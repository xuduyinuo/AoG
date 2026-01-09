import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 在导入其他模块之前设置HF镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

try:
    import dotenv  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError("python-dotenv is required. Please install it with `pip install python-dotenv`." ) from exc

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        "sentence-transformers is required. Install it with `pip install sentence-transformers`."
    ) from exc

from tqdm import tqdm

from agent_pipeline import AoGAgentPipeline
from utils import prepare_dataset

dotenv.load_dotenv()

def repeat_unanswer(dataset, datas, question_string, model_name):
    answered_set = set()
    new_data = []

    file_path = 'AoG_'+dataset+'_'+model_name+'.jsonl'
    
    # 检查文件是否存在
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():  # 确保行不为空
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # 跳过无效的JSON行
                    # Support both legacy schema (dataset-specific key) and new schema ('question')
                    q_val = data.get(question_string) or data.get("question")
                    if isinstance(q_val, str) and q_val:
                        answered_set.add(q_val)
        print(f"Found {len(answered_set)} previously answered questions")
    else:
        print(f"Result file {file_path} not found, processing all questions")

    for x in datas:
        if x[question_string] not in answered_set:
            new_data.append(x)
    print(f"Total questions to process: {len(new_data)}")

    return new_data

def get_one_data(datas, question_string, question):
    for data in datas:
        if data[question_string] == question:
            return [data]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=4096, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.3, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0.3, help="the temperature in reasoning stage.")
    parser.add_argument("--depth", type=int,
                        default=4, help="choose the search depth of AoG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    parser.add_argument("--debug", action="store_true", help="Print ReAct internal prompts, actions and observations.")
    # Optional switches for confidence and KG completion features
    parser.add_argument(
        "--enable-confidence-gate",
        action="store_true",
        default=True,
        help="Enable confidence-based stop criterion (default on).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Confidence threshold in [0,1] to accept an answer when confidence gate is enabled (default: 0.6).",
    )
    parser.add_argument(
        "--enable-kg-completion",
        action="store_true",
        default=True,
        help="Enable KG completion fallback when expansion/pruning yields no candidates (default on).",
    )
    parser.add_argument(
        "--kg-completion-budget",
        type=int,
        default=2,
        help="Max times to use KG completion per question (default: 2).",
    )
    parser.add_argument(
        "--disable-best-guess",
        action="store_true",
        help="Disable automatic best-guess answering when graph evidence is insufficient (default enabled).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of concurrent worker threads to process questions in parallel. Recommended 2-8 depending on your API/IO limits.",
    )

    args = parser.parse_args()
    # Bridge CLI flags to attributes used inside the pipeline via getattr
    # Keep names consistent with agent pipeline expectations
    setattr(args, "enable_confidence_gate", getattr(args, "enable_confidence_gate", True))
    setattr(args, "confidence_threshold", getattr(args, "confidence_threshold", 0.6))
    setattr(args, "enable_kg_completion", getattr(args, "enable_kg_completion", True))
    setattr(args, "kg_completion_budget", getattr(args, "kg_completion_budget", 2))
    setattr(args, "auto_best_guess_on_null", not getattr(args, "disable_best_guess", False))
    datas, question_string = prepare_dataset(args.dataset)
    datas = repeat_unanswer(args.dataset, datas, question_string, args.LLM_type)

    model = SentenceTransformer('msmarco-distilbert-base-tas-b')
    pipeline = AoGAgentPipeline(args, model)

    part_q = False
    if part_q:
        q_set = []
        with open('../eval/analysis_question', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                q_set.append(line.strip())

    print("Start Running AoG agents on %s dataset." % args.dataset)
    # If num_workers == 1, keep sequential behavior for easier debugging
    if args.num_workers <= 1:
        for data in tqdm(datas):
            question = data[question_string]
            if part_q and question not in q_set:
                continue
            print('New question start:', question)
            try:
                pipeline.process_question(data, question_string)
            except Exception as exc:
                print(f"Failed to process question '{question}': {exc}")
    else:
        # Threaded execution: process per-question in a thread pool
        # Note: Most steps are I/O bound (LLM calls, SPARQL), so threads help. Ensure your API rate limits are respected.
        # Filter by part_q if enabled
        work_items = [d for d in datas if (not part_q or d[question_string] in q_set)]
        total = len(work_items)
        print(f"Running with {args.num_workers} worker threads on {total} questions…")

        # Lock for tqdm updates
        pbar_lock = threading.Lock()
        with tqdm(total=total) as pbar:
            def _task(d):
                q = d[question_string]
                print('New question start:', q)
                try:
                    pipeline.process_question(d, question_string)
                except Exception as exc:
                    print(f"Failed to process question '{q}': {exc}")
                finally:
                    with pbar_lock:
                        pbar.update(1)

            with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
                futures = [executor.submit(_task, d) for d in work_items]
                # Optionally, wait and surface first exceptions early
                for f in as_completed(futures):
                    _ = f.result()  # re-raise exceptions if any already printed
