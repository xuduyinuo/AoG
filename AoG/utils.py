from prompt_list import *
import json
import time
from openai import OpenAI, APIError, APITimeoutError, RateLimitError, APIConnectionError, OpenAIError
import re
import requests
import random
from prompt_list import *
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import os
import dotenv
dotenv.load_dotenv()

color_yellow = "\033[93m"
color_green = "\033[92m"
color_red= "\033[91m"
color_end = "\033[0m"

def retrieve_top_docs(query, docs, model, width=3):
    query_emb = model.encode(query)
    doc_emb = model.encode(docs)
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)
    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]
    return top_docs, top_scores

def run_llm(
        prompt,
        temperature,
        max_tokens,
        opeani_api_keys,
        engine="qwen-plus",
        print_in=True,
        print_out=True,
        max_retries=4,
        retry_interval=3.0,
        backoff_factor=2.0,
    ):
    if print_in:
        print(color_green+prompt+color_end)

    # 使用阿里云百炼平台的调用方式
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),  # 使用百炼API Key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    
    last_error = None
    attempt_delay = retry_interval

    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0,
                # Enable model-side web search (DashScope-compatible non-standard arg)
                extra_body={"enable_search": True},
            )

            result = completion.choices[0].message.content
            token_num = {
                "total": completion.usage.total_tokens,
                "input": completion.usage.prompt_tokens,
                "output": completion.usage.completion_tokens,
            }

            if print_out:
                print(color_yellow + result + color_end)
            return result, token_num

        except (APITimeoutError, APIConnectionError, RateLimitError, APIError, OpenAIError) as exc:  # pylint: disable=broad-except
            last_error = exc
            status_code = getattr(exc, "status_code", None)
            error_message = str(exc)

            should_retry = False
            if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
                should_retry = True
            elif isinstance(exc, APIError) and status_code in {429, 500, 502, 503, 504}:
                should_retry = True
            elif any(keyword in error_message for keyword in ["Too many requests", "throttled", "capacity", "ServiceUnavailable", "503"]):
                should_retry = True

            if not should_retry or attempt >= max_retries:
                break

            print(color_red + f"LLM call failed (attempt {attempt}/{max_retries}) with retryable error: {error_message}" + color_end)
            time.sleep(attempt_delay)
            attempt_delay *= backoff_factor
        except Exception as exc:  # pylint: disable=broad-except
            # Non-OpenAI errors are re-raised immediately
            raise

    error_text = str(last_error) if last_error else "Unknown error"
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {error_text}")


def convert_dict_name(ent_rel_ent_dict, entid_name):
    name_dict = {}
    for topic_e, h_t_dict in ent_rel_ent_dict.items():
        if entid_name[topic_e] not in name_dict.keys():
            name_dict[entid_name[topic_e]] = {}

        for h_t, r_e_dict in h_t_dict.items():
            if h_t not in name_dict[entid_name[topic_e]].keys():
                name_dict[entid_name[topic_e]][h_t] = {}
            
            for rela, e_list in r_e_dict.items():
                if rela not in name_dict[entid_name[topic_e]][h_t].keys():
                    name_dict[entid_name[topic_e]][h_t][rela] = []
                for ent in e_list:
                    if entid_name[ent] not in name_dict[entid_name[topic_e]][h_t][rela]:
                        name_dict[entid_name[topic_e]][h_t][rela].append(entid_name[ent])
    return name_dict

    

def _stringify_ground_truth_value(val):
    """Convert various answer value shapes into a user-friendly string."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        # Normalize floats like 2014.0 -> 2014
        if isinstance(val, float) and val.is_integer():
            return str(int(val))
        return str(val)
    if isinstance(val, str):
        return val
    if isinstance(val, (list, tuple, set)):
        parts = [p for p in ([_stringify_ground_truth_value(x) for x in val]) if p]
        # Deduplicate while preserving order
        seen = set()
        ordered = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        return ", ".join(ordered)
    if isinstance(val, dict):
        # Prefer common keys
        for key in ("EntityName", "AnswerArgument", "answer", "text", "string", "value"):
            if key in val and val[key] not in (None, ""):
                return _stringify_ground_truth_value(val[key])
        # Fallback to any string-like value
        values = [v for v in val.values() if isinstance(v, (str, int, float))]
        if values:
            return _stringify_ground_truth_value(values[0])
    return str(val)


def extract_ground_truth_from_data(data: dict | None) -> str | None:
    """Best-effort extraction of ground-truth answer text from a dataset sample.

    Supports WebQSP-style (Parses -> Answers with EntityName/AnswerArgument),
    CWQ/GrailQA variants with top-level answers/answer fields, and generic shapes.
    Returns a single string (comma-separated if multiple); None if not found.
    """
    if not data or not isinstance(data, dict):
        return None

    # 1) Direct top-level keys that commonly hold answers
    for key in ("ground_truth", "gold_answer", "final_answer", "answer"):
        if key in data and data[key] not in (None, ""):
            return _stringify_ground_truth_value(data[key])

    # 2) Common list-based fields
    for key in ("answers", "Answers"):
        if key in data and data[key]:
            ans_list = data[key]
            if isinstance(ans_list, list):
                parts = []
                for item in ans_list:
                    if isinstance(item, dict):
                        # Prefer human-readable name
                        if "EntityName" in item and item["EntityName"]:
                            parts.append(_stringify_ground_truth_value(item["EntityName"]))
                        elif "AnswerArgument" in item and item["AnswerArgument"]:
                            parts.append(_stringify_ground_truth_value(item["AnswerArgument"]))
                        else:
                            # Any other common textual field
                            for k in ("answer", "text", "string", "value"):
                                if k in item and item[k]:
                                    parts.append(_stringify_ground_truth_value(item[k]))
                                    break
                    else:
                        parts.append(_stringify_ground_truth_value(item))
                parts = [p for p in parts if p]
                return _stringify_ground_truth_value(parts)

    # 3) WebQSP-style nested parses
    parses = data.get("Parses") or data.get("parses")
    if isinstance(parses, list) and parses:
        # Choose the first parse that has answers (or just the first)
        target = None
        for p in parses:
            if isinstance(p, dict) and p.get("Answers"):
                target = p
                break
        if target is None:
            target = parses[0] if isinstance(parses[0], dict) else None
        if isinstance(target, dict):
            answers = target.get("Answers")
            if isinstance(answers, list) and answers:
                parts = []
                for item in answers:
                    if isinstance(item, dict):
                        if "EntityName" in item and item["EntityName"]:
                            parts.append(_stringify_ground_truth_value(item["EntityName"]))
                        elif "AnswerArgument" in item and item["AnswerArgument"]:
                            parts.append(_stringify_ground_truth_value(item["AnswerArgument"]))
                        else:
                            for k in ("answer", "text", "string", "value"):
                                if k in item and item[k]:
                                    parts.append(_stringify_ground_truth_value(item[k]))
                                    break
                    else:
                        parts.append(_stringify_ground_truth_value(item))
                parts = [p for p in parts if p]
                return _stringify_ground_truth_value(parts)

    return None


def save_2_jsonl(question, question_string, results, cluster_chain_of_entities, call_num, all_t, start_time, file_name, data=None):
    """Save a single QA result in the unified flattened schema.

    New schema fields:
      - question: str
      - ground_truth: str
      - Sufficient: str
      - Answer: str
      - R: str
      - reasoning_chains, call_num, token stats, time

    Backward compatibility: `data` is optional; when not provided, ground_truth may be empty.
    """
    tt = time.time() - start_time

    # Parse reasoning result to extract A.Sufficient, A.Answer, and R
    sufficient_val = "Unknown"
    answer_val: str | None = None
    reason_val = None

    # Try JSON parse first
    try:
        if isinstance(results, str):
            first = results.find('{')
            last = results.rfind('}')
            candidate = results[first:last+1] if (first != -1 and last != -1 and last > first) else results
            parsed = json.loads(candidate)
        elif isinstance(results, dict):
            parsed = results
        else:
            parsed = None
        if isinstance(parsed, dict):
            A = parsed.get("A") if isinstance(parsed.get("A"), dict) else None
            if A:
                sufficient_val = str(A.get("Sufficient", sufficient_val))
                av = A.get("Answer")
                if isinstance(av, list):
                    answer_val = ", ".join([_stringify_ground_truth_value(x) or "" for x in av if x]) or None
                elif av is not None:
                    answer_val = _stringify_ground_truth_value(av)
            if "R" in parsed:
                reason_val = _stringify_ground_truth_value(parsed.get("R"))
    except Exception:
        parsed = None

    # Fallback: regex-based extractor
    if answer_val is None or reason_val is None:
        try:
            ans, rsn, suf = extract_reason_and_anwer(results if isinstance(results, str) else json.dumps(results, ensure_ascii=False))
            sufficient_val = suf or sufficient_val
            # ans might be a JSON-like list string; normalize
            ans_str = ans.strip() if isinstance(ans, str) else _stringify_ground_truth_value(ans)
            if ans_str and ans_str.startswith('[') and ans_str.endswith(']'):
                try:
                    ans_list = json.loads(ans_str)
                    answer_val = ", ".join([_stringify_ground_truth_value(x) or "" for x in ans_list if x]) or None
                except Exception:
                    answer_val = ans_str
            else:
                answer_val = ans_str
            reason_val = rsn if rsn else reason_val
        except Exception:
            # Last resort: store raw text into R, keep Answer Null
            if reason_val is None:
                reason_val = results if isinstance(results, str) else json.dumps(results, ensure_ascii=False)
            if answer_val is None:
                answer_val = "Null"

    # Compute ground truth
    ground_truth = extract_ground_truth_from_data(data) if data else None
    if ground_truth is None:
        ground_truth = ""

    out = {
        "question": question,
        "ground_truth": ground_truth,
        "Sufficient": str(sufficient_val),
        "Answer": _stringify_ground_truth_value(answer_val) if answer_val is not None else "Null",
        "R": reason_val if isinstance(reason_val, str) else _stringify_ground_truth_value(reason_val) or "",
        "reasoning_chains": cluster_chain_of_entities,
        "call_num": call_num,
        "total_token": all_t.get('total', 0),
        "input_token": all_t.get('input', 0),
        "output_token": all_t.get('output', 0),
        "time": tt,
    }

    with open("PoG_{}.jsonl".format(file_name), "a", encoding="utf-8") as outfile:
        json_str = json.dumps(out, ensure_ascii=False)
        outfile.write(json_str + "\n")

def extract_add_ent(string):
    first_brace_p = string.find('[')
    last_brace_p = string.rfind(']')
    string = string[first_brace_p:last_brace_p+1]
    try:
        new_string = eval(string)
    except:
        s_list = string.split('\', \'')
        if len(s_list) == 1:
            new_string = [s_list[0].strip('[\'').strip('\']')]
        else:
            new_string = [s.strip('[\'').strip('\']') for s in s_list]
    return new_string

def extract_memory(string):
    first_brace_p = string.find('{')
    last_brace_p = string.rfind('}')
    string = string[first_brace_p:last_brace_p+1]
    return string

def extract_reason_and_anwer(string):
    first_brace_p = string.find('{')
    last_brace_p = string.rfind('}')
    string = string[first_brace_p:last_brace_p+1]
    answer = re.search(r'"Answer":\s*"(.*?)"', string)
    if answer:
        answer = answer.group(1)
    else:
        answer = re.search(r'"Answer":\s*(\[[^\]]+\])', string).group(1)

    reason = re.search(r'"R":\s*"(.*?)"', string).group(1)
    sufficient = re.search(r'"Sufficient":\s*"(.*?)"', string).group(1)
    print("Answer:", answer)
    print("Reason:", reason)
    print("Sufficient:", sufficient)
    return answer, reason, sufficient

def extract_add_and_reason(string):
    first_brace_p = string.find('{')
    last_brace_p = string.rfind('}')
    string = string[first_brace_p:last_brace_p+1]
    flag = re.search(r'"Add":\s*"(.*?)"', string).group(1)
    reason = re.search(r'"Reason":\s*"(.*?)"', string).group(1)

    print("Add:", flag)
    print("Reason:", reason)
    if 'yes' in flag.lower():
        return True, reason
    else:
        return False, reason
    


def generate_without_explored_paths(question, subquestions, args):
    prompt = cot_prompt + question 

    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False)
    return response, token_num

def break_question(question, args): 
    prompt = subobjective_prompt + question
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False, False)
    first_brace_p = response.find('[')
    last_brace_p = response.rfind(']')
    response = response[first_brace_p:last_brace_p+1]
    return response, token_num

def get_subquestions(q_mem_f_path, question, args):
    sub_questions, token_num = break_question(question, args)
    with open(q_mem_f_path+'/'+'subq', 'w', encoding='utf-8') as f:
        f.write(str(sub_questions))

    return sub_questions, token_num

def if_finish_list(question, lst, depth_ent_rel_ent_dict, entid_name, name_entid, q_mem_f_path, results, cluster_chain_of_entities, args, model):
    cur_call_time = 0
    cur_token = {'total': 0, 'input': 0, 'output': 0}

    with open(q_mem_f_path+'/mem', 'r', encoding='utf-8') as f:
        his_mem = f.read()

    if all(elem == "[FINISH_ID]" for elem in lst):
        new_lst = []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
    
    all_ent_set = set()
    for dep, ent_rel_ent_dict in depth_ent_rel_ent_dict.items():
        for topic_e, h_t_dict in ent_rel_ent_dict.items():
            all_ent_set.add(topic_e)
            for h_t, r_e_dict in h_t_dict.items():
                for rela, e_list in r_e_dict.items():
                    if all(entid_name[item].startswith('m.') for item in e_list) and len(e_list)>10:
                        e_list = random.sample(e_list, 10)
                        
                    if len(e_list) > 70:
                        print('··········exceed 70 entities··········')
                        sorted_e_list = [entid_name[e_id] for e_id in e_list]
                        topn_entities, topn_scores = retrieve_top_docs(question, sorted_e_list, model, 70)
                        print('sentence:', topn_entities)
                        e_list = [name_entid[e_n] for e_n in topn_entities]
                        all_ent_set |= (set(e_list))

    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    
    prompt = judge_reverse+question+'\nEntities set to be retrieved: ' + str(list(set(sorted([entid_name[ent_i] for ent_i in new_lst])))) +'\nMemory: '+his_mem+'\nKnowledge Triplets:'+chain_prompt

    cur_call_time += 1
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    for kk in token_num.keys():
        cur_token[kk] += token_num[kk]

    flag, reason = extract_add_and_reason(response)

    if flag:
        other_entities = sorted(list(all_ent_set - set(new_lst)))
        other_entities_name = [entid_name[ent_i] for ent_i in other_entities]
        
        print('filter already', [entid_name[ent_i] for ent_i in new_lst], [entid_name[ent_i] for ent_i in all_ent_set], other_entities_name)

        prompt = add_ent_prompt+question+'\nReason: '+reason+'\nCandidate Entities: ' + str(sorted(other_entities_name))+'\nMemory: '+his_mem

        cur_call_time += 1
        response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)

        for kk in token_num.keys():
            cur_token[kk] += token_num[kk]

        add_ent_list = extract_add_ent(response)
        add_ent_list = [name_entid[ent_i] for ent_i in add_ent_list if ent_i in other_entities_name]
        add_ent_list = sorted(add_ent_list)
        if add_ent_list:
            print('add reverse ent:', len(add_ent_list), [entid_name[ent_i] for ent_i in add_ent_list])
            return new_lst, add_ent_list, cur_call_time, cur_token
    return new_lst, [], cur_call_time, cur_token

    
def prepare_dataset(dataset_name):
    if dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    else:
        print("dataset not found, you should pick from {simpleqa, webqsp, grailqa}.")
        exit(-1)
    return datas, question_string

