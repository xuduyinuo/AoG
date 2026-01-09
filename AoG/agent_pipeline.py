import json
import os
import pprint
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple
import threading

from sentence_transformers import SentenceTransformer

import freebase_func as fb
from prompt_list import react_agent_prompt
from utils import (
    convert_dict_name,
    generate_without_explored_paths,
    get_subquestions,
    if_finish_list,
    run_llm,
    save_2_jsonl,
)

# 全局锁，用于在多线程模式下确保向JSONL结果文件追加内容时的线程安全
_SAVE_LOCK = threading.Lock()


@dataclass
class AgentLogger:
    """跟踪agent的LLM调用统计信息。"""

    call_num: int = 0
    tokens: Dict[str, int] = field(default_factory=lambda: {"total": 0, "input": 0, "output": 0})

    def record_call(self, token_num: Dict[str, int] | None) -> None:
        self.call_num += 1
        if not token_num:
            return
        for key in ("total", "input", "output"):
            self.tokens[key] += token_num.get(key, 0)

    def record_batch(self, call_count: int, token_num: Dict[str, int]) -> None:
        self.call_num += call_count
        for key in ("total", "input", "output"):
            self.tokens[key] += token_num.get(key, 0)


@dataclass
class AgentContext:
    """保存agent中的共享状态。"""

    args: Any
    question: str
    question_string: str
    topic_entity: Dict[str, str]
    entid_name: Dict[str, str]
    name_entid: Dict[str, str]
    q_mem_f_path: str
    max_depth: int = 0
    depth: int = 0
    subquestions: str | None = None
    cluster_chain_of_entities: List[List[List[Tuple[str, str, str]]]] = field(default_factory=list)
    depth_ent_rel_ent_dict: Dict[int, Dict] = field(default_factory=dict)
    reverse_rec: Dict[str, Any] = field(default_factory=lambda: {"time": 0, "ent": []})
    logger: AgentLogger = field(default_factory=AgentLogger)
    start_time: float = field(default_factory=time.time)
    pre_relations: List[str] = field(default_factory=list)
    pre_heads: List[int] = field(default_factory=list)
    pending_entities_id: List[str] = field(default_factory=list)
    latest_results: str | None = None
    latest_answer: str | None = None
    latest_sufficient: str | None = None
    latest_status: str | None = None
    ready_to_finish: bool = False
    termination_kind: str | None = None
    final_answer: str | None = None
    final_saved: bool = False
    # 新增：置信度评分和知识图谱补全记录
    latest_confidence: float | None = None
    kg_completion_rec: Dict[str, Any] = field(default_factory=lambda: {"time": 0, "added": 0})
    # 显式的完成标志，避免动态创建属性
    finished: bool = False

    def ensure_memory_store(self) -> None:
        os.makedirs(self.q_mem_f_path, exist_ok=True)
        open(os.path.join(self.q_mem_f_path, "mem"), "w", encoding="utf-8").close()

    def finalize_reverse_entities(self) -> None:
        if not self.reverse_rec["ent"]:
            return
        converted = []
        for ent_id in self.reverse_rec["ent"]:
            converted.append(self.entid_name.get(ent_id, ent_id))
        self.reverse_rec["ent"] = converted

    def __post_init__(self) -> None:
        self.max_depth = getattr(self.args, "depth", 4)


@dataclass
class RelationExpansionResult:
    ent_rel_ent_dict: Dict[str, Dict]
    total_candidates: List[str]
    total_relations: List[str]
    total_entities_id: List[str]
    total_topic_entities: List[str]
    total_head: List[bool]


@dataclass
class EntityPruningResult:
    flag: bool
    chain_of_entities: List[List[Tuple[str, str, str]]]
    entities_id: List[str]
    pre_relations: List[str]
    pre_heads: List[bool]
    ent_rel_ent_dict: Dict[str, Dict]
    cur_call_time: int
    cur_token: Dict[str, int]


@dataclass
class ReActTool:
    name: str
    description: str
    func: Callable[["PoGAgentPipeline", AgentContext, Dict[str, Any] | None], str]


class PlanningAgent:
    def __init__(self, args: Any):
        self.args = args

    def act(self, ctx: AgentContext) -> None:
        sub_questions, token_num = get_subquestions(ctx.q_mem_f_path, ctx.question, self.args)
        ctx.logger.record_call(token_num)
        ctx.subquestions = sub_questions


class RelationExpansionAgent:
    def __init__(self, args: Any):
        self.args = args

    def act(
        self,
        ctx: AgentContext,
        topic_entity: Dict[str, str],
        pre_relations: List[str],
        pre_heads: List[int | bool],
    ) -> RelationExpansionResult:
        current_entity_relations_list: List[Dict[str, Any]] = []
        for idx, (entity_id, entity_name) in enumerate(topic_entity.items()):
            if entity_id == "[FINISH_ID]":
                continue
            token_num = None
            relations: List[Dict[str, Any]] = []
            retrieve_relations, token_num = fb.relation_search_prune(
                entity_id,
                ctx.subquestions,
                entity_name,
                pre_relations,
                pre_heads[idx] if idx < len(pre_heads) else -1,
                ctx.question,
                self.args,
            )
            ctx.logger.record_call(token_num)
            relations.extend(retrieve_relations)
            current_entity_relations_list.extend(relations)

        total_candidates: List[str] = []
        total_relations: List[str] = []
        total_entities_id: List[str] = []
        total_topic_entities: List[str] = []
        total_head: List[bool] = []
        ent_rel_ent_dict: Dict[str, Dict] = {}

        for ent_rel in current_entity_relations_list:
            if ent_rel["entity"] not in ent_rel_ent_dict:
                ent_rel_ent_dict[ent_rel["entity"]] = {}

            if ent_rel["head"]:
                head_or_tail = "head"
                entity_candidates_id = fb.entity_search(ent_rel["entity"], ent_rel["relation"], True)
            else:
                head_or_tail = "tail"
                entity_candidates_id = fb.entity_search(ent_rel["entity"], ent_rel["relation"], False)

            if not entity_candidates_id:
                print("the relations without tail entity:", ent_rel)
                continue

            entity_candidates, entity_candidates_id = fb.provide_triple(entity_candidates_id, ent_rel["relation"])

            ctx.name_entid.update(dict(zip(entity_candidates, entity_candidates_id)))
            ctx.entid_name.update(dict(zip(entity_candidates_id, entity_candidates)))

            if head_or_tail not in ent_rel_ent_dict[ent_rel["entity"]]:
                ent_rel_ent_dict[ent_rel["entity"]][head_or_tail] = {}

            if ent_rel["relation"] not in ent_rel_ent_dict[ent_rel["entity"]][head_or_tail]:
                ent_rel_ent_dict[ent_rel["entity"]][head_or_tail][ent_rel["relation"]] = []

            for retrive_ent in entity_candidates_id:
                if (
                    retrive_ent
                    not in ent_rel_ent_dict[ent_rel["entity"]][head_or_tail][ent_rel["relation"]]
                ):
                    ent_rel_ent_dict[ent_rel["entity"]][head_or_tail][ent_rel["relation"]].append(retrive_ent)

            total_candidates, total_relations, total_entities_id, total_topic_entities, total_head = fb.update_history(
                entity_candidates,
                ent_rel,
                entity_candidates_id,
                total_candidates,
                total_relations,
                total_entities_id,
                total_topic_entities,
                total_head,
            )

        return RelationExpansionResult(
            ent_rel_ent_dict=ent_rel_ent_dict,
            total_candidates=total_candidates,
            total_relations=total_relations,
            total_entities_id=total_entities_id,
            total_topic_entities=total_topic_entities,
            total_head=total_head,
        )


class EntityPruningAgent:
    def __init__(self, args: Any, model: SentenceTransformer):
        self.args = args
        self.model = model

    def act(
        self,
        ctx: AgentContext,
        expansion_result: RelationExpansionResult,
    ) -> EntityPruningResult:
        (
            flag,
            chain_of_entities,
            entities_id,
            pre_relations,
            pre_heads,
            new_ent_rel_ent_dict,
            cur_call_time,
            cur_token,
        ) = fb.entity_condition_prune(
            ctx.question,
            expansion_result.total_entities_id,
            expansion_result.total_relations,
            expansion_result.total_candidates,
            expansion_result.total_topic_entities,
            expansion_result.total_head,
            expansion_result.ent_rel_ent_dict,
            ctx.entid_name,
            ctx.name_entid,
            self.args,
            self.model,
        )
        return EntityPruningResult(
            flag=flag,
            chain_of_entities=chain_of_entities,
            entities_id=entities_id,
            pre_relations=pre_relations,
            pre_heads=pre_heads,
            ent_rel_ent_dict=new_ent_rel_ent_dict,
            cur_call_time=cur_call_time,
            cur_token=cur_token,
        )


class MemoryAgent:
    def __init__(self, args: Any):
        self.args = args

    def act(self, ctx: AgentContext, ent_rel_ent_dict: Dict[str, Dict]) -> Dict[str, int]:
        token_num = fb.update_memory(
            ctx.question,
            ctx.subquestions,
            ent_rel_ent_dict,
            ctx.entid_name,
            ctx.cluster_chain_of_entities,
            ctx.q_mem_f_path,
            self.args,
        )
        ctx.logger.record_call(token_num)
        return token_num


class ReasoningAgent:
    def __init__(self, args: Any):
        self.args = args

    def act(
        self,
        ctx: AgentContext,
        ent_rel_ent_dict: Dict[str, Dict],
    ) -> Tuple[str, str, str, Dict[str, int]]:
        response, answer, sufficient, token_num = fb.reasoning(
            ctx.question,
            ctx.subquestions,
            ent_rel_ent_dict,
            ctx.entid_name,
            ctx.cluster_chain_of_entities,
            ctx.q_mem_f_path,
            self.args,
        )
        ctx.logger.record_call(token_num)
        return response, answer, sufficient, token_num


class ConfidenceAgent:
    """
    置信度评分

    基于问题、推理和答案，生成一个[0,1]范围内的置信度。
    """

    def __init__(self, args: Any, model: SentenceTransformer):
        self.args = args
        self.model = model

    def act(self, ctx: AgentContext, results: str, answer: str, sufficient: str) -> Tuple[float, Dict[str, int]]:
        scoring_prompt = (
            "You are a strict judge. Given the question, a proposed answer, and reasoning, "
            "output a single number in [0,1] representing confidence that the answer is correct.\n"
            f"Question: {ctx.question}\n"
            f"Answer: {answer}\n"
            f"Sufficient: {sufficient}\n"
            f"Reasoning: {results}\n"
            "Confidence (only a number, no text):"
        )
        try:
            response, token_num = run_llm(
                scoring_prompt,
                getattr(self.args, "temperature_reasoning", 0.2),
                64,
                getattr(self.args, "opeani_api_keys", None),
                getattr(self.args, "LLM_type", "qwen-plus"),
                print_in=False,
                print_out=False,
            )
            ctx.logger.record_call(token_num)
            line = (response or "").strip().splitlines()[-1].strip()
            conf = float(line)
        except Exception:
            # 解析失败时的后备启发式方法
            conf = 0.75 if "yes" in str(sufficient).lower() else 0.35
            token_num = {"total": 0, "input": 0, "output": 0}
        conf = max(0.0, min(1.0, conf))
        return conf, token_num


class BestGuessAgent:
    """
    当图证据不足时，使用内部知识生成答案
    """

    def __init__(self, args: Any):
        self.args = args

    def act(self, ctx: AgentContext) -> Dict[str, Any] | None:
        knowledge_text = self._format_knowledge(ctx)
        prior_reasoning = ctx.latest_results or "None"
        prompt = (
            "You are a knowledgeable assistant. The knowledge graph retrieval may be incomplete or contain IDs. "
            "Use both the provided facts and your own world knowledge to answer the question as accurately as possible. "
            "If unsure, give the most plausible answer and state your confidence level (High/Medium/Low).\n"
            f"Question: {ctx.question}\n"
            "Knowledge (may include identifiers):\n"
            f"{knowledge_text}\n"
            "Previous reasoning attempt (may be empty):\n"
            f"{prior_reasoning}\n\n"
            "Respond in strict JSON format:\n"
            "{\n"
            "  \"A\": {\n"
            "    \"Answer\": \"<concise answer>\",\n"
            "    \"Confidence\": \"High|Medium|Low\"\n"
            "  },\n"
            "  \"R\": \"<brief reasoning>\"\n"
            "}"
        )
        try:
            response, token_num = run_llm(
                prompt,
                getattr(self.args, "temperature_reasoning", 0.3),
                getattr(self.args, "max_length", 512),
                getattr(self.args, "opeani_api_keys", None),
                getattr(self.args, "LLM_type", "qwen-plus"),
                print_in=False,
                print_out=False,
            )
            ctx.logger.record_call(token_num)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"BestGuessAgent调用LLM失败: {exc}")
            return None

        parsed = self._parse_response(response)
        answer_text = parsed.get("answer")
        confidence = parsed.get("confidence")
        result_json = parsed.get("json")
        if not result_json:
            return None
        return {
            "result_json": result_json,
            "answer": answer_text,
            "confidence": confidence,
            "raw": response,
        }

    def _format_knowledge(self, ctx: AgentContext) -> str:
        triples: List[str] = []
        for chain_group in ctx.cluster_chain_of_entities[-6:]:  # 保持最近的上下文简洁
            for chain in chain_group:
                if len(chain) == 3:
                    subj, rel, obj = chain
                    triples.append(f"{subj} --{rel}--> {obj}")
        if not triples:
            return "(No structured knowledge available)"
        unique_order: List[str] = []
        for item in triples:
            if item not in unique_order:
                unique_order.append(item)
        return "\n".join(unique_order[-30:])

    def _parse_response(self, response: str) -> Dict[str, Any]:
        text = (response or "").strip()
        if not text:
            return {}
        json_segment = self._extract_json_segment(text)
        data: Dict[str, Any] | None = None
        if json_segment:
            try:
                data = json.loads(json_segment)
            except json.JSONDecodeError:
                data = None
        if not data or not isinstance(data, dict):
            # 后备方案：将原始文本包装到模板中
            data = {
                "A": {"Answer": text, "Confidence": "Low"},
                "R": "Provided without JSON structure; treat as low confidence best-guess."
            }
        a_block = data.get("A", {}) if isinstance(data.get("A"), dict) else {}
        answer = a_block.get("Answer")
        conf_str = str(a_block.get("Confidence", "")).strip()
        confidence = self._map_confidence(conf_str)
        result_json = json.dumps(data, ensure_ascii=False)
        return {"answer": answer, "confidence": confidence, "json": result_json}

    def _extract_json_segment(self, text: str) -> str | None:
        first = text.find('{')
        last = text.rfind('}')
        if first == -1 or last == -1 or last <= first:
            return None
        return text[first:last + 1]

    def _map_confidence(self, conf: str) -> float | None:
        if not conf:
            return None
        mapping = {"high": 0.9, "medium": 0.6, "low": 0.3}
        return mapping.get(conf.lower())


class ReverseAgent:
    def __init__(self, args: Any, model: SentenceTransformer):
        self.args = args
        self.model = model

    def act(
        self,
        ctx: AgentContext,
        entities_id: List[str],
        results: str,
    ) -> Tuple[List[str], List[str]]:
        entities_id, add_ent_list, cur_call_time, cur_token = if_finish_list(
            ctx.question,
            entities_id,
            ctx.depth_ent_rel_ent_dict,
            ctx.entid_name,
            ctx.name_entid,
            ctx.q_mem_f_path,
            results,
            ctx.cluster_chain_of_entities,
            self.args,
            self.model,
        )
        ctx.logger.record_batch(cur_call_time, cur_token)
        return entities_id, add_ent_list


class KGCompletionAgent:
    """
    尝试通过知识图谱补全来补全缺失的三元组/实体。
    """

    def __init__(self, args: Any, model: SentenceTransformer):
        self.args = args
        self.model = model

    def act(
        self,
        ctx: AgentContext,
        ent_rel_ent_dict: Dict[str, Dict],
    ) -> Tuple[Dict[str, Dict], List[str], List[str], List[bool], int, Dict[str, int]]:
        if hasattr(fb, "kg_completion"):
            new_dict, add_entities_id, add_pre_relations, add_pre_heads, cur_call_time, cur_token = fb.kg_completion(
                ctx.question,
                ctx.subquestions,
                ent_rel_ent_dict,
                ctx.entid_name,
                ctx.name_entid,
                ctx.q_mem_f_path,
                self.args,
                self.model,
            )
            ctx.logger.record_batch(cur_call_time, cur_token)
            return new_dict, add_entities_id, add_pre_relations, add_pre_heads, cur_call_time, cur_token
        # 安全的后备方案：不进行补全
        return ent_rel_ent_dict, [], [], [], 0, {"total": 0, "input": 0, "output": 0}


class TerminationAgent:
    def __init__(self, args: Any):
        self.args = args

    def partial(self, ctx: AgentContext) -> str:
        print("No new knowledge added during current search depth, stop searching.")
        # 首先尝试基于图的回答
        answer, token_num = fb.generate_answer(
            ctx.question,
            ctx.subquestions,
            ctx.cluster_chain_of_entities,
            self.args,
        )
        ctx.logger.record_call(token_num)

        # 检测空/空答案，并可选地回退到通用的链式思维
        try:
            ans_lower = answer.lower()
        except Exception:
            ans_lower = str(answer)
        is_null = False
        # 对常见模式进行简单检查
        if '"Answer"' in ans_lower:
            # 涵盖 "Answer": "Null" 和 "Answer": null 的情况
            is_null = (
                '"answer"\s*:\s*"null"' in ans_lower
                or '"answer"\s*:\s*null' in ans_lower
                or '"answer"\s*:\s*"\[\]"' in ans_lower
            )

        if is_null and getattr(self.args, "enable_auto_fallback", True):
            # 使用模型自身知识的链式思维后备方案
            response, token_num_fb = generate_without_explored_paths(
                ctx.question,
                ctx.subquestions,
                self.args,
            )
            ctx.logger.record_call(token_num_fb)
            # 重置链以标志非知识图谱答案并保存
            ctx.cluster_chain_of_entities = []
            self._save(ctx, response)
            return response

        # 否则保持基于图的答案
        self._save(ctx, answer)
        return answer

    def fallback(self, ctx: AgentContext) -> str:
        response, token_num = generate_without_explored_paths(
            ctx.question,
            ctx.subquestions,
            self.args,
        )
        ctx.logger.record_call(token_num)
        ctx.cluster_chain_of_entities = []
        self._save(ctx, response)
        return response

    def success(self, ctx: AgentContext, results: str) -> None:
        self._save(ctx, results)

    def _save(self, ctx: AgentContext, results: str) -> None:
        ctx.finalize_reverse_entities()
        # 保护写入，以便多个线程不会交错JSONL行
        with _SAVE_LOCK:
            save_2_jsonl(
                ctx.question,
                ctx.question_string,
                results,
                ctx.cluster_chain_of_entities,
                ctx.logger.call_num,
                ctx.logger.tokens,
                ctx.start_time,
                file_name=f"{ctx.args.dataset}_{ctx.args.LLM_type}",
                data=getattr(ctx, "_raw_data_sample", None),
            )


class AoGAgentPipeline:
    """
    实现ReAct框架的AoG代理流水线。
    """

    def __init__(self, args: Any, model: SentenceTransformer):
        self.args = args
        self.model = model
        self.planner = PlanningAgent(args)
        self.relation_agent = RelationExpansionAgent(args)
        self.entity_agent = EntityPruningAgent(args, model)
        self.memory_agent = MemoryAgent(args)
        self.reasoning_agent = ReasoningAgent(args)
        self.reverse_agent = ReverseAgent(args, model)
        self.terminator = TerminationAgent(args)

        self.confidence_agent = ConfidenceAgent(args, model)
        self.kg_completion_agent = KGCompletionAgent(args, model)
        self.best_guess_agent = BestGuessAgent(args)

        self.tools: List[ReActTool] = [
            ReActTool(
                name="PlanSubquestions",
                description="Break the question into sub-objectives and initialise working memory.",
                func=PoGAgentPipeline._tool_plan_subquestions,
            ),
            ReActTool(
                name="SearchGraphDepth",
                description="Explore one search depth: expand relations, prune entities, update memory, and attempt reasoning.",
                func=PoGAgentPipeline._tool_search_graph_depth,
            ),
            ReActTool(
                name="ReverseSearch",
                description="Add additional entities for consideration using reverse retrieval heuristics when current candidates are insufficient.",
                func=PoGAgentPipeline._tool_reverse_search,
            ),
            ReActTool(
                name="GeneratePartialAnswer",
                description="Provide the best-effort answer given current knowledge when search cannot progress further.",
                func=PoGAgentPipeline._tool_generate_partial_answer,
            ),
            ReActTool(
                name="FallbackCOT",
                description="Skip graph exploration and answer directly via chain-of-thought reasoning.",
                func=PoGAgentPipeline._tool_fallback_answer,
            ),
            # 新工具
            ReActTool(
                name="CompleteKG",
                description="Try to complete missing triples/entities when expansion/pruning yields no candidates.",
                func=PoGAgentPipeline._tool_complete_kg,
            ),
            ReActTool(
                name="AssessConfidence",
                description="Score confidence [0,1] of the latest answer based on reasoning.",
                func=PoGAgentPipeline._tool_assess_confidence,
            ),
        ]
        self.tool_map = {tool.name: tool for tool in self.tools}

    def process_question(self, data: Dict[str, Any], question_string: str) -> None:
        question = data[question_string]
        topic_entity = data.get("topic_entity", {}) or {}
        entid_name = {e_id: e_name for e_id, e_name in topic_entity.items()}
        name_entid = {e_name: e_id for e_id, e_name in topic_entity.items()}

        mem_dir = self._resolve_memory_dir(question)
        ctx = AgentContext(
            args=self.args,
            question=question,
            question_string=question_string,
            topic_entity=dict(topic_entity),
            entid_name=entid_name,
            name_entid=name_entid,
            q_mem_f_path=mem_dir,
        )
        # 真实标签提取
        setattr(ctx, "_raw_data_sample", data)
        ctx.ensure_memory_store()
        ctx.start_time = time.time()
        ctx.pre_heads = [-1] * len(topic_entity)

        if not topic_entity:
            ctx.latest_status = "no_topic_entities"

        history: List[Dict[str, Any]] = []
        max_steps = max(6, ctx.max_depth * 4)

        step = 0
        while step < max_steps:
            prompt = self._build_prompt(ctx, history)
            if getattr(self.args, "debug", False):
                print("\n================ ReAct Step %d Prompt ================" % (step + 1))
                print(prompt)
                print("================ End Prompt ================")
            response, token_num = run_llm(
                prompt,
                self.args.temperature_reasoning,
                self.args.max_length,
                self.args.opeani_api_keys,
                self.args.LLM_type,
                print_in=False,
                print_out=False,
            )
            ctx.logger.record_call(token_num)

            thought, action, action_input, action_input_raw, final_answer = self._parse_react_response(response)
            if getattr(self.args, "debug", False):
                print("\n>>> LLM Raw Response:\n", response)
                print("Parsed Thought:", thought)
                print("Parsed Action:", action)
                print("Parsed Action Input Raw:", action_input_raw)
                print("Parsed Action Input (object):", action_input)
                if final_answer:
                    print("Parsed Final Answer (inline):", final_answer)
            entry: Dict[str, Any] = {
                "thought": thought or "",
                "action": action or "",
                "input_raw": action_input_raw,
                "observation": "",
            }

            if not action:
                entry["observation"] = (
                    "No action detected in response. Valid options: "
                    + ", ".join(self.tool_map.keys())
                    + ", Finish"
                )
                if getattr(self.args, "debug", False):
                    print("[Debug] No valid action parsed. Observation appended.")
                history.append(entry)
                step += 1
                continue

            if action.lower() == "finish":
                answer_candidate: str | None = None
                if isinstance(action_input, dict):
                    answer_candidate = action_input.get("answer")
                elif isinstance(action_input, str) and action_input.lower() not in {"", "null"}:
                    answer_candidate = action_input
                if final_answer and not answer_candidate:
                    answer_candidate = final_answer
                if answer_candidate is not None:
                    ctx.final_answer = self._sanitize_final_answer(ctx, answer_candidate)
                # 当结构化答案为空或明确不充分时自动回退
                try:
                    fa_lower = (ctx.final_answer or "").lower()
                except Exception:
                    fa_lower = str(ctx.final_answer)
                null_like = (
                    '"answer"\s*:\s*"null"' in fa_lower
                    or '"answer"\s*:\s*null' in fa_lower
                    or '"sufficient"\s*:\s*"no"' in fa_lower
                )
                if null_like and getattr(self.args, "enable_auto_fallback", True):
                    # 使用通用链式思维回退方案，从模型自身知识回答
                    ctx.final_answer = response
                    ctx.termination_kind = "fallback"
                    ctx.final_saved = True
                ctx.ready_to_finish = True
                ctx.finished = True
                entry["observation"] = "Finish acknowledged."
                if getattr(self.args, "debug", False):
                    print("[Debug] Finish detected. Candidate answer:", ctx.final_answer)
                history.append(entry)
                break

            observation = self._execute_tool(action, action_input, ctx)
            entry["observation"] = observation
            if getattr(self.args, "debug", False):
                print("[Tool Execution] Action:", action)
                print("[Tool Execution] Observation:\n", observation)
            history.append(entry)

            if ctx.ready_to_finish and ctx.termination_kind in {"pending_success", "partial", "fallback"}:
                # 允许agent在下一步调用Finish
                step += 1
                continue

            step += 1

        if not ctx.finished and not ctx.final_saved:
            if ctx.termination_kind == "pending_success" and ctx.final_answer:
                self.terminator.success(ctx, ctx.final_answer)
                ctx.final_saved = True
                if getattr(self.args, "debug", False):
                    print("[Debug] Saved successful final answer.")
            else:
                result = self.terminator.partial(ctx)
                ctx.final_answer = result
                ctx.termination_kind = "partial"
                ctx.final_saved = True
                ctx.ready_to_finish = True
                ctx.finished = True
                if getattr(self.args, "debug", False):
                    print("[Debug] Performed partial termination, answer saved.")

        if ctx.final_answer and not ctx.final_saved:
            self.terminator.success(ctx, ctx.final_answer)
            ctx.final_saved = True
            if getattr(self.args, "debug", False):
                print("[Debug] Final answer saved at end of loop.")

    def _build_prompt(self, ctx: AgentContext, history: List[Dict[str, Any]]) -> str:
        tool_desc = "\n".join(f"- {tool.name}: {tool.description}" for tool in self.tools)
        topic_names = list(ctx.topic_entity.values())
        topic_text = self._format_entity_names(topic_names)
        state_lines = [
            f"- Depth explored: {ctx.depth}/{ctx.max_depth}",
            f"- Topic entities in focus: {topic_text}",
            f"- Last status: {ctx.latest_status or 'None'}",
            f"- Ready to finish: {'Yes' if ctx.ready_to_finish else 'No'}",
            f"- Reverse searches used: {ctx.reverse_rec['time']} / 5",
            f"- KG completion used: {ctx.kg_completion_rec['time']} / {getattr(self.args, 'kg_completion_budget', 2)}",
        ]
        if ctx.subquestions:
            state_lines.append(f"- Subquestions: {ctx.subquestions}")
        else:
            state_lines.append("- Subquestions: (not planned yet)")
        if ctx.latest_answer:
            state_lines.append(
                f"- Latest answer candidate: {ctx.latest_answer} (sufficient={ctx.latest_sufficient})"
            )
        if ctx.latest_confidence is not None:
            state_lines.append(f"- Latest confidence: {ctx.latest_confidence:.2f}")
        if ctx.latest_results:
            preview = ctx.latest_results.strip()
            if len(preview) > 200:
                preview = preview[:200] + "..."
            state_lines.append(f"- Latest reasoning output: {preview}")

        history_text = self._format_history(history)

        prompt = (
            f"{react_agent_prompt}\n\n"
            f"Available tools:\n{tool_desc}\n\n"
            f"Question: {ctx.question}\n"
            f"Dataset: {self.args.dataset}\n\n"
            f"Current State:\n" + "\n".join(state_lines) + "\n\n"
            f"Interaction History:\n{history_text}\n\n"
            "Respond with the next Thought, Action, and Action Input."
        )
        return prompt

    def _format_entity_names(self, names: List[str], max_items: int = 5) -> str:
        if not names:
            return "None"
        display = names[:max_items]
        if len(names) > max_items:
            display.append(f"... (+{len(names) - max_items} more)")
        return ", ".join(display)

    def _format_history(self, history: List[Dict[str, Any]], max_steps: int = 6) -> str:
        if not history:
            return "No previous actions."
        trimmed = history[-max_steps:]
        lines: List[str] = []
        for idx, step in enumerate(trimmed, start=len(history) - len(trimmed) + 1):
            lines.append(f"Step {idx} Thought: {step['thought']}")
            lines.append(f"Step {idx} Action: {step['action']}")
            lines.append(f"Step {idx} Action Input: {step['input_raw']}")
            lines.append(f"Step {idx} Observation: {step['observation']}")
        return "\n".join(lines)

    def _parse_react_response(
        self, response: str
    ) -> Tuple[str, str, Any, str, str | None]:
        lines = response.strip().splitlines()
        thought_lines: List[str] = []
        action = ""
        action_input_raw = "null"
        final_answer_lines: List[str] = []
        current = None

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Thought:"):
                thought_lines = [stripped[len("Thought:") :].strip()]
                current = "thought"
            elif stripped.startswith("Action:"):
                action = stripped[len("Action:") :].strip()
                current = "action"
            elif stripped.startswith("Action Input:"):
                action_input_raw = stripped[len("Action Input:") :].strip() or "null"
                current = "action_input"
            elif stripped.startswith("Final Answer:"):
                final_answer_lines = [stripped[len("Final Answer:") :].strip()]
                current = "final"
            else:
                if current == "thought":
                    thought_lines.append(stripped)
                elif current == "action_input":
                    action_input_raw += " " + stripped
                elif current == "final":
                    final_answer_lines.append(stripped)

        thought = " ".join(filter(None, thought_lines)).strip()
        final_answer = " ".join(filter(None, final_answer_lines)).strip() or None

        action_input_raw_clean = action_input_raw.strip()
        if action_input_raw_clean.lower() == "null" or not action_input_raw_clean:
            parsed_input: Any = None
        else:
            try:
                parsed_input = json.loads(action_input_raw_clean)
            except json.JSONDecodeError:
                parsed_input = action_input_raw_clean

        return thought, action, parsed_input, action_input_raw_clean, final_answer

    def _execute_tool(self, action: str, action_input: Any, ctx: AgentContext) -> str:
        tool = self.tool_map.get(action)
        if not tool:
            ctx.latest_status = f"unknown_tool:{action}"
            return f"Unknown tool '{action}'. Available tools: {', '.join(self.tool_map.keys())}"

        params: Dict[str, Any]
        if isinstance(action_input, dict):
            params = action_input
        elif action_input is None:
            params = {}
        else:
            params = {"input": action_input}

        try:
            observation = tool.func(self, ctx, params)
        except Exception as exc:  
            ctx.latest_status = f"tool_error:{action}"
            print(f"Tool {action} raised an exception: {exc}")
            observation = f"Error executing {action}: {exc}"
        return observation

    def _sanitize_final_answer(self, ctx: AgentContext, answer_candidate: Any) -> str:
        if isinstance(answer_candidate, str):
            cleaned = answer_candidate.strip()
            if cleaned.startswith("{") and '"A"' in cleaned and '"R"' in cleaned:
                return cleaned
            if ctx.latest_results and '"A"' in ctx.latest_results and '"R"' in ctx.latest_results:
                return ctx.latest_results
            answer_text = cleaned or (ctx.latest_answer or "Null")
        elif isinstance(answer_candidate, (list, tuple)):
            answer_text = ", ".join(str(item) for item in answer_candidate if item)
        elif isinstance(answer_candidate, dict):
            if 'A' in answer_candidate and 'R' in answer_candidate:
                return json.dumps(answer_candidate, ensure_ascii=False)
            answer_text = answer_candidate.get("answer") or ctx.latest_answer or "Null"
        else:
            answer_text = str(answer_candidate)

        reason_text = ctx.latest_results
        if reason_text and '"A"' in reason_text and '"R"' in reason_text:
            return reason_text

        reasoning = ctx.latest_results or ctx.latest_answer or "No detailed reasoning provided."
        a_block = {
            "Sufficient": "Unknown",
            "Answer": answer_text,
        }
        if ctx.latest_confidence is not None:
            try:
                a_block["Confidence"] = round(float(ctx.latest_confidence), 3)
            except Exception:
                a_block["Confidence"] = ctx.latest_confidence
        structured = {"A": a_block, "R": reasoning}
        return json.dumps(structured, ensure_ascii=False)

    def _tool_plan_subquestions(self, ctx: AgentContext, params: Dict[str, Any]) -> str:
        if ctx.subquestions:
            return f"Subquestions already available: {ctx.subquestions}"
        self.planner.act(ctx)
        ctx.latest_status = "planned_subquestions"
        return f"Identified subquestions: {ctx.subquestions}"

    def _tool_search_graph_depth(self, ctx: AgentContext, params: Dict[str, Any]) -> str:
        if ctx.depth >= ctx.max_depth:
            ctx.latest_status = "max_depth_reached"
            return (
                f"Maximum depth {ctx.max_depth} reached. Consider ReverseSearch, GeneratePartialAnswer, or FallbackCOT."
            )
        if not ctx.topic_entity:
            ctx.latest_status = "no_topic_entities"
            return "No topic entities available. Consider ReverseSearch or FallbackCOT."

        ctx.depth += 1
        print(f"===== ReAct Depth {ctx.depth} Exploration =====")
        expansion_result = self.relation_agent.act(ctx, ctx.topic_entity, ctx.pre_relations, ctx.pre_heads or [-1])
        ctx.depth_ent_rel_ent_dict[ctx.depth] = expansion_result.ent_rel_ent_dict
        pprint.pprint(convert_dict_name(expansion_result.ent_rel_ent_dict, ctx.entid_name))

        if not expansion_result.total_candidates:
            # 当扩展返回空结果时尝试知识图谱补全
            can_kc = getattr(self.args, "enable_kg_completion", False)
            budget = getattr(self.args, "kg_completion_budget", 2)
            if can_kc and ctx.kg_completion_rec["time"] < budget:
                new_dict, add_ids, add_rels, add_heads, _, _ = self.kg_completion_agent.act(
                    ctx, expansion_result.ent_rel_ent_dict
                )
                if add_ids:
                    ctx.kg_completion_rec["time"] += 1
                    ctx.kg_completion_rec["added"] += len(add_ids)
                    ctx.depth_ent_rel_ent_dict[ctx.depth] = new_dict
                    next_topic, next_pre_relations, next_pre_heads = self._prepare_next_entities(
                        ctx, add_ids, add_rels, add_heads
                    )
                    ctx.topic_entity = next_topic
                    ctx.pre_relations = next_pre_relations
                    ctx.pre_heads = next_pre_heads
                    ctx.latest_status = "kg_completion_added_entities"
                    return (
                        f"Depth {ctx.depth}: no candidates from expansion, but KG completion added "
                        f"{len(add_ids)} entities. Next topic entities: "
                        f"{self._format_entity_names(list(ctx.topic_entity.values()))}."
                    )
            # 否则回退
            ctx.latest_status = "no_candidates"
            ctx.topic_entity = {}
            ctx.pending_entities_id = []
            ctx.pre_relations = []
            ctx.pre_heads = []
            ctx.ready_to_finish = False
            return (
                f"Depth {ctx.depth}: no candidate entities found. Consider ReverseSearch, GeneratePartialAnswer, or FallbackCOT."
            )

        pruning_result = self.entity_agent.act(ctx, expansion_result)
        ctx.logger.record_batch(pruning_result.cur_call_time, pruning_result.cur_token)
        ctx.cluster_chain_of_entities.append(pruning_result.chain_of_entities)
        pprint.pprint(convert_dict_name(pruning_result.ent_rel_ent_dict, ctx.entid_name))

        if not pruning_result.flag:
            # 当剪枝返回空结果时尝试知识图谱补全
            can_kc = getattr(self.args, "enable_kg_completion", False)
            budget = getattr(self.args, "kg_completion_budget", 2)
            if can_kc and ctx.kg_completion_rec["time"] < budget:
                new_dict, add_ids, add_rels, add_heads, _, _ = self.kg_completion_agent.act(
                    ctx, pruning_result.ent_rel_ent_dict
                )
                if add_ids:
                    ctx.kg_completion_rec["time"] += 1
                    ctx.kg_completion_rec["added"] += len(add_ids)
                    ctx.depth_ent_rel_ent_dict[ctx.depth] = new_dict
                    next_topic, next_pre_relations, next_pre_heads = self._prepare_next_entities(
                        ctx, add_ids, add_rels, add_heads
                    )
                    ctx.topic_entity = next_topic
                    ctx.pre_relations = next_pre_relations
                    ctx.pre_heads = next_pre_heads
                    ctx.latest_status = "kg_completion_after_prune"
                    return (
                        f"Depth {ctx.depth}: pruning empty, but KG completion added "
                        f"{len(add_ids)} entities. Next topic entities: "
                        f"{self._format_entity_names(list(ctx.topic_entity.values()))}."
                    )
            # 否则回退
            ctx.latest_status = "pruning_empty"
            ctx.topic_entity = {}
            ctx.pending_entities_id = []
            ctx.pre_relations = []
            ctx.pre_heads = []
            ctx.ready_to_finish = False
            return (
                f"Depth {ctx.depth}: entity pruning removed all candidates. Consider ReverseSearch or partial answer."
            )

        ctx.depth_ent_rel_ent_dict[ctx.depth] = pruning_result.ent_rel_ent_dict
        self.memory_agent.act(ctx, pruning_result.ent_rel_ent_dict)
        results, answer, sufficient, _ = self.reasoning_agent.act(ctx, pruning_result.ent_rel_ent_dict)

        ctx.latest_results = results
        normalized_answer = self._humanize_answer(ctx, answer)
        if normalized_answer:
            ctx.latest_results = self._update_results_answer(ctx.latest_results, normalized_answer)
        ctx.latest_answer = normalized_answer if normalized_answer is not None else (answer if answer not in (None, "Null") else None)
        ctx.latest_sufficient = sufficient
        ctx.pending_entities_id = pruning_result.entities_id
        ctx.latest_status = "reasoning_complete"
        ctx.ready_to_finish = False

        # 评估置信度
        try:
            conf, _tok = self.confidence_agent.act(ctx, ctx.latest_results, ctx.latest_answer or "Null", sufficient)
            ctx.latest_confidence = conf
        except Exception:
            ctx.latest_confidence = None

        display_answer = ctx.latest_answer if ctx.latest_answer not in (None, "") else (answer or "Null")
        message = (
            f"Depth {ctx.depth}: reasoning produced answer candidate '{display_answer}' "
            f"(sufficient={sufficient}, confidence={ctx.latest_confidence if ctx.latest_confidence is not None else 'N/A'})."
        )

        if self._should_stop(ctx, display_answer, sufficient):
            ctx.termination_kind = "pending_success"
            ctx.final_answer = ctx.latest_results
            ctx.ready_to_finish = True
            message += " Answer judged sufficient. Call Finish to conclude."
        else:
                message += " Answer insufficient. Consider ReverseSearch or another SearchGraphDepth."
                
                if getattr(self.args, "enable_bridge_answer", True):
                    bridge_resp, tok = generate_without_explored_paths(
                        ctx.question,
                        ctx.subquestions,
                        self.args,
                    )
                    ctx.logger.record_call(tok)
                    # 仅在尚未充分时设置为候选答案
                    ctx.final_answer = bridge_resp
                    message += " A bridge answer from model knowledge is available; you may Finish to output it."

        next_topic, next_pre_relations, next_pre_heads = self._prepare_next_entities(
            ctx,
            pruning_result.entities_id,
            pruning_result.pre_relations,
            pruning_result.pre_heads,
        )

        ctx.topic_entity = next_topic
        ctx.pre_relations = next_pre_relations
        ctx.pre_heads = next_pre_heads

        if not ctx.topic_entity and not ctx.ready_to_finish:
            ctx.latest_status = "no_next_topic_entities"
            message += " No additional topic entities ready; consider ReverseSearch or ending the search."
        else:
            message += (
                f" Next topic entities: {self._format_entity_names(list(ctx.topic_entity.values()))}."
            )

        if self._should_invoke_best_guess(ctx, normalized_answer, sufficient):
            guess = self.best_guess_agent.act(ctx)
            if guess:
                ctx.latest_results = guess["result_json"]
                best_answer = guess["answer"] or "Unknown"
                ctx.latest_answer = guess["answer"] or best_answer
                if guess["confidence"] is not None:
                    ctx.latest_confidence = guess["confidence"]
                ctx.termination_kind = "best_guess"
                ctx.final_answer = ctx.latest_results
                ctx.ready_to_finish = True
                ctx.finished = True
                ctx.topic_entity = {}
                ctx.pending_entities_id = []
                ctx.pre_relations = []
                ctx.pre_heads = []
                ctx.latest_status = "best_guess"
                message += f" Applied general-knowledge best guess to provide an answer: {best_answer}."
                return message

        return message

    def _humanize_answer(self, ctx: AgentContext, answer: Any) -> str | None:
        if answer is None:
            return None
        if isinstance(answer, (list, tuple)):
            resolved = [self._resolve_mid_token(ctx, str(item)) for item in answer]
            resolved = [item for item in resolved if item]
            if resolved:
                return ", ".join(dict.fromkeys(resolved))
            return None
        text = str(answer).strip()
        if not text:
            return None
        lower = text.lower()
        if lower in {"null", "none", "unknown"}:
            return None
        # 尝试解析常见的JSON结构
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                resolved = [self._resolve_mid_token(ctx, str(item)) for item in parsed]
                resolved = [item for item in resolved if item]
                if resolved:
                    return ", ".join(dict.fromkeys(resolved))
                return None
            if isinstance(parsed, dict):
                maybe_answer = parsed.get("Answer") or parsed.get("answer")
                if maybe_answer:
                    return self._humanize_answer(ctx, maybe_answer)
        except json.JSONDecodeError:
            pass
        # 替换文本中嵌入的mid
        if self._looks_like_mid(text):
            mids = re.findall(r"m\.[0-9a-z_]+", text, flags=re.IGNORECASE)
            if mids:
                resolved_tokens = [self._resolve_mid_token(ctx, mid) for mid in mids]
                resolved_tokens = [tok for tok in resolved_tokens if tok and not self._looks_like_mid(tok)]
                if resolved_tokens:
                    return ", ".join(dict.fromkeys(resolved_tokens))
            resolved = self._resolve_mid_token(ctx, text)
            return resolved if resolved else None
        return text

    def _resolve_mid_token(self, ctx: AgentContext, token: str) -> str:
        cleaned = token.strip().strip("'").strip('"')
        if not cleaned:
            return ""
        cached = ctx.entid_name.get(cleaned)
        if cached and not self._looks_like_mid(cached):
            return cached
        if cleaned.startswith("m."):
            try:
                resolved = fb.id2entity_name_or_type(cleaned)
                if resolved and not resolved.startswith("http"):
                    ctx.entid_name[cleaned] = resolved
                    ctx.name_entid[resolved] = cleaned
                    return resolved
            except Exception:
                pass
        return cleaned

    def _looks_like_mid(self, text: str) -> bool:
        if not text:
            return False
        return bool(re.fullmatch(r"m\.[0-9a-z_]+", text.strip(), flags=re.IGNORECASE))

    def _update_results_answer(self, results_text: str | None, answer: str | None) -> str | None:
        if not results_text or not answer:
            return results_text
        try:
            data = json.loads(results_text)
        except json.JSONDecodeError:
            return results_text
        if isinstance(data, dict):
            a_block = data.get("A")
            if isinstance(a_block, dict):
                a_block["Answer"] = answer
                data["A"] = a_block
                return json.dumps(data, ensure_ascii=False)
        return results_text

    def _should_invoke_best_guess(self, ctx: AgentContext, normalized_answer: str | None, sufficient: str) -> bool:
        if not getattr(self.args, "auto_best_guess_on_null", True):
            return False
        if "yes" in str(sufficient).lower():
            return False
        answer_text = (normalized_answer or "").strip().lower()
        needs_guess = not answer_text or answer_text in {"null", "none", "unknown"} or self._looks_like_mid(normalized_answer or "")
        if not needs_guess:
            return False
        if ctx.latest_confidence is not None:
            threshold = getattr(self.args, "confidence_threshold", 0.6)
            if ctx.latest_confidence >= max(0.5, threshold * 0.8):
                return False
        if ctx.depth >= ctx.max_depth:
            return True
        if not ctx.topic_entity:
            return True
        return False

    def _tool_reverse_search(self, ctx: AgentContext, params: Dict[str, Any]) -> str:
        if not ctx.pending_entities_id:
            return "No pending entities available for reverse search. Perform SearchGraphDepth first."
        if ctx.reverse_rec["time"] >= 5:
            return "Reverse search budget exhausted (5 attempts)."

        results_context = ctx.latest_results or ""
        entities_id, add_ent_list = self.reverse_agent.act(ctx, ctx.pending_entities_id, results_context)

        if not add_ent_list:
            ctx.pending_entities_id = entities_id
            ctx.latest_status = "reverse_no_new_entities"
            return "Reverse search did not add new entities."

        ctx.reverse_rec["time"] += 1
        ctx.reverse_rec["ent"].extend(add_ent_list)

        add_entities_id, add_pre_relations, add_pre_heads, new_ent_rel_ent_dict = fb.add_pre_info(
            add_ent_list,
            ctx.depth_ent_rel_ent_dict,
            ctx.depth_ent_rel_ent_dict.get(ctx.depth, {}),
            ctx.entid_name,
            ctx.name_entid,
            self.args,
        )

        ctx.depth_ent_rel_ent_dict[ctx.depth] = new_ent_rel_ent_dict
        ctx.pending_entities_id = entities_id + add_entities_id
        ctx.pre_relations.extend(add_pre_relations)
        ctx.pre_heads.extend(add_pre_heads)

        next_topic, next_pre_relations, next_pre_heads = self._prepare_next_entities(
            ctx,
            ctx.pending_entities_id,
            ctx.pre_relations,
            ctx.pre_heads,
        )
        ctx.topic_entity = next_topic
        ctx.pre_relations = next_pre_relations
        ctx.pre_heads = next_pre_heads
        ctx.latest_status = "reverse_added_entities"

        return (
            "Reverse search added new entities. Updated topic entity focus: "
            + self._format_entity_names(list(ctx.topic_entity.values()))
        )

    def _tool_generate_partial_answer(self, ctx: AgentContext, params: Dict[str, Any]) -> str:
        answer = self.terminator.partial(ctx)
        ctx.termination_kind = "partial"
        ctx.final_answer = answer
        ctx.final_saved = True
        ctx.ready_to_finish = True
        return "Generated partial answer and saved results. Call Finish to exit."

    def _tool_fallback_answer(self, ctx: AgentContext, params: Dict[str, Any]) -> str:
        response = self.terminator.fallback(ctx)
        ctx.termination_kind = "fallback"
        ctx.final_answer = response
        ctx.final_saved = True
        ctx.ready_to_finish = True
        return "Fallback chain-of-thought answer generated and saved. Call Finish to conclude."

    def _tool_complete_kg(self, ctx: AgentContext, params: Dict[str, Any]) -> str:
        if ctx.depth == 0:
            return "Run SearchGraphDepth at least once before KG completion."
        budget = getattr(self.args, "kg_completion_budget", 2)
        if ctx.kg_completion_rec["time"] >= budget:
            return f"KG completion budget exhausted ({budget})."
        cur_dict = ctx.depth_ent_rel_ent_dict.get(ctx.depth, {})
        new_dict, add_ids, add_rels, add_heads, _, _ = self.kg_completion_agent.act(ctx, cur_dict)
        if not add_ids:
            ctx.latest_status = "kg_completion_noop"
            return "KG completion did not add new entities."
        ctx.kg_completion_rec["time"] += 1
        ctx.kg_completion_rec["added"] += len(add_ids)
        ctx.depth_ent_rel_ent_dict[ctx.depth] = new_dict
        next_topic, next_pre_relations, next_pre_heads = self._prepare_next_entities(
            ctx, add_ids, add_rels, add_heads
        )
        ctx.topic_entity = next_topic
        ctx.pre_relations = next_pre_relations
        ctx.pre_heads = next_pre_heads
        ctx.latest_status = "kg_completion_added_entities"
        return (
            "KG completion added new entities. Updated topic focus: "
            + self._format_entity_names(list(ctx.topic_entity.values()))
        )

    def _tool_assess_confidence(self, ctx: AgentContext, params: Dict[str, Any]) -> str:
        if not ctx.latest_results or ctx.latest_answer is None:
            return "No reasoning result to assess. Run SearchGraphDepth first."
        conf, _ = self.confidence_agent.act(ctx, ctx.latest_results, ctx.latest_answer, ctx.latest_sufficient or "")
        ctx.latest_confidence = conf
        return f"Assessed confidence: {conf:.2f}."

    def _prepare_next_entities(
        self,
        ctx: AgentContext,
        entities_id: List[str],
        pre_relations: List[str],
        pre_heads: List[bool],
    ) -> Tuple[Dict[str, str], List[str], List[bool]]:
        topic_entity: Dict[str, str] = {}
        next_pre_relations: List[str] = []
        next_pre_heads: List[bool] = []

        for ent, rel, head in zip(entities_id, pre_relations, pre_heads):
            if ent == "[FINISH_ID]" or fb.if_topic_non_retrieve(ent):
                continue
            if not ent.startswith("m."):
                continue
            if ent not in ctx.entid_name:
                continue
            if ent in topic_entity:
                continue
            topic_entity[ent] = ctx.entid_name[ent]
            next_pre_relations.append(rel)
            next_pre_heads.append(head)

        return topic_entity, next_pre_relations, next_pre_heads

    def _should_stop(self, ctx: AgentContext, answer: str, sufficient: str) -> bool:
        if answer is None:
            return False
        answer_lower = str(answer).lower()
        if answer_lower in {"null", "none"}:
            return False
        if answer_lower.startswith("m.") or answer_lower.startswith('["m.') or answer_lower.startswith("['m."):
            return False
        if "yes" not in str(sufficient).lower():
            return False
        # 考虑置信度
        if getattr(self.args, "enable_confidence_gate", False):
            threshold = getattr(self.args, "confidence_threshold", 0.6)
            if ctx.latest_confidence is None:
                return True
            return ctx.latest_confidence >= threshold
        # 默认充分即停止
        return True

    def _resolve_memory_dir(self, question: str) -> str:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mem"))
        safe_question = question[:255]
        mem_dir = os.path.join(base_dir, self.args.dataset, self.args.LLM_type, safe_question)
        return mem_dir
