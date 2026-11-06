"""Provider adapters for structured generation and research decisions."""

import json
from datetime import datetime
from itertools import count
import logging
from threading import Lock
import time
from hashlib import sha256
from pathlib import Path

from .config import ModelConfig
from .logging_setup import conversation_path, save_model_exchange
from .parameter_search import normalize_plan
from .transport import JsonTransport, RemoteError

log = logging.getLogger(__name__)
_request_numbers = count(1)
_request_lock = Lock()


def _next_request_number() -> int:
    with _request_lock:
        return next(_request_numbers)


GENERATE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"candidates": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "properties": {"hypothesis": {"type": "string"}, "expression_template": {"type": "string"},
                       "parameters": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                           "properties": {"name": {"type": "string"}, "default": {"type": "number"},
                                          "values": {"type": "array", "items": {"type": "number"}}},
                           "required": ["name", "default", "values"]}}},
        "required": ["hypothesis", "expression_template", "parameters"]}}}, "required": ["candidates"]}
LEGACY_GENERATE_SCHEMA = json.loads(json.dumps(GENERATE_SCHEMA))
_legacy_candidate = LEGACY_GENERATE_SCHEMA["properties"]["candidates"]["items"]
_legacy_candidate["properties"]["period"] = {
    "type": "object", "additionalProperties": False,
    "properties": {"name": {"type": "string"}, "default": {"type": "integer"}},
    "required": ["name", "default"]}
_legacy_candidate["required"].append("period")
DECIDE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"action": {"type": "string", "enum": ["NEW_DIRECTION", "OPTIMIZE", "TUNE", "SUBMIT"]},
                   "reason": {"type": "string"}, "target_id": {"type": "string"},
                   "next_direction": {"type": "string"},
                   "optimized_hypothesis": {"type": "string"},
                   "optimized_expression": {"type": "string"},
                   "optimized_candidates": {"type": "array", "maxItems": 2,
                       "items": {"type": "object", "additionalProperties": False,
                       "properties": {"hypothesis": {"type": "string"}, "expression": {"type": "string"}},
                       "required": ["hypothesis", "expression"]}},
                   "parameter_changes": {"type": "object", "additionalProperties": False,
                       "properties": {"decay": {"type": "integer"}, "truncation": {"type": "number"},
                                      "neutralization": {"type": "string"}},
                       "required": ["decay", "truncation", "neutralization"]}},
    "required": ["action", "reason", "target_id", "next_direction", "parameter_changes",
                 "optimized_hypothesis", "optimized_expression", "optimized_candidates"]}
PARAMETER_DECISION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"explore": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["explore", "reason"]}
SUBMISSION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"target_id": {"type": "string"}, "reason": {"type": "string"}},
    "required": ["target_id", "reason"]}
TOPIC_SCHEMA = {"type": "object", "additionalProperties": False,
                "properties": {"title": {"type": "string"}, "objective": {"type": "string"},
                               "rationale": {"type": "string"}},
                "required": ["title", "objective", "rationale"]}
DIRECTIONS_SCHEMA = {"type": "object", "additionalProperties": False,
                     "properties": {"directions": {"type": "array", "items": {
                         "type": "object", "additionalProperties": False,
                         "properties": {"title": {"type": "string"}, "hypothesis": {"type": "string"},
                                        "mechanism": {"type": "string"}},
                         "required": ["title", "hypothesis", "mechanism"]}}},
                     "required": ["directions"]}
FEEDBACK_SCHEMA = {"type": "object", "additionalProperties": False,
                   "properties": {"action": {"type": "string", "enum": ["EXPLORE_PARAM2", "ITERATE", "SWITCH_DIRECTION", "COMPLETE_TOPIC"]},
                                  "reason": {"type": "string"}, "next_goal": {"type": "string"},
                                  "conclusion": {"type": "string"},
                                  "optimized_candidates": {"type": "array", "maxItems": 2, "items": {
                                      "type": "object", "additionalProperties": False,
                                      "properties": {"parent_candidate_id": {"type": "string"},
                                                     "hypothesis": {"type": "string"},
                                                     "expression_template": {"type": "string"},
                                                     "parameters": GENERATE_SCHEMA["properties"]["candidates"]["items"]["properties"]["parameters"]},
                                      "required": ["parent_candidate_id", "hypothesis", "expression_template", "parameters"]}}},
                   "required": ["action", "reason", "next_goal", "conclusion", "optimized_candidates"]}
REPAIR_SCHEMA = {"type": "object", "additionalProperties": False,
                 "properties": {"hypothesis": {"type": "string"}, "expression": {"type": ["string", "null"]},
                                "reason": {"type": "string"}},
                 "required": ["hypothesis", "expression", "reason"]}
BRAINSTORM_SCHEMA = {"type": "object", "additionalProperties": False,
                     "properties": {"response": {"type": "string"}, "topic": TOPIC_SCHEMA},
                     "required": ["response", "topic"]}


class LLM:
    def __init__(self, config: ModelConfig, transport: JsonTransport | None = None):
        if not config.api_key:
            raise ValueError(f"API key is required for {config.provider}")
        self.config = config
        self.http = transport or JsonTransport(timeout=90)

    @staticmethod
    def _prompt(path: Path) -> str:
        path = Path(path)
        try:
            value = path.read_text(encoding="utf-8-sig").strip()
        except OSError as exc:
            raise ValueError(f"Cannot read prompt file: {path}") from exc
        if not value:
            raise ValueError(f"Prompt file is empty: {path}")
        log.info("提示词文件 path=%s sha256=%s", path, sha256(value.encode()).hexdigest()[:12])
        return value

    def _call(self, system: str, user: dict, schema: dict, name: str) -> dict:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]
        base = self.config.base_url
        if not base.endswith("/v1") and self.config.provider == "openai":
            base += "/v1"
        if self.config.api_format == "responses":
            url = base + "/responses"
            payload = {"model": self.config.model, "instructions": system,
                       "input": [{"role": "user", "content": messages[1]["content"]}]}
            mode = "json_schema" if self.config.json_mode == "auto" else self.config.json_mode
            if mode == "json_schema":
                payload["text"] = {"format": {"type": "json_schema", "name": name,
                                              "strict": True, "schema": schema}}
            elif mode == "json_object":
                payload["text"] = {"format": {"type": "json_object"}}
        else:
            url = base + "/chat/completions"
            payload = {"model": self.config.model, "messages": messages}
            mode = "json_object" if self.config.json_mode == "auto" else self.config.json_mode
            if mode == "json_schema":
                payload["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": name, "strict": True, "schema": schema}}
            elif mode == "json_object":
                payload["response_format"] = {"type": "json_object"}
        if self.config.disable_response_storage:
            payload["store"] = False
        headers = {"Authorization": "Bearer " + self.config.api_key}
        started = time.monotonic()

        def send(attempt: str):
            request_no = _next_request_number()
            exchange = {"timestamp": datetime.now().astimezone().isoformat(),
                        "request_no": request_no, "stage": name, "attempt": attempt,
                        "provider": self.config.provider, "model": self.config.model,
                        "api_format": self.config.api_format, "messages": messages}
            def journal(record: dict):
                try:
                    save_model_exchange(record)
                except OSError as journal_error:
                    log.error("模型对话保存失败 path=%s reason=%s", conversation_path(), journal_error)

            journal({**exchange, "event": "request"})
            log.info("模型请求 #%s stage=%s attempt=%s provider=%s model=%s format=%s",
                     request_no, name, attempt, self.config.provider, self.config.model, self.config.api_format)
            log.debug("模型输入 request_no=%s stage=%s system=%s context=%s", request_no, name,
                      system, json.dumps(user, ensure_ascii=False))
            try:
                _, _, response = self.http.request("POST", url, payload, headers)
            except RemoteError as error:
                exchange["error"] = {"status": error.status, "message": str(error)}
                journal({**exchange, "event": "failed"})
                log.warning("模型请求失败 #%s stage=%s http=%s reason=%s", request_no, name,
                            error.status, error)
                raise
            exchange["response"] = response
            journal({**exchange, "event": "completed"})
            return response, request_no

        try:
            data, request_no = send("initial")
        except RemoteError as exc:
            # OpenAI-compatible gateways vary in structured-output support.
            # In auto mode, retain the JSON prompt and local validation.
            error = str(exc).lower()
            format_rejected = exc.status in {400, 422} and any(
                token in error for token in ("response_format", "json_schema", "json_object", "text.format")
            )
            if self.config.json_mode != "auto" or not format_rejected:
                raise
            log.warning("模型网关拒绝结构化格式，尝试兼容模式 stage=%s http=%s", name, exc.status)
            if self.config.api_format == "responses" and mode == "json_schema":
                payload["text"] = {"format": {"type": "json_object"}}
                try:
                    data, request_no = send("json_object_fallback")
                except RemoteError as second:
                    if second.status not in {400, 422} or not any(
                        token in str(second).lower() for token in ("json_object", "text.format")
                    ):
                        raise
                    payload.pop("text", None)
                    data, request_no = send("plain_json_fallback")
            else:
                payload.pop("response_format", None)
                data, request_no = send("plain_json_fallback")
        try:
            if self.config.api_format == "responses":
                parts = [c["text"] for item in data.get("output", []) if item.get("type") == "message"
                         for c in item.get("content", []) if c.get("type") == "output_text"]
                raw = "".join(parts) or data.get("output_text", "")
            else:
                raw = data["choices"][0]["message"]["content"]
            if not raw or not raw.strip():
                raise ValueError("empty model output")
            log.debug("模型原始文本 request_no=%s stage=%s text=%s", request_no, name, raw)
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("model output must be an object")
            log.debug("模型输出 request_no=%s stage=%s response=%s", request_no, name,
                      json.dumps(value, ensure_ascii=False))
            log.info("模型响应 #%s stage=%s elapsed_seconds=%.1f", request_no, name,
                     time.monotonic() - started)
            return value
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise RemoteError(0, f"Invalid {self.config.provider} response: {exc}") from exc

    def probe(self) -> bool:
        """Make one minimal model request without touching BRAIN."""
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
        result = self._call(self._prompt(self.config.probe_prompt_file), {"task": "connectivity check"},
                            schema, "connectivity_check")
        return result.get("ok") is True

    def generate(self, context: dict) -> list[dict]:
        system = self._prompt(self.config.generate_prompt_file)
        result = self._call(system, context, LEGACY_GENERATE_SCHEMA, "alpha_candidates")
        candidates = result.get("candidates")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= int(context["requested_count"]):
            count = len(candidates) if isinstance(candidates, list) else type(candidates).__name__
            raise ValueError(f"candidates must contain 1..{context['requested_count']} items; got {count}")
        normalized = []
        for index, candidate in enumerate(candidates, 1):
            if not isinstance(candidate, dict):
                raise ValueError(f"candidate {index}: expected object; got {type(candidate).__name__}")
            hypothesis = candidate.get("hypothesis")
            if not isinstance(hypothesis, str) or not hypothesis.strip():
                hypothesis = candidate.get("economic_hypothesis")
                if isinstance(hypothesis, str) and hypothesis.strip():
                    log.info("模型字段映射 candidate=%s economic_hypothesis -> hypothesis", index)
            if not isinstance(hypothesis, str) or not hypothesis.strip():
                raise ValueError(f"candidate {index}: missing nonempty hypothesis/economic_hypothesis; keys={list(candidate)}")
            try:
                plan = normalize_plan(candidate)
            except ValueError as exc:
                raise ValueError(f"candidate {index}: {exc}") from exc
            normalized.append({"hypothesis": hypothesis.strip(), **plan})
        return normalized

    def decide(self, context: dict) -> dict:
        system = self._prompt(self.config.decide_prompt_file)
        result = self._call(system, context, DECIDE_SCHEMA, "research_decision")
        if result.get("action") not in {"NEW_DIRECTION", "OPTIMIZE", "TUNE", "SUBMIT"}:
            raise ValueError("Invalid decision action")
        if not isinstance(result.get("reason"), str) or not result["reason"].strip():
            raise ValueError("Decision needs a reason")
        if result.get("next_direction") is None:
            result["next_direction"] = ""
            log.info("模型决策字段映射 next_direction=null -> 保持当前方向")
        if result.get("target_id") is None and result["action"] == "NEW_DIRECTION":
            result["target_id"] = ""
            log.info("模型决策字段映射 target_id=null -> 无目标候选")
        for field in ("target_id", "next_direction"):
            if not isinstance(result.get(field), str):
                raise ValueError(f"Decision {field} must be a string; got {type(result.get(field)).__name__}")
        changes = result.get("parameter_changes")
        if changes is None and result["action"] != "TUNE":
            changes = {"decay": 0, "truncation": 0.08, "neutralization": "SUBINDUSTRY"}
            result["parameter_changes"] = changes
            log.info("模型决策字段映射 parameter_changes=null -> 默认未使用参数")
        if not isinstance(changes, dict) or set(changes) != {"decay", "truncation", "neutralization"}:
            raise ValueError("Invalid parameter changes")
        if (type(changes["decay"]) is not int or type(changes["truncation"]) not in {int, float}
                or not isinstance(changes["neutralization"], str)):
            raise ValueError("Invalid parameter change types")
        for field in ("optimized_hypothesis", "optimized_expression"):
            if result.get(field) is None and result["action"] != "OPTIMIZE":
                result[field] = ""
            if not isinstance(result.get(field), str):
                raise ValueError(f"Decision {field} must be a string; got {type(result.get(field)).__name__}")
        candidates = result.get("optimized_candidates", [])
        if not isinstance(candidates, list) or len(candidates) > 2:
            raise ValueError("optimized_candidates must contain at most two items")
        if result["action"] == "OPTIMIZE" and not candidates:
            if not all(result[field].strip() for field in ("optimized_hypothesis", "optimized_expression")):
                raise ValueError("OPTIMIZE requires optimized_candidates or a nonempty legacy hypothesis and expression")
            candidates = [{"hypothesis": result["optimized_hypothesis"],
                           "expression": result["optimized_expression"]}]
        for index, candidate in enumerate(candidates, 1):
            if (not isinstance(candidate, dict) or not isinstance(candidate.get("hypothesis"), str)
                    or not candidate["hypothesis"].strip() or not isinstance(candidate.get("expression"), str)
                    or not candidate["expression"].strip()):
                raise ValueError(f"optimized candidate {index} needs nonempty hypothesis and expression")
        if result["action"] != "OPTIMIZE" and candidates:
            raise ValueError("optimized_candidates is only valid for OPTIMIZE")
        result["optimized_candidates"] = candidates
        return result

    def decide_parameter(self, context: dict) -> dict:
        result = self._call(self._prompt(self.config.parameter_prompt_file), context,
                            PARAMETER_DECISION_SCHEMA, "parameter_decision")
        if type(result.get("explore")) is not bool or not isinstance(result.get("reason"), str) or not result["reason"].strip():
            raise ValueError("Parameter decision requires explore boolean and nonempty reason")
        return result

    def choose_submission(self, context: dict) -> dict:
        result = self._call(self._prompt(self.config.submission_prompt_file), context,
                            SUBMISSION_SCHEMA, "submission_selection")
        if not isinstance(result.get("target_id"), str) or not result["target_id"].strip():
            raise ValueError("Submission selection requires target_id")
        if not isinstance(result.get("reason"), str) or not result["reason"].strip():
            raise ValueError("Submission selection requires reason")
        return result

    def generate_topic(self, context: dict) -> dict:
        result = self._call(self._prompt(self.config.topic_prompt_file), context, TOPIC_SCHEMA, "research_topic")
        if any(not isinstance(result.get(key), str) or not result[key].strip()
               for key in ("title", "objective", "rationale")):
            raise ValueError("Topic needs title, objective and rationale")
        return result

    def propose_directions(self, context: dict) -> list[dict]:
        result = self._call(self._prompt(self.config.directions_prompt_file), context,
                            DIRECTIONS_SCHEMA, "research_directions")
        items = result.get("directions")
        if not isinstance(items, list) or not 1 <= len(items) <= context["max_directions"]:
            raise ValueError("Invalid direction count")
        for item in items:
            if not isinstance(item, dict) or any(not isinstance(item.get(key), str) or not item[key].strip()
                                                 for key in ("title", "hypothesis", "mechanism")):
                raise ValueError("Each direction needs title, hypothesis and mechanism")
        return items

    def analyze_batch(self, context: dict) -> dict:
        result = self._call(self._prompt(self.config.feedback_prompt_file), context,
                            FEEDBACK_SCHEMA, "research_feedback")
        if result.get("action") not in {"EXPLORE_PARAM2", "ITERATE", "SWITCH_DIRECTION", "COMPLETE_TOPIC"}:
            raise ValueError("Invalid batch decision action")
        if any(not isinstance(result.get(key), str) for key in ("reason", "next_goal", "conclusion")) or not result["reason"].strip():
            raise ValueError("Batch decision needs reason and string fields")
        candidates = result.get("optimized_candidates", [])
        if not isinstance(candidates, list) or len(candidates) > 2:
            raise ValueError("Optimized candidates must contain at most two formulas")
        if result["action"] == "ITERATE" and not candidates:
            raise ValueError("ITERATE requires optimized_candidates")
        allowed = {candidate["id"] for candidate in context["statistics"]["candidates"]}
        normalized = []
        for candidate in candidates:
            if not isinstance(candidate, dict) or candidate.get("parent_candidate_id") not in allowed:
                raise ValueError("Optimized candidate must reference a candidate in this experiment")
            if "period" in candidate:
                raise ValueError("Optimized candidate must use model-chosen parameters only")
            hypothesis = candidate.get("hypothesis")
            if not isinstance(hypothesis, str) or not hypothesis.strip():
                raise ValueError("Optimized candidate needs a hypothesis")
            normalized.append({"parent_candidate_id": candidate["parent_candidate_id"],
                               "hypothesis": hypothesis.strip(), **normalize_plan(candidate)})
        if result["action"] != "ITERATE" and normalized:
            raise ValueError("Only ITERATE may contain optimized candidates")
        result["optimized_candidates"] = normalized
        return result

    def generate_variants(self, context: dict) -> list[dict]:
        result = self._call(self._prompt(self.config.variants_prompt_file), context,
                            GENERATE_SCHEMA, "alpha_variants")
        items = result.get("candidates")
        if not isinstance(items, list) or not 1 <= len(items) <= context["requested_count"]:
            raise ValueError("Invalid structural variant count")
        normalized = []
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("hypothesis"), str) or not item["hypothesis"].strip():
                raise ValueError("Variant needs a hypothesis")
            if "period" in item:
                raise ValueError("Variant must use model-chosen parameters only")
            normalized.append({"hypothesis": item["hypothesis"].strip(), **normalize_plan(item)})
        return normalized

    def repair_expression(self, context: dict) -> dict:
        result = self._call(self._prompt(self.config.repair_prompt_file), context,
                            REPAIR_SCHEMA, "alpha_repair")
        if any(not isinstance(result.get(key), str) or not result[key].strip()
               for key in ("hypothesis", "reason")):
            raise ValueError("Repair needs hypothesis and reason")
        if "expression" not in result or (result["expression"] is not None and not isinstance(result["expression"], str)):
            raise ValueError("Repair expression must be a string or null")
        return result

    def brainstorm_topic(self, context: dict) -> dict:
        result = self._call(self._prompt(self.config.brainstorm_prompt_file), context,
                            BRAINSTORM_SCHEMA, "topic_brainstorm")
        if not isinstance(result.get("response"), str) or not result["response"].strip():
            raise ValueError("Brainstorm needs a response")
        topic = result.get("topic")
        if not isinstance(topic, dict) or any(not isinstance(topic.get(key), str) or not topic[key].strip()
                                             for key in ("title", "objective", "rationale")):
            raise ValueError("Brainstorm needs a proposed topic")
        return result
