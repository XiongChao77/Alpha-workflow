"""Provider adapters for structured generation and research decisions."""

import json
import logging
import time
from hashlib import sha256
from pathlib import Path

from .config import ModelConfig
from .parameter_search import normalize_plan
from .transport import JsonTransport, RemoteError

log = logging.getLogger(__name__)


GENERATE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"candidates": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "properties": {"hypothesis": {"type": "string"}, "expression_template": {"type": "string"},
                       "period": {"type": "object", "additionalProperties": False,
                                  "properties": {"name": {"type": "string"}, "default": {"type": "integer"}},
                                  "required": ["name", "default"]},
                       "parameters": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                           "properties": {"name": {"type": "string"}, "default": {"type": "number"},
                                          "values": {"type": "array", "items": {"type": "number"}}},
                           "required": ["name", "default", "values"]}}},
        "required": ["hypothesis", "expression_template", "period", "parameters"]}}}, "required": ["candidates"]}
DECIDE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"action": {"type": "string", "enum": ["NEW_DIRECTION", "OPTIMIZE", "TUNE", "SUBMIT"]},
                   "reason": {"type": "string"}, "target_id": {"type": "string"},
                   "next_direction": {"type": "string"},
                   "optimized_hypothesis": {"type": "string"},
                   "optimized_expression": {"type": "string"},
                   "parameter_changes": {"type": "object", "additionalProperties": False,
                       "properties": {"decay": {"type": "integer"}, "truncation": {"type": "number"},
                                      "neutralization": {"type": "string"}},
                       "required": ["decay", "truncation", "neutralization"]}},
    "required": ["action", "reason", "target_id", "next_direction", "parameter_changes",
                 "optimized_hypothesis", "optimized_expression"]}
PARAMETER_DECISION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"explore": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["explore", "reason"]}
SUBMISSION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"target_id": {"type": "string"}, "reason": {"type": "string"}},
    "required": ["target_id", "reason"]}


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
        log.info("模型请求 stage=%s provider=%s model=%s format=%s", name, self.config.provider,
                 self.config.model, self.config.api_format)
        log.debug("模型输入 stage=%s system=%s context=%s", name, system, json.dumps(user, ensure_ascii=False))
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
        try:
            _, _, data = self.http.request("POST", url, payload, headers)
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
                    _, _, data = self.http.request("POST", url, payload, headers)
                except RemoteError as second:
                    if second.status not in {400, 422} or not any(
                        token in str(second).lower() for token in ("json_object", "text.format")
                    ):
                        raise
                    payload.pop("text", None)
                    _, _, data = self.http.request("POST", url, payload, headers)
            else:
                payload.pop("response_format", None)
                _, _, data = self.http.request("POST", url, payload, headers)
        try:
            if self.config.api_format == "responses":
                parts = [c["text"] for item in data.get("output", []) if item.get("type") == "message"
                         for c in item.get("content", []) if c.get("type") == "output_text"]
                raw = "".join(parts) or data.get("output_text", "")
            else:
                raw = data["choices"][0]["message"]["content"]
            if not raw or not raw.strip():
                raise ValueError("empty model output")
            log.debug("模型原始文本 stage=%s text=%s", name, raw)
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("model output must be an object")
            log.debug("模型输出 stage=%s response=%s", name, json.dumps(value, ensure_ascii=False))
            log.info("模型响应 stage=%s elapsed_seconds=%.1f", name, time.monotonic() - started)
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
        result = self._call(system, context, GENERATE_SCHEMA, "alpha_candidates")
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
        if result["action"] == "OPTIMIZE" and not all(result[field].strip() for field in
                                                    ("optimized_hypothesis", "optimized_expression")):
            raise ValueError("OPTIMIZE requires nonempty optimized_hypothesis and optimized_expression")
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
