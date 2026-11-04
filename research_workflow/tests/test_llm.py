import json
import unittest
from dataclasses import replace
from pathlib import Path
import tempfile

from research_workflow.config import ModelConfig
from research_workflow.llm import LLM
from research_workflow.transport import RemoteError


class FakeTransport:
    def __init__(self, response, reject_json_mode=False):
        self.response = response
        self.reject_json_mode = reject_json_mode
        self.calls = []

    def request(self, method, url, payload=None, headers=None):
        self.calls.append((method, url, payload, headers))
        if self.reject_json_mode and len(self.calls) == 1:
            raise RemoteError(400, "response_format json_object unsupported")
        return 200, {}, self.response


class LLMCompatibilityTests(unittest.TestCase):
    def test_batch_optimization_returns_direction_and_formula_in_one_call(self):
        response = {"action": "ITERATE", "reason": "Invert and smooth", "next_goal": "test a stable inverse",
                    "conclusion": "Original sign was wrong", "optimized_candidates": [{
                        "parent_candidate_id": "candidate1", "hypothesis": "Inverse impact",
                        "expression_template": "-rank(ts_mean(returns,{Period}))",
                        "period": {"name": "Period", "default": 45}, "parameters": []}]}
        fake = FakeTransport({"choices": [{"message": {"content": json.dumps(response)}}]})
        model = LLM(ModelConfig("openai", "test-model", "fake-key", "https://example.test/v1",
                                "chat_completions"), fake)
        result = model.analyze_batch({"statistics": {"candidates": [{"id": "candidate1"}]}})
        self.assertEqual(result["optimized_candidates"][0]["parent_candidate_id"], "candidate1")
        self.assertEqual(len(fake.calls), 1)

    def test_third_party_chat_completions_base_url(self):
        fake = FakeTransport({"choices": [{"message": {"content": '{"action":"NEW_DIRECTION"}'}}]})
        config = ModelConfig("openai", "gpt-6.1-sol", "fake-key", "https://example.test/v1",
                             "chat_completions", "auto", True)
        result = LLM(config, fake)._call("Return JSON", {"topic": "alpha"}, {}, "test")
        self.assertEqual(result["action"], "NEW_DIRECTION")
        method, url, payload, headers = fake.calls[0]
        self.assertEqual(url, "https://example.test/v1/chat/completions")
        self.assertEqual(payload["model"], "gpt-6.1-sol")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertIs(payload["store"], False)
        self.assertEqual(headers["Authorization"], "Bearer fake-key")

    def test_responses_api_uses_configured_gateway_path(self):
        fake = FakeTransport({"output": [{"type": "message", "content": [
            {"type": "output_text", "text": '{"candidates":[]}'}]}]})
        config = ModelConfig("openai", "gpt-5.5", "fake-key", "https://example.test/v1",
                             "responses", "auto", False)
        self.assertEqual(LLM(config, fake)._call("Return JSON", {}, {}, "test"), {"candidates": []})
        _, url, payload, _ = fake.calls[0]
        self.assertEqual(url, "https://example.test/v1/responses")
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertNotIn("store", payload)

    def test_auto_falls_back_when_gateway_rejects_json_mode(self):
        fake = FakeTransport({"choices": [{"message": {"content": '{"ok":true}'}}]}, True)
        config = ModelConfig("openai", "any-model", "fake-key", "https://example.test/v1",
                             "chat_completions", "auto", False)
        self.assertEqual(LLM(config, fake)._call("Return JSON", {}, {}, "test"), {"ok": True})
        self.assertEqual(len(fake.calls), 2)
        self.assertNotIn("response_format", fake.calls[1][2])

    def test_probe_only_calls_model_endpoint(self):
        fake = FakeTransport({"choices": [{"message": {"content": '{"ok":true}'}}]})
        config = ModelConfig("openai", "any-model", "fake-key", "https://example.test/v1",
                             "chat_completions", "auto", False)
        self.assertTrue(LLM(config, fake).probe())
        self.assertEqual(len(fake.calls), 1)
        self.assertTrue(fake.calls[0][1].endswith("/chat/completions"))

    def test_generation_accepts_economic_hypothesis_alias(self):
        fake = FakeTransport({"choices": [{"message": {"content": (
            '{"candidates":[{"id":"idea_1","economic_hypothesis":"Volume reversal",'
            '"expression_template":"-rank(ts_mean(returns, {Period}))",'
            '"period":{"name":"Period","default":45},"parameters":[]}]}'
        )}}]})
        config = ModelConfig("openai", "test-model", "fake-key", "https://example.test/v1",
                             "chat_completions")
        candidates = LLM(config, fake).generate({"requested_count": 3})
        self.assertEqual(candidates[0]["hypothesis"], "Volume reversal")
        self.assertEqual(candidates[0]["period"]["default"], 45)
        self.assertEqual(candidates[0]["expression_template"], "-rank(ts_mean(returns, {Period}))")

    def test_generation_reports_candidate_index_and_missing_field(self):
        fake = FakeTransport({"choices": [{"message": {"content": (
            '{"candidates":[{"hypothesis":"Idea","expression_template":"rank(ts_mean(close, {Period}))",'
            '"period":{"name":"Period","default":30},"parameters":[]},'
            '{"economic_hypothesis":"Another idea"}]}'
        )}}]})
        config = ModelConfig("openai", "test-model", "fake-key", "https://example.test/v1",
                             "chat_completions")
        with self.assertRaisesRegex(ValueError, "candidate 2: expression_template must be a nonempty string"):
            LLM(config, fake).generate({"requested_count": 3})

    def test_tune_decision_accepts_null_optional_direction(self):
        fake = FakeTransport({"choices": [{"message": {"content": (
            '{"action":"TUNE","reason":"Lower turnover","target_id":"candidate123",'
            '"next_direction":null,"parameter_changes":{"decay":5,"truncation":0.08,'
            '"neutralization":"SUBINDUSTRY"}}'
        )}}]})
        config = ModelConfig("openai", "test-model", "fake-key", "https://example.test/v1",
                             "chat_completions")
        decision = LLM(config, fake).decide({})
        self.assertEqual(decision["next_direction"], "")
        self.assertEqual(decision["parameter_changes"]["decay"], 5)

    def test_tune_still_requires_target_id(self):
        fake = FakeTransport({"choices": [{"message": {"content": (
            '{"action":"TUNE","reason":"Lower turnover","target_id":null,'
            '"next_direction":null,"parameter_changes":{"decay":5,"truncation":0.08,'
            '"neutralization":"SUBINDUSTRY"}}'
        )}}]})
        config = ModelConfig("openai", "test-model", "fake-key", "https://example.test/v1",
                             "chat_completions")
        with self.assertRaisesRegex(ValueError, "Decision target_id must be a string"):
            LLM(config, fake).decide({})

    def test_generate_prompt_file_is_reloaded_for_each_call(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generate.txt"
            path.write_text("First custom prompt", encoding="utf-8")
            fake = FakeTransport({"choices": [{"message": {"content": (
                '{"candidates":[{"hypothesis":"Idea","expression_template":"rank(ts_mean(close, {Period}))",'
                '"period":{"name":"Period","default":30},"parameters":[]}]}'
            )}}]})
            config = replace(ModelConfig("openai", "test-model", "fake-key", "https://example.test/v1",
                                         "chat_completions"), generate_prompt_file=path)
            model = LLM(config, fake)
            model.generate({"requested_count": 1})
            path.write_text("Second custom prompt", encoding="utf-8")
            model.generate({"requested_count": 1})
            self.assertEqual(fake.calls[0][2]["messages"][0]["content"], "First custom prompt")
            self.assertEqual(fake.calls[1][2]["messages"][0]["content"], "Second custom prompt")

    def test_optimize_requires_new_hypothesis_and_expression(self):
        fake = FakeTransport({"choices": [{"message": {"content": (
            '{"action":"OPTIMIZE","reason":"Reduce turnover","target_id":"candidate123",'
            '"next_direction":"","parameter_changes":{"decay":0,"truncation":0.08,'
            '"neutralization":"SUBINDUSTRY"},"optimized_hypothesis":"Smoother reversal",'
            '"optimized_expression":"-rank(ts_mean(returns, 5))"}'
        )}}]})
        config = ModelConfig("openai", "test-model", "fake-key", "https://example.test/v1",
                             "chat_completions")
        decision = LLM(config, fake).decide({})
        self.assertEqual(decision["optimized_hypothesis"], "Smoother reversal")
        self.assertEqual(decision["optimized_expression"], "-rank(ts_mean(returns, 5))")

    def test_optimize_accepts_two_new_candidates(self):
        fake = FakeTransport({"choices": [{"message": {"content": (
            '{"action":"OPTIMIZE","reason":"Compare two variations","target_id":"candidate123",'
            '"next_direction":"","parameter_changes":{"decay":0,"truncation":0.08,'
            '"neutralization":"SUBINDUSTRY"},"optimized_hypothesis":"",'
            '"optimized_expression":"","optimized_candidates":['
            '{"hypothesis":"Center rank","expression":"0.5-rank(close/vwap)"},'
            '{"hypothesis":"Smooth rank","expression":"ts_mean(rank(close/vwap),3)"}]}'
        )}}]})
        config = ModelConfig("openai", "test-model", "fake-key", "https://example.test/v1",
                             "chat_completions")
        decision = LLM(config, fake).decide({})
        self.assertEqual(len(decision["optimized_candidates"]), 2)


if __name__ == "__main__":
    unittest.main()
