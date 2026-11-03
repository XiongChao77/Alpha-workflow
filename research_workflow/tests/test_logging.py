import logging
import os
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_workflow.config import Config, ModelConfig
from research_workflow.llm import LLM
from research_workflow.logging_setup import setup_logging
from research_workflow.tests.test_llm import FakeTransport
from research_workflow.tests.test_workflow import Brain, Generator, Judge
from research_workflow.engine import Engine
from research_workflow.store import Store


class LoggingTests(unittest.TestCase):
    def test_model_and_brain_activity_is_recorded_without_credentials(self):
        with tempfile.TemporaryDirectory() as temp:
            self._check(Path(temp))

    def _check(self, root):
        log_path = root / "workflow.log"
        config = replace(Config(),
                         db_path=root / "state.sqlite3", library_dir=root / "library", auto_submit=True)
        store = Store(config.db_path)
        run_id = store.create_run("test alpha mining", "volume reversal")
        secret = "test-secret-value-123"
        with patch.dict(os.environ, {"LLM_API_KEY": secret, "WQ_BRAIN_PASSWORD": "test-password-123"}):
            setup_logging(log_path, "WARNING")
            try:
                fake = FakeTransport({"choices": [{"message": {"content": '{"ok":true}'}}]})
                model = LLM(ModelConfig("openai", "test-model", secret, "https://example.test/v1",
                                        "chat_completions"), fake)
                self.assertTrue(model.probe())
                engine = Engine(config, store, Generator(), Judge(), Brain())
                for _ in range(5):
                    engine.step(run_id)
                logging.getLogger("research_workflow").info("credential echo %s", secret)
                for handler in logging.getLogger("research_workflow").handlers:
                    handler.flush()
                contents = log_path.read_text(encoding="utf-8")
                self.assertIn("模型输入 stage=connectivity_check", contents)
                self.assertIn("模型输出 stage=connectivity_check", contents)
                self.assertIn("探索方向", contents)
                self.assertIn("sharpe", contents)
                self.assertIn("FITNESS", contents)
                self.assertIn("模型决策", contents)
                self.assertIn("test another reversal horizon", contents)
                self.assertNotIn(secret, contents)
                self.assertIn("[REDACTED]", contents)
            finally:
                logger = logging.getLogger("research_workflow")
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                    handler.close()


if __name__ == "__main__":
    unittest.main()
