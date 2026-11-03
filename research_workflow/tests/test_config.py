import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_workflow.config import Config, load_config_file, model_config


class ConfigFileTests(unittest.TestCase):
    def test_file_loads_one_model_and_brain_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.env"
            path.write_text("""# example
LLM_PROVIDER=openai
LLM_MODEL=gpt-6.1-sol
LLM_API_KEY=fake-model-key
WQ_BRAIN_EMAIL=sample@example.com
WQ_BRAIN_PASSWORD='fake password # kept'
WQ_AUTO_SUBMIT=0
""", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                load_config_file(path, required=True)
                self.assertEqual(model_config().provider, "openai")
                self.assertEqual(model_config().model, "gpt-6.1-sol")
                self.assertEqual(model_config().api_key, "fake-model-key")
                self.assertEqual(Config().password, "fake password # kept")
                self.assertFalse(Config().auto_submit)

    def test_process_environment_overrides_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.env"
            path.write_text("WQ_REGION=USA\n", encoding="utf-8")
            with patch.dict(os.environ, {"WQ_REGION": "EUR"}, clear=True):
                load_config_file(path, required=True)
                self.assertEqual(Config().region, "EUR")

    def test_deepseek_is_alternative_single_provider(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "deepseek", "LLM_API_KEY": "fake-key"}, clear=True):
            self.assertEqual(model_config().provider, "deepseek")
            self.assertEqual(model_config().model, "deepseek-flash")

    def test_third_party_gateway_configuration(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-5.5",
                                     "LLM_API_KEY": "fake-key", "LLM_BASE_URL": "https://example.test/v1",
                                     "LLM_API_FORMAT": "chat_completions",
                                     "LLM_DISABLE_RESPONSE_STORAGE": "true"}, clear=True):
            config = model_config()
            self.assertEqual(config.base_url, "https://example.test/v1")
            self.assertEqual(config.api_format, "chat_completions")
            self.assertTrue(config.disable_response_storage)

    def test_missing_explicit_file_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_config_file(Path(directory) / "missing.env", required=True)


if __name__ == "__main__":
    unittest.main()
