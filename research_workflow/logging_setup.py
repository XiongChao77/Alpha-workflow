"""Console and rotating file logs for the research worker."""

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path


class SecretFormatter(logging.Formatter):
    """Redact configured credentials even when an upstream error echoes one."""

    def format(self, record):
        message = super().format(record)
        for key in ("LLM_API_KEY", "WQ_BRAIN_PASSWORD", "QUANTGPT_TOKEN"):
            secret = os.getenv(key, "")
            if secret and len(secret) >= 4:
                message = message.replace(secret, "[REDACTED]")
        return message


def setup_logging(log_file: Path, level: str = "INFO") -> None:
    logger = logging.getLogger("research_workflow")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console = logging.StreamHandler()
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(SecretFormatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logger.addHandler(console)

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(SecretFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(file_handler)
