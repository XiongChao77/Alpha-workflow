"""Console and rotating file logs for the research worker."""

import logging
from logging.handlers import RotatingFileHandler
import json
import os
from datetime import datetime
from pathlib import Path
from threading import Lock


_conversation_path: Path | None = None
_conversation_lock = Lock()


def conversation_path() -> Path | None:
    return _conversation_path


def save_model_exchange(record: dict) -> None:
    """Append one complete HTTP exchange without authorization headers."""
    if _conversation_path is None:
        return
    line = json.dumps(record, ensure_ascii=False, default=str)
    for key in ("LLM_API_KEY", "WQ_BRAIN_PASSWORD", "QUANTGPT_TOKEN"):
        secret = os.getenv(key, "")
        if secret and len(secret) >= 4:
            line = line.replace(secret, "[REDACTED]")
    with _conversation_lock:
        with _conversation_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


class SecretFormatter(logging.Formatter):
    """Redact configured credentials even when an upstream error echoes one."""

    def format(self, record):
        message = super().format(record)
        for key in ("LLM_API_KEY", "WQ_BRAIN_PASSWORD", "QUANTGPT_TOKEN"):
            secret = os.getenv(key, "")
            if secret and len(secret) >= 4:
                message = message.replace(secret, "[REDACTED]")
        return message


def setup_logging(log_file: Path, level: str = "INFO") -> Path:
    global _conversation_path
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
    started = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    session_path = path.with_name(f"{path.stem}_{started}_{os.getpid()}{path.suffix or '.log'}")
    _conversation_path = session_path.with_name(f"model_conversations_{started}_{os.getpid()}.jsonl")
    file_handler = RotatingFileHandler(session_path, maxBytes=10 * 1024 * 1024,
                                       backupCount=5, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(SecretFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(file_handler)
    return session_path
