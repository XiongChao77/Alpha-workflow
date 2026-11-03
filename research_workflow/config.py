"""Environment-backed configuration. Secrets are never written to the database."""

from dataclasses import dataclass, field
from pathlib import Path
import os
import re


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_FILE = Path(__file__).resolve().parent / "config.env"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def load_config_file(path: Path, *, required: bool = False) -> None:
    """Load simple KEY=VALUE lines; process environment takes precedence."""
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"Invalid config line {line_number}: expected KEY=VALUE")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    api_key: str
    base_url: str
    api_format: str = "responses"
    json_mode: str = "auto"
    disable_response_storage: bool = False
    generate_prompt_file: Path = PROMPT_DIR / "generate.txt"
    decide_prompt_file: Path = PROMPT_DIR / "decide.txt"
    probe_prompt_file: Path = PROMPT_DIR / "probe.txt"
    parameter_prompt_file: Path = PROMPT_DIR / "parameter_decide.txt"
    submission_prompt_file: Path = PROMPT_DIR / "select_submission.txt"


def model_config() -> ModelConfig:
    """One provider and key are shared by generation and research decisions."""
    provider = os.getenv("LLM_PROVIDER", "").lower()
    if provider not in {"openai", "deepseek"}:
        raise ValueError("LLM_PROVIDER must be openai or deepseek")
    fallback_model = "deepseek-flash" if provider == "deepseek" else "gpt-4o"
    key = os.getenv("LLM_API_KEY", "")
    base = os.getenv("LLM_BASE_URL") or (
        "https://api.deepseek.com" if provider == "deepseek" else "https://api.openai.com"
    )
    api_format = os.getenv("LLM_API_FORMAT") or ("chat_completions" if provider == "deepseek" else "responses")
    api_format = api_format.lower()
    if api_format not in {"responses", "chat_completions"}:
        raise ValueError("LLM_API_FORMAT must be responses or chat_completions")
    json_mode = os.getenv("LLM_JSON_MODE", "auto").lower()
    if json_mode not in {"auto", "json_schema", "json_object", "none"}:
        raise ValueError("LLM_JSON_MODE must be auto, json_schema, json_object or none")
    disable_storage = os.getenv("LLM_DISABLE_RESPONSE_STORAGE", "false").lower() in {"1", "true", "yes"}
    return ModelConfig(provider, os.getenv("LLM_MODEL") or fallback_model, key, base.rstrip("/"),
                       api_format, json_mode, disable_storage,
                       Path(os.getenv("LLM_GENERATE_PROMPT_FILE") or PROMPT_DIR / "generate.txt"),
                       Path(os.getenv("LLM_DECIDE_PROMPT_FILE") or PROMPT_DIR / "decide.txt"),
                       Path(os.getenv("LLM_PROBE_PROMPT_FILE") or PROMPT_DIR / "probe.txt"),
                       Path(os.getenv("LLM_PARAMETER_PROMPT_FILE") or PROMPT_DIR / "parameter_decide.txt"),
                       Path(os.getenv("LLM_SUBMISSION_PROMPT_FILE") or PROMPT_DIR / "select_submission.txt"))


@dataclass(frozen=True)
class Config:
    log_file: Path = field(default_factory=lambda: Path(os.getenv("ALPHA_LOG_FILE") or ROOT / "research_workflow" / "logs" / "workflow.log"))
    log_level: str = field(default_factory=lambda: os.getenv("ALPHA_LOG_LEVEL", "INFO"))
    db_path: Path = field(default_factory=lambda: Path(os.getenv("ALPHA_WORKFLOW_DB", ROOT / "research_workflow.sqlite3")))
    library_dir: Path = field(default_factory=lambda: Path(os.getenv("ALPHA_LIBRARY_DIR", ROOT / "alpha_libraries")))
    account: str = field(default_factory=lambda: os.getenv("WQ_ACCOUNT", "primary"))
    email: str = field(default_factory=lambda: os.getenv("WQ_BRAIN_EMAIL", ""))
    password: str = field(default_factory=lambda: os.getenv("WQ_BRAIN_PASSWORD", ""))
    brain_url: str = field(default_factory=lambda: os.getenv("WQ_BRAIN_URL", "https://api.worldquantbrain.com").rstrip("/"))
    region: str = field(default_factory=lambda: os.getenv("WQ_REGION", "USA"))
    universe: str = field(default_factory=lambda: os.getenv("WQ_UNIVERSE", "TOP3000"))
    delay: int = field(default_factory=lambda: int(os.getenv("WQ_DELAY", "1")))
    neutralization: str = field(default_factory=lambda: os.getenv("WQ_NEUTRALIZATION", "SUBINDUSTRY"))
    auto_submit: bool = field(default_factory=lambda: os.getenv("WQ_AUTO_SUBMIT", "1") == "1")
    batch_size: int = field(default_factory=lambda: int(os.getenv("ALPHA_BATCH_SIZE", "3")))
    max_simulations_day: int = field(default_factory=lambda: int(os.getenv("ALPHA_MAX_SIMULATIONS_DAY", "20")))
    max_concurrent_simulations: int = field(default_factory=lambda: int(os.getenv("ALPHA_MAX_CONCURRENT_SIMULATIONS", "3")))
    max_rounds: int = field(default_factory=lambda: int(os.getenv("ALPHA_MAX_ROUNDS", "0")))
    max_pending: int = field(default_factory=lambda: int(os.getenv("ALPHA_MAX_PENDING", "30")))
    period_grid: tuple[int, ...] = field(default_factory=lambda: tuple(
        int(value.strip()) for value in os.getenv("ALPHA_PERIOD_GRID", "30,60,90").split(",")))
    family_max_stagnant: int = field(default_factory=lambda: int(os.getenv("ALPHA_FAMILY_MAX_STAGNANT", "5")))
    submission_review_limit: int = field(default_factory=lambda: int(os.getenv("ALPHA_SUBMISSION_REVIEW_LIMIT", "30")))
    idle_seconds: int = field(default_factory=lambda: int(os.getenv("ALPHA_IDLE_SECONDS", "30")))
    quantgpt_url: str = field(default_factory=lambda: os.getenv("QUANTGPT_URL", "").rstrip("/"))
    quantgpt_token: str = field(default_factory=lambda: os.getenv("QUANTGPT_TOKEN", ""))
    quantgpt_universe: str = field(default_factory=lambda: os.getenv("QUANTGPT_UNIVERSE", "hs300"))
    quantgpt_min_score: float = field(default_factory=lambda: float(os.getenv("QUANTGPT_MIN_SCORE", "0")))

    def research_objective(self) -> str:
        return os.getenv("ALPHA_OBJECTIVE") or f"挖掘低相关、可提交到WorldQuant的 {self.region} {self.universe} D{self.delay} 因子"

    def research_direction(self) -> str:
        return os.getenv("ALPHA_DIRECTION") or "量价维度的因子"

    def settings(self) -> dict:
        return {"instrumentType": "EQUITY", "region": self.region, "universe": self.universe,
                "delay": self.delay, "decay": 0, "neutralization": self.neutralization,
                "truncation": 0.08, "pasteurization": "ON", "unitHandling": "VERIFY",
                "nanHandling": "OFF", "language": "FASTEXPR", "visualization": False}
