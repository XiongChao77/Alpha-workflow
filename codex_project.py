"""Launch Codex CLI with this project's existing model configuration."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "research_workflow" / "config.env"


def read_config(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"Missing project config: {path}")
    settings: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if not match:
            continue
        value = match.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        settings[match.group(1)] = value
    return settings


def main(argv: list[str]) -> int:
    try:
        settings = read_config(CONFIG)
        api_key = settings.get("LLM_API_KEY", "")
        base_url = settings.get("LLM_BASE_URL", "").rstrip("/")
        model = os.environ.get("CODEX_PROJECT_MODEL") or settings.get("LLM_MODEL", "")
        if not api_key or not model or not base_url.startswith("https://"):
            raise ValueError(
                "LLM_API_KEY, LLM_MODEL and an HTTPS LLM_BASE_URL are required "
                "in research_workflow/config.env."
            )
        codex = shutil.which("codex")
        if codex is None:
            raise ValueError("Codex CLI is not on PATH. Install it with npm install -g @openai/codex.")

        env = os.environ.copy()
        env["PROJECT_CODEX_API_KEY"] = api_key
        command = [
            codex,
            "-c", "model_provider=project_api",
            "-c", "model_providers.project_api.name=Project API",
            "-c", f"model_providers.project_api.base_url={base_url}",
            "-c", "model_providers.project_api.env_key=PROJECT_CODEX_API_KEY",
            "-c", "model_providers.project_api.wire_api=responses",
            "-c", "model_providers.project_api.requires_openai_auth=false",
            "-c", "model_providers.project_api.supports_websockets=false",
            "-c", "shell_environment_policy.ignore_default_excludes=false",
            "-m", model,
            *argv,
        ]
        return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode
    except (OSError, ValueError) as exc:
        print(f"codex_project: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
