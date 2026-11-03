"""CLI: one-command start, with optional diagnostics and advanced run control."""

import argparse
import logging
import time

from .brain import Brain
from .config import Config, DEFAULT_CONFIG_FILE, load_config_file, model_config
from .engine import Engine
from .heartbeat import start_heartbeat
from .llm import LLM
from .library import export_candidates
from .logging_setup import setup_logging
from .quantgpt import QuantGPT
from .store import Store


def main():
    parser = argparse.ArgumentParser(description="Standalone WorldQuant alpha research workflow")
    parser.add_argument("--config", type=str, default=None,
                        help="KEY=VALUE config file (default: research_workflow/config.env if present)")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("start", help="Automatically create or resume research and keep running")
    sub.add_parser("check", help="Validate configuration without printing secrets or connecting")
    sub.add_parser("probe", help="Make one small model API request; does not call BRAIN")
    sub.add_parser("export", help="Write structured JSON for all existing candidates; no API calls")
    create = sub.add_parser("create", help="Create a persistent research run")
    create.add_argument("objective")
    create.add_argument("--direction", default="price-volume signals")
    once = sub.add_parser("once", help="Advance one durable workflow step")
    once.add_argument("run_id")
    run = sub.add_parser("run", help="Continue mining until interrupted")
    run.add_argument("run_id")
    status = sub.add_parser("status", help="Show run, candidates and today's submission")
    status.add_argument("run_id", nargs="?")
    args = parser.parse_args()
    command = args.command or "start"
    load_config_file(args.config or DEFAULT_CONFIG_FILE, required=args.config is not None)
    config = Config()
    session_log_file = setup_logging(config.log_file, config.log_level)
    log = logging.getLogger(__package__)
    log.info("本次运行日志 path=%s", session_log_file)
    if command == "check":
        model = model_config()
        print({"model": (model.provider, model.model, bool(model.api_key)),
               "model_base_url": model.base_url, "model_api_format": model.api_format,
               "model_json_mode": model.json_mode,
               "prompt_files": {"generate": str(model.generate_prompt_file),
                                "decide": str(model.decide_prompt_file),
                                "probe": str(model.probe_prompt_file),
                                "parameter": str(model.parameter_prompt_file),
                                "submission": str(model.submission_prompt_file)},
               "disable_response_storage": model.disable_response_storage,
               "brain_credentials_present": bool(config.email and config.password),
               "brain_target": (config.region, config.universe, config.delay),
               "max_concurrent_simulations": config.max_concurrent_simulations,
               "auto_submit": config.auto_submit,
               "quantgpt_enabled": bool(config.quantgpt_url),
               "database": str(config.db_path), "library": str(config.library_dir)})
        if not (model.api_key and config.email and config.password):
            raise SystemExit("Missing LLM_API_KEY or BRAIN credentials in config")
        return
    if command == "probe":
        from .transport import RemoteError

        try:
            success = LLM(model_config()).probe()
        except RemoteError as exc:
            raise SystemExit(f"Model probe failed: HTTP {exc.status} (check gateway format, model ID and key)") from exc
        print("Model probe passed" if success else "Model probe returned unexpected JSON")
        if not success:
            raise SystemExit(1)
        return
    store = Store(config.db_path)
    if command == "export":
        count = export_candidates(store, config.library_dir)
        print(f"Exported {count} candidate records to {config.library_dir / 'candidates'}")
        return
    if command == "create":
        print(store.create_run(args.objective, args.direction))
        return
    if command == "status":
        run_id = args.run_id or ((store.latest_run() or {}).get("id"))
        if not run_id:
            print("No research run yet. Start with: python -m research_workflow")
            return
        print(store.one("SELECT * FROM runs WHERE id=?", (run_id,)))
        print(store.all("SELECT id,round,expression,status,alpha_id FROM candidates WHERE run_id=? ORDER BY created_at", (run_id,)))
        print(store.submission_today(config.account))
        return
    selected_model = model_config()
    if not selected_model.api_key or not config.email or not config.password:
        raise SystemExit("Fill LLM_API_KEY, WQ_BRAIN_EMAIL and WQ_BRAIN_PASSWORD in config.env first")
    if command == "start":
        run_id = store.get_or_create_run(config.research_objective(), config.research_direction())
    else:
        run_id = args.run_id
    log.info("研究任务启动 run_id=%s objective=%s direction=%s", run_id,
             config.research_objective(), config.research_direction())
    model = LLM(selected_model)
    engine = Engine(config, store, model, model, Brain(config), QuantGPT(config) if config.quantgpt_url else None)
    if command == "once":
        log.info("工作流：%s", engine.step(run_id))
        return
    heartbeat_stop, heartbeat_thread = start_heartbeat(store, 10)
    try:
        while True:
            result = engine.step(run_id)
            log.info("工作流：%s", result)
            if result.startswith("run ") or result == "round budget reached":
                break
            time.sleep(config.idle_seconds)
    except KeyboardInterrupt:
        log.info("工作流已停止；状态保存在 SQLite，下一次启动可恢复")
    except Exception:
        log.exception("工作流发生未处理异常 run_id=%s", run_id)
        raise
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2)


if __name__ == "__main__":
    main()
