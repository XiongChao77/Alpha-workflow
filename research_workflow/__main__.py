"""CLI: one-command start, with optional diagnostics and advanced run control."""

import argparse
import json
import logging
import time

from .brain import Brain
from .config import Config, DEFAULT_CONFIG_FILE, load_config_file, model_config
from .engine import Engine
from .heartbeat import start_heartbeat
from .llm import LLM
from .library import export_candidates
from .logging_setup import conversation_path, setup_logging
from .quantgpt import QuantGPT
from .research import AlphaAgent, ResearchAgent, ResearchWorkflow
from .store import Store
from .transport import RemoteError


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
    chat = sub.add_parser("chat", help="Discuss and queue a Topic while research runs")
    chat.add_argument("discussion_id", nargs="?", help="Resume a saved discussion")
    topic_cmd = sub.add_parser("topic", help="Manage researcher-approved research topics")
    topic_sub = topic_cmd.add_subparsers(dest="topic_command", required=True)
    topic_add = topic_sub.add_parser("add", help="Queue a human topic without interrupting active research")
    topic_add.add_argument("title")
    topic_add.add_argument("--objective", default=None)
    topic_add.add_argument("--rationale", default="研究员提交")
    topic_add.add_argument("--priority", type=int, default=100)
    topic_sub.add_parser("list", help="Show queued and historical topics")
    topic_sub.add_parser("suggest", help="Ask Research Agent for a topic without enqueuing it")
    topic_chat = topic_sub.add_parser("chat", help="Discuss and queue a Topic while research runs")
    topic_chat.add_argument("discussion_id", nargs="?", help="Resume a saved discussion")
    topic_brainstorm = topic_sub.add_parser("brainstorm", help="Alias for topic chat")
    topic_brainstorm.add_argument("discussion_id", nargs="?", help="Resume a saved discussion")
    args = parser.parse_args()
    command = args.command or "start"
    load_config_file(args.config or DEFAULT_CONFIG_FILE, required=args.config is not None)
    config = Config()
    session_log_file = setup_logging(config.log_file, config.log_level)
    log = logging.getLogger(__package__)
    log.info("本次运行日志 path=%s", session_log_file)
    log.info("模型对话记录 path=%s", conversation_path())
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
               "research_prompt_files": {"topic": str(model.topic_prompt_file),
                                         "directions": str(model.directions_prompt_file),
                                         "feedback": str(model.feedback_prompt_file),
                                         "variants": str(model.variants_prompt_file),
                                         "repair": str(model.repair_prompt_file)},
               "brainstorm_prompt_file": str(model.brainstorm_prompt_file),
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
    if command in {"topic", "chat"}:
        topic_action = args.topic_command if command == "topic" else "chat"
        if topic_action == "list":
            print(store.all("SELECT id,title,objective,priority,source,status,summary FROM topics "
                            "ORDER BY status,priority DESC,created_at"))
            return
        if topic_action == "add":
            queue = ResearchWorkflow(config, store, None, None, None)
            print(queue.add_topic(args.title, args.objective, args.rationale, args.priority))
            return
        selected_model = model_config()
        if not selected_model.api_key:
            raise SystemExit("Fill LLM_API_KEY in config.env first")
        if topic_action in {"brainstorm", "chat"}:
            model = LLM(selected_model)
            queue = ResearchWorkflow(config, store, None, None, None)
            discussion_id = args.discussion_id or queue.start_discussion()
            session, history = queue.discussion(discussion_id)
            if session["status"] != "open":
                print("讨论已提交，Topic:", session["topic_id"])
                return
            proposal = json.loads(session["proposal"]) if session["proposal"] else None
            print(f"讨论 ID: {discussion_id}。输入研究想法；/queue 入库，/quit 退出。可用此 ID 恢复讨论。")
            while True:
                try:
                    line = input("研究员> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return
                if line == "/quit":
                    return
                if line == "/queue":
                    if proposal is None:
                        print("尚无 Topic 草案")
                        continue
                    print("已入队:", queue.submit_discussion(discussion_id))
                    return
                if not line:
                    continue
                queue.append_discussion(discussion_id, "researcher", line)
                _, history = queue.discussion(discussion_id)
                try:
                    answer = model.brainstorm_topic({"history": history[-20:], "target": config.settings(),
                                                     "recent_topics": store.all("SELECT title,objective FROM topics ORDER BY created_at DESC LIMIT 10")})
                except (ValueError, RemoteError) as exc:
                    log.warning("Topic 讨论模型调用失败 discussion_id=%s reason=%s", discussion_id, exc)
                    print("模型暂未回复；讨论已保存，可稍后使用讨论 ID 恢复。", exc)
                    return
                proposal = answer["topic"]
                queue.append_discussion(discussion_id, "agent", answer["response"], proposal)
                print("Agent:", answer["response"])
                print("当前草案:", proposal)
            return
        suggestion = ResearchAgent(LLM(selected_model)).topic({
            "default_objective": config.research_objective(),
            "default_direction": config.research_direction(),
            "recent_topics": store.all("SELECT title,objective,summary FROM topics ORDER BY created_at DESC LIMIT 10"),
            "recent_feedback": store.all("SELECT statistics,decision FROM feedback ORDER BY created_at DESC LIMIT 8")})
        print(suggestion)
        return
    if command == "export":
        count = export_candidates(store, config.library_dir)
        print(f"Exported {count} candidate records to {config.library_dir / 'candidates'}")
        return
    if command == "create":
        print(store.create_run(args.objective, args.direction))
        return
    if command == "status":
        print("Topics:", store.all("SELECT id,title,priority,source,status,summary FROM topics ORDER BY created_at DESC LIMIT 10"))
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
    model = LLM(selected_model)
    engine = Engine(config, store, model, model, Brain(config), QuantGPT(config) if config.quantgpt_url else None)
    if command == "start":
        workflow = ResearchWorkflow(config, store, ResearchAgent(model), AlphaAgent(model), engine)
        log.info("自主研究工作流启动 pending_topics=%s", store.one("SELECT COUNT(*) AS n FROM topics WHERE status='pending'")["n"])
        heartbeat_stop, heartbeat_thread = start_heartbeat(store, 10)
        try:
            while True:
                log.info("工作流：%s", workflow.step())
                time.sleep(config.idle_seconds)
        except KeyboardInterrupt:
            log.info("工作流已停止；状态保存在 SQLite，下一次启动可恢复")
        except Exception:
            log.exception("自主研究工作流发生未处理异常")
            raise
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=2)
        return
    run_id = args.run_id
    log.info("旧版研究任务启动 run_id=%s", run_id)
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
