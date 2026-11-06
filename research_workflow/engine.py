"""A durable step-based research loop. One process should drive a run."""

from datetime import datetime, timezone
import json
import logging
import re
import sqlite3
import uuid

from .brain import is_checks_passed
from .library import archive, eligible_records, export_candidates
from .parameter_search import best_candidate, period_values, render_expression, sharpe
from .store import group_families, now, today
from .transport import RemoteError


TERMINAL = {"passed", "rejected", "failed", "unknown", "local_rejected"}
log = logging.getLogger(__name__)


def result_summary(data):
    """Compact BRAIN metrics/checks for the console; full result goes to the file."""
    is_data = data.get("is") or {}
    return {"sharpe": is_data.get("sharpe"), "fitness": is_data.get("fitness"),
            "returns": is_data.get("returns"), "turnover": is_data.get("turnover"),
            "checks": [{"name": c.get("name"), "result": c.get("result"),
                        "value": c.get("value"), "limit": c.get("limit")}
                       for c in (is_data.get("checks") or [])]}


def retain_local_result(candidate, brain_result):
    """Keep the optional QuantGPT result when BRAIN replaces the latest result."""
    previous = json.loads(candidate["result"] or "{}")
    if isinstance(previous, dict) and "quantgpt" in previous:
        return {**brain_result, "quantgpt": previous["quantgpt"]}
    return brain_result


def check_quality(candidate):
    """Prefer fewer failed checks, then a smaller total normalized threshold violation."""
    result = json.loads(candidate["result"] or "{}")
    checks = (result.get("is") or {}).get("checks") or []
    if not checks:
        return None
    failed = [check for check in checks if str(check.get("result", "")).upper() == "FAIL"]
    violation = 0.0
    for check in failed:
        value, limit = check.get("value"), check.get("limit")
        name = str(check.get("name") or "").upper()
        if isinstance(value, (int, float)) and isinstance(limit, (int, float)) and limit:
            if name.startswith("LOW_"):
                violation += max(0.0, (limit - value) / abs(limit))
            elif name.startswith("HIGH_"):
                violation += max(0.0, (value - limit) / abs(limit))
            else:
                violation += 1.0
        else:
            violation += 1.0
    return (-len(failed), -round(violation, 6))


def stagnant_count(family):
    """Count trailing evaluated descendants that did not beat the family best."""
    best = None
    stale = 0
    evaluated = 0
    for candidate in family:
        if candidate["status"] not in TERMINAL:
            continue
        quality = check_quality(candidate)
        if candidate.get("origin_action") in {"PERIOD", "PARAM1", "PARAM2"}:
            if quality is not None and (best is None or quality > best):
                best = quality
            continue
        if evaluated == 0:
            if quality is not None and (best is None or quality > best):
                best = quality
        elif quality is not None and (best is None or quality > best):
            best = quality
            stale = 0
        else:
            stale += 1
        evaluated += 1
    return stale


def valid_expression(expression: str) -> bool:
    # BRAIN validates its own expression size and syntax. Keep only a generous
    # local guard so multi-statement FASTEXPR is not discarded before POST.
    if not 3 <= len(expression) <= 4096 or expression.count("(") != expression.count(")"):
        return False
    depth = 0
    for char in expression:
        depth += (char == "(") - (char == ")")
        if depth < 0:
            return False
    return bool(re.fullmatch(r"[A-Za-z0-9_.,;+*/()<>!=&|\s-]+", expression)) and "__" not in expression


class Engine:
    def __init__(self, config, store, generator, judge, brain, quantgpt=None):
        if config.max_concurrent_simulations < 1:
            raise ValueError("ALPHA_MAX_CONCURRENT_SIMULATIONS must be positive")
        self.config = config
        self.store = store
        self.generator = generator
        self.judge = judge
        self.brain = brain
        self.quantgpt = quantgpt

    def _run(self, run_id):
        run = self.store.one("SELECT * FROM runs WHERE id=?", (run_id,))
        if not run:
            raise ValueError("Unknown run ID")
        return run

    def _today_simulations(self):
        from .store import CST
        midnight = datetime.now(CST).replace(hour=0, minute=0, second=0, microsecond=0)
        start = midnight.astimezone(timezone.utc).isoformat()
        row = self.store.one("SELECT COUNT(*) AS n FROM candidates WHERE simulated_at>=?", (start,))
        return row["n"]

    def _inflight_simulations(self):
        """Reserve slots for POSTs whose outcome could still be active."""
        row = self.store.one("SELECT COUNT(*) AS n FROM candidates WHERE status IN "
                             "('starting','simulating','unknown')")
        return row["n"]

    def _advance_ready_candidates(self, candidates):
        ready = [candidate for candidate in candidates
                 if not candidate["not_before"] or candidate["not_before"] <= now()]
        results = []
        max_operations = self.config.max_concurrent_simulations
        handled = set()

        # Local prefilter does not consume a BRAIN simulation slot.
        for candidate in ready:
            if len(results) >= max_operations:
                break
            if candidate["status"] == "local_wait" or (
                candidate["status"] == "queued" and self.quantgpt and not candidate["quantgpt_task_id"]
            ):
                if self.store.cooling("quantgpt"):
                    continue
                results.append(self._advance_candidate(candidate))
                handled.add(candidate["id"])
                if self.store.cooling("quantgpt"):
                    break

        # Poll each due simulation once; a completed one releases its slot.
        for candidate in ready:
            if len(results) >= max_operations or self.store.cooling("brain"):
                break
            if candidate["id"] not in handled and candidate["status"] in {"simulating", "checking"}:
                results.append(self._advance_candidate(candidate))
                handled.add(candidate["id"])

        # Fill remaining BRAIN slots with queued simulations in the same step.
        for candidate in ready:
            if candidate["id"] in handled or candidate["status"] != "queued":
                continue
            if self.quantgpt and not candidate["quantgpt_task_id"]:
                continue
            if self.store.cooling("brain") or self._today_simulations() >= self.config.max_simulations_day:
                break
            in_flight = self._inflight_simulations()
            if in_flight >= self.config.max_concurrent_simulations:
                break
            result = self._advance_candidate(candidate)
            results.append(result)
            handled.add(candidate["id"])
            log.info("BRAIN 并发槽位 in_flight=%s limit=%s", self._inflight_simulations(),
                     self.config.max_concurrent_simulations)
            if self.store.cooling("brain"):
                break
        return results

    def _summary(self, run_id, family_id=None):
        if family_id is None:
            rows = self.store.all("SELECT * FROM candidates WHERE run_id=? ORDER BY created_at DESC LIMIT 30", (run_id,))
        else:
            all_rows = self.store.all("SELECT * FROM candidates WHERE run_id=? ORDER BY created_at,id", (run_id,))
            rows = group_families(all_rows).get(family_id, [])
        submitted = {row["alpha_id"]: row["status"] for row in self.store.all("SELECT alpha_id,status FROM submissions")}
        summary = []
        for c in rows:
            result = json.loads(c["result"] or "{}")
            is_data = result.get("is") or {}
            summary.append({"id": c["id"], "round": c["round"], "hypothesis": c["hypothesis"],
                            "expression": c["expression"], "settings": json.loads(c["settings"]),
                            "parent_candidate_id": c.get("parent_candidate_id"),
                            "origin_action": c.get("origin_action") or "GENERATE",
                            "parameter_values": json.loads(c["parameter_values"] or "{}"),
                            "family_id": family_id,
                            "status": c["status"], "alpha_id": c["alpha_id"],
                            "submission_status": submitted.get(c["alpha_id"]),
                            "metrics": {key: is_data.get(key) for key in ("sharpe", "fitness", "returns", "turnover")},
                            "checks": [{"name": x.get("name"), "result": x.get("result"),
                                        "value": x.get("value"), "limit": x.get("limit")}
                                       for x in (is_data.get("checks") or [])],
                            "error": str(result.get("error") or
                                         (result.get("message") if c["status"] == "failed" else "") or "")[:300]})
        return summary

    def _next_family(self, run_id):
        candidates = self.store.all("SELECT * FROM candidates WHERE run_id=? ORDER BY created_at,id", (run_id,))
        if not candidates:
            return None
        families = group_families(candidates)
        searches = {row["family_id"]: row for row in self.store.all(
            "SELECT family_id,stage FROM parameter_searches WHERE run_id=?", (run_id,))}
        decided = {}
        stopped = set()
        for row in self.store.all("SELECT context,decision FROM decisions WHERE run_id=? ORDER BY created_at,id", (run_id,)):
            context = json.loads(row["context"])
            decision = json.loads(row["decision"])
            family_id = context.get("family_id")
            if decision.get("action") == "STOP_FAMILY" and family_id:
                stopped.add(family_id)
            if family_id and context.get("latest_candidate_id"):
                decided[family_id] = context["latest_candidate_id"]
            elif decision.get("target_id") and decision.get("action") != "OPTIMIZE":
                # Avoid replaying decisions saved before family-scoped context existed.
                target_id = decision["target_id"]
                for root_id, members in families.items():
                    if any(member["id"] == target_id for member in members):
                        decided[root_id] = target_id
                        break
        for family_id, members in families.items():
            if family_id in stopped:
                continue
            search = searches.get(family_id)
            if search and search["stage"] != "READY":
                continue
            latest = members[-1]
            if latest["status"] not in TERMINAL or latest["status"] == "unknown":
                continue
            stale = stagnant_count(members)
            if self.config.family_max_stagnant > 0 and stale >= self.config.family_max_stagnant:
                context = {"family_id": family_id, "latest_candidate_id": latest["id"],
                           "stagnation_count": stale}
                decision = {"action": "STOP_FAMILY", "reason": "consecutive candidates did not improve BRAIN checks"}
                with self.store.tx() as db:
                    db.execute("INSERT INTO decisions VALUES (?,?,?,?,?)",
                               (uuid.uuid4().hex[:12], run_id, json.dumps(context), json.dumps(decision), now()))
                log.info("假设族停止 family_id=%s stagnant=%s limit=%s latest=%s", family_id, stale,
                         self.config.family_max_stagnant, latest["id"])
                continue
            if decided.get(family_id) != latest["id"]:
                return {"id": family_id, "latest_candidate_id": latest["id"],
                        "stagnation_count": stale, "members": members}
        return None

    def _throttle(self, service: str, exc: RemoteError):
        self.store.set_cooldown(service, exc.retry_after or 60, str(exc))
        log.warning("服务退让 service=%s http=%s wait_seconds=%s detail=%s", service, exc.status,
                    exc.retry_after or 60, exc)
        return f"{service} cooldown: {exc}"

    def _archive(self, candidate, stage, decision=None):
        alpha = candidate["alpha_id"]
        platform = self.brain.get_alpha(alpha)
        archive(self.config.library_dir, candidate, platform, stage, decision)
        return platform

    def _reconcile_submission(self):
        item = self.store.one("SELECT * FROM submissions WHERE status NOT IN ('active','failed','rejected','dry_run') ORDER BY created_at LIMIT 1")
        if not item:
            return None
        if self.store.cooling("brain"):
            return "BRAIN cooling before submission reconciliation"
        try:
            platform = self.brain.get_alpha(item["alpha_id"])
        except RemoteError as exc:
            return self._throttle("brain", exc)
        status = str(platform.get("status") or "").upper()
        checks = (platform.get("is") or {}).get("checks") or []
        if status == "ACTIVE":
            new_status = "active"
        elif any(c.get("name") == "SELF_CORRELATION" and c.get("result") == "FAIL" for c in checks):
            new_status = "rejected"
        elif status == "UNSUBMITTED" and item["status"] in {"reserved", "uncertain"}:
            new_status = "uncertain"  # A POST may have reached BRAIN; never retry blindly.
        else:
            new_status = "pending"
        self.store.update_submission(item["account"], item["day"], new_status, status)
        log.info("提交回查 alpha_id=%s platform_status=%s state=%s checks=%s", item["alpha_id"],
                 status, new_status, result_summary(platform)["checks"])
        candidate = self.store.one("SELECT * FROM candidates WHERE id=?", (item["candidate_id"],))
        if candidate:
            matching = None
            for row in self.store.all("SELECT decision FROM decisions WHERE run_id=? ORDER BY created_at DESC LIMIT 30", (candidate["run_id"],)):
                value = json.loads(row["decision"])
                if value.get("action") == "SUBMIT" and value.get("target_id") == candidate["id"]:
                    matching = value
                    break
            archive(self.config.library_dir, candidate, platform,
                    {"active": "已提交，平台 ACTIVE", "rejected": "提交后检查未通过"}.get(new_status, "已发起提交，等待平台确认"), matching)
        return f"submission {item['alpha_id']}: {new_status}"

    def _submit_eligible(self):
        if self.store.submission_today(self.config.account) or not self.config.auto_submit:
            return None
        searches = {row["family_id"]: row["stage"] for row in self.store.all(
            "SELECT family_id,stage FROM parameter_searches")}
        eligible = [record for record in eligible_records(self.store)
                    if record["submission_status"] == "not_submitted" and not record["component_already_used"] and
                    searches.get(record["family_id"], "READY") == "READY"]
        eligible = [record for record in eligible if not (
            (experiment := self.store.one("SELECT e.stage,e.status FROM candidates c JOIN experiments e "
                                          "ON e.id=c.experiment_id WHERE c.id=?", (record["candidate_id"],)))
            and experiment["status"] != "completed")]
        eligible = eligible[:max(1, self.config.submission_review_limit)]
        if not eligible or not hasattr(self.judge, "choose_submission"):
            return None
        if self.store.cooling("llm"):
            return None
        context = {"eligible": eligible, "today_submission_used": False}
        try:
            decision = self.judge.choose_submission(context)
        except RemoteError as exc:
            return self._throttle("llm", exc)
        except ValueError as exc:
            self.store.set_cooldown("llm", 60, str(exc))
            log.warning("合格因子选择无效 reason=%s", exc)
            return f"invalid submission selection: {exc}"
        candidate_id = decision["target_id"]
        if candidate_id not in {record["candidate_id"] for record in eligible}:
            self.store.set_cooldown("llm", 60, "submission ID not eligible")
            return "invalid submission selection: target not eligible"
        candidate = self.store.one("SELECT * FROM candidates WHERE id=?", (candidate_id,))
        log.info("今日提交候选 candidate=%s alpha_id=%s reason=%s", candidate_id,
                 candidate["alpha_id"], decision["reason"])
        return self._submit(candidate, decision)

    def step(self, run_id: str) -> str:
        try:
            return self._step(run_id)
        finally:
            self.sync_candidate_records(run_id)

    def sync_candidate_records(self, run_id: str) -> None:
        """Export current state and backfill candidates created before JSON export existed."""
        export_candidates(self.store, self.config.library_dir, run_id)

    def _step(self, run_id: str) -> str:
        run = self._run(run_id)
        if run["status"] != "running":
            return f"run {run['status']}"

        # A worker crash after a non-idempotent POST can leave no Location.
        # Keep that candidate for manual reconciliation, never POST it again.
        with self.store.tx() as db:
            db.execute("UPDATE candidates SET status='unknown', result=? WHERE run_id=? AND status='starting'",
                       (json.dumps({"error": "worker stopped during simulation POST; check BRAIN before retrying"}), run_id))

        reconciliation = self._reconcile_submission()
        if reconciliation and "cooling" not in reconciliation:
            # Reconciliation is independent of mining; keep advancing the latter.
            pass

        submission = self._submit_eligible()
        if submission:
            return submission

        candidates = self.store.all("SELECT * FROM candidates WHERE run_id=? AND status IN ('simulating','checking','local_wait','queued') ORDER BY created_at,id", (run_id,))
        progressed = self._advance_ready_candidates(candidates)
        if progressed:
            return " | ".join(progressed)

        pending_count = len(candidates)
        if not candidates:
            search_result = self._advance_parameter_searches(run_id)
            if search_result:
                return search_result
            family = self._next_family(run_id)
            if family:
                return self._decide(run, family)
        if self._today_simulations() >= self.config.max_simulations_day:
            if pending_count >= self.config.max_pending:
                return "daily simulation budget reached; backlog full"
            # Continue hypothesis generation while BRAIN is unavailable, up to the backlog cap.
            return self._generate(run, min(self.config.batch_size, self.config.max_pending - pending_count))
        if not candidates:
            return self._generate(run)
        return "waiting for next poll or daily simulation budget"

    def _generate(self, run, requested_count=None):
        if self.config.max_rounds and run["round"] >= self.config.max_rounds:
            with self.store.tx() as db:
                db.execute("UPDATE runs SET status='finished' WHERE id=?", (run["id"],))
            return "round budget reached"
        if self.store.cooling("llm"):
            return "LLM cooling"
        context = {"objective": run["objective"], "direction": run["direction"],
                   "requested_count": requested_count or min(self.config.batch_size, self.config.max_pending),
                   "settings": self.config.settings(), "history": self._summary(run["id"])}
        log.info("探索方向 run_id=%s round=%s direction=%s count=%s", run["id"], run["round"] + 1,
                 run["direction"], context["requested_count"])
        try:
            generated = self.generator.generate(context)
        except RemoteError as exc:
            return self._throttle("llm", exc)
        except ValueError as exc:
            self.store.set_cooldown("llm", 60, str(exc))
            log.warning("模型输出格式校验失败（尚未调用 BRAIN）reason=%s", exc)
            return f"model output validation failed before BRAIN: {exc}"
        next_round = run["round"] + 1
        accepted = 0
        for item in generated:
            if "expression_template" in item:
                try:
                    defaults = {param["name"]: param["default"] for param in item["parameters"]}
                    trials = []
                    for period in period_values(self.config.period_grid, item["period"]["default"]):
                        values = {"Period": period, **defaults}
                        expression = render_expression(item["expression_template"], values)
                        if not valid_expression(expression):
                            raise ValueError(f"Invalid rendered expression for Period={period}")
                        trials.append((values, expression))
                    family_id = self.store.create_parameter_family(
                        run["id"], next_round, item["hypothesis"].strip(), item, trials, self.config.settings())
                    accepted += len(trials)
                    log.info("新假设族 family_id=%s period_values=%s hypothesis=%s template=%s extra_parameters=%s",
                             family_id, [values["Period"] for values, _ in trials], item["hypothesis"],
                             item["expression_template"], item["parameters"])
                except (ValueError, sqlite3.IntegrityError) as exc:
                    log.warning("参数化候选未入队 hypothesis=%s reason=%s", item["hypothesis"], exc)
            else:
                expr = item["expression"].strip()
                if valid_expression(expr) and self.store.add_candidate(run["id"], next_round, item["hypothesis"].strip(), expr, self.config.settings()):
                    accepted += 1
                    log.info("新候选 round=%s hypothesis=%s formula=%s", next_round, item["hypothesis"], expr)
                else:
                    log.warning("候选未入队 round=%s formula=%s", next_round, expr)
        with self.store.tx() as db:
            db.execute("UPDATE runs SET round=?, decision_due=? WHERE id=?", (next_round, int(accepted > 0), run["id"]))
        if accepted == 0:
            self.store.set_cooldown("llm", 300, "no novel valid expressions")
        return f"generated {accepted} candidates in round {next_round}"

    def _search_trials(self, search):
        ids = json.loads(search["stage_trials"])
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.store.all(f"SELECT * FROM candidates WHERE id IN ({placeholders})", tuple(ids))
        by_id = {row["id"]: row for row in rows}
        return [by_id[candidate_id] for candidate_id in ids if candidate_id in by_id]

    def _set_search_stage(self, search, stage, selected=None):
        with self.store.tx() as db:
            db.execute("UPDATE parameter_searches SET stage=?,selected_params=?,selected_period=?,updated_at=? "
                       "WHERE family_id=?", (stage, json.dumps(selected or json.loads(search["selected_params"]), sort_keys=True),
                                             (selected or json.loads(search["selected_params"])).get("Period") or
                                             search["selected_period"], now(), search["family_id"]))

    def _queue_dimension(self, search, index, baseline):
        parameters = json.loads(search["parameters"])
        parameter = parameters[index]
        selected = json.loads(baseline["parameter_values"] or "{}")
        trials = []
        for value in parameter["values"]:
            values = {**selected, parameter["name"]: value}
            expression = render_expression(search["expression_template"], values)
            if not valid_expression(expression):
                raise ValueError(f"Invalid rendered {parameter['name']}={value} expression")
            trials.append((values, expression))
        run = self._run(search["run_id"])
        root = self.store.one("SELECT hypothesis FROM candidates WHERE id=?", (search["family_id"],))
        stage = f"PARAM{index + 1}"
        ids = self.store.enqueue_parameter_stage(
            search["family_id"], search["run_id"], run["round"] + 1, root["hypothesis"],
            trials, json.loads(baseline["settings"]), baseline["id"], stage,
            selected["Period"], selected)
        with self.store.tx() as db:
            db.execute("UPDATE runs SET round=?,decision_due=1 WHERE id=?", (run["round"] + 1, run["id"]))
        log.info("参数搜索已排队 family_id=%s stage=%s parameter=%s values=%s baseline=%s candidate_ids=%s",
                 search["family_id"], stage, parameter["name"], parameter["values"], baseline["id"], ids)
        return f"{stage} search queued for family {search['family_id']}"

    def _advance_parameter_searches(self, run_id):
        searches = self.store.all("SELECT * FROM parameter_searches WHERE run_id=? AND stage NOT IN "
                                  "('READY','ABANDONED') ORDER BY updated_at,family_id", (run_id,))
        for search in searches:
            stage = search["stage"]
            trials = self._search_trials(search)
            if stage != "PARAM2_DECISION" and (not trials or any(c["status"] not in TERMINAL for c in trials)):
                continue
            parameters = json.loads(search["parameters"])
            if stage == "PERIOD":
                passing = [candidate for candidate in trials if sharpe(candidate) is not None and abs(sharpe(candidate)) > self.config.sharpe_gate]
                if not passing:
                    self._set_search_stage(search, "ABANDONED")
                    log.info("假设族放弃 family_id=%s reason=Period 测试均未达到 abs(Sharpe)>%s results=%s",
                             search["family_id"], self.config.sharpe_gate,
                             [(c["parameter_values"], sharpe(c)) for c in trials])
                    return f"Period gate failed for family {search['family_id']}"
                best = best_candidate(passing)
                selected = json.loads(best["parameter_values"])
                log.info("Period 搜索通过 family_id=%s best_period=%s best_sharpe=%s candidate=%s",
                         search["family_id"], selected["Period"], sharpe(best), best["id"])
                if parameters:
                    return self._queue_dimension(search, 0, best)
                self._set_search_stage(search, "READY", selected)
                return f"Period search completed for family {search['family_id']}"
            if stage == "PARAM1":
                best = best_candidate(trials)
                selected = json.loads(best["parameter_values"])
                next_stage = "PARAM2_DECISION" if len(parameters) > 1 else "READY"
                self._set_search_stage(search, next_stage, selected)
                log.info("第二维搜索完成 family_id=%s selected=%s sharpe=%s next=%s",
                         search["family_id"], selected, sharpe(best), next_stage)
                return f"PARAM1 search completed for family {search['family_id']}"
            if stage == "PARAM2_DECISION":
                if self.store.cooling("llm"):
                    return "LLM cooling"
                context = {"family_id": search["family_id"], "history": self._summary(run_id, search["family_id"]),
                           "selected_values": json.loads(search["selected_params"]),
                           "next_parameter": parameters[1],
                           "daily_simulations_remaining": max(0, self.config.max_simulations_day - self._today_simulations())}
                try:
                    decision = self.judge.decide_parameter(context)
                except RemoteError as exc:
                    return self._throttle("llm", exc)
                except ValueError as exc:
                    self.store.set_cooldown("llm", 60, str(exc))
                    return f"invalid parameter decision: {exc}"
                log.info("第三维探索决策 family_id=%s explore=%s reason=%s", search["family_id"],
                         decision["explore"], decision["reason"])
                if decision["explore"]:
                    baseline = self.store.one("SELECT * FROM candidates WHERE id=?", (best_candidate(trials)["id"],))
                    return self._queue_dimension(search, 1, baseline)
                self._set_search_stage(search, "READY")
                return f"PARAM2 skipped for family {search['family_id']}"
            if stage == "PARAM2":
                best = best_candidate(trials)
                selected = json.loads(best["parameter_values"])
                self._set_search_stage(search, "READY", selected)
                log.info("第三维搜索完成 family_id=%s selected=%s sharpe=%s",
                         search["family_id"], selected, sharpe(best))
                return f"PARAM2 search completed for family {search['family_id']}"
        return None

    def _advance_candidate(self, candidate):
        if self.store.cooling("brain") and candidate["status"] not in {"local_wait"} and not (
            candidate["status"] == "queued" and self.quantgpt and not candidate["quantgpt_task_id"]
        ):
            return "BRAIN cooling"
        status = candidate["status"]
        try:
            if status == "queued" and self.quantgpt and not candidate["quantgpt_task_id"]:
                log.info("QuantGPT 本地回测开始 candidate=%s formula=%s", candidate["id"], candidate["expression"])
                task_id = self.quantgpt.start(candidate["expression"])
                self.store.update_candidate(candidate["id"], status="local_wait", quantgpt_task_id=task_id)
                return f"QuantGPT backtest started: {candidate['id']}"
            if status == "local_wait":
                result = self.quantgpt.poll(candidate["quantgpt_task_id"])
                log.debug("QuantGPT 返回 candidate=%s result=%s", candidate["id"], json.dumps(result, ensure_ascii=False))
                if result.get("status") == "failed":
                    self.store.update_candidate(candidate["id"], status="local_rejected", result=result)
                    return f"QuantGPT rejected: {candidate['id']}"
                if result.get("status") != "completed":
                    self.store.update_candidate(candidate["id"], not_before=self._later(10))
                    return "QuantGPT backtest pending"
                summary = result.get("result") or {}
                score = summary.get("score", (summary.get("scoring") or {}).get("score", 0))
                log.info("QuantGPT 结果 candidate=%s score=%s threshold=%s", candidate["id"], score,
                         self.config.quantgpt_min_score)
                if float(score or 0) < self.config.quantgpt_min_score:
                    self.store.update_candidate(candidate["id"], status="local_rejected", result=result)
                    return f"QuantGPT score below threshold: {candidate['id']}"
                self.store.update_candidate(candidate["id"], status="queued", result={"quantgpt": summary})
                return f"QuantGPT passed: {candidate['id']}"
            if status == "queued":
                daily_test_number = self._today_simulations() + 1
                log.info("BRAIN 模拟请求 今日第%s次测试 date=%s candidate=%s formula=%s settings=%s",
                         daily_test_number, today(), candidate["id"],
                         candidate["expression"], candidate["settings"])
                # Claim before POST. A crash between POST and Location persistence leaves
                # an unknown outcome, which is intentionally not submitted again.
                self.store.update_candidate(candidate["id"], status="starting", simulated_at=now())
                try:
                    location, wait = self.brain.start_simulation(candidate["expression"], json.loads(candidate["settings"]))
                except RemoteError as exc:
                    limit_error = exc.status in {400, 403} and any(
                        phrase in str(exc).upper() for phrase in ("RATE_LIMIT", "SIMULATION_LIMIT", "QUOTA")
                    )
                    if exc.status == 429 or limit_error:
                        self.store.update_candidate(candidate["id"], status="queued", simulated_at=None)
                        return self._throttle("brain", exc)
                    if exc.status == 401:
                        self.store.update_candidate(candidate["id"], status="queued", simulated_at=None)
                        self.store.set_cooldown("brain", 300, "BRAIN authentication failed")
                        return "BRAIN authentication failed; retry after cooldown"
                    if exc.status in {400, 403, 422}:
                        self.store.update_candidate(candidate["id"], status="failed", result={"error": str(exc), "http_status": exc.status})
                        log.warning("BRAIN 拒绝公式 candidate=%s http=%s detail=%s", candidate["id"], exc.status, exc)
                        return f"BRAIN rejected expression: {candidate['id']}"
                    self.store.update_candidate(candidate["id"], status="unknown", result={"error": str(exc)})
                    if exc.status == 503:
                        self._throttle("brain", exc)
                    return f"simulation outcome unknown: {candidate['id']}"
                self.store.update_candidate(candidate["id"], status="simulating", simulation_url=location, not_before=self._later(wait))
                log.info("BRAIN 模拟已启动 今日第%s次测试 candidate=%s poll_after=%ss",
                         daily_test_number, candidate["id"], wait)
                return f"BRAIN simulation started: {candidate['id']}"
            if status == "simulating":
                data, wait = self.brain.poll_simulation(candidate["simulation_url"])
                log.debug("BRAIN 模拟返回 candidate=%s result=%s", candidate["id"], json.dumps(data, ensure_ascii=False))
                platform_status = str(data.get("status") or "").upper()
                log.info("BRAIN 模拟状态 candidate=%s status=%s progress=%s alpha_id=%s metrics=%s", candidate["id"],
                         platform_status or "(pending)", data.get("progress"), data.get("alpha_id"),
                         result_summary(data))
                if platform_status == "WARNING":
                    log.info("BRAIN 模拟警告 candidate=%s alpha_id=%s message=%s location=%s；继续轮询",
                             candidate["id"], data.get("alpha_id"), data.get("message"), data.get("location"))
                if platform_status in {"ERROR", "FAILED"} or str(data.get("alpha_status") or "").upper() in {"ERROR", "FAILED"}:
                    log.warning("BRAIN 模拟失败 candidate=%s status=%s message=%s location=%s",
                                candidate["id"], data.get("alpha_status") or platform_status,
                                data.get("alpha_message") or data.get("message"), data.get("location"))
                    self.store.update_candidate(candidate["id"], status="failed", alpha_id=data.get("alpha_id"),
                                                result=retain_local_result(candidate, data))
                    return f"BRAIN simulation failed: {candidate['id']}"
                if platform_status in {"DONE", "COMPLETE"} or (
                    platform_status == "WARNING" and data.get("alpha_id") and data.get("is")
                ):
                    self.store.update_candidate(candidate["id"], status="checking", alpha_id=data.get("alpha_id"),
                                                result=retain_local_result(candidate, data), not_before=self._later(wait))
                    return f"BRAIN simulation complete: {candidate['id']}"
                self.store.update_candidate(candidate["id"], not_before=self._later(wait))
                return f"BRAIN simulation pending: {candidate['id']}"
            if status == "checking":
                if not candidate["alpha_id"]:
                    self.store.update_candidate(candidate["id"], status="unknown")
                    return "BRAIN did not return an alpha ID"
                platform = self.brain.get_alpha(candidate["alpha_id"])
                log.debug("BRAIN Alpha 详情 candidate=%s result=%s", candidate["id"], json.dumps(platform, ensure_ascii=False))
                is_data = platform.get("is") or {}
                checks = is_data.get("checks") or []
                log.info("BRAIN 检查 candidate=%s alpha_id=%s metrics=%s", candidate["id"],
                         candidate["alpha_id"], result_summary(platform))
                if is_checks_passed(is_data):
                    self.store.update_candidate(candidate["id"], status="passed", result=retain_local_result(candidate, platform))
                    updated = self.store.one("SELECT * FROM candidates WHERE id=?", (candidate["id"],))
                    archive(self.config.library_dir, updated, platform, "BRAIN IS 检查通过，待模型决策")
                    return f"BRAIN checks passed: {candidate['id']}"
                if checks and any(str(c.get("result", "")).upper() == "FAIL" for c in checks):
                    self.store.update_candidate(candidate["id"], status="rejected", result=retain_local_result(candidate, platform))
                    return f"BRAIN checks failed: {candidate['id']}"
                self.store.update_candidate(candidate["id"], not_before=self._later(30))
                return f"BRAIN checks pending: {candidate['id']}"
        except RemoteError as exc:
            if exc.status in {429, 503}:
                return self._throttle("quantgpt" if self.quantgpt and status in {"queued", "local_wait"} else "brain", exc)
            self.store.update_candidate(candidate["id"], not_before=self._later(60))
            return f"remote error; retry read: {exc}"
        raise RuntimeError(f"Unhandled candidate status: {status}")

    @staticmethod
    def _later(seconds):
        from datetime import timedelta
        return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()

    def _decide(self, run, family):
        if self.store.cooling("llm"):
            return "LLM cooling"
        context = {"objective": run["objective"], "direction": run["direction"],
                   "family_id": family["id"], "latest_candidate_id": family["latest_candidate_id"],
                   "stagnation_count": family["stagnation_count"],
                   "max_stagnant": self.config.family_max_stagnant,
                   "history": self._summary(run["id"], family["id"]),
                   "today_submission_used": bool(self.store.submission_today(self.config.account)),
                   "daily_simulations_remaining": max(0, self.config.max_simulations_day - self._today_simulations())}
        search = self.store.one("SELECT stage,selected_params FROM parameter_searches WHERE family_id=?",
                                (family["id"],))
        if search:
            context["parameter_search"] = {"stage": search["stage"],
                                           "selected_values": json.loads(search["selected_params"])}
        log.info("假设族复盘 family_id=%s candidates=%s stagnant=%s/%s latest=%s",
                 family["id"], len(family["members"]), family["stagnation_count"],
                 self.config.family_max_stagnant, family["latest_candidate_id"])
        try:
            decision = self.judge.decide(context)
        except RemoteError as exc:
            return self._throttle("llm", exc)
        except ValueError as exc:
            self.store.set_cooldown("llm", 60, str(exc))
            log.warning("模型决策结果无效 reason=%s", exc)
            return f"invalid decision response: {exc}"
        action = decision["action"]
        log.info("模型决策 run_id=%s action=%s reason=%s target=%s next_direction=%s changes=%s optimized_candidates=%s optimized_hypothesis=%s optimized_expression=%s",
                 run["id"], action, decision["reason"], decision["target_id"],
                 decision["next_direction"], decision["parameter_changes"],
                 decision.get("optimized_candidates"),
                 decision.get("optimized_hypothesis"), decision.get("optimized_expression"))
        log.debug("模型决策输入 context=%s", json.dumps(context, ensure_ascii=False))
        target = self.store.one("SELECT * FROM candidates WHERE id=? AND run_id=?", (decision["target_id"], run["id"])) if decision["target_id"] else None
        if action in {"TUNE", "SUBMIT", "OPTIMIZE"} and not target:
            self.store.set_cooldown("llm", 60, "decision target missing")
            log.warning("模型决策未执行：候选不存在 target=%s", decision["target_id"])
            return "invalid model decision: target candidate missing"
        if target and target["id"] not in {member["id"] for member in family["members"]}:
            self.store.set_cooldown("llm", 60, "target outside current family")
            log.warning("模型决策未执行：目标不属于当前假设族 family_id=%s target=%s", family["id"], target["id"])
            return "invalid model decision: target outside current family"
        if action == "SUBMIT" and target["status"] != "passed":
            self.store.set_cooldown("llm", 60, "candidate checks not passed")
            log.warning("模型决策未执行：候选检查未通过 target=%s status=%s", target["id"], target["status"])
            return "invalid model decision: candidate checks have not passed"
        if action == "TUNE":
            changes = decision["parameter_changes"]
            if not (0 <= changes["decay"] <= 20 and 0 <= changes["truncation"] <= 0.5 and
                    changes["neutralization"] in {"NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"}):
                self.store.set_cooldown("llm", 60, "invalid parameter changes")
                return "invalid model decision: parameter changes out of range"
            settings = json.loads(target["settings"])
            settings.update(changes)
            new_id = self.store.add_candidate(run["id"], run["round"] + 1,
                                              target["hypothesis"] + " [parameter tuning]", target["expression"],
                                              settings, target["id"], "TUNE")
            if not new_id:
                self.store.set_cooldown("llm", 60, "duplicate parameter combination")
                return "invalid model decision: duplicate parameter combination"
        if action == "OPTIMIZE":
            proposals = decision.get("optimized_candidates") or [{
                "hypothesis": decision.get("optimized_hypothesis", ""),
                "expression": decision.get("optimized_expression", "")
            }]
            if not 1 <= len(proposals) <= 2:
                self.store.set_cooldown("llm", 60, "invalid optimized candidate count")
                return "invalid model decision: expected one or two optimized candidates"
            normalized = []
            seen = set()
            for proposal in proposals:
                hypothesis = proposal.get("hypothesis", "").strip()
                expression = proposal.get("expression", "").strip()
                key = "".join(expression.split()).lower()
                if (not hypothesis or not valid_expression(expression) or
                        key == "".join(target["expression"].split()).lower() or key in seen):
                    self.store.set_cooldown("llm", 60, "invalid optimized candidate")
                    return "invalid model decision: optimized hypotheses or expressions are invalid or repeated"
                seen.add(key)
                normalized.append((hypothesis, expression, key))
            settings = json.loads(target["settings"])
            settings_json = json.dumps(settings, sort_keys=True)
            for _, _, key in normalized:
                if self.store.one("SELECT id FROM candidates WHERE run_id=? AND expression_key=? AND settings=?",
                                  (run["id"], key, settings_json)):
                    self.store.set_cooldown("llm", 60, "duplicate optimized expression")
                    return "invalid model decision: optimized expression already tested"
            for hypothesis, expression, _ in normalized:
                new_id = self.store.add_candidate(run["id"], run["round"] + 1, hypothesis, expression,
                                                  settings, target["id"], "OPTIMIZE")
                if not new_id:
                    self.store.set_cooldown("llm", 60, "duplicate optimized expression")
                    return "invalid model decision: optimized expression already tested"
                log.info("模型优化产生新候选 parent=%s candidate=%s hypothesis=%s formula=%s", target["id"],
                         new_id, hypothesis, expression)
        with self.store.tx() as db:
            db.execute("INSERT INTO decisions VALUES (?,?,?,?,?)", (uuid.uuid4().hex[:12], run["id"], json.dumps(context, ensure_ascii=False), json.dumps(decision, ensure_ascii=False), now()))
            db.execute("UPDATE runs SET decision_due=0, direction=? WHERE id=?", (decision["next_direction"] or run["direction"], run["id"]))
            if action in {"TUNE", "OPTIMIZE"}:
                db.execute("UPDATE runs SET round=?, decision_due=1 WHERE id=?", (run["round"] + 1, run["id"]))
        if action == "SUBMIT":
            return self._submit(target, decision)
        return f"model chose {action}: {decision['reason']}"

    def _submit(self, candidate, decision):
        log.info("提交评估 candidate=%s alpha_id=%s auto_submit=%s", candidate["id"],
                 candidate["alpha_id"], self.config.auto_submit)
        if not self.config.auto_submit:
            return f"model recommended submission (dry run): {candidate['alpha_id']}"
        previous = self.store.one("SELECT status FROM submissions WHERE alpha_id=?", (candidate["alpha_id"],))
        if previous:
            return f"alpha already has a submission record: {candidate['alpha_id']} ({previous['status']})"
        if self.store.components_used(candidate["id"]):
            return f"factor already used by another submission: {candidate['id']}"
        day = today()
        if not self.store.reserve_submission(self.config.account, candidate["id"], candidate["alpha_id"], day):
            return "daily submission slot already used; continue mining"
        try:
            # Confirm the platform checks immediately before the irreversible POST.
            platform = self.brain.get_alpha(candidate["alpha_id"])
            log.info("提交前 BRAIN 检查 alpha_id=%s metrics=%s", candidate["alpha_id"], result_summary(platform))
            if not is_checks_passed(platform.get("is") or {}):
                self.store.update_submission(self.config.account, day, "rejected", "checks changed before submission")
                return "submission blocked: checks no longer pass"
            if str(platform.get("status") or "").upper() in {"ACTIVE", "SUBMITTED"}:
                self.store.update_submission(self.config.account, day, "rejected", "platform already shows submitted")
                return "submission blocked: platform already shows submitted"
            if today() != day:
                self.store.update_submission(self.config.account, day, "failed", "midnight crossed before POST")
                return "midnight crossed before POST; decide again on the new day"
            self.brain.submit(candidate["alpha_id"])
            log.info("BRAIN 正式提交已发送 alpha_id=%s day=%s", candidate["alpha_id"], day)
            self.store.mark_submission_sent(self.config.account, day)
            archive(self.config.library_dir, candidate, platform, "已发起正式提交，等待平台确认", decision)
            return f"submitted one alpha today: {candidate['alpha_id']}"
        except RemoteError as exc:
            self.store.update_submission(self.config.account, day, "uncertain", str(exc))
            if exc.status in {429, 503}:
                self._throttle("brain", exc)
            return f"submission outcome uncertain; platform reconciliation required: {exc}"
