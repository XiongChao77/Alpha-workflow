"""Autonomous Topic -> Direction -> Experiment research orchestration.

The agents are small decision interfaces over the configured LLM. All BRAIN
operations, parameter expansion, persistence and budgets remain deterministic.
"""

from collections import Counter, defaultdict
import json
import logging
from statistics import median
import uuid

from .engine import TERMINAL, valid_expression
from .parameter_search import period_values, render_expression, sharpe
from .store import now
from .transport import RemoteError

log = logging.getLogger(__name__)


class ResearchAgent:
    def __init__(self, model):
        self.model = model

    def topic(self, context):
        return self.model.generate_topic(context)

    def directions(self, context):
        return self.model.propose_directions(context)

    def feedback(self, context):
        return self.model.analyze_batch(context)


class AlphaAgent:
    def __init__(self, model):
        self.model = model

    def variants(self, context):
        return self.model.generate_variants(context)

    def repair(self, context):
        return self.model.repair_expression(context)


def batch_statistics(rows):
    """Report distributions and parameter regions; never invent correlations."""
    status = Counter(row["status"] for row in rows)
    metric_values = defaultdict(list)
    checks = Counter()
    by_structure = defaultdict(list)
    by_period = defaultdict(list)
    correlation = []
    details = []
    for row in rows:
        result = json.loads(row["result"] or "{}")
        is_data = result.get("is") or {}
        params = json.loads(row["parameter_values"] or "{}")
        metrics = {name: is_data.get(name) for name in ("sharpe", "fitness", "returns", "turnover")}
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metric_values[name].append(float(value))
        for check in is_data.get("checks") or []:
            if check.get("result") == "FAIL":
                checks[check.get("name") or "UNKNOWN"] += 1
            if "CORRELATION" in str(check.get("name") or "") and check.get("value") is not None:
                correlation.append({"candidate_id": row["id"], "name": check["name"],
                                    "result": check.get("result"), "value": check["value"]})
        value = sharpe(row)
        if value is not None:
            by_structure[str(row.get("variant_index"))].append(value)
            if "Period" in params:
                by_period[str(params["Period"])].append(value)
        details.append({"id": row["id"], "variant_index": row.get("variant_index"),
                        "parent_candidate_id": row.get("parent_candidate_id"),
                        "origin_action": row.get("origin_action"), "hypothesis": row["hypothesis"],
                        "expression": row["expression"], "parameter_values": params,
                        "status": row["status"], "alpha_id": row["alpha_id"],
                        "metrics": metrics, "checks": is_data.get("checks") or [],
                        "error": result.get("error") or result.get("message")})
    def distribution(values):
        return {"count": len(values), "min": min(values), "median": median(values),
                "max": max(values), "mean": round(sum(values) / len(values), 4)} if values else None
    return {"count": len(rows), "statuses": dict(status), "passed_rate": round(status["passed"] / len(rows), 4) if rows else 0,
            "metrics": {key: distribution(values) for key, values in metric_values.items()},
            "failed_checks": dict(checks),
            "structure_sharpe": {key: distribution(values) for key, values in by_structure.items()},
            "period_sharpe": {key: distribution(values) for key, values in by_period.items()},
            "correlation_evidence": correlation or "unavailable; BRAIN supplied no correlation values",
            "candidates": details}


def batch_score(statistics):
    """Direction progress favors repeatable quality over a lone high Sharpe."""
    sharpe_stats = statistics.get("metrics", {}).get("sharpe") or {}
    return (statistics.get("passed_rate", 0),
            abs(sharpe_stats["median"]) if sharpe_stats.get("median") is not None else float("-inf"),
            max(abs(sharpe_stats["min"]), abs(sharpe_stats["max"])) if sharpe_stats else float("-inf"))


class ResearchWorkflow:
    def __init__(self, config, store, research_agent, alpha_agent, execution_engine):
        if config.max_combination_pairs < 0 or any(not 0 < weight < 1 for weight in config.combination_weights):
            raise ValueError("Combination pairs and weights must be nonnegative and within (0,1)")
        self.config, self.store = config, store
        self.research, self.alpha, self.engine = research_agent, alpha_agent, execution_engine

    def add_topic(self, title, objective=None, rationale="研究员提交", priority=0, source="researcher"):
        topic_id = uuid.uuid4().hex[:12]
        with self.store.tx() as db:
            db.execute("INSERT INTO topics VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (topic_id, None, title.strip(), (objective or title).strip(), rationale.strip(),
                        priority, source, "pending", None, now(), now()))
        return topic_id

    def start_discussion(self):
        discussion_id = uuid.uuid4().hex[:12]
        with self.store.tx() as db:
            db.execute("INSERT INTO topic_discussions VALUES (?,?,?,?,?,?)",
                       (discussion_id, "open", None, None, now(), now()))
        return discussion_id

    def discussion(self, discussion_id):
        session = self.store.one("SELECT * FROM topic_discussions WHERE id=?", (discussion_id,))
        if not session:
            raise ValueError("Unknown discussion ID")
        messages = self.store.all("SELECT role,content,proposal FROM topic_messages WHERE discussion_id=? "
                                  "ORDER BY created_at,rowid", (discussion_id,))
        history = [{"role": row["role"], "content": row["content"],
                    **({"topic": json.loads(row["proposal"])} if row["proposal"] else {})}
                   for row in messages]
        return session, history

    def append_discussion(self, discussion_id, role, content, proposal=None):
        if role not in {"researcher", "agent"} or not content.strip():
            raise ValueError("Invalid discussion message")
        with self.store.tx() as db:
            session = db.execute("SELECT status FROM topic_discussions WHERE id=?", (discussion_id,)).fetchone()
            if not session or session["status"] != "open":
                raise ValueError("Discussion is not open")
            db.execute("INSERT INTO topic_messages VALUES (?,?,?,?,?,?)",
                       (uuid.uuid4().hex[:12], discussion_id, role, content,
                        json.dumps(proposal, ensure_ascii=False) if proposal else None, now()))
            db.execute("UPDATE topic_discussions SET proposal=COALESCE(?,proposal),updated_at=? WHERE id=?",
                       (json.dumps(proposal, ensure_ascii=False) if proposal else None, now(), discussion_id))

    def submit_discussion(self, discussion_id, priority=100):
        """Confirm a draft and queue it exactly once, without touching active work."""
        with self.store.tx() as db:
            db.execute("BEGIN IMMEDIATE")
            session = db.execute("SELECT * FROM topic_discussions WHERE id=?", (discussion_id,)).fetchone()
            if not session:
                raise ValueError("Unknown discussion ID")
            if session["topic_id"]:
                return session["topic_id"]
            if session["status"] != "open" or not session["proposal"]:
                raise ValueError("Discussion has no confirmed Topic draft")
            proposal = json.loads(session["proposal"])
            topic_id = uuid.uuid4().hex[:12]
            db.execute("INSERT INTO topics VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (topic_id, None, proposal["title"].strip(), proposal["objective"].strip(),
                        proposal["rationale"].strip(), priority, "researcher", "pending", None, now(), now()))
            db.execute("UPDATE topic_discussions SET status='submitted',topic_id=?,updated_at=? WHERE id=?",
                       (topic_id, now(), discussion_id))
            return topic_id

    def _model(self, call):
        if self.store.cooling("llm"):
            return None, "LLM cooling"
        try:
            return call(), None
        except RemoteError as exc:
            return None, self.engine._throttle("llm", exc)
        except ValueError as exc:
            self.store.set_cooldown("llm", 60, str(exc))
            log.warning("研究模型输出无效 reason=%s", exc)
            return None, f"invalid model output: {exc}"

    def _active_topic(self):
        return self.store.one("SELECT * FROM topics WHERE status='active' ORDER BY created_at LIMIT 1")

    def _next_topic(self):
        pending = self.store.one("SELECT * FROM topics WHERE status='pending' ORDER BY priority DESC,created_at,id LIMIT 1")
        if pending:
            return pending, None
        legacy = self.store.one("SELECT r.* FROM runs r LEFT JOIN topics t ON t.run_id=r.id "
                                "WHERE r.status='running' AND t.id IS NULL ORDER BY r.created_at DESC LIMIT 1")
        if legacy:
            topic_id = uuid.uuid4().hex[:12]
            with self.store.tx() as db:
                db.execute("INSERT INTO topics VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                           (topic_id, legacy["id"], legacy["direction"], legacy["objective"],
                            "从旧版运行任务迁入", -1, "legacy", "pending", None, now(), now()))
            log.info("旧研究任务纳入 Topic Queue topic_id=%s run_id=%s", topic_id, legacy["id"])
            return self.store.one("SELECT * FROM topics WHERE id=?", (topic_id,)), None
        history = self.store.all("SELECT title,objective,summary FROM topics WHERE status='completed' ORDER BY updated_at DESC LIMIT 10")
        feedback = self.store.all("SELECT statistics,decision FROM feedback ORDER BY created_at DESC LIMIT 8")
        suggestion, error = self._model(lambda: self.research.topic({
            "default_objective": self.config.research_objective(), "default_direction": self.config.research_direction(),
            "recent_topics": history, "recent_feedback": feedback}))
        if error:
            return None, error
        topic_id = self.add_topic(suggestion["title"], suggestion["objective"], suggestion["rationale"], source="agent")
        log.info("自主生成 Topic topic_id=%s title=%s rationale=%s", topic_id, suggestion["title"], suggestion["rationale"])
        return self.store.one("SELECT * FROM topics WHERE id=?", (topic_id,)), None

    def _activate_topic(self, topic):
        run_id = topic["run_id"] or self.store.create_run(topic["objective"], topic["title"])
        with self.store.tx() as db:
            db.execute("UPDATE topics SET status='active',run_id=?,updated_at=? WHERE id=?",
                       (run_id, now(), topic["id"]))
            if topic["source"] == "legacy":
                existing = db.execute("SELECT COUNT(*) AS n FROM candidates WHERE run_id=? AND topic_id IS NULL",
                                      (run_id,)).fetchone()["n"]
                if existing:
                    direction_id, experiment_id = uuid.uuid4().hex[:12], uuid.uuid4().hex[:12]
                    db.execute("INSERT INTO directions VALUES (?,?,?,?,?,?,?,?,?,?)",
                               (direction_id, topic["id"], topic["title"], topic["objective"],
                                "旧版候选批次；先完成在途结果，再分析", "active", 1, None, now(), now()))
                    db.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?)",
                               (experiment_id, direction_id, 1, "完成迁入候选并集中复盘", "LEGACY", "active",
                                "[]", None, now(), now()))
                    db.execute("UPDATE candidates SET topic_id=?,direction_id=?,experiment_id=? "
                               "WHERE run_id=? AND topic_id IS NULL",
                               (topic["id"], direction_id, experiment_id, run_id))
                    log.info("旧候选迁入 experiment_id=%s count=%s", experiment_id, existing)
        log.info("Topic 启动 topic_id=%s run_id=%s title=%s priority=%s", topic["id"], run_id,
                 topic["title"], topic["priority"])
        return self.store.one("SELECT * FROM topics WHERE id=?", (topic["id"],))

    def _directions(self, topic):
        rows = self.store.all("SELECT * FROM directions WHERE topic_id=? ORDER BY created_at,id", (topic["id"],))
        if not rows:
            items, error = self._model(lambda: self.research.directions({
                "topic": {key: topic[key] for key in ("title", "objective", "rationale")},
                "max_directions": self.config.max_directions_topic,
                "previous_feedback": self.store.all("SELECT statistics,decision FROM feedback ORDER BY created_at DESC LIMIT 5")}))
            if error:
                return None, error
            with self.store.tx() as db:
                for item in items:
                    db.execute("INSERT INTO directions VALUES (?,?,?,?,?,?,?,?,?,?)",
                               (uuid.uuid4().hex[:12], topic["id"], item["title"], item["hypothesis"],
                                item["mechanism"], "pending", 0, None, now(), now()))
            log.info("Direction 已规划 topic_id=%s count=%s titles=%s", topic["id"], len(items),
                     [item["title"] for item in items])
            rows = self.store.all("SELECT * FROM directions WHERE topic_id=? ORDER BY created_at,id", (topic["id"],))
        active = next((row for row in rows if row["status"] == "active"), None)
        if active:
            return active, None
        pending = next((row for row in rows if row["status"] == "pending"), None)
        if pending:
            with self.store.tx() as db:
                db.execute("UPDATE directions SET status='active',updated_at=? WHERE id=?", (now(), pending["id"]))
            log.info("Direction 启动 direction_id=%s title=%s hypothesis=%s", pending["id"],
                     pending["title"], pending["hypothesis"])
            return self.store.one("SELECT * FROM directions WHERE id=?", (pending["id"],)), None
        summary = self.store.all("SELECT title,conclusion FROM directions WHERE topic_id=?", (topic["id"],))
        with self.store.tx() as db:
            db.execute("UPDATE topics SET status='completed',summary=?,updated_at=? WHERE id=?",
                       (json.dumps(summary, ensure_ascii=False), now(), topic["id"]))
            db.execute("UPDATE runs SET status='finished' WHERE id=?", (topic["run_id"],))
        log.info("Topic 完成 topic_id=%s summary=%s", topic["id"], summary)
        return None, f"Topic completed: {topic['id']}"

    def _create_experiment(self, topic, direction, goal, plans_override=None):
        requested = min(self.config.batch_size, max(1, self.config.max_experiment_candidates // 4))
        context = {"topic": topic["title"], "direction": {key: direction[key] for key in
                    ("title", "hypothesis", "mechanism")}, "goal": goal,
                   "requested_count": requested, "settings": self.config.settings(),
                   "allowed_fields": self.config.allowed_fields,
                   "previous_feedback": self.store.all(
                       "SELECT f.statistics,f.decision FROM feedback f JOIN experiments e ON e.id=f.experiment_id "
                       "WHERE e.direction_id=? ORDER BY f.created_at DESC LIMIT 3", (direction["id"],))}
        if plans_override is None:
            plans, error = self._model(lambda: self.alpha.variants(context))
            if error:
                return error
        else:
            plans = plans_override
        for index, plan in enumerate(plans):
            log.info("Alpha 结构变体 direction_id=%s index=%s hypothesis=%s template=%s period=%s parameters=%s",
                     direction["id"], index, plan["hypothesis"], plan["expression_template"],
                     plan["period"]["default"], plan["parameters"])
        experiment_id = uuid.uuid4().hex[:12]
        ordinal = direction["iteration"] + 1
        trials = []
        for index, plan in enumerate(plans):
            defaults = {item["name"]: item["default"] for item in plan["parameters"]}
            for period in period_values(self.config.period_grid, plan["period"]["default"]):
                values = {"Period": period, **defaults}
                expression = render_expression(plan["expression_template"], values)
                if valid_expression(expression):
                    trials.append((index, plan["hypothesis"], expression, values,
                                   plan.get("parent_candidate_id")))
        if not trials:
            self.store.set_cooldown("llm", 300, "no valid expressions")
            return "no valid expressions"
        accepted = 0
        with self.store.tx() as db:
            db.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (experiment_id, direction["id"], ordinal, goal, "PERIOD", "active",
                        json.dumps(plans, ensure_ascii=False), None, now(), now()))
            db.execute("UPDATE directions SET iteration=?,updated_at=? WHERE id=?", (ordinal, now(), direction["id"]))
            db.execute("UPDATE runs SET round=round+1 WHERE id=?", (topic["run_id"],))
            for index, hypothesis, expression, values, parent in trials:
                accepted += self._insert_candidate(db, topic, direction, experiment_id, ordinal,
                                                   index, hypothesis, expression, values, parent, "PERIOD")
            if not accepted:
                db.execute("UPDATE experiments SET stage='READY',status='failed',updated_at=? WHERE id=?", (now(), experiment_id))
        if not accepted:
            self.store.set_cooldown("llm", 300, "no valid novel expressions")
            return "no valid novel expressions"
        log.info("实验批次已排队 experiment_id=%s direction_id=%s ordinal=%s structures=%s candidates=%s goal=%s",
                 experiment_id, direction["id"], ordinal, len(plans), accepted, goal)
        return f"experiment queued: {experiment_id} ({accepted} candidates)"

    def _stage_rows(self, experiment, stage=None):
        if (stage or experiment["stage"]) == "LEGACY":
            return self.store.all("SELECT * FROM candidates WHERE experiment_id=? AND origin_action NOT LIKE 'INVERT_%' "
                                  "ORDER BY created_at,id", (experiment["id"],))
        return self.store.all("SELECT * FROM candidates WHERE experiment_id=? AND origin_action=? ORDER BY created_at,id",
                              (experiment["id"], stage or experiment["stage"]))

    def _insert_candidate(self, db, topic, direction, experiment_id, ordinal, variant_index,
                          hypothesis, expression, values, parent_id, stage):
        expression = expression.strip()
        key = "".join(expression.split()).lower()
        result = db.execute(
            "INSERT OR IGNORE INTO candidates (id,run_id,round,hypothesis,expression,expression_key,settings,status,"
            "parent_candidate_id,origin_action,parameter_values,topic_id,direction_id,experiment_id,variant_index,repair_attempts,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex[:12], topic["run_id"], ordinal, hypothesis, expression, key,
             json.dumps(self.config.settings(), sort_keys=True), "queued", parent_id, stage,
             json.dumps(values, sort_keys=True), topic["id"], direction["id"], experiment_id,
             variant_index, 0, now()))
        return result.rowcount

    def _expand_parameter(self, topic, direction, experiment, stage, selected):
        plans = json.loads(experiment["plans"])
        index = 0 if stage == "PARAM1" else 1
        trials = []
        for variant_index, baseline in selected.items():
            plan = plans[variant_index]
            if len(plan["parameters"]) <= index:
                continue
            parameter = plan["parameters"][index]
            baseline_values = json.loads(baseline["parameter_values"] or "{}")
            for value in parameter["values"]:
                values = {**baseline_values, parameter["name"]: value}
                render_values = {name: values[name] for name in ["Period", *(item["name"] for item in plan["parameters"])]}
                expression = render_expression(plan["expression_template"], render_values)
                if values.get("_inverted"):
                    expression = f"-({expression})"
                if not valid_expression(expression):
                    continue
                trials.append((variant_index, plan["hypothesis"], expression, values, baseline["id"]))
        accepted = 0
        with self.store.tx() as db:
            for variant_index, hypothesis, expression, values, parent_id in trials:
                accepted += self._insert_candidate(db, topic, direction, experiment["id"],
                                                   experiment["ordinal"], variant_index, hypothesis,
                                                   expression, values, parent_id, stage)
            db.execute("UPDATE experiments SET stage=?,updated_at=? WHERE id=?",
                       (stage if accepted else experiment["stage"], now(), experiment["id"]))
        log.info("参数阶段 experiment_id=%s stage=%s accepted=%s selected=%s", experiment["id"], stage,
                 accepted, {key: value["id"] for key, value in selected.items()})
        return accepted

    def _best_by_variant(self, rows, threshold=None):
        groups = defaultdict(list)
        for row in rows:
            if row["variant_index"] is not None and sharpe(row) is not None and (threshold is None or abs(sharpe(row)) > threshold):
                groups[row["variant_index"]].append(row)
        return {key: max(items, key=lambda row: abs(sharpe(row))) for key, items in groups.items()}

    def _positive_by_variant(self, experiment, stage):
        stages = ["PERIOD", "INVERT_PERIOD"]
        if stage in {"PARAM1", "INVERT_PARAM1", "PARAM2", "INVERT_PARAM2"}:
            stages += ["PARAM1", "INVERT_PARAM1"]
        if stage in {"PARAM2", "INVERT_PARAM2"}:
            stages += ["PARAM2", "INVERT_PARAM2"]
        rows = [row for name in stages for row in self._stage_rows(experiment, name)
                if sharpe(row) is not None and sharpe(row) > self.config.sharpe_gate]
        groups = defaultdict(list)
        for row in rows:
            groups[row["variant_index"]].append(row)
        return {index: max(items, key=sharpe) for index, items in groups.items()}

    @staticmethod
    def _fitness(row):
        value = (json.loads(row["result"] or "{}").get("is") or {}).get("fitness")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    @staticmethod
    def _turnover(row):
        value = (json.loads(row["result"] or "{}").get("is") or {}).get("turnover")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def _combination_stopped(self, candidate_id):
        rows = self.store.all("SELECT improved FROM combination_attempts WHERE parent_id=? "
                              "ORDER BY created_at DESC,rowid DESC LIMIT 2", (candidate_id,))
        return len(rows) == 2 and all(not row["improved"] for row in rows)

    def _schedule_combinations(self, topic, direction, experiment):
        if self.config.max_combination_pairs <= 0:
            return 0
        pool = self.store.all(
            "SELECT * FROM candidates WHERE direction_id=? AND origin_action!='COMBINATION' "
            "AND status IN ('passed','rejected') AND alpha_id IS NOT NULL ORDER BY created_at,id",
            (direction["id"],))
        pool = [row for row in pool if sharpe(row) is not None and sharpe(row) > 0 and
                self._fitness(row) is not None and self._turnover(row) is not None and
                len(self.store.component_ids(row["id"])) == 1 and
                not self._combination_stopped(row["id"]) and not self.store.components_used(row["id"]) and
                ";" not in row["expression"]]
        choices = []
        for left_pos, left in enumerate(pool):
            for right in pool[left_pos + 1:]:
                pair_left, pair_right = (left, right) if left["id"] < right["id"] else (right, left)
                left_id, right_id = pair_left["id"], pair_right["id"]
                if self.store.one("SELECT id FROM combination_pairs WHERE direction_id=? AND left_id=? AND right_id=?",
                                  (direction["id"], left_id, right_id)):
                    continue
                choices.append((abs(self._turnover(left) - self._turnover(right)), pair_left, pair_right))
        choices.sort(key=lambda item: item[0], reverse=True)
        selected = choices[:self.config.max_combination_pairs]
        accepted = 0
        with self.store.tx() as db:
            for _, left, right in selected:
                pair_id = uuid.uuid4().hex[:12]
                pair_candidates = 0
                for weight in self.config.combination_weights:
                    if not 0 < weight < 1:
                        continue
                    expression = (f"{weight}*(rank(({left['expression']}))-0.5) + "
                                  f"{round(1-weight, 6)}*(rank(({right['expression']}))-0.5)")
                    if not valid_expression(expression):
                        continue
                    key = "".join(expression.split()).lower()
                    candidate_id = uuid.uuid4().hex[:12]
                    result = db.execute(
                        "INSERT OR IGNORE INTO candidates (id,run_id,round,hypothesis,expression,expression_key,settings,status,"
                        "parent_candidate_id,origin_action,parameter_values,topic_id,direction_id,experiment_id,variant_index,repair_attempts,created_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (candidate_id, topic["run_id"], experiment["ordinal"],
                         f"组合：{left['hypothesis']} + {right['hypothesis']}", expression, key,
                         json.dumps(self.config.settings(), sort_keys=True), "queued", left["id"], "COMBINATION",
                         json.dumps({"pair_id": pair_id, "left_id": left["id"], "right_id": right["id"],
                                     "weight_left": weight}, sort_keys=True), topic["id"], direction["id"],
                         experiment["id"], None, 0, now()))
                    if result.rowcount:
                        db.execute("INSERT INTO combination_members VALUES (?,?,?)", (candidate_id, pair_id, weight))
                        pair_candidates += 1
                if pair_candidates:
                    db.execute("INSERT INTO combination_pairs VALUES (?,?,?,?,?,?,?,?)",
                               (pair_id, direction["id"], experiment["id"], left["id"], right["id"],
                                "active", now(), None))
                    accepted += pair_candidates
                    log.info("同方向因子组合 pair_id=%s left=%s right=%s turnover=(%s,%s) settings=统一使用当前配置 candidates=%s",
                             pair_id, left["id"], right["id"], self._turnover(left), self._turnover(right), pair_candidates)
            if accepted:
                db.execute("UPDATE experiments SET stage='COMBINATION',updated_at=? WHERE id=?", (now(), experiment["id"]))
        return accepted

    def _finish_combinations(self, experiment):
        pairs = self.store.all("SELECT * FROM combination_pairs WHERE experiment_id=? AND status='active'", (experiment["id"],))
        for pair in pairs:
            members = self.store.all(
                "WITH RECURSIVE family(id) AS ("
                "SELECT candidate_id FROM combination_members WHERE pair_id=? "
                "UNION ALL SELECT c.id FROM candidates c JOIN family f ON c.parent_candidate_id=f.id "
                "WHERE c.experiment_id=?) SELECT c.* FROM candidates c JOIN family f ON c.id=f.id",
                (pair["id"], experiment["id"]))
            values = [value for row in members if (value := self._fitness(row)) is not None]
            best = max(values) if values else None
            with self.store.tx() as db:
                for parent_id in (pair["left_id"], pair["right_id"]):
                    parent = db.execute("SELECT result FROM candidates WHERE id=?", (parent_id,)).fetchone()
                    parent_fitness = (json.loads(parent["result"] or "{}").get("is") or {}).get("fitness")
                    if not isinstance(parent_fitness, (int, float)) or isinstance(parent_fitness, bool):
                        parent_fitness = None
                    improved = int(best is not None and parent_fitness is not None and best > parent_fitness)
                    db.execute("INSERT OR IGNORE INTO combination_attempts VALUES (?,?,?,?,?,?)",
                               (pair["id"], parent_id, improved, parent_fitness, best, now()))
                    log.info("组合 Fitness 反馈 pair_id=%s parent=%s parent_fitness=%s best_combo_fitness=%s improved=%s",
                             pair["id"], parent_id, parent_fitness, best, bool(improved))
                db.execute("UPDATE combination_pairs SET status='completed',evaluated_at=? WHERE id=?", (now(), pair["id"]))

    def _queue_inversions(self, topic, direction, experiment, stage, rows):
        """Re-simulate the opposite formula; never infer reverse metrics."""
        accepted = 0
        inverse_stage = "INVERT_" + stage
        with self.store.tx() as db:
            for source in rows:
                if (source["variant_index"] is None and stage != "LEGACY") or sharpe(source) is None or sharpe(source) >= -self.config.sharpe_gate:
                    continue
                expression = source["expression"].strip()
                values = json.loads(source["parameter_values"] or "{}")
                if values.get("_inverted") and expression.startswith("-(") and expression.endswith(")"):
                    expression = expression[2:-1]
                    values.pop("_inverted", None)
                else:
                    expression = f"-({expression})"
                    values["_inverted"] = True
                if valid_expression(expression):
                    accepted += self._insert_candidate(
                        db, topic, direction, experiment["id"], experiment["ordinal"], source["variant_index"],
                        source["hypothesis"] + " [反向验证]", expression, values, source["id"], inverse_stage)
            if accepted:
                db.execute("UPDATE experiments SET stage=?,updated_at=? WHERE id=?",
                           (inverse_stage, now(), experiment["id"]))
        if accepted:
            log.info("负 Sharpe 反向回测 experiment_id=%s stage=%s candidates=%s", experiment["id"], inverse_stage, accepted)
        return accepted

    def _save_feedback(self, experiment, statistics, decision):
        feedback_id = uuid.uuid4().hex[:12]
        with self.store.tx() as db:
            db.execute("INSERT INTO feedback VALUES (?,?,?,?,?,?)",
                       (feedback_id, experiment["id"], experiment["stage"],
                        json.dumps(statistics, ensure_ascii=False), json.dumps(decision, ensure_ascii=False), now()))
            db.execute("UPDATE experiments SET feedback_id=?,updated_at=? WHERE id=?", (feedback_id, now(), experiment["id"]))
        log.info("研究反馈 experiment_id=%s stage=%s action=%s reason=%s conclusion=%s",
                 experiment["id"], experiment["stage"], decision["action"], decision["reason"], decision["conclusion"])

    def _analyze(self, topic, direction, experiment):
        rows = self.store.all("SELECT * FROM candidates WHERE experiment_id=? ORDER BY created_at,id", (experiment["id"],))
        statistics = batch_statistics(rows)
        current_ids = {row["id"] for row in rows}
        related = self.store.all("SELECT * FROM candidates WHERE direction_id=? ORDER BY created_at,id", (direction["id"],))
        by_id = {row["id"]: row for row in related}
        seeds = {}
        for row in related:
            if row["variant_index"] is not None:
                seeds.setdefault((row["experiment_id"], row["variant_index"]), row["id"])
        def family_root(row):
            seen = set()
            while row["parent_candidate_id"] in by_id and row["id"] not in seen:
                seen.add(row["id"])
                row = by_id[row["parent_candidate_id"]]
            return seeds.get((row["experiment_id"], row["variant_index"]), row["id"])
        families = defaultdict(list)
        for row in related:
            families[family_root(row)].append(row)
        family_history = [{"family_id": root, "statistics": batch_statistics(members)}
                          for root, members in families.items() if any(member["id"] in current_ids for member in members)]
        log.info("实验批次统计 experiment_id=%s stage=%s count=%s statuses=%s sharpe=%s failed_checks=%s correlation=%s",
                 experiment["id"], experiment["stage"], statistics["count"], statistics["statuses"],
                 statistics["metrics"].get("sharpe"), statistics["failed_checks"],
                 statistics["correlation_evidence"])
        plans = json.loads(experiment["plans"])
        passing_variants = set(self._positive_by_variant(experiment, experiment["stage"]))
        can_param2 = experiment["stage"] in {"PARAM1", "INVERT_PARAM1"} and any(
            len(plans[index]["parameters"]) > 1 for index in passing_variants)
        completed = self.store.all(
            "SELECT f.statistics FROM experiments e JOIN feedback f ON f.id=e.feedback_id "
            "WHERE e.direction_id=? AND e.status='completed' ORDER BY e.ordinal", (direction["id"],))
        best_score, stagnant = None, 0
        for item in completed:
            score = batch_score(json.loads(item["statistics"]))
            if best_score is None or score > best_score:
                best_score, stagnant = score, 0
            else:
                stagnant += 1
        score = batch_score(statistics)
        stagnant = 0 if best_score is None or score > best_score else stagnant + 1
        context = {"topic": topic["title"], "direction": {key: direction[key] for key in
                    ("title", "hypothesis", "mechanism")}, "experiment": {"id": experiment["id"],
                    "goal": experiment["goal"], "stage": experiment["stage"], "ordinal": experiment["ordinal"]},
                   "statistics": statistics, "can_explore_param2": can_param2,
                   "family_history": family_history,
                   "families": [{"variant_index": index, "hypothesis": plan["hypothesis"],
                                 "expression_template": plan["expression_template"],
                                 "candidates": [item for item in statistics["candidates"] if item["variant_index"] == index]}
                                for index, plan in enumerate(plans)],
                   "stagnant_experiments": stagnant,
                   "max_stagnant_experiments": self.config.family_max_stagnant,
                   "experiments_remaining": max(0, self.config.max_experiments_direction - direction["iteration"]),
                   "daily_simulations_remaining": max(0, self.config.max_simulations_day - self.engine._today_simulations())}
        previous = self.store.one("SELECT stage,decision FROM feedback WHERE experiment_id=? "
                                  "ORDER BY created_at DESC,rowid DESC LIMIT 1", (experiment["id"],))
        if previous and previous["stage"] == experiment["stage"]:
            decision = json.loads(previous["decision"])
        else:
            decision, error = self._model(lambda: self.research.feedback(context))
            if error:
                return error
            if decision["action"] == "EXPLORE_PARAM2" and not can_param2:
                self.store.set_cooldown("llm", 60, "PARAM2 unavailable")
                return "invalid research decision: PARAM2 unavailable"
            if (decision["action"] == "ITERATE" and self.config.family_max_stagnant > 0
                    and stagnant >= self.config.family_max_stagnant):
                decision = {**decision, "action": "SWITCH_DIRECTION",
                            "reason": decision["reason"] + f"; {stagnant} consecutive batches did not improve",
                            "optimized_candidates": []}
            if decision["action"] == "ITERATE" and direction["iteration"] >= self.config.max_experiments_direction:
                decision = {**decision, "action": "SWITCH_DIRECTION",
                            "reason": decision["reason"] + "; direction experiment budget reached",
                            "optimized_candidates": []}
            self._save_feedback(experiment, statistics, decision)
        if decision["action"] == "EXPLORE_PARAM2":
            selected = self._positive_by_variant(experiment, experiment["stage"])
            if self._expand_parameter(topic, direction, experiment, "PARAM2", selected):
                return f"PARAM2 queued: {experiment['id']}"
            decision = {**decision, "action": "ITERATE", "reason": decision["reason"] + "; no novel PARAM2 trials"}
        if experiment["stage"] in {"PARAM1", "INVERT_PARAM1"}:
            count = self._schedule_combinations(topic, direction, experiment)
            if count:
                return f"combination batch queued: {experiment['id']} ({count} candidates)"
        if decision["action"] == "COMPLETE_TOPIC":
            with self.store.tx() as db:
                db.execute("UPDATE experiments SET stage='READY',status='completed',updated_at=? WHERE id=?", (now(), experiment["id"]))
                db.execute("UPDATE directions SET status='completed',conclusion=?,updated_at=? WHERE id=?",
                           (decision["conclusion"], now(), direction["id"]))
                db.execute("UPDATE topics SET status='completed',summary=?,updated_at=? WHERE id=?",
                           (decision["conclusion"], now(), topic["id"]))
                db.execute("UPDATE runs SET status='finished' WHERE id=?", (topic["run_id"],))
            return f"Topic completed: {topic['id']}"
        if decision["action"] == "ITERATE" and direction["iteration"] < self.config.max_experiments_direction:
            with self.store.tx() as db:
                db.execute("UPDATE experiments SET stage='READY',status='completed',updated_at=? WHERE id=?", (now(), experiment["id"]))
            return self._create_experiment(topic, direction, decision["next_goal"] or "研究新的结构变体",
                                           decision["optimized_candidates"])
        with self.store.tx() as db:
            db.execute("UPDATE experiments SET stage='READY',status='completed',updated_at=? WHERE id=?", (now(), experiment["id"]))
            db.execute("UPDATE directions SET status='completed',conclusion=?,updated_at=? WHERE id=?",
                       (decision["conclusion"], now(), direction["id"]))
        return f"direction completed: {direction['id']}"

    def _advance_experiment(self, topic, direction, experiment):
        stage = experiment["stage"]
        if stage == "READY":
            return "experiment complete"
        rows = self._stage_rows(experiment)
        if self.store.one("SELECT id FROM candidates WHERE experiment_id=? AND status NOT IN ('passed','rejected','failed','unknown','local_rejected') LIMIT 1",
                          (experiment["id"],)):
            return "experiment awaiting batch results"
        if any(row["status"] not in TERMINAL for row in rows):
            return "experiment awaiting batch results"
        if stage in {"PERIOD", "PARAM1", "PARAM2", "LEGACY"}:
            if self._queue_inversions(topic, direction, experiment, stage, rows):
                return f"inverse verification queued: {experiment['id']}"
        if stage in {"PERIOD", "INVERT_PERIOD"}:
            selected = self._positive_by_variant(experiment, stage)
            if selected and any(json.loads(experiment["plans"])[index]["parameters"] for index in selected):
                if self._expand_parameter(topic, direction, experiment, "PARAM1", selected):
                    return f"PARAM1 queued: {experiment['id']}"
            count = self._schedule_combinations(topic, direction, experiment)
            if count:
                return f"combination batch queued: {experiment['id']} ({count} candidates)"
            return self._analyze(topic, direction, experiment)
        if stage in {"PARAM2", "INVERT_PARAM2", "LEGACY", "INVERT_LEGACY"}:
            count = self._schedule_combinations(topic, direction, experiment)
            if count:
                return f"combination batch queued: {experiment['id']} ({count} candidates)"
        if stage == "COMBINATION":
            self._finish_combinations(experiment)
        return self._analyze(topic, direction, experiment)

    def _repair_one(self):
        if self.config.max_repair_attempts <= 0:
            return None
        failed = self.store.all(
            "SELECT c.* FROM candidates c JOIN experiments e ON e.id=c.experiment_id "
            "WHERE e.status='active' AND c.status='failed' AND c.repair_attempts<? "
            "ORDER BY c.created_at,c.id LIMIT 10", (self.config.max_repair_attempts,))
        for candidate in failed:
            result = json.loads(candidate["result"] or "{}")
            error = str(result.get("error") or result.get("message") or "")
            if not error or result.get("http_status") not in {400, 422, None}:
                continue
            if not any(token in error.lower() for token in ("unit", "syntax", "expression", "operator", "field", "type", "parse")):
                continue
            if self.store.one("SELECT id FROM candidates WHERE parent_candidate_id=? AND origin_action='REPAIR'", (candidate["id"],)):
                continue
            repaired, failure = self._model(lambda: self.alpha.repair({
                "hypothesis": candidate["hypothesis"], "expression": candidate["expression"],
                "error": error, "attempt": candidate["repair_attempts"] + 1,
                "max_attempts": self.config.max_repair_attempts, "settings": json.loads(candidate["settings"]),
                "allowed_fields": self.config.allowed_fields}))
            if failure:
                return failure
            expression = repaired["expression"].strip()
            self.store.update_candidate(candidate["id"], repair_attempts=self.config.max_repair_attempts)
            if not valid_expression(expression):
                return f"repair invalid: {candidate['id']}"
            child = self.store.add_candidate(
                candidate["run_id"], candidate["round"], repaired["hypothesis"], expression,
                json.loads(candidate["settings"]), parent_candidate_id=candidate["id"],
                origin_action="REPAIR", parameter_values=json.loads(candidate["parameter_values"] or "{}"),
                topic_id=candidate["topic_id"], direction_id=candidate["direction_id"],
                experiment_id=candidate["experiment_id"], variant_index=candidate["variant_index"],
                repair_attempts=candidate["repair_attempts"] + 1)
            log.info("公式修复 parent=%s child=%s attempt=%s reason=%s formula=%s", candidate["id"], child,
                     candidate["repair_attempts"] + 1, repaired["reason"], expression)
            return f"formula repaired: {child or 'duplicate'}"
        return None

    def step(self):
        try:
            return self._step()
        finally:
            active = self._active_topic()
            if active and active["run_id"]:
                self.engine.sync_candidate_records(active["run_id"])

    def _step(self):
        with self.store.tx() as db:
            db.execute("UPDATE candidates SET status='unknown',result=? WHERE status='starting' AND topic_id IS NOT NULL",
                       (json.dumps({"error": "worker stopped during BRAIN simulation POST; reconcile manually"}),))
        self.engine._reconcile_submission()
        submission = self.engine._submit_eligible()
        if submission:
            return submission
        candidates = self.store.all(
            "SELECT c.* FROM candidates c JOIN topics t ON t.id=c.topic_id "
            "WHERE t.status='active' AND c.status IN ('simulating','checking','local_wait','queued') "
            "ORDER BY c.created_at,c.id")
        progressed = self.engine._advance_ready_candidates(candidates)
        if progressed:
            return " | ".join(progressed)
        repaired = self._repair_one()
        if repaired:
            return repaired
        topic = self._active_topic()
        if not topic:
            topic, error = self._next_topic()
            if error:
                return error
            topic = self._activate_topic(topic)
        direction, error = self._directions(topic)
        if error:
            return error
        experiment = self.store.one("SELECT * FROM experiments WHERE direction_id=? AND status='active' ORDER BY ordinal DESC LIMIT 1",
                                    (direction["id"],))
        if experiment:
            return self._advance_experiment(topic, direction, experiment)
        if direction["iteration"] >= self.config.max_experiments_direction:
            with self.store.tx() as db:
                db.execute("UPDATE directions SET status='completed',conclusion=?,updated_at=? WHERE id=?",
                           ("实验次数达到配置上限", now(), direction["id"]))
            return f"direction budget reached: {direction['id']}"
        previous = self.store.one("SELECT f.decision FROM feedback f JOIN experiments e ON e.id=f.experiment_id "
                                  "WHERE e.direction_id=? ORDER BY f.created_at DESC,f.rowid DESC LIMIT 1", (direction["id"],))
        next_goal = direction["hypothesis"]
        plans_override = None
        if previous:
            decision = json.loads(previous["decision"])
            if decision.get("action") == "ITERATE":
                next_goal = decision.get("next_goal") or next_goal
                plans_override = decision.get("optimized_candidates") or None
        return self._create_experiment(topic, direction, next_goal, plans_override)
