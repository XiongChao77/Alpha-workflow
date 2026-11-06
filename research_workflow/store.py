"""SQLite journal and daily submission reservation."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import uuid


CST = timezone(timedelta(hours=8), "Asia/Shanghai")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def today() -> str:
    return datetime.now(CST).date().isoformat()


def tomorrow_utc() -> str:
    tomorrow = datetime.now(CST).date() + timedelta(days=1)
    return datetime.combine(tomorrow, datetime.min.time(), CST).astimezone(timezone.utc).isoformat()


def group_families(candidates):
    """Resolve every candidate to its root hypothesis via parent_candidate_id."""
    by_id = {candidate["id"]: candidate for candidate in candidates}
    roots = {}

    def root(candidate_id, seen=None):
        if candidate_id in roots:
            return roots[candidate_id]
        seen = seen or set()
        if candidate_id in seen:
            raise ValueError("Cycle in candidate lineage")
        seen.add(candidate_id)
        candidate = by_id[candidate_id]
        parent = candidate.get("parent_candidate_id")
        roots[candidate_id] = root(parent, seen) if parent in by_id else candidate_id
        return roots[candidate_id]

    families = {}
    for candidate in candidates:
        families.setdefault(root(candidate["id"]), []).append(candidate)
    return families


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.tx() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, objective TEXT NOT NULL, direction TEXT NOT NULL,
                    status TEXT NOT NULL, round INTEGER NOT NULL DEFAULT 0,
                    decision_due INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, round INTEGER NOT NULL,
                    hypothesis TEXT NOT NULL,
                    expression TEXT NOT NULL, expression_key TEXT NOT NULL,
                    settings TEXT NOT NULL, status TEXT NOT NULL,
                    simulation_url TEXT, alpha_id TEXT, result TEXT, not_before TEXT,
                    quantgpt_task_id TEXT, simulated_at TEXT,
                    parent_candidate_id TEXT, origin_action TEXT, parameter_values TEXT,
                    created_at TEXT NOT NULL, FOREIGN KEY(run_id) REFERENCES runs(id));
                CREATE INDEX IF NOT EXISTS ix_candidates_run ON candidates(run_id, status);
                CREATE UNIQUE INDEX IF NOT EXISTS ux_candidates_formula ON candidates(run_id, expression_key, settings);
                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, context TEXT NOT NULL,
                    decision TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS submissions (
                    account TEXT NOT NULL, day TEXT NOT NULL, candidate_id TEXT NOT NULL,
                    alpha_id TEXT NOT NULL, status TEXT NOT NULL, detail TEXT,
                    created_at TEXT NOT NULL, submitted_at TEXT, PRIMARY KEY(day));
                CREATE TABLE IF NOT EXISTS submission_uses (
                    component_id TEXT PRIMARY KEY, submitted_candidate_id TEXT NOT NULL,
                    day TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS combination_pairs (
                    id TEXT PRIMARY KEY, direction_id TEXT NOT NULL, experiment_id TEXT NOT NULL,
                    left_id TEXT NOT NULL, right_id TEXT NOT NULL, status TEXT NOT NULL,
                    created_at TEXT NOT NULL, evaluated_at TEXT,
                    UNIQUE(direction_id,left_id,right_id));
                CREATE TABLE IF NOT EXISTS combination_members (
                    candidate_id TEXT PRIMARY KEY, pair_id TEXT NOT NULL, weight_left REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS combination_attempts (
                    pair_id TEXT NOT NULL, parent_id TEXT NOT NULL, improved INTEGER NOT NULL,
                    parent_fitness REAL, best_fitness REAL, created_at TEXT NOT NULL,
                    PRIMARY KEY(pair_id,parent_id));
                CREATE TABLE IF NOT EXISTS cooldowns (
                    service TEXT PRIMARY KEY, not_before TEXT NOT NULL, reason TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS parameter_searches (
                    family_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, expression_template TEXT NOT NULL,
                    period_default INTEGER NOT NULL, parameters TEXT NOT NULL,
                    stage TEXT NOT NULL, stage_trials TEXT NOT NULL,
                    selected_period INTEGER, selected_params TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS topics (
                    id TEXT PRIMARY KEY, run_id TEXT UNIQUE, title TEXT NOT NULL, objective TEXT NOT NULL,
                    rationale TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL, status TEXT NOT NULL, summary TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS ix_topics_queue ON topics(status,priority,created_at);
                CREATE TABLE IF NOT EXISTS topic_discussions (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL, proposal TEXT,
                    topic_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS topic_messages (
                    id TEXT PRIMARY KEY, discussion_id TEXT NOT NULL, role TEXT NOT NULL,
                    content TEXT NOT NULL, proposal TEXT, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS ix_topic_messages_discussion ON topic_messages(discussion_id,created_at);
                CREATE TABLE IF NOT EXISTS directions (
                    id TEXT PRIMARY KEY, topic_id TEXT NOT NULL, title TEXT NOT NULL,
                    hypothesis TEXT NOT NULL, mechanism TEXT NOT NULL, status TEXT NOT NULL,
                    iteration INTEGER NOT NULL DEFAULT 0, conclusion TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS ix_directions_topic ON directions(topic_id,status);
                CREATE TABLE IF NOT EXISTS direction_generation_errors (
                    direction_id TEXT PRIMARY KEY, consecutive INTEGER NOT NULL,
                    last_reason TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY, direction_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
                    goal TEXT NOT NULL, stage TEXT NOT NULL, status TEXT NOT NULL,
                    plans TEXT NOT NULL, feedback_id TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS ix_experiments_direction ON experiments(direction_id,status);
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, stage TEXT NOT NULL,
                    statistics TEXT NOT NULL, decision TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS ix_feedback_experiment ON feedback(experiment_id,created_at);
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(candidates)")}
            if "parent_candidate_id" not in columns:
                db.execute("ALTER TABLE candidates ADD COLUMN parent_candidate_id TEXT")
            if "origin_action" not in columns:
                db.execute("ALTER TABLE candidates ADD COLUMN origin_action TEXT")
            if "parameter_values" not in columns:
                db.execute("ALTER TABLE candidates ADD COLUMN parameter_values TEXT")
            for name, ddl in (("topic_id", "TEXT"), ("direction_id", "TEXT"),
                              ("experiment_id", "TEXT"), ("variant_index", "INTEGER"),
                              ("repair_attempts", "INTEGER NOT NULL DEFAULT 0")):
                if name not in columns:
                    db.execute(f"ALTER TABLE candidates ADD COLUMN {name} {ddl}")
            submission_columns = {row["name"] for row in db.execute("PRAGMA table_info(submissions)")}
            if "submitted_at" not in submission_columns:
                db.execute("ALTER TABLE submissions ADD COLUMN submitted_at TEXT")
            for item in db.execute("SELECT candidate_id,alpha_id,day FROM submissions").fetchall():
                db.execute("INSERT OR IGNORE INTO submission_uses VALUES (?,?,?)",
                           (item["candidate_id"], item["candidate_id"], item["day"]))
                db.execute("INSERT OR IGNORE INTO submission_uses VALUES (?,?,?)",
                           ("alpha:" + item["alpha_id"], item["candidate_id"], item["day"]))
            # Older TUNE records predate explicit lineage; connect them to the
            # nearest earlier version of the same formula within their run.
            legacy_tunes = db.execute(
                "SELECT id,run_id,round,expression_key FROM candidates "
                "WHERE parent_candidate_id IS NULL AND origin_action IS NULL "
                "AND hypothesis LIKE '% [parameter tuning]'"
            ).fetchall()
            for child in legacy_tunes:
                parent = db.execute(
                    "SELECT id FROM candidates WHERE run_id=? AND expression_key=? AND round<? "
                    "ORDER BY round DESC,created_at DESC LIMIT 1",
                    (child["run_id"], child["expression_key"], child["round"]),
                ).fetchone()
                if parent:
                    db.execute("UPDATE candidates SET parent_candidate_id=?,origin_action='TUNE' WHERE id=?",
                               (parent["id"], child["id"]))

    @contextmanager
    def tx(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def create_run(self, objective: str, direction: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self.tx() as db:
            db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?)", (run_id, objective, direction, "running", 0, 0, now()))
        return run_id

    def latest_run(self) -> dict | None:
        return self.one("SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT 1")

    def get_or_create_run(self, objective: str, direction: str) -> str:
        """Resume the matching active research task or create it once."""
        with self.tx() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT id FROM runs WHERE status='running' AND objective=? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (objective,),
            ).fetchone()
            if row:
                return row["id"]
            run_id = uuid.uuid4().hex[:12]
            db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?)", (run_id, objective, direction, "running", 0, 0, now()))
            return run_id

    def one(self, sql: str, args: tuple = ()) -> dict | None:
        with self.tx() as db:
            row = db.execute(sql, args).fetchone()
        return dict(row) if row else None

    def all(self, sql: str, args: tuple = ()) -> list[dict]:
        with self.tx() as db:
            return [dict(row) for row in db.execute(sql, args).fetchall()]

    def add_candidate(self, run_id: str, round_number: int, hypothesis: str, expression: str, settings: dict,
                      parent_candidate_id: str | None = None, origin_action: str = "GENERATE",
                      parameter_values: dict | None = None, topic_id: str | None = None,
                      direction_id: str | None = None, experiment_id: str | None = None,
                      variant_index: int | None = None, repair_attempts: int = 0) -> str | None:
        candidate_id = uuid.uuid4().hex[:12]
        expression = expression.strip()
        key = "".join(expression.split()).lower()
        with self.tx() as db:
            result = db.execute("INSERT OR IGNORE INTO candidates (id,run_id,round,hypothesis,expression,expression_key,settings,status,parent_candidate_id,origin_action,parameter_values,topic_id,direction_id,experiment_id,variant_index,repair_attempts,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                (candidate_id, run_id, round_number, hypothesis, expression, key,
                                 json.dumps(settings, sort_keys=True), "queued", parent_candidate_id, origin_action,
                                 json.dumps(parameter_values, sort_keys=True) if parameter_values else None,
                                 topic_id, direction_id, experiment_id, variant_index, repair_attempts, now()))
            return candidate_id if result.rowcount else None

    def create_parameter_family(self, run_id, round_number, hypothesis, plan, trials, settings):
        """Create all Period variants and their durable search plan in one transaction."""
        family_id = uuid.uuid4().hex[:12]
        settings_json = json.dumps(settings, sort_keys=True)
        ids = []
        with self.tx() as db:
            for index, (values, expression) in enumerate(trials):
                candidate_id = family_id if index == 0 else uuid.uuid4().hex[:12]
                db.execute(
                    "INSERT INTO candidates (id,run_id,round,hypothesis,expression,expression_key,settings,status,"
                    "parent_candidate_id,origin_action,parameter_values,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (candidate_id, run_id, round_number, hypothesis, expression,
                     "".join(expression.split()).lower(), settings_json, "queued",
                     None if index == 0 else family_id, "GENERATE" if index == 0 else "PERIOD",
                     json.dumps(values, sort_keys=True), now()),
                )
                ids.append(candidate_id)
            db.execute("INSERT INTO parameter_searches VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (family_id, run_id, plan["expression_template"], plan["period"]["default"],
                        json.dumps(plan["parameters"], ensure_ascii=False), "PERIOD", json.dumps(ids),
                        None, json.dumps({}), now()))
        return family_id

    def enqueue_parameter_stage(self, family_id, run_id, round_number, hypothesis, trials, settings,
                                baseline_id, stage, selected_period, selected_params):
        """Queue one coordinate sweep and persist its candidate IDs atomically."""
        ids = [baseline_id]
        settings_json = json.dumps(settings, sort_keys=True)
        with self.tx() as db:
            for values, expression in trials:
                key = "".join(expression.split()).lower()
                existing = db.execute("SELECT id FROM candidates WHERE run_id=? AND expression_key=? AND settings=?",
                                      (run_id, key, settings_json)).fetchone()
                if existing:
                    ancestor = existing["id"]
                    while ancestor != family_id:
                        row = db.execute("SELECT parent_candidate_id FROM candidates WHERE id=?", (ancestor,)).fetchone()
                        if not row or not row["parent_candidate_id"]:
                            ancestor = None
                            break
                        ancestor = row["parent_candidate_id"]
                    if ancestor == family_id and existing["id"] not in ids:
                        ids.append(existing["id"])
                    continue
                candidate_id = uuid.uuid4().hex[:12]
                db.execute(
                    "INSERT INTO candidates (id,run_id,round,hypothesis,expression,expression_key,settings,status,"
                    "parent_candidate_id,origin_action,parameter_values,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (candidate_id, run_id, round_number, hypothesis, expression, key, settings_json, "queued",
                     baseline_id, stage, json.dumps(values, sort_keys=True), now()),
                )
                ids.append(candidate_id)
            db.execute("UPDATE parameter_searches SET stage=?,stage_trials=?,selected_period=?,"
                       "selected_params=?,updated_at=? WHERE family_id=?",
                       (stage, json.dumps(ids), selected_period, json.dumps(selected_params, sort_keys=True),
                        now(), family_id))
        return ids

    def update_candidate(self, candidate_id: str, **fields):
        if not fields:
            return
        if "settings" in fields:
            fields["settings"] = json.dumps(fields["settings"], sort_keys=True)
        if "result" in fields:
            fields["result"] = json.dumps(fields["result"], ensure_ascii=False)
        with self.tx() as db:
            db.execute("UPDATE candidates SET " + ", ".join(f"{key}=?" for key in fields) + " WHERE id=?",
                       (*fields.values(), candidate_id))

    def set_cooldown(self, service: str, seconds: int, reason: str):
        until = (datetime.now(timezone.utc) + timedelta(seconds=max(1, seconds))).isoformat()
        with self.tx() as db:
            db.execute("INSERT INTO cooldowns VALUES (?,?,?) ON CONFLICT(service) DO UPDATE SET not_before=excluded.not_before, reason=excluded.reason",
                       (service, until, reason))

    def cooling(self, service: str) -> bool:
        row = self.one("SELECT not_before FROM cooldowns WHERE service=?", (service,))
        return bool(row and row["not_before"] > now())

    def submission_today(self, account: str) -> dict | None:
        return self.one("SELECT * FROM submissions WHERE day=?", (today(),))

    @staticmethod
    def _components(db, candidate_id, seen=None):
        seen = seen or set()
        if candidate_id in seen:
            raise ValueError("Cycle in combination lineage")
        seen.add(candidate_id)
        member = db.execute("SELECT pair_id FROM combination_members WHERE candidate_id=?", (candidate_id,)).fetchone()
        if member:
            pair = db.execute("SELECT left_id,right_id FROM combination_pairs WHERE id=?", (member["pair_id"],)).fetchone()
            return sorted(set(Store._components(db, pair["left_id"], seen.copy()) +
                              Store._components(db, pair["right_id"], seen.copy())))
        parent = db.execute("SELECT parent_candidate_id FROM candidates WHERE id=?", (candidate_id,)).fetchone()
        if parent and parent["parent_candidate_id"]:
            inherited = Store._components(db, parent["parent_candidate_id"], seen)
            if len(inherited) > 1:
                return inherited
        return [candidate_id]

    def component_ids(self, candidate_id):
        with self.tx() as db:
            return self._components(db, candidate_id)

    @staticmethod
    def _component_keys(db, candidate_id):
        keys = []
        for component in Store._components(db, candidate_id):
            keys.append(component)
            row = db.execute("SELECT alpha_id FROM candidates WHERE id=?", (component,)).fetchone()
            if row and row["alpha_id"]:
                keys.append("alpha:" + row["alpha_id"])
        return list(dict.fromkeys(keys))

    def components_used(self, candidate_id):
        with self.tx() as db:
            return any(db.execute("SELECT 1 FROM submission_uses WHERE component_id=?", (item,)).fetchone()
                       for item in self._component_keys(db, candidate_id))

    def reserve_submission(self, account: str, candidate_id: str, alpha_id: str, day: str | None = None) -> bool:
        day = day or today()
        with self.tx() as db:
            db.execute("BEGIN IMMEDIATE")
            components = self._component_keys(db, candidate_id)
            if any(db.execute("SELECT 1 FROM submission_uses WHERE component_id=?", (item,)).fetchone()
                   for item in components):
                return False
            result = db.execute("INSERT OR IGNORE INTO submissions (account,day,candidate_id,alpha_id,status,detail,created_at) VALUES (?,?,?,?,?,?,?)",
                                (account, day, candidate_id, alpha_id, "reserved", None, now()))
            if result.rowcount:
                for item in components:
                    db.execute("INSERT INTO submission_uses VALUES (?,?,?)", (item, candidate_id, day))
            return result.rowcount == 1

    def mark_submission_sent(self, account: str, day: str):
        with self.tx() as db:
            db.execute("UPDATE submissions SET status='sent',submitted_at=? WHERE account=? AND day=?",
                       (now(), account, day))

    def update_submission(self, account: str, day: str, status: str, detail: str | None = None):
        with self.tx() as db:
            db.execute("UPDATE submissions SET status=?, detail=?, "
                       "submitted_at=CASE WHEN ?='active' THEN COALESCE(submitted_at,?) ELSE submitted_at END "
                       "WHERE account=? AND day=?", (status, detail, status, now(), account, day))
