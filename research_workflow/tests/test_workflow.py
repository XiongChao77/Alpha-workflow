import json
from dataclasses import replace
from pathlib import Path
import sqlite3
import tempfile
import unittest

from research_workflow.brain import is_checks_passed
from research_workflow.config import Config
from research_workflow.engine import Engine, stagnant_count
from research_workflow.library import eligible_counts
from research_workflow.store import Store, today
from research_workflow.transport import RemoteError, retry_seconds


PASS_IS = {"sharpe": 1.5, "fitness": 1.2, "turnover": 0.2,
           "checks": [{"name": "SHARPE", "result": "PASS"}, {"name": "FITNESS", "result": "PASS"}]}


class Generator:
    def generate(self, context):
        return [{"hypothesis": "High volume reversal", "expression": "-rank(close / ts_mean(close, 5))"}]


class BatchGenerator:
    def generate(self, context):
        return [{"hypothesis": f"Idea {period}", "expression": f"rank(ts_mean(close, {period}))"}
                for period in (5, 10, 20)]


class Judge:
    def decide(self, context):
        candidate = next(c for c in context["history"] if c["status"] == "passed")
        return {"action": "SUBMIT", "reason": "All reported IS checks passed", "target_id": candidate["id"],
                "next_direction": "test another reversal horizon",
                "parameter_changes": {"decay": 0, "truncation": 0.08, "neutralization": "SUBINDUSTRY"}}


class OptimizeJudge:
    def decide(self, context):
        target = next(c for c in context["history"] if c["status"] == "rejected")
        assert target["metrics"]["fitness"] == 0.36
        assert target["hypothesis"] == "High volume reversal"
        assert target["expression"] == "-rank(close / ts_mean(close, 5))"
        return {"action": "OPTIMIZE", "reason": "Lower turnover and improve fitness",
                "target_id": target["id"], "next_direction": "",
                "parameter_changes": {"decay": 0, "truncation": 0.08, "neutralization": "SUBINDUSTRY"},
                "optimized_hypothesis": "Smoothed reversal has less turnover",
                "optimized_expression": "-rank(ts_mean(returns, 5))"}


class FamilyJudge:
    def __init__(self):
        self.contexts = []

    def decide(self, context):
        self.contexts.append(context)
        return {"action": "NEW_DIRECTION", "reason": "Move to the next hypothesis family",
                "target_id": "", "next_direction": "new hypothesis direction",
                "parameter_changes": {"decay": 0, "truncation": 0.08, "neutralization": "SUBINDUSTRY"}}


class ParameterGenerator:
    def generate(self, context):
        return [{"hypothesis": "Parameterized volume signal",
                 "expression_template": "rank(ts_mean(close, {Period}) * {Scale} + {Offset})",
                 "period": {"name": "Period", "default": 45},
                 "parameters": [{"name": "Scale", "default": 1, "values": [1, 2]},
                                {"name": "Offset", "default": 0, "values": [0, 1]}]}]


class ParameterJudge:
    def __init__(self):
        self.contexts = []

    def decide_parameter(self, context):
        self.contexts.append(context)
        return {"explore": True, "reason": "Second dimension improved Sharpe"}


class SubmissionJudge:
    def choose_submission(self, context):
        return {"target_id": context["eligible"][0]["candidate_id"],
                "reason": "Best eligible candidate today"}


class Brain:
    def __init__(self):
        self.submits = 0
        self.starts = 0

    def start_simulation(self, expression, settings):
        self.starts += 1
        return "https://api.worldquantbrain.com/simulations/s1", 0

    def poll_simulation(self, url):
        return {"status": "DONE", "alpha": "/alphas/A123", "alpha_id": "A123", "is": PASS_IS}, 0

    def get_alpha(self, alpha_id):
        return {"status": "ACTIVE" if self.submits else "UNSUBMITTED", "is": PASS_IS}

    def submit(self, alpha_id):
        self.submits += 1
        return {"http_status": 201}


class RateLimitedBrain(Brain):
    def start_simulation(self, expression, settings):
        raise RemoteError(429, "CONCURRENT_SIMULATION_LIMIT", 120)


class InvalidExpressionBrain(Brain):
    def start_simulation(self, expression, settings):
        raise RemoteError(400, "invalid expression")


class FailedChecksBrain(Brain):
    def get_alpha(self, alpha_id):
        return {"status": "UNSUBMITTED", "is": {"sharpe": 1.43, "fitness": 0.36,
                "turnover": 1.2106, "checks": [
                    {"name": "LOW_FITNESS", "result": "FAIL", "value": 0.36, "limit": 1.0},
                    {"name": "HIGH_TURNOVER", "result": "FAIL", "value": 1.2106, "limit": 0.7}]}}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.config = replace(Config(), db_path=root / "state.sqlite3", library_dir=root / "library",
                              max_simulations_day=20, max_rounds=0, auto_submit=True)
        self.store = Store(self.config.db_path)
        self.run_id = self.store.create_run("test alpha mining", "volume reversal")

    def tearDown(self):
        self.temp.cleanup()

    def test_full_cycle_archives_and_submits_once(self):
        brain = Brain()
        engine = Engine(self.config, self.store, Generator(), Judge(), brain)
        self.assertIn("generated 1", engine.step(self.run_id))
        candidate = self.store.one("SELECT * FROM candidates WHERE run_id=?", (self.run_id,))
        record_json = self.config.library_dir / "candidates" / f"{candidate['id']}.json"
        created = json.loads(record_json.read_text(encoding="utf-8"))
        self.assertEqual(created["hypothesis"], "High volume reversal")
        self.assertEqual(created["expression"], "-rank(close / ts_mean(close, 5))")
        self.assertEqual(created["status"], "queued")
        self.assertIn("simulation started", engine.step(self.run_id))
        self.assertIn("simulation complete", engine.step(self.run_id))
        self.assertIn("checks passed", engine.step(self.run_id))
        record = self.config.library_dir / "wq_A123.md"
        self.assertTrue(record.exists())
        self.assertIn("High volume reversal", record.read_text(encoding="utf-8"))
        self.assertIn("submitted one alpha", engine.step(self.run_id))
        self.assertEqual(brain.submits, 1)
        candidate = self.store.one("SELECT * FROM candidates WHERE alpha_id='A123'")
        self.assertIn("already has a submission record", engine._submit(candidate, {"reason": "retry"}))
        self.assertEqual(brain.submits, 1)
        engine.step(self.run_id)
        self.assertEqual(self.store.submission_today(self.config.account)["status"], "active")
        exported = json.loads(record_json.read_text(encoding="utf-8"))
        self.assertEqual(exported["worldquant"]["metrics"]["fitness"], 1.2)
        self.assertEqual(exported["worldquant"]["checks"][0]["result"], "PASS")
        self.assertEqual(exported["submission"]["status"], "active")
        counts = eligible_counts(self.store)
        self.assertEqual(counts["eligible_total"], 1)
        self.assertEqual(counts["submitted_total"], 1)
        self.assertEqual(counts["submitted_today"], 1)
        index = json.loads((self.config.library_dir / "eligible_alphas.json").read_text(encoding="utf-8"))
        self.assertTrue(index["alphas"][0]["submitted"])
        self.assertIn("平台 ACTIVE", record.read_text(encoding="utf-8"))
        self.assertIn("All reported IS checks passed", record.read_text(encoding="utf-8"))

    def test_failed_checks_are_exported_as_structured_json(self):
        engine = Engine(self.config, self.store, Generator(), Judge(), FailedChecksBrain())
        for _ in range(4):
            engine.step(self.run_id)
        candidate = self.store.one("SELECT * FROM candidates WHERE run_id=?", (self.run_id,))
        record = json.loads((self.config.library_dir / "candidates" / f"{candidate['id']}.json")
                            .read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "rejected")
        self.assertEqual(record["worldquant"]["metrics"]["fitness"], 0.36)
        self.assertEqual([c["name"] for c in record["worldquant"]["checks"]],
                         ["LOW_FITNESS", "HIGH_TURNOVER"])
        self.assertFalse((self.config.library_dir / "wq_A123.md").exists())

    def test_optimize_creates_backtest_candidate_with_parent_lineage(self):
        engine = Engine(self.config, self.store, Generator(), OptimizeJudge(), FailedChecksBrain())
        for _ in range(4):
            engine.step(self.run_id)
        original = self.store.one("SELECT * FROM candidates WHERE status='rejected'")
        self.assertIn("model chose OPTIMIZE", engine.step(self.run_id))
        optimized = self.store.one("SELECT * FROM candidates WHERE origin_action='OPTIMIZE'")
        self.assertEqual(optimized["parent_candidate_id"], original["id"])
        self.assertEqual(optimized["hypothesis"], "Smoothed reversal has less turnover")
        self.assertEqual(optimized["expression"], "-rank(ts_mean(returns, 5))")
        exported = json.loads((self.config.library_dir / "candidates" / f"{optimized['id']}.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(exported["parent_candidate_id"], original["id"])
        self.assertIn("simulation started", engine.step(self.run_id))

    def test_existing_database_adds_candidate_lineage_columns(self):
        legacy_path = Path(self.temp.name) / "legacy.sqlite3"
        with sqlite3.connect(legacy_path) as db:
            db.execute("CREATE TABLE candidates (id TEXT PRIMARY KEY, run_id TEXT, round INTEGER, "
                       "hypothesis TEXT, expression TEXT, expression_key TEXT, settings TEXT, "
                       "status TEXT, simulation_url TEXT, alpha_id TEXT, result TEXT, "
                       "not_before TEXT, quantgpt_task_id TEXT, simulated_at TEXT, created_at TEXT)")
            for candidate_id, round_number, hypothesis in (
                ("root", 1, "Original idea"), ("child", 2, "Original idea [parameter tuning]")
            ):
                db.execute("INSERT INTO candidates (id,run_id,round,hypothesis,expression,expression_key,"
                           "settings,status,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                           (candidate_id, "legacy-run", round_number, hypothesis, "rank(close)",
                            "rank(close)", json.dumps({"decay": round_number}), "rejected",
                            f"2026-01-01T00:00:0{round_number}+00:00"))
        Store(legacy_path)
        with sqlite3.connect(legacy_path) as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(candidates)")}
            parent, origin = db.execute("SELECT parent_candidate_id,origin_action FROM candidates WHERE id='child'").fetchone()
        self.assertIn("parent_candidate_id", columns)
        self.assertIn("origin_action", columns)
        self.assertEqual((parent, origin), ("root", "TUNE"))

    def test_decision_context_contains_only_one_hypothesis_family(self):
        first = self.store.add_candidate(self.run_id, 1, "First family", "rank(close)", self.config.settings())
        second = self.store.add_candidate(self.run_id, 1, "Second family", "rank(volume)", self.config.settings())
        for candidate_id in (first, second):
            self.store.update_candidate(candidate_id, status="rejected", result={"is": {
                "fitness": 0.3, "checks": [{"name": "LOW_FITNESS", "result": "FAIL", "value": 0.3, "limit": 1.0}]}})
        judge = FamilyJudge()
        engine = Engine(self.config, self.store, Generator(), judge, Brain())
        engine.step(self.run_id)
        engine.step(self.run_id)
        self.assertEqual([context["family_id"] for context in judge.contexts], [first, second])
        self.assertEqual([[candidate["id"] for candidate in context["history"]] for context in judge.contexts],
                         [[first], [second]])

    def test_family_stops_after_five_consecutive_nonimprovements(self):
        root = self.store.add_candidate(self.run_id, 1, "Stagnant family", "rank(close)", self.config.settings())
        previous = root
        result = {"is": {"fitness": 0.3, "checks": [
            {"name": "LOW_FITNESS", "result": "FAIL", "value": 0.3, "limit": 1.0}]}}
        self.store.update_candidate(root, status="rejected", result=result)
        for index in range(5):
            child = self.store.add_candidate(self.run_id, index + 2, f"Attempt {index + 1}",
                                             f"rank(ts_mean(close, {index + 2}))", self.config.settings(),
                                             previous, "OPTIMIZE")
            self.store.update_candidate(child, status="rejected", result=result)
            previous = child
        other = self.store.add_candidate(self.run_id, 1, "Other family", "rank(volume)", self.config.settings())
        self.store.update_candidate(other, status="rejected", result=result)
        judge = FamilyJudge()
        engine = Engine(self.config, self.store, Generator(), judge, Brain())
        engine.step(self.run_id)
        self.assertEqual(judge.contexts[0]["family_id"], other)
        decisions = [json.loads(row["decision"]) for row in self.store.all("SELECT decision FROM decisions")]
        self.assertEqual(sum(decision["action"] == "STOP_FAMILY" for decision in decisions), 1)
        exported = json.loads((self.config.library_dir / "candidates" / f"{previous}.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(exported["family_id"], root)

    def test_family_improvement_resets_stagnation_count(self):
        def candidate(value):
            return {"status": "rejected", "result": json.dumps({"is": {"checks": [
                {"name": "LOW_FITNESS", "result": "FAIL", "value": value, "limit": 1.0}]}})}
        family = [candidate(0.3), candidate(0.2), candidate(0.4), candidate(0.35)]
        self.assertEqual(stagnant_count(family), 1)

    def test_parameter_grid_best_is_baseline_for_later_improvement(self):
        def candidate(value, origin):
            return {"status": "rejected", "origin_action": origin,
                    "result": json.dumps({"is": {"checks": [
                        {"name": "LOW_FITNESS", "result": "FAIL", "value": value, "limit": 1}]}})}
        family = [candidate(0.2, "GENERATE"), candidate(0.6, "PERIOD"),
                  candidate(0.4, "OPTIMIZE")]
        self.assertEqual(stagnant_count(family), 1)

    def test_daily_submission_keeps_all_eligible_in_index(self):
        for index in range(2):
            candidate_id = self.store.add_candidate(self.run_id, 1, f"Idea {index}",
                                                    f"rank(ts_mean(close, {index + 2}))", self.config.settings())
            self.store.update_candidate(candidate_id, status="passed", alpha_id=f"A{index}",
                                        result={"is": PASS_IS})
        brain = Brain()
        engine = Engine(self.config, self.store, Generator(), SubmissionJudge(), brain)
        self.assertIn("submitted one alpha", engine.step(self.run_id))
        self.assertEqual(brain.submits, 1)
        self.assertIsNone(engine._submit_eligible())
        index = json.loads((self.config.library_dir / "eligible_alphas.json").read_text(encoding="utf-8"))
        self.assertEqual(index["eligible_total"], 2)
        self.assertEqual(index["submitted_total"], 1)
        self.assertEqual(index["submitted_today"], 1)
        self.assertEqual(sum(record["submitted"] for record in index["alphas"]), 1)

    def test_default_three_simulation_slots_fill_in_one_step(self):
        brain = Brain()
        engine = Engine(self.config, self.store, BatchGenerator(), FamilyJudge(), brain)
        self.assertIn("generated 3", engine.step(self.run_id))
        result = engine.step(self.run_id)
        self.assertEqual(result.count("BRAIN simulation started"), 3)
        self.assertEqual(brain.starts, 3)
        self.assertEqual(self.store.one("SELECT COUNT(*) AS n FROM candidates WHERE status='simulating'")["n"], 3)

    def test_completed_simulation_frees_slot_for_queued_candidate(self):
        config = replace(self.config, max_concurrent_simulations=2)
        brain = Brain()
        engine = Engine(config, self.store, BatchGenerator(), FamilyJudge(), brain)
        engine.step(self.run_id)
        engine.step(self.run_id)
        self.assertEqual(brain.starts, 2)
        result = engine.step(self.run_id)
        self.assertIn("BRAIN simulation started", result)
        self.assertEqual(brain.starts, 3)
        self.assertEqual(engine._inflight_simulations(), 1)

    def test_configured_slot_limit_counts_other_runs(self):
        other_run = self.store.create_run("another objective", "another direction")
        other_id = self.store.add_candidate(other_run, 1, "old", "rank(open)", self.config.settings())
        self.store.update_candidate(other_id, status="simulating", simulation_url="https://api.worldquantbrain.com/simulations/other")
        config = replace(self.config, max_concurrent_simulations=2)
        brain = Brain()
        engine = Engine(config, self.store, BatchGenerator(), FamilyJudge(), brain)
        engine.step(self.run_id)
        result = engine.step(self.run_id)
        self.assertEqual(result.count("BRAIN simulation started"), 1)
        self.assertEqual(brain.starts, 1)
        self.assertEqual(engine._inflight_simulations(), 2)

    def test_invalid_simulation_slot_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ALPHA_MAX_CONCURRENT_SIMULATIONS"):
            Engine(replace(self.config, max_concurrent_simulations=0), self.store,
                   Generator(), Judge(), Brain())

    def test_submission_counts_are_independent_of_current_eligibility(self):
        self.assertTrue(self.store.reserve_submission("primary", "old-candidate", "OLD-ALPHA"))
        self.store.mark_submission_sent("primary", today())
        self.assertEqual(eligible_counts(self.store),
                         {"eligible_total": 0, "submitted_total": 1, "submitted_today": 1})

    def test_submission_waits_for_parameter_search_to_finish(self):
        brain = Brain()
        engine = Engine(self.config, self.store, ParameterGenerator(), SubmissionJudge(), brain)
        engine.step(self.run_id)
        first = self.store.one("SELECT * FROM candidates WHERE origin_action='GENERATE'")
        self.store.update_candidate(first["id"], status="passed", alpha_id="A123", result={"is": PASS_IS})
        self.assertIsNone(engine._submit_eligible())
        self.assertEqual(brain.submits, 0)

    def test_parameter_search_period_then_two_dimensions(self):
        judge = ParameterJudge()
        engine = Engine(self.config, self.store, ParameterGenerator(), judge, Brain())
        self.assertIn("generated 4", engine.step(self.run_id))
        search = self.store.one("SELECT * FROM parameter_searches")
        period_candidates = self.store.all("SELECT * FROM candidates ORDER BY created_at,id")
        self.assertEqual({json.loads(c["parameter_values"])["Period"] for c in period_candidates},
                         {30, 60, 90, 45})
        root_record = json.loads((self.config.library_dir / "candidates" / f"{search['family_id']}.json")
                                 .read_text(encoding="utf-8"))
        self.assertEqual(root_record["parameter_search"]["period_default"], 45)
        self.assertEqual(root_record["parameter_search"]["parameters"][0]["name"], "Scale")
        for candidate in period_candidates:
            period = json.loads(candidate["parameter_values"])["Period"]
            score = 1.2 if period == 60 else 0.8
            self.store.update_candidate(candidate["id"], status="rejected", result={"is": {
                "sharpe": score, "checks": [{"name": "LOW_FITNESS", "result": "FAIL", "value": 0.3, "limit": 1}]}})
        self.assertIn("PARAM1 search queued", engine.step(self.run_id))
        search = self.store.one("SELECT * FROM parameter_searches")
        self.assertEqual(search["stage"], "PARAM1")
        self.assertEqual(search["selected_period"], 60)
        for candidate in self.store.all("SELECT * FROM candidates WHERE status='queued'"):
            values = json.loads(candidate["parameter_values"])
            self.assertEqual(values["Period"], 60)
            self.store.update_candidate(candidate["id"], status="rejected", result={"is": {
                "sharpe": 1.4, "checks": [{"name": "LOW_FITNESS", "result": "FAIL", "value": 0.3, "limit": 1}]}})
        self.assertIn("PARAM1 search completed", engine.step(self.run_id))
        self.assertEqual(self.store.one("SELECT stage FROM parameter_searches")["stage"], "PARAM2_DECISION")
        self.assertIn("PARAM2 search queued", engine.step(self.run_id))
        self.assertEqual(len(judge.contexts), 1)
        self.assertEqual(judge.contexts[0]["selected_values"]["Scale"], 2)
        for candidate in self.store.all("SELECT * FROM candidates WHERE status='queued'"):
            values = json.loads(candidate["parameter_values"])
            self.assertEqual((values["Period"], values["Scale"]), (60, 2))
            self.store.update_candidate(candidate["id"], status="rejected", result={"is": {
                "sharpe": 1.5, "checks": [{"name": "LOW_FITNESS", "result": "FAIL", "value": 0.3, "limit": 1}]}})
        self.assertIn("PARAM2 search completed", engine.step(self.run_id))
        self.assertEqual(self.store.one("SELECT stage FROM parameter_searches")["stage"], "READY")

    def test_period_gate_abandons_family_when_no_sharpe_above_one(self):
        engine = Engine(self.config, self.store, ParameterGenerator(), ParameterJudge(), Brain())
        engine.step(self.run_id)
        for candidate in self.store.all("SELECT * FROM candidates"):
            self.store.update_candidate(candidate["id"], status="rejected", result={"is": {
                "sharpe": 1.0, "checks": [{"name": "LOW_SHARPE", "result": "FAIL", "value": 1, "limit": 1.25}]}})
        self.assertIn("Period gate failed", engine.step(self.run_id))
        self.assertEqual(self.store.one("SELECT stage FROM parameter_searches")["stage"], "ABANDONED")

    def test_existing_candidate_is_backfilled_on_next_step(self):
        candidate_id = self.store.add_candidate(self.run_id, 1, "older idea", "rank(volume)", self.config.settings())
        self.store.update_candidate(candidate_id, status="rejected", alpha_id="OLD123",
                                    result={"is": {"fitness": 0.2, "checks": [
                                        {"name": "LOW_FITNESS", "result": "FAIL", "value": 0.2, "limit": 1.0}]}})
        Engine(self.config, self.store, Generator(), FamilyJudge(), Brain()).step(self.run_id)
        record = json.loads((self.config.library_dir / "candidates" / f"{candidate_id}.json")
                            .read_text(encoding="utf-8"))
        self.assertEqual(record["hypothesis"], "older idea")
        self.assertEqual(record["worldquant"]["metrics"]["fitness"], 0.2)

    def test_rate_limit_keeps_candidate_and_persists_cooldown(self):
        engine = Engine(self.config, self.store, Generator(), Judge(), RateLimitedBrain())
        engine.step(self.run_id)
        self.assertIn("cooldown", engine.step(self.run_id))
        candidate = self.store.one("SELECT * FROM candidates WHERE run_id=?", (self.run_id,))
        self.assertEqual(candidate["status"], "queued")
        self.assertTrue(self.store.cooling("brain"))

    def test_definite_bad_expression_is_recorded_as_failed(self):
        engine = Engine(self.config, self.store, Generator(), Judge(), InvalidExpressionBrain())
        engine.step(self.run_id)
        self.assertIn("rejected expression", engine.step(self.run_id))
        candidate = self.store.one("SELECT * FROM candidates WHERE run_id=?", (self.run_id,))
        self.assertEqual(candidate["status"], "failed")

    def test_checks_require_nonempty_all_pass(self):
        self.assertFalse(is_checks_passed({}))
        self.assertFalse(is_checks_passed({"checks": [{"result": "PASS"}, {"result": "PENDING"}]}))
        self.assertTrue(is_checks_passed(PASS_IS))

    def test_retry_after_header(self):
        self.assertEqual(retry_seconds("17"), 17)
        self.assertEqual(retry_seconds("bad", 45), 45)

    def test_daily_submission_slot_is_global_to_database(self):
        self.assertTrue(self.store.reserve_submission("primary", "c1", "A1"))
        self.assertFalse(self.store.reserve_submission("other", "c2", "A2"))

    def test_automatic_start_resumes_matching_active_run(self):
        self.assertEqual(self.store.get_or_create_run("test alpha mining", "different starter direction"), self.run_id)
        other = self.store.get_or_create_run("different target", "price volume")
        self.assertNotEqual(other, self.run_id)

    def test_restart_does_not_repeat_unknown_simulation_post(self):
        candidate_id = self.store.add_candidate(self.run_id, 1, "idea", "rank(close)", self.config.settings())
        self.store.update_candidate(candidate_id, status="starting")
        engine = Engine(self.config, self.store, Generator(), Judge(), Brain())
        engine.step(self.run_id)
        candidate = self.store.one("SELECT * FROM candidates WHERE id=?", (candidate_id,))
        self.assertEqual(candidate["status"], "unknown")
        self.assertEqual(engine.brain.starts, 0)

    def test_prior_day_submission_is_reconciled(self):
        candidate_id = self.store.add_candidate(self.run_id, 1, "old idea", "rank(close)", self.config.settings())
        self.store.update_candidate(candidate_id, status="passed", alpha_id="A123", result={"is": PASS_IS})
        self.assertTrue(self.store.reserve_submission("primary", candidate_id, "A123", "2020-01-01"))
        brain = Brain()
        brain.submits = 1
        engine = Engine(self.config, self.store, Generator(), Judge(), brain)
        engine.step(self.run_id)
        row = self.store.one("SELECT * FROM submissions WHERE day='2020-01-01'")
        self.assertEqual(row["status"], "active")


if __name__ == "__main__":
    unittest.main()
