import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from research_workflow.research import ResearchAgent, AlphaAgent, ResearchWorkflow, batch_statistics
from research_workflow.store import Store


class FakeConfig:
    batch_size = 2
    max_experiment_candidates = 24
    max_directions_topic = 2
    max_experiments_direction = 2
    max_repair_attempts = 2
    max_invalid_generations = 3
    family_max_stagnant = 5
    sharpe_gate = 1.2
    max_combination_pairs = 0
    combination_weights = (0.25, 0.5, 0.75)
    max_simulations_day = 20
    period_grid = (30, 60, 90)

    def settings(self):
        return {"region": "USA", "delay": 1}

    def research_objective(self):
        return "Explore reversal"

    def research_direction(self):
        return "Volume reversal"


class FakeModel:
    def __init__(self):
        self.feedback_calls = []
        self.variants_calls = 0
        self.repair_contexts = []

    def generate_topic(self, context):
        return {"title": "Autonomous topic", "objective": "Explore reversal", "rationale": "market mechanism"}

    def propose_directions(self, context):
        return [{"title": "Volume conditioned reversal", "hypothesis": "Illiquid selling reverts",
                 "mechanism": "Temporary price impact"}]

    def generate_variants(self, context):
        self.variants_calls += 1
        return [{"hypothesis": "Temporary impact reverts", "expression_template":
                 "rank(close / ts_mean(close,{Period})) * {Scale}",
                 "period": {"name": "Period", "default": 45},
                 "parameters": [{"name": "Scale", "default": 1, "values": [1, 2]}]}]

    def analyze_batch(self, context):
        self.feedback_calls.append(context)
        return {"action": "SWITCH_DIRECTION", "reason": "Weak distribution",
                "next_goal": "", "conclusion": "No stable signal"}

    def repair_expression(self, context):
        self.repair_contexts.append(context)
        return {"hypothesis": context["hypothesis"], "expression": "rank(close / ts_mean(close,30))",
                "reason": "Remove incompatible term"}


class FakeEngine:
    def _reconcile_submission(self):
        return None

    def _submit_eligible(self):
        return None

    def _advance_ready_candidates(self, candidates):
        return []

    def _today_simulations(self):
        return 0

    def sync_candidate_records(self, run_id):
        return None

    def _throttle(self, service, exc):
        return "cooldown"


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "workflow.db")
        self.model = FakeModel()
        self.workflow = ResearchWorkflow(FakeConfig(), self.store, ResearchAgent(self.model),
                                         AlphaAgent(self.model), FakeEngine())

    def test_new_plans_search_model_parameters_without_fixed_grid(self):
        class ChosenParametersModel(FakeModel):
            def generate_variants(self, context):
                return [{"hypothesis": "Short horizon reversal",
                         "expression_template": "rank(ts_mean(close,{Window})/close)*{Weight}",
                         "parameters": [{"name": "Window", "default": 12, "values": [8, 12, 24]},
                                        {"name": "Weight", "default": 1, "values": [0.5, 1, 2]}]}]

            def analyze_batch(self, context):
                self.feedback_calls.append(context)
                if context["can_explore_param2"]:
                    return {"action": "EXPLORE_PARAM2", "reason": "Test weight", "next_goal": "",
                            "conclusion": "", "optimized_candidates": []}
                return {"action": "SWITCH_DIRECTION", "reason": "Done", "next_goal": "",
                        "conclusion": "Done", "optimized_candidates": []}

        self.model = ChosenParametersModel()
        self.workflow = ResearchWorkflow(FakeConfig(), self.store, ResearchAgent(self.model),
                                         AlphaAgent(self.model), FakeEngine())
        self.workflow.add_topic("Model chosen parameters")
        self.assertIn("experiment queued", self.workflow.step())
        experiment = self.store.one("SELECT * FROM experiments LIMIT 1")
        self.assertEqual(experiment["stage"], "PARAM1")
        rows = self.store.all("SELECT * FROM candidates WHERE experiment_id=?", (experiment["id"],))
        self.assertEqual({json.loads(row["parameter_values"])["Window"] for row in rows}, {8, 12, 24})
        self.assertTrue(all(json.loads(row["parameter_values"])["Weight"] == 1 for row in rows))
        self.assertTrue(all("Period" not in row["expression"] for row in rows))
        for row in rows:
            window = json.loads(row["parameter_values"])["Window"]
            self.store.update_candidate(row["id"], status="rejected",
                                        result={"is": {"sharpe": 1.4 if window == 12 else 0.6}})
        self.assertIn("PARAM2 queued", self.workflow.step())
        second = self.store.all("SELECT * FROM candidates WHERE experiment_id=? AND origin_action='PARAM2'",
                                (experiment["id"],))
        self.assertEqual({json.loads(row["parameter_values"])["Weight"] for row in second}, {0.5, 2})
        self.assertTrue(all(json.loads(row["parameter_values"])["Window"] == 12 for row in second))

    def test_topic_queue_batch_parameter_gate_and_feedback(self):
        topic_id = self.workflow.add_topic("Researcher topic", priority=3)
        self.assertIn("experiment queued", self.workflow.step())
        topic = self.store.one("SELECT * FROM topics WHERE id=?", (topic_id,))
        self.assertEqual(topic["status"], "active")
        experiment = self.store.one("SELECT * FROM experiments LIMIT 1")
        rows = self.store.all("SELECT * FROM candidates WHERE experiment_id=?", (experiment["id"],))
        self.assertEqual({json.loads(row["parameter_values"])["Period"] for row in rows}, {30, 45, 60, 90})
        for row in rows:
            period = json.loads(row["parameter_values"])["Period"]
            self.store.update_candidate(row["id"], status="rejected", result={"is": {"sharpe": 1.3 if period == 45 else 0.8,
                                                                                        "checks": [{"name": "LOW_FITNESS", "result": "FAIL"}]}})
        self.assertIn("PARAM1 queued", self.workflow.step())
        param_rows = self.store.all("SELECT * FROM candidates WHERE experiment_id=? AND origin_action='PARAM1'", (experiment["id"],))
        self.assertEqual(len(param_rows), 1)
        self.assertEqual(json.loads(param_rows[0]["parameter_values"])["Period"], 45)
        self.store.update_candidate(param_rows[0]["id"], status="rejected", result={"is": {"sharpe": 0.7}})
        self.assertIn("direction completed", self.workflow.step())
        self.assertEqual(len(self.model.feedback_calls), 1)
        stats = self.model.feedback_calls[0]["statistics"]
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["correlation_evidence"], "unavailable; BRAIN supplied no correlation values")
        self.assertIn("Topic completed", self.workflow.step())

    def test_autonomous_topic_and_limited_formula_repair(self):
        self.assertIn("experiment queued", self.workflow.step())
        self.assertEqual(self.store.one("SELECT source FROM topics")["source"], "agent")
        row = self.store.one("SELECT * FROM candidates ORDER BY created_at,id LIMIT 1")
        self.store.update_candidate(row["id"], status="failed", result={"error": "Incompatible unit for add", "http_status": 400})
        self.assertIn("formula repaired", self.workflow.step())
        child = self.store.one("SELECT * FROM candidates WHERE parent_candidate_id=?", (row["id"],))
        self.assertEqual(child["repair_attempts"], 1)
        self.assertEqual(child["experiment_id"], row["experiment_id"])

    def test_brain_error_is_sent_verbatim_to_alpha_repair(self):
        self.workflow.step()
        row = self.store.one("SELECT * FROM candidates ORDER BY created_at,id LIMIT 1")
        message = "Expression with 113 operators exceeds limit of 64."
        self.store.update_candidate(row["id"], status="failed", result={
            "status": "ERROR", "message": message})
        self.assertIn("formula repaired", self.workflow.step())
        self.assertEqual(self.model.repair_contexts[0]["error"], message)
        self.assertNotIn("operator_budget", self.model.repair_contexts[0])
        self.assertNotIn("allowed_fields", self.model.repair_contexts[0])

    def test_model_can_decline_unrepairable_formula_without_retry_loop(self):
        class CannotRepair(FakeModel):
            def repair_expression(self, context):
                self.repair_contexts.append(context)
                return {"hypothesis": context["hypothesis"], "expression": None,
                        "reason": "No viable correction for this platform error"}

        self.model = CannotRepair()
        self.workflow = ResearchWorkflow(FakeConfig(), self.store, ResearchAgent(self.model),
                                         AlphaAgent(self.model), FakeEngine())
        self.workflow.step()
        row = self.store.one("SELECT * FROM candidates ORDER BY created_at,id LIMIT 1")
        self.store.update_candidate(row["id"], status="failed", result={"message": "Unknown operator"})
        self.assertIn("formula unrepaired", self.workflow.step())
        self.assertEqual(self.store.one("SELECT repair_attempts FROM candidates WHERE id=?", (row["id"],))["repair_attempts"],
                         FakeConfig.max_repair_attempts)
        self.assertNotIn("allowed_fields", self.model.repair_contexts[0])
        self.assertEqual(len(self.model.repair_contexts), 1)

    def test_existing_running_run_is_adopted_with_its_candidates(self):
        run_id = self.store.create_run("Old objective", "Old direction")
        candidate_id = self.store.add_candidate(run_id, 1, "Old hypothesis", "rank(close)", {"region": "USA"})
        self.assertIn("experiment awaiting batch results", self.workflow.step())
        topic = self.store.one("SELECT * FROM topics")
        candidate = self.store.one("SELECT * FROM candidates WHERE id=?", (candidate_id,))
        self.assertEqual((topic["source"], topic["run_id"]), ("legacy", run_id))
        self.assertEqual(candidate["topic_id"], topic["id"])
        self.assertIsNotNone(candidate["experiment_id"])

    def test_adopted_legacy_negative_candidate_is_inverted(self):
        run_id = self.store.create_run("Old objective", "Old direction")
        candidate_id = self.store.add_candidate(run_id, 1, "Old hypothesis", "rank(close)", {"region": "USA"})
        self.workflow.step()
        self.store.update_candidate(candidate_id, status="rejected", result={"is": {"sharpe": -1.4}})
        self.assertIn("inverse verification queued", self.workflow.step())
        inverse = self.store.one("SELECT * FROM candidates WHERE parent_candidate_id=? AND origin_action='INVERT_LEGACY'",
                                 (candidate_id,))
        self.assertEqual(inverse["expression"], "-(rank(close))")

    def test_discussion_is_persistent_and_submission_is_idempotent(self):
        discussion_id = self.workflow.start_discussion()
        self.workflow.append_discussion(discussion_id, "researcher", "Explore low volume reversal")
        proposal = {"title": "Low volume reversal", "objective": "Find stable reversal",
                    "rationale": "Temporary impact"}
        self.workflow.append_discussion(discussion_id, "agent", "Test conditional reversal", proposal)
        resumed = ResearchWorkflow(FakeConfig(), self.store, ResearchAgent(self.model),
                                   AlphaAgent(self.model), FakeEngine())
        session, history = resumed.discussion(discussion_id)
        self.assertEqual((session["status"], len(history)), ("open", 2))
        topic_id = resumed.submit_discussion(discussion_id)
        self.assertEqual(resumed.submit_discussion(discussion_id), topic_id)
        topic = self.store.one("SELECT * FROM topics WHERE id=?", (topic_id,))
        self.assertEqual((topic["source"], topic["priority"], topic["status"]),
                         ("researcher", 100, "pending"))
        self.assertEqual(self.store.one("SELECT COUNT(*) AS n FROM topics")["n"], 1)

    def test_new_human_topic_waits_until_current_topic_finishes(self):
        current_id = self.workflow.add_topic("Current", priority=0)
        self.workflow.step()
        next_id = self.workflow.add_topic("Next", priority=100)
        self.assertEqual(self.store.one("SELECT status FROM topics WHERE id=?", (current_id,))["status"], "active")
        self.assertEqual(self.store.one("SELECT status FROM topics WHERE id=?", (next_id,))["status"], "pending")
        self.workflow.step()
        self.assertEqual(self.store.one("SELECT status FROM topics WHERE id=?", (next_id,))["status"], "pending")

    def test_batch_statistics_distinguishes_distribution_from_correlation(self):
        rows = [{"id": "a", "status": "passed", "result": json.dumps({"is": {"sharpe": 1.5, "fitness": 1.2,
                 "checks": [{"name": "SELF_CORRELATION", "result": "PENDING", "value": None}]}}),
                 "parameter_values": '{"Period":30}', "variant_index": 0, "origin_action": "PERIOD",
                 "hypothesis": "h", "expression": "x", "alpha_id": "A"}]
        result = batch_statistics(rows)
        self.assertEqual(result["metrics"]["sharpe"]["median"], 1.5)
        self.assertEqual(result["passed_rate"], 1)
        self.assertIn("unavailable", result["correlation_evidence"])

    def test_negative_sharpe_passes_period_gate_and_selects_largest_magnitude(self):
        self.workflow.add_topic("Negative signal")
        self.workflow.step()
        experiment = self.store.one("SELECT * FROM experiments")
        for row in self.store.all("SELECT * FROM candidates WHERE experiment_id=?", (experiment["id"],)):
            period = json.loads(row["parameter_values"])["Period"]
            score = -1.4 if period == 60 else (0.8 if period == 30 else 0.5)
            self.store.update_candidate(row["id"], status="rejected", result={"is": {"sharpe": score}})
        self.assertIn("inverse verification queued", self.workflow.step())
        inverse = self.store.one("SELECT * FROM candidates WHERE experiment_id=? AND origin_action='INVERT_PERIOD'", (experiment["id"],))
        self.assertEqual(inverse["parent_candidate_id"], self.store.one(
            "SELECT id FROM candidates WHERE experiment_id=? AND parameter_values LIKE '%\"Period\": 60%' AND origin_action='PERIOD'",
            (experiment["id"],))["id"])
        self.assertTrue(inverse["expression"].startswith("-("))
        self.store.update_candidate(inverse["id"], status="rejected", result={"is": {"sharpe": 1.35}})
        self.assertIn("PARAM1 queued", self.workflow.step())
        param = self.store.one("SELECT * FROM candidates WHERE experiment_id=? AND origin_action='PARAM1'", (experiment["id"],))
        self.assertEqual(json.loads(param["parameter_values"])["Period"], 60)
        self.assertTrue(param["expression"].startswith("-("))

    def test_batch_decision_returns_next_alpha_without_second_generation_call(self):
        class IteratingModel(FakeModel):
            def analyze_batch(self, context):
                self.feedback_calls.append(context)
                parent = context["families"][0]["candidates"][0]["id"]
                return {"action": "ITERATE", "reason": "Try volume conditioning", "next_goal": "volume mechanism",
                        "conclusion": "Price-only signal weak", "optimized_candidates": [{
                            "parent_candidate_id": parent, "hypothesis": "Volume conditioned reversal",
                            "expression_template": "rank(volume / ts_mean(volume,{Period}))",
                            "period": {"name": "Period", "default": 35}, "parameters": []}]}

        self.model = IteratingModel()
        self.workflow = ResearchWorkflow(FakeConfig(), self.store, ResearchAgent(self.model),
                                         AlphaAgent(self.model), FakeEngine())
        self.workflow.add_topic("Iterate")
        self.workflow.step()
        first = self.store.one("SELECT * FROM experiments ORDER BY ordinal LIMIT 1")
        original = self.store.all("SELECT * FROM candidates WHERE experiment_id=?", (first["id"],))
        for row in original:
            self.store.update_candidate(row["id"], status="rejected", result={"is": {"sharpe": 0.2}})
        self.assertIn("experiment queued", self.workflow.step())
        self.assertEqual(self.model.variants_calls, 1)
        self.assertEqual(len(self.model.feedback_calls), 1)
        second = self.store.one("SELECT * FROM experiments WHERE ordinal=2")
        children = self.store.all("SELECT * FROM candidates WHERE experiment_id=?", (second["id"],))
        self.assertTrue(children)
        self.assertEqual({row["parent_candidate_id"] for row in children}, {original[0]["id"]})
        for row in children:
            self.store.update_candidate(row["id"], status="rejected", result={"is": {"sharpe": 0.1}})
        self.assertIn("direction completed", self.workflow.step())
        self.assertEqual(len(self.model.feedback_calls[1]["family_history"]), 1)
        self.assertEqual(self.model.feedback_calls[1]["family_history"][0]["statistics"]["count"], 8)

    def test_invalid_optimized_plan_is_not_replayed_forever(self):
        self.workflow.add_topic("Invalid optimization")
        self.workflow.step()
        topic = self.store.one("SELECT * FROM topics WHERE status='active'")
        direction = self.store.one("SELECT * FROM directions WHERE status='active'")
        invalid = [{"parent_candidate_id": "old", "hypothesis": "invalid syntax",
                    "expression_template": "rank(close,{Period}) #invalid",
                    "period": {"name": "Period", "default": 30}, "parameters": []}]
        result = self.workflow._create_experiment(topic, direction, "try repair", invalid)
        self.assertIn("direction completed", result)
        self.assertEqual(self.store.one("SELECT status FROM directions WHERE id=?", (direction["id"],))["status"],
                         "completed")

    def test_repeated_invalid_initial_plans_stop_direction_after_three_attempts(self):
        class InvalidModel(FakeModel):
            def generate_variants(self, context):
                return [{"hypothesis": "bad syntax", "expression_template": "rank(close,{Period}) #invalid",
                         "period": {"name": "Period", "default": 30}, "parameters": []}]

        self.model = InvalidModel()
        self.workflow = ResearchWorkflow(FakeConfig(), self.store, ResearchAgent(self.model),
                                         AlphaAgent(self.model), FakeEngine())
        self.workflow.add_topic("Invalid initial plans")
        self.assertEqual(self.workflow.step(), "no valid expressions")
        direction = self.store.one("SELECT * FROM directions WHERE status='active'")
        for _ in range(2):
            with self.store.tx() as db:
                db.execute("DELETE FROM cooldowns WHERE service='llm'")
            result = self.workflow.step()
        self.assertIn("direction completed", result)
        self.assertEqual(self.store.one("SELECT consecutive FROM direction_generation_errors WHERE direction_id=?",
                                        (direction["id"],))["consecutive"], 3)

    def test_failed_inverse_does_not_enter_parameter_search(self):
        self.workflow.add_topic("Negative signal")
        self.workflow.step()
        experiment = self.store.one("SELECT * FROM experiments")
        for row in self.store.all("SELECT * FROM candidates WHERE experiment_id=?", (experiment["id"],)):
            self.store.update_candidate(row["id"], status="rejected", result={"is": {"sharpe": -1.4}})
        self.assertIn("inverse verification queued", self.workflow.step())
        inverses = self.store.all("SELECT * FROM candidates WHERE experiment_id=? AND origin_action='INVERT_PERIOD'", (experiment["id"],))
        self.assertEqual(len(inverses), 4)
        for inverse in inverses:
            self.store.update_candidate(inverse["id"], status="rejected", result={"is": {"sharpe": 0.3}})
        self.assertIn("direction completed", self.workflow.step())
        self.assertIsNone(self.store.one("SELECT id FROM candidates WHERE experiment_id=? AND origin_action='PARAM1'", (experiment["id"],)))

    def test_negative_parameter_result_is_verified_in_reverse_once(self):
        self.workflow.add_topic("Parameter inversion")
        self.workflow.step()
        experiment = self.store.one("SELECT * FROM experiments")
        for row in self.store.all("SELECT * FROM candidates WHERE experiment_id=?", (experiment["id"],)):
            period = json.loads(row["parameter_values"])["Period"]
            self.store.update_candidate(row["id"], status="rejected", result={"is": {"sharpe": 1.3 if period == 60 else 0.2}})
        self.assertIn("PARAM1 queued", self.workflow.step())
        parameter = self.store.one("SELECT * FROM candidates WHERE experiment_id=? AND origin_action='PARAM1'", (experiment["id"],))
        self.store.update_candidate(parameter["id"], status="rejected", result={"is": {"sharpe": -1.4}})
        self.assertIn("inverse verification queued", self.workflow.step())
        inverse = self.store.one("SELECT * FROM candidates WHERE experiment_id=? AND origin_action='INVERT_PARAM1'", (experiment["id"],))
        self.assertEqual(inverse["parent_candidate_id"], parameter["id"])
        self.store.update_candidate(inverse["id"], status="rejected", result={"is": {"sharpe": 1.3}})
        self.assertIn("direction completed", self.workflow.step())
        self.assertEqual(self.store.one("SELECT COUNT(*) AS n FROM candidates WHERE experiment_id=? AND origin_action='INVERT_PARAM1'", (experiment["id"],))["n"], 1)

    def test_same_direction_mixed_settings_combinations_stop_after_two_fitness_failures(self):
        class ComboConfig(FakeConfig):
            max_combination_pairs = 2

        self.workflow = ResearchWorkflow(ComboConfig(), self.store, ResearchAgent(self.model),
                                         AlphaAgent(self.model), FakeEngine())
        self.workflow.add_topic("Combine")
        self.workflow.step()
        topic = self.store.one("SELECT * FROM topics WHERE status='active'")
        direction = self.store.one("SELECT * FROM directions WHERE status='active'")
        experiment = self.store.one("SELECT * FROM experiments WHERE status='active'")
        first = self.store.one("SELECT * FROM candidates WHERE experiment_id=? ORDER BY created_at,id LIMIT 1", (experiment["id"],))
        ids = [first["id"]]
        for index, expression in enumerate(("rank(volume)", "rank(vwap / close)"), 1):
            ids.append(self.store.add_candidate(topic["run_id"], 1, f"mechanism {index}", expression,
                                                {"region": "USA", "neutralization": "MARKET" if index == 1 else "NONE"},
                                                topic_id=topic["id"], direction_id=direction["id"],
                                                experiment_id=experiment["id"], variant_index=index))
        for index, candidate_id in enumerate(ids):
            self.store.update_candidate(candidate_id, status="rejected", alpha_id=f"A{index}",
                                        result={"is": {"sharpe": 1.4, "fitness": 0.4,
                                                       "turnover": (0.1, 0.5, 0.9)[index]}})
        self.assertEqual(self.workflow._schedule_combinations(topic, direction, experiment), 6)
        pairs = self.store.all("SELECT * FROM combination_pairs ORDER BY created_at,rowid")
        self.assertEqual(len(pairs), 2)
        combos = self.store.all("SELECT * FROM candidates WHERE origin_action='COMBINATION'")
        self.assertEqual(len(combos), 6)
        self.assertTrue(all(json.loads(row["settings"]) == ComboConfig().settings() for row in combos))
        for row in combos:
            self.store.update_candidate(row["id"], status="rejected", alpha_id="C" + row["id"],
                                        result={"is": {"sharpe": 1.0, "fitness": 0.3, "turnover": 0.2}})
        self.workflow._finish_combinations(experiment)
        attempts = self.store.all("SELECT parent_id,improved FROM combination_attempts")
        counts = {candidate_id: sum(row["parent_id"] == candidate_id for row in attempts) for candidate_id in ids}
        twice = next(candidate_id for candidate_id, count in counts.items() if count == 2)
        self.assertTrue(self.workflow._combination_stopped(twice))
        selected_combo = next(row for row in combos if twice in self.store.component_ids(row["id"]))
        self.assertTrue(self.store.reserve_submission("primary", selected_combo["id"],
                                                      "C" + selected_combo["id"], "2020-01-01"))
        self.assertFalse(self.store.reserve_submission("primary", twice, "A0", "2020-01-02"))
        duplicate = self.store.add_candidate(topic["run_id"], 2, "Duplicate platform alpha", "rank(high)",
                                             {"region": "USA"})
        original_alpha = self.store.one("SELECT alpha_id FROM candidates WHERE id=?", (twice,))["alpha_id"]
        self.store.update_candidate(duplicate, alpha_id=original_alpha)
        self.assertFalse(self.store.reserve_submission("primary", duplicate, original_alpha, "2020-01-03"))

    def test_completed_single_factor_stage_schedules_combinations_before_feedback(self):
        class ComboConfig(FakeConfig):
            max_combination_pairs = 1

        self.workflow = ResearchWorkflow(ComboConfig(), self.store, ResearchAgent(self.model),
                                         AlphaAgent(self.model), FakeEngine())
        self.workflow.add_topic("Combine")
        self.workflow.step()
        topic = self.store.one("SELECT * FROM topics WHERE status='active'")
        direction = self.store.one("SELECT * FROM directions WHERE status='active'")
        experiment = self.store.one("SELECT * FROM experiments WHERE status='active'")
        original = self.store.all("SELECT * FROM candidates WHERE experiment_id=?", (experiment["id"],))
        for row in original:
            self.store.update_candidate(row["id"], status="rejected", alpha_id="A" + row["id"],
                                        result={"is": {"sharpe": 0.3, "fitness": 0.3, "turnover": 0.1}})
        other = self.store.add_candidate(topic["run_id"], 1, "Different mechanism", "rank(volume)",
                                         {"neutralization": "MARKET"}, topic_id=topic["id"],
                                         direction_id=direction["id"], experiment_id=experiment["id"], variant_index=1)
        self.store.update_candidate(other, status="rejected", alpha_id="AOTHER",
                                    result={"is": {"sharpe": 0.4, "fitness": 0.2, "turnover": 0.9}})
        self.assertIn("combination batch queued", self.workflow.step())
        self.assertEqual(self.store.one("SELECT stage FROM experiments WHERE id=?", (experiment["id"],))["stage"], "COMBINATION")
        self.assertEqual(self.store.one("SELECT COUNT(*) AS n FROM combination_members")["n"], 3)
        self.assertFalse(self.model.feedback_calls)


if __name__ == "__main__":
    unittest.main()
