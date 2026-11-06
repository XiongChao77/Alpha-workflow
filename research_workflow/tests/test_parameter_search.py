import unittest

from research_workflow.parameter_search import normalize_plan, period_values, render_expression


class ParameterSearchTests(unittest.TestCase):
    def test_model_selects_one_or_two_parameters_without_fixed_window(self):
        plan = normalize_plan({
            "expression_template": "rank(ts_mean(close,{Window}) / close) * {Weight}",
            "parameters": [{"name": "Window", "default": 12, "values": [5, 12, 24]},
                           {"name": "Weight", "default": 1, "values": [0.5, 1]}],
        })
        self.assertNotIn("period", plan)
        self.assertEqual(render_expression(plan["expression_template"], {"Window": 24, "Weight": 1}),
                         "rank(ts_mean(close,24) / close) * 1")
        with self.assertRaisesRegex(ValueError, "1 or 2"):
            normalize_plan({"expression_template": "rank(close)", "parameters": []})

    def test_model_period_is_added_once_to_local_grid(self):
        self.assertEqual(period_values((30, 60, 90), 45), [30, 60, 90, 45])
        self.assertEqual(period_values((30, 60, 90), 60), [30, 60, 90])

    def test_named_placeholders_render_without_changing_other_numbers(self):
        plan = normalize_plan({
            "expression_template": "rank(ts_mean(close, {Period}) / ts_mean(volume, 20)) * {Scale}",
            "period": {"name": "Period", "default": 45},
            "parameters": [{"name": "Scale", "default": 1, "values": [1, 2]}],
        })
        self.assertEqual(render_expression(plan["expression_template"], {"Period": 60, "Scale": 2}),
                         "rank(ts_mean(close, 60) / ts_mean(volume, 20)) * 2")

    def test_undeclared_or_missing_parameter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "placeholders must match"):
            normalize_plan({"expression_template": "rank(ts_mean(close, {Other}))",
                            "period": {"name": "Period", "default": 30}, "parameters": []})
        with self.assertRaisesRegex(ValueError, "at most two"):
            normalize_plan({"expression_template": "rank(ts_mean(close, {Period}))",
                            "period": {"name": "Period", "default": 30},
                            "parameters": [{"name": name, "default": 1, "values": [1]}
                                           for name in ("A", "B", "C")]})


if __name__ == "__main__":
    unittest.main()
