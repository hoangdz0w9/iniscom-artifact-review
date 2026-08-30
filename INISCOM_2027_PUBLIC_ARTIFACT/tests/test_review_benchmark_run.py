import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("benchmark_review", SCRIPTS / "review_benchmark_run.py")
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class ReviewTests(unittest.TestCase):
    def test_expected_protocol_has_all_30800_calls(self):
        groups = review.expected_groups()
        self.assertEqual(len(groups), 540)
        self.assertEqual(sum(groups.values()), 30800)

    def test_exact_gate_boundaries_are_not_forwarded(self):
        setting = {"model_key": "cascade", "low1": .25, "low2": .4}
        scores = {"dt": np.array([.25, .26, .5, .75]), "xgb300": np.array([.5, .4, .5, .5])}
        second, third = review.route_masks(scores, setting)
        np.testing.assert_array_equal(second, [False, True, True, False])
        np.testing.assert_array_equal(third, [False, False, True, False])

    def test_disabled_stage_has_empty_mask(self):
        setting = {"model_key": "cascade", "low1": .1, "low2": .5}
        second, third = review.route_masks({"dt": np.array([.5, .5]), "xgb300": np.array([.5, .7])}, setting)
        self.assertEqual(second.sum(), 2)
        self.assertEqual(third.sum(), 0)

    def test_standalone_has_no_forwarded_rows(self):
        second, third = review.route_masks({"rf30": np.array([.2, .5, .8])}, {"model_key": "rf30"})
        self.assertEqual(second.sum() + third.sum(), 0)

    def test_missing_call_group_is_rejected(self):
        raw = pd.DataFrame([{**dict(zip(review.GROUP_KEYS, key)), "call_index": 0, "input_n": 82332}
                            for key in review.expected_groups()])
        with self.assertRaisesRegex(ValueError, "Missing, duplicated"):
            review.verify_groups(raw)

    def test_bad_summary_cannot_pass_numeric_comparison(self):
        with self.assertRaises(AssertionError):
            review.compare_frames(pd.DataFrame({"seed": [1], "mean": [1.0]}),
                                  pd.DataFrame({"seed": [1], "mean": [1.01]}), ["seed"])


if __name__ == "__main__":
    unittest.main()
