import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from review_class_weight_ablation import thresholds, metrics, interval, compare_frames
from score_shift_analysis import select_policy_thresholds


class ClassWeightReviewTests(unittest.TestCase):
    def test_independent_selector_matches_exhaustive_reference(self):
        rng = np.random.default_rng(17)
        for _ in range(20):
            y = np.r_[np.zeros(41, dtype=int), np.ones(59, dtype=int)]
            score = np.round(rng.uniform(size=100), 2).astype(np.float32)
            self.assertEqual(thresholds(y, score, .37), select_policy_thresholds(y, score, .37))

    def test_equality_and_all_equal_scores(self):
        y, s = np.array([0, 0, 1, 1]), np.full(4, .5)
        self.assertEqual(thresholds(y, s, .5), select_policy_thresholds(y, s, .5))
        self.assertEqual(metrics(y, s >= .5)["TP"], 2)
        self.assertEqual(metrics(y, s >= .5)["FP"], 2)

    def test_interval_rejects_missing_values(self):
        for values in ([1], [1, np.nan], [1, np.inf]):
            with self.assertRaises(ValueError):
                interval(values)
        self.assertEqual(interval([2, 2, 2])["ci95_low"], 2.)

    def test_compare_matches_seed_labels_not_order(self):
        a = pd.DataFrame({"seed": [42, 21], "F1": [90., 88.]})
        compare_frames(a, a.iloc[::-1], ["seed"])

    def test_compare_rejects_edited_metric_and_duplicate(self):
        a = pd.DataFrame({"seed": [21, 42], "F1": [88., 90.]})
        b = a.copy()
        b.loc[0, "F1"] = 89.
        with self.assertRaises(ValueError):
            compare_frames(a, b, ["seed"])
        with self.assertRaises(ValueError):
            compare_frames(a, pd.concat([a, a]), ["seed"])

    def test_metrics_are_percent_not_fraction(self):
        result = metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
        self.assertEqual(result["F1_weighted_pct"], 50.)
        self.assertEqual(result["FAR_pct"], 50.)


if __name__ == "__main__":
    unittest.main()
