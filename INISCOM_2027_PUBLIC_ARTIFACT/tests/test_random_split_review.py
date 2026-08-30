import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from review_random_split_control import interval, metrics, select_thresholds


class RandomSplitReviewTests(unittest.TestCase):
    def test_threshold_equality_is_attack(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([.1, .5, .5, .9], dtype=np.float32)
        selected = select_thresholds(y, score, .5)
        self.assertIn(selected["global_validation_f1"], set(score) | {np.inf})
        result = metrics(y, score >= .5)
        self.assertEqual((result["TN"], result["FP"], result["FN"], result["TP"]),
                         (1, 1, 0, 2))

    def test_oracle_equals_validation_when_priors_match(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([.1, .4, .6, .9], dtype=np.float32)
        selected = select_thresholds(y, score, float(y.mean()))
        self.assertEqual(selected["global_validation_f1"],
                         selected["global_oracle_prior_weighted_validation_f1"])

    def test_seed_interval_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            interval([1.0, np.nan])


if __name__ == "__main__":
    unittest.main()
