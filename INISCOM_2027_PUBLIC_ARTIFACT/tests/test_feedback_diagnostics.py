import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from feedback_diagnostics import threshold_metrics, pareto_flags, stage_bin


class FeedbackDiagnosticsTests(unittest.TestCase):
    def test_threshold_equality_is_attack(self):
        result = threshold_metrics([0, 0, 1, 1], [.4, .5, .5, .8], [.5])
        self.assertEqual(result["TN"][0], 1)
        self.assertEqual(result["FP"][0], 1)
        self.assertEqual(result["TP"][0], 2)

    def test_threshold_band_preserves_half_open_endpoints(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([.4, .5, .4, .5])
        result = threshold_metrics(y, score, [.4, .5])
        self.assertEqual(result["FP"][0] - result["FP"][1], 1)
        self.assertEqual(result["FN"][1] - result["FN"][0], 1)

    def test_extreme_thresholds(self):
        result = threshold_metrics([0, 1], [.2, .8], [0., 1.])
        np.testing.assert_equal(result["FP"], [1, 0])
        np.testing.assert_equal(result["FN"], [0, 1])

    def test_rejects_unaligned_or_nonfinite(self):
        for y, score, thresholds in [([0, 1], [.2], [.5]), ([0, 1], [.2, np.nan], [.5]),
                                      ([0, 1], [.2, .8], [np.inf]), ([0, 0], [.2, .8], [.5])]:
            with self.assertRaises(ValueError):
                threshold_metrics(y, score, thresholds)

    def test_tradeoffs_not_deleted_from_frontier(self):
        frame = pd.DataFrame({"F1_mean": [88.607, 89.023, 90.234, 90.356, 87.227],
                              "reference_seconds": [.033, .084, .088, .807, .154]})
        np.testing.assert_equal(pareto_flags(frame), [True, True, True, True, False])

    def test_identical_points_do_not_dominate_each_other(self):
        frame = pd.DataFrame({"F1_mean": [90., 90.], "reference_seconds": [.1, .1]})
        np.testing.assert_equal(pareto_flags(frame), [True, True])

    def test_stage_bins_boundaries(self):
        self.assertEqual([stage_bin(x) for x in [0, 1, 2, 4, 5, 16, 17, 64, 65]],
                         ["0", "1", "2-4", "2-4", "5-16", "5-16", "17-64", "17-64", "65+"])
        for value in [-1, 1.5]:
            with self.assertRaises(ValueError):
                stage_bin(value)


if __name__ == "__main__":
    unittest.main()
