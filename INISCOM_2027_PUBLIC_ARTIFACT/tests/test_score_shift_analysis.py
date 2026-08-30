"""Fast numerical/unit checks; no experiment data or training required."""
from pathlib import Path
from io import StringIO
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import confusion_matrix, f1_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import score_shift_analysis as analysis


class WeightedCountsTests(unittest.TestCase):
    def test_metrics_match_sklearn_sample_weights(self):
        rng = np.random.default_rng(123)
        y = np.r_[np.zeros(30, dtype=int), np.ones(70, dtype=int)]
        pred = rng.integers(0, 2, size=y.size)
        target = .41
        w0, w1 = analysis.prior_weights(float(y.mean()), target)
        weights = np.where(y == 1, w1, w0)
        got = analysis.prediction_metrics(y, pred, target_prior=target)
        counts = confusion_matrix(y, pred, sample_weight=weights, labels=[0, 1]).ravel()
        np.testing.assert_allclose([got[key] for key in analysis.COUNT_COLS], counts)
        for average, key in (("weighted", "F1_weighted_pct"), ("macro", "F1_macro_pct"), ("binary", "F1_attack_pct")):
            self.assertAlmostEqual(got[key], 100 * f1_score(y, pred, sample_weight=weights, average=average, zero_division=0), places=12)
        self.assertAlmostEqual(got["attack_prevalence_pct"], 100 * target, places=12)

    def test_weighting_changes_f1_denominators_not_just_supports(self):
        y = np.array([0, 0, 1, 1, 1, 1])
        pred = np.array([0, 1, 0, 1, 1, 1])
        raw = analysis.prediction_metrics(y, pred)
        weighted = analysis.prediction_metrics(y, pred, target_prior=.2)
        incorrect = .2 * raw["F1_attack_pct"] + .8 * raw["F1_normal_pct"]
        self.assertGreater(abs(weighted["F1_weighted_pct"] - incorrect), .1)
        self.assertEqual(weighted["FAR_pct"], raw["FAR_pct"])
        self.assertEqual(weighted["Attack_recall_pct"], raw["Attack_recall_pct"])

    def test_weighted_table_matches_direct_weighted_predictions(self):
        y = np.array([0, 0, 1, 1, 1])
        score = np.array([.2, .8, .4, .8, .9])
        table = analysis.exact_threshold_table(y, score)
        weighted = analysis.weighted_threshold_table(table, float(y.mean()), .3)
        for row in weighted.itertuples():
            direct = analysis.prediction_metrics(y, score >= row.threshold, target_prior=.3)
            self.assertAlmostEqual(row.F1_weighted_pct, direct["F1_weighted_pct"], places=12)


class PriorCorrectionTests(unittest.TestCase):
    def test_identical_priors_are_exact_identity(self):
        score = np.array([0, .1, .5, .999, 1.])
        np.testing.assert_array_equal(analysis.prior_correct(score, .6, .6), score)

    def test_monotonicity_endpoints_and_known_half_cutoff(self):
        score = np.linspace(0, 1, 1001)
        corrected = analysis.prior_correct(score, .68063, .55060)
        self.assertTrue(np.all(np.diff(corrected) > 0))
        self.assertEqual(corrected[0], 0)
        self.assertEqual(corrected[-1], 1)
        inverse_half = .6349642262491256
        self.assertAlmostEqual(analysis.prior_correct(np.array([inverse_half]), .68063, .55060)[0], .5)

    def test_invalid_priors_and_scores_fail(self):
        for prior in (0., 1., -.1, np.nan):
            with self.assertRaises(ValueError):
                analysis.prior_correct(np.array([.1, .5]), .6, prior)
        for score in (np.array([-.001, .5]), np.array([1.001, .5]), np.array([np.inf, .5]), np.array([[.2, .3]])):
            with self.assertRaises(ValueError):
                analysis.require_scores(score)


class ThresholdTests(unittest.TestCase):
    def test_equal_score_blocks_are_never_split(self):
        y = np.array([0, 1, 0, 1])
        score = np.array([.2, .8, .8, .8])
        table = analysis.exact_threshold_table(y, score)
        self.assertEqual(len(table), 3)
        np.testing.assert_array_equal((table.TP + table.FP).to_numpy(), [0, 3, 4])

    def test_f1_tie_is_deterministic_and_prefers_lower_far(self):
        y = np.array([1, 1, 0, 0])
        score = np.array([.9, .7, .7, .1])
        selected = analysis.select_policy_thresholds(y, score, .5)
        self.assertEqual(selected[analysis.POLICIES[1]], .9)
        perm = np.array([2, 0, 3, 1])
        self.assertEqual(selected, analysis.select_policy_thresholds(y[perm], score[perm], .5))

    def test_threshold_selector_has_no_test_row_input(self):
        y = np.array([0, 0, 1, 1, 1, 1])
        score = np.array([.1, .6, .4, .5, .8, .9])
        test_labels_a = np.array([0, 1, 1, 0, 1])
        test_labels_b = test_labels_a[::-1]
        a = analysis.select_policy_thresholds(y, score, float(test_labels_a.mean()))
        b = analysis.select_policy_thresholds(y, score, float(test_labels_b.mean()))
        self.assertEqual(a, b)
        different_prior = analysis.select_policy_thresholds(y, score, .2)
        for policy in analysis.POLICIES[:2]:
            self.assertEqual(a[policy], different_prior[policy])

    def test_monotone_map_preserves_constraints_and_test_predictions(self):
        rng = np.random.default_rng(42)
        y = np.r_[np.zeros(30, dtype=int), np.ones(70, dtype=int)]
        score = np.round(rng.random(y.size), 2)
        test_score = rng.random(150)
        corrected = analysis.prior_correct(score, .7, .55)
        corrected_test = analysis.prior_correct(test_score, .7, .55)
        raw_table = analysis.exact_threshold_table(y, score)
        corrected_table = analysis.exact_threshold_table(y, corrected)
        np.testing.assert_array_equal(raw_table[analysis.COUNT_COLS], corrected_table[analysis.COUNT_COLS])
        for kind, target in analysis.CONSTRAINTS:
            raw = analysis.select_constraint_row(raw_table, kind, target)
            transformed = analysis.select_constraint_row(corrected_table, kind, target)
            np.testing.assert_array_equal(test_score >= raw.threshold, corrected_test >= transformed.threshold)

    def test_all_negative_classifier_is_available_at_zero_far(self):
        y, score = np.array([0, 1]), np.array([.9, .8])
        chosen = analysis.select_constraint_row(analysis.exact_threshold_table(y, score), "far_max", 0)
        self.assertEqual(chosen.threshold, np.inf)

    def test_equality_and_nextafter_neighbors_use_exact_ge_comparison(self):
        score = np.array([0., np.nextafter(.5, 0.), .5, np.nextafter(.5, 1.), 1.])
        y = np.array([0, 1, 0, 1, 0])
        table = analysis.exact_threshold_table(y, score)
        self.assertEqual(len(table), 6)
        for row in table.itertuples():
            expected = confusion_matrix(y, score >= row.threshold, labels=[0, 1]).ravel()
            np.testing.assert_array_equal([getattr(row, key) for key in analysis.COUNT_COLS], expected)
        at_half = table.loc[table.threshold == .5].iloc[0]
        self.assertEqual(at_half.TP + at_half.FP, 3)
        at_inf = table.loc[np.isinf(table.threshold)].iloc[0]
        self.assertEqual(at_inf.TP + at_inf.FP, 0)

    def test_archived_float32_neighbors_remain_distinct_after_float64_map(self):
        half = np.float32(.5)
        score = np.array([0., np.nextafter(half, np.float32(0.)), half,
                          np.nextafter(half, np.float32(1.)), 1.], dtype=np.float32)
        original = score.copy()
        mapped = analysis.prior_correct(score, .68063, .55060)
        self.assertTrue(np.all(np.diff(mapped) > 0))
        for threshold, corrected_threshold in zip(score.astype(float), mapped, strict=True):
            np.testing.assert_array_equal(score.astype(float) >= threshold, mapped >= corrected_threshold)
        np.testing.assert_array_equal(score, original)

    def test_csv_threshold_roundtrip_preserves_ties_and_infinity(self):
        # A 15-significant-digit CSV raises this actual DT threshold by two
        # float64 ULPs and would exclude all samples tied at the threshold.
        threshold = float(np.float32(.56))
        values = [threshold, .5, np.inf]
        frame = pd.DataFrame({"threshold": values, "threshold_hex": [value.hex() for value in values]})
        buffer = StringIO()
        frame.to_csv(buffer, index=False, float_format=analysis.CSV_FLOAT_FORMAT)
        decoded = pd.read_csv(StringIO(buffer.getvalue()), float_precision="round_trip")
        np.testing.assert_array_equal(decoded.threshold, values)
        decoded_hex = [float.fromhex(value) for value in decoded.threshold_hex]
        np.testing.assert_array_equal(decoded_hex, values)
        self.assertTrue(threshold >= decoded_hex[0])


class DiagnosticTests(unittest.TestCase):
    def test_ks_distance_matches_scipy_with_ties(self):
        a = np.array([0, .1, .1, .5, 1])
        b = np.array([.1, .2, .2, .9, 1, 1])
        got = analysis.ks_effect(a, b)
        self.assertAlmostEqual(got["KS_D"], ks_2samp(a, b).statistic, places=14)
        self.assertNotIn("pvalue", got)

    def test_calibration_endpoint_bins_and_loss(self):
        y, score = np.array([0, 1, 0, 1]), np.array([0, 1, .2, .8])
        metrics, bins = analysis.calibration_metrics(y, score)
        self.assertEqual(sum(row["count"] for row in bins), y.size)
        self.assertEqual(bins[-1]["count"], 1)
        self.assertAlmostEqual(metrics["ECE_15_equal_width"], .1)
        self.assertAlmostEqual(metrics["Brier_score"], .02)
        self.assertAlmostEqual(metrics["log_loss"], log_loss(y, np.clip(score, 1e-15, 1-1e-15)), places=13)

    def test_invalid_labels_fail_before_silent_cast(self):
        for labels in (np.array([0, .2, 1]), np.array([0, 2]), np.array([[0, 1]])):
            with self.assertRaises(ValueError):
                analysis.require_binary(labels, "labels", both=True)


if __name__ == "__main__":
    unittest.main()
