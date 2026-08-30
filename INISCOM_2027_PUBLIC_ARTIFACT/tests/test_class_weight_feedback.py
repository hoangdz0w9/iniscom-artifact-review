import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rf_class_weight_ablation import arm_parameters, paired_difference, check_control, analyze_pairs, ARMS, POLICIES, SEEDS
from build_paper_assets import band_rates, merge_sparse_bins


class ClassWeightFeedbackTests(unittest.TestCase):
    def test_only_class_weight_changes(self):
        params = {"class_weight": "balanced_subsample", "n_estimators": 30, "n_jobs": 1,
                  "random_state": 21, "max_depth": 10, "min_samples_leaf": 10}
        before = dict(params)
        changed = arm_parameters(params, "none")
        self.assertEqual([k for k in params if params[k] != changed[k]], ["class_weight"])
        self.assertIsNone(changed["class_weight"])
        self.assertEqual(params, before)
        self.assertEqual(arm_parameters(params, "balanced_subsample"), params)

    def test_refuse_wrong_rf_or_arm(self):
        base = {"class_weight": "balanced_subsample", "n_estimators": 30, "n_jobs": 1}
        for changed in ({**base, "n_estimators": 300}, {**base, "class_weight": None}, {**base, "n_jobs": 8}):
            with self.assertRaises(ValueError):
                arm_parameters(changed, "none")
        with self.assertRaises(ValueError):
            arm_parameters(base, "balanced")

    def test_seed_pairing_by_id_not_position(self):
        left = pd.Series([10., 20.], index=[42, 21])
        right = pd.Series([17., 9.], index=[21, 42])
        np.testing.assert_equal(paired_difference(left, right).to_numpy(), [3., 1.])

    def test_reject_bad_pairs(self):
        for left, right in ((pd.Series([1, 2], index=[21, 21]), pd.Series([1, 2], index=[21, 42])),
                            (pd.Series([1], index=[21]), pd.Series([1], index=[42]))):
            with self.assertRaises(ValueError):
                paired_difference(left, right)

    def test_control_checks_scores_labels_ids(self):
        old = SimpleNamespace(y=np.array([0, 1]), score=np.array([.2, .8]), row_id=np.array([4, 8]))
        self.assertEqual(check_control(old.score, old, old.y, old.row_id), 0.)
        for score, y, ids in ((old.score+.001, old.y, old.row_id), (old.score, old.y[::-1], old.row_id), (old.score, old.y, old.row_id[::-1])):
            with self.assertRaises(ValueError):
                check_control(score, old, y, ids)

    def test_paired_gap_change_direction(self):
        rows = []
        for arm in ARMS:
            for policy in POLICIES:
                for seed in SEEDS:
                    value = 90. if policy == POLICIES[0] else (87. if arm == "balanced_subsample" else 89.)
                    rows.append({"arm": arm, "seed": seed, "policy": policy, "cutoff": .5, "test_F1_weighted_pct": value})
        result = analyze_pairs(rows).set_index("contrast")
        self.assertEqual(result.loc["change in default-minus-validation gap: none minus balanced_subsample", "mean"], -2.)

    def test_exact_denominators(self):
        bands = pd.DataFrame([{"model_key": "rf30", "seed": 21, "split": "validation", "normal_rows_reclassified": 425, "attack_rows_reclassified": 555}])
        audit = pd.DataFrame([{"model_key": "rf30", "seed": 21, "split": "validation", "n_normal": 11200, "n_attack": 23869}])
        result = band_rates(bands, audit).iloc[0]
        self.assertAlmostEqual(result.normal_band_pct, 100*425/11200)
        self.assertAlmostEqual(result.attack_band_pct, 100*555/23869)
        with self.assertRaises(ValueError):
            band_rates(bands, audit.iloc[0:0])

    def test_actual_bin_support_and_zero_retention(self):
        internal = ROOT / "generated/feedback_diagnostics/drr_residual_bins_by_seed.csv"
        public = ROOT / "results/feedback_diagnostics/drr_residual_bins_by_seed.csv"
        original = pd.read_csv(internal if internal.is_file() else public)
        merged = merge_sparse_bins(original)
        counts = merged.groupby(["input_n", "stage3_bin"]).n_calls.sum()
        self.assertEqual(int(counts.sum()), 200)
        self.assertTrue((counts >= 10).all())
        self.assertEqual(set(counts.loc[32].index), {"0", "1", "2-16"})
        self.assertEqual(set(counts.loc[1000].index), {"0", "1-16", "17-64", "65+"})
        self.assertEqual(int(counts.loc[32, "0"]), 53)
        self.assertEqual(int(counts.loc[1000, "0"]), 19)

    def test_merge_recomputes_error_not_average_errors(self):
        original = pd.DataFrame([
            {"seed": 21, "input_n": 32, "stage3_bin": "1", "n_calls": 5, "stage3_n_mean": 1., "measured_mean_s": 1., "predicted_mean_s": 2., "signed_error_pct": 100.},
            {"seed": 21, "input_n": 32, "stage3_bin": "2-4", "n_calls": 5, "stage3_n_mean": 2., "measured_mean_s": 3., "predicted_mean_s": 3., "signed_error_pct": 0.}])
        result = merge_sparse_bins(original).iloc[0]
        self.assertEqual(result.n_calls, 10)
        self.assertEqual(result.stage3_bin, "1-4")
        self.assertAlmostEqual(result.signed_error_pct, 25.)


if __name__ == "__main__":
    unittest.main()
