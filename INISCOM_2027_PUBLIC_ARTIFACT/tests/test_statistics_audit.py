"""Statistical unit checks; numerical fixtures are tests, never research data."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from statistics_audit import (  # noqa: E402
    grid_dominance, holm_adjust, metrics, paired_statistics, parse_args, precision_power,
    require_close, tost,
)
from routing_category_analysis import resolve_analysis_paths  # noqa: E402


class StatisticsTests(unittest.TestCase):
    def test_cli_defaults_match_data_parent_contract(self):
        args = parse_args([])
        self.assertEqual(args.project_root, Path(__file__).resolve().parents[2])
        self.assertEqual(args.revision_root, Path(__file__).resolve().parents[1])

    def test_cli_explicit_revision_root_and_out_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            revision = Path(tmp) / "revision"
            output = revision / "generated" / "custom_statistics"
            args = parse_args(["--project-root", str(data), "--revision-root", str(revision), "--out-dir", str(output)])
            project, rev, source, test_csv, out = resolve_analysis_paths(args, "statistics")
            self.assertEqual(project, data.resolve())
            self.assertEqual(rev, revision.resolve())
            self.assertEqual(source, (data / "FINAL3").resolve())
            self.assertEqual(test_csv, (data / "UNSW_NB15_testing-set.csv").resolve())
            self.assertEqual(out, output.resolve())

    def test_paired_interval_uses_sample_sd_and_seed_df(self):
        result = paired_statistics(np.array([-1., 0., 1.]))
        self.assertEqual(result["sd_delta_pp"], 1.)
        self.assertAlmostEqual(result["SE_pp"], 1 / np.sqrt(3))
        self.assertAlmostEqual(result["CI95_high_pp"], student_t.ppf(.975, 2) / np.sqrt(3))
        self.assertEqual(result["p_two_sided_raw"], 1.)
        self.assertEqual(result["A_better_seeds"], 1)
        self.assertEqual(result["B_better_seeds"], 1)

    def test_tost_can_pass_with_ten_low_variance_pairs(self):
        delta = np.linspace(-.01, .01, 10)
        result = tost(delta, .1)
        self.assertTrue(result["equivalent_alpha_0_05"])
        self.assertTrue(result["CI90_strictly_inside_margin"])
        self.assertFalse(tost(np.linspace(-2, 2, 10), .1)["equivalent_alpha_0_05"])

    def test_tost_zero_variance_cases(self):
        self.assertTrue(tost(np.zeros(10), .1)["equivalent_alpha_0_05"])
        self.assertFalse(tost(np.ones(10), .1)["equivalent_alpha_0_05"])

    def test_holm_is_step_down_cumulative_max(self):
        raw = np.array([.04613833784951293, .0019243083077850303, .04934354657021314])
        result = holm_adjust(raw)
        np.testing.assert_allclose(result, [.09227667569902586, .005772924923355091, .09227667569902586], rtol=1e-13)
        result4 = holm_adjust(np.r_[.27853572047756947, raw])
        np.testing.assert_allclose(result4, [.27853572047756947, .13841501354853877, .007697233231140121, .13841501354853877], rtol=1e-13)
        with self.assertRaises(ValueError):
            holm_adjust(np.array([np.nan, .1]))

    def test_precision_and_power_are_distinct(self):
        row = precision_power(1.312530150969073, .1, 10)
        self.assertEqual(row["fixed_SD_t_precision_min_n"], 468)
        self.assertEqual(row["normal_approx_power_n_ceiling"], 1476)
        self.assertAlmostEqual(row["observed_n_CI90_halfwidth_pp"], .7608490628601346)
        self.assertAlmostEqual(row["SD_upper_bound_at_observed_n_and_zero_mean_pp"], .17250861110810795)
        self.assertEqual(row["assumed_true_mean_difference_pp"], 0)

    def test_grid_dominance_counts_ties_and_crossings(self):
        result = grid_dominance(np.array([100, 98, 55]), np.array([100, 97, 55]))
        self.assertTrue(result["A_weakly_dominates_mean_grid"])
        self.assertFalse(result["A_strictly_higher_at_every_grid_point"])
        self.assertEqual(result["tied_points_at_tolerance"], 2)
        result = grid_dominance(np.array([99, 98]), np.array([100, 97]))
        self.assertFalse(result["A_weakly_dominates_mean_grid"])
        self.assertEqual(result["A_higher_points"], 1)
        self.assertEqual(result["B_higher_points"], 1)

    def test_metrics_use_percent_and_ordinary_MCC(self):
        y = np.array([0, 0, 1, 1])
        result = metrics(y, y, np.array([.1, .2, .8, .9]))
        self.assertEqual(result["F1_weighted"], 100.)
        self.assertEqual(result["PR_AUC"], 100.)
        self.assertEqual(result["MCC"], 1.)
        self.assertEqual(result["FAR"], 0.)

    def test_integrity_mismatch_and_nonfinite_fail(self):
        with self.assertRaises(RuntimeError):
            require_close(1., 2., "unit fixture")
        with self.assertRaises(RuntimeError):
            require_close(np.nan, 1., "unit fixture")
        with self.assertRaises(ValueError):
            paired_statistics(np.array([1., np.nan]))


if __name__ == "__main__":
    unittest.main()
