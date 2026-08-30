import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_paper_assets.py"
spec = importlib.util.spec_from_file_location("paper_assets", SCRIPT)
assets = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = assets
spec.loader.exec_module(assets)


class AssetTests(unittest.TestCase):
    def threshold_inputs(self):
        policy, thresholds, cost = [], [], []
        for k in assets.ORDER:
            cost.append(dict(model_key=k, F1_mean=90., F1_sd=.2))
            for i, p in enumerate(assets.POLICIES):
                policy.append(dict(model_key=k, policy=p, test_F1_weighted_pct_mean=90.-i,
                                   test_F1_weighted_pct_sd=.2, test_FAR_pct_mean=10.+i,
                                   test_Attack_recall_pct_mean=98.-i))
                thresholds.append(dict(model_key=k, policy=p, cutoff_mean=.5+i*.01, cutoff_sd=i*.001))
        return pd.DataFrame(policy), pd.DataFrame(thresholds), pd.DataFrame(cost)

    def test_threshold_main_omits_only_default_and_keeps_model_labels(self):
        main = assets.threshold_table_rows(*self.threshold_inputs())
        main = [r for r in main if " & " in r]
        self.assertEqual(len(main), 16)
        self.assertFalse(any("Default 0.5" in r for r in main))
        for i, k in enumerate(assets.ORDER):
            self.assertTrue(main[2*i].startswith(assets.TEX[k] + " & Global val-F1 cutoff"))
            self.assertTrue(main[2*i+1].startswith(" & Oracle prior"))
            self.assertIn(assets.pm(.51, .001, 4), main[2*i])
            self.assertIn(assets.pm(89., .2), main[2*i])
            self.assertIn("11.00 & 97.00", main[2*i])
            self.assertIn("12.00 & 96.00", main[2*i+1])

    def test_threshold_dedup_refuses_inconsistent_default_mean_or_sd(self):
        for field in ("F1_mean", "F1_sd"):
            policy, thresholds, cost = self.threshold_inputs()
            cost.loc[0, field] += .01
            with self.assertRaisesRegex(ValueError, "Default F1/SD disagree"):
                assets.threshold_table_rows(policy, thresholds, cost)

    def test_threshold_dedup_refuses_missing_or_duplicate_policy(self):
        policy, thresholds, cost = self.threshold_inputs()
        for broken in (policy.iloc[1:], pd.concat([policy, policy.iloc[[0]]])):
            with self.assertRaises(ValueError):
                assets.threshold_table_rows(broken, thresholds, cost)

    def test_nonfinite_numbers_are_not_printed(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                assets.number(value)

    def test_tex_rows_have_two_backslashes(self):
        self.assertEqual(assets.row("a", "b"), "a & b " + "\\" * 2)

    def test_tiny_positive_routing_is_not_reported_as_zero(self):
        self.assertEqual(assets.small_rate(.0005), r"$<0.01$")
        self.assertEqual(assets.small_rate(0), "0.00")
        self.assertEqual(assets.small_rate(.01), "0.01")

    def test_unique_rejects_ambiguous_data(self):
        frame = pd.DataFrame({"model": ["a", "a"], "value": [1, 2]})
        with self.assertRaises(ValueError):
            assets.unique(frame, model="a")

    def test_manifest_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / "metric.csv"
            csv.write_text("x\n1\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"status": "complete", "outputs": [
                {"file": "metric.csv", "sha256": assets.digest(csv)}]}), encoding="utf-8")
            self.assertEqual(assets.verify_analysis_manifest(manifest), 1)
            csv.write_text("x\n2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                assets.verify_analysis_manifest(manifest)

    def test_manifest_rejects_incomplete_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text('{"status":"running"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                assets.verify_analysis_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
