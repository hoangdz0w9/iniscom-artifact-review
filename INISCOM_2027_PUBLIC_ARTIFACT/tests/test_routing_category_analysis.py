"""Unit checks for row alignment, routing boundaries, and rate denominators."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from routing_category_analysis import (  # noqa: E402
    FrozenModel, Provenance, category_row, compose_route, descriptive_summary,
    load_clean_test, parse_args, resolve_analysis_paths, validate_prediction,
)


class RoutingTests(unittest.TestCase):
    def test_cli_defaults_separate_data_and_revision_roots(self):
        args = parse_args([])
        self.assertEqual(args.project_root, Path(__file__).resolve().parents[2])
        self.assertEqual(args.revision_root, Path(__file__).resolve().parents[1])

    def test_cli_data_parent_and_explicit_output_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            revision = data / "revision"
            args = parse_args(["--project-root", str(data), "--revision-root", str(revision)])
            project, rev, source, test_csv, out = resolve_analysis_paths(args, "routing_categories")
            self.assertEqual(project, data.resolve())
            self.assertEqual(rev, revision.resolve())
            self.assertEqual(source, (data / "FINAL3").resolve())
            self.assertEqual(test_csv, (data / "UNSW_NB15_testing-set.csv").resolve())
            self.assertEqual(out, (revision / "generated" / "routing_categories").resolve())
            requested = revision / "generated" / "custom"
            args = parse_args(["--project-root", str(data), "--revision-root", str(revision),
                               "--out-dir", str(requested), "--source-root", str(data / "archive"),
                               "--test-csv", str(data / "explicit.csv")])
            _, _, source, test_csv, out = resolve_analysis_paths(args, "routing_categories")
            self.assertEqual(out, requested.resolve())
            self.assertEqual(source, (data / "archive").resolve())
            self.assertEqual(test_csv, (data / "explicit.csv").resolve())

    def test_cli_never_writes_data_parent_or_source_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            revision = data / "revision"
            cases = (
                ["--revision-root", str(data)],
                ["--revision-root", str(data / "FINAL3" / "nested")],
                ["--revision-root", str(revision), "--out-dir", str(data / "outside")],
                ["--revision-root", str(revision), "--out-dir", str(revision / "generated")],
                ["--revision-root", str(revision), "--source-root", str(revision / "generated")],
            )
            for flags in cases:
                with self.subTest(flags=flags), self.assertRaises(ValueError):
                    resolve_analysis_paths(parse_args(["--project-root", str(data), *flags]), "routing_categories")
            self.assertFalse((revision / "generated").exists())

    def test_float32_gate_boundaries_exit_and_interiors_route(self):
        first = np.array([0, .2, .3, .5, .8, 1], dtype=np.float32)
        second = np.array([.9, .9, .7, .4, .1, .1], dtype=np.float32)
        pred, score, r2, r3 = compose_route(first, second, .2)
        np.testing.assert_array_equal(r2, [False, False, True, True, False, False])
        np.testing.assert_array_equal(pred, [0, 0, 1, 0, 1, 1])
        np.testing.assert_array_equal(score, np.array([0, .2, .7, .4, .8, 1], np.float32))
        self.assertEqual(score.dtype, np.float32)
        self.assertFalse(r3.any())

    def test_stage3_disabled_at_half_and_exact_half_is_attack(self):
        first = np.array([.5, .5, .5], np.float32)
        second = np.array([.49, .5, .51], np.float32)
        third = np.array([1, 0, 0], np.float32)
        pred, score, r2, r3 = compose_route(first, second, .1, third, .5)
        np.testing.assert_array_equal(pred, [0, 1, 1])
        np.testing.assert_array_equal(score, second)
        self.assertTrue(r2.all())
        self.assertFalse(r3.any())

    def test_three_stage_nested_masks(self):
        first = np.array([.01, .5, .5, .9], np.float32)
        second = np.array([.5, .5, .2, .5], np.float32)
        third = np.array([.9, .9, .9, .9], np.float32)
        pred, score, r2, r3 = compose_route(first, second, .1, third, .2)
        np.testing.assert_array_equal(r2, [False, True, True, False])
        np.testing.assert_array_equal(r3, [False, True, False, False])
        np.testing.assert_array_equal(pred, [0, 1, 0, 1])
        np.testing.assert_array_equal(score, np.array([.01, .9, .2, .9], np.float32))

    def test_invalid_gates_and_shapes_fail(self):
        x = np.array([.5], np.float32)
        for low in (-.1, .6, np.nan):
            with self.assertRaises(ValueError):
                compose_route(x, x, low)
        with self.assertRaises(ValueError):
            compose_route(x, x, .1, x, None)
        with self.assertRaises(ValueError):
            compose_route(x, np.array([.2, .4]), .1)

    def test_category_denominators_and_enrichment(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1], np.int8)
        arrays = {"y_true": y, "y_pred": y.copy(), "y_score": y.astype(np.float32), "test_row_id": np.arange(8)}
        r2 = np.array([0, 0, 1, 0, 1, 1, 0, 0], bool)
        r3 = np.array([0, 0, 0, 0, 1, 0, 0, 0], bool)
        model = FrozenModel("dt_rf30_rf300", 21, arrays, r2, r3, None, .1, .2)
        cats = np.array(["Normal"] * 4 + ["Rare"] + ["DoS"] * 3)
        row = category_row(model, cats, "Rare")
        self.assertEqual(row["N"], 1)
        self.assertEqual(row["stage3_reach_pct"], 100)
        self.assertEqual(row["stage3_composition_pct"], 100)
        self.assertEqual(row["stage3_enrichment"], 8)
        self.assertTrue(np.isnan(row["native_normal_specificity_pct"]))
        row = category_row(model, cats, "Normal")
        self.assertEqual(row["stage2_reach_pct"], 25)
        self.assertAlmostEqual(row["stage2_composition_pct"], 100 / 3)
        self.assertAlmostEqual(row["stage2_enrichment"], 2 / 3)
        self.assertTrue(np.isnan(row["native_attack_recall_pct"]))
        model.reach3 = np.zeros(8, bool)
        row = category_row(model, cats, "Rare")
        self.assertEqual(row["stage3_rows"], 0)
        self.assertEqual(row["stage3_reach_pct"], 0)
        self.assertTrue(np.isnan(row["stage3_composition_pct"]))
        self.assertTrue(np.isnan(row["stage3_enrichment"]))

    def test_row_ids_validated_against_cleaned_labels(self):
        labels = np.array([0, 0, 1], np.int8)
        item = {"y_true": np.array([1, 0, 0]), "y_pred": np.array([1, 0, 0]), "y_score": np.array([.9, .1, .2]), "test_row_id": np.array([2, 0, 1])}
        validate_prediction(item, labels, "fixture")
        item["test_row_id"] = np.array([2, 0, 0])
        with self.assertRaises(ValueError):
            validate_prediction(item, labels, "duplicate row")
        item["test_row_id"] = np.array([0, 1, 2])
        with self.assertRaises(ValueError):
            validate_prediction(item, labels, "wrong labels")

    def test_cleaning_order_and_category_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.csv"
            pd.DataFrame({"id": [1, 2, 3, 4], "label": [0, "bad", 1, 4], "attack_cat": [" Normal ", "x", "Backdoors", "x"]}).to_csv(path, index=False)
            y, cats, info = load_clean_test(path)
            np.testing.assert_array_equal(y, [0, 1])
            np.testing.assert_array_equal(cats, ["Normal", "Backdoor"])
            self.assertEqual(info["invalid_label_rows_dropped"], 2)

    def test_summary_tracks_undefined_seeds_and_constant_N(self):
        frame = pd.DataFrame({"model": ["x", "x"], "seed": [1, 2], "N": [10, 10], "value": [2., np.nan]})
        result = descriptive_summary(frame, ["model"], ["value"], ["N"]).iloc[0]
        self.assertEqual(result["n_seeds"], 2)
        self.assertEqual(result["value_valid_seeds"], 1)
        self.assertEqual(result["value_mean"], 2.)
        self.assertTrue(np.isnan(result["value_sd"]))
        frame.loc[1, "N"] = 12
        with self.assertRaises(ValueError):
            descriptive_summary(frame, ["model"], ["value"], ["N"])

    def test_input_mutation_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.txt"
            path.write_text("before", encoding="utf-8")
            p = Provenance()
            p.add(path, "unit fixture")
            p.verify_unchanged()
            path.write_text("after", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                p.verify_unchanged()


if __name__ == "__main__":
    unittest.main()
