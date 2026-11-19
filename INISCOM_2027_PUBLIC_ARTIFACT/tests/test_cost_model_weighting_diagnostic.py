import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cost_model_weighting_diagnostic.py"
spec = importlib.util.spec_from_file_location("cost_model_weighting_diagnostic", SCRIPT)
diag = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diag
spec.loader.exec_module(diag)


class RelativeFitTests(unittest.TestCase):
    sizes = np.array([1, 4, 16, 64, 100, 10_000, 82_332], dtype=float)

    def test_exact_affine_times_are_recovered(self):
        times = 3.4e-4 + 4.2e-6 * self.sizes
        fit = diag.fit_relative(self.sizes, times)
        self.assertAlmostEqual(fit["fixed_call_s"], 3.4e-4, places=12)
        self.assertAlmostEqual(fit["per_row_s"], 4.2e-6, places=15)
        self.assertLess(fit["fit_max_absolute_percentage_error"], 1e-6)

    def test_objective_is_squared_relative_error(self):
        rng = np.random.default_rng(0)
        times = (3.4e-4 + 4.2e-6 * self.sizes) * (1 + rng.normal(0, .05, self.sizes.size))
        fit = diag.fit_relative(self.sizes, times)

        def objective(a, b):
            return float(np.sum(((a + b * self.sizes - times) / times) ** 2))

        best = objective(fit["fixed_call_s"], fit["per_row_s"])
        for da, db in ((1e-6, 0), (-1e-6, 0), (0, 1e-9), (0, -1e-9)):
            self.assertLessEqual(best, objective(max(fit["fixed_call_s"] + da, 0), max(fit["per_row_s"] + db, 0)) + 1e-15)

    def test_relative_fit_keeps_the_small_call_intercept(self):
        # Curved data: absolute least squares lets the largest batch dominate.
        times = 3.4e-4 + 4.2e-6 * self.sizes ** 1.03
        relative = diag.fit_relative(self.sizes, times)
        design = np.column_stack([np.ones(self.sizes.size), self.sizes])
        absolute = np.linalg.lstsq(design, times, rcond=None)[0]
        self.assertLess(abs(relative["fixed_call_s"] - times[0]), abs(absolute[0] - times[0]))

    def test_nonnegative_coefficients(self):
        times = np.array([1e-3, 9e-4, 8e-4, 7e-4, 7e-4, 2e-2, 1.6e-1])
        fit = diag.fit_relative(self.sizes, times)
        self.assertGreaterEqual(fit["fixed_call_s"], 0)
        self.assertGreaterEqual(fit["per_row_s"], 0)


class CompareOutputTests(unittest.TestCase):
    def write_outputs(self, directory: Path, value: float, table: str):
        frame = pd.DataFrame({"model_key": ["dt"], "mape": [value]})
        for name in diag.COMPARED_CSVS:
            frame.to_csv(directory / name, index=False)
        (directory / "t17.tex").write_text(table, encoding="utf-8")

    def test_tiny_floating_point_differences_are_accepted(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            a, b = Path(a), Path(b)
            self.write_outputs(a, 79.62962003241978, "x")
            self.write_outputs(b, 79.629620032419666, "x")
            checked = diag.compare_outputs(a, a / "t17.tex", b, b / "t17.tex")
            self.assertEqual(len(checked), 4)

    def test_changed_numbers_or_table_are_rejected(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            a, b = Path(a), Path(b)
            self.write_outputs(a, 79.6, "x")
            self.write_outputs(b, 79.7, "x")
            with self.assertRaises(ValueError):
                diag.compare_outputs(a, a / "t17.tex", b, None)
            self.write_outputs(b, 79.6, "y")
            with self.assertRaises(ValueError):
                diag.compare_outputs(a, a / "t17.tex", None, b / "t17.tex")


class ArchivedAssessmentTests(unittest.TestCase):
    def setUp(self):
        records, heldout = [], []
        keys = ([('component', model) for model in diag.ab.COMPONENTS]
                + [('whole_policy', model) for model in diag.ab.STAGES])
        for kind, model in keys:
            records.append({"kind": kind, "model_key": model, "n_seeds": 5,
                            "heldout_batch_sizes": "32;1000", "mean_absolute_percentage_error": 30.0,
                            "max_absolute_percentage_error": 70.0, "threshold_mape_percent": 20,
                            "threshold_max_error_percent": 50, "status": "model_inadequate_or_insufficient"})
            for seed in (21, 42, 84, 168, 336):
                for size in (32, 1000):
                    heldout.append({"kind": kind, "model_key": model, "seed": seed,
                                    "input_n": size, "n_calls": 20})
        self.reproduced = pd.DataFrame(records)
        self.archived = self.reproduced.copy(deep=True)
        self.heldout = pd.DataFrame(heldout)

    def test_complete_matching_original_assessment_is_accepted(self):
        result = diag.validate_archived_assessment(self.reproduced, self.archived, self.heldout)
        self.assertEqual(len(result), 12)

    def test_changed_maximum_error_is_rejected_even_when_mape_matches(self):
        self.archived.loc[0, "max_absolute_percentage_error"] = 71.0
        with self.assertRaisesRegex(ValueError, "max_absolute_percentage_error"):
            diag.validate_archived_assessment(self.reproduced, self.archived, self.heldout)

    def test_missing_or_duplicate_assessment_is_rejected(self):
        for invalid in (self.archived.iloc[:-1], pd.concat([self.archived, self.archived.iloc[:1]])):
            with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                diag.validate_archived_assessment(self.reproduced, invalid, self.heldout)

    def test_changed_status_or_adequate_original_fit_is_rejected(self):
        self.archived.loc[0, "status"] = "descriptively_adequate_on_tested_batches"
        with self.assertRaises(ValueError):
            diag.validate_archived_assessment(self.reproduced, self.archived, self.heldout)
        self.archived = self.reproduced.copy(deep=True)
        for table in (self.reproduced, self.archived):
            table.loc[0, ["mean_absolute_percentage_error", "max_absolute_percentage_error"]] = [10, 30]
        with self.assertRaises(ValueError):
            diag.validate_archived_assessment(self.reproduced, self.archived, self.heldout)

    def test_incomplete_or_changed_seed_batch_call_coverage_is_rejected(self):
        missing = self.heldout.iloc[:-1]
        changed_seed = self.heldout.copy(deep=True)
        changed_seed.loc[0, "seed"] = 999
        fewer_calls = self.heldout.copy(deep=True)
        fewer_calls.loc[0, "n_calls"] = 19
        for invalid in (missing, changed_seed, fewer_calls):
            with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                diag.validate_archived_assessment(self.reproduced, self.archived, invalid)

    def test_changed_threshold_or_seed_count_metadata_is_rejected(self):
        for column, value in (("threshold_mape_percent", 21), ("n_seeds", 4)):
            invalid = self.archived.copy(deep=True)
            invalid.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                diag.validate_archived_assessment(self.reproduced, invalid, self.heldout)


class RecordedDiagnosticTests(unittest.TestCase):
    def test_recorded_table_is_emitted_as_lf_with_windows_text_defaults(self):
        root = SCRIPT.parents[1]
        defaults = diag.layout_defaults(root)
        expected_dir = defaults["expected_dir"] or root / "generated" / "cost_model_diagnostic"
        expected_table = defaults["expected_table"] or root / "generated" / "tables" / "cost_model_weighting.tex"
        original_write_text = Path.write_text

        def windows_default_write_text(path, data, encoding=None, errors=None, newline=None):
            # Exercise the actual Table 17 writer with Windows newline translation
            # even when the test happens to run on a Unix host.
            return original_write_text(path, data, encoding=encoding, errors=errors,
                                       newline="\r\n" if newline is None else newline)

        with tempfile.TemporaryDirectory(dir=root) as directory:
            outdir = Path(directory) / "diagnostic"
            table = Path(directory) / "table17.tex"
            argv = [str(SCRIPT), "--root", str(root), "--out-dir", str(outdir), "--table", str(table),
                    "--expected-dir", str(expected_dir), "--expected-table", str(expected_table)]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(Path, "write_text", new=windows_default_write_text), \
                    redirect_stdout(io.StringIO()):
                diag.main()
            data = table.read_bytes()
            self.assertIn(b"\n", data)
            self.assertNotIn(b"\r", data)
            self.assertEqual(data, expected_table.read_bytes())
            manifest = (outdir / "manifest.json").read_bytes()
            self.assertNotIn(b"\r", manifest)


if __name__ == "__main__":
    unittest.main()
