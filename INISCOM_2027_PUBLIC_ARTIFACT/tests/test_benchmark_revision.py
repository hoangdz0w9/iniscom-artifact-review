"""Synthetic correctness tests; no research CSVs or fitted research models used."""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))
import benchmark_revision as bench
import analyze_benchmark as analyze

bench.load_dependencies()
analyze.load_dependencies()
np, pd = bench.np, bench.pd
from benchmark_helpers import LegacyPreprocessor


class FakeComponent:
    def __init__(self, column):
        self.column = column
        self.calls = []

    def predict_proba(self, X):
        self.calls.append(len(X))
        score = np.asarray(X[:, self.column], dtype=float)
        return np.column_stack([1-score, score])


def fake_models():
    return {"dt": FakeComponent(0), "rf30": FakeComponent(1),
            "rf300": FakeComponent(2), "xgb300": FakeComponent(3)}


def frozen_rows():
    original = []
    for model in bench.MODELS:
        original.append({
            "seed": "21", "model_key": model, "dataset": "UNSW-NB15",
            "protocol": "official_train_test", "mode_raw": "no_reduction",
            "selected_threshold": ".6", "threshold_dt_low": ".2",
            "threshold_xgb_low": ".49", "threshold_rf30_low": ".49",
        })
    closed = [{"seed": "21", "architecture": model, "threshold_dt_low": ".15",
               "threshold_stage2_low": ".5"} for model in ("cascade", "dt_rf30_rf300")]
    return original, closed


class BenchmarkTests(unittest.TestCase):
    def test_closed_gates_take_precedence(self):
        original, closed = frozen_rows()
        settings = bench.settings_from_tables(original, closed, (21,))
        for model in ("cascade", "dt_rf30_rf300"):
            self.assertEqual(settings[(21, model)].low1, .15)
            self.assertEqual(settings[(21, model)].low2, .5)
        self.assertEqual(settings[(21, "dt_rf30")].low1, .2)
        self.assertEqual(settings[(21, "rf30")].cutoff, .6)

    def test_missing_closed_row_never_falls_back(self):
        original, closed = frozen_rows()
        with self.assertRaisesRegex(ValueError, "FINAL closed"):
            bench.settings_from_tables(original, closed[:1], (21,))

    def test_duplicate_or_wrong_protocol_rejected(self):
        original, closed = frozen_rows()
        original[0]["mode_raw"] = "ipca"
        with self.assertRaisesRegex(ValueError, "Wrong experiment"):
            bench.settings_from_tables(original, closed, (21,))
        original, closed = frozen_rows()
        with self.assertRaisesRegex(ValueError, "Expected one"):
            bench.settings_from_tables(original + [original[0]], closed, (21,))

    def test_setting_hash_changes_with_gate(self):
        a = bench.Setting("cascade", 21, low1=.1, low2=.49)
        b = bench.Setting("cascade", 21, low1=.1, low2=.5)
        self.assertNotEqual(a.digest(), b.digest())

    def test_stage3_disabled_and_half_score_is_attack(self):
        X = np.array([[.1,.1,.1,.1], [.3,.2,.9,.6], [.6,.5,.1,.5], [.9,.4,.3,.9]])
        models = fake_models()
        setting = bench.Setting("dt_rf30_rf300", 21, low1=.2, low2=.5)
        pred, score, n2, n3 = bench.predict_policy(models, X, setting)
        np.testing.assert_array_equal(pred, [0,0,1,1])
        np.testing.assert_allclose(score, [.1,.2,.5,.9])
        self.assertEqual((n2,n3), (2,0))
        self.assertEqual(models["rf300"].calls, [])

    def test_nonempty_subbatch_call_is_once_not_one_per_row(self):
        X = np.array([[.1,.1,.2,.1], [.3,.2,.4,.1], [.4,.4,.6,.1],
                      [.6,.5,.9,.1], [.7,.8,.6,.1], [.9,.8,.1,.1]])
        models = fake_models()
        pred, score, n2, n3 = bench.predict_policy(models, X, bench.Setting("dt_rf30_rf300",21,low1=.2,low2=.4))
        self.assertEqual((n2,n3), (4,1))
        self.assertEqual(models["dt"].calls, [6])
        self.assertEqual(models["rf30"].calls, [4])
        self.assertEqual(models["rf300"].calls, [1])
        np.testing.assert_array_equal(pred, [0,0,0,1,1,1])

    def test_all_gate_stops_no_expert_call(self):
        X = np.array([[.0,.4,.8,.7],[.1,.5,.8,.7],[.9,.2,.3,.3],[1,.2,.3,.3]])
        models = fake_models()
        _, _, n2, n3 = bench.predict_policy(models,X,bench.Setting("cascade",21,low1=.2,low2=.3))
        self.assertEqual((n2,n3),(0,0))
        self.assertEqual(models["xgb300"].calls, [])
        self.assertEqual(models["rf30"].calls, [])

    def test_standalone_uses_frozen_cutoff(self):
        X = np.array([[.1,.5,.1,.1],[.9,.7,.1,.1]])
        pred, _, n2, n3 = bench.predict_policy(fake_models(),X,bench.Setting("rf30",21,cutoff=.6))
        np.testing.assert_array_equal(pred,[0,1])
        self.assertEqual((n2,n3),(0,0))

    def test_route_counts_retain_saved_float32_rule(self):
        scores = {"dt":np.array([.1,.4,.7,.9],dtype=np.float32),
                  "rf30":np.array([.1,.5,.3,.7],dtype=np.float32)}
        self.assertEqual(bench.route_counts_from_scores(scores,bench.Setting("dt_rf30_rf300",21,low1=.2,low2=.5)),(2,0))

    def test_mismatch_is_recorded_not_hidden_by_close_f1(self):
        y, ids = np.array([0,1]), np.array([0,1])
        archive = dict(y_true=y, y_pred=np.array([0,1]), y_score=np.array([.2,.8],dtype=np.float32),test_row_id=ids)
        audit = bench.compare_prediction(y,ids,np.array([1,0]),np.array([.2,.8]),1,0,archive,(1,0),1e-7)
        self.assertEqual(audit["status"], "new_run_quality")
        self.assertEqual(audit["prediction_mismatch_rows"], 2)
        audit = bench.compare_prediction(y,ids,np.array([0,1]),np.array([.2,.8]),0,0,archive,(1,0),1e-7)
        self.assertFalse(audit["route_counts_match"])
        self.assertEqual(audit["status"], "new_run_quality")

    def test_label_or_id_mismatch_is_fatal(self):
        archive = dict(y_true=np.array([0,1]),y_pred=np.array([0,1]),y_score=np.array([.2,.8]),test_row_id=np.array([0,1]))
        with self.assertRaisesRegex(ValueError,"label pairing"):
            bench.compare_prediction(np.array([1,0]),np.array([0,1]),archive["y_pred"],archive["y_score"],0,0,archive,(0,0),1e-7)
        with self.assertRaisesRegex(ValueError,"test_row_id"):
            bench.compare_prediction(archive["y_true"],np.array([1,0]),archive["y_pred"],archive["y_score"],0,0,archive,(0,0),1e-7)

    def test_preprocessing_train_only_and_serializable(self):
        import joblib
        train = pd.DataFrame({"id":[1,2,3],"attack_cat":["Normal","A","B"],"_row_id":[0,1,2],
                              "dur":[1.,2.,3.],"proto":["tcp","udp","tcp"],"service":["http","dns","http"],
                              "state":["FIN","INT","FIN"],"label":[0,1,1]})
        test = train.iloc[[0]].copy()
        test["proto"] = "unknown_test_only"
        processor = LegacyPreprocessor()
        Xtr = processor.fit_transform(train)
        Xte = processor.transform(test)
        self.assertEqual(Xtr.dtype, np.float32)
        self.assertEqual(Xte.shape[1], Xtr.shape[1])
        self.assertFalse(any("unknown_test_only" in col for col in processor.feature_columns))
        self.assertFalse(any(col in processor.feature_columns for col in ("id","attack_cat","_row_id","label")))
        np.testing.assert_allclose(Xtr.mean(axis=0),0,atol=1e-7)
        with tempfile.TemporaryDirectory(prefix="benchmark unit ") as directory:
            path = Path(directory)/"preprocessor.joblib"
            joblib.dump(processor,path)
            loaded = joblib.load(path)
            np.testing.assert_array_equal(loaded.transform(test),Xte)

    def test_batch_plans_are_shared_reproducible_no_replacement(self):
        a = bench.make_batches(100,10,4,21)
        b = bench.make_batches(100,10,4,21)
        for first, second in zip(a,b):
            np.testing.assert_array_equal(first,second)
            self.assertEqual(len(set(first)),10)
        self.assertEqual(bench.make_batches(100,100,3,21),[None,None,None])

    def test_telemetry_actual_calls_and_timer_fields(self):
        X = np.array([[.1,.1,.1,.1],[.3,.2,.9,.6],[.6,.5,.1,.5],[.9,.4,.3,.9]])
        rows = bench.measure_group(fake_models(),X,bench.Setting("dt_rf30_rf300",21,low1=.2,low2=.5),[None,None],{})
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]["input_n"],4)
        self.assertEqual(rows[0]["stage2_n"],2)
        self.assertEqual(rows[0]["stage2_calls"],1)
        self.assertEqual(rows[0]["stage3_calls"],0)
        self.assertGreater(rows[0]["elapsed_total_s"],0)

    def test_new_directories_never_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="benchmark paths with spaces ") as directory:
            first = bench.new_run_directory(directory)
            (first/"sentinel.txt").write_text("preserve",encoding="utf-8")
            second = bench.new_run_directory(directory)
            self.assertNotEqual(first,second)
            self.assertEqual((first/"sentinel.txt").read_text(encoding="utf-8"),"preserve")

    def test_json_represents_nan_defaults_without_invalid_json(self):
        parsed = json.loads(bench.json_text({"missing":float("nan")}))
        self.assertEqual(parsed["missing"],{"__nonfinite_float__":"NaN"})


class AnalysisTests(unittest.TestCase):
    def test_p95_withheld_for_ten_or_99_calls(self):
        self.assertIsNone(analyze.summarize_durations(np.ones(10),100)["p95_s"])
        self.assertIsNone(analyze.summarize_durations(np.ones(99),100)["p95_s"])
        self.assertEqual(analyze.summarize_durations(np.ones(100),100)["p95_s"],1.0)

    def test_rare_route_cost_not_discarded_by_mean(self):
        report = analyze.summarize_durations([.00005]*96+[.012]*4,1)
        self.assertGreater(report["mean_s"],report["median_s"]*5)

    def test_component_fit_recovers_known_cost(self):
        n = np.array([1,4,16,100,10000],dtype=float)
        par = analyze.fit_component_cost(n,.002+0.000003*n)
        self.assertAlmostEqual(par["fixed_call_s"],.002,places=10)
        self.assertAlmostEqual(par["per_row_s"],.000003,places=12)

    def test_skipped_stage_has_zero_fixed_cost(self):
        parameters = {"dt":{"fixed_call_s":1.,"per_row_s":.1},
                      "rf30":{"fixed_call_s":5.,"per_row_s":.2},
                      "rf300":{"fixed_call_s":100.,"per_row_s":1.}}
        self.assertAlmostEqual(analyze.predict_route_cost(parameters,"dt_rf30_rf300",10,2,0),7.4)
        self.assertAlmostEqual(analyze.predict_route_cost(parameters,"dt_rf30_rf300",10,0,0),2.0)
        self.assertAlmostEqual(analyze.predict_route_cost(parameters,"dt_rf30_rf300",10,2,1),108.4)

    def test_heldout_is_not_fitted_and_bad_fit_is_labeled(self):
        rows = []
        for size in (1,4,16,100):
            for _ in range(3):
                rows.append(dict(seed=21,regime="component_calibration",data_role="fit",model_key="dt",input_n=size,n_jobs=1,
                                 elapsed_total_s=.002+.000003*size,stage2_n=0,stage3_n=0,stage2_calls=0,stage3_calls=0))
        for size in (32,1000):
            for _ in range(3):
                rows.append(dict(seed=21,regime="component_calibration",data_role="holdout",model_key="dt",input_n=size,n_jobs=1,
                                 elapsed_total_s=10*(.002+.000003*size),stage2_n=0,stage3_n=0,stage2_calls=0,stage3_calls=0))
        fits, heldout, assessments = analyze.overhead_analysis(pd.DataFrame(rows))
        self.assertAlmostEqual(fits[0]["fixed_call_s"],.002,places=9)
        self.assertNotIn("32",fits[0]["fit_batch_sizes"].split(";"))
        self.assertEqual(len(heldout),2)
        self.assertEqual(assessments[0]["status"],"model_inadequate_or_insufficient")

    def test_analyzer_end_to_end_with_synthetic_csv(self):
        with tempfile.TemporaryDirectory(prefix="analysis unit ") as directory:
            path = Path(directory)
            rows=[]
            for i in range(10):
                rows.append(dict(run_id="synthetic",seed=21,regime="primary_fullbatch",data_role="primary",model_key="dt",
                                 batch_label="full",n_jobs=1,call_index=i,input_n=100,elapsed_total_s=.001+i*.000001,
                                 stage2_n=0,stage3_n=0,stage2_calls=0,stage3_calls=0,quality_status="matched_archive",config_sha256="fixed"))
            bench.write_csv(path/"raw_calls.csv",rows)
            bench.write_json(path/"run_manifest.json",dict(run_id="synthetic",status="complete",smoke=True,
                                                          manuscript_eligible=False,quality_scope="matched_archive"))
            args=analyze.parse_args(["--run-dir",str(path),"--latex"])
            output=analyze.execute(args)
            self.assertTrue((output/"benchmark_fullbatch.tex").is_file())
            summary=pd.read_csv(output/"timing_by_seed.csv")
            self.assertTrue(summary["p95_s"].isna().all())
            self.assertEqual(len(pd.read_csv(path/"raw_calls.csv")),10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
