"""Post-hoc threshold and residual diagnostics from immutable saved results.

No training, timing, test-threshold selection, or cost-model refitting occurs.
The fixed threshold grid is an explanatory display, never a selection rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from score_shift_analysis import load_artifact, metrics_from_counts
from analyze_benchmark import predict_route_cost

POLICIES = ["global_default_0p5", "global_validation_f1", "global_oracle_prior_weighted_validation_f1"]
MODELS = ["dt", "xgb300", "rf30", "rf300", "dt_xgb", "cascade", "dt_rf30", "dt_rf30_rf300"]
GRID = np.arange(60, 141, dtype=float) / 200  # 0.300..0.700, spacing 0.005
BIN_LABELS = ["0", "1", "2-4", "5-16", "17-64", "65+"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def threshold_metrics(y, score, thresholds):
    y, score, thresholds = np.asarray(y), np.asarray(score), np.asarray(thresholds, dtype=float)
    if y.ndim != 1 or score.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid aligned binary labels/scores")
    if not np.isfinite(score).all() or not np.isfinite(thresholds).all():
        raise ValueError("Nonfinite threshold or score")
    normal, attack = np.sort(score[y == 0]), np.sort(score[y == 1])
    if not len(normal) or not len(attack):
        raise ValueError("Both classes are required")
    tn = np.searchsorted(normal, thresholds, side="left")
    fn = np.searchsorted(attack, thresholds, side="left")
    return metrics_from_counts(tn, len(normal) - tn, fn, len(attack) - fn)


def pareto_flags(frame):
    """Descriptive strict dominance of mean points; no epsilon or CI claims."""
    quality = frame.F1_mean.to_numpy(float)
    cost = frame.reference_seconds.to_numpy(float)
    if not np.isfinite(quality).all() or not np.isfinite(cost).all() or (cost <= 0).any():
        raise ValueError("Invalid quality/cost point")
    return np.array([not np.any((quality >= q) & (cost <= c) & ((quality > q) | (cost < c)))
                     for q, c in zip(quality, cost)])


def stage_bin(n):
    if int(n) != n or n < 0:
        raise ValueError("Stage size must be a nonnegative integer")
    return BIN_LABELS[int(np.searchsorted([0, 1, 4, 16, 64], n, side="left"))]


def run(root):
    inputs = {}
    def track(path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Input hash mismatch: {path}")
        inputs[str(path)] = {"sha256": actual, "bytes": path.stat().st_size}
        return path
    def read(path):
        return pd.read_csv(track(path), float_precision="round_trip")
    def manifest(path):
        record = json.loads(track(path).read_text(encoding="utf-8"))
        for output in record.get("outputs", []):
            track(path.parent / Path(output.get("file") or output["path"]).name, output["sha256"])
        return record

    out = root / "generated" / "feedback_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    score_root = root / "generated" / "score_diagnostics"
    previous = manifest(score_root / "manifest.json")
    if previous["status"] != "complete":
        raise ValueError("Incomplete score analysis")
    track(Path(__file__))
    track(Path(__file__).with_name("score_shift_analysis.py"))
    track(Path(__file__).with_name("analyze_benchmark.py"))
    policy = read(score_root / "threshold_policy_by_seed.csv")
    native = read(score_root / "native_policy_by_seed.csv")
    audit = read(score_root / "input_audit.csv")
    if len(policy) != 240 or policy.duplicated(["model_key", "policy", "seed"]).any():
        raise ValueError("Expected exactly 8 models x 3 policies x 10 seeds")
    if set(policy.model_key) != set(MODELS) or set(policy.policy) != set(POLICIES):
        raise ValueError("Unexpected models or threshold policies")
    route_native = native[native.model_key.isin(MODELS[4:])]
    if len(route_native) != 40 or route_native.native_vs_default_global_changed_test_predictions.ne(0).any():
        raise ValueError("Native-route/default-cutoff identity failed")
    stats = policy.groupby(["model_key", "policy"]).agg(
        n_seeds=("seed", "nunique"), cutoff_mean=("selected_threshold", "mean"),
        cutoff_sd=("selected_threshold", "std"), cutoff_min=("selected_threshold", "min"),
        cutoff_max=("selected_threshold", "max"), F1_mean=("test_F1_weighted_pct", "mean"),
        F1_sd=("test_F1_weighted_pct", "std"), FAR_mean=("test_FAR_pct", "mean"),
        recall_mean=("test_Attack_recall_pct", "mean")).reset_index()
    if stats.n_seeds.ne(10).any():
        raise ValueError("Incomplete seed coverage")

    review_root = root / "generated" / "benchmark_review"
    review = json.loads(track(review_root / "review_manifest.json").read_text(encoding="utf-8"))
    if review["status"] != "passed" or not review["full_benchmark_completed"]:
        raise ValueError("Full benchmark review is required")
    for name, expected in review["outputs"].items():
        track(review_root / name, expected)
    timing = read(review_root / "timing_across_seeds.csv")
    timing = timing[timing.regime.eq("primary_fullbatch")]
    comparison = stats[stats.policy.eq(POLICIES[0])].merge(
        timing[["model_key", "mean_of_seed_medians_s", "sd_seed_medians_s"]], on="model_key", validate="one_to_one")
    comparison = comparison.rename(columns={"mean_of_seed_medians_s": "reference_seconds", "sd_seed_medians_s": "reference_seconds_sd"})
    comparison["nondominated_mean_point"] = pareto_flags(comparison)
    comparison["timing_scope"] = "native-policy measured reference; standalone cutoff 0.5 not separately timed"

    curves, bands = [], []
    for model in ("rf30", "rf300"):
        for seed in sorted(policy.seed.unique()):
            choices = policy[(policy.model_key == model) & (policy.seed == seed)].set_index("policy")
            cutoff = float(choices.loc[POLICIES[1], "selected_threshold"])
            for split in ("validation", "test"):
                record = audit[(audit.model_key == model) & (audit.seed == seed) & (audit.split == split)]
                if len(record) != 1:
                    raise ValueError("Ambiguous score input")
                record = record.iloc[0]
                artifact = load_artifact(track(Path(record.path), record.sha256), split)
                values = threshold_metrics(artifact.y, artifact.score.astype(float), GRID)
                for i, threshold in enumerate(GRID):
                    curves.append({"model_key": model, "seed": seed, "split": split, "threshold": threshold,
                                   "F1_weighted_pct": values["F1_weighted_pct"][i],
                                   "FAR_pct": values["FAR_pct"][i], "recall_pct": values["Attack_recall_pct"][i]})
                check = threshold_metrics(artifact.y, artifact.score.astype(float), [cutoff, .5])
                for i, name in enumerate((POLICIES[1], POLICIES[0])):
                    for metric in ("TN", "FP", "FN", "TP", "F1_weighted_pct", "FAR_pct", "Attack_recall_pct"):
                        if abs(float(check[metric][i]) - float(choices.loc[name, split + "_" + metric])) > 1e-8:
                            raise ValueError(f"Score/CSV mismatch: {model}/{seed}/{split}/{metric}")
                if not cutoff < .5:
                    raise ValueError("RF band interpretation requires validation cutoff below 0.5")
                band = (artifact.score.astype(float) >= cutoff) & (artifact.score.astype(float) < .5)
                normal = int(np.count_nonzero(band & (artifact.y == 0)))
                attack = int(np.count_nonzero(band & (artifact.y == 1)))
                if normal != int(check["FP"][0] - check["FP"][1]) or attack != int(check["FN"][1] - check["FN"][0]):
                    raise ValueError("Threshold-band count identity failed")
                bands.append({"model_key": model, "seed": seed, "split": split, "validation_cutoff": cutoff,
                              "normal_rows_reclassified": normal, "attack_rows_reclassified": attack,
                              "F1_default_minus_validation_pp": check["F1_weighted_pct"][1] - check["F1_weighted_pct"][0],
                              "FAR_default_minus_validation_pp": check["FAR_pct"][1] - check["FAR_pct"][0],
                              "recall_default_minus_validation_pp": check["Attack_recall_pct"][1] - check["Attack_recall_pct"][0]})
    curves = pd.DataFrame(curves)
    bands = pd.DataFrame(bands)
    band_summary = bands.groupby(["model_key", "split"]).agg(
        n_seeds=("seed", "nunique"), normal_rows_mean=("normal_rows_reclassified", "mean"),
        attack_rows_mean=("attack_rows_reclassified", "mean"), F1_change_mean=("F1_default_minus_validation_pp", "mean"),
        FAR_change_mean=("FAR_default_minus_validation_pp", "mean"), recall_change_mean=("recall_default_minus_validation_pp", "mean")).reset_index()

    heldout = read(review_root / "overhead_heldout_predictions.csv")
    heldout["signed_percentage_error"] = 100 * heldout.signed_error_s / heldout.observed_mean_s
    residual_summary = heldout.groupby(["kind", "model_key", "input_n"]).agg(
        n_seeds=("seed", "nunique"), signed_error_pct_mean=("signed_percentage_error", "mean"),
        signed_error_pct_sd=("signed_percentage_error", "std")).reset_index()
    fits = read(review_root / "overhead_component_fits.csv")
    raw_path = Path(review["run_directory"]) / "raw_calls.csv"
    track(raw_path, review["file_hashes"][str(raw_path)])
    raw = read(raw_path)
    subset = raw[(raw.model_key == "dt_rf30_rf300") & (raw.regime == "batch_holdout")].copy()
    if len(subset) != 200 or set(subset.input_n) != {32, 1000}:
        raise ValueError("Expected 200 DRR holdout calls")
    estimates = []
    for call in subset.itertuples():
        parameters = fits[fits.seed == call.seed].set_index("component").to_dict("index")
        estimates.append(predict_route_cost(parameters, call.model_key, call.input_n, call.stage2_n, call.stage3_n))
    subset["predicted_s"] = estimates
    subset["stage3_bin"] = subset.stage3_n.map(stage_bin)
    route_bins = subset.groupby(["seed", "input_n", "stage3_bin"], observed=True).agg(
        n_calls=("elapsed_total_s", "size"), stage3_n_mean=("stage3_n", "mean"),
        measured_mean_s=("elapsed_total_s", "mean"), predicted_mean_s=("predicted_s", "mean")).reset_index()
    route_bins["signed_error_pct"] = 100 * (route_bins.predicted_mean_s - route_bins.measured_mean_s) / route_bins.measured_mean_s
    if route_bins.n_calls.sum() != 200:
        raise ValueError("Residual bins lost calls")
    # Reconcile unbinned predictions with the original held-out analyzer.
    for (seed, size), group in subset.groupby(["seed", "input_n"]):
        original = heldout[(heldout.seed == seed) & (heldout.input_n == size) &
                           (heldout.kind == "whole_policy") & (heldout.model_key == "dt_rf30_rf300")].iloc[0]
        if not np.isclose(group.predicted_s.mean(), original.predicted_mean_s, rtol=1e-12, atol=1e-15):
            raise ValueError("Residual predictions differ from frozen cost model")

    frames = {"threshold_summary.csv": stats, "default_quality_cost.csv": comparison,
              "threshold_curves_by_seed.csv": curves, "threshold_band_by_seed.csv": bands,
              "threshold_band_summary.csv": band_summary, "residual_by_size.csv": residual_summary,
              "drr_residual_bins_by_seed.csv": route_bins}
    outputs = []
    for name, frame in frames.items():
        destination = out / name
        frame.to_csv(destination, index=False, float_format="%.17g")
        outputs.append({"file": name, "sha256": sha(destination), "rows": len(frame)})
    for path, expected in inputs.items():
        if sha(path) != expected["sha256"]:
            raise ValueError(f"Input changed during analysis: {path}")
    record = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "script": str(Path(__file__).resolve()), "script_sha256": sha(__file__),
              "input_files": [{"path": path, **value} for path, value in inputs.items()], "outputs": outputs,
              "benchmark_run_id": review["run_id"], "route_default_identity_checks": 40,
              "raw_inputs_modified": False, "new_model_execution": False, "test_threshold_selected": False,
              "cost_model_refitted": False, "residual_calls": len(subset),
              "limitations": ["Post-hoc diagnostics on an already inspected test set",
                              "Native timing is a reference, not a new standalone-0.5 measurement",
                              "DRR bins condition whole-call residuals on stage size, not measured per-stage latency",
                              "Bin means give equal weight to represented seeds; support varies across bins"]}
    (out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Feedback diagnostics complete: {len(outputs)} outputs; no fitting or new timing.")
    print(band_summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-root", type=Path, default=Path(__file__).resolve().parents[1])
    run(parser.parse_args().revision_root.resolve())
