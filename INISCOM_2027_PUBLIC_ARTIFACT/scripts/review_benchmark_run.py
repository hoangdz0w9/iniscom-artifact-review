"""Audit a completed full benchmark and recompute its report without refitting.

Raw run files are read-only. The output is a new, explicit review directory;
failed overhead prediction is retained as a result, not a reason to retune.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef

import analyze_benchmark as analyzer

MODELS = tuple(analyzer.STAGES)
SEEDS = (21, 42, 84, 168, 336, 348184071, 362797685, 883000906, 1365468824, 1486213216)
SENSITIVITY = (21, 42, 84, 168, 336)
GROUP_KEYS = ["seed", "regime", "data_role", "model_key", "batch_label", "n_jobs"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_groups():
    result = {}
    def add(seed, regime, role, label, repeats, jobs=1, models=MODELS):
        for model in models:
            result[(seed, regime, role, model, label, jobs)] = repeats
    for seed in SEEDS:
        add(seed, "primary_fullbatch", "primary", "full", 10)
        if seed in SENSITIVITY:
            for label, count in (("1", 500), ("100", 100), ("10000", 10), ("full", 10)):
                add(seed, "batch_sensitivity", "descriptive", label, count)
            for label in ("32", "1000"):
                add(seed, "batch_holdout", "holdout", label, 20)
            for label in ("1", "4", "16", "64", "100", "10000", "full", "32", "1000"):
                role = "holdout" if label in ("32", "1000") else "fit"
                add(seed, "component_calibration", role, label, 10 if label in ("10000", "full") else 20,
                    models=analyzer.COMPONENTS)
            add(seed, "parallel_fullbatch", "sensitivity", "full", 10, jobs=8)
    return result


def verify_groups(raw):
    actual = raw.groupby(GROUP_KEYS, sort=False).size().to_dict()
    expected = expected_groups()
    require(actual == expected, "Missing, duplicated or unexpected seed/model/batch/repetition groups")
    for key, group in raw.groupby(GROUP_KEYS, sort=False):
        require(sorted(group.call_index.tolist()) == list(range(expected[key])), f"Bad call indices: {key}")
        size = 82332 if key[4] == "full" else int(key[4])
        require(group.input_n.eq(size).all(), f"Input-size mismatch: {key}")
    return len(actual)


def route_masks(scores, setting):
    stages = analyzer.STAGES[setting["model_key"]]
    n = len(scores[stages[0]])
    masks = [np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)]
    if len(stages) > 1:
        score = scores[stages[0]]
        masks[0] = (score > setting["low1"]) & (score < 1 - setting["low1"])
        if len(stages) > 2:
            score = scores[stages[1]]
            masks[1] = masks[0] & (score > setting["low2"]) & (score < 1 - setting["low2"])
    return masks


def calculate_quality(data):
    y, pred, score = data["y_true"], data["y_pred"], data["y_score_native"]
    return {"F1_weighted": 100*f1_score(y, pred, average="weighted", zero_division=0),
            "F1_macro": 100*f1_score(y, pred, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y, pred),
            "FAR": 100*pred[y == 0].mean(), "Attack_recall": 100*pred[y == 1].mean(),
            "PR_AUC": 100*average_precision_score(y, score)}


def compare_frames(expected, actual, keys):
    expected = expected.sort_values(keys).reset_index(drop=True)
    actual = actual.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_exact=False, rtol=1e-11, atol=1e-12)


def run_review(run_dir: Path, output: Path, analysis_dir: Path | None = None, overwrite: bool = False):
    analyzer.load_dependencies()
    run_dir, output = run_dir.resolve(), output.resolve()
    require(not output.is_relative_to(run_dir), "Keep computed reviews outside the immutable raw-run directory")
    require(not output.exists() or overwrite, "Review output already exists; choose a new directory or --overwrite for computed review files only")
    manifest = read_json(run_dir / "run_manifest.json")
    require(manifest["status"] == "complete" and not manifest["smoke"], "Not a completed full run")
    require(set(manifest["seeds"]) == set(SEEDS) and set(manifest["sensitivity_seeds"]) == set(SENSITIVITY), "Seed set differs")
    require(manifest["sample_fraction"] == 1 and manifest["policy"] == "final_expanded_grid_frozen_no_retuning", "Wrong data/gate protocol")
    require(manifest["quality_scope"] == "matched_archive" and manifest["archived_quality_reusable"], "Changed-quality run requires separate review")
    require(not manifest["arguments"]["allow_quality_change"] and manifest["arguments"]["score_atol"] == 1e-7, "Unexpected quality override")
    require(manifest["arguments"]["parallel_jobs"] == 8 and not manifest["arguments"]["skip_sensitivity"], "Wrong thread/sensitivity protocol")
    hashes = {}
    input_manifest = read_json(run_dir / "input_manifest.json")
    inventory = read_json(run_dir / "output_hashes.json")
    sources = [item for entries in input_manifest.values() for item in entries]
    for record in sources + inventory:
        path = Path(record["path"]).resolve()
        actual = hashes.setdefault(str(path), analyzer.sha256_file(path))
        require(actual == record["sha256"] and path.stat().st_size == record.get("bytes", path.stat().st_size), f"Source/artifact changed: {path}")
    print(f"Verified {len(sources)} input/source records and {len(inventory)} run artifacts", flush=True)
    raw = analyzer.validate_raw(pd.read_csv(run_dir / "raw_calls.csv", keep_default_na=False))
    require(len(raw) == manifest["raw_call_count"] == 30800 and raw.run_id.eq(manifest["run_id"]).all(), "Run ID/call count mismatch")
    require(raw.quality_status.eq("matched_archive").all(), "A timed group has unmatched quality")
    require(raw.stage1_n.eq(raw.input_n).all() and raw.stage1_calls.eq(1).all(), "Invalid first-stage telemetry")
    group_count = verify_groups(raw)
    settings = {(r["seed"], r["model_key"]): r for r in read_json(run_dir / "frozen_settings.json")}
    require(set(settings) == {(s, m) for s in SEEDS for m in MODELS}, "Frozen setting coverage differs")
    for (seed, model), group in raw.groupby(["seed", "model_key"], sort=False):
        digest = hashlib.sha256(json.dumps(settings[seed, model], sort_keys=True).encode()).hexdigest()
        require(group.config_sha256.eq(digest).all(), f"Configuration hash mismatch: {seed}/{model}")
        stages = analyzer.STAGES[model]
        for i in range(3):
            require(group[f"stage{i+1}_component"].eq(stages[i] if i < len(stages) else "").all(), "Stage component mismatch")
    checks = pd.read_csv(run_dir / "reproduction_checks.csv")
    expected_checks = {(s, m, "test") for s in SEEDS for m in MODELS}
    expected_checks |= {(s, m, "validation") for s in SEEDS for m in analyzer.COMPONENTS}
    expected_checks |= {(s, m, "parallel_vs_new_single") for s in SENSITIVITY for m in MODELS}
    require(len(checks) == 160 and set(checks[["seed", "model_key", "split"]].itertuples(index=False, name=None)) == expected_checks, "Reproduction-check coverage differs")
    require(checks.status.eq("matched_archive").all() and checks.route_counts_match.eq(True).all(), "Failed reproduction check")
    require(checks.score_mismatch_rows.eq(0).all() and checks.prediction_mismatch_rows.dropna().eq(0).all(), "Prediction/score mismatch")
    require(checks.score_atol.eq(1e-7).all() and checks.max_abs_score_error.le(1e-7).all(), "Score tolerance failed")
    quality = pd.read_csv(run_dir / "new_run_quality.csv")
    require(len(quality) == 80 and not quality.duplicated(["seed", "model_key"]).any(), "Quality table coverage differs")
    project_root = Path(manifest["arguments"]["project_root"])
    archived = project_root / "FINAL3" / "teacher_final_official_10seeds_FINAL"
    final_predictions = project_root / "FINAL3" / "Teacher_Final_Revision_STAGE2" / "final_gap_closure" / "final_predictions"
    audit_rows, route_rows = [], 0
    for seed in SEEDS:
        data = {}
        for model in MODELS:
            with np.load(run_dir / "predictions" / f"seed_{seed}" / f"{model}.npz", allow_pickle=False) as npz:
                data[model] = dict(npz)
            new = data[model]
            reference_dir = final_predictions if model in {"cascade", "dt_rf30_rf300"} else archived / "predictions"
            with np.load(reference_dir / f"{model}__seed_{seed}.npz", allow_pickle=False) as ref:
                for field in ("y_true", "y_pred", "test_row_id"):
                    require(np.array_equal(new[field], ref[field]), f"Independent {field} comparison failed: {model}/{seed}")
                require(np.max(np.abs(new["y_score_native"] - ref["y_score"].astype(float))) <= 1e-7, "Independent score comparison failed")
            require(np.array_equal(new["test_row_id"], np.arange(82332)), "Test row index differs")
            calculated = calculate_quality(new)
            reported = quality.loc[(quality.seed == seed) & (quality.model_key == model)].iloc[0]
            error = max(abs(float(reported[k]) - float(v)) for k, v in calculated.items())
            require(error < 1e-9, f"Quality recomputation failed: {model}/{seed}")
            audit_rows.append({"seed": seed, "model_key": model, "max_quality_abs_error": error, "status": "passed"})
        scores = {m: data[m]["y_score_native"] for m in analyzer.COMPONENTS}
        masks = {m: route_masks(scores, settings[seed, m]) for m in MODELS}
        for plan_key, plan_group in raw.loc[raw.seed.eq(seed)].groupby("batch_plan_key", sort=False):
            with np.load(run_dir / "batch_plans" / f"seed_{seed}__{plan_key}.npz", allow_pickle=False) as plan:
                rows = len(plan_group) // plan_group.model_key.nunique()
                if "indices" in plan:
                    indices = plan["indices"]
                    require(len(indices) == rows and np.all((indices >= 0) & (indices < 82332)), "Bad batch plan indices")
                    require(all(len(np.unique(index)) == len(index) for index in indices), "Repeated rows inside no-replacement batch")
                else:
                    require(int(plan["full_input_n"][0]) == 82332 and int(plan["repeats"][0]) == rows, "Bad full-batch plan")
                    indices = None
                for model, group in plan_group.groupby("model_key", sort=False):
                    group = group.sort_values("call_index")
                    for i, mask in enumerate(masks[model]):
                        counts = np.full(rows, mask.sum()) if indices is None else mask[indices].sum(axis=1)
                        require(np.array_equal(counts, group[f"stage{i+2}_n"].to_numpy()), f"Per-call route count differs from saved scores/plan: {seed}/{model}/{plan_key}")
                    route_rows += len(group)
        print(f"Recomputed quality and all timed route counts: seed {seed}", flush=True)
    parallel = pd.read_csv(run_dir / "new_run_parallel_quality.csv")
    require(len(parallel) == 40 and parallel.vs_new_single_status.eq("matched_archive").all(), "Parallel quality check failed")
    for _, row in parallel.iterrows():
        single = quality.loc[(quality.seed == row.seed) & (quality.model_key == row.model_key)].iloc[0]
        for column in ("F1_weighted", "F1_macro", "MCC", "FAR", "Attack_recall", "stage2_n", "stage3_n"):
            require(abs(row[column] - single[column]) < 1e-9, "Parallel quality differs")
    threads = read_json(run_dir / "thread_regimes.json")
    require(len(threads) == 15, "Thread record count differs")
    for record in threads:
        expected_jobs = 8 if record["regime"] == "parallel_fullbatch" else 1
        require(record["jobs"] == expected_jobs, "Thread policy differs")
        for pool in record["threadpools"]:
            expected = 1 if pool["user_api"] == "blas" else expected_jobs
            require(pool["num_threads"] == expected, "Recorded thread pool violates protocol")
    seed_rows = analyzer.per_seed_summaries(raw)
    across = analyzer.across_seed_summaries(seed_rows)
    fits, heldout, assessment = analyzer.overhead_analysis(raw, max_mape=20.0)
    if analysis_dir is None:
        candidates = list((run_dir / "analysis").glob("*/analysis_manifest.json"))
        require(len(candidates) == 1, "Select --analysis-dir explicitly when the run has zero or multiple analyses")
        original_analysis = candidates[0].parent.resolve()
    else:
        original_analysis = analysis_dir.resolve()
    require(original_analysis.is_relative_to(run_dir / "analysis"), "Analysis does not belong to this run")
    original_manifest = read_json(original_analysis / "analysis_manifest.json")
    require(original_manifest["run_id"] == manifest["run_id"] and
            original_manifest["source_raw_sha256"] == analyzer.sha256_file(run_dir / "raw_calls.csv"), "Original analysis has different source data")
    require(original_manifest["analyzer_sha256"] == analyzer.sha256_file(Path(analyzer.__file__)), "Analyzer source changed since original summary")
    compare_frames(pd.DataFrame(seed_rows), pd.read_csv(original_analysis / "timing_by_seed.csv"), GROUP_KEYS)
    compare_frames(pd.DataFrame(across), pd.read_csv(original_analysis / "timing_across_seeds.csv"), GROUP_KEYS[1:])
    compare_frames(pd.DataFrame(fits), pd.read_csv(original_analysis / "overhead_component_fits.csv"), ["seed", "component"])
    compare_frames(pd.DataFrame(heldout), pd.read_csv(original_analysis / "overhead_heldout_predictions.csv"), ["seed", "kind", "model_key", "input_n"])
    compare_frames(pd.DataFrame(assessment), pd.read_csv(original_analysis / "overhead_assessment.csv"), ["kind", "model_key"])
    environment = read_json(run_dir / "environment.json")
    for path in original_analysis.iterdir():
        if path.is_file():
            hashes[str(path)] = analyzer.sha256_file(path)
    output.mkdir(parents=True, exist_ok=overwrite)
    outputs = {}
    tables = {"timing_by_seed.csv": seed_rows, "timing_across_seeds.csv": across,
              "overhead_component_fits.csv": fits, "overhead_heldout_predictions.csv": heldout,
              "overhead_assessment.csv": assessment, "independent_quality_audit.csv": audit_rows}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(output / name, index=False, float_format="%.17g")
        outputs[name] = analyzer.sha256_file(output / name)
    review = {
        "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest["run_id"], "run_directory": str(run_dir), "original_analysis_directory": str(original_analysis),
        "run_manifest_sha256": analyzer.sha256_file(run_dir / "run_manifest.json"),
        "script_sha256": analyzer.sha256_file(Path(__file__)), "analyzer_sha256": analyzer.sha256_file(Path(analyzer.__file__)),
        "input_source_records_checked": len(sources), "run_artifacts_checked": len(inventory), "file_hashes": hashes,
        "reproduction_checks": len(checks), "max_score_error": float(checks.max_abs_score_error.max()),
        "quality_rows_independently_recomputed": 80, "per_call_routes_independently_recomputed": route_rows,
        "raw_calls": len(raw), "complete_groups": group_count, "overhead_component_fits": len(fits), "heldout_comparisons": len(heldout),
        "overhead_assessments_passed": sum(r["status"] == "descriptively_adequate_on_tested_batches" for r in assessment),
        "overhead_assessments_total": len(assessment), "overhead_criteria": "mean APE <=20%, max APE <=50%, two held-out sizes per seed",
        "environment": {"cpu": json.loads(environment["windows_cpu"]["stdout"])["Name"],
                        "physical_cores": environment["hardware"]["physical_cpu_count"],
                        "logical_cpus": environment["hardware"]["logical_cpu_count"],
                        "ram_bytes": environment["hardware"]["memory_total_bytes"],
                        "power_plugged_at_start": environment["hardware"]["battery"]["power_plugged"],
                        "power_scheme": environment["power_active_scheme"]["stdout"], "packages": environment["packages"]},
        "limitations": ["Power, temperature and competing load were not continuously recorded; startup snapshot is not proof of stable conditions throughout.",
                        "One host; repeated calls and seeds do not represent independent deployments/hardware.",
                        "Negative overhead validation is retained; no tuning on holdout.",
                        "Validation and parallel reproduction checks are hashed run records; independent saved-NPZ recomputation covers test predictions and all route telemetry, not a new model execution."],
        "full_benchmark_completed": True, "timing_usable_for_descriptive_same_host_results": True,
        "submission_ready_claimed": False, "outputs": outputs,
    }
    (output / "review_manifest.json").write_text(json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in review.items() if k not in {"file_hashes", "environment", "outputs"}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, help="Existing analyzer output; inferred only when exactly one exists")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate this script's computed review files, never raw measurements")
    args = parser.parse_args()
    run_review(args.run_dir, args.output_dir, args.analysis_dir, args.overwrite)


if __name__ == "__main__":
    main()
