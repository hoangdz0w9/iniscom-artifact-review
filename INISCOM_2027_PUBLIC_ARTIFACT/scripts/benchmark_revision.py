#!/usr/bin/env python3
"""Reproducible CPU benchmark of the final, frozen UNSW routing policies.

Default: full 10-seed primary control plus five-seed batch/parallel/overhead
sensitivity.  --smoke fits ONE real, full-size seed with short timing repeats;
it is not a reduced-data result and is never manuscript eligible.  --self-test
uses synthetic unit tests only and does not read the research data or fit models.
No existing experimental artifact is overwritten.  No package is installed.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
import uuid


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PACKAGE_ROOT / "vendor"
SEEDS = (21, 42, 84, 168, 336, 1365468824, 1486213216, 362797685, 883000906, 348184071)
SENSITIVITY_SEEDS = SEEDS[:5]
COMPONENTS = ("dt", "xgb300", "rf30", "rf300")
MODELS = COMPONENTS + ("dt_xgb", "cascade", "dt_rf30", "dt_rf30_rf300")
STAGES = {
    "dt": ("dt",), "xgb300": ("xgb300",), "rf30": ("rf30",), "rf300": ("rf300",),
    "dt_xgb": ("dt", "xgb300"), "cascade": ("dt", "xgb300", "rf30"),
    "dt_rf30": ("dt", "rf30"), "dt_rf30_rf300": ("dt", "rf30", "rf300"),
}
THREAD_KEYS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    "OMP_DYNAMIC", "OMP_PROC_BIND", "KMP_AFFINITY", "JOBLIB_TEMP_FOLDER",
)
RAW_FIELDS = (
    "run_id", "seed", "regime", "data_role", "model_key", "batch_label", "n_jobs",
    "call_index", "input_n", "elapsed_total_s", "stage1_component", "stage2_component",
    "stage3_component", "stage1_n", "stage2_n", "stage3_n", "stage1_calls",
    "stage2_calls", "stage3_calls", "quality_status", "config_sha256", "batch_plan_key",
)
np = None
pd = None


def load_dependencies():
    global np, pd
    if np is not None:
        return
    try:
        import numpy as numpy_module
        import pandas as pandas_module
        import joblib  # noqa: F401
        import scipy  # noqa: F401
        import sklearn  # noqa: F401
        import threadpoolctl  # noqa: F401
        import xgboost  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Missing scientific Python dependency. Use your existing research Python "
            "environment (numpy, pandas, scipy, scikit-learn, xgboost, joblib, threadpoolctl). "
            "This script does not install anything. Original error: " + str(exc)
        ) from exc
    np, pd = numpy_module, pandas_module
    if str(VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(VENDOR_ROOT))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        # XGBoost's legitimate default missing=np.nan must remain explicit.
        return {"__nonfinite_float__": "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def json_text(value):
    return json.dumps(json_safe(value), ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n"


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json_text(value))


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def new_run_directory(output_root):
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = output_root / name
    run_dir.mkdir(exist_ok=False)
    return run_dir


@dataclass(frozen=True)
class Setting:
    model_key: str
    seed: int
    cutoff: float | None = None
    low1: float | None = None
    low2: float | None = None
    source: str = ""

    def as_dict(self):
        return dict(model_key=self.model_key, seed=self.seed, cutoff=self.cutoff,
                    low1=self.low1, low2=self.low2, source=self.source)

    def digest(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()


def required_float(row, key, upper=1.0):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing/invalid frozen parameter {key}: {row}") from exc
    if not math.isfinite(value) or value < 0 or value > upper:
        raise ValueError(f"Frozen parameter outside [0,{upper}]: {key}={value}")
    return value


def settings_from_tables(original, closed, seeds):
    """Use the original table ONLY for standalone/two-stage configurations."""
    result = {}
    for seed in seeds:
        for model in MODELS:
            if model in {"cascade", "dt_rf30_rf300"}:
                rows = [r for r in closed if r.get("architecture") == model and int(r["seed"]) == seed]
                if len(rows) != 1:
                    raise ValueError(f"Expected one FINAL closed gate row for {model}/{seed}, got {len(rows)}")
                setting = Setting(model, seed, low1=required_float(rows[0], "threshold_dt_low", .5),
                                  low2=required_float(rows[0], "threshold_stage2_low", .5),
                                  source="final_gap_closure/threshold_extended_selected_by_seed.csv")
            else:
                rows = [r for r in original if r.get("model_key") == model and int(r["seed"]) == seed]
                if len(rows) != 1:
                    raise ValueError(f"Expected one original selected row for {model}/{seed}, got {len(rows)}")
                row = rows[0]
                if any(row.get(k) != v for k, v in {
                    "dataset": "UNSW-NB15", "protocol": "official_train_test", "mode_raw": "no_reduction"
                }.items()):
                    raise ValueError(f"Wrong experiment protocol in frozen table for {model}/{seed}")
                source = "teacher_final_official_10seeds_FINAL/all_results.csv"
                setting = (Setting(model, seed, cutoff=required_float(row, "selected_threshold"), source=source)
                           if model in COMPONENTS else
                           Setting(model, seed, low1=required_float(row, "threshold_dt_low", .5), source=source))
            result[(seed, model)] = setting
    return result


def component_score(model, X):
    proba = model.predict_proba(X)
    # Match teacher_final_revision.predict_proba_score, including float64 gates.
    return np.asarray(proba[:, 1], dtype=float)


def predict_policy(models, X, setting):
    """Actual serving path. Indexing, masks and component calls remain inside."""
    stages = STAGES[setting.model_key]
    first = component_score(models[stages[0]], X)
    if len(stages) == 1:
        return (first >= setting.cutoff).astype(np.int8), first, 0, 0
    n = len(first)
    pred = np.empty(n, dtype=np.int8)
    score = np.empty(n, dtype=float)
    normal1 = first <= setting.low1
    attack1 = first >= 1.0 - setting.low1
    stop1 = normal1 | attack1
    pred[normal1], pred[attack1] = 0, 1
    score[stop1] = first[stop1]
    idx2 = np.flatnonzero(~stop1)
    n3 = 0
    if len(idx2):
        second = component_score(models[stages[1]], X[idx2])
        if len(stages) == 2:
            pred[idx2] = (second >= .5).astype(np.int8)
            score[idx2] = second
        else:
            normal2 = second <= setting.low2
            attack2 = second >= 1.0 - setting.low2
            stop2 = normal2 | attack2
            # At low2=.5 an exact .5 is Attack, as in the archived runner.
            pred[idx2[normal2]], pred[idx2[attack2]] = 0, 1
            score[idx2[stop2]] = second[stop2]
            idx3 = idx2[~stop2]
            n3 = len(idx3)
            if n3:
                third = component_score(models[stages[2]], X[idx3])
                pred[idx3] = (third >= .5).astype(np.int8)
                score[idx3] = third
    return pred, score, len(idx2), n3


def route_counts_from_scores(scores, setting):
    """Counts from the saved float32 arrays, preserving archived boundaries."""
    stages = STAGES[setting.model_key]
    if len(stages) == 1:
        return 0, 0
    first = np.asarray(scores[stages[0]])
    idx2 = np.flatnonzero(~((first <= setting.low1) | (first >= 1.0 - setting.low1)))
    if len(stages) == 2 or not len(idx2):
        return len(idx2), 0
    second = np.asarray(scores[stages[1]])[idx2]
    n3 = int((~((second <= setting.low2) | (second >= 1.0 - setting.low2))).sum())
    return len(idx2), n3


def compare_prediction(y_true, row_ids, pred, score, n2, n3, archived, expected_counts, atol):
    if not np.array_equal(y_true, archived["y_true"]):
        raise ValueError("Archived label pairing mismatch; --allow-quality-change cannot waive this")
    if "test_row_id" in archived and not np.array_equal(row_ids, archived["test_row_id"]):
        raise ValueError("Archived test_row_id mismatch; refusing a misaligned comparison")
    if score.shape != archived["y_score"].shape or not np.isfinite(score).all():
        raise ValueError("Invalid/new score shape or non-finite scores")
    error = np.abs(np.asarray(score, dtype=float) - np.asarray(archived["y_score"], dtype=float))
    pred_mismatch = int((pred != archived["y_pred"]).sum()) if "y_pred" in archived else None
    counts_ok = expected_counts is None or (n2, n3) == tuple(expected_counts)
    score_mismatch = int((error > atol).sum())
    passed = score_mismatch == 0 and (pred_mismatch is None or pred_mismatch == 0) and counts_ok
    return {
        "status": "matched_archive" if passed else "new_run_quality",
        "rows": len(y_true), "score_atol": atol,
        "score_mismatch_rows": score_mismatch, "max_abs_score_error": float(error.max(initial=0)),
        "prediction_mismatch_rows": pred_mismatch,
        "new_stage2_n": n2, "new_stage3_n": n3,
        "archived_stage2_n": None if expected_counts is None else expected_counts[0],
        "archived_stage3_n": None if expected_counts is None else expected_counts[1],
        "route_counts_match": counts_ok,
    }


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def metrics(y, pred, score):
    from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef
    normal = y == 0
    attack = y == 1
    return {
        "F1_weighted": 100.0 * float(f1_score(y, pred, average="weighted", zero_division=0)),
        "F1_macro": 100.0 * float(f1_score(y, pred, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(y, pred)),
        "FAR": 100.0 * float(pred[normal].mean()),
        "Attack_recall": 100.0 * float(pred[attack].mean()),
        "PR_AUC": 100.0 * float(average_precision_score(y, score)),
    }


def command_capture(command, timeout=20):
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=timeout,
                                   shell=False)
        return {"command": command, "returncode": completed.returncode,
                "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def capture_environment(run_dir, args, original_thread_environment):
    from threadpoolctl import threadpool_info
    report = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "THIS new benchmark only; does not establish the 2026 original-run environment",
        "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "thread_environment_before": original_thread_environment,
        "thread_environment_effective": {key: os.environ.get(key) for key in THREAD_KEYS},
        "threadpools_after_import": threadpool_info(),
        "packages": {name: importlib.metadata.version(name) for name in (
            "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib", "threadpoolctl"
        )},
        "arguments": vars(args),
        "power_policy_modified_by_script": False,
        "thread_policy": "BLAS=1; OpenMP and model n_jobs explicitly set to 1 or parallel_jobs per regime",
    }
    try:
        import psutil
        process = psutil.Process()
        battery = psutil.sensors_battery()
        report["hardware"] = {
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "memory_total_bytes": psutil.virtual_memory().total,
            "cpu_frequency": None if psutil.cpu_freq() is None else psutil.cpu_freq()._asdict(),
            "cpu_affinity": process.cpu_affinity() if hasattr(process, "cpu_affinity") else None,
            "process_priority": process.nice(),
            "battery": None if battery is None else battery._asdict(),
            "psutil_version": psutil.__version__,
        }
    except (ImportError, OSError, AttributeError) as exc:
        report["hardware_probe_error"] = str(exc)
    if os.name == "nt":
        report["windows_cpu"] = command_capture([
            "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress",
        ])
        report["power_active_scheme"] = command_capture(["powercfg", "/GETACTIVESCHEME"])
        report["power_processor_settings"] = command_capture(["powercfg", "/QUERY", "SCHEME_CURRENT", "SUB_PROCESSOR"])
    freeze = command_capture([sys.executable, "-m", "pip", "freeze", "--all"], timeout=30)
    report["pip_freeze_status"] = {key: value for key, value in freeze.items() if key != "stdout"}
    with (run_dir / "pip_freeze.txt").open("x", encoding="utf-8") as handle:
        handle.write(freeze.get("stdout", "") + "\n")
    write_json(run_dir / "environment.json", report)
    if freeze.get("returncode") != 0:
        raise RuntimeError("Could not capture exact pip freeze. Inspect environment.json; no model training started.")
    return report


@contextlib.contextmanager
def thread_regime(models, jobs):
    from threadpoolctl import threadpool_limits
    for name in ("rf30", "rf300", "xgb300"):
        models[name].set_params(n_jobs=jobs)
    with threadpool_limits(limits=1, user_api="blas"):
        with threadpool_limits(limits=jobs, user_api="openmp"):
            yield


def make_batches(n, batch_size, repeats, seed):
    if batch_size > n or batch_size < 1 or repeats < 1:
        raise ValueError("Invalid batch plan")
    if batch_size == n:
        return [None] * repeats
    rng = np.random.default_rng(seed)
    return [rng.choice(n, size=batch_size, replace=False).astype(np.int64) for _ in range(repeats)]


def measure_group(models, X, setting, batches, context, warmup_rows=4096, component_only=False):
    """Return buffered telemetry: CSV writes and input materialization are untimed."""
    first_batch = X if batches[0] is None else X[batches[0]]
    if component_only:
        call = lambda array: (None, component_score(models[setting.model_key], array), 0, 0)
    else:
        call = lambda array: predict_policy(models, array, setting)
    call(first_batch[:min(len(first_batch), warmup_rows)])
    call(first_batch)
    rows = []
    stages = STAGES[setting.model_key]
    for index, batch in enumerate(batches):
        # External batch preparation/indexing explicitly lies OUTSIDE the timer.
        Xb = X if batch is None else X[batch]
        started = time.perf_counter_ns()
        pred, score, n2, n3 = call(Xb)
        elapsed = (time.perf_counter_ns() - started) / 1e9
        # Result checks, telemetry formatting and I/O are also outside the timer.
        if len(score) != len(Xb) or elapsed <= 0:
            raise RuntimeError("Invalid measured result or clock")
        rows.append({
            **context, "model_key": setting.model_key, "call_index": index,
            "input_n": len(Xb), "elapsed_total_s": elapsed,
            "stage1_component": stages[0],
            "stage2_component": stages[1] if len(stages) > 1 else "",
            "stage3_component": stages[2] if len(stages) > 2 else "",
            "stage1_n": len(Xb), "stage2_n": n2, "stage3_n": n3,
            "stage1_calls": 1, "stage2_calls": int(n2 > 0), "stage3_calls": int(n3 > 0),
            "config_sha256": setting.digest(),
        })
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path("DATA_ROOT"), help="Root containing UNSW CSVs and FINAL3")
    parser.add_argument("--output-root", type=Path, help="Parent for a NEW timestamped run (default: package generated/benchmarks)")
    parser.add_argument("--seed", type=int, action="append", help="Repeat to choose a subset; default is the archived ten seeds")
    parser.add_argument("--smoke", action="store_true", help="Fit one real full-size seed, short repeats; never manuscript eligible")
    parser.add_argument("--self-test", action="store_true", help="Synthetic unit tests only; no data or model fitting")
    parser.add_argument("--parallel-jobs", type=int, default=8, help="Fixed model workers for full-batch sensitivity (default 8)")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Only primary full-batch single-thread control")
    parser.add_argument("--score-atol", type=float, default=1e-7, help="Tolerance against saved float32 scores; predictions/routes must match exactly")
    parser.add_argument("--allow-quality-change", action="store_true", help="Explicitly keep changed predictions as NEW-RUN quality; never blend with archived quality")
    parser.add_argument("--analyze", action="store_true", help="After completion, run the companion analyzer and generate LaTeX tables")
    args = parser.parse_args(argv)
    if args.parallel_jobs < 2:
        parser.error("--parallel-jobs must be >=2 (single-thread is already a separate control)")
    if not math.isfinite(args.score_atol) or args.score_atol < 0:
        parser.error("--score-atol must be finite and nonnegative")
    if args.seed and (len(args.seed) != len(set(args.seed)) or any(seed not in SEEDS for seed in args.seed)):
        parser.error("--seed must list distinct archived seeds")
    if args.smoke and args.seed and len(args.seed) != 1:
        parser.error("--smoke accepts exactly one --seed")
    return args


def required_paths(project_root, seeds):
    final = project_root / "FINAL3"
    original = final / "teacher_final_official_10seeds_FINAL"
    gap = final / "Teacher_Final_Revision_STAGE2" / "final_gap_closure"
    paths = [project_root / "UNSW_NB15_training-set.csv", project_root / "UNSW_NB15_testing-set.csv",
             original / "run_config.json", original / "all_results.csv",
             gap / "threshold_extended_selected_by_seed.csv"]
    for seed in seeds:
        for model in MODELS:
            directory = gap / "final_predictions" if model in {"cascade", "dt_rf30_rf300"} else original / "predictions"
            paths.append(directory / f"{model}__seed_{seed}.npz")
        paths.extend(original / "validation_scores" / f"{model}__seed_{seed}.npz" for model in COMPONENTS)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required archive/data artifact is missing: {path}")
    return original, gap, paths


def capture_input_manifest(paths):
    return [{"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths]


def save_batch_plan(run_dir, seed, key, batches, input_n):
    path = run_dir / "batch_plans" / f"seed_{seed}__{key}.npz"
    path.parent.mkdir(exist_ok=True)
    if batches[0] is None:
        np.savez_compressed(path, full_input_n=np.array([input_n]), repeats=np.array([len(batches)]))
    else:
        np.savez_compressed(path, indices=np.stack(batches))


def execute(args):
    original_threads = {key: os.environ.get(key) for key in THREAD_KEYS}
    # Process-local limits only. We never change the user's persistent settings.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    load_dependencies()
    from benchmark_helpers import LegacyPreprocessor, build_components, read_official_csv
    import joblib
    from sklearn.model_selection import train_test_split
    from threadpoolctl import threadpool_info

    seeds = tuple(args.seed or ((21,) if args.smoke else SEEDS))
    project_root = args.project_root.resolve()
    original, gap, paths = required_paths(project_root, seeds)
    settings = settings_from_tables(load_csv(original / "all_results.csv"),
                                    load_csv(gap / "threshold_extended_selected_by_seed.csv"), seeds)
    if not args.skip_sensitivity and args.parallel_jobs > (os.cpu_count() or 1):
        raise ValueError("--parallel-jobs exceeds available logical CPUs; choose a fixed smaller value")
    run_dir = new_run_directory(args.output_root or PACKAGE_ROOT / "generated" / "benchmarks")
    print(f"RUN_DIRECTORY={run_dir}", flush=True)
    print("New immutable-source run; archived primary measurements will not be replaced.", flush=True)
    start = time.perf_counter()
    run_id = run_dir.name
    manifest = {
        "schema_version": 1, "run_id": run_id, "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
        "seeds": list(seeds), "sensitivity_seeds": [s for s in seeds if s in SENSITIVITY_SEEDS],
        "policy": "final_expanded_grid_frozen_no_retuning", "same_run_models": list(MODELS),
        "smoke": args.smoke, "sample_fraction": 1.0,
        "quality_scope": "matched_archive_until_checked", "manuscript_eligible": False,
        "timed_region": "component score generation + decision thresholding + routing masks + internal sub-batch slicing; per-call counts returned",
        "excluded_region": "training, preprocessing, external input batching, result checks, CSV/JSON output, telemetry formatting",
        "profiling": "No per-stage clock instrumentation in headline measurements; component costs measured separately",
        "p95_policy": "Only report a per-seed p95 if at least 100 calls; this is descriptive, not a precision guarantee",
        "warmup_policy": "Two untimed calls per group: first min(batch,4096) rows, then the complete first input batch",
        "overhead_fit_batches": [1, 4, 16, 64, 100, 10000, "full"],
        "overhead_holdout_batches": [32, 1000],
    }
    write_json(run_dir / "run_started.json", manifest)
    quality_rows, validation_rows, parallel_quality_rows, training_rows, audits, saved_models, thread_records = [], [], [], [], [], [], []
    raw_count = 0
    matched = True
    try:
        capture_environment(run_dir, args, original_threads)
        own_sources = [Path(__file__), VENDOR_ROOT / "benchmark_helpers" / "__init__.py"]
        legacy_dir = project_root / "FINAL3" / "Teacher_Final_Revision_STAGE2" / "scripts"
        legacy = [legacy_dir / name for name in ("teacher_final_revision.py", "week1_repeated_seed_experiment.py", "week2_baseline_experiment.py")]
        write_json(run_dir / "input_manifest.json", {
            "inputs": capture_input_manifest(paths), "executed_sources": capture_input_manifest(own_sources),
            "reference_legacy_sources_not_imported": capture_input_manifest([p for p in legacy if p.is_file()]),
        })
        write_json(run_dir / "frozen_settings.json", [settings[key].as_dict() for key in settings])
        train_full, dropped_train = read_official_csv(paths[0])
        test_full, dropped_test = read_official_csv(paths[1])
        if len(train_full) != 175341 or len(test_full) != 82332:
            raise ValueError("Official dataset row counts differ from the locked protocol (175341/82332)")
        write_json(run_dir / "data_record.json", {
            "official_train_rows": len(train_full), "official_test_rows": len(test_full),
            "invalid_label_rows_train": dropped_train, "invalid_label_rows_test": dropped_test,
            "test_row_id_definition": "zero-based row index after original label cleaning; NOT raw id",
        })
        with (run_dir / "raw_calls.csv").open("x", newline="", encoding="utf-8") as raw_handle:
            writer = csv.DictWriter(raw_handle, fieldnames=RAW_FIELDS)
            writer.writeheader()
            for seed in seeds:
                print(f"seed={seed}: fitting four shared components once; no gate search", flush=True)
                random.seed(seed)
                np.random.seed(seed)
                train, val = train_test_split(train_full, test_size=.2, random_state=seed,
                                             stratify=train_full["label"], shuffle=True)
                train, val = train.reset_index(drop=True), val.reset_index(drop=True)
                processor = LegacyPreprocessor()
                Xtr = processor.fit_transform(train)
                Xv, Xte = processor.transform(val), processor.transform(test_full)
                ytr = train["label"].to_numpy(dtype=int)
                yv, yte = val["label"].to_numpy(dtype=int), test_full["label"].to_numpy(dtype=int)
                ids = test_full["_row_id"].to_numpy(dtype=np.int64)
                seed_dir = run_dir / "models" / f"seed_{seed}"
                seed_dir.mkdir(parents=True)
                processor_path = seed_dir / "preprocessor.joblib"
                joblib.dump(processor, processor_path, compress=3)
                write_json(seed_dir / "preprocessing.json", {**processor.metadata(), "train_rows": len(ytr), "validation_rows": len(yv)})
                np.savez_compressed(seed_dir / "split_row_ids.npz", train_row_id=train["_row_id"].to_numpy(),
                                    validation_row_id=val["_row_id"].to_numpy(), test_row_id=ids)
                saved_models.append({"path": str(processor_path), "sha256": sha256_file(processor_path)})
                models = build_components(seed)
                with thread_regime(models, 1):
                    for component in COMPONENTS:
                        fit_start = time.perf_counter()
                        models[component].fit(Xtr, ytr)
                        duration = time.perf_counter() - fit_start
                        if not np.array_equal(models[component].classes_, np.array([0, 1])):
                            raise ValueError("Unexpected component class order")
                        training_rows.append({"seed": seed, "component": component, "train_seconds": duration,
                                              "train_rows": len(ytr), "feature_count": Xtr.shape[1], "n_jobs": 1})
                        model_path = seed_dir / f"{component}.joblib"
                        joblib.dump(models[component], model_path, compress=3)
                        saved_models.append({"path": str(model_path), "sha256": sha256_file(model_path)})
                        write_json(seed_dir / f"{component}_params.json", models[component].get_params())
                        if component == "xgb300":
                            native_path = seed_dir / "xgb300.ubj"
                            models[component].save_model(native_path)
                            saved_models.append({"path": str(native_path), "sha256": sha256_file(native_path)})
                    del Xtr
                    archive_components = {m: load_npz(original / "predictions" / f"{m}__seed_{seed}.npz") for m in COMPONENTS}
                    archive_scores = {m: archive_components[m]["y_score"] for m in COMPONENTS}
                    for component in COMPONENTS:
                        archived_val = load_npz(original / "validation_scores" / f"{component}__seed_{seed}.npz")
                        new_score = component_score(models[component], Xv)
                        audit = compare_prediction(yv, None, np.zeros(len(yv)), new_score, 0, 0, archived_val, None, args.score_atol)
                        audits.append({"seed": seed, "model_key": component, "split": "validation", **audit})
                    pred_dir = run_dir / "predictions" / f"seed_{seed}"
                    pred_dir.mkdir(parents=True)
                    for model in MODELS:
                        setting = settings[(seed, model)]
                        pred, score, n2, n3 = predict_policy(models, Xte, setting)
                        directory = gap / "final_predictions" if model in {"cascade", "dt_rf30_rf300"} else original / "predictions"
                        archived = load_npz(directory / f"{model}__seed_{seed}.npz")
                        audit = compare_prediction(yte, ids, pred, score, n2, n3, archived,
                                                   route_counts_from_scores(archive_scores, setting), args.score_atol)
                        audits.append({"seed": seed, "model_key": model, "split": "test", **audit})
                        quality_rows.append({"seed": seed, "model_key": model, **metrics(yte, pred, score),
                                             "stage2_n": n2, "stage3_n": n3, "stage2_pct": 100*n2/len(yte),
                                             "stage3_pct": 100*n3/len(yte), "quality_status": audit["status"],
                                             "config_sha256": setting.digest()})
                        np.savez_compressed(pred_dir / f"{model}.npz", y_true=yte.astype(np.int8), y_pred=pred,
                                            y_score=score.astype(np.float32), y_score_native=score, test_row_id=ids)
                        pv, sv, n2v, n3v = predict_policy(models, Xv, setting)
                        validation_rows.append({"seed": seed, "model_key": model, **metrics(yv, pv, sv),
                                                "stage2_n": n2v, "stage3_n": n3v})
                    seed_audits = [row for row in audits if row["seed"] == seed]
                    seed_match = all(row["status"] == "matched_archive" for row in seed_audits)
                    matched = matched and seed_match
                    seed_quality = "matched_archive" if seed_match else "new_run_quality"
                    if not seed_match and not args.allow_quality_change:
                        raise RuntimeError(
                            f"seed={seed}: score/prediction/route mismatch. No timing for this seed was accepted. "
                            "See reproduction_checks.csv. Investigate first; --allow-quality-change explicitly "
                            "starts a separate NEW-RUN-quality experiment, never an archived-quality replacement."
                        )
                    print(f"seed={seed}: reproduction={seed_quality}; benchmarking frozen policies", flush=True)
                    order = np.random.default_rng(seed ^ 0x43AC).permutation(MODELS).tolist()

                    def run_plan(regime, role, label, batch_n, repeats, jobs, component_only=False):
                        nonlocal raw_count
                        key = f"{regime}_{role}_{label}_j{jobs}"
                        plan_seed = seed ^ int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                        batches = make_batches(len(Xte), batch_n, repeats, plan_seed)
                        save_batch_plan(run_dir, seed, key, batches, len(Xte))
                        context = dict(run_id=run_id, seed=seed, regime=regime, data_role=role, batch_label=label,
                                       n_jobs=jobs, quality_status=seed_quality, batch_plan_key=key)
                        for model in (COMPONENTS if component_only else order):
                            rows = measure_group(models, Xte, settings[(seed, model)], batches, context,
                                                 component_only=component_only)
                            writer.writerows(rows)
                            raw_handle.flush()
                            raw_count += len(rows)
                        print(f"  {regime}: batch={label}, jobs={jobs}, calls/model={repeats}", flush=True)

                    thread_records.append({"seed": seed, "regime": "single_thread", "jobs": 1,
                                           "threadpools": threadpool_info(), "model_order": order})
                    run_plan("primary_fullbatch", "primary", "full", len(Xte), 2 if args.smoke else 10, 1)
                    if not args.skip_sensitivity and seed in SENSITIVITY_SEEDS:
                        for label, batch_n, count in (("1", 1, 500), ("100", 100, 100),
                                                     ("10000", 10000, 10), ("full", len(Xte), 10)):
                            repeats = (8 if batch_n == 1 else 5 if batch_n == 100 else 2) if args.smoke else count
                            run_plan("batch_sensitivity", "descriptive", label, batch_n, repeats, 1)
                        # These batch sizes are locked in advance and NEVER used for model fitting.
                        for batch_n in (32, 1000):
                            run_plan("batch_holdout", "holdout", str(batch_n), batch_n, 3 if args.smoke else 20, 1)
                        for batch_n in (1, 4, 16, 64, 100, 10000, len(Xte), 32, 1000):
                            label = "full" if batch_n == len(Xte) else str(batch_n)
                            role = "holdout" if batch_n in (32, 1000) else "fit"
                            count = 20 if batch_n <= 1000 else 10
                            run_plan("component_calibration", role, label, batch_n, 2 if args.smoke else count, 1, True)
                if not args.skip_sensitivity and seed in SENSITIVITY_SEEDS:
                    with thread_regime(models, args.parallel_jobs):
                        thread_records.append({"seed": seed, "regime": "parallel_fullbatch", "jobs": args.parallel_jobs,
                                               "threadpools": threadpool_info()})
                        # Check parallel quality too: summation/order changes must be recorded.
                        for model in MODELS:
                            setting = settings[(seed, model)]
                            pred, score, n2, n3 = predict_policy(models, Xte, setting)
                            single = load_npz(pred_dir / f"{model}.npz")
                            single_quality = next(row for row in quality_rows if row["seed"] == seed and row["model_key"] == model)
                            audit = compare_prediction(yte, ids, pred, score, n2, n3, single,
                                                       (single_quality["stage2_n"], single_quality["stage3_n"]), args.score_atol)
                            audits.append({"seed": seed, "model_key": model, "split": "parallel_vs_new_single", **audit})
                            parallel_quality_rows.append({"seed": seed, "model_key": model, "n_jobs": args.parallel_jobs,
                                                          **metrics(yte, pred, score), "stage2_n": n2, "stage3_n": n3,
                                                          "vs_new_single_status": audit["status"],
                                                          "config_sha256": setting.digest()})
                            if audit["status"] != "matched_archive":
                                parallel_pred_dir = run_dir / "parallel_predictions" / f"seed_{seed}"
                                parallel_pred_dir.mkdir(parents=True, exist_ok=True)
                                np.savez_compressed(parallel_pred_dir / f"{model}.npz", y_true=yte.astype(np.int8),
                                                    y_pred=pred, y_score=score.astype(np.float32), y_score_native=score,
                                                    test_row_id=ids)
                            if audit["status"] != "matched_archive":
                                matched = False
                                if not args.allow_quality_change:
                                    raise RuntimeError(f"Parallel prediction mismatch for {model}/{seed}; inspect reproduction_checks.csv")
                                seed_quality = "new_run_quality"
                        run_plan("parallel_fullbatch", "sensitivity", "full", len(Xte), 2 if args.smoke else 10, args.parallel_jobs)
                del models, Xte, Xv
        manifest.update(status="complete", quality_scope="matched_archive" if matched else "new_run_quality",
                        manuscript_eligible=not args.smoke and set(seeds) == set(SEEDS),
                        archived_quality_reusable=matched and not args.smoke)
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc), error_type=type(exc).__name__,
                        quality_scope="incomplete_or_changed", manuscript_eligible=False, archived_quality_reusable=False)
        with (run_dir / "error_traceback.txt").open("x", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        raise
    finally:
        write_csv(run_dir / "reproduction_checks.csv", audits)
        write_csv(run_dir / "new_run_quality.csv", quality_rows)
        write_csv(run_dir / "new_run_validation_quality.csv", validation_rows)
        write_csv(run_dir / "new_run_parallel_quality.csv", parallel_quality_rows)
        write_csv(run_dir / "training_times.csv", training_rows)
        write_json(run_dir / "model_manifest.json", saved_models)
        write_json(run_dir / "thread_regimes.json", thread_records)
        manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                        wall_seconds=time.perf_counter()-start, raw_call_count=raw_count)
        write_json(run_dir / "run_manifest.json", manifest)
        inventory = [p for p in run_dir.rglob("*") if p.is_file() and p.name != "output_hashes.json"]
        write_json(run_dir / "output_hashes.json", capture_input_manifest(inventory))
    print(f"COMPLETE: {run_dir}", flush=True)
    print("Use NEW-run quality and timing together. Do not silently replace archived values.", flush=True)
    return run_dir


def run_self_test():
    import unittest
    load_dependencies()
    tests = unittest.defaultTestLoader.discover(str(PACKAGE_ROOT / "tests"), pattern="test_benchmark_revision.py")
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    try:
        run_dir = execute(args)
        if args.analyze:
            command = [sys.executable, str(PACKAGE_ROOT / "scripts" / "analyze_benchmark.py"),
                       "--run-dir", str(run_dir), "--latex"]
            subprocess.run(command, check=True, shell=False)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"BENCHMARK STOPPED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
