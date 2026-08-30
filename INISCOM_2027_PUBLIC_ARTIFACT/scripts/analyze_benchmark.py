#!/usr/bin/env python3
"""Summarize a NEW benchmark run without changing its raw measurements.

Reports means and medians separately; per-seed p95 requires >=100 calls.  A
component-specific nonnegative fixed-call + per-row model is fitted ONLY on
declared calibration batches, then checked on held-out component and routed
batches (32 and 1000).  Poor fit is reported as model inadequacy, not repaired
by moving held-out batches into training or claiming a universal timing law.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid


np = None
pd = None
P95_MIN_CALLS = 100
COMPONENTS = ("dt", "xgb300", "rf30", "rf300")
STAGES = {
    "dt": ("dt",), "xgb300": ("xgb300",), "rf30": ("rf30",), "rf300": ("rf300",),
    "dt_xgb": ("dt", "xgb300"), "cascade": ("dt", "xgb300", "rf30"),
    "dt_rf30": ("dt", "rf30"), "dt_rf30_rf300": ("dt", "rf30", "rf300"),
}
LABELS = {
    "dt": "DT", "xgb300": "XGB-300", "rf30": "RF-30", "rf300": "RF-300",
    "dt_xgb": r"DT$\rightarrow$XGB-300", "cascade": "Cascade-A",
    "dt_rf30": r"DT$\rightarrow$RF-30", "dt_rf30_rf300": r"DT$\rightarrow$RF-30$\rightarrow$RF-300",
}


def load_dependencies():
    global np, pd
    if np is not None:
        return
    try:
        import numpy as numpy_module
        import pandas as pandas_module
        from scipy.optimize import nnls  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Analyzer needs the existing numpy/pandas/scipy environment; no installation is automatic") from exc
    np, pd = numpy_module, pandas_module


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n")


def write_rows(path, rows):
    rows = list(rows)
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def summarize_durations(durations, input_n, min_p95_calls=P95_MIN_CALLS):
    values = np.asarray(durations, dtype=float)
    if len(values) < 1 or not np.isfinite(values).all() or (values <= 0).any() or input_n < 1:
        raise ValueError("Invalid duration sample")
    mean, median = float(values.mean()), float(np.median(values))
    enough = len(values) >= min_p95_calls
    return {
        "n_calls": len(values), "input_n": int(input_n),
        "mean_s": mean, "median_s": median,
        "sd_s": float(values.std(ddof=1)) if len(values) > 1 else None,
        "min_s": float(values.min()), "max_s": float(values.max()),
        "p95_s": float(np.percentile(values, 95)) if enough else None,
        "p95_status": "descriptive_only" if enough else f"withheld_fewer_than_{min_p95_calls}_calls",
        "mean_per_flow_us": 1e6*mean/input_n, "median_per_flow_us": 1e6*median/input_n,
        "throughput_from_mean_flows_s": input_n/mean,
        "throughput_from_median_flows_s": input_n/median,
    }


def per_seed_summaries(raw):
    keys = ["seed", "regime", "data_role", "model_key", "batch_label", "n_jobs"]
    rows = []
    for key, group in raw.groupby(keys, sort=False, dropna=False):
        if group["input_n"].nunique() != 1:
            raise ValueError(f"Mixed input sizes in telemetry group: {key}")
        summary = summarize_durations(group["elapsed_total_s"], int(group["input_n"].iloc[0]))
        total_rows = float(group["input_n"].sum())
        rows.append({
            **dict(zip(keys, key)), **summary,
            "stage2_pct_of_rows": 100*float(group["stage2_n"].sum())/total_rows,
            "stage3_pct_of_rows": 100*float(group["stage3_n"].sum())/total_rows,
            "stage2_nonempty_call_fraction": float(group["stage2_calls"].mean()),
            "stage3_nonempty_call_fraction": float(group["stage3_calls"].mean()),
            "quality_status": "matched_archive" if group["quality_status"].eq("matched_archive").all() else "new_run_quality",
        })
    return rows


def across_seed_summaries(seed_rows):
    frame = pd.DataFrame(seed_rows)
    rows = []
    keys = ["regime", "data_role", "model_key", "batch_label", "n_jobs"]
    for key, group in frame.groupby(keys, sort=False, dropna=False):
        if group["seed"].duplicated().any():
            raise ValueError("Duplicate seed-level summary")
        n = len(group)
        row = {**dict(zip(keys, key)), "n_seeds": n,
               "mean_of_seed_means_s": float(group["mean_s"].mean()),
               "mean_of_seed_medians_s": float(group["median_s"].mean()),
               "mean_of_seed_mean_per_flow_us": float(group["mean_per_flow_us"].mean()),
               "mean_of_seed_median_per_flow_us": float(group["median_per_flow_us"].mean()),
               "sd_seed_means_s": float(group["mean_s"].std(ddof=1)) if n > 1 else None,
               "sd_seed_medians_s": float(group["median_s"].std(ddof=1)) if n > 1 else None,
               "stage2_mean_nonempty_call_fraction": float(group["stage2_nonempty_call_fraction"].mean()),
               "stage3_mean_nonempty_call_fraction": float(group["stage3_nonempty_call_fraction"].mean()),
               "aggregation": "equal weight per seed; p95 is never pooled across seeds"}
        rows.append(row)
    return rows


def fit_component_cost(batch_sizes, mean_durations):
    """Nonnegative T(n)=fixed_call + per_row*n; each batch size has equal weight."""
    from scipy.optimize import nnls
    sizes = np.asarray(batch_sizes, dtype=float)
    durations = np.asarray(mean_durations, dtype=float)
    if len(sizes) < 3 or len(np.unique(sizes)) < 3 or (sizes <= 0).any():
        raise ValueError("At least three distinct positive fit batch sizes are required")
    if not np.isfinite(durations).all() or (durations <= 0).any():
        raise ValueError("Invalid calibration durations")
    scale = float(sizes.max())
    design = np.column_stack([np.ones(len(sizes)), sizes/scale])
    params, _ = nnls(design, durations)
    predicted = design @ params
    error = predicted - durations
    return {
        "fixed_call_s": float(params[0]), "per_row_s": float(params[1]/scale),
        "fit_batch_count": len(sizes), "fit_batch_sizes": ";".join(str(int(n)) for n in sizes),
        "fit_rmse_s": float(np.sqrt(np.mean(error**2))),
        "fit_mean_absolute_percentage_error": float(100*np.mean(np.abs(error)/durations)),
        "fit_max_absolute_percentage_error": float(100*np.max(np.abs(error)/durations)),
        "fit_weighting": "equal weight per batch-size mean, not per raw call",
    }


def predict_route_cost(parameters, model, input_n, stage2_n, stage3_n):
    stages = STAGES[model]
    ns = (int(input_n), int(stage2_n), int(stage3_n))
    if ns[0] < 1 or ns[1] < 0 or ns[2] < 0 or ns[2] > ns[1] or ns[1] > ns[0]:
        raise ValueError("Invalid route counts")
    result = 0.0
    for index, component in enumerate(stages):
        # Fixed overhead is charged once for a NONEMPTY component call, not
        # multiplied by routed fraction and not charged for skipped stages.
        if ns[index] > 0:
            par = parameters[component]
            result += par["fixed_call_s"] + par["per_row_s"]*ns[index]
    return result


def overhead_analysis(raw, max_mape=20.0):
    fits, heldout, assessments = [], [], []
    parameters = {}
    calibration = raw.loc[raw["regime"].eq("component_calibration") & raw["n_jobs"].eq(1)]
    for (seed, component), group in calibration.groupby(["seed", "model_key"], sort=False):
        fit = group.loc[group["data_role"].eq("fit")].groupby("input_n", sort=True)["elapsed_total_s"].mean()
        if len(fit) < 3:
            assessments.append({"seed": int(seed), "model_key": component, "status": "insufficient_fit_batches"})
            continue
        par = fit_component_cost(fit.index.to_numpy(), fit.to_numpy())
        fits.append({"seed": int(seed), "component": component, **par})
        parameters.setdefault(int(seed), {})[component] = par
        test = group.loc[group["data_role"].eq("holdout")]
        for size, subset in test.groupby("input_n", sort=True):
            observed = float(subset["elapsed_total_s"].mean())
            predicted = par["fixed_call_s"] + par["per_row_s"]*int(size)
            heldout.append({
                "seed": int(seed), "kind": "component", "model_key": component, "input_n": int(size),
                "n_calls": len(subset), "observed_mean_s": observed, "predicted_mean_s": predicted,
                "signed_error_s": predicted-observed,
                "absolute_percentage_error": 100*abs(predicted-observed)/observed,
            })
    # Strictly held-out sizes only; primary or fit-size rows are not reused as a test.
    test = raw.loc[raw["regime"].eq("batch_holdout") & raw["data_role"].eq("holdout") & raw["n_jobs"].eq(1)]
    for (seed, model, size), group in test.groupby(["seed", "model_key", "input_n"], sort=False):
        pars = parameters.get(int(seed), {})
        if any(component not in pars for component in STAGES[model]):
            continue
        predicted = [predict_route_cost(pars, model, row.input_n, row.stage2_n, row.stage3_n)
                     for row in group.itertuples()]
        observed = float(group["elapsed_total_s"].mean())
        estimate = float(np.mean(predicted))
        heldout.append({
            "seed": int(seed), "kind": "whole_policy", "model_key": model, "input_n": int(size),
            "n_calls": len(group), "observed_mean_s": observed, "predicted_mean_s": estimate,
            "signed_error_s": estimate-observed,
            "absolute_percentage_error": 100*abs(estimate-observed)/observed,
            "stage2_nonempty_call_fraction": float(group["stage2_calls"].mean()),
            "stage3_nonempty_call_fraction": float(group["stage3_calls"].mean()),
        })
    if heldout:
        table = pd.DataFrame(heldout)
        for (kind, model), group in table.groupby(["kind", "model_key"], sort=False):
            enough = group.groupby("seed")["input_n"].nunique().ge(2).all()
            mape = float(group["absolute_percentage_error"].mean())
            maximum = float(group["absolute_percentage_error"].max())
            status = "descriptively_adequate_on_tested_batches" if enough and mape <= max_mape and maximum <= 2.5*max_mape else "model_inadequate_or_insufficient"
            assessments.append({
                "kind": kind, "model_key": model, "n_seeds": int(group["seed"].nunique()),
                "heldout_batch_sizes": ";".join(str(int(n)) for n in sorted(group["input_n"].unique())),
                "mean_absolute_percentage_error": mape, "max_absolute_percentage_error": maximum,
                "threshold_mape_percent": max_mape, "threshold_max_error_percent": 2.5*max_mape,
                "status": status,
                "interpretation": "descriptive predeclared-size check only; no universal or causal timing law",
            })
    return fits, heldout, assessments


def validate_raw(raw):
    required = {"seed", "regime", "data_role", "model_key", "batch_label", "n_jobs", "input_n",
                "elapsed_total_s", "stage2_n", "stage3_n", "stage2_calls", "stage3_calls", "quality_status"}
    if missing := required - set(raw.columns):
        raise ValueError(f"Missing raw telemetry columns: {sorted(missing)}")
    if raw.empty:
        raise ValueError("No raw calls")
    for col in ("seed", "n_jobs", "input_n", "stage2_n", "stage3_n", "stage2_calls", "stage3_calls"):
        raw[col] = pd.to_numeric(raw[col], errors="raise").astype(int)
    raw["elapsed_total_s"] = pd.to_numeric(raw["elapsed_total_s"], errors="raise")
    if not np.isfinite(raw["elapsed_total_s"]).all() or (raw["elapsed_total_s"] <= 0).any():
        raise ValueError("Non-finite/nonpositive duration")
    if not raw["stage2_calls"].eq(raw["stage2_n"].gt(0).astype(int)).all():
        raise ValueError("Stage-2 nonempty call counts inconsistent with row counts")
    if not raw["stage3_calls"].eq(raw["stage3_n"].gt(0).astype(int)).all():
        raise ValueError("Stage-3 nonempty call counts inconsistent with row counts")
    if (raw["stage2_n"] > raw["input_n"]).any() or (raw["stage3_n"] > raw["stage2_n"]).any():
        raise ValueError("Impossible routed counts")
    return raw


def make_latex(path, rows, manifest):
    full = [r for r in rows if r["regime"] == "primary_fullbatch" and r["n_jobs"] == 1]
    lookup = {row["model_key"]: row for row in full}
    text = [
        "% Generated from THIS run's raw calls; never pair with old timing/quality silently.",
        "% Means and medians are both equal-weight summaries over seeds.",
        "% No p95 is reported from only ten full-batch repetitions per seed.",
        "% Run: " + manifest["run_id"],
        "% Quality scope: " + manifest.get("quality_scope", "unknown"),
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Model & Seeds & Mean (ms) & Median (ms) \\", r"\midrule",
    ]
    for key in STAGES:
        if key in lookup:
            row = lookup[key]
            text.append(f"{LABELS[key]} & {row['n_seeds']} & {1000*row['mean_of_seed_means_s']:.3f} & {1000*row['mean_of_seed_medians_s']:.3f} " + r"\\")
    text.extend([r"\bottomrule", r"\end{tabular}", ""])
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(text))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="Timestamped directory produced by benchmark_revision.py")
    parser.add_argument("--overhead-max-mape", type=float, default=20.0,
                        help="Predeclared descriptive adequacy threshold (default20%%); report changes, do not tune against holdout")
    parser.add_argument("--latex", action="store_true", help="Generate an optional full-batch LaTeX tabular")
    parser.add_argument("--allow-incomplete", action="store_true", help="Analyze partial failed-run calls, explicitly non-publication")
    args = parser.parse_args(argv)
    if not math.isfinite(args.overhead_max_mape) or args.overhead_max_mape <= 0:
        parser.error("--overhead-max-mape must be finite and positive")
    return args


def execute(args):
    load_dependencies()
    run_dir = args.run_dir.resolve()
    with (run_dir / "run_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "complete" and not args.allow_incomplete:
        raise ValueError("Run is incomplete/failed. Investigate first; --allow-incomplete is descriptive only")
    raw_path = run_dir / "raw_calls.csv"
    raw = validate_raw(pd.read_csv(raw_path, keep_default_na=False))
    if "run_id" in raw and not raw["run_id"].eq(manifest["run_id"]).all():
        raise ValueError("Raw calls mix different run IDs")
    if "config_sha256" in raw:
        counts = raw.groupby(["seed", "model_key"])["config_sha256"].nunique()
        if (counts > 1).any():
            raise ValueError("Raw calls mix different configurations within a seed/model")
    # Verify the frozen raw artifact when the producer supplied its inventory.
    inventory_path = run_dir / "output_hashes.json"
    if inventory_path.is_file():
        with inventory_path.open(encoding="utf-8") as handle:
            inventory = json.load(handle)
        matches = [entry for entry in inventory if str(entry["path"]).replace("\\", "/").endswith("/raw_calls.csv")]
        if len(matches) != 1 or matches[0]["sha256"] != sha256_file(raw_path):
            raise ValueError("Raw-call hash differs from the completed run inventory")
    out_dir = run_dir / "analysis" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8])
    out_dir.mkdir(parents=True, exist_ok=False)
    seed_rows = per_seed_summaries(raw)
    across = across_seed_summaries(seed_rows)
    fits, heldout, assessments = overhead_analysis(raw, args.overhead_max_mape)
    write_rows(out_dir / "timing_by_seed.csv", seed_rows)
    write_rows(out_dir / "timing_across_seeds.csv", across)
    write_rows(out_dir / "overhead_component_fits.csv", fits)
    write_rows(out_dir / "overhead_heldout_predictions.csv", heldout)
    write_rows(out_dir / "overhead_assessment.csv", assessments)
    warnings = [
        "Mean-of-seed-medians is not mean latency; rare routed calls can disappear from the median.",
        "p95 is withheld below 100 repetitions per seed; >=100 does not guarantee a precise tail estimate.",
        "Overhead fit omits explicit mask/allocation/cache/residual-difficulty terms; test errors may show inadequacy.",
        "Repeated calls on one host are not independent hardware environments; no cross-hardware inference.",
    ]
    if manifest.get("smoke"):
        warnings.append("SMOKE RUN: exact full-size fitting but short one-seed timing; NOT manuscript evidence.")
    if manifest.get("status") != "complete":
        warnings.append("INCOMPLETE RUN: descriptive debugging only, not publication evidence.")
    if manifest.get("quality_scope") != "matched_archive":
        warnings.append("Use new_run_quality.csv with these times; archived quality is NOT a matching source.")
    if not fits or not heldout:
        warnings.append("No adequate component/held-out records; overhead hypothesis has NOT been tested.")
    inadequate = any(row.get("status") != "descriptively_adequate_on_tested_batches" for row in assessments)
    if inadequate:
        warnings.append("The two-parameter component model is inadequate/insufficient for at least one tested case; do not call it a predictive law.")
    report = {
        "run_id": manifest["run_id"], "run_status": manifest.get("status"),
        "quality_scope": manifest.get("quality_scope"),
        "manuscript_eligible": bool(manifest.get("manuscript_eligible")) and manifest.get("status") == "complete",
        "source_raw_calls": str(raw_path), "source_raw_sha256": sha256_file(raw_path),
        "source_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
        "analyzer_sha256": sha256_file(Path(__file__)),
        "raw_call_count": len(raw), "per_seed_summary_count": len(seed_rows),
        "p95_min_calls": P95_MIN_CALLS, "overhead_max_mape_percent": args.overhead_max_mape,
        "overhead_fit_count": len(fits), "overhead_holdout_count": len(heldout),
        "overhead_any_inadequate": inadequate,
        "warnings": warnings,
    }
    write_json(out_dir / "analysis_manifest.json", report)
    with (out_dir / "NOTES_VI.md").open("x", encoding="utf-8") as handle:
        handle.write("# Ghi chú sử dụng kết quả timing\n\n")
        handle.write("Mỗi kết quả thuộc đúng run này; không ghép timing mới với quality cũ khi đối chiếu không khớp.\n\n")
        handle.write("- `timing_by_seed.csv`: mean, median, p95 có điều kiện và xác suất tầng được gọi.\n")
        handle.write("- `timing_across_seeds.csv`: mỗi seed có trọng số bằng nhau; mean và median được giữ riêng.\n")
        handle.write("- `overhead_heldout_predictions.csv`: kiểm tra trên batch 32/1000 không dùng để khớp mô hình.\n")
        handle.write("- `overhead_assessment.csv`: kết quả kiểm tra, không phải chứng minh nhân quả hay quy luật phổ quát.\n\n")
        handle.write("\n".join("- " + warning for warning in warnings) + "\n")
    if args.latex:
        make_latex(out_dir / "benchmark_fullbatch.tex", across, manifest)
    print(f"ANALYSIS_DIRECTORY={out_dir}")
    print(f"Calls={len(raw)}; overhead fits={len(fits)}; held-out comparisons={len(heldout)}")
    for warning in warnings:
        print("NOTE: " + warning)
    return out_dir


def main(argv=None):
    try:
        execute(parse_args(argv))
        return 0
    except Exception as exc:
        print(f"ANALYSIS STOPPED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
