#!/usr/bin/env python3
"""Post-hoc diagnostic: why does the pre-specified linear cost model fail?

Reviewer R1.3 asked why a fixed-invocation-plus-row latency model fails in
practice. This script answers from the archived raw timing calls only; nothing
is re-measured, and the pre-specified verdict (all 12 held-out assessments
fail) is NOT replaced.

Steps
1. Reproduce the archived assessment exactly with ``analyze_benchmark.py``
   (absolute-time NNLS, equal weight per calibration batch-size mean) and
   check it against ``generated/benchmark_review/overhead_assessment.csv``.
2. Report what the archived fit implies: fitted fixed-call costs versus the
   measured single-row call time, and in-sample percentage error.
3. Refit the SAME two-parameter model on the SAME calibration sizes with one
   change of objective: minimize over a >= 0, b >= 0
       sum_k [ (a + b*n_k - Tbar_k) / Tbar_k ]^2
   where n_k is the calibration batch size and Tbar_k the mean measured time
   at that size (weighted least squares with weights 1/Tbar_k^2). Re-apply the same held-out sizes,
   route accounting and adequacy rule (MAPE <= 20%, max <= 50%).

Step 3 was designed after the held-out failure had been inspected. It is a
diagnostic of the failure mechanism, not a confirmatory model, and it does not
show that the latency model transfers to other hosts or execution regimes.

Inputs are recorded timings of the archived benchmark run (one row per timed
call: batch size, rows reaching each stage, elapsed seconds). They contain no
network-traffic records. Nothing is re-measured and no NIDS model is retrained.

Usage
  Revision tree (writes generated/cost_model_diagnostic and
  generated/tables/cost_model_weighting.tex):
      python scripts/cost_model_weighting_diagnostic.py
  Public artifact (reads results/, writes reproduced/ and compares the result
  with the shipped results/cost_model_diagnostic and Table 17):
      python scripts/cost_model_weighting_diagnostic.py
  Any layout:
      python scripts/cost_model_weighting_diagnostic.py --raw-calls R.csv \
          --assessment A.csv --out-dir OUT --table OUT/table.tex \
          [--expected-dir DIR --expected-table T.tex]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
RUN_ID = "20260827T122554_803811Z_edd09c6d"
spec = importlib.util.spec_from_file_location("analyze_benchmark", HERE / "analyze_benchmark.py")
ab = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ab
spec.loader.exec_module(ab)
ab.load_dependencies()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_relative(sizes, durations):
    """Same nonnegative model; minimizes sum_k ((a + b*n_k - T_k)/T_k)^2."""
    sizes = np.asarray(sizes, float)
    durations = np.asarray(durations, float)
    scale = float(sizes.max())
    design = np.column_stack([np.ones(len(sizes)), sizes / scale])
    weights = 1.0 / durations
    params, _ = nnls(design * weights[:, None], durations * weights)
    predicted = design @ params
    ape = 100 * np.abs(predicted - durations) / durations
    return {"fixed_call_s": float(params[0]), "per_row_s": float(params[1] / scale),
            "fit_mean_absolute_percentage_error": float(ape.mean()),
            "fit_max_absolute_percentage_error": float(ape.max())}


def heldout_errors(raw, parameters):
    rows = []
    calibration = raw.loc[raw["regime"].eq("component_calibration") & raw["n_jobs"].eq(1)]
    for (seed, component), group in calibration.groupby(["seed", "model_key"], sort=False):
        par = parameters[int(seed)][component]
        for size, subset in group.loc[group["data_role"].eq("holdout")].groupby("input_n", sort=True):
            observed = float(subset["elapsed_total_s"].mean())
            predicted = par["fixed_call_s"] + par["per_row_s"] * int(size)
            rows.append({"seed": int(seed), "kind": "component", "model_key": component, "input_n": int(size),
                         "observed_mean_s": observed, "predicted_mean_s": predicted,
                         "signed_error_pct": 100 * (predicted - observed) / observed})
    test = raw.loc[raw["regime"].eq("batch_holdout") & raw["data_role"].eq("holdout") & raw["n_jobs"].eq(1)]
    for (seed, model, size), group in test.groupby(["seed", "model_key", "input_n"], sort=False):
        pars = parameters[int(seed)]
        predicted = np.mean([ab.predict_route_cost(pars, model, r.input_n, r.stage2_n, r.stage3_n)
                             for r in group.itertuples()])
        observed = float(group["elapsed_total_s"].mean())
        rows.append({"seed": int(seed), "kind": "whole_policy", "model_key": model, "input_n": int(size),
                     "observed_mean_s": observed, "predicted_mean_s": float(predicted),
                     "signed_error_pct": 100 * (predicted - observed) / observed})
    table = pd.DataFrame(rows)
    table["absolute_percentage_error"] = table.signed_error_pct.abs()
    return table


def assess(table):
    out = table.groupby(["kind", "model_key"], sort=False).agg(
        mape=("absolute_percentage_error", "mean"), max_ape=("absolute_percentage_error", "max")).reset_index()
    out["adequate"] = (out.mape <= 20) & (out.max_ape <= 50)
    return out


def validate_archived_assessment(reproduced, archived, heldout):
    """Require the same 12 verdicts, errors and complete seed/holdout coverage."""
    keys = ["kind", "model_key"]
    expected_keys = ({("component", model) for model in ab.COMPONENTS}
                     | {("whole_policy", model) for model in ab.STAGES})
    expected_pairs = {(seed, size) for seed in (21, 42, 84, 168, 336)
                      for size in (32, 1000)}
    numeric = ["mean_absolute_percentage_error", "max_absolute_percentage_error",
               "threshold_mape_percent", "threshold_max_error_percent"]
    metadata = ["n_seeds", "heldout_batch_sizes", "status"]
    for label, table in (("reproduced", reproduced), ("archived", archived)):
        if (not set(keys + numeric + metadata).issubset(table.columns)
                or table.duplicated(keys).any()
                or set(table[keys].itertuples(index=False, name=None)) != expected_keys):
            raise ValueError(f"Invalid {label} cost-model assessment coverage")
        if (not np.isfinite(table[numeric].to_numpy(float)).all()
                or (table[numeric] < 0).any().any()
                or not table.n_seeds.eq(5).all()
                or not table.heldout_batch_sizes.eq("32;1000").all()
                or not table.threshold_mape_percent.eq(20).all()
                or not table.threshold_max_error_percent.eq(50).all()):
            raise ValueError(f"Invalid {label} cost-model assessment metadata")
        adequate = ((table.mean_absolute_percentage_error <= 20)
                    & (table.max_absolute_percentage_error <= 50))
        if adequate.any() or not table.status.eq("model_inadequate_or_insufficient").all():
            raise ValueError(f"The {label} assessment does not retain all 12 failed verdicts")

    heldout = pd.DataFrame(heldout)
    if (not set(keys + ["seed", "input_n", "n_calls"]).issubset(heldout.columns)
            or heldout.duplicated(keys + ["seed", "input_n"]).any()
            or set(heldout[keys].itertuples(index=False, name=None)) != expected_keys):
        raise ValueError("Invalid held-out cost-model comparison coverage")
    for _, group in heldout.groupby(keys, sort=False):
        if (set(group[["seed", "input_n"]].itertuples(index=False, name=None)) != expected_pairs
                or not group.n_calls.eq(20).all()):
            raise ValueError("Each assessment requires five seeds and both 20-call held-out batches")

    merged = reproduced.merge(archived, on=keys, suffixes=("", "_archived"), validate="one_to_one")
    for column in numeric:
        if not np.allclose(merged[column], merged[column + "_archived"], atol=1e-9, rtol=0):
            raise ValueError(f"Archived cost-model {column} was not reproduced; stop")
    for column in metadata:
        if not merged[column].equals(merged[column + "_archived"]):
            raise ValueError(f"Archived cost-model {column} was not reproduced; stop")
    return merged


def relative_name(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), root)).as_posix()


def layout_defaults(root: Path) -> dict:
    """Revision tree (generated/) or public artifact (results/)."""
    if (root / "generated").is_dir():
        return {"raw": root / "generated" / "benchmarks" / RUN_ID / "raw_calls.csv",
                "assessment": root / "generated" / "benchmark_review" / "overhead_assessment.csv",
                "out": root / "generated" / "cost_model_diagnostic",
                "table": root / "generated" / "tables" / "cost_model_weighting.tex",
                "expected_dir": None, "expected_table": None}
    return {"raw": root / "results" / "benchmark_timing" / "raw_calls.csv",
            "assessment": root / "results" / "benchmark_review" / "overhead_assessment.csv",
            "out": root / "reproduced" / "cost_model_diagnostic",
            "table": root / "reproduced" / "cost_model_weighting.tex",
            "expected_dir": root / "results" / "cost_model_diagnostic",
            "expected_table": root / "paper_assets" / "tables" / "cost_model_weighting.tex"}


COMPARED_CSVS = ("component_fit_comparison.csv", "component_fit_comparison_by_seed.csv",
                 "heldout_weighting_comparison.csv")


def compare_outputs(outdir: Path, table_path: Path, expected_dir, expected_table) -> list[str]:
    """Compare regenerated outputs with the shipped ones.

    CSVs are compared value by value (identical columns and text; numbers to a
    relative tolerance of 1e-9, since the last binary digits of floating-point
    sums can differ between machines and BLAS builds). The LaTeX table, which
    holds the displayed rounded values, must match byte for byte.
    """
    checked = []
    if expected_dir is not None:
        for name in COMPARED_CSVS:
            got, want = pd.read_csv(outdir / name), pd.read_csv(Path(expected_dir) / name)
            if list(got.columns) != list(want.columns) or got.shape != want.shape:
                raise ValueError(f"Regenerated {name} has a different layout from {expected_dir}")
            numeric = want.select_dtypes("number").columns
            text = [c for c in want.columns if c not in numeric]
            if not got[text].astype(str).equals(want[text].astype(str)) or not np.allclose(
                    got[numeric].to_numpy(float), want[numeric].to_numpy(float), rtol=1e-9, atol=1e-15):
                raise ValueError(f"Regenerated {name} differs from {expected_dir}")
            checked.append(name)
    if expected_table is not None:
        if sha(table_path) != sha(Path(expected_table)):
            raise ValueError(f"Regenerated table differs from {expected_table}")
        checked.append(Path(expected_table).name)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=HERE.parent, help="Revision or artifact root")
    parser.add_argument("--raw-calls", type=Path, help="Recorded per-call benchmark timings (raw_calls.csv)")
    parser.add_argument("--assessment", type=Path, help="Archived overhead_assessment.csv to reproduce first")
    parser.add_argument("--out-dir", type=Path, help="Directory for the diagnostic CSVs and manifest")
    parser.add_argument("--table", type=Path, help="Output path of the LaTeX Table 17")
    parser.add_argument("--expected-dir", type=Path, help="Compare regenerated CSVs with this directory")
    parser.add_argument("--expected-table", type=Path, help="Compare the regenerated table with this file")
    args = parser.parse_args()
    root = args.root.resolve()
    default = layout_defaults(root)
    raw_path = (args.raw_calls or default["raw"]).resolve()
    archived_path = (args.assessment or default["assessment"]).resolve()
    outdir = (args.out_dir or default["out"]).resolve()
    table_path = (args.table or default["table"]).resolve()
    expected_dir = args.expected_dir if args.expected_dir else default["expected_dir"]
    expected_table = args.expected_table if args.expected_table else default["expected_table"]
    raw = pd.read_csv(raw_path, float_precision="round_trip")
    archived = pd.read_csv(archived_path)
    if set(raw["run_id"].unique()) != {RUN_ID} or len(raw) != 30800:
        raise ValueError("Input is not the 30,800-call benchmark run used in the paper")

    # 1. Exact reproduction of the archived verdict.
    fits, heldout, assessments = ab.overhead_analysis(raw)
    repro = pd.DataFrame([a for a in assessments if "kind" in a])
    merged = validate_archived_assessment(repro, archived, heldout)

    # 2. What the archived (absolute) fit implies.
    fits = pd.DataFrame(fits)
    calibration = raw.loc[raw["regime"].eq("component_calibration") & raw["n_jobs"].eq(1)]
    one_row = (calibration[calibration.input_n.eq(1)].groupby(["seed", "model_key"]).elapsed_total_s.mean()
               .rename("measured_one_row_s").reset_index().rename(columns={"model_key": "component"}))
    # 3. Relative-error refit on the same calibration means.
    rel_parameters, rel_fits = {}, []
    for (seed, component), group in calibration.groupby(["seed", "model_key"], sort=False):
        fit = group.loc[group["data_role"].eq("fit")].groupby("input_n", sort=True)["elapsed_total_s"].mean()
        par = fit_relative(fit.index.to_numpy(), fit.to_numpy())
        rel_parameters.setdefault(int(seed), {})[component] = par
        rel_fits.append({"seed": int(seed), "component": component, **par})
    rel_fits = pd.DataFrame(rel_fits)
    comp = (fits[["seed", "component", "fixed_call_s", "per_row_s", "fit_mean_absolute_percentage_error",
                  "fit_max_absolute_percentage_error"]]
            .merge(rel_fits, on=["seed", "component"], suffixes=("_absolute", "_relative"))
            .merge(one_row, on=["seed", "component"]))
    component_summary = comp.groupby("component", sort=False).mean(numeric_only=True).drop(columns="seed").reset_index()

    rel_assess = assess(heldout_errors(raw, rel_parameters))
    summary = merged[["kind", "model_key", "mean_absolute_percentage_error", "max_absolute_percentage_error"]].rename(
        columns={"mean_absolute_percentage_error": "absolute_fit_mape", "max_absolute_percentage_error": "absolute_fit_max_ape"})
    summary = summary.merge(rel_assess.rename(columns={"mape": "relative_fit_mape", "max_ape": "relative_fit_max_ape",
                                                       "adequate": "relative_fit_adequate"}), on=["kind", "model_key"])
    absolute_adequate = int(((summary.absolute_fit_mape <= 20) & (summary.absolute_fit_max_ape <= 50)).sum())
    if len(summary) != 12 or absolute_adequate != 0 or not summary.relative_fit_adequate.all():
        raise ValueError("The recorded diagnostic does not reproduce 12 failed original and 12 passing relative fits")

    outdir.mkdir(parents=True, exist_ok=True)
    component_summary.to_csv(outdir / "component_fit_comparison.csv", index=False, float_format="%.17g", lineterminator="\n")
    comp.to_csv(outdir / "component_fit_comparison_by_seed.csv", index=False, float_format="%.17g", lineterminator="\n")
    summary.to_csv(outdir / "heldout_weighting_comparison.csv", index=False, float_format="%.17g", lineterminator="\n")

    # LaTeX table for the appendix.
    label = {**{k: v for k, v in ab.LABELS.items()}}
    lines = [r"% Generated by scripts/cost_model_weighting_diagnostic.py; do not hand-edit.",
             r"\begin{table}[!htbp]", r"\centering",
             r"\caption{Post-hoc diagnosis of the cost-model failure (five seeds; same calibration sizes, held-out sizes 32 and 1,000, and adequacy rule MAPE $\le20\%$, max APE $\le50\%$). ``Absolute'' is the archived pre-specified fit (equal weight per size mean on seconds); ``relative'' minimizes squared relative error on the same size means (Eq.~\ref{eq:relfit}). Fixed cost is the mean fitted per-call intercept; the measured one-row call is given for reference. All 12 absolute-fit rows fail and all 12 relative-fit rows pass the rule. The relative refit was designed after inspecting the held-out failure and does not replace the pre-specified negative result.}",
             r"\label{tab:costdiag}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{Fixed call cost ($\mu$s)} & \multicolumn{2}{c}{Held-out MAPE / max (\%)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(l){5-6}",
             r"Model & Absolute & Relative & 1 row & Absolute & Relative \\", r"\midrule"]
    def f(v, d=1):
        return f"{v:.{d}f}"
    for kind in ("component", "whole_policy"):
        if kind == "whole_policy":
            lines.append(r"\addlinespace")
        lines.append(r"\multicolumn{6}{@{}l}{\emph{" + ("Component score calls" if kind == "component" else "Whole decision policies") + r"}} \\")
        order = list(ab.COMPONENTS) if kind == "component" else list(ab.STAGES)
        rows = summary[summary.kind.eq(kind)].set_index("model_key").loc[order].reset_index()
        for _, r in rows.iterrows():
            name = r"\quad " + label[r.model_key]
            if kind == "component":
                c = component_summary.set_index("component").loc[r.model_key]
                fixed = [f(1e6 * c.fixed_call_s_absolute), f(1e6 * c.fixed_call_s_relative), f(1e6 * c.measured_one_row_s)]
            else:
                fixed = ["", "", ""]
            lines.append(" & ".join([name, *fixed,
                                     f"{f(r.absolute_fit_mape)} / {f(r.absolute_fit_max_ape)}",
                                     f"{f(r.relative_fit_mape)} / {f(r.relative_fit_max_ape)}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    facts = {
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_run_id": RUN_ID, "post_hoc": True,
        "archived_assessment_reproduced": True,
        "absolute_fit_adequate": absolute_adequate,
        "relative_fit_adequate": int(summary.relative_fit_adequate.sum()),
        "relative_fit_assessments": int(len(summary)),
        "component_relative_mape_range": [float(summary[summary.kind.eq("component")].relative_fit_mape.min()),
                                          float(summary[summary.kind.eq("component")].relative_fit_mape.max())],
        "policy_relative_mape_range": [float(summary[summary.kind.eq("whole_policy")].relative_fit_mape.min()),
                                       float(summary[summary.kind.eq("whole_policy")].relative_fit_mape.max())],
        "component_in_sample_mape_absolute": component_summary.set_index("component").fit_mean_absolute_percentage_error_absolute.to_dict(),
        "component_in_sample_mape_relative": component_summary.set_index("component").fit_mean_absolute_percentage_error_relative.to_dict(),
        "inputs": {relative_name(p, root): sha(p) for p in (raw_path, archived_path, HERE / "analyze_benchmark.py")},
        "script_sha256": sha(Path(__file__)),
        "outputs": {relative_name(p, root): sha(p) for p in
                    [outdir / "component_fit_comparison.csv", outdir / "component_fit_comparison_by_seed.csv",
                     outdir / "heldout_weighting_comparison.csv", table_path]},
    }
    facts["compared_with_shipped_outputs"] = compare_outputs(outdir, table_path, expected_dir, expected_table)
    (outdir / "manifest.json").write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: facts[k] for k in ("relative_fit_adequate", "relative_fit_assessments",
                                              "component_relative_mape_range", "policy_relative_mape_range")}, indent=1))
    print(component_summary.round(7).to_string())
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
