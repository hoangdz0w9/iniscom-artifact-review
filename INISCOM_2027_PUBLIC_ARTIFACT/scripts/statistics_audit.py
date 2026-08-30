#!/usr/bin/env python3
"""Recompute manuscript statistics from frozen scores; no fits or new seeds.

Preserves the original four-contrast Holm family. A three-exploratory-contrast
family and a +/-1-point equivalence margin are explicitly POST-HOC sensitivities.
Default 0.5 policies are diagnostics, not replacements chosen on test outcomes.
PR summaries use the archived 1001-point interpolation convention; grid dominance
is not a claim about every threshold or statistically established superiority.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_recall_curve, roc_auc_score

from routing_category_analysis import (
    Archive, COMPONENTS, MODELS, MODEL_NAMES, Provenance, descriptive_summary,
    environment, json_clean, resolve_analysis_paths, sha256, write_json,
)


PRIMARY = ("rf30", "cascade")
SELECTED_FAMILY = (PRIMARY, ("dt_rf30", "cascade"), ("dt_rf30_rf300", "cascade"), ("cascade", "dt_xgb"))
DIAGNOSTIC_CONTRASTS = (
    ("rf30@0.5", "cascade"), ("rf30@0.5", "rf30"), ("rf300@0.5", "rf300"),
    ("rf300@0.5", "rf30@0.5"), ("rf300@0.5", "cascade"),
)
CHECKPOINTS_PCT = np.array([80., 90., 95., 97., 99., 99.5])
RECALL_GRID = np.linspace(0., 1., 1001)


def metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    if y.shape != pred.shape or y.shape != score.shape or y.ndim != 1 or not y.size:
        raise ValueError("Metric inputs must be aligned nonempty vectors")
    if not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Metrics require binary labels with both classes and binary predictions")
    tn, fp = int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())
    fn, tp = int(((y == 1) & (pred == 0)).sum()), int(((y == 1) & (pred == 1)).sum())
    return {
        "N": len(y), "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "F1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0) * 100),
        "F1_macro": float(f1_score(y, pred, average="macro", zero_division=0) * 100),
        "MCC": float(matthews_corrcoef(y, pred)), "Accuracy": float((pred == y).mean() * 100),
        "Attack_precision": float(100 * tp / (tp + fp)) if tp + fp else 0.,
        "Attack_recall": float(100 * tp / (tp + fn)), "FAR": float(100 * fp / (fp + tn)),
        "ROC_AUC": float(roc_auc_score(y, score) * 100),
        "PR_AUC": float(average_precision_score(y, score) * 100),
    }


def paired_statistics(delta: np.ndarray) -> dict:
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 1 or delta.size < 2 or not np.isfinite(delta).all():
        raise ValueError("Paired differences require at least two finite observations")
    n, mean, sd = len(delta), float(delta.mean()), float(delta.std(ddof=1))
    se = sd / math.sqrt(n)
    half95, half90 = float(student_t.ppf(.975, n - 1) * se), float(student_t.ppf(.95, n - 1) * se)
    if se:
        t_value, p_value = mean / se, float(2 * student_t.sf(abs(mean / se), n - 1))
    else:
        t_value, p_value = (0., 1.) if mean == 0 else (math.copysign(float("inf"), mean), 0.)
    return {
        "n_seeds": n, "mean_delta_pp": mean, "sd_delta_pp": sd, "SE_pp": se,
        "CI95_low_pp": mean - half95, "CI95_high_pp": mean + half95,
        "CI90_low_pp": mean - half90, "CI90_high_pp": mean + half90,
        "t_statistic": t_value, "p_two_sided_raw": p_value,
        "A_better_seeds": int((delta > 0).sum()), "B_better_seeds": int((delta < 0).sum()),
        "tied_seeds": int((delta == 0).sum()),
    }


def tost(delta: np.ndarray, margin: float) -> dict:
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("Equivalence margin must be finite and positive")
    stats = paired_statistics(delta)
    mean, se, df = stats["mean_delta_pp"], stats["SE_pp"], stats["n_seeds"] - 1
    if se:
        p_lower = float(student_t.sf((mean + margin) / se, df))
        p_upper = float(student_t.cdf((mean - margin) / se, df))
    else:
        p_lower = 0. if mean > -margin else (.5 if mean == -margin else 1.)
        p_upper = 0. if mean < margin else (.5 if mean == margin else 1.)
    result = {
        "margin_pp": margin, "p_lower": p_lower, "p_upper": p_upper, "TOST_p": max(p_lower, p_upper),
        "CI90_low_pp": stats["CI90_low_pp"], "CI90_high_pp": stats["CI90_high_pp"],
        "equivalent_alpha_0_05": max(p_lower, p_upper) < .05,
        "CI90_strictly_inside_margin": stats["CI90_low_pp"] > -margin and stats["CI90_high_pp"] < margin,
    }
    if result["equivalent_alpha_0_05"] != result["CI90_strictly_inside_margin"]:
        raise RuntimeError("TOST decision and 90% interval containment disagree")
    return result


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not p.size or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Holm requires a nonempty vector of p values in [0, 1]")
    order = np.argsort(p, kind="stable")
    ordered = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(p.size, 0, -1)))
    adjusted = np.empty_like(ordered)
    adjusted[order] = ordered
    return adjusted


def precision_power(sd: float, margin: float, n_observed: int, power: float = .8, alpha: float = .05) -> dict:
    """Conditional precision and NORMAL-APPROXIMATION TOST power planning.

    Paired differences assumed iid normal, true mean ZERO, fixed SD equal to
    observed estimate. These are sensitivity calculations, not a prospective
    power analysis or a guarantee that more data will demonstrate equivalence.
    """
    if sd <= 0 or margin <= 0 or n_observed < 2 or not 0 < power < 1 or not 0 < alpha < .5:
        raise ValueError("Invalid precision/power planning inputs")
    z = norm.ppf(1 - alpha)
    approximate_precision_n = float((z * sd / margin) ** 2)
    n = max(2, math.floor(approximate_precision_n))
    while student_t.ppf(1 - alpha, n - 1) * sd / math.sqrt(n) >= margin:
        n += 1
    approximate_power_n = float(((z + norm.ppf((1 + power) / 2)) * sd / margin) ** 2)
    return {
        "margin_pp": margin, "observed_paired_SD_pp": sd, "observed_n": n_observed,
        "assumed_true_mean_difference_pp": 0., "alpha_one_sided": alpha, "desired_power": power,
        "observed_n_CI90_halfwidth_pp": float(student_t.ppf(1 - alpha, n_observed - 1) * sd / math.sqrt(n_observed)),
        "SD_upper_bound_at_observed_n_and_zero_mean_pp": float(margin * math.sqrt(n_observed) / student_t.ppf(1 - alpha, n_observed - 1)),
        "normal_precision_n_unrounded": approximate_precision_n, "fixed_SD_t_precision_min_n": n,
        "normal_approx_power_n_unrounded": approximate_power_n,
        "normal_approx_power_n_ceiling": math.ceil(approximate_power_n),
        "n550_CI90_halfwidth_pp": float(student_t.ppf(1 - alpha, 549) * sd / math.sqrt(550)),
        "precision_formula": "t_(1-alpha,n-1) * observed_SD / sqrt(n) < margin; interval centered at zero",
        "power_formula": "n ~= ((z_(1-alpha)+z_((1+power)/2))*observed_SD/margin)^2",
        "assumptions": "POST-HOC planning illustration; iid normal paired seed differences; true difference=0; SD fixed at estimate from ten seeds; fixed official test, not population/test-set uncertainty",
    }


def curve_on_grid(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    precision, recall, _ = precision_recall_curve(y, score)
    # Reproduce the archived interpolation exactly, including tied recall values.
    return np.interp(RECALL_GRID, recall[::-1], precision[::-1]) * 100


def grid_dominance(a: np.ndarray, b: np.ndarray, tolerance: float = 1e-10) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all() or tolerance < 0:
        raise ValueError("Invalid mean-grid comparison")
    d = a - b
    wins, losses = int((d > tolerance).sum()), int((d < -tolerance).sum())
    return {
        "n_grid_points": len(d), "A_higher_points": wins, "B_higher_points": losses,
        "tied_points_at_tolerance": len(d) - wins - losses,
        "minimum_precision_difference_pp": float(d.min()), "maximum_precision_difference_pp": float(d.max()),
        "A_weakly_dominates_mean_grid": losses == 0 and wins > 0,
        "A_strictly_higher_at_every_grid_point": wins == len(d), "tolerance_pp": tolerance,
        "interpretation": "descriptive mean interpolated grid only; not all possible thresholds, not every seed, not statistical or causal dominance",
    }


def require_close(actual: float, archived: float, context: str, tolerance: float = 1e-9) -> float:
    error = abs(float(actual) - float(archived))
    if not np.isfinite(error) or error > tolerance:
        raise RuntimeError(f"Archived numeric check failed ({context}): {actual} vs {archived}, tolerance {tolerance}")
    return error


def run_analysis(project: Path, revision: Path, source: Path, test_csv: Path, out: Path) -> dict:
    provenance = Provenance()
    provenance.add(Path(__file__), "statistics implementation")
    provenance.add(Path(__file__).with_name("routing_category_analysis.py"), "shared exact reconstruction implementation")
    archive = Archive(source, test_csv, provenance)
    archived_quality = pd.read_csv(provenance.add(archive.gap / "final_quality_threshold_closed_by_seed.csv", "authoritative native quality values"))
    archive._validate_table(archived_quality, "model_key", MODELS)
    quality_index = archived_quality.set_index(["model_key", "seed"])
    records, curve_records, validations = [], [], []
    curves = {model: [] for model in MODELS}
    for seed in archive.seeds:
        loaded = archive.load_seed(seed)
        for model in MODELS:
            frozen = loaded[model]
            y, pred, score = (frozen.arrays[k] for k in ("y_true", "y_pred", "y_score"))
            values = metrics(y, pred, score)
            expected = quality_index.loc[(model, seed)]
            error = max(require_close(values[k], expected[k], f"{model}/{seed}/{k}") for k in (
                "TN", "FP", "FN", "TP", "F1_weighted", "F1_macro", "MCC", "Attack_precision", "Attack_recall", "FAR", "ROC_AUC", "PR_AUC"))
            validations.append({"check": "native_quality", "model_key": model, "seed": seed, "max_abs_error": error, "status": "pass"})
            records.append({"model_key": model, "base_model": model, "model": MODEL_NAMES[model], "policy": "native_validation_selected", "seed": seed, "selected_cutoff": frozen.cutoff, **values})
            curve = curve_on_grid(y, score.astype(float))
            curves[model].append(curve)
            for recall in CHECKPOINTS_PCT:
                grid_index = int(round(recall * 10))
                require_close(RECALL_GRID[grid_index] * 100, recall, "checkpoint grid alignment")
                curve_records.append({"model_key": model, "seed": seed, "recall_pct": recall, "precision_pct": float(curve[grid_index])})
            if model in COMPONENTS:
                default_values = metrics(y, (score >= .5).astype(np.int8), score)
                records.append({"model_key": model + "@0.5", "base_model": model, "model": MODEL_NAMES[model] + "@0.5", "policy": "diagnostic_fixed_0.5", "seed": seed, "selected_cutoff": .5, **default_values})
        print(f"Recomputed frozen statistics and PR grid: seed {seed}", flush=True)
    by_seed = pd.DataFrame(records)
    numeric_metrics = ["F1_weighted", "F1_macro", "MCC", "Accuracy", "Attack_precision", "Attack_recall", "FAR", "ROC_AUC", "PR_AUC"]
    summary = descriptive_summary(by_seed, ["model_key"], numeric_metrics, ["base_model", "model", "policy", "N"])
    pivot = by_seed.pivot(index="seed", columns="model_key", values="F1_weighted").sort_index()
    if pivot.isna().any().any() or len(pivot) != len(archive.seeds):
        raise RuntimeError("Incomplete pairing across native/default policies")
    stats_rows, delta_rows = [], []
    for a, b in SELECTED_FAMILY + DIAGNOSTIC_CONTRASTS:
        delta = (pivot[a] - pivot[b]).to_numpy()
        role = "primary_archived" if (a, b) == PRIMARY else ("exploratory_architecture" if (a, b) in SELECTED_FAMILY else "posthoc_default_diagnostic")
        stats_rows.append({"comparison": f"{a} - {b}", "model_A": a, "model_B": b, "role": role, **paired_statistics(delta)})
        delta_rows.extend({"comparison": f"{a} - {b}", "seed": seed, "F1W_A": float(pivot.loc[seed, a]), "F1W_B": float(pivot.loc[seed, b]), "delta_pp": float(pivot.loc[seed, a] - pivot.loc[seed, b])} for seed in pivot.index)
    stats = pd.DataFrame(stats_rows)
    primary_delta = (pivot[PRIMARY[0]] - pivot[PRIMARY[1]]).to_numpy()
    tost_rows = [{"comparison": "rf30 - cascade", "margin_role": role, **tost(primary_delta, margin)} for margin, role in (
        (.1, "original_archived_margin; prospective_registration_not_established"), (1., "POSTHOC_margin_sensitivity_not_a_replacement"))]
    primary_stats = paired_statistics(primary_delta)
    planning = pd.DataFrame([precision_power(primary_stats["sd_delta_pp"], m, primary_stats["n_seeds"]) for m in (.1, 1.)])
    holm_rows = []
    for name, contrasts in (("retained_four_contrast_family", SELECTED_FAMILY), ("POSTHOC_three_exploratory_family", SELECTED_FAMILY[1:])):
        names = [f"{a} - {b}" for a, b in contrasts]
        chosen = stats.set_index("comparison").loc[names].reset_index()
        adjusted = holm_adjust(chosen["p_two_sided_raw"].to_numpy())
        for row, p_holm in zip(chosen.to_dict("records"), adjusted):
            holm_rows.append({"family": name, "family_size": len(contrasts), "comparison": row["comparison"], "p_raw": row["p_two_sided_raw"], "p_holm": p_holm, "significant_alpha_0_05": p_holm < .05})
    holm = pd.DataFrame(holm_rows)
    original_stats = pd.read_csv(provenance.add(source / "overleaf_revised" / "generated" / "selected_point_statistics" / "selected_point_statistics_95ci_holm.csv", "archived four-family paired statistics"))
    if set(original_stats["comparison"]) != {f"{a} - {b}" for a, b in SELECTED_FAMILY} or len(original_stats) != 4:
        raise RuntimeError("Archived four-contrast family changed")
    for row in original_stats.to_dict("records"):
        own = stats.set_index("comparison").loc[row["comparison"]]
        checks = [require_close(own[a], row[b], f"paired/{row['comparison']}/{a}") for a, b in (
            ("mean_delta_pp", "mean_delta_A_minus_B_pp"), ("sd_delta_pp", "sd_delta_pp"),
            ("CI95_low_pp", "CI95_low_pp"), ("CI95_high_pp", "CI95_high_pp"), ("p_two_sided_raw", "p_two_sided_raw"))]
        p_own = holm.loc[(holm["family"] == "retained_four_contrast_family") & (holm["comparison"] == row["comparison"]), "p_holm"].item()
        checks.append(require_close(p_own, row["p_two_sided_holm"], "retained Holm"))
        validations.append({"check": "archived_paired_and_Holm4", "model_key": row["comparison"], "seed": None, "max_abs_error": max(checks), "status": "pass"})
    checkpoint_frame = pd.DataFrame(curve_records)
    checkpoint_summary = descriptive_summary(checkpoint_frame, ["model_key", "recall_pct"], ["precision_pct"])
    grid_rows = []
    mean_curves = {}
    for model, values in curves.items():
        stack = np.asarray(values)
        mean_curves[model] = stack.mean(axis=0)
        sd_curve = stack.std(axis=0, ddof=1)
        grid_rows.extend({"model_key": model, "recall_pct": float(r * 100), "precision_mean": float(m), "precision_sd": float(s)} for r, m, s in zip(RECALL_GRID, mean_curves[model], sd_curve))
    grid = pd.DataFrame(grid_rows)
    archived_grid = pd.read_csv(provenance.add(archive.gap / "pr_curve_summary.csv", "archived mean PR grid"))
    for model in MODELS:
        previous = archived_grid.loc[archived_grid["model_key"] == model].sort_values("recall")
        own = grid.loc[grid["model_key"] == model].sort_values("recall_pct")
        if len(previous) != 1001 or previous["recall"].duplicated().any():
            raise RuntimeError(f"Incomplete archived PR grid for {model}")
        errors = [float(np.max(np.abs(previous[a].to_numpy() - own[b].to_numpy()))) for a, b in (("recall", "recall_pct"), ("precision_mean", "precision_mean"), ("precision_sd", "precision_sd"))]
        require_close(max(errors), 0., f"PR grid/{model}")
        validations.append({"check": "archived_PR_grid", "model_key": model, "seed": None, "max_abs_error": max(errors), "status": "pass"})
    dominance = pd.DataFrame([{"model_A": a, "model_B": b, **grid_dominance(mean_curves[a], mean_curves[b])} for a, b in itertools.permutations(MODELS, 2)])
    summary_index = summary.set_index("model_key")
    sd_pairs = (("rf30", "dt_xgb"), ("rf30", "rf30@0.5"), ("rf300", "rf300@0.5"), ("dt_rf30", "rf30"), ("xgb300", "dt_xgb"), ("xgb300@0.5", "dt_xgb"))
    sd_rows = []
    for a, b in sd_pairs:
        sa, sb = float(summary_index.loc[a, "F1_weighted_sd"]), float(summary_index.loc[b, "F1_weighted_sd"])
        sd_rows.append({"model_A": a, "model_B": b, "SD_A_pp": sa, "SD_B_pp": sb, "SD_ratio_A_over_B": sa / sb if sb else np.nan, "variance_ratio_A_over_B": (sa / sb) ** 2 if sb else np.nan, "interpretation": "descriptive between-seed dispersion under each policy; no causal routing-stabilization claim"})
    files = {
        "native_and_default_by_seed.csv": by_seed, "native_and_default_summary.csv": summary,
        "paired_deltas_by_seed.csv": pd.DataFrame(delta_rows), "paired_statistics.csv": stats,
        "tost_sensitivity.csv": pd.DataFrame(tost_rows), "precision_power_assumptions.csv": planning,
        "holm_sensitivity.csv": holm, "pr_checkpoints_by_seed.csv": checkpoint_frame,
        "pr_checkpoints_summary.csv": checkpoint_summary, "pr_mean_grid.csv": grid,
        "pr_mean_grid_dominance.csv": dominance, "sd_comparisons.csv": pd.DataFrame(sd_rows),
        "validation_checks.csv": pd.DataFrame(validations), "reconstruction_checks.csv": pd.DataFrame(archive.checks),
    }
    for name, data in files.items():
        data.to_csv(out / name, index=False, encoding="utf-8")
    native = summary.loc[summary["policy"] == "native_validation_selected"].sort_values("F1_weighted_mean", ascending=False)
    relevant_dominance = dominance.loc[(dominance["model_A"] == "cascade") & dominance["model_B"].isin(("dt_rf30", "dt_rf30_rf300"))]
    key = {
        "status": "complete", "dataset": archive.dataset_info, "seeds": archive.seeds,
        "primary_paired": primary_stats, "equivalence": tost_rows, "precision_power": planning.to_dict("records"),
        "default_and_native_RF_metrics": summary.loc[summary["model_key"].isin(("rf30", "rf30@0.5", "rf300", "rf300@0.5", "cascade"))].to_dict("records"),
        "default_paired_diagnostics": stats.loc[stats["role"] == "posthoc_default_diagnostic"].to_dict("records"),
        "best_native_selected_model": native.iloc[0][["model_key", "F1_weighted_mean", "F1_weighted_sd"]].to_dict(),
        "holm": holm.to_dict("records"), "cascade_vs_RF_route_mean_grid": relevant_dominance.to_dict("records"),
        "pr_checkpoints": checkpoint_summary.to_dict("records"), "SD_comparisons": sd_rows,
        "integrity": {"native_model_seed_pairs": len(archive.checks), "exact_score_and_prediction_reconstruction": True, "all_archived_numeric_checks_passed": True},
        "interpretation": "Primary estimand and Holm4 retained. Fixed-default, Holm3, wider-margin and planning analyses are post-hoc diagnostics/sensitivities. Intervals concern seed variability on one fixed official test; no post-hoc primary replacement or deployment recommendation.",
    }
    write_json(out / "key_results.json", key)
    provenance.verify_unchanged()
    write_json(out / "MANIFEST.json", {
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(), "analysis": "statistics_audit",
        "project_root": str(project), "revision_root": str(revision), "source_root": str(source), "environment": environment(),
        "input_files": list(provenance.items.values()), "seeds": archive.seeds, "dataset": archive.dataset_info,
        "outputs": [{"path": str(out / n), "rows": len(df), "sha256": sha256(out / n)} for n, df in files.items()],
        "key_results_sha256": sha256(out / "key_results.json"),
        "definitions": {"F1_units": "percentage points; weighted F1 percentage", "PR_AUC": "average_precision_score * 100 (not trapezoidal PR area)", "PR_grid": "precision_recall_curve; reverse recall/precision; np.interp on 1001-point linspace(0,1); average interpolated curves over seeds", "CI": "two-sided paired seed-level Student-t; 95% for differences, 90% for TOST at alpha=.05", "TOST_margin": "original archived +/-0.10 points retained; +/-1.0 explicitly post-hoc", "Holm": "original four selected contrasts retained; separate three-exploratory family post-hoc sensitivity", "power": "normal-approximation planning illustration only, assumes true paired mean zero and SD fixed at current estimate"},
        "limitations": key["interpretation"],
    })
    return key


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="Data parent containing FINAL3 and the official test CSV, normally DATA_ROOT")
    parser.add_argument("--revision-root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Revision/output project, normally .")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args(argv)


def main() -> int:
    project, revision, source, test_csv, out = resolve_analysis_paths(parse_args(), "statistics")
    context = {"analysis": "statistics_audit", "project_root": str(project), "revision_root": str(revision), "source_root": str(source)}
    write_json(out / "MANIFEST.json", {"status": "running", **context})
    write_json(out / "key_results.json", {"status": "running", **context})
    try:
        key = run_analysis(project, revision, source, test_csv, out)
    except Exception as exc:
        failure = {"status": "failed", **context, "error": str(exc), "error_type": type(exc).__name__}
        write_json(out / "MANIFEST.json", failure)
        write_json(out / "key_results.json", failure)
        raise
    print(json.dumps(json_clean({"status": key["status"], "output_dir": str(out), "primary_paired": key["primary_paired"], "integrity": key["integrity"]}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
