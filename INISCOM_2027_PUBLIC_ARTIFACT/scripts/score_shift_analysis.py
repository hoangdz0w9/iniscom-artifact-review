#!/usr/bin/env python3
"""Read-only score analysis for the INISCOM revision; never trains a model.

The default --project-root is the parent of this revision directory (DATA_ROOT
in the original workspace). Inputs are the 8 x 10 reconstructed validation/
test artifact pairs under FINAL3. Original NPZs are never modified.

Three *global-score* policies use the same frozen internal routing gates:
0.5, exhaustive validation F1-W, and exhaustive validation F1-W after oracle
target-prior weighting. Archived native-policy y_pred is a separate reference;
it must not be relabelled as one of these newly recomputed policies.

Only the oracle branches consume the target label prevalence, never individual
target labels or target scores during threshold selection. Consequently they
are retrospective oracle diagnostics, not deployable estimates or upper bounds.
All reported intervals describe seed variation on the SAME fixed test set.

Constraint-selection logic is adapted from FINAL3/overleaf_revised/scripts/
validation_constrained_operating_points.py, whose hash is recorded if available.
No PR/ROC curve interpolation, chart generation, or calibration fitting occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


MODELS = ("dt", "xgb300", "rf30", "rf300", "dt_xgb", "cascade", "dt_rf30", "dt_rf30_rf300")
SEEDS = (21, 42, 84, 168, 336, 348184071, 362797685, 883000906, 1365468824, 1486213216)
MODEL_LABELS = {
    "dt": "DT", "xgb300": "XGB-300", "rf30": "RF-30", "rf300": "RF-300",
    "dt_xgb": "DT->XGB-300", "cascade": "DT->XGB-300->RF-30",
    "dt_rf30": "DT->RF-30", "dt_rf30_rf300": "DT->RF-30->RF-300",
}
FILE_RE = re.compile(r"^(?P<model>.+)__seed_(?P<seed>-?\d+)\.npz$")
POLICIES = ("global_default_0p5", "global_validation_f1", "global_oracle_prior_weighted_validation_f1")
CONSTRAINTS = (("far_max", 10.0), ("far_max", 20.0), ("recall_min", 97.0))
METRIC_NAMES = (
    "F1_weighted_pct", "F1_macro_pct", "F1_attack_pct", "F1_normal_pct",
    "Attack_precision_pct", "Attack_recall_pct", "FAR_pct", "Accuracy_pct",
    "Balanced_accuracy_pct", "MCC", "attack_prevalence_pct",
)
REPORT_METRICS = ("F1_weighted_pct", "FAR_pct", "Attack_recall_pct", "Attack_precision_pct")
CALIBRATION_METRICS = ("Brier_score", "log_loss", "ECE_15_equal_width", "mean_score", "calibration_in_large")
CLASS_GRID = np.linspace(0.0, 1.0, 1001)
QUANTILES = np.asarray([0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1])
COUNT_COLS = ["TN", "FP", "FN", "TP"]
CSV_FLOAT_FORMAT = "%.17g"
SCOPE_NOTES = [
    "Post-hoc score-only diagnostics, not a prospective confirmation or retraining experiment.",
    "Internal routing gates and component models remain frozen; a global threshold is not native gate selection.",
    "Native y_pred is preserved separately; exhaustive score thresholds may differ from the archived threshold grid.",
    "Oracle branches use target label prevalence only. No individual test label or test score chooses a threshold.",
    "Oracle-prior results are not an upper bound and are not deployable without estimating a target prior.",
    "Weighting acts on all confusion counts before both class F1 denominators and target support weights are recomputed.",
    "Prior posterior correction assumes an adequately calibrated source score and stable within-class score distributions.",
    "Class-weighted RF scores and routed mixtures need not be calibrated source posteriors; no calibrator is fitted here.",
    "Improvement or lack of improvement after correction does not identify a cause of distribution shift.",
    "Pure label shift preserves FAR and recall for a fixed score function and threshold at population level.",
    "The same strictly increasing score map on validation and test does not change an exhaustive constrained operating point.",
    "Class-conditional KS values are effect sizes only; no IID row-level significance tests are performed.",
    "Flows may be dependent or duplicated; seeds reuse the same test rows. Seed intervals are not population intervals.",
    "Validation also selected native gates, so its apparent performance can contain selection optimism.",
    "Validation row IDs were not archived in these inputs: label-order checks cannot independently prove row identity.",
]


@dataclass(frozen=True)
class Artifact:
    path: Path
    y: np.ndarray
    score: np.ndarray
    pred: np.ndarray | None
    row_id: np.ndarray | None
    row_id_key: str | None
    sha256: str


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_binary(values: np.ndarray, name: str, both: bool = False) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    labels = np.unique(array)
    if not np.isin(labels, [0, 1]).all() or (both and labels.size != 2):
        raise ValueError(f"{name} must contain {'both ' if both else ''}binary labels; found {labels}")
    return array.astype(np.int8, copy=False)


def require_scores(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "fiu":
        raise ValueError("Scores must be a nonempty one-dimensional numeric array")
    if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError("Scores must be finite and within [0,1]; raw values are never clipped")
    return array


def load_artifact(path: Path, split: str) -> Artifact:
    initial_hash = file_hash(path)
    with np.load(path, allow_pickle=False) as archive:
        if not {"y_true", "y_score"}.issubset(archive.files):
            raise ValueError(f"Missing y_true/y_score: {path}")
        y = require_binary(archive["y_true"], "y_true", both=True)
        score = require_scores(archive["y_score"])
        pred = require_binary(archive["y_pred"], "y_pred") if "y_pred" in archive.files else None
        id_keys = [key for key in ("test_row_id", "validation_row_id", "val_row_id", "row_id", "_row_id") if key in archive.files]
        row_id_key = id_keys[0] if id_keys else None
        row_id = np.asarray(archive[row_id_key]) if row_id_key else None
        for key in id_keys[1:]:
            if not np.array_equal(archive[key], row_id):
                raise ValueError(f"Conflicting ID arrays in {path}")
    if score.size != y.size or (pred is not None and pred.size != y.size):
        raise ValueError(f"Array length mismatch: {path}")
    if split == "test" and (row_id is None or pred is None):
        raise ValueError(f"Test artifacts must preserve row IDs and native y_pred: {path}")
    if row_id is not None:
        if row_id.ndim != 1 or row_id.size != y.size or row_id.dtype.kind not in "iu":
            raise ValueError(f"Invalid row-ID shape/type: {path}")
        if np.unique(row_id).size != row_id.size:
            raise ValueError(f"Duplicate row IDs: {path}")
    for array in (y, score, pred, row_id):
        if array is not None:
            array.setflags(write=False)
    return Artifact(path, y, score, pred, row_id, row_id_key, initial_hash)


def scan_exact(directory: Path) -> dict[tuple[str, int], Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    found: dict[tuple[str, int], Path] = {}
    for path in sorted(directory.glob("*.npz")):
        match = FILE_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected NPZ filename: {path}")
        key = (match["model"], int(match["seed"]))
        if key in found:
            raise ValueError(f"Duplicate model/seed pair: {key}")
        found[key] = path
    expected = {(model, seed) for model in MODELS for seed in SEEDS}
    if set(found) != expected:
        raise ValueError(f"Expected exactly 8 x 10 pairs in {directory}; missing={sorted(expected-set(found))}; extra={sorted(set(found)-expected)}")
    return found


def load_and_verify(validation_dir: Path, test_dir: Path) -> tuple[dict[tuple[str, int, str], Artifact], pd.DataFrame]:
    paths = {"validation": scan_exact(validation_dir), "test": scan_exact(test_dir)}
    artifacts: dict[tuple[str, int, str], Artifact] = {}
    audits: list[dict[str, Any]] = []
    shared_test: Artifact | None = None
    for seed in SEEDS:
        for split in ("validation", "test"):
            reference: Artifact | None = None
            for model in MODELS:
                item = load_artifact(paths[split][model, seed], split)
                if reference is not None:
                    if not np.array_equal(item.y, reference.y):
                        raise ValueError(f"Label/order mismatch: {model}, {seed}, {split}")
                    if (item.row_id is None) != (reference.row_id is None):
                        raise ValueError(f"Inconsistent row-ID availability: {model}, {seed}, {split}")
                    if item.row_id is not None and not np.array_equal(item.row_id, reference.row_id):
                        raise ValueError(f"Row-ID/order mismatch: {model}, {seed}, {split}")
                else:
                    reference = item
                if split == "test":
                    if shared_test is not None and (not np.array_equal(item.y, shared_test.y) or not np.array_equal(item.row_id, shared_test.row_id)):
                        raise ValueError(f"The fixed test set changed across seeds: {model}, {seed}")
                    shared_test = item
                artifacts[model, seed, split] = item
                audits.append({
                    "model_key": model, "seed": seed, "split": split, "path": str(item.path),
                    "sha256": item.sha256, "file_bytes": item.path.stat().st_size,
                    "n_rows": item.y.size, "n_normal": int((item.y == 0).sum()),
                    "n_attack": int(item.y.sum()), "attack_prevalence": float(item.y.mean()),
                    "raw_score_dtype": str(item.score.dtype), "raw_score_min": float(item.score.min()),
                    "raw_score_max": float(item.score.max()), "native_y_pred_available": item.pred is not None,
                    "row_id_key": item.row_id_key, "row_ids_unique": item.row_id is not None,
                    "row_identity_status": "IDs and labels aligned" if item.row_id is not None else "IDs not archived; labels/order only",
                })
    return artifacts, pd.DataFrame(audits)


def divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)


def metrics_from_counts(tn: Any, fp: Any, fn: Any, tp: Any) -> dict[str, np.ndarray]:
    tn, fp, fn, tp = np.broadcast_arrays(*(np.asarray(value, dtype=float) for value in (tn, fp, fn, tp)))
    if any(np.any(value < 0) or not np.isfinite(value).all() for value in (tn, fp, fn, tp)):
        raise ValueError("Confusion counts must be finite and nonnegative")
    normal, attack = tn + fp, tp + fn
    total = normal + attack
    precision = divide(tp, tp + fp)
    recall = divide(tp, attack)
    far = divide(fp, normal)
    f1_attack = divide(2 * tp, 2 * tp + fp + fn)
    f1_normal = divide(2 * tn, 2 * tn + fp + fn)
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp, "sample_mass": total,
        "F1_weighted_pct": 100 * divide(attack * f1_attack + normal * f1_normal, total),
        "F1_macro_pct": 50 * (f1_attack + f1_normal),
        "F1_attack_pct": 100 * f1_attack, "F1_normal_pct": 100 * f1_normal,
        "Attack_precision_pct": 100 * precision, "Attack_recall_pct": 100 * recall,
        "FAR_pct": 100 * far, "Accuracy_pct": 100 * divide(tp + tn, total),
        "Balanced_accuracy_pct": 50 * (recall + divide(tn, normal)),
        "MCC": divide(tp * tn - fp * fn, np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))),
        "attack_prevalence_pct": 100 * divide(attack, total),
    }


def require_prior(prior: float) -> float:
    if not np.isfinite(prior) or not 0 < prior < 1:
        raise ValueError("Both source and target priors must lie strictly inside (0,1)")
    return float(prior)


def prior_weights(source_prior: float, target_prior: float) -> tuple[float, float]:
    source_prior, target_prior = require_prior(source_prior), require_prior(target_prior)
    return (1 - target_prior) / (1 - source_prior), target_prior / source_prior


def prediction_metrics(y: np.ndarray, pred: np.ndarray, target_prior: float | None = None) -> dict[str, float]:
    y = require_binary(y, "y_true", both=True)
    pred = require_binary(pred, "prediction")
    if pred.size != y.size:
        raise ValueError("Prediction/label length mismatch")
    tn = float(np.count_nonzero((y == 0) & (pred == 0)))
    fp = float(np.count_nonzero((y == 0) & (pred == 1)))
    fn = float(np.count_nonzero((y == 1) & (pred == 0)))
    tp = float(np.count_nonzero((y == 1) & (pred == 1)))
    if target_prior is not None:
        w0, w1 = prior_weights(float(y.mean()), target_prior)
        tn, fp, fn, tp = tn * w0, fp * w0, fn * w1, tp * w1
    return {key: float(value) for key, value in metrics_from_counts(tn, fp, fn, tp).items()}


def exact_threshold_table(y: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """All distinct >= classifiers, including +inf (all negative), without interpolation."""
    y, score = require_binary(y, "y_true", both=True), require_scores(score).astype(float, copy=False)
    if y.size != score.size:
        raise ValueError("Score/label length mismatch")
    order = np.argsort(-score, kind="stable")
    sorted_score, sorted_y = score[order], y[order].astype(np.int64)
    block_end = np.flatnonzero(np.r_[sorted_score[1:] != sorted_score[:-1], True])
    tp = np.r_[0, np.cumsum(sorted_y)[block_end]]
    selected = np.r_[0, block_end + 1]
    fp = selected - tp
    fn = int(y.sum()) - tp
    tn = int(y.size - y.sum()) - fp
    frame = pd.DataFrame(metrics_from_counts(tn, fp, fn, tp))
    frame.insert(0, "threshold", np.r_[np.inf, sorted_score[block_end]])
    frame.insert(1, "candidate_index", np.arange(len(frame)))
    return frame


def weighted_threshold_table(table: pd.DataFrame, source_prior: float, target_prior: float) -> pd.DataFrame:
    w0, w1 = prior_weights(source_prior, target_prior)
    frame = pd.DataFrame(metrics_from_counts(table.TN.to_numpy() * w0, table.FP.to_numpy() * w0, table.FN.to_numpy() * w1, table.TP.to_numpy() * w1))
    frame.insert(0, "threshold", table.threshold.to_numpy(copy=True))
    frame.insert(1, "candidate_index", table.candidate_index.to_numpy(copy=True))
    return frame


def select_f1_row(table: pd.DataFrame) -> pd.Series:
    ordered = table.assign(distance_from_half=(table.threshold - .5).abs()).sort_values(
        ["F1_weighted_pct", "FAR_pct", "Attack_recall_pct", "distance_from_half", "threshold"],
        ascending=[False, True, False, True, False], kind="stable",
    )
    return ordered.iloc[0]


def select_policy_thresholds(y_val: np.ndarray, score_val: np.ndarray, target_prior: float) -> dict[str, float]:
    """Pure selector: no test scores or individual test labels are accepted."""
    table = exact_threshold_table(y_val, score_val)
    weighted = weighted_threshold_table(table, float(np.mean(y_val)), target_prior)
    return {
        POLICIES[0]: .5,
        POLICIES[1]: float(select_f1_row(table).threshold),
        POLICIES[2]: float(select_f1_row(weighted).threshold),
    }


def select_constraint_row(table: pd.DataFrame, kind: str, target_pct: float) -> pd.Series:
    """Selection/tie rules copied from the archived transfer script, not F1 maximization."""
    if kind == "far_max":
        feasible = table.loc[table.FAR_pct <= target_pct + 1e-12].copy()
        columns = ["Attack_recall_pct", "F1_weighted_pct", "FAR_pct", "distance_from_half", "threshold"]
        ascending = [False, False, True, True, False]
    elif kind == "recall_min":
        feasible = table.loc[table.Attack_recall_pct >= target_pct - 1e-12].copy()
        columns = ["FAR_pct", "F1_weighted_pct", "Attack_recall_pct", "distance_from_half", "threshold"]
        ascending = [True, False, False, True, False]
    else:
        raise ValueError(f"Unknown constraint: {kind}")
    if feasible.empty:
        raise ValueError(f"No feasible threshold for {kind}={target_pct}")
    feasible["distance_from_half"] = (feasible.threshold - .5).abs()
    return feasible.sort_values(columns, ascending=ascending, kind="stable").iloc[0]


def prior_correct(score: np.ndarray, source_prior: float, target_prior: float) -> np.ndarray:
    score = require_scores(score).astype(np.float64, copy=True)
    w0, w1 = prior_weights(source_prior, target_prior)
    if source_prior == target_prior:
        return score
    return w1 * score / (w1 * score + w0 * (1 - score))


def calibration_metrics(y: np.ndarray, score: np.ndarray) -> tuple[dict[str, float], list[dict[str, Any]]]:
    y, score = require_binary(y, "y_true", both=True), require_scores(score).astype(float, copy=False)
    if y.size != score.size:
        raise ValueError("Calibration score/label length mismatch")
    clipped_for_log_only = np.clip(score, 1e-15, 1 - 1e-15)
    bin_id = np.minimum((score * 15).astype(int), 14)
    ece = 0.0
    bins: list[dict[str, Any]] = []
    for index in range(15):
        mask = bin_id == index
        count = int(mask.sum())
        predicted = float(score[mask].mean()) if count else math.nan
        observed = float(y[mask].mean()) if count else math.nan
        gap = abs(predicted - observed) if count else math.nan
        if count:
            ece += count / y.size * gap
        bins.append({"bin": index + 1, "bin_lower": index / 15, "bin_upper": (index + 1) / 15,
                     "count": count, "mean_score": predicted, "observed_attack_fraction": observed, "absolute_gap": gap})
    return {"Brier_score": float(np.mean((score - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(clipped_for_log_only) + (1-y) * np.log1p(-clipped_for_log_only))),
            "ECE_15_equal_width": float(ece), "mean_score": float(score.mean()),
            "calibration_in_large": float(score.mean() - y.mean())}, bins


def ks_effect(validation_scores: np.ndarray, test_scores: np.ndarray) -> dict[str, float]:
    """Exact empirical KS distance with no IID null distribution or p-value."""
    val, test = np.sort(validation_scores), np.sort(test_scores)
    support = np.union1d(val, test)
    differences = np.searchsorted(test, support, side="right") / test.size - np.searchsorted(val, support, side="right") / val.size
    index = int(np.argmax(np.abs(differences)))
    return {"KS_D": float(abs(differences[index])), "KS_at_score": float(support[index]),
            "test_minus_validation_cdf_at_KS": float(differences[index])}


def add_prefixed(row: dict[str, Any], prefix: str, metrics: dict[str, float]) -> None:
    row.update({prefix + key: value for key, value in metrics.items()})


def seed_statistics(values: Any) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    mean = float(array.mean()) if n else math.nan
    sd = float(array.std(ddof=1)) if n > 1 else math.nan
    se = sd / math.sqrt(n) if n > 1 else math.nan
    half = float(student_t.ppf(.975, n - 1) * se) if n > 1 else math.nan
    return {"n": n, "mean": mean, "sd": sd, "se": se, "ci95_low": mean - half, "ci95_high": mean + half}


def summarize(frame: pd.DataFrame, keys: list[str], metrics: list[str] | tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for group_key, group in frame.groupby(keys, sort=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(keys, group_key, strict=True))
        row["n_seeds"] = int(group.seed.nunique())
        for metric in metrics:
            values = group[metric] if metric in group else []
            row.update({f"{metric}_{stat}": value for stat, value in seed_statistics(values).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_policy_results(policies: pd.DataFrame, native: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([policies, native.assign(policy="native_archived")], ignore_index=True).set_index(["model_key", "seed", "policy"])
    contrasts: list[tuple[str, str, str, str, str]] = []
    for model in MODELS:
        for label, a, b in (("default_minus_validation", POLICIES[0], POLICIES[1]),
                            ("oracle_minus_validation", POLICIES[2], POLICIES[1]),
                            ("oracle_minus_default", POLICIES[2], POLICIES[0]),
                            ("default_minus_native", POLICIES[0], "native_archived"),
                            ("validation_minus_native", POLICIES[1], "native_archived")):
            contrasts.append((label, model, a, model, b))
    contrasts.append(("RF30_default_minus_Cascade_native", "rf30", POLICIES[0], "cascade", "native_archived"))
    rows = []
    for label, model_a, policy_a, model_b, policy_b in contrasts:
        for seed in SEEDS:
            a = combined.loc[(model_a, seed, policy_a)]
            b = combined.loc[(model_b, seed, policy_b)]
            row = {"contrast": label, "model_A": model_a, "policy_A": policy_a,
                   "model_B": model_b, "policy_B": policy_b, "seed": seed}
            for metric in REPORT_METRICS:
                row["delta_" + metric.replace("_pct", "_pp")] = float(a["test_" + metric] - b["test_" + metric])
            rows.append(row)
    by_seed = pd.DataFrame(rows)
    summary = summarize(by_seed, ["contrast", "model_A", "policy_A", "model_B", "policy_B"],
                        ["delta_" + metric.replace("_pct", "_pp") for metric in REPORT_METRICS])
    return by_seed, summary


def analyse(artifacts: dict[tuple[str, int, str], Artifact]) -> dict[str, pd.DataFrame]:
    policy_rows: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    ecdf_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    constraint_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for seed in SEEDS:
            val, test = artifacts[model, seed, "validation"], artifacts[model, seed, "test"]
            yv, yt = val.y, test.y
            sv, st = val.score.astype(float), test.score.astype(float)
            pv, pt = float(yv.mean()), float(yt.mean())
            base = {"model_key": model, "model": MODEL_LABELS[model], "seed": seed,
                    "source_prior": pv, "target_prior": pt,
                    "source_prior_hex": pv.hex(), "target_prior_hex": pt.hex(), "internal_routing": "frozen",
                    "validation_rows": yv.size, "test_rows": yt.size}
            thresholds = select_policy_thresholds(yv, sv, pt)
            for policy, threshold in thresholds.items():
                val_pred, test_pred = sv >= threshold, st >= threshold
                row = {**base, "policy": policy, "selected_threshold": threshold,
                       "selected_threshold_hex": threshold.hex(),
                       "threshold_comparison": "score >= threshold", "oracle_target_prevalence_used": policy == POLICIES[2],
                       "threshold_selection_set": "none" if policy == POLICIES[0] else "validation",
                       "individual_test_labels_used_for_selection": False, "test_scores_used_for_selection": False,
                       "candidate_rule": "fixed 0.5" if policy == POLICIES[0] else "all distinct validation scores plus +inf",
                       "objective": "fixed" if policy == POLICIES[0] else "weighted F1 on " + ("target-prior-weighted validation" if policy == POLICIES[2] else "raw validation")}
                actual_val = prediction_metrics(yv, val_pred)
                actual_test = prediction_metrics(yt, test_pred)
                expected_target = prediction_metrics(yv, val_pred, target_prior=pt)
                add_prefixed(row, "validation_", actual_val)
                add_prefixed(row, "test_", actual_test)
                add_prefixed(row, "pure_prior_expected_target_", expected_target)
                for metric in REPORT_METRICS:
                    row["test_minus_pure_prior_expected_" + metric.replace("_pct", "_pp")] = actual_test[metric] - expected_target[metric]
                policy_rows.append(row)

            assert test.pred is not None
            native = {**base, "policy": "native_archived", "native_predictions_source": str(test.path),
                      "validation_native_prediction_available": val.pred is not None,
                      "native_vs_default_global_changed_test_predictions": int(np.count_nonzero(test.pred != (st >= .5))),
                      "native_vs_exact_validation_global_changed_test_predictions": int(np.count_nonzero(test.pred != (st >= thresholds[POLICIES[1]]))),
                      "validation_status": "stored native y_pred" if val.pred is not None else "not archived; not imputed"}
            add_prefixed(native, "test_", prediction_metrics(yt, test.pred))
            if val.pred is not None:
                add_prefixed(native, "validation_", prediction_metrics(yv, val.pred))
                add_prefixed(native, "pure_prior_expected_target_", prediction_metrics(yv, val.pred, target_prior=pt))
            native_rows.append(native)

            corrected_val, corrected_test = prior_correct(sv, pv, pt), prior_correct(st, pv, pt)
            for split, method, y, score in (("validation", "raw", yv, sv), ("test", "raw", yt, st), ("test", "oracle_prior_corrected", yt, corrected_test)):
                metrics, bins = calibration_metrics(y, score)
                tag = {**base, "split": split, "method": method, "calibrator_fit_set": "none",
                       "oracle_target_prevalence_used": method != "raw"}
                calibration_rows.append({**tag, **metrics})
                bin_rows.extend({**tag, **item} for item in bins)

            for label, label_name in ((0, "Normal"), (1, "Attack")):
                scores_val, scores_test = sv[yv == label], st[yt == label]
                shift_rows.append({**base, "class_label": label, "class_name": label_name,
                                   "validation_class_rows": scores_val.size, "test_class_rows": scores_test.size,
                                   "validation_mean_score": float(scores_val.mean()), "test_mean_score": float(scores_test.mean()),
                                   "test_minus_validation_mean_score": float(scores_test.mean() - scores_val.mean()),
                                   **ks_effect(scores_val, scores_test)})
                sorted_val, sorted_test = np.sort(scores_val), np.sort(scores_test)
                ecdf_val = np.searchsorted(sorted_val, CLASS_GRID, side="right") / scores_val.size
                ecdf_test = np.searchsorted(sorted_test, CLASS_GRID, side="right") / scores_test.size
                for point, value_val, value_test in zip(CLASS_GRID, ecdf_val, ecdf_test, strict=True):
                    ecdf_rows.append({"model_key": model, "seed": seed, "class_label": label, "class_name": label_name,
                                      "score": point, "validation_ecdf": value_val, "test_ecdf": value_test,
                                      "test_minus_validation_ecdf": value_test - value_val})
                qv, qt = np.quantile(scores_val, QUANTILES, method="linear"), np.quantile(scores_test, QUANTILES, method="linear")
                for q, value_val, value_test in zip(QUANTILES, qv, qt, strict=True):
                    quantile_rows.append({"model_key": model, "seed": seed, "class_label": label, "class_name": label_name,
                                          "quantile": q, "validation_score": value_val, "test_score": value_test,
                                          "test_minus_validation_score": value_test - value_val})

            table_raw = exact_threshold_table(yv, sv)
            table_corrected = exact_threshold_table(yv, corrected_val)
            candidate_identity = np.array_equal(table_raw[COUNT_COLS].to_numpy(), table_corrected[COUNT_COLS].to_numpy())
            if not candidate_identity:
                raise RuntimeError(f"Monotone correction changed candidate partitions: {model}/{seed}")
            for kind, target in CONSTRAINTS:
                selected_raw = select_constraint_row(table_raw, kind, target)
                selected_corrected = select_constraint_row(table_corrected, kind, target)
                tr, tc = float(selected_raw.threshold), float(selected_corrected.threshold)
                vp_raw, vp_corrected = sv >= tr, corrected_val >= tc
                tp_raw, tp_corrected = st >= tr, corrected_test >= tc
                changed_val = int(np.count_nonzero(vp_raw != vp_corrected))
                changed_test = int(np.count_nonzero(tp_raw != tp_corrected))
                if changed_val or changed_test:
                    raise RuntimeError(f"Expected invariance violated: {model}/{seed}/{kind}/{target}; val={changed_val}; test={changed_test}")
                row = {**base, "constraint_kind": kind, "constraint_target_pct": target,
                       "raw_selected_threshold": tr, "corrected_selected_threshold": tc,
                       "raw_selected_threshold_hex": tr.hex(), "corrected_selected_threshold_hex": tc.hex(),
                       "validation_candidate_count": len(table_raw), "candidate_partitions_identical": candidate_identity,
                       "changed_validation_predictions": changed_val, "changed_test_predictions": changed_test,
                       "score_map_applied_to": "same map on validation and test",
                       "result_role": "invariance sanity check, not a test of the label-shift hypothesis"}
                actual_val = prediction_metrics(yv, vp_raw)
                actual_test = prediction_metrics(yt, tp_raw)
                expected_target = prediction_metrics(yv, vp_raw, target_prior=pt)
                add_prefixed(row, "validation_", actual_val)
                add_prefixed(row, "test_", actual_test)
                add_prefixed(row, "corrected_test_", prediction_metrics(yt, tp_corrected))
                add_prefixed(row, "pure_prior_expected_target_", expected_target)
                for metric in REPORT_METRICS:
                    row["test_minus_pure_prior_expected_" + metric.replace("_pct", "_pp")] = actual_test[metric] - expected_target[metric]
                constraint_rows.append(row)
        print(f"Analysed {MODEL_LABELS[model]}: {len(SEEDS)} seeds", flush=True)

    policies, native = pd.DataFrame(policy_rows), pd.DataFrame(native_rows)
    calibrations, shifts, constraints = pd.DataFrame(calibration_rows), pd.DataFrame(shift_rows), pd.DataFrame(constraint_rows)
    paired, paired_summary = paired_policy_results(policies, native)
    quality_columns = [prefix + metric for prefix in ("validation_", "test_", "pure_prior_expected_target_") for metric in REPORT_METRICS]
    residual_columns = ["test_minus_pure_prior_expected_" + metric.replace("_pct", "_pp") for metric in REPORT_METRICS]
    return {
        "threshold_policy_by_seed.csv": policies,
        "threshold_policy_summary.csv": summarize(policies, ["model_key", "model", "policy"], quality_columns + residual_columns),
        "native_policy_by_seed.csv": native,
        "native_policy_summary.csv": summarize(native, ["model_key", "model"], quality_columns),
        "policy_paired_differences_by_seed.csv": paired,
        "policy_paired_summary.csv": paired_summary,
        "calibration_by_seed.csv": calibrations,
        "calibration_summary.csv": summarize(calibrations, ["model_key", "model", "split", "method"], CALIBRATION_METRICS),
        "reliability_bins_by_seed.csv": pd.DataFrame(bin_rows),
        "class_conditional_shift_by_seed.csv": shifts,
        "class_conditional_shift_summary.csv": summarize(shifts, ["model_key", "model", "class_label", "class_name"],
                                                        ["KS_D", "validation_mean_score", "test_mean_score", "test_minus_validation_mean_score"]),
        "class_conditional_ecdf.csv": pd.DataFrame(ecdf_rows),
        "class_conditional_quantiles.csv": pd.DataFrame(quantile_rows),
        "constraint_transfer_invariance_by_seed.csv": constraints,
        "constraint_transfer_invariance_summary.csv": summarize(constraints, ["model_key", "model", "constraint_kind", "constraint_target_pct"],
                                                               quality_columns + residual_columns + ["changed_validation_predictions", "changed_test_predictions"]),
    }


def key_results(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    policies = frames["threshold_policy_by_seed.csv"]
    native = frames["native_policy_by_seed.csv"]
    calibration = frames["calibration_by_seed.csv"]
    shifts = frames["class_conditional_shift_by_seed.csv"]
    constraints = frames["constraint_transfer_invariance_by_seed.csv"]
    quality = {}
    for model in MODELS:
        quality[model] = {}
        for policy in POLICIES:
            group = policies.loc[(policies.model_key == model) & (policies.policy == policy)]
            quality[model][policy] = {metric: seed_statistics(group["test_" + metric]) for metric in REPORT_METRICS}
        quality[model]["native_archived"] = {metric: seed_statistics(native.loc[native.model_key == model, "test_" + metric]) for metric in REPORT_METRICS}
    cascade_calibration = {}
    for split, method in (("validation", "raw"), ("test", "raw"), ("test", "oracle_prior_corrected")):
        group = calibration.loc[(calibration.model_key == "cascade") & (calibration.split == split) & (calibration.method == method)]
        cascade_calibration[f"{split}_{method}"] = {metric: seed_statistics(group[metric]) for metric in CALIBRATION_METRICS}
    far10 = constraints.loc[(constraints.model_key == "cascade") & (constraints.constraint_kind == "far_max") & (constraints.constraint_target_pct == 10)]
    cross = frames["policy_paired_summary.csv"]
    cross = cross.loc[cross.contrast == "RF30_default_minus_Cascade_native"].to_dict(orient="records")
    return {
        "status": "complete", "analysis_role": "post-hoc score-only diagnostics; no causal attribution",
        "n_models": len(MODELS), "n_seeds": len(SEEDS), "paired_model_seeds": len(native),
        "source_prevalence": float(policies.source_prior.iloc[0]), "target_prevalence": float(policies.target_prior.iloc[0]),
        "test_quality_by_model_and_policy": quality,
        "RF30_default_minus_Cascade_native": cross[0],
        "cascade_calibration": cascade_calibration,
        "invariance": {"selected_operating_points": len(constraints),
                       "changed_test_predictions_total": int(constraints.changed_test_predictions.sum()),
                       "changed_validation_predictions_total": int(constraints.changed_validation_predictions.sum()),
                       "all_candidate_partitions_identical": bool(constraints.candidate_partitions_identical.all()),
                       "cascade_FAR10_validation_FAR_pct": seed_statistics(far10.validation_FAR_pct),
                       "cascade_FAR10_test_FAR_pct": seed_statistics(far10.test_FAR_pct)},
        "class_conditional_KS_D": {model: {label: seed_statistics(shifts.loc[(shifts.model_key == model) & (shifts.class_name == label), "KS_D"])
                                           for label in ("Normal", "Attack")} for model in MODELS},
        "scope_notes": SCOPE_NOTES,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main() -> None:
    revision_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=revision_root.parent,
                        help="Parent containing FINAL3 and INISCOM_2027; defaults to this revision's parent")
    parser.add_argument("--out-dir", type=Path, default=revision_root / "generated" / "score_diagnostics")
    parser.add_argument("--validation-dir", type=Path, help="Explicit override for portable artifact copies")
    parser.add_argument("--test-dir", type=Path, help="Explicit override for portable artifact copies")
    parser.add_argument("--overwrite", action="store_true", help="Replace only this script's generated output files")
    args = parser.parse_args()
    project_root, out_dir = args.project_root.resolve(), args.out_dir.resolve()
    score_root = project_root / "FINAL3" / "overleaf_revised" / "generated" / "reconstructed_scores"
    validation_dir = (args.validation_dir or score_root / "validation_scores").resolve()
    test_dir = (args.test_dir or score_root / "predictions").resolve()
    if out_dir in (project_root, revision_root, validation_dir, test_dir):
        raise ValueError("Use a dedicated generated-output directory, not a source/project directory")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_dir}; use --overwrite for generated files")
    started = datetime.now(timezone.utc).isoformat()
    artifacts, audits = load_and_verify(validation_dir, test_dir)
    print(f"Verified {len(artifacts)} NPZ inputs; test IDs/labels aligned; validation IDs not assumed", flush=True)
    frames = analyse(artifacts)
    frames["input_audit.csv"] = audits
    # A second read detects any concurrent modification, and also verifies that
    # the analysis has left all archived raw artifacts unchanged.
    for item in artifacts.values():
        if file_hash(item.path) != item.sha256:
            raise RuntimeError(f"An input changed during analysis: {item.path}")
    results = key_results(frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(out_dir / name, index=False, encoding="utf-8-sig", float_format=CSV_FLOAT_FORMAT)
    (out_dir / "key_results.json").write_text(json.dumps(json_safe(results), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    provenance = project_root / "FINAL3" / "overleaf_revised" / "scripts" / "validation_constrained_operating_points.py"
    source_manifest = score_root / "MANIFEST.json"
    output_names = sorted([*frames, "key_results.json"])
    manifest = {
        "schema_version": "1.1", "status": "complete", "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "script": str(Path(__file__).resolve()),
        "script_sha256": file_hash(Path(__file__)), "command": sys.argv,
        "project_root": str(project_root), "validation_dir": str(validation_dir), "test_dir": str(test_dir), "out_dir": str(out_dir),
        "models": list(MODELS), "seeds": list(SEEDS), "paired_model_seeds": len(artifacts) // 2,
        "input_artifacts_unchanged_after_analysis": True, "input_hash_table": "input_audit.csv",
        "fixed_test_row_ids_verified_across_models_and_seeds": True,
        "validation_row_ids_present": bool(audits.loc[audits.split == "validation", "row_id_key"].notna().all()),
        "raw_scores_modified_or_clipped": False, "analysis_comparison_precision": "float64 promoted from archived score dtype",
        "numeric_serialization": {"CSV": "17 significant digits; read with a correctly rounded parser, e.g. pandas float_precision='round_trip'",
                                  "exact_thresholds_and_priors": "Use float.fromhex on the corresponding *_hex column for lossless reuse, including the +inf sentinel",
                                  "reason": "Rounding an observed threshold upward can exclude an entire equal-score block; display-rounded values must not be reapplied"},
        "native_policy_reference": "exact stored test y_pred; unavailable component validation predictions are not imputed",
        "threshold_candidate_rule": "all distinct validation scores plus +inf; score >= threshold; equal-score blocks remain intact",
        "F1_tie_rule": "F1-W descending, FAR ascending, recall descending, distance to 0.5 ascending, threshold descending",
        "constraint_tie_rule_provenance": {"path": str(provenance), "sha256": file_hash(provenance) if provenance.is_file() else None},
        "reconstruction_manifest": {"path": str(source_manifest), "sha256": file_hash(source_manifest) if source_manifest.is_file() else None},
        "calibration": {"bins": 15, "binning": "equal width; last bin includes score 1", "log_loss_clip_only": [1e-15, 1-1e-15], "calibrator_fitted": False},
        "class_conditional_ecdf": {"grid_points": len(CLASS_GRID), "grid_start": 0.0, "grid_end": 1.0, "grid_step": .001,
                                   "definition": "F(s)=P(score<=s); searchsorted side=right; a grid-sampled empirical CDF, not the full unique-score staircase",
                                   "FAR_warning": "FAR(t)=1-F(t-), not 1-F(t), when scores tie at threshold t",
                                   "KS_computation": "exact union of observed class-conditional scores, independent of this display grid"},
        "quantile_method": "numpy.quantile(method='linear')",
        "oracle_prior_source": "mean of test y_true; sole target-label information used for weighting/correction",
        "oracle_flag_scope": "In threshold-policy rows the flag refers to threshold selection; in calibration rows it refers to score transformation. Expected-target metrics always use the stated oracle prior.",
        "individual_test_labels_used_for_selection": False, "test_scores_used_for_selection": False,
        "invariance_sanity_check": results["invariance"], "KS_significance_tests": "not performed",
        "confidence_intervals": "two-sided Student-t 95% intervals across paired training/split seeds on the fixed shared test set; not population intervals",
        "environment": {"python_executable": sys.executable, "python": platform.python_version(), "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")}},
        "scope_notes": SCOPE_NOTES,
        "outputs": [{"file": name, "rows": len(frames[name]) if name in frames else None, "sha256": file_hash(out_dir / name)} for name in output_names],
    }
    (out_dir / "manifest.json").write_text(json.dumps(json_safe(manifest), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    compact = {"status": "complete", "out_dir": str(out_dir), "paired_model_seeds": len(artifacts)//2,
               "invariance_changed_test_predictions": results["invariance"]["changed_test_predictions_total"],
               "RF30_test_F1_weighted_pct": {policy: quality["F1_weighted_pct"]["mean"] for policy, quality in results["test_quality_by_model_and_policy"]["rf30"].items()},
               "Cascade_test_ECE_raw": results["cascade_calibration"]["test_raw"]["ECE_15_equal_width"]["mean"],
               "Cascade_test_ECE_prior_corrected": results["cascade_calibration"]["test_oracle_prior_corrected"]["ECE_15_equal_width"]["mean"]}
    print(json.dumps(json_safe(compact), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
