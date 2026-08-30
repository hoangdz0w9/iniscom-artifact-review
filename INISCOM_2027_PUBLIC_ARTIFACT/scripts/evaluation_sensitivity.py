"""Frozen-prediction unique-only audit and paired test-row bootstrap.

No training, gate/cutoff selection, timing or source-data mutation. Bootstrap
conditions on the ten fitted systems and their archived cutoffs. It resamples
test rows jointly across ALL systems and seeds, NOT ten independent test sets.
Joint label/prediction patterns compress the exact ordinary row-bootstrap law;
they are computational categories, never claimed to be independent sessions.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from score_shift_analysis import (
    MODELS, SEEDS, POLICIES, file_hash, load_artifact,
    metrics_from_counts, prediction_metrics,
)

CONFIGS = (("rf30", "native_archived"), ("cascade", "native_archived"),
           ("rf30", POLICIES[0]), ("rf300", POLICIES[0]), ("rf30", POLICIES[1]))
CONTRASTS = (("RF30_native_minus_Cascade_native", 0, 1),
             ("RF30_default_minus_Cascade_native", 2, 1),
             ("RF300_default_minus_RF30_default", 3, 2),
             ("RF30_default_minus_RF30_global_val", 2, 4))


def feature_hash(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Reproduce the existing 42-predictor audit (no tolerance/rounding)."""
    subset = frame.loc[:, columns].copy()
    for col in columns:
        if subset[col].dtype == "object" or isinstance(subset[col].dtype, pd.StringDtype):
            subset[col] = subset[col].astype("string").fillna("<NA>").str.strip()
    return pd.util.hash_pandas_object(subset, index=False).to_numpy(np.uint64)


def aligned_unique_mask(train, test, row_ids, y):
    train = train.copy()
    test = test.copy()
    for frame in (train, test):
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        if frame.columns.duplicated().any():
            raise ValueError("Duplicate normalized columns")
    if set(train.columns) != set(test.columns):
        raise ValueError("Train/test columns differ")
    columns = [c for c in train.columns if c not in {"id", "label", "attack_cat"}]
    if len(columns) != 42:
        raise ValueError("Expected exactly 42 official model-visible predictors")
    row_ids = np.asarray(row_ids)
    if not np.array_equal(np.sort(row_ids), np.arange(len(test))):
        raise ValueError("Score IDs must be a permutation of zero-based official test positions")
    if not np.array_equal(test["id"].to_numpy(), np.arange(1, len(test) + 1)):
        raise ValueError("Official CSV IDs do not confirm file order")
    if not np.array_equal(test["label"].to_numpy()[row_ids], y):
        raise ValueError("Official test labels and score row IDs disagree")
    if not np.isin(train["label"], [0, 1]).all():
        raise ValueError("Nonbinary official training labels")
    train_hash, test_hash = feature_hash(train, columns), feature_hash(test, columns)
    return ~np.isin(test_hash[row_ids], np.unique(train_hash)), columns


def joint_patterns(y, predictions):
    """predictions[configuration, seed, row]; preserve every pairing."""
    y, predictions = np.asarray(y), np.asarray(predictions)
    if predictions.ndim != 3 or predictions.shape[2] != y.size or y.ndim != 1:
        raise ValueError("Expected aligned configuration x seed x row predictions")
    if not y.size or not np.isin(y, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Nonempty binary arrays required")
    flat = predictions.reshape(-1, y.size)
    packed = np.packbits(np.column_stack((y, flat.T)), axis=1)
    _, first, inverse, counts = np.unique(packed, axis=0, return_index=True,
                                          return_inverse=True, return_counts=True)
    return y[first], flat[:, first].T, counts, inverse


def weighted_f1_from_patterns(pattern_y, pattern_pred, weights):
    """Return [replicate, configuration*seed] F1-W, from joint multiplicities."""
    weights = np.atleast_2d(weights).astype(float)
    pred = pattern_pred.astype(float)
    attack = pattern_y.astype(float)
    tp = weights @ (pred * attack[:, None])
    fp = weights @ (pred * (1 - attack[:, None]))
    n1 = (weights @ attack)[:, None]
    n0 = weights.sum(axis=1)[:, None] - n1
    return metrics_from_counts(n0 - fp, fp, n1 - tp, tp)["F1_weighted_pct"]


def paired_row_bootstrap(y, predictions, n_bootstrap, rng_seed, contrasts=CONTRASTS):
    if n_bootstrap < 1 or rng_seed < 0:
        raise ValueError("Positive replicate count and nonnegative RNG seed required")
    py, pp, counts, _ = joint_patterns(y, predictions)
    configs, seeds, n_rows = predictions.shape
    full = weighted_f1_from_patterns(py, pp, counts).reshape(configs, seeds)
    rng = np.random.default_rng(rng_seed)
    draws = np.empty((n_bootstrap, len(contrasts)), dtype=float)
    for start in range(0, n_bootstrap, 128):
        stop = min(start + 128, n_bootstrap)
        # Exactly the joint multiplicities induced by N IID draws with replacement.
        multiplicities = rng.multinomial(n_rows, counts / n_rows, size=stop - start)
        f1 = weighted_f1_from_patterns(py, pp, multiplicities).reshape(-1, configs, seeds)
        for i, (_, a, b) in enumerate(contrasts):
            draws[start:stop, i] = (f1[:, a, :] - f1[:, b, :]).mean(axis=1)
    rows = []
    for i, (name, a, b) in enumerate(contrasts):
        low, high = np.quantile(draws[:, i], [.025, .975], method="linear")
        rows.append({"contrast": name, "n_rows": n_rows, "n_seeds": seeds,
                     "mean_difference_pp": float((full[a] - full[b]).mean()),
                     "bootstrap_ci95_low_pp": low, "bootstrap_ci95_high_pp": high,
                     "bootstrap_sd_pp": float(draws[:, i].std(ddof=1)) if n_bootstrap > 1 else 0.,
                     "replicates": n_bootstrap, "rng_seed": rng_seed,
                     "joint_prediction_patterns": len(counts)})
    return pd.DataFrame(rows), pd.DataFrame(draws, columns=[c[0] for c in contrasts])


def run(root, project, out, n_bootstrap, rng_seed):
    inputs = {}
    def track(path, expected=None):
        path = Path(path).resolve()
        actual = file_hash(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Input hash mismatch: {path}")
        inputs[str(path)] = actual
        return path
    def read(path):
        return pd.read_csv(track(path), float_precision="round_trip", low_memory=False)
    score_root = root / "generated/score_diagnostics"
    manifest = json.loads(track(score_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Incomplete score analysis")
    for entry in manifest["outputs"]:
        track(score_root / Path(entry.get("file") or entry["path"]).name, entry["sha256"])
    policy = read(score_root / "threshold_policy_by_seed.csv")
    native = read(score_root / "native_policy_by_seed.csv")
    audit = read(score_root / "input_audit.csv")
    if len(policy) != 240 or policy.duplicated(["model_key", "policy", "seed"]).any():
        raise ValueError("Expected unique 8 x 3 x 10 policy rows")
    policy = policy.set_index(["model_key", "policy", "seed"])
    native = native.set_index(["model_key", "seed"])
    artifacts = {}
    for model in MODELS:
        for seed in SEEDS:
            record = audit[(audit.model_key == model) & (audit.seed == seed) & audit.split.eq("test")]
            if len(record) != 1:
                raise ValueError("Missing/ambiguous test artifact")
            record = record.iloc[0]
            item = load_artifact(track(record.path, record.sha256), "test")
            if artifacts:
                first = next(iter(artifacts.values()))
                if not np.array_equal(item.row_id, first.row_id) or not np.array_equal(item.y, first.y):
                    raise ValueError("Test row pairing changed across models/seeds")
            artifacts[model, seed] = item
    first = next(iter(artifacts.values()))
    train = read(project / "UNSW_NB15_training-set.csv")
    test = read(project / "UNSW_NB15_testing-set.csv")
    mask, columns = aligned_unique_mask(train, test, first.row_id, first.y)
    if len(mask) != 82332 or np.count_nonzero(~mask) != 8541 or len(train) != 175341:
        raise ValueError("Official split/overlap does not reproduce the audited counts")
    unique_rows = []
    predictions = {}
    checked = 0
    for (model, seed), item in artifacts.items():
        for name in (*POLICIES, "native_archived"):
            saved = native.loc[model, seed] if name == "native_archived" else policy.loc[model, name, seed]
            cutoff = None if name == "native_archived" else float.fromhex(saved.selected_threshold_hex)
            pred = item.pred if cutoff is None else item.score.astype(float) >= cutoff
            actual = prediction_metrics(item.y, pred)
            for metric in ("TN", "FP", "FN", "TP", "F1_weighted_pct", "FAR_pct", "Attack_recall_pct"):
                if not np.isclose(actual[metric], saved["test_" + metric], rtol=0, atol=1e-10):
                    raise ValueError(f"Frozen prediction reconstruction failed: {model}/{seed}/{name}/{metric}")
            checked += 1
            predictions[model, name, seed] = pred
            selected = prediction_metrics(item.y[mask], pred[mask])
            unique_rows.append({"model_key": model, "seed": seed, "policy": name,
                                "selected_threshold_hex": "native" if cutoff is None else cutoff.hex(),
                                "full_F1_weighted_pct": actual["F1_weighted_pct"],
                                "unique_minus_full_F1_pp": selected["F1_weighted_pct"] - actual["F1_weighted_pct"],
                                **selected})
    unique_frame = pd.DataFrame(unique_rows)
    summary = unique_frame.groupby(["model_key", "policy"], sort=False).agg(
        n_seeds=("seed", "nunique"), F1_mean=("F1_weighted_pct", "mean"),
        F1_sd=("F1_weighted_pct", "std"), attack_F1_mean=("F1_attack_pct", "mean"),
        FAR_mean=("FAR_pct", "mean"), recall_mean=("Attack_recall_pct", "mean"),
        FP_mean=("FP", "mean"), FN_mean=("FN", "mean"),
        full_F1_mean=("full_F1_weighted_pct", "mean"), delta_F1_mean=("unique_minus_full_F1_pp", "mean")).reset_index()
    summary["rank_unique"] = summary.groupby("policy").F1_mean.rank(method="min", ascending=False)
    summary["rank_full"] = summary.groupby("policy").full_F1_mean.rank(method="min", ascending=False)
    cube = np.array([[predictions[m, p, seed] for seed in SEEDS] for m, p in CONFIGS], dtype=np.int8)
    boot_summaries, boot_draws = [], []
    with threadpool_limits(limits=1):
        for i, (subset, keep) in enumerate((("full_test", np.ones(len(mask), dtype=bool)), ("feature_unique", mask))):
            print(f"Bootstrap {subset}: {n_bootstrap:,} shared-row resamples; fixed ten fits/cutoffs", flush=True)
            report, draws = paired_row_bootstrap(first.y[keep], cube[:, :, keep], n_bootstrap, rng_seed + i)
            boot_summaries.append(report.assign(subset=subset))
            boot_draws.append(draws.assign(subset=subset, replicate=np.arange(n_bootstrap)))
    out.mkdir(parents=True, exist_ok=False)
    outputs = {}
    def write_csv(name, frame):
        path = out / name
        frame.to_csv(path, index=False, float_format="%.17g")
        outputs[name] = file_hash(path)
    write_csv("unique_by_seed.csv", unique_frame)
    write_csv("unique_summary.csv", summary)
    write_csv("bootstrap_summary.csv", pd.concat(boot_summaries, ignore_index=True))
    write_csv("bootstrap_replicates.csv", pd.concat(boot_draws, ignore_index=True))
    write_csv("test_subset_mask.csv", pd.DataFrame({"test_row_id": first.row_id, "label": first.y, "feature_unique": mask}))
    # Count paired operational tradeoffs directly; counts are labeled rows, not incidents.
    tradeoff = []
    for seed in SEEDS:
        a = prediction_metrics(first.y, predictions["rf30", POLICIES[0], seed])
        b = prediction_metrics(first.y, predictions["cascade", "native_archived", seed])
        tradeoff.append({"seed": seed, "RF_FP": a["FP"], "Cascade_FP": b["FP"],
                         "RF_FN": a["FN"], "Cascade_FN": b["FN"],
                         "fewer_false_positives": b["FP"] - a["FP"],
                         "additional_false_negatives": a["FN"] - b["FN"]})
    write_csv("headline_error_counts.csv", pd.DataFrame(tradeoff))
    track(Path(__file__))
    track(Path(__file__).with_name("score_shift_analysis.py"))
    record = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "script": str(Path(__file__).resolve()), "input_hashes": inputs, "outputs": outputs,
              "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
              "full_test_rows": len(mask), "overlapping_rows": int((~mask).sum()),
              "unique_rows": int(mask.sum()), "unique_normal": int(((first.y == 0) & mask).sum()),
              "unique_attack": int(((first.y == 1) & mask).sum()), "feature_columns": columns,
              "frozen_reconstruction_checks": checked, "bootstrap_replicates_per_subset": n_bootstrap,
              "bootstrap_rng_seed": rng_seed, "training_or_threshold_reselection": False,
              "scope": ["Post-hoc composition sensitivity; no deduplicated retraining.",
                        "Unique means absent from official training, not internally deduplicated test.",
                        "Oracle cutoffs retain original full-test prior; subset priors never retune them.",
                        "Bootstrap is ordinary unstratified paired row resampling with replacement.",
                        "All models/seeds share each row draw; mean of paired seed F1 differences, not pooled predictions.",
                        "95% percentile intervals condition on fitted systems/cutoffs; seeds are not resampled.",
                        "IID-row sensitivity only: flow/session dependence is not modeled; not a deployment population CI.",
                        "No combination of seed-t and row-bootstrap intervals, no new confirmatory significance claims.",
                        "64-bit feature hashes reproduce the existing audit, without separate collision verification."]}
    (out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("unique_rows", "unique_normal", "unique_attack", "frozen_reconstruction_checks")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=20260828)
    args = parser.parse_args()
    root = args.revision_root.resolve()
    out = (args.output_dir or root / "generated/evaluation_sensitivity").resolve()
    if out.exists():
        raise FileExistsError("Output exists; choose a new --output-dir to preserve the reviewed run")
    if args.replicates < 1000:
        raise ValueError("Use at least 1,000 replicates for the reported percentile intervals")
    run(root, (args.project_root or root.parent).resolve(), out, args.replicates, args.rng_seed)


if __name__ == "__main__":
    main()
