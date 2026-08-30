"""Independently audit a completed RF-30 random-split control run.

The reviewer never fits a model. It verifies immutable hashes, reconstructs the
64/16/20 row partitions, independently enumerates validation cutoffs, and
recomputes sklearn metrics, class-conditional KS statistics, summaries, and the
paired seed interval from the saved float32 scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, t
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score)
from sklearn.model_selection import train_test_split


SEEDS = (21, 42, 84, 168, 336, 348184071, 362797685, 883000906,
         1365468824, 1486213216)
POLICIES = ("global_default_0p5", "global_validation_f1",
            "global_oracle_prior_weighted_validation_f1")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interval(values) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("A seed interval needs at least two finite values")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / np.sqrt(len(values))
    half = float(t.ppf(.975, len(values) - 1) * se)
    return {"n": len(values), "mean": mean, "sd": sd, "se": se,
            "ci95_low": mean - half, "ci95_high": mean + half}


def select_thresholds(y, score, target_prior) -> dict[str, float]:
    """Exhaustive selector independent of the experiment implementation."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    if y.shape != score.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Invalid binary validation arrays")
    if not np.isfinite(score).all() or (score < 0).any() or (score > 1).any():
        raise ValueError("Scores must be finite and in [0,1]")
    candidates = np.r_[np.unique(score), np.inf]
    normal = np.sort(score[y == 0])
    attack = np.sort(score[y == 1])
    tn = np.searchsorted(normal, candidates, side="left").astype(float)
    fn = np.searchsorted(attack, candidates, side="left").astype(float)
    fp, tp = len(normal) - tn, len(attack) - fn

    def choose(w0, w1):
        a, b, c, d = tn * w0, fp * w0, fn * w1, tp * w1
        f0 = np.divide(2 * a, 2 * a + b + c, out=np.zeros_like(a),
                       where=(2 * a + b + c) != 0)
        f1 = np.divide(2 * d, 2 * d + b + c, out=np.zeros_like(d),
                       where=(2 * d + b + c) != 0)
        weighted = ((a + b) * f0 + (c + d) * f1) / (a + b + c + d)
        # Max F1, then min FAR, max recall, nearest 0.5, then larger cutoff.
        order = np.lexsort((-candidates, np.abs(candidates - .5),
                            -d / (c + d), b / (a + b), -weighted))
        return float(candidates[order[0]])

    source_prior = float(y.mean())
    oracle = choose((1 - target_prior) / (1 - source_prior),
                    target_prior / source_prior)
    return dict(zip(POLICIES, (.5, choose(1., 1.), oracle)))


def metrics(y, pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "TN": float(tn), "FP": float(fp), "FN": float(fn), "TP": float(tp),
        "sample_mass": float(len(y)),
        "F1_weighted_pct": 100 * f1_score(y, pred, average="weighted", zero_division=0),
        "F1_macro_pct": 100 * f1_score(y, pred, average="macro", zero_division=0),
        "F1_attack_pct": 100 * f1_score(y, pred, pos_label=1, zero_division=0),
        "F1_normal_pct": 100 * f1_score(y, pred, pos_label=0, zero_division=0),
        "Attack_precision_pct": 100 * precision_score(y, pred, zero_division=0),
        "Attack_recall_pct": 100 * recall_score(y, pred, zero_division=0),
        "FAR_pct": 100 * fp / (tn + fp),
        "Accuracy_pct": 100 * accuracy_score(y, pred),
        "Balanced_accuracy_pct": 100 * balanced_accuracy_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
        "attack_prevalence_pct": 100 * np.mean(y),
    }


def summarize(frame, groups, columns):
    rows = []
    grouped = frame.groupby(groups, sort=False, dropna=False) if groups else [((), frame)]
    for key, values in grouped:
        key = (key,) if groups and not isinstance(key, tuple) else key
        row = {name: value for name, value in zip(groups, key)}
        row["n_seeds"] = int(values.seed.nunique())
        for column in columns:
            row[f"{column}_mean"] = float(values[column].mean())
            row[f"{column}_sd"] = float(values[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def compare_frames(actual, saved, keys):
    if actual.duplicated(keys).any() or saved.duplicated(keys).any():
        raise ValueError(f"Duplicate records for {keys}")
    actual = actual.set_index(keys).sort_index()
    saved = saved.set_index(keys).sort_index()
    if not actual.index.equals(saved.index):
        raise ValueError(f"Record keys differ for {keys}")
    for column in actual.columns:
        if column not in saved:
            raise ValueError(f"Saved output lacks {column}")
        if pd.api.types.is_numeric_dtype(actual[column]):
            if not np.allclose(actual[column].to_numpy(float),
                               saved[column].to_numpy(float),
                               atol=1e-10, rtol=1e-10):
                raise ValueError(f"Recalculation differs: {column}")
        elif not actual[column].astype(str).equals(saved[column].astype(str)):
            raise ValueError(f"Text differs: {column}")


def review(root: Path, run: Path, output: Path):
    if output == run or output.is_relative_to(run):
        raise ValueError("Review output must be outside the immutable run")
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("seeds") != list(SEEDS):
        raise ValueError("Incomplete or different seed design")
    if manifest.get("partition") != {"fit": .64, "validation": .16, "heldout_test": .20}:
        raise ValueError("Unexpected random-split design")
    if (manifest.get("fitted_models"), manifest.get("policy_rows"),
            manifest.get("diagnostic_rows")) != (10, 30, 10):
        raise ValueError("Unexpected run dimensions")

    consumed = {str(manifest_path.resolve()): sha(manifest_path)}
    for name, expected in manifest["input_hashes"].items():
        path = Path(name).resolve()
        if sha(path) != expected:
            raise ValueError(f"Changed experiment input: {path}")
        consumed[str(path)] = expected
    for name, expected in manifest["outputs"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or sha(path) != expected:
            raise ValueError(f"Changed or unsafe experiment output: {name}")
        consumed[str(path)] = expected

    expected_scores = {f"rf30__seed_{seed}.npz" for seed in SEEDS}
    if {p.name for p in (run / "scores").glob("*.npz")} != expected_scores:
        raise ValueError("Score files differ from the ten-seed design")

    train_path = next(Path(p) for p in manifest["input_hashes"]
                      if Path(p).name == "UNSW_NB15_training-set.csv")
    frame = pd.read_csv(train_path, usecols=["label"])
    frame["_row_id"] = np.arange(len(frame), dtype=np.int64)
    if len(frame) != 175341 or set(frame.label.unique()) != {0, 1}:
        raise ValueError("Unexpected official training file")

    policy_rows, diagnostic_rows, split_checks = [], [], []
    for seed in SEEDS:
        development, heldout = train_test_split(
            frame, test_size=.20, random_state=seed, stratify=frame.label, shuffle=True)
        fit, validation = train_test_split(
            development, test_size=.20, random_state=seed + 1,
            stratify=development.label, shuffle=True)
        with np.load(run / "scores" / f"rf30__seed_{seed}.npz", allow_pickle=False) as saved:
            arrays = {name: saved[name].copy() for name in saved.files}
        for split, expected in (("validation", validation), ("test", heldout)):
            y = arrays[f"{split}_y"]
            score = arrays[f"{split}_score"]
            row_id = arrays[f"{split}_row_id"]
            if score.dtype != np.float32 or y.dtype.kind not in "iu" or row_id.dtype.kind not in "iu":
                raise ValueError("Unexpected saved score dtypes")
            if not np.array_equal(row_id, expected._row_id.to_numpy()) or not np.array_equal(y, expected.label.to_numpy()):
                raise ValueError(f"Saved {split} identity differs at seed {seed}")
            if not np.isfinite(score).all() or (score < 0).any() or (score > 1).any():
                raise ValueError("Invalid saved scores")
        yv, sv = arrays["validation_y"], arrays["validation_score"]
        yt, st = arrays["test_y"], arrays["test_score"]
        selected = select_thresholds(yv, sv, float(yt.mean()))
        for policy, cutoff in selected.items():
            result = {"seed": seed, "policy": policy, "cutoff": cutoff,
                      "cutoff_hex": cutoff.hex(),
                      "validation_attack_prior": float(yv.mean()),
                      "test_attack_prior": float(yt.mean())}
            for split, y, score in (("validation", yv, sv), ("test", yt, st)):
                result.update({f"{split}_{name}": value
                               for name, value in metrics(y, score >= cutoff).items()})
            policy_rows.append(result)
        diagnostic_rows.append({
            "seed": seed,
            "normal_KS_D": float(ks_2samp(sv[yv == 0], st[yt == 0], method="asymp").statistic),
            "attack_KS_D": float(ks_2samp(sv[yv == 1], st[yt == 1], method="asymp").statistic),
        })
        split_checks.append({"seed": seed, "fit_rows": len(fit),
                             "validation_rows": len(validation), "test_rows": len(heldout),
                             "row_ids_disjoint": not (set(fit._row_id) & set(validation._row_id)
                                                      or set(fit._row_id) & set(heldout._row_id)
                                                      or set(validation._row_id) & set(heldout._row_id))})

    policies = pd.DataFrame(policy_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    summary = summarize(policies, ["policy"], [
        "cutoff", "validation_F1_weighted_pct", "test_F1_weighted_pct",
        "validation_FAR_pct", "test_FAR_pct",
        "validation_Attack_recall_pct", "test_Attack_recall_pct"])
    diagnostic_summary = summarize(diagnostics, [], ["normal_KS_D", "attack_KS_D"])
    indexed = policies.pivot(index="seed", columns="policy", values="test_F1_weighted_pct")
    contrast = pd.DataFrame([{"contrast": "default minus global validation",
                              **interval(indexed[POLICIES[0]] - indexed[POLICIES[1]])}])

    read = lambda name: pd.read_csv(run / name, float_precision="round_trip")
    compare_frames(policies, read("policies_by_seed.csv"), ["seed", "policy"])
    compare_frames(diagnostics, read("conditional_shift_by_seed.csv"), ["seed"])
    compare_frames(summary, read("policy_summary.csv"), ["policy"])
    compare_frames(diagnostic_summary, read("conditional_shift_summary.csv"), ["n_seeds"])
    compare_frames(contrast, read("paired_contrasts.csv"), ["contrast"])

    output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, data in (("policies_by_seed", policies), ("policy_summary", summary),
                       ("conditional_shift_by_seed", diagnostics),
                       ("conditional_shift_summary", diagnostic_summary),
                       ("paired_contrasts", contrast),
                       ("split_identity_checks", pd.DataFrame(split_checks))):
        path = output / f"{name}.csv"
        data.to_csv(path, index=False, float_format="%.17g")
        outputs[path.name] = sha(path)
    record = {
        "status": "passed", "reviewed_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest["run_id"], "run_directory": str(run),
        "script": str(Path(__file__).resolve()), "script_sha256": sha(__file__),
        "input_hashes": consumed, "outputs": outputs,
        "input_and_run_files_verified": len(consumed), "score_files_checked": 10,
        "partition_identity_checks": 20, "policies_recomputed": 30,
        "metric_sets_recomputed": 60, "ks_statistics_recomputed": 20,
        "paired_contrasts_recomputed": 1, "new_training_performed": False,
        "independent_implementation": "independent exhaustive cutoff enumeration, sklearn metrics, scipy KS, reconstructed sklearn row splits",
        "limitations": [
            "Saved scores and run provenance are verified; fitted model objects were not archived and no independent refit was performed.",
            "Random-row within-training partitions can share duplicates or collection context and are not independent deployments."
        ]
    }
    (output / "review_manifest.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items()
                      if k not in ("input_hashes", "outputs")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--revision-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.revision_root.resolve()
    review(root, args.run_dir.resolve(), root / "generated/random_split_review")


if __name__ == "__main__":
    main()
