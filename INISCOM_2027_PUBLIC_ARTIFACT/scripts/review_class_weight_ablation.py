"""Audit a completed class-weight run without refitting or modifying its files.

Independent NumPy cutoff enumeration and sklearn metrics check saved scores;
paired intervals and all published run summaries are recomputed. Outputs are
review records, not additional experimental runs or independent test datasets.
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
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef)

SEEDS = (21, 42, 84, 168, 336, 348184071, 362797685, 883000906, 1365468824, 1486213216)
ARMS = ("balanced_subsample", "none")
POLICIES = ("global_default_0p5", "global_validation_f1", "global_oracle_prior_weighted_validation_f1")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def interval(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("Intervals require at least two finite seed observations")
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    se = sd / np.sqrt(len(a))
    half = float(t.ppf(.975, len(a)-1) * se)
    return dict(n=len(a), mean=mean, sd=sd, se=se, ci95_low=mean-half, ci95_high=mean+half)


def thresholds(y, score, target_prior):
    """Independent exhaustive search, including score equality and +inf."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    if y.shape != score.shape or set(np.unique(y)) != {0, 1} or not np.isfinite(score).all():
        raise ValueError("Invalid validation arrays")
    candidates = np.r_[np.unique(score), np.inf]
    normal, attack = np.sort(score[y == 0]), np.sort(score[y == 1])
    tn = np.searchsorted(normal, candidates, side="left").astype(float)
    fn = np.searchsorted(attack, candidates, side="left").astype(float)
    fp, tp = len(normal)-tn, len(attack)-fn

    def choose(w0, w1):
        a, b, c, d = tn*w0, fp*w0, fn*w1, tp*w1
        f0 = np.divide(2*a, 2*a+b+c, out=np.zeros_like(a), where=(2*a+b+c) != 0)
        f1 = np.divide(2*d, 2*d+b+c, out=np.zeros_like(d), where=(2*d+b+c) != 0)
        weighted = ((a+b)*f0+(c+d)*f1)/(a+b+c+d)
        # Same objective/tie hierarchy, independently implemented via lexsort.
        order = np.lexsort((-candidates, np.abs(candidates-.5), -d/(c+d), b/(a+b), -weighted))
        return float(candidates[order[0]])

    prior = float(y.mean())
    return dict(zip(POLICIES, (.5, choose(1., 1.), choose((1-target_prior)/(1-prior), target_prior/prior))))


def metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(TN=float(tn), FP=float(fp), FN=float(fn), TP=float(tp), sample_mass=float(len(y)),
                F1_weighted_pct=100*f1_score(y, pred, average="weighted", zero_division=0),
                F1_macro_pct=100*f1_score(y, pred, average="macro", zero_division=0),
                F1_attack_pct=100*f1_score(y, pred, pos_label=1, zero_division=0),
                F1_normal_pct=100*f1_score(y, pred, pos_label=0, zero_division=0),
                Attack_precision_pct=100*precision_score(y, pred, zero_division=0),
                Attack_recall_pct=100*recall_score(y, pred, zero_division=0),
                FAR_pct=100*fp/(tn+fp), Accuracy_pct=100*accuracy_score(y, pred),
                Balanced_accuracy_pct=100*balanced_accuracy_score(y, pred),
                MCC=matthews_corrcoef(y, pred), attack_prevalence_pct=100*np.mean(y))


def calibration(y, s):
    s = s.astype(float)
    p = np.clip(s, 1e-15, 1-1e-15)
    bins = np.minimum((s*15).astype(int), 14)
    ece = sum(abs(float(s[bins == k].mean()-y[bins == k].mean()))*np.mean(bins == k)
              for k in range(15) if (bins == k).any())
    return dict(Brier_score=float(np.mean((s-y)**2)), log_loss=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
                ECE_15_equal_width=float(ece), mean_score=float(s.mean()), calibration_in_large=float(s.mean()-y.mean()))


def compare_frames(actual, saved, keys):
    if actual.duplicated(keys).any() or saved.duplicated(keys).any():
        raise ValueError(f"Duplicate rows: {keys}")
    a, b = actual.set_index(keys).sort_index(), saved.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise ValueError(f"Record keys changed: {keys}")
    for col in a.columns:
        if col not in b:
            raise ValueError(f"Missing saved column: {col}")
        if pd.api.types.is_numeric_dtype(a[col]):
            if not np.allclose(a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float), atol=1e-10, rtol=1e-10):
                raise ValueError(f"Recalculation differs from archive: {col}")
        elif not (a[col].astype(str) == b[col].astype(str)).all():
            raise ValueError(f"Text field differs: {col}")


def paired(frame):
    pivot = frame.pivot(index="seed", columns=["arm", "policy"], values="test_F1_weighted_pct").sort_index()
    if set(pivot.index) != set(SEEDS) or pivot.isna().any().any():
        raise ValueError("Pairing requires the complete ten-seed design")
    rows = []
    for p in POLICIES:
        rows.append(dict(contrast=f"none minus balanced_subsample: {p}", **interval(pivot["none", p]-pivot["balanced_subsample", p])))
    gaps = {}
    for arm in ARMS:
        gaps[arm] = pivot[arm, POLICIES[0]]-pivot[arm, POLICIES[1]]
        rows.append(dict(contrast=f"default minus global validation: {arm}", **interval(gaps[arm])))
    rows.append(dict(contrast="change in default-minus-validation gap: none minus balanced_subsample", **interval(gaps["none"]-gaps["balanced_subsample"])))
    for p in POLICIES[1:]:
        cuts = frame[frame.policy == p].pivot(index="seed", columns="arm", values="cutoff").sort_index()
        rows.append(dict(contrast=f"cutoff change (score units), none minus balanced_subsample: {p}", **interval(cuts["none"]-cuts["balanced_subsample"])))
    return pd.DataFrame(rows)


def review(root, run, output):
    if output == run or output.is_relative_to(run):
        raise ValueError("Review must not write inside the immutable run")
    consumed = {}

    def check(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected and digest != expected:
            raise ValueError(f"Hash mismatch: {path}")
        consumed[str(path)] = digest
        return path

    manifest = json.loads(check(run / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["seeds"] != list(SEEDS) or manifest["arms"] != list(ARMS):
        raise ValueError("Incomplete or different run design")
    if (manifest["fitted_models"], manifest["policy_rows"], manifest["control_checks"]) != (20, 60, 20):
        raise ValueError("Unexpected run dimensions")
    for p, h in manifest["input_hashes"].items():
        check(p, h)
    for name, h in manifest["outputs"].items():
        p = (run / name).resolve()
        if not p.is_relative_to(run):
            raise ValueError("Unsafe output path in manifest")
        check(p, h)
    expected_scores = {f"{arm}__seed_{seed}__{split}.npz" for arm in ARMS for seed in SEEDS for split in ("validation", "test")}
    if {p.name for p in (run / "scores").glob("*.npz")} != expected_scores:
        raise ValueError("Score file set differs from 2 x 10 x 2 design")
    benchmark = json.loads((root / "generated/benchmark_review/review_manifest.json").read_text())
    train_path = next(Path(p) for p in manifest["input_hashes"] if Path(p).name == "UNSW_NB15_training-set.csv")
    test_path = next(Path(p) for p in manifest["input_hashes"] if Path(p).name == "UNSW_NB15_testing-set.csv")
    train_labels = pd.read_csv(train_path, usecols=["label"]).label.to_numpy()
    test_labels = pd.read_csv(test_path, usecols=["label"]).label.to_numpy()
    audit = pd.read_csv(root / "generated/score_diagnostics/input_audit.csv")
    old_policies = pd.read_csv(root / "generated/score_diagnostics/threshold_policy_by_seed.csv", float_precision="round_trip")
    rows, controls, diagnostics = [], [], []
    for seed in SEEDS:
        seed_dir = Path(benchmark["run_directory"]) / "models" / f"seed_{seed}"
        original_params = json.loads((seed_dir / "rf30_params.json").read_text())
        original_preproc = json.loads((seed_dir / "preprocessing.json").read_text())
        preproc = json.loads((run / f"preprocessing_{seed}.json").read_text())
        if any(original_preproc.get(k) != v for k, v in preproc.items()):
            raise ValueError("Preprocessing metadata changed")
        with np.load(seed_dir / "split_row_ids.npz", allow_pickle=False) as z:
            ids = {s: z[f"{s}_row_id"].copy() for s in ("validation", "test")}
        scores = {}
        for arm in ARMS:
            params = json.loads((run / f"params_{arm}_{seed}.json").read_text())
            expected_params = {**original_params, "class_weight": None if arm == "none" else "balanced_subsample"}
            if params != expected_params:
                raise ValueError("Intervention changed more than class_weight")
            for split in ("validation", "test"):
                with np.load(run / "scores" / f"{arm}__seed_{seed}__{split}.npz", allow_pickle=False) as z:
                    y, score, raw, row_id = [z[k].copy() for k in ("y_true", "y_score", "y_score_float64", "row_id")]
                truth = train_labels[ids[split]] if split == "validation" else test_labels
                if not np.array_equal(row_id, ids[split]) or not np.array_equal(y, truth):
                    raise ValueError("Score row-ID/label mismatch")
                if score.shape != truth.shape or score.dtype != np.float32 or not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any() or not np.array_equal(raw.astype(np.float32), score):
                    raise ValueError("Invalid score/precision preservation")
                scores[arm, split] = (y, score)
                if arm == "balanced_subsample":
                    match = audit[(audit.model_key == "rf30") & (audit.seed == seed) & (audit.split == split)]
                    if len(match) != 1:
                        raise ValueError("Missing original control score")
                    with np.load(match.iloc[0].path, allow_pickle=False) as z:
                        if not np.array_equal(y, z["y_true"]):
                            raise ValueError("Original control labels differ")
                        error = float(np.max(np.abs(score.astype(float)-z["y_score"].astype(float))))
                    if error > manifest["score_atol"]:
                        raise ValueError("Weighted control reproduction failed")
                    controls.append(dict(seed=seed, split=split, max_abs_score_error=error, passed=True))
            yv, sv = scores[arm, "validation"]
            yt, st = scores[arm, "test"]
            selected = thresholds(yv, sv, float(yt.mean()))
            for policy, cutoff in selected.items():
                r = dict(seed=seed, arm=arm, policy=policy, cutoff=cutoff, cutoff_hex=cutoff.hex(), all_negative_cutoff=bool(np.isposinf(cutoff)), validation_prior=float(yv.mean()), test_prior=float(yt.mean()))
                for split in ("validation", "test"):
                    y, s = scores[arm, split]
                    r.update({f"{split}_{k}": v for k, v in metrics(y, s >= cutoff).items()})
                if arm == "balanced_subsample":
                    old = old_policies[(old_policies.model_key == "rf30") & (old_policies.seed == seed) & (old_policies.policy == policy)].iloc[0]
                    if cutoff != old.selected_threshold:
                        raise ValueError("Original control selected threshold differs")
                    for key in ("test_TN", "test_FP", "test_FN", "test_TP"):
                        if r[key] != old[key]:
                            raise ValueError("Original control predictions/counts differ")
                rows.append(r)
            d = dict(seed=seed, arm=arm)
            for label, name in ((0, "normal"), (1, "attack")):
                d[f"{name}_KS_D"] = float(ks_2samp(sv[yv == label], st[yt == label], method="asymp").statistic)
            for split in ("validation", "test"):
                y, s = scores[arm, split]
                d.update({f"{split}_{k}": v for k, v in calibration(y, s).items()})
            diagnostics.append(d)
    policies, control, diag = pd.DataFrame(rows), pd.DataFrame(controls), pd.DataFrame(diagnostics)
    compare_frames(policies, pd.read_csv(run / "policies_by_seed.csv", float_precision="round_trip"), ["seed", "arm", "policy"])
    compare_frames(control, pd.read_csv(run / "control_checks.csv"), ["seed", "split"])
    compare_frames(diag, pd.read_csv(run / "score_diagnostics_by_seed.csv", float_precision="round_trip"), ["seed", "arm"])
    summary_rows = []
    for (arm, policy), values in policies.groupby(["arm", "policy"]):
        r = dict(arm=arm, policy=policy, n_seeds=10, all_negative_cutoffs=int(values.all_negative_cutoff.sum()))
        for key in ("cutoff", "test_F1_weighted_pct", "test_FAR_pct", "test_Attack_recall_pct"):
            r.update({f"{key}_{k}": v for k, v in interval(values[key]).items()})
        summary_rows.append(r)
    summary = pd.DataFrame(summary_rows)
    contrasts = paired(policies)
    compare_frames(summary, pd.read_csv(run / "policy_summary.csv", float_precision="round_trip"), ["arm", "policy"])
    compare_frames(contrasts, pd.read_csv(run / "paired_contrasts.csv", float_precision="round_trip"), ["contrast"])
    diag_summary = diag.drop(columns="seed").groupby("arm").mean().reset_index()
    output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, data in (("policies_by_seed", policies), ("policy_summary", summary), ("paired_contrasts", contrasts), ("control_checks", control), ("score_diagnostics_by_seed", diag), ("score_diagnostics_summary", diag_summary)):
        path = output / f"{name}.csv"
        data.to_csv(path, index=False, float_format="%.17g")
        outputs[path.name] = sha(path)
    record = dict(status="passed", reviewed_utc=datetime.now(timezone.utc).isoformat(), run_id=manifest["run_id"], run_directory=str(run),
                  script=str(Path(__file__).resolve()), script_sha256=sha(__file__), input_hashes=consumed, outputs=outputs,
                  input_and_run_files_verified=len(consumed), score_files_checked=40, policies_recomputed=60,
                  metric_sets_recomputed=120, weighted_control_score_checks=20, max_control_score_error=float(control.max_abs_score_error.max()),
                  paired_contrasts_recomputed=8, only_class_weight_changed=True, new_training_performed=False,
                  independent_implementation="NumPy cutoff enumeration/lexsort, sklearn metrics, scipy KS and paired-t intervals",
                  limitations=["Saved scores/parameters and code are verified; fitted model objects were not archived, so this is not an independent refit.", "Same previously inspected official test set; CIs describe seed variation, not deployment uncertainty."])
    (output / "review_manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k not in ("input_hashes", "outputs")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--revision-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root, run = args.revision_root.resolve(), args.run_dir.resolve()
    review(root, run, root / "generated/class_weight_review")


if __name__ == "__main__":
    main()
