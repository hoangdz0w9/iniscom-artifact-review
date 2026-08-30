"""Matched RF-30 class-weight ablation. Explicit --run is required to fit models.

Read docs/CLASS_WEIGHT_ABLATION_VI.md first. Nothing is downloaded, installed,
overwritten, or inserted into the manuscript automatically. --check-only audits
all required inputs but does not train. Each --run creates an immutable run folder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from benchmark_helpers import LegacyPreprocessor, read_official_csv
from score_shift_analysis import (POLICIES, SEEDS, load_artifact, prediction_metrics,
                                  select_policy_thresholds, seed_statistics, summarize,
                                  calibration_metrics, ks_effect)

ARMS = ("balanced_subsample", "none")
SCORE_ATOL = 1e-7


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def arm_parameters(original, arm):
    if arm not in ARMS or original["class_weight"] != "balanced_subsample":
        raise ValueError("Unexpected arm or original class-weight setting")
    if original["n_estimators"] != 30 or original["n_jobs"] != 1:
        raise ValueError("Expected single-thread RF-30")
    result = dict(original)
    result["class_weight"] = None if arm == "none" else "balanced_subsample"
    return result


def paired_difference(left, right):
    if not left.index.is_unique or not right.index.is_unique or set(left.index) != set(right.index):
        raise ValueError("Paired comparison requires unique, identical seed sets")
    return left.sort_index() - right.sort_index()


def check_control(score, archived, y, row_id):
    if not np.array_equal(y, archived.y):
        raise ValueError("Control label/order mismatch")
    if archived.row_id is not None and not np.array_equal(row_id, archived.row_id):
        raise ValueError("Control row-ID mismatch")
    delta = float(np.max(np.abs(score.astype(float) - archived.score.astype(float))))
    if delta > SCORE_ATOL:
        raise ValueError(f"Control scores failed reproduction: max_abs={delta}; tolerance={SCORE_ATOL}")
    return delta


def preflight(root, data_root):
    """Hash-check every consumed input, including all seeds, before any fit."""
    consumed = {}

    def record(path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Input hash mismatch: {path}")
        consumed[str(path)] = actual
        return path

    review_path = record(root / "generated/benchmark_review/review_manifest.json")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("status") != "passed" or not review.get("full_benchmark_completed"):
        raise ValueError("A completed, passed benchmark review is required")
    expected = {str(Path(p).resolve()): h for p, h in review["file_hashes"].items()}

    def reviewed(path):
        path = Path(path).resolve()
        if str(path) not in expected:
            raise ValueError(f"Input absent from accepted benchmark hashes: {path}")
        return record(path, expected[str(path)])

    score_manifest_path = record(root / "generated/score_diagnostics/manifest.json")
    score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    if score_manifest.get("status") != "complete":
        raise ValueError("Score diagnostic archive is incomplete")
    for item in score_manifest["outputs"]:
        record(root / "generated/score_diagnostics" / Path(item.get("file") or item["path"]).name, item["sha256"])
    record(root / "scripts/score_shift_analysis.py", score_manifest["script_sha256"])
    record(root / "vendor/benchmark_helpers/__init__.py")
    record(Path(__file__))
    record(root / "docs/CLASS_WEIGHT_ABLATION_VI.md")
    audit = pd.read_csv(root / "generated/score_diagnostics/input_audit.csv")
    archive = {}
    for seed in SEEDS:
        for split in ("validation", "test"):
            rows = audit[(audit.model_key == "rf30") & (audit.seed == seed) & (audit.split == split)]
            if len(rows) != 1:
                raise ValueError(f"Missing/duplicate archived RF-30: {seed}/{split}")
            r = rows.iloc[0]
            archive[seed, split] = record(r.path, r.sha256)
    originals = pd.read_csv(root / "generated/score_diagnostics/threshold_policy_by_seed.csv", float_precision="round_trip")
    originals = originals[originals.model_key == "rf30"]
    if len(originals) != 30 or originals.duplicated(["seed", "policy"]).any():
        raise ValueError("Expected exactly 10 x 3 original RF policies")
    train_path = reviewed(data_root / "UNSW_NB15_training-set.csv")
    test_path = reviewed(data_root / "UNSW_NB15_testing-set.csv")
    train, dropped_train = read_official_csv(train_path)
    test, dropped_test = read_official_csv(test_path)
    if (len(train), len(test), dropped_train, dropped_test) != (175341, 82332, 0, 0):
        raise ValueError("Official input dimensions changed")
    params, splits, metadata = {}, {}, {}
    for seed in SEEDS:
        directory = Path(review["run_directory"]) / "models" / f"seed_{seed}"
        params[seed] = json.loads(reviewed(directory / "rf30_params.json").read_text())
        arm_parameters(params[seed], "none")
        if params[seed]["random_state"] != seed:
            raise ValueError("Wrong archived random_state")
        with np.load(reviewed(directory / "split_row_ids.npz"), allow_pickle=False) as saved:
            splits[seed] = {k: saved[k].copy() for k in saved.files}
        metadata[seed] = json.loads(reviewed(directory / "preprocessing.json").read_text())
        tr, val = train_test_split(train, test_size=.2, random_state=seed, stratify=train.label, shuffle=True)
        for split, frame, key in (("train", tr, "train_row_id"), ("validation", val, "validation_row_id"), ("test", test, "test_row_id")):
            if not np.array_equal(frame._row_id.to_numpy(), splits[seed][key]):
                raise ValueError(f"Split row IDs changed: {seed}/{split}")
            if split != "train":
                saved = load_artifact(archive[seed, split], split)
                check_control(saved.score, saved, frame.label.to_numpy(), frame._row_id.to_numpy())
    return consumed, train, test, params, archive, originals, metadata


def analyze_pairs(rows):
    frame = pd.DataFrame(rows)
    values = frame.set_index(["arm", "policy", "seed"]).test_F1_weighted_pct
    results = []
    for policy in POLICIES:
        delta = paired_difference(values.loc["none", policy], values.loc["balanced_subsample", policy])
        results.append({"contrast": f"none minus balanced_subsample: {policy}", **seed_statistics(delta)})
    gaps = {}
    for arm in ARMS:
        gaps[arm] = paired_difference(values.loc[arm, POLICIES[0]], values.loc[arm, POLICIES[1]])
        results.append({"contrast": f"default minus global validation: {arm}", **seed_statistics(gaps[arm])})
    results.append({"contrast": "change in default-minus-validation gap: none minus balanced_subsample",
                    **seed_statistics(paired_difference(gaps["none"], gaps["balanced_subsample"]))})
    for policy in POLICIES[1:]:
        thresholds = frame[frame.policy == policy].set_index(["arm", "seed"]).cutoff
        delta = paired_difference(thresholds.loc["none"], thresholds.loc["balanced_subsample"])
        # Never silently discard an all-negative (infinite) selected cutoff.
        if np.isfinite(delta).all():
            results.append({"contrast": f"cutoff change (score units), none minus balanced_subsample: {policy}", **seed_statistics(delta)})
    return pd.DataFrame(results)


def run_experiment(root, checked):
    consumed, train, test, params, archive, originals, metadata = checked
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:8]
    out = root / "generated/class_weight_ablation" / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "scores").mkdir()
    manifest = {"status": "running", "run_id": run_id, "seeds": list(SEEDS), "arms": list(ARMS),
                "started_utc": datetime.now(timezone.utc).isoformat(), "input_hashes": consumed,
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__, "score_atol": SCORE_ATOL,
                "policy_score_dtype": "float32, matching archived global-cutoff diagnostics",
                "raw_scores_also_saved": "float64", "new_timing_claims": False,
                "individual_test_labels_or_test_scores_used_to_select_thresholds": False,
                "oracle_aggregate_test_prevalence_used": True,
                "interpretation": "post-hoc matched intervention on one previously inspected test split; not external confirmation"}
    write_json(out / "manifest.json", manifest)
    rows, controls, shifts = [], [], []
    print(f"RUN_DIRECTORY={out}", flush=True)
    try:
        for index, seed in enumerate(SEEDS, 1):
            tr, val = train_test_split(train, test_size=.2, random_state=seed, stratify=train.label, shuffle=True)
            processor = LegacyPreprocessor()
            with threadpool_limits(limits=1):
                Xtr = processor.fit_transform(tr.reset_index(drop=True))
                Xv, Xt = processor.transform(val.reset_index(drop=True)), processor.transform(test)
                if processor.metadata()["feature_columns"] != metadata[seed]["feature_columns"]:
                    raise ValueError(f"Preprocessing vocabulary/order changed: seed {seed}")
                yv, yt = val.label.to_numpy(), test.label.to_numpy()
                for arm in ARMS:
                    print(f"[{index}/10] seed={seed} class_weight={arm}: fitting RF-30", flush=True)
                    model = RandomForestClassifier(**arm_parameters(params[seed], arm))
                    model.fit(Xtr, tr.label.to_numpy())
                    if list(model.classes_) != [0, 1]:
                        raise ValueError("Unexpected probability-column order")
                    raw_v, raw_t = model.predict_proba(Xv)[:, 1], model.predict_proba(Xt)[:, 1]
                    sv, st = raw_v.astype(np.float32), raw_t.astype(np.float32)
                    thresholds = select_policy_thresholds(yv, sv, float(yt.mean()))
                    for split, scores, y, ids, raw in (("validation", sv, yv, val._row_id.to_numpy(), raw_v), ("test", st, yt, test._row_id.to_numpy(), raw_t)):
                        if arm == "balanced_subsample":
                            old = load_artifact(archive[seed, split], split)
                            delta = check_control(scores, old, y, ids)
                            controls.append({"seed": seed, "split": split, "max_abs_score_error": delta, "passed": True})
                        np.savez_compressed(out / "scores" / f"{arm}__seed_{seed}__{split}.npz",
                                            y_true=y, y_score=scores, y_score_float64=raw, row_id=ids)
                    write_json(out / f"params_{arm}_{seed}.json", model.get_params())
                    write_json(out / f"preprocessing_{seed}.json", processor.metadata())
                    for policy, cutoff in thresholds.items():
                        result = {"seed": seed, "arm": arm, "policy": policy, "cutoff": cutoff,
                                  "cutoff_hex": float(cutoff).hex(), "all_negative_cutoff": bool(np.isposinf(cutoff)),
                                  "validation_prior": float(yv.mean()), "test_prior": float(yt.mean())}
                        for split, y, score in (("validation", yv, sv), ("test", yt, st)):
                            pred = (score >= cutoff).astype(np.int8)
                            metrics = prediction_metrics(y, pred)
                            if not np.isclose(metrics["F1_weighted_pct"], 100*f1_score(y, pred, average="weighted", zero_division=0), atol=1e-10, rtol=0):
                                raise ValueError("Independent sklearn F1 check failed")
                            result.update({f"{split}_{k}": v for k, v in metrics.items()})
                            if arm == "balanced_subsample":
                                original = originals[(originals.seed == seed) & (originals.policy == policy)]
                                if len(original) != 1 or cutoff != float(original.iloc[0].selected_threshold):
                                    raise ValueError(f"Control selected cutoff changed: {seed}/{policy}")
                                old = load_artifact(archive[seed, split], split)
                                if not np.array_equal(pred, old.score >= cutoff):
                                    raise ValueError("Control policy decisions changed")
                                for metric in ("TN", "FP", "FN", "TP", "F1_weighted_pct"):
                                    if not np.isclose(result[f"{split}_{metric}"], original.iloc[0][f"{split}_{metric}"], atol=1e-9, rtol=0):
                                        raise ValueError(f"Control metric changed: {metric}")
                        rows.append(result)
                    diag = {"seed": seed, "arm": arm}
                    for label, name in ((0, "normal"), (1, "attack")):
                        diag[f"{name}_KS_D"] = ks_effect(sv[yv == label], st[yt == label])["KS_D"]
                    for split, y, score in (("validation", yv, sv), ("test", yt, st)):
                        metrics, _ = calibration_metrics(y, score)
                        diag.update({f"{split}_{k}": v for k, v in metrics.items()})
                    shifts.append(diag)
            pd.DataFrame(rows).to_csv(out / "policies_by_seed.csv", index=False, float_format="%.17g")
            pd.DataFrame(controls).to_csv(out / "control_checks.csv", index=False, float_format="%.17g")
            pd.DataFrame(shifts).to_csv(out / "score_diagnostics_by_seed.csv", index=False, float_format="%.17g")
        if len(rows) != 60 or len(controls) != 20 or len(shifts) != 20:
            raise ValueError("Incomplete ablation; refusing complete status")
        frame = pd.DataFrame(rows)
        summary = summarize(frame, ["arm", "policy"], ["cutoff", "test_F1_weighted_pct", "test_FAR_pct", "test_Attack_recall_pct"])
        summary["all_negative_cutoffs"] = [int(frame[(frame.arm == r.arm) & (frame.policy == r.policy)].all_negative_cutoff.sum()) for r in summary.itertuples()]
        summary.to_csv(out / "policy_summary.csv", index=False, float_format="%.17g")
        analyze_pairs(rows).to_csv(out / "paired_contrasts.csv", index=False, float_format="%.17g")
        # Reject input/code changes that occurred during a run.
        for path, expected in consumed.items():
            if sha(path) != expected:
                raise ValueError(f"Input changed during run: {path}")
        manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                        fitted_models=20, policy_rows=60, control_checks=20,
                        outputs={p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"})
        write_json(out / "manifest.json", manifest)
        print(summary.to_string(index=False), flush=True)
        print(f"COMPLETE. Send this full folder for review: {out}", flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-root", type=Path, default=ROOT)
    parser.add_argument("--project-root", type=Path, default=ROOT.parent)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    checked = preflight(args.revision_root.resolve(), args.project_root.resolve())
    print(f"Preflight passed: {len(checked[0])} hashed inputs, 10 matched splits; no models fitted yet.", flush=True)
    if args.run:
        run_experiment(args.revision_root.resolve(), checked)


if __name__ == "__main__":
    main()
