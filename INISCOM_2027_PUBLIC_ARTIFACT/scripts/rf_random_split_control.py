"""Controlled within-training RF-30 split experiment.

This post-hoc control partitions the official UNSW-NB15 training file into
64% fit, 16% validation, and 20% held-out test rows for each archived seed.
It preserves the reviewed RF-30 parameters, preprocessing, and three global
cutoff policies.  Explicit --run is required; every run is immutable and is
never inserted into the manuscript automatically.
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
from score_shift_analysis import (POLICIES, SEEDS, ks_effect,
                                  prediction_metrics, seed_statistics,
                                  select_policy_thresholds)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def summarize(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    groups = frame.groupby(keys, sort=False) if keys else [((), frame)]
    for values, group in groups:
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(keys, values))
        row["n_seeds"] = int(group.seed.nunique())
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def preflight(revision_root: Path, data_root: Path):
    consumed: dict[str, str] = {}

    def record(path: Path) -> Path:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        consumed[str(path)] = sha(path)
        return path

    train_path = record(data_root / "UNSW_NB15_training-set.csv")
    review_path = record(revision_root / "generated/benchmark_review/review_manifest.json")
    record(revision_root / "vendor/benchmark_helpers/__init__.py")
    record(revision_root / "scripts/score_shift_analysis.py")
    record(Path(__file__))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("status") != "passed" or not review.get("full_benchmark_completed"):
        raise ValueError("A passed full benchmark review is required for RF parameters")
    reviewed_hashes = {str(Path(path).resolve()): value for path, value in review["file_hashes"].items()}
    if str(train_path) not in reviewed_hashes or sha(train_path) != reviewed_hashes[str(train_path)]:
        raise ValueError("Training CSV does not match the reviewed input")

    frame, dropped = read_official_csv(train_path)
    if len(frame) != 175341 or dropped != 0 or set(frame.label.unique()) != {0, 1}:
        raise ValueError("Unexpected official training data")
    params = {}
    for seed in SEEDS:
        path = Path(review["run_directory"]) / "models" / f"seed_{seed}" / "rf30_params.json"
        path = record(path)
        if str(path) not in reviewed_hashes or consumed[str(path)] != reviewed_hashes[str(path)]:
            raise ValueError(f"RF-30 parameters are not the reviewed artifact: seed {seed}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if (value.get("n_estimators"), value.get("class_weight"), value.get("random_state"), value.get("n_jobs")) != (30, "balanced_subsample", seed, 1):
            raise ValueError(f"Unexpected RF-30 parameters: seed {seed}")
        params[seed] = value
    return consumed, frame, params


def split_frame(frame: pd.DataFrame, seed: int):
    development, heldout = train_test_split(
        frame, test_size=.20, random_state=seed, stratify=frame.label, shuffle=True)
    fit, validation = train_test_split(
        development, test_size=.20, random_state=seed + 1,
        stratify=development.label, shuffle=True)
    if (len(fit), len(validation), len(heldout)) != (112217, 28055, 35069):
        raise ValueError("Unexpected 64/16/20 split sizes")
    ids = [set(part._row_id.tolist()) for part in (fit, validation, heldout)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2] or len(set.union(*ids)) != len(frame):
        raise ValueError("Split row IDs overlap or are incomplete")
    return fit, validation, heldout


def run_experiment(revision_root: Path, checked) -> Path:
    consumed, frame, params = checked
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:8]
    out = revision_root / "generated/random_split_control" / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "scores").mkdir()
    manifest = {
        "status": "running", "run_id": run_id, "seeds": list(SEEDS),
        "partition": {"fit": .64, "validation": .16, "heldout_test": .20},
        "split_stage_1_random_state": "seed", "split_stage_2_random_state": "seed + 1",
        "input_hashes": consumed, "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "scope": "post-hoc random-row within-training control; not an independent external test",
        "individual_heldout_labels_used_for_selection": False,
        "aggregate_heldout_prevalence_used_only_by_oracle_policy": True,
    }
    write_json(out / "manifest.json", manifest)
    policy_rows, diagnostic_rows, split_rows = [], [], []
    print(f"RUN_DIRECTORY={out}", flush=True)
    try:
        for index, seed in enumerate(SEEDS, 1):
            fit, validation, heldout = split_frame(frame, seed)
            processor = LegacyPreprocessor()
            with threadpool_limits(limits=1):
                x_fit = processor.fit_transform(fit.reset_index(drop=True))
                x_val = processor.transform(validation.reset_index(drop=True))
                x_test = processor.transform(heldout.reset_index(drop=True))
                model = RandomForestClassifier(**params[seed])
                print(f"[{index}/{len(SEEDS)}] seed={seed}: fitting RF-30", flush=True)
                model.fit(x_fit, fit.label.to_numpy())
                if list(model.classes_) != [0, 1]:
                    raise ValueError("Unexpected probability-column order")
                score_val = model.predict_proba(x_val)[:, 1].astype(np.float32)
                score_test = model.predict_proba(x_test)[:, 1].astype(np.float32)
            y_val, y_test = validation.label.to_numpy(), heldout.label.to_numpy()
            thresholds = select_policy_thresholds(y_val, score_val, float(y_test.mean()))
            np.savez_compressed(out / "scores" / f"rf30__seed_{seed}.npz",
                                validation_y=y_val, validation_score=score_val,
                                validation_row_id=validation._row_id.to_numpy(),
                                test_y=y_test, test_score=score_test,
                                test_row_id=heldout._row_id.to_numpy())
            write_json(out / f"preprocessing_{seed}.json", processor.metadata())
            split_rows.append({
                "seed": seed, "fit_rows": len(fit), "validation_rows": len(validation),
                "test_rows": len(heldout), "fit_attack_prior": float(fit.label.mean()),
                "validation_attack_prior": float(y_val.mean()), "test_attack_prior": float(y_test.mean())})
            for policy, cutoff in thresholds.items():
                result = {"seed": seed, "policy": policy, "cutoff": cutoff,
                          "cutoff_hex": float(cutoff).hex(),
                          "validation_attack_prior": float(y_val.mean()),
                          "test_attack_prior": float(y_test.mean())}
                for name, y, score in (("validation", y_val, score_val), ("test", y_test, score_test)):
                    pred = (score >= cutoff).astype(np.int8)
                    metrics = prediction_metrics(y, pred)
                    independent = 100 * f1_score(y, pred, average="weighted", zero_division=0)
                    if not np.isclose(metrics["F1_weighted_pct"], independent, atol=1e-10, rtol=0):
                        raise ValueError("Independent sklearn F1 check failed")
                    result.update({f"{name}_{key}": value for key, value in metrics.items()})
                policy_rows.append(result)
            diag = {"seed": seed}
            for label, name in ((0, "normal"), (1, "attack")):
                diag[f"{name}_KS_D"] = ks_effect(
                    score_val[y_val == label], score_test[y_test == label])["KS_D"]
            diagnostic_rows.append(diag)

            pd.DataFrame(policy_rows).to_csv(out / "policies_by_seed.csv", index=False, float_format="%.17g")
            pd.DataFrame(diagnostic_rows).to_csv(out / "conditional_shift_by_seed.csv", index=False, float_format="%.17g")
            pd.DataFrame(split_rows).to_csv(out / "split_summary_by_seed.csv", index=False, float_format="%.17g")

        policies = pd.DataFrame(policy_rows)
        diagnostics = pd.DataFrame(diagnostic_rows)
        if len(policies) != 30 or len(diagnostics) != 10 or policies.duplicated(["seed", "policy"]).any():
            raise ValueError("Incomplete controlled experiment")
        summary = summarize(policies, ["policy"], [
            "cutoff", "validation_F1_weighted_pct", "test_F1_weighted_pct",
            "validation_FAR_pct", "test_FAR_pct",
            "validation_Attack_recall_pct", "test_Attack_recall_pct"])
        summary.to_csv(out / "policy_summary.csv", index=False, float_format="%.17g")
        diag_summary = summarize(diagnostics, [], ["normal_KS_D", "attack_KS_D"])
        diag_summary.to_csv(out / "conditional_shift_summary.csv", index=False, float_format="%.17g")

        indexed = policies.set_index(["policy", "seed"]).test_F1_weighted_pct
        default_minus_val = indexed.loc[POLICIES[0]].sort_index() - indexed.loc[POLICIES[1]].sort_index()
        contrast = pd.DataFrame([{"contrast": "default minus global validation", **seed_statistics(default_minus_val)}])
        contrast.to_csv(out / "paired_contrasts.csv", index=False, float_format="%.17g")
        for path, expected in consumed.items():
            if sha(Path(path)) != expected:
                raise ValueError(f"Input changed during run: {path}")
        manifest.update(
            status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
            fitted_models=10, policy_rows=30, diagnostic_rows=10,
            outputs={p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*")
                     if p.is_file() and p.name != "manifest.json"})
        write_json(out / "manifest.json", manifest)
        print(summary.to_string(index=False), flush=True)
        print(diag_summary.to_string(index=False), flush=True)
        print(f"COMPLETE. Send this full folder for review: {out}", flush=True)
        return out
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                        finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / "manifest.json", manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-root", type=Path, default=ROOT)
    parser.add_argument("--project-root", type=Path, default=ROOT.parent)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    checked = preflight(args.revision_root.resolve(), args.project_root.resolve())
    print(f"Preflight passed: {len(checked[0])} hashed inputs; no models fitted yet.", flush=True)
    if args.run:
        run_experiment(args.revision_root.resolve(), checked)


if __name__ == "__main__":
    main()
