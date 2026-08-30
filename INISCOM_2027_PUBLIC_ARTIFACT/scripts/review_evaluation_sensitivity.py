"""Separate verification: exact feature join, sklearn metrics, saved bootstrap draws.

Reads the completed sensitivity run; never fits models or selects thresholds.
The exact feature merge independently checks the original hash-based mask.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    root = args.revision_root.resolve()
    run = (args.run_dir or root / "generated/evaluation_sensitivity").resolve()
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Incomplete sensitivity run")
    inputs = {str(manifest_path): sha(manifest_path), str(Path(__file__).resolve()): sha(__file__)}
    for filename, expected in manifest["input_hashes"].items():
        if sha(filename) != expected:
            raise ValueError(f"Changed source: {filename}")
        inputs[filename] = expected
    for name, expected in manifest["outputs"].items():
        if sha(run / name) != expected:
            raise ValueError(f"Changed result: {name}")
        inputs[str(run / name)] = expected
    def csv(path):
        return pd.read_csv(path, float_precision="round_trip", low_memory=False)
    train_path = next(p for p in inputs if Path(p).name == "UNSW_NB15_training-set.csv")
    test_path = next(p for p in inputs if Path(p).name == "UNSW_NB15_testing-set.csv")
    train, test = csv(train_path), csv(test_path)
    columns = manifest["feature_columns"]
    left, right = test[columns].copy(), train[columns].copy()
    for frame in (left, right):
        for name in columns:
            if frame[name].dtype == "object":
                frame[name] = frame[name].astype("string").fillna("<NA>").str.strip()
    left["position"] = np.arange(len(left))
    joined = left.merge(right.drop_duplicates(), on=columns, how="left", indicator=True,
                        sort=False, validate="many_to_one").sort_values("position")
    exact_unique = joined["_merge"].eq("left_only").to_numpy()
    mask_frame = csv(run / "test_subset_mask.csv")
    ids = mask_frame.test_row_id.to_numpy(int)
    if not np.array_equal(mask_frame.feature_unique, exact_unique[ids]):
        raise ValueError("Direct exact-feature join disagrees with hash-based mask")
    unique = csv(run / "unique_by_seed.csv")
    audit_path = next(p for p in inputs if Path(p).name == "input_audit.csv")
    audit = csv(audit_path)
    keep = exact_unique[ids]
    checked = 0
    for (model, seed), group in unique.groupby(["model_key", "seed"]):
        match = audit[audit.model_key.eq(model) & audit.seed.eq(seed) & audit.split.eq("test")]
        if len(match) != 1:
            raise ValueError("Missing test score file")
        with np.load(match.iloc[0].path, allow_pickle=False) as data:
            y = data["y_true"]
            if not np.array_equal(data["test_row_id"], ids) or not np.array_equal(y, test.label.to_numpy()[ids]):
                raise ValueError("Separate score/CSV identity check failed")
            for row in group.itertuples():
                pred = data["y_pred"] if row.policy == "native_archived" else data["y_score"].astype(float) >= float.fromhex(row.selected_threshold_hex)
                tn, fp, fn, tp = confusion_matrix(y[keep], pred[keep], labels=[0, 1]).ravel()
                values = {"TN": tn, "FP": fp, "FN": fn, "TP": tp,
                          "F1_weighted_pct": 100 * f1_score(y[keep], pred[keep], average="weighted"),
                          "F1_attack_pct": 100 * f1_score(y[keep], pred[keep]),
                          "Attack_recall_pct": 100 * recall_score(y[keep], pred[keep]),
                          "FAR_pct": 100 * fp / (fp + tn)}
                for name, value in values.items():
                    if not np.isclose(value, getattr(row, name), atol=1e-10, rtol=0):
                        raise ValueError(f"Independent metric mismatch: {model}/{seed}/{row.policy}/{name}")
                checked += 1
    draws = csv(run / "bootstrap_replicates.csv")
    boot = csv(run / "bootstrap_summary.csv")
    for row in boot.itertuples():
        selected = draws[draws.subset.eq(row.subset)]
        if len(selected) != row.replicates or selected.replicate.nunique() != row.replicates:
            raise ValueError("Missing/duplicate bootstrap replicates")
        limits = np.quantile(selected[row.contrast], [.025, .975], method="linear")
        np.testing.assert_allclose(limits, [row.bootstrap_ci95_low_pp, row.bootstrap_ci95_high_pp], atol=1e-12, rtol=0)
    if checked != 320:
        raise ValueError("Expected 320 separately recomputed model-policy-seed rows")
    report = {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "run_directory": str(run), "input_hashes": inputs,
              "exact_feature_merge_matches_hash_mask": True, "unique_rows": int(keep.sum()),
              "sklearn_metric_rows_recomputed": checked, "bootstrap_intervals_recomputed": len(boot),
              "scope": "Verifies results, not IID-row validity or deployment generalization."}
    (run / "review_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "input_hashes"}, indent=2))


if __name__ == "__main__":
    main()
