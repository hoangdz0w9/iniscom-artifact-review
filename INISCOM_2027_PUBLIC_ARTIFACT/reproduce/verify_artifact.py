"""Verify the extracted anonymous artifact and selected reported facts."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PUBLIC_ARTIFACT_HASHES.json"
FORBIDDEN = re.compile(
    b"C:" + br"[\\/]" + b"Users"
    + b"|D:" + br"[\\/]" + b"NCKH"
    + b"|Dell" + b" Precision"
    + br"|\.co" + b"dex"
    + b"|Chat" + b"GPT"
    + b"|Open" + b"AI",
    re.IGNORECASE,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, tolerance=5e-10):
    if not np.isclose(float(actual), float(expected), atol=tolerance, rtol=0):
        raise ValueError(f"Numeric audit failed: {actual} != {expected}")


def main():
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual_files = {
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if p.is_file()
        and p != MANIFEST
        and "__pycache__" not in p.parts
        and ".pytest_cache" not in p.parts
        and p.suffix.lower() not in {".pyc", ".pyo"}
    }
    if actual_files != set(expected):
        raise ValueError("Artifact members differ from the hash manifest")
    for name, digest in expected.items():
        path = ROOT / name
        if sha(path) != digest:
            raise ValueError(f"Hash mismatch: {name}")
        if path.suffix.lower() in {".py", ".md", ".txt", ".json", ".csv", ".tex"}:
            if FORBIDDEN.search(path.read_bytes()):
                raise ValueError(f"Anonymous-release marker found: {name}")

    threshold = pd.read_csv(ROOT / "results/score_diagnostics/threshold_policy_summary.csv")
    if len(threshold) != 24 or threshold[["model_key", "policy"]].duplicated().any():
        raise ValueError("Expected 8 models x 3 global policies")

    random_policy = pd.read_csv(ROOT / "results/random_split/policy_summary.csv")
    random_shift = pd.read_csv(ROOT / "results/random_split/conditional_shift_summary.csv").iloc[0]
    random_pair = pd.read_csv(ROOT / "results/random_split/paired_contrasts.csv").iloc[0]
    if len(random_policy) != 3 or not random_policy.n_seeds.eq(10).all():
        raise ValueError("Incomplete ten-seed random-split summary")
    selected = random_policy[random_policy.policy.eq("global_validation_f1")].iloc[0]
    close(selected.validation_FAR_pct_mean, 9.637276785714285)
    close(selected.test_FAR_pct_mean, 9.651785714285715)
    close(selected.validation_F1_weighted_pct_mean, 94.46944549379506)
    close(selected.test_F1_weighted_pct_mean, 94.46503407333981)
    close(random_shift.normal_KS_D_mean, 0.010953125)
    close(random_shift.attack_KS_D_mean, 0.007998601645485503)
    close(random_pair["mean"], -0.23117520152818685)

    class_weight = pd.read_csv(ROOT / "results/class_weight/policy_summary.csv")
    bootstrap = pd.read_csv(ROOT / "results/evaluation_sensitivity/bootstrap_summary.csv")
    if len(class_weight) != 6 or len(bootstrap) != 8:
        raise ValueError("Incomplete class-weight or bootstrap summary")

    table = (ROOT / "paper_assets/tables/threshold_policies.tex").read_text(encoding="utf-8")
    if "tab:thresholdfull" in table or "Test F1-W & FAR & Recall" not in table:
        raise ValueError("Merged cutoff/F1/FAR/recall table is missing")

    report = {
        "status": "passed",
        "members_verified": len(expected),
        "threshold_policy_rows": len(threshold),
        "random_split_policy_rows": len(random_policy),
        "class_weight_rows": len(class_weight),
        "bootstrap_rows": len(bootstrap),
        "anonymous_marker_scan": "passed",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
