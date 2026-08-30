"""Audit lossless merge of non-default cutoff quality and error columns."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [[cell.strip() for cell in line.rstrip().removesuffix(r"\\").split("&")]
            for line in path.read_text(encoding="utf-8").splitlines()
            if " & " in line and line.rstrip().endswith(r"\\") and not line.startswith("Model &")]


def policies(path):
    result, model = {}, ""
    for cells in rows(path):
        if cells[0]:
            model = cells[0]
        key = model, cells[1]
        if key in result:
            raise ValueError(f"Duplicate table row: {key}")
        result[key] = cells[2:]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--revision-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root, old = args.revision_root.resolve(), args.baseline.resolve()
    relative = Path("generated/tables")
    old_policy = policies(old / relative / "threshold_policies.tex")
    old_errors = policies(old / relative / "threshold_policy_errors.tex")
    new_policy = policies(root / relative / "threshold_policies.tex")
    retained = {k: v for k, v in old_policy.items() if k[1] != "Default 0.5"}
    expected = {k: values + old_errors[k][-2:] for k, values in retained.items()}
    cost = {r[0]: r[1:] for r in rows(root / relative / "default_quality_cost.tex")}
    checks = {
        "baseline_has_24_quality_and_24_error_rows": len(old_policy) == 24 and len(old_errors) == 24,
        "sixteen_nondefault_rows_combined_losslessly": len(new_policy) == 16 and new_policy == expected,
        "eight_defaults_preserved_in_table2": len(cost) == 8,
        "table2_byte_identical": sha(old / relative / "default_quality_cost.tex") == sha(root / relative / "default_quality_cost.tex"),
        "separate_error_table_removed": not (root / relative / "threshold_policy_errors.tex").exists(),
        "archived_quality_rows_unchanged": rows(old / relative / "quality.tex") == rows(root / relative / "quality.tex"),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": str(old), "current_main_sha256": sha(root / "main.tex"),
        "script_sha256": sha(Path(__file__)), "checks": checks,
        "baseline_main_sha256": sha(old / "main.tex"),
        "current_table_sha256": {p: sha(root / relative / p) for p in sorted({"threshold_policies.tex", "quality.tex", "default_quality_cost.tex"})},
        "baseline_table_sha256": {
            p: sha(old / relative / p)
            for p in ("threshold_policies.tex", "threshold_policy_errors.tex", "quality.tex", "default_quality_cost.tex")
        },
        "note": "Audits only the lossless Table 3/Table 9 presentation merge against the archived baseline. Later unrelated revision assets are outside this audit. No score, threshold, metric, timing or CI is recomputed.",
    }
    (root / "generated/table_dedup_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit("Table deduplication audit failed.")


if __name__ == "__main__":
    main()
