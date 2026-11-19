"""Redraw the four summary-table figures from reviewed CSVs only.

Camera-ready revision (reviewer R2.7/R2.8): the quality--cost heatmap, the
fixed-recall precision heatmap and the two cost-model residual figures are
redrawn with per-cell contrast-aware text colour, true minus signs and margins
that keep every label inside the canvas.

No training, threshold selection, timing or metric computation happens here.
The plotted numbers are read unchanged from the CSVs that
``build_paper_assets.py`` already uses; the drawing code is the same functions
that ``build_paper_assets.py`` calls, so a later full rebuild produces the same
figures. Usage (from the revision root)::

    python scripts/redraw_figures.py            # writes figures/*.pdf and *.png
    python scripts/redraw_figures.py --check    # also asserts plotted values
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("paper_assets", HERE / "build_paper_assets.py")
assets = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = assets
spec.loader.exec_module(assets)

INPUTS = {
    "policy": "generated/score_diagnostics/threshold_policy_summary.csv",
    "default_cost": "generated/feedback_diagnostics/default_quality_cost.csv",
    "pr": "generated/statistics/pr_checkpoints_summary.csv",
    "residual_sizes": "generated/feedback_diagnostics/residual_by_size.csv",
    "residual_bins": "generated/feedback_revision/drr_merged_bins_by_seed.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE.parent, help="Revision root (contains generated/ and figures/)")
    parser.add_argument("--check", action="store_true", help="Assert a few plotted values against the manuscript tables")
    args = parser.parse_args()
    root = args.root.resolve()
    data = {name: assets.read_csv(root / rel) for name, rel in INPUTS.items()}
    if (data["residual_bins"].groupby(["input_n", "stage3_bin"]).n_calls.sum() < 10).any():
        raise ValueError("Displayed residual bin has fewer than 10 calls")
    if args.check:
        policy = data["policy"]
        rf = assets.unique(policy, model_key="rf30", policy=assets.POLICIES[0]).test_F1_weighted_pct_mean
        ca = assets.unique(policy, model_key="cascade", policy=assets.POLICIES[0]).test_F1_weighted_pct_mean
        if f"{rf:.3f}" != "90.234" or f"{ca:.3f}" != "87.227":
            raise ValueError("Policy summary no longer matches the manuscript; rebuild all assets instead")
    figures = root / "figures"
    assets.plot_quality_cost(data["policy"], data["default_cost"], figures / "quality_cost_policies.pdf")
    assets.plot_pr_heatmap(data["pr"], figures / "pr_checkpoint_heatmap.pdf")
    assets.plot_cost_residual(data["residual_sizes"], figures / "cost_residual_by_size.pdf")
    assets.plot_drr_residual(data["residual_bins"], figures / "drr_residual_by_stage3.pdf")
    record = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "camera-ready redraw of summary figures (R2.7/R2.8); no values recomputed",
        "script_sha256": assets.digest(Path(__file__)),
        "drawing_module_sha256": assets.digest(HERE / "build_paper_assets.py"),
        "inputs": {rel: assets.digest(root / rel) for rel in INPUTS.values()},
        "outputs": {str(p.relative_to(root)).replace("\\", "/"): assets.digest(p) for p in assets.OUTPUTS},
    }
    out = root / "generated" / "figure_redraw_manifest.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Redrew {len(assets.OUTPUTS)} figure files; manifest: {out}")


if __name__ == "__main__":
    main()
