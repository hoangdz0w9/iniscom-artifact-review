#!/usr/bin/env python3
"""Audit frozen routing by attack category, without training or selecting policies.

The expanded, threshold-closed archive is authoritative for the two three-stage
routes. Two-stage routes and standalone cutoffs come from the original run.
All test scores/predictions are reconstructed exactly before category summaries
are trusted. ``test_row_id`` indexes the cleaned official CSV, not its ``id``
column. Seed summaries describe repeated fits on ONE fixed test set.

Example (defaults resolve from this script's location):
    python scripts/routing_category_analysis.py --project-root DATA_ROOT
    # --revision-root . selects the generated-output project.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd


COMPONENTS = ("dt", "xgb300", "rf30", "rf300")
ROUTES = ("dt_xgb", "cascade", "dt_rf30", "dt_rf30_rf300")
MODELS = COMPONENTS + ROUTES
THREE_STAGE = ("cascade", "dt_rf30_rf300")
STAGE2_MODEL = {"dt_xgb": "xgb300", "cascade": "xgb300", "dt_rf30": "rf30", "dt_rf30_rf300": "rf30"}
STAGE3_MODEL = {"cascade": "rf30", "dt_rf30_rf300": "rf300"}
MODEL_NAMES = {
    "dt": "DT", "xgb300": "XGB-300", "rf30": "RF-30", "rf300": "RF-300",
    "dt_xgb": "DT->XGB-300", "cascade": "DT->XGB-300->RF-30",
    "dt_rf30": "DT->RF-30", "dt_rf30_rf300": "DT->RF-30->RF-300",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Provenance:
    def __init__(self) -> None:
        self.items: dict[str, dict] = {}

    def add(self, path: Path, role: str) -> Path:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        key = str(path)
        if key not in self.items:
            self.items[key] = {"path": key, "role": role, "bytes": path.stat().st_size, "sha256": sha256(path)}
        return path

    def verify_unchanged(self) -> None:
        for item in self.items.values():
            if sha256(Path(item["path"])) != item["sha256"]:
                raise RuntimeError(f"Input changed during analysis: {item['path']}")


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(json_clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def environment() -> dict:
    versions = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed / not used by this analysis"
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": versions}


def load_clean_test(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Match historical clean_label: numeric, integer cast, binary filter, reset."""
    raw = pd.read_csv(path, low_memory=False)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if not {"label", "attack_cat"} <= set(raw.columns):
        raise ValueError("Test CSV must contain label and attack_cat")
    numeric = pd.to_numeric(raw["label"], errors="coerce")
    clean = raw.loc[numeric.notna()].copy()
    clean["label"] = numeric.loc[numeric.notna()].astype(int).to_numpy()
    clean = clean.loc[clean["label"].isin([0, 1])].reset_index(drop=True)
    labels = clean["label"].to_numpy(np.int8)
    if not len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Cleaned official test must contain both binary classes")
    cat = clean["attack_cat"].fillna("").astype(str).str.strip().replace({"Backdoors": "Backdoor"})
    if ((labels == 1) & cat.str.lower().eq("normal").to_numpy()).any():
        raise ValueError("An Attack label has attack_cat=Normal")
    cat.loc[labels == 0] = "Normal"
    cat = cat.replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    info = {
        "raw_rows": len(raw), "cleaned_rows": len(clean), "invalid_label_rows_dropped": len(raw) - len(clean),
        "label_rule": "numeric label; drop nonnumeric; integer cast; retain 0/1; reset index (historical loader)",
        "row_id_rule": "zero-based position AFTER label cleaning; CSV id column is not test_row_id",
        "normal_rows": int((labels == 0).sum()), "attack_rows": int((labels == 1).sum()),
        "unknown_category_rows": int(cat.eq("Unknown").sum()),
    }
    return labels, cat.to_numpy(str), info


def validate_prediction(item: dict[str, np.ndarray], labels: np.ndarray, source: str) -> None:
    required = {"y_true", "y_pred", "y_score", "test_row_id"}
    if not required <= set(item):
        raise ValueError(f"Missing arrays in {source}: {sorted(required - set(item))}")
    n = len(labels)
    if any(item[k].shape != (n,) for k in required):
        raise ValueError(f"Array shape/official row count mismatch in {source}")
    ids = item["test_row_id"]
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"Noninteger row IDs in {source}")
    if not np.array_equal(np.sort(ids), np.arange(n)):
        raise ValueError(f"Row IDs must form the complete cleaned test set without duplicates: {source}")
    if not np.array_equal(item["y_true"], labels[ids]):
        raise ValueError(f"Labels do not match cleaned CSV row IDs in {source}")
    if not np.isin(item["y_pred"], [0, 1]).all():
        raise ValueError(f"Nonbinary predictions in {source}")
    score = item["y_score"]
    if not np.issubdtype(score.dtype, np.floating) or not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError(f"Invalid probability-like scores in {source}")


def compose_route(score1: np.ndarray, score2: np.ndarray, low1: float,
                  score3: np.ndarray | None = None, low2: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return native decisions, mixed scores, stage-2 reach, stage-3 reach.

    Do not promote float32 before gate comparisons: that changes boundary rows.
    At a 0.5 gate an exact 0.5 exits as Attack, matching assignment order in the
    experiment. An empty stage is an actual zero, not missing evidence.
    """
    if not 0 <= low1 <= .5 or (low2 is not None and not 0 <= low2 <= .5):
        raise ValueError("Routing gate must be in [0, .5]")
    if (score3 is None) != (low2 is None):
        raise ValueError("Third-stage scores and second gate must be supplied together")
    if score1.ndim != 1 or score1.shape != score2.shape or (score3 is not None and score3.shape != score1.shape):
        raise ValueError("Routing score shapes differ")
    normal1, attack1 = score1 <= low1, score1 >= 1 - low1
    reach2 = ~(normal1 | attack1)
    reach3 = np.zeros(score1.size, dtype=bool)
    pred = np.empty(score1.size, dtype=np.int8)
    pred[normal1], pred[attack1] = 0, 1
    mixed = score1.copy()
    idx2 = np.flatnonzero(reach2)
    if score3 is None:
        pred[idx2] = (score2[idx2] >= .5).astype(np.int8)
        mixed[idx2] = score2[idx2]
    else:
        normal2, attack2 = score2[idx2] <= low2, score2[idx2] >= 1 - low2
        pred[idx2[normal2]], pred[idx2[attack2]] = 0, 1
        stop2 = normal2 | attack2
        mixed[idx2[stop2]] = score2[idx2[stop2]]
        idx3 = idx2[~stop2]
        reach3[idx3] = True
        pred[idx3] = (score3[idx3] >= .5).astype(np.int8)
        mixed[idx3] = score3[idx3]
    return pred, mixed, reach2, reach3


@dataclass
class FrozenModel:
    key: str
    seed: int
    arrays: dict[str, np.ndarray]
    reach2: np.ndarray
    reach3: np.ndarray
    cutoff: float | None
    low1: float | None
    low2: float | None


class Archive:
    """Shared strict loader for this analysis and statistics_audit.py."""
    def __init__(self, source_root: Path, test_csv: Path, provenance: Provenance):
        self.source_root, self.provenance = source_root.resolve(), provenance
        self.run = self.source_root / "teacher_final_official_10seeds_FINAL"
        self.gap = self.source_root / "Teacher_Final_Revision_STAGE2" / "final_gap_closure"
        self.labels, self.categories, self.dataset_info = load_clean_test(provenance.add(test_csv, "official test CSV"))
        config = json.loads(provenance.add(self.run / "run_config.json", "primary run configuration").read_text(encoding="utf-8-sig"))
        if config.get("dataset") != "unsw_official":
            raise ValueError("Source run is not the official UNSW regime")
        configured = config["seeds_parsed"] if "seeds_parsed" in config else [int(s) for s in config["seeds"].split(",")]
        self.seeds = sorted(int(s) for s in configured)
        if len(self.seeds) != 10 or len(set(self.seeds)) != 10:
            raise ValueError("This audit requires the ten distinct archived primary seeds")
        self.original_path = provenance.add(self.run / "all_results.csv", "standalone/two-stage selection table")
        self.closed_path = provenance.add(self.gap / "threshold_extended_selected_by_seed.csv", "authoritative expanded three-stage gates")
        self.original = pd.read_csv(self.original_path)
        self.closed = pd.read_csv(self.closed_path)
        self._validate_table(self.original, "model_key", MODELS)
        self._validate_table(self.closed, "architecture", THREE_STAGE)
        self.checks: list[dict] = []

    def _validate_table(self, table: pd.DataFrame, model_column: str, models) -> None:
        if not {model_column, "seed"} <= set(table):
            raise ValueError(f"Missing model/seed columns in {model_column} table")
        pairs = [(str(m), int(s)) for m, s in zip(table[model_column], table["seed"])]
        expected = {(m, s) for m in models for s in self.seeds}
        if len(pairs) != len(set(pairs)) or set(pairs) != expected:
            raise ValueError(f"Incomplete, extra, or duplicate model/seed pairs in {model_column} table")

    def _row(self, table, column, key, seed):
        chosen = table.loc[(table[column] == key) & (table["seed"] == seed)]
        if len(chosen) != 1:
            raise ValueError(f"Expected exactly one threshold row: {key}/{seed}")
        return chosen.iloc[0]

    def _load(self, key: str, seed: int) -> tuple[dict, Path]:
        directory = self.gap / "final_predictions" if key in THREE_STAGE else self.run / "predictions"
        path = self.provenance.add(directory / f"{key}__seed_{seed}.npz", "authoritative test predictions and scores")
        with np.load(path, allow_pickle=False) as archive:
            item = {k: np.asarray(archive[k]) for k in archive.files}
        validate_prediction(item, self.labels, str(path))
        return item, path

    def load_seed(self, seed: int) -> dict[str, FrozenModel]:
        if seed not in self.seeds:
            raise ValueError(f"Unconfigured seed {seed}")
        result = {}
        for key in MODELS:
            item, path = self._load(key, seed)
            if result and not np.array_equal(item["test_row_id"], result["dt"].arrays["test_row_id"]):
                raise RuntimeError(f"Cross-model row order mismatch: {key}/{seed}")
            cutoff = low1 = low2 = None
            threshold_source = self.original_path
            if key in COMPONENTS:
                row = self._row(self.original, "model_key", key, seed)
                cutoff = float(row["selected_threshold"])
                if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
                    raise ValueError(f"Invalid selected cutoff {key}/{seed}")
                pred = (item["y_score"] >= cutoff).astype(np.int8)
                mixed = item["y_score"]
                r2 = r3 = np.zeros(len(pred), dtype=bool)
            else:
                if key in THREE_STAGE:
                    row = self._row(self.closed, "architecture", key, seed)
                    low2 = float(row["threshold_stage2_low"])
                    threshold_source = self.closed_path
                else:
                    row = self._row(self.original, "model_key", key, seed)
                low1 = float(row["threshold_dt_low"])
                s1 = result["dt"].arrays["y_score"]
                s2 = result[STAGE2_MODEL[key]].arrays["y_score"]
                s3 = result[STAGE3_MODEL[key]].arrays["y_score"] if key in THREE_STAGE else None
                pred, mixed, r2, r3 = compose_route(s1, s2, low1, s3, low2)
            prediction_mismatches = int(np.count_nonzero(pred != item["y_pred"]))
            score_mismatches = int(np.count_nonzero(mixed != item["y_score"]))
            max_score_error = float(np.max(np.abs(mixed.astype(float) - item["y_score"].astype(float))))
            if prediction_mismatches or score_mismatches:
                raise RuntimeError(f"Exact reconstruction failed for {key}/{seed}: predictions={prediction_mismatches}, scores={score_mismatches}")
            result[key] = FrozenModel(key, seed, item, r2, r3, cutoff, low1, low2)
            self.checks.append({
                "model_key": key, "seed": seed, "n_rows": len(pred), "test_source": str(path),
                "threshold_source": str(threshold_source), "selected_cutoff": cutoff,
                "threshold_dt_low": low1, "threshold_stage2_low": low2,
                "row_ids_and_csv_labels_exact": True, "prediction_mismatches": prediction_mismatches,
                "score_mismatches": score_mismatches, "max_abs_score_error": max_score_error,
                "stage2_rows": int(r2.sum()), "stage3_rows": int(r3.sum()),
            })
        return result


def pct(n: int | float, denominator: int | float) -> float:
    return float(100 * n / denominator) if denominator else np.nan


def category_row(model: FrozenModel, categories: np.ndarray, category: str) -> dict:
    take = categories == category
    y, pred = model.arrays["y_true"], model.arrays["y_pred"]
    n, errors = int(take.sum()), pred != y
    if not n:
        raise ValueError("Attempted summary of absent category")
    labels = np.unique(y[take])
    if len(labels) != 1:
        raise ValueError(f"Category mixes binary labels: {category}")
    out = {
        "model_key": model.key, "model": MODEL_NAMES[model.key], "seed": model.seed,
        "attack_cat": category, "label": int(labels[0]), "N": n,
        "category_test_share_pct": pct(n, len(y)), "native_errors": int(errors[take].sum()),
        "native_error_pct": pct(errors[take].sum(), n),
        "native_attack_recall_pct": pct(pred[take].sum(), n) if labels[0] else np.nan,
        "native_normal_specificity_pct": pct((pred[take] == 0).sum(), n) if not labels[0] else np.nan,
        "native_FAR_pct": pct(pred[take].sum(), n) if not labels[0] else np.nan,
    }
    for stage, reach in ((2, model.reach2), (3, model.reach3)):
        count, total = int((take & reach).sum()), int(reach.sum())
        out.update({
            f"stage{stage}_available": model.key in (ROUTES if stage == 2 else THREE_STAGE),
            f"stage{stage}_rows": count, f"stage{stage}_total_rows": total,
            f"stage{stage}_reach_pct": pct(count, n),
            f"stage{stage}_composition_pct": pct(count, total),
            f"stage{stage}_enrichment": float((count / total) / (n / len(y))) if total else np.nan,
            f"stage{stage}_native_error_pct": pct(errors[take & reach].sum(), count),
        })
    return out


def descriptive_summary(frame: pd.DataFrame, keys: list[str], metrics: list[str], constants: list[str] = ()) -> pd.DataFrame:
    rows = []
    for values, group in frame.groupby(keys, sort=True, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(keys, values))
        row["n_seeds"] = int(group["seed"].nunique())
        if row["n_seeds"] != len(group):
            raise ValueError(f"Duplicate seed within summary group {row}")
        for name in constants:
            if group[name].nunique(dropna=False) != 1:
                raise ValueError(f"Nonconstant {name} within fixed-test summary group {row}")
            row[name] = group[name].iloc[0]
        for metric in metrics:
            v = group[metric].dropna().astype(float)
            row[f"{metric}_valid_seeds"] = len(v)
            row[f"{metric}_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{metric}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
            row[f"{metric}_min"] = float(v.min()) if len(v) else np.nan
            row[f"{metric}_max"] = float(v.max()) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_error_row(model: FrozenModel, seed_models: dict, categories: np.ndarray, category: str, stage: int) -> dict:
    reach = model.reach2 if stage == 2 else model.reach3
    mask = reach & (categories == category)
    upstream = "dt" if stage == 2 else STAGE2_MODEL[model.key]
    y = model.arrays["y_true"][mask]
    before = seed_models[upstream].arrays["y_score"][mask] >= .5
    after = model.arrays["y_pred"][mask]
    eb, ea = before != y, after != y
    return {
        "model_key": model.key, "seed": model.seed, "attack_cat": category, "stage": stage,
        "upstream_component": upstream, "upstream_diagnostic_cutoff": .5, "N_routed": len(y),
        "upstream_errors": int(eb.sum()), "final_errors": int(ea.sum()),
        "errors_corrected": int((eb & ~ea).sum()), "errors_introduced": int((~eb & ea).sum()),
        "net_errors_corrected": int(eb.sum()) - int(ea.sum()),
        "upstream_error_pct": pct(eb.sum(), len(y)), "final_error_pct": pct(ea.sum(), len(y)),
        "error_reduction_pp": pct(int(eb.sum()) - int(ea.sum()), len(y)),
        "interpretation": "paired fixed-prediction descriptive contrast on the SAME routed rows; not a causal retraining effect",
    }


def output_directory(revision: Path, requested: Path | None, name: str, source: Path | None = None) -> Path:
    out = (requested or revision / "generated" / name).resolve()
    if not out.is_relative_to((revision / "generated").resolve()) or out == (revision / "generated").resolve():
        raise ValueError("Output must be a dedicated subdirectory under revision-root/generated")
    if source is not None and out.is_relative_to(source.resolve()):
        raise ValueError("Output cannot be inside the source archive")
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_analysis_paths(args: argparse.Namespace, name: str) -> tuple[Path, Path, Path, Path, Path]:
    """Data parent and revision output directory are intentionally distinct."""
    project = args.project_root.resolve()
    revision = args.revision_root.resolve()
    source = (args.source_root or project / "FINAL3").resolve()
    test_csv = (args.test_csv or project / "UNSW_NB15_testing-set.csv").resolve()
    if revision == project or revision.is_relative_to(source):
        raise ValueError("revision-root must be a dedicated output project, not the data parent or a source-archive descendant")
    out = output_directory(revision, args.out_dir, name, source)
    return project, revision, source, test_csv, out


def run_analysis(project: Path, revision: Path, source: Path, test_csv: Path, out: Path) -> dict:
    provenance = Provenance()
    provenance.add(Path(__file__), "analysis implementation")
    archive = Archive(source, test_csv, provenance)
    counts = pd.Series(archive.categories).value_counts().rename_axis("attack_cat").rename("N").reset_index()
    counts["test_share_pct"] = counts["N"] * 100 / len(archive.labels)
    counts = counts.sort_values("attack_cat")
    categories = counts["attack_cat"].tolist()
    rows, stages, paired = [], [], []
    for seed in archive.seeds:
        models = archive.load_seed(seed)
        cat = archive.categories[models["dt"].arrays["test_row_id"]]
        for key, model in models.items():
            rows.extend(category_row(model, cat, c) for c in categories)
            stages.append({
                "model_key": key, "seed": seed, "N": len(archive.labels),
                "stage2_rows": int(model.reach2.sum()), "stage3_rows": int(model.reach3.sum()),
                "stage2_reach_pct": pct(model.reach2.sum(), len(cat)), "stage3_reach_pct": pct(model.reach3.sum(), len(cat)),
                "stage1_resolved_rows": int((~model.reach2).sum()),
                "stage2_resolved_rows": int((model.reach2 & ~model.reach3).sum()),
                "stage3_resolved_rows": int(model.reach3.sum()),
            })
            if key in ROUTES:
                for stage in (2, 3) if key in THREE_STAGE else (2,):
                    paired.extend(paired_error_row(model, models, cat, c, stage) for c in categories)
        print(f"Verified and summarized routing: seed {seed}", flush=True)
    frame, stage_frame, paired_frame = pd.DataFrame(rows), pd.DataFrame(stages), pd.DataFrame(paired)
    metrics = ["native_errors", "native_error_pct", "native_attack_recall_pct", "native_normal_specificity_pct", "native_FAR_pct"]
    metrics += [f"stage{s}_{m}" for s in (2, 3) for m in ("rows", "reach_pct", "composition_pct", "enrichment", "native_error_pct")]
    summary = descriptive_summary(frame, ["model_key", "attack_cat"], metrics, ["model", "label", "N", "category_test_share_pct", "stage2_available", "stage3_available"])
    stage_summary = descriptive_summary(stage_frame, ["model_key"], ["stage2_rows", "stage3_rows", "stage2_reach_pct", "stage3_reach_pct"], ["N"])
    paired_summary = descriptive_summary(paired_frame, ["model_key", "attack_cat", "stage"], ["N_routed", "upstream_errors", "final_errors", "errors_corrected", "errors_introduced", "net_errors_corrected", "error_reduction_pp"], ["upstream_component"])
    files = {
        "category_counts.csv": counts, "routing_by_seed.csv": frame, "routing_summary.csv": summary,
        "stage_totals_by_seed.csv": stage_frame, "stage_totals_summary.csv": stage_summary,
        "paired_routed_errors_by_seed.csv": paired_frame, "paired_routed_errors_summary.csv": paired_summary,
        "reconstruction_checks.csv": pd.DataFrame(archive.checks),
    }
    for name, data in files.items():
        data.to_csv(out / name, index=False, encoding="utf-8")
    interesting = summary.loc[summary["model_key"].isin(THREE_STAGE), [
        "model_key", "attack_cat", "N", "stage2_reach_pct_mean", "stage3_reach_pct_mean",
        "stage3_composition_pct_mean", "stage3_enrichment_mean", "stage3_enrichment_valid_seeds", "native_error_pct_mean",
    ]]
    key = {
        "status": "complete", "dataset": archive.dataset_info, "seeds": archive.seeds,
        "models": list(MODELS), "category_counts": counts.to_dict("records"),
        "stage_totals": stage_summary.to_dict("records"), "three_stage_category_summary": interesting.to_dict("records"),
        "exact_reconstruction": {"model_seed_pairs": len(archive.checks), "prediction_mismatches": 0, "score_mismatches": 0},
        "interpretation": "All rates are descriptive on one fixed test set; N is not multiplied by ten. Undefined composition/enrichment for zero-reach seeds is null and valid_seeds is reported. Same-row error contrasts do not prove a causal routing effect.",
    }
    write_json(out / "key_results.json", key)
    provenance.verify_unchanged()
    manifest = {
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(), "analysis": "routing_category_analysis",
        "project_root": str(project), "revision_root": str(revision), "source_root": str(source), "environment": environment(),
        "dataset": archive.dataset_info, "seeds": archive.seeds, "models": list(MODELS),
        "input_files": list(provenance.items.values()),
        "outputs": [{"path": str(out / n), "rows": len(df), "sha256": sha256(out / n)} for n, df in files.items()],
        "key_results_sha256": sha256(out / "key_results.json"),
        "definitions": {
            "reach_pct": "100 * category rows reaching stage / category N",
            "composition_pct": "100 * category rows reaching stage / all rows reaching that stage, within each seed",
            "enrichment": "within-stage category fraction / full-test category fraction; null when no flow reaches stage",
            "summary": "arithmetic mean and sample SD over seeds, NOT pooled independent rows",
            "native_error": "saved validation-selected binary prediction != true binary label",
            "native_attack_recall": "defined only for Attack categories; Normal receives specificity and FAR instead",
            "zero_routing": "count and reach=0; composition/enrichment/conditional-error undefined (blank CSV/null JSON)",
        },
        "limitations": key["interpretation"],
    }
    write_json(out / "MANIFEST.json", manifest)
    return key


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="Data parent containing FINAL3 and the official test CSV, normally DATA_ROOT")
    parser.add_argument("--revision-root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Revision/output project, normally .")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args(argv)


def main() -> int:
    project, revision, source, test_csv, out = resolve_analysis_paths(parse_args(), "routing_categories")
    context = {"analysis": "routing_category_analysis", "project_root": str(project), "revision_root": str(revision), "source_root": str(source)}
    write_json(out / "MANIFEST.json", {"status": "running", **context})
    write_json(out / "key_results.json", {"status": "running", **context})
    try:
        key = run_analysis(project, revision, source, test_csv, out)
    except Exception as exc:
        failure = {"status": "failed", **context, "error": str(exc), "error_type": type(exc).__name__}
        write_json(out / "MANIFEST.json", failure)
        write_json(out / "key_results.json", failure)
        raise
    print(json.dumps({"status": key["status"], "output_dir": str(out), "exact_reconstruction": key["exact_reconstruction"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
