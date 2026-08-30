"""Minimal UNSW helpers adapted from the archived FINAL3 experiment.

Provenance: FINAL3/Teacher_Final_Revision_STAGE2/scripts/
  week1_repeated_seed_experiment.py: clean_label, normalize_objects, split_official
  week2_baseline_experiment.py: normalize_and_encode, make_gate, make_stage_rf,
                               make_xgb

Unlike the archived preprocessing function, this adapter preserves the fitted
scaler and feature vocabulary so a new run can actually be reproduced.  It is
deliberately restricted to the official, no-reduction UNSW protocol.  No
training, downloads, environment changes, or file access occur on import.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DROP_COLUMNS = (
    "id", "attack_cat", "srcip", "dstip", "sport", "dsport", "stime", "ltime",
    "source_file", "_row_id",
)
CATEGORICAL_COLUMNS = ("proto", "state", "service")


def read_official_csv(path):
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    if "label" not in frame:
        raise ValueError(f"Official CSV has no label column: {path}")
    lab = pd.to_numeric(frame["label"], errors="coerce")
    before = len(frame)
    frame = frame.loc[lab.notna()].copy()
    frame["label"] = lab.loc[lab.notna()].astype(int).values
    frame = frame.loc[frame["label"].isin([0, 1])].reset_index(drop=True)
    # This is the archived test_row_id convention, not the CSV's id column.
    frame["_row_id"] = np.arange(len(frame), dtype=np.int64)
    return frame, before - len(frame)


def normalize_objects(frame):
    frame = frame.copy()
    for col in frame.columns:
        # Retain the original dtype rule.  Pandas' newer StringDtype is not
        # silently treated as object; environment differences are audited.
        if frame[col].dtype == "object":
            frame[col] = frame[col].astype(str).str.strip().replace(
                {"": "0", " ": "0", "-": "0", "nan": "0", "None": "0"}
            )
    return frame


class LegacyPreprocessor:
    """Train-only one-hot vocabulary and StandardScaler, serializable by joblib."""

    def __init__(self):
        self.drop_columns = []
        self.categorical_columns = []
        self.feature_columns = []
        self.input_columns = []
        self.scaler = StandardScaler()

    def _encode(self, frame):
        frame = normalize_objects(frame)
        frame = frame.drop(columns=self.drop_columns + ["label"], errors="ignore")
        for col in self.input_columns:
            if col not in frame.columns:
                raise ValueError(f"Missing input feature: {col}")
            if col not in self.categorical_columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0)
        frame = pd.get_dummies(frame, columns=self.categorical_columns)
        return frame.loc[:, ~frame.columns.duplicated()].copy()

    def fit_transform(self, train):
        self.drop_columns = [col for col in DROP_COLUMNS if col in train.columns]
        self.categorical_columns = [col for col in CATEGORICAL_COLUMNS if col in train.columns]
        self.input_columns = [
            col for col in train.columns if col not in self.drop_columns and col != "label"
        ]
        encoded = self._encode(train)
        self.feature_columns = list(encoded.columns)
        matrix = encoded.to_numpy(dtype=np.float32)
        return self.scaler.fit_transform(matrix).astype(np.float32)

    def transform(self, frame):
        if not self.feature_columns:
            raise RuntimeError("Preprocessor has not been fitted")
        encoded = self._encode(frame).reindex(columns=self.feature_columns, fill_value=0)
        return self.scaler.transform(encoded.to_numpy(dtype=np.float32)).astype(np.float32)

    def metadata(self):
        return {
            "drop_columns": self.drop_columns,
            "categorical_columns": self.categorical_columns,
            "input_columns": self.input_columns,
            "feature_columns": self.feature_columns,
            "feature_count": len(self.feature_columns),
            "matrix_dtype": "float32",
            "normalization_rule": "archived object-dtype-only string normalization",
            "reduction": "none",
        }


def build_components(seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    common_rf = dict(
        max_depth=10, min_samples_leaf=10, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=1, random_state=int(seed),
    )
    return {
        "dt": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=50, min_samples_split=4,
            random_state=int(seed),
        ),
        "xgb300": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", device="cpu",
            n_jobs=1, random_state=int(seed), verbosity=0,
        ),
        "rf30": RandomForestClassifier(n_estimators=30, **common_rf),
        "rf300": RandomForestClassifier(n_estimators=300, **common_rf),
    }
