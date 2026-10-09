"""
======================================================================
ORATORIQ — V2 CONTRASTIVE FLAW TYPE TRAINING
======================================================================

Trains ONLY the six-class flaw-type classifier using:

    artifacts\contrastive_features_v2.pkl

Does NOT modify the existing model:

    artifacts\models\contrastive_flaw_type_classifier.joblib

Saves the new model as:

    artifacts\models\contrastive_v2_flaw_type_classifier.joblib

======================================================================
"""

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit


# ======================================================================
# PATHS
# ======================================================================

ROOT = Path(r"D:\OratorIQ")

ARTIFACTS = ROOT / "artifacts"

FEATURE_FILE = (
    ARTIFACTS
    / "contrastive_features_v2.pkl"
)

MODEL_DIR = (
    ARTIFACTS
    / "models"
)

RESULT_DIR = (
    ARTIFACTS
    / "results"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = (
    MODEL_DIR
    / "contrastive_v2_flaw_type_classifier.joblib"
)

REPORT_FILE = (
    RESULT_DIR
    / "contrastive_v2_flaw_type_report.json"
)

CONFUSION_FILE = (
    RESULT_DIR
    / "contrastive_v2_flaw_type_confusion.csv"
)


# ======================================================================
# CONFIGURATION
# ======================================================================

RANDOM_STATE = 42

# Only actual flawed windows are used for flaw-type classification.
# Normal windows do not have a flaw type.
VALID_FLAW_TYPES = [
    "fast_pacing",
    "slow_pacing",
    "high_volume",
    "low_volume",
    "pitch_deviation",
    "long_pause",
]


# ======================================================================
# HELPERS
# ======================================================================

def header(text):
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def progress(current, total, start_time):

    elapsed = time.time() - start_time

    ratio = current / max(total, 1)

    width = 30

    filled = int(
        ratio * width
    )

    bar = (
        "=" * filled
        + "-" * (width - filled)
    )

    if ratio > 0:
        remaining = (
            elapsed / ratio
        ) - elapsed
    else:
        remaining = 0

    print(
        f"\r[{bar}] "
        f"{ratio * 100:6.2f}% "
        f"| {current:,}/{total:,} "
        f"| ETA {int(max(remaining, 0))}s",
        end="",
        flush=True,
    )

    if current == total:
        print()


# ======================================================================
# LOAD DATA
# ======================================================================

header(
    "ORATORIQ — V2 FLAW TYPE TRAINING"
)

if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"V2 feature file not found:\n"
        f"{FEATURE_FILE}"
    )

print(
    f"Loading:\n{FEATURE_FILE}"
)

df = joblib.load(
    FEATURE_FILE
)

print(
    f"Total rows: "
    f"{len(df):,}"
)

print(
    f"Total columns: "
    f"{len(df.columns):,}"
)


# ======================================================================
# FILTER FLAWED WINDOWS
# ======================================================================

df = df[
    df["flaw_type"].isin(
        VALID_FLAW_TYPES
    )
].copy()

df = df.reset_index(
    drop=True
)

print()

print(
    f"Flawed rows available: "
    f"{len(df):,}"
)


# ======================================================================
# CLASS DISTRIBUTION
# ======================================================================

header(
    "CLASS DISTRIBUTION"
)

class_counts = (
    df["flaw_type"]
    .value_counts()
    .reindex(
        VALID_FLAW_TYPES,
        fill_value=0
    )
)

for label, count in class_counts.items():

    print(
        f"{label:<20} {count:>8,}"
    )


# ======================================================================
# IDENTIFY MODEL FEATURES
# ======================================================================

METADATA_COLUMNS = {
    "record_id",
    "pair_id",
    "speaker_id",
    "window_start",
    "window_end",
    "relative_position",
    "flaw_type",
    "severity",
    "temporal_label",
    "good_window_start",
    "good_window_end",
}


feature_columns = []

for column in df.columns:

    if column in METADATA_COLUMNS:
        continue

    if pd.api.types.is_numeric_dtype(
        df[column]
    ):

        feature_columns.append(
            column
        )


print()

print(
    f"Model features: "
    f"{len(feature_columns):,}"
)


# ======================================================================
# CLEAN FEATURES
# ======================================================================

X = (
    df[feature_columns]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0.0)
)

y = df[
    "flaw_type"
].astype(str)

groups = df[
    "speaker_id"
].astype(str)


# ======================================================================
# SPEAKER-SAFE TRAIN / VAL / TEST SPLIT
# ======================================================================

header(
    "CREATING SPEAKER-SAFE SPLIT"
)

# First split:
# 80% train+validation
# 20% test

splitter_1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_val_idx, test_idx = next(
    splitter_1.split(
        X,
        y,
        groups=groups
    )
)


X_train_val = X.iloc[
    train_val_idx
]

y_train_val = y.iloc[
    train_val_idx
]

groups_train_val = groups.iloc[
    train_val_idx
]


X_test = X.iloc[
    test_idx
]

y_test = y.iloc[
    test_idx
]


# Second split:
# 75% train
# 25% validation
# of the 80% train+validation portion.

splitter_2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=RANDOM_STATE
)

train_idx, val_idx = next(
    splitter_2.split(
        X_train_val,
        y_train_val,
        groups=groups_train_val
    )
)


X_train = X_train_val.iloc[
    train_idx
]

y_train = y_train_val.iloc[
    train_idx
]

X_val = X_train_val.iloc[
    val_idx
]

y_val = y_train_val.iloc[
    val_idx
]


# ======================================================================
# SPLIT SUMMARY
# ======================================================================

print(
    f"Train rows:      {len(X_train):,}"
)

print(
    f"Validation rows: {len(X_val):,}"
)

print(
    f"Test rows:       {len(X_test):,}"
)

print()

print(
    f"Train speakers: "
    f"{groups.iloc[train_val_idx[train_idx]].nunique():,}"
)

print(
    f"Validation speakers: "
    f"{groups.iloc[train_val_idx[val_idx]].nunique():,}"
)

print(
    f"Test speakers: "
    f"{groups.iloc[test_idx].nunique():,}"
)


# ======================================================================
# MODEL
# ======================================================================

header(
    "TRAINING HISTGRADIENTBOOSTING"
)

model = HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_iter=300,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=RANDOM_STATE,
)


print(
    f"Training with "
    f"{len(feature_columns):,} features..."
)

start_time = time.time()

model.fit(
    X_train,
    y_train
)

elapsed = time.time() - start_time

print()

print(
    f"Training completed in "
    f"{elapsed:.1f} seconds."
)


# ======================================================================
# VALIDATION
# ======================================================================

header(
    "VALIDATION RESULTS"
)

val_pred = model.predict(
    X_val
)

val_accuracy = accuracy_score(
    y_val,
    val_pred
)

val_macro_f1 = f1_score(
    y_val,
    val_pred,
    average="macro"
)

print(
    f"Validation accuracy: "
    f"{val_accuracy:.4f}"
)

print(
    f"Validation macro F1: "
    f"{val_macro_f1:.4f}"
)

print()

print(
    classification_report(
        y_val,
        val_pred,
        labels=VALID_FLAW_TYPES,
        digits=4,
        zero_division=0
    )
)


# ======================================================================
# TEST
# ======================================================================

header(
    "TEST RESULTS"
)

test_pred = model.predict(
    X_test
)

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

test_macro_f1 = f1_score(
    y_test,
    test_pred,
    labels=VALID_FLAW_TYPES,
    average="macro"
)

print(
    f"Test accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Test macro F1: "
    f"{test_macro_f1:.4f}"
)

print()

test_report = classification_report(
    y_test,
    test_pred,
    labels=VALID_FLAW_TYPES,
    output_dict=True,
    zero_division=0
)

print(
    classification_report(
        y_test,
        test_pred,
        labels=VALID_FLAW_TYPES,
        digits=4,
        zero_division=0
    )
)


# ======================================================================
# CONFUSION MATRIX
# ======================================================================

header(
    "TEST CONFUSION MATRIX"
)

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=VALID_FLAW_TYPES
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"actual_{x}"
        for x in VALID_FLAW_TYPES
    ],
    columns=[
        f"pred_{x}"
        for x in VALID_FLAW_TYPES
    ]
)

print(
    cm_df.to_string()
)


# ======================================================================
# SAVE CONFUSION MATRIX
# ======================================================================

cm_df.to_csv(
    CONFUSION_FILE
)

print()

print(
    f"Confusion matrix saved:\n"
    f"{CONFUSION_FILE}"
)


# ======================================================================
# SAVE MODEL
# ======================================================================

header(
    "SAVING V2 MODEL"
)

joblib.dump(
    {
        "model": model,
        "feature_names": feature_columns,
        "classes": VALID_FLAW_TYPES,
    },
    MODEL_FILE,
    compress=3
)

print(
    f"Model saved:\n"
    f"{MODEL_FILE}"
)


# ======================================================================
# SAVE REPORT
# ======================================================================

report = {
    "model": "HistGradientBoostingClassifier",

    "feature_file": str(
        FEATURE_FILE
    ),

    "model_file": str(
        MODEL_FILE
    ),

    "rows": int(
        len(df)
    ),

    "features": int(
        len(feature_columns)
    ),

    "train_rows": int(
        len(X_train)
    ),

    "validation_rows": int(
        len(X_val)
    ),

    "test_rows": int(
        len(X_test)
    ),

    "validation_accuracy": float(
        val_accuracy
    ),

    "validation_macro_f1": float(
        val_macro_f1
    ),

    "test_accuracy": float(
        test_accuracy
    ),

    "test_macro_f1": float(
        test_macro_f1
    ),

    "classification_report": test_report,

    "classes": VALID_FLAW_TYPES,

    "class_distribution": {
        key: int(value)
        for key, value
        in class_counts.items()
    },

    "feature_names": feature_columns,
}


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2,
        allow_nan=False
    )


# ======================================================================
# FINAL
# ======================================================================

header(
    "V2 FLAW TYPE TRAINING COMPLETE"
)

print(
    f"""
V2 test accuracy:

    {test_accuracy:.4f}

V2 test macro F1:

    {test_macro_f1:.4f}

V2 model:

    {MODEL_FILE}

Report:

    {REPORT_FILE}

Confusion matrix:

    {CONFUSION_FILE}

IMPORTANT:

The original model was NOT modified:

    artifacts\\models\\contrastive_flaw_type_classifier.joblib
"""
)

print(
    "[OK] V2 flaw-type experiment complete."
)