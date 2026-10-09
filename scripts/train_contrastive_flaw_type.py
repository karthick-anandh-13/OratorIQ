from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\OratorIQ")

INPUT_FILE = (
    ROOT / "artifacts" / "contrastive_features.pkl"
)

MODEL_DIR = (
    ROOT / "artifacts" / "models"
)

RESULT_DIR = (
    ROOT / "artifacts" / "results"
)

MODEL_FILE = (
    MODEL_DIR / "contrastive_flaw_type_classifier.joblib"
)

REPORT_FILE = (
    RESULT_DIR / "contrastive_flaw_type_report.json"
)

CONFUSION_FILE = (
    RESULT_DIR / "contrastive_flaw_type_confusion.csv"
)


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42


META_COLUMNS = {
    "record_id",
    "pair_id",
    "speaker_id",
    "window_start",
    "window_end",
    "good_window_start",
    "good_window_end",
    "relative_position",
    "flaw_type",
    "severity",
    "temporal_label",
}


# ============================================================
# HEADER
# ============================================================

def header(text):

    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


# ============================================================
# LOAD
# ============================================================

header(
    "ORATORIQ — CONTRASTIVE FLAW TYPE TRAINING"
)

print(
    f"Loading:\n{INPUT_FILE}"
)

df = joblib.load(
    INPUT_FILE
)

print(
    f"\nTotal rows: {len(df):,}"
)


# ============================================================
# KEEP ONLY FLAWED WINDOWS
# ============================================================

df = df[
    df["temporal_label"] == "flawed"
].copy()

print(
    f"Flawed windows: {len(df):,}"
)


print(
    "\nFlaw distribution:"
)

print(
    df["flaw_type"].value_counts()
)


# ============================================================
# FEATURE SELECTION
# ============================================================

feature_columns = [

    column

    for column in df.columns

    if column not in META_COLUMNS

    and pd.api.types.is_numeric_dtype(
        df[column]
    )
]


print(
    f"\nModel features: "
    f"{len(feature_columns)}"
)


if len(feature_columns) != 138:

    raise RuntimeError(
        f"Expected 138 contrastive features, "
        f"found {len(feature_columns)}"
    )


# ============================================================
# SPEAKER SAFE SPLIT
# ============================================================

header(
    "CREATING SPEAKER-SAFE SPLITS"
)

groups = df["speaker_id"].values


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)


train_idx, test_idx = next(
    splitter.split(
        df,
        groups=groups
    )
)


train_all = df.iloc[
    train_idx
].copy()


test_df = df.iloc[
    test_idx
].copy()


splitter2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=RANDOM_STATE
)


train_idx2, val_idx = next(
    splitter2.split(
        train_all,
        groups=train_all["speaker_id"].values
    )
)


train_df = train_all.iloc[
    train_idx2
].copy()


val_df = train_all.iloc[
    val_idx
].copy()


print(
    f"Train:      {len(train_df):,}"
)

print(
    f"Validation: {len(val_df):,}"
)

print(
    f"Test:       {len(test_df):,}"
)

print()

print(
    f"Train speakers: "
    f"{train_df.speaker_id.nunique()}"
)

print(
    f"Validation speakers: "
    f"{val_df.speaker_id.nunique()}"
)

print(
    f"Test speakers: "
    f"{test_df.speaker_id.nunique()}"
)


# ============================================================
# DATA
# ============================================================

X_train = train_df[
    feature_columns
]

y_train = train_df[
    "flaw_type"
]

X_val = val_df[
    feature_columns
]

y_val = val_df[
    "flaw_type"
]

X_test = test_df[
    feature_columns
]

y_test = test_df[
    "flaw_type"
]


# ============================================================
# MODEL
# ============================================================

header(
    "TRAINING HISTGRADIENTBOOSTING"
)

model = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "model",
        HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE
        )
    )
])


print(
    "Training..."
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# VALIDATION
# ============================================================

val_pred = model.predict(
    X_val
)

val_accuracy = accuracy_score(
    y_val,
    val_pred
)

val_f1 = f1_score(
    y_val,
    val_pred,
    average="macro"
)


# ============================================================
# TEST
# ============================================================

test_pred = model.predict(
    X_test
)

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

test_f1 = f1_score(
    y_test,
    test_pred,
    average="macro"
)


# ============================================================
# RESULTS
# ============================================================

header(
    "RESULTS"
)

print(
    f"Validation Accuracy: "
    f"{val_accuracy:.4f}"
)

print(
    f"Validation Macro F1: "
    f"{val_f1:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Test Macro F1: "
    f"{test_f1:.4f}"
)


print(
    "\nClassification report:"
)

print(
    classification_report(
        y_test,
        test_pred
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = sorted(
    y_test.unique()
)

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(
    "\nConfusion matrix:"
)

print(
    cm_df
)


# ============================================================
# SAVE
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


joblib.dump(
    model,
    MODEL_FILE
)


cm_df.to_csv(
    CONFUSION_FILE
)


import json

report = {

    "model":
        "HistGradientBoosting",

    "validation_accuracy":
        float(val_accuracy),

    "validation_macro_f1":
        float(val_f1),

    "test_accuracy":
        float(test_accuracy),

    "test_macro_f1":
        float(test_f1),

    "feature_count":
        len(feature_columns),

    "classes":
        labels,

    "classification_report":
        classification_report(
            y_test,
            test_pred,
            output_dict=True
        )
}


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


print()
print(
    "Model saved:"
)

print(
    MODEL_FILE
)

print(
    "\nReport:"
)

print(
    REPORT_FILE
)

print(
    "\nConfusion matrix:"
)

print(
    CONFUSION_FILE
)

print()
print(
    "✅ Contrastive flaw-type model complete."
)