"""
===============================================================
ORATORIQ — CONTRASTIVE SEVERITY TRAINING
===============================================================

Trains severity classification using the already-created
contrastive feature dataset.

Classes:

    slight
    medium
    bad
    extreme

Uses speaker-safe splitting.

No audio processing.
No feature extraction.

===============================================================
"""

from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")


# ===============================================================
# PATHS
# ===============================================================

ROOT = Path(r"D:\OratorIQ")

ARTIFACTS = ROOT / "artifacts"

INPUT_FILE = (
    ARTIFACTS / "contrastive_features.pkl"
)

MODEL_DIR = (
    ARTIFACTS / "models"
)

RESULT_DIR = (
    ARTIFACTS / "results"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ===============================================================
# CONFIG
# ===============================================================

RANDOM_STATE = 42


# ===============================================================
# PRINT
# ===============================================================

def header(text):

    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


# ===============================================================
# LOAD DATA
# ===============================================================

header(
    "ORATORIQ — CONTRASTIVE SEVERITY TRAINING"
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


# ===============================================================
# ONLY FLAWED WINDOWS
# ===============================================================

df = df[
    df["temporal_label"] == "flawed"
].copy()


print(
    f"Flawed windows: {len(df):,}"
)


# ===============================================================
# CHECK SEVERITY
# ===============================================================

print(
    "\nSeverity distribution:"
)

print(
    df["severity"].value_counts()
)


# ===============================================================
# FEATURE SELECTION
# ===============================================================

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


feature_columns = [

    c

    for c in df.columns

    if c not in META_COLUMNS

    and pd.api.types.is_numeric_dtype(
        df[c]
    )
]


print(
    f"\nModel features: "
    f"{len(feature_columns)}"
)


# ===============================================================
# SPEAKER-SAFE SPLIT
# ===============================================================

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


train_groups = (
    train_all["speaker_id"].values
)


train_idx2, val_idx = next(
    splitter2.split(
        train_all,
        groups=train_groups
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


print(
    f"\nTrain speakers: "
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


# ===============================================================
# MODELS
# ===============================================================

models = {

    "ExtraTrees": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",
            ExtraTreesClassifier(

                n_estimators=500,

                max_features="sqrt",

                min_samples_leaf=2,

                class_weight="balanced",

                n_jobs=-1,

                random_state=RANDOM_STATE

            )
        )

    ]),


    "RandomForest": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",
            RandomForestClassifier(

                n_estimators=400,

                max_features="sqrt",

                min_samples_leaf=2,

                class_weight="balanced",

                n_jobs=-1,

                random_state=RANDOM_STATE

            )
        )

    ]),


    "HistGradientBoosting": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",
            HistGradientBoostingClassifier(

                max_iter=400,

                learning_rate=0.05,

                max_leaf_nodes=31,

                l2_regularization=1.0,

                random_state=RANDOM_STATE

            )
        )

    ])

}


# ===============================================================
# DATA
# ===============================================================

X_train = train_df[
    feature_columns
]

X_val = val_df[
    feature_columns
]

X_test = test_df[
    feature_columns
]


y_train = train_df[
    "severity"
]

y_val = val_df[
    "severity"
]

y_test = test_df[
    "severity"
]


# ===============================================================
# TRAIN
# ===============================================================

header(
    "TRAINING SEVERITY MODELS"
)


results = {}


for name, model in models.items():

    print()
    print(
        "-" * 60
    )

    print(
        f"Training {name}..."
    )

    start = time.time()


    model.fit(
        X_train,
        y_train
    )


    val_pred = model.predict(
        X_val
    )

    test_pred = model.predict(
        X_test
    )


    val_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0
    )


    test_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )


    val_accuracy = accuracy_score(
        y_val,
        val_pred
    )


    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )


    print(
        f"Training time: "
        f"{int(time.time() - start)}s"
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


    results[name] = {

        "model": model,

        "val_f1": val_f1,

        "test_f1": test_f1,

        "val_accuracy":
            val_accuracy,

        "test_accuracy":
            test_accuracy,

        "test_pred":
            test_pred

    }


# ===============================================================
# SELECT BEST
# ===============================================================

best_name = max(

    results,

    key=lambda name:
        results[name]["val_f1"]

)


best = results[
    best_name
]


header(
    "BEST SEVERITY MODEL"
)


print(
    f"Model: {best_name}"
)

print(
    f"Validation Macro F1: "
    f"{best['val_f1']:.4f}"
)

print(
    f"Test Macro F1: "
    f"{best['test_f1']:.4f}"
)

print(
    f"Test Accuracy: "
    f"{best['test_accuracy']:.4f}"
)


# ===============================================================
# CLASSIFICATION REPORT
# ===============================================================

print(
    "\nClassification report:"
)


report = classification_report(

    y_test,

    best["test_pred"],

    zero_division=0

)


print(
    report
)


# ===============================================================
# CONFUSION MATRIX
# ===============================================================

labels = [
    "slight",
    "medium",
    "bad",
    "extreme"
]


cm = confusion_matrix(

    y_test,

    best["test_pred"],

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


# ===============================================================
# SAVE MODEL
# ===============================================================

model_path = (
    MODEL_DIR
    / "contrastive_severity_classifier.joblib"
)


joblib.dump(
    best["model"],
    model_path
)


print(
    f"\nModel saved:\n{model_path}"
)


# ===============================================================
# SAVE REPORT
# ===============================================================

report_dict = classification_report(

    y_test,

    best["test_pred"],

    output_dict=True,

    zero_division=0

)


report_path = (
    RESULT_DIR
    / "contrastive_severity_report.json"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report_dict,
        f,
        indent=2
    )


cm_path = (
    RESULT_DIR
    / "contrastive_severity_confusion.csv"
)


cm_df.to_csv(
    cm_path
)


# ===============================================================
# SAVE SUMMARY
# ===============================================================

summary = {

    "best_model":
        best_name,

    "validation_f1":
        best["val_f1"],

    "test_f1":
        best["test_f1"],

    "test_accuracy":
        best["test_accuracy"],

    "classes":
        labels,

    "model_features":
        len(feature_columns)

}


summary_path = (
    RESULT_DIR
    / "contrastive_severity_summary.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ===============================================================
# COMPLETE
# ===============================================================

header(
    "SEVERITY TRAINING COMPLETE"
)


print(
    f"""
Best model:
    {best_name}

Validation F1:
    {best["val_f1"]:.4f}

Test F1:
    {best["test_f1"]:.4f}

Test Accuracy:
    {best["test_accuracy"]:.4f}

Model:
    {model_path}

Report:
    {report_path}

Confusion matrix:
    {cm_path}
"""
)


print(
    "✅ Severity model complete."
)