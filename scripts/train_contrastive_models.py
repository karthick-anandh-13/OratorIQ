"""
===============================================================
ORATORIQ — CONTRASTIVE MODEL TRAINING
===============================================================

Uses:

    artifacts/contrastive_features.pkl

Trains:

1. Temporal flaw detector
2. Flaw-type classifier
3. Severity classifier

The model learns from:

    flawed features
    +
    good-vs-flawed feature deltas
    +
    relative changes

Speaker-safe splitting is used.

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
    precision_score,
    recall_score,
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
# PRINT HELPERS
# ===============================================================

def header(text):

    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def elapsed(seconds):

    seconds = int(seconds)

    minutes = seconds // 60

    seconds = seconds % 60

    if minutes:
        return f"{minutes}m {seconds}s"

    return f"{seconds}s"


# ===============================================================
# LOAD DATA
# ===============================================================

header(
    "ORATORIQ — CONTRASTIVE MODEL TRAINING"
)

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Contrastive dataset not found:\n"
        f"{INPUT_FILE}"
    )


print(
    f"Loading:\n{INPUT_FILE}"
)

df = joblib.load(
    INPUT_FILE
)

print(
    f"\nRows: "
    f"{len(df):,}"
)

print(
    f"Columns: "
    f"{len(df.columns):,}"
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


# ===============================================================
# CHECK DATA
# ===============================================================

print(
    "\nTemporal labels:"
)

print(
    df["temporal_label"]
    .value_counts()
)


print(
    "\nFlaw types:"
)

print(
    df["flaw_type"]
    .value_counts()
)


print(
    "\nSeverity:"
)

print(
    df["severity"]
    .value_counts()
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
# MODEL FACTORIES
# ===============================================================

def make_models():

    models = {}

    models["ExtraTrees"] = Pipeline([

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

                random_state=RANDOM_STATE,

            )
        )
    ])


    models["RandomForest"] = Pipeline([

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

                random_state=RANDOM_STATE,

            )
        )
    ])


    models["HistGradientBoosting"] = Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",
            HistGradientBoostingClassifier(

                max_iter=350,

                learning_rate=0.05,

                max_leaf_nodes=31,

                l2_regularization=1.0,

                random_state=RANDOM_STATE,

            )
        )
    ])


    # -----------------------------------------------------------
    # XGBoost
    # -----------------------------------------------------------

    try:

        from xgboost import XGBClassifier

        models["XGBoost"] = Pipeline([

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "model",
                XGBClassifier(

                    n_estimators=500,

                    max_depth=7,

                    learning_rate=0.04,

                    subsample=0.85,

                    colsample_bytree=0.85,

                    objective="binary:logistic",

                    eval_metric="logloss",

                    tree_method="hist",

                    n_jobs=-1,

                    random_state=RANDOM_STATE,

                )
            )
        ])

    except Exception:

        print(
            "\nXGBoost unavailable."
        )


    return models


# ===============================================================
# GENERIC MODEL EVALUATION
# ===============================================================

def train_models(
    train_df,
    val_df,
    test_df,
    target_column,
    task_name,
    models,
):

    header(
        f"TRAINING — {task_name}"
    )


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
        target_column
    ]

    y_val = val_df[
        target_column
    ]

    y_test = test_df[
        target_column
    ]


    results = {}


    for name, model in models.items():

        print(
            f"\n{'-' * 60}"
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


        val_acc = accuracy_score(
            y_val,
            val_pred
        )


        test_acc = accuracy_score(
            y_test,
            test_pred
        )


        print(
            f"Training time: "
            f"{elapsed(time.time() - start)}"
        )


        print(
            f"Validation Accuracy: "
            f"{val_acc:.4f}"
        )

        print(
            f"Validation Macro F1: "
            f"{val_f1:.4f}"
        )

        print(
            f"Test Accuracy: "
            f"{test_acc:.4f}"
        )

        print(
            f"Test Macro F1: "
            f"{test_f1:.4f}"
        )


        results[name] = {

            "model":
                model,

            "val_f1":
                val_f1,

            "test_f1":
                test_f1,

            "val_accuracy":
                val_acc,

            "test_accuracy":
                test_acc,

            "test_predictions":
                test_pred,

        }


    # -----------------------------------------------------------
    # Select best validation model
    # -----------------------------------------------------------

    best_name = max(

        results,

        key=lambda name:
            results[name]["val_f1"]

    )


    best = results[
        best_name
    ]


    print(
        "\n🏆 BEST MODEL:"
    )

    print(
        best_name
    )

    print(
        f"Validation Macro F1: "
        f"{best['val_f1']:.4f}"
    )

    print(
        f"Test Macro F1: "
        f"{best['test_f1']:.4f}"
    )


    # -----------------------------------------------------------
    # Save model
    # -----------------------------------------------------------

    safe_name = (
        task_name
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )


    model_path = (
        MODEL_DIR
        / f"contrastive_{safe_name}.joblib"
    )


    joblib.dump(
        best["model"],
        model_path
    )


    print(
        f"\nSaved:\n{model_path}"
    )


    # -----------------------------------------------------------
    # Save report
    # -----------------------------------------------------------

    report = classification_report(

        y_test,

        best["test_predictions"],

        output_dict=True,

        zero_division=0

    )


    report_path = (
        RESULT_DIR
        / f"contrastive_{safe_name}_report.json"
    )


    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )


    # -----------------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------------

    labels = sorted(
        y_test.unique()
    )


    cm = confusion_matrix(

        y_test,

        best["test_predictions"],

        labels=labels

    )


    cm_df = pd.DataFrame(

        cm,

        index=labels,

        columns=labels

    )


    cm_path = (
        RESULT_DIR
        / f"contrastive_{safe_name}_confusion.csv"
    )


    cm_df.to_csv(
        cm_path
    )


    print(
        f"Report:\n{report_path}"
    )

    print(
        f"Confusion matrix:\n{cm_path}"
    )


    return best["model"], best


# ===============================================================
# PREPARE TEMPORAL DATA
# ===============================================================

header(
    "PREPARING TEMPORAL DETECTION DATA"
)


temporal_train = train_df.copy()
temporal_val = val_df.copy()
temporal_test = test_df.copy()


# ---------------------------------------------------------------
# Binary target
# ---------------------------------------------------------------

for dataset in [
    temporal_train,
    temporal_val,
    temporal_test,
]:

    dataset["target"] = (
        dataset["temporal_label"]
        == "flawed"
    ).astype(int)


# ---------------------------------------------------------------
# Train temporal detector
# ---------------------------------------------------------------

models = make_models()


temporal_model, temporal_result = (
    train_models(

        temporal_train,

        temporal_val,

        temporal_test,

        "target",

        "temporal_detector",

        models,

    )
)


# ===============================================================
# FLAW TYPE
# ===============================================================

header(
    "PREPARING FLAW TYPE DATA"
)


type_train = train_df[
    train_df["temporal_label"]
    == "flawed"
].copy()


type_val = val_df[
    val_df["temporal_label"]
    == "flawed"
].copy()


type_test = test_df[
    test_df["temporal_label"]
    == "flawed"
].copy()


print(
    f"Flawed train windows: "
    f"{len(type_train):,}"
)

print(
    f"Flawed validation windows: "
    f"{len(type_val):,}"
)

print(
    f"Flawed test windows: "
    f"{len(type_test):,}"
)


models = make_models()


type_model, type_result = (
    train_models(

        type_train,

        type_val,

        type_test,

        "flaw_type",

        "flaw_type",

        models,

    )
)


# ===============================================================
# SEVERITY
# ===============================================================

header(
    "PREPARING SEVERITY DATA"
)


severity_train = train_df[
    train_df["temporal_label"]
    == "flawed"
].copy()


severity_val = val_df[
    val_df["temporal_label"]
    == "flawed"
].copy()


severity_test = test_df[
    test_df["temporal_label"]
    == "flawed"
].copy()


models = make_models()


severity_model, severity_result = (
    train_models(

        severity_train,

        severity_val,

        severity_test,

        "severity",

        "severity",

        models,

    )
)


# ===============================================================
# SAVE SUMMARY
# ===============================================================

summary = {

    "temporal": {

        "best_model":
            max(
                [
                    temporal_result
                ],
                key=lambda x:
                    x["val_f1"]
            ) is not None,

        "validation_f1":
            temporal_result["val_f1"],

        "test_f1":
            temporal_result["test_f1"],

        "test_accuracy":
            temporal_result["test_accuracy"],

    },

    "flaw_type": {

        "validation_f1":
            type_result["val_f1"],

        "test_f1":
            type_result["test_f1"],

        "test_accuracy":
            type_result["test_accuracy"],

    },

    "severity": {

        "validation_f1":
            severity_result["val_f1"],

        "test_f1":
            severity_result["test_f1"],

        "test_accuracy":
            severity_result["test_accuracy"],

    },

}


summary_path = (
    RESULT_DIR
    / "contrastive_training_summary.json"
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
    "CONTRASTIVE TRAINING COMPLETE"
)


print(
    f"""
Contrastive dataset:
    {len(df):,} windows

Features:
    {len(feature_columns)}

Temporal detector:
    Test F1 = {temporal_result["test_f1"]:.4f}

Flaw type:
    Test F1 = {type_result["test_f1"]:.4f}

Severity:
    Test F1 = {severity_result["test_f1"]:.4f}

Summary:
    {summary_path}
"""
)


print(
    "✅ Contrastive models trained successfully."
)