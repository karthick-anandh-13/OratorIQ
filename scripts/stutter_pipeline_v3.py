"""
ORATORIQ STUTTER V3
===================

Multi-label fluency/stutter event classification.

V3 improvements over V2:
1. Reuses the existing V2 feature cache.
2. Uses the original annotation columns directly.
3. Treats stutter events as MULTI-LABEL instead of forcing one primary event.
4. Trains separate binary classifiers for:
      - Prolongation
      - Block
      - SoundRep
      - WordRep
      - Interjection
5. Uses class balancing.
6. Tunes decision thresholds on validation data.
7. Keeps episode-safe train/validation/test split.
8. Evaluates on untouched test data.
9. Saves models, predictions, reports and threshold information.

NO AUDIO FEATURE EXTRACTION IS PERFORMED.
"""

from __future__ import annotations

import json
import pickle
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(r"D:\OratorIQ")

METADATA_PATH = (
    ROOT
    / "data"
    / "stutter"
    / "metadata"
    / "stutter_metadata.csv"
)

FEATURE_CACHE_PATH = (
    ROOT
    / "data"
    / "stutter"
    / "features"
    / "stutter_features_v2.pkl"
)

RESULTS_DIR = (
    ROOT
    / "artifacts"
    / "stutter_results"
)

MODELS_DIR = (
    ROOT
    / "artifacts"
    / "stutter_models"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Original SEP-28k event columns
EVENT_COLUMNS = [
    "Prolongation",
    "Block",
    "SoundRep",
    "WordRep",
    "Interjection",
]


RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def banner(title: str):
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def timer():
    return time.time()


def elapsed_seconds(start_time):
    return time.time() - start_time


def save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_path_string(value):
    """
    Normalize Windows paths for matching.
    """
    if pd.isna(value):
        return ""

    value = str(value).strip()

    return value.replace("/", "\\").lower()


def find_feature_dataframe(obj):
    """
    Attempt to locate the feature DataFrame inside the V2 cache.

    Supports common formats:
      - DataFrame
      - dict containing DataFrame
      - dict containing 'data'
      - dict containing 'features'
    """

    if isinstance(obj, pd.DataFrame):
        return obj

    if isinstance(obj, dict):

        preferred_keys = [
            "data",
            "features",
            "feature_df",
            "df",
            "feature_data",
        ]

        for key in preferred_keys:
            if key in obj:
                value = obj[key]

                if isinstance(value, pd.DataFrame):
                    return value

                if isinstance(value, np.ndarray):
                    return pd.DataFrame(value)

        # Search any dictionary value
        for key, value in obj.items():

            if isinstance(value, pd.DataFrame):
                return value

    raise ValueError(
        "Could not locate a pandas DataFrame inside the V2 feature cache."
    )


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    banner("[1/8] Loading metadata")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_PATH}"
        )

    metadata = pd.read_csv(METADATA_PATH)

    print(f"Metadata rows : {len(metadata):,}")
    print(f"Columns       : {len(metadata.columns)}")

    required_columns = [
        "audio_path",
        "filename",
        "show",
        "episode_id",
        "clip_id",
        "dataset",
        "has_stutter",
        *EVENT_COLUMNS,
    ]

    missing = [
        c for c in required_columns
        if c not in metadata.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required metadata columns: {missing}"
        )

    print("\nEvent distribution:")

    for event in EVENT_COLUMNS:
        count = int(metadata[event].fillna(0).sum())
        print(f"  {event:<16}: {count:,}")

    print("\nStutter distribution:")

    print(
        metadata["has_stutter"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return metadata


# ============================================================
# LOAD V2 FEATURES
# ============================================================

def load_v2_features():

    banner("[2/8] Loading existing V2 feature cache")

    if not FEATURE_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"V2 feature cache not found:\n{FEATURE_CACHE_PATH}"
        )

    start = timer()

    print(f"Loading:\n{FEATURE_CACHE_PATH}")

    obj = load_pickle(FEATURE_CACHE_PATH)

    features = find_feature_dataframe(obj)

    print(f"[OK] Feature rows : {len(features):,}")
    print(f"[OK] Feature cols : {len(features.columns):,}")

    print(
        f"[TIME] Loaded in {elapsed_seconds(start):.2f}s"
    )

    return features


# ============================================================
# MATCH FEATURES WITH METADATA
# ============================================================

def match_features_to_metadata(metadata, features):

    banner("[3/8] Matching feature rows with metadata")

    start = timer()

    metadata = metadata.copy()
    features = features.copy()

    print("Feature columns:")
    print(list(features.columns[:15]))

    # --------------------------------------------------------
    # Find likely identity column
    # --------------------------------------------------------

    possible_keys = [
        "audio_path",
        "filename",
        "file",
        "path",
    ]

    feature_key = None

    for key in possible_keys:
        if key in features.columns:
            feature_key = key
            break

    # --------------------------------------------------------
    # Direct audio path matching
    # --------------------------------------------------------

    if feature_key is not None and "audio_path" in metadata.columns:

        print(
            f"[MATCH] Using feature key: {feature_key}"
        )

        features["_match_key"] = (
            features[feature_key]
            .map(normalize_path_string)
        )

        metadata["_match_key"] = (
            metadata["audio_path"]
            .map(normalize_path_string)
        )

    else:

        # ----------------------------------------------------
        # Try filename matching
        # ----------------------------------------------------

        if (
            "filename" in features.columns
            and "filename" in metadata.columns
        ):

            print("[MATCH] Using filename")

            features["_match_key"] = (
                features["filename"]
                .map(normalize_path_string)
            )

            metadata["_match_key"] = (
                metadata["filename"]
                .map(normalize_path_string)
            )

        else:

            # ------------------------------------------------
            # Try composite key
            # ------------------------------------------------

            required = [
                "show",
                "episode_id",
                "clip_id",
            ]

            if all(c in features.columns for c in required):

                print(
                    "[MATCH] Using show + episode_id + clip_id"
                )

                features["_match_key"] = (
                    features["show"].astype(str)
                    + "|"
                    + features["episode_id"].astype(str)
                    + "|"
                    + features["clip_id"].astype(str)
                )

                metadata["_match_key"] = (
                    metadata["show"].astype(str)
                    + "|"
                    + metadata["episode_id"].astype(str)
                    + "|"
                    + metadata["clip_id"].astype(str)
                )

            else:

                raise ValueError(
                    "Could not determine a safe feature/metadata matching key."
                )

    # --------------------------------------------------------
    # Remove duplicate feature keys
    # --------------------------------------------------------

    duplicate_features = (
        features["_match_key"].duplicated().sum()
    )

    if duplicate_features:
        print(
            f"[WARNING] Duplicate feature keys: "
            f"{duplicate_features:,}"
        )

        features = (
            features
            .drop_duplicates("_match_key")
            .copy()
        )

    duplicate_metadata = (
        metadata["_match_key"].duplicated().sum()
    )

    if duplicate_metadata:
        print(
            f"[WARNING] Duplicate metadata keys: "
            f"{duplicate_metadata:,}"
        )

        metadata = (
            metadata
            .drop_duplicates("_match_key")
            .copy()
        )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = metadata.merge(
        features,
        on="_match_key",
        how="inner",
        suffixes=("", "_feature"),
    )

    print()
    print(f"Metadata rows          : {len(metadata):,}")
    print(f"Feature rows           : {len(features):,}")
    print(f"Matched rows           : {len(merged):,}")

    match_rate = (
        len(merged) / max(len(metadata), 1)
    )

    print(
        f"Match rate             : {match_rate * 100:.2f}%"
    )

    if match_rate < 0.95:
        raise RuntimeError(
            "Feature/metadata match rate is below 95%. "
            "Stopping before training."
        )

    merged = merged.drop(
        columns=["_match_key"],
        errors="ignore",
    )

    print(
        f"[TIME] Matching completed in "
        f"{elapsed_seconds(start):.2f}s"
    )

    return merged


# ============================================================
# IDENTIFY MODEL FEATURES
# ============================================================

def prepare_model_matrix(data):

    banner("[4/8] Preparing V3 feature matrix")

    # Metadata columns that must never become model features
    metadata_columns = {
        "audio_path",
        "filename",
        "show",
        "episode_id",
        "clip_id",
        "dataset",
        "start",
        "stop",
        "has_stutter",
        "primary_event",
        *EVENT_COLUMNS,

        # possible duplicated metadata columns
        "audio_path_feature",
        "filename_feature",
        "show_feature",
        "episode_id_feature",
        "clip_id_feature",
        "dataset_feature",
        "start_feature",
        "stop_feature",
        "has_stutter_feature",
        "primary_event_feature",

        # identity columns
        "record_id",
        "path",
        "file",
    }

    numeric_candidates = []

    for column in data.columns:

        if column in metadata_columns:
            continue

        if column.endswith("_feature"):
            continue

        if pd.api.types.is_numeric_dtype(
            data[column]
        ):
            numeric_candidates.append(column)

    if not numeric_candidates:
        raise ValueError(
            "No numeric model features found."
        )

    X = data[numeric_candidates].copy()

    # Replace invalid values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Median imputation
    X = X.fillna(
        X.median(numeric_only=True)
    )

    # Remaining NaNs
    X = X.fillna(0.0)

    print(
        f"Numeric model features : {X.shape[1]:,}"
    )

    print(
        f"Training rows           : {X.shape[0]:,}"
    )

    return X, numeric_candidates


# ============================================================
# EPISODE-SAFE SPLIT
# ============================================================

def create_episode_safe_split(data):

    banner("[5/8] Creating episode-safe split")

    groups = (
        data["show"].astype(str)
        + "||"
        + data["episode_id"].astype(str)
    )

    # First:
    # 80% train, 20% temporary
    splitter_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_idx, temp_idx = next(
        splitter_1.split(
            data,
            groups=groups,
        )
    )

    # Temporary = 20%
    # Split it into validation/test equally
    temp_groups = groups.iloc[temp_idx]

    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )

    val_relative, test_relative = next(
        splitter_2.split(
            temp_idx,
            groups=temp_groups,
        )
    )

    val_idx = temp_idx[val_relative]
    test_idx = temp_idx[test_relative]

    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    test_idx = np.asarray(test_idx)

    train_groups = set(groups.iloc[train_idx])
    val_groups = set(groups.iloc[val_idx])
    test_groups = set(groups.iloc[test_idx])

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if train_groups & val_groups:
        raise RuntimeError(
            "Episode leakage detected: train/validation overlap."
        )

    if train_groups & test_groups:
        raise RuntimeError(
            "Episode leakage detected: train/test overlap."
        )

    if val_groups & test_groups:
        raise RuntimeError(
            "Episode leakage detected: validation/test overlap."
        )

    print(
        f"Train      : {len(train_idx):,}"
    )

    print(
        f"Validation : {len(val_idx):,}"
    )

    print(
        f"Test       : {len(test_idx):,}"
    )

    print(
        f"Train episodes      : {len(train_groups):,}"
    )

    print(
        f"Validation episodes : {len(val_groups):,}"
    )

    print(
        f"Test episodes       : {len(test_groups):,}"
    )

    return train_idx, val_idx, test_idx


# ============================================================
# THRESHOLD TUNING
# ============================================================

def tune_threshold(
    y_true,
    probabilities,
    min_threshold=0.10,
    max_threshold=0.90,
    step=0.01,
):

    thresholds = np.arange(
        min_threshold,
        max_threshold + step / 2,
        step,
    )

    best_threshold = 0.50
    best_f1 = -1.0

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        results.append(
            {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold, best_f1, results


# ============================================================
# DETECTOR
# ============================================================

def train_stutter_detector(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
):

    banner("[6/8] Training balanced stutter detector")

    models = {}

    # --------------------------------------------------------
    # ExtraTrees
    # --------------------------------------------------------

    print("\n[MODEL] ExtraTrees")

    model_et = ExtraTreesClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    )

    start = timer()

    model_et.fit(
        X_train,
        y_train,
    )

    print(
        f"Training time: "
        f"{elapsed_seconds(start):.2f}s"
    )

    val_prob_et = model_et.predict_proba(
        X_val
    )[:, 1]

    test_prob_et = model_et.predict_proba(
        X_test
    )[:, 1]

    threshold_et, _, _ = tune_threshold(
        y_val,
        val_prob_et,
    )

    val_pred_et = (
        val_prob_et >= threshold_et
    ).astype(int)

    test_pred_et = (
        test_prob_et >= threshold_et
    ).astype(int)

    et_val_f1 = f1_score(
        y_val,
        val_pred_et,
        average="macro",
        zero_division=0,
    )

    et_test_f1 = f1_score(
        y_test,
        test_pred_et,
        average="macro",
        zero_division=0,
    )

    print(
        f"Threshold             : {threshold_et:.2f}"
    )

    print(
        f"Validation macro F1   : {et_val_f1:.4f}"
    )

    print(
        f"Test macro F1         : {et_test_f1:.4f}"
    )

    models["ExtraTrees"] = {
        "model": model_et,
        "threshold": threshold_et,
        "validation_macro_f1": et_val_f1,
        "test_macro_f1": et_test_f1,
        "test_predictions": test_pred_et,
        "test_probabilities": test_prob_et,
    }

    # --------------------------------------------------------
    # RandomForest
    # --------------------------------------------------------

    print("\n[MODEL] RandomForest")

    model_rf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    )

    start = timer()

    model_rf.fit(
        X_train,
        y_train,
    )

    print(
        f"Training time: "
        f"{elapsed_seconds(start):.2f}s"
    )

    val_prob_rf = model_rf.predict_proba(
        X_val
    )[:, 1]

    test_prob_rf = model_rf.predict_proba(
        X_test
    )[:, 1]

    threshold_rf, _, _ = tune_threshold(
        y_val,
        val_prob_rf,
    )

    val_pred_rf = (
        val_prob_rf >= threshold_rf
    ).astype(int)

    test_pred_rf = (
        test_prob_rf >= threshold_rf
    ).astype(int)

    rf_val_f1 = f1_score(
        y_val,
        val_pred_rf,
        average="macro",
        zero_division=0,
    )

    rf_test_f1 = f1_score(
        y_test,
        test_pred_rf,
        average="macro",
        zero_division=0,
    )

    print(
        f"Threshold             : {threshold_rf:.2f}"
    )

    print(
        f"Validation macro F1   : {rf_val_f1:.4f}"
    )

    print(
        f"Test macro F1         : {rf_test_f1:.4f}"
    )

    models["RandomForest"] = {
        "model": model_rf,
        "threshold": threshold_rf,
        "validation_macro_f1": rf_val_f1,
        "test_macro_f1": rf_test_f1,
        "test_predictions": test_pred_rf,
        "test_probabilities": test_prob_rf,
    }

    # --------------------------------------------------------
    # Select best by validation macro F1
    # --------------------------------------------------------

    best_name = max(
        models,
        key=lambda name:
        models[name]["validation_macro_f1"],
    )

    best = models[best_name]

    print()
    print(
        f"[BEST DETECTOR] {best_name}"
    )

    print(
        f"Validation macro F1 : "
        f"{best['validation_macro_f1']:.4f}"
    )

    print(
        f"Test macro F1       : "
        f"{best['test_macro_f1']:.4f}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODELS_DIR
        / "stutter_detector_v3.joblib"
    )

    joblib.dump(
        {
            "model": best["model"],
            "threshold": best["threshold"],
            "version": "v3",
            "model_name": best_name,
            "random_state": RANDOM_STATE,
        },
        model_path,
    )

    print(
        f"[SAVED] {model_path}"
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    test_predictions = best["test_predictions"]

    report = classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_test,
        test_predictions,
    )

    detector_report = {
        "version": "v3",
        "model": best_name,
        "threshold": float(
            best["threshold"]
        ),
        "validation_macro_f1": float(
            best["validation_macro_f1"]
        ),
        "test_macro_f1": float(
            best["test_macro_f1"]
        ),
        "test_accuracy": float(
            accuracy_score(
                y_test,
                test_predictions,
            )
        ),
        "classification_report": report,
    }

    save_json(
        RESULTS_DIR
        / "stutter_v3_detector_report.json",
        detector_report,
    )

    pd.DataFrame(
        confusion,
        index=["normal", "stutter"],
        columns=["normal", "stutter"],
    ).to_csv(
        RESULTS_DIR
        / "stutter_v3_detector_confusion.csv"
    )

    return best


# ============================================================
# MULTI-LABEL EVENT CLASSIFIERS
# ============================================================

def train_event_models(
    X_train,
    X_val,
    X_test,
    data_train,
    data_val,
    data_test,
):

    banner("[7/8] Training multi-label event classifiers")

    all_results = {}

    prediction_table = pd.DataFrame(
        index=data_test.index
    )

    # Store true labels
    for event in EVENT_COLUMNS:

        prediction_table[
            f"true_{event}"
        ] = (
            data_test[event]
            .astype(int)
            .values
        )

    for event_number, event in enumerate(
        EVENT_COLUMNS,
        start=1,
    ):

        print()
        print(
            "-" * 70
        )

        print(
            f"[EVENT {event_number}/{len(EVENT_COLUMNS)}] "
            f"{event}"
        )

        y_train = (
            data_train[event]
            .astype(int)
            .values
        )

        y_val = (
            data_val[event]
            .astype(int)
            .values
        )

        y_test = (
            data_test[event]
            .astype(int)
            .values
        )

        print(
            f"Train positives : {y_train.sum():,} "
            f"/ {len(y_train):,}"
        )

        print(
            f"Val positives   : {y_val.sum():,} "
            f"/ {len(y_val):,}"
        )

        print(
            f"Test positives  : {y_test.sum():,} "
            f"/ {len(y_test):,}"
        )

        candidates = {}

        # ----------------------------------------------------
        # ExtraTrees
        # ----------------------------------------------------

        print("\n[MODEL] ExtraTrees")

        model_et = ExtraTreesClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=2,
        )

        model_et.fit(
            X_train,
            y_train,
        )

        val_prob_et = model_et.predict_proba(
            X_val
        )[:, 1]

        test_prob_et = model_et.predict_proba(
            X_test
        )[:, 1]

        threshold_et, _, threshold_results_et = (
            tune_threshold(
                y_val,
                val_prob_et,
            )
        )

        val_pred_et = (
            val_prob_et >= threshold_et
        ).astype(int)

        test_pred_et = (
            test_prob_et >= threshold_et
        ).astype(int)

        et_val_f1 = f1_score(
            y_val,
            val_pred_et,
            zero_division=0,
        )

        et_test_f1 = f1_score(
            y_test,
            test_pred_et,
            zero_division=0,
        )

        print(
            f"Threshold         : {threshold_et:.2f}"
        )

        print(
            f"Validation F1     : {et_val_f1:.4f}"
        )

        print(
            f"Test F1           : {et_test_f1:.4f}"
        )

        candidates["ExtraTrees"] = {
            "model": model_et,
            "threshold": threshold_et,
            "validation_f1": et_val_f1,
            "test_f1": et_test_f1,
            "test_predictions": test_pred_et,
            "test_probabilities": test_prob_et,
            "threshold_results": threshold_results_et,
        }

        # ----------------------------------------------------
        # RandomForest
        # ----------------------------------------------------

        print("\n[MODEL] RandomForest")

        model_rf = RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=2,
        )

        model_rf.fit(
            X_train,
            y_train,
        )

        val_prob_rf = model_rf.predict_proba(
            X_val
        )[:, 1]

        test_prob_rf = model_rf.predict_proba(
            X_test
        )[:, 1]

        threshold_rf, _, threshold_results_rf = (
            tune_threshold(
                y_val,
                val_prob_rf,
            )
        )

        val_pred_rf = (
            val_prob_rf >= threshold_rf
        ).astype(int)

        test_pred_rf = (
            test_prob_rf >= threshold_rf
        ).astype(int)

        rf_val_f1 = f1_score(
            y_val,
            val_pred_rf,
            zero_division=0,
        )

        rf_test_f1 = f1_score(
            y_test,
            test_pred_rf,
            zero_division=0,
        )

        print(
            f"Threshold         : {threshold_rf:.2f}"
        )

        print(
            f"Validation F1     : {rf_val_f1:.4f}"
        )

        print(
            f"Test F1           : {rf_test_f1:.4f}"
        )

        candidates["RandomForest"] = {
            "model": model_rf,
            "threshold": threshold_rf,
            "validation_f1": rf_val_f1,
            "test_f1": rf_test_f1,
            "test_predictions": test_pred_rf,
            "test_probabilities": test_prob_rf,
            "threshold_results": threshold_results_rf,
        }

        # ----------------------------------------------------
        # Select best by validation F1
        # ----------------------------------------------------

        best_name = max(
            candidates,
            key=lambda name:
            candidates[name]["validation_f1"],
        )

        best = candidates[best_name]

        print()
        print(
            f"[BEST] {event}: {best_name}"
        )

        print(
            f"Validation F1 : "
            f"{best['validation_f1']:.4f}"
        )

        print(
            f"Test F1       : "
            f"{best['test_f1']:.4f}"
        )

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        safe_event = event.lower()

        model_path = (
            MODELS_DIR
            / f"stutter_v3_event_{safe_event}.joblib"
        )

        joblib.dump(
            {
                "model": best["model"],
                "threshold": best["threshold"],
                "event": event,
                "version": "v3",
                "model_name": best_name,
                "random_state": RANDOM_STATE,
            },
            model_path,
        )

        print(
            f"[SAVED] {model_path}"
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        test_pred = best[
            "test_predictions"
        ]

        report = classification_report(
            y_test,
            test_pred,
            output_dict=True,
            zero_division=0,
        )

        confusion = confusion_matrix(
            y_test,
            test_pred,
        )

        precision = precision_score(
            y_test,
            test_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            test_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            test_pred,
            zero_division=0,
        )

        all_results[event] = {
            "model": best_name,
            "threshold": float(
                best["threshold"]
            ),
            "validation_f1": float(
                best["validation_f1"]
            ),
            "test_f1": float(f1),
            "test_precision": float(
                precision
            ),
            "test_recall": float(
                recall
            ),
            "test_support": int(
                y_test.sum()
            ),
            "classification_report": report,
        }

        prediction_table[
            f"prob_{event}"
        ] = best[
            "test_probabilities"
        ]

        prediction_table[
            f"pred_{event}"
        ] = test_pred

        # Save event confusion
        pd.DataFrame(
            confusion,
            index=["0", "1"],
            columns=["0", "1"],
        ).to_csv(
            RESULTS_DIR
            / f"stutter_v3_{safe_event}_confusion.csv"
        )

        # Save threshold sweep
        pd.DataFrame(
            best["threshold_results"]
        ).to_csv(
            RESULTS_DIR
            / f"stutter_v3_{safe_event}_thresholds.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_path = (
        RESULTS_DIR
        / "stutter_v3_multilabel_predictions.csv"
    )

    prediction_table.to_csv(
        prediction_path,
        index=False,
    )

    print()
    print(
        f"[SAVED] {prediction_path}"
    )

    return all_results


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_event_feature_importance(
    event_models,
    feature_names,
):

    banner("[8/8] Saving V3 feature importance")

    rows = []

    for event, result in event_models.items():

        model_path = (
            MODELS_DIR
            / f"stutter_v3_event_{event.lower()}.joblib"
        )

        if not model_path.exists():
            continue

        bundle = joblib.load(
            model_path
        )

        model = bundle["model"]

        if not hasattr(
            model,
            "feature_importances_",
        ):
            continue

        importances = (
            model.feature_importances_
        )

        for feature, importance in zip(
            feature_names,
            importances,
        ):
            rows.append(
                {
                    "event": event,
                    "feature": feature,
                    "importance": float(
                        importance
                    ),
                }
            )

    if not rows:
        print(
            "[WARNING] No feature importance available."
        )
        return

    importance_df = pd.DataFrame(rows)

    importance_df = (
        importance_df
        .sort_values(
            ["event", "importance"],
            ascending=[True, False],
        )
    )

    output_path = (
        RESULTS_DIR
        / "stutter_v3_event_feature_importance.csv"
    )

    importance_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"[SAVED] {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = timer()

    banner(
        "ORATORIQ STUTTER V3"
    )

    print(
        "Multi-label fluency/stutter event pipeline"
    )

    print(
        "Reusing existing V2 feature cache"
    )

    print(
        f"\nProject root:\n{ROOT}"
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = load_metadata()

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    features = load_v2_features()

    # --------------------------------------------------------
    # Match
    # --------------------------------------------------------

    data = match_features_to_metadata(
        metadata,
        features,
    )

    # --------------------------------------------------------
    # Model matrix
    # --------------------------------------------------------

    X, feature_names = (
        prepare_model_matrix(data)
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_idx, val_idx, test_idx = (
        create_episode_safe_split(data)
    )

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    X_test = X.iloc[test_idx]

    data_train = data.iloc[train_idx]
    data_val = data.iloc[val_idx]
    data_test = data.iloc[test_idx]

    # --------------------------------------------------------
    # Detector target
    # --------------------------------------------------------

    y_train_detector = (
        data_train["has_stutter"]
        .astype(int)
        .values
    )

    y_val_detector = (
        data_val["has_stutter"]
        .astype(int)
        .values
    )

    y_test_detector = (
        data_test["has_stutter"]
        .astype(int)
        .values
    )

    # --------------------------------------------------------
    # Detector
    # --------------------------------------------------------

    best_detector = train_stutter_detector(
        X_train,
        X_val,
        X_test,
        y_train_detector,
        y_val_detector,
        y_test_detector,
    )

    # --------------------------------------------------------
    # Multi-label event models
    # --------------------------------------------------------

    event_results = train_event_models(
        X_train,
        X_val,
        X_test,
        data_train,
        data_val,
        data_test,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    save_event_feature_importance(
        event_results,
        feature_names,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    detector_test_f1 = (
        best_detector["test_macro_f1"]
    )

    detector_accuracy = (
        accuracy_score(
            y_test_detector,
            best_detector["test_predictions"],
        )
    )

    event_f1_values = [
        result["test_f1"]
        for result in event_results.values()
    ]

    macro_event_f1 = float(
        np.mean(event_f1_values)
    )

    summary = {
        "version": "v3",
        "feature_cache": str(
            FEATURE_CACHE_PATH
        ),
        "feature_rows": int(
            len(data)
        ),
        "feature_count": int(
            len(feature_names)
        ),
        "events": EVENT_COLUMNS,
        "detector": {
            "model": "best_validation_model",
            "threshold": float(
                best_detector["threshold"]
            ),
            "test_accuracy": float(
                detector_accuracy
            ),
            "test_macro_f1": float(
                detector_test_f1
            ),
        },
        "multi_label_events": {
            "macro_f1": macro_event_f1,
            "per_event": event_results,
        },
        "split": {
            "train_rows": int(
                len(train_idx)
            ),
            "validation_rows": int(
                len(val_idx)
            ),
            "test_rows": int(
                len(test_idx)
            ),
        },
    }

    save_json(
        RESULTS_DIR
        / "stutter_v3_summary.json",
        summary,
    )

    # --------------------------------------------------------
    # Final console
    # --------------------------------------------------------

    banner(
        "ORATORIQ STUTTER V3 COMPLETE"
    )

    print(
        f"Feature rows       : {len(data):,}"
    )

    print(
        f"Features           : {len(feature_names):,}"
    )

    print(
        f"Detector accuracy  : {detector_accuracy:.4f}"
    )

    print(
        f"Detector macro F1  : {detector_test_f1:.4f}"
    )

    print(
        f"Event macro F1     : {macro_event_f1:.4f}"
    )

    print()
    print(
        "Per-event F1:"
    )

    for event, result in event_results.items():

        print(
            f"  {event:<16} "
            f"{result['test_f1']:.4f}"
        )

    print()
    print(
        f"Total runtime: "
        f"{elapsed_seconds(total_start) / 60:.2f} minutes"
    )

    print()
    print(
        "Models:"
    )

    print(
        f"  {MODELS_DIR / 'stutter_detector_v3.joblib'}"
    )

    for event in EVENT_COLUMNS:

        print(
            f"  {MODELS_DIR / f'stutter_v3_event_{event.lower()}.joblib'}"
        )

    print()
    print(
        "Results:"
    )

    print(
        f"  {RESULTS_DIR}"
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":
    main()