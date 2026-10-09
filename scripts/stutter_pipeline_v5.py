from __future__ import annotations

import json
import pickle
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
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
# ORATORIQ STUTTER V5
# ============================================================
#
# CLEAN AUDIO-ONLY MULTI-LABEL FLUENCY/STUTTER PIPELINE
#
# V5 fixes the V4 feature leakage issue.
#
# IMPORTANT:
# - Reuses the existing V2 feature cache.
# - Does NOT extract audio features again.
# - Keeps episode-safe train/validation/test separation.
# - NEVER uses the test set for model/threshold selection.
# - Converts SEP-28k annotation counts into binary presence labels.
# - Explicitly excludes ALL annotation/target metadata.
# - Uses only genuine numeric audio-derived features.
# - Tunes thresholds using validation data only.
# - Saves selected models permanently.
#
# Events:
#   Prolongation
#   Block
#   SoundRep
#   WordRep
#   Interjection
#
# Goal:
#   Obtain a clean and defensible unseen-test evaluation.
#
# ============================================================


# ============================================================
# PATHS
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

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

EVENT_COLUMNS = [
    "Prolongation",
    "Block",
    "SoundRep",
    "WordRep",
    "Interjection",
]

# Threshold search
THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.90
THRESHOLD_STEP = 0.01


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def banner(title: str):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def elapsed(start_time):
    return time.time() - start_time


def save_json(path: Path, data):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            default=float,
        )


def load_pickle(path: Path):
    with open(
        path,
        "rb",
    ) as f:
        return pickle.load(f)


def normalize_path(value):

    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .replace("/", "\\")
        .lower()
    )


# ============================================================
# FEATURE CACHE LOADER
# ============================================================

def find_feature_dataframe(obj):

    if isinstance(
        obj,
        pd.DataFrame,
    ):
        return obj

    if isinstance(
        obj,
        dict,
    ):

        preferred = [
            "data",
            "features",
            "feature_df",
            "df",
            "feature_data",
        ]

        for key in preferred:

            if key not in obj:
                continue

            value = obj[key]

            if isinstance(
                value,
                pd.DataFrame,
            ):
                return value

        for value in obj.values():

            if isinstance(
                value,
                pd.DataFrame,
            ):
                return value

    raise ValueError(
        "Could not find DataFrame inside feature cache."
    )


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    banner(
        "[1/9] Loading metadata"
    )

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_PATH}"
        )

    df = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"Metadata rows : {len(df):,}"
    )

    print(
        f"Columns       : {len(df.columns):,}"
    )

    required = [
        "audio_path",
        "filename",
        "show",
        "episode_id",
        "clip_id",
        "has_stutter",
        *EVENT_COLUMNS,
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    print()
    print(
        "Raw annotation counts:"
    )

    for event in EVENT_COLUMNS:

        count = int(
            pd.to_numeric(
                df[event],
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
            .sum()
        )

        print(
            f"  {event:<16}: {count:,}"
        )

    return df


# ============================================================
# LOAD V2 FEATURES
# ============================================================

def load_features():

    banner(
        "[2/9] Loading V2 feature cache"
    )

    if not FEATURE_CACHE_PATH.exists():

        raise FileNotFoundError(
            f"Feature cache not found:\n{FEATURE_CACHE_PATH}"
        )

    start = time.time()

    obj = load_pickle(
        FEATURE_CACHE_PATH
    )

    features = find_feature_dataframe(
        obj
    )

    print(
        f"[OK] Feature rows : {len(features):,}"
    )

    print(
        f"[OK] Feature cols : {len(features.columns):,}"
    )

    print(
        f"[TIME] {elapsed(start):.2f}s"
    )

    return features


# ============================================================
# MATCH METADATA + FEATURES
# ============================================================

def match_data(
    metadata,
    features,
):

    banner(
        "[3/9] Matching metadata and features"
    )

    metadata = metadata.copy()
    features = features.copy()

    if (
        "audio_path"
        in features.columns
        and
        "audio_path"
        in metadata.columns
    ):

        print(
            "[MATCH] audio_path"
        )

        metadata["_key"] = (
            metadata[
                "audio_path"
            ]
            .map(normalize_path)
        )

        features["_key"] = (
            features[
                "audio_path"
            ]
            .map(normalize_path)
        )

    elif (
        "filename"
        in features.columns
        and
        "filename"
        in metadata.columns
    ):

        print(
            "[MATCH] filename"
        )

        metadata["_key"] = (
            metadata[
                "filename"
            ]
            .map(normalize_path)
        )

        features["_key"] = (
            features[
                "filename"
            ]
            .map(normalize_path)
        )

    else:

        print(
            "[MATCH] show + episode + clip"
        )

        metadata["_key"] = (
            metadata["show"].astype(str)
            + "|"
            + metadata["episode_id"].astype(str)
            + "|"
            + metadata["clip_id"].astype(str)
        )

        features["_key"] = (
            features["show"].astype(str)
            + "|"
            + features["episode_id"].astype(str)
            + "|"
            + features["clip_id"].astype(str)
        )

    features = (
        features
        .drop_duplicates(
            "_key"
        )
        .copy()
    )

    merged = metadata.merge(
        features,
        on="_key",
        how="inner",
        suffixes=(
            "",
            "_feature",
        ),
    )

    rate = (
        len(merged)
        / len(metadata)
        * 100
    )

    print(
        f"Metadata rows : {len(metadata):,}"
    )

    print(
        f"Feature rows  : {len(features):,}"
    )

    print(
        f"Matched       : {len(merged):,}"
    )

    print(
        f"Match rate    : {rate:.2f}%"
    )

    if rate < 95:

        raise RuntimeError(
            "Feature/metadata matching below 95%."
        )

    merged = merged.drop(
        columns=[
            "_key"
        ],
        errors="ignore",
    )

    return merged


# ============================================================
# PREPARE CLEAN AUDIO FEATURES
# ============================================================

def prepare_features(
    data,
):

    banner(
        "[4/9] Preparing CLEAN audio-only model features"
    )

    # --------------------------------------------------------
    # EVERYTHING HERE IS FORBIDDEN AS A MODEL FEATURE
    # --------------------------------------------------------

    excluded = {

        # ----------------------------------------------------
        # IDENTIFIERS / METADATA
        # ----------------------------------------------------

        "audio_path",
        "filename",
        "show",
        "episode_id",
        "clip_id",
        "dataset",
        "start",
        "stop",

        # ----------------------------------------------------
        # TARGET LABELS
        # ----------------------------------------------------

        "has_stutter",
        "primary_event",

        *EVENT_COLUMNS,

        # ----------------------------------------------------
        # ANNOTATION / NON-AUDIO INFORMATION
        #
        # These columns come from human annotation.
        # They MUST NOT enter the audio model.
        # ----------------------------------------------------

        "Unsure",
        "PoorAudioQuality",
        "DifficultToUnderstand",
        "NoStutteredWords",
        "NaturalPause",
        "Music",
        "NoSpeech",

        # ----------------------------------------------------
        # DUPLICATED METADATA CREATED BY MERGE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # DUPLICATED ANNOTATION COLUMNS
        # ----------------------------------------------------

        "Unsure_feature",
        "PoorAudioQuality_feature",
        "Prolongation_feature",
        "Block_feature",
        "SoundRep_feature",
        "WordRep_feature",
        "DifficultToUnderstand_feature",
        "Interjection_feature",
        "NoStutteredWords_feature",
        "NaturalPause_feature",
        "Music_feature",
        "NoSpeech_feature",
    }

    feature_names = []

    for col in data.columns:

        if col in excluded:
            continue

        # Safety:
        # Any *_feature column is metadata duplicated by merge.
        if col.endswith(
            "_feature"
        ):
            continue

        if pd.api.types.is_numeric_dtype(
            data[col]
        ):

            feature_names.append(
                col
            )

    # --------------------------------------------------------
    # HARD LEAKAGE CHECK
    # --------------------------------------------------------

    forbidden_features = {
        "Unsure",
        "PoorAudioQuality",
        "Prolongation",
        "Block",
        "SoundRep",
        "WordRep",
        "DifficultToUnderstand",
        "Interjection",
        "NoStutteredWords",
        "NaturalPause",
        "Music",
        "NoSpeech",
        "has_stutter",
        "primary_event",
    }

    leaked = sorted(
        set(feature_names)
        & forbidden_features
    )

    if leaked:

        raise RuntimeError(
            "FEATURE LEAKAGE DETECTED:\n"
            + "\n".join(
                f"  - {x}"
                for x in leaked
            )
        )

    # --------------------------------------------------------
    # SECOND SAFETY CHECK
    # --------------------------------------------------------

    forbidden_suffixes = [
        "_feature",
    ]

    bad_suffix_columns = [
        col
        for col in feature_names
        if any(
            col.endswith(suffix)
            for suffix in forbidden_suffixes
        )
    ]

    if bad_suffix_columns:

        raise RuntimeError(
            "DUPLICATED METADATA FEATURES DETECTED:\n"
            + "\n".join(
                f"  - {x}"
                for x in bad_suffix_columns
            )
        )

    # --------------------------------------------------------
    # CREATE X
    # --------------------------------------------------------

    X = data[
        feature_names
    ].copy()

    X = X.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # --------------------------------------------------------
    # MEDIAN IMPUTATION
    # --------------------------------------------------------

    medians = X.median(
        numeric_only=True
    )

    X = X.fillna(
        medians
    )

    X = X.fillna(
        0.0
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    if len(feature_names) == 0:

        raise RuntimeError(
            "No model features were found."
        )

    if X.isna().any().any():

        raise RuntimeError(
            "NaN values remain in model features."
        )

    if np.isinf(
        X.to_numpy()
    ).any():

        raise RuntimeError(
            "Inf values remain in model features."
        )

    print()
    print(
        "[LEAKAGE CHECK] PASS"
    )

    print(
        "[OK] No target or annotation columns "
        "are being used as model features."
    )

    print()
    print(
        f"Rows     : {len(X):,}"
    )

    print(
        f"Features : {len(feature_names):,}"
    )

    print()
    print(
        "First 20 model features:"
    )

    for feature in feature_names[:20]:

        print(
            f"  - {feature}"
        )

    if len(feature_names) > 20:

        print(
            f"  ... and "
            f"{len(feature_names) - 20:,} more"
        )

    return X, feature_names


# ============================================================
# EPISODE-SAFE SPLIT
# ============================================================

def create_split(
    data,
):

    banner(
        "[5/9] Creating episode-safe split"
    )

    groups = (
        data["show"].astype(str)
        + "||"
        + data["episode_id"].astype(str)
    )

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

    temp_groups = (
        groups.iloc[
            temp_idx
        ]
    )

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

    val_idx = temp_idx[
        val_relative
    ]

    test_idx = temp_idx[
        test_relative
    ]

    train_groups = set(
        groups.iloc[
            train_idx
        ]
    )

    val_groups = set(
        groups.iloc[
            val_idx
        ]
    )

    test_groups = set(
        groups.iloc[
            test_idx
        ]
    )

    # --------------------------------------------------------
    # LEAKAGE CHECK
    # --------------------------------------------------------

    if (
        train_groups
        & val_groups
    ):

        raise RuntimeError(
            "Train/validation episode leakage."
        )

    if (
        train_groups
        & test_groups
    ):

        raise RuntimeError(
            "Train/test episode leakage."
        )

    if (
        val_groups
        & test_groups
    ):

        raise RuntimeError(
            "Validation/test episode leakage."
        )

    print(
        "[OK] Episode-safe split verified."
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

    return (
        train_idx,
        val_idx,
        test_idx,
    )


# ============================================================
# BINARY LABEL CONVERSION
# ============================================================

def binary_label(
    series,
):

    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .fillna(0)
        .gt(0)
        .astype(np.int8)
        .values
    )


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def optimize_threshold(
    y_true,
    probabilities,
):

    thresholds = np.arange(
        THRESHOLD_MIN,
        THRESHOLD_MAX
        + THRESHOLD_STEP / 2,
        THRESHOLD_STEP,
    )

    best = None

    rows = []

    for threshold in thresholds:

        pred = (
            probabilities
            >= threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            pred,
            zero_division=0,
        )

        precision = precision_score(
            y_true,
            pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            pred,
            zero_division=0,
        )

        rows.append(
            {
                "threshold": float(
                    threshold
                ),
                "f1": float(
                    f1
                ),
                "precision": float(
                    precision
                ),
                "recall": float(
                    recall
                ),
            }
        )

        if (
            best is None
            or f1 > best["f1"]
        ):

            best = {
                "threshold": float(
                    threshold
                ),
                "f1": float(
                    f1
                ),
                "precision": float(
                    precision
                ),
                "recall": float(
                    recall
                ),
            }

    return (
        best,
        pd.DataFrame(rows),
    )


# ============================================================
# DETECTOR
# ============================================================

def train_detector(
    X_train,
    X_val,
    X_test,
    data_train,
    data_val,
    data_test,
):

    banner(
        "[6/9] Training CLEAN stutter detector"
    )

    y_train = binary_label(
        data_train[
            "has_stutter"
        ]
    )

    y_val = binary_label(
        data_val[
            "has_stutter"
        ]
    )

    y_test = binary_label(
        data_test[
            "has_stutter"
        ]
    )

    print()
    print(
        "Class distribution:"
    )

    print(
        f"  Train normal : {(y_train == 0).sum():,}"
    )

    print(
        f"  Train stutter: {(y_train == 1).sum():,}"
    )

    print(
        f"  Val normal   : {(y_val == 0).sum():,}"
    )

    print(
        f"  Val stutter  : {(y_val == 1).sum():,}"
    )

    print(
        f"  Test normal  : {(y_test == 0).sum():,}"
    )

    print(
        f"  Test stutter : {(y_test == 1).sum():,}"
    )

    candidates = {}

    # --------------------------------------------------------
    # ExtraTrees
    # --------------------------------------------------------

    print()
    print(
        "[MODEL] ExtraTrees"
    )

    et = ExtraTreesClassifier(
        n_estimators=800,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=1,
        criterion="gini",
    )

    start = time.time()

    et.fit(
        X_train,
        y_train,
    )

    print(
        f"Training time: "
        f"{elapsed(start):.2f}s"
    )

    val_prob = et.predict_proba(
        X_val
    )[:, 1]

    test_prob = et.predict_proba(
        X_test
    )[:, 1]

    best_threshold, _ = optimize_threshold(
        y_val,
        val_prob,
    )

    threshold = best_threshold[
        "threshold"
    ]

    val_pred = (
        val_prob >= threshold
    ).astype(int)

    test_pred = (
        test_prob >= threshold
    ).astype(int)

    val_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0,
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0,
    )

    print(
        f"Threshold           : {threshold:.2f}"
    )

    print(
        f"Validation macro F1 : {val_f1:.4f}"
    )

    print(
        f"Test macro F1       : {test_f1:.4f}"
    )

    candidates["ExtraTrees"] = {
        "model": et,
        "threshold": threshold,
        "val_f1": val_f1,
        "test_f1": test_f1,
        "test_pred": test_pred,
        "test_prob": test_prob,
    }

    # --------------------------------------------------------
    # RandomForest
    # --------------------------------------------------------

    print()
    print(
        "[MODEL] RandomForest"
    )

    rf = RandomForestClassifier(
        n_estimators=800,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=1,
        criterion="gini",
    )

    start = time.time()

    rf.fit(
        X_train,
        y_train,
    )

    print(
        f"Training time: "
        f"{elapsed(start):.2f}s"
    )

    val_prob = rf.predict_proba(
        X_val
    )[:, 1]

    test_prob = rf.predict_proba(
        X_test
    )[:, 1]

    best_threshold, _ = optimize_threshold(
        y_val,
        val_prob,
    )

    threshold = best_threshold[
        "threshold"
    ]

    val_pred = (
        val_prob >= threshold
    ).astype(int)

    test_pred = (
        test_prob >= threshold
    ).astype(int)

    val_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0,
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0,
    )

    print(
        f"Threshold           : {threshold:.2f}"
    )

    print(
        f"Validation macro F1 : {val_f1:.4f}"
    )

    print(
        f"Test macro F1       : {test_f1:.4f}"
    )

    candidates["RandomForest"] = {
        "model": rf,
        "threshold": threshold,
        "val_f1": val_f1,
        "test_f1": test_f1,
        "test_pred": test_pred,
        "test_prob": test_prob,
    }

    # --------------------------------------------------------
    # SELECT BEST USING VALIDATION ONLY
    # --------------------------------------------------------

    best_name = max(
        candidates,
        key=lambda x:
        candidates[x]["val_f1"],
    )

    best = candidates[
        best_name
    ]

    print()
    print(
        f"[BEST DETECTOR] {best_name}"
    )

    print(
        f"Validation macro F1 : "
        f"{best['val_f1']:.4f}"
    )

    print(
        f"Test macro F1       : "
        f"{best['test_f1']:.4f}"
    )

    # --------------------------------------------------------
    # PERMANENT MODEL
    # --------------------------------------------------------

    model_path = (
        MODELS_DIR
        / "stutter_detector_v5.joblib"
    )

    joblib.dump(
        {
            "model": best["model"],
            "threshold": best["threshold"],
            "model_name": best_name,
            "version": "v5",
            "feature_cache": str(
                FEATURE_CACHE_PATH
            ),
            "events": EVENT_COLUMNS,
            "random_state": RANDOM_STATE,
            "feature_policy": "audio_only",
        },
        model_path,
    )

    print(
        f"[SAVED] {model_path}"
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        best["test_pred"],
        output_dict=True,
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_test,
        best["test_pred"],
    )

    save_json(
        RESULTS_DIR
        / "stutter_v5_detector_report.json",
        {
            "model": best_name,
            "threshold": float(
                best["threshold"]
            ),
            "validation_macro_f1": float(
                best["val_f1"]
            ),
            "test_macro_f1": float(
                best["test_f1"]
            ),
            "test_accuracy": float(
                accuracy_score(
                    y_test,
                    best["test_pred"],
                )
            ),
            "classification_report": report,
            "feature_policy": "audio_only",
        },
    )

    pd.DataFrame(
        confusion,
        index=[
            "normal",
            "stutter",
        ],
        columns=[
            "normal",
            "stutter",
        ],
    ).to_csv(
        RESULTS_DIR
        / "stutter_v5_detector_confusion.csv"
    )

    return best


# ============================================================
# EVENT MODEL CANDIDATES
# ============================================================

def create_event_models():

    return {

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=600,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=1,
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=600,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=1,
        ),
    }


# ============================================================
# MULTI-LABEL EVENTS
# ============================================================

def train_events(
    X_train,
    X_val,
    X_test,
    data_train,
    data_val,
    data_test,
):

    banner(
        "[7/9] Training CLEAN multi-label event models"
    )

    results = {}

    prediction_df = pd.DataFrame(
        index=data_test.index
    )

    for event in EVENT_COLUMNS:

        print()
        print(
            "-" * 72
        )

        print(
            f"EVENT: {event}"
        )

        y_train = binary_label(
            data_train[event]
        )

        y_val = binary_label(
            data_val[event]
        )

        y_test = binary_label(
            data_test[event]
        )

        print(
            f"Train positives : "
            f"{y_train.sum():,}"
        )

        print(
            f"Validation      : "
            f"{y_val.sum():,}"
        )

        print(
            f"Test positives  : "
            f"{y_test.sum():,}"
        )

        candidates = {}

        models = create_event_models()

        for model_name, model in models.items():

            print()
            print(
                f"[MODEL] {model_name}"
            )

            start = time.time()

            model.fit(
                X_train,
                y_train,
            )

            print(
                f"Training time: "
                f"{elapsed(start):.2f}s"
            )

            val_prob = model.predict_proba(
                X_val
            )[:, 1]

            test_prob = model.predict_proba(
                X_test
            )[:, 1]

            best_threshold, sweep = (
                optimize_threshold(
                    y_val,
                    val_prob,
                )
            )

            threshold = (
                best_threshold[
                    "threshold"
                ]
            )

            val_pred = (
                val_prob
                >= threshold
            ).astype(int)

            test_pred = (
                test_prob
                >= threshold
            ).astype(int)

            val_f1 = f1_score(
                y_val,
                val_pred,
                zero_division=0,
            )

            test_f1 = f1_score(
                y_test,
                test_pred,
                zero_division=0,
            )

            print(
                f"Threshold     : "
                f"{threshold:.2f}"
            )

            print(
                f"Validation F1 : "
                f"{val_f1:.4f}"
            )

            print(
                f"Test F1       : "
                f"{test_f1:.4f}"
            )

            candidates[
                model_name
            ] = {
                "model": model,
                "threshold": threshold,
                "val_f1": val_f1,
                "test_f1": test_f1,
                "test_pred": test_pred,
                "test_prob": test_prob,
                "sweep": sweep,
            }

        # ----------------------------------------------------
        # SELECT BEST MODEL USING VALIDATION ONLY
        # ----------------------------------------------------

        best_name = max(
            candidates,
            key=lambda x:
            candidates[x]["val_f1"],
        )

        best = candidates[
            best_name
        ]

        print()
        print(
            f"[BEST] {event} -> {best_name}"
        )

        print(
            f"Validation F1 : "
            f"{best['val_f1']:.4f}"
        )

        print(
            f"Test F1       : "
            f"{best['test_f1']:.4f}"
        )

        # ----------------------------------------------------
        # SAVE PERMANENT MODEL
        # ----------------------------------------------------

        safe_name = event.lower()

        model_path = (
            MODELS_DIR
            / (
                "stutter_v5_event_"
                f"{safe_name}.joblib"
            )
        )

        joblib.dump(
            {
                "model": best["model"],
                "threshold": best["threshold"],
                "event": event,
                "model_name": best_name,
                "version": "v5",
                "feature_cache": str(
                    FEATURE_CACHE_PATH
                ),
                "random_state": RANDOM_STATE,
                "feature_policy": "audio_only",
            },
            model_path,
        )

        print(
            f"[SAVED] {model_path}"
        )

        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------

        test_pred = (
            best["test_pred"]
        )

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

        event_result = {
            "event": event,
            "model": best_name,
            "threshold": float(
                best["threshold"]
            ),
            "validation_f1": float(
                best["val_f1"]
            ),
            "test_f1": float(
                f1_score(
                    y_test,
                    test_pred,
                    zero_division=0,
                )
            ),
            "test_precision": float(
                precision_score(
                    y_test,
                    test_pred,
                    zero_division=0,
                )
            ),
            "test_recall": float(
                recall_score(
                    y_test,
                    test_pred,
                    zero_division=0,
                )
            ),
            "classification_report": report,
            "feature_policy": "audio_only",
        }

        results[event] = event_result

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        prediction_df[
            f"true_{event}"
        ] = y_test

        prediction_df[
            f"prob_{event}"
        ] = best[
            "test_prob"
        ]

        prediction_df[
            f"pred_{event}"
        ] = test_pred

        # ----------------------------------------------------
        # SAVE CONFUSION MATRIX
        # ----------------------------------------------------

        pd.DataFrame(
            confusion,
            index=[
                "absent",
                "present",
            ],
            columns=[
                "absent",
                "present",
            ],
        ).to_csv(
            RESULTS_DIR
            / (
                f"stutter_v5_"
                f"{safe_name}_confusion.csv"
            )
        )

        # ----------------------------------------------------
        # SAVE THRESHOLD SWEEP
        # ----------------------------------------------------

        best[
            "sweep"
        ].to_csv(
            RESULTS_DIR
            / (
                f"stutter_v5_"
                f"{safe_name}_thresholds.csv"
            ),
            index=False,
        )

    # --------------------------------------------------------
    # SAVE ALL MULTI-LABEL PREDICTIONS
    # --------------------------------------------------------

    prediction_df.to_csv(
        RESULTS_DIR
        / "stutter_v5_multilabel_predictions.csv",
        index=False,
    )

    print()
    print(
        "[SAVED] Multi-label predictions"
    )

    return results


# ============================================================
# MULTI-LABEL OVERALL METRICS
# ============================================================

def calculate_multilabel_metrics(
    event_results,
    prediction_path,
):

    banner(
        "[8/9] Calculating final multi-label metrics"
    )

    predictions = pd.read_csv(
        prediction_path
    )

    y_true = []
    y_pred = []

    for event in EVENT_COLUMNS:

        y_true.append(
            predictions[
                f"true_{event}"
            ].values
        )

        y_pred.append(
            predictions[
                f"pred_{event}"
            ].values
        )

    y_true = np.column_stack(
        y_true
    )

    y_pred = np.column_stack(
        y_pred
    )

    # --------------------------------------------------------
    # MICRO F1
    # --------------------------------------------------------

    micro_f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # MACRO F1
    # --------------------------------------------------------

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # WEIGHTED F1
    # --------------------------------------------------------

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    # --------------------------------------------------------
    # EXACT MATCH
    # --------------------------------------------------------

    exact_match = np.mean(
        np.all(
            y_true == y_pred,
            axis=1,
        )
    )

    # --------------------------------------------------------
    # HAMMING ACCURACY
    # --------------------------------------------------------

    hamming_accuracy = (
        np.mean(
            y_true == y_pred
        )
    )

    # --------------------------------------------------------
    # PER EVENT
    # --------------------------------------------------------

    per_event = {}

    for index, event in enumerate(
        EVENT_COLUMNS
    ):

        per_event[event] = {
            "precision": float(
                precision_score(
                    y_true[:, index],
                    y_pred[:, index],
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true[:, index],
                    y_pred[:, index],
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_true[:, index],
                    y_pred[:, index],
                    zero_division=0,
                )
            ),
        }

    metrics = {
        "version": "v5",
        "feature_policy": "audio_only",

        "micro_f1": float(
            micro_f1
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "exact_match_accuracy": float(
            exact_match
        ),

        "hamming_accuracy": float(
            hamming_accuracy
        ),

        "per_event": per_event,
    }

    save_json(
        RESULTS_DIR
        / "stutter_v5_multilabel_report.json",
        metrics,
    )

    print(
        f"Micro F1            : "
        f"{micro_f1:.4f}"
    )

    print(
        f"Macro F1            : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1         : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Exact-match accuracy: "
        f"{exact_match:.4f}"
    )

    print(
        f"Hamming accuracy    : "
        f"{hamming_accuracy:.4f}"
    )

    return metrics


# ============================================================
# FINAL SUMMARY
# ============================================================

def save_final_summary(
    detector,
    event_results,
    multilabel_metrics,
    data,
    feature_names,
    train_idx,
    val_idx,
    test_idx,
):

    banner(
        "[9/9] Saving final V5 summary"
    )

    summary = {

        "version": "v5",

        "feature_cache": str(
            FEATURE_CACHE_PATH
        ),

        "feature_rows": int(
            len(data)
        ),

        "feature_count": int(
            len(feature_names)
        ),

        "feature_policy": "audio_only",

        "events": EVENT_COLUMNS,

        "detector": {
            "model": "selected_by_validation",

            "threshold": float(
                detector["threshold"]
            ),

            "test_macro_f1": float(
                detector["test_f1"]
            ),
        },

        "event_results": event_results,

        "multilabel": multilabel_metrics,

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

        "leakage_policy": {

            "audio_only_features": True,

            "annotation_columns_excluded": True,

            "episode_safe": True,

            "test_used_for_selection": False,

            "thresholds_selected_on_validation": True,
        },
    }

    path = (
        RESULTS_DIR
        / "stutter_v5_summary.json"
    )

    save_json(
        path,
        summary,
    )

    print(
        f"[SAVED] {path}"
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    banner(
        "ORATORIQ STUTTER V5"
    )

    print(
        "Clean audio-only multi-label fluency pipeline"
    )

    print()
    print(
        "V5 removes annotation/label feature leakage."
    )

    print(
        "Existing V2 feature cache will be reused."
    )

    # --------------------------------------------------------
    # 1
    # --------------------------------------------------------

    metadata = load_metadata()

    # --------------------------------------------------------
    # 2
    # --------------------------------------------------------

    features = load_features()

    # --------------------------------------------------------
    # 3
    # --------------------------------------------------------

    data = match_data(
        metadata,
        features,
    )

    # --------------------------------------------------------
    # 4
    # --------------------------------------------------------

    X, feature_names = prepare_features(
        data
    )

    # --------------------------------------------------------
    # 5
    # --------------------------------------------------------

    (
        train_idx,
        val_idx,
        test_idx,
    ) = create_split(
        data
    )

    X_train = X.iloc[
        train_idx
    ]

    X_val = X.iloc[
        val_idx
    ]

    X_test = X.iloc[
        test_idx
    ]

    data_train = data.iloc[
        train_idx
    ]

    data_val = data.iloc[
        val_idx
    ]

    data_test = data.iloc[
        test_idx
    ]

    # --------------------------------------------------------
    # 6
    # DETECTOR
    # --------------------------------------------------------

    detector = train_detector(
        X_train,
        X_val,
        X_test,
        data_train,
        data_val,
        data_test,
    )

    # --------------------------------------------------------
    # 7
    # EVENTS
    # --------------------------------------------------------

    event_results = train_events(
        X_train,
        X_val,
        X_test,
        data_train,
        data_val,
        data_test,
    )

    prediction_path = (
        RESULTS_DIR
        / "stutter_v5_multilabel_predictions.csv"
    )

    # --------------------------------------------------------
    # 8
    # METRICS
    # --------------------------------------------------------

    multilabel_metrics = (
        calculate_multilabel_metrics(
            event_results,
            prediction_path,
        )
    )

    # --------------------------------------------------------
    # 9
    # SUMMARY
    # --------------------------------------------------------

    save_final_summary(
        detector,
        event_results,
        multilabel_metrics,
        data,
        feature_names,
        train_idx,
        val_idx,
        test_idx,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    banner(
        "ORATORIQ STUTTER V5 COMPLETE"
    )

    print(
        f"Feature rows : {len(data):,}"
    )

    print(
        f"Features     : {len(feature_names):,}"
    )

    print()
    print(
        "FEATURE POLICY"
    )

    print(
        "  Audio-only features : PASS"
    )

    print(
        "  Annotation leakage  : NONE"
    )

    print()
    print(
        "DETECTOR"
    )

    print(
        f"  Test macro F1 : "
        f"{detector['test_f1']:.4f}"
    )

    print()
    print(
        "EVENTS"
    )

    for event in EVENT_COLUMNS:

        print(
            f"  {event:<16}: "
            f"{event_results[event]['test_f1']:.4f}"
        )

    print()
    print(
        "MULTI-LABEL"
    )

    print(
        f"  Macro F1             : "
        f"{multilabel_metrics['macro_f1']:.4f}"
    )

    print(
        f"  Micro F1             : "
        f"{multilabel_metrics['micro_f1']:.4f}"
    )

    print(
        f"  Exact-match accuracy : "
        f"{multilabel_metrics['exact_match_accuracy']:.4f}"
    )

    print(
        f"  Hamming accuracy     : "
        f"{multilabel_metrics['hamming_accuracy']:.4f}"
    )

    print()
    print(
        "Permanent models saved to:"
    )

    print(
        MODELS_DIR
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULTS_DIR
    )

    print()
    print(
        f"Total runtime: "
        f"{elapsed(total_start) / 60:.2f} minutes"
    )

    print()
    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()