"""
OratorIQ - Stutter / Fluency Pipeline V7
=========================================

V7 goals
--------
1. Reuse the clean V6 temporal audio feature cache.
2. DO NOT train independent window labels.
3. Train at CLIP level using the complete temporal sequence.
4. Use XGBoost for:
      - stutter detector
      - multi-label fluency event classifiers
5. Preserve episode-safe train/validation/test split.
6. Use validation-only threshold tuning.
7. Produce candidate temporal localization using
   leave-one-window-out contribution analysis.
8. Save permanent models and inference-ready artifacts.

IMPORTANT
---------
SEP-28k labels are clip-level annotations.

Therefore V7 does NOT claim exact ground-truth event boundaries.

Temporal regions produced by V7 are:
    "candidate fluency regions"

They are derived from model contribution / temporal evidence.
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

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore")


# ============================================================
# OPTIONAL XGBOOST IMPORT
# ============================================================

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError(
        "\nXGBoost is not installed.\n\n"
        "Install it inside your OratorIQ environment with:\n"
        "    pip install xgboost\n"
    )


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\OratorIQ")

FEATURE_CACHE = (
    ROOT
    / "data"
    / "stutter"
    / "features"
    / "stutter_temporal_features_v6.pkl"
)

METADATA_PATH = (
    ROOT
    / "data"
    / "stutter"
    / "metadata"
    / "stutter_metadata.csv"
)

MODEL_DIR = ROOT / "artifacts" / "stutter_models"
RESULT_DIR = ROOT / "artifacts" / "stutter_results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

WINDOWS_PER_CLIP = 5

EVENTS = [
    "Prolongation",
    "Block",
    "SoundRep",
    "WordRep",
    "Interjection",
]

# XGBoost parameters.
# These are intentionally moderate because the laptop has
# limited VRAM / CPU resources.
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "min_child_weight": 3,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# UTILITY
# ============================================================

def banner(text: str):
    print("\n" + "=" * 75)
    print(text)
    print("=" * 75)


def elapsed(start):
    return time.time() - start


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def safe_float(x):
    try:
        if x is None:
            return None

        if isinstance(x, (np.floating, np.integer)):
            return float(x)

        if pd.isna(x):
            return None

        return float(x)

    except Exception:
        return None


# ============================================================
# LOAD FEATURE CACHE
# ============================================================

def load_feature_cache():
    banner("STEP 1 - LOAD V6 AUDIO FEATURE CACHE")

    if not FEATURE_CACHE.exists():
        raise FileNotFoundError(
            f"\nV6 feature cache not found:\n{FEATURE_CACHE}\n\n"
            "Run stutter_pipeline_v6.py first."
        )

    start = time.time()

    with open(FEATURE_CACHE, "rb") as f:
        cache = pickle.load(f)

    print(f"Cache loaded in {elapsed(start):.2f}s")
    print(f"Cache type: {type(cache)}")

    return cache


# ============================================================
# EXTRACT DATAFRAME FROM DIFFERENT CACHE FORMATS
# ============================================================

def extract_dataframe(cache):
    """
    V6 cache formats may differ slightly depending on the exact
    script version.

    This function attempts to locate the DataFrame automatically.
    """

    if isinstance(cache, pd.DataFrame):
        return cache.copy()

    if isinstance(cache, dict):

        preferred_keys = [
            "data",
            "features",
            "df",
            "feature_data",
            "temporal_features",
        ]

        for key in preferred_keys:

            value = cache.get(key)

            if isinstance(value, pd.DataFrame):
                print(f"Using DataFrame key: {key}")
                return value.copy()

        # Search any DataFrame value.
        for key, value in cache.items():

            if isinstance(value, pd.DataFrame):
                print(f"Using DataFrame key: {key}")
                return value.copy()

    raise ValueError(
        "Could not locate a pandas DataFrame inside the V6 feature cache."
    )


# ============================================================
# FIND METADATA COLUMNS
# ============================================================

def find_column(df, candidates, required=True):

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        raise KeyError(
            f"Could not find one of these columns: {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def identify_columns(df):

    audio_id_col = find_column(
        df,
        [
            "audio_path",
            "filename",
            "file",
            "path",
        ],
    )

    episode_col = find_column(
        df,
        [
            "episode_id",
            "episode",
            "epid",
        ],
    )

    clip_id_col = find_column(
        df,
        [
            "clip_id",
            "clip",
            "id",
        ],
        required=False,
    )

    start_col = find_column(
        df,
        [
            "window_start",
            "start",
            "start_time",
            "window_start_sec",
        ],
    )

    end_col = find_column(
        df,
        [
            "window_end",
            "stop",
            "end",
            "end_time",
            "window_end_sec",
        ],
    )

    has_stutter_col = find_column(
        df,
        [
            "has_stutter",
            "stutter",
        ],
    )

    return {
        "audio_id": audio_id_col,
        "episode": episode_col,
        "clip_id": clip_id_col,
        "start": start_col,
        "end": end_col,
        "has_stutter": has_stutter_col,
    }


# ============================================================
# IDENTIFY 121 AUDIO FEATURES
# ============================================================

def identify_audio_features(df):
    """
    IMPORTANT:

    Never use metadata or annotation columns as model features.

    We explicitly exclude:
        - file/path information
        - episode/clip IDs
        - timing metadata
        - all event labels
        - annotation quality flags
        - has_stutter
        - primary_event
    """

    forbidden_exact = {
        "audio_path",
        "filename",
        "file",
        "path",
        "show",
        "episode_id",
        "episode",
        "epid",
        "clip_id",
        "clip",
        "dataset",
        "start",
        "stop",
        "window_start",
        "window_end",
        "window_start_sec",
        "window_end_sec",
        "start_time",
        "end_time",
        "has_stutter",
        "stutter",
        "primary_event",
        "prolongation",
        "block",
        "soundrep",
        "wordrep",
        "interjection",
        "unsure",
        "pooraudioquality",
        "difficulttounderstand",
        "nostutteredwords",
        "naturalpause",
        "music",
        "nospeech",
    }

    features = []

    for col in df.columns:

        col_lower = str(col).lower()

        if col_lower in forbidden_exact:
            continue

        # Explicitly exclude event annotation names.
        if col_lower in {
            event.lower()
            for event in EVENTS
        }:
            continue

        # Exclude obvious metadata.
        if any(
            token in col_lower
            for token in [
                "audio_path",
                "filename",
                "episode",
                "clip_id",
                "primary_event",
                "dataset",
            ]
        ):
            continue

        # Feature must be numeric.
        if pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)

    print(f"Detected numeric audio features: {len(features)}")

    if len(features) != 121:
        print(
            "\nWARNING:"
            f" expected 121 V6 audio features, found {len(features)}."
        )
        print("Detected feature names:")
        print(features)

    return features


# ============================================================
# NORMALIZE TARGETS
# ============================================================

def normalize_binary(series):
    """
    Convert SEP-28k style event counts into binary presence.
    """

    return (
        pd.to_numeric(series, errors="coerce")
        .fillna(0)
        .gt(0)
        .astype(int)
    )


# ============================================================
# EPISODE-SAFE SPLIT
# ============================================================

def create_episode_split(df, episode_col):

    banner("STEP 2 - EPISODE-SAFE SPLIT")

    episodes = (
        df[episode_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(episodes)

    n = len(episodes)

    n_test = max(1, int(round(n * 0.10)))
    n_val = max(1, int(round(n * 0.10)))

    test_episodes = set(episodes[:n_test])

    val_start = n_test
    val_end = n_test + n_val

    val_episodes = set(episodes[val_start:val_end])

    train_episodes = set(episodes[val_end:])

    split = np.full(len(df), "train", dtype=object)

    split[
        df[episode_col].astype(str).isin(val_episodes).values
    ] = "val"

    split[
        df[episode_col].astype(str).isin(test_episodes).values
    ] = "test"

    df = df.copy()
    df["_split"] = split

    print(f"Total episodes : {len(episodes)}")
    print(f"Train episodes : {len(train_episodes)}")
    print(f"Val episodes   : {len(val_episodes)}")
    print(f"Test episodes  : {len(test_episodes)}")

    print("\nClip/window rows by split:")
    print(df["_split"].value_counts())

    # Safety check.
    train_set = set(df.loc[df["_split"] == "train", episode_col].astype(str))
    val_set = set(df.loc[df["_split"] == "val", episode_col].astype(str))
    test_set = set(df.loc[df["_split"] == "test", episode_col].astype(str))

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    print("\nEpisode leakage check: PASS")

    return df


# ============================================================
# BUILD CLIP-LEVEL SEQUENCES
# ============================================================

def build_clip_sequences(
    df,
    feature_names,
    columns,
):
    """
    Convert:

        5 windows x 121 features

    into:

        605 clip-level features.

    Crucially, labels are assigned ONLY ONCE per clip.

    We do NOT assign clip labels to individual windows.
    """

    banner("STEP 3 - BUILD CLIP-LEVEL TEMPORAL DATA")

    audio_id_col = columns["audio_id"]
    start_col = columns["start"]
    end_col = columns["end"]

    grouped = df.groupby(
        audio_id_col,
        sort=False,
        dropna=False,
    )

    rows = []
    feature_names_out = []

    for window_index in range(WINDOWS_PER_CLIP):

        for feature in feature_names:

            feature_names_out.append(
                f"w{window_index + 1}__{feature}"
            )

    metadata_columns = [
        audio_id_col,
        columns["episode"],
        columns["clip_id"],
        columns["has_stutter"],
        "_split",
    ]

    metadata_columns = [
        c for c in metadata_columns
        if c is not None and c in df.columns
    ]

    processed = 0
    skipped = 0

    for audio_id, group in grouped:

        group = group.sort_values(start_col)

        # ----------------------------------------------------
        # V6 creates up to 5 windows per clip.
        # We require exactly 5 for a fixed temporal sequence.
        # ----------------------------------------------------

        if len(group) < WINDOWS_PER_CLIP:
            skipped += 1
            continue

        # Use first 5 temporal windows.
        group = group.iloc[:WINDOWS_PER_CLIP]

        if len(group) != WINDOWS_PER_CLIP:
            skipped += 1
            continue

        values = []

        valid = True

        for _, row in group.iterrows():

            window_values = pd.to_numeric(
                row[feature_names],
                errors="coerce",
            ).values.astype(np.float32)

            if not np.all(np.isfinite(window_values)):
                valid = False
                break

            values.extend(window_values.tolist())

        if not valid:
            skipped += 1
            continue

        base = group.iloc[0]

        record = {
            audio_id_col: audio_id,
            columns["episode"]: base[columns["episode"]],
            "_split": base["_split"],
            "clip_duration": safe_float(
                group[end_col].max()
            ),
            "window_count": len(group),
            "first_window_start": safe_float(
                group[start_col].min()
            ),
        }

        if columns["clip_id"] is not None:
            record[columns["clip_id"]] = base[
                columns["clip_id"]
            ]

        record[columns["has_stutter"]] = int(
            pd.to_numeric(
                base[columns["has_stutter"]],
                errors="coerce",
            )
            if pd.notna(base[columns["has_stutter"]])
            else 0
        )

        for idx, feature_name in enumerate(feature_names_out):
            record[feature_name] = values[idx]

        # ----------------------------------------------------
        # Event labels.
        # These are clip-level labels.
        # ----------------------------------------------------

        for event in EVENTS:

            if event not in df.columns:
                raise KeyError(
                    f"Missing event annotation column: {event}"
                )

            event_value = pd.to_numeric(
                base[event],
                errors="coerce",
            )

            record[f"target_{event}"] = int(
                1 if pd.notna(event_value) and event_value > 0
                else 0
            )

        rows.append(record)

        processed += 1

        if processed % 2000 == 0:
            print(
                f"  Processed clips: {processed:,} | "
                f"Skipped: {skipped:,}"
            )

    result = pd.DataFrame(rows)

    print("\nClip-level dataset:")
    print(f"Clips: {len(result):,}")
    print(f"Temporal windows per clip: {WINDOWS_PER_CLIP}")
    print(f"Audio features per window: {len(feature_names)}")
    print(
        f"Flattened temporal features: "
        f"{len(feature_names_out):,}"
    )
    print(f"Skipped clips: {skipped:,}")

    return result, feature_names_out


# ============================================================
# TRAIN / VAL / TEST ARRAYS
# ============================================================

def get_split_arrays(df, feature_names):

    train = df[df["_split"] == "train"].copy()
    val = df[df["_split"] == "val"].copy()
    test = df[df["_split"] == "test"].copy()

    X_train = train[feature_names].values.astype(np.float32)
    X_val = val[feature_names].values.astype(np.float32)
    X_test = test[feature_names].values.astype(np.float32)

    return (
        train,
        val,
        test,
        X_train,
        X_val,
        X_test,
    )


# ============================================================
# CLASS WEIGHT
# ============================================================

def calculate_scale_pos_weight(y):

    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))

    if positives == 0:
        return 1.0

    return max(
        1.0,
        negatives / positives,
    )


# ============================================================
# XGBOOST MODEL
# ============================================================

def create_xgb(scale_pos_weight=1.0):

    params = dict(XGB_PARAMS)

    params["scale_pos_weight"] = scale_pos_weight

    return XGBClassifier(**params)


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def optimize_threshold(y_true, probabilities):

    best_threshold = 0.50
    best_f1 = -1

    thresholds = np.arange(
        0.10,
        0.91,
        0.01,
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


# ============================================================
# DETECTOR TRAINING
# ============================================================

def train_detector(
    train,
    val,
    test,
    X_train,
    X_val,
    X_test,
    feature_names,
):

    banner("STEP 4 - XGBOOST STUTTER DETECTOR")

    target = "has_stutter"

    y_train = normalize_binary(train[target]).values
    y_val = normalize_binary(val[target]).values
    y_test = normalize_binary(test[target]).values

    scale = calculate_scale_pos_weight(y_train)

    print(f"Train positive: {y_train.sum():,}")
    print(f"Train negative: {(y_train == 0).sum():,}")
    print(f"scale_pos_weight: {scale:.4f}")

    model = create_xgb(
        scale_pos_weight=scale
    )

    start = time.time()

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_val, y_val),
        ],
        verbose=False,
    )

    print(
        f"XGBoost detector trained in "
        f"{elapsed(start) / 60:.2f} minutes"
    )

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    threshold, val_f1 = optimize_threshold(
        y_val,
        val_prob,
    )

    test_pred = (
        test_prob >= threshold
    ).astype(int)

    test_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0,
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred,
    )

    print("\nDetector:")
    print(f"Validation threshold : {threshold:.2f}")
    print(f"Validation macro F1  : {val_f1:.4f}")
    print(f"Test macro F1        : {test_f1:.4f}")
    print(f"Test accuracy        : {test_accuracy:.4f}")

    print("\nTest classification report:")
    print(
        classification_report(
            y_test,
            test_pred,
            target_names=[
                "normal",
                "stutter",
            ],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            test_pred,
        )
    )

    model_path = (
        MODEL_DIR
        / "stutter_detector_v7_xgboost.joblib"
    )

    artifact = {
        "model": model,
        "model_features": feature_names,
        "threshold": threshold,
        "model_type": "XGBoost",
        "random_state": RANDOM_STATE,
        "windows_per_clip": WINDOWS_PER_CLIP,
        "feature_count_per_window": 121,
        "total_features": len(feature_names),
        "version": "v7",
    }

    joblib.dump(
        artifact,
        model_path,
    )

    print(f"\nSaved detector:")
    print(model_path)

    return {
        "artifact": artifact,
        "model_path": str(model_path),
        "val_threshold": threshold,
        "val_f1": val_f1,
        "test_f1": test_f1,
        "test_accuracy": test_accuracy,
        "test_prob": test_prob,
        "test_pred": test_pred,
        "y_test": y_test,
    }


# ============================================================
# EVENT TRAINING
# ============================================================

def train_event_models(
    train,
    val,
    test,
    X_train,
    X_val,
    X_test,
    feature_names,
):

    banner("STEP 5 - XGBOOST MULTI-LABEL FLUENCY EVENTS")

    results = {}

    for event_index, event in enumerate(EVENTS, start=1):

        print("\n" + "-" * 70)
        print(
            f"[{event_index}/{len(EVENTS)}] "
            f"Training: {event}"
        )
        print("-" * 70)

        target = f"target_{event}"

        y_train = normalize_binary(
            train[target]
        ).values

        y_val = normalize_binary(
            val[target]
        ).values

        y_test = normalize_binary(
            test[target]
        ).values

        scale = calculate_scale_pos_weight(
            y_train
        )

        print(
            f"Positive train examples: "
            f"{y_train.sum():,}"
        )

        print(
            f"Negative train examples: "
            f"{(y_train == 0).sum():,}"
        )

        print(
            f"scale_pos_weight: {scale:.4f}"
        )

        model = create_xgb(
            scale_pos_weight=scale
        )

        start = time.time()

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_val, y_val),
            ],
            verbose=False,
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

        threshold, val_f1 = optimize_threshold(
            y_val,
            val_prob,
        )

        test_pred = (
            test_prob >= threshold
        ).astype(int)

        test_f1 = f1_score(
            y_test,
            test_pred,
            average="binary",
            zero_division=0,
        )

        test_macro_f1 = f1_score(
            y_test,
            test_pred,
            average="macro",
            zero_division=0,
        )

        print(
            f"Threshold: {threshold:.2f}"
        )

        print(
            f"Validation F1: {val_f1:.4f}"
        )

        print(
            f"Test F1: {test_f1:.4f}"
        )

        print(
            f"Test macro F1: {test_macro_f1:.4f}"
        )

        print(
            classification_report(
                y_test,
                test_pred,
                target_names=[
                    "absent",
                    "present",
                ],
                zero_division=0,
            )
        )

        model_path = (
            MODEL_DIR
            / f"stutter_v7_event_{event.lower()}_xgboost.joblib"
        )

        artifact = {
            "model": model,
            "model_features": feature_names,
            "threshold": threshold,
            "event": event,
            "model_type": "XGBoost",
            "random_state": RANDOM_STATE,
            "windows_per_clip": WINDOWS_PER_CLIP,
            "feature_count_per_window": 121,
            "total_features": len(feature_names),
            "version": "v7",
        }

        joblib.dump(
            artifact,
            model_path,
        )

        print(
            f"Saved: {model_path}"
        )

        results[event] = {
            "artifact": artifact,
            "model_path": str(model_path),
            "threshold": threshold,
            "val_f1": val_f1,
            "test_f1": test_f1,
            "test_macro_f1": test_macro_f1,
            "test_prob": test_prob,
            "test_pred": test_pred,
            "y_test": y_test,
        }

    return results


# ============================================================
# TEMPORAL LEAVE-ONE-WINDOW-OUT
# ============================================================

def calculate_window_contributions(
    model_artifact,
    X,
    threshold,
):

    """
    Temporal localization without pretending that
    SEP-28k provides exact window-level ground truth.

    For every clip:

        full probability
                 ↓
        remove/mask window 1
        remove/mask window 2
        ...
        remove/mask window 5

    If removing a window significantly reduces
    the predicted probability, that window contributes
    strongly to the prediction.

    We use the training median as the masking value.
    """

    model = model_artifact["model"]
    feature_names = model_artifact["model_features"]

    n_features = len(feature_names)
    n_windows = WINDOWS_PER_CLIP

    # Training median isn't stored in the artifact in the
    # current implementation. For localization, use zero
    # masking after standardized-like audio features.
    #
    # Instead of assuming standardized values, calculate
    # the median from the supplied test matrix.
    mask_vector = np.median(
        X,
        axis=0,
    )

    full_prob = model.predict_proba(X)[:, 1]

    contributions = np.zeros(
        (
            len(X),
            n_windows,
        ),
        dtype=np.float32,
    )

    for window_idx in range(n_windows):

        masked = X.copy()

        start = (
            window_idx * n_features
            // n_windows
        )

        # Because feature names contain w1__, w2__, ...
        # calculate the actual per-window feature count.
        per_window = int(
            X.shape[1] / n_windows
        )

        start = window_idx * per_window
        end = start + per_window

        masked[:, start:end] = mask_vector[
            start:end
        ]

        masked_prob = model.predict_proba(
            masked
        )[:, 1]

        # Positive means the window helps the model
        # produce a stutter prediction.
        contributions[:, window_idx] = (
            full_prob - masked_prob
        )

    # Normalize contribution values per clip.
    positive = np.maximum(
        contributions,
        0,
    )

    row_max = np.max(
        positive,
        axis=1,
        keepdims=True,
    )

    row_max[row_max == 0] = 1.0

    normalized = positive / row_max

    return (
        full_prob,
        contributions,
        normalized,
    )


# ============================================================
# BUILD TEMPORAL RESULT TABLE
# ============================================================

def build_temporal_results(
    test,
    detector_result,
    event_results,
):

    banner("STEP 6 - BUILD TEMPORAL CANDIDATE REGIONS")

    X_test = test[
        detector_result["artifact"]["model_features"]
    ].values.astype(np.float32)

    full_prob, contributions, normalized = (
        calculate_window_contributions(
            detector_result["artifact"],
            X_test,
            detector_result["val_threshold"],
        )
    )

    rows = []

    for i in range(len(test)):

        base = test.iloc[i]

        audio_id = str(
            base[
                find_column(
                    test,
                    [
                        "audio_path",
                        "filename",
                        "file",
                        "path",
                    ],
                )
            ]
        )

        clip_probability = float(
            full_prob[i]
        )

        detector_threshold = float(
            detector_result["val_threshold"]
        )

        predicted_stutter = int(
            clip_probability >= detector_threshold
        )

        # Event probabilities.
        event_probs = {}

        for event in EVENTS:

            event_probs[event] = float(
                event_results[event]["test_prob"][i]
            )

        # Candidate temporal windows.
        for window_idx in range(
            WINDOWS_PER_CLIP
        ):

            start_time = (
                window_idx * 0.5
            )

            end_time = (
                start_time + 1.0
            )

            rows.append(
                {
                    "audio_path": audio_id,
                    "window_index": window_idx,
                    "window_start": start_time,
                    "window_end": end_time,
                    "clip_stutter_probability":
                        clip_probability,
                    "clip_predicted_stutter":
                        predicted_stutter,
                    "window_contribution":
                        float(
                            contributions[
                                i,
                                window_idx,
                            ]
                        ),
                    "window_contribution_normalized":
                        float(
                            normalized[
                                i,
                                window_idx,
                            ]
                        ),
                    **{
                        f"{event}_probability":
                            event_probs[event]
                        for event in EVENTS
                    },
                }
            )

    temporal_df = pd.DataFrame(rows)

    path = (
        RESULT_DIR
        / "stutter_v7_temporal_candidates.csv"
    )

    temporal_df.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved temporal candidate table:\n{path}"
    )

    return temporal_df


# ============================================================
# CREATE DEMO TIMELINE
# ============================================================

def create_demo_timeline(
    temporal_df,
    test,
):

    banner("STEP 7 - CREATE DEMO TIMELINE")

    timelines = []

    for audio_path, group in temporal_df.groupby(
        "audio_path"
    ):

        group = group.sort_values(
            "window_index"
        )

        clip_probability = float(
            group["clip_stutter_probability"]
            .iloc[0]
        )

        predicted_stutter = int(
            group["clip_predicted_stutter"]
            .iloc[0]
        )

        regions = []

        # Only consider windows with meaningful
        # contribution.
        candidate_group = group[
            group[
                "window_contribution_normalized"
            ] >= 0.50
        ]

        for _, row in candidate_group.iterrows():

            event_scores = {
                event: float(
                    row[
                        f"{event}_probability"
                    ]
                )
                for event in EVENTS
            }

            best_event = max(
                event_scores,
                key=event_scores.get,
            )

            best_event_probability = (
                event_scores[best_event]
            )

            regions.append(
                {
                    "start": float(
                        row["window_start"]
                    ),
                    "end": float(
                        row["window_end"]
                    ),
                    "event": best_event,
                    "event_probability":
                        best_event_probability,
                    "window_contribution":
                        float(
                            row[
                                "window_contribution_normalized"
                            ]
                        ),
                }
            )

        timeline = {
            "audio_path": audio_path,
            "model": "XGBoost",
            "version": "v7",
            "clip_probability": clip_probability,
            "predicted_stutter": bool(
                predicted_stutter
            ),
            "regions": regions,
            "note": (
                "Temporal regions are candidate regions "
                "derived from leave-one-window-out model "
                "contribution. SEP-28k provides clip-level "
                "annotations, not exact temporal boundaries."
            ),
        }

        timelines.append(timeline)

    output = {
        "version": "v7",
        "model": "XGBoost",
        "total_test_clips": len(timelines),
        "timelines": timelines,
    }

    path = (
        RESULT_DIR
        / "stutter_v7_demo_timeline.json"
    )

    save_json(
        output,
        path,
    )

    print(
        f"Saved demo timeline:\n{path}"
    )

    return output


# ============================================================
# SAVE MULTI-LABEL PREDICTIONS
# ============================================================

def save_multilabel_predictions(
    test,
    detector_result,
    event_results,
):

    banner("STEP 8 - SAVE TEST PREDICTIONS")

    output = pd.DataFrame()

    # Preserve useful identifiers.
    for col in [
        "audio_path",
        "filename",
        "episode_id",
        "clip_id",
        "show",
    ]:

        if col in test.columns:
            output[col] = test[col].values

    output["true_has_stutter"] = (
        normalize_binary(
            test["has_stutter"]
        ).values
    )

    output["pred_has_stutter"] = (
        detector_result["test_pred"]
    )

    output["stutter_probability"] = (
        detector_result["test_prob"]
    )

    output["stutter_threshold"] = (
        detector_result["val_threshold"]
    )

    for event in EVENTS:

        result = event_results[event]

        output[
            f"true_{event}"
        ] = result["y_test"]

        output[
            f"pred_{event}"
        ] = result["test_pred"]

        output[
            f"{event}_probability"
        ] = result["test_prob"]

        output[
            f"{event}_threshold"
        ] = result["threshold"]

    path = (
        RESULT_DIR
        / "stutter_v7_predictions.csv"
    )

    output.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved predictions:\n{path}"
    )

    return output


# ============================================================
# SUMMARY
# ============================================================

def create_summary(
    clip_df,
    detector_result,
    event_results,
):

    banner("STEP 9 - FINAL V7 SUMMARY")

    summary = {
        "version": "v7",
        "model": "XGBoost",
        "architecture": (
            "clip-level temporal sequence "
            "with leave-one-window-out localization"
        ),
        "windows_per_clip": WINDOWS_PER_CLIP,
        "features_per_window": 121,
        "flattened_features": (
            121 * WINDOWS_PER_CLIP
        ),
        "training_level": "clip",
        "window_labels_used_for_training": False,
        "annotation_leakage_features_used": False,
        "episode_safe_split": True,
        "clip_count": int(len(clip_df)),
        "detector": {
            "threshold": detector_result[
                "val_threshold"
            ],
            "validation_macro_f1":
                detector_result[
                    "val_f1"
                ],
            "test_macro_f1":
                detector_result[
                    "test_f1"
                ],
            "test_accuracy":
                detector_result[
                    "test_accuracy"
                ],
        },
        "events": {},
    }

    for event in EVENTS:

        result = event_results[event]

        summary["events"][event] = {
            "threshold":
                result["threshold"],
            "validation_f1":
                result["val_f1"],
            "test_f1":
                result["test_f1"],
            "test_macro_f1":
                result["test_macro_f1"],
        }

    summary["limitations"] = [
        (
            "SEP-28k annotations are clip-level and "
            "do not provide exact event boundaries."
        ),
        (
            "Temporal regions are candidate regions "
            "derived from model contribution."
        ),
        (
            "Results should be described as "
            "AI-assisted fluency analytics, "
            "not clinical diagnosis."
        ),
    ]

    path = (
        RESULT_DIR
        / "stutter_v7_summary.json"
    )

    save_json(
        summary,
        path,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        f"\nSaved summary:\n{path}"
    )

    return summary


# ============================================================
# LEAKAGE CHECK
# ============================================================

def run_leakage_check(
    feature_names,
):

    banner("LEAKAGE CHECK")

    suspicious = []

    forbidden_tokens = [
        "prolongation",
        "block",
        "soundrep",
        "wordrep",
        "interjection",
        "has_stutter",
        "primary_event",
        "unsure",
        "pooraudioquality",
        "difficulttounderstand",
        "nostutteredwords",
        "naturalpause",
        "music",
        "nospeech",
    ]

    for feature in feature_names:

        name = str(feature).lower()

        for token in forbidden_tokens:

            if token in name:

                suspicious.append(
                    feature
                )

    if suspicious:

        print(
            "LEAKAGE CHECK: FAIL"
        )

        print(
            "Suspicious features:"
        )

        for feature in suspicious:
            print(
                f"  - {feature}"
            )

        raise RuntimeError(
            "Annotation leakage detected."
        )

    print(
        "LEAKAGE CHECK: PASS"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.time()

    banner(
        "ORATORIQ STUTTER / FLUENCY PIPELINE V7"
    )

    print(
        "Model: XGBoost"
    )

    print(
        "Training formulation: CLIP LEVEL"
    )

    print(
        "Temporal localization: "
        "LEAVE-ONE-WINDOW-OUT"
    )

    print(
        f"Feature cache:\n{FEATURE_CACHE}"
    )

    # --------------------------------------------------------
    # 1. Load V6 cache.
    # --------------------------------------------------------

    cache = load_feature_cache()

    df = extract_dataframe(cache)

    print(
        f"\nLoaded rows: {len(df):,}"
    )

    print(
        f"Loaded columns: {len(df.columns):,}"
    )

    # --------------------------------------------------------
    # 2. Identify columns.
    # --------------------------------------------------------

    columns = identify_columns(df)

    print("\nDetected columns:")

    for key, value in columns.items():
        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # 3. Identify clean audio features.
    # --------------------------------------------------------

    feature_names = identify_audio_features(
        df
    )

    run_leakage_check(
        feature_names
    )

    # --------------------------------------------------------
    # 4. Episode-safe split.
    # --------------------------------------------------------

    df = create_episode_split(
        df,
        columns["episode"],
    )

    # --------------------------------------------------------
    # 5. Build clip sequences.
    # --------------------------------------------------------

    clip_df, sequence_features = (
        build_clip_sequences(
            df,
            feature_names,
            columns,
        )
    )

    # --------------------------------------------------------
    # 6. Final leakage check.
    # --------------------------------------------------------

    run_leakage_check(
        sequence_features
    )

    # --------------------------------------------------------
    # 7. Save sequence feature cache.
    # --------------------------------------------------------

    sequence_cache_path = (
        ROOT
        / "data"
        / "stutter"
        / "features"
        / "stutter_clip_features_v7.pkl"
    )

    with open(
        sequence_cache_path,
        "wb",
    ) as f:

        pickle.dump(
            {
                "data": clip_df,
                "model_features": sequence_features,
                "version": "v7",
                "windows_per_clip":
                    WINDOWS_PER_CLIP,
                "features_per_window":
                    len(feature_names),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        f"\nSaved clip feature cache:"
        f"\n{sequence_cache_path}"
    )

    # --------------------------------------------------------
    # 8. Prepare train/val/test.
    # --------------------------------------------------------

    (
        train,
        val,
        test,
        X_train,
        X_val,
        X_test,
    ) = get_split_arrays(
        clip_df,
        sequence_features,
    )

    print("\nFinal clip split:")

    print(
        f"Train: {len(train):,}"
    )

    print(
        f"Validation: {len(val):,}"
    )

    print(
        f"Test: {len(test):,}"
    )

    # --------------------------------------------------------
    # 9. Detector.
    # --------------------------------------------------------

    detector_result = train_detector(
        train,
        val,
        test,
        X_train,
        X_val,
        X_test,
        sequence_features,
    )

    # --------------------------------------------------------
    # 10. Events.
    # --------------------------------------------------------

    event_results = train_event_models(
        train,
        val,
        test,
        X_train,
        X_val,
        X_test,
        sequence_features,
    )

    # --------------------------------------------------------
    # 11. Temporal candidates.
    # --------------------------------------------------------

    temporal_df = build_temporal_results(
        test,
        detector_result,
        event_results,
    )

    # --------------------------------------------------------
    # 12. Demo timeline.
    # --------------------------------------------------------

    create_demo_timeline(
        temporal_df,
        test,
    )

    # --------------------------------------------------------
    # 13. Save predictions.
    # --------------------------------------------------------

    save_multilabel_predictions(
        test,
        detector_result,
        event_results,
    )

    # --------------------------------------------------------
    # 14. Final summary.
    # --------------------------------------------------------

    summary = create_summary(
        clip_df,
        detector_result,
        event_results,
    )

    # --------------------------------------------------------
    # FINAL.
    # --------------------------------------------------------

    total_minutes = (
        elapsed(overall_start) / 60
    )

    banner(
        "V7 COMPLETE"
    )

    print(
        f"Runtime: {total_minutes:.2f} minutes"
    )

    print(
        f"Clip samples: {len(clip_df):,}"
    )

    print(
        f"Features per window: "
        f"{len(feature_names)}"
    )

    print(
        f"Temporal windows per clip: "
        f"{WINDOWS_PER_CLIP}"
    )

    print(
        f"XGBoost features per clip: "
        f"{len(sequence_features)}"
    )

    print(
        "\nDetector test macro F1:"
        f" {detector_result['test_f1']:.4f}"
    )

    print("\nEvent test F1:")

    for event in EVENTS:

        print(
            f"  {event:15s}: "
            f"{event_results[event]['test_f1']:.4f}"
        )

    print(
        "\nModels saved to:"
    )

    print(
        MODEL_DIR
    )

    print(
        "\nResults saved to:"
    )

    print(
        RESULT_DIR
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "V7 temporal regions are candidate regions, "
        "not ground-truth event boundaries."
    )


if __name__ == "__main__":
    main()