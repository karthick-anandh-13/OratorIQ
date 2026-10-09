"""
===============================================================
ORATORIQ — MASTER TEMPORAL SPEECH ANALYTICS PIPELINE
===============================================================

Pipeline:

1. Dataset validation
2. Actual audio-duration validation
3. Temporal window generation
4. Acoustic feature extraction
5. Speaker normalization
6. Speaker-safe train/validation/test split
7. Binary temporal flaw detection
8. Flaw-type classification
9. Severity classification
10. Model benchmarking
11. Best-model selection
12. Temporal prediction export
13. Checkpointing / resume support

Designed for:
    D:\OratorIQ

Expected metadata:
    data\contrastive\metadata.jsonl

===============================================================
"""

from __future__ import annotations

import os
import json
import time
import warnings
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ===============================================================
# CONFIGURATION
# ===============================================================

ROOT = Path(r"D:\OratorIQ")

METADATA = ROOT / "data" / "contrastive" / "metadata.jsonl"

ARTIFACTS = ROOT / "artifacts"
CACHE_DIR = ARTIFACTS / "feature_cache"
MODEL_DIR = ARTIFACTS / "models"
RESULT_DIR = ARTIFACTS / "results"

ARTIFACTS.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


# Audio processing
TARGET_SR = 16000

# Temporal windows
WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.5

# Minimum percentage of window that must overlap flaw interval
FLAW_OVERLAP_THRESHOLD = 0.30

# Boundary windows are ignored when overlap is ambiguous
BOUNDARY_LOW = 0.05
BOUNDARY_HIGH = 0.30

# Number of MFCC coefficients
N_MFCC = 13

# Resume processing
USE_CACHE = True

# Save checkpoint every N recordings
CHECKPOINT_EVERY = 100

# Random seed
RANDOM_STATE = 42


# ===============================================================
# UTILITY FUNCTIONS
# ===============================================================

def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def elapsed_string(seconds):
    seconds = int(seconds)

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m}m {s}s"

    if m > 0:
        return f"{m}m {s}s"

    return f"{s}s"


def progress(current, total, start_time, prefix=""):
    elapsed = time.time() - start_time

    ratio = current / max(total, 1)

    width = 25
    filled = int(width * ratio)

    bar = "█" * filled + "░" * (width - filled)

    if current > 0:
        estimated_total = elapsed / ratio
        remaining = max(0, estimated_total - elapsed)
    else:
        remaining = 0

    print(
        f"\r{prefix} "
        f"[{bar}] "
        f"{ratio * 100:6.2f}% "
        f"| {current:,}/{total:,} "
        f"| elapsed {elapsed_string(elapsed)} "
        f"| ETA {elapsed_string(remaining)}",
        end="",
        flush=True,
    )

    if current == total:
        print()


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


# ===============================================================
# DATASET LOADING
# ===============================================================

def load_metadata():

    banner("[1/8] LOADING METADATA")

    if not METADATA.exists():
        raise FileNotFoundError(
            f"Metadata not found:\n{METADATA}"
        )

    records = []

    with open(METADATA, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    df = pd.DataFrame(records)

    print(f"Metadata records: {len(df):,}")

    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())

    print("\nFlaw distribution:")
    print(df["flaw_type"].fillna("none").value_counts())

    return df


# ===============================================================
# SPEAKER ID
# ===============================================================

def get_speaker_id(pair_id):

    """
    LibriSpeech IDs normally look like:

        1272-128104-0000

    First component is the speaker ID.

    If an unexpected format occurs, fall back to pair_id.
    """

    if not pair_id:
        return "unknown"

    parts = str(pair_id).split("-")

    if len(parts) >= 3:
        return parts[0]

    return str(pair_id)


# ===============================================================
# AUDIO PATH
# ===============================================================

def resolve_audio_path(audio_path):

    path = Path(str(audio_path))

    if path.is_absolute() and path.exists():
        return path

    # Metadata normally contains:
    # data\contrastive\audio\...
    candidate = ROOT / path

    if candidate.exists():
        return candidate

    # Windows path separators can occasionally cause issues
    normalized = str(audio_path).replace("\\", os.sep).replace("/", os.sep)

    candidate = ROOT / normalized

    if candidate.exists():
        return candidate

    return None


# ===============================================================
# AUDIO VALIDATION
# ===============================================================

def validate_audio_dataset(df):

    banner("[2/8] VALIDATING AUDIO DATASET")

    start_time = time.time()

    valid_rows = []
    invalid = []

    total = len(df)

    for i, row in df.iterrows():

        audio_path = resolve_audio_path(row["audio"])

        if audio_path is None:

            invalid.append({
                "id": row["id"],
                "reason": "AUDIO_NOT_FOUND",
                "audio": row["audio"],
            })

            progress(
                i + 1,
                total,
                start_time,
                "Audio validation"
            )

            continue

        try:

            info = sf.info(str(audio_path))

            actual_duration = float(info.duration)

            if actual_duration <= 0:

                invalid.append({
                    "id": row["id"],
                    "reason": "INVALID_DURATION",
                    "audio": str(audio_path),
                })

                continue

            row = row.copy()

            row["audio_resolved"] = str(audio_path)

            row["actual_duration"] = actual_duration

            # Re-check temporal metadata against ACTUAL audio
            if row["label"] != "good":

                start = safe_float(row.get("flaw_start"))
                end = safe_float(row.get("flaw_end"))

                if start < 0:
                    start = 0

                if end > actual_duration:

                    # Do not silently destroy metadata.
                    # Clip only to actual audio boundary.

                    end = actual_duration

                if end <= start:

                    invalid.append({
                        "id": row["id"],
                        "reason": "INVALID_TEMPORAL_RANGE",
                        "audio": str(audio_path),
                    })

                    continue

                row["flaw_start_actual"] = start
                row["flaw_end_actual"] = end

            else:

                row["flaw_start_actual"] = np.nan
                row["flaw_end_actual"] = np.nan

            valid_rows.append(row)

        except Exception as e:

            invalid.append({
                "id": row["id"],
                "reason": f"AUDIO_READ_ERROR: {str(e)}",
                "audio": str(audio_path),
            })

        progress(
            i + 1,
            total,
            start_time,
            "Audio validation"
        )

    valid_df = pd.DataFrame(valid_rows)

    print(f"\n\nValid audio records: {len(valid_df):,}")
    print(f"Invalid records:     {len(invalid):,}")

    if invalid:

        invalid_path = RESULT_DIR / "invalid_audio_records.json"

        with open(invalid_path, "w", encoding="utf-8") as f:
            json.dump(invalid, f, indent=2)

        print(f"Invalid records saved to:")
        print(invalid_path)

    return valid_df


# ===============================================================
# TEMPORAL LABELING
# ===============================================================

def determine_window_label(
    window_start,
    window_end,
    flaw_start,
    flaw_end,
    is_good
):

    if is_good:
        return "normal"

    overlap_start = max(window_start, flaw_start)
    overlap_end = min(window_end, flaw_end)

    overlap = max(0.0, overlap_end - overlap_start)

    window_length = window_end - window_start

    overlap_ratio = overlap / max(window_length, 1e-6)

    if overlap_ratio >= FLAW_OVERLAP_THRESHOLD:

        return "flawed"

    if overlap_ratio > BOUNDARY_LOW:

        return "boundary"

    return "normal"


# ===============================================================
# FEATURE EXTRACTION
# ===============================================================

def extract_features(
    audio,
    sr,
    start_sample,
    end_sample,
    transcript,
    total_duration
):

    y = audio[start_sample:end_sample]

    if len(y) < int(sr * 0.15):

        return None

    # -----------------------------------------------------------
    # Basic signal
    # -----------------------------------------------------------

    duration = len(y) / sr

    rms = librosa.feature.rms(
        y=y,
        frame_length=min(1024, len(y)),
        hop_length=256
    )[0]

    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    rms_max = float(np.max(rms)) if len(rms) else 0

    # -----------------------------------------------------------
    # Zero crossing rate
    # -----------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=min(1024, len(y)),
        hop_length=256
    )[0]

    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    # -----------------------------------------------------------
    # MFCC
    # -----------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=min(1024, len(y)),
        hop_length=256
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # -----------------------------------------------------------
    # MFCC delta
    # -----------------------------------------------------------

    if mfcc.shape[1] >= 3:

        delta = librosa.feature.delta(mfcc)

        delta_mean = np.mean(delta, axis=1)
        delta_std = np.std(delta, axis=1)

    else:

        delta_mean = np.zeros(N_MFCC)
        delta_std = np.zeros(N_MFCC)

    # -----------------------------------------------------------
    # Spectral features
    # -----------------------------------------------------------

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        hop_length=256
    )[0]

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        hop_length=256
    )[0]

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        hop_length=256
    )[0]

    spectral_flatness = librosa.feature.spectral_flatness(
        y=y,
        hop_length=256
    )[0]

    # -----------------------------------------------------------
    # Pitch / F0
    # -----------------------------------------------------------

    try:

        f0 = librosa.yin(
            y,
            fmin=70,
            fmax=400,
            sr=sr,
            frame_length=1024,
            hop_length=256
        )

        f0 = f0[np.isfinite(f0)]

        if len(f0):

            f0_mean = float(np.mean(f0))
            f0_std = float(np.std(f0))
            f0_min = float(np.percentile(f0, 10))
            f0_max = float(np.percentile(f0, 90))

        else:

            f0_mean = 0
            f0_std = 0
            f0_min = 0
            f0_max = 0

    except Exception:

        f0_mean = 0
        f0_std = 0
        f0_min = 0
        f0_max = 0

    # -----------------------------------------------------------
    # Silence / pause statistics
    # -----------------------------------------------------------

    try:

        intervals = librosa.effects.split(
            y,
            top_db=30
        )

        voiced_samples = sum(
            max(0, end - start)
            for start, end in intervals
        )

        voiced_ratio = voiced_samples / max(len(y), 1)

        silence_ratio = 1.0 - voiced_ratio

    except Exception:

        voiced_ratio = 0
        silence_ratio = 1

    # -----------------------------------------------------------
    # Approximate speech rate
    #
    # This is a recording-level feature repeated for each window.
    # True local word timing requires forced alignment and will be
    # added later if needed.
    # -----------------------------------------------------------

    words = str(transcript).split()

    word_count = len(words)

    global_wpm = (
        word_count / max(total_duration, 0.1)
    ) * 60.0

    # -----------------------------------------------------------
    # Build feature vector
    # -----------------------------------------------------------

    features = {}

    features["duration"] = duration

    features["rms_mean"] = rms_mean
    features["rms_std"] = rms_std
    features["rms_max"] = rms_max

    features["zcr_mean"] = zcr_mean
    features["zcr_std"] = zcr_std

    features["spectral_centroid"] = float(
        np.mean(spectral_centroid)
    )

    features["spectral_bandwidth"] = float(
        np.mean(spectral_bandwidth)
    )

    features["spectral_rolloff"] = float(
        np.mean(spectral_rolloff)
    )

    features["spectral_flatness"] = float(
        np.mean(spectral_flatness)
    )

    features["f0_mean"] = f0_mean
    features["f0_std"] = f0_std
    features["f0_min"] = f0_min
    features["f0_max"] = f0_max

    features["voiced_ratio"] = voiced_ratio
    features["silence_ratio"] = silence_ratio

    features["global_wpm"] = global_wpm

    for i in range(N_MFCC):

        features[f"mfcc_{i+1}_mean"] = float(
            mfcc_mean[i]
        )

        features[f"mfcc_{i+1}_std"] = float(
            mfcc_std[i]
        )

        features[f"delta_{i+1}_mean"] = float(
            delta_mean[i]
        )

        features[f"delta_{i+1}_std"] = float(
            delta_std[i]
        )

    return features


# ===============================================================
# PROCESS ONE RECORDING
# ===============================================================

def process_record(row):

    record_id = row["id"]

    cache_file = CACHE_DIR / f"{record_id}.pkl"

    if USE_CACHE and cache_file.exists():

        try:

            return joblib.load(cache_file)

        except Exception:

            pass

    audio_path = row["audio_resolved"]

    try:

        audio, sr = librosa.load(
            audio_path,
            sr=TARGET_SR,
            mono=True
        )

    except Exception as e:

        print(
            f"\nWARNING: Could not load {audio_path}: {e}"
        )

        return []

    actual_duration = len(audio) / sr

    window_samples = int(WINDOW_SECONDS * sr)
    hop_samples = int(HOP_SECONDS * sr)

    transcript = row.get("transcript", "")

    is_good = row["label"] == "good"

    if is_good:

        flaw_start = None
        flaw_end = None

    else:

        flaw_start = safe_float(
            row.get("flaw_start_actual")
        )

        flaw_end = safe_float(
            row.get("flaw_end_actual")
        )

    windows = []

    start_sample = 0

    while start_sample + window_samples <= len(audio):

        end_sample = start_sample + window_samples

        start_sec = start_sample / sr
        end_sec = end_sample / sr

        if is_good:

            temporal_label = "normal"

        else:

            temporal_label = determine_window_label(
                start_sec,
                end_sec,
                flaw_start,
                flaw_end,
                False
            )

        # Ignore ambiguous boundary windows
        if temporal_label != "boundary":

            features = extract_features(
                audio,
                sr,
                start_sample,
                end_sample,
                transcript,
                actual_duration
            )

            if features is not None:

                features["record_id"] = record_id
                features["pair_id"] = row["pair_id"]
                features["speaker_id"] = get_speaker_id(
                    row["pair_id"]
                )

                features["window_start"] = start_sec
                features["window_end"] = end_sec

                features["temporal_label"] = temporal_label

                features["recording_label"] = row["label"]

                features["severity"] = safe_float(
                    row.get("severity")
                )

                features["flaw_type"] = (
                    row["flaw_type"]
                    if row["flaw_type"]
                    else "none"
                )

                features["transcript"] = transcript

                windows.append(features)

        start_sample += hop_samples

    if USE_CACHE:

        joblib.dump(
            windows,
            cache_file,
            compress=3
        )

    return windows


# ===============================================================
# FEATURE EXTRACTION STAGE
# ===============================================================

def build_temporal_dataset(df):

    banner("[3/8] EXTRACTING TEMPORAL ACOUSTIC FEATURES")

    print(
        f"Window: {WINDOW_SECONDS:.1f}s | "
        f"Hop: {HOP_SECONDS:.1f}s"
    )

    print(
        f"Flaw overlap threshold: "
        f"{FLAW_OVERLAP_THRESHOLD:.0%}"
    )

    print("\nFeatures:")
    print("  MFCC + Delta MFCC")
    print("  F0 / Pitch")
    print("  RMS Energy")
    print("  ZCR")
    print("  Spectral features")
    print("  Silence / Voiced ratio")
    print("  Global speech rate")

    all_windows = []

    start_time = time.time()

    total = len(df)

    for i, (_, row) in enumerate(df.iterrows(), start=1):

        windows = process_record(row)

        all_windows.extend(windows)

        progress(
            i,
            total,
            start_time,
            "Feature extraction"
        )

    feature_df = pd.DataFrame(all_windows)

    checkpoint_path = (
        ARTIFACTS / "temporal_features.pkl"
    )

    joblib.dump(
        feature_df,
        checkpoint_path,
        compress=3
    )

    print(
        f"\n\nGenerated windows: "
        f"{len(feature_df):,}"
    )

    print("\nTemporal label distribution:")
    print(
        feature_df["temporal_label"]
        .value_counts()
    )

    print("\nFeature matrix shape:")
    print(feature_df.shape)

    print(
        f"\nCheckpoint saved:\n"
        f"{checkpoint_path}"
    )

    return feature_df


# ===============================================================
# SPEAKER NORMALIZATION
# ===============================================================

def speaker_normalize(df, feature_columns):

    banner("[4/8] SPEAKER NORMALIZATION")

    df = df.copy()

    numeric_features = [
        c for c in feature_columns
        if c in df.columns
    ]

    print(
        f"Normalizing {len(numeric_features)} acoustic features..."
    )

    # Speaker-level z-normalization.
    #
    # This reduces differences caused by:
    #   - naturally high/low pitch
    #   - loud/quiet speakers
    #   - microphone differences
    #   - individual speaking style

    for feature in numeric_features:

        grouped = df.groupby("speaker_id")[feature]

        mean = grouped.transform("mean")

        std = grouped.transform("std")

        std = std.replace(
            [np.inf, -np.inf, 0],
            np.nan
        ).fillna(1.0)

        df[feature + "_spkz"] = (
            df[feature] - mean
        ) / std

    normalized_columns = [
        c for c in df.columns
        if c.endswith("_spkz")
    ]

    print(
        f"Created {len(normalized_columns)} "
        f"speaker-normalized features."
    )

    return df, normalized_columns


# ===============================================================
# GROUP SPLIT
# ===============================================================

def create_splits(df):

    banner("[5/8] CREATING SPEAKER-SAFE SPLITS")

    # One speaker must not appear in multiple sets.

    groups = df["speaker_id"].values

    splitter1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter1.split(
            df,
            groups=groups
        )
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE
    )

    groups_train = train_df["speaker_id"].values

    train2_idx, val_idx = next(
        splitter2.split(
            train_df,
            groups=groups_train
        )
    )

    final_train = train_df.iloc[train2_idx].copy()
    val_df = train_df.iloc[val_idx].copy()

    print(
        f"Train:      {len(final_train):,} windows"
    )

    print(
        f"Validation: {len(val_df):,} windows"
    )

    print(
        f"Test:       {len(test_df):,} windows"
    )

    print()

    print(
        f"Train speakers: "
        f"{final_train.speaker_id.nunique()}"
    )

    print(
        f"Validation speakers: "
        f"{val_df.speaker_id.nunique()}"
    )

    print(
        f"Test speakers: "
        f"{test_df.speaker_id.nunique()}"
    )

    return final_train, val_df, test_df


# ===============================================================
# MODEL FACTORY
# ===============================================================

def get_models():

    models = {

        "ExtraTrees": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),

            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=350,
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
                SimpleImputer(strategy="median")
            ),

            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
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
                SimpleImputer(strategy="median")
            ),

            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.06,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE
                )
            )
        ])
    }

    # XGBoost is optional.
    try:

        from xgboost import XGBClassifier

        models["XGBoost"] = Pipeline([

            (
                "imputer",
                SimpleImputer(strategy="median")
            ),

            (
                "model",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=7,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=RANDOM_STATE
                )
            )
        ])

    except Exception:

        print(
            "\nXGBoost unavailable. "
            "Continuing with sklearn models."
        )

    return models


# ===============================================================
# MODEL EVALUATION
# ===============================================================

def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):

    print(
        f"\n{'-' * 65}"
    )

    print(
        f"Training {name}..."
    )

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    elapsed = time.time() - start

    val_pred = model.predict(X_val)

    test_pred = model.predict(X_test)

    val_f1 = f1_score(
        y_val,
        val_pred,
        average="macro"
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        average="macro"
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
        f"Training time: {elapsed_string(elapsed)}"
    )

    print(
        f"Validation Accuracy: {val_acc:.4f}"
    )

    print(
        f"Validation Macro F1: {val_f1:.4f}"
    )

    print(
        f"Test Accuracy:       {test_acc:.4f}"
    )

    print(
        f"Test Macro F1:       {test_f1:.4f}"
    )

    return {
        "model": model,
        "val_f1": val_f1,
        "test_f1": test_f1,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "val_predictions": val_pred,
        "test_predictions": test_pred,
    }


# ===============================================================
# PRIMARY BINARY DETECTOR
# ===============================================================

def train_temporal_detector(
    train_df,
    val_df,
    test_df,
    feature_columns
):

    banner("[6/8] TRAINING TEMPORAL FLAW DETECTOR")

    X_train = train_df[feature_columns]
    X_val = val_df[feature_columns]
    X_test = test_df[feature_columns]

    y_train = (
        train_df["temporal_label"] == "flawed"
    ).astype(int)

    y_val = (
        val_df["temporal_label"] == "flawed"
    ).astype(int)

    y_test = (
        test_df["temporal_label"] == "flawed"
    ).astype(int)

    print("\nBinary target:")
    print("0 = normal")
    print("1 = flawed")

    print("\nTrain distribution:")
    print(y_train.value_counts())

    models = get_models()

    results = {}

    for name, model in models.items():

        results[name] = evaluate_model(
            name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test
        )

    # Select by validation F1.
    best_name = max(
        results,
        key=lambda name: results[name]["val_f1"]
    )

    best = results[best_name]

    print(
        f"\n🏆 BEST TEMPORAL MODEL: "
        f"{best_name}"
    )

    print(
        f"Validation F1: "
        f"{best['val_f1']:.4f}"
    )

    print(
        f"Test F1: "
        f"{best['test_f1']:.4f}"
    )

    # Save
    model_path = (
        MODEL_DIR / "temporal_flaw_detector.joblib"
    )

    joblib.dump(
        best["model"],
        model_path
    )

    print(
        f"\nModel saved:\n{model_path}"
    )

    # Save evaluation report
    report = classification_report(
        y_test,
        best["test_predictions"],
        target_names=[
            "normal",
            "flawed"
        ],
        output_dict=True
    )

    with open(
        RESULT_DIR / "temporal_detector_report.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    return best["model"]


# ===============================================================
# FLAW TYPE MODEL
# ===============================================================

def train_flaw_type_model(
    train_df,
    val_df,
    test_df,
    feature_columns
):

    banner("[7/8] TRAINING FLAW-TYPE CLASSIFIER")

    # Only actual flawed windows.
    train = train_df[
        train_df["temporal_label"] == "flawed"
    ].copy()

    val = val_df[
        val_df["temporal_label"] == "flawed"
    ].copy()

    test = test_df[
        test_df["temporal_label"] == "flawed"
    ].copy()

    if len(train) == 0:

        print(
            "No flawed training windows found."
        )

        return None

    # Remove rare classes from evaluation if necessary.
    valid_classes = (
        train["flaw_type"]
        .value_counts()
    )

    valid_classes = valid_classes[
        valid_classes >= 5
    ].index

    train = train[
        train["flaw_type"].isin(valid_classes)
    ]

    val = val[
        val["flaw_type"].isin(valid_classes)
    ]

    test = test[
        test["flaw_type"].isin(valid_classes)
    ]

    X_train = train[feature_columns]
    X_val = val[feature_columns]
    X_test = test[feature_columns]

    y_train = train["flaw_type"]
    y_val = val["flaw_type"]
    y_test = test["flaw_type"]

    models = {

        "ExtraTrees": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),

            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=350,
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
                SimpleImputer(strategy="median")
            ),

            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE
                )
            )
        ])
    }

    results = {}

    for name, model in models.items():

        print(
            f"\nTraining flaw-type model: {name}"
        )

        model.fit(
            X_train,
            y_train
        )

        val_pred = model.predict(X_val)

        test_pred = model.predict(X_test)

        val_f1 = f1_score(
            y_val,
            val_pred,
            average="macro"
        )

        test_f1 = f1_score(
            y_test,
            test_pred,
            average="macro"
        )

        print(
            f"Validation Macro F1: "
            f"{val_f1:.4f}"
        )

        print(
            f"Test Macro F1: "
            f"{test_f1:.4f}"
        )

        results[name] = {
            "model": model,
            "val_f1": val_f1,
            "test_f1": test_f1,
            "test_predictions": test_pred,
        }

    best_name = max(
        results,
        key=lambda name: results[name]["val_f1"]
    )

    best = results[best_name]

    print(
        f"\n🏆 BEST FLAW TYPE MODEL: "
        f"{best_name}"
    )

    print(
        f"Test Macro F1: "
        f"{best['test_f1']:.4f}"
    )

    model_path = (
        MODEL_DIR / "flaw_type_classifier.joblib"
    )

    joblib.dump(
        best["model"],
        model_path
    )

    labels_path = (
        MODEL_DIR / "flaw_type_labels.json"
    )

    with open(
        labels_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            sorted(valid_classes.tolist()),
            f,
            indent=2
        )

    print(
        f"Saved:\n{model_path}"
    )

    return best["model"]


# ===============================================================
# SEVERITY MODEL
# ===============================================================

def train_severity_model(
    train_df,
    val_df,
    test_df,
    feature_columns
):

    banner("[8/8] TRAINING SEVERITY CLASSIFIER")

    # Use flawed temporal windows only.
    train = train_df[
        train_df["temporal_label"] == "flawed"
    ].copy()

    val = val_df[
        val_df["temporal_label"] == "flawed"
    ].copy()

    test = test_df[
        test_df["temporal_label"] == "flawed"
    ].copy()

    if len(train) == 0:

        print(
            "No flawed windows available."
        )

        return None

    X_train = train[feature_columns]
    X_val = val[feature_columns]
    X_test = test[feature_columns]

    y_train = train["recording_label"]
    y_val = val["recording_label"]
    y_test = test["recording_label"]

    model = Pipeline([

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "model",
            ExtraTreesClassifier(
                n_estimators=400,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE
            )
        )
    ])

    print("Training severity model...")

    model.fit(
        X_train,
        y_train
    )

    val_pred = model.predict(X_val)

    test_pred = model.predict(X_test)

    val_f1 = f1_score(
        y_val,
        val_pred,
        average="macro"
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        average="macro"
    )

    print(
        f"Validation Macro F1: {val_f1:.4f}"
    )

    print(
        f"Test Macro F1:       {test_f1:.4f}"
    )

    model_path = (
        MODEL_DIR / "severity_classifier.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"Saved:\n{model_path}"
    )

    return model


# ===============================================================
# TEMPORAL PREDICTIONS
# ===============================================================

def generate_temporal_predictions(
    df,
    detector,
    flaw_model,
    severity_model,
    feature_columns
):

    banner("GENERATING TEMPORAL PREDICTIONS")

    result = df.copy()

    X = result[feature_columns]

    # -----------------------------------------------------------
    # Flaw probability
    # -----------------------------------------------------------

    if hasattr(detector, "predict_proba"):

        probabilities = detector.predict_proba(X)

        if probabilities.shape[1] == 2:

            result["flaw_probability"] = (
                probabilities[:, 1]
            )

        else:

            result["flaw_probability"] = 0.0

    else:

        result["flaw_probability"] = (
            detector.predict(X)
        )

    result["predicted_flaw"] = (
        result["flaw_probability"] >= 0.50
    )

    # -----------------------------------------------------------
    # Flaw type
    # -----------------------------------------------------------

    if flaw_model is not None:

        result["predicted_flaw_type"] = (
            flaw_model.predict(X)
        )

    else:

        result["predicted_flaw_type"] = "unknown"

    # -----------------------------------------------------------
    # Severity
    # -----------------------------------------------------------

    if severity_model is not None:

        result["predicted_severity"] = (
            severity_model.predict(X)
        )

    else:

        result["predicted_severity"] = "unknown"

    # -----------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------

    output_path = (
        RESULT_DIR / "temporal_predictions.csv"
    )

    result.to_csv(
        output_path,
        index=False
    )

    print(
        f"Temporal predictions saved:\n"
        f"{output_path}"
    )

    return result


# ===============================================================
# SAVE DATASET INFORMATION
# ===============================================================

def save_pipeline_info(
    feature_columns,
    normalized_columns
):

    info = {

        "target_sample_rate": TARGET_SR,

        "window_seconds": WINDOW_SECONDS,

        "hop_seconds": HOP_SECONDS,

        "flaw_overlap_threshold":
            FLAW_OVERLAP_THRESHOLD,

        "mfcc_count": N_MFCC,

        "feature_count":
            len(feature_columns),

        "normalized_feature_count":
            len(normalized_columns),

        "feature_columns":
            feature_columns,

        "normalized_columns":
            normalized_columns,

        "random_state":
            RANDOM_STATE,
    }

    path = (
        ARTIFACTS / "pipeline_config.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            info,
            f,
            indent=2
        )

    print(
        f"Pipeline configuration saved:\n{path}"
    )


# ===============================================================
# MAIN
# ===============================================================

def main():

    overall_start = time.time()

    banner(
        "ORATORIQ — MASTER TEMPORAL "
        "SPEECH ANALYTICS PIPELINE"
    )

    print(
        f"Project: {ROOT}"
    )

    print(
        f"Metadata: {METADATA}"
    )

    print(
        f"Window: {WINDOW_SECONDS}s"
    )

    print(
        f"Hop: {HOP_SECONDS}s"
    )

    print(
        "\nCheckpointing: ENABLED"
    )

    print(
        "Speaker-safe split: ENABLED"
    )

    # -----------------------------------------------------------
    # 1. Metadata
    # -----------------------------------------------------------

    df = load_metadata()

    # -----------------------------------------------------------
    # 2. Validate audio
    # -----------------------------------------------------------

    df = validate_audio_dataset(df)

    if len(df) == 0:

        raise RuntimeError(
            "No valid audio records found."
        )

    # -----------------------------------------------------------
    # 3. Feature extraction
    # -----------------------------------------------------------

    feature_checkpoint = (
        ARTIFACTS / "temporal_features.pkl"
    )

    if (
        USE_CACHE
        and feature_checkpoint.exists()
    ):

        print(
            "\nExisting temporal feature "
            "checkpoint detected."
        )

        print(
            "Loading checkpoint instead of "
            "extracting everything again..."
        )

        feature_df = joblib.load(
            feature_checkpoint
        )

        print(
            f"Loaded {len(feature_df):,} windows."
        )

    else:

        feature_df = build_temporal_dataset(df)

    # -----------------------------------------------------------
    # Feature columns
    # -----------------------------------------------------------

    excluded = {

        "record_id",
        "pair_id",
        "speaker_id",

        "window_start",
        "window_end",

        "temporal_label",
        "recording_label",
        "severity",
        "flaw_type",

        "transcript",
    }

    feature_columns = [
        c
        for c in feature_df.columns
        if c not in excluded
        and pd.api.types.is_numeric_dtype(
            feature_df[c]
        )
    ]

    print(
        f"\nBase numeric features: "
        f"{len(feature_columns)}"
    )

    # -----------------------------------------------------------
    # Speaker normalization
    # -----------------------------------------------------------

    feature_df, normalized_columns = (
        speaker_normalize(
            feature_df,
            feature_columns
        )
    )

    # Use normalized features + selected raw features.
    #
    # Keeping some raw features allows the model to learn
    # absolute acoustic changes while normalized features
    # reduce speaker-specific differences.

    model_features = (
        feature_columns
        + normalized_columns
    )

    print(
        f"Total model features: "
        f"{len(model_features)}"
    )

    # -----------------------------------------------------------
    # Save feature information
    # -----------------------------------------------------------

    save_pipeline_info(
        model_features,
        normalized_columns
    )

    # -----------------------------------------------------------
    # Split
    # -----------------------------------------------------------

    train_df, val_df, test_df = (
        create_splits(feature_df)
    )

    # -----------------------------------------------------------
    # Binary temporal detector
    # -----------------------------------------------------------

    detector = train_temporal_detector(
        train_df,
        val_df,
        test_df,
        model_features
    )

    # -----------------------------------------------------------
    # Flaw type
    # -----------------------------------------------------------

    flaw_model = train_flaw_type_model(
        train_df,
        val_df,
        test_df,
        model_features
    )

    # -----------------------------------------------------------
    # Severity
    # -----------------------------------------------------------

    severity_model = train_severity_model(
        train_df,
        val_df,
        test_df,
        model_features
    )

    # -----------------------------------------------------------
    # Predictions
    # -----------------------------------------------------------

    prediction_df = generate_temporal_predictions(
        test_df,
        detector,
        flaw_model,
        severity_model,
        model_features
    )

    # -----------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------

    banner("ORATORIQ PIPELINE COMPLETE")

    print(
        f"Total runtime: "
        f"{elapsed_string(time.time() - overall_start)}"
    )

    print("\nArtifacts:")

    print(
        f"  Feature cache: {CACHE_DIR}"
    )

    print(
        f"  Models:        {MODEL_DIR}"
    )

    print(
        f"  Results:       {RESULT_DIR}"
    )

    print("\nMain files:")

    print(
        "  temporal_features.pkl"
    )

    print(
        "  models\\temporal_flaw_detector.joblib"
    )

    print(
        "  models\\flaw_type_classifier.joblib"
    )

    print(
        "  models\\severity_classifier.joblib"
    )

    print(
        "  results\\temporal_predictions.csv"
    )

    print(
        "\n✅ Training pipeline finished successfully."
    )


# ===============================================================
# ENTRY POINT
# ===============================================================

if __name__ == "__main__":

    main()