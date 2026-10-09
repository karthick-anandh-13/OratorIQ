from pathlib import Path
import json
import time
import warnings

import joblib
import librosa
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")


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

FEATURE_DIR = (
    ROOT
    / "data"
    / "stutter"
    / "features"
)

MODEL_DIR = (
    ROOT
    / "artifacts"
    / "stutter_models"
)

RESULT_DIR = (
    ROOT
    / "artifacts"
    / "stutter_results"
)

FEATURE_PATH = FEATURE_DIR / "stutter_features.pkl"


FEATURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

SR = 16000

N_MFCC = 13

RANDOM_STATE = 42

MAX_FILES = None

CHECKPOINT_EVERY = 250

# Keep mixed samples out of the first event classifier.
EVENT_CLASSES = [
    "Prolongation",
    "Block",
    "SoundRep",
    "WordRep",
    "Interjection",
]


# ============================================================
# UTILITY
# ============================================================

def elapsed_text(seconds):

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = seconds / 60

    if minutes < 60:
        return f"{minutes:.1f}m"

    hours = minutes / 60

    return f"{hours:.2f}h"


def print_progress(current, total, started):

    elapsed = time.time() - started

    speed = current / elapsed if elapsed > 0 else 0

    remaining = total - current

    eta = remaining / speed if speed > 0 else 0

    percent = current / total * 100

    print(
        f"[FEATURES] "
        f"{current:,}/{total:,} "
        f"({percent:6.2f}%) | "
        f"{speed:.2f} files/s | "
        f"elapsed {elapsed_text(elapsed)} | "
        f"ETA {elapsed_text(eta)}"
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def safe_mean(x):

    x = np.asarray(x)

    if x.size == 0:
        return 0.0

    return float(np.nanmean(x))


def safe_std(x):

    x = np.asarray(x)

    if x.size == 0:
        return 0.0

    return float(np.nanstd(x))


def safe_median(x):

    x = np.asarray(x)

    if x.size == 0:
        return 0.0

    return float(np.nanmedian(x))


def extract_features(audio_path):

    try:

        y, sr = librosa.load(
            audio_path,
            sr=SR,
            mono=True,
        )

        if len(y) < int(0.1 * SR):
            return None

        duration = len(y) / sr

        # ----------------------------------------------------
        # Energy
        # ----------------------------------------------------

        rms = librosa.feature.rms(
            y=y,
            frame_length=512,
            hop_length=256,
        )[0]

        # ----------------------------------------------------
        # Zero crossing
        # ----------------------------------------------------

        zcr = librosa.feature.zero_crossing_rate(
            y,
            frame_length=512,
            hop_length=256,
        )[0]

        # ----------------------------------------------------
        # MFCC
        # ----------------------------------------------------

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=512,
            hop_length=256,
        )

        # ----------------------------------------------------
        # Delta MFCC
        # ----------------------------------------------------

        delta = librosa.feature.delta(mfcc)

        # ----------------------------------------------------
        # Spectral features
        # ----------------------------------------------------

        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=512,
            hop_length=256,
        )[0]

        bandwidth = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr,
            n_fft=512,
            hop_length=256,
        )[0]

        rolloff = librosa.feature.spectral_rolloff(
            y=y,
            sr=sr,
            n_fft=512,
            hop_length=256,
        )[0]

        flatness = librosa.feature.spectral_flatness(
            y=y,
            n_fft=512,
            hop_length=256,
        )[0]

        # ----------------------------------------------------
        # Pitch
        # ----------------------------------------------------

        try:

            f0, voiced_flag, voiced_prob = librosa.pyin(
                y,
                fmin=65,
                fmax=500,
                sr=sr,
                frame_length=1024,
                hop_length=256,
            )

            valid_f0 = f0[np.isfinite(f0)]

        except Exception:

            valid_f0 = np.array([])

        # ----------------------------------------------------
        # Silence
        # ----------------------------------------------------

        intervals = librosa.effects.split(
            y,
            top_db=30,
        )

        voiced_samples = 0

        if len(intervals):

            voiced_samples = sum(
                end - start
                for start, end in intervals
            )

        voiced_ratio = (
            voiced_samples / len(y)
            if len(y)
            else 0
        )

        silence_ratio = 1.0 - voiced_ratio

        # ----------------------------------------------------
        # Zero crossing / energy statistics
        # ----------------------------------------------------

        features = {

            "duration": duration,

            "rms_mean": safe_mean(rms),
            "rms_std": safe_std(rms),
            "rms_median": safe_median(rms),

            "zcr_mean": safe_mean(zcr),
            "zcr_std": safe_std(zcr),
            "zcr_median": safe_median(zcr),

            "spectral_centroid_mean": safe_mean(centroid),
            "spectral_centroid_std": safe_std(centroid),

            "spectral_bandwidth_mean": safe_mean(bandwidth),
            "spectral_bandwidth_std": safe_std(bandwidth),

            "spectral_rolloff_mean": safe_mean(rolloff),
            "spectral_rolloff_std": safe_std(rolloff),

            "spectral_flatness_mean": safe_mean(flatness),
            "spectral_flatness_std": safe_std(flatness),

            "f0_mean": safe_mean(valid_f0),
            "f0_std": safe_std(valid_f0),
            "f0_median": safe_median(valid_f0),

            "voiced_ratio": voiced_ratio,
            "silence_ratio": silence_ratio,
        }

        # ----------------------------------------------------
        # MFCC statistics
        # ----------------------------------------------------

        for i in range(N_MFCC):

            features[f"mfcc_{i+1}_mean"] = safe_mean(
                mfcc[i]
            )

            features[f"mfcc_{i+1}_std"] = safe_std(
                mfcc[i]
            )

            features[f"mfcc_{i+1}_median"] = safe_median(
                mfcc[i]
            )

            features[f"delta_mfcc_{i+1}_mean"] = safe_mean(
                delta[i]
            )

            features[f"delta_mfcc_{i+1}_std"] = safe_std(
                delta[i]
            )

        return features

    except Exception as e:

        print(
            f"[WARNING] Feature extraction failed: "
            f"{audio_path} | {e}"
        )

        return None


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 75)
print("ORATORIQ — STUTTER ANALYSIS MASTER PIPELINE")
print("=" * 75)

print("\n[1/6] Loading metadata...")

if not METADATA_PATH.exists():

    raise FileNotFoundError(
        f"Metadata not found:\n{METADATA_PATH}"
    )

metadata = pd.read_csv(METADATA_PATH)

print(
    f"[OK] Metadata rows: "
    f"{len(metadata):,}"
)


# ============================================================
# LIMIT DATASET IF REQUESTED
# ============================================================

if MAX_FILES is not None:

    metadata = metadata.head(MAX_FILES).copy()

    print(
        f"[INFO] MAX_FILES enabled: "
        f"{len(metadata):,}"
    )


# ============================================================
# FEATURE CACHE
# ============================================================

print("\n[2/6] Preparing feature cache...")

if FEATURE_PATH.exists():

    print(
        f"[CACHE] Loading existing features:\n"
        f"        {FEATURE_PATH}"
    )

    feature_df = pd.read_pickle(FEATURE_PATH)

    print(
        f"[CACHE] Loaded "
        f"{len(feature_df):,} feature rows."
    )

else:

    print(
        "[CACHE] No feature cache found."
    )

    feature_records = []

    total = len(metadata)

    started = time.time()

    for index, row in metadata.iterrows():

        features = extract_features(
            row["audio_path"]
        )

        if features is None:
            continue

        record = {
            "filename": row["filename"],
            "audio_path": row["audio_path"],
            "show": row["show"],
            "episode_id": row["episode_id"],
            "clip_id": row["clip_id"],
            "dataset": row["dataset"],
            "has_stutter": int(row["has_stutter"]),
            "primary_event": row["primary_event"],
        }

        record.update(features)

        feature_records.append(record)

        current = index + 1

        if current % CHECKPOINT_EVERY == 0:

            print_progress(
                current,
                total,
                started,
            )

            # Save checkpoint.
            checkpoint_df = pd.DataFrame(
                feature_records
            )

            checkpoint_df.to_pickle(
                FEATURE_PATH
            )

    feature_df = pd.DataFrame(
        feature_records
    )

    feature_df.to_pickle(
        FEATURE_PATH
    )

    print_progress(
        total,
        total,
        started,
    )

    print(
        f"\n[OK] Feature extraction complete."
    )

    print(
        f"[OK] Valid feature rows: "
        f"{len(feature_df):,}"
    )


# ============================================================
# CLEAN FEATURES
# ============================================================

print("\n[3/6] Cleaning feature matrix...")

metadata_columns = [
    "filename",
    "audio_path",
    "show",
    "episode_id",
    "clip_id",
    "dataset",
    "has_stutter",
    "primary_event",
]

feature_columns = [
    column
    for column in feature_df.columns
    if column not in metadata_columns
]

X_all = feature_df[
    feature_columns
].replace(
    [np.inf, -np.inf],
    np.nan,
)

X_all = X_all.fillna(0)

feature_df[feature_columns] = X_all

print(
    f"[OK] Numeric features: "
    f"{len(feature_columns)}"
)


# ============================================================
# EPISODE-SAFE SPLIT
# ============================================================

print("\n[4/6] Creating episode-safe train/validation/test split...")

groups = (
    feature_df["show"].astype(str)
    + "_"
    + feature_df["episode_id"].astype(str)
)

# First: train vs temporary
splitter_1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

train_idx, temp_idx = next(
    splitter_1.split(
        feature_df,
        groups=groups,
    )
)

train_df = feature_df.iloc[
    train_idx
].copy()

temp_df = feature_df.iloc[
    temp_idx
].copy()

temp_groups = groups.iloc[
    temp_idx
]

# Second: validation vs test
splitter_2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=RANDOM_STATE,
)

val_relative_idx, test_relative_idx = next(
    splitter_2.split(
        temp_df,
        groups=temp_groups,
    )
)

val_df = temp_df.iloc[
    val_relative_idx
].copy()

test_df = temp_df.iloc[
    test_relative_idx
].copy()


print(
    f"Train      : {len(train_df):,}"
)

print(
    f"Validation : {len(val_df):,}"
)

print(
    f"Test       : {len(test_df):,}"
)


def group_count(df):

    return (
        df["show"].astype(str)
        + "_"
        + df["episode_id"].astype(str)
    ).nunique()


print(
    f"Train episodes      : {group_count(train_df):,}"
)

print(
    f"Validation episodes : {group_count(val_df):,}"
)

print(
    f"Test episodes       : {group_count(test_df):,}"
)


# ============================================================
# SAVE SPLIT
# ============================================================

split_output = {
    "train_rows": len(train_df),
    "validation_rows": len(val_df),
    "test_rows": len(test_df),
    "train_episodes": group_count(train_df),
    "validation_episodes": group_count(val_df),
    "test_episodes": group_count(test_df),
}

with open(
    RESULT_DIR / "stutter_split.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        split_output,
        f,
        indent=2,
    )


# ============================================================
# MODEL 1 — STUTTER DETECTOR
# ============================================================

print("\n[5/6] Training stutter detector...")

X_train = train_df[feature_columns]
X_val = val_df[feature_columns]
X_test = test_df[feature_columns]

y_train = train_df["has_stutter"]
y_val = val_df["has_stutter"]
y_test = test_df["has_stutter"]


detector = ExtraTreesClassifier(
    n_estimators=350,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced",
    min_samples_leaf=2,
    max_features="sqrt",
)

print("[MODEL] ExtraTrees stutter detector")

detector.fit(
    X_train,
    y_train,
)

val_pred = detector.predict(X_val)
test_pred = detector.predict(X_test)

val_f1 = f1_score(
    y_val,
    val_pred,
    average="macro",
)

test_f1 = f1_score(
    y_test,
    test_pred,
    average="macro",
)

print(
    f"[RESULT] Validation accuracy: "
    f"{accuracy_score(y_val, val_pred):.4f}"
)

print(
    f"[RESULT] Validation macro F1: "
    f"{val_f1:.4f}"
)

print(
    f"[RESULT] Test accuracy: "
    f"{accuracy_score(y_test, test_pred):.4f}"
)

print(
    f"[RESULT] Test macro F1: "
    f"{test_f1:.4f}"
)


detector_report = classification_report(
    y_test,
    test_pred,
    output_dict=True,
)

detector_cm = confusion_matrix(
    y_test,
    test_pred,
)

with open(
    RESULT_DIR / "stutter_detector_report.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        detector_report,
        f,
        indent=2,
    )

np.savetxt(
    RESULT_DIR / "stutter_detector_confusion.csv",
    detector_cm,
    delimiter=",",
    fmt="%d",
)


joblib.dump(
    {
        "model": detector,
        "model_name": "ExtraTrees",
        "feature_columns": feature_columns,
        "random_state": RANDOM_STATE,
        "task": "binary_stutter_detection",
    },
    MODEL_DIR / "stutter_detector.joblib",
)


print(
    "[SAVED] "
    f"{MODEL_DIR / 'stutter_detector.joblib'}"
)


# ============================================================
# MODEL 2 — EVENT CLASSIFIER
# ============================================================

print("\n[6/6] Training stutter event classifier...")

# Only use clean single-event samples.
event_mask_train = train_df[
    "primary_event"
].isin(EVENT_CLASSES)

event_mask_val = val_df[
    "primary_event"
].isin(EVENT_CLASSES)

event_mask_test = test_df[
    "primary_event"
].isin(EVENT_CLASSES)


event_train = train_df[
    event_mask_train
].copy()

event_val = val_df[
    event_mask_val
].copy()

event_test = test_df[
    event_mask_test
].copy()


print(
    f"Event training rows   : "
    f"{len(event_train):,}"
)

print(
    f"Event validation rows : "
    f"{len(event_val):,}"
)

print(
    f"Event test rows       : "
    f"{len(event_test):,}"
)


if len(event_train) == 0:

    print(
        "[WARNING] No event-training samples."
    )

else:

    event_encoder = LabelEncoder()

    event_encoder.fit(
        EVENT_CLASSES
    )

    X_event_train = event_train[
        feature_columns
    ]

    X_event_val = event_val[
        feature_columns
    ]

    X_event_test = event_test[
        feature_columns
    ]

    y_event_train = event_encoder.transform(
        event_train["primary_event"]
    )

    y_event_val = event_encoder.transform(
        event_val["primary_event"]
    )

    y_event_test = event_encoder.transform(
        event_test["primary_event"]
    )

    event_model = ExtraTreesClassifier(
        n_estimators=350,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt",
    )

    print(
        "[MODEL] ExtraTrees event classifier"
    )

    event_model.fit(
        X_event_train,
        y_event_train,
    )

    event_val_pred = event_model.predict(
        X_event_val
    )

    event_test_pred = event_model.predict(
        X_event_test
    )

    event_val_f1 = f1_score(
        y_event_val,
        event_val_pred,
        average="macro",
    )

    event_test_f1 = f1_score(
        y_event_test,
        event_test_pred,
        average="macro",
    )

    print(
        f"[RESULT] Event validation accuracy: "
        f"{accuracy_score(y_event_val, event_val_pred):.4f}"
    )

    print(
        f"[RESULT] Event validation macro F1: "
        f"{event_val_f1:.4f}"
    )

    print(
        f"[RESULT] Event test accuracy: "
        f"{accuracy_score(y_event_test, event_test_pred):.4f}"
    )

    print(
        f"[RESULT] Event test macro F1: "
        f"{event_test_f1:.4f}"
    )

    event_report = classification_report(
        y_event_test,
        event_test_pred,
        target_names=event_encoder.classes_,
        output_dict=True,
    )

    event_cm = confusion_matrix(
        y_event_test,
        event_test_pred,
    )

    with open(
        RESULT_DIR / "stutter_event_report.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            event_report,
            f,
            indent=2,
        )

    np.savetxt(
        RESULT_DIR / "stutter_event_confusion.csv",
        event_cm,
        delimiter=",",
        fmt="%d",
    )

    joblib.dump(
        {
            "model": event_model,
            "label_encoder": event_encoder,
            "model_name": "ExtraTrees",
            "feature_columns": feature_columns,
            "classes": EVENT_CLASSES,
            "task": "stutter_event_classification",
            "random_state": RANDOM_STATE,
        },
        MODEL_DIR / "stutter_event_classifier.joblib",
    )

    print(
        "[SAVED] "
        f"{MODEL_DIR / 'stutter_event_classifier.joblib'}"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = detector.feature_importances_

importance_df = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance": importance,
    }
).sort_values(
    "importance",
    ascending=False,
)

importance_df.to_csv(
    RESULT_DIR / "stutter_feature_importance.csv",
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {

    "dataset": {
        "total_metadata_rows": int(len(metadata)),
        "feature_rows": int(len(feature_df)),
    },

    "features": {
        "count": len(feature_columns),
        "feature_file": str(FEATURE_PATH),
    },

    "split": split_output,

    "stutter_detector": {
        "model": "ExtraTrees",
        "validation_macro_f1": float(val_f1),
        "test_macro_f1": float(test_f1),
        "test_accuracy": float(
            accuracy_score(
                y_test,
                test_pred,
            )
        ),
    },

    "event_classifier": {
        "classes": EVENT_CLASSES,
        "test_rows": int(len(event_test)),
    },

}


with open(
    RESULT_DIR / "stutter_pipeline_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print("\n" + "=" * 75)
print("STUTTER PIPELINE COMPLETE")
print("=" * 75)

print(
    f"Features : {len(feature_columns)}"
)

print(
    f"Rows     : {len(feature_df):,}"
)

print(
    f"Detector test F1 : {test_f1:.4f}"
)

print(
    "\nModels:"
)

print(
    MODEL_DIR / "stutter_detector.joblib"
)

print(
    MODEL_DIR / "stutter_event_classifier.joblib"
)

print(
    "\nResults:"
)

print(
    RESULT_DIR
)

print("=" * 75)