from pathlib import Path
import json
import time
import warnings

import joblib
import librosa
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ============================================================
# ORATORIQ STUTTER PIPELINE V2
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

FEATURE_PATH = (
    FEATURE_DIR
    / "stutter_features_v2.pkl"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

SR = 16000

N_MFCC = 13

RANDOM_STATE = 42

CHECKPOINT_EVERY = 250

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


def safe_percentile(x, p):

    x = np.asarray(x)

    if x.size == 0:
        return 0.0

    return float(np.nanpercentile(x, p))


def elapsed_text(seconds):

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = seconds / 60

    if minutes < 60:
        return f"{minutes:.1f}m"

    return f"{minutes / 60:.2f}h"


def progress(current, total, started):

    elapsed = time.time() - started

    speed = (
        current / elapsed
        if elapsed > 0
        else 0
    )

    remaining = total - current

    eta = (
        remaining / speed
        if speed > 0
        else 0
    )

    percent = (
        current / total * 100
        if total
        else 0
    )

    print(
        f"[FEATURES] "
        f"{current:,}/{total:,} "
        f"({percent:6.2f}%) | "
        f"{speed:.2f} files/s | "
        f"elapsed {elapsed_text(elapsed)} | "
        f"ETA {elapsed_text(eta)}"
    )


# ============================================================
# TEMPORAL FEATURE HELPERS
# ============================================================

def temporal_statistics(signal):

    """
    Calculate statistics from a frame-level signal.
    """

    signal = np.asarray(signal)

    if signal.size == 0:

        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "range": 0.0,
        }

    return {
        "mean": safe_mean(signal),
        "std": safe_std(signal),
        "median": safe_median(signal),
        "p10": safe_percentile(signal, 10),
        "p90": safe_percentile(signal, 90),
        "range": float(
            np.nanmax(signal)
            - np.nanmin(signal)
        ),
    }


def count_segments(mask):

    """
    Count transitions from False -> True.
    """

    mask = np.asarray(mask).astype(bool)

    if mask.size == 0:
        return 0

    padded = np.concatenate(
        [[False], mask, [False]]
    )

    changes = np.diff(
        padded.astype(np.int8)
    )

    starts = np.where(
        changes == 1
    )[0]

    return int(len(starts))


def calculate_pause_features(
    y,
    sr,
):

    """
    Estimate pauses using short-time energy.

    This is not clinical pause detection.
    It is an acoustic proxy.
    """

    frame_length = 512
    hop_length = 256

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    if len(rms) == 0:

        return {
            "pause_count": 0.0,
            "pause_ratio": 0.0,
            "mean_pause_duration": 0.0,
            "max_pause_duration": 0.0,
            "short_pause_count": 0.0,
            "long_pause_count": 0.0,
        }

    threshold = np.percentile(
        rms,
        20,
    )

    silence_mask = rms <= threshold

    # Remove very short isolated silent frames.
    min_frames = max(
        1,
        int(
            0.12
            * sr
            / hop_length
        ),
    )

    pause_durations = []

    run = 0

    for value in silence_mask:

        if value:

            run += 1

        else:

            if run >= min_frames:

                duration = (
                    run
                    * hop_length
                    / sr
                )

                pause_durations.append(
                    duration
                )

            run = 0

    if run >= min_frames:

        duration = (
            run
            * hop_length
            / sr
        )

        pause_durations.append(
            duration
        )

    pause_durations = np.asarray(
        pause_durations
    )

    if pause_durations.size == 0:

        return {
            "pause_count": 0.0,
            "pause_ratio": 0.0,
            "mean_pause_duration": 0.0,
            "max_pause_duration": 0.0,
            "short_pause_count": 0.0,
            "long_pause_count": 0.0,
        }

    return {
        "pause_count": float(
            len(pause_durations)
        ),

        "pause_ratio": float(
            np.sum(pause_durations)
            / max(
                len(y) / sr,
                1e-6,
            )
        ),

        "mean_pause_duration": safe_mean(
            pause_durations
        ),

        "max_pause_duration": float(
            np.max(pause_durations)
        ),

        "short_pause_count": float(
            np.sum(
                pause_durations < 0.50
            )
        ),

        "long_pause_count": float(
            np.sum(
                pause_durations >= 0.50
            )
        ),
    }


def calculate_rate_features(
    y,
    sr,
):

    """
    Acoustic speech-rate proxies.

    These are deliberately described as proxies,
    because real WPM requires reliable transcription.
    """

    duration = len(y) / sr

    if duration <= 0:

        return {
            "energy_peak_count": 0.0,
            "energy_peak_rate": 0.0,
            "energy_variation": 0.0,
            "energy_change_rate": 0.0,
        }

    rms = librosa.feature.rms(
        y=y,
        frame_length=512,
        hop_length=256,
    )[0]

    if len(rms) < 3:

        return {
            "energy_peak_count": 0.0,
            "energy_peak_rate": 0.0,
            "energy_variation": 0.0,
            "energy_change_rate": 0.0,
        }

    # Smooth energy.
    kernel_size = 5

    kernel = np.ones(
        kernel_size
    ) / kernel_size

    smooth = np.convolve(
        rms,
        kernel,
        mode="same",
    )

    threshold = (
        np.mean(smooth)
        + 0.5 * np.std(smooth)
    )

    peaks = (
        (smooth[1:-1] > smooth[:-2])
        &
        (smooth[1:-1] >= smooth[2:])
        &
        (smooth[1:-1] > threshold)
    )

    peak_count = int(
        np.sum(peaks)
    )

    changes = np.abs(
        np.diff(smooth)
    )

    return {
        "energy_peak_count": float(
            peak_count
        ),

        "energy_peak_rate": float(
            peak_count / duration
        ),

        "energy_variation": safe_std(
            smooth
        ),

        "energy_change_rate": safe_mean(
            changes
        ),
    }


# ============================================================
# MAIN FEATURE EXTRACTION
# ============================================================

def extract_features(audio_path):

    try:

        y, sr = librosa.load(
            audio_path,
            sr=SR,
            mono=True,
        )

        if len(y) < int(
            0.15 * sr
        ):

            return None

        duration = len(y) / sr

        # ----------------------------------------------------
        # RMS
        # ----------------------------------------------------

        rms = librosa.feature.rms(
            y=y,
            frame_length=512,
            hop_length=256,
        )[0]

        # ----------------------------------------------------
        # ZCR
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

        delta = librosa.feature.delta(
            mfcc
        )

        delta2 = librosa.feature.delta(
            mfcc,
            order=2,
        )

        # ----------------------------------------------------
        # Spectral
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

            valid_f0 = f0[
                np.isfinite(f0)
            ]

        except Exception:

            f0 = np.full(
                max(1, len(y) // 256),
                np.nan,
            )

            valid_f0 = np.array([])

        # ----------------------------------------------------
        # Voicing
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
            voiced_samples
            / len(y)
        )

        silence_ratio = (
            1.0
            - voiced_ratio
        )

        # ----------------------------------------------------
        # Basic features
        # ----------------------------------------------------

        features = {

            "duration": duration,

            "rms_mean": safe_mean(rms),
            "rms_std": safe_std(rms),
            "rms_median": safe_median(rms),

            "zcr_mean": safe_mean(zcr),
            "zcr_std": safe_std(zcr),
            "zcr_median": safe_median(zcr),

            "centroid_mean": safe_mean(
                centroid
            ),

            "centroid_std": safe_std(
                centroid
            ),

            "bandwidth_mean": safe_mean(
                bandwidth
            ),

            "bandwidth_std": safe_std(
                bandwidth
            ),

            "rolloff_mean": safe_mean(
                rolloff
            ),

            "rolloff_std": safe_std(
                rolloff
            ),

            "flatness_mean": safe_mean(
                flatness
            ),

            "flatness_std": safe_std(
                flatness
            ),

            "f0_mean": safe_mean(
                valid_f0
            ),

            "f0_std": safe_std(
                valid_f0
            ),

            "f0_median": safe_median(
                valid_f0
            ),

            "f0_p10": safe_percentile(
                valid_f0,
                10,
            ),

            "f0_p90": safe_percentile(
                valid_f0,
                90,
            ),

            "voiced_ratio": voiced_ratio,

            "silence_ratio": silence_ratio,
        }

        # ----------------------------------------------------
        # MFCC + temporal MFCC
        # ----------------------------------------------------

        for i in range(N_MFCC):

            statistics = temporal_statistics(
                mfcc[i]
            )

            features[
                f"mfcc_{i+1}_mean"
            ] = statistics["mean"]

            features[
                f"mfcc_{i+1}_std"
            ] = statistics["std"]

            features[
                f"mfcc_{i+1}_range"
            ] = statistics["range"]

            features[
                f"delta_mfcc_{i+1}_mean"
            ] = safe_mean(
                delta[i]
            )

            features[
                f"delta_mfcc_{i+1}_std"
            ] = safe_std(
                delta[i]
            )

            features[
                f"delta2_mfcc_{i+1}_std"
            ] = safe_std(
                delta2[i]
            )

        # ----------------------------------------------------
        # Temporal RMS features
        # ----------------------------------------------------

        rms_stats = temporal_statistics(
            rms
        )

        features[
            "rms_range"
        ] = rms_stats["range"]

        features[
            "rms_p10"
        ] = rms_stats["p10"]

        features[
            "rms_p90"
        ] = rms_stats["p90"]

        if len(rms) > 1:

            rms_change = np.abs(
                np.diff(rms)
            )

            features[
                "rms_change_mean"
            ] = safe_mean(
                rms_change
            )

            features[
                "rms_change_std"
            ] = safe_std(
                rms_change
            )

            features[
                "rms_change_max"
            ] = float(
                np.max(rms_change)
            )

        else:

            features[
                "rms_change_mean"
            ] = 0.0

            features[
                "rms_change_std"
            ] = 0.0

            features[
                "rms_change_max"
            ] = 0.0

        # ----------------------------------------------------
        # F0 temporal variation
        # ----------------------------------------------------

        if len(valid_f0) > 1:

            f0_changes = np.abs(
                np.diff(valid_f0)
            )

            features[
                "f0_change_mean"
            ] = safe_mean(
                f0_changes
            )

            features[
                "f0_change_std"
            ] = safe_std(
                f0_changes
            )

            features[
                "f0_change_max"
            ] = float(
                np.max(f0_changes)
            )

        else:

            features[
                "f0_change_mean"
            ] = 0.0

            features[
                "f0_change_std"
            ] = 0.0

            features[
                "f0_change_max"
            ] = 0.0

        # ----------------------------------------------------
        # Pause features
        # ----------------------------------------------------

        features.update(
            calculate_pause_features(
                y,
                sr,
            )
        )

        # ----------------------------------------------------
        # Rate proxies
        # ----------------------------------------------------

        features.update(
            calculate_rate_features(
                y,
                sr,
            )
        )

        # ----------------------------------------------------
        # Zero crossing variation
        # ----------------------------------------------------

        if len(zcr) > 1:

            zcr_change = np.abs(
                np.diff(zcr)
            )

            features[
                "zcr_change_mean"
            ] = safe_mean(
                zcr_change
            )

            features[
                "zcr_change_std"
            ] = safe_std(
                zcr_change
            )

        else:

            features[
                "zcr_change_mean"
            ] = 0.0

            features[
                "zcr_change_std"
            ] = 0.0

        return features

    except Exception as e:

        print(
            f"[WARNING] Failed: "
            f"{audio_path}"
            f"\n          {e}"
        )

        return None


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("ORATORIQ — STUTTER ANALYSIS PIPELINE V2")
print("=" * 75)

print("\n[1/6] Loading metadata...")

metadata = pd.read_csv(
    METADATA_PATH
)

print(
    f"[OK] Metadata rows: "
    f"{len(metadata):,}"
)


# ============================================================
# FEATURE CACHE
# ============================================================

print("\n[2/6] Preparing V2 feature cache...")

if FEATURE_PATH.exists():

    print(
        "[CACHE] Existing V2 feature cache found:"
    )

    print(
        FEATURE_PATH
    )

    feature_df = pd.read_pickle(
        FEATURE_PATH
    )

    print(
        f"[CACHE] Loaded "
        f"{len(feature_df):,} rows."
    )

else:

    print(
        "[CACHE] No V2 cache found."
    )

    records = []

    total = len(metadata)

    started = time.time()

    for index, row in metadata.iterrows():

        features = extract_features(
            row["audio_path"]
        )

        if features is not None:

            record = {

                "filename": row["filename"],

                "audio_path": row["audio_path"],

                "show": row["show"],

                "episode_id": row["episode_id"],

                "clip_id": row["clip_id"],

                "dataset": row["dataset"],

                "has_stutter": int(
                    row["has_stutter"]
                ),

                "primary_event": row[
                    "primary_event"
                ],
            }

            record.update(
                features
            )

            records.append(
                record
            )

        current = index + 1

        if (
            current
            % CHECKPOINT_EVERY
            == 0
        ):

            progress(
                current,
                total,
                started,
            )

            checkpoint = pd.DataFrame(
                records
            )

            checkpoint.to_pickle(
                FEATURE_PATH
            )

    feature_df = pd.DataFrame(
        records
    )

    feature_df.to_pickle(
        FEATURE_PATH
    )

    progress(
        total,
        total,
        started,
    )

    print(
        f"\n[OK] V2 extraction complete."
    )

    print(
        f"[OK] Feature rows: "
        f"{len(feature_df):,}"
    )


# ============================================================
# CLEAN FEATURES
# ============================================================

print("\n[3/6] Cleaning V2 feature matrix...")

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
    c
    for c in feature_df.columns
    if c not in metadata_columns
]

X = feature_df[
    feature_columns
].replace(
    [np.inf, -np.inf],
    np.nan,
)

X = X.fillna(0)

feature_df[
    feature_columns
] = X

print(
    f"[OK] V2 numeric features: "
    f"{len(feature_columns)}"
)


# ============================================================
# EPISODE-SAFE SPLIT
# ============================================================

print(
    "\n[4/6] Creating episode-safe split..."
)

groups = (
    feature_df["show"].astype(str)
    + "_"
    + feature_df["episode_id"].astype(str)
)


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


splitter_2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=RANDOM_STATE,
)

val_idx, test_idx = next(
    splitter_2.split(
        temp_df,
        groups=temp_groups,
    )
)

val_df = temp_df.iloc[
    val_idx
].copy()

test_df = temp_df.iloc[
    test_idx
].copy()


def count_groups(df):

    return (
        df["show"].astype(str)
        + "_"
        + df["episode_id"].astype(str)
    ).nunique()


print(
    f"Train      : {len(train_df):,}"
)

print(
    f"Validation : {len(val_df):,}"
)

print(
    f"Test       : {len(test_df):,}"
)

print(
    f"Train episodes      : "
    f"{count_groups(train_df):,}"
)

print(
    f"Validation episodes : "
    f"{count_groups(val_df):,}"
)

print(
    f"Test episodes       : "
    f"{count_groups(test_df):,}"
)


# ============================================================
# STUTTER DETECTOR
# ============================================================

print(
    "\n[5/6] Training V2 stutter detector..."
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
    "has_stutter"
]

y_val = val_df[
    "has_stutter"
]

y_test = test_df[
    "has_stutter"
]


detector_candidates = {

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt",
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt",
    ),
}


detector_results = {}

best_detector = None

best_detector_name = None

best_detector_f1 = -1


for name, model in detector_candidates.items():

    print(
        f"\n[MODEL] Training {name}..."
    )

    model.fit(
        X_train,
        y_train,
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
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
    )

    val_accuracy = accuracy_score(
        y_val,
        val_pred,
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred,
    )

    detector_results[
        name
    ] = {

        "validation_accuracy":
            float(val_accuracy),

        "validation_macro_f1":
            float(val_f1),

        "test_accuracy":
            float(test_accuracy),

        "test_macro_f1":
            float(test_f1),
    }

    print(
        f"[RESULT] {name}"
    )

    print(
        f"  Validation accuracy : "
        f"{val_accuracy:.4f}"
    )

    print(
        f"  Validation macro F1 : "
        f"{val_f1:.4f}"
    )

    print(
        f"  Test accuracy       : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"  Test macro F1       : "
        f"{test_f1:.4f}"
    )

    if val_f1 > best_detector_f1:

        best_detector_f1 = val_f1

        best_detector = model

        best_detector_name = name

        best_detector_test_pred = (
            test_pred
        )


print(
    f"\n[BEST DETECTOR] "
    f"{best_detector_name}"
)


detector_report = classification_report(
    y_test,
    best_detector_test_pred,
    output_dict=True,
)

detector_cm = confusion_matrix(
    y_test,
    best_detector_test_pred,
)

with open(
    RESULT_DIR
    / "stutter_v2_detector_report.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        detector_report,
        f,
        indent=2,
    )


np.savetxt(
    RESULT_DIR
    / "stutter_v2_detector_confusion.csv",
    detector_cm,
    delimiter=",",
    fmt="%d",
)


joblib.dump(
    {
        "model": best_detector,
        "model_name": best_detector_name,
        "feature_columns": feature_columns,
        "task": "binary_stutter_detection",
        "version": "v2",
        "random_state": RANDOM_STATE,
    },
    MODEL_DIR
    / "stutter_detector_v2.joblib",
)


with open(
    RESULT_DIR
    / "stutter_v2_detector_comparison.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        detector_results,
        f,
        indent=2,
    )


# ============================================================
# EVENT CLASSIFIER
# ============================================================

print(
    "\n[6/6] Training V2 event classifier..."
)

event_train = train_df[
    train_df[
        "primary_event"
    ].isin(EVENT_CLASSES)
].copy()

event_val = val_df[
    val_df[
        "primary_event"
    ].isin(EVENT_CLASSES)
].copy()

event_test = test_df[
    test_df[
        "primary_event"
    ].isin(EVENT_CLASSES)
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


encoder = LabelEncoder()

encoder.fit(
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


y_event_train = encoder.transform(
    event_train[
        "primary_event"
    ]
)

y_event_val = encoder.transform(
    event_val[
        "primary_event"
    ]
)

y_event_test = encoder.transform(
    event_test[
        "primary_event"
    ]
)


event_candidates = {

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt",
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt",
    ),
}


event_results = {}

best_event_model = None

best_event_name = None

best_event_f1 = -1

best_event_test_pred = None


for name, model in event_candidates.items():

    print(
        f"\n[MODEL] Training {name}..."
    )

    model.fit(
        X_event_train,
        y_event_train,
    )

    val_pred = model.predict(
        X_event_val
    )

    test_pred = model.predict(
        X_event_test
    )

    val_f1 = f1_score(
        y_event_val,
        val_pred,
        average="macro",
    )

    test_f1 = f1_score(
        y_event_test,
        test_pred,
        average="macro",
    )

    val_accuracy = accuracy_score(
        y_event_val,
        val_pred,
    )

    test_accuracy = accuracy_score(
        y_event_test,
        test_pred,
    )

    event_results[
        name
    ] = {

        "validation_accuracy":
            float(val_accuracy),

        "validation_macro_f1":
            float(val_f1),

        "test_accuracy":
            float(test_accuracy),

        "test_macro_f1":
            float(test_f1),
    }

    print(
        f"[RESULT] {name}"
    )

    print(
        f"  Validation accuracy : "
        f"{val_accuracy:.4f}"
    )

    print(
        f"  Validation macro F1 : "
        f"{val_f1:.4f}"
    )

    print(
        f"  Test accuracy       : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"  Test macro F1       : "
        f"{test_f1:.4f}"
    )

    if val_f1 > best_event_f1:

        best_event_f1 = val_f1

        best_event_model = model

        best_event_name = name

        best_event_test_pred = (
            test_pred
        )


print(
    f"\n[BEST EVENT MODEL] "
    f"{best_event_name}"
)


event_report = classification_report(
    y_event_test,
    best_event_test_pred,
    target_names=encoder.classes_,
    output_dict=True,
)

event_cm = confusion_matrix(
    y_event_test,
    best_event_test_pred,
)


with open(
    RESULT_DIR
    / "stutter_v2_event_report.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        event_report,
        f,
        indent=2,
    )


np.savetxt(
    RESULT_DIR
    / "stutter_v2_event_confusion.csv",
    event_cm,
    delimiter=",",
    fmt="%d",
)


joblib.dump(
    {
        "model": best_event_model,
        "label_encoder": encoder,
        "model_name": best_event_name,
        "feature_columns": feature_columns,
        "classes": EVENT_CLASSES,
        "task": "stutter_event_classification",
        "version": "v2",
        "random_state": RANDOM_STATE,
    },
    MODEL_DIR
    / "stutter_event_classifier_v2.joblib",
)


with open(
    RESULT_DIR
    / "stutter_v2_event_comparison.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        event_results,
        f,
        indent=2,
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = (
    best_detector
    .feature_importances_
)

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
    RESULT_DIR
    / "stutter_v2_feature_importance.csv",
    index=False,
)


# ============================================================
# SPLIT INFORMATION
# ============================================================

split_info = {

    "train_rows":
        int(len(train_df)),

    "validation_rows":
        int(len(val_df)),

    "test_rows":
        int(len(test_df)),

    "train_episodes":
        int(count_groups(train_df)),

    "validation_episodes":
        int(count_groups(val_df)),

    "test_episodes":
        int(count_groups(test_df)),
}


with open(
    RESULT_DIR
    / "stutter_v2_split.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        split_info,
        f,
        indent=2,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {

    "version": "v2",

    "metadata_rows":
        int(len(metadata)),

    "feature_rows":
        int(len(feature_df)),

    "feature_count":
        int(len(feature_columns)),

    "detector_models":
        detector_results,

    "best_detector":
        best_detector_name,

    "event_models":
        event_results,

    "best_event_model":
        best_event_name,

    "event_classes":
        EVENT_CLASSES,

    "split":
        split_info,

    "feature_cache":
        str(FEATURE_PATH),

    "detector_model":
        str(
            MODEL_DIR
            / "stutter_detector_v2.joblib"
        ),

    "event_model":
        str(
            MODEL_DIR
            / "stutter_event_classifier_v2.joblib"
        ),
}


with open(
    RESULT_DIR
    / "stutter_v2_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print("\n" + "=" * 75)
print("ORATORIQ STUTTER V2 COMPLETE")
print("=" * 75)

print(
    f"Feature rows : "
    f"{len(feature_df):,}"
)

print(
    f"Features     : "
    f"{len(feature_columns)}"
)

print(
    f"Best detector: "
    f"{best_detector_name}"
)

print(
    f"Best event model: "
    f"{best_event_name}"
)

print(
    "\nModels saved:"
)

print(
    MODEL_DIR
    / "stutter_detector_v2.joblib"
)

print(
    MODEL_DIR
    / "stutter_event_classifier_v2.joblib"
)

print(
    "\nResults saved:"
)

print(
    RESULT_DIR
)

print("=" * 75)