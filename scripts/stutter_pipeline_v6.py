"""
ORATORIQ STUTTER V6
===================

WEAKLY-SUPERVISED TEMPORAL FLUENCY / STUTTER MODEL

Purpose:
    Convert the existing 3-second SEP-28k clips into short
    overlapping temporal windows and train models that can
    produce a probability timeline.

IMPORTANT:
    SEP-28k provides clip-level event annotations.
    It does NOT provide exact event start/end timestamps.

Therefore V6 is:

    CLIP LABEL
        ->
    TEMPORAL WINDOWS
        ->
    WEAK WINDOW LABEL
        ->
    TEMPORAL MODEL
        ->
    CANDIDATE EVENT REGIONS

These regions must NOT be described as clinically verified
stutter boundaries.

Design:
    - Episode-safe train/validation/test split
    - No annotation columns used as features
    - Audio-only features
    - 121 temporal/audio features
    - 1.0 second windows
    - 0.5 second hop
    - Validation-only threshold selection
    - ExtraTrees + RandomForest
    - Permanent model saving
    - Window-level probabilities
    - Region merging
    - Progress + ETA
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import librosa
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

TEMPORAL_FEATURE_CACHE = (
    ROOT
    / "data"
    / "stutter"
    / "features"
    / "stutter_temporal_features_v6.pkl"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TEMPORAL_FEATURE_CACHE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

SR = 16000

WINDOW_SECONDS = 1.0

HOP_SECONDS = 0.5

MIN_AUDIO_SECONDS = 0.20

N_MFCC = 13

EVENT_COLUMNS = [
    "Prolongation",
    "Block",
    "SoundRep",
    "WordRep",
    "Interjection",
]

THRESHOLD_MIN = 0.10

THRESHOLD_MAX = 0.90

THRESHOLD_STEP = 0.01

MERGE_GAP_SECONDS = 0.50

MAX_WINDOWS_PER_CLIP = 5


# ============================================================
# UTILITY
# ============================================================

def banner(title: str):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def save_json(
    path: Path,
    data,
):

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


def elapsed(
    start_time,
):

    return time.time() - start_time


def format_seconds(
    seconds,
):

    seconds = max(
        0,
        float(seconds),
    )

    minutes = int(
        seconds // 60
    )

    secs = int(
        seconds % 60
    )

    return (
        f"{minutes}m {secs}s"
    )


def normalize_path(
    value,
):

    if pd.isna(value):

        return ""

    return (
        str(value)
        .strip()
        .replace(
            "/",
            "\\",
        )
        .lower()
    )


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
        .astype(
            np.int8
        )
        .values
    )


# ============================================================
# METADATA
# ============================================================

def load_metadata():

    banner(
        "[1/8] Loading metadata"
    )

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata not found:\n{METADATA_PATH}"
        )

    df = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"Metadata rows : {len(df):,}"
    )

    print(
        f"Metadata cols : {len(df.columns):,}"
    )

    required = [
        "audio_path",
        "filename",
        "show",
        "episode_id",
        "clip_id",
        "start",
        "stop",
        "has_stutter",
        *EVENT_COLUMNS,
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing metadata columns:\n"
            + "\n".join(
                f"  - {x}"
                for x in missing
            )
        )

    return df


# ============================================================
# EPISODE-SAFE SPLIT
# ============================================================

def create_episode_split(
    data,
):

    banner(
        "[2/8] Creating episode-safe split"
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

    if train_groups & val_groups:

        raise RuntimeError(
            "Train/validation episode leakage."
        )

    if train_groups & test_groups:

        raise RuntimeError(
            "Train/test episode leakage."
        )

    if val_groups & test_groups:

        raise RuntimeError(
            "Validation/test episode leakage."
        )

    print(
        "[OK] Episode-safe split verified."
    )

    print(
        f"Train clips      : {len(train_idx):,}"
    )

    print(
        f"Validation clips : {len(val_idx):,}"
    )

    print(
        f"Test clips       : {len(test_idx):,}"
    )

    print(
        f"Train episodes   : {len(train_groups):,}"
    )

    print(
        f"Val episodes     : {len(val_groups):,}"
    )

    print(
        f"Test episodes    : {len(test_groups):,}"
    )

    return (
        train_idx,
        val_idx,
        test_idx,
    )


# ============================================================
# 121-FEATURE AUDIO EXTRACTOR
# ============================================================

def safe_stats(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.median(values)),
        float(np.min(values)),
        float(np.max(values)),
        float(np.percentile(values, 90)),
    ]


def extract_121_features(
    y,
    sr,
):

    required_samples = int(
        sr * MIN_AUDIO_SECONDS
    )

    if len(y) < required_samples:

        return None

    # --------------------------------------------------------
    # Normalize safely
    # --------------------------------------------------------

    y = np.asarray(
        y,
        dtype=np.float32,
    )

    y = np.nan_to_num(
        y
    )

    duration = (
        len(y) / sr
    )

    # --------------------------------------------------------
    # RMS
    # 6
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=y,
        frame_length=min(
            1024,
            len(y),
        ),
        hop_length=256,
    )[0]

    rms_features = safe_stats(
        rms
    )

    # --------------------------------------------------------
    # ZCR
    # 6
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=min(
            1024,
            len(y),
        ),
        hop_length=256,
    )[0]

    zcr_features = safe_stats(
        zcr
    )

    # --------------------------------------------------------
    # Spectral features
    #
    # centroid 5
    # bandwidth 5
    # rolloff 5
    # flatness 5
    #
    # 20
    # --------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        hop_length=256,
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        hop_length=256,
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        hop_length=256,
    )[0]

    flatness = librosa.feature.spectral_flatness(
        y=y,
        hop_length=256,
    )[0]

    def five_stats(
        values,
    ):

        values = np.asarray(
            values,
            dtype=np.float32,
        )

        values = values[
            np.isfinite(values)
        ]

        if len(values) == 0:

            return [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]

        return [
            float(np.mean(values)),
            float(np.std(values)),
            float(np.median(values)),
            float(np.percentile(values, 10)),
            float(np.percentile(values, 90)),
        ]

    centroid_features = five_stats(
        centroid
    )

    bandwidth_features = five_stats(
        bandwidth
    )

    rolloff_features = five_stats(
        rolloff
    )

    flatness_features = five_stats(
        flatness
    )

    # --------------------------------------------------------
    # F0
    # 8
    # --------------------------------------------------------

    try:

        f0 = librosa.yin(
            y,
            fmin=70,
            fmax=400,
            sr=sr,
            frame_length=1024,
            hop_length=256,
        )

        f0 = f0[
            np.isfinite(f0)
        ]

        f0 = f0[
            f0 > 0
        ]

        if len(f0):

            f0_features = [
                float(np.mean(f0)),
                float(np.std(f0)),
                float(np.median(f0)),
                float(np.min(f0)),
                float(np.max(f0)),
                float(np.percentile(f0, 10)),
                float(np.percentile(f0, 90)),
                float(
                    np.mean(
                        np.abs(
                            np.diff(f0)
                        )
                    )
                )
                if len(f0) > 1
                else 0.0,
            ]

        else:

            f0_features = [
                0.0
            ] * 8

    except Exception:

        f0_features = [
            0.0
        ] * 8

    # --------------------------------------------------------
    # Voicing / silence
    # 4
    # --------------------------------------------------------

    try:

        intervals = librosa.effects.split(
            y,
            top_db=30,
        )

        voiced_samples = sum(
            max(
                0,
                end - start,
            )
            for start, end in intervals
        )

        voiced_ratio = (
            voiced_samples
            / max(
                len(y),
                1,
            )
        )

        silence_ratio = (
            1.0
            - voiced_ratio
        )

    except Exception:

        voiced_ratio = 0.0

        silence_ratio = 1.0

    # --------------------------------------------------------
    # Energy temporal dynamics
    # 8
    # --------------------------------------------------------

    if len(rms) > 1:

        rms_diff = np.diff(
            rms
        )

        rms_abs_diff = np.abs(
            rms_diff
        )

        energy_dynamic = [
            float(np.mean(rms_diff)),
            float(np.std(rms_diff)),
            float(np.mean(rms_abs_diff)),
            float(np.max(rms_abs_diff)),
            float(
                np.percentile(
                    rms_abs_diff,
                    90,
                )
            ),
            float(
                np.mean(
                    rms_diff > 0
                )
            ),
            float(
                np.mean(
                    rms_diff < 0
                )
            ),
            float(
                np.mean(
                    rms > np.mean(rms)
                )
            ),
        ]

    else:

        energy_dynamic = [
            0.0
        ] * 8

    # --------------------------------------------------------
    # Pause / silence dynamics
    # 8
    # --------------------------------------------------------

    silence_mask = (
        rms
        < (
            np.max(rms)
            * 0.10
            if len(rms)
            else 0.0
        )
    )

    if len(silence_mask):

        changes = np.diff(
            silence_mask.astype(
                np.int8
            )
        )

        pause_starts = np.where(
            changes == 1
        )[0]

        pause_ends = np.where(
            changes == -1
        )[0]

        pause_count = len(
            pause_starts
        )

        silence_frames = int(
            np.sum(
                silence_mask
            )
        )

        silence_frame_ratio = (
            silence_frames
            / len(silence_mask)
        )

        longest_silence = 0

        current = 0

        for value in silence_mask:

            if value:

                current += 1

                longest_silence = max(
                    longest_silence,
                    current,
                )

            else:

                current = 0

        pause_durations = []

        for start in pause_starts:

            end_candidates = (
                pause_ends[
                    pause_ends > start
                ]
            )

            if len(
                end_candidates
            ):

                end = end_candidates[0]

                pause_durations.append(
                    float(
                        end - start
                    )
                )

        pause_stats = [
            float(pause_count),
            float(silence_frame_ratio),
            float(longest_silence),
            float(
                np.mean(
                    pause_durations
                )
            )
            if pause_durations
            else 0.0,
            float(
                np.max(
                    pause_durations
                )
            )
            if pause_durations
            else 0.0,
            float(
                np.std(
                    pause_durations
                )
            )
            if pause_durations
            else 0.0,
            float(
                np.sum(
                    silence_mask
                )
            ),
            float(
                np.mean(
                    np.diff(
                        np.where(
                            silence_mask
                        )[0]
                    )
                )
            )
            if np.sum(
                silence_mask
            ) > 1
            else 0.0,
        ]

    else:

        pause_stats = [
            0.0
        ] * 8

    # --------------------------------------------------------
    # Spectral temporal dynamics
    # 8
    # --------------------------------------------------------

    spectral_stack = np.vstack(
        [
            centroid,
            bandwidth,
            rolloff,
            flatness,
        ]
    )

    spectral_dynamic = []

    for row in spectral_stack:

        if len(row) > 1:

            diff = np.diff(
                row
            )

            spectral_dynamic.extend(
                [
                    float(
                        np.mean(diff)
                    ),
                    float(
                        np.std(diff)
                    ),
                ]
            )

        else:

            spectral_dynamic.extend(
                [
                    0.0,
                    0.0,
                ]
            )

    spectral_dynamic = (
        spectral_dynamic[:8]
    )

    while len(
        spectral_dynamic
    ) < 8:

        spectral_dynamic.append(
            0.0
        )

    # --------------------------------------------------------
    # MFCC
    #
    # mean 13
    # std 13
    # delta mean 13
    # delta std 13
    #
    # 52
    # --------------------------------------------------------

    try:

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=min(
                1024,
                len(y),
            ),
            hop_length=256,
        )

        mfcc_mean = np.mean(
            mfcc,
            axis=1,
        )

        mfcc_std = np.std(
            mfcc,
            axis=1,
        )

        if mfcc.shape[1] >= 3:

            delta = librosa.feature.delta(
                mfcc
            )

            delta_mean = np.mean(
                delta,
                axis=1,
            )

            delta_std = np.std(
                delta,
                axis=1,
            )

        else:

            delta_mean = np.zeros(
                N_MFCC
            )

            delta_std = np.zeros(
                N_MFCC
            )

    except Exception:

        mfcc_mean = np.zeros(
            N_MFCC
        )

        mfcc_std = np.zeros(
            N_MFCC
        )

        delta_mean = np.zeros(
            N_MFCC
        )

        delta_std = np.zeros(
            N_MFCC
        )

    # --------------------------------------------------------
    # BUILD EXACTLY 121 FEATURES
    # --------------------------------------------------------

    features = {}

    # 1
    features[
        "duration"
    ] = float(duration)

    # 6
    rms_names = [
        "rms_mean",
        "rms_std",
        "rms_median",
        "rms_min",
        "rms_max",
        "rms_p90",
    ]

    for name, value in zip(
        rms_names,
        rms_features,
    ):

        features[name] = value

    # 6
    zcr_names = [
        "zcr_mean",
        "zcr_std",
        "zcr_median",
        "zcr_min",
        "zcr_max",
        "zcr_p90",
    ]

    for name, value in zip(
        zcr_names,
        zcr_features,
    ):

        features[name] = value

    # 20
    spectral_groups = [
        (
            "spectral_centroid",
            centroid_features,
        ),
        (
            "spectral_bandwidth",
            bandwidth_features,
        ),
        (
            "spectral_rolloff",
            rolloff_features,
        ),
        (
            "spectral_flatness",
            flatness_features,
        ),
    ]

    spectral_suffixes = [
        "mean",
        "std",
        "median",
        "p10",
        "p90",
    ]

    for prefix, values in (
        spectral_groups
    ):

        for suffix, value in zip(
            spectral_suffixes,
            values,
        ):

            features[
                f"{prefix}_{suffix}"
            ] = value

    # 8
    f0_names = [
        "f0_mean",
        "f0_std",
        "f0_median",
        "f0_min",
        "f0_max",
        "f0_p10",
        "f0_p90",
        "f0_abs_change",
    ]

    for name, value in zip(
        f0_names,
        f0_features,
    ):

        features[name] = value

    # 4
    features[
        "voiced_ratio"
    ] = float(
        voiced_ratio
    )

    features[
        "silence_ratio"
    ] = float(
        silence_ratio
    )

    features[
        "speech_activity"
    ] = float(
        voiced_ratio
    )

    features[
        "silence_activity"
    ] = float(
        silence_ratio
    )

    # 8
    energy_names = [
        "energy_diff_mean",
        "energy_diff_std",
        "energy_abs_diff_mean",
        "energy_abs_diff_max",
        "energy_abs_diff_p90",
        "energy_rising_ratio",
        "energy_falling_ratio",
        "energy_above_mean_ratio",
    ]

    for name, value in zip(
        energy_names,
        energy_dynamic,
    ):

        features[name] = value

    # 8
    pause_names = [
        "pause_count",
        "pause_ratio",
        "longest_pause_frames",
        "pause_duration_mean",
        "pause_duration_max",
        "pause_duration_std",
        "silence_frames",
        "silence_spacing",
    ]

    for name, value in zip(
        pause_names,
        pause_stats,
    ):

        features[name] = value

    # 8
    spectral_dynamic_names = [
        "centroid_diff_mean",
        "centroid_diff_std",
        "bandwidth_diff_mean",
        "bandwidth_diff_std",
        "rolloff_diff_mean",
        "rolloff_diff_std",
        "flatness_diff_mean",
        "flatness_diff_std",
    ]

    for name, value in zip(
        spectral_dynamic_names,
        spectral_dynamic,
    ):

        features[name] = value

    # 13 + 13
    for i in range(
        N_MFCC
    ):

        features[
            f"mfcc_{i + 1}_mean"
        ] = float(
            mfcc_mean[i]
        )

        features[
            f"mfcc_{i + 1}_std"
        ] = float(
            mfcc_std[i]
        )

    # 13 + 13
    for i in range(
        N_MFCC
    ):

        features[
            f"delta_{i + 1}_mean"
        ] = float(
            delta_mean[i]
        )

        features[
            f"delta_{i + 1}_std"
        ] = float(
            delta_std[i]
        )

    # --------------------------------------------------------
    # FINAL COUNT CHECK
    # --------------------------------------------------------

    if len(features) != 121:

        raise RuntimeError(
            "Feature count error. "
            f"Expected 121, got {len(features)}."
        )

    return features


# ============================================================
# TEMPORAL WINDOW EXTRACTION
# ============================================================

def extract_clip_windows(
    audio_path,
):

    try:

        audio, sr = librosa.load(
            audio_path,
            sr=SR,
            mono=True,
        )

    except Exception:

        return []

    if len(audio) == 0:

        return []

    total_duration = (
        len(audio) / sr
    )

    window_samples = int(
        WINDOW_SECONDS * sr
    )

    hop_samples = int(
        HOP_SECONDS * sr
    )

    rows = []

    starts = list(
        range(
            0,
            max(
                1,
                len(audio)
                - window_samples
                + 1,
            ),
            hop_samples,
        )
    )

    # Ensure the final part is covered.
    if (
        len(audio) > window_samples
    ):

        final_start = (
            len(audio)
            - window_samples
        )

        if final_start not in starts:

            starts.append(
                final_start
            )

    starts = sorted(
        set(starts)
    )

    starts = starts[
        :MAX_WINDOWS_PER_CLIP
    ]

    for start_sample in starts:

        end_sample = min(
            start_sample
            + window_samples,
            len(audio),
        )

        y = audio[
            start_sample:end_sample
        ]

        feature_dict = (
            extract_121_features(
                y,
                sr,
            )
        )

        if feature_dict is None:

            continue

        start_sec = (
            start_sample / sr
        )

        end_sec = (
            end_sample / sr
        )

        feature_dict[
            "window_start"
        ] = float(
            start_sec
        )

        feature_dict[
            "window_end"
        ] = float(
            end_sec
        )

        feature_dict[
            "window_center"
        ] = float(
            (start_sec + end_sec)
            / 2.0
        )

        feature_dict[
            "clip_duration"
        ] = float(
            total_duration
        )

        rows.append(
            feature_dict
        )

    return rows


# ============================================================
# BUILD TEMPORAL DATASET
# ============================================================

def build_temporal_dataset(
    metadata,
    train_idx,
    val_idx,
    test_idx,
):

    banner(
        "[3/8] Building temporal window dataset"
    )

    split_lookup = {}

    for idx in train_idx:

        split_lookup[
            int(idx)
        ] = "train"

    for idx in val_idx:

        split_lookup[
            int(idx)
        ] = "validation"

    for idx in test_idx:

        split_lookup[
            int(idx)
        ] = "test"

    total = (
        len(train_idx)
        + len(val_idx)
        + len(test_idx)
    )

    print(
        f"Source clips : {total:,}"
    )

    print(
        f"Window size  : "
        f"{WINDOW_SECONDS:.2f}s"
    )

    print(
        f"Window hop   : "
        f"{HOP_SECONDS:.2f}s"
    )

    print()

    rows = []

    start_time = time.time()

    processed = 0

    failed = 0

    last_print = time.time()

    for row_index, row in metadata.iterrows():

        processed += 1

        audio_path = Path(
            str(
                row[
                    "audio_path"
                ]
            )
        )

        if not audio_path.exists():

            failed += 1

            continue

        windows = (
            extract_clip_windows(
                audio_path
            )
        )

        split_name = (
            split_lookup[
                int(row_index)
            ]
        )

        # ----------------------------------------------------
        # Weak labels
        #
        # Because the original label applies to the entire
        # 3-second clip, each temporal window inherits it.
        # ----------------------------------------------------

        for window in windows:

            record = dict(
                window
            )

            record[
                "audio_path"
            ] = str(
                audio_path
            )

            record[
                "filename"
            ] = str(
                row[
                    "filename"
                ]
            )

            record[
                "show"
            ] = str(
                row[
                    "show"
                ]
            )

            record[
                "episode_id"
            ] = str(
                row[
                    "episode_id"
                ]
            )

            record[
                "clip_id"
            ] = str(
                row[
                    "clip_id"
                ]
            )

            record[
                "split"
            ] = split_name

            record[
                "has_stutter"
            ] = int(
                pd.to_numeric(
                    row[
                        "has_stutter"
                    ],
                    errors="coerce",
                )
                if not pd.isna(
                    row[
                        "has_stutter"
                    ]
                )
                else 0
            )

            for event in EVENT_COLUMNS:

                record[
                    event
                ] = int(
                    pd.to_numeric(
                        row[
                            event
                        ],
                        errors="coerce",
                    )
                    > 0
                )

            rows.append(
                record
            )

        now = time.time()

        if (
            now - last_print
            >= 5.0
            or processed == total
        ):

            elapsed_time = (
                now - start_time
            )

            rate = (
                processed
                / max(
                    elapsed_time,
                    0.001,
                )
            )

            remaining = (
                total
                - processed
            )

            eta = (
                remaining
                / max(
                    rate,
                    0.001,
                )
            )

            print(
                f"\rProcessed "
                f"{processed:,}/{total:,} "
                f"clips "
                f"({processed / total * 100:6.2f}%) "
                f"| windows "
                f"{len(rows):,} "
                f"| failed "
                f"{failed:,} "
                f"| rate "
                f"{rate:.2f} clips/s "
                f"| ETA "
                f"{format_seconds(eta)}",
                end="",
                flush=True,
            )

            last_print = now

    print()

    if not rows:

        raise RuntimeError(
            "No temporal windows were generated."
        )

    temporal_df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Save cache
    # --------------------------------------------------------

    temporal_df.to_pickle(
        TEMPORAL_FEATURE_CACHE
    )

    print()
    print(
        f"[SAVED] {TEMPORAL_FEATURE_CACHE}"
    )

    print(
        f"Temporal rows : "
        f"{len(temporal_df):,}"
    )

    print(
        f"Columns       : "
        f"{len(temporal_df.columns):,}"
    )

    print(
        f"Failed clips  : "
        f"{failed:,}"
    )

    return temporal_df


# ============================================================
# LOAD / BUILD CACHE
# ============================================================

def load_or_build_temporal_dataset(
    metadata,
    train_idx,
    val_idx,
    test_idx,
):

    if TEMPORAL_FEATURE_CACHE.exists():

        banner(
            "[3/8] Loading V6 temporal feature cache"
        )

        print(
            f"Cache: {TEMPORAL_FEATURE_CACHE}"
        )

        df = pd.read_pickle(
            TEMPORAL_FEATURE_CACHE
        )

        print(
            f"Temporal rows : {len(df):,}"
        )

        print(
            f"Columns       : {len(df.columns):,}"
        )

        return df

    return build_temporal_dataset(
        metadata,
        train_idx,
        val_idx,
        test_idx,
    )


# ============================================================
# PREPARE MODEL MATRIX
# ============================================================

def prepare_model_matrix(
    temporal_df,
):

    banner(
        "[4/8] Preparing audio-only temporal features"
    )

    forbidden = {
        "audio_path",
        "filename",
        "show",
        "episode_id",
        "clip_id",
        "split",

        "window_start",
        "window_end",
        "window_center",
        "clip_duration",

        "has_stutter",
        *EVENT_COLUMNS,
    }

    feature_names = []

    for col in temporal_df.columns:

        if col in forbidden:

            continue

        if not pd.api.types.is_numeric_dtype(
            temporal_df[col]
        ):

            continue

        feature_names.append(
            col
        )

    # --------------------------------------------------------
    # Leakage safety
    # --------------------------------------------------------

    forbidden_words = [
        "stutter",
        "prolongation",
        "block",
        "soundrep",
        "wordrep",
        "interjection",
        "label",
        "event",
    ]

    suspicious = []

    for col in feature_names:

        lower = col.lower()

        for word in forbidden_words:

            if word in lower:

                suspicious.append(
                    col
                )

                break

    if suspicious:

        raise RuntimeError(
            "Potential label-derived feature detected:\n"
            + "\n".join(
                f"  - {x}"
                for x in suspicious
            )
        )

    X = temporal_df[
        feature_names
    ].copy()

    X = X.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    medians = X.median(
        numeric_only=True
    )

    X = X.fillna(
        medians
    )

    X = X.fillna(
        0.0
    )

    if X.isna().any().any():

        raise RuntimeError(
            "NaN values remain."
        )

    print(
        "[LEAKAGE CHECK] PASS"
    )

    print(
        "[OK] Only audio-derived numeric features."
    )

    print(
        f"Rows     : {len(X):,}"
    )

    print(
        f"Features : {len(feature_names):,}"
    )

    return (
        X,
        feature_names,
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

    sweep = []

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

        sweep.append(
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
            or f1 > best[
                "f1"
            ]
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
        pd.DataFrame(
            sweep
        ),
    )


# ============================================================
# MODEL CANDIDATES
# ============================================================

def detector_models():

    return {

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=2,
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=2,
        ),

    }


# ============================================================
# TRAIN DETECTOR
# ============================================================

def train_detector(
    temporal_df,
    X,
    feature_names,
):

    banner(
        "[5/8] Training temporal stutter detector"
    )

    train_mask = (
        temporal_df[
            "split"
        ]
        == "train"
    )

    val_mask = (
        temporal_df[
            "split"
        ]
        == "validation"
    )

    test_mask = (
        temporal_df[
            "split"
        ]
        == "test"
    )

    X_train = X.loc[
        train_mask
    ]

    X_val = X.loc[
        val_mask
    ]

    X_test = X.loc[
        test_mask
    ]

    y_train = binary_label(
        temporal_df.loc[
            train_mask,
            "has_stutter",
        ]
    )

    y_val = binary_label(
        temporal_df.loc[
            val_mask,
            "has_stutter",
        ]
    )

    y_test = binary_label(
        temporal_df.loc[
            test_mask,
            "has_stutter",
        ]
    )

    print(
        f"Train windows : {len(X_train):,}"
    )

    print(
        f"Val windows   : {len(X_val):,}"
    )

    print(
        f"Test windows  : {len(X_test):,}"
    )

    candidates = {}

    for name, model in detector_models().items():

        print()
        print(
            f"[MODEL] {name}"
        )

        start_time = time.time()

        model.fit(
            X_train,
            y_train,
        )

        print(
            f"Training time: "
            f"{elapsed(start_time):.2f}s"
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
            f"Threshold           : "
            f"{threshold:.2f}"
        )

        print(
            f"Validation macro F1 : "
            f"{val_f1:.4f}"
        )

        print(
            f"Test macro F1       : "
            f"{test_f1:.4f}"
        )

        candidates[name] = {
            "model": model,
            "threshold": threshold,
            "val_f1": val_f1,
            "test_f1": test_f1,
            "test_pred": test_pred,
            "test_prob": test_prob,
            "sweep": sweep,
        }

    best_name = max(
        candidates,
        key=lambda name:
        candidates[
            name
        ][
            "val_f1"
        ],
    )

    best = candidates[
        best_name
    ]

    print()
    print(
        f"[BEST TEMPORAL DETECTOR] "
        f"{best_name}"
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
    # Permanent model
    # --------------------------------------------------------

    model_path = (
        MODELS_DIR
        / "stutter_temporal_detector_v6.joblib"
    )

    joblib.dump(
        {
            "model": best[
                "model"
            ],
            "threshold": best[
                "threshold"
            ],
            "model_name": best_name,
            "version": "v6",
            "feature_names": feature_names,
            "window_seconds": WINDOW_SECONDS,
            "hop_seconds": HOP_SECONDS,
            "feature_count": len(
                feature_names
            ),
            "feature_policy": "audio_only",
            "weak_supervision": True,
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

    report = classification_report(
        y_test,
        best[
            "test_pred"
        ],
        output_dict=True,
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_test,
        best[
            "test_pred"
        ],
    )

    save_json(
        RESULTS_DIR
        / "stutter_v6_temporal_detector_report.json",
        {
            "version": "v6",
            "model": best_name,
            "threshold": float(
                best[
                    "threshold"
                ]
            ),
            "validation_macro_f1": float(
                best[
                    "val_f1"
                ]
            ),
            "test_macro_f1": float(
                best[
                    "test_f1"
                ]
            ),
            "test_accuracy": float(
                accuracy_score(
                    y_test,
                    best[
                        "test_pred"
                    ],
                )
            ),
            "classification_report": report,
            "feature_count": len(
                feature_names
            ),
            "window_seconds": WINDOW_SECONDS,
            "hop_seconds": HOP_SECONDS,
            "weak_supervision": True,
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
        / "stutter_v6_temporal_detector_confusion.csv"
    )

    best[
        "sweep"
    ].to_csv(
        RESULTS_DIR
        / "stutter_v6_temporal_detector_thresholds.csv",
        index=False,
    )

    return best


# ============================================================
# TRAIN EVENT MODELS
# ============================================================

def train_event_models(
    temporal_df,
    X,
    feature_names,
):

    banner(
        "[6/8] Training temporal event models"
    )

    train_mask = (
        temporal_df[
            "split"
        ]
        == "train"
    )

    val_mask = (
        temporal_df[
            "split"
        ]
        == "validation"
    )

    test_mask = (
        temporal_df[
            "split"
        ]
        == "test"
    )

    X_train = X.loc[
        train_mask
    ]

    X_val = X.loc[
        val_mask
    ]

    X_test = X.loc[
        test_mask
    ]

    results = {}

    prediction_df = temporal_df.loc[
        test_mask,
        [
            "audio_path",
            "filename",
            "show",
            "episode_id",
            "clip_id",
            "window_start",
            "window_end",
            "window_center",
        ],
    ].copy()

    for event in EVENT_COLUMNS:

        print()
        print(
            "-" * 72
        )

        print(
            f"EVENT: {event}"
        )

        y_train = binary_label(
            temporal_df.loc[
                train_mask,
                event,
            ]
        )

        y_val = binary_label(
            temporal_df.loc[
                val_mask,
                event,
            ]
        )

        y_test = binary_label(
            temporal_df.loc[
                test_mask,
                event,
            ]
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

        for name, model in detector_models().items():

            print()
            print(
                f"[MODEL] {name}"
            )

            start_time = time.time()

            model.fit(
                X_train,
                y_train,
            )

            print(
                f"Training time: "
                f"{elapsed(start_time):.2f}s"
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

            candidates[name] = {
                "model": model,
                "threshold": threshold,
                "val_f1": val_f1,
                "test_f1": test_f1,
                "test_pred": test_pred,
                "test_prob": test_prob,
                "sweep": sweep,
            }

        best_name = max(
            candidates,
            key=lambda name:
            candidates[
                name
            ][
                "val_f1"
            ],
        )

        best = candidates[
            best_name
        ]

        print()
        print(
            f"[BEST] {event} -> "
            f"{best_name}"
        )

        print(
            f"Validation F1 : "
            f"{best['val_f1']:.4f}"
        )

        print(
            f"Test F1       : "
            f"{best['test_f1']:.4f}"
        )

        safe_name = (
            event.lower()
        )

        model_path = (
            MODELS_DIR
            / (
                "stutter_v6_event_"
                f"{safe_name}.joblib"
            )
        )

        joblib.dump(
            {
                "model": best[
                    "model"
                ],
                "threshold": best[
                    "threshold"
                ],
                "event": event,
                "model_name": best_name,
                "version": "v6",
                "feature_names": feature_names,
                "window_seconds": WINDOW_SECONDS,
                "hop_seconds": HOP_SECONDS,
                "feature_count": len(
                    feature_names
                ),
                "feature_policy": "audio_only",
                "weak_supervision": True,
                "random_state": RANDOM_STATE,
            },
            model_path,
        )

        print(
            f"[SAVED] {model_path}"
        )

        report = classification_report(
            y_test,
            best[
                "test_pred"
            ],
            output_dict=True,
            zero_division=0,
        )

        confusion = confusion_matrix(
            y_test,
            best[
                "test_pred"
            ],
        )

        results[event] = {
            "event": event,
            "model": best_name,
            "threshold": float(
                best[
                    "threshold"
                ]
            ),
            "validation_f1": float(
                best[
                    "val_f1"
                ]
            ),
            "test_f1": float(
                best[
                    "test_f1"
                ]
            ),
            "test_precision": float(
                precision_score(
                    y_test,
                    best[
                        "test_pred"
                    ],
                    zero_division=0,
                )
            ),
            "test_recall": float(
                recall_score(
                    y_test,
                    best[
                        "test_pred"
                    ],
                    zero_division=0,
                )
            ),
            "classification_report": report,
        }

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
        ] = best[
            "test_pred"
        ]

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
                f"stutter_v6_"
                f"{safe_name}_confusion.csv"
            )
        )

        best[
            "sweep"
        ].to_csv(
            RESULTS_DIR
            / (
                f"stutter_v6_"
                f"{safe_name}_thresholds.csv"
            ),
            index=False,
        )

    prediction_path = (
        RESULTS_DIR
        / "stutter_v6_temporal_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    print()
    print(
        f"[SAVED] {prediction_path}"
    )

    return results


# ============================================================
# TEMPORAL REGION MERGING
# ============================================================

def merge_regions(
    windows,
    threshold,
):

    detected = [
        row
        for row in windows
        if row[
            "probability"
        ] >= threshold
    ]

    if not detected:

        return []

    detected = sorted(
        detected,
        key=lambda row:
        row[
            "start"
        ],
    )

    regions = []

    current = dict(
        detected[0]
    )

    for row in detected[1:]:

        gap = (
            row["start"]
            - current["end"]
        )

        if gap <= MERGE_GAP_SECONDS:

            current[
                "end"
            ] = max(
                current["end"],
                row["end"],
            )

            current[
                "max_probability"
            ] = max(
                current[
                    "max_probability"
                ],
                row[
                    "probability"
                ],
            )

            current[
                "mean_probability"
            ] = (
                current[
                    "mean_probability"
                ]
                + row[
                    "probability"
                ]
            ) / 2.0

            current[
                "window_count"
            ] += 1

        else:

            regions.append(
                current
            )

            current = dict(
                row
            )

    regions.append(
        current
    )

    return regions


# ============================================================
# SAVE TEMPORAL DEMO TIMELINE
# ============================================================

def build_demo_timeline(
    temporal_df,
    detector,
    event_results,
):

    banner(
        "[7/8] Building temporal candidate timeline"
    )

    # --------------------------------------------------------
    # Select test windows
    # --------------------------------------------------------

    test_df = temporal_df[
        temporal_df[
            "split"
        ]
        == "test"
    ].copy()

    # We create a representative timeline from the
    # first test clip so the output remains compact.
    # The full predictions are already saved separately.
    # --------------------------------------------------------

    if test_df.empty:

        return

    first_clip = (
        test_df[
            [
                "audio_path",
                "filename",
                "show",
                "episode_id",
                "clip_id",
            ]
        ]
        .drop_duplicates()
        .iloc[0]
    )

    clip_path = (
        first_clip[
            "audio_path"
        ]
    )

    clip_df = test_df[
        test_df[
            "audio_path"
        ]
        == clip_path
    ].copy()

    threshold = detector[
        "threshold"
    ]

    # --------------------------------------------------------
    # Reconstruct probabilities
    # --------------------------------------------------------

    feature_names = None

    detector_model = detector[
        "model"
    ]

    # We cannot directly use the original X here because
    # this function intentionally stays compact.
    #
    # The full prediction table is already saved in
    # stutter_v6_temporal_predictions.csv.
    #
    # So use that file.
    # --------------------------------------------------------

    prediction_path = (
        RESULTS_DIR
        / "stutter_v6_temporal_predictions.csv"
    )

    predictions = pd.read_csv(
        prediction_path
    )

    predictions = predictions[
        predictions[
            "audio_path"
        ]
        == clip_path
    ].copy()

    if predictions.empty:

        return

    # --------------------------------------------------------
    # Detector timeline cannot be recovered from the CSV
    # because feature inference values aren't stored there.
    #
    # Instead create an event timeline from the event
    # probability columns.
    # --------------------------------------------------------

    timeline = []

    for _, row in predictions.iterrows():

        event_candidates = []

        for event in EVENT_COLUMNS:

            prob = float(
                row[
                    f"prob_{event}"
                ]
            )

            event_candidates.append(
                (
                    event,
                    prob,
                )
            )

        event_candidates.sort(
            key=lambda x:
            x[1],
            reverse=True,
        )

        best_event = (
            event_candidates[0]
        )

        timeline.append(
            {
                "start": float(
                    row[
                        "window_start"
                    ]
                ),
                "end": float(
                    row[
                        "window_end"
                    ]
                ),
                "top_event": best_event[0],
                "top_event_probability": best_event[1],
            }
        )

    output = {
        "version": "v6",
        "weak_supervision": True,
        "description": (
            "Candidate temporal event timeline "
            "from weakly-supervised clip labels."
        ),
        "audio_path": clip_path,
        "windows": timeline,
    }

    save_json(
        RESULTS_DIR
        / "stutter_v6_demo_timeline.json",
        output,
    )

    print(
        "[SAVED] "
        "stutter_v6_demo_timeline.json"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def save_summary(
    temporal_df,
    feature_names,
    detector,
    event_results,
    train_idx,
    val_idx,
    test_idx,
):

    banner(
        "[8/8] Saving V6 summary"
    )

    summary = {

        "version": "v6",

        "architecture": (
            "weakly_supervised_temporal"
        ),

        "feature_policy": (
            "audio_only"
        ),

        "feature_count": int(
            len(feature_names)
        ),

        "window_seconds": (
            WINDOW_SECONDS
        ),

        "hop_seconds": (
            HOP_SECONDS
        ),

        "temporal_rows": int(
            len(temporal_df)
        ),

        "detector": {
            "model": (
                detector[
                    "model_name"
                ]
                if "model_name"
                in detector
                else "selected_by_validation"
            ),
            "threshold": float(
                detector[
                    "threshold"
                ]
            ),
            "test_macro_f1": float(
                detector[
                    "test_f1"
                ]
            ),
        },

        "events": event_results,

        "source_split": {
            "train_clips": int(
                len(train_idx)
            ),
            "validation_clips": int(
                len(val_idx)
            ),
            "test_clips": int(
                len(test_idx)
            ),
        },

        "leakage_policy": {

            "audio_only": True,

            "annotation_features": False,

            "episode_safe": True,

            "test_used_for_selection": False,

            "thresholds_selected_on_validation": True,
        },

        "label_policy": {

            "clip_level_labels": True,

            "window_labels_are_weak": True,

            "exact_event_boundaries_available": False,
        },
    }

    path = (
        RESULTS_DIR
        / "stutter_v6_summary.json"
    )

    save_json(
        path,
        summary,
    )

    print(
        f"[SAVED] {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    banner(
        "ORATORIQ STUTTER V6"
    )

    print(
        "Weakly-supervised temporal fluency pipeline"
    )

    print()
    print(
        "121 audio-only features"
    )

    print(
        "1.0s windows / 0.5s hop"
    )

    print(
        "Episode-safe split"
    )

    print(
        "Validation-only model selection"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "SEP-28k labels are clip-level."
    )

    print(
        "V6 therefore produces candidate temporal regions,"
    )

    print(
        "not ground-truth stutter boundaries."
    )

    # --------------------------------------------------------
    # 1
    # --------------------------------------------------------

    metadata = load_metadata()

    # --------------------------------------------------------
    # 2
    # --------------------------------------------------------

    (
        train_idx,
        val_idx,
        test_idx,
    ) = create_episode_split(
        metadata
    )

    # --------------------------------------------------------
    # 3
    # --------------------------------------------------------

    temporal_df = (
        load_or_build_temporal_dataset(
            metadata,
            train_idx,
            val_idx,
            test_idx,
        )
    )

    # --------------------------------------------------------
    # Validate cache
    # --------------------------------------------------------

    required_temporal_columns = [
        "audio_path",
        "filename",
        "show",
        "episode_id",
        "clip_id",
        "split",
        "window_start",
        "window_end",
        "window_center",
        "has_stutter",
        *EVENT_COLUMNS,
    ]

    missing = [
        col
        for col in required_temporal_columns
        if col not in temporal_df.columns
    ]

    if missing:

        raise RuntimeError(
            "Temporal cache missing columns:\n"
            + "\n".join(
                f"  - {x}"
                for x in missing
            )
        )

    # --------------------------------------------------------
    # 4
    # --------------------------------------------------------

    (
        X,
        feature_names,
    ) = prepare_model_matrix(
        temporal_df
    )

    # --------------------------------------------------------
    # 5
    # --------------------------------------------------------

    detector = train_detector(
        temporal_df,
        X,
        feature_names,
    )

    # --------------------------------------------------------
    # 6
    # --------------------------------------------------------

    event_results = train_event_models(
        temporal_df,
        X,
        feature_names,
    )

    # --------------------------------------------------------
    # 7
    # --------------------------------------------------------

    build_demo_timeline(
        temporal_df,
        detector,
        event_results,
    )

    # --------------------------------------------------------
    # 8
    # --------------------------------------------------------

    save_summary(
        temporal_df,
        feature_names,
        detector,
        event_results,
        train_idx,
        val_idx,
        test_idx,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    banner(
        "ORATORIQ STUTTER V6 COMPLETE"
    )

    print(
        f"Temporal rows : "
        f"{len(temporal_df):,}"
    )

    print(
        f"Features      : "
        f"{len(feature_names):,}"
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
        "MODELS:"
    )

    print(
        MODELS_DIR
    )

    print()
    print(
        "RESULTS:"
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
        "=" * 80
    )


if __name__ == "__main__":
    main()