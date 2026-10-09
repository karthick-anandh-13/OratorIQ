from pathlib import Path
import argparse
import json
import warnings

import joblib
import librosa
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\OratorIQ")

ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models"
RESULTS = ARTIFACTS / "results"

TEMPORAL_MODEL = MODELS / "contrastive_temporal_detector.joblib"

# Original 6-class model
FLAW_MODEL = MODELS / "contrastive_flaw_type_classifier.joblib"

# NEW V2.1 family model
FAMILY_MODEL = MODELS / "contrastive_v21_flaw_family_classifier.joblib"

# Original severity model
SEVERITY_MODEL = MODELS / "contrastive_severity_classifier.joblib"

OUTPUT_JSON = RESULTS / "inference_hybrid_result.json"


# ============================================================
# CONFIG
# ============================================================

SR = 16000
N_MFCC = 13

WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.5

MERGE_GAP_SECONDS = 0.75

TEMPORAL_THRESHOLD = 0.50


# ============================================================
# FAMILY DEFINITIONS
# ============================================================

TEMPORAL_FLAWS = {
    "fast_pacing",
    "slow_pacing",
    "long_pause",
}

ACOUSTIC_FLAWS = {
    "high_volume",
    "low_volume",
    "pitch_deviation",
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

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

    duration = len(y) / sr

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=y,
        frame_length=min(1024, len(y)),
        hop_length=256
    )[0]

    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    rms_max = float(np.max(rms)) if len(rms) else 0.0

    # --------------------------------------------------------
    # ZCR
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        y=y,
        frame_length=min(1024, len(y)),
        hop_length=256
    )[0]

    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=min(1024, len(y)),
        hop_length=256
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # --------------------------------------------------------
    # MFCC DELTA
    # --------------------------------------------------------

    if mfcc.shape[1] >= 3:

        delta = librosa.feature.delta(mfcc)

        delta_mean = np.mean(delta, axis=1)
        delta_std = np.std(delta, axis=1)

    else:

        delta_mean = np.zeros(N_MFCC)
        delta_std = np.zeros(N_MFCC)

    # --------------------------------------------------------
    # SPECTRAL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # F0
    # --------------------------------------------------------

    try:

        f0 = librosa.yin(
            y=y,
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

            f0_mean = 0.0
            f0_std = 0.0
            f0_min = 0.0
            f0_max = 0.0

    except Exception:

        f0_mean = 0.0
        f0_std = 0.0
        f0_min = 0.0
        f0_max = 0.0

    # --------------------------------------------------------
    # SILENCE
    # --------------------------------------------------------

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

        voiced_ratio = 0.0
        silence_ratio = 1.0

    # --------------------------------------------------------
    # GLOBAL WPM
    # --------------------------------------------------------

    words = str(transcript).split()

    word_count = len(words)

    global_wpm = (
        word_count / max(total_duration, 0.1)
    ) * 60.0

    # --------------------------------------------------------
    # BUILD FEATURE VECTOR
    # --------------------------------------------------------

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


# ============================================================
# AUDIO WINDOWS
# ============================================================

def extract_audio_windows(audio_path, transcript=""):

    print()
    print("=" * 70)
    print("EXTRACTING AUDIO FEATURES")
    print("=" * 70)

    print(f"Audio: {audio_path}")

    audio, sr = librosa.load(
        audio_path,
        sr=SR,
        mono=True
    )

    total_duration = len(audio) / sr

    print(f"Sample rate: {sr}")
    print(f"Duration: {total_duration:.2f}s")

    window_samples = int(
        WINDOW_SECONDS * sr
    )

    hop_samples = int(
        HOP_SECONDS * sr
    )

    rows = []

    starts = range(
        0,
        max(
            1,
            len(audio) - window_samples + 1
        ),
        hop_samples
    )

    starts = list(starts)

    total = len(starts)

    for index, start_sample in enumerate(
        starts,
        start=1
    ):

        end_sample = min(
            start_sample + window_samples,
            len(audio)
        )

        features = extract_features(
            audio,
            sr,
            start_sample,
            end_sample,
            transcript,
            total_duration
        )

        if features is None:
            continue

        start_time = start_sample / sr
        end_time = end_sample / sr

        features["window_start"] = start_time
        features["window_end"] = end_time

        rows.append(features)

        if index == total or index % 25 == 0:

            print(
                f"\rProcessing windows: "
                f"{index}/{total}",
                end="",
                flush=True
            )

    print()

    df = pd.DataFrame(rows)

    print(
        f"Extracted windows: {len(df):,}"
    )

    print(
        f"Acoustic features: "
        f"{len(df.columns) - 2}"
    )

    return df


# ============================================================
# ORIGINAL 138 CONTRASTIVE FEATURES
# ============================================================

def build_contrastive_features(
    flawed_df,
    good_df
):

    print()
    print("=" * 70)
    print("BUILDING CONTRASTIVE FEATURES")
    print("=" * 70)

    excluded = {
        "window_start",
        "window_end"
    }

    feature_columns = []

    for column in flawed_df.columns:

        if column in excluded:
            continue

        if pd.api.types.is_numeric_dtype(
            flawed_df[column]
        ):
            feature_columns.append(column)

    print(
        f"Base acoustic features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Recording durations
    # --------------------------------------------------------

    good_duration = float(
        good_df["window_end"].max()
    )

    flawed_duration = float(
        flawed_df["window_end"].max()
    )

    good_df = good_df.copy()

    good_df["_relative_position"] = (
        (
            good_df["window_start"]
            + good_df["window_end"]
        ) / 2.0
    ) / max(good_duration, 1e-6)

    results = []

    total = len(flawed_df)

    for counter, (_, row) in enumerate(
        flawed_df.iterrows(),
        start=1
    ):

        flawed_center = (
            row["window_start"]
            + row["window_end"]
        ) / 2.0

        relative_position = (
            flawed_center
            / max(flawed_duration, 1e-6)
        )

        distances = np.abs(
            good_df["_relative_position"].values
            - relative_position
        )

        nearest_index = np.argmin(
            distances
        )

        good_row = good_df.iloc[
            nearest_index
        ]

        output = {}

        output["window_start"] = (
            row["window_start"]
        )

        output["window_end"] = (
            row["window_end"]
        )

        output["relative_position"] = (
            relative_position
        )

        # ----------------------------------------------------
        # Original 69 delta features
        # ----------------------------------------------------

        for feature in feature_columns:

            flawed_value = float(
                row[feature]
            )

            good_value = float(
                good_row[feature]
            )

            delta = (
                flawed_value
                - good_value
            )

            relative_delta = (
                delta
                / (
                    abs(good_value)
                    + 1e-6
                )
            )

            output[
                f"delta_{feature}"
            ] = delta

            output[
                f"relative_delta_{feature}"
            ] = relative_delta

        # ----------------------------------------------------
        # V2.1 deployment-safe features
        # ----------------------------------------------------

        good_window_start = float(
            good_row["window_start"]
        )

        good_window_end = float(
            good_row["window_end"]
        )

        flawed_window_start = float(
            row["window_start"]
        )

        flawed_window_end = float(
            row["window_end"]
        )

        good_window_duration = (
            good_window_end
            - good_window_start
        )

        flawed_window_duration = (
            flawed_window_end
            - flawed_window_start
        )

        # Recording-level timing
        output["flawed_recording_duration"] = (
            flawed_duration
        )

        output["good_recording_duration"] = (
            good_duration
        )

        output["recording_duration_delta"] = (
            flawed_duration
            - good_duration
        )

        output["recording_duration_ratio"] = (
            flawed_duration
            / max(good_duration, 1e-6)
        )

        # Window timing
        output["flawed_window_duration"] = (
            flawed_window_duration
        )

        output["good_window_duration"] = (
            good_window_duration
        )

        output["window_duration_delta"] = (
            flawed_window_duration
            - good_window_duration
        )

        output["window_duration_ratio"] = (
            flawed_window_duration
            / max(good_window_duration, 1e-6)
        )

        output["window_compression_ratio"] = (
            flawed_window_duration
            / max(
                good_window_duration,
                1e-6
            )
        )

        # Alignment
        output["alignment_error"] = (
            abs(
                relative_position
                - float(
                    good_row["_relative_position"]
                )
            )
        )

        # Relative start/end
        flawed_relative_start = (
            flawed_window_start
            / max(flawed_duration, 1e-6)
        )

        flawed_relative_end = (
            flawed_window_end
            / max(flawed_duration, 1e-6)
        )

        good_relative_start = (
            good_window_start
            / max(good_duration, 1e-6)
        )

        good_relative_end = (
            good_window_end
            / max(good_duration, 1e-6)
        )

        output["flawed_relative_start"] = (
            flawed_relative_start
        )

        output["flawed_relative_end"] = (
            flawed_relative_end
        )

        output["good_relative_start"] = (
            good_relative_start
        )

        output["good_relative_end"] = (
            good_relative_end
        )

        output["relative_position_delta"] = (
            relative_position
            - float(
                good_row["_relative_position"]
            )
        )

        # ----------------------------------------------------
        # Temporal acoustic features
        # ----------------------------------------------------

        def delta_and_ratio(
            flawed_value,
            good_value,
            delta_name,
            ratio_name
        ):

            output[delta_name] = (
                flawed_value
                - good_value
            )

            output[ratio_name] = (
                flawed_value
                / max(
                    abs(good_value),
                    1e-6
                )
            )

        # V2.1 deployment-safe feature:
        # keep RMS delta only; temporal_rms_ratio was removed.
        output["temporal_rms_delta"] = (
            float(row["rms_mean"])
            - float(good_row["rms_mean"])
        )

        delta_and_ratio(
            float(row["silence_ratio"]),
            float(good_row["silence_ratio"]),
            "temporal_silence_delta",
            "temporal_silence_ratio"
        )

        delta_and_ratio(
            float(row["f0_mean"]),
            float(good_row["f0_mean"]),
            "temporal_f0_delta",
            "temporal_f0_ratio"
        )

        delta_and_ratio(
            float(row["global_wpm"]),
            float(good_row["global_wpm"]),
            "temporal_wpm_delta",
            "temporal_wpm_ratio"
        )

        # ----------------------------------------------------
        # Previous / next acoustic context
        #
        # IMPORTANT:
        # These are NOT labels.
        # They are calculated from neighboring acoustic
        # windows only.
        # ----------------------------------------------------

        output["_flawed_rms"] = float(
            row["rms_mean"]
        )

        output["_good_rms"] = float(
            good_row["rms_mean"]
        )

        output["_flawed_f0"] = float(
            row["f0_mean"]
        )

        output["_good_f0"] = float(
            good_row["f0_mean"]
        )

        results.append(output)

        if counter == total or counter % 25 == 0:

            print(
                f"\rContrastive windows: "
                f"{counter}/{total}",
                end="",
                flush=True
            )

    print()

    result_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Neighbor acoustic features
    # --------------------------------------------------------

    for index in range(len(result_df)):

        if index > 0:

            result_df.loc[
                index,
                "previous_rms_delta_from_current"
            ] = (
                result_df.loc[
                    index,
                    "_flawed_rms"
                ]
                -
                result_df.loc[
                    index - 1,
                    "_flawed_rms"
                ]
            )

            result_df.loc[
                index,
                "previous_f0_delta_from_current"
            ] = (
                result_df.loc[
                    index,
                    "_flawed_f0"
                ]
                -
                result_df.loc[
                    index - 1,
                    "_flawed_f0"
                ]
            )

        else:

            result_df.loc[
                index,
                "previous_rms_delta_from_current"
            ] = 0.0

            result_df.loc[
                index,
                "previous_f0_delta_from_current"
            ] = 0.0

        if index < len(result_df) - 1:

            result_df.loc[
                index,
                "next_rms_delta_from_current"
            ] = (
                result_df.loc[
                    index + 1,
                    "_flawed_rms"
                ]
                -
                result_df.loc[
                    index,
                    "_flawed_rms"
                ]
            )

            result_df.loc[
                index,
                "next_f0_delta_from_current"
            ] = (
                result_df.loc[
                    index + 1,
                    "_flawed_f0"
                ]
                -
                result_df.loc[
                    index,
                    "_flawed_f0"
                ]
            )

        else:

            result_df.loc[
                index,
                "next_rms_delta_from_current"
            ] = 0.0

            result_df.loc[
                index,
                "next_f0_delta_from_current"
            ] = 0.0

    # Remove internal helper columns
    result_df = result_df.drop(
        columns=[
            "_flawed_rms",
            "_good_rms",
            "_flawed_f0",
            "_good_f0"
        ],
        errors="ignore"
    )

    # Replace numerical NaN/inf
    numeric_columns = result_df.select_dtypes(
        include=[np.number]
    ).columns

    result_df[numeric_columns] = (
        result_df[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0.0)
    )

    return result_df


# ============================================================
# LOAD MODELS
# ============================================================

def unwrap_model(model_object, model_name):
    """
    Models may be saved directly or inside a dictionary.

    Returns:
        (sklearn_model, metadata_dict)
    """
    if hasattr(model_object, "predict"):
        print(f"[OK] {model_name}: loaded direct model")
        return model_object, {}

    if isinstance(model_object, dict):
        print(f"[INFO] {model_name}: saved as dictionary")
        print(f"[INFO] Dictionary keys: {list(model_object.keys())}")

        possible_model_keys = [
            "model",
            "classifier",
            "estimator",
            "best_model",
            "family_model",
            "trained_model",
        ]

        for key in possible_model_keys:
            if key in model_object and hasattr(
                model_object[key], "predict"
            ):
                print(f"[OK] Using dictionary key: '{key}'")
                return model_object[key], model_object

        for key, value in model_object.items():
            if hasattr(value, "predict"):
                print(f"[OK] Found model under dictionary key: '{key}'")
                return value, model_object

        raise TypeError(
            f"Could not find a trained sklearn model inside "
            f"{model_name}. Dictionary keys: "
            f"{list(model_object.keys())}"
        )

    raise TypeError(
        f"Unsupported {model_name} object type: "
        f"{type(model_object)}"
    )


def load_models():
    print()
    print("=" * 70)
    print("LOADING ORATORIQ HYBRID MODELS")
    print("=" * 70)

    paths = [
        TEMPORAL_MODEL,
        FLAW_MODEL,
        FAMILY_MODEL,
        SEVERITY_MODEL,
    ]

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found:\n{path}"
            )

    temporal_raw = joblib.load(TEMPORAL_MODEL)
    flaw_raw = joblib.load(FLAW_MODEL)
    family_raw = joblib.load(FAMILY_MODEL)
    severity_raw = joblib.load(SEVERITY_MODEL)

    temporal_model, temporal_meta = unwrap_model(
        temporal_raw, "Temporal model"
    )
    flaw_model, flaw_meta = unwrap_model(
        flaw_raw, "Flaw-type model"
    )
    family_model, family_meta = unwrap_model(
        family_raw, "Family model"
    )
    severity_model, severity_meta = unwrap_model(
        severity_raw, "Severity model"
    )

    print()
    print("[MODEL INFO]")
    for name, model in [
        ("Temporal", temporal_model),
        ("Flaw type", flaw_model),
        ("Family", family_model),
        ("Severity", severity_model),
    ]:
        n = getattr(model, "n_features_in_", None)
        classes = getattr(model, "classes_", None)
        print(
            f"{name:12s} | "
            f"features={n if n is not None else 'unknown'} | "
            f"classes={list(classes) if classes is not None else 'unknown'}"
        )

    print()
    print("[OK] All models loaded successfully.")

    return (
        temporal_model,
        flaw_model,
        family_model,
        severity_model,
        temporal_meta,
        flaw_meta,
        family_meta,
        severity_meta,
    )


def _find_saved_feature_names(meta):
    """Find a saved feature-name list in a dictionary artifact."""
    if not isinstance(meta, dict):
        return None

    for key in [
        "model_features",
        "feature_names",
        "features",
        "feature_columns",
        "columns",
    ]:
        value = meta.get(key)
        if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
            values = [str(x) for x in value]
            if values:
                return values

    return None


def prepare_model_features(
    contrastive_df,
    model,
    metadata=None,
    model_name="model",
):
    """
    Prepare exactly the feature set expected by the trained estimator.

    Priority:
      1. estimator.feature_names_in_
      2. saved artifact feature-name metadata
      3. generated feature order, only if the model's feature count matches
    """
    excluded = {
        "window_start",
        "window_end",
        "relative_position",
    }

    columns = [
        column
        for column in contrastive_df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            contrastive_df[column]
        )
    ]

    X = contrastive_df[columns].copy()

    expected = getattr(model, "feature_names_in_", None)

    if expected is not None:
        expected = [str(x) for x in expected]
        source = "model.feature_names_in_"
    else:
        expected = _find_saved_feature_names(metadata)
        source = "saved artifact metadata"

    if expected is not None:
        missing = [
            column
            for column in expected
            if column not in X.columns
        ]

        if missing:
            raise ValueError(
                "\n"
                f"{model_name.upper()} FEATURE MISMATCH\n"
                f"Missing features ({len(missing)}):\n"
                + "\n".join(
                    f"  - {x}" for x in missing
                )
                + "\n\n"
                f"Generated numeric features: {len(X.columns)}\n"
                f"Expected features: {len(expected)}\n"
                f"Feature source: {source}"
            )

        X = X[expected]

    else:
        expected_count = getattr(
            model,
            "n_features_in_",
            None
        )

        if (
            expected_count is not None
            and len(X.columns) != expected_count
        ):
            raise ValueError(
                "\n"
                f"{model_name.upper()} FEATURE COUNT MISMATCH\n"
                f"Generated: {len(X.columns)}\n"
                f"Model expects: {expected_count}\n"
                "The estimator exposes no feature names, so "
                "automatic reordering is unsafe."
            )

        print(
            f"[WARNING] {model_name} has no saved feature names. "
            f"Using generated order ({len(X.columns)} features)."
        )

    X = (
        X.replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0.0)
    )

    return X


def get_flawed_probability(
    model,
    X
):

    probabilities = (
        model.predict_proba(X)
    )

    classes = list(
        model.classes_
    )

    print(
        f"Temporal model classes: "
        f"{classes}"
    )

    if "flawed" in classes:

        index = classes.index(
            "flawed"
        )

        return probabilities[
            :,
            index
        ]

    if 1 in classes:

        index = classes.index(1)

        return probabilities[
            :,
            index
        ]

    raise ValueError(
        "Could not find flawed class "
        f"in {classes}"
    )


# ============================================================
# PROBABILITY HELPER
# ============================================================

def probability_dict(
    model,
    X,
    row_index
):

    probabilities = (
        model.predict_proba(X)
    )

    classes = list(
        model.classes_
    )

    return {
        str(classes[i]):
        float(probabilities[row_index, i])
        for i in range(
            len(classes)
        )
    }


# ============================================================
# FAMILY PREDICTION
# ============================================================

def predict_family(
    family_model,
    family_X,
    family_meta=None
):
    """
    Predict flaw family and convert encoded numeric classes
    into human-readable labels.

    The saved family model contains:
        - label_encoder
        - family_mapping

    We prefer the saved mapping/encoder rather than assuming
    that 0 or 1 corresponds to a particular family.
    """

    probabilities = family_model.predict_proba(
        family_X
    )

    classes = list(
        family_model.classes_
    )

    raw_predictions = family_model.predict(
        family_X
    )

    # --------------------------------------------------------
    # Find the saved mapping
    # --------------------------------------------------------

    family_mapping = None
    label_encoder = None

    if isinstance(family_meta, dict):

        family_mapping = family_meta.get(
            "family_mapping"
        )

        label_encoder = family_meta.get(
            "label_encoder"
        )

    # --------------------------------------------------------
    # Convert predictions to readable labels
    # --------------------------------------------------------

    predictions = []

    for prediction in raw_predictions:

        label = None

        # ----------------------------------------------------
        # First: label encoder
        # ----------------------------------------------------

        if label_encoder is not None:

            try:

                label = label_encoder.inverse_transform(
                    [int(prediction)]
                )[0]

            except Exception:

                label = None

        # ----------------------------------------------------
        # Second: explicit family mapping
        # ----------------------------------------------------

        if label is None and isinstance(
            family_mapping,
            dict
        ):

            # Try numeric key
            if prediction in family_mapping:

                label = family_mapping[
                    prediction
                ]

            # Try string numeric key
            elif str(prediction) in family_mapping:

                label = family_mapping[
                    str(prediction)
                ]

            # Try reverse mapping
            else:

                for key, value in family_mapping.items():

                    if str(value) == str(prediction):

                        label = key
                        break

        # ----------------------------------------------------
        # Third: safe fallback
        # ----------------------------------------------------

        if label is None:

            # If classes themselves are strings
            if isinstance(
                prediction,
                str
            ):

                label = prediction

            else:

                label = str(
                    prediction
                )

        predictions.append(
            str(label)
        )

    return (
        np.array(predictions),
        probabilities,
        classes
    )


# ============================================================
# FAMILY-CONSTRAINED FLAW TYPE
# ============================================================

def constrained_flaw_type(
    original_flaw_model,
    X,
    family_prediction,
    family_probabilities,
    family_classes
):

    flaw_probabilities = (
        original_flaw_model.predict_proba(X)
    )

    flaw_classes = list(
        original_flaw_model.classes_
    )

    final_types = []
    final_confidences = []

    family_confidences = []

    for index in range(
        len(X)
    ):

        family_name = str(
            family_prediction[index]
        )

        family_probability = float(
            np.max(
                family_probabilities[index]
            )
        )

        family_confidences.append(
            family_probability
        )

        if family_name == "temporal":

            allowed = (
                TEMPORAL_FLAWS
            )

        elif family_name == "acoustic":

            allowed = (
                ACOUSTIC_FLAWS
            )

        else:

            # Safe fallback
            allowed = set(
                str(x)
                for x in flaw_classes
            )

        candidates = []

        for class_index, class_name in enumerate(
            flaw_classes
        ):

            class_name_string = str(
                class_name
            )

            if class_name_string in allowed:

                candidates.append(
                    (
                        class_name_string,
                        float(
                            flaw_probabilities[
                                index,
                                class_index
                            ]
                        )
                    )
                )

        if not candidates:

            best_index = int(
                np.argmax(
                    flaw_probabilities[index]
                )
            )

            final_types.append(
                str(
                    flaw_classes[
                        best_index
                    ]
                )
            )

            final_confidences.append(
                float(
                    flaw_probabilities[
                        index,
                        best_index
                    ]
                )
            )

        else:

            candidates.sort(
                key=lambda x: x[1],
                reverse=True
            )

            final_types.append(
                candidates[0][0]
            )

            final_confidences.append(
                candidates[0][1]
            )

    return (
        final_types,
        final_confidences,
        family_confidences
    )


# ============================================================
# PRINT TYPE PROBABILITIES
# ============================================================

def print_type_diagnostic(
    predictions,
    family_predictions,
    family_probabilities,
    family_classes
):

    print()
    print("=" * 70)
    print("HYBRID FAMILY DIAGNOSTIC")
    print("=" * 70)

    for index in range(
        len(predictions)
    ):

        start = float(
            predictions.iloc[index][
                "window_start"
            ]
        )

        end = float(
            predictions.iloc[index][
                "window_end"
            ]
        )

        family = str(
            family_predictions[index]
        )

        confidence = float(
            np.max(
                family_probabilities[index]
            )
        )

        print(
            f"Window {index + 1:02d} | "
            f"{start:.2f}s -> {end:.2f}s | "
            f"family={family} | "
            f"confidence={confidence:.3f}"
        )

        pairs = sorted(
            zip(
                family_classes,
                family_probabilities[index]
            ),
            key=lambda x: x[1],
            reverse=True
        )

        for name, probability in pairs:

            print(
                f"    {str(name):12s} "
                f"{probability * 100:6.2f}%"
            )


# ============================================================
# PREDICTION
# ============================================================

def predict(
    contrastive_df,
    temporal_model,
    flaw_model,
    family_model,
    severity_model,
    temporal_meta=None,
    flaw_meta=None,
    family_meta=None,
    severity_meta=None,
):

    print()
    print("=" * 70)
    print("RUNNING ORATORIQ HYBRID MODELS")
    print("=" * 70)

    # --------------------------------------------------------
    # Original temporal model
    # --------------------------------------------------------

    temporal_X = prepare_model_features(
        contrastive_df,
        temporal_model,
        temporal_meta,
        "temporal model",
    )

    print(
        f"Temporal input shape: "
        f"{temporal_X.shape}"
    )

    temporal_prediction = (
        temporal_model.predict(
            temporal_X
        )
    )

    temporal_probability = (
        get_flawed_probability(
            temporal_model,
            temporal_X
        )
    )

    # --------------------------------------------------------
    # V21 family model
    # --------------------------------------------------------

    family_X = prepare_model_features(
        contrastive_df,
        family_model,
        family_meta,
        "V2.1 family model",
    )

    print(
        f"Family input shape: "
        f"{family_X.shape}"
    )

    (
        family_prediction,
        family_probabilities,
        family_classes
    ) = predict_family(
        family_model,
        family_X,
        family_meta
    )

    # --------------------------------------------------------
    # Original six-class model
    # --------------------------------------------------------

    flaw_X = prepare_model_features(
        contrastive_df,
        flaw_model,
        flaw_meta,
        "original flaw-type model",
    )

    original_flaw_prediction = (
        flaw_model.predict(
            flaw_X
        )
    )

    # --------------------------------------------------------
    # Family constrained prediction
    # --------------------------------------------------------

    (
        final_flaw_type,
        final_flaw_confidence,
        family_confidence
    ) = constrained_flaw_type(
        flaw_model,
        flaw_X,
        family_prediction,
        family_probabilities,
        family_classes
    )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    severity_X = prepare_model_features(
        contrastive_df,
        severity_model,
        severity_meta,
        "severity model",
    )

    severity_prediction = (
        severity_model.predict(
            severity_X
        )
    )

    severity_probabilities = (
        severity_model.predict_proba(
            severity_X
        )
    )

    severity_classes = list(
        severity_model.classes_
    )

    severity_confidence = (
        np.max(
            severity_probabilities,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Build dataframe
    # --------------------------------------------------------

    result = contrastive_df.copy()

    print()
    print("=" * 70)
    print("EVIDENCE FEATURE CHECK")
    print("=" * 70)

    evidence_columns = [
        "temporal_rms_delta",
        "temporal_silence_delta",
        "temporal_silence_ratio",
        "temporal_f0_delta",
        "temporal_f0_ratio",
        "temporal_wpm_delta",
        "temporal_wpm_ratio",
        "window_duration_delta",
        "window_duration_ratio",
        "alignment_error",
    ]

    for column in evidence_columns:

        if column in result.columns:

            values = result[column].astype(float)

            print(
                f"{column:30s} "
                f"min={values.min():.6f} "
                f"max={values.max():.6f}"
            )

        else:

            print(
                f"{column:30s} MISSING"
            )

    result[
        "temporal_prediction"
    ] = temporal_prediction

    result[
        "temporal_confidence"
    ] = temporal_probability

    result[
        "family_prediction"
    ] = family_prediction

    result[
        "family_confidence"
    ] = family_confidence

    result[
        "original_flaw_type"
    ] = original_flaw_prediction

    result[
        "flaw_type"
    ] = final_flaw_type

    result[
        "flaw_type_confidence"
    ] = final_flaw_confidence

    result[
        "severity"
    ] = severity_prediction

    result[
        "severity_confidence"
    ] = severity_confidence

    # --------------------------------------------------------
    # Family probabilities
    # --------------------------------------------------------

    for index, class_name in enumerate(
        family_classes
    ):

        result[
            f"family_prob_{class_name}"
        ] = family_probabilities[
            :,
            index
        ]

    # --------------------------------------------------------
    # Severity probabilities
    # --------------------------------------------------------

    for index, class_name in enumerate(
        severity_classes
    ):

        result[
            f"severity_prob_{class_name}"
        ] = severity_probabilities[
            :,
            index
        ]

    # --------------------------------------------------------
    # Original six-class probabilities
    # --------------------------------------------------------

    flaw_probabilities = (
        flaw_model.predict_proba(
            flaw_X
        )
    )

    flaw_classes = list(
        flaw_model.classes_
    )

    for index, class_name in enumerate(
        flaw_classes
    ):

        result[
            f"prob_{class_name}"
        ] = flaw_probabilities[
            :,
            index
        ]

    print_type_diagnostic(
        result,
        family_prediction,
        family_probabilities,
        family_classes
    )

    # --------------------------------------------------------
    # Temporal diagnostic
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEMPORAL DIAGNOSTIC")
    print("=" * 70)

    for index, row in result.iterrows():

        print(
            f"Window {index + 1:02d} | "
            f"{row['window_start']:.2f}s -> "
            f"{row['window_end']:.2f}s | "
            f"flawed="
            f"{row['temporal_confidence']:.4f} | "
            f"family="
            f"{row['family_prediction']} | "
            f"type="
            f"{row['flaw_type']} | "
            f"severity="
            f"{row['severity']}"
        )

    # --------------------------------------------------------
    # Highest risk
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HIGHEST RISK WINDOWS")
    print("=" * 70)

    top_rows = result.sort_values(
        "temporal_confidence",
        ascending=False
    ).head(5)

    for _, row in top_rows.iterrows():

        print(
            f"{row['window_start']:.2f}s"
            f" -> "
            f"{row['window_end']:.2f}s | "
            f"{row['temporal_confidence']:.4f} | "
            f"{row['family_prediction']} | "
            f"{row['flaw_type']} | "
            f"{row['severity']}"
        )

    return result


# ============================================================
# MERGE TEMPORAL DETECTIONS
# ============================================================

def merge_detections(df):

    print()
    print("=" * 70)
    print("MERGING TEMPORAL DETECTIONS")
    print("=" * 70)

    detected = df[
        (
            df[
                "temporal_prediction"
            ].astype(str).isin(
                ["1", "flawed"]
            )
        )
        &
        (
            df[
                "temporal_confidence"
            ]
            >= TEMPORAL_THRESHOLD
        )
    ].copy()

    print(
        f"Threshold: "
        f"{TEMPORAL_THRESHOLD:.2f}"
    )

    print(
        f"Windows above threshold: "
        f"{len(detected)}"
    )

    if detected.empty:

        print(
            "No temporal flaws detected."
        )

        return []

    detected = detected.sort_values(
        "window_start"
    )

    regions = []

    current = None

    for _, row in detected.iterrows():

        start = float(
            row["window_start"]
        )

        end = float(
            row["window_end"]
        )

        if current is None:

            current = {
                "start": start,
                "end": end,
                "rows": [row]
            }

            continue

        gap = (
            start
            - current["end"]
        )

        if gap <= MERGE_GAP_SECONDS:

            current["end"] = max(
                current["end"],
                end
            )

            current[
                "rows"
            ].append(row)

        else:

            regions.append(
                current
            )

            current = {
                "start": start,
                "end": end,
                "rows": [row]
            }

    if current is not None:

        regions.append(
            current
        )

    # --------------------------------------------------------
    # Build final regions
    # --------------------------------------------------------

    final_regions = []

    for region in regions:

        rows = pd.DataFrame(
            region["rows"]
        )

        # Family
        family = (
            rows[
                "family_prediction"
            ]
            .value_counts()
            .idxmax()
        )

        # Final type
        flaw_type = (
            rows[
                "flaw_type"
            ]
            .value_counts()
            .idxmax()
        )

        # Severity
        severity = (
            rows[
                "severity"
            ]
            .value_counts()
            .idxmax()
        )

        temporal_confidence = float(
            rows[
                "temporal_confidence"
            ].mean()
        )

        family_confidence = float(
            rows[
                "family_confidence"
            ].mean()
        )

        flaw_type_confidence = float(
            rows[
                "flaw_type_confidence"
            ].mean()
        )

        severity_confidence = float(
            rows[
                "severity_confidence"
            ].mean()
        )

        final_regions.append({

            "start": round(
                region["start"],
                3
            ),

            "end": round(
                region["end"],
                3
            ),

            "duration": round(
                region["end"]
                - region["start"],
                3
            ),

            "family": str(
                family
            ),

            "flaw_type": str(
                flaw_type
            ),

            "severity": str(
                severity
            ),

            "temporal_confidence": round(
                temporal_confidence,
                4
            ),

            "family_confidence": round(
                family_confidence,
                4
            ),

            "flaw_type_confidence": round(
                flaw_type_confidence,
                4
            ),

            "severity_confidence": round(
                severity_confidence,
                4
            )
        })

    return final_regions


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    audio_path,
    duration,
    regions,
    predictions
):

    RESULTS.mkdir(
        parents=True,
        exist_ok=True
    )

    window_predictions = []

    for _, row in predictions.iterrows():

        record = {

            # ----------------------------------------------------
            # NUMERIC ACOUSTIC EVIDENCE
            # ----------------------------------------------------

            "evidence": {
                "rms_delta": round(
                    float(
                        row.get(
                            "temporal_rms_delta",
                            0.0
                        )
                    ),
                    6
                ),

                "silence_delta": round(
                    float(
                        row.get(
                            "temporal_silence_delta",
                            0.0
                        )
                    ),
                    6
                ),

                "silence_ratio": round(
                    float(
                        row.get(
                            "temporal_silence_ratio",
                            0.0
                        )
                    ),
                    6
                ),

                "f0_delta": round(
                    float(
                        row.get(
                            "temporal_f0_delta",
                            0.0
                        )
                    ),
                    6
                ),

                "f0_ratio": round(
                    float(
                        row.get(
                            "temporal_f0_ratio",
                            0.0
                        )
                    ),
                    6
                ),

                "wpm_delta": round(
                    float(
                        row.get(
                            "temporal_wpm_delta",
                            0.0
                        )
                    ),
                    6
                ),

                "wpm_ratio": round(
                    float(
                        row.get(
                            "temporal_wpm_ratio",
                            0.0
                        )
                    ),
                    6
                ),

                "window_duration_delta": round(
                    float(
                        row.get(
                            "window_duration_delta",
                            0.0
                        )
                    ),
                    6
                ),

                "window_duration_ratio": round(
                    float(
                        row.get(
                            "window_duration_ratio",
                            0.0
                        )
                    ),
                    6
                ),

                "alignment_error": round(
                    float(
                        row.get(
                            "alignment_error",
                            0.0
                        )
                    ),
                    6
                )
            },

            "start": round(
                float(
                    row["window_start"]
                ),
                3
            ),

            "end": round(
                float(
                    row["window_end"]
                ),
                3
            ),

            "temporal_prediction": str(
                row[
                    "temporal_prediction"
                ]
            ),

            "temporal_confidence": round(
                float(
                    row[
                        "temporal_confidence"
                    ]
                ),
                4
            ),

            "family": str(
                row[
                    "family_prediction"
                ]
            ),

            "family_confidence": round(
                float(
                    row[
                        "family_confidence"
                    ]
                ),
                4
            ),

            "original_flaw_type": str(
                row[
                    "original_flaw_type"
                ]
            ),

            "final_flaw_type": str(
                row[
                    "flaw_type"
                ]
            ),

            "flaw_type_confidence": round(
                float(
                    row[
                        "flaw_type_confidence"
                    ]
                ),
                4
            ),

            "severity": str(
                row[
                    "severity"
                ]
            ),

            "severity_confidence": round(
                float(
                    row[
                        "severity_confidence"
                    ]
                ),
                4
            )
        }

        # All probability columns
        for column in predictions.columns:

            if (
                column.startswith("prob_")
                or column.startswith(
                    "family_prob_"
                )
                or column.startswith(
                    "severity_prob_"
                )
            ):

                record[column] = round(
                    float(
                        row[column]
                    ),
                    6
                )

        window_predictions.append(
            record
        )

    output = {

        "system": "OratorIQ",

        "version": "hybrid_v21",

        "audio": str(
            audio_path
        ),

        "duration": round(
            duration,
            3
        ),

        "temporal_threshold":
            TEMPORAL_THRESHOLD,

        "num_detected_regions":
            len(regions),

        "regions":
            regions,

        "window_predictions":
            window_predictions
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    return output


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(result):

    print()
    print("=" * 70)
    print("ORATORIQ HYBRID ANALYSIS RESULT")
    print("=" * 70)

    print(
        f"Audio: "
        f"{result['audio']}"
    )

    print(
        f"Duration: "
        f"{result['duration']:.2f}s"
    )

    print(
        f"Detected regions: "
        f"{result['num_detected_regions']}"
    )

    print()

    if not result["regions"]:

        print(
            "[OK] No significant "
            "temporal flaws detected."
        )

    else:

        for index, region in enumerate(
            result["regions"],
            start=1
        ):

            print(
                f"[{index}] "
                f"{region['start']:.2f}s "
                f"-> "
                f"{region['end']:.2f}s"
            )

            print(
                f"    Family: "
                f"{region['family']}"
            )

            print(
                f"    Flaw: "
                f"{region['flaw_type']}"
            )

            print(
                f"    Severity: "
                f"{region['severity']}"
            )

            print(
                f"    Temporal Confidence: "
                f"{region['temporal_confidence'] * 100:.1f}%"
            )

            print(
                f"    Family Confidence: "
                f"{region['family_confidence'] * 100:.1f}%"
            )

            print(
                f"    Flaw-Type Confidence: "
                f"{region['flaw_type_confidence'] * 100:.1f}%"
            )

            print(
                f"    Severity Confidence: "
                f"{region['severity_confidence'] * 100:.1f}%"
            )

            print()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "OratorIQ hybrid V2.1 "
            "contrastive speech inference"
        )
    )

    parser.add_argument(
        "--good",
        required=True,
        help="Good/reference WAV file"
    )

    parser.add_argument(
        "--flawed",
        required=True,
        help="Flawed WAV file"
    )

    parser.add_argument(
        "--transcript",
        default="",
        help="Optional transcript"
    )

    args = parser.parse_args()

    good_path = Path(
        args.good
    )

    flawed_path = Path(
        args.flawed
    )

    if not good_path.exists():

        raise FileNotFoundError(
            f"Good audio not found:\n"
            f"{good_path}"
        )

    if not flawed_path.exists():

        raise FileNotFoundError(
            f"Flawed audio not found:\n"
            f"{flawed_path}"
        )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    (
        temporal_model,
        flaw_model,
        family_model,
        severity_model,
        temporal_meta,
        flaw_meta,
        family_meta,
        severity_meta,
    ) = load_models()

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    good_df = extract_audio_windows(
        good_path,
        args.transcript
    )

    flawed_df = extract_audio_windows(
        flawed_path,
        args.transcript
    )

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    contrastive_df = (
        build_contrastive_features(
            flawed_df,
            good_df
        )
    )

    print()

    print(
        f"Generated inference features: "
        f"{contrastive_df.shape}"
    )

    numeric_model_columns = [
        c
        for c in contrastive_df.columns
        if c not in {
            "window_start",
            "window_end",
            "relative_position",
        }
        and pd.api.types.is_numeric_dtype(
            contrastive_df[c]
        )
    ]

    print(
        f"Generated numeric model features: "
        f"{len(numeric_model_columns)}"
    )

    if "temporal_rms_ratio" in contrastive_df.columns:
        raise RuntimeError(
            "Legacy feature 'temporal_rms_ratio' is present. "
            "V2.1 deployment features must use only "
            "'temporal_rms_delta'."
        )

    if "temporal_rms_delta" not in contrastive_df.columns:
        raise RuntimeError(
            "Required V2.1 feature 'temporal_rms_delta' "
            "was not generated."
        )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = predict(
        contrastive_df,
        temporal_model,
        flaw_model,
        family_model,
        severity_model,
        temporal_meta,
        flaw_meta,
        family_meta,
        severity_meta,
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    regions = merge_detections(
        predictions
    )

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    flawed_audio, sr = librosa.load(
        flawed_path,
        sr=SR,
        mono=True
    )

    duration = (
        len(flawed_audio)
        / sr
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result = save_result(
        flawed_path,
        duration,
        regions,
        predictions
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_result(
        result
    )

    print()

    print(
        "JSON saved to:"
    )

    print(
        OUTPUT_JSON
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()