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
FLAW_MODEL = MODELS / "contrastive_flaw_type_classifier.joblib"
SEVERITY_MODEL = MODELS / "contrastive_severity_classifier.joblib"

OUTPUT_JSON = RESULTS / "inference_result.json"


# ============================================================
# CONFIG
# ============================================================

SR = 16000
N_MFCC = 13

WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.5

MERGE_GAP_SECONDS = 0.75

# Main decision threshold
TEMPORAL_THRESHOLD = 0.50


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
# AUDIO FEATURE TABLE
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
# CONTRASTIVE FEATURES
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
    # Good recording relative positions
    # --------------------------------------------------------

    good_duration = (
        good_df["window_end"].max()
    )

    good_df = good_df.copy()

    good_df["_relative_position"] = (
        (
            good_df["window_start"]
            + good_df["window_end"]
        ) / 2.0
    ) / max(good_duration, 1e-6)

    flawed_duration = (
        flawed_df["window_end"].max()
    )

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

    return result_df


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    print()
    print("=" * 70)
    print("LOADING ORATORIQ MODELS")
    print("=" * 70)

    for path in [
        TEMPORAL_MODEL,
        FLAW_MODEL,
        SEVERITY_MODEL
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Model not found:\n{path}"
            )

        # ASCII output to avoid Windows encoding problems
        print(f"[OK] {path.name}")

    temporal_model = joblib.load(
        TEMPORAL_MODEL
    )

    flaw_model = joblib.load(
        FLAW_MODEL
    )

    severity_model = joblib.load(
        SEVERITY_MODEL
    )

    return (
        temporal_model,
        flaw_model,
        severity_model
    )


# ============================================================
# MODEL FEATURE PREPARATION
# ============================================================

def prepare_model_features(
    contrastive_df,
    temporal_model
):

    excluded = {
        "window_start",
        "window_end",
        "relative_position"
    }

    columns = []

    for column in contrastive_df.columns:

        if column in excluded:
            continue

        if pd.api.types.is_numeric_dtype(
            contrastive_df[column]
        ):
            columns.append(column)

    X = contrastive_df[columns].copy()

    # --------------------------------------------------------
    # IMPORTANT:
    # Reorder features exactly as the model saw them.
    # --------------------------------------------------------

    expected = getattr(
        temporal_model,
        "feature_names_in_",
        None
    )

    if expected is not None:

        expected = list(expected)

        missing = [
            c for c in expected
            if c not in X.columns
        ]

        extra = [
            c for c in X.columns
            if c not in expected
        ]

        if missing:

            raise ValueError(
                "Missing model features:\n"
                + "\n".join(missing)
            )

        if extra:

            print(
                f"Note: ignoring {len(extra)} "
                f"extra feature(s)."
            )

        X = X[expected]

    return X


# ============================================================
# GET FLAWED PROBABILITY SAFELY
# ============================================================

def get_flawed_probability(
    model,
    X
):

    probabilities = model.predict_proba(X)

    classes = list(model.classes_)

    print()
    print(
        f"Temporal model classes: {classes}"
    )

    if "flawed" in classes:

        flawed_index = classes.index(
            "flawed"
        )

        return probabilities[:, flawed_index]

    # Fallback for numeric labels
    if 1 in classes:

        flawed_index = classes.index(1)

        return probabilities[:, flawed_index]

    raise ValueError(
        "Could not find the 'flawed' class "
        f"in model classes: {classes}"
    )


# ============================================================
# FLAW TYPE PROBABILITY DIAGNOSTIC
# ============================================================

def print_flaw_type_probabilities(
    contrastive_df,
    X,
    flaw_model
):

    probabilities = flaw_model.predict_proba(X)
    classes = list(flaw_model.classes_)

    print()
    print("=" * 70)
    print("FLAW TYPE PROBABILITY DIAGNOSTIC")
    print("=" * 70)

    print(
        "Classes:",
        classes
    )

    for index in range(len(X)):

        probability_pairs = sorted(
            zip(
                classes,
                probabilities[index]
            ),
            key=lambda item: item[1],
            reverse=True
        )

        start = float(
            contrastive_df.iloc[index]["window_start"]
        )

        end = float(
            contrastive_df.iloc[index]["window_end"]
        )

        print()
        print(
            f"Window {index + 1:02d} | "
            f"{start:.2f}s → {end:.2f}s"
        )

        for class_name, probability in probability_pairs:

            print(
                f"    {str(class_name):18s} "
                f"{probability * 100:6.2f}%"
            )


# ============================================================
# PREDICTION
# ============================================================

def predict(
    contrastive_df,
    temporal_model,
    flaw_model,
    severity_model
):

    print()
    print("=" * 70)
    print("RUNNING ORATORIQ MODELS")
    print("=" * 70)

    X = prepare_model_features(
        contrastive_df,
        temporal_model
    )

    print(
        f"Model input shape: {X.shape}"
    )

    # --------------------------------------------------------
    # Temporal detector
    # --------------------------------------------------------

    temporal_pred = (
        temporal_model.predict(X)
    )

    temporal_prob = (
        get_flawed_probability(
            temporal_model,
            X
        )
    )

    # --------------------------------------------------------
    # Flaw type
    # --------------------------------------------------------

    flaw_pred = flaw_model.predict(X)

    # NEW:
    # Calculate and display probabilities for ALL
    # flaw-type classes without changing prediction logic.
    print_flaw_type_probabilities(
        contrastive_df,
        X,
        flaw_model
    )

    flaw_probabilities = flaw_model.predict_proba(X)
    flaw_classes = list(flaw_model.classes_)

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    severity_pred = severity_model.predict(X)

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    result = contrastive_df[
        [
            "window_start",
            "window_end"
        ]
    ].copy()

    result["temporal_prediction"] = (
        temporal_pred
    )

    result["temporal_confidence"] = (
        temporal_prob
    )

    result["flaw_type"] = flaw_pred

    result["severity"] = severity_pred

    # Store the highest flaw-type probability
    # without changing the actual class prediction.
    result["flaw_type_confidence"] = np.max(
        flaw_probabilities,
        axis=1
    )

    # Store the full probability distribution.
    for class_index, class_name in enumerate(
        flaw_classes
    ):

        safe_name = str(class_name)

        result[
            f"prob_{safe_name}"
        ] = flaw_probabilities[
            :,
            class_index
        ]

    # --------------------------------------------------------
    # DIAGNOSTIC OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEMPORAL PROBABILITY DIAGNOSTIC")
    print("=" * 70)

    for index, row in result.iterrows():

        print(
            f"Window {index + 1:02d} | "
            f"{row['window_start']:.2f}s"
            f" → "
            f"{row['window_end']:.2f}s | "
            f"flawed={row['temporal_confidence']:.4f} | "
            f"prediction={row['temporal_prediction']} | "
            f"type={row['flaw_type']} | "
            f"severity={row['severity']}"
        )

    # --------------------------------------------------------
    # Highest-risk windows
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
            f" → "
            f"{row['window_end']:.2f}s | "
            f"{row['temporal_confidence']:.4f} | "
            f"{row['flaw_type']} | "
            f"{row['severity']}"
        )

    return result


# ============================================================
# MERGE TEMPORAL WINDOWS
# ============================================================

def merge_detections(df):

    print()
    print("=" * 70)
    print("MERGING TEMPORAL DETECTIONS")
    print("=" * 70)

    detected = df[
        (
            df["temporal_prediction"].astype(str)
            .isin(["1", "flawed"])
        )
        &
        (
            df["temporal_confidence"]
            >= TEMPORAL_THRESHOLD
        )
    ].copy()

    print(
        f"Threshold: {TEMPORAL_THRESHOLD:.2f}"
    )

    print(
        f"Windows above threshold: "
        f"{len(detected)}"
    )

    if detected.empty:

        print(
            "No temporal flaws detected "
            "above threshold."
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

            current["rows"].append(
                row
            )

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
        regions.append(current)

    # --------------------------------------------------------
    # Build final regions
    # --------------------------------------------------------

    final_regions = []

    for region in regions:

        rows = pd.DataFrame(
            region["rows"]
        )

        flaw_type = (
            rows["flaw_type"]
            .value_counts()
            .idxmax()
        )

        severity = (
            rows["severity"]
            .value_counts()
            .idxmax()
        )

        confidence = float(
            rows[
                "temporal_confidence"
            ].mean()
        )

        # Average flaw-type confidence
        # for windows in this detected region.
        flaw_type_confidence = float(
            rows[
                "flaw_type_confidence"
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

            "flaw_type": str(
                flaw_type
            ),

            "severity": str(
                severity
            ),

            "confidence": round(
                confidence,
                4
            ),

            "flaw_type_confidence": round(
                flaw_type_confidence,
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

    # Convert window predictions to JSON-safe records

    window_predictions = []

    for _, row in predictions.iterrows():

        record = {

            "start": round(
                float(row["window_start"]),
                3
            ),

            "end": round(
                float(row["window_end"]),
                3
            ),

            "temporal_prediction": str(
                row["temporal_prediction"]
            ),

            "temporal_confidence": round(
                float(row["temporal_confidence"]),
                4
            ),

            "flaw_type": str(
                row["flaw_type"]
            ),

            "severity": str(
                row["severity"]
            ),

            "flaw_type_confidence": round(
                float(row["flaw_type_confidence"]),
                4
            )
        }

        # Add every flaw-type probability.
        for column in predictions.columns:

            if column.startswith("prob_"):

                record[column] = round(
                    float(row[column]),
                    6
                )

        window_predictions.append(
            record
        )

    output = {

        "audio": str(
            audio_path
        ),

        "duration": round(
            duration,
            3
        ),

        "temporal_threshold":
            TEMPORAL_THRESHOLD,

        "num_detected_regions": len(
            regions
        ),

        "regions": regions,

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
# DISPLAY
# ============================================================

def display_result(result):

    print()
    print("=" * 70)
    print("ORATORIQ ANALYSIS RESULT")
    print("=" * 70)

    print(
        f"Audio: {result['audio']}"
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
            "[OK] No significant temporal flaws "
            "detected above threshold."
        )

    else:

        for index, region in enumerate(
            result["regions"],
            start=1
        ):

            print(
                f"[{index}] "
                f"{region['start']:.2f}s "
                f"→ "
                f"{region['end']:.2f}s"
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
                f"{region['confidence'] * 100:.1f}%"
            )

            if "flaw_type_confidence" in region:

                print(
                    f"    Flaw-Type Confidence: "
                    f"{region['flaw_type_confidence'] * 100:.1f}%"
                )

            print()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "OratorIQ contrastive speech inference"
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
        severity_model
    ) = load_models()

    # --------------------------------------------------------
    # Extract audio features
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
    # Contrastive features
    # --------------------------------------------------------

    contrastive_df = build_contrastive_features(
        flawed_df,
        good_df
    )

    print()

    print(
        f"Contrastive feature shape: "
        f"{contrastive_df.shape}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = predict(
        contrastive_df,
        temporal_model,
        flaw_model,
        severity_model
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

    duration = len(
        flawed_audio
    ) / sr

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result = save_result(
        flawed_path,
        duration,
        regions,
        predictions
    )

    display_result(
        result
    )

    print()

    print(
        f"JSON saved to:\n"
        f"{OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()