import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

warnings.filterwarnings("ignore")


# ============================================================
# ORATORIQ — REGION-LEVEL FLAW ACOUSTIC DIAGNOSTIC
# ============================================================
#
# Purpose:
# Compare the exact labeled flaw region in a FLAWED recording
# against the corresponding region in its GOOD paired recording.
#
# This is a diagnostic only.
# It does NOT modify the dataset.
# It does NOT retrain any model.
#
# Dataset:
#   D:\OratorIQ\data\contrastive
#
# Metadata:
#   data\contrastive\metadata.jsonl
#
# ============================================================


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = PROJECT_ROOT / "data" / "contrastive" / "metadata.jsonl"

TARGET_FLAWS = [
    "fast_pacing",
    "slow_pacing",
    "high_volume",
    "low_volume",
    "pitch_deviation",
    "long_pause",
]

# Audio analysis settings
TARGET_SR = 16000

# F0 analysis
F0_MIN = 50
F0_MAX = 500

# Silence threshold in dB
SILENCE_DB = -40.0

# STFT frame settings
FRAME_LENGTH = 1024
HOP_LENGTH = 256

# Minimum samples required for useful analysis
MIN_SAMPLES = 512


# ------------------------------------------------------------
# PRINT HELPERS
# ------------------------------------------------------------

def print_line(char="=", width=78):
    print(char * width)


def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


# ------------------------------------------------------------
# LOAD METADATA
# ------------------------------------------------------------

def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_PATH}"
        )

    records = []

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"[WARNING] Invalid JSON at line {line_number}: {e}"
                )

    return records


# ------------------------------------------------------------
# AUDIO PATH
# ------------------------------------------------------------

def resolve_audio_path(record):
    audio_path = record.get("audio")

    if not audio_path:
        return None

    # Metadata uses Windows-style paths.
    path_string = str(audio_path).replace("\\", os.sep)

    path = Path(path_string)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# ------------------------------------------------------------
# LOAD AUDIO
# ------------------------------------------------------------

def load_audio(path):
    if path is None:
        return None, None

    if not path.exists():
        print(f"[WARNING] Audio not found: {path}")
        return None, None

    try:
        y, sr = librosa.load(
            str(path),
            sr=TARGET_SR,
            mono=True
        )

        return y.astype(np.float32), sr

    except Exception as e:
        print(f"[WARNING] Could not load {path}")
        print(f"          {e}")
        return None, None


# ------------------------------------------------------------
# REGION EXTRACTION
# ------------------------------------------------------------

def extract_region_by_relative_position(
    good_audio,
    flawed_audio,
    flaw_start,
    flaw_end,
):
    """
    The flaw timestamps belong to the flawed recording.

    We map the labeled flawed interval into the GOOD recording
    using relative position.

    This avoids assuming that both recordings have exactly
    identical timing after synthetic modification.
    """

    if good_audio is None or flawed_audio is None:
        return None, None, None

    flawed_duration = len(flawed_audio) / TARGET_SR
    good_duration = len(good_audio) / TARGET_SR

    if flawed_duration <= 0 or good_duration <= 0:
        return None, None, None

    flaw_start = max(0.0, float(flaw_start))
    flaw_end = min(float(flaw_end), flawed_duration)

    if flaw_end <= flaw_start:
        return None, None, None

    # Relative positions in flawed recording
    start_ratio = flaw_start / flawed_duration
    end_ratio = flaw_end / flawed_duration

    # Map to GOOD recording
    good_start = start_ratio * good_duration
    good_end = end_ratio * good_duration

    # Convert to samples
    flawed_start_sample = int(
        flaw_start * TARGET_SR
    )

    flawed_end_sample = int(
        flaw_end * TARGET_SR
    )

    good_start_sample = int(
        good_start * TARGET_SR
    )

    good_end_sample = int(
        good_end * TARGET_SR
    )

    # Clamp
    flawed_start_sample = max(
        0,
        min(flawed_start_sample, len(flawed_audio))
    )

    flawed_end_sample = max(
        flawed_start_sample,
        min(flawed_end_sample, len(flawed_audio))
    )

    good_start_sample = max(
        0,
        min(good_start_sample, len(good_audio))
    )

    good_end_sample = max(
        good_start_sample,
        min(good_end_sample, len(good_audio))
    )

    flawed_region = flawed_audio[
        flawed_start_sample:flawed_end_sample
    ]

    good_region = good_audio[
        good_start_sample:good_end_sample
    ]

    mapping = {
        "flawed_start": flaw_start,
        "flawed_end": flaw_end,
        "flawed_duration": flawed_duration,
        "good_start": good_start,
        "good_end": good_end,
        "good_duration": good_duration,
    }

    return good_region, flawed_region, mapping


# ------------------------------------------------------------
# RMS
# ------------------------------------------------------------

def calculate_rms_db(y):
    if y is None or len(y) == 0:
        return np.nan

    rms = np.sqrt(np.mean(np.square(y)) + 1e-12)

    return float(
        20.0 * np.log10(rms + 1e-12)
    )


# ------------------------------------------------------------
# SILENCE RATIO
# ------------------------------------------------------------

def calculate_silence_ratio(y):
    if y is None or len(y) < MIN_SAMPLES:
        return np.nan

    rms = librosa.feature.rms(
        y=y,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]

    rms_db = librosa.amplitude_to_db(
        rms,
        ref=1.0
    )

    silence = rms_db < SILENCE_DB

    return float(np.mean(silence))


# ------------------------------------------------------------
# F0 FEATURES
# ------------------------------------------------------------

def calculate_f0_features(y, sr):
    if y is None or len(y) < MIN_SAMPLES:
        return {
            "mean_f0": np.nan,
            "median_f0": np.nan,
            "std_f0": np.nan,
            "voiced_ratio": np.nan,
        }

    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=F0_MIN,
            fmax=F0_MAX,
            sr=sr,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )

        valid_f0 = f0[
            np.isfinite(f0)
        ]

        if len(valid_f0) == 0:
            return {
                "mean_f0": np.nan,
                "median_f0": np.nan,
                "std_f0": np.nan,
                "voiced_ratio": 0.0,
            }

        return {
            "mean_f0": float(np.mean(valid_f0)),
            "median_f0": float(np.median(valid_f0)),
            "std_f0": float(np.std(valid_f0)),
            "voiced_ratio": float(
                len(valid_f0) / max(len(f0), 1)
            ),
        }

    except Exception:
        return {
            "mean_f0": np.nan,
            "median_f0": np.nan,
            "std_f0": np.nan,
            "voiced_ratio": np.nan,
        }


# ------------------------------------------------------------
# ZERO CROSSING RATE
# ------------------------------------------------------------

def calculate_zcr(y):
    if y is None or len(y) < MIN_SAMPLES:
        return np.nan

    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]

    return float(np.mean(zcr))


# ------------------------------------------------------------
# SPECTRAL FEATURES
# ------------------------------------------------------------

def calculate_spectral_features(y, sr):
    if y is None or len(y) < MIN_SAMPLES:
        return {
            "spectral_centroid": np.nan,
            "spectral_bandwidth": np.nan,
            "spectral_rolloff": np.nan,
        }

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]

    return {
        "spectral_centroid": float(np.mean(centroid)),
        "spectral_bandwidth": float(np.mean(bandwidth)),
        "spectral_rolloff": float(np.mean(rolloff)),
    }


# ------------------------------------------------------------
# MFCC FEATURES
# ------------------------------------------------------------

def calculate_mfcc_features(y, sr):
    if y is None or len(y) < MIN_SAMPLES:
        return {
            f"mfcc_{i}_mean": np.nan
            for i in range(1, 14)
        }

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )

    features = {}

    for i in range(13):
        features[
            f"mfcc_{i + 1}_mean"
        ] = float(np.mean(mfcc[i]))

    return features


# ------------------------------------------------------------
# SPEAKING RATE PROXY
# ------------------------------------------------------------

def calculate_speaking_rate_proxy(y, sr):
    """
    Simple acoustic proxy.

    Counts energy transitions across frames.

    This is NOT a true word-per-minute measurement.
    It is useful only for comparing the paired regions.
    """

    if y is None or len(y) < MIN_SAMPLES:
        return np.nan

    rms = librosa.feature.rms(
        y=y,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]

    if len(rms) < 3:
        return np.nan

    threshold = np.median(rms)

    above = rms > threshold

    transitions = np.sum(
        above[1:] != above[:-1]
    )

    duration = len(y) / sr

    if duration <= 0:
        return np.nan

    return float(
        transitions / duration
    )


# ------------------------------------------------------------
# ZERO CROSSING VARIABILITY
# ------------------------------------------------------------

def calculate_zcr_std(y):
    if y is None or len(y) < MIN_SAMPLES:
        return np.nan

    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]

    return float(np.std(zcr))


# ------------------------------------------------------------
# EXTRACT ALL REGION FEATURES
# ------------------------------------------------------------

def extract_region_features(y, sr):

    if y is None or len(y) < MIN_SAMPLES:
        return {}

    duration = len(y) / sr

    features = {
        "duration": float(duration),

        "rms_db": calculate_rms_db(y),

        "silence_ratio": calculate_silence_ratio(y),

        "zcr": calculate_zcr(y),

        "zcr_std": calculate_zcr_std(y),

        "speaking_rate_proxy":
            calculate_speaking_rate_proxy(y, sr),
    }

    features.update(
        calculate_f0_features(y, sr)
    )

    features.update(
        calculate_spectral_features(y, sr)
    )

    features.update(
        calculate_mfcc_features(y, sr)
    )

    return features


# ------------------------------------------------------------
# PERCENT CHANGE
# ------------------------------------------------------------

def percent_change(good, flawed):

    if not np.isfinite(good):
        return np.nan

    if abs(good) < 1e-9:
        return np.nan

    return float(
        ((flawed - good) / abs(good)) * 100.0
    )


# ------------------------------------------------------------
# DISPLAY FEATURE COMPARISON
# ------------------------------------------------------------

def display_comparison(
    flaw_type,
    good_features,
    flawed_features,
):

    print()
    print("-" * 78)
    print(f"REGION COMPARISON: {flaw_type.upper()}")
    print("-" * 78)

    print(
        f"{'FEATURE':<26}"
        f"{'GOOD':>14}"
        f"{'FLAWED':>14}"
        f"{'DELTA':>14}"
        f"{'% CHANGE':>10}"
    )

    print("-" * 78)

    feature_order = [
        "duration",
        "rms_db",
        "silence_ratio",
        "mean_f0",
        "median_f0",
        "std_f0",
        "voiced_ratio",
        "zcr",
        "zcr_std",
        "speaking_rate_proxy",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
    ]

    for feature in feature_order:

        good = good_features.get(
            feature,
            np.nan
        )

        flawed = flawed_features.get(
            feature,
            np.nan
        )

        if np.isfinite(good) and np.isfinite(flawed):

            delta = flawed - good

            pct = percent_change(
                good,
                flawed
            )

            print(
                f"{feature:<26}"
                f"{good:>14.4f}"
                f"{flawed:>14.4f}"
                f"{delta:>14.4f}"
                f"{pct:>9.2f}%"
            )

        else:

            print(
                f"{feature:<26}"
                f"{'N/A':>14}"
                f"{'N/A':>14}"
                f"{'N/A':>14}"
                f"{'N/A':>10}"
            )

    print("-" * 78)


# ------------------------------------------------------------
# DIAGNOSTIC INTERPRETATION
# ------------------------------------------------------------

def interpret_flaw(
    flaw_type,
    good_features,
    flawed_features,
):

    print()
    print("INTERPRETATION")

    def value(name):
        return (
            good_features.get(name, np.nan),
            flawed_features.get(name, np.nan)
        )

    good_duration, flawed_duration = value("duration")
    good_rms, flawed_rms = value("rms_db")
    good_silence, flawed_silence = value("silence_ratio")
    good_f0, flawed_f0 = value("mean_f0")
    good_zcr, flawed_zcr = value("zcr")
    good_rate, flawed_rate = value(
        "speaking_rate_proxy"
    )

    if flaw_type == "fast_pacing":

        if np.isfinite(good_duration) and np.isfinite(flawed_duration):
            delta = flawed_duration - good_duration

            if delta < 0:
                print(
                    "[+] Duration decreased -> supports FAST_PACING"
                )
            else:
                print(
                    "[-] Duration did not decrease"
                )

        if np.isfinite(good_rate) and np.isfinite(flawed_rate):

            if flawed_rate > good_rate:
                print(
                    "[+] Acoustic transition rate increased -> supports FAST_PACING"
                )
            else:
                print(
                    "[-] Acoustic transition rate did not increase"
                )

    elif flaw_type == "slow_pacing":

        if np.isfinite(good_duration) and np.isfinite(flawed_duration):

            delta = flawed_duration - good_duration

            if delta > 0:
                print(
                    "[+] Duration increased -> supports SLOW_PACING"
                )
            else:
                print(
                    "[-] Duration did not increase"
                )

        if np.isfinite(good_rate) and np.isfinite(flawed_rate):

            if flawed_rate < good_rate:
                print(
                    "[+] Acoustic transition rate decreased -> supports SLOW_PACING"
                )
            else:
                print(
                    "[-] Acoustic transition rate did not decrease"
                )

    elif flaw_type == "high_volume":

        if np.isfinite(good_rms) and np.isfinite(flawed_rms):

            delta = flawed_rms - good_rms

            if delta > 0:
                print(
                    "[+] RMS increased -> supports HIGH_VOLUME"
                )
            else:
                print(
                    "[-] RMS did not increase"
                )

    elif flaw_type == "low_volume":

        if np.isfinite(good_rms) and np.isfinite(flawed_rms):

            delta = flawed_rms - good_rms

            if delta < 0:
                print(
                    "[+] RMS decreased -> supports LOW_VOLUME"
                )
            else:
                print(
                    "[-] RMS did not decrease"
                )

    elif flaw_type == "pitch_deviation":

        if np.isfinite(good_f0) and np.isfinite(flawed_f0):

            delta = flawed_f0 - good_f0

            if abs(delta) > 3.0:
                print(
                    "[+] F0 changed meaningfully -> supports PITCH_DEVIATION"
                )
            else:
                print(
                    "[!] F0 change is small in this example"
                )

    elif flaw_type == "long_pause":

        if np.isfinite(good_silence) and np.isfinite(flawed_silence):

            delta = flawed_silence - good_silence

            if delta > 0:
                print(
                    "[+] Silence ratio increased -> supports LONG_PAUSE"
                )
            else:
                print(
                    "[-] Silence ratio did not increase"
                )

        if np.isfinite(good_duration) and np.isfinite(flawed_duration):

            delta = flawed_duration - good_duration

            if delta > 0:
                print(
                    "[+] Region duration increased -> supports LONG_PAUSE"
                )
            else:
                print(
                    "[-] Region duration did not increase"
                )


# ------------------------------------------------------------
# FIND GOOD PAIR
# ------------------------------------------------------------

def find_good_record(records, pair_id):

    candidates = [
        r for r in records
        if r.get("pair_id") == pair_id
        and r.get("label") == "good"
    ]

    if not candidates:
        return None

    return candidates[0]


# ------------------------------------------------------------
# SELECT EXAMPLE FOR EACH FLAW
# ------------------------------------------------------------

def find_example(records, flaw_type):

    candidates = [
        r for r in records
        if r.get("flaw_type") == flaw_type
        and r.get("label") != "good"
        and r.get("flaw_start") is not None
        and r.get("flaw_end") is not None
    ]

    if not candidates:
        return None

    # Prefer a medium-severity example when possible.
    medium = [
        r for r in candidates
        if abs(
            float(r.get("severity", 0.0)) - 0.5
        ) < 1e-9
    ]

    if medium:
        return medium[0]

    return candidates[0]


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print_line()
    print(
        "ORATORIQ — REGION-LEVEL FLAW ACOUSTIC DIAGNOSTIC"
    )
    print_line()

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Metadata:     {METADATA_PATH}"
    )

    print()

    records = load_metadata()

    print(
        f"Metadata records: {len(records)}"
    )

    print()

    all_results = []

    # --------------------------------------------------------
    # Process each flaw type
    # --------------------------------------------------------

    for flaw_type in TARGET_FLAWS:

        print_line()
        print(
            f"CHECKING REGION: {flaw_type.upper()}"
        )
        print_line()

        flawed_record = find_example(
            records,
            flaw_type
        )

        if flawed_record is None:

            print(
                f"[WARNING] No example found for {flaw_type}"
            )

            continue

        pair_id = flawed_record.get(
            "pair_id"
        )

        good_record = find_good_record(
            records,
            pair_id
        )

        if good_record is None:

            print(
                f"[WARNING] No GOOD pair found for {pair_id}"
            )

            continue

        good_path = resolve_audio_path(
            good_record
        )

        flawed_path = resolve_audio_path(
            flawed_record
        )

        print(
            f"Pair ID:       {pair_id}"
        )

        print(
            f"Good audio:    {good_path}"
        )

        print(
            f"Flawed audio:  {flawed_path}"
        )

        print()

        flaw_start = safe_float(
            flawed_record.get("flaw_start")
        )

        flaw_end = safe_float(
            flawed_record.get("flaw_end")
        )

        severity = safe_float(
            flawed_record.get("severity")
        )

        print(
            f"Labeled flaw start: {flaw_start:.4f}s"
        )

        print(
            f"Labeled flaw end:   {flaw_end:.4f}s"
        )

        print(
            f"Labeled duration:   {flaw_end - flaw_start:.4f}s"
        )

        print(
            f"Severity:            {severity}"
        )

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        print()
        print("Loading audio...")

        good_audio, good_sr = load_audio(
            good_path
        )

        flawed_audio, flawed_sr = load_audio(
            flawed_path
        )

        if good_audio is None or flawed_audio is None:

            print(
                "[WARNING] Skipping because audio could not be loaded."
            )

            continue

        # ----------------------------------------------------
        # Extract paired regions
        # ----------------------------------------------------

        (
            good_region,
            flawed_region,
            mapping
        ) = extract_region_by_relative_position(
            good_audio,
            flawed_audio,
            flaw_start,
            flaw_end,
        )

        if good_region is None or flawed_region is None:

            print(
                "[WARNING] Could not extract regions."
            )

            continue

        print()
        print("REGION MAPPING")
        print("-" * 78)

        print(
            f"GOOD region:   "
            f"{mapping['good_start']:.4f}s "
            f"-> "
            f"{mapping['good_end']:.4f}s"
        )

        print(
            f"FLAWED region: "
            f"{mapping['flawed_start']:.4f}s "
            f"-> "
            f"{mapping['flawed_end']:.4f}s"
        )

        print(
            f"GOOD region duration:   "
            f"{len(good_region) / TARGET_SR:.4f}s"
        )

        print(
            f"FLAWED region duration: "
            f"{len(flawed_region) / TARGET_SR:.4f}s"
        )

        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        print()
        print("Extracting acoustic features...")

        good_features = extract_region_features(
            good_region,
            good_sr
        )

        flawed_features = extract_region_features(
            flawed_region,
            flawed_sr
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        display_comparison(
            flaw_type,
            good_features,
            flawed_features
        )

        interpret_flaw(
            flaw_type,
            good_features,
            flawed_features
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        result = {
            "flaw_type": flaw_type,
            "pair_id": pair_id,
            "severity": severity,

            "good_audio": str(good_path),
            "flawed_audio": str(flawed_path),

            "labeled_flaw_start": flaw_start,
            "labeled_flaw_end": flaw_end,

            "good_region_start": mapping[
                "good_start"
            ],

            "good_region_end": mapping[
                "good_end"
            ],

            "good_features": good_features,
            "flawed_features": flawed_features,
        }

        # Add numeric deltas
        deltas = {}

        for feature in good_features:

            good_value = good_features.get(
                feature,
                np.nan
            )

            flawed_value = flawed_features.get(
                feature,
                np.nan
            )

            if (
                np.isfinite(good_value)
                and np.isfinite(flawed_value)
            ):

                deltas[feature] = (
                    flawed_value - good_value
                )

        result["deltas"] = deltas

        all_results.append(result)

        print()

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print_line()
    print("SUMMARY")
    print_line()

    print(
        f"{'FLAW TYPE':<20}"
        f"{'DURATION Δ':>14}"
        f"{'RMS Δ':>14}"
        f"{'SILENCE Δ':>14}"
        f"{'F0 Δ':>14}"
    )

    print("-" * 78)

    for result in all_results:

        flaw_type = result["flaw_type"]

        deltas = result.get(
            "deltas",
            {}
        )

        duration_delta = deltas.get(
            "duration",
            np.nan
        )

        rms_delta = deltas.get(
            "rms_db",
            np.nan
        )

        silence_delta = deltas.get(
            "silence_ratio",
            np.nan
        )

        f0_delta = deltas.get(
            "mean_f0",
            np.nan
        )

        def fmt(value):

            if np.isfinite(value):
                return f"{value:+.4f}"

            return "N/A"

        print(
            f"{flaw_type:<20}"
            f"{fmt(duration_delta):>14}"
            f"{fmt(rms_delta):>14}"
            f"{fmt(silence_delta):>14}"
            f"{fmt(f0_delta):>14}"
        )

    print()

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / "flaw_region_diagnostic.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=True
        )

    print(
        f"Detailed JSON saved to:"
    )

    print(
        output_path
    )

    print()

    print_line()
    print(
        "REGION DIAGNOSTIC COMPLETE"
    )
    print_line()


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    main()