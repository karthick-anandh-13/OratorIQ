import json
import logging
import re
import sys
import uuid
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"
CONTRASTIVE_DIR = DATA_DIR / "contrastive"
GOOD_AUDIO_DIR = CONTRASTIVE_DIR / "audio" / "good"
METADATA_PATH = CONTRASTIVE_DIR / "metadata.jsonl"

ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ARTIFACTS_DIR / "results"
UPLOADS_DIR = ARTIFACTS_DIR / "api_uploads"

SCRIPTS_DIR = ROOT / "scripts"

STUTTER_RESULT_PATH = (
    ARTIFACTS_DIR
    / "stutter_results"
    / "stutter_inference_result.json"
)


# ============================================================
# FILES
# ============================================================

INFERENCE_RESULT_PATH = (
    RESULTS_DIR / "inference_hybrid_result.json"
)

EXPLANATION_RESULT_PATH = (
    RESULTS_DIR / "oratoriq_explanation.json"
)

TIMELINE_RESULT_PATH = (
    RESULTS_DIR / "oratoriq_timeline.json"
)


# ============================================================
# CONFIG
# ============================================================

TEMPORAL_THRESHOLD = 0.50
MAX_STANDALONE_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_STANDALONE_DURATION_SECONDS = 300
MIN_STANDALONE_DURATION_SECONDS = 0.15
STANDALONE_AUDIO_SUFFIXES = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
}

logger = logging.getLogger(__name__)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="OratorIQ API",
    version="1.0.0",
    description=(
        "OratorIQ contrastive speech analytics, "
        "temporal flaw grounding, and AI-assisted "
        "speech fluency analytics API."
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# UTILITIES
# ============================================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_json(path: Path):
    if not path.exists():
        return None

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            data,
            f,
            indent=2
        )


# ============================================================
# LOAD DELIVERY INFERENCE MODULE
# ============================================================

_INFERENCE_MODULE = None


def load_inference_module():
    """
    Load the existing OratorIQ hybrid inference script.

    We support the filenames used during development so the
    backend does not depend on one exact script filename.
    """

    global _INFERENCE_MODULE

    if _INFERENCE_MODULE is not None:
        return _INFERENCE_MODULE

    candidates = [
        SCRIPTS_DIR / "infer_oratoriq_hybrid.py",
        SCRIPTS_DIR / "infer_oratoriq.py",
        SCRIPTS_DIR / "infer_oratoriq_hybrid_updated.py",
    ]

    inference_path = None

    for candidate in candidates:
        if candidate.exists():
            inference_path = candidate
            break

    if inference_path is None:

        raise FileNotFoundError(
            "Could not find the OratorIQ inference script.\n\n"
            "Expected one of:\n"
            + "\n".join(
                f"  - {path}"
                for path in candidates
            )
        )

    print()
    print("=" * 70)
    print("LOADING ORATORIQ INFERENCE ENGINE")
    print("=" * 70)

    print(
        f"Inference script: {inference_path}"
    )

    try:
        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace"
        )
    except Exception:
        pass

    spec = importlib.util.spec_from_file_location(
        "oratoriq_inference_engine",
        inference_path
    )

    if spec is None or spec.loader is None:

        raise RuntimeError(
            "Could not load inference module."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        "oratoriq_inference_engine"
    ] = module

    spec.loader.exec_module(module)

    _INFERENCE_MODULE = module

    print(
        "[OK] OratorIQ inference engine loaded."
    )

    return module


# ============================================================
# TRANSCRIPT / PAIR IDENTIFICATION
# ============================================================

def extract_pair_id(filename: str):
    """
    Expected synthetic dataset filenames:

        1272-128104-0000_medium.wav
        1272-128104-0000_slight.wav
        1272-128104-0000_extreme.wav

    Returns:

        1272-128104-0000
    """

    stem = Path(filename).stem

    match = re.match(
        r"^(.+?)_(good|slight|medium|bad|extreme)$",
        stem,
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1)

    return None


def find_good_reference(filename: str):
    """
    Find the corresponding clean/reference recording.
    """

    pair_id = extract_pair_id(filename)

    if not pair_id:
        return None, None

    exact = GOOD_AUDIO_DIR / f"{pair_id}.wav"

    if exact.exists():
        return exact, pair_id

    matches = list(
        GOOD_AUDIO_DIR.rglob(
            f"{pair_id}.wav"
        )
    )

    if matches:
        return matches[0], pair_id

    return None, pair_id


def load_transcript(pair_id):
    """
    Load the transcript for a known LibriSpeech pair.
    """

    if not pair_id:
        return ""

    if not METADATA_PATH.exists():
        return ""

    try:

        with METADATA_PATH.open(
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                if not line.strip():
                    continue

                record = json.loads(line)

                if (
                    str(record.get("pair_id"))
                    == str(pair_id)
                ):

                    transcript = record.get(
                        "transcript",
                        ""
                    )

                    return str(
                        transcript or ""
                    )

    except Exception as exc:

        print(
            f"[WARNING] Could not load transcript: "
            f"{exc}"
        )

    return ""


# ============================================================
# SAVE UPLOADED AUDIO
# ============================================================

async def save_upload(
    audio: UploadFile,
    max_size_bytes: int | None = None,
):
    """
    Save uploaded audio into artifacts/api_uploads.
    """

    if not audio.filename:
        raise HTTPException(
            status_code=400,
            detail="An audio file is required."
        )

    original_name = Path(
        audio.filename
    ).name

    suffix = Path(
        original_name
    ).suffix.lower()

    if suffix not in {
        ".wav",
        ".flac",
        ".mp3",
        ".ogg",
        ".m4a"
    }:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. "
                "Use WAV, FLAC, MP3, OGG, or M4A."
            )
        )

    UPLOADS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    unique_name = (
        f"{uuid.uuid4().hex[:12]}_"
        f"{original_name}"
    )

    destination = (
        UPLOADS_DIR / unique_name
    )

    try:

        with destination.open(
            "wb"
        ) as output:

            bytes_written = 0

            while True:

                chunk = await audio.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                bytes_written += len(chunk)

                if (
                    max_size_bytes is not None
                    and bytes_written > max_size_bytes
                ):
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "The uploaded audio exceeds the "
                            "25 MiB file-size limit."
                        ),
                    )

                output.write(chunk)

    except HTTPException:
        destination.unlink(missing_ok=True)
        raise

    except Exception as exc:
        destination.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not save uploaded audio."
            )
        ) from exc

    return (
        destination,
        original_name
    )


# ============================================================
# AUDIO STATISTICS
# ============================================================

def read_audio_statistics(
    audio_path: Path
):
    """
    Read basic audio information.
    """

    try:

        with sf.SoundFile(
            str(audio_path)
        ) as sound:

            sample_rate = int(
                sound.samplerate
            )

            channels = int(
                sound.channels
            )

            samples = sound.read(
                dtype="float64"
            )

            if (
                sample_rate
                and channels
            ):

                duration = (
                    len(samples)
                    /
                    (
                        sample_rate
                        * channels
                    )
                )

            else:

                duration = 0.0

            rms = (
                float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                samples
                            )
                        )
                    )
                )
                if len(samples)
                else 0.0
            )

            return {
                "sample_rate": sample_rate,
                "channels": channels,
                "duration": float(
                    duration
                ),
                "rms": rms,
            }

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file is not "
                "readable audio."
            )
        ) from exc


def validate_standalone_audio(audio_path: Path) -> dict[str, Any]:
    """Validate a decodable, non-silent recording within service limits."""
    try:
        with sf.SoundFile(str(audio_path)) as sound:
            sample_rate = int(sound.samplerate)
            channels = int(sound.channels)
            duration = (
                sound.frames / sample_rate
                if sample_rate > 0
                else 0.0
            )

            if not sound.frames or not sample_rate or not channels:
                raise HTTPException(
                    status_code=400,
                    detail="The uploaded audio contains no usable samples.",
                )

            if duration < MIN_STANDALONE_DURATION_SECONDS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "The recording is too short for analysis; "
                        "provide at least 0.15 seconds of audio."
                    ),
                )

            if duration > MAX_STANDALONE_DURATION_SECONDS:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "The recording exceeds the 5-minute "
                        "duration limit."
                    ),
                )

            squared_sum = 0.0
            sample_count = 0

            while True:
                samples = sound.read(
                    65536,
                    dtype="float32",
                    always_2d=True,
                )

                if not len(samples):
                    break

                if not np.isfinite(samples).all():
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "The uploaded audio contains invalid "
                            "non-finite samples."
                        ),
                    )

                squared_sum += float(
                    np.square(
                        samples,
                        dtype=np.float64,
                    ).sum()
                )
                sample_count += samples.size

            rms = (
                float(np.sqrt(squared_sum / sample_count))
                if sample_count
                else 0.0
            )

            if rms <= 1e-5:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "The recording is silent or too quiet "
                        "to analyze."
                    ),
                )

            return {
                "sample_rate": sample_rate,
                "channels": channels,
                "duration": float(duration),
                "rms": rms,
            }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file could not be decoded as WAV, "
                "FLAC, MP3, or OGG audio."
            ),
        ) from exc


# ============================================================
# FAMILY NORMALIZATION
# ============================================================

def normalize_family(
    value,
    family_model=None,
    family_meta=None
):
    """
    Convert numeric family predictions such as:

        0
        1

    into:

        acoustic
        temporal
    """

    value_string = str(
        value
    ).lower()

    if value_string in {
        "temporal",
        "acoustic"
    }:

        return value_string

    if isinstance(
        family_meta,
        dict
    ):

        mapping = family_meta.get(
            "family_mapping"
        )

        if isinstance(
            mapping,
            dict
        ):

            if value in mapping:

                mapped = mapping[value]

                if str(mapped).lower() in {
                    "temporal",
                    "acoustic"
                }:

                    return str(
                        mapped
                    ).lower()

            if value_string in mapping:

                mapped = mapping[
                    value_string
                ]

                if str(mapped).lower() in {
                    "temporal",
                    "acoustic"
                }:

                    return str(
                        mapped
                    ).lower()

            for key, mapped in mapping.items():

                if str(key) == value_string:

                    if str(mapped).lower() in {
                        "temporal",
                        "acoustic"
                    }:

                        return str(
                            mapped
                        ).lower()

        label_encoder = family_meta.get(
            "label_encoder"
        )

        if label_encoder is not None:

            try:

                decoded = label_encoder.inverse_transform(
                    np.array(
                        [value]
                    )
                )[0]

                decoded = str(
                    decoded
                ).lower()

                if decoded in {
                    "temporal",
                    "acoustic"
                }:

                    return decoded

            except Exception:
                pass

    if value_string == "1":
        return "temporal"

    if value_string == "0":
        return "acoustic"

    return value_string


# ============================================================
# EVIDENCE
# ============================================================

def evidence_from_row(
    row
):
    """
    Extract deployment-safe quantitative evidence.
    """

    def get(
        *names,
        default=0.0
    ):

        for name in names:

            if name in row.index:

                return safe_float(
                    row[name],
                    default
                )

        return default

    return {
        "rms_delta": round(
            get(
                "temporal_rms_delta",
                "rms_delta"
            ),
            6
        ),

        "silence_delta": round(
            get(
                "temporal_silence_delta",
                "silence_delta"
            ),
            6
        ),

        "silence_ratio": round(
            get(
                "temporal_silence_ratio",
                "silence_ratio"
            ),
            6
        ),

        "f0_delta": round(
            get(
                "temporal_f0_delta",
                "f0_delta"
            ),
            6
        ),

        "f0_ratio": round(
            get(
                "temporal_f0_ratio",
                "f0_ratio"
            ),
            6
        ),

        "wpm_delta": round(
            get(
                "temporal_wpm_delta",
                "wpm_delta"
            ),
            6
        ),

        "wpm_ratio": round(
            get(
                "temporal_wpm_ratio",
                "wpm_ratio"
            ),
            6
        ),

        "window_duration_delta": round(
            get(
                "window_duration_delta"
            ),
            6
        ),

        "window_duration_ratio": round(
            get(
                "window_duration_ratio"
            ),
            6
        ),

        "alignment_error": round(
            get(
                "alignment_error"
            ),
            6
        ),
    }


# ============================================================
# BUILD WINDOW OUTPUT
# ============================================================

def build_window_predictions(
    predictions,
    family_meta=None,
    family_model=None
):
    """
    Convert pandas predictions into JSON-safe dictionaries.
    """

    windows = []

    for _, row in predictions.iterrows():

        family = normalize_family(
            row.get(
                "family_prediction",
                "unknown"
            ),
            family_model,
            family_meta
        )

        record = {

            "start": round(
                safe_float(
                    row.get(
                        "window_start"
                    )
                ),
                3
            ),

            "end": round(
                safe_float(
                    row.get(
                        "window_end"
                    )
                ),
                3
            ),

            "temporal_prediction": str(
                row.get(
                    "temporal_prediction",
                    ""
                )
            ),

            "temporal_confidence": round(
                safe_float(
                    row.get(
                        "temporal_confidence"
                    )
                ),
                4
            ),

            "family": family,

            "family_confidence": round(
                safe_float(
                    row.get(
                        "family_confidence"
                    )
                ),
                4
            ),

            "original_flaw_type": str(
                row.get(
                    "original_flaw_type",
                    ""
                )
            ),

            "final_flaw_type": str(
                row.get(
                    "flaw_type",
                    ""
                )
            ),

            "flaw_type_confidence": round(
                safe_float(
                    row.get(
                        "flaw_type_confidence"
                    )
                ),
                4
            ),

            "severity": str(
                row.get(
                    "severity",
                    ""
                )
            ),

            "severity_confidence": round(
                safe_float(
                    row.get(
                        "severity_confidence"
                    )
                ),
                4
            ),

            "evidence": evidence_from_row(
                row
            ),
        }

        for column in predictions.columns:

            if (
                column.startswith(
                    "prob_"
                )
                or column.startswith(
                    "family_prob_"
                )
                or column.startswith(
                    "severity_prob_"
                )
            ):

                value = row.get(
                    column
                )

                if value is not None:

                    record[column] = round(
                        safe_float(
                            value
                        ),
                        6
                    )

        windows.append(
            record
        )

    return windows


# ============================================================
# ADD EVIDENCE TO MERGED REGIONS
# ============================================================

def attach_region_evidence(
    regions,
    windows
):
    """
    Attach overlap-weighted evidence to each detected region.
    """

    evidence_keys = [
        "rms_delta",
        "silence_delta",
        "silence_ratio",
        "f0_delta",
        "f0_ratio",
        "wpm_delta",
        "wpm_ratio",
        "window_duration_delta",
        "window_duration_ratio",
        "alignment_error",
    ]

    for region in regions:

        region_start = safe_float(
            region.get(
                "start"
            )
        )

        region_end = safe_float(
            region.get(
                "end"
            )
        )

        weighted = {
            key: 0.0
            for key in evidence_keys
        }

        total_weight = 0.0

        for window in windows:

            window_start = safe_float(
                window.get(
                    "start"
                )
            )

            window_end = safe_float(
                window.get(
                    "end"
                )
            )

            overlap = max(
                0.0,
                min(
                    region_end,
                    window_end
                )
                -
                max(
                    region_start,
                    window_start
                )
            )

            if overlap <= 0:
                continue

            evidence = window.get(
                "evidence",
                {}
            )

            total_weight += overlap

            for key in evidence_keys:

                weighted[key] += (
                    safe_float(
                        evidence.get(
                            key
                        )
                    )
                    * overlap
                )

        if total_weight > 0:

            region["evidence"] = {
                key: round(
                    value / total_weight,
                    6
                )
                for key, value
                in weighted.items()
            }

        else:

            region["evidence"] = {
                key: 0.0
                for key in evidence_keys
            }

    return regions


# ============================================================
# BUILD TIMELINE
# ============================================================

def build_timeline(
    duration,
    windows,
    regions,
    overall_confidence
):
    """
    Build the JSON timeline used by the React dashboard.
    """

    timeline_windows = []

    for window in windows:

        probability = safe_float(
            window.get(
                "temporal_confidence"
            )
        )

        detected = (
            probability
            >= TEMPORAL_THRESHOLD
        )

        timeline_windows.append({

            "start": window[
                "start"
            ],

            "end": window[
                "end"
            ],

            "temporal_probability": round(
                probability,
                4
            ),

            "status": (
                "detected"
                if detected
                else "normal"
            ),

            "flaw_type": (
                window.get(
                    "final_flaw_type"
                )
                if detected
                else None
            ),

            "family": (
                window.get(
                    "family"
                )
                if detected
                else None
            ),

            "severity": (
                window.get(
                    "severity"
                )
                if detected
                else None
            ),
        })

    return {

        "system": "OratorIQ",

        "version": "timeline_api_v1",

        "duration": round(
            duration,
            3
        ),

        "total_windows": len(
            timeline_windows
        ),

        "detected_windows": sum(
            1
            for window in timeline_windows
            if window["status"]
            == "detected"
        ),

        "detected_regions": len(
            regions
        ),

        "overall_confidence": round(
            overall_confidence,
            4
        ),

        "overall_confidence_percent": round(
            overall_confidence * 100,
            1
        ),

        "regions": regions,

        "windows": timeline_windows,
    }


# ============================================================
# GENERATE EXPLANATION
# ============================================================

def generate_explanation(
    inference_result
):
    """
    Use the existing OratorIQ explainability engine.
    """

    explanation_script = (
        SCRIPTS_DIR
        / "generate_explanation.py"
    )

    if not explanation_script.exists():

        print(
            "[WARNING] "
            "generate_explanation.py not found."
        )

        return None

    try:

        spec = (
            importlib.util.spec_from_file_location(
                "oratoriq_explanation_engine",
                explanation_script
            )
        )

        if spec is None or spec.loader is None:
            return None

        module = (
            importlib.util.module_from_spec(
                spec
            )
        )

        spec.loader.exec_module(
            module
        )

        if not hasattr(
            module,
            "generate_explanation"
        ):

            return None

        explanation = (
            module.generate_explanation(
                inference_result
            )
        )

        if hasattr(
            module,
            "save_explanation"
        ):

            module.save_explanation(
                explanation,
                EXPLANATION_RESULT_PATH
            )

        else:

            save_json(
                EXPLANATION_RESULT_PATH,
                explanation
            )

        return explanation

    except Exception as exc:

        print(
            "[WARNING] "
            f"Explainability engine failed: {exc}"
        )

        return None


# ============================================================
# V4 FLUENCY / STUTTER ANALYSIS
# ============================================================

_STUTTER_MODULE = None


def load_stutter_inference_module():
    """
    Load scripts/stutter_infer.py.

    stutter_infer.py already exposes:
        run_v4_inference(audio_path)

    The V4 module internally caches its models, so repeated
    API requests do not need to reload the models every time.
    """

    global _STUTTER_MODULE

    if _STUTTER_MODULE is not None:
        return _STUTTER_MODULE

    inference_path = (
        SCRIPTS_DIR
        / "stutter_infer.py"
    )

    if not inference_path.exists():

        raise FileNotFoundError(
            "Stutter inference script not found: "
            f"{inference_path}"
        )

    print()
    print("=" * 70)
    print("LOADING ORATORIQ V4 FLUENCY ENGINE")
    print("=" * 70)

    print(
        f"Fluency script: {inference_path}"
    )

    spec = importlib.util.spec_from_file_location(
        "oratoriq_stutter_inference",
        inference_path,
    )

    if spec is None or spec.loader is None:

        raise RuntimeError(
            "Could not load stutter inference module."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        "oratoriq_stutter_inference"
    ] = module

    spec.loader.exec_module(
        module
    )

    if not hasattr(
        module,
        "run_v4_inference"
    ):

        raise RuntimeError(
            "stutter_infer.py does not expose "
            "run_v4_inference()."
        )

    _STUTTER_MODULE = module

    print(
        "[OK] V4 fluency engine loaded."
    )

    return module


def run_fluency_analysis(
    audio_path: Path
):
    """
    Run the V4 fluency/stutter analysis on the uploaded audio.

    Returns a JSON-safe result.

    IMPORTANT:
    V4 is currently an audio-only demo adapter. Its seven
    annotation-derived features are zero-filled by
    stutter_infer.py. Therefore the API does not describe
    this as a medical diagnosis.
    """

    try:

        module = (
            load_stutter_inference_module()
        )

        result = (
            module.run_v4_inference(
                audio_path
            )
        )

        if not isinstance(
            result,
            dict
        ):

            raise RuntimeError(
                "V4 inference returned an invalid result."
            )

        prediction = result.get(
            "prediction",
            {}
        )

        events = prediction.get(
            "events",
            {}
        )

        possible_events = []

        for name, event in events.items():

            if isinstance(
                event,
                dict
            ) and event.get(
                "detected",
                False
            ):

                possible_events.append(
                    name
                )

        fluency_result = {

            "available": True,

            "model": result.get(
                "model",
                "V4"
            ),

            "mode": result.get(
                "mode",
                "audio_only_demo"
            ),

            "detected": bool(
                prediction.get(
                    "detected",
                    False
                )
            ),

            "stutter_probability": round(
                safe_float(
                    prediction.get(
                        "stutter_probability"
                    )
                ),
                6
            ),

            "stutter_probability_percent": round(
                safe_float(
                    prediction.get(
                        "stutter_probability_percent"
                    )
                ),
                2
            ),

            "threshold": round(
                safe_float(
                    prediction.get(
                        "threshold"
                    )
                ),
                4
            ),

            "events": events,

            "possible_events": (
                possible_events
            ),

            "warning": result.get(
                "warning",
                (
                    "V4 is an AI-assisted fluency "
                    "analytics demo and is not a "
                    "medical diagnosis."
                )
            ),

            "runtime_seconds": round(
                safe_float(
                    result.get(
                        "runtime_seconds"
                    )
                ),
                3
            ),
        }

        # Save standalone V4 result.
        save_json(
            STUTTER_RESULT_PATH,
            {
                "system": "OratorIQ",
                "model": fluency_result[
                    "model"
                ],
                "mode": fluency_result[
                    "mode"
                ],
                "audio": str(
                    audio_path
                ),
                "warning": fluency_result[
                    "warning"
                ],
                "prediction": prediction,
                "possible_events": (
                    possible_events
                ),
                "runtime_seconds": fluency_result[
                    "runtime_seconds"
                ],
            }
        )

        return fluency_result

    except Exception as exc:

        print()
        print(
            "[WARNING] V4 fluency analysis failed:"
        )
        print(
            f"{type(exc).__name__}: {exc}"
        )

        # Do NOT fail the entire delivery analysis
        # if the optional fluency module fails.
        return {
            "available": False,

            "model": "V4",

            "mode": "audio_only_demo",

            "detected": False,

            "stutter_probability": 0.0,

            "stutter_probability_percent": 0.0,

            "threshold": 0.67,

            "events": {},

            "possible_events": [],

            "warning": (
                "Fluency analysis was unavailable "
                "for this audio. Delivery analysis "
                "was completed normally."
            ),

            "runtime_seconds": 0.0,

            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health_check():

    return {
        "status": "ok"
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    """
    Return dataset statistics for the existing dashboard.
    """

    entries = []

    manifest_candidates = [
        MANIFEST_PATH,
        CONTRASTIVE_DIR / "metadata.jsonl",
    ]

    manifest_path = None

    for candidate in manifest_candidates:

        if candidate.exists():

            manifest_path = candidate
            break

    if manifest_path:

        with manifest_path.open(
            "r",
            encoding="utf-8"
        ) as manifest_file:

            for line in manifest_file:

                if line.strip():

                    entries.append(
                        json.loads(line)
                    )

    degraded = [
        entry
        for entry in entries
        if safe_float(
            entry.get(
                "severity",
                0.0
            )
        ) > 0
    ]

    average_severity = (

        sum(
            safe_float(
                entry.get(
                    "severity",
                    0.0
                )
            )
            for entry in degraded
        )
        /
        len(degraded)

        if degraded
        else 0.0
    )

    return {

        "entries": len(
            entries
        ),

        "degraded_entries": len(
            degraded
        ),

        "average_severity": round(
            average_severity,
            4
        ),

        "recent_entries": (
            entries[-5:][::-1]
        ),
    }


# ============================================================
# RAG COACHING INTEGRATION
# ============================================================

def generate_rag_coaching(inference_result, fluency):
    """
    Add retrieval-grounded coaching without making RAG a dependency
    for the existing speech-analysis pipeline.

    Delivery regions and fluency events are coached separately so that
    one kind of finding does not hide the other. Any RAG failure is
    isolated and does not fail /api/analyze.
    """
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        from rag.coach import generate_coaching

        delivery_result = generate_coaching(inference_result)
        fluency_result = generate_coaching({"fluency": fluency})

        combined = [delivery_result, fluency_result]
        valid = [
            item for item in combined
            if isinstance(item, dict) and item.get("status") == "ok"
        ]

        if not valid:
            return {
                "status": "no_findings",
                "summary": "No supported findings were available for coaching.",
                "observations": [],
                "recommendations": [],
                "exercises": [],
                "sources": [],
                "retrieval": {
                    "method": "TF-IDF",
                    "knowledge_chunks_used": 0,
                    "top_k_per_finding": 4,
                },
                "disclaimer": (
                    "Automated observations are estimates, not a medical diagnosis. "
                    "Fluency-event predictions are possibilities, not confirmed clinical "
                    "findings. For persistent concerns, consult a qualified speech-language "
                    "pathologist."
                ),
            }

        observations = []
        recommendations = []
        exercises = []
        sources_by_file = {}
        recommendation_seen = set()
        exercise_seen = set()
        chunk_count = 0
        disclaimers = []

        for item in valid:
            observations.extend(item.get("observations", []))
            chunk_count += item.get("retrieval", {}).get(
                "knowledge_chunks_used", 0
            )

            for recommendation in item.get("recommendations", []):
                identity = (
                    recommendation.get("text", "").strip().lower(),
                    recommendation.get("source", ""),
                )
                if identity not in recommendation_seen:
                    recommendation_seen.add(identity)
                    recommendations.append(recommendation)

            for exercise in item.get("exercises", []):
                identity = (
                    exercise.get("title", ""),
                    exercise.get("source", ""),
                )
                if identity not in exercise_seen:
                    exercise_seen.add(identity)
                    exercises.append(exercise)

            for source in item.get("sources", []):
                filename = source.get("file", "unknown")
                if filename not in sources_by_file:
                    sources_by_file[filename] = {
                        "title": source.get("title", filename),
                        "file": filename,
                        "sections": [],
                    }

                for section in source.get("sections", []):
                    if section not in sources_by_file[filename]["sections"]:
                        sources_by_file[filename]["sections"].append(section)

            disclaimer = item.get("disclaimer")
            if disclaimer and disclaimer not in disclaimers:
                disclaimers.append(disclaimer)

        finding_names = []
        for observation in observations:
            title = observation.get("title")
            if title and title not in finding_names:
                finding_names.append(title)

        return {
            "status": "ok" if observations else "no_findings",
            "summary": (
                "Coaching guidance for: " + ", ".join(finding_names) + "."
                if finding_names
                else "No supported findings were available for coaching."
            ),
            "observations": observations,
            "recommendations": recommendations[:8],
            "exercises": exercises[:4],
            "sources": list(sources_by_file.values()),
            "retrieval": {
                "method": "TF-IDF",
                "knowledge_chunks_used": chunk_count,
                "top_k_per_finding": 4,
            },
            "disclaimer": " ".join(disclaimers) if disclaimers else (
                "Automated observations are estimates, not a medical diagnosis."
            ),
        }

    except Exception as exc:
        # Coaching is optional: never let it break the existing API result.
        print(
            "[WARNING] RAG coaching unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
        return {
            "status": "unavailable",
            "summary": "Speech analysis completed, but coaching is unavailable.",
            "observations": [],
            "recommendations": [],
            "exercises": [],
            "sources": [],
            "error": f"{type(exc).__name__}: {exc}",
            "disclaimer": (
                "Automated speech observations are estimates, not a medical diagnosis."
            ),
        }


# ============================================================
# MAIN ORATORIQ ANALYSIS API
# ============================================================

@app.post("/api/analyze")
async def analyze_audio(
    audio: UploadFile = File(...)
) -> dict[str, Any]:
    """
    Analyze an uploaded audio file using:

    1. Contrastive Delivery Analysis
       - Temporal flaw detection
       - Flaw family
       - Flaw type
       - Severity
       - Quantitative evidence
       - Timeline
       - Explanation

    2. V4 Fluency Analysis
       - Overall fluency/stutter probability
       - Prolongation
       - Block
       - Sound repetition
       - Word repetition
       - Interjection

    The two analyses are returned together in one API response.
    """

    # --------------------------------------------------------
    # Save upload
    # --------------------------------------------------------

    uploaded_path, original_name = (
        await save_upload(
            audio
        )
    )

    print()
    print("=" * 70)
    print("ORATORIQ API ANALYSIS")
    print("=" * 70)

    print(
        f"Uploaded audio: {original_name}"
    )

    print(
        f"Saved audio: {uploaded_path}"
    )

    # --------------------------------------------------------
    # Find reference
    # --------------------------------------------------------

    reference_path, pair_id = (
        find_good_reference(
            original_name
        )
    )

    if reference_path is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "OratorIQ contrastive analysis needs "
                "a clean reference recording. "
                "For the current demo, upload one of the "
                "synthetic contrastive files such as "
                "'1272-128104-0000_medium.wav'. "
                "No matching clean reference was found "
                f"for '{original_name}'."
            )
        )

    print(
        f"Pair ID: {pair_id}"
    )

    print(
        f"Reference: {reference_path}"
    )

    # --------------------------------------------------------
    # Transcript
    # --------------------------------------------------------

    transcript = load_transcript(
        pair_id
    )

    if transcript:

        print(
            "[OK] Reference transcript loaded."
        )

    else:

        print(
            "[WARNING] No transcript found."
        )

    # --------------------------------------------------------
    # Audio statistics
    # --------------------------------------------------------

    audio_stats = read_audio_statistics(
        uploaded_path
    )

    # --------------------------------------------------------
    # RUN V4 FLUENCY ANALYSIS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING V4 FLUENCY ANALYSIS")
    print("=" * 70)

    fluency = run_fluency_analysis(
        uploaded_path
    )

    print(
        "[OK] Fluency analysis completed."
    )

    print(
        f"Stutter probability: "
        f"{fluency.get('stutter_probability_percent', 0):.2f}%"
    )

    print(
        f"Possible events: "
        f"{', '.join(fluency.get('possible_events', [])) or 'None'}"
    )

    # --------------------------------------------------------
    # Load delivery inference engine
    # --------------------------------------------------------

    try:

        inference = (
            load_inference_module()
        )

        # ----------------------------------------------------
        # Models
        # ----------------------------------------------------

        (
            temporal_model,
            flaw_model,
            family_model,
            severity_model,
            temporal_meta,
            flaw_meta,
            family_meta,
            severity_meta,
        ) = inference.load_models()

        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        good_df = (
            inference.extract_audio_windows(
                reference_path,
                transcript
            )
        )

        flawed_df = (
            inference.extract_audio_windows(
                uploaded_path,
                transcript
            )
        )

        # ----------------------------------------------------
        # Contrastive features
        # ----------------------------------------------------

        contrastive_df = (
            inference.build_contrastive_features(
                flawed_df,
                good_df
            )
        )

        print()
        print(
            f"[OK] Contrastive features: "
            f"{contrastive_df.shape}"
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = inference.predict(
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

        # ----------------------------------------------------
        # Temporal regions
        # ----------------------------------------------------

        regions = (
            inference.merge_detections(
                predictions
            )
        )

        # ----------------------------------------------------
        # Normalize family names
        # ----------------------------------------------------

        for region in regions:

            region["family"] = (
                normalize_family(
                    region.get(
                        "family",
                        "unknown"
                    ),
                    family_model,
                    family_meta
                )
            )

        # ----------------------------------------------------
        # Window JSON
        # ----------------------------------------------------

        windows = (
            build_window_predictions(
                predictions,
                family_meta,
                family_model
            )
        )

        # ----------------------------------------------------
        # Evidence
        # ----------------------------------------------------

        regions = attach_region_evidence(
            regions,
            windows
        )

        # ----------------------------------------------------
        # Overall confidence
        # ----------------------------------------------------

        if regions:

            confidence_values = []

            for region in regions:

                values = [

                    safe_float(
                        region.get(
                            "temporal_confidence"
                        )
                    ),

                    safe_float(
                        region.get(
                            "family_confidence"
                        )
                    ),

                    safe_float(
                        region.get(
                            "flaw_type_confidence"
                        )
                    ),

                    safe_float(
                        region.get(
                            "severity_confidence"
                        )
                    ),
                ]

                confidence_values.append(
                    sum(values)
                    / len(values)
                )

            overall_confidence = (
                sum(
                    confidence_values
                )
                /
                len(
                    confidence_values
                )
            )

        else:

            overall_confidence = 0.0

        # ----------------------------------------------------
        # Delivery inference result
        # ----------------------------------------------------

        inference_result = {

            "system": "OratorIQ",

            "version": "hybrid_v21_api",

            "audio": str(
                uploaded_path
            ),

            "audio_name": original_name,

            "reference_audio": str(
                reference_path
            ),

            "pair_id": pair_id,

            "duration": round(
                audio_stats[
                    "duration"
                ],
                3
            ),

            "sample_rate": audio_stats[
                "sample_rate"
            ],

            "channels": audio_stats[
                "channels"
            ],

            "rms": audio_stats[
                "rms"
            ],

            "temporal_threshold": (
                TEMPORAL_THRESHOLD
            ),

            "num_detected_regions": len(
                regions
            ),

            "regions": regions,

            "window_predictions": windows,
        }

        # ----------------------------------------------------
        # Save delivery inference JSON
        # ----------------------------------------------------

        save_json(
            INFERENCE_RESULT_PATH,
            inference_result
        )

        print(
            "[OK] Delivery inference JSON saved:"
        )

        print(
            INFERENCE_RESULT_PATH
        )

        # ----------------------------------------------------
        # Explanation
        # ----------------------------------------------------

        explanation = (
            generate_explanation(
                inference_result
            )
        )

        # ----------------------------------------------------
        # Timeline
        # ----------------------------------------------------

        timeline = build_timeline(
            audio_stats["duration"],
            windows,
            regions,
            overall_confidence
        )

        save_json(
            TIMELINE_RESULT_PATH,
            timeline
        )

        # ----------------------------------------------------
        # API analysis response
        # ----------------------------------------------------

        detected = (
            len(regions) > 0
        )

        flaw_list = []

        for region in regions:

            flaw_list.append({

                "start": region.get(
                    "start"
                ),

                "end": region.get(
                    "end"
                ),

                "flaw_type": region.get(
                    "flaw_type",
                    ""
                ),

                "family": region.get(
                    "family",
                    ""
                ),

                "severity": region.get(
                    "severity",
                    ""
                ),

                "confidence": round(
                    (
                        safe_float(
                            region.get(
                                "temporal_confidence"
                            )
                        )
                        +
                        safe_float(
                            region.get(
                                "family_confidence"
                            )
                        )
                        +
                        safe_float(
                            region.get(
                                "flaw_type_confidence"
                            )
                        )
                        +
                        safe_float(
                            region.get(
                                "severity_confidence"
                            )
                        )
                    )
                    / 4,
                    4
                ),

                "evidence": region.get(
                    "evidence",
                    {}
                ),
            })

        # ----------------------------------------------------
        # FINAL COMBINED RESPONSE
        # ----------------------------------------------------

        response = {

            # ------------------------------------------------
            # Basic audio information
            # ------------------------------------------------

            "audio_path": original_name,

            "audio_name": original_name,

            "reference_audio": (
                reference_path.name
            ),

            "pair_id": pair_id,

            "transcript": transcript,

            "sample_rate": audio_stats[
                "sample_rate"
            ],

            "duration": audio_stats[
                "duration"
            ],

            "rms": audio_stats[
                "rms"
            ],

            "channels": audio_stats[
                "channels"
            ],

            # ------------------------------------------------
            # Backward-compatible delivery fields
            # ------------------------------------------------

            "flaws_detected": flaw_list,

            "score": round(
                (
                    100.0
                    if not detected
                    else
                    max(
                        0.0,
                        100.0
                        * (
                            1.0
                            - overall_confidence
                        )
                    )
                ),
                1
            ),

            # ------------------------------------------------
            # DELIVERY ANALYSIS
            # ------------------------------------------------

            "delivery": {

                "detected": detected,

                "overall_confidence": round(
                    overall_confidence,
                    4
                ),

                "overall_confidence_percent": round(
                    overall_confidence * 100,
                    1
                ),

                "regions": flaw_list,

                "window_predictions": windows,

                "explanation": (
                    explanation
                ),

                "timeline": timeline,
            },

            # ------------------------------------------------
            # FLUENCY ANALYSIS
            # ------------------------------------------------

            "fluency": fluency,

            # ------------------------------------------------
            # Backward-compatible analysis object
            # ------------------------------------------------

            "analysis": {

                "detected": detected,

                "overall_confidence": round(
                    overall_confidence,
                    4
                ),

                "overall_confidence_percent": round(
                    overall_confidence * 100,
                    1
                ),

                "regions": flaw_list,

                "window_predictions": windows,

                "explanation": (
                    explanation
                ),

                "timeline": timeline,

                "fluency": fluency,
            },

            # ------------------------------------------------
            # Direct timeline
            # ------------------------------------------------

            "timeline": timeline,

            # ------------------------------------------------
            # Direct explanation
            # ------------------------------------------------

            "explanation": explanation,
        }

        # ----------------------------------------------------
        # RAG COACHING (optional; isolated from ML inference)
        # ----------------------------------------------------

        response["coaching"] = generate_rag_coaching(
            inference_result,
            fluency,
        )

        print(
            f"[RAG] Coaching status: "
            f"{response['coaching'].get('status', 'unknown')}"
        )

        # ----------------------------------------------------
        # Console summary
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("ORATORIQ API RESULT")
        print("=" * 70)

        print(
            f"Delivery regions: "
            f"{len(regions)}"
        )

        print(
            f"Delivery confidence: "
            f"{overall_confidence * 100:.1f}%"
        )

        print(
            f"Fluency probability: "
            f"{fluency.get('stutter_probability_percent', 0):.2f}%"
        )

        print(
            "Possible fluency events: "
            f"{', '.join(fluency.get('possible_events', [])) or 'None'}"
        )

        for index, region in enumerate(
            regions,
            start=1
        ):

            print(
                f"[DELIVERY {index}] "
                f"{region.get('start', 0):.2f}s "
                f"-> "
                f"{region.get('end', 0):.2f}s | "
                f"{region.get('family', '')} | "
                f"{region.get('flaw_type', '')} | "
                f"{region.get('severity', '')}"
            )

        print()
        print(
            "[OK] OratorIQ combined analysis complete."
        )

        return response

    except HTTPException:
        raise

    except Exception as exc:

        print()
        print("=" * 70)
        print("ORATORIQ ANALYSIS ERROR")
        print("=" * 70)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "OratorIQ analysis failed. "
                f"{type(exc).__name__}: {exc}"
            )
        ) from exc


# ============================================================
# EVALUATION
# ============================================================

@app.post("/api/analyze/standalone")
async def analyze_audio_standalone(
    audio: UploadFile = File(...),
) -> dict[str, Any]:
    """Analyze one recording without a clean reference recording."""
    uploaded_path = None
    original_name = audio.filename or ""

    try:
        suffix = Path(original_name).suffix.lower()

        if suffix not in STANDALONE_AUDIO_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported standalone audio format. "
                    "Use WAV, FLAC, MP3, or OGG."
                ),
            )

        uploaded_path, original_name = await save_upload(
            audio,
            max_size_bytes=MAX_STANDALONE_UPLOAD_BYTES,
        )
        audio_stats = validate_standalone_audio(uploaded_path)

        logger.info(
            "Starting standalone analysis for %s",
            original_name,
        )

        fluency = run_fluency_analysis(uploaded_path)
        analysis_status = (
            "completed"
            if fluency.get("available")
            else "partial"
        )
        if not fluency.get("available"):
            logger.warning(
                "Standalone fluency inference unavailable for %s: %s",
                original_name,
                fluency.get("error") or fluency.get("warning"),
            )

        unavailable_reason = (
            "The delivery models require a matched clean reference "
            "and compare audio windows against it. No reference "
            "comparison was performed."
        )
        timeline_reason = (
            "V4 provides recording-level predictions only and does "
            "not return event timestamps."
        )
        delivery = {
            "available": False,
            "status": "unavailable",
            "reason": unavailable_reason,
            "detected": None,
            "overall_confidence": None,
            "regions": [],
            "window_predictions": [],
        }
        timeline = {
            "available": False,
            "status": "unavailable",
            "reason": timeline_reason,
            "regions": [],
            "windows": [],
        }
        coaching = generate_rag_coaching(
            {
                "analysis_mode": "standalone",
                "regions": [],
            },
            fluency,
        )

        return {
            "analysis_mode": "standalone",
            "analysis_mode_label": "Standalone Analysis",
            "processing_status": analysis_status,
            "audio_path": original_name,
            "audio_name": original_name,
            "reference_audio": None,
            "pair_id": None,
            "transcript": None,
            "sample_rate": audio_stats["sample_rate"],
            "duration": audio_stats["duration"],
            "rms": audio_stats["rms"],
            "channels": audio_stats["channels"],
            "audio_validation": {
                "status": "decoded_non_silent",
                "speech_presence_verified": False,
                "message": (
                    "The file was decoded and contains non-silent audio. "
                    "Speech presence is not independently verified."
                ),
            },
            "score": None,
            "flaws_detected": [],
            "delivery": delivery,
            "comparison": {
                "available": False,
                "status": "unavailable",
                "reason": unavailable_reason,
            },
            "fluency": fluency,
            "analysis": {
                "mode": "standalone",
                "status": analysis_status,
                "detected": None,
                "confidence": None,
                "delivery": delivery,
                "fluency": fluency,
            },
            "timeline": timeline,
            "explanation": None,
            "coaching": coaching,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Standalone analysis failed for %s",
            original_name or "(unnamed upload)",
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Standalone audio analysis failed. "
                "Check the backend logs for details."
            ),
        ) from exc
    finally:
        if uploaded_path is not None:
            try:
                uploaded_path.unlink(missing_ok=True)
            except OSError:
                logger.exception(
                    "Could not remove temporary standalone upload %s",
                    uploaded_path,
                )


@app.post("/api/analyze/auto")
async def analyze_audio_auto(
    audio: UploadFile = File(...),
) -> dict[str, Any]:
    """Use contrastive analysis when its matching clean reference exists."""
    reference_path, _ = find_good_reference(audio.filename or "")

    if reference_path is not None:
        return await analyze_audio(audio)

    return await analyze_audio_standalone(audio)


@app.get("/api/evaluation")
def evaluation() -> dict[str, Any]:
    """
    Keep the existing evaluation endpoint.
    """

    try:

        from speechlab.evaluation import (
            run_evaluation
        )

        return run_evaluation(
            str(DATA_DIR)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Evaluation failed: "
                f"{exc}"
            )
        )