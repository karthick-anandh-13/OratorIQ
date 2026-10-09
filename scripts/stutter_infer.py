import ast
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

V2_SCRIPT = ROOT / "scripts" / "stutter_pipeline_v2.py"
MODEL_DIR = ROOT / "artifacts" / "stutter_models"
RESULT_DIR = ROOT / "artifacts" / "stutter_results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD ONLY extract_features() FROM V2
# ============================================================

def load_v2_extractor():

    print("[1/5] Loading V2 feature extractor...")

    source = V2_SCRIPT.read_text(encoding="utf-8")

    tree = ast.parse(source)

    function_node = None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "extract_features":
                function_node = node
                break

    if function_node is None:
        raise RuntimeError(
            "Could not find extract_features() in stutter_pipeline_v2.py"
        )

    # We also need the helper functions used by extract_features:
    # safe_mean, safe_std, safe_median, safe_percentile
    helper_names = {
        "safe_mean",
        "safe_std",
        "safe_median",
        "safe_percentile",
        "temporal_statistics",
        "count_segments",
        "calculate_pause_features",
        "calculate_rate_features",
    }

    selected_nodes = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name in helper_names or node.name == "extract_features":
                selected_nodes.append(node)

    # Execute only imports + constants + helper functions + extractor.
    safe_nodes = []

    for node in tree.body:

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            safe_nodes.append(node)

        elif isinstance(node, ast.Assign):
            # Keep constants such as SR and N_MFCC.
            names = []

            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)

            if any(
                name in {
                    "SR",
                    "N_MFCC",
                }
                for name in names
            ):
                safe_nodes.append(node)

    safe_nodes.extend(selected_nodes)

    module = ast.Module(
        body=safe_nodes,
        type_ignores=[],
    )

    ast.fix_missing_locations(module)

    namespace = {
        "__file__": str(V2_SCRIPT),
        "__name__": "stutter_feature_extractor",
    }

    compiled = compile(
        module,
        str(V2_SCRIPT),
        "exec",
    )

    exec(compiled, namespace)

    extractor = namespace.get("extract_features")

    if extractor is None:
        raise RuntimeError(
            "extract_features() could not be loaded."
        )

    print(
        "      OK - V2 training code was NOT executed."
    )

    return extractor


# ============================================================
# LOAD V4 MODELS
# ============================================================

def load_models():

    print("[2/5] Loading V4 models...")

    detector_artifact = joblib.load(
        MODEL_DIR / "stutter_detector_v4.joblib"
    )

    detector = detector_artifact["model"]
    detector_threshold = float(
        detector_artifact["threshold"]
    )

    event_files = {
        "Prolongation":
            "stutter_v4_event_prolongation.joblib",

        "Block":
            "stutter_v4_event_block.joblib",

        "SoundRep":
            "stutter_v4_event_soundrep.joblib",

        "WordRep":
            "stutter_v4_event_wordrep.joblib",

        "Interjection":
            "stutter_v4_event_interjection.joblib",
    }

    event_models = {}

    for event_name, filename in event_files.items():

        artifact = joblib.load(
            MODEL_DIR / filename
        )

        event_models[event_name] = {
            "model": artifact["model"],
            "threshold": float(
                artifact["threshold"]
            ),
        }

    print(
        f"      Detector features: "
        f"{detector.n_features_in_}"
    )

    print(
        f"      Detector threshold: "
        f"{detector_threshold:.2f}"
    )

    for name, info in event_models.items():

        print(
            f"      {name:18s}: "
            f"{info['model'].n_features_in_} features, "
            f"threshold={info['threshold']:.2f}"
        )

    return (
        detector,
        detector_threshold,
        event_models,
    )


# ============================================================
# BUILD V4 VECTOR
# ============================================================

def build_vector(features, expected_features):

    # The V2 extractor returns exactly 121 audio features.
    audio = pd.DataFrame([features])

    audio = audio.apply(
        pd.to_numeric,
        errors="coerce",
    )

    audio = audio.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    audio = audio.fillna(0.0)

    audio_features = list(audio.columns)

    print(
        f"      Audio features: {len(audio_features)}"
    )

    if len(audio_features) != 121:

        raise RuntimeError(
            f"Expected 121 audio features, "
            f"got {len(audio_features)}"
        )

    # V4 has 7 additional annotation-derived inputs.
    #
    # These cannot be obtained from a new WAV.
    # The V4 demo adapter uses zero.
    annotation_features = [
        "Unsure",
        "PoorAudioQuality",
        "DifficultToUnderstand",
        "NoStutteredWords",
        "NaturalPause",
        "Music",
        "NoSpeech",
    ]

    values = []

    for name in annotation_features:
        values.append(0.0)

    for name in audio_features:
        values.append(
            float(audio.iloc[0][name])
        )

    X = np.asarray(
        values,
        dtype=np.float32,
    ).reshape(1, -1)

    print(
        f"      Final V4 vector: {X.shape[1]}"
    )

    if X.shape[1] != expected_features:

        raise RuntimeError(
            f"V4 expects {expected_features} features "
            f"but generated {X.shape[1]}"
        )

    return X


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    detector,
    detector_threshold,
    event_models,
    X,
):

    print("[4/5] Running inference...")

    probability = float(
        detector.predict_proba(X)[0, 1]
    )

    detected = (
        probability >= detector_threshold
    )

    events = {}

    for name, info in event_models.items():

        model = info["model"]
        threshold = info["threshold"]

        p = float(
            model.predict_proba(X)[0, 1]
        )

        events[name] = {
            "probability": p,
            "probability_percent": p * 100,
            "threshold": threshold,
            "detected": p >= threshold,
        }

    return {
        "stutter_probability": probability,
        "stutter_probability_percent":
            probability * 100,
        "threshold": detector_threshold,
        "detected": detected,
        "events": events,
    }


# ============================================================
# API / REUSABLE INFERENCE
# ============================================================

_V4_CACHE = None


def run_v4_inference(audio_path):
    """
    Reusable V4 fluency inference for the FastAPI backend.

    Returns a JSON-serializable dictionary.
    """

    global _V4_CACHE

    started = time.time()

    print()
    print("=" * 70)
    print("ORATORIQ - V4 FLUENCY API INFERENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load models/extractor only once
    # --------------------------------------------------------

    if _V4_CACHE is None:

        print("[1/4] Loading V2 feature extractor...")
        extractor = load_v2_extractor()

        print("[2/4] Loading V4 models...")
        (
            detector,
            detector_threshold,
            event_models,
        ) = load_models()

        _V4_CACHE = {
            "extractor": extractor,
            "detector": detector,
            "detector_threshold": detector_threshold,
            "event_models": event_models,
        }

        print("      V4 models cached.")

    else:

        print("[1/4] Using cached V4 models...")
        extractor = _V4_CACHE["extractor"]
        detector = _V4_CACHE["detector"]
        detector_threshold = _V4_CACHE["detector_threshold"]
        event_models = _V4_CACHE["event_models"]

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    print("[3/4] Extracting audio features...")

    features = extractor(str(audio_path))

    if features is None:
        raise RuntimeError(
            "V4 feature extraction failed."
        )

    print(
        f"      Extracted: {len(features)} audio features"
    )

    X = build_vector(
        features,
        detector.n_features_in_,
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("[4/4] Running V4 inference...")

    prediction = run_inference(
        detector,
        detector_threshold,
        event_models,
        X,
    )

    elapsed = time.time() - started

    # --------------------------------------------------------
    # JSON-safe response
    # --------------------------------------------------------

    result = {
        "model": "V4",
        "mode": "audio_only_demo",
        "warning": (
            "V4 includes seven annotation-derived features "
            "unavailable from raw audio. They were set to zero "
            "for this demo."
        ),
        "prediction": prediction,
        "runtime_seconds": round(
            elapsed,
            3,
        ),
    }

    print()
    print("V4 RESULT")
    print(
        f"Stutter probability : "
        f"{prediction['stutter_probability_percent']:.2f}%"
    )

    print(
        f"Detected            : "
        f"{prediction['detected']}"
    )

    print("Possible events:")

    for name, event in prediction["events"].items():

        marker = (
            " <-- POSSIBLE"
            if event["detected"]
            else ""
        )

        print(
            f"  {name:18s} "
            f"{event['probability_percent']:.2f}%"
            f"{marker}"
        )

    print(
        f"Runtime             : "
        f"{elapsed:.2f} seconds"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    started = time.time()

    print()
    print("=" * 70)
    print("ORATORIQ - V4 STUTTER / FLUENCY INFERENCE")
    print("=" * 70)

    if len(sys.argv) < 2:

        print(
            'Usage: python scripts\\stutter_infer.py "audio.wav"'
        )

        return

    audio_path = Path(sys.argv[1])

    if not audio_path.is_absolute():
        audio_path = ROOT / audio_path

    audio_path = audio_path.resolve()

    print()
    print(
        f"Input: {audio_path}"
    )

    if not audio_path.exists():

        raise FileNotFoundError(
            audio_path
        )

    extractor = load_v2_extractor()

    (
        detector,
        detector_threshold,
        event_models,
    ) = load_models()

    print("[3/5] Extracting audio features...")

    features = extractor(
        str(audio_path)
    )

    if features is None:

        raise RuntimeError(
            "Feature extraction failed."
        )

    print(
        f"      Extracted: {len(features)} features"
    )

    X = build_vector(
        features,
        detector.n_features_in_,
    )

    result = run_inference(
        detector,
        detector_threshold,
        event_models,
        X,
    )

    elapsed = time.time() - started

    output = {
        "system": "OratorIQ",
        "model": "V4",
        "mode": "audio_only_demo",
        "audio": str(audio_path),
        "warning": (
            "V4 includes seven annotation-derived "
            "features unavailable from raw audio. "
            "They were set to zero for this demo."
        ),
        "prediction": result,
        "runtime_seconds": elapsed,
    }

    output_path = (
        RESULT_DIR /
        "stutter_inference_result.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print("[5/5] Complete.")

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"Stutter probability : "
        f"{result['stutter_probability_percent']:.2f}%"
    )

    print(
        f"Detected            : "
        f"{result['detected']}"
    )

    print()
    print("Possible events:")

    for name, event in result["events"].items():

        marker = (
            " <-- POSSIBLE"
            if event["detected"]
            else ""
        )

        print(
            f"  {name:18s} "
            f"{event['probability_percent']:6.2f}%"
            f"{marker}"
        )

    print()
    print(
        f"Runtime : {elapsed:.2f} seconds"
    )

    print(
        f"JSON    : {output_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()