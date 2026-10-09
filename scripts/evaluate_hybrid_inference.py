"""
OratorIQ - Hybrid Inference Evaluator
=====================================

Runs the existing hybrid inference pipeline against one known example
for each of the 6 synthetic flaw types.

This version is Windows-safe:
- Forces UTF-8 for the child inference process.
- Uses ASCII-only console markers.
- Uses metadata.jsonl to locate real audio files.
- Uses the transcript stored in metadata.
- Runs infer_oratoriq_hybrid.py once per flaw.
- Saves individual inference JSON files.
- Produces final CSV + JSON summaries.
- Uses the final console RESULT as the primary prediction source.
"""

from __future__ import annotations

# ============================================================
# WINDOWS UTF-8 FIX
# ============================================================

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

METADATA_FILE = (
    ROOT
    / "data"
    / "contrastive"
    / "metadata.jsonl"
)

INFERENCE_SCRIPT = (
    ROOT
    / "scripts"
    / "infer_oratoriq_hybrid.py"
)

RESULT_DIR = (
    ROOT
    / "artifacts"
    / "results"
    / "hybrid_evaluation"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_JSON = (
    RESULT_DIR
    / "hybrid_evaluation_summary.json"
)

SUMMARY_CSV = (
    RESULT_DIR
    / "hybrid_evaluation_summary.csv"
)

LIVE_RESULT_JSON = (
    ROOT
    / "artifacts"
    / "results"
    / "inference_hybrid_result.json"
)


# ============================================================
# REPRESENTATIVE TEST CASES
# ============================================================

TARGET_PAIRS = {
    "fast_pacing": "1272-128104-0000",
    "slow_pacing": "1272-128104-0001",
    "high_volume": "1272-128104-0005",
    "low_volume": "1272-128104-0014",
    "pitch_deviation": "1272-128104-0009",
    "long_pause": "1272-128104-0012",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def normalize_path(value: str) -> Path:
    """
    Convert metadata paths such as:

        data\\contrastive\\audio\\good\\abc.wav

    into an absolute path under ROOT.
    """

    p = Path(
        str(value).replace("\\", "/")
    )

    if p.is_absolute():
        return p

    return ROOT / p


# ============================================================
# METADATA
# ============================================================

def load_metadata() -> List[Dict[str, Any]]:
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_FILE}"
        )

    records: List[Dict[str, Any]] = []

    with METADATA_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line_no, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:
                print(
                    f"WARNING: bad JSON at line "
                    f"{line_no}: {exc}"
                )

    print(
        f"Loaded metadata records: "
        f"{len(records):,}"
    )

    return records


# ============================================================
# FIND GOOD + FLAWED RECORDS
# ============================================================

def find_records_for_pair(
    records: List[Dict[str, Any]],
    pair_id: str,
    flaw_type: str,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any]
]:

    pair_records = [
        r
        for r in records
        if str(
            r.get("pair_id", "")
        ) == pair_id
    ]

    if not pair_records:
        raise RuntimeError(
            f"No metadata records found "
            f"for pair_id={pair_id}"
        )

    good_candidates = [
        r
        for r in pair_records
        if (
            str(
                r.get("label", "")
            ).lower() == "good"
            or
            str(
                r.get("flaw_type", "")
            ).lower()
            in {
                "",
                "none",
                "null",
            }
        )
    ]

    flawed_candidates = [
        r
        for r in pair_records
        if str(
            r.get("flaw_type", "")
        ).lower()
        == flaw_type.lower()
    ]

    if not good_candidates:
        raise RuntimeError(
            f"No GOOD record found "
            f"for pair_id={pair_id}"
        )

    if not flawed_candidates:

        available = sorted(
            {
                str(
                    r.get("flaw_type")
                )
                for r in pair_records
                if r.get("flaw_type")
            }
        )

        raise RuntimeError(
            f"No {flaw_type} record found "
            f"for pair_id={pair_id}. "
            f"Available flaw types: "
            f"{available}"
        )

    return (
        good_candidates[0],
        flawed_candidates[0],
    )


# ============================================================
# AUDIO PATH RESOLUTION
# ============================================================

def resolve_audio_path(
    record: Dict[str, Any]
) -> Path:

    raw_audio = str(
        record.get("audio", "")
    ).strip()

    if not raw_audio:
        raise FileNotFoundError(
            "Metadata record has no audio path: "
            f"{record.get('id')}"
        )

    direct = normalize_path(
        raw_audio
    )

    # --------------------------------------------------------
    # Exact metadata path
    # --------------------------------------------------------

    if direct.exists():
        return direct

    # --------------------------------------------------------
    # Fallback: exact filename
    # --------------------------------------------------------

    audio_root = (
        ROOT
        / "data"
        / "contrastive"
        / "audio"
    )

    if audio_root.exists():

        exact_matches = list(
            audio_root.rglob(
                direct.name
            )
        )

        if exact_matches:
            return exact_matches[0]

    # --------------------------------------------------------
    # Fallback: pair ID + flaw type
    # --------------------------------------------------------

    pair_id = str(
        record.get(
            "pair_id",
            ""
        )
    ).strip()

    flaw_type = str(
        record.get(
            "flaw_type",
            ""
        )
    ).strip()

    if audio_root.exists() and pair_id:

        candidates = list(
            audio_root.rglob(
                f"{pair_id}*.wav"
            )
        )

        if flaw_type:

            typed = [
                p
                for p in candidates
                if flaw_type.lower()
                in p.name.lower()
            ]

            if typed:
                return typed[0]

        if candidates:
            return candidates[0]

    raise FileNotFoundError(
        "Could not resolve audio file.\n"
        f"Metadata path: {raw_audio}\n"
        f"Resolved path: {direct}"
    )


# ============================================================
# GROUND TRUTH
# ============================================================

def get_ground_truth(
    record: Dict[str, Any]
) -> Dict[str, Any]:

    start = record.get(
        "flaw_start"
    )

    end = record.get(
        "flaw_end"
    )

    try:
        start = (
            float(start)
            if start is not None
            else None
        )
    except (
        TypeError,
        ValueError,
    ):
        start = None

    try:
        end = (
            float(end)
            if end is not None
            else None
        )
    except (
        TypeError,
        ValueError,
    ):
        end = None

    return {
        "flaw_type": record.get(
            "flaw_type"
        ),
        "severity": record.get(
            "severity"
        ),
        "flaw_start": start,
        "flaw_end": end,
        "original_duration": record.get(
            "original_duration"
        ),
    }


# ============================================================
# JSON HELPERS
# ============================================================

def safe_json_load(
    path: Path
) -> Optional[Dict[str, Any]]:

    if not path.exists():
        return None

    try:

        with path.open(
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as exc:

        print(
            f"WARNING: Could not read "
            f"{path}: {exc}"
        )

        return None


def recursive_find(
    obj: Any,
    wanted_keys: set[str],
) -> List[Tuple[str, Any]]:

    found: List[
        Tuple[str, Any]
    ] = []

    if isinstance(obj, dict):

        for key, value in obj.items():

            if str(key).lower() in wanted_keys:

                found.append(
                    (
                        str(key),
                        value,
                    )
                )

            found.extend(
                recursive_find(
                    value,
                    wanted_keys
                )
            )

    elif isinstance(obj, list):

        for item in obj:

            found.extend(
                recursive_find(
                    item,
                    wanted_keys
                )
            )

    return found


def first_value(
    data: Optional[Dict[str, Any]],
    keys: set[str],
    default: Any = None,
) -> Any:

    if not data:
        return default

    values = recursive_find(
        data,
        {
            k.lower()
            for k in keys
        },
    )

    return (
        values[0][1]
        if values
        else default
    )


# ============================================================
# FAMILY DECODER
# ============================================================

def decode_family(
    value: Any
) -> str:

    if value is None:
        return "unknown"

    text = str(
        value
    ).strip().lower()

    if text in {
        "acoustic",
        "temporal",
    }:
        return text

    if text == "0":
        return "acoustic"

    if text == "1":
        return "temporal"

    return str(value)


# ============================================================
# TEMPORAL METRICS
# ============================================================

def overlap_seconds(
    a_start: Optional[float],
    a_end: Optional[float],
    b_start: Optional[float],
    b_end: Optional[float],
) -> float:

    if None in (
        a_start,
        a_end,
        b_start,
        b_end,
    ):
        return 0.0

    left = max(
        float(a_start),
        float(b_start)
    )

    right = min(
        float(a_end),
        float(b_end)
    )

    return max(
        0.0,
        right - left
    )


def interval_iou(
    pred_start: Optional[float],
    pred_end: Optional[float],
    gt_start: Optional[float],
    gt_end: Optional[float],
) -> Optional[float]:

    if None in (
        pred_start,
        pred_end,
        gt_start,
        gt_end,
    ):
        return None

    pred_start = float(
        pred_start
    )

    pred_end = float(
        pred_end
    )

    gt_start = float(
        gt_start
    )

    gt_end = float(
        gt_end
    )

    if (
        pred_end <= pred_start
        or
        gt_end <= gt_start
    ):
        return 0.0

    intersection = overlap_seconds(
        pred_start,
        pred_end,
        gt_start,
        gt_end,
    )

    union = (
        (pred_end - pred_start)
        +
        (gt_end - gt_start)
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection / union
    )


# ============================================================
# EXTRACT PREDICTION FROM JSON
# ============================================================

def extract_prediction(
    result: Optional[Dict[str, Any]]
) -> Dict[str, Any]:

    empty = {
        "pred_flaw_type": None,
        "pred_family": None,
        "pred_severity": None,
        "pred_start": None,
        "pred_end": None,
        "temporal_confidence": None,
        "family_confidence": None,
        "flaw_type_confidence": None,
        "severity_confidence": None,
    }

    if not result:
        return empty

    pred_type = first_value(
        result,
        {
            "flaw",
            "flaw_type",
            "predicted_flaw_type",
            "prediction_flaw_type",
        },
    )

    family = first_value(
        result,
        {
            "family",
            "flaw_family",
            "predicted_family",
            "prediction_family",
        },
    )

    severity = first_value(
        result,
        {
            "severity",
            "predicted_severity",
            "prediction_severity",
        },
    )

    pred_start = first_value(
        result,
        {
            "start",
            "region_start",
            "flaw_start",
            "predicted_start",
            "prediction_start",
        },
    )

    pred_end = first_value(
        result,
        {
            "end",
            "region_end",
            "flaw_end",
            "predicted_end",
            "prediction_end",
        },
    )

    temporal_conf = first_value(
        result,
        {
            "temporal_confidence",
            "temporal_conf",
        },
    )

    family_conf = first_value(
        result,
        {
            "family_confidence",
            "family_conf",
        },
    )

    type_conf = first_value(
        result,
        {
            "flaw_type_confidence",
            "type_confidence",
            "flaw_confidence",
        },
    )

    severity_conf = first_value(
        result,
        {
            "severity_confidence",
            "severity_conf",
        },
    )

    # --------------------------------------------------------
    # Prefer detected region if available
    # --------------------------------------------------------

    region_candidates = []

    if isinstance(result, dict):

        for key in (
            "detected_regions",
            "regions",
            "detections",
        ):

            value = result.get(key)

            if isinstance(value, list):

                region_candidates.extend(
                    item
                    for item in value
                    if isinstance(
                        item,
                        dict
                    )
                )

    if region_candidates:

        best = region_candidates[0]

        pred_type = (
            best.get("flaw_type")
            or
            best.get("flaw")
            or
            best.get(
                "predicted_flaw_type"
            )
            or
            pred_type
        )

        family = (
            best.get("family")
            or
            best.get("flaw_family")
            or
            best.get(
                "predicted_family"
            )
            or
            family
        )

        severity = (
            best.get("severity")
            or
            best.get(
                "predicted_severity"
            )
            or
            severity
        )

        if best.get("start") is not None:
            pred_start = best.get(
                "start"
            )

        elif best.get(
            "region_start"
        ) is not None:

            pred_start = best.get(
                "region_start"
            )

        elif best.get(
            "flaw_start"
        ) is not None:

            pred_start = best.get(
                "flaw_start"
            )

        if best.get("end") is not None:
            pred_end = best.get(
                "end"
            )

        elif best.get(
            "region_end"
        ) is not None:

            pred_end = best.get(
                "region_end"
            )

        elif best.get(
            "flaw_end"
        ) is not None:

            pred_end = best.get(
                "flaw_end"
            )

    def to_float(value):

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return value

    return {
        "pred_flaw_type": (
            str(pred_type)
            if pred_type is not None
            else None
        ),

        "pred_family": decode_family(
            family
        ),

        "pred_severity": (
            str(severity)
            if severity is not None
            else None
        ),

        "pred_start": to_float(
            pred_start
        ),

        "pred_end": to_float(
            pred_end
        ),

        "temporal_confidence": to_float(
            temporal_conf
        ),

        "family_confidence": to_float(
            family_conf
        ),

        "flaw_type_confidence": to_float(
            type_conf
        ),

        "severity_confidence": to_float(
            severity_conf
        ),
    }


# ============================================================
# PARSE FINAL CONSOLE RESULT
# ============================================================

def parse_console_summary(
    stdout: str
) -> Dict[str, Any]:

    out: Dict[str, Any] = {}

    # --------------------------------------------------------
    # Region
    #
    # Example:
    #
    # 1 detected region 1.50-3.00s
    # --------------------------------------------------------

    m = re.search(
        r"(\d+)\s+detected\s+region[s]?"
        r"(?:\s+([0-9.]+)\s*-\s*([0-9.]+)s)?",
        stdout,
        re.IGNORECASE,
    )

    if m:

        if (
            m.group(2) is not None
            and
            m.group(3) is not None
        ):

            out["pred_start"] = float(
                m.group(2)
            )

            out["pred_end"] = float(
                m.group(3)
            )

    # --------------------------------------------------------
    # Family
    # --------------------------------------------------------

    m = re.search(
        r"Family:\s*([^\r\n]+)",
        stdout,
        re.IGNORECASE,
    )

    if m:

        family_text = (
            m.group(1)
            .strip()
        )

        # Remove accidental confidence/text after family.
        family_text = (
            family_text
            .split("Confidence")[0]
            .strip()
        )

        out["pred_family"] = (
            decode_family(
                family_text
            )
        )

    # --------------------------------------------------------
    # Flaw type
    # --------------------------------------------------------

    m = re.search(
        r"Flaw:\s*([^\r\n]+)",
        stdout,
        re.IGNORECASE,
    )

    if m:

        flaw_text = (
            m.group(1)
            .strip()
        )

        flaw_text = (
            flaw_text
            .split("Confidence")[0]
            .strip()
        )

        out["pred_flaw_type"] = (
            flaw_text
        )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    m = re.search(
        r"Severity:\s*([^\r\n]+)",
        stdout,
        re.IGNORECASE,
    )

    if m:

        severity_text = (
            m.group(1)
            .strip()
        )

        severity_text = (
            severity_text
            .split("Confidence")[0]
            .strip()
        )

        out["pred_severity"] = (
            severity_text
        )

    # --------------------------------------------------------
    # Confidence values
    # --------------------------------------------------------

    confidence_patterns = {
        "temporal_confidence":
            r"Temporal Confidence\s+([0-9.]+)%",
        "family_confidence":
            r"Family Confidence\s+([0-9.]+)%",
        "flaw_type_confidence":
            r"Flaw-Type Confidence\s+([0-9.]+)%",
        "severity_confidence":
            r"Severity Confidence\s+([0-9.]+)%",
    }

    for key, pattern in (
        confidence_patterns.items()
    ):

        m = re.search(
            pattern,
            stdout,
            re.IGNORECASE,
        )

        if m:

            out[key] = (
                float(
                    m.group(1)
                )
                / 100.0
            )

    return out


# ============================================================
# MERGE PREDICTIONS
# ============================================================

def merge_prediction(
    json_prediction: Dict[str, Any],
    console_prediction: Dict[str, Any],
) -> Dict[str, Any]:

    # IMPORTANT:
    #
    # Console RESULT is the primary source.
    #
    # This avoids accidentally selecting the first
    # per-window prediction from the JSON file.

    merged = dict(
        console_prediction
    )

    for key, value in json_prediction.items():

        if (
            key not in merged
            or
            merged[key]
            in (
                None,
                "",
                "unknown",
            )
        ):

            merged[key] = value

    return merged


# ============================================================
# RUN ONE CASE
# ============================================================

def run_single_case(
    flaw_type: str,
    pair_id: str,
    good_record: Dict[str, Any],
    flawed_record: Dict[str, Any],
) -> Dict[str, Any]:

    good_audio = (
        resolve_audio_path(
            good_record
        )
    )

    flawed_audio = (
        resolve_audio_path(
            flawed_record
        )
    )

    transcript = str(
        flawed_record.get(
            "transcript"
        )
        or
        good_record.get(
            "transcript"
        )
        or
        ""
    )

    gt = get_ground_truth(
        flawed_record
    )

    case_json = (
        RESULT_DIR
        / f"{flaw_type}_inference.json"
    )

    # --------------------------------------------------------
    # Remove previous live result
    # --------------------------------------------------------

    if LIVE_RESULT_JSON.exists():

        try:
            LIVE_RESULT_JSON.unlink()

        except OSError:
            pass

    # --------------------------------------------------------
    # Command
    # --------------------------------------------------------

    command = [
        sys.executable,
        str(INFERENCE_SCRIPT),

        "--good",
        str(good_audio),

        "--flawed",
        str(flawed_audio),

        "--transcript",
        transcript,
    ]

    print()
    print("-" * 78)

    print(
        f"CASE: {flaw_type.upper()}"
    )

    print(
        f"Pair: {pair_id}"
    )

    print(
        f"Good : {good_audio}"
    )

    print(
        f"Flawed: {flawed_audio}"
    )

    if (
        gt["flaw_start"] is not None
        and
        gt["flaw_end"] is not None
    ):

        print(
            "Ground truth region: "
            f"{gt['flaw_start']:.4f}s - "
            f"{gt['flaw_end']:.4f}s"
        )

    else:

        print(
            "Ground truth region: "
            "unavailable"
        )

    print("-" * 78)

    # --------------------------------------------------------
    # UTF-8 child environment
    # --------------------------------------------------------

    child_env = os.environ.copy()

    child_env[
        "PYTHONIOENCODING"
    ] = "utf-8"

    child_env[
        "PYTHONUTF8"
    ] = "1"

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    completed = subprocess.run(
        command,
        cwd=ROOT,

        capture_output=True,

        text=True,

        encoding="utf-8",

        errors="replace",

        env=child_env,
    )

    # --------------------------------------------------------
    # Print child output
    # --------------------------------------------------------

    if completed.stdout:

        print(
            completed.stdout
        )

    if completed.stderr:

        print(
            "STDERR:"
        )

        print(
            completed.stderr
        )

    # --------------------------------------------------------
    # Failure
    # --------------------------------------------------------

    if completed.returncode != 0:

        return {
            "flaw_type": flaw_type,
            "pair_id": pair_id,
            "status": "FAILED",

            "error": (
                "infer_oratoriq_hybrid.py "
                f"exited with code "
                f"{completed.returncode}"
            ),

            "good_audio": str(
                good_audio
            ),

            "flawed_audio": str(
                flawed_audio
            ),

            "gt_start": gt[
                "flaw_start"
            ],

            "gt_end": gt[
                "flaw_end"
            ],
        }

    # --------------------------------------------------------
    # Read fresh JSON
    # --------------------------------------------------------

    result = safe_json_load(
        LIVE_RESULT_JSON
    )

    # --------------------------------------------------------
    # Save case-specific JSON
    # --------------------------------------------------------

    if result is not None:

        with case_json.open(
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False,
            )

    # --------------------------------------------------------
    # Extract predictions
    # --------------------------------------------------------

    json_prediction = (
        extract_prediction(
            result
        )
    )

    console_prediction = (
        parse_console_summary(
            completed.stdout
        )
    )

    prediction = merge_prediction(
        json_prediction,
        console_prediction,
    )

    # --------------------------------------------------------
    # Normalize prediction
    # --------------------------------------------------------

    pred_type = (
        str(
            prediction.get(
                "pred_flaw_type"
            )
        )
        .strip()
        .lower()
        if prediction.get(
            "pred_flaw_type"
        ) is not None
        else None
    )

    pred_family = (
        str(
            prediction.get(
                "pred_family"
            )
        )
        .strip()
        .lower()
        if prediction.get(
            "pred_family"
        ) is not None
        else None
    )

    # --------------------------------------------------------
    # Expected family
    # --------------------------------------------------------

    temporal_flaws = {
        "fast_pacing",
        "slow_pacing",
        "long_pause",
    }

    expected_family = (
        "temporal"
        if flaw_type in temporal_flaws
        else "acoustic"
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    type_correct = (
        pred_type == flaw_type.lower()
        if pred_type is not None
        else False
    )

    family_correct = (
        pred_family == expected_family
    )

    # --------------------------------------------------------
    # Temporal IoU
    # --------------------------------------------------------

    iou = interval_iou(
        prediction.get(
            "pred_start"
        ),

        prediction.get(
            "pred_end"
        ),

        gt[
            "flaw_start"
        ],

        gt[
            "flaw_end"
        ],
    )

    # --------------------------------------------------------
    # Temporal overlap
    # --------------------------------------------------------

    temporal_overlap = False

    if all(
        x is not None
        for x in (
            prediction.get(
                "pred_start"
            ),

            prediction.get(
                "pred_end"
            ),

            gt[
                "flaw_start"
            ],

            gt[
                "flaw_end"
            ],
        )
    ):

        temporal_overlap = (
            overlap_seconds(
                prediction.get(
                    "pred_start"
                ),

                prediction.get(
                    "pred_end"
                ),

                gt[
                    "flaw_start"
                ],

                gt[
                    "flaw_end"
                ],
            )
            > 0
        )

    # --------------------------------------------------------
    # Result row
    # --------------------------------------------------------

    result_row = {

        "flaw_type":
            flaw_type,

        "pair_id":
            pair_id,

        "status":
            "PASS",

        "family_expected":
            expected_family,

        "family_predicted":
            prediction.get(
                "pred_family"
            ),

        "family_correct":
            family_correct,

        "flaw_type_predicted":
            prediction.get(
                "pred_flaw_type"
            ),

        "flaw_type_correct":
            type_correct,

        "severity_expected":
            gt[
                "severity"
            ],

        "severity_predicted":
            prediction.get(
                "pred_severity"
            ),

        "gt_start":
            gt[
                "flaw_start"
            ],

        "gt_end":
            gt[
                "flaw_end"
            ],

        "pred_start":
            prediction.get(
                "pred_start"
            ),

        "pred_end":
            prediction.get(
                "pred_end"
            ),

        "temporal_overlap":
            temporal_overlap,

        "temporal_iou":
            iou,

        "temporal_confidence":
            prediction.get(
                "temporal_confidence"
            ),

        "family_confidence":
            prediction.get(
                "family_confidence"
            ),

        "flaw_type_confidence":
            prediction.get(
                "flaw_type_confidence"
            ),

        "severity_confidence":
            prediction.get(
                "severity_confidence"
            ),

        "good_audio":
            str(good_audio),

        "flawed_audio":
            str(flawed_audio),

        "result_json":
            str(case_json),
    }

    # --------------------------------------------------------
    # Case result display
    # --------------------------------------------------------

    print()
    print("CASE RESULT")

    print(
        f"  Family : "
        f"{prediction.get('pred_family')} "
        f"({'PASS' if family_correct else 'FAIL'})"
    )

    print(
        f"  Flaw   : "
        f"{prediction.get('pred_flaw_type')} "
        f"({'PASS' if type_correct else 'FAIL'})"
    )

    print(
        f"  Region : "
        f"{prediction.get('pred_start')} - "
        f"{prediction.get('pred_end')} s"
    )

    print(
        f"  GT     : "
        f"{gt['flaw_start']} - "
        f"{gt['flaw_end']} s"
    )

    print(
        f"  Overlap: "
        f"{'PASS' if temporal_overlap else 'FAIL'}"
    )

    if iou is not None:

        print(
            f"  IoU    : {iou:.3f}"
        )

    else:

        print(
            "  IoU    : unavailable"
        )

    return result_row


# ============================================================
# CSV
# ============================================================

def write_csv(
    rows: List[Dict[str, Any]]
) -> None:

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# SUMMARY
# ============================================================

def write_summary(
    rows: List[Dict[str, Any]]
) -> Dict[str, Any]:

    valid = [
        r
        for r in rows
        if r.get("status") == "PASS"
    ]

    total_cases = len(rows)

    successful_cases = len(
        valid
    )

    failed_cases = (
        total_cases
        - successful_cases
    )

    family_correct = sum(
        bool(
            r.get(
                "family_correct"
            )
        )
        for r in valid
    )

    type_correct = sum(
        bool(
            r.get(
                "flaw_type_correct"
            )
        )
        for r in valid
    )

    temporal_overlap = sum(
        bool(
            r.get(
                "temporal_overlap"
            )
        )
        for r in valid
    )

    ious = [
        float(
            r["temporal_iou"]
        )

        for r in valid

        if r.get(
            "temporal_iou"
        ) is not None
    ]

    summary = {

        "project":
            "OratorIQ",

        "evaluation":
            "6-flaw hybrid inference",

        "total_cases":
            total_cases,

        "successful_cases":
            successful_cases,

        "failed_cases":
            failed_cases,

        "family_accuracy":
            (
                family_correct
                / successful_cases
                if successful_cases
                else 0.0
            ),

        "flaw_type_accuracy":
            (
                type_correct
                / successful_cases
                if successful_cases
                else 0.0
            ),

        "temporal_overlap_rate":
            (
                temporal_overlap
                / successful_cases
                if successful_cases
                else 0.0
            ),

        "mean_temporal_iou":
            (
                sum(ious)
                / len(ious)
                if ious
                else None
            ),

        "case_results":
            rows,
    }

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    write_csv(
        rows
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print_header(
        "ORATORIQ - 6-FLAW HYBRID INFERENCE EVALUATION"
    )

    # --------------------------------------------------------
    # Check inference script
    # --------------------------------------------------------

    if not INFERENCE_SCRIPT.exists():

        raise FileNotFoundError(
            "Hybrid inference script not found:\n"
            f"{INFERENCE_SCRIPT}"
        )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    records = load_metadata()

    rows: List[
        Dict[str, Any]
    ] = []

    total_targets = len(
        TARGET_PAIRS
    )

    # --------------------------------------------------------
    # Run six cases
    # --------------------------------------------------------

    for index, (
        flaw_type,
        pair_id,
    ) in enumerate(
        TARGET_PAIRS.items(),
        start=1,
    ):

        print()

        print(
            f"[{index}/{total_targets}] "
            f"Preparing {flaw_type} "
            f"(pair {pair_id})..."
        )

        try:

            good_record, flawed_record = (
                find_records_for_pair(
                    records,
                    pair_id,
                    flaw_type,
                )
            )

            row = run_single_case(
                flaw_type=flaw_type,
                pair_id=pair_id,
                good_record=good_record,
                flawed_record=flawed_record,
            )

            rows.append(
                row
            )

        except Exception as exc:

            print()

            print(
                f"ERROR in "
                f"{flaw_type}: {exc}"
            )

            rows.append(
                {
                    "flaw_type":
                        flaw_type,

                    "pair_id":
                        pair_id,

                    "status":
                        "FAILED",

                    "error":
                        str(exc),
                }
            )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = write_summary(
        rows
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print_header(
        "FINAL 6-FLAW SUMMARY"
    )

    print(
        f"Cases completed : "
        f"{summary['successful_cases']}/"
        f"{summary['total_cases']}"
    )

    print(
        f"Family accuracy : "
        f"{summary['family_accuracy'] * 100:.1f}%"
    )

    print(
        f"Flaw accuracy   : "
        f"{summary['flaw_type_accuracy'] * 100:.1f}%"
    )

    print(
        f"Temporal overlap: "
        f"{summary['temporal_overlap_rate'] * 100:.1f}%"
    )

    if (
        summary[
            "mean_temporal_iou"
        ]
        is not None
    ):

        print(
            f"Mean temporal IoU: "
            f"{summary['mean_temporal_iou']:.3f}"
        )

    print()

    print(
        f"CSV summary : "
        f"{SUMMARY_CSV}"
    )

    print(
        f"JSON summary: "
        f"{SUMMARY_JSON}"
    )

    print(
        f"Individual inference JSON files: "
        f"{RESULT_DIR}"
    )

    # --------------------------------------------------------
    # Per-flaw result
    # --------------------------------------------------------

    print()

    print(
        "PER-FLAW RESULT"
    )

    print(
        "-" * 78
    )

    for row in rows:

        if row.get(
            "status"
        ) != "PASS":

            print(
                f"  {row.get('flaw_type', 'unknown'):18s}"
                f" | FAILED | "
                f"{row.get('error', 'unknown error')}"
            )

            continue

        family_mark = (
            "PASS"
            if row.get(
                "family_correct"
            )
            else "FAIL"
        )

        type_mark = (
            "PASS"
            if row.get(
                "flaw_type_correct"
            )
            else "FAIL"
        )

        overlap_mark = (
            "PASS"
            if row.get(
                "temporal_overlap"
            )
            else "FAIL"
        )

        iou = row.get(
            "temporal_iou"
        )

        if iou is None:
            iou_text = "N/A"
        else:
            iou_text = f"{float(iou):.3f}"

        print(
            f"  {row['flaw_type']:18s}"
            f" | family {family_mark:4s}"
            f" | type {type_mark:4s}"
            f" | temporal {overlap_mark:4s}"
            f" | IoU {iou_text}"
        )

    print()

    print(
        "Evaluation finished."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()