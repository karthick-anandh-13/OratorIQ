from pathlib import Path
import json
import argparse


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\OratorIQ")
RESULTS = ROOT / "artifacts" / "results"

DEFAULT_INFERENCE = RESULTS / "inference_hybrid_result.json"
DEFAULT_EXPLANATION = RESULTS / "oratoriq_explanation.json"
DEFAULT_OUTPUT = RESULTS / "oratoriq_timeline.json"


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# WINDOW PROCESSING
# ============================================================

def build_window_timeline(inference):
    """
    Convert inference window predictions into a clean,
    frontend-friendly timeline.
    """

    windows = inference.get(
        "window_predictions",
        []
    )

    timeline = []

    for index, window in enumerate(
        windows,
        start=1
    ):

        start = safe_float(
            window.get("start")
        )

        end = safe_float(
            window.get("end")
        )

        temporal_confidence = safe_float(
            window.get(
                "temporal_confidence"
            )
        )

        detected = temporal_confidence >= 0.50

        flaw_type = window.get(
            "final_flaw_type",
            window.get(
                "original_flaw_type"
            )
        )

        family = window.get(
            "family"
        )

        severity = window.get(
            "severity"
        )

        evidence = window.get(
            "evidence",
            {}
        )

        timeline.append({
            "window_id": index,

            "start": round(
                start,
                3
            ),

            "end": round(
                end,
                3
            ),

            "duration": round(
                max(0.0, end - start),
                3
            ),

            "detected": detected,

            "temporal_probability": round(
                temporal_confidence,
                4
            ),

            "family": family,

            "flaw_type": flaw_type,

            "severity": severity,

            "confidence": {
                "temporal": round(
                    temporal_confidence,
                    4
                ),

                "family": round(
                    safe_float(
                        window.get(
                            "family_confidence"
                        )
                    ),
                    4
                ),

                "flaw_type": round(
                    safe_float(
                        window.get(
                            "flaw_type_confidence"
                        )
                    ),
                    4
                ),

                "severity": round(
                    safe_float(
                        window.get(
                            "severity_confidence"
                        )
                    ),
                    4
                ),
            },

            "evidence": {
                "rms_delta": round(
                    safe_float(
                        evidence.get(
                            "rms_delta"
                        )
                    ),
                    6
                ),

                "silence_delta": round(
                    safe_float(
                        evidence.get(
                            "silence_delta"
                        )
                    ),
                    6
                ),

                "silence_ratio": round(
                    safe_float(
                        evidence.get(
                            "silence_ratio"
                        )
                    ),
                    6
                ),

                "f0_delta": round(
                    safe_float(
                        evidence.get(
                            "f0_delta"
                        )
                    ),
                    6
                ),

                "f0_ratio": round(
                    safe_float(
                        evidence.get(
                            "f0_ratio"
                        )
                    ),
                    6
                ),

                "wpm_delta": round(
                    safe_float(
                        evidence.get(
                            "wpm_delta"
                        )
                    ),
                    6
                ),

                "wpm_ratio": round(
                    safe_float(
                        evidence.get(
                            "wpm_ratio"
                        )
                    ),
                    6
                ),

                "window_duration_delta": round(
                    safe_float(
                        evidence.get(
                            "window_duration_delta"
                        )
                    ),
                    6
                ),

                "window_duration_ratio": round(
                    safe_float(
                        evidence.get(
                            "window_duration_ratio"
                        )
                    ),
                    6
                ),

                "alignment_error": round(
                    safe_float(
                        evidence.get(
                            "alignment_error"
                        )
                    ),
                    6
                ),
            },
        })

    return timeline


# ============================================================
# DETECTED REGIONS
# ============================================================

def build_regions(
    inference,
    explanation
):
    """
    Prefer merged regions from the explanation layer.

    This gives the dashboard:
        1.50 -> 3.00
    rather than displaying two overlapping windows:
        1.50 -> 2.50
        2.00 -> 3.00
    """

    regions = explanation.get(
        "regions",
        []
    )

    output = []

    for index, region in enumerate(
        regions,
        start=1
    ):

        time = region.get(
            "time",
            {}
        )

        detection = region.get(
            "detection",
            {}
        )

        confidence = region.get(
            "confidence",
            {}
        )

        explanation_data = region.get(
            "explanation",
            {}
        )

        output.append({

            "region_id": index,

            "start": safe_float(
                time.get("start")
            ),

            "end": safe_float(
                time.get("end")
            ),

            "duration": safe_float(
                time.get(
                    "duration"
                )
            ),

            "flaw_type": detection.get(
                "flaw_type"
            ),

            "flaw_label": detection.get(
                "flaw_label"
            ),

            "family": detection.get(
                "family"
            ),

            "family_label": detection.get(
                "family_label"
            ),

            "severity": detection.get(
                "severity"
            ),

            "severity_label": detection.get(
                "severity_label"
            ),

            "confidence": {
                "overall": safe_float(
                    confidence.get(
                        "overall"
                    )
                ),

                "overall_percent": safe_float(
                    confidence.get(
                        "overall_percent"
                    )
                ),

                "level": confidence.get(
                    "level"
                ),
            },

            "explanation": {
                "summary": explanation_data.get(
                    "summary"
                ),

                "evidence": explanation_data.get(
                    "evidence",
                    []
                ),

                "recommendation":
                    explanation_data.get(
                        "recommendation"
                    ),

                "evidence_values":
                    explanation_data.get(
                        "evidence_values",
                        {}
                    ),
            },
        })

    return output


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    inference,
    explanation,
    windows,
    regions
):

    duration = safe_float(
        inference.get(
            "duration"
        )
    )

    detected_windows = [
        w for w in windows
        if w["detected"]
    ]

    flaw_counts = {}

    for region in regions:

        flaw = region.get(
            "flaw_type"
        )

        if not flaw:
            continue

        flaw_counts[flaw] = (
            flaw_counts.get(
                flaw,
                0
            ) + 1
        )

    return {

        "duration": round(
            duration,
            3
        ),

        "total_windows": len(
            windows
        ),

        "detected_windows": len(
            detected_windows
        ),

        "detected_regions": len(
            regions
        ),

        "overall_confidence":
            safe_float(
                explanation.get(
                    "summary",
                    {}
                ).get(
                    "overall_confidence"
                )
            ),

        "overall_confidence_percent":
            safe_float(
                explanation.get(
                    "summary",
                    {}
                ).get(
                    "overall_confidence_percent"
                )
            ),

        "flaw_counts":
            flaw_counts,
    }


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_timeline(
    inference,
    explanation
):

    windows = build_window_timeline(
        inference
    )

    regions = build_regions(
        inference,
        explanation
    )

    summary = build_summary(
        inference,
        explanation,
        windows,
        regions
    )

    return {

        "system": "OratorIQ",

        "version":
            "timeline_v1",

        "audio":
            inference.get(
                "audio",
                ""
            ),

        "duration":
            safe_float(
                inference.get(
                    "duration"
                )
            ),

        "summary":
            summary,

        "regions":
            regions,

        "windows":
            windows,
    }


# ============================================================
# DISPLAY
# ============================================================

def display_timeline(
    timeline
):

    print()
    print("=" * 70)
    print("ORATORIQ TIMELINE")
    print("=" * 70)

    summary = timeline[
        "summary"
    ]

    print()

    print(
        f"Audio duration: "
        f"{summary['duration']:.2f}s"
    )

    print(
        f"Total windows: "
        f"{summary['total_windows']}"
    )

    print(
        f"Detected windows: "
        f"{summary['detected_windows']}"
    )

    print(
        f"Detected regions: "
        f"{summary['detected_regions']}"
    )

    print(
        f"Overall confidence: "
        f"{summary['overall_confidence_percent']:.1f}%"
    )

    print()

    print("-" * 70)
    print("DETECTED REGIONS")
    print("-" * 70)

    if not timeline["regions"]:

        print(
            "[OK] No detected speech flaws."
        )

    else:

        for region in timeline[
            "regions"
        ]:

            print(
                f"[{region['region_id']}] "
                f"{region['start']:.2f}s -> "
                f"{region['end']:.2f}s | "
                f"{region['flaw_label']} | "
                f"{region['severity_label']} | "
                f"{region['confidence']['overall_percent']:.1f}%"
            )

    print()

    print("-" * 70)
    print("WINDOW TIMELINE")
    print("-" * 70)

    for window in timeline[
        "windows"
    ]:

        status = (
            "DETECTED"
            if window["detected"]
            else "normal"
        )

        flaw = (
            window["flaw_type"]
            if window["detected"]
            else "-"
        )

        print(
            f"{window['start']:>5.2f}s -> "
            f"{window['end']:>5.2f}s | "
            f"{window['temporal_probability']:.3f} | "
            f"{status:<8} | "
            f"{flaw}"
        )

    print()

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate a frontend-ready "
            "OratorIQ timeline JSON."
        )
    )

    parser.add_argument(
        "--inference",
        default=str(
            DEFAULT_INFERENCE
        ),
        help=(
            "Path to inference_hybrid_result.json"
        )
    )

    parser.add_argument(
        "--explanation",
        default=str(
            DEFAULT_EXPLANATION
        ),
        help=(
            "Path to oratoriq_explanation.json"
        )
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT
        ),
        help=(
            "Path to output timeline JSON"
        )
    )

    args = parser.parse_args()

    inference_path = Path(
        args.inference
    )

    explanation_path = Path(
        args.explanation
    )

    output_path = Path(
        args.output
    )

    print()
    print("=" * 70)
    print("ORATORIQ TIMELINE GENERATOR")
    print("=" * 70)

    print()

    print(
        f"Inference:   {inference_path}"
    )

    print(
        f"Explanation: {explanation_path}"
    )

    print(
        f"Output:      {output_path}"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    inference = load_json(
        inference_path
    )

    print(
        "[OK] Inference result loaded."
    )

    explanation = load_json(
        explanation_path
    )

    print(
        "[OK] Explanation result loaded."
    )

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    timeline = generate_timeline(
        inference,
        explanation
    )

    print(
        "[OK] Timeline generated."
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            timeline,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "[OK] Timeline JSON saved."
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    display_timeline(
        timeline
    )

    print()

    print(
        f"Saved to:\n{output_path}"
    )


if __name__ == "__main__":
    main()