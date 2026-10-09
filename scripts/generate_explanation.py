from pathlib import Path
import json
import argparse
import statistics


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\OratorIQ")
RESULTS = ROOT / "artifacts" / "results"

DEFAULT_INPUT = RESULTS / "inference_hybrid_result.json"
DEFAULT_OUTPUT = RESULTS / "oratoriq_explanation.json"


# ============================================================
# HUMAN-READABLE LABELS
# ============================================================

FLAW_LABELS = {
    "fast_pacing": "Fast Pacing",
    "slow_pacing": "Slow Pacing",
    "long_pause": "Long Pause",
    "high_volume": "High Volume",
    "low_volume": "Low Volume",
    "pitch_deviation": "Pitch Deviation",
}

FAMILY_LABELS = {
    "temporal": "Temporal",
    "acoustic": "Acoustic",
}

SEVERITY_LABELS = {
    "slight": "Slight",
    "medium": "Medium",
    "bad": "Bad",
    "extreme": "Extreme",
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percentage(value):
    return round(safe_float(value) * 100.0, 1)


def signed(value, decimals=2):
    return f"{safe_float(value):+.{decimals}f}"


def confidence_label(value):
    value = safe_float(value)

    if value >= 0.90:
        return "Very High"
    if value >= 0.75:
        return "High"
    if value >= 0.60:
        return "Moderate"
    return "Low"


def load_result(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Inference result not found:\n{path}\n\n"
            "Run infer_oratoriq_hybrid.py first."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# EVIDENCE HANDLING
# ============================================================

EVIDENCE_KEYS = (
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
)


def get_region_evidence(region):
    """
    Read evidence directly from a merged region when available.
    This keeps compatibility with future inference outputs that
    already attach evidence to merged regions.
    """
    evidence = region.get("evidence")

    if isinstance(evidence, dict):
        return {
            key: safe_float(evidence.get(key))
            for key in EVIDENCE_KEYS
        }

    return None


def overlap(a_start, a_end, b_start, b_end):
    return max(
        0.0,
        min(a_end, b_end) - max(a_start, b_start)
    )


def aggregate_window_evidence(region, window_predictions):
    """
    Build evidence for a merged region from its overlapping
    window_predictions.

    We use overlap-weighted evidence rather than simply taking
    one arbitrary window. This is important because OratorIQ
    uses 1-second windows with 0.5-second overlap.
    """

    if not window_predictions:
        return None

    region_start = safe_float(region.get("start"))
    region_end = safe_float(region.get("end"))

    candidates = []

    for window in window_predictions:

        start = safe_float(window.get("start"))
        end = safe_float(window.get("end"))

        ov = overlap(
            region_start,
            region_end,
            start,
            end
        )

        if ov <= 0:
            continue

        evidence = window.get("evidence")

        if not isinstance(evidence, dict):
            continue

        temporal_conf = safe_float(
            window.get("temporal_confidence")
        )

        candidates.append(
            (ov, temporal_conf, evidence)
        )

    if not candidates:
        return None

    # Weighted by temporal overlap. Confidence is used only as
    # a secondary stabilizer so a weak edge window does not
    # dominate the explanation.
    weights = []
    values = {key: [] for key in EVIDENCE_KEYS}

    for ov, conf, evidence in candidates:

        weight = ov * max(conf, 0.10)

        weights.append(weight)

        for key in EVIDENCE_KEYS:
            values[key].append(
                (
                    safe_float(evidence.get(key)),
                    weight
                )
            )

    total_weight = sum(weights)

    if total_weight <= 0:
        return None

    aggregated = {}

    for key in EVIDENCE_KEYS:

        weighted_sum = sum(
            value * weight
            for value, weight in values[key]
        )

        aggregated[key] = weighted_sum / total_weight

    return aggregated


def get_evidence(region, window_predictions):
    """
    Priority:
      1. Evidence already attached to merged region
      2. Aggregate overlapping window evidence
      3. Zero fallback

    The fallback is explicitly marked as unavailable so the
    explanation engine never silently claims zero as real evidence.
    """

    direct = get_region_evidence(region)

    if direct is not None:
        return direct, "region"

    aggregated = aggregate_window_evidence(
        region,
        window_predictions
    )

    if aggregated is not None:
        return aggregated, "window_aggregate"

    return (
        {key: 0.0 for key in EVIDENCE_KEYS},
        "unavailable"
    )


# ============================================================
# FLAW-SPECIFIC EXPLANATIONS
# ============================================================

def build_explanation_text(
    flaw_type,
    evidence
):
    rms_delta = safe_float(evidence["rms_delta"])
    silence_delta = safe_float(evidence["silence_delta"])
    silence_ratio = safe_float(evidence["silence_ratio"])
    f0_delta = safe_float(evidence["f0_delta"])
    f0_ratio = safe_float(evidence["f0_ratio"])
    wpm_delta = safe_float(evidence["wpm_delta"])
    wpm_ratio = safe_float(evidence["wpm_ratio"])
    duration_delta = safe_float(
        evidence["window_duration_delta"]
    )
    duration_ratio = safe_float(
        evidence["window_duration_ratio"]
    )
    alignment_error = safe_float(
        evidence["alignment_error"]
    )

    evidence_points = []

    if flaw_type == "fast_pacing":

        summary = (
            "The speaker's delivery is faster than the "
            "reference in this region."
        )

        evidence_points.append(
            f"Estimated speaking-rate difference: "
            f"{signed(wpm_delta)} WPM."
        )

        evidence_points.append(
            f"Relative speaking-rate pattern: "
            f"{wpm_ratio:.3f}× the reference."
        )

        if duration_delta != 0:
            evidence_points.append(
                f"Window timing difference: "
                f"{signed(duration_delta)} seconds."
            )

        recommendation = (
            "Slow down slightly and add natural pauses "
            "between phrases."
        )

    elif flaw_type == "slow_pacing":

        summary = (
            "The speaker's delivery is slower than the "
            "reference in this region."
        )

        evidence_points.append(
            f"Estimated speaking-rate difference: "
            f"{signed(wpm_delta)} WPM."
        )

        evidence_points.append(
            f"Relative speaking-rate pattern: "
            f"{wpm_ratio:.3f}× the reference."
        )

        if duration_delta != 0:
            evidence_points.append(
                f"Window timing difference: "
                f"{signed(duration_delta)} seconds."
            )

        recommendation = (
            "Increase your speaking pace slightly while "
            "keeping the speech clear and natural."
        )

    elif flaw_type == "long_pause":

        summary = (
            "An unusually long silent interval was detected "
            "relative to the reference."
        )

        evidence_points.append(
            f"Silence difference: "
            f"{signed(silence_delta, 3)}."
        )

        evidence_points.append(
            f"Silence ratio: "
            f"{silence_ratio:.3f}× the reference."
        )

        recommendation = (
            "Reduce unnecessary silence and maintain a "
            "smoother speaking flow."
        )

    elif flaw_type == "high_volume":

        summary = (
            "The speaker's vocal intensity is higher than "
            "the reference in this region."
        )

        evidence_points.append(
            f"RMS energy difference: "
            f"{signed(rms_delta, 4)}."
        )

        recommendation = (
            "Reduce vocal intensity slightly and maintain "
            "a comfortable speaking volume."
        )

    elif flaw_type == "low_volume":

        summary = (
            "The speaker's vocal intensity is lower than "
            "the reference in this region."
        )

        evidence_points.append(
            f"RMS energy difference: "
            f"{signed(rms_delta, 4)}."
        )

        recommendation = (
            "Increase vocal projection slightly so your "
            "speech remains clear and audible."
        )

    elif flaw_type == "pitch_deviation":

        summary = (
            "The speaker's pitch pattern differs noticeably "
            "from the reference in this region."
        )

        evidence_points.append(
            f"F0 difference: "
            f"{signed(f0_delta, 2)} Hz."
        )

        evidence_points.append(
            f"Relative F0 pattern: "
            f"{f0_ratio:.3f}× the reference."
        )

        recommendation = (
            "Try to maintain a more controlled and natural "
            "pitch pattern."
        )

    else:

        summary = (
            "An abnormal speech pattern was detected, but "
            "no specific explanation rule is available yet."
        )

        evidence_points.append(
            "The hybrid model identified a speech anomaly."
        )

        recommendation = (
            "Review the highlighted region manually."
        )

    # Add alignment context only when it is meaningful.
    if alignment_error > 0:
        evidence_points.append(
            f"Reference alignment error: "
            f"{alignment_error:.3f}s."
        )

    return (
        summary,
        evidence_points,
        recommendation
    )


# ============================================================
# REGION EXPLANATION
# ============================================================

def explain_region(
    region,
    region_number,
    window_predictions
):
    """
    Convert one detected merged region into a human-readable
    explanation while preserving the existing output schema.
    """

    # Support both the current merged-region keys and the
    # window-level keys used by inference.
    flaw_type = str(
        region.get(
            "flaw_type",
            region.get(
                "final_flaw_type",
                "unknown"
            )
        )
    )

    family = str(
        region.get(
            "family",
            "unknown"
        )
    )

    severity = str(
        region.get(
            "severity",
            "unknown"
        )
    )

    start = safe_float(region.get("start"))
    end = safe_float(region.get("end"))

    duration = safe_float(
        region.get(
            "duration",
            end - start
        )
    )

    temporal_confidence = safe_float(
        region.get("temporal_confidence")
    )

    family_confidence = safe_float(
        region.get("family_confidence")
    )

    flaw_confidence = safe_float(
        region.get("flaw_type_confidence")
    )

    severity_confidence = safe_float(
        region.get("severity_confidence")
    )

    # If merged region confidence fields are unavailable,
    # derive them from overlapping windows.
    if window_predictions:

        overlapping = []

        for window in window_predictions:

            ov = overlap(
                start,
                end,
                safe_float(window.get("start")),
                safe_float(window.get("end"))
            )

            if ov > 0:
                overlapping.append(
                    (ov, window)
                )

        if overlapping:

            def weighted_conf(key):
                total = 0.0
                weight_total = 0.0

                for ov, window in overlapping:
                    value = safe_float(
                        window.get(key)
                    )
                    weight = ov * max(
                        safe_float(
                            window.get(
                                "temporal_confidence"
                            )
                        ),
                        0.10
                    )

                    total += value * weight
                    weight_total += weight

                return (
                    total / weight_total
                    if weight_total
                    else 0.0
                )

            if temporal_confidence == 0:
                temporal_confidence = weighted_conf(
                    "temporal_confidence"
                )

            if family_confidence == 0:
                family_confidence = weighted_conf(
                    "family_confidence"
                )

            if flaw_confidence == 0:
                flaw_confidence = weighted_conf(
                    "flaw_type_confidence"
                )

            if severity_confidence == 0:
                severity_confidence = weighted_conf(
                    "severity_confidence"
                )

    evidence, evidence_source = get_evidence(
        region,
        window_predictions
    )

    overall_confidence = statistics.mean([
        temporal_confidence,
        family_confidence,
        flaw_confidence,
        severity_confidence
    ])

    summary, evidence_points, recommendation = (
        build_explanation_text(
            flaw_type,
            evidence
        )
    )

    return {
        "region_id": region_number,

        "time": {
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(duration, 3),
        },

        "detection": {
            "flaw_type": flaw_type,

            "flaw_label": FLAW_LABELS.get(
                flaw_type,
                flaw_type.replace("_", " ").title()
            ),

            "family": family,

            "family_label": FAMILY_LABELS.get(
                family,
                family.title()
            ),

            "severity": severity,

            "severity_label": SEVERITY_LABELS.get(
                severity,
                severity.title()
            ),
        },

        "confidence": {
            "overall": round(
                overall_confidence,
                4
            ),

            "overall_percent": percentage(
                overall_confidence
            ),

            "level": confidence_label(
                overall_confidence
            ),

            "temporal": round(
                temporal_confidence,
                4
            ),

            "temporal_percent": percentage(
                temporal_confidence
            ),

            "family": round(
                family_confidence,
                4
            ),

            "family_percent": percentage(
                family_confidence
            ),

            "flaw_type": round(
                flaw_confidence,
                4
            ),

            "flaw_type_percent": percentage(
                flaw_confidence
            ),

            "severity": round(
                severity_confidence,
                4
            ),

            "severity_percent": percentage(
                severity_confidence
            ),
        },

        "explanation": {
            "summary": summary,

            # IMPORTANT:
            # This remains a list exactly like the old script,
            # so existing dashboard consumers do not break.
            "evidence": evidence_points,

            "recommendation": recommendation,

            # New quantitative evidence for the dashboard.
            "evidence_values": {
                key: round(
                    safe_float(evidence[key]),
                    6
                )
                for key in EVIDENCE_KEYS
            },

            "evidence_source": evidence_source,
        },
    }


# ============================================================
# OVERALL ANALYSIS
# ============================================================

def generate_explanation(result):
    """
    Generate complete explainability output.

    Compatibility is intentionally preserved:
      - same top-level fields
      - same region structure
      - same explanation.evidence list
      - same CLI arguments
      - same output path by default

    New:
      - quantitative evidence_values
      - actual evidence aggregation from window_predictions
      - evidence_source
    """

    regions = result.get("regions", [])

    window_predictions = result.get(
        "window_predictions",
        []
    )

    # --------------------------------------------------------
    # Fallback only if the inference JSON has no merged regions.
    #
    # Instead of creating one explanation per overlapping
    # window, contiguous detected windows are merged here.
    # --------------------------------------------------------

    if not regions and window_predictions:

        detected = [
            w for w in window_predictions
            if safe_float(
                w.get("temporal_confidence")
            ) >= 0.50
        ]

        detected.sort(
            key=lambda x: safe_float(
                x.get("start")
            )
        )

        merged = []

        for window in detected:

            start = safe_float(
                window.get("start")
            )

            end = safe_float(
                window.get("end")
            )

            if not merged:

                merged.append({
                    "start": start,
                    "end": end,
                    "flaw_type": window.get(
                        "final_flaw_type",
                        window.get(
                            "original_flaw_type",
                            "unknown"
                        )
                    ),
                    "family": window.get(
                        "family",
                        "unknown"
                    ),
                    "severity": window.get(
                        "severity",
                        "unknown"
                    ),
                    "temporal_confidence": safe_float(
                        window.get(
                            "temporal_confidence"
                        )
                    ),
                    "family_confidence": safe_float(
                        window.get(
                            "family_confidence"
                        )
                    ),
                    "flaw_type_confidence": safe_float(
                        window.get(
                            "flaw_type_confidence"
                        )
                    ),
                    "severity_confidence": safe_float(
                        window.get(
                            "severity_confidence"
                        )
                    ),
                })

                continue

            previous = merged[-1]

            # Same merging spirit as inference: overlapping or
            # closely adjacent detected windows become one region.
            if start <= previous["end"] + 0.75:

                previous["end"] = max(
                    previous["end"],
                    end
                )

                # Keep the strongest classification/confidence
                # within the merged region.
                if (
                    safe_float(
                        window.get(
                            "temporal_confidence"
                        )
                    )
                    >
                    safe_float(
                        previous.get(
                            "temporal_confidence"
                        )
                    )
                ):
                    previous[
                        "flaw_type"
                    ] = window.get(
                        "final_flaw_type",
                        window.get(
                            "original_flaw_type",
                            previous["flaw_type"]
                        )
                    )

                    previous[
                        "family"
                    ] = window.get(
                        "family",
                        previous["family"]
                    )

                    previous[
                        "severity"
                    ] = window.get(
                        "severity",
                        previous["severity"]
                    )

                    previous[
                        "temporal_confidence"
                    ] = safe_float(
                        window.get(
                            "temporal_confidence"
                        )
                    )

            else:
                merged.append({
                    "start": start,
                    "end": end,
                    "flaw_type": window.get(
                        "final_flaw_type",
                        window.get(
                            "original_flaw_type",
                            "unknown"
                        )
                    ),
                    "family": window.get(
                        "family",
                        "unknown"
                    ),
                    "severity": window.get(
                        "severity",
                        "unknown"
                    ),
                    "temporal_confidence": safe_float(
                        window.get(
                            "temporal_confidence"
                        )
                    ),
                    "family_confidence": safe_float(
                        window.get(
                            "family_confidence"
                        )
                    ),
                    "flaw_type_confidence": safe_float(
                        window.get(
                            "flaw_type_confidence"
                        )
                    ),
                    "severity_confidence": safe_float(
                        window.get(
                            "severity_confidence"
                        )
                    ),
                })

        regions = merged

    explanations = []

    for index, region in enumerate(
        regions,
        start=1
    ):

        explanations.append(
            explain_region(
                region,
                index,
                window_predictions
            )
        )

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    if explanations:

        confidence_values = [
            item["confidence"]["overall"]
            for item in explanations
        ]

        overall_confidence = statistics.mean(
            confidence_values
        )

    else:
        overall_confidence = 0.0

    flaw_counts = {}

    for item in explanations:

        flaw = item[
            "detection"
        ][
            "flaw_type"
        ]

        flaw_counts[flaw] = (
            flaw_counts.get(
                flaw,
                0
            ) + 1
        )

    return {
        "system": "OratorIQ",

        # Keep the existing version name so downstream code
        # does not need to change.
        "version": "explainability_v1",

        "source_result": str(
            DEFAULT_INPUT
        ),

        "audio": result.get(
            "audio",
            ""
        ),

        "duration": safe_float(
            result.get(
                "duration"
            )
        ),

        "summary": {
            "detected_regions": len(
                explanations
            ),

            "overall_confidence": round(
                overall_confidence,
                4
            ),

            "overall_confidence_percent":
                percentage(
                    overall_confidence
                ),

            "flaw_counts": flaw_counts,
        },

        "regions": explanations,
    }


# ============================================================
# SAVE
# ============================================================

def save_explanation(
    explanation,
    output_path
):

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
            explanation,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# DISPLAY
# ============================================================

def display_explanation(
    explanation
):

    print()
    print("=" * 70)
    print("ORATORIQ EXPLAINABILITY REPORT")
    print("=" * 70)

    print()

    summary = explanation[
        "summary"
    ]

    print(
        f"Detected regions: "
        f"{summary['detected_regions']}"
    )

    print(
        f"Overall confidence: "
        f"{summary['overall_confidence_percent']:.1f}%"
    )

    print()

    if not explanation["regions"]:

        print(
            "[OK] No significant speech flaws detected."
        )

        print()

        return

    for region in explanation[
        "regions"
    ]:

        detection = region[
            "detection"
        ]

        confidence = region[
            "confidence"
        ]

        explanation_data = region[
            "explanation"
        ]

        time = region[
            "time"
        ]

        print("-" * 70)

        print(
            f"[Region {region['region_id']}] "
            f"{time['start']:.2f}s -> "
            f"{time['end']:.2f}s"
        )

        print()

        print(
            f"Issue: "
            f"{detection['flaw_label']}"
        )

        print(
            f"Family: "
            f"{detection['family_label']}"
        )

        print(
            f"Severity: "
            f"{detection['severity_label']}"
        )

        print(
            f"Confidence: "
            f"{confidence['overall_percent']:.1f}% "
            f"({confidence['level']})"
        )

        print()

        print("WHY:")

        print(
            f"  {explanation_data['summary']}"
        )

        print()

        print("EVIDENCE:")

        for evidence in explanation_data[
            "evidence"
        ]:

            print(
                f"  - {evidence}"
            )

        print()

        print("RECOMMENDATION:")

        print(
            f"  {explanation_data['recommendation']}"
        )

        print()

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate human-readable "
            "explanations from OratorIQ "
            "hybrid inference results."
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT
        ),
        help=(
            "Path to "
            "inference_hybrid_result.json"
        )
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT
        ),
        help=(
            "Path to explanation JSON"
        )
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    print()
    print("=" * 70)
    print("ORATORIQ EXPLAINABILITY ENGINE")
    print("=" * 70)

    print()

    print(
        f"Input:  {input_path}"
    )

    print(
        f"Output: {output_path}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    result = load_result(
        input_path
    )

    print(
        "[OK] Inference result loaded."
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    explanation = generate_explanation(
        result
    )

    print(
        "[OK] Explanations generated."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_explanation(
        explanation,
        output_path
    )

    print(
        "[OK] Explanation JSON saved."
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_explanation(
        explanation
    )

    print()

    print(
        f"Saved to:\n{output_path}"
    )


if __name__ == "__main__":
    main()
