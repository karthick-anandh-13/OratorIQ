"""Template-based, retrieval-grounded OratorIQ coaching."""
from pathlib import Path
from typing import Any

try:
    from .retrieve import retrieve
except ImportError:
    from retrieve import retrieve


TOPICS = {
    "fast_pacing": (
        "pacing",
        "fast speaking rate comfortable pace natural pauses",
    ),
    "slow_pacing": (
        "pacing",
        "slow speaking rate comfortable pace clear delivery",
    ),
    "long_pause": (
        "pauses",
        "long pauses natural phrasing pause placement",
    ),
    "pitch_deviation": (
        "pitch",
        "pitch pattern fundamental frequency vocal variation",
    ),
    "low_volume": (
        "volume",
        "low volume loudness audibility projection",
    ),
    "high_volume": (
        "volume",
        "high volume loudness comfortable projection",
    ),
    "prolongation": (
        "prolongation",
        "possible prolonged sounds fluency",
    ),
    "block": (
        "blocking",
        "possible speech block fluency",
    ),
    "blocking": (
        "blocking",
        "possible speech block fluency",
    ),
    "soundrep": (
        "repetitions",
        "possible sound repetition fluency",
    ),
    "wordrep": (
        "repetitions",
        "possible word repetition fluency",
    ),
    "interjection": (
        "fluency",
        "interjections fillers speech fluency",
    ),
}

NAMES = {
    "fast_pacing": "Fast pacing",
    "slow_pacing": "Slow pacing",
    "long_pause": "Long pause",
    "pitch_deviation": "Pitch variation",
    "low_volume": "Low volume",
    "high_volume": "High volume",
    "prolongation": "Possible prolongation",
    "block": "Possible blocking event",
    "blocking": "Possible blocking event",
    "soundrep": "Possible sound repetition",
    "wordrep": "Possible word repetition",
    "interjection": "Possible interjection",
}


def d(value):
    """Return a dictionary or an empty dictionary."""
    return value if isinstance(value, dict) else {}


def key(value):
    """Normalize a finding name."""
    return (
        str(value or "")
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def pick(obj, *keys, default=None):
    """Return the first available, non-None dictionary value."""
    obj = d(obj)

    for name in keys:
        if obj.get(name) is not None:
            return obj[name]

    return default


def _delivery(analysis):
    """Support existing API response formats."""
    return (
        d(analysis.get("delivery"))
        or d(analysis.get("analysis"))
        or analysis
    )


def _evidence_text(finding):
    """Convert model evidence into readable text."""
    evidence = d(finding.get("evidence"))
    finding_key = finding["key"]

    parts = []

    if finding_key in ("fast_pacing", "slow_pacing"):
        value = pick(
            evidence,
            "wpm_delta",
            "temporal_wpm_delta",
            "delta_global_wpm",
        )
        ratio = pick(
            evidence,
            "wpm_ratio",
            "temporal_wpm_ratio",
        )

        try:
            parts.append(
                f"Estimated speaking-rate difference: "
                f"{float(value):+.2f} WPM."
            )
        except (TypeError, ValueError):
            pass

        try:
            parts.append(
                f"Estimated rate ratio: {float(ratio):.3f}x "
                "the reference."
            )
        except (TypeError, ValueError):
            pass

    elif finding_key == "pitch_deviation":
        value = pick(evidence, "f0_delta", "temporal_f0_delta")
        ratio = pick(evidence, "f0_ratio", "temporal_f0_ratio")

        try:
            parts.append(
                f"Estimated fundamental-frequency difference: "
                f"{float(value):+.2f} Hz."
            )
        except (TypeError, ValueError):
            pass

        try:
            parts.append(
                f"Estimated pitch ratio: {float(ratio):.3f}x "
                "the reference."
            )
        except (TypeError, ValueError):
            pass

    elif finding_key == "long_pause":
        value = pick(evidence, "silence_delta")
        ratio = pick(evidence, "silence_ratio")

        try:
            parts.append(
                f"Estimated silence-feature difference: "
                f"{float(value):+.3f}."
            )
        except (TypeError, ValueError):
            pass

        try:
            parts.append(
                f"Estimated silence ratio: {float(ratio):.3f}x "
                "the reference."
            )
        except (TypeError, ValueError):
            pass

    if parts:
        return " ".join(parts)

    start = finding.get("start")
    end = finding.get("end")

    try:
        return (
            f"The model marked a region from "
            f"{float(start):.2f}s to {float(end):.2f}s."
        )
    except (TypeError, ValueError):
        return (
            "Review the marked audio region. "
            "This is an automated observation."
        )


def _bullets(chunks, limit=5):
    """Extract useful, deduplicated coaching recommendations."""
    output = []
    seen = set()

    for chunk in chunks:
        content = chunk.get("content", "")

        lines = [
            line.strip().lstrip("-*• ").strip()
            for line in content.splitlines()
            if line.strip().startswith(("-", "*", "•"))
        ]

        if not lines:
            lines = [content.replace("\n", " ").strip()]

        for line in lines:
            if len(line) < 24:
                continue

            normalized = line.lower()

            if normalized in seen:
                continue

            seen.add(normalized)

            output.append(
                {
                    "text": line[:500],
                    "source": chunk.get("source", "knowledge base"),
                    "section": chunk.get("section", "General guidance"),
                }
            )

            if len(output) >= limit:
                return output

    return output


def _extract_region_findings(analysis):
    """Read the regions produced by OratorIQ's hybrid inference model."""
    findings = []

    regions = analysis.get("regions", [])

    if not isinstance(regions, list):
        return findings

    for region in regions:
        region = d(region)

        flaw = key(
            pick(
                region,
                "flaw_type",
                "final_flaw_type",
                "original_flaw_type",
            )
        )

        if not flaw:
            continue

        confidence = pick(
            region,
            "flaw_type_confidence",
            "temporal_confidence",
            "family_confidence",
        )

        findings.append(
            {
                "key": flaw,
                "name": NAMES.get(
                    flaw,
                    flaw.replace("_", " ").title(),
                ),
                "severity": pick(region, "severity"),
                "confidence": confidence,
                "start": pick(region, "start", "start_time"),
                "end": pick(region, "end", "end_time"),
                "evidence": d(region.get("evidence")),
            }
        )

    return findings


def _extract_legacy_findings(analysis):
    """Read the earlier delivery/fluency API response format."""
    findings = []

    delivery = _delivery(analysis)
    fluency = d(analysis.get("fluency"))

    flaw = key(
        pick(delivery, "flaw_type", "primary_flaw", "label")
    )

    detected = pick(
        delivery,
        "detected",
        "has_flaw",
        default=False,
    )

    if flaw and detected:
        region = d(
            pick(
                delivery,
                "region",
                "primary_region",
                default={},
            )
        )

        findings.append(
            {
                "key": flaw,
                "name": NAMES.get(
                    flaw,
                    flaw.replace("_", " ").title(),
                ),
                "severity": pick(
                    delivery,
                    "severity",
                    "severity_label",
                ),
                "confidence": pick(
                    delivery,
                    "confidence",
                    "overall_confidence",
                ),
                "start": pick(
                    region,
                    "start",
                    "start_time",
                    default=pick(delivery, "start", "start_time"),
                ),
                "end": pick(
                    region,
                    "end",
                    "end_time",
                    default=pick(delivery, "end", "end_time"),
                ),
                "evidence": d(
                    pick(
                        delivery,
                        "evidence",
                        "evidence_values",
                        default={},
                    )
                ),
            }
        )

    prediction = d(fluency.get("prediction")) or fluency

    for raw_name, raw_event in d(prediction.get("events")).items():
        event = d(raw_event)
        event_key = key(raw_name)

        if event.get("detected"):
            findings.append(
                {
                    "key": event_key,
                    "name": NAMES.get(
                        event_key,
                        f"Possible {raw_name} event",
                    ),
                    "confidence": pick(
                        event,
                        "probability",
                        "confidence",
                    ),
                    "evidence": {},
                }
            )

    return findings


def _extract_findings(analysis):
    """
    Prefer the hybrid model's aggregated regions.
    Fall back to the existing API response format.
    """
    findings = _extract_region_findings(analysis)

    if findings:
        return findings

    return _extract_legacy_findings(analysis)


def generate_coaching(
    analysis: dict[str, Any],
    top_k: int = 4,
) -> dict:
    """
    Generate coaching from a saved inference result or API response.

    The ML model supplies the findings.
    TF-IDF retrieval supplies relevant knowledge.
    This function does not diagnose speech disorders.
    """
    analysis = d(analysis)
    findings = _extract_findings(analysis)

    disclaimer = (
        "Automated observations are estimates, not a medical diagnosis. "
        "Fluency-event predictions are possibilities, not confirmed "
        "clinical findings. For persistent concerns, consult a qualified "
        "speech-language pathologist."
    )

    if not findings:
        return {
            "status": "no_findings",
            "summary": "No supported findings were available for coaching.",
            "observations": [],
            "recommendations": [],
            "exercises": [],
            "sources": [],
            "disclaimer": disclaimer,
        }

    observations = []
    all_chunks = []
    seen_chunks = set()

    for finding in findings:
        topic, terms = TOPICS.get(
            finding["key"],
            (
                "presentation_coaching",
                "speech communication practice",
            ),
        )

        evidence = d(finding.get("evidence"))

        query = (
            f"{finding['name']} {terms} "
            f"{finding.get('severity', '')} "
            + " ".join(
                f"{name} {value}"
                for name, value in evidence.items()
            )
        )

        chunks = retrieve(query, top_k=top_k)

        observations.append(
            {
                "type": finding["key"],
                "title": finding["name"],
                "start": finding.get("start"),
                "end": finding.get("end"),
                "severity": finding.get("severity"),
                "confidence": finding.get("confidence"),
                "evidence": _evidence_text(finding),
                "retrieval_query": query,
                "retrieved_sources": [
                    {
                        "title": chunk.get("title"),
                        "file": chunk.get("source"),
                        "section": chunk.get("section"),
                        "score": chunk.get("score"),
                    }
                    for chunk in chunks
                ],
            }
        )

        for chunk in chunks:
            identity = (
                chunk.get("source"),
                chunk.get("section"),
            )

            if identity not in seen_chunks:
                all_chunks.append(chunk)
                seen_chunks.add(identity)

    recommendations = _bullets(all_chunks)

    if not recommendations:
        recommendations = [
            {
                "text": (
                    "Review the marked audio and practise a short passage "
                    "at a comfortable pace, focusing on clear communication."
                ),
                "source": "OratorIQ fallback",
                "section": "General practice",
            }
        ]

    sources = []

    for chunk in all_chunks:
        source_name = chunk.get("source", "knowledge base")
        section_name = chunk.get("section", "General guidance")

        existing = next(
            (
                source
                for source in sources
                if source["file"] == source_name
            ),
            None,
        )

        if existing is None:
            sources.append(
                {
                    "title": chunk.get("title", source_name),
                    "file": source_name,
                    "sections": [section_name],
                }
            )
        elif section_name not in existing["sections"]:
            existing["sections"].append(section_name)

    exercises = [
        {
            "title": f"Practice from {chunk.get('title', 'knowledge base')}",
            "instructions": chunk.get("content", "")[:700],
            "source": chunk.get("source"),
        }
        for chunk in all_chunks
        if (
            "exercise" in chunk.get("section", "").lower()
            or "practice exercise" in chunk.get("content", "").lower()
        )
    ][:2]

    return {
        "status": "ok",
        "summary": (
            "Coaching guidance for: "
            + ", ".join(finding["name"] for finding in findings)
            + "."
        ),
        "observations": observations,
        "recommendations": recommendations,
        "exercises": exercises,
        "sources": sources,
        "retrieval": {
            "method": "TF-IDF",
            "knowledge_chunks_used": len(all_chunks),
            "top_k_per_finding": top_k,
        },
        "disclaimer": disclaimer,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "analysis_json",
        help="Path to a saved OratorIQ analysis JSON",
    )
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    analysis = json.loads(
        Path(args.analysis_json).read_text(encoding="utf-8")
    )

    result = generate_coaching(analysis, top_k=args.top_k)

    print(json.dumps(result, indent=2, ensure_ascii=False))