"""
OratorIQ - Visual Timeline Generator

Reads:
    artifacts/results/oratoriq_timeline.json

Creates:
    artifacts/results/oratoriq_timeline.png

This script does NOT modify any existing pipeline/model files.
"""

from pathlib import Path
import json
import sys

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:
    print("[ERROR] matplotlib is not installed.")
    print()
    print("Install it with:")
    print("    pip install matplotlib")
    sys.exit(1)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\OratorIQ")

TIMELINE_JSON = (
    ROOT
    / "artifacts"
    / "results"
    / "oratoriq_timeline.json"
)

OUTPUT_PNG = (
    ROOT
    / "artifacts"
    / "results"
    / "oratoriq_timeline.png"
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def format_seconds(value):
    return f"{value:.2f}s"


def load_json(path):
    if not path.exists():
        print(f"[ERROR] File not found:")
        print(f"        {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# LOAD TIMELINE
# ============================================================

print("=" * 70)
print("ORATORIQ VISUAL TIMELINE GENERATOR")
print("=" * 70)

print(f"Input : {TIMELINE_JSON}")
print(f"Output: {OUTPUT_PNG}")
print()

timeline = load_json(TIMELINE_JSON)

print("[OK] Timeline JSON loaded.")


# ============================================================
# EXTRACT DATA
# ============================================================

duration = safe_float(timeline.get("duration"))

windows = timeline.get("windows", [])
regions = timeline.get("regions", [])

overall_confidence = safe_float(
    timeline.get("overall_confidence")
)

overall_confidence_percent = safe_float(
    timeline.get("overall_confidence_percent"),
    overall_confidence * 100
)

print(f"[INFO] Audio duration     : {duration:.2f}s")
print(f"[INFO] Total windows      : {len(windows)}")
print(f"[INFO] Detected regions   : {len(regions)}")
print(f"[INFO] Overall confidence: {overall_confidence_percent:.1f}%")
print()


# ============================================================
# WINDOW DATA
# ============================================================

window_starts = []
window_ends = []
window_probs = []

for window in windows:
    start = safe_float(window.get("start"))
    end = safe_float(window.get("end"))

    probability = safe_float(
        window.get("temporal_probability")
    )

    window_starts.append(start)
    window_ends.append(end)
    window_probs.append(probability)


# Use window centers for probability graph

window_centers = [
    (start + end) / 2
    for start, end in zip(window_starts, window_ends)
]


# ============================================================
# CREATE FIGURE
# ============================================================

fig = plt.figure(
    figsize=(15, 10)
)

fig.suptitle(
    "OratorIQ — Contrastive Speech Analysis",
    fontsize=20,
    fontweight="bold",
    y=0.97
)


# ============================================================
# PANEL 1 — SPEECH TIMELINE
# ============================================================

ax1 = plt.subplot2grid(
    (4, 1),
    (0, 0),
    rowspan=1
)

ax1.set_title(
    "Temporal Flaw Timeline",
    fontsize=14,
    fontweight="bold",
    loc="left"
)

# Base timeline
ax1.add_patch(
    Rectangle(
        (0, 0),
        duration,
        1,
        alpha=0.15
    )
)

# Normal label
ax1.text(
    duration * 0.01,
    0.5,
    "Speech",
    va="center",
    fontsize=10
)


# Detected regions

for region in regions:

    start = safe_float(region.get("start"))
    end = safe_float(region.get("end"))

    flaw_label = region.get(
        "flaw_label",
        region.get("flaw_type", "Detected")
    )

    severity = region.get(
        "severity_label",
        region.get("severity", "")
    )

    confidence = safe_float(
        region.get("confidence", {}).get("overall_percent")
    )

    # Highlight detected region
    ax1.add_patch(
        Rectangle(
            (start, 0),
            end - start,
            1,
            alpha=0.65
        )
    )

    # Label above region
    ax1.text(
        (start + end) / 2,
        1.08,
        flaw_label,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

    ax1.text(
        (start + end) / 2,
        0.50,
        f"{severity} | {confidence:.1f}%",
        ha="center",
        va="center",
        fontsize=9
    )

    # Start/end markers
    ax1.axvline(
        start,
        linestyle="--",
        linewidth=1
    )

    ax1.axvline(
        end,
        linestyle="--",
        linewidth=1
    )


ax1.set_xlim(
    0,
    max(duration, 0.1)
)

ax1.set_ylim(
    -0.05,
    1.35
)

ax1.set_yticks([])

ax1.set_xlabel(
    "Time (seconds)"
)

ax1.grid(
    axis="x",
    alpha=0.25
)


# ============================================================
# PANEL 2 — DETECTION PROBABILITY
# ============================================================

ax2 = plt.subplot2grid(
    (4, 1),
    (1, 0),
    rowspan=1
)

ax2.set_title(
    "Temporal Flaw Detection Probability",
    fontsize=14,
    fontweight="bold",
    loc="left"
)

if window_centers:

    ax2.plot(
        window_centers,
        window_probs,
        marker="o",
        linewidth=2,
        markersize=5
    )

    ax2.fill_between(
        window_centers,
        window_probs,
        alpha=0.12
    )


# Detection threshold
threshold = 0.50

ax2.axhline(
    threshold,
    linestyle="--",
    linewidth=1.5,
    label="Detection threshold (0.50)"
)


# Highlight detected regions

for region in regions:

    start = safe_float(region.get("start"))
    end = safe_float(region.get("end"))

    ax2.axvspan(
        start,
        end,
        alpha=0.15
    )


ax2.set_xlim(
    0,
    max(duration, 0.1)
)

ax2.set_ylim(
    0,
    1.05
)

ax2.set_ylabel(
    "Probability"
)

ax2.set_xlabel(
    "Time (seconds)"
)

ax2.grid(
    alpha=0.25
)

ax2.legend(
    loc="upper right"
)


# ============================================================
# PANEL 3 — DETECTION SUMMARY
# ============================================================

ax3 = plt.subplot2grid(
    (4, 1),
    (2, 0),
    rowspan=1
)

ax3.axis("off")

ax3.set_title(
    "Detection Summary",
    fontsize=14,
    fontweight="bold",
    loc="left"
)


summary_lines = []

if regions:

    for index, region in enumerate(regions, start=1):

        flaw_label = region.get(
            "flaw_label",
            region.get("flaw_type", "Unknown")
        )

        family_label = region.get(
            "family_label",
            region.get("family", "Unknown")
        )

        severity_label = region.get(
            "severity_label",
            region.get("severity", "Unknown")
        )

        start = safe_float(
            region.get("start")
        )

        end = safe_float(
            region.get("end")
        )

        confidence = safe_float(
            region.get(
                "confidence",
                {}
            ).get(
                "overall_percent"
            )
        )

        summary_lines.append(
            f"Region {index}: "
            f"{flaw_label} | "
            f"{family_label} | "
            f"{severity_label} | "
            f"{start:.2f}s → {end:.2f}s | "
            f"{confidence:.1f}% confidence"
        )

else:

    summary_lines.append(
        "No significant temporal flaw detected."
    )


summary_text = "\n".join(summary_lines)

ax3.text(
    0.01,
    0.70,
    summary_text,
    transform=ax3.transAxes,
    fontsize=12,
    va="top"
)


# ============================================================
# PANEL 4 — EVIDENCE
# ============================================================

ax4 = plt.subplot2grid(
    (4, 1),
    (3, 0),
    rowspan=1
)

ax4.axis("off")

ax4.set_title(
    "Acoustic / Temporal Evidence",
    fontsize=14,
    fontweight="bold",
    loc="left"
)


# Collect evidence from first detected region

evidence = {}

if regions:

    evidence = (
        regions[0]
        .get("explanation", {})
        .get("evidence_values", {})
    )


def evidence_value(key, default=0.0):
    return safe_float(
        evidence.get(key),
        default
    )


evidence_lines = [

    f"Speaking-rate difference : "
    f"{evidence_value('wpm_delta'):+.2f} WPM",

    f"Speaking-rate ratio      : "
    f"{evidence_value('wpm_ratio'):.3f}× reference",

    f"F0 difference            : "
    f"{evidence_value('f0_delta'):+.2f} Hz",

    f"F0 ratio                 : "
    f"{evidence_value('f0_ratio'):.3f}× reference",

    f"RMS difference           : "
    f"{evidence_value('rms_delta'):+.4f}",

    f"Silence difference       : "
    f"{evidence_value('silence_delta'):+.4f}",

    f"Window duration delta    : "
    f"{evidence_value('window_duration_delta'):+.3f}s",

    f"Alignment error          : "
    f"{evidence_value('alignment_error'):.3f}s",
]


# Split evidence into two columns

left_lines = evidence_lines[:4]
right_lines = evidence_lines[4:]


ax4.text(
    0.02,
    0.78,
    "\n".join(left_lines),
    transform=ax4.transAxes,
    fontsize=11,
    va="top"
)

ax4.text(
    0.52,
    0.78,
    "\n".join(right_lines),
    transform=ax4.transAxes,
    fontsize=11,
    va="top"
)


# ============================================================
# FOOTER
# ============================================================

fig.text(
    0.5,
    0.015,
    (
        f"OratorIQ | Duration: {duration:.2f}s | "
        f"Overall confidence: {overall_confidence_percent:.1f}%"
    ),
    ha="center",
    fontsize=9
)


# ============================================================
# SAVE
# ============================================================

plt.tight_layout(
    rect=[0, 0.03, 1, 0.95]
)

OUTPUT_PNG.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    OUTPUT_PNG,
    dpi=180,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# DONE
# ============================================================

print("[OK] Visual timeline generated.")
print()
print(f"Saved to:")
print(f"    {OUTPUT_PNG}")
print()

if regions:

    print("DETECTED REGIONS")

    for index, region in enumerate(regions, start=1):

        start = safe_float(region.get("start"))
        end = safe_float(region.get("end"))

        flaw = region.get(
            "flaw_label",
            region.get("flaw_type", "Unknown")
        )

        severity = region.get(
            "severity_label",
            region.get("severity", "Unknown")
        )

        print(
            f"[{index}] "
            f"{start:.2f}s -> {end:.2f}s | "
            f"{flaw} | "
            f"{severity}"
        )

else:

    print("No detected regions.")

print()
print("=" * 70)
print("DONE")
print("=" * 70)