"""
===============================================================
ORATORIQ — CONTRASTIVE FEATURE BUILDER
===============================================================

Uses the existing temporal feature checkpoint.

For every flawed recording:

    GOOD paired recording
             vs
    FLAWED recording

creates:

    flawed feature
    good baseline feature
    absolute delta
    relative delta

No audio extraction is performed here.

This means this stage should be much faster than the
original 14-minute feature extraction.

===============================================================
"""

from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd


# ===============================================================
# PATHS
# ===============================================================

ROOT = Path(r"D:\OratorIQ")

ARTIFACTS = ROOT / "artifacts"

FEATURE_FILE = (
    ARTIFACTS / "temporal_features.pkl"
)

OUTPUT_FILE = (
    ARTIFACTS / "contrastive_features.pkl"
)


# ===============================================================
# CONFIG
# ===============================================================

# Ignore metadata / labels.
EXCLUDED_COLUMNS = {
    "record_id",
    "pair_id",
    "speaker_id",
    "window_start",
    "window_end",
    "temporal_label",
    "recording_label",
    "severity",
    "flaw_type",
    "transcript",
}


# ===============================================================
# HELPERS
# ===============================================================

def header(text):

    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def progress(current, total, start_time):

    elapsed = time.time() - start_time

    ratio = current / max(total, 1)

    width = 25

    filled = int(
        ratio * width
    )

    bar = (
        "█" * filled
        + "░" * (width - filled)
    )

    if ratio > 0:

        remaining = (
            elapsed / ratio
        ) - elapsed

    else:

        remaining = 0

    print(
        f"\r[{bar}] "
        f"{ratio * 100:6.2f}% "
        f"| {current:,}/{total:,} "
        f"| ETA {int(max(remaining, 0))}s",
        end="",
        flush=True,
    )

    if current == total:
        print()


# ===============================================================
# LOAD
# ===============================================================

header(
    "ORATORIQ — BUILDING CONTRASTIVE FEATURES"
)

if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"Feature checkpoint not found:\n"
        f"{FEATURE_FILE}"
    )


print(
    f"Loading:\n{FEATURE_FILE}"
)

df = joblib.load(
    FEATURE_FILE
)

print(
    f"Total temporal windows: "
    f"{len(df):,}"
)


# ===============================================================
# FIND NUMERIC FEATURES
# ===============================================================

feature_columns = []

for column in df.columns:

    if column in EXCLUDED_COLUMNS:
        continue

    if pd.api.types.is_numeric_dtype(
        df[column]
    ):

        feature_columns.append(column)


print(
    f"Numeric acoustic features: "
    f"{len(feature_columns)}"
)


# ===============================================================
# IDENTIFY GOOD RECORDINGS
# ===============================================================

good_df = df[
    df["recording_label"] == "good"
].copy()

flawed_df = df[
    df["recording_label"] != "good"
].copy()


print(
    f"Good windows:   {len(good_df):,}"
)

print(
    f"Flawed windows: {len(flawed_df):,}"
)


# ===============================================================
# CREATE GOOD BASELINE LOOKUP
# ===============================================================

header(
    "CREATING PAIRED GOOD-SPEECH BASELINES"
)

"""
Each pair_id should contain:

    good
    slight
    medium
    bad
    extreme

The flawed recording may have a different duration.

Therefore we match using RELATIVE POSITION rather than
blindly using the exact timestamp.

Example:

Flawed:
    window center = 60% of recording

Good:
    use window closest to 60% of good recording
"""

good_lookup = {}


for pair_id, group in good_df.groupby(
    "pair_id"
):

    group = group.sort_values(
        "window_start"
    ).copy()

    duration = (
        group["window_end"].max()
    )

    if duration <= 0:
        continue

    group["_relative_position"] = (
        (
            group["window_start"]
            + group["window_end"]
        ) / 2.0
    ) / duration

    good_lookup[pair_id] = group


print(
    f"Paired good recordings available: "
    f"{len(good_lookup):,}"
)


# ===============================================================
# BUILD DELTAS
# ===============================================================

header(
    "BUILDING CONTRASTIVE DELTA FEATURES"
)

results = []

start_time = time.time()

total = len(flawed_df)

matched = 0
unmatched = 0


for counter, (_, row) in enumerate(
    flawed_df.iterrows(),
    start=1
):

    pair_id = row["pair_id"]

    if pair_id not in good_lookup:

        unmatched += 1

        progress(
            counter,
            total,
            start_time
        )

        continue


    good_windows = good_lookup[pair_id]

    # -----------------------------------------------------------
    # Relative position of flawed window
    # -----------------------------------------------------------

    flawed_duration = (
        flawed_df[
            flawed_df["pair_id"] == pair_id
        ]["window_end"].max()
    )

    if flawed_duration <= 0:

        unmatched += 1

        continue


    flawed_center = (
        row["window_start"]
        + row["window_end"]
    ) / 2.0

    relative_position = (
        flawed_center
        / flawed_duration
    )


    # -----------------------------------------------------------
    # Find closest good window
    # -----------------------------------------------------------

    distance = np.abs(
        good_windows[
            "_relative_position"
        ].values
        - relative_position
    )

    nearest_index = np.argmin(
        distance
    )

    good_row = good_windows.iloc[
        nearest_index
    ]


    # -----------------------------------------------------------
    # Build result
    # -----------------------------------------------------------

    output = {

        "record_id":
            row["record_id"],

        "pair_id":
            pair_id,

        "speaker_id":
            row["speaker_id"],

        "window_start":
            row["window_start"],

        "window_end":
            row["window_end"],

        "relative_position":
            relative_position,

        "flaw_type":
            row["flaw_type"],

        "severity":
            row["recording_label"],

        "temporal_label":
            row["temporal_label"],

        "good_window_start":
            good_row["window_start"],

        "good_window_end":
            good_row["window_end"],

    }


    # -----------------------------------------------------------
    # Delta features
    # -----------------------------------------------------------

    for feature in feature_columns:

        flawed_value = row[feature]

        good_value = good_row[feature]

        if (
            pd.isna(flawed_value)
            or pd.isna(good_value)
        ):

            delta = 0.0

            relative_delta = 0.0

        else:

            flawed_value = float(
                flawed_value
            )

            good_value = float(
                good_value
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

    matched += 1

    progress(
        counter,
        total,
        start_time
    )


# ===============================================================
# SAVE
# ===============================================================

contrastive_df = pd.DataFrame(
    results
)


print()

print(
    f"\nMatched windows:   "
    f"{matched:,}"
)

print(
    f"Unmatched windows: "
    f"{unmatched:,}"
)

print(
    f"Contrastive rows:  "
    f"{len(contrastive_df):,}"
)

print(
    f"Contrastive features: "
    f"{len(contrastive_df.columns):,}"
)


joblib.dump(
    contrastive_df,
    OUTPUT_FILE,
    compress=3
)


print(
    f"\nSaved:\n{OUTPUT_FILE}"
)


# ===============================================================
# SUMMARY
# ===============================================================

header(
    "CONTRASTIVE DATASET COMPLETE"
)

print(
    f"""
Original windows:
    {len(df):,}

Flawed windows:
    {len(flawed_df):,}

Successfully paired:
    {matched:,}

Unmatched:
    {unmatched:,}

Output:
    {OUTPUT_FILE}
"""
)

print(
    "✅ Contrastive feature construction complete."
)