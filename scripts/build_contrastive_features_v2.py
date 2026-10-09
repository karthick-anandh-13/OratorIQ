"""
======================================================================
ORATORIQ — CONTRASTIVE FEATURE BUILDER V2
======================================================================

Purpose:
    Build an improved contrastive feature dataset for OratorIQ.

This version:

    1. Loads the existing temporal feature checkpoint.
    2. Keeps the original 69 acoustic features.
    3. Keeps the original:
           delta_<feature>
           relative_delta_<feature>
       features.
    4. Adds explicit temporal-aware features.
    5. Uses relative-position matching between GOOD and FLAWED audio.
    6. Adds duration / pacing / silence / F0 / RMS comparison features.
    7. Adds neighboring-window context.
    8. Precomputes recording durations for speed.
    9. Does NOT modify the original contrastive_features.pkl.
   10. Saves to:
           artifacts\contrastive_features_v2.pkl

IMPORTANT:
    This script does NOT modify:
        - audio files
        - metadata.jsonl
        - temporal_features.pkl
        - existing models
        - existing contrastive_features.pkl

======================================================================
"""

from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd


# ======================================================================
# PATHS
# ======================================================================

ROOT = Path(r"D:\OratorIQ")

ARTIFACTS = ROOT / "artifacts"

FEATURE_FILE = (
    ARTIFACTS / "temporal_features.pkl"
)

OUTPUT_FILE = (
    ARTIFACTS / "contrastive_features_v2.pkl"
)


# ======================================================================
# CONFIGURATION
# ======================================================================

# Metadata / labels that must NOT become model features.
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

# Small epsilon to prevent division by zero.
EPS = 1e-6


# ======================================================================
# PRINT HELPERS
# ======================================================================

def header(text):
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def progress(current, total, start_time):

    elapsed = time.time() - start_time

    ratio = current / max(total, 1)

    width = 30

    filled = int(
        ratio * width
    )

    bar = (
        "=" * filled
        + "-" * (width - filled)
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


# ======================================================================
# SAFE NUMERIC VALUE
# ======================================================================

def safe_float(value):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

        return 0.0

    except Exception:

        return 0.0


# ======================================================================
# RELATIVE DIFFERENCE
# ======================================================================

def relative_difference(
    flawed_value,
    good_value
):

    flawed_value = safe_float(
        flawed_value
    )

    good_value = safe_float(
        good_value
    )

    return (
        flawed_value - good_value
    ) / (
        abs(good_value) + EPS
    )


# ======================================================================
# RATIO
# ======================================================================

def safe_ratio(
    flawed_value,
    good_value
):

    flawed_value = safe_float(
        flawed_value
    )

    good_value = safe_float(
        good_value
    )

    return (
        flawed_value
        / (
            abs(good_value) + EPS
        )
    )


# ======================================================================
# LOAD FEATURES
# ======================================================================

header(
    "ORATORIQ — BUILDING CONTRASTIVE FEATURES V2"
)

if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"Feature checkpoint not found:\n"
        f"{FEATURE_FILE}"
    )


print(
    f"Loading feature checkpoint:\n"
    f"{FEATURE_FILE}"
)

df = joblib.load(
    FEATURE_FILE
)

print(
    f"Total temporal windows: "
    f"{len(df):,}"
)


# ======================================================================
# BASIC VALIDATION
# ======================================================================

required_columns = {
    "record_id",
    "pair_id",
    "speaker_id",
    "window_start",
    "window_end",
    "recording_label",
    "temporal_label",
    "severity",
    "flaw_type",
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(
            sorted(missing_columns)
        )
    )


# ======================================================================
# SORT DATA
# ======================================================================

df = df.sort_values(
    [
        "pair_id",
        "record_id",
        "window_start",
    ]
).reset_index(
    drop=True
)


# ======================================================================
# FIND NUMERIC ACOUSTIC FEATURES
# ======================================================================

feature_columns = []

for column in df.columns:

    if column in EXCLUDED_COLUMNS:
        continue

    if pd.api.types.is_numeric_dtype(
        df[column]
    ):

        feature_columns.append(
            column
        )


print(
    f"Numeric acoustic features: "
    f"{len(feature_columns)}"
)

print()

print(
    "First acoustic features:"
)

for feature in feature_columns[:15]:

    print(
        f"    {feature}"
    )

if len(feature_columns) > 15:

    print(
        f"    ... "
        f"{len(feature_columns) - 15} more"
    )


# ======================================================================
# SPLIT GOOD / FLAWED
# ======================================================================

good_df = df[
    df["recording_label"] == "good"
].copy()

flawed_df = df[
    df["recording_label"] != "good"
].copy()


print()

print(
    f"Good windows:   "
    f"{len(good_df):,}"
)

print(
    f"Flawed windows: "
    f"{len(flawed_df):,}"
)


# ======================================================================
# PRECOMPUTE RECORDING DURATIONS
# ======================================================================

header(
    "PRECOMPUTING RECORDING DURATIONS"
)

# The old builder repeatedly scanned the dataframe for
# every flawed window. This version calculates durations once.

recording_durations = (
    df.groupby(
        "record_id"
    )[
        "window_end"
    ]
    .max()
    .to_dict()
)

print(
    f"Recording durations calculated: "
    f"{len(recording_durations):,}"
)


# ======================================================================
# BUILD GOOD WINDOW LOOKUP
# ======================================================================

header(
    "CREATING PAIRED GOOD-SPEECH BASELINES"
)

"""
For every pair_id:

    GOOD recording
        |
        +-- window 1
        +-- window 2
        +-- window 3
        ...

A flawed window is matched using relative position.

Example:

    flawed window center = 60% of flawed recording

    GOOD baseline window
        = closest window around 60% of GOOD recording
"""


good_lookup = {}

for pair_id, group in good_df.groupby(
    "pair_id"
):

    group = group.sort_values(
        "window_start"
    ).copy()

    good_record_ids = (
        group["record_id"]
        .unique()
    )

    if len(good_record_ids) == 0:
        continue

    good_record_id = (
        good_record_ids[0]
    )

    good_duration = (
        recording_durations.get(
            good_record_id,
            0.0
        )
    )

    if good_duration <= 0:
        continue

    group[
        "_relative_position"
    ] = (
        (
            group["window_start"]
            + group["window_end"]
        )
        / 2.0
    ) / good_duration

    good_lookup[
        pair_id
    ] = group


print(
    f"Paired good recordings available: "
    f"{len(good_lookup):,}"
)


# ======================================================================
# CREATE INDEXED LOOKUP FOR FASTER NEIGHBOR SEARCH
# ======================================================================

header(
    "PREPARING TEMPORAL CONTEXT LOOKUPS"
)

# Store windows grouped by recording.
#
# This allows us to find previous / current / next
# windows without repeatedly scanning the entire dataframe.

recording_lookup = {}

for record_id, group in df.groupby(
    "record_id"
):

    group = group.sort_values(
        "window_start"
    ).copy()

    group = group.reset_index(
        drop=True
    )

    recording_lookup[
        record_id
    ] = group


print(
    f"Recording groups prepared: "
    f"{len(recording_lookup):,}"
)


# ======================================================================
# HELPER — FIND NEAREST GOOD WINDOW
# ======================================================================

def find_nearest_good_window(
    good_windows,
    relative_position
):

    positions = (
        good_windows[
            "_relative_position"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if len(positions) == 0:
        return None

    distance = np.abs(
        positions
        - relative_position
    )

    nearest_index = int(
        np.argmin(distance)
    )

    return good_windows.iloc[
        nearest_index
    ]


# ======================================================================
# HELPER — GET NEIGHBOR WINDOW
# ======================================================================

def get_neighbor(
    group,
    current_start,
    direction
):

    if group is None or len(group) == 0:
        return None

    starts = (
        group["window_start"]
        .to_numpy(
            dtype=np.float64
        )
    )

    if direction == "previous":

        candidates = np.where(
            starts < current_start
        )[0]

        if len(candidates) == 0:
            return None

        return group.iloc[
            candidates[-1]
        ]

    if direction == "next":

        candidates = np.where(
            starts > current_start
        )[0]

        if len(candidates) == 0:
            return None

        return group.iloc[
            candidates[0]
        ]

    return None


# ======================================================================
# BUILD CONTRASTIVE DATASET
# ======================================================================

header(
    "BUILDING V2 CONTRASTIVE FEATURES"
)

results = []

start_time = time.time()

total = len(flawed_df)

matched = 0

unmatched = 0


# ======================================================================
# MAIN LOOP
# ======================================================================

for counter, (
    _,
    row
) in enumerate(
    flawed_df.iterrows(),
    start=1
):

    pair_id = row[
        "pair_id"
    ]

    # --------------------------------------------------------------
    # Check paired GOOD recording
    # --------------------------------------------------------------

    if pair_id not in good_lookup:

        unmatched += 1

        progress(
            counter,
            total,
            start_time
        )

        continue


    good_windows = (
        good_lookup[
            pair_id
        ]
    )


    # --------------------------------------------------------------
    # Flawed recording duration
    # --------------------------------------------------------------

    record_id = row[
        "record_id"
    ]

    flawed_duration = (
        recording_durations.get(
            record_id,
            0.0
        )
    )

    if flawed_duration <= 0:

        unmatched += 1

        progress(
            counter,
            total,
            start_time
        )

        continue


    # --------------------------------------------------------------
    # Good recording duration
    # --------------------------------------------------------------

    good_record_ids = (
        good_df[
            good_df["pair_id"] == pair_id
        ]["record_id"]
        .unique()
    )

    if len(good_record_ids) == 0:

        unmatched += 1

        progress(
            counter,
            total,
            start_time
        )

        continue


    good_record_id = (
        good_record_ids[0]
    )

    good_duration = (
        recording_durations.get(
            good_record_id,
            0.0
        )
    )

    if good_duration <= 0:

        unmatched += 1

        progress(
            counter,
            total,
            start_time
        )

        continue


    # --------------------------------------------------------------
    # Flawed window timing
    # --------------------------------------------------------------

    flawed_window_start = safe_float(
        row["window_start"]
    )

    flawed_window_end = safe_float(
        row["window_end"]
    )

    flawed_window_duration = (
        flawed_window_end
        - flawed_window_start
    )

    flawed_center = (
        flawed_window_start
        + flawed_window_end
    ) / 2.0


    # --------------------------------------------------------------
    # Relative position
    # --------------------------------------------------------------

    relative_position = (
        flawed_center
        / flawed_duration
    )


    # --------------------------------------------------------------
    # Find closest GOOD window
    # --------------------------------------------------------------

    good_row = (
        find_nearest_good_window(
            good_windows,
            relative_position
        )
    )

    if good_row is None:

        unmatched += 1

        progress(
            counter,
            total,
            start_time
        )

        continue


    # --------------------------------------------------------------
    # Good window timing
    # --------------------------------------------------------------

    good_window_start = safe_float(
        good_row["window_start"]
    )

    good_window_end = safe_float(
        good_row["window_end"]
    )

    good_window_duration = (
        good_window_end
        - good_window_start
    )


    # --------------------------------------------------------------
    # BASIC OUTPUT
    # --------------------------------------------------------------

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
            row["severity"],

        "temporal_label":
            row["temporal_label"],

        "good_window_start":
            good_window_start,

        "good_window_end":
            good_window_end,

        # ----------------------------------------------------------
        # NEW TEMPORAL FEATURES
        # ----------------------------------------------------------

        "flawed_recording_duration":
            flawed_duration,

        "good_recording_duration":
            good_duration,

        "recording_duration_delta":
            flawed_duration
            - good_duration,

        "recording_duration_ratio":
            flawed_duration
            / (
                good_duration
                + EPS
            ),

        "flawed_window_duration":
            flawed_window_duration,

        "good_window_duration":
            good_window_duration,

        "window_duration_delta":
            flawed_window_duration
            - good_window_duration,

        "window_duration_ratio":
            flawed_window_duration
            / (
                good_window_duration
                + EPS
            ),

        "window_compression_ratio":
            good_window_duration
            / (
                flawed_window_duration
                + EPS
            ),

        # How far the matched GOOD window is from
        # the ideal relative position.
        "alignment_error":
            abs(
                relative_position
                - safe_float(
                    good_row[
                        "_relative_position"
                    ]
                )
            ),
    }


    # ==================================================================
    # ORIGINAL DELTA FEATURES
    # ==================================================================

    for feature in feature_columns:

        flawed_value = row[
            feature
        ]

        good_value = good_row[
            feature
        ]

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
                    + EPS
                )
            )


        output[
            f"delta_{feature}"
        ] = delta

        output[
            f"relative_delta_{feature}"
        ] = relative_delta


    # ==================================================================
    # EXPLICIT TEMPORAL / ACOUSTIC FEATURES
    # ==================================================================

    def get_feature(
        source,
        feature_name
    ):

        if feature_name not in source:

            return 0.0

        return safe_float(
            source[
                feature_name
            ]
        )


    # --------------------------------------------------------------
    # Useful acoustic features
    # --------------------------------------------------------------

    flawed_rms = get_feature(
        row,
        "rms_db"
    )

    good_rms = get_feature(
        good_row,
        "rms_db"
    )

    flawed_silence = get_feature(
        row,
        "silence_ratio"
    )

    good_silence = get_feature(
        good_row,
        "silence_ratio"
    )

    flawed_f0 = get_feature(
        row,
        "f0_mean"
    )

    good_f0 = get_feature(
        good_row,
        "f0_mean"
    )

    flawed_wpm = get_feature(
        row,
        "global_wpm"
    )

    good_wpm = get_feature(
        good_row,
        "global_wpm"
    )


    # Some checkpoints may use slightly different
    # F0 / WPM naming. Try alternatives.

    if (
        flawed_f0 == 0.0
        and "mean_f0" in row.index
    ):

        flawed_f0 = get_feature(
            row,
            "mean_f0"
        )

        good_f0 = get_feature(
            good_row,
            "mean_f0"
        )


    if (
        flawed_wpm == 0.0
        and "wpm" in row.index
    ):

        flawed_wpm = get_feature(
            row,
            "wpm"
        )

        good_wpm = get_feature(
            good_row,
            "wpm"
        )


    # --------------------------------------------------------------
    # RMS
    # --------------------------------------------------------------

    output[
        "temporal_rms_delta"
    ] = (
        flawed_rms
        - good_rms
    )

    output[
        "temporal_rms_ratio"
    ] = (
        flawed_rms
        - good_rms
    )

    # --------------------------------------------------------------
    # Silence
    # --------------------------------------------------------------

    output[
        "temporal_silence_delta"
    ] = (
        flawed_silence
        - good_silence
    )

    output[
        "temporal_silence_ratio"
    ] = (
        flawed_silence
        / (
            good_silence
            + EPS
        )
    )

    # --------------------------------------------------------------
    # F0
    # --------------------------------------------------------------

    output[
        "temporal_f0_delta"
    ] = (
        flawed_f0
        - good_f0
    )

    output[
        "temporal_f0_ratio"
    ] = (
        flawed_f0
        / (
            abs(good_f0)
            + EPS
        )
    )

    # --------------------------------------------------------------
    # Global WPM
    # --------------------------------------------------------------

    output[
        "temporal_wpm_delta"
    ] = (
        flawed_wpm
        - good_wpm
    )

    output[
        "temporal_wpm_ratio"
    ] = (
        flawed_wpm
        / (
            abs(good_wpm)
            + EPS
        )
    )


    # ==================================================================
    # TEMPORAL POSITION FEATURES
    # ==================================================================

    output[
        "flawed_relative_start"
    ] = (
        flawed_window_start
        / (
            flawed_duration
            + EPS
        )
    )

    output[
        "flawed_relative_end"
    ] = (
        flawed_window_end
        / (
            flawed_duration
            + EPS
        )
    )

    output[
        "good_relative_start"
    ] = (
        good_window_start
        / (
            good_duration
            + EPS
        )
    )

    output[
        "good_relative_end"
    ] = (
        good_window_end
        / (
            good_duration
            + EPS
        )
    )


    output[
        "relative_position_delta"
    ] = (
        output[
            "flawed_relative_start"
        ]
        - output[
            "good_relative_start"
        ]
    )


    # ==================================================================
    # NEIGHBOR CONTEXT
    # ==================================================================

    current_group = (
        recording_lookup.get(
            record_id
        )
    )

    previous_row = get_neighbor(
        current_group,
        flawed_window_start,
        "previous"
    )

    next_row = get_neighbor(
        current_group,
        flawed_window_start,
        "next"
    )


    # --------------------------------------------------------------
    # Previous / next temporal labels
    # --------------------------------------------------------------

    if previous_row is not None:

        output[
            "previous_temporal_label"
        ] = safe_float(
            previous_row[
                "temporal_label"
            ]
        )

    else:

        output[
            "previous_temporal_label"
        ] = 0.0


    if next_row is not None:

        output[
            "next_temporal_label"
        ] = safe_float(
            next_row[
                "temporal_label"
            ]
        )

    else:

        output[
            "next_temporal_label"
        ] = 0.0


    # --------------------------------------------------------------
    # Previous / next acoustic changes
    # --------------------------------------------------------------

    if previous_row is not None:

        previous_rms = get_feature(
            previous_row,
            "rms_db"
        )

        previous_f0 = get_feature(
            previous_row,
            "f0_mean"
        )

        output[
            "previous_rms_delta_from_current"
        ] = (
            flawed_rms
            - previous_rms
        )

        output[
            "previous_f0_delta_from_current"
        ] = (
            flawed_f0
            - previous_f0
        )

    else:

        output[
            "previous_rms_delta_from_current"
        ] = 0.0

        output[
            "previous_f0_delta_from_current"
        ] = 0.0


    if next_row is not None:

        next_rms = get_feature(
            next_row,
            "rms_db"
        )

        next_f0 = get_feature(
            next_row,
            "f0_mean"
        )

        output[
            "next_rms_delta_from_current"
        ] = (
            flawed_rms
            - next_rms
        )

        output[
            "next_f0_delta_from_current"
        ] = (
            flawed_f0
            - next_f0
        )

    else:

        output[
            "next_rms_delta_from_current"
        ] = 0.0

        output[
            "next_f0_delta_from_current"
        ] = 0.0


    # ==================================================================
    # SAVE RESULT
    # ==================================================================

    results.append(
        output
    )

    matched += 1

    progress(
        counter,
        total,
        start_time
    )


# ======================================================================
# CREATE DATAFRAME
# ======================================================================

contrastive_df = pd.DataFrame(
    results
)


# ======================================================================
# CLEAN DATA
# ======================================================================

# Replace infinite values.

contrastive_df = (
    contrastive_df
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


# Fill numeric NaNs.

numeric_columns = (
    contrastive_df
    .select_dtypes(
        include=[np.number]
    )
    .columns
)

contrastive_df[
    numeric_columns
] = (
    contrastive_df[
        numeric_columns
    ]
    .fillna(0.0)
)


# ======================================================================
# SAVE
# ======================================================================

print()

print(
    f"Matched windows:   "
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
    f"Total columns:     "
    f"{len(contrastive_df.columns):,}"
)


# Count model features.

metadata_columns = {
    "record_id",
    "pair_id",
    "speaker_id",
    "window_start",
    "window_end",
    "relative_position",
    "flaw_type",
    "severity",
    "temporal_label",
    "good_window_start",
    "good_window_end",
}

model_columns = [
    column
    for column in contrastive_df.columns
    if column not in metadata_columns
]


print(
    f"Model feature columns: "
    f"{len(model_columns):,}"
)


# ======================================================================
# SAFETY CHECKS
# ======================================================================

header(
    "V2 DATASET SAFETY CHECK"
)


print(
    f"Expected original acoustic features: "
    f"{len(feature_columns):,}"
)


original_delta_count = sum(
    1
    for column in contrastive_df.columns
    if column.startswith(
        "delta_"
    )
    and not column.startswith(
        "delta_delta_"
    )
)


original_relative_delta_count = sum(
    1
    for column in contrastive_df.columns
    if column.startswith(
        "relative_delta_"
    )
)


print(
    f"Delta features found: "
    f"{original_delta_count:,}"
)

print(
    f"Relative delta features found: "
    f"{original_relative_delta_count:,}"
)


print()

print(
    "New temporal features:"
)

new_temporal_features = [
    "recording_duration_delta",
    "recording_duration_ratio",
    "window_duration_delta",
    "window_duration_ratio",
    "window_compression_ratio",
    "alignment_error",
    "temporal_rms_delta",
    "temporal_rms_ratio",
    "temporal_silence_delta",
    "temporal_silence_ratio",
    "temporal_f0_delta",
    "temporal_f0_ratio",
    "temporal_wpm_delta",
    "temporal_wpm_ratio",
    "relative_position_delta",
    "previous_temporal_label",
    "next_temporal_label",
    "previous_rms_delta_from_current",
    "previous_f0_delta_from_current",
    "next_rms_delta_from_current",
    "next_f0_delta_from_current",
]

for feature in new_temporal_features:

    exists = feature in contrastive_df.columns

    status = "OK" if exists else "MISSING"

    print(
        f"    [{status}] {feature}"
    )


# ======================================================================
# SAVE
# ======================================================================

joblib.dump(
    contrastive_df,
    OUTPUT_FILE,
    compress=3
)


# ======================================================================
# FINAL SUMMARY
# ======================================================================

header(
    "CONTRASTIVE V2 DATASET COMPLETE"
)

print(
    f"""
Original temporal windows:

    {len(df):,}

Good windows:

    {len(good_df):,}

Flawed windows:

    {len(flawed_df):,}

Successfully paired:

    {matched:,}

Unmatched:

    {unmatched:,}

Output rows:

    {len(contrastive_df):,}

Total columns:

    {len(contrastive_df.columns):,}

Model features:

    {len(model_columns):,}

Output:

    {OUTPUT_FILE}
"""
)

print(
    "IMPORTANT:"
)

print(
    "The original contrastive_features.pkl was NOT modified."
)

print()

print(
    "[OK] Contrastive feature construction V2 complete."
)