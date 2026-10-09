import os
import pickle
import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = r"artifacts\contrastive_features_v2.pkl"
OUTPUT_FILE = r"artifacts\contrastive_features_v21.pkl"

# These are derived from ground-truth labels.
# They MUST NOT be used as model inputs.
LEAKAGE_COLUMNS = [
    "previous_temporal_label",
    "next_temporal_label",
]


# ============================================================
# HELPERS
# ============================================================

def log(message):
    print(f"[V2.1] {message}", flush=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_feature_file(path):

    log("Trying standard pickle loader...")

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)

        log("Standard pickle loader: SUCCESS")
        return data

    except (pickle.UnpicklingError, EOFError, ValueError, AttributeError) as e:

        log(
            f"Standard pickle loader failed: {type(e).__name__}"
        )

        log("Trying joblib loader...")

        try:
            data = joblib.load(path)

            log("Joblib loader: SUCCESS")
            return data

        except Exception as joblib_error:

            raise RuntimeError(
                "Could not load the V2 feature file using "
                "either pickle or joblib.\n\n"
                f"Pickle error: {e}\n"
                f"Joblib error: {joblib_error}"
            )


# ============================================================
# IDENTIFY DATAFRAME
# ============================================================

def extract_dataframe(data):

    # Direct DataFrame
    if isinstance(data, pd.DataFrame):

        return data.copy()

    # Dictionary containing DataFrame
    if isinstance(data, dict):

        possible_keys = [
            "data",
            "df",
            "features",
            "contrastive_features",
        ]

        for key in possible_keys:

            if key in data:

                value = data[key]

                if isinstance(value, pd.DataFrame):

                    return value.copy()

        # Sometimes the dictionary itself may contain
        # a dataframe-like object under another key.
        for key, value in data.items():

            if isinstance(value, pd.DataFrame):

                log(
                    f"Found DataFrame under dictionary key: {key}"
                )

                return value.copy()

        raise ValueError(
            "A dictionary was loaded, but no pandas DataFrame "
            "was found inside it.\n"
            f"Available keys: {list(data.keys())}"
        )

    raise ValueError(
        f"Unsupported loaded object type: {type(data)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ORATORIQ — CONTRASTIVE FEATURES V2.1")
    print("Clean deployment-safe feature set")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    log(f"Loading: {INPUT_FILE}")

    # --------------------------------------------------------
    # Load V2 file
    # --------------------------------------------------------

    data = load_feature_file(INPUT_FILE)

    # --------------------------------------------------------
    # Identify dataframe
    # --------------------------------------------------------

    df = extract_dataframe(data)

    log(f"Rows loaded: {len(df):,}")
    log(f"Columns loaded: {len(df.columns):,}")

    # --------------------------------------------------------
    # Show original columns
    # --------------------------------------------------------

    log("")
    log("Checking for label-derived features...")

    # --------------------------------------------------------
    # Find leakage columns
    # --------------------------------------------------------

    found_leakage = [
        col
        for col in LEAKAGE_COLUMNS
        if col in df.columns
    ]

    log(
        f"Leakage columns found: {len(found_leakage)}"
    )

    for col in found_leakage:

        log(f"  Removing: {col}")

    # --------------------------------------------------------
    # Remove label-derived features
    # --------------------------------------------------------

    df = df.drop(
        columns=found_leakage,
        errors="ignore"
    )

    # --------------------------------------------------------
    # Fix temporal RMS feature naming
    #
    # V2 called this:
    #
    #     temporal_rms_ratio
    #
    # But its actual calculation was:
    #
    #     flawed_rms - good_rms
    #
    # Therefore it is a DELTA, not a ratio.
    # --------------------------------------------------------

    if "temporal_rms_ratio" in df.columns:

        if "temporal_rms_delta" not in df.columns:

            df["temporal_rms_delta"] = (
                df["temporal_rms_ratio"]
            )

        df = df.drop(
            columns=["temporal_rms_ratio"]
        )

        log(
            "Fixed feature name:"
        )

        log(
            "  temporal_rms_ratio -> "
            "temporal_rms_delta"
        )

    # --------------------------------------------------------
    # Metadata columns
    # --------------------------------------------------------

    metadata_columns = [
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
    ]

    metadata_columns = [
        col
        for col in metadata_columns
        if col in df.columns
    ]

    # --------------------------------------------------------
    # Identify model features
    # --------------------------------------------------------

    model_features = [
        col
        for col in df.columns
        if col not in metadata_columns
    ]

    # --------------------------------------------------------
    # Keep numeric model features only
    # --------------------------------------------------------

    numeric_features = []

    for col in model_features:

        if pd.api.types.is_numeric_dtype(df[col]):

            numeric_features.append(col)

    model_features = numeric_features

    # --------------------------------------------------------
    # Check for accidental label leakage
    # --------------------------------------------------------

    forbidden_words = [
        "temporal_label",
        "previous_temporal",
        "next_temporal",
    ]

    suspicious = []

    for col in model_features:

        lower = col.lower()

        for word in forbidden_words:

            if word in lower:

                suspicious.append(col)

                break

    if suspicious:

        log("")
        log(
            "WARNING — suspicious label-derived "
            "features found:"
        )

        for col in suspicious:

            log(f"  {col}")

        raise ValueError(
            "Potential label leakage detected.\n"
            "Remove these features before training."
        )

    # --------------------------------------------------------
    # NaN / infinite cleanup
    # --------------------------------------------------------

    log("")
    log("Checking NaN and infinite values...")

    numeric_matrix = (
        df[model_features]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    nan_count = int(
        numeric_matrix.isna().sum().sum()
    )

    if nan_count > 0:

        log(
            f"Found {nan_count:,} NaN values."
        )

        log(
            "Replacing NaN values with "
            "column medians..."
        )

        numeric_matrix = (
            numeric_matrix.fillna(
                numeric_matrix.median()
            )
        )

        df[model_features] = numeric_matrix

    else:

        log("No NaN values found.")

    # --------------------------------------------------------
    # Dataset statistics
    # --------------------------------------------------------

    log("")
    log("Dataset summary")
    log("-" * 50)

    log(
        f"Rows:              {len(df):,}"
    )

    log(
        f"Total columns:     {len(df.columns):,}"
    )

    log(
        f"Metadata columns:  {len(metadata_columns):,}"
    )

    log(
        f"Model features:    {len(model_features):,}"
    )

    # --------------------------------------------------------
    # Flaw distribution
    # --------------------------------------------------------

    if "flaw_type" in df.columns:

        log("")
        log("Flaw distribution:")
        log("-" * 50)

        counts = (
            df["flaw_type"]
            .value_counts()
        )

        for label, count in counts.items():

            log(
                f"  {str(label):20s} {count:,}"
            )

    # --------------------------------------------------------
    # Temporal label distribution
    # --------------------------------------------------------

    if "temporal_label" in df.columns:

        log("")
        log("Temporal label distribution:")
        log("-" * 50)

        counts = (
            df["temporal_label"]
            .value_counts()
        )

        for label, count in counts.items():

            log(
                f"  {str(label):20s} {count:,}"
            )

    # --------------------------------------------------------
    # Save V2.1
    # --------------------------------------------------------

    output = {
        "data": df,
        "model_features": model_features,
        "metadata_columns": metadata_columns,
        "removed_leakage_columns": found_leakage,
        "version": "V2.1",
    }

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    log("")
    log(
        f"Saving V2.1 feature file:"
    )

    log(
        f"  {OUTPUT_FILE}"
    )

    with open(
        OUTPUT_FILE,
        "wb"
    ) as f:

        pickle.dump(
            output,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    # --------------------------------------------------------
    # Verify saved file
    # --------------------------------------------------------

    log("")
    log("Verifying saved file...")

    with open(
        OUTPUT_FILE,
        "rb"
    ) as f:

        check = pickle.load(f)

    check_df = check["data"]

    log(
        f"Verified rows:     {len(check_df):,}"
    )

    log(
        f"Verified features: "
        f"{len(check['model_features']):,}"
    )

    # --------------------------------------------------------
    # Final leakage check
    # --------------------------------------------------------

    remaining_leakage = []

    for col in check["model_features"]:

        if col in LEAKAGE_COLUMNS:

            remaining_leakage.append(col)

    if remaining_leakage:

        raise ValueError(
            f"Leakage still present: "
            f"{remaining_leakage}"
        )

    # Additional string-based check
    remaining_suspicious = []

    for col in check["model_features"]:

        lower = col.lower()

        if (
            "previous_temporal" in lower
            or "next_temporal" in lower
            or "temporal_label" in lower
        ):

            remaining_suspicious.append(col)

    if remaining_suspicious:

        raise ValueError(
            "Suspicious label-derived features "
            f"still present: {remaining_suspicious}"
        )

    log("")
    log("Leakage check: PASS")

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("")
    print("=" * 70)
    print("V2.1 FEATURE BUILD COMPLETE")
    print("=" * 70)

    print("")
    print("Output:")
    print(f"  {OUTPUT_FILE}")

    print("")
    print("Removed leakage features:")

    if found_leakage:

        for col in found_leakage:

            print(f"  - {col}")

    else:

        print("  None found")

    print("")
    print(
        f"Final model features: "
        f"{len(model_features):,}"
    )

    print("")
    print("Next step:")
    print("  Train the clean flaw-family classifier.")

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()