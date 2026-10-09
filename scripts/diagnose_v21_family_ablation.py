import os
import pickle
import joblib
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# ORATORIQ
# V2.1 FLAW FAMILY ABLATION / ROBUSTNESS TEST
#
# Goal:
# Determine whether the 100% family classification result
# is robust or mainly caused by timing/recording features.
#
# Tests:
#
#   TEST A:
#       All V2.1 features
#
#   TEST B:
#       Remove timing / recording / alignment features
#
#   TEST C:
#       Acoustic contrastive features only
#
# All tests use the SAME speaker-safe split.
# ============================================================


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = r"artifacts\contrastive_features_v21.pkl"

OUTPUT_REPORT = (
    r"artifacts\results\v21_family_ablation_report.json"
)

OUTPUT_CSV = (
    r"artifacts\results\v21_family_ablation_results.csv"
)

RANDOM_STATE = 42


# ============================================================
# FAMILY DEFINITIONS
# ============================================================

TEMPORAL_FLAWS = {
    "fast_pacing",
    "slow_pacing",
    "long_pause",
}

ACOUSTIC_FLAWS = {
    "high_volume",
    "low_volume",
    "pitch_deviation",
}


# ============================================================
# LOG
# ============================================================

def log(message):
    print(
        f"[V2.1-ABLATION] {message}",
        flush=True
    )


# ============================================================
# LOAD FEATURE FILE
# ============================================================

def load_feature_file(path):

    log(f"Loading: {path}")

    try:

        with open(path, "rb") as f:
            data = pickle.load(f)

        log("Loaded using pickle.")

        return data

    except Exception as pickle_error:

        log(
            "Pickle loading failed. "
            "Trying joblib..."
        )

        try:

            data = joblib.load(path)

            log("Loaded using joblib.")

            return data

        except Exception as joblib_error:

            raise RuntimeError(
                "Could not load feature file.\n\n"
                f"Pickle error: {pickle_error}\n"
                f"Joblib error: {joblib_error}"
            )


# ============================================================
# EXTRACT DATAFRAME
# ============================================================

def extract_dataframe(data):

    if isinstance(data, pd.DataFrame):

        return data.copy()

    if isinstance(data, dict):

        possible_keys = [
            "data",
            "df",
            "features",
            "contrastive_features",
        ]

        for key in possible_keys:

            if key in data:

                if isinstance(
                    data[key],
                    pd.DataFrame
                ):

                    return data[key].copy()

        for key, value in data.items():

            if isinstance(
                value,
                pd.DataFrame
            ):

                return value.copy()

    raise ValueError(
        "Could not find a pandas DataFrame "
        "inside the feature file."
    )


# ============================================================
# CREATE FAMILY LABEL
# ============================================================

def create_family_label(flaw_type):

    if flaw_type in TEMPORAL_FLAWS:

        return "temporal"

    if flaw_type in ACOUSTIC_FLAWS:

        return "acoustic"

    return None


# ============================================================
# FIND MODEL FEATURES
# ============================================================

def get_model_features(df):

    metadata = {
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
        "family",
    }

    features = []

    for column in df.columns:

        if column in metadata:
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            features.append(column)

    return features


# ============================================================
# SPEAKER SAFE SPLIT
#
# IMPORTANT:
# This follows the same 60/20/20 speaker split logic
# used by the family classifier.
# ============================================================

def speaker_safe_split(
    df,
    random_state=42,
):

    speakers = (
        df["speaker_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    speakers = np.array(
        speakers
    )

    rng = np.random.RandomState(
        random_state
    )

    rng.shuffle(
        speakers
    )

    n_speakers = len(
        speakers
    )

    if n_speakers < 3:

        raise ValueError(
            f"Need at least 3 speakers. "
            f"Found {n_speakers}."
        )

    n_train = max(
        1,
        int(round(
            n_speakers * 0.60
        ))
    )

    n_val = max(
        1,
        int(round(
            n_speakers * 0.20
        ))
    )

    if (
        n_train + n_val
        >= n_speakers
    ):

        n_val = max(
            1,
            n_speakers
            - n_train
            - 1
        )

    train_speakers = set(
        speakers[:n_train]
    )

    val_speakers = set(
        speakers[
            n_train:
            n_train + n_val
        ]
    )

    test_speakers = set(
        speakers[
            n_train + n_val:
        ]
    )

    train_df = df[
        df["speaker_id"]
        .astype(str)
        .isin(train_speakers)
    ].copy()

    val_df = df[
        df["speaker_id"]
        .astype(str)
        .isin(val_speakers)
    ].copy()

    test_df = df[
        df["speaker_id"]
        .astype(str)
        .isin(test_speakers)
    ].copy()

    return (
        train_df,
        val_df,
        test_df,
        train_speakers,
        val_speakers,
        test_speakers,
    )


# ============================================================
# FEATURE GROUPING
# ============================================================

def is_timing_feature(feature):

    name = feature.lower()

    timing_keywords = [
        "global_wpm",
        "temporal_wpm",
        "recording_duration",
        "window_duration",
        "compression",
        "alignment_error",
        "relative_position",
        "position_delta",
        "duration",
        "pacing",
        "silence",
        "voiced_ratio",
    ]

    return any(
        keyword in name
        for keyword in timing_keywords
    )


def is_acoustic_feature(feature):

    name = feature.lower()

    acoustic_keywords = [
        "rms",
        "zcr",
        "spectral",
        "mfcc",
        "f0",
        "pitch",
        "delta_",
        "relative_delta_",
    ]

    return any(
        keyword in name
        for keyword in acoustic_keywords
    )


# ============================================================
# PREPARE X / Y
# ============================================================

def prepare_xy(
    df,
    feature_columns,
):

    X = (
        df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0.0)
    )

    y = (
        df["family"]
        .astype(str)
    )

    return X, y


# ============================================================
# TRAIN + EVALUATE
# ============================================================

def run_experiment(
    name,
    feature_columns,
    train_df,
    val_df,
    test_df,
):

    print("")
    print("=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)

    log(
        f"Features used: {len(feature_columns):,}"
    )

    if len(feature_columns) == 0:

        raise ValueError(
            f"No features available for experiment: {name}"
        )

    X_train, y_train = prepare_xy(
        train_df,
        feature_columns
    )

    X_val, y_val = prepare_xy(
        val_df,
        feature_columns
    )

    X_test, y_test = prepare_xy(
        test_df,
        feature_columns
    )

    # --------------------------------------------------------
    # Random Forest
    # Same model that achieved 1.0 previously.
    # --------------------------------------------------------

    log(
        "Training RandomForest..."
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    log(
        "Training complete."
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_pred = model.predict(
        X_val
    )

    val_accuracy = accuracy_score(
        y_val,
        val_pred
    )

    val_precision = precision_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0,
    )

    val_recall = recall_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0,
    )

    val_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_pred = model.predict(
        X_test
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    test_precision = precision_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0,
    )

    test_recall = recall_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0,
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("")
    print("Validation:")
    print(
        f"  Accuracy:       {val_accuracy:.4f}"
    )
    print(
        f"  Macro Precision:{val_precision:.4f}"
    )
    print(
        f"  Macro Recall:   {val_recall:.4f}"
    )
    print(
        f"  Macro F1:       {val_f1:.4f}"
    )

    print("")
    print("Test:")
    print(
        f"  Accuracy:       {test_accuracy:.4f}"
    )
    print(
        f"  Macro Precision:{test_precision:.4f}"
    )
    print(
        f"  Macro Recall:   {test_recall:.4f}"
    )
    print(
        f"  Macro F1:       {test_f1:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("")
    print("Test classification report:")
    print(
        classification_report(
            y_test,
            test_pred,
            zero_division=0
        ),
        flush=True
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    labels = [
        "acoustic",
        "temporal",
    ]

    cm = confusion_matrix(
        y_test,
        test_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "actual_acoustic",
            "actual_temporal",
        ],
        columns=[
            "predicted_acoustic",
            "predicted_temporal",
        ],
    )

    print(
        "Confusion matrix:"
    )

    print(
        cm_df.to_string(),
        flush=True
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    top_features = []

    if hasattr(
        model,
        "feature_importances_"
    ):

        pairs = list(
            zip(
                feature_columns,
                model.feature_importances_,
            )
        )

        pairs.sort(
            key=lambda x: x[1],
            reverse=True
        )

        top_features = [
            {
                "feature": feature,
                "importance": float(
                    importance
                ),
            }
            for feature, importance
            in pairs[:20]
        ]

        print("")
        print(
            "Top 20 features:"
        )

        for item in top_features:

            print(
                f"  "
                f"{item['feature']:50s}"
                f" {item['importance']:.6f}"
            )

    return {
        "experiment": name,
        "feature_count": len(
            feature_columns
        ),

        "validation_accuracy": float(
            val_accuracy
        ),

        "validation_macro_precision": float(
            val_precision
        ),

        "validation_macro_recall": float(
            val_recall
        ),

        "validation_macro_f1": float(
            val_f1
        ),

        "test_accuracy": float(
            test_accuracy
        ),

        "test_macro_precision": float(
            test_precision
        ),

        "test_macro_recall": float(
            test_recall
        ),

        "test_macro_f1": float(
            test_f1
        ),

        "confusion_matrix": cm.tolist(),

        "top_features": top_features,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore"
    )

    print("")
    print("=" * 70)
    print(
        "ORATORIQ — V2.1 FAMILY ROBUSTNESS TEST"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.exists(
        INPUT_FILE
    ):

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    os.makedirs(
        os.path.dirname(
            OUTPUT_REPORT
        ),
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_feature_file(
        INPUT_FILE
    )

    df = extract_dataframe(
        data
    )

    log(
        f"Total rows: {len(df):,}"
    )

    log(
        f"Total columns: {len(df.columns):,}"
    )

    # --------------------------------------------------------
    # Keep flawed windows
    # --------------------------------------------------------

    df = df[
        df["flaw_type"].notna()
    ].copy()

    # --------------------------------------------------------
    # Family labels
    # --------------------------------------------------------

    df["family"] = (
        df["flaw_type"]
        .astype(str)
        .map(
            create_family_label
        )
    )

    unknown = df[
        df["family"].isna()
    ]

    if len(unknown) > 0:

        raise ValueError(
            "Unknown flaw types found: "
            f"{unknown['flaw_type'].unique().tolist()}"
        )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    all_features = get_model_features(
        df
    )

    log(
        f"All model features: "
        f"{len(all_features):,}"
    )

    # --------------------------------------------------------
    # Explicit leakage check
    # --------------------------------------------------------

    leakage_features = []

    for feature in all_features:

        lower = feature.lower()

        if (
            "temporal_label" in lower
            or "previous_temporal" in lower
            or "next_temporal" in lower
        ):

            leakage_features.append(
                feature
            )

    if leakage_features:

        raise ValueError(
            "LEAKAGE FEATURES FOUND:\n"
            + "\n".join(
                leakage_features
            )
        )

    log(
        "Label leakage check: PASS"
    )

    # --------------------------------------------------------
    # Speaker split
    # --------------------------------------------------------

    (
        train_df,
        val_df,
        test_df,
        train_speakers,
        val_speakers,
        test_speakers,
    ) = speaker_safe_split(
        df,
        random_state=RANDOM_STATE,
    )

    log("")
    log("Speaker-safe split:")
    log(
        f"  Train speakers: {len(train_speakers)}"
    )
    log(
        f"  Validation speakers: {len(val_speakers)}"
    )
    log(
        f"  Test speakers: {len(test_speakers)}"
    )

    log("")
    log("Window split:")
    log(
        f"  Train: {len(train_df):,}"
    )
    log(
        f"  Validation: {len(val_df):,}"
    )
    log(
        f"  Test: {len(test_df):,}"
    )

    # --------------------------------------------------------
    # Verify speaker isolation
    # --------------------------------------------------------

    train_set = set(
        train_df["speaker_id"]
        .astype(str)
    )

    val_set = set(
        val_df["speaker_id"]
        .astype(str)
    )

    test_set = set(
        test_df["speaker_id"]
        .astype(str)
    )

    assert train_set.isdisjoint(
        val_set
    )

    assert train_set.isdisjoint(
        test_set
    )

    assert val_set.isdisjoint(
        test_set
    )

    log(
        "Speaker leakage check: PASS"
    )

    # ========================================================
    # FEATURE GROUPS
    # ========================================================

    timing_features = [
        feature
        for feature in all_features
        if is_timing_feature(feature)
    ]

    acoustic_features = [
        feature
        for feature in all_features
        if is_acoustic_feature(feature)
    ]

    # Remove timing features from all features
    # for Experiment B.
    no_timing_features = [
        feature
        for feature in all_features
        if feature not in timing_features
    ]

    # Remove anything that wasn't confidently assigned
    # to acoustic features for Experiment C.
    acoustic_only_features = [
        feature
        for feature in acoustic_features
        if feature in all_features
    ]

    # Remove possible timing features accidentally caught
    # by broad acoustic keywords.
    acoustic_only_features = [
        feature
        for feature in acoustic_only_features
        if not is_timing_feature(feature)
    ]

    # --------------------------------------------------------
    # Print feature groups
    # --------------------------------------------------------

    print("")
    print("=" * 70)
    print("FEATURE GROUP ANALYSIS")
    print("=" * 70)

    print(
        f"All features:              "
        f"{len(all_features):4d}"
    )

    print(
        f"Timing/recording features: "
        f"{len(timing_features):4d}"
    )

    print(
        f"No-timing features:        "
        f"{len(no_timing_features):4d}"
    )

    print(
        f"Acoustic-only features:     "
        f"{len(acoustic_only_features):4d}"
    )

    print("")
    print("Timing features removed in Test B:")

    for feature in timing_features:

        print(
            f"  - {feature}"
        )

    # ========================================================
    # RUN EXPERIMENT A
    # ========================================================

    result_a = run_experiment(
        "A_ALL_FEATURES",
        all_features,
        train_df,
        val_df,
        test_df,
    )

    # ========================================================
    # RUN EXPERIMENT B
    # ========================================================

    result_b = run_experiment(
        "B_NO_TIMING_FEATURES",
        no_timing_features,
        train_df,
        val_df,
        test_df,
    )

    # ========================================================
    # RUN EXPERIMENT C
    # ========================================================

    result_c = run_experiment(
        "C_ACOUSTIC_ONLY",
        acoustic_only_features,
        train_df,
        val_df,
        test_df,
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    results = [
        result_a,
        result_b,
        result_c,
    ]

    comparison_df = pd.DataFrame(
        [
            {
                "experiment": r[
                    "experiment"
                ],
                "features": r[
                    "feature_count"
                ],
                "val_accuracy": r[
                    "validation_accuracy"
                ],
                "val_macro_f1": r[
                    "validation_macro_f1"
                ],
                "test_accuracy": r[
                    "test_accuracy"
                ],
                "test_macro_f1": r[
                    "test_macro_f1"
                ],
            }
            for r in results
        ]
    )

    print("")
    print("=" * 70)
    print("ABLATION COMPARISON")
    print("=" * 70)

    print(
        comparison_df.to_string(
            index=False
        ),
        flush=True
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    score_a = result_a[
        "test_macro_f1"
    ]

    score_b = result_b[
        "test_macro_f1"
    ]

    score_c = result_c[
        "test_macro_f1"
    ]

    drop_no_timing = (
        score_a - score_b
    )

    drop_acoustic = (
        score_a - score_c
    )

    if score_b >= 0.90:

        timing_interpretation = (
            "STRONG: family classification remains "
            "high after removing timing/recording "
            "features."
        )

    elif score_b >= 0.75:

        timing_interpretation = (
            "MODERATE: removing timing features "
            "reduces performance, but the family "
            "signal remains useful."
        )

    else:

        timing_interpretation = (
            "WEAK: performance drops substantially "
            "without timing features. The original "
            "100% result is strongly dependent on "
            "timing/recording information."
        )

    if score_c >= 0.90:

        acoustic_interpretation = (
            "STRONG: acoustic features alone "
            "separate the two families very well."
        )

    elif score_c >= 0.75:

        acoustic_interpretation = (
            "MODERATE: acoustic features contain "
            "useful family information."
        )

    else:

        acoustic_interpretation = (
            "WEAK: acoustic features alone do not "
            "reliably separate the families."
        )

    print("")
    print("=" * 70)
    print("ROBUSTNESS INTERPRETATION")
    print("=" * 70)

    print("")
    print(
        f"Test A → Test B F1 drop: "
        f"{drop_no_timing:.4f}"
    )

    print(
        f"Test A → Test C F1 drop: "
        f"{drop_acoustic:.4f}"
    )

    print("")
    print(
        f"Timing robustness: "
        f"{timing_interpretation}"
    )

    print("")
    print(
        f"Acoustic robustness: "
        f"{acoustic_interpretation}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    comparison_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    report = {
        "version": "V2.1",
        "random_state": RANDOM_STATE,

        "rows_total": int(
            len(df)
        ),

        "rows_train": int(
            len(train_df)
        ),

        "rows_validation": int(
            len(val_df)
        ),

        "rows_test": int(
            len(test_df)
        ),

        "train_speakers": int(
            len(train_speakers)
        ),

        "validation_speakers": int(
            len(val_speakers)
        ),

        "test_speakers": int(
            len(test_speakers)
        ),

        "all_feature_count": int(
            len(all_features)
        ),

        "timing_feature_count": int(
            len(timing_features)
        ),

        "no_timing_feature_count": int(
            len(no_timing_features)
        ),

        "acoustic_only_feature_count": int(
            len(acoustic_only_features)
        ),

        "timing_features": timing_features,

        "experiments": results,

        "test_f1_drop_without_timing":
            float(drop_no_timing),

        "test_f1_drop_acoustic_only":
            float(drop_acoustic),

        "timing_interpretation":
            timing_interpretation,

        "acoustic_interpretation":
            acoustic_interpretation,
    }

    import json

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # ========================================================
    # FINAL
    # ========================================================

    print("")
    print("=" * 70)
    print("V2.1 ABLATION TEST COMPLETE")
    print("=" * 70)

    print("")
    print("Saved comparison:")
    print(
        f"  {OUTPUT_CSV}"
    )

    print("")
    print("Saved report:")
    print(
        f"  {OUTPUT_REPORT}"
    )

    print("")
    print("IMPORTANT:")
    print(
        "Do NOT replace the production family model yet."
    )

    print(
        "Use the ablation results to decide whether "
        "the 1.0000 score is robust."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()