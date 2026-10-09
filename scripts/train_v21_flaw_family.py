import os
import json
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.preprocessing import LabelEncoder


# ============================================================
# ORATORIQ
# V2.1 CLEAN FLAW FAMILY CLASSIFIER
#
# Families:
#
#   TEMPORAL:
#       fast_pacing
#       slow_pacing
#       long_pause
#
#   ACOUSTIC:
#       high_volume
#       low_volume
#       pitch_deviation
#
# IMPORTANT:
#   - Uses V2.1 features
#   - Does NOT use ground-truth temporal labels as features
#   - Speaker-safe train/validation/test split
#   - Normal windows are excluded from this classifier
# ============================================================


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = r"artifacts\contrastive_features_v21.pkl"

MODEL_DIR = r"artifacts\models"
RESULT_DIR = r"artifacts\results"

BEST_MODEL_FILE = (
    r"artifacts\models\contrastive_v21_flaw_family_classifier.joblib"
)

REPORT_FILE = (
    r"artifacts\results\contrastive_v21_flaw_family_report.json"
)

CONFUSION_FILE = (
    r"artifacts\results\contrastive_v21_flaw_family_confusion.csv"
)

PREDICTIONS_FILE = (
    r"artifacts\results\contrastive_v21_flaw_family_predictions.csv"
)

RANDOM_STATE = 42


# ============================================================
# FAMILY MAPPING
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
# LOGGING
# ============================================================

def log(message):
    print(f"[V2.1-FAMILY] {message}", flush=True)


# ============================================================
# LOAD FEATURE FILE
# ============================================================

def load_feature_file(path):

    log(f"Loading feature file: {path}")

    try:

        with open(path, "rb") as f:
            data = pickle.load(f)

        log("Loaded using pickle.")

        return data

    except Exception as pickle_error:

        log(
            f"Pickle loading failed: "
            f"{type(pickle_error).__name__}"
        )

        log("Trying joblib loader...")

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

                if isinstance(data[key], pd.DataFrame):

                    return data[key].copy()

        for key, value in data.items():

            if isinstance(value, pd.DataFrame):

                log(
                    f"Found dataframe under key: {key}"
                )

                return value.copy()

        raise ValueError(
            "No pandas DataFrame found inside feature file."
        )

    raise ValueError(
        f"Unsupported feature file type: {type(data)}"
    )


# ============================================================
# FIND MODEL FEATURES
# ============================================================

def get_model_features(df):

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

    model_features = []

    for column in df.columns:

        if column in metadata_columns:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):

            model_features.append(column)

    return model_features


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
# SPEAKER-SAFE SPLIT
# ============================================================

def speaker_safe_split(
    df,
    train_fraction=0.60,
    val_fraction=0.20,
    random_state=42,
):

    speakers = (
        df["speaker_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    rng = np.random.RandomState(random_state)

    speakers = np.array(speakers)

    rng.shuffle(speakers)

    n_speakers = len(speakers)

    if n_speakers < 3:

        raise ValueError(
            f"Need at least 3 speakers. "
            f"Found {n_speakers}."
        )

    n_train = max(
        1,
        int(round(n_speakers * train_fraction))
    )

    n_val = max(
        1,
        int(round(n_speakers * val_fraction))
    )

    # Make sure at least one speaker remains for test.
    if n_train + n_val >= n_speakers:

        n_val = max(
            1,
            n_speakers - n_train - 1
        )

    train_speakers = set(
        speakers[:n_train]
    )

    val_speakers = set(
        speakers[
            n_train:n_train + n_val
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
# PREPARE FEATURES
# ============================================================

def prepare_xy(df, feature_columns):

    X = (
        df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # Fill missing values using training-independent
    # median calculation for robustness.
    X = X.fillna(0.0)

    y = df["family"].astype(str)

    return X, y


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    dataset_name,
):

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    precision = precision_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    log("")
    log("=" * 60)
    log(f"{dataset_name} RESULTS")
    log("=" * 60)

    log(
        f"Accuracy:      {accuracy:.4f}"
    )

    log(
        f"Macro Precision: {precision:.4f}"
    )

    log(
        f"Macro Recall:    {recall:.4f}"
    )

    log(
        f"Macro F1:      {macro_f1:.4f}"
    )

    log(
        f"Weighted F1:   {weighted_f1:.4f}"
    )

    log("")
    log("Classification report:")

    report_text = classification_report(
        y,
        predictions,
        zero_division=0,
    )

    print(
        report_text,
        flush=True
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "predictions": predictions,
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
    print("ORATORIQ — V2.1 FLAW FAMILY CLASSIFIER")
    print("=" * 70)

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Feature file not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data = load_feature_file(
        INPUT_FILE
    )

    df = extract_dataframe(
        data
    )

    log(
        f"Loaded rows: {len(df):,}"
    )

    log(
        f"Loaded columns: {len(df.columns):,}"
    )

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "flaw_type",
        "speaker_id",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Keep only flawed windows
    #
    # Normal windows have no flaw type and therefore cannot
    # belong to either family.
    # --------------------------------------------------------

    log("")
    log(
        "Filtering to flawed windows..."
    )

    df = df[
        df["flaw_type"].notna()
    ].copy()

    log(
        f"Flawed rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # Create family labels
    # --------------------------------------------------------

    df["family"] = (
        df["flaw_type"]
        .astype(str)
        .map(create_family_label)
    )

    # --------------------------------------------------------
    # Check unknown flaw types
    # --------------------------------------------------------

    unknown_rows = df[
        df["family"].isna()
    ]

    if len(unknown_rows) > 0:

        unknown_types = (
            unknown_rows["flaw_type"]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unknown flaw types found:\n"
            f"{unknown_types}"
        )

    # --------------------------------------------------------
    # Family distribution
    # --------------------------------------------------------

    log("")
    log("Family distribution:")
    log("-" * 50)

    family_counts = (
        df["family"]
        .value_counts()
    )

    for family, count in family_counts.items():

        log(
            f"  {family:15s} {count:,}"
        )

    # --------------------------------------------------------
    # Flaw distribution
    # --------------------------------------------------------

    log("")
    log("Flaw distribution:")
    log("-" * 50)

    flaw_counts = (
        df["flaw_type"]
        .value_counts()
    )

    for flaw, count in flaw_counts.items():

        log(
            f"  {flaw:20s} {count:,}"
        )

    # --------------------------------------------------------
    # Find model features
    # --------------------------------------------------------

    model_features = get_model_features(
        df
    )

    if len(model_features) == 0:

        raise ValueError(
            "No numeric model features found."
        )

    log("")
    log(
        f"Model features: {len(model_features):,}"
    )

    # --------------------------------------------------------
    # Explicit leakage protection
    # --------------------------------------------------------

    forbidden_features = [
        "temporal_label",
        "previous_temporal_label",
        "next_temporal_label",
    ]

    for feature in model_features:

        lower = feature.lower()

        for forbidden in forbidden_features:

            if forbidden in lower:

                raise ValueError(
                    "LABEL LEAKAGE DETECTED:\n"
                    f"Feature: {feature}"
                )

    # --------------------------------------------------------
    # Speaker-safe split
    # --------------------------------------------------------

    log("")
    log(
        "Creating speaker-safe train/validation/test split..."
    )

    (
        train_df,
        val_df,
        test_df,
        train_speakers,
        val_speakers,
        test_speakers,
    ) = speaker_safe_split(
        df,
        train_fraction=0.60,
        val_fraction=0.20,
        random_state=RANDOM_STATE,
    )

    log("")
    log("Speaker split:")
    log("-" * 50)

    log(
        f"Train speakers: {len(train_speakers)}"
    )

    log(
        f"Validation speakers: {len(val_speakers)}"
    )

    log(
        f"Test speakers: {len(test_speakers)}"
    )

    log("")
    log("Window split:")
    log("-" * 50)

    log(
        f"Train windows: {len(train_df):,}"
    )

    log(
        f"Validation windows: {len(val_df):,}"
    )

    log(
        f"Test windows: {len(test_df):,}"
    )

    # --------------------------------------------------------
    # Verify speaker isolation
    # --------------------------------------------------------

    train_speaker_set = set(
        train_df["speaker_id"]
        .astype(str)
    )

    val_speaker_set = set(
        val_df["speaker_id"]
        .astype(str)
    )

    test_speaker_set = set(
        test_df["speaker_id"]
        .astype(str)
    )

    assert (
        train_speaker_set
        .isdisjoint(val_speaker_set)
    )

    assert (
        train_speaker_set
        .isdisjoint(test_speaker_set)
    )

    assert (
        val_speaker_set
        .isdisjoint(test_speaker_set)
    )

    log(
        "Speaker leakage check: PASS"
    )

    # --------------------------------------------------------
    # Prepare datasets
    # --------------------------------------------------------

    X_train, y_train = prepare_xy(
        train_df,
        model_features
    )

    X_val, y_val = prepare_xy(
        val_df,
        model_features
    )

    X_test, y_test = prepare_xy(
        test_df,
        model_features
    )

    # --------------------------------------------------------
    # Encode labels
    # --------------------------------------------------------

    encoder = LabelEncoder()

    y_train_encoded = encoder.fit_transform(
        y_train
    )

    y_val_encoded = encoder.transform(
        y_val
    )

    y_test_encoded = encoder.transform(
        y_test
    )

    class_names = encoder.classes_

    log("")
    log(
        f"Classes: {list(class_names)}"
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    log("")
    log("=" * 70)
    log("TRAINING MODELS")
    log("=" * 70)

    models = {

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }

    results = {}

    best_model = None
    best_model_name = None
    best_val_f1 = -1.0

    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    for model_name, model in models.items():

        log("")
        log("=" * 60)
        log(
            f"Training: {model_name}"
        )
        log("=" * 60)

        model.fit(
            X_train,
            y_train_encoded
        )

        log(
            f"{model_name} training complete."
        )

        # ----------------------------------------------------
        # Validation prediction
        # ----------------------------------------------------

        val_predictions = model.predict(
            X_val
        )

        val_accuracy = accuracy_score(
            y_val_encoded,
            val_predictions
        )

        val_f1 = f1_score(
            y_val_encoded,
            val_predictions,
            average="macro",
            zero_division=0,
        )

        val_precision = precision_score(
            y_val_encoded,
            val_predictions,
            average="macro",
            zero_division=0,
        )

        val_recall = recall_score(
            y_val_encoded,
            val_predictions,
            average="macro",
            zero_division=0,
        )

        log(
            f"Validation accuracy: "
            f"{val_accuracy:.4f}"
        )

        log(
            f"Validation precision: "
            f"{val_precision:.4f}"
        )

        log(
            f"Validation recall: "
            f"{val_recall:.4f}"
        )

        log(
            f"Validation macro F1: "
            f"{val_f1:.4f}"
        )

        results[model_name] = {
            "validation_accuracy": float(
                val_accuracy
            ),
            "validation_precision": float(
                val_precision
            ),
            "validation_recall": float(
                val_recall
            ),
            "validation_macro_f1": float(
                val_f1
            ),
        }

        # ----------------------------------------------------
        # Select best model
        # ----------------------------------------------------

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            best_model = model

            best_model_name = model_name

            log(
                f"NEW BEST MODEL: {model_name}"
            )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    log("")
    log("=" * 70)
    log("BEST MODEL")
    log("=" * 70)

    log(
        f"Model: {best_model_name}"
    )

    log(
        f"Validation Macro F1: "
        f"{best_val_f1:.4f}"
    )

    # --------------------------------------------------------
    # Validation report
    # --------------------------------------------------------

    val_predictions = best_model.predict(
        X_val
    )

    val_report = classification_report(
        y_val_encoded,
        val_predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    log("")
    log("=" * 70)
    log("FINAL TEST EVALUATION")
    log("=" * 70)

    test_predictions = best_model.predict(
        X_test
    )

    test_accuracy = accuracy_score(
        y_test_encoded,
        test_predictions
    )

    test_precision = precision_score(
        y_test_encoded,
        test_predictions,
        average="macro",
        zero_division=0,
    )

    test_recall = recall_score(
        y_test_encoded,
        test_predictions,
        average="macro",
        zero_division=0,
    )

    test_f1 = f1_score(
        y_test_encoded,
        test_predictions,
        average="macro",
        zero_division=0,
    )

    test_weighted_f1 = f1_score(
        y_test_encoded,
        test_predictions,
        average="weighted",
        zero_division=0,
    )

    log("")
    log(
        f"Test accuracy:       {test_accuracy:.4f}"
    )

    log(
        f"Test macro precision: "
        f"{test_precision:.4f}"
    )

    log(
        f"Test macro recall:    "
        f"{test_recall:.4f}"
    )

    log(
        f"Test macro F1:        "
        f"{test_f1:.4f}"
    )

    log(
        f"Test weighted F1:     "
        f"{test_weighted_f1:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    test_report = classification_report(
        y_test_encoded,
        test_predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    print("")
    print(
        classification_report(
            y_test_encoded,
            test_predictions,
            target_names=class_names,
            zero_division=0,
        ),
        flush=True
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test_encoded,
        test_predictions,
    )

    log("")
    log("Confusion matrix:")

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"actual_{x}"
            for x in class_names
        ],
        columns=[
            f"predicted_{x}"
            for x in class_names
        ],
    )

    print(
        cm_df.to_string(),
        flush=True
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    cm_df.to_csv(
        CONFUSION_FILE
    )

    # --------------------------------------------------------
    # Prediction dataframe
    # --------------------------------------------------------

    prediction_df = test_df[
        [
            col
            for col in [
                "record_id",
                "pair_id",
                "speaker_id",
                "window_start",
                "window_end",
                "relative_position",
                "flaw_type",
                "severity",
            ]
            if col in test_df.columns
        ]
    ].copy()

    prediction_df[
        "actual_family"
    ] = y_test.values

    prediction_df[
        "predicted_family"
    ] = encoder.inverse_transform(
        test_predictions
    )

    if hasattr(
        best_model,
        "predict_proba"
    ):

        probabilities = (
            best_model.predict_proba(
                X_test
            )
        )

        for index, class_name in enumerate(
            class_names
        ):

            prediction_df[
                f"prob_{class_name}"
            ] = probabilities[:, index]

        prediction_df[
            "confidence"
        ] = probabilities.max(
            axis=1
        )

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    feature_importance = {}

    if hasattr(
        best_model,
        "feature_importances_"
    ):

        importance_values = (
            best_model.feature_importances_
        )

        feature_pairs = list(
            zip(
                model_features,
                importance_values
            )
        )

        feature_pairs.sort(
            key=lambda x: x[1],
            reverse=True
        )

        feature_importance = {
            feature: float(value)
            for feature, value
            in feature_pairs[:30]
        }

        log("")
        log(
            "Top 30 feature importances:"
        )

        for feature, value in feature_pairs[:30]:

            log(
                f"  {feature:45s} "
                f"{value:.6f}"
            )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_package = {
        "model": best_model,
        "label_encoder": encoder,
        "model_features": model_features,
        "family_mapping": {
            "temporal": sorted(
                TEMPORAL_FLAWS
            ),
            "acoustic": sorted(
                ACOUSTIC_FLAWS
            ),
        },
        "random_state": RANDOM_STATE,
        "version": "V2.1",
        "model_name": best_model_name,
    }

    joblib.dump(
        model_package,
        BEST_MODEL_FILE
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report = {
        "version": "V2.1",

        "model_name": best_model_name,

        "input_file": INPUT_FILE,

        "rows_total": int(len(df)),

        "rows_train": int(len(train_df)),

        "rows_validation": int(len(val_df)),

        "rows_test": int(len(test_df)),

        "train_speakers": int(
            len(train_speakers)
        ),

        "validation_speakers": int(
            len(val_speakers)
        ),

        "test_speakers": int(
            len(test_speakers)
        ),

        "model_features": int(
            len(model_features)
        ),

        "family_distribution": {
            str(k): int(v)
            for k, v
            in family_counts.items()
        },

        "flaw_distribution": {
            str(k): int(v)
            for k, v
            in flaw_counts.items()
        },

        "validation_results": results,

        "best_validation_macro_f1": float(
            best_val_f1
        ),

        "validation_classification_report":
            val_report,

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

        "test_weighted_f1": float(
            test_weighted_f1
        ),

        "test_classification_report":
            test_report,

        "confusion_matrix":
            cm.tolist(),

        "classes":
            class_names.tolist(),

        "top_feature_importance":
            feature_importance,
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("")
    print("=" * 70)
    print("V2.1 FLAW FAMILY TRAINING COMPLETE")
    print("=" * 70)

    print("")
    print(
        f"Best model: "
        f"{best_model_name}"
    )

    print(
        f"Validation Macro F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test Macro F1: "
        f"{test_f1:.4f}"
    )

    print("")
    print("Saved model:")
    print(
        f"  {BEST_MODEL_FILE}"
    )

    print("")
    print("Saved report:")
    print(
        f"  {REPORT_FILE}"
    )

    print("")
    print("Saved confusion matrix:")
    print(
        f"  {CONFUSION_FILE}"
    )

    print("")
    print("Saved predictions:")
    print(
        f"  {PREDICTIONS_FILE}"
    )

    print("")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()