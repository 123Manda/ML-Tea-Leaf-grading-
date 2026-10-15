from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# =========================================================
# CONFIGURATION
# =========================================================

CLASS_NAMES = [
    "Best",
    "Below_Best",
    "Poor",
]


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
)

TEST_FILE = (
    FEATURE_DIR
    / "test_features.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

MODEL_PATH = (
    MODEL_DIR
    / "xgboost_baseline.pkl"
)

LABEL_ENCODER_PATH = (
    MODEL_DIR
    / "xgboost_label_encoder.pkl"
)

FEATURE_COLUMNS_PATH = (
    MODEL_DIR
    / "xgboost_feature_columns.json"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# LOAD TEST DATA
# =========================================================

def load_test_data():

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test feature file not found:\n{TEST_FILE}"
        )

    test_df = pd.read_csv(
        TEST_FILE
    )

    required_columns = {
        "file_name",
        "class",
        "split",
    }

    missing_columns = (
        required_columns
        - set(test_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return test_df


# =========================================================
# LOAD TRAINED COMPONENTS
# =========================================================

def load_model_components():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost model not found:\n{MODEL_PATH}"
        )

    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Label encoder not found:\n{LABEL_ENCODER_PATH}"
        )

    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Feature column file not found:\n"
            f"{FEATURE_COLUMNS_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    label_encoder = joblib.load(
        LABEL_ENCODER_PATH
    )

    with open(
        FEATURE_COLUMNS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        feature_columns = json.load(
            file
        )

    return (
        model,
        label_encoder,
        feature_columns,
    )


# =========================================================
# PREPARE TEST FEATURES
# =========================================================

def prepare_test_features(
    test_df,
    feature_columns,
):

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in test_df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing feature columns in test dataset: "
            f"{missing_features}"
        )

    X_test = test_df[
        feature_columns
    ].copy()

    y_test = test_df[
        "class"
    ].copy()

    for column in feature_columns:

        X_test[column] = pd.to_numeric(
            X_test[column],
            errors="coerce",
        )

    X_test = X_test.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    invalid_values = int(
        X_test.isna().sum().sum()
    )

    if invalid_values > 0:

        raise ValueError(
            f"Test dataset contains "
            f"{invalid_values} invalid values."
        )

    return (
        X_test,
        y_test,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("FINAL XGBOOST TEST EVALUATION")
    print("=" * 70)

    test_df = load_test_data()

    (
        model,
        label_encoder,
        feature_columns,
    ) = load_model_components()

    (
        X_test,
        y_test,
    ) = prepare_test_features(
        test_df,
        feature_columns,
    )

    print()
    print("Test Dataset")
    print("-" * 45)

    print(
        f"Test samples  : {len(X_test)}"
    )

    print(
        f"Feature count : {len(feature_columns)}"
    )

    print()

    print(
        "Test class distribution:"
    )

    print(
        y_test.value_counts().to_string()
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    encoded_predictions = model.predict(
        X_test
    ).astype(
        int
    )

    predictions = label_encoder.inverse_transform(
        encoded_predictions
    )

    # -----------------------------------------------------
    # Prediction probabilities
    # -----------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )

    confidence_scores = np.max(
        probabilities,
        axis=1,
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy        : "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro Precision : "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall    : "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1-score  : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1     : "
        f"{weighted_f1:.4f}"
    )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    report = classification_report(
        y_test,
        predictions,
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    print()
    print("=" * 70)
    print("TEST CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        report_df.to_string()
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=CLASS_NAMES,
    )

    confusion_df = pd.DataFrame(
        cm,
        index=[
            f"Actual_{name}"
            for name in CLASS_NAMES
        ],
        columns=[
            f"Predicted_{name}"
            for name in CLASS_NAMES
        ],
    )

    print()
    print("=" * 70)
    print("TEST CONFUSION MATRIX")
    print("=" * 70)

    print(
        confusion_df.to_string()
    )

    # -----------------------------------------------------
    # Save final metrics
    # -----------------------------------------------------

    metrics = {
        "selected_model": "XGBoost",
        "test_samples": int(
            len(X_test)
        ),
        "feature_count": int(
            len(feature_columns)
        ),
        "test_accuracy": float(
            accuracy
        ),
        "test_macro_precision": float(
            macro_precision
        ),
        "test_macro_recall": float(
            macro_recall
        ),
        "test_macro_f1": float(
            macro_f1
        ),
        "test_weighted_f1": float(
            weighted_f1
        ),
    }

    metrics_path = (
        RESULTS_DIR
        / "xgboost_final_test_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Save classification report
    # -----------------------------------------------------

    report_path = (
        RESULTS_DIR
        / "xgboost_final_test_classification_report.csv"
    )

    report_df.to_csv(
        report_path
    )

    # -----------------------------------------------------
    # Save confusion matrix
    # -----------------------------------------------------

    confusion_path = (
        RESULTS_DIR
        / "xgboost_final_test_confusion_matrix.csv"
    )

    confusion_df.to_csv(
        confusion_path
    )

    # -----------------------------------------------------
    # Save individual predictions
    # -----------------------------------------------------

    predictions_df = pd.DataFrame(
        {
            "file_name":
                test_df[
                    "file_name"
                ].values,

            "actual_class":
                y_test.values,

            "predicted_class":
                predictions,

            "confidence":
                confidence_scores,
        }
    )

    predictions_df[
        "correct"
    ] = (
        predictions_df[
            "actual_class"
        ]
        ==
        predictions_df[
            "predicted_class"
        ]
    )

    prediction_path = (
        RESULTS_DIR
        / "xgboost_final_test_predictions.csv"
    )

    predictions_df.to_csv(
        prediction_path,
        index=False,
    )

    # -----------------------------------------------------
    # Save average confidence
    # -----------------------------------------------------

    correct_mask = (
        predictions_df[
            "correct"
        ]
    )

    correct_confidence = (
        predictions_df.loc[
            correct_mask,
            "confidence",
        ].mean()
    )

    incorrect_confidence = (
        predictions_df.loc[
            ~correct_mask,
            "confidence",
        ].mean()
    )

    print()
    print("=" * 70)
    print("CONFIDENCE SUMMARY")
    print("=" * 70)

    print(
        f"Average confidence "
        f"(correct predictions)   : "
        f"{correct_confidence:.4f}"
    )

    if np.isnan(
        incorrect_confidence
    ):

        print(
            "Average confidence "
            "(incorrect predictions) : "
            "No incorrect predictions"
        )

    else:

        print(
            f"Average confidence "
            f"(incorrect predictions) : "
            f"{incorrect_confidence:.4f}"
        )

    # -----------------------------------------------------
    # Final
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Final metrics:\n"
        f"{metrics_path}"
    )

    print()

    print(
        f"Classification report:\n"
        f"{report_path}"
    )

    print()

    print(
        f"Confusion matrix:\n"
        f"{confusion_path}"
    )

    print()

    print(
        f"Test predictions:\n"
        f"{prediction_path}"
    )

    print()

    print(
        "FINAL XGBoost test evaluation "
        "completed successfully."
    )


if __name__ == "__main__":
    main()