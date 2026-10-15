from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

RANDOM_SEED = 42

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

TRAIN_FILE = (
    FEATURE_DIR
    / "train_features.csv"
)

VALIDATION_FILE = (
    FEATURE_DIR
    / "validation_features.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# LOAD DATA
# =========================================================

def load_dataset(file_path):

    if not file_path.exists():

        raise FileNotFoundError(
            f"Dataset file not found:\n{file_path}"
        )

    dataframe = pd.read_csv(
        file_path
    )

    required_columns = {
        "file_name",
        "class",
        "split",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    return dataframe


# =========================================================
# PREPARE FEATURES
# =========================================================

def prepare_xy(dataframe):

    metadata_columns = [
        "file_name",
        "class",
        "split",
    ]

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in metadata_columns
    ]

    X = dataframe[
        feature_columns
    ].copy()

    y = dataframe[
        "class"
    ].copy()

    # Ensure all features are numeric
    for column in feature_columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # Replace infinity with NaN
    X = X.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # Check missing values
    missing_values = int(
        X.isna().sum().sum()
    )

    if missing_values > 0:

        raise ValueError(
            f"Feature dataset contains "
            f"{missing_values} missing/invalid values."
        )

    return (
        X,
        y,
        feature_columns,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SVM BASELINE MODEL TRAINING")
    print("=" * 70)

    # -----------------------------------------------------
    # Load train and validation datasets
    # -----------------------------------------------------

    train_df = load_dataset(
        TRAIN_FILE
    )

    validation_df = load_dataset(
        VALIDATION_FILE
    )

    (
        X_train,
        y_train,
        feature_columns,
    ) = prepare_xy(
        train_df
    )

    (
        X_validation,
        y_validation,
        validation_feature_columns,
    ) = prepare_xy(
        validation_df
    )

    if (
        feature_columns
        != validation_feature_columns
    ):

        raise ValueError(
            "Train and validation feature columns "
            "do not match."
        )

    print()
    print("Dataset")
    print("-" * 45)

    print(
        f"Training samples   : "
        f"{len(X_train)}"
    )

    print(
        f"Validation samples : "
        f"{len(X_validation)}"
    )

    print(
        f"Feature count      : "
        f"{len(feature_columns)}"
    )

    print()

    print("Training class distribution:")
    print(
        y_train.value_counts().to_string()
    )

    print()

    print("Validation class distribution:")
    print(
        y_validation.value_counts().to_string()
    )

    # -----------------------------------------------------
    # SVM Pipeline
    # -----------------------------------------------------

    svm_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    print()
    print("=" * 70)
    print("TRAINING SVM")
    print("=" * 70)

    svm_pipeline.fit(
        X_train,
        y_train,
    )

    print()
    print(
        "SVM training completed."
    )

    # -----------------------------------------------------
    # Validation prediction
    # -----------------------------------------------------

    validation_predictions = (
        svm_pipeline.predict(
            X_validation
        )
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_validation,
        validation_predictions,
    )

    macro_precision = precision_score(
        y_validation,
        validation_predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_validation,
        validation_predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_validation,
        validation_predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_validation,
        validation_predictions,
        average="weighted",
        zero_division=0,
    )

    print()
    print("=" * 70)
    print("VALIDATION RESULTS")
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
        y_validation,
        validation_predictions,
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
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        report_df.to_string()
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        y_validation,
        validation_predictions,
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
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        confusion_df.to_string()
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    model_path = (
        MODEL_DIR
        / "svm_baseline.pkl"
    )

    joblib.dump(
        svm_pipeline,
        model_path,
    )

    # -----------------------------------------------------
    # Save feature names
    # -----------------------------------------------------

    feature_path = (
        MODEL_DIR
        / "svm_feature_columns.json"
    )

    with open(
        feature_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            feature_columns,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Save summary metrics
    # -----------------------------------------------------

    metrics = {
        "model": "SVM",
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "class_weight": "balanced",
        "random_seed": RANDOM_SEED,
        "training_samples": len(
            X_train
        ),
        "validation_samples": len(
            X_validation
        ),
        "feature_count": len(
            feature_columns
        ),
        "validation_accuracy": float(
            accuracy
        ),
        "validation_macro_precision": float(
            macro_precision
        ),
        "validation_macro_recall": float(
            macro_recall
        ),
        "validation_macro_f1": float(
            macro_f1
        ),
        "validation_weighted_f1": float(
            weighted_f1
        ),
    }

    metrics_path = (
        RESULTS_DIR
        / "svm_validation_metrics.json"
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

    classification_report_path = (
        RESULTS_DIR
        / "svm_validation_classification_report.csv"
    )

    report_df.to_csv(
        classification_report_path
    )

    # -----------------------------------------------------
    # Save confusion matrix
    # -----------------------------------------------------

    confusion_matrix_path = (
        RESULTS_DIR
        / "svm_validation_confusion_matrix.csv"
    )

    confusion_df.to_csv(
        confusion_matrix_path
    )

    # -----------------------------------------------------
    # Save individual predictions
    # -----------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "file_name":
                validation_df[
                    "file_name"
                ].values,
            "actual_class":
                y_validation.values,
            "predicted_class":
                validation_predictions,
        }
    )

    prediction_df[
        "correct"
    ] = (
        prediction_df[
            "actual_class"
        ]
        ==
        prediction_df[
            "predicted_class"
        ]
    )

    prediction_path = (
        RESULTS_DIR
        / "svm_validation_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Model:\n{model_path}"
    )

    print()

    print(
        f"Metrics:\n{metrics_path}"
    )

    print()

    print(
        f"Classification report:\n"
        f"{classification_report_path}"
    )

    print()

    print(
        f"Confusion matrix:\n"
        f"{confusion_matrix_path}"
    )

    print()

    print(
        f"Validation predictions:\n"
        f"{prediction_path}"
    )

    print()

    print(
        "IMPORTANT: Test dataset has NOT been used."
    )

    print()

    print(
        "SVM baseline experiment completed successfully."
    )


if __name__ == "__main__":
    main()