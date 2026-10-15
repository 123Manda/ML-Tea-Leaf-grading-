from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier


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

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# FEATURE GROUPS
# =========================================================

COLOR_FEATURES = [
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",

    "mean_h",
    "mean_s",
    "mean_v",
    "std_h",
    "std_s",
    "std_v",

    "mean_lab_l",
    "mean_lab_a",
    "mean_lab_b",
    "std_lab_l",
    "std_lab_a",
    "std_lab_b",

    "green_ratio",
    "yellow_brown_ratio",
    "dark_ratio",
    "bright_ratio",
]


SHAPE_SIZE_FEATURES = [
    "area_ratio",
    "perimeter_normalized",
    "bbox_width_ratio",
    "bbox_height_ratio",
    "aspect_ratio",
    "extent",
    "solidity",
    "circularity",
    "equivalent_diameter_normalized",
    "ellipse_axis_ratio",
]


TEXTURE_FEATURES = [
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    "gray_mean",
    "gray_std",
    "entropy",
    "laplacian_variance",
]


FEATURE_SETS = {
    "Color Only":
        COLOR_FEATURES,

    "Shape + Size Only":
        SHAPE_SIZE_FEATURES,

    "Texture Only":
        TEXTURE_FEATURES,

    "Color + Shape/Size":
        COLOR_FEATURES
        + SHAPE_SIZE_FEATURES,

    "Color + Texture":
        COLOR_FEATURES
        + TEXTURE_FEATURES,

    "Shape/Size + Texture":
        SHAPE_SIZE_FEATURES
        + TEXTURE_FEATURES,

    "All Features":
        COLOR_FEATURES
        + SHAPE_SIZE_FEATURES
        + TEXTURE_FEATURES,
}


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found:\n{TRAIN_FILE}"
        )

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found:\n"
            f"{VALIDATION_FILE}"
        )

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    return (
        train_df,
        validation_df,
    )


# =========================================================
# VERIFY FEATURES
# =========================================================

def verify_features(
    train_df,
    validation_df,
):

    all_required_features = set()

    for features in FEATURE_SETS.values():

        all_required_features.update(
            features
        )

    missing_train = (
        all_required_features
        - set(
            train_df.columns
        )
    )

    missing_validation = (
        all_required_features
        - set(
            validation_df.columns
        )
    )

    if missing_train:

        raise ValueError(
            "Training dataset is missing features: "
            f"{sorted(missing_train)}"
        )

    if missing_validation:

        raise ValueError(
            "Validation dataset is missing features: "
            f"{sorted(missing_validation)}"
        )


# =========================================================
# PREPARE FEATURE MATRIX
# =========================================================

def prepare_x(
    dataframe,
    feature_columns,
):

    X = dataframe[
        feature_columns
    ].copy()

    for column in feature_columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    invalid_count = int(
        X.isna().sum().sum()
    )

    if invalid_count > 0:

        raise ValueError(
            f"Found {invalid_count} invalid values "
            f"in selected feature set."
        )

    return X


# =========================================================
# CREATE XGBOOST MODEL
# =========================================================

def create_model():

    return XGBClassifier(
        objective="multi:softprob",

        num_class=len(
            CLASS_NAMES
        ),

        n_estimators=500,

        learning_rate=0.05,

        max_depth=5,

        min_child_weight=1,

        subsample=0.85,

        colsample_bytree=0.85,

        reg_alpha=0.0,

        reg_lambda=1.0,

        eval_metric="mlogloss",

        tree_method="hist",

        random_state=RANDOM_SEED,

        n_jobs=-1,
    )


# =========================================================
# EVALUATE ONE FEATURE SET
# =========================================================

def evaluate_feature_set(
    name,
    feature_columns,
    train_df,
    validation_df,
    label_encoder,
):

    print()
    print("=" * 70)

    print(
        f"ABLATION EXPERIMENT: {name}"
    )

    print("=" * 70)

    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )

    X_train = prepare_x(
        train_df,
        feature_columns,
    )

    X_validation = prepare_x(
        validation_df,
        feature_columns,
    )

    y_train_text = (
        train_df[
            "class"
        ]
    )

    y_validation_text = (
        validation_df[
            "class"
        ]
    )

    y_train = (
        label_encoder.transform(
            y_train_text
        )
    )

    model = create_model()

    model.fit(
        X_train,
        y_train,
    )

    encoded_predictions = (
        model.predict(
            X_validation
        )
        .astype(
            int
        )
    )

    predictions = (
        label_encoder.inverse_transform(
            encoded_predictions
        )
    )

    accuracy = accuracy_score(
        y_validation_text,
        predictions,
    )

    macro_precision = precision_score(
        y_validation_text,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_validation_text,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_validation_text,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_validation_text,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print()

    print(
        f"Accuracy       : "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro Precision: "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall   : "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1       : "
        f"{macro_f1:.4f}"
    )

    return {
        "feature_set":
            name,

        "feature_count":
            len(
                feature_columns
            ),

        "accuracy":
            float(
                accuracy
            ),

        "accuracy_percent":
            float(
                accuracy * 100
            ),

        "macro_precision":
            float(
                macro_precision
            ),

        "macro_recall":
            float(
                macro_recall
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "weighted_f1":
            float(
                weighted_f1
            ),
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)

    print(
        "XGBOOST FEATURE ABLATION STUDY"
    )

    print("=" * 70)

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Only training and validation datasets "
        "are used in this study."
    )

    print(
        "The final test dataset is not used "
        "for ablation or model tuning."
    )

    (
        train_df,
        validation_df,
    ) = load_data()

    verify_features(
        train_df,
        validation_df,
    )

    label_encoder = LabelEncoder()

    label_encoder.fit(
        CLASS_NAMES
    )

    results = []

    for (
        feature_set_name,
        feature_columns,
    ) in FEATURE_SETS.items():

        result = evaluate_feature_set(
            feature_set_name,
            feature_columns,
            train_df,
            validation_df,
            label_encoder,
        )

        results.append(
            result
        )

    # -----------------------------------------------------
    # Results table
    # -----------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            by="macro_f1",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    results_df[
        "rank"
    ] = (
        np.arange(
            1,
            len(
                results_df
            )
            + 1
        )
    )

    display_columns = [
        "rank",
        "feature_set",
        "feature_count",
        "accuracy_percent",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]

    results_df = results_df[
        display_columns
    ]

    print()
    print("=" * 70)

    print(
        "FINAL ABLATION STUDY RESULTS"
    )

    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    csv_path = (
        RESULTS_DIR
        / "xgboost_feature_ablation_results.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    # -----------------------------------------------------
    # Save JSON
    # -----------------------------------------------------

    json_path = (
        RESULTS_DIR
        / "xgboost_feature_ablation_results.json"
    )

    json_results = []

    for row in results_df.to_dict(
        orient="records"
    ):

        json_results.append(
            {
                key: (
                    value.item()
                    if isinstance(
                        value,
                        np.generic,
                    )
                    else value
                )
                for key, value in row.items()
            }
        )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_results,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Best feature set
    # -----------------------------------------------------

    best_row = (
        results_df.iloc[
            0
        ]
    )

    print()
    print("=" * 70)

    print(
        "BEST FEATURE SET"
    )

    print("=" * 70)

    print(
        f"Feature set : "
        f"{best_row['feature_set']}"
    )

    print(
        f"Features    : "
        f"{int(best_row['feature_count'])}"
    )

    print(
        f"Accuracy    : "
        f"{best_row['accuracy_percent']:.2f}%"
    )

    print(
        f"Macro F1    : "
        f"{best_row['macro_f1']:.4f}"
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        csv_path
    )

    print(
        json_path
    )

    print()

    print(
        "Feature ablation study completed successfully."
    )


if __name__ == "__main__":
    main()