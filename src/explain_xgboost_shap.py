from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import shap


# =========================================================
# CONFIGURATION
# =========================================================

TOP_GLOBAL_FEATURES = 15
TOP_LOCAL_FEATURES = 5


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

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

TEST_FEATURE_FILE = (
    FEATURE_DIR
    / "test_features.csv"
)

MODEL_FILE = (
    MODEL_DIR
    / "xgboost_baseline.pkl"
)

LABEL_ENCODER_FILE = (
    MODEL_DIR
    / "xgboost_label_encoder.pkl"
)

FEATURE_COLUMNS_FILE = (
    MODEL_DIR
    / "xgboost_feature_columns.json"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# LOAD FILES
# =========================================================

def load_components():

    if not TEST_FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Test feature file not found:\n"
            f"{TEST_FEATURE_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"XGBoost model not found:\n"
            f"{MODEL_FILE}"
        )

    if not LABEL_ENCODER_FILE.exists():
        raise FileNotFoundError(
            f"Label encoder not found:\n"
            f"{LABEL_ENCODER_FILE}"
        )

    if not FEATURE_COLUMNS_FILE.exists():
        raise FileNotFoundError(
            f"Feature column file not found:\n"
            f"{FEATURE_COLUMNS_FILE}"
        )

    test_df = pd.read_csv(
        TEST_FEATURE_FILE
    )

    model = joblib.load(
        MODEL_FILE
    )

    label_encoder = joblib.load(
        LABEL_ENCODER_FILE
    )

    with open(
        FEATURE_COLUMNS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        feature_columns = json.load(
            file
        )

    return (
        test_df,
        model,
        label_encoder,
        feature_columns,
    )


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_test_data(
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
            "Missing feature columns: "
            f"{missing_features}"
        )

    X_test = test_df[
        feature_columns
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

    invalid_count = int(
        X_test.isna().sum().sum()
    )

    if invalid_count > 0:

        raise ValueError(
            f"Test feature data contains "
            f"{invalid_count} invalid values."
        )

    return X_test


# =========================================================
# NORMALIZE SHAP ARRAY FORMAT
# =========================================================

def normalize_shap_values(
    shap_output,
    n_samples,
    n_features,
    n_classes,
):
    """
    Convert SHAP output into standard shape:

        samples x features x classes

    Different SHAP versions may return:
        - Explanation
        - numpy array
        - list of arrays
    """

    if hasattr(
        shap_output,
        "values"
    ):

        values = shap_output.values

    else:

        values = shap_output

    # -----------------------------------------------------
    # Older SHAP format:
    # list[class] -> (samples, features)
    # -----------------------------------------------------

    if isinstance(
        values,
        list
    ):

        if len(values) != n_classes:

            raise ValueError(
                "Unexpected SHAP class count."
            )

        stacked = np.stack(
            values,
            axis=2,
        )

        return stacked

    values = np.asarray(
        values
    )

    # -----------------------------------------------------
    # Expected new format
    # samples x features x classes
    # -----------------------------------------------------

    if (
        values.ndim == 3
        and values.shape[0] == n_samples
        and values.shape[1] == n_features
        and values.shape[2] == n_classes
    ):

        return values

    # -----------------------------------------------------
    # Alternative:
    # classes x samples x features
    # -----------------------------------------------------

    if (
        values.ndim == 3
        and values.shape[0] == n_classes
        and values.shape[1] == n_samples
        and values.shape[2] == n_features
    ):

        return np.transpose(
            values,
            (
                1,
                2,
                0,
            ),
        )

    # -----------------------------------------------------
    # Alternative:
    # samples x classes x features
    # -----------------------------------------------------

    if (
        values.ndim == 3
        and values.shape[0] == n_samples
        and values.shape[1] == n_classes
        and values.shape[2] == n_features
    ):

        return np.transpose(
            values,
            (
                0,
                2,
                1,
            ),
        )

    raise ValueError(
        "Unexpected SHAP value shape: "
        f"{values.shape}"
    )


# =========================================================
# GLOBAL IMPORTANCE
# =========================================================

def calculate_global_importance(
    shap_values,
    feature_columns,
):

    # Mean absolute SHAP value across:
    # all samples and all output classes
    global_importance = np.mean(
        np.abs(
            shap_values
        ),
        axis=(
            0,
            2,
        ),
    )

    global_df = pd.DataFrame(
        {
            "feature":
                feature_columns,

            "mean_abs_shap":
                global_importance,
        }
    )

    global_df = (
        global_df
        .sort_values(
            by="mean_abs_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return global_df


# =========================================================
# CLASS-WISE IMPORTANCE
# =========================================================

def calculate_class_importance(
    shap_values,
    feature_columns,
    class_names,
):

    rows = []

    for class_index, class_name in enumerate(
        class_names
    ):

        class_values = (
            shap_values[
                :,
                :,
                class_index
            ]
        )

        importance = np.mean(
            np.abs(
                class_values
            ),
            axis=0,
        )

        for feature, value in zip(
            feature_columns,
            importance,
        ):

            rows.append(
                {
                    "class":
                        class_name,

                    "feature":
                        feature,

                    "mean_abs_shap":
                        float(
                            value
                        ),
                }
            )

    class_df = pd.DataFrame(
        rows
    )

    class_df = (
        class_df
        .sort_values(
            by=[
                "class",
                "mean_abs_shap",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return class_df


# =========================================================
# LOCAL EXPLANATIONS
# =========================================================

def create_local_explanations(
    test_df,
    X_test,
    model,
    label_encoder,
    shap_values,
    feature_columns,
):

    encoded_predictions = model.predict(
        X_test
    ).astype(
        int
    )

    predicted_classes = (
        label_encoder.inverse_transform(
            encoded_predictions
        )
    )

    probabilities = model.predict_proba(
        X_test
    )

    rows = []

    summary_rows = []

    for sample_index in range(
        len(X_test)
    ):

        predicted_index = int(
            encoded_predictions[
                sample_index
            ]
        )

        predicted_class = (
            predicted_classes[
                sample_index
            ]
        )

        actual_class = (
            test_df.iloc[
                sample_index
            ]["class"]
        )

        file_name = (
            test_df.iloc[
                sample_index
            ]["file_name"]
        )

        confidence = float(
            probabilities[
                sample_index,
                predicted_index
            ]
        )

        sample_shap = (
            shap_values[
                sample_index,
                :,
                predicted_index
            ]
        )

        feature_values = (
            X_test.iloc[
                sample_index
            ].values
        )

        local_df = pd.DataFrame(
            {
                "feature":
                    feature_columns,

                "feature_value":
                    feature_values,

                "shap_value":
                    sample_shap,
            }
        )

        local_df[
            "abs_shap"
        ] = np.abs(
            local_df[
                "shap_value"
            ]
        )

        local_df = (
            local_df
            .sort_values(
                by="abs_shap",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        top_local = local_df.head(
            TOP_LOCAL_FEATURES
        )

        for rank, row in enumerate(
            top_local.itertuples(
                index=False
            ),
            start=1,
        ):

            direction = (
                "supports_predicted_class"
                if row.shap_value > 0
                else
                "opposes_predicted_class"
            )

            rows.append(
                {
                    "file_name":
                        file_name,

                    "actual_class":
                        actual_class,

                    "predicted_class":
                        predicted_class,

                    "confidence":
                        confidence,

                    "correct":
                        actual_class
                        == predicted_class,

                    "rank":
                        rank,

                    "feature":
                        row.feature,

                    "feature_value":
                        float(
                            row.feature_value
                        ),

                    "shap_value":
                        float(
                            row.shap_value
                        ),

                    "direction":
                        direction,
                }
            )

        top_feature_names = ", ".join(
            top_local[
                "feature"
            ].tolist()
        )

        summary_rows.append(
            {
                "file_name":
                    file_name,

                "actual_class":
                    actual_class,

                "predicted_class":
                    predicted_class,

                "confidence":
                    confidence,

                "correct":
                    actual_class
                    == predicted_class,

                "top_5_features":
                    top_feature_names,
            }
        )

    local_long_df = pd.DataFrame(
        rows
    )

    local_summary_df = pd.DataFrame(
        summary_rows
    )

    return (
        local_long_df,
        local_summary_df,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("XGBOOST SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 70)

    (
        test_df,
        model,
        label_encoder,
        feature_columns,
    ) = load_components()

    X_test = prepare_test_data(
        test_df,
        feature_columns,
    )

    class_names = list(
        label_encoder.classes_
    )

    print()
    print("Dataset")
    print("-" * 45)

    print(
        f"Test samples : "
        f"{len(X_test)}"
    )

    print(
        f"Features     : "
        f"{len(feature_columns)}"
    )

    print(
        f"Classes      : "
        f"{class_names}"
    )

    # -----------------------------------------------------
    # SHAP EXPLAINER
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("CALCULATING SHAP VALUES")
    print("=" * 70)

    explainer = shap.TreeExplainer(
        model
    )

    raw_shap_output = explainer(
        X_test
    )

    shap_values = normalize_shap_values(
        raw_shap_output,
        n_samples=len(
            X_test
        ),
        n_features=len(
            feature_columns
        ),
        n_classes=len(
            class_names
        ),
    )

    print()
    print(
        "SHAP values calculated successfully."
    )

    print(
        f"Normalized SHAP shape: "
        f"{shap_values.shape}"
    )

    # -----------------------------------------------------
    # GLOBAL IMPORTANCE
    # -----------------------------------------------------

    global_df = calculate_global_importance(
        shap_values,
        feature_columns,
    )

    global_path = (
        RESULTS_DIR
        / "shap_global_feature_importance.csv"
    )

    global_df.to_csv(
        global_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("TOP GLOBAL SHAP FEATURES")
    print("=" * 70)

    print(
        global_df
        .head(
            TOP_GLOBAL_FEATURES
        )
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # CLASS-WISE IMPORTANCE
    # -----------------------------------------------------

    class_df = calculate_class_importance(
        shap_values,
        feature_columns,
        class_names,
    )

    class_path = (
        RESULTS_DIR
        / "shap_class_feature_importance.csv"
    )

    class_df.to_csv(
        class_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("TOP 5 FEATURES PER CLASS")
    print("=" * 70)

    for class_name in class_names:

        print()
        print(
            class_name
        )

        print(
            "-" * 40
        )

        top_class = (
            class_df[
                class_df[
                    "class"
                ]
                == class_name
            ]
            .head(
                5
            )
        )

        print(
            top_class[
                [
                    "feature",
                    "mean_abs_shap",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # -----------------------------------------------------
    # LOCAL EXPLANATIONS
    # -----------------------------------------------------

    (
        local_long_df,
        local_summary_df,
    ) = create_local_explanations(
        test_df,
        X_test,
        model,
        label_encoder,
        shap_values,
        feature_columns,
    )

    local_long_path = (
        RESULTS_DIR
        / "shap_local_explanations.csv"
    )

    local_summary_path = (
        RESULTS_DIR
        / "shap_prediction_explanation_summary.csv"
    )

    local_long_df.to_csv(
        local_long_path,
        index=False,
    )

    local_summary_df.to_csv(
        local_summary_path,
        index=False,
    )

    # -----------------------------------------------------
    # Save summary JSON
    # -----------------------------------------------------

    top_global_records = []

    for row in global_df.head(
        10
    ).itertuples(
        index=False
    ):

        top_global_records.append(
            {
                "feature":
                    row.feature,

                "mean_abs_shap":
                    float(
                        row.mean_abs_shap
                    ),
            }
        )

    top_features_by_class = {}

    for class_name in class_names:

        class_top = (
            class_df[
                class_df[
                    "class"
                ]
                == class_name
            ]
            .head(
                5
            )
        )

        top_features_by_class[
            class_name
        ] = [
            {
                "feature":
                    row.feature,

                "mean_abs_shap":
                    float(
                        row.mean_abs_shap
                    ),
            }
            for row in class_top.itertuples(
                index=False
            )
        ]

    summary = {
        "model":
            "XGBoost",

        "explanation_method":
            "SHAP TreeExplainer",

        "test_samples":
            int(
                len(X_test)
            ),

        "feature_count":
            int(
                len(feature_columns)
            ),

        "classes":
            class_names,

        "top_10_global_features":
            top_global_records,

        "top_5_features_by_class":
            top_features_by_class,
    }

    summary_path = (
        RESULTS_DIR
        / "shap_explanation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # FINAL OUTPUT
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Global SHAP importance:\n"
        f"{global_path}"
    )

    print()

    print(
        f"Class-wise SHAP importance:\n"
        f"{class_path}"
    )

    print()

    print(
        f"Local explanations:\n"
        f"{local_long_path}"
    )

    print()

    print(
        f"Prediction explanation summary:\n"
        f"{local_summary_path}"
    )

    print()

    print(
        f"SHAP JSON summary:\n"
        f"{summary_path}"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "Positive SHAP values support the analysed "
        "class score, while negative SHAP values "
        "oppose that class score."
    )

    print()

    print(
        "XGBoost SHAP analysis completed successfully."
    )


if __name__ == "__main__":
    main()