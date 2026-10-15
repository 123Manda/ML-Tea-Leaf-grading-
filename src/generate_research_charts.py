from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

CHARTS_DIR = (
    RESULTS_DIR
    / "charts"
)

CHARTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# FILE PATHS
# =========================================================

SVM_METRICS_FILE = (
    RESULTS_DIR
    / "svm_validation_metrics.json"
)

RF_METRICS_FILE = (
    RESULTS_DIR
    / "random_forest_validation_metrics.json"
)

XGB_METRICS_FILE = (
    RESULTS_DIR
    / "xgboost_validation_metrics.json"
)

ABLATION_FILE = (
    RESULTS_DIR
    / "xgboost_feature_ablation_results.csv"
)

TEST_CONFUSION_FILE = (
    RESULTS_DIR
    / "xgboost_final_test_confusion_matrix.csv"
)

TEST_CLASSIFICATION_FILE = (
    RESULTS_DIR
    / "xgboost_final_test_classification_report.csv"
)

SHAP_GLOBAL_FILE = (
    RESULTS_DIR
    / "shap_global_feature_importance.csv"
)


# =========================================================
# UTILITY
# =========================================================

def load_json(file_path):

    if not file_path.exists():

        raise FileNotFoundError(
            f"File not found:\n{file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def save_figure(
    file_name,
):

    output_path = (
        CHARTS_DIR
        / file_name
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# =========================================================
# 1. MODEL COMPARISON
# =========================================================

def create_model_comparison_chart():

    svm = load_json(
        SVM_METRICS_FILE
    )

    rf = load_json(
        RF_METRICS_FILE
    )

    xgb = load_json(
        XGB_METRICS_FILE
    )

    models = [
        "SVM",
        "Random Forest",
        "XGBoost",
    ]

    accuracies = [
        svm[
            "validation_accuracy"
        ] * 100,

        rf[
            "validation_accuracy"
        ] * 100,

        xgb[
            "validation_accuracy"
        ] * 100,
    ]

    macro_f1_scores = [
        svm[
            "validation_macro_f1"
        ] * 100,

        rf[
            "validation_macro_f1"
        ] * 100,

        xgb[
            "validation_macro_f1"
        ] * 100,
    ]

    x = np.arange(
        len(models)
    )

    width = 0.35

    plt.figure(
        figsize=(
            9,
            6,
        )
    )

    bars_accuracy = plt.bar(
        x - width / 2,
        accuracies,
        width,
        label="Accuracy",
    )

    bars_f1 = plt.bar(
        x + width / 2,
        macro_f1_scores,
        width,
        label="Macro F1",
    )

    plt.title(
        "Validation Performance Comparison of ML Models",
        fontsize=14,
    )

    plt.xlabel(
        "Machine Learning Model"
    )

    plt.ylabel(
        "Performance (%)"
    )

    plt.xticks(
        x,
        models,
    )

    plt.ylim(
        0,
        100,
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    for bars in [
        bars_accuracy,
        bars_f1,
    ]:

        for bar in bars:

            height = bar.get_height()

            plt.text(
                bar.get_x()
                + bar.get_width()
                / 2,

                height + 0.8,

                f"{height:.2f}",

                ha="center",
                va="bottom",
                fontsize=9,
            )

    save_figure(
        "01_model_comparison.png"
    )


# =========================================================
# 2. FEATURE ABLATION STUDY
# =========================================================

def create_ablation_chart():

    if not ABLATION_FILE.exists():

        raise FileNotFoundError(
            f"Ablation file not found:\n"
            f"{ABLATION_FILE}"
        )

    df = pd.read_csv(
        ABLATION_FILE
    )

    df = df.sort_values(
        by="accuracy_percent",
        ascending=True,
    )

    plt.figure(
        figsize=(
            10,
            7,
        )
    )

    bars = plt.barh(
        df[
            "feature_set"
        ],
        df[
            "accuracy_percent"
        ],
    )

    plt.title(
        "XGBoost Feature Ablation Study",
        fontsize=14,
    )

    plt.xlabel(
        "Validation Accuracy (%)"
    )

    plt.ylabel(
        "Feature Set"
    )

    plt.xlim(
        0,
        100,
    )

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    for bar in bars:

        width = bar.get_width()

        plt.text(
            width + 0.6,
            bar.get_y()
            + bar.get_height()
            / 2,

            f"{width:.2f}%",

            va="center",
            fontsize=9,
        )

    save_figure(
        "02_feature_ablation.png"
    )


# =========================================================
# 3. FINAL TEST CONFUSION MATRIX
# =========================================================

def create_confusion_matrix_chart():

    if not TEST_CONFUSION_FILE.exists():

        raise FileNotFoundError(
            f"Confusion matrix not found:\n"
            f"{TEST_CONFUSION_FILE}"
        )

    df = pd.read_csv(
        TEST_CONFUSION_FILE,
        index_col=0,
    )

    matrix = df.values

    row_labels = [
        label.replace(
            "Actual_",
            "",
        ).replace(
            "_",
            " ",
        )
        for label in df.index
    ]

    column_labels = [
        label.replace(
            "Predicted_",
            "",
        ).replace(
            "_",
            " ",
        )
        for label in df.columns
    ]

    plt.figure(
        figsize=(
            8,
            7,
        )
    )

    image = plt.imshow(
        matrix
    )

    plt.colorbar(
        image,
        label="Number of Images",
    )

    plt.title(
        "XGBoost Final Test Confusion Matrix",
        fontsize=14,
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )

    plt.xticks(
        np.arange(
            len(
                column_labels
            )
        ),
        column_labels,
    )

    plt.yticks(
        np.arange(
            len(
                row_labels
            )
        ),
        row_labels,
    )

    threshold = (
        matrix.max()
        / 2.0
    )

    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            value = matrix[
                i,
                j,
            ]

            text_color = (
                "white"
                if value > threshold
                else "black"
            )

            plt.text(
                j,
                i,
                str(
                    value
                ),
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
            )

    save_figure(
        "03_final_test_confusion_matrix.png"
    )


# =========================================================
# 4. SHAP GLOBAL FEATURE IMPORTANCE
# =========================================================

def create_shap_chart():

    if not SHAP_GLOBAL_FILE.exists():

        raise FileNotFoundError(
            f"SHAP file not found:\n"
            f"{SHAP_GLOBAL_FILE}"
        )

    df = pd.read_csv(
        SHAP_GLOBAL_FILE
    )

    top_features = (
        df
        .head(
            15
        )
        .sort_values(
            by="mean_abs_shap",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(
            10,
            8,
        )
    )

    bars = plt.barh(
        top_features[
            "feature"
        ],
        top_features[
            "mean_abs_shap"
        ],
    )

    plt.title(
        "Top 15 Global Features Based on SHAP Importance",
        fontsize=14,
    )

    plt.xlabel(
        "Mean Absolute SHAP Value"
    )

    plt.ylabel(
        "Feature"
    )

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    for bar in bars:

        width = bar.get_width()

        plt.text(
            width + 0.01,
            bar.get_y()
            + bar.get_height()
            / 2,

            f"{width:.3f}",

            va="center",
            fontsize=8,
        )

    save_figure(
        "04_shap_global_feature_importance.png"
    )


# =========================================================
# 5. FINAL TEST CLASS-WISE F1 SCORE
# =========================================================

def create_class_f1_chart():

    if not TEST_CLASSIFICATION_FILE.exists():

        raise FileNotFoundError(
            f"Classification report not found:\n"
            f"{TEST_CLASSIFICATION_FILE}"
        )

    df = pd.read_csv(
        TEST_CLASSIFICATION_FILE,
        index_col=0,
    )

    classes = [
        "Best",
        "Below_Best",
        "Poor",
    ]

    f1_scores = [
        float(
            df.loc[
                class_name,
                "f1-score",
            ]
        ) * 100
        for class_name in classes
    ]

    display_classes = [
        name.replace(
            "_",
            " ",
        )
        for name in classes
    ]

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    bars = plt.bar(
        display_classes,
        f1_scores,
    )

    plt.title(
        "XGBoost Final Test F1-Score by Tea Leaf Quality Class",
        fontsize=14,
    )

    plt.xlabel(
        "Tea Leaf Quality Class"
    )

    plt.ylabel(
        "F1-Score (%)"
    )

    plt.ylim(
        0,
        100,
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x()
            + bar.get_width()
            / 2,

            height + 1,

            f"{height:.2f}%",

            ha="center",
            va="bottom",
            fontsize=10,
        )

    save_figure(
        "05_test_class_f1_scores.png"
    )


# =========================================================
# 6. MODEL COMPARISON TABLE CSV
# =========================================================

def create_model_comparison_table():

    svm = load_json(
        SVM_METRICS_FILE
    )

    rf = load_json(
        RF_METRICS_FILE
    )

    xgb = load_json(
        XGB_METRICS_FILE
    )

    rows = [
        {
            "Model":
                "SVM",

            "Validation Accuracy (%)":
                svm[
                    "validation_accuracy"
                ] * 100,

            "Macro Precision":
                svm[
                    "validation_macro_precision"
                ],

            "Macro Recall":
                svm[
                    "validation_macro_recall"
                ],

            "Macro F1":
                svm[
                    "validation_macro_f1"
                ],
        },

        {
            "Model":
                "Random Forest",

            "Validation Accuracy (%)":
                rf[
                    "validation_accuracy"
                ] * 100,

            "Macro Precision":
                rf[
                    "validation_macro_precision"
                ],

            "Macro Recall":
                rf[
                    "validation_macro_recall"
                ],

            "Macro F1":
                rf[
                    "validation_macro_f1"
                ],
        },

        {
            "Model":
                "XGBoost",

            "Validation Accuracy (%)":
                xgb[
                    "validation_accuracy"
                ] * 100,

            "Macro Precision":
                xgb[
                    "validation_macro_precision"
                ],

            "Macro Recall":
                xgb[
                    "validation_macro_recall"
                ],

            "Macro F1":
                xgb[
                    "validation_macro_f1"
                ],
        },
    ]

    comparison_df = pd.DataFrame(
        rows
    )

    comparison_path = (
        RESULTS_DIR
        / "final_model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    print(
        f"Saved: {comparison_path}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)

    print(
        "GENERATING RESEARCH RESULT CHARTS"
    )

    print("=" * 70)

    print()

    create_model_comparison_chart()

    create_ablation_chart()

    create_confusion_matrix_chart()

    create_shap_chart()

    create_class_f1_chart()

    create_model_comparison_table()

    print()

    print("=" * 70)

    print(
        "CHART GENERATION COMPLETED"
    )

    print("=" * 70)

    print()

    print(
        "Charts saved to:"
    )

    print(
        CHARTS_DIR
    )

    print()

    print(
        "Generated charts:"
    )

    print(
        "1. Model comparison"
    )

    print(
        "2. Feature ablation study"
    )

    print(
        "3. Final test confusion matrix"
    )

    print(
        "4. SHAP global feature importance"
    )

    print(
        "5. Final test class-wise F1 scores"
    )

    print()

    print(
        "All figures were exported at 300 DPI."
    )


if __name__ == "__main__":
    main()