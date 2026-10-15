from pathlib import Path
import argparse
import json

import cv2
import joblib
import numpy as np

from extract_features import extract_features_from_image


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_baseline.pkl"
)

LABEL_ENCODER_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_label_encoder.pkl"
)

FEATURE_COLUMNS_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_feature_columns.json"
)


# =========================================================
# LOAD MODEL FILES
# =========================================================

def load_feature_columns():

    with open(
        FEATURE_COLUMNS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    # Support both:
    # ["feature1", "feature2", ...]
    # or
    # {"feature_columns": [...]}

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        if "feature_columns" in data:
            return data["feature_columns"]

    raise ValueError(
        "Could not read feature columns "
        "from xgboost_feature_columns.json"
    )


def load_prediction_assets():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(
            "XGBoost label encoder not found: "
            f"{LABEL_ENCODER_PATH}"
        )

    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            "Feature column file not found: "
            f"{FEATURE_COLUMNS_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    label_encoder = joblib.load(
        LABEL_ENCODER_PATH
    )

    feature_columns = (
        load_feature_columns()
    )

    return (
        model,
        label_encoder,
        feature_columns,
    )


# =========================================================
# LEAF MEASUREMENTS
# =========================================================

def calculate_leaf_measurements(
    contour,
):

    area_px2 = float(
        cv2.contourArea(
            contour
        )
    )

    perimeter_px = float(
        cv2.arcLength(
            contour,
            True,
        )
    )

    rect = cv2.minAreaRect(
        contour
    )

    width, height = rect[1]

    width = float(width)
    height = float(height)

    leaf_length_px = max(
        width,
        height,
    )

    leaf_width_px = min(
        width,
        height,
    )

    if leaf_width_px > 0:

        aspect_ratio = (
            leaf_length_px
            / leaf_width_px
        )

    else:

        aspect_ratio = 0.0

    return {
        "length_px": round(
            leaf_length_px,
            2,
        ),
        "width_px": round(
            leaf_width_px,
            2,
        ),
        "area_px2": round(
            area_px2,
            2,
        ),
        "perimeter_px": round(
            perimeter_px,
            2,
        ),
        "aspect_ratio": round(
            aspect_ratio,
            4,
        ),
    }


# =========================================================
# PREPARE MODEL INPUT
# =========================================================

def prepare_feature_vector(
    features,
    feature_columns,
):

    missing_features = [
        feature_name
        for feature_name
        in feature_columns
        if feature_name
        not in features
    ]

    if missing_features:

        raise ValueError(
            "Missing features required "
            f"by XGBoost: {missing_features}"
        )

    feature_vector = [
        float(
            features[
                feature_name
            ]
        )
        for feature_name
        in feature_columns
    ]

    feature_array = np.array(
        [feature_vector],
        dtype=np.float64,
    )

    return feature_array


# =========================================================
# SINGLE IMAGE PREDICTION
# =========================================================

def predict_single_image(
    image_path,
):

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    (
        model,
        label_encoder,
        feature_columns,
    ) = load_prediction_assets()

    (
        features,
        processed_image,
        contour,
        segmentation_method,
    ) = extract_features_from_image(
        image_path
    )

    feature_array = (
        prepare_feature_vector(
            features,
            feature_columns,
        )
    )

    predicted_encoded = (
        model.predict(
            feature_array
        )[0]
    )

    predicted_grade = (
        label_encoder
        .inverse_transform(
            [
                int(
                    predicted_encoded
                )
            ]
        )[0]
    )

    probability_values = (
        model.predict_proba(
            feature_array
        )[0]
    )

    model_classes = (
        model.classes_
    )

    probabilities = {}

    for encoded_class, probability in zip(
        model_classes,
        probability_values,
    ):

        class_name = (
            label_encoder
            .inverse_transform(
                [
                    int(
                        encoded_class
                    )
                ]
            )[0]
        )

        probabilities[
            str(class_name)
        ] = round(
            float(
                probability
            ),
            6,
        )

    confidence = float(
        np.max(
            probability_values
        )
    )

    measurements = (
        calculate_leaf_measurements(
            contour
        )
    )

    result = {
        "image_name": (
            image_path.name
        ),
        "predicted_grade": str(
            predicted_grade
        ),
        "confidence": round(
            confidence,
            6,
        ),
        "confidence_percent": round(
            confidence * 100,
            2,
        ),
        "probabilities": (
            probabilities
        ),
        "segmentation_method": (
            segmentation_method
        ),
        "leaf_measurements": (
            measurements
        ),
    }

    return result


# =========================================================
# COMMAND LINE
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Predict the quality grade "
            "of a single tea leaf image."
        )
    )

    parser.add_argument(
        "image",
        help=(
            "Path to the tea leaf image."
        ),
    )

    parser.add_argument(
        "--output",
        help=(
            "Optional path to save "
            "prediction result as JSON."
        ),
        default=None,
    )

    args = parser.parse_args()

    try:

        result = (
            predict_single_image(
                args.image
            )
        )

        result_json = json.dumps(
            result,
            indent=4,
            ensure_ascii=False,
        )

        print(
            "\n"
            "======================================"
        )

        print(
            "SMART TEA LEAF GRADING RESULT"
        )

        print(
            "======================================"
        )

        print(
            result_json
        )

        print(
            "======================================"
        )

        if args.output:

            output_path = Path(
                args.output
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(
                output_path,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    result,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

            print(
                "\nPrediction JSON saved to:"
            )

            print(
                output_path
            )

    except Exception as error:

        print(
            "\nPrediction failed:"
        )

        print(
            str(
                error
            )
        )

        raise


if __name__ == "__main__":
    main()