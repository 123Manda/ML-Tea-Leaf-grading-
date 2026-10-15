from pathlib import Path
import json

from backend.storage import save_prediction_with_image


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "test"
    / "Best"
    / "best_0007.png"
)

PREDICTION_PATH = (
    PROJECT_ROOT
    / "results"
    / "single_prediction_test.json"
)


def main():

    # Load image
    with open(
        IMAGE_PATH,
        "rb",
    ) as image_file:

        image_bytes = (
            image_file.read()
        )

    # Load prediction result
    with open(
        PREDICTION_PATH,
        "r",
        encoding="utf-8",
    ) as prediction_file:

        prediction_result = (
            json.load(
                prediction_file
            )
        )

    # Save image + prediction
    saved_result = (
        save_prediction_with_image(
            image_bytes=image_bytes,
            filename=IMAGE_PATH.name,
            content_type="image/png",
            prediction_result=prediction_result,
        )
    )

    print(
        "MongoDB save successful."
    )

    print(
        f"Prediction ID: "
        f"{saved_result['prediction_id']}"
    )

    print(
        f"Image File ID: "
        f"{saved_result['image_file_id']}"
    )


if __name__ == "__main__":
    main()