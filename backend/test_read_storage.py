from backend.database import (
    image_fs,
    predictions_collection,
)


def main():

    # Latest prediction record
    prediction = (
        predictions_collection
        .find_one(
            sort=[
                (
                    "created_at",
                    -1,
                )
            ]
        )
    )

    if prediction is None:

        print(
            "No prediction records found."
        )

        return

    print(
        "\n======================================"
    )

    print(
        "MONGODB PREDICTION RECORD"
    )

    print(
        "======================================"
    )

    print(
        f"Prediction ID: "
        f"{prediction['_id']}"
    )

    print(
        f"Image Name: "
        f"{prediction['image_name']}"
    )

    print(
        f"Predicted Grade: "
        f"{prediction['predicted_grade']}"
    )

    print(
        f"Confidence: "
        f"{prediction['confidence_percent']}%"
    )

    print(
        f"Segmentation Method: "
        f"{prediction['segmentation_method']}"
    )

    print(
        f"Image File ID: "
        f"{prediction['image_file_id']}"
    )

    # Check GridFS image
    image_file = image_fs.get(
        prediction[
            "image_file_id"
        ]
    )

    image_bytes = (
        image_file.read()
    )

    print(
        f"Stored Image Size: "
        f"{len(image_bytes)} bytes"
    )

    print(
        "Image found in GridFS: YES"
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()