from datetime import datetime, timezone

from backend.database import (
    image_fs,
    predictions_collection,
)


# =========================================================
# SAVE IMAGE + PREDICTION
# =========================================================

def save_prediction_with_image(
    image_bytes,
    filename,
    content_type,
    prediction_result,
):

    image_file_id = None

    try:
        # -------------------------------------------------
        # SAVE ORIGINAL IMAGE TO GRIDFS
        # -------------------------------------------------

        image_file_id = image_fs.put(
            image_bytes,
            filename=filename,
            content_type=content_type,
            upload_date=datetime.now(
                timezone.utc
            ),
        )

        # -------------------------------------------------
        # CREATE PREDICTION DOCUMENT
        # -------------------------------------------------

        prediction_document = {
            "image_file_id": image_file_id,
            "image_name": filename,
            "content_type": content_type,

            "predicted_grade": (
                prediction_result[
                    "predicted_grade"
                ]
            ),

            "confidence": (
                prediction_result[
                    "confidence"
                ]
            ),

            "confidence_percent": (
                prediction_result[
                    "confidence_percent"
                ]
            ),

            "probabilities": (
                prediction_result[
                    "probabilities"
                ]
            ),

            "segmentation_method": (
                prediction_result[
                    "segmentation_method"
                ]
            ),

            "leaf_measurements": (
                prediction_result[
                    "leaf_measurements"
                ]
            ),

            "created_at": datetime.now(
                timezone.utc
            ),
        }

        # -------------------------------------------------
        # SAVE PREDICTION RESULT
        # -------------------------------------------------

        inserted = (
            predictions_collection
            .insert_one(
                prediction_document
            )
        )

        return {
            "prediction_id": str(
                inserted.inserted_id
            ),
            "image_file_id": str(
                image_file_id
            ),
        }

    except Exception:

        # If prediction saving fails after
        # image upload, remove orphan image.
        if image_file_id is not None:

            try:
                image_fs.delete(
                    image_file_id
                )
            except Exception:
                pass

        raise