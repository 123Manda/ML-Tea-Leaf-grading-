from pathlib import Path
import sys
import tempfile
import os

from bson import ObjectId
from bson.errors import InvalidId

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from fastapi.responses import (
    Response,
)
from fastapi.responses import (
    Response,
    StreamingResponse,
)

from gridfs.errors import NoFile


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


# =========================================================
# PROJECT IMPORTS
# =========================================================

from predict_single_image import (  # noqa: E402
    predict_single_image,
)

from backend.database import (  # noqa: E402
    ping_database,
    predictions_collection,
    image_fs,
)

from backend.storage import (  # noqa: E402
    save_prediction_with_image,
)

from backend.report_generator import (
    generate_prediction_pdf,
)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Smart Tea Leaf Grading API",
    description=(
        "ML-based tea leaf grading API using "
        "OpenCV feature extraction and XGBoost."
    ),
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# SETTINGS
# =========================================================

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

MAX_IMAGE_SIZE = (
    10 * 1024 * 1024
)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "system": (
            "Smart Tea Leaf Grading API"
        ),
        "status": "running",
        "version": "1.0.0",
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():

    try:

        database_status = (
            ping_database()
        )

        return {
            "status": "healthy",
            "database": (
                database_status
            ),
            "model": "XGBoost",
            "classes": [
                "Best",
                "Below_Best",
                "Poor",
            ],
        }

    except Exception as error:

        raise HTTPException(
            status_code=503,
            detail=(
                "Database connection failed: "
                f"{str(error)}"
            ),
        )


# =========================================================
# PREDICT
# =========================================================

@app.post("/predict")
async def predict_leaf(
    file: UploadFile = File(...),
):

    if (
        file.content_type
        not in ALLOWED_CONTENT_TYPES
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid file type. "
                "Please upload JPG, PNG, "
                "or WEBP image."
            ),
        )

    image_bytes = await file.read()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:

        raise HTTPException(
            status_code=400,
            detail=(
                "Image is too large. "
                "Maximum allowed size is 10 MB."
            ),
        )

    original_filename = (
        file.filename
        or "uploaded_leaf.jpg"
    )

    suffix = (
        Path(original_filename)
        .suffix
        .lower()
    )

    if suffix not in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }:

        suffix = ".jpg"

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_file.write(
                image_bytes
            )

            temp_path = (
                temp_file.name
            )

        prediction_result = (
            predict_single_image(
                temp_path
            )
        )

        prediction_result[
            "image_name"
        ] = original_filename

        database_result = (
            save_prediction_with_image(
                image_bytes=image_bytes,
                filename=original_filename,
                content_type=(
                    file.content_type
                ),
                prediction_result=(
                    prediction_result
                ),
            )
        )

        image_file_id = (
            database_result[
                "image_file_id"
            ]
        )

        return {
            "success": True,
            "message": (
                "Tea leaf graded successfully."
            ),
            "prediction_id": (
                database_result[
                    "prediction_id"
                ]
            ),
            "image_file_id": (
                image_file_id
            ),
            "image_url": (
                f"/images/{image_file_id}"
            ),
            "result": (
                prediction_result
            ),
        }

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                f"{str(error)}"
            ),
        )

    finally:

        if (
            temp_path
            and os.path.exists(
                temp_path
            )
        ):

            try:
                os.remove(
                    temp_path
                )
            except OSError:
                pass


# =========================================================
# PREDICTION HISTORY
# =========================================================

@app.get("/predictions")
def get_prediction_history(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
):

    try:

        cursor = (
            predictions_collection
            .find()
            .sort(
                "created_at",
                -1,
            )
            .limit(
                limit
            )
        )

        predictions = []

        for record in cursor:

            created_at = (
                record.get(
                    "created_at"
                )
            )

            image_file_id = str(
                record[
                    "image_file_id"
                ]
            )

            predictions.append(
                {
                    "prediction_id": str(
                        record["_id"]
                    ),
                    "image_file_id": (
                        image_file_id
                    ),
                    "image_url": (
                        f"/images/"
                        f"{image_file_id}"
                    ),
                    "image_name": (
                        record.get(
                            "image_name"
                        )
                    ),
                    "content_type": (
                        record.get(
                            "content_type"
                        )
                    ),
                    "predicted_grade": (
                        record.get(
                            "predicted_grade"
                        )
                    ),
                    "confidence": (
                        record.get(
                            "confidence"
                        )
                    ),
                    "confidence_percent": (
                        record.get(
                            "confidence_percent"
                        )
                    ),
                    "probabilities": (
                        record.get(
                            "probabilities",
                            {},
                        )
                    ),
                    "segmentation_method": (
                        record.get(
                            "segmentation_method"
                        )
                    ),
                    "leaf_measurements": (
                        record.get(
                            "leaf_measurements",
                            {},
                        )
                    ),
                    "created_at": (
                        created_at.isoformat()
                        if created_at
                        else None
                    ),
                }
            )

        return {
            "success": True,
            "count": len(
                predictions
            ),
            "predictions": (
                predictions
            ),
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to retrieve "
                "prediction history: "
                f"{str(error)}"
            ),
        )


# =========================================================
# GET STORED IMAGE
# =========================================================

@app.get("/images/{image_file_id}")
def get_stored_image(
    image_file_id: str,
):

    try:

        object_id = ObjectId(
            image_file_id
        )

    except InvalidId:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file ID.",
        )

    try:

        image_file = (
            image_fs.get(
                object_id
            )
        )

        image_bytes = (
            image_file.read()
        )

        content_type = getattr(
            image_file,
            "content_type",
            None,
        )

        if not content_type:

            content_type = (
                "application/octet-stream"
            )

        return Response(
            content=image_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": (
                    f'inline; filename="'
                    f'{image_file.filename}"'
                )
            },
        )

    except NoFile:

        raise HTTPException(
            status_code=404,
            detail="Stored image not found.",
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to retrieve image: "
                f"{str(error)}"
            ),
        )


        # =========================================================
# GET SINGLE PREDICTION
# =========================================================

@app.get("/predictions/{prediction_id}")
def get_single_prediction(
    prediction_id: str,
):

    try:

        object_id = ObjectId(
            prediction_id
        )

    except InvalidId:

        raise HTTPException(
            status_code=400,
            detail="Invalid prediction ID.",
        )

    try:

        record = (
            predictions_collection
            .find_one(
                {
                    "_id": object_id
                }
            )
        )

        if record is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Prediction record "
                    "not found."
                ),
            )

        created_at = (
            record.get(
                "created_at"
            )
        )

        image_file_id = str(
            record[
                "image_file_id"
            ]
        )

        return {
            "success": True,
            "prediction": {
                "prediction_id": str(
                    record["_id"]
                ),
                "image_file_id": (
                    image_file_id
                ),
                "image_url": (
                    f"/images/"
                    f"{image_file_id}"
                ),
                "image_name": (
                    record.get(
                        "image_name"
                    )
                ),
                "content_type": (
                    record.get(
                        "content_type"
                    )
                ),
                "predicted_grade": (
                    record.get(
                        "predicted_grade"
                    )
                ),
                "confidence": (
                    record.get(
                        "confidence"
                    )
                ),
                "confidence_percent": (
                    record.get(
                        "confidence_percent"
                    )
                ),
                "probabilities": (
                    record.get(
                        "probabilities",
                        {},
                    )
                ),
                "segmentation_method": (
                    record.get(
                        "segmentation_method"
                    )
                ),
                "leaf_measurements": (
                    record.get(
                        "leaf_measurements",
                        {},
                    )
                ),
                "created_at": (
                    created_at.isoformat()
                    if created_at
                    else None
                ),
            },
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to retrieve "
                "prediction: "
                f"{str(error)}"
            ),
        )

        # =========================================================
# DELETE PREDICTION + STORED IMAGE
# =========================================================

@app.delete("/predictions/{prediction_id}")
def delete_prediction(
    prediction_id: str,
):

    try:

        object_id = ObjectId(
            prediction_id
        )

    except InvalidId:

        raise HTTPException(
            status_code=400,
            detail="Invalid prediction ID.",
        )

    try:

        record = (
            predictions_collection
            .find_one(
                {
                    "_id": object_id
                }
            )
        )

        if record is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Prediction record "
                    "not found."
                ),
            )

        image_file_id = (
            record.get(
                "image_file_id"
            )
        )

        # Delete stored image from GridFS
        if image_file_id is not None:

            image_fs.delete(
                image_file_id
            )

        # Delete prediction document
        delete_result = (
            predictions_collection
            .delete_one(
                {
                    "_id": object_id
                }
            )
        )

        if delete_result.deleted_count != 1:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Prediction record "
                    "could not be deleted."
                ),
            )

        return {
            "success": True,
            "message": (
                "Prediction and stored "
                "image deleted successfully."
            ),
            "prediction_id": (
                prediction_id
            ),
            "image_file_id": (
                str(image_file_id)
                if image_file_id
                else None
            ),
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to delete "
                "prediction: "
                f"{str(error)}"
            ),
        )

        # =========================================================
# DOWNLOAD PREDICTION PDF REPORT
# =========================================================

@app.get("/predictions/{prediction_id}/report")
def download_prediction_report(
    prediction_id: str,
):

    try:
        object_id = ObjectId(
            prediction_id
        )

    except InvalidId:
        raise HTTPException(
            status_code=400,
            detail="Invalid prediction ID.",
        )

    try:
        prediction_record = (
            predictions_collection
            .find_one(
                {
                    "_id": object_id
                }
            )
        )

        if prediction_record is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Prediction record "
                    "not found."
                ),
            )

        image_bytes = None

        image_file_id = (
            prediction_record.get(
                "image_file_id"
            )
        )

        if image_file_id is not None:
            try:
                stored_image = (
                    image_fs.get(
                        image_file_id
                    )
                )

                image_bytes = (
                    stored_image.read()
                )

            except NoFile:
                image_bytes = None

        pdf_buffer = (
            generate_prediction_pdf(
                prediction_record=(
                    prediction_record
                ),
                image_bytes=(
                    image_bytes
                ),
            )
        )

        report_filename = (
            "tea_leaf_grading_report_"
            f"{prediction_id}.pdf"
        )

        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="'
                    f'{report_filename}"'
                )
            },
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to generate PDF report: "
                f"{str(error)}"
            ),
        )