from pathlib import Path
import os

from dotenv import load_dotenv
from pymongo import MongoClient
import gridfs


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv(ENV_PATH)

MONGODB_URI = os.getenv("MONGODB_URI")

MONGODB_DB_NAME = os.getenv(
    "MONGODB_DB_NAME",
    "tea_leaf_grading_db",
)


if not MONGODB_URI:
    raise ValueError(
        "MONGODB_URI was not found in .env"
    )


# =========================================================
# MONGODB CONNECTION
# =========================================================

client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=10000,
)

database = client[
    MONGODB_DB_NAME
]


# =========================================================
# COLLECTIONS
# =========================================================

predictions_collection = database[
    "predictions"
]


# =========================================================
# GRIDFS
# =========================================================

image_fs = gridfs.GridFS(
    database,
    collection="leaf_images",
)


# =========================================================
# CONNECTION TEST
# =========================================================

def ping_database():

    client.admin.command(
        "ping"
    )

    return {
        "status": "connected",
        "database": database.name,
    }