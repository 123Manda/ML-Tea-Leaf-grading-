from pathlib import Path
import os

from dotenv import load_dotenv
from pymongo import MongoClient


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv(
    "MONGODB_DB_NAME",
    "tea_leaf_grading_db",
)


# =========================================================
# TEST MONGODB CONNECTION
# =========================================================

def main():

    if not MONGODB_URI:
        raise ValueError(
            "MONGODB_URI was not found in .env"
        )

    print("Connecting to MongoDB Atlas...")

    client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=10000,
    )

    # Test connection
    client.admin.command("ping")

    database = client[
        MONGODB_DB_NAME
    ]

    print("MongoDB connection successful.")
    print(
        f"Database name: {database.name}"
    )

    client.close()


if __name__ == "__main__":
    main()