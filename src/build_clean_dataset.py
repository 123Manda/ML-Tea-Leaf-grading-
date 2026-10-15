from pathlib import Path
from PIL import Image
import hashlib
import shutil
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "clean_unique"
RESULTS_DIR = PROJECT_ROOT / "results"

CLASSES = [
    "Best",
    "Below_Best",
    "Poor",
]

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def calculate_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def is_augmented(file_name):
    name = file_name.lower()

    patterns = [
        "__aug__",
        "_aug_",
        "-aug-",
        "augmented",
        "augmentation",
    ]

    return any(pattern in name for pattern in patterns)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("BUILDING CLEAN UNIQUE DATASET")
    print("=" * 70)

    CLEAN_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = []

    manifest_rows = []

    for class_name in CLASSES:

        source_dir = RAW_DATA_DIR / class_name
        target_dir = CLEAN_DATA_DIR / class_name

        target_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Remove old files if script is rerun
        for old_file in target_dir.iterdir():
            if old_file.is_file():
                old_file.unlink()

        files = sorted(
            [
                path
                for path in source_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
        )

        seen_hashes = set()

        total = 0
        augmented_removed = 0
        duplicate_removed = 0
        invalid_removed = 0
        kept = 0

        for file_path in files:

            total += 1

            # ---------------------------------------------
            # Remove augmented images
            # ---------------------------------------------

            if is_augmented(file_path.name):

                augmented_removed += 1

                manifest_rows.append(
                    {
                        "class": class_name,
                        "file_name": file_path.name,
                        "status": "REMOVED_AUGMENTED",
                        "hash": "",
                    }
                )

                continue

            # ---------------------------------------------
            # Verify image
            # ---------------------------------------------

            try:

                with Image.open(file_path) as image:
                    image.verify()

            except Exception:

                invalid_removed += 1

                manifest_rows.append(
                    {
                        "class": class_name,
                        "file_name": file_path.name,
                        "status": "REMOVED_INVALID",
                        "hash": "",
                    }
                )

                continue

            # ---------------------------------------------
            # Exact duplicate detection
            # ---------------------------------------------

            file_hash = calculate_hash(file_path)

            if file_hash in seen_hashes:

                duplicate_removed += 1

                manifest_rows.append(
                    {
                        "class": class_name,
                        "file_name": file_path.name,
                        "status": "REMOVED_DUPLICATE",
                        "hash": file_hash,
                    }
                )

                continue

            seen_hashes.add(file_hash)

            # ---------------------------------------------
            # Keep unique original
            # ---------------------------------------------

            new_name = (
                f"{class_name.lower()}_"
                f"{kept + 1:04d}"
                f"{file_path.suffix.lower()}"
            )

            destination = target_dir / new_name

            shutil.copy2(
                file_path,
                destination,
            )

            kept += 1

            manifest_rows.append(
                {
                    "class": class_name,
                    "file_name": file_path.name,
                    "status": "KEPT",
                    "hash": file_hash,
                    "new_file_name": new_name,
                }
            )

        summary.append(
            {
                "class": class_name,
                "raw_images": total,
                "augmented_removed": augmented_removed,
                "duplicate_copies_removed": duplicate_removed,
                "invalid_removed": invalid_removed,
                "unique_original_images": kept,
            }
        )

        print()
        print(f"{class_name}")
        print("-" * 50)
        print(f"Raw images              : {total}")
        print(f"Augmented removed       : {augmented_removed}")
        print(f"Duplicate copies removed: {duplicate_removed}")
        print(f"Invalid removed         : {invalid_removed}")
        print(f"Unique originals kept   : {kept}")

    # -----------------------------------------------------
    # Save reports
    # -----------------------------------------------------

    summary_df = pd.DataFrame(summary)

    summary_path = (
        RESULTS_DIR
        / "clean_dataset_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    manifest_df = pd.DataFrame(manifest_rows)

    manifest_path = (
        RESULTS_DIR
        / "clean_dataset_manifest.csv"
    )

    manifest_df.to_csv(
        manifest_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("FINAL CLEAN DATASET SUMMARY")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("Clean images saved to:")
    print(CLEAN_DATA_DIR)

    print()
    print("Reports:")
    print(summary_path)
    print(manifest_path)

    print()
    print("Cleaning completed successfully.")


if __name__ == "__main__":
    main()