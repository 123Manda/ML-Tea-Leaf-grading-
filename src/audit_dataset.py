from pathlib import Path
from PIL import Image, UnidentifiedImageError
import pandas as pd
import hashlib
import re

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

def calculate_file_hash(file_path):
    """
    Calculate SHA-256 hash of the complete image file.
    Exact duplicate files will have the same hash.
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def detect_augmented_file(file_name):
    """
    Detect common naming patterns used for augmented images.
    """
    name = file_name.lower()

    patterns = [
        "__aug__",
        "_aug_",
        "-aug-",
        "augmented",
        "augmentation",
    ]

    return any(pattern in name for pattern in patterns)


def get_original_family_name(file_name):
    """
    Attempts to identify the original image family.

    Example:
    Gray_blight_000311__aug__00001.jpg
    becomes:
    Gray_blight_000311
    """

    stem = Path(file_name).stem

    stem = re.split(
        r"__aug__|_aug_|-aug-|augmented|augmentation",
        stem,
        flags=re.IGNORECASE,
    )[0]

    return stem


def inspect_image(image_path, class_name):
    """
    Inspect one image and return metadata.
    """

    record = {
        "class": class_name,
        "file_name": image_path.name,
        "file_path": str(image_path),
        "extension": image_path.suffix.lower(),
        "valid_image": False,
        "width": None,
        "height": None,
        "channels_mode": None,
        "file_size_kb": None,
        "sha256": None,
        "is_augmented": False,
        "original_family": None,
        "error": "",
    }

    try:
        record["file_size_kb"] = round(
            image_path.stat().st_size / 1024,
            2,
        )

        record["sha256"] = calculate_file_hash(image_path)

        record["is_augmented"] = detect_augmented_file(
            image_path.name
        )

        record["original_family"] = get_original_family_name(
            image_path.name
        )

        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            record["width"] = image.width
            record["height"] = image.height
            record["channels_mode"] = image.mode

        record["valid_image"] = True

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Exception,
    ) as error:
        record["error"] = str(error)

    return record


# ---------------------------------------------------------
# Main audit
# ---------------------------------------------------------

def main():

    all_records = []

    print("=" * 65)
    print("TEA LEAF GRADING DATASET AUDIT")
    print("=" * 65)

    for class_name in CLASSES:

        class_dir = RAW_DATA_DIR / class_name

        if not class_dir.exists():
            print(
                f"\nERROR: Folder not found: {class_dir}"
            )
            continue

        image_files = [
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        print(
            f"\nScanning {class_name}: "
            f"{len(image_files)} supported image files"
        )

        for image_path in image_files:

            record = inspect_image(
                image_path=image_path,
                class_name=class_name,
            )

            all_records.append(record)

    if not all_records:
        print("\nNo supported image files were found.")
        return

    df = pd.DataFrame(all_records)

    # -----------------------------------------------------
    # Exact duplicates
    # -----------------------------------------------------

    valid_hashes = (
        df["valid_image"]
        & df["sha256"].notna()
    )

    df["exact_duplicate"] = False

    df.loc[
        valid_hashes,
        "exact_duplicate",
    ] = (
        df.loc[
            valid_hashes,
            "sha256",
        ]
        .duplicated(
            keep=False
        )
    )

    # -----------------------------------------------------
    # Cross-class duplicate warning
    # -----------------------------------------------------

    df["cross_class_duplicate"] = False

    valid_df = df[
        df["valid_image"]
        & df["sha256"].notna()
    ].copy()

    hash_class_counts = (
        valid_df
        .groupby("sha256")["class"]
        .nunique()
    )

    cross_class_hashes = set(
        hash_class_counts[
            hash_class_counts > 1
        ].index
    )

    df.loc[
        df["sha256"].isin(cross_class_hashes),
        "cross_class_duplicate",
    ] = True

    # -----------------------------------------------------
    # Low-resolution warning
    # -----------------------------------------------------

    df["low_resolution"] = False

    valid_dimensions = (
        df["width"].notna()
        & df["height"].notna()
    )

    df.loc[
        valid_dimensions,
        "low_resolution",
    ] = (
        (df.loc[valid_dimensions, "width"] < 224)
        |
        (df.loc[valid_dimensions, "height"] < 224)
    )

    # -----------------------------------------------------
    # Review status
    # -----------------------------------------------------

    def assign_review_status(row):

        if not row["valid_image"]:
            return "EXCLUDE_CORRUPT"

        if row["cross_class_duplicate"]:
            return "REVIEW_CROSS_CLASS_DUPLICATE"

        if row["is_augmented"]:
            return "EXCLUDE_AUGMENTED_FROM_ORIGINAL_COUNT"

        if row["exact_duplicate"]:
            return "REVIEW_DUPLICATE"

        if row["low_resolution"]:
            return "REVIEW_LOW_RESOLUTION"

        return "OK"

    df["audit_status"] = df.apply(
        assign_review_status,
        axis=1,
    )

    # -----------------------------------------------------
    # Save full audit report
    # -----------------------------------------------------

    audit_csv = (
        RESULTS_DIR
        / "dataset_audit.csv"
    )

    df.to_csv(
        audit_csv,
        index=False,
    )

    # -----------------------------------------------------
    # Class summary
    # -----------------------------------------------------

    summary_rows = []

    for class_name in CLASSES:

        class_df = df[
            df["class"] == class_name
        ]

        summary_rows.append(
            {
                "class": class_name,
                "total_files": len(class_df),
                "valid_images": int(
                    class_df["valid_image"].sum()
                ),
                "corrupt_images": int(
                    (~class_df["valid_image"]).sum()
                ),
                "augmented_images": int(
                    class_df["is_augmented"].sum()
                ),
                "exact_duplicate_files": int(
                    class_df["exact_duplicate"].sum()
                ),
                "cross_class_duplicates": int(
                    class_df[
                        "cross_class_duplicate"
                    ].sum()
                ),
                "low_resolution_images": int(
                    class_df[
                        "low_resolution"
                    ].sum()
                ),
                "audit_ok_images": int(
                    (
                        class_df["audit_status"]
                        == "OK"
                    ).sum()
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_csv = (
        RESULTS_DIR
        / "dataset_audit_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    # -----------------------------------------------------
    # Duplicate details
    # -----------------------------------------------------

    duplicate_df = df[
        df["exact_duplicate"]
        | df["cross_class_duplicate"]
    ].copy()

    duplicate_csv = (
        RESULTS_DIR
        / "duplicate_images.csv"
    )

    duplicate_df.to_csv(
        duplicate_csv,
        index=False,
    )

    # -----------------------------------------------------
    # Augmented image details
    # -----------------------------------------------------

    augmented_df = df[
        df["is_augmented"]
    ].copy()

    augmented_csv = (
        RESULTS_DIR
        / "augmented_images.csv"
    )

    augmented_df.to_csv(
        augmented_csv,
        index=False,
    )

    # -----------------------------------------------------
    # Invalid image details
    # -----------------------------------------------------

    invalid_df = df[
        ~df["valid_image"]
    ].copy()

    invalid_csv = (
        RESULTS_DIR
        / "invalid_images.csv"
    )

    invalid_df.to_csv(
        invalid_csv,
        index=False,
    )

    # -----------------------------------------------------
    # Print report
    # -----------------------------------------------------

    print("\n")
    print("=" * 65)
    print("DATASET AUDIT SUMMARY")
    print("=" * 65)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\n")
    print("Reports created:")

    print(
        f"1. {audit_csv}"
    )

    print(
        f"2. {summary_csv}"
    )

    print(
        f"3. {duplicate_csv}"
    )

    print(
        f"4. {augmented_csv}"
    )

    print(
        f"5. {invalid_csv}"
    )

    print("\nAudit completed successfully.")


if __name__ == "__main__":
    main()