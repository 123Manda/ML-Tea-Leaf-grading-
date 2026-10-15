from pathlib import Path
import random
import shutil
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

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
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clean_unique"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def clear_directory(directory):
    """
    Remove an existing split directory so rerunning the
    script does not create duplicate files.
    """

    if directory.exists():
        shutil.rmtree(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def copy_files(files, destination, class_name, split_name, manifest):
    """
    Copy selected files to the split folder and record
    their split information.
    """

    class_destination = (
        destination
        / split_name
        / class_name
    )

    class_destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    for file_path in files:

        target_path = (
            class_destination
            / file_path.name
        )

        shutil.copy2(
            file_path,
            target_path,
        )

        manifest.append(
            {
                "file_name": file_path.name,
                "class": class_name,
                "split": split_name,
                "source_path": str(file_path),
                "split_path": str(target_path),
            }
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("CREATING TRAIN / VALIDATION / TEST SPLITS")
    print("=" * 70)

    random.seed(RANDOM_SEED)

    if not SOURCE_DIR.exists():

        print(
            f"ERROR: Clean dataset folder not found:\n{SOURCE_DIR}"
        )

        return

    # -----------------------------------------------------
    # Reset split directory
    # -----------------------------------------------------

    clear_directory(SPLIT_DIR)

    manifest = []
    summary_rows = []

    total_train = 0
    total_val = 0
    total_test = 0

    # -----------------------------------------------------
    # Process each class separately
    # -----------------------------------------------------

    for class_name in CLASSES:

        class_dir = (
            SOURCE_DIR
            / class_name
        )

        if not class_dir.exists():

            print(
                f"ERROR: Missing class folder: {class_dir}"
            )

            continue

        files = sorted(
            [
                path
                for path in class_dir.iterdir()
                if path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            ]
        )

        random.shuffle(files)

        total_count = len(files)

        train_count = int(
            total_count * TRAIN_RATIO
        )

        val_count = int(
            total_count * VAL_RATIO
        )

        test_count = (
            total_count
            - train_count
            - val_count
        )

        # -------------------------------------------------
        # Split
        # -------------------------------------------------

        train_files = files[
            :train_count
        ]

        val_files = files[
            train_count:
            train_count + val_count
        ]

        test_files = files[
            train_count + val_count:
        ]

        # -------------------------------------------------
        # Copy
        # -------------------------------------------------

        copy_files(
            train_files,
            SPLIT_DIR,
            class_name,
            "train",
            manifest,
        )

        copy_files(
            val_files,
            SPLIT_DIR,
            class_name,
            "validation",
            manifest,
        )

        copy_files(
            test_files,
            SPLIT_DIR,
            class_name,
            "test",
            manifest,
        )

        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)

        summary_rows.append(
            {
                "class": class_name,
                "total_images": total_count,
                "train_images": len(train_files),
                "validation_images": len(val_files),
                "test_images": len(test_files),
            }
        )

        print()
        print(class_name)
        print("-" * 45)
        print(
            f"Total      : {total_count}"
        )
        print(
            f"Train      : {len(train_files)}"
        )
        print(
            f"Validation : {len(val_files)}"
        )
        print(
            f"Test       : {len(test_files)}"
        )

    # -----------------------------------------------------
    # Save manifest
    # -----------------------------------------------------

    manifest_df = pd.DataFrame(
        manifest
    )

    manifest_path = (
        RESULTS_DIR
        / "dataset_split_manifest.csv"
    )

    manifest_df.to_csv(
        manifest_path,
        index=False,
    )

    # -----------------------------------------------------
    # Save summary
    # -----------------------------------------------------

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        RESULTS_DIR
        / "dataset_split_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # -----------------------------------------------------
    # Print final summary
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL DATASET SPLIT SUMMARY")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Total train images      : {total_train}"
    )

    print(
        f"Total validation images : {total_val}"
    )

    print(
        f"Total test images       : {total_test}"
    )

    print(
        f"Grand total             : "
        f"{total_train + total_val + total_test}"
    )

    print()
    print("Split folders created at:")
    print(SPLIT_DIR)

    print()
    print("Reports created:")
    print(manifest_path)
    print(summary_path)

    print()
    print(
        f"Random seed used: {RANDOM_SEED}"
    )

    print()
    print("Dataset split completed successfully.")


if __name__ == "__main__":
    main()