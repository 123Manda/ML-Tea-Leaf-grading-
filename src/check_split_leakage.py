from pathlib import Path
from PIL import Image
import imagehash
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

HASH_SIZE = 16

# Smaller distance = more visually similar.
# Start conservatively.
NEAR_DUPLICATE_THRESHOLD = 5

CLASSES = [
    "Best",
    "Below_Best",
    "Poor",
]

SPLITS = [
    "train",
    "validation",
    "test",
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

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Generate perceptual hashes
# ---------------------------------------------------------

def get_phash(image_path):
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")

            return imagehash.phash(
                image,
                hash_size=HASH_SIZE,
            )

    except Exception as error:
        print(
            f"Could not hash {image_path}: {error}"
        )

        return None


# ---------------------------------------------------------
# Load images
# ---------------------------------------------------------

def collect_images():

    records = []

    for split_name in SPLITS:

        for class_name in CLASSES:

            folder = (
                SPLIT_DIR
                / split_name
                / class_name
            )

            if not folder.exists():
                continue

            files = [
                file_path
                for file_path in folder.iterdir()
                if file_path.is_file()
                and file_path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            ]

            for file_path in files:

                phash = get_phash(
                    file_path
                )

                if phash is None:
                    continue

                records.append(
                    {
                        "split": split_name,
                        "class": class_name,
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "phash": phash,
                    }
                )

    return records


# ---------------------------------------------------------
# Compare two split groups
# ---------------------------------------------------------

def compare_groups(
    first_group,
    second_group,
    first_name,
    second_name,
):

    matches = []

    print()
    print(
        f"Checking {first_name} "
        f"vs {second_name}..."
    )

    for index_a, image_a in enumerate(
        first_group
    ):

        if index_a % 100 == 0:

            print(
                f"Processed "
                f"{index_a}/{len(first_group)} "
                f"{first_name} images"
            )

        for image_b in second_group:

            distance = (
                image_a["phash"]
                - image_b["phash"]
            )

            if (
                distance
                <= NEAR_DUPLICATE_THRESHOLD
            ):

                matches.append(
                    {
                        "split_1": first_name,
                        "class_1": image_a["class"],
                        "file_1": image_a["file_name"],
                        "path_1": image_a["file_path"],

                        "split_2": second_name,
                        "class_2": image_b["class"],
                        "file_2": image_b["file_name"],
                        "path_2": image_b["file_path"],

                        "phash_distance": distance,

                        "same_class": (
                            image_a["class"]
                            == image_b["class"]
                        ),
                    }
                )

    return matches


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("TRAIN / VALIDATION / TEST LEAKAGE CHECK")
    print("=" * 70)

    print()
    print(
        "Calculating perceptual hashes..."
    )

    records = collect_images()

    print()
    print(
        f"Images hashed: {len(records)}"
    )

    train_images = [
        item
        for item in records
        if item["split"] == "train"
    ]

    validation_images = [
        item
        for item in records
        if item["split"] == "validation"
    ]

    test_images = [
        item
        for item in records
        if item["split"] == "test"
    ]

    print(
        f"Train      : {len(train_images)}"
    )

    print(
        f"Validation : {len(validation_images)}"
    )

    print(
        f"Test       : {len(test_images)}"
    )

    # -----------------------------------------------------
    # Cross-split comparisons
    # -----------------------------------------------------

    all_matches = []

    all_matches.extend(
        compare_groups(
            train_images,
            validation_images,
            "train",
            "validation",
        )
    )

    all_matches.extend(
        compare_groups(
            train_images,
            test_images,
            "train",
            "test",
        )
    )

    all_matches.extend(
        compare_groups(
            validation_images,
            test_images,
            "validation",
            "test",
        )
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    output_path = (
        RESULTS_DIR
        / "near_duplicate_leakage.csv"
    )

    if all_matches:

        df = pd.DataFrame(
            all_matches
        )

        df = df.sort_values(
            by="phash_distance"
        )

        df.to_csv(
            output_path,
            index=False,
        )

        same_class_count = int(
            df["same_class"].sum()
        )

        different_class_count = int(
            (~df["same_class"]).sum()
        )

        print()
        print("=" * 70)
        print("POTENTIAL LEAKAGE FOUND")
        print("=" * 70)

        print(
            f"Potential near-duplicate pairs: "
            f"{len(df)}"
        )

        print(
            f"Same-class pairs             : "
            f"{same_class_count}"
        )

        print(
            f"Different-class pairs        : "
            f"{different_class_count}"
        )

        print()
        print(
            f"Review report saved to:\n"
            f"{output_path}"
        )

        print()
        print(
            "Do NOT delete files automatically. "
            "Review the reported pairs first."
        )

    else:

        empty_df = pd.DataFrame(
            columns=[
                "split_1",
                "class_1",
                "file_1",
                "path_1",
                "split_2",
                "class_2",
                "file_2",
                "path_2",
                "phash_distance",
                "same_class",
            ]
        )

        empty_df.to_csv(
            output_path,
            index=False,
        )

        print()
        print("=" * 70)
        print("NO NEAR-DUPLICATE LEAKAGE FOUND")
        print("=" * 70)

        print(
            "No cross-split image pairs were "
            f"found within perceptual-hash "
            f"distance <= "
            f"{NEAR_DUPLICATE_THRESHOLD}."
        )

    print()
    print("Leakage check completed.")


if __name__ == "__main__":
    main()