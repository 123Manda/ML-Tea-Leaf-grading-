from pathlib import Path
import math
import shutil

import cv2
import numpy as np
import pandas as pd

from skimage.feature import graycomatrix, graycoprops


# =========================================================
# CONFIGURATION
# =========================================================

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

MAX_IMAGE_SIZE = 800
TEXTURE_SIZE = 128
PREVIEW_IMAGES_PER_CLASS = 4
FALLBACK_PREVIEW_LIMIT = 12

BORDER_MARGIN = 5


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)

FEATURE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

PREVIEW_DIR = (
    RESULTS_DIR
    / "segmentation_preview"
)

FALLBACK_PREVIEW_DIR = (
    RESULTS_DIR
    / "fallback_segmentation_preview"
)

FEATURE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# UTILITY
# =========================================================

def resize_image(image, max_size=MAX_IMAGE_SIZE):

    height, width = image.shape[:2]

    max_dimension = max(
        height,
        width,
    )

    if max_dimension <= max_size:
        return image

    scale = max_size / max_dimension

    new_width = int(
        round(width * scale)
    )

    new_height = int(
        round(height * scale)
    )

    return cv2.resize(
        image,
        (
            new_width,
            new_height,
        ),
        interpolation=cv2.INTER_AREA,
    )


def contour_is_valid(
    contour,
    image_shape,
):

    if contour is None:
        return False

    height, width = image_shape[:2]

    image_area = (
        height * width
    )

    area = cv2.contourArea(
        contour
    )

    if area <= 0:
        return False

    area_ratio = (
        area / image_area
    )

    if area_ratio < 0.003:
        return False

    if area_ratio > 0.70:
        return False

    x, y, w, h = cv2.boundingRect(
        contour
    )

    touches_border = (
        x <= BORDER_MARGIN
        or y <= BORDER_MARGIN
        or (
            x + w
            >= width - BORDER_MARGIN
        )
        or (
            y + h
            >= height - BORDER_MARGIN
        )
    )

    if touches_border:
        return False

    return True


# =========================================================
# COMPONENT CLEANING
# =========================================================

def remove_border_components(mask):

    height, width = mask.shape[:2]

    number_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    cleaned = np.zeros_like(
        mask
    )

    for label_id in range(
        1,
        number_labels,
    ):

        x = stats[
            label_id,
            cv2.CC_STAT_LEFT
        ]

        y = stats[
            label_id,
            cv2.CC_STAT_TOP
        ]

        w = stats[
            label_id,
            cv2.CC_STAT_WIDTH
        ]

        h = stats[
            label_id,
            cv2.CC_STAT_HEIGHT
        ]

        area = stats[
            label_id,
            cv2.CC_STAT_AREA
        ]

        if area < 100:
            continue

        touches_border = (
            x <= BORDER_MARGIN
            or y <= BORDER_MARGIN
            or (
                x + w
                >= width - BORDER_MARGIN
            )
            or (
                y + h
                >= height - BORDER_MARGIN
            )
        )

        if touches_border:
            continue

        cleaned[
            labels == label_id
        ] = 255

    return cleaned


def select_best_component(
    mask,
):

    height, width = mask.shape[:2]

    image_area = (
        height * width
    )

    image_center_x = (
        width / 2.0
    )

    image_center_y = (
        height / 2.0
    )

    max_distance = math.sqrt(
        image_center_x ** 2
        +
        image_center_y ** 2
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        area_ratio = (
            area / image_area
        )

        if area_ratio < 0.003:
            continue

        if area_ratio > 0.70:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        touches_border = (
            x <= BORDER_MARGIN
            or y <= BORDER_MARGIN
            or (
                x + w
                >= width - BORDER_MARGIN
            )
            or (
                y + h
                >= height - BORDER_MARGIN
            )
        )

        if touches_border:
            continue

        moments = cv2.moments(
            contour
        )

        if moments["m00"] != 0:

            cx = (
                moments["m10"]
                / moments["m00"]
            )

            cy = (
                moments["m01"]
                / moments["m00"]
            )

        else:

            cx = (
                x + w / 2.0
            )

            cy = (
                y + h / 2.0
            )

        distance = math.sqrt(
            (
                cx - image_center_x
            ) ** 2
            +
            (
                cy - image_center_y
            ) ** 2
        )

        center_score = (
            1.0
            -
            min(
                distance / max_distance,
                1.0,
            )
        )

        score = (
            area
            *
            (
                1.0
                +
                0.40 * center_score
            )
        )

        candidates.append(
            (
                score,
                contour,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return candidates[0][1]


def contour_to_mask(
    contour,
    image_shape,
):

    mask = np.zeros(
        image_shape[:2],
        dtype=np.uint8,
    )

    cv2.drawContours(
        mask,
        [contour],
        -1,
        255,
        thickness=-1,
    )

    return mask


# =========================================================
# PRIMARY SEGMENTATION
# =========================================================

def create_primary_colour_mask(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    green_mask = (
        (h >= 22)
        &
        (h <= 105)
        &
        (s >= 32)
        &
        (v >= 20)
    )

    brown_mask = (
        (h >= 3)
        &
        (h <= 35)
        &
        (s >= 45)
        &
        (v >= 20)
    )

    dark_coloured_mask = (
        (s >= 40)
        &
        (v >= 15)
        &
        (v <= 105)
    )

    combined = (
        green_mask
        |
        brown_mask
        |
        dark_coloured_mask
    )

    mask = (
        combined.astype(
            np.uint8
        )
        * 255
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    return mask


def primary_segmentation(image):

    mask = create_primary_colour_mask(
        image
    )

    mask = remove_border_components(
        mask
    )

    contour = select_best_component(
        mask
    )

    if not contour_is_valid(
        contour,
        image.shape,
    ):
        return None, None

    final_mask = contour_to_mask(
        contour,
        image.shape,
    )

    return (
        final_mask,
        contour,
    )


# =========================================================
# FALLBACK 1 — LAB BACKGROUND DISTANCE
# =========================================================

def estimate_background_lab(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    ).astype(
        np.float32
    )

    height, width = lab.shape[:2]

    border_size = max(
        5,
        int(
            min(
                height,
                width,
            )
            * 0.04
        ),
    )

    top = lab[
        :border_size,
        :,
    ].reshape(
        -1,
        3,
    )

    bottom = lab[
        height - border_size:,
        :,
    ].reshape(
        -1,
        3,
    )

    left = lab[
        :,
        :border_size,
    ].reshape(
        -1,
        3,
    )

    right = lab[
        :,
        width - border_size:,
    ].reshape(
        -1,
        3,
    )

    border_pixels = np.vstack(
        [
            top,
            bottom,
            left,
            right,
        ]
    )

    background_lab = np.median(
        border_pixels,
        axis=0,
    )

    return (
        lab,
        background_lab,
    )


def lab_fallback_segmentation(image):

    lab, background_lab = (
        estimate_background_lab(
            image
        )
    )

    difference = np.sqrt(
        np.sum(
            (
                lab
                - background_lab
            ) ** 2,
            axis=2,
        )
    )

    difference_8bit = np.clip(
        difference * 3.0,
        0,
        255,
    ).astype(
        np.uint8
    )

    # Automatic threshold
    _, otsu_mask = cv2.threshold(
        difference_8bit,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    # Minimum colour difference
    manual_mask = (
        difference > 10.0
    ).astype(
        np.uint8
    ) * 255

    mask = cv2.bitwise_and(
        otsu_mask,
        manual_mask,
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11, 11),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    mask = remove_border_components(
        mask
    )

    contour = select_best_component(
        mask
    )

    if not contour_is_valid(
        contour,
        image.shape,
    ):
        return None, None

    final_mask = contour_to_mask(
        contour,
        image.shape,
    )

    return (
        final_mask,
        contour,
    )


# =========================================================
# FALLBACK 2 — GRABCUT
# =========================================================

def grabcut_fallback_segmentation(
    image,
):

    height, width = image.shape[:2]

    margin_x = max(
        5,
        int(
            width * 0.05
        ),
    )

    margin_y = max(
        5,
        int(
            height * 0.05
        ),
    )

    rectangle = (
        margin_x,
        margin_y,
        width - 2 * margin_x,
        height - 2 * margin_y,
    )

    grabcut_mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    bg_model = np.zeros(
        (
            1,
            65,
        ),
        dtype=np.float64,
    )

    fg_model = np.zeros(
        (
            1,
            65,
        ),
        dtype=np.float64,
    )

    try:

        cv2.grabCut(
            image,
            grabcut_mask,
            rectangle,
            bg_model,
            fg_model,
            5,
            cv2.GC_INIT_WITH_RECT,
        )

    except Exception:

        return (
            None,
            None,
        )

    binary_mask = np.where(
        (
            grabcut_mask == cv2.GC_FGD
        )
        |
        (
            grabcut_mask == cv2.GC_PR_FGD
        ),
        255,
        0,
    ).astype(
        np.uint8
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        open_kernel,
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    binary_mask = remove_border_components(
        binary_mask
    )

    contour = select_best_component(
        binary_mask
    )

    if not contour_is_valid(
        contour,
        image.shape,
    ):
        return (
            None,
            None,
        )

    final_mask = contour_to_mask(
        contour,
        image.shape,
    )

    return (
        final_mask,
        contour,
    )


# =========================================================
# COMPLETE SEGMENTATION
# =========================================================

def segment_leaf(image):

    # -----------------------------------------------------
    # Method 1 — Primary
    # -----------------------------------------------------

    mask, contour = primary_segmentation(
        image
    )

    if contour is not None:

        return (
            mask,
            contour,
            "primary_hsv",
        )

    # -----------------------------------------------------
    # Method 2 — LAB fallback
    # -----------------------------------------------------

    mask, contour = (
        lab_fallback_segmentation(
            image
        )
    )

    if contour is not None:

        return (
            mask,
            contour,
            "fallback_lab",
        )

    # -----------------------------------------------------
    # Method 3 — GrabCut fallback
    # -----------------------------------------------------

    mask, contour = (
        grabcut_fallback_segmentation(
            image
        )
    )

    if contour is not None:

        return (
            mask,
            contour,
            "fallback_grabcut",
        )

    return (
        None,
        None,
        "failed",
    )


# =========================================================
# COLOR FEATURES
# =========================================================

def calculate_color_features(
    image,
    mask,
):

    features = {}

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    valid_pixels = (
        mask > 0
    )

    if np.sum(
        valid_pixels
    ) == 0:
        return None

    rgb_pixels = rgb[
        valid_pixels
    ]

    features["mean_r"] = float(
        np.mean(
            rgb_pixels[:, 0]
        )
    )

    features["mean_g"] = float(
        np.mean(
            rgb_pixels[:, 1]
        )
    )

    features["mean_b"] = float(
        np.mean(
            rgb_pixels[:, 2]
        )
    )

    features["std_r"] = float(
        np.std(
            rgb_pixels[:, 0]
        )
    )

    features["std_g"] = float(
        np.std(
            rgb_pixels[:, 1]
        )
    )

    features["std_b"] = float(
        np.std(
            rgb_pixels[:, 2]
        )
    )

    hsv_pixels = hsv[
        valid_pixels
    ]

    features["mean_h"] = float(
        np.mean(
            hsv_pixels[:, 0]
        )
    )

    features["mean_s"] = float(
        np.mean(
            hsv_pixels[:, 1]
        )
    )

    features["mean_v"] = float(
        np.mean(
            hsv_pixels[:, 2]
        )
    )

    features["std_h"] = float(
        np.std(
            hsv_pixels[:, 0]
        )
    )

    features["std_s"] = float(
        np.std(
            hsv_pixels[:, 1]
        )
    )

    features["std_v"] = float(
        np.std(
            hsv_pixels[:, 2]
        )
    )

    lab_pixels = lab[
        valid_pixels
    ]

    features["mean_lab_l"] = float(
        np.mean(
            lab_pixels[:, 0]
        )
    )

    features["mean_lab_a"] = float(
        np.mean(
            lab_pixels[:, 1]
        )
    )

    features["mean_lab_b"] = float(
        np.mean(
            lab_pixels[:, 2]
        )
    )

    features["std_lab_l"] = float(
        np.std(
            lab_pixels[:, 0]
        )
    )

    features["std_lab_a"] = float(
        np.std(
            lab_pixels[:, 1]
        )
    )

    features["std_lab_b"] = float(
        np.std(
            lab_pixels[:, 2]
        )
    )

    hue = hsv_pixels[:, 0]
    sat = hsv_pixels[:, 1]
    val = hsv_pixels[:, 2]

    green_pixels = (
        (hue >= 30)
        &
        (hue <= 95)
        &
        (sat >= 30)
    )

    yellow_brown_pixels = (
        (hue >= 5)
        &
        (hue < 30)
        &
        (sat >= 35)
    )

    dark_pixels = (
        val < 80
    )

    bright_pixels = (
        val > 180
    )

    total_pixels = len(
        hsv_pixels
    )

    features["green_ratio"] = float(
        np.sum(
            green_pixels
        )
        / total_pixels
    )

    features["yellow_brown_ratio"] = float(
        np.sum(
            yellow_brown_pixels
        )
        / total_pixels
    )

    features["dark_ratio"] = float(
        np.sum(
            dark_pixels
        )
        / total_pixels
    )

    features["bright_ratio"] = float(
        np.sum(
            bright_pixels
        )
        / total_pixels
    )

    return features


# =========================================================
# SHAPE / SIZE FEATURES
# =========================================================

def calculate_shape_features(
    contour,
    image_shape,
):

    features = {}

    height, width = (
        image_shape[:2]
    )

    image_area = (
        height * width
    )

    area = cv2.contourArea(
        contour
    )

    perimeter = cv2.arcLength(
        contour,
        True,
    )

    x, y, w, h = cv2.boundingRect(
        contour
    )

    bounding_area = (
        w * h
    )

    hull = cv2.convexHull(
        contour
    )

    hull_area = cv2.contourArea(
        hull
    )

    features["area_ratio"] = float(
        area / image_area
    )

    features[
        "perimeter_normalized"
    ] = float(
        perimeter
        /
        math.sqrt(
            image_area
        )
    )

    features[
        "bbox_width_ratio"
    ] = float(
        w / width
    )

    features[
        "bbox_height_ratio"
    ] = float(
        h / height
    )

    features["aspect_ratio"] = (
        float(
            w / h
        )
        if h > 0
        else 0.0
    )

    features["extent"] = (
        float(
            area
            / bounding_area
        )
        if bounding_area > 0
        else 0.0
    )

    features["solidity"] = (
        float(
            area
            / hull_area
        )
        if hull_area > 0
        else 0.0
    )

    features["circularity"] = (
        float(
            (
                4
                * math.pi
                * area
            )
            /
            (
                perimeter ** 2
            )
        )
        if perimeter > 0
        else 0.0
    )

    equivalent_diameter = math.sqrt(
        (
            4
            * area
        )
        / math.pi
    )

    features[
        "equivalent_diameter_normalized"
    ] = float(
        equivalent_diameter
        /
        math.sqrt(
            image_area
        )
    )

    if len(contour) >= 5:

        ellipse = cv2.fitEllipse(
            contour
        )

        _, axes, _ = ellipse

        major_axis = max(
            axes
        )

        minor_axis = min(
            axes
        )

        features[
            "ellipse_axis_ratio"
        ] = (
            float(
                minor_axis
                / major_axis
            )
            if major_axis > 0
            else 0.0
        )

    else:

        features[
            "ellipse_axis_ratio"
        ] = 0.0

    return features


# =========================================================
# TEXTURE FEATURES
# =========================================================

def calculate_texture_features(
    image,
    mask,
    contour,
):

    features = {}

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    x, y, w, h = cv2.boundingRect(
        contour
    )

    gray_crop = grayscale[
        y:y + h,
        x:x + w,
    ]

    mask_crop = mask[
        y:y + h,
        x:x + w,
    ]

    valid_pixels = gray_crop[
        mask_crop > 0
    ]

    if len(
        valid_pixels
    ) == 0:
        return None

    mean_leaf_gray = int(
        np.mean(
            valid_pixels
        )
    )

    texture_image = (
        gray_crop.copy()
    )

    texture_image[
        mask_crop == 0
    ] = mean_leaf_gray

    texture_image = cv2.resize(
        texture_image,
        (
            TEXTURE_SIZE,
            TEXTURE_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    quantized = (
        texture_image
        // 16
    ).astype(
        np.uint8
    )

    glcm = graycomatrix(
        quantized,
        distances=[
            1,
            2,
        ],
        angles=[
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4,
        ],
        levels=16,
        symmetric=True,
        normed=True,
    )

    features[
        "glcm_contrast"
    ] = float(
        np.mean(
            graycoprops(
                glcm,
                "contrast",
            )
        )
    )

    features[
        "glcm_homogeneity"
    ] = float(
        np.mean(
            graycoprops(
                glcm,
                "homogeneity",
            )
        )
    )

    features[
        "glcm_energy"
    ] = float(
        np.mean(
            graycoprops(
                glcm,
                "energy",
            )
        )
    )

    features[
        "glcm_correlation"
    ] = float(
        np.mean(
            graycoprops(
                glcm,
                "correlation",
            )
        )
    )

    features["gray_mean"] = float(
        np.mean(
            valid_pixels
        )
    )

    features["gray_std"] = float(
        np.std(
            valid_pixels
        )
    )

    histogram = cv2.calcHist(
        [
            valid_pixels.astype(
                np.uint8
            )
        ],
        [0],
        None,
        [256],
        [0, 256],
    )

    histogram = (
        histogram.flatten()
    )

    histogram_sum = np.sum(
        histogram
    )

    if histogram_sum > 0:

        probability = (
            histogram
            / histogram_sum
        )

        probability = probability[
            probability > 0
        ]

        entropy = -np.sum(
            probability
            * np.log2(
                probability
            )
        )

    else:

        entropy = 0.0

    features["entropy"] = float(
        entropy
    )

    eroded_mask = cv2.erode(
        mask,
        np.ones(
            (
                5,
                5,
            ),
            np.uint8,
        ),
        iterations=1,
    )

    laplacian = cv2.Laplacian(
        grayscale,
        cv2.CV_64F,
    )

    laplacian_pixels = laplacian[
        eroded_mask > 0
    ]

    if len(
        laplacian_pixels
    ) > 0:

        features[
            "laplacian_variance"
        ] = float(
            np.var(
                laplacian_pixels
            )
        )

    else:

        features[
            "laplacian_variance"
        ] = 0.0

    return features


# =========================================================
# PREVIEW
# =========================================================

def save_preview(
    image,
    contour,
    destination,
):

    preview = image.copy()

    cv2.drawContours(
        preview,
        [contour],
        -1,
        (
            0,
            255,
            0,
        ),
        thickness=3,
    )

    x, y, w, h = cv2.boundingRect(
        contour
    )

    cv2.rectangle(
        preview,
        (
            x,
            y,
        ),
        (
            x + w,
            y + h,
        ),
        (
            255,
            0,
            0,
        ),
        thickness=2,
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(destination),
        preview,
    )


# =========================================================
# SINGLE IMAGE
# =========================================================

def extract_features_from_image(
    image_path,
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            "OpenCV could not read image."
        )

    image = resize_image(
        image
    )

    (
        mask,
        contour,
        segmentation_method,
    ) = segment_leaf(
        image
    )

    if (
        mask is None
        or contour is None
    ):

        raise ValueError(
            "Leaf segmentation failed."
        )

    color_features = (
        calculate_color_features(
            image,
            mask,
        )
    )

    shape_features = (
        calculate_shape_features(
            contour,
            image.shape,
        )
    )

    texture_features = (
        calculate_texture_features(
            image,
            mask,
            contour,
        )
    )

    if color_features is None:

        raise ValueError(
            "Color feature extraction failed."
        )

    if texture_features is None:

        raise ValueError(
            "Texture feature extraction failed."
        )

    features = {}

    features.update(
        color_features
    )

    features.update(
        shape_features
    )

    features.update(
        texture_features
    )

    return (
        features,
        image,
        contour,
        segmentation_method,
    )


# =========================================================
# PROCESS SPLIT
# =========================================================

def process_split(
    split_name,
):

    print()
    print("=" * 70)

    print(
        f"PROCESSING SPLIT: "
        f"{split_name.upper()}"
    )

    print("=" * 70)

    records = []
    failures = []
    segmentation_records = []

    fallback_preview_count = 0

    for class_name in CLASSES:

        class_dir = (
            SPLIT_DIR
            / split_name
            / class_name
        )

        if not class_dir.exists():

            print(
                f"Missing folder: "
                f"{class_dir}"
            )

            continue

        image_files = sorted(
            [
                path
                for path
                in class_dir.iterdir()
                if (
                    path.is_file()
                    and
                    path.suffix.lower()
                    in SUPPORTED_EXTENSIONS
                )
            ]
        )

        print()

        print(
            f"{class_name}: "
            f"{len(image_files)} images"
        )

        successful_count = 0

        for index, image_path in enumerate(
            image_files,
            start=1,
        ):

            try:

                (
                    features,
                    image,
                    contour,
                    method,
                ) = extract_features_from_image(
                    image_path
                )

                row = {
                    "file_name":
                        image_path.name,
                    "class":
                        class_name,
                    "split":
                        split_name,
                }

                row.update(
                    features
                )

                records.append(
                    row
                )

                segmentation_records.append(
                    {
                        "split":
                            split_name,
                        "class":
                            class_name,
                        "file_name":
                            image_path.name,
                        "segmentation_method":
                            method,
                    }
                )

                successful_count += 1

                # Normal previews
                if (
                    successful_count
                    <= PREVIEW_IMAGES_PER_CLASS
                ):

                    preview_path = (
                        PREVIEW_DIR
                        / split_name
                        / class_name
                        / (
                            image_path.stem
                            + "_preview.jpg"
                        )
                    )

                    save_preview(
                        image,
                        contour,
                        preview_path,
                    )

                # Fallback previews
                if (
                    method
                    != "primary_hsv"
                    and
                    fallback_preview_count
                    < FALLBACK_PREVIEW_LIMIT
                ):

                    fallback_preview_path = (
                        FALLBACK_PREVIEW_DIR
                        / split_name
                        / class_name
                        / (
                            image_path.stem
                            + "_"
                            + method
                            + ".jpg"
                        )
                    )

                    save_preview(
                        image,
                        contour,
                        fallback_preview_path,
                    )

                    fallback_preview_count += 1

            except Exception as error:

                failures.append(
                    {
                        "split":
                            split_name,
                        "class":
                            class_name,
                        "file_name":
                            image_path.name,
                        "file_path":
                            str(
                                image_path
                            ),
                        "error":
                            str(
                                error
                            ),
                    }
                )

            if (
                index % 50 == 0
                or
                index
                == len(
                    image_files
                )
            ):

                print(
                    f"Processed "
                    f"{index}/"
                    f"{len(image_files)}"
                )

        print(
            f"Successful: "
            f"{successful_count}"
        )

    return (
        records,
        failures,
        segmentation_records,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)

    print(
        "TEA LEAF FEATURE EXTRACTION"
    )

    print("=" * 70)

    print()

    print(
        "Segmentation pipeline:"
    )

    print(
        "1. Primary HSV segmentation"
    )

    print(
        "2. LAB background-distance fallback"
    )

    print(
        "3. GrabCut final fallback"
    )

    print()

    if PREVIEW_DIR.exists():

        shutil.rmtree(
            PREVIEW_DIR
        )

    if FALLBACK_PREVIEW_DIR.exists():

        shutil.rmtree(
            FALLBACK_PREVIEW_DIR
        )

    PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FALLBACK_PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_failures = []
    all_segmentation_records = []

    summary_rows = []

    for split_name in SPLITS:

        (
            records,
            failures,
            segmentation_records,
        ) = process_split(
            split_name
        )

        output_df = pd.DataFrame(
            records
        )

        output_path = (
            FEATURE_OUTPUT_DIR
            / f"{split_name}_features.csv"
        )

        output_df.to_csv(
            output_path,
            index=False,
        )

        all_failures.extend(
            failures
        )

        all_segmentation_records.extend(
            segmentation_records
        )

        feature_count = (
            len(
                output_df.columns
            )
            - 3
            if len(
                output_df.columns
            ) >= 3
            else 0
        )

        summary_rows.append(
            {
                "split":
                    split_name,
                "features_created":
                    len(
                        records
                    ),
                "failures":
                    len(
                        failures
                    ),
                "feature_columns":
                    feature_count,
            }
        )

        print()

        print(
            f"Saved: "
            f"{output_path}"
        )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        RESULTS_DIR
        / "feature_extraction_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # -----------------------------------------------------
    # FAILURES
    # -----------------------------------------------------

    failure_path = (
        RESULTS_DIR
        / "feature_extraction_failures.csv"
    )

    failure_df = pd.DataFrame(
        all_failures,
        columns=[
            "split",
            "class",
            "file_name",
            "file_path",
            "error",
        ],
    )

    failure_df.to_csv(
        failure_path,
        index=False,
    )

    # -----------------------------------------------------
    # SEGMENTATION METHOD REPORT
    # -----------------------------------------------------

    segmentation_df = pd.DataFrame(
        all_segmentation_records
    )

    segmentation_report_path = (
        RESULTS_DIR
        / "segmentation_method_report.csv"
    )

    segmentation_df.to_csv(
        segmentation_report_path,
        index=False,
    )

    print()

    print("=" * 70)

    print(
        "FEATURE EXTRACTION SUMMARY"
    )

    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()

    print(
        f"Total failures: "
        f"{len(all_failures)}"
    )

    if not segmentation_df.empty:

        print()

        print("=" * 70)

        print(
            "SEGMENTATION METHOD COUNTS"
        )

        print("=" * 70)

        print(
            segmentation_df[
                "segmentation_method"
            ]
            .value_counts()
            .to_string()
        )

    print()

    print(
        "Fallback preview folder:"
    )

    print(
        FALLBACK_PREVIEW_DIR
    )

    print()

    print(
        "Feature extraction completed."
    )


if __name__ == "__main__":
    main()