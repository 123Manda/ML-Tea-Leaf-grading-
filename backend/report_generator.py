from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# =========================================================
# HELPERS
# =========================================================

def format_grade(value):

    if not value:
        return "N/A"

    return str(value).replace(
        "_",
        " ",
    )


def format_datetime(value):

    if value is None:
        return "N/A"

    if hasattr(
        value,
        "strftime",
    ):
        return value.strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

    return str(value)


# =========================================================
# PDF GENERATOR
# =========================================================

def generate_prediction_pdf(
    prediction_record,
    image_bytes=None,
):

    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=(
            "Smart Tea Leaf Grading Report"
        ),
        author=(
            "ML Powered Tea Factory "
            "Management System"
        ),
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor(
            "#1F5E32"
        ),
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor(
            "#667268"
        ),
        spaceAfter=18,
    )

    section_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor(
            "#245C33"
        ),
        spaceBefore=10,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "NormalText",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor(
            "#333333"
        ),
    )

    note_style = ParagraphStyle(
        "NoteText",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor(
            "#666666"
        ),
        spaceBefore=8,
    )

    story = []


    # =====================================================
    # HEADER
    # =====================================================

    story.append(
        Paragraph(
            "Smart Tea Leaf Grading Report",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "ML Powered Tea Factory Management System"
            "<br/>"
            "OpenCV Feature Extraction + "
            "XGBoost Classification",
            subtitle_style,
        )
    )


    # =====================================================
    # IMAGE
    # =====================================================

    if image_bytes:

        try:

            image_buffer = BytesIO(
                image_bytes
            )

            report_image = Image(
                image_buffer
            )

            max_width = 90 * mm
            max_height = 70 * mm

            image_width = (
                report_image.imageWidth
            )

            image_height = (
                report_image.imageHeight
            )

            scale = min(
                max_width / image_width,
                max_height / image_height,
                1,
            )

            report_image.drawWidth = (
                image_width * scale
            )

            report_image.drawHeight = (
                image_height * scale
            )

            report_image.hAlign = "CENTER"

            story.append(
                report_image
            )

            story.append(
                Spacer(
                    1,
                    10,
                )
            )

        except Exception:

            story.append(
                Paragraph(
                    "Uploaded image could not "
                    "be rendered in this report.",
                    note_style,
                )
            )


    # =====================================================
    # PREDICTION SUMMARY
    # =====================================================

    story.append(
        Paragraph(
            "Prediction Summary",
            section_style,
        )
    )

    summary_data = [
        [
            "Prediction ID",
            str(
                prediction_record.get(
                    "_id",
                    "N/A",
                )
            ),
        ],
        [
            "Image Name",
            prediction_record.get(
                "image_name",
                "N/A",
            ),
        ],
        [
            "Predicted Grade",
            format_grade(
                prediction_record.get(
                    "predicted_grade"
                )
            ),
        ],
        [
            "Confidence",
            (
                f"{prediction_record.get(
                    'confidence_percent',
                    0
                )}%"
            ),
        ],
        [
            "Segmentation Method",
            prediction_record.get(
                "segmentation_method",
                "N/A",
            ),
        ],
        [
            "Model",
            "XGBoost",
        ],
        [
            "Generated At",
            format_datetime(
                prediction_record.get(
                    "created_at"
                )
            ),
        ],
    ]

    summary_table = Table(
        summary_data,
        colWidths=[
            55 * mm,
            115 * mm,
        ],
    )

    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor(
                        "#EEF5EF"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (0, -1),
                    colors.HexColor(
                        "#244C2E"
                    ),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (1, 0),
                    (1, -1),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#D8E4DA"
                    ),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(
        summary_table
    )

    story.append(
        Spacer(
            1,
            12,
        )
    )


    # =====================================================
    # CLASS PROBABILITIES
    # =====================================================

    story.append(
        Paragraph(
            "Class Probabilities",
            section_style,
        )
    )

    probabilities = (
        prediction_record.get(
            "probabilities",
            {},
        )
    )

    probability_data = [
        [
            "Class",
            "Probability",
        ]
    ]

    for class_name in [
        "Best",
        "Below_Best",
        "Poor",
    ]:

        probability = float(
            probabilities.get(
                class_name,
                0,
            )
        )

        probability_data.append(
            [
                format_grade(
                    class_name
                ),
                f"{probability * 100:.2f}%",
            ]
        )

    probability_table = Table(
        probability_data,
        colWidths=[
            85 * mm,
            85 * mm,
        ],
    )

    probability_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#245C33"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#D8E4DA"
                    ),
                ),
                (
                    "ALIGN",
                    (1, 1),
                    (1, -1),
                    "RIGHT",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(
        probability_table
    )

    story.append(
        Spacer(
            1,
            12,
        )
    )


    # =====================================================
    # LEAF MEASUREMENTS
    # =====================================================

    story.append(
        Paragraph(
            "Leaf Measurements",
            section_style,
        )
    )

    measurements = (
        prediction_record.get(
            "leaf_measurements",
            {},
        )
    )

    measurement_data = [
        [
            "Measurement",
            "Value",
        ],
        [
            "Length",
            (
                f"{measurements.get(
                    'length_px',
                    'N/A'
                )} px"
            ),
        ],
        [
            "Width",
            (
                f"{measurements.get(
                    'width_px',
                    'N/A'
                )} px"
            ),
        ],
        [
            "Area",
            (
                f"{measurements.get(
                    'area_px2',
                    'N/A'
                )} px²"
            ),
        ],
        [
            "Perimeter",
            (
                f"{measurements.get(
                    'perimeter_px',
                    'N/A'
                )} px"
            ),
        ],
        [
            "Aspect Ratio",
            str(
                measurements.get(
                    "aspect_ratio",
                    "N/A",
                )
            ),
        ],
    ]

    measurement_table = Table(
        measurement_data,
        colWidths=[
            85 * mm,
            85 * mm,
        ],
    )

    measurement_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#245C33"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#D8E4DA"
                    ),
                ),
                (
                    "ALIGN",
                    (1, 1),
                    (1, -1),
                    "RIGHT",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(
        measurement_table
    )


    # =====================================================
    # RESEARCH NOTE
    # =====================================================

    story.append(
        Spacer(
            1,
            14,
        )
    )

    story.append(
        Paragraph(
            "<b>Processing Method:</b> "
            "The uploaded tea leaf image was segmented "
            "using the OpenCV-based segmentation pipeline. "
            "Color, shape, size, and texture features were "
            "extracted and classified using the selected "
            "XGBoost model.",
            normal_style,
        )
    )

    story.append(
        Paragraph(
            "<b>Measurement Note:</b> "
            "Length, width, area, and perimeter values "
            "are image-space pixel measurements. "
            "They should not be interpreted as physical "
            "millimetres or centimetres without camera "
            "calibration or a known scale reference.",
            note_style,
        )
    )


    # =====================================================
    # BUILD PDF
    # =====================================================

    document.build(
        story
    )

    pdf_buffer.seek(
        0
    )

    return pdf_buffer