"""Generate Review 3 PowerPoint Presentation and export to PDF.

Follows formal academic presentation standards for MIT Anna University:
- 16:9 widescreen layout
- Structured slide cards, tables, and metric callouts
- High-resolution embedded engineering diagrams and output plots
- No emojis, clean formal styling, consistent deep navy / slate gray palette
- Automatic native PDF export via PowerPoint COM
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pptx
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PPTX_PATH = OUT_DIR / "Review_3_Presentation.pptx"
PDF_PATH = OUT_DIR / "Review_3_Presentation.pdf"

# Formal Color Palette
NAVY = RGBColor(15, 41, 66)       # #0F2942
SLATE_BLUE = RGBColor(30, 64, 175) # #1E40AF
DARK_GRAY = RGBColor(30, 41, 59)   # #1E293B
MUTED_GRAY = RGBColor(71, 85, 105) # #475569
LIGHT_BG = RGBColor(248, 250, 252) # #F8FAFC
CARD_BG = RGBColor(241, 245, 249)  # #F1F5F9
BORDER_COL = RGBColor(203, 213, 225) # #CBD5E1
WHITE = RGBColor(255, 255, 255)
GREEN = RGBColor(4, 120, 87)       # #047857
LIGHT_GREEN = RGBColor(240, 253, 244) # #F0FDF4


def create_base_presentation() -> pptx.Presentation:
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    return prs


def add_slide_header(slide, title_text: str, subtitle_text: str) -> None:
    """Add formal header banner to slide."""
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.95))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_title = tf.paragraphs[0]
    p_title.text = title_text
    p_title.font.name = "Calibri"
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = NAVY

    p_sub = tf.add_paragraph()
    p_sub.text = subtitle_text
    p_sub.font.name = "Calibri"
    p_sub.font.size = Pt(11.5)
    p_sub.font.color.rgb = MUTED_GRAY

    # Top accent line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.35), Inches(11.733), Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = SLATE_BLUE
    line.line.color.rgb = SLATE_BLUE


def add_slide_footer(slide, slide_num: int, total_slides: int = 16) -> None:
    """Add formal footer to slide."""
    # Bottom separator line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(7.0), Inches(11.733), Inches(0.015))
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER_COL
    line.line.color.rgb = BORDER_COL

    footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.05), Inches(10.0), Inches(0.35))
    tf = footer_box.text_frame
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = "Review 3 | Wearable Fall Detection & Severity System | MIT Anna University"
    p.font.name = "Calibri"
    p.font.size = Pt(9.5)
    p.font.color.rgb = MUTED_GRAY

    num_box = slide.shapes.add_textbox(Inches(11.0), Inches(7.05), Inches(1.533), Inches(0.35))
    tf_num = num_box.text_frame
    tf_num.margin_left = tf_num.margin_top = tf_num.margin_right = tf_num.margin_bottom = 0
    p_num = tf_num.paragraphs[0]
    p_num.text = f"{slide_num} / {total_slides}"
    p_num.alignment = PP_ALIGN.RIGHT
    p_num.font.name = "Calibri"
    p_num.font.size = Pt(9.5)
    p_num.font.color.rgb = MUTED_GRAY


def add_card(slide, left: float, top: float, width: float, height: float, title: str, items: list[str], bg_color=CARD_BG, border_color=BORDER_COL, title_color=NAVY):
    """Add a structured content card with bullet points."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1.2)

    tb = slide.shapes.add_textbox(Inches(left + 0.15), Inches(top + 0.12), Inches(width - 0.3), Inches(height - 0.24))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_title = tf.paragraphs[0]
    p_title.text = title
    p_title.font.name = "Calibri"
    p_title.font.size = Pt(12.5)
    p_title.font.bold = True
    p_title.font.color.rgb = title_color
    p_title.space_after = Pt(6)

    for item in items:
        p = tf.add_paragraph()
        p.text = f"- {item}"
        p.font.name = "Calibri"
        p.font.size = Pt(10.0)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(3.5)
        p.line_spacing = 1.15


def add_table_shape(slide, left: float, top: float, width: float, height: float, headers: list[str], rows: list[list[str]], col_widths: list[float] | None = None):
    """Add a formal clean academic table."""
    num_rows = len(rows) + 1
    num_cols = len(headers)
    table_shape = slide.shapes.add_table(num_rows, num_cols, Inches(left), Inches(top), Inches(width), Inches(height))
    table = table_shape.table

    if col_widths and len(col_widths) == num_cols:
        for idx, w in enumerate(col_widths):
            table.columns[idx].width = Inches(w)

    # Style Header Row
    for idx, header in enumerate(headers):
        cell = table.cell(0, idx)
        cell.text = header
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            p.font.name = "Calibri"
            p.font.size = Pt(10)
            p.font.bold = True
            p.font.color.rgb = WHITE

    # Style Body Rows
    for row_idx, row_data in enumerate(rows):
        is_even = row_idx % 2 == 0
        row_bg = CARD_BG if is_even else WHITE
        for col_idx, val in enumerate(row_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = row_bg
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
                p.font.name = "Calibri"
                p.font.size = Pt(9.5)
                p.font.color.rgb = DARK_GRAY


# ==============================================================================
# SLIDE BUILDERS
# ==============================================================================

def build_slide_1_title(prs) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Top institutional banner
    banner = slide.shapes.add_textbox(Inches(0.8), Inches(0.6), Inches(11.733), Inches(1.2))
    tf_b = banner.text_frame
    tf_b.word_wrap = True
    p1 = tf_b.paragraphs[0]
    p1.text = "DEPARTMENT OF INFORMATION TECHNOLOGY"
    p1.alignment = PP_ALIGN.CENTER
    p1.font.name = "Calibri"
    p1.font.size = Pt(17)
    p1.font.bold = True
    p1.font.color.rgb = NAVY

    p2 = tf_b.add_paragraph()
    p2.text = "MADRAS INSTITUTE OF TECHNOLOGY, ANNA UNIVERSITY"
    p2.alignment = PP_ALIGN.CENTER
    p2.font.name = "Calibri"
    p2.font.size = Pt(15)
    p2.font.bold = True
    p2.font.color.rgb = SLATE_BLUE

    # Review Badge & Academic details
    badge_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.4), Inches(2.0), Inches(2.5), Inches(0.45))
    badge_box.fill.solid()
    badge_box.fill.fore_color.rgb = CARD_BG
    badge_box.line.color.rgb = SLATE_BLUE
    badge_box.line.width = Pt(1.5)
    tf_badge = badge_box.text_frame
    p_badge = tf_badge.paragraphs[0]
    p_badge.text = "THIRD REVIEW"
    p_badge.alignment = PP_ALIGN.CENTER
    p_badge.font.name = "Calibri"
    p_badge.font.size = Pt(12)
    p_badge.font.bold = True
    p_badge.font.color.rgb = NAVY

    sub_info = slide.shapes.add_textbox(Inches(0.8), Inches(2.55), Inches(11.733), Inches(0.4))
    tf_sub = sub_info.text_frame
    p_sub = tf_sub.paragraphs[0]
    p_sub.text = "IT5712 - PROJECT 1   |   ACADEMIC YEAR: 2023 - 2027"
    p_sub.alignment = PP_ALIGN.CENTER
    p_sub.font.name = "Calibri"
    p_sub.font.size = Pt(11)
    p_sub.font.bold = True
    p_sub.font.color.rgb = MUTED_GRAY

    # Main Project Title
    title_box = slide.shapes.add_textbox(Inches(1.0), Inches(3.1), Inches(11.333), Inches(1.3))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = "Wearable Fall Detection and Severity Classification\nSystem for Elderly People Using Deep Learning"
    p_title.alignment = PP_ALIGN.CENTER
    p_title.font.name = "Calibri"
    p_title.font.size = Pt(25)
    p_title.font.bold = True
    p_title.font.color.rgb = NAVY
    p_title.line_spacing = 1.25

    # Separator Line
    sep = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(2.5), Inches(4.7), Inches(8.333), Inches(0.025))
    sep.fill.solid()
    sep.fill.fore_color.rgb = SLATE_BLUE
    sep.line.color.rgb = SLATE_BLUE

    # Team Members Card (Left)
    add_card(
        slide,
        left=1.5, top=5.0, width=4.8, height=1.9,
        title="TEAM MEMBERS",
        items=[
            "1. Sarath R (2023506065)",
            "2. Ajay Kumar S (2023506105)",
            "3. Arjun A (2023506063)",
            "4. Kris Virshak (2023506124)",
        ],
        title_color=NAVY
    )

    # Guided By Card (Right)
    add_card(
        slide,
        left=7.0, top=5.0, width=4.8, height=1.9,
        title="GUIDED BY",
        items=[
            "Dr. S. Umamaheswari,",
            "Associate Professor,",
            "Department of Information Technology,",
            "Madras Institute of Technology, Anna University",
        ],
        title_color=NAVY
    )


def build_slide_2_scope(prs) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide,
        "Review 3 Scope and Key Deliverables",
        "Transition from Review 2 prototype components to fully deployed, verified end-to-end edge architecture."
    )

    w = 5.7
    h = 2.1
    top1 = 1.6
    top2 = 3.9

    add_card(
        slide, 0.8, top1, w, h,
        "1. Multi-Task Fall & Severity Deep Learning",
        [
            "Unified deep CNN-LSTM backbone trained end-to-end directly on raw IMU windows.",
            "Dual-head architecture: Binary Sigmoid (Fall vs ADL) + Softmax (Mild vs Severe).",
            "Custom Masked Loss excludes non-fall ADLs and sensor outliers from severity gradients.",
            "Severity Macro-F1 = 0.8660 (Mild: 0.8346, Severe: 0.8973; 0.9020 recording-level)."
        ]
    )

    add_card(
        slide, 6.8, top1, w, h,
        "2. Strict Operating Threshold Policy (Decision D2)",
        [
            "Shifted from arbitrary 0.50 threshold to rigorous validation-selected operating rule.",
            "Threshold selected strictly on validation split to maximize F2, frozen to test.",
            "Selected threshold 0.25 on CNN-LSTM reduces missed falls by 33% (9 to 6).",
            "High test sensitivity: 0.9950 recall and 0.9879 F2 with 17.74 false alarms/hour."
        ]
    )

    add_card(
        slide, 0.8, top2, w, h,
        "3. Edge Quantization & Microcontroller C Export",
        [
            "Evaluated 4 TFLite variants: Float32, Dynamic-Range, Float16, Full-Integer INT8.",
            "Dynamic-Range chosen via strict gating: 111.0 KB (2.12x comp), 0.000 F2/recall drop.",
            "Single-window latency: 0.026 ms on 1 thread (enables >1,000x real-time speed).",
            "Exported C byte array header (model_data.h) for direct ESP32 firmware integration."
        ]
    )

    add_card(
        slide, 6.8, top2, w, h,
        "4. Real-Time Streaming Engine & Caregiver UI",
        [
            "64-sample circular ring buffer @ 20 Hz, evaluating every 16 samples (0.8 s latency).",
            "Multi-stage noise filtering: N-consecutive positive windows + 10.0 s alert cooldown.",
            "Bitwise mathematical parity verified: max abs diff <= 1e-6 vs offline TFLite.",
            "Interactive Streamlit dashboard (app/demo.py) with real-time audio-visual alerts."
        ]
    )

    # Bottom KPI Summary Bar
    kpi_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.15), Inches(11.733), Inches(0.7))
    kpi_box.fill.solid()
    kpi_box.fill.fore_color.rgb = LIGHT_GREEN
    kpi_box.line.color.rgb = GREEN
    kpi_box.line.width = Pt(1.5)
    tf_kpi = kpi_box.text_frame
    p_kpi = tf_kpi.paragraphs[0]
    p_kpi.text = "Dataset: 84,123 Windows   |   Edge Model: 111.0 KB (TFLite)   |   Test Recall: 0.9950   |   Severity Macro-F1: 0.8660   |   Test Suite: 54/54 Passed"
    p_kpi.alignment = PP_ALIGN.CENTER
    p_kpi.font.name = "Calibri"
    p_kpi.font.size = Pt(11.5)
    p_kpi.font.bold = True
    p_kpi.font.color.rgb = GREEN

    add_slide_footer(slide, 2)


def build_slide_3_architecture(prs) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide,
        "Updated End-to-End System Architecture",
        "Complete deployed pipeline from raw 6-axis wearable motion data to real-time caregiver alerts."
    )

    # Insert system architecture diagram
    img_path = ROOT / "docs" / "presentation" / "figures" / "review3_system_architecture.png"
    if img_path.exists():
        slide.shapes.add_picture(str(img_path), Inches(0.8), Inches(1.5), Inches(11.733), Inches(3.6))

    # Architecture narrative cards below
    add_card(
        slide, 0.8, 5.25, 3.75, 1.6,
        "Signal Conditioning",
        [
            "6 retained channels: acc1_x/y/z, gyro_x/y/z.",
            "Butterworth 5 Hz LPF removes tremor noise.",
            "200 Hz -> 20 Hz decimation (10x reduction).",
            "StandardScaler fit exclusively on train split."
        ]
    )

    add_card(
        slide, 4.8, 5.25, 3.75, 1.6,
        "Deep Multi-Task Backbone",
        [
            "Sliding window: 64 samples (3.2 s), 75% overlap.",
            "2 Conv1D blocks extract spatial transients.",
            "Unrolled LSTM (64 units) models sequence.",
            "Dual heads output fall prob & K=2 severity."
        ]
    )

    add_card(
        slide, 8.8, 5.25, 3.733, 1.6,
        "Edge Deployment & Alert",
        [
            "Dynamic-range quantized TFLite (111.0 KB).",
            "Ring buffer evaluates every 16 samples (0.8 s).",
            "Cooldown timer prevents repetitive alert spam.",
            "SimulatedCaregiverSink logs alert dispatch."
        ]
    )

    add_slide_footer(slide, 3)


def build_slide_4_dataset(prs) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide,
        "Dataset Partitioning & Preprocessing Rigor",
        "SisFall subject-wise split ensuring zero participant leakage and standardized signal conditioning."
    )

    # Left: Tables
    headers_split = ["Split", "Subjects", "Recordings", "Windows", "ADL", "Fall"]
    rows_split = [
        ["Train", "26", "2,942", "55,423", "50,939", "4,484"],
        ["Validation", "6", "829", "15,073", "13,573", "1,500"],
        ["Test", "6", "734", "13,627", "12,427", "1,200"],
        ["Total", "38", "4,505", "84,123", "76,939", "7,184"],
    ]
    add_table_shape(slide, 0.8, 1.55, 6.2, 1.8, headers_split, rows_split, [1.0, 1.0, 1.1, 1.1, 1.0, 1.0])

    headers_prep = ["Parameter", "Value / Design Decision"]
    rows_prep = [
        ["Original Sampling Rate", "200 Hz (from SisFall DAQ)"],
        ["Downsampled Rate", "20 Hz (efficient edge processing)"],
        ["Digital Filter", "Butterworth Low-Pass, 2nd order, 5 Hz cutoff"],
        ["Retained Channels", "6 (acc1_x/y/z, gyro_x/y/z)"],
        ["Dropped Channels", "3 (redundant second accelerometer acc2)"],
        ["Standardization", "StandardScaler (mean/std fit on train split only)"],
        ["Window / Stride", "64 samples (3.2 s), stride 16 (0.8 s, 75% overlap)"],
    ]
    add_table_shape(slide, 0.8, 3.6, 6.2, 3.2, headers_prep, rows_prep, [2.4, 3.8])

    # Right: Class balance image
    img_path = ROOT / "docs" / "reports" / "project_report" / "report_assets" / "class_balance.png"
    if img_path.exists():
        slide.shapes.add_picture(str(img_path), Inches(7.3), Inches(1.55), Inches(5.2), Inches(3.2))

    add_card(
        slide, 7.3, 4.95, 5.2, 1.85,
        "Dataset Integrity Highlights",
        [
            "Strict 26 / 6 / 6 subject-wise split (Decision D3).",
            "Zero subject overlap between training and evaluation.",
            "Extreme imbalance handled naturally (91.5% ADL vs 8.5% Fall).",
            "Training balance: Offline class-weighting and balanced sampling.",
            "All test evaluations conducted on unseen subjects."
        ]
    )

    add_slide_footer(slide, 4)


def build_slide_5_windowing(prs) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide,
        "Impact-Centered Windowing & Fall Signal Dynamics",
        "Formulating 3.2 s window boundaries centered on peak impact to separate falls from ordinary activities."
    )

    img_win = ROOT / "docs" / "reports" / "project_report" / "report_assets" / "preprocessing_windowing_example.png"
    if img_win.exists():
        slide.shapes.add_picture(str(img_win), Inches(0.8), Inches(1.55), Inches(5.7), Inches(2.6))

    img_wave = ROOT / "docs" / "reports" / "project_report" / "report_assets" / "sample_waveforms_fall_vs_adl.png"
    if img_wave.exists():
        slide.shapes.add_picture(str(img_wave), Inches(6.8), Inches(1.55), Inches(5.7), Inches(2.6))

    add_card(
        slide, 0.8, 4.35, 5.7, 2.45,
        "Impact-Centered Fall Labeling",
        [
            "Signal Magnitude Vector: SMV(t) = sqrt(ax^2 + ay^2 + az^2).",
            "Peak Impact Detection: The instant of maximum acceleration in each fall trial serves as the anchor point.",
            "Impact Window Labeling: Only windows overlapping the peak impact point are labeled as positive fall (label=1).",
            "Background Rejection: Pre-fall gait instability and post-fall motionless periods are preserved as background.",
            "Overlap Strategy: 75% overlap (16-sample stride) ensures impacts near window boundaries are never missed."
        ]
    )

    add_card(
        slide, 6.8, 4.35, 5.7, 2.45,
        "Observed Kinematic Characteristics",
        [
            "Impulsive Acceleration Spikes: Falls produce sharp acceleration transients (>11g in SisFall) during floor impact.",
            "Rotational Kinematics: Rapid angular velocity surges (>300 deg/s) accompany loss of balance prior to collision.",
            "ADL Regularity: Walking, sitting, and transitions exhibit periodic, low-amplitude oscillations (peak SMV < 2.0g).",
            "Dynamic Contrast: The temporal CNN-LSTM architecture captures both local impact shock and pre/post temporal context."
        ]
    )

    add_slide_footer(slide, 5)


def build_slide_6_multitask_arch(prs) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide,
        "Multi-Task CNN-LSTM Model Architecture",
        "Joint representation learning for simultaneous fall detection and injury severity classification."
    )

    img_arch = ROOT / "docs" / "presentation" / "figures" / "review3_multitask_architecture.png"
    if img_arch.exists():
        slide.shapes.add_picture(str(img_arch), Inches(0.8), Inches(1.55), Inches(11.733), Inches(3.6))

    add_card(
        slide, 0.8, 5.3, 3.75, 1.55,
        "Convolutional Feature Extractor",
        [
            "Block 1: Conv1D (32, k=3) + BN + ReLU + MaxPool.",
            "Block 2: Conv1D (64, k=3) + BN + ReLU + MaxPool.",
            "Extracts localized high-frequency shock transients."
        ]
    )

    add_card(
        slide, 4.8, 5.3, 3.75, 1.55,
        "Temporal Sequence Modeling",
        [
            "LSTM: 64 hidden units, return_sequences=True.",
            "Unrolled LSTM (unroll=True) for edge TFLite compatibility.",
            "GlobalAveragePooling1D aggregates temporal trajectory."
        ]
    )

    add_card(
        slide, 8.8, 5.3, 3.733, 1.55,
        "Multi-Task Prediction Heads",
        [
            "Shared Dense trunk (32 units, ReLU) + Dropout (0.3).",
            "Head 1: Dense(1, Sigmoid) for binary fall probability.",
            "Head 2: Dense(2, Softmax) for Mild vs Severe classes."
        ]
    )

    add_slide_footer(slide, 6)


def build_slide_7_masked_loss(prs) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide,
        "Masked Severity Loss Formulation & Training",
        "Preventing gradient interference from non-fall windows and sensor outliers during joint optimization."
    )

    # Left: Mathematical formulation
    add_card(
        slide, 0.8, 1.55, 6.0, 5.25,
        "Loss Formulation & Gradient Masking",
        [
            "Challenge: 91.5% of windows are ADLs with no severity label. Sensor outliers (-1) from gyro saturation must not pollute severity weights.",
            "Total Multi-Task Objective Function:\n  L_total = L_binary + lambda * L_severity_masked",
            "Binary Fall Loss (L_binary):\n  Standard Binary Cross-Entropy evaluated across all training windows.",
            "Masked Severity Loss (L_severity_masked):\n  Evaluated strictly on confident fall windows:\n  mask = (y_binary == 1) & (y_severity in {0, 1})\n  L_severity_masked = sum(mask * CCE(y_sev, p_sev)) / (sum(mask) + eps)",
            "Zero Gradient Backpropagation: ADL windows (y_bin=0) and extreme gyro outliers (y_sev=-1) contribute 0.0 loss.",
            "Optimization: Adam (lr=1e-3), batch size 64, lambda=1.0.",
            "Graph Acceleration: Compiled via @tf.function, reducing full training epoch time to 103.4 seconds."
        ]
    )

    # Right: Training curves
    img_curves = ROOT / "docs" / "reports" / "project_report" / "report_assets" / "cnn_lstm_training_curves.png"
    if img_curves.exists():
        slide.shapes.add_picture(str(img_curves), Inches(7.1), Inches(1.55), Inches(5.4), Inches(3.2))

    add_card(
        slide, 7.1, 4.95, 5.4, 1.85,
        "Training Behavior Observations",
        [
            "Smooth Convergence: Loss decreases stably without divergence.",
            "Validation Accuracy: Stabilizes above 99.4% after early epochs.",
            "Generalization: Train and validation losses track closely.",
            "Balanced Sampling: Applied X_train_balanced (59,907 windows) ensuring high gradient signal for minority fall class."
        ]
    )

    add_slide_footer(slide, 7)


def build_slide_8_severity_results(prs) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide,
        "Multi-Task Severity Classification Results",
        "Quantitative evaluation on held-out test subjects against statistical baselines and anomaly analysis."
    )

    headers = ["Classifier / Model", "Feature Representation", "Macro-F1", "Mild F1", "Severe F1", "Severity Gain"]
    rows = [
        ["Majority Class Baseline", "None (Constant Severe Predictor)", "0.3794", "0.0000", "0.7588", "Reference Baseline"],
        ["Logistic Regression", "Signal Features (Peak Acc + Gyro)", "0.6109", "0.5966", "0.6251", "+0.2315"],
        ["Multi-Task CNN-LSTM (Ours)", "End-to-End Standardized IMU Windows", "0.8660", "0.8346", "0.8973", "+0.4866 vs Majority"],
    ]
    add_table_shape(slide, 0.8, 1.55, 6.2, 1.6, headers, rows, [1.8, 1.8, 0.8, 0.8, 0.8, 1.2])

    # Left Bottom: Key observations
    add_card(
        slide, 0.8, 3.35, 6.2, 3.45,
        "Severity Classification Performance",
        [
            "High Test Discrimination: Severity Macro-F1 reaches 0.8660 on confident fall windows (Mild F1: 0.8346, Severe F1: 0.8973).",
            "Test Window Confusion Matrix (n=1,176):\n  - True Mild (Cluster 0): 376 Correct, 81 Misclassified (82.3% recall)\n  - True Severe (Cluster 1): 651 Correct, 68 Misclassified (90.5% recall)\n  - Overall Severity Accuracy: 87.3% on unseen test subjects.",
            "Recording-Level Severity: Aggregating windows per recording achieves Macro-F1 = 0.9020 (95 Mild, 175 Severe correctly classified).",
            "Outlier Safety (Isolation Forest -1): All 24 gyro-outlier fall windows are detected as falls by the binary branch with 100% recall.",
            "Scientific Honesty: Ground-truth severity was generated via unsupervised K-Means on signal features, so deep learning provides learned feature extraction without manual thresholds."
        ]
    )

    # Right: PCA Plot
    img_pca = ROOT / "docs" / "presentation" / "figures" / "severity" / "kmeans_severity_test_pca.png"
    if img_pca.exists():
        slide.shapes.add_picture(str(img_pca), Inches(7.3), Inches(1.55), Inches(5.2), Inches(3.4))

    add_card(
        slide, 7.3, 5.15, 5.2, 1.65,
        "Cluster Separation & Anomaly Screening",
        [
            "K=2 PCA feature space shows distinct cluster separation.",
            "Mild cluster corresponds to lower-energy knees/hands falls.",
            "Severe cluster corresponds to high-impact backward/syncope falls.",
            "Isolation Forest detects unusual anomalies with ~2% flag rate."
        ]
    )

    add_slide_footer(slide, 8)


def build_slide_9_model_comparison(prs) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide,
        "Cross-Model Benchmarking & Deployment Decision",
        "Evaluating candidate models under Decision D2 validation-selected thresholding rules."
    )

    headers = ["Model Architecture", "Val Threshold", "Accuracy", "Precision", "Recall", "F1", "F2", "FA/hr", "AUROC", "Parameters", "Severity F1"]
    rows = [
        ["CNN (1D Baseline)", "0.20", "0.9965", "0.9683", "0.9925", "0.9802", "0.9876", "14.12", "0.9998", "33,729", "N/A"],
        ["CNN-LSTM (Binary Final)", "0.25", "0.9960", "0.9606", "0.9950", "0.9775", "0.9879", "17.74", "0.9998", "46,817", "N/A"],
        ["CNN-LSTM + Severity Head", "0.10", "0.9935", "0.9580", "0.9692", "0.9635", "0.9669", "18.47", "0.9994", "46,883", "0.8660"],
        ["FiLM (k=30, Exploratory)", "0.75", "0.9943", "0.9693", "0.9692", "0.9787", "0.9880", "14.03", "0.9998", "46,945", "N/A"],
    ]
    add_table_shape(slide, 0.8, 1.55, 11.733, 1.8, headers, rows, [2.1, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8, 1.2, 1.1])

    # Left Bottom: Decision Rationale
    add_card(
        slide, 0.8, 3.55, 6.2, 3.25,
        "Authoritative Deployment Gating Decision (Task P5)",
        [
            "Gating Protocol: Deploy the multi-task model if its test binary recall did not regress vs baseline CNN-LSTM, else deploy the binary CNN-LSTM.",
            "Recall Trade-off Analysis:\n  - Baseline CNN-LSTM Recall: 0.9950 (only 6 missed falls out of 1,200)\n  - Multi-Task CNN-LSTM Recall: 0.9692 (37 missed falls out of 1,200)\n  - Recall Regression: True (0.0258 drop, representing 31 additional missed falls).",
            "Clinical Safety Priority: In elderly fall monitoring, false alarms cause nuisance, but a missed fall can lead to an unassisted long-lie and fatal complications.",
            "Conclusion: The binary CNN-LSTM remains the authoritative deployed edge model, while the multi-task head is preserved for post-detection severity staging."
        ]
    )

    # Right Bottom: Confusion matrix
    img_cm = ROOT / "docs" / "reports" / "project_report" / "report_assets" / "cnn_lstm_confusion_matrix.png"
    if img_cm.exists():
        slide.shapes.add_picture(str(img_cm), Inches(7.3), Inches(3.55), Inches(5.2), Inches(3.25))

    add_slide_footer(slide, 9)


def build_slide_10_film_evaluation(prs) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide,
        "Scientific Evaluation of FiLM Personalization",
        "Rigorous multi-seed benchmarking and statistical hypothesis testing of the personalization prototype."
    )

    add_card(
        slide, 0.8, 1.55, 6.0, 5.25,
        "FiLM Methodology & Experimental Findings",
        [
            "Architecture: Feature-wise Linear Modulation applied after the 2nd CNN block: h_film = gamma * h + beta.",
            "UserEncoder: Lightweight side-network (2,352 params) trained once on simulated user profiles; per-user adaptation uses k ADL windows without gradient steps.",
            "Calibration Sweep across k in {0, 1, 5, 10, 30} across 3 random seeds:\n  - k=0 (Identity Init): Perfectly reproduces global CNN-LSTM (AUROC=0.9998)\n  - k=1: F1 = 0.9738, F2 = 0.9840, FA/hr = 16.64\n  - k=30: F1 = 0.9787, F2 = 0.9880, FA/hr = 14.03.",
            "Paired Wilcoxon Signed-Rank Hypothesis Test:\n  - Evaluated across n=4 held-out test subjects with falls.\n  - Test Statistic W = 0, p-value = 0.125 (> 0.05 alpha threshold).",
            "Formal Closure (Decision D4):\n  - While FiLM proves that affine modulation can reduce false alarms as k increases, the small test fall cohort prevents claiming statistically significant superiority over the global CNN-LSTM baseline.\n  - The prototype is formally closed as exploratory-inconclusive."
        ]
    )

    # Right: FiLM F2 vs k plot
    img_film = ROOT / "docs" / "presentation" / "figures" / "personalization" / "01_f2_vs_k.png"
    if img_film.exists():
        slide.shapes.add_picture(str(img_film), Inches(7.1), Inches(1.55), Inches(5.4), Inches(3.2))

    add_card(
        slide, 7.1, 4.95, 5.4, 1.85,
        "Statistical Inference Takeaways",
        [
            "Sample Size Limitation: With only 4 fall test subjects, statistical significance cannot be established at alpha=0.05.",
            "Scientific Integrity: Accurately reporting negative or inconclusive findings prevents overclaiming unverified edge gains.",
            "Architectural Soundness: The modular FiLM interface remains available for future multi-center studies with larger cohorts."
        ]
    )

    add_slide_footer(slide, 10)


def build_slide_11_quantization(prs) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide,
        "Edge Model Quantization & Optimization (TFLite)",
        "Benchmarking post-training quantization variants for edge deployment on low-power microcontrollers."
    )

    headers = ["Quantization Variant", "Size (KB)", "Comp Ratio", "Val Threshold", "Agreement", "Test Recall", "Test F2", "FA/hr", "Latency (ms)", "Gate Decision"]
    rows = [
        ["Float32 TFLite", "235.6", "1.00x", "0.25", "100.00%", "0.9950", "0.9879", "17.74", "0.063", "PASS"],
        ["Dynamic-Range (Winner)", "111.0", "2.12x", "0.25", "100.00%", "0.9950", "0.9879", "17.74", "0.026", "DEPLOYED"],
        ["Float16 TFLite", "147.8", "1.59x", "0.25", "100.00%", "0.9950", "0.9879", "17.74", "0.062", "PASS"],
        ["Full-Integer INT8", "136.9", "1.72x", "0.30", "99.97%", "0.9950", "0.9879", "17.74", "0.033", "PASS"],
    ]
    add_table_shape(slide, 0.8, 1.55, 11.733, 1.8, headers, rows, [1.9, 0.9, 0.9, 1.1, 1.1, 1.0, 0.9, 0.8, 1.1, 1.3])

    # Left Bottom: Quantization comparison chart
    img_quant = ROOT / "docs" / "presentation" / "figures" / "review3_quantization_comparison.png"
    if img_quant.exists():
        slide.shapes.add_picture(str(img_quant), Inches(0.8), Inches(3.55), Inches(6.2), Inches(3.25))

    # Right Bottom: Gating Protocol Details
    add_card(
        slide, 7.3, 3.55, 5.2, 3.25,
        "Gating Protocol & Variant Selection",
        [
            "Strict Gating Constraints:\n  - F2 drop <= 0.005 vs Keras float32 on full test set.\n  - Recall drop <= 0.010 vs Keras float32 on full test set.\n  - Winner selected as smallest file size passing both gates.",
            "Dynamic-Range Selected as Winner:\n  - Exactly 0.000 drop in F2 and Recall vs Keras float.\n  - 100.0% class agreement across all 13,627 test windows.\n  - Mean absolute probability difference: 0.00003.\n  - 2.12x compression: 235.6 KB -> 111.0 KB.",
            "INT8 Representative Calibration: Full INT8 converted using 500 stratified training windows (8.5% fall, 91.5% ADL).",
            "Single-Window CPU Latency: Measured on 1 thread over 1,000 runs after 50 warmups: median 0.026 ms, p95 0.039 ms."
        ]
    )

    add_slide_footer(slide, 11)


def build_slide_12_esp32_export(prs) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide,
        "Microcontroller Export & Hardware Suitability",
        "Autonomous wearable deployment readiness on resource-constrained embedded microcontrollers."
    )

    add_card(
        slide, 0.8, 1.55, 5.7, 5.25,
        "ESP32 C Byte Array Header Export",
        [
            "Export File: results/quantization/run_v1/model_data.h",
            "C Symbol: const unsigned char g_model_data[] __attribute__((aligned(16)))",
            "Exact Size: const unsigned int g_model_data_len = 113,704 bytes (111.04 KB).",
            "Target Microcontroller: Espressif ESP32-WROOM-32 / ESP32-S3 (240 MHz dual-core Xtensa LX7, 520 KB SRAM).",
            "SRAM Budget Analysis:\n  - Internal SRAM: 520 KB\n  - Model Footprint: 111 KB (~21.3% of available SRAM)\n  - TFLite Arena Buffer: ~48 KB\n  - Total RAM Required: ~159 KB (easily fits without external PSRAM).",
            "Compute Budget Analysis:\n  - CPU latency on desktop: 0.026 ms\n  - Projected ESP32 execution latency: <15 ms\n  - With an 800 ms evaluation stride, the microcontroller remains idle >98% of the time, allowing deep sleep and multi-day battery life."
        ]
    )

    add_card(
        slide, 6.8, 1.55, 5.7, 5.25,
        "Deployment Metadata & Edge Autonomy",
        [
            "Deployment Metadata File: deployed_model_meta.json.",
            "Stored Hyperparameters for Embedded Runtime:\n  - Input Window: 64 samples x 6 channels (3.2 seconds)\n  - Stride: 16 samples (0.8 second decision step)\n  - Sampling Frequency: 20 Hz\n  - Operating Threshold: tau = 0.25 (Decision D2)\n  - Input Quantization Scale / Zero-Point: (0.0, 0)\n  - StandardScaler Parameters: channel mean and standard deviation vector stored for on-device standardization.",
            "Complete Edge Autonomy:\n  - Zero Cloud Dependency: All inference, sliding windowing, and alert logic execute locally on the wearable sensor.\n  - Privacy-Preserving: Motion data never leaves the device; only emergency alert notifications are transmitted."
        ]
    )

    add_slide_footer(slide, 12)


def build_slide_13_streaming_engine(prs) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide,
        "Real-Time Edge Streaming Engine Architecture",
        "Circular ring buffer mechanism with multi-stage noise suppression and caregiver notification sinks."
    )

    img_stream = ROOT / "docs" / "presentation" / "figures" / "review3_streaming_pipeline.png"
    if img_stream.exists():
        slide.shapes.add_picture(str(img_stream), Inches(0.8), Inches(1.55), Inches(11.733), Inches(3.2))

    add_card(
        slide, 0.8, 4.95, 3.75, 1.85,
        "Ring Buffer Mechanics",
        [
            "Sample-by-sample ingestion at 20 Hz.",
            "Capacity: 64 samples (3.2 s).",
            "Warmup: Strictly returns None for t < 3.2 s.",
            "Stride: Evaluates window every 16 samples (0.8 s step latency)."
        ]
    )

    add_card(
        slide, 4.8, 4.95, 3.75, 1.85,
        "Configurable Alert Policy",
        [
            "Validation Threshold: tau = 0.25.",
            "N-Consecutive Filter: Requires 1, 2, or 3 consecutive positive windows.",
            "10.0 s Cooldown Refractory Period prevents duplicate alert spam."
        ]
    )

    add_card(
        slide, 8.8, 4.95, 3.733, 1.85,
        "Notification Sink Interfaces",
        [
            "Abstract AlertSink interface.",
            "ConsoleSink: Real-time terminal logging.",
            "SimulatedCaregiverSink: In-memory timestamped dispatch log.",
            "Integrated with Streamlit demo UI."
        ]
    )

    add_slide_footer(slide, 13)


def build_slide_14_parity_demo(prs) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide,
        "System Parity, Replay Validation & Dashboard",
        "Exact bitwise validation across test trials and interactive live caregiver monitoring dashboard."
    )

    headers = ["Trial ID", "Activity Description", "Category", "Duration", "Windows", "Max Prob", "Alert Status", "Expected Behavior"]
    rows = [
        ["SA02:D01", "Walking", "ADL", "15.0 s", "15", "0.0001", "0 Alerts", "Clean Rejection (Pass)"],
        ["SA02:D07", "Standing to Sitting", "ADL", "15.0 s", "15", "0.0002", "0 Alerts", "Clean Rejection (Pass)"],
        ["SA02:D12", "Stumbling / Tripping", "Hard-Negative", "15.0 s", "15", "0.0001", "0 Alerts", "Clean Recovery (Pass)"],
        ["SA02:D15", "Vertical Jumping", "Hard-Negative", "15.0 s", "15", "0.0435", "0 Alerts", "Suppressed below tau=0.25 (Pass)"],
        ["SA02:F01", "Fall Forward on Knees", "Mild Fall", "15.0 s", "15", "0.9544", "1 Alert (7.15s)", "Immediate Dispatch (Detected)"],
        ["SA02:F03", "Fall Backward on Buttocks", "Severe Fall", "15.0 s", "15", "0.9292", "1 Alert (7.15s)", "Immediate Dispatch (Detected)"],
    ]
    add_table_shape(slide, 0.8, 1.55, 11.733, 2.2, headers, rows, [1.1, 2.2, 1.3, 1.0, 1.0, 1.1, 1.7, 2.3])

    add_card(
        slide, 0.8, 3.95, 5.7, 2.85,
        "Bitwise Offline-Online Parity",
        [
            "Parity Requirement: Streaming probabilities must equal offline batch predictions down to floating point tolerances.",
            "Verification Result: Evaluated across 3 complete test recordings (D07, D15, F01).\n  - Maximum Absolute Probability Difference: <= 1.0e-6.\n  - Window count and temporal step alignment: 100% exact match.",
            "Dynamic Jump Suppression: Jumping causes violent landing impacts (>2.5g), producing a transient spike of 0.0435. Operating threshold tau=0.25 correctly prevents false alarms."
        ]
    )

    add_card(
        slide, 6.8, 3.95, 5.7, 2.85,
        "Interactive Caregiver Dashboard (app/demo.py)",
        [
            "Built with Streamlit for live panel demonstration.",
            "Rolling 6-Channel Waveform Chart: Continuous 3-axis accelerometer and gyroscope displays.",
            "Probability Timeline: Real-time probability line against horizontal threshold boundary (tau=0.25).",
            "Status Badges: MONITORING (Green) -> FALL DETECTED (Pulsing Red Alert).",
            "Caregiver Notification Log: Timestamped emergency dispatch table.",
            "Replay Speeds: 1x real-time, 5x, 20x, or custom IMU CSV upload."
        ]
    )

    add_slide_footer(slide, 14)


def build_slide_15_testing_limitations(prs) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide,
        "Verification Rigor, Limitations & Future Scope",
        "Automated test coverage, critical scientific boundaries, and the next-phase deployment roadmap."
    )

    add_card(
        slide, 0.8, 1.55, 3.75, 5.25,
        "Automated Verification Suite",
        [
            "54 out of 54 Pytest Tests Passing:",
            "test_streaming.py (8/8):\n  - Warmup silence for t < 3.2 s\n  - Offline-online parity (3 trials)\n  - Alert policy N-consecutive\n  - 10 s alert cooldown\n  - Streamlit headless AppTest.",
            "test_quantization.py (6/6):\n  - TFLite file existence & size\n  - Tensor shapes (1, 64, 6)\n  - 50-window parity vs Keras float\n  - Metadata schema & C header.",
            "test_severity_multitask.py (8/8):\n  - Gradient masking for ADLs\n  - Extreme gyro outlier masking\n  - Class-balanced weights.",
            "test_film_personalization.py (16/16)\n  - Frozen backbone integrity\n  - Identity init at k=0.",
            "test_sisfall_pipeline.py (16/16)\n  - Subject split & windowing."
        ]
    )

    add_card(
        slide, 4.8, 1.55, 3.75, 5.25,
        "Honest Limitations Disclosures",
        [
            "Laboratory Fall Domain Shift:\n  - SisFall falls were performed by young adult participants onto safety mattresses under controlled laboratory conditions.\n  - Real-world elderly falls involve pre-fall gait instability, different impact damping, and prolonged unassisted lying.",
            "Small Test Fall Cohort (n=6 subjects):\n  - Only 4 test subjects had falls; 2 were ADL-only.\n  - Sample size is insufficient for formal cross-subject statistical power (Wilcoxon p=0.125).",
            "Clustering-Derived Severity Ground Truth:\n  - Severity labels were obtained via unsupervised K-Means on signal-derived features, not hospital injury records.",
            "Extreme Gyro Outliers (-1):\n  - Masked from severity loss due to sensor saturation; preserved in binary detection."
        ]
    )

    add_card(
        slide, 8.8, 1.55, 3.733, 5.25,
        "Future Development Roadmap",
        [
            "Physical ESP32 Wearable Firmware:\n  - Compile exported model_data.h with TensorFlow Lite for Microcontrollers (TFLM).\n  - Integrate with MPU-6050 / ICM-42688 IMU over I2C/SPI.",
            "Battery & Power Optimization:\n  - Implement hardware interrupts on high acceleration thresholds to wake MCU from deep sleep.",
            "Clinical Pilot Validation:\n  - Partner with geriatric care facilities to record real-world movement and natural stumble events.",
            "Cellular / BLE Dispatch:\n  - Connect alert sinks to LTE-M / NB-IoT modules for direct SMS and caregiver mobile push alerts."
        ]
    )

    add_slide_footer(slide, 15)


def build_slide_16_conclusion(prs) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_header(
        slide,
        "Conclusion and Project Completion",
        "Delivering a complete, validated, and quantized deep learning system for edge fall detection."
    )

    # Deliverables recap table
    headers = ["Milestone Component", "Review 2 Baseline Status", "Review 3 Delivered Outcome", "Performance / Impact"]
    rows = [
        ["Model Architecture", "Binary CNN-LSTM baseline", "Multi-Task CNN-LSTM (Binary + Severity)", "Severity Macro-F1 = 0.8660 (87.3% accuracy)"],
        ["Operating Policy", "Uncalibrated 0.50 threshold", "Validation-Selected tau = 0.25 (Decision D2)", "Test Recall = 0.9950 (F2 = 0.9879, 6 misses / 1200)"],
        ["FiLM Personalization", "Early proof-of-concept prototype", "Multi-seed statistical evaluation", "Closed as inconclusive (p=0.125, n=4 test cohort)"],
        ["Edge Quantization", "Uncompressed float models only", "Dynamic-Range TFLite Deployment", "111.0 KB size (2.12x comp), 0.026 ms CPU latency"],
        ["Embedded Readiness", "No firmware export", "C array header exported (model_data.h)", "Fits in 21% ESP32 SRAM; zero cloud dependency"],
        ["Streaming Simulation", "Offline batch processing only", "Real-time circular ring buffer engine", "0.8 s step latency; bitwise parity (<= 1e-6 diff)"],
        ["Interactive Interface", "Command line scripts only", "Streamlit Web Caregiver Dashboard", "Real-time chart streaming, alert cooldown & logs"],
        ["Verification Suite", "Partial unit tests", "54 / 54 Pytest tests passing cleanly", "Zero leakage across 38 subjects, complete git sync"],
    ]
    add_table_shape(slide, 0.8, 1.55, 11.733, 3.2, headers, rows, [2.1, 2.5, 3.5, 3.6])

    # Thank you / closing box
    close_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.95), Inches(11.733), Inches(1.85))
    close_box.fill.solid()
    close_box.fill.fore_color.rgb = LIGHT_BG
    close_box.line.color.rgb = NAVY
    close_box.line.width = Pt(1.5)

    tf_close = close_box.text_frame
    p_c1 = tf_close.paragraphs[0]
    p_c1.text = "THANK YOU"
    p_c1.alignment = PP_ALIGN.CENTER
    p_c1.font.name = "Calibri"
    p_c1.font.size = Pt(20)
    p_c1.font.bold = True
    p_c1.font.color.rgb = NAVY

    p_c2 = tf_close.add_paragraph()
    p_c2.text = "Wearable Fall Detection and Severity Classification System for Elderly People Using Deep Learning\nDepartment of Information Technology, Madras Institute of Technology, Anna University\nGitHub Repository: https://github.com/ArjunArul55/fall-detect"
    p_c2.alignment = PP_ALIGN.CENTER
    p_c2.font.name = "Calibri"
    p_c2.font.size = Pt(11.5)
    p_c2.font.color.rgb = DARK_GRAY
    p_c2.line_spacing = 1.3

    p_c3 = tf_close.add_paragraph()
    p_c3.text = "Open for Questions and Discussion"
    p_c3.alignment = PP_ALIGN.CENTER
    p_c3.font.name = "Calibri"
    p_c3.font.size = Pt(12)
    p_c3.font.bold = True
    p_c3.font.color.rgb = SLATE_BLUE

    add_slide_footer(slide, 16)


# ==============================================================================
# MAIN EXPORT ROUTINE
# ==============================================================================

def export_pptx_to_pdf(pptx_path: Path, pdf_path: Path) -> bool:
    """Export PPTX to PDF using native PowerPoint COM automation."""
    try:
        cmd = f"""
        $ppt = New-Object -ComObject PowerPoint.Application
        $pres = $ppt.Presentations.Open('{pptx_path.resolve()}', 0, 0, 0)
        $pres.SaveAs('{pdf_path.resolve()}', 32)
        $pres.Close()
        $ppt.Quit()
        """
        proc = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True)
        if proc.returncode == 0 and pdf_path.exists():
            print(f"Successfully converted to PDF: {pdf_path}")
            return True
        else:
            print(f"PowerPoint COM export failed: {proc.stderr}")
            return False
    except Exception as exc:
        print(f"Error during PDF conversion: {exc}")
        return False


def main():
    print("Building formal Review 3 PowerPoint Presentation...")
    prs = create_base_presentation()

    build_slide_1_title(prs)
    build_slide_2_scope(prs)
    build_slide_3_architecture(prs)
    build_slide_4_dataset(prs)
    build_slide_5_windowing(prs)
    build_slide_6_multitask_arch(prs)
    build_slide_7_masked_loss(prs)
    build_slide_8_severity_results(prs)
    build_slide_9_model_comparison(prs)
    build_slide_10_film_evaluation(prs)
    build_slide_11_quantization(prs)
    build_slide_12_esp32_export(prs)
    build_slide_13_streaming_engine(prs)
    build_slide_14_parity_demo(prs)
    build_slide_15_testing_limitations(prs)
    build_slide_16_conclusion(prs)

    prs.save(str(PPTX_PATH))
    print(f"Presentation saved to: {PPTX_PATH}")

    print("Converting presentation to PDF via native PowerPoint...")
    pdf_success = export_pptx_to_pdf(PPTX_PATH, PDF_PATH)
    if pdf_success:
        print(f"Review 3 PDF available at: {PDF_PATH}")
    else:
        print("Note: PDF conversion was not completed automatically. The PPTX file is ready.")


if __name__ == "__main__":
    main()
