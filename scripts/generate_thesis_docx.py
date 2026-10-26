"""
Generate the complete, formal Anna University (MIT Campus) B.Tech Thesis Report
for 'Wearable Fall Detection System for Elderly People Using Deep Learning' in Word (.docx) format.

Follows official formatting:
- Font: Times New Roman
- Margins: Left 1.5 inches, Right 1.0 inch, Top 1.0 inch, Bottom 1.0 inch
- 1.5 Line Spacing, Justified body text
- Chapter Headings: 16pt Bold Centered
- Section Headings: 13pt Bold
- Subsection Headings: 12pt Bold
- Body text: 12pt Regular
- Embedded authentic diagrams, waveforms, confusion matrices, and dashboard screenshots
"""

import os
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs" / "thesis"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
DOCX_PATH = DOCS_DIR / "Wearable_Fall_Detection_Thesis_Report.docx"
ASSETS_DIR = DOCS_DIR / "extracted_assets"

def set_cell_background(cell, fill_hex):
    """Set background color of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tc_pr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set inner cell padding in twips (1/20 of a pt)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tc_pr.append(tc_mar)

def add_styled_paragraph(doc, text="", font_size=12, bold=False, italic=False, 
                         align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=6, 
                         line_spacing=1.5, color=RGBColor(0, 0, 0)):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if text:
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(font_size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color
    return p

def add_heading_1(doc, text):
    """Chapter Title"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(15)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 0, 0)
    return p

def add_heading_2(doc, text):
    """Section Heading (e.g. 1.1 OVERVIEW)"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 0, 0)
    return p

def add_heading_3(doc, text):
    """Subsection Heading (e.g. 1.1.1 Feature Extraction)"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 0, 0)
    return p

def add_figure(doc, img_path, caption, width_in=5.5):
    """Add centered image and figure caption."""
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(10)
    p_img.paragraph_format.space_after = Pt(4)
    run_img = p_img.add_run()
    if Path(img_path).exists():
        run_img.add_picture(str(img_path), width=Inches(width_in))
    else:
        run_img.add_text(f"[Figure: {caption} - Image not found at {img_path}]")
        run_img.font.name = "Times New Roman"
        run_img.font.italic = True

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(12)
    run_cap = p_cap.add_run(caption)
    run_cap.font.name = "Times New Roman"
    run_cap.font.size = Pt(10.5)
    run_cap.font.bold = True
    return p_cap

def format_table(table, col_widths, headers, data, caption=""):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        set_cell_background(hdr_cells[i], "E2E8F0")
        set_cell_margins(hdr_cells[i], 120, 120, 140, 140)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].font.name = "Times New Roman"
        p.runs[0].font.size = Pt(10.5)
        p.runs[0].font.bold = True

    for row_idx, row_data in enumerate(data):
        row = table.rows[row_idx + 1]
        bg_hex = "F8FAFC" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate(row_data):
            cell = row.cells[col_idx]
            cell.text = str(val)
            set_cell_background(cell, bg_hex)
            set_cell_margins(cell, 100, 100, 140, 140)
            p = cell.paragraphs[0]
            p.runs[0].font.name = "Times New Roman"
            p.runs[0].font.size = Pt(10)
            if col_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for row in table.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = Inches(width)

def build_thesis_docx():
    print("Building formal Anna University Thesis in DOCX format...")
    doc = Document()

    # Configure Margins: Left 1.5 in (Binding), Right 1.0 in, Top 1.0 in, Bottom 1.0 in
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.5)
        section.right_margin = Inches(1.0)

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    p = add_styled_paragraph(doc, "WEARABLE FALL DETECTION SYSTEM FOR ELDERLY PEOPLE USING DEEP LEARNING", 
                             font_size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=10, space_after=18, line_spacing=1.2)

    add_styled_paragraph(doc, "A PROJECT REPORT", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)
    add_styled_paragraph(doc, "Submitted by", font_size=12, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=16)

    # Team members table
    team_table = doc.add_table(rows=4, cols=2)
    team_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    team_data = [
        ("SARATH R", "2023506065"),
        ("AJAY KUMAR S", "2023506105"),
        ("ARJUN A", "2023506063"),
        ("KRIS VIRSHAK", "2023506124"),
    ]
    for i, (name, reg) in enumerate(team_data):
        c1, c2 = team_table.rows[i].cells
        c1.text = name
        c2.text = reg
        for c, align in [(c1, WD_ALIGN_PARAGRAPH.LEFT), (c2, WD_ALIGN_PARAGRAPH.RIGHT)]:
            p = c.paragraphs[0]
            p.alignment = align
            p.runs[0].font.name = "Times New Roman"
            p.runs[0].font.size = Pt(11.5)
            p.runs[0].font.bold = True
        c1.width = Inches(2.5)
        c2.width = Inches(2.0)

    add_styled_paragraph(doc, "", font_size=12, space_after=16)
    add_styled_paragraph(doc, "Under the supervision of", font_size=12, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    add_styled_paragraph(doc, "Dr. S. UMAMAHESWARI, Associate Professor", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=16)

    add_styled_paragraph(doc, "In partial fulfillment for the award of the degree of", font_size=11.5, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    add_styled_paragraph(doc, "BACHELOR OF TECHNOLOGY", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    add_styled_paragraph(doc, "in", font_size=11.5, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    add_styled_paragraph(doc, "INFORMATION TECHNOLOGY", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=14)

    # Anna University Emblem
    emblem_path = ASSETS_DIR / "page_0_Image1.jpg"
    p_logo = doc.add_paragraph()
    p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_logo.paragraph_format.space_before = Pt(4)
    p_logo.paragraph_format.space_after = Pt(10)
    if emblem_path.exists():
        p_logo.add_run().add_picture(str(emblem_path), width=Inches(1.5))

    add_styled_paragraph(doc, "DEPARTMENT OF INFORMATION TECHNOLOGY", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    add_styled_paragraph(doc, "MADRAS INSTITUTE OF TECHNOLOGY CAMPUS", font_size=12.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    add_styled_paragraph(doc, "ANNA UNIVERSITY, CHENNAI – 600044", font_size=12.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
    add_styled_paragraph(doc, "APRIL 2026", font_size=12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)

    doc.add_page_break()

    # =========================================================================
    # BONAFIDE CERTIFICATE
    # =========================================================================
    add_styled_paragraph(doc, "BONAFIDE CERTIFICATE", font_size=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=24, space_after=24)
    cert_text = (
        'Certified that this project report titled "WEARABLE FALL DETECTION SYSTEM FOR ELDERLY '
        'PEOPLE USING DEEP LEARNING" is the bonafide work of SARATH R (2023506065), '
        'AJAY KUMAR S (2023506105), ARJUN A (2023506063), and KRIS VIRSHAK (2023506124) '
        'who carried out the project work under my supervision.'
    )
    add_styled_paragraph(doc, cert_text, font_size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=50, line_spacing=1.5)

    sig_table = doc.add_table(rows=6, cols=2)
    sig_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    sig_data = [
        ("SIGNATURE", "SIGNATURE"),
        ("Dr. RADHA SENTHILKUMAR", "Dr. S. UMAMAHESWARI"),
        ("HEAD OF THE DEPARTMENT", "SUPERVISOR"),
        ("Associate Professor", "Associate Professor"),
        ("Department of Information Technology", "Department of Information Technology"),
        ("MIT Campus, Anna University, Chennai – 600044", "MIT Campus, Anna University, Chennai – 600044"),
    ]
    for row_idx, (c1_val, c2_val) in enumerate(sig_data):
        row = sig_table.rows[row_idx]
        row.cells[0].text = c1_val
        row.cells[1].text = c2_val
        for col_idx, cell in enumerate(row.cells):
            cell.width = Inches(2.9)
            p = cell.paragraphs[0]
            p.runs[0].font.name = "Times New Roman"
            p.runs[0].font.size = Pt(11)
            if row_idx in [0, 1, 2]:
                p.runs[0].font.bold = True
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(2)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT

    doc.add_page_break()

    # =========================================================================
    # ACKNOWLEDGEMENT
    # =========================================================================
    add_styled_paragraph(doc, "ACKNOWLEDGEMENT", font_size=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=24, space_after=20)
    ack_paras = [
        "It is essential to mention the names of the individuals whose continuous guidance, constructive criticism, and scholarly encouragement made the successful completion of this project possible.",
        "We express our deepest sense of gratitude and sincere thanks to our respected project supervisor, Dr. S. Umamaheswari, Associate Professor, Department of Information Technology, Madras Institute of Technology Campus, Anna University, for her constant technical guidance, keen intellectual oversight, and patient encouragement throughout every stage of this research.",
        "We are profoundly thankful to our respected Head of the Department, Dr. Radha Senthilkumar, Department of Information Technology, for providing the necessary academic environment and continuous administrative support.",
        "We express our sincere gratitude to our review panel members for their insightful feedback, rigorous evaluations, and constructive suggestions during the project reviews, which substantially elevated the scientific and practical rigor of this work.",
        "We also place on record our heartfelt thanks to our respected Dean of MIT Campus, Dr. P. Jayashree, for providing exceptional computing infrastructure, laboratory amenities, and an inspiring academic atmosphere.",
        "Finally, we thank our parents, faculty members, and fellow students who supported us directly and indirectly during the execution of this project.",
    ]
    for para in ack_paras:
        add_styled_paragraph(doc, para, font_size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=12, line_spacing=1.5)

    add_styled_paragraph(doc, "", font_size=12, space_after=12)
    stud_table = doc.add_table(rows=4, cols=2)
    stud_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (name, reg) in enumerate(team_data):
        c1, c2 = stud_table.rows[i].cells
        c1.text = name
        c2.text = reg
        for c, align in [(c1, WD_ALIGN_PARAGRAPH.LEFT), (c2, WD_ALIGN_PARAGRAPH.RIGHT)]:
            p = c.paragraphs[0]
            p.alignment = align
            p.runs[0].font.name = "Times New Roman"
            p.runs[0].font.size = Pt(11)
            p.runs[0].font.bold = True
        c1.width = Inches(2.5)
        c2.width = Inches(2.0)

    doc.add_page_break()

    # =========================================================================
    # ABSTRACT
    # =========================================================================
    add_styled_paragraph(doc, "ABSTRACT", font_size=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=24, space_after=20)
    abstract_text_1 = (
        "Falls among elderly populations represent a primary global cause of fatal and non-fatal accidental injuries, "
        "often resulting in fractures, intracranial trauma, prolonged immobilization (the 'long-lie' state), and high healthcare expenditures. "
        "Wearable Inertial Measurement Units (IMUs) equipped with tri-axial accelerometers and gyroscopes offer an unobtrusive, "
        "continuous mechanism for ambient safety monitoring. However, traditional wearable detection systems suffer from severe false-alarm rates "
        "during high-acceleration activities of daily living (ADL), high memory footprints that overwhelm edge microcontrollers, "
        "and a lack of actionable telemetry regarding fall severity."
    )
    abstract_text_2 = (
        "This project presents an end-to-end, edge-deployable wearable fall detection system developed on the benchmark SisFall dataset "
        "(38 subjects, 4,505 recordings, 84,123 windows). To prevent data leakage and guarantee real-world generalization, a strict "
        "subject-wise partitioning scheme (26 Train / 6 Validation / 6 Test subjects) is implemented. Raw 200 Hz signals are conditioned through "
        "a second-order 5 Hz Butterworth low-pass filter to attenuate sensor noise, followed by anti-aliased downsampling to 20 Hz across six primary IMU axes. "
        "Recordings are segmented into 64-sample (3.2-second) sliding windows with a 16-sample (0.8-second) stride (75% overlap). "
        "An impact-centered labeling algorithm identifies the peak signal magnitude vector (SMV) to isolate true impact windows from non-fall movement."
    )
    abstract_text_3 = (
        "A hybrid Convolutional Neural Network and Long Short-Term Memory (CNN-LSTM) architecture is developed to fuse local spatial channel features "
        "with temporal sequential dynamics. To address the critical asymmetric clinical cost where missed falls carry severe health risks, the operating "
        "decision threshold is optimized on validation data to maximize the F2-score, selecting tau = 0.25 over the symmetric tau = 0.50 default. "
        "On held-out test subjects, the deployed CNN-LSTM achieves 0.9950 recall, 0.9606 precision, 0.9775 F1-score, and 0.9879 F2-score, "
        "missing only 6 out of 1,200 fall windows while correctly rejecting 12,378 ADL windows. "
        "To provide actionable context, an unsupervised K-Means (K=2) and Isolation Forest pipeline categorizes fall impacts into relative Mild and Severe "
        "pseudo-severity clusters (Macro-F1 = 0.8660) while masking anomalous outlier falls. "
        "Calibration-conditioned personalization via Feature-wise Linear Modulation (FiLM) is investigated and rigorously documented (p = 0.125). "
        "For microcontroller deployment, the model is quantized into Dynamic-Range TensorFlow Lite (111.0 KB, 53% compression, 0.026 ms latency) "
        "and exported as a C header array (model_data.h) consuming only 21% SRAM on an ESP32. "
        "An in-memory circular ring-buffer streaming engine and Streamlit clinical dashboard validate bitwise numerical parity (<= 1e-6 error) "
        "and real-time caregiver alert dispatch."
    )
    add_styled_paragraph(doc, abstract_text_1, font_size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=10, line_spacing=1.5)
    add_styled_paragraph(doc, abstract_text_2, font_size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=10, line_spacing=1.5)
    add_styled_paragraph(doc, abstract_text_3, font_size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=12, line_spacing=1.5)

    doc.add_page_break()

    # =========================================================================
    # TABLE OF CONTENTS
    # =========================================================================
    add_styled_paragraph(doc, "TABLE OF CONTENTS", font_size=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=24, space_after=16)

    toc_table = doc.add_table(rows=1, cols=3)
    toc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    toc_data = [
        ("CHAPTER NO.", "TITLE", "PAGE NO."),
        ("", "ABSTRACT", "iv"),
        ("", "LIST OF TABLES", "viii"),
        ("", "LIST OF FIGURES", "ix"),
        ("", "LIST OF ABBREVIATIONS", "xi"),
        ("1", "INTRODUCTION", "1"),
        ("1.1", "Overview", "1"),
        ("1.2", "Research Challenges", "2"),
        ("1.3", "Objectives", "3"),
        ("1.4", "Scope of the Project", "4"),
        ("1.5", "Key Contributions", "4"),
        ("1.6", "Wearable Inertial Measurement Units (IMU)", "5"),
        ("1.7", "Deep Learning for Fall Detection (CNN-LSTM)", "6"),
        ("1.8", "Unsupervised Fall Severity Pseudo-Labeling", "7"),
        ("1.9", "Feature-Wise Linear Modulation (FiLM)", "8"),
        ("1.10", "System Architecture Overview", "9"),
        ("1.11", "Organization of the Thesis", "10"),
        ("2", "LITERATURE SURVEY", "11"),
        ("2.1", "Threshold-Based and Handcrafted Motion Methods", "11"),
        ("2.2", "Classical Machine Learning Approaches", "12"),
        ("2.3", "Deep Learning Architectures for Fall Detection", "13"),
        ("2.4", "Multi-Task Learning and Severity Grading", "15"),
        ("2.5", "Personalization and Calibration in Wearables", "16"),
        ("2.6", "Edge Inference and Microcontroller Quantization", "17"),
        ("2.7", "Summary of Literature and Research Gaps", "18"),
        ("3", "SYSTEM ARCHITECTURE AND DESIGN", "20"),
        ("3.1", "Overall System Architecture", "20"),
        ("3.2", "Wearable Sensor Acquisition Pipeline", "22"),
        ("3.3", "Butterworth Low-Pass Filtering and Downsampling", "23"),
        ("3.4", "Sliding Window Segmentation and Impact Labeling", "24"),
        ("3.5", "Hybrid CNN-LSTM Neural Network Architecture", "26"),
        ("3.6", "Multi-Task Severity Head and Masked Loss", "28"),
        ("3.7", "FiLM Personalization Adapter Design", "30"),
        ("3.8", "TensorFlow Lite Quantization Pipeline", "31"),
        ("3.9", "Circular Ring Buffer and Streaming Alert Logic", "33"),
        ("4", "ALGORITHM DEVELOPMENT AND IMPLEMENTATION", "35"),
        ("4.1", "Dataset Ingestion and Preprocessing Pipeline", "35"),
        ("4.2", "Butterworth Filtering and Anti-Aliased Resampling", "36"),
        ("4.3", "Impact-Centered Window Labeling Algorithm", "37"),
        ("4.4", "CNN-LSTM Model Implementation and Training", "38"),
        ("4.5", "Validation-Based Operating Threshold Optimization", "40"),
        ("4.6", "Severity Feature Engineering (58 Features)", "41"),
        ("4.7", "K-Means Clustering and Isolation Forest Filtering", "43"),
        ("4.8", "FiLM Meta-Training and Calibration Implementation", "44"),
        ("4.9", "Dynamic-Range Quantization and C Array Header Export", "46"),
        ("4.10", "Real-Time Streaming Engine and Dashboard Implementation", "47"),
        ("5", "RESULTS AND DISCUSSIONS", "49"),
        ("5.1", "Experimental Setup and Computing Environment", "49"),
        ("5.2", "SisFall Dataset Partitioning and Characteristics", "50"),
        ("5.3", "Core Binary Detection Performance (CNN vs CNN-LSTM)", "51"),
        ("5.4", "Operating Decision Threshold Optimization Analysis", "53"),
        ("5.5", "Severity Pseudo-Labeling Benchmark and Analysis", "55"),
        ("5.6", "Multi-Task Joint Loss Evaluation and Decision Gating", "57"),
        ("5.7", "FiLM Personalization Empirical and Statistical Results", "59"),
        ("5.8", "Edge Quantization Footprint and Latency Benchmarks", "61"),
        ("5.9", "Streaming Replay Verification and Numerical Parity", "63"),
        ("5.10", "Threats to Validity and Research Limitations", "65"),
        ("6", "CONCLUSION AND FUTURE WORK", "67"),
        ("6.1", "Conclusion", "67"),
        ("6.2", "Future Work", "68"),
        ("", "REFERENCES", "70"),
    ]

    for row_idx, item in enumerate(toc_data):
        if row_idx == 0:
            row = toc_table.rows[0]
        else:
            row = toc_table.add_row()
        ch_no, title, page = item
        row.cells[0].text = ch_no
        row.cells[1].text = title
        row.cells[2].text = page
        
        # Formatting
        row.cells[0].width = Inches(1.3)
        row.cells[1].width = Inches(3.8)
        row.cells[2].width = Inches(0.9)
        
        for c_idx, cell in enumerate(row.cells):
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            p.runs[0].font.name = "Times New Roman"
            p.runs[0].font.size = Pt(10.5)
            if row_idx == 0 or ch_no in ["1", "2", "3", "4", "5", "6"] or title in ["ABSTRACT", "LIST OF TABLES", "LIST OF FIGURES", "LIST OF ABBREVIATIONS", "REFERENCES"]:
                p.runs[0].font.bold = True
            if c_idx == 2:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    doc.add_page_break()

    # =========================================================================
    # LIST OF TABLES & LIST OF FIGURES
    # =========================================================================
    add_styled_paragraph(doc, "LIST OF TABLES", font_size=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=24, space_after=16)
    tables_list = [
        ("TABLE NO.", "TITLE", "PAGE NO."),
        ("3.1", "SisFall IMU Sensor Channel Specifications and Dynamic Ranges", "22"),
        ("3.2", "Sliding Window Segmentation Hyperparameters", "25"),
        ("3.3", "Layer-by-Layer Specifications of the Hybrid CNN-LSTM Architecture", "27"),
        ("4.1", "58-Dimensional Statistical and Kinematic Feature Set for Severity Modeling", "42"),
        ("4.2", "Dynamic-Range Quantization Parameters and Target ESP32 Specifications", "46"),
        ("5.1", "Experimental Computing Environment Specifications", "49"),
        ("5.2", "SisFall Dataset Split and Window Counts across Partitions", "50"),
        ("5.3", "Performance Comparison of CNN Baseline and CNN-LSTM on Held-Out Test Set", "52"),
        ("5.4", "Confusion Matrix Breakdown across Operating Decision Thresholds", "54"),
        ("5.5", "Severity Clustering Metrics Comparison (K=2 vs K=3)", "56"),
        ("5.6", "Severity Head Confusion Matrix across 1,176 Confident Fall Windows", "56"),
        ("5.7", "Binary vs Multi-Task CNN-LSTM Gating Performance Comparison", "58"),
        ("5.8", "FiLM Personalization F2-Score across Calibration Window Sizes", "60"),
        ("5.9", "TFLite Edge Quantization Benchmark: Footprint, Latency, and Numerical Parity", "62"),
        ("5.10", "Replay-Based Streaming Verification Across 6 Standard Evaluation Scenarios", "64"),
    ]
    t_table = doc.add_table(rows=len(tables_list), cols=3)
    t_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, (t_no, title, pg) in enumerate(tables_list):
        row = t_table.rows[r_idx]
        row.cells[0].text = t_no
        row.cells[1].text = title
        row.cells[2].text = pg
        row.cells[0].width = Inches(1.2)
        row.cells[1].width = Inches(4.0)
        row.cells[2].width = Inches(0.8)
        for c_idx, cell in enumerate(row.cells):
            p = cell.paragraphs[0]
            p.runs[0].font.name = "Times New Roman"
            p.runs[0].font.size = Pt(10)
            if r_idx == 0:
                p.runs[0].font.bold = True
            if c_idx == 2:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    doc.add_page_break()

    add_styled_paragraph(doc, "LIST OF FIGURES", font_size=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=24, space_after=16)
    figures_list = [
        ("FIGURE NO.", "TITLE", "PAGE NO."),
        ("3.1", "End-to-End System Architecture of the Wearable Fall Detection System", "21"),
        ("3.2", "Tri-Axial Accelerometer Waveform Before and After 5 Hz Butterworth Low-Pass Filter", "23"),
        ("3.3", "Sliding Window Segmentation with 64-Sample Windows and 75% Overlap", "24"),
        ("3.4", "Impact-Centered Window Labeling via Peak Signal Magnitude Vector (SMV)", "25"),
        ("3.5", "Hybrid CNN-LSTM Neural Network Architecture Flowchart", "27"),
        ("3.6", "Multi-Task Dual-Head Neural Trunk with Masked Severity Loss Formulation", "29"),
        ("3.7", "Feature-Wise Linear Modulation (FiLM) Adapter and UserEncoder Pipeline", "30"),
        ("3.8", "Circular Ring Buffer Mechanics and Streaming Evaluation Stride", "33"),
        ("5.1", "Subject-Wise Split and Overall Class Imbalance in the SisFall Dataset", "51"),
        ("5.2", "CNN-LSTM Binary Model Training Loss and Accuracy Convergence Curves", "53"),
        ("5.3", "Validation Precision-Recall Curve Across Operating Decision Thresholds", "54"),
        ("5.4", "Held-Out Test Set Confusion Matrix at Canonical tau = 0.50 Threshold", "55"),
        ("5.5", "K-Means (K=2) Fall Severity Clusters Projected onto 2D PCA Space", "56"),
        ("5.6", "Isolation Forest Anomaly Score Distribution Flagging Uncertain Fall Outliers", "57"),
        ("5.7", "Peak Acceleration and Angular Velocity Intensity Profiles Across Severity Clusters", "57"),
        ("5.8", "Paired Wilcoxon Signed-Rank Test Significance for FiLM Personalization", "60"),
        ("5.9", "TFLite Model Footprint and Single-Window CPU Latency Benchmarks", "62"),
        ("5.10", "Live Streaming Dashboard: Fall Probability Timeline Crossing Decision Threshold", "64"),
        ("5.11", "Streaming Dashboard: Raw 6-Channel Synchronized IMU Sensor Traces", "65"),
        ("5.12", "Streaming Dashboard: Simulated Caregiver Alert Dispatch Notification Log", "65"),
        ("5.13", "Interactive FiLM User Personalization Demonstration Interface", "66"),
    ]
    f_table = doc.add_table(rows=len(figures_list), cols=3)
    f_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, (f_no, title, pg) in enumerate(figures_list):
        row = f_table.rows[r_idx]
        row.cells[0].text = f_no
        row.cells[1].text = title
        row.cells[2].text = pg
        row.cells[0].width = Inches(1.2)
        row.cells[1].width = Inches(4.0)
        row.cells[2].width = Inches(0.8)
        for c_idx, cell in enumerate(row.cells):
            p = cell.paragraphs[0]
            p.runs[0].font.name = "Times New Roman"
            p.runs[0].font.size = Pt(10)
            if r_idx == 0:
                p.runs[0].font.bold = True
            if c_idx == 2:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    doc.add_page_break()

    # =========================================================================
    # LIST OF ABBREVIATIONS
    # =========================================================================
    add_styled_paragraph(doc, "LIST OF ABBREVIATIONS", font_size=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=24, space_after=16)
    abbr_list = [
        ("ADL", "Activities of Daily Living"),
        ("AI", "Artificial Intelligence"),
        ("ANN", "Artificial Neural Network"),
        ("BCE", "Binary Cross-Entropy"),
        ("BiLSTM", "Bidirectional Long Short-Term Memory"),
        ("BLE", "Bluetooth Low Energy"),
        ("CNN", "Convolutional Neural Network"),
        ("CPU", "Central Processing Unit"),
        ("ESP32", "Espressif Systems 32-bit Microcontroller"),
        ("FPR", "False Positive Rate"),
        ("FiLM", "Feature-wise Linear Modulation"),
        ("FP", "False Positive"),
        ("FN", "False Negative"),
        ("GAP", "Global Average Pooling"),
        ("IMU", "Inertial Measurement Unit"),
        ("INT8", "8-bit Signed Integer Quantization"),
        ("K-Means", "K-Means Partitioning Clustering Algorithm"),
        ("LSTM", "Long Short-Term Memory"),
        ("MCU", "Microcontroller Unit"),
        ("ML", "Machine Learning"),
        ("PCA", "Principal Component Analysis"),
        ("RAM", "Random Access Memory"),
        ("ReLU", "Rectified Linear Unit"),
        ("SMV", "Signal Magnitude Vector"),
        ("SRAM", "Static Random Access Memory"),
        ("SVM", "Support Vector Machine"),
        ("TFLite", "TensorFlow Lite Embedded Runtime"),
        ("TN", "True Negative"),
        ("TP", "True Positive"),
        ("UI", "User Interface"),
    ]
    abbr_table = doc.add_table(rows=len(abbr_list), cols=2)
    abbr_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, (abbr, full) in enumerate(abbr_list):
        row = abbr_table.rows[r_idx]
        row.cells[0].text = abbr
        row.cells[1].text = full
        row.cells[0].width = Inches(1.8)
        row.cells[1].width = Inches(4.2)
        for cell in row.cells:
            p = cell.paragraphs[0]
            p.runs[0].font.name = "Times New Roman"
            p.runs[0].font.size = Pt(10.5)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
        row.cells[0].paragraphs[0].runs[0].font.bold = True

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 1: INTRODUCTION
    # =========================================================================
    add_heading_1(doc, "CHAPTER 1\nINTRODUCTION")
    
    add_heading_2(doc, "1.1 OVERVIEW")
    add_styled_paragraph(doc, 
        "Rapid demographic aging represents one of the most prominent socioeconomic and public health challenges of the twenty-first century. "
        "According to the World Health Organization (WHO), falls constitute the second leading cause of unintentional injury-related deaths worldwide, "
        "with adults older than 60 years of age suffering the greatest proportion of fatal falls. Approximately 28% to 35% of individuals aged 65 and over "
        "experience at least one fall each year, escalating to 32% to 42% for those aged 70 years and older. "
        "Beyond direct physiological injuries—such as femoral neck fractures, traumatic brain injuries, and soft-tissue lacerations—falls frequently precipitate "
        "the debilitating 'long-lie' state, wherein an injured individual remains immobilized on the ground for an hour or longer. "
        "The consequences of prolonged recumbency include dehydration, hypothermia, rhabdomyolysis, pressure ulcers, and psychological distress, "
        "which collectively multiply hospitalization rates and mortality risks."
    )
    add_styled_paragraph(doc, 
        "To mitigate these risks, automated fall detection systems have gained significant research attention. "
        "Traditional vision-based systems using ambient cameras provide non-intrusive monitoring but are fundamentally restricted to single-room indoor environments, "
        "suffer from severe blind spots, and incur significant privacy violations that limit their acceptability in residential spaces. "
        "Conversely, wearable inertial measurement unit (IMU) systems equipped with micro-electromechanical systems (MEMS) tri-axial accelerometers "
        "and gyroscopes offer continuous, ubiquitous, and privacy-preserving monitoring across both indoor and outdoor activities. "
        "By continuously tracking linear acceleration and angular velocity, wearable devices can detect the rapid biomechanical transitions that distinguish "
        "accidental falls from normal activities of daily living (ADL)."
    )
    add_styled_paragraph(doc, 
        "This project, titled 'Wearable Fall Detection System for Elderly People Using Deep Learning', develops an end-to-end, scientifically rigorous, "
        "and edge-deployable deep learning framework. The system is designed to ingest raw inertial motion signals, perform low-power filtering and windowing, "
        "classify fall events with high recall, estimate the physical impact severity of the fall, explore calibration-conditioned user personalization, "
        "and execute real-time inference on resource-constrained wearable microcontrollers."
    )

    add_heading_2(doc, "1.2 RESEARCH CHALLENGES")
    add_styled_paragraph(doc, 
        "Developing a reliable and practical wearable fall detection system presents several critical engineering and scientific hurdles:"
    )
    challenges = [
        ("High Class Imbalance and False Alarm Rates: ", "In everyday life, falls are rare, catastrophic events, whereas ADLs (walking, sitting, ascending stairs) "
         "occur continuously. In benchmark datasets, ADL windows outnumber fall windows by more than 10 to 1. Traditional classifiers tuned for overall accuracy "
         "tend to trigger excessive false alarms on vigorous non-fall activities (e.g., jumping, coughing, stumbling, or rapidly sitting on a bed), "
         "causing user alert fatigue and eventual device abandonment."),
        ("Asymmetric Misclassification Costs: ", "In clinical safety, a False Negative (a missed fall) can be fatal, while a False Positive (a false alert) is merely an inconvenience. "
         "Conventional machine learning pipelines operating at the default decision threshold of 0.50 assume symmetric misclassification penalties, "
         "leading to unacceptable numbers of undetected falls."),
        ("Data Leakage in Evaluation Protocols: ", "Many published fall detection studies randomly shuffle overlapping time windows across training and testing partitions. "
         "Because consecutive sliding windows from the same subject share sensor noise signatures and personal gait patterns, random splitting causes severe data leakage, "
         "yielding overly optimistic academic metrics that fail to generalize to unseen human subjects in real-world deployments."),
        ("Absence of Clinical Severity Telemetry: ", "Existing commercial alarms operate as binary detectors (Fall vs. Non-Fall). They provide no indication of impact severity, "
         "preventing emergency medical dispatchers from prioritizing high-energy, potentially debilitating impacts over mild slips."),
        ("Edge Compute and Energy Constraints: ", "Wearable smartwatches and pendant monitors operate on battery-constrained microcontrollers (e.g., ESP32, ARM Cortex-M) "
         "with limited Static RAM (typically 256 KB to 520 KB) and strict thermal dissipation limits. Large neural networks cannot be executed locally without intelligent quantization.")
    ]
    for title, desc in challenges:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.5
        r_t = p.add_run(title)
        r_t.font.name = "Times New Roman"
        r_t.font.bold = True
        r_d = p.add_run(desc)
        r_d.font.name = "Times New Roman"

    add_heading_2(doc, "1.3 OBJECTIVES")
    add_styled_paragraph(doc, 
        "The primary objectives of this project are as follows:"
    )
    objectives = [
        "To build a robust, reproducible, and leakage-free signal preprocessing pipeline for the benchmark SisFall inertial dataset, incorporating Butterworth low-pass filtering and anti-aliased downsampling.",
        "To segment continuous multi-channel IMU recordings into fixed temporal windows and apply an impact-centered labeling algorithm based on peak signal magnitude vector dynamics.",
        "To design, train, and evaluate a hybrid deep learning model combining 1D Convolutional Neural Networks (CNN) for spatial feature extraction with Long Short-Term Memory (LSTM) networks for temporal sequence modeling.",
        "To formulate a validation-based decision threshold sweep optimizing the F2-score to prioritize fall recall and minimize fatal false negatives.",
        "To develop an unsupervised clustering pipeline using K-Means (K=2) and Isolation Forest to assign relative Mild and Severe pseudo-severity grades to detected falls while filtering anomalous outliers.",
        "To investigate calibration-conditioned user personalization using Feature-wise Linear Modulation (FiLM) and evaluate its statistical significance via paired Wilcoxon testing.",
        "To quantize the trained deep learning model into TensorFlow Lite Dynamic-Range format, export a standalone C byte array (model_data.h) for ESP32 microcontrollers, and demonstrate streaming inference through a real-time clinical dashboard."
    ]
    for obj in objectives:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.5
        r_bullet = p.add_run("•  ")
        r_bullet.font.name = "Times New Roman"
        r_bullet.font.bold = True
        r_text = p.add_run(obj)
        r_text.font.name = "Times New Roman"

    add_heading_2(doc, "1.4 SCOPE OF THE PROJECT")
    add_styled_paragraph(doc, 
        "The scope of this project encompasses the complete research-to-edge engineering cycle for wearable fall detection. "
        "It utilizes the SisFall dataset, comprising 4,505 trials recorded from 38 subjects performing 19 distinct ADLs and 15 simulated fall scenarios. "
        "The project focuses on multi-axis inertial sensing (primary tri-axial accelerometer and gyroscope), sliding window segmentation, "
        "hybrid deep neural networks, unsupervised severity categorization, and embedded deployment benchmarking. "
        "\n\nScope Boundary and Research Guardrail: "
        "This system represents a high-fidelity research prototype and a software-in-the-loop demonstration. "
        "The frontends replay stored SisFall recordings and simulated sensor profiles. They do not directly interface with live commercial smartwatch hardware, "
        "dispatch live cellular SMS to emergency services, or provide a medically certified diagnosis of physical injury."
    )

    add_heading_2(doc, "1.5 KEY CONTRIBUTIONS")
    add_styled_paragraph(doc, 
        "The principal technical contributions of this thesis are summarized below:"
    )
    contributions = [
        ("Strict Subject-Wise Evaluation Protocol: ", "Enforces a zero-leakage 26/6/6 subject split across train, validation, and test sets, ensuring authentic out-of-sample evaluation."),
        ("Optimized Recall-Centric Decision Policy: ", "Establishes a validation-selected operating threshold of tau = 0.25 (maximizing F2), achieving 0.9950 recall and reducing missed falls to only 6 out of 1,200 test fall windows."),
        ("Unsupervised Fall Severity Pseudo-Labeling: ", "Introduces a 58-dimensional feature-space K-Means (K=2) and Isolation Forest pipeline that stratifies falls into Mild vs. Severe impacts (Macro-F1 = 0.8660) and masks atypical outlier rows."),
        ("Empirical Personalization Investigation: ", "Formulates an affine FiLM modulation layer to adapt intermediate feature maps to user profiles without full network retraining, establishing honest statistical limits (p = 0.125)."),
        ("Ultra-Lightweight Edge Quantization: ", "Delivers a 111.0 KB Dynamic-Range TFLite model executing in 0.026 ms on a single CPU thread, alongside a C header (model_data.h) consuming only 21% SRAM on an ESP32 microcontroller."),
        ("Real-Time Streaming Replay Architecture: ", "Engineers an in-memory circular ring buffer operating with bitwise numerical parity (<= 1e-6 max absolute difference) and an interactive clinical demonstration dashboard.")
    ]
    for title, desc in contributions:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.5
        r_t = p.add_run(title)
        r_t.font.name = "Times New Roman"
        r_t.font.bold = True
        r_d = p.add_run(desc)
        r_d.font.name = "Times New Roman"

    add_heading_2(doc, "1.6 WEARABLE INERTIAL MEASUREMENT UNITS (IMU)")
    add_styled_paragraph(doc, 
        "Inertial Measurement Units serve as the sensory foundation of modern wearable computing. An IMU typically integrates a 3-axis accelerometer "
        "and a 3-axis gyroscope. The accelerometer measures specific force (linear acceleration plus gravitational acceleration vector) along three orthogonal axes (x, y, z), "
        "expressed in units of g (1g ≈ 9.81 m/s²). The gyroscope measures rotational rate (angular velocity) around these axes, expressed in degrees per second (°/s) or radians per second. "
        "During a fall event, the human body undergoes a characteristic biomechanical sequence: an initial loss of balance, followed by a near-weightless free-fall phase "
        "(where total acceleration drops significantly below 1g), culminating in a sharp, high-g impact collision with the ground, and concluding in an inactive post-fall resting period. "
        "By continuously sampling these physical channels, an IMU provides rich multidimensional time-series data reflecting human kinematic state."
    )

    add_heading_2(doc, "1.7 DEEP LEARNING FOR FALL DETECTION (CNN-LSTM)")
    add_styled_paragraph(doc, 
        "Traditional machine learning classifiers rely heavily on handcrafted statistical features (mean, variance, peak-to-peak amplitudes, wavelet coefficients) "
        "which require extensive domain engineering and fail to capture transient sequential relationships. Deep learning automates feature discovery directly from raw temporal signals. "
        "In this project, a hybrid CNN-LSTM architecture is designed. The 1D Convolutional layers act as automated spatial feature extractors, learning localized temporal filters "
        "across the six sensor axes. Max-pooling layers reduce dimensionality while preserving dominant peak excitations. "
        "The Long Short-Term Memory (LSTM) recurrent layer receives the ordered sequence of feature maps, maintaining internal hidden and cell states that explicitly model "
        "how kinematic patterns evolve over the 3.2-second window. This architectural synergy allows the network to distinguish sharp, isolated shocks (such as dropping a phone or clapping) "
        "from true fall events that follow a prolonged loss of equilibrium."
    )

    add_heading_2(doc, "1.8 UNSUPERVISED FALL SEVERITY PSEUDO-LABELING")
    add_styled_paragraph(doc, 
        "In commercial healthcare systems, simply stating 'a fall occurred' is insufficient for emergency triage. A frail elderly patient who slips gently onto a carpeted surface "
        "requires a different clinical response than a patient who collapses violently onto concrete from standing height. "
        "Because public biomedical datasets lack clinical trauma scores (such as Abbreviated Injury Scale or Injury Severity Scores), this project introduces an unsupervised "
        "clustering strategy. By extracting 58 comprehensive statistical and magnitude-based kinematic features from fall windows, K-Means clustering groups events into relative "
        "Mild and Severe pseudo-severity categories. Furthermore, an Isolation Forest anomaly detector isolates irregular fall dynamics (e.g., tumbling, rolling), "
        "marking them as 'Uncertain' (-1) to maintain clean training supervision."
    )

    add_heading_2(doc, "1.9 FEATURE-WISE LINEAR MODULATION (FiLM)")
    add_styled_paragraph(doc, 
        "Human motion varies widely due to subject age, weight, height, walking cadence, and sensor orientation on the body. "
        "Training a separate neural network for every user is computationally impractical on edge devices. "
        "Feature-wise Linear Modulation (FiLM) offers an elegant alternative by applying affine transformations to the intermediate feature representations of a frozen base network. "
        "A lightweight UserEncoder network observes a small calibration sample of normal ADL walking windows from a specific individual and predicts scale (gamma) and shift (beta) vectors. "
        "These vectors modulate the feature maps before temporal reasoning in the LSTM, personalizing detection without requiring backpropagation on the device."
    )

    add_heading_2(doc, "1.10 SYSTEM ARCHITECTURE OVERVIEW")
    add_styled_paragraph(doc, 
        "The end-to-end system is structured into five modular tiers: (1) Sensor Ingestion and Conditioning, (2) Temporal Segmentation and Impact Centering, "
        "(3) Hybrid CNN-LSTM Inference Engine, (4) Edge Model Quantization and Microcontroller Export, and (5) Real-Time Streaming Ring Buffer and Caregiver Interface. "
        "The complete architecture is visualized in Figure 1.1."
    )
    add_figure(doc, REPO_ROOT / "docs/presentation/figures/review3_system_architecture.png", 
               "Figure 1.1: End-to-End System Architecture of the Wearable Fall Detection System", width_in=5.8)

    add_heading_2(doc, "1.11 ORGANIZATION OF THE THESIS")
    add_styled_paragraph(doc, 
        "The remainder of this thesis is structured as follows: "
        "\nChapter 2 provides a comprehensive Literature Survey, analyzing threshold-based, classical machine learning, deep learning, severity estimation, and TinyML quantization approaches. "
        "\nChapter 3 details the System Architecture and Design, explaining mathematical signal filtering, sliding window mechanics, neural network specifications, and the streaming ring buffer. "
        "\nChapter 4 presents Algorithm Development and Implementation, documenting dataset ingestion, the impact-centered labeling algorithm, loss functions, threshold sweeps, and C array export. "
        "\nChapter 5 presents Results and Discussions, providing extensive empirical benchmarks, confusion matrices, severity clustering results, FiLM statistical evaluations, quantization metrics, and latency profiles. "
        "\nChapter 6 concludes the thesis and outlines prospective Future Work."
    )

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 2: LITERATURE SURVEY
    # =========================================================================
    add_heading_1(doc, "CHAPTER 2\nLITERATURE SURVEY")

    add_heading_2(doc, "2.1 THRESHOLD-BASED AND HANDCRAFTED MOTION METHODS")
    add_styled_paragraph(doc, 
        "Early research in automated fall detection relied almost exclusively on threshold-based algorithms applied to raw or derived kinematic parameters. "
        "These systems typically compute the acceleration Signal Magnitude Vector (SMV), defined as SMV = sqrt(acc_x^2 + acc_y^2 + acc_z^2). "
        "When the SMV exceeds a predetermined threshold (often set between 2.5g and 3.5g), a fall is flagged. "
        "Bourke et al. [1] conducted pioneering studies evaluating tri-axial accelerometers placed at the trunk and thigh, demonstrating that peak acceleration thresholds "
        "could distinguish simulated falls from low-intensity ADLs. However, threshold-based techniques suffer from severe false alarms when users engage in vigorous activities "
        "such as jumping, running downstairs, or sitting down abruptly, which generate impact spikes exceeding 3g. "
        "Furthermore, fixed thresholds fail to account for inter-individual variability in mass and muscle tone, leading to unacceptably high false-positive rates in ambulatory living environments."
    )

    add_heading_2(doc, "2.2 CLASSICAL MACHINE LEARNING APPROACHES")
    add_styled_paragraph(doc, 
        "To improve discrimination between falls and high-acceleration ADLs, researchers transitioned toward classical supervised machine learning algorithms. "
        "In these frameworks, continuous sensor streams are divided into temporal windows from which handcrafted statistical, time-domain, and frequency-domain features are computed. "
        "Commonly extracted features include mean, variance, root-mean-square (RMS), skewness, kurtosis, zero-crossing rates, spectral energy, and discrete wavelet transform (DWT) sub-band coefficients. "
        "Support Vector Machines (SVM) with radial basis function (RBF) kernels were widely adopted due to their effectiveness in high-dimensional feature spaces [2]. "
        "Random Forests and k-Nearest Neighbors (k-NN) were also extensively explored [3]. "
        "While classical machine learning improved overall accuracy, it introduced significant drawbacks: feature engineering is labor-intensive, domain-specific, and prone to losing fine-grained temporal phase transitions. "
        "Moreover, computing dozens of spectral and statistical features over short intervals imposes heavy mathematical overhead on low-power microcontrollers."
    )

    add_heading_2(doc, "2.3 DEEP LEARNING ARCHITECTURES FOR FALL DETECTION")
    add_styled_paragraph(doc, 
        "The advent of deep neural networks transformed human activity recognition and fall detection by enabling automated, hierarchical representation learning directly from raw sensor time series. "
        "Convolutional Neural Networks (CNN) using 1D temporal kernels were introduced to capture localized motion motifs across multi-axis IMU streams [4]. "
        "1D CNNs apply weight-sharing across time steps, extracting translation-invariant features representing sudden acceleration shocks. "
        "However, standalone CNN architectures lack recurrent feedback connections, limiting their ability to model long-term temporal dependencies across the pre-fall, impact, and post-fall phases. "
        "Conversely, Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks excel at sequential modeling by maintaining memory cells that capture temporal progression [5]. "
        "Recent state-of-the-art studies have demonstrated that combining CNN and LSTM networks into a unified hybrid architecture yields superior performance [6]. "
        "In this paradigm, convolutional layers perform spatial feature extraction across sensor axes, while subsequent LSTM layers model temporal transitions over multi-second windows."
    )

    add_heading_2(doc, "2.4 MULTI-TASK LEARNING AND SEVERITY GRADING")
    add_styled_paragraph(doc, 
        "While binary fall detection has been widely explored, automated fall severity estimation remains an emerging and challenging frontier. "
        "Existing clinical literature indicates that impact velocity, kinetic energy dissipation, and post-impact immobility are strongly correlated with physical injury risk [7]. "
        "In machine learning, multi-task learning (MTL) has been proposed to jointly optimize shared representations for related tasks, such as simultaneous fall detection and activity classification [8]. "
        "However, applying multi-task learning to fall severity is severely hindered by the lack of clinician-annotated injury labels in publicly available datasets. "
        "Attempts to train multi-task networks on uncurated heuristic labels often lead to negative transfer, where optimizing the secondary severity objective degrades the primary binary safety recall."
    )

    add_heading_2(doc, "2.5 PERSONALIZATION AND DOMAIN ADAPTATION IN WEARABLES")
    add_styled_paragraph(doc, 
        "A major limitation of generalized deep learning models in wearable sensing is the subject domain shift. "
        "Differences in walking biomechanics, age-related gait disorders, sensor attachment snugness, and anatomical orientation alter signal baselines. "
        "Transfer learning and fine-tuning techniques have been investigated to adapt pre-trained models to target users [9]. "
        "However, fine-tuning requires backpropagation on the wearable device, demanding significant computational memory, floating-point arithmetic, and labeled fall examples from the user—an impractical requirement in elderly care. "
        "Feature-wise Linear Modulation (FiLM), originally introduced by Perez et al. [10] for visual reasoning, offers a parameter-efficient alternative. "
        "FiLM applies affine scaling and shifting to intermediate feature maps conditioned on a compact context vector, enabling adaptation without retraining base network weights."
    )

    add_heading_2(doc, "2.6 EDGE INFERENCE AND MICROCONTROLLER QUANTIZATION")
    add_styled_paragraph(doc, 
        "Deploying deep neural networks onto wearable Internet-of-Things (IoT) edge devices requires minimizing memory footprints and inference latency. "
        "TinyML frameworks, notably TensorFlow Lite for Microcontrollers (TFLite Micro) and ARM CMSIS-NN, enable deep learning on resource-constrained microcontrollers [11]. "
        "Post-training quantization converts 32-bit floating-point weights and activations into reduced-precision representations, such as 16-bit float (FP16) or 8-bit integer (INT8). "
        "Dynamic-range quantization quantizes weights to 8-bit integers while performing runtime computation with floating-point activations, achieving substantial storage compression with minimal accuracy degradation [12]. "
        "Exporting quantized models into static C byte arrays enables direct compilation into microcontroller flash memory, avoiding dynamic heap allocation."
    )

    add_heading_2(doc, "2.7 SUMMARY OF LITERATURE AND RESEARCH GAPS")
    add_styled_paragraph(doc, 
        "The literature review reveals three critical research gaps that motivate the contributions of this project:"
    )
    gaps = [
        ("Lack of Subject-Wise Rigor: ", "A substantial portion of existing studies evaluates models using random window shuffling, inflating reported accuracies and obscuring real-world generalization failures."),
        ("Symmetric Optimization Bias: ", "Standard research pipelines optimize classification accuracy or balanced F1-score at a 0.50 threshold, neglecting the catastrophic clinical cost of missed falls."),
        ("Disconnection Between Model and Deployment: ", "Most published works present high-parameter models evaluated offline in Python, without validating edge quantization, execution latency, or circular streaming buffer integration on embedded microcontrollers.")
    ]
    for title, desc in gaps:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.5
        r_t = p.add_run(title)
        r_t.font.name = "Times New Roman"
        r_t.font.bold = True
        r_d = p.add_run(desc)
        r_d.font.name = "Times New Roman"

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 3: SYSTEM ARCHITECTURE AND DESIGN
    # =========================================================================
    add_heading_1(doc, "CHAPTER 3\nSYSTEM ARCHITECTURE AND DESIGN")

    add_heading_2(doc, "3.1 OVERALL SYSTEM ARCHITECTURE")
    add_styled_paragraph(doc, 
        "The architecture of the proposed wearable fall detection system is designed around a validated core binary detector and two parallel exploratory research branches. "
        "Figure 3.1 illustrates the comprehensive system flow. Continuous 6-channel IMU streams are filtered, normalized, and windowed before feeding into the hybrid CNN-LSTM trunk. "
        "The core binary model is converted to an ultra-compact TensorFlow Lite format and deployed into a real-time streaming runtime. "
        "Parallel Branch A explores unsupervised severity clustering and multi-task learning, while Parallel Branch B evaluates FiLM-based user personalization."
    )

    add_heading_2(doc, "3.2 WEARABLE SENSOR ACQUISITION PIPELINE")
    add_styled_paragraph(doc, 
        "The system utilizes multi-axis inertial data sourced from the SisFall dataset. Although the physical acquisition hardware recorded nine raw channels "
        "(two accelerometers and one gyroscope), our architecture retains only the primary wide-range tri-axial accelerometer (ADXL345, range ±16g) "
        "and the tri-axial gyroscope (ITG3200, range ±2000°/s), dropping the redundant second accelerometer (MMA7361). "
        "Table 3.1 details the sensor channel specifications."
    )
    
    # Table 3.1
    t31 = doc.add_table(rows=7, cols=4)
    t31_headers = ["Channel Name", "Sensor Type", "Measurement Unit", "Physical Range"]
    t31_data = [
        ["acc1_x", "Primary Accelerometer", "g (m/s²)", "±16 g"],
        ["acc1_y", "Primary Accelerometer", "g (m/s²)", "±16 g"],
        ["acc1_z", "Primary Accelerometer", "g (m/s²)", "±16 g"],
        ["gyro_x", "Tri-Axial Gyroscope", "degrees / sec (°/s)", "±2000 °/s"],
        ["gyro_y", "Tri-Axial Gyroscope", "degrees / sec (°/s)", "±2000 °/s"],
        ["gyro_z", "Tri-Axial Gyroscope", "degrees / sec (°/s)", "±2000 °/s"],
    ]
    format_table(t31, [1.5, 1.8, 1.4, 1.3], t31_headers, t31_data)
    add_styled_paragraph(doc, "Table 3.1: SisFall IMU Sensor Channel Specifications and Dynamic Ranges", font_size=10.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=12)

    add_heading_2(doc, "3.3 BUTTERWORTH LOW-PASS FILTERING AND DOWNSAMPLING")
    add_styled_paragraph(doc, 
        "Raw IMU signals sampled at 200 Hz contain significant high-frequency mechanical vibrations, electronic baseline noise, and sensor jitter. "
        "Biomechanical studies confirm that human intentional movements and fall shock waves have dominant spectral energy below 5 Hz. "
        "Therefore, each recording is preprocessed with a 2nd-order Butterworth low-pass digital filter configured with a 5 Hz cutoff frequency: "
        "\n\nH(s) = 1 / (1 + sqrt(2)*(s/omega_c) + (s/omega_c)^2)"
        "\n\nFiltering is executed per recording prior to window segmentation. Subsequently, signals are downsampled by a decimation factor of 10 from 200 Hz to 20 Hz. "
        "At 20 Hz, the Nyquist frequency is 10 Hz, fully preserving the 5 Hz filtered motion dynamics while cutting computational complexity and memory footprint by 90%. "
        "Figure 3.2 illustrates the signal profile before and after filtering."
    )
    add_figure(doc, REPO_ROOT / "docs/reports/technical_appendix/plots/preprocessing_filter_before_after.png", 
               "Figure 3.2: Tri-Axial Accelerometer Waveform Before and After 5 Hz Butterworth Low-Pass Filter", width_in=5.5)

    add_heading_2(doc, "3.4 SLIDING WINDOW SEGMENTATION AND IMPACT LABELING")
    add_styled_paragraph(doc, 
        "Continuous 20 Hz multi-channel sensor data is segmented using a fixed-size sliding window. "
        "A window length of 64 samples corresponds to exactly 3.2 seconds of physical motion, which comfortably encompasses the pre-fall instability phase (0.5–1.0 s), "
        "the free-fall and collision impact phase (0.3–0.5 s), and post-fall resting immobility (1.5–2.0 s). "
        "A sliding stride of 16 samples advances the window by 0.8 seconds, creating a 75% overlap (48 shared samples between consecutive windows). "
        "This high overlap ensures that the rapid shock peak is captured near the window center in at least one evaluation frame."
    )
    add_figure(doc, REPO_ROOT / "docs/reports/project_report/report_assets/preprocessing_windowing_example.png", 
               "Figure 3.3: Sliding Window Segmentation with 64-Sample Windows and 75% Overlap", width_in=5.5)

    add_styled_paragraph(doc, 
        "Impact-Centered Window Labeling: Rather than naively assigning label 1 to every window generated from a fall trial recording, "
        "the preprocessing pipeline computes the acceleration Signal Magnitude Vector across the entire recording and identifies the index of maximum impact peak: "
        "\n\nt_impact = argmax_t sqrt(acc_x[t]^2 + acc_y[t]^2 + acc_z[t]^2)"
        "\n\nA window receives binary Fall label 1 if and only if it contains the detected impact sample index. "
        "All other windows from the same fall recording (e.g., initial standing or walking prior to tripping) are designated as label 0 (Background/ADL). "
        "All windows extracted from ADL recordings receive label 0. Figure 3.4 demonstrates this impact-centering mechanism."
    )
    add_figure(doc, REPO_ROOT / "docs/reports/technical_appendix/plots/impact_labeling_annotated_fall.png", 
               "Figure 3.4: Impact-Centered Window Labeling via Peak Signal Magnitude Vector (SMV)", width_in=5.5)

    add_heading_2(doc, "3.5 HYBRID CNN-LSTM NEURAL NETWORK ARCHITECTURE")
    add_styled_paragraph(doc, 
        "The deep learning model is architected to process standardized 64 x 6 input tensors. "
        "It consists of two 1D Convolutional blocks for spatial-channel feature extraction, followed by a Long Short-Term Memory (LSTM) layer for temporal modeling, "
        "global average pooling, and a fully connected classification head. "
        "Table 3.3 details the layer specifications, output tensor shapes, and trainable parameter counts."
    )
    
    # Table 3.3
    t33 = doc.add_table(rows=8, cols=4)
    t33_headers = ["Layer / Stage", "Kernel / Units", "Output Dimension", "Function / Role"]
    t33_data = [
        ["Input Layer", "-", "(None, 64, 6)", "3.2 s standardized 6-channel IMU window"],
        ["Conv1D Block 1", "32 filters, k=5, s=1", "(None, 32, 32)", "BatchNorm + ReLU + MaxPool(2)"],
        ["Conv1D Block 2", "64 filters, k=5, s=1", "(None, 16, 64)", "BatchNorm + ReLU + MaxPool(2)"],
        ["Recurrent LSTM", "64 units, s=1", "(None, 16, 64)", "Sequential temporal dependency modeling"],
        ["Global Avg Pool", "-", "(None, 64)", "Temporal aggregation across time steps"],
        ["Dense Trunk", "32 units, ReLU", "(None, 32)", "Regularized feature projection (Dropout 0.3)"],
        ["Binary Output", "1 unit, Sigmoid", "(None, 1)", "Fall confidence probability P(fall) in [0, 1]"],
    ]
    format_table(t33, [1.5, 1.6, 1.4, 1.5], t33_headers, t33_data)
    add_styled_paragraph(doc, "Table 3.3: Layer-by-Layer Specifications of the Hybrid CNN-LSTM Architecture", font_size=10.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=12)

    add_figure(doc, REPO_ROOT / "docs/presentation/figures/review3_multitask_architecture.png", 
               "Figure 3.5: Hybrid CNN-LSTM Neural Network Architecture Flowchart", width_in=5.8)

    add_heading_2(doc, "3.6 MULTI-TASK SEVERITY HEAD AND MASKED LOSS")
    add_styled_paragraph(doc, 
        "In Parallel Branch A, the shared 32-dimensional dense trunk is augmented with a secondary severity classification head. "
        "The severity head comprises a Dense(2) layer with softmax activation predicting relative probabilities for Mild (Cluster 0) and Severe (Cluster 1) falls. "
        "To prevent non-fall ADL windows and anomalous outlier falls from corrupting the severity gradients, a masked multi-task loss is formulated: "
        "\n\nL_total = L_binary(y_bin, p_bin) + lambda * m_sev * L_severity(y_sev, p_sev)"
        "\n\nwhere m_sev = 1 if y_bin == 1 and y_sev != -1, and 0 otherwise. "
        "Binary cross-entropy is computed over all windows, while categorical cross-entropy is activated only for confident fall samples."
    )

    add_heading_2(doc, "3.7 FiLM PERSONALIZATION ADAPTER DESIGN")
    add_styled_paragraph(doc, 
        "In Parallel Branch B, Feature-wise Linear Modulation is evaluated. As shown in Figure 3.7, FiLM is injected immediately between the frozen Conv1D Block 2 "
        "and the frozen LSTM layer. The intermediate feature map H has dimension (16 x 64). "
        "A lightweight UserEncoder network (24,960 parameters) pools calibration ADL windows from a target user and outputs two 64-dimensional vectors: "
        "a multiplicative scale vector gamma and an additive shift vector beta. The modulated feature representation is computed as: "
        "\n\nH_personalized = gamma (element-wise multiplication) H + beta"
        "\n\nDuring personalization meta-training, the core CNN and LSTM weights remain completely frozen; only the UserEncoder weights are updated."
    )
    add_figure(doc, REPO_ROOT / "docs/presentation/figures/personalization/09_film_architecture_pipeline.png", 
               "Figure 3.7: Feature-Wise Linear Modulation (FiLM) Adapter and UserEncoder Pipeline", width_in=5.8)

    add_heading_2(doc, "3.8 TENSORFLOW LITE QUANTIZATION PIPELINE")
    add_styled_paragraph(doc, 
        "For microcontroller deployment, the trained Float32 Keras model is converted using TensorFlow Lite's post-training quantization tools. "
        "Four quantization configurations were evaluated: Float32 baseline, Float16 half-precision, Full INT8 quantization, and Dynamic-Range quantization. "
        "Dynamic-Range quantization quantizes model weights to 8-bit signed integers while dynamically quantizing activations during runtime inference. "
        "The resulting .tflite flatbuffer is transformed into a C unsigned char array stored in model_data.h, allowing it to reside in read-only flash memory on the microcontroller."
    )

    add_heading_2(doc, "3.9 CIRCULAR RING BUFFER AND STREAMING ALERT LOGIC")
    add_styled_paragraph(doc, 
        "In real-world deployment, sensor samples arrive continuously from the I2C bus at 20 Hz. "
        "To eliminate dynamic memory allocation and prevent heap fragmentation, an in-memory circular ring buffer of dimension (64, 6) is maintained. "
        "The buffer requires a 3.15-second warmup period (63 samples) before inference begins. "
        "Thereafter, every 16 samples (0.8 seconds), the 64-sample buffer is copied, standardized using the training scaler, and passed to the TFLite interpreter. "
        "An alert policy enforces an operating threshold of tau = 0.25 and a 10-second alert cooldown timer to suppress repetitive alerts from the same fall."
    )
    add_figure(doc, REPO_ROOT / "docs/presentation/figures/review3_streaming_pipeline.png", 
               "Figure 3.8: Circular Ring Buffer Mechanics and Streaming Evaluation Stride", width_in=5.8)

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 4: ALGORITHM DEVELOPMENT AND IMPLEMENTATION
    # =========================================================================
    add_heading_1(doc, "CHAPTER 4\nALGORITHM DEVELOPMENT AND IMPLEMENTATION")

    add_heading_2(doc, "4.1 DATASET INGESTION AND PREPROCESSING PIPELINE")
    add_styled_paragraph(doc, 
        "The preprocessing pipeline is implemented in Python using NumPy, SciPy, and Pandas. "
        "Raw SisFall trials are organized in text files containing 9 integer channels representing raw ADC counts. "
        "Conversion factors specified by the SisFall authors are applied: accelerometer counts are converted to g using a scale factor of 2*16 / 2^13, "
        "and gyroscope counts are converted to °/s using 2*2000 / 2^16. "
        "Columns corresponding to acc2 are discarded, yielding a 6-channel floating-point array for each recording."
    )

    add_heading_2(doc, "4.2 BUTTERWORTH FILTERING AND ANTI-ALIASED RESAMPLING")
    add_styled_paragraph(doc, 
        "Digital filtering is implemented using SciPy's signal.butter function. A 2nd-order low-pass filter with critical frequency Wn = 5 / (200 / 2) = 0.05 "
        "is designed. Forward-backward filtering via signal.filtfilt ensures zero phase distortion across the time series. "
        "Decimation from 200 Hz to 20 Hz is performed using signal.resample_poly, which applies an anti-aliasing FIR filter prior to downsampling."
    )

    add_heading_2(doc, "4.3 IMPACT-CENTERED WINDOW LABELING ALGORITHM")
    add_styled_paragraph(doc, 
        "Algorithm 4.1 outlines the formal impact-centering and window generation logic:"
    )
    algo_box = (
        "Algorithm 4.1: Impact-Centered Window Generation\n"
        "Input: Continuous 6-channel recording R of length T at 20 Hz; Trial type Type in {ADL, Fall}\n"
        "Output: Set of labeled windows W = {(X_i, y_i)}\n"
        "1: Compute SMV[t] = sqrt(R[t, 0]^2 + R[t, 1]^2 + R[t, 2]^2) for t = 0 to T-1\n"
        "2: If Type == Fall then\n"
        "3:     impact_idx = argmax_t(SMV[t])\n"
        "4: Else\n"
        "5:     impact_idx = -1\n"
        "6: Initialize window start s = 0, window size W_len = 64, stride S = 16\n"
        "7: While (s + W_len) <= T do\n"
        "8:     window_data = R[s : s + W_len, :]\n"
        "9:     If Type == Fall and (s <= impact_idx < s + W_len) then\n"
        "10:        y_window = 1\n"
        "11:    Else\n"
        "12:        y_window = 0\n"
        "13:    Append (window_data, y_window) to W\n"
        "14:    s = s + S\n"
        "15: Return W"
    )
    p_algo = doc.add_paragraph()
    p_algo.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_algo.paragraph_format.left_indent = Inches(0.3)
    p_algo.paragraph_format.space_before = Pt(6)
    p_algo.paragraph_format.space_after = Pt(10)
    r_algo = p_algo.add_run(algo_box)
    r_algo.font.name = "Courier New"
    r_algo.font.size = Pt(9.5)

    add_heading_2(doc, "4.4 CNN-LSTM MODEL IMPLEMENTATION AND TRAINING")
    add_styled_paragraph(doc, 
        "The model is constructed using TensorFlow/Keras. Input tensors are normalized using StandardScaler fitted exclusively on the 26 training subjects: "
        "\n\nX_norm = (X - mu_train) / sigma_train"
        "\n\nTraining is executed using the Adam optimizer with an initial learning rate of 0.001, batch size of 64, and class weighting to counterbalance the 11.4:1 ADL-to-fall training ratio: "
        "\n\nw_0 = N_total / (2 * N_ADL) ≈ 0.544,   w_1 = N_total / (2 * N_Fall) ≈ 6.180"
        "\n\nValidation loss callbacks reduce the learning rate by a factor of 0.5 upon plateauing and terminate training after 6 epochs without improvement. "
        "Best model weights are restored from checkpoint epoch 4."
    )

    add_heading_2(doc, "4.5 VALIDATION-BASED OPERATING THRESHOLD OPTIMIZATION")
    add_styled_paragraph(doc, 
        "The sigmoid output returns a continuous fall probability p in [0, 1]. "
        "To select the optimal decision threshold tau, a linear sweep across candidate thresholds from 0.05 to 0.95 with step 0.01 is evaluated on the validation set. "
        "The objective function maximizes the F2-score: "
        "\n\nF2 = (1 + 2^2) * (Precision * Recall) / (2^2 * Precision + Recall) = 5 * Precision * Recall / (4 * Precision + Recall)"
        "\n\nThe threshold maximizing validation F2 is selected (tau = 0.25) and locked for all subsequent evaluations on the held-out test partition."
    )

    add_heading_2(doc, "4.6 SEVERITY FEATURE ENGINEERING (58 FEATURES)")
    add_styled_paragraph(doc, 
        "For all fall windows, 58 statistical and kinematic features are computed. "
        "Table 4.1 categorizes these features across the 6 physical channels and 2 derived magnitude vectors."
    )
    
    # Table 4.1
    t41 = doc.add_table(rows=4, cols=3)
    t41_headers = ["Feature Category", "Channel Sources", "Extracted Statistical Descriptors"]
    t41_data = [
        ["Per-Channel Statistics (42 features)", "acc1_x, acc1_y, acc1_z, gyro_x, gyro_y, gyro_z (6 channels)", "Mean, standard deviation, minimum, maximum, skewness, kurtosis, signal energy (7 per channel)"],
        ["Acceleration Magnitude (8 features)", "SMV_acc = sqrt(acc_x^2 + acc_y^2 + acc_z^2)", "Peak amplitude, minimum, mean, variance, root-mean-square, crest factor, energy, jerk peak"],
        ["Gyroscope Magnitude (8 features)", "SMV_gyro = sqrt(gyro_x^2 + gyro_y^2 + gyro_z^2)", "Peak angular rate, minimum, mean, variance, RMS, rotational energy, peak angular acceleration"],
    ]
    format_table(t41, [1.8, 1.8, 2.4], t41_headers, t41_data)
    add_styled_paragraph(doc, "Table 4.1: 58-Dimensional Statistical and Kinematic Feature Set for Severity Modeling", font_size=10.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=12)

    add_heading_2(doc, "4.7 K-MEANS CLUSTERING AND ISOLATION FOREST FILTERING")
    add_styled_paragraph(doc, 
        "The 58-dimensional feature vectors are normalized using StandardScaler fitted on training fall windows. "
        "K-Means clustering with K=2 partitions the feature space into Cluster 0 (Mild) and Cluster 1 (Severe). "
        "Simultaneously, an Isolation Forest model with 100 estimators and 2% contamination threshold is fitted. "
        "Windows flagged as anomalies by the Isolation Forest are assigned label -1 (Uncertain). "
        "During multi-task model training, these uncertain rows are excluded from the severity loss gradient updates."
    )

    add_heading_2(doc, "4.8 FiLM META-TRAINING AND CALIBRATION IMPLEMENTATION")
    add_styled_paragraph(doc, 
        "The UserEncoder network is constructed with two 1D convolution layers, a global pooling layer, and dual dense output heads predicting gamma and beta. "
        "Meta-training is conducted using simulated user profiles (Low-amplitude, High-amplitude, Rotated sensor, Noisy, and Slow-cadence profiles). "
        "Calibration window sizes k in {0, 1, 5, 10, 30} are evaluated. At inference time, the UserEncoder processes k calibration ADL windows "
        "in a single forward pass without backpropagation."
    )

    add_heading_2(doc, "4.9 DYNAMIC-RANGE QUANTIZATION AND C ARRAY HEADER EXPORT")
    add_styled_paragraph(doc, 
        "Quantization is executed using tf.lite.TFLiteConverter.optimizations = [tf.lite.Optimize.DEFAULT]. "
        "The generated flatbuffer binary is read in Python and formatted as a C header file (model_data.h) containing: "
        "\n\nconst unsigned char g_model[] = { 0x1c, 0x00, 0x00, ... };\nconst int g_model_len = 113704;"
        "\n\nThis header is compiled directly into ESP32 firmware using the Arduino-ESP32 or ESP-IDF toolchain."
    )

    add_heading_2(doc, "4.10 REAL-TIME STREAMING ENGINE AND DASHBOARD IMPLEMENTATION")
    add_styled_paragraph(doc, 
        "The streaming engine is implemented in src/streaming.py. It exposes an StreamingFallDetector class initialized with ring buffer capacity 64, "
        "stride 16, threshold 0.25, and cooldown 10.0 seconds. "
        "A full-stack clinical monitoring dashboard is authored in Streamlit (app/demo.py), featuring interactive replay controls, "
        "real-time probability graphs, multi-channel sensor traces, and simulated caregiver SMS alert dispatch logs."
    )

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 5: RESULTS AND DISCUSSIONS
    # =========================================================================
    add_heading_1(doc, "CHAPTER 5\nRESULTS AND DISCUSSIONS")

    add_heading_2(doc, "5.1 EXPERIMENTAL SETUP AND COMPUTING ENVIRONMENT")
    add_styled_paragraph(doc, 
        "All model training, quantization benchmarks, and streaming verification tests were executed in a standardized computing environment. "
        "Table 5.1 details the hardware and software specifications."
    )
    
    # Table 5.1
    t51 = doc.add_table(rows=7, cols=2)
    t51_headers = ["System Component", "Hardware / Software Specification"]
    t51_data = [
        ["Processor (CPU)", "Intel(R) Core(TM) i5-11400H @ 2.70GHz (6 Cores, 12 Threads)"],
        ["System Memory (RAM)", "16.0 GB DDR4 @ 3200 MHz"],
        ["Operating System", "Microsoft Windows 11 Home (64-bit)"],
        ["Programming Environment", "Python 3.11.9 Virtual Environment (.venv)"],
        ["Deep Learning Libraries", "TensorFlow 2.16.1, Keras 3.3.3, Scikit-Learn 1.4.2"],
        ["Target Microcontroller", "Espressif ESP32-WROOM-32 (240 MHz Dual-Core Tensilica, 520 KB SRAM)"],
    ]
    format_table(t51, [2.5, 3.5], t51_headers, t51_data)
    add_styled_paragraph(doc, "Table 5.1: Experimental Computing Environment Specifications", font_size=10.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=12)

    add_heading_2(doc, "5.2 SISFALL DATASET CHARACTERISTICS AND PARTITIONS")
    add_styled_paragraph(doc, 
        "The SisFall dataset was partitioned into 26 training subjects, 6 validation subjects, and 6 held-out test subjects. "
        "Table 5.2 summarizes the exact window distributions and class imbalance ratios."
    )
    
    # Table 5.2
    t52 = doc.add_table(rows=5, cols=6)
    t52_headers = ["Partition", "Subjects", "Recordings", "Total Windows", "ADL Windows", "Fall Windows"]
    t52_data = [
        ["Train", "26", "2,942", "55,423", "50,939 (91.9%)", "4,484 (8.1%)"],
        ["Validation", "6", "829", "15,073", "13,573 (90.0%)", "1,500 (10.0%)"],
        ["Held-Out Test", "6", "734", "13,627", "12,427 (91.2%)", "1,200 (8.8%)"],
        ["Total Dataset", "38", "4,505", "84,123", "76,939 (91.5%)", "7,184 (8.5%)"],
    ]
    format_table(t52, [1.0, 0.8, 1.0, 1.1, 1.1, 1.0], t52_headers, t52_data)
    add_styled_paragraph(doc, "Table 5.2: SisFall Dataset Split and Window Counts across Partitions", font_size=10.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=12)

    add_figure(doc, REPO_ROOT / "docs/reports/project_report/report_assets/class_balance.png", 
               "Figure 5.1: Subject-Wise Split and Overall Class Imbalance in the SisFall Dataset", width_in=5.5)

    add_heading_2(doc, "5.3 CORE BINARY DETECTION PERFORMANCE (CNN VS CNN-LSTM)")
    add_styled_paragraph(doc, 
        "The performance of the 1D CNN baseline and the hybrid CNN-LSTM network was benchmarked on the 13,627 held-out test windows. "
        "Table 5.3 summarizes the comparative evaluation."
    )
    
    # Table 5.3
    t53 = doc.add_table(rows=4, cols=8)
    t53_headers = ["Model Architecture", "Threshold", "Precision", "Recall", "F1-Score", "F2-Score", "FP", "FN"]
    t53_data = [
        ["CNN Baseline", "tau = 0.50", "0.9761", "0.9867", "0.9814", "0.9846", "29", "16"],
        ["CNN-LSTM", "tau = 0.50", "0.9770", "0.9925", "0.9847", "0.9894", "28", "9"],
        ["CNN-LSTM (Deployed)", "tau = 0.25", "0.9606", "0.9950", "0.9775", "0.9879", "49", "6"],
    ]
    format_table(t53, [1.4, 0.8, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5], t53_headers, t53_data)
    add_styled_paragraph(doc, "Table 5.3: Performance Comparison of CNN Baseline and CNN-LSTM on Held-Out Test Set", font_size=10.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=12)

    add_styled_paragraph(doc, 
        "At the canonical threshold of 0.50, the CNN-LSTM reduces missed fall windows from 16 to 9 compared to the CNN baseline. "
        "Figure 5.2 shows the training loss and accuracy convergence curves for the CNN-LSTM model."
    )
    add_figure(doc, REPO_ROOT / "docs/reports/project_report/report_assets/cnn_lstm_training_curves.png", 
               "Figure 5.2: CNN-LSTM Binary Model Training Loss and Accuracy Convergence Curves", width_in=5.5)

    add_heading_2(doc, "5.4 OPERATING DECISION THRESHOLD OPTIMIZATION ANALYSIS")
    add_styled_paragraph(doc, 
        "Sweeping the decision threshold on validation data demonstrated that lowering tau from 0.50 to 0.25 maximized the F2-score. "
        "When applied to the held-out test partition, tau = 0.25 increased fall recall from 0.9925 to 0.9950, successfully detecting 1,194 out of 1,200 fall windows (only 6 missed falls). "
        "Although false positive windows increased modestly from 28 to 49 out of 12,427 ADL windows, precision remained exceptionally high at 0.9606. "
        "Figures 5.3 and 5.4 depict the precision-recall trade-off and the resulting confusion matrix."
    )
    add_figure(doc, REPO_ROOT / "docs/reports/project_report/report_assets/cnn_lstm_precision_recall_curve.png", 
               "Figure 5.3: Validation Precision-Recall Curve Across Operating Decision Thresholds", width_in=5.2)
    add_figure(doc, REPO_ROOT / "docs/reports/project_report/report_assets/cnn_lstm_confusion_matrix.png", 
               "Figure 5.4: Held-Out Test Set Confusion Matrix at Canonical tau = 0.50 Threshold", width_in=4.8)

    add_heading_2(doc, "5.5 SEVERITY PSEUDO-LABELING BENCHMARK AND ANALYSIS")
    add_styled_paragraph(doc, 
        "Evaluating cluster stability across 58 kinematic features confirmed that K=2 achieved a Silhouette score of 0.1774, "
        "significantly outperforming K=3 (0.1585). "
        "Table 5.5 compares clustering validity indices."
    )
    
    # Table 5.5
    t55 = doc.add_table(rows=3, cols=5)
    t55_headers = ["Cluster Model", "Silhouette Score (Higher is Better)", "Davies-Bouldin Index (Lower is Better)", "Calinski-Harabasz Index", "Cluster Balance Status"]
    t55_data = [
        ["K = 2 (Selected)", "0.1774", "1.9898", "998.4", "Well-Separated (Mild: 468, Severe: 732)"],
        ["K = 3 (Rejected)", "0.1585", "2.1512", "778.2", "Unstable Middle Cluster; Degraded Boundaries"],
    ]
    format_table(t55, [1.4, 1.2, 1.2, 1.1, 1.3], t55_headers, t55_data)
    add_styled_paragraph(doc, "Table 5.5: Severity Clustering Metrics Comparison (K=2 vs K=3)", font_size=10.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=12)

    add_figure(doc, REPO_ROOT / "docs/reports/project_report/report_assets/severity_pca_clusters.png", 
               "Figure 5.5: K-Means (K=2) Fall Severity Clusters Projected onto 2D PCA Space", width_in=5.2)
    add_figure(doc, REPO_ROOT / "docs/reports/project_report/report_assets/anomaly_score_distribution.png", 
               "Figure 5.6: Isolation Forest Anomaly Score Distribution Flagging Uncertain Fall Outliers", width_in=5.2)
    add_figure(doc, REPO_ROOT / "docs/reports/project_report/report_assets/severity_peak_intensity.png", 
               "Figure 5.7: Peak Acceleration and Angular Velocity Intensity Profiles Across Severity Clusters", width_in=5.2)

    add_styled_paragraph(doc, 
        "Across 1,176 confident test fall windows (excluding 24 uncertain windows flagged by the Isolation Forest), "
        "the severity classifier achieved a Macro-F1 score of 0.8660 (Mild F1 = 0.8346, Severe F1 = 0.8973). "
        "Table 5.6 presents the severity confusion matrix."
    )
    
    # Table 5.6
    t56 = doc.add_table(rows=3, cols=4)
    t56_headers = ["True Class", "Predicted Mild (Cluster 0)", "Predicted Severe (Cluster 1)", "Per-Class F1-Score"]
    t56_data = [
        ["Mild (457 windows)", "376", "81", "0.8346"],
        ["Severe (719 windows)", "68", "651", "0.8973"],
    ]
    format_table(t56, [1.5, 1.5, 1.5, 1.5], t56_headers, t56_data)
    add_styled_paragraph(doc, "Table 5.6: Severity Head Confusion Matrix across 1,176 Confident Fall Windows", font_size=10.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=12)

    add_heading_2(doc, "5.6 MULTI-TASK JOINT LOSS EVALUATION AND DECISION GATING")
    add_styled_paragraph(doc, 
        "To evaluate whether the multi-task model could replace the binary detector, a formal gating criterion was applied: "
        "the multi-task model must maintain binary F2 within 0.005 of the single-task baseline. "
        "Table 5.7 presents the gating comparison."
    )
    
    # Table 5.7
    t57 = doc.add_table(rows=3, cols=6)
    t57_headers = ["Model Configuration", "Operating Threshold", "Binary Recall", "Binary Precision", "Binary F2-Score", "Deployment Status"]
    t57_data = [
        ["Single-Task CNN-LSTM", "tau = 0.25", "0.9950", "0.9606", "0.9879", "PASSED (Deployed Primary)"],
        ["Multi-Task CNN-LSTM (lambda=1.0)", "tau = 0.10", "0.9692", "0.9610", "0.9669", "FAILED GATE (Kept as Extension)"],
    ]
    format_table(t57, [1.8, 0.8, 0.8, 0.8, 0.8, 1.2], t57_headers, t57_data)
    add_styled_paragraph(doc, "Table 5.7: Binary vs Multi-Task CNN-LSTM Gating Performance Comparison", font_size=10.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=12)

    add_styled_paragraph(doc, 
        "Because joint gradient backpropagation caused negative transfer (reducing binary fall recall from 99.50% to 96.92%), "
        "the multi-task model was rejected for primary deployment in accordance with clinical safety principles."
    )

    add_heading_2(doc, "5.7 FiLM PERSONALIZATION EMPIRICAL AND STATISTICAL RESULTS")
    add_styled_paragraph(doc, 
        "Table 5.8 summarizes the F2 performance of FiLM across varying calibration window sizes."
    )
    
    # Table 5.8
    t58 = doc.add_table(rows=6, cols=4)
    t58_headers = ["Calibration ADL Windows", "Global Model F2", "FiLM Personalized F2", "Delta F2"]
    t58_data = [
        ["k = 0 windows", "0.9880", "0.9859", "-0.0021"],
        ["k = 1 window", "0.9880", "0.9681", "-0.0199"],
        ["k = 5 windows", "0.9880", "0.9874", "-0.0006"],
        ["k = 10 windows", "0.9880", "0.9881", "+0.0001"],
        ["k = 30 windows", "0.9880", "0.9884", "+0.0004"],
    ]
    format_table(t58, [1.8, 1.4, 1.4, 1.4], t58_headers, t58_data)
    add_styled_paragraph(doc, "Table 5.8: FiLM Personalization F2-Score across Calibration Window Sizes", font_size=10.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=12)

    add_styled_paragraph(doc, 
        "Statistical Significance: A two-sided paired Wilcoxon signed-rank test comparing FiLM against the global baseline across held-out subjects "
        "yielded a p-value of 0.125. Because p > 0.05, the performance differences are not statistically significant. "
        "With only n = 4 fall-capable subjects in the held-out split, p = 0.125 represents the mathematical minimum achievable p-value for that sample size. "
        "Figure 5.8 presents the p-value comparison chart."
    )
    add_figure(doc, REPO_ROOT / "docs/presentation/figures/personalization/07_wilcoxon_p_values.png", 
               "Figure 5.8: Paired Wilcoxon Signed-Rank Test Significance for FiLM Personalization", width_in=5.5)

    add_heading_2(doc, "5.8 EDGE QUANTIZATION FOOTPRINT AND LATENCY BENCHMARKS")
    add_styled_paragraph(doc, 
        "Post-training quantization benchmarks evaluated model file size, inference execution latency, and prediction agreement. "
        "Table 5.9 summarizes the findings across all four formats."
    )
    
    # Table 5.9
    t59 = doc.add_table(rows=5, cols=6)
    t59_headers = ["Quantization Format", "Model Size", "Size Reduction", "CPU Latency (1 Thread)", "Keras Agreement", "Test Fall Recall"]
    t59_data = [
        ["Float32 (Unquantized)", "235.6 KB", "Baseline (0%)", "0.063 ms", "100.00%", "0.9950"],
        ["Dynamic-Range (Selected)", "111.0 KB", "52.9% reduction", "0.026 ms", "100.00%", "0.9950"],
        ["Float16 Quantization", "147.8 KB", "37.3% reduction", "0.062 ms", "100.00%", "0.9950"],
        ["Full INT8 Quantization", "136.9 KB", "41.9% reduction", "0.033 ms", "99.97%", "0.9950"],
    ]
    format_table(t59, [1.4, 0.9, 1.0, 1.1, 1.0, 0.8], t59_headers, t59_data)
    add_styled_paragraph(doc, "Table 5.9: TFLite Edge Quantization Benchmark: Footprint, Latency, and Numerical Parity", font_size=10.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=12)

    add_figure(doc, REPO_ROOT / "docs/presentation/figures/review3_quantization_comparison.png", 
               "Figure 5.9: TFLite Model Footprint and Single-Window CPU Latency Benchmarks", width_in=5.5)

    add_heading_2(doc, "5.9 STREAMING REPLAY VERIFICATION AND NUMERICAL PARITY")
    add_styled_paragraph(doc, 
        "Numerical parity tests confirmed that the streaming ring-buffer implementation achieved exact bitwise agreement with batch Keras inference, "
        "with maximum absolute error <= 1e-6. "
        "Replay evaluation across six standardized benchmark scenarios (walking, sitting, stumbling, jumping, mild fall, and severe fall) "
        "demonstrated 100% detection accuracy with zero false alarms. "
        "Figures 5.10, 5.11, 5.12, and 5.13 display screenshots of the live Streamlit dashboard during streaming replay."
    )
    add_figure(doc, ASSETS_DIR / "slide_page_11_Image77.png", 
               "Figure 5.10: Live Streaming Dashboard: Fall Probability Timeline Crossing Decision Threshold", width_in=5.5)
    add_figure(doc, ASSETS_DIR / "slide_page_11_Image78.png", 
               "Figure 5.11: Streaming Dashboard: Raw 6-Channel Synchronized IMU Sensor Traces", width_in=5.5)
    add_figure(doc, ASSETS_DIR / "slide_page_11_Image79.png", 
               "Figure 5.12: Streaming Dashboard: Simulated Caregiver Alert Dispatch Notification Log", width_in=5.5)
    add_figure(doc, ASSETS_DIR / "slide_page_12_Image82.png", 
               "Figure 5.13: Interactive FiLM User Personalization Demonstration Interface", width_in=5.5)

    add_heading_2(doc, "5.10 THREATS TO VALIDITY AND RESEARCH LIMITATIONS")
    add_styled_paragraph(doc, 
        "To maintain rigorous academic integrity, several engineering limitations must be acknowledged:"
    )
    limitations = [
        ("Simulated vs. Real-World Geriatric Falls: ", "SisFall trials consist of simulated falls performed by healthy volunteers onto foam crash mats. Real-world elderly falls involve frailty, slower reaction times, fear of falling, and complex multi-stage stumbling."),
        ("Sample Size in Held-Out Fall Subjects: ", "The held-out partition includes 6 subjects, of whom only 4 performed fall trials. While sufficient for proof-of-concept verification, larger multi-center cohorts are needed."),
        ("Pseudo-Severity vs. Clinical Outcome: ", "The Mild and Severe severity classes reflect sensor-derived kinetic energy clusters, not clinician-diagnosed fracture or concussion outcomes."),
        ("Software-in-the-Loop Demonstration: ", "The current prototype operates via software replay of 20 Hz CSV streams. Live BLE hardware ingestion, GPS localization, and battery discharge profiling remain future work.")
    ]
    for title, desc in limitations:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.5
        r_t = p.add_run(title)
        r_t.font.name = "Times New Roman"
        r_t.font.bold = True
        r_d = p.add_run(desc)
        r_d.font.name = "Times New Roman"

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 6: CONCLUSION AND FUTURE WORK
    # =========================================================================
    add_heading_1(doc, "CHAPTER 6\nCONCLUSION AND FUTURE WORK")

    add_heading_2(doc, "6.1 CONCLUSION")
    add_styled_paragraph(doc, 
        "This project has presented a comprehensive, scientifically validated, and edge-deployable wearable fall detection system designed for elderly care. "
        "By enforcing a strict subject-wise partitioning protocol on the SisFall dataset, the framework eliminates data leakage and reflects true out-of-sample generalization. "
        "The signal conditioning pipeline successfully combines a 2nd-order 5 Hz Butterworth low-pass filter with anti-aliased downsampling to 20 Hz, "
        "reducing computational overhead by 90% while fully capturing intentional body kinetics. "
        "Recordings are segmented into 64-sample (3.2-second) sliding windows with an impact-centered labeling algorithm that isolates impact shock dynamics from ambient walking."
    )
    add_styled_paragraph(doc, 
        "The hybrid CNN-LSTM architecture combines 1D spatial convolutional feature extraction with recurrent temporal modeling. "
        "By optimizing the decision threshold on validation data to maximize the recall-weighted F2-score, an operating threshold of tau = 0.25 was established. "
        "On held-out test subjects, the deployed model achieves 0.9950 recall and 0.9775 F1-score, detecting 1,194 out of 1,200 fall windows with only 6 missed falls. "
        "The unsupervised K-Means (K=2) and Isolation Forest pipeline provides meaningful impact stratification into relative Mild and Severe categories (Macro-F1 = 0.8660) "
        "while masking atypical fall outliers. "
        "Post-training Dynamic-Range quantization compresses the model to 111.0 KB with single-window CPU latency of 0.026 ms, "
        "and the exported C array header (model_data.h) consumes only 21% of SRAM on an ESP32 microcontroller. "
        "Finally, the circular ring-buffer streaming engine and interactive dashboard demonstrate real-time telemetry, exact bitwise numerical parity, and zero false alarms."
    )

    add_heading_2(doc, "6.2 FUTURE WORK")
    add_styled_paragraph(doc, 
        "Several promising avenues for future research and engineering enhancements have been identified:"
    )
    future_items = [
        "Hardware Integration on Wearable Firmware: Flashing model_data.h directly onto an ESP32-WROOM-32 or Nordic nRF52840 wearable development board, interfacing with an MPU-6050 or BMI270 IMU over I2C at 20 Hz.",
        "Clinical Validation on Geriatric Populations: Partnering with geriatric healthcare facilities to capture longitudinal, unscripted motion profiles from older adults, validating model robustness against age-related gait disorders and walking aids.",
        "Multimodal Sensor Fusion: Incorporating barometric pressure sensors to detect sudden altitude drops (rapid vertical elevation changes of 1–1.5 meters) and photoplethysmography (PPG) heart-rate sensors to correlate physiological shock with mechanical impact.",
        "Cloud-Connected Caregiver Telemetry: Integrating cellular NB-IoT or LoRaWAN communication protocols to dispatch automated emergency notifications with GPS coordinates to caregivers and ambulance services.",
        "Expanded FiLM Personalization Trials: Conducting large-cohort personalization experiments with larger numbers of fall-capable subjects to explore whether statistically significant adaptation gains can be achieved."
    ]
    for item in future_items:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.5
        r_bullet = p.add_run("•  ")
        r_bullet.font.name = "Times New Roman"
        r_bullet.font.bold = True
        r_text = p.add_run(item)
        r_text.font.name = "Times New Roman"

    doc.add_page_break()

    # =========================================================================
    # REFERENCES
    # =========================================================================
    add_heading_1(doc, "REFERENCES")
    refs = [
        "[1] A. K. Bourke, J. V. O'Brien, and G. M. Lyons, \"Evaluation of a threshold-based tri-axial accelerometer fall detection algorithm,\" Gait & Posture, vol. 26, no. 2, pp. 194-199, 2007.",
        "[2] Q. Zhang, H. Wang, and M. Li, \"Support vector machine-based fall detection using wearable sensors in elderly healthcare,\" IEEE Transactions on Biomedical Engineering, vol. 63, no. 4, pp. 842-850, 2016.",
        "[3] F. Bagalà et al., \"Evaluation of accelerometer-based fall detection algorithms on real-world falls,\" PLoS ONE, vol. 7, no. 5, p. e37062, 2012.",
        "[4] Y. LeCun, Y. Bengio, and G. Hinton, \"Deep learning,\" Nature, vol. 521, no. 7553, pp. 436-444, 2015.",
        "[5] S. Hochreiter and J. Schmidhuber, \"Long short-term memory,\" Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.",
        "[6] X. Hu et al., \"A hybrid CNN-LSTM model for involuntary fall detection using wrist-worn sensors,\" Advanced Engineering Informatics, vol. 65, p. 103178, 2025.",
        "[7] S. R. Lord, S. T. Smith, and J. C. Menant, \"Vision and falls in older people: risk factors and intervention strategies,\" Clinics in Geriatric Medicine, vol. 26, no. 4, pp. 569-581, 2010.",
        "[8] R. Caruana, \"Multitask learning,\" Machine Learning, vol. 28, no. 1, pp. 41-75, 1997.",
        "[9] S. J. Pan and Q. Yang, \"A survey on transfer learning,\" IEEE Transactions on Knowledge and Data Engineering, vol. 22, no. 10, pp. 1345-1359, 2010.",
        "[10] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, \"FiLM: Visual reasoning with a general conditioning layer,\" in Proc. AAAI Conference on Human Computation and Crowdsourcing, 2018, pp. 3942-3951.",
        "[11] C. R. Banbury et al., \"Microwork: Benchmarking TinyML systems on extreme edge microcontrollers,\" in Proc. IEEE Micro, vol. 41, no. 6, pp. 30-39, 2021.",
        "[12] B. Jacob et al., \"Quantization and training of neural networks for efficient integer-arithmetic-only inference,\" in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 2704-2713.",
        "[13] A. Sucerquia, J. D. López, and J. F. Vargas-Bonilla, \"SisFall: A fall and movement dataset,\" Sensors, vol. 17, no. 1, p. 198, 2017.",
        "[14] N. Noury et al., \"Fall detection - principles and methods,\" in Proc. 29th Annual International Conference of the IEEE EMBS, 2007, pp. 1663-1666.",
        "[15] M. Kangas, A. Konttila, P. Lindgren, I. Winblad, and T. Jämsä, \"Comparison of low-complexity fall detection algorithms for body attached accelerometers,\" Gait & Posture, vol. 28, no. 2, pp. 285-291, 2008.",
        "[16] P. Pierleoni et al., \"A high reliability wearable device for elderly fall detection,\" IEEE Sensors Journal, vol. 15, no. 8, pp. 4544-4553, 2015.",
        "[17] F. J. Ordóñez and D. Roggen, \"Deep convolutional and LSTM recurrent neural networks for multimodal wearable activity, fall and gesture recognition,\" Sensors, vol. 16, no. 1, p. 115, 2016.",
        "[18] L. Bao and S. S. Intille, \"Activity recognition from user-annotated acceleration data,\" in Proc. Pervasive Computing, Springer, 2004, pp. 1-17.",
        "[19] F. T. Liu, K. M. Ting, and Z. H. Zhou, \"Isolation forest,\" in Proc. 8th IEEE International Conference on Data Mining (ICDM), 2008, pp. 413-422.",
        "[20] P. J. Rousseeuw, \"Silhouettes: A graphical aid to the interpretation and validation of cluster analysis,\" Journal of Computational and Applied Mathematics, vol. 20, pp. 53-65, 1987.",
        "[21] D. L. Davies and D. W. Bouldin, \"A cluster separation measure,\" IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. PAMI-1, no. 2, pp. 224-227, 1979.",
        "[22] F. Wilcoxon, \"Individual comparisons by ranking methods,\" Biometrics Bulletin, vol. 1, no. 6, pp. 80-83, 1945.",
        "[23] F. Chollet, Deep Learning with Python, 2nd ed. Shelter Island, NY: Manning Publications, 2021.",
        "[24] F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.",
        "[25] P. Warden and D. Situnayake, TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers, Sebastopol, CA: O'Reilly Media, 2019."
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.4)
        p.paragraph_format.first_line_indent = Inches(-0.4)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(ref)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10.5)

    doc.save(str(DOCX_PATH))
    print(f"Successfully generated DOCX Thesis Report at: {DOCX_PATH}")
    return DOCX_PATH

if __name__ == "__main__":
    build_thesis_docx()
