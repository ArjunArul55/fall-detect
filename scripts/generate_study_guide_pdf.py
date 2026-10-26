"""
Generate the comprehensive Study & Viva Defense Handbook PDF:
'Wearable Fall Detection System - Project Review & Viva Defense Study Guide'
Includes deep technical explanations, 'why we used it' rationales, embedded visualizations,
slide-by-slide presentation scripts, and the top 20 master defense questions.
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs" / "thesis"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = DOCS_DIR / "Wearable_Fall_Detection_Study_and_Defense_Guide.pdf"
ASSETS_DIR = DOCS_DIR / "extracted_assets"

class StudyGuideCanvas(canvas.Canvas):
    """Numbered canvas with running header and footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, total_pages):
        page_num = self._pageNumber
        if page_num == 1:
            return  # Cover page

        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#1E293B"))

        # Header
        self.drawString(1.0 * inch, 11.69 * inch - 0.65 * inch, "PROJECT REVIEW 3 — VIVA & PRESENTATION STUDY GUIDE")
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawRightString(8.27 * inch - 1.0 * inch, 11.69 * inch - 0.65 * inch, "Wearable Fall Detection System")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(1.0 * inch, 11.69 * inch - 0.72 * inch, 8.27 * inch - 1.0 * inch, 11.69 * inch - 0.72 * inch)

        # Footer
        self.line(1.0 * inch, 0.75 * inch, 8.27 * inch - 1.0 * inch, 0.75 * inch)
        self.drawString(1.0 * inch, 0.55 * inch, "Madras Institute of Technology, Anna University — Dept. of Information Technology")
        self.drawRightString(8.27 * inch - 1.0 * inch, 0.55 * inch, f"Page {page_num} of {total_pages}")
        self.restoreState()

def build_study_guide_pdf():
    print(f"Generating comprehensive Study Guide PDF at: {PDF_PATH}...")
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=1.0 * inch,
        rightMargin=1.0 * inch,
        topMargin=1.0 * inch,
        bottomMargin=1.0 * inch
    )

    styles = getSampleStyleSheet()

    # Typography Styles
    cover_title = ParagraphStyle('CoverTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=20, leading=26, alignment=TA_CENTER, textColor=colors.HexColor("#0B2545"))
    cover_sub = ParagraphStyle('CoverSub', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=12, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#1E3A8A"), spaceAfter=10)
    cover_meta = ParagraphStyle('CoverMeta', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=15, alignment=TA_CENTER, textColor=colors.HexColor("#334155"))
    
    h1 = ParagraphStyle('H1', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=14, leading=18, alignment=TA_LEFT, textColor=colors.HexColor("#0B2545"), spaceBefore=14, spaceAfter=8)
    h2 = ParagraphStyle('H2', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11.5, leading=15, alignment=TA_LEFT, textColor=colors.HexColor("#1E3A8A"), spaceBefore=10, spaceAfter=4)
    h3 = ParagraphStyle('H3', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, leading=13, alignment=TA_LEFT, textColor=colors.HexColor("#0F172A"), spaceBefore=6, spaceAfter=2)
    
    body = ParagraphStyle('Body', parent=styles['Normal'], fontName='Helvetica', fontSize=9.5, leading=14, alignment=TA_JUSTIFY, textColor=colors.HexColor("#1E293B"), spaceAfter=6)
    body_bold = ParagraphStyle('BodyBold', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9.5, leading=14, alignment=TA_JUSTIFY, textColor=colors.HexColor("#0F172A"), spaceAfter=6)
    say_script = ParagraphStyle('SayScript', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=9.5, leading=14, alignment=TA_JUSTIFY, textColor=colors.HexColor("#0369A1"), spaceAfter=6)
    
    callout = ParagraphStyle('Callout', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=13.5, alignment=TA_JUSTIFY, textColor=colors.HexColor("#0F172A"), leftIndent=12, rightIndent=12)
    caption = ParagraphStyle('Caption', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#475569"), spaceBefore=3, spaceAfter=10)

    story = []

    # =========================================================================
    # COVER / HEADER PAGE
    # =========================================================================
    story.append(Spacer(1, 0.4 * inch))
    emblem_path = ASSETS_DIR / "page_0_Image1.jpg"
    if emblem_path.exists():
        story.append(Image(str(emblem_path), width=1.3 * inch, height=1.3 * inch))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("PROJECT REVIEW 3 & VIVA DEFENSE STUDY GUIDE", cover_title))
    story.append(Paragraph("Wearable Fall Detection System for Elderly People Using Deep Learning", cover_sub))
    story.append(Spacer(1, 0.15 * inch))

    meta_box = [
        [Paragraph("<b>Institution:</b> Madras Institute of Technology Campus, Anna University", cover_meta)],
        [Paragraph("<b>Department:</b> Department of Information Technology (Academic Year: 2023–2027)", cover_meta)],
        [Paragraph("<b>Course:</b> IT23702 / IT5712 - PROJECT 1", cover_meta)],
        [Paragraph("<b>Supervisor:</b> Dr. S. Umamaheswari, Associate Professor", cover_meta)],
        [Paragraph("<b>Team Members:</b> Sarath R (2023506065), Ajay Kumar S (2023506105), Arjun A (2023506063), Kris Virshak (2023506124)", cover_meta)],
    ]
    t_meta = Table(meta_box, colWidths=[6.2 * inch])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 0.25 * inch))

    intro_box = (
        "<b>PURPOSE OF THIS STUDY GUIDE:</b><br/>"
        "This document is authored specifically for the project team to master every theoretical, mathematical, architectural, "
        "and empirical detail of the Wearable Fall Detection System before the Review 3 committee and external viva examiners. "
        "It breaks down the overarching project story, provides explicit technical justifications for every technology chosen, "
        "explains all embedded figures and visualizations, walks through every presentation slide with verbatim spoken scripts, "
        "and answers the top 20 toughest defense questions with bulletproof mathematical and clinical justifications."
    )
    p_intro = Paragraph(intro_box, callout)
    t_intro = Table([[p_intro]], colWidths=[6.2 * inch])
    t_intro.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#BFDBFE")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_intro)
    story.append(PageBreak())

    # =========================================================================
    # PART 1: THE BIG PICTURE & SYSTEM ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("PART 1: THE CORE PROJECT STORY & ARCHITECTURE", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0B2545"), spaceAfter=10))

    story.append(Paragraph("1.1 The Clinical Problem We Solve", h2))
    story.append(Paragraph(
        "Falls in older adults are the second leading cause of unintentional injury deaths worldwide. In elderly populations, "
        "falls often trigger the <b>'long-lie state'</b>—remaining on the floor unable to rise for more than an hour. "
        "Prolonged immobilization leads to hypothermia, pressure sores, dehydration, muscle breakdown (rhabdomyolysis), and high mortality. "
        "Existing smartwatches suffer from high false-alarm rates during vigorous activities of daily living (ADLs) and lack any indication of impact severity. "
        "Our objective is to deliver an ultra-high recall, lightweight deep learning model that detects falls in real time on low-cost wearable microcontrollers.", body))

    story.append(Paragraph("1.2 The System Architecture at a Glance", h2))
    story.append(Paragraph(
        "The system processes continuous 6-axis inertial data (accelerometer and gyroscope) through an 8-stage pipeline: "
        "Acquisition -> Butterworth 5 Hz Filtering -> Downsampling to 20 Hz -> 64-sample Windowing -> CNN-LSTM Inference -> "
        "Validation-Selected Thresholding (tau = 0.25) -> Dynamic-Range TFLite Microcontroller Execution -> Circular Ring-Buffer Streaming with Caregiver Alerts.", body))

    arch_img = REPO_ROOT / "docs/presentation/figures/review3_system_architecture.png"
    if arch_img.exists():
        story.append(Image(str(arch_img), width=6.0 * inch, height=2.6 * inch))
        story.append(Paragraph("Figure 1.1: Complete 8-Stage End-to-End Fall Detection and Edge Deployment Architecture", caption))

    story.append(Paragraph("1.3 Core Guardrail: Validated Baseline vs. Exploratory Extensions", h2))
    story.append(Paragraph(
        "A critical hallmark of academic rigor is distinguishing between what is <b>fully validated for deployment</b> versus what is an <b>exploratory research extension</b>:<br/>"
        "• <b>Validated Core (Deployed):</b> Binary CNN-LSTM detector operating at threshold tau = 0.25 (0.9950 recall, 0.9879 F2) quantized to Dynamic-Range TFLite (111 KB, 0.026 ms latency). "
        "This is the sole model powering our real-time streaming dashboard.<br/>"
        "• <b>Parallel Branch A (Severity):</b> Unsupervised K-Means (K=2) and Isolation Forest clustering that stratifies fall impacts into Mild and Severe pseudo-labels.<br/>"
        "• <b>Parallel Branch B (Personalization):</b> FiLM feature modulation exploring zero-backpropagation calibration adjustments (honest scientific result: p = 0.125).", body))

    story.append(Spacer(1, 0.1 * inch))
    story.append(PageBreak())

    # =========================================================================
    # PART 2: WHY WE USED EACH TECHNOLOGY & WHY IT MATTERS
    # =========================================================================
    story.append(Paragraph("PART 2: TECHNICAL DEEP DIVE — 'WHY WE USED IT' & 'WHY IT MATTERS'", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0B2545"), spaceAfter=10))

    tech_items = [
        ("1. SisFall Dataset (38 Subjects, 4,505 Trials)",
         "Why We Used It: SisFall is the global gold-standard benchmark for wearable fall detection. It contains synchronized recordings from 38 subjects (23 young adults aged 19-30 and 15 elderly adults aged 60-75) performing 19 distinct ADLs (walking, stumbling, jumping, sitting) and 15 simulated fall types (forward, backward, lateral, syncope/fainting).\n"
         "Why It Matters: Having both young and elderly subjects ensures realistic anthropometric variability. With 4,505 trials (84,123 windows), it allows robust subject-wise partitioning without synthetic oversampling."),
        
        ("2. Six Inertial Channels (Acc1 + Gyro) and Dropping Acc2",
         "Why We Used It: We retain the primary wide-range tri-axial accelerometer (ADXL345, ±16g) and tri-axial gyroscope (ITG3200, ±2000°/s). We deliberately discarded the second accelerometer (MMA7361, ±6g).\n"
         "Why It Matters: Acc2 has a narrow ±6g range that clips and saturates during high-impact falls (which often exceed 8g–10g). Keeping redundant channels wastes microcontroller I2C bandwidth, increases battery drain, and doubles input tensor size for zero accuracy gain."),
        
        ("3. 2nd-Order 5 Hz Butterworth Low-Pass Filter",
         "Why We Used It: Human intentional voluntary motion and fall shock waves have fundamental frequencies below 4 Hz to 5 Hz. Mechanical vibrations from sensor rattling and high-frequency electronic noise operate above 8 Hz.\n"
         "Why It Matters: A Butterworth filter provides a maximally flat frequency response in the passband with zero ripple, removing noise without distorting real impact peaks. Forward-backward filtering (filtfilt) ensures zero phase shift."),
        
        ("4. Downsampling from 200 Hz to 20 Hz",
         "Why We Used It: Nyquist theorem dictates that a 5 Hz filtered signal requires at least 10 Hz sampling. Sampling at 20 Hz provides a comfortable 4x oversampling factor.\n"
         "Why It Matters: Downsampling by a factor of 10 cuts raw data volume by 90%. This reduces memory consumption, speeds up inference by 10x, and enables continuous 24/7 battery life on microcontrollers."),
        
        ("5. Strict Subject-Wise Split (26 Train / 6 Val / 6 Test)",
         "Why We Used It: We partition data strictly by human participants: 26 subjects for training, 6 for validation, and 6 held-out test subjects. No participant's data ever crosses partition boundaries.\n"
         "Why It Matters: Random window shuffling leaks subject-specific gait, mass, and sensor attachment artifacts into the test set, creating artificially inflated 99.9% accuracies that collapse in real life. Subject-wise splitting guarantees true out-of-sample clinical generalization."),
        
        ("6. 64-Sample Window (3.2 s) with 16-Sample Stride (0.8 s, 75% Overlap)",
         "Why We Used It: At 20 Hz, 64 samples equals exactly 3.2 seconds. A fall comprises three distinct kinematic phases: pre-fall instability (0.5–1.0 s), impact shock (0.3–0.5 s), and post-fall resting immobility (1.5–2.0 s). Stride is 16 samples (0.8 s), giving 75% overlap.\n"
         "Why It Matters: 3.2 seconds captures the entire transition from motion to resting, allowing the model to differentiate a fall from jumping or sitting. The 0.8-second stride ensures that the impact peak is centered in at least one window and triggers alerts within under one second."),
        
        ("7. Impact-Centered Window Labeling via Peak SMV",
         "Why We Used It: In a 15-second fall trial, the person walks normally for 10 seconds, falls at t = 11 s, and rests. If we labeled all windows from that trial as 'Fall', the network would be trained to treat normal walking as a fall!\n"
         "Why It Matters: We compute the Signal Magnitude Vector (SMV = sqrt(acc_x^2 + acc_y^2 + acc_z^2)) and locate the maximum impact index. Only windows containing that impact sample receive Fall label 1. This provides clean, unambiguous supervision."),
        
        ("8. Hybrid CNN-LSTM Architecture (46,817 Parameters)",
         "Why We Used It: We combine two 1D Convolutional blocks with a 64-unit Long Short-Term Memory (LSTM) layer and Global Average Pooling.\n"
         "Why It Matters: CNNs excel at extracting localized spatial relationships across the 6 axes (identifying sudden acceleration spikes). LSTMs excel at sequential temporal reasoning (tracking: normal walking -> freefall dip -> sharp impact -> zero motion post-fall). Standalone CNNs miss the temporal sequence; standalone LSTMs struggle with raw high-frequency noise. Together, they achieve optimal synergy."),
        
        ("9. Validation Decision Threshold tau = 0.25 (Maximizing F2-Score)",
         "Why We Used It: Standard classifiers use a threshold of 0.50. However, in healthcare, the cost of a False Negative (a missed fall resulting in death) is vastly worse than a False Positive (a caregiver checking on an elderly person). We swept thresholds on validation data to maximize the F2-score, which weights recall twice as heavily as precision.\n"
         "Why It Matters: At tau = 0.25, test recall reached 0.9950 (F2 = 0.9879), reducing missed fall windows from 9 down to only 6 out of 1,200. Precision remained high at 0.9606."),
        
        ("10. Unsupervised Severity Clustering: K-Means (K=2) + Isolation Forest",
         "Why We Used It: SisFall lacks clinician-annotated trauma grades. We extracted 58 kinematic features (peak acceleration, rotational energy, jerk, kurtosis) and evaluated K=2 to K=5. K=2 achieved the highest Silhouette score (0.1774 vs 0.1585 for K=3). Isolation Forest isolates the top 2% anomalous falls (label -1 Uncertain).\n"
         "Why It Matters: Stratifies falls into Mild (lower energy slip) vs. Severe (high acceleration collision) to prioritize emergency medical triage while masking noisy outliers."),
        
        ("11. Dynamic-Range TensorFlow Lite Quantization & C Header Export",
         "Why We Used It: Post-training quantization converts 32-bit floating-point weights into 8-bit integers while computing with float activations at runtime. We exported the binary into a C byte array header (model_data.h, 113,704 bytes).\n"
         "Why It Matters: Cuts model size by 53% (from 235.6 KB down to 111.0 KB) and single-window CPU latency to 0.026 ms with 100% agreement with Keras. On an ESP32 microcontroller with 520 KB SRAM, it occupies only 21% of memory, leaving ample space for FreeRTOS and BLE/WiFi communication."),
        
        ("12. Circular Ring Buffer Streaming Engine (Zero Dynamic Memory)",
         "Why We Used It: Embedded devices crash from heap fragmentation if memory is allocated dynamically. We engineered a pre-allocated circular ring buffer of size (64 x 6) that continuously stores arriving samples at 20 Hz.\n"
         "Why It Matters: Every 16 samples (0.8 s), the buffer is evaluated with zero memory leaks. A 10-second alert cooldown prevents spamming caregivers while maintaining bitwise numerical parity (<= 1e-6 error) with batch inference.")
    ]

    for title, desc in tech_items:
        story.append(Paragraph(f"<b>{title}</b>", h3))
        story.append(Paragraph(desc.replace("\n", "<br/>"), body))
        story.append(Spacer(1, 0.04 * inch))

    story.append(PageBreak())

    # =========================================================================
    # PART 3: VISUALIZATIONS GALLERY & INTERPRETATION
    # =========================================================================
    story.append(Paragraph("PART 3: VISUALIZATION GALLERY & DIAGRAM INTERPRETATIONS", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0B2545"), spaceAfter=10))

    story.append(Paragraph("3.1 Preprocessing: Filtering and Impact Windowing", h2))
    p_wave = REPO_ROOT / "docs/reports/project_report/report_assets/sample_waveforms_fall_vs_adl.png"
    if p_wave.exists():
        story.append(Image(str(p_wave), width=6.0 * inch, height=2.4 * inch))
        story.append(Paragraph("Figure 3.1: Raw 6-Channel Motion Waveforms: High-G Fall Transient vs. Periodic ADL Walking", caption))
    story.append(Paragraph(
        "<b>Interpretation:</b> Notice how the fall signal (top) exhibits a quiet walking phase, a sudden drop during free-fall, "
        "and an enormous acceleration spike (> 8g) across axes, followed by flat immobility. In contrast, the ADL walking signal (bottom) "
        "displays steady, rhythmic periodic oscillations within ±1.5g. Our CNN-LSTM easily discriminates these biomechanical signatures.", body))

    story.append(Paragraph("3.2 Model Training Convergence and Decision Thresholding", h2))
    p_curves = REPO_ROOT / "docs/reports/project_report/report_assets/cnn_lstm_training_curves.png"
    if p_curves.exists():
        story.append(Image(str(p_curves), width=5.8 * inch, height=2.2 * inch))
        story.append(Paragraph("Figure 3.2: CNN-LSTM Training Loss and Accuracy Curves (Optimal Checkpoint at Epoch 4)", caption))
    story.append(Paragraph(
        "<b>Interpretation:</b> Training converges smoothly in 14 epochs. Validation loss reaches its minimum at Epoch 4 (0.016) "
        "and begins plateauing. The early stopping callback restores the best Epoch 4 weights, preventing overfitting.", body))

    story.append(Paragraph("3.3 Confusion Matrix & Precision-Recall Curve", h2))
    p_cm = REPO_ROOT / "docs/reports/project_report/report_assets/cnn_lstm_confusion_matrix.png"
    if p_cm.exists():
        story.append(Image(str(p_cm), width=4.5 * inch, height=2.8 * inch))
        story.append(Paragraph("Figure 3.3: Held-Out Test Confusion Matrix at tau = 0.50 (9 Missed Falls; Reduced to 6 at tau = 0.25)", caption))

    story.append(Paragraph("3.4 Fall Severity Clustering & Uncertainty Detection", h2))
    p_pca = REPO_ROOT / "docs/reports/project_report/report_assets/severity_pca_clusters.png"
    if p_pca.exists():
        story.append(Image(str(p_pca), width=5.2 * inch, height=2.6 * inch))
        story.append(Paragraph("Figure 3.4: K-Means (K=2) Fall Severity Clusters Projected onto 2D PCA Feature Space", caption))
    story.append(Paragraph(
        "<b>Interpretation:</b> The 58-dimensional feature space cleanly separates into Cluster 0 (Mild, blue) and Cluster 1 (Severe, red) "
        "along Principal Component 1 (25.4% variance). Severe falls exhibit statistically significant increases in peak acceleration and angular velocity.", body))

    story.append(Paragraph("3.5 Edge Quantization Benchmarks", h2))
    p_quant = REPO_ROOT / "docs/presentation/figures/review3_quantization_comparison.png"
    if p_quant.exists():
        story.append(Image(str(p_quant), width=5.8 * inch, height=2.4 * inch))
        story.append(Paragraph("Figure 3.5: Model Footprint (KB) and Single-Window CPU Latency (ms) Across TFLite Quantization Formats", caption))
    story.append(Paragraph(
        "<b>Interpretation:</b> Dynamic-Range quantization cuts size from 235.6 KB to 111.0 KB and latency to 0.026 ms while preserving "
        "100% prediction agreement with the unquantized model. Full INT8 is larger (136.9 KB) because storing lookup tables for LSTM gates "
        "offsets integer weight savings.", body))

    story.append(Paragraph("3.6 Live Replay Dashboard Screenshots", h2))
    p_d1 = ASSETS_DIR / "slide_page_11_Image77.png"
    p_d3 = ASSETS_DIR / "slide_page_11_Image79.png"
    if p_d1.exists() and p_d3.exists():
        story.append(Image(str(p_d1), width=5.6 * inch, height=2.2 * inch))
        story.append(Paragraph("Figure 3.6: Live Dashboard Timeline: Probability Spikes from 0.0 to 1.0 at Fall Impact (t = 5.55 s)", caption))
        story.append(Image(str(p_d3), width=5.6 * inch, height=2.0 * inch))
        story.append(Paragraph("Figure 3.7: Simulated Caregiver Alert Dispatch Log Recording Alert #1 with Severe Classification", caption))

    story.append(PageBreak())

    # =========================================================================
    # PART 4: SLIDE-BY-SLIDE PRESENTATION WALKTHROUGH & SCRIPTS
    # =========================================================================
    story.append(Paragraph("PART 4: SLIDE-BY-SLIDE PRESENTATION GUIDE & TALKING SCRIPTS", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0B2545"), spaceAfter=10))

    slides_guide = [
        ("Slide 1: Title Slide",
         "What is on the Slide: Project Title, Anna University Header, Team Members, and Supervisor Dr. S. Umamaheswari.\n"
         "What to Say (Verbatim Script): 'Respected committee members, our supervisor Dr. Umamaheswari, and faculty, good morning. We present Review 3 of our project: Wearable Fall Detection System for Elderly People Using Deep Learning. In this review, we demonstrate our high-recall CNN-LSTM detector, edge microcontroller quantization, real-time streaming engine, and scientific evaluations of severity clustering and FiLM personalization.'\n"
         "Why It Matters: Sets a formal, professional academic tone immediately."),

        ("Slide 2: Review 3 Scope and Status",
         "What is on the Slide: Three research pillars (Severity, FiLM, Edge Streaming) and the critical distinction box.\n"
         "What to Say: 'Review 3 encompasses three engineering thrusts around our core detector. First, fall severity pseudo-labeling; second, calibration-conditioned personalization via FiLM; and third, edge quantization and streaming replay. Crucially, we maintain complete scientific honesty: our core binary CNN-LSTM is the sole validated model powering deployment, while severity and FiLM remain exploratory research branches.'\n"
         "Why It Matters: Prevents the committee from thinking you built a messy all-in-one network; shows mature engineering judgment."),

        ("Slide 3 & Slide 21: Review 3 System Map & Pipeline",
         "What is on the Slide: Block diagram from raw 200 Hz SisFall trials through preprocessing, subject split, CNN-LSTM, thresholding, to TFLite streaming.\n"
         "What to Say: 'Here is our end-to-end architecture. Raw 9-channel recordings are filtered with a 5 Hz Butterworth low-pass filter and downsampled to 20 Hz across six primary axes. We enforce a zero-leakage 26/6/6 subject split. 64-sample sliding windows feed the CNN-LSTM. Operating threshold tau = 0.25 was selected on validation data to maximize recall. The binary model is converted to TFLite for real-time streaming.'\n"
         "Why It Matters: Demonstrates complete pipeline mastery from raw physics to deployed edge inference."),

        ("Slide 4: Severity Training Pipeline",
         "What is on the Slide: Fall windows -> 58 Feature Extraction -> K-Means (K=2) -> Isolation Forest Mask -> Multi-Task Loss.\n"
         "What to Say: 'Because SisFall lacks medical trauma labels, we developed an unsupervised pseudo-labeling pipeline. For every true fall window, we extract 58 statistical and kinematic features. We train K-Means to cluster falls into Mild and Severe categories. An Isolation Forest identifies the top 2% anomalous falls, tagging them as Uncertain (-1) so irregular falls do not destabilize model training.'\n"
         "Why It Matters: Solves the lack of ground-truth clinical severity labels without hallucinating medical outcomes."),

        ("Slide 5: Why K = 2 Was Selected",
         "What is on the Slide: 4 clustering metric plots: Elbow Inertia, Silhouette vs. k, Davies-Bouldin, Calinski-Harabasz.\n"
         "What to Say: 'We systematically evaluated cluster counts from k=2 to k=5. As shown in the Silhouette plot, k=2 achieves a score of 0.177, whereas k=3 drops to 0.158 and produces an unstable, ill-defined middle cluster. A 2-cluster partition cleanly separates low-impact falls (Mild) from high-impact falls (Severe). We refer to them as sensor-derived pseudo-labels rather than medical injury grades.'\n"
         "Why It Matters: Provides mathematical proof (Silhouette & Davies-Bouldin) for why K=2 was chosen over K=3."),

        ("Slide 6: Severity Clusters and Uncertainty",
         "What is on the Slide: 2D PCA cluster scatter plot, Isolation Forest anomaly distribution, and peak intensity boxplots.\n"
         "What to Say: 'On the top left, the PCA projection demonstrates clear separation between Cluster 0 (Mild, 468 test windows) and Cluster 1 (Severe, 732 test windows). The boxplots confirm that Severe falls exhibit significantly higher peak acceleration and angular velocity. The Isolation Forest histogram shows the 2% anomaly contamination threshold, isolating irregular falls so they do not destabilize model training.'\n"
         "Why It Matters: Visually proves that the K-Means clusters represent genuine physical differences in impact kinetic energy."),

        ("Slide 19 & Slide 7: CNN-LSTM Architecture & Severity Evaluation",
         "What is on the Slide: Layer-by-layer architectural flow and severity confusion matrix (Macro-F1 = 0.8660).\n"
         "What to Say: 'Our neural trunk shares two Conv1D blocks with batch normalization and max-pooling, followed by a 64-unit LSTM and global pooling. The shared representation feeds a sigmoid binary head and a 2-class softmax severity head. As seen in the confusion matrix on Slide 7, the severity head achieves a Macro-F1 of 0.8660 across 1,176 confident fall windows.'\n"
         "Why It Matters: Demonstrates strong multi-task classification while properly evaluating on masked confident fall windows."),

        ("Slide 8: Severity Head Trade-off & Gating Decision",
         "What is on the Slide: Comparison table between Binary Baseline (Recall 0.9950) vs. Multi-Task (Recall 0.9692).\n"
         "What to Say: 'Here is our core engineering decision. Although the multi-task model classifies severity well, its binary fall recall fell from 99.50% to 96.92% on held-out test data. Under our pre-established engineering gate (Decision D1/D2), any degradation in binary safety recall disqualifies a model from primary deployment. Therefore, we retain the binary CNN-LSTM as our deployed model.'\n"
         "Why It Matters: The most impressive moment in your viva: you demonstrate that safety trumps novelty."),

        ("Slide 9 & Slide 20: FiLM Personalization Method & Architecture",
         "What is on the Slide: UserEncoder predicting gamma and beta vectors to modulate CNN feature sequence H before the frozen LSTM.\n"
         "What to Say: 'Wearable sensors vary by user gait, weight, and sensor placement. Rather than retraining a full neural network for each person, we investigated Feature-wise Linear Modulation (FiLM). A lightweight UserEncoder (24,960 parameters) pools calibration ADL windows from a specific user and predicts affine scale gamma and shift beta vectors. These modulate the CNN feature maps before passing into the frozen LSTM.'\n"
         "Why It Matters: Demonstrates cutting-edge parameter-efficient fine-tuning (PEFT) without backpropagation on edge devices."),

        ("Slide 10 & Slide 11: FiLM Experiment Design & Statistical Results",
         "What is on the Slide: Wilcoxon paired test p-value chart (p = 0.125) and F2 across calibration windows.\n"
         "What to Say: 'We tested calibration sizes of 0, 1, 5, 10, and 30 ADL windows. While false alarms slightly decreased for some simulated profiles, our paired Wilcoxon signed-rank test on held-out subjects yielded p = 0.125. Because p > 0.05, the improvement is not statistically significant. Rather than claiming an unverified benefit, we formally document this as an exploratory proof-of-concept that requires larger clinical trials.'\n"
         "Why It Matters: Proves statistical rigor. Reviewers respect students who acknowledge p > 0.05 rather than faking 100% success."),

        ("Slide 12 & Slide 13: TFLite Quantization Benchmarks",
         "What is on the Slide: Bar charts comparing Model Size (KB) and Latency (ms) across Float32, Dynamic-Range, Float16, and Full INT8.\n"
         "What to Say: 'To deploy onto microcontrollers, we converted our model into 4 TFLite formats. Unquantized Float32 is 235.6 KB with 0.063 ms latency. Dynamic-Range quantization reduced the model footprint to 111.0 KB—a 53% compression—while cutting inference latency to 0.026 ms per window. Crucially, as shown on Slide 13, Dynamic-Range quantization maintained 100% numerical agreement with the original Keras model, preserving our 0.9950 recall exactly.'\n"
         "Why It Matters: Concrete evidence that the model is ready for real embedded hardware."),

        ("Slide 14: Streaming Inference & Ring Buffer Architecture",
         "What is on the Slide: Circular Ring Buffer (64 x 6), Evaluation Stride (16 samples), TFLite edge model, Alert Cooldown.\n"
         "What to Say: 'In real-world deployment, sensor samples arrive continuously at 20 Hz, not as pre-cut matrices. We implemented an in-memory circular ring buffer that continuously holds the latest 64 samples. Every 16 samples (0.8 seconds), a window is extracted, normalized using the stored training scaler, and evaluated by the TFLite engine. If the probability exceeds 0.25, an alert is triggered, and a 10-second cooldown prevents repeated spamming.'\n"
         "Why It Matters: Demonstrates software engineering maturity and production embedded design."),

        ("Slide 15 & Slide 16: Live Streaming Dashboard Demonstration",
         "What is on the Slide: Streamlit UI showing probability timeline spiking to 1.0 at t=5.55s and caregiver alert log.\n"
         "What to Say: 'These slides show our software-in-the-loop streaming dashboard. On Slide 15, we replay a held-out SisFall recording. For normal motion, probability stays near 0. At t = 5.55 s, the impact occurs; probability spikes to 1.0, crossing the threshold line. Slide 16 shows the caregiver dispatch log recording the alert timestamp, probability, and predicted severity.'\n"
         "Why It Matters: Connects mathematics to a tangible clinical user experience."),

        ("Slide 17: FiLM User Personalization Frontend",
         "What is on the Slide: Interactive UI showing subject selection, noisy profile simulation, and comparison timeline.\n"
         "What to Say: 'This frontend demonstrates our FiLM proof-of-concept. The user can select a held-out subject, inject simulated sensor noise or rotated profiles, provide 5 calibration ADL windows, and observe how the modulation shifts the false-positive rate on remaining windows.'\n"
         "Why It Matters: Proves that the FiLM adapter is fully functional in software."),

        ("Slide 18: Conclusions, Limitations, and Next Steps",
         "What is on the Slide: Summary table of Validated Core vs. Exploratory Extensions, Limitations, and Next Steps.\n"
         "What to Say: 'In conclusion, our project delivers a complete, reproducible pipeline from raw inertial sensors to a quantized edge detector achieving 0.9950 recall on held-out subjects. We have rigorously bounded our findings: SisFall consists of simulated falls by healthy volunteers, and our frontends operate in software replay. In future work, we plan to validate on real-world elderly motion recordings and deploy the C-array header onto physical ESP32 hardware. Thank you, and we welcome your questions.'\n"
         "Why It Matters: Professional, humble, and scientifically airtight conclusion.")
    ]

    for title, desc in slides_guide:
        story.append(Paragraph(f"<b>{title}</b>", h2))
        lines = desc.split("\n")
        for line in lines:
            if line.startswith("What is on the Slide:"):
                story.append(Paragraph(f"<b>{line[:21]}</b>{line[21:]}", body))
            elif line.startswith("What to Say:"):
                story.append(Paragraph(f"<b>{line[:12]}</b> <i>\"{line[13:]}\"</i>", say_script))
            elif line.startswith("Why It Matters:"):
                story.append(Paragraph(f"<b>{line[:15]}</b>{line[15:]}", body_bold))
            else:
                story.append(Paragraph(line, body))
        story.append(Spacer(1, 0.05 * inch))

    story.append(PageBreak())

    # =========================================================================
    # PART 5: MASTER VIVA DEFENSE QUESTION BANK (TOP 20 QUESTIONS)
    # =========================================================================
    story.append(Paragraph("PART 5: MASTER VIVA DEFENSE QUESTION BANK (TOP 20 QUESTIONS)", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0B2545"), spaceAfter=10))
    story.append(Paragraph("Below are the top 20 questions external examiners and faculty typically ask, paired with authoritative answers grounded in your data.", body))

    viva_qna = [
        ("Q1: Why did you choose a hybrid CNN-LSTM instead of a pure CNN or pure LSTM?",
         "Answer: A 1D CNN acts as an automatic feature extractor across the 6 channels, identifying sharp acceleration shocks, but lacks recurrent memory over multi-second temporal windows. A pure LSTM can model sequences but struggles with raw high-frequency noise and has higher parameter complexity. By feeding pooled CNN feature maps into an LSTM, the CNN extracts spatial motion primitives and the LSTM models the progression from walking to falling to immobility, achieving 0.9950 recall."),

        ("Q2: Why is your operating threshold 0.25 instead of the default 0.50?",
         "Answer: Default tau = 0.50 assumes symmetric cost between false alarms and missed falls. In elderly safety, a False Negative (missed fall) can lead to death from prolonged immobility, while a False Positive is merely an inconvenience. We swept candidate thresholds on validation data to maximize the recall-weighted F2-score (F2 = 5*P*R / (4*P + R)). tau = 0.25 was selected, reducing missed fall windows from 9 down to 6 on test data while maintaining 0.9606 precision."),

        ("Q3: How did you ensure there is zero data leakage between training and testing?",
         "Answer: We enforced two strict boundaries: First, a subject-wise split where 26 subjects are Train, 6 are Validation, and 6 are Test. No person's windows appear in more than one partition. Second, our StandardScaler was fitted exclusively on the 26 training subjects and reused without re-fitting on validation, test, and streaming inference."),

        ("Q4: Why did you drop the second accelerometer (acc2) from the SisFall dataset?",
         "Answer: The SisFall hardware had two accelerometers: ADXL345 (range ±16g) and MMA7361 (range ±6g). Human falls generate impact forces between 6g and 12g, causing the ±6g MMA7361 to saturate and clip at peak impact. The wide-range ADXL345 captures the complete dynamic range cleanly. Dropping acc2 reduces input dimensions from 9 to 6, saving 33% computation and microcontroller memory."),

        ("Q5: Why did you select a 5 Hz Butterworth low-pass filter?",
         "Answer: Clinical biomechanics literature confirms that voluntary human intentional movements operate below 3 Hz to 4 Hz, with fall impact transients rarely exceeding 5 Hz. High frequencies above 5 Hz represent mechanical resonance, clothing flutter, and electrical sensor noise. A 2nd-order Butterworth filter provides a maximally flat passband with zero ripple, removing noise without altering peak impact amplitudes."),

        ("Q6: Why did you downsample the signal from 200 Hz to 20 Hz?",
         "Answer: According to the Nyquist-Shannon theorem, a 5 Hz low-pass filtered signal requires a minimum sampling rate of 10 Hz. Sampling at 20 Hz provides a 4x oversampling safety margin. Decimating from 200 Hz to 20 Hz reduces memory buffer requirements by 90% (from 640 floats to 64 floats per window) and reduces microcontroller inference operations by a factor of 10."),

        ("Q7: What is impact-centered labeling and why is it necessary?",
         "Answer: A 15-second fall recording in SisFall consists of normal walking, a 0.5s loss of balance, a 0.3s collision, and 3s of resting. If all windows from that trial were labeled 'Fall', normal walking would be penalized as a fall. We compute the peak Signal Magnitude Vector (SMV) and assign Fall label 1 only to windows containing that peak. Pre-fall walking and post-fall resting remain label 0, eliminating label ambiguity."),

        ("Q8: Why did you use K-Means with K=2 rather than K=3 for severity?",
         "Answer: We evaluated K=2 through K=5 across 58 kinematic features. K=2 achieved a Silhouette score of 0.1774, whereas K=3 dropped to 0.1585 and created an unstable, overlapping middle cluster with poor Davies-Bouldin separation. K=2 provides a clean, defensible partition between low-energy slips (Mild) and violent high-acceleration collisions (Severe)."),

        ("Q9: What is the purpose of the Isolation Forest in your severity pipeline?",
         "Answer: Real-world falls contain atypical anomalies—such as rolling off a bed, sliding down a wall, or sensor dislodgement—that do not fit clean impact dynamics. The Isolation Forest isolates the top 2% anomalous fall feature vectors and assigns them label -1 (Uncertain). These windows are retained as binary falls but masked out of the severity loss, preventing noisy outliers from distorting cluster centroids."),

        ("Q10: Why was the multi-task CNN-LSTM rejected for primary deployment?",
         "Answer: In our gating benchmark (Decision D1/D2), the multi-task model suffered negative transfer: while it achieved 0.8660 severity Macro-F1, its binary fall recall dropped from 0.9950 to 0.9692 (missing 37 falls instead of 6). In safety-critical healthcare, binary recall is non-negotiable. Therefore, the single-task binary detector is deployed, and severity is retained as a research extension."),

        ("Q11: What is FiLM personalization and how does it work without retraining?",
         "Answer: Feature-wise Linear Modulation applies an affine transformation (H_pers = gamma * H + beta) to the intermediate CNN feature maps before the frozen LSTM. A lightweight UserEncoder processes k calibration ADL windows from a user and predicts gamma and beta in a single forward pass. Because the base CNN and LSTM remain completely frozen, adaptation happens instantly without backpropagation on the edge device."),

        ("Q12: Your FiLM evaluation showed p = 0.125. Why did it not achieve statistical significance?",
         "Answer: In the held-out test partition of SisFall, only 4 subjects performed actual falls (the other 2 performed ADLs only). For a sample size of n = 4, the minimum mathematically achievable two-sided Wilcoxon p-value is (0.5)^3 = 0.125. Because p > 0.05, we transparently document the result as exploratory-inconclusive rather than over-claiming unverified gains."),

        ("Q13: Why did you select Dynamic-Range quantization over Full INT8?",
         "Answer: Dynamic-Range quantization compresses the model from 235.6 KB to 111.0 KB (53% compression) and executes in 0.026 ms on CPU with 100% agreement with Keras. Full INT8 was actually larger (136.9 KB) and slower (0.033 ms) because quantizing recurrent LSTM sigmoid and tanh gate tables requires additional dequantization lookups on ARM/x86 architectures."),

        ("Q14: How does your model run on an ESP32 microcontroller?",
         "Answer: We converted the Dynamic-Range TFLite flatbuffer into a static C byte array (const unsigned char g_model[] in model_data.h, 113,704 bytes). This array resides in read-only flash memory. On an ESP32 with 520 KB of SRAM, the runtime tensor arena occupies approximately 110 KB (21% of SRAM), leaving ample memory for FreeRTOS and communication."),

        ("Q15: How does the circular ring buffer streaming engine prevent memory leaks?",
         "Answer: The StreamingFallDetector pre-allocates a fixed (64 x 6) float array in memory. When a new 20 Hz sample arrives, it overwrites the oldest sample via modulo indexing. Every 16 samples (0.8s), a sliding window view is extracted and normalized. Zero heap memory is allocated or freed during streaming, completely eliminating heap fragmentation."),

        ("Q16: Why is 'Impact-to-Alert Latency' shown as N/A in the dashboard screenshot?",
         "Answer: In our software replay mode, SisFall recordings are read from CSV files without synchronized hardware real-time clock timestamps embedded in each row. To preserve scientific honesty, the dashboard displays N/A for live packet latency rather than showing an unverified fabricated number. Model inference latency itself was rigorously benchmarked at 0.026 ms."),

        ("Q17: How does your system prevent repeated alert spamming during a fall?",
         "Answer: The streaming engine enforces an alert cooldown timer of 10.0 seconds. Once a fall is detected (probability > 0.25), subsequent windows are evaluated for logging, but additional caregiver alarms are suppressed until the 10-second window elapses."),

        ("Q18: What is the difference between your pseudo-severity labels and clinical injury grades?",
         "Answer: SisFall recorded young and elderly volunteers falling onto foam crash pads without real fractures or concussions. Our Mild and Severe labels reflect physical impact energy (peak acceleration and angular velocity), not medical trauma. We explicitly document them as sensor-derived pseudo-labels."),

        ("Q19: What are the primary threats to validity of this research?",
         "Answer: First, SisFall consists of simulated falls by healthy volunteers, which may differ from frail geriatric falls involving slow stumbling. Second, the held-out test split has 6 subjects, of whom 4 had falls. Third, the current prototype operates via software replay rather than live on-body firmware."),

        ("Q20: What are your immediate next steps for future work?",
         "Answer: First, flashing model_data.h directly onto physical ESP32 hardware with an I2C MPU-6050 IMU. Second, conducting clinical trials with frail elderly residents in assisted living facilities. Third, fusing a barometric pressure sensor to detect rapid vertical altitude drops (1.0m to 1.5m), further reducing false alarms.")
    ]

    for q, a in viva_qna:
        story.append(Paragraph(f"<b>{q}</b>", h3))
        story.append(Paragraph(a, body))
        story.append(Spacer(1, 0.05 * inch))

    doc.build(story, canvasmaker=StudyGuideCanvas)
    print(f"Successfully compiled Study & Defense Guide PDF at: {PDF_PATH}")
    return PDF_PATH

if __name__ == "__main__":
    build_study_guide_pdf()
