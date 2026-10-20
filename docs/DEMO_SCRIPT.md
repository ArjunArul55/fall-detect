# AI Wearable Fall Detection - Demo Script & Walkthrough

This document outlines a **5-minute live demonstration script** showcasing real-time, low-latency streaming fall detection using the deployed quantized TFLite model (`results/quantization/run_v1/deployed_model.tflite`).

---

## 1. System & Architecture Overview (1 Minute)

- **Input Modality**: 6-channel IMU ($a_x, a_y, a_z$ in $g$, $g_x, g_y, g_z$ in $\text{rad/s}$) sampled at 20 Hz.
- **Windowing & Latency**:
  - Window size: 64 samples ($3.2\text{ s}$).
  - Stride: 16 samples ($0.8\text{ s}$ update interval).
  - Algorithmic latency: $0.8\text{ s}$ per sliding step after a $3.2\text{ s}$ buffer warmup.
  - Deployed model size: $111.04\text{ KB}$ (Dynamic-range quantized CNN-LSTM).
  - CPU inference latency: $\approx 0.025\text{ ms}$ per window.
- **Alert Policy**:
  - Validation-selected threshold: $\tau = 0.10$ (selected to optimize validation $F_2$, balancing high recall with low false alarm rate).
  - Consecutive positive requirement: $N=1$ (default; configurable to 2 or 3 for enhanced noise rejection).
  - Alert cooldown: $10.0\text{ s}$ refractory period preventing duplicate alerts during a single fall event.
- **Severity Classification**:
  - Deployed model: Binary CNN-LSTM trunk deployed per gating protocol (preserved 100% recall parity with float baseline).
  - Multitask severity head context: Evaluated in Task P2/P3 with K=2 canonical labels (Mild vs. Severe).

---

## 2. Launching the Demo

### Option A: Interactive Streamlit Dashboard (Recommended)
```powershell
.\.venv\Scripts\streamlit run app/demo.py
```
Open your browser to `http://localhost:8501`.

### Option B: Command-Line Interface Replay
```powershell
.\.venv\Scripts\python.exe scripts/run_demo_cli.py --csv-path data/demo_recordings/SA02_F01_MildFall.csv --speed 5.0
```

---

## 3. Five-Minute Walkthrough: 6 Curated Test Recordings

All demo recordings are extracted from the **unseen SisFall test set** (Subject `SA02`, test split). Each recording is a continuous 15.0 s sequence (300 samples at 20 Hz, yielding 15 sliding windows).

| # | Recording File | Subject | Activity Code | Activity Description | Category | Expected Outcome | Actual Observed Max Prob | Actual Alert Status |
|---|---|---|---|---|---|---|---|---|
| **1** | `SA02_D01_Walking.csv` | SA02 | `D01` | Walking | ADL (Baseline) | Monitoring, 0 alerts | `0.0001` | 0 Alerts (Clean pass) |
| **2** | `SA02_D07_Sitting.csv` | SA02 | `D07` | Standing to sitting | ADL (Transition) | Monitoring, 0 alerts | `0.0002` | 0 Alerts (Clean pass) |
| **3** | `SA02_D12_Stumble.csv` | SA02 | `D12` | Stumbling / tripping | Hard-Negative ADL | Monitoring, 0 alerts | `0.0001` | 0 Alerts (Clean pass) |
| **4** | `SA02_D15_Jump.csv` | SA02 | `D15` | Vertical jumping | Hard-Negative ADL | Elevated prob, no alert | `0.0435` | 0 Alerts (Below $\tau=0.10$) |
| **5** | `SA02_F01_MildFall.csv` | SA02 | `F01` | Fall forward on knees | Fall (Mild) | Alert at impact ($t \approx 7.2\text{s}$) | `0.9999` | 1 Alert fired ($t = 7.15\text{s}$) |
| **6** | `SA02_F03_SevereFall.csv` | SA02 | `F03` | Fall backward on buttocks | Fall (Severe) | Alert at impact ($t \approx 7.0\text{s}$) | `0.9292` | 1 Alert fired ($t = 7.15\text{s}$) |

---

## 4. Step-by-Step Live Demo Narrative

### Minute 1: Normal Daily Activities (ADL Rejection)
1. In the sidebar dropdown, select **`SA02_D01_Walking (ADL)`**.
2. Set replay speed to **5x** and click **Start Streaming Replay**.
3. **Presenter Commentary**:
   > *"Notice the continuous 3-axis accelerometer and gyroscope waveforms. During standard walking, acceleration oscillates rhythmically within $[-1.5g, +0.5g]$. The model evaluates every 16 samples ($0.8\text{ s}$). Fall probability stays below $0.0002$. The status remains MONITORING with zero false alarms."*
4. Select **`SA02_D07_Sitting (ADL)`** and click replay.
5. **Presenter Commentary**:
   > *"Transitions like sitting down involve vertical acceleration dips. Here the maximum fall probability reaches only $0.0002$, well below the operating threshold of $\tau = 0.10$."*

### Minute 2: Challenging Dynamic Hard Negatives (Jump & Stumble)
1. Select **`SA02_D12_Stumble (Hard-Negative ADL)`** and run replay.
2. **Presenter Commentary**:
   > *"Tripping or stumbling involves sharp inertial spikes. A naive threshold on peak acceleration might trigger a false alarm, but the temporal LSTM dynamics correctly recognize recovery. Probability stays negligible at $0.0001$."*
3. Select **`SA02_D15_Jump (Hard-Negative ADL)`** and run replay.
4. **Presenter Commentary**:
   > *"Vertical jumping causes substantial impact shock upon landing ($> 2.5g$). Notice the probability timeline spikes up to $0.0435$ around the landing phase. However, because our validation-selected threshold is $\tau = 0.10$, the system successfully suppresses the false alarm."*

### Minute 3: Real Fall Event (Mild Fall Detection)
1. Select **`SA02_F01_MildFall (Fall)`** and set speed to **1x** or **5x**.
2. Click **Start Streaming Replay**.
3. **Presenter Commentary**:
   > *"In this trial, the subject walks forward and falls forward onto their knees at $t \approx 7.2\text{ s}$. As the 64-sample window slides over the impact phase, fall probability shoots from $<0.0001$ to $0.9999$. At $t = 7.15\text{ s}$, the status badge flips to 'FALL DETECTED' with an audio-visual alert. The Simulated Caregiver Sink immediately logs the dispatch event: 'Fall detected, severity Mild, probability 0.9999'."*

### Minute 4: High-Impact Fall Event (Severe Fall Detection)
1. Select **`SA02_F03_SevereFall (Fall)`** and run replay.
2. **Presenter Commentary**:
   > *"This trial simulates a severe backward fall onto the buttocks with immediate cessation of movement. At $t = 7.15\text{ s}$, probability reaches $0.9292$, triggering an immediate alert. Despite multiple subsequent overlapping windows maintaining high probability, our $10\text{ s}$ cooldown policy suppresses duplicate spam alerts."*

### Minute 5: Interactive Stress-Testing & Parameter Tuning
1. Demonstrate the effect of **$N$-consecutive positives**:
   - Change $N$-consecutive from 1 to 2.
   - Re-run `SA02_F01_MildFall.csv`.
   - Explain how requiring 2 consecutive windows ($1.6\text{ s}$) filters isolated transient shock artifacts while still notifying caregivers within $1.6\text{ s}$ of impact.
2. Demonstrate **Custom CSV Upload**:
   - Point out the CSV file uploader in the sidebar, which accepts arbitrary 6-channel IMU recordings sampled at 20 Hz.

---

## 5. Honest Evaluation & Known Failure Modes

In accordance with scientific integrity guidelines:
1. **Lab vs. Real-World Domain Shift**: The SisFall dataset consists of simulated falls performed onto safety mattresses by young adult subjects and controlled ADLs. Real-world elderly falls often exhibit prolonged pre-fall sway and post-fall immobility that are not fully captured in controlled lab settings.
2. **Hard-Negative Jump Spikes**: Vigorous jumping (`D15`) reached a peak probability of $0.0435$. In high-athleticism scenarios or lower threshold configurations ($\tau < 0.05$), jumping can register as a transient positive. Setting $N=2$ consecutive positives provides robust mitigation.
3. **Test Subject Constraints**: The SisFall test split comprises 6 subjects, 4 of whom performed falls and 2 who performed ADLs only. While empirical recall on the test set is high ($100\%$ on tested fall trials with D2 threshold), clinical deployment requires wider cohort validation.
