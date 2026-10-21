# Model Comparison Summary (Task P3)

This report consolidates the test-set performance across models developed in the project, evaluated under locked decision rules:
- **D1**: Canonical severity scheme is K=2 (`k2_20260914T055946Z`: Mild=0, Severe=1, Uncertain=-1).
- **D2**: Every model's operating threshold is selected strictly on the **validation split** (maximum $F_2$, tie-breaking lower false alarms/hr, then precision), frozen, and evaluated on the test set. 0.50 serves as a reference point.
- **D4**: FiLM personalization uses a frozen CNN-LSTM trunk, a small user encoder trained once, and inference-only adaptation.

---

## 1. Primary Model Comparison (Validation-Selected Threshold)

| Model | Val-Selected Threshold | Accuracy | Precision | Recall | F1 | F2 | FA/hr | AUROC | AUPRC | Parameters | Severity Macro-F1 | Status / Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CNN (binary)** | 0.20 | 0.9965 | 0.9683 | 0.9925 | 0.9802 | 0.9876 | 14.12 | *UNVERIFIED* | *UNVERIFIED* | 33,729 | N/A | Saved baseline run (`results/cnn_baseline/run_metadata.json`) |
| **CNN-LSTM (binary)** | 0.25 | 0.9960 | 0.9606 | 0.9950 | 0.9775 | 0.9879 | 17.74 | 0.9998 | 0.9987 | 46,817 | N/A | Saved baseline run (`results/cnn_lstm_baseline/run_metadata.json`) |
| **CNN-LSTM + severity head** | 0.20 | 0.0990 | 0.0883 | 0.9900 | 0.1621 | 0.3254 | 4441.70 | 0.5036 | 0.4631 | 46,883 | 0.3794 | Trained run (`results/severity_multitask/run_v1/run_metadata.json`) |
| **FiLM (k=1, exploratory)** | 0.60 (mean user) | 0.9932 | 0.9720 | 0.9554 | 0.9738 | 0.9840 | 16.64 | 0.9998 | 0.9985 | 46,817 (+128) | N/A | Mean across 3 seeds (`results/film_personalization/full_run_v1`) |
| **FiLM (k=30, exploratory)** | 0.75 (mean user) | 0.9943 | 0.9693 | 0.9692 | 0.9787 | 0.9880 | 14.03 | 0.9998 | 0.9986 | 46,817 (+128) | N/A | Mean across 3 seeds (`results/film_personalization/full_run_v1`) |

> **Footnote**: Evaluated on 6 held-out test subjects (4 young adults with falls: SA02, SA03, SA09, SA14; 2 elderly subjects with ADL-only: SE11, SE14). Fall-sensitive metrics ($F_1$, $F_2$, Recall) are computed strictly across fall-containing subjects. With $n=4$ fall subjects, statistical significance cannot be claimed.

---

## 2. Reference Operating Point (Threshold = 0.50)

For cross-model consistency, the standard 0.50 threshold is provided below as a reference operating point:

| Model | Threshold | Accuracy | Precision | Recall | F1 | F2 | FA/hr | Missed Falls (out of 1,200) | False Alarms (out of 12,427 ADL) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CNN (binary)** | 0.50 | 0.9967 | 0.9761 | 0.9867 | 0.9814 | 0.9845 | 10.50 | 16 | 29 |
| **CNN-LSTM (binary)** | 0.50 | 0.9973 | 0.9770 | 0.9925 | 0.9847 | 0.9894 | 10.14 | 9 | 28 |
| **CNN-LSTM + severity head** | 0.50 | 0.9119 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 1,200 | 0 |
| **FiLM (k=1, exploratory)** | 0.25 (canonical ref) | 0.9872 | 0.9108 | 0.9821 | 0.9514 | 0.9768 | 44.65 | 21 | 124 |
| **FiLM (k=30, exploratory)** | 0.25 (canonical ref) | 0.9894 | 0.9092 | 0.9889 | 0.9535 | 0.9805 | 36.90 | 13 | 102 |

---

## 3. Analysis & Key Observations

1. **CNN vs CNN-LSTM Baseline**:
   - At validation-selected thresholds (0.20 for CNN, 0.25 for CNN-LSTM), the CNN-LSTM reduces missed falls from 9 to 6 (test recall improves from $0.9925 \to 0.9950$, with $F_2$ rising from $0.9876 \to 0.9879$).
   - The false alarm rate rises from 14.12 to 17.74 FA/hr (+10 false positive windows across 2.76 hours of negative activity), which diagnostics confirm are single-window transient blips.
2. **CNN-LSTM Severity Head (Task P2)**:
   - Evaluated on real dataset training (`results/severity_multitask/run_v1`).
   - The multi-task model achieved severity macro-F1 of **0.3794** on confident fall windows (matching the majority class baseline).
   - However, binary test recall regressed from 0.9950 to 0.9900, with a steep surge in false alarms (4441.70 FA/hr). Therefore, per Task P5 deployment rules, the binary CNN-LSTM remains the authoritative deployed model.
3. **FiLM Personalization Status**:
   - Closed as exploratory/inconclusive. When per-user validation calibration is applied, FiLM achieves AUROC ~0.9998, but does not provide statistically significant improvements over Global CNN-LSTM on the 4 held-out fall subjects.
   - At the fixed 0.25 reference threshold, FiLM exhibits elevated false alarms (44.65 FA/hr at $k=1$, 36.90 FA/hr at $k=30$).

---

## 4. Artifact Provenance

- **CNN Baseline**: `results/cnn_baseline/run_metadata.json`
- **CNN-LSTM Baseline**: `results/cnn_lstm_baseline/run_metadata.json`
- **CNN-LSTM Severity Head**: `results/severity_multitask/run_v1/run_metadata.json`
- **FiLM Personalization**: `results/film_personalization/full_run_v1/analysis/test_metric_summary.csv`
