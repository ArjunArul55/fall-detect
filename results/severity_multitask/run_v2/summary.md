# Multi-Task Severity Head Run Summary (Task P2)

- **Run Directory**: `results/severity_multitask/run_v2`
- **Generated (UTC)**: 2026-10-21T12:41:54.871833+00:00
- **Best Lambda**: 1.0 (Epoch 1)
- **Validation-Selected Binary Threshold**: 0.10 (0.50 reference)
- **Training Time**: 103.39 s

## 1. Binary Fall Detection: Baseline CNN-LSTM vs Multi-Task

| Model | Threshold | Accuracy | Precision | Recall | F1 | F2 | FA/hr | AUROC | AUPRC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CNN-LSTM Baseline | 0.25 (val-sel) | 0.9960 | 0.9606 | 0.9950 | 0.9775 | 0.9879 | 17.74 | 0.9998 | 0.9984 |
| **Multi-Task CNN-LSTM** | **0.10 (val-sel)** | **0.9935** | **0.9580** | **0.9692** | **0.9635** | **0.9669** | **18.47** | **0.9994** | **0.9943** |
| Multi-Task CNN-LSTM | 0.50 (ref) | 0.9870 | 0.9904 | 0.8608 | 0.9211 | 0.8840 | 3.62 | 0.9994 | 0.9943 |

- **Recall Regression Check**: Baseline Recall = 0.9950, Multi-Task Recall = 0.9692. Regressed: `True`.

## 2. Severity Classification on Confident Fall Windows

| Level | Evaluated Samples | Mild P/R/F1 | Severe P/R/F1 | Macro-F1 | Confusion Matrix (Mild, Severe) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Window | 1176 | 0.8468 / 0.8228 / 0.8346 | 0.8893 / 0.9054 / 0.8973 | **0.8660** | [[376, 81], [68, 651]] |
| Recording | 297 | - | - | **0.9020** | [[95, 19], [8, 175]] |
| Subject | 6 | - | - | - | [[0, 0], [0, 4]] |

### Per-Subject Severity Window Breakdown

| Subject ID | Total Windows | True Mild | True Severe | Pred Mild | Pred Severe | Macro-F1 | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SA02 | 285 | 109 | 176 | 101 | 184 | 0.8718 | Young adult (with falls) |
| SA03 | 297 | 128 | 169 | 133 | 164 | 0.9146 | Young adult (with falls) |
| SA09 | 297 | 124 | 173 | 120 | 177 | 0.8331 | Young adult (with falls) |
| SA14 | 297 | 96 | 201 | 90 | 207 | 0.8356 | Young adult (with falls) |
| SE11 | 0 | 0 | 0 | 0 | 0 | N/A | Elderly subject (ADL-only) |
| SE14 | 0 | 0 | 0 | 0 | 0 | N/A | Elderly subject (ADL-only) |

## 3. Severity Classification Context Baselines

| Classifier | Features | Macro-F1 | Mild F1 | Severe F1 | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Majority Class | None (Severe always) | 0.3794 | 0.0000 | 0.7588 | Constant predictor |
| Logistic Regression | Peak SMV + Peak Gyro Mag | 0.6109 | 0.5966 | 0.6251 | Partially circular with K-Means |
| **Multi-Task CNN-LSTM** | End-to-end IMU windows | **0.8660** | **0.8346** | **0.8973** | Joint representation |

> Ground-truth severity labels were obtained via unsupervised K-Means on signal-derived features (peak acc, gyro, energy). Therefore, feature-based Logistic Regression partly re-predicts the clustering input features. The multi-task neural network's gain must be evaluated honestly against this baseline without claiming clinical ground truth.

## 4. Anomaly Fall Windows (-1) Binary Recall

- **Total -1 Fall Windows in Test**: 24
- **Detected as Fall by Binary Head**: 24
- **Binary Recall on -1 Windows**: **1.0000**
- *Note*: Uncertain (-1) windows are extreme gyro outliers; they stay in binary detection but are masked from severity loss.