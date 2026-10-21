# Multi-Task Severity Head Run Summary (Task P2)

- **Run Directory**: `results/severity_multitask/run_v1`
- **Generated (UTC)**: 2026-10-21T12:20:59.455639+00:00
- **Best Lambda**: 1.0 (Epoch 6)
- **Validation-Selected Binary Threshold**: 0.20 (0.50 reference)
- **Training Time**: 111.29 s

## 1. Binary Fall Detection: Baseline CNN-LSTM vs Multi-Task

| Model | Threshold | Accuracy | Precision | Recall | F1 | F2 | FA/hr | AUROC | AUPRC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CNN-LSTM Baseline | 0.25 (val-sel) | 0.9960 | 0.9606 | 0.9950 | 0.9775 | 0.9879 | 17.74 | 0.9998 | 0.9984 |
| **Multi-Task CNN-LSTM** | **0.20 (val-sel)** | **0.0990** | **0.0883** | **0.9900** | **0.1621** | **0.3254** | **4441.70** | **0.5036** | **0.4631** |
| Multi-Task CNN-LSTM | 0.50 (ref) | 0.9119 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.5036 | 0.4631 |

- **Recall Regression Check**: Baseline Recall = 0.9950, Multi-Task Recall = 0.9900. Regressed: `True`.

## 2. Severity Classification on Confident Fall Windows

| Level | Evaluated Samples | Mild P/R/F1 | Severe P/R/F1 | Macro-F1 | Confusion Matrix (Mild, Severe) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Window | 1176 | 0.0000 / 0.0000 / 0.0000 | 0.6114 / 1.0000 / 0.7588 | **0.3794** | [[0, 457], [0, 719]] |
| Recording | 297 | - | - | **0.3812** | [[0, 114], [0, 183]] |
| Subject | 6 | - | - | - | [[0, 0], [0, 4]] |

### Per-Subject Severity Window Breakdown

| Subject ID | Total Windows | True Mild | True Severe | Pred Mild | Pred Severe | Macro-F1 | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SA02 | 285 | 109 | 176 | 0 | 285 | 0.3818 | Young adult (with falls) |
| SA03 | 297 | 128 | 169 | 0 | 297 | 0.3627 | Young adult (with falls) |
| SA09 | 297 | 124 | 173 | 0 | 297 | 0.3681 | Young adult (with falls) |
| SA14 | 297 | 96 | 201 | 0 | 297 | 0.4036 | Young adult (with falls) |
| SE11 | 0 | 0 | 0 | 0 | 0 | N/A | Elderly subject (ADL-only) |
| SE14 | 0 | 0 | 0 | 0 | 0 | N/A | Elderly subject (ADL-only) |

## 3. Severity Classification Context Baselines

| Classifier | Features | Macro-F1 | Mild F1 | Severe F1 | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Majority Class | None (Severe always) | 0.3794 | 0.0000 | 0.7588 | Constant predictor |
| Logistic Regression | Peak SMV + Peak Gyro Mag | 0.4468 | 0.4628 | 0.4308 | Partially circular with K-Means |
| **Multi-Task CNN-LSTM** | End-to-end IMU windows | **0.3794** | **0.0000** | **0.7588** | Joint representation |

> Ground-truth severity labels were obtained via unsupervised K-Means on signal-derived features (peak acc, gyro, energy). Therefore, feature-based Logistic Regression partly re-predicts the clustering input features. The multi-task neural network's gain must be evaluated honestly against this baseline without claiming clinical ground truth.

## 4. Anomaly Fall Windows (-1) Binary Recall

- **Total -1 Fall Windows in Test**: 24
- **Detected as Fall by Binary Head**: 23
- **Binary Recall on -1 Windows**: **0.9583**
- *Note*: Uncertain (-1) windows are extreme gyro outliers; they stay in binary detection but are masked from severity loss.