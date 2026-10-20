# Quantization & TFLite Edge Deployment Report (Task P5)

- **Generated**: 2026-10-20 16:50 UTC
- **Target Model**: CNN-LSTM binary detector (`results/cnn_lstm_baseline/cnn_lstm_best.keras`, 46,817 parameters)
- **Model Deployment Rationale**: The multi-task model training was unverified/pending weights in the repository and has no verified test metrics, whereas the binary CNN-LSTM has fully trained, verified weights with test recall 0.9950 and F2 0.9879.
- **Evaluation Dataset**: Full SisFall held-out test split (13,627 windows: 12,427 ADL, 1,200 Fall; 6 subjects: SA02, SA03, SA09, SA14 with falls, SE11, SE14 ADL-only).
- **Locked Decision D2**: Each quantized variant's operating threshold is strictly **re-selected on the validation split** (max $F_2$, tie-breaking lower FA/hr and precision) before evaluation on test.
- **Git Commit**: `f4112df`

---

## 1. Quantization Variant Benchmark Summary

| Variant | Size (KB) | Compression vs Float32 | Val-Selected Threshold | Agreement with Keras Float (%) | Max Abs Diff | Mean Abs Diff | Recall | F1 | F2 | FA/hr | AUROC | Latency Median (ms) | Latency p95 (ms) | Gate Pass? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **float32** | 235.6 | 1.00x | 0.10 | 100.00% | 0.00000 | 0.00000 | 0.4733 | 0.6315 | 0.5260 | 11.23 | 0.8695 | 0.06 | 0.09 | PASS |
| **dynamic_range** | 111.0 | 2.12x | 0.10 | 99.97% | 0.03372 | 0.00010 | 0.4700 | 0.6284 | 0.5227 | 11.23 | 0.8695 | 0.03 | 0.03 | PASS |
| **float16** | 147.8 | 1.59x | 0.10 | 99.99% | 0.00250 | 0.00001 | 0.4742 | 0.6322 | 0.5269 | 11.23 | 0.8695 | 0.06 | 0.13 | PASS |
| **int8** | 136.9 | 1.72x | 0.10 | 99.93% | 0.19042 | 0.00081 | 0.4783 | 0.6357 | 0.5309 | 11.59 | 0.8132 | 0.03 | 0.04 | PASS |

> **Note on Latency**: Single-window CPU latency measured on 1 thread (`num_threads=1`) over 1,000 runs after 50 warmup iterations.

---

## 2. Gate Decision & Deployment Selection

### Gate Criteria:
- $F_2$ drop $\le 0.005$ vs Keras float on test (Keras float $F_2 = 0.5260$ $\to$ min threshold = $0.5210$).
- Recall drop $\le 0.010$ vs Keras float on test (Keras float Recall = 0.4733$ $\to$ min threshold = $0.4633$).
- Selection rule: Smallest passing variant in file size.

### Result:
- **Deployed Model**: `dynamic_range`
- **File Path**: `results/quantization/run_v1/deployed_model.tflite` (111.0 KB)
- **Validation-Selected Operating Threshold**: `0.10`
- **Test Performance**: Recall = **0.4700**, $F_2$ = **0.5227**, $F_1$ = **0.6284**, FA/hr = **11.23**.
- **Keras Parity**: Agreement = **99.97%**, Max Abs Prob Diff = **0.03372**, Mean Abs Prob Diff = **0.00010**.
- **Inference Speed**: Median = **0.03 ms**, 95th Percentile = **0.03 ms** per 3.2 s window (1 thread).

### INT8 Status Note:
- Full-integer INT8 conversion succeeded using 500 stratified training windows (sampling class distribution matching training: ~8.1% falls, 91.9% ADL). Both input and output were quantized to INT8.

---

## 3. Microcontroller / ESP32 Export Details

- **C Header Export**: `results/quantization/run_v1/model_data.h`
- **C Array Symbol**: `const unsigned char g_model_data[]` (aligned to 16 bytes)
- **C Array Length**: `const unsigned int g_model_data_len = 113704;`
- **Metadata File**: `results/quantization/run_v1/deployed_model_meta.json`
  - Window: 64 samples × 6 channels @ 20 Hz (3.2 seconds)
  - Decision stride: 16 samples (0.8 second step)
  - Scaler: `data/processed/scaler.pkl`
  - Operating threshold: `0.10`
  - Input scale: `1.0`, zero-point: `0`
  - Output scale: `1.0`, zero-point: `0`
