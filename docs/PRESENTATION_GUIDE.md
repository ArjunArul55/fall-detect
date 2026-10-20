# Fall Detection Project — Presentation Guide

## Recommended presentation order

1. **Problem and aim** — detect fall impacts from wearable inertial sensor
   streams while distinguishing them from ordinary activities of daily living
   (ADLs).
2. **Dataset** — introduce SisFall and the six retained channels:
   accelerometer `acc1_x/y/z` and gyroscope `gyro_x/y/z`.
3. **Signal preparation** — explain filtering, downsampling, subject-wise
   splitting, train-only standardization, and impact-centered window labels.
4. **Model** — describe the CNN as a local-pattern extractor and the LSTM as
   the part that learns how those patterns evolve over time.
5. **Evaluation** — report window metrics and recording-level detection
  separately, using the held-out subject split.
6. **Additional study** — present severity analysis and FiLM personalization
   as separate research extensions, not as capabilities of the binary
   CNN-LSTM itself.
7. **Limitations and next steps** — discuss controlled-dataset generalization,
   false triggers, severity-label validity, and validation on more diverse
   real-world users.

## Key facts to explain

- Input window: **64 time steps × 6 channels**, sampled at 20 Hz.
- Window duration: **3.2 seconds**; stride: **16 samples / 0.8 seconds**;
  windows overlap by 75%.
- Subject split: **26 train / 6 validation / 6 test subjects**.
- A fall window is positive when it contains the peak-acceleration impact;
  other windows from that fall recording remain background.
- CNN-LSTM structure: two Conv1D blocks with pooling, one 64-unit LSTM,
  global average pooling, a 32-unit dense layer, dropout, and a sigmoid output.
- Headline comparisons use threshold **0.25**, selected by maximum F2 on
  validation and applied unchanged to test. Show **0.50** as a reference column;
  **0.20** is only a FiLM sensitivity row.
- Stored CNN-LSTM reference results at threshold 0.50: fall precision **0.9770**, recall
  **0.9925**, F1 **0.9847**. The stored evaluation summary reports 300/300 fall
  recordings detected. This is a particular held-out split, not a guarantee of
  performance in uncontrolled real-world deployments.

## Suggested live demo

1. Show a short ADL and fall waveform from
   [the project report figures](reports/project_report/report_assets/).
2. Trace one sample window from its six sensor channels to its binary label.
3. Open the model definition in
   [train_cnn_lstm_baseline.py](../scripts/train_cnn_lstm_baseline.py).
4. Show the validation-selected threshold and held-out evaluation artifacts
   under `results/cnn_lstm_baseline/`.
5. Close with the limitations and the distinction between binary fall
   detection, severity analysis, and the FiLM personalization prototype.

## Figure shortlist

- Dataset balance and representative signals:
  [class balance](reports/project_report/report_assets/class_balance.png) and
  [sample waveforms](reports/project_report/report_assets/sample_waveforms_fall_vs_adl.png).
- Windowing and model performance:
  [windowing example](reports/project_report/report_assets/preprocessing_windowing_example.png),
  [CNN-LSTM confusion matrix](reports/project_report/report_assets/cnn_lstm_confusion_matrix.png),
  and [CNN-LSTM training curves](reports/project_report/report_assets/cnn_lstm_training_curves.png).
- Severity extension:
  [severity PCA](presentation/figures/severity/kmeans_severity_test_pca.png)
  and [severity evaluation metrics](presentation/figures/severity/cnn_lstm_severity_evaluation_metrics.png).
- Optional personalization extension:
  [FiLM pipeline](presentation/figures/personalization/09_film_architecture_pipeline.png)
  and [F2 across calibration sizes](presentation/figures/personalization/01_f2_vs_k.png).

For a full report with plots and tables, see
[the project report](reports/project_report/PROJECT_REPORT.md) and
[the technical appendix](reports/technical_appendix/REVIEW2_REPORT.md).

## Metric caveat

The saved project reports do not agree on every recording-level false-trigger
count. Before presenting those counts as a central result, reconcile
`results/cnn_lstm_baseline/recording_level_eval/` with the false-positive
diagnostics generated for the same model and threshold. Prefer metrics read
directly from the run metadata and timestamped evaluation artifacts, and state
the threshold and evaluation level for every number.

Use K=2 (`Mild`, `Severe`, `Uncertain (-1)`) as the canonical severity scheme;
Moderate/K=3 is archived exploration only. The canonical synthetic-user method
is parametric per-channel gain/offset on SisFall windows; TimeGAN/TimeVAE is
future work only. The CNN-LSTM backbone is frozen; a small user encoder is
trained once on simulated users; per-user adaptation at deployment is
inference-only (no gradient steps).
