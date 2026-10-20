# FiLM personalization experiment

This report is generated from `metrics_summary.csv` and `paired_wilcoxon.csv`.
No expected-performance curve is prefilled. Read the saved per-user CSV for seed-level results.

## Primary results: real held-out SisFall users

| Model | k | F1 mean +/- std | F2 mean +/- std | False alarms/hour mean +/- std |
|---|---:|---:|---:|---:|
| FiLM (canonical threshold) | 0 | 0.9760 +/- 0.0023 | 0.9873 +/- 0.0026 | 16.3434 +/- 12.0499 |
| FiLM (canonical threshold) | 1 | 0.9498 +/- 0.0186 | 0.9789 +/- 0.0080 | 39.3727 +/- 34.1944 |
| FiLM (canonical threshold) | 5 | 0.9610 +/- 0.0067 | 0.9835 +/- 0.0035 | 29.8390 +/- 22.2914 |
| FiLM (canonical threshold) | 10 | 0.9626 +/- 0.0059 | 0.9842 +/- 0.0033 | 28.5390 +/- 21.1500 |
| FiLM (canonical threshold) | 30 | 0.9625 +/- 0.0055 | 0.9842 +/- 0.0031 | 28.6009 +/- 21.1056 |
| FiLM (validation-retuned threshold) | 0 | 0.9798 +/- 0.0071 | 0.9759 +/- 0.0090 | 4.9525 +/- 4.5839 |
| FiLM (validation-retuned threshold) | 1 | 0.9801 +/- 0.0071 | 0.9878 +/- 0.0044 | 12.4433 +/- 10.8199 |
| FiLM (validation-retuned threshold) | 5 | 0.9831 +/- 0.0045 | 0.9880 +/- 0.0037 | 9.4717 +/- 7.4981 |
| FiLM (validation-retuned threshold) | 10 | 0.9839 +/- 0.0046 | 0.9879 +/- 0.0035 | 8.6050 +/- 6.9402 |
| FiLM (validation-retuned threshold) | 30 | 0.9836 +/- 0.0044 | 0.9878 +/- 0.0033 | 8.7908 +/- 7.0835 |
| Full fine-tune (+2 falls) | 0 | 0.9777 +/- 0.0030 | 0.9880 +/- 0.0030 | 22.3793 +/- 3.2154 |
| Full fine-tune (+2 falls) | 1 | 0.9769 +/- 0.0029 | 0.9877 +/- 0.0030 | 23.2150 +/- 3.1018 |
| Full fine-tune (+2 falls) | 5 | 0.9764 +/- 0.0023 | 0.9874 +/- 0.0028 | 23.8650 +/- 2.7793 |
| Full fine-tune (+2 falls) | 10 | 0.9768 +/- 0.0030 | 0.9876 +/- 0.0030 | 23.4007 +/- 3.3223 |
| Full fine-tune (+2 falls) | 30 | 0.9769 +/- 0.0027 | 0.9877 +/- 0.0030 | 23.2150 +/- 2.8582 |
| Full fine-tune (ADL-only) | 0 | 0.9776 +/- 0.0037 | 0.9880 +/- 0.0034 | 15.1052 +/- 11.2360 |
| Full fine-tune (ADL-only) | 1 | 0.9813 +/- 0.0032 | 0.9893 +/- 0.0031 | 12.1337 +/- 9.1601 |
| Full fine-tune (ADL-only) | 5 | 0.9806 +/- 0.0040 | 0.9878 +/- 0.0057 | 11.8861 +/- 8.9879 |
| Full fine-tune (ADL-only) | 10 | 0.9812 +/- 0.0034 | 0.9882 +/- 0.0048 | 11.4527 +/- 8.5468 |
| Full fine-tune (ADL-only) | 30 | 0.9808 +/- 0.0046 | 0.9876 +/- 0.0062 | 11.5147 +/- 8.5928 |
| Global | 0 | 0.9776 +/- 0.0037 | 0.9880 +/- 0.0034 | 15.1052 +/- 11.2360 |
| Global | 1 | 0.9776 +/- 0.0037 | 0.9880 +/- 0.0034 | 15.1052 +/- 11.2360 |
| Global | 5 | 0.9776 +/- 0.0037 | 0.9880 +/- 0.0034 | 15.1052 +/- 11.2360 |
| Global | 10 | 0.9776 +/- 0.0037 | 0.9880 +/- 0.0034 | 15.1052 +/- 11.2360 |
| Global | 30 | 0.9776 +/- 0.0037 | 0.9880 +/- 0.0034 | 15.1052 +/- 11.2360 |
| Last-layer fine-tune (+2 falls) | 0 | 0.9774 +/- 0.0037 | 0.9879 +/- 0.0034 | 22.6579 +/- 3.5443 |
| Last-layer fine-tune (+2 falls) | 1 | 0.9774 +/- 0.0037 | 0.9879 +/- 0.0034 | 22.6579 +/- 3.5443 |
| Last-layer fine-tune (+2 falls) | 5 | 0.9774 +/- 0.0037 | 0.9879 +/- 0.0034 | 22.6579 +/- 3.5443 |
| Last-layer fine-tune (+2 falls) | 10 | 0.9774 +/- 0.0037 | 0.9879 +/- 0.0034 | 22.6579 +/- 3.5443 |
| Last-layer fine-tune (+2 falls) | 30 | 0.9774 +/- 0.0037 | 0.9879 +/- 0.0034 | 22.6579 +/- 3.5443 |
| Last-layer fine-tune (ADL-only) | 0 | 0.9776 +/- 0.0037 | 0.9880 +/- 0.0034 | 15.1052 +/- 11.2360 |
| Last-layer fine-tune (ADL-only) | 1 | 0.9780 +/- 0.0035 | 0.9881 +/- 0.0033 | 14.7957 +/- 10.9980 |
| Last-layer fine-tune (ADL-only) | 5 | 0.9780 +/- 0.0035 | 0.9881 +/- 0.0033 | 14.7957 +/- 10.9980 |
| Last-layer fine-tune (ADL-only) | 10 | 0.9780 +/- 0.0035 | 0.9881 +/- 0.0033 | 14.7957 +/- 10.9980 |
| Last-layer fine-tune (ADL-only) | 30 | 0.9780 +/- 0.0035 | 0.9881 +/- 0.0033 | 14.7957 +/- 10.9980 |

## Paired Wilcoxon comparisons

| Evaluation | Comparison | Subjects | Statistic | p-value |
|---|---|---:|---:|---:|
| real_heldout | FiLM vs Full fine-tune (+2 falls) | 4 | 0 | 0.125 |
| real_heldout | FiLM vs Full fine-tune (ADL-only) | 4 | 0 | 0.125 |
| real_heldout | FiLM vs Global | 4 | 0 | 0.125 |
| real_heldout | FiLM vs Last-layer fine-tune (+2 falls) | 4 | 0 | 0.125 |
| real_heldout | FiLM vs Last-layer fine-tune (ADL-only) | 4 | 0 | 0.125 |
| simulated_users | FiLM vs Full fine-tune (+2 falls) | 4 | 4 | 0.875 |
| simulated_users | FiLM vs Full fine-tune (ADL-only) | 4 | 5 | 1 |
| simulated_users | FiLM vs Global | 4 | 5 | 1 |
| simulated_users | FiLM vs Last-layer fine-tune (+2 falls) | 4 | 5 | 1 |
| simulated_users | FiLM vs Last-layer fine-tune (ADL-only) | 4 | 5 | 1 |

## Validation tuning and probability shifts

Fine-tuning configurations were selected on validation F2 at threshold 0.25; each method tried 9 configs. Test probability shifts are reporting-only.

| Comparison method | Selection/tuning data | Configurations tried |
|---|---|---:|
| Global | Frozen canonical baseline; shared threshold 0.25 | 0 new configs |
| Last-layer FT | Validation F2 at 0.25; one ADL calibration window/user | 9 |
| Full FT | Validation F2 at 0.25; one ADL calibration window/user | 9 |
| FiLM | Train-subject episodes; validation early stopping/threshold | 1 run configuration |

| Method | Selected learning rate | Selected epochs | Validation F2 | Validation FA/hr |
|---|---:|---:|---:|---:|
| Last-layer FT | 1e-02 | 1 | 0.9848 | 21.7170 |
| Full FT | 1e-03 | 1 | 0.9853 | 19.2700 |

| Method | Test subset | Signed mean shift | Absolute mean shift | Threshold crossings |
|---|---|---:|---:|---:|
| full_model | ADL | -0.000529 | 0.000531 | 48 |
| full_model | Fall | -0.005762 | 0.005811 | 2 |
| last_layer | ADL | -0.000059 | 0.000059 | 5 |
| last_layer | Fall | -0.000210 | 0.000210 | 0 |

## Model and deployment size

- Frozen backbone parameters: 46817.
- Shared UserEncoder parameters: 24960.
- Per-user state: 128 float32 values (512 bytes).
- TFLite head sizes: 195164 bytes base; 197940 bytes FiLM; overhead 2776 bytes.
- TFLite max absolute output errors: base 5.96e-08; FiLM 0.

## Evaluation details

The real held-out evaluation contains 6 subjects. False alarms/hour uses a stride of 16 samples at 20 Hz (0.8 seconds per row).
The positive-calibration baselines add 2 fall windows and are labeled as such.
Global baselines and FiLM canonical-threshold rows use 0.250; the FiLM validation-retuned threshold is 0.850.

## Reproduction

```powershell
"C:\Users\drums\Desktop\fall detection project\.venv\Scripts\python.exe" scripts\run_film_personalization.py --seed 42 --epochs 1 --episodes-per-epoch 30 --validation-episodes 12 --patience 5 --finetune-epochs 3 --calibration-seeds 17 31 47 71 101 --timevae-epochs 0
```

Calibration uses ADL windows only for the primary FiLM and fine-tuning baselines. Positive-calibration fine-tuning rows are explicitly labeled.

## Limitations

- Only the existing held-out subject split is used. It has 4 subjects with fall windows and 2 ADL-only subjects; fall-sensitive metrics are N/A for those ADL-only users.
- User profiles are parametric sensor shifts, not evidence of realistic new-user physiology.
- The optional TimeVAE path is future work only; the canonical synthetic users use parametric shifts.
- The last-layer and full fine-tuning baselines with ADL-only calibration have no positive examples.
