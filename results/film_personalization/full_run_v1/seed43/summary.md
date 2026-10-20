# FiLM personalization experiment

This report is generated from `metrics_summary.csv` and `paired_wilcoxon.csv`.
No expected-performance curve is prefilled. Read the saved per-user CSV for seed-level results.

## Primary results: real held-out SisFall users

| Model | k | F1 mean +/- std | F2 mean +/- std | False alarms/hour mean +/- std |
|---|---:|---:|---:|---:|
| FiLM (canonical threshold) | 0 | 0.9702 +/- 0.0063 | 0.9864 +/- 0.0032 | 21.8531 +/- 16.6734 |
| FiLM (canonical threshold) | 1 | 0.9502 +/- 0.0127 | 0.9794 +/- 0.0054 | 39.7674 +/- 29.9926 |
| FiLM (canonical threshold) | 5 | 0.9512 +/- 0.0114 | 0.9798 +/- 0.0049 | 38.2584 +/- 29.7976 |
| FiLM (canonical threshold) | 10 | 0.9525 +/- 0.0111 | 0.9804 +/- 0.0047 | 37.1440 +/- 28.8954 |
| FiLM (canonical threshold) | 30 | 0.9552 +/- 0.0076 | 0.9816 +/- 0.0032 | 34.8535 +/- 26.1863 |
| FiLM (validation-retuned threshold) | 0 | 0.9834 +/- 0.0047 | 0.9843 +/- 0.0060 | 6.8097 +/- 5.1922 |
| FiLM (validation-retuned threshold) | 1 | 0.9791 +/- 0.0055 | 0.9853 +/- 0.0057 | 11.7623 +/- 9.4237 |
| FiLM (validation-retuned threshold) | 5 | 0.9790 +/- 0.0069 | 0.9869 +/- 0.0047 | 13.0004 +/- 11.0144 |
| FiLM (validation-retuned threshold) | 10 | 0.9803 +/- 0.0058 | 0.9872 +/- 0.0047 | 11.8242 +/- 9.7238 |
| FiLM (validation-retuned threshold) | 30 | 0.9806 +/- 0.0055 | 0.9872 +/- 0.0043 | 11.5147 +/- 9.4120 |
| Full fine-tune (+2 falls) | 0 | 0.9777 +/- 0.0030 | 0.9880 +/- 0.0030 | 22.3793 +/- 3.2154 |
| Full fine-tune (+2 falls) | 1 | 0.9770 +/- 0.0028 | 0.9877 +/- 0.0030 | 23.1222 +/- 2.9194 |
| Full fine-tune (+2 falls) | 5 | 0.9763 +/- 0.0024 | 0.9874 +/- 0.0028 | 23.9579 +/- 2.9458 |
| Full fine-tune (+2 falls) | 10 | 0.9769 +/- 0.0029 | 0.9876 +/- 0.0030 | 23.3079 +/- 3.1004 |
| Full fine-tune (+2 falls) | 30 | 0.9771 +/- 0.0029 | 0.9877 +/- 0.0031 | 23.0293 +/- 2.9764 |
| Full fine-tune (ADL-only) | 0 | 0.9776 +/- 0.0037 | 0.9880 +/- 0.0034 | 15.1052 +/- 11.2360 |
| Full fine-tune (ADL-only) | 1 | 0.9780 +/- 0.0035 | 0.9881 +/- 0.0033 | 14.7957 +/- 10.9980 |
| Full fine-tune (ADL-only) | 5 | 0.9779 +/- 0.0036 | 0.9881 +/- 0.0033 | 14.8576 +/- 11.0467 |
| Full fine-tune (ADL-only) | 10 | 0.9780 +/- 0.0035 | 0.9881 +/- 0.0033 | 14.7957 +/- 10.9980 |
| Full fine-tune (ADL-only) | 30 | 0.9780 +/- 0.0035 | 0.9881 +/- 0.0033 | 14.7957 +/- 10.9980 |
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

Fine-tuning configurations were selected on validation F2 at threshold 0.25; each method tried 12 configs. Test probability shifts are reporting-only.

| Comparison method | Selection/tuning data | Configurations tried |
|---|---|---:|
| Global | Frozen canonical baseline; shared threshold 0.25 | 0 new configs |
| Last-layer FT | Validation F2 at 0.25; one ADL calibration window/user | 12 |
| Full FT | Validation F2 at 0.25; one ADL calibration window/user | 12 |
| FiLM | Train-subject episodes; validation early stopping/threshold | 1 run configuration |

| Method | Selected learning rate | Selected epochs | Validation F2 | Validation FA/hr |
|---|---:|---:|---:|---:|
| Last-layer FT | 1e-02 | 1 | 0.9848 | 21.7170 |
| Full FT | 1e-04 | 1 | 0.9848 | 21.7170 |

| Method | Test subset | Signed mean shift | Absolute mean shift | Threshold crossings |
|---|---|---:|---:|---:|
| full_model | ADL | -0.000033 | 0.000053 | 5 |
| full_model | Fall | -0.000139 | 0.000419 | 0 |
| last_layer | ADL | -0.000066 | 0.000066 | 5 |
| last_layer | Fall | -0.000222 | 0.000222 | 0 |

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
"C:\Users\drums\Desktop\fall detection project\.venv\Scripts\python.exe" scripts\run_film_personalization.py --output-dir "C:\Users\drums\Desktop\fall detection project\results\film_personalization\full_run_v1\seed43" --seed 43 --epochs 30 --episodes-per-epoch 30 --validation-episodes 12 --patience 5 --finetune-epochs 3 --calibration-seeds 17 31 47 71 101 --timevae-epochs 0
```

Calibration uses ADL windows only for the primary FiLM and fine-tuning baselines. Positive-calibration fine-tuning rows are explicitly labeled.

## Limitations

- Only the existing held-out subject split is used. It has 4 subjects with fall windows and 2 ADL-only subjects; fall-sensitive metrics are N/A for those ADL-only users.
- User profiles are parametric sensor shifts, not evidence of realistic new-user physiology.
- The optional TimeVAE path is future work only; the canonical synthetic users use parametric shifts.
- The last-layer and full fine-tuning baselines with ADL-only calibration have no positive examples.
