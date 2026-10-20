# FiLM personalization experiment

This report is generated from `metrics_summary.csv` and `paired_wilcoxon.csv`.
No expected-performance curve is prefilled. Read the saved per-user CSV for seed-level results.

## Primary results: real held-out SisFall users

| Model | k | F1 mean +/- std | F2 mean +/- std | False alarms/hour mean +/- std |
|---|---:|---:|---:|---:|
| FiLM (canonical threshold) | 0 | 0.9595 +/- 0.0054 | 0.9829 +/- 0.0031 | 31.0772 +/- 22.8344 |
| FiLM (canonical threshold) | 1 | 0.9332 +/- 0.0565 | 0.9705 +/- 0.0256 | 76.9175 +/- 85.0871 |
| FiLM (canonical threshold) | 5 | 0.9358 +/- 0.0227 | 0.9731 +/- 0.0100 | 53.9044 +/- 42.0241 |
| FiLM (canonical threshold) | 10 | 0.9343 +/- 0.0198 | 0.9725 +/- 0.0087 | 53.3249 +/- 42.3109 |
| FiLM (canonical threshold) | 30 | 0.9363 +/- 0.0147 | 0.9735 +/- 0.0063 | 52.5802 +/- 37.0194 |
| FiLM (validation-retuned threshold) | 0 | 0.9810 +/- 0.0045 | 0.9854 +/- 0.0038 | 9.9051 +/- 7.8391 |
| FiLM (validation-retuned threshold) | 1 | 0.9755 +/- 0.0135 | 0.9848 +/- 0.0053 | 15.8846 +/- 18.1641 |
| FiLM (validation-retuned threshold) | 5 | 0.9747 +/- 0.0112 | 0.9864 +/- 0.0048 | 17.1482 +/- 16.6662 |
| FiLM (validation-retuned threshold) | 10 | 0.9743 +/- 0.0085 | 0.9862 +/- 0.0040 | 17.4577 +/- 15.1085 |
| FiLM (validation-retuned threshold) | 30 | 0.9750 +/- 0.0070 | 0.9867 +/- 0.0038 | 16.9624 +/- 13.9214 |
| Full fine-tune (+2 falls) | 0 | 0.9777 +/- 0.0029 | 0.9880 +/- 0.0030 | 22.2864 +/- 3.1884 |
| Full fine-tune (+2 falls) | 1 | 0.9769 +/- 0.0029 | 0.9877 +/- 0.0030 | 23.2150 +/- 3.1018 |
| Full fine-tune (+2 falls) | 5 | 0.9763 +/- 0.0022 | 0.9874 +/- 0.0027 | 23.9579 +/- 2.7547 |
| Full fine-tune (+2 falls) | 10 | 0.9768 +/- 0.0029 | 0.9876 +/- 0.0030 | 23.4007 +/- 3.0960 |
| Full fine-tune (+2 falls) | 30 | 0.9771 +/- 0.0030 | 0.9877 +/- 0.0031 | 23.0293 +/- 3.0960 |
| Full fine-tune (ADL-only) | 0 | 0.9776 +/- 0.0037 | 0.9880 +/- 0.0034 | 15.1052 +/- 11.2360 |
| Full fine-tune (ADL-only) | 1 | 0.9780 +/- 0.0025 | 0.9881 +/- 0.0029 | 14.7338 +/- 10.8175 |
| Full fine-tune (ADL-only) | 5 | 0.9786 +/- 0.0039 | 0.9884 +/- 0.0034 | 14.3005 +/- 10.7205 |
| Full fine-tune (ADL-only) | 10 | 0.9787 +/- 0.0039 | 0.9883 +/- 0.0035 | 14.1766 +/- 10.6239 |
| Full fine-tune (ADL-only) | 30 | 0.9784 +/- 0.0034 | 0.9882 +/- 0.0034 | 14.3624 +/- 10.6066 |
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
| simulated_users | FiLM vs Full fine-tune (+2 falls) | 4 | 1 | 0.25 |
| simulated_users | FiLM vs Full fine-tune (ADL-only) | 4 | 2 | 0.375 |
| simulated_users | FiLM vs Global | 4 | 1 | 0.25 |
| simulated_users | FiLM vs Last-layer fine-tune (+2 falls) | 4 | 1 | 0.25 |
| simulated_users | FiLM vs Last-layer fine-tune (ADL-only) | 4 | 2 | 0.375 |

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
| Full FT | 1e-04 | 5 | 0.9850 | 22.3287 |

| Method | Test subset | Signed mean shift | Absolute mean shift | Threshold crossings |
|---|---|---:|---:|---:|
| full_model | ADL | -0.000131 | 0.000230 | 18 |
| full_model | Fall | -0.000833 | 0.002237 | 0 |
| last_layer | ADL | -0.000067 | 0.000067 | 5 |
| last_layer | Fall | -0.000241 | 0.000241 | 0 |

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
"C:\Users\drums\Desktop\fall detection project\.venv\Scripts\python.exe" scripts\run_film_personalization.py --output-dir "C:\Users\drums\Desktop\fall detection project\results\film_personalization\full_run_v1\seed44" --seed 44 --epochs 30 --episodes-per-epoch 30 --validation-episodes 12 --patience 5 --finetune-epochs 3 --calibration-seeds 17 31 47 71 101 --timevae-epochs 0
```

Calibration uses ADL windows only for the primary FiLM and fine-tuning baselines. Positive-calibration fine-tuning rows are explicitly labeled.

## Limitations

- Only the existing held-out subject split is used. It has 4 subjects with fall windows and 2 ADL-only subjects; fall-sensitive metrics are N/A for those ADL-only users.
- User profiles are parametric sensor shifts, not evidence of realistic new-user physiology.
- The optional TimeVAE path is future work only; the canonical synthetic users use parametric shifts.
- The last-layer and full fine-tuning baselines with ADL-only calibration have no positive examples.
