# FiLM personalization experiment

This report is generated from `metrics_summary.csv` and `paired_wilcoxon.csv`.
No expected-performance curve is prefilled. Read the saved per-user CSV for seed-level results.

## Primary results: real held-out SisFall users

| Model | k | F1 mean +/- std | F2 mean +/- std | False alarms/hour mean +/- std |
|---|---:|---:|---:|---:|
| FiLM (canonical threshold) | 0 | 0.9780 +/- 0.0035 | 0.9881 +/- 0.0033 | 14.7957 +/- 10.9980 |
| FiLM (canonical threshold) | 1 | 0.9806 +/- 0.0032 | 0.9891 +/- 0.0033 | 12.6909 +/- 9.4835 |
| FiLM (canonical threshold) | 5 | 0.9798 +/- 0.0032 | 0.9889 +/- 0.0031 | 13.3719 +/- 9.9882 |
| FiLM (canonical threshold) | 10 | 0.9797 +/- 0.0033 | 0.9888 +/- 0.0032 | 13.4957 +/- 10.0893 |
| FiLM (canonical threshold) | 30 | 0.9797 +/- 0.0033 | 0.9888 +/- 0.0031 | 13.4957 +/- 10.1011 |
| FiLM (validation-retuned threshold) | 0 | 0.9756 +/- 0.0019 | 0.9871 +/- 0.0023 | 16.6529 +/- 12.2756 |
| FiLM (validation-retuned threshold) | 1 | 0.9793 +/- 0.0032 | 0.9887 +/- 0.0029 | 13.7433 +/- 10.3301 |
| FiLM (validation-retuned threshold) | 5 | 0.9777 +/- 0.0034 | 0.9880 +/- 0.0030 | 14.9814 +/- 11.2064 |
| FiLM (validation-retuned threshold) | 10 | 0.9776 +/- 0.0034 | 0.9880 +/- 0.0030 | 15.1052 +/- 11.3099 |
| FiLM (validation-retuned threshold) | 30 | 0.9773 +/- 0.0034 | 0.9879 +/- 0.0030 | 15.2910 +/- 11.4349 |
| Full fine-tune (+2 falls) | 0 | 0.9774 +/- 0.0034 | 0.9879 +/- 0.0033 | 22.6579 +/- 3.3332 |
| Full fine-tune (+2 falls) | 1 | 0.9772 +/- 0.0032 | 0.9878 +/- 0.0032 | 22.9364 +/- 3.1469 |
| Full fine-tune (+2 falls) | 5 | 0.9773 +/- 0.0035 | 0.9878 +/- 0.0033 | 22.7507 +/- 3.3481 |
| Full fine-tune (+2 falls) | 10 | 0.9774 +/- 0.0034 | 0.9879 +/- 0.0033 | 22.6579 +/- 3.3332 |
| Full fine-tune (+2 falls) | 30 | 0.9774 +/- 0.0034 | 0.9879 +/- 0.0033 | 22.6579 +/- 3.3332 |
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
| Last-layer fine-tune (ADL-only) | 1 | 0.9776 +/- 0.0037 | 0.9880 +/- 0.0034 | 15.1052 +/- 11.2360 |
| Last-layer fine-tune (ADL-only) | 5 | 0.9776 +/- 0.0037 | 0.9880 +/- 0.0034 | 15.1052 +/- 11.2360 |
| Last-layer fine-tune (ADL-only) | 10 | 0.9776 +/- 0.0037 | 0.9880 +/- 0.0034 | 15.1052 +/- 11.2360 |
| Last-layer fine-tune (ADL-only) | 30 | 0.9776 +/- 0.0037 | 0.9880 +/- 0.0034 | 15.1052 +/- 11.2360 |

## Paired Wilcoxon comparisons

| Evaluation | Comparison | Subjects | Statistic | p-value |
|---|---|---:|---:|---:|
| real_heldout | FiLM vs Full fine-tune (+2 falls) | 4 | 0 | 0.125 |
| real_heldout | FiLM vs Full fine-tune (ADL-only) | 4 | 0 | 0.125 |
| real_heldout | FiLM vs Global | 4 | 0 | 0.125 |
| real_heldout | FiLM vs Last-layer fine-tune (+2 falls) | 4 | 0 | 0.125 |
| real_heldout | FiLM vs Last-layer fine-tune (ADL-only) | 4 | 0 | 0.125 |
| simulated_users | FiLM vs Full fine-tune (+2 falls) | 4 | 3 | 0.625 |
| simulated_users | FiLM vs Full fine-tune (ADL-only) | 4 | 5 | 1 |
| simulated_users | FiLM vs Global | 4 | 4 | 0.875 |
| simulated_users | FiLM vs Last-layer fine-tune (+2 falls) | 4 | 3 | 0.625 |
| simulated_users | FiLM vs Last-layer fine-tune (ADL-only) | 4 | 4 | 0.875 |

## Model and deployment size

- Frozen backbone parameters: 46817.
- Shared UserEncoder parameters: 24960.
- Per-user state: 128 float32 values (512 bytes).
- TFLite head sizes: 194896 bytes base; 197940 bytes FiLM; overhead 3044 bytes.
- TFLite max absolute output errors: base 5.96e-08; FiLM 0.

## Evaluation details

The real held-out evaluation contains 6 subjects. False alarms/hour uses a stride of 16 samples at 20 Hz (0.8 seconds per row).
The positive-calibration baselines add 2 fall windows and are labeled as such.
Global baselines and FiLM canonical-threshold rows use 0.250; the FiLM validation-retuned threshold is 0.200.

## Reproduction

```powershell
"C:\Users\drums\Desktop\fall detection project\.venv\Scripts\python.exe" scripts\run_film_personalization.py --seed 42 --epochs 2 --episodes-per-epoch 3 --validation-episodes 2 --patience 1 --finetune-epochs 1 --calibration-seeds 17 31 47 71 101 --timevae-epochs 0
```

Calibration uses ADL windows only for the primary FiLM and fine-tuning baselines. Positive-calibration fine-tuning rows are explicitly labeled.

## Limitations

- Only the existing held-out subject split is used. It has 4 subjects with fall windows and 2 ADL-only subjects; fall-sensitive metrics are N/A for those ADL-only users.
- User profiles are parametric sensor shifts, not evidence of realistic new-user physiology.
- The optional TimeVAE can generate fall-labeled windows but does not validate real fall realism.
- The last-layer and full fine-tuning baselines with ADL-only calibration have no positive examples.
