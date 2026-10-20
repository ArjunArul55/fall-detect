# Fall Detection Project: Complete Status and Data Guide

> Generated from the files currently present in the repository on 2026-10-20. This is the consolidated project report; older generated Markdown reports were removed to avoid conflicting summaries.

## 1. Executive Summary

**Locked reporting policy:** canonical severity is K=2 (`k2_20260914T055946Z`: Mild, Severe, Uncertain (-1)); K=3 is archived and non-canonical. All headline model metrics use threshold 0.25, selected on validation by maximum F2 and held fixed for test. The FiLM sensitivity row reports that run's validation-retuned threshold as selected; 0.50 is reference only.

This project includes raw SisFall ingestion, signal preprocessing, subject-wise splitting, normalized sliding-window generation, exploratory data analysis, impact-centered binary labeling, unsupervised fall-severity/anomaly analysis, and supervised CNN/CNN-LSTM fall-detection baselines.

The current canonical binary dataset contains **84,123 windows** with shape `(64, 6)`: **76,939 ADL/background** windows and **7,184 impact-centered fall** windows. The six channels are normalized accelerometer and gyroscope channels. Severity labels are derived only for the 7,184 fall windows.

## 2. What Has Been Done

1. **Raw dataset discovery and loading:** SisFall subject directories and activity text files were discovered and parsed. Each raw row has nine numeric sensor values separated by commas and terminated by a semicolon.
2. **Metadata construction:** Subject ID, activity code, activity name, recording ID, source file/path, binary label, and per-recording timestamp were added.
3. **Activity mapping:** `D01`-`D19` are ADL activities and `F01`-`F15` are falls. Binary mapping is `0 = ADL/background`, `1 = fall`.
4. **Channel selection:** The first accelerometer (`acc1`) and gyroscope (`gyro`) tri-axial channels were retained. The second accelerometer (`acc2`, MMA8451Q) was removed, leaving six input channels.
5. **Filtering:** A Butterworth low-pass filter was applied independently per recording before downsampling. The configured cutoff is 5 Hz; the generated filter plot provides a visual check.
6. **Downsampling:** Signals were reduced from 200 Hz to 20 Hz using a factor of 10, independently per recording.
7. **Leakage-resistant splitting:** Subjects were assigned to train/validation/test at approximately 70/15/15, so a subject cannot occur in more than one split.
8. **Normalization:** A training-only `StandardScaler` was fitted on the six selected channels and then applied to all splits.
9. **Windowing:** Windows contain 64 samples with stride 16. At 20 Hz, each window is 3.2 seconds and each stride is 0.8 seconds, producing 75% overlap.
10. **Impact-centered labels:** For fall recordings, the raw acceleration-magnitude peak identifies the impact. Only windows containing that impact receive label 1; other windows from the same recording remain background label 0.
11. **EDA and quality checks:** Class balance, signal distributions, autocorrelation, outliers, acceleration peaks, channel health, and integrity checks were exported to CSV and plots.
12. **Unsupervised severity labeling:** Fall windows were transformed into 58 statistical features, scaled using train fall windows, clustered with KMeans, and ordered by acceleration/gyroscope intensity into Mild and Severe (K=2, canonical run `k2_20260914T055946Z`). The earlier K=3 run is archived for reference.
13. **Anomaly refinement:** A train-only Isolation Forest flags unusual fall windows. Anomalous fall severity is changed to `-1` in refined labels, while the original cluster labels remain available.
14. **Automated testing:** pytest run on 2026-10-20: **40 tests** (36 passed, 0 failed, 0 errors, 4 skipped; overall passed).

## 3. Current Dataset Statistics

| split | windows | shape | subjects | recordings | ADL/background | fall | fall_% | NaN | Inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 55423 | (55423, 64, 6) [metadata] | 26 | 2942 | 50939 | 4484 | 8.0905 | UNVERIFIED (.npy missing) | UNVERIFIED (.npy missing) |
| val | 15073 | (15073, 64, 6) [metadata] | 6 | 829 | 13573 | 1500 | 9.95157 | UNVERIFIED (.npy missing) | UNVERIFIED (.npy missing) |
| test | 13627 | (13627, 64, 6) [metadata] | 6 | 734 | 12427 | 1200 | 8.80605 | UNVERIFIED (.npy missing) | UNVERIFIED (.npy missing) |

Total windows: **84,123**. Total ADL/background: **76,939**. Total fall: **7,184**. Overall fall percentage: **8.54%**.

Integrity result: all processed binary arrays have zero NaN and zero infinite values. The array-level checks also report no flat-line or constant windows in the previous verification output.

## 4. Feature Map and Units

| Raw columns | Sensor | Physical interpretation | Unit in source dataset | Used? |
| --- | --- | --- | --- | --- |
| `adxl345_x/y/z` | ADXL345 accelerometer | Linear acceleration by axis | Dataset sensor units; confirm against SisFall documentation before converting to g | Yes, renamed `acc1_x/y/z` |
| `itg3200_x/y/z` | ITG3200 gyroscope | Angular velocity by axis | Dataset sensor units; confirm against SisFall documentation before converting to deg/s or rad/s | Yes, renamed `gyro_x/y/z` |
| `mma8451q_x/y/z` | MMA8451Q accelerometer | Second accelerometer triad | Dataset sensor units | No, removed from model input |

The final tensor is `(number_of_windows, 64, 6)`. Each timestep has `acc1_x`, `acc1_y`, `acc1_z`, `gyro_x`, `gyro_y`, and `gyro_z`. Values are **standardized z-scores**, calculated as `(value - training_mean) / training_std`; therefore they are dimensionless. Do not interpret normalized values directly as g or angular velocity without applying the saved scaler in reverse.

Each fall window becomes **58 features**: 42 per-channel values (six channels times mean, standard deviation, minimum, maximum, range, RMS, and absolute peak) plus 16 magnitude values (accelerometer and gyroscope magnitude times mean, standard deviation, minimum, maximum, range, RMS, peak, and mean-square energy). Feature names are stored in `data/processed/severity/runs/k2_20260914T055946Z/feature_names.json`.

Magnitude definitions: `acc_mag = sqrt(acc1_x^2 + acc1_y^2 + acc1_z^2)` and `gyro_mag = sqrt(gyro_x^2 + gyro_y^2 + gyro_z^2)`. Since the input is normalized, these magnitudes are normalized-coordinate magnitudes, not physical acceleration or angular-velocity units.

## 5. Severity and Anomaly Results

The canonical current result is `0 = Mild`, `1 = Severe`, and `-1 = non-applicable or uncertain`. ADL/background windows are always `-1` for severity. KMeans is fitted only on train fall windows with `k=2`, `random_state=42`, and `n_init=20`. The Isolation Forest uses contamination 0.02, 256 estimators, and random state 42.

The canonical post-Isolation Forest fall export contains 7,184 rows: train 4,484, validation 1,500, and test 1,200. Anomaly counts are train 90 (2.01%), validation 23 (1.53%), and test 24 (2.00%). Anomalous fall windows are uncertain, not automatically bad data; review or exclude them deliberately during modeling.

Uncertain (`-1`) fall windows show elevated peak gyro magnitude (split medians about 5.9–7.0k, versus about 4.2–4.6k for Severe) and are enriched 6.5–7.4x in the top-decile gyro windows relative to their overall split share. They are masked from severity loss and reported separately; this indicates an anomaly association, not clinical severity.

### Important artifact consistency note

The repository contains artifacts from multiple severity-analysis runs. The canonical result is the versioned K=2 run under `data/processed/severity/runs/k2_20260914T055946Z/`; the earlier K=3 run is archived under `data/processed/deprecated_severity/`. Some unversioned files directly under `data/processed/severity/` may belong to older runs. Do not mix them with the canonical K=2 outputs; use one timestamped run directory for analysis.

## 6. Stored Arrays and Models

### Array inventory

| artifact | shape | dtype | min | max |
| --- | --- | --- | --- | --- |
| data/processed/severity/runs/k2_20260914T055946Z/clustering_features/test_fall_features.npy | (1200, 58) | float64 | -48.5212 | 99.9793 |
| data/processed/severity/runs/k2_20260914T055946Z/clustering_features/test_fall_indices.npy | (1200,) | int64 | 1636 | 11002 |
| data/processed/severity/runs/k2_20260914T055946Z/clustering_features/train_fall_features.npy | (4484, 58) | float64 | -36.8906 | 71.3756 |
| data/processed/severity/runs/k2_20260914T055946Z/clustering_features/train_fall_indices.npy | (4484,) | int64 | 1633 | 47694 |
| data/processed/severity/runs/k2_20260914T055946Z/clustering_features/val_fall_features.npy | (1500, 58) | float64 | -18.9812 | 29.4566 |
| data/processed/severity/runs/k2_20260914T055946Z/clustering_features/val_fall_indices.npy | (1500,) | int64 | 1634 | 13757 |
| data/processed/severity/runs/k2_20260914T055946Z/test_anomaly_flags.npy | (1200,) | int64 | 0 | 1 |
| data/processed/severity/runs/k2_20260914T055946Z/test_anomaly_scores.npy | (1200,) | float64 | -0.0602334 | 0.135932 |
| data/processed/severity/runs/k2_20260914T055946Z/test_refined_severity_labels.npy | (13627,) | int64 | -1 | 1 |
| data/processed/severity/runs/k2_20260914T055946Z/test_severity_labels.npy | (13627,) | int64 | -1 | 1 |
| data/processed/severity/runs/k2_20260914T055946Z/train_anomaly_flags.npy | (4484,) | int64 | 0 | 1 |
| data/processed/severity/runs/k2_20260914T055946Z/train_anomaly_scores.npy | (4484,) | float64 | -0.0730676 | 0.142574 |
| data/processed/severity/runs/k2_20260914T055946Z/train_refined_severity_labels.npy | (55423,) | int64 | -1 | 1 |
| data/processed/severity/runs/k2_20260914T055946Z/train_severity_labels.npy | (55423,) | int64 | -1 | 1 |
| data/processed/severity/runs/k2_20260914T055946Z/val_anomaly_flags.npy | (1500,) | int64 | 0 | 1 |
| data/processed/severity/runs/k2_20260914T055946Z/val_anomaly_scores.npy | (1500,) | float64 | -0.0325907 | 0.136228 |
| data/processed/severity/runs/k2_20260914T055946Z/val_refined_severity_labels.npy | (15073,) | int64 | -1 | 1 |
| data/processed/severity/runs/k2_20260914T055946Z/val_severity_labels.npy | (15073,) | int64 | -1 | 1 |

`train.npy`, `val.npy`, and `test.npy` are the primary model inputs. Their matching label, subject, and recording arrays must remain aligned by row. `X_train_balanced.npy` and `y_train_balanced.npy` are additional class-balancing artifacts and should be used only when an experiment explicitly selects them.

### Pickled estimator inventory

| artifact | python_type | details |
| --- | --- | --- |
| data/processed/severity/runs/k2_20260914T055946Z/feature_scaler.pkl | StandardScaler | features=58 |
| data/processed/severity/runs/k2_20260914T055946Z/kmeans_model.pkl | KMeans | features=58, clusters=2 |
| data/processed/severity/runs/k2_20260914T055946Z/severity_kmeans_model.pkl | KMeans | features=58, clusters=2 |
| data/processed/severity/runs/k2_20260914T055946Z/severity_scaler.pkl | StandardScaler | features=58 |

Stored model/scaler meanings: `scaler.pkl` is the six-channel preprocessing scaler; `feature_scaler.pkl` and `severity_scaler.pkl` are severity-feature scalers from different runs/naming conventions; `kmeans_model.pkl` and `severity_kmeans_model.pkl` are KMeans artifacts from different runs; `isolation_forest.pkl` is the anomaly estimator. These are unsupervised preprocessing/label-generation artifacts, not a trained fall-detection classifier.

## 7. CSV Inventory, Statistics, and Samples

The following sections cover every project CSV outside `.venv`. For each file, `Shape` gives rows and columns, `Missing cells` counts blank/NaN values, numeric statistics describe each numeric column, and the sample is the first 10 rows. Blank cells in the sample are intentional missing/non-applicable values.

### `data/processed/channel_statistics.csv`

- Shape: `18 rows x 13 columns`
- Missing cells: `0`
- Columns: `split, channel, mean, std, min, max, p25, p50, p75, samples, windows, label_0, label_1`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| mean | 18 | -0.0917422 | 0.0164689 | 0.000120642 | 0.0909627 | 0.324814 |
| std | 18 | 0.989853 | 1.06547 | 1.05912 | 0.0649212 | 1.19414 |
| min | 18 | -48.5212 | -17.5627 | -12.9511 | 12.3762 | -5.63588 |
| max | 18 | 6.51522 | 16.1045 | 13.3402 | 9.68905 | 38.1595 |
| p25 | 18 | -0.589018 | -0.279547 | -0.195697 | 0.177561 | -0.106129 |
| p50 | 18 | -0.416074 | -0.0394165 | 0.00209213 | 0.185826 | 0.355942 |
| p75 | 18 | 0.119958 | 0.374788 | 0.237884 | 0.290536 | 0.950644 |
| samples | 18 | 13627 | 28041 | 15073 | 19932.6 | 55423 |
| windows | 18 | 13627 | 28041 | 15073 | 19932.6 | 55423 |
| label_0 | 18 | 12427 | 25646.3 | 13573 | 18409.4 | 50939 |
| label_1 | 18 | 1200 | 2394.67 | 1500 | 1525.43 | 4484 |

First 10 rows:

| split | channel | mean | std | min | max | p25 | p50 | p75 | samples | windows | label_0 | label_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | acc1_x | 0.000451948 | 1.00911 | -13.8934 | 11.6163 | -0.199323 | 0.0297179 | 0.246185 | 55423 | 55423 | 50939 | 4484 |
| train | acc1_y | -0.0120695 | 1.0142 | -6.89975 | 6.94432 | -0.554049 | -0.377854 | 0.817334 | 55423 | 55423 | 50939 | 4484 |
| train | acc1_z | -0.0322971 | 0.989853 | -10.3763 | 8.83929 | -0.587955 | -0.0817941 | 0.438412 | 55423 | 55423 | 50939 | 4484 |
| train | gyro_x | 0.000523963 | 1.07316 | -36.8906 | 38.1595 | -0.192071 | 6.42881e-05 | 0.229583 | 55423 | 55423 | 50939 | 4484 |
| train | gyro_y | -0.00172154 | 1.06964 | -29.6888 | 27.5684 | -0.163348 | 0.00198487 | 0.169796 | 55423 | 55423 | 50939 | 4484 |
| train | gyro_z | -0.000210665 | 1.07406 | -38.0956 | 37.6419 | -0.136826 | 0.00219938 | 0.133601 | 55423 | 55423 | 50939 | 4484 |
| val | acc1_x | -0.0917422 | 1.08453 | -12.0087 | 11.1325 | -0.293318 | -0.0360446 | 0.199054 | 15073 | 15073 | 13573 | 1500 |
| val | acc1_y | 0.00972959 | 1.04219 | -7.25511 | 6.51522 | -0.567018 | -0.382135 | 0.950644 | 15073 | 15073 | 13573 | 1500 |
| val | acc1_z | 0.324814 | 1.00022 | -8.20313 | 9.20574 | -0.209686 | 0.355942 | 0.869386 | 15073 | 15073 | 13573 | 1500 |
| val | gyro_x | -0.00131147 | 1.17436 | -18.9812 | 17.5454 | -0.172721 | 0.00356407 | 0.207723 | 15073 | 15073 | 13573 | 1500 |
### `data/processed/reports/eda_acceleration_spikes.csv`

- Shape: `6 rows x 9 columns`
- Missing cells: `0`
- Columns: `split, class_name, windows, peak_mean, peak_median, peak_std, peak_p95, peak_max, magnitude_mean`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| windows | 6 | 1200 | 14020.5 | 8455.5 | 18856.2 | 50939 |
| peak_mean | 6 | 2.08205 | 4.5393 | 4.40994 | 2.6668 | 7.16641 |
| peak_median | 6 | 1.84132 | 4.34407 | 4.23187 | 2.70943 | 7.0617 |
| peak_std | 6 | 0.964197 | 1.32677 | 1.26785 | 0.361296 | 1.84286 |
| peak_p95 | 6 | 3.59748 | 6.89294 | 6.75199 | 3.42116 | 10.4178 |
| peak_max | 6 | 7.68074 | 10.7937 | 10.5206 | 2.99567 | 15.3133 |
| magnitude_mean | 6 | 1.36619 | 1.7886 | 1.81179 | 0.420321 | 2.17611 |

First 10 rows:

| split | class_name | windows | peak_mean | peak_median | peak_std | peak_p95 | peak_max | magnitude_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | ADL | 50939 | 2.09587 | 1.88398 | 1.02748 | 3.76154 | 9.26133 | 1.36619 |
| train | Fall | 4484 | 7.16641 | 7.0617 | 1.84286 | 10.4178 | 15.3133 | 2.17172 |
| val | ADL | 13573 | 2.15129 | 1.89952 | 1.05707 | 3.99028 | 8.01174 | 1.4574 |
| val | Fall | 1500 | 7.07161 | 6.81366 | 1.59038 | 10.0768 | 12.7154 | 2.17611 |
| test | ADL | 12427 | 2.08205 | 1.84132 | 0.964197 | 3.59748 | 7.68074 | 1.39398 |
| test | Fall | 1200 | 6.66859 | 6.56422 | 1.47864 | 9.51371 | 11.7798 | 2.16618 |
### `data/processed/reports/eda_autocorrelation.csv`

- Shape: `594 rows x 4 columns`
- Missing cells: `0`
- Columns: `split, channel, lag, autocorrelation`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| lag | 594 | 0 | 16 | 16 | 9.52993 | 32 |
| autocorrelation | 594 | -0.129891 | 0.0565435 | -0.0362337 | 0.240654 | 1 |

First 10 rows:

| split | channel | lag | autocorrelation |
| --- | --- | --- | --- |
| train | acc1_x | 0 | 1 |
| train | acc1_x | 1 | 0.698383 |
| train | acc1_x | 2 | 0.319105 |
| train | acc1_x | 3 | 0.217049 |
| train | acc1_x | 4 | 0.222961 |
| train | acc1_x | 5 | 0.211051 |
| train | acc1_x | 6 | 0.162659 |
| train | acc1_x | 7 | 0.0943122 |
| train | acc1_x | 8 | 0.0706545 |
| train | acc1_x | 9 | 0.0665171 |
### `data/processed/reports/eda_class_distribution.csv`

- Shape: `6 rows x 6 columns`
- Missing cells: `0`
- Columns: `split, class_label, class_name, window_count, percentage, total_windows`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| class_label | 6 | 0 | 0.5 | 0.5 | 0.547723 | 1 |
| window_count | 6 | 1200 | 14020.5 | 8455.5 | 18856.2 | 50939 |
| percentage | 6 | 8.0905 | 50 | 50 | 44.9765 | 91.9095 |
| total_windows | 6 | 13627 | 28041 | 15073 | 21219.9 | 55423 |

First 10 rows:

| split | class_label | class_name | window_count | percentage | total_windows |
| --- | --- | --- | --- | --- | --- |
| train | 0 | ADL | 50939 | 91.9095 | 55423 |
| train | 1 | Fall | 4484 | 8.0905 | 55423 |
| val | 0 | ADL | 13573 | 90.0484 | 15073 |
| val | 1 | Fall | 1500 | 9.95157 | 15073 |
| test | 0 | ADL | 12427 | 91.194 | 13627 |
| test | 1 | Fall | 1200 | 8.80605 | 13627 |
### `data/processed/reports/eda_outliers.csv`

- Shape: `18 rows x 10 columns`
- Missing cells: `0`
- Columns: `split, channel, sample_count, iqr_outliers, iqr_outlier_percentage, iqr_lower_bound, iqr_upper_bound, zscore_outliers, zscore_outlier_percentage, zscore_threshold`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| sample_count | 18 | 872128 | 1.79462e+06 | 964672 | 1.27569e+06 | 3.54707e+06 |
| iqr_outliers | 18 | 8108 | 302079 | 227674 | 301372 | 1.01039e+06 |
| iqr_outlier_percentage | 18 | 0.840493 | 17.2336 | 19.111 | 11.3363 | 32.5828 |
| iqr_lower_bound | 18 | -2.84351 | -1.26105 | -0.84897 | 0.828493 | -0.445259 |
| iqr_upper_bound | 18 | 0.459089 | 1.35629 | 0.911229 | 0.947594 | 3.22714 |
| zscore_outliers | 18 | 1391 | 21674.6 | 18044 | 23231 | 77014 |
| zscore_outlier_percentage | 18 | 0.159495 | 1.24592 | 1.05996 | 0.914204 | 2.43852 |
| zscore_threshold | 18 | 3 | 3 | 3 | 0 | 3 |

First 10 rows:

| split | channel | sample_count | iqr_outliers | iqr_outlier_percentage | iqr_lower_bound | iqr_upper_bound | zscore_outliers | zscore_outlier_percentage | zscore_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | acc1_x | 3547072 | 608827 | 17.1642 | -0.867586 | 0.914448 | 15934 | 0.449216 | 3 |
| train | acc1_y | 3547072 | 44835 | 1.264 | -2.61113 | 2.87441 | 21348 | 0.601849 | 3 |
| train | acc1_z | 3547072 | 242806 | 6.84525 | -2.12751 | 1.97796 | 6566 | 0.18511 | 3 |
| train | gyro_x | 3547072 | 678660 | 19.133 | -0.824552 | 0.862064 | 77014 | 2.1712 | 3 |
| train | gyro_y | 3547072 | 904997 | 25.5139 | -0.663064 | 0.669512 | 75297 | 2.12279 | 3 |
| train | gyro_z | 3547072 | 1010394 | 28.4853 | -0.542467 | 0.539242 | 52181 | 1.4711 | 3 |
| val | acc1_x | 964672 | 184146 | 19.089 | -1.03188 | 0.937612 | 3332 | 0.345402 | 3 |
| val | acc1_y | 964672 | 8108 | 0.840493 | -2.84351 | 3.22714 | 6259 | 0.648822 | 3 |
| val | acc1_z | 964672 | 44289 | 4.59109 | -1.82829 | 2.48799 | 2200 | 0.228057 | 3 |
| val | gyro_x | 964672 | 212543 | 22.0327 | -0.743388 | 0.77839 | 21805 | 2.26035 | 3 |
### `data/processed/reports/eda_signal_health_statistics.csv`

- Shape: `54 rows x 13 columns`
- Missing cells: `0`
- Columns: `gibesplit, class_name, channel, mean, median, std, variance, rms, min, max, skewness, kurtosis, sample_count`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| mean | 54 | -0.187893 | 0.0460568 | 0.000120643 | 0.188245 | 0.714509 |
| median | 54 | -0.449582 | 0.0235487 | 0.00219971 | 0.268481 | 0.926878 |
| std | 54 | 0.900062 | 1.258 | 1.05912 | 0.421572 | 2.34148 |
| variance | 54 | 0.810112 | 1.757 | 1.12184 | 1.31148 | 5.48253 |
| rms | 54 | 0.900091 | 1.27294 | 1.06064 | 0.420471 | 2.34305 |
| min | 54 | -48.5212 | -15.6077 | -11.7561 | 11.2308 | -5.13218 |
| max | 54 | 4.06029 | 14.4481 | 11.666 | 9.00355 | 38.1595 |
| skewness | 54 | -1.33691 | 0.0129641 | -0.0177335 | 0.372372 | 0.992024 |
| kurtosis | 54 | 0.086231 | 14.4383 | 7.13257 | 25.9191 | 155.066 |
| sample_count | 54 | 76800 | 1.19642e+06 | 868672 | 1.23638e+06 | 3.54707e+06 |

First 10 rows:

| gibesplit | class_name | channel | mean | median | std | variance | rms | min | max | skewness | kurtosis | sample_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | All | acc1_x | 0.000451951 | 0.0297179 | 1.00911 | 1.01829 | 1.00911 | -13.8934 | 11.6163 | -0.172358 | 5.89572 | 3547072 |
| train | All | acc1_y | -0.0120695 | -0.377854 | 1.0142 | 1.0286 | 1.01427 | -6.89975 | 6.94432 | 0.202759 | 0.725326 | 3547072 |
| train | All | acc1_z | -0.0322971 | -0.0817941 | 0.989853 | 0.979808 | 0.990379 | -10.3763 | 8.83929 | 0.314065 | 1.22191 | 3547072 |
| train | All | gyro_x | 0.000523961 | 6.42881e-05 | 1.07316 | 1.15167 | 1.07316 | -36.8906 | 38.1595 | 0.450478 | 38.5635 | 3547072 |
| train | All | gyro_y | -0.00172154 | 0.00198487 | 1.06964 | 1.14412 | 1.06964 | -29.6888 | 27.5684 | -0.145794 | 37.5338 | 3547072 |
| train | All | gyro_z | -0.000210666 | 0.00219938 | 1.07406 | 1.15361 | 1.07406 | -38.0956 | 37.6419 | -0.510598 | 101.801 | 3547072 |
| train | ADL | acc1_x | 0.0102339 | 0.0317026 | 0.932267 | 0.869121 | 0.932323 | -9.02316 | 6.12925 | -0.0609529 | 3.86334 | 3260096 |
| train | ADL | acc1_y | -0.0760279 | -0.408438 | 0.977297 | 0.955109 | 0.980249 | -5.59689 | 5.42834 | 0.175544 | 0.766639 | 3260096 |
| train | ADL | acc1_z | -0.0294212 | -0.0773949 | 0.941856 | 0.887093 | 0.942315 | -6.59289 | 5.82372 | 0.404459 | 0.614009 | 3260096 |
| train | ADL | gyro_x | 0.00723441 | 0.00125309 | 0.900062 | 0.810112 | 0.900091 | -36.8906 | 38.1595 | -0.0575259 | 35.124 | 3260096 |
### `data/processed/severity/runs/k2_20260914T055946Z/anomaly_by_severity.csv`

- Shape: `6 rows x 6 columns`
- Missing cells: `0`
- Columns: `split, severity_label, severity_name, fall_windows, anomalous_windows, anomaly_rate`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| severity_label | 6 | 0 | 0.5 | 0.5 | 0.547723 | 1 |
| fall_windows | 6 | 468 | 1197.33 | 834.5 | 893.904 | 2787 |
| anomalous_windows | 6 | 5 | 22.8333 | 15.5 | 17.7135 | 47 |
| anomaly_rate | 6 | 0.00888099 | 0.018593 | 0.0184849 | 0.00579681 | 0.0253388 |

First 10 rows:

| split | severity_label | severity_name | fall_windows | anomalous_windows | anomaly_rate |
| --- | --- | --- | --- | --- | --- |
| train | 0 | Mild | 1697 | 43 | 0.0253388 |
| train | 1 | Severe | 2787 | 47 | 0.016864 |
| val | 0 | Mild | 563 | 5 | 0.00888099 |
| val | 1 | Severe | 937 | 18 | 0.0192102 |
| test | 0 | Mild | 468 | 11 | 0.0235043 |
| test | 1 | Severe | 732 | 13 | 0.0177596 |
### `data/processed/severity/runs/k2_20260914T055946Z/anomaly_by_subject.csv`

- Shape: `24 rows x 5 columns`
- Missing cells: `0`
- Columns: `split, subject_id, fall_windows, anomalous_windows, anomaly_rate`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| fall_windows | 24 | 292 | 299.333 | 300 | 1.78561 | 300 |
| anomalous_windows | 24 | 1 | 5.70833 | 3.5 | 5.1876 | 21 |
| anomaly_rate | 24 | 0.00333333 | 0.0190636 | 0.0116667 | 0.0172965 | 0.07 |

First 10 rows:

| split | subject_id | fall_windows | anomalous_windows | anomaly_rate |
| --- | --- | --- | --- | --- |
| train | SA01 | 300 | 12 | 0.04 |
| train | SA04 | 300 | 8 | 0.0266667 |
| train | SA05 | 300 | 3 | 0.01 |
| train | SA06 | 299 | 12 | 0.0401338 |
| train | SA07 | 299 | 2 | 0.00668896 |
| train | SA08 | 300 | 2 | 0.00666667 |
| train | SA10 | 299 | 10 | 0.0334448 |
| train | SA11 | 300 | 21 | 0.07 |
| train | SA16 | 300 | 1 | 0.00333333 |
| train | SA17 | 300 | 3 | 0.01 |
### `data/processed/severity/runs/k2_20260914T055946Z/anomaly_summary.csv`

- Shape: `3 rows x 4 columns`
- Missing cells: `0`
- Columns: `split, fall_windows, anomalous_windows, anomaly_rate`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| fall_windows | 3 | 1200 | 2394.67 | 1500 | 1815.62 | 4484 |
| anomalous_windows | 3 | 23 | 45.6667 | 24 | 38.397 | 90 |
| anomaly_rate | 3 | 0.0153333 | 0.0184682 | 0.02 | 0.00271514 | 0.0200714 |

First 10 rows:

| split | fall_windows | anomalous_windows | anomaly_rate |
| --- | --- | --- | --- |
| train | 4484 | 90 | 0.0200714 |
| val | 1500 | 23 | 0.0153333 |
| test | 1200 | 24 | 0.02 |
### `data/processed/severity/runs/k2_20260914T055946Z/cluster_statistics.csv`

- Shape: `2 rows x 14 columns`
- Missing cells: `0`
- Columns: `cluster_id, train_count, val_count, test_count, total_count, acc_mag_mean, acc_mag_rms_mean, acc_mag_peak_mean, gyro_mag_mean, gyro_mag_rms_mean, gyro_mag_peak_mean, cluster_intensity_score, severity_label, severity_name`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| cluster_id | 2 | 0 | 0.5 | 0.5 | 0.707107 | 1 |
| train_count | 2 | 1697 | 2242 | 2242 | 770.746 | 2787 |
| val_count | 2 | 563 | 750 | 750 | 264.458 | 937 |
| test_count | 2 | 468 | 600 | 600 | 186.676 | 732 |
| total_count | 2 | 2728 | 3592 | 3592 | 1221.88 | 4456 |
| acc_mag_mean | 2 | 2.04996 | 2.14791 | 2.14791 | 0.138521 | 2.24586 |
| acc_mag_rms_mean | 2 | 2.33736 | 2.47483 | 2.47483 | 0.194404 | 2.61229 |
| acc_mag_peak_mean | 2 | 6.39667 | 7.01589 | 7.01589 | 0.875704 | 7.63511 |
| gyro_mag_mean | 2 | 1.90934 | 2.16416 | 2.16416 | 0.360364 | 2.41897 |
| gyro_mag_rms_mean | 2 | 3.04433 | 3.40727 | 3.40727 | 0.513262 | 3.7702 |
| gyro_mag_peak_mean | 2 | 10.1551 | 11.4295 | 11.4295 | 1.80228 | 12.7039 |
| cluster_intensity_score | 2 | 16.5517 | 18.4454 | 18.4454 | 2.67799 | 20.339 |
| severity_label | 2 | 0 | 0.5 | 0.5 | 0.707107 | 1 |

First 10 rows:

| cluster_id | train_count | val_count | test_count | total_count | acc_mag_mean | acc_mag_rms_mean | acc_mag_peak_mean | gyro_mag_mean | gyro_mag_rms_mean | gyro_mag_peak_mean | cluster_intensity_score | severity_label | severity_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1697 | 563 | 468 | 2728 | 2.04996 | 2.33736 | 6.39667 | 1.90934 | 3.04433 | 10.1551 | 16.5517 | 0 | Mild |
| 1 | 2787 | 937 | 732 | 4456 | 2.24586 | 2.61229 | 7.63511 | 2.41897 | 3.7702 | 12.7039 | 20.339 | 1 | Severe |
### `data/processed/severity/runs/k2_20260914T055946Z/reports/anomalous_feature_statistics.csv`

- Shape: `6 rows x 3 columns`
- Missing cells: `0`
- Columns: `split, status, count`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| count | 6 | 23 | 1197.33 | 633 | 1689.93 | 4394 |

First 10 rows:

| split | status | count |
| --- | --- | --- |
| train | normal | 4394 |
| train | anomalous | 90 |
| val | normal | 1477 |
| val | anomalous | 23 |
| test | normal | 1176 |
| test | anomalous | 24 |
### `data/processed/severity/runs/k2_20260914T055946Z/reports/anomaly_by_severity_report.csv`

- Shape: `6 rows x 6 columns`
- Missing cells: `0`
- Columns: `split, severity_label, severity_name, fall_windows, anomalous_windows, anomaly_rate`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| severity_label | 6 | 0 | 0.5 | 0.5 | 0.547723 | 1 |
| fall_windows | 6 | 468 | 1197.33 | 834.5 | 893.904 | 2787 |
| anomalous_windows | 6 | 5 | 22.8333 | 15.5 | 17.7135 | 47 |
| anomaly_rate | 6 | 0.00888099 | 0.018593 | 0.0184849 | 0.00579681 | 0.0253388 |

First 10 rows:

| split | severity_label | severity_name | fall_windows | anomalous_windows | anomaly_rate |
| --- | --- | --- | --- | --- | --- |
| train | 0 | Mild | 1697 | 43 | 0.0253388 |
| train | 1 | Severe | 2787 | 47 | 0.016864 |
| val | 0 | Mild | 563 | 5 | 0.00888099 |
| val | 1 | Severe | 937 | 18 | 0.0192102 |
| test | 0 | Mild | 468 | 11 | 0.0235043 |
| test | 1 | Severe | 732 | 13 | 0.0177596 |
### `data/processed/severity/runs/k2_20260914T055946Z/reports/anomaly_by_subject_report.csv`

- Shape: `24 rows x 5 columns`
- Missing cells: `0`
- Columns: `split, subject_id, fall_windows, anomalous_windows, anomaly_rate`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| fall_windows | 24 | 292 | 299.333 | 300 | 1.78561 | 300 |
| anomalous_windows | 24 | 1 | 5.70833 | 3.5 | 5.1876 | 21 |
| anomaly_rate | 24 | 0.00333333 | 0.0190636 | 0.0116667 | 0.0172965 | 0.07 |

First 10 rows:

| split | subject_id | fall_windows | anomalous_windows | anomaly_rate |
| --- | --- | --- | --- | --- |
| train | SA01 | 300 | 12 | 0.04 |
| train | SA04 | 300 | 8 | 0.0266667 |
| train | SA05 | 300 | 3 | 0.01 |
| train | SA06 | 299 | 12 | 0.0401338 |
| train | SA07 | 299 | 2 | 0.00668896 |
| train | SA08 | 300 | 2 | 0.00666667 |
| train | SA10 | 299 | 10 | 0.0334448 |
| train | SA11 | 300 | 21 | 0.07 |
| train | SA16 | 300 | 1 | 0.00333333 |
| train | SA17 | 300 | 3 | 0.01 |
### `data/processed/severity/runs/k2_20260914T055946Z/reports/anomaly_summary_report.csv`

- Shape: `3 rows x 4 columns`
- Missing cells: `0`
- Columns: `split, fall_windows, anomalous_windows, anomaly_rate`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| fall_windows | 3 | 1200 | 2394.67 | 1500 | 1815.62 | 4484 |
| anomalous_windows | 3 | 23 | 45.6667 | 24 | 38.397 | 90 |
| anomaly_rate | 3 | 0.0153333 | 0.0184682 | 0.02 | 0.00271514 | 0.0200714 |

First 10 rows:

| split | fall_windows | anomalous_windows | anomaly_rate |
| --- | --- | --- | --- |
| train | 4484 | 90 | 0.0200714 |
| val | 1500 | 23 | 0.0153333 |
| test | 1200 | 24 | 0.02 |
### `data/processed/severity/runs/k2_20260914T055946Z/reports/cluster_feature_report.csv`

- Shape: `6 rows x 13 columns`
- Missing cells: `0`
- Columns: `split, cluster_id, severity_label, severity_name, count, acc_mag_peak_mean, acc_mag_peak_std, gyro_mag_peak_mean, gyro_mag_peak_std, acc_mag_rms_mean, acc_mag_rms_std, gyro_mag_rms_mean, gyro_mag_rms_std`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| cluster_id | 6 | 0 | 0.5 | 0.5 | 0.547723 | 1 |
| severity_label | 6 | 0 | 0.5 | 0.5 | 0.547723 | 1 |
| count | 6 | 468 | 1197.33 | 834.5 | 893.904 | 2787 |
| acc_mag_peak_mean | 6 | 6.32927 | 6.86178 | 6.70411 | 0.548575 | 7.63511 |
| acc_mag_peak_std | 6 | 1.35625 | 1.56297 | 1.60197 | 0.170896 | 1.79457 |
| gyro_mag_peak_mean | 6 | 9.83922 | 11.3195 | 11.282 | 1.36017 | 12.7039 |
| gyro_mag_peak_std | 6 | 2.40716 | 3.36371 | 3.33773 | 0.675133 | 4.27923 |
| acc_mag_rms_mean | 6 | 2.32863 | 2.47215 | 2.46299 | 0.136142 | 2.62864 |
| acc_mag_rms_std | 6 | 0.470104 | 0.495941 | 0.498416 | 0.0152783 | 0.515789 |
| gyro_mag_rms_mean | 6 | 3.02488 | 3.44211 | 3.41222 | 0.442268 | 3.92943 |
| gyro_mag_rms_std | 6 | 0.647976 | 0.782749 | 0.77085 | 0.122793 | 0.955065 |

First 10 rows:

| split | cluster_id | severity_label | severity_name | count | acc_mag_peak_mean | acc_mag_peak_std | gyro_mag_peak_mean | gyro_mag_peak_std | acc_mag_rms_mean | acc_mag_rms_std | gyro_mag_rms_mean | gyro_mag_rms_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | Mild | 1697 | 6.39667 | 1.65265 | 10.1551 | 3.65488 | 2.33736 | 0.470104 | 3.04433 | 0.83982 |
| train | 1 | 1 | Severe | 2787 | 7.63511 | 1.79457 | 12.7039 | 3.02058 | 2.61229 | 0.498995 | 3.7702 | 0.701879 |
| val | 0 | 0 | Mild | 563 | 6.5227 | 1.35625 | 10.2819 | 3.01081 | 2.32863 | 0.489672 | 3.05425 | 0.681957 |
| val | 1 | 1 | Severe | 937 | 7.40142 | 1.62912 | 12.6549 | 2.40716 | 2.62864 | 0.515789 | 3.82959 | 0.647976 |
| test | 0 | 0 | Mild | 468 | 6.32927 | 1.57482 | 9.83922 | 4.27923 | 2.39367 | 0.50325 | 3.02488 | 0.955065 |
| test | 1 | 1 | Severe | 732 | 6.88552 | 1.37038 | 12.2821 | 3.80957 | 2.53231 | 0.497837 | 3.92943 | 0.869798 |
### `data/processed/severity/runs/k2_20260914T055946Z/reports/post_isolation_forest_all_windows.csv`

- Shape: `84,123 rows x 16 columns`
- Missing cells: `307,756`
- Columns: `split, window_index, subject_id, recording_id, class_label, class_name, is_fall_window, fall_window_order, cluster_id, severity_label, severity_class, anomaly_score, anomaly_flag, anomaly_marking, refined_severity_label, refined_severity_class`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| window_index | 84123 | 0 | 20710.8 | 14217 | 16408 | 55422 |
| class_label | 84123 | 0 | 0.0853988 | 0 | 0.279476 | 1 |
| fall_window_order | 7184 | 0 | 1655.7 | 1197 | 1295.08 | 4483 |
| cluster_id | 7184 | 0 | 0.620267 | 1 | 0.485354 | 1 |
| severity_label | 84123 | -1 | -0.861631 | -1 | 0.474516 | 1 |
| anomaly_score | 7184 | -0.0730676 | 0.0793206 | 0.084564 | 0.0311061 | 0.142574 |
| anomaly_flag | 7184 | 0 | 0.0190702 | 0 | 0.136781 | 1 |
| refined_severity_label | 84123 | -1 | -0.864187 | -1 | 0.470591 | 1 |

First 10 rows:

| split | window_index | subject_id | recording_id | class_label | class_name | is_fall_window | fall_window_order | cluster_id | severity_label | severity_class | anomaly_score | anomaly_flag | anomaly_marking | refined_severity_label | refined_severity_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable |  |  | Not evaluated | -1 | Non-applicable |
| train | 1 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable |  |  | Not evaluated | -1 | Non-applicable |
| train | 2 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable |  |  | Not evaluated | -1 | Non-applicable |
| train | 3 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable |  |  | Not evaluated | -1 | Non-applicable |
| train | 4 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable |  |  | Not evaluated | -1 | Non-applicable |
| train | 5 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable |  |  | Not evaluated | -1 | Non-applicable |
| train | 6 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable |  |  | Not evaluated | -1 | Non-applicable |
| train | 7 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable |  |  | Not evaluated | -1 | Non-applicable |
| train | 8 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable |  |  | Not evaluated | -1 | Non-applicable |
| train | 9 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable |  |  | Not evaluated | -1 | Non-applicable |
### `data/processed/severity/runs/k2_20260914T055946Z/reports/post_isolation_forest_fall_windows.csv`

- Shape: `7,184 rows x 16 columns`
- Missing cells: `0`
- Columns: `split, window_index, fall_window_order, subject_id, recording_id, class_label, class_name, cluster_id, cluster_mapped_severity_label, severity_label, severity_class, anomaly_score, anomaly_flag, anomaly_marking, refined_severity_label, refined_severity_class`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| window_index | 7184 | 1633 | 16282.6 | 11002 | 12473.6 | 47694 |
| fall_window_order | 7184 | 0 | 1655.7 | 1197 | 1295.08 | 4483 |
| class_label | 7184 | 1 | 1 | 1 | 0 | 1 |
| cluster_id | 7184 | 0 | 0.620267 | 1 | 0.485354 | 1 |
| cluster_mapped_severity_label | 7184 | 0 | 0.620267 | 1 | 0.485354 | 1 |
| severity_label | 7184 | 0 | 0.620267 | 1 | 0.485354 | 1 |
| anomaly_score | 7184 | -0.0730676 | 0.0793206 | 0.084564 | 0.0311061 | 0.142574 |
| anomaly_flag | 7184 | 0 | 0.0190702 | 0 | 0.136781 | 1 |
| refined_severity_label | 7184 | -1 | 0.59034 | 1 | 0.529167 | 1 |

First 10 rows:

| split | window_index | fall_window_order | subject_id | recording_id | class_label | class_name | cluster_id | cluster_mapped_severity_label | severity_label | severity_class | anomaly_score | anomaly_flag | anomaly_marking | refined_severity_label | refined_severity_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1633 | 0 | SA01 | SA01:F01_SA01_R01.txt | 1 | Fall | 1 | 1 | 1 | Severe | 0.0274336 | 0 | Normal | 1 | Severe |
| train | 1634 | 1 | SA01 | SA01:F01_SA01_R01.txt | 1 | Fall | 1 | 1 | 1 | Severe | -0.0653269 | 1 | Anomalous | -1 | Non-applicable |
| train | 1635 | 2 | SA01 | SA01:F01_SA01_R01.txt | 1 | Fall | 1 | 1 | 1 | Severe | -0.0656995 | 1 | Anomalous | -1 | Non-applicable |
| train | 1636 | 3 | SA01 | SA01:F01_SA01_R01.txt | 1 | Fall | 1 | 1 | 1 | Severe | -0.0730676 | 1 | Anomalous | -1 | Non-applicable |
| train | 1649 | 4 | SA01 | SA01:F01_SA01_R02.txt | 1 | Fall | 0 | 0 | 0 | Mild | 0.0967213 | 0 | Normal | 0 | Mild |
| train | 1650 | 5 | SA01 | SA01:F01_SA01_R02.txt | 1 | Fall | 0 | 0 | 0 | Mild | 0.111812 | 0 | Normal | 0 | Mild |
| train | 1651 | 6 | SA01 | SA01:F01_SA01_R02.txt | 1 | Fall | 0 | 0 | 0 | Mild | 0.106299 | 0 | Normal | 0 | Mild |
| train | 1652 | 7 | SA01 | SA01:F01_SA01_R02.txt | 1 | Fall | 0 | 0 | 0 | Mild | 0.0389802 | 0 | Normal | 0 | Mild |
| train | 1665 | 8 | SA01 | SA01:F01_SA01_R03.txt | 1 | Fall | 0 | 0 | 0 | Mild | 0.0906103 | 0 | Normal | 0 | Mild |
| train | 1666 | 9 | SA01 | SA01:F01_SA01_R03.txt | 1 | Fall | 0 | 0 | 0 | Mild | 0.0981735 | 0 | Normal | 0 | Mild |
### `data/processed/severity/runs/k2_20260914T055946Z/reports/post_isolation_forest_window_summary.csv`

- Shape: `12 rows x 4 columns`
- Missing cells: `0`
- Columns: `split, severity_class, anomaly_marking, window_count`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| window_count | 12 | 5 | 598.667 | 252 | 843.635 | 2740 |

First 10 rows:

| split | severity_class | anomaly_marking | window_count |
| --- | --- | --- | --- |
| test | Mild | Anomalous | 11 |
| test | Mild | Normal | 457 |
| test | Severe | Anomalous | 13 |
| test | Severe | Normal | 719 |
| train | Mild | Anomalous | 43 |
| train | Mild | Normal | 1654 |
| train | Severe | Anomalous | 47 |
| train | Severe | Normal | 2740 |
| val | Mild | Anomalous | 5 |
| val | Mild | Normal | 558 |
### `data/processed/severity/runs/k2_20260914T055946Z/severity_cluster_profiles.csv`

- Shape: `2 rows x 11 columns`
- Missing cells: `8`
- Columns: `cluster_id, size, peak_accel_mean, peak_accel_median, acc_rms_mean, gyro_peak_mean, gyro_rms_mean, energy_mean, delta_velocity_mean, sma_mean, jerk_mean`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| cluster_id | 2 | 0 | 0.5 | 0.5 | 0.707107 | 1 |
| size | 2 | 1697 | 2242 | 2242 | 770.746 | 2787 |
| peak_accel_mean | 2 | 6.39667 | 7.01589 | 7.01589 | 0.875704 | 7.63511 |
| peak_accel_median | 2 | 6.42979 | 6.96732 | 6.96732 | 0.760173 | 7.50484 |
| acc_rms_mean | 2 | 2.33736 | 2.47483 | 2.47483 | 0.194404 | 2.61229 |
| gyro_peak_mean | 2 | 10.1551 | 11.4295 | 11.4295 | 1.80228 | 12.7039 |
| gyro_rms_mean | 2 | 3.04433 | 3.40727 | 3.40727 | 0.513262 | 3.7702 |
| energy_mean | 0 |  |  |  |  |  |
| delta_velocity_mean | 0 |  |  |  |  |  |
| sma_mean | 0 |  |  |  |  |  |
| jerk_mean | 0 |  |  |  |  |  |

First 10 rows:

| cluster_id | size | peak_accel_mean | peak_accel_median | acc_rms_mean | gyro_peak_mean | gyro_rms_mean | energy_mean | delta_velocity_mean | sma_mean | jerk_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1697 | 6.39667 | 6.42979 | 2.33736 | 10.1551 | 3.04433 |  |  |  |  |
| 1 | 2787 | 7.63511 | 7.50484 | 2.61229 | 12.7039 | 3.7702 |  |  |  |  |
### `data/processed/severity/runs/k2_20260914T055946Z/severity_cluster_sizes.csv`

- Shape: `2 rows x 3 columns`
- Missing cells: `0`
- Columns: `cluster_id, cluster_size, cluster_percentage`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| cluster_id | 2 | 0 | 0.5 | 0.5 | 0.707107 | 1 |
| cluster_size | 2 | 1697 | 2242 | 2242 | 770.746 | 2787 |
| cluster_percentage | 2 | 0.378457 | 0.5 | 0.5 | 0.171888 | 0.621543 |

First 10 rows:

| cluster_id | cluster_size | cluster_percentage |
| --- | --- | --- |
| 0 | 1697 | 0.378457 |
| 1 | 2787 | 0.621543 |
### `data/processed/severity/runs/k2_20260914T055946Z/severity_k_comparison.csv`

- Shape: `4 rows x 11 columns`
- Missing cells: `0`
- Columns: `k, silhouette, davies_bouldin, calinski_harabasz, silhouette_std, davies_bouldin_std, calinski_harabasz_std, mean_inertia, min_cluster_pct, max_cluster_pct, cluster_size_range`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 4 | 2 | 3.5 | 3.5 | 1.29099 | 5 |
| silhouette | 4 | 0.158482 | 0.167195 | 0.166459 | 0.00795482 | 0.177382 |
| davies_bouldin | 4 | 1.87068 | 2.01183 | 2.01243 | 0.1162 | 2.15179 |
| calinski_harabasz | 4 | 657.654 | 787.934 | 747.133 | 149.613 | 999.814 |
| silhouette_std | 4 | 0 | 1.47074e-05 | 1.45894e-06 | 2.74816e-05 | 5.59119e-05 |
| davies_bouldin_std | 4 | 0 | 0.000866472 | 7.53795e-05 | 0.00163362 | 0.00331513 |
| calinski_harabasz_std | 4 | 0 | 0.00363828 | 0.000205427 | 0.00700333 | 0.0141423 |
| mean_inertia | 4 | 163843 | 186324 | 184407 | 21239.4 | 212638 |
| min_cluster_pct | 4 | 0.118421 | 0.241826 | 0.235169 | 0.113287 | 0.378546 |
| max_cluster_pct | 4 | 0.301918 | 0.404561 | 0.347435 | 0.147334 | 0.621454 |

First 10 rows:

| k | silhouette | davies_bouldin | calinski_harabasz | silhouette_std | davies_bouldin_std | calinski_harabasz_std | mean_inertia | min_cluster_pct | max_cluster_pct | cluster_size_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.177382 | 1.99011 | 999.814 | 9.33745e-07 | 0.000137985 | 0.000195193 | 212638 | 0.378546 | 0.621454 | [1697, 2787] |
| 3 | 0.158482 | 2.15179 | 778.411 | 0 | 0 | 0 | 193014 | 0.28256 | 0.369982 | [1267, 1659] |
| 4 | 0.164331 | 2.03475 | 715.856 | 1.98413e-06 | 1.27735e-05 | 0.000215661 | 175799 | 0.187779 | 0.324888 | [842, 1457] |
| 5 | 0.168586 | 1.87068 | 657.654 | 5.59119e-05 | 0.00331513 | 0.0141423 | 163843 | 0.118421 | 0.301918 | [526, 1361] |
### `data/processed/severity/runs/k2_20260914T055946Z/severity_k_stability.csv`

- Shape: `4 rows x 9 columns`
- Missing cells: `0`
- Columns: `k, mean_silhouette, std_silhouette, mean_davies_bouldin, std_davies_bouldin, mean_calinski_harabasz, std_calinski_harabasz, mean_cluster_size_min_pct, mean_cluster_size_max_pct`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 4 | 2 | 3.5 | 3.5 | 1.29099 | 5 |
| mean_silhouette | 4 | 0.158482 | 0.167195 | 0.166459 | 0.00795482 | 0.177382 |
| std_silhouette | 4 | 0 | 1.47074e-05 | 1.45894e-06 | 2.74816e-05 | 5.59119e-05 |
| mean_davies_bouldin | 4 | 1.87068 | 2.01183 | 2.01243 | 0.1162 | 2.15179 |
| std_davies_bouldin | 4 | 0 | 0.000866472 | 7.53795e-05 | 0.00163362 | 0.00331513 |
| mean_calinski_harabasz | 4 | 657.654 | 787.934 | 747.133 | 149.613 | 999.814 |
| std_calinski_harabasz | 4 | 0 | 0.00363828 | 0.000205427 | 0.00700333 | 0.0141423 |
| mean_cluster_size_min_pct | 4 | 0.118421 | 0.241826 | 0.235169 | 0.113287 | 0.378546 |
| mean_cluster_size_max_pct | 4 | 0.301918 | 0.404561 | 0.347435 | 0.147334 | 0.621454 |

First 10 rows:

| k | mean_silhouette | std_silhouette | mean_davies_bouldin | std_davies_bouldin | mean_calinski_harabasz | std_calinski_harabasz | mean_cluster_size_min_pct | mean_cluster_size_max_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.177382 | 9.33745e-07 | 1.99011 | 0.000137985 | 999.814 | 0.000195193 | 0.378546 | 0.621454 |
| 3 | 0.158482 | 0 | 2.15179 | 0 | 778.411 | 0 | 0.28256 | 0.369982 |
| 4 | 0.164331 | 1.98413e-06 | 2.03475 | 1.27735e-05 | 715.856 | 0.000215661 | 0.187779 | 0.324888 |
| 5 | 0.168586 | 5.59119e-05 | 1.87068 | 0.00331513 | 657.654 | 0.0141423 | 0.118421 | 0.301918 |
### `docs/reports/technical_appendix/tables/cnn_baseline_architecture.csv`

- Shape: `15 rows x 4 columns`
- Missing cells: `0`
- Columns: `layer, type, output_shape, params`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| params | 15 | 0 | 2248.6 | 33 | 5240.4 | 18528 |

First 10 rows:

| layer | type | output_shape | params |
| --- | --- | --- | --- |
| conv1d | Conv1D | (None, 64, 32) | 992 |
| batch_normalization | BatchNormalization | (None, 64, 32) | 128 |
| re_lu | ReLU | (None, 64, 32) | 0 |
| max_pooling1d | MaxPooling1D | (None, 32, 32) | 0 |
| conv1d_1 | Conv1D | (None, 32, 64) | 10304 |
| batch_normalization_1 | BatchNormalization | (None, 32, 64) | 256 |
| re_lu_1 | ReLU | (None, 32, 64) | 0 |
| max_pooling1d_1 | MaxPooling1D | (None, 16, 64) | 0 |
| conv1d_2 | Conv1D | (None, 16, 96) | 18528 |
| batch_normalization_2 | BatchNormalization | (None, 16, 96) | 384 |
### `docs/reports/technical_appendix/tables/cnn_baseline_test_metrics.csv`

- Shape: `1 rows x 5 columns`
- Missing cells: `0`
- Columns: `threshold, precision, recall, f1, confusion_matrix_TN_FP_FN_TP`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| threshold | 1 | 0.5 | 0.5 | 0.5 |  | 0.5 |
| precision | 1 | 0.9761 | 0.9761 | 0.9761 |  | 0.9761 |
| recall | 1 | 0.9867 | 0.9867 | 0.9867 |  | 0.9867 |
| f1 | 1 | 0.9814 | 0.9814 | 0.9814 |  | 0.9814 |

First 10 rows:

| threshold | precision | recall | f1 | confusion_matrix_TN_FP_FN_TP |
| --- | --- | --- | --- | --- |
| 0.5 | 0.9761 | 0.9867 | 0.9814 | [[12398, 29], [16, 1184]] |
### `docs/reports/technical_appendix/tables/cnn_lstm_architecture.csv`

- Shape: `13 rows x 4 columns`
- Missing cells: `0`
- Columns: `layer, type, output_shape, params`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| params | 13 | 0 | 3601.31 | 33 | 9280.57 | 33024 |

First 10 rows:

| layer | type | output_shape | params |
| --- | --- | --- | --- |
| conv1d_3 | Conv1D | (None, 64, 32) | 992 |
| batch_normalization_3 | BatchNormalization | (None, 64, 32) | 128 |
| re_lu_3 | ReLU | (None, 64, 32) | 0 |
| max_pooling1d_2 | MaxPooling1D | (None, 32, 32) | 0 |
| conv1d_4 | Conv1D | (None, 32, 64) | 10304 |
| batch_normalization_4 | BatchNormalization | (None, 32, 64) | 256 |
| re_lu_4 | ReLU | (None, 32, 64) | 0 |
| max_pooling1d_3 | MaxPooling1D | (None, 16, 64) | 0 |
| lstm | LSTM | (None, 16, 64) | 33024 |
| global_average_pooling1d_1 | GlobalAveragePooling1D | (None, 64) | 0 |
### `docs/reports/technical_appendix/tables/cnn_lstm_recording_level_eval.csv`

- Shape: `2 rows x 6 columns`
- Missing cells: `0`
- Columns: `threshold, total_fall_recordings, fall_recordings_detected, fall_recordings_fully_missed, recording_level_recall, adl_recordings_falsely_triggered`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| threshold | 2 | 0.25 | 0.375 | 0.375 | 0.176777 | 0.5 |
| total_fall_recordings | 2 | 300 | 300 | 300 | 0 | 300 |
| fall_recordings_detected | 2 | 300 | 300 | 300 | 0 | 300 |
| fall_recordings_fully_missed | 2 | 0 | 0 | 0 | 0 | 0 |
| recording_level_recall | 2 | 1 | 1 | 1 | 0 | 1 |
| adl_recordings_falsely_triggered | 2 | 0 | 0 | 0 | 0 | 0 |

First 10 rows:

| threshold | total_fall_recordings | fall_recordings_detected | fall_recordings_fully_missed | recording_level_recall | adl_recordings_falsely_triggered |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 300 | 300 | 0 | 1 | 0 |
| 0.25 | 300 | 300 | 0 | 1 | 0 |
### `docs/reports/technical_appendix/tables/cnn_lstm_test_metrics.csv`

- Shape: `2 rows x 5 columns`
- Missing cells: `0`
- Columns: `threshold, precision, recall, f1, confusion_matrix_TN_FP_FN_TP`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| threshold | 2 | 0.25 | 0.375 | 0.375 | 0.176777 | 0.5 |
| precision | 2 | 0.9606 | 0.9688 | 0.9688 | 0.0115966 | 0.977 |
| recall | 2 | 0.9925 | 0.99375 | 0.99375 | 0.00176777 | 0.995 |
| f1 | 2 | 0.9775 | 0.9811 | 0.9811 | 0.00509117 | 0.9847 |

First 10 rows:

| threshold | precision | recall | f1 | confusion_matrix_TN_FP_FN_TP |
| --- | --- | --- | --- | --- |
| 0.5 | 0.977 | 0.9925 | 0.9847 | [[12399, 28], [9, 1191]] |
| 0.25 | 0.9606 | 0.995 | 0.9775 | [[12378, 49], [6, 1194]] |
### `docs/reports/technical_appendix/tables/dataset_raw_sample.csv`

- Shape: `10 rows x 7 columns`
- Missing cells: `0`
- Columns: `row_index, acc1_x, acc1_y, acc1_z, gyro_x, gyro_y, gyro_z`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| row_index | 10 | 0 | 4.5 | 4.5 | 3.02765 | 9 |
| acc1_x | 10 | -55 | -22.1 | -28.5 | 26.6644 | 17 |
| acc1_y | 10 | -339 | -229 | -208 | 60.685 | -174 |
| acc1_z | 10 | -99 | -60.5 | -61 | 26.3492 | -19 |
| gyro_x | 10 | -214 | -131.3 | -137 | 66.55 | -18 |
| gyro_y | 10 | -764 | -668.3 | -687.5 | 85.3243 | -504 |
| gyro_z | 10 | -352 | -190.6 | -175 | 95.9643 | -72 |

First 10 rows:

| row_index | acc1_x | acc1_y | acc1_z | gyro_x | gyro_y | gyro_z |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 17 | -179 | -99 | -18 | -504 | -352 |
| 1 | 15 | -174 | -90 | -53 | -568 | -306 |
| 2 | 1 | -176 | -81 | -84 | -613 | -271 |
| 3 | -10 | -180 | -77 | -104 | -647 | -227 |
| 4 | -21 | -191 | -63 | -128 | -675 | -191 |
| 5 | -37 | -225 | -59 | -146 | -700 | -159 |
| 6 | -36 | -243 | -46 | -166 | -722 | -131 |
| 7 | -44 | -271 | -38 | -190 | -738 | -107 |
| 8 | -51 | -312 | -33 | -210 | -752 | -90 |
| 9 | -55 | -339 | -19 | -214 | -764 | -72 |
### `docs/reports/technical_appendix/tables/dataset_subject_split.csv`

- Shape: `3 rows x 6 columns`
- Missing cells: `0`
- Columns: `split, subjects, recordings, windows_total, windows_adl, windows_fall`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| subjects | 3 | 6 | 12.6667 | 6 | 11.547 | 26 |
| recordings | 3 | 734 | 1501.67 | 829 | 1248.27 | 2942 |
| windows_total | 3 | 13627 | 28041 | 15073 | 23724.5 | 55423 |
| windows_adl | 3 | 12427 | 25646.3 | 13573 | 21911.6 | 50939 |
| windows_fall | 3 | 1200 | 2394.67 | 1500 | 1815.62 | 4484 |

First 10 rows:

| split | subjects | recordings | windows_total | windows_adl | windows_fall |
| --- | --- | --- | --- | --- | --- |
| train | 26 | 2942 | 55423 | 50939 | 4484 |
| val | 6 | 829 | 15073 | 13573 | 1500 |
| test | 6 | 734 | 13627 | 12427 | 1200 |
### `docs/reports/technical_appendix/tables/impact_labeling_sample_rows.csv`

- Shape: `8 rows x 5 columns`
- Missing cells: `0`
- Columns: `window_id, subject_id, recording_id, peak_index_in_window, assigned_label`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| window_id | 8 | 1636 | 6042.12 | 6041 | 3302.76 | 10447 |
| peak_index_in_window | 8 | 17 | 39.875 | 40 | 18.5275 | 61 |
| assigned_label | 8 | 1 | 1 | 1 | 0 | 1 |

First 10 rows:

| window_id | subject_id | recording_id | peak_index_in_window | assigned_label |
| --- | --- | --- | --- | --- |
| 1636 | SA02 | SA02:F01_SA02_R01.txt | 60 | 1 |
| 2190 | SA02 | SA02:F08_SA02_R03.txt | 25 | 1 |
| 4392 | SA03 | SA03:F01_SA03_R01.txt | 52 | 1 |
| 4942 | SA03 | SA03:F08_SA03_R03.txt | 28 | 1 |
| 7140 | SA09 | SA09:F01_SA09_R01.txt | 61 | 1 |
| 7698 | SA09 | SA09:F08_SA09_R03.txt | 17 | 1 |
| 9892 | SA14 | SA14:F01_SA14_R01.txt | 54 | 1 |
| 10447 | SA14 | SA14:F08_SA14_R03.txt | 22 | 1 |
### `docs/reports/technical_appendix/tables/preprocessing_parameters.csv`

- Shape: `10 rows x 2 columns`
- Missing cells: `0`
- Columns: `parameter, value`

Numeric statistics:

_No rows._

First 10 rows:

| parameter | value |
| --- | --- |
| Original sampling rate | 200 Hz |
| Downsampled rate | 20 Hz |
| Butterworth filter type | low-pass |
| Butterworth filter order | 2 (as invoked; function default is 4, not used) |
| Butterworth cutoff frequency | 5 Hz |
| Channels retained | 6 (acc1_x/y/z, gyro_x/y/z) |
| Channels dropped | 3 (acc2_x/y/z, second accelerometer) |
| Scaler | StandardScaler, fit on train split only |
| Window size | 64 samples (3.2 s @ 20 Hz) |
| Window stride | 16 samples (0.8 s @ 20 Hz, 75% overlap) |
### `docs/reports/technical_appendix/tables/severity_cluster_anomaly_k2.csv`

- Shape: `6 rows x 5 columns`
- Missing cells: `0`
- Columns: `split, severity, cluster_size, anomalous_windows, anomaly_rate`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| cluster_size | 6 | 468 | 1197.33 | 834.5 | 893.904 | 2787 |
| anomalous_windows | 6 | 5 | 22.8333 | 15.5 | 17.7135 | 47 |
| anomaly_rate | 6 | 0.0089 | 0.0186 | 0.0185 | 0.0057772 | 0.0253 |

First 10 rows:

| split | severity | cluster_size | anomalous_windows | anomaly_rate |
| --- | --- | --- | --- | --- |
| train | Mild | 1697 | 43 | 0.0253 |
| train | Severe | 2787 | 47 | 0.0169 |
| val | Mild | 563 | 5 | 0.0089 |
| val | Severe | 937 | 18 | 0.0192 |
| test | Mild | 468 | 11 | 0.0235 |
| test | Severe | 732 | 13 | 0.0178 |
### `docs/reports/technical_appendix/tables/summary_cnn_vs_cnn_lstm.csv`

- Shape: `2 rows x 6 columns`
- Missing cells: `0`
- Columns: `model, threshold, precision, recall, f1, recording_level_recall`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| threshold | 2 | 0.5 | 0.5 | 0.5 | 0 | 0.5 |
| precision | 2 | 0.9761 | 0.97655 | 0.97655 | 0.000636396 | 0.977 |
| recall | 2 | 0.9867 | 0.9896 | 0.9896 | 0.00410122 | 0.9925 |
| f1 | 2 | 0.9814 | 0.98305 | 0.98305 | 0.00233345 | 0.9847 |
| recording_level_recall | 2 | 0.9967 | 0.99835 | 0.99835 | 0.00233345 | 1 |

First 10 rows:

| model | threshold | precision | recall | f1 | recording_level_recall |
| --- | --- | --- | --- | --- | --- |
| CNN baseline | 0.5 | 0.9761 | 0.9867 | 0.9814 | 0.9967 |
| CNN-LSTM (final) | 0.5 | 0.977 | 0.9925 | 0.9847 | 1 |
### `docs/reports/technical_appendix/tables/window_formation_counts.csv`

- Shape: `3 rows x 5 columns`
- Missing cells: `0`
- Columns: `split, tensor_shape, windows_total, windows_adl, windows_fall`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| windows_total | 3 | 13627 | 28041 | 15073 | 23724.5 | 55423 |
| windows_adl | 3 | 12427 | 25646.3 | 13573 | 21911.6 | 50939 |
| windows_fall | 3 | 1200 | 2394.67 | 1500 | 1815.62 | 4484 |

First 10 rows:

| split | tensor_shape | windows_total | windows_adl | windows_fall |
| --- | --- | --- | --- | --- |
| train | (64, 6) | 55423 | 50939 | 4484 |
| val | (64, 6) | 15073 | 13573 | 1500 |
| test | (64, 6) | 13627 | 12427 | 1200 |
### `results/cnn_baseline/recording_level_eval/false_trigger_only_recordings.csv`

- Shape: `0 rows x 4 columns`
- Missing cells: `0`
- Columns: `threshold, subject_id, recording_id, num_fall_windows`

Numeric statistics:

_No rows._

First 10 rows:

_No rows._
### `results/cnn_baseline/recording_level_eval/impact_window_verification.csv`

- Shape: `300 rows x 9 columns`
- Missing cells: `0`
- Columns: `subject_id, recording_id, num_fall_windows, predicted_positive_threshold_0_5, num_detected_fall_windows_threshold_0_5, detection_type_threshold_0_5, predicted_positive_threshold_0_2, num_detected_fall_windows_threshold_0_2, detection_type_threshold_0_2`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| num_fall_windows | 300 | 4 | 4 | 4 | 0 | 4 |
| predicted_positive_threshold_0_5 | 300 | 0 | 0.996667 | 1 | 0.057735 | 1 |
| num_detected_fall_windows_threshold_0_5 | 300 | 0 | 3.94667 | 4 | 0.301316 | 4 |
| predicted_positive_threshold_0_2 | 300 | 1 | 1 | 1 | 0 | 1 |
| num_detected_fall_windows_threshold_0_2 | 300 | 3 | 3.97 | 4 | 0.170872 | 4 |

First 10 rows:

| subject_id | recording_id | num_fall_windows | predicted_positive_threshold_0_5 | num_detected_fall_windows_threshold_0_5 | detection_type_threshold_0_5 | predicted_positive_threshold_0_2 | num_detected_fall_windows_threshold_0_2 | detection_type_threshold_0_2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SA02 | SA02:F01_SA02_R01.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
| SA02 | SA02:F01_SA02_R02.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
| SA02 | SA02:F01_SA02_R03.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
| SA02 | SA02:F01_SA02_R04.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
| SA02 | SA02:F01_SA02_R05.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
| SA02 | SA02:F02_SA02_R01.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
| SA02 | SA02:F02_SA02_R02.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
| SA02 | SA02:F02_SA02_R03.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
| SA02 | SA02:F02_SA02_R04.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
| SA02 | SA02:F02_SA02_R05.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
### `results/cnn_baseline/recording_level_eval/missed_fall_recordings.csv`

- Shape: `1 rows x 3 columns`
- Missing cells: `0`
- Columns: `subject_id, recording_id, num_fall_windows`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| num_fall_windows | 1 | 4 | 4 | 4 |  | 4 |

First 10 rows:

| subject_id | recording_id | num_fall_windows |
| --- | --- | --- |
| SA14 | SA14:F01_SA14_R02.txt | 4 |
### `results/cnn_baseline/recording_level_eval/recording_level_metrics.csv`

- Shape: `734 rows x 15 columns`
- Missing cells: `0`
- Columns: `recording_id, subject_id, true_label, num_windows, num_fall_windows, predicted_positive_threshold_0_5, num_detected_fall_windows_threshold_0_5, detected_fall_recording_threshold_0_5, fully_missed_fall_recording_threshold_0_5, false_positive_recording_threshold_0_5, predicted_positive_threshold_0_2, num_detected_fall_windows_threshold_0_2, detected_fall_recording_threshold_0_2, fully_missed_fall_recording_threshold_0_2, false_positive_recording_threshold_0_2`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| true_label | 734 | 0 | 0.408719 | 0 | 0.491932 | 1 |
| num_windows | 734 | 12 | 18.5654 | 15 | 19.6046 | 122 |
| num_fall_windows | 734 | 0 | 1.63488 | 0 | 1.96773 | 4 |
| predicted_positive_threshold_0_5 | 734 | 0 | 0.407357 | 0 | 0.491677 | 1 |
| num_detected_fall_windows_threshold_0_5 | 734 | 0 | 1.61308 | 0 | 1.95101 | 4 |
| detected_fall_recording_threshold_0_5 | 734 | 0 | 0.407357 | 0 | 0.491677 | 1 |
| fully_missed_fall_recording_threshold_0_5 | 734 | 0 | 0.0013624 | 0 | 0.0369107 | 1 |
| false_positive_recording_threshold_0_5 | 734 | 0 | 0 | 0 | 0 | 0 |
| predicted_positive_threshold_0_2 | 734 | 0 | 0.408719 | 0 | 0.491932 | 1 |
| num_detected_fall_windows_threshold_0_2 | 734 | 0 | 1.62262 | 0 | 1.95602 | 4 |
| detected_fall_recording_threshold_0_2 | 734 | 0 | 0.408719 | 0 | 0.491932 | 1 |
| fully_missed_fall_recording_threshold_0_2 | 734 | 0 | 0 | 0 | 0 | 0 |
| false_positive_recording_threshold_0_2 | 734 | 0 | 0 | 0 | 0 | 0 |

First 10 rows:

| recording_id | subject_id | true_label | num_windows | num_fall_windows | predicted_positive_threshold_0_5 | num_detected_fall_windows_threshold_0_5 | detected_fall_recording_threshold_0_5 | fully_missed_fall_recording_threshold_0_5 | false_positive_recording_threshold_0_5 | predicted_positive_threshold_0_2 | num_detected_fall_windows_threshold_0_2 | detected_fall_recording_threshold_0_2 | fully_missed_fall_recording_threshold_0_2 | false_positive_recording_threshold_0_2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SA02:D01_SA02_R01.txt | SA02 | 0 | 122 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SA02:D02_SA02_R01.txt | SA02 | 0 | 122 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SA02:D03_SA02_R01.txt | SA02 | 0 | 122 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SA02:D04_SA02_R01.txt | SA02 | 0 | 122 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SA02:D05_SA02_R01.txt | SA02 | 0 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SA02:D05_SA02_R02.txt | SA02 | 0 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SA02:D05_SA02_R03.txt | SA02 | 0 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SA02:D05_SA02_R04.txt | SA02 | 0 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SA02:D05_SA02_R05.txt | SA02 | 0 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SA02:D06_SA02_R01.txt | SA02 | 0 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
### `results/cnn_baseline/recording_level_eval/subject_level_metrics.csv`

- Shape: `6 rows x 12 columns`
- Missing cells: `4`
- Columns: `subject_id, total_recordings, fall_recordings, adl_recordings, fall_recordings_missed_threshold_0_5, fall_recordings_detected_threshold_0_5, subject_recall_threshold_0_5, review_below_90_percent_threshold_0_5, fall_recordings_missed_threshold_0_2, fall_recordings_detected_threshold_0_2, subject_recall_threshold_0_2, review_below_90_percent_threshold_0_2`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| total_recordings | 6 | 59 | 122.333 | 154 | 49.0578 | 154 |
| fall_recordings | 6 | 0 | 50 | 75 | 38.7298 | 75 |
| adl_recordings | 6 | 59 | 72.3333 | 79 | 10.328 | 79 |
| fall_recordings_missed_threshold_0_5 | 6 | 0 | 0.166667 | 0 | 0.408248 | 1 |
| fall_recordings_detected_threshold_0_5 | 6 | 0 | 49.8333 | 74.5 | 38.6027 | 75 |
| subject_recall_threshold_0_5 | 4 | 0.986667 | 0.996667 | 1 | 0.00666667 | 1 |
| fall_recordings_missed_threshold_0_2 | 6 | 0 | 0 | 0 | 0 | 0 |
| fall_recordings_detected_threshold_0_2 | 6 | 0 | 50 | 75 | 38.7298 | 75 |
| subject_recall_threshold_0_2 | 4 | 1 | 1 | 1 | 0 | 1 |

First 10 rows:

| subject_id | total_recordings | fall_recordings | adl_recordings | fall_recordings_missed_threshold_0_5 | fall_recordings_detected_threshold_0_5 | subject_recall_threshold_0_5 | review_below_90_percent_threshold_0_5 | fall_recordings_missed_threshold_0_2 | fall_recordings_detected_threshold_0_2 | subject_recall_threshold_0_2 | review_below_90_percent_threshold_0_2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SA02 | 154 | 75 | 79 | 0 | 75 | 1 | False | 0 | 75 | 1 | False |
| SA03 | 154 | 75 | 79 | 0 | 75 | 1 | False | 0 | 75 | 1 | False |
| SA09 | 154 | 75 | 79 | 0 | 75 | 1 | False | 0 | 75 | 1 | False |
| SA14 | 154 | 75 | 79 | 1 | 74 | 0.986667 | False | 0 | 75 | 1 | False |
| SE11 | 59 | 0 | 59 | 0 | 0 |  | False | 0 | 0 |  | False |
| SE14 | 59 | 0 | 59 | 0 | 0 |  | False | 0 | 0 |  | False |
### `results/cnn_baseline/recording_level_eval/summary_comparison.csv`

- Shape: `3 rows x 3 columns`
- Missing cells: `0`
- Columns: `level, threshold_0_5_recall, chosen_threshold_recall`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| threshold_0_5_recall | 3 | 0.986667 | 0.993333 | 0.996667 | 0.0057735 | 0.996667 |
| chosen_threshold_recall | 3 | 0.9925 | 0.9975 | 1 | 0.00433013 | 1 |

First 10 rows:

| level | threshold_0_5_recall | chosen_threshold_recall |
| --- | --- | --- |
| window | 0.986667 | 0.9925 |
| recording | 0.996667 | 1 |
| subject | 0.996667 | 1 |
### `results/cnn_baseline/threshold_sweep.csv`

- Shape: `17 rows x 4 columns`
- Missing cells: `0`
- Columns: `threshold, precision, recall, f2`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| threshold | 17 | 0.1 | 0.5 | 0.5 | 0.252488 | 0.9 |
| precision | 17 | 0.949937 | 0.966828 | 0.966905 | 0.00944801 | 0.982609 |
| recall | 17 | 0.979333 | 0.992549 | 0.993333 | 0.00550876 | 0.999333 |
| f2 | 17 | 0.979987 | 0.987261 | 0.988195 | 0.00262201 | 0.989824 |

First 10 rows:

| threshold | precision | recall | f2 |
| --- | --- | --- | --- |
| 0.1 | 0.949937 | 0.999333 | 0.989047 |
| 0.15 | 0.951111 | 0.998667 | 0.988779 |
| 0.2 | 0.955967 | 0.998667 | 0.989824 |
| 0.25 | 0.958948 | 0.996667 | 0.988887 |
| 0.3 | 0.96139 | 0.996 | 0.98888 |
| 0.35 | 0.963226 | 0.995333 | 0.988742 |
| 0.4 | 0.965696 | 0.994667 | 0.988734 |
| 0.45 | 0.966905 | 0.993333 | 0.987933 |
| 0.5 | 0.966905 | 0.993333 | 0.987933 |
| 0.55 | 0.968161 | 0.993333 | 0.988195 |
### `results/cnn_baseline/training_history.csv`

- Shape: `18 rows x 12 columns`
- Missing cells: `0`
- Columns: `epoch, accuracy, loss, pr_auc, precision, recall, val_accuracy, val_loss, val_pr_auc, val_precision, val_recall, learning_rate`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| epoch | 18 | 1 | 9.5 | 9.5 | 5.33854 | 18 |
| accuracy | 18 | 0.971943 | 0.994769 | 0.996121 | 0.00613151 | 0.999134 |
| loss | 18 | 0.00234021 | 0.0156647 | 0.0110875 | 0.018087 | 0.0819537 |
| pr_auc | 18 | 0.953124 | 0.989598 | 0.99079 | 0.00998757 | 0.998308 |
| precision | 18 | 0.750642 | 0.945006 | 0.95541 | 0.0539908 | 0.989409 |
| recall | 18 | 0.978145 | 0.99689 | 0.998327 | 0.00502329 | 1 |
| val_accuracy | 18 | 0.968487 | 0.993823 | 0.99519 | 0.0064075 | 0.996882 |
| val_loss | 18 | 0.0125432 | 0.0222352 | 0.016968 | 0.020163 | 0.102116 |
| val_pr_auc | 18 | 0.97244 | 0.989992 | 0.99039 | 0.00517809 | 0.995538 |
| val_precision | 18 | 0.760284 | 0.953887 | 0.960192 | 0.0495743 | 0.983367 |
| val_recall | 18 | 0.980667 | 0.989148 | 0.988667 | 0.00513253 | 0.998 |
| learning_rate | 18 | 0.00025 | 0.000805556 | 0.001 | 0.000291492 | 0.001 |

First 10 rows:

| epoch | accuracy | loss | pr_auc | precision | recall | val_accuracy | val_loss | val_pr_auc | val_precision | val_recall | learning_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.971943 | 0.0819537 | 0.953124 | 0.750642 | 0.978145 | 0.968487 | 0.102116 | 0.97244 | 0.760284 | 0.998 | 0.001 |
| 2 | 0.990401 | 0.029943 | 0.982565 | 0.89887 | 0.993087 | 0.994228 | 0.0168517 | 0.995216 | 0.952015 | 0.992 | 0.001 |
| 3 | 0.992819 | 0.0229119 | 0.985409 | 0.921933 | 0.99554 | 0.994693 | 0.0157809 | 0.993443 | 0.951654 | 0.997333 | 0.001 |
| 4 | 0.993342 | 0.0209338 | 0.987679 | 0.927488 | 0.99554 | 0.994162 | 0.0169088 | 0.995538 | 0.953145 | 0.99 | 0.001 |
| 5 | 0.995309 | 0.0147522 | 0.989897 | 0.947079 | 0.99777 | 0.99529 | 0.0144497 | 0.994357 | 0.958895 | 0.995333 | 0.001 |
| 6 | 0.994858 | 0.0159059 | 0.990152 | 0.943213 | 0.996432 | 0.994427 | 0.0179363 | 0.992385 | 0.955013 | 0.990667 | 0.001 |
| 7 | 0.995435 | 0.0141873 | 0.989647 | 0.948674 | 0.997547 | 0.995091 | 0.0178831 | 0.98774 | 0.958226 | 0.994 | 0.001 |
| 8 | 0.995543 | 0.0132131 | 0.989856 | 0.949692 | 0.99777 | 0.995953 | 0.0125432 | 0.994481 | 0.966905 | 0.993333 | 0.001 |
| 9 | 0.995958 | 0.0114798 | 0.990261 | 0.953964 | 0.998216 | 0.994294 | 0.0185966 | 0.989911 | 0.961488 | 0.982 | 0.001 |
| 10 | 0.996283 | 0.00999102 | 0.992049 | 0.956856 | 0.999108 | 0.995555 | 0.0166878 | 0.991579 | 0.967384 | 0.988667 | 0.001 |
### `results/cnn_lstm_baseline/cnn_vs_cnn_lstm_comparison.csv`

- Shape: `4 rows x 11 columns`
- Missing cells: `2`
- Columns: `model, threshold, parameters, window_precision, window_recall, window_f1, window_f2, recording_recall, impact_recall, training_time_seconds, epochs`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| parameters | 4 | 33729 | 40273 | 40273 | 7556.36 | 46817 |
| window_precision | 4 | 0.960579 | 0.970499 | 0.972193 | 0.00768577 | 0.97703 |
| window_recall | 4 | 0.986667 | 0.991667 | 0.9925 | 0.00353553 | 0.995 |
| window_f1 | 4 | 0.977487 | 0.980947 | 0.980799 | 0.00298582 | 0.984704 |
| window_f2 | 4 | 0.984534 | 0.987346 | 0.987741 | 0.00203065 | 0.989367 |
| recording_recall | 4 | 0.996667 | 0.999167 | 1 | 0.00166667 | 1 |
| impact_recall | 4 | 0.996667 | 0.999167 | 1 | 0.00166667 | 1 |
| training_time_seconds | 2 | 466.117 | 466.117 | 466.117 | 0 | 466.117 |
| epochs | 4 | 8 | 11 | 11 | 3.4641 | 14 |

First 10 rows:

| model | threshold | parameters | window_precision | window_recall | window_f1 | window_f2 | recording_recall | impact_recall | training_time_seconds | epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN | 0.5 | 33729 | 0.976092 | 0.986667 | 0.981351 | 0.984534 | 0.996667 | 0.996667 |  | 8 |
| CNN-LSTM | 0.5 | 46817 | 0.97703 | 0.9925 | 0.984704 | 0.989367 | 1 | 1 | 466.117 | 14 |
| CNN | chosen | 33729 | 0.968293 | 0.9925 | 0.980247 | 0.987562 | 1 | 1 |  | 8 |
| CNN-LSTM | chosen | 46817 | 0.960579 | 0.995 | 0.977487 | 0.98792 | 1 | 1 | 466.117 | 14 |
### `results/cnn_lstm_baseline/false_positive_diagnostics/false_positive_activity_breakdown.csv`

- Shape: `4 rows x 13 columns`
- Missing cells: `0`
- Columns: `threshold_label, threshold, threshold_name, subject_id, recording_id, true_label, activity_code, activity_name, data_integrity_bug, activity_judgment, judgment_rationale, total_windows, num_fall_windows`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| threshold | 4 | 0.25 | 0.3125 | 0.25 | 0.125 | 0.5 |
| true_label | 4 | 0 | 0 | 0 | 0 | 0 |
| total_windows | 4 | 12 | 12 | 12 | 0 | 12 |
| num_fall_windows | 4 | 0 | 0 | 0 | 0 | 0 |

First 10 rows:

| threshold_label | threshold | threshold_name | subject_id | recording_id | true_label | activity_code | activity_name | data_integrity_bug | activity_judgment | judgment_rationale | total_windows | num_fall_windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.5 | threshold_0_5 | SA03 | SA03:D14_SA03_R05.txt | 0 | D14 | Being on one’s back change to lateral position, wait a moment, and change to one’s back | False | unexpected | Lower-impact ADL pattern should be less fall-like. | 12 | 0 |
| chosen | 0.25 | threshold_0_25 | SA03 | SA03:D13_SA03_R04.txt | 0 | D13 | Sitting a moment, lying quickly, wait a moment, and sit again | False | plausible_borderline | Rapid, abrupt, stumbling, collapse-into-chair, or jumping motion can resemble fall dynamics. | 12 | 0 |
| chosen | 0.25 | threshold_0_25 | SA03 | SA03:D14_SA03_R05.txt | 0 | D14 | Being on one’s back change to lateral position, wait a moment, and change to one’s back | False | unexpected | Lower-impact ADL pattern should be less fall-like. | 12 | 0 |
| chosen | 0.25 | threshold_0_25 | SA09 | SA09:D13_SA09_R04.txt | 0 | D13 | Sitting a moment, lying quickly, wait a moment, and sit again | False | plausible_borderline | Rapid, abrupt, stumbling, collapse-into-chair, or jumping motion can resemble fall dynamics. | 12 | 0 |
### `results/cnn_lstm_baseline/false_positive_diagnostics/full_window_level_comparison.csv`

- Shape: `4 rows x 9 columns`
- Missing cells: `0`
- Columns: `model_threshold, precision, recall, F1, F2, recording_recall, impact_verified_recall, adl_false_trigger_count, source`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| precision | 4 | 0.968293 | 0.973466 | 0.973459 | 0.00474475 | 0.978654 |
| recall | 4 | 0.986667 | 0.992708 | 0.992917 | 0.00478109 | 0.998333 |
| F1 | 4 | 0.980247 | 0.982981 | 0.982869 | 0.00263691 | 0.985939 |
| F2 | 4 | 0.984534 | 0.988791 | 0.988962 | 0.00353307 | 0.992708 |
| recording_recall | 4 | 0.996667 | 0.999167 | 1 | 0.00166667 | 1 |
| impact_verified_recall | 4 | 0.996667 | 0.999167 | 1 | 0.00166667 | 1 |
| adl_false_trigger_count | 4 | 0 | 1 | 0.5 | 1.41421 | 3 |

First 10 rows:

| model_threshold | precision | recall | F1 | F2 | recording_recall | impact_verified_recall | adl_false_trigger_count | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN_baseline@0.50 | 0.976092 | 0.986667 | 0.981351 | 0.984534 | 0.996667 | 0.996667 | 0 | reused |
| CNN_baseline@0.20 | 0.968293 | 0.9925 | 0.980247 | 0.987562 | 1 | 1 | 0 | reused |
| CNN_LSTM@0.50 | 0.978654 | 0.993333 | 0.985939 | 0.990362 | 1 | 1 | 1 | reused |
| CNN_LSTM@0.25 | 0.970827 | 0.998333 | 0.984388 | 0.992708 | 1 | 1 | 3 | reused |
### `results/cnn_lstm_baseline/recording_level_eval/false_trigger_only_recordings.csv`

- Shape: `0 rows x 4 columns`
- Missing cells: `0`
- Columns: `threshold, subject_id, recording_id, num_fall_windows`

Numeric statistics:

_No rows._

First 10 rows:

_No rows._
### `results/cnn_lstm_baseline/recording_level_eval/impact_window_verification.csv`

- Shape: `300 rows x 9 columns`
- Missing cells: `0`
- Columns: `subject_id, recording_id, num_fall_windows, predicted_positive_threshold_0_5, num_detected_fall_windows_threshold_0_5, detection_type_threshold_0_5, predicted_positive_threshold_0_25, num_detected_fall_windows_threshold_0_25, detection_type_threshold_0_25`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| num_fall_windows | 300 | 4 | 4 | 4 | 0 | 4 |
| predicted_positive_threshold_0_5 | 300 | 1 | 1 | 1 | 0 | 1 |
| num_detected_fall_windows_threshold_0_5 | 300 | 3 | 3.97 | 4 | 0.170872 | 4 |
| predicted_positive_threshold_0_25 | 300 | 1 | 1 | 1 | 0 | 1 |
| num_detected_fall_windows_threshold_0_25 | 300 | 3 | 3.98 | 4 | 0.140234 | 4 |

First 10 rows:

| subject_id | recording_id | num_fall_windows | predicted_positive_threshold_0_5 | num_detected_fall_windows_threshold_0_5 | detection_type_threshold_0_5 | predicted_positive_threshold_0_25 | num_detected_fall_windows_threshold_0_25 | detection_type_threshold_0_25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SA02 | SA02:F01_SA02_R01.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
| SA02 | SA02:F01_SA02_R02.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
| SA02 | SA02:F01_SA02_R03.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
| SA02 | SA02:F01_SA02_R04.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
| SA02 | SA02:F01_SA02_R05.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
| SA02 | SA02:F02_SA02_R01.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
| SA02 | SA02:F02_SA02_R02.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
| SA02 | SA02:F02_SA02_R03.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
| SA02 | SA02:F02_SA02_R04.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
| SA02 | SA02:F02_SA02_R05.txt | 4 | 1 | 4 | genuine_impact | 1 | 4 | genuine_impact |
### `results/cnn_lstm_baseline/recording_level_eval/missed_fall_recordings.csv`

- Shape: `0 rows x 3 columns`
- Missing cells: `0`
- Columns: `subject_id, recording_id, num_fall_windows`

Numeric statistics:

_No rows._

First 10 rows:

_No rows._
### `results/cnn_lstm_baseline/recording_level_eval/recording_level_metrics.csv`

- Shape: `734 rows x 15 columns`
- Missing cells: `0`
- Columns: `recording_id, subject_id, true_label, num_windows, num_fall_windows, predicted_positive_threshold_0_5, num_detected_fall_windows_threshold_0_5, predicted_positive_threshold_0_25, num_detected_fall_windows_threshold_0_25, detected_fall_recording_threshold_0_5, fully_missed_fall_recording_threshold_0_5, false_positive_recording_threshold_0_5, detected_fall_recording_threshold_0_25, fully_missed_fall_recording_threshold_0_25, false_positive_recording_threshold_0_25`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| true_label | 734 | 0 | 0.408719 | 0 | 0.491932 | 1 |
| num_windows | 734 | 12 | 18.5654 | 15 | 19.6046 | 122 |
| num_fall_windows | 734 | 0 | 1.63488 | 0 | 1.96773 | 4 |
| predicted_positive_threshold_0_5 | 734 | 0 | 0.408719 | 0 | 0.491932 | 1 |
| num_detected_fall_windows_threshold_0_5 | 734 | 0 | 1.62262 | 0 | 1.95602 | 4 |
| predicted_positive_threshold_0_25 | 734 | 0 | 0.408719 | 0 | 0.491932 | 1 |
| num_detected_fall_windows_threshold_0_25 | 734 | 0 | 1.6267 | 0 | 1.95994 | 4 |
| detected_fall_recording_threshold_0_5 | 734 | 0 | 0.408719 | 0 | 0.491932 | 1 |
| fully_missed_fall_recording_threshold_0_5 | 734 | 0 | 0 | 0 | 0 | 0 |
| false_positive_recording_threshold_0_5 | 734 | 0 | 0 | 0 | 0 | 0 |
| detected_fall_recording_threshold_0_25 | 734 | 0 | 0.408719 | 0 | 0.491932 | 1 |
| fully_missed_fall_recording_threshold_0_25 | 734 | 0 | 0 | 0 | 0 | 0 |
| false_positive_recording_threshold_0_25 | 734 | 0 | 0 | 0 | 0 | 0 |

First 10 rows:

| recording_id | subject_id | true_label | num_windows | num_fall_windows | predicted_positive_threshold_0_5 | num_detected_fall_windows_threshold_0_5 | predicted_positive_threshold_0_25 | num_detected_fall_windows_threshold_0_25 | detected_fall_recording_threshold_0_5 | fully_missed_fall_recording_threshold_0_5 | false_positive_recording_threshold_0_5 | detected_fall_recording_threshold_0_25 | fully_missed_fall_recording_threshold_0_25 | false_positive_recording_threshold_0_25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SA02:D01_SA02_R01.txt | SA02 | 0 | 122 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SA02:D02_SA02_R01.txt | SA02 | 0 | 122 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SA02:D03_SA02_R01.txt | SA02 | 0 | 122 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SA02:D04_SA02_R01.txt | SA02 | 0 | 122 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SA02:D05_SA02_R01.txt | SA02 | 0 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SA02:D05_SA02_R02.txt | SA02 | 0 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SA02:D05_SA02_R03.txt | SA02 | 0 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SA02:D05_SA02_R04.txt | SA02 | 0 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SA02:D05_SA02_R05.txt | SA02 | 0 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SA02:D06_SA02_R01.txt | SA02 | 0 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
### `results/cnn_lstm_baseline/recording_level_eval/subject_level_metrics.csv`

- Shape: `6 rows x 12 columns`
- Missing cells: `4`
- Columns: `subject_id, total_recordings, fall_recordings, adl_recordings, fall_recordings_missed_threshold_0_5, fall_recordings_detected_threshold_0_5, subject_recall_threshold_0_5, review_below_90_percent_threshold_0_5, fall_recordings_missed_threshold_0_25, fall_recordings_detected_threshold_0_25, subject_recall_threshold_0_25, review_below_90_percent_threshold_0_25`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| total_recordings | 6 | 59 | 122.333 | 154 | 49.0578 | 154 |
| fall_recordings | 6 | 0 | 50 | 75 | 38.7298 | 75 |
| adl_recordings | 6 | 59 | 72.3333 | 79 | 10.328 | 79 |
| fall_recordings_missed_threshold_0_5 | 6 | 0 | 0 | 0 | 0 | 0 |
| fall_recordings_detected_threshold_0_5 | 6 | 0 | 50 | 75 | 38.7298 | 75 |
| subject_recall_threshold_0_5 | 4 | 1 | 1 | 1 | 0 | 1 |
| fall_recordings_missed_threshold_0_25 | 6 | 0 | 0 | 0 | 0 | 0 |
| fall_recordings_detected_threshold_0_25 | 6 | 0 | 50 | 75 | 38.7298 | 75 |
| subject_recall_threshold_0_25 | 4 | 1 | 1 | 1 | 0 | 1 |

First 10 rows:

| subject_id | total_recordings | fall_recordings | adl_recordings | fall_recordings_missed_threshold_0_5 | fall_recordings_detected_threshold_0_5 | subject_recall_threshold_0_5 | review_below_90_percent_threshold_0_5 | fall_recordings_missed_threshold_0_25 | fall_recordings_detected_threshold_0_25 | subject_recall_threshold_0_25 | review_below_90_percent_threshold_0_25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SA02 | 154 | 75 | 79 | 0 | 75 | 1 | False | 0 | 75 | 1 | False |
| SA03 | 154 | 75 | 79 | 0 | 75 | 1 | False | 0 | 75 | 1 | False |
| SA09 | 154 | 75 | 79 | 0 | 75 | 1 | False | 0 | 75 | 1 | False |
| SA14 | 154 | 75 | 79 | 0 | 75 | 1 | False | 0 | 75 | 1 | False |
| SE11 | 59 | 0 | 59 | 0 | 0 |  | False | 0 | 0 |  | False |
| SE14 | 59 | 0 | 59 | 0 | 0 |  | False | 0 | 0 |  | False |
### `results/cnn_lstm_baseline/recording_level_eval/summary_comparison.csv`

- Shape: `3 rows x 3 columns`
- Missing cells: `0`
- Columns: `level, threshold_0_5_recall, chosen_threshold_recall`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| threshold_0_5_recall | 3 | 0.9925 | 0.9975 | 1 | 0.00433013 | 1 |
| chosen_threshold_recall | 3 | 0.995 | 0.998333 | 1 | 0.00288675 | 1 |

First 10 rows:

| level | threshold_0_5_recall | chosen_threshold_recall |
| --- | --- | --- |
| window | 0.9925 | 0.995 |
| recording | 1 | 1 |
| subject | 1 | 1 |
### `results/cnn_lstm_baseline/severity_eval/k2_20260914T055946Z/overall_recording_metrics.csv`

- Shape: `1,468 rows x 7 columns`
- Missing cells: `0`
- Columns: `threshold_role, threshold, recording_id, subject_id, true_fall_recording, predicted_positive, two_consecutive_alert`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| threshold | 1468 | 0.25 | 0.375 | 0.375 | 0.125043 | 0.5 |

First 10 rows:

| threshold_role | threshold | recording_id | subject_id | true_fall_recording | predicted_positive | two_consecutive_alert |
| --- | --- | --- | --- | --- | --- | --- |
| validation_selected | 0.25 | SA02:D01_SA02_R01.txt | SA02 | False | True | True |
| validation_selected | 0.25 | SA02:D02_SA02_R01.txt | SA02 | False | False | False |
| validation_selected | 0.25 | SA02:D03_SA02_R01.txt | SA02 | False | False | False |
| validation_selected | 0.25 | SA02:D04_SA02_R01.txt | SA02 | False | False | False |
| validation_selected | 0.25 | SA02:D05_SA02_R01.txt | SA02 | False | False | False |
| validation_selected | 0.25 | SA02:D05_SA02_R02.txt | SA02 | False | False | False |
| validation_selected | 0.25 | SA02:D05_SA02_R03.txt | SA02 | False | False | False |
| validation_selected | 0.25 | SA02:D05_SA02_R04.txt | SA02 | False | False | False |
| validation_selected | 0.25 | SA02:D05_SA02_R05.txt | SA02 | False | False | False |
| validation_selected | 0.25 | SA02:D06_SA02_R01.txt | SA02 | False | False | False |
### `results/cnn_lstm_baseline/severity_eval/k2_20260914T055946Z/recording_metrics_by_severity.csv`

- Shape: `650 rows x 11 columns`
- Missing cells: `0`
- Columns: `threshold_role, threshold, severity_label, severity, recording_id, subject_id, severity_windows, detected_severity_windows, recording_detected, impact_verified, two_consecutive_alert`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| threshold | 650 | 0.25 | 0.375 | 0.375 | 0.125096 | 0.5 |
| severity_label | 650 | -1 | 0.532308 | 1 | 0.573989 | 1 |
| severity_windows | 650 | 1 | 3.69231 | 4 | 0.814413 | 4 |
| detected_severity_windows | 650 | 0 | 3.66923 | 4 | 0.859896 | 4 |

First 10 rows:

| threshold_role | threshold | severity_label | severity | recording_id | subject_id | severity_windows | detected_severity_windows | recording_detected | impact_verified | two_consecutive_alert |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation_selected | 0.25 | -1 | Uncertain (-1) | SA02:F01_SA02_R01.txt | SA02 | 4 | 4 | True | True | True |
| validation_selected | 0.25 | -1 | Uncertain (-1) | SA02:F05_SA02_R02.txt | SA02 | 1 | 1 | True | True | False |
| validation_selected | 0.25 | -1 | Uncertain (-1) | SA02:F05_SA02_R03.txt | SA02 | 4 | 4 | True | True | True |
| validation_selected | 0.25 | -1 | Uncertain (-1) | SA02:F05_SA02_R04.txt | SA02 | 4 | 4 | True | True | True |
| validation_selected | 0.25 | -1 | Uncertain (-1) | SA02:F05_SA02_R05.txt | SA02 | 1 | 1 | True | True | False |
| validation_selected | 0.25 | -1 | Uncertain (-1) | SA02:F14_SA02_R05.txt | SA02 | 1 | 1 | True | True | False |
| validation_selected | 0.25 | -1 | Uncertain (-1) | SA03:F01_SA03_R03.txt | SA03 | 1 | 1 | True | True | False |
| validation_selected | 0.25 | -1 | Uncertain (-1) | SA03:F08_SA03_R04.txt | SA03 | 1 | 1 | True | True | False |
| validation_selected | 0.25 | -1 | Uncertain (-1) | SA03:F08_SA03_R05.txt | SA03 | 1 | 1 | True | True | False |
| validation_selected | 0.25 | -1 | Uncertain (-1) | SA09:F04_SA09_R01.txt | SA09 | 1 | 1 | True | True | False |
### `results/cnn_lstm_baseline/severity_eval/k2_20260914T055946Z/recording_summary_by_severity.csv`

- Shape: `6 rows x 8 columns`
- Missing cells: `0`
- Columns: `threshold_role, threshold, severity_label, severity, recordings, detected_recordings, impact_verified_recordings, two_consecutive_recordings`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| threshold | 6 | 0.25 | 0.375 | 0.375 | 0.136931 | 0.5 |
| severity_label | 6 | -1 | 0 | 0 | 0.894427 | 1 |
| recordings | 6 | 13 | 108.333 | 126 | 78.5689 | 186 |
| detected_recordings | 6 | 13 | 107.5 | 123.5 | 78.355 | 186 |
| impact_verified_recordings | 6 | 13 | 107.5 | 123.5 | 78.355 | 186 |
| two_consecutive_recordings | 6 | 4 | 100.167 | 112.5 | 81.064 | 184 |

First 10 rows:

| threshold_role | threshold | severity_label | severity | recordings | detected_recordings | impact_verified_recordings | two_consecutive_recordings |
| --- | --- | --- | --- | --- | --- | --- | --- |
| validation_selected | 0.25 | -1 | Uncertain (-1) | 13 | 13 | 13 | 4 |
| validation_selected | 0.25 | 0 | Mild | 126 | 124 | 124 | 113 |
| validation_selected | 0.25 | 1 | Severe | 186 | 186 | 186 | 184 |
| reference_0.50 | 0.5 | -1 | Uncertain (-1) | 13 | 13 | 13 | 4 |
| reference_0.50 | 0.5 | 0 | Mild | 126 | 123 | 123 | 112 |
| reference_0.50 | 0.5 | 1 | Severe | 186 | 186 | 186 | 184 |
### `results/cnn_lstm_baseline/severity_eval/k2_20260914T055946Z/subject_metrics_by_severity.csv`

- Shape: `24 rows x 9 columns`
- Missing cells: `0`
- Columns: `threshold_role, threshold, severity_label, severity, subject_id, severity_windows, detected_severity_windows, subject_recall, impact_verified`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| threshold | 24 | 0.25 | 0.375 | 0.375 | 0.127688 | 0.5 |
| severity_label | 24 | -1 | 0 | 0 | 0.834058 | 1 |
| severity_windows | 24 | 3 | 100 | 116.5 | 74.0012 | 201 |
| detected_severity_windows | 24 | 2 | 99.375 | 116.5 | 73.9752 | 200 |
| subject_recall | 24 | 0.666667 | 0.96726 | 1 | 0.0937765 | 1 |

First 10 rows:

| threshold_role | threshold | severity_label | severity | subject_id | severity_windows | detected_severity_windows | subject_recall | impact_verified |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation_selected | 0.25 | -1 | Uncertain (-1) | SA02 | 15 | 15 | 1 | True |
| validation_selected | 0.25 | -1 | Uncertain (-1) | SA03 | 3 | 3 | 1 | True |
| validation_selected | 0.25 | -1 | Uncertain (-1) | SA09 | 3 | 3 | 1 | True |
| validation_selected | 0.25 | -1 | Uncertain (-1) | SA14 | 3 | 2 | 0.666667 | True |
| validation_selected | 0.25 | 0 | Mild | SA02 | 109 | 109 | 1 | True |
| validation_selected | 0.25 | 0 | Mild | SA03 | 128 | 128 | 1 | True |
| validation_selected | 0.25 | 0 | Mild | SA09 | 124 | 124 | 1 | True |
| validation_selected | 0.25 | 0 | Mild | SA14 | 96 | 92 | 0.958333 | True |
| validation_selected | 0.25 | 1 | Severe | SA02 | 176 | 176 | 1 | True |
| validation_selected | 0.25 | 1 | Severe | SA03 | 169 | 169 | 1 | True |
### `results/cnn_lstm_baseline/severity_eval/k2_20260914T055946Z/subject_summary_by_severity.csv`

- Shape: `6 rows x 8 columns`
- Missing cells: `0`
- Columns: `threshold_role, threshold, severity_label, severity, subjects, mean_subject_recall, min_subject_recall, impact_verified_subjects`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| threshold | 6 | 0.25 | 0.375 | 0.375 | 0.136931 | 0.5 |
| severity_label | 6 | -1 | 0 | 0 | 0.894427 | 1 |
| subjects | 6 | 4 | 4 | 4 | 0 | 4 |
| mean_subject_recall | 6 | 0.916667 | 0.96726 | 0.986979 | 0.039542 | 0.998756 |
| min_subject_recall | 6 | 0.666667 | 0.86904 | 0.947917 | 0.158168 | 0.995025 |
| impact_verified_subjects | 6 | 4 | 4 | 4 | 0 | 4 |

First 10 rows:

| threshold_role | threshold | severity_label | severity | subjects | mean_subject_recall | min_subject_recall | impact_verified_subjects |
| --- | --- | --- | --- | --- | --- | --- | --- |
| validation_selected | 0.25 | -1 | Uncertain (-1) | 4 | 0.916667 | 0.666667 | 4 |
| validation_selected | 0.25 | 0 | Mild | 4 | 0.989583 | 0.958333 | 4 |
| validation_selected | 0.25 | 1 | Severe | 4 | 0.998756 | 0.995025 | 4 |
| reference_0.50 | 0.5 | -1 | Uncertain (-1) | 4 | 0.916667 | 0.666667 | 4 |
| reference_0.50 | 0.5 | 0 | Mild | 4 | 0.984375 | 0.9375 | 4 |
| reference_0.50 | 0.5 | 1 | Severe | 4 | 0.997512 | 0.99005 | 4 |
### `results/cnn_lstm_baseline/severity_eval/k2_20260914T055946Z/window_metrics_by_severity.csv`

- Shape: `6 rows x 9 columns`
- Missing cells: `0`
- Columns: `threshold_role, threshold, severity_label, severity, windows, detected_windows, window_recall, window_precision, window_f1`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| threshold | 6 | 0.25 | 0.375 | 0.375 | 0.136931 | 0.5 |
| severity_label | 6 | -1 | 0 | 0 | 0.894427 | 1 |
| windows | 6 | 24 | 400 | 457 | 313.934 | 719 |
| detected_windows | 6 | 23 | 397.5 | 452 | 313.446 | 718 |
| window_recall | 6 | 0.958333 | 0.981769 | 0.989059 | 0.0186365 | 0.998609 |
| window_precision | 6 | 0.0185036 | 0.322935 | 0.367208 | 0.254677 | 0.588187 |
| window_f1 | 6 | 0.0363062 | 0.436048 | 0.535564 | 0.322096 | 0.739938 |

First 10 rows:

| threshold_role | threshold | severity_label | severity | windows | detected_windows | window_recall | window_precision | window_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation_selected | 0.25 | -1 | Uncertain (-1) | 24 | 23 | 0.958333 | 0.0185036 | 0.0363062 |
| validation_selected | 0.25 | 0 | Mild | 457 | 453 | 0.991247 | 0.364441 | 0.532941 |
| validation_selected | 0.25 | 1 | Severe | 719 | 718 | 0.998609 | 0.577635 | 0.731906 |
| reference_0.50 | 0.5 | -1 | Uncertain (-1) | 24 | 23 | 0.958333 | 0.0188679 | 0.0370072 |
| reference_0.50 | 0.5 | 0 | Mild | 457 | 451 | 0.986871 | 0.369975 | 0.538186 |
| reference_0.50 | 0.5 | 1 | Severe | 719 | 717 | 0.997218 | 0.588187 | 0.739938 |
### `results/cnn_lstm_baseline/threshold_sweep.csv`

- Shape: `17 rows x 5 columns`
- Missing cells: `0`
- Columns: `threshold, precision, recall, f1, f2`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| threshold | 17 | 0.1 | 0.5 | 0.5 | 0.252488 | 0.9 |
| precision | 17 | 0.93601 | 0.966717 | 0.969261 | 0.0143948 | 0.98762 |
| recall | 17 | 0.957333 | 0.984627 | 0.988 | 0.010142 | 0.994667 |
| f1 | 17 | 0.964447 | 0.975456 | 0.975706 | 0.0043475 | 0.980782 |
| f2 | 17 | 0.963241 | 0.980905 | 0.983813 | 0.00591207 | 0.984658 |

First 10 rows:

| threshold | precision | recall | f1 | f2 |
| --- | --- | --- | --- | --- |
| 0.1 | 0.93601 | 0.994667 | 0.964447 | 0.982354 |
| 0.15 | 0.945466 | 0.994 | 0.969126 | 0.983899 |
| 0.2 | 0.952046 | 0.992667 | 0.971932 | 0.984268 |
| 0.25 | 0.953876 | 0.992667 | 0.972885 | 0.984658 |
| 0.3 | 0.957502 | 0.991333 | 0.974124 | 0.984377 |
| 0.35 | 0.96119 | 0.990667 | 0.975706 | 0.984628 |
| 0.4 | 0.962362 | 0.988667 | 0.975337 | 0.983291 |
| 0.45 | 0.964867 | 0.988667 | 0.976622 | 0.983813 |
| 0.5 | 0.969261 | 0.988 | 0.978541 | 0.984194 |
| 0.55 | 0.970511 | 0.987333 | 0.97885 | 0.983922 |
### `results/cnn_lstm_baseline/training_history.csv`

- Shape: `14 rows x 12 columns`
- Missing cells: `0`
- Columns: `epoch, accuracy, loss, pr_auc, precision, recall, val_accuracy, val_loss, val_pr_auc, val_precision, val_recall, learning_rate`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| epoch | 14 | 1 | 7.5 | 7.5 | 4.1833 | 14 |
| accuracy | 14 | 0.967288 | 0.993227 | 0.995065 | 0.00808388 | 0.998863 |
| loss | 14 | 0.00278467 | 0.01987 | 0.0136613 | 0.0247246 | 0.0992646 |
| pr_auc | 14 | 0.938433 | 0.988026 | 0.991482 | 0.0151103 | 0.998305 |
| precision | 14 | 0.72262 | 0.931521 | 0.944665 | 0.0678011 | 0.986359 |
| recall | 14 | 0.966771 | 0.995014 | 0.997658 | 0.00863184 | 0.999777 |
| val_accuracy | 14 | 0.993299 | 0.995105 | 0.995389 | 0.00103409 | 0.996948 |
| val_loss | 14 | 0.0132467 | 0.0167747 | 0.0163359 | 0.00210086 | 0.0212579 |
| val_pr_auc | 14 | 0.988497 | 0.992027 | 0.992258 | 0.00269219 | 0.997035 |
| val_precision | 14 | 0.953926 | 0.967541 | 0.968944 | 0.00925301 | 0.982095 |
| val_recall | 14 | 0.976 | 0.983905 | 0.984 | 0.00423947 | 0.989333 |
| learning_rate | 14 | 0.00025 | 0.00075 | 0.001 | 0.000310087 | 0.001 |

First 10 rows:

| epoch | accuracy | loss | pr_auc | precision | recall | val_accuracy | val_loss | val_pr_auc | val_precision | val_recall | learning_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.967288 | 0.0992646 | 0.938433 | 0.72262 | 0.966771 | 0.993299 | 0.020017 | 0.995829 | 0.953926 | 0.98 | 0.001 |
| 2 | 0.988001 | 0.036211 | 0.980198 | 0.877894 | 0.989295 | 0.994029 | 0.0157665 | 0.997035 | 0.956606 | 0.984667 | 0.001 |
| 3 | 0.990527 | 0.0271336 | 0.985882 | 0.900303 | 0.992864 | 0.994361 | 0.0160793 | 0.994648 | 0.961513 | 0.982667 | 0.001 |
| 4 | 0.993414 | 0.0205384 | 0.989791 | 0.927548 | 0.996432 | 0.995688 | 0.0132467 | 0.994389 | 0.969261 | 0.988 | 0.001 |
| 5 | 0.99419 | 0.0164116 | 0.989633 | 0.935174 | 0.997324 | 0.994228 | 0.018903 | 0.990001 | 0.954926 | 0.988667 | 0.001 |
| 6 | 0.994822 | 0.0161413 | 0.988451 | 0.942441 | 0.996878 | 0.993697 | 0.0212579 | 0.989312 | 0.961261 | 0.976 | 0.001 |
| 7 | 0.995309 | 0.0127755 | 0.992199 | 0.94689 | 0.997993 | 0.995091 | 0.0162134 | 0.992 | 0.970317 | 0.980667 | 0.001 |
| 8 | 0.994605 | 0.0145472 | 0.990764 | 0.939878 | 0.997101 | 0.995356 | 0.0164584 | 0.992524 | 0.970395 | 0.983333 | 0.001 |
| 9 | 0.996085 | 0.0109764 | 0.99405 | 0.95539 | 0.998216 | 0.995621 | 0.0151712 | 0.992517 | 0.968627 | 0.988 | 0.0005 |
| 10 | 0.997672 | 0.00618751 | 0.996218 | 0.972445 | 0.999554 | 0.995422 | 0.0149539 | 0.992534 | 0.965517 | 0.989333 | 0.0005 |
### `results/comparison/model_comparison.csv`

- Shape: `5 rows x 13 columns`
- Missing cells: `4`
- Columns: `model, val_selected_threshold, accuracy, precision, recall, f1, f2, fa_per_hour, auroc, auprc, parameters, severity_macro_f1, status`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| val_selected_threshold | 5 | 0.2 | 0.41 | 0.25 | 0.248495 | 0.75 |

First 10 rows:

| model | val_selected_threshold | accuracy | precision | recall | f1 | f2 | fa_per_hour | auroc | auprc | parameters | severity_macro_f1 | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN (binary) | 0.2 | 0.9965 | 0.9683 | 0.9925 | 0.9802 | 0.9876 | 14.12 | UNVERIFIED | UNVERIFIED | 33729 |  | reproduced from results/cnn_baseline/run_metadata.json |
| CNN-LSTM (binary) | 0.25 | 0.996 | 0.9606 | 0.995 | 0.9775 | 0.9879 | 17.74 | 0.9998 | 0.9987 | 46817 |  | reproduced from results/cnn_lstm_baseline/run_metadata.json |
| CNN-LSTM + severity head | 0.25 | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | 46883 | UNVERIFIED | architecture/tests/pipeline implemented; training UNVERIFIED pending data/processed/*.npy tensors |
| FiLM (k=1, exploratory) | 0.6 | 0.9932 | 0.972 | 0.9554 | 0.9738 | 0.984 | 16.64 | 0.9998 | 0.9985 | 46817 (+128 affine) |  | reproduced from results/film_personalization/full_run_v1/analysis/test_metric_summary.csv |
| FiLM (k=30, exploratory) | 0.75 | 0.9943 | 0.9693 | 0.9692 | 0.9787 | 0.988 | 14.03 | 0.9998 | 0.9986 | 46817 (+128 affine) |  | reproduced from results/film_personalization/full_run_v1/analysis/test_metric_summary.csv |
### `results/film_personalization/full_run_v1/aggregate_summary.csv`

- Shape: `30 rows x 19 columns`
- Missing cells: `0`
- Columns: `model, k, headline_threshold, n_seeds, n_fall_subjects, n_adl_only_subjects, f1_mean_across_seeds, f1_std_across_seeds, f1_mean_plus_minus_std, f2_mean_across_seeds, f2_std_across_seeds, f2_mean_plus_minus_std, false_alarms_per_hour_mean_across_seeds, false_alarms_per_hour_std_across_seeds, false_alarms_per_hour_mean_plus_minus_std, film_vs_global_f1_wilcoxon_n_fall_subjects, film_vs_global_f1_wilcoxon_statistic, film_vs_global_f1_wilcoxon_p_value, fall_sensitive_metrics_note`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 30 | 0 | 9.2 | 5 | 11.1677 | 30 |
| headline_threshold | 30 | 0.25 | 0.25 | 0.25 | 0 | 0.25 |
| n_seeds | 30 | 3 | 3 | 3 | 0 | 3 |
| n_fall_subjects | 30 | 4 | 4 | 4 | 0 | 4 |
| n_adl_only_subjects | 30 | 0 | 1.33333 | 2 | 0.958927 | 2 |
| f1_mean_across_seeds | 30 | 0.951388 | 0.973849 | 0.977572 | 0.00885547 | 0.978212 |
| f1_std_across_seeds | 30 | 0 | 0.00257928 | 1.35974e-16 | 0.00600678 | 0.0188347 |
| f2_mean_across_seeds | 30 | 0.976791 | 0.986693 | 0.987952 | 0.00308004 | 0.988192 |
| f2_std_across_seeds | 30 | 0 | 0.000940971 | 1.35974e-16 | 0.00221471 | 0.006839 |
| false_alarms_per_hour_mean_across_seeds | 30 | 14.5894 | 21.1637 | 18.7113 | 8.31601 | 44.6523 |
| false_alarms_per_hour_std_across_seeds | 30 | 0 | 2.75156 | 0 | 6.94654 | 30.1213 |
| film_vs_global_f1_wilcoxon_n_fall_subjects | 30 | 4 | 4 | 4 | 0 | 4 |
| film_vs_global_f1_wilcoxon_statistic | 30 | 0 | 0 | 0 | 0 | 0 |
| film_vs_global_f1_wilcoxon_p_value | 30 | 0.125 | 0.125 | 0.125 | 0 | 0.125 |

First 10 rows:

| model | k | headline_threshold | n_seeds | n_fall_subjects | n_adl_only_subjects | f1_mean_across_seeds | f1_std_across_seeds | f1_mean_plus_minus_std | f2_mean_across_seeds | f2_std_across_seeds | f2_mean_plus_minus_std | false_alarms_per_hour_mean_across_seeds | false_alarms_per_hour_std_across_seeds | false_alarms_per_hour_mean_plus_minus_std | film_vs_global_f1_wilcoxon_n_fall_subjects | film_vs_global_f1_wilcoxon_statistic | film_vs_global_f1_wilcoxon_p_value | fall_sensitive_metrics_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Global | 0 | 0.25 | 3 | 4 | 2 | 0.977572 | 1.35974e-16 | 0.977572 +/- 0.000000 | 0.987952 | 0 | 0.987952 +/- 0.000000 | 15.1052 | 0 | 15.105241 +/- 0.000000 | 4 | 0 | 0.125 | ADL-only held-out users (n=2) excluded; F1/F2 N/A for those users. |
| Global | 1 | 0.25 | 3 | 4 | 2 | 0.977572 | 1.35974e-16 | 0.977572 +/- 0.000000 | 0.987952 | 0 | 0.987952 +/- 0.000000 | 15.1052 | 0 | 15.105241 +/- 0.000000 | 4 | 0 | 0.125 | ADL-only held-out users (n=2) excluded; F1/F2 N/A for those users. |
| Global | 5 | 0.25 | 3 | 4 | 2 | 0.977572 | 1.35974e-16 | 0.977572 +/- 0.000000 | 0.987952 | 0 | 0.987952 +/- 0.000000 | 15.1052 | 0 | 15.105241 +/- 0.000000 | 4 | 0 | 0.125 | ADL-only held-out users (n=2) excluded; F1/F2 N/A for those users. |
| Global | 10 | 0.25 | 3 | 4 | 2 | 0.977572 | 1.35974e-16 | 0.977572 +/- 0.000000 | 0.987952 | 0 | 0.987952 +/- 0.000000 | 15.1052 | 0 | 15.105241 +/- 0.000000 | 4 | 0 | 0.125 | ADL-only held-out users (n=2) excluded; F1/F2 N/A for those users. |
| Global | 30 | 0.25 | 3 | 4 | 2 | 0.977572 | 1.35974e-16 | 0.977572 +/- 0.000000 | 0.987952 | 0 | 0.987952 +/- 0.000000 | 15.1052 | 0 | 15.105241 +/- 0.000000 | 4 | 0 | 0.125 | ADL-only held-out users (n=2) excluded; F1/F2 N/A for those users. |
| Last-layer FT (ADL-only) | 0 | 0.25 | 3 | 4 | 2 | 0.977572 | 1.35974e-16 | 0.977572 +/- 0.000000 | 0.987952 | 0 | 0.987952 +/- 0.000000 | 15.1052 | 0 | 15.105241 +/- 0.000000 | 4 | 0 | 0.125 | ADL-only held-out users (n=2) excluded; F1/F2 N/A for those users. |
| Last-layer FT (ADL-only) | 1 | 0.25 | 3 | 4 | 2 | 0.977972 | 0 | 0.977972 +/- 0.000000 | 0.988115 | 1.35974e-16 | 0.988115 +/- 0.000000 | 14.7957 | 0 | 14.795708 +/- 0.000000 | 4 | 0 | 0.125 | ADL-only held-out users (n=2) excluded; F1/F2 N/A for those users. |
| Last-layer FT (ADL-only) | 5 | 0.25 | 3 | 4 | 2 | 0.977972 | 0 | 0.977972 +/- 0.000000 | 0.988115 | 1.35974e-16 | 0.988115 +/- 0.000000 | 14.7957 | 0 | 14.795708 +/- 0.000000 | 4 | 0 | 0.125 | ADL-only held-out users (n=2) excluded; F1/F2 N/A for those users. |
| Last-layer FT (ADL-only) | 10 | 0.25 | 3 | 4 | 2 | 0.977972 | 0 | 0.977972 +/- 0.000000 | 0.988115 | 1.35974e-16 | 0.988115 +/- 0.000000 | 14.7957 | 0 | 14.795708 +/- 0.000000 | 4 | 0 | 0.125 | ADL-only held-out users (n=2) excluded; F1/F2 N/A for those users. |
| Last-layer FT (ADL-only) | 30 | 0.25 | 3 | 4 | 2 | 0.977972 | 0 | 0.977972 +/- 0.000000 | 0.988115 | 1.35974e-16 | 0.988115 +/- 0.000000 | 14.7957 | 0 | 14.795708 +/- 0.000000 | 4 | 0 | 0.125 | ADL-only held-out users (n=2) excluded; F1/F2 N/A for those users. |
### `results/film_personalization/full_run_v1/analysis/affine_diagnostics.csv`

- Shape: `450 rows x 8 columns`
- Missing cells: `0`
- Columns: `meta_seed, subject_id, calibration_sampling_seed, k, gamma, beta, mean_abs_gamma_minus_one, mean_abs_beta`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| meta_seed | 450 | 42 | 43 | 43 | 0.817405 | 44 |
| calibration_sampling_seed | 450 | 17 | 53.4 | 47 | 29.8432 | 101 |
| k | 450 | 0 | 9.2 | 5 | 10.9922 | 30 |
| mean_abs_gamma_minus_one | 450 | 0.0258933 | 0.127146 | 0.125155 | 0.059175 | 0.319175 |
| mean_abs_beta | 450 | 0.0279901 | 0.112096 | 0.115246 | 0.0438733 | 0.246145 |

First 10 rows:

| meta_seed | subject_id | calibration_sampling_seed | k | gamma | beta | mean_abs_gamma_minus_one | mean_abs_beta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | SA02 | 17 | 0 | [0.9744612574577332, 0.9802280068397522, 0.9593567848205566, 1.0454673767089844, 1.0255341529846191, 1.031937837600708, 1.0401533842086792, 0.9731156826019287, 1.0476120710372925, 0.9761656522750854, 1.0119636058807373, 1.0322586297988892, 1.0331684350967407, 1.0437836647033691, 0.9692011475563049, 1.0122201442718506, 0.9954627156257629, 1.044032335281372, 1.029148817062378, 0.9930559992790222, 0.9944395422935486, 0.9800451397895813, 0.961567223072052, 0.9801180362701416, 0.9741944074630737, 1.0463500022888184, 1.0012630224227905, 0.9754196405410767, 1.0339101552963257, 0.9611222743988037, 1.047689437866211, 1.001063346862793, 0.9724512100219727, 1.0340135097503662, 0.9895067811012268, 1.029173731803894, 1.0410135984420776, 0.9738740921020508, 0.9564855694770813, 0.970777153968811, 0.9691944718360901, 1.0369713306427002, 1.0321849584579468, 1.0311028957366943, 0.9752775430679321, 0.9516848921775818, 0.9990354776382446, 1.0156683921813965, 1.0366023778915405, 0.9934179782867432, 1.0469506978988647, 0.964000403881073, 0.9691071510314941, 1.033030390739441, 1.008920431137085, 1.0221143960952759, 0.9997050166130066, 1.0321297645568848, 1.0120289325714111, 0.9646422266960144, 1.023935317993164, 1.041111946105957, 1.0479092597961426, 1.0377202033996582] | [-0.03292996436357498, -0.02982385829091072, -0.03446139022707939, 0.029744066298007965, -0.0006125548388808966, 0.033527523279190063, 0.03338836878538132, -0.028857622295618057, 0.024015622213482857, -0.03933580592274666, -0.026273243129253387, 0.026726435869932175, 0.04256751760840416, 0.03019598498940468, -0.03609147667884827, 0.003985585179179907, -0.026524312794208527, 0.035950712859630585, 0.015597132965922356, -0.029416467994451523, 0.004911990836262703, -0.03452745079994202, -0.041573841124773026, -0.040359579026699066, -0.039328135550022125, 0.01641952246427536, -0.017257893458008766, -0.019454708322882652, 0.031532976776361465, -0.0468190535902977, 0.02810540422797203, 0.017404763028025627, -0.03303492069244385, -0.009630413725972176, 0.028016678988933563, 0.0390566922724247, 0.024360697716474533, -0.038484297692775726, -0.03976278007030487, -0.03601759672164917, -0.03912078216671944, 0.034828007221221924, -0.014034286141395569, 0.0024721561931073666, -0.03839993476867676, -0.03913610056042671, 0.02313634753227234, -0.0037996077444404364, -0.014344106428325176, -0.014935285784304142, 0.03189972788095474, -0.038225457072257996, -0.04328679293394089, 0.023010503500699997, -0.030273962765932083, 0.0319533534348011, -0.03416471183300018, 0.02661481499671936, 0.007039680611342192, -0.03863117843866348, 0.024126455187797546, 0.017851099371910095, 0.03714950382709503, 0.03684605658054352] | 0.028016 | 0.0279901 |
| 42 | SA02 | 17 | 1 | [1.2090435028076172, 1.1805113554000854, 0.9412974715232849, 1.4240649938583374, 0.807903528213501, 0.811212420463562, 0.8670021891593933, 1.351897120475769, 0.9438644051551819, 1.0978937149047852, 1.80544114112854, 0.6011703014373779, 1.3414617776870728, 1.154415249824524, 1.2737735509872437, 1.391076922416687, 0.7016183137893677, 1.1509722471237183, 0.6888017654418945, 0.8706644773483276, 0.7984134554862976, 1.4215574264526367, 1.146626591682434, 1.1549886465072632, 1.2756929397583008, 1.301247000694275, 1.212103009223938, 0.7209784984588623, 1.0569629669189453, 1.1531187295913696, 1.3287502527236938, 0.970507025718689, 1.3886442184448242, 1.157761812210083, 0.517569899559021, 0.919887125492096, 1.2826910018920898, 1.135076642036438, 0.899574875831604, 0.988297700881958, 1.371907353401184, 0.9196459650993347, 1.119571328163147, 0.958102822303772, 1.2142680883407593, 0.5988000631332397, 0.917788028717041, 0.7461933493614197, 1.2328227758407593, 1.340700387954712, 1.3347828388214111, 0.9777088165283203, 1.1682143211364746, 0.7733490467071533, 1.356346607208252, 1.125527024269104, 1.4851925373077393, 0.8980796933174133, 1.3186190128326416, 0.9973382353782654, 0.9750611782073975, 1.77423894405365, 1.1965727806091309, 1.2711381912231445] | [0.267560213804245, 0.3700733780860901, 0.07279906421899796, 0.24208226799964905, -0.20718319714069366, -0.14077356457710266, -0.3000888526439667, 0.34130528569221497, -0.2249308079481125, 0.15251977741718292, 0.05539078265428543, -0.33989623188972473, 0.4555070996284485, -0.060493696480989456, 0.06516645848751068, 0.5996175408363342, -0.2446221262216568, 0.15949659049510956, -0.46421512961387634, 0.1424967646598816, -0.5219622850418091, 0.09212789684534073, 0.03510461002588272, 0.027948755770921707, 0.08001247048377991, -0.42955270409584045, 0.024955347180366516, 0.1505199819803238, 0.05229853838682175, -0.1954006403684616, -0.12606626749038696, -0.07473008334636688, 0.11170575767755508, 0.24469125270843506, -0.06486739218235016, -0.07438895106315613, -0.18405546247959137, 0.10096778720617294, 0.11184637248516083, 0.19924227893352509, 0.11701241880655289, -0.08370749652385712, -0.6717528700828552, -0.22605961561203003, 0.04480445757508278, -0.11589756608009338, -0.06511196494102478, -0.2664525806903839, -0.05946142598986626, 0.12487560510635376, -0.15601815283298492, 0.08887869864702225, 0.10617811977863312, -0.48176297545433044, 0.3993367552757263, 0.27758824825286865, 0.06959887593984604, 0.11562423408031464, 0.7093827724456787, -0.03365209698677063, -0.23835404217243195, 0.9272144436836243, -0.14869937300682068, 0.18544037640094757] | 0.232732 | 0.211274 |
| 42 | SA02 | 17 | 5 | [1.0185842514038086, 0.9750725626945496, 0.8619488477706909, 1.3020786046981812, 0.9745774269104004, 1.0098419189453125, 1.0148279666900635, 1.0681116580963135, 1.0950027704238892, 0.9896303415298462, 1.3852499723434448, 0.8882195353507996, 1.201035737991333, 1.1918959617614746, 0.9696496725082397, 1.2073721885681152, 0.7904712557792664, 1.2204958200454712, 0.9507132768630981, 0.9706882834434509, 0.8226084113121033, 1.0707131624221802, 0.9671540856361389, 0.9541782140731812, 0.981078028678894, 1.2746953964233398, 1.1368471384048462, 0.8475116491317749, 1.148938775062561, 0.9480422735214233, 1.274869680404663, 1.0068566799163818, 1.0533314943313599, 1.1881464719772339, 0.7281434535980225, 1.0530339479446411, 1.2203580141067505, 0.9213225841522217, 0.8017123341560364, 0.8665050268173218, 1.0595346689224243, 1.082104206085205, 1.1182940006256104, 1.0651733875274658, 0.9341403841972351, 0.6953930854797363, 0.9621261954307556, 0.9304240345954895, 1.173048734664917, 1.0134285688400269, 1.2513113021850586, 0.8830902576446533, 0.9312648177146912, 0.9990829825401306, 1.1536812782287598, 1.1503368616104126, 1.1517224311828613, 1.0302075147628784, 1.1730130910873413, 0.9042813181877136, 0.9798980355262756, 1.464005947113037, 1.2271473407745361, 1.2381983995437622] | [0.023756351321935654, 0.05987117439508438, -0.056845277547836304, 0.20563940703868866, -0.09635788202285767, 0.021883537992835045, -0.044048625975847244, 0.06578364968299866, -0.10794375836849213, -0.058967575430870056, -0.04610591381788254, -0.09909084439277649, 0.28914982080459595, 0.056210558861494064, -0.09570472687482834, 0.2814787030220032, -0.24076837301254272, 0.1643747240304947, -0.14136645197868347, -0.0322321280837059, -0.2667362689971924, -0.08194710314273834, -0.10023277997970581, -0.12633705139160156, -0.08845792710781097, -0.15951155126094818, -0.030533384531736374, 0.057340990751981735, 0.12063263356685638, -0.20655640959739685, -0.0042762067168951035, 0.06479138135910034, -0.06737533211708069, 0.10857746005058289, 0.0908639207482338, 0.08994997292757034, 0.013438470661640167, -0.0833168774843216, -0.06693438440561295, -0.030349768698215485, -0.07050616294145584, 0.08488032221794128, -0.35965782403945923, -0.14571256935596466, -0.12436585128307343, -0.12869811058044434, 0.05169965326786041, -0.14302799105644226, -0.05855794623494148, -0.057813577353954315, -0.02273956872522831, -0.07224493473768234, -0.09846819937229156, -0.1527193784713745, 0.07646334171295166, 0.24830539524555206, -0.0895528718829155, 0.18956436216831207, 0.3458710014820099, -0.1342211663722992, -0.07951189577579498, 0.48964381217956543, 0.07256422191858292, 0.16552533209323883] | 0.126009 | 0.117313 |
| 42 | SA02 | 17 | 10 | [0.9954720735549927, 0.9574624300003052, 0.8703460693359375, 1.2755553722381592, 0.9925063848495483, 1.0313454866409302, 1.0354034900665283, 1.0317221879959106, 1.1052988767623901, 0.9697928428649902, 1.3206452131271362, 0.9260305762290955, 1.1775667667388916, 1.1907275915145874, 0.9297034740447998, 1.1791245937347412, 0.8161213994026184, 1.218512773513794, 0.9793369174003601, 0.9711581468582153, 0.833720862865448, 1.0284477472305298, 0.9494370818138123, 0.9284670948982239, 0.9481959342956543, 1.2601373195648193, 1.1258008480072021, 0.860485315322876, 1.1560810804367065, 0.9203874468803406, 1.2611956596374512, 1.0083788633346558, 1.011937141418457, 1.1878019571304321, 0.7664641737937927, 1.0709764957427979, 1.2076542377471924, 0.8981767296791077, 0.7949019074440002, 0.8574950098991394, 1.0255659818649292, 1.0961965322494507, 1.1010479927062988, 1.074914574623108, 0.9048125743865967, 0.7136269807815552, 0.9755328297615051, 0.952903687953949, 1.163560390472412, 0.9669991731643677, 1.230310320854187, 0.8673413395881653, 0.9096181988716125, 1.0266878604888916, 1.1292898654937744, 1.1516391038894653, 1.1032906770706177, 1.0395619869232178, 1.1490846872329712, 0.8959366083145142, 0.9876498579978943, 1.4013640880584717, 1.223719835281372, 1.2246348857879639] | [-0.0034800292924046516, 0.02471858821809292, -0.0604710653424263, 0.19913533329963684, -0.07341304421424866, 0.0402318611741066, -0.01563197746872902, 0.03285771235823631, -0.0950385257601738, -0.0816778764128685, -0.05272257328033447, -0.07392185926437378, 0.2669053375720978, 0.07076551020145416, -0.11619347333908081, 0.24008403718471527, -0.23629692196846008, 0.1662464588880539, -0.09777340292930603, -0.058841854333877563, -0.224946066737175, -0.10266108810901642, -0.11474880576133728, -0.14271464943885803, -0.10902488231658936, -0.12703606486320496, -0.031800899654626846, 0.050994064658880234, 0.1301288902759552, -0.20200157165527344, 0.007327220402657986, 0.07233718782663345, -0.08562120050191879, 0.09006641805171967, 0.10482384264469147, 0.11033936589956284, 0.036183539777994156, -0.1044919490814209, -0.0877324640750885, -0.054962776601314545, -0.09275680035352707, 0.10128019750118256, -0.3226792812347412, -0.12599380314350128, -0.14159372448921204, -0.1298414021730423, 0.07089482992887497, -0.11676586419343948, -0.0541086383163929, -0.08084221184253693, -0.0104445219039917, -0.09202590584754944, -0.11834453046321869, -0.11511144042015076, 0.04445166885852814, 0.23820115625858307, -0.10234324634075165, 0.19392447173595428, 0.2887779176235199, -0.1437177062034607, -0.05938216298818588, 0.42290523648262024, 0.09628079831600189, 0.16510947048664093] | 0.12008 | 0.114439 |
| 42 | SA02 | 17 | 30 | [1.0020467042922974, 0.9720591902732849, 0.8876081705093384, 1.271248698234558, 0.9831200838088989, 1.0131973028182983, 1.023829460144043, 1.0410752296447754, 1.0897583961486816, 0.9647189378738403, 1.3302843570709229, 0.9012811183929443, 1.182517409324646, 1.1745363473892212, 0.9506812691688538, 1.1884419918060303, 0.823468029499054, 1.199721336364746, 0.9609461426734924, 0.9628847241401672, 0.8362454175949097, 1.0482593774795532, 0.9622260928153992, 0.9369900226593018, 0.9656700491905212, 1.2486979961395264, 1.118510127067566, 0.8517190217971802, 1.1425458192825317, 0.9349561333656311, 1.257305383682251, 1.0030415058135986, 1.030853033065796, 1.1704641580581665, 0.7644711136817932, 1.0558847188949585, 1.206925868988037, 0.9126404523849487, 0.8099497556686401, 0.8711873888969421, 1.0457093715667725, 1.077484130859375, 1.0854809284210205, 1.0703948736190796, 0.9228168725967407, 0.7166074514389038, 0.9732486009597778, 0.9390909075737, 1.1666725873947144, 0.986095130443573, 1.2231727838516235, 0.8700321316719055, 0.9301359057426453, 1.009303092956543, 1.1313207149505615, 1.1417696475982666, 1.1139005422592163, 1.0208415985107422, 1.142796277999878, 0.9060734510421753, 0.9943836331367493, 1.3962500095367432, 1.211694359779358, 1.2131569385528564] | [0.010744204744696617, 0.0440797284245491, -0.044360265135765076, 0.19658616185188293, -0.07329364120960236, 0.031341198831796646, -0.031462159007787704, 0.04637107253074646, -0.09579212963581085, -0.06776558607816696, -0.0470743402838707, -0.08732473850250244, 0.2648872137069702, 0.05628754571080208, -0.10586360841989517, 0.25481611490249634, -0.22982940077781677, 0.1629319041967392, -0.10915221273899078, -0.05169161409139633, -0.2282487154006958, -0.09242572635412216, -0.10561409592628479, -0.13137120008468628, -0.09944222867488861, -0.13716654479503632, -0.026742607355117798, 0.055160943418741226, 0.12347643822431564, -0.19377018511295319, 0.0034188441932201385, 0.048298247158527374, -0.07484401017427444, 0.08299054950475693, 0.092048279941082, 0.09839985519647598, 0.023532498627901077, -0.09293748438358307, -0.07539056241512299, -0.040674228221178055, -0.0806698352098465, 0.09212937951087952, -0.33453458547592163, -0.11482612788677216, -0.12888123095035553, -0.12444895505905151, 0.06120118498802185, -0.12056757509708405, -0.04455389827489853, -0.07294756174087524, -0.021604904904961586, -0.08189700543880463, -0.10406909883022308, -0.13160835206508636, 0.06520220637321472, 0.2316778004169464, -0.09290043264627457, 0.18530477583408356, 0.2936202585697174, -0.13734763860702515, -0.06416550278663635, 0.4361090660095215, 0.08131186664104462, 0.15937010943889618] | 0.114965 | 0.110915 |
| 42 | SA02 | 31 | 0 | [0.9744612574577332, 0.9802280068397522, 0.9593567848205566, 1.0454673767089844, 1.0255341529846191, 1.031937837600708, 1.0401533842086792, 0.9731156826019287, 1.0476120710372925, 0.9761656522750854, 1.0119636058807373, 1.0322586297988892, 1.0331684350967407, 1.0437836647033691, 0.9692011475563049, 1.0122201442718506, 0.9954627156257629, 1.044032335281372, 1.029148817062378, 0.9930559992790222, 0.9944395422935486, 0.9800451397895813, 0.961567223072052, 0.9801180362701416, 0.9741944074630737, 1.0463500022888184, 1.0012630224227905, 0.9754196405410767, 1.0339101552963257, 0.9611222743988037, 1.047689437866211, 1.001063346862793, 0.9724512100219727, 1.0340135097503662, 0.9895067811012268, 1.029173731803894, 1.0410135984420776, 0.9738740921020508, 0.9564855694770813, 0.970777153968811, 0.9691944718360901, 1.0369713306427002, 1.0321849584579468, 1.0311028957366943, 0.9752775430679321, 0.9516848921775818, 0.9990354776382446, 1.0156683921813965, 1.0366023778915405, 0.9934179782867432, 1.0469506978988647, 0.964000403881073, 0.9691071510314941, 1.033030390739441, 1.008920431137085, 1.0221143960952759, 0.9997050166130066, 1.0321297645568848, 1.0120289325714111, 0.9646422266960144, 1.023935317993164, 1.041111946105957, 1.0479092597961426, 1.0377202033996582] | [-0.03292996436357498, -0.02982385829091072, -0.03446139022707939, 0.029744066298007965, -0.0006125548388808966, 0.033527523279190063, 0.03338836878538132, -0.028857622295618057, 0.024015622213482857, -0.03933580592274666, -0.026273243129253387, 0.026726435869932175, 0.04256751760840416, 0.03019598498940468, -0.03609147667884827, 0.003985585179179907, -0.026524312794208527, 0.035950712859630585, 0.015597132965922356, -0.029416467994451523, 0.004911990836262703, -0.03452745079994202, -0.041573841124773026, -0.040359579026699066, -0.039328135550022125, 0.01641952246427536, -0.017257893458008766, -0.019454708322882652, 0.031532976776361465, -0.0468190535902977, 0.02810540422797203, 0.017404763028025627, -0.03303492069244385, -0.009630413725972176, 0.028016678988933563, 0.0390566922724247, 0.024360697716474533, -0.038484297692775726, -0.03976278007030487, -0.03601759672164917, -0.03912078216671944, 0.034828007221221924, -0.014034286141395569, 0.0024721561931073666, -0.03839993476867676, -0.03913610056042671, 0.02313634753227234, -0.0037996077444404364, -0.014344106428325176, -0.014935285784304142, 0.03189972788095474, -0.038225457072257996, -0.04328679293394089, 0.023010503500699997, -0.030273962765932083, 0.0319533534348011, -0.03416471183300018, 0.02661481499671936, 0.007039680611342192, -0.03863117843866348, 0.024126455187797546, 0.017851099371910095, 0.03714950382709503, 0.03684605658054352] | 0.028016 | 0.0279901 |
| 42 | SA02 | 31 | 1 | [1.1753343343734741, 1.155066967010498, 0.9467715620994568, 1.4147093296051025, 0.8440330624580383, 0.8539263010025024, 0.9137656092643738, 1.3048897981643677, 0.9892616868019104, 1.0601400136947632, 1.6907458305358887, 0.6555609703063965, 1.333062767982483, 1.1509040594100952, 1.2410428524017334, 1.37617826461792, 0.7311196327209473, 1.1772891283035278, 0.741836667060852, 0.8946864604949951, 0.8091866374015808, 1.3499751091003418, 1.10883629322052, 1.1188610792160034, 1.2201671600341797, 1.3029968738555908, 1.1522350311279297, 0.7375985383987427, 1.0841023921966553, 1.1319503784179688, 1.3229230642318726, 0.9695513844490051, 1.2997256517410278, 1.1455051898956299, 0.5586426258087158, 0.9633381366729736, 1.289090633392334, 1.0932939052581787, 0.8793500661849976, 0.9748744368553162, 1.3091728687286377, 0.9492810964584351, 1.1079847812652588, 0.9702009558677673, 1.1583800315856934, 0.6169559955596924, 0.9180589914321899, 0.7911179661750793, 1.2479705810546875, 1.303585410118103, 1.3275104761123657, 0.9480074644088745, 1.1354374885559082, 0.8301196098327637, 1.3270916938781738, 1.1166223287582397, 1.4102047681808472, 0.9241187572479248, 1.2559176683425903, 0.9578983783721924, 1.0109941959381104, 1.7205979824066162, 1.2103756666183472, 1.2643485069274902] | [0.19969484210014343, 0.3008210062980652, 0.05821613222360611, 0.24796268343925476, -0.18886534869670868, -0.08668266236782074, -0.2491896152496338, 0.2729298770427704, -0.17861999571323395, 0.09431596100330353, 0.011547170579433441, -0.27334967255592346, 0.42280980944633484, -0.03895490616559982, 0.01493341475725174, 0.5473219156265259, -0.25048795342445374, 0.17795240879058838, -0.3816725015640259, 0.10495772957801819, -0.46127748489379883, 0.037240512669086456, -0.010684526525437832, -0.022581499069929123, 0.026951361447572708, -0.36069387197494507, -0.0005314666777849197, 0.11783675849437714, 0.07822209596633911, -0.20745046436786652, -0.08359173685312271, -0.07522495836019516, 0.05620121210813522, 0.1876387596130371, -0.018009476363658905, -0.032147616147994995, -0.14254292845726013, 0.044496163725852966, 0.05714483931660652, 0.13658477365970612, 0.06012553721666336, -0.048745881766080856, -0.6157119274139404, -0.18570001423358917, -0.008820043876767159, -0.12212716042995453, -0.04978575557470322, -0.2532521188259125, -0.04801696166396141, 0.06799361854791641, -0.11721071600914001, 0.03167467564344406, 0.056978851556777954, -0.4018908441066742, 0.3279479444026947, 0.2749325633049011, 0.014284832403063774, 0.1435728371143341, 0.6199033260345459, -0.07499279826879501, -0.18065345287322998, 0.8255760669708252, -0.0865548849105835, 0.19210684299468994] | 0.206812 | 0.17292 |
| 42 | SA02 | 31 | 5 | [1.0087376832962036, 0.9939343333244324, 0.9179943799972534, 1.2687348127365112, 0.9659050703048706, 0.9871953725814819, 1.0091660022735596, 1.05416738986969, 1.057929515838623, 0.9512774348258972, 1.3650213479995728, 0.8648723363876343, 1.196846842765808, 1.1667075157165527, 0.9524774551391602, 1.2040501832962036, 0.8391363620758057, 1.1808775663375854, 0.9226748943328857, 0.9298023581504822, 0.8302762508392334, 1.073930263519287, 0.9783562421798706, 0.9331993460655212, 0.9819738268852234, 1.239454984664917, 1.1325781345367432, 0.822466254234314, 1.1329317092895508, 0.9312644004821777, 1.266583800315857, 0.9900311827659607, 1.0561825037002563, 1.1640877723693848, 0.7490682005882263, 1.0387250185012817, 1.2159247398376465, 0.9220553636550903, 0.8103311061859131, 0.8778694272041321, 1.0796977281570435, 1.0529156923294067, 1.043248176574707, 1.0678460597991943, 0.9385311007499695, 0.6946841478347778, 0.9805686473846436, 0.9101259112358093, 1.176522135734558, 0.9843688607215881, 1.2153065204620361, 0.8555759787559509, 0.9551693797111511, 0.982930600643158, 1.143815040588379, 1.1477060317993164, 1.1264402866363525, 0.9834657907485962, 1.1511027812957764, 0.9157888293266296, 0.9957417845726013, 1.3907842636108398, 1.2055716514587402, 1.2080016136169434] | [0.039328671991825104, 0.0770299956202507, -0.011785969138145447, 0.2092811018228531, -0.06233177334070206, 0.015042170882225037, -0.06514100730419159, 0.07449428737163544, -0.12347452342510223, -0.049334242939949036, -0.026945149526000023, -0.1339007318019867, 0.2876189053058624, 0.040804699063301086, -0.10069005936384201, 0.29745936393737793, -0.23722180724143982, 0.17544826865196228, -0.133457213640213, -0.05554671585559845, -0.24556894600391388, -0.08235348761081696, -0.09558652341365814, -0.12186549603939056, -0.09116218984127045, -0.17019562423229218, -0.014060646295547485, 0.08429649472236633, 0.12705406546592712, -0.19365422427654266, -0.010798182338476181, 0.016765594482421875, -0.060133595019578934, 0.09253869205713272, 0.07159025967121124, 0.09142429381608963, 0.008087985217571259, -0.08047178387641907, -0.06056028604507446, -0.01706554740667343, -0.06755238026380539, 0.08020836114883423, -0.3830516040325165, -0.10434293746948242, -0.1176990270614624, -0.13170373439788818, 0.06654554605484009, -0.1190788745880127, -0.03463611379265785, -0.06711539626121521, -0.04957854002714157, -0.072129026055336, -0.08674654364585876, -0.1782020926475525, 0.11359064280986786, 0.23451662063598633, -0.07284645736217499, 0.18505637347698212, 0.323341429233551, -0.13664627075195312, -0.08710584789514542, 0.4854677617549896, 0.06351946294307709, 0.17260557413101196] | 0.11707 | 0.115388 |
| 42 | SA02 | 31 | 10 | [1.021108627319336, 1.0164306163787842, 0.9353451728820801, 1.252817153930664, 0.9474802017211914, 0.9594647884368896, 0.9852179288864136, 1.0721102952957153, 1.0203955173492432, 0.9572663903236389, 1.3906441926956177, 0.8356623649597168, 1.196084976196289, 1.1480329036712646, 0.972659707069397, 1.2024080753326416, 0.8485805988311768, 1.1539294719696045, 0.8939883708953857, 0.9108960628509521, 0.8355286121368408, 1.1016710996627808, 0.9995481967926025, 0.9492176175117493, 1.0077886581420898, 1.22102952003479, 1.1482088565826416, 0.8188548684120178, 1.1112768650054932, 0.9419873952865601, 1.2579275369644165, 0.9896737933158875, 1.0938674211502075, 1.147184133529663, 0.7449484467506409, 1.0109750032424927, 1.2025654315948486, 0.9444282054901123, 0.8317626714706421, 0.8921599984169006, 1.1087700128555298, 1.0295876264572144, 1.0345842838287354, 1.0513015985488892, 0.9626330733299255, 0.7028790712356567, 0.9827202558517456, 0.892497718334198, 1.1643964052200317, 1.0023748874664307, 1.1984214782714844, 0.8663265705108643, 0.9799166321754456, 0.952587366104126, 1.1461470127105713, 1.1409144401550293, 1.1477910280227661, 0.9577785730361938, 1.161670207977295, 0.9380776286125183, 0.9912105798721313, 1.3811061382293701, 1.183801531791687, 1.194343090057373] | [0.06812060624361038, 0.1097220778465271, 0.009261226281523705, 0.19533196091651917, -0.06961669027805328, -0.011140294373035431, -0.09131675958633423, 0.10071524977684021, -0.14184458553791046, -0.019566621631383896, -0.005383388139307499, -0.16212037205696106, 0.29202812910079956, 0.01931741088628769, -0.07457435131072998, 0.3075834810733795, -0.22196586430072784, 0.1613922119140625, -0.1617085337638855, -0.043291859328746796, -0.25429069995880127, -0.05457395315170288, -0.06930068880319595, -0.09306252002716064, -0.06445230543613434, -0.19776742160320282, 0.006945916451513767, 0.09409411251544952, 0.10867692530155182, -0.177224263548851, -0.03109835647046566, 0.0032468526624143124, -0.031452476978302, 0.10151426494121552, 0.04475659877061844, 0.06810346245765686, -0.018015209585428238, -0.05196504294872284, -0.030880674719810486, 0.013499025255441666, -0.037373486906290054, 0.06163150444626808, -0.3962583541870117, -0.10982181876897812, -0.08797487616539001, -0.11915402114391327, 0.058762721717357635, -0.11850277334451675, -0.039620548486709595, -0.04259749874472618, -0.07830575853586197, -0.044129371643066406, -0.05601383373141289, -0.21109642088413239, 0.1479577273130417, 0.21948808431625366, -0.04569166898727417, 0.16741666197776794, 0.33509865403175354, -0.11729267984628677, -0.10653289407491684, 0.504009485244751, 0.031641680747270584, 0.16274146735668182] | 0.114068 | 0.110626 |
| 42 | SA02 | 31 | 30 | [1.0038292407989502, 0.9876070022583008, 0.9108665585517883, 1.2574002742767334, 0.969220757484436, 0.9916867613792419, 1.0100809335708618, 1.0456830263137817, 1.057848572731018, 0.955583393573761, 1.3529808521270752, 0.876393735408783, 1.1869632005691528, 1.1654679775238037, 0.9465523362159729, 1.1919220685958862, 0.8413139581680298, 1.1786231994628906, 0.9335767030715942, 0.9375913143157959, 0.8355711102485657, 1.0649569034576416, 0.9753851294517517, 0.9346554279327393, 0.9763931632041931, 1.2352904081344604, 1.1343492269515991, 0.8351003527641296, 1.1301276683807373, 0.926044225692749, 1.259401798248291, 0.9983097314834595, 1.0544298887252808, 1.1616801023483276, 0.76056307554245, 1.0354948043823242, 1.2057911157608032, 0.9200894236564636, 0.8134269714355469, 0.8778509497642517, 1.069204568862915, 1.057729959487915, 1.0545575618743896, 1.0667541027069092, 0.9331474900245667, 0.7044575810432434, 0.9811493754386902, 0.9201124310493469, 1.1651486158370972, 0.9790903925895691, 1.2075517177581787, 0.8601178526878357, 0.9479261636734009, 0.9860922694206238, 1.133224368095398, 1.1466401815414429, 1.1216676235198975, 0.9917428493499756, 1.1511552333831787, 0.9182459115982056, 0.993494987487793, 1.3788954019546509, 1.1994247436523438, 1.203836441040039] | [0.0338432602584362, 0.0702948048710823, -0.019658280536532402, 0.1986437737941742, -0.06580943614244461, 0.014830532483756542, -0.055742066353559494, 0.06720508635044098, -0.11780282855033875, -0.049556486308574677, -0.027747564017772675, -0.12230324745178223, 0.2792174518108368, 0.0403236523270607, -0.09633920341730118, 0.27727794647216797, -0.22795933485031128, 0.16680675745010376, -0.1273295283317566, -0.05392637103796005, -0.23366552591323853, -0.07941827923059464, -0.09216627478599548, -0.11778804659843445, -0.08785219490528107, -0.15967440605163574, -0.010153578594326973, 0.07584844529628754, 0.12146025151014328, -0.1864587515592575, -0.010565878823399544, 0.030139848589897156, -0.0579608753323555, 0.08912583440542221, 0.07130542397499084, 0.08923819661140442, 0.010081079788506031, -0.07837877422571182, -0.058487050235271454, -0.019153909757733345, -0.06516443192958832, 0.08372914791107178, -0.36433538794517517, -0.10914678126573563, -0.11386655271053314, -0.1263333261013031, 0.0675416812300682, -0.11239828169345856, -0.04292432591319084, -0.06510680913925171, -0.04883481562137604, -0.06822758167982101, -0.08675578981637955, -0.16548903286457062, 0.09989018738269806, 0.22888119518756866, -0.07274577021598816, 0.18124526739120483, 0.3065682649612427, -0.13129204511642456, -0.0821320042014122, 0.4651167392730713, 0.0630050003528595, 0.16481627523899078] | 0.113574 | 0.111048 |
### `results/film_personalization/full_run_v1/analysis/affine_variability.csv`

- Shape: `3 rows x 9 columns`
- Missing cells: `0`
- Columns: `k, within_user_gamma_std_mean_channels, between_user_gamma_std_mean_channels, within_user_beta_std_mean_channels, between_user_beta_std_mean_channels, mean_abs_gamma_minus_one, mean_abs_beta, users, replicates_per_user`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 3 | 0 | 10.3333 | 1 | 17.0392 | 30 |
| within_user_gamma_std_mean_channels | 3 | 0.0190693 | 0.0816159 | 0.0935283 | 0.057523 | 0.13225 |
| between_user_gamma_std_mean_channels | 3 | 0 | 0.0105806 | 0.0106934 | 0.0105247 | 0.0210485 |
| within_user_beta_std_mean_channels | 3 | 0.0183402 | 0.0732349 | 0.0813398 | 0.0513245 | 0.120025 |
| between_user_beta_std_mean_channels | 3 | 0 | 0.0109495 | 0.0124117 | 0.0102966 | 0.0204369 |
| mean_abs_gamma_minus_one | 3 | 0.0358526 | 0.116205 | 0.140647 | 0.0713439 | 0.172116 |
| mean_abs_beta | 3 | 0.0388092 | 0.102793 | 0.123983 | 0.0564547 | 0.145587 |
| users | 3 | 6 | 6 | 6 | 0 | 6 |
| replicates_per_user | 3 | 15 | 15 | 15 | 0 | 15 |

First 10 rows:

| k | within_user_gamma_std_mean_channels | between_user_gamma_std_mean_channels | within_user_beta_std_mean_channels | between_user_beta_std_mean_channels | mean_abs_gamma_minus_one | mean_abs_beta | users | replicates_per_user |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0190693 | 0 | 0.0183402 | 0 | 0.0358526 | 0.0388092 | 6 | 15 |
| 1 | 0.13225 | 0.0210485 | 0.120025 | 0.0204369 | 0.172116 | 0.145587 | 6 | 15 |
| 30 | 0.0935283 | 0.0106934 | 0.0813398 | 0.0124117 | 0.140647 | 0.123983 | 6 | 15 |
### `results/film_personalization/full_run_v1/analysis/affine_variability_by_channel.csv`

- Shape: `192 rows x 6 columns`
- Missing cells: `0`
- Columns: `k, feature_channel, within_user_gamma_std, between_user_gamma_std, within_user_beta_std, between_user_beta_std`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 192 | 0 | 10.3333 | 1 | 13.9488 | 30 |
| feature_channel | 192 | 0 | 31.5 | 31.5 | 18.5212 | 63 |
| within_user_gamma_std | 192 | 0.00415498 | 0.0816159 | 0.0682871 | 0.0696946 | 0.392111 |
| between_user_gamma_std | 192 | 0 | 0.0105806 | 0.00805775 | 0.0107634 | 0.0467292 |
| within_user_beta_std | 192 | 0.00387721 | 0.0732349 | 0.0654738 | 0.0612289 | 0.311204 |
| between_user_beta_std | 192 | 0 | 0.0109495 | 0.0101219 | 0.0104648 | 0.042924 |

First 10 rows:

| k | feature_channel | within_user_gamma_std | between_user_gamma_std | within_user_beta_std | between_user_beta_std |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.0140385 | 0 | 0.0206334 | 0 |
| 0 | 1 | 0.015575 | 0 | 0.0267555 | 0 |
| 0 | 2 | 0.03977 | 0 | 0.0189221 | 0 |
| 0 | 3 | 0.0247171 | 0 | 0.00387721 | 0 |
| 0 | 4 | 0.00952176 | 0 | 0.0150798 | 0 |
| 0 | 5 | 0.0246108 | 0 | 0.0140278 | 0 |
| 0 | 6 | 0.0302074 | 0 | 0.0121381 | 0 |
| 0 | 7 | 0.0257453 | 0 | 0.0139842 | 0 |
| 0 | 8 | 0.0124146 | 0 | 0.0276393 | 0 |
| 0 | 9 | 0.0223461 | 0 | 0.02994 | 0 |
### `results/film_personalization/full_run_v1/analysis/probability_diagnostics.csv`

- Shape: `900 rows x 9 columns`
- Missing cells: `0`
- Columns: `meta_seed, subject_id, sampling_seed, k, model, class, mean_probability, median_probability, windows`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| meta_seed | 900 | 42 | 43 | 43 | 0.816951 | 44 |
| sampling_seed | 900 | 17 | 53.4 | 47 | 29.8266 | 101 |
| k | 900 | 0 | 10.3333 | 1 | 13.9202 | 30 |
| mean_probability | 900 | 4.67672e-05 | 0.397795 | 0.00804647 | 0.481419 | 0.998959 |
| median_probability | 900 | 5.92717e-06 | 0.400087 | 0.000424151 | 0.489972 | 0.999993 |
| windows | 900 | 300 | 1344.7 | 1277.5 | 950.557 | 2423 |

First 10 rows:

| meta_seed | subject_id | sampling_seed | k | model | class | mean_probability | median_probability | windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | SA02 | 17 | 0 | Global | ADL | 0.00386878 | 6.36937e-05 | 2423 |
| 42 | SA02 | 17 | 0 | Global | Fall | 0.976517 | 0.999816 | 300 |
| 42 | SA02 | 17 | 0 | FiLM | ADL | 0.00564637 | 9.40259e-05 | 2423 |
| 42 | SA02 | 17 | 0 | FiLM | Fall | 0.986258 | 0.999921 | 300 |
| 42 | SA02 | 17 | 1 | Global | ADL | 0.00386878 | 6.36937e-05 | 2423 |
| 42 | SA02 | 17 | 1 | Global | Fall | 0.976517 | 0.999816 | 300 |
| 42 | SA02 | 17 | 1 | FiLM | ADL | 0.000945012 | 5.92717e-06 | 2423 |
| 42 | SA02 | 17 | 1 | FiLM | Fall | 0.885117 | 0.998586 | 300 |
| 42 | SA02 | 17 | 30 | Global | ADL | 0.00386878 | 6.36937e-05 | 2423 |
| 42 | SA02 | 17 | 30 | Global | Fall | 0.976517 | 0.999816 | 300 |
### `results/film_personalization/full_run_v1/analysis/real_test_channel_shifts.csv`

- Shape: `36 rows x 8 columns`
- Missing cells: `0`
- Columns: `subject_id, channel_index, train_mean, train_std, test_subject_mean, mean_shift_in_train_sd, test_subject_std, std_ratio_to_train`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| channel_index | 36 | 0 | 2.5 | 2.5 | 1.73205 | 5 |
| train_mean | 36 | -0.0322958 | -0.00755339 | -0.000966431 | 0.0120701 | 0.000524308 |
| train_std | 36 | 0.987466 | 1.03564 | 1.03956 | 0.0355027 | 1.07233 |
| test_subject_mean | 36 | -0.326296 | -0.00761222 | -0.00577664 | 0.180345 | 0.542896 |
| mean_shift_in_train_sd | 36 | -0.297732 | 0.000171502 | -0.00334846 | 0.182996 | 0.582493 |
| test_subject_std | 36 | 0.680676 | 1.03155 | 1.02704 | 0.18325 | 1.5847 |
| std_ratio_to_train | 36 | 0.677423 | 0.994161 | 0.992409 | 0.160546 | 1.48084 |

First 10 rows:

| subject_id | channel_index | train_mean | train_std | test_subject_mean | mean_shift_in_train_sd | test_subject_std | std_ratio_to_train |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SA02 | 0 | 0.000452647 | 1.0048 | -0.000939409 | -0.0013854 | 1.04514 | 1.04014 |
| SA02 | 1 | -0.0120686 | 1.01205 | 0.00817099 | 0.0199986 | 1.04148 | 1.02908 |
| SA02 | 2 | -0.0322958 | 0.987466 | 0.542896 | 0.582493 | 0.895184 | 0.906547 |
| SA02 | 3 | 0.000524308 | 1.07014 | 0.137332 | 0.127841 | 1.3031 | 1.21769 |
| SA02 | 4 | -0.00172253 | 1.06707 | -0.0672086 | -0.0613698 | 1.0994 | 1.0303 |
| SA02 | 5 | -0.000210335 | 1.07233 | 0.00546265 | 0.00529032 | 1.35848 | 1.26684 |
| SA03 | 0 | 0.000452647 | 1.0048 | -0.0504273 | -0.0506369 | 0.987983 | 0.983262 |
| SA03 | 1 | -0.0120686 | 1.01205 | -0.0463478 | -0.033871 | 1.0668 | 1.0541 |
| SA03 | 2 | -0.0322958 | 0.987466 | -0.326296 | -0.297732 | 0.980592 | 0.993039 |
| SA03 | 3 | 0.000524308 | 1.07014 | -0.101643 | -0.0954712 | 1.09928 | 1.02724 |
### `results/film_personalization/full_run_v1/analysis/test_metric_summary.csv`

- Shape: `100 rows x 9 columns`
- Missing cells: `0`
- Columns: `model, k, scope, metric, validation_calibrated_mean, validation_calibrated_std, observations, reference_0_25_mean, reference_0_25_std`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 100 | 0 | 9.2 | 5 | 11.0353 | 30 |
| validation_calibrated_mean | 100 | 0.973825 | 3.70163 | 0.998748 | 5.46537 | 16.6444 |
| validation_calibrated_std | 100 | 3.58509e-05 | 2.24059 | 0.00331338 | 4.54622 | 16.1594 |
| observations | 100 | 60 | 66 | 60 | 12.0605 | 90 |
| reference_0_25_mean | 100 | 0.951388 | 4.85004 | 0.998748 | 8.94482 | 44.6523 |
| reference_0_25_std | 100 | 3.58509e-05 | 3.38826 | 0.00331905 | 8.71883 | 57.8559 |

First 10 rows:

| model | k | scope | metric | validation_calibrated_mean | validation_calibrated_std | observations | reference_0_25_mean | reference_0_25_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FiLM | 0 | per_user | f1 | 0.980836 | 0.00447724 | 60 | 0.967143 | 0.00782245 |
| FiLM | 0 | per_user | f2 | 0.988293 | 0.00386527 | 60 | 0.985441 | 0.00349955 |
| FiLM | 0 | per_user | false_alarms_per_hour | 11.948 | 9.12413 | 90 | 24.5151 | 18.9974 |
| FiLM | 0 | per_user | auroc | 0.99985 | 3.62979e-05 | 60 | 0.99985 | 3.62979e-05 |
| FiLM | 0 | per_user | auprc | 0.998794 | 0.000313903 | 60 | 0.998794 | 0.000313903 |
| FiLM | 1 | per_user | f1 | 0.973825 | 0.010922 | 60 | 0.951388 | 0.0367167 |
| FiLM | 1 | per_user | f2 | 0.983961 | 0.0107883 | 60 | 0.976791 | 0.0182968 |
| FiLM | 1 | per_user | false_alarms_per_hour | 16.6444 | 16.1594 | 90 | 44.6523 | 57.8559 |
| FiLM | 1 | per_user | auroc | 0.999815 | 9.65022e-05 | 60 | 0.999815 | 9.65022e-05 |
| FiLM | 1 | per_user | auprc | 0.998469 | 0.000837436 | 60 | 0.998469 | 0.000837436 |
### `results/film_personalization/full_run_v1/analysis/test_threshold_comparison.csv`

- Shape: `1,800 rows x 20 columns`
- Missing cells: `4,800`
- Columns: `meta_seed, model, k, scope, subject_id, calibration_sampling_seed, validation_threshold, rows_validation_calibrated, selected_f1, selected_f2, selected_false_alarms_per_hour, selected_auroc, selected_auprc, reference_0_25_f1, reference_0_25_f2, reference_0_25_false_alarms_per_hour, reference_0_25_auroc, reference_0_25_auprc, fall_windows, adl_windows`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| meta_seed | 1800 | 42 | 43 | 43 | 0.816723 | 44 |
| k | 1800 | 0 | 9.2 | 5 | 10.983 | 30 |
| calibration_sampling_seed | 1800 | 17 | 53.4 | 47 | 29.8183 | 101 |
| validation_threshold | 1800 | 0.15 | 0.369167 | 0.25 | 0.210235 | 0.9 |
| selected_f1 | 1200 | 0.940439 | 0.978125 | 0.977049 | 0.00513025 | 0.990099 |
| selected_f2 | 1200 | 0.935179 | 0.987941 | 0.987409 | 0.00420416 | 0.996016 |
| selected_false_alarms_per_hour | 1800 | 0 | 14.5435 | 18.572 | 11.2689 | 70.5737 |
| selected_auroc | 1200 | 0.999558 | 0.999842 | 0.99984 | 4.89611e-05 | 0.99997 |
| selected_auprc | 1200 | 0.995805 | 0.998718 | 0.998645 | 0.000433662 | 0.999763 |
| reference_0_25_f1 | 1200 | 0.767263 | 0.97217 | 0.97561 | 0.0154031 | 0.985222 |
| reference_0_25_f2 | 1200 | 0.891795 | 0.98615 | 0.986755 | 0.00687529 | 0.994036 |
| reference_0_25_false_alarms_per_hour | 1800 | 0 | 20.2933 | 20.4292 | 22.9585 | 338.011 |
| reference_0_25_auroc | 1200 | 0.999558 | 0.999842 | 0.99984 | 4.89611e-05 | 0.99997 |
| reference_0_25_auprc | 1200 | 0.995805 | 0.998718 | 0.998645 | 0.000433662 | 0.999763 |
| fall_windows | 1800 | 0 | 200 | 300 | 141.461 | 300 |
| adl_windows | 1800 | 1277 | 2041.17 | 2423 | 540.144 | 2423 |

First 10 rows:

| meta_seed | model | k | scope | subject_id | calibration_sampling_seed | validation_threshold | rows_validation_calibrated | selected_f1 | selected_f2 | selected_false_alarms_per_hour | selected_auroc | selected_auprc | reference_0_25_f1 | reference_0_25_f2 | reference_0_25_false_alarms_per_hour | reference_0_25_auroc | reference_0_25_auprc | fall_windows | adl_windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | Global | 0 | per_user | SA02 | 17 | 0.25 | True | 0.97545 | 0.986102 | 24.1436 | 0.99979 | 0.998288 | 0.97545 | 0.986102 | 24.1436 | 0.99979 | 0.998288 | 300 | 2423 |
| 42 | Last-layer FT (ADL-only) | 0 | per_user | SA02 | 17 | 0.25 | True | 0.97545 | 0.986102 | 24.1436 | 0.99979 | 0.998288 | 0.97545 | 0.986102 | 24.1436 | 0.99979 | 0.998288 | 300 | 2423 |
| 42 | Full FT (ADL-only) | 0 | per_user | SA02 | 17 | 0.25 | True | 0.97545 | 0.986102 | 24.1436 | 0.99979 | 0.998288 | 0.97545 | 0.986102 | 24.1436 | 0.99979 | 0.998288 | 300 | 2423 |
| 42 | FiLM | 0 | per_user | SA02 | 17 | 0.6 | True | 0.976974 | 0.984748 | 20.4292 | 0.999799 | 0.998374 | 0.972358 | 0.986799 | 29.7152 | 0.999799 | 0.998374 | 300 | 2423 |
| 42 | Global | 1 | per_user | SA02 | 17 | 0.25 | True | 0.97545 | 0.986102 | 24.1436 | 0.99979 | 0.998288 | 0.97545 | 0.986102 | 24.1436 | 0.99979 | 0.998288 | 300 | 2423 |
| 42 | Last-layer FT (ADL-only) | 1 | per_user | SA02 | 17 | 0.25 | True | 0.977049 | 0.986755 | 22.2864 | 0.999791 | 0.998298 | 0.977049 | 0.986755 | 22.2864 | 0.999791 | 0.998298 | 300 | 2423 |
| 42 | Full FT (ADL-only) | 1 | per_user | SA02 | 17 | 0.25 | True | 0.977049 | 0.986755 | 22.2864 | 0.999798 | 0.998357 | 0.977049 | 0.986755 | 22.2864 | 0.999798 | 0.998357 | 300 | 2423 |
| 42 | FiLM | 1 | per_user | SA02 | 17 | 0.2 | True | 0.953528 | 0.935179 | 7.42881 | 0.999578 | 0.996651 | 0.955172 | 0.935811 | 5.57161 | 0.999578 | 0.996651 | 300 | 2423 |
| 42 | Global | 5 | per_user | SA02 | 17 | 0.25 | True | 0.97545 | 0.986102 | 24.1436 | 0.99979 | 0.998288 | 0.97545 | 0.986102 | 24.1436 | 0.99979 | 0.998288 | 300 | 2423 |
| 42 | Last-layer FT (ADL-only) | 5 | per_user | SA02 | 17 | 0.3 | True | 0.978654 | 0.987409 | 20.4292 | 0.999794 | 0.998321 | 0.977049 | 0.986755 | 22.2864 | 0.999794 | 0.998321 | 300 | 2423 |
### `results/film_personalization/full_run_v1/analysis/training_curve_summary.csv`

- Shape: `3 rows x 8 columns`
- Missing cells: `0`
- Columns: `meta_seed, epochs_run, configured_epochs, best_epoch, best_validation_loss, last_validation_loss, early_stopped, validation_loss_after_best`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| meta_seed | 3 | 42 | 43 | 43 | 1 | 44 |
| epochs_run | 3 | 9 | 14 | 13 | 5.56776 | 20 |
| configured_epochs | 3 | 30 | 30 | 30 | 0 | 30 |
| best_epoch | 3 | 4 | 9 | 8 | 5.56776 | 15 |
| best_validation_loss | 3 | 0.0114084 | 0.0173245 | 0.0193105 | 0.00521483 | 0.0212544 |
| last_validation_loss | 3 | 0.0385472 | 0.0831705 | 0.0947908 | 0.0400966 | 0.116174 |

First 10 rows:

| meta_seed | epochs_run | configured_epochs | best_epoch | best_validation_loss | last_validation_loss | early_stopped | validation_loss_after_best |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 9 | 30 | 4 | 0.0114084 | 0.0385472 | True | True |
| 43 | 13 | 30 | 8 | 0.0212544 | 0.116174 | True | True |
| 44 | 20 | 30 | 15 | 0.0193105 | 0.0947908 | True | True |
### `results/film_personalization/full_run_v1/analysis/training_curves.csv`

- Shape: `42 rows x 5 columns`
- Missing cells: `0`
- Columns: `meta_seed, epoch, train_loss, val_loss, epoch_seconds`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| meta_seed | 42 | 42 | 43.2619 | 43 | 0.798154 | 44 |
| epoch | 42 | 1 | 8.2381 | 7.5 | 5.23501 | 20 |
| train_loss | 42 | 0.0106318 | 0.0761948 | 0.0776773 | 0.0440087 | 0.195672 |
| val_loss | 42 | 0.0114084 | 0.0898855 | 0.0814648 | 0.0564559 | 0.245065 |
| epoch_seconds | 42 | 2.99985 | 3.80718 | 3.53107 | 0.932599 | 6.7111 |

First 10 rows:

| meta_seed | epoch | train_loss | val_loss | epoch_seconds |
| --- | --- | --- | --- | --- |
| 42 | 1 | 0.145981 | 0.245065 | 3.84597 |
| 42 | 2 | 0.0378917 | 0.178698 | 3.44672 |
| 42 | 3 | 0.0437488 | 0.027715 | 3.25525 |
| 42 | 4 | 0.082741 | 0.0114084 | 3.15391 |
| 42 | 5 | 0.0616718 | 0.0928408 | 3.26851 |
| 42 | 6 | 0.0701789 | 0.02297 | 3.17922 |
| 42 | 7 | 0.0781247 | 0.0617052 | 3.12358 |
| 42 | 8 | 0.195672 | 0.0862091 | 3.30763 |
| 42 | 9 | 0.0949704 | 0.0385472 | 4.35129 |
| 43 | 1 | 0.1544 | 0.177773 | 3.2392 |
### `results/film_personalization/full_run_v1/analysis/validation_thresholds.csv`

- Shape: `60 rows x 10 columns`
- Missing cells: `0`
- Columns: `meta_seed, model, k, validation_threshold, validation_f1, validation_f2, validation_false_alarms_per_hour, threshold_selection, validation_sampling_seeds, rows_validation_calibrated`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| meta_seed | 60 | 42 | 43 | 43 | 0.823387 | 44 |
| k | 60 | 0 | 9.2 | 5 | 11.0726 | 30 |
| validation_threshold | 60 | 0.15 | 0.369167 | 0.25 | 0.211951 | 0.9 |
| validation_f1 | 60 | 0.959059 | 0.973401 | 0.973238 | 0.00268164 | 0.977725 |
| validation_f2 | 60 | 0.973791 | 0.984621 | 0.984788 | 0.00166809 | 0.986424 |
| validation_false_alarms_per_hour | 60 | 18.547 | 23.4302 | 23.7213 | 2.60469 | 34.2044 |

First 10 rows:

| meta_seed | model | k | validation_threshold | validation_f1 | validation_f2 | validation_false_alarms_per_hour | threshold_selection | validation_sampling_seeds | rows_validation_calibrated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | Global | 0 | 0.25 | 0.973203 | 0.984788 | 23.8557 | pooled validation users/seeds; max F2, then lower FA/hr | 17;31;47;71;101 | True |
| 42 | Last-layer FT (ADL-only) | 0 | 0.25 | 0.973203 | 0.984788 | 23.8557 | pooled validation users/seeds; max F2, then lower FA/hr | 17;31;47;71;101 | True |
| 42 | Full FT (ADL-only) | 0 | 0.25 | 0.973203 | 0.984788 | 23.8557 | pooled validation users/seeds; max F2, then lower FA/hr | 17;31;47;71;101 | True |
| 42 | FiLM | 0 | 0.6 | 0.977725 | 0.98624 | 18.7486 | pooled validation users/seeds; max F2, then lower FA/hr | 17;31;47;71;101 | True |
| 42 | Global | 1 | 0.25 | 0.973203 | 0.984788 | 23.8557 | pooled validation users/seeds; max F2, then lower FA/hr | 17;31;47;71;101 | True |
| 42 | Last-layer FT (ADL-only) | 1 | 0.25 | 0.973454 | 0.984812 | 23.5197 | pooled validation users/seeds; max F2, then lower FA/hr | 17;31;47;71;101 | True |
| 42 | Full FT (ADL-only) | 1 | 0.25 | 0.973648 | 0.984971 | 23.3854 | pooled validation users/seeds; max F2, then lower FA/hr | 17;31;47;71;101 | True |
| 42 | FiLM | 1 | 0.2 | 0.959059 | 0.973791 | 34.2044 | pooled validation users/seeds; max F2, then lower FA/hr | 17;31;47;71;101 | True |
| 42 | Global | 5 | 0.25 | 0.973203 | 0.984788 | 23.8557 | pooled validation users/seeds; max F2, then lower FA/hr | 17;31;47;71;101 | True |
| 42 | Last-layer FT (ADL-only) | 5 | 0.3 | 0.975082 | 0.984768 | 21.1678 | pooled validation users/seeds; max F2, then lower FA/hr | 17;31;47;71;101 | True |
### `results/film_personalization/full_run_v1/logs/run_summary.csv`

- Shape: `3 rows x 8 columns`
- Missing cells: `0`
- Columns: `seed, epochs_run, last_layer_lr, last_layer_epochs, full_ft_lr, full_ft_epochs, film_validation_threshold, elapsed_seconds`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| seed | 3 | 42 | 43 | 43 | 1 | 44 |
| epochs_run | 3 | 9 | 14 | 13 | 5.56776 | 20 |
| last_layer_lr | 3 | 0.01 | 0.01 | 0.01 | 0 | 0.01 |
| last_layer_epochs | 3 | 1 | 1 | 1 | 0 | 1 |
| full_ft_lr | 3 | 0.0001 | 0.0001 | 0.0001 | 0 | 0.0001 |
| full_ft_epochs | 3 | 1 | 2.33333 | 1 | 2.3094 | 5 |
| film_validation_threshold | 3 | 0.8 | 0.833333 | 0.85 | 0.0288675 | 0.85 |
| elapsed_seconds | 3 | 957.822 | 980.769 | 988.313 | 20.2578 | 996.173 |

First 10 rows:

| seed | epochs_run | last_layer_lr | last_layer_epochs | full_ft_lr | full_ft_epochs | film_validation_threshold | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 9 | 0.01 | 1 | 0.0001 | 1 | 0.8 | 996.173 |
| 43 | 13 | 0.01 | 1 | 0.0001 | 1 | 0.85 | 988.313 |
| 44 | 20 | 0.01 | 1 | 0.0001 | 5 | 0.85 | 957.822 |
### `results/film_personalization/full_run_v1/seed42/deployment_size.csv`

- Shape: `1 rows x 7 columns`
- Missing cells: `0`
- Columns: `base_tflite_bytes, film_tflite_bytes, tflite_overhead_bytes, user_affine_bytes, channels, base_tflite_max_abs_error, film_tflite_max_abs_error`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| base_tflite_bytes | 1 | 195164 | 195164 | 195164 |  | 195164 |
| film_tflite_bytes | 1 | 197940 | 197940 | 197940 |  | 197940 |
| tflite_overhead_bytes | 1 | 2776 | 2776 | 2776 |  | 2776 |
| user_affine_bytes | 1 | 512 | 512 | 512 |  | 512 |
| channels | 1 | 64 | 64 | 64 |  | 64 |
| base_tflite_max_abs_error | 1 | 5.96046e-08 | 5.96046e-08 | 5.96046e-08 |  | 5.96046e-08 |
| film_tflite_max_abs_error | 1 | 0 | 0 | 0 |  | 0 |

First 10 rows:

| base_tflite_bytes | film_tflite_bytes | tflite_overhead_bytes | user_affine_bytes | channels | base_tflite_max_abs_error | film_tflite_max_abs_error |
| --- | --- | --- | --- | --- | --- | --- |
| 195164 | 197940 | 2776 | 512 | 64 | 5.96046e-08 | 0 |
### `results/film_personalization/full_run_v1/seed42/experiment_config.csv`

- Shape: `23 rows x 2 columns`
- Missing cells: `0`
- Columns: `key, value`

Numeric statistics:

_No rows._

First 10 rows:

| key | value |
| --- | --- |
| seed | 42 |
| calibration_seeds | [17, 31, 47, 71, 101] |
| calibration_sizes | [0, 1, 5, 10, 30] |
| epochs | 30 |
| episodes_per_epoch | 30 |
| validation_episodes | 12 |
| patience | 5 |
| fine_tune_epochs | 3 |
| fine_tune_tuning_json | {"configurations_per_method": 12, "methods": {"full_model": {"configurations_tried": [{"epochs": 1, "learning_rate": 0.0001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.0001, "validation_mean_f2": 0.9845051056462616, "validation_mean_false_alarms_per_hour": 21.10522022838499, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.0001, "validation_mean_f2": 0.9839675638082447, "validation_mean_false_alarms_per_hour": 19.88172920065253, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.001, "validation_mean_f2": 0.9835303182010312, "validation_mean_false_alarms_per_hour": 18.35236541598695, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.001, "validation_mean_f2": 0.9724184578678468, "validation_mean_false_alarms_per_hour": 11.623164763458403, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.001, "validation_mean_f2": 0.9374280600532447, "validation_mean_false_alarms_per_hour": 8.258564437194126, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.01, "validation_mean_f2": 0.8301107802719189, "validation_mean_false_alarms_per_hour": 6.423327895595432, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.01, "validation_mean_f2": 0.11069402736069403, "validation_mean_false_alarms_per_hour": 0.30587275693311583, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.01, "validation_mean_f2": 0.025182778229082048, "validation_mean_false_alarms_per_hour": 0.0, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.1, "validation_mean_f2": 0.19073203492276697, "validation_mean_false_alarms_per_hour": 1.529363784665579, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.1, "validation_mean_f2": 0.0, "validation_mean_false_alarms_per_hour": 0.0, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.1, "validation_mean_f2": 0.0, "validation_mean_false_alarms_per_hour": 0.0, "validation_subjects_with_falls": 5}], "selected_config": {"epochs": 1, "learning_rate": 0.0001}, "selection_metrics": {"validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342}}, "last_layer": {"configurations_tried": [{"epochs": 1, "learning_rate": 0.0001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.0001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.0001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.001, "validation_mean_f2": 0.9847754325282876, "validation_mean_false_alarms_per_hour": 21.716965742251222, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.001, "validation_mean_f2": 0.984243790398388, "validation_mean_false_alarms_per_hour": 21.716965742251222, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.01, "validation_mean_f2": 0.9847754325282876, "validation_mean_false_alarms_per_hour": 21.716965742251222, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.01, "validation_mean_f2": 0.9841006072335541, "validation_mean_false_alarms_per_hour": 20.799347471451878, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.01, "validation_mean_f2": 0.9835332603844064, "validation_mean_false_alarms_per_hour": 18.35236541598695, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.1, "validation_mean_f2": 0.9839535462369682, "validation_mean_false_alarms_per_hour": 19.88172920065253, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.1, "validation_mean_f2": 0.9817591744756253, "validation_mean_false_alarms_per_hour": 13.764274061990212, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.1, "validation_mean_f2": 0.9772855558839574, "validation_mean_false_alarms_per_hour": 9.176182707993474, "validation_subjects_with_falls": 5}], "selected_config": {"epochs": 1, "learning_rate": 0.01}, "selection_metrics": {"validation_mean_f2": 0.9847754325282876, "validation_mean_false_alarms_per_hour": 21.716965742251222}}}, "tie_breaking": ["higher validation mean F2", "lower validation mean false alarms per hour", "fewer epochs", "lower learning rate"], "validation_calibration_policy": "one ADL calibration window per validation subject", "validation_threshold": 0.25} |
| timevae_epochs | 0 |
### `results/film_personalization/full_run_v1/seed42/fine_tune_probability_shift_summary.csv`

- Shape: `8 rows x 8 columns`
- Missing cells: `0`
- Columns: `evaluation, method, label_group, windows, signed_mean_probability_shift, absolute_mean_probability_shift, threshold_crossings_vs_global, subject_seed_rows`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| windows | 8 | 6000 | 37231.2 | 35117.5 | 32097.6 | 72690 |
| signed_mean_probability_shift | 8 | -0.00082462 | -0.000150862 | -8.74094e-05 | 0.000300199 | 0.000227352 |
| absolute_mean_probability_shift | 8 | 5.48846e-05 | 0.000417035 | 0.000161906 | 0.00052839 | 0.00155126 |
| threshold_crossings_vs_global | 8 | 0 | 8.75 | 7.5 | 7.66718 | 21 |
| subject_seed_rows | 8 | 20 | 27.5 | 30 | 4.6291 | 30 |

First 10 rows:

| evaluation | method | label_group | windows | signed_mean_probability_shift | absolute_mean_probability_shift | threshold_crossings_vs_global | subject_seed_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | full_model | ADL | 61235 | -3.27193e-05 | 5.48846e-05 | 5 | 30 |
| real_heldout | full_model | Fall | 6000 | -0.000134168 | 0.000433951 | 0 | 20 |
| real_heldout | last_layer | ADL | 61235 | -5.98713e-05 | 5.98713e-05 | 5 | 30 |
| real_heldout | last_layer | Fall | 6000 | -0.000208052 | 0.000208052 | 0 | 20 |
| simulated_users | full_model | ADL | 72690 | -8.69323e-05 | 0.000115759 | 21 | 30 |
| simulated_users | full_model | Fall | 9000 | 0.000227352 | 0.00155126 | 12 | 30 |
| simulated_users | last_layer | ADL | 72690 | -8.78865e-05 | 8.78865e-05 | 10 | 30 |
| simulated_users | last_layer | Fall | 9000 | -0.00082462 | 0.00082462 | 17 | 30 |
### `results/film_personalization/full_run_v1/seed42/fine_tune_probability_shifts.csv`

- Shape: `220 rows x 10 columns`
- Missing cells: `0`
- Columns: `evaluation, profile, subject_id, seed, method, label_group, windows, signed_mean_probability_shift, absolute_mean_probability_shift, threshold_crossings_vs_global`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| seed | 220 | 17 | 53.4 | 47 | 29.878 | 101 |
| windows | 220 | 300 | 1353.86 | 1277.5 | 1014.7 | 2423 |
| signed_mean_probability_shift | 220 | -0.00364983 | -0.000149022 | -7.63559e-05 | 0.000828161 | 0.00653764 |
| absolute_mean_probability_shift | 220 | 7.08022e-06 | 0.000425766 | 9.78875e-05 | 0.00087015 | 0.00655918 |
| threshold_crossings_vs_global | 220 | 0 | 0.318182 | 0 | 0.610888 | 3 |

First 10 rows:

| evaluation | profile | subject_id | seed | method | label_group | windows | signed_mean_probability_shift | absolute_mean_probability_shift | threshold_crossings_vs_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | Real | SA02 | 17 | last_layer | ADL | 2423 | -8.79983e-05 | 8.79983e-05 | 1 |
| real_heldout | Real | SA02 | 17 | last_layer | Fall | 300 | -0.000290356 | 0.000290356 | 0 |
| real_heldout | Real | SA02 | 17 | full_model | ADL | 2423 | -6.10596e-05 | 9.99363e-05 | 1 |
| real_heldout | Real | SA02 | 17 | full_model | Fall | 300 | -6.98936e-05 | 0.00060224 | 0 |
| real_heldout | Real | SA02 | 31 | last_layer | ADL | 2423 | -6.86674e-05 | 6.86674e-05 | 1 |
| real_heldout | Real | SA02 | 31 | last_layer | Fall | 300 | -0.000217449 | 0.000217449 | 0 |
| real_heldout | Real | SA02 | 31 | full_model | ADL | 2423 | -7.14691e-05 | 9.2182e-05 | 1 |
| real_heldout | Real | SA02 | 31 | full_model | Fall | 300 | -0.000126161 | 0.000466551 | 0 |
| real_heldout | Real | SA02 | 47 | last_layer | ADL | 2423 | -7.72107e-05 | 7.72107e-05 | 1 |
| real_heldout | Real | SA02 | 47 | last_layer | Fall | 300 | -0.000267755 | 0.000267755 | 0 |
### `results/film_personalization/full_run_v1/seed42/meta_training_history.csv`

- Shape: `9 rows x 4 columns`
- Missing cells: `0`
- Columns: `epoch, train_loss, val_loss, epoch_seconds`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| epoch | 9 | 1 | 5 | 5 | 2.73861 | 9 |
| train_loss | 9 | 0.0378917 | 0.0901089 | 0.0781247 | 0.0507057 | 0.195672 |
| val_loss | 9 | 0.0114084 | 0.0850176 | 0.0617052 | 0.0788261 | 0.245065 |
| epoch_seconds | 9 | 3.12358 | 3.4369 | 3.26851 | 0.407164 | 4.35129 |

First 10 rows:

| epoch | train_loss | val_loss | epoch_seconds |
| --- | --- | --- | --- |
| 1 | 0.145981 | 0.245065 | 3.84597 |
| 2 | 0.0378917 | 0.178698 | 3.44672 |
| 3 | 0.0437488 | 0.027715 | 3.25525 |
| 4 | 0.082741 | 0.0114084 | 3.15391 |
| 5 | 0.0616718 | 0.0928408 | 3.26851 |
| 6 | 0.0701789 | 0.02297 | 3.17922 |
| 7 | 0.0781247 | 0.0617052 | 3.12358 |
| 8 | 0.195672 | 0.0862091 | 3.30763 |
| 9 | 0.0949704 | 0.0385472 | 4.35129 |
### `results/film_personalization/full_run_v1/seed42/metrics_summary.csv`

- Shape: `245 rows x 17 columns`
- Missing cells: `0`
- Columns: `evaluation, profile, model, k, threshold, accuracy_mean, accuracy_std, precision_mean, precision_std, recall_mean, recall_std, f1_mean, f1_std, f2_mean, f2_std, false_alarms_per_hour_mean, false_alarms_per_hour_std`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 245 | 0 | 9.2 | 5 | 11.0025 | 30 |
| threshold | 245 | 0.25 | 0.328571 | 0.25 | 0.192854 | 0.8 |
| accuracy_mean | 245 | 0.973926 | 0.989721 | 0.98993 | 0.00564711 | 0.997699 |
| accuracy_std | 245 | 0 | 0.000995427 | 0.000492707 | 0.00131018 | 0.00901951 |
| precision_mean | 245 | 0.874645 | 0.962616 | 0.962539 | 0.0305077 | 1 |
| precision_std | 245 | 0 | 0.00464894 | 0.00293001 | 0.00797414 | 0.0734637 |
| recall_mean | 245 | 0.766667 | 0.944048 | 0.990667 | 0.0630969 | 1 |
| recall_std | 245 | 0 | 0.00523166 | 0.001838 | 0.0109637 | 0.0900432 |
| f1_mean | 245 | 0.86629 | 0.951211 | 0.952001 | 0.0273307 | 0.984444 |
| f1_std | 245 | 0 | 0.00389941 | 0.0021513 | 0.00566099 | 0.0498571 |
| f2_mean | 245 | 0.803634 | 0.94644 | 0.976282 | 0.0481858 | 0.988365 |
| f2_std | 245 | 0 | 0.00453612 | 0.00241882 | 0.00844388 | 0.07543 |
| false_alarms_per_hour_mean | 245 | 0 | 20.9897 | 16.7148 | 19.2148 | 79.8597 |
| false_alarms_per_hour_std | 245 | 0 | 3.58538 | 1.66113 | 6.04006 | 54.8048 |

First 10 rows:

| evaluation | profile | model | k | threshold | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | f2_mean | f2_std | false_alarms_per_hour_mean | false_alarms_per_hour_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | Real | FiLM (canonical threshold) | 0 | 0.25 | 0.99574 | 0.00321546 | 0.94737 | 0.00904719 | 0.9975 | 0.00148087 | 0.971771 | 0.0051839 | 0.987042 | 0.00276879 | 20.6149 | 15.5105 |
| real_heldout | Real | FiLM (canonical threshold) | 1 | 0.25 | 0.995642 | 0.00354711 | 0.955743 | 0.020181 | 0.987167 | 0.0243578 | 0.970786 | 0.00931583 | 0.980449 | 0.0170176 | 17.272 | 16.0071 |
| real_heldout | Real | FiLM (canonical threshold) | 5 | 0.25 | 0.995373 | 0.00361061 | 0.942458 | 0.0138807 | 0.998167 | 0.0029568 | 0.969459 | 0.00737407 | 0.986468 | 0.00366656 | 22.7198 | 17.8115 |
| real_heldout | Real | FiLM (canonical threshold) | 10 | 0.25 | 0.99525 | 0.00364692 | 0.94093 | 0.0122774 | 0.998167 | 0.0029568 | 0.968663 | 0.00673285 | 0.986142 | 0.00364511 | 23.3388 | 17.9181 |
| real_heldout | Real | FiLM (canonical threshold) | 30 | 0.25 | 0.995275 | 0.00361464 | 0.941067 | 0.0115262 | 0.998333 | 0.00296174 | 0.968821 | 0.00649265 | 0.986306 | 0.00367691 | 23.2769 | 17.7584 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 0 | 0.8 | 0.997552 | 0.00190051 | 0.979356 | 0.00493112 | 0.9875 | 0.00505814 | 0.983402 | 0.00399863 | 0.985855 | 0.00442023 | 7.73834 | 6.01473 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 1 | 0.8 | 0.996389 | 0.00428934 | 0.986882 | 0.010308 | 0.964 | 0.0434021 | 0.974667 | 0.0207856 | 0.968099 | 0.0345611 | 4.89063 | 5.90295 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 5 | 0.8 | 0.997625 | 0.00187191 | 0.978296 | 0.00581946 | 0.989667 | 0.00779113 | 0.983921 | 0.00431783 | 0.987354 | 0.00607445 | 8.17169 | 6.48142 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 10 | 0.8 | 0.997674 | 0.00176607 | 0.977983 | 0.00499531 | 0.990667 | 0.00514128 | 0.984269 | 0.00316132 | 0.988094 | 0.0039953 | 8.2955 | 6.41008 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 30 | 0.8 | 0.997699 | 0.00178075 | 0.977989 | 0.00583467 | 0.991 | 0.00288168 | 0.984444 | 0.00363229 | 0.988365 | 0.00287212 | 8.2955 | 6.55683 |
### `results/film_personalization/full_run_v1/seed42/paired_wilcoxon.csv`

- Shape: `10 rows x 6 columns`
- Missing cells: `0`
- Columns: `evaluation, metric, comparison, n_subjects, wilcoxon_statistic, p_value`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| n_subjects | 10 | 4 | 4 | 4 | 0 | 4 |
| wilcoxon_statistic | 10 | 0 | 1.1 | 1 | 1.19722 | 3 |
| p_value | 10 | 0.125 | 0.275 | 0.25 | 0.174801 | 0.625 |

First 10 rows:

| evaluation | metric | comparison | n_subjects | wilcoxon_statistic | p_value |
| --- | --- | --- | --- | --- | --- |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (+2 falls) | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (ADL-only) | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Global | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (+2 falls) | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (ADL-only) | 4 | 0 | 0.125 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (+2 falls) | 4 | 3 | 0.625 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (ADL-only) | 4 | 2 | 0.375 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Global | 4 | 2 | 0.375 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (+2 falls) | 4 | 2 | 0.375 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (ADL-only) | 4 | 2 | 0.375 |
### `results/film_personalization/full_run_v1/seed42/per_user_metrics.csv`

- Shape: `2,000 rows x 16 columns`
- Missing cells: `1,000`
- Columns: `evaluation, profile, subject_id, model, k, seed, threshold, accuracy, precision, recall, f1, f2, false_alarms_per_hour, false_positive_windows, adl_windows, fall_windows`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 2000 | 0 | 9.2 | 5 | 10.9827 | 30 |
| seed | 2000 | 17 | 53.4 | 47 | 29.8175 | 101 |
| threshold | 2000 | 0.25 | 0.3325 | 0.25 | 0.196438 | 0.8 |
| accuracy | 2000 | 0.969152 | 0.992253 | 0.994487 | 0.00579718 | 1 |
| precision | 1750 | 0.78628 | 0.962235 | 0.964286 | 0.0272439 | 1 |
| recall | 1750 | 0.723333 | 0.958921 | 0.993289 | 0.0584047 | 1 |
| f1 | 1750 | 0.837838 | 0.958984 | 0.973941 | 0.0264886 | 0.991625 |
| f2 | 1750 | 0.765162 | 0.958588 | 0.984748 | 0.0451141 | 0.994695 |
| false_alarms_per_hour | 2000 | 0 | 19.1812 | 18.572 | 17.6012 | 150.433 |
| false_positive_windows | 2000 | 0 | 10.328 | 10 | 9.47725 | 81 |
| adl_windows | 2000 | 1277 | 2279.81 | 2423 | 378.933 | 2423 |
| fall_windows | 2000 | 0 | 262 | 300 | 99.0551 | 300 |

First 10 rows:

| evaluation | profile | subject_id | model | k | seed | threshold | accuracy | precision | recall | f1 | f2 | false_alarms_per_hour | false_positive_windows | adl_windows | fall_windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | Real | SA02 | Global | 0 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Last-layer fine-tune (ADL-only) | 0 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Full fine-tune (ADL-only) | 0 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Last-layer fine-tune (+2 falls) | 0 | 17 | 0.25 | 0.994487 | 0.957929 | 0.993289 | 0.975288 | 0.986009 | 24.1436 | 13 | 2423 | 298 |
| real_heldout | Real | SA02 | Full fine-tune (+2 falls) | 0 | 17 | 0.25 | 0.994855 | 0.961039 | 0.993289 | 0.976898 | 0.986667 | 22.2864 | 12 | 2423 | 298 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 0 | 17 | 0.25 | 0.993757 | 0.949206 | 0.996667 | 0.972358 | 0.986799 | 29.7152 | 16 | 2423 | 300 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 0 | 17 | 0.8 | 0.99596 | 0.976898 | 0.986667 | 0.981758 | 0.984697 | 13.0004 | 7 | 2423 | 300 |
| real_heldout | Real | SA02 | Global | 1 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Last-layer fine-tune (ADL-only) | 1 | 17 | 0.25 | 0.994859 | 0.96129 | 0.993333 | 0.977049 | 0.986755 | 22.2864 | 12 | 2423 | 300 |
| real_heldout | Real | SA02 | Full fine-tune (ADL-only) | 1 | 17 | 0.25 | 0.994859 | 0.96129 | 0.993333 | 0.977049 | 0.986755 | 22.2864 | 12 | 2423 | 300 |
### `results/film_personalization/full_run_v1/seed42/per_user_seed_summary.csv`

- Shape: `400 rows x 24 columns`
- Missing cells: `400`
- Columns: `evaluation, profile, subject_id, model, k, threshold, accuracy_mean, accuracy_std, accuracy_count, precision_mean, precision_std, precision_count, recall_mean, recall_std, recall_count, f1_mean, f1_std, f1_count, f2_mean, f2_std, f2_count, false_alarms_per_hour_mean, false_alarms_per_hour_std, false_alarms_per_hour_count`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 400 | 0 | 9.2 | 5 | 10.9937 | 30 |
| threshold | 400 | 0.25 | 0.3325 | 0.25 | 0.196635 | 0.8 |
| accuracy_mean | 400 | 0.973926 | 0.992253 | 0.994487 | 0.00570813 | 1 |
| accuracy_std | 400 | 0 | 0.000529174 | 0.000201147 | 0.00104124 | 0.00901951 |
| accuracy_count | 400 | 5 | 5 | 5 | 0 | 5 |
| precision_mean | 350 | 0.874645 | 0.962235 | 0.964016 | 0.0263066 | 1 |
| precision_std | 350 | 0 | 0.00349476 | 0.00139091 | 0.00725383 | 0.0734637 |
| precision_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| recall_mean | 350 | 0.766667 | 0.958921 | 0.993289 | 0.0576838 | 1 |
| recall_std | 350 | 0 | 0.00367438 | 0 | 0.0100421 | 0.0900432 |
| recall_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| f1_mean | 350 | 0.86629 | 0.958984 | 0.97409 | 0.0259892 | 0.990033 |
| f1_std | 350 | 0 | 0.00280959 | 0.0013959 | 0.00518182 | 0.0498571 |
| f1_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| f2_mean | 350 | 0.803634 | 0.958588 | 0.984646 | 0.0445398 | 0.993377 |
| f2_std | 350 | 0 | 0.00322458 | 0.000717204 | 0.00773049 | 0.07543 |
| f2_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| false_alarms_per_hour_mean | 400 | 0 | 19.1812 | 18.7577 | 17.047 | 79.8597 |
| false_alarms_per_hour_std | 400 | 0 | 1.90077 | 0.415283 | 4.59948 | 54.8048 |
| false_alarms_per_hour_count | 400 | 5 | 5 | 5 | 0 | 5 |

First 10 rows:

| evaluation | profile | subject_id | model | k | threshold | accuracy_mean | accuracy_std | accuracy_count | precision_mean | precision_std | precision_count | recall_mean | recall_std | recall_count | f1_mean | f1_std | f1_count | f2_mean | f2_std | f2_count | false_alarms_per_hour_mean | false_alarms_per_hour_std | false_alarms_per_hour_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 0 | 0.25 | 0.993757 | 0 | 5 | 0.949206 | 0 | 5 | 0.996667 | 0 | 5 | 0.972358 | 0 | 5 | 0.986799 | 0 | 5 | 29.7152 | 0 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 1 | 0.25 | 0.991186 | 0.00137409 | 5 | 0.964194 | 0.0314718 | 5 | 0.957333 | 0.0355434 | 5 | 0.959842 | 0.00636164 | 5 | 0.958115 | 0.0225791 | 5 | 20.8007 | 19.3631 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 5 | 0.25 | 0.992655 | 0.00212557 | 5 | 0.943291 | 0.017237 | 5 | 0.993333 | 0 | 5 | 0.967601 | 0.00907983 | 5 | 0.982861 | 0.00375041 | 5 | 33.4296 | 10.7493 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 10 | 0.25 | 0.992214 | 0.00211603 | 5 | 0.93972 | 0.0171544 | 5 | 0.993333 | 0 | 5 | 0.965719 | 0.00903829 | 5 | 0.982084 | 0.00373351 | 5 | 35.6583 | 10.7011 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 30 | 0.25 | 0.992068 | 0.00041872 | 5 | 0.938297 | 0.00337145 | 5 | 0.993333 | 0 | 5 | 0.965029 | 0.00178238 | 5 | 0.981814 | 0.000737773 | 5 | 36.4012 | 2.11754 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 0 | 0.8 | 0.99596 | 0 | 5 | 0.976898 | 0 | 5 | 0.986667 | 0 | 5 | 0.981758 | 0 | 5 | 0.984697 | 0 | 5 | 13.0004 | 0 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 1 | 0.8 | 0.989864 | 0.00564526 | 5 | 0.988854 | 0.0153008 | 5 | 0.919333 | 0.0658871 | 5 | 0.951407 | 0.0284422 | 5 | 0.931588 | 0.0514318 | 5 | 6.31449 | 8.67136 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 5 | 0.8 | 0.99552 | 0.000796162 | 5 | 0.976196 | 0.00253299 | 5 | 0.983333 | 0.00942809 | 5 | 0.979724 | 0.00374119 | 5 | 0.98188 | 0.00713043 | 5 | 13.3719 | 1.55385 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 10 | 0.8 | 0.995667 | 0.000307257 | 5 | 0.976232 | 0.00340974 | 5 | 0.984667 | 0.0060553 | 5 | 0.980413 | 0.00147268 | 5 | 0.982957 | 0.00418039 | 5 | 13.3719 | 2.03446 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 30 | 0.8 | 0.995593 | 0 | 5 | 0.973684 | 0 | 5 | 0.986667 | 0 | 5 | 0.980132 | 0 | 5 | 0.984043 | 0 | 5 | 14.8576 | 0 | 5 |
### `results/film_personalization/full_run_v1/seed43/deployment_size.csv`

- Shape: `1 rows x 7 columns`
- Missing cells: `0`
- Columns: `base_tflite_bytes, film_tflite_bytes, tflite_overhead_bytes, user_affine_bytes, channels, base_tflite_max_abs_error, film_tflite_max_abs_error`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| base_tflite_bytes | 1 | 195164 | 195164 | 195164 |  | 195164 |
| film_tflite_bytes | 1 | 197940 | 197940 | 197940 |  | 197940 |
| tflite_overhead_bytes | 1 | 2776 | 2776 | 2776 |  | 2776 |
| user_affine_bytes | 1 | 512 | 512 | 512 |  | 512 |
| channels | 1 | 64 | 64 | 64 |  | 64 |
| base_tflite_max_abs_error | 1 | 5.96046e-08 | 5.96046e-08 | 5.96046e-08 |  | 5.96046e-08 |
| film_tflite_max_abs_error | 1 | 0 | 0 | 0 |  | 0 |

First 10 rows:

| base_tflite_bytes | film_tflite_bytes | tflite_overhead_bytes | user_affine_bytes | channels | base_tflite_max_abs_error | film_tflite_max_abs_error |
| --- | --- | --- | --- | --- | --- | --- |
| 195164 | 197940 | 2776 | 512 | 64 | 5.96046e-08 | 0 |
### `results/film_personalization/full_run_v1/seed43/experiment_config.csv`

- Shape: `23 rows x 2 columns`
- Missing cells: `0`
- Columns: `key, value`

Numeric statistics:

_No rows._

First 10 rows:

| key | value |
| --- | --- |
| seed | 43 |
| calibration_seeds | [17, 31, 47, 71, 101] |
| calibration_sizes | [0, 1, 5, 10, 30] |
| epochs | 30 |
| episodes_per_epoch | 30 |
| validation_episodes | 12 |
| patience | 5 |
| fine_tune_epochs | 3 |
| fine_tune_tuning_json | {"configurations_per_method": 12, "methods": {"full_model": {"configurations_tried": [{"epochs": 1, "learning_rate": 0.0001, "validation_mean_f2": 0.9847756870751981, "validation_mean_false_alarms_per_hour": 21.716965742251222, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.0001, "validation_mean_f2": 0.9846331270992463, "validation_mean_false_alarms_per_hour": 20.799347471451878, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.0001, "validation_mean_f2": 0.9839412525898819, "validation_mean_false_alarms_per_hour": 18.658238172920065, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.001, "validation_mean_f2": 0.9823417548925377, "validation_mean_false_alarms_per_hour": 18.658238172920065, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.001, "validation_mean_f2": 0.9720902816703267, "validation_mean_false_alarms_per_hour": 9.787928221859707, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.001, "validation_mean_f2": 0.9185609535499818, "validation_mean_false_alarms_per_hour": 3.976345840130506, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.01, "validation_mean_f2": 0.7772710519635823, "validation_mean_false_alarms_per_hour": 5.8115823817292, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.01, "validation_mean_f2": 0.17341037482104524, "validation_mean_false_alarms_per_hour": 0.9176182707993474, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.01, "validation_mean_f2": 0.09762485750087149, "validation_mean_false_alarms_per_hour": 0.0, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.1, "validation_mean_f2": 0.1989771060012393, "validation_mean_false_alarms_per_hour": 3.058727569331158, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.1, "validation_mean_f2": 0.008264462809917356, "validation_mean_false_alarms_per_hour": 0.0, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.1, "validation_mean_f2": 0.0, "validation_mean_false_alarms_per_hour": 0.0, "validation_subjects_with_falls": 5}], "selected_config": {"epochs": 1, "learning_rate": 0.0001}, "selection_metrics": {"validation_mean_f2": 0.9847756870751981, "validation_mean_false_alarms_per_hour": 21.716965742251222}}, "last_layer": {"configurations_tried": [{"epochs": 1, "learning_rate": 0.0001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.0001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.0001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.001, "validation_mean_f2": 0.984243790398388, "validation_mean_false_alarms_per_hour": 21.716965742251222, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.01, "validation_mean_f2": 0.9847754325282876, "validation_mean_false_alarms_per_hour": 21.716965742251222, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.01, "validation_mean_f2": 0.9839695594208253, "validation_mean_false_alarms_per_hour": 21.105220228384994, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.01, "validation_mean_f2": 0.9837951029808144, "validation_mean_false_alarms_per_hour": 17.740619902120716, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.1, "validation_mean_f2": 0.984620771391346, "validation_mean_false_alarms_per_hour": 19.575856443719413, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.1, "validation_mean_f2": 0.9825608192960826, "validation_mean_false_alarms_per_hour": 13.152528548123982, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.1, "validation_mean_f2": 0.9745709836127938, "validation_mean_false_alarms_per_hour": 10.399673735725939, "validation_subjects_with_falls": 5}], "selected_config": {"epochs": 1, "learning_rate": 0.01}, "selection_metrics": {"validation_mean_f2": 0.9847754325282876, "validation_mean_false_alarms_per_hour": 21.716965742251222}}}, "tie_breaking": ["higher validation mean F2", "lower validation mean false alarms per hour", "fewer epochs", "lower learning rate"], "validation_calibration_policy": "one ADL calibration window per validation subject", "validation_threshold": 0.25} |
| timevae_epochs | 0 |
### `results/film_personalization/full_run_v1/seed43/fine_tune_probability_shift_summary.csv`

- Shape: `8 rows x 8 columns`
- Missing cells: `0`
- Columns: `evaluation, method, label_group, windows, signed_mean_probability_shift, absolute_mean_probability_shift, threshold_crossings_vs_global, subject_seed_rows`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| windows | 8 | 6000 | 37231.2 | 35117.5 | 32097.6 | 72690 |
| signed_mean_probability_shift | 8 | -0.000789605 | -0.000166974 | -9.12413e-05 | 0.000267296 | 9.57309e-05 |
| absolute_mean_probability_shift | 8 | 5.29214e-05 | 0.00040309 | 0.000171539 | 0.000496877 | 0.00146631 |
| threshold_crossings_vs_global | 8 | 0 | 6.5 | 5.5 | 5.58058 | 15 |
| subject_seed_rows | 8 | 20 | 27.5 | 30 | 4.6291 | 30 |

First 10 rows:

| evaluation | method | label_group | windows | signed_mean_probability_shift | absolute_mean_probability_shift | threshold_crossings_vs_global | subject_seed_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | full_model | ADL | 61235 | -3.28201e-05 | 5.29214e-05 | 5 | 30 |
| real_heldout | full_model | Fall | 6000 | -0.000138663 | 0.000419319 | 0 | 20 |
| real_heldout | last_layer | ADL | 61235 | -6.62298e-05 | 6.62298e-05 | 5 | 30 |
| real_heldout | last_layer | Fall | 6000 | -0.000221726 | 0.000221726 | 0 | 20 |
| simulated_users | full_model | ADL | 72690 | -9.52247e-05 | 0.000121352 | 14 | 30 |
| simulated_users | full_model | Fall | 9000 | 9.57309e-05 | 0.00146631 | 15 | 30 |
| simulated_users | last_layer | ADL | 72690 | -8.72579e-05 | 8.72579e-05 | 7 | 30 |
| simulated_users | last_layer | Fall | 9000 | -0.000789605 | 0.000789605 | 6 | 30 |
### `results/film_personalization/full_run_v1/seed43/fine_tune_probability_shifts.csv`

- Shape: `220 rows x 10 columns`
- Missing cells: `0`
- Columns: `evaluation, profile, subject_id, seed, method, label_group, windows, signed_mean_probability_shift, absolute_mean_probability_shift, threshold_crossings_vs_global`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| seed | 220 | 17 | 53.4 | 47 | 29.878 | 101 |
| windows | 220 | 300 | 1353.86 | 1277.5 | 1014.7 | 2423 |
| signed_mean_probability_shift | 220 | -0.00363629 | -0.000165773 | -7.81771e-05 | 0.000737837 | 0.00446905 |
| absolute_mean_probability_shift | 220 | 7.1989e-06 | 0.000410596 | 0.000109823 | 0.00080376 | 0.00514542 |
| threshold_crossings_vs_global | 220 | 0 | 0.236364 | 0 | 0.485916 | 3 |

First 10 rows:

| evaluation | profile | subject_id | seed | method | label_group | windows | signed_mean_probability_shift | absolute_mean_probability_shift | threshold_crossings_vs_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | Real | SA02 | 17 | last_layer | ADL | 2423 | -7.77114e-05 | 7.77114e-05 | 1 |
| real_heldout | Real | SA02 | 17 | last_layer | Fall | 300 | -0.000274678 | 0.000274678 | 0 |
| real_heldout | Real | SA02 | 17 | full_model | ADL | 2423 | -7.35094e-05 | 0.000104677 | 1 |
| real_heldout | Real | SA02 | 17 | full_model | Fall | 300 | -0.000168689 | 0.000609399 | 0 |
| real_heldout | Real | SA02 | 31 | last_layer | ADL | 2423 | -6.83159e-05 | 6.83159e-05 | 1 |
| real_heldout | Real | SA02 | 31 | last_layer | Fall | 300 | -0.000255302 | 0.000255302 | 0 |
| real_heldout | Real | SA02 | 31 | full_model | ADL | 2423 | -7.25785e-05 | 9.38631e-05 | 1 |
| real_heldout | Real | SA02 | 31 | full_model | Fall | 300 | -0.000108464 | 0.000477679 | 0 |
| real_heldout | Real | SA02 | 47 | last_layer | ADL | 2423 | -8.04102e-05 | 8.04102e-05 | 1 |
| real_heldout | Real | SA02 | 47 | last_layer | Fall | 300 | -0.000291608 | 0.000291608 | 0 |
### `results/film_personalization/full_run_v1/seed43/meta_training_history.csv`

- Shape: `13 rows x 4 columns`
- Missing cells: `0`
- Columns: `epoch, train_loss, val_loss, epoch_seconds`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| epoch | 13 | 1 | 7 | 7 | 3.89444 | 13 |
| train_loss | 13 | 0.0106318 | 0.0822725 | 0.0933855 | 0.0462833 | 0.1544 |
| val_loss | 13 | 0.0212544 | 0.0985273 | 0.110536 | 0.0545254 | 0.177773 |
| epoch_seconds | 13 | 2.99985 | 3.45366 | 3.28003 | 0.398262 | 4.43395 |

First 10 rows:

| epoch | train_loss | val_loss | epoch_seconds |
| --- | --- | --- | --- |
| 1 | 0.1544 | 0.177773 | 3.2392 |
| 2 | 0.128448 | 0.107796 | 3.23781 |
| 3 | 0.0936739 | 0.0268531 | 3.11643 |
| 4 | 0.0933855 | 0.132786 | 2.99985 |
| 5 | 0.0106318 | 0.0547313 | 3.21394 |
| 6 | 0.123048 | 0.125403 | 3.82609 |
| 7 | 0.0245079 | 0.175387 | 3.78083 |
| 8 | 0.118569 | 0.0212544 | 3.59767 |
| 9 | 0.0401025 | 0.0644516 | 4.43395 |
| 10 | 0.0256922 | 0.140571 | 3.60404 |
### `results/film_personalization/full_run_v1/seed43/metrics_summary.csv`

- Shape: `245 rows x 17 columns`
- Missing cells: `0`
- Columns: `evaluation, profile, model, k, threshold, accuracy_mean, accuracy_std, precision_mean, precision_std, recall_mean, recall_std, f1_mean, f1_std, f2_mean, f2_std, false_alarms_per_hour_mean, false_alarms_per_hour_std`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 245 | 0 | 9.2 | 5 | 11.0025 | 30 |
| threshold | 245 | 0.25 | 0.335714 | 0.25 | 0.210386 | 0.85 |
| accuracy_mean | 245 | 0.973926 | 0.989402 | 0.989864 | 0.00585019 | 0.997552 |
| accuracy_std | 245 | 0 | 0.000930237 | 0.000479177 | 0.0012326 | 0.00793872 |
| precision_mean | 245 | 0.818781 | 0.95685 | 0.96129 | 0.035939 | 1 |
| precision_std | 245 | 0 | 0.00480778 | 0.0033802 | 0.00714836 | 0.0580056 |
| recall_mean | 245 | 0.766667 | 0.947599 | 0.993289 | 0.0631958 | 1 |
| recall_std | 245 | 0 | 0.00320003 | 0.00182574 | 0.00472828 | 0.0385861 |
| f1_mean | 245 | 0.86629 | 0.950032 | 0.952013 | 0.0277702 | 0.985507 |
| f1_std | 245 | 0 | 0.00329296 | 0.00211443 | 0.00387804 | 0.0288906 |
| f2_mean | 245 | 0.803634 | 0.948071 | 0.976436 | 0.0478697 | 0.992569 |
| f2_std | 245 | 0 | 0.00301529 | 0.00182465 | 0.00356499 | 0.0304638 |
| false_alarms_per_hour_mean | 245 | 0 | 24.5451 | 20.0578 | 23.1206 | 125.175 |
| false_alarms_per_hour_std | 245 | 0 | 3.96558 | 1.8572 | 6.52433 | 45.3857 |

First 10 rows:

| evaluation | profile | model | k | threshold | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | f2_mean | f2_std | false_alarms_per_hour_mean | false_alarms_per_hour_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | Real | FiLM (canonical threshold) | 0 | 0.25 | 0.995495 | 0.00344899 | 0.944417 | 0.0109756 | 0.9975 | 0.00148087 | 0.970206 | 0.00625869 | 0.986393 | 0.00323478 | 21.8531 | 16.6734 |
| real_heldout | Real | FiLM (canonical threshold) | 1 | 0.25 | 0.992119 | 0.00591147 | 0.905399 | 0.0229622 | 1 | 0 | 0.950206 | 0.0127 | 0.979428 | 0.005413 | 39.7674 | 29.9926 |
| real_heldout | Real | FiLM (canonical threshold) | 5 | 0.25 | 0.992435 | 0.00589215 | 0.907071 | 0.0207136 | 1 | 0 | 0.951153 | 0.0114275 | 0.979839 | 0.00486189 | 38.2584 | 29.7976 |
| real_heldout | Real | FiLM (canonical threshold) | 10 | 0.25 | 0.992655 | 0.00571376 | 0.909515 | 0.0201044 | 1 | 0 | 0.952503 | 0.0110503 | 0.980413 | 0.00469007 | 37.144 | 28.8954 |
| real_heldout | Real | FiLM (canonical threshold) | 30 | 0.25 | 0.993108 | 0.00517806 | 0.914419 | 0.0139322 | 1 | 0 | 0.955244 | 0.00763148 | 0.981589 | 0.00323142 | 34.8535 | 26.1863 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 0 | 0.85 | 0.997552 | 0.00194884 | 0.98173 | 0.00379335 | 0.985 | 0.00705036 | 0.983353 | 0.00470502 | 0.984338 | 0.00602426 | 6.80974 | 5.19223 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 1 | 0.85 | 0.996903 | 0.00244113 | 0.969066 | 0.00867321 | 0.9895 | 0.00694801 | 0.979145 | 0.00551013 | 0.985324 | 0.00574627 | 11.7623 | 9.42374 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 5 | 0.85 | 0.996866 | 0.00258734 | 0.966067 | 0.0120038 | 0.992333 | 0.00460104 | 0.978987 | 0.0068907 | 0.986942 | 0.00472275 | 13.0004 | 11.0144 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 10 | 0.85 | 0.997062 | 0.00236432 | 0.968995 | 0.00983115 | 0.991833 | 0.00523986 | 0.980252 | 0.00579835 | 0.987161 | 0.00474509 | 11.8242 | 9.72377 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 30 | 0.85 | 0.997111 | 0.00230603 | 0.96977 | 0.0093436 | 0.991667 | 0.0046516 | 0.980571 | 0.00546915 | 0.987192 | 0.00430651 | 11.5147 | 9.41196 |
### `results/film_personalization/full_run_v1/seed43/paired_wilcoxon.csv`

- Shape: `10 rows x 6 columns`
- Missing cells: `0`
- Columns: `evaluation, metric, comparison, n_subjects, wilcoxon_statistic, p_value`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| n_subjects | 10 | 4 | 4 | 4 | 0 | 4 |
| wilcoxon_statistic | 10 | 0 | 2.4 | 2 | 2.54733 | 5 |
| p_value | 10 | 0.125 | 0.55 | 0.5 | 0.449537 | 1 |

First 10 rows:

| evaluation | metric | comparison | n_subjects | wilcoxon_statistic | p_value |
| --- | --- | --- | --- | --- | --- |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (+2 falls) | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (ADL-only) | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Global | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (+2 falls) | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (ADL-only) | 4 | 0 | 0.125 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (+2 falls) | 4 | 4 | 0.875 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (ADL-only) | 4 | 5 | 1 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Global | 4 | 5 | 1 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (+2 falls) | 4 | 5 | 1 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (ADL-only) | 4 | 5 | 1 |
### `results/film_personalization/full_run_v1/seed43/per_user_metrics.csv`

- Shape: `2,000 rows x 16 columns`
- Missing cells: `1,000`
- Columns: `evaluation, profile, subject_id, model, k, seed, threshold, accuracy, precision, recall, f1, f2, false_alarms_per_hour, false_positive_windows, adl_windows, fall_windows`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 2000 | 0 | 9.2 | 5 | 10.9827 | 30 |
| seed | 2000 | 17 | 53.4 | 47 | 29.8175 | 101 |
| threshold | 2000 | 0.25 | 0.34 | 0.25 | 0.214296 | 0.85 |
| accuracy | 2000 | 0.965846 | 0.991896 | 0.994487 | 0.00600246 | 1 |
| precision | 1750 | 0.763359 | 0.956551 | 0.96129 | 0.0325653 | 1 |
| recall | 1750 | 0.766667 | 0.961798 | 0.993289 | 0.0572849 | 1 |
| f1 | 1750 | 0.865801 | 0.957483 | 0.97377 | 0.0266699 | 0.990033 |
| f2 | 1750 | 0.803634 | 0.959681 | 0.984646 | 0.0439475 | 0.995355 |
| false_alarms_per_hour | 2000 | 0 | 22.3889 | 20.4292 | 21.4973 | 172.72 |
| false_positive_windows | 2000 | 0 | 12.0525 | 11 | 11.5772 | 93 |
| adl_windows | 2000 | 1277 | 2279.81 | 2423 | 378.933 | 2423 |
| fall_windows | 2000 | 0 | 262 | 300 | 99.0551 | 300 |

First 10 rows:

| evaluation | profile | subject_id | model | k | seed | threshold | accuracy | precision | recall | f1 | f2 | false_alarms_per_hour | false_positive_windows | adl_windows | fall_windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | Real | SA02 | Global | 0 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Last-layer fine-tune (ADL-only) | 0 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Full fine-tune (ADL-only) | 0 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Last-layer fine-tune (+2 falls) | 0 | 17 | 0.25 | 0.994487 | 0.957929 | 0.993289 | 0.975288 | 0.986009 | 24.1436 | 13 | 2423 | 298 |
| real_heldout | Real | SA02 | Full fine-tune (+2 falls) | 0 | 17 | 0.25 | 0.994487 | 0.957929 | 0.993289 | 0.975288 | 0.986009 | 24.1436 | 13 | 2423 | 298 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 0 | 17 | 0.25 | 0.993757 | 0.949206 | 0.996667 | 0.972358 | 0.986799 | 29.7152 | 16 | 2423 | 300 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 0 | 17 | 0.85 | 0.995226 | 0.976744 | 0.98 | 0.978369 | 0.979347 | 13.0004 | 7 | 2423 | 300 |
| real_heldout | Real | SA02 | Global | 1 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Last-layer fine-tune (ADL-only) | 1 | 17 | 0.25 | 0.994859 | 0.96129 | 0.993333 | 0.977049 | 0.986755 | 22.2864 | 12 | 2423 | 300 |
| real_heldout | Real | SA02 | Full fine-tune (ADL-only) | 1 | 17 | 0.25 | 0.994859 | 0.96129 | 0.993333 | 0.977049 | 0.986755 | 22.2864 | 12 | 2423 | 300 |
### `results/film_personalization/full_run_v1/seed43/per_user_seed_summary.csv`

- Shape: `400 rows x 24 columns`
- Missing cells: `400`
- Columns: `evaluation, profile, subject_id, model, k, threshold, accuracy_mean, accuracy_std, accuracy_count, precision_mean, precision_std, precision_count, recall_mean, recall_std, recall_count, f1_mean, f1_std, f1_count, f2_mean, f2_std, f2_count, false_alarms_per_hour_mean, false_alarms_per_hour_std, false_alarms_per_hour_count`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 400 | 0 | 9.2 | 5 | 10.9937 | 30 |
| threshold | 400 | 0.25 | 0.34 | 0.25 | 0.214511 | 0.85 |
| accuracy_mean | 400 | 0.973926 | 0.991896 | 0.994487 | 0.00593984 | 1 |
| accuracy_std | 400 | 0 | 0.00048256 | 0.000164356 | 0.000889756 | 0.00793872 |
| accuracy_count | 400 | 5 | 5 | 5 | 0 | 5 |
| precision_mean | 350 | 0.818781 | 0.956551 | 0.96129 | 0.0318735 | 1 |
| precision_std | 350 | 0 | 0.00365526 | 0.00165585 | 0.00673493 | 0.0580056 |
| precision_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| recall_mean | 350 | 0.766667 | 0.961798 | 0.993289 | 0.0571998 | 1 |
| recall_std | 350 | 0 | 0.00197425 | 0 | 0.00420526 | 0.0385861 |
| recall_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| f1_mean | 350 | 0.86629 | 0.957483 | 0.97409 | 0.0264026 | 0.990033 |
| f1_std | 350 | 0 | 0.00239203 | 0.000881497 | 0.00374543 | 0.0288906 |
| f1_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| f2_mean | 350 | 0.803634 | 0.959681 | 0.983802 | 0.0438635 | 0.994555 |
| f2_std | 350 | 0 | 0.00201783 | 0.000464356 | 0.00326709 | 0.0304638 |
| f2_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| false_alarms_per_hour_mean | 400 | 0 | 22.3889 | 20.4292 | 21.0158 | 125.175 |
| false_alarms_per_hour_std | 400 | 0 | 2.10958 | 0.830566 | 4.72034 | 45.3857 |
| false_alarms_per_hour_count | 400 | 5 | 5 | 5 | 0 | 5 |

First 10 rows:

| evaluation | profile | subject_id | model | k | threshold | accuracy_mean | accuracy_std | accuracy_count | precision_mean | precision_std | precision_count | recall_mean | recall_std | recall_count | f1_mean | f1_std | f1_count | f2_mean | f2_std | f2_count | false_alarms_per_hour_mean | false_alarms_per_hour_std | false_alarms_per_hour_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 0 | 0.25 | 0.993757 | 0 | 5 | 0.949206 | 0 | 5 | 0.996667 | 0 | 5 | 0.972358 | 0 | 5 | 0.986799 | 0 | 5 | 29.7152 | 0 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 1 | 0.25 | 0.990011 | 0.00248262 | 5 | 0.917179 | 0.0186813 | 5 | 1 | 0 | 5 | 0.956721 | 0.0102342 | 5 | 0.982205 | 0.00433414 | 5 | 50.5159 | 12.555 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 5 | 0.25 | 0.989497 | 0.00255227 | 5 | 0.913288 | 0.01914 | 5 | 1 | 0 | 5 | 0.954595 | 0.0105033 | 5 | 0.981306 | 0.00445266 | 5 | 53.116 | 12.9072 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 10 | 0.25 | 0.989423 | 0.00234288 | 5 | 0.912685 | 0.0177895 | 5 | 1 | 0 | 5 | 0.954277 | 0.00970475 | 5 | 0.981175 | 0.00409838 | 5 | 53.4874 | 11.8483 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 30 | 0.25 | 0.990525 | 0.000706404 | 5 | 0.920836 | 0.00540773 | 5 | 1 | 0 | 5 | 0.95878 | 0.00293882 | 5 | 0.983092 | 0.00123794 | 5 | 47.9158 | 3.5724 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 0 | 0.85 | 0.995226 | 0 | 5 | 0.976744 | 0 | 5 | 0.98 | 0 | 5 | 0.978369 | 0 | 5 | 0.979347 | 0 | 5 | 13.0004 | 0 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 1 | 0.85 | 0.994418 | 0.000706404 | 5 | 0.967855 | 0.00636637 | 5 | 0.982 | 0.0060553 | 5 | 0.974852 | 0.00316852 | 5 | 0.979123 | 0.00432798 | 5 | 18.2006 | 3.80613 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 5 | 0.85 | 0.994638 | 0.000492707 | 5 | 0.964246 | 0.00534184 | 5 | 0.988 | 0.00182574 | 5 | 0.975969 | 0.00212002 | 5 | 0.98315 | 0.000893877 | 5 | 20.4292 | 3.21677 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 10 | 0.85 | 0.994785 | 0.000307257 | 5 | 0.966122 | 0.00562699 | 5 | 0.987333 | 0.00365148 | 5 | 0.976595 | 0.00128517 | 5 | 0.983006 | 0.00182465 | 5 | 19.3149 | 3.37377 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 30 | 0.85 | 0.995006 | 0.000201147 | 5 | 0.967363 | 5.83043e-05 | 5 | 0.988 | 0.00182574 | 5 | 0.977572 | 0.00092319 | 5 | 0.983802 | 0.00146008 | 5 | 18.572 | 0 | 5 |
### `results/film_personalization/full_run_v1/seed44/deployment_size.csv`

- Shape: `1 rows x 7 columns`
- Missing cells: `0`
- Columns: `base_tflite_bytes, film_tflite_bytes, tflite_overhead_bytes, user_affine_bytes, channels, base_tflite_max_abs_error, film_tflite_max_abs_error`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| base_tflite_bytes | 1 | 195164 | 195164 | 195164 |  | 195164 |
| film_tflite_bytes | 1 | 197940 | 197940 | 197940 |  | 197940 |
| tflite_overhead_bytes | 1 | 2776 | 2776 | 2776 |  | 2776 |
| user_affine_bytes | 1 | 512 | 512 | 512 |  | 512 |
| channels | 1 | 64 | 64 | 64 |  | 64 |
| base_tflite_max_abs_error | 1 | 5.96046e-08 | 5.96046e-08 | 5.96046e-08 |  | 5.96046e-08 |
| film_tflite_max_abs_error | 1 | 0 | 0 | 0 |  | 0 |

First 10 rows:

| base_tflite_bytes | film_tflite_bytes | tflite_overhead_bytes | user_affine_bytes | channels | base_tflite_max_abs_error | film_tflite_max_abs_error |
| --- | --- | --- | --- | --- | --- | --- |
| 195164 | 197940 | 2776 | 512 | 64 | 5.96046e-08 | 0 |
### `results/film_personalization/full_run_v1/seed44/experiment_config.csv`

- Shape: `23 rows x 2 columns`
- Missing cells: `0`
- Columns: `key, value`

Numeric statistics:

_No rows._

First 10 rows:

| key | value |
| --- | --- |
| seed | 44 |
| calibration_seeds | [17, 31, 47, 71, 101] |
| calibration_sizes | [0, 1, 5, 10, 30] |
| epochs | 30 |
| episodes_per_epoch | 30 |
| validation_episodes | 12 |
| patience | 5 |
| fine_tune_epochs | 3 |
| fine_tune_tuning_json | {"configurations_per_method": 12, "methods": {"full_model": {"configurations_tried": [{"epochs": 1, "learning_rate": 0.0001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.0001, "validation_mean_f2": 0.9850429612665825, "validation_mean_false_alarms_per_hour": 22.328711256117458, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.0001, "validation_mean_f2": 0.98426594344652, "validation_mean_false_alarms_per_hour": 22.940456769983687, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.001, "validation_mean_f2": 0.9837951099423824, "validation_mean_false_alarms_per_hour": 17.740619902120716, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.001, "validation_mean_f2": 0.9722041704480923, "validation_mean_false_alarms_per_hour": 11.929037520391518, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.001, "validation_mean_f2": 0.924035919339332, "validation_mean_false_alarms_per_hour": 5.8115823817292, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.01, "validation_mean_f2": 0.8022519133495187, "validation_mean_false_alarms_per_hour": 5.811582381729202, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.01, "validation_mean_f2": 0.06140703150669928, "validation_mean_false_alarms_per_hour": 0.0, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.01, "validation_mean_f2": 0.000832639467110741, "validation_mean_false_alarms_per_hour": 0.0, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.1, "validation_mean_f2": 0.10112359550561797, "validation_mean_false_alarms_per_hour": 0.0, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.1, "validation_mean_f2": 0.0, "validation_mean_false_alarms_per_hour": 0.0, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.1, "validation_mean_f2": 0.0, "validation_mean_false_alarms_per_hour": 0.0, "validation_subjects_with_falls": 5}], "selected_config": {"epochs": 5, "learning_rate": 0.0001}, "selection_metrics": {"validation_mean_f2": 0.9850429612665825, "validation_mean_false_alarms_per_hour": 22.328711256117458}}, "last_layer": {"configurations_tried": [{"epochs": 1, "learning_rate": 0.0001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.0001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.0001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.001, "validation_mean_f2": 0.984243790398388, "validation_mean_false_alarms_per_hour": 21.716965742251222, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.01, "validation_mean_f2": 0.9847754325282876, "validation_mean_false_alarms_per_hour": 21.716965742251222, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.01, "validation_mean_f2": 0.9839695594208253, "validation_mean_false_alarms_per_hour": 21.105220228384994, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.01, "validation_mean_f2": 0.9830010207266795, "validation_mean_false_alarms_per_hour": 18.35236541598695, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.1, "validation_mean_f2": 0.9834163987871101, "validation_mean_false_alarms_per_hour": 19.88172920065253, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.1, "validation_mean_f2": 0.9829583652533118, "validation_mean_false_alarms_per_hour": 13.458401305057096, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.1, "validation_mean_f2": 0.9743035353355884, "validation_mean_false_alarms_per_hour": 9.787928221859707, "validation_subjects_with_falls": 5}], "selected_config": {"epochs": 1, "learning_rate": 0.01}, "selection_metrics": {"validation_mean_f2": 0.9847754325282876, "validation_mean_false_alarms_per_hour": 21.716965742251222}}}, "tie_breaking": ["higher validation mean F2", "lower validation mean false alarms per hour", "fewer epochs", "lower learning rate"], "validation_calibration_policy": "one ADL calibration window per validation subject", "validation_threshold": 0.25} |
| timevae_epochs | 0 |
### `results/film_personalization/full_run_v1/seed44/fine_tune_probability_shift_summary.csv`

- Shape: `8 rows x 8 columns`
- Missing cells: `0`
- Columns: `evaluation, method, label_group, windows, signed_mean_probability_shift, absolute_mean_probability_shift, threshold_crossings_vs_global, subject_seed_rows`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| windows | 8 | 6000 | 37231.2 | 35117.5 | 32097.6 | 72690 |
| signed_mean_probability_shift | 8 | -0.000849036 | -0.000159087 | -0.000185979 | 0.000663678 | 0.00128761 |
| absolute_mean_probability_shift | 8 | 6.74122e-05 | 0.00143456 | 0.000360766 | 0.00247054 | 0.00728771 |
| threshold_crossings_vs_global | 8 | 0 | 21.375 | 8.5 | 27.9895 | 71 |
| subject_seed_rows | 8 | 20 | 27.5 | 30 | 4.6291 | 30 |

First 10 rows:

| evaluation | method | label_group | windows | signed_mean_probability_shift | absolute_mean_probability_shift | threshold_crossings_vs_global | subject_seed_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | full_model | ADL | 61235 | -0.000131333 | 0.00023024 | 18 | 30 |
| real_heldout | full_model | Fall | 6000 | -0.000833254 | 0.00223677 | 0 | 20 |
| real_heldout | last_layer | ADL | 61235 | -6.74122e-05 | 6.74122e-05 | 5 | 30 |
| real_heldout | last_layer | Fall | 6000 | -0.000240625 | 0.000240625 | 0 | 20 |
| simulated_users | full_model | ADL | 72690 | -0.000354876 | 0.000480907 | 60 | 30 |
| simulated_users | full_model | Fall | 9000 | 0.00128761 | 0.00728771 | 71 | 30 |
| simulated_users | last_layer | ADL | 72690 | -8.37687e-05 | 8.37687e-05 | 7 | 30 |
| simulated_users | last_layer | Fall | 9000 | -0.000849036 | 0.000849036 | 10 | 30 |
### `results/film_personalization/full_run_v1/seed44/fine_tune_probability_shifts.csv`

- Shape: `220 rows x 10 columns`
- Missing cells: `0`
- Columns: `evaluation, profile, subject_id, seed, method, label_group, windows, signed_mean_probability_shift, absolute_mean_probability_shift, threshold_crossings_vs_global`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| seed | 220 | 17 | 53.4 | 47 | 29.878 | 101 |
| windows | 220 | 300 | 1353.86 | 1277.5 | 1014.7 | 2423 |
| signed_mean_probability_shift | 220 | -0.015772 | -0.000124737 | -0.000115146 | 0.00311334 | 0.022215 |
| absolute_mean_probability_shift | 220 | 1.32243e-06 | 0.00145236 | 0.000246961 | 0.00367468 | 0.0246218 |
| threshold_crossings_vs_global | 220 | 0 | 0.777273 | 0 | 1.8032 | 10 |

First 10 rows:

| evaluation | profile | subject_id | seed | method | label_group | windows | signed_mean_probability_shift | absolute_mean_probability_shift | threshold_crossings_vs_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | Real | SA02 | 17 | last_layer | ADL | 2423 | -8.58088e-05 | 8.58088e-05 | 1 |
| real_heldout | Real | SA02 | 17 | last_layer | Fall | 300 | -0.000285238 | 0.000285238 | 0 |
| real_heldout | Real | SA02 | 17 | full_model | ADL | 2423 | -0.000200682 | 0.000443297 | 2 |
| real_heldout | Real | SA02 | 17 | full_model | Fall | 300 | -0.000494696 | 0.00362392 | 0 |
| real_heldout | Real | SA02 | 31 | last_layer | ADL | 2423 | -6.78806e-05 | 6.78806e-05 | 1 |
| real_heldout | Real | SA02 | 31 | last_layer | Fall | 300 | -0.000254623 | 0.000254623 | 0 |
| real_heldout | Real | SA02 | 31 | full_model | ADL | 2423 | -0.000240635 | 0.000385671 | 2 |
| real_heldout | Real | SA02 | 31 | full_model | Fall | 300 | -0.000308765 | 0.00254825 | 0 |
| real_heldout | Real | SA02 | 47 | last_layer | ADL | 2423 | -7.76499e-05 | 7.76499e-05 | 1 |
| real_heldout | Real | SA02 | 47 | last_layer | Fall | 300 | -0.000261443 | 0.000261443 | 0 |
### `results/film_personalization/full_run_v1/seed44/meta_training_history.csv`

- Shape: `20 rows x 4 columns`
- Missing cells: `0`
- Columns: `epoch, train_loss, val_loss, epoch_seconds`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| epoch | 20 | 1 | 10.5 | 10.5 | 5.91608 | 20 |
| train_loss | 20 | 0.0152589 | 0.0659829 | 0.0635882 | 0.03895 | 0.13724 |
| val_loss | 20 | 0.0193105 | 0.086459 | 0.0781107 | 0.0480538 | 0.186525 |
| epoch_seconds | 20 | 3.02872 | 4.2036 | 3.80332 | 1.17939 | 6.7111 |

First 10 rows:

| epoch | train_loss | val_loss | epoch_seconds |
| --- | --- | --- | --- |
| 1 | 0.0265498 | 0.079675 | 3.74358 |
| 2 | 0.13724 | 0.134464 | 3.55949 |
| 3 | 0.0772299 | 0.075194 | 6.47188 |
| 4 | 0.115549 | 0.0765465 | 4.11177 |
| 5 | 0.0801089 | 0.148152 | 4.09591 |
| 6 | 0.0834204 | 0.0606548 | 6.62039 |
| 7 | 0.0152589 | 0.0907196 | 5.10771 |
| 8 | 0.0188571 | 0.055263 | 6.7111 |
| 9 | 0.0641012 | 0.161841 | 4.8352 |
| 10 | 0.0489431 | 0.186525 | 4.09403 |
### `results/film_personalization/full_run_v1/seed44/metrics_summary.csv`

- Shape: `245 rows x 17 columns`
- Missing cells: `0`
- Columns: `evaluation, profile, model, k, threshold, accuracy_mean, accuracy_std, precision_mean, precision_std, recall_mean, recall_std, f1_mean, f1_std, f2_mean, f2_std, false_alarms_per_hour_mean, false_alarms_per_hour_std`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 245 | 0 | 9.2 | 5 | 11.0025 | 30 |
| threshold | 245 | 0.25 | 0.335714 | 0.25 | 0.210386 | 0.85 |
| accuracy_mean | 245 | 0.943004 | 0.988357 | 0.989791 | 0.00779698 | 0.997184 |
| accuracy_std | 245 | 0 | 0.00192414 | 0.000658506 | 0.00561571 | 0.0697689 |
| precision_mean | 245 | 0.732609 | 0.949182 | 0.961686 | 0.0496212 | 0.997783 |
| precision_std | 245 | 0 | 0.0100088 | 0.00348367 | 0.023898 | 0.227917 |
| recall_mean | 245 | 0.797333 | 0.949369 | 0.993289 | 0.0635088 | 1 |
| recall_std | 245 | 0 | 0.0035318 | 0.00150072 | 0.00541486 | 0.0494076 |
| f1_mean | 245 | 0.826111 | 0.946429 | 0.951976 | 0.0324254 | 0.980985 |
| f1_std | 245 | 0 | 0.00643187 | 0.00291432 | 0.0153109 | 0.163694 |
| f2_mean | 245 | 0.830308 | 0.947464 | 0.9742 | 0.0479111 | 0.989301 |
| f2_std | 245 | 0 | 0.00439618 | 0.00233745 | 0.00789925 | 0.0847299 |
| false_alarms_per_hour_mean | 245 | 1.11432 | 30.7935 | 20.8007 | 37.9733 | 277.095 |
| false_alarms_per_hour_std | 245 | 0 | 9.14034 | 2.11754 | 29.1418 | 361.161 |

First 10 rows:

| evaluation | profile | model | k | threshold | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | f2_mean | f2_std | false_alarms_per_hour_mean | false_alarms_per_hour_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | Real | FiLM (canonical threshold) | 0 | 0.25 | 0.993794 | 0.00458315 | 0.922805 | 0.00889842 | 0.999167 | 0.00148087 | 0.959453 | 0.00541258 | 0.982889 | 0.0030644 | 31.0772 | 22.8344 |
| real_heldout | Real | FiLM (canonical threshold) | 1 | 0.25 | 0.984175 | 0.0175338 | 0.880532 | 0.0928481 | 0.998667 | 0.00251312 | 0.933172 | 0.056492 | 0.970494 | 0.0256421 | 76.9175 | 85.0871 |
| real_heldout | Real | FiLM (canonical threshold) | 5 | 0.25 | 0.989265 | 0.00827381 | 0.880362 | 0.0389945 | 0.999833 | 0.000745356 | 0.935849 | 0.0227281 | 0.97308 | 0.00999581 | 53.9044 | 42.0241 |
| real_heldout | Real | FiLM (canonical threshold) | 10 | 0.25 | 0.989438 | 0.00834665 | 0.877234 | 0.0345302 | 1 | 0 | 0.934257 | 0.0198109 | 0.972521 | 0.0086559 | 53.3249 | 42.3109 |
| real_heldout | Real | FiLM (canonical threshold) | 30 | 0.25 | 0.989557 | 0.00726305 | 0.880624 | 0.0261891 | 1 | 0 | 0.936329 | 0.0147151 | 0.973465 | 0.00633699 | 52.5802 | 37.0194 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 0 | 0.85 | 0.997184 | 0.00218508 | 0.973767 | 0.00699223 | 0.988333 | 0.0038236 | 0.980985 | 0.00453562 | 0.985378 | 0.00376633 | 9.90508 | 7.83908 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 1 | 0.85 | 0.996199 | 0.00364348 | 0.960786 | 0.0289156 | 0.991167 | 0.00669075 | 0.975458 | 0.0135431 | 0.98475 | 0.00526232 | 15.8846 | 18.1641 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 5 | 0.85 | 0.996193 | 0.00346856 | 0.956151 | 0.0218623 | 0.994333 | 0.00391429 | 0.974728 | 0.0111607 | 0.986361 | 0.00483312 | 17.1482 | 16.6662 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 10 | 0.85 | 0.996132 | 0.00320841 | 0.955164 | 0.0166603 | 0.994333 | 0.00376192 | 0.974273 | 0.0084567 | 0.986189 | 0.00399434 | 17.4577 | 15.1085 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 30 | 0.85 | 0.996254 | 0.00299345 | 0.956295 | 0.0134473 | 0.994667 | 0.00380827 | 0.975049 | 0.00695172 | 0.986712 | 0.00377965 | 16.9624 | 13.9214 |
### `results/film_personalization/full_run_v1/seed44/paired_wilcoxon.csv`

- Shape: `10 rows x 6 columns`
- Missing cells: `0`
- Columns: `evaluation, metric, comparison, n_subjects, wilcoxon_statistic, p_value`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| n_subjects | 10 | 4 | 4 | 4 | 0 | 4 |
| wilcoxon_statistic | 10 | 0 | 0.7 | 0.5 | 0.823273 | 2 |
| p_value | 10 | 0.125 | 0.2125 | 0.1875 | 0.102909 | 0.375 |

First 10 rows:

| evaluation | metric | comparison | n_subjects | wilcoxon_statistic | p_value |
| --- | --- | --- | --- | --- | --- |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (+2 falls) | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (ADL-only) | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Global | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (+2 falls) | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (ADL-only) | 4 | 0 | 0.125 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (+2 falls) | 4 | 1 | 0.25 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (ADL-only) | 4 | 2 | 0.375 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Global | 4 | 1 | 0.25 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (+2 falls) | 4 | 1 | 0.25 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (ADL-only) | 4 | 2 | 0.375 |
### `results/film_personalization/full_run_v1/seed44/per_user_metrics.csv`

- Shape: `2,000 rows x 16 columns`
- Missing cells: `1,000`
- Columns: `evaluation, profile, subject_id, model, k, seed, threshold, accuracy, precision, recall, f1, f2, false_alarms_per_hour, false_positive_windows, adl_windows, fall_windows`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 2000 | 0 | 9.2 | 5 | 10.9827 | 30 |
| seed | 2000 | 17 | 53.4 | 47 | 29.8175 | 101 |
| threshold | 2000 | 0.25 | 0.34 | 0.25 | 0.214296 | 0.85 |
| accuracy | 2000 | 0.822622 | 0.990971 | 0.994487 | 0.00875965 | 1 |
| precision | 1750 | 0.383142 | 0.949566 | 0.96129 | 0.0492676 | 1 |
| recall | 1750 | 0.78 | 0.963143 | 0.993333 | 0.0573169 | 1 |
| f1 | 1750 | 0.554017 | 0.954102 | 0.97377 | 0.0335417 | 0.990066 |
| f2 | 1750 | 0.75643 | 0.958946 | 0.983607 | 0.0444143 | 0.995355 |
| false_alarms_per_hour | 2000 | 0 | 27.6756 | 20.4292 | 40.4365 | 897.028 |
| false_positive_windows | 2000 | 0 | 14.7995 | 11 | 21.5436 | 483 |
| adl_windows | 2000 | 1277 | 2279.81 | 2423 | 378.933 | 2423 |
| fall_windows | 2000 | 0 | 262 | 300 | 99.0551 | 300 |

First 10 rows:

| evaluation | profile | subject_id | model | k | seed | threshold | accuracy | precision | recall | f1 | f2 | false_alarms_per_hour | false_positive_windows | adl_windows | fall_windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | Real | SA02 | Global | 0 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Last-layer fine-tune (ADL-only) | 0 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Full fine-tune (ADL-only) | 0 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Last-layer fine-tune (+2 falls) | 0 | 17 | 0.25 | 0.994487 | 0.957929 | 0.993289 | 0.975288 | 0.986009 | 24.1436 | 13 | 2423 | 298 |
| real_heldout | Real | SA02 | Full fine-tune (+2 falls) | 0 | 17 | 0.25 | 0.994855 | 0.961039 | 0.993289 | 0.976898 | 0.986667 | 22.2864 | 12 | 2423 | 298 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 0 | 17 | 0.25 | 0.990452 | 0.920245 | 1 | 0.958466 | 0.982962 | 48.2872 | 26 | 2423 | 300 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 0 | 17 | 0.85 | 0.994859 | 0.96732 | 0.986667 | 0.976898 | 0.982736 | 18.572 | 10 | 2423 | 300 |
| real_heldout | Real | SA02 | Global | 1 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Last-layer fine-tune (ADL-only) | 1 | 17 | 0.25 | 0.994859 | 0.96129 | 0.993333 | 0.977049 | 0.986755 | 22.2864 | 12 | 2423 | 300 |
| real_heldout | Real | SA02 | Full fine-tune (ADL-only) | 1 | 17 | 0.25 | 0.995226 | 0.964401 | 0.993333 | 0.978654 | 0.987409 | 20.4292 | 11 | 2423 | 300 |
### `results/film_personalization/full_run_v1/seed44/per_user_seed_summary.csv`

- Shape: `400 rows x 24 columns`
- Missing cells: `400`
- Columns: `evaluation, profile, subject_id, model, k, threshold, accuracy_mean, accuracy_std, accuracy_count, precision_mean, precision_std, precision_count, recall_mean, recall_std, recall_count, f1_mean, f1_std, f1_count, f2_mean, f2_std, f2_count, false_alarms_per_hour_mean, false_alarms_per_hour_std, false_alarms_per_hour_count`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 400 | 0 | 9.2 | 5 | 10.9937 | 30 |
| threshold | 400 | 0.25 | 0.34 | 0.25 | 0.214511 | 0.85 |
| accuracy_mean | 400 | 0.943004 | 0.990971 | 0.994418 | 0.00750115 | 1 |
| accuracy_std | 400 | 0 | 0.00137978 | 0.000201221 | 0.00488509 | 0.0697689 |
| accuracy_count | 400 | 5 | 5 | 5 | 0 | 5 |
| precision_mean | 350 | 0.732609 | 0.949566 | 0.961913 | 0.0443147 | 0.997783 |
| precision_std | 350 | 0 | 0.00860378 | 0.00170087 | 0.0226296 | 0.227917 |
| precision_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| recall_mean | 350 | 0.797333 | 0.963143 | 0.993289 | 0.057185 | 1 |
| recall_std | 350 | 0 | 0.00222436 | 0 | 0.00482906 | 0.0494076 |
| recall_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| f1_mean | 350 | 0.826111 | 0.954102 | 0.974064 | 0.0306697 | 0.988391 |
| f1_std | 350 | 0 | 0.005362 | 0.00113434 | 0.0143147 | 0.163694 |
| f1_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| f2_mean | 350 | 0.830308 | 0.958946 | 0.98371 | 0.0438807 | 0.993772 |
| f2_std | 350 | 0 | 0.00329076 | 0.000585678 | 0.00732695 | 0.0847299 |
| f2_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| false_alarms_per_hour_mean | 400 | 0 | 27.6756 | 20.8007 | 33.1796 | 277.095 |
| false_alarms_per_hour_std | 400 | 0 | 6.67724 | 0.830566 | 25.0435 | 361.161 |
| false_alarms_per_hour_count | 400 | 5 | 5 | 5 | 0 | 5 |

First 10 rows:

| evaluation | profile | subject_id | model | k | threshold | accuracy_mean | accuracy_std | accuracy_count | precision_mean | precision_std | precision_count | recall_mean | recall_std | recall_count | f1_mean | f1_std | f1_count | f2_mean | f2_std | f2_count | false_alarms_per_hour_mean | false_alarms_per_hour_std | false_alarms_per_hour_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 0 | 0.25 | 0.990452 | 0 | 5 | 0.920245 | 0 | 5 | 1 | 0 | 5 | 0.958466 | 0 | 5 | 0.982962 | 0 | 5 | 48.2872 | 0 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 1 | 0.25 | 0.982446 | 0.0171298 | 5 | 0.877319 | 0.108862 | 5 | 0.996 | 0.00365148 | 5 | 0.929742 | 0.0631448 | 5 | 0.967408 | 0.0265525 | 5 | 86.5456 | 88.4523 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 5 | 0.25 | 0.985163 | 0.00430316 | 5 | 0.882669 | 0.0311133 | 5 | 0.999333 | 0.00149071 | 5 | 0.937138 | 0.0170776 | 5 | 0.97342 | 0.0068176 | 5 | 74.6595 | 22.3405 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 10 | 0.25 | 0.983474 | 0.00496117 | 5 | 0.87062 | 0.033696 | 5 | 1 | 0 | 5 | 0.930557 | 0.0193598 | 5 | 0.970933 | 0.00846481 | 5 | 83.5741 | 25.0894 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 30 | 0.25 | 0.983107 | 0.00176123 | 5 | 0.867185 | 0.0119634 | 5 | 1 | 0 | 5 | 0.928833 | 0.00687786 | 5 | 0.970253 | 0.00300654 | 5 | 85.4313 | 8.90683 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 0 | 0.85 | 0.994859 | 0 | 5 | 0.96732 | 0 | 5 | 0.986667 | 0 | 5 | 0.976898 | 0 | 5 | 0.982736 | 0 | 5 | 18.572 | 0 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 1 | 0.85 | 0.994198 | 0.0019674 | 5 | 0.963882 | 0.0200692 | 5 | 0.984667 | 0.00505525 | 5 | 0.974038 | 0.00839698 | 5 | 0.980356 | 0.00233745 | 5 | 20.8007 | 12.2068 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 5 | 0.85 | 0.994932 | 0.000837441 | 5 | 0.964955 | 0.00715632 | 5 | 0.99 | 0 | 5 | 0.977306 | 0.00366857 | 5 | 0.98488 | 0.00148981 | 5 | 20.0578 | 4.23507 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 10 | 0.85 | 0.994565 | 0.000796162 | 5 | 0.961826 | 0.00676861 | 5 | 0.99 | 0 | 5 | 0.9757 | 0.00347866 | 5 | 0.984228 | 0.00141489 | 5 | 21.915 | 4.02632 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 30 | 0.85 | 0.994271 | 0.000201147 | 5 | 0.959305 | 0.00169823 | 5 | 0.99 | 0 | 5 | 0.97441 | 0.00087579 | 5 | 0.983704 | 0.000356961 | 5 | 23.4007 | 1.01723 | 5 |
### `results/film_personalization/full_run_v1/timing_pilot_1x30/deployment_size.csv`

- Shape: `1 rows x 7 columns`
- Missing cells: `0`
- Columns: `base_tflite_bytes, film_tflite_bytes, tflite_overhead_bytes, user_affine_bytes, channels, base_tflite_max_abs_error, film_tflite_max_abs_error`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| base_tflite_bytes | 1 | 195164 | 195164 | 195164 |  | 195164 |
| film_tflite_bytes | 1 | 197940 | 197940 | 197940 |  | 197940 |
| tflite_overhead_bytes | 1 | 2776 | 2776 | 2776 |  | 2776 |
| user_affine_bytes | 1 | 512 | 512 | 512 |  | 512 |
| channels | 1 | 64 | 64 | 64 |  | 64 |
| base_tflite_max_abs_error | 1 | 5.96046e-08 | 5.96046e-08 | 5.96046e-08 |  | 5.96046e-08 |
| film_tflite_max_abs_error | 1 | 0 | 0 | 0 |  | 0 |

First 10 rows:

| base_tflite_bytes | film_tflite_bytes | tflite_overhead_bytes | user_affine_bytes | channels | base_tflite_max_abs_error | film_tflite_max_abs_error |
| --- | --- | --- | --- | --- | --- | --- |
| 195164 | 197940 | 2776 | 512 | 64 | 5.96046e-08 | 0 |
### `results/film_personalization/full_run_v1/timing_pilot_1x30/experiment_config.csv`

- Shape: `23 rows x 2 columns`
- Missing cells: `0`
- Columns: `key, value`

Numeric statistics:

_No rows._

First 10 rows:

| key | value |
| --- | --- |
| seed | 42 |
| calibration_seeds | [17, 31, 47, 71, 101] |
| calibration_sizes | [0, 1, 5, 10, 30] |
| epochs | 1 |
| episodes_per_epoch | 30 |
| validation_episodes | 12 |
| patience | 5 |
| fine_tune_epochs | 3 |
| fine_tune_tuning_json | {"configurations_per_method": 9, "methods": {"full_model": {"configurations_tried": [{"epochs": 1, "learning_rate": 0.0001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.0001, "validation_mean_f2": 0.9845051056462616, "validation_mean_false_alarms_per_hour": 21.10522022838499, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.0001, "validation_mean_f2": 0.9842341776428419, "validation_mean_false_alarms_per_hour": 20.49347471451876, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.001, "validation_mean_f2": 0.9852882509284967, "validation_mean_false_alarms_per_hour": 19.269983686786297, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.001, "validation_mean_f2": 0.9780012921834705, "validation_mean_false_alarms_per_hour": 12.54078303425775, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.001, "validation_mean_f2": 0.9432932753694887, "validation_mean_false_alarms_per_hour": 8.564437194127244, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.01, "validation_mean_f2": 0.7582201604606291, "validation_mean_false_alarms_per_hour": 3.364600326264274, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.01, "validation_mean_f2": 0.006639017888924762, "validation_mean_false_alarms_per_hour": 0.0, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.01, "validation_mean_f2": 0.09358490566037736, "validation_mean_false_alarms_per_hour": 0.30587275693311583, "validation_subjects_with_falls": 5}], "selected_config": {"epochs": 1, "learning_rate": 0.001}, "selection_metrics": {"validation_mean_f2": 0.9852882509284967, "validation_mean_false_alarms_per_hour": 19.269983686786297}}, "last_layer": {"configurations_tried": [{"epochs": 1, "learning_rate": 0.0001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.0001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.0001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.001, "validation_mean_f2": 0.984644812606137, "validation_mean_false_alarms_per_hour": 22.022838499184342, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.001, "validation_mean_f2": 0.9847754325282876, "validation_mean_false_alarms_per_hour": 21.716965742251222, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.001, "validation_mean_f2": 0.984243790398388, "validation_mean_false_alarms_per_hour": 21.716965742251222, "validation_subjects_with_falls": 5}, {"epochs": 1, "learning_rate": 0.01, "validation_mean_f2": 0.9847754325282876, "validation_mean_false_alarms_per_hour": 21.716965742251222, "validation_subjects_with_falls": 5}, {"epochs": 5, "learning_rate": 0.01, "validation_mean_f2": 0.9841006072335541, "validation_mean_false_alarms_per_hour": 20.799347471451878, "validation_subjects_with_falls": 5}, {"epochs": 20, "learning_rate": 0.01, "validation_mean_f2": 0.9835332603844064, "validation_mean_false_alarms_per_hour": 18.35236541598695, "validation_subjects_with_falls": 5}], "selected_config": {"epochs": 1, "learning_rate": 0.01}, "selection_metrics": {"validation_mean_f2": 0.9847754325282876, "validation_mean_false_alarms_per_hour": 21.716965742251222}}}, "tie_breaking": ["higher validation mean F2", "lower validation mean false alarms per hour", "fewer epochs", "lower learning rate"], "validation_calibration_policy": "one ADL calibration window per validation subject", "validation_threshold": 0.25} |
| timevae_epochs | 0 |
### `results/film_personalization/full_run_v1/timing_pilot_1x30/fine_tune_probability_shift_summary.csv`

- Shape: `8 rows x 8 columns`
- Missing cells: `0`
- Columns: `evaluation, method, label_group, windows, signed_mean_probability_shift, absolute_mean_probability_shift, threshold_crossings_vs_global, subject_seed_rows`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| windows | 8 | 6000 | 37231.2 | 35117.5 | 32097.6 | 72690 |
| signed_mean_probability_shift | 8 | -0.0189386 | -0.00341376 | -0.000688936 | 0.00655152 | -5.93081e-05 |
| absolute_mean_probability_shift | 8 | 5.93081e-05 | 0.0034496 | 0.00069062 | 0.00663309 | 0.0191727 |
| threshold_crossings_vs_global | 8 | 0 | 44.125 | 14 | 60.7676 | 168 |
| subject_seed_rows | 8 | 20 | 27.5 | 30 | 4.6291 | 30 |

First 10 rows:

| evaluation | method | label_group | windows | signed_mean_probability_shift | absolute_mean_probability_shift | threshold_crossings_vs_global | subject_seed_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | full_model | ADL | 61235 | -0.000528835 | 0.00053135 | 48 | 30 |
| real_heldout | full_model | Fall | 6000 | -0.00576167 | 0.00581088 | 2 | 20 |
| real_heldout | last_layer | ADL | 61235 | -5.93081e-05 | 5.93081e-05 | 5 | 30 |
| real_heldout | last_layer | Fall | 6000 | -0.000209551 | 0.000209551 | 0 | 20 |
| simulated_users | full_model | ADL | 72690 | -0.000849037 | 0.000849891 | 102 | 30 |
| simulated_users | full_model | Fall | 9000 | -0.0189386 | 0.0191727 | 168 | 30 |
| simulated_users | last_layer | ADL | 72690 | -8.81042e-05 | 8.81042e-05 | 8 | 30 |
| simulated_users | last_layer | Fall | 9000 | -0.000875043 | 0.000875043 | 20 | 30 |
### `results/film_personalization/full_run_v1/timing_pilot_1x30/fine_tune_probability_shifts.csv`

- Shape: `220 rows x 10 columns`
- Missing cells: `0`
- Columns: `evaluation, profile, subject_id, seed, method, label_group, windows, signed_mean_probability_shift, absolute_mean_probability_shift, threshold_crossings_vs_global`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| seed | 220 | 17 | 53.4 | 47 | 29.878 | 101 |
| windows | 220 | 300 | 1353.86 | 1277.5 | 1014.7 | 2423 |
| signed_mean_probability_shift | 220 | -0.0971945 | -0.00345269 | -0.000311276 | 0.0104921 | 0.000405759 |
| absolute_mean_probability_shift | 220 | 8.43808e-06 | 0.00348954 | 0.000315727 | 0.0105217 | 0.0971945 |
| threshold_crossings_vs_global | 220 | 0 | 1.60455 | 0 | 4.00718 | 35 |

First 10 rows:

| evaluation | profile | subject_id | seed | method | label_group | windows | signed_mean_probability_shift | absolute_mean_probability_shift | threshold_crossings_vs_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | Real | SA02 | 17 | last_layer | ADL | 2423 | -8.25989e-05 | 8.25989e-05 | 1 |
| real_heldout | Real | SA02 | 17 | last_layer | Fall | 300 | -0.000306373 | 0.000306373 | 0 |
| real_heldout | Real | SA02 | 17 | full_model | ADL | 2423 | -0.000810444 | 0.000810444 | 5 |
| real_heldout | Real | SA02 | 17 | full_model | Fall | 300 | -0.00748538 | 0.00748593 | 0 |
| real_heldout | Real | SA02 | 31 | last_layer | ADL | 2423 | -3.21066e-05 | 3.21066e-05 | 1 |
| real_heldout | Real | SA02 | 31 | last_layer | Fall | 300 | -8.98934e-05 | 8.98934e-05 | 0 |
| real_heldout | Real | SA02 | 31 | full_model | ADL | 2423 | -0.00085801 | 0.00085801 | 5 |
| real_heldout | Real | SA02 | 31 | full_model | Fall | 300 | -0.00787875 | 0.00787875 | 0 |
| real_heldout | Real | SA02 | 47 | last_layer | ADL | 2423 | -8.68134e-05 | 8.68134e-05 | 1 |
| real_heldout | Real | SA02 | 47 | last_layer | Fall | 300 | -0.000285005 | 0.000285005 | 0 |
### `results/film_personalization/full_run_v1/timing_pilot_1x30/meta_training_history.csv`

- Shape: `1 rows x 4 columns`
- Missing cells: `0`
- Columns: `epoch, train_loss, val_loss, epoch_seconds`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| epoch | 1 | 1 | 1 | 1 |  | 1 |
| train_loss | 1 | 0.145981 | 0.145981 | 0.145981 |  | 0.145981 |
| val_loss | 1 | 0.245065 | 0.245065 | 0.245065 |  | 0.245065 |
| epoch_seconds | 1 | 4.23491 | 4.23491 | 4.23491 |  | 4.23491 |

First 10 rows:

| epoch | train_loss | val_loss | epoch_seconds |
| --- | --- | --- | --- |
| 1 | 0.145981 | 0.245065 | 4.23491 |
### `results/film_personalization/full_run_v1/timing_pilot_1x30/metrics_summary.csv`

- Shape: `245 rows x 17 columns`
- Missing cells: `0`
- Columns: `evaluation, profile, model, k, threshold, accuracy_mean, accuracy_std, precision_mean, precision_std, recall_mean, recall_std, f1_mean, f1_std, f2_mean, f2_std, false_alarms_per_hour_mean, false_alarms_per_hour_std`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 245 | 0 | 9.2 | 5 | 11.0025 | 30 |
| threshold | 245 | 0.25 | 0.335714 | 0.25 | 0.210386 | 0.85 |
| accuracy_mean | 245 | 0.962541 | 0.988907 | 0.98993 | 0.00677449 | 0.997613 |
| accuracy_std | 245 | 0 | 0.000992096 | 0.000492707 | 0.00139691 | 0.0136533 |
| precision_mean | 245 | 0.752935 | 0.958976 | 0.963849 | 0.0381111 | 1 |
| precision_std | 245 | 0 | 0.00419048 | 0.00282863 | 0.00722052 | 0.0699177 |
| recall_mean | 245 | 0.66 | 0.941191 | 0.990667 | 0.0706455 | 1 |
| recall_std | 245 | 0 | 0.00402612 | 0.00150072 | 0.00608045 | 0.0369233 |
| f1_mean | 245 | 0.795181 | 0.947384 | 0.952001 | 0.0328795 | 0.983871 |
| f1_std | 245 | 0 | 0.00377946 | 0.00224486 | 0.005142 | 0.0454919 |
| f2_mean | 245 | 0.708155 | 0.94307 | 0.975923 | 0.0545117 | 0.989257 |
| f2_std | 245 | 0 | 0.00383135 | 0.00161499 | 0.00526437 | 0.0328876 |
| false_alarms_per_hour_mean | 245 | 0 | 23.4952 | 16.7148 | 25.3941 | 187.949 |
| false_alarms_per_hour_std | 245 | 0 | 3.4978 | 1.66113 | 6.76295 | 69.047 |

First 10 rows:

| evaluation | profile | model | k | threshold | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | f2_mean | f2_std | false_alarms_per_hour_mean | false_alarms_per_hour_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | Real | FiLM (canonical threshold) | 0 | 0.25 | 0.996401 | 0.00262423 | 0.957691 | 0.00532886 | 0.995 | 0.0038236 | 0.975973 | 0.0023231 | 0.987297 | 0.00256636 | 16.3434 | 12.0499 |
| real_heldout | Real | FiLM (canonical threshold) | 1 | 0.25 | 0.992178 | 0.00677858 | 0.905328 | 0.033151 | 0.9995 | 0.00122116 | 0.949776 | 0.0186327 | 0.978909 | 0.00803039 | 39.3727 | 34.1944 |
| real_heldout | Real | FiLM (canonical threshold) | 5 | 0.25 | 0.994038 | 0.00447672 | 0.925707 | 0.0114696 | 0.999167 | 0.00148087 | 0.961006 | 0.00669441 | 0.983536 | 0.00350071 | 29.839 | 22.2914 |
| real_heldout | Real | FiLM (canonical threshold) | 10 | 0.25 | 0.994296 | 0.00425522 | 0.92868 | 0.00979773 | 0.999167 | 0.00148087 | 0.962615 | 0.0059013 | 0.984213 | 0.00328217 | 28.539 | 21.15 |
| real_heldout | Real | FiLM (canonical threshold) | 30 | 0.25 | 0.994283 | 0.00424531 | 0.928523 | 0.00908052 | 0.999167 | 0.00148087 | 0.962534 | 0.00553101 | 0.98418 | 0.00314105 | 28.6009 | 21.1056 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 0 | 0.85 | 0.997062 | 0.00244934 | 0.986505 | 0.00649842 | 0.973333 | 0.0105409 | 0.979848 | 0.00707098 | 0.975923 | 0.00897354 | 4.95254 | 4.58392 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 1 | 0.85 | 0.997025 | 0.00250258 | 0.967516 | 0.01264 | 0.993 | 0.00388429 | 0.980052 | 0.00705134 | 0.98777 | 0.00435707 | 12.4433 | 10.8199 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 5 | 0.85 | 0.997491 | 0.00198138 | 0.974955 | 0.00672064 | 0.991333 | 0.00365148 | 0.983067 | 0.00451581 | 0.988008 | 0.00373198 | 9.47173 | 7.49813 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 10 | 0.85 | 0.997613 | 0.00190642 | 0.97718 | 0.00677364 | 0.990667 | 0.00298142 | 0.983871 | 0.00458399 | 0.987936 | 0.00348558 | 8.60504 | 6.94016 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 30 | 0.85 | 0.997576 | 0.00191579 | 0.976704 | 0.00687271 | 0.990667 | 0.00298142 | 0.983627 | 0.00439104 | 0.987837 | 0.00329387 | 8.79076 | 7.0835 |
### `results/film_personalization/full_run_v1/timing_pilot_1x30/paired_wilcoxon.csv`

- Shape: `10 rows x 6 columns`
- Missing cells: `0`
- Columns: `evaluation, metric, comparison, n_subjects, wilcoxon_statistic, p_value`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| n_subjects | 10 | 4 | 4 | 4 | 0 | 4 |
| wilcoxon_statistic | 10 | 0 | 2.4 | 2 | 2.54733 | 5 |
| p_value | 10 | 0.125 | 0.55 | 0.5 | 0.449537 | 1 |

First 10 rows:

| evaluation | metric | comparison | n_subjects | wilcoxon_statistic | p_value |
| --- | --- | --- | --- | --- | --- |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (+2 falls) | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (ADL-only) | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Global | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (+2 falls) | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (ADL-only) | 4 | 0 | 0.125 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (+2 falls) | 4 | 4 | 0.875 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (ADL-only) | 4 | 5 | 1 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Global | 4 | 5 | 1 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (+2 falls) | 4 | 5 | 1 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (ADL-only) | 4 | 5 | 1 |
### `results/film_personalization/full_run_v1/timing_pilot_1x30/per_user_metrics.csv`

- Shape: `2,000 rows x 16 columns`
- Missing cells: `1,000`
- Columns: `evaluation, profile, subject_id, model, k, seed, threshold, accuracy, precision, recall, f1, f2, false_alarms_per_hour, false_positive_windows, adl_windows, fall_windows`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 2000 | 0 | 9.2 | 5 | 10.9827 | 30 |
| seed | 2000 | 17 | 53.4 | 47 | 29.8175 | 101 |
| threshold | 2000 | 0.25 | 0.34 | 0.25 | 0.214296 | 0.85 |
| accuracy | 2000 | 0.947484 | 0.991699 | 0.994487 | 0.0067223 | 1 |
| precision | 1750 | 0.677201 | 0.95909 | 0.964286 | 0.0338096 | 1 |
| recall | 1750 | 0.66 | 0.957102 | 0.993289 | 0.064131 | 1 |
| f1 | 1750 | 0.795181 | 0.95608 | 0.97377 | 0.0312296 | 0.991708 |
| f2 | 1750 | 0.708155 | 0.956243 | 0.984646 | 0.050085 | 0.995355 |
| false_alarms_per_hour | 2000 | 0 | 21.0969 | 18.572 | 22.6061 | 265.58 |
| false_positive_windows | 2000 | 0 | 11.3595 | 10 | 12.1721 | 143 |
| adl_windows | 2000 | 1277 | 2279.81 | 2423 | 378.933 | 2423 |
| fall_windows | 2000 | 0 | 262 | 300 | 99.0551 | 300 |

First 10 rows:

| evaluation | profile | subject_id | model | k | seed | threshold | accuracy | precision | recall | f1 | f2 | false_alarms_per_hour | false_positive_windows | adl_windows | fall_windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | Real | SA02 | Global | 0 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Last-layer fine-tune (ADL-only) | 0 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Full fine-tune (ADL-only) | 0 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Last-layer fine-tune (+2 falls) | 0 | 17 | 0.25 | 0.994487 | 0.957929 | 0.993289 | 0.975288 | 0.986009 | 24.1436 | 13 | 2423 | 298 |
| real_heldout | Real | SA02 | Full fine-tune (+2 falls) | 0 | 17 | 0.25 | 0.994487 | 0.957929 | 0.993289 | 0.975288 | 0.986009 | 24.1436 | 13 | 2423 | 298 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 0 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 0 | 17 | 0.85 | 0.993022 | 0.979522 | 0.956667 | 0.96796 | 0.961152 | 11.1432 | 6 | 2423 | 300 |
| real_heldout | Real | SA02 | Global | 1 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Last-layer fine-tune (ADL-only) | 1 | 17 | 0.25 | 0.994859 | 0.96129 | 0.993333 | 0.977049 | 0.986755 | 22.2864 | 12 | 2423 | 300 |
| real_heldout | Real | SA02 | Full fine-tune (ADL-only) | 1 | 17 | 0.25 | 0.996328 | 0.973856 | 0.993333 | 0.983498 | 0.989376 | 14.8576 | 8 | 2423 | 300 |
### `results/film_personalization/full_run_v1/timing_pilot_1x30/per_user_seed_summary.csv`

- Shape: `400 rows x 24 columns`
- Missing cells: `400`
- Columns: `evaluation, profile, subject_id, model, k, threshold, accuracy_mean, accuracy_std, accuracy_count, precision_mean, precision_std, precision_count, recall_mean, recall_std, recall_count, f1_mean, f1_std, f1_count, f2_mean, f2_std, f2_count, false_alarms_per_hour_mean, false_alarms_per_hour_std, false_alarms_per_hour_count`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 400 | 0 | 9.2 | 5 | 10.9937 | 30 |
| threshold | 400 | 0.25 | 0.34 | 0.25 | 0.214511 | 0.85 |
| accuracy_mean | 400 | 0.962541 | 0.991699 | 0.994487 | 0.00664148 | 1 |
| accuracy_std | 400 | 0 | 0.000521315 | 0.000201295 | 0.00109128 | 0.0136533 |
| accuracy_count | 400 | 5 | 5 | 5 | 0 | 5 |
| precision_mean | 350 | 0.752935 | 0.95909 | 0.964052 | 0.0331881 | 1 |
| precision_std | 350 | 0 | 0.00315238 | 0.0017035 | 0.00673501 | 0.0699177 |
| precision_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| recall_mean | 350 | 0.66 | 0.957102 | 0.993289 | 0.0639775 | 1 |
| recall_std | 350 | 0 | 0.00261115 | 0 | 0.00543417 | 0.0369233 |
| recall_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| f1_mean | 350 | 0.795181 | 0.95608 | 0.97409 | 0.0308771 | 0.989696 |
| f1_std | 350 | 0 | 0.00271322 | 0.00138212 | 0.00477266 | 0.0454919 |
| f1_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| f2_mean | 350 | 0.708155 | 0.956243 | 0.983774 | 0.0499075 | 0.993768 |
| f2_std | 350 | 0 | 0.00259713 | 0.000703447 | 0.00475435 | 0.0328876 |
| f2_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| false_alarms_per_hour_mean | 400 | 0 | 21.0969 | 18.572 | 22.1055 | 187.949 |
| false_alarms_per_hour_std | 400 | 0 | 1.81201 | 0.830566 | 5.09578 | 69.047 |
| false_alarms_per_hour_count | 400 | 5 | 5 | 5 | 0 | 5 |

First 10 rows:

| evaluation | profile | subject_id | model | k | threshold | accuracy_mean | accuracy_std | accuracy_count | precision_mean | precision_std | precision_count | recall_mean | recall_std | recall_count | f1_mean | f1_std | f1_count | f2_mean | f2_std | f2_count | false_alarms_per_hour_mean | false_alarms_per_hour_std | false_alarms_per_hour_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 0 | 0.25 | 0.994491 | 0 | 5 | 0.958199 | 0 | 5 | 0.993333 | 0 | 5 | 0.97545 | 0 | 5 | 0.986102 | 0 | 5 | 24.1436 | 0 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 1 | 0.25 | 0.984576 | 0.00731264 | 5 | 0.879567 | 0.0509914 | 5 | 1 | 0 | 5 | 0.935297 | 0.0289305 | 5 | 0.972892 | 0.0125475 | 5 | 78.0025 | 36.9812 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 5 | 0.25 | 0.990599 | 0.00145976 | 5 | 0.921485 | 0.0111354 | 5 | 1 | 0 | 5 | 0.95911 | 0.00606156 | 5 | 0.983225 | 0.00255616 | 5 | 47.5444 | 7.38223 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 10 | 0.25 | 0.99148 | 0.000164236 | 5 | 0.928219 | 0.001282 | 5 | 1 | 0 | 5 | 0.962773 | 0.00069023 | 5 | 0.984769 | 0.000289016 | 5 | 43.0871 | 0.830566 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 30 | 0.25 | 0.991553 | 0 | 5 | 0.928793 | 0 | 5 | 1 | 0 | 5 | 0.963082 | 0 | 5 | 0.984898 | 0 | 5 | 42.7156 | 0 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 0 | 0.85 | 0.993022 | 0 | 5 | 0.979522 | 0 | 5 | 0.956667 | 0 | 5 | 0.96796 | 0 | 5 | 0.961152 | 0 | 5 | 11.1432 | 0 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 1 | 0.85 | 0.994198 | 0.00130874 | 5 | 0.960017 | 0.0124725 | 5 | 0.988667 | 0.00182574 | 5 | 0.974089 | 0.00565605 | 5 | 0.982773 | 0.00150762 | 5 | 23.0293 | 7.49813 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 5 | 0.85 | 0.995006 | 0.000328471 | 5 | 0.968593 | 0.00282863 | 5 | 0.986667 | 0 | 5 | 0.977545 | 0.00144216 | 5 | 0.982997 | 0.000583704 | 5 | 17.8291 | 1.66113 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 10 | 0.85 | 0.995226 | 0.000259679 | 5 | 0.970496 | 0.00225 | 5 | 0.986667 | 0 | 5 | 0.978513 | 0.00114366 | 5 | 0.983389 | 0.000462034 | 5 | 16.7148 | 1.31324 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 30 | 0.85 | 0.995299 | 0.000164236 | 5 | 0.97113 | 0.00142769 | 5 | 0.986667 | 0 | 5 | 0.978836 | 0.00072451 | 5 | 0.983519 | 0.00029241 | 5 | 16.3434 | 0.830566 | 5 |
### `results/film_personalization/prototype_run_final_smoke_v2/deployment_size.csv`

- Shape: `1 rows x 7 columns`
- Missing cells: `0`
- Columns: `base_tflite_bytes, film_tflite_bytes, tflite_overhead_bytes, user_affine_bytes, channels, base_tflite_max_abs_error, film_tflite_max_abs_error`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| base_tflite_bytes | 1 | 194896 | 194896 | 194896 |  | 194896 |
| film_tflite_bytes | 1 | 197940 | 197940 | 197940 |  | 197940 |
| tflite_overhead_bytes | 1 | 3044 | 3044 | 3044 |  | 3044 |
| user_affine_bytes | 1 | 512 | 512 | 512 |  | 512 |
| channels | 1 | 64 | 64 | 64 |  | 64 |
| base_tflite_max_abs_error | 1 | 5.96046e-08 | 5.96046e-08 | 5.96046e-08 |  | 5.96046e-08 |
| film_tflite_max_abs_error | 1 | 0 | 0 | 0 |  | 0 |

First 10 rows:

| base_tflite_bytes | film_tflite_bytes | tflite_overhead_bytes | user_affine_bytes | channels | base_tflite_max_abs_error | film_tflite_max_abs_error |
| --- | --- | --- | --- | --- | --- | --- |
| 194896 | 197940 | 3044 | 512 | 64 | 5.96046e-08 | 0 |
### `results/film_personalization/prototype_run_final_smoke_v2/experiment_config.csv`

- Shape: `22 rows x 2 columns`
- Missing cells: `0`
- Columns: `key, value`

Numeric statistics:

_No rows._

First 10 rows:

| key | value |
| --- | --- |
| seed | 42 |
| calibration_seeds | [17, 31, 47, 71, 101] |
| calibration_sizes | [0, 1, 5, 10, 30] |
| epochs | 2 |
| episodes_per_epoch | 3 |
| validation_episodes | 2 |
| patience | 1 |
| fine_tune_epochs | 1 |
| timevae_epochs | 0 |
| sample_rate_hz | 20.0 |
### `results/film_personalization/prototype_run_final_smoke_v2/meta_training_history.csv`

- Shape: `2 rows x 3 columns`
- Missing cells: `0`
- Columns: `epoch, train_loss, val_loss`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| epoch | 2 | 1 | 1.5 | 1.5 | 0.707107 | 2 |
| train_loss | 2 | 0.0182485 | 0.115525 | 0.115525 | 0.13757 | 0.212802 |
| val_loss | 2 | 0.741122 | 0.954821 | 0.954821 | 0.302216 | 1.16852 |

First 10 rows:

| epoch | train_loss | val_loss |
| --- | --- | --- |
| 1 | 0.0182485 | 1.16852 |
| 2 | 0.212802 | 0.741122 |
### `results/film_personalization/prototype_run_final_smoke_v2/metrics_summary.csv`

- Shape: `245 rows x 17 columns`
- Missing cells: `0`
- Columns: `evaluation, profile, model, k, threshold, accuracy_mean, accuracy_std, precision_mean, precision_std, recall_mean, recall_std, f1_mean, f1_std, f2_mean, f2_std, false_alarms_per_hour_mean, false_alarms_per_hour_std`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 245 | 0 | 9.2 | 5 | 11.0025 | 30 |
| threshold | 245 | 0.2 | 0.242857 | 0.25 | 0.0175322 | 0.25 |
| accuracy_mean | 245 | 0.976129 | 0.989137 | 0.989791 | 0.00628125 | 0.997111 |
| accuracy_std | 245 | 0 | 0.000667809 | 0.000479177 | 0.000690355 | 0.00265608 |
| precision_mean | 245 | 0.88969 | 0.962546 | 0.963579 | 0.0304048 | 0.99708 |
| precision_std | 245 | 0 | 0.00289297 | 0.0030457 | 0.00226051 | 0.00858126 |
| recall_mean | 245 | 0.788667 | 0.938689 | 0.993289 | 0.0699966 | 1 |
| recall_std | 245 | 0 | 0.0025541 | 0 | 0.00314052 | 0.0121564 |
| f1_mean | 245 | 0.879209 | 0.948177 | 0.951612 | 0.030694 | 0.98318 |
| f1_std | 245 | 0 | 0.00232472 | 0.00210624 | 0.00170075 | 0.00709194 |
| f2_mean | 245 | 0.822545 | 0.941945 | 0.976436 | 0.053865 | 0.989245 |
| f2_std | 245 | 0 | 0.00237138 | 0.00122292 | 0.00247505 | 0.0100732 |
| false_alarms_per_hour_mean | 245 | 1.48576 | 20.9212 | 16.7148 | 19.264 | 69.0879 |
| false_alarms_per_hour_std | 245 | 0 | 2.48002 | 1.66113 | 3.14728 | 12.2756 |

First 10 rows:

| evaluation | profile | model | k | threshold | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | f2_mean | f2_std | false_alarms_per_hour_mean | false_alarms_per_hour_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | Real | FiLM (canonical threshold) | 0 | 0.25 | 0.996707 | 0.00245594 | 0.961539 | 0.00569569 | 0.995 | 0.0038236 | 0.977972 | 0.00354615 | 0.988115 | 0.00329407 | 14.7957 | 10.998 |
| real_heldout | Real | FiLM (canonical threshold) | 1 | 0.25 | 0.997111 | 0.0021581 | 0.966827 | 0.0053317 | 0.994833 | 0.00411459 | 0.980618 | 0.00320058 | 0.989095 | 0.00331394 | 12.6909 | 9.4835 |
| real_heldout | Real | FiLM (canonical threshold) | 5 | 0.25 | 0.996989 | 0.00224402 | 0.965116 | 0.0055621 | 0.995 | 0.0038236 | 0.979817 | 0.00320842 | 0.988868 | 0.00309507 | 13.3719 | 9.98825 |
| real_heldout | Real | FiLM (canonical threshold) | 10 | 0.25 | 0.996964 | 0.0022678 | 0.964804 | 0.00568436 | 0.995 | 0.0038236 | 0.979657 | 0.00334887 | 0.988802 | 0.00316327 | 13.4957 | 10.0893 |
| real_heldout | Real | FiLM (canonical threshold) | 30 | 0.25 | 0.996964 | 0.00226369 | 0.964806 | 0.0057539 | 0.995 | 0.0038236 | 0.979657 | 0.00325897 | 0.988802 | 0.00308217 | 13.4957 | 10.1011 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 0 | 0.2 | 0.99634 | 0.00265608 | 0.956928 | 0.00538417 | 0.995 | 0.0038236 | 0.975574 | 0.00187944 | 0.987133 | 0.00229215 | 16.6529 | 12.2756 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 1 | 0.2 | 0.996915 | 0.00229566 | 0.964191 | 0.0061601 | 0.995 | 0.0038236 | 0.979337 | 0.00320367 | 0.988671 | 0.00293582 | 13.7433 | 10.3301 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 5 | 0.2 | 0.99667 | 0.00247377 | 0.961087 | 0.00625685 | 0.995 | 0.0038236 | 0.977734 | 0.00335822 | 0.988017 | 0.00301196 | 14.9814 | 11.2064 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 10 | 0.2 | 0.996646 | 0.00249399 | 0.96078 | 0.00638922 | 0.995 | 0.0038236 | 0.977574 | 0.00342525 | 0.987951 | 0.00302461 | 15.1052 | 11.3099 |
| real_heldout | Real | FiLM (validation-retuned threshold) | 30 | 0.2 | 0.996609 | 0.00251987 | 0.960315 | 0.00635434 | 0.995 | 0.0038236 | 0.977334 | 0.00343084 | 0.987853 | 0.00303719 | 15.291 | 11.4349 |
### `results/film_personalization/prototype_run_final_smoke_v2/paired_wilcoxon.csv`

- Shape: `10 rows x 6 columns`
- Missing cells: `0`
- Columns: `evaluation, metric, comparison, n_subjects, wilcoxon_statistic, p_value`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| n_subjects | 10 | 4 | 4 | 4 | 0 | 4 |
| wilcoxon_statistic | 10 | 0 | 1.9 | 1.5 | 2.079 | 5 |
| p_value | 10 | 0.125 | 0.4625 | 0.375 | 0.372911 | 1 |

First 10 rows:

| evaluation | metric | comparison | n_subjects | wilcoxon_statistic | p_value |
| --- | --- | --- | --- | --- | --- |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (+2 falls) | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (ADL-only) | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Global | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (+2 falls) | 4 | 0 | 0.125 |
| real_heldout | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (ADL-only) | 4 | 0 | 0.125 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (+2 falls) | 4 | 3 | 0.625 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Full fine-tune (ADL-only) | 4 | 5 | 1 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Global | 4 | 4 | 0.875 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (+2 falls) | 4 | 3 | 0.625 |
| simulated_users | subject-mean F1 across k and seeds | FiLM vs Last-layer fine-tune (ADL-only) | 4 | 4 | 0.875 |
### `results/film_personalization/prototype_run_final_smoke_v2/per_user_metrics.csv`

- Shape: `2,000 rows x 16 columns`
- Missing cells: `1,000`
- Columns: `evaluation, profile, subject_id, model, k, seed, threshold, accuracy, precision, recall, f1, f2, false_alarms_per_hour, false_positive_windows, adl_windows, fall_windows`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 2000 | 0 | 9.2 | 5 | 10.9827 | 30 |
| seed | 2000 | 17 | 53.4 | 47 | 29.8175 | 101 |
| threshold | 2000 | 0.2 | 0.2425 | 0.25 | 0.017858 | 0.25 |
| accuracy | 2000 | 0.975395 | 0.991941 | 0.994491 | 0.00625208 | 1 |
| precision | 1750 | 0.886905 | 0.962171 | 0.964286 | 0.0257644 | 1 |
| recall | 1750 | 0.78 | 0.955578 | 0.993333 | 0.064009 | 1 |
| f1 | 1750 | 0.874766 | 0.957075 | 0.975288 | 0.0291657 | 0.985222 |
| f2 | 1750 | 0.815331 | 0.955777 | 0.985705 | 0.0498029 | 0.994036 |
| false_alarms_per_hour | 2000 | 0 | 19.1292 | 18.572 | 16.7986 | 70.5737 |
| false_positive_windows | 2000 | 0 | 10.3 | 10 | 9.0451 | 38 |
| adl_windows | 2000 | 1277 | 2279.81 | 2423 | 378.933 | 2423 |
| fall_windows | 2000 | 0 | 262 | 300 | 99.0551 | 300 |

First 10 rows:

| evaluation | profile | subject_id | model | k | seed | threshold | accuracy | precision | recall | f1 | f2 | false_alarms_per_hour | false_positive_windows | adl_windows | fall_windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | Real | SA02 | Global | 0 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Last-layer fine-tune (ADL-only) | 0 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Full fine-tune (ADL-only) | 0 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Last-layer fine-tune (+2 falls) | 0 | 17 | 0.25 | 0.994487 | 0.957929 | 0.993289 | 0.975288 | 0.986009 | 24.1436 | 13 | 2423 | 298 |
| real_heldout | Real | SA02 | Full fine-tune (+2 falls) | 0 | 17 | 0.25 | 0.994487 | 0.957929 | 0.993289 | 0.975288 | 0.986009 | 24.1436 | 13 | 2423 | 298 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 0 | 17 | 0.25 | 0.994859 | 0.96129 | 0.993333 | 0.977049 | 0.986755 | 22.2864 | 12 | 2423 | 300 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 0 | 17 | 0.2 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Global | 1 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Last-layer fine-tune (ADL-only) | 1 | 17 | 0.25 | 0.994491 | 0.958199 | 0.993333 | 0.97545 | 0.986102 | 24.1436 | 13 | 2423 | 300 |
| real_heldout | Real | SA02 | Full fine-tune (ADL-only) | 1 | 17 | 0.25 | 0.994859 | 0.96129 | 0.993333 | 0.977049 | 0.986755 | 22.2864 | 12 | 2423 | 300 |
### `results/film_personalization/prototype_run_final_smoke_v2/per_user_seed_summary.csv`

- Shape: `400 rows x 24 columns`
- Missing cells: `400`
- Columns: `evaluation, profile, subject_id, model, k, threshold, accuracy_mean, accuracy_std, accuracy_count, precision_mean, precision_std, precision_count, recall_mean, recall_std, recall_count, f1_mean, f1_std, f1_count, f2_mean, f2_std, f2_count, false_alarms_per_hour_mean, false_alarms_per_hour_std, false_alarms_per_hour_count`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 400 | 0 | 9.2 | 5 | 10.9937 | 30 |
| threshold | 400 | 0.2 | 0.2425 | 0.25 | 0.0178759 | 0.25 |
| accuracy_mean | 400 | 0.976129 | 0.991941 | 0.994491 | 0.00624666 | 1 |
| accuracy_std | 400 | 0 | 0.00027072 | 0.000164236 | 0.000330517 | 0.00128797 |
| accuracy_count | 400 | 5 | 5 | 5 | 0 | 5 |
| precision_mean | 350 | 0.88969 | 0.962171 | 0.964286 | 0.0256866 | 0.99708 |
| precision_std | 350 | 0 | 0.00177878 | 0.00136885 | 0.00193117 | 0.00858126 |
| precision_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| recall_mean | 350 | 0.788667 | 0.955578 | 0.993333 | 0.0640198 | 1 |
| recall_std | 350 | 0 | 0.00140821 | 0 | 0.00283094 | 0.0121564 |
| recall_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| f1_mean | 350 | 0.879209 | 0.957075 | 0.975288 | 0.0291332 | 0.984253 |
| f1_std | 350 | 0 | 0.00145067 | 0.000881497 | 0.0016397 | 0.00709194 |
| f1_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| f2_mean | 350 | 0.822545 | 0.955777 | 0.985509 | 0.0498021 | 0.993641 |
| f2_std | 350 | 0 | 0.00140819 | 0.00036004 | 0.00228439 | 0.0100732 |
| f2_count | 400 | 0 | 4.375 | 5 | 1.65567 | 5 |
| false_alarms_per_hour_mean | 400 | 0 | 19.1292 | 19.3149 | 16.7627 | 69.0879 |
| false_alarms_per_hour_std | 400 | 0 | 0.922268 | 0.415283 | 1.16595 | 5.72428 |
| false_alarms_per_hour_count | 400 | 5 | 5 | 5 | 0 | 5 |

First 10 rows:

| evaluation | profile | subject_id | model | k | threshold | accuracy_mean | accuracy_std | accuracy_count | precision_mean | precision_std | precision_count | recall_mean | recall_std | recall_count | f1_mean | f1_std | f1_count | f2_mean | f2_std | f2_count | false_alarms_per_hour_mean | false_alarms_per_hour_std | false_alarms_per_hour_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 0 | 0.25 | 0.994859 | 0 | 5 | 0.96129 | 0 | 5 | 0.993333 | 0 | 5 | 0.977049 | 0 | 5 | 0.986755 | 0 | 5 | 22.2864 | 0 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 1 | 0.25 | 0.996107 | 0.000492707 | 5 | 0.971965 | 0.00422837 | 5 | 0.993333 | 0 | 5 | 0.982529 | 0.00216667 | 5 | 0.988982 | 0.000879647 | 5 | 15.9719 | 2.4917 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 5 | 0.25 | 0.995887 | 0.000478825 | 5 | 0.970066 | 0.0041116 | 5 | 0.993333 | 0 | 5 | 0.981558 | 0.00210624 | 5 | 0.988589 | 0.000854963 | 5 | 17.0863 | 2.42149 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 10 | 0.25 | 0.99574 | 0.00055695 | 5 | 0.968809 | 0.00478105 | 5 | 0.993333 | 0 | 5 | 0.980913 | 0.00244953 | 5 | 0.988327 | 0.0009944 | 5 | 17.8291 | 2.81659 | 5 |
| real_heldout | Real | SA02 | FiLM (canonical threshold) | 30 | 0.25 | 0.995887 | 0.000307257 | 5 | 0.970058 | 0.00264419 | 5 | 0.993333 | 0 | 5 | 0.981556 | 0.00135306 | 5 | 0.988588 | 0.000548868 | 5 | 17.0863 | 1.55385 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 0 | 0.2 | 0.994491 | 0 | 5 | 0.958199 | 0 | 5 | 0.993333 | 0 | 5 | 0.97545 | 0 | 5 | 0.986102 | 0 | 5 | 24.1436 | 0 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 1 | 0.2 | 0.995813 | 0.000715886 | 5 | 0.969452 | 0.00613013 | 5 | 0.993333 | 0 | 5 | 0.98124 | 0.00314457 | 5 | 0.988458 | 0.00127752 | 5 | 17.4577 | 3.62035 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 5 | 0.2 | 0.995226 | 0.000367242 | 5 | 0.964409 | 0.00312108 | 5 | 0.993333 | 0 | 5 | 0.978656 | 0.00160699 | 5 | 0.987409 | 0.000654347 | 5 | 20.4292 | 1.8572 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 10 | 0.2 | 0.995152 | 0.000402293 | 5 | 0.963787 | 0.00341896 | 5 | 0.993333 | 0 | 5 | 0.978335 | 0.00176037 | 5 | 0.987278 | 0.000716801 | 5 | 20.8007 | 2.03446 | 5 |
| real_heldout | Real | SA02 | FiLM (validation-retuned threshold) | 30 | 0.2 | 0.995006 | 0.000201147 | 5 | 0.962535 | 0.00170395 | 5 | 0.993333 | 0 | 5 | 0.977691 | 0.000878739 | 5 | 0.987017 | 0.000358163 | 5 | 21.5435 | 1.01723 | 5 |

## 8. How to Continue

1. Treat `train.npy`, `val.npy`, and `test.npy` plus their aligned metadata arrays as the current binary modeling dataset.
2. Build a severity-aware CNN-LSTM with a dedicated severity-prediction head; the current detector predicts only binary fall/background.
3. Validate the CNN-LSTM and FiLM prototype on additional real-world subjects and sensor placements.
4. Continue reporting window-, recording-, subject-level, and impact-verified performance separately.
5. Keep all transformations train-only: preprocessing scaler, severity feature scaler, KMeans, and Isolation Forest.
6. Record the exact artifact generation command and timestamp whenever the processed arrays or severity outputs are regenerated.

## 9. Key Files

| Purpose | File |
| --- | --- |
| Consolidated guide | `docs/reports/PROJECT_STATUS_REPORT.md` |
| Rebuild this guide | `scripts/generate_project_status_report.py` |
| Raw loader | `src/data/sisfall_loader.py` |
| Preprocessing/windowing | `src/data/sisfall_preprocessing.py` |
| Severity labeling | `src/data/severity_labeling.py` |
| Anomaly refinement | `src/data/severity_anomaly.py` |
| Tests | `tests/` |

