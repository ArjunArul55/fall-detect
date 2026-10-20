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
14. **Automated testing:** pytest run on 2026-10-20: **36 tests** (36 passed, 0 failed, 0 errors; overall passed).

## 3. Current Dataset Statistics

| split | windows | shape | subjects | recordings | ADL/background | fall | fall_% | NaN | Inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 55423 | (55423, 64, 6) | 26 | 2942 | 50939 | 4484 | 8.0905 | 0 | 0 |
| val | 15073 | (15073, 64, 6) | 6 | 829 | 13573 | 1500 | 9.95157 | 0 | 0 |
| test | 13627 | (13627, 64, 6) | 6 | 734 | 12427 | 1200 | 8.80605 | 0 | 0 |

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
| data/processed/test.npy | (13627, 64, 6) | float32 | -48.5212 | 20.0049 |
| data/processed/test_labels.npy | (13627,) | int64 | 0 | 1 |
| data/processed/test_recording_ids.npy | (13627,) | <U21 |  |  |
| data/processed/test_subject_ids.npy | (13627,) | <U4 |  |  |
| data/processed/train.npy | (55423, 64, 6) | float32 | -38.0956 | 38.1595 |
| data/processed/train_labels.npy | (55423,) | int64 | 0 | 1 |
| data/processed/train_recording_ids.npy | (55423,) | <U21 |  |  |
| data/processed/train_subject_ids.npy | (55423,) | <U4 |  |  |
| data/processed/val.npy | (15073, 64, 6) | float32 | -18.9812 | 18.5895 |
| data/processed/val_labels.npy | (15073,) | int64 | 0 | 1 |
| data/processed/val_recording_ids.npy | (15073,) | <U21 |  |  |
| data/processed/val_subject_ids.npy | (15073,) | <U4 |  |  |
| data/processed/X_train_balanced.npy | (59907, 64, 6) | float32 | -39.861 | 38.1595 |
| data/processed/y_train_balanced.npy | (59907,) | int64 | 0 | 1 |

`train.npy`, `val.npy`, and `test.npy` are the primary model inputs. Their matching label, subject, and recording arrays must remain aligned by row. `X_train_balanced.npy` and `y_train_balanced.npy` are additional class-balancing artifacts and should be used only when an experiment explicitly selects them.

### Pickled estimator inventory

| artifact | python_type | details |
| --- | --- | --- |
| data/processed/scaler.pkl | StandardScaler | features=6 |
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

- Shape: `734 rows x 5 columns`
- Missing cells: `0`
- Columns: `recording_id, subject_id, true_fall_recording, predicted_positive, two_consecutive_alert`

Numeric statistics:

_No rows._

First 10 rows:

| recording_id | subject_id | true_fall_recording | predicted_positive | two_consecutive_alert |
| --- | --- | --- | --- | --- |
| SA02:D01_SA02_R01.txt | SA02 | False | False | False |
| SA02:D02_SA02_R01.txt | SA02 | False | False | False |
| SA02:D03_SA02_R01.txt | SA02 | False | False | False |
| SA02:D04_SA02_R01.txt | SA02 | False | False | False |
| SA02:D05_SA02_R01.txt | SA02 | False | False | False |
| SA02:D05_SA02_R02.txt | SA02 | False | False | False |
| SA02:D05_SA02_R03.txt | SA02 | False | False | False |
| SA02:D05_SA02_R04.txt | SA02 | False | False | False |
| SA02:D05_SA02_R05.txt | SA02 | False | False | False |
| SA02:D06_SA02_R01.txt | SA02 | False | False | False |
### `results/cnn_lstm_baseline/severity_eval/k2_20260914T055946Z/recording_metrics_by_severity.csv`

- Shape: `325 rows x 9 columns`
- Missing cells: `0`
- Columns: `severity_label, severity, recording_id, subject_id, severity_windows, detected_severity_windows, recording_detected, impact_verified, two_consecutive_alert`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| severity_label | 325 | -1 | 0.532308 | 1 | 0.574431 | 1 |
| severity_windows | 325 | 1 | 3.69231 | 4 | 0.815041 | 4 |
| detected_severity_windows | 325 | 0 | 3.66462 | 4 | 0.843353 | 4 |

First 10 rows:

| severity_label | severity | recording_id | subject_id | severity_windows | detected_severity_windows | recording_detected | impact_verified | two_consecutive_alert |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -1 | Uncertain (-1) | SA02:F01_SA02_R01.txt | SA02 | 4 | 4 | True | True | True |
| -1 | Uncertain (-1) | SA02:F05_SA02_R02.txt | SA02 | 1 | 1 | True | True | False |
| -1 | Uncertain (-1) | SA02:F05_SA02_R03.txt | SA02 | 4 | 4 | True | True | True |
| -1 | Uncertain (-1) | SA02:F05_SA02_R04.txt | SA02 | 4 | 4 | True | True | True |
| -1 | Uncertain (-1) | SA02:F05_SA02_R05.txt | SA02 | 1 | 1 | True | True | False |
| -1 | Uncertain (-1) | SA02:F14_SA02_R05.txt | SA02 | 1 | 1 | True | True | False |
| -1 | Uncertain (-1) | SA03:F01_SA03_R03.txt | SA03 | 1 | 1 | True | True | False |
| -1 | Uncertain (-1) | SA03:F08_SA03_R04.txt | SA03 | 1 | 1 | True | True | False |
| -1 | Uncertain (-1) | SA03:F08_SA03_R05.txt | SA03 | 1 | 1 | True | True | False |
| -1 | Uncertain (-1) | SA09:F04_SA09_R01.txt | SA09 | 1 | 1 | True | True | False |
### `results/cnn_lstm_baseline/severity_eval/k2_20260914T055946Z/recording_summary_by_severity.csv`

- Shape: `3 rows x 6 columns`
- Missing cells: `0`
- Columns: `severity_label, severity, recordings, detected_recordings, impact_verified_recordings, two_consecutive_recordings`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| severity_label | 3 | -1 | 0 | 0 | 1 | 1 |
| recordings | 3 | 13 | 108.333 | 126 | 87.8427 | 186 |
| detected_recordings | 3 | 12 | 107.667 | 125 | 88.2855 | 186 |
| impact_verified_recordings | 3 | 12 | 107.667 | 125 | 88.2855 | 186 |
| two_consecutive_recordings | 3 | 4 | 100.333 | 113 | 90.6661 | 184 |

First 10 rows:

| severity_label | severity | recordings | detected_recordings | impact_verified_recordings | two_consecutive_recordings |
| --- | --- | --- | --- | --- | --- |
| -1 | Uncertain (-1) | 13 | 12 | 12 | 4 |
| 0 | Mild | 126 | 125 | 125 | 113 |
| 1 | Severe | 186 | 186 | 186 | 184 |
### `results/cnn_lstm_baseline/severity_eval/k2_20260914T055946Z/subject_metrics_by_severity.csv`

- Shape: `12 rows x 7 columns`
- Missing cells: `0`
- Columns: `severity_label, severity, subject_id, severity_windows, detected_severity_windows, subject_recall, impact_verified`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| severity_label | 12 | -1 | 0 | 0 | 0.852803 | 1 |
| severity_windows | 12 | 3 | 100 | 116.5 | 75.6643 | 201 |
| detected_severity_windows | 12 | 2 | 99.25 | 115 | 75.5515 | 201 |
| subject_recall | 12 | 0.666667 | 0.966562 | 1 | 0.0949374 | 1 |

First 10 rows:

| severity_label | severity | subject_id | severity_windows | detected_severity_windows | subject_recall | impact_verified |
| --- | --- | --- | --- | --- | --- | --- |
| -1 | Uncertain (-1) | SA02 | 15 | 15 | 1 | True |
| -1 | Uncertain (-1) | SA03 | 3 | 3 | 1 | True |
| -1 | Uncertain (-1) | SA09 | 3 | 2 | 0.666667 | True |
| -1 | Uncertain (-1) | SA14 | 3 | 3 | 1 | True |
| 0 | Mild | SA02 | 109 | 108 | 0.990826 | True |
| 0 | Mild | SA03 | 128 | 128 | 1 | True |
| 0 | Mild | SA09 | 124 | 122 | 0.983871 | True |
| 0 | Mild | SA14 | 96 | 93 | 0.96875 | True |
| 1 | Severe | SA02 | 176 | 174 | 0.988636 | True |
| 1 | Severe | SA03 | 169 | 169 | 1 | True |
### `results/cnn_lstm_baseline/severity_eval/k2_20260914T055946Z/subject_summary_by_severity.csv`

- Shape: `3 rows x 6 columns`
- Missing cells: `0`
- Columns: `severity_label, severity, subjects, mean_subject_recall, min_subject_recall, impact_verified_subjects`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| severity_label | 3 | -1 | 0 | 0 | 1 | 1 |
| subjects | 3 | 4 | 4 | 4 | 0 | 4 |
| mean_subject_recall | 3 | 0.916667 | 0.966562 | 0.985862 | 0.0435787 | 0.997159 |
| min_subject_recall | 3 | 0.666667 | 0.874684 | 0.96875 | 0.180423 | 0.988636 |
| impact_verified_subjects | 3 | 4 | 4 | 4 | 0 | 4 |

First 10 rows:

| severity_label | severity | subjects | mean_subject_recall | min_subject_recall | impact_verified_subjects |
| --- | --- | --- | --- | --- | --- |
| -1 | Uncertain (-1) | 4 | 0.916667 | 0.666667 | 4 |
| 0 | Mild | 4 | 0.985862 | 0.96875 | 4 |
| 1 | Severe | 4 | 0.997159 | 0.988636 | 4 |
### `results/cnn_lstm_baseline/severity_eval/k2_20260914T055946Z/window_metrics_by_severity.csv`

- Shape: `3 rows x 7 columns`
- Missing cells: `0`
- Columns: `severity_label, severity, windows, detected_windows, window_recall, window_precision, window_f1`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| severity_label | 3 | -1 | 0 | 0 | 1 | 1 |
| windows | 3 | 24 | 400 | 457 | 350.989 | 719 |
| detected_windows | 3 | 23 | 397 | 451 | 350.137 | 717 |
| window_recall | 3 | 0.958333 | 0.980808 | 0.986871 | 0.0201391 | 0.997218 |
| window_precision | 3 | 0.0188679 | 0.325677 | 0.369975 | 0.287233 | 0.588187 |
| window_f1 | 3 | 0.0370072 | 0.438377 | 0.538186 | 0.361938 | 0.739938 |

First 10 rows:

| severity_label | severity | windows | detected_windows | window_recall | window_precision | window_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| -1 | Uncertain (-1) | 24 | 23 | 0.958333 | 0.0188679 | 0.0370072 |
| 0 | Mild | 457 | 451 | 0.986871 | 0.369975 | 0.538186 |
| 1 | Severe | 719 | 717 | 0.997218 | 0.588187 | 0.739938 |
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

