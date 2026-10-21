# Fall Detection

AI-powered wearable fall-detection research using the [SisFall](https://www.sisfall.com/) dataset. The project covers raw data ingestion, preprocessing, exploratory analysis, fall-severity labeling, anomaly detection, and supervised CNN/CNN-LSTM fall-detection baselines.

The current processed dataset contains 84,123 windows of shape `(64, 6)`: 76,939 ADL/background windows and 7,184 impact-centered fall windows. Each window represents 3.2 seconds sampled at 20 Hz with six standardized IMU channels: `acc1_x`, `acc1_y`, `acc1_z`, `gyro_x`, `gyro_y`, and `gyro_z`.

## Current Status

- Raw SisFall parsing, metadata construction, filtering, downsampling, subject-wise splitting, normalization, and sliding-window generation are implemented.
- Fall labels are impact-centered: only windows containing the peak fall impact are labeled as fall.
- Canonical severity labels use K=2 run `k2_20260914T055946Z`: Mild, Severe, and Uncertain (-1). Moderate/K=3 is superseded exploration and is not canonical.
- Two supervised baselines are available: a compact 1D CNN and a compact CNN-LSTM.
- The headline model-comparison threshold is `0.25`, selected on validation by maximum F2 and applied unchanged to test. The FiLM sensitivity row reports that run's validation-retuned threshold as selected; `0.50` is a reference point.

For the presentation roadmap, see [docs/PRESENTATION_GUIDE.md](docs/PRESENTATION_GUIDE.md). For the full generated report, metrics, artifact inventory, and CSV samples, see [docs/reports/PROJECT_STATUS_REPORT.md](docs/reports/PROJECT_STATUS_REPORT.md).

## Repository Layout

```text
.
├── data/
│   ├── raw/SisFall_dataset/         # Downloaded raw SisFall files (local, not tracked)
│   └── processed/                   # Arrays, scaler, EDA, severity and verification artifacts
├── docs/
│   ├── presentation/                # Presentation figures
│   └── reports/                     # Main report, technical appendix, and generated assets
├── results/
│   ├── cnn_baseline/                # CNN checkpoint, metrics, and evaluation
│   ├── cnn_lstm_baseline/           # CNN-LSTM checkpoint, metrics, and evaluation
│   └── film_personalization/        # Separate personalization experiment and run outputs
├── scripts/                         # Runnable preprocessing, training, analysis, and report commands
├── src/
│   ├── data/                        # SisFall loading, preprocessing, severity, anomaly modules
│   └── personalization/             # FiLM personalization prototype
├── tests/                           # Pytest coverage
├── requirements.txt
└── README.md                        # Setup and command-line entry point
```

Keep `data/processed/` and `results/` at these paths: the training, evaluation, and analysis scripts use them as their artifact locations. Reports and figures intended for review are grouped under `docs/`. The workspace hides the large raw dataset, archived severity outputs, and Python environment/cache folders from the Explorer view.

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Python dependencies include NumPy, pandas, SciPy, scikit-learn, matplotlib, tqdm, pytest, and TensorFlow.

## Dataset

Download the SisFall dataset from the official source and extract it as:

```text
data/raw/SisFall_dataset/
├── SA01/
├── SA02/
├── SE01/
└── ...
```

The preprocessing scripts expect the dataset at `data/raw/SisFall_dataset/`. Raw dataset files are not tracked in git.

## Pipeline

Run the full preprocessing pipeline:

```bash
python scripts/run_full_preprocessing.py
```

This generates the primary model inputs under `data/processed/`:

- `train.npy`, `val.npy`, `test.npy`
- `train_labels.npy`, `val_labels.npy`, `test_labels.npy`
- `train_subject_ids.npy`, `val_subject_ids.npy`, `test_subject_ids.npy`
- `train_recording_ids.npy`, `val_recording_ids.npy`, `test_recording_ids.npy`
- `scaler.pkl`

Run EDA and verification:

```bash
python scripts/run_phase2_eda.py
python scripts/verify_preprocessing_eda_pipeline.py
python scripts/inspect_processed_data.py
```

Generate severity labels and anomaly outputs:

```bash
python scripts/run_severity_labeling.py
python scripts/run_severity_anomaly_analysis.py
python scripts/validate_severity_clusters.py
python scripts/run_cluster_plots.py
```

Severity artifacts are written to `data/processed/severity/`, including refined severity labels, anomaly scores/flags, clustering features, KMeans/Isolation Forest models, CSV reports, and plots.
The canonical severity scheme is K=2: Mild, Severe, and Uncertain (-1). Moderate/K=3 artifacts are superseded and archived only.

## Model Training

Train the compact CNN baseline:

```bash
python scripts/train_cnn_baseline.py
```

Train the compact CNN-LSTM baseline:

```bash
python scripts/train_cnn_lstm_baseline.py
```

Evaluate recording-level behavior for the CNN baseline:

```bash
python scripts/evaluate_recording_level.py
```

Model outputs are saved under `results/cnn_baseline/` and `results/cnn_lstm_baseline/`.

## FiLM Personalization Prototype

Run the personalized-user experiment without changing the canonical model or split:

```powershell
.venv\Scripts\python.exe scripts\run_film_personalization.py
```

The script loads the frozen CNN-LSTM only through `src/keras_model_loading.py`, verifies
the CNN/LSTM split on 1,000 windows, then meta-trains a zero-initialized FiLM UserEncoder
on train-subject episodes. The CNN-LSTM backbone is frozen; a small user encoder is trained
once on simulated users; per-user adaptation at deployment is inference-only (no gradient
steps). The canonical synthetic-user protocol applies parametric per-channel gain/offset
transformations to SisFall windows (`x_sim = x * g + b`). It uses the stored 26/6/6
train/validation/test subject split, five calibration-sampling seeds, and calibration sizes
0/1/5/10/30. TimeGAN/TimeVAE is future work only, not part of the canonical experiment.
The headline comparison uses threshold `0.25` for all models, selected by maximum F2 on
validation and applied unchanged to test. The FiLM sensitivity row reports that run's
validation-retuned threshold as selected; it may differ across runs. `0.50` is a reference
point only. Test subjects are used only for
calibration and final evaluation, never for meta-training or generator training.

Results are written to a unique run directory under `results/film_personalization/`:
per-seed user metrics, per-user/per-k mean/std (`per_user_seed_summary.csv`), aggregate
mean/std, paired Wilcoxon comparisons, F1 and log-scale
false-alarm plots, meta-training history, run configuration/git revision, a generated
Markdown summary, and frozen/FiLM TFLite heads. Fine-tuning rows using only ADL calibration
are labeled separately from variants that additionally calibrate on two held-out fall
windows. Experimental TimeVAE code is future work only and is not part of the canonical
synthetic-user method; when explicitly explored, it is train-subject-only and enabled with
`--timevae-epochs N`. For a lighter run, lower `--epochs` and `--episodes-per-epoch` while retaining
at least five `--calibration-seeds`. In the current stored test split, two subjects have
ADL-only windows; their fall-sensitive per-user metrics are recorded as unavailable, not
as zero, and positive-fall calibration variants are omitted for them.

Presentation figures are written to `docs/presentation/figures/personalization/`.
The separate severity review plots are generated under `docs/presentation/figures/severity/`.

## Stored Results Snapshot

These persisted metrics are a snapshot, not the complete standardized comparison at the
locked headline threshold. CNN metrics at `0.25` still need to be evaluated in Phase 1.
The headline threshold for all models is `0.25`; `0.50` rows below are reference results
only. The CNN-LSTM weights currently on disk are recorded in
`results/cnn_lstm_baseline/run_metadata.json` (training commit `6e8b4a96`).

| Model | Threshold | Role | Fall precision | Fall recall | Fall F1 | Fall F2 | Recording recall | Impact-verified recall |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.50 | Reference | 0.9761 | 0.9867 | 0.9814 | 0.9845 | 0.9967 | 0.9967 |
| CNN-LSTM | 0.50 | Reference | 0.9770 | 0.9925 | 0.9847 | 0.9894 | 1.0000 | 1.0000 |
| CNN-LSTM | 0.25 | Headline | 0.9606 | 0.9950 | 0.9775 | 0.9879 | 1.0000 | 1.0000 |

The CNN-LSTM `0.25` threshold is the validation-selected maximum-F2 point in the stored
threshold sweep. A FiLM validation-retuned threshold is sensitivity analysis only and is
reported as selected for its run; the project's `0.50` results are reference columns,
not the headline comparison.

## Interactive Streaming Demo (Task P6)

A real-time edge streaming simulator and interactive dashboard running on the deployed quantized model (`results/quantization/run_v1/deployed_model.tflite`):

### 1. Web Dashboard (Streamlit)
```powershell
streamlit run app/demo.py
```
- **Features**: Real-time sliding ring buffer (64 samples @ 20 Hz, stride 16), rolling 6-channel IMU charts, probability timeline vs. validation-selected threshold ($\tau = 0.25$), visual status badge (`MONITORING` / `FALL DETECTED`), configurable $N$-consecutive positives (1–3), cooldown refractory period (10 s), caregiver notification dispatch log, and custom 6-channel CSV upload.
- **Walkthrough Guide**: See [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) for a 5-minute presentation script covering 6 test recordings (walking, sitting, stumble, jump, mild fall, severe fall).

### 2. Command-Line Replay
```powershell
python scripts/run_demo_cli.py --csv-path data/demo_recordings/SA02_F01_MildFall.csv --speed 5.0
```
Flags:
- `--recording-id`: Replay trial from test set (e.g. `SA02:F01_SA02_R02.txt`).
- `--csv-path`: Replay custom 6-channel IMU CSV file.
- `--speed`: Replay speed multiplier (e.g. `1.0` real-time, `5.0` 5x speed, `0` instantaneous).
- `--n-consecutive`: Required consecutive positive windows before alerting (`1`, `2`, or `3`).
- `--cooldown`: Alert suppression window in seconds (default `10.0s`).

## Limitations

1. **Laboratory Simulated Falls (SisFall)**: Falls were performed by young adult participants landing on safety mattresses under controlled laboratory conditions. Real-world geriatric falls typically feature distinct pre-fall gait instability, different impact dynamics, and post-fall immobility/long-lie periods that are not fully represented in lab data.
2. **Small Test Cohort ($n=6$)**: The subject-wise test split comprises 6 subjects, of whom only 4 performed falls and 2 performed ADL-only activities. Fall-sensitive metrics are unavailable for ADL-only subjects, and sample size is insufficient for formal statistical significance testing.
3. **Clustering-Derived Severity Ground Truth**: Severity categories (Mild vs. Severe) were derived via unsupervised $K=2$ clustering and Isolation Forest anomaly filtering on peak acceleration and angular velocity, rather than verified clinical injury outcomes.
4. **Extreme-Gyro Outliers**: Windows flagged as $-1$ by the Isolation Forest are extreme-gyro outliers and are masked from severity loss/metrics; they remain in the binary fall-detection evaluation.

## Tests

```bash
pytest
```

Run `pytest` to verify the test suite.

## Notes

- Large generated binary artifacts such as `*.npy`, `*.pkl`, and trained model checkpoints may be excluded from git depending on local ignore rules.
- Do not mix severity artifacts from different runs without regenerating them together.
- SisFall dataset usage is subject to the original dataset terms.

## License

Research and educational use.

