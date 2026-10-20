"""Reproducible meta-training and held-out evaluation orchestration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.keras_model_loading import load_keras_model
from src.personalization.evaluation import (
    WINDOW_STRIDE_SAMPLES,
    calculate_metrics,
    export_tflite_head,
    fine_tune_baseline,
    paired_wilcoxon,
    predict_model_probabilities,
    save_metric_plots,
    summarize_per_user_seeds,
    summarize_results,
    tune_finetune_baselines,
    write_results_manifest,
)
from src.personalization.film import (
    CALIBRATION_SIZES,
    CANONICAL_CLASS_WEIGHTS,
    choose_f2_threshold,
    film_probabilities,
    film_probabilities_from_features,
    save_user_encoder_weights,
    train_user_encoder,
)
from src.personalization.model import split_frozen_model
from src.personalization.simulator import (
    USER_PROFILES,
    UserSimulator,
    assert_subject_disjoint,
    split_user_calibration,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
BASELINE_DIR = ROOT / "results" / "cnn_lstm_baseline"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "film_personalization"
POSITIVE_FALL_CALIBRATION = 2
RANDOM_SEEDS = (17, 31, 47, 71, 101)


def _set_seeds(seed: int) -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)


def _load_arrays() -> dict[str, dict[str, np.ndarray]]:
    data: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        windows = np.load(DATA_DIR / f"{split}.npy").astype(np.float32, copy=False)
        labels = np.load(DATA_DIR / f"{split}_labels.npy").astype(np.int32, copy=False)
        subjects = np.load(DATA_DIR / f"{split}_subject_ids.npy", allow_pickle=True).astype(str)
        if windows.shape != (len(labels), 64, 6) or len(labels) != len(subjects):
            raise ValueError(f"{split} arrays must align and have (N,64,6) windows.")
        if not np.isfinite(windows).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{split} arrays contain invalid values.")
        data[split] = {"x": windows, "y": labels, "subject": subjects}
    assert_subject_disjoint(
        data["train"]["subject"], data["val"]["subject"], data["test"]["subject"]
    )
    expected_subject_counts = {"train": 26, "val": 6, "test": 6}
    observed_subject_counts = {
        split: len(np.unique(values["subject"])) for split, values in data.items()
    }
    if observed_subject_counts != expected_subject_counts:
        raise ValueError(
            f"Unexpected subject-wise split {observed_subject_counts}; "
            f"expected {expected_subject_counts}."
        )
    return data


def _threshold_from_metadata() -> float:
    """Read the canonical 0.25 threshold; a run's validation-retuned value is sensitivity-only."""
    metadata = json.loads((BASELINE_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    threshold = metadata.get("chosen_threshold", {}).get("threshold")
    if threshold is None:
        raise ValueError("Canonical baseline metadata does not contain chosen_threshold.threshold.")
    return float(threshold)


def _load_backbone(train_windows: np.ndarray) -> Any:
    import tensorflow as tf

    from scripts.train_cnn_lstm_baseline import build_model

    model_path = BASELINE_DIR / "cnn_lstm_best.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Canonical CNN-LSTM weights not found: {model_path}")
    model = load_keras_model(tf, model_path, build_model)
    return split_frozen_model(model, train_windows)


def _tune_validation_threshold(
    components: Any,
    encoder: Any,
    val_data: dict[str, np.ndarray],
    simulator: UserSimulator,
    seed: int,
) -> float:
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for subject in np.unique(val_data["subject"]):
        indices = np.flatnonzero(val_data["subject"] == subject)
        x, y = val_data["x"][indices], val_data["y"][indices]
        if not np.any(y == 0) or not np.any(y == 1):
            continue
        calibration, _, query, query_labels = split_user_calibration(
            x, y, max_calibration=5, seed=seed
        )
        calibration = simulator.transform(
            calibration, "Normal", user_id=f"val:{subject}", seed=seed
        )
        query = simulator.transform(query, "Normal", user_id=f"val:{subject}", seed=seed + 1)
        prediction, _, _ = film_probabilities(components, encoder, query, calibration)
        probabilities.append(prediction)
        labels.append(query_labels)
    if not probabilities:
        raise ValueError("Validation subjects do not contain both classes for threshold selection.")
    return choose_f2_threshold(np.concatenate(labels), np.concatenate(probabilities))


def _fit_timevae_if_requested(
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    output_dir: Path,
    *,
    epochs: int,
    seed: int,
) -> dict[str, Any] | None:
    if epochs <= 0:
        return None
    from src.personalization.generator import (
        ConditionalTimeVAE,
        evaluate_generator_quality,
        validate_generator_training_split,
    )

    validate_generator_training_split(
        train_data["subject"],
        train_data["subject"],
        np.concatenate((val_data["subject"], test_data["subject"])),
    )
    generator = ConditionalTimeVAE(seed=seed)
    history = generator.fit(
        train_data["x"],
        train_data["y"],
        train_data["subject"],
        held_out_subject_ids=np.concatenate((val_data["subject"], test_data["subject"])),
        epochs=epochs,
    )
    generator.save_weights(output_dir / "timevae")
    pd.DataFrame(history).to_csv(output_dir / "timevae_training_history.csv", index=False)
    labels = np.concatenate((np.zeros(1000, dtype=np.int32), np.ones(1000, dtype=np.int32)))
    generator_simulator = UserSimulator(
        backend="generator",
        seed=seed,
        generator=lambda classes, rng: generator.generate(
            classes, seed=int(rng.integers(0, np.iinfo(np.int32).max))
        ),
    )
    synthetic = generator_simulator.generate(labels, seed=seed)
    quality = evaluate_generator_quality(
        train_data["x"],
        synthetic,
        labels,
        test_data["x"],
        test_data["y"],
        output_dir,
        seed=seed,
    )
    return quality


def _append_metric(
    rows: list[dict[str, Any]],
    *,
    evaluation: str,
    profile: str,
    subject_id: str,
    model_name: str,
    k: int,
    seed: int,
    threshold: float,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    rows.append(
        {
            "evaluation": evaluation,
            "profile": profile,
            "subject_id": subject_id,
            "model": model_name,
            "k": int(k),
            "seed": int(seed),
            "threshold": float(threshold),
            **calculate_metrics(labels, probabilities, threshold),
        }
    )


def evaluate_users(
    components: Any,
    encoder: Any,
    test_data: dict[str, np.ndarray],
    simulator: UserSimulator,
    output_dir: Path,
    *,
    canonical_threshold: float,
    retuned_threshold: float,
    seeds: tuple[int, ...] = RANDOM_SEEDS,
    finetune_epochs: int = 3,
    fine_tune_configs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compare global, fine-tuned, and FiLM systems on real and shifted users."""

    rows: list[dict[str, Any]] = []
    probability_shift_rows: list[dict[str, Any]] = []
    subjects = sorted(np.unique(test_data["subject"]))
    subjects_with_falls = [
        subject
        for subject in subjects
        if np.any(test_data["y"][test_data["subject"] == subject] == 1)
    ]
    if not subjects_with_falls:
        raise ValueError("The held-out split has no subject with fall windows to evaluate.")
    profiles = [("real_heldout", subject, "Real") for subject in subjects]
    profiles.extend(
        ("simulated_users", subjects_with_falls[index % len(subjects_with_falls)], profile)
        for index, profile in enumerate(USER_PROFILES)
    )

    for evaluation, subject, profile in profiles:
        subject_rows = np.flatnonzero(test_data["subject"] == subject)
        subject_windows = test_data["x"][subject_rows]
        subject_labels = test_data["y"][subject_rows]
        for seed in seeds:
            calibration_pool, _, query_raw, query_labels = split_user_calibration(
                subject_windows,
                subject_labels,
                max_calibration=max(CALIBRATION_SIZES),
                seed=seed,
                allow_adl_only=evaluation == "real_heldout",
            )
            transform_seed = seed + 1009
            calibration_pool = simulator.transform(
                calibration_pool,
                profile if profile != "Real" else "Normal",
                user_id=f"{evaluation}:{subject}",
                seed=transform_seed,
            )
            query = simulator.transform(
                query_raw,
                profile if profile != "Real" else "Normal",
                user_id=f"{evaluation}:{subject}",
                seed=transform_seed + 1,
            )

            query_fall_indices = np.flatnonzero(query_labels == 1)
            has_positive_calibration = len(query_fall_indices) >= POSITIVE_FALL_CALIBRATION
            if has_positive_calibration:
                fall_rng = np.random.default_rng(seed + 701)
                fall_indices = fall_rng.choice(
                    query_fall_indices, size=POSITIVE_FALL_CALIBRATION, replace=False
                )
                positive_query_mask = np.ones(len(query), dtype=bool)
                positive_query_mask[fall_indices] = False
                positive_query = query[positive_query_mask]
                positive_query_labels = query_labels[positive_query_mask]
                positive_calibration = query[fall_indices]
                positive_calibration_labels = query_labels[fall_indices]
            else:
                positive_query = query
                positive_query_labels = query_labels
                positive_calibration = np.empty((0, 64, 6), dtype=np.float32)
                positive_calibration_labels = np.empty((0,), dtype=np.int32)
            query_features = components.extractor(query, training=False)
            global_probabilities = components.head(
                query_features, training=False
            ).numpy().reshape(-1)

            for k in CALIBRATION_SIZES:
                calibration = calibration_pool[:k]
                calibration_labels = np.zeros(k, dtype=np.int32)
                _append_metric(
                    rows,
                    evaluation=evaluation,
                    profile=profile,
                    subject_id=subject,
                    model_name="Global",
                    k=k,
                    seed=seed,
                    threshold=canonical_threshold,
                    labels=query_labels,
                    probabilities=global_probabilities,
                )

                fine_tune_variants = [
                    (
                        "Last-layer fine-tune (ADL-only)",
                        False,
                        False,
                        calibration_labels,
                        query,
                        query_labels,
                    ),
                    (
                        "Full fine-tune (ADL-only)",
                        True,
                        False,
                        calibration_labels,
                        query,
                        query_labels,
                    ),
                ]
                if has_positive_calibration:
                    fine_tune_variants.extend(
                        [
                            (
                                f"Last-layer fine-tune (+{POSITIVE_FALL_CALIBRATION} falls)",
                                False,
                                True,
                                np.concatenate((calibration_labels, positive_calibration_labels)),
                                positive_query,
                                positive_query_labels,
                            ),
                            (
                                f"Full fine-tune (+{POSITIVE_FALL_CALIBRATION} falls)",
                                True,
                                True,
                                np.concatenate((calibration_labels, positive_calibration_labels)),
                                positive_query,
                                positive_query_labels,
                            ),
                        ]
                    )
                for model_name, is_full, uses_falls, fit_labels, eval_x, eval_y in fine_tune_variants:
                    fit_x = (
                        np.concatenate((calibration, positive_calibration), axis=0)
                        if uses_falls
                        else calibration
                    )
                    method_key = "full_model" if is_full else "last_layer"
                    selected_config = (
                        fine_tune_configs["methods"][method_key]["selected_config"]
                        if fine_tune_configs and not uses_falls
                        else {"epochs": finetune_epochs, "learning_rate": 1e-4}
                    )
                    adapted = fine_tune_baseline(
                        components.original,
                        fit_x,
                        fit_labels,
                        full_model=is_full,
                        epochs=int(selected_config["epochs"]),
                        learning_rate=float(selected_config["learning_rate"]),
                    )
                    probabilities = predict_model_probabilities(adapted, eval_x)
                    if k == 1 and not uses_falls:
                        for label, label_name in ((0, "ADL"), (1, "Fall")):
                            group = eval_y == label
                            if np.any(group):
                                differences = probabilities[group] - global_probabilities[group]
                                probability_shift_rows.append(
                                    {
                                        "evaluation": evaluation,
                                        "profile": profile,
                                        "subject_id": subject,
                                        "seed": seed,
                                        "method": method_key,
                                        "label_group": label_name,
                                        "windows": int(group.sum()),
                                        "signed_mean_probability_shift": float(
                                            np.mean(differences)
                                        ),
                                        "absolute_mean_probability_shift": float(
                                            np.mean(np.abs(differences))
                                        ),
                                        "threshold_crossings_vs_global": int(
                                            np.count_nonzero(
                                                (probabilities[group] >= canonical_threshold)
                                                != (
                                                    global_probabilities[group]
                                                    >= canonical_threshold
                                                )
                                            )
                                        ),
                                    }
                                )
                    _append_metric(
                        rows,
                        evaluation=evaluation,
                        profile=profile,
                        subject_id=subject,
                        model_name=model_name,
                        k=k,
                        seed=seed,
                        threshold=canonical_threshold,
                        labels=eval_y,
                        probabilities=probabilities,
                    )

                if k:
                    calibration_features = components.extractor(calibration, training=False)
                else:
                    import tensorflow as tf

                    calibration_features = tf.zeros(
                        (0, *components.feature_shape), dtype=tf.float32
                    )
                probabilities, _, _ = film_probabilities_from_features(
                    components, encoder, query_features, calibration_features
                )
                for model_name, threshold in (
                    ("FiLM (canonical threshold)", canonical_threshold),
                    ("FiLM (validation-retuned threshold)", retuned_threshold),
                ):
                    _append_metric(
                        rows,
                        evaluation=evaluation,
                        profile=profile,
                        subject_id=subject,
                        model_name=model_name,
                        k=k,
                        seed=seed,
                        threshold=threshold,
                        labels=query_labels,
                        probabilities=probabilities,
                    )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "per_user_metrics.csv", index=False)
    shifts = pd.DataFrame(probability_shift_rows)
    shifts.to_csv(output_dir / "fine_tune_probability_shifts.csv", index=False)
    if not shifts.empty:
        shift_summary = (
            shifts.groupby(["evaluation", "method", "label_group"], as_index=False)
            .agg(
                windows=("windows", "sum"),
                signed_mean_probability_shift=("signed_mean_probability_shift", "mean"),
                absolute_mean_probability_shift=(
                    "absolute_mean_probability_shift",
                    "mean",
                ),
                threshold_crossings_vs_global=("threshold_crossings_vs_global", "sum"),
                subject_seed_rows=("subject_id", "size"),
            )
        )
        shift_summary.to_csv(
            output_dir / "fine_tune_probability_shift_summary.csv", index=False
        )
    return metrics


def write_summary_from_csvs(output_dir: Path, command: str) -> None:
    """Build the human-readable report exclusively from persisted result CSVs."""

    summary = pd.read_csv(output_dir / "metrics_summary.csv")
    tests = pd.read_csv(output_dir / "paired_wilcoxon.csv")
    config = pd.read_csv(output_dir / "experiment_config.csv").set_index("key")["value"].to_dict()
    deployment = pd.read_csv(output_dir / "deployment_size.csv").iloc[0]
    metrics = pd.read_csv(output_dir / "per_user_metrics.csv")
    test_subject_count = metrics.loc[
        metrics["evaluation"] == "real_heldout", "subject_id"
    ].nunique()
    real_metrics = metrics[metrics["evaluation"] == "real_heldout"]
    subject_fall_support = real_metrics.groupby("subject_id")["fall_windows"].max()
    adl_only_subject_count = int((subject_fall_support == 0).sum())
    canonical_threshold = float(
        metrics.loc[metrics["model"] == "Global", "threshold"].drop_duplicates().iloc[0]
    )
    retuned_threshold = float(
        metrics.loc[
            metrics["model"] == "FiLM (validation-retuned threshold)", "threshold"
        ].drop_duplicates().iloc[0]
    )
    selected = summary[
        (summary["evaluation"] == "real_heldout")
        & (summary["profile"] == "Real")
        & summary["k"].isin(CALIBRATION_SIZES)
    ]
    lines = [
        "# FiLM personalization experiment",
        "",
        "This report is generated from `metrics_summary.csv` and `paired_wilcoxon.csv`.",
        "No expected-performance curve is prefilled. Read the saved per-user CSV for seed-level results.",
        "",
        "## Primary results: real held-out SisFall users",
        "",
        "| Model | k | F1 mean +/- std | F2 mean +/- std | False alarms/hour mean +/- std |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.model} | {int(row.k)} | {row.f1_mean:.4f} +/- {row.f1_std:.4f} "
            f"| {row.f2_mean:.4f} +/- {row.f2_std:.4f} "
            f"| {row.false_alarms_per_hour_mean:.4f} +/- {row.false_alarms_per_hour_std:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired Wilcoxon comparisons",
            "",
            "| Evaluation | Comparison | Subjects | Statistic | p-value |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in tests.itertuples(index=False):
        lines.append(
            f"| {row.evaluation} | {row.comparison} | {int(row.n_subjects)} "
            f"| {row.wilcoxon_statistic:.4g} | {row.p_value:.4g} |"
        )
    tuning_path = output_dir / "fine_tune_tuning.json"
    if tuning_path.is_file():
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                "## Validation tuning and probability shifts",
                "",
                f"Fine-tuning configurations were selected on validation F2 at "
                f"threshold {float(tuning['validation_threshold']):.2f}; "
                f"each method tried {int(tuning['configurations_per_method'])} configs. "
                "Test probability shifts are reporting-only.",
                "",
                "| Comparison method | Selection/tuning data | Configurations tried |",
                "|---|---|---:|",
                "| Global | Frozen canonical baseline; shared threshold 0.25 | 0 new configs |",
                "| Last-layer FT | Validation F2 at 0.25; one ADL calibration window/user | "
                f"{int(tuning['configurations_per_method'])} |",
                "| Full FT | Validation F2 at 0.25; one ADL calibration window/user | "
                f"{int(tuning['configurations_per_method'])} |",
                "| FiLM | Train-subject episodes; validation early stopping/threshold | "
                "1 run configuration |",
                "",
                "| Method | Selected learning rate | Selected epochs | Validation F2 | Validation FA/hr |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method_key, label in (
            ("last_layer", "Last-layer FT"),
            ("full_model", "Full FT"),
        ):
            method = tuning["methods"][method_key]
            config_selected = method["selected_config"]
            metrics_selected = method["selection_metrics"]
            lines.append(
                f"| {label} | {config_selected['learning_rate']:.0e} "
                f"| {config_selected['epochs']} | {metrics_selected['validation_mean_f2']:.4f} "
                f"| {metrics_selected['validation_mean_false_alarms_per_hour']:.4f} |"
            )
        shift_path = output_dir / "fine_tune_probability_shift_summary.csv"
        if shift_path.is_file():
            shifts = pd.read_csv(shift_path)
            lines.extend(
                [
                    "",
                    "| Method | Test subset | Signed mean shift | Absolute mean shift | Threshold crossings |",
                    "|---|---|---:|---:|---:|",
                ]
            )
            for row in shifts[shifts["evaluation"] == "real_heldout"].itertuples(
                index=False
            ):
                lines.append(
                    f"| {row.method} | {row.label_group} | "
                    f"{row.signed_mean_probability_shift:.6f} | "
                    f"{row.absolute_mean_probability_shift:.6f} | "
                    f"{int(row.threshold_crossings_vs_global)} |"
                )
    lines.extend(
        [
            "",
            "## Model and deployment size",
            "",
            f"- Frozen backbone parameters: {int(float(config['frozen_backbone_parameters']))}.",
            f"- Shared UserEncoder parameters: {int(float(config['user_encoder_parameters']))}.",
            f"- Per-user state: {int(float(config['per_user_affine_floats']))} float32 values "
            f"({int(float(config['per_user_affine_bytes']))} bytes).",
            f"- TFLite head sizes: {int(deployment['base_tflite_bytes'])} bytes base; "
            f"{int(deployment['film_tflite_bytes'])} bytes FiLM; "
            f"overhead {int(deployment['tflite_overhead_bytes'])} bytes.",
            f"- TFLite max absolute output errors: base {deployment['base_tflite_max_abs_error']:.3g}; "
            f"FiLM {deployment['film_tflite_max_abs_error']:.3g}.",
            "",
            "## Evaluation details",
            "",
            f"The real held-out evaluation contains {test_subject_count} subjects. False alarms/hour "
            f"uses a stride of {int(float(config['window_stride_samples']))} samples at "
            f"{float(config['sample_rate_hz']):g} Hz "
            f"({float(config['window_stride_seconds']):g} seconds per row).",
            f"The positive-calibration baselines add "
            f"{int(float(config['positive_fall_calibration_windows']))} fall windows and are labeled as such.",
            f"Global baselines and FiLM canonical-threshold rows use {canonical_threshold:.3f}; "
            f"the FiLM validation-retuned threshold is {retuned_threshold:.3f}.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            command,
            "```",
            "",
            "Calibration uses ADL windows only for the primary FiLM and fine-tuning baselines. "
            "Positive-calibration fine-tuning rows are explicitly labeled.",
            "",
            "## Limitations",
            "",
            "- Only the existing held-out subject split is used. "
            f"It has {test_subject_count - adl_only_subject_count} subjects with fall windows and "
            f"{adl_only_subject_count} ADL-only subjects; fall-sensitive metrics are N/A for those ADL-only users.",
            "- User profiles are parametric sensor shifts, not evidence of realistic new-user physiology.",
            "- The optional TimeVAE path is future work only; the canonical synthetic users use parametric shifts.",
            "- The last-layer and full fine-tuning baselines with ADL-only calibration have no positive examples.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    *,
    output_dir: Path | None = None,
    seed: int = 42,
    episodes_per_epoch: int = 30,
    epochs: int = 30,
    validation_episodes: int = 12,
    patience: int = 5,
    seeds: tuple[int, ...] = RANDOM_SEEDS,
    finetune_epochs: int = 3,
    timevae_epochs: int = 0,
    command: str | None = None,
) -> Path:
    if len(seeds) < 5:
        raise ValueError("At least five calibration-sampling seeds are required.")
    _set_seeds(seed)
    output_dir = output_dir or (
        DEFAULT_OUTPUT_DIR
        / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    data = _load_arrays()
    components = _load_backbone(data["train"]["x"])
    simulator = UserSimulator(seed=seed)
    generator_quality = _fit_timevae_if_requested(
        data["train"],
        data["val"],
        data["test"],
        output_dir,
        epochs=timevae_epochs,
        seed=seed,
    )
    encoder, history = train_user_encoder(
        components,
        data["train"]["x"],
        data["train"]["y"],
        data["train"]["subject"],
        data["val"]["x"],
        data["val"]["y"],
        data["val"]["subject"],
        simulator,
        seed=seed,
        epochs=epochs,
        episodes_per_epoch=episodes_per_epoch,
        validation_episodes=validation_episodes,
        patience=patience,
    )
    pd.DataFrame(history).to_csv(output_dir / "meta_training_history.csv", index=False)
    save_user_encoder_weights(encoder, output_dir / "user_encoder.weights.h5")
    canonical_threshold = _threshold_from_metadata()
    retuned_threshold = _tune_validation_threshold(
        components, encoder, data["val"], simulator, seed + 211
    )
    fine_tune_tuning = tune_finetune_baselines(
        components.original,
        data["val"],
        threshold=canonical_threshold,
        calibration_seed=seed + 1701,
    )
    (output_dir / "fine_tune_tuning.json").write_text(
        json.dumps(fine_tune_tuning, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "thresholds.json").write_text(
        json.dumps(
            {
                "canonical_threshold_from_existing_sweep": canonical_threshold,
                "film_threshold_tuned_on_validation_only": retuned_threshold,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = evaluate_users(
        components,
        encoder,
        data["test"],
        simulator,
        output_dir,
        canonical_threshold=canonical_threshold,
        retuned_threshold=retuned_threshold,
        seeds=seeds,
        finetune_epochs=finetune_epochs,
        fine_tune_configs=fine_tune_tuning,
    )
    summarize_per_user_seeds(metrics, output_dir)
    summary = summarize_results(metrics, output_dir)
    tests = paired_wilcoxon(metrics, output_dir)
    save_metric_plots(metrics, output_dir)

    validation_adl = data["val"]["x"][data["val"]["y"] == 0][:5]
    _, gamma, beta = film_probabilities(
        components,
        encoder,
        data["val"]["x"][:1],
        validation_adl,
    )
    np.savez(
        output_dir / "validation_user_film_constants.npz",
        gamma=gamma.astype(np.float32),
        beta=beta.astype(np.float32),
    )
    deployment = export_tflite_head(components, gamma, beta, output_dir)
    pd.DataFrame([deployment]).to_csv(output_dir / "deployment_size.csv", index=False)
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    run_command = command or " ".join([sys.executable, *sys.argv])
    manifest = {
        "git_commit": git_revision,
        "seed": seed,
        "calibration_seeds": list(seeds),
        "calibration_sizes": list(CALIBRATION_SIZES),
        "episodes_per_epoch": episodes_per_epoch,
        "max_epochs": epochs,
        "validation_episodes": validation_episodes,
        "early_stopping_patience": patience,
        "fine_tune_epochs": finetune_epochs,
        "timevae_epochs": timevae_epochs,
        "parameter_count": int(encoder.count_params()),
        "per_user_affine_floats": int(2 * components.feature_shape[-1]),
        "per_user_affine_bytes": int(2 * components.feature_shape[-1] * np.dtype(np.float32).itemsize),
        "frozen_backbone_parameters": int(components.original.count_params()),
        "user_profiles": {name: asdict(profile) for name, profile in USER_PROFILES.items()},
        "simulator_backend": "parametric",
        "meta_learning_rate": 1e-3,
        "query_adl_windows_per_episode": 32,
        "query_fall_sampling_ratio": 4484 / 50939,
        "positive_fall_calibration_windows": POSITIVE_FALL_CALIBRATION,
        "class_weights": {str(key): value for key, value in CANONICAL_CLASS_WEIGHTS.items()},
        "window_stride_samples": WINDOW_STRIDE_SAMPLES,
        "window_stride_seconds_at_20hz": WINDOW_STRIDE_SAMPLES / 20.0,
        "canonical_threshold": canonical_threshold,
        "film_validation_threshold": retuned_threshold,
        "fine_tune_tuning": fine_tune_tuning,
        "generator_quality": generator_quality,
        "deployment": deployment,
        "summary_rows": int(len(summary)),
        "wilcoxon_comparisons": int(len(tests)),
        "command": run_command,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_results_manifest(output_dir, manifest)
    config_rows = [
        ("seed", seed),
        ("calibration_seeds", json.dumps(list(seeds))),
        ("calibration_sizes", json.dumps(list(CALIBRATION_SIZES))),
        ("epochs", epochs),
        ("episodes_per_epoch", episodes_per_epoch),
        ("validation_episodes", validation_episodes),
        ("patience", patience),
        ("fine_tune_epochs", finetune_epochs),
        ("fine_tune_tuning_json", json.dumps(fine_tune_tuning, sort_keys=True)),
        ("timevae_epochs", timevae_epochs),
        ("sample_rate_hz", 20.0),
        ("window_stride_samples", WINDOW_STRIDE_SAMPLES),
        ("window_stride_seconds", WINDOW_STRIDE_SAMPLES / 20.0),
        ("positive_fall_calibration_windows", POSITIVE_FALL_CALIBRATION),
        ("frozen_backbone_parameters", components.original.count_params()),
        ("user_encoder_parameters", encoder.count_params()),
        ("per_user_affine_floats", 2 * components.feature_shape[-1]),
        ("per_user_affine_bytes", 2 * components.feature_shape[-1] * np.dtype(np.float32).itemsize),
        ("user_profiles_json", json.dumps(manifest["user_profiles"], sort_keys=True)),
        ("simulator_backend", "parametric"),
        ("query_adl_windows_per_episode", 32),
        ("query_fall_sampling_ratio", 4484 / 50939),
        ("class_weights_json", json.dumps(
            {str(key): value for key, value in CANONICAL_CLASS_WEIGHTS.items()},
            sort_keys=True,
        )),
    ]
    pd.DataFrame(config_rows, columns=["key", "value"]).to_csv(
        output_dir / "experiment_config.csv", index=False
    )
    write_summary_from_csvs(output_dir, run_command)
    return output_dir
