"""Personalization baselines, metrics, statistics, plots, and TFLite export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score

from src.personalization.film import CANONICAL_CLASS_WEIGHTS


SAMPLE_RATE_HZ = 20.0
WINDOW_STRIDE_SAMPLES = 16
WINDOW_SECONDS_PER_ROW = WINDOW_STRIDE_SAMPLES / SAMPLE_RATE_HZ
FALSE_ALARM_WINDOW_RATE_PER_HOUR = 3600.0 / WINDOW_SECONDS_PER_ROW


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    truth = np.asarray(labels, dtype=np.int32)
    predicted = np.asarray(probabilities) >= threshold
    false_positives = int(np.count_nonzero((truth == 0) & predicted))
    adl_count = int(np.count_nonzero(truth == 0))
    fall_count = int(np.count_nonzero(truth == 1))
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": (
            float(precision_score(truth, predicted, zero_division=0))
            if fall_count
            else float("nan")
        ),
        "recall": (
            float(recall_score(truth, predicted, zero_division=0))
            if fall_count
            else float("nan")
        ),
        "f1": float(f1_score(truth, predicted, zero_division=0)) if fall_count else float("nan"),
        "f2": (
            float(fbeta_score(truth, predicted, beta=2, zero_division=0))
            if fall_count
            else float("nan")
        ),
        "false_alarms_per_hour": (
            false_positives / adl_count * FALSE_ALARM_WINDOW_RATE_PER_HOUR
            if adl_count
            else float("nan")
        ),
        "false_positive_windows": float(false_positives),
        "adl_windows": float(adl_count),
        "fall_windows": float(fall_count),
    }


def fine_tune_baseline(
    original_model: Any,
    calibration_windows: np.ndarray,
    calibration_labels: np.ndarray,
    *,
    full_model: bool,
    epochs: int = 3,
    learning_rate: float = 1e-4,
) -> Any:
    """Clone the baseline and fine-tune only output Dense or all layers."""

    import tensorflow as tf

    if len(calibration_windows) == 0:
        return original_model
    model = tf.keras.models.clone_model(original_model)
    model.set_weights(original_model.get_weights())
    model.trainable = True
    for layer in model.layers:
        layer.trainable = full_model or layer.name == "dense_1"
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    trainable_variables = model.trainable_variables
    if not trainable_variables:
        raise RuntimeError("Fine-tuning configuration produced no trainable model variables.")
    optimizer.build(trainable_variables)
    windows = tf.convert_to_tensor(calibration_windows, dtype=tf.float32)
    labels = tf.convert_to_tensor(calibration_labels.reshape(-1, 1), dtype=tf.float32)
    for _ in range(epochs):
        with tf.GradientTape() as tape:
            probabilities = model(windows, training=True)
            per_example = tf.keras.losses.binary_crossentropy(labels, probabilities)
            weights = tf.where(
                labels[:, 0] > 0.5,
                CANONICAL_CLASS_WEIGHTS[1],
                CANONICAL_CLASS_WEIGHTS[0],
            )
            loss = tf.reduce_mean(per_example * weights)
        gradients = tape.gradient(loss, trainable_variables)
        gradient_pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, trainable_variables)
            if gradient is not None
        ]
        if not gradient_pairs:
            raise RuntimeError("Fine-tuning produced no gradients.")
        optimizer.apply_gradients(gradient_pairs)
    return model


def predict_model_probabilities(model: Any, windows: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    import tensorflow as tf

    values = np.asarray(windows, dtype=np.float32)
    batches = [
        model(tf.convert_to_tensor(values[start : start + batch_size]), training=False).numpy().reshape(-1)
        for start in range(0, len(values), batch_size)
    ]
    return np.concatenate(batches) if batches else np.empty((0,), dtype=np.float32)


def summarize_results(rows: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    metrics = ["accuracy", "precision", "recall", "f1", "f2", "false_alarms_per_hour"]
    group_columns = ["evaluation", "profile", "model", "k", "threshold"]
    summary = (
        rows.groupby(group_columns, dropna=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(item) for item in column if item != "")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    return summary


def summarize_per_user_seeds(rows: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    metrics = ["accuracy", "precision", "recall", "f1", "f2", "false_alarms_per_hour"]
    group_columns = [
        "evaluation",
        "profile",
        "subject_id",
        "model",
        "k",
        "threshold",
    ]
    summary = (
        rows.groupby(group_columns, dropna=False)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(item) for item in column if item != "")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(output_dir / "per_user_seed_summary.csv", index=False)
    return summary


def paired_wilcoxon(rows: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    paired = rows.groupby(["evaluation", "subject_id", "model"], as_index=False)["f1"].mean()
    tests = []
    for evaluation, group in paired.groupby("evaluation"):
        by_model = group.pivot(index="subject_id", columns="model", values="f1")
        film_model = "FiLM (canonical threshold)"
        if film_model not in by_model:
            continue
        baselines = [
            name
            for name in by_model.columns
            if not str(name).startswith("FiLM")
        ]
        for baseline in baselines:
            shared = by_model[[film_model, baseline]].dropna()
            if len(shared) < 2:
                statistic, p_value = float("nan"), float("nan")
            elif np.allclose(shared[film_model], shared[baseline]):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(
                    shared[film_model],
                    shared[baseline],
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                statistic, p_value = float(result.statistic), float(result.pvalue)
            tests.append(
                {
                    "evaluation": evaluation,
                    "metric": "subject-mean F1 across k and seeds",
                    "comparison": f"FiLM vs {baseline}",
                    "n_subjects": int(len(shared)),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
    result_table = pd.DataFrame(tests)
    result_table.to_csv(output_dir / "paired_wilcoxon.csv", index=False)
    return result_table


def save_metric_plots(rows: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    means = (
        rows.groupby(["evaluation", "profile", "subject_id", "model", "k"], as_index=False)
        [["f1", "false_alarms_per_hour"]]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    for (evaluation, profile, subject, model), group in means.groupby(
        ["evaluation", "profile", "subject_id", "model"]
    ):
        ax.plot(
            group["k"],
            group["f1"],
            marker="o",
            alpha=0.55,
            label=f"{evaluation}:{profile}/{subject} - {model}",
        )
    ax.set_xlabel("ADL calibration windows (k)")
    ax.set_ylabel("F1")
    ax.set_xticks([0, 1, 5, 10, 30])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "f1_vs_k_per_user.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    positives = means[means["false_alarms_per_hour"] > 0]
    for (evaluation, profile, subject, model), group in positives.groupby(
        ["evaluation", "profile", "subject_id", "model"]
    ):
        ax.plot(
            group["k"],
            group["false_alarms_per_hour"],
            marker="o",
            alpha=0.55,
            label=f"{evaluation}:{profile}/{subject} - {model}",
        )
    ax.set_xlabel("ADL calibration windows (k)")
    ax.set_ylabel("False alarms per hour of ADL")
    ax.set_xticks([0, 1, 5, 10, 30])
    ax.set_yscale("log")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "false_alarms_per_hour_vs_k.png", dpi=160)
    plt.close(fig)


def export_tflite_head(
    components: Any,
    gamma: np.ndarray,
    beta: np.ndarray,
    output_dir: Path,
) -> dict[str, int | float]:
    """Export the frozen head and a FiLM head with user affine values baked in."""

    import tensorflow as tf

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_input = tf.keras.Input(shape=components.feature_shape, name="cnn_features")
    base_model = tf.keras.Model(feature_input, components.head(feature_input, training=False))
    film_input = tf.keras.Input(shape=components.feature_shape, name="cnn_features")
    gamma_tensor = tf.constant(np.asarray(gamma, dtype=np.float32).reshape(1, 1, -1))
    beta_tensor = tf.constant(np.asarray(beta, dtype=np.float32).reshape(1, 1, -1))
    modulated = tf.keras.layers.Lambda(
        lambda values: values * gamma_tensor + beta_tensor,
        name="precomputed_user_film",
    )(film_input)
    film_model = tf.keras.Model(
        film_input,
        components.head(modulated, training=False),
        name="frozen_film_lstm_head",
    )
    base_path = output_dir / "frozen_lstm_head.tflite"
    film_path = output_dir / "personalized_film_head.tflite"
    base_path.write_bytes(tf.lite.TFLiteConverter.from_keras_model(base_model).convert())
    film_path.write_bytes(tf.lite.TFLiteConverter.from_keras_model(film_model).convert())
    probe = np.random.default_rng(42).normal(size=(1, *components.feature_shape)).astype(np.float32)

    def validate_tflite(path: Path, expected_model: Any) -> float:
        interpreter = tf.lite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        interpreter.set_tensor(input_detail["index"], probe)
        interpreter.invoke()
        actual = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
        expected = expected_model(probe, training=False).numpy()
        maximum_error = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, atol=1e-5, rtol=1e-5):
            raise AssertionError(
                f"TFLite output for {path.name} differs from Keras by {maximum_error:.8g}."
            )
        return maximum_error

    base_error = validate_tflite(base_path, base_model)
    film_error = validate_tflite(film_path, film_model)
    return {
        "base_tflite_bytes": base_path.stat().st_size,
        "film_tflite_bytes": film_path.stat().st_size,
        "tflite_overhead_bytes": film_path.stat().st_size - base_path.stat().st_size,
        "user_affine_bytes": int(2 * len(gamma) * np.dtype(np.float32).itemsize),
        "channels": int(len(gamma)),
        "base_tflite_max_abs_error": base_error,
        "film_tflite_max_abs_error": film_error,
    }


def write_results_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    (output_dir / "run_metadata.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
