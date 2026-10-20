"""Validation-calibrated, read-only analysis of the three completed FiLM runs."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, fbeta_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[4]
RUNS_DIR = ROOT / "results" / "film_personalization" / "full_run_v1"
OUTPUT_DIR = RUNS_DIR / "analysis"
SEEDS = (42, 43, 44)
CALIBRATION_SEEDS = (17, 31, 47, 71, 101)
CALIBRATION_SIZES = (0, 1, 5, 10, 30)
MODELS = ("Global", "Last-layer FT (ADL-only)", "Full FT (ADL-only)", "FiLM")
THRESHOLD_GRID = np.round(np.arange(0.1, 0.901, 0.05), 2)
REFERENCE_THRESHOLD = 0.25
WINDOW_SECONDS = 16 / 20

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.personalization.evaluation import (  # noqa: E402
    fine_tune_baseline,
    predict_model_probabilities,
)
from src.personalization.experiment import _load_arrays, _load_backbone  # noqa: E402
from src.personalization.film import UserEncoder, film_probabilities  # noqa: E402
from src.personalization.simulator import UserSimulator, split_user_calibration  # noqa: E402


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    prediction = probabilities >= threshold
    adl = labels == 0
    falls = labels == 1
    false_alarms_per_hour = (
        float(np.count_nonzero(adl & prediction) / np.count_nonzero(adl) / WINDOW_SECONDS * 3600)
        if np.any(adl)
        else float("nan")
    )
    auroc = (
        float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) == 2
        else float("nan")
    )
    auprc = (
        float(average_precision_score(labels, probabilities))
        if np.any(falls)
        else float("nan")
    )
    return {
        "f1": float(f1_score(labels, prediction, zero_division=0)) if np.any(falls) else float("nan"),
        "f2": float(fbeta_score(labels, prediction, beta=2, zero_division=0))
        if np.any(falls)
        else float("nan"),
        "false_alarms_per_hour": false_alarms_per_hour,
        "auroc": auroc,
        "auprc": auprc,
    }


def _choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = []
    for threshold in THRESHOLD_GRID:
        metrics = _metrics(labels, probabilities, float(threshold))
        candidates.append((float(threshold), metrics))
    # If F2 and FA/hr also tie, prefer the higher threshold as a deterministic
    # conservative-alert tie-break beyond the locked D2 criteria.
    threshold, metrics = min(
        candidates,
        key=lambda row: (
            -row[1]["f2"],
            row[1]["false_alarms_per_hour"],
            -row[0],
        ),
    )
    return threshold, metrics


def _load_encoder(weights_path: Path, channels: int) -> UserEncoder:
    import tensorflow as tf

    encoder = UserEncoder(channels=channels, name="user_encoder")
    features = tf.keras.Input(shape=(16, channels), name="calibration_features")
    gamma, beta = encoder(features)
    wrapper = tf.keras.Model(features, [gamma, beta], name="film_user_encoder")
    wrapper.load_weights(weights_path)
    return encoder


def _subject_partitions(
    data: dict[str, np.ndarray],
    simulator: UserSimulator,
) -> list[dict[str, Any]]:
    partitions = []
    for subject in sorted(np.unique(data["subject"])):
        subject_mask = data["subject"] == subject
        windows = data["x"][subject_mask]
        labels = data["y"][subject_mask]
        for sampling_seed in CALIBRATION_SEEDS:
            calibration, _, query, query_labels = split_user_calibration(
                windows,
                labels,
                max_calibration=max(CALIBRATION_SIZES),
                seed=sampling_seed,
                allow_adl_only=True,
            )
            calibration = simulator.transform(
                calibration,
                "Normal",
                user_id=f"real_heldout:{subject}",
                seed=sampling_seed + 1009,
            )
            query = simulator.transform(
                query,
                "Normal",
                user_id=f"real_heldout:{subject}",
                seed=sampling_seed + 1010,
            )
            partitions.append(
                {
                    "subject_id": str(subject),
                    "sampling_seed": sampling_seed,
                    "calibration": calibration,
                    "query": query,
                    "labels": query_labels.astype(np.int32, copy=False),
                }
            )
    return partitions


def _predict_partition(
    components: Any,
    encoder: UserEncoder,
    partition: dict[str, Any],
    fine_tune_configs: dict[str, dict[str, Any]],
) -> dict[tuple[int, str], np.ndarray]:
    predictions: dict[tuple[int, str], np.ndarray] = {}
    query = partition["query"]
    labels = partition["labels"]
    global_probabilities = predict_model_probabilities(components.original, query)
    for k in CALIBRATION_SIZES:
        calibration = partition["calibration"][:k]
        zero_labels = np.zeros(len(calibration), dtype=np.int32)
        predictions[(k, "Global")] = global_probabilities
        for model, full_model, config_key in (
            ("Last-layer FT (ADL-only)", False, "last_layer"),
            ("Full FT (ADL-only)", True, "full_model"),
        ):
            config = fine_tune_configs[config_key]["selected_config"]
            adapted = fine_tune_baseline(
                components.original,
                calibration,
                zero_labels,
                full_model=full_model,
                epochs=int(config["epochs"]),
                learning_rate=float(config["learning_rate"]),
            )
            predictions[(k, model)] = predict_model_probabilities(adapted, query)
        film_probability, _, _ = film_probabilities(
            components,
            encoder,
            query,
            calibration,
        )
        predictions[(k, "FiLM")] = film_probability
    return predictions


def _validation_thresholds(
    seed: int,
    components: Any,
    encoder: UserEncoder,
    fine_tune_configs: dict[str, dict[str, Any]],
) -> tuple[dict[tuple[int, str], float], list[dict[str, Any]]]:
    val_data = _load_arrays()["val"]
    simulator = UserSimulator(seed=seed)
    pooled: dict[tuple[int, str], tuple[list[np.ndarray], list[np.ndarray]]] = {
        (k, model): ([], []) for k in CALIBRATION_SIZES for model in MODELS
    }
    for partition in _subject_partitions(val_data, simulator):
        predictions = _predict_partition(components, encoder, partition, fine_tune_configs)
        for key, probabilities in predictions.items():
            pooled[key][0].append(partition["labels"])
            pooled[key][1].append(probabilities)

    thresholds = {}
    threshold_rows = []
    for k in CALIBRATION_SIZES:
        for model in MODELS:
            labels = np.concatenate(pooled[(k, model)][0])
            probabilities = np.concatenate(pooled[(k, model)][1])
            threshold, validation_metrics = _choose_threshold(labels, probabilities)
            thresholds[(k, model)] = threshold
            threshold_rows.append(
                {
                    "meta_seed": seed,
                    "model": model,
                    "k": k,
                    "validation_threshold": threshold,
                    "validation_f1": validation_metrics["f1"],
                    "validation_f2": validation_metrics["f2"],
                    "validation_false_alarms_per_hour": validation_metrics[
                        "false_alarms_per_hour"
                    ],
                    "threshold_selection": "pooled validation users/seeds; max F2, then lower FA/hr",
                    "validation_sampling_seeds": ";".join(map(str, CALIBRATION_SEEDS)),
                    "rows_validation_calibrated": True,
                }
            )
    return thresholds, threshold_rows


def _user_metric_row(
    *,
    seed: int,
    model: str,
    k: int,
    scope: str,
    subject_id: str,
    sampling_seed: int | str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    selected = _metrics(labels, probabilities, threshold)
    reference = _metrics(labels, probabilities, REFERENCE_THRESHOLD)
    return {
        "meta_seed": seed,
        "model": model,
        "k": k,
        "scope": scope,
        "subject_id": subject_id,
        "calibration_sampling_seed": sampling_seed,
        "validation_threshold": threshold,
        "rows_validation_calibrated": True,
        **{f"selected_{name}": value for name, value in selected.items()},
        **{f"reference_0_25_{name}": value for name, value in reference.items()},
        "fall_windows": int(np.count_nonzero(labels == 1)),
        "adl_windows": int(np.count_nonzero(labels == 0)),
    }


def _evaluate_test(
    seed: int,
    components: Any,
    encoder: UserEncoder,
    fine_tune_configs: dict[str, dict[str, Any]],
    thresholds: dict[tuple[int, str], float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    test_data = _load_arrays()["test"]
    simulator = UserSimulator(seed=seed)
    user_rows = []
    pooled_rows = []
    affine_rows = []
    pooled_predictions: dict[tuple[int, str], dict[str, list[np.ndarray]]] = {
        (k, model): {"all_labels": [], "all_probabilities": [], "fall_labels": [], "fall_probabilities": []}
        for k in CALIBRATION_SIZES
        for model in MODELS
    }
    for partition in _subject_partitions(test_data, simulator):
        subject_id = partition["subject_id"]
        predictions = _predict_partition(components, encoder, partition, fine_tune_configs)
        for k in CALIBRATION_SIZES:
            film_probability, gamma, beta = film_probabilities(
                components,
                encoder,
                partition["query"],
                partition["calibration"][:k],
            )
            predictions[(k, "FiLM")] = film_probability
            affine_rows.append(
                {
                    "meta_seed": seed,
                    "subject_id": subject_id,
                    "calibration_sampling_seed": partition["sampling_seed"],
                    "k": k,
                    "gamma": json.dumps(gamma.astype(float).tolist()),
                    "beta": json.dumps(beta.astype(float).tolist()),
                    "mean_abs_gamma_minus_one": float(np.mean(np.abs(gamma - 1.0))),
                    "mean_abs_beta": float(np.mean(np.abs(beta))),
                }
            )
            for model in MODELS:
                bucket = pooled_predictions[(k, model)]
                bucket["all_labels"].append(partition["labels"])
                bucket["all_probabilities"].append(predictions[(k, model)])
                if np.any(partition["labels"] == 1):
                    bucket["fall_labels"].append(partition["labels"])
                    bucket["fall_probabilities"].append(predictions[(k, model)])
                user_rows.append(
                    _user_metric_row(
                        seed=seed,
                        model=model,
                        k=k,
                        scope="per_user",
                        subject_id=subject_id,
                        sampling_seed=partition["sampling_seed"],
                        labels=partition["labels"],
                        probabilities=predictions[(k, model)],
                        threshold=thresholds[(k, model)],
                    )
                )

    for k in CALIBRATION_SIZES:
        for model in MODELS:
            bucket = pooled_predictions[(k, model)]
            for scope, labels_parts, probability_parts in (
                ("pooled_all_users", bucket["all_labels"], bucket["all_probabilities"]),
                (
                    "pooled_fall_subjects",
                    bucket["fall_labels"],
                    bucket["fall_probabilities"],
                ),
            ):
                pooled_rows.append(
                    _user_metric_row(
                        seed=seed,
                        model=model,
                        k=k,
                        scope=scope,
                        subject_id="ALL",
                        sampling_seed="all",
                        labels=np.concatenate(labels_parts),
                        probabilities=np.concatenate(probability_parts),
                        threshold=thresholds[(k, model)],
                    )
                )
    return user_rows, pooled_rows, affine_rows


def _summarize_threshold_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    metric_names = ("f1", "f2", "false_alarms_per_hour", "auroc", "auprc")
    for (model, k, scope), group in rows.groupby(["model", "k", "scope"], dropna=False):
        # Per-user summaries retain user-level variation across seeds. ADL-only
        # subjects have NaN fall-sensitive scores and are excluded from F1/F2.
        for metric in metric_names:
            values = group[f"selected_{metric}"].replace([np.inf, -np.inf], np.nan).dropna()
            records.append(
                {
                    "model": model,
                    "k": int(k),
                    "scope": scope,
                    "metric": metric,
                    "validation_calibrated_mean": float(values.mean()) if len(values) else np.nan,
                    "validation_calibrated_std": float(values.std(ddof=1))
                    if len(values) > 1
                    else np.nan,
                    "observations": int(len(values)),
                    "reference_0_25_mean": float(
                        group[f"reference_0_25_{metric}"].mean()
                    ),
                    "reference_0_25_std": float(
                        group[f"reference_0_25_{metric}"].std(ddof=1)
                    )
                    if group[f"reference_0_25_{metric}"].count() > 1
                    else np.nan,
                }
            )
    return pd.DataFrame(records)


def _probability_diagnostics(
    seed: int,
    components: Any,
    encoder: UserEncoder,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    test_data = _load_arrays()["test"]
    simulator = UserSimulator(seed=seed)
    score_rows = []
    affine_rows = []
    for partition in _subject_partitions(test_data, simulator):
        global_probabilities = predict_model_probabilities(
            components.original, partition["query"]
        )
        for k in (0, 1, 30):
            film_probabilities_values, gamma, beta = film_probabilities(
                components,
                encoder,
                partition["query"],
                partition["calibration"][:k],
            )
            affine_rows.append(
                {
                    "meta_seed": seed,
                    "subject_id": partition["subject_id"],
                    "calibration_sampling_seed": partition["sampling_seed"],
                    "k": k,
                    "gamma": gamma.astype(float),
                    "beta": beta.astype(float),
                }
            )
            for model, probabilities in (
                ("Global", global_probabilities),
                ("FiLM", film_probabilities_values),
            ):
                for label, label_name in ((0, "ADL"), (1, "Fall")):
                    values = probabilities[partition["labels"] == label]
                    if len(values):
                        score_rows.append(
                            {
                                "meta_seed": seed,
                                "subject_id": partition["subject_id"],
                                "sampling_seed": partition["sampling_seed"],
                                "k": k,
                                "model": model,
                                "class": label_name,
                                "mean_probability": float(np.mean(values)),
                                "median_probability": float(np.median(values)),
                                "windows": int(len(values)),
                            }
                        )
    return score_rows, affine_rows


def _affine_variability(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(rows)
    summary_rows = []
    channel_rows = []
    for k, group in raw.groupby("k"):
        users = sorted(group.subject_id.unique())
        gamma_by_user = []
        beta_by_user = []
        for user in users:
            user_rows = group[group.subject_id == user]
            gamma_by_user.append(np.stack(user_rows.gamma.to_list()))
            beta_by_user.append(np.stack(user_rows.beta.to_list()))
        gamma_stack = np.stack(gamma_by_user)  # user x replicate x channel
        beta_stack = np.stack(beta_by_user)
        within_gamma = np.std(gamma_stack, axis=1, ddof=1).mean(axis=0)
        within_beta = np.std(beta_stack, axis=1, ddof=1).mean(axis=0)
        user_gamma_means = gamma_stack.mean(axis=1)
        user_beta_means = beta_stack.mean(axis=1)
        between_gamma = np.std(user_gamma_means, axis=0, ddof=1)
        between_beta = np.std(user_beta_means, axis=0, ddof=1)
        summary_rows.append(
            {
                "k": int(k),
                "within_user_gamma_std_mean_channels": float(within_gamma.mean()),
                "between_user_gamma_std_mean_channels": float(between_gamma.mean()),
                "within_user_beta_std_mean_channels": float(within_beta.mean()),
                "between_user_beta_std_mean_channels": float(between_beta.mean()),
                "mean_abs_gamma_minus_one": float(np.mean(np.abs(gamma_stack - 1.0))),
                "mean_abs_beta": float(np.mean(np.abs(beta_stack))),
                "users": len(users),
                "replicates_per_user": int(min(len(values) for values in gamma_by_user)),
            }
        )
        for channel in range(gamma_stack.shape[-1]):
            channel_rows.append(
                {
                    "k": int(k),
                    "feature_channel": channel,
                    "within_user_gamma_std": float(within_gamma[channel]),
                    "between_user_gamma_std": float(between_gamma[channel]),
                    "within_user_beta_std": float(within_beta[channel]),
                    "between_user_beta_std": float(between_beta[channel]),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(channel_rows)


def _training_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = []
    summaries = []
    for seed in SEEDS:
        path = RUNS_DIR / f"seed{seed}" / "meta_training_history.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "meta_seed", seed)
        curves.append(frame)
        best_index = int(frame["val_loss"].idxmin())
        summaries.append(
            {
                "meta_seed": seed,
                "epochs_run": int(len(frame)),
                "configured_epochs": 30,
                "best_epoch": int(frame.loc[best_index, "epoch"]),
                "best_validation_loss": float(frame.loc[best_index, "val_loss"]),
                "last_validation_loss": float(frame.iloc[-1]["val_loss"]),
                "early_stopped": len(frame) < 30,
                "validation_loss_after_best": float(frame.iloc[-1]["val_loss"])
                > float(frame.loc[best_index, "val_loss"]),
            }
        )
    return pd.concat(curves, ignore_index=True), pd.DataFrame(summaries)


def _channel_shift_diagnostics() -> pd.DataFrame:
    data = _load_arrays()
    train = data["train"]["x"].reshape(-1, 6)
    train_mean = train.mean(axis=0)
    train_std = train.std(axis=0)
    rows = []
    for subject in sorted(np.unique(data["test"]["subject"])):
        values = data["test"]["x"][data["test"]["subject"] == subject].reshape(-1, 6)
        means = values.mean(axis=0)
        stds = values.std(axis=0)
        for channel in range(6):
            rows.append(
                {
                    "subject_id": str(subject),
                    "channel_index": channel,
                    "train_mean": float(train_mean[channel]),
                    "train_std": float(train_std[channel]),
                    "test_subject_mean": float(means[channel]),
                    "mean_shift_in_train_sd": float(
                        (means[channel] - train_mean[channel]) / train_std[channel]
                    ),
                    "test_subject_std": float(stds[channel]),
                    "std_ratio_to_train": float(stds[channel] / train_std[channel]),
                }
            )
    return pd.DataFrame(rows)


def _assert_output_files_absent() -> None:
    outputs = (
        "validation_thresholds.csv",
        "test_threshold_comparison.csv",
        "test_metric_summary.csv",
        "probability_diagnostics.csv",
        "affine_diagnostics.csv",
        "affine_variability.csv",
        "affine_variability_by_channel.csv",
        "training_curves.csv",
        "training_curve_summary.csv",
        "real_test_channel_shifts.csv",
    )
    existing = [name for name in outputs if (OUTPUT_DIR / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing analysis outputs: " + ", ".join(existing)
        )


def main() -> None:
    _assert_output_files_absent()
    all_threshold_rows = []
    all_test_rows = []
    all_affine_rows = []
    all_probability_rows = []
    diagnostic_affine_rows = []
    for seed in SEEDS:
        run_dir = RUNS_DIR / f"seed{seed}"
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        components = _load_backbone(_load_arrays()["train"]["x"])
        encoder = _load_encoder(
            run_dir / "user_encoder.weights.h5",
            components.feature_shape[-1],
        )
        ft_configs = metadata["fine_tune_tuning"]["methods"]
        thresholds, threshold_rows = _validation_thresholds(
            seed, components, encoder, ft_configs
        )
        all_threshold_rows.extend(threshold_rows)
        test_rows, _, affine_rows = _evaluate_test(
            seed, components, encoder, ft_configs, thresholds
        )
        all_test_rows.extend(test_rows)
        all_affine_rows.extend(affine_rows)
        score_rows, diag_affine = _probability_diagnostics(
            seed, components, encoder
        )
        all_probability_rows.extend(score_rows)
        diagnostic_affine_rows.extend(diag_affine)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    thresholds_frame = pd.DataFrame(all_threshold_rows)
    test_frame = pd.DataFrame(all_test_rows)
    thresholds_frame.to_csv(OUTPUT_DIR / "validation_thresholds.csv", index=False)
    test_frame.to_csv(OUTPUT_DIR / "test_threshold_comparison.csv", index=False)
    _summarize_threshold_metrics(test_frame).to_csv(
        OUTPUT_DIR / "test_metric_summary.csv", index=False
    )
    probability_frame = pd.DataFrame(all_probability_rows)
    probability_frame.to_csv(
        OUTPUT_DIR / "probability_diagnostics.csv", index=False
    )
    affine_summary, affine_by_channel = _affine_variability(diagnostic_affine_rows)
    affine_summary.to_csv(OUTPUT_DIR / "affine_variability.csv", index=False)
    affine_by_channel.to_csv(
        OUTPUT_DIR / "affine_variability_by_channel.csv", index=False
    )
    pd.DataFrame(all_affine_rows).to_csv(
        OUTPUT_DIR / "affine_diagnostics.csv", index=False
    )
    curves, curve_summary = _training_curves()
    curves.to_csv(OUTPUT_DIR / "training_curves.csv", index=False)
    curve_summary.to_csv(OUTPUT_DIR / "training_curve_summary.csv", index=False)
    _channel_shift_diagnostics().to_csv(
        OUTPUT_DIR / "real_test_channel_shifts.csv", index=False
    )
    print(f"Wrote validation thresholds and test metrics under {OUTPUT_DIR}")
    print(f"Threshold rows: {len(thresholds_frame)}; test per-user rows: {len(test_frame)}")


if __name__ == "__main__":
    main()
