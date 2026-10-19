"""FiLM helpers: frozen CNN-LSTM, once-trained user encoder, inference-only adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import tensorflow as tf

from src.personalization.simulator import USER_PROFILES


CANONICAL_CLASS_WEIGHTS = {0: 0.5440134278254383, 1: 6.180084745762712}
CALIBRATION_SIZES = (0, 1, 5, 10, 30)


class UserEncoder(tf.keras.layers.Layer):
    """Map pooled calibration features (or a learned k=0 embedding) to FiLM."""

    def __init__(self, channels: int = 64, hidden_units: int = 128, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.channels = channels
        self.hidden_units = hidden_units
        self.hidden = tf.keras.layers.Dense(hidden_units, activation="relu", name="user_hidden")
        self.affine = tf.keras.layers.Dense(
            2 * channels,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="film_affine",
        )

    def build(self, input_shape: tf.TensorShape) -> None:
        input_shape = tf.TensorShape(input_shape)
        if input_shape.rank != 3 or input_shape[-1] != self.channels:
            raise ValueError(
                "UserEncoder expects calibration features shaped (K, T, C); "
                f"received {input_shape}."
            )
        self.z0 = self.add_weight(
            name="zero_shot_embedding",
            shape=(self.hidden_units,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, calibration_features: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        pooled_embedding = tf.cond(
            tf.shape(calibration_features)[0] > 0,
            lambda: self.hidden(tf.reduce_mean(calibration_features, axis=(0, 1))[None, :])[0],
            lambda: tf.identity(self.z0),
        )
        affine = self.affine(pooled_embedding[None, :])[0]
        delta_gamma, beta = tf.split(affine, 2, axis=-1)
        return 1.0 + delta_gamma, beta

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"channels": self.channels, "hidden_units": self.hidden_units})
        return config


def film_probabilities(
    components: Any,
    encoder: UserEncoder,
    query_windows: np.ndarray,
    calibration_windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return probabilities and the per-user gamma/beta, without gradient updates."""

    query = np.asarray(query_windows, dtype=np.float32)
    calibration = np.asarray(calibration_windows, dtype=np.float32)
    if query.ndim != 3 or tuple(query.shape[1:]) != (64, 6):
        raise ValueError(f"query_windows must have shape (N, 64, 6), got {query.shape}.")
    if calibration.ndim != 3 or tuple(calibration.shape[1:]) != (64, 6):
        raise ValueError(
            f"calibration_windows must have shape (K, 64, 6), got {calibration.shape}."
        )

    if len(calibration):
        calibration_features = components.extractor(calibration, training=False)
    else:
        calibration_features = tf.zeros((0, *components.feature_shape), dtype=tf.float32)
    query_features = components.extractor(query, training=False)
    return film_probabilities_from_features(
        components, encoder, query_features, calibration_features
    )


def film_probabilities_from_features(
    components: Any,
    encoder: UserEncoder,
    query_features: tf.Tensor,
    calibration_features: tf.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply per-user FiLM when frozen CNN features have already been computed."""

    gamma, beta = encoder(calibration_features, training=False)
    modulated = query_features * gamma[None, None, :] + beta[None, None, :]
    probabilities = components.head(modulated, training=False).numpy().reshape(-1)
    return probabilities, gamma.numpy(), beta.numpy()


def _episode_loss(
    components: Any,
    encoder: UserEncoder,
    query_windows: np.ndarray,
    query_labels: np.ndarray,
    calibration_windows: np.ndarray,
    *,
    training: bool,
) -> tuple[tf.Tensor, tf.Tensor]:
    if len(calibration_windows):
        calibration_features = components.extractor(calibration_windows, training=False)
    else:
        calibration_features = tf.zeros((0, *components.feature_shape), dtype=tf.float32)
    query_features = components.extractor(query_windows, training=False)
    with tf.GradientTape() as tape:
        gamma, beta = encoder(calibration_features, training=training)
        modulated = query_features * gamma[None, None, :] + beta[None, None, :]
        probabilities = components.head(modulated, training=False)
        labels = tf.convert_to_tensor(query_labels.reshape(-1, 1), dtype=tf.float32)
        per_example = tf.keras.losses.binary_crossentropy(labels, probabilities)
        weights = tf.where(labels[:, 0] > 0.5, CANONICAL_CLASS_WEIGHTS[1], CANONICAL_CLASS_WEIGHTS[0])
        loss = tf.reduce_mean(per_example * weights)
    return loss, tape


def _sample_episode(
    windows: np.ndarray,
    labels: np.ndarray,
    subject_indices: Mapping[str, np.ndarray],
    simulator: Any,
    rng: np.random.Generator,
    *,
    query_adl_windows: int,
    profile_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible_subjects = [
        subject
        for subject, subject_rows in subject_indices.items()
        if np.any(labels[subject_rows] == 0) and np.any(labels[subject_rows] == 1)
    ]
    if not eligible_subjects:
        raise ValueError("No subject has both ADL and fall windows for episodic sampling.")
    subject = str(rng.choice(eligible_subjects))
    indices = subject_indices[subject]
    subject_labels = labels[indices]
    adl_indices = indices[subject_labels == 0]
    fall_indices = indices[subject_labels == 1]
    if not len(adl_indices) or not len(fall_indices):
        raise ValueError(f"Meta-training subject {subject} must have ADL and fall windows.")

    profile = str(rng.choice(profile_names))
    calibration_count = int(rng.choice(CALIBRATION_SIZES))
    calibration_count = min(calibration_count, max(0, len(adl_indices) - 1))
    calibration_indices = rng.choice(adl_indices, size=calibration_count, replace=False)
    query_adl_candidates = np.setdiff1d(adl_indices, calibration_indices, assume_unique=True)
    query_adl_count = min(query_adl_windows, len(query_adl_candidates))
    query_fall_count = max(1, int(round(query_adl_count * (4484 / 50939))))
    query_adl = rng.choice(query_adl_candidates, size=query_adl_count, replace=False)
    query_falls = rng.choice(fall_indices, size=min(query_fall_count, len(fall_indices)), replace=False)
    query_indices = np.concatenate((query_adl, query_falls))
    query_indices = rng.permutation(query_indices)
    transform_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    calibration = simulator.transform(
        windows[calibration_indices],
        profile,
        user_id=f"meta:{subject}:{profile}",
        seed=transform_seed,
    )
    query = simulator.transform(
        windows[query_indices],
        profile,
        user_id=f"meta:{subject}:{profile}",
        seed=transform_seed + 1,
    )
    return calibration, query, labels[query_indices]


def _subject_index(subject_ids: np.ndarray) -> dict[str, np.ndarray]:
    strings = np.asarray(subject_ids).astype(str)
    return {subject: np.flatnonzero(strings == subject) for subject in np.unique(strings)}


def train_user_encoder(
    components: Any,
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    train_subject_ids: np.ndarray,
    val_windows: np.ndarray,
    val_labels: np.ndarray,
    val_subject_ids: np.ndarray,
    simulator: Any,
    *,
    seed: int = 42,
    epochs: int = 30,
    episodes_per_epoch: int = 30,
    validation_episodes: int = 12,
    query_adl_windows: int = 32,
    patience: int = 5,
    learning_rate: float = 1e-3,
) -> tuple[UserEncoder, list[dict[str, float]]]:
    """Meta-train only the UserEncoder on train subjects; early-stop on val."""

    train_subject_ids = np.asarray(train_subject_ids).astype(str)
    val_subject_ids = np.asarray(val_subject_ids).astype(str)
    if set(train_subject_ids) & set(val_subject_ids):
        raise ValueError("Meta-training and validation subjects must be disjoint.")
    if not len(train_windows) or len(train_windows) != len(train_labels) or len(train_windows) != len(train_subject_ids):
        raise ValueError("Train windows, labels, and subject IDs must be non-empty and aligned.")
    if len(val_windows) != len(val_labels) or len(val_windows) != len(val_subject_ids):
        raise ValueError("Validation windows, labels, and subject IDs must be aligned.")
    if epochs <= 0 or episodes_per_epoch <= 0 or validation_episodes <= 0:
        raise ValueError("Epoch and episode counts must be positive.")

    components.extractor.trainable = False
    components.head.trainable = False
    encoder = UserEncoder(channels=components.feature_shape[-1], name="user_encoder")
    encoder(tf.zeros((1, *components.feature_shape), dtype=tf.float32))
    encoder(tf.zeros((0, *components.feature_shape), dtype=tf.float32))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer.build(encoder.trainable_variables)
    train_index = _subject_index(train_subject_ids)
    val_index = _subject_index(val_subject_ids)
    train_rng = np.random.default_rng(seed)
    val_rng = np.random.default_rng(seed + 1)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_weights = encoder.get_weights()
    best_epoch = 0

    for epoch in range(epochs):
        training_losses: list[float] = []
        for _ in range(episodes_per_epoch):
            calibration, query, query_labels = _sample_episode(
                train_windows,
                train_labels,
                train_index,
                simulator,
                train_rng,
                query_adl_windows=query_adl_windows,
                profile_names=tuple(USER_PROFILES),
            )
            loss, tape = _episode_loss(
                components, encoder, query, query_labels, calibration, training=True
            )
            gradients = tape.gradient(loss, encoder.trainable_variables)
            gradient_pairs = [
                (gradient, variable)
                for gradient, variable in zip(gradients, encoder.trainable_variables)
                if gradient is not None
            ]
            if not gradient_pairs:
                raise RuntimeError("Meta-training episode produced no UserEncoder gradients.")
            optimizer.apply_gradients(gradient_pairs)
            training_losses.append(float(loss.numpy()))

        validation_losses: list[float] = []
        for _ in range(validation_episodes):
            calibration, query, query_labels = _sample_episode(
                val_windows,
                val_labels,
                val_index,
                simulator,
                val_rng,
                query_adl_windows=query_adl_windows,
                profile_names=tuple(USER_PROFILES),
            )
            loss, _ = _episode_loss(
                components, encoder, query, query_labels, calibration, training=False
            )
            validation_losses.append(float(loss.numpy()))
        train_loss = float(np.mean(training_losses))
        val_loss = float(np.mean(validation_losses))
        history.append({"epoch": float(epoch + 1), "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = encoder.get_weights()
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            break

    encoder.set_weights(best_weights)
    return encoder, history


def choose_f2_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int32)
    probs = np.asarray(probabilities, dtype=np.float64)
    thresholds = np.round(np.arange(0.1, 0.901, 0.05), 2)
    candidates: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        predictions = probs >= threshold
        true_positive = int(np.count_nonzero((labels == 1) & predictions))
        false_positive = int(np.count_nonzero((labels == 0) & predictions))
        false_negative = int(np.count_nonzero((labels == 1) & ~predictions))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
        candidates.append((f2, recall, -threshold))
    return float(-max(candidates)[2])


def save_user_encoder_weights(encoder: UserEncoder, path: Any) -> None:
    """Persist the layer through a Functional wrapper so weights are reloadable."""

    features = tf.keras.Input(shape=(16, encoder.channels), name="calibration_features")
    gamma, beta = encoder(features)
    model = tf.keras.Model(features, [gamma, beta], name="film_user_encoder")
    model.save_weights(path)
