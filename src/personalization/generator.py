"""Optional class-conditional TimeVAE and synthetic-data quality checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def validate_generator_training_split(
    train_subject_ids: np.ndarray,
    generator_subject_ids: np.ndarray,
    held_out_subject_ids: np.ndarray,
) -> None:
    train = set(np.asarray(train_subject_ids).astype(str))
    generator = set(np.asarray(generator_subject_ids).astype(str))
    held_out = set(np.asarray(held_out_subject_ids).astype(str))
    if not generator or not generator.issubset(train):
        raise ValueError("TimeVAE training must use train-subject windows only.")
    if generator & held_out:
        raise ValueError("Held-out subject windows cannot be used to train TimeVAE.")


class ConditionalTimeVAE:
    """Compact class-conditional variational autoencoder for 64x6 windows."""

    def __init__(self, latent_dim: int = 16, seed: int = 42) -> None:
        self.latent_dim = latent_dim
        self.seed = seed
        tf.keras.utils.set_random_seed(seed)
        window_input = tf.keras.Input(shape=(64, 6), name="window")
        label_input = tf.keras.Input(shape=(), dtype=tf.int32, name="label")
        label_vector = tf.keras.layers.CategoryEncoding(
            num_tokens=2, output_mode="one_hot", name="class_condition"
        )(label_input)
        flattened = tf.keras.layers.Flatten()(window_input)
        encoded = tf.keras.layers.Concatenate()([flattened, label_vector])
        encoded = tf.keras.layers.Dense(128, activation="relu")(encoded)
        encoded = tf.keras.layers.Dense(64, activation="relu")(encoded)
        mean = tf.keras.layers.Dense(latent_dim, name="latent_mean")(encoded)
        log_variance = tf.keras.layers.Dense(latent_dim, name="latent_log_variance")(encoded)
        self.encoder = tf.keras.Model([window_input, label_input], [mean, log_variance])

        latent_input = tf.keras.Input(shape=(latent_dim,), name="latent")
        decoder_label = tf.keras.Input(shape=(), dtype=tf.int32, name="label")
        decoder_condition = tf.keras.layers.CategoryEncoding(
            num_tokens=2, output_mode="one_hot"
        )(decoder_label)
        decoded = tf.keras.layers.Concatenate()([latent_input, decoder_condition])
        decoded = tf.keras.layers.Dense(128, activation="relu")(decoded)
        decoded = tf.keras.layers.Dense(64 * 32, activation="relu")(decoded)
        decoded = tf.keras.layers.Reshape((64, 32))(decoded)
        decoded = tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu")(decoded)
        reconstruction = tf.keras.layers.Conv1D(6, 3, padding="same")(decoded)
        self.decoder = tf.keras.Model([latent_input, decoder_label], reconstruction)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def fit(
        self,
        train_windows: np.ndarray,
        train_labels: np.ndarray,
        train_subject_ids: np.ndarray,
        *,
        held_out_subject_ids: np.ndarray,
        epochs: int = 30,
        batch_size: int = 128,
    ) -> list[dict[str, float]]:
        validate_generator_training_split(
            train_subject_ids, train_subject_ids, held_out_subject_ids
        )
        windows = np.asarray(train_windows, dtype=np.float32)
        labels = np.asarray(train_labels, dtype=np.int32)
        if windows.shape != (len(labels), 64, 6) or len(labels) != len(train_subject_ids):
            raise ValueError("TimeVAE train windows, labels, and subject IDs must align.")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("TimeVAE class labels must be binary.")
        rng = np.random.default_rng(self.seed)
        history: list[dict[str, float]] = []
        for epoch in range(epochs):
            order = rng.permutation(len(windows))
            losses = []
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                batch_x = tf.convert_to_tensor(windows[indices])
                batch_y = tf.convert_to_tensor(labels[indices])
                with tf.GradientTape() as tape:
                    mean, log_variance = self.encoder([batch_x, batch_y], training=True)
                    epsilon = tf.random.normal(tf.shape(mean), seed=self.seed + epoch)
                    latent = mean + tf.exp(0.5 * log_variance) * epsilon
                    reconstructed = self.decoder([latent, batch_y], training=True)
                    reconstruction_loss = tf.reduce_mean(tf.square(batch_x - reconstructed))
                    kl_loss = -0.5 * tf.reduce_mean(
                        1.0 + log_variance - tf.square(mean) - tf.exp(log_variance)
                    )
                    loss = reconstruction_loss + 0.01 * kl_loss
                variables = self.encoder.trainable_variables + self.decoder.trainable_variables
                gradients = tape.gradient(loss, variables)
                self.optimizer.apply_gradients(zip(gradients, variables))
                losses.append(float(loss.numpy()))
            history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses))})
        return history

    def generate(self, labels: np.ndarray, *, seed: int = 42) -> np.ndarray:
        classes = np.asarray(labels, dtype=np.int32).reshape(-1)
        if not np.isin(classes, [0, 1]).all():
            raise ValueError("Generation labels must be binary.")
        latent = tf.random.stateless_normal(
            (len(classes), self.latent_dim), seed=(self.seed, int(seed))
        )
        generated = self.decoder(
            [latent, tf.convert_to_tensor(classes)], training=False
        ).numpy()
        return np.asarray(generated, dtype=np.float32)

    def save_weights(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.encoder.save_weights(path.with_name(f"{path.stem}_encoder.weights.h5"))
        self.decoder.save_weights(path.with_name(f"{path.stem}_decoder.weights.h5"))


def evaluate_generator_quality(
    real_windows: np.ndarray,
    synthetic_windows: np.ndarray,
    synthetic_labels: np.ndarray,
    real_test_windows: np.ndarray,
    real_test_labels: np.ndarray,
    output_dir: Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Save channel/SMV/PCA comparisons and train-synthetic-test-real scores."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    real = np.asarray(real_windows, dtype=np.float32)
    synthetic = np.asarray(synthetic_windows, dtype=np.float32)
    test = np.asarray(real_test_windows, dtype=np.float32)
    synthetic_y = np.asarray(synthetic_labels, dtype=np.int32)
    test_y = np.asarray(real_test_labels, dtype=np.int32)
    if real.ndim != 3 or synthetic.shape[1:] != (64, 6) or test.shape[1:] != (64, 6):
        raise ValueError("Real and synthetic quality-check windows must have shape (N, 64, 6).")
    if len(synthetic) != len(synthetic_y) or len(test) != len(test_y):
        raise ValueError("Quality-check labels must align with their windows.")

    rows = []
    for channel in range(6):
        rows.append(
            {
                "channel": channel,
                "real_mean": float(np.mean(real[:, :, channel])),
                "synthetic_mean": float(np.mean(synthetic[:, :, channel])),
                "real_std": float(np.std(real[:, :, channel])),
                "synthetic_std": float(np.std(synthetic[:, :, channel])),
            }
        )
    channel_stats = pd.DataFrame(rows)
    channel_stats.to_csv(output_dir / "timevae_channel_stats.csv", index=False)

    real_smv = np.sqrt(np.sum(real[:, :, :3] ** 2, axis=-1)).reshape(-1)
    synthetic_smv = np.sqrt(np.sum(synthetic[:, :, :3] ** 2, axis=-1)).reshape(-1)
    edges = np.linspace(
        float(min(real_smv.min(), synthetic_smv.min())),
        float(max(real_smv.max(), synthetic_smv.max())),
        51,
    )
    hist_rows = []
    for index, (real_count, synthetic_count) in enumerate(
        zip(np.histogram(real_smv, bins=edges)[0], np.histogram(synthetic_smv, bins=edges)[0])
    ):
        hist_rows.append({"bin": index, "real_count": int(real_count), "synthetic_count": int(synthetic_count)})
    pd.DataFrame(hist_rows).to_csv(output_dir / "timevae_smv_histogram.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(real_smv, bins=edges, density=True, alpha=0.5, label="real train")
    ax.hist(synthetic_smv, bins=edges, density=True, alpha=0.5, label="synthetic")
    ax.set_xlabel("Acceleration magnitude")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "timevae_smv_histogram.png", dpi=150)
    plt.close(fig)

    sample_count = min(1500, len(real), len(synthetic))
    pooled = np.concatenate((real[:sample_count], synthetic[:sample_count]), axis=0).reshape(2 * sample_count, -1)
    coordinates = PCA(n_components=2, random_state=seed).fit_transform(pooled)
    pca_table = pd.DataFrame(
        {
            "pca_1": coordinates[:, 0],
            "pca_2": coordinates[:, 1],
            "source": ["real"] * sample_count + ["synthetic"] * sample_count,
        }
    )
    pca_table.to_csv(output_dir / "timevae_pca.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    for source, group in pca_table.groupby("source"):
        ax.scatter(group["pca_1"], group["pca_2"], s=5, alpha=0.35, label=source)
    ax.legend()
    ax.set_title("Real versus synthetic windows (PCA)")
    fig.tight_layout()
    fig.savefig(output_dir / "timevae_pca.png", dpi=150)
    plt.close(fig)

    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    classifier.fit(synthetic.reshape(len(synthetic), -1), synthetic_y)
    test_predictions = classifier.predict(test.reshape(len(test), -1))
    synthetic_test_metrics = {
        "accuracy": float(accuracy_score(test_y, test_predictions)),
        "precision": float(precision_score(test_y, test_predictions, zero_division=0)),
        "recall": float(recall_score(test_y, test_predictions, zero_division=0)),
        "f1": float(f1_score(test_y, test_predictions, zero_division=0)),
    }
    pd.DataFrame([synthetic_test_metrics]).to_csv(
        output_dir / "timevae_train_synthetic_test_real.csv", index=False
    )
    return {"channel_stats": rows, "train_synthetic_test_real": synthetic_test_metrics}
