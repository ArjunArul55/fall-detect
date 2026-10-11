"""Split and freeze the canonical CNN-LSTM at the CNN/LSTM boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrozenComponents:
    """The frozen feature extractor, temporal classifier head, and source model."""

    extractor: Any
    head: Any
    original: Any
    feature_shape: tuple[int, int]


def split_frozen_model(
    model: Any,
    validation_windows: np.ndarray,
    *,
    expected_windows: int = 1000,
    tolerance: float = 1e-5,
) -> FrozenComponents:
    """Reuse model layers and weights, checking exact forward equivalence."""

    if validation_windows.ndim != 3 or tuple(validation_windows.shape[1:]) != (64, 6):
        raise ValueError(
            "validation_windows must have shape (N, 64, 6); "
            f"received {validation_windows.shape}."
        )
    if len(validation_windows) < expected_windows:
        raise ValueError(
            f"At least {expected_windows} windows are required to verify the split; "
            f"received {len(validation_windows)}."
        )

    cut_layer = next(
        (layer for layer in model.layers if layer.name == "max_pooling1d_1"),
        None,
    )
    if cut_layer is None:
        raise ValueError("Canonical CNN-LSTM layer 'max_pooling1d_1' was not found.")

    import tensorflow as tf

    extractor = tf.keras.Model(
        inputs=model.inputs[0],
        outputs=cut_layer.output,
        name="frozen_cnn_extractor",
    )
    head_input_shape = tuple(int(size) for size in cut_layer.output.shape[1:])
    head_input = tf.keras.Input(shape=head_input_shape, name="cnn_features")
    features = head_input
    cut_index = model.layers.index(cut_layer)
    for layer in model.layers[cut_index + 1 :]:
        features = layer(features)
    head = tf.keras.Model(head_input, features, name="frozen_lstm_head")

    for component in (model, extractor, head):
        component.trainable = False
        for layer in component.layers:
            layer.trainable = False

    sample = np.asarray(validation_windows[:expected_windows], dtype=np.float32)
    original_output = model.predict(sample, batch_size=128, verbose=0)
    split_features = extractor.predict(sample, batch_size=128, verbose=0)
    split_output = head.predict(split_features, batch_size=128, verbose=0)
    max_error = float(np.max(np.abs(original_output - split_output)))
    if not np.allclose(original_output, split_output, atol=tolerance, rtol=0.0):
        raise AssertionError(
            f"Split CNN-LSTM differs from original by {max_error:.8g}; "
            f"tolerance is {tolerance}."
        )

    return FrozenComponents(
        extractor=extractor,
        head=head,
        original=model,
        feature_shape=head_input_shape,
    )
