"""Loss and row-selection helpers for a binary-plus-severity model."""

from __future__ import annotations

import numpy as np


def severity_class_weights(y_sev: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute balanced Mild/Severe weights from confident training labels."""

    labels = np.asarray(y_sev)
    valid = np.array(mask, dtype=bool, copy=True)
    if labels.ndim != 1 or valid.shape != labels.shape:
        raise ValueError("Severity labels and mask must be aligned one-dimensional arrays.")
    if not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("Severity labels must be -1 (uncertain), 0 (Mild), or 1 (Severe).")

    valid &= labels >= 0
    counts = np.bincount(labels[valid].astype(np.int64), minlength=2)
    present = counts > 0
    weights = np.zeros(2, dtype=np.float32)
    if np.any(present):
        weights[present] = counts.sum() / (present.sum() * counts[present])
    return weights


def severity_loss(
    y_sev: np.ndarray,
    logits: np.ndarray,
    mask: np.ndarray,
    class_weights: np.ndarray,
):
    """Weighted sparse cross-entropy; masked and uncertain rows have no gradient."""

    import tensorflow as tf

    labels = tf.convert_to_tensor(y_sev)
    logits_tensor = tf.convert_to_tensor(logits)
    valid = tf.cast(tf.convert_to_tensor(mask), tf.bool)
    weights = tf.cast(tf.convert_to_tensor(class_weights), logits_tensor.dtype)

    if labels.shape.rank != 1:
        raise ValueError("Severity labels must be one-dimensional.")
    if logits_tensor.shape.rank != 2 or logits_tensor.shape[-1] != 2:
        raise ValueError("Severity logits must have shape (batch, 2).")
    if valid.shape.rank != 1 or weights.shape != (2,):
        raise ValueError("Mask must be one-dimensional and class_weights must have shape (2,).")
    if labels.shape[0] != logits_tensor.shape[0] or valid.shape[0] != labels.shape[0]:
        raise ValueError("Severity labels, logits, and mask must have matching batch sizes.")

    labels = tf.cast(labels, tf.int32)
    valid = tf.logical_and(valid, labels >= 0)
    safe_labels = tf.where(valid, labels, tf.zeros_like(labels))
    per_row = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=safe_labels,
        logits=logits_tensor,
    )
    row_weights = tf.gather(weights, safe_labels) * tf.cast(valid, logits_tensor.dtype)
    return tf.math.divide_no_nan(
        tf.reduce_sum(per_row * row_weights),
        tf.reduce_sum(row_weights),
    )


def select_fall_severity_rows(
    binary_labels: np.ndarray,
    severity_labels: np.ndarray,
    head_inputs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Route only fall representations to the severity head, retaining uncertain labels."""

    binary = np.asarray(binary_labels)
    severity = np.asarray(severity_labels)
    values = np.asarray(head_inputs)
    if binary.ndim != 1 or severity.shape != binary.shape:
        raise ValueError("Binary and severity labels must be aligned one-dimensional arrays.")
    if values.ndim < 2 or values.shape[0] != len(binary):
        raise ValueError("Severity head inputs must align with the binary labels.")
    if not np.isin(binary, [0, 1]).all() or not np.isin(severity, [-1, 0, 1]).all():
        raise ValueError("Unexpected binary or severity labels.")
    fall_rows = binary == 1
    return severity[fall_rows].astype(np.int32, copy=False), values[fall_rows]


def select_confident_fall_severity_rows(
    binary_labels: np.ndarray,
    severity_labels: np.ndarray,
    logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select only confident Mild/Severe fall rows for severity evaluation."""

    binary = np.asarray(binary_labels)
    severity = np.asarray(severity_labels)
    values = np.asarray(logits)
    if binary.ndim != 1 or severity.shape != binary.shape:
        raise ValueError("Binary and severity labels must be aligned one-dimensional arrays.")
    if values.ndim != 2 or values.shape != (len(binary), 2):
        raise ValueError("Severity logits must have shape (number_of_rows, 2).")
    if not np.isin(binary, [0, 1]).all() or not np.isin(severity, [-1, 0, 1]).all():
        raise ValueError("Unexpected binary or severity labels.")
    fall_labels, fall_logits = select_fall_severity_rows(binary, severity, values)
    confident = fall_labels >= 0
    return fall_labels[confident], fall_logits[confident]


def count_uncertain_fall_rows(binary_labels: np.ndarray, severity_labels: np.ndarray) -> int:
    """Count binary-positive rows excluded from severity-class metrics as uncertain."""

    binary = np.asarray(binary_labels)
    severity = np.asarray(severity_labels)
    if binary.ndim != 1 or severity.shape != binary.shape:
        raise ValueError("Binary and severity labels must be aligned one-dimensional arrays.")
    if not np.isin(binary, [0, 1]).all() or not np.isin(severity, [-1, 0, 1]).all():
        raise ValueError("Unexpected binary or severity labels.")
    return int(np.count_nonzero((binary == 1) & (severity == -1)))
