import numpy as np
import tensorflow as tf

from src.severity_multitask import (
    count_uncertain_fall_rows,
    select_confident_fall_severity_rows,
    select_fall_severity_rows,
    severity_class_weights,
    severity_loss,
)


def test_masked_rows_have_zero_loss_and_zero_gradient():
    labels = tf.constant([0, -1], dtype=tf.int32)
    logits = tf.Variable([[1.0, 0.0], [0.0, 2.0]])
    with tf.GradientTape() as tape:
        loss = severity_loss(labels, logits, [True, True], [1.0, 1.0])
    gradient = tape.gradient(loss, logits).numpy()
    valid_only_loss = severity_loss([0], [[1.0, 0.0]], [True], [1.0, 1.0])

    assert np.isfinite(loss.numpy())
    assert loss.numpy() == valid_only_loss.numpy()
    assert gradient[0].any()
    assert np.array_equal(gradient[1], np.zeros(2))


def test_all_masked_batch_returns_finite_zero_loss():
    loss = severity_loss([-1, -1], [[1.0, 0.0], [0.0, 1.0]], [False, False], [1.0, 1.0])

    assert np.isfinite(loss.numpy())
    assert loss.numpy() == 0.0


def test_severity_class_weights_use_only_confident_masked_train_labels():
    train_mask = np.asarray([True, True, True, True, False])
    weights = severity_class_weights(
        np.asarray([0, 0, 1, -1, 1]),
        train_mask,
    )

    assert weights.shape == (2,)
    assert np.allclose(weights, [0.75, 1.5])
    assert np.array_equal(train_mask, [True, True, True, True, False])


def test_severity_output_shapes_and_adl_rows_are_excluded_from_evaluation():
    binary = np.asarray([0, 1, 1, 1])
    severity = np.asarray([-1, 0, 1, -1])
    logits = np.arange(8, dtype=np.float32).reshape(4, 2)

    selected_labels, selected_logits = select_confident_fall_severity_rows(
        binary, severity, logits
    )

    assert selected_labels.shape == (2,)
    assert selected_logits.shape == (2, 2)
    assert np.array_equal(selected_labels, [0, 1])
    assert np.array_equal(selected_logits, logits[1:3])
    assert count_uncertain_fall_rows(binary, severity) == 1

    fall_labels, fall_head_inputs = select_fall_severity_rows(binary, severity, logits)
    assert fall_labels.shape == (3,)
    assert fall_head_inputs.shape == (3, 2)
    assert np.array_equal(fall_labels, [0, 1, -1])
    assert np.array_equal(fall_head_inputs, logits[1:])
