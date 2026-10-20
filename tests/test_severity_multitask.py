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


def test_multitask_model_output_shapes():
    from src.severity_multitask import build_multitask_cnn_lstm

    model = build_multitask_cnn_lstm(tf, input_shape=(64, 6), unroll=True)
    batch_x = tf.zeros((4, 64, 6), dtype=tf.float32)
    outputs = model(batch_x, training=False)

    assert "binary_output" in outputs
    assert "severity_logits" in outputs
    assert "severity_output" in outputs
    assert outputs["binary_output"].shape == (4, 1)
    assert outputs["severity_logits"].shape == (4, 2)
    assert outputs["severity_output"].shape == (4, 2)
    assert np.allclose(tf.reduce_sum(outputs["severity_output"], axis=-1).numpy(), 1.0)
    assert model.count_params() == 46883


def test_multitask_head_ignores_adl_and_uncertain_rows_in_loss():
    from src.severity_multitask import compute_multitask_loss

    y_bin = tf.constant([0, 1, 1, 1], dtype=tf.float32)
    y_sev = tf.constant([-1, 0, 1, -1], dtype=tf.int32)
    b_prob = tf.constant([0.1, 0.9, 0.85, 0.95], dtype=tf.float32)
    s_logits = tf.Variable([[1.0, 0.0], [2.0, 0.5], [0.1, 1.5], [3.0, -1.0]])
    class_weights_bin = {0: 0.544, 1: 6.18}
    class_weights_sev = tf.constant([1.32, 0.80], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tot_loss, bin_loss, sev_loss = compute_multitask_loss(
            y_bin, y_sev, b_prob, s_logits, class_weights_bin, class_weights_sev, lambda_weight=0.3
        )
    grads = tape.gradient(sev_loss, s_logits).numpy()

    assert np.isfinite(tot_loss.numpy())
    assert np.isfinite(sev_loss.numpy())
    # ADL row (index 0) and uncertain fall row (index 3) must have exact zero gradients:
    assert np.allclose(grads[0], [0.0, 0.0])
    assert np.allclose(grads[3], [0.0, 0.0])
    # Confident fall rows (indices 1 and 2) must have nonzero gradients:
    assert not np.allclose(grads[1], [0.0, 0.0])
    assert not np.allclose(grads[2], [0.0, 0.0])


def test_multitask_tiny_smoke_train_step():
    from src.severity_multitask import build_multitask_cnn_lstm, compute_multitask_loss

    model = build_multitask_cnn_lstm(tf, input_shape=(64, 6), unroll=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    batch_x = tf.random.normal((4, 64, 6), seed=42)
    y_bin = tf.constant([0, 1, 1, 1], dtype=tf.float32)
    y_sev = tf.constant([-1, 0, 1, -1], dtype=tf.int32)
    class_weights_bin = {0: 0.544, 1: 6.18}
    class_weights_sev = tf.constant([1.32, 0.80], dtype=tf.float32)

    initial_weights = [w.numpy().copy() for w in model.trainable_variables]

    with tf.GradientTape() as tape:
        out = model(batch_x, training=True)
        total_loss, b_loss, s_loss = compute_multitask_loss(
            y_bin,
            y_sev,
            out["binary_output"],
            out["severity_logits"],
            class_weights_bin,
            class_weights_sev,
            lambda_weight=0.3,
        )
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    assert np.isfinite(total_loss.numpy())
    assert total_loss.numpy() > 0.0
    updated_weights = [w.numpy() for w in model.trainable_variables]
    assert any(not np.array_equal(w_init, w_upd) for w_init, w_upd in zip(initial_weights, updated_weights))


def test_multitask_trunk_initialization_matches_baseline():
    from pathlib import Path
    from src.severity_multitask import build_multitask_cnn_lstm
    from src.keras_model_loading import load_keras_model
    from scripts.train_cnn_lstm_baseline import build_model as build_baseline_fn

    baseline_path = Path("results/cnn_lstm_baseline/cnn_lstm_best.keras")
    if not baseline_path.exists():
        pytest.skip("Baseline cnn_lstm_best.keras not found")

    baseline = load_keras_model(tf, baseline_path, build_baseline_fn)
    model = build_multitask_cnn_lstm(tf, baseline_weights_path=baseline_path)

    sample_x = tf.random.normal((6, 64, 6), seed=123)
    baseline_pred = baseline(sample_x, training=False).numpy().ravel()
    multitask_pred = model(sample_x, training=False)["binary_output"].numpy().ravel()

    assert np.allclose(baseline_pred, multitask_pred, atol=1e-6)

