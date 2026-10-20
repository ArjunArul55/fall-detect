from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_cnn_lstm_baseline import build_model
from src.keras_model_loading import load_keras_model
from src.personalization.film import UserEncoder, film_probabilities
from src.personalization.film import train_user_encoder
from src.personalization.evaluation import (
    calculate_metrics,
    fine_tune_config_grid,
    fine_tune_baseline,
    paired_wilcoxon,
    predict_model_probabilities,
    select_fine_tune_config,
)
from src.personalization.generator import (
    ConditionalTimeVAE,
    evaluate_generator_quality,
    validate_generator_training_split,
)
from src.personalization.model import split_frozen_model
from src.personalization.simulator import (
    USER_PROFILES,
    UserSimulator,
    assert_subject_disjoint,
    split_user_calibration,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CANONICAL_MODEL = ROOT / "results" / "cnn_lstm_baseline" / "cnn_lstm_best.keras"


def _load_components():
    import tensorflow as tf

    model = load_keras_model(tf, CANONICAL_MODEL, build_model)
    windows = np.load(PROCESSED / "train.npy", mmap_mode="r")
    return split_frozen_model(model, windows)


def test_frozen_model_split_matches_original_and_freezes_weights():
    components = _load_components()
    assert components.feature_shape == (16, 64)
    assert components.extractor.count_params() + components.head.count_params() == 46817
    assert not components.extractor.trainable
    assert not components.head.trainable
    assert all(not variable.trainable for variable in components.extractor.weights)
    assert all(not variable.trainable for variable in components.head.weights)


def test_subject_ids_are_disjoint_and_aligned_with_windows():
    subject_sets = {}
    for split in ("train", "val", "test"):
        windows = np.load(PROCESSED / f"{split}.npy", mmap_mode="r")
        labels = np.load(PROCESSED / f"{split}_labels.npy", mmap_mode="r")
        subject_ids = np.load(PROCESSED / f"{split}_subject_ids.npy", allow_pickle=True)
        assert len(windows) == len(labels) == len(subject_ids)
        subject_sets[split] = set(subject_ids.astype(str))

    assert subject_sets["train"].isdisjoint(subject_sets["val"])
    assert subject_sets["train"].isdisjoint(subject_sets["test"])
    assert subject_sets["val"].isdisjoint(subject_sets["test"])


def test_finetune_grid_and_selection_tie_breaking():
    configs = fine_tune_config_grid()
    assert len(configs) == 12
    assert {config["learning_rate"] for config in configs} == {
        1e-4,
        1e-3,
        1e-2,
        1e-1,
    }
    assert {config["epochs"] for config in configs} == {1, 5, 20}

    equal_f2 = [
        {
            "learning_rate": 1e-2,
            "epochs": 5,
            "validation_mean_f2": 0.98,
            "validation_mean_false_alarms_per_hour": 1.0,
        },
        {
            "learning_rate": 1e-1,
            "epochs": 1,
            "validation_mean_f2": 0.98,
            "validation_mean_false_alarms_per_hour": 1.0,
        },
        {
            "learning_rate": 1e-3,
            "epochs": 1,
            "validation_mean_f2": 0.98,
            "validation_mean_false_alarms_per_hour": 2.0,
        },
    ]
    selected = select_fine_tune_config(equal_f2)
    assert selected["epochs"] == 1
    assert selected["learning_rate"] == 1e-1
    lower_lr_tie = [
        {
            "learning_rate": 1e-2,
            "epochs": 1,
            "validation_mean_f2": 0.98,
            "validation_mean_false_alarms_per_hour": 1.0,
        },
        {
            "learning_rate": 1e-3,
            "epochs": 1,
            "validation_mean_f2": 0.98,
            "validation_mean_false_alarms_per_hour": 1.0,
        },
    ]
    assert select_fine_tune_config(lower_lr_tie)["learning_rate"] == 1e-3

    higher_f2 = {
        "learning_rate": 1e-1,
        "epochs": 20,
        "validation_mean_f2": 0.981,
        "validation_mean_false_alarms_per_hour": 100.0,
    }
    assert select_fine_tune_config([*equal_f2, higher_f2]) is higher_f2


def test_parametric_user_profiles_are_seeded_and_shape_preserving():
    windows = np.ones((4, 64, 6), dtype=np.float32)
    simulator = UserSimulator(seed=19)
    low_first = simulator.transform(windows, "Low-amp", user_id="SA02", seed=7)
    low_repeat = simulator.transform(windows, "Low-amp", user_id="SA02", seed=7)
    assert np.array_equal(low_first, low_repeat)
    assert low_first.shape == windows.shape
    assert np.allclose(low_first, 0.6, atol=1e-6)
    assert set(USER_PROFILES) == {"Normal", "Low-amp", "High-amp", "Rotated", "Noisy", "Slow"}


def test_generator_backend_is_class_conditional_and_shape_checked():
    simulator = UserSimulator(
        backend="generator",
        seed=11,
        generator=lambda labels, rng: np.broadcast_to(
            labels[:, None, None].astype(np.float32), (len(labels), 64, 6)
        ).copy(),
    )
    generated = simulator.generate(np.asarray([0, 1]), seed=5)
    assert generated.shape == (2, 64, 6)
    assert np.all(generated[0] == 0.0)
    assert np.all(generated[1] == 1.0)
    try:
        simulator.transform(np.zeros((1, 64, 6), dtype=np.float32), "Normal")
    except ValueError as exc:
        assert "generate(labels)" in str(exc)
    else:
        raise AssertionError("Generative backend accepted unlabeled transform input.")


def test_calibration_is_adl_only_disjoint_from_query_and_supports_zero():
    windows = np.arange(12 * 64 * 6, dtype=np.float32).reshape(12, 64, 6)
    labels = np.asarray([0] * 8 + [1] * 4, dtype=np.int32)
    calibration, calibration_labels, query, query_labels = split_user_calibration(
        windows, labels, max_calibration=5, seed=5
    )
    assert len(calibration) == 5
    assert np.all(calibration_labels == 0)
    assert len(query) == 7
    assert set(query_labels) == {0, 1}
    assert set(map(tuple, calibration[:, 0, :])).isdisjoint(map(tuple, query[:, 0, :]))

    empty, empty_labels, _, _ = split_user_calibration(windows, labels, max_calibration=0)
    assert empty.shape == (0, 64, 6)
    assert empty_labels.shape == (0,)

    adl_only, adl_only_labels = split_user_calibration(
        windows[:8], np.zeros(8, dtype=np.int32), max_calibration=3, allow_adl_only=True
    )[0:2]
    assert adl_only.shape == (3, 64, 6)
    assert np.all(adl_only_labels == 0)


def test_leakage_guard_rejects_generator_data_outside_train_subjects():
    assert_subject_disjoint(
        np.asarray(["train"]), np.asarray(["val"]), np.asarray(["test"]),
        generator_subject_ids=np.asarray(["train"]),
    )
    try:
        assert_subject_disjoint(
            np.asarray(["train"]), np.asarray(["val"]), np.asarray(["test"]),
            generator_subject_ids=np.asarray(["test"]),
        )
    except ValueError as exc:
        message = str(exc).lower()
        assert "held-out" in message or "train-subject" in message
    else:
        raise AssertionError("Generator subject leakage was not rejected.")


def test_generator_training_split_rejects_held_out_subjects():
    validate_generator_training_split(
        np.asarray(["train-a", "train-b"]),
        np.asarray(["train-a"]),
        np.asarray(["val-a", "test-a"]),
    )
    try:
        validate_generator_training_split(
            np.asarray(["train-a", "test-a"]),
            np.asarray(["test-a"]),
            np.asarray(["test-a"]),
        )
    except ValueError as exc:
        message = str(exc).lower()
        assert "held-out" in message or "train-subject" in message
    else:
        raise AssertionError("TimeVAE accepted held-out subject windows.")


def test_conditional_timevae_trains_only_on_provided_train_subject_rows():
    generator = ConditionalTimeVAE(latent_dim=4, seed=9)
    windows = np.random.default_rng(9).normal(size=(4, 64, 6)).astype(np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
    history = generator.fit(
        windows,
        labels,
        np.asarray(["train"] * 4),
        held_out_subject_ids=np.asarray(["val", "test"]),
        epochs=1,
        batch_size=2,
    )
    generated = generator.generate(labels, seed=9)
    assert len(history) == 1
    assert np.isfinite(history[0]["loss"])
    assert generated.shape == (4, 64, 6)
    assert np.isfinite(generated).all()


def test_timevae_quality_check_saves_distribution_and_tstr_artifacts(tmp_path):
    rng = np.random.default_rng(14)
    real = rng.normal(size=(8, 64, 6)).astype(np.float32)
    synthetic = rng.normal(size=(4, 64, 6)).astype(np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
    report = evaluate_generator_quality(
        real,
        synthetic,
        labels,
        real[:4],
        labels,
        tmp_path,
    )
    assert set(report) == {"channel_stats", "train_synthetic_test_real"}
    for filename in (
        "timevae_channel_stats.csv",
        "timevae_smv_histogram.csv",
        "timevae_smv_histogram.png",
        "timevae_pca.csv",
        "timevae_pca.png",
        "timevae_train_synthetic_test_real.csv",
    ):
        assert (tmp_path / filename).exists()


def test_false_alarm_rate_uses_preprocessed_stride_at_20hz():
    labels = np.asarray([0] * 10 + [1] * 2)
    probabilities = np.asarray([0.9] + [0.1] * 9 + [0.9, 0.1])
    metrics = calculate_metrics(labels, probabilities, threshold=0.5)
    assert metrics["false_positive_windows"] == 1.0
    assert metrics["false_alarms_per_hour"] == 450.0
    adl_only_metrics = calculate_metrics(np.zeros(3), np.asarray([0.1, 0.2, 0.9]), 0.5)
    assert np.isnan(adl_only_metrics["f1"])
    assert adl_only_metrics["fall_windows"] == 0.0


def test_wilcoxon_pairs_canonical_film_against_baselines_only(tmp_path):
    rows = []
    for subject, film, baseline in (
        ("s1", 0.8, 0.6),
        ("s2", 0.7, 0.65),
        ("s3", 0.9, 0.8),
    ):
        for model, f1 in (
            ("FiLM (canonical threshold)", film),
            ("FiLM (validation-retuned threshold)", film - 0.01),
            ("Global", baseline),
            ("Last-layer fine-tune (ADL-only)", baseline - 0.02),
        ):
            rows.append({"evaluation": "real_heldout", "subject_id": subject, "model": model, "f1": f1})
    result = paired_wilcoxon(pd.DataFrame(rows), tmp_path)
    assert len(result) == 2
    assert set(result["comparison"]) == {
        "FiLM vs Global",
        "FiLM vs Last-layer fine-tune (ADL-only)",
    }


def test_baseline_fine_tuning_runs_without_changing_source_model():
    import tensorflow as tf

    source = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(2,)),
            tf.keras.layers.Dense(1, activation="sigmoid", name="dense_1"),
        ]
    )
    source(np.ones((1, 2), dtype=np.float32))
    source.trainable = False
    original_weights = [weight.copy() for weight in source.get_weights()]
    calibration = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int32)
    adapted = fine_tune_baseline(
        source, calibration, labels, full_model=False, epochs=1
    )
    predictions = predict_model_probabilities(adapted, calibration)
    assert predictions.shape == (2,)
    assert all(
        np.array_equal(before, after)
        for before, after in zip(original_weights, source.get_weights())
    )


def test_film_identity_initialization_matches_frozen_model_for_k_zero_and_nonzero():
    import tensorflow as tf

    components = _load_components()
    encoder = UserEncoder(channels=64)
    encoder(tf.zeros((0, *components.feature_shape), dtype=tf.float32))
    query = np.load(PROCESSED / "test.npy", mmap_mode="r")[:24]
    expected = components.original.predict(query, verbose=0).reshape(-1)

    for calibration in (np.empty((0, 64, 6), dtype=np.float32), query[:3]):
        actual, gamma, beta = film_probabilities(components, encoder, query, calibration)
        assert np.allclose(gamma, 1.0)
        assert np.allclose(beta, 0.0)
        assert np.allclose(actual, expected, atol=1e-5, rtol=0.0)


def test_user_encoder_k_zero_path_returns_finite_identity_parameters():
    import tensorflow as tf

    encoder = UserEncoder(channels=64)
    gamma, beta = encoder(tf.zeros((0, 16, 64), dtype=tf.float32))
    assert gamma.shape == (64,)
    assert beta.shape == (64,)
    assert np.isfinite(gamma.numpy()).all()
    assert np.isfinite(beta.numpy()).all()
    assert np.allclose(gamma.numpy(), 1.0)
    assert np.allclose(beta.numpy(), 0.0)
    assert encoder.z0.trainable


def test_meta_training_updates_encoder_only_with_disjoint_train_and_validation_subjects():
    components = _load_components()
    train_x = np.load(PROCESSED / "train.npy", mmap_mode="r")
    train_y = np.load(PROCESSED / "train_labels.npy", mmap_mode="r")
    train_subjects = np.load(PROCESSED / "train_subject_ids.npy", allow_pickle=True).astype(str)
    val_x = np.load(PROCESSED / "val.npy", mmap_mode="r")
    val_y = np.load(PROCESSED / "val_labels.npy", mmap_mode="r")
    val_subjects = np.load(PROCESSED / "val_subject_ids.npy", allow_pickle=True).astype(str)

    def two_class_rows(windows, labels, subjects):
        for subject in np.unique(subjects):
            indices = np.flatnonzero(subjects == subject)
            adl = indices[labels[indices] == 0][:2]
            falls = indices[labels[indices] == 1][:2]
            if len(adl) == len(falls) == 2:
                selected = np.concatenate((adl, falls))
                return windows[selected], labels[selected], subjects[selected]
        raise AssertionError("No subject with both classes was available.")

    train_x, train_y, train_subjects = two_class_rows(train_x, train_y, train_subjects)
    val_x, val_y, val_subjects = two_class_rows(val_x, val_y, val_subjects)
    encoder, history = train_user_encoder(
        components,
        train_x,
        train_y,
        train_subjects,
        val_x,
        val_y,
        val_subjects,
        UserSimulator(seed=3),
        seed=3,
        epochs=1,
        episodes_per_epoch=1,
        validation_episodes=1,
        query_adl_windows=2,
    )
    assert len(history) == 1
    assert encoder.trainable_variables
    assert np.any(encoder.affine.kernel.numpy() != 0.0) or np.any(
        encoder.affine.bias.numpy() != 0.0
    )
    assert not components.extractor.trainable
    assert not components.head.trainable
