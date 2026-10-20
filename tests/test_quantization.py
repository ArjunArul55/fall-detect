"""Unit tests for TFLite quantization, model loading, tensor shapes, and parity (Task P5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from src.keras_model_loading import load_keras_model
from scripts.train_cnn_lstm_baseline import build_model

ROOT = Path(__file__).resolve().parents[1]
QUANT_DIR = ROOT / "results" / "quantization" / "run_v1"
DEPLOYED_MODEL_PATH = QUANT_DIR / "deployed_model.tflite"
METADATA_PATH = QUANT_DIR / "deployed_model_meta.json"
HEADER_PATH = QUANT_DIR / "model_data.h"
KERAS_MODEL_PATH = ROOT / "results" / "cnn_lstm_baseline" / "cnn_lstm_best.keras"
DATA_DIR = ROOT / "data" / "processed"


def test_deployed_model_file_exists():
    assert DEPLOYED_MODEL_PATH.is_file(), f"Missing deployed TFLite model at {DEPLOYED_MODEL_PATH}"
    assert DEPLOYED_MODEL_PATH.stat().st_size > 50_000, "Deployed model file is unexpectedly small"


def test_deployed_model_loads_in_interpreter():
    interpreter = tf.lite.Interpreter(model_path=str(DEPLOYED_MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    assert len(input_details) == 1, "Expected single input tensor"
    assert len(output_details) == 1, "Expected single output tensor"


def test_deployed_model_tensor_shapes():
    interpreter = tf.lite.Interpreter(model_path=str(DEPLOYED_MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Input tensor shape should match window dimensions (1, 64, 6)
    assert tuple(input_details["shape"]) == (1, 64, 6)
    # Output tensor shape should match binary classification (1, 1) or (1,)
    assert tuple(output_details["shape"]) in ((1, 1), (1,))


def test_50_window_parity_vs_keras():
    assert KERAS_MODEL_PATH.is_file(), f"Missing Keras model at {KERAS_MODEL_PATH}"
    keras_model = load_keras_model(tf, KERAS_MODEL_PATH, build_model)

    test_path = DATA_DIR / "test.npy"
    if test_path.is_file():
        test_x = np.load(test_path)[:50].astype(np.float32)
    else:
        # Fallback to deterministic pseudo-windows if test.npy is absent
        rng = np.random.default_rng(42)
        test_x = rng.standard_normal((50, 64, 6)).astype(np.float32)

    keras_preds = keras_model.predict(test_x, verbose=0).ravel()

    interpreter = tf.lite.Interpreter(model_path=str(DEPLOYED_MODEL_PATH))
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    tflite_preds = []
    for i in range(len(test_x)):
        interpreter.set_tensor(in_idx, test_x[i : i + 1])
        interpreter.invoke()
        tflite_preds.append(interpreter.get_tensor(out_idx).item())
    tflite_preds = np.array(tflite_preds, dtype=np.float32)

    abs_diff = np.abs(keras_preds - tflite_preds)
    assert float(np.max(abs_diff)) < 0.05, f"Max probability diff {np.max(abs_diff)} exceeds tolerance 0.05"
    assert float(np.mean(abs_diff)) < 0.01, f"Mean probability diff {np.mean(abs_diff)} exceeds tolerance 0.01"

    # Read operating threshold from metadata if available
    threshold = 0.10
    if METADATA_PATH.is_file():
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        threshold = float(meta.get("operating_threshold", 0.10))

    class_agreement = np.mean((keras_preds >= threshold) == (tflite_preds >= threshold))
    assert class_agreement >= 0.98, f"Class agreement {class_agreement:.2%} below 98%"


def test_deployed_model_metadata_schema():
    assert METADATA_PATH.is_file(), f"Missing deployed metadata at {METADATA_PATH}"
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    required_keys = [
        "variant",
        "threshold",
        "input_scale",
        "input_zero_point",
        "output_scale",
        "output_zero_point",
        "window_shape",
        "stride",
        "scaler_path",
        "git_commit",
    ]
    for key in required_keys:
        assert key in meta, f"Metadata missing required key: {key}"

    assert list(meta["window_shape"]) == [64, 6]
    assert meta["stride"] == 16
    assert 0.0 <= meta["threshold"] <= 1.0


def test_c_header_structure():
    assert HEADER_PATH.is_file(), f"Missing C header at {HEADER_PATH}"
    content = HEADER_PATH.read_text(encoding="utf-8")

    assert "const unsigned char g_model_data[]" in content
    assert "const unsigned int g_model_data_len" in content
    assert "aligned(16)" in content

    # Verify model length in header matches file size
    model_size = DEPLOYED_MODEL_PATH.stat().st_size
    assert f"g_model_data_len = {model_size};" in content
