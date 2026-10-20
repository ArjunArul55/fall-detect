"""Unit tests for real-time streaming fall detector and alert policy (Task P6).

Tests cover:
1. Warm-up behavior: ring buffer outputs None for the first 63 samples, then evaluates at sample 64.
2. Parity test: streaming probabilities equal offline window-by-window TFLite predictions
   (max abs diff <= 1e-6) on 3 test recordings, with exact window count/positions match.
3. Alert policy: N-consecutive positives requirement (1, 2, 3) and cooldown suppression.
4. AlertSink interface: ConsoleSink and SimulatedCaregiverSink logging.
5. Streamlit app smoke test via AppTest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensorflow as tf

from src.streaming import (
    AlertSink,
    ConsoleSink,
    SimulatedCaregiverSink,
    StreamingDetector,
    reconstruct_recording_stream,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "results" / "quantization" / "run_v1" / "deployed_model.tflite"
META_PATH = ROOT / "results" / "quantization" / "run_v1" / "deployed_model_meta.json"
SCALER_PATH = ROOT / "data" / "processed" / "scaler.pkl"
APP_PATH = ROOT / "app" / "demo.py"


def test_warmup_no_output_before_64_samples():
    """Verify detector produces no predictions until exactly 64 samples have arrived."""
    detector = StreamingDetector(
        model_path=MODEL_PATH,
        meta_path=META_PATH,
        standardize=False,
    )

    rng = np.random.default_rng(101)
    # Stream 63 samples: all must return None
    for i in range(63):
        sample = rng.standard_normal(6).astype(np.float32)
        res = detector.process_sample(sample, timestamp=i * 0.05)
        assert res is None, f"Expected None at sample index {i}, got {res}"

    # 64th sample (index 63) must return first window
    sample_64 = rng.standard_normal(6).astype(np.float32)
    res_64 = detector.process_sample(sample_64, timestamp=63 * 0.05)
    assert res_64 is not None, "Expected window evaluation at sample index 63"
    assert "fall_probability" in res_64
    assert "is_fall" in res_64
    assert "is_alert" in res_64
    assert 0.0 <= res_64["fall_probability"] <= 1.0

    # Next 15 samples (indices 64 to 78) must return None
    for i in range(64, 79):
        sample = rng.standard_normal(6).astype(np.float32)
        res = detector.process_sample(sample, timestamp=i * 0.05)
        assert res is None, f"Expected None during stride at index {i}"

    # Sample 80 (index 79, stride=16) must return second window
    sample_80 = rng.standard_normal(6).astype(np.float32)
    res_80 = detector.process_sample(sample_80, timestamp=79 * 0.05)
    assert res_80 is not None, "Expected window evaluation at sample index 79"


@pytest.mark.parametrize(
    "recording_id",
    [
        "SA02:D07_SA02_R01.txt",
        "SA02:D15_SA02_R01.txt",
        "SA02:F01_SA02_R02.txt",
    ],
)
def test_offline_online_parity_3_recordings(recording_id: str):
    """Verify streaming probabilities match offline window-by-window TFLite inference (max abs diff <= 1e-6)."""
    samples, _, meta = reconstruct_recording_stream(recording_id, split="test")
    num_samples = len(samples)
    expected_windows = (num_samples - 64) // 16 + 1
    assert expected_windows > 0, f"Recording {recording_id} has insufficient samples: {num_samples}"

    # 1. Compute offline window-by-window TFLite predictions
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH), num_threads=1)
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    offline_probs: list[float] = []
    window_starts = list(range(0, num_samples - 64 + 1, 16))
    assert len(window_starts) == expected_windows

    for start in window_starts:
        w = samples[start : start + 64][np.newaxis, :, :].astype(np.float32)
        interpreter.set_tensor(in_idx, w)
        interpreter.invoke()
        p = float(interpreter.get_tensor(out_idx).item())
        offline_probs.append(p)

    # 2. Compute online streaming predictions
    sink = SimulatedCaregiverSink()
    detector = StreamingDetector(
        model_path=MODEL_PATH,
        meta_path=META_PATH,
        sinks=[sink],
        standardize=False,
    )

    online_probs: list[float] = []
    for idx, sample in enumerate(samples):
        res = detector.process_sample(sample, timestamp=idx * 0.05)
        if res is not None:
            online_probs.append(res["fall_probability"])

    # 3. Assert exact count and precision parity
    assert len(online_probs) == len(offline_probs) == expected_windows
    max_diff = float(np.max(np.abs(np.array(online_probs) - np.array(offline_probs))))
    assert max_diff <= 1e-6, f"Parity failure on {recording_id}: max diff {max_diff:.2e} > 1e-6"


def test_alert_policy_n_consecutive():
    """Verify N-consecutive positive window requirement before triggering alerts."""
    sink = SimulatedCaregiverSink()
    # Fixed threshold 0.50 for deterministic testing
    detector = StreamingDetector(
        model_path=MODEL_PATH,
        meta_path=META_PATH,
        threshold=0.50,
        n_consecutive=2,
        cooldown_seconds=10.0,
        sinks=[sink],
        standardize=False,
    )

    # Mock _infer_tflite to control probabilities
    # Sequence of probabilities: [0.80, 0.10, 0.85, 0.90]
    mock_probs = [0.80, 0.10, 0.85, 0.90]
    call_idx = 0

    def mock_infer(window: np.ndarray) -> tuple[float, str]:
        nonlocal call_idx
        p = mock_probs[call_idx]
        call_idx += 1
        return p, "Mild"

    detector._infer_tflite = mock_infer  # type: ignore

    # Provide 64 samples for first window (prob = 0.80)
    for i in range(64):
        res = detector.process_sample(np.zeros(6, dtype=np.float32), timestamp=i * 0.05)
    assert res is not None
    assert res["fall_probability"] == 0.80
    assert res["is_fall"] is True
    assert res["is_alert"] is False, "1st positive with n_consecutive=2 should not alert"
    assert len(sink.alerts) == 0

    # Provide 16 samples for second window (prob = 0.10, below threshold -> resets consecutive count)
    for i in range(64, 80):
        res = detector.process_sample(np.zeros(6, dtype=np.float32), timestamp=i * 0.05)
    assert res is not None
    assert res["fall_probability"] == 0.10
    assert res["is_fall"] is False
    assert res["is_alert"] is False
    assert len(sink.alerts) == 0

    # Provide 16 samples for third window (prob = 0.85 -> consecutive count = 1)
    for i in range(80, 96):
        res = detector.process_sample(np.zeros(6, dtype=np.float32), timestamp=i * 0.05)
    assert res is not None
    assert res["fall_probability"] == 0.85
    assert res["is_fall"] is True
    assert res["is_alert"] is False, "1st positive in new streak should not alert"
    assert len(sink.alerts) == 0

    # Provide 16 samples for fourth window (prob = 0.90 -> consecutive count = 2 -> alert!)
    for i in range(96, 112):
        res = detector.process_sample(np.zeros(6, dtype=np.float32), timestamp=i * 0.05)
    assert res is not None
    assert res["fall_probability"] == 0.90
    assert res["is_fall"] is True
    assert res["is_alert"] is True, "2nd consecutive positive should fire alert"
    assert len(sink.alerts) == 1
    assert sink.alerts[0]["fall_probability"] == 0.90


def test_alert_policy_cooldown():
    """Verify cooldown period suppresses repeated alerts during an ongoing fall event."""
    sink = SimulatedCaregiverSink()
    detector = StreamingDetector(
        model_path=MODEL_PATH,
        meta_path=META_PATH,
        threshold=0.50,
        n_consecutive=1,
        cooldown_seconds=10.0,
        sinks=[sink],
        standardize=False,
    )

    # Mock _infer_tflite to always return high probability
    detector._infer_tflite = lambda w: (0.95, "Severe")  # type: ignore

    # Window 1 at t = 3.20s -> Alert fires
    for i in range(64):
        res = detector.process_sample(np.zeros(6, dtype=np.float32), timestamp=i * 0.05)
    assert res is not None
    assert res["is_alert"] is True
    assert len(sink.alerts) == 1

    # Window 2 at t = 4.00s (within 10s cooldown) -> Alert suppressed
    for i in range(64, 80):
        res = detector.process_sample(np.zeros(6, dtype=np.float32), timestamp=i * 0.05)
    assert res is not None
    assert res["is_fall"] is True
    assert res["is_alert"] is False, "Alert within 10s cooldown must be suppressed"
    assert len(sink.alerts) == 1

    # Window 3 at t = 14.50s (> 10s after first alert at 3.15s) -> Alert fires again
    for i in range(80, 96):
        res = detector.process_sample(np.zeros(6, dtype=np.float32), timestamp=14.0 + (i - 80) * 0.05)
    assert res is not None
    assert res["is_alert"] is True, "Alert after cooldown expiration should fire"
    assert len(sink.alerts) == 2


def test_alert_sinks_interface(capsys):
    """Verify ConsoleSink and SimulatedCaregiverSink record alerts accurately."""
    console = ConsoleSink(prefix="[TEST-ALERT]")
    caregiver = SimulatedCaregiverSink()

    event = {
        "timestamp": 7.15,
        "fall_probability": 0.9542,
        "severity": "Severe",
        "is_fall": True,
        "is_alert": True,
    }

    console.send_alert(event)
    out = capsys.readouterr().out
    assert "[TEST-ALERT] Fall detected at 7.15s | Prob: 0.9542 | Severity: Severe" in out

    caregiver.send_alert(event)
    assert len(caregiver.alerts) == 1
    assert caregiver.alerts[0]["fall_probability"] == 0.9542
    assert "Fall detected, severity Severe, probability 0.9542" in caregiver.alerts[0]["message"]

    caregiver.clear()
    assert len(caregiver.alerts) == 0


def test_streamlit_app_smoke():
    """Verify Streamlit app initializes, loads model, and executes without unhandled errors."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit.testing.v1 is not available in current environment")

    assert APP_PATH.is_file(), f"Missing app file at {APP_PATH}"
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"Streamlit app raised exceptions: {at.exception}"
