"""Real-time streaming fall detection engine with ring buffer and alert policy (Task P6).

Implements:
- Ring buffer of 6-channel IMU samples at 20 Hz (window=64, stride=16).
- TFLite inference using deployed model (handling INT8 or float32 quantization).
- Optional standardization with scaler.pkl.
- Configurable alert policy: validation-selected threshold, N consecutive positives, cooldown.
- AlertSink interface with ConsoleSink and SimulatedCaregiverSink.
- Helpers to reconstruct test recording streams for offline-online parity verification.
"""

from __future__ import annotations

import abc
import json
import pickle
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "results" / "quantization" / "run_v1" / "deployed_model.tflite"
DEFAULT_META_PATH = ROOT / "results" / "quantization" / "run_v1" / "deployed_model_meta.json"
DEFAULT_SCALER_PATH = ROOT / "data" / "processed" / "scaler.pkl"
DATA_DIR = ROOT / "data" / "processed"


class AlertSink(abc.ABC):
    """Abstract interface for receiving fall alert notifications."""

    @abc.abstractmethod
    def send_alert(self, alert_event: dict[str, Any]) -> None:
        """Process a triggered alert event."""
        pass


class ConsoleSink(AlertSink):
    """Prints alert notifications to standard output."""

    def __init__(self, prefix: str = "[ALERT]") -> None:
        self.prefix = prefix

    def send_alert(self, alert_event: dict[str, Any]) -> None:
        t = alert_event.get("timestamp", 0.0)
        p = alert_event.get("fall_probability", 0.0)
        s = alert_event.get("severity", "N/A")
        print(f"{self.prefix} Fall detected at {t:.2f}s | Prob: {p:.4f} | Severity: {s}")


class SimulatedCaregiverSink(AlertSink):
    """In-memory alert log simulating a caregiver mobile app notification feed."""

    def __init__(self) -> None:
        self.alerts: list[dict[str, Any]] = []

    def send_alert(self, alert_event: dict[str, Any]) -> None:
        p = alert_event.get("fall_probability", 0.0)
        s = alert_event.get("severity", "N/A")
        t = alert_event.get("timestamp", 0.0)
        event = dict(alert_event)
        event["message"] = f"Fall detected, severity {s}, probability {p:.4f}"
        event["logged_at"] = time.time()
        self.alerts.append(event)

    def clear(self) -> None:
        self.alerts.clear()


class StreamingDetector:
    """Ring-buffered streaming detector processing single 6-channel IMU samples at 20 Hz."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        meta_path: str | Path | None = DEFAULT_META_PATH,
        scaler_path: str | Path | None = None,
        threshold: float | None = None,
        n_consecutive: int = 1,
        cooldown_seconds: float = 10.0,
        standardize: bool = False,
        sinks: list[AlertSink] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path) if meta_path else None
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.n_consecutive = max(1, int(n_consecutive))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.standardize = standardize
        self.sinks: list[AlertSink] = sinks if sinks is not None else []

        self.metadata: dict[str, Any] = {}
        if self.meta_path and self.meta_path.is_file():
            self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))

        if threshold is not None:
            self.threshold = float(threshold)
        else:
            self.threshold = float(self.metadata.get("threshold", 0.10))

        self.scaler: Any = None
        if self.standardize and self.scaler_path and self.scaler_path.is_file():
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        self._init_tflite()

        self.window_size = 64
        self.stride = 16
        self.sampling_hz = 20.0
        self.dt = 1.0 / self.sampling_hz

        self.reset()

    def _init_tflite(self) -> None:
        import tensorflow as tf

        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path), num_threads=1)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()

        self.is_int8_in = self.input_details["dtype"] == np.int8
        self.in_scale, self.in_zero_point = self.input_details.get("quantization", (0.0, 0))

        # Check for multi-task severity head
        self.has_severity_head = len(self.output_details) > 1

        self.bin_out_details = self.output_details[0]
        self.is_int8_bin_out = self.bin_out_details["dtype"] == np.int8
        self.bin_out_scale, self.bin_out_zero_point = self.bin_out_details.get("quantization", (0.0, 0))

        if self.has_severity_head:
            self.sev_out_details = self.output_details[1]
            self.is_int8_sev_out = self.sev_out_details["dtype"] == np.int8
            self.sev_out_scale, self.sev_out_zero_point = self.sev_out_details.get("quantization", (0.0, 0))
        else:
            self.sev_out_details = None

    def reset(self) -> None:
        """Reset internal streaming buffer and alert tracking state."""
        self.ring_buffer = np.zeros((self.window_size, 6), dtype=np.float32)
        self.sample_count = 0
        self.samples_since_last_inference = 0
        self.window_count = 0
        self.consecutive_positives = 0
        self.last_alert_timestamp = -1e9

    def process_sample(
        self,
        sample: np.ndarray | list[float],
        timestamp: float | None = None,
    ) -> dict[str, Any] | None:
        """Feed a single 6-channel sample (acc1_x, acc1_y, acc1_z, gyro_x, gyro_y, gyro_z) at 20 Hz.

        Returns:
            dict with prediction details when a decision stride completes (every 16 samples after 64),
            or None during warm-up and between strides.
        """
        arr = np.asarray(sample, dtype=np.float32)
        if arr.shape != (6,):
            raise ValueError(f"Expected 6-channel sample, found shape {arr.shape}")

        self.ring_buffer = np.roll(self.ring_buffer, -1, axis=0)
        self.ring_buffer[-1] = arr
        self.sample_count += 1

        curr_time = timestamp if timestamp is not None else (self.sample_count - 1) * self.dt

        # Warm-up phase: require 64 samples before first window evaluation
        if self.sample_count < self.window_size:
            return None

        if self.sample_count == self.window_size:
            self.samples_since_last_inference = 0
            return self._evaluate_window(curr_time)

        self.samples_since_last_inference += 1
        if self.samples_since_last_inference == self.stride:
            self.samples_since_last_inference = 0
            return self._evaluate_window(curr_time)

        return None

    def _infer_tflite(self, window: np.ndarray) -> tuple[float, str | None]:
        """Run raw inference on a (64, 6) window, returning (probability, severity)."""
        if self.is_int8_in:
            window_q = np.clip(
                np.round(window / self.in_scale + self.in_zero_point), -128, 127
            ).astype(np.int8)
            self.interpreter.set_tensor(self.input_details["index"], window_q[None, :])
        else:
            self.interpreter.set_tensor(self.input_details["index"], window[None, :])

        self.interpreter.invoke()

        # Binary output
        raw_bin = self.interpreter.get_tensor(self.bin_out_details["index"]).ravel()[0]
        if self.is_int8_bin_out:
            prob = float((float(raw_bin) - self.bin_out_zero_point) * self.bin_out_scale)
        else:
            prob = float(raw_bin)

        prob = max(0.0, min(1.0, prob))
        is_fall = prob >= self.threshold

        # Severity output (if multi-task head present)
        severity: str | None = None
        if self.has_severity_head and is_fall and self.sev_out_details is not None:
            raw_sev = self.interpreter.get_tensor(self.sev_out_details["index"]).ravel()
            if self.is_int8_sev_out:
                sev_probs = (raw_sev.astype(np.float32) - self.sev_out_zero_point) * self.sev_out_scale
            else:
                sev_probs = raw_sev.astype(np.float32)
            sev_idx = int(np.argmax(sev_probs))
            severity = "Severe" if sev_idx == 1 else "Mild"
        elif is_fall:
            severity = "N/A"

        return prob, severity

    def _evaluate_window(self, timestamp: float) -> dict[str, Any]:
        self.window_count += 1
        window = self.ring_buffer.copy()

        if self.standardize and self.scaler is not None:
            window = self.scaler.transform(window).astype(np.float32)

        prob, severity = self._infer_tflite(window)
        is_fall = prob >= self.threshold

        # Alert policy: consecutive positives & cooldown
        is_alert = False
        if is_fall:
            self.consecutive_positives += 1
        else:
            self.consecutive_positives = 0

        if self.consecutive_positives >= self.n_consecutive:
            time_since_last = timestamp - self.last_alert_timestamp
            if time_since_last >= self.cooldown_seconds:
                is_alert = True
                self.last_alert_timestamp = timestamp

        event = {
            "timestamp": float(timestamp),
            "sample_index": int(self.sample_count),
            "window_index": int(self.window_count),
            "fall_probability": float(prob),
            "is_fall": bool(is_fall),
            "is_alert": bool(is_alert),
            "severity": severity,
            "consecutive_positives": int(self.consecutive_positives),
        }

        if is_alert:
            for sink in self.sinks:
                sink.send_alert(event)

        return event


def reconstruct_recording_stream(
    recording_id: str,
    split: str = "test",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reconstruct a continuous 20 Hz sample stream and ground truth for a recorded trial.

    Returns:
        samples: shape (T, 6) 6-channel IMU array.
        sample_labels: shape (T,) binary ground truth at each timestep.
        meta: dictionary with recording metadata, window count, duration, subject, activity.
    """
    rec_path = DATA_DIR / f"{split}_recording_ids.npy"
    x_path = DATA_DIR / f"{split}.npy"
    y_path = DATA_DIR / f"{split}_labels.npy"
    sub_path = DATA_DIR / f"{split}_subject_ids.npy"

    if not rec_path.is_file() or not x_path.is_file():
        raise FileNotFoundError(f"Missing split arrays in {DATA_DIR}")

    rec_ids = np.load(rec_path, allow_pickle=True)
    mask = rec_ids == recording_id
    if not np.any(mask):
        raise ValueError(f"Recording ID '{recording_id}' not found in {split} split.")

    windows = np.load(x_path)[mask]
    window_labels = np.load(y_path)[mask] if y_path.is_file() else np.zeros(len(windows))
    subject_ids = np.load(sub_path, allow_pickle=True)[mask] if sub_path.is_file() else ["UNKNOWN"]

    n_windows = len(windows)
    # Assemble continuous sample sequence matching window overlap (window 0: 64, subsequent: +16 samples)
    samples_list = [windows[0]]
    for i in range(1, n_windows):
        samples_list.append(windows[i, -16:])
    samples = np.concatenate(samples_list, axis=0)

    # Reconstruct sample-level ground truth label
    sample_labels = np.zeros(len(samples), dtype=np.int32)
    for i, w_lbl in enumerate(window_labels):
        if w_lbl == 1:
            sample_labels[i * 16 : i * 16 + 64] = 1

    subject = str(subject_ids[0])
    activity_code = recording_id.split(":")[-1].split("_")[0] if ":" in recording_id else recording_id.split("_")[0]

    meta = {
        "recording_id": recording_id,
        "subject_id": subject,
        "activity_code": activity_code,
        "num_windows": n_windows,
        "num_samples": len(samples),
        "duration_seconds": len(samples) / 20.0,
        "fall_windows": int(np.sum(window_labels == 1)),
        "is_fall_trial": bool(np.any(window_labels == 1)),
    }

    return samples, sample_labels, meta
