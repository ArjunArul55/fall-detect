"""CLI tool to replay continuous IMU recordings through the StreamingDetector (Task P6).

Usage:
    python scripts/run_demo_cli.py --recording-id "SA02:F01_SA02_R02.txt" --speed 5 --n-consecutive 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.streaming import (
    AlertSink,
    ConsoleSink,
    SimulatedCaregiverSink,
    StreamingDetector,
    reconstruct_recording_stream,
)

DEFAULT_RECORDING = "SA02:F01_SA02_R02.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording-id",
        type=str,
        default=DEFAULT_RECORDING,
        help="Recording ID from test set (e.g. SA02:F01_SA02_R02.txt, SA02:D01_SA02_R01.txt).",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional path to custom 6-channel IMU CSV file (overrides --recording-id).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=5.0,
        help="Replay speed multiplier (e.g. 1.0 = real-time 20 Hz, 5.0 = 5x faster, 0 = instantaneous).",
    )
    parser.add_argument(
        "--n-consecutive",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Number of consecutive positive windows required to trigger an alert (default 1).",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=10.0,
        help="Alert cooldown period in seconds (default 10.0s).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Operating probability threshold (default read from deployed_model_meta.json).",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Apply data/processed/scaler.pkl standard scaling to incoming samples.",
    )
    return parser.parse_args()


def load_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    if args.csv_path is not None:
        p = Path(args.csv_path)
        if not p.is_file():
            raise FileNotFoundError(f"CSV file not found: {p}")
        df = pd.read_csv(p)
        imu_cols = ["acc1_x", "acc1_y", "acc1_z", "gyro_x", "gyro_y", "gyro_z"]
        if not all(c in df.columns for c in imu_cols):
            # Try positional first 6 columns
            samples = df.iloc[:, :6].to_numpy(dtype=np.float32)
        else:
            samples = df[imu_cols].to_numpy(dtype=np.float32)
        labels = np.zeros(len(samples), dtype=np.int32)
        meta = {
            "recording_id": p.name,
            "subject_id": "Custom",
            "activity_code": "Custom CSV",
            "num_windows": max(0, (len(samples) - 64) // 16 + 1),
            "num_samples": len(samples),
            "duration_seconds": len(samples) / 20.0,
            "fall_windows": 0,
            "is_fall_trial": False,
        }
        return samples, labels, meta

    return reconstruct_recording_stream(args.recording_id, split="test")


def main() -> None:
    args = parse_args()
    samples, labels, meta = load_data(args)

    print("=" * 70)
    print("AI WEARABLE FALL DETECTION - REAL-TIME STREAMING REPLAY")
    print("=" * 70)
    print(f"Recording ID       : {meta['recording_id']}")
    print(f"Subject / Activity : {meta['subject_id']} / {meta['activity_code']}")
    print(f"Total Samples      : {meta['num_samples']} ({meta['duration_seconds']:.2f} s @ 20 Hz)")
    print(f"Expected Windows   : {meta['num_windows']}")
    print(f"Ground Truth       : {'FALL' if meta['is_fall_trial'] else 'ADL / Normal'}")
    print(f"Replay Speed       : {'Max speed (no delay)' if args.speed <= 0 else f'{args.speed}x real-time'}")
    print(f"Alert Policy       : N_consecutive={args.n_consecutive}, Cooldown={args.cooldown}s")
    print("=" * 70)

    console_sink = ConsoleSink(prefix="[CARE-ALERT]")
    caregiver_sink = SimulatedCaregiverSink()

    detector = StreamingDetector(
        threshold=args.threshold,
        n_consecutive=args.n_consecutive,
        cooldown_seconds=args.cooldown,
        standardize=args.standardize,
        sinks=[console_sink, caregiver_sink],
    )

    print(f"Model Deployed     : {detector.metadata.get('variant', 'dynamic_range')}")
    print(f"Model Threshold    : {detector.threshold:.4f}")
    print("-" * 70)

    dt_delay = (1.0 / 20.0) / args.speed if args.speed > 0 else 0.0

    windows_evaluated = 0
    alerts_count = 0
    fall_positives = 0

    started = time.perf_counter()
    for idx, sample in enumerate(samples):
        timestamp = idx / 20.0
        result = detector.process_sample(sample, timestamp=timestamp)

        if result is not None:
            windows_evaluated += 1
            prob = result["fall_probability"]
            is_fall = result["is_fall"]
            is_alert = result["is_alert"]
            sev = result["severity"]

            if is_fall:
                fall_positives += 1
            if is_alert:
                alerts_count += 1

            status_str = "MONITORING"
            if is_alert:
                status_str = f"*** ALERT FIRED (Severity: {sev}) ***"
            elif is_fall:
                status_str = f"Positive (waiting for {args.n_consecutive} consec)"

            print(
                f"[t={timestamp:6.2f}s | sample={idx:4d}] Window {windows_evaluated:2d} | "
                f"Prob: {prob:6.4f} | {status_str}"
            )

        if dt_delay > 0:
            time.sleep(dt_delay)

    elapsed = time.perf_counter() - started

    print("=" * 70)
    print("REPLAY SUMMARY")
    print("=" * 70)
    print(f"Replay Elapsed Time: {elapsed:.2f} s")
    print(f"Windows Evaluated  : {windows_evaluated}")
    print(f"Positive Windows   : {fall_positives}")
    print(f"Alerts Triggered   : {alerts_count}")
    print(f"Caregiver Sink Log : {len(caregiver_sink.alerts)} notifications logged")
    for i, alert in enumerate(caregiver_sink.alerts, 1):
        print(f"  {i}. {alert['message']} at t={alert['timestamp']:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
