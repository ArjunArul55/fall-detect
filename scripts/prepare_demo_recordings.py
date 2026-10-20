"""Generate 6 continuous 20 Hz IMU recordings for the Task P6 demo and evaluations.

Creates:
- 2 ADL trials: walking, sitting.
- 2 Hard-negative ADL trials: stumbling, jumping.
- 2 Fall trials: Mild Fall (K=2 severity 0), Severe Fall (K=2 severity 1).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "demo_recordings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

T = 300  # 15.0 seconds at 20 Hz -> exactly 15 windows (64 samples, stride 16)
t = np.linspace(0, (T - 1) / 20.0, T)
gravity = np.array([0.0, -0.98, 0.1], dtype=np.float32)

rng = np.random.RandomState(42)

recordings = {}

# 1. ADL Walking (D01, Subject SA02)
freq = 1.8
acc_walk = gravity + 0.30 * np.sin(2 * np.pi * freq * t[:, None]) + rng.normal(0, 0.08, (T, 3))
gyro_walk = 0.50 * np.cos(2 * np.pi * freq * t[:, None]) + rng.normal(0, 0.08, (T, 3))
recordings["SA02_D01_Walking"] = {
    "samples": np.column_stack([acc_walk, gyro_walk]).astype(np.float32),
    "subject_id": "SA02",
    "activity_code": "D01",
    "activity_name": "Walking (ADL)",
    "category": "ADL",
    "is_fall": False,
    "expected_alert": False,
    "fall_impact_time_s": None,
}

# 2. ADL Sitting (D07, Subject SA02)
acc_sit = np.tile(gravity, (T, 1))
gyro_sit = np.zeros((T, 3), dtype=np.float32)
sit_transition = 1.0 / (1.0 + np.exp(-(t - 6.0) / 0.5))
acc_sit[:, 1] += 0.4 * sit_transition
acc_sit += rng.normal(0, 0.04, (T, 3))
gyro_sit[:, 0] = 0.8 * np.exp(-0.5 * ((t - 6.0) / 0.6) ** 2) + rng.normal(0, 0.03, (T,))
gyro_sit += rng.normal(0, 0.02, (T, 3))
recordings["SA02_D07_Sitting"] = {
    "samples": np.column_stack([acc_sit, gyro_sit]).astype(np.float32),
    "subject_id": "SA02",
    "activity_code": "D07",
    "activity_name": "Standing to sitting (ADL)",
    "category": "ADL",
    "is_fall": False,
    "expected_alert": False,
    "fall_impact_time_s": None,
}

# 3. Hard-Negative Stumble (D12, Subject SA02)
acc_stumble = gravity + 0.25 * np.sin(2 * np.pi * 1.5 * t[:, None]) + rng.normal(0, 0.06, (T, 3))
gyro_stumble = 0.40 * np.cos(2 * np.pi * 1.5 * t[:, None]) + rng.normal(0, 0.06, (T, 3))
spike_stumble = 3.6 * np.exp(-0.5 * ((t - 7.0) / 0.20) ** 2)
acc_stumble[:, 0] += spike_stumble * 0.8
acc_stumble[:, 1] -= spike_stumble * 0.5
gyro_stumble[:, 2] += 4.5 * np.exp(-0.5 * ((t - 7.0) / 0.25) ** 2)
recordings["SA02_D12_Stumble"] = {
    "samples": np.column_stack([acc_stumble, gyro_stumble]).astype(np.float32),
    "subject_id": "SA02",
    "activity_code": "D12",
    "activity_name": "Stumbling / tripping (Hard-Negative ADL)",
    "category": "Hard-Negative ADL",
    "is_fall": False,
    "expected_alert": False,
    "fall_impact_time_s": None,
}

# 4. Hard-Negative Jump (D15, Subject SA02)
acc_jump = np.tile(gravity, (T, 1)) + rng.normal(0, 0.05, (T, 3))
gyro_jump = rng.normal(0, 0.04, (T, 3))
takeoff = -0.7 * np.exp(-0.5 * ((t - 6.0) / 0.2) ** 2)
landing = 4.2 * np.exp(-0.5 * ((t - 6.5) / 0.15) ** 2)
acc_jump[:, 1] += (takeoff + landing).astype(np.float32)
gyro_jump[:, 0] += (1.8 * np.exp(-0.5 * ((t - 6.5) / 0.2) ** 2)).astype(np.float32)
recordings["SA02_D15_Jump"] = {
    "samples": np.column_stack([acc_jump, gyro_jump]).astype(np.float32),
    "subject_id": "SA02",
    "activity_code": "D15",
    "activity_name": "Jumping (Hard-Negative ADL)",
    "category": "Hard-Negative ADL",
    "is_fall": False,
    "expected_alert": False,
    "fall_impact_time_s": None,
}

# 5. Mild Fall (F01, Subject SA02, K=2 Severity: 0)
acc_mild = np.tile(gravity, (T, 1)) + 0.20 * np.sin(2 * np.pi * 1.5 * t[:, None])
gyro_mild = np.tile(0.30 * np.cos(2 * np.pi * 1.5 * t[:, None]), (1, 3))
freefall = -0.8 * np.exp(-0.5 * ((t - 6.9) / 0.18) ** 2)
impact_mild = 5.8 * np.exp(-0.5 * ((t - 7.2) / 0.22) ** 2)
post_mask = t > 7.4
acc_mild[post_mask] = np.array([0.05, 0.0, -0.95], dtype=np.float32) + rng.normal(0, 0.02, (np.sum(post_mask), 3)).astype(np.float32)
gyro_mild[post_mask] = rng.normal(0, 0.02, (np.sum(post_mask), 3)).astype(np.float32)
acc_mild[:, 1] += (freefall + impact_mild).astype(np.float32)
gyro_mild[:, 0] += (7.5 * np.exp(-0.5 * ((t - 7.1) / 0.25) ** 2)).astype(np.float32)
acc_mild += rng.normal(0, 0.05, (T, 3)).astype(np.float32)
gyro_mild += rng.normal(0, 0.05, (T, 3)).astype(np.float32)
recordings["SA02_F01_MildFall"] = {
    "samples": np.column_stack([acc_mild, gyro_mild]).astype(np.float32),
    "subject_id": "SA02",
    "activity_code": "F01",
    "activity_name": "Fall forward on knees (Mild Fall)",
    "category": "Fall",
    "is_fall": True,
    "expected_alert": True,
    "severity": "Mild",
    "fall_impact_time_s": 7.2,
}

# 6. Severe Fall (F03, Subject SA02, K=2 Severity: 1)
acc_sev = np.tile(gravity, (T, 1)) + 0.25 * np.sin(2 * np.pi * 1.6 * t[:, None])
gyro_sev = np.tile(0.35 * np.cos(2 * np.pi * 1.6 * t[:, None]), (1, 3))
freefall_sev = -0.9 * np.exp(-0.5 * ((t - 6.7) / 0.18) ** 2)
impact_sev = 9.5 * np.exp(-0.5 * ((t - 7.0) / 0.18) ** 2)
post_sev = t > 7.3
acc_sev[post_sev] = np.array([-0.85, 0.05, 0.12], dtype=np.float32) + rng.normal(0, 0.02, (np.sum(post_sev), 3)).astype(np.float32)
gyro_sev[post_sev] = rng.normal(0, 0.02, (np.sum(post_sev), 3)).astype(np.float32)
acc_sev[:, 0] -= (impact_sev * 0.7).astype(np.float32)
acc_sev[:, 2] -= (impact_sev * 0.5).astype(np.float32)
gyro_sev[:, 0] += (10.5 * np.exp(-0.5 * ((t - 6.9) / 0.20) ** 2)).astype(np.float32)
gyro_sev[:, 2] += (8.5 * np.exp(-0.5 * ((t - 7.0) / 0.20) ** 2)).astype(np.float32)
acc_sev += rng.normal(0, 0.05, (T, 3)).astype(np.float32)
gyro_sev += rng.normal(0, 0.05, (T, 3)).astype(np.float32)
recordings["SA02_F03_SevereFall"] = {
    "samples": np.column_stack([acc_sev, gyro_sev]).astype(np.float32),
    "subject_id": "SA02",
    "activity_code": "F03",
    "activity_name": "Fall backward on buttocks (Severe Fall)",
    "category": "Fall",
    "is_fall": True,
    "expected_alert": True,
    "severity": "Severe",
    "fall_impact_time_s": 7.0,
}

manifest = {}
channels = ["acc1_x", "acc1_y", "acc1_z", "gyro_x", "gyro_y", "gyro_z"]
for key, item in recordings.items():
    csv_file = OUT_DIR / f"{key}.csv"
    df = pd.DataFrame(item["samples"], columns=channels)
    df.insert(0, "timestamp_s", np.round(t, 2))
    df.to_csv(csv_file, index=False)

    act_code = item["activity_code"]
    manifest[key] = {
        "recording_id": f"SA02:{act_code}_SA02_R01.txt",
        "file_name": f"{key}.csv",
        "subject_id": item["subject_id"],
        "activity_code": item["activity_code"],
        "activity_name": item["activity_name"],
        "category": item["category"],
        "is_fall": item["is_fall"],
        "expected_alert": item["expected_alert"],
        "severity": item.get("severity", "N/A"),
        "fall_impact_time_s": item.get("fall_impact_time_s"),
        "num_samples": T,
        "duration_s": 15.0,
        "sampling_hz": 20.0,
        "expected_windows": 15,
    }

manifest_path = OUT_DIR / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Generated 6 demo recordings and manifest at: {OUT_DIR}")
