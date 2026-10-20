"""Interactive Streamlit Edge Fall Detection Streaming Demo (Task P6).

Run with:
    streamlit run app/demo.py
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import streamlit as st

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.streaming import (
    ConsoleSink,
    SimulatedCaregiverSink,
    StreamingDetector,
)

DEMO_DIR = ROOT / "data" / "demo_recordings"
META_PATH = ROOT / "results" / "quantization" / "run_v1" / "deployed_model_meta.json"
MODEL_PATH = ROOT / "results" / "quantization" / "run_v1" / "deployed_model.tflite"

st.set_page_config(
    page_title="Wearable Fall Detection - Edge Demo",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title { font-size: 28px; font-weight: 700; color: #1E293B; margin-bottom: 0px; }
    .sub-title { font-size: 14px; color: #64748B; margin-bottom: 20px; }
    .badge-normal { background-color: #DEF7EC; color: #03543F; padding: 12px 20px; border-radius: 8px; font-size: 18px; font-weight: 600; border-left: 6px solid #31C48D; }
    .badge-fall { background-color: #FDE8E8; color: #9B1C1C; padding: 12px 20px; border-radius: 8px; font-size: 20px; font-weight: 700; border-left: 6px solid #F05252; animation: pulse 1s infinite; }
    .badge-warmup { background-color: #FEF08A; color: #713F12; padding: 12px 20px; border-radius: 8px; font-size: 16px; font-weight: 600; border-left: 6px solid #EAB308; }
    .metric-card { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_manifest() -> dict[str, Any]:
    manifest_file = DEMO_DIR / "manifest.json"
    if manifest_file.is_file():
        return json.loads(manifest_file.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def load_meta() -> dict[str, Any]:
    if META_PATH.is_file():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    st.markdown('<div class="main-title">🛡️ AI Wearable Fall Detection — Real-Time Edge TFLite Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Software-in-the-loop streaming demo running quantized TFLite inference over 20 Hz IMU ring buffer (Task P6).</div>', unsafe_allow_html=True)

    manifest = load_manifest()
    meta = load_meta()

    # Sidebar
    st.sidebar.header("🕹️ Stream Settings")

    source_type = st.sidebar.radio(
        "Input Data Source",
        ["Preloaded Test Recordings", "Upload Custom 20 Hz CSV"],
        index=0,
    )

    recording_df: pd.DataFrame | None = None
    rec_meta: dict[str, Any] = {}

    if source_type == "Preloaded Test Recordings":
        options = list(manifest.keys()) if manifest else []
        labels_map = {
            k: f"{manifest[k].get('category', 'Trial')}: {manifest[k].get('activity_name', k)}"
            for k in options
        }
        selected_key = st.sidebar.selectbox(
            "Select Test Trial",
            options,
            format_func=lambda x: labels_map.get(x, x),
            index=4 if len(options) > 4 else 0,
        )
        if selected_key:
            rec_meta = manifest[selected_key]
            csv_path = DEMO_DIR / rec_meta["file_name"]
            if csv_path.is_file():
                recording_df = pd.read_csv(csv_path)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload 6-Channel IMU CSV (acc1_x..z, gyro_x..z @ 20 Hz)", type=["csv"])
        if uploaded_file is not None:
            recording_df = pd.read_csv(uploaded_file)
            rec_meta = {
                "activity_name": uploaded_file.name,
                "category": "Custom Upload",
                "is_fall": False,
                "fall_impact_time_s": None,
                "num_samples": len(recording_df),
                "duration_s": len(recording_df) / 20.0,
            }

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Detection Policy")

    default_threshold = float(meta.get("threshold", 0.10))
    threshold = st.sidebar.slider(
        "Operating Threshold",
        min_value=0.01,
        max_value=0.99,
        value=default_threshold,
        step=0.01,
        help="Validation-selected threshold (Decision D2). Predictions >= threshold trigger fall state.",
    )

    n_consecutive = st.sidebar.selectbox(
        "N-Consecutive Positives",
        [1, 2, 3],
        index=0,
        help="Number of consecutive positive windows required before firing alert.",
    )

    cooldown = st.sidebar.slider(
        "Alert Cooldown (seconds)",
        min_value=1.0,
        max_value=30.0,
        value=10.0,
        step=1.0,
        help="Suppresses repeated notifications for N seconds after an alert fires.",
    )

    speed_choice = st.sidebar.select_slider(
        "Replay Speed",
        options=["1x (Real-time)", "5x (Fast)", "20x (Ultra)", "Max (Instant)"],
        value="Max (Instant)",
    )
    speed_map = {"1x (Real-time)": 1.0, "5x (Fast)": 5.0, "20x (Ultra)": 20.0, "Max (Instant)": 0.0}
    speed = speed_map[speed_choice]

    with st.sidebar.expander("📦 Deployed TFLite Model Info", expanded=False):
        st.write(f"**Variant**: `{meta.get('variant', 'dynamic_range')}`")
        st.write(f"**Size**: `{meta.get('size_kb', 111.0):.1f} KB`")
        st.write(f"**Latency**: `{meta.get('metrics', {}).get('latency_median_ms', 0.03):.3f} ms` / window")
        st.write(f"**Window Size**: 64 samples (3.2s @ 20 Hz)")
        st.write(f"**Stride**: 16 samples (0.8s update)")
        st.write(f"**Operating Threshold**: `{default_threshold:.2f}` (Validation-Selected)")
        st.write(f"**Git Commit**: `{meta.get('git_commit', 'UNKNOWN')}`")

    if recording_df is None:
        st.info("Please select or upload a continuous IMU recording to begin streaming.")
        return

    # Extract 6 IMU channels
    imu_cols = ["acc1_x", "acc1_y", "acc1_z", "gyro_x", "gyro_y", "gyro_z"]
    missing = [c for c in imu_cols if c not in recording_df.columns]
    if missing:
        st.error(f"Missing required columns in CSV: {missing}. Expected {imu_cols}")
        return

    raw_samples = recording_df[imu_cols].to_numpy(dtype=np.float32)
    n_samples = len(raw_samples)
    duration_s = n_samples / 20.0

    st.sidebar.markdown(f"**Loaded Trial**: {rec_meta.get('activity_name', 'Trial')}")
    st.sidebar.markdown(f"**Duration**: {duration_s:.1f}s ({n_samples} samples)")

    # Session State
    if "replay_running" not in st.session_state:
        st.session_state.replay_running = False

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    run_clicked = col_btn1.button("▶️ Run / Replay Stream", type="primary")
    reset_clicked = col_btn2.button("🔄 Reset")

    if reset_clicked:
        st.session_state.replay_running = False

    # Execute streaming inference
    if run_clicked or not st.session_state.replay_running:
        caregiver_sink = SimulatedCaregiverSink()
        console_sink = ConsoleSink()

        detector = StreamingDetector(
            threshold=threshold,
            n_consecutive=n_consecutive,
            cooldown_seconds=cooldown,
            standardize=False,
            sinks=[caregiver_sink, console_sink],
        )

        window_history: list[dict[str, Any]] = []
        status_history: list[dict[str, Any]] = []
        first_alert_time: float | None = None

        delay_dt = (1.0 / 20.0) / speed if speed > 0 else 0.0

        for i, s in enumerate(raw_samples):
            t_curr = i / 20.0
            res = detector.process_sample(s, timestamp=t_curr)
            if res is not None:
                window_history.append(res)
                if res["is_alert"] and first_alert_time is None:
                    first_alert_time = t_curr
            if delay_dt > 0 and speed < 20.0:
                time.sleep(delay_dt)

        total_windows = len(window_history)
        total_alerts = len(caregiver_sink.alerts)
        is_fall_ground_truth = rec_meta.get("is_fall", False)
        impact_t = rec_meta.get("fall_impact_time_s", None)

        false_alarms = total_alerts if not is_fall_ground_truth else 0
        latency_str = "N/A"
        if is_fall_ground_truth and first_alert_time is not None and impact_t is not None:
            latency_val = max(0.0, first_alert_time - impact_t)
            latency_str = f"{latency_val:.2f} s"

        # Live Status Banner
        if total_alerts > 0:
            st.markdown(
                f'<div class="badge-fall">🚨 FALL DETECTED! First Alert at t={first_alert_time:.2f}s | Caregiver Notified (Total Alerts: {total_alerts})</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="badge-normal">🟢 MONITORING (NORMAL) — No Fall Detected across {total_windows} decision windows.</div>',
                unsafe_allow_html=True,
            )

        st.write("")

        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Windows Processed", f"{total_windows}")
        m2.metric("Alerts Triggered", f"{total_alerts}", delta="Triggered" if total_alerts > 0 else "Clean")
        m3.metric("False Alarms (ADL)", f"{false_alarms}", delta="0 FA" if false_alarms == 0 else f"+{false_alarms} FA", delta_color="inverse")
        m4.metric("Impact-to-Alert Latency", latency_str)

        st.write("")

        # Charts Layout
        tab1, tab2, tab3 = st.tabs(["📈 Fall Probability Timeline", "🌊 6-Channel IMU Sensor Stream", "📬 Caregiver Notification Log"])

        with tab1:
            st.subheader("Window Fall Probability vs Threshold")
            if window_history:
                prob_df = pd.DataFrame([
                    {
                        "Time (s)": w["timestamp"],
                        "Fall Probability": w["fall_probability"],
                        "Threshold": threshold,
                        "Is Fall": w["is_fall"],
                        "Is Alert": w["is_alert"],
                    }
                    for w in window_history
                ]).set_index("Time (s)")

                st.line_chart(prob_df[["Fall Probability", "Threshold"]], color=["#EF4444", "#3B82F6"])
                if impact_t is not None:
                    st.caption(f"📍 Ground-truth fall impact occurred at **t = {impact_t:.2f} s**")
            else:
                st.write("No windows evaluated yet (requires at least 64 samples / 3.2s).")

        with tab2:
            st.subheader("Raw 6-Channel Sensor Stream (20 Hz)")
            imu_df = pd.DataFrame(raw_samples, columns=imu_cols)
            imu_df["Time (s)"] = np.round(np.arange(n_samples) / 20.0, 2)
            imu_df = imu_df.set_index("Time (s)")

            sub_acc, sub_gyro = st.columns(2)
            with sub_acc:
                st.markdown("**Accelerometer (ADXL345, g)**")
                st.line_chart(imu_df[["acc1_x", "acc1_y", "acc1_z"]])
            with sub_gyro:
                st.markdown("**Gyroscope (ITG3200, rad/s)**")
                st.line_chart(imu_df[["gyro_x", "gyro_y", "gyro_z"]])

        with tab3:
            st.subheader("Simulated Caregiver Alert Dispatch Log")
            if caregiver_sink.alerts:
                alert_df = pd.DataFrame([
                    {
                        "Alert #": idx + 1,
                        "Timestamp (s)": f"{a['timestamp']:.2f} s",
                        "Fall Probability": f"{a['fall_probability']:.4f}",
                        "Severity": a.get("severity", "N/A"),
                        "Message": a["message"],
                    }
                    for idx, a in enumerate(caregiver_sink.alerts)
                ])
                st.dataframe(alert_df, use_container_width=True, hide_index=True)
            else:
                st.write("No caregiver alerts dispatched. Model state remains normal monitoring.")


if __name__ == "__main__":
    main()
