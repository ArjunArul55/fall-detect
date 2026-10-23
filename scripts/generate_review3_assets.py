"""Generate formal engineering diagrams and figures for Review 3 presentation."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "presentation" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
NAVY = "#0F2942"
BLUE = "#1E40AF"
LIGHT_BLUE = "#E0E7FF"
SLATE = "#475569"
LIGHT_BG = "#F8FAFC"
BORDER = "#CBD5E1"
DARK_GRAY = "#1E293B"
GREEN = "#047857"
LIGHT_GREEN = "#D1FAE5"


def generate_system_architecture() -> None:
    """Generate Review 3 end-to-end system architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=DPI)
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")
    ax.axis("off")

    stages = [
        ("1. Data Acquisition", "6-Axis IMU (Acc + Gyro)\nSampled at 200 Hz"),
        ("2. Preprocessing", "Butterworth LPF (5 Hz)\n200 Hz -> 20 Hz Decimation\nStandardScaler (Train Fit)"),
        ("3. Sliding Windowing", "64 Samples (3.2 s)\nStride 16 (0.8 s step)\n75% Overlap Buffer"),
        ("4. Deep Backbone", "2x Conv1D + MaxPool\nLSTM (64, unrolled)\nGlobalAvgPool + Dense"),
        ("5. Multi-Task Heads", "Binary Fall Head (Sigmoid)\nSeverity Head (Softmax)\nMasked Loss Training"),
        ("6. Edge Quantization", "Dynamic-Range TFLite\n111.0 KB (2.12x comp)\n0.026 ms Edge Latency"),
        ("7. Alert Logic", "Threshold tau = 0.25\nN-Consecutive Filter (1-3)\n10.0 s Alert Cooldown"),
        ("8. Caregiver Dispatch", "Streamlit Live Dashboard\nConsole / Remote Sinks\nAudio-Visual Alert"),
    ]

    n = len(stages)
    box_w = 1.35
    box_h = 2.4
    spacing = 0.35
    total_w = n * box_w + (n - 1) * spacing
    start_x = (14.0 - total_w) / 2.0
    start_y = 1.0

    for i, (title, desc) in enumerate(stages):
        x = start_x + i * (box_w + spacing)
        y = start_y

        is_highlight = i in (3, 4, 5)
        bg_col = LIGHT_GREEN if is_highlight else LIGHT_BG
        border_col = GREEN if is_highlight else BORDER
        title_col = GREEN if is_highlight else NAVY

        rect = patches.FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=bg_col,
            edgecolor=border_col,
            linewidth=1.8 if is_highlight else 1.2,
        )
        ax.add_patch(rect)

        # Step badge header
        header_rect = patches.FancyBboxPatch(
            (x, y + box_h - 0.55), box_w, 0.55,
            boxstyle="round,pad=0.04,rounding_size=0.1",
            facecolor=title_col,
            edgecolor="none",
        )
        ax.add_patch(header_rect)

        ax.text(
            x + box_w / 2.0, y + box_h - 0.28,
            title,
            ha="center", va="center",
            fontsize=8.5, fontweight="bold",
            color="#FFFFFF"
        )

        ax.text(
            x + box_w / 2.0, y + (box_h - 0.55) / 2.0,
            desc,
            ha="center", va="center",
            fontsize=8.0,
            color=DARK_GRAY,
            linespacing=1.35
        )

        # Arrow to next stage
        if i < n - 1:
            ax.annotate(
                "",
                xy=(x + box_w + spacing, y + box_h / 2.0),
                xytext=(x + box_w, y + box_h / 2.0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=BLUE,
                    lw=1.8,
                    mutation_scale=14,
                ),
            )

    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 4.5)
    plt.tight_layout()
    out_path = OUT_DIR / "review3_system_architecture.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_multitask_architecture() -> None:
    """Generate Review 3 Multi-Task CNN-LSTM detailed architecture diagram."""
    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=DPI)
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")
    ax.axis("off")

    layers = [
        ("Input Window", "64 x 6\n(3.2 s @ 20 Hz)", "#0F2942"),
        ("Conv1D Block 1", "32 filters, k=3\nBatchNorm + ReLU\nMaxPool1D (pool=2)", "#1E40AF"),
        ("Conv1D Block 2", "64 filters, k=3\nBatchNorm + ReLU\nMaxPool1D (pool=2)", "#1E40AF"),
        ("Recurrent LSTM", "64 Units\n(unroll=True)\nreturn_seq=True", "#1E40AF"),
        ("Global Pooling", "GlobalAveragePool1D\nAggregates 16 steps\nFeature Dim = 64", "#334155"),
        ("Dense Trunk", "Dense (32, ReLU)\nDropout (0.3)\nDense Feature Vector", "#334155"),
    ]

    # Draw trunk horizontally
    x_positions = np.linspace(0.8, 8.5, len(layers))
    y_trunk = 2.4
    b_w = 1.3
    b_h = 1.7

    for idx, ((title, desc, col), x) in enumerate(zip(layers, x_positions)):
        rect = patches.FancyBboxPatch(
            (x - b_w / 2.0, y_trunk - b_h / 2.0), b_w, b_h,
            boxstyle="round,pad=0.06,rounding_size=0.1",
            facecolor=LIGHT_BG,
            edgecolor=col,
            linewidth=1.6,
        )
        ax.add_patch(rect)

        # Header bar
        hbar = patches.FancyBboxPatch(
            (x - b_w / 2.0, y_trunk + b_h / 2.0 - 0.45), b_w, 0.45,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=col,
            edgecolor="none",
        )
        ax.add_patch(hbar)

        ax.text(x, y_trunk + b_h / 2.0 - 0.22, title, ha="center", va="center", fontsize=8.0, fontweight="bold", color="#FFFFFF")
        ax.text(x, y_trunk - 0.2, desc, ha="center", va="center", fontsize=7.5, color=DARK_GRAY, linespacing=1.3)

        if idx < len(layers) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - b_w / 2.0, y_trunk),
                xytext=(x + b_w / 2.0, y_trunk),
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.6, mutation_scale=12),
            )

    # Fork from Dense Trunk into two heads
    trunk_out_x = x_positions[-1] + b_w / 2.0
    head_x = 10.8
    head_w = 1.8
    head_h = 1.4

    # Binary Head (Top)
    y_bin = 3.4
    rect_bin = patches.FancyBboxPatch(
        (head_x - head_w / 2.0, y_bin - head_h / 2.0), head_w, head_h,
        boxstyle="round,pad=0.06,rounding_size=0.1",
        facecolor="#F0FDF4",
        edgecolor=GREEN,
        linewidth=2.0,
    )
    ax.add_patch(rect_bin)
    h_bin = patches.FancyBboxPatch(
        (head_x - head_w / 2.0, y_bin + head_h / 2.0 - 0.42), head_w, 0.42,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor=GREEN,
        edgecolor="none",
    )
    ax.add_patch(h_bin)
    ax.text(head_x, y_bin + head_h / 2.0 - 0.21, "Binary Fall Head", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#FFFFFF")
    ax.text(head_x, y_bin - 0.2, "Dense(1, Sigmoid)\np_fall in [0, 1]\nLoss: BCE (All Windows)", ha="center", va="center", fontsize=7.5, color=DARK_GRAY, linespacing=1.25)

    # Severity Head (Bottom)
    y_sev = 1.4
    rect_sev = patches.FancyBboxPatch(
        (head_x - head_w / 2.0, y_sev - head_h / 2.0), head_w, head_h,
        boxstyle="round,pad=0.06,rounding_size=0.1",
        facecolor="#EFF6FF",
        edgecolor=BLUE,
        linewidth=2.0,
    )
    ax.add_patch(rect_sev)
    h_sev = patches.FancyBboxPatch(
        (head_x - head_w / 2.0, y_sev + head_h / 2.0 - 0.42), head_w, 0.42,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor=BLUE,
        edgecolor="none",
    )
    ax.add_patch(h_sev)
    ax.text(head_x, y_sev + head_h / 2.0 - 0.21, "Severity Head (K=2)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#FFFFFF")
    ax.text(head_x, y_sev - 0.2, "Dense(2, Softmax)\n[p_mild, p_severe]\nLoss: Masked CCE (Falls Only)", ha="center", va="center", fontsize=7.5, color=DARK_GRAY, linespacing=1.25)

    # Branching arrows
    ax.annotate(
        "",
        xy=(head_x - head_w / 2.0, y_bin),
        xytext=(trunk_out_x, y_trunk),
        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.8, mutation_scale=12),
    )
    ax.annotate(
        "",
        xy=(head_x - head_w / 2.0, y_sev),
        xytext=(trunk_out_x, y_trunk),
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.8, mutation_scale=12),
    )

    # Total parameters note box
    note_rect = patches.FancyBboxPatch(
        (0.5, 0.1), 11.5, 0.55,
        boxstyle="round,pad=0.04,rounding_size=0.06",
        facecolor="#F1F5F9",
        edgecolor=BORDER,
        linewidth=1.0,
    )
    ax.add_patch(note_rect)
    ax.text(
        6.25, 0.37,
        "Total Architecture Parameters: 46,883  |  Shared Trunk: 46,817  |  Binary Head: 33  |  Severity Head: 66  |  Edge CPU Latency: 0.026 ms",
        ha="center", va="center",
        fontsize=8.5, fontweight="bold",
        color=NAVY
    )

    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.2)
    plt.tight_layout()
    out_path = OUT_DIR / "review3_multitask_architecture.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_quantization_comparison() -> None:
    """Generate formal quantization benchmark comparison chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=DPI)
    fig.patch.set_facecolor("#FFFFFF")

    variants = ["Float32", "Dynamic-Range\n(Deployed)", "Float16", "Full INT8"]
    sizes_kb = [235.6, 111.0, 147.8, 136.9]
    latencies_ms = [0.063, 0.026, 0.062, 0.033]

    colors_size = [SLATE, BLUE, SLATE, SLATE]
    colors_lat = [SLATE, GREEN, SLATE, SLATE]

    # Subplot 1: Model Size
    bars1 = ax1.bar(variants, sizes_kb, color=colors_size, width=0.55, edgecolor=BORDER, linewidth=1.2)
    ax1.set_ylabel("Model Size (KB)", fontsize=10, fontweight="bold", color=NAVY)
    ax1.set_title("Model Footprint Comparison (TFLite)", fontsize=11, fontweight="bold", color=NAVY, pad=10)
    ax1.set_ylim(0, 280)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)

    for bar, val in zip(bars1, sizes_kb):
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, y + 6, f"{val:.1f} KB", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=DARK_GRAY)

    # Subplot 2: Single-Window Latency
    bars2 = ax2.bar(variants, latencies_ms, color=colors_lat, width=0.55, edgecolor=BORDER, linewidth=1.2)
    ax2.set_ylabel("Inference Latency per Window (ms)", fontsize=10, fontweight="bold", color=NAVY)
    ax2.set_title("Single-Window CPU Latency (1 Thread, 1000 Runs)", fontsize=11, fontweight="bold", color=NAVY, pad=10)
    ax2.set_ylim(0, 0.08)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)

    for bar, val in zip(bars2, latencies_ms):
        y = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, y + 0.002, f"{val:.3f} ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=DARK_GRAY)

    plt.tight_layout()
    out_path = OUT_DIR / "review3_quantization_comparison.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_streaming_diagram() -> None:
    """Generate Review 3 real-time streaming ring buffer diagram."""
    fig, ax = plt.subplots(figsize=(12, 4.0), dpi=DPI)
    fig.patch.set_facecolor("#FFFFFF")
    ax.axis("off")

    # Draw incoming stream
    ax.annotate(
        "Continuous 20 Hz IMU Stream\n(6 Channels: Acc + Gyro)",
        xy=(1.8, 2.0), xytext=(0.2, 2.0),
        arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=2.0),
        fontsize=8.5, fontweight="bold", color=NAVY, va="center"
    )

    # Ring Buffer Box
    box_rb = patches.FancyBboxPatch(
        (2.0, 0.8), 3.2, 2.4,
        boxstyle="round,pad=0.06,rounding_size=0.1",
        facecolor=LIGHT_BG, edgecolor=NAVY, linewidth=1.8
    )
    ax.add_patch(box_rb)
    ax.text(3.6, 2.9, "Circular Ring Buffer", ha="center", va="center", fontsize=9.5, fontweight="bold", color=NAVY)
    ax.text(3.6, 2.0, "Capacity: 64 Samples (3.2 s)\nWarmup: Suppressed for t < 3.2 s\nStride: Every 16 Samples (0.8 s)\nStandardization: scaler.pkl fit", ha="center", va="center", fontsize=8.0, color=DARK_GRAY, linespacing=1.35)

    # Arrow to TFLite Engine
    ax.annotate(
        "Window (64, 6)\nEvery 0.8 s",
        xy=(6.3, 2.0), xytext=(5.2, 2.0),
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.0),
        fontsize=8.0, fontweight="bold", color=BLUE, va="center", ha="center"
    )

    # TFLite Engine Box
    box_tf = patches.FancyBboxPatch(
        (6.5, 0.8), 2.8, 2.4,
        boxstyle="round,pad=0.06,rounding_size=0.1",
        facecolor=LIGHT_BG, edgecolor=BLUE, linewidth=1.8
    )
    ax.add_patch(box_tf)
    ax.text(7.9, 2.9, "Deployed Edge Model", ha="center", va="center", fontsize=9.5, fontweight="bold", color=BLUE)
    ax.text(7.9, 2.0, "TFLite Dynamic-Range\nSize: 111 KB | Latency: 0.026 ms\nThreshold: tau = 0.25 (D2 Rule)\nBinary Fall Prob in [0, 1]", ha="center", va="center", fontsize=8.0, color=DARK_GRAY, linespacing=1.35)

    # Arrow to Alert Policy
    ax.annotate(
        "Prob >= tau?",
        xy=(10.0, 2.0), xytext=(9.3, 2.0),
        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.0),
        fontsize=8.0, fontweight="bold", color=GREEN, va="center", ha="center"
    )

    # Alert Policy & Caregiver Sink Box
    box_pol = patches.FancyBboxPatch(
        (10.2, 0.8), 3.2, 2.4,
        boxstyle="round,pad=0.06,rounding_size=0.1",
        facecolor="#F0FDF4", edgecolor=GREEN, linewidth=2.0
    )
    ax.add_patch(box_pol)
    ax.text(11.8, 2.9, "Alert Policy & Sink", ha="center", va="center", fontsize=9.5, fontweight="bold", color=GREEN)
    ax.text(11.8, 2.0, "N-Consecutive Filter (1-3)\nCooldown Timer: 10.0 s\nCaregiver Notification Log\nAudio-Visual Streamlit Alarm", ha="center", va="center", fontsize=8.0, color=DARK_GRAY, linespacing=1.35)

    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 4.0)
    plt.tight_layout()
    out_path = OUT_DIR / "review3_streaming_pipeline.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    print("Generating Review 3 presentation figure assets...")
    generate_system_architecture()
    generate_multitask_architecture()
    generate_quantization_comparison()
    generate_streaming_diagram()
    print("All assets generated successfully!")
