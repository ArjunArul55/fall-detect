"""Create publication-ready plots from a FiLM personalization experiment run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
K_VALUES = [0, 1, 5, 10, 30]
MODEL_ORDER = [
    "FiLM (canonical threshold)",
    "FiLM (validation-retuned threshold)",
    "Global",
    "Full fine-tune (ADL-only)",
    "Full fine-tune (+2 falls)",
    "Last-layer fine-tune (ADL-only)",
    "Last-layer fine-tune (+2 falls)",
]
MODEL_LABELS = {
    "FiLM (canonical threshold)": "FiLM canonical",
    "FiLM (validation-retuned threshold)": "FiLM retuned",
    "Global": "Global",
    "Full fine-tune (ADL-only)": "Full FT (ADL-only)",
    "Full fine-tune (+2 falls)": "Full FT (+2 falls)",
    "Last-layer fine-tune (ADL-only)": "Last-layer FT (ADL-only)",
    "Last-layer fine-tune (+2 falls)": "Last-layer FT (+2 falls)",
}
MODEL_COLORS = {
    "FiLM (canonical threshold)": "#0072B2",
    "FiLM (validation-retuned threshold)": "#56B4E9",
    "Global": "#D55E00",
    "Full fine-tune (ADL-only)": "#009E73",
    "Full fine-tune (+2 falls)": "#E69F00",
    "Last-layer fine-tune (ADL-only)": "#CC79A7",
    "Last-layer fine-tune (+2 falls)": "#8C564B",
}
MODEL_MARKERS = {
    "FiLM (canonical threshold)": "o",
    "FiLM (validation-retuned threshold)": "s",
    "Global": "^",
    "Full fine-tune (ADL-only)": "D",
    "Full fine-tune (+2 falls)": "v",
    "Last-layer fine-tune (ADL-only)": "P",
    "Last-layer fine-tune (+2 falls)": "X",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/film_personalization/prototype_run_final_smoke_v2"),
        help="Directory containing the experiment CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory in which the 300-dpi PNG figures will be saved.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_csv_with_preview(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required experiment CSV not found: {path}")
    frame = pd.read_csv(path)
    print(f"\n=== {path.name} ===")
    print("Columns:", frame.columns.tolist())
    print(frame.head().to_string(index=False))
    return frame


def require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination


def style_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=15, fontweight="bold", pad=17)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def add_subtitle(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.935, text, ha="center", va="top", fontsize=10.5, color="#444444")


def real_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[metrics["evaluation"].eq("real_heldout")].copy()
    selected["k"] = pd.to_numeric(selected["k"], errors="raise").astype(int)
    return selected


def validate_requested_results(metrics: pd.DataFrame, wilcoxon: pd.DataFrame) -> None:
    data = real_summary(metrics)
    expected_pairs = {(model, k) for model in MODEL_ORDER for k in K_VALUES}
    observed_pairs = set(zip(data["model"], data["k"]))
    missing = sorted(expected_pairs - observed_pairs)
    if missing:
        raise ValueError(f"Real-heldout summary is missing requested model/k rows: {missing}")
    duplicates = data.duplicated(["model", "k"], keep=False)
    if duplicates.any():
        rows = data.loc[duplicates, ["model", "k"]].drop_duplicates().to_dict("records")
        raise ValueError(f"Real-heldout summary has duplicate model/k rows: {rows}")
    missing_evaluations = {"real_heldout", "simulated_users"} - set(wilcoxon["evaluation"])
    if missing_evaluations:
        raise ValueError(f"paired_wilcoxon.csv is missing evaluations: {sorted(missing_evaluations)}")
    wilcoxon["p_value"] = pd.to_numeric(wilcoxon["p_value"], errors="raise")


def model_legend(ax: plt.Axes, models: list[str] | None = None, **kwargs: object) -> None:
    labels = models or MODEL_ORDER
    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linewidth=2,
            markersize=6,
            label=MODEL_LABELS[model],
        )
        for model in labels
    ]
    ax.legend(handles=handles, frameon=False, **kwargs)


def plot_f2_vs_k(metrics: pd.DataFrame, output_dir: Path) -> Path:
    data = real_summary(metrics)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    best = data.loc[data["f2_mean"].idxmax()]
    for model in MODEL_ORDER:
        rows = data.loc[data["model"].eq(model)].set_index("k").reindex(K_VALUES)
        ax.errorbar(
            K_VALUES,
            rows["f2_mean"],
            yerr=rows["f2_std"],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linewidth=2,
            markersize=5,
            capsize=3,
            label=MODEL_LABELS[model],
        )
    style_axis(ax, "Real held-out SisFall: F2 vs calibration size", "Calibration windows (k)", "F2 score")
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0.985, 0.990)
    ax.legend(ncol=2, frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    add_subtitle(
        fig,
        f"Highest observed mean F2 is {best['f2_mean']:.4f} for "
        f"{MODEL_LABELS[best['model']]} at k={int(best['k'])}.",
    )
    return save_figure(fig, output_dir, "01_f2_vs_k.png")


def plot_false_alarms_vs_k(metrics: pd.DataFrame, output_dir: Path) -> Path:
    data = real_summary(metrics)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for model in MODEL_ORDER:
        rows = data.loc[data["model"].eq(model)].set_index("k").reindex(K_VALUES)
        ax.errorbar(
            K_VALUES,
            rows["false_alarms_per_hour_mean"],
            yerr=rows["false_alarms_per_hour_std"],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linewidth=2,
            markersize=5,
            capsize=3,
            label=MODEL_LABELS[model],
        )
    film = data.loc[data["model"].eq("FiLM (canonical threshold)") & data["k"].eq(1)].iloc[0]
    global_row = data.loc[data["model"].eq("Global") & data["k"].eq(1)].iloc[0]
    ax.scatter(
        [1, 1],
        [film["false_alarms_per_hour_mean"], global_row["false_alarms_per_hour_mean"]],
        s=100,
        facecolors="none",
        edgecolors=[MODEL_COLORS["FiLM (canonical threshold)"], MODEL_COLORS["Global"]],
        linewidths=2,
        zorder=5,
    )
    ax.annotate(
        f"FiLM k=1: {film['false_alarms_per_hour_mean']:.2f}",
        (1, film["false_alarms_per_hour_mean"]),
        xytext=(10, -20),
        textcoords="offset points",
        fontsize=9,
        color=MODEL_COLORS["FiLM (canonical threshold)"],
    )
    ax.annotate(
        f"Global: {global_row['false_alarms_per_hour_mean']:.2f}",
        (1, global_row["false_alarms_per_hour_mean"]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        color=MODEL_COLORS["Global"],
    )
    style_axis(
        ax,
        "Real held-out SisFall: false alarms vs calibration size",
        "Calibration windows (k)",
        "False alarms per hour",
    )
    ax.set_xticks(K_VALUES)
    model_legend(ax, ncol=2, fontsize=9, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    add_subtitle(
        fig,
        f"At k=1, canonical FiLM averages {film['false_alarms_per_hour_mean']:.2f} "
        f"vs {global_row['false_alarms_per_hour_mean']:.2f} false alarms/hour for Global.",
    )
    return save_figure(fig, output_dir, "02_false_alarms_per_hour_vs_k.png")


def plot_k1_false_alarm_bars(metrics: pd.DataFrame, output_dir: Path) -> Path:
    data = real_summary(metrics)
    data = data.loc[data["k"].eq(1)].sort_values("false_alarms_per_hour_mean")
    colors = [MODEL_COLORS[model] for model in data["model"]]
    labels = [MODEL_LABELS[model] for model in data["model"]]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(
        labels,
        data["false_alarms_per_hour_mean"],
        yerr=data["false_alarms_per_hour_std"],
        color=colors,
        capsize=4,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    style_axis(
        ax,
        "Real held-out SisFall: false alarms at k=1",
        "Model",
        "False alarms per hour",
    )
    ax.tick_params(axis="x", labelrotation=25)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    best = data.iloc[0]
    add_subtitle(
        fig,
        f"{MODEL_LABELS[best['model']]} has the lowest mean false-alarm rate "
        f"at k=1 ({best['false_alarms_per_hour_mean']:.2f}/hour).",
    )
    return save_figure(fig, output_dir, "03_false_alarms_k1_by_model.png")


def plot_tradeoff(metrics: pd.DataFrame, output_dir: Path) -> Path:
    data = real_summary(metrics)
    fig, ax = plt.subplots(figsize=(11, 7))
    for model in MODEL_ORDER:
        rows = data.loc[data["model"].eq(model)]
        sizes = [max(12, 3 * int(k)) for k in rows["k"]]
        ax.scatter(
            rows["false_alarms_per_hour_mean"],
            rows["f2_mean"],
            s=sizes,
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            alpha=0.82,
            edgecolors="white",
            linewidths=0.7,
            label=MODEL_LABELS[model],
        )
    for model, label, offset in [
        ("FiLM (canonical threshold)", "FiLM k=1", (10, 9)),
        ("Global", "Global k=1", (10, -17)),
    ]:
        row = data.loc[data["model"].eq(model) & data["k"].eq(1)].iloc[0]
        ax.annotate(
            label,
            (row["false_alarms_per_hour_mean"], row["f2_mean"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": MODEL_COLORS[model], "lw": 0.9},
        )
    style_axis(
        ax,
        "Real held-out SisFall: F2 and false-alarm trade-off",
        "False alarms per hour (mean)",
        "F2 score (mean)",
    )
    ax.grid(alpha=0.22)
    model_legend(ax, ncol=2, fontsize=9, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    add_subtitle(
        fig,
        "Each point is one model–k result; marker area increases with k (k=0 uses a visible minimum size).",
    )
    return save_figure(fig, output_dir, "04_f2_false_alarm_tradeoff.png")


def real_seed_rows(per_user: pd.DataFrame) -> pd.DataFrame:
    rows = per_user.loc[
        per_user["evaluation"].eq("real_heldout")
        & pd.to_numeric(per_user["k"], errors="raise").eq(1)
        & per_user["model"].isin(["Global", "FiLM (canonical threshold)"])
    ].copy()
    rows["false_alarms_per_hour"] = pd.to_numeric(
        rows["false_alarms_per_hour"], errors="raise"
    )
    return rows


def mean_per_user_fa(per_user: pd.DataFrame) -> pd.DataFrame:
    rows = real_seed_rows(per_user)
    return (
        rows.groupby(["subject_id", "model"], as_index=False)
        ["false_alarms_per_hour"]
        .mean()
    )


def adl_only_subjects(per_user: pd.DataFrame) -> set[str]:
    rows = real_seed_rows(per_user)
    falls = rows.groupby("subject_id")["fall_windows"].max()
    return set(falls.loc[falls.eq(0)].index.astype(str))


def plot_user_slopes(per_user: pd.DataFrame, output_dir: Path) -> Path:
    means = mean_per_user_fa(per_user)
    subjects = sorted(means["subject_id"].astype(str).unique())
    pivot = means.pivot(index="subject_id", columns="model", values="false_alarms_per_hour")
    required_models = ["Global", "FiLM (canonical threshold)"]
    missing = [model for model in required_models if model not in pivot.columns]
    if missing:
        raise ValueError(f"Per-user data has no k=1 observations for: {missing}")

    fig, ax = plt.subplots(figsize=(9, 6))
    x_positions = [0, 1]
    for subject in subjects:
        y_values = [
            pivot.loc[subject, "Global"],
            pivot.loc[subject, "FiLM (canonical threshold)"],
        ]
        ax.plot(x_positions, y_values, color="#A7A7A7", linewidth=1.4, alpha=0.85, zorder=1)
        ax.scatter(x_positions, y_values, color="#707070", s=30, zorder=2)
        ax.text(1.04, y_values[1], str(subject), va="center", fontsize=8, color="#555555")

    global_mean = pivot["Global"].mean()
    film_mean = pivot["FiLM (canonical threshold)"].mean()
    ax.plot(
        x_positions,
        [global_mean, film_mean],
        color="#111111",
        linewidth=3.2,
        marker="o",
        markersize=8,
        label="Mean across users",
        zorder=4,
    )
    ax.text(0, global_mean, f"  {global_mean:.2f}", va="bottom", fontsize=10, fontweight="bold")
    ax.text(1, film_mean, f"  {film_mean:.2f}", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x_positions, ["Global", "FiLM canonical"])
    style_axis(ax, "Per-user false alarms: Global to FiLM (k=1)", "Model", "False alarms per hour")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    delta = film_mean - global_mean
    verb = "falls" if delta < 0 else "rises"
    add_subtitle(fig, f"Mean user-level FA/hr {verb} by {abs(delta):.2f} from Global to FiLM.")
    return save_figure(fig, output_dir, "05_per_user_global_to_film_slopes.png")


def plot_user_dots(per_user: pd.DataFrame, output_dir: Path) -> Path:
    means = mean_per_user_fa(per_user)
    adl_users = adl_only_subjects(per_user)
    subjects = sorted(means["subject_id"].astype(str).unique())
    positions = {subject: index for index, subject in enumerate(subjects)}
    marker_for = {subject: ("D" if subject in adl_users else "o") for subject in subjects}
    offsets = {"Global": -0.13, "FiLM (canonical threshold)": 0.13}
    fig, ax = plt.subplots(figsize=(11, 6))
    for model in ["Global", "FiLM (canonical threshold)"]:
        rows = means.loc[means["model"].eq(model)]
        for row in rows.itertuples(index=False):
            subject = str(row.subject_id)
            ax.scatter(
                positions[subject] + offsets[model],
                row.false_alarms_per_hour,
                marker=marker_for[subject],
                s=70,
                color=MODEL_COLORS[model],
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
    ax.set_xticks(range(len(subjects)), subjects)
    style_axis(
        ax,
        "Per-user false alarms for Global and FiLM (k=1)",
        "Real held-out user",
        "False alarms per hour (mean across calibration seeds)",
    )
    model_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS[model],
               markeredgecolor="white", markersize=8, label=MODEL_LABELS[model])
        for model in ["Global", "FiLM (canonical threshold)"]
    ]
    marker_handles = [
        Line2D([0], [0], marker="o", color="#666666", linestyle="none", markersize=7,
               label="Fall user"),
        Line2D([0], [0], marker="D", color="#666666", linestyle="none", markersize=7,
               label="ADL-only (fall metrics N/A)"),
    ]
    ax.legend(handles=model_handles + marker_handles, frameon=False, ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    adl_labels = ", ".join(sorted(adl_users)) or "none"
    add_subtitle(
        fig,
        f"ADL-only users ({adl_labels}) have zero fall windows; fall-sensitive metrics are N/A.",
    )
    return save_figure(fig, output_dir, "06_per_user_false_alarm_dots.png")


def plot_wilcoxon(wilcoxon: pd.DataFrame, output_dir: Path) -> Path:
    evaluations = ["real_heldout", "simulated_users"]
    comparisons = list(dict.fromkeys(wilcoxon["comparison"].tolist()))
    fig_height = max(6.2, 0.66 * len(comparisons) + 2.2)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    offsets = {"real_heldout": -0.16, "simulated_users": 0.16}
    colors = {"real_heldout": "#0072B2", "simulated_users": "#D55E00"}
    for evaluation in evaluations:
        rows = wilcoxon.loc[wilcoxon["evaluation"].eq(evaluation)].copy()
        rows["comparison"] = pd.Categorical(rows["comparison"], categories=comparisons, ordered=True)
        rows = rows.sort_values("comparison")
        y_values = [comparisons.index(value) + offsets[evaluation] for value in rows["comparison"].astype(str)]
        bars = ax.barh(
            y_values,
            rows["p_value"],
            height=0.27,
            color=colors[evaluation],
            label=evaluation.replace("_", " "),
            edgecolor="white",
        )
        ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
    ax.axvline(0.05, color="#555555", linestyle="--", linewidth=1.2, label="p=0.05")
    ax.axvline(
        0.125,
        color="#8A4F7D",
        linestyle="--",
        linewidth=1.4,
        label="min achievable p with n=4 (0.125)",
    )
    style_axis(ax, "Paired Wilcoxon comparisons", "p-value", "Comparison")
    ax.set_yticks(range(len(comparisons)), comparisons)
    ax.set_xlim(0, 1.08)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout(rect=(0, 0.12, 1, 0.91))
    minimum = wilcoxon["p_value"].min()
    add_subtitle(fig, f"Smallest reported p-value is {minimum:.3g}; the dashed lines mark 0.05 and 0.125.")
    return save_figure(fig, output_dir, "07_wilcoxon_p_values.png")


def plot_deployment_size(deployment: pd.DataFrame, output_dir: Path) -> Path:
    row = deployment.iloc[0]
    labels = ["Base", "FiLM"]
    values = [int(row["base_tflite_bytes"]), int(row["film_tflite_bytes"])]
    colors = ["#7A8793", MODEL_COLORS["FiLM (canonical threshold)"]]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    bars = ax.bar(labels, values, color=colors, width=0.52, edgecolor="white")
    ax.bar_label(bars, labels=[f"{value:,} B" for value in values], padding=5, fontsize=10)
    style_axis(ax, "TFLite deployment size", "Model", "TFLite model size (bytes)")
    ax.set_ylim(0, max(values) * 1.2)
    ax.annotate(
        f"Overhead: {int(row['tflite_overhead_bytes']):,} B",
        xy=(0.5, max(values) * 1.02),
        xytext=(0.5, max(values) * 1.12),
        ha="center",
        fontsize=10,
        fontweight="bold",
        arrowprops={"arrowstyle": "<->", "color": "#333333", "lw": 1.2},
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    add_subtitle(fig, f"Per-user affine state: {int(row['user_affine_bytes']):,} B.")
    return save_figure(fig, output_dir, "08_deployment_size.png")


def add_pipeline_box(
    ax: plt.Axes, x: float, y: float, width: float, height: float, text: str,
    facecolor: str, fontsize: float = 10,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.2,
        edgecolor="#34495E",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)


def add_pipeline_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color="#34495E",
        )
    )


def plot_pipeline(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    boxes = [
        (0.25, 2.65, 1.45, 1.0, "Calibration\nwindows", "#EAF2F8"),
        (2.05, 2.65, 1.45, 1.0, "UserEncoder", "#D6EAF8"),
        (3.85, 2.65, 1.5, 1.0, "128-d\nprofile", "#D6EAF8"),
        (5.7, 2.65, 1.5, 1.0, "gamma / beta", "#FCF3CF"),
        (7.6, 2.35, 2.6, 1.6, "Frozen CNN + LSTM\nFiLM layers apply\nprofile modulation", "#E8DAEF"),
        (10.65, 2.65, 1.5, 1.0, "Fall\nprobability", "#D5F5E3"),
        (12.55, 2.65, 1.4, 1.0, "Threshold\n0.25", "#FDEBD0"),
        (14.3, 2.65, 1.4, 1.0, "Alert", "#F5B7B1"),
    ]
    for box in boxes:
        add_pipeline_box(ax, *box)
    for left, right in zip(boxes, boxes[1:]):
        add_pipeline_arrow(
            ax,
            (left[0] + left[2] + 0.06, left[1] + left[3] / 2),
            (right[0] - 0.06, right[1] + right[3] / 2),
        )
    ax.text(
        8.9,
        1.65,
        "Base CNN+LSTM weights remain frozen during personalization",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )
    ax.set_title("FiLM personalization and inference pipeline", fontsize=15, fontweight="bold", pad=20)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    add_subtitle(fig, "A compact user profile modulates the frozen backbone before thresholding and alerting.")
    return save_figure(fig, output_dir, "09_film_architecture_pipeline.png")


def main() -> None:
    args = parse_args()
    results_dir = resolve_path(args.results_dir)
    output_dir = resolve_path(args.output_dir)

    metrics = read_csv_with_preview(results_dir / "metrics_summary.csv")
    wilcoxon = read_csv_with_preview(results_dir / "paired_wilcoxon.csv")
    per_user = read_csv_with_preview(results_dir / "per_user_metrics.csv")
    deployment = read_csv_with_preview(results_dir / "deployment_size.csv")

    require_columns(
        metrics,
        [
            "evaluation",
            "model",
            "k",
            "f2_mean",
            "f2_std",
            "false_alarms_per_hour_mean",
            "false_alarms_per_hour_std",
        ],
        "metrics_summary.csv",
    )
    require_columns(wilcoxon, ["evaluation", "comparison", "p_value"], "paired_wilcoxon.csv")
    require_columns(
        per_user,
        ["evaluation", "subject_id", "model", "k", "seed", "fall_windows", "false_alarms_per_hour"],
        "per_user_metrics.csv",
    )
    require_columns(
        deployment,
        ["base_tflite_bytes", "film_tflite_bytes", "tflite_overhead_bytes", "user_affine_bytes"],
        "deployment_size.csv",
    )
    validate_requested_results(metrics, wilcoxon)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "savefig.dpi": 300,
        }
    )
    output_files = [
        plot_f2_vs_k(metrics, output_dir),
        plot_false_alarms_vs_k(metrics, output_dir),
        plot_k1_false_alarm_bars(metrics, output_dir),
        plot_tradeoff(metrics, output_dir),
        plot_user_slopes(per_user, output_dir),
        plot_user_dots(per_user, output_dir),
        plot_wilcoxon(wilcoxon, output_dir),
        plot_deployment_size(deployment, output_dir),
        plot_pipeline(output_dir),
    ]
    print("\nSaved figures:")
    for path in output_files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
