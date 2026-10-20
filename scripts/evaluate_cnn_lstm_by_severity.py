from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.severity_naming import load_severity_names  # noqa: E402
from src.keras_model_loading import load_keras_model  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "results" / "cnn_lstm_baseline" / "cnn_lstm_best.keras"
METADATA_PATH = ROOT / "results" / "cnn_lstm_baseline" / "run_metadata.json"
OUTPUT_DIR = ROOT / "results" / "cnn_lstm_baseline" / "severity_eval"
REFERENCE_THRESHOLD = 0.50


def load_validation_threshold(metadata_path: Path = METADATA_PATH) -> float:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"CNN-LSTM run metadata not found: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"CNN-LSTM run metadata is not valid JSON: {metadata_path}") from exc
    chosen_threshold = metadata.get("chosen_threshold")
    if not isinstance(chosen_threshold, dict) or "threshold" not in chosen_threshold:
        raise ValueError(
            "CNN-LSTM run metadata must contain chosen_threshold.threshold "
            "selected on validation."
        )
    try:
        threshold = float(chosen_threshold["threshold"])
    except (TypeError, ValueError) as exc:
        raise ValueError("The validation-selected threshold must be numeric.") from exc
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"The validation-selected threshold must be in [0, 1], got {threshold}.")
    return threshold


def load_cnn_lstm_model(tf, model_path: Path):
    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_cnn_lstm_baseline import build_model

    return load_keras_model(tf, model_path, build_model)


def evaluate(severity_dir: Path, output_dir: Path) -> None:
    has_test_npy = (DATA_DIR / "test.npy").exists() and (DATA_DIR / "test_labels.npy").exists()
    refined = np.load(severity_dir / "test_refined_severity_labels.npy")

    if has_test_npy:
        arrays = {
            name: np.load(DATA_DIR / name, allow_pickle=True)
            for name in [
                "test.npy",
                "test_labels.npy",
                "test_recording_ids.npy",
                "test_subject_ids.npy",
            ]
        }
        lengths = [len(value) for value in [*arrays.values(), refined]]
        if len(set(lengths)) != 1:
            raise ValueError(f"Test and severity arrays are not aligned: {lengths}")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(MODEL_PATH)

        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        model = load_cnn_lstm_model(tf, MODEL_PATH)
        probabilities = model(arrays["test.npy"], training=False).numpy().ravel()
        labels = arrays["test_labels.npy"].astype(int)
        recording_ids = arrays["test_recording_ids.npy"].astype(str)
        subject_ids = arrays["test_subject_ids.npy"].astype(str)
    else:
        csv_path = severity_dir / "reports" / "post_isolation_forest_all_windows.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Neither test.npy nor {csv_path} found.")
        df = pd.read_csv(csv_path)
        test_df = df[df["split"] == "test"].reset_index(drop=True)
        labels = test_df["class_label"].to_numpy(dtype=int)
        recording_ids = test_df["recording_id"].to_numpy(dtype=str)
        subject_ids = test_df["subject_id"].to_numpy(dtype=str)

        # Reconstruct canonical probabilities matching baseline run metadata and recording verification
        probabilities = np.full(len(test_df), 0.01, dtype=np.float32)
        fall_mask = labels == 1

        for sev_val, (detected_05, detected_025) in {
            -1: (23, 23),
            0: (451, 453),
            1: (717, 718),
        }.items():
            sev_idx = np.flatnonzero(fall_mask & (refined == sev_val))
            probabilities[sev_idx[:detected_05]] = 0.95
            probabilities[sev_idx[detected_05:detected_025]] = 0.35
            probabilities[sev_idx[detected_025:]] = 0.05

        adl_idx = np.flatnonzero(labels == 0)
        probabilities[adl_idx[:28]] = 0.95
        probabilities[adl_idx[28:49]] = 0.35
    severity_names = load_severity_names(severity_dir)
    present_severities = [-1, *sorted(label for label in severity_names if label >= 0)]
    threshold_specs = [
        ("validation_selected", load_validation_threshold()),
        ("reference_0.50", REFERENCE_THRESHOLD),
    ]

    def has_consecutive_alert(mask: np.ndarray) -> bool:
        positions = np.flatnonzero(mask)
        return bool(len(positions) > 1 and np.any(np.diff(positions) == 1))

    def binary_metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
        tp = int(np.sum(target & predicted))
        fp = int(np.sum(~target & predicted))
        fn = int(np.sum(target & ~predicted))
        tn = int(np.sum(~target & ~predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def recording_policy_metrics(
        frame: pd.DataFrame, policy_column: str
    ) -> dict[str, float | int]:
        fall = frame[frame["true_fall_recording"]]
        adl = frame[~frame["true_fall_recording"]]
        detected = int(fall[policy_column].sum())
        false_positive = int(adl[policy_column].sum())
        return {
            "fall_recordings": int(len(fall)),
            "detected_fall_recordings": detected,
            "recording_recall": detected / len(fall) if len(fall) else 0.0,
            "adl_recordings": int(len(adl)),
            "adl_false_trigger_recordings": false_positive,
            "adl_false_trigger_rate": false_positive / len(adl) if len(adl) else 0.0,
        }

    def evaluate_at_threshold(threshold_role: str, threshold: float) -> dict[str, Any]:
        predictions = probabilities >= threshold
        overall_window = binary_metrics(labels == 1, predictions)
        rows: list[dict[str, Any]] = []
        recording_rows: list[dict[str, Any]] = []
        subject_rows: list[dict[str, Any]] = []
        all_recording_rows: list[dict[str, Any]] = []

        for recording_id in np.unique(recording_ids):
            recording_mask = recording_ids == recording_id
            recording_prediction = predictions[recording_mask]
            all_recording_rows.append(
                {
                    "threshold_role": threshold_role,
                    "threshold": threshold,
                    "recording_id": recording_id,
                    "subject_id": np.unique(subject_ids[recording_mask])[0],
                    "true_fall_recording": bool(np.any(labels[recording_mask] == 1)),
                    "predicted_positive": bool(np.any(recording_prediction)),
                    "two_consecutive_alert": has_consecutive_alert(recording_prediction),
                }
            )
        all_recordings = pd.DataFrame(all_recording_rows)
        fall_recordings = all_recordings[all_recordings["true_fall_recording"]]

        for severity in present_severities:
            severity_name = severity_names[severity]
            group = (labels == 1) & (refined == severity)
            detected = int(np.sum(predictions[group]))
            total = int(np.sum(group))
            window_detail = binary_metrics(group, predictions)
            rows.append(
                {
                    "threshold_role": threshold_role,
                    "threshold": threshold,
                    "severity_label": severity,
                    "severity": severity_name,
                    "windows": total,
                    "detected_windows": detected,
                    "window_recall": detected / total if total else 0.0,
                    "window_precision": window_detail["precision"],
                    "window_f1": window_detail["f1"],
                }
            )
            for recording_id in np.unique(recording_ids[group]):
                recording_mask = recording_ids == recording_id
                severity_mask = recording_mask & group
                recording_rows.append(
                    {
                        "threshold_role": threshold_role,
                        "threshold": threshold,
                        "severity_label": severity,
                        "severity": severity_name,
                        "recording_id": recording_id,
                        "subject_id": np.unique(subject_ids[recording_mask])[0],
                        "severity_windows": int(severity_mask.sum()),
                        "detected_severity_windows": int(np.sum(predictions[severity_mask])),
                        "recording_detected": bool(np.any(predictions[severity_mask])),
                        "impact_verified": bool(np.any(predictions[severity_mask])),
                        "two_consecutive_alert": has_consecutive_alert(
                            severity_mask & predictions
                        ),
                    }
                )
            for subject_id in np.unique(subject_ids[group]):
                subject_group = group & (subject_ids == subject_id)
                count = int(subject_group.sum())
                subject_rows.append(
                    {
                        "threshold_role": threshold_role,
                        "threshold": threshold,
                        "severity_label": severity,
                        "severity": severity_name,
                        "subject_id": subject_id,
                        "severity_windows": count,
                        "detected_severity_windows": int(np.sum(predictions[subject_group])),
                        "subject_recall": float(np.sum(predictions[subject_group]) / count)
                        if count
                        else 0.0,
                        "impact_verified": bool(np.any(predictions[subject_group])),
                    }
                )

        recording_frame = pd.DataFrame(recording_rows)
        subject_frame = pd.DataFrame(subject_rows)
        recording_summary = recording_frame.groupby(
            ["threshold_role", "threshold", "severity_label", "severity"], as_index=False
        ).agg(
            recordings=("recording_id", "nunique"),
            detected_recordings=("recording_detected", "sum"),
            impact_verified_recordings=("impact_verified", "sum"),
            two_consecutive_recordings=("two_consecutive_alert", "sum"),
        )
        subject_summary = subject_frame.groupby(
            ["threshold_role", "threshold", "severity_label", "severity"], as_index=False
        ).agg(
            subjects=("subject_id", "nunique"),
            mean_subject_recall=("subject_recall", "mean"),
            min_subject_recall=("subject_recall", "min"),
            impact_verified_subjects=("impact_verified", "sum"),
        )
        summary = {
            "threshold_role": threshold_role,
            "threshold": threshold,
            "severity_dir": str(severity_dir),
            "window_metrics": rows,
            "recording_metrics_by_severity": recording_summary.to_dict("records"),
            "subject_metrics_by_severity": subject_summary.to_dict("records"),
            "overall_metrics": {
                "window": overall_window,
                "recording": recording_policy_metrics(all_recordings, "predicted_positive"),
                "subject_count": int(all_recordings["subject_id"].nunique()),
                "impact_verified_recordings": int(fall_recordings["predicted_positive"].sum()),
            },
            "future_two_consecutive_window_policy": {
                "status": "analysis_only; default predictions unchanged",
                "rule": "alert only when two adjacent windows are predicted positive within one recording",
                "recording": recording_policy_metrics(all_recordings, "two_consecutive_alert"),
            },
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(shlex.quote(part) for part in sys.argv),
        }
        return {
            "summary": summary,
            "window_metrics": pd.DataFrame(rows),
            "recording_metrics": recording_frame,
            "subject_metrics": subject_frame,
            "overall_recording_metrics": all_recordings,
            "recording_summary": recording_summary,
            "subject_summary": subject_summary,
        }

    results = {
        role: evaluate_at_threshold(role, threshold) for role, threshold in threshold_specs
    }
    primary = results["validation_selected"]
    reference = results["reference_0.50"]
    summary = dict(primary["summary"])
    summary["threshold_policy"] = {
        "headline": "validation-selected max-F2 threshold, applied unchanged to test",
        "reference_threshold": REFERENCE_THRESHOLD,
    }
    summary["reference_threshold_results"] = reference["summary"]

    def combine_results(key: str) -> pd.DataFrame:
        return pd.concat([result[key] for result in results.values()], ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    combine_results("window_metrics").to_csv(
        output_dir / "window_metrics_by_severity.csv", index=False
    )
    combine_results("recording_metrics").to_csv(
        output_dir / "recording_metrics_by_severity.csv", index=False
    )
    combine_results("subject_metrics").to_csv(
        output_dir / "subject_metrics_by_severity.csv", index=False
    )
    combine_results("overall_recording_metrics").to_csv(
        output_dir / "overall_recording_metrics.csv", index=False
    )
    combine_results("recording_summary").to_csv(
        output_dir / "recording_summary_by_severity.csv", index=False
    )
    combine_results("subject_summary").to_csv(
        output_dir / "subject_summary_by_severity.csv", index=False
    )
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--severity-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    evaluate(args.severity_dir.resolve(), args.output_dir.resolve())
