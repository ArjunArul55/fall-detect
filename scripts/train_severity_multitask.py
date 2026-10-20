"""Train and evaluate the multi-task CNN-LSTM with binary and severity heads.

Task P2:
- Shared trunk initialized from results/cnn_lstm_baseline/cnn_lstm_best.keras
- Binary sigmoid head (all windows) + severity head (2-class softmax Mild/Severe)
- Loss = weighted BCE (existing weights) + lambda * severity_loss (mask excludes ADL & -1 rows)
- Lambda grid: {0.3, 1.0}
- Validation selection: binary F2 must not drop > 0.005 vs baseline validation F2 (0.98466);
  among eligible epochs/lambdas pick best severity macro-F1 on confident val labels.
- Binary threshold selected by D2 rule on validation, applied to test.
- Baselines: Majority class and Logistic Regression on [peak SMV, peak gyro magnitude].
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.severity_multitask import (
    build_multitask_cnn_lstm,
    compute_multitask_loss,
    severity_class_weights,
    severity_loss,
)

DATA_DIR = ROOT / "data" / "processed"
CANONICAL_SEVERITY_DIR = DATA_DIR / "severity" / "runs" / "k2_20260914T055946Z"
BASELINE_DIR = ROOT / "results" / "cnn_lstm_baseline"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "severity_multitask" / "run_v1"

SEED = 42
INPUT_SHAPE = (64, 6)
BATCH_SIZE = 64
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.05), 2)
WINDOW_STEP_SECONDS = 0.8  # stride 16 @ 20 Hz = 0.8 s


def set_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(smoke: bool = False):
    """Load train, val, and test splits with binary and severity labels."""
    csv_path = CANONICAL_SEVERITY_DIR / "reports" / "post_isolation_forest_all_windows.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Canonical metadata CSV not found: {csv_path}")

    meta_df = pd.read_csv(csv_path)

    arrays: dict[str, dict[str, np.ndarray]] = {}
    has_npy = (DATA_DIR / "train.npy").exists() and (DATA_DIR / "val.npy").exists() and (DATA_DIR / "test.npy").exists()

    if not has_npy and not smoke:
        raise FileNotFoundError(
            f"Required window tensors (train.npy, val.npy, test.npy) not found under {DATA_DIR}. "
            "Use --smoke to run test pipeline on synthetic sample batches, or provide arrays."
        )

    for split in ("train", "val", "test"):
        split_meta = meta_df[meta_df["split"] == split].reset_index(drop=True)
        n_samples = len(split_meta)

        if has_npy and not smoke:
            x = np.load(DATA_DIR / f"{split}.npy").astype(np.float32, copy=False)
        else:
            # Synthetic smoke data aligned to split metadata
            rng = np.random.RandomState(SEED + (1 if split == "val" else (2 if split == "test" else 0)))
            if smoke:
                # Ensure ADL, Mild fall, Severe fall, and -1 are present in smoke mode
                sub_dfs = []
                for sev_val in [-1, 0, 1]:
                    part = split_meta[(split_meta["class_label"] == 1) & (split_meta["refined_severity_label"] == sev_val)]
                    if len(part) > 0:
                        sub_dfs.append(part.head(16))
                adl_part = split_meta[split_meta["class_label"] == 0]
                if len(adl_part) > 0:
                    sub_dfs.append(adl_part.head(32))
                split_meta = pd.concat(sub_dfs).reset_index(drop=True)
                count = len(split_meta)
            else:
                count = n_samples
            x = rng.randn(count, 64, 6).astype(np.float32)

        # Ground truth labels from canonical metadata
        y_bin = split_meta["class_label"].to_numpy(dtype=np.int32)
        y_sev = split_meta["refined_severity_label"].to_numpy(dtype=np.int32)
        subjects = split_meta["subject_id"].to_numpy(dtype=str)
        recordings = split_meta["recording_id"].to_numpy(dtype=str)

        arrays[split] = {
            "x": x,
            "y_bin": y_bin,
            "y_sev": y_sev,
            "subjects": subjects,
            "recordings": recordings,
        }

    return arrays


def threshold_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int32)
    fa_count = np.sum((labels == 0) & (predictions == 1))
    total_negative_hours = np.sum(labels == 0) * WINDOW_STEP_SECONDS / 3600.0
    fa_per_hour = float(fa_count / total_negative_hours) if total_negative_hours > 0 else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "f2": float(fbeta_score(labels, predictions, beta=2, zero_division=0)),
        "false_alarms_per_hour": fa_per_hour,
    }


def choose_d2_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Select threshold on validation by max F2, tie-break lowest false alarms/hr, then precision."""
    sweep = [threshold_metrics(labels, probabilities, t) for t in THRESHOLDS]
    df_sweep = pd.DataFrame(sweep)
    sorted_df = df_sweep.sort_values(
        ["f2", "false_alarms_per_hour", "precision", "threshold"],
        ascending=[False, True, False, True],
    )
    best = sorted_df.iloc[0].to_dict()
    return {k: float(v) for k, v in best.items()}


def compute_binary_eval_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    preds = (probabilities >= threshold).astype(np.int32)
    t_metrics = threshold_metrics(labels, probabilities, threshold)

    accuracy = float(np.mean(labels == preds))
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()

    try:
        auroc = float(roc_auc_score(labels, probabilities))
    except Exception:
        auroc = float("nan")

    try:
        p_curve, r_curve, _ = precision_recall_curve(labels, probabilities)
        auprc = float(auc(r_curve, p_curve))
    except Exception:
        auprc = float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": t_metrics["precision"],
        "recall": t_metrics["recall"],
        "f1": t_metrics["f1"],
        "f2": t_metrics["f2"],
        "fa_per_hour": t_metrics["false_alarms_per_hour"],
        "auroc": auroc,
        "auprc": auprc,
        "confusion_matrix": cm,
    }


def compute_severity_metrics(
    y_true_sev: np.ndarray,
    predicted_sev_labels: np.ndarray,
    confident_mask: np.ndarray,
) -> dict[str, Any]:
    """Compute per-class P/R/F1, macro-F1, and confusion matrix on confident fall windows."""
    y_t = y_true_sev[confident_mask]
    y_p = predicted_sev_labels[confident_mask]

    if len(y_t) == 0:
        return {
            "n_samples": 0,
            "mild": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
            "severe": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
            "macro_f1": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    p_mild = float(precision_score(y_t, y_p, pos_label=0, zero_division=0))
    r_mild = float(recall_score(y_t, y_p, pos_label=0, zero_division=0))
    f1_mild = float(f1_score(y_t, y_p, pos_label=0, zero_division=0))
    supp_mild = int(np.sum(y_t == 0))

    p_sev = float(precision_score(y_t, y_p, pos_label=1, zero_division=0))
    r_sev = float(recall_score(y_t, y_p, pos_label=1, zero_division=0))
    f1_sev = float(f1_score(y_t, y_p, pos_label=1, zero_division=0))
    supp_sev = int(np.sum(y_t == 1))

    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    cm = confusion_matrix(y_t, y_p, labels=[0, 1]).tolist()

    return {
        "n_samples": int(len(y_t)),
        "mild": {"precision": p_mild, "recall": r_mild, "f1": f1_mild, "support": supp_mild},
        "severe": {"precision": p_sev, "recall": r_sev, "f1": f1_sev, "support": supp_sev},
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
    }


def compute_recording_severity_cm(
    recordings: np.ndarray,
    y_true_sev: np.ndarray,
    predicted_sev: np.ndarray,
    fall_mask: np.ndarray,
) -> dict[str, Any]:
    """Compute recording-level severity confusion matrix (majority label per fall recording)."""
    unique_recs = sorted(set(recordings[fall_mask]))
    rec_true = []
    rec_pred = []

    for rec_id in unique_recs:
        mask = (recordings == rec_id) & fall_mask
        true_labels = y_true_sev[mask]
        pred_labels = predicted_sev[mask]

        conf_idx = true_labels >= 0
        if np.any(conf_idx):
            # Majority vote on true and predicted
            rec_true.append(int(pd.Series(true_labels[conf_idx]).mode()[0]))
            rec_pred.append(int(pd.Series(pred_labels[conf_idx]).mode()[0]))

    if rec_true:
        cm = confusion_matrix(rec_true, rec_pred, labels=[0, 1]).tolist()
        macro_f1 = float(f1_score(rec_true, rec_pred, average="macro", zero_division=0))
    else:
        cm = [[0, 0], [0, 0]]
        macro_f1 = 0.0

    return {
        "n_recordings": len(rec_true),
        "confusion_matrix": cm,
        "macro_f1": macro_f1,
    }


def compute_subject_severity_cm(
    subjects: np.ndarray,
    y_true_sev: np.ndarray,
    predicted_sev: np.ndarray,
    fall_mask: np.ndarray,
) -> dict[str, Any]:
    """Compute per-subject severity window counts and subject-level confusion matrix."""
    per_subject: dict[str, Any] = {}
    subj_true = []
    subj_pred = []

    for subj_id in sorted(set(subjects)):
        mask = (subjects == subj_id) & fall_mask & (y_true_sev >= 0)
        t_sub = y_true_sev[mask]
        p_sub = predicted_sev[mask]

        if len(t_sub) > 0:
            sub_cm = confusion_matrix(t_sub, p_sub, labels=[0, 1]).tolist()
            sub_f1 = float(f1_score(t_sub, p_sub, average="macro", zero_division=0))
            per_subject[subj_id] = {
                "n_confident_fall_windows": int(len(t_sub)),
                "true_mild": int(np.sum(t_sub == 0)),
                "true_severe": int(np.sum(t_sub == 1)),
                "pred_mild": int(np.sum(p_sub == 0)),
                "pred_severe": int(np.sum(p_sub == 1)),
                "confusion_matrix": sub_cm,
                "macro_f1": sub_f1,
            }
            subj_true.append(int(pd.Series(t_sub).mode()[0]))
            subj_pred.append(int(pd.Series(p_sub).mode()[0]))
        else:
            per_subject[subj_id] = {
                "n_confident_fall_windows": 0,
                "true_mild": 0,
                "true_severe": 0,
                "pred_mild": 0,
                "pred_severe": 0,
                "confusion_matrix": [[0, 0], [0, 0]],
                "macro_f1": float("nan"),
                "note": "ADL-only subject (no falls)",
            }

    if subj_true:
        subj_cm = confusion_matrix(subj_true, subj_pred, labels=[0, 1]).tolist()
    else:
        subj_cm = [[0, 0], [0, 0]]

    return {
        "per_subject_details": per_subject,
        "subject_level_confusion_matrix": subj_cm,
    }


def compute_severity_baselines(
    train_x: np.ndarray,
    train_y_bin: np.ndarray,
    train_y_sev: np.ndarray,
    test_x: np.ndarray,
    test_y_bin: np.ndarray,
    test_y_sev: np.ndarray,
) -> dict[str, Any]:
    """Compute contextual severity baselines: Majority class & Logistic Regression on peaks."""
    # Confident training falls
    tr_mask = (train_y_bin == 1) & (train_y_sev >= 0)
    te_mask = (test_y_bin == 1) & (test_y_sev >= 0)

    # 1. Majority Class Baseline
    if np.any(te_mask):
        majority_class = int(pd.Series(train_y_sev[tr_mask]).mode()[0]) if np.any(tr_mask) else 1
        maj_preds = np.full(int(np.sum(te_mask)), majority_class, dtype=np.int32)
        maj_cm = confusion_matrix(test_y_sev[te_mask], maj_preds, labels=[0, 1]).tolist()
        maj_macro_f1 = float(f1_score(test_y_sev[te_mask], maj_preds, average="macro", zero_division=0))
    else:
        majority_class = 1
        maj_cm = [[0, 0], [0, 0]]
        maj_macro_f1 = 0.0

    # 2. Logistic Regression on [peak SMV, peak gyro magnitude]
    # SMV: acc magnitude sqrt(ax^2 + ay^2 + az^2)
    # Gyro magnitude: sqrt(gx^2 + gy^2 + gz^2)
    def extract_peaks(windows: np.ndarray) -> np.ndarray:
        acc_mag = np.linalg.norm(windows[:, :, :3], axis=2)
        gyro_mag = np.linalg.norm(windows[:, :, 3:], axis=2)
        peak_smv = np.max(acc_mag, axis=1, keepdims=True)
        peak_gyro = np.max(gyro_mag, axis=1, keepdims=True)
        return np.hstack([peak_smv, peak_gyro])

    tr_feats = extract_peaks(train_x[tr_mask]) if np.any(tr_mask) else np.zeros((1, 2))
    tr_labels = train_y_sev[tr_mask] if np.any(tr_mask) else np.array([0])
    te_feats = extract_peaks(test_x[te_mask]) if np.any(te_mask) else np.zeros((1, 2))
    te_labels = test_y_sev[te_mask] if np.any(te_mask) else np.array([0])

    lr_model = LogisticRegression(class_weight="balanced", random_state=SEED)
    if len(np.unique(tr_labels)) > 1 and np.any(te_mask) and len(te_labels) > 0:
        lr_model.fit(tr_feats, tr_labels)
        lr_preds = lr_model.predict(te_feats)
        lr_cm = confusion_matrix(te_labels, lr_preds, labels=[0, 1]).tolist()
        lr_macro_f1 = float(f1_score(te_labels, lr_preds, average="macro", zero_division=0))
        lr_p = precision_score(te_labels, lr_preds, average=None, zero_division=0).tolist()
        lr_r = recall_score(te_labels, lr_preds, average=None, zero_division=0).tolist()
        lr_f1 = f1_score(te_labels, lr_preds, average=None, zero_division=0).tolist()
    else:
        lr_macro_f1 = 0.0
        lr_cm = [[0, 0], [0, 0]]
        lr_p, lr_r, lr_f1 = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]

    return {
        "majority_class_baseline": {
            "predicted_class": majority_class,
            "macro_f1": maj_macro_f1,
            "confusion_matrix": maj_cm,
        },
        "logistic_regression_baseline": {
            "features": ["peak_smv", "peak_gyro_magnitude"],
            "macro_f1": lr_macro_f1,
            "mild": {"precision": lr_p[0], "recall": lr_r[0], "f1": lr_f1[0]},
            "severe": {"precision": lr_p[1], "recall": lr_r[1], "f1": lr_f1[1]},
            "confusion_matrix": lr_cm,
        },
        "circularity_disclosure": (
            "Ground-truth severity labels were obtained via unsupervised K-Means on signal-derived "
            "features (peak acc, gyro, energy). Therefore, feature-based Logistic Regression partly "
            "re-predicts the clustering input features. The multi-task neural network's gain must be "
            "evaluated honestly against this baseline without claiming clinical ground truth."
        ),
    }


def train_multitask_pipeline(output_dir: Path, smoke: bool = False, max_epochs: int = 20):
    if output_dir.exists():
        raise FileExistsError(f"Target directory already exists: {output_dir}. Refusing to overwrite.")

    output_dir.mkdir(parents=True, exist_ok=False)
    set_seeds(SEED)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)

    # 1. Load data
    data = load_dataset(smoke=smoke)
    train_x, train_y_bin, train_y_sev = data["train"]["x"], data["train"]["y_bin"], data["train"]["y_sev"]
    val_x, val_y_bin, val_y_sev = data["val"]["x"], data["val"]["y_bin"], data["val"]["y_sev"]
    test_x, test_y_bin, test_y_sev = data["test"]["x"], data["test"]["y_bin"], data["test"]["y_sev"]
    test_subj, test_rec = data["test"]["subjects"], data["test"]["recordings"]

    # 2. Baseline reference metadata
    baseline_metadata_path = BASELINE_DIR / "run_metadata.json"
    if baseline_metadata_path.exists():
        baseline_meta = json.loads(baseline_metadata_path.read_text(encoding="utf-8"))
        baseline_val_f2 = float(baseline_meta["chosen_threshold"]["f2"])
        baseline_test_metrics = baseline_meta["metrics"]["test_chosen_threshold"]
    else:
        baseline_val_f2 = 0.984658
        baseline_test_metrics = {}

    min_allowed_val_f2 = baseline_val_f2 - 0.005

    # 3. Class weights
    w_bin = {0: 0.5440134278254383, 1: 6.180084745762712}
    train_fall_mask = train_y_bin == 1
    w_sev = severity_class_weights(train_y_sev, mask=train_fall_mask)

    lambdas = [0.3, 1.0]
    candidate_results: list[dict[str, Any]] = []

    best_candidate: dict[str, Any] | None = None
    best_candidate_model: Any = None

    started_total = time.perf_counter()

    for lam in lambdas:
        print(f"\n--- Training multi-task model with lambda = {lam} ---")
        model = build_multitask_cnn_lstm(
            tf,
            input_shape=INPUT_SHAPE,
            unroll=True,
            baseline_weights_path=BASELINE_DIR / "cnn_lstm_best.keras",
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        train_ds = (
            tf.data.Dataset.from_tensor_slices((train_x, train_y_bin, train_y_sev))
            .shuffle(buffer_size=1024, seed=SEED)
            .batch(BATCH_SIZE)
        )

        n_epochs = 2 if smoke else max_epochs
        best_val_macro_f1 = -1.0
        best_epoch_weights = None
        best_epoch_info = None

        all_candidates_recorded = []
        for epoch in range(1, n_epochs + 1):
            epoch_losses = []
            for bx, by_bin, by_sev in train_ds:
                with tf.GradientTape() as tape:
                    outputs = model(bx, training=True)
                    tot_l, b_l, s_l = compute_multitask_loss(
                        by_bin,
                        by_sev,
                        outputs["binary_output"],
                        outputs["severity_logits"],
                        w_bin,
                        w_sev,
                        lambda_weight=lam,
                    )
                grads = tape.gradient(tot_l, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                epoch_losses.append(float(tot_l.numpy()))

            # Validation evaluation
            val_out = model(val_x, training=False)
            val_b_probs = val_out["binary_output"].numpy().ravel()
            val_s_logits = val_out["severity_logits"].numpy()
            val_s_preds = np.argmax(val_s_logits, axis=-1)

            chosen_thresh = choose_d2_threshold(val_y_bin, val_b_probs)
            val_f2 = chosen_thresh["f2"]

            # Severity macro-F1 on confident fall windows
            val_conf_mask = (val_y_bin == 1) & (val_y_sev >= 0)
            if np.any(val_conf_mask):
                val_sev_f1 = float(f1_score(val_y_sev[val_conf_mask], val_s_preds[val_conf_mask], average="macro", zero_division=0))
            else:
                val_sev_f1 = 0.0

            eligible = bool(val_f2 >= min_allowed_val_f2)
            weights_snapshot = [w.copy() for w in model.get_weights()]
            info = {
                "lambda": lam,
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "val_binary_f2": float(val_f2),
                "val_chosen_threshold": float(chosen_thresh["threshold"]),
                "val_severity_macro_f1": float(val_sev_f1),
                "eligible": eligible,
            }
            all_candidates_recorded.append((info, weights_snapshot, model))
            print(f"Epoch {epoch:2d}/{n_epochs}: Loss={info['train_loss']:.4f}, Val F2={val_f2:.4f} (>= {min_allowed_val_f2:.4f}: {eligible}), Val Sev Macro-F1={val_sev_f1:.4f}")

            if eligible and (val_sev_f1 > best_val_macro_f1):
                best_val_macro_f1 = val_sev_f1
                best_epoch_weights = weights_snapshot
                best_epoch_info = info

        if best_epoch_info is not None:
            candidate_results.append(best_epoch_info)
            if best_candidate is None or (best_epoch_info["val_severity_macro_f1"] > best_candidate["val_severity_macro_f1"]):
                best_candidate = best_epoch_info
                model.set_weights(best_epoch_weights)
                best_candidate_model = model

    if best_candidate is None:
        print("Warning: No candidate met the strict binary F2 constraint. Falling back to highest validation F2 candidate.")
        best_tuple = sorted(all_candidates_recorded, key=lambda x: (x[0]["val_binary_f2"], x[0]["val_severity_macro_f1"]), reverse=True)[0]
        best_candidate = best_tuple[0]
        best_candidate_model = best_tuple[2]
        best_candidate_model.set_weights(best_tuple[1])

    # Save best model
    best_model_path = output_dir / "multitask_best.keras"
    if best_candidate_model is not None:
        best_candidate_model.save(best_model_path)
    else:
        best_model_path = None

    training_time_seconds = time.perf_counter() - started_total

    # 4. Final Evaluation on Test Set using Best Candidate
    selected_thresh = best_candidate["val_chosen_threshold"]
    test_out = best_candidate_model(test_x, training=False)
    test_b_probs = test_out["binary_output"].numpy().ravel()
    test_s_logits = test_out["severity_logits"].numpy()
    test_s_probs = test_out["severity_output"].numpy()
    test_s_preds = np.argmax(test_s_logits, axis=-1)

    # (a) Binary test evaluation vs baseline
    binary_test_metrics = compute_binary_eval_metrics(test_y_bin, test_b_probs, threshold=selected_thresh)
    binary_ref_metrics = compute_binary_eval_metrics(test_y_bin, test_b_probs, threshold=0.50)

    # (b) Severity on confident fall windows: per-class P/R/F1, macro-F1, confusion matrix
    test_conf_mask = (test_y_bin == 1) & (test_y_sev >= 0)
    severity_conf_metrics = compute_severity_metrics(test_y_sev, test_s_preds, test_conf_mask)
    rec_severity_cm = compute_recording_severity_cm(test_rec, test_y_sev, test_s_preds, fall_mask=(test_y_bin == 1))
    subj_severity_cm = compute_subject_severity_cm(test_subj, test_y_sev, test_s_preds, fall_mask=(test_y_bin == 1))

    # (c) Severity on windows predicted FALL
    pred_fall_mask = test_b_probs >= selected_thresh
    predicted_falls_true_sev = test_y_sev[pred_fall_mask]
    predicted_falls_pred_sev = test_s_preds[pred_fall_mask]
    pred_fall_breakdown = {
        "total_windows_predicted_fall": int(np.sum(pred_fall_mask)),
        "true_mild_falls_in_pred": int(np.sum((test_y_bin[pred_fall_mask] == 1) & (predicted_falls_true_sev == 0))),
        "true_severe_falls_in_pred": int(np.sum((test_y_bin[pred_fall_mask] == 1) & (predicted_falls_true_sev == 1))),
        "true_uncertain_falls_in_pred": int(np.sum((test_y_bin[pred_fall_mask] == 1) & (predicted_falls_true_sev == -1))),
        "false_alarm_adl_in_pred": int(np.sum(test_y_bin[pred_fall_mask] == 0)),
        "predicted_mild_count": int(np.sum(predicted_falls_pred_sev == 0)),
        "predicted_severe_count": int(np.sum(predicted_falls_pred_sev == 1)),
    }

    # (d) Binary recall on -1 windows separately
    uncertain_fall_mask = (test_y_bin == 1) & (test_y_sev == -1)
    n_uncertain_falls = int(np.sum(uncertain_fall_mask))
    if n_uncertain_falls > 0:
        uncertain_recall = float(np.mean(test_b_probs[uncertain_fall_mask] >= selected_thresh))
        uncertain_detected = int(np.sum(test_b_probs[uncertain_fall_mask] >= selected_thresh))
    else:
        uncertain_recall = float("nan")
        uncertain_detected = 0

    uncertain_metrics = {
        "total_uncertain_fall_windows": n_uncertain_falls,
        "uncertain_fall_windows_detected_as_fall": uncertain_detected,
        "uncertain_binary_recall": uncertain_recall,
        "note": "Uncertain (-1) windows are extreme gyro outliers; they stay in binary detection but are masked from severity loss.",
    }

    # 5. Severity baselines
    baselines = compute_severity_baselines(
        train_x, train_y_bin, train_y_sev, test_x, test_y_bin, test_y_sev
    )

    # 6. Save per-window test predictions CSV
    test_df = pd.DataFrame(
        {
            "window_index": np.arange(len(test_y_bin)),
            "subject_id": test_subj,
            "recording_id": test_rec,
            "binary_label": test_y_bin,
            "true_severity_label": test_y_sev,
            "predicted_binary_prob": test_b_probs,
            "predicted_binary_label": (test_b_probs >= selected_thresh).astype(int),
            "predicted_severity_mild_prob": test_s_probs[:, 0],
            "predicted_severity_severe_prob": test_s_probs[:, 1],
            "predicted_severity_label": test_s_preds,
        }
    )
    test_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # 7. Metadata JSON
    metadata = {
        "task": "TASK P2: multi-task severity head",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "training_time_seconds": training_time_seconds,
        "best_selection": best_candidate,
        "candidates_evaluated": candidate_results,
        "validation_selection_rule": (
            f"Binary F2 >= {min_allowed_val_f2:.6f} (baseline val F2 - 0.005); "
            "highest severity macro-F1 on confident validation falls."
        ),
        "threshold_policy": "D2 rule (validation max F2, tie-break FA/hr), frozen to test; 0.50 reference.",
        "chosen_threshold": selected_thresh,
        "metrics": {
            "binary_test_chosen_threshold": binary_test_metrics,
            "binary_test_reference_0_50": binary_ref_metrics,
            "baseline_comparison": {
                "cnn_lstm_baseline_test_recall": baseline_test_metrics.get("recall", 0.995),
                "multitask_test_recall": binary_test_metrics["recall"],
                "recall_regressed": bool(binary_test_metrics["recall"] < baseline_test_metrics.get("recall", 0.995)),
            },
            "severity_confident_falls": severity_conf_metrics,
            "severity_recording_level": rec_severity_cm,
            "severity_subject_level": subj_severity_cm,
            "severity_on_predicted_falls": pred_fall_breakdown,
            "uncertain_minus1_binary_recall": uncertain_metrics,
            "severity_context_baselines": baselines,
        },
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 8. Markdown Summary
    summary_md = generate_summary_markdown(metadata)
    (output_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    print(f"\nTask P2 multi-task severity head completed successfully!")
    print(f"Results saved under: {output_dir}")
    return metadata


def generate_summary_markdown(meta: dict[str, Any]) -> str:
    m = meta["metrics"]
    b_chosen = m["binary_test_chosen_threshold"]
    b_ref = m["binary_test_reference_0_50"]
    s_conf = m["severity_confident_falls"]
    s_base = m["severity_context_baselines"]
    unc = m["uncertain_minus1_binary_recall"]

    lines = [
        "# Multi-Task Severity Head Run Summary (Task P2)",
        "",
        f"- **Run Directory**: `results/severity_multitask/run_v1`",
        f"- **Generated (UTC)**: {meta['timestamp_utc']}",
        f"- **Best Lambda**: {meta['best_selection']['lambda']} (Epoch {meta['best_selection']['epoch']})",
        f"- **Validation-Selected Binary Threshold**: {meta['chosen_threshold']:.2f} (0.50 reference)",
        f"- **Training Time**: {meta['training_time_seconds']:.2f} s",
        "",
        "## 1. Binary Fall Detection: Baseline CNN-LSTM vs Multi-Task",
        "",
        "| Model | Threshold | Accuracy | Precision | Recall | F1 | F2 | FA/hr | AUROC | AUPRC |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| CNN-LSTM Baseline | 0.25 (val-sel) | 0.9960 | 0.9606 | 0.9950 | 0.9775 | 0.9879 | 17.74 | 0.9998 | 0.9984 |",
        f"| **Multi-Task CNN-LSTM** | **{meta['chosen_threshold']:.2f} (val-sel)** | **{b_chosen['accuracy']:.4f}** | **{b_chosen['precision']:.4f}** | **{b_chosen['recall']:.4f}** | **{b_chosen['f1']:.4f}** | **{b_chosen['f2']:.4f}** | **{b_chosen['fa_per_hour']:.2f}** | **{b_chosen['auroc']:.4f}** | **{b_chosen['auprc']:.4f}** |",
        f"| Multi-Task CNN-LSTM | 0.50 (ref) | {b_ref['accuracy']:.4f} | {b_ref['precision']:.4f} | {b_ref['recall']:.4f} | {b_ref['f1']:.4f} | {b_ref['f2']:.4f} | {b_ref['fa_per_hour']:.2f} | {b_ref['auroc']:.4f} | {b_ref['auprc']:.4f} |",
        "",
        f"- **Recall Regression Check**: Baseline Recall = 0.9950, Multi-Task Recall = {b_chosen['recall']:.4f}. Regressed: `{m['baseline_comparison']['recall_regressed']}`.",
        "",
        "## 2. Severity Classification on Confident Fall Windows",
        "",
        "| Level | Evaluated Samples | Mild P/R/F1 | Severe P/R/F1 | Macro-F1 | Confusion Matrix (Mild, Severe) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| Window | {s_conf['n_samples']} | {s_conf['mild']['precision']:.4f} / {s_conf['mild']['recall']:.4f} / {s_conf['mild']['f1']:.4f} | {s_conf['severe']['precision']:.4f} / {s_conf['severe']['recall']:.4f} / {s_conf['severe']['f1']:.4f} | **{s_conf['macro_f1']:.4f}** | {s_conf['confusion_matrix']} |",
        f"| Recording | {m['severity_recording_level']['n_recordings']} | - | - | **{m['severity_recording_level']['macro_f1']:.4f}** | {m['severity_recording_level']['confusion_matrix']} |",
        f"| Subject | 6 | - | - | - | {m['severity_subject_level']['subject_level_confusion_matrix']} |",
        "",
        "### Per-Subject Severity Window Breakdown",
        "",
        "| Subject ID | Total Windows | True Mild | True Severe | Pred Mild | Pred Severe | Macro-F1 | Notes |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for subj, info in m["severity_subject_level"]["per_subject_details"].items():
        if info.get("n_confident_fall_windows", 0) > 0:
            lines.append(f"| {subj} | {info['n_confident_fall_windows']} | {info['true_mild']} | {info['true_severe']} | {info['pred_mild']} | {info['pred_severe']} | {info['macro_f1']:.4f} | Young adult (with falls) |")
        else:
            lines.append(f"| {subj} | 0 | 0 | 0 | 0 | 0 | N/A | Elderly subject (ADL-only) |")

    lines.extend([
        "",
        "## 3. Severity Classification Context Baselines",
        "",
        "| Classifier | Features | Macro-F1 | Mild F1 | Severe F1 | Notes |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| Majority Class | None (Severe always) | {s_base['majority_class_baseline']['macro_f1']:.4f} | 0.0000 | {s_base['majority_class_baseline']['macro_f1'] * 2:.4f} | Constant predictor |",
        f"| Logistic Regression | Peak SMV + Peak Gyro Mag | {s_base['logistic_regression_baseline']['macro_f1']:.4f} | {s_base['logistic_regression_baseline']['mild']['f1']:.4f} | {s_base['logistic_regression_baseline']['severe']['f1']:.4f} | Partially circular with K-Means |",
        f"| **Multi-Task CNN-LSTM** | End-to-end IMU windows | **{s_conf['macro_f1']:.4f}** | **{s_conf['mild']['f1']:.4f}** | **{s_conf['severe']['f1']:.4f}** | Joint representation |",
        "",
        f"> {s_base['circularity_disclosure']}",
        "",
        "## 4. Anomaly Fall Windows (-1) Binary Recall",
        "",
        f"- **Total -1 Fall Windows in Test**: {unc['total_uncertain_fall_windows']}",
        f"- **Detected as Fall by Binary Head**: {unc['uncertain_fall_windows_detected_as_fall']}",
        f"- **Binary Recall on -1 Windows**: **{unc['uncertain_binary_recall']:.4f}**",
        f"- *Note*: {unc['note']}",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoke", action="store_true", help="Run small smoke pipeline for testing")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    train_multitask_pipeline(output_dir=args.output_dir, smoke=args.smoke, max_epochs=args.epochs)
