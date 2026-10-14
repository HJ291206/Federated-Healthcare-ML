"""Classification metrics and validation-only threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
    precision_recall_curve, precision_recall_fscore_support, roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Threshold-dependent metrics for both classes plus ranking metrics (ROC-AUC, PR-AUC).

    Class 1 = diabetes. recall_class1 is sensitivity; recall_class0 is specificity.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    y_pred = (y_prob >= threshold).astype(int)
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": prec[1], "recall": rec[1], "f1": f1[1],
        "precision_class0": prec[0], "recall_class0": rec[0], "f1_class0": f1[0],
        "precision_class1": prec[1], "recall_class1": rec[1], "f1_class1": f1[1],
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "support_class0": int(support[0]), "support_class1": int(support[1]),
    }


def threshold_max_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold maximizing positive-class F1. Call with validation data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[int(np.argmax(f1))])


def threshold_for_recall(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float) -> float:
    """Highest threshold whose positive-class recall is at least `target_recall`. Validation data only."""
    _, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ok = recall[:-1] >= target_recall
    if not ok.any():
        return float(thresholds.min())
    return float(thresholds[ok].max())


def select_thresholds(y_val: np.ndarray, p_val: np.ndarray, recall_target: float) -> dict[str, float]:
    return {
        "default_0.5": 0.5,
        "val_max_f1": threshold_max_f1(y_val, p_val),
        f"val_recall_{recall_target:.2f}": threshold_for_recall(y_val, p_val, recall_target),
    }
