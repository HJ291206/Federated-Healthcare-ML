"""Figures for classifier evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

MODEL_COLORS = {"logistic_regression": "#4C72B0", "random_forest": "#55A868", "mlp": "#DD8452"}


def _finish(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(y_true: np.ndarray, probs: dict[str, np.ndarray], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, color=MODEL_COLORS.get(name), label=f"{name} (AUC = {roc_auc_score(y_true, p):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (recall)", title=title)
    ax.legend(loc="lower right", fontsize=8)
    _finish(fig, path)


def plot_pr_curves(y_true: np.ndarray, probs: dict[str, np.ndarray], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax.plot(rec, prec, color=MODEL_COLORS.get(name), label=f"{name} (AP = {average_precision_score(y_true, p):.4f})")
    ax.axhline(np.mean(y_true), color="k", ls="--", lw=0.8, label=f"prevalence = {np.mean(y_true):.4f}")
    ax.set(xlabel="Recall (diabetes)", ylabel="Precision (diabetes)", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper right", fontsize=8)
    _finish(fig, path)


def plot_confusion_matrices(rows: pd.DataFrame, path: Path, title: str) -> None:
    """`rows` has one row per model with tn, fp, fn, tp and threshold columns."""
    fig, axes = plt.subplots(1, len(rows), figsize=(4.6 * len(rows), 4.3))
    axes = np.atleast_1d(axes)
    for ax, (_, r) in zip(axes, rows.iterrows()):
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]], dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j]):,}\n({norm[i, j]:.1%} of row)", ha="center", va="center",
                        color="white" if norm[i, j] > 0.6 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["pred 0: no diabetes", "pred 1: diabetes"], fontsize=8)
        ax.set_yticks([0, 1], ["true 0", "true 1"], fontsize=8)
        ax.set_title(f"{r.model}\nthreshold = {r.threshold:.3f}", fontsize=9)
    fig.suptitle(title, fontsize=10)
    _finish(fig, path)


def plot_mlp_history(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for seed, h in history.groupby("seed"):
        axes[0].plot(h.epoch, h.train_loss, label=f"train, seed {seed}")
        axes[0].plot(h.epoch, h.val_loss, ls="--", label=f"val, seed {seed}")
        axes[1].plot(h.epoch, h.val_roc_auc, label=f"seed {seed}")
    axes[0].set(xlabel="Epoch", ylabel="Class-weighted BCE loss", title="MLP loss")
    axes[1].set(xlabel="Epoch", ylabel="Validation ROC-AUC", title="MLP validation ROC-AUC (early-stopping criterion)")
    for ax in axes:
        ax.legend(fontsize=7)
    _finish(fig, path)


def plot_threshold_tradeoff(y_val: np.ndarray, p_val: np.ndarray, thresholds: dict[str, float],
                            path: Path, title: str) -> None:
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    p, r = prec[:-1], rec[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thr, r, label="recall (diabetes)")
    ax.plot(thr, p, label="precision (diabetes)")
    ax.plot(thr, f1, label="F1 (diabetes)")
    for name, t in thresholds.items():
        ax.axvline(t, color="grey", ls=":", lw=1)
        ax.text(t, 1.01, name, rotation=90, fontsize=7, va="bottom", ha="center")
    ax.set(xlabel="Decision threshold on predicted probability", ylabel="Score", ylim=(0, 1), xlim=(0, 1))
    ax.set_title(title, fontsize=10, pad=55)
    ax.legend(fontsize=8)
    _finish(fig, path)
