"""PyTorch training utilities, written so a federated client can call `train_one_epoch` directly."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from ml.models import DiabetesMLP, build_mlp
from ml.seed import set_global_seed, torch_generator


@dataclass
class MLPTrainConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 60
    patience: int = 6
    torch_threads: int = 4
    device: str = "cpu"  # "cpu", "cuda" or "auto" (cuda if available)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False (CPU-only torch build or no GPU)")
    return torch.device(name)


def positive_class_weight(y_train: np.ndarray) -> float:
    """n_negative / n_positive, computed from training labels only."""
    n_pos = int((y_train == 1).sum())
    if n_pos == 0:
        raise ValueError("training labels contain no positive examples")
    return float((y_train == 0).sum() / n_pos)


def make_loss(pos_weight: float) -> nn.BCEWithLogitsLoss:
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))


def iterate_minibatches(n: int, batch_size: int, generator: torch.Generator, shuffle: bool = True) -> Iterator[torch.Tensor]:
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def train_one_epoch(model: nn.Module, X: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer,
                    loss_fn: nn.Module, batch_size: int, generator: torch.Generator,
                    regularizer: Callable[[nn.Module], torch.Tensor] | None = None) -> float:
    """One pass over (X, y). `regularizer` adds a model-dependent penalty (e.g. a FedProx proximal term)."""
    model.train()
    total, count = 0.0, 0
    for idx in iterate_minibatches(len(X), batch_size, generator):
        xb, yb = X[idx], y[idx]
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        if regularizer is not None:
            loss = loss + regularizer(model)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(idx)
        count += len(idx)
    return total / count


@torch.no_grad()
def predict_logits(model: nn.Module, X: torch.Tensor, batch_size: int = 16384) -> torch.Tensor:
    model.eval()
    return torch.cat([model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])


def predict_proba(model: nn.Module, X: np.ndarray | torch.Tensor) -> np.ndarray:
    device = next(model.parameters()).device
    X_t = torch.as_tensor(X, dtype=torch.float32).to(device)
    return torch.sigmoid(predict_logits(model, X_t)).cpu().numpy().astype(np.float64)


@torch.no_grad()
def evaluate_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module) -> float:
    return float(loss_fn(predict_logits(model, X), y).item())


def fit_mlp(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
            cfg: MLPTrainConfig, seed: int) -> tuple[DiabetesMLP, pd.DataFrame, dict]:
    """Train with class-weighted BCE; early-stop on validation ROC-AUC and restore the best weights."""
    set_global_seed(seed)
    torch.set_num_threads(cfg.torch_threads)
    gen = torch_generator(seed)

    device = resolve_device(cfg.device)
    pos_weight = positive_class_weight(y_train)
    loss_fn = make_loss(pos_weight).to(device)
    model = build_mlp(X_train.shape[1], cfg.hidden_sizes, cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    Xtr = torch.as_tensor(X_train, dtype=torch.float32).to(device)
    ytr = torch.as_tensor(y_train, dtype=torch.float32).to(device)
    Xva = torch.as_tensor(X_val, dtype=torch.float32).to(device)
    yva = torch.as_tensor(y_val, dtype=torch.float32).to(device)

    best_auc, best_epoch, best_state, stale = -np.inf, 0, None, 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, Xtr, ytr, optimizer, loss_fn, cfg.batch_size, gen)
        val_loss = evaluate_loss(model, Xva, yva, loss_fn)
        val_auc = roc_auc_score(y_val, predict_proba(model, Xva))
        history.append({"seed": seed, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_roc_auc": val_auc, "epoch_seconds": time.perf_counter() - t0})
        if val_auc > best_auc + 1e-5:
            best_auc, best_epoch, best_state, stale = val_auc, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.cpu()  # downstream code (saving, SHAP, federated reuse) expects CPU tensors
    info = {"best_epoch": best_epoch, "epochs_run": len(history), "best_val_roc_auc": best_auc,
            "pos_weight": pos_weight, "train_seconds": time.perf_counter() - start, "device": str(device),
            "config": asdict(cfg)}
    return model, pd.DataFrame(history), info
