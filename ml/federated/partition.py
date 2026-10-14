"""Reproducible IID and label-skewed (Dirichlet) partitioning into simulated clients.

Clients are simulated partitions of one national survey (BRFSS 2015). They are not
hospitals or real institutions.

Partitioning operates only on the indices it is given (the global train and validation
splits). The global test split is never passed in, so it can never reach a client.

Dirichlet label skew (Hsu, Qi & Brown, 2019, "Measuring the Effects of Non-Identical
Data Distribution for Federated Visual Classification"): for every class c a vector
q_c ~ Dir(alpha * 1_K) gives the share of class-c rows assigned to each of K clients.
Small alpha gives strongly skewed class mixes (and, because class 0 dominates, unequal
client sizes); alpha -> infinity approaches IID. The same q is applied to the train and
validation indices so each client's validation rows follow its own distribution.
Draws violating the minimum-count constraint are rejected and redrawn; the number of
draws is reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClientPartition:
    client_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray


@dataclass(frozen=True)
class PartitionResult:
    name: str
    method: str
    clients: list[ClientPartition]
    proportions: np.ndarray  # shape (num_clients, num_classes)
    draws: int
    params: dict


def split_by_proportions(idx: np.ndarray, y: np.ndarray, proportions: np.ndarray,
                         rng: np.random.Generator) -> list[np.ndarray]:
    """Assign each class's rows of `idx` to clients according to `proportions[:, class]`."""
    num_clients, num_classes = proportions.shape
    parts: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx_c = rng.permutation(idx[y[idx] == c])
        cuts = np.round(np.cumsum(proportions[:, c])[:-1] * len(idx_c)).astype(int)
        for k, chunk in enumerate(np.split(idx_c, cuts)):
            parts[k].append(chunk)
    return [np.sort(np.concatenate(p)) for p in parts]


def _class_counts(parts: list[np.ndarray], y: np.ndarray, num_classes: int) -> np.ndarray:
    return np.array([[int((y[p] == c).sum()) for c in range(num_classes)] for p in parts])


def partition_iid(train_idx: np.ndarray, val_idx: np.ndarray, y: np.ndarray, num_clients: int,
                  seed: int) -> PartitionResult:
    """Stratified equal split: every client gets ~1/K of each class."""
    rng = np.random.default_rng(seed)
    num_classes = int(y.max()) + 1
    q = np.full((num_clients, num_classes), 1.0 / num_clients)
    train_parts = split_by_proportions(train_idx, y, q, rng)
    val_parts = split_by_proportions(val_idx, y, q, rng)
    clients = [ClientPartition(k, t, v) for k, (t, v) in enumerate(zip(train_parts, val_parts))]
    return PartitionResult("iid", "iid", clients, q, 1, {"num_clients": num_clients, "seed": seed})


def partition_dirichlet(train_idx: np.ndarray, val_idx: np.ndarray, y: np.ndarray, num_clients: int,
                        alpha: float, seed: int, min_train_per_class: int = 20, min_val_per_class: int = 5,
                        max_draws: int = 10_000, name: str | None = None) -> PartitionResult:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    rng = np.random.default_rng(seed)
    num_classes = int(y.max()) + 1
    for draw in range(1, max_draws + 1):
        q = rng.dirichlet(np.full(num_clients, alpha), size=num_classes).T  # (K, C), columns sum to 1
        train_parts = split_by_proportions(train_idx, y, q, rng)
        val_parts = split_by_proportions(val_idx, y, q, rng)
        if (_class_counts(train_parts, y, num_classes) >= min_train_per_class).all() and \
                (_class_counts(val_parts, y, num_classes) >= min_val_per_class).all():
            clients = [ClientPartition(k, t, v) for k, (t, v) in enumerate(zip(train_parts, val_parts))]
            return PartitionResult(name or f"dirichlet_a{alpha:g}", "dirichlet", clients, q, draw, {
                "num_clients": num_clients, "alpha": alpha, "seed": seed,
                "min_train_per_class": min_train_per_class, "min_val_per_class": min_val_per_class})
    raise RuntimeError(f"no Dirichlet(alpha={alpha}) draw met the minimum counts in {max_draws} draws")


def make_partition(spec: dict, train_idx: np.ndarray, val_idx: np.ndarray, y: np.ndarray,
                   num_clients: int, seed: int) -> PartitionResult:
    method = spec["method"]
    if method == "iid":
        return partition_iid(train_idx, val_idx, y, num_clients, seed)
    if method == "dirichlet":
        return partition_dirichlet(train_idx, val_idx, y, num_clients, spec["alpha"], seed,
                                   spec.get("min_train_per_class", 20), spec.get("min_val_per_class", 5),
                                   name=spec.get("name"))
    raise ValueError(f"unknown partition method {method!r}")


def partition_summary(result: PartitionResult, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in result.clients:
        for split, idx in (("train", c.train_idx), ("val", c.val_idx)):
            n_pos = int(y[idx].sum())
            rows.append({"partition": result.name, "method": result.method, "client": c.client_id,
                         "split": split, "n": len(idx), "n_negative": len(idx) - n_pos, "n_positive": n_pos,
                         "positive_rate": n_pos / len(idx) if len(idx) else float("nan"),
                         "dirichlet_draws": result.draws})
    return pd.DataFrame(rows)
