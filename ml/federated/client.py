"""Simulated federated client: holds its own rows and trains the received global model locally."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ml.training import make_loss, train_one_epoch


@dataclass
class LocalTrainConfig:
    lr: float = 0.05
    batch_size: int = 64
    local_epochs: int = 1
    momentum: float = 0.0
    weight_decay: float = 0.0


@dataclass
class ClientUpdate:
    client_id: int
    state: dict[str, torch.Tensor]
    n_samples: int
    train_loss: float
    seconds: float


def derive_seed(*parts: int) -> int:
    """Deterministic, collision-resistant seed from (seed, round, client, ...)."""
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


class FederatedClient:
    def __init__(self, client_id: int, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray, pos_weight: float):
        self.client_id = client_id
        self.X_train = torch.as_tensor(X_train, dtype=torch.float32)
        self.y_train = torch.as_tensor(y_train, dtype=torch.float32)
        self.X_val = torch.as_tensor(X_val, dtype=torch.float32)
        self.y_val = torch.as_tensor(y_val, dtype=torch.float32)
        self.loss_fn = make_loss(pos_weight)

    @property
    def n_train(self) -> int:
        return len(self.y_train)

    def label_counts(self) -> tuple[int, int]:
        """(n_negative, n_positive) of the local training labels."""
        n_pos = int(self.y_train.sum().item())
        return self.n_train - n_pos, n_pos

    def local_train(self, model: nn.Module, cfg: LocalTrainConfig, seed: int,
                    regularizer: Callable[[nn.Module], torch.Tensor] | None = None) -> ClientUpdate:
        """Train `model` (already loaded with the global weights) for cfg.local_epochs epochs."""
        start = time.perf_counter()
        torch.manual_seed(seed)  # dropout masks
        gen = torch.Generator().manual_seed(seed)  # minibatch order
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
        losses = [train_one_epoch(model, self.X_train, self.y_train, optimizer, self.loss_fn,
                                  cfg.batch_size, gen, regularizer) for _ in range(cfg.local_epochs)]
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return ClientUpdate(self.client_id, state, self.n_train, float(np.mean(losses)),
                            time.perf_counter() - start)
