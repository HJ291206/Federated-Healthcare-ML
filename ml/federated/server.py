"""Aggregation server for simulated federated training (sequential, single process).

The server holds only the global model. It never receives raw rows; evaluation on held-out
data is delegated to an `evaluate_fn` supplied by the experiment, which is how the simulation
measures the global model without giving the test set to any client.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ml.federated.client import ClientUpdate, FederatedClient, LocalTrainConfig, derive_seed


def state_nbytes(state: dict[str, torch.Tensor]) -> int:
    """Raw tensor payload in bytes (serialization/protocol overhead excluded)."""
    return int(sum(t.numel() * t.element_size() for t in state.values()))


class FederatedServer:
    def __init__(self, model_fn: Callable[[], nn.Module], clients: list[FederatedClient],
                 local_cfg: LocalTrainConfig, client_fraction: float, seed: int,
                 aggregate: Callable[[Sequence[ClientUpdate]], dict[str, torch.Tensor]],
                 regularizer_factory: Callable[[list[torch.Tensor]], Callable] | None = None,
                 algorithm: str = "fedavg"):
        if not clients:
            raise ValueError("need at least one client")
        if not 0 < client_fraction <= 1:
            raise ValueError("client_fraction must be in (0, 1]")
        self.model_fn = model_fn
        self.clients = clients
        self.local_cfg = local_cfg
        self.client_fraction = client_fraction
        self.seed = seed
        self.aggregate = aggregate
        self.regularizer_factory = regularizer_factory
        self.algorithm = algorithm
        torch.manual_seed(derive_seed(seed, 0xC0FFEE))
        self.global_model = model_fn()  # identical initialization for every algorithm given the seed
        self.round = 0
        self.history: list[dict] = []
        self.bytes_down = 0
        self.bytes_up = 0

    def global_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}

    def sample_clients(self, round_idx: int) -> list[int]:
        k = len(self.clients)
        m = max(1, int(round(self.client_fraction * k)))
        if m == k:
            return list(range(k))
        rng = np.random.default_rng(derive_seed(self.seed, round_idx, 0x5A))
        return sorted(int(i) for i in rng.choice(k, size=m, replace=False))

    def run_round(self) -> dict:
        self.round += 1
        start = time.perf_counter()
        selected = self.sample_clients(self.round)
        global_state = self.global_state()
        payload = state_nbytes(global_state)
        updates = []
        for cid in selected:
            model = self.model_fn()
            model.load_state_dict(global_state)
            reg = None
            if self.regularizer_factory is not None:
                reg = self.regularizer_factory([p for p in model.parameters() if p.requires_grad])
            updates.append(self.clients[cid].local_train(model, self.local_cfg,
                                                         derive_seed(self.seed, self.round, cid), reg))
        self.global_model.load_state_dict(self.aggregate(updates))
        down = payload * len(selected)
        up = sum(state_nbytes(u.state) for u in updates)
        self.bytes_down += down
        self.bytes_up += up
        n = sum(u.n_samples for u in updates)
        return {
            "round": self.round,
            "clients": ",".join(map(str, selected)),
            "n_clients": len(selected),
            "n_samples": n,
            "client_train_loss": sum(u.train_loss * u.n_samples for u in updates) / n,
            "round_seconds": time.perf_counter() - start,
            "max_client_seconds": max(u.seconds for u in updates),
            "bytes_down": down,
            "bytes_up": up,
            "cumulative_bytes": self.bytes_down + self.bytes_up,
        }

    def fit(self, rounds: int, evaluate_fn: Callable[[nn.Module], dict] | None = None,
            checkpoint_dir: Path | None = None, checkpoint_every: int = 0) -> list[dict]:
        """Run `rounds` more rounds (continuing after `load_checkpoint`, if any). If `checkpoint_every` > 0,
        save the global model every k rounds and at the end. Per-round seeds derive from the round index, so a
        run resumed from a checkpoint is identical to an uninterrupted one."""
        target = self.round + rounds
        for _ in range(rounds):
            record = {"algorithm": self.algorithm, **self.run_round()}
            if evaluate_fn is not None:
                record.update(evaluate_fn(self.global_model))
            self.history.append(record)
            if checkpoint_dir is not None and checkpoint_every > 0 and \
                    (self.round % checkpoint_every == 0 or self.round == target):
                self.save_checkpoint(checkpoint_dir / f"{self.algorithm}_round{self.round:04d}.pt")
        return self.history

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"algorithm": self.algorithm, "round": self.round, "seed": self.seed,
                    "state_dict": self.global_state(), "bytes_down": self.bytes_down, "bytes_up": self.bytes_up,
                    "history": self.history}, path)

    def load_checkpoint(self, path: Path) -> int:
        """Restore global weights, round counter, byte counters and history. Returns the restored round."""
        ckpt = torch.load(path, weights_only=False)
        self.global_model.load_state_dict(ckpt["state_dict"])
        self.round, self.bytes_down, self.bytes_up = ckpt["round"], ckpt["bytes_down"], ckpt["bytes_up"]
        self.history = list(ckpt["history"])
        return self.round
