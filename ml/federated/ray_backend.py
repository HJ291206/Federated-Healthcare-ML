"""Optional Ray execution backend: clients train in parallel actor processes.

Distributed execution is NOT a privacy mechanism. Ray only changes *where* each simulated
client's local training runs (one actor process per client, on one machine or a cluster).
The aggregation semantics are unchanged: the same client code, the same seeds, the same
sample-count-weighted FedAvg / FedProx aggregation, so results match the sequential
backend bit for bit (see tests/test_ray_backend.py). What a client sends to the server is
also unchanged (its model parameters), so plain FedAvg/FedProx on Ray has no formal
privacy guarantee either; see docs/differential_privacy.md for that.

Usage:
    with RayFederatedServer(..., num_cpus=4) as server:
        server.fit(rounds, evaluate_fn)
The context manager kills the actors and shuts Ray down if this process started it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import ray
import torch
from torch import nn

from ml.federated.client import ClientUpdate, FederatedClient, LocalTrainConfig, derive_seed
from ml.federated.server import FederatedServer, state_nbytes


@ray.remote
class ClientActor:
    """Holds one client's data for the lifetime of the experiment; only model states leave."""

    def __init__(self, client: FederatedClient, torch_threads: int):
        torch.set_num_threads(torch_threads)
        torch.use_deterministic_algorithms(True)
        self.client = client

    def local_train(self, model_fn: Callable[[], nn.Module], global_state: dict[str, torch.Tensor],
                    cfg: LocalTrainConfig, seed: int, regularizer: Any) -> ClientUpdate:
        model = model_fn()
        model.load_state_dict(global_state)
        return self.client.local_train(model, cfg, seed, regularizer)

    def n_train(self) -> int:
        return self.client.n_train


def init_ray(num_cpus: int | None = None, address: str | None = None) -> bool:
    """Start Ray locally (CPU) or connect to `address`. Returns True if this call started it."""
    if ray.is_initialized():
        return False
    ray.init(address=address, num_cpus=num_cpus, include_dashboard=False, ignore_reinit_error=True,
             log_to_driver=False, configure_logging=False)
    return True


class RayFederatedServer(FederatedServer):
    def __init__(self, model_fn: Callable[[], nn.Module], clients: list[FederatedClient],
                 local_cfg: LocalTrainConfig, client_fraction: float, seed: int,
                 aggregate: Callable[[Sequence[ClientUpdate]], dict[str, torch.Tensor]],
                 regularizer_factory: Callable[[list[torch.Tensor]], Any] | None = None,
                 algorithm: str = "fedavg", num_cpus: int | None = None, address: str | None = None,
                 actor_threads: int = 1):
        super().__init__(model_fn, clients, local_cfg, client_fraction, seed, aggregate, regularizer_factory, algorithm)
        self._started_ray = init_ray(num_cpus, address)
        self.actors = [ClientActor.remote(c, actor_threads) for c in clients]
        probe = model_fn()
        self._param_names = [n for n, p in probe.named_parameters() if p.requires_grad]

    def run_round(self) -> dict:
        self.round += 1
        start = time.perf_counter()
        selected = self.sample_clients(self.round)
        global_state = self.global_state()
        payload = state_nbytes(global_state)
        state_ref = ray.put(global_state)  # broadcast once
        reg = None
        if self.regularizer_factory is not None:
            reg = self.regularizer_factory([global_state[n] for n in self._param_names])
        futures = [self.actors[cid].local_train.remote(self.model_fn, state_ref, self.local_cfg,
                                                       derive_seed(self.seed, self.round, cid), reg)
                   for cid in selected]
        updates = ray.get(futures)  # same order as `selected`, as in the sequential server
        self.global_model.load_state_dict(self.aggregate(updates))
        down, up = payload * len(selected), sum(state_nbytes(u.state) for u in updates)
        self.bytes_down += down
        self.bytes_up += up
        n = sum(u.n_samples for u in updates)
        return {"round": self.round, "clients": ",".join(map(str, selected)), "n_clients": len(selected),
                "n_samples": n, "client_train_loss": sum(u.train_loss * u.n_samples for u in updates) / n,
                "round_seconds": time.perf_counter() - start, "max_client_seconds": max(u.seconds for u in updates),
                "bytes_down": down, "bytes_up": up, "cumulative_bytes": self.bytes_down + self.bytes_up}

    def shutdown(self) -> None:
        for a in self.actors:
            ray.kill(a, no_restart=True)
        self.actors = []
        if self._started_ray and ray.is_initialized():
            ray.shutdown()
            self._started_ray = False

    def __enter__(self) -> "RayFederatedServer":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()
