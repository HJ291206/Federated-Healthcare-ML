"""FedAvg (McMahan et al., 2017): sample-count-weighted averaging of client parameters."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn

from ml.federated.client import ClientUpdate, FederatedClient, LocalTrainConfig
from ml.federated.server import FederatedServer


def weighted_average(states: Sequence[dict[str, torch.Tensor]], weights: Sequence[float]) -> dict[str, torch.Tensor]:
    """sum_k (w_k / sum w) * state_k, computed in float64. Non-floating tensors are copied from the first state."""
    if not states or len(states) != len(weights):
        raise ValueError("need one weight per state and at least one state")
    total = float(sum(weights))
    if total <= 0 or any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative with a positive sum")
    keys = states[0].keys()
    if any(s.keys() != keys for s in states):
        raise ValueError("client states have different keys")
    out = {}
    for k in keys:
        ref = states[0][k]
        if not torch.is_floating_point(ref):
            out[k] = ref.clone()
            continue
        acc = torch.zeros_like(ref, dtype=torch.float64)
        for s, w in zip(states, weights):
            acc += s[k].to(torch.float64) * (w / total)
        out[k] = acc.to(ref.dtype)
    return out


def fedavg_aggregate(updates: Sequence[ClientUpdate]) -> dict[str, torch.Tensor]:
    return weighted_average([u.state for u in updates], [u.n_samples for u in updates])


def build_fedavg_server(model_fn: Callable[[], nn.Module], clients: list[FederatedClient],
                        local_cfg: LocalTrainConfig, client_fraction: float, seed: int) -> FederatedServer:
    return FederatedServer(model_fn, clients, local_cfg, client_fraction, seed,
                           aggregate=fedavg_aggregate, regularizer_factory=None, algorithm="fedavg")
