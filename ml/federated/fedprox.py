"""FedProx (Li et al., 2020): FedAvg aggregation with a proximal term in each client's objective.

local_loss = classification_loss + (mu / 2) * sum_p ||w_p - w_global_p||^2

over the trainable parameters p. With mu = 0 FedProx reduces exactly to FedAvg.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn

from ml.federated.client import FederatedClient, LocalTrainConfig
from ml.federated.fedavg import fedavg_aggregate
from ml.federated.server import FederatedServer


def trainable_parameters(model: nn.Module) -> list[torch.Tensor]:
    return [p for p in model.parameters() if p.requires_grad]


def proximal_term(model: nn.Module, global_params: Sequence[torch.Tensor], mu: float) -> torch.Tensor:
    params = trainable_parameters(model)
    if len(params) != len(global_params):
        raise ValueError("global_params must match the model's trainable parameters")
    sq = sum(((p - g) ** 2).sum() for p, g in zip(params, global_params))
    return 0.5 * mu * sq


def fedprox_local_loss(classification_loss: torch.Tensor, model: nn.Module,
                       global_params: Sequence[torch.Tensor], mu: float) -> torch.Tensor:
    return classification_loss + proximal_term(model, global_params, mu)


class ProximalRegularizer:
    """Callable loss term (mu/2)||w - w_global||^2 with a frozen anchor.

    `add_gradient` applies the term's exact gradient mu * (w - w_global) directly to `.grad`.
    The DP client uses it because the term is data-independent: adding it after per-example
    clipping and noising costs no privacy budget, whereas putting it in the loss would be
    dropped by Opacus (per-sample hooks only see data-dependent gradients).
    """

    def __init__(self, global_params: Sequence[torch.Tensor], mu: float):
        self.anchor = [g.detach().clone() for g in global_params]
        self.mu = float(mu)

    def __call__(self, model: nn.Module) -> torch.Tensor:
        return proximal_term(model, self.anchor, self.mu)

    @torch.no_grad()
    def add_gradient(self, model: nn.Module) -> None:
        params = trainable_parameters(model)
        if len(params) != len(self.anchor):
            raise ValueError("anchor does not match the model's trainable parameters")
        for p, g in zip(params, self.anchor):
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.add_(self.mu * (p - g))


def make_proximal_regularizer(global_params: Sequence[torch.Tensor], mu: float) -> ProximalRegularizer:
    return ProximalRegularizer(global_params, mu)


def build_fedprox_server(model_fn: Callable[[], nn.Module], clients: list[FederatedClient],
                         local_cfg: LocalTrainConfig, client_fraction: float, seed: int,
                         mu: float) -> FederatedServer:
    if mu < 0:
        raise ValueError("mu must be non-negative")
    factory = None if mu == 0 else (lambda gp: make_proximal_regularizer(gp, mu))
    return FederatedServer(model_fn, clients, local_cfg, client_fraction, seed,
                           aggregate=fedavg_aggregate, regularizer_factory=factory,
                           algorithm=f"fedprox_mu{mu:g}")
