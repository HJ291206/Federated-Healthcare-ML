"""Optional example-level differential privacy for client training (DP-SGD via Opacus).

Privacy unit: ONE TRAINING EXAMPLE (one survey respondent's row) of ONE CLIENT.
This is example-level DP, not client-level DP. The guarantee is per client: client k's
sequence of released model updates is (epsilon_k, delta)-DP with respect to adding or
removing one of client k's training rows. Each client keeps its own accountant across
rounds, so epsilon_k accumulates over all local steps of all rounds.

Mechanism (Abadi et al., 2016), implemented by Opacus:
  * minibatches drawn by Poisson sampling with rate q = batch_size / n_k,
  * per-example gradient clipping to L2 norm C (max_grad_norm),
  * Gaussian noise N(0, (sigma * C)^2) added to the summed clipped gradient,
  * accounting with the RDP accountant (epsilon reported for a fixed delta).

Assumptions: the aggregation server and other clients are honest-but-curious; the noise
RNG is torch's default generator (secure_mode=False, not cryptographically secure);
hyperparameters were not tuned under DP, so their choice does not consume budget here but
would in a strict treatment.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch
from opacus import PrivacyEngine
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.federated.client import ClientUpdate, FederatedClient, LocalTrainConfig, derive_seed

# Opacus hooks on the first Linear layer trigger this harmless PyTorch notice (inputs need no grad).
warnings.filterwarnings("ignore", message="Full backward hook is firing")


@dataclass
class DPConfig:
    noise_multiplier: float          # sigma, noise std as a multiple of the clipping norm
    max_grad_norm: float = 1.0       # per-example L2 clipping norm C
    delta: float = 1e-5
    accountant: str = "rdp"
    poisson_sampling: bool = True    # required for the accountant's sampling assumption

    def to_dict(self) -> dict:
        return asdict(self)


class DPFederatedClient(FederatedClient):
    """FederatedClient whose local training is example-level DP-SGD with a persistent accountant."""

    def __init__(self, client: FederatedClient, dp: DPConfig):
        super().__init__(client.client_id, client.X_train.numpy(), client.y_train.numpy().astype(int),
                         client.X_val.numpy(), client.y_val.numpy().astype(int),
                         float(client.loss_fn.pos_weight))
        if dp.noise_multiplier < 0 or dp.max_grad_norm <= 0 or not 0 < dp.delta < 1:
            raise ValueError("invalid DPConfig")
        self.dp = dp
        self.engine = PrivacyEngine(accountant=dp.accountant, secure_mode=False)
        self.steps = 0
        self.sample_rate: float | None = None
        self._dataset = TensorDataset(self.X_train, self.y_train)  # one object: accounting is per dataset

    def local_train(self, model: nn.Module, cfg: LocalTrainConfig, seed: int,
                    regularizer: Callable[[nn.Module], torch.Tensor] | None = None) -> ClientUpdate:
        if regularizer is not None and not hasattr(regularizer, "add_gradient"):
            raise TypeError("DP training supports only data-independent regularizers exposing add_gradient()")
        start = time.perf_counter()
        torch.manual_seed(seed)
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
        loader = DataLoader(self._dataset, batch_size=cfg.batch_size,
                            shuffle=True, generator=torch.Generator().manual_seed(seed))
        gs_model, dp_opt, dp_loader = self.engine.make_private(
            module=model, optimizer=optimizer, data_loader=loader,
            noise_multiplier=self.dp.noise_multiplier, max_grad_norm=self.dp.max_grad_norm,
            poisson_sampling=self.dp.poisson_sampling,
            noise_generator=torch.Generator().manual_seed(derive_seed(seed, 0xD9)),
        )
        self.sample_rate = min(1.0, cfg.batch_size / self.n_train)
        gs_model.train()
        losses, n_steps = [], 0
        for _ in range(cfg.local_epochs):
            for xb, yb in dp_loader:
                dp_opt.zero_grad(set_to_none=True)
                if len(xb) == 0:  # possible under Poisson sampling; accountant treats it as a no-op step
                    dp_opt.step()
                    n_steps += 1
                    continue
                loss = self.loss_fn(gs_model(xb), yb)
                loss.backward()
                if dp_opt.pre_step():  # clip per example, add noise, scale, record step in accountant
                    if regularizer is not None:
                        regularizer.add_gradient(gs_model)  # data-independent; no privacy cost
                    dp_opt.original_optimizer.step()
                losses.append(loss.item())
                n_steps += 1
        self.steps += n_steps
        gs_model.remove_hooks()
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return ClientUpdate(self.client_id, state, self.n_train, float(np.mean(losses)) if losses else float("nan"),
                            time.perf_counter() - start)

    def epsilon(self) -> float:
        return float(self.engine.get_epsilon(self.dp.delta)) if self.steps else 0.0

    def accounting_record(self) -> dict:
        return {"client": self.client_id, "n_train": self.n_train, "sample_rate": self.sample_rate,
                "steps": self.steps, "epsilon": self.epsilon(), "delta": self.dp.delta,
                "noise_multiplier": self.dp.noise_multiplier, "max_grad_norm": self.dp.max_grad_norm,
                "accountant": self.dp.accountant, "poisson_sampling": self.dp.poisson_sampling,
                "privacy_unit": "one training example of this client (example-level DP)"}


def make_dp_clients(clients: list[FederatedClient], dp: DPConfig) -> list[DPFederatedClient]:
    return [DPFederatedClient(c, dp) for c in clients]
