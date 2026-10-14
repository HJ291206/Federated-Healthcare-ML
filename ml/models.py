"""Model definitions shared by the centralized and federated pipelines."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn


class DiabetesMLP(nn.Module):
    """Feed-forward binary classifier returning one logit per row."""

    def __init__(self, n_features: int = 21, hidden_sizes: Sequence[int] = (64, 32), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.config = {"n_features": n_features, "hidden_sizes": list(hidden_sizes), "dropout": dropout}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_mlp(n_features: int, hidden_sizes: Sequence[int], dropout: float) -> DiabetesMLP:
    return DiabetesMLP(n_features=n_features, hidden_sizes=hidden_sizes, dropout=dropout)


def load_mlp(path) -> DiabetesMLP:
    """Load a checkpoint written by train_centralized / federated experiment ({'state_dict', 'config', ...})."""
    ckpt = torch.load(path, weights_only=False)
    model = DiabetesMLP(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_logistic_regression(C: float, seed: int, max_iter: int = 2000) -> LogisticRegression:
    # class_weight="balanced" derives weights from the labels passed to fit(), i.e. training data only.
    return LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter, random_state=seed)


def build_random_forest(seed: int, n_estimators: int, max_depth: int | None, min_samples_leaf: int,
                        max_features: str | float = "sqrt", n_jobs: int = -1) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        max_features=max_features, class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed,
    )
