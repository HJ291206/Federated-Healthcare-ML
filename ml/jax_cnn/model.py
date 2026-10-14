"""JAX/Flax models for the tabular diabetes features.

The 21 BRFSS features have no spatial or sequential structure: their column order is an
arbitrary convention of the CSV. A 1D convolution over them therefore mixes *neighbouring
columns* (e.g. BMI next to Smoker), which has no medical meaning. The CNN here is an
experimental representation used to test whether that inductive bias helps or hurts on
tabular data, not an image-style CNN. A JAX MLP with the PyTorch MLP's architecture is
included as the control that separates the framework from the architecture.
"""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class CNN1D(nn.Module):
    channels: Sequence[int] = (16, 32)
    kernel_size: int = 3
    hidden: int = 32
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        h = x[..., None]  # (batch, 21 features treated as positions, 1 channel)
        for c in self.channels:
            h = nn.relu(nn.Conv(c, kernel_size=(self.kernel_size,), padding="SAME")(h))
        h = h.mean(axis=1)  # global average pooling over the 21 positions
        h = nn.relu(nn.Dense(self.hidden)(h))
        h = nn.Dropout(self.dropout, deterministic=not train)(h)
        return nn.Dense(1)(h).squeeze(-1)


class MLP(nn.Module):
    """Same architecture as ml.models.DiabetesMLP (21 -> 64 -> 32 -> 1)."""
    hidden_sizes: Sequence[int] = (64, 32)
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        h = x
        for width in self.hidden_sizes:
            h = nn.relu(nn.Dense(width)(h))
            h = nn.Dropout(self.dropout, deterministic=not train)(h)
        return nn.Dense(1)(h).squeeze(-1)


def count_parameters(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
