"""Tests for the optional JAX 1D-CNN module (skipped if JAX/Flax are not installed)."""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = jax.numpy

from ml.jax_cnn.experiment import fit_jax, weighted_bce  # noqa: E402
from ml.jax_cnn.model import CNN1D, MLP, count_parameters  # noqa: E402


def test_weighted_bce_matches_torch_pos_weight():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=64).astype(np.float32)
    y = rng.integers(0, 2, size=64).astype(np.float32)
    ours = float(weighted_bce(jnp.asarray(logits), jnp.asarray(y), 6.18))
    ref = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(6.18))(torch.tensor(logits), torch.tensor(y)).item()
    assert ours == pytest.approx(ref, rel=1e-5)


def test_model_shapes_and_param_counts():
    x = jnp.zeros((5, 21), dtype=jnp.float32)
    key = jax.random.PRNGKey(0)
    for model in (CNN1D(channels=(4, 8), kernel_size=3, hidden=8), MLP(hidden_sizes=(64, 32))):
        params = model.init(key, x, train=False)
        out = model.apply(params, x, train=False)
        assert out.shape == (5,)
        assert count_parameters(params) > 0
    mlp_params = MLP(hidden_sizes=(64, 32)).init(key, x, train=False)
    assert count_parameters(mlp_params) == 21 * 64 + 64 + 64 * 32 + 32 + 32 + 1  # same as the PyTorch MLP (3,521)


def test_feature_permutation_changes_cnn_but_not_mlp():
    """Conv kernels couple neighbouring columns, so column order matters for the CNN only."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(8, 21)).astype(np.float32)
    perm = rng.permutation(21)
    key = jax.random.PRNGKey(0)
    cnn = CNN1D(channels=(4, 8), kernel_size=3, hidden=8)
    p = cnn.init(key, jnp.asarray(x), train=False)
    a = cnn.apply(p, jnp.asarray(x), train=False)
    b = cnn.apply(p, jnp.asarray(x[:, perm]), train=False)
    assert not np.allclose(np.asarray(a), np.asarray(b))
    # An MLP with permuted first-layer weights is exactly invariant.
    mlp = MLP(hidden_sizes=(8,))
    q = mlp.init(key, jnp.asarray(x), train=False)
    q_perm = jax.tree_util.tree_map(lambda v: v, q)
    q_perm["params"]["Dense_0"]["kernel"] = q["params"]["Dense_0"]["kernel"][perm]
    assert np.allclose(np.asarray(mlp.apply(q, jnp.asarray(x), train=False)),
                       np.asarray(mlp.apply(q_perm, jnp.asarray(x[:, perm]), train=False)), atol=1e-5)


def test_fit_jax_runs_and_is_deterministic():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 21)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.normal(size=300) > 0.8).astype(np.float32)
    cfg = {"lr": 0.01, "weight_decay": 0.0, "batch_size": 64, "max_epochs": 3, "patience": 3}
    outs = [fit_jax(CNN1D(channels=(4, 8), kernel_size=3, hidden=8), X[:200], y[:200], X[200:], y[200:], cfg, seed=0)
            for _ in range(2)]
    p0, p1 = (o[2]["predict"](X[200:]) for o in outs)
    assert p0.shape == (100,) and np.all((p0 >= 0) & (p0 <= 1))
    assert np.array_equal(p0, p1)
    assert outs[0][2]["epochs_run"] == 3 and len(outs[0][1]) == 3
