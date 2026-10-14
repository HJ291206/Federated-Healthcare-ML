import numpy as np
import pytest
import torch

from ml.federated.client import FederatedClient, LocalTrainConfig
from ml.federated.fedprox import make_proximal_regularizer, trainable_parameters
from ml.federated.privacy import DPConfig, DPFederatedClient
from ml.models import DiabetesMLP

N = 24


def _client():
    gen = torch.Generator().manual_seed(7)
    X = torch.randn(N, 21, generator=gen).numpy()
    y = (X[:, 0] > 0.2).astype(int)
    return FederatedClient(0, X, y, X[:4], y[:4], pos_weight=3.0)


def _model():
    torch.manual_seed(0)
    return DiabetesMLP(21, (8,), 0.0)


FULL_BATCH = LocalTrainConfig(lr=1.0, batch_size=N, local_epochs=1)


def _manual_clipped_mean_grad(model, client, C):
    """Average of per-example gradients, each clipped to L2 norm C."""
    params = trainable_parameters(model)
    total = [torch.zeros_like(p) for p in params]
    for i in range(N):
        model.zero_grad()
        client.loss_fn(model(client.X_train[i:i + 1]), client.y_train[i:i + 1]).backward()
        grads = [p.grad.detach().clone() for p in params]
        norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
        factor = min(1.0, C / (norm.item() + 1e-6))
        for t, g in zip(total, grads):
            t += g * factor
    return [t / N for t in total]


def test_per_example_clipping_matches_manual_computation():
    C = 0.05  # small enough that every example is clipped
    dp_client = DPFederatedClient(_client(), DPConfig(noise_multiplier=0.0, max_grad_norm=C, poisson_sampling=False))
    model = _model()
    before = [p.detach().clone() for p in trainable_parameters(model)]
    expected = _manual_clipped_mean_grad(_model(), _client(), C)
    upd = dp_client.local_train(model, FULL_BATCH, seed=1)
    after = [upd.state[k] for k in upd.state]
    for b, a, g in zip(before, after, expected):
        torch.testing.assert_close(a, b - FULL_BATCH.lr * g, rtol=1e-4, atol=1e-6)


def test_clipping_actually_limits_the_update():
    big = DPFederatedClient(_client(), DPConfig(0.0, max_grad_norm=1e6, poisson_sampling=False))
    small = DPFederatedClient(_client(), DPConfig(0.0, max_grad_norm=0.01, poisson_sampling=False))
    m1, m2 = _model(), _model()
    init = [p.detach().clone() for p in trainable_parameters(m1)]
    u1, u2 = big.local_train(m1, FULL_BATCH, 1), small.local_train(m2, FULL_BATCH, 1)
    step = lambda u: torch.sqrt(sum(((u.state[k] - i) ** 2).sum() for k, i in zip(u.state, init)))
    assert step(u2) <= FULL_BATCH.lr * 0.01 + 1e-6
    assert step(u1) > step(u2)


def test_noise_is_added_and_reproducible():
    cfg = DPConfig(noise_multiplier=1.0, max_grad_norm=1.0, poisson_sampling=False)
    a = DPFederatedClient(_client(), cfg).local_train(_model(), FULL_BATCH, seed=3)
    b = DPFederatedClient(_client(), cfg).local_train(_model(), FULL_BATCH, seed=3)
    c = DPFederatedClient(_client(), cfg).local_train(_model(), FULL_BATCH, seed=4)
    quiet = DPFederatedClient(_client(), DPConfig(0.0, 1.0, poisson_sampling=False)).local_train(_model(), FULL_BATCH, seed=3)
    for k in a.state:
        assert torch.equal(a.state[k], b.state[k])
    assert any(not torch.equal(a.state[k], c.state[k]) for k in a.state)
    assert any(not torch.equal(a.state[k], quiet.state[k]) for k in a.state)


def test_accountant_tracks_steps_and_epsilon_behaves():
    cfg = LocalTrainConfig(lr=0.1, batch_size=8, local_epochs=1)
    low = DPFederatedClient(_client(), DPConfig(noise_multiplier=1.0, delta=1e-3))
    high = DPFederatedClient(_client(), DPConfig(noise_multiplier=4.0, delta=1e-3))
    assert low.epsilon() == 0.0
    low.local_train(_model(), cfg, seed=0)
    e1, s1 = low.epsilon(), low.steps
    low.local_train(_model(), cfg, seed=1)
    e2 = low.epsilon()
    high.local_train(_model(), cfg, seed=0)
    high.local_train(_model(), cfg, seed=1)
    assert s1 == pytest.approx(N / 8, abs=2)  # Poisson sampling: about n/batch_size steps
    assert 0 < e1 < e2 < float("inf")
    assert high.epsilon() < e2
    rec = low.accounting_record()
    assert rec["steps"] == low.steps and rec["privacy_unit"].startswith("one training example")


def test_proximal_gradient_is_applied_under_dp():
    """With no noise and a huge clip norm, DP-FedProx on a full batch equals plain FedProx SGD."""
    mu = 0.5
    cfg = LocalTrainConfig(lr=0.1, batch_size=N, local_epochs=2)  # 2 steps: the term is zero at the anchor
    plain_model, dp_model = _model(), _model()
    anchor = [p.detach().clone() for p in trainable_parameters(plain_model)]
    plain = _client().local_train(plain_model, cfg, seed=2, regularizer=make_proximal_regularizer(anchor, mu))
    dp_client = DPFederatedClient(_client(), DPConfig(0.0, max_grad_norm=1e9, poisson_sampling=False))
    dp = dp_client.local_train(dp_model, cfg, seed=2, regularizer=make_proximal_regularizer(anchor, mu))
    for k in plain.state:
        torch.testing.assert_close(dp.state[k], plain.state[k], rtol=1e-4, atol=1e-6)
    # and the proximal term really changed the result
    no_prox = DPFederatedClient(_client(), DPConfig(0.0, 1e9, poisson_sampling=False)).local_train(_model(), cfg, seed=2)
    assert any(not torch.allclose(dp.state[k], no_prox.state[k]) for k in dp.state)


def test_dp_client_rejects_opaque_regularizers():
    dp_client = DPFederatedClient(_client(), DPConfig(1.0))
    with pytest.raises(TypeError):
        dp_client.local_train(_model(), FULL_BATCH, 0, regularizer=lambda m: torch.tensor(0.0))


@pytest.mark.parametrize("kw", [{"noise_multiplier": -1.0}, {"noise_multiplier": 1.0, "max_grad_norm": 0.0},
                                {"noise_multiplier": 1.0, "delta": 1.0}])
def test_invalid_dp_config_raises(kw):
    with pytest.raises(ValueError):
        DPFederatedClient(_client(), DPConfig(**kw))
