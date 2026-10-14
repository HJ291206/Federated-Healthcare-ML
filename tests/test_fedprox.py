import copy

import pytest
import torch

from ml.federated.client import FederatedClient, LocalTrainConfig
from ml.federated.fedprox import fedprox_local_loss, make_proximal_regularizer, proximal_term, trainable_parameters
from ml.models import DiabetesMLP


def _models():
    torch.manual_seed(0)
    global_model = DiabetesMLP(21, (8,), 0.0)
    local = copy.deepcopy(global_model)
    with torch.no_grad():
        for p in local.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return global_model, local


def test_proximal_term_matches_manual_formula():
    g, l = _models()
    gp = [p.detach() for p in trainable_parameters(g)]
    mu = 0.37
    manual = 0.5 * mu * sum(((a - b) ** 2).sum() for a, b in zip(l.parameters(), g.parameters()))
    torch.testing.assert_close(proximal_term(l, gp, mu), manual)


def test_proximal_term_is_zero_at_global_weights_and_for_mu_zero():
    g, l = _models()
    gp = [p.detach() for p in trainable_parameters(g)]
    assert proximal_term(g, gp, 1.0).item() == 0.0
    assert proximal_term(l, gp, 0.0).item() == 0.0


def test_proximal_gradient_is_mu_times_difference():
    g, l = _models()
    gp = [p.detach().clone() for p in trainable_parameters(g)]
    mu = 0.5
    proximal_term(l, gp, mu).backward()
    for p, q in zip(l.parameters(), gp):
        torch.testing.assert_close(p.grad, mu * (p.detach() - q))


def test_local_loss_adds_proximal_term():
    g, l = _models()
    gp = [p.detach() for p in trainable_parameters(g)]
    ce = torch.tensor(0.8)
    torch.testing.assert_close(fedprox_local_loss(ce, l, gp, 0.2), ce + proximal_term(l, gp, 0.2))


def test_regularizer_freezes_global_copy():
    g, l = _models()
    gp = [p for p in trainable_parameters(g)]
    reg = make_proximal_regularizer(gp, 1.0)
    before = reg(l).item()
    with torch.no_grad():
        for p in g.parameters():
            p.add_(1.0)  # mutating the source must not change the anchor
    assert reg(l).item() == pytest.approx(before)


def test_mismatched_parameters_raise():
    _, l = _models()
    with pytest.raises(ValueError):
        proximal_term(l, [torch.zeros(1)], 1.0)


def _client():
    gen = torch.Generator().manual_seed(1)
    X = torch.randn(200, 21, generator=gen).numpy()
    y = (X[:, 0] > 0.3).astype(int)
    return FederatedClient(0, X, y, X[:20], y[:20], pos_weight=2.0)


def _train(mu):
    torch.manual_seed(0)
    model = DiabetesMLP(21, (8,), 0.1)
    anchor = [p.detach().clone() for p in model.parameters()]
    reg = None if mu is None else make_proximal_regularizer(anchor, mu)
    upd = _client().local_train(model, LocalTrainConfig(lr=0.1, batch_size=32, local_epochs=3), seed=5, regularizer=reg)
    drift = sum(((upd.state[k] - a) ** 2).sum() for k, a in zip(upd.state, anchor)).sqrt().item()
    return upd, drift


def test_fedprox_with_mu_zero_equals_fedavg_update():
    plain, _ = _train(None)
    prox0, _ = _train(0.0)
    for k in plain.state:
        assert torch.equal(plain.state[k], prox0.state[k])


def test_larger_mu_keeps_local_model_closer_to_global():
    drifts = [_train(mu)[1] for mu in (0.0, 1.0, 10.0)]
    assert drifts[0] > drifts[1] > drifts[2]
