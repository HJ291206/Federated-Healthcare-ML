import numpy as np
import pytest
import torch

from ml.models import DiabetesMLP, count_parameters
from ml.seed import set_global_seed, torch_generator
from ml.training import make_loss, positive_class_weight, predict_proba, train_one_epoch


def test_positive_class_weight_uses_given_labels_only():
    assert positive_class_weight(np.array([0, 0, 0, 1])) == 3.0
    with pytest.raises(ValueError):
        positive_class_weight(np.zeros(5, dtype=int))


def test_mlp_shapes_and_parameter_count():
    model = DiabetesMLP(21, (64, 32), 0.1)
    assert model(torch.zeros(7, 21)).shape == (7,)
    assert count_parameters(model) == 21 * 64 + 64 + 64 * 32 + 32 + 32 + 1
    p = predict_proba(model, np.zeros((5, 21), dtype=np.float32))
    assert p.shape == (5,) and ((0 <= p) & (p <= 1)).all()


def _one_epoch(seed, regularizer=None):
    set_global_seed(seed)
    X = torch.randn(256, 21, generator=torch_generator(99))
    y = (X[:, 0] > 0).float()
    model = DiabetesMLP(21, (8,), 0.0)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = train_one_epoch(model, X, y, opt, make_loss(1.0), 32, torch_generator(seed), regularizer)
    return loss, torch.cat([p.detach().flatten() for p in model.parameters()])


def test_train_one_epoch_is_deterministic():
    l1, w1 = _one_epoch(0)
    l2, w2 = _one_epoch(0)
    assert l1 == l2 and torch.equal(w1, w2)


def test_regularizer_changes_training():
    _, w_plain = _one_epoch(0)
    _, w_reg = _one_epoch(0, regularizer=lambda m: 1.0 * sum((p ** 2).sum() for p in m.parameters()))
    assert w_reg.norm() < w_plain.norm()
