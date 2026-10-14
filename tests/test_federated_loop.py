import numpy as np
import pytest
import torch

from ml.federated.client import FederatedClient, LocalTrainConfig, derive_seed
from ml.federated.fedavg import build_fedavg_server
from ml.federated.fedprox import build_fedprox_server
from ml.federated.server import state_nbytes
from ml.models import DiabetesMLP

CFG = LocalTrainConfig(lr=0.1, batch_size=32, local_epochs=1)


def model_fn():
    return DiabetesMLP(21, (8,), 0.0)


def make_clients(sizes=(120, 60, 220)):
    rng = np.random.default_rng(0)
    clients = []
    for k, n in enumerate(sizes):
        X = rng.normal(size=(n, 21)).astype(np.float32)
        y = (X[:, 0] + 0.3 * k > 0.2).astype(int)
        clients.append(FederatedClient(k, X, y, X[:10], y[:10], pos_weight=1.5))
    return clients


def evaluator(model):
    return {"weight_norm": float(sum(p.detach().norm() ** 2 for p in model.parameters()) ** 0.5)}


def test_history_has_one_record_per_round_with_byte_accounting():
    clients = make_clients()
    server = build_fedavg_server(model_fn, clients, CFG, client_fraction=1.0, seed=0)
    hist = server.fit(4, evaluator)
    payload = state_nbytes(model_fn().state_dict())
    assert [h["round"] for h in hist] == [1, 2, 3, 4]
    assert all("weight_norm" in h for h in hist)
    assert all(h["bytes_down"] == h["bytes_up"] == 3 * payload for h in hist)
    assert hist[-1]["cumulative_bytes"] == 4 * 2 * 3 * payload
    assert payload == 4 * sum(p.numel() for p in model_fn().parameters())


def test_checkpoint_resume_reproduces_uninterrupted_run(tmp_path):
    full = build_fedavg_server(model_fn, make_clients(), CFG, 1.0, seed=5)
    full.fit(5, evaluator, checkpoint_dir=tmp_path, checkpoint_every=2)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fedavg_round0002.pt", "fedavg_round0004.pt", "fedavg_round0005.pt"]

    resumed = build_fedavg_server(model_fn, make_clients(), CFG, 1.0, seed=5)
    assert resumed.load_checkpoint(tmp_path / "fedavg_round0002.pt") == 2
    assert len(resumed.history) == 2
    resumed.fit(3, evaluator)
    assert resumed.round == 5 and len(resumed.history) == 5
    assert resumed.history[-1]["cumulative_bytes"] == full.history[-1]["cumulative_bytes"]
    for k, v in full.global_state().items():
        assert torch.equal(v, resumed.global_state()[k])


def test_training_is_deterministic():
    a = build_fedavg_server(model_fn, make_clients(), CFG, 1.0, seed=3)
    b = build_fedavg_server(model_fn, make_clients(), CFG, 1.0, seed=3)
    a.fit(3), b.fit(3)
    for k, v in a.global_state().items():
        assert torch.equal(v, b.global_state()[k])


def test_fedavg_and_fedprox_start_from_same_initialization():
    a = build_fedavg_server(model_fn, make_clients(), CFG, 1.0, seed=9)
    b = build_fedprox_server(model_fn, make_clients(), CFG, 1.0, seed=9, mu=0.1)
    for k, v in a.global_state().items():
        assert torch.equal(v, b.global_state()[k])


def test_fedprox_mu_zero_reproduces_fedavg_exactly():
    a = build_fedavg_server(model_fn, make_clients(), CFG, 1.0, seed=2)
    b = build_fedprox_server(model_fn, make_clients(), CFG, 1.0, seed=2, mu=0.0)
    a.fit(3), b.fit(3)
    for k, v in a.global_state().items():
        assert torch.equal(v, b.global_state()[k])


def test_single_round_equals_manual_weighted_average_of_local_training():
    clients = make_clients()
    server = build_fedavg_server(model_fn, clients, CFG, 1.0, seed=4)
    init = server.global_state()
    expected = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in init.items()}
    total = sum(c.n_train for c in clients)
    for c in clients:
        m = model_fn()
        m.load_state_dict(init)
        upd = c.local_train(m, CFG, derive_seed(4, 1, c.client_id))
        for k in expected:
            expected[k] += upd.state[k].double() * c.n_train / total
    server.run_round()
    for k, v in server.global_state().items():
        torch.testing.assert_close(v, expected[k].float())


def test_partial_participation_samples_requested_fraction():
    server = build_fedavg_server(model_fn, make_clients((50,) * 5), CFG, client_fraction=0.6, seed=0)
    picks = [server.sample_clients(r) for r in range(1, 21)]
    assert all(len(p) == 3 and len(set(p)) == 3 for p in picks)
    assert len({tuple(p) for p in picks}) > 1
    assert picks == [server.sample_clients(r) for r in range(1, 21)]
    rec = server.run_round()
    assert rec["n_clients"] == 3 and rec["bytes_up"] == 3 * state_nbytes(server.global_state())


def test_federated_training_reduces_loss_on_learnable_task():
    clients = make_clients()
    server = build_fedavg_server(model_fn, clients, LocalTrainConfig(lr=0.1, batch_size=32, local_epochs=2), 1.0, seed=0)
    hist = server.fit(8)
    assert hist[-1]["client_train_loss"] < hist[0]["client_train_loss"]


@pytest.mark.parametrize("fraction", [0.0, 1.5])
def test_invalid_fraction_raises(fraction):
    with pytest.raises(ValueError):
        build_fedavg_server(model_fn, make_clients(), CFG, fraction, seed=0)
