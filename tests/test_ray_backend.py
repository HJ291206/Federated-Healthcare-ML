import functools

import numpy as np
import pytest
import torch

ray = pytest.importorskip("ray")

from ml.federated.client import FederatedClient, LocalTrainConfig  # noqa: E402
from ml.federated.fedavg import fedavg_aggregate  # noqa: E402
from ml.federated.fedprox import make_proximal_regularizer  # noqa: E402
from ml.federated.ray_backend import RayFederatedServer  # noqa: E402
from ml.federated.server import FederatedServer  # noqa: E402
from ml.models import build_mlp  # noqa: E402

CFG = LocalTrainConfig(lr=0.1, batch_size=32, local_epochs=1)
# Must be importable by Ray worker processes (a function defined in this test module is not).
model_fn = functools.partial(build_mlp, 21, (8,), 0.1)


def make_clients():
    rng = np.random.default_rng(0)
    out = []
    for k, n in enumerate((120, 60, 220)):
        X = rng.normal(size=(n, 21)).astype(np.float32)
        y = (X[:, 0] + 0.3 * k > 0.2).astype(int)
        out.append(FederatedClient(k, X, y, X[:10], y[:10], pos_weight=1.5))
    return out


@pytest.fixture(scope="module")
def ray_session():
    ray.init(num_cpus=3, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False, configure_logging=False)
    yield
    ray.shutdown()


@pytest.mark.parametrize("mu", [None, 0.05])
def test_ray_backend_matches_sequential_exactly(ray_session, mu):
    factory = None if mu is None else (lambda gp: make_proximal_regularizer(gp, mu))
    seq = FederatedServer(model_fn, make_clients(), CFG, 1.0, 3, fedavg_aggregate, factory)
    seq.fit(3)
    with RayFederatedServer(model_fn, make_clients(), CFG, 1.0, 3, fedavg_aggregate, factory) as par:
        hist = par.fit(3)
        assert len(par.actors) == 3
    assert par.actors == []
    assert ray.is_initialized()  # the fixture started Ray, so the server must not shut it down
    for k, v in seq.global_state().items():
        assert torch.equal(v, par.global_state()[k]), k
    assert hist[-1]["cumulative_bytes"] == seq.history[-1]["cumulative_bytes"]
    assert hist[-1]["n_samples"] == 400


def test_partial_participation_on_ray(ray_session):
    with RayFederatedServer(model_fn, make_clients(), CFG, 0.67, 1, fedavg_aggregate) as par:
        rec = par.run_round()
    assert rec["n_clients"] == 2
