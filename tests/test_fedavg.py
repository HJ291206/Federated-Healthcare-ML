import pytest
import torch

from ml.federated.client import ClientUpdate
from ml.federated.fedavg import fedavg_aggregate, weighted_average


def _state(w, b):
    return {"w": torch.tensor(w, dtype=torch.float32), "b": torch.tensor(b, dtype=torch.float32),
            "steps": torch.tensor(7, dtype=torch.int64)}


def test_weighted_average_matches_hand_computation():
    s1, s2 = _state([1.0, 2.0], [0.0]), _state([3.0, 6.0], [4.0])
    out = weighted_average([s1, s2], [1, 3])
    torch.testing.assert_close(out["w"], torch.tensor([2.5, 5.0]))
    torch.testing.assert_close(out["b"], torch.tensor([3.0]))
    assert out["w"].dtype == torch.float32
    assert out["steps"].item() == 7  # non-float tensors are copied, not averaged


def test_weights_are_normalized_and_scale_invariant():
    s1, s2 = _state([1.0, 0.0], [1.0]), _state([0.0, 1.0], [3.0])
    a = weighted_average([s1, s2], [10, 30])
    b = weighted_average([s1, s2], [1, 3])
    for k in ("w", "b"):
        torch.testing.assert_close(a[k], b[k])


def test_identical_states_are_a_fixed_point():
    s = _state([0.3, -1.2], [0.7])
    out = weighted_average([s, s, s], [5, 1, 100])
    torch.testing.assert_close(out["w"], s["w"])


def test_fedavg_aggregate_weights_by_sample_count():
    updates = [ClientUpdate(0, _state([0.0, 0.0], [0.0]), 100, 0.0, 0.0),
               ClientUpdate(1, _state([4.0, 8.0], [2.0]), 300, 0.0, 0.0)]
    out = fedavg_aggregate(updates)
    torch.testing.assert_close(out["w"], torch.tensor([3.0, 6.0]))


@pytest.mark.parametrize("weights", [[], [1], [0, 0], [-1, 2]])
def test_invalid_weights_raise(weights):
    states = [_state([1.0, 1.0], [1.0]), _state([1.0, 1.0], [1.0])]
    with pytest.raises(ValueError):
        weighted_average(states, weights)


def test_mismatched_keys_raise():
    with pytest.raises(ValueError):
        weighted_average([_state([1.0, 1.0], [1.0]), {"w": torch.zeros(2)}], [1, 1])
