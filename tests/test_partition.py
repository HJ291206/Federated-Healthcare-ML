import numpy as np
import pytest

from ml.data import stratified_split
from ml.federated.partition import make_partition, partition_dirichlet, partition_iid, partition_summary


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    y = (rng.random(20_000) < 0.14).astype(int)
    split = stratified_split(y, 0.15, 0.15, seed=42)
    return y, split


def _all(parts, attr):
    return np.concatenate([getattr(c, attr) for c in parts.clients])


@pytest.mark.parametrize("make", [
    lambda s, y: partition_iid(s.train, s.val, y, 5, seed=1),
    lambda s, y: partition_dirichlet(s.train, s.val, y, 5, alpha=0.3, seed=1),
])
def test_partitions_are_disjoint_complete_and_exclude_test(setup, make):
    y, s = setup
    p = make(s, y)
    assert len(p.clients) == 5
    tr, va = _all(p, "train_idx"), _all(p, "val_idx")
    assert len(tr) == len(np.unique(tr)) and set(tr) == set(s.train)
    assert len(va) == len(np.unique(va)) and set(va) == set(s.val)
    assert not (set(tr) | set(va)) & set(s.test)


def test_iid_clients_match_global_class_rate_and_size(setup):
    y, s = setup
    p = partition_iid(s.train, s.val, y, 5, seed=1)
    sizes = [len(c.train_idx) for c in p.clients]
    assert max(sizes) - min(sizes) <= 2
    for c in p.clients:
        assert y[c.train_idx].mean() == pytest.approx(y[s.train].mean(), abs=0.003)


def test_partition_is_reproducible(setup):
    y, s = setup
    a = partition_dirichlet(s.train, s.val, y, 5, 0.5, seed=3)
    b = partition_dirichlet(s.train, s.val, y, 5, 0.5, seed=3)
    c = partition_dirichlet(s.train, s.val, y, 5, 0.5, seed=4)
    for ca, cb in zip(a.clients, b.clients):
        np.testing.assert_array_equal(ca.train_idx, cb.train_idx)
    assert any(not np.array_equal(ca.train_idx, cc.train_idx) for ca, cc in zip(a.clients, c.clients))


def test_smaller_alpha_gives_more_label_skew(setup):
    y, s = setup

    def spread(alpha):
        rates = []
        for seed in range(5):
            p = partition_dirichlet(s.train, s.val, y, 5, alpha, seed=seed, min_train_per_class=1, min_val_per_class=0)
            rates.append(np.std([y[c.train_idx].mean() for c in p.clients]))
        return np.mean(rates)

    assert spread(0.1) > spread(1.0) > spread(100.0)


def test_minimum_class_counts_are_enforced(setup):
    y, s = setup
    p = partition_dirichlet(s.train, s.val, y, 5, 0.1, seed=0, min_train_per_class=15, min_val_per_class=3)
    for c in p.clients:
        assert np.bincount(y[c.train_idx], minlength=2).min() >= 15
        assert np.bincount(y[c.val_idx], minlength=2).min() >= 3


def test_summary_reports_counts(setup):
    y, s = setup
    p = make_partition({"method": "iid"}, s.train, s.val, y, 5, seed=0)
    summ = partition_summary(p, y)
    assert set(summ.columns) >= {"client", "split", "n", "n_positive", "positive_rate"}
    assert summ[summ.split == "train"].n.sum() == len(s.train)
    assert (summ.n_positive + summ.n_negative == summ.n).all()


def test_invalid_inputs_raise(setup):
    y, s = setup
    with pytest.raises(ValueError):
        partition_dirichlet(s.train, s.val, y, 5, alpha=0, seed=0)
    with pytest.raises(ValueError):
        make_partition({"method": "nope"}, s.train, s.val, y, 5, 0)
