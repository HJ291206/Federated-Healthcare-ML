import numpy as np
import pytest

from ml.data import DataSplit, features_and_target, stratified_split
from ml.data_validation import FEATURES, TARGET


@pytest.fixture
def labels():
    rng = np.random.default_rng(1)
    return (rng.random(10_000) < 0.14).astype(int)


def test_split_is_disjoint_and_complete(labels):
    s = stratified_split(labels, 0.15, 0.15, seed=42)
    all_idx = np.concatenate([s.train, s.val, s.test])
    assert len(all_idx) == len(np.unique(all_idx)) == len(labels)
    assert not set(s.test) & set(s.train) and not set(s.test) & set(s.val) and not set(s.val) & set(s.train)
    assert s.sizes() == {"train": 7000, "val": 1500, "test": 1500}


def test_split_is_stratified(labels):
    s = stratified_split(labels, 0.15, 0.15, seed=42)
    for part in (s.train, s.val, s.test):
        assert labels[part].mean() == pytest.approx(labels.mean(), abs=0.002)


def test_split_is_reproducible(labels, tmp_path):
    a, b = stratified_split(labels, 0.15, 0.15, 7), stratified_split(labels, 0.15, 0.15, 7)
    c = stratified_split(labels, 0.15, 0.15, 8)
    np.testing.assert_array_equal(a.test, b.test)
    assert not np.array_equal(a.test, c.test)
    a.save(tmp_path / "s.npz")
    np.testing.assert_array_equal(DataSplit.load(tmp_path / "s.npz").train, a.train)


@pytest.mark.parametrize("test_size,val_size", [(0, 0.1), (0.5, 0.5), (1.2, 0.1)])
def test_invalid_split_sizes_raise(labels, test_size, val_size):
    with pytest.raises(ValueError):
        stratified_split(labels, test_size, val_size, 0)


def test_features_and_target_excludes_target(toy_frame):
    X, y = features_and_target(toy_frame)
    assert list(X.columns) == FEATURES and TARGET not in X.columns
    assert y.dtype == np.int64


def test_non_binary_target_raises(toy_frame):
    bad = toy_frame.copy()
    bad.loc[0, TARGET] = 2
    with pytest.raises(ValueError):
        features_and_target(bad)
