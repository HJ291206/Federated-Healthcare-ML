import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from ml.data_validation import FEATURES, TARGET
from ml.preprocessing import PASSTHROUGH_FEATURES, SCALED_FEATURES, FeaturePreprocessor


def test_feature_groups_cover_features_exactly_once():
    assert sorted(SCALED_FEATURES + PASSTHROUGH_FEATURES) == sorted(FEATURES)
    assert TARGET not in SCALED_FEATURES + PASSTHROUGH_FEATURES


def test_transform_excludes_target_and_keeps_feature_order(toy_frame):
    out = FeaturePreprocessor().fit_transform(toy_frame)  # frame still contains the target column
    assert out.shape == (len(toy_frame), 21)
    assert out.dtype == np.float32
    for i, col in enumerate(FEATURES):
        if col in PASSTHROUGH_FEATURES:
            np.testing.assert_array_equal(out[:, i], toy_frame[col].to_numpy(np.float32))


def test_scaler_statistics_come_from_training_rows_only(toy_frame):
    train, holdout = toy_frame.iloc[:300], toy_frame.iloc[300:].copy()
    holdout["BMI"] += 100  # a shift that would move the mean if holdout rows leaked into fit()
    pre = FeaturePreprocessor().fit(train)
    bmi = SCALED_FEATURES.index("BMI")
    assert pre.scaler.mean_[bmi] == pytest.approx(train["BMI"].mean())
    scaled_train = pre.transform(train)[:, [FEATURES.index(c) for c in SCALED_FEATURES]]
    np.testing.assert_allclose(scaled_train.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(scaled_train.std(axis=0), 1, atol=1e-4)
    assert pre.transform(holdout)[:, FEATURES.index("BMI")].mean() > 5


def test_unfitted_and_missing_columns_raise(toy_frame):
    with pytest.raises(NotFittedError):
        FeaturePreprocessor().transform(toy_frame)
    with pytest.raises(KeyError):
        FeaturePreprocessor().fit(toy_frame.drop(columns=["BMI"]))


def test_sufficient_statistics_match_pooled_fit(toy_frame):
    parts = [toy_frame.iloc[:50], toy_frame.iloc[50:260], toy_frame.iloc[260:]]
    federated = FeaturePreprocessor.from_sufficient_statistics(
        [FeaturePreprocessor.sufficient_statistics(p) for p in parts])
    pooled = FeaturePreprocessor().fit(toy_frame)
    np.testing.assert_allclose(federated.scaler.mean_, pooled.scaler.mean_, rtol=1e-12)
    np.testing.assert_allclose(federated.scaler.scale_, pooled.scaler.scale_, rtol=1e-9)
    np.testing.assert_allclose(federated.transform(toy_frame), pooled.transform(toy_frame), atol=1e-6)


def test_save_load_roundtrip(toy_frame, tmp_path):
    pre = FeaturePreprocessor().fit(toy_frame)
    pre.save(tmp_path / "pre.joblib")
    loaded = FeaturePreprocessor.load(tmp_path / "pre.joblib")
    np.testing.assert_array_equal(loaded.transform(toy_frame), pre.transform(toy_frame))
