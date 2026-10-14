"""Feature preprocessing shared by the centralized and federated pipelines.

Binary indicators pass through unchanged; ordinal, count and continuous features are
standardized. The scaler can be fitted either on a DataFrame (centralized) or from
per-client sufficient statistics (federated), so no client has to share raw rows.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from ml.data_validation import (
    BINARY_FEATURES, CONTINUOUS_FEATURES, COUNT_FEATURES, FEATURES, ORDINAL_FEATURES, TARGET,
)

SCALED_FEATURES = [*CONTINUOUS_FEATURES, *ORDINAL_FEATURES, *COUNT_FEATURES]
PASSTHROUGH_FEATURES = list(BINARY_FEATURES)


class FeaturePreprocessor:
    def __init__(self) -> None:
        self.feature_names: list[str] = list(FEATURES)
        self.scaled_features: list[str] = list(SCALED_FEATURES)
        self.passthrough_features: list[str] = list(PASSTHROUGH_FEATURES)
        self._scaled_idx = [self.feature_names.index(c) for c in self.scaled_features]
        self.scaler: StandardScaler | None = None

    @staticmethod
    def _check_columns(X: pd.DataFrame) -> None:
        missing = [c for c in FEATURES if c not in X.columns]
        if missing:
            raise KeyError(f"missing feature columns: {missing}")

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        self._check_columns(X)
        self.scaler = StandardScaler().fit(X[self.scaled_features].to_numpy(dtype=np.float64))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Return a float32 array with columns in `feature_names` order; the target is never included."""
        if self.scaler is None:
            raise NotFittedError("FeaturePreprocessor is not fitted")
        self._check_columns(X)
        out = X[self.feature_names].to_numpy(dtype=np.float64, copy=True)
        out[:, self._scaled_idx] = self.scaler.transform(out[:, self._scaled_idx])
        return out.astype(np.float32)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @staticmethod
    def sufficient_statistics(X: pd.DataFrame) -> dict[str, np.ndarray | int]:
        """Count, sum and sum of squares of the scaled features, computable locally by a client."""
        values = X[SCALED_FEATURES].to_numpy(dtype=np.float64)
        return {"n": int(values.shape[0]), "sum": values.sum(axis=0), "sum_sq": (values ** 2).sum(axis=0)}

    @classmethod
    def from_sufficient_statistics(cls, stats: list[dict]) -> "FeaturePreprocessor":
        """Build a preprocessor equivalent to fitting on the union of the clients' rows."""
        n = sum(s["n"] for s in stats)
        if n == 0:
            raise ValueError("no samples in statistics")
        mean = sum(s["sum"] for s in stats) / n
        var = np.maximum(sum(s["sum_sq"] for s in stats) / n - mean ** 2, 0.0)
        scaler = StandardScaler()
        scaler.mean_ = mean
        scaler.var_ = var
        scaler.scale_ = np.where(var > 0, np.sqrt(var), 1.0)
        scaler.n_samples_seen_ = n
        scaler.n_features_in_ = len(SCALED_FEATURES)
        pre = cls()
        pre.scaler = scaler
        return pre

    def to_dict(self) -> dict:
        if self.scaler is None:
            raise NotFittedError("FeaturePreprocessor is not fitted")
        return {
            "feature_names": self.feature_names,
            "scaled_features": self.scaled_features,
            "passthrough_features": self.passthrough_features,
            "mean": dict(zip(self.scaled_features, self.scaler.mean_.round(6).tolist())),
            "scale": dict(zip(self.scaled_features, self.scaler.scale_.round(6).tolist())),
            "n_samples_seen": int(self.scaler.n_samples_seen_),
            "target_excluded": TARGET,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "FeaturePreprocessor":
        return joblib.load(path)
