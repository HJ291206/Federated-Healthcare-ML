"""Loading and stratified splitting of the primary dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data_validation import FEATURES, PRIMARY_FILE, TARGET, load_csv


@dataclass(frozen=True)
class DataSplit:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, train=self.train, val=self.val, test=self.test)

    @classmethod
    def load(cls, path: Path) -> "DataSplit":
        with np.load(path) as f:
            return cls(train=f["train"], val=f["val"], test=f["test"])


def load_primary(file_name: str = PRIMARY_FILE) -> pd.DataFrame:
    return load_csv(file_name)


def features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the 21 input features (target excluded) and the integer target."""
    if TARGET not in df.columns:
        raise KeyError(f"target column {TARGET!r} missing")
    X = df[FEATURES].copy()
    y = df[TARGET].to_numpy().astype(np.int64)
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError(f"{TARGET} must be binary, got {np.unique(y)}")
    return X, y


def stratified_split(y: np.ndarray, test_size: float, val_size: float, seed: int) -> DataSplit:
    """Stratified train/validation/test split of row indices.

    `test_size` and `val_size` are fractions of the full dataset.
    """
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be in (0, 1) and sum to < 1")
    idx = np.arange(len(y))
    trainval, test = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    train, val = train_test_split(trainval, test_size=val_size / (1 - test_size),
                                  stratify=y[trainval], random_state=seed)
    return DataSplit(train=np.sort(train), val=np.sort(val), test=np.sort(test))
