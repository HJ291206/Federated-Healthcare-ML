import numpy as np
import pandas as pd
import pytest

from ml.data_validation import BINARY_FEATURES, COUNT_FEATURES, FEATURES, ORDINAL_FEATURES, TARGET


@pytest.fixture
def toy_frame() -> pd.DataFrame:
    """Small schema-valid frame for unit tests only; never used for analysis or reported results."""
    rng = np.random.default_rng(0)
    n = 400
    data = {c: rng.integers(0, 2, n) for c in BINARY_FEATURES}
    for c, (lo, hi) in {**ORDINAL_FEATURES, **COUNT_FEATURES}.items():
        data[c] = rng.integers(lo, hi + 1, n)
    data["BMI"] = rng.integers(15, 50, n)
    data[TARGET] = (rng.random(n) < 0.2).astype(int)
    return pd.DataFrame(data)[[TARGET] + FEATURES].astype(float)
