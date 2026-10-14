"""Demo inference on the saved models with strict input validation.

The request contains the 21 BRFSS indicator codes typed by the user; nothing is stored or forwarded, and no
training record is ever returned. Outputs are educational model outputs, not a diagnosis.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field, field_validator

from ml.data_validation import BINARY_FEATURES, CONTINUOUS_FEATURES, COUNT_FEATURES, FEATURES, ORDINAL_FEATURES
from ml.models import load_mlp
from ml.preprocessing import FeaturePreprocessor
from ml.training import predict_proba

ModelName = Literal["centralized_mlp", "centralized_logistic_regression", "centralized_random_forest",
                    "iid_fedavg", "dirichlet_a0.5_fedavg", "dirichlet_a0.5_fedprox_mu0.01"]

# BMI codes in the file range 12-98; accept the plausible adult range and reject anything outside it.
BMI_RANGE = (10, 100)

FEATURE_SCHEMA = (
    [{"name": f, "kind": "binary", "min": 0, "max": 1} for f in BINARY_FEATURES]
    + [{"name": f, "kind": "ordinal", "min": lo, "max": hi} for f, (lo, hi) in ORDINAL_FEATURES.items()]
    + [{"name": f, "kind": "count", "min": lo, "max": hi} for f, (lo, hi) in COUNT_FEATURES.items()]
    + [{"name": f, "kind": "continuous", "min": BMI_RANGE[0], "max": BMI_RANGE[1]} for f in CONTINUOUS_FEATURES]
)
_SCHEMA_BY_NAME = {s["name"]: s for s in FEATURE_SCHEMA}


class PredictRequest(BaseModel):
    model: ModelName = "centralized_mlp"
    features: dict[str, float] = Field(..., description="All 21 BRFSS indicator codes, keyed by feature name.")

    @field_validator("features")
    @classmethod
    def check_features(cls, feats: dict[str, float]) -> dict[str, float]:
        missing = [f for f in FEATURES if f not in feats]
        unknown = [f for f in feats if f not in _SCHEMA_BY_NAME]
        if missing or unknown:
            raise ValueError(f"missing features: {missing}; unknown features: {unknown}")
        errors = []
        for name, value in feats.items():
            s = _SCHEMA_BY_NAME[name]
            if value is None or not np.isfinite(value):
                errors.append(f"{name}: value is missing or not a number")
            elif not (s["min"] <= value <= s["max"]):
                errors.append(f"{name}: {value} outside allowed range [{s['min']}, {s['max']}]")
            elif s["kind"] != "continuous" and float(value) != int(value):
                errors.append(f"{name}: must be an integer code, got {value}")
        if errors:
            raise ValueError("; ".join(errors))
        return feats


class PredictResponse(BaseModel):
    model: str
    probability: float
    threshold: float
    threshold_source: str
    predicted_class: int
    predicted_label: str
    disclaimer: str


class ModelRegistry:
    """Loads saved models lazily. Thresholds come from the saved validation-chosen operating points."""

    def __init__(self, models_dir: Path, results_dir: Path, operating_point: str = "val_max_f1"):
        self.models_dir, self.results_dir, self.op = Path(models_dir), Path(results_dir), operating_point

    def available_models(self) -> list[str]:
        out = []
        pre_ok = (self.models_dir / "centralized" / "preprocessor.joblib").is_file()
        if pre_ok:
            for name, rel in (("centralized_mlp", "centralized/mlp_seed0.pt"),
                              ("centralized_logistic_regression", "centralized/logistic_regression_seed0.joblib"),
                              ("centralized_random_forest", "centralized/random_forest_seed0.joblib")):
                if (self.models_dir / rel).is_file():
                    out.append(name)
            for name in ("iid_fedavg", "dirichlet_a0.5_fedavg", "dirichlet_a0.5_fedprox_mu0.01"):
                if self._federated_path(name) is not None:
                    out.append(name)
        return out

    def _federated_path(self, name: str) -> Path | None:
        for tag in ("final", "fast"):
            p = self.models_dir / "federated" / tag / f"{name}_seed0.pt"
            if p.is_file():
                return p
        return None

    @lru_cache(maxsize=8)
    def _load(self, name: str):
        pre = FeaturePreprocessor.load(self.models_dir / "centralized" / "preprocessor.joblib")
        if name == "centralized_mlp" or not name.startswith("centralized_"):
            path = self.models_dir / "centralized" / "mlp_seed0.pt" if name == "centralized_mlp" else self._federated_path(name)
            if path is None or not path.is_file():
                raise FileNotFoundError(name)
            model = load_mlp(path)
            return pre, (lambda X: predict_proba(model, X))
        est = joblib.load(self.models_dir / "centralized" / f"{name.removeprefix('centralized_')}_seed0.joblib")
        return pre, (lambda X: est.predict_proba(X)[:, 1])

    def threshold(self, name: str) -> tuple[float, str]:
        if name.startswith("centralized_"):
            path = self.results_dir / "centralized_results.csv"
            if path.is_file():
                df = pd.read_csv(path)
                row = df[(df.model == name.removeprefix("centralized_")) & (df.seed == 0) & (df.split == "val")
                         & (df.operating_point == self.op)]
                if len(row):
                    return float(row.threshold.iloc[0]), f"{self.op} on validation split (centralized_results.csv)"
        else:
            part, algo = name.split("_", 1) if name.startswith("iid") else ("dirichlet_a0.5", name.removeprefix("dirichlet_a0.5_"))
            for tag in ("final", "fast"):
                path = self.results_dir / "federated" / tag / "federated_results.csv"
                if path.is_file():
                    df = pd.read_csv(path)
                    row = df[(df.partition == part) & (df.method == algo) & (df.seed == 0) & (df.split == "val")
                             & (df.operating_point == self.op)]
                    if len(row):
                        return float(row.threshold.iloc[0]), f"{self.op} on validation split (federated/{tag})"
        return 0.5, "default 0.5 (no saved operating point found)"

    def predict(self, req: PredictRequest) -> PredictResponse:
        if req.model not in self.available_models():
            raise FileNotFoundError(f"model {req.model!r} is not available; train it first")
        pre, fn = self._load(req.model)
        frame = pd.DataFrame([{f: req.features[f] for f in FEATURES}])
        with torch.no_grad():
            prob = float(fn(pre.transform(frame))[0])
        thr, src = self.threshold(req.model)
        cls = int(prob >= thr)
        from api.store import DISCLAIMERS
        return PredictResponse(model=req.model, probability=prob, threshold=thr, threshold_source=src, predicted_class=cls,
                               predicted_label="model output: higher-risk pattern" if cls else "model output: lower-risk pattern",
                               disclaimer=DISCLAIMERS["prediction"])
