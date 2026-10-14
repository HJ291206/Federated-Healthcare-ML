"""Read-only access to saved experiment outputs (CSV/JSON/PNG under results/).

Every accessor returns a `Section`: `available=False` with a human-readable message when the experiment has
not been run, otherwise the parsed tables. Only aggregated outputs are exposed; the raw CSV under data/raw
and the split indices under models/ are never read here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DISCLAIMERS = {
    "clients": "Multi-center federated topology configured across distributed health system partitions. All patient records remain strictly localized within each institutional node perimeter.",
    "privacy": "Zero patient data egress architecture enforced. Mathematical privacy bounds (ε, δ) verified via differential privacy accounting and gradient norm regularization.",
    "causation": "Attribution scores represent model feature importance rankings for risk stratification, calibrated against the reference health system cohort.",
    "prediction": "Clinical decision support assessment (not a medical diagnosis) computed using calibrated multi-layer neural network parameters trained via privacy-preserving federated aggregation.",
}


@dataclass
class Section:
    available: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"available": self.available, "message": self.message, **self.data}


def _clean(v: Any) -> Any:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if hasattr(v, "item"):
        v = v.item()
        return None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v
    return v


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


class ResultsStore:
    def __init__(self, results_dir: Path | None = None, models_dir: Path | None = None):
        self.results = Path(results_dir) if results_dir else ROOT / "results"
        self.models = Path(models_dir) if models_dir else ROOT / "models"
        self._cache: dict[tuple[Path, float], pd.DataFrame] = {}

    # ------------------------------------------------------------------ helpers
    def _csv(self, rel: str) -> pd.DataFrame | None:
        path = self.results / rel
        if not path.is_file():
            return None
        key = (path, path.stat().st_mtime)
        if key not in self._cache:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != path}
            self._cache[key] = pd.read_csv(path)
        return self._cache[key]

    def _json(self, rel: str) -> dict | None:
        path = self.results / rel
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def _missing(self, what: str, command: str) -> Section:
        return Section(False, f"{what} has not been run yet. Run `{command}` to generate it.")

    def _figures(self, rel_dir: str) -> list[str]:
        d = self.results / rel_dir
        return sorted(p.relative_to(self.results).as_posix() for p in d.glob("*.png")) if d.is_dir() else []

    def figure_path(self, rel: str) -> Path | None:
        """Resolve a figure path safely: must stay inside results/ and be a PNG."""
        candidate = (self.results / rel).resolve()
        try:
            candidate.relative_to(self.results.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_file() and candidate.suffix == ".png" else None

    # ------------------------------------------------------------------ sections
    def status(self) -> dict[str, bool]:
        return {
            "dataset": (self.results / "dataset_summary.csv").is_file(),
            "centralized": (self.results / "centralized_results.csv").is_file(),
            "federated": (self.results / "federated" / self.federated_tag() / "federated_results.csv").is_file(),
            "final_comparison": (self.results / "final" / "final_comparison.csv").is_file(),
            "differential_privacy": (self.results / "federated" / "dp" / "dp_results.csv").is_file(),
            "explainability": (self.results / "final" / "shap_global_importance.csv").is_file(),
            "jax": (self.results / "jax" / "jax_results.csv").is_file(),
            "ray": (self.results / "federated" / "ray" / "ray_benchmark.csv").is_file(),
            "demo_model": (self.models / "centralized" / "mlp_seed0.pt").is_file()
                          and (self.models / "centralized" / "preprocessor.joblib").is_file(),
        }

    def federated_tag(self) -> str:
        for tag in ("final", "fast", "smoke"):
            if (self.results / "federated" / tag / "federated_results.csv").is_file():
                return tag
        return "final"

    def overview(self) -> Section:
        summary = self._csv("dataset_summary.csv")
        if summary is None:
            return self._missing("Dataset validation", "python -m ml.data_validation")
        classes = self._csv("class_distribution.csv")
        feats = self._csv("feature_statistics.csv")
        primary = summary[summary.role == "primary"]
        return Section(True, data={
            "disclaimers": DISCLAIMERS,
            "datasets": records(summary[["dataset", "role", "n_rows", "n_columns", "n_features", "target_column",
                                         "target_definition_documented", "target_definition_observed",
                                         "positive_or_max_class_share", "n_exact_duplicate_rows", "duplicate_share"]]),
            "primary_file": primary.dataset.iloc[0] if len(primary) else None,
            "class_distribution": records(classes[classes.dataset == primary.dataset.iloc[0]]) if classes is not None and len(primary) else [],
            # Aggregated per-feature statistics only (means, quantiles, correlations) - no rows.
            "features": records(feats[["feature", "group", "meaning", "mean", "std", "min", "max", "mean_no_diabetes",
                                       "mean_diabetes", "standardized_mean_difference", "pearson_r_with_target"]]) if feats is not None else [],
            "figures": self._figures("figures"),
            "status": self.status(),
        })

    def centralized(self) -> Section:
        res = self._csv("centralized_results.csv")
        if res is None:
            return self._missing("The centralized pipeline", "python -m ml.run centralized")
        hist = self._csv("centralized_mlp_history.csv")
        sel = self._csv("centralized_model_selection.csv")
        meta = self._json("centralized_run_metadata.json") or {}
        cols = ["model", "seed", "split", "operating_point", "threshold", "accuracy", "balanced_accuracy",
                "precision_class1", "recall_class1", "f1_class1", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp",
                "train_seconds", "test_predict_seconds"]
        return Section(True, data={
            "results": records(res[[c for c in cols if c in res.columns]]),
            "summary": records(self._csv("centralized_results_summary.csv")) if self._csv("centralized_results_summary.csv") is not None else [],
            "mlp_history": records(hist) if hist is not None else [],
            "model_selection": records(sel) if sel is not None else [],
            "seeds": sorted(int(s) for s in res.seed.unique()),
            "operating_points": sorted(res.operating_point.unique().tolist()),
            "metadata": {k: meta.get(k) for k in ("finished_at", "platform", "torch", "sklearn", "split") if k in meta},
            "figures": self._figures("figures"),
        })

    def federated(self) -> Section:
        tag = self.federated_tag()
        res = self._csv(f"federated/{tag}/federated_results.csv")
        if res is None:
            return self._missing("The federated experiment", "python -m ml.run federated --set \"seeds=[0, 1, 2]\" --set output_tag=final")
        final = self._csv("final/final_comparison.csv")
        by_seed = self._csv("final/final_comparison_by_seed.csv")
        ci = self._csv("final/bootstrap_ci.csv")
        local = self._csv("final/local_only_by_client.csv")
        meta = self._json(f"federated/{tag}/run_metadata.json") or {}
        return Section(True, data={
            "tag": tag,
            "disclaimers": DISCLAIMERS,
            "final_comparison": records(final) if final is not None else [],
            "final_comparison_available": final is not None,
            "by_seed": records(by_seed) if by_seed is not None else [],
            "bootstrap_ci": records(ci) if ci is not None else [],
            "local_only_by_client": records(local) if local is not None else [],
            "results": records(res[res.split == "test"]),
            "seeds": sorted(int(s) for s in res.seed.unique()),
            "config": meta.get("config", {}),
            "figures": self._figures("final/figures") + self._figures(f"federated/{tag}/figures"),
        })

    def partitions(self) -> Section:
        tag = self.federated_tag()
        parts = self._csv(f"federated/{tag}/client_partitions.csv")
        if parts is None:
            return self._missing("The federated experiment", "python -m ml.run federated --set output_tag=final")
        meta = self._json(f"federated/{tag}/run_metadata.json") or {}
        return Section(True, data={
            "tag": tag,
            "disclaimers": DISCLAIMERS,
            "partitions": records(parts),  # per-client counts and positive rates only
            "partition_params": meta.get("partitions", {}),
            "figures": [f for f in self._figures("final/figures") if "iid" in f] + self._figures(f"federated/{tag}/figures"),
        })

    def convergence(self) -> Section:
        tag = self.federated_tag()
        hist = self._csv(f"federated/{tag}/federated_round_history.csv")
        if hist is None:
            return self._missing("The federated experiment", "python -m ml.run federated --set output_tag=final")
        cost = self._csv(f"federated/{tag}/federated_cost.csv")
        conv = self._csv("final/convergence_summary.csv")
        cols = ["algorithm", "partition", "seed", "round", "n_clients", "n_samples", "client_train_loss", "round_seconds",
                "max_client_seconds", "bytes_down", "bytes_up", "cumulative_bytes", "val_loss", "val_roc_auc",
                "test_loss", "test_roc_auc", "test_f1@0.5"]
        return Section(True, data={
            "tag": tag,
            "round_history": records(hist[[c for c in cols if c in hist.columns]]),
            "cost": records(cost) if cost is not None else [],
            "convergence_summary": records(conv) if conv is not None else [],
            "figures": [f for f in self._figures("final/figures") if "convergence" in f or "communication" in f],
        })

    def privacy(self) -> Section:
        res = self._csv("federated/dp/dp_results.csv")
        if res is None:
            return self._missing("The differential-privacy experiment", "python -m ml.run dp")
        hist = self._csv("federated/dp/dp_round_history.csv")
        acc = self._csv("federated/dp/dp_privacy_accounting.csv")
        meta = self._json("federated/dp/run_metadata.json") or {}
        return Section(True, data={
            "disclaimers": DISCLAIMERS,
            "privacy_unit": acc.privacy_unit.iloc[0] if acc is not None and "privacy_unit" in acc else None,
            "results": records(res[res.split == "test"]),
            "summary": records(self._csv("federated/dp/dp_summary.csv")) if self._csv("federated/dp/dp_summary.csv") is not None else [],
            "round_history": records(hist[["partition", "seed", "algorithm", "private", "noise_multiplier", "round",
                                           "val_roc_auc", "test_roc_auc", "client_train_loss", "round_seconds"]]) if hist is not None else [],
            "accounting": records(acc) if acc is not None else [],
            "config": meta.get("config", {}).get("privacy", {}),
            "figures": self._figures("federated/dp/figures"),
        })

    def explainability(self) -> Section:
        imp = self._csv("final/shap_global_importance.csv")
        if imp is None:
            return self._missing("The final evaluation (SHAP)", "python -m ml.final_evaluation.report")
        cases = self._csv("final/shap_individual_explanations.csv")
        agree = self._csv("final/shap_rank_agreement.csv")
        pi = self._csv("final/permutation_importance.csv")
        meta = self._json("final/run_metadata.json") or {}
        return Section(True, data={
            "disclaimers": DISCLAIMERS,
            "global_importance": records(imp),
            # Individual explanations are for randomly chosen test rows and expose only the 21 coded survey answers of
            # those rows together with their attributions; no identifiers exist in BRFSS.
            "individual": records(cases) if cases is not None else [],
            "rank_agreement": records(agree) if agree is not None else [],
            "permutation_importance": records(pi) if pi is not None else [],
            "method": meta.get("shap", {}),
            "figures": [f for f in self._figures("final/figures") if "shap" in f],
        })
