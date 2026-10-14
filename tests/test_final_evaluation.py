import numpy as np
import pandas as pd
import pytest

from ml.data_validation import FEATURES
from ml.final_evaluation.report import bootstrap_ci, long_table, md_table, summarize

shap = pytest.importorskip("shap")
from ml.final_evaluation.explain import permutation_importance, select_cases, shap_values  # noqa: E402
from ml.models import DiabetesMLP  # noqa: E402


def _metric_row(**kw):
    base = {"accuracy": 0.8, "balanced_accuracy": 0.7, "precision": 0.3, "recall": 0.6, "f1": 0.4,
            "precision_class0": 0.9, "recall_class0": 0.8, "f1_class0": 0.85, "precision_class1": 0.3,
            "recall_class1": 0.6, "f1_class1": 0.4, "roc_auc": 0.82, "pr_auc": 0.4, "tn": 10, "fp": 2, "fn": 3, "tp": 5,
            "threshold": 0.5, "split": "test", "operating_point": "val_max_f1"}
    return {**base, **kw}


def test_long_table_and_summary_aggregate_clients_then_seeds():
    central = pd.DataFrame([_metric_row(model="mlp", seed=s, train_seconds=1.0, roc_auc=0.82 + 0.01 * s) for s in (0, 1)])
    fed = pd.DataFrame(
        [_metric_row(partition="iid", seed=s, method="local_only", client=c, train_seconds=1, rounds=0, total_bytes=0,
                     roc_auc=0.7 + 0.1 * c) for s in (0, 1) for c in (0, 1)]
        + [_metric_row(partition="iid", seed=s, method="fedavg", client=-1, train_seconds=2, rounds=4, total_bytes=100,
                       roc_auc=0.8) for s in (0, 1)]
        + [_metric_row(partition="iid", seed=0, method="centralized", client=-1, train_seconds=1, rounds=0, total_bytes=0)])
    long = long_table(central, fed, None, "val_max_f1")
    assert set(long.family) == {"centralized", "local_only", "fedavg"}
    assert "centralized" not in set(long.method)  # the federated file's reference row is excluded
    summary = summarize(long)
    loc = summary[summary.method == "local_only"].iloc[0]
    assert loc.roc_auc_mean == pytest.approx(0.75) and loc.n_seeds == 2 and loc.roc_auc_std == pytest.approx(0.0)
    cen = summary[summary.method == "centralized_mlp"].iloc[0]
    assert cen.roc_auc_mean == pytest.approx(0.825) and cen.roc_auc_std == pytest.approx(np.std([0.82, 0.83], ddof=1))
    assert summary[summary.method == "fedavg"].total_bytes.iloc[0] == 100


def test_bootstrap_ci_brackets_point_estimate_and_is_deterministic():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.5 * y + 0.3 * rng.normal(size=2000) + 0.25, 0, 1)
    a, b = bootstrap_ci(y, p, 0.5, 50, seed=1), bootstrap_ci(y, p, 0.5, 50, seed=1)
    assert a == b
    assert a["roc_auc_ci_low"] <= a["roc_auc"] <= a["roc_auc_ci_high"]
    assert a["f1_ci_low"] <= a["f1_class1"] <= a["f1_ci_high"]


def test_md_table_formats_nan_and_floats():
    out = md_table(pd.DataFrame({"a": [1.23456, np.nan], "b": ["x", "y"]}), ["a", "b"])
    assert "| 1.2346 | x |" in out and "| – | y |" in out


def test_shap_values_are_additive_for_mlp_and_logistic_regression():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 21)).astype(np.float32)
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    lr = LogisticRegression().fit(X, y)
    sv, base = shap_values("logistic_regression", lr, X[:50], X[50:70], max_evals=215)
    assert sv.shape == (20, 21)
    assert np.allclose(base + sv.sum(axis=1), lr.predict_proba(X[50:70])[:, 1], atol=1e-3)
    mlp = DiabetesMLP(21, (8,), 0.0).eval()
    sv_m, base_m = shap_values("mlp", mlp, X[:50], X[50:60], max_evals=215)
    assert sv_m.shape == (10, 21)
    import torch
    with torch.no_grad():
        pm = torch.sigmoid(mlp(torch.tensor(X[50:60]))).numpy()
    assert np.allclose(base_m + sv_m.sum(axis=1), pm, atol=1e-3)


def test_permutation_importance_ranks_true_signal_first():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 21))
    y = (X[:, 3] + 0.2 * rng.normal(size=2000) > 0).astype(int)
    pi = permutation_importance(lambda Z: 1 / (1 + np.exp(-3 * Z[:, 3])), X, y, repeats=3, seed=0)
    assert pi.loc[pi.auc_drop_mean.idxmax(), "feature"] == FEATURES[3]
    assert (pi.auc_drop_mean.drop(pi.auc_drop_mean.idxmax()).abs() < 0.02).all()


def test_select_cases_picks_expected_rows():
    p = np.array([0.9, 0.1, 0.2, 0.05, 0.7])
    y = np.array([1, 1, 0, 0, 1])
    cases = select_cases(p, y)
    assert cases == {"confident_true_positive": 0, "missed_diabetes_false_negative": 1, "confident_true_negative": 3}
