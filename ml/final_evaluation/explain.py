"""Explainability for the saved centralized and federated models.

Method: SHAP (Lundberg & Lee, 2017), the Shapley-value attribution of each prediction to its 21 inputs.
- Logistic regression and the MLPs (centralized and federated global models) use the model-agnostic
  Permutation explainer on the predicted probability, with an independent background sample from the
  training split (interventional expectation).
- Random forest uses the exact path-dependent TreeExplainer; its leaf values are class probabilities, so all
  attributions are in probability units and comparable across models (the background convention differs:
  tree-path-dependent uses the training-set leaf coverage instead of the explicit background sample).
Global importance is the mean |SHAP| over an explained test subset. A permutation-importance check
(ROC-AUC drop when one column is shuffled) is computed as a method-independent cross-check.

Feature importance describes what the fitted model uses, not what causes diabetes. BRFSS is a
cross-sectional self-report survey; attributions reflect associations and model artefacts.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from ml.data_validation import FEATURES
from ml.models import load_mlp  # noqa: F401  (re-exported for ml.final_evaluation.report)
from ml.training import predict_proba

NOT_CAUSAL = ("Feature importance is NOT causation: it measures how much the fitted model's output depends on "
              "a feature in the BRFSS 2015 survey sample, not whether the feature causes diabetes.")


def proba_fn(kind: str, model):
    if kind == "mlp":
        return lambda X: predict_proba(model, np.asarray(X, dtype=np.float32))
    return lambda X: model.predict_proba(np.asarray(X))[:, 1]


def shap_values(kind: str, model, X_background: np.ndarray, X_explain: np.ndarray, max_evals: int) -> tuple[np.ndarray, float]:
    """Return (shap_values (n, p) in probability units, base value)."""
    if kind == "random_forest":
        # Exact path-dependent TreeExplainer; sklearn RF leaf values are class probabilities, so the output is
        # already in probability units. (The interventional variant with a background sample took ~2 min here.)
        ex = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        vals = ex.shap_values(X_explain, check_additivity=False)
        vals = vals[..., 1] if vals.ndim == 3 else vals
        base = float(np.atleast_1d(ex.expected_value)[-1])
        return np.asarray(vals), base
    masker = shap.maskers.Independent(X_background, max_samples=len(X_background))
    ex = shap.explainers.Permutation(proba_fn(kind, model), masker, feature_names=list(FEATURES))
    res = ex(X_explain, max_evals=max_evals, silent=True)
    return np.asarray(res.values), float(np.atleast_1d(res.base_values)[0])


def permutation_importance(predict, X: np.ndarray, y: np.ndarray, repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, predict(X))
    rows = []
    for j, name in enumerate(FEATURES):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            drops.append(base - roc_auc_score(y, predict(Xp)))
        rows.append({"feature": name, "auc_drop_mean": float(np.mean(drops)), "auc_drop_std": float(np.std(drops)),
                     "baseline_roc_auc": base})
    return pd.DataFrame(rows)


def select_cases(p: np.ndarray, y: np.ndarray) -> dict[str, int]:
    """Indices (into the explained subset) of three illustrative predictions."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return {"confident_true_positive": int(pos[np.argmax(p[pos])]),
            "missed_diabetes_false_negative": int(pos[np.argmin(p[pos])]),
            "confident_true_negative": int(neg[np.argmin(p[neg])])}


def plot_global_importance(imp: pd.DataFrame, path: Path, top: int = 21) -> None:
    piv = imp.pivot(index="feature", columns="model", values="mean_abs_shap")
    order = piv.mean(axis=1).sort_values(ascending=True).index[-top:]
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(9, 7))
    k, h = len(piv.columns), 0.8 / len(piv.columns)
    ypos = np.arange(len(piv))
    for i, col in enumerate(piv.columns):
        ax.barh(ypos + (i - k / 2 + 0.5) * h, piv[col], height=h, label=col)
    ax.set_yticks(ypos, piv.index, fontsize=8)
    ax.set_xlabel("Mean |SHAP value| (change in predicted probability)")
    ax.set_title("Global feature importance (SHAP) on the test subset\n" + NOT_CAUSAL, fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_individual(sv: np.ndarray, base: float, raw_row: pd.Series, prob: float, label: int, title: str, path: Path) -> None:
    order = np.argsort(np.abs(sv))[::-1][:12]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in sv[order]]
    ax.barh(range(len(order))[::-1], sv[order], color=colors)
    ax.set_yticks(range(len(order))[::-1], [f"{FEATURES[j]} = {raw_row[FEATURES[j]]:g}" for j in order], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("SHAP contribution to predicted probability (red raises, blue lowers)")
    ax.set_title(f"{title}\nbase rate {base:.3f} -> predicted P(diabetes) = {prob:.3f}; true label = {label}\n"
                 "Attribution of one prediction, not a causal statement.", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def explain_models(models: dict[str, tuple[str, object]], X_bg: np.ndarray, X_ex: np.ndarray, y_ex: np.ndarray,
                   raw_ex: pd.DataFrame, X_pi: np.ndarray, y_pi: np.ndarray, out_dir: Path, fig_dir: Path,
                   max_evals: int, pi_repeats: int, seed: int) -> dict:
    """models: name -> (kind, fitted model). Writes CSVs and figures; returns a summary dict."""
    imp_rows, case_rows, pi_frames, timings, per_model_imp = [], [], [], {}, {}
    for name, (kind, model) in models.items():
        t0 = time.perf_counter()
        sv, base = shap_values(kind, model, X_bg, X_ex, max_evals)
        timings[name] = time.perf_counter() - t0
        mean_abs = np.abs(sv).mean(axis=0)
        per_model_imp[name] = mean_abs
        for j, f in enumerate(FEATURES):
            imp_rows.append({"model": name, "feature": f, "mean_abs_shap": float(mean_abs[j]),
                             "mean_shap": float(sv[:, j].mean()), "rank": int((-mean_abs).argsort().argsort()[j] + 1)})
        p = proba_fn(kind, model)(X_ex)
        for case, i in select_cases(p, y_ex).items():
            for j, f in enumerate(FEATURES):
                case_rows.append({"model": name, "case": case, "subset_index": i, "true_label": int(y_ex[i]),
                                  "predicted_probability": float(p[i]), "base_value": base, "feature": f,
                                  "feature_value": float(raw_ex.iloc[i][f]), "shap_value": float(sv[i, j])})
            plot_individual(sv[i], base, raw_ex.iloc[i], float(p[i]), int(y_ex[i]), f"{name}: {case.replace('_', ' ')}",
                            fig_dir / f"shap_individual_{name}_{case}.png")
        pi = permutation_importance(proba_fn(kind, model), X_pi, y_pi, pi_repeats, seed).assign(model=name)
        pi_frames.append(pi)
        print(f"explained {name:28s} SHAP {timings[name]:5.1f}s  top-3: "
              + ", ".join(FEATURES[j] for j in (-mean_abs).argsort()[:3]))

    imp = pd.DataFrame(imp_rows)
    imp.to_csv(out_dir / "shap_global_importance.csv", index=False)
    pd.DataFrame(case_rows).to_csv(out_dir / "shap_individual_explanations.csv", index=False)
    pi_all = pd.concat(pi_frames, ignore_index=True)
    pi_all.to_csv(out_dir / "permutation_importance.csv", index=False)
    plot_global_importance(imp, fig_dir / "shap_global_importance.png")

    # Rank agreement between models (Spearman on mean |SHAP|), and SHAP vs permutation importance per model.
    names = list(models)
    agree = [{"model_a": a, "model_b": b,
              "spearman_rho": float(spearmanr(per_model_imp[a], per_model_imp[b]).statistic)}
             for i, a in enumerate(names) for b in names[i + 1:]]
    for n in names:
        pi_n = pi_all[pi_all.model == n].set_index("feature").loc[list(FEATURES), "auc_drop_mean"].to_numpy()
        agree.append({"model_a": n, "model_b": f"{n} (permutation importance)",
                      "spearman_rho": float(spearmanr(per_model_imp[n], pi_n).statistic)})
    pd.DataFrame(agree).to_csv(out_dir / "shap_rank_agreement.csv", index=False)
    return {"shap_seconds": timings, "n_explained": int(len(X_ex)), "n_background": int(len(X_bg)),
            "max_evals": max_evals, "shap_version": shap.__version__}
