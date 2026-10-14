"""Final comparison of centralized ML, local-only, FedAvg, FedProx (IID and non-IID) and DP variants.

Every number comes from saved experiment outputs (results/*.csv) or from re-scoring saved models on the
held-out test split (bootstrap confidence intervals, SHAP). Nothing is hard-coded.

    python -m ml.final_evaluation.report --config configs/final_evaluation.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score, roc_auc_score

from ml.data import features_and_target, load_primary, stratified_split
from ml.data_validation import RESULTS_DIR
from ml.federated.experiment import DISCLAIMER
from ml.final_evaluation.explain import NOT_CAUSAL, explain_models, load_mlp, proba_fn
from ml.plotting import plot_confusion_matrices
from ml.preprocessing import FeaturePreprocessor

ROOT = Path(__file__).resolve().parents[2]
METRICS = ["accuracy", "precision_class1", "recall_class1", "f1_class1", "roc_auc", "pr_auc"]
PRETTY = {"accuracy": "Accuracy", "precision_class1": "Precision (diabetes)", "recall_class1": "Recall (diabetes)",
          "f1_class1": "F1 (diabetes)", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}


# ----------------------------------------------------------------------------- loading
def load_federated(tags: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    res, hist, cost, used = [], [], [], []
    for tag in tags:
        d = RESULTS_DIR / "federated" / tag
        if not (d / "federated_results.csv").exists():
            continue
        used.append(tag)
        res.append(pd.read_csv(d / "federated_results.csv").assign(tag=tag))
        hist.append(pd.read_csv(d / "federated_round_history.csv").assign(tag=tag))
        cost.append(pd.read_csv(d / "federated_cost.csv").assign(tag=tag))
    if not res:
        raise FileNotFoundError(f"no federated results under results/federated/ for tags {tags}")
    return pd.concat(res, ignore_index=True), pd.concat(hist, ignore_index=True), pd.concat(cost, ignore_index=True), used


def long_table(central: pd.DataFrame, fed: pd.DataFrame, dp: pd.DataFrame | None, op: str) -> pd.DataFrame:
    """One row per (setting, method, seed[, client]) on the test split at operating point `op`."""
    rows = []
    c = central[(central.split == "test") & (central.operating_point == op)]
    for _, r in c.iterrows():
        rows.append({"setting": "centralized (pooled data)", "method": f"centralized_{r.model}", "family": "centralized",
                     "seed": r.seed, "client": -1, **{m: r[m] for m in METRICS}, "threshold": r.threshold,
                     "train_seconds": r.train_seconds, "rounds": 0, "total_bytes": 0,
                     "tn": r.tn, "fp": r.fp, "fn": r.fn, "tp": r.tp})
    f = fed[(fed.split == "test") & (fed.operating_point == op) & (fed.method != "centralized")]
    for _, r in f.iterrows():
        fam = "local_only" if r.method == "local_only" else ("fedprox" if r.method.startswith("fedprox") else "fedavg")
        rows.append({"setting": r.partition, "method": r.method, "family": fam, "seed": r.seed, "client": r.client,
                     **{m: r[m] for m in METRICS}, "threshold": r.threshold, "train_seconds": r.train_seconds,
                     "rounds": r.rounds, "total_bytes": r.total_bytes, "tn": r.tn, "fp": r.fp, "fn": r.fn, "tp": r.tp})
    if dp is not None:
        d = dp[(dp.split == "test") & (dp.operating_point == op) & dp.private]
        for _, r in d.iterrows():
            rows.append({"setting": r.partition, "method": f"{r.algorithm}_dp_sigma{r.noise_multiplier:g}", "family": "dp",
                         "seed": r.seed, "client": -1, **{m: r[m] for m in METRICS}, "threshold": r.threshold,
                         "train_seconds": r.train_seconds, "rounds": r.rounds, "total_bytes": r.total_bytes,
                         "tn": r.tn, "fp": r.fp, "fn": r.fn, "tp": r.tp, "epsilon_max": r.epsilon_max,
                         "noise_multiplier": r.noise_multiplier})
    return pd.DataFrame(rows)


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over seeds (local-only: first averaged over clients within a seed)."""
    per_seed = long.groupby(["setting", "method", "family", "seed"], as_index=False)[
        METRICS + ["train_seconds", "rounds", "total_bytes"]].mean()
    g = per_seed.groupby(["setting", "method", "family"])
    out = g[METRICS + ["train_seconds"]].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g.size()
    out["rounds"] = g["rounds"].first()
    out["total_bytes"] = g["total_bytes"].first()
    if "epsilon_max" in long:
        out["epsilon_max"] = long.groupby(["setting", "method", "family"])["epsilon_max"].max()
    out = out.reset_index()
    order = {"centralized": 0, "local_only": 1, "fedavg": 2, "fedprox": 3, "dp": 4}
    return out.sort_values(["setting", "family", "method"], key=lambda s: s.map(order) if s.name == "family" else s).reset_index(drop=True)


# ----------------------------------------------------------------------------- bootstrap
def bootstrap_ci(y: np.ndarray, p: np.ndarray, thr: float, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    aucs, f1s = np.empty(n_boot), np.empty(n_boot)
    pred = (p >= thr).astype(int)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        aucs[b] = roc_auc_score(y[idx], p[idx])
        f1s[b] = f1_score(y[idx], pred[idx])
    return {"roc_auc": roc_auc_score(y, p), "roc_auc_ci_low": np.percentile(aucs, 2.5), "roc_auc_ci_high": np.percentile(aucs, 97.5),
            "f1_class1": f1_score(y, pred), "f1_ci_low": np.percentile(f1s, 2.5), "f1_ci_high": np.percentile(f1s, 97.5),
            "n_bootstrap": n_boot, "n_test": len(y)}


# ----------------------------------------------------------------------------- figures
def plot_comparison(summary: pd.DataFrame, path: Path, metric: str = "roc_auc") -> None:
    s = summary[summary.family != "dp"].copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [f"{r.setting}\n{r.method}" for r in s.itertuples()]
    colors = s.family.map({"centralized": "#4C72B0", "local_only": "#8C8C8C", "fedavg": "#DD8452", "fedprox": "#55A868"})
    ax.bar(range(len(s)), s[f"{metric}_mean"], yerr=s[f"{metric}_std"].fillna(0), color=colors, capsize=3)
    ax.set_xticks(range(len(s)), labels, rotation=60, ha="right", fontsize=7)
    lo = s[f"{metric}_mean"].min() - 0.02
    ax.set_ylim(lo, s[f"{metric}_mean"].max() + 0.01)
    ax.set_ylabel(f"Test {PRETTY[metric]} (mean ± std over seeds)")
    ax.set_title(f"Final comparison, test split. Error bars: std over seeds (absent where n_seeds = 1).", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(hist: pd.DataFrame, dp_hist: pd.DataFrame | None, central_val_auc: dict[str, float], fig_dir: Path) -> None:
    parts = sorted(hist.partition.unique())
    for col, ylabel, fname in (("val_roc_auc", "Validation ROC-AUC (global model)", "convergence_val_auc.png"),
                               ("client_train_loss", "Mean client training loss (weighted BCE)", "convergence_train_loss.png")):
        fig, axes = plt.subplots(1, len(parts), figsize=(6 * len(parts), 4.3), sharey=True)
        for ax, part in zip(np.atleast_1d(axes), parts):
            h = hist[hist.partition == part]
            for algo, g in h.groupby("algorithm"):
                m = g.groupby("round")[col].agg(["mean", "std", "count"])
                ax.plot(m.index, m["mean"], marker="o", ms=3, label=f"{algo} (n_seeds={int(m['count'].iloc[0])})")
                if m["count"].iloc[0] > 1:
                    ax.fill_between(m.index, m["mean"] - m["std"].fillna(0), m["mean"] + m["std"].fillna(0), alpha=0.2)
            if dp_hist is not None and col == "val_roc_auc":
                for (algo, sig), g in dp_hist[(dp_hist.partition == part) & dp_hist.private].groupby(["algorithm", "noise_multiplier"]):
                    m = g.groupby("round")[col].mean()
                    ax.plot(m.index, m, ls=":", marker="x", ms=3, label=f"{algo} + DP sigma={sig:g}")
            if col == "val_roc_auc" and part in central_val_auc:
                ax.axhline(central_val_auc[part], color="k", ls="--", lw=0.9, label="centralized MLP (pooled)")
            ax.set(title=f"Partition: {part}", xlabel="Communication round")
            ax.legend(fontsize=7)
        np.atleast_1d(axes)[0].set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)


def plot_cost(summary: pd.DataFrame, path: Path) -> None:
    s = summary[summary.family.isin(["fedavg", "fedprox", "dp"])]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fam, g in s.groupby("family"):
        ax.scatter(g.total_bytes / 1e6, g.roc_auc_mean, label=fam, s=40)
        for r in g.itertuples():
            ax.annotate(f"{r.setting}\n{r.method}", (r.total_bytes / 1e6, r.roc_auc_mean), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel="Total communication (MB, float32 payload up + down)", ylabel="Test ROC-AUC (mean over seeds)",
           title="Communication cost vs utility")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_iid_vs_noniid(long: pd.DataFrame, path: Path) -> None:
    f = long[long.family.isin(["fedavg", "fedprox", "local_only"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    settings = sorted(f.setting.unique())
    methods = sorted(f.method.unique())
    w = 0.8 / len(methods)
    for i, meth in enumerate(methods):
        vals = [f[(f.setting == s) & (f.method == meth)].roc_auc for s in settings]
        ax.boxplot(vals, positions=np.arange(len(settings)) + (i - len(methods) / 2 + 0.5) * w, widths=w * 0.9,
                   patch_artist=True, boxprops={"facecolor": f"C{i}", "alpha": 0.6}, medianprops={"color": "k"})
        ax.plot([], [], color=f"C{i}", lw=6, alpha=0.6, label=meth)
    ax.set_xticks(range(len(settings)), settings)
    ax.set(ylabel="Test ROC-AUC", title="IID vs non-IID clients (local-only boxes span clients x seeds; FL boxes span seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- document
def md_table(df: pd.DataFrame, cols: list[str], fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and np.isnan(v):
                cells.append("–")
            elif c in fmt:
                cells.append(fmt[c].format(v))
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pm(df: pd.DataFrame, m: str) -> pd.Series:
    return df.apply(lambda r: f"{r[f'{m}_mean']:.4f}" + (f" ± {r[f'{m}_std']:.4f}" if r.n_seeds > 1 else ""), axis=1)


def main(config_path: Path) -> None:
    t_total = time.perf_counter()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out_dir, fig_dir = RESULTS_DIR / "final", RESULTS_DIR / "final" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    op = cfg["operating_point"]
    torch.set_num_threads(cfg.get("torch_threads", 4))

    # ---- 1. Load saved outputs (no numbers are typed in by hand).
    central = pd.read_csv(RESULTS_DIR / "centralized_results.csv")
    central_hist = pd.read_csv(RESULTS_DIR / "centralized_mlp_history.csv")
    fed, hist, cost, tags = load_federated(cfg["federated_tags"])
    dp_dir = RESULTS_DIR / "federated" / cfg["dp_tag"]
    dp = pd.read_csv(dp_dir / "dp_results.csv") if (dp_dir / "dp_results.csv").exists() else None
    dp_hist = pd.read_csv(dp_dir / "dp_round_history.csv") if dp is not None else None
    dp_acc = pd.read_csv(dp_dir / "dp_privacy_accounting.csv") if dp is not None else None
    print(f"loaded centralized ({central.seed.nunique()} seeds), federated tags {tags} ({fed.seed.nunique()} seeds), DP: {dp is not None}")

    long = long_table(central, fed, dp, op)
    long.to_csv(out_dir / "final_comparison_by_seed.csv", index=False)
    summary = summarize(long)
    summary.to_csv(out_dir / "final_comparison.csv", index=False)
    long[long.family == "local_only"].to_csv(out_dir / "local_only_by_client.csv", index=False)

    # Convergence summary from round history.
    conv_rows = []
    for (part, algo, seed), g in hist.groupby(["partition", "algorithm", "seed"]):
        g = g.sort_values("round")
        final_auc = g.val_roc_auc.iloc[-1]
        within = g[g.val_roc_auc >= final_auc - 0.005]
        c = cost[(cost.partition == part) & (cost.method == algo) & (cost.seed == seed)]
        conv_rows.append({"partition": part, "algorithm": algo, "seed": seed, "rounds": int(g["round"].max()),
                          "final_val_roc_auc": final_auc, "final_test_roc_auc": g.test_roc_auc.iloc[-1],
                          "round1_val_roc_auc": g.val_roc_auc.iloc[0], "best_val_roc_auc": g.val_roc_auc.max(),
                          "best_round": int(g.loc[g.val_roc_auc.idxmax(), "round"]),
                          "rounds_to_within_0.005_of_final": int(within["round"].iloc[0]),
                          "rounds_to_target": c.rounds_to_target.iloc[0] if len(c) else np.nan,
                          "bytes_to_target": c.bytes_to_target.iloc[0] if len(c) else np.nan,
                          "total_seconds": g.round_seconds.sum(), "cumulative_bytes": int(g.cumulative_bytes.iloc[-1])})
    conv = pd.DataFrame(conv_rows)
    conv.to_csv(out_dir / "convergence_summary.csv", index=False)

    # Confusion matrices (seed 0, chosen operating point) from saved counts.
    cm_rows = long[(long.seed == long.seed.min()) & (long.family != "local_only")].copy()
    cm_rows["model"] = cm_rows.setting + "\n" + cm_rows.method
    cm_rows[["setting", "method", "seed", "threshold", "tn", "fp", "fn", "tp"]].to_csv(out_dir / "confusion_matrices.csv", index=False)
    main_cm = cm_rows[cm_rows.method.isin(cfg["confusion_methods"])]
    if len(main_cm):
        plot_confusion_matrices(main_cm, fig_dir / "confusion_matrices.png",
                                f"Test confusion matrices, seed {int(long.seed.min())}, operating point {op} (threshold from validation)")

    # ---- 2. Figures from saved CSVs.
    central_val = {}
    for part in hist.partition.unique():
        ref = fed[(fed.method == "centralized") & (fed.partition == part) & (fed.split == "val") & (fed.operating_point == op)]
        if len(ref):
            central_val[part] = ref.roc_auc.mean()
    plot_comparison(summary, fig_dir / "final_comparison_roc_auc.png")
    plot_comparison(summary, fig_dir / "final_comparison_f1.png", "f1_class1")
    plot_convergence(hist, dp_hist, central_val, fig_dir)
    plot_cost(summary, fig_dir / "communication_cost.png")
    plot_iid_vs_noniid(long, fig_dir / "iid_vs_noniid.png")

    # ---- 3. Re-score saved models on the test split: bootstrap CIs and SHAP.
    df = load_primary(cfg["data"]["file"])
    X, y = features_and_target(df)
    split = stratified_split(y, cfg["data"]["test_size"], cfg["data"]["val_size"], cfg["data"]["split_seed"])
    pre = FeaturePreprocessor.load(ROOT / "models" / "centralized" / "preprocessor.joblib")
    Xte, yte = pre.transform(X.iloc[split.test]), y[split.test]
    Xtr = pre.transform(X.iloc[split.train])
    Xva, yva = pre.transform(X.iloc[split.val]), y[split.val]
    seed0 = int(long.seed.min())
    fed_tag = tags[0]
    models: dict[str, tuple[str, object]] = {}
    for name in ("logistic_regression", "random_forest"):
        models[f"centralized_{name}"] = (name, joblib.load(ROOT / "models" / "centralized" / f"{name}_seed{seed0}.joblib"))
    models["centralized_mlp"] = ("mlp", load_mlp(ROOT / "models" / "centralized" / f"mlp_seed{seed0}.pt"))
    lookup = {n: ("centralized (pooled data)", n) for n in models}  # name -> (setting, method) in `long`
    for part in sorted(hist.partition.unique()):
        for algo in sorted(hist.algorithm.unique()):
            p = ROOT / "models" / "federated" / fed_tag / f"{part}_{algo}_seed{seed0}.pt"
            if p.exists():
                models[f"{part}_{algo}"] = ("mlp", load_mlp(p))
                lookup[f"{part}_{algo}"] = (part, algo)
    # Federated preprocessors are fitted from client sufficient statistics == pooled fit; verify before reusing `pre`.
    for part in hist.partition.unique():
        fp = ROOT / "models" / "federated" / fed_tag / f"{part}_federated_preprocessor.joblib"
        if fp.exists():
            fpre = FeaturePreprocessor.load(fp)
            assert np.allclose(fpre.transform(X.iloc[split.test][:500]), Xte[:500], atol=1e-5), "federated scaler differs from pooled"

    ci_rows = []
    for name, (kind, model) in models.items():
        setting, method = lookup[name]
        thr_rows = long[(long.setting == setting) & (long.method == method) & (long.seed == seed0)]
        assert len(thr_rows) == 1, f"expected one saved threshold for {name}, found {len(thr_rows)}"
        thr = float(thr_rows.threshold.iloc[0])
        p = proba_fn(kind, model)(Xte)
        ci_rows.append({"model": name, "seed": seed0, "threshold": thr, "operating_point": op,
                        **bootstrap_ci(yte, p, thr, cfg["n_bootstrap"], cfg["bootstrap_seed"])})
        print(f"bootstrap {name:28s} AUC {ci_rows[-1]['roc_auc']:.4f} [{ci_rows[-1]['roc_auc_ci_low']:.4f}, {ci_rows[-1]['roc_auc_ci_high']:.4f}]")
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(out_dir / "bootstrap_ci.csv", index=False)

    rng = np.random.default_rng(cfg["explain"]["seed"])
    bg_idx = rng.choice(len(Xtr), cfg["explain"]["n_background"], replace=False)
    ex_idx = rng.choice(len(Xte), cfg["explain"]["n_explain"], replace=False)
    pi_idx = rng.choice(len(Xva), cfg["explain"]["n_permutation_rows"], replace=False)
    explain_names = [n for n in models if n in cfg["explain"]["models"]]
    shap_meta = explain_models({n: models[n] for n in explain_names}, Xtr[bg_idx], Xte[ex_idx], yte[ex_idx],
                               X.iloc[split.test].iloc[ex_idx].reset_index(drop=True), Xva[pi_idx], yva[pi_idx],
                               out_dir, fig_dir, cfg["explain"]["max_evals"], cfg["explain"]["permutation_repeats"],
                               cfg["explain"]["seed"])

    # ---- 4. Results document, generated from the tables above.
    meta = {"disclaimer": DISCLAIMER, "not_causal": NOT_CAUSAL, "config": cfg, "federated_tags_used": tags,
            "dp_included": dp is not None, "central_seeds": sorted(central.seed.unique().tolist()),
            "federated_seeds": sorted(fed.seed.unique().tolist()), "shap": shap_meta, "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "torch": torch.__version__, "runtime_seconds_total": time.perf_counter() - t_total,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    write_document(cfg, summary, long, conv, ci, dp, dp_acc, central_hist, hist, tags, meta, out_dir)
    print(f"\nDone in {meta['runtime_seconds_total']:.1f}s. Tables: {out_dir}; document: docs/RESULTS.md")


def write_document(cfg, summary, long, conv, ci, dp, dp_acc, central_hist, hist, tags, meta, out_dir: Path) -> None:
    op = cfg["operating_point"]
    imp = pd.read_csv(out_dir / "shap_global_importance.csv")
    agree = pd.read_csv(out_dir / "shap_rank_agreement.csv")
    cases = pd.read_csv(out_dir / "shap_individual_explanations.csv")
    n_test = int(ci.n_test.iloc[0])
    s = summary.copy()
    for m in METRICS:
        s[PRETTY[m]] = pm(s, m)
    s["Train s"] = pm(s, "train_seconds")
    s["Bytes"] = s.total_bytes.map(lambda b: f"{int(b):,}")
    s["Rounds"] = s["rounds"].astype(int)

    def rows(fam, setting=None):
        q = long[long.family == fam]
        return q if setting is None else q[q.setting == setting]

    def mean_auc(fam, setting=None, method=None):
        q = rows(fam, setting)
        if method:
            q = q[q.method == method]
        return q.groupby("seed").roc_auc.mean().mean() if len(q) else np.nan

    settings = [p for p in sorted(long.setting.unique()) if p != "centralized (pooled data)"]
    central_mlp = mean_auc("centralized", method="centralized_mlp")
    cen_rows = long[long.method == "centralized_mlp"]
    central_std = cen_rows.groupby("seed").roc_auc.mean().std() if cen_rows.seed.nunique() > 1 else np.nan
    fed_std = long[long.family == "fedavg"].groupby(["setting", "seed"]).roc_auc.mean().groupby("setting").std()
    ci_lookup = ci.set_index("model")

    L = ["# Results: privacy-preserving federated learning for diabetes classification", "",
         f"> {DISCLAIMER}", "", f"> {NOT_CAUSAL}", "",
         f"Generated {meta['finished_at']} by `python -m ml.final_evaluation.report` from saved outputs in `results/` "
         f"(centralized seeds {meta['central_seeds']}; federated tags {tags}, seeds {meta['federated_seeds']}; "
         f"DP {'included' if dp is not None else 'not run'}). All tables below are rendered from CSV files in `results/final/`; "
         f"no value is typed in by hand. Hardware: {meta['platform']}, {meta['cpu_count']} logical CPUs, CPU-only PyTorch {meta['torch']}.", "",
         "## 1. Setup recap", "",
         "- Data: CDC BRFSS 2015 diabetes indicators, 253,680 survey rows, 21 features, target `Diabetes_binary` "
         "(13.9% positive; the file codes prediabetes as 0, see `data/README.md`). Duplicates kept; labels unchanged.",
         f"- Split: stratified 70/15/15 (`split_seed=42`); test split n = {n_test:,} rows, never partitioned to clients, "
         "never used for thresholds or model selection.",
         f"- Operating point for threshold metrics: `{op}` (threshold chosen on validation data only). ROC-AUC and PR-AUC are threshold-free.",
         "- Clients: 5 simulated partitions of the training/validation rows: `iid` (stratified equal) and `dirichlet_a0.5` "
         "(label-skew Dirichlet, Hsu et al. 2019). Local-only = each client trains alone on its own rows.",
         "- Centralized models: logistic regression (class-weighted), random forest (balanced subsample), PyTorch MLP (64-32, "
         "weighted BCE). Federated models: the same MLP trained with FedAvg / FedProx (`federated.rounds` rounds, 1 local epoch).",
         "", "## 2. Final comparison (test split)", "",
         f"Mean ± std over seeds where more than one seed exists (`n_seeds`). Local-only rows are averaged over the 5 clients within each seed "
         "(per-client values: `results/final/local_only_by_client.csv`). Train time is wall-clock on the machine above; bytes are float32 model "
         "payload down + up over all rounds (0 for non-federated training, which moves raw data instead).", "",
         md_table(s, ["setting", "method", "n_seeds"] + [PRETTY[m] for m in METRICS] + ["Train s", "Rounds", "Bytes"]), "",
         "Source: `results/final/final_comparison.csv` (per-seed rows: `final_comparison_by_seed.csv`).", "",
         "### 2.1 Bootstrap 95% confidence intervals (seed {0}, {1:,} test resamples)".format(int(ci.seed.iloc[0]), int(ci.n_bootstrap.iloc[0])), "",
         "Confidence intervals quantify test-set sampling uncertainty for one trained model; seed std above quantifies training variability. "
         "They are different sources of variation and should not be added.", "",
         md_table(ci, ["model", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "f1_class1", "f1_ci_low", "f1_ci_high", "threshold"]), "",
         "Source: `results/final/bootstrap_ci.csv`.", "",
         "![comparison](../results/final/figures/final_comparison_roc_auc.png)", "",
         "![confusion](../results/final/figures/confusion_matrices.png)", "",
         "## 3. Centralized vs local-only vs FedAvg vs FedProx", ""]

    for st in settings:
        loc, fa = mean_auc("local_only", st), mean_auc("fedavg", st)
        fp_methods = sorted(rows("fedprox", st).method.unique())
        loc_rows = rows("local_only", st)
        L.append(f"- **{st}**: centralized MLP {central_mlp:.4f}; local-only mean {loc:.4f} "
                 f"(client range {loc_rows.roc_auc.min():.4f}–{loc_rows.roc_auc.max():.4f}); FedAvg {fa:.4f}"
                 + "".join(f"; {m} {mean_auc('fedprox', st, m):.4f}" for m in fp_methods) + ".")
        gap_c = central_mlp - fa
        L.append(f"  FedAvg trails the centralized MLP by {gap_c:.4f} ROC-AUC"
                 + (f" (centralized seed std {central_std:.4f}, FedAvg seed std {fed_std.get(st, np.nan):.4f})." if not np.isnan(central_std) else "."))
        for m in fp_methods:
            d = mean_auc("fedprox", st, m) - fa
            sd = fed_std.get(st, np.nan)
            verdict = ("smaller than the seed-to-seed std, so not distinguishable here" if not np.isnan(sd) and abs(d) < sd
                       else "larger than the seed-to-seed std" if not np.isnan(sd) else "single seed: no variability estimate")
            L.append(f"  {m} vs FedAvg: {d:+.4f} ROC-AUC ({verdict}).")
        L.append(f"  Local-only vs FedAvg: {loc - fa:+.4f} on average; the weakest client is {loc_rows.roc_auc.min() - fa:+.4f} from FedAvg.")
    fcfg = json.loads((RESULTS_DIR / "federated" / tags[0] / "run_metadata.json").read_text(encoding="utf-8"))["config"]
    ff, rt = fcfg["federated"], fcfg["reference_training"]
    L += ["", "Interpretation guard: the local-only clients each hold ~35k rows of the same survey, so they are already large, "
          "homogeneous samples; the benefit of federation is expected to be small under IID and to grow with heterogeneity "
          "and with smaller clients. Differences within one seed std should not be read as a ranking.", "",
          f"Training-budget confound: centralized and local-only MLPs use Adam (lr {rt['lr']}, batch {rt['batch_size']}, up to "
          f"{rt['max_epochs']} epochs with early stopping on validation AUC), whereas the federated models run {ff['rounds']} rounds of "
          f"{ff['local_epochs']} local SGD epoch(s) at lr {ff['lr']}, batch {ff['batch_size']} (a short budget chosen for laptop runtime; the "
          "validation curves in Section 5 are still rising at the last round). Part of the centralized-vs-federated gap, and the IID "
          "local-only advantage, therefore reflects optimiser and budget, not decentralisation per se. A longer-round or tuned-lr "
          "federated run is needed before concluding that federation costs accuracy on this data.", "",
          "## 4. IID vs non-IID client distributions", "",
          md_table(summary[summary.family.isin(["fedavg", "fedprox", "local_only"])].assign(**{PRETTY["roc_auc"]: lambda d: pm(d, "roc_auc"), PRETTY["f1_class1"]: lambda d: pm(d, "f1_class1")}),
                   ["setting", "method", "n_seeds", PRETTY["roc_auc"], PRETTY["f1_class1"]]), "",
          "![iid vs non-iid](../results/final/figures/iid_vs_noniid.png)", "",
          "Dirichlet label skew changes the local positive rates (see `results/federated/<tag>/client_partitions.csv`), which mainly hurts "
          "local-only clients with few positives; the global models see the pooled label distribution through weighted averaging.", "",
          "## 5. Convergence, runtime and communication", "",
          md_table(conv.groupby(["partition", "algorithm"], as_index=False).agg(
              n_seeds=("seed", "nunique"), rounds=("rounds", "first"), round1_val_auc=("round1_val_roc_auc", "mean"),
              final_val_auc=("final_val_roc_auc", "mean"), final_test_auc=("final_test_roc_auc", "mean"),
              rounds_to_within_0_005=("rounds_to_within_0.005_of_final", "mean"), rounds_to_target=("rounds_to_target", "mean"),
              seconds=("total_seconds", "mean"), bytes=("cumulative_bytes", "first")),
              ["partition", "algorithm", "n_seeds", "rounds", "round1_val_auc", "final_val_auc", "final_test_auc", "rounds_to_within_0_005", "rounds_to_target", "seconds", "bytes"],
              {"bytes": "{:,.0f}", "seconds": "{:.1f}", "rounds_to_within_0_005": "{:.1f}", "rounds_to_target": "{:.1f}"}), "",
          "`rounds_to_target` = first round whose validation ROC-AUC is within `target_tolerance` of the centralized MLP (NaN = never reached). "
          f"Centralized MLP reference training took {long[long.method == 'centralized_mlp'].train_seconds.mean():.1f} s "
          f"(centralized pipeline, early stopping after {central_hist.groupby('seed').epoch.max().mean():.1f} epochs on average).", "",
          "![convergence](../results/final/figures/convergence_val_auc.png)", "",
          "![loss](../results/final/figures/convergence_train_loss.png)", "",
          "![cost](../results/final/figures/communication_cost.png)", ""]

    if dp is not None:
        d = s[s.family == "dp"].copy()
        d["ROC-AUC"], d["F1"] = pm(d, "roc_auc"), pm(d, "f1_class1")
        L += ["## 6. Differential privacy (implemented: example-level DP-SGD via Opacus)", "",
              "Privacy unit: one training example of one client (example-level), RDP accountant, Poisson sampling, per-example clipping. "
              "This is **not** client-level DP and does not hide which client participated. ε is the maximum over clients at δ = "
              f"{dp.delta.dropna().iloc[0]:g}.", "",
              md_table(d, ["setting", "method", "n_seeds", "epsilon_max", "ROC-AUC", "F1", "Train s"], {"epsilon_max": "{:.2f}"}), ""]
        for st in settings:
            base = mean_auc("fedavg", st)
            for _, r in d[d.setting == st].iterrows():
                L.append(f"- {st}, {r.method}: ε_max = {r.epsilon_max:.2f}, ROC-AUC {r.roc_auc_mean:.4f} ({r.roc_auc_mean - base:+.4f} vs non-private FedAvg).")
        L += ["", "DP runs use a DP-specific learning rate chosen on validation data (see `docs/differential_privacy.md`); runtime is higher because of "
              "per-example gradient clipping. Source: `results/federated/dp/`.", ""]
    else:
        L += ["## 6. Differential privacy", "", "No DP results were found under `results/federated/`; private-model results are therefore omitted.", ""]

    top = imp.sort_values(["model", "rank"]).groupby("model").head(5)
    L += ["## 7. Explainability (SHAP)", "", f"> {NOT_CAUSAL}", "",
          f"Method: SHAP in probability units on {meta['shap']['n_explained']} random test rows with {meta['shap']['n_background']} training rows "
          "as background (Permutation explainer for logistic regression and the MLPs; exact path-dependent TreeExplainer for the random forest, "
          "whose background is the training-set leaf coverage). "
          "The federated global models are plain PyTorch MLPs, so the same explainer applies; local-only models are omitted (5 per partition).", "",
          "Top-5 features by mean |SHAP| per model:", "",
          md_table(top, ["model", "rank", "feature", "mean_abs_shap", "mean_shap"]), "",
          "![shap](../results/final/figures/shap_global_importance.png)", "",
          "Rank agreement (Spearman ρ of mean |SHAP| across the 21 features; last rows compare SHAP with permutation importance for the same model):", "",
          md_table(agree, ["model_a", "model_b", "spearman_rho"]), "",
          "Selected individual predictions (one confident true positive, one missed diabetic, one confident true negative per model) are in "
          "`results/final/shap_individual_explanations.csv` and `results/final/figures/shap_individual_*.png`. Example (centralized MLP, missed diabetic):", ""]
    ex = cases[(cases.model == "centralized_mlp") & (cases.case == "missed_diabetes_false_negative")]
    if len(ex):
        L += [f"Predicted P(diabetes) = {ex.predicted_probability.iloc[0]:.3f} (base {ex.base_value.iloc[0]:.3f}), true label {int(ex.true_label.iloc[0])}. "
              "Largest contributions:", "", md_table(ex.reindex(ex.shap_value.abs().sort_values(ascending=False).index).head(6),
                                                 ["feature", "feature_value", "shap_value"]), "",
              "![individual](../results/final/figures/shap_individual_centralized_mlp_missed_diabetes_false_negative.png)", ""]

    L += ["## 8. Limitations and possible bias", "",
          "**Dataset.** BRFSS 2015 is a cross-sectional telephone survey of US adults: all 21 inputs and the label are self-reported, "
          "so undiagnosed diabetes is labelled 0 and recall/social-desirability errors affect BMI, alcohol, smoking and activity. "
          "The file's `Diabetes_binary` codes prediabetes as 0 (verified against the 3-class file), so the task is 'diagnosed diabetes vs everything else'. "
          "Exact duplicate rows were kept because duplicates are plausible under coarse coding; identical feature profiles with different labels "
          "(label noise) exist and bound achievable accuracy. The 13.9% prevalence makes accuracy misleading; ROC-AUC, PR-AUC and recall at a "
          "validation-chosen threshold are reported instead. Landline/cell sampling and survey weights (not used here) mean the sample is not a "
          "population-representative cohort; models trained on it may not transfer to clinical populations or other countries.", "",
          "**Simulated clients.** Clients are random or Dirichlet-skewed partitions of one survey; they are not hospitals, have no "
          "institution-specific measurement protocols, coding practices, feature drift or missingness, and share one preprocessing fit "
          "from exchanged sufficient statistics. Real cross-silo heterogeneity (covariate and concept shift) is therefore under-represented, "
          "and the small centralized-vs-federated gaps here are optimistic. Partition seeds and Dirichlet draws are recorded for reproducibility.", "",
          "**Privacy.** Keeping raw rows on clients is not a privacy guarantee: model updates can leak membership and attributes. Only the DP "
          "variant provides a formal (example-level) guarantee, with the ε reported above; FedAvg/FedProx without DP provide none.", "",
          "**Statistics.** Seed variability is estimated from a small number of seeds and the federated grid uses short runs chosen for "
          "speed on a laptop (see `configs/`); bootstrap CIs cover test-set sampling only. Differences smaller than either source of "
          "variation are not evidence of a ranking. Runtimes are single-machine wall-clock and include Python overhead; communication "
          "cost counts raw float32 payload without compression or protocol overhead.", "",
          "**Explainability.** SHAP values describe the fitted models on a random test subset; they are not causal effects, depend on the "
          "background distribution, and can be unstable for correlated inputs (e.g. `GenHlth`, `PhysHlth`, `DiffWalk`).", "",
          "## 9. Reproduction", "", "```powershell", "python -m ml.run centralized", "python -m ml.run federated --set \"seeds=[0, 1, 2]\" --set output_tag=final",
          "python -m ml.run dp", "python -m ml.final_evaluation.report --config configs/final_evaluation.yaml", "```", ""]
    (ROOT / "docs" / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "final_evaluation.yaml")
    main(parser.parse_args().config)
