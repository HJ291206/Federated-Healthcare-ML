"""Centralized vs local-only vs FedAvg vs FedProx on simulated clients.

Run from the repository root:
    python -m ml.federated.experiment --config configs/federated_smoke.yaml   # quick check
    python -m ml.federated.experiment --config configs/federated.yaml         # main experiment

Simulated clients are partitions of one survey dataset (BRFSS 2015), not hospitals.

Protocol (identical for every method):
  * Global split = the centralized split (same file, fractions and split_seed), so the
    global test set is the one used by ml.train_centralized.
  * Client k receives a partition of the global *train* rows and of the global *validation*
    rows. The test rows are never partitioned or passed to any client.
  * Federated preprocessing: the scaler is built from per-client sufficient statistics
    (counts, sums, sums of squares); the class weight from per-client label counts.
  * Thresholds for the global models are chosen on the union of client validation rows
    (in a deployment each client would report its own validation counts), then frozen.
  * Federated models are taken after the final round: no round is selected on the test set.
  * The global test set is used only by the simulation's evaluator, to report metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from ml import plotting
from ml.data import features_and_target, load_primary, stratified_split
from ml.data_validation import FEATURES, RAW_DIR, RESULTS_DIR, TARGET, sha256
from ml.evaluation import classification_metrics, select_thresholds
from ml.federated.client import FederatedClient, LocalTrainConfig
from ml.federated.fedavg import build_fedavg_server
from ml.federated.fedprox import build_fedprox_server
from ml.federated.partition import PartitionResult, make_partition, partition_summary
from ml.federated.server import FederatedServer, state_nbytes
from ml.models import build_mlp, count_parameters
from ml.preprocessing import FeaturePreprocessor
from ml.seed import set_global_seed
from ml.training import MLPTrainConfig, fit_mlp, make_loss, predict_logits, predict_proba

ROOT = Path(__file__).resolve().parents[2]
DISCLAIMER = ("Clients are simulated partitions of a single public survey dataset (CDC BRFSS 2015). "
              "They are not hospitals or real institutions, and federated learning alone does not "
              "guarantee privacy: model updates can still leak information about training data.")


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class GlobalEvaluator:
    """Simulation-side evaluator. Holds the global test set; never shared with clients."""

    def __init__(self, X_val, y_val, X_test, y_test, pos_weight: float):
        self.X_val, self.y_val = torch.as_tensor(X_val), y_val
        self.X_test, self.y_test = torch.as_tensor(X_test), y_test
        self.loss_fn = make_loss(pos_weight)
        self._yv = torch.as_tensor(y_val, dtype=torch.float32)
        self._yt = torch.as_tensor(y_test, dtype=torch.float32)

    def __call__(self, model: torch.nn.Module) -> dict:
        lv, lt = predict_logits(model, self.X_val), predict_logits(model, self.X_test)
        pv, pt = torch.sigmoid(lv).numpy(), torch.sigmoid(lt).numpy()
        m = classification_metrics(self.y_test, pt, 0.5)
        return {
            "val_loss": float(self.loss_fn(lv, self._yv)), "val_roc_auc": roc_auc_score(self.y_val, pv),
            "val_pr_auc": average_precision_score(self.y_val, pv),
            "test_loss": float(self.loss_fn(lt, self._yt)), "test_roc_auc": m["roc_auc"],
            "test_pr_auc": m["pr_auc"], "test_accuracy@0.5": m["accuracy"],
            "test_precision@0.5": m["precision"], "test_recall@0.5": m["recall"], "test_f1@0.5": m["f1"],
        }


def score_rows(p_val, y_val, p_test, y_test, recall_target: float, base: dict) -> list[dict]:
    rows = []
    for op, thr in select_thresholds(y_val, p_val, recall_target).items():  # validation only
        for split, yt, pp in (("val", y_val, p_val), ("test", y_test, p_test)):
            rows.append({**base, "split": split, "operating_point": op, **classification_metrics(yt, pp, thr)})
    return rows


def build_clients(part: PartitionResult, X: pd.DataFrame, y: np.ndarray) -> tuple[list[FederatedClient], FeaturePreprocessor, float]:
    # Each client computes statistics on its own training rows; only these aggregates leave the client.
    stats = [FeaturePreprocessor.sufficient_statistics(X.iloc[c.train_idx]) for c in part.clients]
    pre = FeaturePreprocessor.from_sufficient_statistics(stats)
    counts = np.array([[int((y[c.train_idx] == 0).sum()), int((y[c.train_idx] == 1).sum())] for c in part.clients])
    pos_weight = float(counts[:, 0].sum() / counts[:, 1].sum())
    clients = [FederatedClient(c.client_id, pre.transform(X.iloc[c.train_idx]), y[c.train_idx],
                               pre.transform(X.iloc[c.val_idx]), y[c.val_idx], pos_weight)
               for c in part.clients]
    return clients, pre, pos_weight


def run_federated(server: FederatedServer, rounds: int, evaluator: GlobalEvaluator,
                  checkpoint_dir: Path | None = None, checkpoint_every: int = 0) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    history = server.fit(rounds, evaluator, checkpoint_dir, checkpoint_every)
    return history, time.perf_counter() - t0


def rounds_to_target(history: pd.DataFrame, target: float) -> tuple[float, float]:
    hit = history[history.val_roc_auc >= target]
    if hit.empty:
        return float("nan"), float("nan")
    first = hit.iloc[0]
    return float(first["round"]), float(first["cumulative_bytes"])


# ----------------------------------------------------------------------------- figures

def plot_client_distributions(summary: pd.DataFrame, path: Path) -> None:
    parts = summary.partition.unique()
    fig, axes = plt.subplots(1, len(parts), figsize=(5 * len(parts), 4.2), squeeze=False)
    for ax, name in zip(axes[0], parts):
        s = summary[(summary.partition == name) & (summary.split == "train")]
        ax.bar(s.client, s.n_negative, color=plotting.MODEL_COLORS["logistic_regression"], label="no diabetes (0)")
        ax.bar(s.client, s.n_positive, bottom=s.n_negative, color=plotting.MODEL_COLORS["mlp"], label="diabetes (1)")
        for _, r in s.iterrows():
            ax.text(r.client, r.n, f"{r.positive_rate:.1%}", ha="center", va="bottom", fontsize=8)
        ax.set(title=f"{name}: client training rows\n(label = positive rate)", xlabel="Simulated client",
               ylabel="Rows", xticks=list(s.client))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(history: pd.DataFrame, refs: pd.DataFrame, partition: str, path: Path) -> None:
    h = history[history.partition == partition].groupby(["algorithm", "round"], sort=False).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for algo, g in h.groupby("algorithm", sort=False):
        axes[0].plot(g["round"], g.val_roc_auc, label=algo)
        axes[1].plot(g["round"], g.test_roc_auc, label=algo)
        axes[2].plot(g["round"], g.val_loss, label=algo)
    r = refs[refs.partition == partition]
    for ax, col in ((axes[0], "val_roc_auc"), (axes[1], "test_roc_auc")):
        ax.axhline(r[r.method == "centralized"][col].mean(), color="k", ls="--", lw=1, label="centralized MLP")
        ax.axhline(r[r.method == "local_only"][col].mean(), color="grey", ls=":", lw=1.2, label="local-only (mean of clients)")
    axes[0].set(title="Global model, validation ROC-AUC", xlabel="Communication round", ylabel="ROC-AUC")
    axes[1].set(title="Global model, test ROC-AUC (reporting only)", xlabel="Communication round", ylabel="ROC-AUC")
    axes[2].set(title="Global model, validation loss", xlabel="Communication round", ylabel="Class-weighted BCE")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle(f"Convergence, partition '{partition}' (mean over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(test_rows: pd.DataFrame, path: Path) -> None:
    d = test_rows[test_rows.operating_point == "val_max_f1"]
    parts = list(d.partition.unique())
    methods = list(d.method.unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    width = 0.8 / len(methods)
    for ax, metric, label in ((axes[0], "roc_auc", "Test ROC-AUC"), (axes[1], "recall_class1", "Test recall, diabetes (val max-F1 threshold)")):
        for i, m in enumerate(methods):
            g = d[d.method == m].groupby("partition")[metric]
            mean, lo, hi = g.mean().reindex(parts), g.min().reindex(parts), g.max().reindex(parts)
            x = np.arange(len(parts)) + (i - (len(methods) - 1) / 2) * width
            ax.bar(x, mean, width, label=m, yerr=[mean - lo, hi - mean], capsize=2)
        ax.set_xticks(np.arange(len(parts)), parts)
        ax.set_ylabel(label)
        lo_all = d[metric].min()
        ax.set_ylim(max(0, lo_all - 0.05), min(1, d[metric].max() + 0.03))
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Test split comparison. Bars: mean; whiskers: min-max over seeds (and over clients for local-only)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- report

def write_report(meta: dict, part_summary: pd.DataFrame, summary: pd.DataFrame, comm: pd.DataFrame, path: Path) -> None:
    def f(v, nd=4):
        return "n/a" if pd.isna(v) else f"{v:.{nd}f}"

    L = ["# Federated learning results", "", f"> {DISCLAIMER}", "",
         f"Generated by `python -m ml.federated.experiment --config {meta['config_file']}` on {meta['finished_at']}. "
         "All numbers are read from the CSV files in this folder.", "",
         f"- Data: `{meta['data_file']}` (sha256 `{meta['data_sha256'][:16]}…`), target `{TARGET}`.",
         f"- Global split (seed {meta['split_seed']}): " + ", ".join(f"{k} {v:,}" for k, v in meta["split_sizes"].items())
         + ". Test rows are never assigned to clients.",
         f"- Seeds: {meta['seeds']}. Clients: {meta['num_clients']}. Rounds: {meta['federated']['rounds']}, "
         f"local epochs: {meta['federated']['local_epochs']}, SGD lr {meta['federated']['lr']}, batch "
         f"{meta['federated']['batch_size']}, client fraction {meta['federated']['client_fraction']}.",
         f"- Model: MLP {meta['model']['hidden_sizes']} ({meta['model_parameters']:,} parameters, "
         f"{meta['model_bytes']:,} bytes per float32 copy).", "",
         "## Client partitions (training rows)", "",
         "| Partition | Client | Train n | Train positives | Train positive rate | Val n | Val positive rate |",
         "|---|---|---|---|---|---|---|"]
    for (p, c), g in part_summary.groupby(["partition", "client"], sort=False):
        tr, va = g[g.split == "train"].iloc[0], g[g.split == "val"].iloc[0]
        L.append(f"| {p} | {c} | {tr.n:,} | {tr.n_positive:,} | {tr.positive_rate:.2%} | {va.n:,} | {va.positive_rate:.2%} |")
    L += ["", "Dirichlet partitions: per-class client shares drawn from Dir(alpha); draws with fewer than the minimum "
          "per-class counts are rejected (see `ml/federated/partition.py`).", ""]

    for op in ("val_max_f1", "default_0.5"):
        s = summary[summary.operating_point == op]
        L += [f"## Test metrics, operating point `{op}`", "",
              "Mean over seeds; for local-only, mean over clients and seeds with [min, max] across clients.", "",
              "| Partition | Method | ROC-AUC | Accuracy | Precision (1) | Recall (1) | F1 (1) | Precision (0) | Recall (0) | TN | FP | FN | TP |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in s.iterrows():
            auc = f(r.roc_auc_mean) + (f" [{f(r.roc_auc_min)}, {f(r.roc_auc_max)}]" if r.method == "local_only" else "")
            rec = f(r.recall_class1_mean) + (f" [{f(r.recall_class1_min)}, {f(r.recall_class1_max)}]" if r.method == "local_only" else "")
            L.append(f"| {r.partition} | {r.method} | {auc} | {f(r.accuracy_mean)} | {f(r.precision_class1_mean)} | {rec} | "
                     f"{f(r.f1_class1_mean)} | {f(r.precision_class0_mean)} | {f(r.recall_class0_mean)} | "
                     f"{r.tn_mean:,.0f} | {r.fp_mean:,.0f} | {r.fn_mean:,.0f} | {r.tp_mean:,.0f} |")
        L.append("")

    L += ["## Training time and communication", "",
          f"Bytes are raw float32 tensor payloads (download of the global model + upload of each client model), "
          "excluding serialization and protocol overhead. `rounds_to_target` is the first round whose global "
          f"validation ROC-AUC is within {meta['target_tolerance']} of the same-seed centralized MLP's validation ROC-AUC.", "",
          "| Partition | Method | Training seconds | Rounds | Total MB transmitted | Rounds to target | MB to target |",
          "|---|---|---|---|---|---|---|"]
    for _, r in comm.iterrows():
        L.append(f"| {r.partition} | {r.method} | {r.train_seconds:.1f} | {r.rounds:.0f} | {r.total_bytes / 1e6:.3f} | "
                 f"{f(r.rounds_to_target, 1)} | {f(r.bytes_to_target / 1e6, 3) if not pd.isna(r.bytes_to_target) else 'n/a'} |")
    L += ["", "Centralized and local-only methods transmit no model parameters (0 bytes); centralized training "
          "instead requires moving all raw rows to one place, which federated learning avoids.", "",
          "## Caveats", "",
          "- Training time is wall-clock time of a sequential single-process simulation on CPU; in a real deployment "
          "clients would train in parallel and network latency would dominate.",
          "- The centralized MLP uses Adam with early stopping (the Phase-2 recipe); federated methods use plain SGD "
          "for a fixed number of rounds, the standard FedAvg/FedProx client optimizer.",
          "- Per-round test metrics are logged for convergence plots only; no round, hyperparameter or threshold "
          "is chosen using the test set.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------- main

def main(config_path: Path) -> None:
    t_total = time.perf_counter()
    cfg = load_config(config_path)
    out_dir = RESULTS_DIR / "federated" / cfg["output_tag"]
    fig_dir = out_dir / "figures"
    model_dir = ROOT / "models" / "federated" / cfg["output_tag"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    dcfg, fcfg, mcfg = cfg["data"], cfg["federated"], cfg["model"]
    seeds = [int(s) for s in cfg["seeds"]]
    recall_target = cfg["threshold"]["recall_target"]
    torch.set_num_threads(cfg.get("torch_threads", 4))

    df = load_primary(dcfg["file"])
    X, y = features_and_target(df)
    split = stratified_split(y, dcfg["test_size"], dcfg["val_size"], dcfg["split_seed"])
    test_set = set(split.test.tolist())
    local_cfg = LocalTrainConfig(lr=fcfg["lr"], batch_size=fcfg["batch_size"], local_epochs=fcfg["local_epochs"],
                                 momentum=fcfg.get("momentum", 0.0), weight_decay=fcfg.get("weight_decay", 0.0))
    mlp_cfg = MLPTrainConfig(hidden_sizes=mcfg["hidden_sizes"], dropout=mcfg["dropout"], **cfg["reference_training"])

    def model_fn():
        return build_mlp(len(FEATURES), mcfg["hidden_sizes"], mcfg["dropout"])

    rows, histories, part_tables, comm_rows, refs = [], [], [], [], []
    partitions_meta = {}
    central_cache: dict[int, dict] = {}

    for spec in cfg["partitions"]:
        part = make_partition(spec, split.train, split.val, y, cfg["num_clients"], cfg["partition_seed"])
        for c in part.clients:
            assert not (set(c.train_idx.tolist()) | set(c.val_idx.tolist())) & test_set, "test row reached a client"
        part_tables.append(partition_summary(part, y))
        partitions_meta[part.name] = {**part.params, "draws": part.draws,
                                      "proportions": np.round(part.proportions, 6).tolist()}
        clients, fed_pre, pos_weight = build_clients(part, X, y)
        X_val = np.concatenate([c.X_val.numpy() for c in clients])
        y_val = np.concatenate([c.y_val.numpy() for c in clients]).astype(int)
        X_test = fed_pre.transform(X.iloc[split.test])
        y_test = y[split.test]
        evaluator = GlobalEvaluator(X_val, y_val, X_test, y_test, pos_weight)
        print(f"\n== Partition {part.name} ({part.draws} draw(s)); global pos_weight {pos_weight:.4f}")
        print(partition_summary(part, y).query("split == 'train'")[["client", "n", "n_positive", "positive_rate"]].to_string(index=False))

        for seed in seeds:
            base = {"partition": part.name, "seed": seed}

            # Centralized reference: pooled client training rows == global train split.
            if seed not in central_cache:
                set_global_seed(seed)
                Xtr_c = fed_pre.transform(X.iloc[split.train])
                t0 = time.perf_counter()
                model, _, info = fit_mlp(Xtr_c, y[split.train], fed_pre.transform(X.iloc[split.val]), y[split.val], mlp_cfg, seed)
                central_cache[seed] = {"model": model, "seconds": time.perf_counter() - t0, "info": info}
            cm = central_cache[seed]["model"]
            pv_c, pt_c = predict_proba(cm, evaluator.X_val), predict_proba(cm, evaluator.X_test)
            rows += score_rows(pv_c, y_val, pt_c, y_test, recall_target,
                               {**base, "method": "centralized", "client": -1,
                                "train_seconds": central_cache[seed]["seconds"], "rounds": 0, "total_bytes": 0})
            central_val_auc = roc_auc_score(y_val, pv_c)
            refs.append({**base, "method": "centralized", "val_roc_auc": central_val_auc,
                         "test_roc_auc": roc_auc_score(y_test, pt_c)})
            comm_rows.append({**base, "method": "centralized", "train_seconds": central_cache[seed]["seconds"],
                              "rounds": 0, "total_bytes": 0, "rounds_to_target": np.nan, "bytes_to_target": np.nan})

            # Local-only: each client trains alone on its own rows with its own scaler and class weight.
            if cfg.get("run_local_only", True):
                for c in part.clients:
                    local_pre = FeaturePreprocessor().fit(X.iloc[c.train_idx])
                    Xtr_l, Xva_l = local_pre.transform(X.iloc[c.train_idx]), local_pre.transform(X.iloc[c.val_idx])
                    ytr_l, yva_l = y[c.train_idx], y[c.val_idx]
                    t0 = time.perf_counter()
                    lm, _, _ = fit_mlp(Xtr_l, ytr_l, Xva_l, yva_l, mlp_cfg, seed)
                    secs = time.perf_counter() - t0
                    pv_l = predict_proba(lm, Xva_l)
                    pt_l = predict_proba(lm, local_pre.transform(X.iloc[split.test]))
                    rows += score_rows(pv_l, yva_l, pt_l, y_test, recall_target,
                                       {**base, "method": "local_only", "client": c.client_id,
                                        "train_seconds": secs, "rounds": 0, "total_bytes": 0})
                    refs.append({**base, "method": "local_only", "client": c.client_id,
                                 "val_roc_auc": roc_auc_score(yva_l, pv_l), "test_roc_auc": roc_auc_score(y_test, pt_l)})
                    comm_rows.append({**base, "method": f"local_only_client{c.client_id}", "train_seconds": secs,
                                      "rounds": 0, "total_bytes": 0, "rounds_to_target": np.nan, "bytes_to_target": np.nan})
                print(f"seed {seed} local-only done")

            # FedAvg and FedProx share the model_fn, initialization seed, clients and local settings.
            servers = [build_fedavg_server(model_fn, clients, local_cfg, fcfg["client_fraction"], seed)]
            servers += [build_fedprox_server(model_fn, clients, local_cfg, fcfg["client_fraction"], seed, float(mu))
                        for mu in fcfg["fedprox_mu"]]
            for server in servers:
                hist, secs = run_federated(server, fcfg["rounds"], evaluator,
                                           model_dir / "checkpoints" / f"{part.name}_seed{seed}",
                                           int(fcfg.get("checkpoint_every", 0)))
                h = pd.DataFrame(hist).assign(**base)
                histories.append(h)
                gm = server.global_model
                pv, pt = predict_proba(gm, evaluator.X_val), predict_proba(gm, evaluator.X_test)
                total = server.bytes_down + server.bytes_up
                rows += score_rows(pv, y_val, pt, y_test, recall_target,
                                   {**base, "method": server.algorithm, "client": -1, "train_seconds": secs,
                                    "rounds": fcfg["rounds"], "total_bytes": total})
                r_hit, b_hit = rounds_to_target(h, central_val_auc - cfg["target_tolerance"])
                comm_rows.append({**base, "method": server.algorithm, "train_seconds": secs, "rounds": fcfg["rounds"],
                                  "total_bytes": total, "bytes_down": server.bytes_down, "bytes_up": server.bytes_up,
                                  "rounds_to_target": r_hit, "bytes_to_target": b_hit})
                torch.save({"state_dict": gm.state_dict(), "config": gm.config, "algorithm": server.algorithm,
                            "partition": part.name, "seed": seed, "rounds": fcfg["rounds"]},
                           model_dir / f"{part.name}_{server.algorithm}_seed{seed}.pt")
                last = hist[-1]
                print(f"seed {seed} {server.algorithm:16s} {secs:7.1f}s  final val AUC {last['val_roc_auc']:.4f}  "
                      f"test AUC {last['test_roc_auc']:.4f}  bytes {total:,}")
        fed_pre.save(model_dir / f"{part.name}_federated_preprocessor.joblib")

    results = pd.DataFrame(rows)
    history = pd.concat(histories, ignore_index=True)
    part_summary = pd.concat(part_tables, ignore_index=True)
    comm = pd.DataFrame(comm_rows)
    refs_df = pd.DataFrame(refs)

    metric_cols = ["threshold", "accuracy", "balanced_accuracy", "precision_class0", "recall_class0",
                   "precision_class1", "recall_class1", "f1_class1", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp",
                   "train_seconds", "total_bytes"]
    test = results[results.split == "test"]
    summary = test.groupby(["partition", "method", "operating_point"], sort=False)[metric_cols].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    comm_summary = comm.assign(method=comm.method.str.replace(r"local_only_client\d+", "local_only (sum of clients)", regex=True)) \
        .groupby(["partition", "method", "seed"], sort=False).agg(
            train_seconds=("train_seconds", "sum"), rounds=("rounds", "max"), total_bytes=("total_bytes", "sum"),
            rounds_to_target=("rounds_to_target", "first"), bytes_to_target=("bytes_to_target", "first")) \
        .groupby(["partition", "method"], sort=False).mean().reset_index()

    results.to_csv(out_dir / "federated_results.csv", index=False)
    summary.to_csv(out_dir / "federated_summary.csv", index=False)
    history.to_csv(out_dir / "federated_round_history.csv", index=False)
    part_summary.to_csv(out_dir / "client_partitions.csv", index=False)
    comm.to_csv(out_dir / "federated_cost.csv", index=False)
    comm_summary.to_csv(out_dir / "federated_cost_summary.csv", index=False)

    plot_client_distributions(part_summary, fig_dir / "client_distributions.png")
    for name in part_summary.partition.unique():
        plot_convergence(history, refs_df, name, fig_dir / f"convergence_{name}.png")
        cm_rows = test[(test.partition == name) & (test.seed == seeds[0]) & (test.operating_point == "val_max_f1")
                       & (test.method != "local_only")].rename(columns={"method": "model"})
        plotting.plot_confusion_matrices(cm_rows, fig_dir / f"confusion_{name}_val_max_f1.png",
                                         f"Test confusion matrices, partition '{name}', val max-F1 threshold (seed {seeds[0]})")
    plot_comparison(test, fig_dir / "method_comparison.png")

    probe = model_fn()
    meta = {
        "disclaimer": DISCLAIMER, "config_file": str(config_path.as_posix()), "config": cfg,
        "data_file": dcfg["file"], "data_sha256": sha256(RAW_DIR / dcfg["file"]), "split_seed": dcfg["split_seed"],
        "split_sizes": split.sizes(), "seeds": seeds, "partition_seed": cfg["partition_seed"],
        "num_clients": cfg["num_clients"], "partitions": partitions_meta, "federated": fcfg, "model": mcfg,
        "model_parameters": count_parameters(probe), "model_bytes": state_nbytes(probe.state_dict()),
        "target_tolerance": cfg["target_tolerance"], "runtime_seconds_total": time.perf_counter() - t_total,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
                        "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                        "cuda_used": False, "sklearn": sklearn.__version__, "numpy": np.__version__},
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    write_report(meta, part_summary, summary, comm_summary, out_dir / "report.md")

    view = summary[summary.operating_point == "val_max_f1"][
        ["partition", "method", "roc_auc_mean", "accuracy_mean", "precision_class1_mean", "recall_class1_mean",
         "f1_class1_mean", "roc_auc_min", "roc_auc_max"]]
    with pd.option_context("display.width", 250):
        print("\nTest split, val max-F1 threshold:\n" + view.round(4).to_string(index=False))
        print("\nCost:\n" + comm_summary.round(3).to_string(index=False))
    print(f"\nTotal runtime {meta['runtime_seconds_total']:.1f}s. Outputs in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    main(parser.parse_args().config)
