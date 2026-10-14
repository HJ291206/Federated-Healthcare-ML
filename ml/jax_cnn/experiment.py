"""JAX 1D-CNN vs JAX MLP vs PyTorch MLP on the same split, with a fair timing protocol.

Run from the repository root:
    python -m ml.jax_cnn.experiment [--config configs/jax_cnn.yaml]

Fairness of the benchmark: same rows (centralized split, seed 42), same preprocessing,
same class weighting, same optimizer (Adam + L2 weight decay), same batch size, same
early-stopping rule on validation ROC-AUC, same seeds, same machine, run back to back in
one process. JAX times include JIT compilation (reported separately as the first-epoch
time). CPU thread usage differs between XLA and PyTorch and is not equalized, so timing
differences are indicative, not a definitive framework comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

from ml.data import features_and_target, load_primary, stratified_split
from ml.data_validation import FEATURES, RAW_DIR, RESULTS_DIR, TARGET, sha256
from ml.evaluation import classification_metrics, select_thresholds
from ml.jax_cnn.model import CNN1D, MLP, count_parameters
from ml.models import count_parameters as torch_count_parameters
from ml.preprocessing import FeaturePreprocessor
from ml.training import MLPTrainConfig, fit_mlp, positive_class_weight, predict_proba

ROOT = Path(__file__).resolve().parents[2]


def weighted_bce(logits: jnp.ndarray, y: jnp.ndarray, pos_weight: float) -> jnp.ndarray:
    """Mean of pos_weight-weighted BCE, matching torch.nn.BCEWithLogitsLoss(pos_weight=...)."""
    per_example = optax.sigmoid_binary_cross_entropy(logits, y)
    weights = jnp.where(y == 1, pos_weight, 1.0)
    return jnp.mean(weights * per_example)


def fit_jax(model, X_train, y_train, X_val, y_val, cfg: dict, seed: int) -> tuple[dict, pd.DataFrame, dict]:
    key = jax.random.PRNGKey(seed)
    key, init_key, drop_key = jax.random.split(key, 3)
    params = model.init({"params": init_key, "dropout": drop_key}, jnp.zeros((1, X_train.shape[1]), jnp.float32), train=False)["params"]
    tx = optax.chain(optax.add_decayed_weights(cfg["weight_decay"]), optax.adam(cfg["lr"]))
    opt_state = tx.init(params)
    pos_weight = positive_class_weight(y_train)

    @jax.jit
    def train_step(params, opt_state, xb, yb, dkey):
        def loss_fn(p):
            return weighted_bce(model.apply({"params": p}, xb, train=True, rngs={"dropout": dkey}), yb, pos_weight)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def predict(params, xb):
        return jax.nn.sigmoid(model.apply({"params": params}, xb, train=False))

    def predict_all(params, X):
        return np.concatenate([np.asarray(predict(params, jnp.asarray(X[i:i + 16384]))) for i in range(0, len(X), 16384)])

    rng = np.random.default_rng(seed)
    n, bs = len(X_train), cfg["batch_size"]
    Xtr, ytr = jnp.asarray(X_train), jnp.asarray(y_train, jnp.float32)
    best_auc, best_epoch, best_params, stale, history = -np.inf, 0, params, 0, []
    t_start = time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.perf_counter()
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            key, dkey = jax.random.split(key)
            params, opt_state, loss = train_step(params, opt_state, Xtr[idx], ytr[idx], dkey)
            losses.append(loss)
        train_loss = float(np.mean(np.asarray(losses)))
        val_auc = roc_auc_score(y_val, predict_all(params, X_val))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_roc_auc": val_auc,
                        "epoch_seconds": time.perf_counter() - t0})
        if val_auc > best_auc + 1e-5:
            best_auc, best_epoch, best_params, stale = val_auc, epoch, params, 0
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    info = {"best_epoch": best_epoch, "epochs_run": len(history), "best_val_roc_auc": best_auc,
            "train_seconds": time.perf_counter() - t_start, "first_epoch_seconds": history[0]["epoch_seconds"],
            "n_parameters": count_parameters(best_params), "pos_weight": pos_weight}
    return best_params, pd.DataFrame(history), {**info, "predict": lambda X: predict_all(best_params, X)}


def main(config_path: Path) -> None:
    t_total = time.perf_counter()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out_dir = RESULTS_DIR / "jax"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tcfg = cfg["training"]
    torch.set_num_threads(cfg.get("torch_threads", 4))

    df = load_primary(cfg["data"]["file"])
    X, y = features_and_target(df)
    split = stratified_split(y, cfg["data"]["test_size"], cfg["data"]["val_size"], cfg["data"]["split_seed"])
    pre = FeaturePreprocessor().fit(X.iloc[split.train])
    Xtr, Xva, Xte = (pre.transform(X.iloc[i]) for i in (split.train, split.val, split.test))
    ytr, yva, yte = y[split.train], y[split.val], y[split.test]

    orders = {"original": np.arange(len(FEATURES))}
    for s in cfg["cnn"]["permutation_seeds"]:
        orders[f"permuted_seed{s}"] = np.random.default_rng(s).permutation(len(FEATURES))

    rows, histories = [], []

    def record(name, framework, order_name, seed, p_val, p_test, info):
        thresholds = select_thresholds(yva, p_val, cfg["threshold"]["recall_target"])  # validation only
        for op, thr in thresholds.items():
            for split_name, yt, pp in (("val", yva, p_val), ("test", yte, p_test)):
                rows.append({"model": name, "framework": framework, "feature_order": order_name, "seed": seed,
                             "split": split_name, "operating_point": op, **classification_metrics(yt, pp, thr),
                             "train_seconds": info["train_seconds"], "first_epoch_seconds": info["first_epoch_seconds"],
                             "epochs_run": info["epochs_run"], "best_epoch": info["best_epoch"],
                             "n_parameters": info["n_parameters"]})

    for seed in cfg["seeds"]:
        # PyTorch MLP (same recipe as the centralized phase), timed in this process for fairness.
        mcfg = MLPTrainConfig(hidden_sizes=cfg["mlp"]["hidden_sizes"], dropout=cfg["mlp"]["dropout"], lr=tcfg["lr"],
                              weight_decay=tcfg["weight_decay"], batch_size=tcfg["batch_size"],
                              max_epochs=tcfg["max_epochs"], patience=tcfg["patience"],
                              device=cfg["mlp"].get("device", "cpu"),
                              torch_threads=cfg.get("torch_threads", 4))
        t0 = time.perf_counter()
        tm, th, tinfo = fit_mlp(Xtr, ytr, Xva, yva, mcfg, seed)
        tinfo.update({"train_seconds": time.perf_counter() - t0, "first_epoch_seconds": th.epoch_seconds.iloc[0],
                      "n_parameters": torch_count_parameters(tm)})
        record("mlp", "pytorch", "original", seed, predict_proba(tm, Xva), predict_proba(tm, Xte), tinfo)
        histories.append(th.assign(model="mlp", framework="pytorch", feature_order="original", seed=seed))
        print(f"seed {seed} pytorch mlp   {tinfo['train_seconds']:6.1f}s  {tinfo['epochs_run']} epochs  val AUC {tinfo['best_val_roc_auc']:.4f}")

        # JAX MLP control (same architecture).
        jm = MLP(tuple(cfg["mlp"]["hidden_sizes"]), cfg["mlp"]["dropout"])
        _, jh, jinfo = fit_jax(jm, Xtr, ytr, Xva, yva, tcfg, seed)
        record("mlp", "jax", "original", seed, jinfo["predict"](Xva), jinfo["predict"](Xte), jinfo)
        histories.append(jh.assign(model="mlp", framework="jax", feature_order="original", seed=seed))
        print(f"seed {seed} jax mlp       {jinfo['train_seconds']:6.1f}s  {jinfo['epochs_run']} epochs  val AUC {jinfo['best_val_roc_auc']:.4f}")

        # JAX 1D CNN, original and permuted column orders.
        for order_name, order in orders.items():
            cnn = CNN1D(tuple(cfg["cnn"]["channels"]), cfg["cnn"]["kernel_size"], cfg["cnn"]["hidden"], cfg["cnn"]["dropout"])
            _, ch, cinfo = fit_jax(cnn, Xtr[:, order], ytr, Xva[:, order], yva, tcfg, seed)
            record("cnn1d", "jax", order_name, seed, cinfo["predict"](Xva[:, order]), cinfo["predict"](Xte[:, order]), cinfo)
            histories.append(ch.assign(model="cnn1d", framework="jax", feature_order=order_name, seed=seed))
            print(f"seed {seed} jax cnn1d {order_name:16s} {cinfo['train_seconds']:6.1f}s  {cinfo['epochs_run']} epochs  val AUC {cinfo['best_val_roc_auc']:.4f}")

    results = pd.DataFrame(rows)
    history = pd.concat(histories, ignore_index=True)
    results.to_csv(out_dir / "jax_results.csv", index=False)
    history.to_csv(out_dir / "jax_history.csv", index=False)
    test = results[(results.split == "test") & (results.operating_point == "val_max_f1")]
    summary = test.groupby(["model", "framework", "feature_order"], sort=False).agg(
        roc_auc=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"), pr_auc=("pr_auc", "mean"),
        accuracy=("accuracy", "mean"), recall_class1=("recall_class1", "mean"), f1_class1=("f1_class1", "mean"),
        train_seconds=("train_seconds", "mean"), first_epoch_seconds=("first_epoch_seconds", "mean"),
        epochs_run=("epochs_run", "mean"), n_parameters=("n_parameters", "first"), n_seeds=("seed", "nunique")).reset_index()
    summary["seconds_per_epoch"] = summary.train_seconds / summary.epochs_run
    summary.to_csv(out_dir / "jax_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    labels = [f"{r.framework}\n{r.model}\n{r.feature_order}" for _, r in summary.iterrows()]
    axes[0].bar(labels, summary.roc_auc, yerr=summary.roc_auc_std.fillna(0), capsize=3, color="#4C72B0")
    axes[0].set(ylabel="Test ROC-AUC", ylim=(summary.roc_auc.min() - 0.01, summary.roc_auc.max() + 0.005))
    axes[1].bar(labels, summary.train_seconds, color="#DD8452", label="total (incl. JIT)")
    axes[1].bar(labels, summary.first_epoch_seconds, color="#C44E52", label="first epoch (JIT + warm-up)")
    axes[1].set(ylabel="Training seconds (mean over seeds)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("JAX 1D-CNN vs JAX MLP vs PyTorch MLP, same split / recipe / machine (CPU)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "jax_vs_pytorch.png", dpi=150)
    plt.close(fig)

    meta = {"config": cfg, "data_sha256": sha256(RAW_DIR / cfg["data"]["file"]), "split_sizes": split.sizes(),
            "jax": jax.__version__, "jax_devices": [str(d) for d in jax.devices()], "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(), "cpu_count": os.cpu_count(), "platform": platform.platform(),
            "python": sys.version.split()[0], "runtime_seconds_total": time.perf_counter() - t_total,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    L = ["# JAX 1D-CNN experiment", "",
         "The 21 features are tabular survey answers with an arbitrary column order; a 1D convolution over them "
         "mixes neighbouring columns that have no medical relationship. The CNN is an experimental representation, "
         "not an image-style CNN with a physical meaning. Rows `permuted_seed*` retrain the CNN on shuffled column "
         "orders: if results change, the architecture depends on an arbitrary convention.", "",
         f"Generated on {meta['finished_at']} from `results/jax/jax_results.csv`. Seeds: {cfg['seeds']}. "
         f"JAX {jax.__version__} on {meta['jax_devices']}, PyTorch {torch.__version__} ({torch.get_num_threads()} threads). "
         "Timing is wall-clock on one CPU machine, run back to back; XLA and PyTorch thread usage is not equalized.", "",
         "| Framework | Model | Feature order | Params | Test ROC-AUC | PR-AUC | Accuracy | Recall (1) | F1 (1) | Epochs | Train s | First-epoch s | s/epoch |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        L.append(f"| {r.framework} | {r.model} | {r.feature_order} | {r.n_parameters:,} | {r.roc_auc:.4f} ± {0 if pd.isna(r.roc_auc_std) else r.roc_auc_std:.4f} | "
                 f"{r.pr_auc:.4f} | {r.accuracy:.4f} | {r.recall_class1:.4f} | {r.f1_class1:.4f} | {r.epochs_run:.1f} | "
                 f"{r.train_seconds:.1f} | {r.first_epoch_seconds:.2f} | {r.seconds_per_epoch:.2f} |")
    L += ["", "Operating point: validation max-F1 threshold; thresholds and early stopping use validation data only.", ""]
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    with pd.option_context("display.width", 220):
        print("\n" + summary[["framework", "model", "feature_order", "roc_auc", "recall_class1", "f1_class1",
                              "epochs_run", "train_seconds", "first_epoch_seconds"]].round(4).to_string(index=False))
    print(f"\nTotal runtime {meta['runtime_seconds_total']:.1f}s. Outputs in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "jax_cnn.yaml")
    main(parser.parse_args().config)
