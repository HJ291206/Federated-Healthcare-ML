"""Sequential vs Ray execution of the identical FedAvg / FedProx workload.

Run from the repository root:
    python -m ml.federated.ray_benchmark [--config configs/ray_benchmark.yaml]

Checks that the Ray backend reproduces the sequential global model exactly and reports
wall-clock time for both. Ray adds process start-up and object-transfer overhead, so it
only pays off when per-client training is long enough; the numbers here are for this
laptop's CPU and this small MLP.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import torch
import yaml

from ml.data import features_and_target, load_primary, stratified_split
from ml.data_validation import FEATURES, RESULTS_DIR
from ml.federated.client import LocalTrainConfig
from ml.federated.experiment import DISCLAIMER, GlobalEvaluator, build_clients
from ml.federated.fedavg import fedavg_aggregate
from ml.federated.fedprox import make_proximal_regularizer
from ml.federated.partition import make_partition
from ml.federated.ray_backend import RayFederatedServer, init_ray
from ml.federated.server import FederatedServer
from ml.models import build_mlp

ROOT = Path(__file__).resolve().parents[2]


def main(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out_dir = RESULTS_DIR / "federated" / "ray"
    out_dir.mkdir(parents=True, exist_ok=True)
    dcfg, fcfg = cfg["data"], cfg["federated"]
    torch.set_num_threads(cfg["sequential_threads"])

    df = load_primary(dcfg["file"])
    X, y = features_and_target(df)
    split = stratified_split(y, dcfg["test_size"], dcfg["val_size"], dcfg["split_seed"])
    part = make_partition(cfg["partition"], split.train, split.val, y, cfg["num_clients"], cfg["partition_seed"])
    clients, fed_pre, pos_weight = build_clients(part, X, y)
    X_val = np.concatenate([c.X_val.numpy() for c in clients])
    y_val = np.concatenate([c.y_val.numpy() for c in clients]).astype(int)
    evaluator = GlobalEvaluator(X_val, y_val, fed_pre.transform(X.iloc[split.test]), y[split.test], pos_weight)
    local_cfg = LocalTrainConfig(lr=fcfg["lr"], batch_size=fcfg["batch_size"], local_epochs=fcfg["local_epochs"])

    def model_fn():
        return build_mlp(len(FEATURES), cfg["model"]["hidden_sizes"], cfg["model"]["dropout"])

    algos = {"fedavg": None}
    for mu in fcfg["fedprox_mu"]:
        algos[f"fedprox_mu{float(mu):g}"] = (lambda m: (lambda gp: make_proximal_regularizer(gp, m)))(float(mu))

    t0 = time.perf_counter()
    started = init_ray(num_cpus=cfg["ray_num_cpus"])
    ray_startup = time.perf_counter() - t0
    rows, states = [], {}
    try:
        # "sequential" = speed baseline with cfg["sequential_threads"] torch threads.
        # "sequential_matched" = same torch thread count as each Ray actor; the only fair numerical
        # reference, since intra-op thread count changes floating-point reduction order.
        backends = {"sequential": cfg["sequential_threads"], "sequential_matched": cfg["ray_actor_threads"], "ray": None}
        for name, factory in algos.items():
            for backend, threads in backends.items():
                if backend == "ray":
                    server = RayFederatedServer(model_fn, clients, local_cfg, 1.0, cfg["seed"], fedavg_aggregate, factory,
                                                name, num_cpus=cfg["ray_num_cpus"], actor_threads=cfg["ray_actor_threads"])
                else:
                    torch.set_num_threads(threads)
                    server = FederatedServer(model_fn, clients, local_cfg, 1.0, cfg["seed"], fedavg_aggregate, factory, name)
                t0 = time.perf_counter()
                hist = server.fit(fcfg["rounds"], evaluator)
                secs = time.perf_counter() - t0
                states[(name, backend)] = server.global_state()
                if backend == "ray":
                    server.shutdown()
                rows.append({"algorithm": name, "backend": backend, "torch_threads": threads or cfg["ray_actor_threads"],
                             "rounds": fcfg["rounds"], "seconds": secs, "seconds_per_round": secs / fcfg["rounds"],
                             "mean_max_client_seconds": float(np.mean([h["max_client_seconds"] for h in hist])),
                             "final_val_roc_auc": hist[-1]["val_roc_auc"], "final_test_roc_auc": hist[-1]["test_roc_auc"],
                             "total_bytes": server.bytes_down + server.bytes_up})
                print(f"{name:15s} {backend:19s} {secs:7.1f}s  val AUC {hist[-1]['val_roc_auc']:.4f}")
            par = states[(name, "ray")]
            for r in rows:
                if r["algorithm"] == name:
                    ref = states[(name, r["backend"])]
                    r["max_abs_param_diff_vs_ray"] = max(float((ref[k] - par[k]).abs().max()) for k in ref)
                    if r["backend"] != "ray":
                        print(f"{name}: max |param_{r['backend']} - param_ray| = {r['max_abs_param_diff_vs_ray']:.3e}")
            torch.set_num_threads(cfg["sequential_threads"])
    finally:
        if started and ray.is_initialized():
            ray.shutdown()

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "ray_benchmark.csv", index=False)
    meta = {"disclaimer": DISCLAIMER, "config": cfg, "ray": ray.__version__, "torch": torch.__version__,
            "ray_startup_seconds": ray_startup, "cpu_count": os.cpu_count(), "platform": platform.platform(),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    L = ["# Sequential vs Ray execution", "", f"> {DISCLAIMER}", "",
         "Ray changes where client training runs, not what is shared: it is distributed execution, not a privacy "
         "mechanism. With the same torch thread count per client (`sequential_matched`), the Ray backend must "
         "reproduce the sequential global model exactly (max |Δparam| = 0). The `sequential` row uses more intra-op "
         "threads for speed; any non-zero difference there comes from floating-point reduction order, not from the "
         "algorithm.", "",
         f"Generated {meta['finished_at']}. Ray {ray.__version__}, start-up {ray_startup:.1f} s (not included in the "
         f"per-run times), {cfg['ray_num_cpus']} CPUs for Ray, {cfg['ray_actor_threads']} torch thread(s) per actor, "
         f"{cfg['sequential_threads']} torch threads for the sequential backend, {cfg['num_clients']} clients, "
         f"{fcfg['rounds']} rounds, {fcfg['local_epochs']} local epoch(s), batch {fcfg['batch_size']}.", "",
         "| Algorithm | Backend | Torch threads/client | Seconds | s/round | Mean slowest-client s/round | Final val ROC-AUC | Max |Δparam| vs Ray |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in results.iterrows():
        L.append(f"| {r.algorithm} | {r.backend} | {r.torch_threads} | {r.seconds:.1f} | {r.seconds_per_round:.2f} | "
                 f"{r.mean_max_client_seconds:.2f} | {r.final_val_roc_auc:.4f} | {r.max_abs_param_diff_vs_ray:.1e} |")
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nRay start-up {ray_startup:.1f}s. Outputs in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ray_benchmark.yaml")
    main(parser.parse_args().config)
