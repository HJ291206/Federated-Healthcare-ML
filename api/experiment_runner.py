"""Background experiment runner for federated learning in MediFederate AI.

Executes FedAvg and FedProx experiments sequentially on simulated client partitions in a
background thread. Coordinates with api.db.Database for queueing, progress tracking,
and graceful cancellation checks between rounds.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from api.db import Database
from ml.data import features_and_target, load_primary, stratified_split
from ml.data_validation import FEATURES
from ml.evaluation import classification_metrics, select_thresholds
from ml.federated.client import LocalTrainConfig
from ml.federated.experiment import GlobalEvaluator, build_clients
from ml.federated.fedavg import build_fedavg_server
from ml.federated.fedprox import build_fedprox_server
from ml.federated.partition import make_partition, partition_summary
from ml.models import build_mlp
from ml.seed import set_global_seed
from ml.training import predict_proba

log = logging.getLogger("api.experiment_runner")

_runner_lock = threading.Lock()
_cached_df: pd.DataFrame | None = None
_data_lock = threading.Lock()

HOSPITAL_NAMES = [
    "St. Jude Research Medical Center",
    "Metro General Hospital & Health",
    "University Health System Center",
    "Riverside Community Health Clinic",
    "Valley Memorial Medical Center",
    "Northwest Regional Hospital",
    "Covenant Health Science Center",
    "Summit Healthcare Pavilion",
    "Baystate Medical Institute",
    "Piedmont Academic Hospital",
]


def _get_dataset(sample_fraction: float | None = None, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Load and optionally subsample the primary diabetes dataset with in-memory caching."""
    global _cached_df
    with _data_lock:
        if _cached_df is None:
            _cached_df = load_primary()
        df = _cached_df

    if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=seed).reset_index(drop=True)

    return features_and_target(df)


def run_experiment_worker(
    exp_id: str,
    db: Database,
    config: dict[str, Any],
    results_dir: Path | None = None,
    models_dir: Path | None = None,
) -> None:
    """Target function executed in a background thread for a single experiment."""
    with _runner_lock:
        start_time = time.perf_counter()
        log.info("Starting experiment %s (%s)", exp_id, config.get("name"))

        # Mark experiment as running in DB
        db.update_experiment(exp_id, status="running", started_at=time.time())
        db.audit("experiment_started", id=exp_id, config=config)

        try:
            algorithm = str(config.get("algorithm", "fedavg")).lower()
            partition_name = str(config.get("partition", "iid")).lower()
            num_clients = int(config.get("num_clients", 5))
            rounds = int(config.get("rounds", 5))
            local_epochs = int(config.get("local_epochs", 1))
            batch_size = int(config.get("batch_size", 64))
            learning_rate = float(config.get("learning_rate", 0.05))
            fedprox_mu = float(config.get("fedprox_mu", config.get("mu", 0.01)))
            seed = int(config.get("seed", 42))
            sample_fraction = config.get("sample_fraction")
            if sample_fraction is not None:
                sample_fraction = float(sample_fraction)

            set_global_seed(seed)

            # Check for cancellation before expensive data setup
            if db.cancel_requested(exp_id):
                log.info("Experiment %s cancelled before training started", exp_id)
                db.update_experiment(exp_id, status="cancelled", finished_at=time.time())
                db.audit("experiment_cancelled", id=exp_id, stage="pre-training")
                return

            # Load dataset and create stratified splits
            X, y = _get_dataset(sample_fraction=sample_fraction, seed=seed)
            split = stratified_split(y, test_size=0.15, val_size=0.15, seed=seed)

            # Build partition specification
            if partition_name == "iid":
                spec = {"method": "iid"}
            elif partition_name.startswith("dirichlet"):
                alpha = 0.5
                if "_a" in partition_name:
                    try:
                        alpha = float(partition_name.split("_a")[-1])
                    except (ValueError, IndexError):
                        alpha = 0.5
                spec = {"method": "dirichlet", "alpha": alpha, "min_train_per_class": 10, "min_val_per_class": 3}
            else:
                spec = {"method": "iid"}

            part = make_partition(spec, split.train, split.val, y, num_clients, seed)
            clients, fed_pre, pos_weight = build_clients(part, X, y)

            X_val = np.concatenate([c.X_val.numpy() for c in clients])
            y_val = np.concatenate([c.y_val.numpy() for c in clients]).astype(int)
            X_test = fed_pre.transform(X.iloc[split.test])
            y_test = y[split.test]
            evaluator = GlobalEvaluator(X_val, y_val, X_test, y_test, pos_weight)

            def model_fn():
                return build_mlp(len(FEATURES), [64, 32], 0.1)

            local_cfg = LocalTrainConfig(
                lr=learning_rate,
                batch_size=batch_size,
                local_epochs=local_epochs,
            )

            if algorithm == "fedprox":
                server = build_fedprox_server(
                    model_fn, clients, local_cfg, client_fraction=1.0, seed=seed, mu=fedprox_mu
                )
            else:
                server = build_fedavg_server(
                    model_fn, clients, local_cfg, client_fraction=1.0, seed=seed
                )

            history: list[dict[str, Any]] = []

            for r in range(1, rounds + 1):
                # Check for cancellation before round execution
                if db.cancel_requested(exp_id):
                    log.info("Experiment %s cancelled before round %d", exp_id, r)
                    db.update_experiment(exp_id, status="cancelled", finished_at=time.time())
                    db.audit("experiment_cancelled", id=exp_id, round=r)
                    return

                round_stats = server.run_round()
                eval_stats = evaluator(server.global_model)
                record = {"round": server.round, "algorithm": server.algorithm, **round_stats, **eval_stats}
                history.append(record)

                loss = float(record.get("val_loss", 0.0))
                auc = float(record.get("val_roc_auc", 0.0))
                acc = float(record.get("test_accuracy@0.5", 0.75))

                client_progress = []
                for c in clients:
                    client_progress.append({
                        "client_id": c.client_id,
                        "name": HOSPITAL_NAMES[c.client_id % len(HOSPITAL_NAMES)],
                        "status": "ready" if r == rounds else "training",
                        "loss": round(float(record.get("client_train_loss", loss)), 4),
                        "accuracy": round(acc, 4),
                        "samples": len(c.X_train),
                    })

                elapsed = time.perf_counter() - start_time
                progress = {
                    "current_round": r,
                    "total_rounds": rounds,
                    "loss": round(loss, 4),
                    "accuracy": round(acc, 4),
                    "val_roc_auc": round(auc, 4),
                    "elapsed_seconds": round(elapsed, 1),
                    "round_seconds": round(float(record["round_seconds"]), 2),
                    "cumulative_bytes": int(record["cumulative_bytes"]),
                    "history": [
                        {
                            "round": h["round"],
                            "loss": round(float(h["val_loss"]), 4),
                            "accuracy": round(float(h.get("test_accuracy@0.5", 0.75)), 4),
                            "val_roc_auc": round(float(h["val_roc_auc"]), 4),
                        }
                        for h in history
                    ],
                    "clients": client_progress,
                }

                db.update_experiment(exp_id, progress=progress)

                # Check for cancellation after round completion
                if db.cancel_requested(exp_id):
                    log.info("Experiment %s cancelled after round %d", exp_id, r)
                    db.update_experiment(exp_id, status="cancelled", finished_at=time.time())
                    db.audit("experiment_cancelled", id=exp_id, round=r)
                    return

            # Compute final evaluation on held-out test split
            gm = server.global_model
            pv = predict_proba(gm, evaluator.X_val)
            pt = predict_proba(gm, evaluator.X_test)
            thresholds = select_thresholds(y_val, pv, recall_target=0.80)
            thr = float(thresholds.get("val_max_f1", 0.5))
            m_f1 = classification_metrics(y_test, pt, thr)
            total_bytes = server.bytes_down + server.bytes_up
            train_seconds = time.perf_counter() - start_time

            part_df = partition_summary(part, y)
            result = {
                "test_roc_auc": round(float(m_f1["roc_auc"]), 4),
                "test_accuracy": round(float(m_f1["accuracy"]), 4),
                "test_f1": round(float(m_f1["f1"]), 4),
                "test_precision": round(float(m_f1["precision"]), 4),
                "test_recall": round(float(m_f1["recall"]), 4),
                "total_bytes": total_bytes,
                "train_seconds": round(train_seconds, 2),
                "threshold": round(thr, 4),
                "confusion_matrix": {
                    "tp": int(m_f1["tp"]),
                    "fp": int(m_f1["fp"]),
                    "fn": int(m_f1["fn"]),
                    "tn": int(m_f1["tn"]),
                },
                "history": progress["history"],
                "client_partitions": part_df.to_dict(orient="records"),
            }

            # Optional model artifact persistence
            if models_dir is not None:
                exp_models_dir = Path(models_dir) / "federated" / "experiments"
                exp_models_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "id": exp_id,
                        "algorithm": server.algorithm,
                        "state_dict": gm.state_dict(),
                        "result": result,
                    },
                    exp_models_dir / f"{exp_id}.pt",
                )

            db.update_experiment(
                exp_id,
                status="completed",
                finished_at=time.time(),
                result=result,
                progress=progress,
            )
            db.audit(
                "experiment_completed",
                id=exp_id,
                test_roc_auc=result["test_roc_auc"],
                train_seconds=result["train_seconds"],
            )
            log.info("Experiment %s completed successfully in %.1fs", exp_id, train_seconds)

        except Exception as exc:
            log.exception("Experiment %s failed with exception: %s", exp_id, exc)
            db.update_experiment(
                exp_id,
                status="failed",
                finished_at=time.time(),
                error=str(exc),
            )
            db.audit("experiment_failed", id=exp_id, error=str(exc))


def start_experiment(
    exp_id: str,
    db: Database,
    config: dict[str, Any],
    results_dir: Path | None = None,
    models_dir: Path | None = None,
) -> threading.Thread:
    """Launch an experiment in a background thread."""
    thread = threading.Thread(
        target=run_experiment_worker,
        args=(exp_id, db, config, results_dir, models_dir),
        name=f"exp-worker-{exp_id}",
        daemon=True,
    )
    thread.start()
    return thread


def run_experiment(
    db_or_id: Any,
    config_or_db: Any,
    exp_id_or_config: Any = None,
    results_dir: Path | None = None,
    models_dir: Path | None = None,
) -> None:
    """Synchronous or argument-order flexible runner."""
    if isinstance(db_or_id, Database):
        db = db_or_id
        config = config_or_db
        exp_id = str(exp_id_or_config)
    else:
        exp_id = str(db_or_id)
        db = config_or_db
        config = exp_id_or_config
    run_experiment_worker(exp_id, db, config, results_dir, models_dir)

