"""End-to-end smoke tests on the real primary CSV (skipped if the data is absent).

Runs the federated pipeline through the public entry point with a tiny budget, then checks that every
expected artefact exists, that the test split never reached a client, and that the written metrics are
internally consistent. Outputs go to a throw-away tag that is deleted afterwards.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.data_validation import PRIMARY_FILE, RAW_DIR, RESULTS_DIR

ROOT = Path(__file__).resolve().parents[1]
TAG = "e2e_smoke_test"
pytestmark = pytest.mark.skipif(not (RAW_DIR / PRIMARY_FILE).is_file(), reason="primary CSV not present")


@pytest.fixture(scope="module")
def smoke_run():
    out_dir, model_dir = RESULTS_DIR / "federated" / TAG, ROOT / "models" / "federated" / TAG
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(model_dir, ignore_errors=True)
    cmd = [sys.executable, "-m", "ml.run", "federated", "--config", "configs/federated_fast.yaml",
           "--set", f"output_tag={TAG}", "--set", "seeds=[0]", "--set", "federated.rounds=2",
           "--set", "reference_training.max_epochs=1", "--set", "run_local_only=true",
           "--set", "partitions=[{method: iid}, {method: dirichlet, alpha: 0.5}]"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    yield out_dir, model_dir
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(model_dir, ignore_errors=True)


def test_artifacts_written(smoke_run):
    out_dir, model_dir = smoke_run
    for name in ("federated_results.csv", "federated_summary.csv", "federated_round_history.csv",
                 "client_partitions.csv", "federated_cost.csv", "run_metadata.json", "report.md"):
        assert (out_dir / name).is_file(), name
    assert (out_dir / "figures").is_dir() and any((out_dir / "figures").glob("*.png"))
    assert (model_dir / "iid_fedavg_seed0.pt").is_file() and (model_dir / "iid_fedprox_mu0.01_seed0.pt").is_file()


def test_no_test_rows_reach_clients_and_metrics_are_consistent(smoke_run):
    out_dir, _ = smoke_run
    meta = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    parts = pd.read_csv(out_dir / "client_partitions.csv")
    n_total = 253_680
    test_size = meta["config"]["data"]["test_size"]
    # Each partition's clients hold exactly the train+val rows, i.e. (1 - test_size) of the data.
    for name, g in parts.groupby("partition"):
        assert g.n.sum() == pytest.approx(n_total * (1 - test_size), abs=2), name
        assert (g.n_positive >= 5).all()
    assert parts.client.nunique() >= 5

    res = pd.read_csv(out_dir / "federated_results.csv")
    assert {"centralized", "local_only", "fedavg", "fedprox_mu0.01"} <= set(res.method)
    test = res[res.split == "test"]
    assert (test.support_class0 + test.support_class1 == round(n_total * test_size)).all()
    assert ((test.tn + test.fp + test.fn + test.tp) == test.support_class0 + test.support_class1).all()
    assert test.roc_auc.between(0.5, 1.0).all()
    # Validation-chosen thresholds, re-applied: recall target row must reach the target on validation.
    val_recall = res[(res.split == "val") & (res.operating_point == "val_recall_0.80")]
    assert (val_recall.recall_class1 >= 0.80 - 1e-9).all()

    hist = pd.read_csv(out_dir / "federated_round_history.csv")
    assert set(hist["round"]) == {1, 2}
    payload = 3521 * 4
    assert (hist.bytes_down == 5 * payload).all() and (hist.bytes_up == 5 * payload).all()
    cost = pd.read_csv(out_dir / "federated_cost.csv")
    fed_cost = cost[cost.method.isin(["fedavg", "fedprox_mu0.01"])]
    assert (fed_cost.total_bytes == 2 * 2 * 5 * payload).all()
    assert np.isfinite(fed_cost.train_seconds).all()


def test_cuda_request_fails_loudly_on_cpu_build():
    import torch

    from ml.training import resolve_device

    if torch.cuda.is_available():
        pytest.skip("CUDA available; nothing to check")
    with pytest.raises(RuntimeError, match="CUDA requested"):
        resolve_device("cuda")
    assert resolve_device("auto").type == "cpu"
