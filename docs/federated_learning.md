# Federated learning pipeline: methods

> **Simulated clients, not hospitals.** All clients in this project are partitions of a single public survey dataset (CDC BRFSS 2015, see `data/README.md`). They do not correspond to hospitals, clinics, states or any real institution. The dataset has no site variable, so client heterogeneity is *induced* by the partitioning method below. **Federated learning alone does not guarantee privacy.** Raw rows stay on the client, but model updates can still leak information (e.g. membership inference, gradient inversion). Formal guarantees require differential privacy, which is a separate, later phase.

Measured results are generated, not written here: see `results/federated/<tag>/report.md`.

## Code map

| File | Responsibility |
|---|---|
| `ml/federated/partition.py` | IID and Dirichlet label-skew partitioning of the global train and validation indices; per-client summaries |
| `ml/federated/client.py` | `FederatedClient`: holds its local rows, trains the received global model with SGD for `local_epochs` |
| `ml/federated/server.py` | `FederatedServer`: client sampling, broadcast, aggregation, byte accounting, per-round evaluation hook |
| `ml/federated/fedavg.py` | Sample-count-weighted parameter averaging; FedAvg server factory |
| `ml/federated/fedprox.py` | Proximal term `(mu/2) * sum ||w - w_global||^2` over trainable parameters; FedProx server factory |
| `ml/federated/experiment.py` | End-to-end comparison: centralized, local-only, FedAvg, FedProx |

Shared with the centralized pipeline: `ml/data.py` (split), `ml/preprocessing.py` (scaler), `ml/models.py` (`DiabetesMLP`), `ml/training.py` (`train_one_epoch`), `ml/evaluation.py` (metrics, thresholds).

## Data flow and leakage control

1. **Global split.** The stratified split is the same as in `configs/centralized.yaml` (15% test, 15% validation, seed 42), so all phases share one test set.
2. **Clients.** Each client receives a disjoint part of the global *training* rows and a disjoint part of the global *validation* rows. The union of client training rows is exactly the centralized training set, so centralized versus federated differs only in *where* training happens.
3. **The test set is never partitioned.** It is passed only to the simulation's `GlobalEvaluator`. An assertion in `experiment.py` and the tests in `tests/test_partition.py` check that no test index reaches a client.
4. **Federated preprocessing.** Each client computes the count, sum and sum of squares of its scaled features, and the server combines them into the scaler (`FeaturePreprocessor.from_sufficient_statistics`). The result is identical to fitting on the pooled training rows (unit-tested). The class weight `pos_weight = n_negative / n_positive` is built from per-client label counts. Both are aggregate statistics that leave the clients. This is a disclosure, which is documented here.
5. **Thresholds.** Thresholds for the global model are chosen on the union of the clients' validation rows. In a deployment, each client would report validation counts instead. They are then frozen before the test set is scored.
6. **No test-based selection.** The reported federated model is the model after the final round. Test metrics are logged every round only to draw convergence curves.

## Partitioning

- **IID.** The training rows of each class are split equally across the K clients, so every client has about 1/K of each class.
- **Dirichlet label skew** (Hsu, Qi & Brown, 2019):
  - For each class c, the clients' shares are drawn as `q_c ~ Dir(alpha * 1_K)`, and class-c rows are split accordingly.
  - The same `q` is applied to the validation rows, so each client's validation data matches its own distribution.
  - Smaller `alpha` gives more skew. Because class 0 is 86% of the data, label skew also creates very unequal client sizes.
  - Draws in which any client has fewer than `min_train_per_class` training rows or `min_val_per_class` validation rows of some class are rejected and redrawn. The number of draws is reported, because rejection slightly favours less extreme draws.
- Partitions are fixed by `partition_seed`, independently of the training seeds.

## Algorithms

All federated methods share the same `model_fn`, the same initial global weights for a given seed, the same clients, and the same local SGD settings.

- **FedAvg** (McMahan et al., 2017). Each round:
  1. The server samples `round(client_fraction * K)` clients.
  2. It sends each sampled client the global parameters.
  3. Each client runs `local_epochs` epochs of minibatch SGD on its class-weighted BCE loss.
  4. The server sets the new global parameters to `w = sum_k (n_k / sum_j n_j) * w_k`, where `n_k` is client k's number of training rows.
- **FedProx** (Li et al., 2020). Identical to FedAvg, except that each client minimizes `BCE + (mu/2) * sum_p ||w_p - w_global_p||^2` over its trainable parameters, where `w_global` is a frozen copy of the parameters it received. With `mu = 0`, FedProx is bit-for-bit identical to FedAvg (tested).
- **Local-only.** Each client trains the same MLP architecture alone, with no communication. It uses its own scaler, its own class weight, Adam with early stopping on its own validation rows, and a threshold chosen on its own validation rows. It is evaluated on the global test set.
- **Centralized.** The same MLP, trained on the pooled training rows with the centralized recipe (Adam with early stopping). This is the non-private reference point.

## Measurements

- **Classification metrics on the global test set**, for both classes: accuracy, precision, recall, F1, ROC-AUC, PR-AUC and confusion counts. They are reported at three thresholds: 0.5, the F1-maximizing threshold on validation, and the threshold reaching 80% recall on validation.
- **Convergence.** The global model's validation and test loss and AUC are recorded after every round.
- **Communication.** Raw float32 payload bytes, counting both the download of the global model and the upload of each sampled client's model. Serialization and protocol overhead are excluded. Bytes and rounds are also reported up to the first round whose validation ROC-AUC comes within `target_tolerance` of the centralized MLP.
- **Training time.** Wall-clock time of the sequential single-process CPU simulation. The time of the slowest client per round is also logged, as an estimate of parallel execution time.

## Reproducing

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_fedavg.py tests/test_fedprox.py tests/test_partition.py tests/test_federated_loop.py
.\.venv\Scripts\python.exe -m ml.federated.experiment --config configs/federated_smoke.yaml
.\.venv\Scripts\python.exe -m ml.federated.experiment --config configs/federated.yaml
```

Every setting (rounds, local epochs, learning rate, batch size, client fraction, FedProx `mu`, number of clients, partitions, seeds) is in the YAML config. Use a new `output_tag` to keep runs separate.

## Limitations

- Clients run sequentially in one process. Network effects (latency, dropped clients, stragglers) are not simulated.
- With 5 clients and full participation, the client sampling between rounds is trivial. Set `client_fraction < 1` to study partial participation.
- Sharing the scaler statistics and label counts discloses aggregate information about each client.
- Only label skew is simulated. Feature skew (for example, partitioning by age band) is a natural extension.
