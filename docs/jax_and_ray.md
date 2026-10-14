# Optional extensions: JAX 1D-CNN and Ray execution backend

Both extensions are optional. The core project (`requirements.txt`) does not import JAX or Ray; their tests skip automatically when the packages are absent. Install with `pip install -r requirements-optional.txt` (versions were checked with `pip install --dry-run` to leave torch/numpy/scikit-learn/opacus unchanged).

## Part A: JAX 1D-CNN (`ml/jax_cnn/`)

**Why a CNN on tabular data is questionable.** The 21 BRFSS features are survey answers in an arbitrary column order; a 1D convolution over them mixes neighbouring columns (e.g. `Stroke` next to `HeartDiseaseorAttack` next to `PhysActivity`) that have no spatial or sequential relationship. The CNN is therefore an experimental representation, not a model with a physical interpretation. To expose this, the experiment retrains the CNN on a shuffled column order (`permutation_seeds`); an MLP is exactly invariant to such a permutation, a CNN is not (`tests/test_jax_cnn.py`).

**Implementation.** Flax linen `CNN1D` (two `Conv` layers over the 21 positions, global average pooling, dense head), optax `adam` with decoupled weight decay, the same weighted BCE loss as the PyTorch MLP (`pos_weight = n_neg / n_pos`, unit-tested to match `torch.nn.BCEWithLogitsLoss`), the same 70/15/15 stratified split (`split_seed=42`), the same preprocessing, early stopping on validation ROC-AUC, thresholds chosen on validation only. A JAX `MLP` with the PyTorch architecture (3,521 parameters) is trained as a framework control.

**Measured results** (`results/jax/report.md`, seed 0, 15-epoch budget, CPU; values are from the run and will differ on other hardware):

| Framework | Model | Feature order | Test ROC-AUC | Train s (15 epochs) | First-epoch s |
|---|---|---|---|---|---|
| PyTorch | MLP | original | 0.8276 | 13.0 | 0.73 |
| JAX | MLP | original | 0.8273 | 11.4 | 2.49 |
| JAX | CNN1D | original | 0.8171 | 23.2 | 2.61 |
| JAX | CNN1D | permuted | 0.8088 | 23.2 | 2.40 |

Interpretation:

- The CNN does not beat the MLP (0.817 vs 0.828 test AUC) and loses a further 0.008 AUC when the columns are shuffled, confirming that its result depends on an arbitrary ordering convention.
- The JAX and PyTorch MLPs reach the same AUC (0.8273 vs 0.8276), as expected for the same architecture and loss.
- **Speed: no claim that JAX is faster.** Steady-state JAX epochs were somewhat shorter (0.76 vs 0.87 s/epoch) but JAX's first epoch includes XLA compilation (2.5 s vs 0.7 s), so total time differs by ~1.6 s over 15 epochs. The runs were back to back on one laptop with PyTorch at 4 threads and XLA's default thread pool; this is not a controlled benchmark and a 60-epoch run or a different machine could flip the ordering. The same per-epoch loop structure (NumPy shuffling, full-batch validation each epoch) is used in both.

Run: `python -m ml.run jax` (or `python -m ml.jax_cnn.experiment --config configs/jax_cnn.yaml`).

## Part B: Ray execution backend (`ml/federated/ray_backend.py`)

**What it does.** `RayFederatedServer` subclasses `FederatedServer` and overrides only `run_round`: each client becomes a `ClientActor` holding its own training tensors; the server `ray.put`s the global weights once per round and each actor runs `FederatedClient.local_train` with the identical per-round seed, returning a `ClientUpdate`. Aggregation (`fedavg_aggregate`), FedProx's proximal regularizer, client sampling, byte accounting and history are inherited unchanged. `init_ray` starts a local CPU-only Ray instance (no dashboard, no cluster) and `shutdown()` / the context manager kills actors and stops Ray only if this process started it.

**Semantics preserved.** `tests/test_ray_backend.py` checks that FedAvg and FedProx on Ray produce `torch.equal` parameters to the sequential server, identical byte counters and that actors are released. The benchmark re-checks this on the real data.

**Measured results** (`results/federated/ray/report.md`; 5 IID clients, 4 rounds, batch 64, laptop CPU):

| Algorithm | Backend | Torch threads/client | Seconds | Max |Δparam| vs Ray |
|---|---|---|---|---|
| FedAvg | sequential | 4 | 8.2 | 6.0e-03 |
| FedAvg | sequential_matched | 1 | 6.5 | 0 |
| FedAvg | Ray (5 actors) | 1 | 7.1 | – |
| FedProx μ=0.01 | sequential | 4 | 12.0 | 7.9e-03 |
| FedProx μ=0.01 | sequential_matched | 1 | 11.5 | 0 |
| FedProx μ=0.01 | Ray (5 actors) | 1 | 8.9 | – |

Observations:

- With the same torch thread count per client, Ray reproduces the sequential global model **exactly** for both algorithms. With 4 intra-op threads the sequential model differs by up to 8e-3 after ~2,200 SGD steps per client; that is floating-point reduction order, not an algorithmic difference (validation AUC agrees to 4 decimals).
- Speed depends on per-client work. For FedAvg (≈1.3 s of work per client per round) Ray was **not faster** than single-threaded sequential training (7.1 s vs 6.5 s): actor dispatch, object-store transfer and process scheduling on a laptop eat the parallelism gain. For FedProx, where the proximal term adds per-step work, Ray was 1.3× faster (8.9 s vs 11.5 s). Ray start-up (~3 s) is excluded from the per-run times. The gain grows with bigger models, more local epochs or more clients, and with multiple machines; measure before claiming a speed-up.

**Distributed execution is not federated privacy.** Ray decides *where* a computation runs; federated learning decides *what leaves* each client (model updates, never records). In this simulation all actors live on one machine and the raw tensors are created in the driver process before being handed to actors, so Ray gives no additional data-locality guarantee over the sequential loop. In a real deployment each client's actor would read its own data source and only `ClientUpdate` objects would cross the network; even then, model updates can leak information about training records, so FedAvg/FedProx on Ray does not by itself guarantee privacy (see `docs/differential_privacy.md`).

Run: `python -m ml.run ray-benchmark` (or `python -m ml.federated.ray_benchmark --config configs/ray_benchmark.yaml`).

## Part C: cloud execution

See `docs/LIGHTNING_AI.md`. The single entry point `python -m ml.run <experiment> [--config] [--set key=value] [--device cpu|cuda|auto]` is used locally and in the cloud; per-round federated checkpoints are enabled with `federated.checkpoint_every`.
