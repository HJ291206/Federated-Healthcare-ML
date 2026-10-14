# Running on Lightning AI Cloud (optional)

Everything in this project runs on a laptop CPU without any cloud account; that is the default and the path used for all reported results. Lightning AI is useful for the slower variants (30-round / 3-seed federated grids, DP sweeps, larger CNNs) or for GPU runs.

## Credentials

- **Never put API keys or tokens in source code, configs, or results.** Nothing in this repository reads a credential.
- Authenticate with the Lightning CLI (`lightning login`) or let the Studio session handle it. If a token is required for a CI job, store it as a secret and expose it as the environment variable `LIGHTNING_API_KEY`; do not write it to disk inside the repository.

## One-time setup in a Studio

1. Create a Studio (CPU is fine; choose a GPU machine, e.g. T4/L4, only when you intend to use `--device cuda`).
2. Clone the repository and create the environment:
   ```bash
   git clone <your-private-remote> Federated-Healthcare-ML && cd Federated-Healthcare-ML
   python -m venv .venv && source .venv/bin/activate
   # CPU machine:
   pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
   # GPU machine (pick the CUDA build matching the Studio's driver, e.g. cu126):
   # pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   pip install -r requirements-optional.txt    # only if you need JAX / Ray / DP
   ```
3. **Data.** The raw CSVs are not in git. Upload `diabetes_binary_health_indicators_BRFSS2015.csv` (and, optionally, the two alternative files) to `data/raw/` through the Studio file browser or a private Studio drive. Then verify the copy is byte-identical to the one analysed here:
   ```bash
   python -m ml.data_validation      # sha256_matches_manifest must PASS
   ```
   Do not upload the data to any public bucket or public repository.

## Running experiments

The entry point is the same locally and in the cloud:

```bash
python -m ml.run centralized --device auto                       # GPU if present, else CPU
python -m ml.run federated --config configs/federated.yaml        # full 30-round, 3-seed grid
python -m ml.run federated --set federated.rounds=50 --set seeds=[0,1,2,3,4] --set output_tag=cloud50
python -m ml.run dp --set privacy.noise_multipliers=[0.5,1,2,4] --set output_tag=dp_sweep
python -m ml.run jax
python -m ml.run ray-benchmark --set ray_num_cpus=8
```

- `--device {cpu,cuda,auto}` applies to the PyTorch MLP (`ml/training.py::fit_mlp`). Federated client training runs on CPU: the model has 3,521 parameters and per-client batches are small, so GPU kernel launch overhead would dominate.
- `--set key=value` overrides any YAML key; the merged config is saved in the run's `run_metadata.json`, so cloud runs are reproducible from their outputs.
- Checkpoints: set `federated.checkpoint_every: k` (or `--set federated.checkpoint_every=5`) to save the global model every k rounds to `models/federated/<tag>/checkpoints/`. `FederatedServer.load_checkpoint()` restores weights, round counter, byte counters and history.
- JAX on GPU: install `jax[cuda12]` instead of `jax` (Linux only). On Windows JAX is CPU-only.

## Getting results back

Results are CSV/PNG/MD files under `results/`; models under `models/` (git-ignored). Download `results/` from the Studio, or commit only `results/` (never `data/` or `models/`) to your private remote.

## Cost and fairness notes

- Timing numbers from a cloud machine are not comparable to the laptop numbers in `results/`; report hardware with every timing (the scripts record platform, CPU count and torch/JAX versions in `run_metadata.json`).
- Lightning AI's free tier has limited monthly compute; stop the Studio when idle.
