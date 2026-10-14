# Reproducibility checklist

Tick marks reflect what is implemented and was exercised on the development machine; items marked "not run" are honest gaps.

## Data
- [x] Raw files identified by SHA-256 (`data/raw/MANIFEST.sha256`); `python -m ml.data_validation` fails if a file changes.
- [x] Raw CSVs never modified, never committed (`.gitignore`), never uploaded.
- [x] Target definition verified against the 3-class file and documented (`data/README.md`); labels unchanged.
- [x] Duplicate handling documented (kept; counts in `results/dataset_summary.csv`).
- [x] Source, citation and license recorded (Kaggle CC0 verified; UCI terms / CDC codebook not independently verified, stated as such).

## Splits, seeds and configs
- [x] One stratified 70/15/15 split with `split_seed=42` shared by all experiments; indices saved (`models/centralized/split_indices.npz`).
- [x] Every experiment is driven by a YAML in `configs/`; the merged config is written to each run's `run_metadata.json`.
- [x] Seeds: `set_global_seed` (Python, NumPy, torch, deterministic algorithms); per-round client seeds derived from `(seed, round, client)` via `SeedSequence`.
- [x] Partition draws recorded (`client_partitions.csv`, `dirichlet_draws`), partition seed in config.
- [x] Determinism tests: `tests/test_training.py::test_train_one_epoch_is_deterministic`, `tests/test_federated_loop.py::test_training_is_deterministic`, `tests/test_partition.py::test_partition_is_reproducible`, `tests/test_split.py::test_split_is_reproducible`, `tests/test_privacy.py::test_noise_is_added_and_reproducible`, `tests/test_jax_cnn.py::test_fit_jax_runs_and_is_deterministic`.
- [x] Checkpoint/resume reproduces an uninterrupted run (`tests/test_federated_loop.py::test_checkpoint_resume_reproduces_uninterrupted_run`).

## Environment
- [x] Pinned versions in `requirements.txt`, `requirements-optional.txt`, `requirements-api.txt`; `pyproject.toml` requires Python 3.11.
- [x] CPU-only execution verified for every experiment; GPU code path exists but was **not run** (CPU torch build).
- [x] `run_metadata.json` records platform, CPU count and library versions for timing context.
- [ ] Docker image: not provided.

## Leakage controls
- [x] Scaler fitted on training rows only (`tests/test_preprocessing.py::test_scaler_statistics_come_from_training_rows_only`).
- [x] Thresholds, early stopping and hyperparameters chosen on validation only; test scored once (`ml/evaluation.py`, `ml/train_centralized.py`).
- [x] Test split never partitioned to clients (assertion in `ml/federated/experiment.py`; `tests/test_partition.py::test_partitions_are_disjoint_complete_and_exclude_test`; `tests/test_e2e_smoke.py`).
- [x] Per-round test metrics are logged for reporting only; no decision uses them.

## Results provenance
- [x] Every table and figure is generated from saved CSVs (`ml/final_evaluation/report.py`); `docs/RESULTS.md` names its source files.
- [x] Seed variability (3 seeds) and bootstrap CIs (500 resamples) reported; single-seed rows are labelled `n_seeds = 1`.
- [x] Known confound (training budget of the 8-round federated runs vs early-stopped centralized runs) stated in the report.
- [ ] 30-round / 3-seed federated grid (`configs/federated.yaml`): **not run**.
- [ ] Lightning AI Cloud execution: **not run** (instructions only).

## Tests
- [x] Unit tests for aggregation, proximal term, partitioning, preprocessing, metrics, DP mechanics, JAX, Ray, API.
- [x] End-to-end smoke test on the real CSV through the public entry point (`tests/test_e2e_smoke.py`).
- [x] Frontend tests (vitest) and type-check; production build.
- [x] Last full run: see README §9 for the exact counts and date.

## Exact reproduction from a clean machine

```powershell
git clone <your-private-remote> Federated-Healthcare-ML; cd Federated-Healthcare-ML
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -r requirements-optional.txt -r requirements-api.txt
# place the three Kaggle CSVs in data/raw/, then:
python -m ml.data_validation
python -m ml.exploratory_analysis
python -m ml.run centralized
python -m ml.run federated --set "seeds=[0, 1, 2]" --set output_tag=final
python -m ml.run dp
python -m ml.run jax
python -m ml.run ray-benchmark
python -m ml.final_evaluation.report
python -m pytest -q
cd dashboard; npm install; npm test; npm run build
```

Expected wall-clock on a 12-thread laptop CPU: about 25 minutes for all experiments plus 1–2 minutes of tests. Metrics should match `results/final/final_comparison.csv` to ~1e-3 (floating-point reduction order can differ across CPUs and thread counts; see the Ray report for an example).
