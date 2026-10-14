# Phase 0 Plan: Privacy-Preserving Federated Machine Learning for Diabetes Classification under Heterogeneous Health Data

Status: **planning only, nothing implemented yet**. Every number in this document comes from inspecting the real CSV files on 2026-10-14. No experimental results exist yet.

---

## 1. Findings from data inspection

### 1.1 Location

- `d:\Federated-Healthcare-ML` is **empty**. It has no `data/raw/` folder and is not a git repository.
- The CSVs are in `C:\Users\harsh\Downloads\archive (2)\`. They need to be copied into `data/raw/` (Phase 1, with your permission).

### 1.2 Primary dataset: `diabetes_binary_health_indicators_BRFSS2015.csv`

| Property | Value |
|---|---|
| Shape | 253,680 rows x 22 columns (target + 21 features) |
| Size | 22.7 MB |
| Missing values | 0 |
| Data type | All `float64`, but **every value is an integer code** |
| Target `Diabetes_binary` | 0: 218,334 (86.07%), 1: 35,346 (13.93%), so about a **6.2 : 1 imbalance** |
| Exact duplicate rows | 24,206 (9.5%) |
| Rows with identical features but different labels | 1,566 feature profiles covering 5,218 rows (corrected by `ml/data_validation.py`) |

Feature types:

- **Binary (14):** HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex
- **Ordinal (4):** GenHlth (1-5), Age (13 bands, 1-13), Education (1-6), Income (1-8)
- **Count/continuous (3):** BMI (12-98, 84 unique values), MentHlth (0-30), PhysHlth (0-30)

### 1.3 Secondary datasets

| File | Shape | Finding | Proposed use |
|---|---|---|---|
| `diabetes_012_health_indicators_BRFSS2015.csv` | 253,680 x 22 | **The same respondents in the same row order as the primary file.** Features are identical row by row, and `Diabetes_binary == (Diabetes_012 == 2)`. Prediabetes (4,631 rows) is merged into class 0 in the binary file. | Optional 3-class extension. **Must never be combined with the primary file**, because that would duplicate every respondent. |
| `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` | 70,692 x 22 | Balanced 35,346 / 35,346. The positive count exactly matches the primary file, so it is almost certainly a subsample of it. | Only a separate sensitivity analysis (balanced training versus class weighting). **Must never be mixed into train or test sets built from the primary file**, because that would leak test rows. |
| `online_retail.csv` | about 541,909 rows x 8 columns (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country) | E-commerce transactions with no health content. | **Exclude** from the diabetes research. It is unrelated, and mixing domains would weaken the study. |

### 1.4 Environment

| Item | Found | Implication |
|---|---|---|
| Python | 3.11.8 | Good. Supported by torch, JAX, Flower, Ray, and Opacus. |
| torch | 2.10.0, **CPU-only build** (`cuda False`) | The CPU path works now. CUDA is optional (see Section 8). |
| GPU | RTX 3050 Laptop, 4 GB, driver 610.62 | Enough for these small models, but the speedup will be minor. |
| Already installed | scikit-learn 1.8.0, pandas 3.0.1, numpy 2.4.3, shap 0.51.0, fastapi 0.135.1, uvicorn 0.42.0, pydantic 2.12.5, pytest 9.1.1, matplotlib 3.10.8 | These are very recent versions, so compatibility must be pinned and checked. |
| Not installed | jax, flax, optax, flwr, ray, opacus | Install them as optional extras. |
| Node | v22.14.0 | Good for Vite + React + TypeScript. |

---

## 2. Assumptions (please confirm or correct)

0. **Target definition (verified in Phase 1).** `Diabetes_binary = 1` means diabetes only. Prediabetes is coded 0, contrary to the Kaggle and UCI documentation. Labels are kept as-is. See `data/README.md`, Section 3.
1. **Unit of analysis.** Each row is one anonymous BRFSS 2015 survey respondent. The labels are *self-reported* diagnoses, not clinical ground truth. The model is a research artefact, not a diagnostic tool.
2. **Simulated clients.** Clients are created by partitioning one national survey. Documentation will call them "simulated clients" or "simulated sites", **never hospitals**. Demographic partitions (for example by Age band or Income) imitate heterogeneous populations but are not real institutions.
3. **Duplicates are kept by default.** With 21 coarse integer codes, different people can share an identical profile, so duplicates are plausible and not necessarily data-entry errors. A sensitivity run with duplicates removed will be reported. The 2,425 conflicting-label rows show that the features cannot fully determine the label (irreducible error).
4. **One global held-out test set.** A stratified 20% of the primary data is held out **before** client partitioning. All methods are evaluated on this exact set. Each client also holds a local validation split for early stopping and local metrics.
5. **Imbalance handling.** The default is a class-weighted loss (`pos_weight`) with the decision threshold tuned on validation data only. PR-AUC is reported alongside ROC-AUC, because accuracy is misleading at 86/14.
6. **Preprocessing in the federated setting.** Binary columns pass through unchanged. Ordinal and count columns are standardized. For the federated runs, the scaler's mean and variance are computed from **aggregated per-client sufficient statistics** (sums, sums of squares, counts), so the server never sees raw rows. The centralized baseline fits the scaler on its own training set.
7. **Privacy claims.** Federated learning only keeps raw data local. Model updates can still leak information (membership inference, gradient inversion). Formal guarantees are claimed **only** for runs with differential privacy, and only with the epsilon and delta reported by the accountant.
8. **CPU first.** Every required experiment must finish on a laptop CPU at a reasonable scale. GPU and Lightning AI are optional accelerators.
9. **Seeds.** The default seed list is {0, 1, 2, 3, 4}. Results are reported as mean ± standard deviation, with bootstrap 95% confidence intervals on the test set for the headline numbers.
10. **Data governance.** `data/raw/`, `data/processed/`, and any row-level outputs are git-ignored. Only aggregate metrics, figures, and SHAP summaries go into `results/`. The CDC data is public and de-identified, but your rule (never upload raw data) is followed anyway.

### 2.1 Decisions confirmed (2026-10-14)

- Copy the three diabetes CSVs into `data/raw/` (git-ignored, with SHA-256 checksums recorded).
- Exclude `online_retail.csv` from the project.
- Keep duplicate rows by default, and report a sensitivity run with duplicates removed.
- Initialize git with a `.gitignore` that protects `data/`.

---

## 3. Architecture

```
                    ┌──────────────────────────── configs/*.yaml (pydantic-validated)
                    ▼
  data/raw ──► fedhealth.data ──► global test split ──► partitioners (IID / Dirichlet / feature / quantity)
                (load, schema,        │                         │
                 checksum)            │                         ▼
                                      │                 ClientDataset x K (K ≥ 5)
                                      │                         │
           ┌──────────────────────────┼─────────────────────────┼─────────────────────────┐
           ▼                          ▼                         ▼                         ▼
   Baselines (sklearn)      Centralized (PyTorch)      Local-only (PyTorch)     Federated (PyTorch)
   LogReg, HistGB                                                               FedAvg / FedProx
                                                                                ├─ backend: sequential (default, CPU)
                                                                                ├─ backend: Ray actors (optional)
                                                                                ├─ backend: Flower simulation (cross-check)
                                                                                └─ DP: Opacus DP-SGD + RDP accountant (optional)
           │                          │                         │                         │
           └──────────────► fedhealth.evaluation (metrics, CIs, confusion, comm-cost, timing) ◄┘
                                      │
              JAX/Flax 1D-CNN ────────┤  (separate track, same splits)
              SHAP explainers ────────┤
                                      ▼
                     results/runs/<run_id>/  (config, env, metrics.csv, rounds.csv, …)
                                      │
                     ┌────────────────┴──────────────┐
                     ▼                               ▼
             reports/ (figures, tables,      FastAPI (read-only API over results/)
             final report generated from               ▼
             CSVs, not hand-typed)           React + TypeScript dashboard
```

### 3.1 Key design decisions

**Federated learning: explicit implementation plus a maintained framework.**
- FedAvg and FedProx are implemented **from scratch** in `fedhealth/federated/` as a framework-independent server, client, and aggregation layer:
  - FedAvg uses a weighted average of client parameters, with weights proportional to client sample counts.
  - FedProx adds the proximal term `(mu/2)·||w − w_global||²` to each client's local loss.
- **Flower (`flwr`)** is the maintained framework. A thin adapter wraps the same client logic in Flower's `NumPyClient`, and Flower's built-in `FedAvg` and `FedProx` strategies run as an **equivalence cross-check** in the tests. This satisfies "use a maintained framework" while keeping the algorithms visible and testable.
- Flower's simulation engine runs on Ray, which overlaps with the optional Ray requirement.

**Client execution backends.**
- `sequential`: the default. Deterministic and CPU-friendly.
- `ray`: optional. One Ray actor per client, fully parallel. Results must match `sequential` within floating-point tolerance for the same seed.

**Comparison regimes.**

| Regime | Training data | Evaluation |
|---|---|---|
| Centralized | Union of all client training sets | Global test set |
| Local-only | Each client's own data, no communication | Global test set plus the client's own validation set; report mean, worst client, and spread |
| FedAvg | Federated, mu = 0 | Global test set plus per-client validation |
| FedProx | Federated, mu ∈ {0.001, 0.01, 0.1} | Same as FedAvg |

**Heterogeneity settings** (K = 5 default, plus K = 10 as a scaling check):

| Name | Method | What it simulates |
|---|---|---|
| `iid` | Stratified random split | Control |
| `dirichlet_label` | Label proportions per client drawn from Dirichlet(α), α ∈ {0.1, 0.5, 1.0} | Prevalence differences |
| `feature_age` | Clients defined by Age-band groups | Different patient demographics |
| `feature_income` | Clients defined by Income groups | Socioeconomic heterogeneity |
| `quantity` | Client sizes drawn from a power law | Small versus large sites |

Every partition saves its per-client size and positive rate to a CSV, and a heterogeneity figure is generated for it.

**Models.**
- PyTorch MLP: 21 → 64 → 32 → 1, with ReLU, dropout, and BCEWithLogits loss. About 3.5k parameters, so it is cheap on CPU and gives small communication payloads.
- Logistic regression in PyTorch: the simplest federated model, used for sanity checks.
- scikit-learn baselines: LogisticRegression and HistGradientBoostingClassifier. These are centralized only and serve as reference points.

**Differential privacy (optional).** Opacus is used with an accountant (RDP, plus PRV as a cross-check):
- **Mode A, record-level DP:** DP-SGD inside each client, with per-sample clipping, Gaussian noise, and accounting of each client's epsilon.
- **Mode B, client-level DP-FedAvg:** the server clips client update norms and adds Gaussian noise. Epsilon is accounted with the subsampled-Gaussian RDP accountant, where the sampling rate is the client-sampling fraction. With only 5 to 10 clients, client-level DP gives a **very poor** privacy-utility trade-off. This will be reported honestly rather than tuned away.
- Experiments sweep the noise multiplier and record (epsilon, delta = 1/n) against ROC-AUC. Hyperparameters are tuned without touching the test set, and the privacy cost of tuning is acknowledged as a limitation.

**JAX 1D CNN.** Flax (NNX API) and Optax treat the 21 features as a length-21, single-channel sequence: Conv1D → Conv1D → global pool → Dense. It runs centralized and with FedAvg on the same splits. Planned control experiments:
1. **Feature-permutation sensitivity:** run with 5 random column orders. Convolutions assume neighbouring columns are related. Tabular columns have no natural order, so if results change with the ordering, the architecture is mismatched to the data.
2. **JAX MLP:** separates the effect of the framework (JAX versus PyTorch) from the effect of the architecture (CNN versus MLP).

Documented limitations:
- There is no spatial or temporal structure in tabular data.
- The CNN has more inductive bias than the data justifies.
- Native Windows JAX is CPU-only; CUDA JAX needs Linux, WSL2, or Lightning AI.
- Random number generation differs between frameworks, so results are not bit-identical across them.

**Explainability (SHAP).**
- `TreeExplainer` for HistGB.
- `DeepExplainer` or `GradientExplainer` for the PyTorch models, with `KernelExplainer` on a small subsample as a fallback.
- Compared models: centralized, FedAvg (IID), and FedAvg (most non-IID). Outputs are beeswarm plots, mean |SHAP| bar plots, and rank correlation (Kendall τ) of feature importance between models.
- Limitation: computing SHAP needs a background dataset. A real federated server would not hold one, so in this simulation SHAP uses held-out data. This is stated explicitly. A federated alternative (clients compute local SHAP, the server averages them) is a stretch goal.

**Metrics and costs.**
- **Classification:** accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrices (counts and normalized), all at the validation-tuned threshold and at 0.5.
- **Convergence:** per-round global test loss and AUC curves, plus rounds needed to reach 95% of the centralized AUC.
- **Runtime:** wall-clock time per round and in total, using `time.perf_counter`. Hardware and library versions are logged.
- **Communication cost:** *measured*. The serialized size in bytes of every upload and download is logged per round, cumulated, and cross-checked against `num_params × 4 bytes × clients × 2`.

**Reproducibility.**
- Seeds are set for Python, NumPy, torch, and JAX PRNG keys. The torch deterministic algorithms flag is on, and DataLoader generators are seeded.
- Each run directory stores its resolved config, a config hash, the git commit, package versions, hardware details, and the data file's SHA-256.
- `run_id = <experiment>_<partition>_<algo>_seed<k>_<confighash8>`.
- The final report's tables and figures are generated **only from CSVs** by a script, so no number is ever typed by hand.

**Dashboard.**
- **FastAPI** (read-only):
  - `GET /api/runs`
  - `GET /api/runs/{id}` (config and summary)
  - `GET /api/runs/{id}/rounds`
  - `GET /api/runs/{id}/confusion`
  - `GET /api/compare?ids=…`
  - `GET /api/shap/{id}`
  - `GET /api/privacy` (epsilon versus AUC)
  - `GET /api/health`
  - It serves only aggregate artefacts and never row-level data. CORS is limited to localhost.
  - Launching runs from the UI is an optional stretch goal, limited to a whitelist of small presets.
- **React + TypeScript** (Vite, TanStack Query, Recharts, and types generated from FastAPI's OpenAPI schema). Pages:
  1. Overview
  2. Compare runs (metrics table and bar charts)
  3. Convergence curves
  4. Confusion matrices
  5. Client heterogeneity
  6. Privacy-utility trade-off
  7. SHAP

---

## 4. Dependency plan

A **dedicated virtual environment** (`.venv`, Python 3.11) is used so that the very new global packages (pandas 3, numpy 2.4) cannot break things unexpectedly. Dependencies are declared in `pyproject.toml` with optional extras, and a lock file of exact versions is exported after the first green test run.

| Group | Packages | Notes |
|---|---|---|
| core | torch, numpy, pandas, scikit-learn, pyyaml, pydantic, matplotlib, seaborn, tqdm | The CPU wheel of torch by default |
| fl | flwr[simulation] | Pulls in Ray. Used for the cross-check. |
| ray | ray[default] | Optional backend. Ray on native Windows works but is less mature than on Linux, so this is verified in Phase 4 before relying on it. |
| dp | opacus | **Risk:** Opacus may lag torch 2.10. If it is incompatible, pin to the newest torch version Opacus supports, in the dp environment only. |
| jax | jax, flax, optax | CPU on Windows. On Lightning AI or Linux use `jax[cuda12]`. |
| xai | shap | 0.51 is already installed. Check numpy 2.x compatibility. |
| api | fastapi, uvicorn | |
| dev | pytest, pytest-cov, ruff, mypy, hypothesis (property tests for the partitioners) | |
| frontend | vite, react, typescript, @tanstack/react-query, recharts, openapi-typescript, vitest | Node 22 |

Compatibility is verified at the start of each phase that introduces a new package, and the outcome is recorded in `docs/ENVIRONMENT.md`.

---

## 5. Directory structure

```
Federated-Healthcare-ML/
├── README.md
├── pyproject.toml
├── .gitignore                    # data/raw, data/processed, .venv, node_modules, large artefacts
├── configs/
│   ├── base.yaml
│   ├── data/{iid,dirichlet,feature_age,feature_income,quantity}.yaml
│   ├── algo/{centralized,local_only,fedavg,fedprox}.yaml
│   ├── dp/{off,dpsgd_client,dp_fedavg}.yaml
│   └── experiments/{smoke,main_grid,dp_sweep,jax_cnn,shap}.yaml
├── data/
│   ├── README.md                 # provenance (CDC BRFSS 2015, Kaggle cleaned version), licence, schema
│   ├── raw/                      # git-ignored; plus MANIFEST.sha256 (committed)
│   └── processed/                # git-ignored; cached splits/partitions
├── src/fedhealth/
│   ├── config.py                 # pydantic schemas + YAML loader
│   ├── seed.py                   # reproducibility utilities
│   ├── data/{load.py, schema.py, preprocess.py, split.py, partition.py}
│   ├── models/{mlp.py, logreg.py}
│   ├── training/{trainer.py, centralized.py, local_only.py}
│   ├── federated/{client.py, server.py, aggregation.py, fedprox.py, comm.py,
│   │              backends/{sequential.py, ray_backend.py}, flower_adapter.py}
│   ├── privacy/{dpsgd.py, dp_fedavg.py, accountant.py}
│   ├── jax_models/{cnn1d.py, train.py, federated.py}
│   ├── explain/shap_analysis.py
│   ├── evaluation/{metrics.py, bootstrap.py, timing.py}
│   ├── tracking/{run.py, environment.py}
│   └── viz/{plots.py}
├── scripts/
│   ├── prepare_data.py           # copy check, checksum, schema validation, EDA summary
│   ├── run_experiment.py         # one config → one run
│   ├── run_grid.py               # multi-seed grid
│   ├── aggregate_results.py      # runs → summary CSVs
│   ├── make_figures.py
│   └── build_report.py           # CSVs → report tables/figures
├── tests/
│   ├── test_data_*.py  test_partition.py  test_aggregation.py  test_fedprox.py
│   ├── test_flower_equivalence.py  test_ray_equivalence.py  test_comm_cost.py
│   ├── test_dp_accountant.py  test_metrics.py  test_reproducibility.py
│   ├── test_jax_cnn.py  test_api.py
├── api/                          # FastAPI app (app.py, routers/, schemas.py)
├── dashboard/                    # Vite + React + TS
├── results/                      # git-ignored except summary CSVs
│   ├── runs/<run_id>/{config.yaml, env.json, metrics.csv, rounds.csv, clients.csv, confusion.csv, comm.csv}
│   └── summary/*.csv
├── reports/
│   ├── figures/
│   ├── tables/
│   └── final_report.md (→ PDF via pandoc, optional)
└── docs/
    ├── PLAN.md  ARCHITECTURE.md  ENVIRONMENT.md  REPRODUCIBILITY.md
    ├── LIGHTNING_AI.md  PRIVACY_AND_LIMITATIONS.md  DATA_CARD.md
```

---

## 6. Planned tests (correctness, not just smoke tests)

- **Partitioning:** client partitions are disjoint, cover every training row, and contain no global-test rows. They are deterministic for a given seed. Dirichlet runs with small α produce measurably more skew than large α.
- **Aggregation:**
  - Weighted FedAvg on hand-constructed tensors matches the analytic result.
  - One client with one local epoch matches centralized SGD on the same data and seed.
  - Unequal client sizes are weighted correctly.
- **FedProx:**
  - With mu = 0, FedProx matches FedAvg bit for bit.
  - The proximal gradient equals `mu·(w − w_global)`.
  - Larger mu reduces drift between local and global weights.
- **Flower cross-check:** given the same seeds, the explicit FedAvg and FedProx implementations match Flower's strategies within tolerance after N rounds.
- **Ray:** the Ray backend reproduces sequential results.
- **Communication cost:** measured bytes equal the analytic count.
- **DP:** the accountant's epsilon is monotonic in steps and noise, and matches Opacus reference values. Clipping bounds per-sample gradient norms.
- **Metrics:** match sklearn on fixed fixtures. The confusion-matrix sum equals the test set size.
- **Reproducibility:** two runs with the same config and seed produce identical metrics.
- **API:** endpoints return valid schemas, and no endpoint can serve files from `data/`.

---

## 7. Implementation roadmap

| Phase | Deliverables | Exit criterion |
|---|---|---|
| **1. Scaffolding and data** | git init, `.gitignore`, venv, `pyproject.toml`, copy CSVs into `data/raw` plus SHA-256 manifest, schema validation, EDA summary CSV and figures, data card | `pytest tests/test_data_*` passes, and no raw data is tracked by git |
| **2. Splits, preprocessing, partitioners** | Global test split, federated scaler statistics, all 5 partitioners, heterogeneity plots | Partition tests pass |
| **3. Centralized baselines** | sklearn LogReg and HistGB, PyTorch MLP, threshold tuning, full metrics, bootstrap CIs | Real baseline metrics saved to CSV |
| **4. Federated core** | Local-only, FedAvg, FedProx, sequential backend, communication and runtime logging, Flower cross-check, optional Ray backend | Aggregation, FedProx, equivalence, and communication-cost tests pass |
| **5. Experiment grid** | Multi-seed grid across regimes, partitions, and mu values; summary CSVs; convergence, metric, and confusion figures | `results/summary/*.csv` generated from real runs |
| **6. Differential privacy** | Record-level DP-SGD and client-level DP-FedAvg, accountant, epsilon-versus-utility sweep | Accountant tests pass, and the privacy-utility CSV and figure exist |
| **7. JAX 1D CNN** | Flax CNN plus JAX MLP control, centralized and FedAvg runs, permutation-sensitivity experiment, written limitations | JAX tests pass, and comparison CSV exists |
| **8. SHAP** | Explainers, plots, cross-model importance rank correlation | SHAP CSV and figures exist |
| **9. API and dashboard** | FastAPI read-only API, React + TypeScript dashboard, API tests | Dashboard renders real result files locally |
| **10. Cloud and report** | `LIGHTNING_AI.md` (GPU setup, `jax[cuda12]`, uploading data to a private Studio drive), final report built from CSVs, limitations and ethics section | Report builds end to end from `results/` |

Each phase ends with a stop for your review before the next one starts.

---

## 8. Risks and open points

- **GPU locally:** torch is the CPU build. A 3.5k-parameter MLP on about 200k rows trains fast on CPU, and GPU kernel-launch overhead may even make it slower. GPU matters mainly for DP-SGD sweeps, the CNN, and Ray parallelism, so Lightning AI is documented for those.
- **Library versions:** Opacus with torch 2.10, SHAP with numpy 2.4, and Ray on native Windows all need early verification, with fallbacks pinned in separate extras.
- **Overstated claims to avoid:** "hospitals", "FL guarantees privacy", "clinically validated", and DP guarantees without reported epsilon and delta.
- **Evaluation leakage:** thresholds and hyperparameters are tuned on validation data only, and the global test set is touched only for final reporting.
