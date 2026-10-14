# Final research report: privacy-preserving federated learning for diabetes classification under heterogeneous health data

*Companion documents: [`RESULTS.md`](RESULTS.md) (all tables, generated from CSVs), [`../README.md`](../README.md) (diagrams, commands), [`REQUIREMENTS_MAP.md`](REQUIREMENTS_MAP.md), [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).*

> Clients are simulated partitions of one public survey dataset (CDC BRFSS 2015), not hospitals. Federated learning alone does not guarantee privacy. Feature importance is not causation. Model outputs are educational, not diagnoses. No number in this report is typed in by hand without a generating file being named.

## 1. Problem and research questions

Can a diabetes classifier be trained across several data holders without pooling their rows, and what does that cost in accuracy, communication and runtime? Specifically:

1. How do FedAvg and FedProx compare with centralized training and with each holder training alone (local-only)?
2. How does client heterogeneity (IID vs Dirichlet label skew) change the picture?
3. What is the utility cost of adding example-level differential privacy to local training?
4. Which features drive the models, and do the federated and centralized models agree?

## 2. Data

CDC BRFSS 2015 diabetes health indicators (Kaggle, Alex Teboul, CC0; UCI id 891, DOI 10.24432/C53919): 253,680 adult survey respondents, 21 self-reported indicators (blood pressure, cholesterol, BMI, smoking, stroke, heart disease, activity, diet, alcohol, health-care access, general/mental/physical health, mobility, sex, age group, education, income), target `Diabetes_binary` with 35,346 positives (13.93 %). Validation (`ml/data_validation.py`, `results/validation_checks.csv`): expected shape and columns, no missing values, documented code ranges, SHA-256 manifest. Two findings changed the documentation but not the data: (i) in this file prediabetes is coded 0 (verified by matching the 3-class file), so the task is *diagnosed diabetes vs everything else*; (ii) 24,206 exact duplicate rows exist and were kept, since coarse coding makes identical answer vectors plausible; 1,566 feature profiles carry both labels (5,218 rows), an irreducible label-noise floor. Full details: `data/README.md`.

## 3. Methodology

**Split and leakage control.** One stratified 70/15/15 split (`split_seed=42`) is shared by every experiment (`ml/data.py`; indices saved to `models/centralized/split_indices.npz`). All preprocessing statistics are fitted on the training split only (`ml/preprocessing.py`; test in `tests/test_preprocessing.py`). Hyperparameters, early stopping and decision thresholds use the validation split only; the test split is scored once per final model (`ml/evaluation.py::select_thresholds`, `ml/train_centralized.py`). In the federated setting the test split is never partitioned to clients; an assertion in `ml/federated/experiment.py` checks this and `tests/test_partition.py` tests it.

**Class imbalance.** Weighted losses rather than resampling: `pos_weight = n_neg/n_pos` (6.18) in BCE for the MLPs, `class_weight="balanced"` for logistic regression, `balanced_subsample` for the random forest. Three operating points are reported: default 0.5, validation max-F1 (used in the headline tables), validation recall ≥ 0.80.

**Centralized baselines** (`ml/train_centralized.py`, 3 seeds): logistic regression (C grid → 10, plateau), random forest (200 trees; depth/leaf grid → depth 20, min leaf 20, interior optimum), PyTorch MLP 21-64-32-1 with ReLU, dropout 0.1, Adam 1e-3, weight decay 1e-4, batch 512, early stopping on validation ROC-AUC (patience 6).

**Simulated clients and partitions** (`ml/federated/partition.py`). Five clients; `iid` = stratified equal shares; `dirichlet_a0.5` = per-class Dirichlet(α=0.5) shares (Hsu, Qi & Brown, 2019) applied to train and validation indices, with rejection sampling for minimum class counts (draw count recorded; 1 draw for both). Partitions are a deterministic function of `(partition_seed=42, num_clients, alpha)`.

**Federated preprocessing and loss.** Each client sends `(count, sum, sum of squares)` per feature and its label counts; the server fits the shared scaler from these sufficient statistics (equal to the pooled fit, `tests/test_preprocessing.py::test_sufficient_statistics_match_pooled_fit`) and the global `pos_weight`. These aggregates are the only non-model information exchanged.

**FedAvg** (McMahan et al., 2017; `ml/federated/fedavg.py`): \(w_{r+1} = \sum_k \frac{n_k}{\sum_j n_j} w_k\), computed in float64 (`tests/test_fedavg.py`, `test_federated_loop.py::test_single_round_equals_manual_weighted_average_of_local_training`). **FedProx** (Li et al., 2020; `ml/federated/fedprox.py`): each client minimises \(\ell_k(w) + \frac{\mu}{2}\|w - w_r\|^2\) with the anchor frozen at the broadcast weights; μ = 0 reproduces FedAvg bit-for-bit (`tests/test_fedprox.py`). Both start from the same seeded initialisation; local training is SGD (lr 0.1, batch 2048, 1 local epoch, full participation), 8 rounds. Communication = float32 payload down + up (3,521 parameters = 14,084 B per copy). **Local-only** baselines train the same MLP recipe as the centralized model on each client's own rows with its own scaler and class weight.

**Differential privacy** (`ml/federated/privacy.py`, Opacus 1.6.0): example-level DP-SGD inside each client (Poisson sampling, per-example clipping C = 1, Gaussian noise σ ∈ {1, 4}, RDP accountant persisting across rounds, δ = 1e-5, DP-specific lr 2.0 chosen on validation). ε is reported per client and as the maximum. The FedProx proximal gradient μ(w − w_r) is added after clipping and noising; it is data-independent and costs no privacy budget. This is **not** client-level DP; `secure_mode=False` (non-cryptographic RNG) is documented. Tests check clipping against a manual computation, noise reproducibility, accountant monotonicity and the proximal path (`tests/test_privacy.py`).

**Explainability** (`ml/final_evaluation/explain.py`, SHAP 0.51): Permutation explainer in probability units (100 training-row background, 400 random test rows) for logistic regression and the MLPs (centralized, IID FedAvg, Dirichlet FedAvg and FedProx); exact path-dependent TreeExplainer for the random forest; permutation importance (AUC drop) as a method-independent check; Spearman rank agreement between models; three illustrative single-prediction explanations per model.

**Statistics.** Three training seeds for centralized and federated runs (mean ± std); 500-resample bootstrap 95 % CIs on the test split for seed-0 models; one seed for DP (runtime).

**Optional modules.** JAX/Flax 1D-CNN over the 21 features (documented as non-spatial; column-permutation sensitivity run) with a JAX MLP control; Ray actor backend reproducing the sequential server exactly; Lightning AI Cloud instructions with a single reproducible entry point, `--device`, and per-round checkpoints.

## 4. Results (all from `results/final/*.csv`, `results/jax/`, `results/federated/ray/`; test split n = 38,052; CPU laptop)

### 4.1 Centralized vs local-only vs FedAvg vs FedProx (ROC-AUC, mean ± std over 3 seeds)

| Setting | Centralized MLP | Local-only (mean; min–max over clients) | FedAvg | FedProx μ=0.01 |
|---|---|---|---|---|
| IID | 0.8276 ± 0.0005 | 0.8230 ± 0.0002 (0.8223–0.8238) | 0.8176 ± 0.0009 | 0.8175 ± 0.0009 |
| Dirichlet α=0.5 | 0.8276 ± 0.0005 | 0.8136 ± 0.0023 (0.7837–0.8251) | 0.8181 ± 0.0009 | 0.8180 ± 0.0009 |

Random forest 0.8253 ± 0.0002 and logistic regression 0.8200 ± 0.0000 complete the centralized baselines. Bootstrap 95 % CIs (seed 0): centralized MLP [0.8230, 0.8336], IID FedAvg [0.8132, 0.8239], Dirichlet FedAvg [0.8135, 0.8245]. Threshold metrics at the validation max-F1 point are in `RESULTS.md` §2 (e.g., centralized MLP recall 0.606 / F1 0.468; IID FedAvg recall 0.627 / F1 0.459).

Reading: FedAvg trails the centralized MLP by 0.010 AUC in both settings, more than ten seed standard deviations. FedProx is indistinguishable from FedAvg (|Δ| ≤ 0.0001 < std), as expected with full participation, one local epoch and mild skew. Under IID, local-only clients (0.8230) *beat* the 8-round federated model because each client already holds ~35k homogeneous rows and the local models used Adam with early stopping (≈ 12 epochs) while the federated models used 8 rounds × 1 SGD epoch with validation curves still rising; this is a training-budget confound stated in `RESULTS.md`, not evidence that federation hurts. Under Dirichlet skew the weakest local client drops to 0.784 and the local-only mean falls below FedAvg (0.8136 vs 0.8181), which is the expected benefit of federation under heterogeneity.

### 4.2 Convergence, runtime and communication (`results/final/convergence_summary.csv`)

Validation ROC-AUC rises from ≈ 0.775 (round 1) to ≈ 0.823 (round 8) for every algorithm and partition; runs reach within 0.005 of their final AUC after 4–6 rounds and reach the centralized target (validation AUC − 0.01) in 6–8 rounds (two IID runs only at round 8). Eight rounds cost 2.6–3.1 s of training per run and 1,126,720 bytes of model traffic (8 × 5 clients × 2 directions × 14,084 B), versus 45 s for the centralized MLP, which instead moves the raw rows. The DP runs took 20–58 s because of per-example clipping.

### 4.3 Differential privacy (`results/federated/dp/`, 1 seed)

| Setting | σ | ε_max (δ = 1e-5) | FedAvg + DP AUC | Δ vs non-private FedAvg | FedProx + DP AUC |
|---|---|---|---|---|---|
| IID | 1 | 5.18 | 0.8173 | −0.0003 | 0.8157 |
| IID | 4 | 0.69 | 0.8172 | −0.0004 | 0.8156 |
| Dirichlet α=0.5 | 1 | 14.70 | 0.8165 | −0.0016 | 0.8133 |
| Dirichlet α=0.5 | 4 | 2.34 | 0.8162 | −0.0019 | 0.8133 |

With ~35k rows per client and 144 steps per client, strong example-level guarantees (ε < 1 under IID) cost under 0.002 AUC in this configuration; the skewed partition yields larger ε for the smaller clients (higher sampling rate) at equal σ. Single-seed results; the seed std of the non-private runs (0.0009) is the relevant scale.

### 4.4 Explainability (`results/final/shap_*.csv`)

All six explained models rank `GenHlth` first; `BMI`, `Age`, `HighBP` and `HighChol` fill ranks 2–5 in model-dependent order. Spearman rank agreement of mean |SHAP|: centralized MLP vs logistic regression 0.93, vs random forest 0.89, vs federated MLPs 0.75–0.78; FedAvg vs FedProx 0.999; IID vs Dirichlet FedAvg 0.987. SHAP vs permutation importance within a model: 0.74–0.95. An illustrative missed diabetic (centralized MLP, predicted 0.12 vs base 0.42) is driven down mainly by `Age = 1` (youngest group) and absent hypertension/cholesterol flags. These are descriptions of fitted models on survey data, not causal effects.

### 4.5 Optional modules

- **JAX 1D-CNN** (`results/jax/report.md`, 15-epoch budget, seed 0): CNN test AUC 0.8171, 0.8088 with shuffled columns; JAX MLP 0.8273 vs PyTorch MLP 0.8276. The CNN is worse than the MLP and sensitive to the arbitrary column order, as predicted for non-spatial features. Total training time: JAX MLP 11.4 s vs PyTorch 13.0 s, but JAX's first epoch (2.5 s, XLA compile) is 3× slower; not a controlled benchmark, so no claim that JAX is faster.
- **Ray backend** (`results/federated/ray/report.md`): with matched torch threads Ray reproduces the sequential global model exactly (max |Δparam| = 0) for FedAvg and FedProx; Ray was slower for FedAvg (7.1 s vs 6.5 s) and 1.3× faster for FedProx (8.9 s vs 11.5 s) on 5 clients × 4 rounds. Distributed execution is not a privacy mechanism.
- **Lightning AI Cloud**: documented workflow and entry point only; not executed in the cloud.

## 5. Limitations

- **Survey data.** Self-reported, cross-sectional, US adults; undiagnosed diabetes is labelled 0; prediabetes is coded 0; BMI and behaviours carry recall and social-desirability bias; sampling weights are unused, so the sample is not population-representative; the task is "diagnosed diabetes vs rest". Transfer to clinical populations or other countries is not established.
- **Simulated clients.** Random/Dirichlet partitions of one survey lack institution-specific protocols, coding practices, covariate shift and missingness, and share one preprocessing fit; real cross-silo heterogeneity is under-represented and the centralized–federated gaps here are optimistic.
- **Training budget.** The federated grid used 8 rounds for laptop runtime; curves were still rising. The 30-round / 3-seed configuration (`configs/federated.yaml`) was written but not run. Conclusions about the size of the federated gap therefore need a longer run.
- **Privacy.** Only the DP variant has a formal guarantee, and it is example-level with a non-cryptographic RNG; non-private FedAvg/FedProx offer none. ε values are reported for one seed.
- **Statistics.** Three seeds; bootstrap CIs capture test-set sampling only. Differences smaller than either source of variation are not rankings.
- **Explainability.** SHAP depends on the background sample and can be unstable for correlated inputs (`GenHlth`, `PhysHlth`, `DiffWalk`); it is not causal.
- **Timing.** Single-machine wall-clock, including Python overhead; the earlier "fast" run was slowed by stray processes and was superseded by the `final` run. Byte counts exclude compression and protocol overhead.

## 6. Conclusion

On BRFSS 2015, a 3,521-parameter MLP trained with FedAvg across five simulated clients reaches a test ROC-AUC of 0.818 after eight rounds and ~1.1 MB of model traffic, 0.010 below the centralized MLP (0.828) and within 0.006 of the best local model under IID; under Dirichlet label skew the global model (0.818) beats the average local model (0.814) and protects the weakest client (0.784). FedProx with μ = 0.01 gives no measurable change over FedAvg in this regime. Adding example-level differential privacy with σ = 4 (ε = 0.69 under IID, 2.34 under skew, δ = 1e-5) costs at most 0.002 AUC, at roughly 15–20× the training time. All models agree that self-rated general health, BMI, age, blood pressure and cholesterol dominate the predictions. The results support federated training as a viable alternative to pooling for this kind of tabular survey task, with the explicit caveats that the clients are simulated, the federated runs were short, and only the DP variant offers a formal privacy guarantee.

## References

- McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-efficient learning of deep networks from decentralized data. *AISTATS*.
- Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. *MLSys*.
- Hsu, T.-M. H., Qi, H., & Brown, M. (2019). Measuring the effects of non-identical data distribution for federated visual classification. arXiv:1909.06335.
- Abadi, M., et al. (2016). Deep learning with differential privacy. *ACM CCS*. Opacus: Yousefpour, A., et al. (2021). Opacus: User-friendly differential privacy library in PyTorch. arXiv:2109.12298.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Teboul, A. (2021). Diabetes Health Indicators Dataset (Kaggle, CC0). CDC (2015). Behavioral Risk Factor Surveillance System. UCI ML Repository id 891, DOI 10.24432/C53919.
