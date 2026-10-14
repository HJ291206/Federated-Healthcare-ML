# Optional differential-privacy experiment

Measured results: `results/federated/dp/report.md` (generated). This file explains what is claimed.

## Threat model

- **Participants.** One aggregation server and 5 simulated clients. The clients are partitions of a public survey (CDC BRFSS 2015), not hospitals.
- **Adversary.** Honest-but-curious: the server, or any party that sees a client's model updates, follows the protocol but tries to learn about individual training rows from those updates. Clients do not see each other's data. Secure aggregation and malicious (poisoning) clients are out of scope.
- **What is protected.** Whether any single training row of a client was used, inferred from that client's released model parameters.

## Data locality is not privacy

Ordinary FedAvg and FedProx keep raw rows on the client, but the parameters a client sends are a deterministic function of its data. Membership inference and gradient inversion attacks can recover information about training rows from such updates. **Plain FedAvg/FedProx carry no formal privacy guarantee** and are labelled *non-private* in all outputs.

Differential privacy adds a quantitative guarantee: the distribution of a client's released updates changes by at most a factor e^ε (plus δ) when one training row is added or removed. This holds regardless of what the adversary knows or computes afterwards (post-processing), so the server's aggregation does not weaken it.

## Privacy unit and mechanism

- **Privacy unit: one training example (one survey respondent's row) of one client.** This is **example-level DP**. It is **not client-level DP**: the guarantee does not hide whether an entire client participated, and no claim of that kind is made.
- **Mechanism: DP-SGD (Abadi et al., 2016) via Opacus 1.6.0** in `ml/federated/privacy.py`, applied inside each client's local training:
  - Poisson-sampled minibatches with rate q = batch_size / n_k.
  - Per-example gradient clipping to L2 norm C = `max_grad_norm`.
  - Gaussian noise with standard deviation σ·C added to the summed clipped gradient.
  - RDP accountant, one per client, persisting across communication rounds, so ε_k accumulates over all local steps.
  - Reported quantities: ε per client, δ, σ, C, sample rate, number of steps (`dp_privacy_accounting.csv`). The headline ε is the maximum over clients.
- **FedProx under DP.** The proximal term (μ/2)‖w − w_global‖² does not depend on data, so its exact gradient μ(w − w_global) is added *after* clipping and noising. This costs no privacy budget and is unit-tested (`tests/test_privacy.py::test_proximal_gradient_is_applied_under_dp`).

## Verification (tests/test_privacy.py)

- Per-example clipping matches a manual per-example gradient computation.
- A tiny clipping norm bounds the parameter update.
- Noise changes the update, is reproducible for a fixed seed, and differs across seeds.
- The accountant counts steps, ε grows with more steps and shrinks with larger σ.
- DP-FedProx with no noise and no effective clipping equals plain FedProx on a full batch.
- Opaque loss-based regularizers are rejected by the DP client (they would be silently dropped by Opacus).

## Assumptions and limitations

- Hyperparameters were chosen on validation data (the DP learning rate 2.0 was picked from {0.1, 0.5, 2.0} by validation ROC-AUC); this tuning is not charged to ε, as is common but not strictly rigorous.
- `secure_mode=False`: the noise comes from PyTorch's default RNG, which is not cryptographically secure.
- Small clients spend budget fast: a client with 3,315 rows and batch 2048 has q ≈ 0.62, so its ε is much larger than that of a 72,375-row client at the same σ. Per-client batch sizes or client-specific σ would equalize this; they were not used here.
- Empty Poisson batches are counted as no-op steps.
- The scaler statistics and class counts shared for preprocessing are not covered by the DP guarantee (see `docs/federated_learning.md`).
- The non-private baseline uses a different learning rate (0.1) than the DP runs (2.0), because clipped gradients are much smaller; both were set on validation data.
