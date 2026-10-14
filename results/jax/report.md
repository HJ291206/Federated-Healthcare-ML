# JAX 1D-CNN experiment

The 21 features are tabular survey answers with an arbitrary column order; a 1D convolution over them mixes neighbouring columns that have no medical relationship. The CNN is an experimental representation, not an image-style CNN with a physical meaning. Rows `permuted_seed*` retrain the CNN on shuffled column orders: if results change, the architecture depends on an arbitrary convention.

Generated on 2026-10-15 02:48:13 +0530 from `results/jax/jax_results.csv`. Seeds: [0]. JAX 0.10.2 on ['cpu:0'], PyTorch 2.10.0+cpu (4 threads). Timing is wall-clock on one CPU machine, run back to back; XLA and PyTorch thread usage is not equalized.

| Framework | Model | Feature order | Params | Test ROC-AUC | PR-AUC | Accuracy | Recall (1) | F1 (1) | Epochs | Train s | First-epoch s | s/epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pytorch | mlp | original | 3,521 | 0.8276 ± 0.0000 | 0.4169 | 0.8076 | 0.6075 | 0.4681 | 15.0 | 13.0 | 0.73 | 0.87 |
| jax | mlp | original | 3,521 | 0.8273 ± 0.0000 | 0.4170 | 0.8087 | 0.6039 | 0.4681 | 15.0 | 11.4 | 2.49 | 0.76 |
| jax | cnn1d | original | 2,721 | 0.8171 ± 0.0000 | 0.3913 | 0.7924 | 0.6239 | 0.4558 | 15.0 | 23.2 | 2.61 | 1.55 |
| jax | cnn1d | permuted_seed1 | 2,721 | 0.8088 ± 0.0000 | 0.3795 | 0.7882 | 0.6060 | 0.4436 | 15.0 | 23.2 | 2.40 | 1.55 |

Operating point: validation max-F1 threshold; thresholds and early stopping use validation data only.
