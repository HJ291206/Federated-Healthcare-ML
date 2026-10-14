# Sequential vs Ray execution

> Clients are simulated partitions of a single public survey dataset (CDC BRFSS 2015). They are not hospitals or real institutions, and federated learning alone does not guarantee privacy: model updates can still leak information about training data.

Ray changes where client training runs, not what is shared: it is distributed execution, not a privacy mechanism. With the same torch thread count per client (`sequential_matched`), the Ray backend must reproduce the sequential global model exactly (max |Δparam| = 0). The `sequential` row uses more intra-op threads for speed; any non-zero difference there comes from floating-point reduction order, not from the algorithm.

Generated 2026-10-15 02:52:50 +0530. Ray 2.59.0, start-up 3.2 s (not included in the per-run times), 5 CPUs for Ray, 1 torch thread(s) per actor, 4 torch threads for the sequential backend, 5 clients, 4 rounds, 1 local epoch(s), batch 64.

| Algorithm | Backend | Torch threads/client | Seconds | s/round | Mean slowest-client s/round | Final val ROC-AUC | Max |Δparam| vs Ray |
|---|---|---|---|---|---|---|---|
| fedavg | sequential | 4 | 8.2 | 2.05 | 0.67 | 0.8294 | 6.0e-03 |
| fedavg | sequential_matched | 1 | 6.5 | 1.63 | 0.35 | 0.8294 | 0.0e+00 |
| fedavg | ray | 1 | 7.1 | 1.78 | 0.43 | 0.8294 | 0.0e+00 |
| fedprox_mu0.01 | sequential | 4 | 12.0 | 3.00 | 0.65 | 0.8291 | 7.9e-03 |
| fedprox_mu0.01 | sequential_matched | 1 | 11.5 | 2.88 | 0.60 | 0.8291 | 0.0e+00 |
| fedprox_mu0.01 | ray | 1 | 8.9 | 2.22 | 0.70 | 0.8291 | 0.0e+00 |
