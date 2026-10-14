# Demonstration dashboard

React + TypeScript + Vite + Tailwind frontend (`dashboard/`) on a FastAPI backend (`api/`). The backend only reads the
saved experiment outputs under `results/` and the saved models under `models/`; it never reads `data/raw/`, never
returns training records or split indices, and the demo classifier runs in-process (inputs are not stored or
forwarded). All numbers shown come from CSV/JSON files written by the experiments; a page shows an empty state with the
command to run when its experiment has not been executed.

## Start both services (from the repository root, two terminals)

Backend:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-api.txt   # once
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000
```

Frontend:

```powershell
cd dashboard
npm install                                                          # once
npm run dev                                                          # http://localhost:5173
```

If a port is busy, pick others: `--port 8001` for uvicorn and `$env:VITE_API_URL="http://127.0.0.1:8001"; npx vite --port 5180`.
The Vite dev server proxies `/api/*` to the backend, so the browser talks to one origin and no CORS configuration is
needed; for a split deployment set `DASHBOARD_ORIGINS` (comma-separated) on the backend and `VITE_API_BASE` at build time.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_api.py     # backend: empty states, validation, traversal, real-data checks
cd dashboard; npm test; npm run typecheck; npm run build     # frontend: vitest (jsdom) + tsc + production build
```

## Endpoints

| Route | Content |
|---|---|
| `GET /api/health` | which experiments have saved outputs; which demo models are loadable |
| `GET /api/overview` | dataset summary, class distribution, per-feature aggregates (no rows) |
| `GET /api/centralized` | `centralized_results.csv`, MLP history, hyperparameter selection |
| `GET /api/federated` | `results/final/final_comparison*.csv`, bootstrap CIs, local-only per client |
| `GET /api/partitions` | per-client row counts and positive rates for IID / Dirichlet partitions |
| `GET /api/convergence` | per-round history, communication and runtime costs |
| `GET /api/privacy` | DP results, per-client ε accounting, privacy unit |
| `GET /api/explainability` | SHAP global importance, selected individual explanations, rank agreement |
| `GET /api/figures/{path}` | PNG figures under `results/` only (path traversal and non-PNG rejected) |
| `GET /api/predict/schema` | the 21 features with allowed ranges, available models, disclaimer |
| `POST /api/predict` | validated demo inference; returns probability, validation-chosen threshold and disclaimer |

Validation: every feature must be present, finite, inside its documented code range (binary 0/1, GenHlth 1–5, Age 1–13,
Education 1–6, Income 1–8, MentHlth/PhysHlth 0–30, BMI 10–100) and integer-valued except BMI; violations return HTTP 422
with a field-level message. The frontend mirrors these rules before submitting.

## Pages

1. Overview and dataset (target definition as documented vs observed, class balance, feature dictionary, experiment status)
2. Centralized model performance (metrics per model/seed/operating point, MLP learning curve, model selection, ROC/PR figures)
3. Federated-learning comparison (centralized vs local-only vs FedAvg vs FedProx, seed std, bootstrap CIs)
4. IID vs non-IID client distributions (rows and positive rate per client)
5. Convergence and communication cost (per-round AUC/loss, rounds-to-target, bytes)
6. Differential privacy (privacy–utility scatter, per-client ε accounting, privacy unit stated as example-level)
7. SHAP explainability (global importance per model, individual explanations, rank agreement; "importance is not causation")
8. Demo classifier (21-field validated form; outputs labelled as educational model outputs, not a diagnosis)
