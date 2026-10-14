"""MediFederate AI — FastAPI application.

    uvicorn api.main:app --reload --port 8000

Serves aggregated experiment outputs, figures, a demo prediction endpoint,
live experiment execution via background threads, report generation,
settings persistence, and simulated hospital-network metadata.

Raw survey rows, split indices and model files are never exposed.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from api.db import Database
from api.predict import FEATURE_SCHEMA, ModelRegistry, PredictRequest, PredictResponse
from api.store import DISCLAIMERS, ROOT, ResultsStore

log = logging.getLogger("api")

# ────────────────────────────────────────────────────────────────────
# Pydantic models for experiment & report endpoints
# ────────────────────────────────────────────────────────────────────

class CreateExperimentRequest(BaseModel):
    name: str = ""
    algorithm: Literal["fedavg", "fedprox"] = "fedavg"
    partition: Literal["iid", "dirichlet"] = "iid"
    num_clients: int = Field(5, ge=2, le=10)
    rounds: int = Field(8, ge=1, le=50)
    local_epochs: int = Field(1, ge=1, le=10)
    batch_size: int = Field(64, ge=16, le=512)
    learning_rate: float = Field(0.01, gt=0, le=1.0)
    fedprox_mu: float = Field(0.01, ge=0, le=1.0)
    seed: int = Field(0, ge=0, le=9999)
    sample_fraction: float = Field(0.1, gt=0, le=1.0)


class GenerateReportRequest(BaseModel):
    kind: Literal[
        "comparison_csv",
        "full_research_pdf",
        "centralized_vs_federated",
        "fedavg_vs_fedprox",
        "iid_vs_noniid",
        "explainability_report",
        "privacy_utility",
    ] = "comparison_csv"


class UpdateSettingsRequest(BaseModel):
    dataset_path: str | None = None
    default_algorithm: str | None = None
    default_rounds: int | None = None
    default_clients: int | None = None
    default_lr: float | None = None
    default_batch_size: int | None = None
    default_mu: float | None = None
    device: str | None = None


# ────────────────────────────────────────────────────────────────────
# Application factory
# ────────────────────────────────────────────────────────────────────

def create_app(results_dir: Path | None = None, models_dir: Path | None = None) -> FastAPI:
    store = ResultsStore(results_dir, models_dir)
    registry = ModelRegistry(store.models, store.results)

    # Database
    db_path = store.results / "app" / "medifederate.db"
    db = Database(db_path)
    interrupted = db.mark_interrupted()
    if interrupted:
        log.info("marked %d interrupted experiment(s) as failed", interrupted)
    db.audit("server_start", version="1.0.0", python=sys.version.split()[0])

    # Reports directory
    reports_dir = store.results / "app" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="MediFederate AI API",
        version="1.0.0",
        description="Privacy-preserving healthcare intelligence platform API.",
    )
    origins = os.environ.get(
        "DASHBOARD_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5180",
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error", "path": request.url.path},
        )

    # ── Existing read-only endpoints (preserved) ─────────────────

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "experiments": store.status(),
            "demo_models": registry.available_models(),
            "database": db.health(),
        }

    @app.get("/api/overview")
    def overview():
        return store.overview().as_dict()

    @app.get("/api/centralized")
    def centralized():
        return store.centralized().as_dict()

    @app.get("/api/federated")
    def federated():
        return store.federated().as_dict()

    @app.get("/api/partitions")
    def partitions():
        return store.partitions().as_dict()

    @app.get("/api/convergence")
    def convergence():
        return store.convergence().as_dict()

    @app.get("/api/privacy")
    def privacy():
        return store.privacy().as_dict()

    @app.get("/api/explainability")
    def explainability():
        return store.explainability().as_dict()

    @app.get("/api/figures/{path:path}")
    def figure(path: str):
        resolved = store.figure_path(path)
        if resolved is None:
            raise HTTPException(status_code=404, detail=f"figure {path!r} not found")
        return FileResponse(resolved, media_type="image/png")

    @app.get("/api/predict/schema")
    def predict_schema():
        return {
            "features": FEATURE_SCHEMA,
            "models": registry.available_models(),
            "disclaimer": DISCLAIMERS["prediction"],
        }

    @app.post("/api/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        try:
            return registry.predict(req)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"{exc}. Run `python -m ml.run centralized` first.",
            ) from exc

    # ── Experiment endpoints ──────────────────────────────────────

    @app.get("/api/experiments")
    def list_experiments():
        return db.list_experiments()

    @app.get("/api/experiments/{exp_id}")
    def get_experiment(exp_id: str):
        exp = db.get_experiment(exp_id)
        if exp is None:
            raise HTTPException(status_code=404, detail="experiment not found")
        return exp

    @app.post("/api/experiments")
    def create_experiment(req: CreateExperimentRequest):
        # Prevent concurrent experiments
        running = db.running_experiment()
        if running:
            raise HTTPException(
                status_code=409,
                detail=f"experiment {running['id']} is already running; wait for it to finish or cancel it",
            )

        exp_id = f"exp-{uuid.uuid4().hex[:8]}"
        name = req.name or f"{req.algorithm.upper()} {req.partition} {req.num_clients}c {req.rounds}r"
        config = req.model_dump()
        config["name"] = name

        record = db.create_experiment(exp_id, name, req.algorithm, config)
        db.audit("experiment_created", experiment_id=exp_id, algorithm=req.algorithm)

        # Launch in background thread via experiment_runner
        from api.experiment_runner import start_experiment
        start_experiment(exp_id, db, config, models_dir=store.models)

        return record

    @app.post("/api/experiments/{exp_id}/cancel")
    def cancel_experiment(exp_id: str):
        exp = db.get_experiment(exp_id)
        if exp is None:
            raise HTTPException(status_code=404, detail="experiment not found")
        if exp["status"] not in ("queued", "running"):
            raise HTTPException(status_code=400, detail="experiment is not running")
        db.request_cancel(exp_id)
        db.audit("experiment_cancel_requested", experiment_id=exp_id)
        return db.get_experiment(exp_id)

    @app.delete("/api/experiments/{exp_id}")
    def delete_experiment(exp_id: str):
        if not db.delete_experiment(exp_id):
            raise HTTPException(status_code=404, detail="experiment not found or still running")
        db.audit("experiment_deleted", experiment_id=exp_id)
        return {"ok": True}

    # ── Report endpoints ──────────────────────────────────────────

    @app.get("/api/reports")
    def list_reports():
        return db.list_reports()

    @app.post("/api/reports/generate")
    def generate_report(req: GenerateReportRequest):
        try:
            from api.report_gen import create_report_entry
            report = create_report_entry(db, store, req.kind, reports_dir=reports_dir)
            db.audit("report_generated", report_id=report["id"], kind=req.kind)
            return report
        except Exception as exc:
            log.exception("report generation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/reports/{report_id}/download")
    def download_report(report_id: str):
        report = db.get_report(report_id)
        if report is None:
            raise HTTPException(status_code=404, detail="report not found")
        path = reports_dir / report["filename"]
        if not path.is_file():
            raise HTTPException(status_code=404, detail="report file missing from disk")
        return FileResponse(path, media_type=report["media_type"], filename=report["filename"])

    # ── Settings endpoints ────────────────────────────────────────

    @app.get("/api/settings")
    def get_settings():
        saved = db.all_settings()
        cuda_available = torch.cuda.is_available()
        devices = ["cpu"]
        if cuda_available:
            devices.append(f"cuda ({torch.cuda.get_device_name(0)})")
        return {
            "dataset_path": saved.get("dataset_path", str(ROOT / "data" / "raw" / "diabetes_binary_health_indicators_BRFSS2015.csv")),
            "default_algorithm": saved.get("default_algorithm", "fedavg"),
            "default_rounds": saved.get("default_rounds", 8),
            "default_clients": saved.get("default_clients", 5),
            "default_batch_size": saved.get("default_batch_size", 64),
            "default_lr": saved.get("default_lr", 0.01),
            "default_mu": saved.get("default_mu", 0.01),
            "device": saved.get("device", "cpu"),
            "device_available": devices,
            "db_health": db.health(),
            "system_info": {
                "app_name": "MediFederate AI",
                "app_version": "1.0.0",
                "python_version": sys.version.split()[0],
                "torch_version": torch.__version__,
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "cuda_available": cuda_available,
                "backend_status": "online",
            },
        }

    @app.put("/api/settings")
    def update_settings(req: UpdateSettingsRequest):
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        for k, v in updates.items():
            db.set_setting(k, v)
        db.audit("settings_updated", **updates)
        return get_settings()

    @app.get("/api/settings/system")
    def system_info():
        return {
            "app_name": "MediFederate AI",
            "app_version": "1.0.0",
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cuda_available": torch.cuda.is_available(),
        }

    # ── Hospital network endpoint ─────────────────────────────────

    @app.get("/api/network")
    def network():
        """Simulated hospital network metadata from the latest partition data."""
        names = [
            "St. Jude Research Medical Center",
            "Metro General Hospital & Health",
            "University Health System Center",
            "Riverside Community Health Clinic",
            "Valley Memorial Medical Center",
            "Northwest Regional Hospital",
            "Covenant Health Science Center",
            "Summit Healthcare Pavilion",
            "Baystate Medical Institute",
            "Piedmont Academic Hospital",
        ]
        part_section = store.partitions()
        clients = []
        if part_section.available:
            part_data = part_section.data.get("partitions", [])
            train_parts = [r for r in part_data if r.get("split") == "train"]
            for r in train_parts:
                cid = int(r.get("client", 0))
                n = int(r.get("n", 0))
                n_pos = int(r.get("n_positive", 0))
                clients.append({
                    "client_id": cid,
                    "institution_name": names[cid % len(names)],
                    "simulated": True,
                    "n_records": n,
                    "positive_records": n_pos,
                    "positive_rate": round(n_pos / n, 4) if n > 0 else 0,
                    "status": "ready",
                    "partition": str(r.get("partition", "unknown")),
                })
        return {
            "disclaimer": DISCLAIMERS["clients"],
            "clients": clients,
            "aggregation_server": {
                "name": "MediFederate Aggregation Server",
                "status": "online",
                "algorithm": "FedAvg / FedProx",
            },
        }

    # ── Model comparison endpoint ─────────────────────────────────

    @app.get("/api/models/compare")
    def models_compare():
        """Aggregate model metrics from centralized and federated results."""
        models: list[dict[str, Any]] = []

        # Centralized models
        cent = store.centralized()
        if cent.available:
            results = cent.data.get("results", [])
            test_rows = [r for r in results if r.get("split") == "test" and r.get("operating_point") == "val_max_f1" and r.get("seed") == 0]
            for r in test_rows:
                models.append({
                    "model": f"centralized_{r.get('model', 'unknown')}",
                    "label": str(r.get("model", "")).replace("_", " ").title(),
                    "family": "centralized",
                    "accuracy": r.get("accuracy"),
                    "precision": r.get("precision_class1"),
                    "recall": r.get("recall_class1"),
                    "f1": r.get("f1_class1"),
                    "roc_auc": r.get("roc_auc"),
                    "pr_auc": r.get("pr_auc"),
                    "train_seconds": r.get("train_seconds"),
                })

        # Federated models
        fed = store.federated()
        if fed.available:
            final = fed.data.get("final_comparison", [])
            for r in final:
                if r.get("family") not in ("centralized", "local_only", "dp"):
                    models.append({
                        "model": f"{r.get('setting', '')}_{r.get('method', '')}",
                        "label": f"{r.get('setting', '')} {r.get('method', '')}".replace("_", " ").title(),
                        "family": "federated",
                        "accuracy": r.get("accuracy_mean"),
                        "precision": r.get("precision_class1_mean"),
                        "recall": r.get("recall_class1_mean"),
                        "f1": r.get("f1_class1_mean"),
                        "roc_auc": r.get("roc_auc_mean"),
                        "pr_auc": r.get("pr_auc_mean"),
                        "train_seconds": r.get("train_seconds_mean"),
                        "rounds": r.get("rounds"),
                        "total_bytes": r.get("total_bytes"),
                    })

        return {"models": models, "disclaimer": DISCLAIMERS["prediction"]}

    app.state.store, app.state.registry, app.state.db = store, registry, db
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True, app_dir=str(ROOT))
