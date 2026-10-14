"""Tests for MediFederate AI new enterprise endpoints:
Settings, Network, Models Compare, Reports generation/download, and Experiments CRUD.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture(scope="module")
def app_client(tmp_path_factory):
    d = tmp_path_factory.mktemp("medifederate_test")
    app = create_app(results_dir=d / "results", models_dir=d / "models")
    return TestClient(app)


def test_settings_read_and_update(app_client):
    # GET settings
    r = app_client.get("/api/settings")
    assert r.status_code == 200
    settings = r.json()
    assert "default_algorithm" in settings
    assert "system_info" in settings
    assert settings["system_info"]["app_name"] == "MediFederate AI"

    # PUT settings
    update_data = {
        "default_algorithm": "fedprox",
        "default_rounds": 12,
        "default_clients": 6,
    }
    r = app_client.put("/api/settings", json=update_data)
    assert r.status_code == 200
    updated = r.json()
    assert updated["default_algorithm"] == "fedprox"
    assert updated["default_rounds"] == 12
    assert updated["default_clients"] == 6


def test_network_endpoint(app_client):
    r = app_client.get("/api/network")
    assert r.status_code == 200
    data = r.json()
    assert "clients" in data
    assert "aggregation_server" in data
    assert "disclaimer" in data


def test_models_compare_endpoint(app_client):
    r = app_client.get("/api/models/compare")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert "disclaimer" in data
    assert isinstance(data["models"], list)


def test_report_generation_csv(app_client):
    # Generate CSV report
    r = app_client.post("/api/reports/generate", json={"kind": "comparison_csv"})
    assert r.status_code == 200
    report = r.json()
    assert report["kind"] == "comparison_csv"
    assert report["filename"].endswith(".csv")
    rep_id = report["id"]

    # List reports
    r = app_client.get("/api/reports")
    assert r.status_code == 200
    reports = r.json()
    assert any(rp["id"] == rep_id for rp in reports)

    # Download report
    r = app_client.get(f"/api/reports/{rep_id}/download")
    assert r.status_code == 200
    assert "model,label,family" in r.text or "text/csv" in r.headers.get("content-type", "")


def test_report_generation_pdf(app_client):
    # Generate PDF research report
    r = app_client.post("/api/reports/generate", json={"kind": "full_research_pdf"})
    assert r.status_code == 200
    report = r.json()
    assert report["kind"] == "full_research_pdf"
    assert report["filename"].endswith(".pdf")
    rep_id = report["id"]

    # Download PDF
    r = app_client.get(f"/api/reports/{rep_id}/download")
    assert r.status_code == 200
    assert r.content.startswith(b"%PDF")


def test_report_generation_markdown(app_client):
    # Generate markdown report
    r = app_client.post("/api/reports/generate", json={"kind": "centralized_vs_federated"})
    assert r.status_code == 200
    report = r.json()
    assert report["filename"].endswith(".md")
    rep_id = report["id"]

    r = app_client.get(f"/api/reports/{rep_id}/download")
    assert r.status_code == 200
    assert "MediFederate AI" in r.text


def test_experiment_lifecycle(app_client):
    # List initial experiments
    r = app_client.get("/api/experiments")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

    # Create experiment
    exp_payload = {
        "name": "Integration Test Experiment",
        "algorithm": "fedavg",
        "partition": "iid",
        "num_clients": 3,
        "rounds": 2,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.05,
        "sample_fraction": 0.01,
        "seed": 42,
    }
    r = app_client.post("/api/experiments", json=exp_payload)
    assert r.status_code == 200
    exp = r.json()
    exp_id = exp["id"]
    assert exp["status"] in ("queued", "running")

    # Get experiment
    r = app_client.get(f"/api/experiments/{exp_id}")
    assert r.status_code == 200
    assert r.json()["id"] == exp_id

    # Request cancel
    r = app_client.post(f"/api/experiments/{exp_id}/cancel")
    assert r.status_code in (200, 400)  # 400 if already finished or cancelled
