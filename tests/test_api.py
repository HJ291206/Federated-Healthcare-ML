"""API tests: real saved results when present, plus empty-state and validation behaviour on a temp directory."""

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import create_app  # noqa: E402
from api.store import ROOT  # noqa: E402
from ml.data_validation import FEATURES  # noqa: E402

VALID = {"HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 31, "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
         "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
         "GenHlth": 3, "MentHlth": 0, "PhysHlth": 2, "DiffWalk": 0, "Sex": 0, "Age": 9, "Education": 5, "Income": 6}
SECTIONS = ["overview", "centralized", "federated", "partitions", "convergence", "privacy", "explainability"]


@pytest.fixture(scope="module")
def empty_client(tmp_path_factory):
    d = tmp_path_factory.mktemp("empty")
    return TestClient(create_app(results_dir=d / "results", models_dir=d / "models"))


@pytest.fixture(scope="module")
def real_client():
    return TestClient(create_app())


def test_health_reports_status_flags(empty_client):
    r = empty_client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and set(body["experiments"]) >= {"centralized", "federated", "differential_privacy"}
    assert all(v is False for v in body["experiments"].values())
    assert body["demo_models"] == []


@pytest.mark.parametrize("section", SECTIONS)
def test_empty_state_when_nothing_has_run(empty_client, section):
    r = empty_client.get(f"/api/{section}")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False and "python -m ml." in body["message"]


def test_predict_without_models_returns_503(empty_client):
    r = empty_client.post("/api/predict", json={"model": "centralized_mlp", "features": VALID})
    assert r.status_code == 503
    assert "not available" in r.json()["detail"]


def test_predict_schema_lists_all_features(empty_client):
    body = empty_client.get("/api/predict/schema").json()
    assert sorted(f["name"] for f in body["features"]) == sorted(FEATURES)
    assert "not a medical diagnosis" in body["disclaimer"]


@pytest.mark.parametrize("bad, fragment", [
    ({**VALID, "BMI": 250}, "outside allowed range"),
    ({**VALID, "GenHlth": 0}, "outside allowed range"),
    ({**VALID, "HighBP": 0.5}, "must be an integer code"),
    ({k: v for k, v in VALID.items() if k != "Age"}, "missing features"),
    ({**VALID, "Unknown": 1}, "unknown features"),
])
def test_predict_rejects_invalid_input(empty_client, bad, fragment):
    r = empty_client.post("/api/predict", json={"model": "centralized_mlp", "features": bad})
    assert r.status_code == 422
    assert fragment in str(r.json()["detail"])


def test_predict_rejects_nan_and_strings(empty_client):
    r = empty_client.post("/api/predict", json={"model": "centralized_mlp", "features": {**VALID, "BMI": "abc"}})
    assert r.status_code == 422


def test_figure_path_traversal_is_blocked(real_client):
    assert real_client.get("/api/figures/../data/raw/MANIFEST.sha256").status_code == 404
    assert real_client.get("/api/figures/dataset_summary.csv").status_code == 404


def test_unknown_model_name_is_rejected(empty_client):
    r = empty_client.post("/api/predict", json={"model": "secret_model", "features": VALID})
    assert r.status_code == 422


# ---------------------------------------------------------------- with real saved results (skipped if absent)
needs_results = pytest.mark.skipif(not (ROOT / "results" / "centralized_results.csv").is_file(), reason="centralized results not generated")
needs_models = pytest.mark.skipif(not (ROOT / "models" / "centralized" / "mlp_seed0.pt").is_file(), reason="models not trained")


@needs_results
def test_centralized_section_matches_csv(real_client):
    import pandas as pd

    body = real_client.get("/api/centralized").json()
    assert body["available"] is True
    df = pd.read_csv(ROOT / "results" / "centralized_results.csv")
    assert len(body["results"]) == len(df)
    row = next(r for r in body["results"] if r["model"] == "mlp" and r["seed"] == 0 and r["split"] == "test" and r["operating_point"] == "val_max_f1")
    ref = df[(df.model == "mlp") & (df.seed == 0) & (df.split == "test") & (df.operating_point == "val_max_f1")].iloc[0]
    assert row["roc_auc"] == pytest.approx(ref.roc_auc)


@needs_results
def test_overview_exposes_only_aggregates(real_client):
    body = real_client.get("/api/overview").json()
    assert body["available"] is True
    assert body["primary_file"].endswith(".csv")
    assert all("mean" in f and "feature" in f for f in body["features"])
    assert len(body["features"]) == len(FEATURES)
    text = str(body)
    assert "data/raw" not in text and "split_indices" not in text


@needs_models
def test_predict_returns_probability_with_disclaimer(real_client):
    r = real_client.post("/api/predict", json={"model": "centralized_mlp", "features": VALID})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["predicted_class"] in (0, 1)
    assert "val_max_f1" in body["threshold_source"]
    assert "not a medical diagnosis" in body["disclaimer"]
    # deterministic
    assert real_client.post("/api/predict", json={"model": "centralized_mlp", "features": VALID}).json()["probability"] == body["probability"]


@needs_models
def test_predict_every_available_model(real_client):
    for name in real_client.get("/api/health").json()["demo_models"]:
        r = real_client.post("/api/predict", json={"model": name, "features": VALID})
        assert r.status_code == 200, (name, r.text)


@needs_results
def test_figure_served_when_present(real_client):
    figs = real_client.get("/api/centralized").json()["figures"]
    if not figs:
        pytest.skip("no figures")
    r = real_client.get(f"/api/figures/{figs[0]}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
