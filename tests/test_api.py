import pytest
from fastapi.testclient import TestClient

from api.main import app
from forecasting.model_store import model_dir, model_name

client = TestClient(app)


def _model_exported() -> bool:
    """test_forecast needs a model exported from the MLflow registry into data/models/
    (forecasting.export_model) - true on the deploy host, where every deploy runs the export,
    but never in CI, which has no route to the private MLflow server. Skip rather than fail."""
    return (model_dir(model_name("kelmarsh")) / "model.joblib").exists()


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.skipif(not _model_exported(), reason="no exported model in data/models (needs MLflow)")
def test_forecast():
    resp = client.post("/forecast", json={"farm_id": "kelmarsh", "horizon_hours": 6})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predicted_production_mw"]) == 6
    assert all(v >= 0 for v in body["predicted_production_mw"])


def test_wind_prediction_live_weather():
    resp = client.get("/wind-prediction/live-weather", params={"farm_id": "kelmarsh"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["farm_id"] == "kelmarsh"
    assert len(body["points"]) > 0


def test_wind_prediction_payload():
    # 200 once `python -m wind_prediction.export` has run at least once; 404 (with a helpful
    # message, not a crash) beforehand - both are correct behavior for this endpoint.
    resp = client.get("/wind-prediction")
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        body = resp.json()
        assert "results" in body and "taxonomy" in body


def test_operational_payload():
    resp = client.get("/operational/kelmarsh")
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        body = resp.json()
        assert "electrical_losses" in body and "yaw_status" in body
    assert client.get("/operational/not-a-farm").status_code == 404
