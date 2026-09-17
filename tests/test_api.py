from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


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
