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
