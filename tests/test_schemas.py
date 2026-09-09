from schemas.models import ForecastRequest


def test_forecast_request_valid():
    req = ForecastRequest(farm_id="farm-1", horizon_hours=24)
    assert req.horizon_hours == 24
