"""FastAPI service wrapping the LangGraph agent."""
from fastapi import FastAPI

from forecasting.predict import predict_production
from schemas.models import ForecastRequest, ForecastResult

app = FastAPI(title="Windward")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResult)
def forecast(request: ForecastRequest):
    return predict_production(farm_id=request.farm_id, horizon_hours=request.horizon_hours)
