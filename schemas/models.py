"""Pydantic contracts shared across agents, tools, and the API."""
from datetime import datetime
from pydantic import BaseModel, Field


class WeatherPoint(BaseModel):
    timestamp: datetime
    wind_speed_ms: float
    wind_direction_deg: float
    temperature_c: float
    pressure_hpa: float


class EnergyPricePoint(BaseModel):
    timestamp: datetime
    price_eur_mwh: float


class ProductionPoint(BaseModel):
    timestamp: datetime
    turbine_id: str
    output_mw: float


class ForecastRequest(BaseModel):
    farm_id: str
    horizon_hours: int = Field(gt=0, le=168)


class ForecastResult(BaseModel):
    farm_id: str
    generated_at: datetime
    predicted_production_mw: list[float]
    predicted_price_eur_mwh: list[float]
    model_version: str


class AnomalyFlag(BaseModel):
    turbine_id: str
    detected_at: datetime
    description: str
    severity: str  # low | medium | high


class BladeInspectionResult(BaseModel):
    turbine_id: str
    image_ref: str
    damage_detected: bool
    damage_types: list[str] = Field(default_factory=list)
    confidence: float
    description: str = ""


class SpainPricePeriod(BaseModel):
    timestamp: datetime
    predicted_price_eur_mwh: float


class SpainPriceForecastResult(BaseModel):
    target_day: str
    generated_at: datetime
    periods: list[SpainPricePeriod]


class Recommendation(BaseModel):
    farm_id: str
    action: str
    rationale: str
    supporting_sources: list[str] = Field(default_factory=list)
