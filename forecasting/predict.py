"""Batch inference: fetch forward-looking weather + price, run the registered model for a given farm."""
import pandas as pd

from data_sources.energy_price_client import synthetic_prices
from data_sources.farms import FARMS, Farm
from data_sources.meteo_client import fetch_forecast
from forecasting.features import FEATURE_COLUMNS
from forecasting.registry import load_latest_model
from forecasting.train import model_name
from schemas.models import ForecastResult


def predict_production(farm_id: str = "kelmarsh", horizon_hours: int = 48) -> ForecastResult:
    farm: Farm = FARMS[farm_id]
    weather = fetch_forecast(farm.lat, farm.lon, hours=horizon_hours)
    prices = synthetic_prices(len(weather))
    for p, w in zip(prices, weather):
        p.timestamp = w.timestamp

    w_df = pd.DataFrame([p.model_dump() for p in weather]).set_index("timestamp")
    p_df = pd.DataFrame([p.model_dump() for p in prices]).set_index("timestamp")
    df = w_df.join(p_df, how="inner")
    df["hour_of_day"] = df.index.hour
    df["wind_speed_cubed"] = df["wind_speed_ms"] ** 3

    model = load_latest_model(model_name(farm_id))
    # trained on farm-level totals (features.build_feature_frame sums production across turbines)
    farm_mw = model.predict(df[FEATURE_COLUMNS])

    return ForecastResult(
        farm_id=farm.farm_id,
        generated_at=pd.Timestamp.utcnow().to_pydatetime(),
        predicted_production_mw=[float(v) for v in farm_mw],
        predicted_price_eur_mwh=[p.price_eur_mwh for p in prices],
        model_version="latest",
    )
