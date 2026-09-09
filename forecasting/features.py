"""Join weather + price + production into a per-hour feature frame for training/inference."""
import pandas as pd

from schemas.models import EnergyPricePoint, ProductionPoint, WeatherPoint


def build_feature_frame(
    weather: list[WeatherPoint],
    prices: list[EnergyPricePoint],
    production: list[ProductionPoint],
) -> pd.DataFrame:
    w = pd.DataFrame([p.model_dump() for p in weather]).set_index("timestamp")
    pr = pd.DataFrame([p.model_dump() for p in prices]).set_index("timestamp")

    prod = pd.DataFrame([p.model_dump() for p in production])
    prod_agg = prod.groupby("timestamp")["output_mw"].sum().to_frame("output_mw")

    df = w.join(pr, how="inner").join(prod_agg, how="inner")
    df["hour_of_day"] = df.index.hour
    df["wind_speed_cubed"] = df["wind_speed_ms"] ** 3  # power ~ v^3, physically motivated feature
    return df.dropna()


FEATURE_COLUMNS = [
    "wind_speed_ms",
    "wind_speed_cubed",
    "wind_direction_deg",
    "temperature_c",
    "pressure_hpa",
    "price_eur_mwh",
    "hour_of_day",
]
TARGET_COLUMN = "output_mw"
