"""Derived feature columns shared by training (forecasting/pipeline.py) and inference
(forecasting/predict.py) — one place so the two paths can't silently drift apart."""
import pandas as pd

from analysis.efficiency import air_density_kg_m3


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds hour_of_day, wind_speed_cubed, and air_density_kg_m3 to a weather-indexed frame
    (needs wind_speed_ms, temperature_c, pressure_hpa columns already present). Mutates and
    returns df."""
    df["hour_of_day"] = df.index.hour
    df["wind_speed_cubed"] = df["wind_speed_ms"] ** 3  # power ~ v^3, physically motivated feature
    # Real per-hour air density (ideal gas law) instead of assuming sea-level-standard air —
    # actual wind power available scales with rho, and a farm's real conditions (e.g. a cold
    # front) can diverge from the 1.225 kg/m3 constant enough to matter (see analysis/efficiency.py).
    df["air_density_kg_m3"] = air_density_kg_m3(df["temperature_c"], df["pressure_hpa"])
    return df


FEATURE_COLUMNS = [
    "wind_speed_ms",
    "wind_speed_cubed",
    "wind_direction_deg",
    "temperature_c",
    "pressure_hpa",
    "air_density_kg_m3",
    "price_eur_mwh",
    "hour_of_day",
]
TARGET_COLUMN = "output_mw"
