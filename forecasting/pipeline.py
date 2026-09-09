"""Assembles the training frame for a given farm.

Primary path: real SCADA production (see data_sources/greenbyte_scada.py) joined with real
historical weather (Open-Meteo ERA5 archive) for the same period — this is the actual
forecasting problem: predict real turbine output from external, forecastable meteorological
data (not the turbines' own on-site anemometers).

Day-ahead price is still synthetic: ENTSO-E's GB coverage/format differs post-Brexit from
the EU day-ahead product this client targets, and it isn't needed to demonstrate the
forecasting pipeline — swap in a real GB price feed (Elexon BMRS) if that becomes relevant.
"""
import pandas as pd

from data_sources.energy_price_client import synthetic_prices
from data_sources.farms import Farm
from data_sources.greenbyte_scada import load_farm_hourly_production
from data_sources.meteo_client import fetch_historical
from forecasting.features import FEATURE_COLUMNS, TARGET_COLUMN


def build_training_frame(farm: Farm) -> pd.DataFrame:
    missing = [p for p in farm.scada_zips if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing SCADA file(s) for {farm.farm_id}: {missing} — see README §4 Data Sources for the download command"
        )

    production = load_farm_hourly_production(farm.scada_zips)  # hourly, indexed, output_mw

    start = production.index.min().date().isoformat()
    end = production.index.max().date().isoformat()
    weather = fetch_historical(farm.lat, farm.lon, start, end)
    w_df = pd.DataFrame([p.model_dump() for p in weather]).set_index("timestamp")

    prices = synthetic_prices(len(weather))
    for p, w in zip(prices, weather):
        p.timestamp = w.timestamp
    p_df = pd.DataFrame([p.model_dump() for p in prices]).set_index("timestamp")

    df = w_df.join(p_df, how="inner").join(production, how="inner")
    df["hour_of_day"] = df.index.hour
    df["wind_speed_cubed"] = df["wind_speed_ms"] ** 3
    return df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
