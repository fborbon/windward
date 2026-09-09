"""Open-Meteo client — free, no API key required."""
import httpx

import config
from schemas.models import WeatherPoint

ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com/v1"


def fetch_forecast(lat: float, lon: float, hours: int = 48) -> list[WeatherPoint]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "windspeed_10m,winddirection_10m,temperature_2m,pressure_msl",
        "windspeed_unit": "ms",  # API defaults to km/h — schema is m/s
        "forecast_days": max(1, hours // 24 + 1),
    }
    resp = httpx.get(f"{config.OPEN_METEO_BASE_URL}/forecast", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    return [
        WeatherPoint(
            timestamp=t,
            wind_speed_ms=ws,
            wind_direction_deg=wd,
            temperature_c=temp,
            pressure_hpa=pres,
        )
        for t, ws, wd, temp, pres in zip(
            data["time"],
            data["windspeed_10m"],
            data["winddirection_10m"],
            data["temperature_2m"],
            data["pressure_msl"],
        )
    ][:hours]


def fetch_historical(lat: float, lon: float, start_date: str, end_date: str) -> list[WeatherPoint]:
    """start_date/end_date as 'YYYY-MM-DD'. Backed by ERA5 reanalysis, no key required."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "windspeed_10m,winddirection_10m,temperature_2m,pressure_msl",
        "windspeed_unit": "ms",
    }
    resp = httpx.get(f"{ARCHIVE_BASE_URL}/archive", params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    return [
        WeatherPoint(
            timestamp=t,
            wind_speed_ms=ws,
            wind_direction_deg=wd,
            temperature_c=temp,
            pressure_hpa=pres,
        )
        for t, ws, wd, temp, pres in zip(
            data["time"],
            data["windspeed_10m"],
            data["winddirection_10m"],
            data["temperature_2m"],
            data["pressure_msl"],
        )
        if ws is not None
    ]
