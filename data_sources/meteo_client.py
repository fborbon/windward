"""Open-Meteo client — free, no API key required."""
import time

import httpx

import config
from schemas.models import WeatherPoint

ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com/v1"


def _get_with_retry(url: str, params: dict, timeout: float, max_attempts: int = 3) -> httpx.Response:
    """Open-Meteo has been observed to intermittently 500 on an otherwise-valid request and
    recover within seconds (caught this live: 3 straight 500s from one host, then 200s a
    minute later) — retry 5xx/network errors with a short backoff before giving up. 4xx
    isn't retried; that means the request itself is wrong, retrying won't help."""
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            resp = httpx.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            is_server_side = isinstance(e, httpx.TransportError) or e.response.status_code >= 500
            if not is_server_side or attempt == max_attempts - 1:
                raise
            last_exc = e
            time.sleep(2 * (attempt + 1))
    raise last_exc  # pragma: no cover — unreachable, loop always returns or raises above


def fetch_forecast(lat: float, lon: float, hours: int = 48) -> list[WeatherPoint]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "windspeed_10m,winddirection_10m,temperature_2m,pressure_msl",
        "windspeed_unit": "ms",  # API defaults to km/h — schema is m/s
        "forecast_days": max(1, hours // 24 + 1),
    }
    resp = _get_with_retry(f"{config.OPEN_METEO_BASE_URL}/forecast", params, timeout=30)
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
    resp = _get_with_retry(f"{ARCHIVE_BASE_URL}/archive", params, timeout=60)
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
