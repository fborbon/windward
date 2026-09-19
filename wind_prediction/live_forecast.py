"""Live, forward-looking 48h wind-speed forecast: predicts real wind speed (not farm output_mw
- see evaluate.py for that) for the next 48 hours using three real techniques spanning
naive/classical/foundation-model, and Open-Meteo's own NWP forecast product, then grades all
four against the real ERA5 archive once that window has actually passed. A different exercise
from evaluate.py's backtest: this is genuinely forward-looking (no test-period hindsight, the
target window doesn't exist yet when the forecast is made) and isolates wind-speed forecasting
error from power-curve error.

Two different Open-Meteo products, not one - worth being precise about, since it's easy to
conflate them: `/forecast` (data_sources.meteo_client.fetch_forecast) is a live NWP-model
forecast (ECMWF/GFS/ICON blend) - a genuine independent prediction, exactly like this module's
own techniques, and the second point of comparison below. `/archive` (fetch_historical) is ERA5
REANALYSIS, not a forecast at all - the closest available approximation to "true" past
conditions, confirmed empirically to land with ~1 day of lag (not the 5+ day lag classic ERA5
has), which is what everything gets graded against once real time has passed.

Run via cron (see infra/aws/README.md): `python -m wind_prediction.live_forecast` once a day -
grades any prior forecast whose target window is now safely past the archive's lag, then
records a new one. State lives in data/wind_forecast_log/<farm_id>.json (gitignored, real
runtime data, not committed).
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data_sources.farms import FARMS, Farm
from data_sources.meteo_client import fetch_forecast, fetch_historical

LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "wind_forecast_log"
HORIZON_HOURS = 48
GRADE_BUFFER_HOURS = 30  # past the 48h horizon, before attempting to grade (archive lag + margin)
CONTEXT_HOURS = 24 * 21  # 3 weeks of real recent history to condition the techniques on


def _log_path(farm_id: str) -> Path:
    return LOG_DIR / f"{farm_id}.json"


def _load_log(farm_id: str) -> list[dict]:
    path = _log_path(farm_id)
    return json.loads(path.read_text()) if path.exists() else []


def _save_log(farm_id: str, log: list[dict]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log_path(farm_id).write_text(json.dumps(log, indent=2, default=str))


def _recent_history(farm: Farm) -> pd.Series:
    """Real recent wind speed at the farm's real coordinates from the ERA5 archive - the archive's
    own lag means the last ~24h are typically missing, so this oversamples and trims to what's
    actually there."""
    end = datetime.now(timezone.utc).replace(tzinfo=None).date()
    start = end - timedelta(hours=CONTEXT_HOURS + 72)
    points = fetch_historical(farm.lat, farm.lon, start.isoformat(), end.isoformat())
    s = pd.Series({p.timestamp: p.wind_speed_ms for p in points}).sort_index()
    return s.tail(CONTEXT_HOURS)


def _seasonal_naive_forecast(history: pd.Series, hours: int) -> list[float]:
    """Real recent diurnal pattern (last 24h of real data), repeated - the same seasonal-naive
    concept as wind_prediction.models.naive_baselines, just called once for a genuine forward
    window instead of the backtest's many historical windows."""
    last_24h = history.values[-24:]
    return [float(last_24h[h % 24]) for h in range(hours)]


def _holt_winters_forecast(history: pd.Series, hours: int) -> list[float]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    fitted = ExponentialSmoothing(
        history.reset_index(drop=True), trend="add", seasonal="add", seasonal_periods=24,
        initialization_method="estimated",
    ).fit()
    return [float(v) for v in fitted.forecast(hours).values]


def _chronos_forecast(history: pd.Series, hours: int) -> list[float]:
    import torch
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-tiny", device_map="cpu")
    context = torch.tensor(history.values.astype("float32"))
    _, mean = pipeline.predict_quantiles(inputs=context, prediction_length=hours, quantile_levels=[0.5])
    return [float(v) for v in mean.squeeze(0).numpy()]


def record_forecast(farm_id: str = "kelmarsh") -> dict:
    farm = FARMS[farm_id]
    history = _recent_history(farm)
    if len(history) < 48:
        raise RuntimeError(f"not enough real recent ERA5 archive history for {farm_id}: only {len(history)} hours")

    now = datetime.now(timezone.utc).replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
    target_timestamps = [now + timedelta(hours=h + 1) for h in range(HORIZON_HOURS)]

    # fetch_forecast's `hours` counts from today's UTC midnight, not from `now` - request enough
    # to reach target_timestamps[-1] regardless of what time of day this runs.
    midnight_today = now.replace(hour=0)
    om_hours_needed = int((target_timestamps[-1] - midnight_today).total_seconds() // 3600) + 1
    om_points = fetch_forecast(farm.lat, farm.lon, hours=om_hours_needed)
    om_by_ts = {p.timestamp: p.wind_speed_ms for p in om_points}
    om_values = [om_by_ts.get(ts) for ts in target_timestamps]

    record = {
        "generated_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        "farm_id": farm_id,
        "lat": farm.lat, "lon": farm.lon,
        "history_end": history.index[-1].isoformat(),
        "history_hours_used": len(history),
        "target_timestamps": [ts.isoformat() for ts in target_timestamps],
        "techniques": {
            "seasonal_naive": _seasonal_naive_forecast(history, HORIZON_HOURS),
            "holt_winters": _holt_winters_forecast(history, HORIZON_HOURS),
            "chronos_bolt_tiny": _chronos_forecast(history, HORIZON_HOURS),
        },
        "open_meteo_forecast_ms": om_values,
        "graded": False,
    }
    log = _load_log(farm_id)
    log.append(record)
    _save_log(farm_id, log)
    return record


def _mae(preds: list, actual: list) -> float | None:
    pairs = [(p, a) for p, a in zip(preds, actual) if p is not None and a is not None]
    return float(np.mean([abs(p - a) for p, a in pairs])) if pairs else None


def grade_pending(farm_id: str = "kelmarsh") -> list[dict]:
    log = _load_log(farm_id)
    now = datetime.now(timezone.utc).replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
    farm = FARMS[farm_id]
    graded_now = []

    for record in log:
        if record["graded"]:
            continue
        last_target = datetime.fromisoformat(record["target_timestamps"][-1])
        if now - last_target < timedelta(hours=GRADE_BUFFER_HOURS):
            continue

        start = record["target_timestamps"][0][:10]
        end = record["target_timestamps"][-1][:10]
        actual_points = fetch_historical(farm.lat, farm.lon, start, end)
        actual_by_ts = {p.timestamp.isoformat(): p.wind_speed_ms for p in actual_points}
        actual = [actual_by_ts.get(ts) for ts in record["target_timestamps"]]
        if sum(v is not None for v in actual) < HORIZON_HOURS * 0.9:
            continue  # archive not fully populated yet - try again on a later run

        record["actual_wind_speed_ms"] = actual
        errors = {name: _mae(preds, actual) for name, preds in record["techniques"].items()}
        errors["open_meteo_forecast"] = _mae(record["open_meteo_forecast_ms"], actual)
        record["errors_mae"] = errors
        record["graded"] = True
        graded_now.append(record)

    if graded_now:
        _save_log(farm_id, log)
    return graded_now


def run(farm_id: str = "kelmarsh") -> None:
    for g in grade_pending(farm_id):
        print(f"graded forecast from {g['generated_at']}: {g['errors_mae']}")
    record = record_forecast(farm_id)
    print(f"recorded new forecast at {record['generated_at']}, target window {record['target_timestamps'][0]} .. {record['target_timestamps'][-1]}")


if __name__ == "__main__":
    run()
