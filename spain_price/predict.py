"""Predict Spain day-ahead prices for a target day (defaults to tomorrow) — the actual
day-ahead forecasting use case: at the time OMIE's own auction clears, you'd want your own
forecast of what it's about to publish, or a forecast for days beyond OMIE's own 1-day horizon.
"""
from datetime import date, datetime, timedelta

import pandas as pd

from data_sources.omie_price_client import fetch_historical
from forecasting.registry import load_latest_model
from spain_price.features import FEATURE_COLUMNS, PERIODS_PER_DAY
from spain_price.train import MODEL_NAME


def predict_day(target_day: date | None = None) -> dict:
    target_day = target_day or (date.today() + timedelta(days=1))

    # Need real history back far enough for the 7-day rolling mean feature.
    history_end = target_day - timedelta(days=1)
    history_start = history_end - timedelta(days=14)
    history = fetch_historical(history_start, history_end)
    if history.empty:
        raise RuntimeError(f"no OMIE history available to build features for {target_day}")
    history = history.set_index("timestamp").sort_index()["price_eur_mwh"]

    rows = []
    for period in range(PERIODS_PER_DAY):
        ts = datetime(target_day.year, target_day.month, target_day.day) + timedelta(minutes=15 * period)
        rows.append(
            {
                "timestamp": ts,
                "period_of_day": period,
                "hour_of_day": ts.hour,
                "day_of_week": ts.weekday(),
                "month": ts.month,
                "is_weekend": int(ts.weekday() >= 5),
                "price_lag_1d": _lookup(history, ts - timedelta(days=1)),
                "price_lag_2d": _lookup(history, ts - timedelta(days=2)),
                "price_lag_7d": _lookup(history, ts - timedelta(days=7)),
                "price_rolling_mean_7d": history.loc[ts - timedelta(days=7) : ts - timedelta(minutes=15)].mean(),
            }
        )
    df = pd.DataFrame(rows).dropna(subset=FEATURE_COLUMNS)
    if df.empty:
        raise RuntimeError(f"insufficient real history to build a full forecast for {target_day}")

    # pandas' DatetimeIndex .hour/.dayofweek/.month accessors (used at training time in
    # features.py) yield int32; plain python ints here default to int64 — must match the
    # MLflow model's inferred signature exactly or predict() raises a schema error.
    for col in ("hour_of_day", "day_of_week", "month"):
        df[col] = df[col].astype("int32")

    model = load_latest_model(MODEL_NAME)
    df["predicted_price_eur_mwh"] = model.predict(df[FEATURE_COLUMNS])

    return {
        "target_day": target_day.isoformat(),
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "periods": [
            {"timestamp": r.timestamp.isoformat(), "predicted_price_eur_mwh": float(r.predicted_price_eur_mwh)}
            for r in df.itertuples()
        ],
    }


def _lookup(history: pd.Series, ts: datetime) -> float | None:
    try:
        return float(history.loc[ts])
    except KeyError:
        return None
