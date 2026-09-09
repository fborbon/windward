"""Feature engineering for Spain day-ahead price forecasting (OMIE, 15-minute periods).

Deliberately no weather/demand inputs — those would need their own real data sources.
This is a calendar + autoregressive baseline: period-of-day, day-of-week, month, and lag
features (same period yesterday / a week ago / the trailing weekly mean for that period),
which is a standard, defensible starting point for short-term electricity price forecasting.
"""
import pandas as pd

PERIODS_PER_DAY = 96

FEATURE_COLUMNS = [
    "period_of_day",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "price_lag_1d",
    "price_lag_2d",
    "price_lag_7d",
    "price_rolling_mean_7d",
]
TARGET_COLUMN = "price_eur_mwh"


def build_feature_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """prices: columns [timestamp, price_eur_mwh], one row per 15-min period, sorted."""
    df = prices.set_index("timestamp").sort_index().copy()

    df["period_of_day"] = ((df.index.hour * 60 + df.index.minute) // 15).astype(int)
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # Lags are in units of 15-min periods: 1 day = 96, 7 days = 672.
    df["price_lag_1d"] = df["price_eur_mwh"].shift(PERIODS_PER_DAY)
    df["price_lag_2d"] = df["price_eur_mwh"].shift(2 * PERIODS_PER_DAY)
    df["price_lag_7d"] = df["price_eur_mwh"].shift(7 * PERIODS_PER_DAY)
    df["price_rolling_mean_7d"] = df["price_eur_mwh"].rolling(7 * PERIODS_PER_DAY, min_periods=PERIODS_PER_DAY).mean()

    return df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
