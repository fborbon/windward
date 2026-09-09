"""Assembles the training frame: real OMIE day-ahead prices for Spain, features engineered
in spain_price/features.py.
"""
from datetime import date, timedelta

from data_sources.omie_price_client import fetch_historical
from spain_price.features import build_feature_frame

DEFAULT_HISTORY_DAYS = 120  # >7-day lag/rolling features need real runway before they're usable


def build_training_frame(history_days: int = DEFAULT_HISTORY_DAYS):
    end = date.today() - timedelta(days=1)  # yesterday is the most recent fully-published day
    start = end - timedelta(days=history_days)
    prices = fetch_historical(start, end)
    if prices.empty:
        raise RuntimeError(f"no OMIE data returned for {start}..{end}")
    return build_feature_frame(prices)
