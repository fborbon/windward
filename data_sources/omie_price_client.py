"""Spain (Iberian market, OMIE) day-ahead electricity price — real data, public, no auth.

OMIE publishes one file per day, semicolon-delimited: year;month;day;period;price_pt;price_es;
`period` is a 15-minute slot (1-96/day; up to 100 on the rare day with a DST-extra hour).
Public, no token, no registration — unlike ENTSO-E, which needs a token for the equivalent
EU-wide day-ahead product (see data_sources/energy_price_client.py).
"""
from datetime import date, datetime, timedelta

import httpx
import pandas as pd

BASE_URL = "https://www.omie.es/es/file-download"


def fetch_day(day: date) -> pd.DataFrame:
    """Real day-ahead price for one day, Spain (`price_es`). Returns empty DataFrame if the
    file isn't published yet (e.g. asking for a future date beyond the current auction)."""
    filename = f"marginalpdbc_{day:%Y%m%d}.1"
    resp = httpx.get(
        BASE_URL, params={"parents": "marginalpdbc", "filename": filename}, timeout=30, follow_redirects=True
    )
    if resp.status_code != 200 or not resp.text.strip() or resp.text.startswith("<"):
        return pd.DataFrame(columns=["timestamp", "price_eur_mwh"])

    rows = []
    for line in resp.text.strip().splitlines()[1:]:  # skip "MARGINALPDBC;" header
        parts = line.strip().rstrip(";").split(";")
        if len(parts) < 6:
            continue
        year, month, dom, period, _price_pt, price_es = parts[:6]
        period = int(period)
        # period 1 = 00:00-00:15 local; convert to a UTC-naive timestamp assuming CET/CEST
        # is out of scope here — periods are kept as a simple offset from local midnight,
        # which is what the model actually needs (period-of-day is the feature, not the
        # absolute UTC instant).
        ts = datetime(int(year), int(month), int(dom)) + timedelta(minutes=15 * (period - 1))
        rows.append({"timestamp": ts, "price_eur_mwh": float(price_es)})
    return pd.DataFrame(rows)


def fetch_historical(start: date, end: date) -> pd.DataFrame:
    """Real OMIE prices for every day in [start, end], concatenated. One HTTP request per day."""
    frames = []
    day = start
    while day <= end:
        df = fetch_day(day)
        if not df.empty:
            frames.append(df)
        day += timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=["timestamp", "price_eur_mwh"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset="timestamp").sort_values("timestamp")
