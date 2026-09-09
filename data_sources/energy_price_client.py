"""ENTSO-E day-ahead price client, with a synthetic fallback so the pipeline runs
end-to-end before you've registered for a token.

Get a free token at https://transparency.entsoe.eu (Account Settings -> Web API Security
Token) and set ENTSOE_API_TOKEN in .env to switch from synthetic to real prices.
"""
import random
from datetime import datetime, timedelta

import pandas as pd

import config
from schemas.models import EnergyPricePoint


def fetch_day_ahead_prices(bidding_zone_eic: str, hours: int = 48) -> list[EnergyPricePoint]:
    if not config.ENTSOE_API_TOKEN:
        return synthetic_prices(hours)

    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=config.ENTSOE_API_TOKEN)
    start = pd.Timestamp.now(tz="UTC").floor("h")
    end = start + pd.Timedelta(hours=hours)
    series = client.query_day_ahead_prices(bidding_zone_eic, start=start, end=end)
    return [
        EnergyPricePoint(timestamp=ts.to_pydatetime(), price_eur_mwh=float(price))
        for ts, price in series.items()
    ]


def synthetic_prices(hours: int) -> list[EnergyPricePoint]:
    """Rough day/night price curve: higher during 07-22h, lower overnight, plus noise."""
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    points = []
    for h in range(hours):
        ts = now + timedelta(hours=h)
        base = 90 if 7 <= ts.hour <= 22 else 40
        points.append(EnergyPricePoint(timestamp=ts, price_eur_mwh=max(0.0, base + random.gauss(0, 10))))
    return points
