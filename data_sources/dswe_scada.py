"""Loader for the "Inland-Offshore Wind Farm Dataset1" (Zenodo doi:10.5281/zenodo.5516552,
CC BY 4.0), one of the real open datasets published alongside Yu Ding's *Data Science for Wind
Energy* (Chapter 5, aml.engr.tamu.edu/book-dswe). Six real turbines (WT1-WT6) each paired with
one of three real on-site meteorological masts — WT1+WT2 share Mast A (inland), WT3+WT4 share
Mast B (inland), WT5+WT6 share Mast C (offshore) — real 10-minute wind speed/direction/air
density/turbulence measured at the mast, real turbine power as a % of rated capacity.

Like EDP Wind Farm A, this backs diagnosis/RAG only, not a registered Farm in
data_sources.farms.FARMS: no coordinates are disclosed anywhere in the dataset or the book.
Unlike EDP, rows have no per-row timestamp at all (just a bare "Sequence No.", and the row
count is meaningfully short of what continuous 10-minute coverage over the documented period
would give — e.g. WT1 has 47,542 rows against ~52,704 expected for a full year, so gaps are
real, not evenly spaced), so this dataset can't support real time-matched forecasting either.

What it CAN support, and EDP couldn't: the Zenodo record's own description gives a real
calendar date range per mast (below, quoted from that description, not invented), and the mast
gives a real, independent on-site wind-speed measurement. That's enough for a real
Measure-Correlate-Predict-style ratio (analysis.efficiency.measure_correlate_predict) against
Open-Meteo's ERA5 archive for that same real period, at a location the caller supplies — see
that function's docstring for why no default location is silently assumed.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "dswe_inland_offshore"

# From the dataset's own Zenodo record (doi:10.5281/zenodo.5516552) description — real,
# documented date ranges, not derived from the (timestamp-free) CSV rows themselves.
TURBINES = {
    "WT1": {"file": "Inland Wind Farm Dataset1(WT1).csv", "mast": "A", "site": "inland", "period_start": "2010-07-30", "period_end": "2011-07-31"},
    "WT2": {"file": "Inland Wind Farm Dataset1(WT2).csv", "mast": "A", "site": "inland", "period_start": "2010-07-30", "period_end": "2011-07-31"},
    "WT3": {"file": "Inland Wind Farm Dataset1(WT3).csv", "mast": "B", "site": "inland", "period_start": "2010-04-29", "period_end": "2011-04-30"},
    "WT4": {"file": "Inland Wind Farm Dataset1(WT4).csv", "mast": "B", "site": "inland", "period_start": "2010-04-29", "period_end": "2011-04-30"},
    "WT5": {"file": "Offshore Wind Farm Dataset1(WT5).csv", "mast": "C", "site": "offshore", "period_start": "2009-01-01", "period_end": "2009-12-31"},
    "WT6": {"file": "Offshore Wind Farm Dataset1(WT6).csv", "mast": "C", "site": "offshore", "period_start": "2009-01-01", "period_end": "2009-12-31"},
}


def load_turbine(turbine_id: str) -> pd.DataFrame:
    """Real 10-minute mast + turbine data for one turbine: V (m/s), D (deg), air density,
    humidity (offshore only), I (turbulence intensity), S_a/S_b (wind shear, S_a offshore
    only), and y (% of rated power). No timestamp column — see module docstring."""
    if turbine_id not in TURBINES:
        raise ValueError(f"unknown turbine_id {turbine_id!r}, expected one of {list(TURBINES)}")
    path = DATA_DIR / TURBINES[turbine_id]["file"]
    if not path.exists():
        raise FileNotFoundError(f"missing DSWE dataset file: {path} — download from https://zenodo.org/records/5516552")
    return pd.read_csv(path)
