"""CARE to Compare (Zenodo doi:10.5281/zenodo.15846963, CC BY-SA 4.0; Gück et al. 2024,
doi:10.3390/data9120138): 95 labelled early-fault-detection cases from three wind farms.

- Wind Farm A: EDP's onshore farm in Portugal, 5 turbines, 22 cases, 86 columns.
- Wind Farm B: an offshore farm in Germany, 9 turbines, 15 cases, 257 columns.
- Wind Farm C: an offshore farm in Germany, 22 turbines, 58 cases, 957 columns.

Each case is one turbine's anonymised 10-minute SCADA: about a year of "train" rows, then a
"prediction" window that holds exactly one labelled event, either an anomaly leading up to a
fault (often with an operator-written root cause) or normal behaviour. Timestamps are shifted
per file and power/energy values are rescaled, so none of it can feed the forecasting pipeline;
what it is good for is fault detection with ground truth (analysis/care_nbm.py).

The archive unpacks to about 20 GB, so nothing here needs it unpacked: each case is read
straight out of the zip, one at a time, by the offline precompute (analysis/care_nbm.py). The
API only reads the small committed payload that precompute writes.

    mkdir -p data/care_raw && curl -sL -o data/care_raw/CARE_To_Compare.zip \\
        "https://zenodo.org/records/15846963/files/CARE_To_Compare.zip?download=1"     # 5.5 GB
"""
import zipfile
from functools import lru_cache
from pathlib import Path

import pandas as pd

ZIP_PATH = Path(__file__).resolve().parent.parent / "data" / "care_raw" / "CARE_To_Compare.zip"

STATUS_LABELS = {0: "Normal Operation", 1: "Derated Operation", 2: "Idling", 3: "Service", 4: "Downtime", 5: "Other"}

FARMS = {
    "A": {"name": "Wind Farm A", "where": "onshore, Portugal (EDP)", "turbines": 5,
          "wind": "wind_speed_3_avg", "power": "power_30_avg",
          # For farm A the status labels come from the failure log and are only meant for
          # filtering training data, not for scoring (CARE2Compare FAQ).
          "status_at_prediction": False},
    "B": {"name": "Wind Farm B", "where": "offshore, Germany", "turbines": 9,
          "wind": "wind_speed_61_avg", "power": "power_62_avg", "status_at_prediction": True},
    "C": {"name": "Wind Farm C", "where": "offshore, Germany", "turbines": 22,
          "wind": "wind_speed_236_avg", "power": "power_6_avg", "status_at_prediction": True},
}
META_COLUMNS = ["time_stamp", "asset_id", "id", "train_test", "status_type_id"]


def _zip() -> zipfile.ZipFile:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"{ZIP_PATH} missing - see this module's docstring for the download command")
    return zipfile.ZipFile(ZIP_PATH)


def _member(farm: str, name: str) -> str:
    return f"CARE_To_Compare/Wind Farm {farm}/{name}"


@lru_cache(maxsize=None)
def load_events(farm: str) -> pd.DataFrame:
    with _zip() as z:
        df = pd.read_csv(z.open(_member(farm, "event_info.csv")), sep=";")
    df = df.rename(columns={"asset": "asset_id"})
    df["farm"] = farm
    return df.set_index("event_id")


@lru_cache(maxsize=None)
def load_feature_description(farm: str) -> pd.DataFrame:
    with _zip() as z:
        fd = pd.read_csv(z.open(_member(farm, "feature_description.csv")), sep=";", encoding="utf-8", encoding_errors="replace")
    # the source files already carry U+FFFD where a degree sign was (units "°C", "°")
    for c in ("unit", "description"):
        fd[c] = fd[c].astype(str).str.replace("\ufffd", "°", regex=False)
    return fd


def load_case(farm: str, event_id: int) -> pd.DataFrame:
    """One case's full table (train and prediction rows), float32 sensor columns."""
    with _zip() as z:
        df = pd.read_csv(z.open(_member(farm, f"datasets/{event_id}.csv")), sep=";")
    num = [c for c in df.columns if c not in META_COLUMNS]
    df[num] = df[num].astype("float32")
    return df


def sensor_columns(farm: str, columns: list[str]) -> dict[str, list[str]]:
    """The 10-minute averages to model (min/max/std are unreliable, especially in farm B, per
    the dataset's FAQ), split into plain values and angles (encoded as sin/cos downstream).
    Counters (cumulative energy totals) are left out: they only ever grow."""
    fd = load_feature_description(farm)
    is_angle = dict(zip(fd["sensor_name"], fd["is_angle"].astype(bool)))
    is_counter = dict(zip(fd["sensor_name"], fd["is_counter"].astype(bool)))
    plain, angles = [], []
    for c in columns:
        if c in META_COLUMNS:
            continue
        base = c[:-4] if c.endswith("_avg") else c
        if c.endswith(("_min", "_max", "_std")) or base not in is_angle or is_counter.get(base):
            continue
        (angles if is_angle[base] else plain).append(c)
    return {"plain": plain, "angles": angles}
