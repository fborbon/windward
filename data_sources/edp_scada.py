"""Loader for EDP Wind Farm A — the "Wind Farm A" subset of the CARE-to-Compare wind turbine
SCADA benchmark (Zenodo doi:10.5281/zenodo.15846963, CC BY-SA 4.0, published alongside Gück
et al. 2024, "CARE to Compare: A Real-World Benchmark Dataset for Early Fault Detection in Wind
Turbine Data", MDPI Data 9(12):138). Wind Farm A is described in that paper as EDP's own
onshore wind farm in Portugal — the original edp.com/en/innovation/data "Wind Farm 1" — but
repackaged and anonymized: turbine identity, exact coordinates, rated power/rotor diameter and
calendar timestamps are all stripped (see the dataset's own README.md), and each of the 22 CSVs
is an independent ~1-year anonymized time window ending in one real, labeled event (a fault or
normal operation) rather than one continuous farm-wide series like Windward's other three farms.

That's why this backs diagnosis/RAG only (see docs/ARCHITECTURE.md), not the forecast
pipeline: there's no real location to join real weather against, and no disclosed rated power/
rotor diameter to compute capacity factor or Cp against. What IS real and usable: real 10-minute
wind speed + grid power SCADA, real turbine operational status codes, and 22 real labeled
case studies (11 with a real root-cause fault description) — genuinely better ground truth for
anomaly detection than the other three farms have, which is the point of adding it.

Download (766MB for just this farm's slice, not committed — see README §5):
    curl -sL -o /tmp/care.zip "https://zenodo.org/api/records/15846963/files/CARE_To_Compare.zip/content"
    unzip "/tmp/care.zip" "CARE_To_Compare/Wind Farm A/*" -d data/edp_wind_farm_a_raw
    mv "data/edp_wind_farm_a_raw/CARE_To_Compare/Wind Farm A/datasets" data/edp_wind_farm_a/
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "edp_wind_farm_a"
DATASETS_DIR = DATA_DIR / "datasets"

# From the dataset's own README.md ("Event Data" section) — real, documented status codes.
STATUS_LABELS = {
    0: "Normal Operation",
    1: "Derated Operation",
    2: "Idling",
    3: "Service",
    4: "Downtime",
    5: "Other",
}

_SERIES_COLUMNS = {
    "time_stamp": "timestamp",
    "asset_id": "turbine_id",
    "wind_speed_3_avg": "wind_speed_ms",
    "power_30_avg": "power_frac",
}


def load_events() -> pd.DataFrame:
    """The real event_info.csv: one row per of the 22 real, labeled case studies — anonymized
    turbine id, anomaly/normal label, real fault description where one was given, and
    anonymized (relative-only, not real calendar dates) start/end timestamps."""
    df = pd.read_csv(DATA_DIR / "event_info.csv", sep=";")
    start = pd.to_datetime(df["event_start"])
    end = pd.to_datetime(df["event_end"])
    df["duration_hours"] = (end - start).dt.total_seconds() / 3600
    return df.set_index("event_id")


def load_event_series(event_id: int) -> pd.DataFrame:
    """One case study's real 10-minute SCADA: timestamp (anonymized offset, not a real
    calendar date), turbine_id, wind_speed_ms (real m/s), power_frac, status_type_id,
    train_test. power_frac ranges roughly -0.01 to 0.98, NOT real kW despite the source
    dataset's feature_description.csv labeling it "kW" — the anonymization rescaled power/
    energy channels (to hide the source farm's real turbine capacity) but left the original
    unit label unchanged, so treat it as a normalized fraction of rated capacity, not an
    absolute power reading. Each file covers a single anonymized turbine's own trajectory
    (see module docstring)."""
    path = DATASETS_DIR / f"{event_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing EDP Wind Farm A event file: {path} — see this module's docstring for the download command")
    df = pd.read_csv(
        path, sep=";",
        usecols=list(_SERIES_COLUMNS) + ["train_test", "status_type_id"],
        parse_dates=["time_stamp"],
    )
    return df.rename(columns=_SERIES_COLUMNS)
