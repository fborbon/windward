"""Loader for the Hill of Towie open SCADA dataset (RES, on behalf of TRIG, CC BY 4.0,
Zenodo doi:10.5281/zenodo.14870023) — a different export shape from the Greenbyte one used
by Kelmarsh/Penmanshiel (data_sources/greenbyte_scada.py). RES's historian exports one CSV
per signal-group table per month, covering all 21 turbines together (columns TimeStamp,
StationId, wtc_*), instead of one CSV per turbine covering the whole period. Same output
schema as greenbyte_scada.py's functions, so the rest of the pipeline doesn't need to know
which farm it's looking at — see data_sources/farms.py's loader_for().
"""
import zipfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "hill_of_towie"

_alarm_descriptions_cache: pd.DataFrame | None = None


def _alarm_descriptions() -> pd.DataFrame:
    global _alarm_descriptions_cache
    if _alarm_descriptions_cache is None:
        df = pd.read_csv(DATA_DIR / "Hill_of_Towie_alarms_description.csv")
        _alarm_descriptions_cache = df.set_index("Alarm Code")
    return _alarm_descriptions_cache


def load_turbine_static(farm_id: str) -> pd.DataFrame:
    """Per-turbine static metadata straight from the dataset's own turbine_metadata.csv
    (coordinates, manufacturer, model, rated power, rotor diameter, hub height, commercial-ops
    date), indexed by turbine_id in the same 'HillOfTowie_NN' shape the SCADA loaders use."""
    df = pd.read_csv(DATA_DIR / "Hill_of_Towie_turbine_metadata.csv")
    df["turbine_id"] = "HillOfTowie_" + df["Turbine Name"].str.extract(r"T(\d+)")[0]
    return df.set_index("turbine_id")


def _station_to_turbine() -> dict:
    static = load_turbine_static("hill_of_towie")
    return {row["Station ID"]: turbine_id for turbine_id, row in static.iterrows()}


def _table_members(zf: zipfile.ZipFile, table: str) -> list[str]:
    return sorted(n for n in zf.namelist() if n.startswith(f"{table}_"))


def _read_table(zip_paths: list[Path], table: str, usecols: list[str]) -> pd.DataFrame:
    """A handful of monthly files (e.g. tblSCTurbine_2024_07.csv) mix 'YYYY-MM-DD HH:MM:SS'
    rows with a few bare 'YYYY-MM-DD' rows at a midnight boundary — pandas then infers TimeStamp
    as object dtype instead of datetime64 for that one file, which breaks concat/merge against
    the other months. Parsing as plain strings and coercing with pd.to_datetime afterwards
    (rather than read_csv's parse_dates) handles the mixed formats consistently."""
    frames = []
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zf:
            for member in _table_members(zf, table):
                with zf.open(member) as f:
                    frames.append(pd.read_csv(f, usecols=usecols))
    df = pd.concat(frames, ignore_index=True)
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], format="mixed")
    return df


def load_farm_hourly_production(zip_paths: list[Path]) -> pd.DataFrame:
    """Hourly DataFrame indexed by timestamp with one column, output_mw — summed across all
    21 turbines' active power (tblSCTurGrid.wtc_ActPower_mean, kW)."""
    grid = _read_table(zip_paths, "tblSCTurGrid", ["TimeStamp", "StationId", "wtc_ActPower_mean"])
    wide = grid.pivot_table(index="TimeStamp", columns="StationId", values="wtc_ActPower_mean")
    hourly = wide.resample("1h").mean().clip(lower=0) / 1000.0  # kW -> MW
    farm = hourly.sum(axis=1, min_count=1).rename("output_mw").dropna()
    return farm.to_frame()


def load_status_events(zip_paths: list[Path]) -> pd.DataFrame:
    """Real turbine alarm log, reshaped into the same schema as greenbyte_scada's
    load_status_events (turbine_id, start, end, duration, status, code, message,
    iec_category) even though the underlying RES export (TimeOn/TimeOff/StationNr/Alarmcode
    plus a separate alarm-code lookup table) looks nothing like Greenbyte's per-turbine
    Status_*.csv files. `status` is derived from the alarm-code lookup's 'Stopping' flag
    (1 -> Stop, 0 -> Warning) since RES doesn't label events that way directly."""
    descriptions = _alarm_descriptions()
    station_to_turbine = _station_to_turbine()

    frames = []
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zf:
            for member in _table_members(zf, "tblAlarmLog"):
                with zf.open(member) as f:
                    frames.append(pd.read_csv(f))
    events = pd.concat(frames, ignore_index=True)
    events["TimeOn"] = pd.to_datetime(events["TimeOn"], format="mixed")  # mixed date formats across months, see _read_table
    events["TimeOff"] = pd.to_datetime(events["TimeOff"], format="mixed")

    events["turbine_id"] = events["StationNr"].map(station_to_turbine)
    duration_td = events["TimeOff"] - events["TimeOn"]
    events["duration"] = duration_td.apply(
        lambda td: f"{int(td.total_seconds() // 3600)}:{int(td.total_seconds() % 3600 // 60):02d}:{int(td.total_seconds() % 60):02d}"
        if pd.notna(td) else ""
    )
    stopping = events["Alarmcode"].map(descriptions["Stopping"])
    events["status"] = stopping.map({1: "Stop", 0: "Warning"}).fillna("Warning")
    events["message"] = events["Alarmcode"].map(descriptions["Description"]).fillna("")
    events["iec_category"] = ""

    out = events.rename(columns={"TimeOn": "start", "TimeOff": "end", "Alarmcode": "code"})
    return out[["turbine_id", "start", "end", "duration", "status", "code", "message", "iec_category"]]


def load_turbine_hourly_series(zip_paths: list[Path]) -> pd.DataFrame:
    """Long-format hourly DataFrame: columns turbine_id, timestamp, wind_speed_ms, power_kw."""
    station_to_turbine = _station_to_turbine()
    grid = _read_table(zip_paths, "tblSCTurGrid", ["TimeStamp", "StationId", "wtc_ActPower_mean"])
    wind = _read_table(zip_paths, "tblSCTurbine", ["TimeStamp", "StationId", "wtc_PrWindSp_mean"])

    grid = grid.rename(columns={"wtc_ActPower_mean": "power_kw"})
    wind = wind.rename(columns={"wtc_PrWindSp_mean": "wind_speed_ms"})
    merged = pd.merge(grid, wind, on=["TimeStamp", "StationId"])
    merged["turbine_id"] = merged["StationId"].map(station_to_turbine)
    merged = merged.set_index("TimeStamp")

    hourly = (
        merged.groupby("turbine_id")
        .resample("1h")[["power_kw", "wind_speed_ms"]]
        .mean()
        .reset_index()
        .rename(columns={"TimeStamp": "timestamp"})
    )
    hourly["power_kw"] = hourly["power_kw"].clip(lower=0)
    return hourly.dropna(subset=["wind_speed_ms", "power_kw"])
