"""Generic loader for Greenbyte-exported wind farm SCADA data (the export format used by
both the Kelmarsh and Penmanshiel open datasets — Cubico Sustainable Investments, CC BY 4.0,
Zenodo). Farm-level production is used as the training TARGET; per-turbine wind speed +
power (load_turbine_hourly_series) is used for power-curve / efficiency analysis.
"""
import re
import zipfile
from pathlib import Path

import pandas as pd

HEADER_ROW = 9  # 0-indexed line of the "# Date and time,..." row
COL_TIMESTAMP = 0
COL_WIND_SPEED = 1
COL_POWER = 61  # "Power (kW)" — same position across both datasets (verified against each's dataSignalMapping file)


def load_turbine_static(farm_id: str) -> pd.DataFrame:
    """Per-turbine static metadata (manufacturer, model, hub height, exact coordinates,
    commercial-ops date) from the dataset's own `*_WT_static.csv`, indexed by turbine_id
    in the same 'Farmname_N' shape the SCADA loaders use."""
    data_dir = Path(__file__).resolve().parent.parent / "data" / farm_id
    matches = list(data_dir.glob("*_WT_static.csv"))
    if not matches:
        raise FileNotFoundError(f"no *_WT_static.csv found under {data_dir}")
    df = pd.read_csv(matches[0])
    df["turbine_id"] = df["Title"].str.replace(" ", "_", regex=False)
    return df.set_index("turbine_id")


def _turbine_member_names(zf: zipfile.ZipFile) -> list[str]:
    return sorted(n for n in zf.namelist() if n.startswith("Turbine_Data_"))


def _status_member_names(zf: zipfile.ZipFile) -> list[str]:
    return sorted(n for n in zf.namelist() if n.startswith("Status_"))


def _read_turbine_csv(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with zf.open(member) as f:
        df = pd.read_csv(
            f,
            skiprows=HEADER_ROW,
            header=0,
            usecols=[COL_TIMESTAMP, COL_WIND_SPEED, COL_POWER],
            names=["timestamp", "wind_speed_ms", "power_kw"],
            parse_dates=["timestamp"],
        )
    return df.set_index("timestamp").sort_index()


def load_farm_hourly_production(zip_paths: list[Path]) -> pd.DataFrame:
    """Hourly DataFrame indexed by timestamp with one column, output_mw — summed across
    all turbines and all zips (zips may split turbines into groups, as Penmanshiel does)."""
    per_turbine_hourly = []
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zf:
            for member in _turbine_member_names(zf):
                df = _read_turbine_csv(zf, member)
                hourly = df["power_kw"].resample("1h").mean().clip(lower=0) / 1000.0  # kW -> MW
                turbine_id = re.search(r"Turbine_Data_([A-Za-z]+_\d+)", member).group(1)
                per_turbine_hourly.append(hourly.rename(turbine_id))

    wide = pd.concat(per_turbine_hourly, axis=1)
    farm = wide.sum(axis=1, min_count=1).rename("output_mw").dropna()
    return farm.to_frame()


def load_status_events(zip_paths: list[Path]) -> pd.DataFrame:
    """Real turbine status/fault event log: turbine_id, start, end, duration, status,
    code, message, iec_category. Source for the incident RAG corpus / semantic search."""
    frames = []
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zf:
            for member in _status_member_names(zf):
                with zf.open(member) as f:
                    df = pd.read_csv(
                        f, skiprows=HEADER_ROW, header=0,
                        parse_dates=["Timestamp start", "Timestamp end"], date_format="%Y-%m-%d %H:%M:%S",
                    )
                df.columns = [c.strip() for c in df.columns]
                df["turbine_id"] = re.search(r"Status_([A-Za-z]+_\d+)", member).group(1)
                frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={
        "Timestamp start": "start", "Timestamp end": "end", "Duration": "duration",
        "Status": "status", "Code": "code", "Message": "message", "IEC category": "iec_category",
    })
    return out[["turbine_id", "start", "end", "duration", "status", "code", "message", "iec_category"]]


def load_turbine_hourly_series(zip_paths: list[Path]) -> pd.DataFrame:
    """Long-format hourly DataFrame: columns turbine_id, timestamp, wind_speed_ms, power_kw."""
    frames = []
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zf:
            for member in _turbine_member_names(zf):
                df = _read_turbine_csv(zf, member)
                hourly = df.resample("1h").mean()
                hourly["power_kw"] = hourly["power_kw"].clip(lower=0)
                hourly["turbine_id"] = re.search(r"Turbine_Data_([A-Za-z]+_\d+)", member).group(1)
                frames.append(hourly.reset_index())
    return pd.concat(frames, ignore_index=True).dropna(subset=["wind_speed_ms", "power_kw"])
