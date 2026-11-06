"""Loader for the SMARTEOLE wind farm control open dataset (Zenodo doi:10.5281/zenodo.7342466,
Etalab Open Licence 2.0): Sole du Moulin Vieux, Picardy, France, 7 Senvion MM82 (2050 kW, 82 m
rotor, 80 m hub) in a single north-south line about 350 m (4.3 D) apart. 17 February to 25 May
2020, the last field campaign of the French SMARTEOLE project, which tested wake steering:
turbine SMV6 was yawed off the wind on purpose, toggling the offset on and off, to bend its
wake away from SMV5 just north of it (Simley et al. 2021, Wind Energ. Sci. 6, 1427,
https://wes.copernicus.org/articles/6/1427/2021/).

What makes it different from the other farms:
- 1-minute data, pre-processed by the data owners: every minute in which any turbine was
  stopped was removed, and nacelle position/direction were corrected for north alignment. So
  there are no downtime periods to find, and hourly farm output is "all seven running".
- The control log from SMV6 (applied yaw offset, whether it was active) comes with it, which is
  what makes the wake-steering comparison in operational_assessment/smarteole_wake_steering.py
  possible.
- A per-turbine derating flag (0/1, "suspected derating, conservative estimate"), no status log.
- Three months only: the shortest record in Windward.

The 1-minute CSVs are 220 MB (SCADA) and 40 MB (control log), so convert() aggregates them once
to 10 minutes (data/smarteole/smarteole_10min.csv.gz, committed); a 10-minute row needs at
least 8 of its 10 minutes.

    python -m data_sources.smarteole_scada
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "smarteole"
RAW_DIR = DATA_DIR / "SMARTEOLE-WFC-open-dataset"
CSV_PATH = DATA_DIR / "smarteole_10min.csv.gz"
STATIC_PATH = DATA_DIR / "SMARTEOLE_WakeSteering_Coordinates_staticData.csv"
TURBINES = 7
RATED_KW = 2050.0
ROTOR_D_M = 82.0
HUB_M = 80.0
STEERED, WAKED = 6, 5  # SMV6 is yawed; SMV5 sits in its wake in southerly wind

_EMPTY_EVENTS = ["turbine_id", "start", "end", "duration", "status", "code", "message", "iec_category"]


def turbine_id(n: int) -> str:
    return f"SMV{n}"


def convert(out_path: Path = CSV_PATH) -> pd.DataFrame:
    """1-minute SCADA + SMV6 control log -> 10-minute means. Per turbine: power (kW), nacelle wind
    speed, wind direction, derate share; SMV6: applied yaw offset and the share of the 10 minutes
    the offset was active."""
    keep = ["time"]
    for i in range(1, TURBINES + 1):
        keep += [f"active_power_{i}_avg", f"wind_speed_{i}_avg", f"wind_direction_{i}_avg", f"derate_{i}"]
    scada = pd.read_csv(RAW_DIR / "SMARTEOLE_WakeSteering_SCADA_1minData.csv", usecols=keep, parse_dates=["time"])
    log = pd.read_csv(RAW_DIR / "SMARTEOLE_WakeSteering_ControlLog_1minData.csv",
                      usecols=["time", "control_log_offset_avg", "control_log_offset_active_avg"], parse_dates=["time"])
    df = scada.merge(log, on="time", how="left").set_index("time")
    rename = {"control_log_offset_avg": "offset_deg_6", "control_log_offset_active_avg": "offset_active_6"}
    for i in range(1, TURBINES + 1):
        rename |= {f"active_power_{i}_avg": f"p{i}", f"wind_speed_{i}_avg": f"w{i}", f"derate_{i}": f"derate{i}"}
    df = df.rename(columns=rename)
    # wind direction: average as a vector, not arithmetically (359 and 1 must give 0)
    import numpy as np
    rad = {i: np.deg2rad(df.pop(f"wind_direction_{i}_avg")) for i in range(1, TURBINES + 1)}
    for i, r in rad.items():
        df[f"s{i}"], df[f"c{i}"] = np.sin(r), np.cos(r)
    g = df.resample("10min")
    out = g.mean()
    out = out[g.size() >= 8]
    for i in range(1, TURBINES + 1):
        out[f"d{i}"] = (np.rad2deg(np.arctan2(out.pop(f"s{i}"), out.pop(f"c{i}"))) % 360).round(1)
    out = out.round(3).reset_index().rename(columns={"time": "timestamp"})
    out.to_csv(out_path, index=False, compression="gzip")
    return out


_cache: dict[str, pd.DataFrame] = {}


def load_10min(path: Path = CSV_PATH) -> pd.DataFrame:
    key = str(path)
    if key not in _cache:
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} missing - run `python -m data_sources.smarteole_scada` (needs the Zenodo zip unpacked in {RAW_DIR})")
        _cache[key] = pd.read_csv(path, parse_dates=["timestamp"])
    return _cache[key].copy()


def long_10min(zip_paths: list[Path] | None = None) -> pd.DataFrame:
    """timestamp, turbine_id, power_kw, wind_speed_ms (nacelle), wind_direction_deg, derate."""
    df = pd.concat([load_10min(p) for p in (zip_paths or [CSV_PATH])], ignore_index=True)
    frames = []
    for i in range(1, TURBINES + 1):
        part = df[["timestamp", f"p{i}", f"w{i}", f"d{i}", f"derate{i}"]].rename(
            columns={f"p{i}": "power_kw", f"w{i}": "wind_speed_ms", f"d{i}": "wind_direction_deg", f"derate{i}": "derate"})
        part["turbine_id"] = turbine_id(i)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def load_turbine_static(farm_id: str) -> pd.DataFrame:
    coords = pd.read_csv(STATIC_PATH)
    coords = coords[coords["Turbine"].str.startswith("SMV")]
    return pd.DataFrame([{"turbine_id": r.Turbine, "Title": f"Sole du Moulin Vieux {r.Turbine}", "Manufacturer": "Senvion",
                          "Model": "MM82", "Rated power (kW)": RATED_KW, "Hub Height (m)": HUB_M,
                          "Rotor Diameter (m)": ROTOR_D_M, "Latitude": r.Latitude, "Longitude": r.Longitude,
                          "Commercial Operations Date": None} for r in coords.itertuples()]).set_index("turbine_id")


def load_farm_hourly_production(zip_paths: list[Path]) -> pd.DataFrame:
    """Hourly output_mw = sum of the 7 turbines. The owners removed every minute with a stopped
    turbine, so every remaining row has all seven; an hour needs at least 4 of its 6 ten-minute
    rows to count."""
    df = pd.concat([load_10min(p) for p in zip_paths], ignore_index=True).set_index("timestamp")
    farm_kw = df[[f"p{i}" for i in range(1, TURBINES + 1)]].clip(lower=0).sum(axis=1)
    g = farm_kw.resample("1h")
    hourly = g.mean()[g.count() >= 4]
    return (hourly / 1000.0).rename("output_mw").to_frame()


def load_status_events(zip_paths: list[Path]) -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_EVENTS)


def load_turbine_hourly_series(zip_paths: list[Path]) -> pd.DataFrame:
    long = long_10min(zip_paths).dropna(subset=["power_kw", "wind_speed_ms"])
    hourly = (long.set_index("timestamp").groupby("turbine_id")[["power_kw", "wind_speed_ms"]]
              .resample("1h").mean().reset_index())
    hourly["power_kw"] = hourly["power_kw"].clip(lower=0)
    return hourly.dropna(subset=["wind_speed_ms", "power_kw"])[["turbine_id", "timestamp", "wind_speed_ms", "power_kw"]]


def scada_10min_for_records() -> pd.DataFrame:
    long = long_10min().dropna(subset=["power_kw"])
    return long.rename(columns={"timestamp": "time", "turbine_id": "asset_id", "power_kw": "WTUR_W"})[["time", "asset_id", "WTUR_W"]]


if __name__ == "__main__":
    out = convert()
    print(f"wrote {CSV_PATH} ({CSV_PATH.stat().st_size / 1e6:.1f} MB), {len(out)} rows, "
          f"{out['timestamp'].min()} to {out['timestamp'].max()}")
