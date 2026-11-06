"""Loader for the Delabole open SCADA dataset (DTU Data, doi:10.11583/DTU.14077004): the UK's
first commercial wind farm, 10 Winddane WD34 400 kW turbines at Deli Farm, North Cornwall,
commissioned December 1991. Measured May 1993 to April 1994 for an ETSU/DTI performance study,
10-minute statistics, about 6,800 hours.

The source is one NetCDF4 file (delabole_all.nc) holding turbine active power poa01..poa10 plus
two met masts (mast 1: cups at 10/20/33/44 m, vane at 44 m, temperature; mast 2: cups at 10/33 m).
It differs from the other farms in three ways the rest of the pipeline has to know about:

- No nacelle anemometers. Every turbine gets mast 1's 33 m cup speed (hub height is 32 m), so
  per-turbine power curves share one free-stream wind reading. That makes the farm a clean
  wake/performance comparison (no per-turbine anemometer bias), but a turbine-level anemometer
  calibration finding can't happen here.
- No alarm/status log. load_status_events() returns an empty frame with the shared schema, and
  Farm.has_status_log is False so the agent tools say "not in this dataset" instead of zero.
- Stall-regulated 1991 turbines: 10-minute power reaches about 495 kW on a 400 kW rating, which
  is real overshoot in gusty stall operation, not a unit error.

The NetCDF needs h5py, which the slim API image doesn't carry, so convert() turns it once into
data/delabole/delabole_10min.csv.gz (about 1.5 MB, committed) and the loaders read that:

    pip install h5py && python -m data_sources.delabole_scada   # rebuilds the CSV from the .nc

Licence note: the DTU Data record says CC BY 4.0, while the NetCDF's own `license` attribute says
CC BY NC 4.0. Attribute DTU Data / ETSU either way, and check before any commercial use.
"""
import math
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "delabole"
NETCDF_PATH = DATA_DIR / "delabole_all.nc"
CSV_PATH = DATA_DIR / "delabole_10min.csv.gz"
TURBINE_COUNT = 10
WIND_SPEED_CHANNEL = "ws33_m1"  # mast 1, 33 m: closest cup to the 32 m hub

_EMPTY_EVENTS = ["turbine_id", "start", "end", "duration", "status", "code", "message", "iec_category"]


def turbine_id(n: int) -> str:
    return f"Delabole_{n:02d}"


def convert(nc_path: Path = NETCDF_PATH, out_path: Path = CSV_PATH) -> pd.DataFrame:
    """NetCDF4 -> compact CSV. Times are 'minutes since 1993-05-01 00:00:00', taken as UTC
    (the source doesn't say GMT or BST). Quality flags are all 0 ('not checked'), so nothing
    is dropped here; analysis/qc.py does the screening downstream like for every other farm."""
    import h5py  # offline-only dependency, see the module docstring

    with h5py.File(nc_path, "r") as f:
        minutes = f["time"][:]
        cols = {f"poa{i:02d}": f[f"poa{i:02d}"][:] for i in range(1, TURBINE_COUNT + 1)}
        for name in ("ws44_m1", "ws33_m1", "ws33_m2", "wd44_m1", "ta_05"):
            cols[name] = f[name][:]
    df = pd.DataFrame(cols)
    df.insert(0, "timestamp", pd.Timestamp("1993-05-01") + pd.to_timedelta(minutes, unit="min"))
    num = df.columns[1:]
    df[num] = df[num].round(2)
    df.to_csv(out_path, index=False, compression="gzip")
    return df


_cache: dict[str, pd.DataFrame] = {}


def load_10min(path: Path = CSV_PATH) -> pd.DataFrame:
    """The converted 10-minute table: timestamp, poa01..poa10 (kW), mast channels."""
    key = str(path)
    if key not in _cache:
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run `python -m data_sources.delabole_scada` (needs h5py and {NETCDF_PATH.name})")
        _cache[key] = pd.read_csv(path, parse_dates=["timestamp"])
    return _cache[key].copy()


def _long_10min(zip_paths: list[Path]) -> pd.DataFrame:
    """timestamp, turbine_id, power_kw, wind_speed_ms (mast 1, 33 m), 10-minute rows."""
    df = pd.concat([load_10min(p) for p in zip_paths], ignore_index=True)
    power_cols = [f"poa{i:02d}" for i in range(1, TURBINE_COUNT + 1)]
    long = df.melt(id_vars=["timestamp", WIND_SPEED_CHANNEL], value_vars=power_cols,
                   var_name="channel", value_name="power_kw")
    long["turbine_id"] = long["channel"].str[3:].astype(int).map(turbine_id)
    return long.rename(columns={WIND_SPEED_CHANNEL: "wind_speed_ms"}).drop(columns="channel")


def load_turbine_static(farm_id: str) -> pd.DataFrame:
    """Per-turbine metadata, Kelmarsh_WT_static.csv's layout. Coordinates are derived from the
    DTU site documentation's layout offsets in metres from mast 1 (50.630492 N, 4.708475 W),
    taking x as east and y as north: good to tens of metres, which is all the nearest-neighbour
    ranking in analysis/efficiency.py needs. The source gives the commissioning month only."""
    df = pd.read_csv(DATA_DIR / "Delabole_WT_static.csv")
    df["turbine_id"] = df["Alternative Title"].str[2:].astype(int).map(turbine_id)
    return df.set_index("turbine_id")


def load_farm_hourly_production(zip_paths: list[Path]) -> pd.DataFrame:
    """Hourly output_mw, summed across turbines. An hour needs most of its 10-minute samples
    from every reporting turbine: about 40% of turbine-samples are missing in this campaign, and
    summing only the turbines that happened to report would read as low-output hours. Missing
    turbines are filled with the reporting turbines' mean in that hour, the same scaling a
    farm-level availability correction does, and hours where fewer than 7 of the 10 report
    are dropped."""
    long = _long_10min(zip_paths)
    hourly = (long.set_index("timestamp").groupby("turbine_id")["power_kw"]
              .resample("1h").mean().unstack("turbine_id").clip(lower=0))
    reporting = hourly.notna().sum(axis=1)
    hourly = hourly[reporting >= 7]
    farm_kw = hourly.mean(axis=1) * TURBINE_COUNT
    return (farm_kw / 1000.0).rename("output_mw").to_frame()


def load_status_events(zip_paths: list[Path]) -> pd.DataFrame:
    """The Delabole export has no alarm or status log: an empty frame with the shared schema."""
    return pd.DataFrame(columns=_EMPTY_EVENTS)


def load_turbine_hourly_series(zip_paths: list[Path]) -> pd.DataFrame:
    """Long-format hourly: turbine_id, timestamp, wind_speed_ms (mast 1, 33 m), power_kw."""
    long = _long_10min(zip_paths).dropna(subset=["power_kw", "wind_speed_ms"])
    hourly = (long.set_index("timestamp").groupby("turbine_id")[["power_kw", "wind_speed_ms"]]
              .resample("1h").mean().reset_index())
    hourly["power_kw"] = hourly["power_kw"].clip(lower=0)
    return hourly.dropna(subset=["wind_speed_ms", "power_kw"])[["turbine_id", "timestamp", "wind_speed_ms", "power_kw"]]


def scada_10min_for_records() -> pd.DataFrame:
    """10-minute rows in the OpenOA-style shape data_sources/farm_facts.py reads from
    data/operational_cache for the other farms (time, asset_id, WTUR_W)."""
    long = _long_10min([CSV_PATH]).dropna(subset=["power_kw"])
    return long.rename(columns={"timestamp": "time", "turbine_id": "asset_id", "power_kw": "WTUR_W"})[["time", "asset_id", "WTUR_W"]]


if __name__ == "__main__":
    out = convert()
    print(f"wrote {CSV_PATH} ({CSV_PATH.stat().st_size / 1e6:.1f} MB), {len(out)} rows, "
          f"{out['timestamp'].min()} to {out['timestamp'].max()}")
    assert not math.isnan(out["poa01"].max())
