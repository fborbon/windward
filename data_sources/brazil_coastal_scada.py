"""Loader for two Brazilian coastal wind farms (Zenodo doi:10.5281/zenodo.1475197, CC BY-SA 4.0):
a micrometeorological campaign by the Federal University of Santa Catarina, August 2013 to July
2014, 10-minute data. Trade winds and the sea breeze dominate both sites.

- Pedra do Sal (UEPS), Parnaíba, Piauí: 20 Enercon E-44 (900 kW, 44 m rotor, 55 m hub), 18 MW.
- Beberibe (UEBB), Ceará: 32 Enercon E-48 (800 kW, 48 m rotor, 75 m hub), 25.6 MW.

Each NetCDF has every turbine's 10-minute active power and nacelle wind speed (a 2D sonic on the
nacelle) plus a 100 m IEC met mast with cups and vanes at five or six heights. Things to know:

- The files' metadata was copied between the two farms: both are titled "UEBB Dataset", both
  describe "Enercon E-48 at 75 m", and power is labelled "rpm". The Zenodo record is right
  (E-44 at 55 m for Pedra do Sal), and power is kW (Pedra do Sal tops out at 930, Beberibe 840).
- No turbine coordinates. The NetCDF points to "reference material" that isn't in the record,
  so positions come from OpenStreetMap (© OpenStreetMap contributors, ODbL): the 20- and
  32-turbine clusters at the two sites, saved in *_osm_turbines.csv. OSM doesn't say which
  turbine is which number, so infer_layout() works it out from the data: both farms are a single
  NW-SE line, neighbouring turbines' short-term power fluctuations correlate most (median 0.71 vs
  0.50 for any pair at Pedra do Sal), and in south-east wind along the line the turbines that
  produce most are the high numbers, so the numbering runs NW to SE. The positions are an
  inference, good for nearest-neighbour comparisons, not survey grade.
- No alarm/status log.

The NetCDF needs h5py, so convert() writes one compact CSV per farm (committed):

    pip install h5py && python -m data_sources.brazil_coastal_scada
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "brazil_coastal"

SITES = {
    "pedra_do_sal": {"code": "UEPS", "prefix": "PedraDoSal", "name": "Pedra do Sal", "turbines": 20, "rated_kw": 900.0, "rotor_m": 44.0,
                     "hub_m": 55.0, "model": "E-44", "mast_ws_height": 60.0},
    "beberibe": {"code": "UEBB", "prefix": "Beberibe", "name": "Beberibe", "turbines": 32, "rated_kw": 800.0, "rotor_m": 48.0,
                 "hub_m": 75.0, "model": "E-48", "mast_ws_height": 80.0},
}
MAST_WD_HEIGHT = 100.0

_EMPTY_EVENTS = ["turbine_id", "start", "end", "duration", "status", "code", "message", "iec_category"]


def csv_path(farm_id: str) -> Path:
    return DATA_DIR / f"{farm_id}_10min.csv.gz"


def netcdf_path(farm_id: str) -> Path:
    return DATA_DIR / f"{SITES[farm_id]['code']}_v1.nc"


def static_path(farm_id: str) -> Path:
    return DATA_DIR / f"{farm_id}_turbine_layout.csv"


def turbine_id(farm_id: str, n: int) -> str:
    return f"{SITES[farm_id]['prefix']}_{n:02d}"


def _farm_of(path: Path) -> str:
    return next(f for f in SITES if Path(path).name.startswith(f))


def convert(farm_id: str) -> pd.DataFrame:
    """NetCDF4 -> CSV: timestamp (UTC), p01.. (kW), w01.. (nacelle m/s), mast wind speed near hub
    height and mast direction at 100 m."""
    import h5py  # offline-only dependency, see the module docstring

    site = SITES[farm_id]
    with h5py.File(netcdf_path(farm_id), "r") as f:
        heights = list(f["Height"][:])
        t = pd.to_datetime(f["Time"][:], unit="s").round("min")
        power, nacelle = f["active_power_total"][:], f["wind_speed_nacelle"][:]
        df = pd.DataFrame({"timestamp": t})
        for i in range(site["turbines"]):
            df[f"p{i + 1:02d}"] = power[i]
            df[f"w{i + 1:02d}"] = nacelle[i]
        df["mast_ws"] = f["wind_speed"][heights.index(site["mast_ws_height"])]
        df["mast_wd"] = f["wind_direction"][heights.index(MAST_WD_HEIGHT)]
    num = df.columns[1:]
    df[num] = df[num].round(1)
    df.to_csv(csv_path(farm_id), index=False, compression="gzip")
    return df


def infer_layout(farm_id: str) -> pd.DataFrame:
    """Match the dataset's turbine numbers to the OSM positions (see the module docstring) and
    write {farm_id}_turbine_layout.csv. Steps: order the OSM turbines along the line's principal
    axis from NW to SE; order the dataset's turbines by spectral seriation of the correlation of
    their high-passed 10-minute power (neighbours share gusts); orient that order so low numbers
    come first; assign position k to the k-th turbine in that order."""
    site = SITES[farm_id]
    osm = pd.read_csv(DATA_DIR / f"{farm_id}_osm_turbines.csv")
    lat0, lon0 = osm["latitude"].mean(), osm["longitude"].mean()
    x = (osm["longitude"] - lon0) * 111320 * math.cos(math.radians(lat0))
    y = (osm["latitude"] - lat0) * 110574
    xy = np.c_[x - x.mean(), y - y.mean()]
    axis = np.linalg.svd(xy, full_matrices=False)[2][0]
    if axis[0] < 0:  # point the axis east-ish, so increasing projection runs NW -> SE on these lines
        axis = -axis
    osm = osm.assign(along_m=xy @ axis).sort_values("along_m").reset_index(drop=True)

    p = load_10min(csv_path(farm_id))[[f"p{i:02d}" for i in range(1, site["turbines"] + 1)]]
    hp = (p - p.rolling(6, center=True).mean()).dropna()
    c = np.clip(hp.corr().values, 0, None)
    np.fill_diagonal(c, 0)
    lap = np.diag(c.sum(1)) - c
    fiedler = np.linalg.eigh(lap)[1][:, 1]
    order = np.argsort(fiedler) + 1
    if np.corrcoef(order, np.arange(len(order)))[0, 1] < 0:
        order = order[::-1]
    rows = []
    for k, n in enumerate(order):
        rows.append({"turbine_id": turbine_id(farm_id, int(n)), "Title": f"{site['name']} {int(n):02d}",
                     "Manufacturer": "Enercon", "Model": site["model"], "Rated power (kW)": site["rated_kw"],
                     "Hub Height (m)": site["hub_m"], "Rotor Diameter (m)": site["rotor_m"],
                     "Latitude": osm.loc[k, "latitude"], "Longitude": osm.loc[k, "longitude"],
                     "Position along line (m, NW to SE)": round(float(osm.loc[k, "along_m"]), 0),
                     "OSM node": int(osm.loc[k, "osm_node_id"]),
                     "Commercial Operations Date": None})  # not in the dataset
    out = pd.DataFrame(rows).sort_values("turbine_id")
    out.to_csv(static_path(farm_id), index=False)
    return out


_cache: dict[str, pd.DataFrame] = {}


def load_10min(path: Path) -> pd.DataFrame:
    key = str(path)
    if key not in _cache:
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} missing - run `python -m data_sources.brazil_coastal_scada` (needs h5py and the Zenodo NetCDFs)")
        _cache[key] = pd.read_csv(path, parse_dates=["timestamp"])
    return _cache[key].copy()


def long_10min(zip_paths: list[Path]) -> pd.DataFrame:
    """timestamp, turbine_id, power_kw, wind_speed_ms (nacelle), mast_ws, mast_wd."""
    frames = []
    for path in zip_paths:
        farm_id = _farm_of(path)
        df = load_10min(path)
        for i in range(1, SITES[farm_id]["turbines"] + 1):
            part = df[["timestamp", "mast_ws", "mast_wd", f"p{i:02d}", f"w{i:02d}"]].rename(
                columns={f"p{i:02d}": "power_kw", f"w{i:02d}": "wind_speed_ms"})
            part["turbine_id"] = turbine_id(farm_id, i)
            frames.append(part)
    return pd.concat(frames, ignore_index=True)


def load_turbine_static(farm_id: str) -> pd.DataFrame:
    return pd.read_csv(static_path(farm_id)).set_index("turbine_id")


def load_farm_hourly_production(zip_paths: list[Path]) -> pd.DataFrame:
    """Hourly output_mw: the reporting turbines' mean times the turbine count (an availability
    scaling, as for Delabole), dropping hours where fewer than three quarters report."""
    farm_id = _farm_of(zip_paths[0])
    n = SITES[farm_id]["turbines"]
    long = long_10min(zip_paths)
    hourly = (long.set_index("timestamp").groupby("turbine_id")["power_kw"]
              .resample("1h").mean().unstack("turbine_id").clip(lower=0))
    hourly = hourly[hourly.notna().sum(axis=1) >= math.ceil(0.75 * n)]
    return (hourly.mean(axis=1) * n / 1000.0).rename("output_mw").to_frame()


def load_status_events(zip_paths: list[Path]) -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_EVENTS)


def load_turbine_hourly_series(zip_paths: list[Path]) -> pd.DataFrame:
    """turbine_id, timestamp, wind_speed_ms (nacelle), power_kw, hourly."""
    long = long_10min(zip_paths).dropna(subset=["power_kw", "wind_speed_ms"])
    hourly = (long.set_index("timestamp").groupby("turbine_id")[["power_kw", "wind_speed_ms"]]
              .resample("1h").mean().reset_index())
    hourly["power_kw"] = hourly["power_kw"].clip(lower=0)
    return hourly.dropna(subset=["wind_speed_ms", "power_kw"])[["turbine_id", "timestamp", "wind_speed_ms", "power_kw"]]


def scada_10min_for_records(farm_id: str) -> pd.DataFrame:
    long = long_10min([csv_path(farm_id)]).dropna(subset=["power_kw"])
    return long.rename(columns={"timestamp": "time", "turbine_id": "asset_id", "power_kw": "WTUR_W"})[["time", "asset_id", "WTUR_W"]]


if __name__ == "__main__":
    for farm in SITES:
        out = convert(farm)
        print(f"wrote {csv_path(farm)} ({csv_path(farm).stat().st_size / 1e6:.1f} MB), {len(out)} rows, "
              f"{out['timestamp'].min()} to {out['timestamp'].max()}")
        layout = infer_layout(farm)
        print(f"wrote {static_path(farm)}: {len(layout)} turbines")
