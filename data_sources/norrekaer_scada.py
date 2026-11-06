"""Loader for Nørrekær Enge II (DTU Data, doi:10.11583/DTU.19076756): 42 Nordtank NTK 300F
(300 kW, 28 m rotor, 31 m hub, stall-regulated) on flat reclaimed seabed on the Limfjord, North
Jutland, Denmark. Measured by Risø 1991-12 to 1993-06 to study wakes and turbulence inside a
farm (Højstrup et al. 1993, Risø-I-684(EN),
https://gitlab.windenergy.dtu.dk/fair-data/winddata-revamp/winddata-documentation/-/raw/master/wind_farm/norre_m2/norrekaer_windfarm.pdf).

Layout: six rows A-F of seven turbines, rows running about 166/346 degrees, 6.3 D apart within a
row and 7-8 D between rows, with a ~730 m gap between rows C and D. Turbine and mast positions are
the report's Appendix 5 UTM (zone 32N) coordinates (Norrekaer_turbine_utm32.csv), checked against
its Appendix 4 distance table; the DTU site page's own layout list is partly corrupted.

What makes it different from the other farms:
- Per-turbine nacelle anemometer (`*_wsn`) and yaw misalignment (`*_ym`) for all 42 turbines,
  plus two hub-height (31 m) met masts: M1 upstream of the farm for the prevailing south-west
  wind, M2 deep inside it next to F6.
- Power for 40 turbines: A1 and F6 (the two strain-gauged turbines next to the masts) have
  nacelle wind and yaw only, their power lives in a sibling dataset.
- Risø's acquisition kept 10-minute statistics only for runs with hub-height wind above about
  2.5 m/s at both masts, so calm hours are absent by design, not missing.
- No alarm/status log.

The NetCDF needs h5py, so convert() writes data/norrekaer_enge/norrekaer_10min.csv.gz once
(committed) and the loaders read that:

    pip install h5py && python -m data_sources.norrekaer_scada

Licence note: the DTU Data record says CC BY 4.0; the NetCDF's own `license` attribute says
CC BY NC 4.0. Attribute DTU Data / Risø, and check before any commercial use.
"""
import math
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "norrekaer_enge"
NETCDF_PATH = DATA_DIR / "norre_m2_all.nc"
CSV_PATH = DATA_DIR / "norrekaer_10min.csv.gz"
PARK_CSV_PATH = DATA_DIR / "norrekaer_park_10min.csv.gz"  # continuous park meter, every 10 minutes
UTM_PATH = DATA_DIR / "Norrekaer_turbine_utm32.csv"

ROWS = "ABCDEF"
TURBINES = [f"{r}{i}" for r in ROWS for i in range(1, 8)]  # A1..F7, 42
NO_POWER = {"A1", "F6"}
POWER_TURBINES = [t for t in TURBINES if t not in NO_POWER]  # 40
RATED_KW = 300.0
ROTOR_D_M = 28.0
MAST_CHANNELS = ["s31_1", "s31_2", "d31_1", "d34_2", "park", "connect"]

_EMPTY_EVENTS = ["turbine_id", "start", "end", "duration", "status", "code", "message", "iec_category"]


def turbine_id(name: str) -> str:
    return f"Norrekaer_{name}"


def utm32_to_latlon(easting: float, northing: float) -> tuple[float, float]:
    """WGS84 UTM zone 32N -> lat/lon (standard series inversion, sub-metre at this scale)."""
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    x, y = easting - 500000.0, northing
    m = y / k0
    mu = m / (a * (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu) + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
            + (151 * e1**3 / 96) * math.sin(6 * mu) + (1097 * e1**4 / 512) * math.sin(8 * mu))
    n1 = a / math.sqrt(1 - e2 * math.sin(phi1) ** 2)
    t1 = math.tan(phi1) ** 2
    c1 = ep2 * math.cos(phi1) ** 2
    r1 = a * (1 - e2) / (1 - e2 * math.sin(phi1) ** 2) ** 1.5
    d = x / (n1 * k0)
    lat = phi1 - (n1 * math.tan(phi1) / r1) * (d**2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * ep2) * d**4 / 24
                                               + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * ep2 - 3 * c1**2) * d**6 / 720)
    lon = (d - (1 + 2 * t1 + c1) * d**3 / 6 + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * ep2 + 24 * t1**2) * d**5 / 120) / math.cos(phi1)
    return math.degrees(lat), 9.0 + math.degrees(lon)


def convert(nc_path: Path = NETCDF_PATH, out_path: Path = CSV_PATH) -> pd.DataFrame:
    """NetCDF4 -> compact CSV: mast/park channels plus each turbine's power, nacelle wind and yaw
    misalignment. Rows without any turbine data (the park-only continuous record outside Risø's
    selected runs) are dropped. Times are 'minutes since 1991-12-22 23:36:00', taken as UTC."""
    import h5py  # offline-only dependency, see the module docstring

    with h5py.File(nc_path, "r") as f:
        cols = {c: f[c][:] for c in MAST_CHANNELS}
        for t in TURBINES:
            key = t.lower()
            if t not in NO_POWER:
                cols[f"{t}_pow"] = f[f"{key}_pow"][:]
            cols[f"{t}_wsn"] = f[f"{key}_wsn"][:]
            cols[f"{t}_ym"] = f[f"{key}_ym"][:]
        minutes = f["time"][:]
    df = pd.DataFrame(cols)
    df.insert(0, "timestamp", pd.Timestamp("1991-12-22 23:36") + pd.to_timedelta(minutes, unit="min"))
    df[["timestamp", "park", "connect"]].assign(park=df["park"].round(1)).to_csv(PARK_CSV_PATH, index=False, compression="gzip")
    turbine_cols = [c for c in df.columns if c.endswith(("_pow", "_wsn"))]
    df = df[df[turbine_cols].notna().any(axis=1)]
    num = df.columns[1:]
    df[num] = df[num].round(1)
    df.to_csv(out_path, index=False, compression="gzip")
    return df


_cache: dict[str, pd.DataFrame] = {}


def load_10min(path: Path = CSV_PATH) -> pd.DataFrame:
    key = str(path)
    if key not in _cache:
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run `python -m data_sources.norrekaer_scada` (needs h5py and {NETCDF_PATH.name})")
        _cache[key] = pd.read_csv(path, parse_dates=["timestamp"])
    return _cache[key].copy()


def long_10min(zip_paths: list[Path] | None = None) -> pd.DataFrame:
    """timestamp, turbine_id, power_kw, wind_speed_ms (nacelle), yaw_misalignment_deg, plus the
    mast/park channels on every row. Only the 40 turbines with power."""
    df = pd.concat([load_10min(p) for p in (zip_paths or [CSV_PATH])], ignore_index=True)
    frames = []
    for t in POWER_TURBINES:
        part = df[["timestamp", *MAST_CHANNELS, f"{t}_pow", f"{t}_wsn", f"{t}_ym"]].rename(
            columns={f"{t}_pow": "power_kw", f"{t}_wsn": "wind_speed_ms", f"{t}_ym": "yaw_misalignment_deg"})
        part["turbine_id"] = turbine_id(t)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def load_turbine_static(farm_id: str) -> pd.DataFrame:
    """Kelmarsh_WT_static.csv's layout, from the report's Appendix 1 (specs) and 5 (UTM)."""
    utm = pd.read_csv(UTM_PATH).set_index("turbine")
    rows = []
    for t in POWER_TURBINES:
        lat, lon = utm32_to_latlon(*utm.loc[t, ["easting", "northing"]])
        rows.append({"turbine_id": turbine_id(t), "Title": f"Norrekaer {t}", "Manufacturer": "Nordtank",
                     "Model": "NTK 300F", "Rated power (kW)": RATED_KW, "Hub Height (m)": 31.0,
                     "Rotor Diameter (m)": ROTOR_D_M, "Latitude": round(lat, 6), "Longitude": round(lon, 6),
                     "Easting (m)": utm.loc[t, "easting"], "Northing (m)": utm.loc[t, "northing"],
                     "Commercial Operations Date": "1991"})
    return pd.DataFrame(rows).set_index("turbine_id")


def load_farm_hourly_production(zip_paths: list[Path]) -> pd.DataFrame:
    """Hourly output_mw from the park meter (`park`), not the turbine sum: the turbine channels
    only exist for Risø's selected windy runs (about 5,000 hours, biased high), while the park
    meter runs continuously (about 12,350 hours, calm ones included), which is what a forecast
    model must learn from. The meter covers all 42 turbines; it equals 1.05 x the 40-turbine sum
    (correlation 1.00 over 11,222 rows where all 40 report), so it is scaled by 40/42 to match
    the registered turbine set and capacity. Small negative readings (idle consumption) clip to 0."""
    park = pd.read_csv(PARK_CSV_PATH, parse_dates=["timestamp"]).set_index("timestamp")["park"]
    hourly = park.clip(lower=0).resample("1h").mean().dropna()
    return (hourly * len(POWER_TURBINES) / len(TURBINES) / 1000.0).rename("output_mw").to_frame()


def load_status_events(zip_paths: list[Path]) -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_EVENTS)


def load_turbine_hourly_series(zip_paths: list[Path]) -> pd.DataFrame:
    """turbine_id, timestamp, wind_speed_ms (nacelle), power_kw, hourly."""
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
