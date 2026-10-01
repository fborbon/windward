"""Builds OpenOA `PlantData` inputs (scada / meter / curtail / asset / reanalysis frames) from
this project's raw open SCADA exports, so OpenOA's own analysis classes can run on them.

Runs OFFLINE, inside the isolated OpenOA venv (requirements-openoa.txt), never inside the live
service: OpenOA 3.2 pins scikit-learn<1.7 (and its pygam dependency pins scipy<1.17), which
would downgrade the service's scikit-learn/scipy and risk the registered MLflow models no
longer unpickling. So this module deliberately imports nothing from the rest of the project
(data_sources/ pulls in the service's dependency tree) and only needs pandas + requests.

Signal choices, each verified against the real files (see README §15):
- Revenue meter = the substation PMU's "GMS Energy Export (kWh)" for Kelmarsh/Penmanshiel, NOT
  the "Grid Meter" device: over Kelmarsh's full-coverage 2016 period the Grid Meter channel
  summed to 16.578 GWh against 16.563 GWh of summed turbine export, i.e. *more* energy at the
  grid than the turbines produced, which is physically impossible for a real meter (it behaves
  like a derived/allocated channel). The PMU summed to 16.404 GWh, a plausible ~1% loss.
- Hill of Towie's meter = tblGrid's cumulative ActivePowerExport counter (kWh) at station 91,
  differenced to 10-min energy. Station 91 covers the whole farm (peaks at 47.6 MW for a 48.3 MW
  farm, r=0.99 against the summed turbine active power).
- Availability / curtailment losses (Kelmarsh/Penmanshiel only) = the turbines' own Greenbyte
  "Lost Production to Downtime" / "Lost Production to Curtailment (Total)" channels, summed.
  Hill of Towie's export has no equivalent channel, so MonteCarloAEP isn't run for it.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "operational_cache"

# Greenbyte column name -> OpenOA IEC 61400-25 tag
GREENBYTE_SCADA_COLS = {
    "Power (kW)": "WTUR_W",
    "Wind speed (m/s)": "WMET_HorWdSpd",
    "Wind direction (°)": "WMET_HorWdDir",
    "Vane position 1+2 (°)": "WMET_HorWdDirRel",
    "Blade angle (pitch position) A (°)": "WROT_BlPthAngVal",
    "Nacelle ambient temperature (°C)": "WMET_EnvTmp",
}
GREENBYTE_LOSS_COLS = {
    "Lost Production to Downtime (kWh)": "IAVL_DnWh",
    "Lost Production to Curtailment (Total) (kWh)": "IAVL_ExtPwrDnWh",
}

GREENBYTE_FARMS = {
    "kelmarsh": {
        "scada_zips": ["Kelmarsh_SCADA_2016.zip"] + [f"Kelmarsh_SCADA_{y}_{3083 + y - 2017}.zip" for y in range(2017, 2022)],
        "pmu_zip": "Kelmarsh_PMU_3089.zip",
    },
    "penmanshiel": {
        "scada_zips": [
            "Penmanshiel_SCADA_2016_WT01-10.zip", "Penmanshiel_SCADA_2016_WT11-15.zip",
            "Penmanshiel_SCADA_2017_WT01-10_3114.zip", "Penmanshiel_SCADA_2017_WT11-15_3115.zip",
            "Penmanshiel_SCADA_2018_WT01-10_3113.zip", "Penmanshiel_SCADA_2018_WT11-15_3116.zip",
            "Penmanshiel_SCADA_2019_WT01-10_3112.zip", "Penmanshiel_SCADA_2019_WT11-15_3117.zip",
            "Penmanshiel_SCADA_2020_WT01-10_3109.zip", "Penmanshiel_SCADA_2020_WT11-15_3118.zip",
            "Penmanshiel_SCADA_2021_WT01-10_3108.zip", "Penmanshiel_SCADA_2021_WT11-15_3108.zip",
        ],
        "pmu_zip": "Penmanshiel_PMU_3152.zip",
    },
}

# Reanalysis products available from Open-Meteo's archive at 100 m: ERA5 (global, ~31 km) and
# CERRA (Copernicus European regional reanalysis, ~5.5 km, ends 2021-06-30). Two products is what
# lets MonteCarloAEP sample reanalysis-product choice as an uncertainty component, as in the
# OpenOA benchmark methodology (NREL/TP-5000-78715 §2.3.4, which used ERA-Interim/MERRA-2/GFS).
REANALYSIS_PRODUCTS = ("era5", "cerra")
REANALYSIS_START = "2001-01-01"
REANALYSIS_END = "2021-06-30"


def _greenbyte_header_row(zf: zipfile.ZipFile, member: str) -> int:
    with zf.open(member) as f:
        return next(i for i, line in enumerate(f) if line.startswith(b"# Date and time"))


def _read_greenbyte(zf: zipfile.ZipFile, member: str, wanted: dict[str, str]) -> pd.DataFrame:
    hdr = _greenbyte_header_row(zf, member)
    norm = lambda c: c.replace("\n", " ").strip()
    with zf.open(member) as f:
        df = pd.read_csv(
            f, skiprows=hdr, low_memory=False,
            usecols=lambda c: norm(c) in wanted or c.startswith("# Date and time"),
        )
    df.columns = [norm(c) for c in df.columns]
    df = df.rename(columns={"# Date and time": "time", **wanted})
    df["time"] = pd.to_datetime(df["time"])
    return df


def _turbine_id_from_member(member: str) -> str:
    # Turbine_Data_Kelmarsh_1_2016-...csv -> Kelmarsh_1 ; Turbine_Data_Penmanshiel_01_... -> Penmanshiel_01
    parts = member.split("_")
    return f"{parts[2]}_{parts[3]}"


def load_greenbyte_scada(farm_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(scada, curtail): scada is long-format 10-min per turbine with OpenOA tags; curtail is the
    10-min farm total of downtime + curtailment lost energy (kWh)."""
    cache = CACHE_DIR / f"{farm_id}_scada.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    frames = []
    for name in GREENBYTE_FARMS[farm_id]["scada_zips"]:
        path = DATA_DIR / farm_id / name
        with zipfile.ZipFile(path) as zf:
            for member in sorted(n for n in zf.namelist() if n.startswith("Turbine_Data_")):
                df = _read_greenbyte(zf, member, {**GREENBYTE_SCADA_COLS, **GREENBYTE_LOSS_COLS})
                df["asset_id"] = _turbine_id_from_member(member)
                frames.append(df)
    raw = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time", "asset_id"])
    # Vane angle and pitch to [-180, 180), the same normalisation OpenOA's own ENGIE example applies
    for col in ("WMET_HorWdDirRel", "WROT_BlPthAngVal"):
        raw[col] = ((raw[col] + 180.0) % 360.0) - 180.0
    scada = raw[["time", "asset_id", *GREENBYTE_SCADA_COLS.values()]].copy()
    curtail = raw.groupby("time")[list(GREENBYTE_LOSS_COLS.values())].sum(min_count=1).reset_index()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle((scada, curtail), cache)
    return scada, curtail


def load_greenbyte_meter(farm_id: str) -> pd.DataFrame:
    path = DATA_DIR / farm_id / GREENBYTE_FARMS[farm_id]["pmu_zip"]
    with zipfile.ZipFile(path) as zf:
        member = next(n for n in zf.namelist() if n.startswith("Device_Data_"))
        df = _read_greenbyte(zf, member, {"GMS Energy Export (kWh)": "MMTR_SupWh"})
    return df.dropna(subset=["MMTR_SupWh"])


def load_greenbyte_asset(farm_id: str) -> pd.DataFrame:
    static = pd.read_csv(next((DATA_DIR / farm_id).glob("*_WT_static.csv")), encoding="utf-8-sig")
    static = static.dropna(subset=["Title"])  # Penmanshiel's file ends with blank ",,,," rows
    return pd.DataFrame({
        "asset_id": static["Title"].str.replace(" ", "_", regex=False),
        "latitude": static["Latitude"],
        "longitude": static["Longitude"],
        "rated_power": static["Rated power (kW)"].astype(float),
        "hub_height": static["Hub Height (m)"].astype(float),
        "rotor_diameter": static["Rotor Diameter (m)"].astype(float),
        "elevation": static["Elevation (m)"].astype(float),
        "type": "turbine",
    })


# --- Hill of Towie (RES historian export, one CSV per signal table per month) ---

def _hot_table(table: str, usecols: list[str]) -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(DATA_DIR / "hill_of_towie" / "2024.zip") as zf:
        for member in sorted(n for n in zf.namelist() if n.startswith(f"{table}_")):
            with zf.open(member) as f:
                frames.append(pd.read_csv(f, usecols=usecols))
    df = pd.concat(frames, ignore_index=True)
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], format="mixed")
    return df


def load_hill_of_towie_asset() -> pd.DataFrame:
    meta = pd.read_csv(DATA_DIR / "hill_of_towie" / "Hill_of_Towie_turbine_metadata.csv", encoding="utf-8-sig")
    return pd.DataFrame({
        "asset_id": "HillOfTowie_" + meta["Turbine Name"].str.extract(r"T(\d+)")[0],
        "station_id": meta["Station ID"],
        "latitude": meta["Latitude"],
        "longitude": meta["Longitude"],
        "rated_power": meta["Rated power (kW)"].astype(float),
        "hub_height": meta["Hub Height (m)"].astype(float),
        "rotor_diameter": meta["Rotor Diameter (m)"].astype(float),
        "type": "turbine",
    })


def load_hill_of_towie_scada() -> pd.DataFrame:
    cache = CACHE_DIR / "hill_of_towie_scada.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    keys = ["TimeStamp", "StationId"]
    turb = _hot_table("tblSCTurbine", keys + ["wtc_AcWindSp_mean", "wtc_ActualWindDirection_mean", "wtc_NacelPos_mean", "wtc_PitcPosA_mean"])
    grid = _hot_table("tblSCTurGrid", keys + ["wtc_ActPower_mean"])
    temp = _hot_table("tblSCTurTemp", keys + ["wtc_AmbieTmp_mean"])
    df = turb.merge(grid, on=keys, how="outer").merge(temp, on=keys, how="left")
    asset = load_hill_of_towie_asset()
    df["asset_id"] = df["StationId"].map(dict(zip(asset["station_id"], asset["asset_id"])))
    # This export has no vane channel; the relative wind direction is the absolute wind
    # direction minus the nacelle position, wrapped to [-180, 180).
    rel = ((df["wtc_ActualWindDirection_mean"] - df["wtc_NacelPos_mean"] + 180.0) % 360.0) - 180.0
    scada = pd.DataFrame({
        "time": df["TimeStamp"].dt.floor("10min"),
        "asset_id": df["asset_id"],
        "WTUR_W": df["wtc_ActPower_mean"],
        "WMET_HorWdSpd": df["wtc_AcWindSp_mean"],
        "WMET_HorWdDir": df["wtc_ActualWindDirection_mean"] % 360.0,
        "WMET_HorWdDirRel": rel,
        "WROT_BlPthAngVal": df["wtc_PitcPosA_mean"],
        "WMET_EnvTmp": df["wtc_AmbieTmp_mean"],
    }).dropna(subset=["asset_id"]).drop_duplicates(subset=["time", "asset_id"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    scada.to_pickle(cache)
    return scada


def load_hill_of_towie_meter() -> pd.DataFrame:
    g = _hot_table("tblGrid", ["TimeStamp", "Station", "ActivePowerExport"])
    g = g[g["Station"] == 91].sort_values("TimeStamp").set_index("TimeStamp")
    g.index = g.index.floor("10min")
    g = g[~g.index.duplicated()].asfreq("10min")
    energy = g["ActivePowerExport"].diff()
    # The counter is cumulative kWh: a step outside [0, 48.3 MW x 1/6 h] is a counter reset or a gap
    energy = energy.where((energy >= 0) & (energy <= 48300 / 6 * 1.05))
    return energy.rename("MMTR_SupWh").dropna().rename_axis("time").reset_index()


# --- Reanalysis (Open-Meteo archive) ---

def fetch_reanalysis(farm_id: str, lat: float, lon: float, product: str,
                     start: str = REANALYSIS_START, end: str = REANALYSIS_END) -> pd.DataFrame:
    cache = CACHE_DIR / f"{farm_id}_reanalysis_{product}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": "wind_speed_100m,wind_direction_100m,temperature_2m,surface_pressure",
        "models": product, "wind_speed_unit": "ms", "timezone": "GMT",
    }
    resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=300)
    resp.raise_for_status()
    h = resp.json()["hourly"]
    ws = np.asarray(h["wind_speed_100m"], dtype=float)
    wd = np.asarray(h["wind_direction_100m"], dtype=float)
    df = pd.DataFrame({
        "datetime": pd.to_datetime(h["time"]),
        "ws_100": ws,
        "wd_100": wd,
        # Meteorological convention: direction the wind blows FROM
        "u_100": -ws * np.sin(np.deg2rad(wd)),
        "v_100": -ws * np.cos(np.deg2rad(wd)),
        "t_2m": np.asarray(h["temperature_2m"], dtype=float) + 273.15,  # K, as OpenOA expects
        "surf_pres": np.asarray(h["surface_pressure"], dtype=float) * 100.0,  # hPa -> Pa
    }).dropna()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    return df


def plant_metadata(lat: float, lon: float, capacity_mw: float, products=REANALYSIS_PRODUCTS) -> dict:
    reanalysis_meta = {
        p: {
            "frequency": "h", "time": "datetime",
            "WMETR_HorWdSpd": "ws_100", "WMETR_HorWdDir": "wd_100",
            "WMETR_HorWdSpdU": "u_100", "WMETR_HorWdSpdV": "v_100",
            "WMETR_EnvTmp": "t_2m", "WMETR_EnvPres": "surf_pres", "WMETR_AirDen": "dens",
        }
        for p in products
    }
    return {
        "latitude": lat, "longitude": lon, "capacity": capacity_mw,
        "scada": {"frequency": "10min", "time": "time", "asset_id": "asset_id",
                  **{tag: tag for tag in GREENBYTE_SCADA_COLS.values()}},
        "meter": {"frequency": "10min", "time": "time", "MMTR_SupWh": "MMTR_SupWh"},
        "curtail": {"frequency": "10min", "time": "time", "IAVL_DnWh": "IAVL_DnWh", "IAVL_ExtPwrDnWh": "IAVL_ExtPwrDnWh"},
        "asset": {"asset_id": "asset_id", "latitude": "latitude", "longitude": "longitude",
                  "rated_power": "rated_power", "hub_height": "hub_height", "rotor_diameter": "rotor_diameter",
                  "elevation": "elevation", "type": "type"},
        "reanalysis": reanalysis_meta,
    }
