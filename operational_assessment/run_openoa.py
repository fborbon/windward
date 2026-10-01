"""Runs NREL/NLR's OpenOA operational-assessment methods on the real farm data and writes the
results to dashboard/data/operational_payload.json, which the live service only *reads*
(operational_assessment/payload.py). Offline, in the isolated venv:

    python3 -m venv .venv-openoa && .venv-openoa/bin/pip install -r requirements-openoa.txt
    .venv-openoa/bin/python -m operational_assessment.run_openoa [farm_id ...]

Which OpenOA analyses run, and why (method notes and paper references in README §15):
- ElectricalLosses: summed turbine energy vs the substation meter, all three farms.
- MonteCarloAEP, twice: the industry-standard monthly linear regression the OpenOA benchmark
  used (NREL/TP-5000-78715 §2.3.4), and a daily GAM with temperature as a second input, which
  Bodini et al. 2021 (doi:10.1002/we.2645) found cuts regression uncertainty. Then a
  one-component-at-a-time breakdown of the monthly run, to compare the root-sum-of-squares total
  that assumes uncorrelated components with the full Monte Carlo total that doesn't (Bodini &
  Optis 2020, doi:10.5194/wes-5-1435-2020). Kelmarsh/Penmanshiel only: Hill of Towie's export
  has no availability/curtailment-loss channel and only covers one year.
- WakeLosses (SCADA wind direction), all three farms.
- StaticYawMisalignment, all three farms; OpenOA itself flags this method as not yet validated
  against turbines with known misalignment, which the payload and dashboard repeat.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import openoa
from openoa.analysis import ElectricalLosses, MonteCarloAEP, StaticYawMisalignment, WakeLosses
from openoa.plant import PlantData

from operational_assessment import plant_builder as pb

logging.getLogger("openoa").setLevel(logging.WARNING)
np.random.seed(42)

PAYLOAD_PATH = pb.ROOT / "dashboard" / "data" / "operational_payload.json"

FARM_CONFIG = {
    "kelmarsh": {"lat": 52.4014, "lon": -0.9431, "capacity_mw": 12.3, "products": ("era5", "cerra"),
                 "reanalysis_end": "2021-06-30", "aep": True},
    "penmanshiel": {"lat": 55.9045, "lon": -2.2936, "capacity_mw": 28.7, "products": ("era5", "cerra"),
                    "reanalysis_end": "2021-06-30", "aep": True},
    # CERRA ends 2021-06-30, before Hill of Towie's 2024 period, so ERA5 is the only product
    "hill_of_towie": {"lat": 57.505768, "lon": -3.068384, "capacity_mw": 48.3, "products": ("era5",),
                      "reanalysis_end": "2024-12-31", "aep": False},
}

N_SIM_AEP_MONTHLY = 2000
N_SIM_AEP_DAILY = 300
N_SIM_ELEC = 5000
N_SIM_WAKE = 30
N_SIM_YAW = 50

REFERENCES = {
    "openoa_joss_2021": "Perr-Sauer et al. 2021, OpenOA: An Open-Source Codebase For Operational Analysis of Wind Farms, JOSS 6(58), 2171, doi:10.21105/joss.02171",
    "wp3_benchmark_2021": "Fields et al. 2021, Wind Plant Performance Prediction Benchmark Phase 1 Technical Report, NREL/TP-5000-78715",
    "bodini_2021_power_curves": "Bodini et al. 2021, Lowering post-construction yield assessment uncertainty through better wind plant power curves, Wind Energy, doi:10.1002/we.2645",
    "bodini_optis_2020_uncertainty": "Bodini & Optis 2020, Operational-based annual energy production uncertainty: are its components actually uncorrelated?, Wind Energ. Sci. 5, 1435-1448, doi:10.5194/wes-5-1435-2020",
}


def _f(x, nd=3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def build_plant(farm_id: str, analysis_types: list[str], require_vane: bool = False) -> PlantData:
    cfg = FARM_CONFIG[farm_id]
    if farm_id == "hill_of_towie":
        scada, curtail = pb.load_hill_of_towie_scada(), None
        meter = pb.load_hill_of_towie_meter()
        asset = pb.load_hill_of_towie_asset().drop(columns=["station_id"])
        asset["elevation"] = np.nan
        start = "2004-01-01"
    else:
        scada, curtail = pb.load_greenbyte_scada(farm_id)
        meter = pb.load_greenbyte_meter(farm_id)
        asset = pb.load_greenbyte_asset(farm_id)
        start = pb.REANALYSIS_START
    reanalysis = {p: pb.fetch_reanalysis(farm_id, cfg["lat"], cfg["lon"], p, start=start, end=cfg["reanalysis_end"])
                  for p in cfg["products"]}
    metadata = pb.plant_metadata(cfg["lat"], cfg["lon"], cfg["capacity_mw"], products=cfg["products"])
    if curtail is None:
        metadata.pop("curtail")
    if require_vane:
        # StaticYawMisalignment averages the vane angle with a plain mean, so a single NaN vane
        # reading (about a quarter of Kelmarsh's 10-min rows have none) turns every turbine's
        # result into NaN; give it only rows that actually have a vane and pitch reading.
        scada = scada.dropna(subset=["WMET_HorWdDirRel", "WROT_BlPthAngVal", "WMET_HorWdSpd", "WTUR_W"])
    # WakeLosses slices the (time, asset_id) MultiIndex, which needs it lexsorted
    scada = scada.sort_values(["time", "asset_id"]).reset_index(drop=True)
    return PlantData(
        analysis_type=analysis_types,
        metadata=metadata,
        scada=scada,
        meter=meter,
        curtail=curtail,
        asset=asset,
        reanalysis=reanalysis,
    )


def run_electrical_losses(plant: PlantData) -> dict:
    el = ElectricalLosses(plant, UQ=True, num_sim=N_SIM_ELEC)
    el.run()
    losses = np.asarray(el.electrical_losses).ravel() * 100
    combined = el.combined_energy
    return {
        "mean_pct": _f(losses.mean()),
        "std_pct": _f(losses.std()),
        "p05_pct": _f(np.percentile(losses, 5)),
        "p95_pct": _f(np.percentile(losses, 95)),
        "n_days": int(len(combined)),
        "period_start": str(combined.index.min().date()),
        "period_end": str(combined.index.max().date()),
        "turbine_energy_gwh": _f(el.total_turbine_energy.sum() / 1e6 if np.ndim(el.total_turbine_energy) == 0 else np.mean(el.total_turbine_energy) / 1e6),
        "meter_energy_gwh": _f(el.total_meter_energy.sum() / 1e6 if np.ndim(el.total_meter_energy) == 0 else np.mean(el.total_meter_energy) / 1e6),
        "num_sim": N_SIM_ELEC,
        "uncertainty_meter": el.uncertainty_meter,
        "uncertainty_scada": el.uncertainty_scada,
    }


def _aep_summary(results: pd.DataFrame, num_sim: int) -> dict:
    aep = results["aep_GWh"]
    return {
        "p50_gwh": _f(aep.median()),
        "mean_gwh": _f(aep.mean()),
        "std_gwh": _f(aep.std()),
        "uncertainty_pct": _f(aep.std() / aep.mean() * 100),
        "p90_gwh": _f(np.percentile(aep, 10)),  # P90 = exceeded with 90% probability
        "avail_loss_pct": _f(results["avail_pct"].mean() * 100),
        "curt_loss_pct": _f(results["curt_pct"].mean() * 100),
        "lt_por_ratio": _f(results["lt_por_ratio"].mean()),
        "r2_mean": _f(results["r2"].mean()),
        "n_points_mean": _f(results["n_points"].mean(), 1),
        "iav_pct": _f(results["iav"].mean() * 100),
        "num_sim": num_sim,
    }


def run_aep(plant: PlantData, products: tuple[str, ...]) -> dict:
    products = list(products)
    common = dict(reanalysis_products=products, apply_iav=False)
    out = {}

    t0 = time.time()
    monthly = MonteCarloAEP(plant, time_resolution="MS", reg_model="lin", outlier_detection=True, **common)
    monthly.run(num_sim=N_SIM_AEP_MONTHLY, progress_bar=False)
    out["monthly_linear"] = _aep_summary(monthly.results, N_SIM_AEP_MONTHLY)
    out["monthly_linear"]["runtime_s"] = round(time.time() - t0, 1)
    out["por_start"] = str(monthly.start_por.date())
    out["por_end"] = str(monthly.end_por.date())

    t0 = time.time()
    daily = MonteCarloAEP(plant, time_resolution="D", reg_model="gam", reg_temperature=True,
                          outlier_detection=True, **common)
    daily.run(num_sim=N_SIM_AEP_DAILY, progress_bar=False)
    out["daily_gam_temperature"] = _aep_summary(daily.results, N_SIM_AEP_DAILY)
    out["daily_gam_temperature"]["runtime_s"] = round(time.time() - t0, 1)

    # One component at a time (Bodini & Optis 2020 §2.3). OpenOA 3.2 always bootstraps the
    # regression data, so a run with every other component pinned is the regression-only
    # baseline, and each other component's own contribution is sqrt(var_with_it - var_baseline).
    pinned = dict(uncertainty_meter=0.0, uncertainty_losses=0.0, uncertainty_windiness=(20.0, 20.0),
                  uncertainty_loss_max=(15.0, 15.0), uncertainty_outlier=(2.0, 2.0),
                  reanalysis_products=[products[0]])
    components = {
        "regression": {},
        "revenue_meter": {"uncertainty_meter": 0.005},
        "reported_losses": {"uncertainty_losses": 0.05},
        "windiness_lt_period": {"uncertainty_windiness": (10.0, 20.0)},
        "loss_threshold": {"uncertainty_loss_max": (10.0, 20.0)},
        "outlier_threshold": {"uncertainty_outlier": (1.0, 3.0)},
    }
    if len(products) > 1:
        components["reanalysis_product"] = {"reanalysis_products": products}
    cv = {}
    for name, override in components.items():
        params = {**pinned, **override}
        mc = MonteCarloAEP(plant, time_resolution="MS", reg_model="lin", outlier_detection=True, apply_iav=False,
                           reanalysis_products=params.pop("reanalysis_products"))
        mc.run(num_sim=N_SIM_AEP_MONTHLY, progress_bar=False, **params)
        cv[name] = mc.results["aep_GWh"].std() / mc.results["aep_GWh"].mean() * 100
    reg = cv["regression"]
    breakdown = {"regression": _f(reg)}
    for name, value in cv.items():
        if name != "regression":
            breakdown[name] = _f(np.sqrt(max(value ** 2 - reg ** 2, 0.0)))
    rss = np.sqrt(sum(v ** 2 for v in breakdown.values()))
    out["uncertainty_components_pct"] = breakdown
    out["rss_uncorrelated_total_pct"] = _f(rss)
    out["monte_carlo_total_pct"] = out["monthly_linear"]["uncertainty_pct"]
    return out


def run_wake_losses(plant: PlantData, products: tuple[str, ...]) -> dict:
    wl = WakeLosses(plant, wind_direction_col="WMET_HorWdDir", wind_direction_data_type="scada",
                    UQ=True, num_sim=N_SIM_WAKE, reanalysis_products=list(products))
    wl.run()
    turbine_ids = list(wl.turbine_ids)
    wd_bins = np.arange(0.0, 360.0, wl.wd_bin_width)
    por_wd = np.asarray(wl.wake_losses_por_wd).mean(axis=0) * 100 if np.ndim(wl.wake_losses_por_wd) > 1 else np.asarray(wl.wake_losses_por_wd) * 100
    energy_wd = np.asarray(wl.energy_por_wd).mean(axis=0) if np.ndim(wl.energy_por_wd) > 1 else np.asarray(wl.energy_por_wd)
    energy_share = energy_wd / energy_wd.sum() * 100 if energy_wd.sum() else energy_wd
    return {
        "plant_por_pct": _f(wl.wake_losses_por_mean * 100),
        "plant_por_std_pct": _f(wl.wake_losses_por_std * 100),
        "plant_lt_pct": _f(wl.wake_losses_lt_mean * 100),
        "plant_lt_std_pct": _f(wl.wake_losses_lt_std * 100),
        "turbines": {
            tid: {
                "por_pct": _f(np.asarray(wl.turbine_wake_losses_por_mean)[i] * 100),
                "lt_pct": _f(np.asarray(wl.turbine_wake_losses_lt_mean)[i] * 100),
                "lt_std_pct": _f(np.asarray(wl.turbine_wake_losses_lt_std)[i] * 100),
            }
            for i, tid in enumerate(turbine_ids)
        },
        "by_direction": [
            {"wd_deg": float(wd), "plant_por_pct": _f(l), "energy_share_pct": _f(e)}
            for wd, l, e in zip(wd_bins, por_wd, energy_share)
        ],
        "num_sim": N_SIM_WAKE,
    }


def run_yaw(farm_id: str) -> dict:
    """One StaticYawMisalignment per turbine, so a cosine fit that doesn't converge on one
    turbine (scipy's curve_fit hits maxfev on one Hill of Towie turbine) is recorded for that
    turbine instead of discarding the whole farm's results."""
    plant = build_plant(farm_id, ["StaticYawMisalignment"], require_vane=True)
    out = {}
    ws_bins = None
    for tid in plant.turbine_ids:
        ym = StaticYawMisalignment(plant, turbine_ids=[tid], UQ=True, num_sim=N_SIM_YAW)
        try:
            ym.run(num_sim=N_SIM_YAW)
        except Exception as exc:
            out[tid] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        ws_bins = list(ym.ws_bins)
        sims = np.asarray(ym.yaw_misalignment).reshape(N_SIM_YAW, -1)[:, 0]
        by_ws = np.asarray(ym.yaw_misalignment_avg_ws).reshape(-1)
        out[tid] = {
            "mean_deg": _f(np.asarray(ym.yaw_misalignment_avg).ravel()[0], 2),
            "std_deg": _f(np.asarray(ym.yaw_misalignment_std).ravel()[0], 2),
            "ci95_deg": [_f(np.nanpercentile(sims, 2.5), 2), _f(np.nanpercentile(sims, 97.5), 2)],
            "mean_vane_deg": _f(np.nanmean(np.asarray(ym.mean_vane_angle)), 2),
            "by_ws_deg": {str(ws): _f(v, 2) for ws, v in zip(ym.ws_bins, by_ws)},
            "by_ws_spread_deg": _f(np.nanmax(by_ws) - np.nanmin(by_ws), 2),
        }
    return {
        "turbines": out,
        "ws_bins_ms": ws_bins,
        "num_sim": N_SIM_YAW,
        "caveat": "OpenOA's StaticYawMisalignment is not yet validated against turbines with known static yaw misalignment; it relies on nacelle wind speed, which itself is affected by yaw misalignment. Treat as a screening indicator, not a calibration value.",
    }


def run_farm(farm_id: str) -> dict:
    cfg = FARM_CONFIG[farm_id]
    types = ["ElectricalLosses", "WakeLosses-scada"]
    if cfg["aep"]:
        types += ["MonteCarloAEP-temp"]
    t0 = time.time()
    plant = build_plant(farm_id, types)
    print(f"[{farm_id}] PlantData built in {time.time() - t0:.0f}s", flush=True)
    result = {"reanalysis_products": list(cfg["products"])}
    steps = [("electrical_losses", lambda: run_electrical_losses(plant)),
             ("wake_losses", lambda: run_wake_losses(plant, cfg["products"])),
             ("yaw_misalignment", lambda: run_yaw(farm_id))]
    if cfg["aep"]:
        steps.insert(1, ("aep", lambda: run_aep(plant, cfg["products"])))
    else:
        result["aep"] = {"skipped": "No availability/curtailment-loss channel in this SCADA export, and only one year of data; an operational AEP without loss correction would mix downtime into the power curve regression."}
    for name, fn in steps:
        t0 = time.time()
        try:
            result[name] = fn()
        except Exception as exc:  # keep the other analyses; record the real failure
            result[name] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"[{farm_id}] {name} done in {time.time() - t0:.0f}s: {json.dumps(result[name])[:300]}", flush=True)
    return result


def _result_path(farm_id: str) -> Path:
    return pb.CACHE_DIR / f"openoa_result_{farm_id}.json"


def merge_payload():
    """Farms run as separate processes (they're independent and each takes a while), each
    writing its own result file; this folds whatever results exist into the one payload."""
    payload = json.loads(PAYLOAD_PATH.read_text()) if PAYLOAD_PATH.exists() else {"farms": {}}
    for farm_id in FARM_CONFIG:
        path = _result_path(farm_id)
        if path.exists():
            payload["farms"][farm_id] = json.loads(path.read_text())
    payload.update({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "openoa_version": openoa.__version__,
        "references": REFERENCES,
    })
    PAYLOAD_PATH.write_text(json.dumps(payload, indent=1))
    print(f"wrote {PAYLOAD_PATH} ({', '.join(payload['farms'])})")


def main(farm_ids: list[str]):
    for farm_id in farm_ids:
        pb.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _result_path(farm_id).write_text(json.dumps(run_farm(farm_id), indent=1))
    merge_payload()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--merge"]:
        merge_payload()
    else:
        main(args or list(FARM_CONFIG))
