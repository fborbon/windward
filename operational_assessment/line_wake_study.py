"""Measured wake study for the single-line farms: Pedra do Sal and Beberibe (Brazil) and SMARTEOLE's
Sole du Moulin Vieux (France). Same method as the Nørrekær Enge study
(operational_assessment/norrekaer_wake_study.py), generalised to any layout:

1. Wind direction per 10-minute record: the met mast vane at 100 m for the Brazilian farms; for
   SMARTEOLE, the vector mean of the nacelle directions of SMV1, 2, 3 and 7 (the owners
   corrected them for north alignment; SMV4-6 are left out because SMV6 is the steered turbine).
2. Freestream turbines for that direction: no other turbine upstream within +/-20 degrees and
   20 rotor diameters.
3. Wake loss = 1 - turbine energy / freestream-equivalent energy, with the freestream mean at
   5-90% of rated, per turbine, per 5-degree bin and for the plant.

On top, one farm-specific question each, written as plain-language findings the dashboard shows:
- SMARTEOLE: did yawing SMV6 off the wind help? Steering-on vs steering-off records in SMV5's
  wake sector, binned by reference wind speed and direction so both sides see the same wind.
- Brazil: how steady the trade wind is, and what that does to a line of turbines.

    python -m operational_assessment.line_wake_study      # writes all three into operational_payload.json
"""
import json
import math

import numpy as np
import pandas as pd

from data_sources import brazil_coastal_scada as bc
from data_sources import smarteole_scada as sm
from operational_assessment.payload import PAYLOAD_PATH

SECTOR_HALF_DEG = 20.0
WAKE_RANGE_D = 20.0
FREE_POWER_BAND = (0.05, 0.90)
SMARTEOLE_REF = [1, 2, 3, 7]
STEER_SECTOR = (195.0, 225.0)  # SMV5 sits 302 m from SMV6 at 207.5 deg: its wake sector
STEER_WS = (4.0, 12.0)


def _xy(static: pd.DataFrame) -> pd.DataFrame:
    lat0, lon0 = static["Latitude"].mean(), static["Longitude"].mean()
    return pd.DataFrame({"x": (static["Longitude"] - lon0) * 111320 * math.cos(math.radians(lat0)),
                         "y": (static["Latitude"] - lat0) * 110574}, index=static.index)


def freestream_sets(static: pd.DataFrame, rotor_m: float) -> dict[int, list[str]]:
    pos = _xy(static).to_numpy()
    dx = pos[None, :, 0] - pos[:, None, 0]
    dy = pos[None, :, 1] - pos[:, None, 1]
    dist = np.hypot(dx, dy)
    bearing = np.degrees(np.arctan2(dx, dy)) % 360
    near = (dist > 0) & (dist <= WAKE_RANGE_D * rotor_m)
    out = {}
    for wd in range(360):
        diff = (bearing - wd + 180) % 360 - 180
        waked = (near & (np.abs(diff) <= SECTOR_HALF_DEG)).any(axis=1)
        out[wd] = [t for t, w in zip(static.index, waked) if not w]
    return out


def study(power: pd.DataFrame, wd: pd.Series, static: pd.DataFrame, rated_kw: float, rotor_m: float,
          min_report: int) -> tuple[dict, pd.DataFrame]:
    """power: records x turbines (kW); wd: direction per record. Returns (wake_losses, kept meta)."""
    ok = wd.notna().to_numpy()
    power, wd = power.clip(lower=0)[ok], wd[ok]
    free = freestream_sets(static.loc[power.columns], rotor_m)
    col = {t: i for i, t in enumerate(power.columns)}
    vals = power.to_numpy()
    wd_int = (wd.round() % 360).astype(int).to_numpy()
    ref = np.full(len(power), np.nan)
    for i, d in enumerate(wd_int):
        row = vals[i, [col[t] for t in free[d]]]
        row = row[~np.isnan(row)]
        if len(row):
            ref[i] = row.mean()
    meta = pd.DataFrame({"wd": wd.to_numpy(), "free_kw": ref, "n_report": power.notna().sum(axis=1).to_numpy()}, index=power.index)
    lo, hi = (v * rated_kw for v in FREE_POWER_BAND)
    keep = meta["free_kw"].between(lo, hi) & (meta["n_report"] >= min_report)
    power, meta = power[keep], meta[keep]
    turbines = {}
    for t in power.columns:
        ok = power[t].notna()
        loss = 100 * (1 - power.loc[ok, t].sum() / meta.loc[ok, "free_kw"].sum())
        turbines[t] = {"por_pct": round(float(loss), 2), "lt_pct": round(float(loss), 2)}
    expected = meta["free_kw"] * power.notna().sum(axis=1)
    total = power.sum(axis=1)
    bins = (meta["wd"] // 5 * 5) % 360
    by_dir = []
    for b in range(0, 360, 5):
        m = bins == b
        if m.sum() < 20:
            by_dir.append({"wd_deg": float(b), "plant_por_pct": None, "energy_share_pct": 0.0, "n_records": int(m.sum())})
            continue
        by_dir.append({"wd_deg": float(b), "plant_por_pct": round(float(100 * (1 - total[m].sum() / expected[m].sum())), 2),
                       "energy_share_pct": round(float(100 * total[m].sum() / total.sum()), 2), "n_records": int(m.sum())})
    plant = 100 * (1 - total.sum() / expected.sum())
    return ({"plant_por_pct": round(float(plant), 2), "plant_lt_pct": round(float(plant), 2), "turbines": turbines,
             "by_direction": by_dir, "n_records": int(len(meta))}, meta)


def _direction_summary(wd: pd.Series) -> dict:
    """Prevailing direction and how concentrated the wind is (share of records within +/-30 deg of it)."""
    r = np.deg2rad(wd.dropna())
    mean = float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)
    diff = (wd.dropna() - mean + 180) % 360 - 180
    return {"prevailing_deg": round(mean, 0), "share_within_30_deg_pct": round(float(100 * (diff.abs() <= 30).mean()), 1)}


def _worst_bins(wl: dict, k: int = 3) -> list[dict]:
    bins = [b for b in wl["by_direction"] if b["plant_por_pct"] is not None and b["n_records"] >= 50]
    return sorted(bins, key=lambda b: -b["plant_por_pct"])[:k]


def _line_bearing(static: pd.DataFrame) -> float:
    xy = _xy(static).to_numpy()
    v = np.linalg.svd(xy - xy.mean(0), full_matrices=False)[2][0]
    return round(float(np.degrees(np.arctan2(v[0], v[1])) % 180), 0)


def brazil(farm_id: str) -> dict:
    site = bc.SITES[farm_id]
    df = bc.load_10min(bc.csv_path(farm_id))
    static = bc.load_turbine_static(farm_id)
    power = df[[f"p{i:02d}" for i in range(1, site["turbines"] + 1)]]
    power.columns = [bc.turbine_id(farm_id, i) for i in range(1, site["turbines"] + 1)]
    wl, meta = study(power, df["mast_wd"], static, site["rated_kw"], site["rotor_m"],
                     min_report=math.ceil(0.75 * site["turbines"]))
    dirs = _direction_summary(df["mast_wd"])
    line = _line_bearing(static)
    worst = _worst_bins(wl)
    cf = df[power.columns.str.replace(f"{site['prefix']}_", "p", regex=False)].clip(lower=0).mean().mean() / site["rated_kw"]
    turb = sorted(wl["turbines"].items(), key=lambda kv: kv[1]["lt_pct"])
    findings = [
        f"The trade wind is unusually steady: {dirs['share_within_30_deg_pct']:.0f}% of 10-minute records come from within 30 degrees "
        f"of {dirs['prevailing_deg']:.0f} degrees (mast vane, 100 m). The mean turbine capacity factor over the year is {100 * cf:.0f}%, "
        "high by European standards.",
        f"The turbines stand in one line along {line:.0f}/{(line + 180) % 360:.0f} degrees, so the prevailing wind crosses it at an angle and "
        f"most turbines are freestream most of the time: the farm loses {wl['plant_por_pct']:.1f}% to its own wakes "
        "(freestream mean 5-90% of rated).",
        "The worst directions are the ones closest to the line, where each turbine stands in its neighbour's wake: "
        + ", ".join(f"{b['wd_deg']:.0f} degrees ({b['plant_por_pct']:.1f}%, {b['energy_share_pct']:.1f}% of the energy)" for b in worst) + ".",
        f"Least waked: {', '.join(t for t, _ in turb[:3])}; most waked: {', '.join(t for t, _ in turb[-3:])}. Turbine positions are inferred "
        "(OpenStreetMap positions matched to the dataset's numbers by power correlation, see data_sources/brazil_coastal_scada.py), "
        "so treat single-turbine values as indicative.",
    ]
    return {
        "method": "measured_wake_study",
        "study": "line_farm",
        "title": f"Measured wake study: {site['name']}",
        "description": (f"Wake losses measured directly: for every 10-minute record the wind direction comes from the site's 100 m met mast, "
                        "the turbines with no other turbine upstream (±20°, 20 rotor diameters) are the freestream reference, and every "
                        f"turbine's energy is compared with theirs. {wl['n_records']:,} records, 2013-08 to 2014-07 (Federal University of "
                        "Santa Catarina campaign, Zenodo 1475197)."),
        "kpis": [{"label": "Farm wake loss (5-90% of rated)", "value": f"{wl['plant_por_pct']:.1f}%"},
                 {"label": f"Records within 30° of {dirs['prevailing_deg']:.0f}°", "value": f"{dirs['share_within_30_deg_pct']:.0f}%"},
                 {"label": "Mean capacity factor", "value": f"{100 * cf:.0f}%"}],
        "findings": findings,
        "method_note": ("Not OpenOA: no substation meter, curtailment or availability channels. Period of record only, so lt_pct equals "
                        "por_pct. Turbine positions are inferred, see data_sources/brazil_coastal_scada.py."),
        "aep": {"skipped": "one year of turbine SCADA with no substation meter; the wake study below is measured directly instead"},
        "wake_losses": wl,
        "wake_study": {"direction": dirs, "line_bearing_deg": line, "worst_bins": worst, "capacity_factor": round(float(cf), 3)},
    }


def _binned_ratio(x: pd.DataFrame, cols: list[str]) -> dict:
    """Mean of each column over the reference power, on vs off, with every (2 deg, 1 m/s) bin weighted
    by the smaller of its on/off counts, so both sides are compared at the same wind."""
    x = x.assign(bin=list(zip((x["dref"] // 2).astype(int), (x["wref"] // 1).astype(int))))
    g = x.groupby(["bin", "on"])[cols + ["pref"]].mean().unstack("on").dropna()
    n = x.groupby(["bin", "on"]).size().unstack("on").reindex(g.index)
    w = n.min(axis=1)
    return {on: {c: float((g[(c, on)] * w).sum() / (g[("pref", on)] * w).sum()) for c in cols} for on in (False, True)}


def smarteole() -> dict:
    df = sm.load_10min()
    static = sm.load_turbine_static("smarteole")
    ids = [sm.turbine_id(i) for i in range(1, sm.TURBINES + 1)]
    power = df[[f"p{i}" for i in range(1, sm.TURBINES + 1)]]
    power.columns = ids
    r = np.deg2rad(df[[f"d{i}" for i in SMARTEOLE_REF]])
    wd = pd.Series(np.rad2deg(np.arctan2(np.sin(r).mean(axis=1), np.cos(r).mean(axis=1))) % 360, index=df.index)
    wl, _ = study(power, wd, static, sm.RATED_KW, sm.ROTOR_D_M, min_report=sm.TURBINES)

    x = df.assign(pref=df[[f"p{i}" for i in SMARTEOLE_REF]].mean(axis=1), wref=df[[f"w{i}" for i in SMARTEOLE_REF]].mean(axis=1), dref=wd)
    x = x[(x["offset_active_6"] > 0.9) | (x["offset_active_6"] < 0.1)].assign(on=lambda d: d["offset_active_6"] > 0.9)
    x = x[x["dref"].between(*STEER_SECTOR) & x["wref"].between(*STEER_WS)]
    x = x.dropna(subset=["p5", "p6", "pref"]).assign(pair=lambda d: (d["p5"] + d["p6"]) / 2)
    ratio = _binned_ratio(x, ["p5", "p6", "pair"])
    gain = lambda rr: 100 * (rr[True]["pair"] / rr[False]["pair"] - 1)  # noqa: E731
    days = x["timestamp"].dt.date
    rng = np.random.default_rng(0)
    uniq = days.unique()
    boot = [gain(_binned_ratio(pd.concat([x[days == d] for d in rng.choice(uniq, len(uniq))]), ["p5", "p6", "pair"]))
            for _ in range(300)]
    steer = {
        "sector_deg": list(STEER_SECTOR), "ref_wind_ms": list(STEER_WS),
        "n_records_off": int((~x["on"]).sum()), "n_records_on": int(x["on"].sum()), "n_days": int(len(uniq)),
        "median_offset_deg_when_on": round(float(x.loc[x["on"], "offset_deg_6"].median()), 1),
        "smv6_pct_of_ref": {"off": round(100 * ratio[False]["p6"], 1), "on": round(100 * ratio[True]["p6"], 1)},
        "smv5_pct_of_ref": {"off": round(100 * ratio[False]["p5"], 1), "on": round(100 * ratio[True]["p5"], 1)},
        "pair_gain_pct": round(gain(ratio), 1),
        "pair_gain_90ci_pct": [round(float(np.percentile(boot, 5)), 1), round(float(np.percentile(boot, 95)), 1)],
    }
    smv5_bin = next(b for b in wl["by_direction"] if b["wd_deg"] == 205.0)  # 205-210 holds the 207.5-degree alignment
    findings = [
        f"The line runs north-south with turbines about 350 m (4.3 rotor diameters) apart, so wind along it is costly: the worst bins "
        + ", ".join(f"{b['wd_deg']:.0f} degrees ({b['plant_por_pct']:.1f}%)" for b in _worst_bins(wl)) + f"; {wl['plant_por_pct']:.1f}% over the campaign.",
        f"SMV5 stands 302 m (3.7 rotor diameters) from SMV6 at 207.5 degrees, the closest pair on the line. In that wind, with "
        f"steering off, it makes only {steer['smv5_pct_of_ref']['off']:.0f}% of what the reference turbines (SMV1, 2, 3, 7) make; "
        f"the farm loses {smv5_bin['plant_por_pct']:.1f}% in the 205-degree bin.",
        f"With SMV6 yawed off the wind (median offset {steer['median_offset_deg_when_on']:.0f} degrees), SMV6 goes from "
        f"{steer['smv6_pct_of_ref']['off']:.1f}% to {steer['smv6_pct_of_ref']['on']:.1f}% of the reference and SMV5 from "
        f"{steer['smv5_pct_of_ref']['off']:.1f}% to {steer['smv5_pct_of_ref']['on']:.1f}%: the pair changes by {steer['pair_gain_pct']:+.1f}% "
        f"(90% interval {steer['pair_gain_90ci_pct'][0]:+.1f}% to {steer['pair_gain_90ci_pct'][1]:+.1f}%, resampling {steer['n_days']} days) "
        f"in {STEER_SECTOR[0]:.0f}-{STEER_SECTOR[1]:.0f} degrees and {STEER_WS[0]:.0f}-{STEER_WS[1]:.0f} m/s.",
        (("So steering moved energy from SMV6 to SMV5 and the pair came out ahead, but " if steer["pair_gain_90ci_pct"][0] > 0 else
          "So steering moved energy from SMV6 to SMV5, but whether the pair came out ahead is not settled: the interval includes zero, and ")
         + "the record is three months with one steered turbine, and the owners removed every minute a turbine was stopped. "
           "The campaign's own analysis is Simley et al. 2021 (Wind Energ. Sci. 6, 1427)."),
    ]
    return {
        "method": "measured_wake_study",
        "study": "line_farm",
        "title": "Measured wake study and wake steering: Sole du Moulin Vieux (SMARTEOLE)",
        "description": ("Wake losses measured directly: for every 10-minute record the wind direction is the mean nacelle direction of "
                        "SMV1, 2, 3 and 7 (corrected for north alignment by the data owners), the turbines with no other turbine upstream "
                        f"(±20°, 20 rotor diameters) are the freestream reference. {wl['n_records']:,} records, 2020-02-17 to 2020-05-24."),
        "kpis": [{"label": "Farm wake loss (5-90% of rated)", "value": f"{wl['plant_por_pct']:.1f}%"},
                 {"label": f"SMV5 + SMV6 with wake steering (90% CI {steer['pair_gain_90ci_pct'][0]:+.1f} to {steer['pair_gain_90ci_pct'][1]:+.1f})", "value": f"{steer['pair_gain_pct']:+.1f}%"},
                 {"label": "SMV5 behind SMV6 (steering off)", "value": f"{steer['smv5_pct_of_ref']['off']:.0f}% of ref"}],
        "findings": findings,
        "method_note": ("Not OpenOA: no substation meter and three months only. Period of record, so lt_pct equals por_pct. Wake-steering "
                        "comparison: 10-minute records with the offset fully on vs fully off, weighted by (2°, 1 m/s) reference bins."),
        "aep": {"skipped": "three months of SCADA with stopped minutes removed by the data owners; no substation meter"},
        "wake_losses": wl,
        "turbine_notes": {"SMV6": (f"SMV6 was the wake-steering test turbine: in this campaign it was yawed off the wind on purpose "
                                   f"(up to {x['offset_deg_6'].max():.0f} degrees) to bend its wake away from SMV5, which costs it some "
                                   "output by design. A low SMV6 output here is the experiment, not a fault.")},
        "wake_study": {"wake_steering": steer, "worst_bins": _worst_bins(wl)},
    }


def build() -> dict:
    return {"pedra_do_sal": brazil("pedra_do_sal"), "beberibe": brazil("beberibe"), "smarteole": smarteole()}


def write(results: dict) -> None:
    payload = json.loads(PAYLOAD_PATH.read_text()) if PAYLOAD_PATH.exists() else {"farms": {}}
    payload.setdefault("farms", {}).update(results)
    PAYLOAD_PATH.write_text(json.dumps(payload, indent=1))


if __name__ == "__main__":
    res = build()
    write(res)
    for farm, r in res.items():
        print(f"== {farm}: plant wake loss {r['wake_losses']['plant_por_pct']}% over {r['wake_losses']['n_records']} records")
        for f in r["findings"]:
            print("  -", f)
