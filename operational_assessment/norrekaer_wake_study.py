"""Measured wake study for Nørrekær Enge II: what the farm loses to its own wakes, by turbine and
by wind direction, straight from the data, and how that compares with Risø's 1993 report
(Højstrup et al., Risø-I-684(EN), https://gitlab.windenergy.dtu.dk/fair-data/winddata-revamp/winddata-documentation/-/raw/master/wind_farm/norre_m2/norrekaer_windfarm.pdf).

The other farms get wake losses from OpenOA (operational_assessment/run_openoa.py), which needs a
substation meter, reanalysis and years of data. Nørrekær Enge has what OpenOA lacks instead: a
hub-height met mast upstream of the farm (M1) with a wind vane, a second one deep inside it (M2),
and turbines laid out on a clean grid on flat reclaimed seabed. So the freestream reference can be
chosen geometrically for every 10-minute record:

1. Wind direction: M1's 31 m vane (M2's 34 m vane when M1 is missing).
2. Freestream turbines for that direction: no other turbine upstream within a +/-20 degree
   sector and 20 rotor diameters (560 m). Rows are 6.3 D apart, so this is never borderline.
3. Each turbine's efficiency = its power / the mean power of that record's freestream turbines,
   kept only when the freestream mean is 5-90% of rated (above that every turbine is at rated
   and wakes cost nothing; below it the ratios are noise) and at least 3 freestream turbines and
   35 of 40 turbines report.
4. Wake loss = 1 - sum(turbine energy) / sum(freestream-equivalent energy), per turbine over all
   its records, per 5-degree direction bin, and for the plant.

Written into dashboard/data/operational_payload.json under farms.norrekaer_enge, in the same
`wake_losses` shape OpenOA produces, so the /analysis graph's wake-vs-fault context and the /ask
agent's tools use it unchanged:

    python -m operational_assessment.norrekaer_wake_study
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from data_sources import norrekaer_scada as ns
from operational_assessment.payload import PAYLOAD_PATH

FARM_ID = "norrekaer_enge"
SECTOR_HALF_DEG = 20.0
WAKE_RANGE_D = 20.0
MIN_FREE = 3
MIN_REPORTING = 35
FREE_POWER_BAND = (0.05, 0.90)


def _positions() -> pd.DataFrame:
    utm = pd.read_csv(ns.UTM_PATH).set_index("turbine")
    return utm.loc[ns.TURBINES, ["easting", "northing"]]


def freestream_sets(step_deg: int = 1) -> dict[int, list[str]]:
    """{direction: turbines with no upstream turbine within the sector and range}, all 42 in the
    geometry (A1/F6 shed wakes even though they log no power)."""
    pos = _positions().to_numpy()
    dx = pos[None, :, 0] - pos[:, None, 0]  # [t, u]: vector from turbine t to turbine u
    dy = pos[None, :, 1] - pos[:, None, 1]
    dist = np.hypot(dx, dy)
    bearing = np.degrees(np.arctan2(dx, dy)) % 360
    near = (dist > 0) & (dist <= WAKE_RANGE_D * ns.ROTOR_D_M)
    out = {}
    for wd in range(0, 360, step_deg):
        diff = (bearing - wd + 180) % 360 - 180  # u seen from t, vs the direction the wind comes from
        waked = (near & (np.abs(diff) <= SECTOR_HALF_DEG)).any(axis=1)
        out[wd] = [t for t, w in zip(ns.TURBINES, waked) if not w]
    return out


def _records() -> pd.DataFrame:
    df = ns.load_10min()
    df["wd"] = df["d31_1"].fillna(df["d34_2"])
    df["ws_ref"] = df["s31_1"].fillna(df["s31_2"])
    return df.dropna(subset=["wd"])


def efficiencies() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-record power matrix of the 40 turbines, per-record freestream mean kW + direction)."""
    df = _records()
    free = freestream_sets()
    power = df[[f"{t}_pow" for t in ns.POWER_TURBINES]].clip(lower=0)
    power.columns = ns.POWER_TURBINES
    wd_int = df["wd"].round().astype(int) % 360
    ref = np.full(len(df), np.nan)
    n_free = np.zeros(len(df), dtype=int)
    vals = power.to_numpy()
    col = {t: i for i, t in enumerate(ns.POWER_TURBINES)}
    for i, wd in enumerate(wd_int.to_numpy()):
        idx = [col[t] for t in free[wd] if t in col]
        row = vals[i, idx]
        row = row[~np.isnan(row)]
        n_free[i] = len(row)
        if len(row):
            ref[i] = row.mean()
    meta = pd.DataFrame({"timestamp": df["timestamp"].to_numpy(), "wd": df["wd"].to_numpy(),
                         "ws_ref": df["ws_ref"].to_numpy(), "free_kw": ref, "n_free": n_free,
                         "n_report": power.notna().sum(axis=1).to_numpy()}, index=df.index)
    lo, hi = (v * ns.RATED_KW for v in FREE_POWER_BAND)
    keep = meta["free_kw"].between(lo, hi) & (meta["n_free"] >= MIN_FREE) & (meta["n_report"] >= MIN_REPORTING)
    return power[keep], meta[keep]


def wake_losses(power: pd.DataFrame, meta: pd.DataFrame) -> dict:
    ref = meta["free_kw"]
    turbines = {}
    for t in ns.POWER_TURBINES:
        ok = power[t].notna()
        loss = 100 * (1 - power.loc[ok, t].sum() / ref[ok].sum())
        turbines[ns.turbine_id(t)] = {"por_pct": round(float(loss), 2), "lt_pct": round(float(loss), 2)}
    expected = ref * power.notna().sum(axis=1)
    plant = 100 * (1 - power.sum(axis=1).sum() / expected.sum())
    bins = (meta["wd"] // 5 * 5) % 360
    by_dir = []
    total_energy = power.sum(axis=1).sum()
    for b in range(0, 360, 5):
        m = bins == b
        if m.sum() < 20:
            by_dir.append({"wd_deg": float(b), "plant_por_pct": None, "energy_share_pct": 0.0, "n_records": int(m.sum())})
            continue
        e = power[m].sum(axis=1).sum()
        by_dir.append({"wd_deg": float(b), "plant_por_pct": round(float(100 * (1 - e / expected[m].sum())), 2),
                       "energy_share_pct": round(float(100 * e / total_energy), 2), "n_records": int(m.sum())})
    return {"plant_por_pct": round(float(plant), 2), "plant_lt_pct": round(float(plant), 2),
            "turbines": turbines, "by_direction": by_dir, "n_records": int(len(meta))}


def _row_profile(df: pd.DataFrame, turbines: list[str], wd_center: float, half: float = 7.5,
                 ws: tuple[float, float] = (6.0, 9.0)) -> dict:
    """Nacelle wind speed along a line of turbines, relative to the first one, for wind blowing
    along the line: the same cases as the report's figures 10-12. Nacelle wind, not power,
    because the front turbine of row A (A1) logs no power."""
    diff = (df["wd"] - wd_center + 180) % 360 - 180
    sel = df[(diff.abs() <= half) & df["ws_ref"].between(*ws)]
    cols = [f"{t}_wsn" for t in turbines]
    sel = sel.dropna(subset=cols)
    if len(sel) < 10:
        return {"n_records": int(len(sel))}
    mean = sel[cols].mean()
    rel = (mean / mean.iloc[0] * 100).round(1)
    power_cols = [f"{t}_pow" for t in turbines if t not in ns.NO_POWER]
    return {"direction_deg": wd_center, "n_records": int(len(sel)), "mean_ref_ws_ms": round(float(sel["ws_ref"].mean()), 2),
            "nacelle_wind_pct_of_first": dict(zip(turbines, rel.tolist())),
            "mean_power_kw": {c[:-4]: round(float(sel[c].mean()), 1) for c in power_cols}}


def report_comparisons(power: pd.DataFrame, meta: pd.DataFrame, wl: dict) -> dict:
    df = _records()
    out = {}
    # Report fig. 11 / 12: wind straight along row A, from the south (166 deg) and the north (346 deg)
    row_a = [f"A{i}" for i in range(1, 8)]
    out["row_a_from_south"] = _row_profile(df, row_a, 166.0)
    out["row_a_from_north"] = _row_profile(df, list(reversed(row_a)), 346.0)
    # Report fig. 10: wind along line 1 (A1 B1 C1 D1 E1 F1, ~258 deg)
    out["line_1_from_west"] = _row_profile(df, ["A1", "B1", "C1", "D1", "E1", "F1"], 258.0)
    # Report section 6: coastal fetch. Row A is the front row for westerly wind; the report found
    # >17% difference between its ends in sector 10 (255-285 deg), A7 (nearest the fjord) highest.
    diff = (df["wd"] - 270 + 180) % 360 - 180
    west = df[(diff.abs() <= 15) & df["ws_ref"].between(5.0, 11.0)]
    a_pow = west[[f"A{i}_pow" for i in range(2, 8)]].dropna()
    rel = (a_pow.mean() / a_pow.mean().mean() * 100).round(1)
    out["row_a_coastal_fetch_255_285"] = {"n_records": int(len(a_pow)), "power_pct_of_row_mean": {c[:2]: v for c, v in rel.items()},
                                          "a7_vs_a2_pct": round(float(100 * (a_pow["A7_pow"].mean() / a_pow["A2_pow"].mean() - 1)), 1)}
    # The same south-to-north rise appears in the *unwaked* front row whichever side the wind comes
    # from (row A in westerly wind, row F in easterly), so it's a site gradient, not a wake: terrain
    # up to 40 m and villages just south of the farm, open water to the north (report section 2).
    # It is also the likely reason the report's row-A-from-the-south power "recovers" after turbine
    # 4 (fig. 11), which the authors couldn't explain.
    grad = {}
    for label, center, row in (("westerly_front_row_A", 270.0, "A"), ("easterly_front_row_F", 90.0, "F")):
        d = (df["wd"] - center + 180) % 360 - 180
        cols = [f"{row}{i}_pow" for i in range(1, 8) if f"{row}{i}" not in ns.NO_POWER]
        s = df[(d.abs() <= 15) & df["ws_ref"].between(5.0, 11.0)][cols].dropna()
        m = s.mean()
        grad[label] = {"n_records": int(len(s)), "power_pct_of_row_mean_south_to_north": {c[:2]: round(float(v), 1) for c, v in (m / m.mean() * 100).items()}}
    out["south_north_site_gradient"] = grad
    # Report fig. 13: park output dips (wake alignments) at about 218, 258 and 293 deg
    by = pd.DataFrame([b for b in wl["by_direction"] if b["plant_por_pct"] is not None]).set_index("wd_deg")
    sw = by.loc[200:300, "plant_por_pct"]
    peaks = [float(i) for i in sw.index if sw[i] >= sw.rolling(3, center=True, min_periods=1).max()[i] and sw[i] > sw.median()]
    out["wake_peaks_200_300_deg"] = {"bins_deg": peaks, "loss_pct": [round(float(sw[p]), 1) for p in peaks],
                                     "report_fig13_dips_deg": [218, 258, 293]}
    # The rows run 166/346 deg: wind straight down all six rows at once is the worst case
    # (the report's row-A figures 11-12), but rare. Report the two row-aligned bins with their counts.
    bd = {b["wd_deg"]: b for b in wl["by_direction"]}
    out["row_aligned_wake_loss"] = {f"{int(d)}": {"loss_pct": bd[d]["plant_por_pct"], "n_records": bd[d]["n_records"],
                                                  "energy_share_pct": bd[d]["energy_share_pct"]} for d in (165.0, 345.0)}
    # Mast M2 sits deep inside the farm: its hub-height wind vs M1's, by direction
    both = df.dropna(subset=["s31_1", "s31_2"])
    both = both[both["s31_1"].between(5, 12)]
    d = (both["d31_1"] // 30 * 30) % 360
    out["m2_vs_m1_wind_pct_by_30deg"] = {int(k): round(float(v), 1) for k, v in
                                          (both.groupby(d)["s31_2"].mean() / both.groupby(d)["s31_1"].mean() * 100).items()}
    # Yaw misalignment channel: per-turbine median, a vane/yaw calibration screen
    ym = df[[f"{t}_ym" for t in ns.POWER_TURBINES]]
    ym = ym.where(ym.abs() < 45)
    med = ym.median().round(1)
    out["yaw_misalignment_median_deg"] = {"farm_median": round(float(med.median()), 1),
                                          "largest_abs": {c[:-3]: float(v) for c, v in med.abs().sort_values(ascending=False).head(5).items()}}
    return out


def build() -> dict:
    power, meta = efficiencies()
    wl = wake_losses(power, meta)
    findings = report_comparisons(power, meta, wl)
    return {
        "method": "measured_wake_study",
        "method_note": ("Not OpenOA: wake losses measured directly against geometrically freestream turbines per "
                        "10-minute record (wind direction from met mast M1), see operational_assessment/norrekaer_wake_study.py. "
                        "Period of record only (1991-12 to 1993-06, Risø-selected runs), so lt_pct equals por_pct. "
                        "Per-turbine values also carry a site gradient (output rises ~25-50% from the south to the north end "
                        "of a row in both westerly and easterly wind, see wake_study.south_north_site_gradient): a negative "
                        "value at a northern row end is that gradient, not a turbine beating freestream."),
        "aep": {"skipped": "no substation meter, curtailment or availability channels, and the turbine channels only exist "
                           "for Risø-selected runs; the wake study below is measured directly instead"},
        "wake_losses": wl,
        "wake_study": findings,
    }


def write(result: dict) -> None:
    payload = json.loads(PAYLOAD_PATH.read_text()) if PAYLOAD_PATH.exists() else {"farms": {}}
    payload.setdefault("farms", {})[FARM_ID] = result
    PAYLOAD_PATH.write_text(json.dumps(payload, indent=1))


if __name__ == "__main__":
    res = build()
    write(res)
    print(json.dumps({k: v for k, v in res["wake_losses"].items() if k not in ("by_direction", "turbines")}))
    print(json.dumps(res["wake_study"], indent=1))
