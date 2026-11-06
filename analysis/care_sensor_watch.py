"""Sensor-by-sensor normal-behaviour monitoring with alert rules and time to failure, on the CARE to
Compare cases (data_sources/care_scada.py): the practice condition-monitoring teams use, as
opposed to the single whole-turbine score of analysis/care_nbm.py.

Design, and the papers behind each choice (README §21; PDFs in web/papers/):

1. Data quality first (analysis/sensor_quality.py): gaps, stale, out-of-range, step and erratic
   readings are removed; sensors with too many of them are reported as suspect, not monitored.
2. Two models per sensor, because one input set cannot do both jobs:
   - Cause model, for machine health: the sensor predicted from the operating conditions only
     (wind speed, power, rotor speed, outside temperature; current value, 3-hour and 12-hour
     averages for thermal lag; power and speed squared for losses that grow with load). No other
     temperatures: Felgueira et al. 2019 showed that inputs physically close to the target heat up
     with the fault and hide it; their causal-only model predicted worst and detected best.
   - Peer model, for sensor validation: the sensor predicted from the five sensors that correlate
     with it most in normal operation (MSET's approach, Gross et al. 1997). A sensor that leaves its
     peers while they still agree with each other is a sensor fault, not a machine fault.
   Linear (ridge) models; a sensor whose cause model explains under half its variance (R² < 0.5)
   is not used for machine health.
3. Reference period per case: the benchmark's split. Fit on the first 85% of the training year,
   set each sensor's threshold on the last 15% (about 8 weeks; the 99th percentile of the absolute
   residual in normal operation), then watch the prediction window that follows, which holds the
   labelled event: 9 to 93 days, median about 2 weeks. Both wind papers ask for at least a year of
   training to cover the seasons (Peter et al. 2022, Murgia et al. 2023). A first version here
   fitted on 60% of the year and watched the last four months: healthy turbines warned as often as
   failing ones and warnings grew with time since the fit, i.e. seasonal drift. So lead times here
   are bounded by the prediction window.
4. Alert rule: a residual above the threshold for at least 30 minutes (3 rows) is one episode.
   Faults showed up as a steady trickle of 1 to 3 episodes a day, which a fixed "N per day" rule
   misses, so a sensor warns when Page's CUSUM on its daily episode count, against its own normal
   rate, reaches Wald's SPRT boundary ln((1-beta)/alpha) (the SPRT is MSET's test, Gross et al.
   1997): alpha 0.1%, beta 10%, a 4-fold rate. The fixed daily-count rule is in the grid too.
   Only rows in normal operation count.
5. From sensors to a maintenance ticket: sensor warnings are grouped by component (gearbox,
   generator, transformer, pitch, ...; keyword rules on the sensor descriptions). A component
   warns when one of its sensors warns (the single-temperature models in the papers alarm on one
   signal; 2 sensors within 3 days is in the comparison grid), and a sensor the quality checks or
   its near-duplicate twin mark as faulty that day does not count. The turbine alarm is the first
   component warning; the component with most warning sensor-days is the system's diagnosis,
   compared with the operator's root cause.
6. Time to failure: days from the first alarm to the logged fault (the end of the labelled event
   window), with the same rules run on the normal cases as the false-alarm base rate. Murgia et
   al. found lead times from weeks to years on the same kind of alarm and conclude it suits
   diagnosis more than prognosis; the output is a distribution, not a countdown per turbine.

Two stages: `compute` reads each case from the 5.5 GB archive once and caches residual summaries
under data/care_cache/ (gitignored); `build` applies the rules to the cache and writes
dashboard/data/care_sensor_watch.json. Changing a rule only needs `build`.

    python -m analysis.care_sensor_watch            # compute missing cases (~40 min) then build
    python -m analysis.care_sensor_watch --build    # rules and payload only, under a minute
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from analysis import sensor_quality as sq
from analysis.care_score import care_score, evaluate_event
from data_sources import care_scada as cs
from data_sources.edp_scada import subsystem

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "care_cache"
PAYLOAD_PATH = ROOT / "dashboard" / "data" / "care_sensor_watch.json"
CACHE_VERSION = 3

FIT_SHARE = 0.85  # of the training rows; the rest of the training year sets the thresholds
TOP_K = 5
MIN_R2 = 0.5
EWM_SPANS = (18, 72)  # rows: 3 and 12 hours
THRESHOLD_QUANTILES = (0.99, 0.999)
MIN_RUN = 3  # rows, 30 minutes
MIN_CAL_ROWS = 500
CHART_ROWS = 36  # 6-hour chart buckets
ROWS_PER_DAY = 144
ESCALATION_DAYS = 30
MODELS = ("cause", "peer")

# Operating conditions: the cause model's inputs (base sensor names per farm: wind speed, power,
# outside temperature, rotor speed).
OPERATING = {
    "A": ["wind_speed_3", "power_30", "sensor_0", "sensor_52"],
    "B": ["wind_speed_61", "power_62", "sensor_8", "sensor_25"],
    "C": ["wind_speed_236", "power_6", "sensor_7", "sensor_144"],
}

# Components from the sensor descriptions, first match wins. Names follow the subsystems used for
# the operators' root causes (data_sources/edp_scada.py), so diagnosis can be compared.
COMPONENT_RULES = [
    ("transformer", ("transformer", "transformator", "hv ")),
    ("hydraulic", ("hydraul",)),
    ("gearbox", ("gear", "planetary")),
    ("generator bearing", ("generator bearing",)),
    ("generator", ("generator", "stator", "slip ring", "split ring", "winding")),
    ("rotor or main bearing", ("rotor bearing", "main bearing", "bearing")),
    ("pitch system", ("pitch", "axis", "blade", "hub", "battery")),
    ("converter", ("converter", "inverter", "igbt", "vcs", "vcp", "choke", "dc link", "dc-link", "busbar")),
    ("yaw system", ("yaw",)),
    ("cooling", ("cooling", "cooler", "water", "fan", "aerat")),
    ("brake", ("brake",)),
    ("operating conditions", ("wind", "ambient", "outside", "speed", "rpm", "direction")),
    ("nacelle and controllers", ("cabinet", "controller", "nacelle", "tower", "top", "board")),
    ("power and grid", ("power", "voltage", "current", "grid", "reactive", "frequency", "phase")),
]
NOT_A_FAULT = {"operating conditions"}

# The rule, fixed before scoring. "cusum": Page's CUSUM on the daily episode count against the
# sensor's own normal episode rate (Poisson), alarm level from Wald's SPRT boundary ln((1-beta)/alpha).
# "count": a fixed number of episodes per day, persisting over days (kept for comparison).
RULE = {"method": "cusum", "quantile": 0.99, "rate_factor": 4.0, "alpha": 0.001, "beta": 0.1, "component_min_sensors": 1}
COMPONENT_WINDOW_DAYS = 3
DUPLICATE_R = 0.98  # a peer this close measures the same thing (e.g. three winding phases)
PEER_AGREE_SHARE = 0.5
GRID = ([{"method": "cusum", "quantile": q, "rate_factor": f, "alpha": a, "beta": 0.1, "component_min_sensors": m}
         for q in THRESHOLD_QUANTILES for f in (2.0, 4.0) for a in (0.01, 0.001, 0.0001) for m in (1, 2)]
        + [{"method": "count", "quantile": q, "episodes_per_day": e, "persist_days": p, "component_min_sensors": m}
           for q in THRESHOLD_QUANTILES for e in (2, 3) for p in (1, 2) for m in (1, 2)])


def component(description: str) -> str:
    text = f" {description.lower()} "
    return next((name for name, keys in COMPONENT_RULES if any(k in text for k in keys)), "other")


def _base(col: str) -> str:
    return col[:-4] if col.endswith("_avg") else col


def _cache_path(event_id: int) -> Path:
    return CACHE_DIR / f"{event_id}.npz"


# ---------------------------------------------------------------- stage 1: residuals per case
def _fit(feats: np.ndarray, y: np.ndarray, train: np.ndarray) -> np.ndarray | None:
    valid = ~np.isnan(feats).any(axis=1)
    train = train & valid & ~np.isnan(y)
    if train.sum() < 1000:
        return None
    mu, sd = feats[train].mean(0), feats[train].std(0)
    sd[sd < 1e-9] = 1.0
    model = Ridge(alpha=1.0).fit((feats[train] - mu) / sd, y[train])
    yhat = np.full(len(y), np.nan)
    yhat[valid] = model.predict((feats[valid] - mu) / sd)
    return yhat


def compute_case(farm: str, event_id: int, ev: pd.Series) -> None:
    df = cs.load_case(farm, event_id).sort_values("id").reset_index(drop=True)
    fd = cs.load_feature_description(farm)
    desc, unit = dict(zip(fd["sensor_name"], fd["description"])), dict(zip(fd["sensor_name"], fd["unit"]))
    sensors = cs.sensor_columns(farm, list(df.columns))["plain"]
    n = len(df)
    ts = pd.to_datetime(df["time_stamp"])
    end_ts = pd.to_datetime(ev["event_end"])
    n_train = int((df["train_test"] == "train").sum())
    fit_end, cal_end = int(FIT_SHARE * n_train), n_train
    idx = np.arange(n)
    fit_rows, cal_rows, mon_rows = idx < fit_end, (idx >= fit_end) & (idx < cal_end), idx >= cal_end
    normal = (df["status_type_id"] == 0).to_numpy()
    if not cs.FARMS[farm]["status_at_prediction"]:
        # Farm A's status comes from the failure log and covers whole event windows, so after
        # training the operating state is taken from the machine itself: producing and turning.
        p_col, r_col = (next(c for c in df.columns if _base(c) == b) for b in (OPERATING[farm][1], OPERATING[farm][3]))
        ref_ok = normal & fit_rows
        p_min, r_min = (float(np.nanquantile(df.loc[ref_ok, c], 0.01)) for c in (p_col, r_col))
        running = ((df[p_col] > max(p_min, 0)) & (df[r_col] > r_min)).to_numpy()
        normal = np.where(idx < cal_end, normal, running)

    days = np.floor((ts - end_ts).dt.total_seconds() / 86400).astype(int).to_numpy()
    mon_days = np.unique(days[mon_rows])
    d0 = int(mon_days.min())
    day_idx = days - d0
    n_days = int(mon_days.max()) - d0 + 1

    def per_day(mask, weights=None):
        return np.bincount(day_idx[mask], weights=None if weights is None else weights[mask], minlength=n_days)[:n_days]

    # 1. quality
    raw = df[sensors].to_numpy(dtype="float64")
    clean = raw.copy()
    quality, bad_days = {}, {}
    for j, s in enumerate(sensors):
        checks = sq.check_sensor(raw[:, j], fit_rows, str(unit.get(_base(s), "")))
        bad = sq.bad_mask(checks)
        clean[bad, j] = np.nan
        quality[s] = {k: int(v[mon_rows].sum()) for k, v in checks.items()}
        quality[s]["bad_share"] = round(float(bad[mon_rows].mean()), 4)
        bad_days[s] = per_day(mon_rows & bad & ~checks["gap"])
    X = pd.DataFrame(clean, columns=sensors)

    # 2. correlations in normal fit rows, and the cause-model inputs
    ref = fit_rows & normal
    Xr = X[ref]
    usable = (Xr.isna().mean() <= 0.2) & (Xr.std() > 1e-6)
    cols = [s for s in sensors if usable[s]]
    Z = ((Xr[cols] - Xr[cols].mean()) / Xr[cols].std()).fillna(0.0).to_numpy()
    C = (Z.T @ Z) / max(len(Z) - 1, 1)
    op = [c for c in cols if _base(c) in OPERATING[farm]]
    op_feats = [X[op]] + [X[op].ewm(span=s, min_periods=s // 3, ignore_na=True).mean().add_suffix(f"__ewm{s}") for s in EWM_SPANS]
    squared = [c for c in op if _base(c) in (OPERATING[farm][1], OPERATING[farm][3])]
    cause_all = pd.concat(op_feats + [(X[squared] ** 2).add_suffix("__sq")], axis=1)
    E = X.ewm(span=EWM_SPANS[0], min_periods=EWM_SPANS[0] // 3, ignore_na=True).mean()

    pred_mask = (df["train_test"] == "prediction").to_numpy()
    chart_start = fit_end
    bucket = (idx - chart_start) // CHART_ROWS
    n_buckets = int(bucket.max()) + 1
    okb = idx >= chart_start

    def chart(v, mask):
        m = mask & okb & ~np.isnan(v)
        cnt = np.bincount(bucket[m], minlength=n_buckets)
        return np.bincount(bucket[m], weights=v[m], minlength=n_buckets) / np.where(cnt, cnt, np.nan)

    meta = []
    arrays = {f"{m}_{k}": [] for m in MODELS for k in ("severity", "monitored", *(f"episodes_{q}" for q in THRESHOLD_QUANTILES),
                                                          *(f"pred_episode_{q}" for q in THRESHOLD_QUANTILES))}
    arrays.update({"bad_days": [], "chart_actual": [], "chart_cause": [], "chart_peer": []})
    for i, s in enumerate(cols):
        y = X[s].to_numpy()
        order = np.argsort(-np.abs(C[i]))
        peers = [cols[k] for k in order if cols[k] != s][:TOP_K]
        cause_cols = [c for c in cause_all.columns if not c.startswith(s)]
        preds = {"cause": _fit(cause_all[cause_cols].to_numpy(), y, ref),
                 "peer": _fit(pd.concat([X[peers], E[peers].add_suffix("__ewm")], axis=1).to_numpy(), y, ref)}
        if any(p is None for p in preds.values()):
            continue
        m_meta, ok = {}, True
        for name, yhat in preds.items():
            resid = np.abs(y - yhat)
            cal = cal_rows & normal & ~np.isnan(resid)
            if cal.sum() < MIN_CAL_ROWS:
                ok = False
                break
            r2 = 1.0 - float(np.nanmean((y[cal] - yhat[cal]) ** 2) / max(np.nanvar(y[cal]), 1e-12))
            thr = {q: float(np.quantile(resid[cal], q)) for q in THRESHOLD_QUANTILES}
            # the sensor's own normal episode rate, from the threshold period (for the CUSUM)
            base_rate = {}
            for q in THRESHOLD_QUANTILES:
                ep = sq._long_runs(cal & (resid > thr[q]), MIN_RUN)
                base_rate[q] = float((ep & ~np.concatenate([[False], ep[:-1]])).sum() + 0.5) / float(cal.sum())
            live = mon_rows & normal & ~np.isnan(resid)
            mon = per_day(live)
            arrays[f"{name}_monitored"].append(mon)
            arrays[f"{name}_severity"].append(per_day(live, resid / max(thr[RULE["quantile"]], 1e-9)) / np.maximum(mon, 1))
            for q in THRESHOLD_QUANTILES:
                in_ep = sq._long_runs(live & (resid > thr[q]), MIN_RUN)
                starts = in_ep & ~np.concatenate([[False], in_ep[:-1]])
                arrays[f"{name}_episodes_{q}"].append(per_day(starts))
                arrays[f"{name}_pred_episode_{q}"].append(in_ep[pred_mask])
            m_meta[name] = {"r2": round(r2, 3), "threshold": round(thr[RULE["quantile"]], 4),
                            **{f"rate_{q}": base_rate[q] for q in THRESHOLD_QUANTILES}}
        if not ok:
            for k in arrays:
                if k.startswith(MODELS) and len(arrays[k]) > len(meta):
                    arrays[k].pop()
            continue
        arrays["bad_days"].append(bad_days[s])
        arrays["chart_actual"].append(chart(y, normal))
        arrays["chart_cause"].append(chart(preds["cause"], normal))
        arrays["chart_peer"].append(chart(preds["peer"], normal))
        d = str(desc.get(_base(s), ""))
        meta.append({"sensor": s, "description": d, "unit": str(unit.get(_base(s), "")), "component": component(d),
                     "operating": s in op, "peers": peers,
                     "peer_r": [round(float(abs(C[i, cols.index(p)])), 4) for p in peers], **{f"{k}_{m}": v for m, mm in m_meta.items() for k, v in mm.items()},
                     "quality": quality[s]})

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    info = {"version": CACHE_VERSION, "event_id": int(event_id), "farm": farm, "n_rows": n, "fit_end": fit_end, "cal_end": cal_end,
            "first_day": d0, "n_days": n_days, "n_sensors_total": len(sensors), "n_sensors_usable": len(cols),
            "cause_inputs": [str(desc.get(_base(c), c)) for c in op],
            "chart_hours": [round(float((ts.iloc[min(chart_start + b * CHART_ROWS, n - 1)] - end_ts).total_seconds() / 3600), 1) for b in range(n_buckets)],
            "cal_start_hours": round(float((ts.iloc[fit_end] - end_ts).total_seconds() / 3600), 1),
            "mon_start_hours": round(float((ts.iloc[cal_end] - end_ts).total_seconds() / 3600), 1),
            "sensors": meta,
            "suspect": [{"sensor": s, "description": str(desc.get(_base(s), "")), **quality[s]} for s in sensors
                        if quality[s]["bad_share"] > sq.SUSPECT_SHARE]}
    out = {"pred_ids": df["id"].to_numpy()[pred_mask], "pred_normal": normal[pred_mask]}
    for k, v in arrays.items():
        if "pred_episode" in k:
            out[k] = np.packbits(np.array(v, bool).reshape(len(meta), -1), axis=1)
        elif k.startswith("chart") or k.endswith("severity"):
            out[k] = np.array(v, np.float32).reshape(len(meta), -1)
        else:
            out[k] = np.array(v, np.int16).reshape(len(meta), -1)
    np.savez_compressed(_cache_path(event_id), info=json.dumps(info), **out)


def compute(force: bool = False) -> None:
    t0 = time.time()
    for farm in cs.FARMS:
        for event_id, ev in cs.load_events(farm).iterrows():
            path = _cache_path(int(event_id))
            if not force and path.exists() and json.loads(str(np.load(path)["info"])).get("version") == CACHE_VERSION:
                continue
            compute_case(farm, int(event_id), ev)
            print(f"[{time.time() - t0:5.0f}s] farm {farm} case {event_id}", flush=True)


# ---------------------------------------------------------------- stage 2: rules and payload
def _load(event_id: int) -> tuple[dict, dict]:
    z = np.load(_cache_path(event_id))
    return json.loads(str(z["info"])), {k: z[k] for k in z.files if k != "info"}


def alert_days(episodes: np.ndarray, rule: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per sensor and day: alert day (enough episodes), and warning (an alert day with another
    alert day within the previous persist_days days; persist_days 1 means any alert day)."""
    alert = episodes >= rule["episodes_per_day"]
    warn = alert.copy()
    if rule["persist_days"] > 1:
        prev = np.zeros_like(alert)
        for lag in range(1, rule["persist_days"] + 1):
            prev[:, lag:] |= alert[:, :-lag]
        warn = alert & prev
    return alert, warn


def cusum(episodes: np.ndarray, base_rate: np.ndarray, monitored: np.ndarray, rule: dict) -> tuple[np.ndarray, np.ndarray]:
    """Page's CUSUM for a rise in a Poisson rate, per sensor and day. episodes: episodes started per
    day; base_rate: the sensor's normal episodes per monitored row (from the threshold period);
    monitored: rows watched per day. Each day adds k ln(f) - (f - 1) lambda0, the log-likelihood
    ratio of rate f x lambda0 against lambda0; the sum is floored at 0 and the sensor warns while it
    is at or above Wald's boundary ln((1 - beta) / alpha)."""
    f = rule["rate_factor"]
    lam0 = base_rate[:, None] * monitored
    inc = episodes * np.log(f) - (f - 1.0) * lam0
    h = np.log((1 - rule["beta"]) / rule["alpha"])
    stat = np.zeros_like(inc, dtype=float)
    acc = np.zeros(inc.shape[0])
    for d in range(inc.shape[1]):
        acc = np.maximum(0.0, acc + inc[:, d])
        stat[:, d] = acc
    return stat, stat >= h


def sensor_warnings(arr: dict, info: dict, model: str, rule: dict) -> tuple[np.ndarray, np.ndarray | None]:
    q = rule["quantile"]
    eps = arr[f"{model}_episodes_{q}"]
    if rule["method"] == "count":
        return alert_days(eps, rule)[1], None
    rate = np.array([m[f"rate_{q}_{model}"] for m in info["sensors"]], float)
    stat, warn = cusum(eps, rate, arr[f"{model}_monitored"], rule)
    return warn, stat


def _trailing_any(m: np.ndarray, days: int) -> np.ndarray:
    out = m.copy()
    for lag in range(1, days):
        out[..., lag:] |= m[..., :-lag]
    return out


def sensor_faults(peer_warn: np.ndarray, peers: list[list[int]], peer_r: list[list[float]],
                  bad_share: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two kinds of sensor-level finding per sensor and day.
    fault: the quality checks removed over 20% of the day's readings, or the sensor leaves its
      peer model while a near-duplicate peer (|r| > 0.98, measuring the same thing) stays quiet:
      MSET's redundant-sensor logic (Gross et al. 1997). These readings do not count for the machine.
    check: the sensor leaves its peers while under half of them warn, without a duplicate to
      settle it. A drifting sensor and a single overheating part look the same in 10-minute SCADA,
      so these still count, with a note to check the sensor first."""
    fault = bad_share > sq.SUSPECT_SHARE
    check = np.zeros_like(fault)
    for j, (pj, rj) in enumerate(zip(peers, peer_r)):
        if not pj:
            continue
        dup = [p for p, r in zip(pj, rj) if r > DUPLICATE_R]
        if dup:
            fault[j] |= peer_warn[j] & ~peer_warn[dup].any(axis=0)
        check[j] = peer_warn[j] & (peer_warn[pj].mean(axis=0) < PEER_AGREE_SHARE) & ~fault[j]
    return fault, check


def evaluate(info: dict, arr: dict, ev: pd.Series, rule: dict) -> dict:
    q = rule["quantile"]
    sensors = info["sensors"]
    S, D = len(sensors), info["n_days"]
    day_of = np.arange(D) + info["first_day"]
    upto = day_of <= -1  # whole days before the logged fault
    if S == 0:
        empty = np.zeros((0, D), bool)
        res = evaluate_event(info["event_id"], ev["event_label"], arr["pred_ids"], np.zeros(len(arr["pred_ids"]), bool),
                             np.ones(len(arr["pred_ids"]), bool), int(ev["event_start_id"]), int(ev["event_end_id"]))
        return {"first_alarm_day": None, "warn": empty, "fault": empty, "check": empty, "stat": None, "component_days": {}, "alarm": np.zeros(D, bool),
                "day_of": day_of, "care": res, "eligible": np.zeros(0, bool)}
    name_to_j = {m["sensor"]: j for j, m in enumerate(sensors)}
    peers = [[name_to_j[p] for p in m["peers"] if p in name_to_j] for m in sensors]
    peer_r = [[r for p, r in zip(m["peers"], m["peer_r"]) if p in name_to_j] for m in sensors]
    mon = np.maximum(arr["cause_monitored"], 1)
    peer_warn, _ = sensor_warnings(arr, info, "peer", rule)
    fault, check = sensor_faults(peer_warn, peers, peer_r, arr["bad_days"] / mon)
    eligible = np.array([m["r2_cause"] >= MIN_R2 and m["quality"]["bad_share"] <= sq.SUSPECT_SHARE
                         and m["component"] not in NOT_A_FAULT and not m["operating"] for m in sensors], bool)
    warn, stat = sensor_warnings(arr, info, "cause", rule)
    warn &= eligible[:, None] & ~fault
    recent = _trailing_any(warn, COMPONENT_WINDOW_DAYS)
    comps = sorted({m["component"] for m, e in zip(sensors, eligible) if e})
    component_days = {}
    for c in comps:
        members = np.array([m["component"] == c for m in sensors]) & eligible
        need = min(rule["component_min_sensors"], int(members.sum()))
        on = (recent[members].sum(axis=0) >= need) & warn[members].any(axis=0)
        if on.any():
            component_days[c] = on
    alarm = np.logical_or.reduce(list(component_days.values())) if component_days else np.zeros(D, bool)
    alarm_days = np.flatnonzero(alarm & upto)
    first = int(day_of[alarm_days[0]]) if len(alarm_days) else None
    # CARE flags: a prediction row is flagged on an alarm day when a sensor of an alarmed
    # component is in an episode
    ids, n_pred = arr["pred_ids"], len(arr["pred_ids"])
    ep_rows = np.unpackbits(arr[f"cause_pred_episode_{q}"], axis=1, count=n_pred).astype(bool)
    end_id = int(ev["event_end_id"])
    pd_idx = np.clip(np.floor((ids - end_id) / ROWS_PER_DAY).astype(int) - info["first_day"], 0, D - 1)
    in_alarm = np.zeros((S, D), bool)
    for c, on in component_days.items():
        members = np.array([m["component"] == c for m in sensors]) & eligible
        in_alarm[members] |= on[None, :] & warn[members]
    flags = (ep_rows & in_alarm[:, pd_idx]).any(axis=0)
    normal = arr["pred_normal"] if cs.FARMS[info["farm"]]["status_at_prediction"] else np.ones(n_pred, bool)
    res = evaluate_event(info["event_id"], ev["event_label"], ids, flags, normal, int(ev["event_start_id"]), end_id)
    return {"first_alarm_day": first, "warn": warn, "fault": fault, "check": check, "stat": stat, "component_days": component_days, "alarm": alarm,
            "day_of": day_of, "care": res, "eligible": eligible}


def diagnosis_matches(diagnosis: str | None, operator: str | None) -> bool | None:
    if not diagnosis or not operator or operator == "other":
        return None
    a, b = diagnosis.split()[0], operator.split()[0]
    return diagnosis == operator or a == b


def _summarise(rows: list[dict]) -> dict:
    out = {}
    for farm in [*cs.FARMS, "all"]:
        sel = [r for r in rows if farm == "all" or r["farm"] == farm]
        an = [r for r in sel if r["label"] == "anomaly"]
        no = [r for r in sel if r["label"] == "normal"]
        out[farm] = {"faults": len(an), "faults_alarmed": sum(r["first_alarm_day"] is not None for r in an),
                     "normal_cases": len(no), "normal_with_alarm": sum(r["first_alarm_day"] is not None for r in no),
                     "normal_monitored_days": int(sum(r["monitored_days"] for r in no)),
                     "normal_alarm_days": int(sum(r["alarm_days"] for r in no))}
    return out


def build() -> dict:
    events = {int(i): ev for farm in cs.FARMS for i, ev in cs.load_events(farm).iterrows()}
    loaded = {i: _load(i) for i in sorted(events) if _cache_path(i).exists()}

    def run(rule):
        rows, results = [], []
        for i, (info, arr) in loaded.items():
            r = evaluate(info, arr, events[i], rule)
            upto = r["day_of"] <= -1
            results.append(r)
            rows.append({"event_id": i, "farm": info["farm"], "label": events[i]["event_label"],
                         "first_alarm_day": r["first_alarm_day"], "monitored_days": int(upto.sum()),
                         "alarm_days": int((r["alarm"] & upto).sum())})
        scores = {f: care_score([r["care"] for r, row in zip(results, rows) if f == "all" or row["farm"] == f]) for f in [*cs.FARMS, "all"]}
        return rows, results, scores

    rows, results, scores = run(RULE)
    sensitivity = []
    for g in GRID:
        g_rows, _, g_scores = run(g)
        s = _summarise(g_rows)["all"]
        sensitivity.append({**g, "care_all": g_scores["all"]["care"], "faults_alarmed": s["faults_alarmed"],
                            "normal_with_alarm": s["normal_with_alarm"],
                            "false_alarm_days_per_turbine_month": round(30 * s["normal_alarm_days"] / max(s["normal_monitored_days"], 1), 2)})

    cases_out, ttf, esc_an, esc_no, diag = {}, [], {}, {}, []
    for row, r in zip(rows, results):
        i = row["event_id"]
        info, arr = loaded[i]
        ev = events[i]
        sensors, warn, fault, day_of = info["sensors"], r["warn"], r["fault"], r["day_of"]
        upto = day_of <= -1
        ok = r["eligible"]
        op_desc = None if pd.isna(ev["event_description"]) else ev["event_description"]
        operator_sub = subsystem(op_desc)
        # escalation: share of watched sensors in warning per day, aligned on the event end
        if ok.any():
            share = warn[ok].sum(0) / ok.sum()
            target = esc_an if ev["event_label"] == "anomaly" else esc_no
            for d, v in zip(day_of, share):
                if d <= -1:
                    target.setdefault(int(d), []).append(float(v))
        comp_rows = []
        for c, on in r["component_days"].items():
            first = np.flatnonzero(on & upto)
            if len(first):
                members = [j for j, m in enumerate(sensors) if m["component"] == c and ok[j]]
                comp_rows.append({"component": c, "first_day": int(day_of[first[0]]), "days": int((on & upto).sum()),
                                  "sensor_warning_days": int(warn[members][:, upto].sum())})
        comp_rows.sort(key=lambda x: (x["first_day"], -x["sensor_warning_days"]))
        diagnosis = max(comp_rows, key=lambda x: x["sensor_warning_days"])["component"] if comp_rows else None
        match = diagnosis_matches(diagnosis, operator_sub) if ev["event_label"] == "anomaly" else None
        if ev["event_label"] == "anomaly" and diagnosis:
            diag.append({"event_id": i, "operator": operator_sub, "diagnosis": diagnosis, "match": match})
        sens = []
        for j, m in enumerate(sensors):
            w = np.flatnonzero(warn[j] & upto)
            if len(w):
                sens.append({"sensor": m["sensor"], "description": m["description"], "component": m["component"], "unit": m["unit"],
                             "r2": m["r2_cause"], "first_warning_day": int(day_of[w[0]]), "warning_days": int(len(w)), "index": j})
        sens.sort(key=lambda s: (-s["warning_days"], s["first_warning_day"]))
        charts = []
        for s in sens[:3]:
            j = s.pop("index")
            charts.append({"sensor": s["sensor"], "description": s["description"], "unit": s["unit"],
                           "threshold": sensors[j]["threshold_cause"],
                           "actual": [None if np.isnan(v) else round(float(v), 3) for v in arr["chart_actual"][j]],
                           "predicted": [None if np.isnan(v) else round(float(v), 3) for v in arr["chart_cause"][j]]})
        for s in sens[3:]:
            s.pop("index")
        sensor_faults_out = []
        for j, m in enumerate(sensors):
            fd = np.flatnonzero(fault[j] & upto)
            if len(fd):
                sensor_faults_out.append({"sensor": m["sensor"], "description": m["description"], "days": int(len(fd)),
                                          "first_day": int(day_of[fd[0]]),
                                          "peers": [next((x["description"] for x in sensors if x["sensor"] == p), p) for p in m["peers"][:3]]})
        sensor_faults_out.sort(key=lambda x: -x["days"])
        sensor_checks = []
        for j, m in enumerate(sensors):
            cd = np.flatnonzero(r["check"][j] & upto)
            if len(cd):
                sensor_checks.append({"sensor": m["sensor"], "description": m["description"], "days": int(len(cd)),
                                      "first_day": int(day_of[cd[0]]), "also_machine_warning": bool((warn[j] & upto).any()),
                                      "peers": [next((x["description"] for x in sensors if x["sensor"] == p), p) for p in m["peers"][:3]]})
        sensor_checks.sort(key=lambda x: -x["days"])
        lead = -row["first_alarm_day"] if row["first_alarm_day"] is not None and ev["event_label"] == "anomaly" else None
        if ev["event_label"] == "anomaly" and lead is not None:
            ttf.append({"event_id": i, "farm": info["farm"], "subsystem": operator_sub or "not stated", "lead_days": lead})
        q_rows = [{"sensor": m["sensor"], "description": m["description"], **{k: m["quality"][k] for k in sq.CHECKS}, "bad_share": m["quality"]["bad_share"]}
                  for m in sensors if m["quality"]["bad_share"] > 0.01]
        q_rows.sort(key=lambda x: -x["bad_share"])
        care = r["care"]
        cases_out[str(i)] = {
            "event_id": i, "farm": info["farm"], "label": ev["event_label"], "description": op_desc, "operator_subsystem": operator_sub,
            "first_alarm_day": row["first_alarm_day"], "lead_days": lead, "diagnosis": diagnosis, "diagnosis_matches": match,
            "components": comp_rows, "monitor_start_hours": info["mon_start_hours"], "threshold_start_hours": info["cal_start_hours"],
            "sensors_total": info["n_sensors_total"], "sensors_modelled": len(sensors), "sensors_watched": int(ok.sum()),
            "cause_inputs": info["cause_inputs"],
            "suspect_sensors": info["suspect"], "quality_issues": q_rows[:12], "sensor_faults": sensor_faults_out[:10], "sensor_checks": sensor_checks[:10],
            "warned_sensors": sens[:10], "warned_sensor_count": len(sens),
            "daily": {"day": [int(d) for d in day_of[upto]], "sensors_in_warning": [int(v) for v in warn[:, upto].sum(0)],
                      "sensor_faults": [int(v) for v in fault[:, upto].sum(0)], "turbine_alarm": [bool(v) for v in r["alarm"][upto]]},
            "chart_hours": info["chart_hours"], "charts": charts,
            "care": {"detected": bool(care.detected), "max_criticality": care.max_criticality,
                     "lead_hours": round(care.lead_rows / 6, 1) if care.lead_rows is not None else None},
        }

    ttf.sort(key=lambda x: x["lead_days"])
    leads = np.array([t["lead_days"] for t in ttf], float)
    horizon = int(leads.max()) if len(leads) else 0
    survival = [{"days": d, "share_still_ahead": round(float((leads >= d).mean()), 3)} for d in range(0, horizon + 1)] if len(leads) else []
    by_sub = {}
    for t in ttf:
        by_sub.setdefault(t["subsystem"], []).append(t["lead_days"])
    rng = range(-ESCALATION_DAYS, 0)
    esc = {"days": list(rng),
           "faults": [round(float(np.mean(esc_an[d])), 4) if d in esc_an else None for d in rng],
           "normal": [round(float(np.mean(esc_no[d])), 4) if d in esc_no else None for d in rng],
           "faults_n": [len(esc_an.get(d, [])) for d in rng], "normal_n": [len(esc_no.get(d, [])) for d in rng]}
    judged = [d for d in diag if d["match"] is not None]
    pct = lambda a: None if not len(leads) else float(np.quantile(leads, a))  # noqa: E731
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": {
            "quality_checks": list(sq.CHECKS), "fit_share": FIT_SHARE, "top_k_peers": TOP_K, "min_r2": MIN_R2,
            "min_run_minutes": MIN_RUN * 10, "rule": RULE, "component_window_days": COMPONENT_WINDOW_DAYS,
            "text": ("Two linear models per sensor. The cause model predicts it from the operating conditions only (wind speed, power, "
                     "rotor speed, outside temperature, with 3- and 12-hour averages for thermal lag), so a failing part cannot hide "
                     "behind neighbouring sensors that heat up with it. The peer model predicts it from its 5 best-correlated sensors "
                     "and checks the sensor itself. Fitted on the first 85% of the training year, threshold = 99th percentile of the "
                     "absolute residual on the rest, then the case's prediction window is watched. An episode is 30 minutes above the "
                     "threshold. A sensor warns when its episodes run above its own normal rate: a CUSUM adds up each day's "
                     f"evidence for a {RULE['rate_factor']:.0f}-fold rate and warns at Wald's boundary for a {RULE['alpha']:.1%} false-alarm "
                     f"and {RULE['beta']:.0%} missed-alarm risk, so a steady trickle of episodes counts as much as a burst. A component (gearbox, generator, transformer, ...) warns when "
                     f"{RULE['component_min_sensors']} of its sensors warn within {COMPONENT_WINDOW_DAYS} days; a sensor that leaves "
                     "its near-duplicate twin, or fails the quality checks, does not count. The first component warning is the turbine alarm."),
        },
        "summary": _summarise(rows),
        "care_scores": scores,
        "pca_reference": {"care_all": 0.581, "note": "the whole-turbine PCA detector on the same cases (analysis/care_nbm.py)"},
        "diagnosis": {"alarmed_faults": len(diag), "judged": len(judged), "matches": sum(d["match"] for d in judged), "cases": diag},
        "time_to_failure": {"alarmed_faults": ttf, "median_lead_days": pct(0.5), "p25_lead_days": pct(0.25), "p75_lead_days": pct(0.75),
                            "survival": survival,
                            "by_subsystem": [{"subsystem": k, "faults_alarmed": len(v), "median_lead_days": float(np.median(v)), "min_lead_days": int(min(v))}
                                             for k, v in sorted(by_sub.items(), key=lambda kv: -len(kv[1]))],
                            "escalation": esc},
        "sensitivity": sensitivity,
        "cases": cases_out,
    }


if __name__ == "__main__":
    if "--build" not in sys.argv:
        compute(force="--force" in sys.argv)
    payload = build()
    PAYLOAD_PATH.write_text(json.dumps(payload, separators=(",", ":")))
    print(json.dumps({k: payload[k] for k in ("summary",)}, indent=1))
    print(json.dumps({f: {k: v for k, v in s.items() if k in ("care", "anomalies_detected", "false_alarms")} for f, s in payload["care_scores"].items()}))
    t = payload["time_to_failure"]
    print("lead days median", t["median_lead_days"], "p25", t["p25_lead_days"], "p75", t["p75_lead_days"])
    d = payload["diagnosis"]
    print("diagnosis", d["matches"], "of", d["judged"], "judged;", d["alarmed_faults"], "alarmed faults")
    for g in payload["sensitivity"]:
        print(g)
    print(f"wrote {PAYLOAD_PATH} ({PAYLOAD_PATH.stat().st_size / 1e6:.2f} MB)")
