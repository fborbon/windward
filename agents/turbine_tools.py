"""Per-turbine tools for the /ask agent (agents/qa_agent.py), so visitors can ask about one
turbine ("when was Kelmarsh 3 last maintained?") or compare turbines ("which turbine loses the
most to wakes?") instead of only about the farm as a whole.

Everything comes from data the rest of the project already computes: the farm's live analysis
payload (capacity factor, peak Cp, QC, anomalies), the offline OpenOA payload (wake losses, yaw
misalignment), and the real SCADA status/alarm logs.

What counts as maintenance depends on what each export actually records:
- Kelmarsh/Penmanshiel (Greenbyte): status events whose IEC 61400-26 category is "Scheduled
  Maintenance" (manual on-site stops, manual brake), and "Forced outage" for unplanned stops.
- Hill of Towie (RES historian): no IEC category, and the alarm-code lookup only names one
  maintenance-type stop, 3130 "Pitch lubrication". The lookup's other stopping codes are all
  weather (low/high wind, icing) or cable untwisting, and every other code is undescribed, so
  forced outages can't be identified from this export and are reported as unavailable, not 0.
"""
import re

import pandas as pd

from data_sources.farms import FARMS, loader_for

RANK_METRICS = {
    # metric: (description, higher_is_better)
    "capacity_factor": ("share of rated output actually produced, every hour counted", True),
    "peak_cp": ("95th-percentile power coefficient on QC-cleaned hours (aerodynamic efficiency)", True),
    "wake_loss_lt_pct": ("long-term energy lost to neighbors' wakes (OpenOA WakeLosses)", False),
    "abs_yaw_misalignment_deg": ("magnitude of estimated static yaw offset (OpenOA, screening only)", False),
    "qc_removed_share": ("share of hours removed by SCADA QC (stopped in wind, derated, frozen sensor)", False),
    "forced_outage_hours": ("hours of unplanned stops in the status log", False),
    "maintenance_events": ("number of maintenance stops in the status log", False),
}

HOT_MAINTENANCE_CODES = {3130}  # Pitch lubrication

_events_cache: dict[str, pd.DataFrame] = {}


def _events(farm_id: str) -> pd.DataFrame:
    if farm_id not in _events_cache:
        farm = FARMS[farm_id]
        ev = loader_for(farm).load_status_events(farm.scada_zips)
        # an event still open at the end of the export has a placeholder instead of an end time
        ev["start"] = pd.to_datetime(ev["start"], errors="coerce")
        ev["end"] = pd.to_datetime(ev["end"], errors="coerce")
        ev["hours"] = (ev["end"] - ev["start"]).dt.total_seconds() / 3600
        if farm.data_source == "hill_of_towie":
            ev["is_maintenance"] = ev["code"].isin(HOT_MAINTENANCE_CODES)
            ev["is_forced_outage"] = False
        else:
            ev["is_maintenance"] = ev["iec_category"] == "Scheduled Maintenance"
            ev["is_forced_outage"] = ev["iec_category"] == "Forced outage"
        _events_cache[farm_id] = ev
    return _events_cache[farm_id]


def resolve_turbine_id(farm_id: str, text: str) -> str | None:
    """'Kelmarsh 3', 'kelmarsh_3', 'turbine 3', 'T03', '3' -> 'Kelmarsh_3'."""
    ids = FARMS[farm_id].turbine_ids
    if text in ids:
        return text
    norm = text.strip().lower().replace(" ", "_")
    for tid in ids:
        if tid.lower() == norm:
            return tid
    digits = re.findall(r"\d+", text)
    if digits:
        n = int(digits[-1])
        for tid in ids:
            if int(re.findall(r"\d+", tid)[-1]) == n:
                return tid
    return None


def _event_record(row) -> dict:
    return {
        "start": row["start"].isoformat() if pd.notna(row["start"]) else None,
        "end": row["end"].isoformat() if pd.notna(row["end"]) else None,
        "hours": round(float(row["hours"]), 2) if pd.notna(row["hours"]) else None,
        "message": row["message"],
        "code": str(row["code"]),
    }


def _union_hours(events: pd.DataFrame) -> float:
    """Downtime hours with overlapping events counted once: one stop is often logged as several
    concurrent alarms (Kelmarsh_1's three 'Overload generator fan 1/2/3' events are the same
    218.5 h), so a plain sum would triple-count it."""
    spans = events.dropna(subset=["start", "end"]).sort_values("start")[["start", "end"]].values
    total, cur_start, cur_end = 0.0, None, None
    for start, end in spans:
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += (cur_end - cur_start) / pd.Timedelta(hours=1)
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += (cur_end - cur_start) / pd.Timedelta(hours=1)
    return round(float(total), 1)


def _maintenance_stats(farm_id: str, turbine_id: str) -> dict:
    ev = _events(farm_id)
    t = ev[ev["turbine_id"] == turbine_id]
    maint = t[t["is_maintenance"]].sort_values("start")
    forced = t[t["is_forced_outage"]]
    top_faults = forced.groupby("message")["hours"].agg(["count", "sum"]).sort_values("sum", ascending=False).head(3)
    outages_known = FARMS[farm_id].data_source != "hill_of_towie"
    return {
        "log_period": f"{ev['start'].min():%Y-%m-%d} to {ev['start'].max():%Y-%m-%d}",
        "maintenance_events": int(len(maint)),
        "maintenance_hours": _union_hours(maint),
        "last_maintenance": _event_record(maint.iloc[-1]) if len(maint) else None,
        "forced_outage_events": int(len(forced)) if outages_known else None,
        "forced_outage_hours": _union_hours(forced) if outages_known else None,
        "note": "hours count overlapping events once; per-cause hours can overlap each other (one stop often raises several alarms), so don't add them up",
        "top_forced_outage_causes": [
            {"message": msg or "(no description in this export)", "events": int(r["count"]), "hours": round(float(r["sum"]), 1)}
            for msg, r in top_faults.iterrows()
        ] if outages_known else "not identifiable from this farm's alarm export",
        "maintenance_definition": "Hill of Towie: alarm 3130 'Pitch lubrication' only (the export doesn't label maintenance)"
        if FARMS[farm_id].data_source == "hill_of_towie" else "IEC 61400-26 category 'Scheduled Maintenance'",
    }


def _operational(farm_id: str) -> dict:
    from operational_assessment.payload import farm_operational, yaw_statuses

    op = farm_operational(farm_id) or {}
    return {
        "wake": ((op.get("wake_losses") or {}).get("turbines") or {}),
        "yaw": ((op.get("yaw_misalignment") or {}).get("turbines") or {}),
        "yaw_status": yaw_statuses(op),
    }


def turbine_details(farm_id: str, turbine: str, analysis: dict) -> dict | str:
    turbine_id = resolve_turbine_id(farm_id, turbine)
    if turbine_id is None:
        return f"unknown turbine {turbine!r}; this farm's turbines are {', '.join(FARMS[farm_id].turbine_ids)}"
    eff = next((e for e in analysis.get("efficiency_summary", []) if e["turbine_id"] == turbine_id), {})
    op = _operational(farm_id)
    fleet_cf = [e["capacity_factor"] for e in analysis.get("efficiency_summary", []) if e.get("capacity_factor") is not None]
    yaw = op["yaw"].get(turbine_id) or {}
    return {
        "turbine_id": turbine_id,
        "capacity_factor": eff.get("capacity_factor"),
        "fleet_mean_capacity_factor": sum(fleet_cf) / len(fleet_cf) if fleet_cf else None,
        "capacity_factor_rank": (sorted(fleet_cf, reverse=True).index(eff["capacity_factor"]) + 1) if eff.get("capacity_factor") in fleet_cf else None,
        "fleet_size": len(fleet_cf),
        "peak_cp": eff.get("peak_cp"),
        "over_betz_limit": eff.get("over_betz"),
        "qc_removed_share": ((analysis.get("qc_summary") or {}).get("turbines") or {}).get(turbine_id),
        "anomalies": [a for a in analysis.get("anomalies", []) if a.get("turbine_id") == turbine_id],
        "wake_loss": op["wake"].get(turbine_id),
        "yaw_misalignment": {"mean_deg": yaw.get("mean_deg"), "ci95_deg": yaw.get("ci95_deg"),
                             "status": op["yaw_status"].get(turbine_id)} if yaw else None,
        "status_log": _maintenance_stats(farm_id, turbine_id),
    }


def rank_turbines(farm_id: str, metric: str, analysis: dict) -> dict | str:
    if metric not in RANK_METRICS:
        return f"unknown metric {metric!r}; use one of {', '.join(RANK_METRICS)}"
    description, higher_is_better = RANK_METRICS[metric]
    op = _operational(farm_id)
    qc = (analysis.get("qc_summary") or {}).get("turbines") or {}
    eff = {e["turbine_id"]: e for e in analysis.get("efficiency_summary", [])}
    rows = []
    for tid in FARMS[farm_id].turbine_ids:
        if metric in ("capacity_factor", "peak_cp"):
            value = (eff.get(tid) or {}).get(metric)
        elif metric == "wake_loss_lt_pct":
            value = (op["wake"].get(tid) or {}).get("lt_pct")
        elif metric == "abs_yaw_misalignment_deg":
            mean = (op["yaw"].get(tid) or {}).get("mean_deg")
            value = abs(mean) if mean is not None and op["yaw_status"].get(tid) != "inconclusive" else None
        elif metric == "qc_removed_share":
            value = (qc.get(tid) or {}).get("qc_flag")
        else:
            value = _maintenance_stats(farm_id, tid)[metric]
        if value is not None:
            rows.append({"turbine_id": tid, metric: round(float(value), 4)})
    rows.sort(key=lambda r: r[metric], reverse=True)
    return {
        "metric": metric, "description": description,
        "order": "highest first; " + ("higher is better, so best first" if higher_is_better else "higher is worse, so most affected first"),
        "ranking": rows,
        "note": "yaw estimates marked inconclusive are left out" if metric == "abs_yaw_misalignment_deg" else None,
    }
