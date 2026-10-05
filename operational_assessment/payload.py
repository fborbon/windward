"""Service-side reader for dashboard/data/operational_payload.json (written offline by
operational_assessment/run_openoa.py). Imports nothing from OpenOA, so the live service keeps
its own scikit-learn/scipy versions."""
import json
from pathlib import Path

import numpy as np

PAYLOAD_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "data" / "operational_payload.json"

# A static yaw offset is flagged when its magnitude is at least this AND OpenOA's Monte Carlo
# 95% interval excludes zero. ~3 deg is roughly where the cos^2 rule of thumb starts costing a
# few tenths of a percent of below-rated energy, the range a vane recalibration is worth checking.
YAW_FLAG_DEG = 3.0
# OpenOA estimates the offset separately in each wind-speed bin (5-8 m/s) and averages them. When
# those per-bin estimates disagree by more than this, the cosine peak isn't identifiable from the
# data (on Kelmarsh the 5 and 8 m/s fits land 20-40 deg out, at or past OpenOA's +/-25 deg vane
# window), so the turbine is reported as inconclusive rather than flagged.
YAW_MAX_BIN_SPREAD_DEG = 10.0

_cache: dict = {"mtime": None, "data": None}


def load_payload() -> dict | None:
    if not PAYLOAD_PATH.exists():
        return None
    mtime = PAYLOAD_PATH.stat().st_mtime
    if _cache["mtime"] != mtime:
        _cache["data"] = json.loads(PAYLOAD_PATH.read_text())
        _cache["mtime"] = mtime
    return _cache["data"]


def farm_operational(farm_id: str) -> dict | None:
    payload = load_payload()
    if not payload:
        return None
    farm = payload.get("farms", {}).get(farm_id)
    if farm is None:
        return None
    return {**farm, "generated_at": payload.get("generated_at"), "openoa_version": payload.get("openoa_version"),
            "references": payload.get("references")}


def yaw_status(t: dict) -> str:
    """'flag' | 'ok' | 'inconclusive' for one turbine's StaticYawMisalignment result."""
    mean, ci, spread = t.get("mean_deg"), t.get("ci95_deg") or [None, None], t.get("by_ws_spread_deg")
    if mean is None or None in ci or spread is None or spread > YAW_MAX_BIN_SPREAD_DEG:
        return "inconclusive"
    return "flag" if abs(mean) >= YAW_FLAG_DEG and (ci[0] > 0 or ci[1] < 0) else "ok"


def yaw_statuses(operational: dict | None) -> dict:
    yaw = (operational or {}).get("yaw_misalignment") or {}
    return {tid: yaw_status(t) for tid, t in (yaw.get("turbines") or {}).items()}


def yaw_flags(operational: dict | None) -> list[dict]:
    """Turbines whose estimated static yaw misalignment is material, statistically
    distinguishable from zero, and consistent across wind-speed bins, with a cos^2
    rule-of-thumb below-rated power loss."""
    yaw = (operational or {}).get("yaw_misalignment") or {}
    flags = []
    for turbine_id, t in (yaw.get("turbines") or {}).items():
        if yaw_status(t) == "flag":
            mean, ci = t["mean_deg"], t["ci95_deg"]
            flags.append({
                "turbine_id": turbine_id,
                "yaw_misalignment_deg": mean,
                "ci95_deg": ci,
                "approx_below_rated_loss_pct": round((1 - np.cos(np.deg2rad(mean)) ** 2) * 100, 2),
            })
    return sorted(flags, key=lambda f: -abs(f["yaw_misalignment_deg"]))


def turbine_wake_loss_pct(operational: dict | None, turbine_id: str) -> float | None:
    turbines = ((operational or {}).get("wake_losses") or {}).get("turbines") or {}
    return (turbines.get(turbine_id) or {}).get("lt_pct")


def compact_summary(operational: dict | None) -> dict | None:
    """The payload minus its long per-direction/per-bin arrays: small enough to hand to the
    LLM as a tool result."""
    if not operational:
        return None
    out = {k: v for k, v in operational.items() if k not in ("wake_losses", "yaw_misalignment", "references")}
    # Give the LLM the interval ready-made: asked for "uncertainty", it otherwise does its own
    # (wrong) arithmetic from the coefficient of variation.
    aep = dict(operational.get("aep") or {})
    for method in ("monthly_linear", "daily_gam_temperature"):
        m = aep.get(method)
        if m and m.get("p50_gwh") is not None and m.get("std_gwh") is not None:
            aep[method] = {**m, "interval_95pct_gwh": [round(m["mean_gwh"] - 1.96 * m["std_gwh"], 2),
                                                        round(m["mean_gwh"] + 1.96 * m["std_gwh"], 2)]}
    out["aep"] = aep
    wl = operational.get("wake_losses") or {}
    out["wake_losses"] = {k: v for k, v in wl.items() if k != "by_direction"}
    ym = operational.get("yaw_misalignment") or {}
    statuses = yaw_statuses(operational)
    out["yaw_misalignment"] = {
        "caveat": ym.get("caveat"),
        "not_identifiable": ym.get("not_identifiable"),
        "vane_std_deg": ym.get("vane_std_deg"),
        "turbines": {tid: {"mean_deg": t.get("mean_deg"), "ci95_deg": t.get("ci95_deg"),
                           "by_ws_spread_deg": t.get("by_ws_spread_deg"), "status": statuses.get(tid)}
                     for tid, t in (ym.get("turbines") or {}).items()},
        "status_rule": f"flag if |offset| >= {YAW_FLAG_DEG} deg and 95% CI excludes 0; inconclusive if per-wind-speed-bin estimates spread > {YAW_MAX_BIN_SPREAD_DEG} deg",
    }
    return out
