"""Case studies from the CARE to Compare benchmark (Zenodo doi:10.5281/zenodo.15846963, CC BY-SA
4.0; Gück et al. 2024, "CARE to Compare: A Real-World Benchmark Dataset for Early Fault Detection
in Wind Turbine Data", MDPI Data 9(12):138), as served by the API's /edp/* routes and the
dashboard's case-study tab. The module keeps its first name: the tab started with Wind Farm A
alone, EDP's onshore farm in Portugal (the original edp.com "Wind Farm 1").

Since 2026-11 it covers all three of the benchmark's farms, 95 cases on 36 turbines:
- A: EDP onshore, Portugal, 5 turbines, 22 cases
- B: offshore, Germany, 9 turbines, 15 cases
- C: offshore, Germany, 22 turbines, 58 cases

All anonymised: no coordinates, no rated power, shifted timestamps, rescaled power. So no
forecasting, and diagnosis only. Each case is one turbine's year of 10-minute SCADA ending in one
labelled event (an anomaly leading up to a fault, often with the operator's root cause, or normal
behaviour).

This module reads dashboard/data/care_payload.json, which analysis/care_nbm.py writes offline
from the 5.5 GB archive (data_sources/care_scada.py): per case the power curve, the status
breakdown and the normal-behaviour detector's result. The API host needs no raw CARE data.
"""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

PAYLOAD_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "data" / "care_payload.json"

# From the dataset's README ("Event Data"): status 0 and 2 are normal operation.
STATUS_LABELS = {
    0: "Normal Operation",
    1: "Derated Operation",
    2: "Idling",
    3: "Service",
    4: "Downtime",
    5: "Other",
}


@lru_cache(maxsize=1)
def _payload(mtime: float) -> dict:
    return json.loads(PAYLOAD_PATH.read_text())


def load_payload() -> dict:
    if not PAYLOAD_PATH.exists():
        raise FileNotFoundError(f"{PAYLOAD_PATH} missing - run `python -m analysis.care_nbm` (needs the CARE archive)")
    return _payload(PAYLOAD_PATH.stat().st_mtime)


def load_events(farm: str | None = None) -> pd.DataFrame:
    """One row per case: farm, turbine (anonymised asset id), label, operator's root cause where
    given, event-window duration, and whether the detector caught it."""
    rows = []
    for c in load_payload()["cases"]:
        if farm and c["farm"] != farm:
            continue
        d = c["detector"]
        rows.append({"event_id": c["event_id"], "farm": c["farm"], "asset": c["turbine_id"], "event_label": c["label"],
                     "event_description": c["description"], "duration_hours": c["duration_hours"],
                     "detected": d["detected"], "lead_hours": d["lead_hours"]})
    return pd.DataFrame(rows).set_index("event_id")


def load_case(event_id: int) -> dict | None:
    return next((c for c in load_payload()["cases"] if c["event_id"] == event_id), None)


def detector_summary() -> dict:
    p = load_payload()
    return {"generated_at": p["generated_at"], "farms": p["farms"], **p["detector"]}
