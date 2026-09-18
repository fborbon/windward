"""FastAPI service wrapping the LangGraph agent, plus the live dashboard that replaced the
Claude Artifact demo — this now serves real forecasts, real Betz-limit diagnostics, and a
real LLM narration/Q&A over the agent's own findings, all against production models on this
box, not a sandboxed viewer-billed environment.
"""
import json
import math
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from data_sources.farms import FARMS
from forecasting.predict import predict_production
from schemas.models import ForecastRequest, ForecastResult

app = FastAPI(title="Windward")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
INSPECTION_PHOTO = Path(__file__).resolve().parent.parent / "data" / "sample_inspection_photos" / "rotorblatt_inspection.jpg"
WIND_PREDICTION_PAYLOAD = Path(__file__).resolve().parent.parent / "dashboard" / "data" / "wind_prediction_payload.json"

_analysis_cache: dict[str, dict] = {}
_ANALYSIS_TTL_S = 3600  # SCADA history is static 2016 data; just avoids re-running Bedrock calls on every viewer


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/farms")
def list_farms():
    return [
        {
            "farm_id": f.farm_id,
            "name": f.name,
            "lat": f.lat,
            "lon": f.lon,
            "turbine_count": len(f.turbine_ids),
            "rated_power_kw": f.rated_power_kw,
            "rotor_diameter_m": f.rotor_diameter_m,
            "total_capacity_mw": round(f.rated_power_kw * len(f.turbine_ids) / 1000, 1),
            "source_doi": f.source_doi,
        }
        for f in FARMS.values()
    ]


@app.post("/forecast", response_model=ForecastResult)
def forecast(request: ForecastRequest):
    return predict_production(farm_id=request.farm_id, horizon_hours=request.horizon_hours)


@app.get("/wind-prediction")
def wind_prediction():
    """Precomputed naive-to-foundation-model technique comparison (wind_prediction/), served
    static - see wind_prediction/export.py for why this doesn't need to run per-request."""
    if not WIND_PREDICTION_PAYLOAD.exists():
        raise HTTPException(404, "wind-prediction payload not exported yet - run `python -m wind_prediction.export`")
    return json.loads(WIND_PREDICTION_PAYLOAD.read_text())


@app.get("/wind-prediction/live-weather")
def wind_prediction_live_weather(farm_id: str = "kelmarsh"):
    """The one genuinely live-refreshing piece of this project: a real Open-Meteo forecast call,
    made fresh on every request (not cached/precomputed like the SCADA-bound analysis above) -
    see README.md §13 for why the SCADA production data can't be live the same way."""
    if farm_id not in FARMS:
        raise HTTPException(404, f"unknown farm_id '{farm_id}'")
    from data_sources.meteo_client import fetch_forecast

    farm = FARMS[farm_id]
    points = fetch_forecast(farm.lat, farm.lon, hours=48)
    return {
        "farm_id": farm_id,
        "fetched_at": pd.Timestamp.utcnow().isoformat(),
        "source": "Open-Meteo forecast API (live)",
        "points": [
            {
                "timestamp": p.timestamp.isoformat(),
                "wind_speed_ms": p.wind_speed_ms,
                "wind_direction_deg": p.wind_direction_deg,
                "temperature_c": p.temperature_c,
            }
            for p in points
        ],
    }


def _clean(v):
    """Round-trip pandas/numpy scalars through plain python + drop NaN so FastAPI's default
    JSON encoder (which chokes on numpy types and produces invalid `NaN` tokens) never sees them."""
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    f = float(v)
    return None if math.isnan(f) else f


def _run_analysis(farm_id: str) -> dict:
    if farm_id not in FARMS:
        raise HTTPException(404, f"unknown farm_id '{farm_id}'")

    cached = _analysis_cache.get(farm_id)
    if cached and time.time() - cached["_cached_at"] < _ANALYSIS_TTL_S:
        return cached

    from agents.graph import build_graph, run

    result = run(build_graph(), farm_id)

    comparison: pd.DataFrame = result["comparison"]
    recent = comparison.tail(24 * 14)  # last 2 weeks of the SCADA period, enough for a chart without shipping the whole year
    efficiency: pd.DataFrame = result["efficiency_summary"]

    power_curves = {}
    for turbine_id, curve in result["power_curves"].items():
        power_curves[turbine_id] = [
            {"wind_speed_bin": _clean(idx), "mean_power_kw": _clean(row["mean_power_kw"]), "sample_count": int(row["sample_count"])}
            for idx, row in curve.iterrows()
            if not pd.isna(row["mean_power_kw"])
        ]

    payload = {
        "_cached_at": time.time(),
        "farm_id": farm_id,
        "comparison_series": [
            {
                "timestamp": ts.isoformat(),
                "actual_mw": _clean(row["actual_mw"]),
                "predicted_mw": _clean(row["predicted_mw"]),
            }
            for ts, row in recent.iterrows()
        ],
        "comparison_summary": {
            "mean_actual_mw": _clean(comparison["actual_mw"].mean()),
            "mean_predicted_mw": _clean(comparison["predicted_mw"].mean()),
            "mean_abs_pct_error": _clean((comparison["actual_mw"] - comparison["predicted_mw"]).abs().div(comparison["predicted_mw"].replace(0, float("nan"))).mean()),
            "hours_analyzed": int(len(comparison)),
        },
        "efficiency_summary": [
            {
                "turbine_id": turbine_id,
                "capacity_factor": _clean(row["capacity_factor"]),
                "peak_cp": _clean(row["peak_cp"]),
                "betz_limit": _clean(row["betz_limit"]),
                "hours_observed": int(row["hours_observed"]),
                "over_betz": bool(row["peak_cp"] > row["betz_limit"]) if not pd.isna(row["peak_cp"]) else False,
            }
            for turbine_id, row in efficiency.iterrows()
        ],
        "power_curves": power_curves,
        "wind_rose": [
            {"compass": row["compass"], "direction_deg": _clean(row["direction_deg"]), "energy_kwh": _clean(row["energy_kwh"])}
            for _, row in result["wind_rose"].iterrows()
        ],
        "wind_speed_distribution": result["wind_speed_distribution"],
        "anomalies": result["anomalies"],
        "recommendation": result["recommendation"],
        "explanation": result["explanation"],
        "inspection_results": result["inspection_results"],
        "rag_context": result.get("rag_context", []),
    }
    _analysis_cache[farm_id] = payload
    try:
        _save_dynamo_session(farm_id, payload)
    except Exception as e:  # DynamoDB is a record of the run, not on the critical path to serving it
        print(f"WARN: session write failed for {farm_id}: {e}")
    return payload


def _save_dynamo_session(farm_id: str, payload: dict):
    """Records that this analysis ran — a real DynamoDB write per fresh (non-cached) request,
    not a batch export step; see storage/dynamo_session_store.py."""
    from storage.dynamo_session_store import save_session

    eff = payload["efficiency_summary"]
    save_session(
        f"{farm_id}-{payload['_cached_at']}",
        {
            "farm_id": farm_id,
            "run_at": pd.Timestamp.utcfromtimestamp(payload["_cached_at"]).isoformat(),
            "mean_capacity_factor": sum(r["capacity_factor"] for r in eff) / len(eff),
            "anomaly_count": len(payload["anomalies"]),
            "recommendation": payload["recommendation"],
        },
    )


@app.get("/analysis/{farm_id}")
def analysis(farm_id: str):
    return {k: v for k, v in _run_analysis(farm_id).items() if k != "_cached_at"}


@app.get("/analysis/{farm_id}/inspection-photo")
def inspection_photo(farm_id: str):
    if farm_id not in FARMS:
        raise HTTPException(404, f"unknown farm_id '{farm_id}'")
    return FileResponse(INSPECTION_PHOTO, media_type="image/jpeg")


class AskRequest(BaseModel):
    question: str


@app.post("/analysis/{farm_id}/ask")
def ask(farm_id: str, request: AskRequest):
    if not request.question.strip() or len(request.question) > 500:
        raise HTTPException(400, "question must be 1-500 characters")

    from agents.qa_agent import answer_question

    analysis = _run_analysis(farm_id)
    return answer_question(farm_id, request.question.strip(), analysis)


# --- EDP Wind Farm A: real labeled fault case studies, diagnosis/RAG only, no forecasting ---
# (no disclosed coordinates or rated power to forecast against — see data_sources/edp_scada.py).
# Deliberately not part of data_sources.farms.FARMS or the /farms, /forecast, /analysis routes
# above, which all assume a Farm with real lat/lon/rated_power.


@app.get("/edp/events")
def edp_events():
    from data_sources.edp_scada import load_events

    events = load_events()
    return [
        {
            "event_id": int(event_id),
            "turbine_id": str(row["asset"]),
            "label": row["event_label"],
            "description": row["event_description"] if pd.notna(row["event_description"]) else None,
            "duration_hours": round(row["duration_hours"], 1),
        }
        for event_id, row in events.iterrows()
    ]


@app.get("/edp/events/{event_id}")
def edp_event_detail(event_id: int):
    from analysis.efficiency import binned_power_curve
    from data_sources.edp_scada import STATUS_LABELS, load_event_series, load_events

    events = load_events()
    if event_id not in events.index:
        raise HTTPException(404, f"unknown event_id {event_id}")
    event = events.loc[event_id]
    series = load_event_series(event_id)
    curve = binned_power_curve(series, power_col="power_frac")
    status_counts = series["status_type_id"].value_counts().sort_index()

    return {
        "event_id": event_id,
        "turbine_id": str(event["asset"]),
        "label": event["event_label"],
        "description": event["event_description"] if pd.notna(event["event_description"]) else None,
        "duration_hours": round(event["duration_hours"], 1),
        "rows_observed": len(series),
        "power_curve": [
            {"wind_speed_bin": _clean(idx), "mean_power_frac": _clean(row["mean_power_kw"]), "sample_count": int(row["sample_count"])}
            for idx, row in curve.iterrows()
            if not pd.isna(row["mean_power_kw"])
        ],
        "status_breakdown": {STATUS_LABELS.get(int(sid), str(sid)): int(count) for sid, count in status_counts.items()},
    }


class EdpAskRequest(BaseModel):
    question: str


@app.post("/edp/ask")
def edp_ask(request: EdpAskRequest):
    if not request.question.strip() or len(request.question) > 500:
        raise HTTPException(400, "question must be 1-500 characters")

    from agents.llm_router import complete
    from rag.edp_retriever import get_retriever

    docs = get_retriever().invoke(request.question.strip())
    context = "\n".join(f"- {d.page_content}" for d in docs) or "(no matching case studies found)"
    prompt = (
        "You are the Windward analysis agent for EDP Wind Farm A, a real, anonymized wind "
        "turbine fault-detection benchmark (22 labeled real case studies, diagnosis/RAG only — "
        "no forecasting, since the source farm's location and rated power are undisclosed). "
        "Answer the visitor's question using ONLY the case studies below; if none are relevant, "
        "say so plainly rather than guessing. Keep the answer under 120 words, plain English.\n\n"
        f"Case studies:\n{context}\n\nQuestion: {request.question.strip()}"
    )
    response = complete([{"role": "user", "content": prompt}])
    return {"answer": response.choices[0].message.content, "sources": [d.metadata.get("source", "?") for d in docs]}


# Mounted last: exact-path routes above always win; everything else (including "/") falls
# through to the dashboard's static files.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="dashboard")
