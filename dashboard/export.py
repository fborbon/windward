"""Runs the agent graph for every registered farm and exports a combined dashboard payload.
"""
import json
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

import config
from agents.graph import build_graph, run as run_graph
from data_sources.farms import FARMS
from data_sources.greenbyte_scada import load_status_events
from forecasting.train import model_name
from storage.dynamo_session_store import save_session

EXPORT_DIR = Path(__file__).resolve().parent / "data"


def _latest_model_metrics(farm_id: str) -> dict:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    version = client.get_latest_versions(model_name(farm_id))[0]
    run = client.get_run(version.run_id)
    return {"mae_mw": run.data.metrics.get("mae_mw"), "r2": run.data.metrics.get("r2")}


def _farm_payload(farm_id: str, result: dict) -> dict:
    farm = FARMS[farm_id]

    comparison = result["comparison"]
    daily = comparison.resample("1D").mean(numeric_only=True).dropna()
    daily_out = [
        {"date": d.strftime("%Y-%m-%d"), "actual": round(r.actual_mw, 3), "predicted": round(r.predicted_mw, 3)}
        for d, r in daily.iterrows()
    ]

    curves_out = {}
    for turbine_id, curve in result["power_curves"].items():
        c = curve.dropna(subset=["mean_power_kw"])
        curves_out[turbine_id] = [{"ws": round(ws, 2), "kw": round(row.mean_power_kw, 1)} for ws, row in c.iterrows()]

    eff_out = result["efficiency_summary"].reset_index().round(4).to_dict(orient="records")

    anomalies = result["anomalies"]
    anomalies_out = [{k: (None if pd.isna(v) else v) for k, v in a.items()} for a in anomalies]

    events = load_status_events(farm.scada_zips)
    interventions = events[events["status"].isin(["Stop", "Warning"])].groupby("turbine_id").size()
    maintenance_out = [{"turbine_id": t, "interventions": int(n)} for t, n in interventions.items()]

    return {
        "daily": daily_out,
        "curves": curves_out,
        "efficiency": eff_out,
        "anomalies": anomalies_out,
        "maintenance": maintenance_out,
        "explanation": result["explanation"],
        "recommendation": result["recommendation"],
        "meta": {
            "farm": farm.name,
            "farm_id": farm.farm_id,
            "turbines": len(farm.turbine_ids),
            "rated_mw_each": farm.rated_power_kw / 1000,
            "rotor_m": farm.rotor_diameter_m,
            "total_mw": farm.rated_power_kw * len(farm.turbine_ids) / 1000,
            "source_doi": farm.source_doi,
            "model_metrics": _latest_model_metrics(farm_id),
        },
    }


def _save_dynamo_session(farm_id: str, farm_payload: dict, run_at: str):
    eff = farm_payload["efficiency"]
    save_session(
        f"{farm_id}-{run_at}",
        {
            "farm_id": farm_id,
            "run_at": run_at,
            "mean_capacity_factor": sum(r["capacity_factor"] for r in eff) / len(eff),
            "anomaly_count": len(farm_payload["anomalies"]),
            "recommendation": farm_payload["recommendation"],
            "model_metrics": farm_payload["meta"]["model_metrics"],
        },
    )


def export():
    EXPORT_DIR.mkdir(exist_ok=True)
    graph = build_graph()
    run_at = pd.Timestamp.utcnow().isoformat()

    payload = {"farms": {}}
    for farm_id in FARMS:
        print(f"running graph for {farm_id}...")
        result = run_graph(graph, farm_id)
        payload["farms"][farm_id] = _farm_payload(farm_id, result)
        _save_dynamo_session(farm_id, payload["farms"][farm_id], run_at)

    out_path = EXPORT_DIR / "dashboard_payload.json"
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"Exported {out_path} ({out_path.stat().st_size / 1024:.0f} KB), farms: {list(payload['farms'])}")
    return payload


if __name__ == "__main__":
    export()
