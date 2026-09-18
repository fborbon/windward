"""LangGraph workflow: ingest -> forecast -> diagnose -> {rag, multimodal} -> investigate
-> (conditional) recommend | recommend_retrain -> explain.

This is the "architect generative AI workflows" piece — a declarative, checkpointable state
machine rather than hand-rolled orchestration (contrast with global-news-agent's custom
orchestrator). `investigate` is a real LangGraph conditional-routing node: it decides, from
the farm's actual recent-vs-overall forecast error trend and the registered model's real age
(forecasting.registry.latest_model_info), whether to recommend retraining or fall through to
the normal worst-turbine recommendation — not a fixed linear/parallel run every time.

Runs in analysis/backtest mode over each farm's real SCADA period (the only period we have
real production data for): ingest real weather+production, forecast with the farm's
registered model, diagnose actual-vs-predicted + per-turbine wind-extraction efficiency,
then have the LLM narrate it.

Invoke with an initial state containing farm_id, e.g. graph.invoke({"farm_id": "kelmarsh"}).
"""
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from agents.llm_router import complete
from analysis.efficiency import (
    actual_vs_predicted,
    air_density_kg_m3,
    binned_power_curve,
    fit_power_curve_displacement,
    neighbor_underperformance,
    scada_reanalysis_wind_check,
    smooth_power_curve,
    turbine_efficiency_summary,
    wind_rose_energy_kwh,
    wind_speed_power_distribution,
)
from data_sources.farms import FARMS, loader_for
from forecasting.features import FEATURE_COLUMNS, TARGET_COLUMN
from forecasting.pipeline import build_training_frame
from forecasting.registry import load_latest_model
from forecasting.train import model_name

# Demo-scale thresholds for the retrain decision: the last week's mean absolute pct error
# would need to run at least 25% hotter than the full analysis period's average, AND the
# registered model would need to be older than this, before investigate_node recommends
# retraining rather than the normal worst-turbine action.
RECENT_ERROR_INCREASE_RATIO = 1.25
RETRAIN_AGE_DAYS_THRESHOLD = 90

# Demo-scale thresholds for the two per-turbine/per-hour anomaly checks added alongside the
# original farm-forecast-deviation and over-Betz checks — see analysis/efficiency.py.
NEIGHBOR_UNDERPERFORMANCE_PCT_THRESHOLD = 0.05  # flag a turbine if >5% of its hours are low vs neighbors
REANALYSIS_MISMATCH_PCT_THRESHOLD = 0.02  # flag if >2% of hours diverge >5 m/s from Open-Meteo


class WindwardState(TypedDict, total=False):
    farm_id: str
    feature_frame: pd.DataFrame
    turbine_hourly: pd.DataFrame
    comparison: pd.DataFrame  # actual vs predicted, farm-level
    power_curves: dict
    smooth_power_curves: dict
    efficiency_summary: pd.DataFrame
    anomalies: list
    data_quality: dict
    wind_rose: pd.DataFrame
    wind_speed_distribution: dict
    rag_context: list
    inspection_results: list
    investigation: dict
    recommendation: dict
    explanation: str


def ingest_node(state: WindwardState) -> dict:
    farm = FARMS[state["farm_id"]]
    return {
        "feature_frame": build_training_frame(farm),
        "turbine_hourly": loader_for(farm).load_turbine_hourly_series(farm.scada_zips),
    }


def forecast_node(state: WindwardState) -> dict:
    df = state["feature_frame"]
    model = load_latest_model(model_name(state["farm_id"]))
    predicted = pd.Series(model.predict(df[FEATURE_COLUMNS]), index=df.index)
    return {"comparison": actual_vs_predicted(df[TARGET_COLUMN], predicted)}


def diagnose_node(state: WindwardState) -> dict:
    farm = FARMS[state["farm_id"]]
    turbine_hourly = state["turbine_hourly"]
    feature_frame = state["feature_frame"]

    power_curves = {}
    smooth_power_curves = {}
    for turbine_id, g in turbine_hourly.groupby("turbine_id"):
        curve = binned_power_curve(g)
        power_curves[turbine_id] = curve
        smooth_power_curves[turbine_id] = smooth_power_curve(g, curve.index.values)
    farm_mean_curve = binned_power_curve(turbine_hourly)  # pooled across all turbines — the displacement-fit reference

    # Real per-hour air density (ideal gas law on the farm's actual weather) rather than the
    # sea-level constant — a real UK-winter farm can genuinely diverge from 1.225 kg/m3
    # enough to matter for Cp/Betz-limit readings.
    air_density = air_density_kg_m3(feature_frame["temperature_c"], feature_frame["pressure_hpa"])
    efficiency_summary = turbine_efficiency_summary(
        turbine_hourly, farm.rated_power_kw, farm.rotor_diameter_m, air_density_by_timestamp=air_density
    )

    comparison = state["comparison"]
    bad_hours = comparison[(comparison["pct_of_predicted"] < 0.5) | (comparison["pct_of_predicted"] > 1.5)]
    anomalies = [
        {"type": "farm_forecast_deviation", "count": len(bad_hours), "pct_of_hours": len(bad_hours) / len(comparison)}
    ]

    over_betz = efficiency_summary[efficiency_summary["peak_cp"] > efficiency_summary["betz_limit"]]
    for turbine_id in over_betz.index:
        t = turbine_hourly[turbine_hourly["turbine_id"] == turbine_id]
        displacement_ms = fit_power_curve_displacement(farm_mean_curve, t["wind_speed_ms"], t["power_kw"])
        anomalies.append({
            "type": "anemometer_calibration_suspect", "turbine_id": turbine_id,
            "displacement_ms": round(displacement_ms, 3),
            "detail": (
                f"peak Cp exceeds the Betz limit even after correcting for real air density — the turbine's power "
                f"curve is displaced {displacement_ms:+.2f} m/s from the farm-mean curve, consistent with a real "
                f"nacelle anemometer bias (it sits downstream of the spinning rotor), not over-unity extraction"
            ),
        })

    # Per-turbine, per-hour spatial anomaly: compare each turbine against its k nearest
    # neighbors' concurrent production, not just the farm-level aggregate deviation above.
    static = loader_for(farm).load_turbine_static(farm.farm_id)
    neighbor_flags = neighbor_underperformance(turbine_hourly, static)
    total_hours = turbine_hourly["timestamp"].nunique()
    for turbine_id, flagged_hours in neighbor_flags.items():
        pct = flagged_hours / total_hours if total_hours else 0
        if pct > NEIGHBOR_UNDERPERFORMANCE_PCT_THRESHOLD:
            anomalies.append({
                "type": "underperforms_neighbors", "turbine_id": turbine_id,
                "flagged_hours": flagged_hours, "pct_of_hours": round(pct, 4),
                "detail": f"produced meaningfully less than its nearest neighbor turbines in {flagged_hours} hours ({pct:.1%}) while those neighbors were themselves producing",
            })

    # Cross-reference the turbine SCADA anemometers against an independent source (Open-Meteo
    # reanalysis) — a real QC pattern (compare on-site sensor vs. an independent reference),
    # adapted since there's no second on-site mast here.
    data_quality = scada_reanalysis_wind_check(turbine_hourly, feature_frame)
    if data_quality.get("pct_hours_diverging_gt_5ms", 0) > REANALYSIS_MISMATCH_PCT_THRESHOLD:
        anomalies.append({
            "type": "reanalysis_mismatch",
            "pct_of_hours": data_quality["pct_hours_diverging_gt_5ms"],
            "detail": "farm-mean SCADA wind speed diverges >5 m/s from the independent Open-Meteo reanalysis in a meaningful fraction of hours — possible data alignment or sensor issue",
        })

    total_capacity_mw = farm.rated_power_kw * len(farm.turbine_ids) / 1000
    wind_rose = wind_rose_energy_kwh(feature_frame)
    wind_speed_distribution = wind_speed_power_distribution(feature_frame, total_capacity_mw)

    return {
        "power_curves": power_curves, "smooth_power_curves": smooth_power_curves,
        "efficiency_summary": efficiency_summary,
        "anomalies": anomalies, "data_quality": data_quality,
        "wind_rose": wind_rose, "wind_speed_distribution": wind_speed_distribution,
    }


def rag_node(state: WindwardState) -> dict:
    from rag.langchain_retriever import get_retriever

    retriever = get_retriever(state["farm_id"])
    docs = retriever.invoke("turbine faults, downtime causes, curtailment, extraction efficiency")
    return {"rag_context": [{"source": d.metadata.get("source", "?"), "text": d.page_content} for d in docs]}


SAMPLE_INSPECTION_PHOTO = "data/sample_inspection_photos/rotorblatt_inspection.jpg"


def multimodal_node(state: WindwardState) -> dict:
    """Runs a real vision-LLM inspection pass (multimodal/blade_inspection.py) on the
    worst-performing turbine, using one real, openly-licensed sample photo (Wikimedia
    Commons, CC BY-SA 3.0) — not a live drone feed, which isn't sourced yet (see roadmap)."""
    from multimodal.blade_inspection import inspect_image

    worst_turbine = state["efficiency_summary"]["capacity_factor"].idxmin()
    result = inspect_image(worst_turbine, SAMPLE_INSPECTION_PHOTO)
    return {"inspection_results": [result.model_dump()]}


def investigate_node(state: WindwardState) -> dict:
    """Runs after rag+multimodal (both already in state by the time this fires — same
    fan-in guarantee the old rag/multimodal -> recommend edges gave recommend_node). Computes
    whether the forecast has gotten measurably worse recently and whether the registered model
    is old enough that retraining is worth flagging — real signals, not a fixed check."""
    from forecasting.registry import latest_model_info

    comparison = state["comparison"]
    abs_pct_err = (comparison["actual_mw"] - comparison["predicted_mw"]).abs() / comparison["predicted_mw"].replace(0, float("nan"))
    overall_mape = float(abs_pct_err.mean())
    recent_mape = float(abs_pct_err.tail(24 * 7).mean())  # last week of the SCADA period
    error_increased = recent_mape > overall_mape * RECENT_ERROR_INCREASE_RATIO

    model_info = latest_model_info(model_name(state["farm_id"]))
    stale = (model_info["age_days"] or 0) > RETRAIN_AGE_DAYS_THRESHOLD

    return {
        "investigation": {
            "overall_mape": overall_mape,
            "recent_mape": recent_mape,
            "error_increased": error_increased,
            "model_version": model_info["version"],
            "model_age_days": model_info["age_days"],
            "recommend_retrain": bool(error_increased and stale),
        }
    }


def _route_after_investigation(state: WindwardState) -> str:
    return "recommend_retrain" if state["investigation"]["recommend_retrain"] else "recommend"


def recommend_node(state: WindwardState) -> dict:
    eff = state["efficiency_summary"]
    worst = eff["capacity_factor"].idxmin()
    return {
        "recommendation": {
            "farm_id": state["farm_id"],
            "action": f"Prioritize inspection of {worst}",
            "rationale": (
                f"{worst} has the lowest capacity factor in the fleet "
                f"({eff.loc[worst, 'capacity_factor']:.1%} vs fleet mean {eff['capacity_factor'].mean():.1%})."
            ),
            "supporting_sources": [c["source"] for c in state.get("rag_context", [])],
        }
    }


def recommend_retrain_node(state: WindwardState) -> dict:
    inv = state["investigation"]
    return {
        "recommendation": {
            "farm_id": state["farm_id"],
            "action": f"Retrain the {state['farm_id']} forecasting model",
            "rationale": (
                f"Forecast error over the last 7 days (MAPE {inv['recent_mape']:.1%}) is running "
                f"more than {int((RECENT_ERROR_INCREASE_RATIO - 1) * 100)}% above the full analysis "
                f"period's average ({inv['overall_mape']:.1%}), and the currently registered model "
                f"(v{inv['model_version']}) is {inv['model_age_days']} days old — past the "
                f"{RETRAIN_AGE_DAYS_THRESHOLD}-day point where retraining on more recent data is worth checking."
            ),
            "supporting_sources": [c["source"] for c in state.get("rag_context", [])],
        }
    }


def explain_node(state: WindwardState) -> dict:
    farm = FARMS[state["farm_id"]]
    eff = state["efficiency_summary"]
    comparison = state["comparison"]
    rag_block = "\n".join(f"- ({c['source']}) {c['text'][:300]}" for c in state.get("rag_context", [])) or "(none retrieved)"
    inv = state.get("investigation") or {}
    investigation_block = (
        f"recent-week MAPE {inv['recent_mape']:.1%} vs full-period MAPE {inv['overall_mape']:.1%} "
        f"(error {'has' if inv['error_increased'] else 'has not'} increased meaningfully); "
        f"registered model v{inv['model_version']}, {inv['model_age_days']} days old"
        if inv else "(not computed)"
    )
    prompt = f"""You are a wind farm performance analyst. Write a concise (under 170 words) plain-English
summary of this farm's performance for a technical but non-specialist stakeholder.

Farm: {farm.name} ({len(farm.turbine_ids)}x turbines, {farm.rated_power_kw * len(farm.turbine_ids) / 1000:.1f} MW total)
Farm-level actual vs model-predicted production, over {len(comparison)} hours:
  mean actual: {comparison['actual_mw'].mean():.2f} MW, mean predicted: {comparison['predicted_mw'].mean():.2f} MW
Per-turbine capacity factor and peak power coefficient (Cp, real per-hour air density used, Betz limit {eff['betz_limit'].iloc[0]:.3f}):
{eff[['capacity_factor', 'peak_cp']].round(3).to_string()}
Anomalies detected: {state['anomalies']}
Data quality — farm SCADA wind speed vs. independent Open-Meteo reanalysis: {state.get('data_quality')}
Model-health investigation: {investigation_block}
Blade inspection (vision-LLM pass on the worst-performing turbine's most recent available photo): {state.get('inspection_results')}
Recommendation already decided: {state['recommendation']['action']} — {state['recommendation']['rationale']}
Retrieved reference context (cite it by name if you use it, otherwise ignore):
{rag_block}

Explain what these numbers mean for wind-resource-extraction efficiency, and back the recommendation with the data. If the recommendation is about retraining, say so plainly; otherwise briefly note that the model-health check came back clean."""

    response = complete([{"role": "user", "content": prompt}])
    return {"explanation": response.choices[0].message.content}


def build_graph():
    graph = StateGraph(WindwardState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("rag", rag_node)
    graph.add_node("multimodal", multimodal_node)
    graph.add_node("investigate", investigate_node)
    graph.add_node("recommend", recommend_node)
    graph.add_node("recommend_retrain", recommend_retrain_node)
    graph.add_node("explain", explain_node)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "forecast")
    graph.add_edge("forecast", "diagnose")
    graph.add_edge("diagnose", "rag")
    graph.add_edge("diagnose", "multimodal")
    # investigate is the fan-in point (needs rag_context + inspection_results already in
    # state, same guarantee recommend used to get directly) and its only outgoing routing
    # is conditional — mixing static predecessor edges into a conditional-routing target
    # makes both branches fire regardless of the condition, so recommend/recommend_retrain
    # must not also receive static edges from rag/multimodal.
    graph.add_edge("rag", "investigate")
    graph.add_edge("multimodal", "investigate")
    graph.add_conditional_edges(
        "investigate",
        _route_after_investigation,
        {"recommend": "recommend", "recommend_retrain": "recommend_retrain"},
    )
    graph.add_edge("recommend", "explain")
    graph.add_edge("recommend_retrain", "explain")
    graph.add_edge("explain", END)

    return graph.compile()


def run(graph, farm_id: str) -> dict:
    """Invoke the compiled graph for a farm, tracing to Langfuse when configured
    (see observability/tracing.py — a no-op if LANGFUSE_* isn't set)."""
    from observability.tracing import get_handler

    handler = get_handler()
    run_config = {"callbacks": [handler]} if handler else {}
    return graph.invoke({"farm_id": farm_id}, config=run_config)
