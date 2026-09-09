"""LangGraph workflow: ingest -> forecast -> diagnose -> rag -> multimodal -> recommend -> explain.

This is the "architect generative AI workflows" piece — a declarative, checkpointable state
machine rather than hand-rolled orchestration (contrast with global-news-agent's custom
orchestrator).

Runs in analysis/backtest mode over each farm's real SCADA period (the only period we have
real production data for): ingest real weather+production, forecast with the farm's
registered model, diagnose actual-vs-predicted + per-turbine wind-extraction efficiency,
then have the LLM narrate it. rag/multimodal are wired as pass-throughs until a document
corpus / inspection photos exist (see roadmap) — the graph still runs end-to-end today.

Invoke with an initial state containing farm_id, e.g. graph.invoke({"farm_id": "kelmarsh"}).
"""
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from agents.llm_router import complete
from analysis.efficiency import actual_vs_predicted, binned_power_curve, turbine_efficiency_summary
from data_sources.farms import FARMS, loader_for
from forecasting.features import FEATURE_COLUMNS, TARGET_COLUMN
from forecasting.pipeline import build_training_frame
from forecasting.registry import load_latest_model
from forecasting.train import model_name


class WindwardState(TypedDict, total=False):
    farm_id: str
    feature_frame: pd.DataFrame
    turbine_hourly: pd.DataFrame
    comparison: pd.DataFrame  # actual vs predicted, farm-level
    power_curves: dict
    efficiency_summary: pd.DataFrame
    anomalies: list
    rag_context: list
    inspection_results: list
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

    power_curves = {
        turbine_id: binned_power_curve(g)
        for turbine_id, g in turbine_hourly.groupby("turbine_id")
    }
    efficiency_summary = turbine_efficiency_summary(turbine_hourly, farm.rated_power_kw, farm.rotor_diameter_m)

    comparison = state["comparison"]
    bad_hours = comparison[(comparison["pct_of_predicted"] < 0.5) | (comparison["pct_of_predicted"] > 1.5)]
    anomalies = [
        {"type": "farm_forecast_deviation", "count": len(bad_hours), "pct_of_hours": len(bad_hours) / len(comparison)}
    ]
    over_betz = efficiency_summary[efficiency_summary["peak_cp"] > efficiency_summary["betz_limit"]]
    for turbine_id in over_betz.index:
        anomalies.append(
            {"type": "anemometer_calibration_suspect", "turbine_id": turbine_id,
             "detail": "peak Cp exceeds the Betz limit — likely nacelle anemometer bias, not real over-unity extraction"}
        )
    return {"power_curves": power_curves, "efficiency_summary": efficiency_summary, "anomalies": anomalies}


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


def explain_node(state: WindwardState) -> dict:
    farm = FARMS[state["farm_id"]]
    eff = state["efficiency_summary"]
    comparison = state["comparison"]
    rag_block = "\n".join(f"- ({c['source']}) {c['text'][:300]}" for c in state.get("rag_context", [])) or "(none retrieved)"
    prompt = f"""You are a wind farm performance analyst. Write a concise (under 170 words) plain-English
summary of this farm's performance for a technical but non-specialist stakeholder.

Farm: {farm.name} ({len(farm.turbine_ids)}x turbines, {farm.rated_power_kw * len(farm.turbine_ids) / 1000:.1f} MW total)
Farm-level actual vs model-predicted production, over {len(comparison)} hours:
  mean actual: {comparison['actual_mw'].mean():.2f} MW, mean predicted: {comparison['predicted_mw'].mean():.2f} MW
Per-turbine capacity factor and peak power coefficient (Cp, Betz limit {eff['betz_limit'].iloc[0]:.3f}):
{eff[['capacity_factor', 'peak_cp']].round(3).to_string()}
Anomalies detected: {state['anomalies']}
Blade inspection (vision-LLM pass on the worst-performing turbine's most recent available photo): {state.get('inspection_results')}
Recommendation already decided: {state['recommendation']['action']} — {state['recommendation']['rationale']}
Retrieved reference context (cite it by name if you use it, otherwise ignore):
{rag_block}

Explain what these numbers mean for wind-resource-extraction efficiency, and back the recommendation with the data."""

    response = complete([{"role": "user", "content": prompt}])
    return {"explanation": response.choices[0].message.content}


def build_graph():
    graph = StateGraph(WindwardState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("rag", rag_node)
    graph.add_node("multimodal", multimodal_node)
    graph.add_node("recommend", recommend_node)
    graph.add_node("explain", explain_node)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "forecast")
    graph.add_edge("forecast", "diagnose")
    graph.add_edge("diagnose", "rag")
    graph.add_edge("diagnose", "multimodal")
    graph.add_edge("rag", "recommend")
    graph.add_edge("multimodal", "recommend")
    graph.add_edge("recommend", "explain")
    graph.add_edge("explain", END)

    return graph.compile()


def run(graph, farm_id: str) -> dict:
    """Invoke the compiled graph for a farm, tracing to Langfuse when configured
    (see observability/tracing.py — a no-op if LANGFUSE_* isn't set)."""
    from observability.tracing import get_handler

    handler = get_handler()
    run_config = {"callbacks": [handler]} if handler else {}
    return graph.invoke({"farm_id": farm_id}, config=run_config)
