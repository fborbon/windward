"""Tool-calling QA agent for the public "ask" endpoint.

Before this existed, /analysis/{farm_id}/ask (api/main.py) answered every question with one
direct Bedrock call and the farm's entire precomputed analysis stuffed into the prompt — no
retrieval, no tool selection, none of the RAG/LangChain stack the rest of the project actually
has. This module fixes that: the LLM is given real tools and decides for itself whether a
question needs the RAG-grounded maintenance/incident corpus, the farm's current numeric
analysis, both, or neither.

`search_maintenance_docs` reuses mcp_server.tools.query_maintenance_docs unchanged — the same
FAISS/Bedrock-Titan retriever the MCP server already exposes — so the public dashboard and any
MCP client are grounded by the same retrieval pipeline, not two divergent implementations.
"""
import json

from agents.llm_router import complete
from mcp_server.tools import query_maintenance_docs
from observability.tracing import enable_litellm_tracing

enable_litellm_tracing()  # no-op until LANGFUSE_* keys are set — see observability/tracing.py

MAX_TOOL_ROUNDS = 3

_ANALYSIS_KEYS = ("comparison_summary", "efficiency_summary", "anomalies", "recommendation")

_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_maintenance_docs",
            "description": (
                "Search this farm's real turbine fault/incident log and reference notes for "
                "information about faults, downtime causes, curtailment, or turbine specs. "
                "Use for 'why', 'what caused', or history questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_analysis",
            "description": (
                "Get this farm's current numeric analysis: actual-vs-predicted production, "
                "per-turbine capacity factor and power coefficient (Cp) vs the Betz limit, "
                "detected anomalies, and the existing recommendation. Use for questions about "
                "current performance or numbers."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_SYSTEM_PROMPT = (
    "You are the Windward analysis agent for farm '{farm_id}'. Answer the visitor's question "
    "using the tools available: call search_maintenance_docs for fault/downtime/history "
    "questions, get_current_analysis for numeric/performance questions, both if the question "
    "needs both, or neither for a general question you can already answer. Keep the final "
    "answer under 120 words, plain English, technical but non-specialist."
)


def _run_tool(name: str, args: dict, farm_id: str, analysis: dict, sources: list[str]):
    if name == "search_maintenance_docs":
        sources.append("maintenance-docs")
        return query_maintenance_docs(farm_id, args.get("question") or "")
    if name == "get_current_analysis":
        sources.append("current-analysis")
        return {k: analysis[k] for k in _ANALYSIS_KEYS if k in analysis}
    return f"unknown tool {name!r}"


def _fallback_answer(farm_id: str, question: str, analysis: dict) -> dict:
    """Today's stuffed-prompt behaviour, kept as a safety net if tool-calling errors out."""
    context = (
        f"You are the Windward analysis agent for {farm_id}. Answer using ONLY the data below. "
        f"Keep the answer under 120 words, plain English, technical but non-specialist.\n\n"
        f"Comparison: {analysis.get('comparison_summary')}\n"
        f"Efficiency: {analysis.get('efficiency_summary')}\n"
        f"Anomalies: {analysis.get('anomalies')}\n"
        f"Recommendation: {analysis.get('recommendation')}\n\n"
        f"Visitor question: {question}"
    )
    response = complete([{"role": "user", "content": context}])
    return {"answer": response.choices[0].message.content, "sources": []}


def answer_question(farm_id: str, question: str, analysis: dict) -> dict:
    sources: list[str] = []
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT.format(farm_id=farm_id)},
        {"role": "user", "content": question},
    ]

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = complete(messages, tools=_TOOLS_SCHEMA)
            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                return {"answer": msg.content, "sources": sources}

            messages.append(msg.model_dump())
            for call in tool_calls:
                args = json.loads(call.function.arguments or "{}")
                result = _run_tool(call.function.name, args, farm_id, analysis, sources)
                content = result if isinstance(result, str) else json.dumps(result, default=str)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

        # Exhausted MAX_TOOL_ROUNDS without a final answer — ask once more without tools.
        response = complete(messages)
        return {"answer": response.choices[0].message.content, "sources": sources}
    except Exception as e:  # tool-calling is new — never let it take the endpoint fully down
        print(f"WARN: qa_agent tool-calling failed ({e}); falling back to stuffed-prompt answer")
        return _fallback_answer(farm_id, question, analysis)
