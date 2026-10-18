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
import re

from agents.llm_router import complete
from data_sources.farms import FARMS
from agents.turbine_tools import RANK_METRICS, rank_turbines, turbine_details
from mcp_server.tools import query_maintenance_docs, technical_documents_answer

MAX_TOOL_ROUNDS = 3

_ANALYSIS_KEYS = ("comparison_summary", "efficiency_summary", "qc_summary", "anomalies", "recommendation")

_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_maintenance_docs",
            "description": (
                "Search this farm's real turbine fault/incident log and reference notes for "
                "information about faults, downtime causes, curtailment, or turbine specs. "
                "Use for 'why', 'what caused', or history questions. Its incidents are individual "
                "example events (the farm's longest ones), not totals: for a turbine's total "
                "outage hours, counts or rankings use get_turbine_details / rank_turbines instead."
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
            "name": "search_technical_documents",
            "description": (
                "Search real public PDFs: the manufacturer documents for this farm's turbine "
                "model (design, drivetrain, operating limits, power, upgrades) and public "
                "offshore-wind tender and leasing documents (site conditions, technical and "
                "application requirements for bidders). Answers cite document and page. Use for "
                "turbine design/specification questions, whether Windward's turbine specs match "
                "the datasheet, or wind-project planning and tender requirement questions."
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
            "name": "get_farm_facts",
            "description": (
                "This farm's own dataset metadata and production history: commissioning (commercial "
                "operations) date, turbine model, hub height per turbine, decommissioning information, "
                "upgrades, and production records over the full multi-year SCADA (highest 10-minute "
                "farm output and when, best day, best month, energy per calendar year). Use for when "
                "the farm was built or decommissioned, its hub heights, or its maximum/record/"
                "historical production."
            ),
            "parameters": {"type": "object", "properties": {}},
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
    {
        "type": "function",
        "function": {
            "name": "get_operational_assessment",
            "description": (
                "Get this farm's long-term operational assessment from NREL/NLR's OpenOA, run on "
                "the full multi-year data: electrical losses (turbines vs substation meter), "
                "long-term annual energy production (P50/P90) and its uncertainty components, "
                "per-turbine wake losses, and static yaw misalignment estimates. Use for questions "
                "about AEP, energy yield, losses, wakes, yaw, or uncertainty."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_turbine_details",
            "description": (
                "Everything known about ONE turbine: capacity factor and its rank in the fleet, "
                "peak Cp, QC-removed hours, its anomalies, OpenOA wake loss and yaw misalignment, "
                "and its real status log (last maintenance stop, maintenance count, forced-outage "
                "hours and top causes). Use for any question about a specific turbine. It needs the turbine's "
                "name or number: when the turbine comes from a ranking, call rank_turbines first and this "
                "in a later step."
            ),
            "parameters": {
                "type": "object",
                "properties": {"turbine": {"type": "string", "description": "turbine name or number, e.g. 'Kelmarsh 3' or '3'"}},
                "required": ["turbine"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_turbines",
            "description": (
                "Rank this farm's turbines by one metric, to answer best/worst/most/least "
                "questions across turbines. capacity_factor and peak_cp: higher is better; the "
                "others (wake_loss_lt_pct, abs_yaw_misalignment_deg, qc_removed_share, "
                "forced_outage_hours, maintenance_events) are listed highest first."
            ),
            "parameters": {
                "type": "object",
                "properties": {"metric": {"type": "string", "enum": list(RANK_METRICS)}},
                "required": ["metric"],
            },
        },
    },
]

_SYSTEM_PROMPT = (
    "You are the Windward analysis agent for farm '{farm_id}', whose turbines are {turbines}. A question "
    "about a turbine not in that list is about a turbine that doesn't exist: say so. Questions about "
    "public offshore tenders or leasing rounds (Borssele, Hollandse Kust (zuid), ScotWind) are in scope "
    "whichever farm is selected: answer them with search_technical_documents. Answer the visitor's question "
    "using the tools available: call search_maintenance_docs for fault/downtime/history "
    "questions, search_technical_documents for turbine design/specification questions and "
    "wind-project tender or planning requirements (keep its page citations, and never present "
    "tender documents about other projects as facts about this farm), get_current_analysis for "
    "numeric/performance questions over the 2016 analysis window, get_farm_facts for when the farm was "
    "built or decommissioned, upgrades installed on it (e.g. AeroUp), its hub heights, and record or "
    "multi-year production, "
    "get_operational_assessment for long-term yield/loss/wake/yaw questions, get_turbine_details "
    "for a question about one specific turbine, rank_turbines for best/worst/most-affected "
    "comparisons across turbines, several if the question "
    "needs both, or neither for a general question you can already answer. Keep the final "
    "answer under 120 words, plain English, technical but non-specialist. For totals, counts and "
    "rankings per turbine, the turbine tools are authoritative; if you also cite a single event from "
    "the maintenance docs, say it's one event. Quote numbers exactly "
    "as the tools return them; never compute new statistics (intervals, percentages) yourself. "
    "Keep the page citations search_technical_documents returns, and any condition it "
    "attaches to a fact (\"only for turbines with...\"); never add citations for the other tools. Follow any "
    "note a tool returns. If a tool returns an error or lacks something, say what is missing; never fill the "
    "gap yourself. Don't pad a tool's facts with explanations it didn't give, and don't show its "
    "internal field names (lt_pct, por_pct): say what they mean (long-term corrected, period of record). "
    "Reply with the answer only: no reasoning, plans or tags."
)

# Asked "how close does the best turbine get to Betz?", the agent scanned this unsorted list and
# named the second-highest turbine, and called a Cp above the limit "close to it".
_ANALYSIS_NOTE = (
    "efficiency_summary is not sorted: for best/worst/highest across turbines call rank_turbines. The Betz "
    "limit is 0.593; a peak_cp above it (over_betz true) is physically impossible and points to the nacelle "
    "anemometer reading low (see anomalies), not to a turbine beating Betz."
)

def _highlights(analysis: dict) -> dict:
    """The extremes of the unsorted efficiency_summary, ready-made."""
    eff = [e for e in analysis.get("efficiency_summary") or [] if e.get("peak_cp") is not None]
    if not eff:
        return {}
    by_cf = sorted(eff, key=lambda e: e["capacity_factor"])
    top_cp = max(eff, key=lambda e: e["peak_cp"])
    return {
        "highest_capacity_factor": {k: by_cf[-1][k] for k in ("turbine_id", "capacity_factor")},
        "lowest_capacity_factor": {k: by_cf[0][k] for k in ("turbine_id", "capacity_factor")},
        "highest_peak_cp": {k: top_cp.get(k) for k in ("turbine_id", "peak_cp", "over_betz")},
    }


def _with_citations(answer: str, cited: list[dict]) -> str:
    """The technical-documents tool's answer is page-cited, but the agent's shorter rewrite of it
    often drops the citations. Put the verified ones back when it did."""
    if not cited or re.search(r"\bp\.\s?\d", answer):
        return answer
    by_title: dict[str, list[str]] = {}
    for c in cited:
        by_title.setdefault(c["title"], []).append(f"p.{c['page']}")
    return answer + "\n\nSources: " + "; ".join(f"{t}, {', '.join(p)}" for t, p in by_title.items())


_NO_ANSWER = "The agent returned no answer this time. Please try asking again."

_THINKING = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)


def _clean(text: str | None) -> str:
    """Nova sometimes leaks its reasoning: a <thinking> block, or only the closing tag after
    a paragraph of plans ("I will present this information as..."). Keep what follows it, and
    unwrap citations it formats as broken markdown links."""
    text = _THINKING.sub("", text or "")
    # the prompt's own citation template, echoed back verbatim
    text = re.sub(r"\s*\(document title, p\.N\)", "", text)
    # citations formatted as broken markdown links: "[Title, p.6](document_title, p.6)"
    text = re.sub(r"\[([^\]]+)\]\(document_title[^)]*\)", r"(\1)", text)
    if "</thinking>" in text:
        text = text.rsplit("</thinking>", 1)[1]
    return text.replace("<thinking>", "").strip()


def _run_tool(name: str, args: dict, farm_id: str, analysis: dict, sources: list[str], cited: list[dict]):
    if name == "search_maintenance_docs":
        sources.append("maintenance-docs")
        return query_maintenance_docs(farm_id, args.get("question") or "")
    if name == "search_technical_documents":
        sources.append("technical-docs")
        result = technical_documents_answer(farm_id, args.get("question") or "")
        cited.extend(c for c in result["cited"] if c not in cited)
        return result["text"]
    if name == "get_farm_facts":
        from data_sources.farm_facts import farm_facts

        sources.append("farm-facts")
        return farm_facts(farm_id) or "no farm facts exported for this farm"
    if name == "get_current_analysis":
        sources.append("current-analysis")
        out = {k: analysis[k] for k in _ANALYSIS_KEYS if k in analysis}
        cmp = dict(out.get("comparison_summary") or {})
        if cmp.get("mean_abs_pct_error") is not None:
            # a fraction under a "pct" name: the agent reported 0.504 as "0.504%" even with a note
            cmp["mean_abs_hourly_error_percent"] = round(cmp.pop("mean_abs_pct_error") * 100, 1)
        if cmp.get("mean_actual_mw") is not None and cmp.get("mean_predicted_mw"):
            # and it called 5.654 MW actual vs 5.653 predicted "slightly lower"
            diff = cmp["mean_actual_mw"] - cmp["mean_predicted_mw"]
            cmp["mean_actual_vs_predicted"] = f"actual {'above' if diff >= 0 else 'below'} predicted by {abs(diff) / cmp['mean_predicted_mw'] * 100:.2f}%"
        for k in ("mean_actual_mw", "mean_predicted_mw"):
            if isinstance(cmp.get(k), float):
                cmp[k] = round(cmp[k], 3)
        if cmp:
            out["comparison_summary"] = cmp
        return {**out, "highlights": _highlights(analysis), "note": _ANALYSIS_NOTE}
    if name == "get_operational_assessment":
        from operational_assessment.payload import compact_summary, farm_operational

        sources.append("operational-assessment")
        return compact_summary(farm_operational(farm_id)) or "no operational assessment exported for this farm"
    if name == "get_turbine_details":
        sources.append("turbine-details")
        return turbine_details(farm_id, args.get("turbine") or "", analysis)
    if name == "rank_turbines":
        sources.append("turbine-ranking")
        return rank_turbines(farm_id, args.get("metric") or "", analysis)
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
    return {"answer": _clean(response.choices[0].message.content), "sources": []}


def answer_question(farm_id: str, question: str, analysis: dict) -> dict:
    sources: list[str] = []
    cited: list[dict] = []
    retried = False
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT.format(farm_id=farm_id, turbines=", ".join(FARMS[farm_id].turbine_ids))},
        {"role": "user", "content": question},
    ]

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = complete(messages, tools=_TOOLS_SCHEMA, temperature=0.3 if retried else 0)
            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                answer = _clean(msg.content)
                if not answer and not retried:
                    # Nova Lite now and then returns a bare newline, no answer and no tool call
                    # (seen at temperature 0 on the ScotWind question); one retry clears it.
                    retried = True
                    continue
                return {"answer": _with_citations(answer, cited) or _NO_ANSWER, "sources": sources}

            messages.append(msg.model_dump())
            for call in tool_calls:
                args = json.loads(call.function.arguments or "{}")
                result = _run_tool(call.function.name, args, farm_id, analysis, sources, cited)
                content = result if isinstance(result, str) else json.dumps(result, default=str)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

        # Exhausted MAX_TOOL_ROUNDS without a final answer — ask once more without tools.
        response = complete(messages, temperature=0)
        return {"answer": _with_citations(_clean(response.choices[0].message.content), cited), "sources": sources}
    except Exception as e:  # tool-calling is new — never let it take the endpoint fully down
        print(f"WARN: qa_agent tool-calling failed ({e}); falling back to stuffed-prompt answer", flush=True)
        return _fallback_answer(farm_id, question, analysis)
