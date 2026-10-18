"""Tool implementations exposed by the MCP server — thin wrappers around the graph nodes
and forecasting/RAG modules, so the same logic is callable from the FastAPI service, the
LangGraph agent, or an external MCP client (e.g. Claude)."""
from agents.graph import build_graph, run as run_graph
from agents.llm_router import complete
from data_sources.farms import FARMS
from forecasting.predict import predict_production
from rag.langchain_retriever import get_retriever
from schemas.models import ForecastRequest, ForecastResult

_graph_cache = None


def _graph():
    global _graph_cache
    if _graph_cache is None:
        _graph_cache = build_graph()
    return _graph_cache


def get_forecast(request: ForecastRequest) -> ForecastResult:
    return predict_production(farm_id=request.farm_id, horizon_hours=request.horizon_hours)


def _check_farm(farm_id: str):
    if farm_id not in FARMS:
        raise ValueError(f"unknown farm_id {farm_id!r}, expected one of {list(FARMS)}")


_ANALYSIS_TTL_S = 3600
_analysis_cache: dict[str, tuple[float, dict]] = {}


def _turbine_analysis(farm_id: str) -> dict:
    """The farm's graph run, reduced to what agents/turbine_tools.py reads and cached for an
    hour like the API's /analysis cache, so a client asking about several turbines in a row
    doesn't re-run the graph (and its LLM narration) for each one."""
    import time

    cached = _analysis_cache.get(farm_id)
    if cached and time.time() - cached[0] < _ANALYSIS_TTL_S:
        return cached[1]
    from agents.turbine_tools import analysis_from_graph_state

    analysis = analysis_from_graph_state(run_graph(_graph(), farm_id))
    _analysis_cache[farm_id] = (time.time(), analysis)
    return analysis


def get_recommendation(farm_id: str) -> dict:
    """Runs the full diagnose pipeline for the farm and returns its recommendation +
    supporting narrative — not a cached lookup, this re-derives it from current data."""
    _check_farm(farm_id)
    result = run_graph(_graph(), farm_id)
    return {**result["recommendation"], "explanation": result["explanation"]}


def get_operational_assessment(farm_id: str) -> dict:
    """The offline OpenOA results for the farm (README §15): electrical losses, long-term AEP
    with its uncertainty breakdown, wake losses and static yaw misalignment per turbine."""
    from operational_assessment.payload import compact_summary, farm_operational, yaw_flags

    _check_farm(farm_id)
    op = farm_operational(farm_id)
    if op is None:
        return {"error": "no operational assessment exported for this farm"}
    return {**compact_summary(op), "yaw_flags": yaw_flags(op)}


def get_turbine_details(farm_id: str, turbine: str) -> dict | str:
    from agents.turbine_tools import turbine_details

    _check_farm(farm_id)
    return turbine_details(farm_id, turbine, _turbine_analysis(farm_id))


def rank_turbines(farm_id: str, metric: str) -> dict | str:
    from agents.turbine_tools import RANK_METRICS, rank_turbines as _rank

    _check_farm(farm_id)
    # only the first two metrics need the graph run; the rest come from OpenOA + status logs
    analysis = _turbine_analysis(farm_id) if metric in ("capacity_factor", "peak_cp", "qc_removed_share") else {}
    return _rank(farm_id, metric, analysis)


def query_maintenance_docs(farm_id: str, question: str) -> str:
    """RAG over the farm's real incident log + technical reference notes, then a grounded
    LLM answer citing sources."""
    retriever = get_retriever(farm_id)
    docs = retriever.invoke(question)
    context = "\n".join(f"- ({d.metadata.get('source', '?')}) {d.page_content}" for d in docs)
    prompt = (
        f"Answer the question using only the context below, citing sources by name. "
        f"If the context doesn't cover it, say so.\n\nContext:\n{context}\n\nQuestion: {question}"
    )
    response = complete([{"role": "user", "content": prompt}])
    return response.choices[0].message.content


def technical_documents(farm_id: str) -> dict:
    """The real PDFs indexed for this farm (README §16) and the datasheet-vs-config check."""
    from rag.datasheet_specs import crosscheck
    from rag.pdf_retriever import indexed_documents

    _check_farm(farm_id)
    return {"documents": indexed_documents(farm_id), "datasheet_check": crosscheck(FARMS[farm_id])}


def query_technical_documents(farm_id: str, question: str) -> str:
    """RAG over real public PDFs: the manufacturer documents for this farm's turbine model and
    public offshore-wind tender/leasing documents (rag/pdf_sources.py), answered with page
    citations."""
    from rag.datasheet_specs import crosscheck
    from rag.pdf_retriever import index_built, search

    _check_farm(farm_id)
    if not index_built():
        return "The technical-document index has not been built on this host yet (rag/pdf_index_build.py)."
    docs = search(farm_id, question)
    if not docs:
        return "No passage in the indexed technical documents matches this question."

    context = "\n\n".join(
        f"[{d.metadata['title']}, p.{d.metadata['page']} - {d.metadata['doc_type']}]\n{d.page_content}" for d in docs
    )
    check = crosscheck(FARMS[farm_id])
    if check and check["checks"]:
        lines = [
            f"- {c['field']}: datasheet p.{c['page']} says {c['datasheet']:g}, Windward's config uses "
            f"{c['configured']:g} ({c['status']})"
            for c in check["checks"]
        ]
        context += f"\n\n[Datasheet cross-check for {check['turbine_model']}]\n" + "\n".join(lines)

    prompt = (
        f"Farm: {FARMS[farm_id].name}, turbine model {FARMS[farm_id].manufacturer_model}.\n"
        "Answer the question using only the PDF passages below, citing every fact as (document "
        "title, p.N), and keep any condition or limit the passage attaches to it. Passages of type "
        "tender_site_description or leasing_round are about other, offshore wind projects: answer "
        "questions about tenders, bidding or project planning directly from them, naming the "
        "project, but never present them as facts about this farm. If the passages don't cover "
        "the question, say so.\n\n"
        f"Passages:\n{context}\n\nQuestion: {question}"
    )
    response = complete([{"role": "user", "content": prompt}])
    urls = sorted({f"{d.metadata['title']}: {d.metadata['url']}" for d in docs})
    return response.choices[0].message.content + "\n\nDocuments: " + "; ".join(urls)


def query_edp_incidents(question: str) -> str:
    """RAG over EDP Wind Farm A's real labeled fault case studies (22 anonymized turbine
    windows, 11 with a real root-cause description) — see rag/edp_incident_corpus.py. Separate
    from query_maintenance_docs because this corpus isn't keyed to a Farm dataclass entry."""
    from rag.edp_retriever import get_retriever as get_edp_retriever

    docs = get_edp_retriever().invoke(question)
    context = "\n".join(f"- ({d.metadata.get('source', '?')}) {d.page_content}" for d in docs)
    prompt = (
        f"Answer the question using only the context below, citing sources by name. "
        f"If the context doesn't cover it, say so.\n\nContext:\n{context}\n\nQuestion: {question}"
    )
    response = complete([{"role": "user", "content": prompt}])
    return response.choices[0].message.content


def query_dswe_reference(question: str) -> str:
    """RAG over the DSWE Inland-Offshore dataset's real turbine/mast facts and the
    Measure-Correlate-Predict methodology — see rag/dswe_incident_corpus.py."""
    from rag.dswe_retriever import get_retriever as get_dswe_retriever

    docs = get_dswe_retriever().invoke(question)
    context = "\n".join(f"- ({d.metadata.get('source', '?')}) {d.page_content}" for d in docs)
    prompt = (
        f"Answer the question using only the context below, citing sources by name. "
        f"If the context doesn't cover it, say so.\n\nContext:\n{context}\n\nQuestion: {question}"
    )
    response = complete([{"role": "user", "content": prompt}])
    return response.choices[0].message.content
