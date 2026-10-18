"""Tool implementations exposed by the MCP server — thin wrappers around the graph nodes
and forecasting/RAG modules, so the same logic is callable from the FastAPI service, the
LangGraph agent, or an external MCP client (e.g. Claude)."""
import re

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
    # indexes built before rag/incident_corpus.py handled a missing category say "nan"
    context = "\n".join(
        f"- ({d.metadata.get('source', '?')}) {d.page_content.replace('IEC category: nan.', 'IEC category: unclassified.')}"
        for d in docs
    )
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
    return technical_documents_answer(farm_id, question)["text"]


_CITE = re.compile(r"\bp\.\s?(\d+)")


def check_citations(answer: str, pages: list[dict]) -> tuple[str, list[dict]]:
    """Re-checks every "p.N" citation in an answer against the text of the page it names, the
    way rag/datasheet_specs.verify checks extracted specs: the numbers in the sentence it cites
    must be printed on that page. Nova often cites the page number printed in a document's own
    running header (the Hollandse Kust (zuid) depths are on PDF page 14, printed as page 20); a
    citation whose numbers are on another page in the context is moved there. Returns the
    corrected answer and the (title, url, page) of each citation, for the agent to keep."""
    from rag.datasheet_specs import _numbers

    page_numbers = [(p, _numbers(p["text"])) for p in pages]
    out, cited, last = [], [], 0
    for m in _CITE.finditer(answer):
        n = int(m.group(1))
        segment = answer[last:m.start()]
        for p in pages:  # a title's own numbers ("October 2017") aren't claims
            segment = segment.replace(p["title"], "")
        claims = _numbers(segment[-400:])
        named = [pn for pn in page_numbers if pn[0]["page"] == n]
        # several documents can have a page n: prefer the one whose title the citation names
        before = answer[max(0, m.start() - 160):m.start()]
        named.sort(key=lambda pn: pn[0]["title"][:25] not in before)
        target = named[0][0] if named else None
        if claims and not (named and claims <= named[0][1]):
            moved = next((p for p, nums in page_numbers if claims <= nums and (target is None or p["doc_id"] == target["doc_id"])), None)
            moved = moved or next((p for p, nums in page_numbers if claims <= nums), None)
            if moved is not None:
                target = moved
        out.append(answer[last:m.start()] + (f"p.{target['page']}" if target else m.group(0)))
        last = m.end()
        if target and (target["title"], target["page"]) not in {(c["title"], c["page"]) for c in cited}:
            cited.append({"title": target["title"], "url": target["url"], "page": target["page"]})
    out.append(answer[last:])
    return "".join(out), cited


def technical_documents_answer(farm_id: str, question: str) -> dict:
    """query_technical_documents' answer with its verified citations kept apart (qa_agent
    re-attaches them when its own shorter rewrite drops them)."""
    from rag.datasheet_specs import crosscheck
    from rag.pdf_retriever import index_built, search_pages

    _check_farm(farm_id)
    if not index_built():
        return {"text": "The technical-document index has not been built on this host yet (rag/pdf_index_build.py).", "cited": []}
    pages = search_pages(farm_id, question)
    if not pages:
        return {"text": "No passage in the indexed technical documents matches this question.", "cited": []}

    context = "\n\n".join(f"[{p['title']}, p.{p['page']} - {p['doc_type']}]\n{p['text']}" for p in pages)
    check = crosscheck(FARMS[farm_id])
    if check and check["checks"]:
        lines = [
            f"- {c['field']}: datasheet p.{c['page']} says {c['datasheet']:g}, Windward's config uses "
            f"{c['configured']:g} ({c['status']})"
            for c in check["checks"]
        ]
        context += f"\n\n[Datasheet cross-check for {check['turbine_model']}]\n" + "\n".join(lines)

    farm = FARMS[farm_id]
    prompt = (
        f"Farm: {farm.name}, turbine model {farm.manufacturer_model}.\n"
        "Answer the question from the PDF pages below only, citing every fact as (document title, p.N), "
        "and keep any condition or limit a page attaches to it: a product's scope limits (\"Note: only ... "
        "are within the scope\") are part of the answer.\n"
        "- Use the pages' own words. When a page only lists items, list them; never add explanations of "
        "what each item means that the page doesn't give.\n"
        "- Pages of type datasheet or upgrade_sheet are the manufacturer's documents for this farm's "
        "turbine model.\n"
        "- Pages of type tender_site_description or leasing_round are about other, offshore projects. "
        "When the question asks about one of those projects, or about tenders or leasing, answer it "
        "fully from them and name the project. Only a question about this farm itself must not be "
        "answered from them.\n"
        "- Asked which options or variants the manufacturer offers, list every value the pages give "
        "(\"80 m or site-specific\" is one standard value plus site-specific ones), and say plainly when "
        "only one is given.\n"
        "- In an IEC class such as IA, the Roman numeral is the wind class and the letter the "
        "turbulence category.\n"
        "- Cite the p.N from the [title, p.N] header above each page, never a page number printed in "
        "the page's own text.\n"
        "- Answer every part the pages cover; name only the parts they really don't.\n\n"
        f"Pages:\n{context}\n\nQuestion: {question}"
    )
    response = complete([{"role": "user", "content": prompt}], temperature=0)
    answer, cited = check_citations(response.choices[0].message.content or "", pages)
    urls = sorted({f"{c['title']}: {c['url']}" for c in cited} or {f"{p['title']}: {p['url']}" for p in pages})
    return {"text": answer + "\n\nDocuments: " + "; ".join(urls), "cited": cited}


def get_farm_facts(farm_id: str) -> dict:
    """The dataset's own metadata (commissioning date, hub height per turbine) and production
    records over the full SCADA history (data_sources/farm_facts.py, README §17)."""
    from data_sources.farm_facts import farm_facts

    _check_farm(farm_id)
    return farm_facts(farm_id) or {"error": "farm facts not exported (python -m data_sources.farm_facts)"}


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
