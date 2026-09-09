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


def get_recommendation(farm_id: str) -> dict:
    """Runs the full diagnose pipeline for the farm and returns its recommendation +
    supporting narrative — not a cached lookup, this re-derives it from current data."""
    if farm_id not in FARMS:
        raise ValueError(f"unknown farm_id {farm_id!r}, expected one of {list(FARMS)}")
    result = run_graph(_graph(), farm_id)
    return {**result["recommendation"], "explanation": result["explanation"]}


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
