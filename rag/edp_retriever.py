"""LangChain retrieval over the EDP Wind Farm A incident corpus (rag/edp_incident_corpus.py).
Separate from rag/langchain_retriever.py because this corpus isn't keyed to a Farm dataclass
entry — Wind Farm A has no disclosed coordinates or rated power (see data_sources/edp_scada.py)
so it doesn't go through data_sources.farms.FARMS at all.
"""
from pathlib import Path

from langchain_community.vectorstores import FAISS

from rag.edp_incident_corpus import build_corpus
from rag.embeddings import BedrockTitanEmbeddings

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "rag_index" / "edp_wind_farm_a"

_cache: FAISS | None = None


def get_vectorstore() -> FAISS:
    global _cache
    if _cache is not None:
        return _cache

    embeddings = BedrockTitanEmbeddings()
    if (INDEX_DIR / "index.faiss").exists():
        _cache = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    else:
        docs = build_corpus()
        _cache = FAISS.from_documents(docs, embeddings)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        _cache.save_local(str(INDEX_DIR))
    return _cache


def get_retriever(k: int = 4):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})
