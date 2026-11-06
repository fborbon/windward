"""LangChain retrieval over the CARE to Compare case-study corpus (rag/edp_incident_corpus.py).
Separate from rag/langchain_retriever.py because this corpus isn't keyed to a Farm dataclass
entry: the three farms have no disclosed coordinates or rated power (data_sources/edp_scada.py).
The index lives under a name tied to the corpus version, so a corpus change builds a new one
instead of serving the old Farm-A-only index.
"""
from pathlib import Path

from langchain_community.vectorstores import FAISS

from rag.edp_incident_corpus import build_corpus
from rag.embeddings import BedrockTitanEmbeddings

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "rag_index" / "care_to_compare_v2"

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
