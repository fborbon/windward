"""LangChain retrieval over the DSWE Inland-Offshore corpus (rag/dswe_incident_corpus.py).
Same pattern as rag/edp_retriever.py — this corpus isn't keyed to a Farm dataclass entry
either (no disclosed coordinates — see data_sources/dswe_scada.py).
"""
from pathlib import Path

from langchain_community.vectorstores import FAISS

from rag.dswe_incident_corpus import build_corpus
from rag.embeddings import BedrockTitanEmbeddings

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "rag_index" / "dswe_inland_offshore"

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
