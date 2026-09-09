"""LangChain retrieval over the incident/reference corpus (see rag/incident_corpus.py).
FAISS index is built once per farm and cached to disk (data/rag_index/<farm_id>/) — rebuilding
means re-embedding ~60 documents via Bedrock Titan on every graph run otherwise.
"""
from pathlib import Path

from langchain_community.vectorstores import FAISS

from data_sources.farms import FARMS, Farm
from rag.embeddings import BedrockTitanEmbeddings
from rag.incident_corpus import build_corpus

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "rag_index"

_cache: dict[str, FAISS] = {}


def _build_and_save(farm: Farm, embeddings: BedrockTitanEmbeddings) -> FAISS:
    docs = build_corpus(farm)
    store = FAISS.from_documents(docs, embeddings)
    path = INDEX_DIR / farm.farm_id
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))
    return store


def get_vectorstore(farm_id: str) -> FAISS:
    if farm_id in _cache:
        return _cache[farm_id]

    farm = FARMS[farm_id]
    embeddings = BedrockTitanEmbeddings()
    path = INDEX_DIR / farm_id
    if (path / "index.faiss").exists():
        store = FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    else:
        store = _build_and_save(farm, embeddings)
    _cache[farm_id] = store
    return store


def get_retriever(farm_id: str, k: int = 4):
    return get_vectorstore(farm_id).as_retriever(search_kwargs={"k": k})
