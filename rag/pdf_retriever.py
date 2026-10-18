"""Retrieval over the technical-document PDFs (rag/pdf_sources.py): one FAISS index over every
document, filtered per farm at query time by each chunk's `applies_to`, so Hill of Towie's
questions see the Siemens SWT-2.3-82 VS brochure and the tender documents but never the Senvion
upgrade sheet.

Load-only: the index is built offline by rag/pdf_index_build.py (jobs image, PyMuPDF) on every
deploy, so the slim API image never parses a PDF. Unlike rag/langchain_retriever.py there is no
build-on-first-request fallback - a missing index returns no results and says so.
"""
import json
from pathlib import Path

from langchain_community.vectorstores import FAISS

from rag.embeddings import BedrockTitanEmbeddings
from rag.pdf_sources import ALL_FARMS, BY_ID

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "pdf_index"
MANIFEST_PATH = INDEX_DIR / "manifest.json"

_cache: FAISS | None = None


def index_built() -> bool:
    return (INDEX_DIR / "index.faiss").exists()


def get_vectorstore() -> FAISS | None:
    global _cache
    if _cache is None and index_built():
        _cache = FAISS.load_local(str(INDEX_DIR), BedrockTitanEmbeddings(), allow_dangerous_deserialization=True)
    return _cache


def search(farm_id: str, question: str, k: int = 5) -> list:
    store = get_vectorstore()
    if store is None:
        return []
    # Over-fetch then filter: the corpus is a few hundred chunks, and filtering in Python keeps
    # this independent of which filter syntax the installed langchain FAISS wrapper supports.
    hits = store.similarity_search(question, k=max(k * 6, 30))
    allowed = [d for d in hits if ALL_FARMS in d.metadata["applies_to"] or farm_id in d.metadata["applies_to"]]
    return allowed[:k]


def indexed_documents(farm_id: str | None = None) -> list[dict]:
    """The documents actually in the built index (from its manifest), with page counts."""
    if not MANIFEST_PATH.exists():
        return []
    manifest = json.loads(MANIFEST_PATH.read_text())
    out = []
    for doc_id, info in manifest["documents"].items():
        source = BY_ID.get(doc_id)
        if source is None or (farm_id and ALL_FARMS not in source.applies_to and farm_id not in source.applies_to):
            continue
        out.append({
            "doc_id": doc_id,
            "title": source.title,
            "publisher": source.publisher,
            "doc_type": source.doc_type,
            "url": source.url,
            "pages": info["pages"],
            "chunks": info["chunks"],
        })
    return out
