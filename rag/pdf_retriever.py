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
from rag.pdf_sources import ALL_FARMS, BY_ID, sources_for_farm

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "pdf_index"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
PAGES_PATH = INDEX_DIR / "pages.json"

# Manufacturer documents short enough to read whole: the farm's own turbine documents are always
# in the answer's context, because a spec question ("which hub heights does it come in?") often
# matches the brochure's prose pages while the answer sits in its spec table (README §17).
WHOLE_DOC_TYPES = ("datasheet", "upgrade_sheet")
WHOLE_DOC_MAX_CHARS = 20_000

_cache: FAISS | None = None
_pages: dict | None = None


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


def load_pages() -> dict:
    """{doc_id: {page: text}} written by rag/pdf_index_build.py; empty for an older index."""
    global _pages
    if _pages is None:
        _pages = json.loads(PAGES_PATH.read_text()) if PAGES_PATH.exists() else {}
    return _pages


def _page_hit(doc, text: str) -> dict:
    return {**{k: doc.metadata[k] for k in ("doc_id", "title", "doc_type", "url", "page")}, "text": text}


def search_pages(farm_id: str, question: str, max_pages: int = 4) -> list[dict]:
    """The pages to answer from: the farm's own turbine documents in full, then the pages whose
    chunks match the question best, each read whole. A chunk is 1,200 characters, so the
    sentence that answers a question is often in the chunk next to the one that matched (the
    Hollandse Kust (zuid) mean wind speed sits two chunks after the passage about how it was
    measured); reading the whole page removes that miss."""
    pages = load_pages()
    out, seen = [], set()
    for source in sources_for_farm(farm_id):
        doc_pages = pages.get(source.doc_id) or {}
        if source.doc_type in WHOLE_DOC_TYPES and sum(map(len, doc_pages.values())) <= WHOLE_DOC_MAX_CHARS:
            for page, text in doc_pages.items():
                out.append({"doc_id": source.doc_id, "title": source.title, "doc_type": source.doc_type,
                            "url": source.url, "page": int(page), "text": text})
                seen.add((source.doc_id, int(page)))
    found = 0
    for doc in search(farm_id, question, k=12):
        key = (doc.metadata["doc_id"], doc.metadata["page"])
        if key in seen:
            continue
        seen.add(key)
        text = (pages.get(key[0]) or {}).get(str(key[1]))
        # an index built before pages.json existed: fall back to the chunk itself
        out.append(_page_hit(doc, text if text else doc.page_content))
        found += 1
        if found == max_pages:
            break
    return out


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
