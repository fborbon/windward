"""PDF ingestion for the technical-document RAG: download (sha256-checked), page-level text
extraction with PyMuPDF, watermark cleanup and sentence-aware chunking into LangChain
Documents that carry their source document, page and URL, so every retrieved passage can be
cited back to a real page of a real PDF.

PyMuPDF is only in requirements.txt (the jobs image): this module runs at index-build time
(rag/pdf_index_build.py), never on the API's request path, which only loads the built index
(rag/pdf_retriever.py).
"""
import hashlib
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from rag.pdf_sources import PdfSource

PDF_DIR = Path(__file__).resolve().parent.parent / "data" / "pdfs"

CHUNK_SIZE = 1200  # characters; a datasheet table or a tender requirement fits in one chunk
CHUNK_OVERLAP = 150
MIN_CHUNK_LENGTH = 80


@dataclass
class Page:
    doc_id: str
    page: int  # 1-based, as printed in a PDF viewer
    text: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(source: PdfSource, pdf_dir: Path = PDF_DIR) -> Path | None:
    """Cached download. Returns None (with a warning) when the file can't be fetched or its
    sha256 doesn't match the pinned one: a skipped document beats silently indexing a
    different one."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    path = pdf_dir / f"{source.doc_id}.pdf"
    if path.exists() and _sha256(path) == source.sha256:
        return path

    tmp = path.with_suffix(".part")
    try:
        req = urllib.request.Request(source.url, headers={"User-Agent": "Mozilla/5.0 (windward-rag-indexer)"})
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as f:
            f.write(resp.read())
    except OSError as e:
        print(f"WARN: {source.doc_id}: download failed ({e}); skipped")
        tmp.unlink(missing_ok=True)
        return None

    digest = _sha256(tmp)
    if digest != source.sha256:
        print(f"WARN: {source.doc_id}: sha256 {digest} != pinned {source.sha256}; skipped")
        tmp.unlink()
        return None
    tmp.replace(path)
    return path


def clean_text(text: str, drop_lines: tuple[str, ...] = ()) -> str:
    """Drop watermark-fragment lines, collapse whitespace inside lines, keep paragraph breaks."""
    drop = set(drop_lines)
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line not in drop]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_pages(path: Path, source: PdfSource) -> list[Page]:
    import pymupdf  # build-time only, see module docstring

    pages = []
    with pymupdf.open(str(path)) as doc:
        for i, page in enumerate(doc):
            text = clean_text(page.get_text("text"), source.drop_lines)
            if text:
                pages.append(Page(doc_id=source.doc_id, page=i + 1, text=text))
    return pages


def chunk_page(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP, min_length: int = MIN_CHUNK_LENGTH
) -> list[str]:
    """Fixed-size character windows with overlap, each end pulled back to the last sentence
    (or paragraph) boundary in the window's second half, so a requirement or a spec row isn't
    cut mid-sentence."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("\n\n", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        piece = text[start:end].strip()
        if len(piece) >= min_length:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def page_documents(pages: list[Page], source: PdfSource) -> list[Document]:
    docs = []
    for p in pages:
        for i, piece in enumerate(chunk_page(p.text)):
            docs.append(
                Document(
                    page_content=f"{source.title}, page {p.page}:\n{piece}",
                    metadata={
                        "doc_id": source.doc_id,
                        "title": source.title,
                        "publisher": source.publisher,
                        "doc_type": source.doc_type,
                        "url": source.url,
                        "page": p.page,
                        "chunk": i,
                        "applies_to": list(source.applies_to),
                    },
                )
            )
    return docs
