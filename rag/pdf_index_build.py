"""`python -m rag.pdf_index_build [--force]` - downloads the technical-document PDFs
(rag/pdf_sources.py), extracts and chunks them (rag/pdf_corpus.py), embeds the chunks with
Bedrock Titan into one FAISS index in data/pdf_index/, and extracts structured turbine specs
from each datasheet (rag/datasheet_specs.py).

Runs in the jobs image from infra/deploy_host.sh on every deploy, but only rebuilds when the
source list or chunking parameters changed (manifest.json fingerprint), so an ordinary deploy
costs no Bedrock calls. A document whose download or sha256 check fails is left out of the
index with a warning instead of failing the deploy.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone

from langchain_community.vectorstores import FAISS

from rag import datasheet_specs, pdf_corpus
from rag.embeddings import BedrockTitanEmbeddings
from rag.pdf_retriever import INDEX_DIR, MANIFEST_PATH, index_built
from rag.pdf_sources import SOURCES


def fingerprint() -> str:
    parts = [f"{s.doc_id}:{s.sha256}:{','.join(s.applies_to)}:{','.join(s.drop_lines)}" for s in SOURCES]
    parts.append(f"chunk:{pdf_corpus.CHUNK_SIZE}:{pdf_corpus.CHUNK_OVERLAP}:{pdf_corpus.MIN_CHUNK_LENGTH}")
    parts.append(f"specs:{','.join(datasheet_specs.SPEC_FIELDS)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def up_to_date() -> bool:
    if not (index_built() and MANIFEST_PATH.exists()):
        return False
    manifest = json.loads(MANIFEST_PATH.read_text())
    # a document skipped last time (download failure) is retried on the next deploy
    return manifest.get("fingerprint") == fingerprint() and len(manifest["documents"]) == len(SOURCES)


def build() -> dict:
    from agents.llm_router import complete

    docs, documents, specs = [], {}, {}
    for source in SOURCES:
        path = pdf_corpus.download(source)
        if path is None:
            continue
        pages = pdf_corpus.extract_pages(path, source)
        chunks = pdf_corpus.page_documents(pages, source)
        docs.extend(chunks)
        documents[source.doc_id] = {"pages": len(pages), "chunks": len(chunks), "sha256": source.sha256}
        print(f"{source.doc_id}: {len(pages)} pages -> {len(chunks)} chunks")
        if source.doc_type == "datasheet":
            specs[source.doc_id] = datasheet_specs.extract_specs(pages, complete)
            verified = sum(v["verified"] for v in specs[source.doc_id].values())
            print(f"{source.doc_id}: {len(specs[source.doc_id])} spec fields extracted, {verified} verified on their page")

    if not docs:
        raise SystemExit("no PDF could be downloaded - index left as it was")

    store = FAISS.from_documents(docs, BedrockTitanEmbeddings())
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))
    datasheet_specs.SPECS_PATH.write_text(json.dumps(specs, indent=1))
    manifest = {
        "fingerprint": fingerprint(),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunks": len(docs),
        "embedded_chars": sum(len(d.page_content) for d in docs),
        "documents": documents,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rebuild even if the manifest is current")
    args = parser.parse_args()
    if not args.force and up_to_date():
        print("technical-document index up to date, nothing to do")
        return
    manifest = build()
    print(f"technical-document index built: {manifest['chunks']} chunks from {len(manifest['documents'])} PDFs")


if __name__ == "__main__":
    main()
