"""Technical-document PDF RAG (README §16): ingestion, chunking, per-farm scoping, and the
datasheet extraction checks - no network and no Bedrock calls."""
import json
from dataclasses import replace

import pytest
from langchain_core.documents import Document

from data_sources.farms import FARMS
from rag import datasheet_specs, pdf_corpus, pdf_retriever
from rag.pdf_sources import ALL_FARMS, BY_ID, SOURCES, sources_for_farm


def _make_pdf(path, pages: list[str]):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_registry_is_consistent():
    assert len(BY_ID) == len(SOURCES)
    models = {f.manufacturer_model for f in FARMS.values()}
    for s in SOURCES:
        assert len(s.sha256) == 64 and s.url.startswith("https://")
        assert all(f == ALL_FARMS or f in FARMS for f in s.applies_to)
        if s.doc_type == "datasheet":
            assert s.turbine_model in models


def test_each_farm_sees_only_its_own_turbine_documents():
    hot = {s.doc_id for s in sources_for_farm("hill_of_towie")}
    kel = {s.doc_id for s in sources_for_farm("kelmarsh")}
    assert "siemens_swt_2_3_82_vs" in hot and "siemens_swt_2_3_82_vs" not in kel
    assert "sgre_energyup_senvion" in kel and "sgre_energyup_senvion" not in hot
    assert "rvo_borssele_1_2_psd" in hot & kel


def test_clean_text_drops_watermark_lines():
    text = "Rotor\nM\nDiameter   82.4 m\nW\nor\nld\n\n\n\nGenerator"
    assert pdf_corpus.clean_text(text, ("M", "W", "or", "ld")) == "Rotor\nDiameter 82.4 m\n\nGenerator"


def test_chunk_page_respects_size_overlap_and_sentences():
    text = " ".join(f"Requirement {i} shall be met by the bidder." for i in range(200))
    chunks = pdf_corpus.chunk_page(text, chunk_size=500, overlap=60, min_length=20)
    assert len(chunks) > 5
    assert all(len(c) <= 500 for c in chunks)
    assert all(c.endswith(".") for c in chunks[:-1])  # never cut mid-sentence
    assert chunks[0][-30:] in text and chunks[1][:20] in chunks[0]  # overlap carried over
    assert pdf_corpus.chunk_page("too short", min_length=80) == []


def test_extract_pages_and_documents_carry_citation_metadata(tmp_path):
    path = tmp_path / "sheet.pdf"
    _make_pdf(path, ["Rotor diameter 82.4 m.\nRated power 2,300 kW.\nSwept area 5,300 m2.\nGearbox 3-stage planetary/helical.", "", "Cut-out wind speed 25 m/s.\nCut-in wind speed 3-5 m/s.\nRated power reached at 13-14 m/s."])
    source = BY_ID["siemens_swt_2_3_82_vs"]
    pages = pdf_corpus.extract_pages(path, source)
    assert [p.page for p in pages] == [1, 3]  # blank page skipped, numbering kept
    docs = pdf_corpus.page_documents(pages, replace(source, title="T"))
    assert docs[1].metadata["page"] == 3 and docs[1].metadata["url"] == source.url
    assert docs[0].page_content.startswith("T, page 1:")


def test_download_rejects_a_file_with_the_wrong_sha256(tmp_path):
    src_pdf = tmp_path / "other.pdf"
    _make_pdf(src_pdf, ["not the pinned document"])
    source = replace(BY_ID["rvo_borssele_1_2_psd"], url=src_pdf.as_uri())
    assert pdf_corpus.download(source, tmp_path / "cache") is None
    assert not (tmp_path / "cache" / f"{source.doc_id}.pdf").exists()

    good = replace(source, sha256=pdf_corpus._sha256(src_pdf))
    assert pdf_corpus.download(good, tmp_path / "cache").exists()


def test_search_filters_by_farm(monkeypatch):
    class FakeStore:
        def similarity_search(self, q, k):
            return [
                Document(page_content="senvion", metadata={"applies_to": ["kelmarsh", "penmanshiel"]}),
                Document(page_content="siemens", metadata={"applies_to": ["hill_of_towie"]}),
                Document(page_content="tender", metadata={"applies_to": [ALL_FARMS]}),
            ]

    monkeypatch.setattr(pdf_retriever, "get_vectorstore", lambda: FakeStore())
    assert [d.page_content for d in pdf_retriever.search("hill_of_towie", "q")] == ["siemens", "tender"]
    assert [d.page_content for d in pdf_retriever.search("kelmarsh", "q", k=1)] == ["senvion"]


def test_verify_checks_the_cited_page():
    page = "Diameter 82.4 m\nNominal power 2,300 kW\nGearbox type 3-stage planetary/helical"
    assert datasheet_specs.verify("rotor_diameter_m", 82.4, page)
    assert datasheet_specs.verify("rated_power_kw", 2300, page)
    assert datasheet_specs.verify("gearbox", "3-stage planetary/helical", page)
    assert not datasheet_specs.verify("rotor_diameter_m", 93, page)  # a hallucinated value
    assert datasheet_specs.verify("rated_power_kw", 2300, "Rated power 2.3 MW")  # printed in MW
    assert datasheet_specs.verify("iec_class", "IEC Class IA", "a high-wind (IEC\nClass IA) version")


def test_extract_specs_marks_unverified_values():
    class Msg:
        content = json.dumps({
            "rotor_diameter_m": {"value": 82.4, "page": 6},
            "rated_power_kw": {"value": 2500, "page": 6},  # wrong: not on page 6
            "iec_class": None,
            "rotor_speed_rpm": {"value": "6-18", "page": 3},
        })

    class Resp:
        choices = [type("C", (), {"message": Msg})]

    pages = [pdf_corpus.Page("d", 4, "Rotor speed 6-18 rpm"), pdf_corpus.Page("d", 6, "Diameter 82.4 m\nNominal power 2,300 kW")]
    specs = datasheet_specs.extract_specs(pages, lambda *a, **k: Resp)
    assert specs["rotor_diameter_m"] == {"value": 82.4, "page": 6, "verified": True}
    assert specs["rated_power_kw"]["verified"] is False
    assert "iec_class" not in specs
    # cited the brochure's printed page number, found on the real PDF page instead
    assert specs["rotor_speed_rpm"] == {"value": "6-18", "page": 4, "verified": True, "cited_page": 3}


def test_crosscheck_compares_only_verified_fields(tmp_path, monkeypatch):
    specs_path = tmp_path / "datasheet_specs.json"
    specs_path.write_text(json.dumps({"siemens_swt_2_3_82_vs": {
        "rotor_diameter_m": {"value": 82.4, "page": 6, "verified": True},
        "rated_power_kw": {"value": "2,300", "page": 6, "verified": True},
        "cut_out_wind_speed_ms": {"value": 25, "page": 6, "verified": True},
    }}))
    monkeypatch.setattr(datasheet_specs, "SPECS_PATH", specs_path)

    check = datasheet_specs.crosscheck(FARMS["hill_of_towie"])
    by_field = {c["field"]: c for c in check["checks"]}
    assert by_field["rated_power_kw"]["status"] == "match"
    assert by_field["rotor_diameter_m"]["status"] == "match"  # farms.py corrected to the datasheet's 82.4 m

    rounded = replace(FARMS["hill_of_towie"], rotor_diameter_m=82.0)  # the dataset metadata's value
    by_field = {c["field"]: c for c in datasheet_specs.crosscheck(rounded)["checks"]}
    assert by_field["rotor_diameter_m"] == {
        "field": "rotor_diameter_m", "datasheet": 82.4, "configured": 82.0, "page": 6, "status": "differs",
    }
    assert datasheet_specs.crosscheck(FARMS["kelmarsh"]) is None  # no Senvion datasheet indexed


def test_technical_documents_endpoint():
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    assert client.get("/analysis/nowhere/technical-documents").status_code == 404
    body = client.get("/analysis/hill_of_towie/technical-documents").json()
    assert set(body) == {"documents", "datasheet_check"}  # empty/None until the index is built
