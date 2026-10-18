"""Registry of the real, public PDF documents behind the technical-document RAG (README §16).

Two kinds of document, both real and downloadable from the URL below:
- manufacturer documents for the turbine models this project's farms actually run
  (data_sources/farms.py `manufacturer_model`), scoped to those farms only;
- public offshore-wind tender documents (the Dutch government's Project and Site
  Descriptions, which state the site conditions and the technical requirements a bidder must
  meet) and a seabed-leasing round update, as general wind-project planning reference for
  every farm. None of them is about one of this project's farms, and the agent is told so.

The PDFs are not committed: the manufacturer brochures are the manufacturers' copyright, so
the index build (rag/pdf_index_build.py) downloads them into data/pdfs/ (gitignored) and every
answer cites the original URL and page. sha256 pins the exact file that was read and checked:
if a publisher swaps the file, the build skips it rather than silently indexing a different
document.
"""
from dataclasses import dataclass

ALL_FARMS = "*"


@dataclass(frozen=True)
class PdfSource:
    doc_id: str
    title: str
    publisher: str
    url: str
    sha256: str
    doc_type: str  # "datasheet" | "upgrade_sheet" | "tender_site_description" | "leasing_round"
    applies_to: tuple[str, ...]  # farm_ids, or (ALL_FARMS,)
    turbine_model: str | None = None  # set for a datasheet: the Farm.manufacturer_model it describes
    # Watermark fragments some hosts stamp on every page (one letter or syllable per text line);
    # dropped line by line before chunking so they don't pollute the embeddings.
    drop_lines: tuple[str, ...] = ()
    notes: str = ""


SOURCES: list[PdfSource] = [
    PdfSource(
        doc_id="siemens_swt_2_3_82_vs",
        title="Siemens Wind Turbine SWT-2.3-82 VS - Built on experience (technical specifications)",
        publisher="Siemens AG / Siemens Wind Power A/S, 2009",
        url="https://media.mwps.world/static/2025/04/siemens_wind_turbine_swt-2.3-82_technical_specifications.pdf",
        sha256="34ea7c1a803278e33083114c556e4184efda6c11a72e283bfb2c97ea1279043e",
        doc_type="datasheet",
        applies_to=("hill_of_towie",),
        turbine_model="Siemens SWT-2.3-VS-82",
        drop_lines=("M", "W", "PS", "or", "ld"),
        notes="Manufacturer brochure, copy hosted by MWPS World (its watermark is stripped).",
    ),
    PdfSource(
        doc_id="sgre_energyup_senvion",
        title="EnergyUp Full Power Curve Advanced - Senvion technology (MM82, MM92, MM100 upgrade)",
        publisher="Siemens Gamesa Renewable Energy (Senvion's service successor)",
        url=(
            "https://assets.siemens-energy.com/dam/4efd451c-6c1c-4c16-9c6a-b34e0118222a/"
            "siemens-gamesa-energyup-full-power-curve-advanced-senvion-pdf_Original%20file.pdf"
        ),
        sha256="8228f7715da7bef0535e0ce688f62a4a5d24fbc38a644707f50207f016206ae8",
        doc_type="upgrade_sheet",
        applies_to=("kelmarsh", "penmanshiel"),
        notes=(
            "The only official Senvion-platform document found online; Senvion's own MM82/MM92 "
            "datasheets are not publicly hosted since its 2019 insolvency. An uprate offer, not a "
            "datasheet, so it is never used to cross-check the farms' rated power."
        ),
    ),
    PdfSource(
        doc_id="rvo_borssele_1_2_psd",
        title="Borssele Wind Farm Sites I and II - Project and Site Description, Version 3 (April 2016)",
        publisher="Netherlands Enterprise Agency (RVO)",
        url="https://offshorewind.rvo.nl/file/download/43061512",
        sha256="df0c2dc3678b8d4c8667166a103b3ffea688c61b6c5156d08d4a27e2da18d0c0",
        doc_type="tender_site_description",
        applies_to=(ALL_FARMS,),
    ),
    PdfSource(
        doc_id="rvo_hkz_1_2_psd",
        title="Hollandse Kust (zuid) Wind Farm Sites I and II - Project and Site Description (October 2017)",
        publisher="Netherlands Enterprise Agency (RVO)",
        url="https://offshorewind.rvo.nl/file/download/54309512",
        sha256="dbcca7d9b6eb9b575f7fc90d980d259f9986690e3f95961cc9ae44c40e94a748",
        doc_type="tender_site_description",
        applies_to=(ALL_FARMS,),
    ),
    PdfSource(
        doc_id="ces_scotwind_2019_update",
        title="ScotWind Leasing - April 2019 update (working draft)",
        publisher="Crown Estate Scotland",
        url="https://www.crownestatescotland.com/sites/default/files/2023-07/scotwind-leasing-april-2019-update.pdf",
        sha256="4b0453919f54ac0d41fded825240da641dd82fce1db0ec2f75a9cabb971910a6",
        doc_type="leasing_round",
        applies_to=(ALL_FARMS,),
        notes="Application process and requirements of Scotland's offshore seabed-leasing round.",
    ),
]

BY_ID: dict[str, PdfSource] = {s.doc_id: s for s in SOURCES}


def sources_for_farm(farm_id: str) -> list[PdfSource]:
    return [s for s in SOURCES if ALL_FARMS in s.applies_to or farm_id in s.applies_to]
