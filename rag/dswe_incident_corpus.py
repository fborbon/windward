"""RAG corpus for the DSWE Inland-Offshore dataset (see data_sources/dswe_scada.py) — a
different shape again from the other two corpora: no fault/incident log here (this dataset
documents turbine performance and met-mast environmental variables, not maintenance events),
so the real content is per-turbine/mast facts plus reference notes on the variables and the
Measure-Correlate-Predict methodology used to make the missing location workable.
"""
from langchain_core.documents import Document

from data_sources.dswe_scada import TURBINES, load_turbine


def _turbine_documents() -> list[Document]:
    docs = []
    for turbine_id, meta in TURBINES.items():
        df = load_turbine(turbine_id)
        text = (
            f"DSWE Inland-Offshore Wind Farm Dataset1, turbine {turbine_id}, {meta['site']} site, "
            f"paired with met mast {meta['mast']}. Real 10-minute data from {meta['period_start']} "
            f"to {meta['period_end']} (documented date range; individual rows have no timestamp, "
            f"only a sequence number — {len(df)} real rows). Mean mast wind speed "
            f"{df['V'].mean():.2f} m/s, mean turbine output {df['y (% relative to rated power)'].mean():.1f}% "
            f"of rated power. No geographic coordinates are disclosed for this farm."
        )
        docs.append(Document(page_content=text, metadata={"source": f"dswe-turbine:{turbine_id}"}))
    return docs


def _reference_documents() -> list[Document]:
    provenance = Document(
        page_content=(
            "DSWE Inland-Offshore Wind Farm Dataset1 provenance. Published alongside Yu Ding's "
            "book Data Science for Wind Energy (Chapter 5), Zenodo doi:10.5281/zenodo.5516552, "
            "CC BY 4.0. Six real turbines (WT1-WT6) from two real wind farms (inland and "
            "offshore), each paired with one of three real on-site meteorological masts: WT1+WT2 "
            "with Mast A (inland), WT3+WT4 with Mast B (inland), WT5+WT6 with Mast C (offshore). "
            "No coordinates are disclosed anywhere in the dataset or the book."
        ),
        metadata={"source": "internal-reference:dswe-provenance"},
    )
    variables = Document(
        page_content=(
            "DSWE Inland-Offshore Wind Farm Dataset1 variable definitions (from the dataset's own "
            "documentation). V: wind speed (m/s). D: wind direction (degrees). air density: kg/m3. "
            "humidity: relative humidity (offshore mast only). I: turbulence intensity. S_a: "
            "above-hub-height wind shear (offshore mast only). S_b: below-hub-height wind shear. "
            "y: turbine power output as a percentage of rated capacity. All measured at the mast "
            "except y, which is measured at the turbine."
        ),
        metadata={"source": "internal-reference:dswe-variables"},
    )
    mcp = Document(
        page_content=(
            "Measure-Correlate-Predict (MCP) methodology, as applied to this dataset (see "
            "analysis.efficiency.measure_correlate_predict). A classical wind-resource-assessment "
            "technique: Measure a real but short or incomplete on-site wind-speed record; "
            "Correlate it against a real long-term reference (here, Open-Meteo's ERA5 reanalysis) "
            "over the same real period, producing a scale ratio; Predict local conditions by "
            "applying that ratio to the reference's other values (here, a live forecast). This "
            "dataset has no disclosed location, so the reference point is a real but explicitly "
            "illustrative coordinate the caller supplies, not a claim about the farm's real "
            "siting — the computed ratio and how far it sits from 1.0 is itself evidence of how "
            "representative that reference point is, not a fixed assumption."
        ),
        metadata={"source": "internal-reference:dswe-mcp-methodology"},
    )
    return [provenance, variables, mcp]


def build_corpus() -> list[Document]:
    return _turbine_documents() + _reference_documents()
