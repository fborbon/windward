"""RAG corpus for EDP Wind Farm A's real labeled fault/normal case studies (see
data_sources/edp_scada.py) — a different content shape from the other three farms' rolling
fault-event log: 22 independent, real, windowed case studies, each labeled anomaly or normal,
11 of them with a real root-cause description (Gearbox failure, Generator bearing failure,
Transformer failure, Hydraulic group). Not keyed to a Farm dataclass entry — see
rag/edp_retriever.py.
"""
import pandas as pd
from langchain_core.documents import Document

from data_sources.edp_scada import STATUS_LABELS, load_events


def _event_documents() -> list[Document]:
    events = load_events()
    docs = []
    for event_id, e in events.iterrows():
        cause = f" Real root cause: {e['event_description']}." if pd.notna(e["event_description"]) and e["event_description"] else ""
        text = (
            f"EDP Wind Farm A, turbine (anonymized id) {e['asset']}, case study #{event_id}: "
            f"{e['event_label']} event, {e['duration_hours']:.0f} hours of real 10-minute SCADA "
            f"leading up to it.{cause} Timestamps in this dataset are anonymized (relative "
            "offsets only, not real calendar dates) to protect the source farm's identity."
        )
        docs.append(Document(page_content=text, metadata={"source": f"edp-case-study:{event_id}"}))
    return docs


def _reference_documents() -> list[Document]:
    provenance = Document(
        page_content=(
            "EDP Wind Farm A dataset provenance. Part of the CARE-to-Compare benchmark "
            "(Zenodo doi:10.5281/zenodo.15846963, CC BY-SA 4.0; Gück et al. 2024, 'CARE to "
            "Compare: A Real-World Benchmark Dataset for Early Fault Detection in Wind Turbine "
            "Data', MDPI Data 9(12):138). Wind Farm A is EDP's own onshore wind farm in "
            "Portugal (the original edp.com/en/innovation/data 'Wind Farm 1'), anonymized: "
            "turbine identity, exact coordinates, rated power/rotor diameter, and calendar "
            "timestamps are all stripped, and power/energy sensor channels are rescaled to a "
            "normalized fraction rather than real kW. Wind speed and turbine status codes are "
            "real, unscaled measurements. 22 of the source dataset's 95 case studies belong to "
            "this farm; 44 of the 95 overall are labeled anomalies."
        ),
        metadata={"source": "internal-reference:edp-provenance"},
    )
    status_codes = Document(
        page_content=(
            "EDP Wind Farm A real turbine operational status codes: "
            + "; ".join(f"{code} = {label}" for code, label in STATUS_LABELS.items())
            + ". Codes 0 and 2 are normal operation; 4 (Downtime) means the turbine is down due "
            "to a fault or other reason; 3 (Service) means a service team is on site."
        ),
        metadata={"source": "internal-reference:edp-status-codes"},
    )
    return [provenance, status_codes]


def build_corpus() -> list[Document]:
    return _event_documents() + _reference_documents()
