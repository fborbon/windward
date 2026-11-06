"""RAG corpus for the CARE to Compare case studies (data_sources/edp_scada.py): 95 real, labelled,
anonymised turbine cases from three farms (A: EDP onshore, Portugal; B and C: offshore,
Germany), each an anomaly leading up to a fault (often with the operator's root cause) or normal
behaviour, plus what Windward's normal-behaviour detector (analysis/care_nbm.py) made of it.
Not keyed to a Farm dataclass entry: see rag/edp_retriever.py.
"""
from langchain_core.documents import Document

from data_sources.edp_scada import STATUS_LABELS, detector_summary, load_payload

FARM_WORDS = {"A": "Wind Farm A (EDP, onshore, Portugal)", "B": "Wind Farm B (offshore, Germany)",
              "C": "Wind Farm C (offshore, Germany)"}


def _case_document(c: dict) -> Document:
    d = c["detector"]
    cause = f" Operator's root cause: {c['description']}." if c["description"] else ""
    if c["label"] == "anomaly":
        outcome = (f" Windward's normal-behaviour detector caught it {d['lead_hours']:.0f} hours before the event end"
                   if d["detected"] and d["lead_hours"] is not None else " Windward's normal-behaviour detector missed it")
    else:
        outcome = (" Windward's normal-behaviour detector raised a false alarm on it" if d["detected"]
                   else " Windward's normal-behaviour detector correctly stayed quiet")
    sensors = "; ".join(f"{s['description'] or s['sensor']} ({s['share_of_error_pct']:.0f}%)" for s in d["top_sensors"])
    text = (f"{FARM_WORDS[c['farm']]}, turbine (anonymised id) {c['turbine_id']}, case study #{c['event_id']}: "
            f"{c['label']} event over a {c['duration_hours']:.0f}-hour window.{cause}{outcome} "
            f"(peak criticality {d['max_criticality']}, alarm at 72). Sensors carrying most of the model's error in the "
            f"event window: {sensors}. Timestamps are anonymised (relative offsets, not calendar dates).")
    return Document(page_content=text, metadata={"source": f"care-case-study:{c['farm']}:{c['event_id']}"})


def _reference_documents() -> list[Document]:
    s = detector_summary()
    sc = s["scores"]
    provenance = Document(
        page_content=(
            "CARE to Compare dataset provenance: Zenodo doi:10.5281/zenodo.15846963, CC BY-SA 4.0; Gück et al. 2024, "
            "'CARE to Compare: A Real-World Benchmark Dataset for Early Fault Detection in Wind Turbine Data', MDPI "
            "Data 9(12):138. 95 cases on 36 turbines: Wind Farm A (EDP's onshore farm in Portugal, 5 turbines, 22 cases), "
            "Wind Farm B (offshore Germany, 9 turbines, 15 cases), Wind Farm C (offshore Germany, 22 turbines, 58 cases). "
            "All anonymised: no coordinates, no rated power, shifted timestamps, rescaled power."),
        metadata={"source": "internal-reference:care-provenance"},
    )
    detector = Document(
        page_content=(
            f"Windward's normal-behaviour detector on CARE to Compare: {s['method']} CARE score {sc['all']['care']} over all "
            f"95 cases (A {sc['A']['care']}, B {sc['B']['care']}, C {sc['C']['care']}); it caught "
            f"{sc['all']['anomalies_detected']} of {sc['all']['anomaly_events']} anomaly cases with "
            f"{sc['all']['false_alarms']} false alarms on {sc['all']['normal_events']} normal cases. For reference, the "
            f"benchmark paper's per-farm autoencoder scores {s['reference']['autoencoder_per_farm_paper']} and random "
            f"guessing {s['reference']['random']}."),
        metadata={"source": "internal-reference:care-detector"},
    )
    status_codes = Document(
        page_content=("CARE to Compare turbine operational status codes: "
                      + "; ".join(f"{code} = {label}" for code, label in STATUS_LABELS.items())
                      + ". Codes 0 and 2 are normal operation; 4 (Downtime) means the turbine is down due to a fault "
                        "or other reason; 3 (Service) means a service team is on site."),
        metadata={"source": "internal-reference:care-status-codes"},
    )
    return [provenance, detector, status_codes]


def build_corpus() -> list[Document]:
    return [_case_document(c) for c in load_payload()["cases"]] + _reference_documents()
