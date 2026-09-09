"""Builds the RAG document corpus: real turbine fault/status events (Stop + significant
Warning entries from the SCADA status log) plus a small set of factual technical reference
notes (turbine specs, IEC/Betz methodology). Two different content shapes deliberately —
structured event records vs. prose reference — to exercise realistic retrieval.
"""
from langchain_core.documents import Document

from data_sources.farms import Farm, loader_for

MIN_DURATION_HOURS = 1.0
MAX_EVENTS = 60  # keep the embedding call count small — longest-duration events are the ones worth retrieving


def _duration_hours(duration: str) -> float:
    try:
        h, m, s = duration.split(":")
        return int(h) + int(m) / 60 + int(s) / 3600
    except (ValueError, AttributeError):
        return 0.0


def _event_documents(farm: Farm) -> list[Document]:
    events = loader_for(farm).load_status_events(farm.scada_zips)
    events["duration_hours"] = events["duration"].apply(_duration_hours)
    significant = events[
        (events["status"].isin(["Stop", "Warning"])) & (events["duration_hours"] >= MIN_DURATION_HOURS)
    ].sort_values("duration_hours", ascending=False).head(MAX_EVENTS)

    docs = []
    for _, e in significant.iterrows():
        text = (
            f"{farm.name}, {e['turbine_id'].replace('_', ' ')}: {e['status']} event, "
            f"code {e['code']} — {e['message']}. Started {e['start']:%Y-%m-%d %H:%M}, "
            f"lasted {e['duration_hours']:.1f} hours. IEC category: {e['iec_category'] or 'unclassified'}."
        )
        docs.append(Document(page_content=text, metadata={"source": f"scada-event:{e['turbine_id']}:{e['code']}:{e['start']}"}))
    return docs


def _reference_documents(farm: Farm) -> list[Document]:
    specs = Document(
        page_content=(
            f"{farm.name} turbine reference. {len(farm.turbine_ids)}x {farm.manufacturer_model}, rated "
            f"{farm.rated_power_kw / 1000:.2f} MW each, rotor diameter {farm.rotor_diameter_m} m. "
            "Typical cut-in wind speed ~3 m/s, rated wind speed ~12-13 m/s, cut-out ~25 m/s. "
            "Power curve shape follows the standard cubic ramp between cut-in and rated speed "
            "(P proportional to v^3), then flat at rated capacity until cut-out."
        ),
        metadata={"source": "internal-reference:turbine-spec"},
    )
    methodology = Document(
        page_content=(
            "Wind-resource extraction efficiency methodology (Windward internal reference). "
            "Power curves are binned per IEC 61400-12-1 convention: 0.5 m/s bins from 0-20 m/s, "
            "mean power per bin. The power coefficient Cp = actual power / kinetic power available "
            "in the wind swept by the rotor (P_wind = 0.5 * air_density * swept_area * wind_speed^3). "
            "The Betz limit (16/27, ~0.593) is the theoretical maximum Cp any turbine can achieve; "
            "real turbines typically peak around 0.45-0.50. A measured Cp above the Betz limit is not "
            "real over-unity extraction — it indicates a data or sensor issue, most often nacelle "
            "anemometer bias from being downstream of the rotor."
        ),
        metadata={"source": "internal-reference:efficiency-methodology"},
    )
    curtailment = Document(
        page_content=(
            "GB grid curtailment context (Windward internal reference). UK onshore wind farms can be "
            "curtailed by National Grid ESO for balancing or local network constraint reasons, "
            "independent of wind resource availability. A period of low actual production despite "
            "strong forecast wind should not automatically be read as a turbine fault — cross-check "
            "against curtailment/constraint signals before recommending a physical inspection."
        ),
        metadata={"source": "internal-reference:gb-curtailment"},
    )
    return [specs, methodology, curtailment]


def build_corpus(farm: Farm) -> list[Document]:
    return _event_documents(farm) + _reference_documents(farm)
