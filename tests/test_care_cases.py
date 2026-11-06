"""CARE to Compare case studies (all three farms), served from the committed
dashboard/data/care_payload.json, so these run in CI without the 5.5 GB archive."""
from fastapi.testclient import TestClient

from analysis.care_score import care_score, EventResult
from data_sources.edp_scada import detector_summary, load_case, load_events, load_payload


def test_payload_covers_the_three_farms_and_95_cases():
    events = load_events()
    assert len(events) == 95 and events.index.is_unique
    assert events["farm"].value_counts().to_dict() == {"C": 58, "A": 22, "B": 15}
    assert set(events["event_label"]) == {"anomaly", "normal"}


def test_every_case_has_its_detector_result_and_charts():
    for c in load_payload()["cases"]:
        d = c["detector"]
        assert isinstance(d["detected"], bool) and d["max_criticality"] >= 0
        assert d["detected"] == (d["max_criticality"] >= 72)
        assert len(d["hourly"]["criticality"]) == len(d["hourly"]["hours_to_event_end"]) > 10
        assert d["top_sensors"] and c["power_curve"]
        if c["label"] == "anomaly" and d["detected"]:
            assert d["lead_hours"] is not None and d["lead_hours"] >= 0


def test_stored_scores_recompute_from_the_stored_cases():
    """The per-farm CARE scores in the payload follow from the per-case results it stores."""
    s = detector_summary()["scores"]
    cases = load_payload()["cases"]
    for farm in ("A", "B", "C", "all"):
        res = [EventResult(c["event_id"], c["label"], c["detector"]["detected"], c["detector"]["max_criticality"],
                           c["detector"]["coverage"], c["detector"]["accuracy"], c["detector"]["earliness"], None, None)
               for c in cases if farm == "all" or c["farm"] == farm]
        assert abs(care_score(res)["care"] - s[farm]["care"]) < 2e-3, farm


def test_api_routes():
    import api.main as m
    c = TestClient(m.app)
    allev = c.get("/edp/events").json()
    assert len(allev) == 95 and {"farm", "detected", "lead_hours"} <= set(allev[0])
    assert len(c.get("/edp/events", params={"farm": "B"}).json()) == 15
    assert c.get("/edp/events", params={"farm": "Z"}).status_code == 400
    eid = allev[0]["event_id"]
    assert c.get(f"/edp/events/{eid}").json()["event_id"] == eid
    assert c.get("/edp/events/99999").status_code == 404
    assert c.get("/edp/detector").json()["reference"]["autoencoder_per_farm_paper"] == 0.66


def test_corpus_has_a_document_per_case_plus_references():
    from rag.edp_incident_corpus import build_corpus
    docs = build_corpus()
    assert len(docs) == 95 + 3
    assert any("Wind Farm C (offshore, Germany)" in d.page_content for d in docs)
    assert load_case(load_events().index[0]) is not None


def test_subsystem_tags_follow_the_operator_wording():
    from data_sources.edp_scada import subsystem
    assert subsystem("Failure due to Rotorbrake and Hydraulic problemes") == "hydraulic"  # first rule wins
    assert subsystem("high temperature in transformer cell") == "transformer"
    assert subsystem("21002 : Axis 1 DC-link voltage low, batt") == "pitch system"  # a pitch axis drive
    assert subsystem("Converter Failure from 17.11 12:30 - 18.11. 13:57, Fuse Filter Supply") == "converter"
    assert subsystem("WEC in failure - hub battery charger defect") == "pitch system"
    assert subsystem(None) is None
