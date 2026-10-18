"""The /ask agent's evaluation fixes (README §17): citation re-checking, answer cleaning, the
ranking notes, whole-page retrieval, farm facts, and the evaluation grader - no Bedrock calls."""
from langchain_core.documents import Document

from agents import qa_agent, turbine_tools
from data_sources.farm_facts import farm_facts
from mcp_server.tools import check_citations
from rag import pdf_retriever
from rag_eval.questions import QUESTIONS
from rag_eval.run import grade

HKZ = "Hollandse Kust (zuid) PSD"


def _page(n, text, doc_id="rvo_hkz_1_2_psd", title=HKZ):
    return {"doc_id": doc_id, "title": title, "url": "https://example.org", "page": n, "text": text}


def test_check_citations_moves_a_printed_page_number_to_the_pdf_page():
    pages = [_page(14, "water depth ranging from 18.1 m to 27.9 m bLAT"), _page(20, "sand waves 2 - 6 m"),
             _page(21, "long-term mean wind speed 9.44 ± 0.38 m/s at 100 m MSL")]
    answer = f"Depth 18.1 m to 27.9 m ({HKZ}, p.20). Mean wind 9.44 ± 0.38 m/s at 100 m ({HKZ}, p.21)."
    fixed, cited = check_citations(answer, pages)
    assert f"({HKZ}, p.14)" in fixed and f"({HKZ}, p.21)" in fixed
    assert [c["page"] for c in cited] == [14, 21]
    # a claim whose numbers are on no page in the context keeps its citation untouched
    assert check_citations(f"Capacity 700 MW ({HKZ}, p.20).", pages)[0].endswith("p.20).")


def test_clean_strips_leaked_reasoning_and_broken_links():
    assert qa_agent._clean("I will list them. </thinking>\nThe answer.") == "The answer."
    assert qa_agent._clean("<thinking>plan</thinking>Answer") == "Answer"
    assert qa_agent._clean("Rotor 82.4 m ([Siemens SWT, p.6](document_title, p.6)).") == "Rotor 82.4 m ((Siemens SWT, p.6))."


def test_citations_are_reattached_only_when_the_rewrite_dropped_them():
    cited = [{"title": "Siemens SWT", "url": "u", "page": 6}, {"title": "Siemens SWT", "url": "u", "page": 4}]
    assert qa_agent._with_citations("Cut-out is 25 m/s.", cited).endswith("Sources: Siemens SWT, p.6, p.4")
    assert qa_agent._with_citations("Cut-out is 25 m/s (Siemens SWT, p.6).", cited) == "Cut-out is 25 m/s (Siemens SWT, p.6)."


def test_yaw_ranking_keeps_inconclusive_estimates_labeled():
    out = turbine_tools.rank_turbines("kelmarsh", "abs_yaw_misalignment_deg", {})
    assert out["ranking"], "every Kelmarsh estimate is inconclusive; they must still be listed"
    assert out["ranking"][0]["turbine_id"] == "Kelmarsh_4" and out["ranking"][0]["status"] == "inconclusive"
    assert "no reliable estimate" in out["note"]


def test_peak_cp_ranking_flags_values_above_betz():
    analysis = {"efficiency_summary": [
        {"turbine_id": "HillOfTowie_18", "capacity_factor": 0.19, "peak_cp": 0.5995, "over_betz": True},
        {"turbine_id": "HillOfTowie_20", "capacity_factor": 0.25, "peak_cp": 0.5833, "over_betz": False},
    ]}
    out = turbine_tools.rank_turbines("hill_of_towie", "peak_cp", analysis)
    assert out["ranking"][0] == {"turbine_id": "HillOfTowie_18", "peak_cp": 0.5995, "over_betz_limit": True}
    assert "physically impossible" in out["note"]
    assert qa_agent._highlights(analysis)["highest_peak_cp"]["turbine_id"] == "HillOfTowie_18"


def test_search_pages_reads_whole_pages_and_the_farms_own_datasheet(monkeypatch):
    class FakeStore:
        def similarity_search(self, q, k):
            meta = {"doc_id": "rvo_hkz_1_2_psd", "title": HKZ, "doc_type": "tender_site_description",
                    "url": "u", "applies_to": ["*"]}
            return [Document(page_content="chunk a", metadata={**meta, "page": 21, "chunk": 0}),
                    Document(page_content="chunk b", metadata={**meta, "page": 21, "chunk": 3})]

    monkeypatch.setattr(pdf_retriever, "get_vectorstore", lambda: FakeStore())
    monkeypatch.setattr(pdf_retriever, "_pages", {
        "siemens_swt_2_3_82_vs": {"4": "IEC Class IA", "6": "Diameter 82.4 m"},
        "rvo_hkz_1_2_psd": {"21": "whole page 21 with 9.44 m/s"},
    })
    pages = pdf_retriever.search_pages("hill_of_towie", "mean wind speed?")
    assert [(p["doc_id"], p["page"]) for p in pages] == [
        ("siemens_swt_2_3_82_vs", 4), ("siemens_swt_2_3_82_vs", 6), ("rvo_hkz_1_2_psd", 21)]  # page 21 once
    assert pages[-1]["text"] == "whole page 21 with 9.44 m/s"
    # Kelmarsh never gets the Siemens brochure
    assert all(p["doc_id"] != "siemens_swt_2_3_82_vs" for p in pdf_retriever.search_pages("kelmarsh", "q"))


def test_farm_facts_payload():
    kel = farm_facts("kelmarsh")
    assert kel["turbines"]["commercial_operations_date"] == ["2016-04-15"]
    assert set(kel["turbines"]["hub_heights_m"]) == {"68.5", "78.5"}
    rec = kel["production_records"]
    assert rec["max_10min_farm_output_mw"] <= rec["installed_capacity_mw"] * 1.02
    assert "No decommissioning" in kel["decommissioning"]


def test_grader_needs_a_right_tool_every_fact_and_no_forbidden_claim():
    q15 = next(q for q in QUESTIONS if q["id"] == "q15")
    good = "HillOfTowie_18's peak Cp of 0.5937 is above the Betz limit of 0.5926: an anemometer artifact."
    assert grade(q15, good, ["turbine-ranking"])["pass"]
    assert not grade(q15, good, ["maintenance-docs"])["pass"]
    contradiction = good + " It does not exceed the Betz limit."
    assert grade(q15, contradiction, ["turbine-ranking"])["forbidden_found"]
    assert len({q["id"] for q in QUESTIONS}) == len(QUESTIONS) == 35


def test_rag_eval_endpoint():
    from fastapi.testclient import TestClient

    from api.main import app

    body = TestClient(app).get("/rag-eval").json()
    assert len(body["questions"]) >= 15 and body["answer_model"]["id"] == "eu.amazon.nova-lite-v1:0"


def test_current_analysis_states_error_and_direction_plainly():
    analysis = {"comparison_summary": {"mean_actual_mw": 5.65423, "mean_predicted_mw": 5.65332, "mean_abs_pct_error": 0.5044}}
    out = qa_agent._run_tool("get_current_analysis", {}, "penmanshiel", analysis, [], [])
    cmp = out["comparison_summary"]
    assert cmp["mean_abs_hourly_error_percent"] == 50.4 and "mean_abs_pct_error" not in cmp
    assert cmp["mean_actual_vs_predicted"].startswith("actual above") and cmp["mean_actual_mw"] == 5.654


def test_operational_summary_says_when_no_aep_exists_and_names_extremes():
    from operational_assessment.payload import compact_summary, farm_operational

    hot = compact_summary(farm_operational("hill_of_towie"))["aep"]
    assert "NOT COMPUTED" in hot["long_term_aep_p50_p90"]
    pen = compact_summary(farm_operational("penmanshiel"))
    assert pen["wake_losses"]["largest_turbine_wake_loss_lt"]["turbine_id"] == "Penmanshiel_10"
    assert compact_summary(farm_operational("kelmarsh"))["aep"]["largest_uncertainty_component"]["component"] == "regression"


def test_unknown_turbine_and_tender_scope_are_in_the_system_prompt():
    prompt = qa_agent._SYSTEM_PROMPT.format(farm_id="kelmarsh", turbines=", ".join(qa_agent.FARMS["kelmarsh"].turbine_ids))
    assert "Kelmarsh_6" in prompt and "Kelmarsh_7" not in prompt and "doesn't exist" in prompt
    assert "Borssele" in prompt
    assert qa_agent._clean("18.8% loss. (document title, p.N)") == "18.8% loss."
