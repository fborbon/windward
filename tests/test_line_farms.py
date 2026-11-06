"""Pedra do Sal, Beberibe (Brazil, Zenodo 1475197) and SMARTEOLE (France, Zenodo 7342466): run on
the committed CSVs in data/brazil_coastal and data/smarteole, so they run in CI."""
import pandas as pd

from agents.graph import _wake_source
from data_sources import brazil_coastal_scada as bc
from data_sources import smarteole_scada as sm
from data_sources.farms import FARMS, loader_for
from operational_assessment import line_wake_study as lws
from operational_assessment.payload import farm_operational


def test_registered_with_their_loaders():
    assert loader_for(FARMS["pedra_do_sal"]) is bc and loader_for(FARMS["beberibe"]) is bc
    assert loader_for(FARMS["smarteole"]) is sm
    assert len(FARMS["pedra_do_sal"].turbine_ids) == 20 and len(FARMS["beberibe"].turbine_ids) == 32
    assert FARMS["smarteole"].turbine_ids == [f"SMV{i}" for i in range(1, 8)]
    assert not any(FARMS[f].has_status_log for f in ("pedra_do_sal", "beberibe", "smarteole"))


def test_brazil_production_is_a_full_year_within_capacity():
    for farm_id, cap_mw in (("pedra_do_sal", 18.0), ("beberibe", 25.6)):
        prod = bc.load_farm_hourly_production(FARMS[farm_id].scada_zips)
        assert prod.index.min() >= pd.Timestamp("2013-08-01") and prod.index.max() < pd.Timestamp("2014-08-01")
        assert len(prod) > 8000
        assert prod["output_mw"].min() >= 0 and prod["output_mw"].max() < 1.05 * cap_mw
        assert 0.3 < prod["output_mw"].mean() / cap_mw < 0.5  # trade-wind capacity factor


def test_brazil_layout_is_complete_and_numbering_runs_nw_to_se():
    for farm_id in ("pedra_do_sal", "beberibe"):
        static = bc.load_turbine_static(farm_id)
        assert sorted(static.index) == sorted(FARMS[farm_id].turbine_ids)
        assert static["OSM node"].is_unique
        along = static["Position along line (m, NW to SE)"]
        numbers = static.index.str[-2:].astype(int)
        assert pd.Series(numbers, index=static.index).corr(along, method="spearman") > 0.8


def test_smarteole_is_the_2020_campaign_with_the_steering_log():
    df = sm.load_10min()
    assert df["timestamp"].min() >= pd.Timestamp("2020-02-17") and df["timestamp"].max() < pd.Timestamp("2020-05-26")
    assert {"offset_deg_6", "offset_active_6"} <= set(df.columns)
    prod = sm.load_farm_hourly_production(FARMS["smarteole"].scada_zips)
    assert len(prod) > 2000 and prod["output_mw"].max() < 7 * 2.05 * 1.05


def test_freestream_sets_follow_a_north_south_line():
    free = lws.freestream_sets(sm.load_turbine_static("smarteole"), sm.ROTOR_D_M)
    assert free[0] == ["SMV1"] and free[180] == ["SMV7"]  # along the line only the end turbine is free
    assert len(free[90]) == 7  # across it nobody is waked


def test_wake_studies_are_in_the_payload_and_labelled_as_measured():
    for farm_id in ("pedra_do_sal", "beberibe", "smarteole"):
        op = farm_operational(farm_id)
        assert op["method"] == "measured_wake_study" and op["study"] == "line_farm"
        assert len(op["wake_losses"]["turbines"]) == len(FARMS[farm_id].turbine_ids)
        assert op["findings"] and _wake_source(op) == "the measured wake study"
    steer = farm_operational("smarteole")["wake_study"]["wake_steering"]
    # steering costs the yawed turbine a little and helps the one behind it
    assert steer["smv6_pct_of_ref"]["on"] < steer["smv6_pct_of_ref"]["off"]
    assert steer["smv5_pct_of_ref"]["on"] > steer["smv5_pct_of_ref"]["off"]
    lo, hi = steer["pair_gain_90ci_pct"]
    assert lo < steer["pair_gain_pct"] < hi


def test_recommendation_passes_over_the_steered_turbine():
    from agents.graph import recommend_node

    eff = pd.DataFrame({"capacity_factor": [0.30, 0.289, 0.287]}, index=["SMV1", "SMV5", "SMV6"])
    out = recommend_node({"farm_id": "smarteole", "efficiency_summary": eff, "operational": farm_operational("smarteole")})
    rec = out["recommendation"]
    assert rec["action"] == "Prioritize inspection of SMV5"
    assert "passed over" in rec["rationale"] and "on purpose" in rec["rationale"]
