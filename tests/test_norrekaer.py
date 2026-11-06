"""Nørrekær Enge II (DTU Data NetCDF, Risø 1991-93): runs on the committed CSVs in
data/norrekaer_enge/, so it runs in CI like Delabole."""
import math

import pandas as pd

from agents.graph import _wake_source
from data_sources import norrekaer_scada as ns
from data_sources.farms import FARMS, loader_for
from operational_assessment import norrekaer_wake_study as ws
from operational_assessment.payload import farm_operational

FARM = FARMS["norrekaer_enge"]


def test_registered_with_its_own_loader_and_40_power_turbines():
    assert loader_for(FARM) is ns
    assert len(FARM.turbine_ids) == 40
    assert "Norrekaer_A1" not in FARM.turbine_ids and "Norrekaer_F6" not in FARM.turbine_ids
    assert FARM.wind_speed_source == "nacelle" and FARM.has_status_log is False


def test_report_utm_table_matches_its_own_distance_table():
    utm = pd.read_csv(ns.UTM_PATH).set_index("turbine")
    d = lambda a, b: math.dist(utm.loc[a], utm.loc[b])  # noqa: E731
    # Risø-I-684(EN) Appendix 4: M1-A1 55.0 m, M2-F6 55.0 m, A1-F1 1632.5 m, M1-F7 1953.4 m
    assert round(d("M1", "A1"), 0) == 55 and round(d("M2", "F6"), 0) == 55
    assert abs(d("A1", "F1") - 1632.5) < 2 and abs(d("M1", "F7") - 1953.4) < 2


def test_production_is_the_continuous_park_meter_scaled_to_40_turbines():
    prod = ns.load_farm_hourly_production(FARM.scada_zips)
    assert len(prod) > 12000  # the turbine channels alone cover ~5,000 windy-biased hours
    assert prod["output_mw"].min() >= 0 and prod["output_mw"].max() < 14
    assert 0.12 < prod["output_mw"].mean() / 12.0 < 0.22  # a 1990s Danish capacity factor


def test_freestream_sets_follow_the_layout():
    free = ws.freestream_sets()
    assert set(free[166]) == {"A1", "B1", "C1", "D1", "E1", "F1"}  # wind from the south: southern row ends
    assert set(free[346]) == {"A7", "B7", "C7", "D7", "E7", "F7"}  # from the north: northern ends
    assert {f"A{i}" for i in range(1, 8)} <= set(free[270])  # from the west: row A


def test_wake_study_is_in_the_operational_payload_and_labelled_as_measured():
    op = farm_operational("norrekaer_enge")
    assert op["method"] == "measured_wake_study"
    assert 3 < op["wake_losses"]["plant_lt_pct"] < 10
    assert len(op["wake_losses"]["turbines"]) == 40
    peaks = op["wake_study"]["wake_peaks_200_300_deg"]["bins_deg"]
    for report_dip in (218, 258, 293):  # Risø fig. 13
        assert min(abs(p - report_dip) for p in peaks) <= 5
    assert _wake_source(op) == "the measured wake study"
    assert _wake_source(farm_operational("kelmarsh")) == "OpenOA (long-term)"
