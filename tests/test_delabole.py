"""Delabole (DTU Data NetCDF, 1993-94): runs on the committed data/delabole/delabole_10min.csv.gz,
so unlike the zip-backed farms these tests run in CI too."""
import pandas as pd
import pytest

from agents import turbine_tools
from data_sources import delabole_scada
from data_sources.farms import FARMS, loader_for
from forecasting.features import TARGET_COLUMN

FARM = FARMS["delabole"]


def test_registered_with_its_own_loader():
    assert loader_for(FARM) is delabole_scada
    assert FARM.turbine_ids == [f"Delabole_{i:02d}" for i in range(1, 11)]
    assert FARM.has_status_log is False and FARM.wind_speed_source == "met_mast"


def test_hourly_production_covers_the_campaign_and_respects_capacity():
    prod = delabole_scada.load_farm_hourly_production(FARM.scada_zips)
    assert list(prod.columns) == ["output_mw"]
    assert prod.index.min() >= pd.Timestamp("1993-05-01") and prod.index.max() < pd.Timestamp("1994-05-01")
    assert len(prod) > 5000
    # stall turbines overshoot their 400 kW rating in gusts, but an hourly farm mean stays near 4 MW
    assert 0 <= prod["output_mw"].min() and prod["output_mw"].max() < 4.6


def test_turbine_series_uses_the_mast_for_every_turbine():
    hourly = delabole_scada.load_turbine_hourly_series(FARM.scada_zips)
    assert set(hourly["turbine_id"]) == set(FARM.turbine_ids)
    # one mast reading per 10 minutes for all ten turbines (hourly means then differ slightly,
    # since each turbine's hour averages only the samples it reported power in)
    ten_min = delabole_scada._long_10min(FARM.scada_zips).dropna(subset=["wind_speed_ms"])
    assert (ten_min.groupby("timestamp")["wind_speed_ms"].nunique() == 1).all()


def test_static_has_coordinates_for_neighbor_ranking():
    static = delabole_scada.load_turbine_static("delabole")
    assert list(static.index) == FARM.turbine_ids
    assert static["Latitude"].between(50.62, 50.64).all() and static["Longitude"].between(-4.72, -4.70).all()


def test_no_status_log_is_reported_as_unknown_not_zero():
    assert delabole_scada.load_status_events(FARM.scada_zips).empty
    stats = turbine_tools._maintenance_stats("delabole", "Delabole_03")
    assert "no alarm or status log" in stats["note"]
    ranking = turbine_tools.rank_turbines("delabole", "forced_outage_hours", {"efficiency_summary": []})
    assert ranking["ranking"] == [] and "no alarm or status log" in ranking["note"]


def test_reanalysis_miss_filter_drops_only_calm_full_output_hours():
    pytest.importorskip("mlflow")  # forecasting.train imports it; installed in CI (requirements.txt)
    from forecasting.train import drop_reanalysis_misses

    df = pd.DataFrame({"wind_speed_ms": [0.4, 0.4, 9.0], TARGET_COLUMN: [3.7, 0.0, 3.7]})
    assert drop_reanalysis_misses("delabole", df).index.tolist() == [1, 2]
