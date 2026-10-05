import numpy as np
import pandas as pd

from analysis.qc import bin_filter, scada_qc_flags, unresponsive_flag
from operational_assessment.payload import compact_summary, yaw_flags, yaw_status


def test_unresponsive_flag_marks_whole_frozen_run():
    s = pd.Series([1.0, 2.0, 5.0, 5.0, 5.0, 3.0])
    assert unresponsive_flag(s, threshold=3).tolist() == [False, False, True, True, True, False]


def test_bin_filter_flags_outlier_within_power_bin():
    power = pd.Series([500.0] * 20)
    wind = pd.Series([7.0, 7.1, 6.9, 7.05, 6.95] * 4)
    wind.iloc[3] = 15.0  # same power, far more wind: a derated/curtailed hour
    flag = bin_filter(power, wind, bin_width=100, threshold=2, bin_min=0, bin_max=2000)
    assert flag.iloc[3] and flag.sum() == 1


def test_scada_qc_flags_stopped_in_wind():
    df = pd.DataFrame({
        "turbine_id": "T1",
        "timestamp": pd.date_range("2024-01-01", periods=4, freq="h"),
        "wind_speed_ms": [8.0, 8.5, 9.0, 3.0],
        "power_kw": [900.0, 5.0, 1100.0, 0.0],
    })
    out = scada_qc_flags(df, rated_power_kw=2050)
    assert out["stopped"].tolist() == [False, True, False, False]  # 3 m/s at 0 kW is just calm


def _yaw(mean, ci, spread):
    return {"mean_deg": mean, "ci95_deg": ci, "by_ws_spread_deg": spread}


def test_yaw_status_gates():
    assert yaw_status(_yaw(5.0, [3.0, 7.0], 4.0)) == "flag"
    assert yaw_status(_yaw(1.0, [-1.0, 3.0], 4.0)) == "ok"
    assert yaw_status(_yaw(15.0, [12.0, 18.0], 30.0)) == "inconclusive"  # bins disagree
    assert yaw_status(_yaw(None, [None, None], None)) == "inconclusive"


def test_yaw_flags_and_compact_summary():
    op = {"yaw_misalignment": {"turbines": {"A": _yaw(-6.0, [-8.0, -4.0], 3.0), "B": _yaw(1.0, [-1.0, 2.0], 2.0)}},
          "wake_losses": {"plant_lt_pct": 7.0, "by_direction": [{"wd_deg": 0.0}]}}
    flags = yaw_flags(op)
    assert [f["turbine_id"] for f in flags] == ["A"]
    assert np.isclose(flags[0]["approx_below_rated_loss_pct"], (1 - np.cos(np.deg2rad(6.0)) ** 2) * 100, atol=0.01)
    summary = compact_summary(op)
    assert "by_direction" not in summary["wake_losses"]
    assert summary["yaw_misalignment"]["turbines"]["A"]["status"] == "flag"


def test_resolve_turbine_id_accepts_common_spellings():
    from agents.turbine_tools import resolve_turbine_id

    assert resolve_turbine_id("kelmarsh", "turbine 3") == "Kelmarsh_3"
    assert resolve_turbine_id("kelmarsh", "Kelmarsh 3") == "Kelmarsh_3"
    assert resolve_turbine_id("penmanshiel", "T05") == "Penmanshiel_05"
    assert resolve_turbine_id("penmanshiel", "3") is None  # Penmanshiel has no turbine 03
    assert resolve_turbine_id("kelmarsh", "9") is None


def test_compact_summary_adds_aep_interval():
    op = {"aep": {"monthly_linear": {"p50_gwh": 31.687, "mean_gwh": 31.687, "std_gwh": 0.371}}}
    assert compact_summary(op)["aep"]["monthly_linear"]["interval_95pct_gwh"] == [30.96, 32.41]


def test_union_hours_counts_concurrent_alarms_once():
    import pandas as pd
    from agents.turbine_tools import _union_hours

    ev = pd.DataFrame({
        "start": pd.to_datetime(["2016-01-01 00:00", "2016-01-01 00:00", "2016-01-01 01:00", "2016-01-02 00:00"]),
        "end": pd.to_datetime(["2016-01-01 02:00", "2016-01-01 02:00", "2016-01-01 03:00", "2016-01-02 01:00"]),
    })
    assert _union_hours(ev) == 4.0  # 00:00-03:00 merged, plus one separate hour


def test_mcp_server_exposes_operational_and_turbine_tools():
    import asyncio
    from mcp_server.server import mcp

    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {"get_operational_assessment", "get_turbine_details", "rank_turbines"} <= names
