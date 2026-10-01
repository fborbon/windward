"""SCADA quality control applied before power curves are fit: a lean port of three of OpenOA's
`openoa.utils.filters` functions (NREL/NLR OpenOA, BSD-3-Clause, Perr-Sauer et al. 2021,
doi:10.21105/joss.02171), with the thresholds OpenOA's own example notebooks use.

Ported rather than imported: OpenOA 3.2 pins scikit-learn<1.7 and (via pygam) scipy<1.17, and
the live service runs newer versions that its registered MLflow models were trained under. The
full OpenOA analyses run offline instead (operational_assessment/run_openoa.py).

Flags are computed per turbine on the hourly series diagnose_node already has:
- `frozen`: OpenOA `unresponsive_flag`, the wind speed reading doesn't change for 3+ hours
  (a stuck anemometer or a gap-filled block).
- `stopped`: OpenOA's window-range idea (power < 2% of rated while wind >= 5 m/s): the turbine
  was down or curtailed in wind it should have produced in.
- `power_curve_outlier`: OpenOA `bin_filter`, binned by POWER (6% of rated wide, between 1% and
  90% of rated) and flagging wind speeds more than 2 std from the bin median. Binning by power
  rather than wind speed is what makes derating/curtailment show up as outliers: a curtailed
  turbine produces a power level that normally needs much less wind.

Only the power-curve-shaped outputs (binned/AMK curves, displacement fit, peak Cp) use the
cleaned data. Capacity factor, neighbor underperformance and the forecast target keep every
hour, because downtime and underperformance are exactly what those are meant to see.
"""
import numpy as np
import pandas as pd

FROZEN_MIN_REPEATS = 3
STOPPED_MAX_POWER_FRACTION = 0.02
STOPPED_MIN_WIND_MS = 5.0
BIN_WIDTH_FRACTION = 0.06
BIN_MIN_FRACTION = 0.01
BIN_MAX_FRACTION = 0.90
BIN_THRESHOLD_STD = 2.0


def unresponsive_flag(series: pd.Series, threshold: int = FROZEN_MIN_REPEATS) -> pd.Series:
    """True where the value hasn't changed for `threshold` consecutive samples (OpenOA
    filters.unresponsive_flag, same rolling-diff logic, including flagging the preceding
    samples of each frozen run)."""
    run_end = series.diff().ne(0).rolling(threshold - 1).sum() == 0
    flag = run_end.copy()
    for i in range(threshold - 1):  # shifts taken from the original run-end flags, not compounded
        flag |= run_end.shift(-1 - i, fill_value=False)
    return flag.fillna(False).astype(bool)


def bin_filter(bin_col: pd.Series, value_col: pd.Series, bin_width: float, threshold: float,
               bin_min: float, bin_max: float) -> pd.Series:
    """OpenOA filters.bin_filter with center_type='median', threshold_type='std',
    direction='all'."""
    edges = np.unique(np.clip(np.append(np.arange(bin_min, bin_max, bin_width), bin_max), bin_min, bin_max))
    which = pd.Series(np.digitize(bin_col, edges, right=True), index=bin_col.index)
    grouped = value_col.groupby(which)
    center = grouped.transform("median")
    deviation = grouped.transform(lambda v: v.std(ddof=1)) * threshold
    flag = (value_col > center + deviation) | (value_col < center - deviation)
    flag[(bin_col <= bin_min) | (bin_col > bin_max)] = False
    return flag.fillna(False).astype(bool)


def scada_qc_flags(turbine_hourly: pd.DataFrame, rated_power_kw: float) -> pd.DataFrame:
    """Adds boolean columns frozen / stopped / power_curve_outlier / qc_flag (any of them) to a
    copy of the long-format hourly frame (turbine_id, timestamp, wind_speed_ms, power_kw)."""
    parts = []
    for _, g in turbine_hourly.sort_values("timestamp").groupby("turbine_id", sort=False):
        g = g.copy()
        g["frozen"] = unresponsive_flag(g["wind_speed_ms"])
        g["stopped"] = (g["power_kw"] < STOPPED_MAX_POWER_FRACTION * rated_power_kw) & (g["wind_speed_ms"] >= STOPPED_MIN_WIND_MS)
        g["power_curve_outlier"] = bin_filter(
            g["power_kw"], g["wind_speed_ms"],
            bin_width=BIN_WIDTH_FRACTION * rated_power_kw, threshold=BIN_THRESHOLD_STD,
            bin_min=BIN_MIN_FRACTION * rated_power_kw, bin_max=BIN_MAX_FRACTION * rated_power_kw,
        )
        parts.append(g)
    out = pd.concat(parts)
    out["qc_flag"] = out["frozen"] | out["stopped"] | out["power_curve_outlier"]
    return out


def qc_summary(flagged: pd.DataFrame) -> dict:
    """Per-turbine and farm-wide share of hours each filter removed."""
    cols = ["frozen", "stopped", "power_curve_outlier", "qc_flag"]
    per_turbine = flagged.groupby("turbine_id")[cols].mean().round(4)
    return {
        "farm": {c: round(float(flagged[c].mean()), 4) for c in cols},
        "turbines": per_turbine.to_dict(orient="index"),
        "hours_total": int(len(flagged)),
        "hours_kept": int((~flagged["qc_flag"]).sum()),
        "method": "OpenOA filters (unresponsive_flag, window-range stopped check, power-binned bin_filter), ported in analysis/qc.py",
    }
