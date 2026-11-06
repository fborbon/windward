"""Sensor data-quality checks, run before any sensor is trusted by the per-sensor normal-behaviour
monitor (analysis/care_sensor_watch.py). A residual alarm on a frozen or noisy sensor is a sensor
problem, not a machine problem, so every reading first goes through five checks, each tuned on
the sensor's own reference period (fit rows) rather than on fixed limits:

- gap: no reading (NaN).
- stale: the same value repeated for longer than the sensor ever normally repeats it: at least
  an hour (6 rows at 10 minutes) and at least 3 times the reference period's 99.9th percentile
  of repeat length, since coarse sensors (whole degrees) repeat often. Values that are common in
  the reference period (more than 1% of rows, e.g. 0 A while the turbine idles) are exempt,
  because a stopped machine legitimately repeats them.
- out of range: a temperature outside -50 to 250 °C, or any value further than twice the
  reference spread (0.1st to 99.9th percentile) beyond that spread. A failing component may run
  hotter than ever before, so this envelope is deliberately wide: it catches impossible readings,
  not unusual ones.
- step: a change between consecutive readings more than 3 times the largest change seen in the
  reference period (its 99.9th percentile). Both rows around the jump are flagged, so a one-row
  spike is caught on its way up and down.
- erratic: an hour (6 rows) whose row-to-row changes scatter 4 times more than the reference
  period's 99th percentile of that scatter: a noisy or loose sensor.

Flagged readings become missing for the monitor: they are not used as model inputs and cannot
raise an alarm. A sensor with more than 20% of its monitored readings flagged is reported as
suspect and not monitored at all.
"""
import numpy as np
import pandas as pd

CHECKS = ("gap", "stale", "out_of_range", "step", "erratic")
STALE_ROWS = 6
STALE_FACTOR = 3.0
COMMON_VALUE_SHARE = 0.01
TEMPERATURE_UNITS = {"°C", "Celsius"}
TEMPERATURE_LIMITS = (-50.0, 250.0)
ENVELOPE_SPREADS = 2.0
STEP_FACTOR = 3.0
ERRATIC_WINDOW = 6
ERRATIC_FACTOR = 4.0
SUSPECT_SHARE = 0.2


def _long_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Rows that belong to runs of True at least min_len long."""
    m = np.asarray(mask, bool)
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (ends - starts) >= min_len
    delta = np.zeros(len(m) + 1, np.int32)
    np.add.at(delta, starts[keep], 1)
    np.add.at(delta, ends[keep], -1)
    return np.cumsum(delta[:-1]) > 0


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    d = np.diff(np.concatenate([[0], np.asarray(mask, bool).astype(np.int8), [0]]))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def check_sensor(x: np.ndarray, ref: np.ndarray, unit: str = "") -> dict[str, np.ndarray]:
    """x: the sensor's full series; ref: boolean mask of its reference rows. Returns one boolean
    array per check, True where the reading fails it."""
    x = np.asarray(x, dtype="float64")
    n = len(x)
    out = {k: np.zeros(n, bool) for k in CHECKS}
    out["gap"] = np.isnan(x)
    xr = x[ref & ~out["gap"]]
    if len(xr) < 100:
        return out

    # stale: an exact repeat longer than the sensor's normal repeats (in zero differences)
    same = np.zeros(n, bool)
    same[1:] = (np.diff(x) == 0)
    ref_runs = _run_lengths(same & ref)
    normal_run = float(np.quantile(ref_runs, 0.999)) if len(ref_runs) else 0.0
    run = _long_runs(same, max(STALE_ROWS - 1, int(np.ceil(STALE_FACTOR * normal_run))))
    stale = run | np.concatenate([run[1:], [False]])  # include the first row of each run
    vals, counts = np.unique(xr, return_counts=True)
    common = vals[counts > COMMON_VALUE_SHARE * len(xr)]
    out["stale"] = stale & ~np.isin(x, common)

    lo, hi = np.quantile(xr, [0.001, 0.999])
    spread = max(hi - lo, 1e-6)
    oor = (x < lo - ENVELOPE_SPREADS * spread) | (x > hi + ENVELOPE_SPREADS * spread)
    if unit in TEMPERATURE_UNITS:
        oor |= (x < TEMPERATURE_LIMITS[0]) | (x > TEMPERATURE_LIMITS[1])
    out["out_of_range"] = oor & ~out["gap"]

    dx = np.abs(np.diff(x, prepend=np.nan))
    ref_dx = dx[ref & ~np.isnan(dx)]
    if len(ref_dx) > 100:
        big = dx > STEP_FACTOR * max(float(np.quantile(ref_dx, 0.999)), 1e-6)
        out["step"] = big | np.concatenate([big[1:], [False]])

    scatter = pd.Series(np.diff(x, prepend=np.nan)).rolling(ERRATIC_WINDOW, min_periods=ERRATIC_WINDOW).std().to_numpy()
    ref_sc = scatter[ref & ~np.isnan(scatter)]
    if len(ref_sc) > 100:
        out["erratic"] = scatter > ERRATIC_FACTOR * max(float(np.quantile(ref_sc, 0.99)), 1e-6)
    return out


def bad_mask(checks: dict[str, np.ndarray]) -> np.ndarray:
    return np.logical_or.reduce([checks[k] for k in CHECKS])
