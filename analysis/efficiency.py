"""Wind-resource-extraction efficiency analytics: IEC-style binned power curves and the
power coefficient Cp = P_actual / P_wind_available, benchmarked against the Betz limit.

Methodology follows standard wind-industry practice (IEC 61400-12-1 power-curve binning,
0.5 m/s bins; least-squares power-curve-displacement fitting; KD-tree neighbor-turbine
underperformance detection) — the same class of analysis behind real offshore SCADA
power-curve correction work.
"""
import numpy as np
import pandas as pd

AIR_DENSITY_KG_M3 = 1.225  # sea-level standard — fallback when real temperature/pressure aren't available
DRY_AIR_GAS_CONSTANT_J_KGK = 287.05  # specific gas constant for dry air
BETZ_LIMIT = 16 / 27  # ~0.593, theoretical max fraction of wind power a turbine can extract


def air_density_kg_m3(temperature_c, pressure_hpa):
    """Real air density via the ideal gas law (rho = P / R*T), in place of the sea-level
    constant — Cp/Betz-limit physics is density-dependent, and a farm's actual conditions
    (e.g. a cold UK winter) can meaningfully differ from 1.225 kg/m3. Accepts scalars or
    pandas Series."""
    temperature_k = temperature_c + 273.15
    pressure_pa = pressure_hpa * 100.0
    return pressure_pa / (DRY_AIR_GAS_CONSTANT_J_KGK * temperature_k)


def wind_power_available_kw(wind_speed_ms: pd.Series, rotor_diameter_m: float, air_density=AIR_DENSITY_KG_M3) -> pd.Series:
    """Kinetic power in the wind swept by the rotor: P = 0.5 * rho * A * v^3."""
    area_m2 = np.pi * (rotor_diameter_m / 2) ** 2
    return 0.5 * air_density * area_m2 * wind_speed_ms**3 / 1000.0


def power_coefficient(power_kw: pd.Series, wind_speed_ms: pd.Series, rotor_diameter_m: float, air_density=AIR_DENSITY_KG_M3) -> pd.Series:
    """Cp = actual power / available wind power. NaN below ~3 m/s where the denominator is
    negligible and Cp is numerically meaningless."""
    p_wind = wind_power_available_kw(wind_speed_ms, rotor_diameter_m, air_density)
    cp = power_kw / p_wind
    return cp.where(wind_speed_ms >= 3.0)


def binned_power_curve(df: pd.DataFrame, wind_col: str = "wind_speed_ms", power_col: str = "power_kw") -> pd.DataFrame:
    """IEC 61400-12-1 style: 0.5 m/s bins from 0-20 m/s (coarser above), mean power per bin."""
    bins = np.concatenate([np.arange(0, 20, 0.5), np.arange(20, 30, 2)])
    labels = bins[:-1] + np.diff(bins) / 2
    bin_idx = pd.cut(df[wind_col], bins=bins, labels=labels, include_lowest=True)
    curve = df.groupby(bin_idx, observed=True)[power_col].agg(["mean", "count"])
    curve.index = curve.index.astype(float).rename("wind_speed_bin")
    return curve.rename(columns={"mean": "mean_power_kw", "count": "sample_count"})


def smooth_power_curve(
    df: pd.DataFrame, test_points, wind_col: str = "wind_speed_ms", power_col: str = "power_kw"
) -> pd.Series:
    """A continuous, non-parametric power curve via kernel regression — AMK (Additive
    Multiplicative Kernel), Lee et al. 2015, from Yu Ding's *Data Science for Wind Energy* and
    its companion `dswe` package (MIT license) — as a smoother alternative to the discrete IEC
    binning above (binned_power_curve() has 0.5 m/s buckets with a handful of samples each in
    the tails; AMK borrows strength across nearby wind speeds instead). Evaluated at the same
    wind-speed points passed in (typically a binned_power_curve()'s bin centers) so the two
    curves overlay directly on one chart. Two of this package's other methods, ComparePCurve
    and FunGP, were evaluated and NOT used — real, reproducible bugs against current numpy/
    scipy, confirmed by running them against real Kelmarsh data (see README roadmap); AMK hit
    neither."""
    from dswe import AMK

    clean = df[[wind_col, power_col]].dropna()
    test_x = np.asarray(test_points, dtype=float).reshape(-1, 1)
    # A fixed 0.5 m/s bandwidth (matching binned_power_curve's own bin width) instead of AMK's
    # data-adaptive "dpi" plug-in estimator: dpi's bandwidth search has real, data-dependent
    # cost — one specific real turbine (Kelmarsh_1) took 5.7s under "dpi" vs <50ms for every
    # other turbine across all three farms, confirmed reproducible in diagnose_node's per-
    # turbine loop. A fixed bandwidth sidesteps that entirely (uniformly <50ms per turbine,
    # verified against all three farms) for a curve visually indistinguishable from "dpi"'s.
    model = AMK(X_train=clean[[wind_col]].values, y_train=clean[power_col].values, X_test=test_x, bw=[0.5], fixed_cov=[0])
    return pd.Series(model.predictions, index=np.asarray(test_points, dtype=float), name="mean_power_kw_smooth")


def fit_power_curve_displacement(reference_curve: pd.DataFrame, wind_speed_ms: pd.Series, power_kw: pd.Series) -> float:
    """Least-squares horizontal (wind-speed axis) displacement between a turbine's observed
    power curve and a reference curve (the farm-mean binned curve) — quantifies a systematic
    anemometer calibration bias instead of just flagging that Cp exceeds the Betz limit. A
    turbine reading, say, +0.4 m/s displaced from its peers under the same wind is consistent
    with a real nacelle-anemometer bias (it sits downstream of the spinning rotor), not
    over-unity energy extraction. Mirrors real power-curve-correction practice: fit against a
    reference curve (here the farm's own pooled empirical curve, since no manufacturer power
    curve CSV exists for these open datasets) rather than a raw scatter comparison."""
    from scipy.interpolate import interp1d
    from scipy.optimize import leastsq

    ref = reference_curve.dropna()
    curve_fn = interp1d(
        ref.index.values, ref["mean_power_kw"].values, bounds_error=False,
        fill_value=(ref["mean_power_kw"].iloc[0], ref["mean_power_kw"].iloc[-1]),
    )
    mask = power_kw > 0
    uh = wind_speed_ms[mask].values
    pw = power_kw[mask].values
    uh_min, uh_max = ref.index.min(), ref.index.max()

    def residual(delta):
        shifted = np.clip(uh + delta[0], uh_min, uh_max)
        return curve_fn(shifted) - pw

    displacement = leastsq(residual, x0=[0.0])[0][0]
    return float(displacement)


def neighbor_underperformance(
    turbine_hourly: pd.DataFrame, turbine_coords: pd.DataFrame, k: int = 5, std_factor: float = 2.698,
    min_neighbor_power_kw: float = 50.0,
) -> dict:
    """Real per-turbine, per-hour anomaly detection: for each turbine, compare its power
    against the median of its k nearest spatial neighbors at that same hour — flags hours
    where a turbine produced meaningfully less than nearby turbines that were themselves
    producing (so it's a real underperformance signal, not just calm wind). std_factor=2.698
    matches conventional median +/- factor*std outlier practice. Mirrors real SCADA
    power-correction work (KD-tree neighbor comparison, robust median/std thresholding),
    adapted from farm-level aggregate deviation (the only anomaly signal diagnose_node had
    before) to genuine per-turbine spatial comparison.

    turbine_coords must be indexed by turbine_id with 'Latitude'/'Longitude' columns (see
    data_sources.*_scada.load_turbine_static) — lat/lon distance is an adequate nearest-
    neighbor *ranking* at single-farm scale (a few km across), even though it isn't a true
    metric distance.

    Returns {turbine_id: flagged_hour_count}.
    """
    from scipy.spatial import cKDTree

    pivot = turbine_hourly.pivot_table(index="timestamp", columns="turbine_id", values="power_kw")
    turbine_ids = list(pivot.columns)
    coords = turbine_coords.loc[turbine_ids, ["Latitude", "Longitude"]].values
    tree = cKDTree(coords)

    flagged = {}
    for i, turbine_id in enumerate(turbine_ids):
        _, neighbor_idx = tree.query(coords[i], k=min(k + 1, len(coords)))
        neighbor_ids = [turbine_ids[j] for j in np.atleast_1d(neighbor_idx) if turbine_ids[j] != turbine_id][:k]
        if not neighbor_ids:
            flagged[turbine_id] = 0
            continue

        neighbor_power = pivot[neighbor_ids]
        median = neighbor_power.median(axis=1)
        std = neighbor_power.std(axis=1)
        this = pivot[turbine_id]

        low_threshold = median - std_factor * std
        is_low = (this < low_threshold) & (median > min_neighbor_power_kw)
        flagged[turbine_id] = int(is_low.sum())

    return flagged


def scada_reanalysis_wind_check(turbine_hourly: pd.DataFrame, feature_frame: pd.DataFrame) -> dict:
    """Cross-references farm-mean turbine SCADA wind speed against the independent Open-Meteo
    reanalysis wind speed for the same hours — real QC practice (comparing an on-site sensor
    against an independent reference) adapted to Windward's actual two data sources, since
    there's no second on-site mast/sonic here. A low correlation would flag a data-alignment
    bug or a genuine on-site sensor issue; expected to run fairly high (reanalysis vs a real
    turbine's own anemometer, not a perfect match) but should not be near zero."""
    farm_mean_scada = turbine_hourly.groupby("timestamp")["wind_speed_ms"].mean()
    aligned = pd.concat(
        [farm_mean_scada.rename("scada"), feature_frame["wind_speed_ms"].rename("reanalysis")], axis=1
    ).dropna()
    if len(aligned) < 2:
        return {"correlation": None, "hours_compared": len(aligned)}

    correlation = aligned["scada"].corr(aligned["reanalysis"])
    diff = (aligned["scada"] - aligned["reanalysis"]).abs()
    return {
        "correlation": float(correlation),
        "hours_compared": int(len(aligned)),
        "mean_abs_diff_ms": float(diff.mean()),
        "pct_hours_diverging_gt_5ms": float((diff > 5.0).mean()),
    }


def turbine_efficiency_summary(
    turbine_hourly: pd.DataFrame, rated_power_kw: float, rotor_diameter_m: float, air_density_by_timestamp: pd.Series = None
) -> pd.DataFrame:
    """One row per turbine_id: capacity factor and peak power coefficient (aerodynamic
    efficiency), the latter benchmarked against the Betz limit and typical real-turbine Cp.
    air_density_by_timestamp (indexed by timestamp, e.g. from air_density_kg_m3() applied to
    the farm's weather data): when given, uses the farm's real per-hour air density instead
    of the sea-level constant."""
    rows = []
    for turbine_id, g in turbine_hourly.groupby("turbine_id"):
        if air_density_by_timestamp is not None:
            density = air_density_by_timestamp.reindex(g["timestamp"]).values
        else:
            density = AIR_DENSITY_KG_M3
        cp = power_coefficient(g["power_kw"], g["wind_speed_ms"], rotor_diameter_m, density)
        rows.append(
            {
                "turbine_id": turbine_id,
                "capacity_factor": g["power_kw"].mean() / rated_power_kw,
                "peak_cp": cp.quantile(0.95),  # robust "peak" — avoids single-sample outliers
                "betz_limit": BETZ_LIMIT,
                "hours_observed": len(g),
            }
        )
    return pd.DataFrame(rows).set_index("turbine_id")


def measure_correlate_predict(mast_mean_ms: float, era5_mean_ms: float, era5_series_ms: pd.Series) -> dict:
    """Measure-Correlate-Predict (MCP) — standard wind-resource-assessment practice for
    estimating a site's wind resource from a real but short/incomplete on-site measurement
    campaign: Measure (a real on-site mast reading), Correlate (a ratio against a real
    long-term reference over the SAME period), Predict (scale the reference's other values —
    here, a live forecast — by that ratio to estimate local conditions).

    This is the ratio-of-means variant, not full regression-based MCP (which needs concurrent
    timestamp-paired mast/reference samples). data_sources/dswe_scada.py's turbines have no
    per-row timestamps, only a real documented overall date range per mast — see that module's
    docstring — so a ratio over that real period, not a per-timestamp regression, is what the
    data actually supports. Returns the ratio, both real input means, and the scaled series."""
    ratio = mast_mean_ms / era5_mean_ms
    return {
        "ratio": ratio,
        "mast_mean_ms": mast_mean_ms,
        "era5_mean_ms": era5_mean_ms,
        "predicted_local_ms": (era5_series_ms * ratio).tolist(),
    }


def actual_vs_predicted(actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    """Aligned actual/predicted farm production with residual, for diagnosis/dashboard."""
    df = pd.concat([actual.rename("actual_mw"), predicted.rename("predicted_mw")], axis=1).dropna()
    df["residual_mw"] = df["actual_mw"] - df["predicted_mw"]
    df["pct_of_predicted"] = df["actual_mw"] / df["predicted_mw"].replace(0, np.nan)
    return df


COMPASS_16 = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]


def wind_rose_energy_kwh(
    df: pd.DataFrame, wind_dir_col: str = "wind_direction_deg", power_col: str = "output_mw", n_sectors: int = 16
) -> pd.DataFrame:
    """Cumulative energy (kWh) actually produced per wind-direction sector, over the whole
    period in df - real additive production accounting (sum of power_mw over 1h samples, per
    sector), not a theoretical curve. Bins are centered on each compass point (e.g. "N" covers
    -11.25..11.25 degrees for 16 sectors), matching conventional wind-rose plotting."""
    sector_width = 360.0 / n_sectors
    shifted = (df[wind_dir_col] % 360 + sector_width / 2) % 360
    sector_idx = (shifted // sector_width).astype(int).clip(0, n_sectors - 1)
    energy_mwh = df.assign(_sector=sector_idx).groupby("_sector")[power_col].sum()  # MW over 1h samples = MWh
    energy_kwh = (energy_mwh * 1000).reindex(range(n_sectors), fill_value=0.0)
    labels = COMPASS_16 if n_sectors == 16 else [f"{i * sector_width:.0f}°" for i in range(n_sectors)]
    return pd.DataFrame({
        "sector": range(n_sectors),
        "compass": labels,
        "direction_deg": [i * sector_width for i in range(n_sectors)],
        "energy_kwh": energy_kwh.values,
    })


def wind_speed_power_distribution(
    df: pd.DataFrame, rated_capacity_mw: float, wind_col: str = "wind_speed_ms",
    power_col: str = "output_mw", bin_width: float = 0.5,
) -> dict:
    """Site wind-speed distribution (with a fitted Weibull curve) paired with the farm's real
    power output (as a share of rated capacity) at each wind speed, sharing the same x-axis -
    the resource-assessment pair from demo_notebook/'s "Recurso eolico del emplazamiento"
    section, computed here for the live dashboard instead of a notebook."""
    from scipy.stats import weibull_min

    wind_speed = df[wind_col].dropna()
    shape, loc, scale = weibull_min.fit(wind_speed, floc=0)

    hist_edges = np.arange(0, wind_speed.max() + bin_width, bin_width)
    density, edges = np.histogram(wind_speed, bins=hist_edges, density=True)
    bin_centers = edges[:-1] + bin_width / 2
    weibull_pdf = weibull_min.pdf(bin_centers, shape, loc, scale)

    power_binned = df.groupby(pd.cut(df[wind_col], bins=hist_edges), observed=True)[power_col].mean()
    share_of_capacity = (power_binned / rated_capacity_mw).reindex(bin_centers, fill_value=np.nan)

    return {
        "weibull_shape": float(shape),
        "weibull_scale": float(scale),
        "wind_speed_bins": [round(float(v), 3) for v in bin_centers],
        "wind_speed_density": [round(float(v), 5) for v in density],
        "weibull_pdf": [round(float(v), 5) for v in weibull_pdf],
        "power_share_of_capacity": [None if pd.isna(v) else round(float(v), 4) for v in share_of_capacity.values],
    }
