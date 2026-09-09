"""Wind-resource-extraction efficiency analytics: IEC-style binned power curves and the
power coefficient Cp = P_actual / P_wind_available, benchmarked against the Betz limit.

Methodology follows standard wind-industry practice (IEC 61400-12-1 power-curve binning,
0.5 m/s bins) — the same class of analysis behind SCADA power-curve correction work.
"""
import numpy as np
import pandas as pd

AIR_DENSITY_KG_M3 = 1.225  # sea-level standard
BETZ_LIMIT = 16 / 27  # ~0.593, theoretical max fraction of wind power a turbine can extract


def wind_power_available_kw(wind_speed_ms: pd.Series, rotor_diameter_m: float) -> pd.Series:
    """Kinetic power in the wind swept by the rotor: P = 0.5 * rho * A * v^3."""
    area_m2 = np.pi * (rotor_diameter_m / 2) ** 2
    return 0.5 * AIR_DENSITY_KG_M3 * area_m2 * wind_speed_ms**3 / 1000.0


def power_coefficient(power_kw: pd.Series, wind_speed_ms: pd.Series, rotor_diameter_m: float) -> pd.Series:
    """Cp = actual power / available wind power. NaN below ~3 m/s where the denominator is
    negligible and Cp is numerically meaningless."""
    p_wind = wind_power_available_kw(wind_speed_ms, rotor_diameter_m)
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


def turbine_efficiency_summary(
    turbine_hourly: pd.DataFrame, rated_power_kw: float, rotor_diameter_m: float
) -> pd.DataFrame:
    """One row per turbine_id: capacity factor and peak power coefficient (aerodynamic
    efficiency), the latter benchmarked against the Betz limit and typical real-turbine Cp."""
    rows = []
    for turbine_id, g in turbine_hourly.groupby("turbine_id"):
        cp = power_coefficient(g["power_kw"], g["wind_speed_ms"], rotor_diameter_m)
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


def actual_vs_predicted(actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    """Aligned actual/predicted farm production with residual, for diagnosis/dashboard."""
    df = pd.concat([actual.rename("actual_mw"), predicted.rename("predicted_mw")], axis=1).dropna()
    df["residual_mw"] = df["actual_mw"] - df["predicted_mw"]
    df["pct_of_predicted"] = df["actual_mw"] / df["predicted_mw"].replace(0, np.nan)
    return df
