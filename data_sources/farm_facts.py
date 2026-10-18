"""Farm facts for the /ask agent (README §17): what each dataset's own metadata says about the
farm (commissioning date, turbine model, hub height per turbine) and its production records over
the full multi-year SCADA history (highest 10-minute output, best day, month and year).

Before this, "when was the farm built?" or "what's its highest production ever?" had no source:
the agent either refused or answered from the 2016 analysis window. The SCADA history lives in
data/operational_cache (the OpenOA inputs, README §15, several GB and not on the API host), so
the records are exported once to a small JSON the slim API reads:

    python -m data_sources.farm_facts   # writes dashboard/data/farm_facts_payload.json
"""
import json
from pathlib import Path

import pandas as pd

from data_sources.farms import DATA_DIR, FARMS

PAYLOAD_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "data" / "farm_facts_payload.json"
CACHE_DIR = DATA_DIR / "operational_cache"

_STATIC = {
    "kelmarsh": DATA_DIR / "kelmarsh" / "Kelmarsh_WT_static.csv",
    "penmanshiel": DATA_DIR / "penmanshiel" / "Penmanshiel_WT_static.csv",
    "hill_of_towie": DATA_DIR / "hill_of_towie" / "Hill_of_Towie_turbine_metadata.csv",
}
_AEROUP = DATA_DIR / "hill_of_towie" / "Hill_of_Towie_AeroUp_install_dates.csv"

# A day/month/year only counts toward a record when the SCADA covers it almost fully, so a
# partial first month can't win "best month" and a half-logged day can't look like a calm one.
MIN_COVERAGE = 0.95


def _turbines(farm_id: str) -> dict:
    df = pd.read_csv(_STATIC[farm_id], encoding="utf-8-sig").dropna(how="all")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    name_col = "Title" if "Title" in df else "Turbine Name"
    cod = pd.to_datetime(df["Commercial Operations Date"], dayfirst=farm_id != "hill_of_towie")
    hub = df.groupby("Hub Height (m)")[name_col].apply(list)
    return {
        "source": f"{_STATIC[farm_id].name} (the dataset's turbine metadata)",
        "count": int(len(df)),
        "manufacturer": df["Manufacturer"].iloc[0],
        "model": df["Model"].iloc[0],
        "rated_power_kw": float(df["Rated power (kW)"].iloc[0]),
        "rotor_diameter_m_in_metadata": float(df["Rotor Diameter (m)"].iloc[0]),
        "hub_heights_m": {f"{h:g}": names for h, names in hub.items()},
        "commercial_operations_date": sorted({d.date().isoformat() for d in cod}),
    }


def _records(farm_id: str) -> dict:
    scada = pd.read_pickle(CACHE_DIR / f"{farm_id}_scada.pkl")
    if isinstance(scada, tuple):
        scada = scada[0]
    n_turbines = len(FARMS[farm_id].turbine_ids)
    by_time = scada.groupby("time")["WTUR_W"].agg(["sum", "count"])
    by_time["mwh"] = by_time["sum"] / 1000 / 6

    peak_t = by_time["sum"].idxmax()
    peak_row = scada.loc[scada["WTUR_W"].idxmax()]

    def best(freq: str, label_fmt: str) -> dict | None:
        g = by_time.groupby(pd.Grouper(freq=freq))
        energy, samples = g["mwh"].sum(), g["count"].sum()
        # turbine-samples a fully logged period would have (days, months: calendar length)
        steps = pd.Series([(p + pd.tseries.frequencies.to_offset(freq) - p) / pd.Timedelta("10min") for p in energy.index],
                          index=energy.index)
        full = energy[samples >= MIN_COVERAGE * steps * n_turbines]
        if full.empty:
            return None
        return {"period": full.idxmax().strftime(label_fmt), "energy_mwh": round(float(full.max()), 1)}

    years = by_time.groupby(by_time.index.year)
    annual = {}
    for year, g in years:
        expected = len(pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="10min", inclusive="left")) * n_turbines
        coverage = g["count"].sum() / expected
        annual[str(year)] = {"energy_gwh": round(float(g["mwh"].sum() / 1000), 2), "scada_coverage": round(float(coverage), 3)}

    return {
        "source": "10-minute turbine SCADA (the OpenOA input, README §15), farm output = sum of turbine active power",
        "period": f"{scada['time'].min():%Y-%m-%d} to {scada['time'].max():%Y-%m-%d}",
        "max_10min_farm_output_mw": round(float(by_time.loc[peak_t, "sum"]) / 1000, 2),
        "max_10min_farm_output_at": f"{peak_t:%Y-%m-%d %H:%M}",
        "turbines_reporting_then": int(by_time.loc[peak_t, "count"]),
        "installed_capacity_mw": round(FARMS[farm_id].rated_power_kw * n_turbines / 1000, 2),
        "max_10min_single_turbine_kw": round(float(peak_row["WTUR_W"]), 1),
        "max_10min_single_turbine": f"{peak_row['asset_id']} at {peak_row['time']:%Y-%m-%d %H:%M}",
        "best_day": best("D", "%Y-%m-%d"),
        "best_month": best("MS", "%Y-%m"),
        "energy_by_calendar_year": annual,
        "note": "energy is from turbine SCADA, before the few-% electrical losses to the export meter; a year "
                "with scada_coverage well below 1 is partial (data gaps or the export starting/ending mid-year)",
    }


def build() -> dict:
    farms = {}
    for farm_id in FARMS:
        facts = {"turbines": _turbines(farm_id), "production_records": _records(farm_id),
                 "decommissioning": "No decommissioning or end-of-consent date appears in any source Windward holds "
                                    "for this farm (dataset metadata, SCADA, the indexed PDFs)."}
        if farm_id == "hill_of_towie" and _AEROUP.exists():
            aero = pd.read_csv(_AEROUP)
            facts["upgrades"] = [{
                "name": "AeroUp (blade aerodynamic upgrade)",
                "installed": f"{aero.iloc[:, 1].min()} to {aero.iloc[:, 2].max()}",
                "turbines": int(len(aero)),
                "source": _AEROUP.name,
            }]
        farms[farm_id] = facts
    return {"farms": farms, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}


_cache: dict = {}


def farm_facts(farm_id: str) -> dict | None:
    if "data" not in _cache:
        _cache["data"] = json.loads(PAYLOAD_PATH.read_text()) if PAYLOAD_PATH.exists() else {"farms": {}}
    return _cache["data"]["farms"].get(farm_id)


if __name__ == "__main__":
    PAYLOAD_PATH.write_text(json.dumps(build(), indent=1))
    print(PAYLOAD_PATH.read_text())
