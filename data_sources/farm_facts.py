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
    "delabole": DATA_DIR / "delabole" / "Delabole_WT_static.csv",
    "norrekaer_enge": None,  # built from the report's specs and UTM table, see data_sources/norrekaer_scada.py
    "pedra_do_sal": None,  # Zenodo record specs + OpenStreetMap positions, see data_sources/brazil_coastal_scada.py
    "beberibe": None,
    "smarteole": None,  # the dataset's coordinates file + its README's turbine model
}
# The DTU site documentation gives Delabole's commissioning as "December 1991", no day; the
# static CSV carries the 1st only because the column needs a date.
_COMMISSIONING_MONTH_ONLY = {"delabole": "1991-12 (month only in the source)",
                             "norrekaer_enge": "1991 (year only: commissioning ran through summer 1991, Risø report section 5)",
                             "pedra_do_sal": "not stated in the dataset",
                             "beberibe": "not stated in the dataset",
                             "smarteole": "not stated in the dataset"}
_STATIC_SOURCE = {
    "norrekaer_enge": "Risø report Risø-I-684(EN) Appendix 1 (specs) and 5 (UTM positions); 40 of 42 turbines log power",
    "pedra_do_sal": ("Zenodo 1475197 record (20 Enercon E-44, 55 m hub); positions from OpenStreetMap, matched to the dataset's "
                     "turbine numbers by power correlation (inferred, see data_sources/brazil_coastal_scada.py)"),
    "beberibe": ("Zenodo 1475197 record (32 Enercon E-48, 75 m hub); positions from OpenStreetMap, matched to the dataset's "
                 "turbine numbers by power correlation (inferred, see data_sources/brazil_coastal_scada.py)"),
    "smarteole": "SMARTEOLE dataset coordinates file and README (7 Senvion MM82, 82 m rotor, 80 m hub)",
}
_AEROUP = DATA_DIR / "hill_of_towie" / "Hill_of_Towie_AeroUp_install_dates.csv"

# A day/month/year only counts toward a record when the SCADA covers it almost fully, so a
# partial first month can't win "best month" and a half-logged day can't look like a calm one.
MIN_COVERAGE = 0.95


def _turbines(farm_id: str) -> dict:
    if _STATIC[farm_id] is None:
        from data_sources.farms import loader_for
        df = loader_for(FARMS[farm_id]).load_turbine_static(farm_id).reset_index()
    else:
        df = pd.read_csv(_STATIC[farm_id], encoding="utf-8-sig").dropna(how="all")
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    name_col = "Title" if "Title" in df else "Turbine Name"
    cod = pd.to_datetime(df["Commercial Operations Date"], dayfirst=farm_id != "hill_of_towie")
    hub = df.groupby("Hub Height (m)")[name_col].apply(list)
    return {
        "source": (_STATIC_SOURCE[farm_id] if _STATIC[farm_id] is None else
                   f"{_STATIC[farm_id].name} (" + ("built from the DTU site documentation's turbine list and layout"
                                                   if farm_id == "delabole" else "the dataset's turbine metadata") + ")"),
        "count": int(len(df)),
        "manufacturer": df["Manufacturer"].iloc[0],
        "model": df["Model"].iloc[0],
        "rated_power_kw": float(df["Rated power (kW)"].iloc[0]),
        "rotor_diameter_m_in_metadata": float(df["Rotor Diameter (m)"].iloc[0]),
        "hub_heights_m": {f"{h:g}": names for h, names in hub.items()},
        "commercial_operations_date": [_COMMISSIONING_MONTH_ONLY[farm_id]] if farm_id in _COMMISSIONING_MONTH_ONLY
                                      else sorted({d.date().isoformat() for d in cod}),
    }


def _records(farm_id: str) -> dict:
    if FARMS[farm_id].data_source in ("delabole", "norrekaer", "smarteole"):
        # Small enough to read straight from the committed CSV; it never went through OpenOA.
        from data_sources.farms import loader_for
        scada = loader_for(FARMS[farm_id]).scada_10min_for_records()
    elif FARMS[farm_id].data_source == "brazil_coastal":
        from data_sources import brazil_coastal_scada
        scada = brazil_coastal_scada.scada_10min_for_records(farm_id)
    else:
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

    # Asked for "the year that produced the most", the agent picked 2020 over Penmanshiel's 2017
    # from the year list; partial years (export start/end) can't count, so give it ready-made.
    full_years = {y: a for y, a in annual.items() if a["scada_coverage"] >= MIN_COVERAGE}
    best_year = max(full_years, key=lambda y: full_years[y]["energy_gwh"]) if full_years else None
    return {
        "source": ("10-minute means of 1-minute turbine SCADA (SMARTEOLE, 2020; the data owners removed every minute in which "
                   "any turbine was stopped, so totals undercount and coverage reads low by design)" if FARMS[farm_id].data_source == "smarteole"
                   else "10-minute turbine SCADA (Zenodo 1475197, 2013-14 Federal University of Santa Catarina campaign)"
                   if FARMS[farm_id].data_source == "brazil_coastal"
                   else "10-minute turbine SCADA (DTU Data NetCDF, 1991-93 Risø campaign; only runs with hub wind above "
                   "~2.5 m/s were logged, so totals and coverage undercount by design)" if FARMS[farm_id].data_source == "norrekaer"
                   else "10-minute turbine SCADA (DTU Data NetCDF, 1993-94 ETSU campaign)" if FARMS[farm_id].data_source == "delabole"
                   else "10-minute turbine SCADA (the OpenOA input, README §15)") + ", farm output = sum of turbine active power",
        "period": f"{scada['time'].min():%Y-%m-%d} to {scada['time'].max():%Y-%m-%d}",
        "max_10min_farm_output_mw": round(float(by_time.loc[peak_t, "sum"]) / 1000, 2),
        "max_10min_farm_output_at": f"{peak_t:%Y-%m-%d %H:%M}",
        "turbines_reporting_then": int(by_time.loc[peak_t, "count"]),
        "installed_capacity_mw": round(FARMS[farm_id].rated_power_kw * n_turbines / 1000, 2),
        "max_10min_single_turbine_kw": round(float(peak_row["WTUR_W"]), 1),
        "max_10min_single_turbine": f"{peak_row['asset_id']} at {peak_row['time']:%Y-%m-%d %H:%M}",
        "best_day": best("D", "%Y-%m-%d"),
        "best_month": best("MS", "%Y-%m"),
        "best_full_calendar_year": {"year": best_year, **full_years[best_year]} if best_year else
                                   "no calendar year with at least 95% SCADA coverage",
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
