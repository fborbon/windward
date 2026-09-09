"""Farm registry. Every entry is a real, open, CC BY 4.0 SCADA dataset — real turbines,
real coordinates, real production — but not all of them share one export format. Kelmarsh
and Penmanshiel are both Cubico Sustainable Investments' Greenbyte export
(data_sources/greenbyte_scada.py); Hill of Towie is RES's own historian export
(data_sources/hill_of_towie_scada.py), a different file layout entirely. Farm.data_source
says which data_sources/<data_source>_scada.py module a farm's zips need — use loader_for()
rather than importing a *_scada module directly, so callers work for any registered farm."""
from dataclasses import dataclass
from pathlib import Path

from data_sources import greenbyte_scada, hill_of_towie_scada

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Farm:
    farm_id: str
    name: str
    lat: float
    lon: float
    bidding_zone_eic: str  # ENTSO-E bidding zone
    turbine_ids: list[str]
    rated_power_kw: float
    rotor_diameter_m: float
    manufacturer_model: str
    scada_zips: list[Path]  # one or more zips covering the analysis period, same turbine set
    source_doi: str
    data_source: str = "greenbyte"  # which data_sources/<data_source>_scada.py module parses scada_zips


KELMARSH_FARM = Farm(
    farm_id="kelmarsh",
    name="Kelmarsh Wind Farm",
    lat=52.400604,
    lon=-0.947133,
    bidding_zone_eic="10YGB----------A",
    turbine_ids=[f"Kelmarsh_{i}" for i in range(1, 7)],
    rated_power_kw=2050.0,
    rotor_diameter_m=92.0,
    manufacturer_model="Senvion MM92",
    scada_zips=[DATA_DIR / "kelmarsh" / "Kelmarsh_SCADA_2016.zip"],
    source_doi="10.5281/zenodo.5841834",
)

PENMANSHIEL_FARM = Farm(
    farm_id="penmanshiel",
    name="Penmanshiel Wind Farm",
    lat=55.902502,  # Penmanshiel 01
    lon=-2.306389,
    bidding_zone_eic="10YGB----------A",
    turbine_ids=[f"Penmanshiel_{i:02d}" for i in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]],
    rated_power_kw=2050.0,
    rotor_diameter_m=82.0,
    manufacturer_model="Senvion MM82",
    scada_zips=[
        DATA_DIR / "penmanshiel" / "Penmanshiel_SCADA_2016_WT01-10.zip",
        DATA_DIR / "penmanshiel" / "Penmanshiel_SCADA_2016_WT11-15.zip",
    ],
    source_doi="10.5281/zenodo.5946808",
)

HILL_OF_TOWIE_FARM = Farm(
    farm_id="hill_of_towie",
    name="Hill of Towie Wind Farm",
    lat=57.505768,  # mean of the 21 turbine coordinates (Hill_of_Towie_turbine_metadata.csv)
    lon=-3.068384,
    bidding_zone_eic="10YGB----------A",
    turbine_ids=[f"HillOfTowie_{i:02d}" for i in range(1, 22)],
    rated_power_kw=2300.0,
    rotor_diameter_m=82.0,
    manufacturer_model="Siemens SWT-2.3-VS-82",
    scada_zips=[DATA_DIR / "hill_of_towie" / "2024.zip"],
    source_doi="10.5281/zenodo.14870023",
    data_source="hill_of_towie",
)

FARMS: dict[str, Farm] = {f.farm_id: f for f in [KELMARSH_FARM, PENMANSHIEL_FARM, HILL_OF_TOWIE_FARM]}

# Kept for older imports / synthetic-data code paths.
DEMO_FARM = KELMARSH_FARM

_LOADER_MODULES = {"greenbyte": greenbyte_scada, "hill_of_towie": hill_of_towie_scada}


def loader_for(farm: Farm):
    """Returns the data_sources.<data_source>_scada module registered for this farm, so
    callers (forecasting, dashboard, RAG) don't need to hardcode which SCADA export format
    a given farm uses."""
    return _LOADER_MODULES[farm.data_source]
