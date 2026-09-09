"""Farm registry. All entries so far are Greenbyte-exported SCADA (see
data_sources/greenbyte_scada.py) from Cubico Sustainable Investments, published on
Zenodo under CC BY 4.0 — real turbines, real coordinates, real production."""
from dataclasses import dataclass, field
from pathlib import Path

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
    scada_zips: list[Path]  # one or more zips covering the analysis period, same turbine set
    source_doi: str


KELMARSH_FARM = Farm(
    farm_id="kelmarsh",
    name="Kelmarsh Wind Farm",
    lat=52.400604,
    lon=-0.947133,
    bidding_zone_eic="10YGB----------A",
    turbine_ids=[f"Kelmarsh_{i}" for i in range(1, 7)],
    rated_power_kw=2050.0,
    rotor_diameter_m=92.0,
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
    scada_zips=[
        DATA_DIR / "penmanshiel" / "Penmanshiel_SCADA_2016_WT01-10.zip",
        DATA_DIR / "penmanshiel" / "Penmanshiel_SCADA_2016_WT11-15.zip",
    ],
    source_doi="10.5281/zenodo.5946808",
)

FARMS: dict[str, Farm] = {f.farm_id: f for f in [KELMARSH_FARM, PENMANSHIEL_FARM]}

# Kept for older imports / synthetic-data code paths.
DEMO_FARM = KELMARSH_FARM
