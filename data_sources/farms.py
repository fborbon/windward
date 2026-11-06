"""Farm registry. Every entry is a real, open SCADA dataset — real turbines,
real coordinates, real production — but not all of them share one export format. Kelmarsh
and Penmanshiel are both Cubico Sustainable Investments' Greenbyte export
(data_sources/greenbyte_scada.py); Hill of Towie is RES's own historian export
(data_sources/hill_of_towie_scada.py), a different file layout entirely; Delabole is DTU Data's
NetCDF of a 1993-94 ETSU campaign (data_sources/delabole_scada.py); Nørrekær Enge II is a Risø
1991-93 wake campaign, DTU NetCDF too (data_sources/norrekaer_scada.py); Pedra do Sal and Beberibe
are a Brazilian university campaign's NetCDFs (data_sources/brazil_coastal_scada.py); SMARTEOLE is
a French wake-steering experiment's 1-minute CSVs (data_sources/smarteole_scada.py). Farm.data_source
says which data_sources/<data_source>_scada.py module a farm's zips need — use loader_for()
rather than importing a *_scada module directly, so callers work for any registered farm."""
from dataclasses import dataclass
from pathlib import Path

from data_sources import (brazil_coastal_scada, delabole_scada, greenbyte_scada, hill_of_towie_scada, norrekaer_scada,
                          smarteole_scada)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Farm:
    farm_id: str
    name: str
    lat: float
    lon: float
    bidding_zone_eic: str  # ENTSO-E bidding zone; "" outside the ENTSO-E area (Brazil)
    turbine_ids: list[str]
    rated_power_kw: float
    rotor_diameter_m: float
    manufacturer_model: str
    scada_zips: list[Path]  # one or more zips covering the analysis period, same turbine set
    source_doi: str
    data_source: str = "greenbyte"  # which data_sources/<data_source>_scada.py module parses scada_zips
    has_status_log: bool = True  # False: the export has no alarm/status log at all (Delabole)
    wind_speed_source: str = "nacelle"  # "met_mast": one mast cup stands in for every turbine (Delabole)


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
    # The manufacturer's datasheet says 82.4 m (rag/datasheet_specs.py, README §16.3); the dataset's
    # metadata CSV rounds it to the model name's 82. Swept area goes with the diameter squared, so
    # 82.0 overstated every Hill of Towie Cp by about 1%.
    rotor_diameter_m=82.4,
    manufacturer_model="Siemens SWT-2.3-VS-82",
    scada_zips=[DATA_DIR / "hill_of_towie" / "2024.zip"],
    source_doi="10.5281/zenodo.14870023",
    data_source="hill_of_towie",
)

DELABOLE_FARM = Farm(
    farm_id="delabole",
    name="Delabole Wind Farm",
    lat=50.630492,  # met mast 1, the origin of the site layout (DTU site documentation)
    lon=-4.708475,
    bidding_zone_eic="10YGB----------A",
    turbine_ids=[delabole_scada.turbine_id(i) for i in range(1, 11)],
    rated_power_kw=400.0,
    rotor_diameter_m=38.8,
    manufacturer_model="Winddane WD34",
    # Not a zip: the NetCDF converted once to a committed CSV (see data_sources/delabole_scada.py).
    # 1993-05 to 1994-04; no nacelle anemometers (mast 1 at 33 m for every turbine), no status log.
    scada_zips=[delabole_scada.CSV_PATH],
    source_doi="10.11583/DTU.14077004",
    data_source="delabole",
    has_status_log=False,
    wind_speed_source="met_mast",
)

NORREKAER_FARM = Farm(
    farm_id="norrekaer_enge",
    name="Nørrekær Enge II Wind Farm",
    lat=57.010,  # centre of the layout (report Appendix 5 UTM, see data_sources/norrekaer_scada.py)
    lon=9.421,
    bidding_zone_eic="10YDK-1--------W",  # DK1, West Denmark
    # 40 of 42: A1 and F6 log nacelle wind and yaw but no power in this dataset
    turbine_ids=[norrekaer_scada.turbine_id(t) for t in norrekaer_scada.POWER_TURBINES],
    rated_power_kw=300.0,
    rotor_diameter_m=28.0,
    manufacturer_model="Nordtank NTK 300F",
    scada_zips=[norrekaer_scada.CSV_PATH],
    source_doi="10.11583/DTU.19076756",
    data_source="norrekaer",
    has_status_log=False,
)

PEDRA_DO_SAL_FARM = Farm(
    farm_id="pedra_do_sal",
    name="Pedra do Sal Wind Farm",
    lat=-2.8282,  # centre of the 20 OpenStreetMap turbine positions (data/brazil_coastal)
    lon=-41.7091,
    bidding_zone_eic="",  # Brazil: no ENTSO-E zone
    turbine_ids=[brazil_coastal_scada.turbine_id("pedra_do_sal", i) for i in range(1, 21)],
    rated_power_kw=900.0,
    rotor_diameter_m=44.0,
    manufacturer_model="Enercon E-44",
    scada_zips=[brazil_coastal_scada.csv_path("pedra_do_sal")],
    source_doi="10.5281/zenodo.1475197",
    data_source="brazil_coastal",
    has_status_log=False,
)

BEBERIBE_FARM = Farm(
    farm_id="beberibe",
    name="Beberibe Wind Farm",
    lat=-4.1946,  # centre of the 32 OpenStreetMap turbine positions
    lon=-38.0779,
    bidding_zone_eic="",
    turbine_ids=[brazil_coastal_scada.turbine_id("beberibe", i) for i in range(1, 33)],
    rated_power_kw=800.0,
    rotor_diameter_m=48.0,
    manufacturer_model="Enercon E-48",
    scada_zips=[brazil_coastal_scada.csv_path("beberibe")],
    source_doi="10.5281/zenodo.1475197",
    data_source="brazil_coastal",
    has_status_log=False,
)

SMARTEOLE_FARM = Farm(
    farm_id="smarteole",
    name="Sole du Moulin Vieux (SMARTEOLE)",
    lat=49.8406,  # mean of the 7 turbine coordinates in the dataset
    lon=2.8008,
    bidding_zone_eic="10YFR-RTE------C",  # France
    turbine_ids=[smarteole_scada.turbine_id(i) for i in range(1, 8)],
    rated_power_kw=2050.0,
    rotor_diameter_m=82.0,
    manufacturer_model="Senvion MM82",
    scada_zips=[smarteole_scada.CSV_PATH],  # 2020-02-17 to 2020-05-24, stopped minutes removed by the owners
    source_doi="10.5281/zenodo.7342466",
    data_source="smarteole",
    has_status_log=False,
)

FARMS: dict[str, Farm] = {f.farm_id: f for f in [KELMARSH_FARM, PENMANSHIEL_FARM, HILL_OF_TOWIE_FARM, DELABOLE_FARM, NORREKAER_FARM,
                                                 PEDRA_DO_SAL_FARM, BEBERIBE_FARM, SMARTEOLE_FARM]}

# Kept for older imports / synthetic-data code paths.
DEMO_FARM = KELMARSH_FARM

_LOADER_MODULES = {"greenbyte": greenbyte_scada, "hill_of_towie": hill_of_towie_scada, "delabole": delabole_scada,
                   "norrekaer": norrekaer_scada, "brazil_coastal": brazil_coastal_scada, "smarteole": smarteole_scada}


def loader_for(farm: Farm):
    """Returns the data_sources.<data_source>_scada module registered for this farm, so
    callers (forecasting, dashboard, RAG) don't need to hardcode which SCADA export format
    a given farm uses."""
    return _LOADER_MODULES[farm.data_source]
