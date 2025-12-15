"""
SMPS Data Sources Package.

This package provides access to various environmental data sources
for soil moisture prediction.

Data Sources:
-------------
- iSDA Africa: High-resolution (30m) soil data for Africa
- SoilGrids: Global soil data at 250m resolution
- Google Earth Engine: Satellite data (NDVI, LAI, LST)
- Open-Meteo: Weather data (precipitation, temperature, ET₀)
- ISMN: Validation soil moisture measurements

Usage:
------
>>> from smps.data.sources import (
...     IsdaAfricaAuthenticatedSource,
...     SoilGridsGlobalSource,
...     GoogleEarthEngineSatelliteSource,
...     OpenMeteoWeatherSource,
...     ValidationDataManager
... )

>>> # Fetch soil data from iSDA
>>> isda = IsdaAfricaAuthenticatedSource()
>>> profile = isda.fetch_soil_profile("site1", latitude=-1.29, longitude=36.82)

>>> # Fetch weather data
>>> weather = OpenMeteoWeatherSource()
>>> data = weather.fetch("site1", latitude=-1.29, longitude=36.82,
...                      start_date=start, end_date=end)
"""

# Soil data sources
from smps.data.sources.isda_authenticated import (
    IsdaAfricaAuthenticatedSource,
    get_isda_soil_data,
)
from smps.data.sources.soilgrids import (
    SoilGridsGlobalSource,
    get_soilgrids_profile,
)
from smps.data.sources.soil import (
    MockSoilSource,
)

# Satellite data sources
from smps.data.sources.gee_satellite import (
    GoogleEarthEngineSatelliteSource,
    setup_gee_authentication,
)
from smps.data.sources.satellite import (
    MODISNDVISource,
)

# Weather data sources
from smps.data.sources.weather import (
    OpenMeteoSource as OpenMeteoWeatherSource,  # Alias for consistency
)

# Validation data sources
from smps.data.sources.validation_sources import (
    ISMNDataSource,
    FluxnetDataSource,
    ValidationDataManager,
    ValidationObservation,
    print_attribute_guide,
    SOIL_MOISTURE_PREDICTION_ATTRIBUTES,
)

# FLDAS soil moisture reference data
from smps.data.sources.fldas import (
    FLDASSource,
    FLDASObservation,
    load_fldas_observation,
)

# Base classes
from smps.data.sources.base import (
    DataSource,
    DataFetchRequest,
    DataFetchResult,
    SoilSource,
    WeatherSource,
    RemoteSensingSource as SatelliteSource,  # Alias for clarity
)

__all__ = [
    # Soil sources
    "IsdaAfricaAuthenticatedSource",
    "SoilGridsGlobalSource",
    "MockSoilSource",
    "get_isda_soil_data",
    "get_soilgrids_profile",

    # Satellite sources
    "GoogleEarthEngineSatelliteSource",
    "MODISNDVISource",
    "setup_gee_authentication",

    # Weather sources
    "OpenMeteoWeatherSource",

    # Validation sources
    "ISMNDataSource",
    "FluxnetDataSource",
    "ValidationDataManager",
    "ValidationObservation",

    # FLDAS reference data
    "FLDASSource",
    "FLDASObservation",
    "load_fldas_observation",

    # Utilities
    "print_attribute_guide",
    "SOIL_MOISTURE_PREDICTION_ATTRIBUTES",

    # Base classes
    "DataSource",
    "DataFetchRequest",
    "DataFetchResult",
    "SoilSource",
    "WeatherSource",
    "SatelliteSource",
]
