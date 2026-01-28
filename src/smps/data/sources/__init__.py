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

from smps.data.sources.base import (
    DataSource,
    DataFetchRequest,
    DataFetchResult,
    SoilSource,
    WeatherSource,
    RemoteSensingSource as SatelliteSource,  # Alias for clarity
)
from smps.data.sources.ismn_loader import (
    ISMNStationLoader,
    ISMNStationData,
    ISMNSensorMetadata,
    ISMNSoilProperties,
    load_ismn_station,
    get_daily_soil_moisture,
)
from smps.data.sources.validation_sources import (
    ISMNDataSource,
    FluxnetDataSource,
    ValidationDataManager,
    ValidationObservation,
    print_attribute_guide,
    SOIL_MOISTURE_PREDICTION_ATTRIBUTES,
)
from smps.data.sources.spaceiotbox import (
    SpaceIoTBoxClient,
    SpaceIoTBoxConfig,
    SpaceIoTBoxWeatherSource,
    SpaceIoTBoxCopernicusSource,
    SpaceIoTBoxAgroSource,
    SpaceIoTBoxDatasetsSource,
    SpaceIoTBoxUnifiedSource,
    get_spaceiotbox_weather,
    get_spaceiotbox_satellite,
)
from smps.data.sources.weather import (
    OpenMeteoSource as OpenMeteoWeatherSource,  # Alias for consistency
)
from smps.data.sources.satellite import (
    MODISNDVISource,
)
from smps.data.sources.soilgrids import (
    SoilGridsGlobalSource,
    get_soilgrids_profile,
)
from smps.data.sources.isda_authenticated import (
    IsdaAfricaAuthenticatedSource,
    get_isda_soil_data,
)
import logging

logger = logging.getLogger(__name__)

# Soil data sources

# Satellite data sources
try:
    import ee  # Test if earthengine-api is available
    from smps.data.sources.gee_satellite import (
        GoogleEarthEngineSatelliteSource,
        setup_gee_authentication,
    )
except ImportError as e:
    logger.warning(f"Could not import Google Earth Engine: {e}")
    GoogleEarthEngineSatelliteSource = None
    setup_gee_authentication = None

# Weather data sources

# SpaceIoTBox API data sources

# Validation data sources

# ISMN file loader

# Base classes

__all__ = [
    # Soil sources
    "IsdaAfricaAuthenticatedSource",
    "SoilGridsGlobalSource",
    "get_isda_soil_data",
    "get_soilgrids_profile",

    # Satellite sources
    "GoogleEarthEngineSatelliteSource",
    "MODISNDVISource",
    "setup_gee_authentication",

    # Weather sources
    "OpenMeteoWeatherSource",

    # SpaceIoTBox sources
    "SpaceIoTBoxClient",
    "SpaceIoTBoxConfig",
    "SpaceIoTBoxWeatherSource",
    "SpaceIoTBoxCopernicusSource",
    "SpaceIoTBoxAgroSource",
    "SpaceIoTBoxDatasetsSource",
    "SpaceIoTBoxUnifiedSource",
    "get_spaceiotbox_weather",
    "get_spaceiotbox_satellite",

    # Validation sources
    "ISMNDataSource",
    "FluxnetDataSource",
    "ValidationDataManager",
    "ValidationObservation",

    # ISMN file loader
    "ISMNStationLoader",
    "ISMNStationData",
    "ISMNSensorMetadata",
    "ISMNSoilProperties",
    "load_ismn_station",
    "get_daily_soil_moisture",

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
