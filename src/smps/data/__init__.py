"""
SWPPS Data Module.

Provides data acquisition and quality control:
- Weather data from Open-Meteo API
- Sensor data from WaziGate IoT
- Quality control pipeline
"""

from smps.data.weather import (
    OpenMeteoClient,
    WeatherFetchRequest,
    fetch_weather_for_plot,
)
from smps.data.sensors import (
    WaziGateClient,
    SensorDataManager,
)
from smps.data.quality import (
    QualityControlPipeline,
    QCConfig,
    QCResult,
    QCFlags,
    run_qc_pipeline,
    WeatherGapFiller,
    run_weather_qc,
)
from smps.data.pipeline import (
    DataPipeline,
    DataPipelineConfig,
)
from smps.data.sources.base import (
    DataSource,
    DataSourceConfig,
)
from smps.data.site_manager import (
    SiteManager,
    SiteMetadata,
)
from smps.data.preprocessor import (
    DataPreprocessor,
    TemporalSplitConfig,
)

__all__ = [
    # Weather
    "OpenMeteoClient",
    "WeatherFetchRequest",
    "fetch_weather_for_plot",
    # Sensors
    "WaziGateClient",
    "SensorDataManager",
    # Quality control
    "QualityControlPipeline",
    "QCConfig",
    "QCResult",
    "QCFlags",
    "run_qc_pipeline",
    # Pipeline
    "DataPipeline",
    "DataPipelineConfig",
    # Sources
    "DataSource",
    "DataSourceConfig",
    # New data science modules
    "SiteManager",
    "SiteMetadata",
    "DataPreprocessor",
    "TemporalSplitConfig",
]
