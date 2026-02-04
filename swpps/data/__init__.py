"""
SWPPS Data Module.

Provides data acquisition and quality control:
- Weather data from Open-Meteo API
- Sensor data from WaziGate IoT
- Quality control pipeline
"""

from swpps.data.weather import (
    OpenMeteoClient,
    WeatherFetchRequest,
    fetch_weather_for_plot,
)
from swpps.data.sensors import (
    WaziGateClient,
    SensorDataManager,
)
from swpps.data.quality import (
    QualityControlPipeline,
    QCConfig,
    QCResult,
    QCFlags,
    run_qc_pipeline,
    WeatherGapFiller,
    run_weather_qc,
)
from swpps.data.pipeline import (
    DataPipeline,
    DataPipelineConfig,
)
from swpps.data.sources.base import (
    DataSource,
    DataSourceConfig,
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
]
