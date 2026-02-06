"""Core module for SWPPS."""

from smps.core.config import SWPPSConfig, PlotConfig, get_config, init_config, load_config, save_config
from smps.core.constants import (
    IRRIGATION_THRESHOLDS,
    CROP_THRESHOLDS,
    MatricPotentialRange,
    PHYSICAL_CONSTANTS,
    MODEL_DEFAULTS,
    IrrigationAction,
)
from smps.core.types import (
    MatricPotential,
    SoilTensionKPa,
    VolumetricWaterContent,
    SoilMoistureStatus,
    DepthZone,
    VanGenuchtenParams,
    SoilProfile,
    DailyWeather,
    SensorReading,
    PhysicsModelOutput,
    PredictionResult,
    IrrigationDecision,
)
from smps.core.exceptions import (
    SWPPSError,
    ConfigurationError,
    SensorError,
    DataFetchError,
    PhysicsModelError,
    MLModelError,
    InsufficientDataError,
    ActuatorError,
)

__all__ = [
    # Config
    "SWPPSConfig",
    "PlotConfig",
    "get_config",
    "init_config",
    "load_config",
    "save_config",
    # Constants
    "IRRIGATION_THRESHOLDS",
    "CROP_THRESHOLDS",
    "MatricPotentialRange",
    "PHYSICAL_CONSTANTS",
    "MODEL_DEFAULTS",
    "IrrigationAction",
    # Types
    "MatricPotential",
    "SoilTensionKPa",
    "VolumetricWaterContent",
    "SoilMoistureStatus",
    "DepthZone",
    "VanGenuchtenParams",
    "SoilProfile",
    "DailyWeather",
    "SensorReading",
    "PhysicsModelOutput",
    "PredictionResult",
    "IrrigationDecision",
    # Exceptions
    "SWPPSError",
    "ConfigurationError",
    "SensorError",
    "DataFetchError",
    "PhysicsModelError",
    "MLModelError",
    "InsufficientDataError",
    "ActuatorError",
]
