"""
Configuration management for SWPPS.

Handles loading, validation, and management of system configuration
including plot-specific settings and model parameters.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from smps.core.constants import CROP_THRESHOLDS, MODEL_DEFAULTS

logger = logging.getLogger("swpps.core.config")


@dataclass
class SensorConfig:
    """Configuration for a sensor device."""
    device_id: str
    sensor_id: str
    sensor_type: str  # "tension", "capacitive", etc.
    depth_cm: int = 20
    unit: str = "cbar"

    @property
    def full_id(self) -> str:
        """Full sensor identifier."""
        return f"{self.device_id}/{self.sensor_id}"


@dataclass
class PlotConfig:
    """Configuration for a single agricultural plot."""
    # Identity
    plot_id: int
    name: str

    # Location
    latitude: float
    longitude: float
    timezone: Optional[str] = None

    # Sensors
    moisture_sensors: List[SensorConfig] = field(default_factory=list)
    temperature_sensors: List[SensorConfig] = field(default_factory=list)
    flow_sensors: List[SensorConfig] = field(default_factory=list)

    # Crop and soil
    crop_type: str = "default"
    soil_type: str = "loam"

    # Irrigation thresholds (kPa, will be populated from crop type)
    irrigate_at_kpa: float = -80.0
    refill_to_kpa: float = -25.0
    critical_at_kpa: float = -200.0

    # Irrigation settings
    irrigation_amount_liters: float = 100.0
    look_ahead_hours: float = 24.0

    # Training settings
    start_date: Optional[str] = None
    train_period_days: int = 30
    predict_period_hours: int = 6

    # Advanced soil parameters (optional, will be estimated if not provided)
    sand_percent: Optional[float] = None
    clay_percent: Optional[float] = None
    organic_matter_percent: Optional[float] = None

    def __post_init__(self):
        """Apply crop-specific thresholds if not explicitly set."""
        if self.crop_type in CROP_THRESHOLDS:
            crop_defaults = CROP_THRESHOLDS[self.crop_type]
            # Only override if using default values
            if self.irrigate_at_kpa == -80.0:
                self.irrigate_at_kpa = crop_defaults.get(
                    "irrigate_below_kpa", -80.0)
            if self.refill_to_kpa == -25.0:
                self.refill_to_kpa = crop_defaults.get("refill_to_kpa", -25.0)
            if self.critical_at_kpa == -200.0:
                self.critical_at_kpa = crop_defaults.get(
                    "stress_threshold_kpa", -200.0)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], plot_id: int) -> "PlotConfig":
        """Create PlotConfig from dictionary (e.g., loaded from JSON)."""
        # Parse sensor configurations
        moisture_sensors = []
        for sensor_path in data.get("DeviceAndSensorIdsMoisture", []):
            if "/" in sensor_path:
                device_id, sensor_id = sensor_path.split("/", 1)
                moisture_sensors.append(SensorConfig(
                    device_id=device_id,
                    sensor_id=sensor_id,
                    sensor_type=data.get("Sensor_kind", "tension"),
                ))

        temperature_sensors = []
        for sensor_path in data.get("DeviceAndSensorIdsTemp", []):
            if "/" in sensor_path:
                device_id, sensor_id = sensor_path.split("/", 1)
                temperature_sensors.append(SensorConfig(
                    device_id=device_id,
                    sensor_id=sensor_id,
                    sensor_type="temperature",
                    unit="celsius",
                ))

        flow_sensors = []
        for sensor_path in data.get("DeviceAndSensorIdsFlow", []):
            if "/" in sensor_path:
                device_id, sensor_id = sensor_path.split("/", 1)
                flow_sensors.append(SensorConfig(
                    device_id=device_id,
                    sensor_id=sensor_id,
                    sensor_type="flow",
                    unit="liters",
                ))

        # Parse GPS
        gps_info = data.get("Gps_info", {})
        latitude = float(gps_info.get(
            "lattitude", gps_info.get("latitude", 0)))
        longitude = float(gps_info.get("longitude", 0))

        return cls(
            plot_id=plot_id,
            name=data.get("Name", f"Plot {plot_id}"),
            latitude=latitude,
            longitude=longitude,
            moisture_sensors=moisture_sensors,
            temperature_sensors=temperature_sensors,
            flow_sensors=flow_sensors,
            crop_type=data.get("Crop_type", "default"),
            soil_type=data.get("Soil_type", "loam"),
            # Convert to kPa
            irrigate_at_kpa=-float(data.get("Threshold", 80)),
            irrigation_amount_liters=float(data.get("Irrigation_amount", 100)),
            look_ahead_hours=float(data.get("Look_ahead_time", 24)),
            start_date=data.get("Start_date"),
            train_period_days=int(data.get("Period", 30)) or 30,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "plot_id": self.plot_id,
            "Name": self.name,
            "Gps_info": {
                "lattitude": str(self.latitude),
                "longitude": str(self.longitude),
            },
            "DeviceAndSensorIdsMoisture": [s.full_id for s in self.moisture_sensors],
            "DeviceAndSensorIdsTemp": [s.full_id for s in self.temperature_sensors],
            "DeviceAndSensorIdsFlow": [s.full_id for s in self.flow_sensors],
            "Crop_type": self.crop_type,
            "Soil_type": self.soil_type,
            "Threshold": abs(self.irrigate_at_kpa),  # Store as positive cbar
            "Irrigation_amount": self.irrigation_amount_liters,
            "Look_ahead_time": self.look_ahead_hours,
            "Start_date": self.start_date,
            "Period": self.train_period_days,
        }


@dataclass
class SWPPSConfig:
    """Main configuration for the SWPPS system."""

    # API settings
    api_url: str = "http://wazigate/"
    api_token: Optional[str] = None

    # Paths
    config_dir: Path = field(default_factory=lambda: Path("config"))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))

    # Plot configurations
    plots: Dict[int, PlotConfig] = field(default_factory=dict)

    # Model settings
    physics_enabled: bool = True
    ml_enabled: bool = True
    hybrid_mode: bool = True  # Physics + ML residual

    # Training settings
    auto_retrain: bool = True
    retrain_interval_days: int = 7
    min_training_samples: int = 30

    # Prediction settings
    forecast_horizons_hours: List[int] = field(
        default_factory=lambda: [0, 6, 24, 72, 168]
    )
    prediction_interval_hours: int = 6

    # Uncertainty quantification
    uncertainty_enabled: bool = True
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Logging
    verbose: bool = True
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "SWPPSConfig":
        """Load configuration from environment variables and .env file."""
        from dotenv import load_dotenv
        load_dotenv()

        return cls(
            api_url=os.getenv("API_URL", "http://wazigate/"),
            api_token=os.getenv("API_TOKEN"),
            verbose=os.getenv("VERBOSE", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def load_plots(self) -> None:
        """Load plot configurations from config directory."""
        if not self.config_dir.exists():
            logger.warning("Config directory %s does not exist",
                           self.config_dir)
            return

        for config_file in self.config_dir.glob("current_config_plot*.json"):
            try:
                # Extract plot ID from filename
                import re
                match = re.search(r'plot(\d+)\.json$', config_file.name)
                if not match:
                    continue
                plot_id = int(match.group(1))

                # Load configuration
                with open(config_file, 'r') as f:
                    data = json.load(f)

                plot_config = PlotConfig.from_dict(data, plot_id)
                self.plots[plot_id] = plot_config

                logger.info("Loaded configuration for plot %d: %s",
                            plot_id, plot_config.name)

            except Exception as e:
                logger.error("Failed to load config from %s: %s",
                             config_file, e)

    def save_plot(self, plot_config: PlotConfig) -> None:
        """Save a plot configuration to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

        config_file = self.config_dir / \
            f"current_config_plot{plot_config.plot_id}.json"

        with open(config_file, 'w') as f:
            json.dump(plot_config.to_dict(), f, indent=4)

        self.plots[plot_config.plot_id] = plot_config
        logger.info("Saved configuration for plot %d", plot_config.plot_id)

    def get_plot(self, plot_id: int) -> Optional[PlotConfig]:
        """Get plot configuration by ID."""
        return self.plots.get(plot_id)


def get_config() -> SWPPSConfig:
    """Get or create the global configuration instance."""
    if not hasattr(get_config, "_instance"):
        get_config._instance = SWPPSConfig.from_env()
        get_config._instance.load_plots()
    return get_config._instance


def init_config(config: SWPPSConfig) -> None:
    """Initialize with a specific configuration."""
    get_config._instance = config


def load_config(path: Path) -> SWPPSConfig:
    """
    Load configuration from a JSON file.

    Args:
        path: Path to JSON configuration file

    Returns:
        SWPPSConfig instance
    """
    with open(path, 'r') as f:
        data = json.load(f)

    config = SWPPSConfig(**data)
    return config


def save_config(config: SWPPSConfig, path: Path) -> None:
    """
    Save configuration to a JSON file.

    Args:
        config: Configuration to save
        path: Path to save JSON file
    """
    import dataclasses

    data = dataclasses.asdict(config)

    # Convert Path objects to strings
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)

    # Don't save plot instances
    data.pop("plots", None)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
