"""
Validation Configuration Module.

Centralizes all configurable parameters for ISMN and physics validation.
Eliminates hardcoded values throughout the codebase.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from datetime import date


@dataclass
class LayerDepthConfig:
    """Configuration for soil layer depths."""
    surface_depth_m: float = 0.10  # 0-10 cm
    root_zone_depth_m: float = 0.40  # 10-50 cm (40 cm thick)
    deep_depth_m: float = 0.50  # 50-100 cm (50 cm thick)

    @property
    def total_depth_m(self) -> float:
        return self.surface_depth_m + self.root_zone_depth_m + self.deep_depth_m

    @property
    def layer_boundaries_m(self) -> List[float]:
        """Return layer boundaries [0, 0.10, 0.40, 1.0]"""
        return [
            0.0,
            self.surface_depth_m,
            self.surface_depth_m + self.root_zone_depth_m,
            self.total_depth_m
        ]


@dataclass
class SensorDepthMapping:
    """Configuration for mapping sensor depths to model layers."""
    # Thresholds for assigning sensors to layers
    surface_max_depth_m: float = 0.10
    root_zone_max_depth_m: float = 0.50
    deep_max_depth_m: float = 1.00

    def get_model_layer(self, sensor_depth_m: float) -> str:
        """Determine which model layer a sensor depth corresponds to."""
        if sensor_depth_m <= self.surface_max_depth_m:
            return "surface"
        elif sensor_depth_m <= self.root_zone_max_depth_m:
            return "root_zone"
        else:
            return "deep"

    def get_layer_weights(self, sensor_depth_m: float) -> Dict[str, float]:
        """
        Get interpolation weights for blending layer values.

        Returns dict with weights for surface, root_zone, deep.
        """
        if sensor_depth_m <= 0.05:
            return {"surface": 1.0, "root_zone": 0.0, "deep": 0.0}
        elif sensor_depth_m <= 0.10:
            # Pure surface
            return {"surface": 1.0, "root_zone": 0.0, "deep": 0.0}
        elif sensor_depth_m <= 0.20:
            # Blend surface and root zone
            w = (sensor_depth_m - 0.10) / 0.10
            return {"surface": 1.0 - w, "root_zone": w, "deep": 0.0}
        elif sensor_depth_m <= 0.30:
            # Mostly root zone
            return {"surface": 0.0, "root_zone": 1.0, "deep": 0.0}
        elif sensor_depth_m <= 0.50:
            # Blend root zone and deep
            w = (sensor_depth_m - 0.30) / 0.20
            return {"surface": 0.0, "root_zone": 1.0 - w, "deep": w}
        else:
            # Deep layer
            return {"surface": 0.0, "root_zone": 0.0, "deep": 1.0}


@dataclass
class ISMNValidationConfig:
    """Configuration for ISMN validation runs."""
    # Date range
    start_date: date = field(default_factory=lambda: date(2020, 1, 1))
    end_date: date = field(default_factory=lambda: date(2021, 12, 31))

    # Site selection
    max_sites: int = 20
    min_observation_days: int = 100
    required_quality_flags: List[str] = field(
        default_factory=lambda: ["G", "M", "1"])

    # Depth distribution (how many sites per depth)
    depth_distribution: Dict[float, int] = field(default_factory=lambda: {
        0.05: 4,   # 5cm - surface
        0.10: 10,  # 10cm - surface
        0.20: 6,   # 20cm - root zone
        0.30: 4,   # 30cm - root zone
        0.60: 2,   # 60cm - deep
    })

    # Layer depths
    layer_depths: LayerDepthConfig = field(default_factory=LayerDepthConfig)
    sensor_mapping: SensorDepthMapping = field(
        default_factory=SensorDepthMapping)

    # Caching
    cache_dir: Path = field(default_factory=lambda: Path("./data/cache"))
    weather_cache_ttl_days: int = 30

    # Rate limiting
    openmeteo_rate_limit_seconds: float = 1.0
    isda_rate_limit_seconds: float = 0.5

    # Output
    output_dir: Path = field(
        default_factory=lambda: Path("./scripts/data/ismn"))
    save_daily_results: bool = True
    save_summary: bool = True


@dataclass
class DefaultSoilParameters:
    """Default soil parameters when data sources fail."""
    # Typical tropical loamy soil
    sand_percent: float = 40.0
    clay_percent: float = 30.0
    silt_percent: float = 30.0
    porosity: float = 0.45
    field_capacity: float = 0.30
    wilting_point: float = 0.12
    organic_carbon_percent: float = 1.5
    bulk_density_g_cm3: float = 1.3

    @classmethod
    def for_region(cls, region: str) -> "DefaultSoilParameters":
        """Get region-specific defaults."""
        regions = {
            "east_africa": cls(sand_percent=35, clay_percent=35),
            "west_africa": cls(sand_percent=45, clay_percent=25),
            "southern_africa": cls(sand_percent=40, clay_percent=30),
            "sahel": cls(sand_percent=60, clay_percent=15, field_capacity=0.22),
            "default": cls(),
        }
        return regions.get(region.lower(), regions["default"])


@dataclass
class NDVIConfig:
    """Configuration for NDVI data and fallbacks."""
    # NDVI bounds for fc calculation
    ndvi_min_bare_soil: float = 0.15
    ndvi_max_vegetation: float = 0.90

    # Seasonal pattern parameters for synthetic NDVI
    tropical_base: float = 0.55
    tropical_amplitude: float = 0.15
    temperate_base: float = 0.35
    temperate_amplitude: float = 0.35

    # Phase shifts (day of year offsets)
    tropical_phase_offset: int = 60
    northern_phase_offset: int = 100
    southern_phase_offset: int = -80  # Opposite season


@dataclass
class PhysicsModelConfig:
    """Configuration for physics model parameters."""
    # Infiltration
    default_max_infiltration_mm_hr: float = 20.0

    # ET partitioning
    min_crop_coefficient: float = 0.30
    max_crop_coefficient: float = 1.20

    # Drainage
    drainage_coefficient: float = 0.10
    deep_percolation_fraction: float = 0.20

    # Root distribution (fraction in each layer)
    root_fraction_surface: float = 0.40
    root_fraction_root_zone: float = 0.50
    root_fraction_deep: float = 0.10

    # Water balance tolerance
    water_balance_tolerance_mm: float = 0.1


@dataclass
class ValidationMetricsThresholds:
    """Thresholds for validation metric quality assessment."""
    # Good performance thresholds
    rmse_good: float = 0.05  # m³/m³
    rmse_acceptable: float = 0.08
    rmse_poor: float = 0.12

    mae_good: float = 0.04
    mae_acceptable: float = 0.06

    bias_good: float = 0.02
    bias_acceptable: float = 0.05

    correlation_good: float = 0.80
    correlation_acceptable: float = 0.60


# Global default configurations
DEFAULT_VALIDATION_CONFIG = ISMNValidationConfig()
DEFAULT_SOIL_PARAMS = DefaultSoilParameters()
DEFAULT_LAYER_DEPTHS = LayerDepthConfig()
DEFAULT_SENSOR_MAPPING = SensorDepthMapping()
DEFAULT_PHYSICS_CONFIG = PhysicsModelConfig()
DEFAULT_METRICS_THRESHOLDS = ValidationMetricsThresholds()


def get_validation_config(
    config_path: Optional[Path] = None,
    **overrides
) -> ISMNValidationConfig:
    """
    Get validation configuration with optional overrides.

    Args:
        config_path: Path to YAML config file (optional)
        **overrides: Parameter overrides

    Returns:
        ISMNValidationConfig instance
    """
    if config_path and config_path.exists():
        import yaml
        with open(config_path) as f:
            file_config = yaml.safe_load(f)
        return ISMNValidationConfig(**{**file_config, **overrides})

    if overrides:
        return ISMNValidationConfig(**overrides)

    return DEFAULT_VALIDATION_CONFIG
