"""
Configuration system with validation and environment awareness.
Based on Pydantic Settings for robust configuration management.
"""
from pathlib import Path
import yaml
from pydantic import Field, field_validator, model_validator, ValidationInfo, ConfigDict
from pydantic_settings import BaseSettings
from typing import List, Dict, Optional, Literal, Union
from enum import Enum



class DataSourceType(str, Enum):
    ISDA_SOIL = "isdasoil"
    SOILGRIDS = "soilgrids"
    ERA5 = "era5"
    OPEN_METEO = "open_meteo"
    GRAFS = "grafs"
    SMAP = "smap"
    SENTINEL2 = "sentinel2"
    SENTINEL1 = "sentinel1"
    MODIS = "modis"
    SENSLOG = "senslog"
    WAZIUP = "waziup"




class DataConfig(BaseSettings):
    """Configuration for dynamic data sources (Africa-focused)"""

    # REQUIRED CREDENTIALS
    isdasoil_username: Optional[str] = Field(None, description="iSDAsoil API username")
    isdasoil_password: Optional[str] = Field(None, description="iSDAsoil API password (from env)")
    era5_cds_uid: Optional[str] = Field(None, description="Copernicus CDS UID")
    era5_cds_key: Optional[str] = Field(None, description="Copernicus CDS API key")

    # OPTIONAL: Google Earth Engine (for Sentinel-2, MODIS, GRAFS)
    gee_project_id: Optional[str] = Field(None, description="GEE project ID (if using GEE)")
    use_gee: bool = Field(False, description="Enable Google Earth Engine for RS data")

     # DATA SOURCE PRIORITY (dynamic > fallback > constants)
    soil_data_priority: List[DataSourceType] = Field(
        default=[DataSourceType.ISDA_SOIL, DataSourceType.SOILGRIDS],
        description="Priority order for soil property fetching"
    )

    precipitation_source: DataSourceType = Field(
        default=DataSourceType.ERA5,
        description="Primary precipitation source"
    )

    et0_source: DataSourceType = Field(
        default=DataSourceType.ERA5,
        description="Primary ET0 source (calculated from ERA5)"
    )

    # REFERENCE SOIL MOISTURE (for assimilation)
    observation_sources: List[DataSourceType] = Field(
        default=[DataSourceType.GRAFS, DataSourceType.SMAP],
        description="Data sources for validation/assimilation"
    )

    # REMOTE SENSING (vegetation indices)
    ndvi_source: DataSourceType = Field(
        default=DataSourceType.SENTINEL2,
        description="NDVI source (Sentinel-2 or MODIS)"
    )

    # CACHING & PERFORMANCE
    cache_soil_properties_days: int = Field(
        default=30,
        description="Cache soil properties (slow to change)"
    )
    cache_forcing_days: int = Field(
        default=1,
        description="Cache daily forcing (weather updates daily)"
    )
    cache_dir: Path = Field(default=Path("./data/cache"))

    # FALLBACKS
    fallback_texture_class: str = Field(
        default="loam",
        description="Fallback if all data sources fail"
    )

    max_retries_data_fetch: int = Field(default=3)
    timeout_seconds: int = Field(default=30)

    # Temporal settings
    target_frequency: str = Field("1D", description="Target frequency for aggregation")
    min_data_coverage: float = Field(0.8, ge=0, le=1, description="Minimum data coverage required")

    # Quality control
    outlier_sigma: float = Field(3.0, gt=0, description="Sigma for outlier detection")
    max_gap_days: int = Field(7, gt=0, description="Maximum gap to interpolate")


    feature_store_dir: Path = Field(Path("./data/features"), description="Feature store directory")



    model_config = ConfigDict(env_prefix="SMPS_DATA_", case_sensitive=False)

class ValidationStrategy(str, Enum):
    """Cross-validation strategies"""
    LEAVE_SITE_OUT = "leave_site_out"
    TEMPORAL_SPLIT = "temporal_split"
    K_FOLD = "k_fold"
    NESTED_CV = "nested_cv"


class PhysicsPriorConfig(BaseSettings):
    """Configuration for physics prior model"""

    # Layer definitions
    surface_depth_m: float = Field(0.1, gt=0, description="Surface layer depth in meters")
    root_zone_depth_m: float = Field(0.3, gt=0, description="Root zone depth in meters")



    # ET partitioning
    ndvi_et_partitioning_lambda: float = Field(0.5, ge=0, le=2)

    # Numerical stability
    min_soil_moisture: float = Field(0.01, ge=0, description="Minimum soil moisture (residual)")
    max_soil_moisture: float = Field(0.5, le=1, description="Maximum soil moisture (porosity)")

    # Runtime options
    enforce_water_balance: bool = Field(True, description="Ensure water balance closure")
    max_iterations: int = Field(100, gt=0, description="Maximum iterations for numerical solution")


class FeatureConfig(BaseSettings):
    """Configuration for feature engineering"""

    # Temporal features
    lag_days: List[int] = Field([1, 3, 7, 14, 30], description="Days to lag soil moisture")
    rolling_windows: List[int] = Field([3, 7, 14, 30], description="Rolling window sizes")
    cumulative_windows: List[int] = Field([1, 3, 7, 14, 30], description="Cumulative windows")

    # Physics features
    include_physics_fluxes: bool = Field(True, description="Include flux terms from physics model")
    include_physics_residuals: bool = Field(True, description="Include physics model residuals")

    # Remote sensing features
    ndvi_features: List[str] = Field(
        ["current", "anomaly", "slope_14d"],
        description="NDVI features to compute"
    )

    # Feature selection
    max_features: int = Field(50, gt=0, description="Maximum number of features to use")
    min_feature_importance: float = Field(0.001, ge=0, description="Minimum SHAP importance")


class ModelConfig(BaseSettings):
    """Configuration for machine learning models"""

    # Model selection
    primary_model: Literal["lightgbm", "xgboost", "catboost"] = "lightgbm"
    use_ensemble: bool = Field(True, description="Use ensemble of models")
    ensemble_members: List[str] = Field(["lightgbm", "xgboost", "catboost"])

    # LightGBM specific
    lightgbm_params: Dict = Field(
        default={
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": 8,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 20,
            "verbose": -1
        }
    )

    # Training parameters
    validation_strategy: ValidationStrategy = ValidationStrategy.LEAVE_SITE_OUT
    n_folds: int = Field(5, ge=2, description="Number of CV folds")
    test_size_days: int = Field(90, gt=0, description="Days to hold out for testing")

    # Early stopping
    early_stopping_rounds: int = Field(100, gt=0)
    early_stopping_tolerance: float = Field(0.001, ge=0)

    # Physics-informed loss
    physics_loss_weight: float = Field(0.1, ge=0, description="Weight for physics regularization")
    penalty_depth_violation: float = Field(10.0, ge=0, description="Penalty for depth ordering violations")
    penalty_unphysical: float = Field(5.0, ge=0, description="Penalty for unphysical predictions")


class UncertaintyConfig(BaseSettings):
    """Configuration for uncertainty quantification"""

    method: Literal["conformal", "quantile", "ensemble", "bayesian"] = "conformal"
    confidence_level: float = Field(0.9, gt=0, lt=1, description="Prediction interval coverage")

    # Conformal prediction
    calibration_size: float = Field(0.2, gt=0, lt=1, description="Fraction of data for calibration")

    # Quantile regression
    quantiles: List[float] = Field([0.1, 0.5, 0.9], description="Quantiles to predict")

    # Ensemble settings
    n_ensemble_members: int = Field(10, ge=1, description="Number of ensemble members")
    ensemble_diversity_method: Literal["bagging", "random_seeds", "hyperparameter_variation"] = "bagging"



class MonitoringConfig(BaseSettings):
    """Configuration for monitoring and observability"""

    # Metrics collection
    track_feature_importance: bool = Field(True)
    track_prediction_distribution: bool = Field(True)
    track_data_coverage: bool = Field(True)

    # Alert thresholds
    rmse_alert_threshold: float = Field(0.1)
    data_coverage_alert_threshold: float = Field(0.5)
    feature_drift_threshold: float = Field(0.3)

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # External monitoring
    enable_prometheus: bool = Field(False)
    prometheus_port: int = Field(9090)
    enable_sentry: bool = Field(False)
    sentry_dsn: Optional[str] = None

class EdgeConfig(BaseSettings):
    """Configuration for edge deployment"""

    device_type: Literal["wazigate", "raspberry_pi", "custom"] = "wazigate"
    max_memory_mb: int = Field(512, description="Maximum memory for edge model")
    max_model_size_mb: int = Field(50, description="Maximum model file size")

    # Fallback behavior
    fallback_to_physics: bool = Field(True)
    cache_days: int = Field(7, description="Days of cached data to keep")

    # Sync settings
    sync_interval_minutes: int = Field(60)
    sync_on_wifi_only: bool = Field(True)

    # Battery optimization
    inference_interval_hours: int = Field(6)
    sleep_during_night: bool = Field(True)

class APIConfig(BaseSettings):
    """Configuration for API endpoints"""

    host: str = Field("0.0.0.0")
    port: int = Field(8000)
    workers: int = Field(4)

    # Rate limiting
    rate_limit_per_minute: int = Field(60)
    enable_auth: bool = Field(False)

    # Response caching
    cache_ttl_seconds: int = Field(300)
    max_response_size_mb: int = Field(10)

    # Documentation
    enable_swagger: bool = Field(True)
    enable_redoc: bool = Field(False)


class SmpsConfig(BaseSettings):
    """Main configuration for the Smps system"""

    # System
    project_name: str = "smps"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    random_seed: int = Field(42, description="Random seed for reproducibility")

    # Component configurations
    physics: PhysicsPriorConfig = PhysicsPriorConfig()
    features: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()
    uncertainty: UncertaintyConfig = UncertaintyConfig()
    data: DataConfig = Field(default_factory=DataConfig)

    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent.parent
    data_dir: Path = Field(None)
    models_dir: Path = Field(None)
    logs_dir: Path = Field(None)

    # SITE METADATA
    site_configs: Dict[str, Dict] = Field(
        default_factory=dict,
        description="Site-specific overrides (crop type, irrigation, etc.)"
    )

    model_config = ConfigDict(
        env_prefix="SMPS_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )


    @field_validator("data_dir", "models_dir", "logs_dir", mode="before")
    @classmethod
    def set_paths(cls, v, info: ValidationInfo):
        """Set default paths relative to base directory"""
        if v is not None:
            return Path(v)

        base = info.data.get("base_dir", Path.cwd())
        field_name = info.field_name

        if field_name == "data_dir":
            return base / "data"
        if field_name == "models_dir":
            return base / "models"
        if field_name == "logs_dir":
            return base / "logs"

    @model_validator(mode="after")
    def validate_config(self):
        """Cross-field validation"""
        if self.environment == "production" and self.debug:
            raise ValueError("Debug mode cannot be enabled in production")

        # Ensure cache directory exists
        cache_dir = self.data.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        return self

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "SmpsConfig":
        """Load configuration from YAML file"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)

        return cls(**yaml_config)

    def to_yaml(self, yaml_path: Union[str, Path]):
        """Save configuration to YAML file"""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def get_site_config(self, site_id: str) -> Dict:
        """Get site-specific configuration overrides"""
        if self.site_configs is None:
            return {}
        return self.site_configs.get(site_id, {})

# USAGE: Environment variables override defaults
# export SMPS_DATA__ISDASOIL_USERNAME=...
# export SMPS_DATA__ERA5_CDS_UID=...
config = SmpsConfig()

# Global configuration instance
_config: Optional[SmpsConfig] = None


def get_config(config_path: Optional[Path] = None) -> SmpsConfig:
    """Get or create configuration instance (singleton pattern)"""
    global _config

    if _config is None:
        if config_path and config_path.exists():
            _config = SmpsConfig.from_yaml(config_path)
        else:
            # Try to load from environment
            _config = SmpsConfig()

    return _config


def set_config(config: SmpsConfig):
    """Set configuration (useful for testing)"""
    global _config
    _config = config