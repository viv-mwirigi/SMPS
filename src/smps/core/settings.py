"""
Global settings for SMPS using pydantic-settings.

This provides centralized configuration management with environment variable
support and validation.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class SMPSSettings(BaseSettings):
    """Global settings for SMPS."""

    # Project paths
    project_root: Path = Path(__file__).parent.parent.parent.parent
    data_dir: Path = project_root / "data"
    results_dir: Path = project_root / "results"
    logs_dir: Path = project_root / "logs"
    models_dir: Path = results_dir / "models"

    # Data sources
    cache_dir: Path = data_dir / "cache"
    weather_cache_dir: Optional[Path] = None
    soil_cache_dir: Optional[Path] = None
    satellite_cache_dir: Optional[Path] = None

    # API keys and credentials
    openmeteo_api_key: Optional[str] = None
    earthengine_service_account: Optional[str] = None
    isda_username: Optional[str] = None
    isda_password: Optional[str] = None
    google_cloud_project: Optional[str] = None
    spaceiotbox_username: Optional[str] = None
    spaceiotbox_password: Optional[str] = None

    # Model training settings
    default_n_estimators: int = 1000
    default_learning_rate: float = 0.03
    default_max_depth: int = 6
    default_cv_folds: int = 5

    # Physics settings
    default_vg_ensemble_size: int = 5
    enable_tropical_corrections: bool = True

    # Validation settings
    validation_horizons_hours: list[int] = [24, 72, 168]
    physics_pass_kge_min: float = 0.40
    physics_pass_nse_min: float = 0.30

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SMPS_",
        case_sensitive=False,
    )

    def ensure_directories(self):
        """Ensure all required directories exist."""
        for dir_path in [self.data_dir, self.results_dir, self.logs_dir,
                         self.models_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = SMPSSettings()
settings.ensure_directories()
