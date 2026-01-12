"""
Type definitions and type aliases for the Smps system.
Provides strong typing throughout the codebase.
"""
from pathlib import Path
from datetime import date, datetime
from typing import TypedDict, NamedTuple, Protocol, runtime_checkable, List, Dict, Any, Tuple, Union, Literal, Optional
from enum import Enum, auto
from dataclasses import dataclass
from typing_extensions import TypeAlias
import numpy as np
import pandas as pd
from pydantic import BaseModel


# Type aliases for clarity
SiteID: TypeAlias = str
Date: TypeAlias = date
DateTime: TypeAlias = datetime
DepthCm: TypeAlias = int
SoilMoistureVWC: TypeAlias = float  # m³/m³
SoilTensionKPa: TypeAlias = float
PrecipitationMm: TypeAlias = float
ET0Mm: TypeAlias = float

# Array types for static typing with numpy
FloatArray: TypeAlias = np.ndarray  # Shape: (n_samples, n_features)
SoilMoistureArray: TypeAlias = np.ndarray  # Shape: (n_timesteps, n_depths)
FluxArray: TypeAlias = np.ndarray  # Shape: (n_timesteps, n_fluxes)


@dataclass(frozen=True)
class SiteMetadata:
    """Immutable site metadata"""
    site_id: SiteID
    latitude: float
    longitude: float
    elevation_m: Optional[float] = None
    soil_texture_class: Optional[str] = None
    crop_type: Optional[str] = None
    planting_date: Optional[Date] = None
    harvest_date: Optional[Date] = None


@dataclass(frozen=True)
class SoilParameters:
    """Soil hydraulic parameters"""
    sand_percent: float
    silt_percent: float
    clay_percent: float
    porosity: float
    field_capacity: float
    wilting_point: float
    saturated_hydraulic_conductivity_cm_day: float
    van_genuchten_alpha: Optional[float] = None
    van_genuchten_n: Optional[float] = None
    bulk_density_g_cm3: Optional[float] = None
    organic_matter_percent: Optional[float] = None

    @property
    def sand_fraction(self) -> float:
        """Sand content as fraction"""
        return self.sand_percent / 100.0

    @property
    def clay_fraction(self) -> float:
        """Clay content as fraction"""
        return self.clay_percent / 100.0


class SoilLayer(str, Enum):
    """Soil layer identifiers"""
    SURFACE = "surface"  # 0-10cm
    UPPER_ROOT = "upper_root"  # 10-30cm
    LOWER_ROOT = "lower_root"  # 30-50cm
    TRANSITION = "transition"  # 50-75cm
    DEEP = "deep"  # 75-100cm

    @property
    def depth_range_cm(self) -> Tuple[int, int]:
        ranges = {
            SoilLayer.SURFACE: (0, 10),
            SoilLayer.UPPER_ROOT: (10, 30),
            SoilLayer.LOWER_ROOT: (30, 50),
            SoilLayer.TRANSITION: (50, 75),
            SoilLayer.DEEP: (75, 100)
        }
        return ranges[self]


@dataclass
class PhysicsPriorResult:
    """Results from physics prior model"""
    date: Date
    theta_surface: SoilMoistureVWC
    theta_root: SoilMoistureVWC
    theta_deep: Optional[SoilMoistureVWC] = None  # For 3-layer models
    fluxes: Dict[str, float] = None  # evapotranspiration, drainage, etc.
    water_balance_error: float = 0.0  # Should be near zero
    converged: bool = True

    def __post_init__(self):
        if self.fluxes is None:
            self.fluxes = {}


@dataclass
class ModelPrediction:
    """Complete model prediction with uncertainty"""
    date: Date
    point_prediction: SoilMoistureVWC
    lower_bound: SoilMoistureVWC
    upper_bound: SoilMoistureVWC
    confidence: float
    feature_contributions: Dict[str, float]  # SHAP values
    physics_prior: SoilMoistureVWC
    residuals: SoilMoistureVWC


class DataQualityFlag(Enum):
    """Data quality flags"""
    OK = auto()
    MISSING = auto()
    OUTLIER = auto()
    INTERPOLATED = auto()
    CLIMATOLOGY = auto()
    UNCERTAIN = auto()
    FAILED_QC = auto()
    FLAGGED = auto()  # Generic flag for questionable data


@dataclass
class QualityControlResult:
    """Results of data quality control"""
    flag: DataQualityFlag
    original_value: float
    corrected_value: Optional[float] = None
    confidence: float = 1.0
    reason: Optional[str] = None


# Protocol definitions for dependency injection
@runtime_checkable
class DataSource(Protocol):
    """Protocol for data sources"""

    def fetch(self, site_id: SiteID, start_date: Date, end_date: Date) -> pd.DataFrame:
        """Fetch data for given site and date range"""
        ...

    def get_metadata(self) -> Dict[str, Any]:
        """Get source metadata (resolution, coverage, etc.)"""
        ...


@runtime_checkable
class SoilMoistureModel(Protocol):
    """Protocol for soil moisture prediction models"""

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "SoilMoistureModel":
        """Train model"""
        ...

    def predict(self, X: pd.DataFrame, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions"""
        ...

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        """Explain predictions (SHAP, feature importance)"""
        ...

    def save(self, path: Path):
        """Save model to disk"""
        ...

    @classmethod
    def load(cls, path: Path) -> "SoilMoistureModel":
        """Load model from disk"""
        ...


# Pydantic models for API serialization
class PredictionRequest(BaseModel):
    """Request for soil moisture prediction"""
    site_id: SiteID
    start_date: Date
    end_date: Date
    include_uncertainty: bool = True
    include_explanations: bool = True
    return_format: Literal["json", "csv", "parquet"] = "json"


class PredictionResponse(BaseModel):
    """Response with soil moisture predictions"""
    site_id: SiteID
    predictions: List[ModelPrediction]
    metadata: Dict[str, Any]
    model_version: str
    processing_time_ms: float
