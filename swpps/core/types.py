"""
Core types and type aliases for SWPPS.

Uses matric potential (soil water tension) as the primary representation,
eliminating the need for soil-specific calibration.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, TypeAlias, Union
import numpy as np


# =============================================================================
# TYPE ALIASES
# =============================================================================

# Matric Potential in kPa (negative values, more negative = drier)
# Common range: 0 to -1500 kPa (saturation to permanent wilting)
MatricPotential: TypeAlias = float  # kPa
SoilTensionKPa: TypeAlias = float   # Alias for clarity

# Volumetric water content (m³/m³), used internally for physics
VolumetricWaterContent: TypeAlias = float

# Weather inputs
PrecipitationMm: TypeAlias = float
EvapotranspirationMm: TypeAlias = float
TemperatureCelsius: TypeAlias = float

# Site identifiers
SiteID: TypeAlias = str
PlotID: TypeAlias = Union[str, int]

# Array types
FloatArray: TypeAlias = np.ndarray


# =============================================================================
# ENUMS
# =============================================================================

class SoilMoistureStatus(Enum):
    """Plant-centric soil moisture status based on matric potential."""
    SATURATED = auto()        # ψ > -5 kPa: Too wet, potential root issues
    OPTIMAL_WET = auto()      # -5 to -33 kPa: Upper optimal range
    OPTIMAL = auto()          # -33 to -100 kPa: Ideal range
    OPTIMAL_DRY = auto()      # -100 to -200 kPa: Lower optimal, approaching stress
    MILD_STRESS = auto()      # -200 to -500 kPa: Begin stress
    MODERATE_STRESS = auto()  # -500 to -1000 kPa: Significant stress
    SEVERE_STRESS = auto()    # -1000 to -1500 kPa: Severe stress
    WILTING = auto()          # < -1500 kPa: Permanent wilting

    @classmethod
    def from_potential(cls, psi_kpa: float) -> "SoilMoistureStatus":
        """Determine status from matric potential value."""
        if psi_kpa > -5:
            return cls.SATURATED
        elif psi_kpa > -33:
            return cls.OPTIMAL_WET
        elif psi_kpa > -100:
            return cls.OPTIMAL
        elif psi_kpa > -200:
            return cls.OPTIMAL_DRY
        elif psi_kpa > -500:
            return cls.MILD_STRESS
        elif psi_kpa > -1000:
            return cls.MODERATE_STRESS
        elif psi_kpa > -1500:
            return cls.SEVERE_STRESS
        else:
            return cls.WILTING


class DepthZone(Enum):
    """Soil depth zones for multi-layer modeling."""
    SURFACE = "surface"         # 0-10 cm: Evaporation zone
    SHALLOW_ROOT = "shallow"    # 10-30 cm: Shallow root zone
    DEEP_ROOT = "deep"          # 30-60 cm: Deep root zone
    SUBSOIL = "subsoil"         # 60-100 cm: Below root zone

    @property
    def depth_range_cm(self) -> Tuple[int, int]:
        """Return (top, bottom) depth in cm."""
        ranges = {
            DepthZone.SURFACE: (0, 10),
            DepthZone.SHALLOW_ROOT: (10, 30),
            DepthZone.DEEP_ROOT: (30, 60),
            DepthZone.SUBSOIL: (60, 100),
        }
        return ranges[self]


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class SiteMetadata:
    """Immutable site/plot metadata."""
    site_id: SiteID
    name: str
    latitude: float
    longitude: float
    elevation_m: Optional[float] = None
    timezone: Optional[str] = None
    crop_type: Optional[str] = None
    soil_texture: Optional[str] = None


@dataclass
class VanGenuchtenParams:
    """
    Van Genuchten soil hydraulic parameters.

    These define the relationship between matric potential (ψ) and
    volumetric water content (θ):

    θ(ψ) = θ_r + (θ_s - θ_r) / [1 + (α|ψ|)^n]^m

    where m = 1 - 1/n (Mualem constraint)
    """
    theta_r: float  # Residual water content (m³/m³)
    theta_s: float  # Saturated water content (m³/m³)
    alpha: float    # Air entry parameter (1/kPa)
    n: float        # Pore size distribution index (-)
    K_sat: float    # Saturated hydraulic conductivity (mm/day)

    @property
    def m(self) -> float:
        """Mualem parameter."""
        return 1.0 - 1.0 / self.n

    def theta_from_psi(self, psi_kpa: float) -> float:
        """
        Water content from matric potential using Van Genuchten equation.

        Args:
            psi_kpa: Matric potential in kPa (negative values)

        Returns:
            Volumetric water content (m³/m³)
        """
        if psi_kpa >= 0:
            return self.theta_s

        h = abs(psi_kpa)
        Se = (1.0 + (self.alpha * h) ** self.n) ** (-self.m)
        return self.theta_r + (self.theta_s - self.theta_r) * Se

    def psi_from_theta(self, theta: float) -> float:
        """
        Matric potential from water content (inverse Van Genuchten).

        Args:
            theta: Volumetric water content (m³/m³)

        Returns:
            Matric potential in kPa (negative)
        """
        theta = np.clip(theta, self.theta_r + 1e-6, self.theta_s - 1e-6)
        Se = (theta - self.theta_r) / (self.theta_s - self.theta_r)

        if Se >= 1.0:
            return 0.0

        psi = -((Se ** (-1.0 / self.m) - 1.0) ** (1.0 / self.n)) / self.alpha
        return psi

    def K_from_theta(self, theta: float) -> float:
        """
        Hydraulic conductivity from water content (Van Genuchten-Mualem).

        Args:
            theta: Volumetric water content (m³/m³)

        Returns:
            Hydraulic conductivity (mm/day)
        """
        theta = np.clip(theta, self.theta_r + 1e-6, self.theta_s - 1e-6)
        Se = (theta - self.theta_r) / (self.theta_s - self.theta_r)

        # Mualem model with tortuosity L=0.5
        return self.K_sat * (Se ** 0.5) * (1.0 - (1.0 - Se ** (1.0/self.m)) ** self.m) ** 2

    @property
    def field_capacity_kpa(self) -> float:
        """Field capacity at -33 kPa."""
        return -33.0

    @property
    def wilting_point_kpa(self) -> float:
        """Permanent wilting point at -1500 kPa."""
        return -1500.0

    @property
    def theta_fc(self) -> float:
        """Water content at field capacity."""
        return self.theta_from_psi(-33.0)

    @property
    def theta_wp(self) -> float:
        """Water content at wilting point."""
        return self.theta_from_psi(-1500.0)


@dataclass
class SoilProfile:
    """Complete soil profile with hydraulic parameters."""
    site_id: SiteID
    sand_percent: float
    clay_percent: float
    silt_percent: float
    organic_matter_percent: float
    bulk_density_g_cm3: float
    van_genuchten: VanGenuchtenParams

    @property
    def texture_class(self) -> str:
        """USDA texture class from particle sizes."""
        sand, clay = self.sand_percent, self.clay_percent

        if sand >= 85 and clay < 10:
            return "sand"
        elif sand >= 70 and clay < 15:
            return "loamy_sand"
        elif sand >= 50 and clay < 20:
            return "sandy_loam"
        elif clay >= 40:
            return "clay"
        elif clay >= 35:
            return "clay_loam"
        elif sand < 50 and clay >= 27:
            return "silty_clay"
        else:
            return "loam"


@dataclass
class DailyWeather:
    """Daily weather observation/forecast."""
    date: date
    precipitation_mm: float
    et0_mm: float  # Reference evapotranspiration
    temperature_mean_c: float
    temperature_min_c: float
    temperature_max_c: float
    relative_humidity_mean: float
    solar_radiation_mj_m2: float
    wind_speed_m_s: float

    # Optional advanced fields
    cloud_cover_percent: Optional[float] = None
    vapor_pressure_deficit_kpa: Optional[float] = None


@dataclass
class SensorReading:
    """Single sensor reading from IoT device."""
    timestamp: datetime
    sensor_id: str
    device_id: str
    value: float
    unit: str
    quality_flag: int = 0  # 0 = good, 1 = suspect, 2 = bad

    @property
    def matric_potential_kpa(self) -> Optional[float]:
        """Convert reading to matric potential if applicable."""
        if self.unit in ("cbar", "centibars"):
            return -self.value  # cbar to kPa (1 cbar = 1 kPa)
        elif self.unit in ("kPa", "kpa"):
            return -abs(self.value)  # Ensure negative
        elif self.unit in ("hPa", "hpa", "mbar"):
            return -self.value / 10.0  # hPa to kPa
        return None


@dataclass
class PhysicsModelOutput:
    """Output from physics-based water balance model."""
    date: date
    psi_surface_kpa: MatricPotential      # Surface layer potential
    psi_root_kpa: MatricPotential         # Root zone potential
    psi_deep_kpa: Optional[MatricPotential] = None  # Deep layer

    # Flux terms (mm/day)
    precipitation_mm: float = 0.0
    infiltration_mm: float = 0.0
    runoff_mm: float = 0.0
    evaporation_mm: float = 0.0
    transpiration_mm: float = 0.0
    drainage_mm: float = 0.0

    # Quality indicators
    water_balance_error_mm: float = 0.0
    converged: bool = True

    # Optional θ outputs (computed via VG conversion for backward compatibility)
    theta_surface: Optional[float] = None
    theta_root: Optional[float] = None
    theta_deep: Optional[float] = None

    def compute_theta_from_psi(
        self,
        vg_params: Optional["VanGenuchtenParams"] = None,  # type: ignore
    ) -> "PhysicsModelOutput":
        """
        Convert ψ to θ using Van Genuchten.

        This provides backward compatibility for systems expecting
        volumetric water content instead of matric potential.

        Args:
            vg_params: Van Genuchten parameters (uses defaults if None)

        Returns:
            Self with theta fields populated
        """
        # Delayed import to avoid circular dependency
        from swpps.physics.van_genuchten import (
            VanGenuchtenParams, water_content_from_potential
        )

        if vg_params is None:
            vg_params = VanGenuchtenParams()

        self.theta_surface = water_content_from_potential(
            self.psi_surface_kpa, vg_params
        )
        self.theta_root = water_content_from_potential(
            self.psi_root_kpa, vg_params
        )
        if self.psi_deep_kpa is not None:
            self.theta_deep = water_content_from_potential(
                self.psi_deep_kpa, vg_params
            )

        return self


@dataclass
class PredictionResult:
    """
    Complete prediction result with uncertainty.

    The core output of SWPPS - matric potential prediction with
    uncertainty bounds and irrigation recommendation.
    """
    timestamp: datetime
    horizon_hours: int  # Forecast horizon (0 = nowcast)

    # Matric potential predictions (kPa, negative values)
    psi_predicted_kpa: MatricPotential         # Point estimate
    psi_lower_bound_kpa: MatricPotential       # 10th percentile
    psi_upper_bound_kpa: MatricPotential       # 90th percentile

    # Component predictions
    psi_physics_kpa: MatricPotential           # Physics model
    psi_ml_residual_kpa: float                 # ML correction

    # Status and recommendation
    status: SoilMoistureStatus

    # Confidence metrics
    confidence: float  # 0-1 confidence in prediction
    uncertainty_kpa: float  # Standard deviation

    # Model info
    model_version: str = "1.0.0"

    # Optional θ output (for backward compatibility)
    theta_predicted: Optional[float] = None

    @property
    def is_critical(self) -> bool:
        """Check if moisture status is critical (needs attention)."""
        return self.status in (
            SoilMoistureStatus.MODERATE_STRESS,
            SoilMoistureStatus.SEVERE_STRESS,
            SoilMoistureStatus.WILTING,
        )

    def compute_theta(
        self,
        vg_params: Optional["VanGenuchtenParams"] = None,  # type: ignore
    ) -> "PredictionResult":
        """
        Convert ψ prediction to θ using Van Genuchten.

        Provides backward compatibility for systems expecting
        volumetric water content.

        Args:
            vg_params: Van Genuchten parameters

        Returns:
            Self with theta_predicted populated
        """
        from swpps.physics.van_genuchten import (
            VanGenuchtenParams, water_content_from_potential
        )

        if vg_params is None:
            vg_params = VanGenuchtenParams()

        self.theta_predicted = water_content_from_potential(
            self.psi_predicted_kpa, vg_params
        )

        return self


@dataclass
class IrrigationDecision:
    """Irrigation decision with timing and amount."""
    should_irrigate: bool
    urgency: str  # "immediate", "soon", "scheduled", "none"

    # When to irrigate
    recommended_time: Optional[datetime] = None
    time_until_critical_hours: Optional[float] = None

    # How much
    recommended_amount_mm: Optional[float] = None
    recommended_amount_liters: Optional[float] = None

    # Target
    target_potential_kpa: Optional[float] = None

    # Reasoning
    reason: str = ""

    # Forecast context
    prediction: Optional[PredictionResult] = None
    forecast_horizon_hours: int = 168  # 7 days default


@dataclass
class CanonicalTableRow:
    """
    Single row of the canonical table - unified data structure.

    Contains all data needed for prediction:
    - Site/temporal identifiers
    - Sensor observations (if available)
    - Weather data
    - Soil properties
    - Physics model outputs
    - Engineered features (added during processing)
    """
    # Identifiers
    date: date
    site_id: SiteID

    # Observed matric potential (target variable)
    psi_observed_kpa: Optional[MatricPotential] = None
    observation_quality: int = 0

    # Weather
    precipitation_mm: float = 0.0
    et0_mm: float = 0.0
    temperature_mean_c: float = 20.0
    temperature_min_c: float = 15.0
    temperature_max_c: float = 25.0
    relative_humidity: float = 60.0
    solar_radiation_mj_m2: float = 15.0
    wind_speed_m_s: float = 2.0

    # Soil properties
    sand_percent: float = 40.0
    clay_percent: float = 25.0
    organic_matter_percent: float = 2.0

    # Physics model outputs
    psi_physics_surface_kpa: Optional[MatricPotential] = None
    psi_physics_root_kpa: Optional[MatricPotential] = None
    physics_drainage_mm: float = 0.0
    physics_runoff_mm: float = 0.0
    physics_et_actual_mm: float = 0.0

    # Temporal features (filled during engineering)
    day_of_year: int = 1
    month: int = 1

    # Engineered features (added dynamically)
    features: Dict[str, float] = field(default_factory=dict)
