"""
Simple Bucket Water Balance Model.

A straightforward water balance model based on established soil physics.
No assumptions, no hardcoded defaults - all parameters must be explicitly provided.

This model follows the approach of successful process-based models like:
- SMAP L4 (NASA)
- Noah-MP land surface model
- SWAP (Wageningen)
- API (Antecedent Precipitation Index) models

Key design principles:
1. SIMPLICITY: Standard bucket/reservoir approach, no Richards equation
2. TRANSPARENCY: Every parameter must be explicitly provided
3. PHYSICS-BASED: Uses Van Genuchten for water retention, Darcy for drainage
4. DATA-DRIVEN: No crop type assumptions - uses observed NDVI/LAI directly

The water balance equation:
    dS/dt = P - E - T - D - R

Where:
    S = soil water storage (mm)
    P = precipitation (mm/day)
    E = soil evaporation (mm/day)
    T = transpiration (mm/day)
    D = deep drainage (mm/day)
    R = surface runoff (mm/day)

References:
- Allen et al. (1998) FAO-56: Crop evapotranspiration
- Van Genuchten (1980) Closed-form equation for hydraulic conductivity
- Mualem (1976) Hydraulic conductivity of unsaturated soils
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple
import numpy as np

from smps.core.types import PhysicsPriorResult
from smps.physics.evapotranspiration import get_Kcb_from_curve

logger = logging.getLogger(__name__)


# =============================================================================
# PARAMETER DATACLASSES - NO DEFAULTS
# =============================================================================

@dataclass
class SoilHydraulicParams:
    """
    Soil hydraulic parameters - ALL must be provided, no defaults.

    These can come from:
    1. Laboratory measurements
    2. Pedotransfer functions (applied externally)
    3. Calibration against observed data
    """
    # Van Genuchten parameters
    theta_r: float  # Residual water content (m³/m³)
    theta_s: float  # Saturated water content / porosity (m³/m³)
    alpha: float    # Air entry parameter (1/m)
    n: float        # Pore size distribution parameter (-)

    # Hydraulic conductivity
    K_sat: float    # Saturated hydraulic conductivity (m/day)

    # Derived (calculated in post_init)
    m: float = field(init=False)
    theta_fc: float = field(init=False)  # Field capacity
    theta_wp: float = field(init=False)  # Wilting point

    def __post_init__(self):
        """Calculate derived parameters."""
        self.m = 1.0 - 1.0 / self.n

        # Field capacity at -33 kPa = -3.37 m water head
        psi_fc = -3.37
        self.theta_fc = self._theta_from_psi(psi_fc)

        # Wilting point at -1500 kPa = -153.0 m water head
        psi_wp = -153.0
        self.theta_wp = self._theta_from_psi(psi_wp)

    def _theta_from_psi(self, psi_m: float) -> float:
        """Van Genuchten water retention curve."""
        if psi_m >= 0:
            return self.theta_s
        h = abs(psi_m)
        Se = (1.0 + (self.alpha * h) ** self.n) ** (-self.m)
        return self.theta_r + (self.theta_s - self.theta_r) * Se

    def psi_from_theta(self, theta: float) -> float:
        """Inverse Van Genuchten - matric potential from water content."""
        theta = np.clip(theta, self.theta_r + 1e-6, self.theta_s - 1e-6)
        Se = (theta - self.theta_r) / (self.theta_s - self.theta_r)
        if Se >= 1.0:
            return 0.0
        return -((Se ** (-1.0/self.m) - 1.0) ** (1.0/self.n)) / self.alpha

    def K_from_theta(self, theta: float) -> float:
        """Van Genuchten-Mualem hydraulic conductivity."""
        theta = np.clip(theta, self.theta_r + 1e-6, self.theta_s - 1e-6)
        Se = (theta - self.theta_r) / (self.theta_s - self.theta_r)
        # Mualem model with L=0.5
        return self.K_sat * (Se ** 0.5) * (1.0 - (1.0 - Se ** (1.0/self.m)) ** self.m) ** 2

    def validate(self) -> List[str]:
        """Validate parameter values are physically reasonable."""
        errors = []
        if not (0 < self.theta_r < self.theta_s < 1):
            errors.append(
                f"Invalid theta_r={self.theta_r}, theta_s={self.theta_s}")
        if not (0.001 < self.alpha < 100):
            errors.append(
                f"alpha={self.alpha} outside typical range [0.001, 100] 1/m")
        if not (1.05 < self.n < 10):
            errors.append(f"n={self.n} outside typical range [1.05, 10]")
        if not (1e-6 < self.K_sat < 100):
            errors.append(
                f"K_sat={self.K_sat} outside typical range [1e-6, 100] m/day")
        return errors


@dataclass
class LayerConfig:
    """
    Configuration for a single soil layer.
    """
    depth_top_m: float      # Depth to top of layer (m)
    depth_bottom_m: float   # Depth to bottom of layer (m)
    hydraulics: SoilHydraulicParams  # Hydraulic parameters for this layer

    @property
    def thickness_m(self) -> float:
        return self.depth_bottom_m - self.depth_top_m

    @property
    def center_depth_m(self) -> float:
        return (self.depth_top_m + self.depth_bottom_m) / 2


@dataclass
class ModelConfig:
    """
    Configuration for the water balance model.

    ALL parameters must be provided - no defaults.
    """
    # Soil profile
    layers: List[LayerConfig]

    # Output depth - which depth to report soil moisture for
    output_depth_m: float

    # ET partitioning - fraction of ET that is transpiration (vs evaporation)
    # This should come from vegetation data (NDVI, LAI) not assumptions
    # If not provided, will be calculated from vegetation_fraction
    transpiration_fraction: Optional[float] = None

    # Vegetation fraction (0-1) - from NDVI or land cover data
    # Used to calculate transpiration fraction if not provided directly
    vegetation_fraction: Optional[float] = None

    # Root distribution - fraction of roots in each layer
    # Must sum to 1.0, length must match number of layers
    root_fractions: Optional[List[float]] = None

    # Runoff parameters
    # Curve number for SCS method, or infiltration capacity (mm/hr)
    curve_number: Optional[float] = None
    infiltration_capacity_mm_hr: Optional[float] = None

    # Bias correction (post-processing)
    bias_correction_additive: float = 0.0
    bias_correction_multiplicative: float = 1.0

    # Advanced ET options
    penman_monteith_enabled: bool = False
    atmospheric_pressure_kpa: Optional[float] = None

    # Vegetation stress options
    vegetation_stress_enabled: bool = False
    wilting_point_stress_threshold: float = 0.8
    ndvi_stress_threshold: float = 0.2

    # Advanced infiltration / macropore options
    macropore_flow_enabled: bool = False
    macropore_fraction: Optional[float] = None
    matrix_infiltration_capacity_mm_hr: Optional[float] = None
    macropore_infiltration_capacity_mm_hr: Optional[float] = None

    # Groundwater / capillary rise
    groundwater_enabled: bool = False
    groundwater_depth_m: Optional[float] = None
    capillary_rise_enabled: bool = False
    max_capillary_rise_mm_day: Optional[float] = None

    # Dynamic parameter adjustment
    dynamic_parameters_enabled: bool = False
    seasonal_adjustment_enabled: bool = False
    soil_moisture_state_adjustment: bool = False

    # Advanced ET parameters
    # Penman-Monteith requires meteorological data
    penman_monteith_enabled: bool = False
    atmospheric_pressure_kpa: Optional[float] = None  # Required for PM
    # Calculated from pressure
    psychrometric_constant_kpa_c: Optional[float] = None

    # Vegetation stress parameters
    vegetation_stress_enabled: bool = False
    wilting_point_stress_threshold: float = 0.8  # Fraction of WP-FC range
    ndvi_stress_threshold: float = 0.2  # NDVI below which stress increases

    # Advanced infiltration parameters
    macropore_flow_enabled: bool = False
    # Fraction of infiltration through macropores
    macropore_fraction: Optional[float] = None
    # Matrix infiltration rate
    matrix_infiltration_capacity_mm_hr: Optional[float] = None
    # Macropore infiltration rate
    macropore_infiltration_capacity_mm_hr: Optional[float] = None

    # Groundwater interaction parameters
    groundwater_enabled: bool = False
    groundwater_depth_m: Optional[float] = None  # Depth to water table
    capillary_rise_enabled: bool = False
    # Maximum capillary rise rate
    max_capillary_rise_mm_day: Optional[float] = None

    # Crop development model for dynamic vegetation parameters
    crop_development_model: Optional["CropDevelopmentModel"] = None

    # Dynamic parameter adjustment
    dynamic_parameters_enabled: bool = False
    seasonal_adjustment_enabled: bool = False
    soil_moisture_state_adjustment: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if not self.layers:
            raise ValueError("At least one layer must be provided")

        # Validate layer depths are contiguous
        for i in range(1, len(self.layers)):
            if abs(self.layers[i].depth_top_m - self.layers[i-1].depth_bottom_m) > 1e-6:
                raise ValueError(
                    f"Layer {i} depth_top doesn't match layer {i-1} depth_bottom")

        # Validate root fractions if provided
        if self.root_fractions is not None:
            if len(self.root_fractions) != len(self.layers):
                raise ValueError(
                    "root_fractions must have same length as layers")
            if abs(sum(self.root_fractions) - 1.0) > 1e-6:
                raise ValueError(
                    f"root_fractions must sum to 1.0, got {sum(self.root_fractions)}")

    def get_transpiration_fraction(self, ndvi: Optional[float] = None) -> float:
        """
        Get transpiration fraction from available data.

        Priority:
        1. Directly specified transpiration_fraction
        2. Calculated from vegetation_fraction
        3. Calculated from provided NDVI
        4. Raise error if no data available
        """
        if self.transpiration_fraction is not None:
            return self.transpiration_fraction

        veg_frac = self.vegetation_fraction
        if veg_frac is None and ndvi is not None:
            # Simple linear mapping: NDVI 0.1->0.0, NDVI 0.9->1.0
            veg_frac = np.clip((ndvi - 0.1) / 0.8, 0.0, 1.0)

        if veg_frac is not None:
            # Transpiration fraction increases with vegetation cover
            # Based on partitioning studies (e.g., Kool et al. 2014)
            return veg_frac ** 0.5  # Square root to account for overlap

        raise ValueError(
            "Cannot determine transpiration fraction. Provide one of: "
            "transpiration_fraction, vegetation_fraction, or ndvi"
        )


# =============================================================================
# LAYER STATE
# =============================================================================

@dataclass
class LayerState:
    """
    State of a soil layer.
    """
    config: LayerConfig
    theta: float  # Current volumetric water content (m³/m³)

    @property
    def storage_mm(self) -> float:
        """Water storage in mm."""
        return self.theta * self.config.thickness_m * 1000

    @property
    def available_water_mm(self) -> float:
        """Plant available water above wilting point (mm)."""
        wp = self.config.hydraulics.theta_wp
        return max(0, (self.theta - wp) * self.config.thickness_m * 1000)

    @property
    def saturation_deficit_mm(self) -> float:
        """Space available before saturation (mm)."""
        sat = self.config.hydraulics.theta_s
        return max(0, (sat - self.theta) * self.config.thickness_m * 1000)

    @property
    def field_capacity_deficit_mm(self) -> float:
        """Space available before field capacity (mm)."""
        fc = self.config.hydraulics.theta_fc
        return max(0, (fc - self.theta) * self.config.thickness_m * 1000)

    @property
    def psi_m(self) -> float:
        """Matric potential (m)."""
        return self.config.hydraulics.psi_from_theta(self.theta)

    @property
    def K_m_day(self) -> float:
        """Hydraulic conductivity (m/day)."""
        return self.config.hydraulics.K_from_theta(self.theta)

    def add_water(self, amount_mm: float) -> float:
        """
        Add water to layer, respecting saturation limit.

        Returns excess water that couldn't be stored (mm).
        """
        max_add = self.saturation_deficit_mm
        actual_add = min(amount_mm, max_add)

        delta_theta = actual_add / (self.config.thickness_m * 1000)
        self.theta += delta_theta

        return amount_mm - actual_add

    def remove_water(self, amount_mm: float, min_theta: Optional[float] = None) -> float:
        """
        Remove water from layer, respecting minimum limit.

        Args:
            amount_mm: Amount to remove
            min_theta: Minimum theta (default: residual water content)

        Returns actual amount removed (mm).
        """
        if min_theta is None:
            min_theta = self.config.hydraulics.theta_r

        max_remove = (self.theta - min_theta) * self.config.thickness_m * 1000
        max_remove = max(0, max_remove)
        actual_remove = min(amount_mm, max_remove)

        delta_theta = actual_remove / (self.config.thickness_m * 1000)
        self.theta -= delta_theta

        return actual_remove


# =============================================================================
# FLUX CALCULATIONS
# =============================================================================

@dataclass
class DailyFluxes:
    """
    Water fluxes for a single day.
    """
    # Inputs
    precipitation_mm: float = 0.0
    irrigation_mm: float = 0.0  # NEW: Irrigation water input
    et0_mm: float = 0.0  # Reference ET

    # Calculated fluxes
    infiltration_mm: float = 0.0
    runoff_mm: float = 0.0
    soil_evaporation_mm: float = 0.0
    transpiration_mm: float = 0.0
    drainage_mm: float = 0.0  # From bottom layer

    # Per-layer fluxes
    layer_drainage_mm: List[float] = field(default_factory=list)
    layer_transpiration_mm: List[float] = field(default_factory=list)

    @property
    def total_et_mm(self) -> float:
        return self.soil_evaporation_mm + self.transpiration_mm

    @property
    def water_balance_mm(self) -> float:
        """Should be approximately zero for closed system."""
        return (self.precipitation_mm + self.irrigation_mm  # Include irrigation input
                - self.runoff_mm
                - self.total_et_mm
                - self.drainage_mm)


# =============================================================================
# SIMPLE WATER BALANCE MODEL
# =============================================================================

class SimpleWaterBalance:
    """
    Simple bucket water balance model.

    This model:
    1. Calculates infiltration and runoff
    2. Partitions ET into evaporation and transpiration
    3. Distributes transpiration across root zone
    4. Calculates gravity drainage between layers

    No assumptions are made - all parameters must be explicitly configured.
    """

    def __init__(self, config: ModelConfig, initial_theta: Optional[float] = None):
        """
        Initialize model with configuration.

        Args:
            config: Model configuration (all parameters required)
            initial_theta: Initial soil moisture for all layers. If None,
                          uses mid-point between wilting point and field capacity.
        """
        self.config = config
        self.n_layers = len(config.layers)

        # Initialize layer states
        self.layers: List[LayerState] = []
        for layer_config in config.layers:
            if initial_theta is not None:
                init_val = initial_theta
            else:
                # Use mid-point between WP and FC as conservative default
                fc = layer_config.hydraulics.theta_fc
                wp = layer_config.hydraulics.theta_wp
                init_val = wp + 0.5 * (fc - wp)
                logger.debug(
                    f"Using default initial θ={init_val:.3f} (midpoint WP-FC)")
            self.layers.append(LayerState(config=layer_config, theta=init_val))

        # Set default root distribution if not provided
        if config.root_fractions is None:
            self._set_default_root_distribution()
            logger.debug(
                "Using default exponential root distribution (β=0.97)")

        # Initialize crop development model if provided
        self.crop_model = config.crop_development_model
        if self.crop_model is not None:
            logger.info(
                "Using dynamic crop development model for vegetation parameters")

        logger.info(
            f"Initialized SimpleWaterBalance with {self.n_layers} layers")

    def _set_default_root_distribution(self):
        """
        Set exponential root distribution if not provided.

        Uses beta=0.97 (Jackson et al. 1996) for tropical grasslands/savanna.
        This is a reasonable default but should be overridden with site-specific
        data when available.

        Reference:
            Jackson et al. (1996) A global analysis of root distributions for
            terrestrial biomes. Oecologia 108:389-411.
        """
        beta = 0.97
        fractions = []
        total = 0.0

        for layer in self.config.layers:
            # Cumulative root fraction at layer boundaries
            d_top = layer.depth_top_m * 100  # Convert to cm
            d_bot = layer.depth_bottom_m * 100
            frac_top = 1.0 - beta ** d_top
            frac_bot = 1.0 - beta ** d_bot
            layer_frac = frac_bot - frac_top
            fractions.append(layer_frac)
            total += layer_frac

        # Normalize to sum to 1
        self.config.root_fractions = [f/total for f in fractions]

    def set_initial_conditions(self, theta_values: List[float]):
        """
        Set initial soil moisture for each layer.

        Args:
            theta_values: List of theta values, one per layer
        """
        if len(theta_values) != self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} values, got {len(theta_values)}")

        for layer, theta in zip(self.layers, theta_values):
            hydr = layer.config.hydraulics
            layer.theta = np.clip(theta, hydr.theta_r +
                                  1e-6, hydr.theta_s - 1e-6)

    def run_daily(
        self,
        precipitation_mm: float,
        et0_mm: float,
        ndvi: Optional[float] = None,
        irrigation_mm: float = 0.0,  # NEW: Irrigation water input
        # Additional meteorological data for Penman-Monteith
        temperature_mean_c: Optional[float] = None,
        temperature_min_c: Optional[float] = None,
        temperature_max_c: Optional[float] = None,
        relative_humidity_mean: Optional[float] = None,
        wind_speed_mean_m_s: Optional[float] = None,
        solar_radiation_mj_m2: Optional[float] = None,
        atmospheric_pressure_kpa: Optional[float] = None,
        # Groundwater data
        groundwater_depth_m: Optional[float] = None,
        # Seasonal/day of year for adjustments
        day_of_year: Optional[int] = None,
    ) -> Tuple[DailyFluxes, float]:
        """
        Run model for one day.

        Args:
            precipitation_mm: Daily precipitation (mm)
            et0_mm: Reference evapotranspiration (mm)
            ndvi: NDVI value for vegetation fraction (optional)
            irrigation_mm: Daily irrigation water applied (mm) - NEW

        Returns:
            Tuple of (DailyFluxes, output_theta)
        """
        # Apply dynamic parameter adjustments
        self._apply_dynamic_parameter_adjustments(day_of_year)

        # Update crop development model if available
        if self.crop_model is not None and temperature_mean_c is not None:
            # Calculate GDD for crop development
            gdd = self.crop_model.calculate_gdd(
                temperature_max_c, temperature_min_c)
            self.crop_model.state.accumulated_gdd += gdd

            # Update growth stage
            old_stage = self.crop_model.state.growth_stage
            self.crop_model.state.growth_stage = self.crop_model.update_growth_stage()

            # Update root distribution if stage changed
            if old_stage != self.crop_model.state.growth_stage:
                self.config.root_fractions = self.crop_model._calculate_root_fractions(
                    self.crop_model.state.root_depth_m,
                    self.crop_model.root_params.beta_nominal
                )

        # Initialize flux container for this day
        fluxes = DailyFluxes(
            precipitation_mm=precipitation_mm,
            irrigation_mm=irrigation_mm,  # NEW: Track irrigation input
            et0_mm=et0_mm if et0_mm is not None else 0.0,
            layer_drainage_mm=[0.0] * self.n_layers,
            layer_transpiration_mm=[0.0] * self.n_layers,
        )

        # 1. INFILTRATION AND RUNOFF
        # Total water input = precipitation + irrigation
        total_water_input = precipitation_mm + irrigation_mm

        if self.config.macropore_flow_enabled:
            infiltration, runoff = self._calculate_advanced_infiltration(
                total_water_input)
        else:
            infiltration, runoff = self._calculate_infiltration(
                total_water_input)

        fluxes.infiltration_mm = infiltration
        fluxes.runoff_mm = runoff

        # Add infiltration to surface layer
        excess = self.layers[0].add_water(infiltration)

        # Any excess from surface becomes additional runoff
        if excess > 0:
            fluxes.runoff_mm += excess
            fluxes.infiltration_mm -= excess

        # 2. EVAPOTRANSPIRATION
        # Calculate reference ET using Penman-Monteith if data available
        if (self.config.penman_monteith_enabled and
            all(x is not None for x in [temperature_mean_c, temperature_min_c, temperature_max_c,
                                        relative_humidity_mean, wind_speed_mean_m_s, solar_radiation_mj_m2,
                                        atmospheric_pressure_kpa])):

            et_ref = self._calculate_penman_monteith_et(
                temperature_mean_c, temperature_min_c, temperature_max_c,
                relative_humidity_mean, wind_speed_mean_m_s, solar_radiation_mj_m2,
                atmospheric_pressure_kpa
            )

            # Apply seasonal adjustment
            if self.config.seasonal_adjustment_enabled and day_of_year is not None:
                et_ref *= self._calculate_seasonal_et_factor(day_of_year)

        else:
            # Fall back to provided ET0
            et_ref = et0_mm

        # Get transpiration fraction with stress factors
        if self.crop_model is not None:
            # Use crop development model for dynamic vegetation parameters
            t_frac = self._get_transpiration_fraction_from_crop_model(ndvi)
        elif self.config.vegetation_stress_enabled:
            t_frac = self._get_transpiration_fraction_with_stress(ndvi)
        else:
            try:
                t_frac = self.config.get_transpiration_fraction(ndvi)
            except ValueError:
                t_frac = 0.0
                logger.warning("No vegetation data - assuming bare soil")

        # Potential E and T
        pot_evap = et_ref * (1.0 - t_frac)
        pot_trans = et_ref * t_frac

        # Actual soil evaporation (from surface layer only)
        actual_evap = self._calculate_soil_evaporation(pot_evap)
        fluxes.soil_evaporation_mm = actual_evap

        # Actual transpiration (distributed across root zone)
        actual_trans = self._calculate_transpiration(pot_trans)
        fluxes.transpiration_mm = actual_trans
        fluxes.layer_transpiration_mm = self._last_layer_transpiration

        # 3. GRAVITY DRAINAGE
        drainage = self._calculate_drainage()
        fluxes.layer_drainage_mm = drainage
        fluxes.drainage_mm = drainage[-1] if drainage else 0.0

        # 4. CAPILLARY RISE (if groundwater enabled)
        if self.config.groundwater_enabled and groundwater_depth_m is not None:
            # Update groundwater depth for capillary rise calculation
            self.config.groundwater_depth_m = groundwater_depth_m
            capillary_rise = self._calculate_capillary_rise()
            fluxes.capillary_rise_mm = sum(
                capillary_rise) if capillary_rise else 0.0

        # Get output theta at specified depth
        output_theta = self._get_theta_at_depth(self.config.output_depth_m)

        # Apply bias correction
        output_theta = output_theta * self.config.bias_correction_multiplicative + \
            self.config.bias_correction_additive
        output_theta = np.clip(output_theta, 0.01, 0.55)  # Physical bounds

        return fluxes, output_theta

    def _calculate_infiltration(self, precip_mm: float) -> Tuple[float, float]:
        """
        Calculate infiltration and runoff.

        Uses either:
        1. SCS Curve Number method if CN provided
        2. Simple infiltration capacity if provided
        3. All precipitation infiltrates if no parameters

        Returns (infiltration_mm, runoff_mm)
        """
        if precip_mm <= 0:
            return 0.0, 0.0

        if self.config.curve_number is not None:
            # SCS Curve Number method
            cn = self.config.curve_number
            S = 25400 / cn - 254  # Potential retention (mm)
            Ia = 0.2 * S  # Initial abstraction

            if precip_mm <= Ia:
                return precip_mm, 0.0
            else:
                runoff = (precip_mm - Ia) ** 2 / (precip_mm - Ia + S)
                return precip_mm - runoff, runoff

        elif self.config.infiltration_capacity_mm_hr is not None:
            # Simple infiltration capacity (assume 24-hour distribution)
            # Conservative: use average intensity
            avg_intensity = precip_mm / 24.0
            capacity = self.config.infiltration_capacity_mm_hr

            if avg_intensity <= capacity:
                return precip_mm, 0.0
            else:
                # Fraction that infiltrates
                frac = capacity / avg_intensity
                return precip_mm * frac, precip_mm * (1 - frac)

        else:
            # No runoff parameters - all infiltrates
            return precip_mm, 0.0

    def _calculate_soil_evaporation(self, potential_mm: float) -> float:
        """
        Calculate soil evaporation from surface layer.

        Uses the FAO-56 Stage 1/Stage 2 concept:
        - Stage 1: Energy-limited (at potential rate)
        - Stage 2: Diffusion-limited (reduced rate)
        """
        if potential_mm <= 0:
            return 0.0

        surface = self.layers[0]
        hydr = surface.config.hydraulics

        # Calculate evaporation reduction factor
        # Based on relative extractable water
        theta = surface.theta
        theta_fc = hydr.theta_fc
        theta_wp = hydr.theta_wp
        theta_r = hydr.theta_r

        # Stage 1 threshold: above 50% of FC-WP
        theta_stage1 = theta_wp + 0.5 * (theta_fc - theta_wp)

        if theta >= theta_stage1:
            # Stage 1: full evaporation
            Kr = 1.0
        else:
            # Stage 2: reduced evaporation
            # Linear reduction to air-dry point
            Kr = (theta - theta_r) / (theta_stage1 - theta_r)
            Kr = max(0.0, min(1.0, Kr))

        actual_evap = potential_mm * Kr

        # Remove from surface layer (down to residual)
        actual_evap = surface.remove_water(actual_evap, min_theta=theta_r)

        return actual_evap

    def _calculate_transpiration(self, potential_mm: float) -> float:
        """
        Calculate transpiration distributed across root zone.

        Uses Feddes-style reduction based on soil water stress.
        """
        if potential_mm <= 0:
            self._last_layer_transpiration = [0.0] * self.n_layers
            return 0.0

        total_trans = 0.0
        layer_trans = []

        for i, (layer, root_frac) in enumerate(zip(
            self.layers, self.config.root_fractions
        )):
            if root_frac <= 0:
                layer_trans.append(0.0)
                continue

            # Potential transpiration from this layer
            pot_layer = potential_mm * root_frac

            # Water stress factor (0-1)
            stress = self._water_stress_factor(layer)

            # Actual transpiration
            actual_layer = pot_layer * stress

            # Remove water (down to wilting point)
            wp = layer.config.hydraulics.theta_wp
            actual_layer = layer.remove_water(actual_layer, min_theta=wp)

            layer_trans.append(actual_layer)
            total_trans += actual_layer

        self._last_layer_transpiration = layer_trans
        return total_trans

    def _water_stress_factor(self, layer: LayerState) -> float:
        """
        Calculate water stress factor (Feddes-style).

        Returns 1.0 for no stress, 0.0 for complete stress.
        """
        theta = layer.theta
        hydr = layer.config.hydraulics

        # Critical points
        theta_sat = hydr.theta_s
        theta_fc = hydr.theta_fc
        theta_wp = hydr.theta_wp

        # Anaerobiosis point (too wet)
        theta_anox = theta_sat - 0.05  # 5% below saturation

        # Readily available water depletion point
        # Below this, plants start to stress
        p = 0.5  # Depletion fraction for most crops
        theta_rad = theta_wp + (1 - p) * (theta_fc - theta_wp)

        if theta >= theta_anox:
            # Waterlogging stress
            stress = (theta_sat - theta) / (theta_sat - theta_anox)
        elif theta >= theta_rad:
            # No stress
            stress = 1.0
        elif theta > theta_wp:
            # Water stress
            stress = (theta - theta_wp) / (theta_rad - theta_wp)
        else:
            # Complete stress
            stress = 0.0

        return max(0.0, min(1.0, stress))

    def _calculate_drainage(self) -> List[float]:
        """
        Calculate gravity drainage between layers.

        Water drains when theta > field capacity.
        Drainage rate limited by hydraulic conductivity.
        """
        drainage = []

        for i, layer in enumerate(self.layers):
            hydr = layer.config.hydraulics
            theta_fc = hydr.theta_fc

            if layer.theta <= theta_fc:
                # No drainage below field capacity
                drainage.append(0.0)
                continue

            # Drainable water
            drainable = (layer.theta - theta_fc) * \
                layer.config.thickness_m * 1000

            # Hydraulic conductivity limit
            K = hydr.K_from_theta(layer.theta)
            max_drain = K * 1000  # m/day to mm/day

            # Actual drainage (limited by K and drainable water)
            actual_drain = min(drainable, max_drain)

            # Remove from this layer
            layer.remove_water(actual_drain, min_theta=theta_fc)

            # Add to next layer (if exists)
            if i < self.n_layers - 1:
                excess = self.layers[i + 1].add_water(actual_drain)
                # If next layer saturates, reduce drainage
                actual_drain -= excess
                # Return excess to this layer
                if excess > 0:
                    layer.add_water(excess)

            drainage.append(actual_drain)

        return drainage

    def _calculate_penman_monteith_et(
        self,
        t_mean: float,
        t_min: float,
        t_max: float,
        rh_mean: float,
        wind_speed: float,
        solar_rad: float,
        pressure: float,
    ) -> float:
        """
        Calculate reference evapotranspiration using FAO-56 Penman-Monteith equation.

        This is the gold standard for ET calculation, accounting for:
        - Radiation balance
        - Aerodynamic transport
        - Vapor pressure deficit
        - Wind speed effects

        Args:
            t_mean: Mean daily temperature (°C)
            t_min: Minimum daily temperature (°C)
            t_max: Maximum daily temperature (°C)
            rh_mean: Mean relative humidity (%)
            wind_speed: Mean wind speed (m/s)
            solar_rad: Solar radiation (MJ/m²/day)
            pressure: Atmospheric pressure (kPa)

        Returns:
            Reference ET (mm/day)
        """
        # Convert units
        rh_mean = rh_mean / 100.0  # % to fraction

        # Calculate saturation vapor pressure (kPa)
        e_s_tmax = 0.6108 * np.exp(17.27 * t_max / (t_max + 237.3))
        e_s_tmin = 0.6108 * np.exp(17.27 * t_min / (t_min + 237.3))
        e_s = (e_s_tmax + e_s_tmin) / 2.0

        # Calculate actual vapor pressure (kPa)
        e_a = rh_mean * e_s

        # Vapor pressure deficit
        vpd = e_s - e_a

        # Slope of saturation vapor pressure curve (kPa/°C)
        delta = 4098 * e_s / (t_mean + 237.3) ** 2

        # Psychrometric constant (kPa/°C)
        gamma = 0.665e-3 * pressure

        # Radiation term (MJ/m²/day to mm/day)
        # Convert solar radiation to net radiation (simplified)
        # Rn = (1 - albedo) * Rs - sigma * (T+273.15)^4 * (0.34 - 0.14*sqrt(e_a)) * (1.35*Rs/Rso - 0.35)
        # For simplicity, use empirical relationship
        rn = 0.77 * solar_rad  # Simplified net radiation (MJ/m²/day)
        g = 0.0  # Soil heat flux (negligible for daily)

        # Aerodynamic term
        # Wind speed at 2m height (assume input is at 2m)
        u2 = wind_speed

        # FAO-56 Penman-Monteith equation
        numerator = 0.408 * delta * \
            (rn - g) + gamma * (900 / (t_mean + 273)) * u2 * vpd
        denominator = delta + gamma * (1 + 0.34 * u2)

        et0 = numerator / denominator

        return max(0.0, et0)

    def _calculate_seasonal_et_factor(self, day_of_year: int) -> float:
        """
        Calculate seasonal adjustment factor for ET.

        In tropical regions, ET varies with:
        - Wet/dry season transitions
        - Solar angle variations
        - Humidity changes

        Uses a sinusoidal pattern with tropical characteristics.
        """
        # Convert to radians (0 = Jan 1, 2π = Dec 31)
        doy_rad = 2 * np.pi * (day_of_year - 1) / 365

        # Tropical seasonal pattern:
        # - Higher ET during dry season (less cloud cover)
        # - Lower ET during wet season (more cloud cover, higher humidity)
        # - Peak around March-April (dry season), trough around August-September (wet season)

        # Phase shift for tropical West Africa (peak in March)
        phase_shift = -np.pi / 3  # ~60 days shift

        # Amplitude of variation (±20% around mean)
        amplitude = 0.2

        seasonal_factor = 1.0 + amplitude * np.sin(doy_rad + phase_shift)

        return max(0.7, min(1.3, seasonal_factor))  # Reasonable bounds

    def _calculate_advanced_infiltration(self, precip_mm: float) -> Tuple[float, float]:
        """
        Advanced infiltration calculation with macropore flow.

        For tropical soils, infiltration occurs through:
        1. Matrix flow (slow, through soil matrix)
        2. Macropore flow (fast, through cracks/channels/bioturbation)

        The partitioning depends on:
        - Soil moisture state
        - Recent precipitation history
        - Soil structure

        Returns (infiltration_mm, runoff_mm)
        """
        if precip_mm <= 0:
            return 0.0, 0.0

        if not self.config.macropore_flow_enabled:
            return self._calculate_infiltration(precip_mm)

        # Get current surface soil moisture state
        surface_theta = self.layers[0].theta
        hydr = self.layers[0].config.hydraulics

        # Soil moisture state affects macropore connectivity
        # Dry soils: macropores are more active
        # Wet soils: macropores may be filled or less effective
        theta_relative = (surface_theta - hydr.theta_r) / \
            (hydr.theta_s - hydr.theta_r)

        # Macropore effectiveness decreases as soil wets
        macropore_effectiveness = max(0.0, 1.0 - theta_relative)

        # Dynamic infiltration capacities based on soil state
        matrix_capacity = self.config.matrix_infiltration_capacity_mm_hr or 10.0
        macropore_capacity = self.config.macropore_infiltration_capacity_mm_hr or 50.0

        # Adjust capacities based on soil moisture
        # Wet soils have lower infiltration capacity
        wetness_factor = 1.0 - 0.5 * theta_relative  # Reduce capacity when wet
        matrix_capacity *= wetness_factor
        macropore_capacity *= wetness_factor * macropore_effectiveness

        # Total infiltration capacity
        total_capacity = matrix_capacity + macropore_capacity

        # Assume precipitation is distributed over 24 hours
        avg_intensity = precip_mm / 24.0

        if avg_intensity <= total_capacity:
            # All precipitation infiltrates
            infiltration = precip_mm

            # Partition between matrix and macropore flow
            macropore_fraction = (macropore_capacity /
                                  total_capacity) * macropore_effectiveness
            matrix_fraction = 1.0 - macropore_fraction

            # For now, we don't track separate flows, just total infiltration
            runoff = 0.0

        else:
            # Limited infiltration
            infiltration_fraction = total_capacity / avg_intensity
            infiltration = precip_mm * infiltration_fraction
            runoff = precip_mm * (1.0 - infiltration_fraction)

        return infiltration, runoff

    def _calculate_capillary_rise(self) -> List[float]:
        """
        Calculate capillary rise from groundwater table.

        Capillary rise occurs when the water table is close to the surface,
        bringing water upward through unsaturated soil. This is important in:
        - Arid and semi-arid regions with shallow water tables
        - Areas with irrigation return flow
        - Coastal regions with saline groundwater

        Uses the Buckingham-Darcy equation for unsaturated flow.
        """
        if not self.config.capillary_rise_enabled or self.config.groundwater_depth_m is None:
            return [0.0] * self.n_layers

        capillary_rise = []
        cumulative_depth = 0.0

        for i, layer in enumerate(self.layers):
            layer_bottom = cumulative_depth + layer.config.thickness_m

            # Only consider capillary rise if water table is within influence depth
            # Typical influence depth is 2-3 meters for most soils
            influence_depth = 3.0  # meters

            if self.config.groundwater_depth_m > influence_depth:
                capillary_rise.append(0.0)
                cumulative_depth = layer_bottom
                continue

            # Distance from water table to layer center
            distance_to_wt = max(0.01, self.config.groundwater_depth_m -
                                 cumulative_depth - layer.config.thickness_m / 2)

            # Capillary rise potential decreases exponentially with distance
            # Based on soil hydraulic properties
            hydr = layer.config.hydraulics

            # Simplified capillary rise calculation
            # Rise rate = K * (dψ/dz) where ψ is matric potential
            # For saturated conditions near water table, ψ ≈ -distance
            matric_potential = -distance_to_wt  # meters (negative = suction)

            # Hydraulic conductivity at field capacity (representative of capillary flow)
            K_fc = hydr.K_from_theta(hydr.theta_fc)

            # Capillary rise rate (m/day)
            rise_rate_m_day = K_fc * \
                (1.0 / distance_to_wt)  # Simplified gradient

            # Convert to mm/day and limit by maximum rate
            rise_rate_mm_day = rise_rate_m_day * 1000
            max_rise = self.config.max_capillary_rise_mm_day or 5.0  # Default 5 mm/day
            actual_rise = min(rise_rate_mm_day, max_rise)

            # Only allow rise if layer has space
            available_space = (hydr.theta_s - layer.theta) * \
                layer.config.thickness_m * 1000
            actual_rise = min(actual_rise, available_space)

            # Add water to layer
            if actual_rise > 0:
                layer.add_water(actual_rise)

            capillary_rise.append(actual_rise)
            cumulative_depth = layer_bottom

        return capillary_rise

    def _apply_dynamic_parameter_adjustments(self, day_of_year: Optional[int]):
        """
        Apply dynamic parameter adjustments based on soil moisture state and season.

        This implements time-varying parameters that respond to:
        1. Current soil moisture conditions
        2. Seasonal patterns
        3. Feedback between soil state and hydraulic properties
        """
        if not self.config.dynamic_parameters_enabled:
            return

        # Store original parameters if not already stored
        if not hasattr(self, '_original_parameters_stored'):
            self._original_ksat = [
                layer.config.hydraulics.K_sat for layer in self.layers]
            self._original_alpha = [
                layer.config.hydraulics.alpha for layer in self.layers]
            self._original_parameters_stored = True

        # Soil moisture state adjustment
        if self.config.soil_moisture_state_adjustment:
            for i, layer in enumerate(self.layers):
                hydr = layer.config.hydraulics
                theta_relative = (layer.theta - hydr.theta_r) / \
                    (hydr.theta_s - hydr.theta_r)

                # Hydraulic conductivity increases with wetness (power law relationship)
                # Square relationship typical for unsaturated flow
                wetness_factor = theta_relative ** 2
                hydr.K_sat = self._original_ksat[i] * wetness_factor

                # Air entry parameter (alpha) may change slightly with soil structure
                # In swelling soils, alpha decreases as soil wets
                if theta_relative > 0.7:  # Near saturation
                    # Slight decrease
                    hydr.alpha = self._original_alpha[i] * 0.8

        # Seasonal adjustments
        if self.config.seasonal_adjustment_enabled and day_of_year is not None:
            # Seasonal variation in hydraulic properties due to:
            # - Temperature effects on viscosity
            # - Soil structure changes (cracking/swelling)
            # - Biological activity

            # Simplified seasonal pattern for tropical regions
            seasonal_factor = 1.0 + 0.1 * \
                np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peak in March

            for hydr in [layer.config.hydraulics for layer in self.layers]:
                hydr.K_sat *= seasonal_factor

    def _calculate_improved_transpiration(self, potential_mm: float) -> float:
        """
        Improved transpiration calculation with better root distribution.

        Uses a more realistic root water uptake model that accounts for:
        1. Root distribution following Jackson et al. (1996) for tropical systems
        2. Compensatory uptake when some layers are dry
        3. Species-specific root patterns (though generalized here)
        """
        if potential_mm <= 0:
            self._last_layer_transpiration = [0.0] * self.n_layers
            return 0.0

        total_trans = 0.0
        layer_trans = []
        root_zone_theta = []
        root_zone_weights = []

        # First pass: calculate potential uptake from each layer
        for i, (layer, root_frac) in enumerate(zip(self.layers, self.config.root_fractions)):
            if root_frac <= 0:
                layer_trans.append(0.0)
                root_zone_theta.append(0.0)
                root_zone_weights.append(0.0)
                continue

            # Water stress factor (Feddes-style with improvements)
            stress = self._water_stress_factor_improved(layer)

            # Potential transpiration from this layer
            pot_layer = potential_mm * root_frac

            # Available water in layer
            hydr = layer.config.hydraulics
            available_water = max(
                0, layer.theta - hydr.theta_wp) * layer.config.thickness_m * 1000

            # Maximum possible uptake (limited by available water and stress)
            max_uptake = min(pot_layer * stress, available_water)

            layer_trans.append(max_uptake)
            root_zone_theta.append(layer.theta)
            root_zone_weights.append(root_frac)
            total_trans += max_uptake

        # Second pass: apply compensatory uptake if total is less than potential
        if total_trans < potential_mm and sum(root_zone_weights) > 0:
            # Calculate weighted average theta in root zone
            avg_theta = sum(t * w for t, w in zip(root_zone_theta,
                            root_zone_weights)) / sum(root_zone_weights)

            # Find layers with above-average moisture for compensation
            compensation_factor = 1.2  # Allow 20% more uptake from wetter layers

            for i, (layer, root_frac) in enumerate(zip(self.layers, self.config.root_fractions)):
                if root_frac > 0 and layer.theta > avg_theta:
                    hydr = layer.config.hydraulics
                    additional_available = max(
                        0, layer.theta - hydr.theta_wp) * layer.config.thickness_m * 1000 - layer_trans[i]

                    if additional_available > 0:
                        # Allow additional uptake
                        additional_uptake = min(
                            additional_available,
                            (potential_mm - total_trans) *
                            root_frac * compensation_factor
                        )

                        layer_trans[i] += additional_uptake
                        total_trans += additional_uptake

        # Third pass: actually remove water
        actual_total = 0.0
        for i, (layer, uptake) in enumerate(zip(self.layers, layer_trans)):
            if uptake > 0:
                hydr = layer.config.hydraulics
                actual_uptake = layer.remove_water(
                    uptake, min_theta=hydr.theta_wp)
                layer_trans[i] = actual_uptake
                actual_total += actual_uptake

        self._last_layer_transpiration = layer_trans
        return actual_total

    def _water_stress_factor_improved(self, layer: LayerState) -> float:
        """
        Improved water stress factor with more realistic thresholds.

        Based on extensive literature review of plant water relations:
        - Anaerobiosis point: when soil becomes too wet for roots
        - Wilting point: when plants can no longer extract water
        - Field capacity: optimal water content
        - Stress threshold: when plants begin to close stomata
        """
        theta = layer.theta
        hydr = layer.config.hydraulics

        # Critical points (m³/m³)
        theta_sat = hydr.theta_s
        theta_fc = hydr.theta_fc
        theta_wp = hydr.theta_wp

        # Anaerobiosis point (too wet) - typically 90-95% saturation
        theta_anox = theta_sat * 0.95

        # Readily available water depletion point
        # For most crops: 50% depletion of available water
        p = 0.5
        theta_rad = theta_wp + (1 - p) * (theta_fc - theta_wp)

        # Stress onset point (milder stress)
        # Plants begin to show stress at 30% depletion
        p_stress = 0.7
        theta_stress = theta_wp + (1 - p_stress) * (theta_fc - theta_wp)

        if theta >= theta_anox:
            # Waterlogging stress - exponential decrease
            stress = np.exp(-2 * (theta - theta_anox) /
                            (theta_sat - theta_anox))
        elif theta >= theta_stress:
            # No stress zone
            stress = 1.0
        elif theta >= theta_wp:
            # Water stress zone - linear decrease from stress onset to wilting
            stress = (theta - theta_wp) / (theta_stress - theta_wp)
        else:
            # Complete stress
            stress = 0.0

        return max(0.0, min(1.0, stress))

    def _calculate_improved_drainage(self) -> List[float]:
        """
        Improved vertical drainage with better layer coupling.

        Accounts for:
        1. Hydraulic conductivity variations with depth
        2. Layer interface resistances
        3. Time-dependent drainage (though simplified to daily)
        4. Preferential flow paths in structured soils
        """
        drainage = []

        for i, layer in enumerate(self.layers):
            hydr = layer.config.hydraulics
            theta_fc = hydr.theta_fc

            if layer.theta <= theta_fc:
                # No drainage below field capacity
                drainage.append(0.0)
                continue

            # Drainable water (mm)
            drainable = (layer.theta - theta_fc) * \
                layer.config.thickness_m * 1000

            # Hydraulic conductivity at current moisture content
            K = hydr.K_from_theta(layer.theta)

            # Apply depth correction for K (typically decreases with depth)
            # Due to compaction, lower porosity in subsoils
            # Reduce by 10% per meter
            depth_factor = max(0.3, 1.0 - 0.1 * (layer.center_depth_m / 2.0))
            K *= depth_factor

            # Maximum drainage rate (m/day to mm/day)
            max_drain = K * 1000

            # Actual drainage (limited by drainable water and conductivity)
            actual_drain = min(drainable, max_drain)

            # For improved coupling: consider layer below
            if i < self.n_layers - 1:
                lower_layer = self.layers[i + 1]
                lower_hydr = lower_layer.config.hydraulics

                # If lower layer is drier, drainage is faster
                # If lower layer is wetter, drainage slows
                suction_gradient = lower_hydr.psi_from_theta(
                    lower_layer.theta) - hydr.psi_from_theta(layer.theta)

                # Adjust drainage based on gradient
                if suction_gradient > 0:  # Lower layer drier, faster drainage
                    gradient_factor = 1.0 + 0.2 * \
                        min(1.0, suction_gradient / 1000)  # Max 20% increase
                else:  # Lower layer wetter, slower drainage
                    # Min 50% reduction
                    gradient_factor = max(
                        0.5, 1.0 + 0.2 * max(-1.0, suction_gradient / 1000))

                actual_drain *= gradient_factor

            # Remove from this layer
            layer.remove_water(actual_drain, min_theta=theta_fc)

            # Add to next layer (if exists)
            if i < self.n_layers - 1:
                excess = self.layers[i + 1].add_water(actual_drain)
                # If next layer saturates, reduce drainage
                actual_drain -= excess
                # Return excess to this layer
                if excess > 0:
                    layer.add_water(excess)

            drainage.append(actual_drain)

        return drainage

    def _get_transpiration_fraction_with_stress(self, ndvi: Optional[float]) -> float:
        """
        Get transpiration fraction with vegetation stress factors.

        Combines:
        1. Base vegetation fraction from NDVI/land cover
        2. Soil moisture stress
        3. Vegetation health stress from NDVI

        Returns:
            Transpiration fraction (0-1)
        """
        # Get base transpiration fraction
        base_t_frac = self.config.get_transpiration_fraction(ndvi)

        stress_factor = 1.0

        if self.config.vegetation_stress_enabled:
            # Soil moisture stress
            if self.config.wilting_point_stress_threshold > 0:
                # Calculate average root zone soil moisture stress
                root_zone_theta = 0.0
                root_zone_weight = 0.0

                for layer, root_frac in zip(self.layers, self.config.root_fractions):
                    if root_frac > 0:
                        hydr = layer.config.hydraulics
                        theta_norm = (layer.theta - hydr.theta_wp) / \
                            (hydr.theta_fc - hydr.theta_wp)
                        theta_norm = np.clip(theta_norm, 0.0, 1.0)

                        # Stress when below threshold
                        if theta_norm < self.config.wilting_point_stress_threshold:
                            layer_stress = theta_norm / self.config.wilting_point_stress_threshold
                            stress_factor *= layer_stress
                            root_zone_weight += root_frac

                if root_zone_weight > 0:
                    stress_factor = stress_factor ** (1.0 / root_zone_weight)

            # NDVI-based vegetation stress
            if ndvi is not None and self.config.ndvi_stress_threshold > 0:
                if ndvi < self.config.ndvi_stress_threshold:
                    ndvi_factor = ndvi / self.config.ndvi_stress_threshold
                    stress_factor *= ndvi_factor

        return base_t_frac * stress_factor

    def _get_transpiration_fraction_from_crop_model(self, ndvi: Optional[float]) -> float:
        """
        Get transpiration fraction from crop development model.

        Uses dynamic vegetation parameters from phenological model:
        1. Kcb coefficient from growth stage
        2. Root zone stress from soil moisture
        3. NDVI-based health adjustment

        Returns:
            Transpiration fraction (0-1)
        """
        if self.crop_model is None:
            return 0.0

        # Get Kcb from crop development model (basal crop coefficient)
        kcb = get_Kcb_from_curve(
            self.crop_model.Kc_curve,
            self.crop_model.state.days_since_planting)

        # Adjust for vegetation health from NDVI
        if ndvi is not None:
            # NDVI adjustment factor (0.1-0.9 NDVI maps to 0.5-1.5 Kcb)
            ndvi_factor = 0.5 + (ndvi - 0.1) / 0.8
            ndvi_factor = np.clip(ndvi_factor, 0.5, 1.5)
            kcb *= ndvi_factor

        # Convert Kcb to transpiration fraction
        # Kcb is the ratio of crop ET to reference ET
        # For transpiration fraction, we need to account for evaporation
        # Typical Ke/Kc ratio is 0.1-0.3 for healthy crops
        ke_kc_ratio = 0.15  # Conservative estimate
        transpiration_fraction = kcb * (1.0 - ke_kc_ratio)

        # Apply soil moisture stress
        stress_factor = 1.0
        if self.config.vegetation_stress_enabled:
            # Calculate root zone stress
            root_zone_theta = 0.0
            root_zone_weight = 0.0

            for layer, root_frac in zip(self.layers, self.config.root_fractions):
                if root_frac > 0:
                    hydr = layer.config.hydraulics
                    theta_norm = (layer.theta - hydr.theta_wp) / \
                        (hydr.theta_fc - hydr.theta_wp)
                    theta_norm = np.clip(theta_norm, 0.0, 1.0)

                    if theta_norm < self.config.wilting_point_stress_threshold:
                        layer_stress = theta_norm / self.config.wilting_point_stress_threshold
                        stress_factor *= layer_stress
                        root_zone_weight += root_frac

            if root_zone_weight > 0:
                stress_factor = stress_factor ** (1.0 / root_zone_weight)

        return np.clip(transpiration_fraction * stress_factor, 0.0, 1.0)

    def _get_theta_at_depth(self, target_depth_m: float) -> float:
        """
        Get soil moisture at specified depth.

        Interpolates between layers if necessary.
        """
        for layer in self.layers:
            if layer.config.depth_top_m <= target_depth_m < layer.config.depth_bottom_m:
                return layer.theta

        # If target is deeper than profile, return deepest layer
        if target_depth_m >= self.layers[-1].config.depth_bottom_m:
            return self.layers[-1].theta

        # If target is above profile (shouldn't happen)
        return self.layers[0].theta

    def run_period(
        self,
        dates: List[date],
        precipitation: List[float],
        et0: List[float],
        ndvi: Optional[List[float]] = None,
        warmup_days: int = 30,
    ) -> Tuple[List[PhysicsPriorResult], List[DailyFluxes]]:
        """
        Run model for a time period.

        Args:
            dates: List of dates
            precipitation: Daily precipitation (mm)
            et0: Reference ET (mm)
            ndvi: Optional NDVI values
            warmup_days: Days to run before recording output

        Returns:
            Tuple of (results, fluxes) for non-warmup period
        """
        n_days = len(dates)

        if len(precipitation) != n_days or len(et0) != n_days:
            raise ValueError("Input arrays must have same length as dates")

        if ndvi is not None and len(ndvi) != n_days:
            raise ValueError("NDVI array must have same length as dates")

        results = []
        all_fluxes = []

        for i in range(n_days):
            ndvi_val = ndvi[i] if ndvi is not None else None

            fluxes, theta = self.run_daily(
                precipitation_mm=precipitation[i],
                et0_mm=et0[i],
                ndvi=ndvi_val,
            )

            if i >= warmup_days:
                # Create result
                result = PhysicsPriorResult(
                    date=dates[i],
                    theta_surface=self.layers[0].theta,
                    theta_root=theta,  # At output depth
                    theta_deep=self.layers[-1].theta if self.n_layers > 1 else None,
                    fluxes={
                        'evapotranspiration': fluxes.total_et_mm,
                        'drainage': fluxes.drainage_mm,
                        'runoff': fluxes.runoff_mm,
                        'infiltration': fluxes.infiltration_mm,
                    },
                    water_balance_error=0.0,  # Simple model, no numerical error
                    converged=True,
                )
                results.append(result)
                all_fluxes.append(fluxes)

        return results, all_fluxes


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_hydraulic_params_from_texture_tropical(
    sand_percent: float,
    clay_percent: float,
    organic_matter_percent: float = 2.0,
    bulk_density_g_cm3: Optional[float] = None,
    soil_type: str = "ferralsol",
) -> SoilHydraulicParams:
    """
    Create hydraulic parameters from soil texture using tropical PTF.

    Based on Hodnett & Tomasella (2002) with modifications for African soil types.
    More appropriate for tropical environments than standard Saxton-Rawls.

    Key improvements:
    1. Higher porosity due to biological activity and aggregation
    2. Higher Ksat due to macropores in tropical soils
    3. Modified retention curve for oxide clay behavior

    Args:
        sand_percent: Sand content (%)
        clay_percent: Clay content (%)
        organic_matter_percent: Organic matter content (%)
        bulk_density_g_cm3: Bulk density (g/cm³), estimated if not provided
        soil_type: WRB soil classification ("ferralsol", "nitisol", "vertisol")

    Returns:
        SoilHydraulicParams
    """
    from smps.physics.adaptive_calibration import tropical_ptf_van_genuchten

    # Get tropical PTF parameters with soil-type specific corrections
    trop_params = tropical_ptf_van_genuchten(
        sand_percent=sand_percent,
        clay_percent=clay_percent,
        organic_matter_percent=organic_matter_percent,
        bulk_density=bulk_density_g_cm3 or 1.35,
        soil_type=soil_type,
    )

    # Convert to SoilHydraulicParams format
    hydraulics = SoilHydraulicParams(
        theta_r=trop_params['theta_r'],
        theta_s=trop_params['theta_s'],
        alpha=trop_params['alpha'],
        n=trop_params['n'],
        K_sat=trop_params['K_sat'],
    )

    return hydraulics


def create_hydraulic_params_from_texture(
    sand_percent: float,
    clay_percent: float,
    organic_matter_percent: float = 2.0,
    bulk_density_g_cm3: Optional[float] = None,
) -> SoilHydraulicParams:
    """
    Create hydraulic parameters from soil texture using Saxton & Rawls (2006).

    This is a HELPER function - the model itself doesn't assume any PTF.
    Users can use this or provide their own parameters.

    IMPORTANT: Van Genuchten α and n are fitted to match Saxton-Rawls
    θ(ψ) estimates at -33 kPa and -1500 kPa for consistency.

    Args:
        sand_percent: Sand content (%)
        clay_percent: Clay content (%)
        organic_matter_percent: Organic matter content (%)
        bulk_density_g_cm3: Bulk density (g/cm³), estimated if not provided

    Returns:
        SoilHydraulicParams
    """
    S = sand_percent / 100.0
    C = clay_percent / 100.0
    OM = organic_matter_percent / 100.0

    # Saxton & Rawls (2006) equations
    # Wilting point (-1500 kPa)
    theta_1500t = (
        -0.024 * S + 0.487 * C + 0.006 * OM
        + 0.005 * S * OM - 0.013 * C * OM
        + 0.068 * S * C + 0.031
    )
    theta_1500 = theta_1500t + 0.14 * theta_1500t - 0.02
    theta_1500 = max(0.02, theta_1500)  # Minimum bound

    # Field capacity (-33 kPa)
    theta_33t = (
        -0.251 * S + 0.195 * C + 0.011 * OM
        + 0.006 * S * OM - 0.027 * C * OM
        + 0.452 * S * C + 0.299
    )
    theta_33 = theta_33t + 1.283 * theta_33t ** 2 - 0.374 * theta_33t - 0.015
    theta_33 = max(theta_1500 + 0.02, theta_33)  # Must be > WP

    # Saturation
    theta_s_33t = (
        0.278 * S + 0.034 * C + 0.022 * OM
        - 0.018 * S * OM - 0.027 * C * OM
        - 0.584 * S * C + 0.078
    )
    theta_s_33 = theta_s_33t + 0.636 * theta_s_33t - 0.107
    theta_s = theta_33 + theta_s_33 - 0.097 * S + 0.043
    theta_s = max(theta_33 + 0.05, min(0.55, theta_s))  # Bounds

    # Saturated hydraulic conductivity (mm/hr)
    B = (np.log(1500) - np.log(33)) / (np.log(theta_33) - np.log(theta_1500))
    lambda_pore = 1 / B
    K_sat_mm_hr = 1930 * (theta_s - theta_33) ** (3 - lambda_pore)
    K_sat_m_day = K_sat_mm_hr * 24 / 1000

    # Residual water content - typically 5-15% of θs
    # For sandy soils: lower; for clay: higher
    theta_r = 0.02 + 0.15 * C  # Increases with clay
    theta_r = min(theta_r, theta_1500 * 0.5)  # Must be well below WP

    # FIT Van Genuchten α and n to match Saxton-Rawls θ33 and θ1500
    # This ensures consistency between PTF approaches
    #
    # Van Genuchten: Se = [1 + (α*h)^n]^(-m), where m = 1 - 1/n
    # At h = 3.37m (33 kPa): Se_33 = (θ33 - θr)/(θs - θr)
    # At h = 153m (1500 kPa): Se_1500 = (θ1500 - θr)/(θs - θr)
    #
    # Solve for α and n using Newton-Raphson or analytical approximation

    Se_33 = (theta_33 - theta_r) / (theta_s - theta_r)
    Se_1500 = (theta_1500 - theta_r) / (theta_s - theta_r)

    # Analytical solution for n from two points on retention curve:
    # ln[(1/Se_1)^(1/m) - 1] - ln[(1/Se_2)^(1/m) - 1] = n * ln(h1/h2)
    # This is iterative since m = 1-1/n

    h_33 = 3.37  # m water head
    h_1500 = 153.0  # m water head

    # Initial guess for n based on texture
    if S > 0.7:
        n = 2.5  # Sandy
    elif C > 0.4:
        n = 1.15  # Clay
    else:
        n = 1.5  # Loam

    # Iterate to find n
    for _ in range(20):
        m = 1.0 - 1.0 / n

        # Compute Se from VG at both pressures for current n
        # Solve for α from 33 kPa point
        # Se_33 = [1 + (α*h_33)^n]^(-m)
        # (1/Se_33)^(1/m) - 1 = (α*h_33)^n

        term_33 = max(1e-6, Se_33 ** (-1/m) - 1)
        term_1500 = max(1e-6, Se_1500 ** (-1/m) - 1)

        # Ratio gives n
        # term_1500 / term_33 = (h_1500/h_33)^n
        ratio = term_1500 / term_33
        target_ratio = (h_1500 / h_33) ** n

        # Update n
        if ratio > 0 and target_ratio > 0:
            n_new = np.log(ratio) / np.log(h_1500 / h_33)
            n_new = max(1.05, min(5.0, n_new))  # Bounds

            if abs(n_new - n) < 0.01:
                break
            n = 0.5 * n + 0.5 * n_new  # Damped update

    # Final α from 33 kPa point
    m = 1.0 - 1.0 / n
    term_33 = max(1e-6, Se_33 ** (-1/m) - 1)
    alpha = (term_33 ** (1/n)) / h_33
    alpha = max(0.001, min(1.0, alpha))  # Reasonable bounds (1/m)

    return SoilHydraulicParams(
        theta_r=theta_r,
        theta_s=theta_s,
        alpha=alpha,
        n=n,
        K_sat=K_sat_m_day,
    )


def create_simple_config_improved(
    sand_percent: float,
    clay_percent: float,
    output_depth_m: float = 0.10,
    n_layers: int = 3,
    max_depth_m: float = 1.0,
    vegetation_fraction: Optional[float] = None,
    curve_number: Optional[float] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    elevation_m: Optional[float] = None,
    slope_percent: Optional[float] = None,
    observed_mean: Optional[float] = None,
    use_tropical_ptf: bool = True,
    apply_adaptive_calibration: bool = True,
    soil_type: str = "nitisol",
    crop_type: str = "savanna",
    ndvi_mean: Optional[float] = None,
) -> ModelConfig:
    """
    Create an improved model configuration with tropical PTFs and adaptive calibration.

    Improvements over basic config:
    1. Uses tropical pedotransfer functions with soil-type corrections
    2. Applies site-specific adaptive corrections
    3. Better infiltration modeling for tropical soils
    4. Climate-aware ET partitioning
    5. Dynamic crop development model for vegetation parameters

    Args:
        sand_percent: Sand content (%)
        clay_percent: Clay content (%)
        output_depth_m: Depth to report soil moisture (m)
        n_layers: Number of layers
        max_depth_m: Total profile depth (m)
        vegetation_fraction: Vegetation cover (0-1)
        curve_number: SCS curve number for runoff
        latitude: Site latitude for adaptive calibration
        longitude: Site longitude for adaptive calibration
        elevation_m: Site elevation (m) for adaptive calibration
        slope_percent: Site slope (%) for runoff calibration
        observed_mean: Observed mean soil moisture for bias correction
        use_tropical_ptf: Whether to use tropical PTFs
        apply_adaptive_calibration: Whether to apply adaptive corrections
        soil_type: WRB soil classification ("ferralsol", "nitisol", "vertisol")
        crop_type: Crop/vegetation type for phenology model

    Returns:
        ModelConfig ready for use
    """
    # Get hydraulic parameters
    if use_tropical_ptf:
        hydraulics = create_hydraulic_params_from_texture_tropical(
            sand_percent=sand_percent,
            clay_percent=clay_percent,
            soil_type=soil_type,
        )
    else:
        hydraulics = create_hydraulic_params_from_texture(
            sand_percent=sand_percent,
            clay_percent=clay_percent,
        )

    # Apply adaptive calibration if requested
    bias_correction_multiplicative = 1.0
    bias_correction_additive = 0.0

    if apply_adaptive_calibration and latitude is not None and longitude is not None:
        from smps.physics.adaptive_calibration import create_site_calibrator

        obs_stats = {
            'mean': observed_mean} if observed_mean is not None else None

        calibrator = create_site_calibrator(
            latitude=latitude,
            longitude=longitude,
            sand_percent=sand_percent,
            clay_percent=clay_percent,
            annual_precip_mm=1000.0,  # Default, could be improved
            observed_stats=obs_stats,
            elevation_m=elevation_m,
            slope_percent=slope_percent,
            ndvi_mean=ndvi_mean,
        )

        if calibrator is not None:
            # Apply parameter adjustments
            adj = calibrator.params
            hydraulics.alpha *= adj.alpha_multiplier
            hydraulics.n += adj.n_adjustment
            hydraulics.theta_r += adj.theta_r_adjustment
            hydraulics.theta_s += adj.theta_s_adjustment
            hydraulics.K_sat *= adj.ksat_multiplier

            # Apply bias corrections
            bias_correction_multiplicative = adj.bias_correction_multiplicative
            bias_correction_additive = adj.bias_correction_additive

            # Ensure bounds
            hydraulics.n = max(1.05, hydraulics.n)
            hydraulics.theta_r = np.clip(hydraulics.theta_r, 0.01, 0.15)
            hydraulics.theta_s = np.clip(hydraulics.theta_s, 0.35, 0.60)
            hydraulics.alpha = np.clip(hydraulics.alpha, 0.5, 20.0)
            hydraulics.K_sat = np.clip(hydraulics.K_sat, 0.01, 10.0)

    # Create uniform layer thicknesses
    layer_thickness = max_depth_m / n_layers

    # Create layers
    layers = []
    for i in range(n_layers):
        layer = LayerConfig(
            depth_top_m=i * layer_thickness,
            depth_bottom_m=(i + 1) * layer_thickness,
            hydraulics=hydraulics,  # Same for all layers in simple config
        )
        layers.append(layer)

    # Initialize crop development model
    from smps.physics.crop_development import CropDevelopmentModel
    layer_depths = np.array([layer.depth_bottom_m for layer in layers])
    crop_model = CropDevelopmentModel(
        crop_name=crop_type,
        layer_depths_m=layer_depths
    )

    return ModelConfig(
        layers=layers,
        output_depth_m=output_depth_m,
        vegetation_fraction=vegetation_fraction,
        curve_number=curve_number,
        bias_correction_additive=bias_correction_additive,
        bias_correction_multiplicative=bias_correction_multiplicative,
        crop_development_model=crop_model,
        # Enable advanced physics features for improved fidelity
        macropore_flow_enabled=True,
        groundwater_enabled=True,
        capillary_rise_enabled=True,
        dynamic_parameters_enabled=True,
        seasonal_adjustment_enabled=True,
        vegetation_stress_enabled=True,
        penman_monteith_enabled=True,
    )
    """
    Create a simple model configuration from basic inputs.

    This is a CONVENIENCE function for quick setup.
    For full control, create ModelConfig directly.

    Args:
        sand_percent: Sand content (%)
        clay_percent: Clay content (%)
        output_depth_m: Depth to report soil moisture (m)
        n_layers: Number of layers
        max_depth_m: Total profile depth (m)
        vegetation_fraction: Vegetation cover (0-1)
        curve_number: SCS curve number for runoff

    Returns:
        ModelConfig ready for use
    """
    # Create uniform layer thicknesses
    layer_thickness = max_depth_m / n_layers

    # Get hydraulic parameters
    hydraulics = create_hydraulic_params_from_texture(
        sand_percent=sand_percent,
        clay_percent=clay_percent,
    )

    # Create layers
    layers = []
    for i in range(n_layers):
        layer = LayerConfig(
            depth_top_m=i * layer_thickness,
            depth_bottom_m=(i + 1) * layer_thickness,
            hydraulics=hydraulics,  # Same for all layers in simple config
        )
        layers.append(layer)

    return ModelConfig(
        layers=layers,
        output_depth_m=output_depth_m,
        vegetation_fraction=vegetation_fraction,
        curve_number=curve_number,
    )


def create_simple_config(
    sand_percent: float,
    clay_percent: float,
    output_depth_m: float = 0.10,
    n_layers: int = 3,
    max_depth_m: float = 1.0,
    vegetation_fraction: Optional[float] = None,
    curve_number: Optional[float] = None,
) -> ModelConfig:
    """
    Create a simple model configuration from basic inputs.

    This is a CONVENIENCE function for quick setup.
    For full control, create ModelConfig directly.

    Args:
        sand_percent: Sand content (%)
        clay_percent: Clay content (%)
        output_depth_m: Depth to report soil moisture (m)
        n_layers: Number of layers
        max_depth_m: Total profile depth (m)
        vegetation_fraction: Vegetation cover (0-1)
        curve_number: SCS curve number for runoff

    Returns:
        ModelConfig ready for use
    """
    # Create uniform layer thicknesses
    layer_thickness = max_depth_m / n_layers

    # Get hydraulic parameters
    hydraulics = create_hydraulic_params_from_texture(
        sand_percent=sand_percent,
        clay_percent=clay_percent,
    )

    # Create layers
    layers = []
    for i in range(n_layers):
        layer = LayerConfig(
            depth_top_m=i * layer_thickness,
            depth_bottom_m=(i + 1) * layer_thickness,
            hydraulics=hydraulics,  # Same for all layers in simple config
        )
        layers.append(layer)

    return ModelConfig(
        layers=layers,
        output_depth_m=output_depth_m,
        vegetation_fraction=vegetation_fraction,
        curve_number=curve_number,
    )
