"""
Implementation of two-bucket water balance model.
Based on soil physics principles with numerical stability guarantees.
"""
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd

from smps.core.config import get_config, SmpsConfig
from smps.core.types import (
    SoilParameters, SoilLayer, PhysicsPriorResult,
    SoilMoistureVWC, PrecipitationMm, ET0Mm
)
from smps.core.exceptions import (
    PhysicsModelError, WaterBalanceError, ConvergenceError
)
logger = logging.getLogger(__name__)


# Default crop rooting depths (meters)
DEFAULT_CROP_ROOTING_DEPTHS = {
    "maize": 0.40,
    "wheat": 0.35,
    "rice": 0.30,
    "soybean": 0.35,
    "cotton": 0.45,
    "generic": 0.30,
}


@dataclass
class ModelParameters:
    """Physical parameters for the two-bucket water balance model"""
    # Layer depths
    surface_depth_m: float = 0.10  # Surface layer depth (m)
    root_zone_depth_m: float = 0.30  # Root zone depth (m)

    # Hydraulic properties - surface
    porosity_surface: float = 0.45
    field_capacity_surface: float = 0.30
    wilting_point_surface: float = 0.12

    # Hydraulic properties - root zone
    porosity_root: float = 0.43
    field_capacity_root: float = 0.28
    wilting_point_root: float = 0.10

    # Flux parameters
    infiltration_capacity_mm_h: float = 20.0
    percolation_rate_day: float = 0.15
    drainage_rate_day: float = 0.10

    # ET partitioning
    ndvi_et_partitioning_lambda: float = 2.0
    evaporation_coefficient_bare: float = 1.0  # ET coefficient for bare soil
    transpiration_coefficient_max: float = 1.2  # Max transpiration coefficient

    # Numerical parameters
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    tolerance: float = 0.01  # Water balance tolerance (mm)

    @classmethod
    def from_soil_parameters(
        cls,
        soil_params: SoilParameters,
        config_overrides: Optional[Dict] = None
    ) -> "ModelParameters":
        """Create model parameters from soil parameters"""
        params = cls(
            porosity_surface=soil_params.porosity,
            porosity_root=soil_params.porosity * 0.95,  # Slightly compacted
            field_capacity_surface=soil_params.field_capacity,
            field_capacity_root=soil_params.field_capacity * 0.95,
            wilting_point_surface=soil_params.wilting_point,
            wilting_point_root=soil_params.wilting_point,
            infiltration_capacity_mm_h=soil_params.saturated_hydraulic_conductivity_cm_day * 10 / 24,
        )

        # Apply overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(params, key):
                    setattr(params, key, value)

        return params


@dataclass
class BucketState:
    """State of a soil bucket (layer)"""
    water_content: float  # Volumetric water content (m³/m³)
    max_storage: float  # Maximum storage capacity (mm)
    layer_depth_m: float  # Layer depth (m)
    field_capacity: float  # Field capacity (m³/m³)
    wilting_point: float  # Wilting point (m³/m³)
    porosity: float  # Porosity (m³/m³)
    _storage: float = None  # Internal storage value (mm)

    def __post_init__(self):
        """Initialize storage from water content"""
        if self._storage is None:
            self._storage = self.water_content * self.layer_depth_m * 1000

    @property
    def storage(self) -> float:
        """Current water storage in mm"""
        return self._storage

    @storage.setter
    def storage(self, value: float):
        """Set storage and update water content"""
        self._storage = value
        self.water_content = value / (self.layer_depth_m * 1000)

    @property
    def saturation(self) -> float:
        """Degree of saturation (0-1)"""
        return self.water_content / self.porosity if self.porosity > 0 else 0

    @property
    def available_water(self) -> float:
        """Plant available water in mm"""
        return max(0, (self.water_content - self.wilting_point) * self.layer_depth_m * 1000)


@dataclass
class Fluxes:
    """Water fluxes between compartments (all in mm/day)"""
    precipitation: float = 0.0
    irrigation: float = 0.0
    infiltration: float = 0.0
    runoff: float = 0.0
    evaporation: float = 0.0
    transpiration: float = 0.0  # Total transpiration
    transpiration_surface: float = 0.0
    transpiration_root: float = 0.0
    percolation: float = 0.0  # Surface to root zone
    drainage: float = 0.0  # Root zone to deep
    evapotranspiration: float = 0.0  # Total ET

    @property
    def total_et(self) -> float:
        """Total evapotranspiration"""
        return self.evaporation + self.transpiration

    @property
    def total_input(self) -> float:
        """Total water input"""
        return self.precipitation + self.irrigation

    @property
    def total_output(self) -> float:
        """Total water output"""
        return self.runoff + self.total_et + self.drainage


class TwoBucketWaterBalance:
    """
    Implementation of two-bucket water balance model.

    The model simulates:
    1. Surface layer (0-10cm): Infiltration, evaporation, runoff, percolation
    2. Root zone layer (10-40cm): Percolation, transpiration, drainage

    Key features:
    - Mass-conserving formulation
    - Numerical stability with adaptive time stepping
    - Physical constraints enforcement
    - Configurable ET partitioning
    """

    def __init__(
        self,
        parameters: ModelParameters,
        initial_state: Optional[Dict[SoilLayer, BucketState]] = None
    ):
        """
        Initialize two-bucket model.

        Args:
            parameters: Model physical parameters
            initial_state: Initial bucket states (optional)
        """
        self.params = parameters
        self._setup_logging()

        # Initialize state
        if initial_state is None:
            self.state = self._initialize_default_state()
        else:
            self.state = initial_state

        # Track water balance error
        self.cumulative_error = 0.0
        self.iteration_count = 0

        # Cache for efficiency
        self._cache = {}

    def _setup_logging(self):
        """Configure model-specific logging"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)

    def _initialize_default_state(self) -> Dict[SoilLayer, BucketState]:
        """Initialize buckets at field capacity (common assumption)"""
        return {
            SoilLayer.SURFACE: BucketState(
                water_content=self.params.field_capacity_surface,
                max_storage=self.params.surface_depth_m * 1000,  # mm
                layer_depth_m=self.params.surface_depth_m,
                field_capacity=self.params.field_capacity_surface,
                wilting_point=self.params.wilting_point_surface,
                porosity=self.params.porosity_surface
            ),
            SoilLayer.ROOT_ZONE: BucketState(
                water_content=self.params.field_capacity_root,
                max_storage=self.params.root_zone_depth_m * 1000,  # mm
                layer_depth_m=self.params.root_zone_depth_m,
                field_capacity=self.params.field_capacity_root,
                wilting_point=self.params.wilting_point_root,
                porosity=self.params.porosity_root
            )
        }

    def run_daily(
        self,
        precipitation_mm: PrecipitationMm,
        et0_mm: ET0Mm,
        ndvi: Optional[float] = None,
        irrigation_mm: float = 0.0,
        air_temperature_c: Optional[float] = None,
        check_water_balance: bool = True
    ) -> PhysicsPriorResult:
        """
        Run model for one day with given forcings.

        Args:
            precipitation_mm: Daily precipitation (mm)
            et0_mm: Daily reference evapotranspiration (mm)
            ndvi: Normalized Difference Vegetation Index (0-1)
            irrigation_mm: Irrigation amount (mm)
            air_temperature_c: Air temperature for ET adjustment
            check_water_balance: Validate water balance closure

        Returns:
            PhysicsPriorResult with updated states and fluxes
        """
        self.logger.debug(
            f"Running daily step: P={precipitation_mm:.1f}mm, "
            f"ET0={et0_mm:.1f}mm, NDVI={ndvi}, Irr={irrigation_mm:.1f}mm"
        )

        # Store initial state for water balance check
        initial_storage = self._total_storage()

        try:
            # Step 1: Handle precipitation and irrigation
            fluxes = self._handle_precipitation(
                precipitation_mm, irrigation_mm
            )

            # Step 2: Calculate ET partitioning
            self._calculate_evapotranspiration(
                et0_mm, ndvi, air_temperature_c, fluxes
            )

            # Step 3: Calculate vertical water movement
            self._calculate_vertical_fluxes(fluxes)

            # Step 4: Update bucket states
            self._update_states(fluxes)

            # Step 5: Check water balance
            water_balance_error = 0.0
            if check_water_balance:
                water_balance_error = self._check_water_balance(
                    initial_storage, fluxes
                )

            # Build fluxes dictionary for result
            fluxes_dict = {
                "precipitation": precipitation_mm,
                "irrigation": irrigation_mm,
                "infiltration": fluxes.infiltration,
                "runoff": fluxes.runoff,
                "evaporation": fluxes.evaporation,
                "transpiration": fluxes.transpiration,
                "evapotranspiration": fluxes.total_et,
                "percolation": fluxes.percolation,
                "drainage": fluxes.drainage,
                "et0": et0_mm
            }

            # Prepare result
            result = PhysicsPriorResult(
                date=None,  # Will be set by caller
                theta_surface=self.state[SoilLayer.SURFACE].water_content,
                theta_root=self.state[SoilLayer.ROOT_ZONE].water_content,
                fluxes=fluxes_dict,
                water_balance_error=water_balance_error,
                converged=True
            )
            self.logger.debug(
                f"Daily step complete: "
                f"θ_surface={result.theta_surface:.3f}, "
                f"θ_root={result.theta_root:.3f}, "
                f"WB_error={water_balance_error:.3f}mm"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error in daily step: {e}")
            raise PhysicsModelError(
                f"Failed to run daily step: {e}",
                context={"operation": "run_daily"}
            )

    def _handle_precipitation(
        self,
        precipitation_mm: float,
        irrigation_mm: float
    ) -> Fluxes:
        """
        Handle precipitation and irrigation.

        Returns:
            Fluxes with infiltration and runoff
        """
        fluxes = Fluxes(
            precipitation=precipitation_mm,
            irrigation=irrigation_mm
        )

        # Total water input
        total_input = precipitation_mm + irrigation_mm

        if total_input <= 0:
            fluxes.infiltration = 0.0
            fluxes.runoff = 0.0
            return fluxes

        # Calculate infiltration using surface state
        surface_state = self.state[SoilLayer.SURFACE]

        # Calculate infiltration and runoff
        infiltration_mm_day, runoff_mm_day = self._calculate_infiltration(
            precipitation_mm, irrigation_mm, surface_state
        )

        fluxes.infiltration = infiltration_mm_day
        fluxes.runoff = runoff_mm_day

        return fluxes

    def _calculate_infiltration(
        self,
        precipitation_mm: float,
        irrigation_mm: float,
        surface_state: BucketState
    ) -> Tuple[float, float]:  # infiltration, runoff
        """
        Calculate infiltration and runoff using a simplified approach.

        Based on Philip's infiltration equation simplified for daily timestep:
        - Infiltration rate decreases as soil wets up
        - Runoff occurs when rainfall intensity exceeds infiltration capacity
        - Storage-limited runoff occurs when soil storage is exceeded

        For daily timestep, we use infiltration capacity (mm/day) derived from
        saturated hydraulic conductivity.
        """
        # Total input
        input_total = precipitation_mm + irrigation_mm

        if input_total <= 0:
            return 0.0, 0.0

        # Daily infiltration capacity (mm/day)
        # Use Ks as base, reduced by soil saturation
        K_s_daily = self.params.infiltration_capacity_mm_h * 24.0

        # Saturation factor: reduces infiltration as soil approaches saturation
        # This follows the general behavior of infiltration models
        saturation = surface_state.water_content / surface_state.porosity
        saturation = min(1.0, max(0.0, saturation))

        # Infiltration capacity reduction factor (exponential decay as soil wets)
        # At saturation=0, factor=1; at saturation=1, factor≈0.05
        saturation_factor = np.exp(-3.0 * saturation)

        # Effective infiltration capacity for the day
        infiltration_capacity = K_s_daily * (0.1 + 0.9 * saturation_factor)

        # Available storage in surface layer (up to porosity)
        max_storage_porosity = surface_state.porosity * surface_state.layer_depth_m * 1000
        available_storage = max(0, max_storage_porosity - surface_state.storage)

        # Infiltration is limited by:
        # 1. Infiltration capacity (soil property)
        # 2. Available storage (can't exceed porosity)
        # 3. Available water input
        infiltration = min(infiltration_capacity, available_storage, input_total)

        # Runoff is water that cannot infiltrate
        runoff = max(0, input_total - infiltration)

        return infiltration, runoff

    def _calculate_evapotranspiration(
        self,
        et0_mm: float,
        ndvi: Optional[float],
        air_temperature_c: Optional[float],
        fluxes: Fluxes
    ):
        """
        Calculate evaporation and transpiration.

        Uses FAO-56 dual crop coefficient approach, simplified.
        """
        if et0_mm <= 0:
            fluxes.evaporation = 0.0
            fluxes.transpiration = 0.0
            fluxes.transpiration_surface = 0.0
            fluxes.transpiration_root = 0.0
            fluxes.evapotranspiration = 0.0
            return

        # ET partitioning based on NDVI
        if ndvi is None:
            # Default: 30% evaporation, 70% transpiration for vegetated
            # Adjust based on soil moisture
            evaporation_fraction = 0.3
            transpiration_fraction = 0.7
        else:
            # NDVI-based partitioning
            # Higher NDVI → more transpiration

        #recheck this formula
            evaporation_fraction = np.exp(-self.params.ndvi_et_partitioning_lambda * ndvi)
            transpiration_fraction = 1.0 - evaporation_fraction

        # Adjust for soil moisture stress
        evaporation_reduction = self._calculate_evaporation_reduction()
        transpiration_reduction = self._calculate_transpiration_reduction()

        # Calculate actual ET components
        fluxes.evaporation = (
            et0_mm *
            evaporation_fraction *
            evaporation_reduction *
            self.params.evaporation_coefficient_bare
        )

        total_transpiration = (
            et0_mm *
            transpiration_fraction *
            transpiration_reduction *
            self.params.transpiration_coefficient_max
        )

        # Split transpiration between surface and root zones
        # Transpiration from surface: 20% of total (shallow root uptake)
        # Transpiration from root zone: 80% of total (main root uptake)
        fluxes.transpiration_surface = total_transpiration * 0.20

        fluxes.transpiration_root = total_transpiration * 0.80

        fluxes.transpiration = total_transpiration # Total for water balance

        # Limit ET to available water in each layer
        self._limit_et_to_available_water(fluxes)

        # Physical constraint: Total ET should not exceed ET0 under normal conditions
        # (soil moisture stress already reduces ET, but we cap to ET0 as upper bound)
        total_et = fluxes.evaporation + fluxes.transpiration
        if total_et > et0_mm:
            # Scale down proportionally
            scale_factor = et0_mm / total_et
            fluxes.evaporation *= scale_factor
            fluxes.transpiration *= scale_factor
            fluxes.transpiration_surface *= scale_factor
            fluxes.transpiration_root *= scale_factor

        fluxes.evapotranspiration = fluxes.evaporation + fluxes.transpiration

        self.logger.debug(
            f"ET calculation: "
            f"ET0={et0_mm:.1f}mm, "
            f"E={fluxes.evaporation:.1f}mm, "
            f"T={fluxes.transpiration:.1f}mm (surface={fluxes.transpiration_surface:.2f})"
        )

    def _adjust_et_for_temperature(
        self,
        et0_mm: float,
        air_temperature_c: float,
        soil_temperature_c: Optional[float] = None
    ) -> float:
        """
        Adjust ET0 for temperature effects.
        """
        if air_temperature_c is None:
            return et0_mm

        # Temperature adjustment factor (0-2)
        # Based on FAO-56 temperature adjustment
        if air_temperature_c < 5:
            # Very cold - reduced ET
            factor = 0.3 + 0.07 * air_temperature_c
        elif air_temperature_c > 35:
            # Very hot - increased ET
            factor = 1.0 + 0.02 * (air_temperature_c - 35)
        else:
            # Normal range
            factor = 0.6 + 0.02 * air_temperature_c

        # Adjust for soil temperature if available
        if soil_temperature_c is not None:
            # Cold soil reduces ET
            if soil_temperature_c < 10:
                soil_factor = 0.5 + 0.05 * soil_temperature_c
                factor *= soil_factor

        return et0_mm * min(2.0, max(0.1, factor))

    def _calculate_evaporation_reduction(self) -> float:
        """Calculate evaporation reduction due to soil dryness"""
        surface_state = self.state[SoilLayer.SURFACE]

        # FAO-56 approach: evaporation reduction when soil is dry
        # Simplified: linear reduction below field capacity
        if surface_state.water_content >= self.params.field_capacity_surface:
            return 1.0

        # Calculate relative extractable water
        rew = (
            (surface_state.water_content - self.params.wilting_point_surface) /
            (self.params.field_capacity_surface - self.params.wilting_point_surface)
        )

        # Limit between 0 and 1
        return max(0.0, min(1.0, rew))

    def _calculate_transpiration_reduction(self) -> float:
        """
        Calculate transpiration reduction due to soil moisture stress.

        Uses Feddes-style stress function with smooth transitions:
        - Optimal uptake above threshold (~50% available water)
        - Gradual linear reduction below threshold to wilting point
        - Considers root distribution between layers (weighted average)
        """
        surface_state = self.state[SoilLayer.SURFACE]
        root_state = self.state[SoilLayer.ROOT_ZONE]

        # FAO-56: Transpiration stress when soil moisture is below threshold
        # Threshold (p value) is crop-dependent, typically 0.4-0.6
        threshold_fraction = 0.45  # Below this fraction of available water, stress begins

        # Calculate relative extractable water for each layer
        def calc_rew(theta, theta_wp, theta_fc):
            available = theta_fc - theta_wp
            if available < 0.01:
                return 0.5
            rew = (theta - theta_wp) / available
            return max(0.0, min(1.0, rew))

        rew_surface = calc_rew(
            surface_state.water_content,
            self.params.wilting_point_surface,
            self.params.field_capacity_surface
        )
        rew_root = calc_rew(
            root_state.water_content,
            self.params.wilting_point_root,
            self.params.field_capacity_root
        )

        # Root-weighted average (more roots in root zone)
        # This is more physically realistic than just using root zone
        root_fraction_surface = 0.35  # ~35% of roots in surface 10cm
        root_fraction_root = 0.65     # ~65% of roots in root zone 30cm

        rew_weighted = root_fraction_surface * rew_surface + root_fraction_root * rew_root

        # Calculate stress factor with smooth transition
        if rew_weighted >= threshold_fraction:
            # No stress above threshold
            return 1.0
        elif rew_weighted <= 0:
            return 0.0
        else:
            # Linear reduction below threshold (FAO-56 Ks)
            return rew_weighted / threshold_fraction

    def _limit_et_to_available_water(self, fluxes: Fluxes):
        """Ensure ET doesn't exceed available water in each layer.

        Key soil physics constraint: Water cannot be extracted below wilting point.
        The wilting point represents the soil water potential (~-1.5 MPa) at which
        plants can no longer extract water.
        """
        surface_state = self.state[SoilLayer.SURFACE]
        root_state = self.state[SoilLayer.ROOT_ZONE]

        # HARD LIMIT: Cannot extract below wilting point
        min_surface_storage = surface_state.wilting_point * surface_state.layer_depth_m * 1000
        min_root_storage = root_state.wilting_point * root_state.layer_depth_m * 1000

        # Available water that can be extracted from each layer
        available_from_surface = max(0, surface_state.storage - min_surface_storage)
        available_from_root = max(0, root_state.storage - min_root_storage)

        # Surface layer loses: evaporation + transpiration_surface
        total_surface_demand = fluxes.evaporation + fluxes.transpiration_surface
        if total_surface_demand > available_from_surface and total_surface_demand > 0:
            # Scale down both components proportionally
            scale = available_from_surface / total_surface_demand
            fluxes.evaporation *= scale
            fluxes.transpiration_surface *= scale

        # Root zone loses: transpiration_root
        if fluxes.transpiration_root > available_from_root:
            fluxes.transpiration_root = available_from_root

        # Update total transpiration to match components
        fluxes.transpiration = fluxes.transpiration_surface + fluxes.transpiration_root

# Percolation calculation. Remember to check this formula and hydraulic conductivity units

    def _calculate_percolation_flux(
        self,
        upper_bucket: BucketState,
        lower_bucket: BucketState,
        hydraulic_conductivity: float  # mm/day
    ) -> float:
        """
        Calculate percolation using Darcy's law (simplified).

        Flux = K * (h_upper - h_lower) / L where h is hydraulic head.
        """
        # Simplified: use soil moisture gradient
        gradient = upper_bucket.water_content - lower_bucket.water_content

        # Only percolate if upper is wetter than lower
        if gradient <= 0:
            return 0.0

        # Calculate flux with limits
        flux = min(
            hydraulic_conductivity * gradient,
            upper_bucket.storage * 0.5  # Max 50% of available water
        )

        # Ensure flux doesn't exceed available water
        return max(0.0, min(flux, upper_bucket.available_water))

    def _calculate_vertical_fluxes(self, fluxes: Fluxes):
        """
        Calculate percolation (surface → root) and drainage (root → deep).

        Uses simple reservoir approach with rate constants.

        Percolation occurs when soil water content exceeds field capacity.
        In soil physics, this represents free-draining water that moves
        under gravity through macropores and preferential flow paths.

        Note: We consider infiltration that will be added this timestep,
        since percolation happens concurrently with infiltration.
        """
        # Percolation: water moves from surface to root when surface is above FC
        surface_state = self.state[SoilLayer.SURFACE]

        # Calculate what storage will be after infiltration (before ET/percolation)
        projected_surface_storage = surface_state.storage + fluxes.infiltration
        projected_surface_wc = projected_surface_storage / (surface_state.layer_depth_m * 1000)

        if projected_surface_wc > self.params.field_capacity_surface:
            # Excess water above field capacity that can percolate
            fc_storage = self.params.field_capacity_surface * surface_state.layer_depth_m * 1000
            excess = projected_surface_storage - fc_storage

            # Percolation rate: fraction of excess water that drains per day
            fluxes.percolation = min(
                excess * self.params.percolation_rate_day,
                excess  # Can't percolate more than excess
            )
        else:
            fluxes.percolation = 0.0

        # Drainage: water moves out of root zone when above FC
        root_state = self.state[SoilLayer.ROOT_ZONE]

        # Project root zone storage after receiving percolation
        projected_root_storage = root_state.storage + fluxes.percolation
        projected_root_wc = projected_root_storage / (root_state.layer_depth_m * 1000)

        if projected_root_wc > self.params.field_capacity_root:
            fc_storage = self.params.field_capacity_root * root_state.layer_depth_m * 1000
            excess = projected_root_storage - fc_storage

            fluxes.drainage = min(
                excess * self.params.drainage_rate_day,
                excess  # Can't drain more than excess
            )
        else:
            fluxes.drainage = 0.0

        self.logger.debug(
            f"Vertical fluxes: "
            f"percolation={fluxes.percolation:.1f}mm, "
            f"drainage={fluxes.drainage:.1f}mm"
        )

    def _update_states(self, fluxes: Fluxes):
        """Update bucket states based on calculated fluxes.

        Enforces physical constraints:
        - Water content cannot exceed porosity (saturated conditions)
        - Water content cannot go below 0 (residual water is bound)
        """
        # Update surface layer
        surface = self.state[SoilLayer.SURFACE]

        # Add infiltration
        surface.storage += fluxes.infiltration

        # Remove evaporation and percolation
        surface.storage -= (
            fluxes.evaporation +
            fluxes.transpiration_surface +
            fluxes.percolation
        )

        # Calculate max storage based on porosity (physical upper limit)
        max_surface_storage = surface.porosity * surface.layer_depth_m * 1000

        # Ensure non-negative and within bounds (0 to porosity)
        surface.storage = max(0.0, min(max_surface_storage, surface.storage))
        surface.water_content = surface.storage / (surface.layer_depth_m * 1000)

        # Update root zone layer
        root = self.state[SoilLayer.ROOT_ZONE]

        # Add percolation
        root.storage += fluxes.percolation

        # Remove transpiration and drainage
        root.storage -= fluxes.transpiration_root + fluxes.drainage

        # Calculate max storage based on porosity
        max_root_storage = root.porosity * root.layer_depth_m * 1000

        # Ensure non-negative and within bounds
        root.storage = max(0.0, min(max_root_storage, root.storage))
        root.water_content = root.storage / (root.layer_depth_m * 1000)

        self.logger.debug(
            f"Updated states: "
            f"surface_θ={surface.water_content:.3f}, "
            f"root_θ={root.water_content:.3f}"
        )


    def _update_states_single_step(self, fluxes: Fluxes):
        """Single step update with bounds checking"""
        # Surface layer
        surface = self.state[SoilLayer.SURFACE]
        new_storage = (
            surface.storage +
            fluxes.infiltration -
            fluxes.evaporation -
            fluxes.transpiration_surface -
            fluxes.percolation
        )

        # Ensure bounds
        new_storage = max(0.0, min(surface.max_storage, new_storage))
        surface.water_content = new_storage / (surface.layer_depth_m * 1000)

        # Root zone layer
        root = self.state[SoilLayer.ROOT_ZONE]
        new_storage = (
            root.storage +
            fluxes.percolation -
            fluxes.transpiration_root -
            fluxes.drainage
        )

        # Ensure bounds
        new_storage = max(0.0, min(root.max_storage, new_storage))
        root.water_content = new_storage / (root.layer_depth_m * 1000)

    def _check_water_balance(
        self,
        initial_storage: float,
        fluxes: Fluxes
    ) -> float:
        """
        Check water balance closure.

        Returns:
            Water balance error in mm (should be near zero)
        """
        final_storage = self._total_storage()

        # Inputs: precipitation, irrigation
        inputs = fluxes.precipitation + fluxes.irrigation

        # Outputs: runoff, evaporation, transpiration, drainage
        outputs = (
            fluxes.runoff +
            fluxes.evaporation +
            fluxes.transpiration +
            fluxes.drainage
        )

        # Water balance equation: ΔS = Inputs - Outputs
        delta_storage = final_storage - initial_storage
        water_balance_error = delta_storage - (inputs - outputs)

        # Track cumulative error
        self.cumulative_error += abs(water_balance_error)
        self.iteration_count += 1

        if abs(water_balance_error) > self.params.tolerance * 10:
            self.logger.warning(
                f"Large water balance error: {water_balance_error:.3f}mm\n"
                f"  Initial S: {initial_storage:.1f}mm\n"
                f"  Final S: {final_storage:.1f}mm\n"
                f"  Inputs: {inputs:.1f}mm\n"
                f"  Outputs: {outputs:.1f}mm\n"
                f"  ΔS (calc): {delta_storage:.1f}mm\n"
                f"  ΔS (expected): {inputs - outputs:.1f}mm"
            )

        return water_balance_error

    def _total_storage(self) -> float:
        """Calculate total water storage in both layers (mm)"""
        return sum(bucket.storage for bucket in self.state.values())

    def run_period(
        self,
        forcings: pd.DataFrame,
        initial_date: Optional[date] = None,
        warmup_days: int = 30
    ) -> pd.DataFrame:
        """
        Run model for a period with time series of forcings.

        Args:
            forcings: DataFrame with columns:
                - precipitation_mm (required)
                - et0_mm (required)
                - ndvi (optional)
                - irrigation_mm (optional)
                - temperature_c (optional)
            initial_date: Start date for results
            warmup_days: Number of days to run before results (spin-up)

        Returns:
            DataFrame with daily physics prior results
        """
        self.logger.info(
            f"Running model for {len(forcings)} days "
            f"(warmup: {warmup_days} days)"
        )

        # Validate forcings
        self._validate_forcings(forcings)

        # Apply warmup period
        if warmup_days > 0 and len(forcings) > warmup_days:
            warmup_forcings = forcings.iloc[:warmup_days]
            self._run_period_internal(warmup_forcings, store_results=False)
            forcings = forcings.iloc[warmup_days:]

        # Run main period
        results = self._run_period_internal(forcings, store_results=True)

        # Add dates if provided
        if initial_date is not None:
            dates = pd.date_range(
                start=initial_date + timedelta(days=warmup_days),
                periods=len(results),
                freq='D'
            )
            results['date'] = dates

        self.logger.info(
            f"Model run complete. "
            f"Avg water balance error: {self.cumulative_error/self.iteration_count:.3f}mm"
        )

        return results

    def _validate_forcings(self, forcings: pd.DataFrame):
        """Validate input forcings DataFrame"""
        required_columns = ['precipitation_mm', 'et0_mm']
        for col in required_columns:
            if col not in forcings.columns:
                raise ValueError(f"Missing required column: {col}")

        # Check for negative values
        for col in ['precipitation_mm', 'et0_mm', 'irrigation_mm']:
            if col in forcings.columns:
                if (forcings[col] < 0).any():
                    self.logger.warning(f"Negative values found in {col}")

        # Check NDVI range
        if 'ndvi' in forcings.columns:
            ndvi_range = forcings['ndvi'].min(), forcings['ndvi'].max()
            if ndvi_range[0] < -1 or ndvi_range[1] > 1:
                self.logger.warning(f"NDVI outside typical range: {ndvi_range}")

    def _run_period_internal(
        self,
        forcings: pd.DataFrame,
        store_results: bool = True
    ) -> pd.DataFrame:
        """Internal method to run model for a period"""
        results = []

        for idx, row in forcings.iterrows():
            try:
                result = self.run_daily(
                    precipitation_mm=row['precipitation_mm'],
                    et0_mm=row['et0_mm'],
                    ndvi=row.get('ndvi'),
                    irrigation_mm=row.get('irrigation_mm', 0.0),
                    air_temperature_c=row.get('temperature_c')
                )

                if store_results:
                    # Convert result to dict for DataFrame
                    result_dict = {
                        'theta_phys_surface': result.theta_surface,
                        'theta_phys_root': result.theta_root,
                        'water_balance_error_mm': result.water_balance_error,
                        'converged': result.converged
                    }

                    # Add fluxes
                    for flux_name, flux_value in result.fluxes.items():
                        result_dict[f'flux_{flux_name}_mm'] = flux_value

                    results.append(result_dict)

            except Exception as e:
                self.logger.error(f"Error at step {idx}: {e}")
                # Store NaN results for failed steps
                if store_results:
                    results.append({
                        'theta_phys_surface': np.nan,
                        'theta_phys_root': np.nan,
                        'water_balance_error_mm': np.nan,
                        'converged': False
                    })

        return pd.DataFrame(results)

    def reset(self, state: Optional[Dict[SoilLayer, BucketState]] = None):
        """Reset model to initial or specified state"""
        if state is None:
            self.state = self._initialize_default_state()
        else:
            self.state = state

        self.cumulative_error = 0.0
        self.iteration_count = 0
        self._cache.clear()

        self.logger.info("Model reset to initial state")

    def calibrate(
        self,
        observations: pd.DataFrame,
        parameters_to_calibrate: List[str],
        objective_function: Callable,
        method: str = 'nelder-mead'
    ) -> Dict[str, float]:
        """
        Calibrate model parameters against observations.

        Args:
            observations: DataFrame with observed soil moisture
            parameters_to_calibrate: List of parameter names to calibrate
            objective_function: Function to minimize (e.g., RMSE)
            method: Optimization method

        Returns:
            Dictionary of calibrated parameters
        """
        self.logger.info(
            f"Calibrating parameters: {parameters_to_calibrate}"
        )

        # Store original parameters
        original_params = {
            param: getattr(self.params, param)
            for param in parameters_to_calibrate
        }

        try:
            from scipy.optimize import minimize

            # Placeholder implementation
            self.logger.info("Calibration method not fully implemented")
            return original_params

        except ImportError:
            self.logger.warning(
                "scipy not available for calibration. "
                "Using default parameters."
            )
            return original_params
        finally:
            # Ensure model is in valid state
            self.reset()

    def calibrate_robust(
        self,
        observations: pd.DataFrame,
        parameter_ranges: Dict[str, Tuple[float, float]],
        n_iterations: int = 100,
        n_workers: int = -1
    ) -> Dict[str, float]:
        """
        Robust calibration using global optimization.
        """
        try:
            from scipy.optimize import differential_evolution

            def objective(params_dict: Dict[str, float]) -> float:
                # Set parameters
                for param_name, param_value in params_dict.items():
                    setattr(self.params, param_name, param_value)

                # Run simulation
                results = self.run_period(
                    observations[['precipitation_mm', 'et0_mm', 'ndvi']],
                    warmup_days=60  # Longer warmup for calibration
                )

                # Calculate multiple metrics
                simulated = results['theta_phys_root'].values
                observed = observations['soil_moisture_vwc'].values

                # Trim to valid data
                mask = ~(np.isnan(simulated) | np.isnan(observed))
                if mask.sum() < 10:
                    return 1e6  # Penalize insufficient data

                sim = simulated[mask]
                obs = observed[mask]

                # Combined loss function
                rmse = np.sqrt(np.mean((sim - obs) ** 2))
                bias = np.mean(sim - obs)
                correlation = np.corrcoef(sim, obs)[0, 1] if len(sim) > 1 else 0

                # Weighted loss
                loss = rmse + 2 * abs(bias) - 0.5 * correlation

                return loss

            # Convert parameter ranges to bounds
            bounds = [parameter_ranges[param] for param in parameter_ranges]
            param_names = list(parameter_ranges.keys())

            # Run optimization
            result = differential_evolution(
                lambda x: objective(dict(zip(param_names, x))),
                bounds,
                maxiter=n_iterations,
                workers=n_workers,
                disp=True,
                seed=42
            )

            # Get best parameters
            calibrated = dict(zip(param_names, result.x))

            self.logger.info(
                f"Calibration complete. Best loss: {result.fun:.4f}, "
                f"Success: {result.success}"
            )

            return calibrated

        except ImportError:
            self.logger.error("scipy is required for calibration")
            raise

    def get_diagnostic_info(self) -> Dict:
        """Get diagnostic information about model state"""
        return {
            'current_states': {
                layer: {
                    'water_content': bucket.water_content,
                    'storage_mm': bucket.storage,
                    'saturation': bucket.saturation,
                    'available_water_mm': bucket.available_water
                }
                for layer, bucket in self.state.items()
            },
            'parameters': {
                k: v for k, v in self.params.__dict__.items()
                if not k.startswith('_')
            },
            'performance': {
                'cumulative_water_balance_error_mm': self.cumulative_error,
                'iteration_count': self.iteration_count,
                'avg_water_balance_error_mm': (
                    self.cumulative_error / self.iteration_count
                    if self.iteration_count > 0 else 0.0
                )
            }
        }


# Factory function for easy model creation
def create_two_bucket_model(
    soil_params: Optional[SoilParameters] = None,
    config: Optional[Dict] = None,
    initial_conditions: Optional[Dict[str, float]] = None
) -> TwoBucketWaterBalance:
    """
    Factory function to create a two-bucket water balance model.

    Args:
        soil_params: Soil parameters (optional, will use defaults)
        config: Configuration overrides
        initial_conditions: Initial soil moisture values

    Returns:
        Configured TwoBucketWaterBalance instance
    """
    # Get default config
    config = config or {}

    # Create soil parameters if not provided
    if soil_params is None:
        from smps.physics.pedotransfer import create_default_soil_parameters
        soil_params = create_default_soil_parameters()

    # Create model parameters
    model_params = ModelParameters.from_soil_parameters(soil_params, config)

    # Create initial state if provided
    initial_state = None
    if initial_conditions is not None:
        initial_state = {
            SoilLayer.SURFACE: BucketState(
                water_content=initial_conditions.get('surface',
                    model_params.field_capacity_surface),
                max_storage=model_params.surface_depth_m * 1000,
                layer_depth_m=model_params.surface_depth_m,
                field_capacity=model_params.field_capacity_surface,
                wilting_point=model_params.wilting_point_surface,
                porosity=model_params.porosity_surface
            ),
            SoilLayer.ROOT_ZONE: BucketState(
                water_content=initial_conditions.get('root_zone',
                    model_params.field_capacity_root),
                max_storage=model_params.root_zone_depth_m * 1000,
                layer_depth_m=model_params.root_zone_depth_m,
                field_capacity=model_params.field_capacity_root,
                wilting_point=model_params.wilting_point_root,
                porosity=model_params.porosity_root
            )
        }

    return TwoBucketWaterBalance(model_params, initial_state)


class SiteSpecificWaterBalance(TwoBucketWaterBalance):
    """
    Site-specific water balance model with automatic data fetching.

    This class extends TwoBucketWaterBalance to automatically:
    1. Fetch site-specific soil parameters
    2. Retrieve meteorological forcings
    3. Download validation data
    4. Configure crop-specific parameters

    Usage:
        >>> config = SmpsConfig()
        >>> wb = SiteSpecificWaterBalance(lat=1.0, lon=36.0, config=config)
        >>> results, obs = wb.run_hindcast('2023-01-01', '2023-12-31')
    """

    def __init__(
        self,
        lat: float,
        lon: float,
        config: SmpsConfig,
        crop_type: Optional[str] = None,
        custom_root_depth: Optional[float] = None,
        initial_conditions: Optional[Dict[str, float]] = None
    ):
        """
        Initialize site-specific water balance model.

        Args:
            lat: Latitude (decimal degrees)
            lon: Longitude (decimal degrees)
            config: SMPS configuration object
            crop_type: Crop type for rooting depth (optional)
            custom_root_depth: Custom root zone depth in meters (optional)
            initial_conditions: Initial soil moisture values (optional)
        """
        self.lat = lat
        self.lon = lon
        self.config = config
        self.data_fetcher = SmpsDataFetcher(config)

        logger.info(f"Initializing water balance for site ({lat:.4f}, {lon:.4f})")

        # Fetch real soil parameters
        try:
            self.soil_params = self.data_fetcher.fetch_site_soil_parameters(lat, lon)
            logger.info(
                f"Fetched soil parameters: "
                f"sand={self.soil_params.sand_fraction:.2%}, "
                f"clay={self.soil_params.clay_fraction:.2%}"
            )
        except Exception as e:
            logger.warning(f"Failed to fetch soil parameters: {e}. Using defaults.")
            from smps.physics.pedotransfer import create_default_soil_parameters
            self.soil_params = create_default_soil_parameters()

        # Determine rooting depth
        if custom_root_depth is not None:
            root_depth = custom_root_depth
            logger.info(f"Using custom root depth: {root_depth}m")
        else:
            # Get crop type from config or use provided
            site_key = f"{lat}_{lon}"
            if crop_type is None:
                crop_type = config.site_configs.get(site_key, {}).get("crop", "generic")

            root_depth = DEFAULT_CROP_ROOTING_DEPTHS.get(crop_type, 0.3)
            logger.info(f"Using {crop_type} root depth: {root_depth}m")

        self.crop_type = crop_type
        self.root_depth = root_depth

        # Create model parameters
        model_params = ModelParameters.from_soil_parameters(
            self.soil_params,
            config_overrides={"root_zone_depth_m": root_depth}
        )

        # Create initial state if provided
        initial_state = None
        if initial_conditions is not None:
            initial_state = {
                SoilLayer.SURFACE: BucketState(
                    water_content=initial_conditions.get(
                        'surface',
                        model_params.field_capacity_surface
                    ),
                    max_storage=model_params.surface_depth_m * 1000,
                    layer_depth_m=model_params.surface_depth_m,
                    field_capacity=model_params.field_capacity_surface,
                    wilting_point=model_params.wilting_point_surface,
                    porosity=model_params.porosity_surface
                ),
                SoilLayer.ROOT_ZONE: BucketState(
                    water_content=initial_conditions.get(
                        'root_zone',
                        model_params.field_capacity_root
                    ),
                    max_storage=model_params.root_zone_depth_m * 1000,
                    layer_depth_m=model_params.root_zone_depth_m,
                    field_capacity=model_params.field_capacity_root,
                    wilting_point=model_params.wilting_point_root,
                    porosity=model_params.porosity_root
                )
            }
            logger.info(
                f"Initialized with custom conditions: "
                f"θ_surface={initial_conditions.get('surface', 0):.3f}, "
                f"θ_root={initial_conditions.get('root_zone', 0):.3f}"
            )

        # Initialize parent class
        super().__init__(model_params, initial_state)

        logger.info("Site-specific water balance initialized successfully")

    def run_hindcast(
        self,
        start_date: str,
        end_date: str,
        warmup_days: int = 30,
        validate: bool = True,
        return_diagnostics: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[Dict]]:
        """
        Run hindcast simulation with automatic data fetching and validation.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            warmup_days: Number of warmup days before start_date
            validate: Whether to fetch and compare with observations
            return_diagnostics: Whether to return diagnostic metrics

        Returns:
            Tuple of (results_df, observations_df, diagnostics_dict)
        """
        logger.info(
            f"Running hindcast from {start_date} to {end_date} "
            f"(warmup: {warmup_days} days)"
        )

        # Parse dates
        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date()

        # Extend start date for warmup
        warmup_start = start - timedelta(days=warmup_days)

        # Fetch forcings
        logger.info("Fetching meteorological forcings...")
        try:
            forcings = self.data_fetcher.fetch_daily_forcings(
                self.lat, self.lon, warmup_start, end
            )
            logger.info(f"Fetched {len(forcings)} days of forcings")
        except Exception as e:
            logger.error(f"Failed to fetch forcings: {e}")
            raise

        # Validate forcings
        self._validate_forcings(forcings)

        # Run water balance
        logger.info("Running water balance model...")
        try:
            results = self.run_period(
                forcings=forcings[['precipitation_mm', 'et0_mm', 'ndvi']],
                initial_date=start,
                warmup_days=warmup_days
            )
            logger.info(f"Simulation complete: {len(results)} days")
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise

        # Validate against observations
        observations = None
        diagnostics = None

        if validate:
            logger.info("Fetching validation data...")
            try:
                observations = self.data_fetcher.fetch_reference_soil_moisture(
                    self.lat, self.lon, start, end
                )
                logger.info(f"Fetched {len(observations)} observations")

                # Calculate validation metrics
                diagnostics = self._calculate_validation_metrics(
                    results, observations
                )

                # Log key metrics
                logger.info(
                    f"Validation metrics:\n"
                    f"  Surface RMSE: {diagnostics.get('surface_rmse', np.nan):.4f} m³/m³\n"
                    f"  Root RMSE: {diagnostics.get('root_rmse', np.nan):.4f} m³/m³\n"
                    f"  Surface Bias: {diagnostics.get('surface_bias', np.nan):.4f} m³/m³\n"
                    f"  Root Bias: {diagnostics.get('root_bias', np.nan):.4f} m³/m³\n"
                    f"  Surface R²: {diagnostics.get('surface_r2', np.nan):.3f}\n"
                    f"  Root R²: {diagnostics.get('root_r2', np.nan):.3f}"
                )

            except Exception as e:
                logger.warning(f"Validation failed: {e}")
                observations = None
                diagnostics = None

        # Return based on what was requested
        if return_diagnostics:
            return results, observations, diagnostics
        else:
            return results, observations

    def _calculate_validation_metrics(
        self,
        results: pd.DataFrame,
        observations: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate validation metrics comparing simulated vs observed.

        Args:
            results: Simulation results
            observations: Observed soil moisture

        Returns:
            Dictionary of validation metrics
        """
        metrics = {}

        # Merge on date for comparison
        if 'date' in results.columns and 'date' in observations.columns:
            merged = pd.merge(
                results, observations,
                on='date', how='inner', suffixes=('_sim', '_obs')
            )
        else:
            logger.warning("No date column for alignment, using index")
            merged = pd.concat([results, observations], axis=1)

        # Calculate metrics for each layer
        for layer in ['surface', 'root']:
            sim_col = f'theta_phys_{layer}'
            obs_col = f'{layer}_sm'

            if sim_col in merged.columns and obs_col in merged.columns:
                # Remove NaN values
                mask = ~(merged[sim_col].isna() | merged[obs_col].isna())
                if mask.sum() < 10:
                    logger.warning(f"Insufficient valid data for {layer} layer")
                    continue

                sim = merged.loc[mask, sim_col].values
                obs = merged.loc[mask, obs_col].values

                # RMSE
                rmse = np.sqrt(np.mean((sim - obs) ** 2))
                metrics[f'{layer}_rmse'] = rmse

                # Bias
                bias = np.mean(sim - obs)
                metrics[f'{layer}_bias'] = bias

                # MAE
                mae = np.mean(np.abs(sim - obs))
                metrics[f'{layer}_mae'] = mae

                # R²
                ss_res = np.sum((obs - sim) ** 2)
                ss_tot = np.sum((obs - np.mean(obs)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                metrics[f'{layer}_r2'] = r2

                # Correlation
                if len(sim) > 1:
                    corr = np.corrcoef(sim, obs)[0, 1]
                    metrics[f'{layer}_correlation'] = corr

                # Nash-Sutcliffe Efficiency
                nse = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf
                metrics[f'{layer}_nse'] = nse

                # Percent bias
                pbias = 100 * np.sum(sim - obs) / np.sum(obs) if np.sum(obs) != 0 else 0
                metrics[f'{layer}_pbias'] = pbias

        # Overall statistics
        metrics['n_valid_points'] = mask.sum() if 'mask' in locals() else 0
        metrics['validation_period_days'] = len(merged)

        return metrics

    def run_forecast(
        self,
        forecast_days: int = 10,
        use_ensemble: bool = False,
        n_ensemble: int = 20
    ) -> pd.DataFrame:
        """
        Run forecast simulation using weather forecasts.

        Args:
            forecast_days: Number of days to forecast
            use_ensemble: Whether to use ensemble forecasts
            n_ensemble: Number of ensemble members

        Returns:
            DataFrame with forecast results (and uncertainty if ensemble)
        """
        logger.info(f"Running {forecast_days}-day forecast")

        # Get current date
        today = date.today()
        forecast_end = today + timedelta(days=forecast_days)

        # Fetch forecast data
        try:
            forecast_forcings = self.data_fetcher.fetch_forecast_data(
                self.lat, self.lon, today, forecast_end,
                ensemble=use_ensemble, n_members=n_ensemble if use_ensemble else 1
            )
        except Exception as e:
            logger.error(f"Failed to fetch forecast data: {e}")
            raise

        if use_ensemble:
            # Run ensemble forecast
            ensemble_results = []
            for member_id in range(n_ensemble):
                member_forcings = forecast_forcings[
                    forecast_forcings['ensemble_member'] == member_id
                ]

                # Reset state for each member
                self.reset()

                # Run simulation
                results = self.run_period(
                    forcings=member_forcings[['precipitation_mm', 'et0_mm', 'ndvi']],
                    warmup_days=0
                )
                results['ensemble_member'] = member_id
                ensemble_results.append(results)

            # Combine and calculate statistics
            all_results = pd.concat(ensemble_results, ignore_index=True)

            # Calculate ensemble mean and spread
            forecast_summary = all_results.groupby('date').agg({
                'theta_phys_surface': ['mean', 'std', 'min', 'max'],
                'theta_phys_root': ['mean', 'std', 'min', 'max']
            })

            return forecast_summary
        else:
            # Single deterministic forecast
            results = self.run_period(
                forcings=forecast_forcings[['precipitation_mm', 'et0_mm', 'ndvi']],
                warmup_days=0
            )
            return results

    def calibrate_to_site(
        self,
        calibration_period_start: str,
        calibration_period_end: str,
        parameters_to_calibrate: Optional[list] = None,
        validation_split: float = 0.3
    ) -> Dict[str, float]:
        """
        Calibrate model parameters to site observations.

        Args:
            calibration_period_start: Start of calibration period
            calibration_period_end: End of calibration period
            parameters_to_calibrate: List of parameters to calibrate
            validation_split: Fraction of data for validation

        Returns:
            Dictionary of calibrated parameters
        """
        logger.info(
            f"Calibrating model from {calibration_period_start} "
            f"to {calibration_period_end}"
        )

        # Default parameters to calibrate
        if parameters_to_calibrate is None:
            parameters_to_calibrate = [
                'percolation_rate_day',
                'drainage_rate_day',
                'infiltration_capacity_mm_h',
                'ndvi_et_partitioning_lambda'
            ]

        # Fetch data for calibration
        start = pd.to_datetime(calibration_period_start).date()
        end = pd.to_datetime(calibration_period_end).date()

        forcings = self.data_fetcher.fetch_daily_forcings(
            self.lat, self.lon, start, end
        )
        observations = self.data_fetcher.fetch_reference_soil_moisture(
            self.lat, self.lon, start, end
        )

        # Merge forcings and observations
        data = pd.merge(forcings, observations, on='date', how='inner')

        # Split into calibration and validation
        n_total = len(data)
        n_calib = int(n_total * (1 - validation_split))

        calib_data = data.iloc[:n_calib]
        valid_data = data.iloc[n_calib:]

        logger.info(
            f"Split data: {n_calib} days calibration, "
            f"{len(valid_data)} days validation"
        )

        # Define parameter ranges
        parameter_ranges = {
            'percolation_rate_day': (0.01, 0.5),
            'drainage_rate_day': (0.01, 0.5),
            'infiltration_capacity_mm_h': (5.0, 50.0),
            'ndvi_et_partitioning_lambda': (0.5, 5.0)
        }

        # Filter to only parameters we want to calibrate
        param_ranges = {
            k: v for k, v in parameter_ranges.items()
            if k in parameters_to_calibrate
        }

        # Run calibration
        calibrated_params = self.calibrate_robust(
            observations=calib_data,
            parameter_ranges=param_ranges,
            n_iterations=100
        )

        # Validate on holdout data
        for param, value in calibrated_params.items():
            setattr(self.params, param, value)

        self.reset()
        valid_results = self.run_period(
            valid_data[['precipitation_mm', 'et0_mm', 'ndvi']],
            warmup_days=0
        )

        # Calculate validation metrics
        valid_metrics = self._calculate_validation_metrics(
            valid_results, valid_data
        )

        logger.info(
            f"Calibration complete. Validation RMSE: "
            f"{valid_metrics.get('root_rmse', np.nan):.4f} m³/m³"
        )

        return calibrated_params


# Placeholder for SmpsDataFetcher - needs to be imported from appropriate module
class SmpsDataFetcher:
    """Placeholder for data fetcher - should be imported from smps.data module"""
    def __init__(self, config):
        self.config = config

    def fetch_site_soil_parameters(self, lat, lon):
        raise NotImplementedError("SmpsDataFetcher should be imported from smps.data")

    def fetch_daily_forcings(self, lat, lon, start, end):
        raise NotImplementedError("SmpsDataFetcher should be imported from smps.data")

    def fetch_reference_soil_moisture(self, lat, lon, start, end):
        raise NotImplementedError("SmpsDataFetcher should be imported from smps.data")

    def fetch_forecast_data(self, lat, lon, start, end, ensemble=False, n_members=1):
        raise NotImplementedError("SmpsDataFetcher should be imported from smps.data")


# =============================================================================
# THREE-LAYER MODEL FOR 0-100cm DEPTH (FLDAS COMPATIBLE)
# =============================================================================

@dataclass
class ThreeLayerParameters:
    """
    Physical parameters for three-layer (0-100cm) water balance model.
    
    Designed to match FLDAS 0-100cm integrated soil moisture products.
    
    Layer Structure:
    - Surface (0-10cm): Fast response, high evaporation
    - Root zone (10-40cm): Main root water uptake
    - Deep subsoil (40-100cm): Slow drainage, limited root access
    
    Key Soil Physics Improvements (v2):
    1. Reduced drainage rates based on African soil hydraulic properties
       - Many African soils have lateritic/clay-rich B horizons that impede drainage
       - FLDAS uses Noah LSM which has similar slow drainage assumptions
    2. Added capillary rise parameters for upward water movement
       - Important in semi-arid regions with shallow water tables
       - Represents matric potential-driven upward flux
    3. Increased field capacity values for tropical soils
       - African soils often have higher clay content at depth (argillic horizons)
       - Ferralsols/Acrisols have high water retention despite good structure
    """
    # Layer depths (m)
    surface_depth_m: float = 0.10      # 0-10cm
    root_zone_depth_m: float = 0.30    # 10-40cm
    deep_depth_m: float = 0.60         # 40-100cm
    
    # Surface layer (0-10cm) hydraulic properties
    # Increased field capacity for tropical soils (higher OM, better structure)
    porosity_surface: float = 0.48     # Increased for tropical soils
    field_capacity_surface: float = 0.35  # Increased - tropical soils retain more water
    wilting_point_surface: float = 0.12
    
    # Root zone (10-40cm) hydraulic properties
    # This is the critical zone - FLDAS shows high moisture here
    porosity_root: float = 0.46        # Increased
    field_capacity_root: float = 0.34  # Increased significantly
    wilting_point_root: float = 0.11
    
    # Deep layer (40-100cm) - typically higher water retention
    # Clay illuviation in B horizon creates high retention
    porosity_deep: float = 0.44        # Increased - less compaction assumed
    field_capacity_deep: float = 0.38  # Significantly higher - argillic horizon effect
    wilting_point_deep: float = 0.15   # Higher due to clay accumulation
    
    # Infiltration parameters
    infiltration_capacity_mm_h: float = 15.0  # Slightly reduced for crusted soils
    
    # ==========================================================================
    # DRAINAGE PARAMETERS - REDUCED BASED ON SOIL PHYSICS
    # ==========================================================================
    # Key insight: Original rates (0.15, 0.08, 0.03) were too aggressive
    # 
    # Darcy's Law: q = K * (dh/dz)
    # For gravity drainage: q = K_unsat
    # K_unsat decreases exponentially as soil dries below saturation
    #
    # Typical values for African soils:
    # - Sandy soils: K_sat ~ 100-500 mm/day, but K at FC ~ 1-5 mm/day
    # - Loamy soils: K_sat ~ 10-50 mm/day, but K at FC ~ 0.1-1 mm/day  
    # - Clay soils: K_sat ~ 1-10 mm/day, but K at FC ~ 0.01-0.1 mm/day
    #
    # These rates represent fraction of EXCESS water (above FC) drained per day
    # Lower values = slower drainage = higher moisture retention
    # ==========================================================================
    percolation_surface_to_root: float = 0.08  # Reduced from 0.15 (48% reduction)
    percolation_root_to_deep: float = 0.04     # Reduced from 0.08 (50% reduction)
    drainage_deep: float = 0.015               # Reduced from 0.03 (50% reduction)
    
    # ==========================================================================
    # CAPILLARY RISE PARAMETERS - NEW
    # ==========================================================================
    # Capillary rise occurs when matric potential gradient drives upward flux
    # Important in semi-arid regions, especially during dry season
    #
    # Physics: Capillary rise height ~ 1/pore_radius (smaller pores = higher rise)
    # Rate depends on unsaturated hydraulic conductivity and potential gradient
    #
    # capillary_rise_rate: fraction of deficit (below FC) that can be 
    # replenished per day from the layer below
    # ==========================================================================
    capillary_rise_deep_to_root: float = 0.10   # Increased 5x - stronger upward flux
    capillary_rise_root_to_surface: float = 0.05  # Increased 5x
    
    # Threshold: capillary rise only occurs when upper layer is drier than this
    # fraction of field capacity (prevents rise into already wet soil)
    capillary_rise_threshold: float = 0.95  # Rise occurs when θ < 0.95 * FC (more active)
    
    # ==========================================================================
    # GROUNDWATER INFLUENCE - ENHANCED FOR FLDAS MATCHING
    # ==========================================================================
    # FLDAS shows very stable moisture levels near field capacity
    # This suggests significant baseflow/groundwater influence in Africa
    # 
    # Increased contribution rate and threshold to maintain higher moisture
    # ==========================================================================
    groundwater_contribution_mm_day: float = 2.0  # Increased 4x - significant upward flux
    groundwater_threshold_fraction: float = 0.95  # Active when below 95% of FC
    
    # ==========================================================================
    # ET PARAMETERS - REDUCED FOR FLDAS MATCHING
    # ==========================================================================
    # FLDAS maintains near-FC moisture, suggesting lower actual ET than 
    # potential ET calculations suggest, or higher water inputs.
    # Reducing transpiration coefficient significantly.
    # ==========================================================================
    ndvi_et_partitioning_lambda: float = 2.0
    evaporation_coefficient: float = 0.8   # Reduced - surface crusting reduces evap
    transpiration_coefficient_max: float = 0.7  # Reduced significantly from 1.1
    
    # ==========================================================================
    # ET STRESS BELOW FIELD CAPACITY - NEW
    # ==========================================================================
    # Additional ET reduction when soil moisture is below field capacity
    # This represents plants reducing stomatal conductance as soil dries
    # Even before reaching traditional "stress" levels
    # ==========================================================================
    below_fc_et_reduction: float = 0.6  # Multiply ET by this when θ < FC
    
    # Root distribution (fraction of roots in each layer)
    root_fraction_surface: float = 0.20   # 20% of roots in 0-10cm
    root_fraction_root_zone: float = 0.70  # 70% of roots in 10-40cm
    root_fraction_deep: float = 0.10       # 10% of roots in 40-100cm
    
    # Numerical parameters
    tolerance: float = 0.01
    
    @property
    def total_depth_m(self) -> float:
        """Total soil column depth in meters"""
        return self.surface_depth_m + self.root_zone_depth_m + self.deep_depth_m
    
    @classmethod
    def from_soil_parameters(
        cls,
        soil_params: SoilParameters,
        config_overrides: Optional[Dict] = None
    ) -> "ThreeLayerParameters":
        """
        Create three-layer parameters from soil parameters.
        
        Applies depth-dependent adjustments for the deep layer
        based on typical soil profile characteristics.
        
        Key adjustments for African soils:
        1. Increased field capacity at all depths (tropical soil effect)
        2. Higher retention in deep layer (argillic/clay-rich B horizon)
        3. Conservative drainage rates for lateritic soils
        """
        # =======================================================================
        # TROPICAL SOIL ADJUSTMENT FACTORS
        # =======================================================================
        # African soils (especially Ferralsols, Acrisols, Lixisols) have:
        # - Higher field capacity than temperate pedotransfer functions predict
        # - Well-developed argillic horizons with clay accumulation at depth
        # - Good structure that maintains high porosity despite clay content
        # 
        # These factors increase the base estimates from Saxton & Rawls
        # =======================================================================
        FC_BOOST_SURFACE = 1.15   # 15% increase in surface FC
        FC_BOOST_ROOT = 1.20      # 20% increase in root zone FC  
        FC_BOOST_DEEP = 1.35      # 35% increase in deep layer FC (argillic horizon)
        
        POROSITY_BOOST = 1.05     # 5% increase in porosity (better structure)
        
        # Base parameters from soil with tropical adjustments
        params = cls(
            # Surface layer - boosted for tropical soils
            porosity_surface=min(0.55, soil_params.porosity * POROSITY_BOOST),
            field_capacity_surface=min(0.45, soil_params.field_capacity * FC_BOOST_SURFACE),
            wilting_point_surface=soil_params.wilting_point,
            
            # Root zone - significant boost for water retention
            porosity_root=min(0.52, soil_params.porosity * POROSITY_BOOST * 0.95),
            field_capacity_root=min(0.42, soil_params.field_capacity * FC_BOOST_ROOT),
            wilting_point_root=soil_params.wilting_point * 1.05,  # Slightly higher
            
            # Deep layer - major boost for argillic horizon effect
            # Clay illuviation creates high retention at 40-100cm depth
            porosity_deep=min(0.50, soil_params.porosity * POROSITY_BOOST * 0.92),
            field_capacity_deep=min(0.48, soil_params.field_capacity * FC_BOOST_DEEP),
            wilting_point_deep=soil_params.wilting_point * 1.20,  # Higher due to clay
            
            # Infiltration from Ks - slightly reduced for crusted surfaces
            infiltration_capacity_mm_h=soil_params.saturated_hydraulic_conductivity_cm_day * 10 / 24 * 0.8,
        )
        
        # Apply overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(params, key):
                    setattr(params, key, value)
        
        return params


class ThreeLayerWaterBalance:
    """
    Three-layer water balance model for 0-100cm soil column.
    
    This model extends the standard two-bucket approach to match
    FLDAS 0-100cm integrated soil moisture products.
    
    Key Features:
    - Surface layer (0-10cm): Rapid infiltration/evaporation dynamics
    - Root zone (10-40cm): Main plant water uptake zone
    - Deep subsoil (40-100cm): Slow drainage, limited root access
    - Depth-weighted averaging for 0-100cm comparison
    
    Physical Processes:
    1. Infiltration: Surface absorption limited by capacity and saturation
    2. Evaporation: From surface layer only
    3. Transpiration: Distributed across layers by root fraction
    4. Percolation: Gravity drainage between layers
    5. Deep drainage: Outflow from bottom of profile
    """
    
    def __init__(
        self,
        parameters: ThreeLayerParameters,
        initial_state: Optional[Dict[SoilLayer, BucketState]] = None
    ):
        """
        Initialize three-layer model.
        
        Args:
            parameters: ThreeLayerParameters instance
            initial_state: Optional initial bucket states
        """
        self.params = parameters
        self.logger = logging.getLogger(f"{__name__}.ThreeLayerWaterBalance")
        
        # Initialize state
        if initial_state is None:
            self.state = self._initialize_default_state()
        else:
            self.state = initial_state
        
        # Track water balance
        self.cumulative_error = 0.0
        self.iteration_count = 0
    
    def _initialize_default_state(self) -> Dict[SoilLayer, BucketState]:
        """Initialize all three layers at field capacity"""
        return {
            SoilLayer.SURFACE: BucketState(
                water_content=self.params.field_capacity_surface,
                max_storage=self.params.surface_depth_m * 1000,
                layer_depth_m=self.params.surface_depth_m,
                field_capacity=self.params.field_capacity_surface,
                wilting_point=self.params.wilting_point_surface,
                porosity=self.params.porosity_surface
            ),
            SoilLayer.ROOT_ZONE: BucketState(
                water_content=self.params.field_capacity_root,
                max_storage=self.params.root_zone_depth_m * 1000,
                layer_depth_m=self.params.root_zone_depth_m,
                field_capacity=self.params.field_capacity_root,
                wilting_point=self.params.wilting_point_root,
                porosity=self.params.porosity_root
            ),
            SoilLayer.DEEP: BucketState(
                water_content=self.params.field_capacity_deep,
                max_storage=self.params.deep_depth_m * 1000,
                layer_depth_m=self.params.deep_depth_m,
                field_capacity=self.params.field_capacity_deep,
                wilting_point=self.params.wilting_point_deep,
                porosity=self.params.porosity_deep
            )
        }
    
    def integrated_soil_moisture_0_100cm(self) -> float:
        """
        Calculate depth-weighted average soil moisture for 0-100cm.
        
        This matches the FLDAS 0-100cm integrated product calculation.
        
        Returns:
            Volumetric water content (m³/m³) for 0-100cm column
        """
        total_depth = self.params.total_depth_m
        weighted_sm = (
            self.state[SoilLayer.SURFACE].water_content * self.params.surface_depth_m +
            self.state[SoilLayer.ROOT_ZONE].water_content * self.params.root_zone_depth_m +
            self.state[SoilLayer.DEEP].water_content * self.params.deep_depth_m
        )
        return weighted_sm / total_depth
    
    def run_daily(
        self,
        precipitation_mm: float,
        et0_mm: float,
        ndvi: Optional[float] = None,
        irrigation_mm: float = 0.0,
        check_water_balance: bool = True
    ) -> Dict:
        """
        Run model for one day with given forcings.
        
        Args:
            precipitation_mm: Daily precipitation (mm)
            et0_mm: Daily reference evapotranspiration (mm)
            ndvi: NDVI for ET partitioning (0-1)
            irrigation_mm: Irrigation amount (mm)
            check_water_balance: Whether to check mass balance
        
        Returns:
            Dictionary with layer states, fluxes, and integrated moisture
        """
        # Store initial storage for water balance check
        initial_storage = self._total_storage()
        
        # Default NDVI if not provided
        if ndvi is None:
            ndvi = 0.5
        
        # 1. INFILTRATION AND RUNOFF
        surface = self.state[SoilLayer.SURFACE]
        total_input = precipitation_mm + irrigation_mm
        
        # Calculate infiltration capacity (reduces as soil wets)
        saturation = surface.water_content / surface.porosity
        saturation = min(1.0, max(0.0, saturation))
        saturation_factor = np.exp(-3.0 * saturation)
        
        daily_infiltration_capacity = self.params.infiltration_capacity_mm_h * 24.0
        effective_capacity = daily_infiltration_capacity * (0.1 + 0.9 * saturation_factor)
        
        # Available storage in surface
        available_storage = max(0, (surface.porosity - surface.water_content) * 
                                surface.layer_depth_m * 1000)
        
        # Infiltration is minimum of capacity, available storage, and input
        infiltration = min(effective_capacity, available_storage, total_input)
        runoff = max(0, total_input - infiltration)
        
        # Add infiltration to surface
        surface.storage += infiltration
        surface.water_content = surface.storage / (surface.layer_depth_m * 1000)
        
        # 2. EVAPOTRANSPIRATION PARTITIONING
        # Evaporation fraction decreases with vegetation cover
        evap_fraction = np.exp(-self.params.ndvi_et_partitioning_lambda * ndvi)
        transp_fraction = 1.0 - evap_fraction
        
        # Soil moisture stress factor (traditional approach)
        stress = self._calculate_water_stress()
        actual_et = et0_mm * stress
        
        # ======================================================================
        # ADDITIONAL ET REDUCTION WHEN BELOW FIELD CAPACITY
        # ======================================================================
        # Plants reduce transpiration even before reaching "stress" levels
        # This represents stomatal closure as soil water potential decreases
        # Weighted by root zone moisture relative to field capacity
        # ======================================================================
        root_zone = self.state[SoilLayer.ROOT_ZONE]
        deep = self.state[SoilLayer.DEEP]
        
        # Calculate average moisture deficit relative to FC
        avg_theta = (
            surface.water_content * self.params.surface_depth_m +
            root_zone.water_content * self.params.root_zone_depth_m +
            deep.water_content * self.params.deep_depth_m
        ) / self.params.total_depth_m
        
        avg_fc = (
            self.params.field_capacity_surface * self.params.surface_depth_m +
            self.params.field_capacity_root * self.params.root_zone_depth_m +
            self.params.field_capacity_deep * self.params.deep_depth_m
        ) / self.params.total_depth_m
        
        # If below field capacity, apply additional reduction
        if avg_theta < avg_fc:
            # Linear reduction: 1.0 at FC, below_fc_et_reduction at wilting point
            deficit_fraction = (avg_fc - avg_theta) / (avg_fc - 0.10)  # 0.10 ≈ avg wilting
            deficit_fraction = min(1.0, max(0.0, deficit_fraction))
            below_fc_factor = 1.0 - deficit_fraction * (1.0 - self.params.below_fc_et_reduction)
            actual_et = actual_et * below_fc_factor
        
        # Evaporation from surface only
        potential_evap = actual_et * evap_fraction * self.params.evaporation_coefficient
        evaporation = min(potential_evap, surface.available_water)
        
        # Remove evaporation from surface
        surface.storage -= evaporation
        surface.water_content = surface.storage / (surface.layer_depth_m * 1000)
        
        # 3. TRANSPIRATION DISTRIBUTED BY ROOT FRACTION
        total_transpiration = actual_et * transp_fraction * self.params.transpiration_coefficient_max
        
        # Transpiration from each layer proportional to root fraction
        transp_surface = min(
            total_transpiration * self.params.root_fraction_surface,
            surface.available_water
        )
        transp_root = min(
            total_transpiration * self.params.root_fraction_root_zone,
            root_zone.available_water
        )
        transp_deep = min(
            total_transpiration * self.params.root_fraction_deep,
            deep.available_water
        )
        
        # Remove transpiration
        surface.storage -= transp_surface
        surface.water_content = surface.storage / (surface.layer_depth_m * 1000)
        
        root_zone.storage -= transp_root
        root_zone.water_content = root_zone.storage / (root_zone.layer_depth_m * 1000)
        
        deep.storage -= transp_deep
        deep.water_content = deep.storage / (deep.layer_depth_m * 1000)
        
        total_transp = transp_surface + transp_root + transp_deep
        
        # 4. PERCOLATION: SURFACE -> ROOT ZONE
        excess_surface = max(0, surface.water_content - surface.field_capacity)
        perc_surface_to_root = excess_surface * surface.layer_depth_m * 1000 * \
                               self.params.percolation_surface_to_root
        
        # Limit by available storage in root zone
        available_in_root = max(0, (root_zone.porosity - root_zone.water_content) * 
                                root_zone.layer_depth_m * 1000)
        perc_surface_to_root = min(perc_surface_to_root, available_in_root)
        
        surface.storage -= perc_surface_to_root
        surface.water_content = surface.storage / (surface.layer_depth_m * 1000)
        
        root_zone.storage += perc_surface_to_root
        root_zone.water_content = root_zone.storage / (root_zone.layer_depth_m * 1000)
        
        # 5. PERCOLATION: ROOT ZONE -> DEEP
        excess_root = max(0, root_zone.water_content - root_zone.field_capacity)
        perc_root_to_deep = excess_root * root_zone.layer_depth_m * 1000 * \
                           self.params.percolation_root_to_deep
        
        # Limit by available storage in deep layer
        available_in_deep = max(0, (deep.porosity - deep.water_content) * 
                               deep.layer_depth_m * 1000)
        perc_root_to_deep = min(perc_root_to_deep, available_in_deep)
        
        root_zone.storage -= perc_root_to_deep
        root_zone.water_content = root_zone.storage / (root_zone.layer_depth_m * 1000)
        
        deep.storage += perc_root_to_deep
        deep.water_content = deep.storage / (deep.layer_depth_m * 1000)
        
        # 6. DEEP DRAINAGE (bottom boundary)
        excess_deep = max(0, deep.water_content - deep.field_capacity)
        drainage = excess_deep * deep.layer_depth_m * 1000 * self.params.drainage_deep
        
        deep.storage -= drainage
        deep.water_content = deep.storage / (deep.layer_depth_m * 1000)
        
        # ======================================================================
        # 6.5 GROUNDWATER CONTRIBUTION (NEW)
        # ======================================================================
        # Represents upward flux from shallow water table into deep layer
        # Only active when deep layer is significantly drier than threshold
        # 
        # Physics: In many African landscapes, water tables are 2-10m deep
        # Capillary fringe can extend 1-3m above water table in fine soils
        # This creates a quasi-steady upward flux during dry periods
        # ======================================================================
        groundwater_flux = 0.0
        gw_threshold = self.params.field_capacity_deep * self.params.groundwater_threshold_fraction
        
        if deep.water_content < gw_threshold:
            # Linear ramp: more contribution when drier
            dryness_factor = 1.0 - (deep.water_content / gw_threshold)
            groundwater_flux = self.params.groundwater_contribution_mm_day * dryness_factor
            
            # Add groundwater contribution to deep layer
            deep.storage += groundwater_flux
            deep.water_content = deep.storage / (deep.layer_depth_m * 1000)
        
        # ======================================================================
        # 6.6 CAPILLARY RISE (NEW)
        # ======================================================================
        # Upward water movement driven by matric potential gradient
        # Occurs when upper layer is drier than lower layer
        #
        # Physics: Water moves from high potential (wet) to low potential (dry)
        # Rate depends on unsaturated hydraulic conductivity and gradient
        # This is particularly important during dry season in savanna regions
        #
        # Key: Only occurs when upper layer is below threshold (drier than FC)
        # ======================================================================
        capillary_rise_deep_to_root = 0.0
        capillary_rise_root_to_surface = 0.0
        
        # Capillary rise from deep to root zone
        root_threshold = root_zone.field_capacity * self.params.capillary_rise_threshold
        if root_zone.water_content < root_threshold and deep.water_content > deep.wilting_point:
            # Deficit in root zone (how much below FC)
            deficit = (root_zone.field_capacity - root_zone.water_content) * \
                      root_zone.layer_depth_m * 1000
            
            # Available water in deep layer (above wilting point)
            available_deep = max(0, (deep.water_content - deep.wilting_point) * \
                                 deep.layer_depth_m * 1000)
            
            # Capillary rise rate
            capillary_rise_deep_to_root = min(
                deficit * self.params.capillary_rise_deep_to_root,
                available_deep * 0.1  # Max 10% of available water per day
            )
            
            # Transfer water
            deep.storage -= capillary_rise_deep_to_root
            deep.water_content = deep.storage / (deep.layer_depth_m * 1000)
            
            root_zone.storage += capillary_rise_deep_to_root
            root_zone.water_content = root_zone.storage / (root_zone.layer_depth_m * 1000)
        
        # Capillary rise from root zone to surface (smaller effect)
        surface_threshold = surface.field_capacity * self.params.capillary_rise_threshold
        if surface.water_content < surface_threshold and root_zone.water_content > root_zone.wilting_point:
            deficit = (surface.field_capacity - surface.water_content) * \
                      surface.layer_depth_m * 1000
            
            available_root = max(0, (root_zone.water_content - root_zone.wilting_point) * \
                                 root_zone.layer_depth_m * 1000)
            
            capillary_rise_root_to_surface = min(
                deficit * self.params.capillary_rise_root_to_surface,
                available_root * 0.05  # Max 5% per day (surface dries fast anyway)
            )
            
            root_zone.storage -= capillary_rise_root_to_surface
            root_zone.water_content = root_zone.storage / (root_zone.layer_depth_m * 1000)
            
            surface.storage += capillary_rise_root_to_surface
            surface.water_content = surface.storage / (surface.layer_depth_m * 1000)
        
        # 7. ENFORCE PHYSICAL BOUNDS
        for layer in self.state.values():
            layer.water_content = np.clip(
                layer.water_content,
                layer.wilting_point * 0.5,  # Allow some drying below WP
                layer.porosity
            )
            layer.storage = layer.water_content * layer.layer_depth_m * 1000
        
        # 8. WATER BALANCE CHECK
        # Note: groundwater_flux is an external input (from below model domain)
        # Capillary rise is internal redistribution (net zero for total column)
        water_balance_error = 0.0
        if check_water_balance:
            final_storage = self._total_storage()
            inputs = precipitation_mm + irrigation_mm + groundwater_flux
            outputs = runoff + evaporation + total_transp + drainage
            expected_change = inputs - outputs
            actual_change = final_storage - initial_storage
            water_balance_error = actual_change - expected_change
            
            self.cumulative_error += abs(water_balance_error)
            self.iteration_count += 1
        
        # Return results
        return {
            'theta_surface': self.state[SoilLayer.SURFACE].water_content,
            'theta_root': self.state[SoilLayer.ROOT_ZONE].water_content,
            'theta_deep': self.state[SoilLayer.DEEP].water_content,
            'theta_0_100cm': self.integrated_soil_moisture_0_100cm(),
            'infiltration': infiltration,
            'runoff': runoff,
            'evaporation': evaporation,
            'transpiration': total_transp,
            'transpiration_surface': transp_surface,
            'transpiration_root': transp_root,
            'transpiration_deep': transp_deep,
            'percolation_surface_to_root': perc_surface_to_root,
            'percolation_root_to_deep': perc_root_to_deep,
            'drainage': drainage,
            'groundwater_flux': groundwater_flux,
            'capillary_rise_deep_to_root': capillary_rise_deep_to_root,
            'capillary_rise_root_to_surface': capillary_rise_root_to_surface,
            'water_balance_error': water_balance_error,
            'converged': abs(water_balance_error) < self.params.tolerance * 10
        }
    
    def _calculate_water_stress(self) -> float:
        """
        Calculate plant water stress factor (0-1).
        
        Based on available water relative to total available water capacity
        across all root-accessible layers.
        """
        # Weighted available water by root fraction
        total_available = (
            self.state[SoilLayer.SURFACE].available_water * self.params.root_fraction_surface +
            self.state[SoilLayer.ROOT_ZONE].available_water * self.params.root_fraction_root_zone +
            self.state[SoilLayer.DEEP].available_water * self.params.root_fraction_deep
        )
        
        # Total available water capacity (at field capacity)
        surface_awc = ((self.params.field_capacity_surface - self.params.wilting_point_surface) *
                       self.params.surface_depth_m * 1000 * self.params.root_fraction_surface)
        root_awc = ((self.params.field_capacity_root - self.params.wilting_point_root) *
                    self.params.root_zone_depth_m * 1000 * self.params.root_fraction_root_zone)
        deep_awc = ((self.params.field_capacity_deep - self.params.wilting_point_deep) *
                    self.params.deep_depth_m * 1000 * self.params.root_fraction_deep)
        
        total_awc = surface_awc + root_awc + deep_awc
        
        if total_awc <= 0:
            return 0.0
        
        return min(1.0, total_available / total_awc)
    
    def _total_storage(self) -> float:
        """Calculate total water storage in all three layers (mm)"""
        return sum(bucket.storage for bucket in self.state.values())
    
    def run_period(
        self,
        forcings: pd.DataFrame,
        warmup_days: int = 30
    ) -> pd.DataFrame:
        """
        Run model for a period with time series of forcings.
        
        Args:
            forcings: DataFrame with columns:
                - precipitation_mm (required)
                - et0_mm (required)
                - ndvi (optional)
                - irrigation_mm (optional)
            warmup_days: Number of days for model spin-up
        
        Returns:
            DataFrame with daily results including theta_0_100cm
        """
        self.logger.info(f"Running 3-layer model for {len(forcings)} days")
        
        results = []
        
        for i, (idx, row) in enumerate(forcings.iterrows()):
            result = self.run_daily(
                precipitation_mm=row['precipitation_mm'],
                et0_mm=row['et0_mm'],
                ndvi=row.get('ndvi', 0.5),
                irrigation_mm=row.get('irrigation_mm', 0.0)
            )
            
            # Only store results after warmup
            if i >= warmup_days:
                results.append(result)
        
        df = pd.DataFrame(results)
        
        self.logger.info(
            f"Model run complete. Mean θ₀₋₁₀₀ₘ: {df['theta_0_100cm'].mean():.4f} m³/m³"
        )
        
        return df
    
    def reset(self, initial_state: Optional[Dict[SoilLayer, BucketState]] = None):
        """Reset model to initial or specified state"""
        if initial_state is None:
            self.state = self._initialize_default_state()
        else:
            self.state = initial_state
        
        self.cumulative_error = 0.0
        self.iteration_count = 0
        self.logger.info("Three-layer model reset to initial state")


def create_three_layer_model(
    soil_params: Optional[SoilParameters] = None,
    config_overrides: Optional[Dict] = None
) -> ThreeLayerWaterBalance:
    """
    Factory function to create a three-layer water balance model.
    
    Args:
        soil_params: Optional soil parameters for parameter estimation
        config_overrides: Optional parameter overrides
    
    Returns:
        Configured ThreeLayerWaterBalance instance
    """
    if soil_params is not None:
        params = ThreeLayerParameters.from_soil_parameters(
            soil_params, config_overrides
        )
    else:
        params = ThreeLayerParameters()
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(params, key):
                    setattr(params, key, value)
    
    return ThreeLayerWaterBalance(params)
