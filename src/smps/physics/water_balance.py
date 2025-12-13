"""
Implementation of two-bucket water balance model.
Based on soil physics principles with numerical stability guarantees.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from datetime import date, timedelta
import logging

from smps.core.config import get_config
from smps.core.types import (
    SoilParameters, SoilLayer, PhysicsPriorResult,
    SoilMoistureVWC, PrecipitationMm, ET0Mm
)
from smps.core.exceptions import (
    PhysicsModelError, WaterBalanceError, ConvergenceError, ParameterError
)
from smps.core.config import SmpsConfig
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
       """Calculate transpiration reduction due to soil moisture stress"""
       root_state = self.state[SoilLayer.ROOT_ZONE]

       # FAO-56: Transpiration stress when soil moisture is below threshold
       # Threshold is often around 50% of available water
       threshold = (
           self.params.wilting_point_root +
           0.5 * (self.params.field_capacity_root - self.params.wilting_point_root)
       )

       if root_state.water_content >= threshold:
           return 1.0

       # Linear reduction below threshold
       rew = (
           (root_state.water_content - self.params.wilting_point_root) /
           (threshold - self.params.wilting_point_root)
       )

       return max(0.0, min(1.0, rew))

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


