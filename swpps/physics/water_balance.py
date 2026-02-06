"""
Tension-Space Water Balance Model for SWPPS.

This model performs water balance calculations in matric potential (ψ) space
rather than volumetric water content (θ) space. The key advantage is that
the outputs are directly in the units used for irrigation decisions.

Water Balance Equation:
    dS/dt = P - E - T - D - R

Where:
    S = soil water storage (mm)
    P = precipitation (mm/day)
    E = soil evaporation (mm/day)
    T = transpiration (mm/day)
    D = deep drainage (mm/day)
    R = surface runoff (mm/day)

The model maintains water balance in mm (via θ) but reports outputs in kPa (via ψ).

Enhancements:
- FAO-56 dual crop coefficient ET partitioning
- Tropical soil corrections for African soils
- Macropore flow and preferential infiltration
- Capillary rise from water table

References:
- Allen et al. (1998) FAO-56: Crop evapotranspiration
- Van Genuchten (1980) Closed-form equation for hydraulic conductivity
- Hodnett & Tomasella (2002) Tropical PTFs
"""

import numpy as np
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple
import logging

from swpps.core.types import (
    VanGenuchtenParams,
    PhysicsModelOutput,
)
from swpps.physics.van_genuchten import (
    water_content_from_potential,
    potential_from_water_content,
    hydraulic_conductivity_from_content,
    estimate_van_genuchten_params,
)
from swpps.physics.evapotranspiration import (
    CropCoefficients,
    compute_et_partitioning,
)
from swpps.physics.tropical import (
    TropicalSoilCorrections,
    partition_infiltration,
)

logger = logging.getLogger("swpps.physics.water_balance")


@dataclass
class LayerConfig:
    """Configuration for a single soil layer."""
    depth_top_m: float
    depth_bottom_m: float
    van_genuchten: VanGenuchtenParams

    @property
    def thickness_m(self) -> float:
        return self.depth_bottom_m - self.depth_top_m

    @property
    def thickness_mm(self) -> float:
        return self.thickness_m * 1000


@dataclass
class LayerState:
    """Current state of a soil layer."""
    config: LayerConfig
    theta: float  # Current volumetric water content

    @property
    def psi_kpa(self) -> float:
        """Current matric potential."""
        return potential_from_water_content(self.theta, self.config.van_genuchten)

    @property
    def storage_mm(self) -> float:
        """Water storage in mm."""
        return self.theta * self.config.thickness_mm

    @property
    def available_water_mm(self) -> float:
        """Plant available water above wilting point."""
        wp = self.config.van_genuchten.theta_wp
        return max(0, (self.theta - wp) * self.config.thickness_mm)

    @property
    def deficit_to_fc_mm(self) -> float:
        """Water deficit to reach field capacity."""
        fc = self.config.van_genuchten.theta_fc
        return max(0, (fc - self.theta) * self.config.thickness_mm)

    def add_water(self, amount_mm: float) -> float:
        """Add water to layer, return excess."""
        max_add = (self.config.van_genuchten.theta_s -
                   self.theta) * self.config.thickness_mm
        actual_add = min(amount_mm, max_add)
        self.theta += actual_add / self.config.thickness_mm
        return amount_mm - actual_add

    def remove_water(self, amount_mm: float, min_theta: Optional[float] = None) -> float:
        """Remove water from layer, return actual removed."""
        if min_theta is None:
            min_theta = self.config.van_genuchten.theta_r
        max_remove = (self.theta - min_theta) * self.config.thickness_mm
        actual_remove = min(amount_mm, max(0, max_remove))
        self.theta -= actual_remove / self.config.thickness_mm
        return actual_remove


@dataclass
class WaterBalanceConfig:
    """Configuration for water balance model."""
    # Layer configurations
    layers: List[LayerConfig]

    # Root distribution (fraction in each layer)
    root_fractions: Optional[List[float]] = None

    # Crop/vegetation parameters
    crop_coefficients: Optional[CropCoefficients] = None
    planting_doy: Optional[int] = None
    default_crop_coefficient: float = 0.8  # Fallback Kc

    # Runoff (SCS curve number method)
    curve_number: float = 70.0

    # Drainage parameters
    drainage_coefficient: float = 0.5  # Fraction of excess water that drains per day

    # Tropical soil corrections (optional)
    tropical_corrections: Optional[TropicalSoilCorrections] = None

    # Macropore flow
    enable_macropore_flow: bool = False
    precip_duration_hr: float = 4.0  # Default storm duration

    # Capillary rise from water table
    water_table_depth_m: Optional[float] = None  # None = no water table

    # Initial conditions
    initial_psi_kpa: float = -50.0  # Initial matric potential

    def __post_init__(self):
        # Default root distribution if not provided
        if self.root_fractions is None:
            n_layers = len(self.layers)
            if n_layers == 1:
                self.root_fractions = [1.0]
            elif n_layers == 2:
                self.root_fractions = [0.6, 0.4]
            elif n_layers == 3:
                self.root_fractions = [0.5, 0.35, 0.15]
            else:
                # Exponential decrease with depth
                self.root_fractions = [0.5 ** i for i in range(n_layers)]
                total = sum(self.root_fractions)
                self.root_fractions = [r / total for r in self.root_fractions]


class TensionSpaceWaterBalance:
    """
    Water balance model that operates in tension space.

    The model maintains internal water content (θ) for flux calculations
    but reports outputs as matric potential (ψ) for direct use in
    irrigation decisions.
    """

    def __init__(self, config: WaterBalanceConfig):
        self.config = config
        self.layers: List[LayerState] = []
        self._initialize_layers()

    def _initialize_layers(self):
        """Initialize layer states from config."""
        self.layers = []
        for layer_config in self.config.layers:
            # Initialize each layer at the specified initial potential
            theta = water_content_from_potential(
                self.config.initial_psi_kpa,
                layer_config.van_genuchten
            )
            self.layers.append(LayerState(config=layer_config, theta=theta))

    def reset(self, initial_psi_kpa: Optional[float] = None):
        """Reset model state."""
        if initial_psi_kpa is not None:
            self.config.initial_psi_kpa = initial_psi_kpa
        self._initialize_layers()

    def step(
        self,
        current_date: date,
        precipitation_mm: float,
        et0_mm: float,
        ndvi: Optional[float] = None,
        irrigation_mm: float = 0.0,
    ) -> PhysicsModelOutput:
        """
        Perform one day water balance step.

        Args:
            current_date: Current simulation date
            precipitation_mm: Daily precipitation (mm)
            et0_mm: Reference evapotranspiration (mm)
            ndvi: Normalized Difference Vegetation Index (optional)
            irrigation_mm: Irrigation applied (mm)

        Returns:
            PhysicsModelOutput with updated potentials and fluxes
        """
        # Store initial storage for balance check
        initial_storage = sum(l.storage_mm for l in self.layers)

        # Total water input
        total_input_mm = precipitation_mm + irrigation_mm

        # 1. Calculate runoff (SCS curve number method)
        runoff_mm = self._calculate_runoff(total_input_mm)
        infiltration_mm = total_input_mm - runoff_mm

        # 2. Partition infiltration (matrix vs macropore)
        macropore_mm = 0.0
        if self.config.enable_macropore_flow and infiltration_mm > 0:
            vg = self.layers[0].config.van_genuchten
            infil_result = partition_infiltration(
                precip_mm=infiltration_mm,
                precip_duration_hr=self.config.precip_duration_hr,
                Ksat_mm_day=vg.K_sat,
                theta_current=self.layers[0].theta,
                theta_sat=vg.theta_s,
                tropical_corrections=self.config.tropical_corrections,
            )
            matrix_infiltration = infil_result['matrix_mm']
            macropore_mm = infil_result['macropore_mm']
            runoff_mm += infil_result['runoff_mm']
        else:
            matrix_infiltration = infiltration_mm

        # 3. Infiltration into surface layer (matrix flow)
        excess_mm = self.layers[0].add_water(matrix_infiltration)

        # 4. Macropore bypass to deeper layers
        if macropore_mm > 0 and len(self.layers) > 1:
            # Macropore flow bypasses surface, goes to subsurface
            bypass_excess = self.layers[-1].add_water(macropore_mm * 0.7)
            if len(self.layers) > 2:
                self.layers[1].add_water(macropore_mm * 0.3)
            excess_mm += bypass_excess

        # 5. Percolation through layers
        drainage_mm = self._calculate_drainage(excess_mm)

        # 6. Calculate ET with FAO-56 dual coefficient
        et_result = self._calculate_et_fao56(et0_mm, ndvi, current_date)
        evaporation_mm = et_result['evaporation_mm']
        transpiration_mm = et_result['transpiration_mm']
        Ks = et_result.get('Ks', 1.0)

        # 7. Remove water for evaporation (from surface)
        actual_evap = self.layers[0].remove_water(evaporation_mm)

        # 8. Remove water for transpiration (from root zone)
        actual_transp = self._remove_transpiration(transpiration_mm)

        # 9. Capillary rise (if water table present)
        capillary_rise_mm = 0.0
        if self.config.water_table_depth_m is not None:
            capillary_rise_mm = self._calculate_capillary_rise()

        # 10. Calculate water balance error
        final_storage = sum(l.storage_mm for l in self.layers)
        storage_change = final_storage - initial_storage
        input_total = total_input_mm + capillary_rise_mm
        output_total = runoff_mm + drainage_mm + actual_evap + actual_transp
        balance_error = storage_change - (input_total - output_total)

        # 11. Build output with potentials (the key innovation!)
        return PhysicsModelOutput(
            date=current_date,
            psi_surface_kpa=self.layers[0].psi_kpa,
            psi_root_kpa=self._get_root_zone_potential(),
            psi_deep_kpa=self.layers[-1].psi_kpa if len(
                self.layers) > 2 else None,
            precipitation_mm=precipitation_mm,
            infiltration_mm=infiltration_mm,
            runoff_mm=runoff_mm,
            evaporation_mm=actual_evap,
            transpiration_mm=actual_transp,
            drainage_mm=drainage_mm,
            water_balance_error_mm=balance_error,
            converged=abs(balance_error) < 1.0,
        )

    def _calculate_runoff(self, precipitation_mm: float) -> float:
        """Calculate runoff using SCS curve number method."""
        CN = self.config.curve_number

        # Maximum retention (mm)
        S = 25400.0 / CN - 254.0

        # Initial abstraction (typically 0.2*S)
        Ia = 0.2 * S

        if precipitation_mm <= Ia:
            return 0.0

        # SCS runoff equation
        runoff = ((precipitation_mm - Ia) ** 2) / (precipitation_mm - Ia + S)

        return max(0.0, runoff)

    def _calculate_et_fao56(
        self,
        et0_mm: float,
        ndvi: Optional[float],
        current_date: date,
    ) -> Dict[str, float]:
        """
        Calculate ET using FAO-56 dual crop coefficient.

        Args:
            et0_mm: Reference evapotranspiration
            ndvi: NDVI value (optional)
            current_date: Current date for growth stage

        Returns:
            Dict with evaporation_mm, transpiration_mm, Kcb, Ks, Ke
        """
        # Get soil state for stress calculation
        theta_surface = self.layers[0].theta
        theta_root = self._get_root_zone_theta()
        vg = self.layers[0].config.van_genuchten

        # Use FAO-56 module
        result = compute_et_partitioning(
            et0_mm=et0_mm,
            ndvi=ndvi,
            crop_coef=self.config.crop_coefficients,
            day_of_year=current_date.timetuple().tm_yday if current_date else None,
            planting_doy=self.config.planting_doy,
            theta_surface=theta_surface,
            theta_root=theta_root,
            theta_fc=vg.theta_fc,
            theta_wp=vg.theta_wp,
        )

        return result

    def _get_root_zone_theta(self) -> float:
        """Get weighted average water content in root zone."""
        total_weight = 0.0
        weighted_theta = 0.0

        for layer, root_frac in zip(self.layers, self.config.root_fractions):
            if root_frac > 0:
                weighted_theta += layer.theta * root_frac
                total_weight += root_frac

        if total_weight > 0:
            return weighted_theta / total_weight
        return self.layers[0].theta

    def _calculate_capillary_rise(self) -> float:
        """
        Calculate capillary rise from water table.

        Uses simplified approach based on water table depth
        and soil hydraulic properties.
        """
        if self.config.water_table_depth_m is None:
            return 0.0

        wt_depth = self.config.water_table_depth_m
        bottom_layer = self.layers[-1]
        vg = bottom_layer.config.van_genuchten

        # Distance from bottom of root zone to water table
        root_zone_bottom = bottom_layer.config.depth_bottom_m
        distance_to_wt = wt_depth - root_zone_bottom

        if distance_to_wt <= 0:
            # Water table within root zone - no capillary rise needed
            return 0.0

        # Capillary rise flux depends on:
        # 1. Distance to water table (decreases with depth)
        # 2. Unsaturated conductivity
        # 3. Gradient (dψ/dz)

        # Simplified: exponential decay with depth
        # Maximum rise ~5 mm/day for shallow water tables
        max_rise = 5.0  # mm/day
        decay_factor = 0.5  # per meter

        rise_potential = max_rise * np.exp(-decay_factor * distance_to_wt)

        # Limited by capacity of bottom layer
        deficit = (vg.theta_fc - bottom_layer.theta) * \
            bottom_layer.config.thickness_mm

        actual_rise = min(rise_potential, deficit)

        # Add to bottom layer
        if actual_rise > 0:
            bottom_layer.add_water(actual_rise)

        return actual_rise

    def _calculate_drainage(self, initial_excess_mm: float) -> float:
        """Calculate drainage between layers and out of bottom."""
        excess = initial_excess_mm
        total_drainage = 0.0

        for i, layer in enumerate(self.layers):
            # Add excess from above
            new_excess = layer.add_water(excess)

            # Calculate gravity drainage if above field capacity
            vg = layer.config.van_genuchten
            if layer.theta > vg.theta_fc:
                excess_theta = layer.theta - vg.theta_fc

                # Drainage rate depends on hydraulic conductivity
                K = hydraulic_conductivity_from_content(layer.theta, vg)
                drainage_rate = min(
                    excess_theta, K * self.config.drainage_coefficient / 1000)

                drainage_mm = drainage_rate * layer.config.thickness_mm
                layer.theta -= drainage_rate

                if i < len(self.layers) - 1:
                    excess = drainage_mm + new_excess
                else:
                    total_drainage = drainage_mm + new_excess
            else:
                excess = new_excess
                if i == len(self.layers) - 1:
                    total_drainage = excess

        return total_drainage

    def _get_crop_coefficient(self, ndvi: Optional[float]) -> float:
        """Get crop coefficient from NDVI or default."""
        if ndvi is None:
            return self.config.default_crop_coefficient

        # Linear interpolation: NDVI 0.1 -> Kc 0.2, NDVI 0.9 -> Kc 1.2
        Kc = 0.2 + 1.0 * np.clip((ndvi - 0.1) / 0.8, 0, 1)
        return Kc

    def _partition_et(
        self,
        et_actual_mm: float,
        ndvi: Optional[float] = None
    ) -> Tuple[float, float]:
        """Partition actual ET into evaporation and transpiration."""
        # Vegetation fraction from NDVI
        if ndvi is not None:
            veg_frac = np.clip((ndvi - 0.1) / 0.8, 0, 1)
        else:
            veg_frac = 0.5

        # Transpiration fraction increases with vegetation cover
        transp_frac = veg_frac ** 0.5

        transpiration_mm = et_actual_mm * transp_frac
        evaporation_mm = et_actual_mm * (1 - transp_frac)

        return evaporation_mm, transpiration_mm

    def _remove_transpiration(self, transpiration_mm: float) -> float:
        """Remove transpiration from root zone according to root distribution."""
        total_removed = 0.0
        remaining = transpiration_mm

        for i, (layer, root_frac) in enumerate(
            zip(self.layers, self.config.root_fractions)
        ):
            # Amount this layer should supply
            target = transpiration_mm * root_frac

            # Limited by water availability (above wilting point)
            actual = layer.remove_water(
                target, layer.config.van_genuchten.theta_wp)
            total_removed += actual

            # If this layer couldn't supply enough, try to get more from others
            remaining -= actual

        # Redistribute remaining demand to wetter layers
        if remaining > 0.1:
            for layer in sorted(self.layers, key=lambda l: l.theta, reverse=True):
                if remaining <= 0:
                    break
                actual = layer.remove_water(
                    remaining, layer.config.van_genuchten.theta_wp)
                total_removed += actual
                remaining -= actual

        return total_removed

    def _get_root_zone_potential(self) -> float:
        """Get weighted average potential in root zone."""
        total_weight = 0.0
        weighted_psi = 0.0

        for layer, root_frac in zip(self.layers, self.config.root_fractions):
            if root_frac > 0:
                weighted_psi += layer.psi_kpa * root_frac
                total_weight += root_frac

        if total_weight > 0:
            return weighted_psi / total_weight
        return self.layers[0].psi_kpa

    def run_period(
        self,
        dates: List[date],
        precipitation: List[float],
        et0: List[float],
        ndvi: Optional[List[float]] = None,
        irrigation: Optional[List[float]] = None,
        warmup_days: int = 30,
    ) -> List[PhysicsModelOutput]:
        """
        Run model for a period of time.

        Args:
            dates: List of dates
            precipitation: Daily precipitation (mm)
            et0: Reference ET (mm)
            ndvi: NDVI values (optional)
            irrigation: Irrigation amounts (optional)
            warmup_days: Number of days for model warmup

        Returns:
            List of PhysicsModelOutput for each day
        """
        n_days = len(dates)

        # Prepare optional inputs
        if ndvi is None:
            ndvi = [None] * n_days
        if irrigation is None:
            irrigation = [0.0] * n_days

        # Run simulation
        results = []
        for i in range(n_days):
            output = self.step(
                current_date=dates[i],
                precipitation_mm=precipitation[i],
                et0_mm=et0[i],
                ndvi=ndvi[i],
                irrigation_mm=irrigation[i],
            )
            results.append(output)

        # Discard warmup period
        if warmup_days > 0 and len(results) > warmup_days:
            return results[warmup_days:]

        return results


def create_water_balance_model(
    sand_percent: float,
    clay_percent: float,
    organic_matter_percent: float = 2.0,
    n_layers: int = 3,
    max_depth_m: float = 1.0,
    initial_psi_kpa: float = -50.0,
) -> TensionSpaceWaterBalance:
    """
    Create a water balance model from soil texture.

    Args:
        sand_percent: Sand content (%)
        clay_percent: Clay content (%)
        organic_matter_percent: Organic matter (%)
        n_layers: Number of soil layers
        max_depth_m: Total soil depth (m)
        initial_psi_kpa: Initial matric potential

    Returns:
        Configured TensionSpaceWaterBalance model
    """
    # Estimate Van Genuchten parameters
    vg_params = estimate_van_genuchten_params(
        sand_percent=sand_percent,
        clay_percent=clay_percent,
        organic_matter_percent=organic_matter_percent,
    )

    # Create layer configurations
    layer_thickness = max_depth_m / n_layers
    layers = []

    for i in range(n_layers):
        depth_top = i * layer_thickness
        depth_bottom = (i + 1) * layer_thickness

        layers.append(LayerConfig(
            depth_top_m=depth_top,
            depth_bottom_m=depth_bottom,
            van_genuchten=vg_params,
        ))

    # Create config
    config = WaterBalanceConfig(
        layers=layers,
        initial_psi_kpa=initial_psi_kpa,
    )

    return TensionSpaceWaterBalance(config)
