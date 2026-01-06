"""
Vertical Water Flux Module with Darcy-based flow and hysteresis.

This module implements physically-based vertical water movement including:
1. Darcy's law with unsaturated hydraulic conductivity
2. Van Genuchten-Mualem K(θ) relationships
3. Gradient-driven percolation and capillary rise
4. Hysteresis in water retention
5. Macropore drainage at near-saturation

References:
- Darcy, H. (1856). Les fontaines publiques de la ville de Dijon. Dalmont, Paris.
- Richards, L.A. (1931). Capillary conduction of liquids through porous mediums.
  Physics 1:318-333.
- Van Genuchten, M.Th. (1980). A closed-form equation for predicting the
  hydraulic conductivity of unsaturated soils.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import logging

from smps.physics.soil_hydraulics import (
    VanGenuchtenParameters,
    HysteresisParameters,
    HysteresisState,
    van_genuchten_psi_from_theta,
    van_genuchten_theta_from_psi,
    van_genuchten_mualem_K,
    darcy_flux_between_layers,
    get_effective_alpha,
)

logger = logging.getLogger(__name__)


@dataclass
class VerticalFluxParameters:
    """
    Parameters for vertical water flux calculations.
    """
    # Time stepping
    dt_seconds: float = 3600.0  # Internal timestep (1 hour default)

    # Macropore parameters
    macropore_enabled: bool = True
    macropore_threshold_saturation: float = 0.90  # Activate above 90% saturation
    macropore_conductivity_factor: float = 10.0  # K_macro = factor × K_sat

    # Hysteresis
    hysteresis_enabled: bool = False
    alpha_wetting_ratio: float = 2.0  # α_wet / α_dry

    # Capillary rise parameters
    capillary_rise_enabled: bool = True
    max_capillary_rise_m_day: float = 0.005  # 5 mm/day maximum

    # Water table (if present)
    water_table_depth_m: Optional[float] = None

    # Numerical stability
    max_flux_fraction: float = 0.5  # Max fraction of layer water per timestep
    convergence_tolerance: float = 1e-6


@dataclass
class LayerState:
    """
    State of a soil layer for flux calculations.
    """
    depth_top_m: float  # Depth to top of layer (m from surface)
    depth_bottom_m: float  # Depth to bottom of layer
    theta: float  # Current water content (m³/m³)
    theta_previous: float = None  # Previous timestep for hysteresis
    vg_params: VanGenuchtenParameters = None
    hysteresis_state: HysteresisState = HysteresisState.UNKNOWN

    def __post_init__(self):
        if self.theta_previous is None:
            self.theta_previous = self.theta

    @property
    def thickness_m(self) -> float:
        """Layer thickness (m)"""
        return self.depth_bottom_m - self.depth_top_m

    @property
    def depth_center_m(self) -> float:
        """Center depth of layer (m)"""
        return (self.depth_top_m + self.depth_bottom_m) / 2

    @property
    def elevation_m(self) -> float:
        """Elevation relative to surface (m, negative below surface)"""
        return -self.depth_center_m

    @property
    def storage_mm(self) -> float:
        """Water storage in layer (mm)"""
        return self.theta * self.thickness_m * 1000

    @property
    def psi_m(self) -> float:
        """Matric potential (m, negative for unsaturated)"""
        if self.vg_params is None:
            raise ValueError("VG parameters required for psi calculation")
        return van_genuchten_psi_from_theta(self.theta, self.vg_params)

    @property
    def hydraulic_head_m(self) -> float:
        """Total hydraulic head H = ψ + z (m)"""
        return self.psi_m + self.elevation_m

    @property
    def K_m_day(self) -> float:
        """Unsaturated hydraulic conductivity (m/day)"""
        if self.vg_params is None:
            raise ValueError("VG parameters required for K calculation")
        return van_genuchten_mualem_K(self.theta, self.vg_params)

    @property
    def saturation(self) -> float:
        """Relative saturation S = θ/θ_s"""
        if self.vg_params is None:
            return self.theta / 0.45  # Assume typical θ_s
        return self.theta / self.vg_params.theta_s

    def available_water_mm(self) -> float:
        """Water available above residual (mm)"""
        if self.vg_params is None:
            theta_r = 0.05
        else:
            theta_r = self.vg_params.theta_r
        return max(0, (self.theta - theta_r) * self.thickness_m * 1000)

    def space_available_mm(self) -> float:
        """Space available below saturation (mm)"""
        if self.vg_params is None:
            theta_s = 0.45
        else:
            theta_s = self.vg_params.theta_s
        return max(0, (theta_s - self.theta) * self.thickness_m * 1000)


def calculate_interlayer_K(
    K_upper: float,
    K_lower: float,
    method: str = "geometric"
) -> float:
    """
    Calculate effective hydraulic conductivity between layers.

    Options:
    - "geometric": sqrt(K1 × K2) - commonly used
    - "harmonic": 2×K1×K2/(K1+K2) - more conservative for layered soils
    - "arithmetic": (K1+K2)/2 - tends to overestimate
    - "upstream": Use upstream K (direction-dependent)

    Args:
        K_upper: K of upper layer (m/day)
        K_lower: K of lower layer (m/day)
        method: Averaging method

    Returns:
        Effective interlayer K (m/day)
    """
    if method == "geometric":
        return np.sqrt(K_upper * K_lower)
    elif method == "harmonic":
        return 2 * K_upper * K_lower / (K_upper + K_lower + 1e-15)
    elif method == "arithmetic":
        return (K_upper + K_lower) / 2
    else:
        # Upstream: use K of layer water is flowing from
        # (Need gradient info - default to geometric)
        return np.sqrt(K_upper * K_lower)


def darcy_flux_vertical(
    layer_upper: LayerState,
    layer_lower: LayerState,
    params: VerticalFluxParameters
) -> float:
    """
    Calculate Darcy flux between two vertically adjacent layers.

    q = -K × dH/dz

    Positive flux = downward (from upper to lower layer).

    Args:
        layer_upper: Upper soil layer
        layer_lower: Lower soil layer (below upper)
        params: Flux calculation parameters

    Returns:
        Darcy flux (m/day, positive = downward)
    """
    # Distance between layer centers
    # Positive (upper is less negative)
    dz = layer_upper.elevation_m - layer_lower.elevation_m

    if dz <= 0:
        logger.warning("Invalid layer geometry: upper layer below lower layer")
        return 0.0

    # Calculate hydraulic heads
    H_upper = layer_upper.hydraulic_head_m
    H_lower = layer_lower.hydraulic_head_m

    # Hydraulic gradient (positive if water flows downward)
    dH_dz = (H_upper - H_lower) / dz

    # Get effective K between layers
    K_upper = layer_upper.K_m_day
    K_lower = layer_lower.K_m_day
    K_eff = calculate_interlayer_K(K_upper, K_lower, method="geometric")

    # Darcy flux
    q = K_eff * dH_dz  # q > 0 means downward flow

    return q


def calculate_macropore_drainage(
    layer: LayerState,
    params: VerticalFluxParameters
) -> float:
    """
    Calculate macropore drainage for near-saturated conditions.

    Macropores (cracks, root channels, worm holes) bypass the soil
    matrix and drain rapidly when soil is near saturation.

    Args:
        layer: Soil layer
        params: Flux parameters

    Returns:
        Macropore drainage rate (m/day)
    """
    if not params.macropore_enabled:
        return 0.0

    saturation = layer.saturation

    if saturation < params.macropore_threshold_saturation:
        return 0.0

    # Macropore flow increases rapidly above threshold
    excess_saturation = saturation - params.macropore_threshold_saturation
    relative_excess = excess_saturation / \
        (1.0 - params.macropore_threshold_saturation)

    # Macropore K increases exponentially near saturation
    K_macro = (
        layer.vg_params.K_sat *
        params.macropore_conductivity_factor *
        relative_excess ** 2
    )

    # Assume unit gradient for rapid drainage
    return K_macro


def calculate_capillary_rise(
    layer_upper: LayerState,
    layer_lower: LayerState,
    params: VerticalFluxParameters
) -> float:
    """
    Calculate capillary rise from lower to upper layer.

    Capillary rise occurs when the matric potential gradient drives
    upward flow against gravity. Common in dry seasons with shallow
    water tables or wet subsoil.

    Args:
        layer_upper: Upper (drier) layer
        layer_lower: Lower (wetter) layer
        params: Flux parameters

    Returns:
        Capillary rise rate (m/day, positive = upward flow)
    """
    if not params.capillary_rise_enabled:
        return 0.0

    # Only calculate if upper layer is drier
    psi_upper = layer_upper.psi_m
    psi_lower = layer_lower.psi_m

    # Matric potential gradient (positive when upper is drier)
    psi_gradient = psi_upper - psi_lower

    if psi_gradient >= 0:
        # Upper layer is wetter or equal - no capillary rise
        return 0.0

    # Use full Darcy flux calculation
    # Negative psi_gradient means lower layer has higher (less negative) ψ
    q = darcy_flux_vertical(layer_upper, layer_lower, params)

    # If q is negative, it means upward flow
    if q >= 0:
        return 0.0  # No upward flow

    capillary_rise = -q  # Convert to positive (upward)

    # Limit capillary rise
    capillary_rise = min(capillary_rise, params.max_capillary_rise_m_day)

    return capillary_rise


def calculate_water_table_flux(
    bottom_layer: LayerState,
    params: VerticalFluxParameters
) -> float:
    """
    Calculate flux from/to water table at bottom boundary.

    If water table is present:
    - Drainage to water table if layer is wet
    - Capillary rise from water table if layer is dry

    Args:
        bottom_layer: Lowest soil layer
        params: Flux parameters

    Returns:
        Net flux at bottom (m/day, positive = downward/drainage)
    """
    if params.water_table_depth_m is None:
        # Free drainage boundary - use unit gradient
        return bottom_layer.K_m_day

    # Distance to water table
    depth_to_wt = params.water_table_depth_m - bottom_layer.depth_bottom_m

    if depth_to_wt <= 0:
        # Water table is at or above bottom of layer
        # Saturated conditions
        return 0.0

    # Calculate flux assuming saturated water table (ψ=0)
    H_layer = bottom_layer.hydraulic_head_m
    H_wt = -params.water_table_depth_m  # z coordinate of water table

    dz = bottom_layer.elevation_m - (-params.water_table_depth_m)

    if dz == 0:
        return 0.0

    dH_dz = (H_layer - H_wt) / dz

    # Use layer K (may need better estimation for flux to WT)
    return bottom_layer.K_m_day * dH_dz


@dataclass
class VerticalFluxResult:
    """Results from vertical flux calculation"""
    percolation_mm: List[float]  # Percolation between layers (mm/day)
    capillary_rise_mm: List[float]  # Capillary rise between layers (mm/day)
    macropore_drainage_mm: List[float]  # Macropore drainage per layer (mm/day)
    bottom_drainage_mm: float  # Drainage at bottom boundary (mm/day)
    net_flux_mm: List[float]  # Net flux into each layer (mm/day)
    layer_K_m_day: List[float]  # Hydraulic conductivity per layer


class VerticalFluxModel:
    """
    Complete vertical water flux model using Darcy's law.
    """

    def __init__(
        self,
        params: Optional[VerticalFluxParameters] = None
    ):
        """
        Initialize vertical flux model.

        Args:
            params: Model parameters
        """
        self.params = params or VerticalFluxParameters()
        self.logger = logging.getLogger(f"{__name__}.VerticalFluxModel")

    def calculate_fluxes(
        self,
        layers: List[LayerState],
        dt_days: float = 1.0
    ) -> VerticalFluxResult:
        """
        Calculate all vertical fluxes between layers.

        Args:
            layers: List of soil layers (ordered top to bottom)
            dt_days: Timestep (days)

        Returns:
            VerticalFluxResult with all flux components
        """
        n = len(layers)

        if n < 1:
            return VerticalFluxResult([], [], [], 0.0, [], [])

        # Initialize results
        percolation = []  # n-1 values (between adjacent layers)
        capillary_rise = []  # n-1 values
        macropore_drainage = [0.0] * n
        net_flux = [0.0] * n
        layer_K = [layer.K_m_day for layer in layers]

        # Calculate fluxes between adjacent layers
        for i in range(n - 1):
            upper = layers[i]
            lower = layers[i + 1]

            # Darcy flux (positive = downward)
            q = darcy_flux_vertical(upper, lower, self.params)

            if q >= 0:
                # Downward flow (percolation)
                # Limit to available water in upper layer
                max_flux = upper.available_water_mm() * self.params.max_flux_fraction / dt_days
                q_limited = min(q * 1000, max_flux) / 1000  # Convert mm to m

                # Also limit to space in lower layer
                max_space = lower.space_available_mm() / dt_days
                q_limited = min(q_limited * 1000, max_space) / 1000

                percolation.append(q_limited * 1000)  # Convert to mm/day
                capillary_rise.append(0.0)
            else:
                # Upward flow (capillary rise)
                cap_rise = calculate_capillary_rise(upper, lower, self.params)

                # Limit to available water in lower layer
                max_flux = lower.available_water_mm() * self.params.max_flux_fraction / dt_days
                cap_rise_limited = min(cap_rise * 1000, max_flux) / 1000

                percolation.append(0.0)
                capillary_rise.append(cap_rise_limited * 1000)  # mm/day

            # Macropore drainage from upper layer
            if self.params.macropore_enabled:
                macro = calculate_macropore_drainage(upper, self.params)
                macropore_drainage[i] = macro * 1000  # mm/day

        # Bottom boundary flux
        bottom_drainage = calculate_water_table_flux(layers[-1], self.params)

        # Limit bottom drainage
        max_bottom = layers[-1].available_water_mm() * \
            self.params.max_flux_fraction / dt_days
        bottom_drainage_mm = min(bottom_drainage * 1000, max_bottom)

        # Macropore from bottom layer
        if self.params.macropore_enabled:
            macro_bottom = calculate_macropore_drainage(
                layers[-1], self.params)
            macropore_drainage[-1] = macro_bottom * 1000

        # Calculate net flux for each layer
        # Layer 0: -percolation[0] - macro[0] + capillary[0]
        # Layer i: +percolation[i-1] - percolation[i] + cap[i] - cap[i-1] - macro[i]
        # Layer n-1: +percolation[n-2] - bottom_drainage + cap[n-2] - macro[n-1]

        for i in range(n):
            flux_in = 0.0
            flux_out = 0.0

            # From layer above (percolation into this layer)
            if i > 0:
                flux_in += percolation[i-1]

            # To layer below (percolation out of this layer)
            if i < n - 1:
                flux_out += percolation[i]

            # Capillary rise into this layer (from below)
            if i < n - 1:
                flux_in += capillary_rise[i]

            # Capillary rise out of this layer (to above)
            if i > 0:
                flux_out += capillary_rise[i-1]

            # Macropore drainage
            flux_out += macropore_drainage[i]

            # Bottom boundary
            if i == n - 1:
                flux_out += bottom_drainage_mm

            net_flux[i] = flux_in - flux_out

        return VerticalFluxResult(
            percolation_mm=percolation,
            capillary_rise_mm=capillary_rise,
            macropore_drainage_mm=macropore_drainage,
            bottom_drainage_mm=bottom_drainage_mm,
            net_flux_mm=net_flux,
            layer_K_m_day=layer_K
        )

    def apply_fluxes(
        self,
        layers: List[LayerState],
        result: VerticalFluxResult,
        dt_days: float = 1.0
    ) -> List[LayerState]:
        """
        Apply calculated fluxes to update layer water contents.

        Args:
            layers: Soil layers
            result: Flux calculation result
            dt_days: Timestep (days)

        Returns:
            Updated layers
        """
        for i, layer in enumerate(layers):
            # Update previous theta for hysteresis
            layer.theta_previous = layer.theta

            # Apply net flux
            d_storage = result.net_flux_mm[i] * dt_days  # mm
            d_theta = d_storage / (layer.thickness_m * 1000)

            new_theta = layer.theta + d_theta

            # Enforce bounds
            layer.theta = np.clip(
                new_theta,
                layer.vg_params.theta_r * 1.01,
                layer.vg_params.theta_s * 0.99
            )

            # Update hysteresis state
            if layer.theta > layer.theta_previous:
                layer.hysteresis_state = HysteresisState.WETTING
            elif layer.theta < layer.theta_previous:
                layer.hysteresis_state = HysteresisState.DRYING

        return layers


def compare_simple_vs_darcy_percolation(
    theta_upper: float,
    theta_lower: float,
    theta_FC: float,
    vg_params: VanGenuchtenParameters,
    layer_thickness_m: float = 0.10,
    simple_rate: float = 0.15
) -> Dict[str, float]:
    """
    Compare simple rate-based percolation vs Darcy's law.

    Simple approach:
        perc = (θ - FC) × thickness × rate   if θ > FC

    Darcy approach:
        perc = -K(θ) × dH/dz

    Args:
        theta_upper: Upper layer water content
        theta_lower: Lower layer water content
        theta_FC: Field capacity
        vg_params: Van Genuchten parameters
        layer_thickness_m: Layer thickness
        simple_rate: Simple model percolation rate (1/day)

    Returns:
        Dictionary comparing both approaches
    """
    # Simple approach
    if theta_upper > theta_FC:
        excess = (theta_upper - theta_FC) * layer_thickness_m * 1000  # mm
        simple_perc = excess * simple_rate
    else:
        simple_perc = 0.0

    # Darcy approach
    upper_layer = LayerState(
        depth_top_m=0.0,
        depth_bottom_m=layer_thickness_m,
        theta=theta_upper,
        vg_params=vg_params
    )
    lower_layer = LayerState(
        depth_top_m=layer_thickness_m,
        depth_bottom_m=2 * layer_thickness_m,
        theta=theta_lower,
        vg_params=vg_params
    )

    params = VerticalFluxParameters()
    q = darcy_flux_vertical(upper_layer, lower_layer, params)
    darcy_perc = max(0, q * 1000)  # mm/day, positive = downward

    return {
        'theta_upper': theta_upper,
        'theta_lower': theta_lower,
        'simple_percolation_mm_day': simple_perc,
        'darcy_percolation_mm_day': darcy_perc,
        'difference_mm_day': darcy_perc - simple_perc,
        'ratio': darcy_perc / (simple_perc + 0.01),
        'K_upper_m_day': upper_layer.K_m_day,
        'psi_upper_m': upper_layer.psi_m,
        'psi_lower_m': lower_layer.psi_m,
        'hydraulic_gradient': (upper_layer.hydraulic_head_m - lower_layer.hydraulic_head_m) / layer_thickness_m
    }
