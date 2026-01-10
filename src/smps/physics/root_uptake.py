"""
Root Water Uptake Model with pressure head-based stress function.

This module implements physically-based root water uptake including:
1. Feddes stress function using matric potential
2. Dynamic root distribution that responds to moisture
3. Compensatory uptake from wet layers
4. Hydraulic lift (nighttime redistribution)

References:
- Feddes, R.A. et al. (1978). Simulation of field water use and crop yield.
- Jarvis, N.J. (1989). A simple empirical model of root water uptake.
- Šimůnek, J. and Hopmans, J.W. (2009). Modeling compensated root water
  and nutrient uptake. Ecological Modelling.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import logging

from smps.physics.soil_hydraulics import (
    VanGenuchtenParameters,
    FeddesParameters,
    van_genuchten_psi_from_theta,
    van_genuchten_theta_from_psi,
    van_genuchten_mualem_K,
    feddes_stress_factor,
    s_shape_stress_factor,
)

logger = logging.getLogger(__name__)


@dataclass
class RootDistributionParameters:
    """
    Parameters for root distribution and dynamics.

    Root density typically follows an exponential or linear decrease with depth.
    Roots can also redistribute toward moisture (root plasticity).

    v2.2 Deep Layer Improvements:
    - Increased default max_depth for deeper rooting crops
    - Lower beta for better deep root representation
    - Higher minimum root fraction in deep layers
    - Enhanced compensation from deep layers during drought
    """
    # Static root distribution type
    distribution_type: str = "exponential"  # "exponential", "linear", "uniform"

    # Exponential decay parameter (1/m)
    # Lower values = more roots at depth (changed from 5.0 to 3.5)
    # β=3.5 gives ~30% of roots below 30cm vs ~5% with β=5.0
    beta: float = 3.5

    # Maximum rooting depth (m) - increased for deep-rooted crops
    # Maize roots can reach 1.5-2.0m, wheat 1.0-1.5m
    max_depth: float = 1.5

    # Root length density at surface (m/m³)
    RLD_surface: float = 5000.0

    # Root plasticity parameters
    plasticity_enabled: bool = True
    plasticity_rate: float = 0.15  # Increased from 0.1 for faster adaptation

    # Minimum root fraction in any layer (prevents complete abandonment)
    # Increased from 0.05 to ensure deep layers have meaningful root presence
    min_root_fraction: float = 0.08

    # Deep layer compensation enhancement
    deep_compensation_enabled: bool = True
    deep_compensation_factor: float = 1.5  # Extra compensation from deep layers
    # Layers below this depth get enhanced compensation
    deep_layer_threshold_m: float = 0.40


@dataclass
class SoilLayer:
    """Soil layer for root uptake calculations"""
    depth_top: float  # Top of layer (m, from surface)
    depth_bottom: float  # Bottom of layer (m, from surface)
    theta: float  # Volumetric water content (m³/m³)
    vg_params: VanGenuchtenParameters
    root_fraction: float = 0.0  # Fraction of roots in this layer

    @property
    def thickness(self) -> float:
        """Layer thickness (m)"""
        return self.depth_bottom - self.depth_top

    @property
    def depth_center(self) -> float:
        """Center depth of layer (m)"""
        return (self.depth_top + self.depth_bottom) / 2

    @property
    def psi(self) -> float:
        """Matric potential (m)"""
        return van_genuchten_psi_from_theta(self.theta, self.vg_params)

    @property
    def K(self) -> float:
        """Unsaturated hydraulic conductivity (m/day)"""
        return van_genuchten_mualem_K(self.theta, self.vg_params)


def calculate_static_root_distribution(
    layers: List[SoilLayer],
    params: RootDistributionParameters
) -> List[float]:
    """
    Calculate static root distribution based on exponential or linear model.

    Exponential model (Gale & Grigal, 1987):
        Y = 1 - β^d

    where Y is cumulative root fraction to depth d, β is shape parameter.

    Args:
        layers: List of soil layers
        params: Root distribution parameters

    Returns:
        List of root fractions for each layer
    """
    root_fractions = []

    if params.distribution_type == "exponential":
        # Exponential decay
        for layer in layers:
            # Cumulative root fraction at top and bottom
            if layer.depth_top >= params.max_depth:
                frac = 0.0
            else:
                # Exponential decay: f(z) = β × exp(-β × z)
                # Integrate over layer thickness
                z_top = min(layer.depth_top, params.max_depth)
                z_bot = min(layer.depth_bottom, params.max_depth)

                # Cumulative fraction
                cum_top = 1.0 - np.exp(-params.beta * z_top)
                cum_bot = 1.0 - np.exp(-params.beta * z_bot)

                frac = cum_bot - cum_top

            root_fractions.append(max(params.min_root_fraction, frac))

    elif params.distribution_type == "linear":
        # Linear decrease with depth
        for layer in layers:
            if layer.depth_bottom <= params.max_depth:
                z_mid = layer.depth_center
                frac = max(0, 1.0 - z_mid / params.max_depth)
            else:
                frac = 0.0
            root_fractions.append(max(params.min_root_fraction, frac))

    else:  # uniform
        n_active = sum(1 for l in layers if l.depth_center <= params.max_depth)
        frac = 1.0 / max(1, n_active)
        root_fractions = [frac if l.depth_center <= params.max_depth
                          else params.min_root_fraction for l in layers]

    # Normalize to sum to 1
    total = sum(root_fractions)
    if total > 0:
        root_fractions = [f / total for f in root_fractions]

    return root_fractions


def update_dynamic_root_distribution(
    layers: List[SoilLayer],
    current_fractions: List[float],
    params: RootDistributionParameters,
    feddes_params: FeddesParameters
) -> List[float]:
    """
    Update root distribution based on soil moisture (root plasticity).

    Roots grow toward layers with favorable water status.

    Args:
        layers: Soil layers with current moisture
        current_fractions: Current root fractions
        params: Root distribution parameters
        feddes_params: Feddes stress parameters

    Returns:
        Updated root fractions
    """
    if not params.plasticity_enabled:
        return current_fractions

    # Calculate water availability index for each layer
    # Based on Feddes stress function (0-1)
    availability = []
    for layer in layers:
        alpha = feddes_stress_factor(layer.psi, feddes_params)
        availability.append(alpha)

    # Calculate target distribution weighted by availability
    weighted_avail = [a * f for a, f in zip(availability, current_fractions)]
    total_weighted = sum(weighted_avail)

    if total_weighted > 0:
        target_fractions = [w / total_weighted for w in weighted_avail]
    else:
        target_fractions = current_fractions

    # Gradually move toward target (prevents sudden changes)
    new_fractions = []
    for current, target in zip(current_fractions, target_fractions):
        new_frac = current + params.plasticity_rate * (target - current)
        new_frac = max(params.min_root_fraction, new_frac)
        new_fractions.append(new_frac)

    # Normalize
    total = sum(new_fractions)
    if total > 0:
        new_fractions = [f / total for f in new_fractions]

    return new_fractions


def calculate_layer_uptake_feddes(
    layer: SoilLayer,
    T_pot: float,
    root_fraction: float,
    feddes_params: FeddesParameters
) -> float:
    """
    Calculate potential water uptake from a single layer using Feddes model.

    S = α(ψ) × T_pot × root_fraction / Δz

    where S is sink term (1/day) and α is stress factor.

    Args:
        layer: Soil layer
        T_pot: Potential transpiration (mm/day)
        root_fraction: Fraction of roots in this layer
        feddes_params: Feddes stress parameters

    Returns:
        Potential uptake from this layer (mm/day)
    """
    if root_fraction <= 0 or T_pot <= 0:
        return 0.0

    # Get stress factor from Feddes function
    alpha = feddes_stress_factor(layer.psi, feddes_params, T_pot)

    # Layer potential uptake
    uptake = alpha * T_pot * root_fraction

    return max(0.0, uptake)


def calculate_compensatory_uptake(
    layers: List[SoilLayer],
    T_pot: float,
    root_fractions: List[float],
    feddes_params: FeddesParameters,
    omega_c: float = 1.0,
    deep_compensation_factor: float = 1.5,
    deep_threshold_m: float = 0.40
) -> List[float]:
    """
    Calculate root water uptake with enhanced compensation from deep wet layers.

    When some layers are stressed, plants can compensate by extracting
    more water from layers with adequate moisture.

    v2.2 Enhancement: Deep layers (>40cm) receive enhanced compensation weight
    to improve deep soil moisture dynamics during drought periods.

    Compensatory uptake (Jarvis, 1989; Šimůnek & Hopmans, 2009):
        S_i = α_i × w_i × T_pot × [1 + ω_c × (1 - T_act/T_pot)]

    where w_i is weighted root fraction that increases for wet layers.

    Args:
        layers: List of soil layers
        T_pot: Potential transpiration (mm/day)
        root_fractions: Root fractions per layer
        feddes_params: Feddes stress parameters
        omega_c: Compensation factor (0 = no compensation, 1 = full)
        deep_compensation_factor: Extra compensation multiplier for deep layers
        deep_threshold_m: Depth threshold for "deep" layers (m)

    Returns:
        List of uptake amounts per layer (mm/day)
    """
    if T_pot <= 0:
        return [0.0] * len(layers)

    # Calculate stress factor for each layer
    alphas = []
    for layer in layers:
        alpha = feddes_stress_factor(layer.psi, feddes_params, T_pot)
        alphas.append(alpha)

    # Non-compensated uptake first (to calculate deficit)
    non_comp_uptakes = []
    for alpha, rf in zip(alphas, root_fractions):
        non_comp_uptakes.append(alpha * rf * T_pot)

    T_non_comp = sum(non_comp_uptakes)
    deficit_fraction = 1.0 - T_non_comp / T_pot if T_pot > 0 else 0.0

    if deficit_fraction <= 0 or omega_c <= 0:
        # No compensation needed
        return non_comp_uptakes

    # Calculate compensation weights with deep layer enhancement
    # Layers with higher alpha (less stressed) get higher weights
    # Deep layers get additional weight multiplier
    weighted_alphas = []
    for i, (layer, alpha, rf) in enumerate(zip(layers, alphas, root_fractions)):
        weight = alpha * rf

        # Apply deep layer enhancement
        if layer.depth_center > deep_threshold_m:
            # Deep layers get enhanced weight when they are wet (high alpha)
            # This allows roots to extract more water from deep layers during drought
            depth_enhancement = 1.0 + (deep_compensation_factor - 1.0) * alpha
            weight *= depth_enhancement

        weighted_alphas.append(weight)

    sum_weighted = sum(weighted_alphas)

    if sum_weighted <= 0:
        return non_comp_uptakes

    # Compensatory uptake with deep layer enhancement
    compensated_uptakes = []
    compensation_factor = 1.0 + omega_c * deficit_fraction

    for i, (layer, alpha, rf) in enumerate(zip(layers, alphas, root_fractions)):
        # Extra weight for unstressed layers, with deep layer bonus
        base_weight = (alpha * rf / sum_weighted) if sum_weighted > 0 else 0

        # Deep layer enhancement factor
        if layer.depth_center > deep_threshold_m:
            depth_mult = 1.0 + (deep_compensation_factor - 1.0) * (
                layer.depth_center - deep_threshold_m) / (1.0 - deep_threshold_m)
            depth_mult = min(depth_mult, deep_compensation_factor)
        else:
            depth_mult = 1.0

        # Compensated uptake with deep enhancement
        comp_uptake = alpha * rf * T_pot * compensation_factor * depth_mult * (
            1 + omega_c * (alpha - np.mean(alphas)) / (max(alphas) + 0.01)
        )

        # Limit to available water (allow up to 60% extraction from deep layers)
        available = max(0, (layer.theta - layer.vg_params.theta_r) *
                        layer.thickness * 1000)

        # Allow higher extraction from deep layers
        max_extraction_frac = 0.6 if layer.depth_center > deep_threshold_m else 0.5
        compensated_uptakes.append(
            min(comp_uptake, available * max_extraction_frac))

    # Scale to not exceed potential transpiration significantly
    total_comp = sum(compensated_uptakes)
    if total_comp > T_pot * 1.1:  # Allow slight overshoot
        scale = T_pot / total_comp
        compensated_uptakes = [u * scale for u in compensated_uptakes]

    return compensated_uptakes


def calculate_hydraulic_lift(
    layers: List[SoilLayer],
    root_fractions: List[float],
    max_lift_mm_day: float = 0.5
) -> List[float]:
    """
    Calculate hydraulic lift (hydraulic redistribution at night).

    Water moves from wet deep layers through roots to dry shallow layers.
    Typically 5-20% of daily transpiration, occurring at night.

    Based on potential gradient between layers through root pathway.

    Args:
        layers: Soil layers
        root_fractions: Root fractions
        max_lift_mm_day: Maximum hydraulic lift (mm/day)

    Returns:
        List of water fluxes per layer (positive = gain, negative = loss)
    """
    n = len(layers)
    fluxes = [0.0] * n

    if n < 2:
        return fluxes

    # Find layers that can donate (wet) and receive (dry)
    psi_values = [layer.psi for layer in layers]
    mean_psi = np.mean(psi_values)

    # Donors: wetter than average (less negative psi)
    # Receivers: drier than average (more negative psi)

    for i in range(n):
        if root_fractions[i] < 0.01:
            continue  # No roots to transport water

        psi_i = psi_values[i]

        for j in range(n):
            if i == j or root_fractions[j] < 0.01:
                continue

            psi_j = psi_values[j]

            # Gradient (positive = j is drier than i, so flow from i to j)
            gradient = psi_i - psi_j

            if gradient > 0:
                # Water moves from i to j through roots
                # Limited by root connectivity and resistance
                connectivity = np.sqrt(root_fractions[i] * root_fractions[j])

                # Flux proportional to gradient and connectivity
                flux = 0.1 * gradient * connectivity * max_lift_mm_day
                flux = min(flux, max_lift_mm_day / n)  # Limit per layer pair

                fluxes[i] -= flux
                fluxes[j] += flux

    return fluxes


@dataclass
class RootUptakeResult:
    """Results from root water uptake calculation"""
    total_uptake: float  # Total transpiration (mm/day)
    layer_uptakes: List[float]  # Uptake per layer (mm/day)
    layer_stress: List[float]  # Stress factor per layer (0-1)
    compensatory_factor: float  # Overall compensation achieved
    hydraulic_lift: List[float]  # Hydraulic lift fluxes (mm/day)
    effective_root_fractions: List[float]  # Current root distribution


class RootWaterUptakeModel:
    """
    Complete root water uptake model integrating all components.
    """

    def __init__(
        self,
        root_params: Optional[RootDistributionParameters] = None,
        feddes_params: Optional[FeddesParameters] = None,
        enable_compensation: bool = True,
        enable_hydraulic_lift: bool = True,
        compensation_factor: float = 0.8
    ):
        """
        Initialize root uptake model.

        Args:
            root_params: Root distribution parameters
            feddes_params: Feddes stress parameters
            enable_compensation: Enable compensatory uptake
            enable_hydraulic_lift: Enable nighttime hydraulic lift
            compensation_factor: Strength of compensation (0-1)
        """
        self.root_params = root_params or RootDistributionParameters()
        self.feddes_params = feddes_params or FeddesParameters.for_crop(
            'maize')
        self.enable_compensation = enable_compensation
        self.enable_hydraulic_lift = enable_hydraulic_lift
        self.compensation_factor = compensation_factor

        self.logger = logging.getLogger(f"{__name__}.RootWaterUptakeModel")

    def calculate_uptake(
        self,
        layers: List[SoilLayer],
        T_pot: float,
        update_root_distribution: bool = True
    ) -> RootUptakeResult:
        """
        Calculate root water uptake for all layers.

        Args:
            layers: Soil layers with current moisture state
            T_pot: Potential transpiration (mm/day)
            update_root_distribution: Whether to update dynamic roots

        Returns:
            RootUptakeResult with uptake distribution
        """
        n = len(layers)

        # Get current root fractions
        current_fractions = [l.root_fraction for l in layers]

        # Initialize if needed
        if sum(current_fractions) < 0.01:
            current_fractions = calculate_static_root_distribution(
                layers, self.root_params
            )
            for i, frac in enumerate(current_fractions):
                layers[i].root_fraction = frac

        # Update dynamic distribution if enabled
        if update_root_distribution and self.root_params.plasticity_enabled:
            new_fractions = update_dynamic_root_distribution(
                layers, current_fractions,
                self.root_params, self.feddes_params
            )
            for i, frac in enumerate(new_fractions):
                layers[i].root_fraction = frac

        # Calculate layer stress factors
        stress_factors = []
        for layer in layers:
            alpha = feddes_stress_factor(layer.psi, self.feddes_params, T_pot)
            stress_factors.append(alpha)

        # Calculate uptake (with or without compensation)
        if self.enable_compensation:
            layer_uptakes = calculate_compensatory_uptake(
                layers, T_pot,
                [l.root_fraction for l in layers],
                self.feddes_params,
                self.compensation_factor
            )
        else:
            layer_uptakes = []
            for layer in layers:
                uptake = calculate_layer_uptake_feddes(
                    layer, T_pot, layer.root_fraction, self.feddes_params
                )
                layer_uptakes.append(uptake)

        total_uptake = sum(layer_uptakes)

        # Calculate compensation factor achieved
        no_stress_uptake = T_pot
        comp_factor = total_uptake / no_stress_uptake if no_stress_uptake > 0 else 0

        # Calculate hydraulic lift if enabled
        if self.enable_hydraulic_lift:
            hydraulic_lift = calculate_hydraulic_lift(
                layers, [l.root_fraction for l in layers]
            )
        else:
            hydraulic_lift = [0.0] * n

        return RootUptakeResult(
            total_uptake=total_uptake,
            layer_uptakes=layer_uptakes,
            layer_stress=stress_factors,
            compensatory_factor=comp_factor,
            hydraulic_lift=hydraulic_lift,
            effective_root_fractions=[l.root_fraction for l in layers]
        )

    def apply_uptake(
        self,
        layers: List[SoilLayer],
        result: RootUptakeResult
    ) -> List[SoilLayer]:
        """
        Apply uptake results to update layer water contents.

        Args:
            layers: Soil layers
            result: Uptake calculation result

        Returns:
            Updated layers
        """
        for i, layer in enumerate(layers):
            # Subtract uptake
            uptake_m3m3 = result.layer_uptakes[i] / (layer.thickness * 1000)

            # Add/subtract hydraulic lift
            lift_m3m3 = result.hydraulic_lift[i] / (layer.thickness * 1000)

            new_theta = layer.theta - uptake_m3m3 + lift_m3m3

            # Enforce bounds
            layer.theta = np.clip(
                new_theta,
                layer.vg_params.theta_r * 1.01,
                layer.vg_params.theta_s
            )

        return layers


def compare_water_content_vs_pressure_stress(
    theta: float,
    theta_FC: float,
    theta_WP: float,
    vg_params: VanGenuchtenParameters,
    feddes_params: FeddesParameters
) -> Dict[str, float]:
    """
    Compare stress calculation using water content vs pressure head.

    Demonstrates why pressure-based approach is more physically correct.

    Args:
        theta: Current water content
        theta_FC: Field capacity
        theta_WP: Wilting point
        vg_params: Van Genuchten parameters
        feddes_params: Feddes parameters

    Returns:
        Dictionary comparing both approaches
    """
    # Water content-based (original simple approach)
    available_range = theta_FC - theta_WP
    if available_range > 0:
        rew = (theta - theta_WP) / available_range
        rew = np.clip(rew, 0, 1)

        # Simple linear stress
        threshold = 0.45
        if rew >= threshold:
            stress_wc = 1.0
        else:
            stress_wc = rew / threshold
    else:
        stress_wc = 0.0

    # Pressure head-based (physically correct)
    psi = van_genuchten_psi_from_theta(theta, vg_params)
    stress_psi = feddes_stress_factor(psi, feddes_params)

    return {
        'theta': theta,
        'psi_m': psi,
        'psi_kPa': psi * 9.81,  # Convert m to kPa approximately
        'stress_water_content': stress_wc,
        'stress_pressure_head': stress_psi,
        'difference': stress_psi - stress_wc,
        'relative_difference_pct': 100 * (stress_psi - stress_wc) / (stress_wc + 0.01)
    }
