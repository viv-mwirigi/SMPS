"""
Green-Ampt infiltration model with rainfall intensity consideration.

This module implements the Green-Ampt infiltration equation with support for:
1. Cumulative infiltration tracking
2. Time-to-ponding calculation
3. Sub-daily rainfall intensity distribution estimation
4. Macropore bypass flow for high-intensity events

References:
- Green, W.H. and Ampt, G.A. (1911). Studies on soil physics: I. Flow of air
  and water through soils. Journal of Agricultural Science, 4:1-24.
- Mein, R.G. and Larson, C.L. (1973). Modeling infiltration during a steady
  rain. Water Resources Research, 9(2):384-394.
- Rawls, W.J. et al. (1983). Green-Ampt infiltration parameters from soils
  data. Journal of Hydraulic Engineering, 109(1):62-70.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum
import logging

from smps.physics.soil_hydraulics import (
    VanGenuchtenParameters,
    van_genuchten_theta_from_psi,
    van_genuchten_psi_from_theta,
)

logger = logging.getLogger(__name__)


@dataclass
class GreenAmptParameters:
    """
    Green-Ampt infiltration model parameters.

    The Green-Ampt equation for cumulative infiltration F is:
        F = K_s * t + ψ_f * Δθ * ln(1 + F / (ψ_f * Δθ))

    Infiltration rate:
        f = K_s * [1 + (ψ_f * Δθ) / F]

    Time to ponding (for constant rainfall intensity i):
        t_p = K_s * ψ_f * Δθ / [i * (i - K_s)]  for i > K_s

    Parameters:
        K_s: Saturated hydraulic conductivity (mm/h)
        psi_f: Wetting front suction head (mm, positive value)
        theta_s: Saturated water content (m³/m³)
        theta_i: Initial water content (m³/m³)
        Δθ: theta_s - theta_i (moisture deficit)

    Note: Working in mm and hours for rainfall intensity compatibility.
    """
    K_s: float  # Saturated hydraulic conductivity (mm/h)
    psi_f: float  # Wetting front suction head (mm, positive)
    theta_s: float  # Saturated water content (m³/m³)
    theta_i: float  # Initial water content (m³/m³)

    # Macropore parameters
    macropore_fraction: float = 0.05  # Fraction of pore space as macropores
    macropore_threshold_mm_h: float = 30.0  # Intensity threshold for bypass

    @property
    def delta_theta(self) -> float:
        """Moisture deficit (m³/m³)"""
        return max(0.01, self.theta_s - self.theta_i)

    @property
    def sorptivity_parameter(self) -> float:
        """ψ_f × Δθ term (mm)"""
        return self.psi_f * self.delta_theta

    @classmethod
    def from_texture_class(
        cls,
        texture_class: str,
        theta_i: float,
        porosity: Optional[float] = None
    ) -> "GreenAmptParameters":
        """
        Get Green-Ampt parameters for USDA texture class.

        Values from Rawls et al. (1983) - widely used defaults.

        Args:
            texture_class: USDA texture class
            theta_i: Initial water content
            porosity: Override for theta_s

        Returns:
            GreenAmptParameters
        """
        # Rawls et al. (1983) Table 1
        # Format: (K_s [cm/h], ψ_f [cm], porosity, effective_porosity)
        params = {
            'sand': (11.78, 4.95, 0.437, 0.417),
            'loamy_sand': (2.99, 6.13, 0.437, 0.401),
            'sandy_loam': (1.09, 11.01, 0.453, 0.412),
            'loam': (0.34, 8.89, 0.463, 0.434),
            'silt_loam': (0.65, 16.68, 0.501, 0.486),
            'sandy_clay_loam': (0.15, 21.85, 0.398, 0.330),
            'clay_loam': (0.10, 20.88, 0.464, 0.390),
            'silty_clay_loam': (0.10, 27.30, 0.471, 0.432),
            'sandy_clay': (0.06, 23.90, 0.430, 0.321),
            'silty_clay': (0.05, 29.22, 0.479, 0.423),
            'clay': (0.03, 31.63, 0.475, 0.385),
        }

        texture_key = texture_class.lower().replace(' ', '_')

        if texture_key not in params:
            logger.warning(f"Unknown texture '{texture_class}', using 'loam'")
            texture_key = 'loam'

        K_s_cm_h, psi_f_cm, default_porosity, _ = params[texture_key]

        theta_s = porosity if porosity is not None else default_porosity

        return cls(
            K_s=K_s_cm_h * 10,  # cm/h → mm/h
            psi_f=psi_f_cm * 10,  # cm → mm
            theta_s=theta_s,
            theta_i=theta_i
        )

    @classmethod
    def from_van_genuchten(
        cls,
        vg_params: VanGenuchtenParameters,
        theta_i: float
    ) -> "GreenAmptParameters":
        """
        Derive Green-Ampt parameters from Van Genuchten parameters.

        Uses the relationships from Rawls & Brakensiek (1989):
            ψ_f ≈ (2 + 3λ) / (1 + 3λ) × (2/α) × S_e^(1/λ)

        where λ = m/(1-m) = n-1

        Args:
            vg_params: Van Genuchten parameters
            theta_i: Initial water content

        Returns:
            GreenAmptParameters
        """
        # Convert K_sat from m/day to mm/h
        K_s_mm_h = vg_params.K_sat * 1000 / 24

        # Estimate wetting front suction
        # Using approximation: ψ_f ≈ 0.76 / α (from Morel-Seytoux & Khanji, 1974)
        # Convert from 1/m to mm
        psi_f_mm = 0.76 / vg_params.alpha * 1000

        # Clamp to reasonable range
        psi_f_mm = np.clip(psi_f_mm, 10, 500)

        return cls(
            K_s=K_s_mm_h,
            psi_f=psi_f_mm,
            theta_s=vg_params.theta_s,
            theta_i=theta_i
        )

    def update_initial_moisture(self, theta_i: float):
        """Update initial moisture content"""
        self.theta_i = np.clip(theta_i, 0.01, self.theta_s - 0.01)


@dataclass
class InfiltrationState:
    """
    Tracks infiltration state for cumulative calculations.
    """
    cumulative_infiltration: float = 0.0  # Total infiltration (mm)
    cumulative_runoff: float = 0.0  # Total runoff (mm)
    time_since_rain_start: float = 0.0  # Hours since rain began
    ponding_occurred: bool = False  # Whether ponding has occurred
    wetting_front_depth: float = 0.0  # Depth of wetting front (mm)

    def reset(self):
        """Reset state for new event"""
        self.cumulative_infiltration = 0.0
        self.cumulative_runoff = 0.0
        self.time_since_rain_start = 0.0
        self.ponding_occurred = False
        self.wetting_front_depth = 0.0


class RainfallIntensityDistribution(Enum):
    """Common rainfall intensity distributions for daily disaggregation"""
    UNIFORM = "uniform"  # Spread evenly across 24h
    TRIANGULAR = "triangular"  # Peak in middle
    FRONTAL = "frontal"  # Front-loaded (morning rain)
    CONVECTIVE = "convective"  # Short intense bursts
    BIMODAL = "bimodal"  # Two peaks (tropical pattern)


def estimate_rainfall_intensity_distribution(
    daily_precip_mm: float,
    max_temperature_c: Optional[float] = None,
    season: Optional[str] = None,
    distribution: RainfallIntensityDistribution = RainfallIntensityDistribution.TRIANGULAR
) -> List[float]:
    """
    Estimate sub-daily rainfall intensity distribution from daily total.

    Returns hourly intensities for a 24-hour period.

    Args:
        daily_precip_mm: Total daily precipitation (mm)
        max_temperature_c: Maximum temperature (for convective inference)
        season: Season name (wet/dry for tropical inference)
        distribution: Assumed distribution type

    Returns:
        List of 24 hourly intensities (mm/h)
    """
    if daily_precip_mm <= 0:
        return [0.0] * 24

    # Auto-select distribution based on conditions
    if distribution == RainfallIntensityDistribution.CONVECTIVE or (
        max_temperature_c is not None and max_temperature_c > 30 and daily_precip_mm > 20
    ):
        # Convective: short intense burst (4-6 hours)
        # Peak in afternoon (hours 14-18)
        hours = np.arange(24)
        # Gaussian centered at hour 15 with std=2
        weights = np.exp(-0.5 * ((hours - 15) / 2) ** 2)
        # Only rain for ~6 hours
        weights[hours < 12] = 0
        weights[hours > 20] = 0

    elif distribution == RainfallIntensityDistribution.BIMODAL:
        # Bimodal: morning and afternoon peaks (tropical)
        hours = np.arange(24)
        morning_peak = np.exp(-0.5 * ((hours - 8) / 2) ** 2)
        afternoon_peak = np.exp(-0.5 * ((hours - 16) / 2) ** 2)
        weights = 0.4 * morning_peak + 0.6 * afternoon_peak

    elif distribution == RainfallIntensityDistribution.FRONTAL:
        # Frontal: early peak, long tail
        hours = np.arange(24)
        weights = np.exp(-hours / 8)  # Exponential decay

    elif distribution == RainfallIntensityDistribution.TRIANGULAR:
        # Triangular: peak in middle of event
        hours = np.arange(24)
        # Assume 8-hour event centered at noon
        weights = np.zeros(24)
        event_start = 8
        event_end = 16
        peak = 12
        for h in range(event_start, event_end):
            if h <= peak:
                weights[h] = (h - event_start + 1) / (peak - event_start + 1)
            else:
                weights[h] = (event_end - h) / (event_end - peak)
    else:
        # Uniform distribution
        weights = np.ones(24)

    # Normalize to sum to daily total
    weights = weights / weights.sum()
    intensities = daily_precip_mm * weights

    return list(intensities)


def time_to_ponding(
    rainfall_intensity_mm_h: float,
    params: GreenAmptParameters
) -> float:
    """
    Calculate time to ponding for constant rainfall intensity.

    t_p = K_s × ψ_f × Δθ / [i × (i - K_s)]

    Ponding occurs when rainfall intensity exceeds infiltration capacity.

    Args:
        rainfall_intensity_mm_h: Rainfall intensity (mm/h)
        params: Green-Ampt parameters

    Returns:
        Time to ponding (hours), or inf if no ponding
    """
    i = rainfall_intensity_mm_h
    K_s = params.K_s

    if i <= K_s:
        # Intensity less than K_s: no ponding
        return float('inf')

    numerator = K_s * params.sorptivity_parameter
    denominator = i * (i - K_s)

    if denominator <= 0:
        return float('inf')

    return numerator / denominator


def infiltration_before_ponding(
    rainfall_intensity_mm_h: float,
    params: GreenAmptParameters
) -> float:
    """
    Calculate cumulative infiltration at time of ponding.

    F_p = K_s × ψ_f × Δθ / (i - K_s)

    Args:
        rainfall_intensity_mm_h: Rainfall intensity (mm/h)
        params: Green-Ampt parameters

    Returns:
        Cumulative infiltration at ponding (mm)
    """
    i = rainfall_intensity_mm_h
    K_s = params.K_s

    if i <= K_s:
        return float('inf')

    return K_s * params.sorptivity_parameter / (i - K_s)


def green_ampt_infiltration_rate(
    cumulative_infiltration: float,
    params: GreenAmptParameters
) -> float:
    """
    Calculate infiltration rate from cumulative infiltration.

    f = K_s × [1 + (ψ_f × Δθ) / F]

    Args:
        cumulative_infiltration: Cumulative infiltration F (mm)
        params: Green-Ampt parameters

    Returns:
        Infiltration rate (mm/h)
    """
    F = max(cumulative_infiltration, 0.1)  # Avoid division by zero

    return params.K_s * (1 + params.sorptivity_parameter / F)


def green_ampt_cumulative_infiltration(
    time_hours: float,
    params: GreenAmptParameters,
    F_initial: float = 0.0
) -> float:
    """
    Calculate cumulative infiltration using iterative solution.

    The Green-Ampt equation is implicit in F:
        F = K_s × t + ψ_f × Δθ × ln(1 + F / (ψ_f × Δθ))

    Solved iteratively using Newton-Raphson.

    Args:
        time_hours: Time since ponding (hours)
        params: Green-Ampt parameters
        F_initial: Initial cumulative infiltration (mm)

    Returns:
        Cumulative infiltration (mm)
    """
    if time_hours <= 0:
        return F_initial

    K_s = params.K_s
    S = params.sorptivity_parameter  # ψ_f × Δθ

    # Initial guess
    F = F_initial + K_s * time_hours

    # Newton-Raphson iteration
    for _ in range(50):
        f_F = F - K_s * time_hours - S * np.log(1 + F / S)
        f_prime = 1 - S / (S + F)

        if abs(f_prime) < 1e-10:
            break

        F_new = F - f_F / f_prime

        if abs(F_new - F) < 1e-6:
            break

        F = max(0.1, F_new)  # Ensure positive

    return F


def macropore_bypass_flow(
    rainfall_intensity_mm_h: float,
    infiltration_rate_mm_h: float,
    params: GreenAmptParameters,
    surface_saturation: float = 0.0
) -> float:
    """
    Calculate macropore bypass flow during high-intensity events.

    When rainfall intensity greatly exceeds matrix infiltration capacity,
    water can bypass the soil matrix through macropores (cracks, root
    channels, worm burrows).

    Args:
        rainfall_intensity_mm_h: Rainfall intensity (mm/h)
        infiltration_rate_mm_h: Matrix infiltration rate (mm/h)
        params: Green-Ampt parameters
        surface_saturation: Surface layer saturation (0-1)

    Returns:
        Macropore bypass flow rate (mm/h)
    """
    if rainfall_intensity_mm_h < params.macropore_threshold_mm_h:
        return 0.0

    # Excess intensity beyond matrix capacity
    excess = rainfall_intensity_mm_h - infiltration_rate_mm_h

    if excess <= 0:
        return 0.0

    # Macropore flow increases with saturation and excess intensity
    # At low saturation, macropores can accept more flow
    saturation_factor = 1.0 - 0.5 * surface_saturation

    # Fraction of excess that enters macropores
    bypass_fraction = params.macropore_fraction * saturation_factor

    return excess * bypass_fraction


def simulate_infiltration_event(
    hourly_rainfall_mm: List[float],
    params: GreenAmptParameters,
    initial_state: Optional[InfiltrationState] = None,
    dt_hours: float = 1.0
) -> Tuple[float, float, InfiltrationState]:
    """
    Simulate infiltration for a rainfall event with variable intensity.

    Uses Green-Ampt equation with ponding time calculation.

    Args:
        hourly_rainfall_mm: List of hourly rainfall amounts (mm)
        params: Green-Ampt parameters
        initial_state: Previous infiltration state
        dt_hours: Time step (hours)

    Returns:
        Tuple of (total_infiltration, total_runoff, final_state)
    """
    state = initial_state if initial_state else InfiltrationState()

    # Reset if dry period preceded
    if hourly_rainfall_mm[0] > 0 and state.time_since_rain_start > 24:
        state.reset()
        params.update_initial_moisture(params.theta_i)  # Would need current θ

    total_infiltration = 0.0
    total_runoff = 0.0

    for hour, rainfall_mm in enumerate(hourly_rainfall_mm):
        if rainfall_mm <= 0:
            continue

        # Convert to intensity
        intensity_mm_h = rainfall_mm / dt_hours

        if not state.ponding_occurred:
            # Pre-ponding: check if ponding occurs this timestep
            t_p = time_to_ponding(intensity_mm_h, params)

            if t_p >= dt_hours:
                # No ponding this timestep - all rain infiltrates
                infiltration = rainfall_mm
                runoff = 0.0
                state.cumulative_infiltration += infiltration
            else:
                # Ponding occurs during this timestep
                state.ponding_occurred = True

                # Infiltration before ponding
                F_p = infiltration_before_ponding(intensity_mm_h, params)

                # Time remaining after ponding
                t_remaining = dt_hours - t_p

                # Infiltration after ponding using Green-Ampt
                F_after = green_ampt_cumulative_infiltration(
                    t_remaining, params, F_p
                )

                infiltration = F_after
                runoff = rainfall_mm - infiltration
                state.cumulative_infiltration = F_after
        else:
            # Post-ponding: use Green-Ampt
            F_new = green_ampt_cumulative_infiltration(
                dt_hours, params, state.cumulative_infiltration
            )

            infiltration = F_new - state.cumulative_infiltration

            # Check for macropore bypass
            matrix_rate = green_ampt_infiltration_rate(
                state.cumulative_infiltration, params
            )
            bypass = macropore_bypass_flow(
                intensity_mm_h, matrix_rate, params
            )

            # Total potential infiltration
            total_potential = infiltration + bypass * dt_hours

            # Cannot infiltrate more than rainfall
            actual_infiltration = min(total_potential, rainfall_mm)
            runoff = rainfall_mm - actual_infiltration

            state.cumulative_infiltration += actual_infiltration

        total_infiltration += infiltration
        total_runoff += max(0, runoff)
        state.time_since_rain_start += dt_hours

        # Update wetting front depth
        state.wetting_front_depth = (
            state.cumulative_infiltration / (params.delta_theta * 1000)
        )  # Convert mm to m, divide by Δθ

    state.cumulative_runoff += total_runoff

    return total_infiltration, total_runoff, state


def daily_infiltration_green_ampt(
    daily_precip_mm: float,
    params: GreenAmptParameters,
    distribution: RainfallIntensityDistribution = RainfallIntensityDistribution.TRIANGULAR,
    max_temperature_c: Optional[float] = None
) -> Tuple[float, float]:
    """
    Calculate daily infiltration and runoff using Green-Ampt with
    estimated sub-daily intensity distribution.

    This is the main function for daily-timestep models.

    Args:
        daily_precip_mm: Daily precipitation (mm)
        params: Green-Ampt parameters
        distribution: Rainfall distribution type
        max_temperature_c: Maximum temperature for distribution inference

    Returns:
        Tuple of (infiltration_mm, runoff_mm)
    """
    if daily_precip_mm <= 0:
        return 0.0, 0.0

    # Estimate hourly distribution
    hourly_rain = estimate_rainfall_intensity_distribution(
        daily_precip_mm, max_temperature_c, distribution=distribution
    )

    # Simulate infiltration
    infiltration, runoff, _ = simulate_infiltration_event(
        hourly_rain, params
    )

    return infiltration, runoff


def compare_simple_vs_green_ampt(
    daily_precip_mm: float,
    surface_theta: float,
    params: GreenAmptParameters,
    simple_K_s_mm_day: float
) -> Dict[str, float]:
    """
    Compare simple capacity approach vs Green-Ampt infiltration.

    Useful for understanding the impact of intensity consideration.

    Args:
        daily_precip_mm: Daily precipitation (mm)
        surface_theta: Current surface water content
        params: Green-Ampt parameters
        simple_K_s_mm_day: Simple model K_s (mm/day)

    Returns:
        Dictionary with both estimates and difference
    """
    # Simple capacity approach (original method)
    saturation = surface_theta / params.theta_s
    saturation_factor = np.exp(-3.0 * saturation)
    simple_capacity = simple_K_s_mm_day * (0.1 + 0.9 * saturation_factor)
    simple_infiltration = min(simple_capacity, daily_precip_mm)
    simple_runoff = daily_precip_mm - simple_infiltration

    # Green-Ampt with intensity distribution
    ga_infiltration, ga_runoff = daily_infiltration_green_ampt(
        daily_precip_mm, params
    )

    return {
        'simple_infiltration': simple_infiltration,
        'simple_runoff': simple_runoff,
        'green_ampt_infiltration': ga_infiltration,
        'green_ampt_runoff': ga_runoff,
        'infiltration_difference': ga_infiltration - simple_infiltration,
        'runoff_difference': ga_runoff - simple_runoff,
        'relative_difference_percent': (
            100 * (ga_infiltration - simple_infiltration) /
            (simple_infiltration + 1e-6)
        )
    }
