"""
FAO-56 Evapotranspiration Model for SWPPS.

Implements proper ET partitioning following FAO-56:
- Dual crop coefficient (Kcb + Ke)
- Water stress coefficient (Ks)
- Soil evaporation reduction (Kr)

References:
- Allen et al. (1998) FAO-56: Crop evapotranspiration
- Allen (2000) Dual crop coefficient method
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("swpps.physics.evapotranspiration")


@dataclass
class CropCoefficients:
    """
    FAO-56 crop coefficient parameters.

    The dual coefficient separates:
    - Kcb: Basal crop coefficient (transpiration when soil is dry)
    - Ke: Soil evaporation coefficient
    - Kc = Kcb × Ks + Ke (where Ks is water stress coefficient)
    """
    # Basal crop coefficients
    Kcb_ini: float = 0.15   # Initial stage
    Kcb_mid: float = 1.15   # Mid-season
    Kcb_end: float = 0.50   # End of season

    # Growth stage lengths (days)
    L_ini: int = 30         # Initial
    L_dev: int = 40         # Development
    L_mid: int = 50         # Mid-season
    L_late: int = 30        # Late season

    # Crop parameters
    crop_height_m: float = 2.0
    root_depth_max_m: float = 1.0
    p_standard: float = 0.55  # Depletion fraction at ET0=5 mm/day

    @property
    def total_season(self) -> int:
        return self.L_ini + self.L_dev + self.L_mid + self.L_late

    @classmethod
    def for_crop(cls, crop_name: str) -> "CropCoefficients":
        """Get FAO-56 standard coefficients for crop."""
        # FAO-56 Tables 12 and 17
        # Format: (Kcb_ini, Kcb_mid, Kcb_end, L_ini, L_dev, L_mid, L_late, height, root_depth, p)
        crops = {
            'maize': (0.15, 1.15, 0.50, 30, 40, 50, 30, 2.0, 1.0, 0.55),
            'wheat': (0.15, 1.10, 0.25, 20, 35, 40, 30, 1.0, 1.0, 0.55),
            'rice': (1.00, 1.15, 0.90, 30, 30, 60, 30, 1.0, 0.5, 0.20),
            'soybean': (0.15, 1.10, 0.30, 20, 35, 60, 25, 0.8, 0.6, 0.50),
            'cotton': (0.15, 1.15, 0.50, 30, 50, 55, 45, 1.3, 1.3, 0.65),
            'sorghum': (0.15, 1.05, 0.35, 20, 35, 40, 30, 1.5, 1.0, 0.55),
            'millet': (0.15, 1.00, 0.30, 15, 25, 40, 25, 1.5, 1.0, 0.55),
            'groundnut': (0.15, 1.10, 0.50, 25, 35, 45, 25, 0.4, 0.5, 0.50),
            'cassava': (0.15, 0.80, 0.30, 20, 40, 90, 60, 1.0, 0.7, 0.35),
            'beans': (0.15, 1.05, 0.30, 20, 30, 30, 10, 0.4, 0.5, 0.45),
            'potato': (0.15, 1.10, 0.65, 25, 30, 45, 30, 0.6, 0.4, 0.35),
            'tomato': (0.15, 1.10, 0.70, 30, 40, 45, 30, 0.6, 0.7, 0.40),
            'grassland': (0.30, 0.95, 0.95, 10, 20, 200, 30, 0.3, 0.5, 0.50),
            'savanna': (0.20, 0.70, 0.40, 30, 60, 120, 60, 1.0, 1.5, 0.60),
            'generic': (0.20, 0.90, 0.50, 25, 35, 45, 25, 0.8, 0.6, 0.50),
        }

        key = crop_name.lower().replace(' ', '_')
        if key not in crops:
            logger.warning(f"Unknown crop '{crop_name}', using 'generic'")
            key = 'generic'

        data = crops[key]
        return cls(
            Kcb_ini=data[0], Kcb_mid=data[1], Kcb_end=data[2],
            L_ini=data[3], L_dev=data[4], L_mid=data[5], L_late=data[6],
            crop_height_m=data[7], root_depth_max_m=data[8], p_standard=data[9]
        )


def get_Kcb_from_doy(
    day_of_year: int,
    planting_doy: int,
    crop_coef: CropCoefficients,
) -> float:
    """
    Get basal crop coefficient for day of year.

    Args:
        day_of_year: Day of year (1-365)
        planting_doy: Day of year when crop was planted
        crop_coef: Crop coefficient parameters

    Returns:
        Kcb for the day
    """
    days_since_planting = day_of_year - planting_doy

    if days_since_planting < 0:
        days_since_planting += 365  # Wrap around year

    L_ini = crop_coef.L_ini
    L_dev = crop_coef.L_dev
    L_mid = crop_coef.L_mid
    L_late = crop_coef.L_late

    # Determine growth stage
    if days_since_planting <= L_ini:
        # Initial stage
        return crop_coef.Kcb_ini

    elif days_since_planting <= L_ini + L_dev:
        # Development stage - linear interpolation
        days_in_dev = days_since_planting - L_ini
        frac = days_in_dev / L_dev
        return crop_coef.Kcb_ini + frac * (crop_coef.Kcb_mid - crop_coef.Kcb_ini)

    elif days_since_planting <= L_ini + L_dev + L_mid:
        # Mid-season
        return crop_coef.Kcb_mid

    elif days_since_planting <= L_ini + L_dev + L_mid + L_late:
        # Late season - linear interpolation
        days_in_late = days_since_planting - L_ini - L_dev - L_mid
        frac = days_in_late / L_late
        return crop_coef.Kcb_mid + frac * (crop_coef.Kcb_end - crop_coef.Kcb_mid)

    else:
        # After harvest
        return 0.15


def get_Kcb_from_ndvi(
    ndvi: float,
    ndvi_min: float = 0.15,
    ndvi_max: float = 0.85,
    Kcb_min: float = 0.15,
    Kcb_max: float = 1.20,
) -> float:
    """
    Estimate Kcb from NDVI.

    Uses linear relationship validated by many studies:
    Kcb = Kcb_min + (Kcb_max - Kcb_min) × (NDVI - NDVImin) / (NDVImax - NDVImin)

    Args:
        ndvi: NDVI value (0-1)
        ndvi_min: NDVI for bare soil (default 0.15)
        ndvi_max: NDVI at full cover (default 0.85)
        Kcb_min: Minimum Kcb (default 0.15)
        Kcb_max: Maximum Kcb (default 1.20)

    Returns:
        Estimated Kcb
    """
    if ndvi <= ndvi_min:
        return Kcb_min
    elif ndvi >= ndvi_max:
        return Kcb_max

    frac = (ndvi - ndvi_min) / (ndvi_max - ndvi_min)
    return Kcb_min + frac * (Kcb_max - Kcb_min)


def ndvi_to_fractional_cover(
    ndvi: float,
    ndvi_min: float = 0.15,
    ndvi_max: float = 0.85,
) -> float:
    """
    Convert NDVI to fractional vegetation cover.

    fc = ((NDVI - NDVImin) / (NDVImax - NDVImin))²
    """
    if ndvi <= ndvi_min:
        return 0.0
    elif ndvi >= ndvi_max:
        return 1.0

    frac = (ndvi - ndvi_min) / (ndvi_max - ndvi_min)
    return frac ** 2


def compute_water_stress_coefficient(
    theta: float,
    theta_fc: float,
    theta_wp: float,
    p: float = 0.55,
    et0: float = 5.0,
) -> float:
    """
    Compute water stress coefficient Ks (θ-based, FAO-56 standard).

    FAO-56 Eq. 84:
    Ks = (TAW - Dr) / ((1-p) × TAW)   when Dr > RAW
    Ks = 1                              when Dr <= RAW

    where:
        TAW = total available water = (θfc - θwp) × Zr
        RAW = readily available water = p × TAW
        Dr = root zone depletion

    Simplified for theta directly:
    Ks = (θ - θwp) / ((1-p) × (θfc - θwp))   when θ < θfc - p×(θfc-θwp)
    Ks = 1                                    otherwise

    Args:
        theta: Current water content (m³/m³)
        theta_fc: Field capacity
        theta_wp: Wilting point
        p: Depletion fraction (adjusted for ET0)
        et0: Reference ET (mm/day) for p adjustment

    Returns:
        Ks (0-1)
    """
    # Adjust p for ET0 (FAO-56 Eq. 86)
    p_adj = p + 0.04 * (5.0 - et0)
    p_adj = np.clip(p_adj, 0.1, 0.8)

    # Total available water
    TAW = theta_fc - theta_wp

    # Readily available water threshold
    theta_raw = theta_fc - p_adj * TAW

    if theta >= theta_raw:
        return 1.0
    elif theta <= theta_wp:
        return 0.0
    else:
        return (theta - theta_wp) / ((1 - p_adj) * TAW)


def compute_water_stress_from_potential(
    psi_kpa: float,
    psi_fc_kpa: float = -33.0,
    psi_wp_kpa: float = -1500.0,
    psi_critical_kpa: float = -100.0,
    stress_shape: float = 2.0,
) -> float:
    """
    Compute water stress coefficient Ks directly from matric potential (ψ-based).

    This is the PREFERRED method for tension-space models because:
    1. Plants respond to ψ, not θ (hydraulic continuity)
    2. ψ is soil-type independent
    3. Stress thresholds are more universal in ψ space

    Uses sigmoid function for smooth transition:
    Ks = 1 / (1 + exp(-shape × (ψ - ψ_critical) / (ψ_fc - ψ_critical)))

    Key thresholds (typical values):
    - ψ_fc ≈ -10 to -33 kPa (field capacity, no stress)
    - ψ_critical ≈ -100 to -200 kPa (stress onset)
    - ψ_wp ≈ -1500 kPa (wilting point, Ks → 0)

    Args:
        psi_kpa: Current matric potential (kPa, negative)
        psi_fc_kpa: Field capacity potential (kPa, typically -10 to -33)
        psi_wp_kpa: Wilting point potential (kPa, typically -1500)
        psi_critical_kpa: Potential at stress onset (kPa, typically -100)
        stress_shape: Shape parameter controlling transition steepness

    Returns:
        Ks (0-1): Water stress coefficient
    """
    # No stress above field capacity
    if psi_kpa >= psi_fc_kpa:
        return 1.0

    # Complete stress below wilting point
    if psi_kpa <= psi_wp_kpa:
        return 0.0

    # Linear stress in log-ψ space (plants respond logarithmically)
    # This is more physiologically accurate than linear ψ
    log_psi = np.log10(abs(psi_kpa))
    log_fc = np.log10(abs(psi_fc_kpa))
    log_wp = np.log10(abs(psi_wp_kpa))
    log_crit = np.log10(abs(psi_critical_kpa))

    # Above critical point: no stress
    if psi_kpa >= psi_critical_kpa:
        return 1.0

    # Linear decrease in log space from critical to wilting
    Ks = (log_wp - log_psi) / (log_wp - log_crit)

    return float(np.clip(Ks, 0.0, 1.0))


def compute_soil_evaporation_coefficient(
    theta_surface: float,
    theta_fc: float,
    theta_wp: float,
    theta_residual: float,
    fc: float,
    precip_mm: float = 0.0,
    few: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute soil evaporation coefficient Ke.

    FAO-56 approach:
    Ke = Kr × (Kcmax - Kcb) × few

    where:
        Kr = evaporation reduction coefficient
        Kcmax = maximum Kc when soil is wet
        few = fraction of soil exposed and wetted

    Args:
        theta_surface: Surface layer water content
        theta_fc: Field capacity
        theta_wp: Wilting point
        theta_residual: Residual water content
        fc: Fractional vegetation cover
        precip_mm: Recent precipitation
        few: Fraction exposed and wetted (computed if None)

    Returns:
        Tuple of (Ke, Kr)
    """
    # Evaporation reduction coefficient
    # Kr = 1 when surface wet, decreases as surface dries
    # Readily evaporable water (~8-12 mm)
    REW = 0.1 * (theta_fc - theta_residual) * 100
    TEW = 0.5 * (theta_fc - theta_residual) * 100  # Total evaporable water

    if TEW <= 0:
        TEW = 10.0
    if REW <= 0:
        REW = min(8.0, TEW * 0.5)

    # Estimate cumulative evaporation from surface drying
    Se = (theta_fc - theta_surface) / (theta_fc - theta_residual)
    De = Se * TEW  # Approximate depletion

    if De <= REW:
        Kr = 1.0
    else:
        Kr = max(0.0, (TEW - De) / (TEW - REW))

    # Fraction exposed and wetted
    if few is None:
        few_dry = 1.0 - fc  # Exposed fraction
        if precip_mm > 5:  # Wetted by rain
            few = min(1.0, few_dry)
        else:
            few = few_dry

    # Maximum Kc (when soil surface is wet)
    Kcmax = 1.20

    # Ke is limited by energy available above transpiration
    Kcb = 0.2 + fc * 1.0  # Approximate Kcb
    Ke = Kr * (Kcmax - Kcb) * few
    Ke = max(0.0, min(Ke, 1.0))  # Bound

    return Ke, Kr


def compute_et_partitioning(
    et0_mm: float,
    ndvi: Optional[float] = None,
    crop_coef: Optional[CropCoefficients] = None,
    day_of_year: Optional[int] = None,
    planting_doy: Optional[int] = None,
    theta_surface: Optional[float] = None,
    theta_root: Optional[float] = None,
    theta_fc: float = 0.30,
    theta_wp: float = 0.10,
) -> Dict[str, float]:
    """
    Compute ET partitioning into evaporation and transpiration.

    Args:
        et0_mm: Reference evapotranspiration (mm/day)
        ndvi: NDVI value (if available)
        crop_coef: Crop coefficients
        day_of_year: Current DOY
        planting_doy: Planting DOY
        theta_surface: Surface water content
        theta_root: Root zone water content
        theta_fc: Field capacity
        theta_wp: Wilting point

    Returns:
        Dict with evaporation_mm, transpiration_mm, etc.
    """
    # Get Kcb from NDVI or growth stage
    if ndvi is not None:
        Kcb = get_Kcb_from_ndvi(ndvi)
        fc = ndvi_to_fractional_cover(ndvi)
    elif crop_coef is not None and day_of_year is not None:
        planting = planting_doy or 1
        Kcb = get_Kcb_from_doy(day_of_year, planting, crop_coef)
        # Estimate fc from Kcb
        fc = np.clip((Kcb - 0.15) / 1.0, 0, 1)
    else:
        # Default
        Kcb = 0.6
        fc = 0.4

    # Water stress coefficient
    if theta_root is not None:
        p = crop_coef.p_standard if crop_coef else 0.55
        Ks = compute_water_stress_coefficient(
            theta_root, theta_fc, theta_wp, p, et0_mm
        )
    else:
        Ks = 1.0

    # Soil evaporation coefficient
    if theta_surface is not None:
        Ke, Kr = compute_soil_evaporation_coefficient(
            theta_surface, theta_fc, theta_wp, 0.05, fc
        )
    else:
        Ke = 0.2 * (1 - fc)  # Simple estimate
        Kr = 0.8

    # Actual transpiration
    transpiration_mm = Kcb * Ks * et0_mm

    # Actual evaporation
    evaporation_mm = Ke * et0_mm

    # Total ET
    et_actual_mm = transpiration_mm + evaporation_mm

    # Ensure reasonable bounds
    et_actual_mm = min(et_actual_mm, et0_mm * 1.3)  # Max ~130% ET0

    return {
        "transpiration_mm": transpiration_mm,
        "evaporation_mm": evaporation_mm,
        "et_actual_mm": et_actual_mm,
        "Kcb": Kcb,
        "Ks": Ks,
        "Ke": Ke,
        "Kr": Kr,
        "fc": fc,
    }
