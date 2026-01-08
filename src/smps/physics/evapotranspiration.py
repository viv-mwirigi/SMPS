"""
FAO-56 Dual Crop Coefficient Evapotranspiration Model.

This module implements the FAO-56 methodology for calculating
evapotranspiration with proper partitioning between:
1. Basal crop transpiration (Kcb × ET0)
2. Soil evaporation (Ke × ET0)
3. Canopy interception

References:
- Allen, R.G., Pereira, L.S., Raes, D. and Smith, M. (1998).
  Crop evapotranspiration - Guidelines for computing crop water requirements.
  FAO Irrigation and drainage paper 56. FAO, Rome.
- Allen, R.G. (2000). Using the FAO-56 dual crop coefficient method over an
  irrigated region as part of an evapotranspiration intercomparison study.
  Journal of Hydrology, 229:27-41.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CROP COEFFICIENT PARAMETERS
# =============================================================================

@dataclass
class CropCoefficientCurve:
    """
    FAO-56 crop coefficient curve parameters.

    The crop coefficient (Kc) follows a trapezoidal pattern:
    - Initial stage (Kc_ini): From planting to ~10% ground cover
    - Development stage: Linear increase from Kc_ini to Kc_mid
    - Mid-season (Kc_mid): From full cover to start of maturity
    - Late season: Linear decrease from Kc_mid to Kc_end

    The dual coefficient separates:
    - Kcb: Basal crop coefficient (transpiration when soil surface is dry)
    - Ke: Soil evaporation coefficient
    - Kc = Kcb × Ks + Ke (where Ks is water stress coefficient)
    """
    # Basal crop coefficients (transpiration)
    Kcb_ini: float = 0.15  # Initial stage
    Kcb_mid: float = 1.15  # Mid-season
    Kcb_end: float = 0.50  # End of season

    # Single crop coefficients (for reference)
    Kc_ini: float = 0.30
    Kc_mid: float = 1.20
    Kc_end: float = 0.55

    # Growth stage lengths (days)
    L_ini: int = 30  # Initial stage
    L_dev: int = 40  # Development stage
    L_mid: int = 50  # Mid-season
    L_late: int = 30  # Late season

    # Maximum crop height (m) for wind/humidity adjustments
    crop_height_m: float = 2.0

    # Rooting depth parameters
    root_depth_min_m: float = 0.10  # At emergence
    root_depth_max_m: float = 1.0  # At full development

    # Depletion fraction (p) - when stress begins
    p_standard: float = 0.55  # At ET0 = 5 mm/day

    @property
    def total_growing_season(self) -> int:
        """Total length of growing season (days)"""
        return self.L_ini + self.L_dev + self.L_mid + self.L_late

    @classmethod
    def for_crop(cls, crop_name: str) -> "CropCoefficientCurve":
        """
        Get standard crop coefficients from FAO-56 Table 12/17.

        Args:
            crop_name: Crop name (e.g., 'maize', 'wheat')

        Returns:
            CropCoefficientCurve with FAO-56 standard values
        """
        # FAO-56 Tables 12 and 17
        # Format: (Kcb_ini, Kcb_mid, Kcb_end, Kc_ini, Kc_mid, Kc_end,
        #          L_ini, L_dev, L_mid, L_late, height_m, root_depth_m, p)
        crops = {
            'maize': (0.15, 1.15, 0.50, 0.30, 1.20, 0.55, 30, 40, 50, 30, 2.0, 1.0, 0.55),
            'wheat': (0.15, 1.10, 0.25, 0.30, 1.15, 0.30, 20, 35, 40, 30, 1.0, 1.0, 0.55),
            'rice': (1.00, 1.15, 0.90, 1.05, 1.20, 0.95, 30, 30, 60, 30, 1.0, 0.5, 0.20),
            'soybean': (0.15, 1.10, 0.30, 0.40, 1.15, 0.50, 20, 35, 60, 25, 0.8, 0.6, 0.50),
            'cotton': (0.15, 1.15, 0.50, 0.35, 1.15, 0.70, 30, 50, 55, 45, 1.3, 1.3, 0.65),
            'sorghum': (0.15, 1.05, 0.35, 0.30, 1.10, 0.55, 20, 35, 40, 30, 1.5, 1.0, 0.55),
            'millet': (0.15, 1.00, 0.30, 0.30, 1.00, 0.30, 15, 25, 40, 25, 1.5, 1.0, 0.55),
            'groundnut': (0.15, 1.10, 0.50, 0.40, 1.15, 0.60, 25, 35, 45, 25, 0.4, 0.5, 0.50),
            'cassava': (0.15, 0.80, 0.30, 0.30, 1.10, 0.50, 20, 40, 90, 60, 1.0, 0.7, 0.35),
            'beans': (0.15, 1.05, 0.30, 0.40, 1.15, 0.35, 20, 30, 30, 10, 0.4, 0.5, 0.45),
            'potato': (0.15, 1.10, 0.65, 0.50, 1.15, 0.75, 25, 30, 45, 30, 0.6, 0.4, 0.35),
            'tomato': (0.15, 1.10, 0.70, 0.60, 1.15, 0.80, 30, 40, 45, 30, 0.6, 0.7, 0.40),
            'grassland': (0.30, 0.95, 0.95, 0.40, 1.00, 1.00, 10, 20, 200, 30, 0.3, 0.5, 0.50),
            'savanna': (0.20, 0.70, 0.40, 0.30, 0.80, 0.45, 30, 60, 120, 60, 1.0, 1.5, 0.60),
            'forest': (0.95, 1.00, 1.00, 1.00, 1.05, 1.05, 30, 60, 200, 30, 10.0, 2.0, 0.70),
        }

        crop_key = crop_name.lower().replace(' ', '_')

        if crop_key not in crops:
            logger.warning(f"Unknown crop '{crop_name}', using 'maize'")
            crop_key = 'maize'

        data = crops[crop_key]

        return cls(
            Kcb_ini=data[0], Kcb_mid=data[1], Kcb_end=data[2],
            Kc_ini=data[3], Kc_mid=data[4], Kc_end=data[5],
            L_ini=data[6], L_dev=data[7], L_mid=data[8], L_late=data[9],
            crop_height_m=data[10], root_depth_max_m=data[11], p_standard=data[12]
        )


def ndvi_to_lai(
    ndvi: float,
    ndvi_min: float = 0.15,
    ndvi_max: float = 0.90,
    extinction_coeff: float = 0.5
) -> float:
    """
    Estimate Leaf Area Index (LAI) from NDVI.

    Uses the relationship from Carlson & Ripley (1997):
        LAI = -ln[(NDVI_max - NDVI)/(NDVI_max - NDVI_min)] / k

    Simplified form commonly used:
        LAI = -2.0 × ln(1 - fc)

    where fc (fractional cover) ≈ (NDVI - NDVI_min)/(NDVI_max - NDVI_min)

    Args:
        ndvi: NDVI value (0-1)
        ndvi_min: NDVI value for bare soil (default: 0.15)
        ndvi_max: NDVI value for full vegetation (default: 0.90)
        extinction_coeff: Light extinction coefficient (default: 0.5)

    Returns:
        Estimated LAI (m²/m²)
    """
    ndvi = np.clip(ndvi, 0.0, 1.0)

    # Fractional cover
    fc = (ndvi - ndvi_min) / (ndvi_max - ndvi_min)
    fc = np.clip(fc, 0.01, 0.99)

    # LAI estimation using Beer-Lambert law inversion
    lai = -np.log(1 - fc) / extinction_coeff

    # Clamp to realistic range
    return np.clip(lai, 0.0, 8.0)


def lai_to_kcb(
    lai: float,
    Kcb_full: float = 1.15,
    lai_full: float = 3.0
) -> float:
    """
    Estimate basal crop coefficient from LAI.

    Based on FAO-56 Eq. 97:
        Kcb = Kcb_full × min(1, LAI/LAI_full)^0.5

    Or using the relationship:
        Kcb = Kcb_min + (Kcb_full - Kcb_min) × fc

    Args:
        lai: Leaf Area Index (m²/m²)
        Kcb_full: Kcb at full cover
        lai_full: LAI at full cover

    Returns:
        Estimated Kcb
    """
    if lai <= 0:
        return 0.15  # Bare soil Kcb

    # FAO approach
    fc = min(1.0, (lai / lai_full) ** 0.5)
    Kcb_min = 0.15

    return Kcb_min + (Kcb_full - Kcb_min) * fc


def adjust_Kcb_for_climate(
    Kcb: float,
    u2: float = 2.0,  # Wind speed at 2m (m/s)
    RH_min: float = 45.0,  # Minimum relative humidity (%)
    crop_height_m: float = 2.0
) -> float:
    """
    Adjust Kcb for local climate conditions (FAO-56 Eq. 70).

    Kcb_adj = Kcb + [0.04(u2 - 2) - 0.004(RH_min - 45)] × (h/3)^0.3

    This adjustment accounts for:
    - Higher evaporative demand in dry, windy conditions
    - Reduced demand in humid conditions

    Args:
        Kcb: Standard Kcb value
        u2: Wind speed at 2m height (m/s)
        RH_min: Minimum daily relative humidity (%)
        crop_height_m: Crop height (m)

    Returns:
        Climate-adjusted Kcb
    """
    # Clamp inputs
    u2 = np.clip(u2, 1.0, 6.0)
    RH_min = np.clip(RH_min, 20.0, 80.0)
    h = np.clip(crop_height_m, 0.1, 10.0)

    # FAO-56 Equation 70
    adjustment = (0.04 * (u2 - 2) - 0.004 * (RH_min - 45)) * (h / 3) ** 0.3

    return Kcb + adjustment


# =============================================================================
# SOIL EVAPORATION MODEL
# =============================================================================

class SoilEvaporationStage(Enum):
    """Three stages of soil evaporation"""
    STAGE1 = 1  # Energy-limited (wet soil)
    STAGE2 = 2  # Falling rate (drying)
    STAGE3 = 3  # Vapor diffusion (very dry)


@dataclass
class SoilEvaporationState:
    """
    Tracks cumulative evaporation for multi-day soil drying.

    FAO-56 uses a two-stage model:
    - Stage 1: Energy-limited evaporation at potential rate
    - Stage 2: Soil-limited evaporation (falling rate)

    We extend this with Stage 3 for very dry conditions.
    """
    cumulative_evaporation: float = 0.0  # Since last wetting (mm)
    days_since_wetting: int = 0
    current_stage: SoilEvaporationStage = SoilEvaporationStage.STAGE1

    # FAO-56 parameters
    REW: float = 9.0  # Readily evaporable water (mm), Stage 1 limit
    TEW: float = 25.0  # Total evaporable water (mm), Stage 2 limit

    def reset_after_wetting(self, wetting_mm: float):
        """Reset after significant wetting event"""
        if wetting_mm > self.REW / 2:
            # Significant wetting resets the drying cycle
            self.cumulative_evaporation = max(
                0, self.cumulative_evaporation - wetting_mm
            )
            self.days_since_wetting = 0

            if self.cumulative_evaporation < self.REW:
                self.current_stage = SoilEvaporationStage.STAGE1

    @classmethod
    def from_soil_properties(
        cls,
        theta_FC: float,
        theta_WP: float,
        Ze: float = 0.10  # Evaporating layer depth (m)
    ) -> "SoilEvaporationState":
        """
        Initialize evaporation parameters from soil properties.

        FAO-56 Equations 73 and 74:
            TEW = 1000 × (θ_FC - 0.5×θ_WP) × Ze
            REW = estimated from texture (typically 8-12 mm)

        Args:
            theta_FC: Field capacity (m³/m³)
            theta_WP: Wilting point (m³/m³)
            Ze: Depth of surface evaporation layer (m)

        Returns:
            SoilEvaporationState with calculated parameters
        """
        # Total evaporable water (FAO-56 Eq. 73)
        TEW = 1000 * (theta_FC - 0.5 * theta_WP) * Ze
        TEW = np.clip(TEW, 10, 50)

        # REW is typically 30-40% of TEW for most soils
        REW = TEW * 0.35
        REW = np.clip(REW, 6, 15)

        return cls(REW=REW, TEW=TEW)


def calculate_Kr(
    evap_state: SoilEvaporationState
) -> float:
    """
    Calculate evaporation reduction coefficient Kr.

    FAO-56 Equation 71:
        Kr = (TEW - De) / (TEW - REW)  for De > REW
        Kr = 1                          for De ≤ REW

    where De = cumulative depth of evaporation (mm)

    Args:
        evap_state: Current soil evaporation state

    Returns:
        Kr coefficient (0-1)
    """
    De = evap_state.cumulative_evaporation
    REW = evap_state.REW
    TEW = evap_state.TEW

    if De <= REW:
        return 1.0  # Stage 1
    elif De >= TEW:
        return 0.0  # Stage 3 (essentially zero evaporation)
    else:
        # Stage 2: linear reduction
        return (TEW - De) / (TEW - REW)


def calculate_Ke(
    Kcb: float,
    Kr: float,
    fc: float,
    Kc_max: float = 1.20,
    few: Optional[float] = None
) -> float:
    """
    Calculate soil evaporation coefficient Ke.

    FAO-56 Equations 71-76:
        Ke = min(Kr × (Kc_max - Kcb), few × Kc_max)

    The Ke is limited by:
    1. Energy available for evaporation after transpiration (Kc_max - Kcb)
    2. Exposed wet soil fraction (few)

    Args:
        Kcb: Basal crop coefficient
        Kr: Evaporation reduction coefficient
        fc: Fractional vegetation cover (0-1)
        Kc_max: Maximum Kc value (~1.2 for most conditions)
        few: Fraction of exposed wetted soil (defaults to 1-fc)

    Returns:
        Soil evaporation coefficient Ke
    """
    # Fraction of soil exposed to evaporation
    if few is None:
        few = 1.0 - fc
    few = np.clip(few, 0.01, 1.0)

    # Upper limit on Ke based on energy balance
    Ke_upper = Kc_max - Kcb

    # Ke limited by available energy and exposed area
    Ke = min(Kr * Ke_upper, few * Kc_max)

    return max(0.0, Ke)


def calculate_Kc_max(
    u2: float = 2.0,
    RH_min: float = 45.0,
    crop_height_m: float = 2.0
) -> float:
    """
    Calculate maximum Kc value for climate (FAO-56 Eq. 72).

    Kc_max = max{1.2 + [0.04(u2-2) - 0.004(RH_min-45)](h/3)^0.3, Kcb + 0.05}

    Args:
        u2: Wind speed at 2m (m/s)
        RH_min: Minimum relative humidity (%)
        crop_height_m: Crop height (m)

    Returns:
        Kc_max value
    """
    u2 = np.clip(u2, 1.0, 6.0)
    RH_min = np.clip(RH_min, 20.0, 80.0)
    h = np.clip(crop_height_m, 0.1, 10.0)

    adjustment = (0.04 * (u2 - 2) - 0.004 * (RH_min - 45)) * (h / 3) ** 0.3

    return 1.2 + adjustment


# =============================================================================
# CANOPY INTERCEPTION
# =============================================================================

@dataclass
class InterceptionParameters:
    """Parameters for canopy rainfall interception"""
    # Storage capacity (mm)
    storage_capacity_min: float = 0.5  # Sparse canopy
    storage_capacity_max: float = 2.5  # Dense canopy

    # Throughfall coefficient
    throughfall_coef: float = 0.1  # Fraction that falls through immediately

    # Current storage
    current_storage: float = 0.0


def canopy_interception(
    precipitation_mm: float,
    lai: float,
    et0_mm: float,
    params: InterceptionParameters
) -> Tuple[float, float, float]:
    """
    Calculate canopy interception, throughfall, and interception evaporation.

    Based on the Rutter model simplified approach:
        I_max = S × LAI
        I = min(P, I_max - current_storage)

    Interception evaporates at potential rate.

    Args:
        precipitation_mm: Daily precipitation (mm)
        lai: Leaf Area Index (m²/m²)
        et0_mm: Reference ET (mm/day)
        params: Interception parameters

    Returns:
        Tuple of (throughfall_mm, interception_evap_mm, new_storage_mm)
    """
    if precipitation_mm <= 0:
        # No precip - only evaporate existing storage
        # Wet canopy evaporates fast
        evap = min(params.current_storage, et0_mm * 1.2)
        params.current_storage -= evap
        return 0.0, evap, params.current_storage

    # Storage capacity depends on LAI
    # S typically 0.5-2.5 mm, scaling with LAI
    fc = min(1.0, lai / 3.0)  # Fractional cover
    storage_capacity = (
        params.storage_capacity_min +
        (params.storage_capacity_max - params.storage_capacity_min) * fc
    )

    # Direct throughfall (between gaps)
    direct_throughfall = precipitation_mm * params.throughfall_coef * (1 - fc)

    # Available for interception
    available = precipitation_mm - direct_throughfall

    # Interception limited by remaining storage capacity
    space_available = max(0, storage_capacity - params.current_storage)
    interception = min(available, space_available)

    # Update storage
    params.current_storage += interception

    # Drainage from canopy (excess)
    drainage = available - interception

    # Total throughfall
    throughfall = direct_throughfall + drainage

    # Interception evaporation (at ~1.2 × ET0 for wet canopy)
    # But limited to what's stored
    evap_potential = et0_mm * 1.2 * fc
    interception_evap = min(params.current_storage, evap_potential)
    params.current_storage -= interception_evap

    return throughfall, interception_evap, params.current_storage


# =============================================================================
# WATER STRESS COEFFICIENT
# =============================================================================

def calculate_p_adjusted(
    p_standard: float,
    ET0: float,
    ET0_ref: float = 5.0
) -> float:
    """
    Adjust depletion fraction p for ET demand (FAO-56 Eq. 84).

    p = p_standard + 0.04 × (5 - ET0)

    Higher ET demand → stress begins sooner (lower p)
    Lower ET demand → stress begins later (higher p)

    Args:
        p_standard: Standard depletion fraction at ET0=5 mm/day
        ET0: Current reference ET (mm/day)
        ET0_ref: Reference ET for standard p (5 mm/day)

    Returns:
        Adjusted p value
    """
    p_adj = p_standard + 0.04 * (ET0_ref - ET0)
    return np.clip(p_adj, 0.1, 0.8)


def calculate_Ks(
    theta: float,
    theta_FC: float,
    theta_WP: float,
    p: float = 0.55
) -> float:
    """
    Calculate water stress coefficient Ks (FAO-56).

    When θ is above threshold (θ_t = θ_FC - p×TAW):
        Ks = 1 (no stress)
    When θ is between θ_t and θ_WP:
        Ks = (θ - θ_WP) / (θ_t - θ_WP)
    When θ is below θ_WP:
        Ks = 0 (complete stress)

    Args:
        theta: Current water content (m³/m³)
        theta_FC: Field capacity (m³/m³)
        theta_WP: Wilting point (m³/m³)
        p: Depletion fraction (when stress begins)

    Returns:
        Ks coefficient (0-1)
    """
    # Total available water
    TAW = theta_FC - theta_WP

    if TAW <= 0:
        return 0.0

    # Threshold water content
    theta_t = theta_FC - p * TAW

    if theta >= theta_t:
        return 1.0  # No stress
    elif theta <= theta_WP:
        return 0.0  # Complete stress
    else:
        # Linear reduction
        RAW = theta_t - theta_WP  # Readily available water range
        return (theta - theta_WP) / RAW


def calculate_Ks_from_pressure_head(
    psi: float,
    psi_3: float = -100.0,  # -1000 kPa ≈ -100 m
    psi_4: float = -150.0   # -1500 kPa ≈ -150 m
) -> float:
    """
    Calculate Ks using pressure head (matric potential).

    This is more physically accurate than water content-based
    because wilting point is actually defined by pressure, not θ.

    Args:
        psi: Current pressure head (m, negative)
        psi_3: Pressure head at stress onset (m)
        psi_4: Pressure head at wilting point (m)

    Returns:
        Ks coefficient (0-1)
    """
    if psi >= psi_3:
        return 1.0  # No stress
    elif psi <= psi_4:
        return 0.0  # Complete stress
    else:
        return (psi - psi_4) / (psi_3 - psi_4)


# =============================================================================
# MAIN ET CALCULATION FUNCTIONS
# =============================================================================

@dataclass
class ETResult:
    """Results from ET calculation"""
    ET_c: float  # Total crop ET (mm/day)
    E_s: float  # Soil evaporation (mm/day)
    T_c: float  # Crop transpiration (mm/day)
    E_int: float  # Interception evaporation (mm/day)
    Kcb: float  # Basal crop coefficient
    Ke: float  # Soil evaporation coefficient
    Ks: float  # Water stress coefficient
    Kr: float  # Evaporation reduction coefficient


def calculate_et_fao56_dual(
    ET0: float,
    ndvi: Optional[float] = None,
    lai: Optional[float] = None,
    theta_surface: float = 0.25,
    theta_FC: float = 0.30,
    theta_WP: float = 0.10,
    evap_state: Optional[SoilEvaporationState] = None,
    crop_params: Optional[CropCoefficientCurve] = None,
    day_of_season: int = 60,
    precipitation_mm: float = 0.0,
    u2: float = 2.0,
    RH_min: float = 45.0
) -> ETResult:
    """
    Calculate ET using FAO-56 dual crop coefficient method.

    ET_c = (Ks × Kcb + Ke) × ET0

    This is the main function for daily ET partitioning.

    Args:
        ET0: Reference evapotranspiration (mm/day)
        ndvi: NDVI value (0-1), used if LAI not provided
        lai: Leaf Area Index (m²/m²)
        theta_surface: Surface layer water content (m³/m³)
        theta_FC: Field capacity (m³/m³)
        theta_WP: Wilting point (m³/m³)
        evap_state: Soil evaporation state tracker
        crop_params: Crop coefficient curve
        day_of_season: Day since planting/emergence
        precipitation_mm: Daily precipitation (mm)
        u2: Wind speed at 2m (m/s)
        RH_min: Minimum relative humidity (%)

    Returns:
        ETResult with partitioned ET components
    """
    if ET0 <= 0:
        return ETResult(0, 0, 0, 0, 0, 0, 1.0, 1.0)

    # Default crop parameters if not provided
    if crop_params is None:
        crop_params = CropCoefficientCurve.for_crop('grassland')

    # Initialize evaporation state if needed
    if evap_state is None:
        evap_state = SoilEvaporationState.from_soil_properties(
            theta_FC, theta_WP)

    # Estimate LAI from NDVI if not provided
    if lai is None:
        if ndvi is not None:
            lai = ndvi_to_lai(ndvi)
        else:
            lai = 2.0  # Default moderate vegetation

    # Calculate fractional cover from LAI
    fc = 1.0 - np.exp(-0.5 * lai)  # Beer-Lambert with k=0.5
    fc = np.clip(fc, 0.01, 0.99)

    # Get Kcb from crop curve or estimate from LAI
    Kcb = get_Kcb_from_curve(crop_params, day_of_season)

    # Alternatively, estimate from LAI if mid-season
    Kcb_from_lai = lai_to_kcb(lai, crop_params.Kcb_mid)

    # Use LAI-based Kcb if significantly different and LAI is reliable
    if ndvi is not None and 0.3 < ndvi < 0.85:
        # Blend curve-based and LAI-based estimates
        Kcb = 0.5 * Kcb + 0.5 * Kcb_from_lai

    # Adjust Kcb for climate
    Kcb_adj = adjust_Kcb_for_climate(
        Kcb, u2, RH_min, crop_params.crop_height_m)

    # Calculate water stress coefficient
    p_adj = calculate_p_adjusted(crop_params.p_standard, ET0)
    Ks = calculate_Ks(theta_surface, theta_FC, theta_WP, p_adj)

    # Update evaporation state for wetting
    if precipitation_mm > 0:
        evap_state.reset_after_wetting(precipitation_mm)

    # Calculate Kr and Ke
    Kr = calculate_Kr(evap_state)
    Kc_max = calculate_Kc_max(u2, RH_min, crop_params.crop_height_m)
    Ke = calculate_Ke(Kcb_adj, Kr, fc, Kc_max)

    # Calculate ET components
    T_c = Ks * Kcb_adj * ET0  # Transpiration
    E_s = Ke * ET0  # Soil evaporation

    # Update cumulative evaporation for next timestep
    evap_state.cumulative_evaporation += E_s
    evap_state.days_since_wetting += 1

    # Total ET
    ET_c = T_c + E_s

    return ETResult(
        ET_c=ET_c,
        E_s=E_s,
        T_c=T_c,
        E_int=0.0,  # Calculated separately if canopy interception used
        Kcb=Kcb_adj,
        Ke=Ke,
        Ks=Ks,
        Kr=Kr
    )


def get_Kcb_from_curve(
    params: CropCoefficientCurve,
    day_of_season: int
) -> float:
    """
    Get Kcb value from trapezoidal crop curve.

    Args:
        params: Crop coefficient parameters
        day_of_season: Days since planting/emergence

    Returns:
        Kcb value for current growth stage
    """
    L_ini = params.L_ini
    L_dev = params.L_dev
    L_mid = params.L_mid
    L_late = params.L_late

    # Cumulative days at end of each stage
    end_ini = L_ini
    end_dev = end_ini + L_dev
    end_mid = end_dev + L_mid
    end_late = end_mid + L_late

    day = min(day_of_season, end_late)

    if day <= end_ini:
        # Initial stage
        return params.Kcb_ini

    elif day <= end_dev:
        # Development stage - linear increase
        progress = (day - end_ini) / L_dev
        return params.Kcb_ini + progress * (params.Kcb_mid - params.Kcb_ini)

    elif day <= end_mid:
        # Mid-season
        return params.Kcb_mid

    else:
        # Late season - linear decrease
        progress = (day - end_mid) / L_late
        return params.Kcb_mid - progress * (params.Kcb_mid - params.Kcb_end)


def compare_single_vs_dual_Kc(
    ET0: float,
    ndvi: float,
    theta_surface: float,
    theta_FC: float = 0.30,
    theta_WP: float = 0.10,
    lambda_ndvi: float = 2.0
) -> Dict[str, float]:
    """
    Compare simple NDVI-based partitioning vs FAO-56 dual coefficient.

    Useful for understanding the improvement from FAO-56.

    Args:
        ET0: Reference ET (mm/day)
        ndvi: NDVI value
        theta_surface: Surface water content
        theta_FC: Field capacity
        theta_WP: Wilting point
        lambda_ndvi: Exponential coefficient for simple method

    Returns:
        Dictionary comparing both approaches
    """
    # Simple method (original)
    evap_frac_simple = np.exp(-lambda_ndvi * ndvi)
    transp_frac_simple = 1.0 - evap_frac_simple

    E_simple = ET0 * evap_frac_simple
    T_simple = ET0 * transp_frac_simple

    # FAO-56 dual method
    result_fao = calculate_et_fao56_dual(
        ET0=ET0,
        ndvi=ndvi,
        theta_surface=theta_surface,
        theta_FC=theta_FC,
        theta_WP=theta_WP
    )

    return {
        'simple_evaporation': E_simple,
        'simple_transpiration': T_simple,
        'fao56_evaporation': result_fao.E_s,
        'fao56_transpiration': result_fao.T_c,
        'evaporation_difference': result_fao.E_s - E_simple,
        'transpiration_difference': result_fao.T_c - T_simple,
        'Kcb': result_fao.Kcb,
        'Ke': result_fao.Ke,
        'Ks': result_fao.Ks
    }
