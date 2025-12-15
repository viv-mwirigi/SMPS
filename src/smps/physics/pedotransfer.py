"""
Pedotransfer functions for estimating soil hydraulic parameters from texture.
Implements multiple established methods with uncertainty estimation.
"""
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from smps.core.types import SoilParameters



@dataclass
class TextureClass:
    """Soil texture classification"""
    name: str
    sand_range: Tuple[float, float]
    clay_range: Tuple[float, float]
    silt_range: Tuple[float, float]
    usda_class: str
    fao_class: str


# USDA soil texture classes
TEXTURE_CLASSES = [
    TextureClass("sand", (85, 100), (0, 10), (0, 15), "Sand", "Sand"),
    TextureClass("loamy_sand", (70, 90), (0, 15), (0, 30), "Loamy Sand", "Loamy Sand"),
    TextureClass("sandy_loam", (43, 85), (0, 20), (0, 50), "Sandy Loam", "Sandy Loam"),
    TextureClass("loam", (23, 52), (7, 27), (28, 50), "Loam", "Loam"),
    TextureClass("silt_loam", (0, 50), (0, 27), (50, 88), "Silt Loam", "Silt Loam"),
    TextureClass("silt", (0, 20), (0, 12), (80, 100), "Silt", "Silt"),
    TextureClass("clay_loam", (20, 45), (27, 40), (15, 53), "Clay Loam", "Clay Loam"),
    TextureClass("silty_clay_loam", (0, 20), (27, 40), (40, 73), "Silty Clay Loam", "Silty Clay Loam"),
    TextureClass("clay", (0, 45), (40, 100), (0, 40), "Clay", "Clay"),
]


def classify_soil_texture(
    sand_percent: float,
    clay_percent: float,
    silt_percent: Optional[float] = None
) -> str:
    """
    Classify soil texture using USDA texture triangle.

    Args:
        sand_percent: Sand content (%)
        clay_percent: Clay content (%)
        silt_percent: Silt content (%), optional

    Returns:
        Texture class name
    """
    # Calculate silt if not provided
    if silt_percent is None:
        silt_percent = 100 - sand_percent - clay_percent

    # Normalize to 100%
    total = sand_percent + clay_percent + silt_percent
    if abs(total - 100) > 1:
        sand_percent = sand_percent / total * 100
        clay_percent = clay_percent / total * 100
        silt_percent = silt_percent / total * 100

    # Check each texture class
    for texture_class in TEXTURE_CLASSES:
        if (texture_class.sand_range[0] <= sand_percent <= texture_class.sand_range[1] and
            texture_class.clay_range[0] <= clay_percent <= texture_class.clay_range[1] and
            texture_class.silt_range[0] <= silt_percent <= texture_class.silt_range[1]):
            return texture_class.name

    # If no exact match, find closest class
    return _find_closest_texture_class(sand_percent, clay_percent, silt_percent)


def _find_closest_texture_class(
    sand: float,
    clay: float,
    silt: float
) -> str:
    """Find closest texture class using Euclidean distance in texture triangle"""
    # Texture triangle coordinates (simplified)
    texture_coords = {
        "sand": (90, 10, 0),
        "loamy_sand": (80, 10, 10),
        "sandy_loam": (60, 10, 30),
        "loam": (40, 20, 40),
        "silt_loam": (20, 10, 70),
        "silt": (10, 5, 85),
        "clay_loam": (30, 35, 35),
        "silty_clay_loam": (10, 35, 55),
        "clay": (20, 60, 20),
    }

    point = (sand, clay, silt)
    distances = {}

    for texture, coords in texture_coords.items():
        distance = np.sqrt(
            (point[0] - coords[0])**2 +
            (point[1] - coords[1])**2 +
            (point[2] - coords[2])**2
        )
        distances[texture] = distance

    return min(distances.items(), key=lambda x: x[1])[0]


def estimate_soil_parameters_saxton(
    sand_percent: float,
    clay_percent: float,
    organic_matter_percent: float = 2.0,
    bulk_density_g_cm3: Optional[float] = None,
    is_tropical_oxide_soil: bool = False
) -> SoilParameters:
    """
    Estimate soil hydraulic parameters using Saxton & Rawls (2006) pedotransfer functions.

    Reference: Saxton, K.E., Rawls, W.J., 2006. Soil Water Characteristic Estimates by
               Texture and Organic Matter for Hydrologic Solutions.

    Args:
        sand_percent: Sand content (%)
        clay_percent: Clay content (%)
        organic_matter_percent: Organic matter content (%)
        bulk_density_g_cm3: Bulk density (g/cm³), optional

    Returns:
        SoilParameters with estimated values
    """
    # Convert to fraction
    sand = sand_percent / 100.0
    clay = clay_percent / 100.0
    om = organic_matter_percent / 100.0

    # Equation 1: Saturated water content (θ_sat)
    # θ_sat = 0.332 - 7.251e-4 * SAND + 0.1276 * log10(CLAY) + 0.002 * OM
    clay_for_log = max(1.0, clay_percent)  # Avoid log10(0)
    porosity = (
        0.332 -
        7.251e-4 * sand_percent +
        0.1276 * np.log10(clay_for_log) +
        0.002 * om
    )

    # Equation 2: Field capacity at -33 kPa (θ_33)
    # θ_33 = 0.2576 - 0.002 * SAND + 0.0036 * CLAY + 0.0299 * OM
    field_capacity = (
        0.2576 -
        0.002 * sand_percent +
        0.0036 * clay_percent +
        0.0299 * om
    )

    # Equation 3: Wilting point at -1500 kPa (θ_1500)
    # θ_1500 = 0.026 + 0.005 * CLAY + 0.0158 * OM
    wilting_point = (
        0.026 +
        0.005 * clay_percent +
        0.0158 * om
    )

    # Equation for saturated hydraulic conductivity (K_sat)
    # log10(K_sat) = 0.6 + 0.0126 * SAND - 0.0064 * CLAY - 0.0127 * OM
    # K_sat in mm/hour, convert to cm/day
    log_k_sat = (
        0.6 +
        0.0126 * sand_percent -
        0.0064 * clay_percent -
        0.0127 * om
    )
    k_sat_mm_hr = 10 ** log_k_sat
    k_sat_cm_day = k_sat_mm_hr * 2.4  # mm/hr → cm/day

    # Van Genuchten parameters from Saxton & Rawls Appendix
    # α = exp(-0.784 + 0.0178 * SAND - 0.0112 * CLAY - 0.0074 * OM) [1/cm]
    log_alpha = (
        -0.784 +
        0.0178 * sand_percent -
        0.0112 * clay_percent -
        0.0074 * om
    )
    alpha_cm = np.exp(log_alpha)
    alpha_kpa = alpha_cm * 10.2  # 1/cm → 1/kPa

    # n = 1 + exp(-0.63 - 0.0063 * SAND + 0.0121 * CLAY - 0.0064 * OM)
    log_n = (
        -0.63 -
        0.0063 * sand_percent +
        0.0121 * clay_percent -
        0.0064 * om
    )
    n = 1 + np.exp(log_n)

    # Bulk density estimation (Eq. 4)
    if bulk_density_g_cm3 is None:
        # BD = 1.51 + 0.0045 * SAND - 0.0057 * CLAY - 0.0135 * OM
        bulk_density_g_cm3 = (
            1.51 +
            0.0045 * sand_percent -
            0.0057 * clay_percent -
            0.0135 * om
        )

    if is_tropical_oxide_soil:
        # Correction for Ferralsols/Oxisols (common in humid Africa)
        # These soils have high clay but drain fast due to micro-aggregation.

        # Increase Ksat (drainage speed) significantly
        # Experimental multiplier often 2x to 10x; start conservative.
        k_sat_cm_day *= 2.0

        # Decrease Field Capacity (holds less water against gravity than expected)
        field_capacity *= 0.85

        # Adjust Porosity slightly up (granular structure)
        porosity = min(porosity * 1.05, 0.65)

    # Apply physical bounds
    porosity = np.clip(porosity, 0.30, 0.60)
    field_capacity = np.clip(field_capacity, 0.05, 0.50)
    wilting_point = np.clip(wilting_point, 0.01, 0.35)
    k_sat_cm_day = np.clip(k_sat_cm_day, 0.1, 1000.0)
    alpha_kpa = np.clip(alpha_kpa, 0.001, 0.5)
    n = np.clip(n, 1.05, 2.8)
    bulk_density_g_cm3 = np.clip(bulk_density_g_cm3, 1.0, 1.8)

    return SoilParameters(
        sand_percent=sand_percent,
        silt_percent=100 - sand_percent - clay_percent,
        clay_percent=clay_percent,
        porosity=porosity,
        field_capacity=field_capacity,
        wilting_point=wilting_point,
        saturated_hydraulic_conductivity_cm_day=k_sat_cm_day,
        van_genuchten_alpha=alpha_kpa,
        van_genuchten_n=n,
        bulk_density_g_cm3=bulk_density_g_cm3,
        organic_matter_percent=om
    )


def estimate_van_genuchten_parameters(
    sand_percent: float,
    clay_percent: float,
    porosity: float
) -> Tuple[float, float]:
    """
    Estimate van Genuchten parameters (α, n) from texture and porosity.

    Based on ROSETTA model (Schaap et al., 2001) simplified relationships.

    Returns:
        Tuple of (α [1/cm], n [-])
    """
    # Simplified relationships
    # α: inversely related to clay content
    alpha_cm = 0.1 * np.exp(-0.05 * clay_percent)  # 1/cm

    # Convert to 1/kPa for consistency
    alpha_kpa = alpha_cm * 10.2  # 1/cm → 1/kPa (approx)

    # n: related to pore size distribution
    n = 1.2 + 0.02 * sand_percent - 0.01 * clay_percent

    # Ensure bounds
    alpha_kpa = np.clip(alpha_kpa, 0.001, 0.5)
    n = np.clip(n, 1.1, 2.5)

    return alpha_kpa, n


from typing import Literal

def estimate_soil_parameters_rosetta(
    sand_percent: float,
    clay_percent: float,
    bulk_density_g_cm3: Optional[float] = None,
    organic_matter_percent: float = 2.0,
    model_level: Literal[1, 2, 3, 4, 5] = 2
) -> SoilParameters:
    """
    Estimate soil hydraulic parameters using ROSETTA model (Schaap et al., 2001).

    Model levels:
      1: Texture class only
      2: Texture class + bulk density
      3: Sand, clay, BD
      4: Sand, clay, silt, BD
      5: Sand, clay, silt, BD, θ_33, θ_1500

    Note: This is a simplified implementation. For production, use the actual ROSETTA model.
    """
    silt_percent = 100 - sand_percent - clay_percent

    # Default bulk density if not provided
    if bulk_density_g_cm3 is None:
        # Estimate from texture (Rawls et al., 1982)
        bulk_density_g_cm3 = (
            1.0 +
            0.004 * sand_percent +
            0.002 * clay_percent -
            0.01 * organic_matter_percent
        )

    # Van Genuchten parameters from ROSETTA Level 2 (neural network)
    # Simplified relationships based on ROSETTA

    # α parameter (1/cm)
    log_alpha = (
        0.65 -
        0.78 * (sand_percent/100) +
        0.60 * (clay_percent/100) -
        0.13 * bulk_density_g_cm3
    )
    alpha_cm = np.exp(log_alpha)
    alpha_kpa = alpha_cm * 10.2

    # n parameter
    n = (
        1.1 +
        0.4 * (sand_percent/100) -
        0.3 * (clay_percent/100) +
        0.1 * bulk_density_g_cm3
    )

    # θ_s (saturation)
    porosity = (
        0.81 -
        0.283 * bulk_density_g_cm3 +
        0.001 * clay_percent
    )

    # θ_r (residual)
    theta_r = 0.01 + 0.005 * (clay_percent/100)

    # K_s (saturated hydraulic conductivity, cm/day)
    log_k_sat = (
        1.5 -
        0.95 * (clay_percent/100) +
        0.65 * (sand_percent/100) -
        0.2 * bulk_density_g_cm3
    )
    k_sat_cm_day = 10 ** log_k_sat

    # Field capacity and wilting point from van Genuchten
    # θ at -33 kPa and -1500 kPa
    from smps.physics.van_genuchten import calculate_theta_from_pressure

    field_capacity = calculate_theta_from_pressure(
        pressure_head=-33,  # kPa
        theta_r=theta_r,
        theta_s=porosity,
        alpha=alpha_kpa,
        n=n
    )

    wilting_point = calculate_theta_from_pressure(
        pressure_head=-1500,  # kPa
        theta_r=theta_r,
        theta_s=porosity,
        alpha=alpha_kpa,
        n=n
    )

    return SoilParameters(
        sand_percent=sand_percent,
        silt_percent=silt_percent,
        clay_percent=clay_percent,
        porosity=porosity,
        field_capacity=field_capacity,
        wilting_point=wilting_point,
        saturated_hydraulic_conductivity_cm_day=k_sat_cm_day,
        van_genuchten_alpha=alpha_kpa,
        van_genuchten_n=n,
        bulk_density_g_cm3=bulk_density_g_cm3
    )




def classify_soil_texture_usda(
    sand_percent: float,
    clay_percent: float
) -> str:
    """
    USDA soil texture classification using texture triangle.

    Based on USDA NRCS texture triangle with exact boundaries.
    """
    silt_percent = 100 - sand_percent - clay_percent

    # USDA texture triangle decision tree
    if clay_percent >= 40:
        if silt_percent >= 40:
            return "silty_clay"
        elif sand_percent >= 45:
            return "sandy_clay"
        else:
            return "clay"

    elif clay_percent >= 27:
        if silt_percent >= 40:
            return "silty_clay_loam"
        elif sand_percent >= 45:
            return "sandy_clay_loam"
        else:
            return "clay_loam"

    elif clay_percent >= 20:
        if silt_percent >= 28:
            return "silt_loam"
        elif sand_percent >= 52:
            return "sandy_loam"
        else:
            return "loam"

    elif clay_percent >= 7:
        if silt_percent >= 50:
            return "silt_loam"
        elif sand_percent >= 52:
            return "sandy_loam"
        elif sand_percent >= 43:
            return "loam"
        else:
            return "silt_loam"

    else:  # clay < 7%
        if silt_percent >= 50:
            return "silt"
        elif sand_percent >= 85:
            return "sand"
        elif sand_percent >= 70:
            return "loamy_sand"
        else:
            return "sandy_loam"




def validate_soil_parameters(params: SoilParameters) -> Dict[str, str]:
    """
    Validate soil parameters for physical plausibility.

    Returns:
        Dictionary of warnings for suspicious values
    """
    warnings = {}

    # Check texture percentages sum to 100%
    total_texture = params.sand_percent + params.silt_percent + params.clay_percent
    if abs(total_texture - 100) > 1:
        warnings['texture_sum'] = f"Texture percentages sum to {total_texture:.1f}%, not 100%"

    # Check porosity bounds
    if params.porosity < 0.3 or params.porosity > 0.6:
        warnings['porosity'] = f"Porosity {params.porosity:.3f} outside typical range (0.3-0.6)"

    # Check field capacity > wilting point
    if params.field_capacity <= params.wilting_point:
        warnings['water_retention'] = (
            f"Field capacity ({params.field_capacity:.3f}) ≤ "
            f"wilting point ({params.wilting_point:.3f})"
        )

    # Check porosity > field capacity
    if params.porosity <= params.field_capacity:
        warnings['saturation'] = (
            f"Porosity ({params.porosity:.3f}) ≤ "
            f"field capacity ({params.field_capacity:.3f})"
        )

    # Check van Genuchten parameters
    if params.van_genuchten_alpha <= 0 or params.van_genuchten_alpha > 1:
        warnings['van_genuchten_alpha'] = (
            f"α parameter {params.van_genuchten_alpha:.3f} outside typical range (0.001-0.5)"
        )

    if params.van_genuchten_n < 1.1 or params.van_genuchten_n > 3.0:
        warnings['van_genuchten_n'] = (
            f"n parameter {params.van_genuchten_n:.3f} outside typical range (1.1-3.0)"
        )

    return warnings


def check_consistency_across_methods(
    sand_percent: float,
    clay_percent: float,
    methods: List[str] = ["saxton", "texture_class", "rosetta"]
) -> Dict[str, Dict[str, float]]:
    """
    Compare parameter estimates from different PTFs.

    Useful for identifying outliers and building consensus estimates.
    """
    results = {}

    for method in methods:
        try:
            if method == "saxton":
                params = estimate_soil_parameters_saxton(sand_percent, clay_percent)

            elif method == "rosetta":
                params = estimate_soil_parameters_rosetta(sand_percent, clay_percent)
            else:
                continue

            results[method] = {
                'porosity': params.porosity,
                'field_capacity': params.field_capacity,
                'wilting_point': params.wilting_point,
                'k_sat': params.saturated_hydraulic_conductivity_cm_day,
                'alpha': params.van_genuchten_alpha,
                'n': params.van_genuchten_n
            }
        except Exception as e:
            results[method] = {'error': str(e)}

    return results

def convert_units(
    params: SoilParameters,
    target_units: Dict[str, str]
) -> SoilParameters:
    """
    Convert soil parameters between different unit systems.

    Supported conversions:
    - k_sat: cm/day ↔ mm/hour ↔ m/s
    - alpha: 1/kPa ↔ 1/cm
    - water content: m³/m³ ↔ % by volume ↔ mm water per m depth
    """
    converted = params.copy()

    for param_name, target_unit in target_units.items():
        value = getattr(params, param_name)

        if param_name == 'saturated_hydraulic_conductivity_cm_day':
            if target_unit == 'mm_per_hour':
                converted_value = value / 2.4
            elif target_unit == 'm_per_s':
                converted_value = value * 1.157e-7
            else:
                continue
            setattr(converted, param_name, converted_value)

        elif param_name == 'van_genuchten_alpha':
            if target_unit == 'per_cm' and params.van_genuchten_alpha is not None:
                # Assuming alpha is in 1/kPa
                converted_value = params.van_genuchten_alpha / 10.2
                setattr(converted, param_name, converted_value)

        elif param_name in ['porosity', 'field_capacity', 'wilting_point']:
            if target_unit == 'percent':
                converted_value = value * 100
                setattr(converted, param_name, converted_value)
            elif target_unit == 'mm_per_m':
                # mm water per m soil depth
                converted_value = value * 1000
                setattr(converted, param_name, converted_value)

    return converted

def estimate_parameter_uncertainty(
    soil_params: SoilParameters,
    confidence_level: float = 0.95
) -> Dict[str, Tuple[float, float]]:
    """
    Estimate uncertainty ranges for soil parameters.

    Based on typical coefficients of variation from pedotransfer functions.

    Returns:
        Dictionary with (lower_bound, upper_bound) for each parameter
    """
    # Typical coefficients of variation (CV) for pedotransfer estimates
    cv_map = {
        "porosity": 0.10,  # ±10%
        "field_capacity": 0.15,  # ±15%
        "wilting_point": 0.20,  # ±20%
        "saturated_hydraulic_conductivity_cm_day": 0.50,  # ±50% (high uncertainty)
        "van_genuchten_alpha": 0.30,
        "van_genuchten_n": 0.15,
    }

    # Z-score for confidence level
    z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence_level, 1.96)

    uncertainties = {}

    for param_name, cv in cv_map.items():
        value = getattr(soil_params, param_name)
        if value is not None:
            std_dev = value * cv
            margin = z_score * std_dev
            uncertainties[param_name] = (
                max(0, value - margin),
                value + margin
            )

    return uncertainties


# Convenience alias used by other modules
def estimate_soil_parameters(
    sand_percent: float,
    clay_percent: float,
    organic_matter_percent: float = 2.0,
    bulk_density_g_cm3: Optional[float] = None,
    is_tropical_oxide_soil: bool = False
) -> SoilParameters:
    """
    Default soil parameter estimator (Saxton & Rawls 2006).
    Kept for backward compatibility with callers expecting `estimate_soil_parameters`.
    """
    return estimate_soil_parameters_saxton(
        sand_percent=sand_percent,
        clay_percent=clay_percent,
        organic_matter_percent=organic_matter_percent,
        bulk_density_g_cm3=bulk_density_g_cm3,
        is_tropical_oxide_soil=is_tropical_oxide_soil
    )


def create_default_soil_parameters() -> SoilParameters:
    """Fallback loam-like soil parameters for initialization."""
    return SoilParameters(
        sand_percent=40,
        silt_percent=40,
        clay_percent=20,
        porosity=0.45,
        field_capacity=0.27,
        wilting_point=0.15,
        saturated_hydraulic_conductivity_cm_day=25.0,
        van_genuchten_alpha=0.04,
        van_genuchten_n=1.5,
        bulk_density_g_cm3=1.4,
        organic_matter_percent=0.02
    )