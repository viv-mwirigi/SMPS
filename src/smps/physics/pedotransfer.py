"""
Pedotransfer functions for estimating soil hydraulic parameters from texture.

Implements multiple established methods with uncertainty estimation.
Includes tropical soil corrections (Gap 6) for:
- Enhanced organic matter effects on water retention
- Structure factor for Ksat adjustment
- Parameter distribution representation

References:
- Saxton & Rawls (2006) Soil Water Characteristic Estimates
- Wosten et al. (1999) Development of pedotransfer functions for HYPRES
- Hodnett & Tomasella (2002) PTFs for tropical soils
- Minasny & Hartemink (2011) Tropical soil properties
"""
from typing import Literal
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass, field
import logging

from smps.core.types import SoilParameters

logger = logging.getLogger(__name__)


# =============================================================================
# TROPICAL SOIL CORRECTIONS (Gap 6)
# =============================================================================

@dataclass
class TropicalSoilCorrections:
    """
    Corrections for tropical soil properties not captured by standard PTFs.

    African tropical soils often have:
    - 2-3× higher organic matter in surface layers
    - Well-aggregated structure (oxide clays act like sand)
    - Higher infiltration due to biological activity

    These corrections should be applied when:
    - Location is in tropical zone (23.5°N to 23.5°S)
    - Soils are weathered (Ferralsols, Acrisols, Nitisols)
    """
    # Organic matter correction factor
    # Increase FC by this amount per 1% OM above baseline
    fc_per_percent_om: float = 0.01  # m³/m³ per % OM

    # Structure factor for Ksat (1.0 = no correction)
    # Well-aggregated tropical soils: 1.5-3.0
    # Massive/compacted soils: 0.5-1.0
    structure_factor: float = 1.0

    # Baseline OM for which Saxton-Rawls was calibrated
    baseline_om_percent: float = 2.0

    # Oxide clay aggregation factor
    # Reduces effective clay content for water retention
    oxide_aggregation: float = 1.0  # 1.0 = standard, < 1.0 = more aggregated

    # Biological macropore factor
    # Increases Ksat due to termite/ant activity
    macropore_factor: float = 1.0  # 1.0 = standard, > 1.0 = more macropores

    @classmethod
    def for_african_soil(cls, soil_type: str = "ferralsol") -> "TropicalSoilCorrections":
        """
        Get standard corrections for African soil types.

        Args:
            soil_type: WRB soil classification

        Returns:
            TropicalSoilCorrections configured for soil type
        """
        soil_params = {
            # Well-aggregated oxide soils
            'ferralsol': (0.01, 2.0, 0.6, 2.0),  # fc_om, struct, oxide, macro
            'acrisol': (0.01, 1.5, 0.7, 1.5),
            'nitisol': (0.01, 1.8, 0.65, 1.8),

            # Vertic (shrink-swell) soils
            'vertisol': (0.012, 0.8, 1.0, 0.8),

            # Sandy coastal soils
            'arenosol': (0.008, 1.0, 1.0, 1.2),

            # Young volcanic soils
            'andosol': (0.015, 1.3, 0.8, 1.5),

            # Default tropical
            'default': (0.01, 1.5, 0.8, 1.5),
        }

        params = soil_params.get(soil_type.lower(), soil_params['default'])

        return cls(
            fc_per_percent_om=params[0],
            structure_factor=params[1],
            oxide_aggregation=params[2],
            macropore_factor=params[3]
        )


@dataclass
class SoilParameterDistribution:
    """
    Represent soil parameters as distributions rather than point estimates.

    This addresses the uncertainty inherent in pedotransfer functions
    (typically CV = 20-50% for Ksat, 10-20% for water retention).

    Uses log-normal distribution for Ksat (always positive, right-skewed)
    and truncated normal for water contents (bounded 0-1).
    """
    # Mean (expected) values
    porosity_mean: float = 0.45
    field_capacity_mean: float = 0.25
    wilting_point_mean: float = 0.12
    ksat_mean_cm_day: float = 50.0

    # Coefficients of variation (CV = std/mean)
    porosity_cv: float = 0.10
    field_capacity_cv: float = 0.15
    wilting_point_cv: float = 0.20
    ksat_cv: float = 0.50  # High uncertainty

    # Van Genuchten parameters
    vg_alpha_mean: float = 0.04
    vg_alpha_cv: float = 0.30
    vg_n_mean: float = 1.5
    vg_n_cv: float = 0.15

    def sample(self, n_samples: int = 1, seed: Optional[int] = None) -> List[SoilParameters]:
        """
        Generate random samples from parameter distributions.

        Args:
            n_samples: Number of parameter sets to generate
            seed: Random seed for reproducibility

        Returns:
            List of SoilParameters samples
        """
        rng = np.random.default_rng(seed)
        samples = []

        for _ in range(n_samples):
            # Sample each parameter
            # Use truncated normal for bounded parameters

            porosity = self._sample_truncated_normal(
                rng, self.porosity_mean, self.porosity_mean * self.porosity_cv,
                0.25, 0.65
            )

            fc = self._sample_truncated_normal(
                rng, self.field_capacity_mean,
                self.field_capacity_mean * self.field_capacity_cv,
                0.05, porosity - 0.02
            )

            wp = self._sample_truncated_normal(
                rng, self.wilting_point_mean,
                self.wilting_point_mean * self.wilting_point_cv,
                0.01, fc - 0.02
            )

            # Log-normal for Ksat
            ksat = self._sample_lognormal(
                rng, self.ksat_mean_cm_day, self.ksat_cv
            )

            # VG parameters
            alpha = self._sample_lognormal(
                rng, self.vg_alpha_mean, self.vg_alpha_cv
            )
            n = self._sample_truncated_normal(
                rng, self.vg_n_mean, self.vg_n_mean * self.vg_n_cv,
                1.05, 3.0
            )

            samples.append(SoilParameters(
                sand_percent=0,  # Not sampled
                silt_percent=0,
                clay_percent=0,
                porosity=porosity,
                field_capacity=fc,
                wilting_point=wp,
                saturated_hydraulic_conductivity_cm_day=ksat,
                van_genuchten_alpha=alpha,
                van_genuchten_n=n,
                bulk_density_g_cm3=1.4  # Not sampled
            ))

        return samples

    def _sample_truncated_normal(
        self, rng, mean: float, std: float, lower: float, upper: float
    ) -> float:
        """Sample from truncated normal distribution."""
        for _ in range(100):  # Max attempts
            sample = rng.normal(mean, std)
            if lower <= sample <= upper:
                return sample
        return np.clip(mean, lower, upper)

    def _sample_lognormal(
        self, rng, mean: float, cv: float
    ) -> float:
        """Sample from log-normal distribution given mean and CV."""
        # Convert mean and cv to log-normal parameters
        sigma_sq = np.log(1 + cv**2)
        mu = np.log(mean) - sigma_sq / 2
        sigma = np.sqrt(sigma_sq)
        return rng.lognormal(mu, sigma)

    def get_percentiles(
        self, percentiles: List[float] = [5, 25, 50, 75, 95]
    ) -> Dict[str, Dict[float, float]]:
        """
        Get parameter values at specified percentiles.

        Args:
            percentiles: List of percentile values (0-100)

        Returns:
            Dictionary mapping parameter names to percentile values
        """
        # Generate many samples
        samples = self.sample(n_samples=1000, seed=42)

        result = {}
        param_names = ['porosity', 'field_capacity', 'wilting_point',
                       'saturated_hydraulic_conductivity_cm_day',
                       'van_genuchten_alpha', 'van_genuchten_n']

        for param in param_names:
            values = [getattr(s, param) for s in samples]
            result[param] = {
                p: np.percentile(values, p) for p in percentiles
            }

        return result


def create_parameter_distribution(
    soil_params: SoilParameters,
    uncertainty_level: str = "moderate"
) -> SoilParameterDistribution:
    """
    Create a parameter distribution from point estimates.

    Args:
        soil_params: Point estimates from PTF
        uncertainty_level: 'low', 'moderate', or 'high'

    Returns:
        SoilParameterDistribution for Monte Carlo analysis
    """
    # CV multipliers by uncertainty level
    cv_mult = {'low': 0.5, 'moderate': 1.0, 'high': 1.5}[uncertainty_level]

    return SoilParameterDistribution(
        porosity_mean=soil_params.porosity,
        field_capacity_mean=soil_params.field_capacity,
        wilting_point_mean=soil_params.wilting_point,
        ksat_mean_cm_day=soil_params.saturated_hydraulic_conductivity_cm_day,
        vg_alpha_mean=soil_params.van_genuchten_alpha or 0.04,
        vg_n_mean=soil_params.van_genuchten_n or 1.5,
        porosity_cv=0.10 * cv_mult,
        field_capacity_cv=0.15 * cv_mult,
        wilting_point_cv=0.20 * cv_mult,
        ksat_cv=0.50 * cv_mult,
        vg_alpha_cv=0.30 * cv_mult,
        vg_n_cv=0.15 * cv_mult,
    )


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
    TextureClass("loamy_sand", (70, 90), (0, 15),
                 (0, 30), "Loamy Sand", "Loamy Sand"),
    TextureClass("sandy_loam", (43, 85), (0, 20),
                 (0, 50), "Sandy Loam", "Sandy Loam"),
    TextureClass("loam", (23, 52), (7, 27), (28, 50), "Loam", "Loam"),
    TextureClass("silt_loam", (0, 50), (0, 27),
                 (50, 88), "Silt Loam", "Silt Loam"),
    TextureClass("silt", (0, 20), (0, 12), (80, 100), "Silt", "Silt"),
    TextureClass("clay_loam", (20, 45), (27, 40),
                 (15, 53), "Clay Loam", "Clay Loam"),
    TextureClass("silty_clay_loam", (0, 20), (27, 40), (40, 73),
                 "Silty Clay Loam", "Silty Clay Loam"),
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


def is_tropical_location(lat: float, lon: float) -> bool:
    """
    Determine if a location is in the tropical zone where oxide soils dominate.

    This is a simple heuristic based on:
    - Latitude between ~23.5°N and ~23.5°S (tropics)
    - Refinements for Africa where oxide soils (Ferralsols, Acrisols) are common

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees

    Returns:
        True if location is likely to have tropical oxide soil characteristics
    """
    # Basic tropical zone check
    if abs(lat) > 25:
        return False

    # Africa-specific refinements (where our TAHMO data is)
    # Most of Sub-Saharan Africa has weathered oxide soils
    if -20 <= lon <= 55:  # Africa longitude range
        if -35 <= lat <= 25:  # Africa latitude range
            return True

    # Other tropical regions (Southeast Asia, South America)
    # These also have extensive oxide soils
    if -15 <= lat <= 15:
        return True

    return False


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
    # Saxton & Rawls (2006) uses texture in percent and OM in percent.
    # Do NOT convert OM to a fraction here, otherwise OM effects become ~100× too small.
    om = float(organic_matter_percent)

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
        # Tropical soil corrections based on TAHMO validation feedback
        # Two separate phenomena need different corrections:
        #
        # 1. OXIDE CLAY EFFECT (clay > 30%):
        #    Ferralsols/Oxisols have micro-aggregated clays that drain like sand
        #    Only apply to HIGH CLAY soils - low clay sites don't need this
        #
        # 2. SANDY COASTAL EFFECT (sand > 50% AND clay > 18%):
        #    Sandy soils with moderate clay in hot coastal climates drain very fast
        #    Pure sandy soils (low clay) are already predicted correctly by Saxton-Rawls

        # OXIDE CLAY CORRECTION - only for clay > 30%
        if clay_percent > 30:
            # Scale effect from 30% to 45% clay
            clay_effect = min((clay_percent - 30) / 15.0, 1.0)

            # Increase Ksat due to oxide aggregation
            ksat_multiplier = 1.5 + 2.0 * clay_effect
            k_sat_cm_day *= ksat_multiplier

            # Reduce FC - oxide clays hold less water than temperate equivalents
            fc_multiplier = 0.75 - 0.15 * \
                clay_effect  # 0.75 at 30% clay, 0.60 at 45%
            field_capacity *= fc_multiplier

            # Reduce WP proportionally
            wp_multiplier = 0.85 - 0.10 * clay_effect
            wilting_point *= wp_multiplier

            # Slight porosity reduction
            porosity *= 0.95

        # SANDY COASTAL CORRECTION - sand > 50% AND clay > 18%
        # This targets sites like Likoni/BaseTitanium (sandy but with enough clay
        # to have been incorrectly estimated). Pure sandy soils (clay < 18%)
        # like Nyankpala/Walembelle are already correct with standard PTF.
        if sand_percent > 50 and clay_percent > 18:
            # Scale effect from 50% to 70% sand
            sand_effect = min((sand_percent - 50) / 20.0, 1.0)

            # Sandy soils in tropics drain faster
            k_sat_cm_day *= (1.3 + 0.7 * sand_effect)

            # Clamp FC for sandy tropical soils - they hold very little water
            # Likoni/BaseTitanium reality: FC ~0.05-0.08
            max_fc_sandy = 0.10 - 0.02 * sand_effect  # 0.10 at 50% sand, 0.08 at 70%
            if field_capacity > max_fc_sandy:
                field_capacity = max_fc_sandy

            # WP also very low for sandy soils
            max_wp_sandy = 0.05 - 0.01 * sand_effect
            if wilting_point > max_wp_sandy:
                wilting_point = max_wp_sandy

        # Ensure physical constraint: FC must be above WP
        if field_capacity <= wilting_point * 1.15:
            field_capacity = wilting_point * 1.20

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
        organic_matter_percent=organic_matter_percent
    )


def estimate_soil_parameters_tropical(
    sand_percent: float,
    clay_percent: float,
    organic_matter_percent: float = 4.0,  # Higher default for tropical
    bulk_density_g_cm3: Optional[float] = None,
    tropical_corrections: Optional[TropicalSoilCorrections] = None,
    structure_factor: float = 1.5,
    return_distribution: bool = False
) -> Union[SoilParameters, Tuple[SoilParameters, SoilParameterDistribution]]:
    """
    Estimate soil hydraulic parameters for tropical soils with corrections.

    This function addresses Gap 6 by:
    1. Adding organic matter correction (FC increases ~0.01 per 1% OM)
    2. Applying structure factor for Ksat (1.0-3.0)
    3. Accounting for oxide clay aggregation
    4. Optionally returning parameter distributions

    References:
    - Hodnett & Tomasella (2002) Marked differences between PTFs for tropical soils
    - Minasny & Hartemink (2011) Predicting soil properties in the tropics

    Args:
        sand_percent: Sand content (%)
        clay_percent: Clay content (%)
        organic_matter_percent: Organic matter content (%), default 4% for tropics
        bulk_density_g_cm3: Bulk density (g/cm³), optional
        tropical_corrections: TropicalSoilCorrections object
        structure_factor: Ksat multiplier for soil structure (1.0-3.0)
        return_distribution: If True, also return parameter distribution

    Returns:
        SoilParameters, or (SoilParameters, SoilParameterDistribution) if return_distribution
    """
    # Use default tropical corrections if not provided
    if tropical_corrections is None:
        tropical_corrections = TropicalSoilCorrections.for_african_soil(
            'default')

    # Start with standard Saxton-Rawls estimate using baseline OM
    baseline_om = tropical_corrections.baseline_om_percent

    # Get base parameters (without tropical correction flag to avoid double correction)
    base_params = estimate_soil_parameters_saxton(
        sand_percent=sand_percent,
        clay_percent=clay_percent,
        organic_matter_percent=baseline_om,
        bulk_density_g_cm3=bulk_density_g_cm3,
        is_tropical_oxide_soil=False  # We apply our own corrections
    )

    # ===== ORGANIC MATTER CORRECTION =====
    # Tropical soils often have 2-3× higher OM which increases water retention
    om_excess = organic_matter_percent - baseline_om

    # Increase FC by fc_per_percent_om for each 1% OM above baseline
    fc_correction = tropical_corrections.fc_per_percent_om * om_excess
    field_capacity = base_params.field_capacity + fc_correction

    # WP also increases but less than FC (more PAW with higher OM)
    wp_correction = 0.5 * tropical_corrections.fc_per_percent_om * om_excess
    wilting_point = base_params.wilting_point + wp_correction

    # Porosity increases slightly with OM
    porosity_correction = 0.005 * om_excess
    porosity = base_params.porosity + porosity_correction

    # ===== HIGH-CLAY OXIDE SOIL CORRECTION (FERRALSOLS/NITISOLS) =====
    # African high-clay soils are dominated by oxide/kaolinite clays that:
    # - Form stable micro-aggregates that drain like sand
    # - Have much lower water retention than temperate 2:1 clays
    # - Can dry to very low moisture levels (obs data shows <0.05 m³/m³)
    #
    # Evidence from ISMN validation:
    # - Kitabi_College (40% clay): obs_mean=0.028, standard PTF gives WP=0.26
    # - ES_Mibilizi_A (57% clay): obs_mean=0.042, standard PTF gives WP=0.34
    # - These soils actually behave like sandy loams for water retention
    #
    # The correction needs to be very aggressive because:
    # - Model theta_min = max(theta_r, 0.5*WP)
    # - To reach obs=0.028, we need 0.5*WP < 0.028 → WP < 0.056
    if clay_percent > 30:
        # Progressive reduction from 30% to 60% clay
        clay_effect = min((clay_percent - 30) / 30.0, 1.0)

        # Reduce WP very aggressively - oxide clays hold almost no water
        # Target: WP ≈ 0.03-0.06 for high-clay tropical soils
        # At 60% clay, reduce WP to ~10% of original (effectively like sand)
        wp_reduction = 0.90 * clay_effect  # 0% reduction at 30% clay, 90% at 60%
        wilting_point *= (1.0 - wp_reduction)

        # Reduce FC proportionally
        # At 60% clay, reduce FC to ~35% of original
        fc_reduction = 0.65 * clay_effect
        field_capacity *= (1.0 - fc_reduction)

        # Increase Ksat - well-aggregated clays drain very fast
        ksat_multiplier = 1.0 + 5.0 * clay_effect  # Up to 6× at 60% clay
        k_sat = base_params.saturated_hydraulic_conductivity_cm_day * ksat_multiplier
    else:
        k_sat = base_params.saturated_hydraulic_conductivity_cm_day

    # ===== STRUCTURE FACTOR FOR KSAT =====
    # Well-aggregated tropical soils have higher Ksat
    # Termite/ant activity creates macropores
    effective_structure = structure_factor * tropical_corrections.macropore_factor
    k_sat = k_sat * effective_structure

    # ===== OXIDE CLAY AGGREGATION =====
    # Oxide clays in Ferralsols behave more like sand
    # Reduce effective clay for VG parameters
    if tropical_corrections.oxide_aggregation < 1.0:
        effective_clay = clay_percent * tropical_corrections.oxide_aggregation

        # Recalculate VG parameters with effective clay
        # Higher alpha (faster drainage) with oxide aggregation
        log_alpha = (
            -0.784 +
            0.0178 * sand_percent -
            0.0112 * effective_clay -
            0.0074 * organic_matter_percent
        )
        alpha_kpa = np.exp(log_alpha) * 10.2

        # Slightly higher n (narrower pore size distribution)
        log_n = (
            -0.63 -
            0.0063 * sand_percent +
            0.0121 * effective_clay -
            0.0064 * organic_matter_percent
        )
        n = 1 + np.exp(log_n)
    else:
        alpha_kpa = base_params.van_genuchten_alpha
        n = base_params.van_genuchten_n

    # ===== APPLY PHYSICAL BOUNDS =====
    porosity = np.clip(porosity, 0.30, 0.65)
    # Lower minimum for oxide clays
    field_capacity = np.clip(field_capacity, 0.05, 0.55)
    # Lower minimum for oxide clays
    wilting_point = np.clip(wilting_point, 0.01, 0.40)
    # Higher upper bound for structured tropical
    k_sat = np.clip(k_sat, 0.5, 2000.0)
    alpha_kpa = np.clip(alpha_kpa, 0.001, 0.5)
    n = np.clip(n, 1.05, 2.8)

    # Ensure FC > WP with adequate PAW
    if field_capacity <= wilting_point * 1.2:
        field_capacity = wilting_point * 1.3

    # Bulk density typically lower in high-OM tropical soils
    if bulk_density_g_cm3 is None:
        bulk_density_g_cm3 = 1.4 - 0.03 * (organic_matter_percent - 2)
        bulk_density_g_cm3 = np.clip(bulk_density_g_cm3, 1.0, 1.6)

    soil_params = SoilParameters(
        sand_percent=sand_percent,
        silt_percent=100 - sand_percent - clay_percent,
        clay_percent=clay_percent,
        porosity=porosity,
        field_capacity=field_capacity,
        wilting_point=wilting_point,
        saturated_hydraulic_conductivity_cm_day=k_sat,
        van_genuchten_alpha=alpha_kpa,
        van_genuchten_n=n,
        bulk_density_g_cm3=bulk_density_g_cm3,
        organic_matter_percent=organic_matter_percent
    )

    if return_distribution:
        # Create distribution with higher uncertainty for tropical
        dist = create_parameter_distribution(
            soil_params, uncertainty_level='high')
        return soil_params, dist

    return soil_params


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
        warnings[
            'porosity'] = f"Porosity {params.porosity:.3f} outside typical range (0.3-0.6)"

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
                params = estimate_soil_parameters_saxton(
                    sand_percent, clay_percent)

            elif method == "rosetta":
                params = estimate_soil_parameters_rosetta(
                    sand_percent, clay_percent)
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
        # ±50% (high uncertainty)
        "saturated_hydraulic_conductivity_cm_day": 0.50,
        "van_genuchten_alpha": 0.30,
        "van_genuchten_n": 0.15,
    }

    # Z-score for confidence level
    z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(
        confidence_level, 1.96)

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
        organic_matter_percent=2.0
    )
