"""
Adaptive Physics Model Calibration Module.

This module addresses critical issues in physics-based soil moisture modeling:
1. Site-specific bias correction using climatological constraints
2. Adaptive infiltration parameters based on rainfall intensity and soil state
3. Climate-aware ET partitioning for tropical environments
4. Improved pedotransfer functions for African soils
5. Dynamic root water uptake calibration

The key insight is that generic physics parameters often fail across diverse sites
because:
- Pedotransfer functions were developed on temperate soils
- ET partitioning doesn't account for local vegetation dynamics
- Infiltration parameters don't capture macropore flow in tropical soils
- Root distributions vary significantly with land cover type

This module provides adaptive corrections that can be applied without full
parameter optimization, using physical constraints and site characteristics.

References:
- Hodnett & Tomasella (2002): Tropical soil PTFs
- Minasny & Hartemink (2011): Tropical soil properties
- Seneviratne et al. (2010): Soil moisture-climate interactions
- Andreasen et al. (2013): Bias correction for soil moisture
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ClimateZone(Enum):
    """Köppen-Geiger climate zones relevant for soil moisture modeling."""
    TROPICAL_WET = "Af"              # Tropical rainforest
    TROPICAL_MONSOON = "Am"          # Tropical monsoon
    TROPICAL_SAVANNA = "Aw"          # Tropical savanna (wet-dry)
    SEMI_ARID_HOT = "BSh"            # Hot semi-arid (Sahel)
    ARID_HOT = "BWh"                 # Hot desert
    HUMID_SUBTROPICAL = "Cfa"        # Humid subtropical
    MEDITERRANEAN = "Cs"             # Mediterranean
    TEMPERATE = "Cfb"                # Oceanic/temperate


class LandCoverType(Enum):
    """Land cover types affecting ET and root distribution."""
    CROPLAND = "cropland"            # Agricultural crops
    GRASSLAND = "grassland"          # Natural grassland
    SAVANNA = "savanna"              # Savanna woodland
    FOREST = "forest"                # Dense forest
    SHRUBLAND = "shrubland"          # Shrubs and bushes
    BARE_SOIL = "bare"               # Bare/sparse vegetation


@dataclass
class SiteCharacteristics:
    """
    Site-specific characteristics for adaptive calibration.

    These are derived from ancillary data (climate, land cover, soil surveys)
    and used to adjust physics parameters without full optimization.
    """
    latitude: float
    longitude: float
    elevation_m: float = 200.0
    slope_percent: float = 5.0  # Slope in percent for runoff calculations

    # Climate characteristics
    climate_zone: ClimateZone = ClimateZone.TROPICAL_SAVANNA
    mean_annual_precip_mm: float = 1000.0
    mean_annual_temp_c: float = 27.0
    precip_seasonality: float = 0.7  # 0=uniform, 1=highly seasonal
    dry_season_length_months: int = 4

    # Soil characteristics (can be from ISDA or local surveys)
    sand_percent: float = 40.0
    clay_percent: float = 25.0
    organic_matter_percent: float = 2.0
    soil_depth_m: float = 1.5

    # Land cover
    land_cover: LandCoverType = LandCoverType.SAVANNA
    vegetation_fraction: float = 0.6  # Mean annual vegetation cover

    # Historical soil moisture statistics (if available)
    sm_mean_obs: Optional[float] = None  # Mean observed soil moisture
    sm_std_obs: Optional[float] = None   # Std of observed soil moisture
    sm_min_obs: Optional[float] = None   # Minimum observed
    sm_max_obs: Optional[float] = None   # Maximum observed

    @classmethod
    def estimate_from_location(
        cls,
        latitude: float,
        longitude: float,
        sand_percent: float = 40.0,
        clay_percent: float = 25.0,
        elevation_m: float = 200.0,
        ndvi_mean: Optional[float] = None,
        annual_precip_mm: Optional[float] = None,
    ) -> "SiteCharacteristics":
        """
        Estimate site characteristics from location and basic inputs.

        Uses geographic heuristics for African sites when detailed data
        is not available.
        """
        # Estimate climate zone from latitude, elevation, and region (improved for Africa)
        # East Africa has complex topography - higher elevations can be much drier

        # Base classification by latitude
        if abs(latitude) < 5:
            base_climate = ClimateZone.TROPICAL_WET
            base_map = 2000.0
            base_seasonality = 0.3
            base_dry_months = 1
        elif abs(latitude) < 10:
            base_climate = ClimateZone.TROPICAL_SAVANNA
            base_map = 1200.0
            base_seasonality = 0.6
            base_dry_months = 4
        elif abs(latitude) < 15:
            base_climate = ClimateZone.SEMI_ARID_HOT
            base_map = 800.0
            base_seasonality = 0.8
            base_dry_months = 6
        else:
            base_climate = ClimateZone.ARID_HOT
            base_map = 400.0
            base_seasonality = 0.9
            base_dry_months = 8

        # Regional adjustments for East Africa (longitude 25°E to 45°E)
        # East Africa has Rift Valley, highlands, and coastal variations
        if 25 <= longitude <= 45:  # East Africa
            # High elevation correction (>1500m = cooler/drier)
            if elevation_m > 1500:
                if base_climate == ClimateZone.TROPICAL_WET:
                    climate_zone = ClimateZone.TROPICAL_SAVANNA
                    map_est = base_map * 0.7  # Highlands are drier
                    seasonality = base_seasonality + 0.2
                    dry_months = base_dry_months + 2
                else:
                    climate_zone = base_climate
                    map_est = base_map * 0.8
                    seasonality = base_seasonality + 0.1
                    dry_months = base_dry_months + 1
            # Coastal areas (near Indian Ocean)
            elif latitude < 5 and longitude > 35:
                climate_zone = ClimateZone.TROPICAL_WET  # Coastal wet
                map_est = base_map * 1.2
                seasonality = base_seasonality * 0.8
                dry_months = max(0, base_dry_months - 1)
            # Northern East Africa (more arid)
            elif latitude > 0:
                climate_zone = ClimateZone.TROPICAL_SAVANNA
                map_est = base_map * 0.9
                seasonality = base_seasonality + 0.1
                dry_months = base_dry_months + 1
            else:
                climate_zone = base_climate
                map_est = base_map
                seasonality = base_seasonality
                dry_months = base_dry_months
        else:
            # West Africa and other regions - use base classification
            climate_zone = base_climate
            map_est = base_map
            seasonality = base_seasonality
            dry_months = base_dry_months

        # Use provided precipitation if available
        if annual_precip_mm is not None:
            map_est = annual_precip_mm

        # Estimate land cover from NDVI
        if ndvi_mean is not None:
            if ndvi_mean > 0.6:
                land_cover = LandCoverType.FOREST
                veg_frac = 0.85
            elif ndvi_mean > 0.4:
                land_cover = LandCoverType.SAVANNA
                veg_frac = 0.6
            elif ndvi_mean > 0.25:
                land_cover = LandCoverType.GRASSLAND
                veg_frac = 0.4
            elif ndvi_mean > 0.15:
                land_cover = LandCoverType.SHRUBLAND
                veg_frac = 0.25
            else:
                land_cover = LandCoverType.BARE_SOIL
                veg_frac = 0.1
        else:
            land_cover = LandCoverType.SAVANNA
            veg_frac = 0.5

        # Estimate mean temperature from latitude and elevation
        # Base temp ~30°C at equator, -0.65°C per 100m elevation
        mat = 30.0 - abs(latitude) * 0.3 - elevation_m * 0.0065
        mat = np.clip(mat, 15.0, 35.0)

        return cls(
            latitude=latitude,
            longitude=longitude,
            elevation_m=elevation_m,
            slope_percent=5.0,  # Default slope - would be estimated from DEM
            climate_zone=climate_zone,
            mean_annual_precip_mm=map_est,
            mean_annual_temp_c=mat,
            precip_seasonality=seasonality,
            dry_season_length_months=dry_months,
            sand_percent=sand_percent,
            clay_percent=clay_percent,
            organic_matter_percent=2.0,
            land_cover=land_cover,
            vegetation_fraction=veg_frac,
        )


@dataclass
class AdaptiveCalibrationParameters:
    """
    Adaptive calibration parameters derived from site characteristics.

    These multipliers/corrections are applied to the base physics model
    to account for site-specific conditions without full parameter optimization.
    """
    # Infiltration adjustments
    # K_sat adjustment (macropores, structure)
    ksat_multiplier: float = 1.0
    # Fraction of rain that infiltrates (vs interception)
    infiltration_efficiency: float = 0.85
    macropore_fraction: float = 0.15     # Enhanced macropore flow fraction
    runoff_curve_number: float = 70.0    # SCS curve number for excess runoff

    # ET partitioning adjustments
    kcb_multiplier: float = 1.0          # Basal crop coefficient adjustment
    ke_multiplier: float = 1.0           # Soil evaporation coefficient adjustment
    et_stress_factor: float = 1.0        # Overall ET stress adjustment
    transpiration_fraction: float = 0.7   # T/(T+E) partitioning

    # Soil hydraulic adjustments
    theta_s_adjustment: float = 0.0      # Porosity adjustment (additive)
    theta_r_adjustment: float = 0.0      # Residual WC adjustment (additive)
    alpha_multiplier: float = 1.0        # VG alpha adjustment
    n_adjustment: float = 0.0            # VG n adjustment (additive)

    # Root water uptake adjustments
    root_depth_multiplier: float = 1.0   # Maximum root depth adjustment
    # Root exponential decay (lower = deeper roots)
    root_distribution_beta: float = 3.5
    compensation_factor: float = 1.0     # Root compensation strength

    # Drainage adjustments
    drainage_coefficient: float = 1.0    # Deep drainage multiplier
    capillary_rise_max_mm: float = 3.0   # Maximum capillary rise (mm/day)

    # Bias correction (applied as post-processing)
    bias_correction_additive: float = 0.0  # Added to predictions
    bias_correction_multiplicative: float = 1.0  # Multiplied to predictions

    # Dynamic constraints
    theta_min_physical: float = 0.02     # Physical minimum (air-dry)
    theta_max_physical: float = 0.55     # Physical maximum (saturation)


class AdaptivePhysicsCalibrator:
    """
    Adaptive calibration system for physics-based soil moisture models.

    This class derives physics parameter adjustments from site characteristics
    and observed data statistics, without requiring full optimization.

    Key principles:
    1. Use physical constraints to bound predictions
    2. Apply climate-aware corrections to ET and infiltration
    3. Correct for tropical soil biases in pedotransfer functions
    4. Enable dynamic parameter adjustment based on soil moisture state
    """

    def __init__(
        self,
        site_chars: SiteCharacteristics,
        observed_stats: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize adaptive calibrator.

        Args:
            site_chars: Site characteristics
            observed_stats: Dictionary with 'mean', 'std', 'min', 'max' of observed SM
        """
        self.site = site_chars
        self.obs_stats = observed_stats or {}
        self.params = self._derive_parameters()

        logger.info(
            f"Initialized AdaptivePhysicsCalibrator for "
            f"lat={site_chars.latitude:.2f}, lon={site_chars.longitude:.2f}, "
            f"climate={site_chars.climate_zone.name}, "
            f"land_cover={site_chars.land_cover.name}"
        )

    def _derive_parameters(self) -> AdaptiveCalibrationParameters:
        """Derive adaptive parameters from site characteristics."""
        params = AdaptiveCalibrationParameters()
        site = self.site

        # ================================================================
        # 1. INFILTRATION ADJUSTMENTS
        # ================================================================
        # Tropical soils often have enhanced macroporosity due to:
        # - Termite/ant activity
        # - Root channels
        # - Aggregate structure from oxide clays

        # Ksat multiplier based on climate and soil
        if site.climate_zone in [ClimateZone.TROPICAL_WET, ClimateZone.TROPICAL_MONSOON]:
            # High biological activity increases Ksat
            params.ksat_multiplier = 1.8
            params.macropore_fraction = 0.25
        elif site.climate_zone == ClimateZone.TROPICAL_SAVANNA:
            # Moderate enhancement
            params.ksat_multiplier = 1.5
            params.macropore_fraction = 0.20
        elif site.climate_zone == ClimateZone.SEMI_ARID_HOT:
            # Surface sealing can reduce infiltration
            params.ksat_multiplier = 0.9
            params.macropore_fraction = 0.10
        else:
            params.ksat_multiplier = 1.0
            params.macropore_fraction = 0.10

        # Sandy soils have higher Ksat but less macropore effect
        if site.sand_percent > 70:
            params.ksat_multiplier *= 1.3
            params.macropore_fraction *= 0.7
        elif site.clay_percent > 40:
            params.ksat_multiplier *= 0.7
            params.macropore_fraction *= 1.3

        # Infiltration efficiency depends on vegetation cover
        params.infiltration_efficiency = 0.70 + 0.25 * site.vegetation_fraction

        # Slope affects runoff potential (SCS curve number)
        # Steeper slopes increase runoff due to faster overland flow
        if hasattr(site, 'slope_percent') and site.slope_percent is not None:
            slope = site.slope_percent
            # SCS adjustment: CN increases with slope
            # For slopes > 5%, CN increases by ~5-10 points
            if slope > 10:
                params.runoff_curve_number = min(
                    95, params.runoff_curve_number + 10)
            elif slope > 5:
                params.runoff_curve_number = min(
                    90, params.runoff_curve_number + 5)
            elif slope < 2:
                # Flat areas have lower runoff
                params.runoff_curve_number = max(
                    60, params.runoff_curve_number - 5)

        # ================================================================
        # 2. ET PARTITIONING ADJUSTMENTS
        # ================================================================
        # Critical for bias control - most physics model biases come from ET

        # Kcb adjustment based on land cover and climate
        land_cover_kcb = {
            LandCoverType.FOREST: 1.05,
            LandCoverType.SAVANNA: 0.85,
            LandCoverType.GRASSLAND: 0.80,
            LandCoverType.CROPLAND: 1.00,
            LandCoverType.SHRUBLAND: 0.75,
            LandCoverType.BARE_SOIL: 0.40,
        }
        params.kcb_multiplier = land_cover_kcb.get(site.land_cover, 0.90)

        # Climate adjustment for Kcb (FAO-56 Eq. 70 inspired)
        # Higher temperatures and lower humidity increase ET demand
        # But tropical plants are often adapted and have stomatal regulation
        if site.mean_annual_temp_c > 30:
            # Hot conditions - plants limit transpiration
            params.kcb_multiplier *= 0.90

        # Seasonal adjustment
        if site.precip_seasonality > 0.7:
            # Highly seasonal - reduce ET in dry season
            params.et_stress_factor = 0.85

        # Soil evaporation adjustment
        # Sandy soils dry faster (higher Ke initially but drops quickly)
        if site.sand_percent > 60:
            params.ke_multiplier = 1.1
        elif site.clay_percent > 40:
            params.ke_multiplier = 0.85  # Clay holds moisture, slower evap

        # T/(T+E) partitioning based on vegetation
        params.transpiration_fraction = 0.3 + 0.5 * site.vegetation_fraction

        # ================================================================
        # 3. SOIL HYDRAULIC ADJUSTMENTS
        # ================================================================
        # Tropical soils often have different properties than PTF predictions

        # Porosity adjustment for tropical soils
        # Hodnett & Tomasella (2002) found tropical soils have ~5% higher porosity
        if site.climate_zone in [
            ClimateZone.TROPICAL_WET,
            ClimateZone.TROPICAL_MONSOON,
            ClimateZone.TROPICAL_SAVANNA
        ]:
            params.theta_s_adjustment = 0.03  # +3% porosity

        # Residual water content adjustment
        # Tropical clay-rich soils have higher residual due to oxide aggregation
        if site.clay_percent > 35:
            params.theta_r_adjustment = 0.02

        # VG alpha adjustment (affects air entry and drainage)
        # Tropical soils often have better structure = larger alpha
        if site.clay_percent > 40 and site.climate_zone in [
            ClimateZone.TROPICAL_WET, ClimateZone.TROPICAL_SAVANNA
        ]:
            # Well-aggregated tropical clays act more like loams
            params.alpha_multiplier = 1.5
            params.n_adjustment = 0.1

        # ================================================================
        # 3b. EAST AFRICAN HIGHLAND SOIL CORRECTIONS
        # ================================================================
        # East African highland soils (Nitisols, Ferralsols, Andosols) have
        # very different hydraulic properties than standard PTFs predict:
        # - Oxide clays form stable micro-aggregates
        # - Soils drain like sandy loams despite high clay content
        # - Much lower water retention than temperate clays
        # - Can dry to θ < 0.05 m³/m³ even with 40%+ clay
        #
        # Evidence from ISMN validation shows physics model overestimates
        # water content by 0.02-0.12 m³/m³ for East African sites.
        is_east_africa = 25 <= site.longitude <= 45
        is_highland = site.elevation_m > 1000 if hasattr(
            site, 'elevation_m') else site.mean_annual_temp_c < 25
        has_oxide_clay = site.clay_percent > 25 and is_east_africa

        if is_east_africa and has_oxide_clay:
            logger.info(
                f"Applying East African oxide clay corrections: "
                f"clay={site.clay_percent}%, highland={is_highland}"
            )

            # Progressive correction based on clay content
            # More clay = MORE correction needed (opposite of temperate soils!)
            # 0 at 25%, 1 at 50%+
            clay_factor = min(1.0, (site.clay_percent - 25) / 25.0)

            # 1. DRASTICALLY reduce water retention for oxide clays
            # Standard PTFs give WP=0.26 for 40% clay
            # Oxide clays: WP should be ~0.05-0.10
            # This manifests as negative theta_s adjustment (reduces saturation)
            params.theta_s_adjustment -= 0.05 * clay_factor  # Reduce porosity effect

            # 2. INCREASE drainage dramatically
            # Oxide clays have stable aggregates = fast drainage like sand
            drainage_boost = 2.0 + 3.0 * clay_factor  # 2× to 5× drainage
            params.ksat_multiplier *= drainage_boost
            params.drainage_coefficient *= (1.5 + clay_factor)
            params.macropore_fraction += 0.10 * clay_factor

            # 3. INCREASE alpha (faster unsaturated drainage)
            # Oxide aggregates = larger pores = higher alpha
            params.alpha_multiplier *= (1.5 + clay_factor)

            # 4. Highland-specific: INCREASE ET
            # East African highlands have high solar radiation and wind
            if is_highland:
                params.kcb_multiplier *= 1.15  # More transpiration
                params.ke_multiplier *= 1.20   # More evaporation
                params.et_stress_factor *= 1.10

            # 5. Adjust physical bounds for oxide clays
            # These soils can get MUCH drier than temperate clays
            params.theta_min_physical = max(
                0.02, 0.02 + 0.001 * site.clay_percent)
            params.theta_max_physical = min(
                0.50, 0.40 + 0.002 * (100 - site.sand_percent))

            logger.info(
                f"  → ksat_mult={params.ksat_multiplier:.2f}, "
                f"drainage={params.drainage_coefficient:.2f}, "
                f"alpha_mult={params.alpha_multiplier:.2f}"
            )

        # ================================================================
        # 4. ROOT WATER UPTAKE ADJUSTMENTS
        # ================================================================
        # Root distribution varies significantly with vegetation type

        root_params = {
            # Deep roots, uniform dist
            LandCoverType.FOREST: (1.5, 2.5, 1.2),
            # Medium depth, some compensation
            LandCoverType.SAVANNA: (1.3, 3.0, 1.1),
            LandCoverType.GRASSLAND: (0.8, 4.0, 0.9),   # Shallow, concentrated
            LandCoverType.CROPLAND: (1.0, 3.5, 1.0),    # Moderate
            LandCoverType.SHRUBLAND: (1.2, 3.0, 1.0),   # Medium
            LandCoverType.BARE_SOIL: (0.5, 5.0, 0.5),   # Minimal
        }
        depth_mult, beta, comp = root_params.get(
            site.land_cover, (1.0, 3.5, 1.0))

        params.root_depth_multiplier = depth_mult
        params.root_distribution_beta = beta
        params.compensation_factor = comp

        # Adjust for climate - deeper roots in seasonal climates
        if site.dry_season_length_months > 4:
            params.root_depth_multiplier *= 1.2
            params.root_distribution_beta *= 0.85  # More uniform with depth

        # ================================================================
        # 5. DRAINAGE ADJUSTMENTS
        # ================================================================
        # Bottom boundary conditions affect deep soil dynamics

        # Capillary rise depends on water table (estimated from climate)
        if site.mean_annual_precip_mm > 1500:
            # Likely shallow water table
            params.capillary_rise_max_mm = 5.0
        elif site.mean_annual_precip_mm < 600:
            # Deep water table
            params.capillary_rise_max_mm = 1.0
        else:
            params.capillary_rise_max_mm = 3.0

        # Drainage coefficient based on soil depth and texture
        if site.sand_percent > 60:
            params.drainage_coefficient = 1.3  # Fast drainage
        elif site.clay_percent > 40:
            params.drainage_coefficient = 0.6  # Slow drainage

        # ================================================================
        # 6. BIAS CORRECTION FROM OBSERVATIONS
        # ================================================================
        if self.obs_stats:
            self._apply_observation_based_corrections(params)

        # ================================================================
        # 7. PHYSICAL CONSTRAINTS
        # ================================================================
        # Set reasonable bounds based on soil texture
        params.theta_min_physical = 0.01 + 0.002 * site.clay_percent
        params.theta_max_physical = 0.35 + 0.004 * (100 - site.sand_percent)

        # Update self attributes for compatibility
        self.bias_correction_multiplicative = params.bias_correction_multiplicative
        self.bias_correction_additive = params.bias_correction_additive

        return params

    def _apply_observation_based_corrections(
        self,
        params: AdaptiveCalibrationParameters
    ) -> None:
        """
        Apply corrections based on observed soil moisture statistics.

        This implements a "soft calibration" that adjusts parameters to match
        observed climatology without full time-series optimization.

        Key insight: Large mean biases in tropical Africa often indicate:
        1. Soils drain faster than PTFs predict (higher Ksat needed)
        2. ET is higher than expected (vegetation stress adaptation)
        3. Lower field capacity (less water holding)

        We correct physics parameters rather than just shifting predictions.
        """
        obs = self.obs_stats

        if 'mean' in obs and obs['mean'] is not None:
            # Estimate expected mean from physics (simplified)
            # Using Budyko-type relationship
            site = self.site
            dryness_index = site.mean_annual_temp_c * \
                30 / max(100, site.mean_annual_precip_mm)
            expected_mean = 0.35 * np.exp(-0.3 * dryness_index)
            expected_mean = np.clip(expected_mean, 0.10, 0.45)

            # If observed mean differs significantly, apply correction
            obs_mean = obs['mean']
            diff = obs_mean - expected_mean

            if abs(diff) > 0.03:  # Threshold for correction
                logger.info(
                    f"Mean bias detected: obs={obs_mean:.3f}, expected={expected_mean:.3f}")

                # Calculate correction factor (multiplicative is better than additive)
                # This preserves dynamics while shifting the mean
                correction_ratio = obs_mean / expected_mean if expected_mean > 0.05 else 1.0

                # Allow stronger corrections for extreme biases (especially in East Africa)
                # East African sites can be much drier than expected due to topography
                if abs(site.longitude - 30) < 15:  # East Africa longitude range
                    # Allow much stronger corrections for very dry sites
                    if obs_mean < 0.05:  # Extremely dry sites
                        correction_ratio = np.clip(
                            correction_ratio, 0.01, 3.0)  # Allow down to 0.01x
                    else:
                        # Wider range for East Africa
                        correction_ratio = np.clip(correction_ratio, 0.2, 3.0)
                else:
                    correction_ratio = np.clip(
                        correction_ratio, 0.5, 2.0)  # Standard range

                if diff < 0:  # Observed drier than expected
                    # The physics model is predicting too wet
                    # Options: increase ET, increase Ksat, lower field capacity

                    dryness_factor = expected_mean / obs_mean if obs_mean > 0.05 else 2.0

                    # Allow stronger corrections for extreme dryness (East Africa issue)
                    if obs_mean < 0.08:  # Very dry sites
                        # Allow up to 5x correction
                        dryness_factor = np.clip(dryness_factor, 1.0, 5.0)
                    else:
                        dryness_factor = np.clip(dryness_factor, 1.0, 3.0)

                    # 1. Increase ET more aggressively for very dry sites
                    et_increase = 0.1 * (dryness_factor - 1)
                    if obs_mean < 0.08:  # Very dry sites need stronger ET correction
                        et_increase *= 2.0
                    params.kcb_multiplier *= min(1.0 + et_increase, 1.8)
                    params.ke_multiplier *= min(1.0 + et_increase, 1.8)

                    # 2. Increase drainage more aggressively
                    drainage_increase = dryness_factor
                    if obs_mean < 0.08:
                        drainage_increase *= 1.5  # Extra drainage for very dry sites
                    params.ksat_multiplier *= min(drainage_increase, 3.0)
                    params.drainage_coefficient *= min(
                        1.0 + 0.2 * (dryness_factor - 1), 2.0)

                    # 3. Lower water holding capacity more for very dry sites
                    capacity_reduction = -0.02 * (dryness_factor - 1)
                    if obs_mean < 0.08:
                        capacity_reduction *= 2.0  # Double reduction for very dry sites
                    params.theta_s_adjustment = capacity_reduction

                    # 4. Apply stronger multiplicative bias correction for extreme cases
                    if obs_mean < 0.03:  # Extremely dry sites (like Kitabi)
                        params.bias_correction_multiplicative = correction_ratio ** 1.0  # Direct correction
                    elif obs_mean < 0.05:  # Very dry sites
                        params.bias_correction_multiplicative = correction_ratio ** 1.5  # Stronger correction
                    elif obs_mean < 0.08:  # Dry sites
                        params.bias_correction_multiplicative = correction_ratio ** 1.0  # Moderate correction
                    else:
                        params.bias_correction_multiplicative = np.sqrt(
                            correction_ratio)

                else:  # Observed wetter than expected
                    # Model is predicting too dry
                    wetness_factor = obs_mean / expected_mean if expected_mean > 0.05 else 1.5
                    wetness_factor = np.clip(wetness_factor, 1.0, 2.0)

                    # 1. Reduce ET
                    params.kcb_multiplier *= max(1.0 -
                                                 0.1 * (wetness_factor - 1), 0.7)
                    params.ke_multiplier *= max(1.0 -
                                                0.1 * (wetness_factor - 1), 0.7)

                    # 2. Reduce drainage
                    params.ksat_multiplier *= max(1.0 / wetness_factor, 0.6)

                    # 3. Modest bias correction
                    params.bias_correction_multiplicative = np.sqrt(
                        correction_ratio)

                # Don't use large additive corrections - they distort dynamics
                params.bias_correction_additive = 0.0

        if 'std' in obs and obs['std'] is not None:
            # If observed variability differs, adjust dynamics
            obs_std = obs['std']

            # Expected std based on climate (rough estimate)
            expected_std = 0.04 + 0.02 * self.site.precip_seasonality

            if obs_std < expected_std * 0.7:
                # Low variability - soil has high water holding, slow dynamics
                params.drainage_coefficient *= 0.8
                params.compensation_factor *= 1.1
            elif obs_std > expected_std * 1.5:
                # High variability - fast response
                params.drainage_coefficient *= 1.2
                params.macropore_fraction *= 1.1

        if 'min' in obs and 'max' in obs:
            if obs['min'] is not None and obs['max'] is not None:
                # Use observed range to set physical bounds
                # But don't make them too tight - leave room for extremes
                params.theta_min_physical = max(0.02, obs['min'] - 0.03)
                params.theta_max_physical = min(0.55, obs['max'] + 0.03)

    def get_adjusted_vg_params(
        self,
        base_params: "VanGenuchtenParameters"  # type hint
    ) -> Dict[str, float]:
        """
        Get adjusted Van Genuchten parameters.

        Args:
            base_params: Base VG parameters from pedotransfer function

        Returns:
            Dictionary with adjusted parameters
        """
        p = self.params

        return {
            'alpha': base_params.alpha * p.alpha_multiplier,
            'n': base_params.n + p.n_adjustment,
            'theta_r': base_params.theta_r + p.theta_r_adjustment,
            'theta_s': min(0.60, base_params.theta_s + p.theta_s_adjustment),
            'K_sat': base_params.K_sat * p.ksat_multiplier,
        }

    def adjust_infiltration(
        self,
        potential_infiltration_mm: float,
        surface_theta: float,
        theta_s: float,
        rainfall_intensity_mm_hr: float = 5.0,
    ) -> Tuple[float, float]:
        """
        Adjust infiltration based on adaptive parameters.

        Returns:
            Tuple of (actual_infiltration_mm, runoff_mm)
        """
        p = self.params

        # Calculate saturation
        saturation = surface_theta / theta_s if theta_s > 0 else 1.0
        saturation = np.clip(saturation, 0, 1)

        # Base infiltration capacity
        # Reduces as saturation increases (Dunne-type runoff)
        infil_capacity_fraction = 1.0 - saturation ** 2

        # Macropore bypass when nearly saturated
        if saturation > 0.7:
            macropore_bypass = p.macropore_fraction * (saturation - 0.7) / 0.3
        else:
            macropore_bypass = 0.0

        # Effective infiltration efficiency
        eff_efficiency = p.infiltration_efficiency * (
            infil_capacity_fraction + macropore_bypass
        )
        eff_efficiency = np.clip(eff_efficiency, 0.1, 1.0)

        # Calculate actual infiltration
        actual_infiltration = potential_infiltration_mm * eff_efficiency
        runoff = potential_infiltration_mm - actual_infiltration

        return actual_infiltration, max(0, runoff)

    def adjust_et(
        self,
        et0_mm: float,
        kcb_base: float,
        ke_base: float,
        soil_moisture: float,
        ndvi: Optional[float] = None,
        day_of_year: int = 180,
    ) -> Tuple[float, float]:
        """
        Adjust ET components based on adaptive parameters.

        Returns:
            Tuple of (transpiration_mm, soil_evaporation_mm)
        """
        p = self.params

        # Adjust crop coefficients
        kcb_adj = kcb_base * p.kcb_multiplier
        ke_adj = ke_base * p.ke_multiplier

        # Seasonal adjustment for savanna/grassland
        # Vegetation senescence in dry season
        if self.site.land_cover in [LandCoverType.SAVANNA, LandCoverType.GRASSLAND]:
            # Simple phenology - reduced Kcb during dry months
            # Assuming dry season centered around day 350-60 (Dec-Feb for N hemisphere tropics)
            # or 170-230 (Jun-Aug for S hemisphere)
            if self.site.latitude > 0:
                dry_center = 30  # January
            else:
                dry_center = 200  # July

            # Distance from dry season center (circular)
            dist_from_dry = min(
                abs(day_of_year - dry_center),
                365 - abs(day_of_year - dry_center)
            )
            # Max 180 days from center
            seasonal_factor = 0.5 + 0.5 * min(1.0, dist_from_dry / 90)
            kcb_adj *= seasonal_factor

        # NDVI-based adjustment if available
        if ndvi is not None:
            # Scale Kcb with NDVI
            ndvi_factor = np.clip(ndvi / 0.6, 0.3, 1.0)
            kcb_adj *= ndvi_factor

        # Soil moisture stress function
        # More realistic s-shaped stress curve
        fc_est = 0.30  # Estimated field capacity
        wp_est = 0.10  # Estimated wilting point

        if soil_moisture < wp_est:
            stress = 0.05  # Minimal transpiration
        elif soil_moisture > fc_est:
            stress = 1.0
        else:
            # S-curve stress function
            rel_depletion = (fc_est - soil_moisture) / (fc_est - wp_est)
            stress = 1.0 - rel_depletion ** 2
            stress = np.clip(stress, 0.05, 1.0)

        # Calculate ET components
        transpiration = et0_mm * kcb_adj * stress * p.et_stress_factor

        # Soil evaporation - higher when vegetation is low
        veg_cover = 0.3 + 0.5 * (ndvi if ndvi else 0.4)
        exposed_soil = 1.0 - veg_cover
        soil_evaporation = et0_mm * ke_adj * exposed_soil * p.et_stress_factor

        # Apply partitioning constraint
        total_et = transpiration + soil_evaporation
        if total_et > 0:
            actual_t_frac = transpiration / total_et
            target_t_frac = p.transpiration_fraction * \
                (ndvi if ndvi else 0.4) / 0.4
            target_t_frac = np.clip(target_t_frac, 0.2, 0.8)

            # Blend toward target partitioning
            blend_factor = 0.5
            new_t_frac = actual_t_frac * \
                (1 - blend_factor) + target_t_frac * blend_factor

            transpiration = total_et * new_t_frac
            soil_evaporation = total_et * (1 - new_t_frac)

        return transpiration, soil_evaporation

    def apply_bias_correction(
        self,
        prediction: float,
    ) -> float:
        """
        Apply post-hoc bias correction to prediction.

        Args:
            prediction: Raw model prediction

        Returns:
            Bias-corrected prediction
        """
        p = self.params

        # Apply corrections
        corrected = prediction * p.bias_correction_multiplicative
        corrected += p.bias_correction_additive

        # Enforce physical bounds
        corrected = np.clip(corrected, p.theta_min_physical,
                            p.theta_max_physical)

        return corrected

    def constrain_prediction(
        self,
        prediction: float,
        previous_prediction: Optional[float] = None,
        precipitation_mm: float = 0.0,
        dt_days: float = 1.0,
    ) -> float:
        """
        Apply physical constraints to prediction.

        Constraints:
        1. Within physical bounds (theta_r to theta_s)
        2. Maximum drying rate (based on ET0 limit)
        3. Maximum wetting rate (based on Ksat)
        4. Temporal smoothness (no unrealistic jumps)

        Args:
            prediction: Model prediction
            previous_prediction: Previous timestep prediction
            precipitation_mm: Precipitation for this timestep
            dt_days: Timestep length

        Returns:
            Constrained prediction
        """
        p = self.params

        # Physical bounds
        constrained = np.clip(
            prediction, p.theta_min_physical, p.theta_max_physical)

        if previous_prediction is not None:
            # Maximum change constraints
            # Maximum drying: ~5mm/day ET / 100mm layer = 0.05/day
            max_drying_rate = 0.05 * dt_days

            # Maximum wetting: depends on precipitation
            # Assume ~100mm effective soil depth, Ksat ~ 50mm/day
            max_wetting_rate = min(precipitation_mm / 100.0, 0.15) * dt_days

            # If no precip, can still wet slightly from capillary rise
            if precipitation_mm < 1:
                max_wetting_rate = p.capillary_rise_max_mm / 100.0 * dt_days

            # Apply constraints
            delta = constrained - previous_prediction

            if delta < 0:  # Drying
                delta = max(delta, -max_drying_rate)
            else:  # Wetting
                delta = min(delta, max_wetting_rate)

            constrained = previous_prediction + delta

        # Final bounds check
        constrained = np.clip(
            constrained, p.theta_min_physical, p.theta_max_physical)

        return constrained


def create_site_calibrator(
    latitude: float,
    longitude: float,
    sand_percent: float,
    clay_percent: float,
    ndvi_mean: Optional[float] = None,
    annual_precip_mm: Optional[float] = None,
    observed_stats: Optional[Dict[str, float]] = None,
    elevation_m: Optional[float] = None,
    slope_percent: Optional[float] = None,
) -> AdaptivePhysicsCalibrator:
    """
    Factory function to create a site-specific calibrator.

    Args:
        latitude: Site latitude
        longitude: Site longitude
        sand_percent: Soil sand content (%)
        clay_percent: Soil clay content (%)
        ndvi_mean: Mean NDVI (optional)
        annual_precip_mm: Annual precipitation (optional)
        observed_stats: Dict with 'mean', 'std', 'min', 'max' of observed SM
        elevation_m: Site elevation (m, optional)
        slope_percent: Site slope (%, optional)

    Returns:
        Configured AdaptivePhysicsCalibrator
    """
    site_chars = SiteCharacteristics.estimate_from_location(
        latitude=latitude,
        longitude=longitude,
        sand_percent=sand_percent,
        clay_percent=clay_percent,
        ndvi_mean=ndvi_mean,
        annual_precip_mm=annual_precip_mm,
    )

    # Override with provided values if available
    if elevation_m is not None:
        site_chars.elevation_m = elevation_m
    if slope_percent is not None:
        site_chars.slope_percent = slope_percent

    return AdaptivePhysicsCalibrator(
        site_chars=site_chars,
        observed_stats=observed_stats,
    )


# =============================================================================
# IMPROVED PEDOTRANSFER FUNCTIONS FOR TROPICAL SOILS
# =============================================================================

def tropical_ptf_van_genuchten(
    sand_percent: float,
    clay_percent: float,
    organic_matter_percent: float = 2.0,
    bulk_density: float = 1.35,
    soil_type: str = "ferralsol",
) -> Dict[str, float]:
    """
    Pedotransfer function for tropical soils with soil-type specific corrections.

    Based on Hodnett & Tomasella (2002) with modifications for African soil types.

    Key differences by soil type:
    - Ferralsols: Well-aggregated oxide clays, high porosity, high Ksat
    - Nitisols: Moderately weathered, less aggregated, moderate properties
    - Vertisols: Shrink-swell clays, high water retention, variable Ksat

    Args:
        sand_percent: Sand content (%)
        clay_percent: Clay content (%)
        organic_matter_percent: Organic matter content (%)
        bulk_density: Bulk density (g/cm³)
        soil_type: WRB soil classification ("ferralsol", "nitisol", "vertisol")

    Returns:
        Dictionary with VG parameters
    """
    silt_percent = 100 - sand_percent - clay_percent
    om = organic_matter_percent
    bd = bulk_density

    # Soil-type specific corrections
    # todo: Use the iSDA Africa Soil Database API (primary source) to pull the soil_type or classification field.
    # todo: Map the iSDA classification to this soil_corrections dictionary rather than using the current blanket clay > 25% rule
    soil_corrections = {
        'ferralsol': {
            'aggregation_factor': 1.0,  # Well aggregated
            'macropore_factor': 2.0,   # High macroporosity
            'porosity_bonus': 0.03,    # Higher porosity
            'alpha_adjustment': 0.3,   # Higher alpha (coarser behavior)
            'n_adjustment': 0.1,       # Narrower pore distribution
        },
        'nitisol': {
            'aggregation_factor': 0.7,  # Moderately aggregated
            'macropore_factor': 1.3,    # Moderate macroporosity
            'porosity_bonus': 0.01,     # Lower porosity than ferralsols
            'alpha_adjustment': 0.1,    # Moderate alpha increase
            'n_adjustment': 0.05,       # Slight narrowing
        },
        'vertisol': {
            'aggregation_factor': 0.5,  # Poor aggregation (cracks when dry)
            'macropore_factor': 0.8,    # Lower macroporosity when wet
            'porosity_bonus': 0.02,     # High total porosity but variable
            'alpha_adjustment': -0.2,   # Lower alpha (finer behavior)
            'n_adjustment': -0.1,       # Wider pore distribution
        },
        'default': {
            'aggregation_factor': 0.8,
            'macropore_factor': 1.5,
            'porosity_bonus': 0.02,
            'alpha_adjustment': 0.2,
            'n_adjustment': 0.05,
        }
    }

    corrections = soil_corrections.get(
        soil_type.lower(), soil_corrections['default'])

    # Hodnett & Tomasella (2002) base equations
    theta_s = (
        0.81 -
        0.283 * bd +
        0.001 * clay_percent +
        0.003 * om
    )

    # Apply soil-type specific porosity adjustment
    theta_s += corrections['porosity_bonus']
    theta_s = np.clip(theta_s, 0.35, 0.65)

    # Residual water content
    theta_r = (
        0.015 +
        0.005 * clay_percent +
        0.014 * om -
        0.001 * sand_percent
    )
    theta_r = np.clip(theta_r, 0.01, 0.20)

    # Van Genuchten alpha (1/m) with soil-type corrections
    log_alpha = (
        -0.6 +
        0.013 * sand_percent -
        0.007 * clay_percent -
        0.01 * om
    )

    # Apply soil-type specific alpha adjustment
    log_alpha += corrections['alpha_adjustment']

    alpha = 10 ** log_alpha * 100  # Convert to 1/m
    alpha = np.clip(alpha, 0.5, 20.0)

    # Van Genuchten n with soil-type corrections
    n = (
        1.15 +
        0.01 * sand_percent -
        0.006 * clay_percent -
        0.005 * om
    )

    n += corrections['n_adjustment']
    n = np.clip(n, 1.05, 2.5)

    # Saturated hydraulic conductivity with soil-type corrections
    log_ksat = (
        0.5 +
        0.02 * sand_percent -
        0.015 * clay_percent -
        0.01 * om
    )

    ksat_cm_day = (10 ** log_ksat) * corrections['macropore_factor']
    ksat_m_day = ksat_cm_day / 100
    ksat_m_day = np.clip(ksat_m_day, 0.01, 10.0)

    return {
        'alpha': alpha,
        'n': n,
        'theta_r': theta_r,
        'theta_s': theta_s,
        'K_sat': ksat_m_day,
    }


def estimate_field_capacity_tropical(
    sand_percent: float,
    clay_percent: float,
    organic_matter_percent: float = 2.0,
) -> float:
    """
    Estimate field capacity for tropical soils.

    Field capacity in tropical soils is often higher than temperate
    predictions due to:
    1. Higher organic matter in surface
    2. Better structure from oxide aggregation
    3. Different clay mineralogy
    """
    om = organic_matter_percent

    # Base from Saxton-Rawls type
    fc = (
        0.20 +
        0.002 * clay_percent +
        0.001 * (100 - sand_percent) +
        0.01 * om
    )

    # Tropical adjustment
    if clay_percent > 35:
        fc += 0.03  # More water retention in tropical clays

    return np.clip(fc, 0.15, 0.45)


def estimate_wilting_point_tropical(
    sand_percent: float,
    clay_percent: float,
    organic_matter_percent: float = 2.0,
) -> float:
    """
    Estimate wilting point for tropical soils.

    Wilting point is less affected by tropical conditions but
    aggregated clays may have lower WP than expected.
    """
    wp = (
        0.05 +
        0.003 * clay_percent +
        0.005 * organic_matter_percent
    )

    # Aggregated tropical clays may release water more easily
    if clay_percent > 40:
        wp *= 0.9

    return np.clip(wp, 0.03, 0.25)
