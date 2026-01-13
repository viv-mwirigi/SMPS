"""
Dynamic Crop Development Model with Phenology-Driven Parameters.


1. Time-varying Kcb curves with phenological stages
2. GDD-based root depth growth
3. Dynamic root distribution based on soil moisture
4. Residue cover effects on surface evaporation

References:
- FAO-56: Allen et al. (1998) Crop evapotranspiration
- Ritchie (1991) Soil water availability. Plant and Soil 58:327-338
- Feddes & Raats (2004) Parameterizing the soil-water-plant root system
- Simunek et al. (2016) HYDRUS Technical Manual
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# PHENOLOGICAL STAGES
# =============================================================================

class GrowthStage(Enum):
    """Crop growth stages following FAO-56"""
    INITIAL = "initial"      # Planting to 10% ground cover
    DEVELOPMENT = "development"  # 10% to effective full cover
    MID_SEASON = "mid_season"    # Full cover to start of maturity
    LATE_SEASON = "late_season"  # Maturity to harvest
    DORMANT = "dormant"          # Off-season / fallow


@dataclass
class PhenologyParameters:
    """
    Crop phenology parameters for GDD-based development.

    Growing Degree Days (GDD) calculated as:
        GDD = max(0, (Tmax + Tmin)/2 - Tbase)

    Accumulated GDD drives stage transitions.
    """
    # Base temperature for GDD accumulation (°C)
    T_base: float = 10.0

    # Optimum temperature (°C)
    T_opt: float = 30.0

    # Maximum temperature cutoff (°C)
    T_max: float = 40.0

    # GDD requirements for stage transitions
    GDD_emergence: float = 100.0     # Planting to emergence
    GDD_initial_end: float = 300.0   # End of initial stage
    GDD_dev_end: float = 700.0       # End of development stage
    GDD_mid_end: float = 1200.0      # End of mid-season
    GDD_maturity: float = 1500.0     # Physiological maturity

    # Frost sensitivity threshold (°C)
    T_frost_kill: float = -2.0

    @classmethod
    def for_crop(cls, crop_name: str) -> "PhenologyParameters":
        """Get crop-specific phenology parameters."""
        # GDD requirements from crop literature
        # Format: (T_base, T_opt, T_max, GDD_emerg, GDD_ini, GDD_dev, GDD_mid, GDD_mat)
        crops = {
            'maize': (10.0, 30.0, 40.0, 80, 300, 700, 1200, 1500),
            'wheat': (0.0, 25.0, 35.0, 100, 350, 800, 1100, 1400),
            'rice': (10.0, 30.0, 40.0, 100, 400, 900, 1500, 1800),
            'sorghum': (10.0, 32.0, 42.0, 100, 350, 750, 1100, 1400),
            'millet': (10.0, 32.0, 42.0, 80, 300, 600, 1000, 1200),
            'soybean': (10.0, 28.0, 38.0, 100, 350, 800, 1200, 1500),
            'groundnut': (13.0, 30.0, 40.0, 120, 400, 800, 1200, 1500),
            'cotton': (12.0, 30.0, 40.0, 100, 400, 900, 1400, 1800),
            'cassava': (15.0, 30.0, 38.0, 150, 500, 1500, 3500, 4500),
            'beans': (10.0, 25.0, 35.0, 80, 250, 500, 800, 1000),
            'grassland': (5.0, 25.0, 35.0, 50, 200, 400, 2000, 2500),
            'savanna': (10.0, 30.0, 40.0, 100, 400, 800, 2000, 2500),
        }

        crop_key = crop_name.lower().replace(' ', '_')
        if crop_key not in crops:
            logger.warning(f"Unknown crop '{crop_name}', using 'maize'")
            crop_key = 'maize'

        data = crops[crop_key]
        return cls(
            T_base=data[0], T_opt=data[1], T_max=data[2],
            GDD_emergence=data[3], GDD_initial_end=data[4],
            GDD_dev_end=data[5], GDD_mid_end=data[6], GDD_maturity=data[7]
        )


@dataclass
class RootGrowthParameters:
    """
    Parameters for dynamic root depth growth.

    Root depth increases with accumulated GDD until maximum:
        depth(t) = min(max_depth, initial + growth_rate × GDD)

    Root distribution adjusts based on soil moisture profile.
    """
    # Initial root depth at emergence (m)
    initial_depth_m: float = 0.10

    # Maximum root depth (m)
    max_depth_m: float = 1.0

    # Root growth rate (m per 100 GDD)
    growth_rate_m_per_100gdd: float = 0.08

    # Root distribution shape parameter
    # Lower = more uniform, Higher = more surface-concentrated
    beta_nominal: float = 0.97

    # Maximum root density (m root / m³ soil) at peak development
    max_root_density: float = 5000.0

    # Root senescence rate during late season (fraction per day)
    senescence_rate: float = 0.02

    # Plasticity parameter - how much roots adjust to moisture
    # 0 = static distribution, 1 = fully plastic
    plasticity: float = 0.3

    @classmethod
    def for_crop(cls, crop_name: str) -> "RootGrowthParameters":
        """Get crop-specific root growth parameters."""
        # Format: (init_depth, max_depth, growth_rate, beta, max_density, senescence)
        crops = {
            'maize': (0.10, 1.0, 0.08, 0.97, 5000, 0.02),
            'wheat': (0.08, 1.0, 0.06, 0.96, 6000, 0.015),
            'rice': (0.10, 0.5, 0.04, 0.94, 4000, 0.01),
            'sorghum': (0.10, 1.0, 0.10, 0.97, 4500, 0.02),
            'millet': (0.08, 1.0, 0.12, 0.97, 4000, 0.025),
            'soybean': (0.08, 0.6, 0.05, 0.95, 5000, 0.02),
            'groundnut': (0.10, 0.5, 0.04, 0.94, 4500, 0.02),
            'cotton': (0.10, 1.3, 0.10, 0.97, 5000, 0.015),
            'cassava': (0.15, 0.7, 0.03, 0.93, 3500, 0.01),
            'beans': (0.08, 0.5, 0.05, 0.95, 4500, 0.025),
            'grassland': (0.15, 0.5, 0.02, 0.93, 8000, 0.005),
            'savanna': (0.20, 1.5, 0.05, 0.95, 3000, 0.01),
            'forest': (0.30, 2.0, 0.03, 0.96, 2500, 0.005),
        }

        crop_key = crop_name.lower().replace(' ', '_')
        if crop_key not in crops:
            crop_key = 'maize'

        data = crops[crop_key]
        return cls(
            initial_depth_m=data[0], max_depth_m=data[1],
            growth_rate_m_per_100gdd=data[2], beta_nominal=data[3],
            max_root_density=data[4], senescence_rate=data[5]
        )


@dataclass
class ResidueCoverParameters:
    """
    Parameters for crop residue affecting surface evaporation.

    Residue cover reduces soil evaporation by:
    1. Reducing incoming radiation at soil surface
    2. Increasing surface resistance to vapor diffusion
    3. Reducing wind speed at soil surface
    """
    # Residue mass (kg/ha)
    residue_mass_kg_ha: float = 0.0

    # Cover factor per unit residue mass (m²/kg)
    # Typical: 2-4 ×10⁻⁴ for cereal stover
    cover_factor: float = 3.0e-4

    # Maximum fractional cover (even with high residue)
    max_cover_fraction: float = 0.95

    # Decay rate of residue (fraction per day at 25°C)
    decay_rate_25C: float = 0.005

    @property
    def fractional_cover(self) -> float:
        """Calculate fractional ground cover from residue."""
        # Gregory (1982) exponential relationship
        # fc = 1 - exp(-k × M)
        k = self.cover_factor
        M = self.residue_mass_kg_ha
        fc = 1 - np.exp(-k * M)
        return min(fc, self.max_cover_fraction)


# =============================================================================
# CROP DEVELOPMENT STATE
# =============================================================================

@dataclass
class CropState:
    """Current state of crop development."""
    # Accumulated GDD since planting
    accumulated_gdd: float = 0.0

    # Days since planting
    days_since_planting: int = 0

    # Current growth stage
    growth_stage: GrowthStage = GrowthStage.INITIAL

    # Current root depth (m)
    current_root_depth_m: float = 0.10

    # Current Kcb value
    current_Kcb: float = 0.15

    # Current LAI
    current_LAI: float = 0.1

    # Fractional canopy cover
    fractional_cover: float = 0.0

    # Root distribution by layer (fractions summing to 1)
    root_fractions: np.ndarray = field(default_factory=lambda: np.array([1.0]))

    # Accumulated stress days
    stress_days: int = 0

    # Is crop active (vs dormant/harvested)
    is_active: bool = True

    # Residue cover state
    residue_mass_kg_ha: float = 0.0


# =============================================================================
# CROP DEVELOPMENT MODEL
# =============================================================================

class CropDevelopmentModel:
    """
    Dynamic crop development model integrating phenology and root growth.

    Implements Gap 7 requirements:
    - Time-varying Kcb curves with phenological stages
    - GDD-based root depth growth
    - Dynamic root distribution based on soil moisture
    - Residue cover effects
    """

    def __init__(
        self,
        crop_name: str = "maize",
        phenology_params: Optional[PhenologyParameters] = None,
        root_params: Optional[RootGrowthParameters] = None,
        residue_params: Optional[ResidueCoverParameters] = None,
        layer_depths_m: Optional[np.ndarray] = None
    ):
        """
        Initialize crop development model.

        Args:
            crop_name: Crop type for default parameters
            phenology_params: Override phenology parameters
            root_params: Override root growth parameters
            residue_params: Override residue parameters
            layer_depths_m: Depth to bottom of each soil layer
        """
        self.crop_name = crop_name

        # Load crop-specific parameters
        self.phenology = phenology_params or PhenologyParameters.for_crop(
            crop_name)
        self.root_params = root_params or RootGrowthParameters.for_crop(
            crop_name)
        self.residue_params = residue_params or ResidueCoverParameters()

        # Soil layer structure (default 3 layers)
        self.layer_depths_m = layer_depths_m if layer_depths_m is not None else \
            np.array([0.1, 0.4, 1.0])

        # FAO-56 crop coefficients
        from smps.physics.evapotranspiration import CropCoefficientCurve
        self.Kc_curve = CropCoefficientCurve.for_crop(crop_name)

        # Initialize state
        self.state = CropState(
            root_fractions=self._calculate_initial_root_fractions()
        )

    def _calculate_initial_root_fractions(self) -> np.ndarray:
        """Calculate initial root distribution."""
        n_layers = len(self.layer_depths_m)
        fractions = np.zeros(n_layers)

        # Initial roots only in first layer
        initial_depth = self.root_params.initial_depth_m
        if self.layer_depths_m[0] >= initial_depth:
            fractions[0] = 1.0
        else:
            # Distribute if initial depth spans multiple layers
            fractions = self._calculate_root_fractions(
                initial_depth,
                self.root_params.beta_nominal
            )

        return fractions

    def _calculate_root_fractions(
        self,
        root_depth: float,
        beta: float
    ) -> np.ndarray:
        """
        Calculate root fraction in each soil layer using Gale-Grigal model.

        Y = 1 - β^d
        where Y is cumulative root fraction and d is depth
        """
        n_layers = len(self.layer_depths_m)
        fractions = np.zeros(n_layers)

        # Calculate cumulative root fraction at each layer boundary
        prev_depth = 0.0
        prev_cum_frac = 0.0

        for i, layer_bottom in enumerate(self.layer_depths_m):
            if layer_bottom <= root_depth:
                # Layer fully within rooting zone
                # depth in cm for β
                cum_frac = 1 - beta ** (layer_bottom * 100)
                fractions[i] = cum_frac - prev_cum_frac
                prev_cum_frac = cum_frac
            elif prev_depth < root_depth:
                # Layer partially within rooting zone
                # Interpolate to root depth
                cum_frac_at_root = 1 - beta ** (root_depth * 100)
                fractions[i] = cum_frac_at_root - prev_cum_frac
                prev_cum_frac = cum_frac_at_root
            else:
                # Layer below rooting zone
                fractions[i] = 0.0

            prev_depth = layer_bottom

        # Normalize to sum to 1
        total = np.sum(fractions)
        if total > 0:
            fractions = fractions / total
        else:
            fractions[0] = 1.0

        return fractions

    def calculate_gdd(
        self,
        T_max: float,
        T_min: float
    ) -> float:
        """
        Calculate Growing Degree Days for one day.

        Uses standard method with cutoffs:
            GDD = max(0, (min(Tmax, Tupper) + max(Tmin, Tbase)) / 2 - Tbase)

        Args:
            T_max: Maximum daily temperature (°C)
            T_min: Minimum daily temperature (°C)

        Returns:
            GDD for the day
        """
        # Apply temperature bounds
        T_max_adj = min(T_max, self.phenology.T_max)
        T_min_adj = max(T_min, self.phenology.T_base)

        # Average temperature method
        T_avg = (T_max_adj + T_min_adj) / 2
        gdd = max(0.0, T_avg - self.phenology.T_base)

        return gdd

    def update_growth_stage(self) -> GrowthStage:
        """Determine current growth stage from accumulated GDD."""
        gdd = self.state.accumulated_gdd

        if gdd < self.phenology.GDD_emergence:
            return GrowthStage.INITIAL
        elif gdd < self.phenology.GDD_initial_end:
            return GrowthStage.INITIAL
        elif gdd < self.phenology.GDD_dev_end:
            return GrowthStage.DEVELOPMENT
        elif gdd < self.phenology.GDD_mid_end:
            return GrowthStage.MID_SEASON
        elif gdd < self.phenology.GDD_maturity:
            return GrowthStage.LATE_SEASON
        else:
            return GrowthStage.DORMANT

    def calculate_Kcb(self) -> float:
        """
        Calculate current basal crop coefficient based on GDD.

        Uses linear interpolation between stages similar to FAO-56
        but driven by GDD rather than fixed days.
        """
        gdd = self.state.accumulated_gdd
        Kcb = self.Kc_curve

        # Pre-emergence
        if gdd < self.phenology.GDD_emergence:
            return Kcb.Kcb_ini * 0.5

        # Initial stage
        if gdd < self.phenology.GDD_initial_end:
            return Kcb.Kcb_ini

        # Development stage - linear increase
        if gdd < self.phenology.GDD_dev_end:
            frac = (gdd - self.phenology.GDD_initial_end) / \
                   (self.phenology.GDD_dev_end - self.phenology.GDD_initial_end)
            return Kcb.Kcb_ini + frac * (Kcb.Kcb_mid - Kcb.Kcb_ini)

        # Mid-season
        if gdd < self.phenology.GDD_mid_end:
            return Kcb.Kcb_mid

        # Late season - linear decrease
        if gdd < self.phenology.GDD_maturity:
            frac = (gdd - self.phenology.GDD_mid_end) / \
                   (self.phenology.GDD_maturity - self.phenology.GDD_mid_end)
            return Kcb.Kcb_mid + frac * (Kcb.Kcb_end - Kcb.Kcb_mid)

        # Post-maturity
        return Kcb.Kcb_end * 0.5

    def calculate_root_depth(self) -> float:
        """
        Calculate current root depth based on GDD.

        depth(t) = min(max_depth, initial + growth_rate × GDD)

        During late season, effective depth may decrease due to senescence.
        """
        gdd = self.state.accumulated_gdd
        params = self.root_params

        # Pre-emergence - minimal roots
        if gdd < self.phenology.GDD_emergence:
            return params.initial_depth_m * 0.5

        # Calculate potential depth from GDD
        gdd_since_emergence = gdd - self.phenology.GDD_emergence
        potential_depth = (
            params.initial_depth_m +
            params.growth_rate_m_per_100gdd * (gdd_since_emergence / 100)
        )

        # Apply maximum depth constraint
        depth = min(potential_depth, params.max_depth_m)

        # Late season reduction due to senescence
        if self.state.growth_stage == GrowthStage.LATE_SEASON:
            senescence_factor = 1 - params.senescence_rate * self.state.days_since_planting
            senescence_factor = max(
                0.5, senescence_factor)  # Don't go below 50%
            depth *= senescence_factor

        return depth

    def calculate_LAI(self) -> float:
        """
        Estimate LAI from Kcb following FAO-56 relationship.

        LAI ≈ -ln(1 - Kcb/1.2) / 0.5 for crops
        """
        Kcb = self.state.current_Kcb

        # Empirical relationship
        if Kcb < 0.01:
            return 0.0

        # Clamp Kcb for calculation
        Kcb_clamp = min(Kcb, 1.15)

        # FAO approximation
        fc = min(0.99, Kcb_clamp / 1.2)  # Fractional cover estimate
        if fc < 0.01:
            return 0.0

        LAI = -np.log(1 - fc) / 0.5

        return min(LAI, 8.0)  # Cap at realistic maximum

    def adjust_root_distribution_for_moisture(
        self,
        layer_moisture_fractions: np.ndarray
    ) -> np.ndarray:
        """
        Adjust root distribution based on soil moisture profile.

        Roots proliferate in layers with more available water.
        This implements "root plasticity" - the ability of roots to
        respond to soil moisture heterogeneity.

        Args:
            layer_moisture_fractions: Relative available water (0-1) in each layer

        Returns:
            Adjusted root fractions
        """
        nominal_fractions = self._calculate_root_fractions(
            self.state.current_root_depth_m,
            self.root_params.beta_nominal
        )

        if self.root_params.plasticity == 0:
            return nominal_fractions

        # Calculate moisture-weighted adjustment
        # More roots in wetter layers
        moisture = np.clip(layer_moisture_fractions, 0.1, 1.0)

        # Weight nominal distribution by moisture
        weighted = nominal_fractions * moisture ** self.root_params.plasticity

        # Renormalize
        total = np.sum(weighted)
        if total > 0:
            adjusted = weighted / total
        else:
            adjusted = nominal_fractions

        # Blend with nominal based on plasticity
        p = self.root_params.plasticity
        result = (1 - p) * nominal_fractions + p * adjusted

        return result / np.sum(result)  # Ensure sums to 1

    def update_residue(
        self,
        T_avg: float,
        precipitation_mm: float = 0.0,
        added_residue_kg_ha: float = 0.0
    ) -> float:
        """
        Update residue mass accounting for decomposition.

        Decomposition rate follows Arrhenius-type temperature dependence.

        Args:
            T_avg: Average temperature (°C)
            precipitation_mm: Daily precipitation
            added_residue_kg_ha: Newly added residue (e.g., from harvest)

        Returns:
            Updated residue mass (kg/ha)
        """
        params = self.residue_params

        # Add new residue
        mass = self.state.residue_mass_kg_ha + added_residue_kg_ha

        # Temperature adjustment for decay rate
        # Q10 ≈ 2 (rate doubles per 10°C)
        T_adj = (T_avg - 25) / 10
        decay_rate = params.decay_rate_25C * (2 ** T_adj)

        # Moisture effect - decomposition faster when wet
        if precipitation_mm > 5:
            decay_rate *= 1.5

        # Apply decay
        mass *= (1 - decay_rate)

        self.state.residue_mass_kg_ha = max(0, mass)

        return self.state.residue_mass_kg_ha

    def calculate_residue_evaporation_reduction(self) -> float:
        """
        Calculate reduction factor for soil evaporation due to residue cover.

        Returns:
            Reduction factor (0-1, where 1 = no reduction)
        """
        # Update residue parameters with current mass
        self.residue_params.residue_mass_kg_ha = self.state.residue_mass_kg_ha

        fc_residue = self.residue_params.fractional_cover

        # Exponential reduction in evaporation with cover
        # E_reduction = exp(-0.5 × LAI_residue)
        # where LAI_residue ≈ -ln(1 - fc) / 0.4
        if fc_residue > 0.01:
            LAI_equiv = -np.log(1 - fc_residue) / 0.4
            reduction = np.exp(-0.5 * LAI_equiv)
        else:
            reduction = 1.0

        return reduction

    def advance_day(
        self,
        T_max: float,
        T_min: float,
        layer_moisture_fractions: Optional[np.ndarray] = None,
        water_stress_factor: float = 1.0
    ) -> CropState:
        """
        Advance crop development by one day.

        Args:
            T_max: Maximum daily temperature (°C)
            T_min: Minimum daily temperature (°C)
            layer_moisture_fractions: Relative water content per layer
            water_stress_factor: Overall stress factor (0-1)

        Returns:
            Updated CropState
        """
        if not self.state.is_active:
            return self.state

        # Calculate GDD
        gdd_today = self.calculate_gdd(T_max, T_min)

        # Reduce GDD accumulation under severe stress
        if water_stress_factor < 0.5:
            gdd_today *= (0.5 + water_stress_factor)

        # Update accumulated GDD
        self.state.accumulated_gdd += gdd_today
        self.state.days_since_planting += 1

        # Update growth stage
        self.state.growth_stage = self.update_growth_stage()

        # Update Kcb
        self.state.current_Kcb = self.calculate_Kcb()

        # Update root depth
        self.state.current_root_depth_m = self.calculate_root_depth()

        # Update LAI
        self.state.current_LAI = self.calculate_LAI()

        # Update fractional cover
        self.state.fractional_cover = 1 - np.exp(-0.5 * self.state.current_LAI)

        # Update root distribution
        if layer_moisture_fractions is not None:
            self.state.root_fractions = self.adjust_root_distribution_for_moisture(
                layer_moisture_fractions
            )
        else:
            self.state.root_fractions = self._calculate_root_fractions(
                self.state.current_root_depth_m,
                self.root_params.beta_nominal
            )

        # Track stress
        if water_stress_factor < 0.8:
            self.state.stress_days += 1

        # Update residue
        T_avg = (T_max + T_min) / 2
        self.update_residue(T_avg)

        # Check for maturity
        if self.state.accumulated_gdd >= self.phenology.GDD_maturity:
            self.state.is_active = False
            # Add harvest residue
            self.state.residue_mass_kg_ha += 3000  # Typical residue addition

        # Check for frost damage
        if T_min < self.phenology.T_frost_kill:
            if self.state.growth_stage in [GrowthStage.INITIAL, GrowthStage.DEVELOPMENT]:
                logger.warning(f"Frost damage at {T_min:.1f}°C")
                self.state.current_Kcb *= 0.5
                self.state.current_LAI *= 0.5

        return self.state

    def reset(
        self,
        initial_residue_kg_ha: float = 0.0,
        planting_day: int = 0
    ):
        """Reset model for new growing season."""
        self.state = CropState(
            root_fractions=self._calculate_initial_root_fractions(),
            residue_mass_kg_ha=initial_residue_kg_ha,
            days_since_planting=planting_day
        )

    def get_current_parameters(self) -> Dict[str, float]:
        """Get current crop parameters as dictionary."""
        return {
            'accumulated_gdd': self.state.accumulated_gdd,
            'days_since_planting': self.state.days_since_planting,
            'growth_stage': self.state.growth_stage.value,
            'Kcb': self.state.current_Kcb,
            'root_depth_m': self.state.current_root_depth_m,
            'LAI': self.state.current_LAI,
            'fractional_cover': self.state.fractional_cover,
            'root_fractions': list(self.state.root_fractions),
            'residue_cover': self.residue_params.fractional_cover,
            'evap_reduction_factor': self.calculate_residue_evaporation_reduction(),
            'is_active': self.state.is_active
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_crop_model(
    crop_name: str,
    layer_depths_m: Optional[np.ndarray] = None,
    initial_residue_kg_ha: float = 0.0
) -> CropDevelopmentModel:
    """
    Create a crop development model with standard parameters.

    Args:
        crop_name: Crop type (maize, wheat, rice, etc.)
        layer_depths_m: Depth to bottom of each soil layer
        initial_residue_kg_ha: Initial residue from previous crop

    Returns:
        Configured CropDevelopmentModel
    """
    model = CropDevelopmentModel(
        crop_name=crop_name,
        layer_depths_m=layer_depths_m,
        residue_params=ResidueCoverParameters(
            residue_mass_kg_ha=initial_residue_kg_ha
        )
    )
    return model


def estimate_planting_window(
    crop_name: str,
    latitude: float,
    rainfall_onset_doy: Optional[int] = None
) -> Tuple[int, int]:
    """
    Estimate typical planting window for a crop based on location.

    Args:
        crop_name: Crop type
        latitude: Location latitude
        rainfall_onset_doy: Day of year when rains typically start

    Returns:
        Tuple of (earliest_doy, latest_doy) for planting
    """
    # Simple heuristics based on tropical vs temperate
    if abs(latitude) < 23.5:  # Tropics
        if rainfall_onset_doy is not None:
            earliest = rainfall_onset_doy
            latest = rainfall_onset_doy + 45
        else:
            # Assume bimodal rainfall
            # First season: March-April
            # Second season: September-October
            earliest = 60  # March
            latest = 120   # April
    else:
        # Temperate - spring planting
        if latitude > 0:  # Northern hemisphere
            earliest = 90   # April
            latest = 150    # May
        else:  # Southern hemisphere
            earliest = 270  # October
            latest = 330    # November

    return (int(earliest), int(latest))
