"""
Tropical Soil Corrections for SWPPS.

Implements corrections for tropical/African soils:
- Oxide aggregation effects (ferralsols, nitisols)
- Macropore flow adjustments
- Regional climate corrections

Based on research from:
- Hodnett & Tomasella (2002) - Tropical PTFs
- Saxton & Rawls (2006) - With tropical modifications
- FAO World Reference Base soil corrections
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger("swpps.physics.tropical")


@dataclass
class TropicalSoilCorrections:
    """
    Corrections for tropical and African soils.

    CRITICAL: Standard Van Genuchten PTFs were developed for temperate soils
    with montmorillonite (2:1 swelling) clays. African tropical soils are
    fundamentally different:

    1. KAOLINITE DOMINANCE: African soils are 70-95% kaolinite (1:1 clay)
       - Non-swelling, low CEC (1-15 vs 20-100 meq/100g for montmorillonite)
       - Holds ~20% as much water per unit clay as montmorillonite
       - Creates stable microaggregates ("pseudosand" behavior)

    2. Fe/Al OXIDE AGGREGATION: Ferralsols have 15-40% Fe₂O₃ + Al₂O₃
       - Oxide-clay bonding creates stable aggregates
       - Bimodal porosity: large inter-aggregate + small intra-aggregate pores
       - Higher θr due to water adsorption on oxide surfaces

    3. MACROPORE NETWORKS: Extensive macropores from:
       - Termite activity (very high in African savannas)
       - Root channels from deep-rooted vegetation
       - Wet-dry cracking cycles

    These corrections are based on Hodnett & Tomasella (2002) and subsequent
    tropical soil research.

    References:
    - Hodnett & Tomasella (2002) Geoderma 107:157-166
    - Minasny & Hartemink (2011) Earth-Sci Rev 106:52-62
    - FAO World Reference Base soil descriptions
    """
    # Base soil texture
    clay_fraction: float
    sand_fraction: float
    silt_fraction: float = field(default=None)

    # CRITICAL: Clay mineralogy
    # Fraction of clay that is kaolinite (0-1)
    kaolinite_fraction: float = 0.80
    # African tropics: 0.7-0.95 typically

    # Fe/Al oxide content (% by weight)
    fe_oxide_pct: float = 10.0          # Fe₂O₃ content (%)
    al_oxide_pct: float = 8.0           # Al₂O₃ content (%)
    oxide_content: float = 0.1          # Combined 0-1 scale (legacy)

    organic_carbon_pct: float = 1.5     # Organic carbon (%)
    cec_cmol_kg: float = 15.0           # Cation exchange capacity

    # Environmental factors
    mean_annual_precip_mm: float = 1000.0
    mean_annual_temp_c: float = 25.0
    wet_dry_cycles: bool = True         # Distinct wet/dry seasons

    # Biological activity
    termite_activity: str = "moderate"  # none, low, moderate, high
    root_channels: bool = True

    def __post_init__(self):
        if self.silt_fraction is None:
            self.silt_fraction = 1.0 - self.clay_fraction - self.sand_fraction

    @classmethod
    def for_african_soil(
        cls,
        soil_type: str,
        clay_fraction: float,
        sand_fraction: float,
        organic_carbon_pct: float = 1.5,
        mean_annual_precip_mm: float = 1000.0,
    ) -> "TropicalSoilCorrections":
        """
        Create corrections for African soil type.

        Args:
            soil_type: One of 'ferralsol', 'nitisol', 'acrisol', 'vertisol', 'arenosol', 'generic'
            clay_fraction: Clay fraction (0-1)
            sand_fraction: Sand fraction (0-1)
            organic_carbon_pct: Organic carbon percentage
            mean_annual_precip_mm: Mean annual precipitation
        """
        # Soil type specific defaults based on FAO WRB and African soil surveys
        # Values reflect typical African soil characteristics
        type_params = {
            'ferralsol': {
                'kaolinite_fraction': 0.90,  # Almost pure kaolinite
                'fe_oxide_pct': 18.0,        # High Fe oxides (red color)
                'al_oxide_pct': 25.0,        # High Al oxides
                'oxide_content': 0.35,
                'cec_cmol_kg': 5.0,          # Very low CEC typical of ferralsols
                'termite_activity': 'high',
            },
            'nitisol': {
                'kaolinite_fraction': 0.85,
                'fe_oxide_pct': 12.0,
                'al_oxide_pct': 18.0,
                'oxide_content': 0.25,
                'cec_cmol_kg': 10.0,
                'termite_activity': 'moderate',
            },
            'acrisol': {
                'kaolinite_fraction': 0.80,
                'fe_oxide_pct': 8.0,
                'al_oxide_pct': 12.0,
                'oxide_content': 0.15,
                'cec_cmol_kg': 8.0,
                'termite_activity': 'moderate',
            },
            'vertisol': {
                'kaolinite_fraction': 0.30,  # More montmorillonite (swelling)
                'fe_oxide_pct': 5.0,
                'al_oxide_pct': 8.0,
                'oxide_content': 0.05,
                'cec_cmol_kg': 35.0,         # High CEC from smectite clays
                'termite_activity': 'low',
            },
            'arenosol': {
                'kaolinite_fraction': 0.70,
                'fe_oxide_pct': 3.0,
                'al_oxide_pct': 5.0,
                'oxide_content': 0.05,
                'cec_cmol_kg': 3.0,
                'termite_activity': 'low',
            },
            'generic': {
                'kaolinite_fraction': 0.75,  # Default for African tropics
                'fe_oxide_pct': 10.0,
                'al_oxide_pct': 12.0,
                'oxide_content': 0.10,
                'cec_cmol_kg': 12.0,
                'termite_activity': 'moderate',
            },
        }

        key = soil_type.lower()
        if key not in type_params:
            logger.warning(f"Unknown soil type '{soil_type}', using 'generic'")
            key = 'generic'

        params = type_params[key]

        return cls(
            clay_fraction=clay_fraction,
            sand_fraction=sand_fraction,
            kaolinite_fraction=params['kaolinite_fraction'],
            fe_oxide_pct=params['fe_oxide_pct'],
            al_oxide_pct=params['al_oxide_pct'],
            oxide_content=params['oxide_content'],
            organic_carbon_pct=organic_carbon_pct,
            cec_cmol_kg=params['cec_cmol_kg'],
            mean_annual_precip_mm=mean_annual_precip_mm,
            termite_activity=params['termite_activity'],
            root_channels=True,
            wet_dry_cycles=mean_annual_precip_mm < 1800,
        )

    def get_oxide_aggregation_factor(self) -> float:
        """
        Calculate aggregation factor from Fe/Al oxide content.

        Iron and aluminum oxides in tropical soils:
        1. Coat clay particles, reducing their effective surface area
        2. Create stable microaggregates ("pseudosand")
        3. Form Fe-O-Si and Al-O-Si bonds that resist dispersion

        High oxide ferralsols (>30% Fe₂O₃+Al₂O₃) behave like sandy soils
        despite having >50% clay content!

        Returns:
            Aggregation factor (1.0 = no effect, >1 = increased macropores)
        """
        # Use actual oxide percentages if available
        total_oxide_pct = self.fe_oxide_pct + self.al_oxide_pct

        # Threshold effects: aggregation increases nonlinearly above ~15%
        if total_oxide_pct > 15:
            oxide_effect = 0.2 + 0.03 * (total_oxide_pct - 15)
        else:
            oxide_effect = total_oxide_pct * 0.013

        # Clay-oxide interaction (more clay + oxide = stronger aggregation)
        # This is the key mechanism in ferralsols
        clay_oxide_effect = self.clay_fraction * (total_oxide_pct / 100) * 2.0

        # Organic matter enhances aggregation
        om_effect = min(self.organic_carbon_pct / 3.0, 0.25)

        # Kaolinite + oxide = very stable aggregates
        # Montmorillonite + oxide = less stable (swelling disrupts)
        mineralogy_factor = 0.7 + 0.3 * self.kaolinite_fraction

        factor = 1.0 + (oxide_effect + clay_oxide_effect +
                        om_effect) * mineralogy_factor
        return min(factor, 2.5)  # Cap at 2.5x

    def get_macropore_factor(self) -> float:
        """
        Calculate macropore enhancement factor.

        Tropical soils often have extensive macropore networks from:
        - Termite activity
        - Root channels
        - Wet-dry cracking
        - Biological activity

        Returns:
            Macropore factor (1.0 = normal, >1 = enhanced macropores)
        """
        base = 1.0

        # Termite contribution
        termite_factors = {
            'none': 0.0,
            'low': 0.1,
            'moderate': 0.25,
            'high': 0.45,
        }
        base += termite_factors.get(self.termite_activity, 0.1)

        # Root channels
        if self.root_channels:
            base += 0.15

        # Wet-dry cracking (especially for clayey soils)
        # But ONLY for montmorillonite-rich soils - kaolinite doesn't crack
        if self.wet_dry_cycles and self.clay_fraction > 0.25:
            montmorillonite_effect = (
                1 - self.kaolinite_fraction) * self.clay_fraction
            base += 0.3 * montmorillonite_effect  # Only non-kaolinite cracks

        return base

    def get_effective_clay_fraction(self) -> float:
        """
        Calculate effective clay fraction accounting for kaolinite mineralogy.

        CRITICAL INSIGHT: Standard PTFs assume montmorillonite (2:1) clay behavior.
        African tropical soils are dominated by kaolinite (1:1) which has:
        - ~20% of montmorillonite's water retention capacity per unit clay
        - ~10% of montmorillonite's CEC
        - No swelling/shrinking behavior

        A 50% clay ferralsol with 90% kaolinite behaves hydraulically like
        a ~15% clay temperate soil!

        Returns:
            Effective clay fraction for PTF purposes (0-1)
        """
        # Kaolinite contributes ~20% of montmorillonite's hydraulic effect
        # This is the key correction for African soils
        kaolinite_effect = self.clay_fraction * self.kaolinite_fraction * 0.20
        montmorillonite_effect = self.clay_fraction * \
            (1 - self.kaolinite_fraction) * 1.0

        effective_clay = kaolinite_effect + montmorillonite_effect
        return effective_clay

    def get_theta_r_correction_factor(self) -> float:
        """
        Calculate θr correction factor for tropical soils.

        Tropical soils have HIGHER θr than standard PTFs predict because:
        1. Fe/Al oxides adsorb water strongly on their surfaces
        2. Micropores within aggregates retain water at high tensions

        But kaolinite itself holds LESS water than montmorillonite.
        Net effect depends on oxide content.

        Returns:
            Multiplicative factor for θr (typically 0.8-1.5)
        """
        # Oxide effect: increases θr due to surface adsorption
        total_oxide = self.fe_oxide_pct + self.al_oxide_pct
        if total_oxide > 10:
            oxide_factor = 1.0 + 0.02 * (total_oxide - 10)
        else:
            oxide_factor = 1.0

        # Kaolinite effect: decreases θr (less water per unit clay)
        # But this is partially offset by micropore trapping
        kao_factor = 1.0 - 0.2 * self.kaolinite_fraction

        # Net factor
        return np.clip(oxide_factor * kao_factor, 0.6, 1.8)

    def correct_theta_sat(self, theta_sat_standard: float) -> float:
        """
        Correct saturated water content for tropical soils.

        Tropical soils often have higher porosity due to:
        - Oxide aggregation
        - Macropore networks
        - Biological activity

        Args:
            theta_sat_standard: θs from standard PTF

        Returns:
            Corrected θs
        """
        agg_factor = self.get_oxide_aggregation_factor()
        macro_factor = self.get_macropore_factor()

        # Combined increase (typically 5-25%)
        increase = (agg_factor - 1.0) * 0.1 + (macro_factor - 1.0) * 0.08

        corrected = theta_sat_standard * (1.0 + increase)
        return min(corrected, 0.65)  # Physical limit

    def correct_alpha(self, alpha_standard: float) -> float:
        """
        Correct Van Genuchten α for tropical soils.

        Increased macroporosity typically increases α (earlier
        desaturation), but oxide aggregation can decrease it.

        Args:
            alpha_standard: α from standard PTF (1/kPa)

        Returns:
            Corrected α
        """
        macro_factor = self.get_macropore_factor()
        agg_factor = self.get_oxide_aggregation_factor()

        # Macropores increase α (larger pores drain earlier)
        macro_effect = (macro_factor - 1.0) * 0.4

        # Aggregation decreases α (aggregates hold water in micropores)
        agg_effect = (agg_factor - 1.0) * -0.15

        correction = 1.0 + macro_effect + agg_effect
        return alpha_standard * correction

    def correct_n(self, n_standard: float) -> float:
        """
        Correct Van Genuchten n for tropical soils.

        Aggregation creates bimodal pore distribution,
        which typically broadens the retention curve (decreases n).

        Args:
            n_standard: n from standard PTF

        Returns:
            Corrected n
        """
        agg_factor = self.get_oxide_aggregation_factor()

        # Strong aggregation broadens curve
        if agg_factor > 1.3:
            reduction = 0.1 * (agg_factor - 1.0)
        else:
            reduction = 0.05 * (agg_factor - 1.0)

        corrected = n_standard * (1.0 - reduction)
        return max(corrected, 1.05)  # Physical minimum

    def correct_Ksat(self, Ksat_standard_mm_day: float) -> float:
        """
        Correct saturated hydraulic conductivity for tropical soils.

        Macropores dramatically increase Ksat in tropical soils.

        Args:
            Ksat_standard_mm_day: Ksat from standard PTF (mm/day)

        Returns:
            Corrected Ksat (mm/day)
        """
        macro_factor = self.get_macropore_factor()
        agg_factor = self.get_oxide_aggregation_factor()

        # Macropore effect is exponential on Ksat
        macro_multiplier = np.exp((macro_factor - 1.0) * 2.0)

        # Aggregation also increases Ksat
        agg_multiplier = agg_factor ** 0.5

        corrected = Ksat_standard_mm_day * macro_multiplier * agg_multiplier

        # Apply texture-based limits
        max_Ksat = 5000.0 if self.sand_fraction > 0.6 else 2000.0
        return min(corrected, max_Ksat)

    def apply_all_corrections(
        self,
        theta_sat: float,
        theta_res: float,
        alpha: float,
        n: float,
        Ksat_mm_day: float,
    ) -> Dict[str, float]:
        """
        Apply all tropical corrections to VG parameters.

        Args:
            theta_sat: Saturated water content
            theta_res: Residual water content
            alpha: VG α parameter (1/kPa)
            n: VG n parameter
            Ksat_mm_day: Saturated conductivity (mm/day)

        Returns:
            Dict of corrected parameters
        """
        theta_sat_corr = self.correct_theta_sat(theta_sat)
        alpha_corr = self.correct_alpha(alpha)
        n_corr = self.correct_n(n)
        Ksat_corr = self.correct_Ksat(Ksat_mm_day)

        # θr correction for tropical soils (NOT unchanged!)
        # Oxide surfaces retain water, but kaolinite holds less than montmorillonite
        theta_r_factor = self.get_theta_r_correction_factor()
        theta_res_corr = theta_res * theta_r_factor
        theta_res_corr = np.clip(theta_res_corr, 0.01, 0.30)

        return {
            'theta_sat': theta_sat_corr,
            'theta_res': theta_res_corr,
            'alpha': alpha_corr,
            'n': n_corr,
            'Ksat_mm_day': Ksat_corr,
            'aggregation_factor': self.get_oxide_aggregation_factor(),
            'macropore_factor': self.get_macropore_factor(),
            'effective_clay_fraction': self.get_effective_clay_fraction(),
            'theta_r_correction_factor': theta_r_factor,
        }


def estimate_macropore_flow_fraction(
    precip_intensity_mm_hr: float,
    Ksat_matrix_mm_hr: float,
    theta_current: float,
    theta_sat: float,
    macropore_factor: float = 1.0,
) -> float:
    """
    Estimate fraction of infiltration through macropores.

    When precipitation exceeds matrix infiltration capacity,
    water flows through macropores bypassing the matrix.

    Args:
        precip_intensity_mm_hr: Precipitation intensity
        Ksat_matrix_mm_hr: Matrix Ksat
        theta_current: Current water content
        theta_sat: Saturated water content
        macropore_factor: Macropore enhancement factor

    Returns:
        Fraction of flow through macropores (0-1)
    """
    # Matrix infiltration capacity decreases as soil wets
    saturation = theta_current / theta_sat
    matrix_capacity = Ksat_matrix_mm_hr * (1.0 - saturation ** 2)

    if precip_intensity_mm_hr <= matrix_capacity:
        return 0.0

    # Excess goes through macropores
    excess = precip_intensity_mm_hr - matrix_capacity
    macro_fraction = excess / precip_intensity_mm_hr

    # Macropore factor increases bypass potential
    macro_fraction *= min(macropore_factor, 2.0)

    return min(macro_fraction, 0.9)  # Max 90% bypass


def partition_infiltration(
    precip_mm: float,
    precip_duration_hr: float,
    Ksat_mm_day: float,
    theta_current: float,
    theta_sat: float,
    tropical_corrections: Optional[TropicalSoilCorrections] = None,
) -> Dict[str, float]:
    """
    Partition infiltration into matrix and macropore components.

    Args:
        precip_mm: Total precipitation (mm)
        precip_duration_hr: Duration (hours)
        Ksat_mm_day: Saturated conductivity
        theta_current: Current water content
        theta_sat: Saturated water content
        tropical_corrections: Tropical corrections (optional)

    Returns:
        Dict with matrix_mm, macropore_mm, runoff_mm
    """
    if precip_duration_hr <= 0 or precip_mm <= 0:
        return {
            'matrix_mm': 0.0,
            'macropore_mm': 0.0,
            'runoff_mm': 0.0,
        }

    # Precipitation intensity
    intensity_mm_hr = precip_mm / precip_duration_hr
    Ksat_mm_hr = Ksat_mm_day / 24.0

    # Get macropore factor
    if tropical_corrections:
        macro_factor = tropical_corrections.get_macropore_factor()
    else:
        macro_factor = 1.0

    # Macropore flow fraction
    macro_frac = estimate_macropore_flow_fraction(
        intensity_mm_hr, Ksat_mm_hr, theta_current, theta_sat, macro_factor
    )

    # Total infiltration capacity (matrix + macropores)
    saturation = theta_current / theta_sat
    matrix_capacity = Ksat_mm_hr * (1.0 - saturation ** 2)
    macro_capacity = macro_factor * Ksat_mm_hr * 2.0  # Macropores can handle more
    total_capacity = matrix_capacity + macro_capacity

    # Infiltration and runoff
    if intensity_mm_hr <= total_capacity:
        infiltration = precip_mm
        runoff = 0.0
    else:
        infiltration = total_capacity * precip_duration_hr
        runoff = precip_mm - infiltration

    # Partition infiltration
    matrix_mm = infiltration * (1.0 - macro_frac)
    macropore_mm = infiltration * macro_frac

    return {
        'matrix_mm': matrix_mm,
        'macropore_mm': macropore_mm,
        'runoff_mm': max(0.0, runoff),
        'macropore_fraction': macro_frac,
    }
