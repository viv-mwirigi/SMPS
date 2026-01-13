"""
Advanced soil hydraulic functions implementing rigorous soil physics.

This module provides the core hydraulic relationships needed for accurate
water balance modeling, including:

1. Van Genuchten water retention curve: θ(ψ) relationship
2. Van Genuchten-Mualem unsaturated hydraulic conductivity: K(θ)
3. Brooks-Corey relationships (alternative parameterization)
4. Matric potential calculations and conversions
5. Feddes root water uptake stress function

References:
- Van Genuchten, M.Th. (1980). A closed-form equation for predicting the
  hydraulic conductivity of unsaturated soils. Soil Sci. Soc. Am. J. 44:892-898.
- Mualem, Y. (1976). A new model for predicting the hydraulic conductivity
  of unsaturated porous media. Water Resources Research, 12(3):513-522.
- Feddes, R.A. et al. (1978). Simulation of field water use and crop yield.
  PUDOC, Wageningen, 189 pp.
- Brooks, R.H. and Corey, A.T. (1964). Hydraulic properties of porous media.
  Hydrology Paper No. 3, Colorado State University.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Numerical safety / sanity bounds
# =============================================================================

# Recommended Van Genuchten parameter ranges (units: alpha in 1/m)
VG_ALPHA_RECOMMENDED_RANGE_M_INV = (0.1, 10.0)  # ~0.001–0.1 cm^-1
VG_ALPHA_HARD_RANGE_M_INV = (1e-6, 1e4)
VG_N_MIN = 1.1
VG_N_HARD_RANGE = (1.01, 10.0)

# Prevent K(psi) underflowing to zero
K_MIN_RELATIVE_TO_KSAT = 1e-6

# Avoid exp/log overflow in float64
_LOG_MAX_FLOAT64 = 709.0

# Physical constants
WATER_DENSITY = 1000.0  # kg/m³
GRAVITY = 9.81  # m/s²
ATMOSPHERIC_PRESSURE = 101.325  # kPa

# Pressure head conversions
# 1 kPa = 0.102 m water column (h = P / (ρg))
KPA_TO_M_HEAD = 0.102
M_HEAD_TO_KPA = 1.0 / KPA_TO_M_HEAD  # ≈ 9.81

# Standard pressure heads (m of water, negative for suction)
PRESSURE_HEAD_SATURATION = 0.0  # ψ at saturation
PRESSURE_HEAD_FIELD_CAPACITY = -3.37  # -33 kPa ≈ -3.37 m
# -1500 kPa ≈ -153 m (permanent wilting point)
PRESSURE_HEAD_WILTING_POINT = -153.0
PRESSURE_HEAD_AIR_DRY = -10000.0  # ~-100 MPa (oven dry)


@dataclass
class VanGenuchtenParameters:
    """
    Van Genuchten model parameters for soil water retention curve.

    The Van Genuchten (1980) model is:
        Se = [1 + (α|ψ|)^n]^(-m)

    where:
        Se = (θ - θ_r) / (θ_s - θ_r)  [effective saturation]
        ψ = pressure head (m, negative for suction)
        α = inverse of air entry pressure (1/m)
        n = pore size distribution parameter (>1)
        m = 1 - 1/n (Mualem constraint)
        θ_r = residual water content
        θ_s = saturated water content (≈ porosity)

    Units:
        - alpha: 1/m (note: some sources use 1/cm)
        - n: dimensionless
        - theta_r, theta_s: m³/m³
        - K_sat: m/day
    """
    alpha: float  # 1/m - inverse air entry pressure
    n: float  # Pore size distribution parameter
    theta_r: float  # Residual water content (m³/m³)
    theta_s: float  # Saturated water content (m³/m³)
    K_sat: float  # Saturated hydraulic conductivity (m/day)
    L: float = 0.5  # Pore connectivity parameter (Mualem default)

    def __post_init__(self) -> None:
        # Basic finite/positive checks
        for name in ("alpha", "n", "theta_r", "theta_s", "K_sat", "L"):
            value = getattr(self, name)
            if value is None or not np.isfinite(value):
                raise ValueError(
                    f"VanGenuchtenParameters.{name} must be finite")

        # Clamp hard ranges first
        self.alpha = float(np.clip(self.alpha, *VG_ALPHA_HARD_RANGE_M_INV))
        self.n = float(np.clip(self.n, *VG_N_HARD_RANGE))
        self.K_sat = float(max(self.K_sat, 1e-12))
        self.L = float(np.clip(self.L, -5.0, 5.0))

        # Recommended-range warnings + enforcement
        if self.alpha < VG_ALPHA_RECOMMENDED_RANGE_M_INV[0] or self.alpha > VG_ALPHA_RECOMMENDED_RANGE_M_INV[1]:
            logger.warning(
                "VG alpha=%.4g 1/m outside recommended range [%.3g, %.3g] 1/m",
                self.alpha,
                VG_ALPHA_RECOMMENDED_RANGE_M_INV[0],
                VG_ALPHA_RECOMMENDED_RANGE_M_INV[1],
            )

        if self.n < VG_N_MIN:
            logger.warning(
                "VG n=%.4g < %.3g; clamping to %.3g for stability",
                self.n,
                VG_N_MIN,
                VG_N_MIN,
            )
            self.n = float(VG_N_MIN)

        # Water content sanity
        self.theta_s = float(np.clip(self.theta_s, 0.05, 0.80))
        self.theta_r = float(np.clip(self.theta_r, 0.0, self.theta_s - 1e-6))

        # Ensure usable storage
        if self.theta_s - self.theta_r < 1e-6:
            raise ValueError(
                f"Invalid VG parameters: theta_s ({self.theta_s}) must exceed theta_r ({self.theta_r})"
            )

    def with_multipliers(
        self,
        *,
        theta_s_adj: float = 1.0,
        Ks_adj: float = 1.0,
        alpha_adj: float = 1.0,
        n_adj: float = 1.0,
        theta_r_adj: float = 1.0,
    ) -> "VanGenuchtenParameters":
        """Return a copy with simple multiplicative adjustments applied."""
        return VanGenuchtenParameters(
            alpha=float(self.alpha) * float(alpha_adj),
            n=float(self.n) * float(n_adj),
            theta_r=float(self.theta_r) * float(theta_r_adj),
            theta_s=float(self.theta_s) * float(theta_s_adj),
            K_sat=float(self.K_sat) * float(Ks_adj),
            L=float(self.L),
        )

    @property
    def m(self) -> float:
        """Mualem constraint: m = 1 - 1/n"""
        return 1.0 - 1.0 / self.n

    @property
    def available_water_capacity(self) -> float:
        """θ_s - θ_r: total available water capacity"""
        return self.theta_s - self.theta_r

    @classmethod
    def from_texture(
        cls,
        sand_percent: float,
        clay_percent: float,
        organic_matter_percent: float = 2.0,
        bulk_density: Optional[float] = None
    ) -> "VanGenuchtenParameters":
        """
        Estimate Van Genuchten parameters from soil texture using
        Saxton & Rawls (2006) and Carsel & Parrish (1988).

        Args:
            sand_percent: Sand content (%)
            clay_percent: Clay content (%)
            organic_matter_percent: Organic matter (%)
            bulk_density: Bulk density (g/cm³), estimated if None

        Returns:
            VanGenuchtenParameters instance
        """
        om = organic_matter_percent / 100.0

        # Porosity (saturated water content) from Saxton & Rawls
        clay_for_log = max(1.0, clay_percent)
        theta_s = (
            0.332 -
            7.251e-4 * sand_percent +
            0.1276 * np.log10(clay_for_log) +
            0.002 * organic_matter_percent
        )
        theta_s = np.clip(theta_s, 0.30, 0.60)

        # Residual water content (approximately wilting point / 3)
        theta_r = (0.026 + 0.005 * clay_percent +
                   0.0158 * organic_matter_percent) / 3.0
        theta_r = np.clip(theta_r, 0.01, 0.15)

        # Saturated hydraulic conductivity
        log_k_sat = (
            0.6 +
            0.0126 * sand_percent -
            0.0064 * clay_percent -
            0.0127 * organic_matter_percent
        )
        K_sat_mm_hr = 10 ** log_k_sat
        K_sat_m_day = K_sat_mm_hr * 24 / 1000  # mm/hr → m/day
        K_sat_m_day = np.clip(K_sat_m_day, 0.001, 10.0)

        # Alpha parameter (1/m) from Saxton & Rawls
        log_alpha = (
            -0.784 +
            0.0178 * sand_percent -
            0.0112 * clay_percent -
            0.0074 * organic_matter_percent
        )
        alpha_cm = np.exp(log_alpha)
        alpha_m = alpha_cm * 100  # 1/cm → 1/m
        alpha_m = np.clip(alpha_m, 0.1, 50.0)

        # n parameter
        log_n = (
            -0.63 -
            0.0063 * sand_percent +
            0.0121 * clay_percent -
            0.0064 * organic_matter_percent
        )
        n = 1 + np.exp(log_n)
        n = np.clip(n, 1.05, 2.8)

        return cls(
            alpha=alpha_m,
            n=n,
            theta_r=theta_r,
            theta_s=theta_s,
            K_sat=K_sat_m_day
        )

    @classmethod
    def from_texture_class(cls, texture_class: str) -> "VanGenuchtenParameters":
        """
        Get Van Genuchten parameters for USDA texture class.

        Values from Carsel & Parrish (1988) - widely used defaults.

        Args:
            texture_class: USDA texture class name

        Returns:
            VanGenuchtenParameters for the texture class
        """
        # Carsel & Parrish (1988) parameters
        # NOTE: alpha values here are in 1/m (not 1/cm). Many published tables use 1/cm;
        # those would be ~100x smaller.
        # Format: (alpha [1/m], n, theta_r, theta_s, K_sat [cm/day])
        params = {
            'sand': (14.5, 2.68, 0.045, 0.43, 712.8),
            'loamy_sand': (12.4, 2.28, 0.057, 0.41, 350.2),
            'sandy_loam': (7.5, 1.89, 0.065, 0.41, 106.1),
            'loam': (3.6, 1.56, 0.078, 0.43, 24.96),
            'silt_loam': (2.0, 1.41, 0.067, 0.45, 10.80),
            'silt': (1.6, 1.37, 0.034, 0.46, 6.00),
            'sandy_clay_loam': (5.9, 1.48, 0.100, 0.39, 31.44),
            'clay_loam': (1.9, 1.31, 0.095, 0.41, 6.24),
            'silty_clay_loam': (1.0, 1.23, 0.089, 0.43, 1.68),
            'sandy_clay': (2.7, 1.23, 0.100, 0.38, 2.88),
            'silty_clay': (0.5, 1.09, 0.070, 0.36, 0.48),
            'clay': (0.8, 1.09, 0.068, 0.38, 4.80),
        }

        # Normalize texture class name
        texture_key = texture_class.lower().replace(' ', '_').replace('-', '_')

        if texture_key not in params:
            logger.warning(
                f"Unknown texture class '{texture_class}', using 'loam'")
            texture_key = 'loam'

        alpha_cm, n, theta_r, theta_s, K_sat_cm_day = params[texture_key]

        return cls(
            alpha=alpha_cm,  # already 1/m
            n=n,
            theta_r=theta_r,
            theta_s=theta_s,
            K_sat=K_sat_cm_day / 100  # cm/day → m/day
        )

    @classmethod
    def for_depth(
        cls,
        surface_params: "VanGenuchtenParameters",
        depth_m: float,
        horizon_type: str = "Bt"
    ) -> "VanGenuchtenParameters":
        """
        Calculate depth-dependent Van Genuchten parameters.

        Accounts for pedogenic processes that modify soil properties with depth:
        - Clay illuviation (Bt horizon) increases clay content
        - Bulk density increases with depth (compaction)
        - Organic matter decreases with depth
        - Macroporosity decreases with depth

        Based on research from Vereecken et al. (2010), Schaap et al. (2001),
        and field observations from USDA-NRCS soil surveys.

        Args:
            surface_params: Van Genuchten parameters for surface soil
            depth_m: Depth from surface (m)
            horizon_type: Soil horizon type ("A", "E", "Bt", "C")
                         Bt = clay-enriched B horizon (most common in agricultural soils)

        Returns:
            Depth-adjusted VanGenuchtenParameters
        """
        # Depth adjustment factors based on typical soil profiles
        # These factors represent changes relative to surface soil

        if depth_m <= 0.10:
            # Surface layer - use surface params directly
            return surface_params

        elif depth_m <= 0.30:
            # Upper root zone (10-30 cm)
            # Slight changes from surface
            alpha_factor = 0.90  # Slight reduction in alpha (smaller pores)
            n_factor = 0.98  # n nearly unchanged
            theta_s_factor = 0.97  # Slight porosity reduction
            theta_r_factor = 1.05  # Slight increase in residual water
            K_sat_factor = 0.70  # K_sat decreases with depth

        elif depth_m <= 0.50:
            # Lower root zone (30-50 cm) - Bt horizon effects begin
            if horizon_type == "Bt":
                # Clay illuviation zone
                # Significant alpha reduction (finer pores)
                alpha_factor = 0.70
                n_factor = 0.92  # n decreases (broader pore distribution)
                theta_s_factor = 0.92  # Porosity decreases
                theta_r_factor = 1.20  # Higher residual due to more clay
                K_sat_factor = 0.30  # Significant K_sat reduction
            else:
                alpha_factor = 0.80
                n_factor = 0.95
                theta_s_factor = 0.95
                theta_r_factor = 1.10
                K_sat_factor = 0.50

        elif depth_m <= 0.75:
            # Transition zone (50-75 cm) - maximum Bt effect
            if horizon_type == "Bt":
                alpha_factor = 0.55  # Strong alpha reduction
                n_factor = 0.88  # Broader pore distribution
                theta_s_factor = 0.88  # Lower porosity
                theta_r_factor = 1.35  # Higher residual (clay)
                K_sat_factor = 0.15  # Much lower K_sat
            else:
                alpha_factor = 0.70
                n_factor = 0.92
                theta_s_factor = 0.92
                theta_r_factor = 1.15
                K_sat_factor = 0.35

        else:
            # Deep zone (>75 cm) - approaching C horizon or continued Bt
            if horizon_type == "Bt":
                alpha_factor = 0.50  # Very fine pores
                n_factor = 0.85  # Broadest pore distribution
                theta_s_factor = 0.85  # Lowest porosity
                theta_r_factor = 1.40  # Highest residual
                K_sat_factor = 0.10  # Very low K_sat
            elif horizon_type == "C":
                # C horizon - parent material, often sandier
                alpha_factor = 1.20  # Larger pores possible
                n_factor = 1.05
                theta_s_factor = 0.90
                theta_r_factor = 0.90
                K_sat_factor = 0.50
            else:
                alpha_factor = 0.60
                n_factor = 0.90
                theta_s_factor = 0.90
                theta_r_factor = 1.20
                K_sat_factor = 0.25

        # Apply factors with bounds checking
        new_alpha = surface_params.alpha * alpha_factor
        new_n = max(1.05, surface_params.n * n_factor)  # n must be > 1
        new_theta_s = np.clip(surface_params.theta_s *
                              theta_s_factor, 0.25, 0.55)
        new_theta_r = np.clip(surface_params.theta_r *
                              theta_r_factor, 0.01, 0.20)
        new_K_sat = max(0.0001, surface_params.K_sat *
                        K_sat_factor)  # m/day, minimum value

        # Ensure theta_s > theta_r
        if new_theta_s <= new_theta_r + 0.05:
            new_theta_r = new_theta_s - 0.10
            new_theta_r = max(0.01, new_theta_r)

        return cls(
            alpha=new_alpha,
            n=new_n,
            theta_r=new_theta_r,
            theta_s=new_theta_s,
            K_sat=new_K_sat,
            L=surface_params.L
        )

    @classmethod
    def create_depth_profile(
        cls,
        sand_percent: float,
        clay_percent: float,
        layer_depths_m: List[float],
        organic_matter_surface: float = 3.0,
        horizon_type: str = "Bt"
    ) -> List["VanGenuchtenParameters"]:
        """
        Create a complete soil hydraulic profile with depth-dependent parameters.

        This method creates VG parameters for multiple layers, accounting for
        typical pedogenic changes with depth.

        Args:
            sand_percent: Sand content at surface (%)
            clay_percent: Clay content at surface (%)
            layer_depths_m: List of layer thicknesses (m)
            organic_matter_surface: OM at surface (%)
            horizon_type: Dominant subsurface horizon type

        Returns:
            List of VanGenuchtenParameters, one per layer
        """
        # Create surface layer parameters
        surface_vg = cls.from_texture(
            sand_percent, clay_percent, organic_matter_surface
        )

        vg_params_list = []
        cumulative_depth = 0.0

        for thickness in layer_depths_m:
            layer_center_depth = cumulative_depth + thickness / 2

            # Get depth-adjusted parameters
            layer_vg = cls.for_depth(
                surface_vg, layer_center_depth, horizon_type)
            vg_params_list.append(layer_vg)

            cumulative_depth += thickness

        return vg_params_list


@dataclass
class BrooksCoreyParameters:
    """
    Brooks-Corey model parameters for soil water retention.

    The Brooks-Corey (1964) model is:
        Se = (ψ_b / ψ)^λ  for ψ < ψ_b
        Se = 1            for ψ >= ψ_b

    where:
        Se = effective saturation
        ψ_b = air entry pressure head (bubbling pressure, m)
        λ = pore size distribution index
    """
    psi_b: float  # Air entry pressure head (m, positive value)
    lambda_bc: float  # Pore size distribution index
    theta_r: float  # Residual water content
    theta_s: float  # Saturated water content
    K_sat: float  # Saturated hydraulic conductivity (m/day)

    @classmethod
    def from_van_genuchten(cls, vg: VanGenuchtenParameters) -> "BrooksCoreyParameters":
        """
        Convert Van Genuchten parameters to Brooks-Corey equivalent.

        Using the relationships from Lenhard et al. (1989).
        """
        # Approximate conversion
        psi_b = 1.0 / vg.alpha  # Air entry ≈ 1/α
        lambda_bc = vg.n - 1.0  # λ ≈ n - 1

        return cls(
            psi_b=psi_b,
            lambda_bc=lambda_bc,
            theta_r=vg.theta_r,
            theta_s=vg.theta_s,
            K_sat=vg.K_sat
        )


# =============================================================================
# VAN GENUCHTEN WATER RETENTION FUNCTIONS
# =============================================================================

def van_genuchten_theta_from_psi(
    psi: float,
    params: VanGenuchtenParameters
) -> float:
    """
    Calculate water content from pressure head using Van Genuchten model.

    θ(ψ) = θ_r + (θ_s - θ_r) / [1 + (α|ψ|)^n]^m

    Args:
        psi: Pressure head (m, negative for suction)
        params: Van Genuchten parameters

    Returns:
        Volumetric water content (m³/m³)
    """
    if psi >= 0:
        # Saturated or positive pressure
        return params.theta_s

    # Effective saturation, computed in a log-safe way.
    x = float(params.alpha) * float(abs(psi))
    if x <= 0.0:
        Se = 1.0
    else:
        log_x = float(np.log(x))
        log_alpha_psi_n = float(params.n) * log_x
        if log_alpha_psi_n >= _LOG_MAX_FLOAT64:
            alpha_psi_n = float("inf")
        else:
            alpha_psi_n = float(np.exp(log_alpha_psi_n))
        # For very large alpha_psi_n, Se -> 0; avoid warnings from inf arithmetic.
        if not np.isfinite(alpha_psi_n):
            Se = 0.0
        else:
            Se = float((1.0 + alpha_psi_n) ** (-params.m))

    # Convert to water content
    theta = params.theta_r + (params.theta_s - params.theta_r) * Se

    return np.clip(theta, params.theta_r, params.theta_s)


def van_genuchten_psi_from_theta(
    theta: float,
    params: VanGenuchtenParameters
) -> float:
    """
    Calculate pressure head from water content (inverse Van Genuchten).

    ψ(θ) = -(1/α) * [Se^(-1/m) - 1]^(1/n)

    Args:
        theta: Volumetric water content (m³/m³)
        params: Van Genuchten parameters

    Returns:
        Pressure head (m, negative for suction)
    """
    # Clamp theta to valid range
    theta = np.clip(theta, params.theta_r * 1.001, params.theta_s * 0.999)

    # Effective saturation
    Se = (theta - params.theta_r) / (params.theta_s - params.theta_r)
    Se = np.clip(Se, 1e-10, 1.0 - 1e-10)

    # Inverse VG relationship
    # Se = [1 + (α|ψ|)^n]^(-m)
    # => Se^(-1/m) = 1 + (α|ψ|)^n
    # => (α|ψ|)^n = Se^(-1/m) - 1
    # => |ψ| = (1/α) * [Se^(-1/m) - 1]^(1/n)

    # log-safe computation of Se^(-1/m)
    log_Se = float(np.log(Se))
    log_Se_pow = (-1.0 / float(params.m)) * log_Se
    if log_Se_pow >= _LOG_MAX_FLOAT64:
        Se_pow = float("inf")
    else:
        Se_pow = float(np.exp(log_Se_pow))

    term = Se_pow - 1.0

    if term <= 0:
        return 0.0  # At or above saturation

    psi_abs = (1.0 / params.alpha) * (term ** (1.0 / params.n))

    return -psi_abs  # Negative for suction


def van_genuchten_mualem_K(
    theta: float,
    params: VanGenuchtenParameters
) -> float:
    """
    Calculate unsaturated hydraulic conductivity using Van Genuchten-Mualem model.

    K(θ) = K_sat * Se^L * [1 - (1 - Se^(1/m))^m]²

    This is the widely-used closed-form solution combining Van Genuchten
    retention with Mualem's pore connectivity model.

    Args:
        theta: Volumetric water content (m³/m³)
        params: Van Genuchten parameters

    Returns:
        Unsaturated hydraulic conductivity (m/day)
    """
    # Clamp theta
    theta = np.clip(theta, params.theta_r * 1.001, params.theta_s)

    # Effective saturation
    Se = (theta - params.theta_r) / (params.theta_s - params.theta_r)
    Se = np.clip(Se, 1e-10, 1.0)

    if Se >= 1.0 - 1e-10:
        return params.K_sat

    # Mualem-Van Genuchten equation
    # K(Se) = K_sat * Se^L * [1 - (1 - Se^(1/m))^m]²

    Se_inv_m = Se ** (1.0 / params.m)
    inner_term = 1.0 - Se_inv_m

    if inner_term <= 0:
        return params.K_sat

    outer_term = 1.0 - (inner_term ** params.m)

    K_rel = (Se ** params.L) * (outer_term ** 2)

    K_unsat = float(params.K_sat) * float(np.clip(K_rel, 0.0, 1.0))

    # Prevent underflow to (near) zero which can destabilize numerical fluxes.
    K_min = float(params.K_sat) * float(K_MIN_RELATIVE_TO_KSAT)
    return float(max(K_unsat, K_min))


def van_genuchten_mualem_K_from_psi(
    psi: float,
    params: VanGenuchtenParameters
) -> float:
    """
    Calculate unsaturated K directly from pressure head.

    Args:
        psi: Pressure head (m, negative)
        params: Van Genuchten parameters

    Returns:
        Unsaturated hydraulic conductivity (m/day)
    """
    theta = van_genuchten_theta_from_psi(psi, params)
    return van_genuchten_mualem_K(theta, params)


# =============================================================================
# BROOKS-COREY HYDRAULIC FUNCTIONS
# =============================================================================

def brooks_corey_theta_from_psi(
    psi: float,
    params: BrooksCoreyParameters
) -> float:
    """
    Calculate water content from pressure head using Brooks-Corey model.

    Args:
        psi: Pressure head (m, negative for suction)
        params: Brooks-Corey parameters

    Returns:
        Volumetric water content (m³/m³)
    """
    if psi >= -params.psi_b:
        return params.theta_s

    Se = (params.psi_b / abs(psi)) ** params.lambda_bc
    theta = params.theta_r + (params.theta_s - params.theta_r) * Se

    return np.clip(theta, params.theta_r, params.theta_s)


def brooks_corey_K(
    theta: float,
    params: BrooksCoreyParameters
) -> float:
    """
    Calculate unsaturated K using Brooks-Corey model.

    K(Se) = K_sat * Se^(3 + 2/λ)

    Args:
        theta: Volumetric water content (m³/m³)
        params: Brooks-Corey parameters

    Returns:
        Unsaturated hydraulic conductivity (m/day)
    """
    Se = (theta - params.theta_r) / (params.theta_s - params.theta_r)
    Se = np.clip(Se, 1e-10, 1.0)

    exponent = 3.0 + 2.0 / params.lambda_bc
    K_rel = Se ** exponent

    return params.K_sat * K_rel


# =============================================================================
# SPECIFIC WATER CAPACITY (dθ/dψ)
# =============================================================================

def specific_water_capacity(
    psi: float,
    params: VanGenuchtenParameters
) -> float:
    """
    Calculate specific water capacity C(ψ) = dθ/dψ.

    This is the slope of the water retention curve, important for
    Richards' equation solutions.

    C(ψ) = α*m*n*(θ_s - θ_r) * (α|ψ|)^(n-1) / [1 + (α|ψ|)^n]^(m+1)

    Args:
        psi: Pressure head (m, negative)
        params: Van Genuchten parameters

    Returns:
        Specific water capacity (1/m)
    """
    if psi >= 0:
        return 0.0  # No change at saturation

    alpha_psi = float(params.alpha) * float(abs(psi))
    if alpha_psi <= 0.0:
        return 0.0

    log_alpha_psi = float(np.log(alpha_psi))
    log_alpha_psi_n = float(params.n) * log_alpha_psi
    if log_alpha_psi_n >= _LOG_MAX_FLOAT64:
        alpha_psi_n = float("inf")
    else:
        alpha_psi_n = float(np.exp(log_alpha_psi_n))

    # alpha_psi^(n-1) computed log-safe
    log_alpha_psi_n_minus_1 = float(params.n - 1.0) * log_alpha_psi
    if log_alpha_psi_n_minus_1 >= _LOG_MAX_FLOAT64:
        alpha_psi_n_minus_1 = float("inf")
    else:
        alpha_psi_n_minus_1 = float(np.exp(log_alpha_psi_n_minus_1))

    numerator = (
        float(params.alpha) * float(params.m) * float(params.n) *
        float(params.theta_s - params.theta_r) *
        float(alpha_psi_n_minus_1)
    )

    if not np.isfinite(alpha_psi_n):
        return 0.0

    denominator = float((1.0 + alpha_psi_n) ** (float(params.m) + 1.0))

    if denominator <= 0.0 or not np.isfinite(denominator) or not np.isfinite(numerator):
        return 0.0

    return float(numerator / denominator)


# =============================================================================
# FEDDES ROOT WATER UPTAKE STRESS FUNCTION
# =============================================================================

@dataclass
class FeddesParameters:
    """
    Feddes root water uptake parameters.

    The Feddes function defines plant water uptake as a function of
    pressure head with four critical values:

    α(ψ) = 0                          for ψ > ψ_1 or ψ < ψ_4
    α(ψ) = (ψ - ψ_1)/(ψ_2 - ψ_1)     for ψ_1 > ψ > ψ_2
    α(ψ) = 1                          for ψ_2 >= ψ >= ψ_3
    α(ψ) = (ψ - ψ_4)/(ψ_3 - ψ_4)     for ψ_3 > ψ > ψ_4

    Parameters (all in m pressure head):
        ψ_1: Anaerobiosis point (too wet, ~-0.1 to -0.5 m)
        ψ_2: Reduction point high (start of optimal, ~-0.25 to -1.0 m)
        ψ_3h: Reduction point low, high transpiration (~-3 to -6 m)
        ψ_3l: Reduction point low, low transpiration (~-6 to -15 m)
        ψ_4: Wilting point (~-80 to -160 m, i.e., -800 to -1600 kPa)

    The ψ_3 value depends on transpiration rate:
        ψ_3 = ψ_3h for T_pot > T_high
        ψ_3 = ψ_3l for T_pot < T_low
        Linear interpolation between
    """
    psi_1: float = -0.1  # Anaerobiosis point (m)
    psi_2: float = -0.25  # Optimal upper (m)
    psi_3h: float = -3.0  # Stress onset, high T (m)
    psi_3l: float = -6.0  # Stress onset, low T (m)
    psi_4: float = -160.0  # Wilting point (m)
    T_high: float = 5.0  # High transpiration rate (mm/day)
    T_low: float = 1.0  # Low transpiration rate (mm/day)

    @classmethod
    def for_crop(cls, crop_type: str) -> "FeddesParameters":
        """
        Get Feddes parameters for common crop types.

        Values from Wesseling et al. (1991) and Taylor & Ashcroft (1972).

        Args:
            crop_type: Crop name (e.g., 'maize', 'wheat', 'grassland')

        Returns:
            FeddesParameters for the crop
        """
        # Format: (psi_1, psi_2, psi_3h, psi_3l, psi_4)
        # All values in meters
        crop_params = {
            'maize': (-0.15, -0.30, -6.0, -10.0, -160.0),
            'wheat': (-0.10, -0.25, -5.0, -9.0, -160.0),
            'rice': (0.0, -0.10, -2.0, -3.0, -80.0),  # Rice tolerates flooding
            'soybean': (-0.10, -0.25, -4.0, -8.0, -160.0),
            'cotton': (-0.15, -0.30, -8.0, -12.0, -160.0),
            'grassland': (-0.10, -0.25, -5.0, -8.0, -80.0),
            'potato': (-0.10, -0.25, -3.0, -5.0, -160.0),
            # Drought tolerant
            'cassava': (-0.15, -0.30, -10.0, -15.0, -160.0),
            'generic': (-0.10, -0.25, -5.0, -9.0, -160.0),
        }

        crop_key = crop_type.lower().replace(' ', '_')

        if crop_key not in crop_params:
            logger.warning(f"Unknown crop '{crop_type}', using 'generic'")
            crop_key = 'generic'

        psi_1, psi_2, psi_3h, psi_3l, psi_4 = crop_params[crop_key]

        return cls(psi_1=psi_1, psi_2=psi_2, psi_3h=psi_3h, psi_3l=psi_3l, psi_4=psi_4)


def feddes_stress_factor(
    psi: float,
    params: FeddesParameters,
    T_pot: float = 5.0
) -> float:
    """
    Calculate Feddes root water uptake stress factor.

    The stress factor α ranges from 0 (no uptake) to 1 (optimal uptake).

    Args:
        psi: Pressure head (m, negative for suction)
        params: Feddes parameters
        T_pot: Potential transpiration rate (mm/day)

    Returns:
        Stress factor α (0-1)
    """
    # Interpolate ψ_3 based on potential transpiration
    if T_pot >= params.T_high:
        psi_3 = params.psi_3h
    elif T_pot <= params.T_low:
        psi_3 = params.psi_3l
    else:
        # Linear interpolation
        t_frac = (T_pot - params.T_low) / (params.T_high - params.T_low)
        psi_3 = params.psi_3l + t_frac * (params.psi_3h - params.psi_3l)

    # Calculate stress factor
    if psi > params.psi_1:
        # Too wet (anaerobic)
        return 0.0
    elif psi > params.psi_2:
        # Linear ramp from anaerobiosis to optimal
        return (psi - params.psi_1) / (params.psi_2 - params.psi_1)
    elif psi >= psi_3:
        # Optimal zone
        return 1.0
    elif psi > params.psi_4:
        # Stress zone - linear decline
        return (psi - params.psi_4) / (psi_3 - params.psi_4)
    else:
        # Below wilting point
        return 0.0


def s_shape_stress_factor(
    psi: float,
    psi_50: float = -10.0,
    p: float = 3.0
) -> float:
    """
    S-shaped (van Genuchten-type) root water uptake stress function.

    α(ψ) = 1 / [1 + (ψ/ψ_50)^p]

    This smooth function is often more stable numerically than Feddes.

    Args:
        psi: Pressure head (m, negative)
        psi_50: Pressure head at 50% reduction (m)
        p: Exponent controlling curve steepness

    Returns:
        Stress factor (0-1)
    """
    if psi >= 0:
        return 1.0

    ratio = abs(psi) / abs(psi_50)
    return 1.0 / (1.0 + ratio ** p)


# =============================================================================
# HYDRAULIC HEAD AND DARCY FLUX CALCULATIONS
# =============================================================================

def hydraulic_head(
    psi: float,
    z: float
) -> float:
    """
    Calculate total hydraulic head.

    H = ψ + z

    where:
        ψ = pressure head (m, negative for suction)
        z = elevation head (m, positive upward from reference)
        H = total hydraulic head (m)

    Args:
        psi: Pressure head (m)
        z: Elevation (m)

    Returns:
        Total hydraulic head (m)
    """
    return psi + z


def darcy_flux(
    K: float,
    dH_dz: float
) -> float:
    """
    Calculate Darcy flux.

    q_down = K * dH/dz

    Sign convention used in SMPS: positive flux is downward.

    Note: In many hydrology texts, Darcy flux is defined positive upward,
    which would introduce a leading minus sign. Here we use a downward-positive
    convention to match the rest of the codebase (e.g. `vertical_flux`).

    Args:
        K: Hydraulic conductivity (m/day)
        dH_dz: Hydraulic head gradient (m/m)

    Returns:
        Darcy flux (m/day, positive = downward)
    """
    return K * dH_dz


def darcy_flux_between_layers(
    theta_upper: float,
    theta_lower: float,
    z_upper: float,
    z_lower: float,
    params_upper: VanGenuchtenParameters,
    params_lower: VanGenuchtenParameters,
    use_geometric_mean_K: bool = True
) -> float:
    """
    Calculate Darcy flux between two soil layers.

    Args:
        theta_upper: Water content in upper layer (m³/m³)
        theta_lower: Water content in lower layer (m³/m³)
        z_upper: Elevation of upper layer center (m)
        z_lower: Elevation of lower layer center (m)
        params_upper: VG parameters for upper layer
        params_lower: VG parameters for lower layer
        use_geometric_mean_K: Use geometric mean of K values

    Returns:
        Flux between layers (m/day, positive = downward)
    """
    # Calculate pressure heads
    psi_upper = van_genuchten_psi_from_theta(theta_upper, params_upper)
    psi_lower = van_genuchten_psi_from_theta(theta_lower, params_lower)

    # Total hydraulic heads
    H_upper = hydraulic_head(psi_upper, z_upper)
    H_lower = hydraulic_head(psi_lower, z_lower)

    # Distance between layer centers
    dz = z_upper - z_lower  # Positive

    if dz <= 0:
        return 0.0

    # Hydraulic gradient (dH/dz)
    dH_dz = (H_upper - H_lower) / dz

    # Get unsaturated K for both layers
    K_upper = van_genuchten_mualem_K(theta_upper, params_upper)
    K_lower = van_genuchten_mualem_K(theta_lower, params_lower)

    # Inter-layer conductivity
    if use_geometric_mean_K:
        K_inter = np.sqrt(K_upper * K_lower)
    else:
        # Harmonic mean (more conservative)
        K_inter = 2 * K_upper * K_lower / (K_upper + K_lower + 1e-15)

    # Darcy flux (positive = downward)
    q = darcy_flux(K_inter, dH_dz)

    return q


# =============================================================================
# HYSTERESIS FUNCTIONS
# =============================================================================

class HysteresisState(Enum):
    """Current state of soil wetting/drying"""
    WETTING = "wetting"
    DRYING = "drying"
    UNKNOWN = "unknown"


@dataclass
class HysteresisParameters:
    """
    Parameters for hysteretic water retention.

    Simple approach using different α values for wetting/drying.
    Based on Kool & Parker (1987) scaling approach.

    Typically: α_wet ≈ 2 × α_dry (wetting curve is shifted)
    """
    alpha_drying: float  # α for main drying curve (1/m)
    alpha_wetting: float  # α for main wetting curve (1/m)
    n: float  # n is usually assumed the same
    theta_r: float
    theta_s: float
    K_sat: float

    @classmethod
    def from_vg_params(
        cls,
        params: VanGenuchtenParameters,
        alpha_ratio: float = 2.0
    ) -> "HysteresisParameters":
        """
        Create hysteresis parameters from standard VG params.

        Args:
            params: Base VG parameters (assumed to be drying)
            alpha_ratio: Ratio of α_wetting to α_drying

        Returns:
            HysteresisParameters
        """
        return cls(
            alpha_drying=params.alpha,
            alpha_wetting=params.alpha * alpha_ratio,
            n=params.n,
            theta_r=params.theta_r,
            theta_s=params.theta_s,
            K_sat=params.K_sat
        )


def get_effective_alpha(
    current_theta: float,
    previous_theta: float,
    params: HysteresisParameters,
    current_state: HysteresisState = HysteresisState.UNKNOWN
) -> Tuple[float, HysteresisState]:
    """
    Determine effective α value based on wetting/drying state.

    Args:
        current_theta: Current water content
        previous_theta: Previous timestep water content
        params: Hysteresis parameters
        current_state: Current hysteresis state

    Returns:
        Tuple of (effective_alpha, new_state)
    """
    theta_change = current_theta - previous_theta

    if abs(theta_change) < 1e-8:
        # No significant change - maintain state
        if current_state == HysteresisState.WETTING:
            return params.alpha_wetting, current_state
        else:
            return params.alpha_drying, current_state

    if theta_change > 0:
        return params.alpha_wetting, HysteresisState.WETTING
    else:
        return params.alpha_drying, HysteresisState.DRYING


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def field_capacity_psi() -> float:
    """Standard field capacity pressure head (m)"""
    return PRESSURE_HEAD_FIELD_CAPACITY


def wilting_point_psi() -> float:
    """Standard permanent wilting point pressure head (m)"""
    return PRESSURE_HEAD_WILTING_POINT


def theta_at_field_capacity(params: VanGenuchtenParameters) -> float:
    """Calculate water content at field capacity (-33 kPa)"""
    return van_genuchten_theta_from_psi(PRESSURE_HEAD_FIELD_CAPACITY, params)


def theta_at_wilting_point(params: VanGenuchtenParameters) -> float:
    """Calculate water content at permanent wilting point (-1500 kPa)"""
    return van_genuchten_theta_from_psi(PRESSURE_HEAD_WILTING_POINT, params)


def plant_available_water(params: VanGenuchtenParameters) -> float:
    """
    Calculate plant available water capacity (m³/m³).

    PAW = θ_FC - θ_WP
    """
    theta_fc = theta_at_field_capacity(params)
    theta_wp = theta_at_wilting_point(params)
    return theta_fc - theta_wp


def relative_saturation(
    theta: float,
    params: VanGenuchtenParameters
) -> float:
    """
    Calculate relative saturation (actual saturation degree).

    S = θ / θ_s

    Note: This is different from effective saturation Se.
    """
    return theta / params.theta_s


def effective_saturation(
    theta: float,
    params: VanGenuchtenParameters
) -> float:
    """
    Calculate effective saturation.

    Se = (θ - θ_r) / (θ_s - θ_r)
    """
    Se = (theta - params.theta_r) / (params.theta_s - params.theta_r)
    return np.clip(Se, 0.0, 1.0)
