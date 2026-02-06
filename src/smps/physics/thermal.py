"""Soil thermal conductivity calculations (Johansen model)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ThermalParams:
    """Thermal parameters for Johansen conductivity model."""

    bulk_density_g_cm3: float
    theta_s: float
    theta_r: float
    quartz_fraction: float = 0.6


def thermal_conductivity_johansen(theta: float, params: ThermalParams) -> float:
    """Compute soil thermal conductivity (lambda) in W/m/K.

    Johansen model:
    lambda = lambda_dry + Sr * (lambda_sat - lambda_dry)
    """
    # Constants
    lambda_water = 0.57
    lambda_quartz = 7.7
    lambda_other = 2.0

    rho_b = params.bulk_density_g_cm3 * 1000.0
    porosity = float(np.clip(params.theta_s, 0.05, 0.8))

    # Solid thermal conductivity from quartz fraction
    quartz_frac = float(np.clip(params.quartz_fraction, 0.0, 1.0))
    lambda_solid = (lambda_quartz ** quartz_frac) * \
        (lambda_other ** (1.0 - quartz_frac))

    # Saturated conductivity (geometric mean)
    lambda_sat = (lambda_solid ** (1.0 - porosity)) * \
        (lambda_water ** porosity)

    # Dry conductivity (Johansen empirical)
    lambda_dry = (0.135 * rho_b + 64.7) / (2700.0 - 0.947 * rho_b)

    # Degree of saturation
    denom = max(params.theta_s - params.theta_r, 1e-8)
    Sr = float(np.clip((theta - params.theta_r) / denom, 0.0, 1.0))

    return float(lambda_dry + Sr * (lambda_sat - lambda_dry))
