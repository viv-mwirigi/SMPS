"""
Improved Van Genuchten (iVG) formulation with a dry-end correction.

This module implements a PDI-style retention curve that blends a wet VG term
with a dry-end term using smooth weighting. The goal is to avoid unrealistic
behavior at very low water contents while keeping standard VG behavior near
field capacity and saturation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class IVGParams:
    """Parameters for the improved Van Genuchten (iVG) model."""

    theta_r: float
    theta_s: float
    alpha: float
    n: float
    K_sat: float

    # Dry-end behavior controls
    theta_dry: float = 0.01
    psi_dry_transition_kpa: float = 10000.0
    psi_air_dry_kpa: float = 1.0e6
    dry_exponent: float = 0.75
    dry_weight_exponent: float = 2.0

    @property
    def m(self) -> float:
        return 1.0 - 1.0 / self.n


def _se_wet(psi_kpa: float, params: IVGParams) -> float:
    if psi_kpa >= 0:
        return 1.0
    h = abs(psi_kpa)
    return (1.0 + (params.alpha * h) ** params.n) ** (-params.m)


def _se_dry(psi_kpa: float, params: IVGParams) -> float:
    if psi_kpa >= 0:
        return 1.0
    h = abs(psi_kpa)
    if h <= params.psi_dry_transition_kpa:
        return 1.0
    # Exponential decay in log-space toward air-dry conditions
    log_ratio = np.log1p(h / params.psi_dry_transition_kpa)
    se = np.exp(-params.dry_exponent * log_ratio)
    return float(np.clip(se, 0.0, 1.0))


def _dry_weight(psi_kpa: float, params: IVGParams) -> float:
    if psi_kpa >= 0:
        return 0.0
    h = abs(psi_kpa)
    ratio = h / params.psi_dry_transition_kpa
    w_wet = 1.0 / (1.0 + ratio ** params.dry_weight_exponent)
    return float(np.clip(1.0 - w_wet, 0.0, 1.0))


def effective_saturation(psi_kpa: float, params: IVGParams) -> float:
    """Compute iVG effective saturation Se(psi)."""
    se_wet = _se_wet(psi_kpa, params)
    se_dry = _se_dry(psi_kpa, params)
    w_dry = _dry_weight(psi_kpa, params)
    se = (w_dry * se_dry) + ((1.0 - w_dry) * se_wet)
    return float(np.clip(se, 0.0, 1.0))


def theta_from_psi(psi_kpa: float, params: IVGParams) -> float:
    """Convert matric potential (kPa) to volumetric water content (theta)."""
    se = effective_saturation(psi_kpa, params)
    theta = params.theta_dry + (params.theta_s - params.theta_dry) * se
    return float(np.clip(theta, params.theta_dry, params.theta_s))


def psi_from_theta(theta: float, params: IVGParams) -> float:
    """Invert iVG with a robust bisection solver."""
    theta_safe = float(np.clip(theta, params.theta_dry +
                       1e-8, params.theta_s - 1e-8))

    # Bisection bounds in kPa (negative suction range)
    lo = -abs(params.psi_air_dry_kpa)
    hi = -1.0e-6

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        theta_mid = theta_from_psi(mid, params)
        if theta_mid > theta_safe:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)


def hydraulic_conductivity_from_psi(psi_kpa: float, params: IVGParams) -> float:
    """Unsaturated hydraulic conductivity using Mualem with iVG Se."""
    se = effective_saturation(psi_kpa, params)
    if se <= 0.0:
        return 0.0
    if se >= 1.0:
        return params.K_sat
    m = params.m
    L = 0.5
    k = params.K_sat * (se ** L) * (1.0 - (1.0 - se ** (1.0 / m)) ** m) ** 2
    return float(max(0.0, k))


def theta_series_from_psi(psi_kpa: Iterable[float], params: IVGParams) -> np.ndarray:
    """Vectorized theta from psi for arrays."""
    return np.array([theta_from_psi(float(p), params) for p in psi_kpa], dtype=float)


def psi_series_from_theta(theta: Iterable[float], params: IVGParams) -> np.ndarray:
    """Vectorized psi from theta for arrays."""
    return np.array([psi_from_theta(float(t), params) for t in theta], dtype=float)


def ivg_from_vg(theta_r: float, theta_s: float, alpha: float, n: float, K_sat: float) -> IVGParams:
    """Helper to build iVG params from classic VG inputs."""
    theta_dry = max(0.0, min(0.5 * theta_r, theta_r - 0.005)
                    ) if theta_r > 0.01 else 0.005
    return IVGParams(
        theta_r=theta_r,
        theta_s=theta_s,
        alpha=alpha,
        n=n,
        K_sat=K_sat,
        theta_dry=theta_dry,
    )
