"""Space adapters between θ-space (volumetric water content) and ψ-space (matric potential).

Why this exists
---------------
In this codebase there are multiple places that need to convert between:
- θ: volumetric water content (m³/m³)
- ψ: matric potential (kPa, negative for suction)

Conversions are physically defined by a soil water retention curve (Van Genuchten).
However, the inverse θ→ψ is numerically ill-conditioned near saturation (θ→θs)
where tiny θ errors map to large ψ swings.

This module provides a single, explicit, numerically-safe conversion surface
for use at training boundaries and deployment boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import numpy as np

from smps.core.types import VanGenuchtenParams
from smps.physics.van_genuchten import potential_from_water_content, water_content_from_potential


@dataclass(frozen=True)
class SpaceClips:
    """Recommended safety clips for irrigation decision use-cases."""

    # θ clips are always applied before inversion to avoid division by zero.
    theta_eps: float = 1e-6

    # ψ is typically only meaningful for crops in about [-2000, 0] kPa.
    # Keeping a wider range is fine for QC/plots; tighter is fine for decisions.
    psi_min_kpa: float = -2000.0
    psi_max_kpa: float = -0.1


def clip_theta(theta: float | np.ndarray, params: VanGenuchtenParams, *, eps: float = 1e-6):
    """Clip θ into the open interval (θr, θs) for stable inversion."""
    return np.clip(theta, params.theta_r + eps, params.theta_s - eps)


def theta_to_psi(
    theta: float | np.ndarray,
    params: VanGenuchtenParams,
    *,
    clips: Optional[SpaceClips] = SpaceClips(),
    clip_output: bool = True,
) -> float | np.ndarray:
    """Convert θ → ψ (kPa).

    Notes:
    - Applies θ clipping before inversion.
    - Optionally clips ψ into a decision-safe range.
    """
    theta_safe = clip_theta(theta, params, eps=(clips.theta_eps if clips else 1e-6))

    if np.isscalar(theta_safe):
        psi = potential_from_water_content(float(theta_safe), params)
        if clip_output and clips is not None:
            psi = float(np.clip(psi, clips.psi_min_kpa, clips.psi_max_kpa))
        return psi

    psi_arr = np.array([potential_from_water_content(float(t), params) for t in np.asarray(theta_safe, dtype=float)])
    if clip_output and clips is not None:
        psi_arr = np.clip(psi_arr, clips.psi_min_kpa, clips.psi_max_kpa)
    return psi_arr


def psi_to_theta(
    psi_kpa: float | np.ndarray,
    params: VanGenuchtenParams,
    *,
    clip_output: bool = True,
) -> float | np.ndarray:
    """Convert ψ (kPa) → θ (m³/m³)."""

    if np.isscalar(psi_kpa):
        theta = water_content_from_potential(float(psi_kpa), params)
        if clip_output:
            theta = float(np.clip(theta, 0.0, params.theta_s))
        return theta

    theta_arr = np.array([water_content_from_potential(float(p), params) for p in np.asarray(psi_kpa, dtype=float)])
    if clip_output:
        theta_arr = np.clip(theta_arr, 0.0, params.theta_s)
    return theta_arr


def theta_thresholds_for_psi(
    params: VanGenuchtenParams,
    psi_thresholds_kpa: Mapping[str, float] | Iterable[tuple[str, float]],
) -> dict[str, float]:
    """Precompute θ thresholds equivalent to given ψ thresholds.

    This is often better than converting every predicted θ to ψ for decisions.
    You convert *the thresholds once*, then compare θ directly.
    """
    items = psi_thresholds_kpa.items() if hasattr(psi_thresholds_kpa, "items") else psi_thresholds_kpa
    return {name: psi_to_theta(float(psi), params) for name, psi in items}
