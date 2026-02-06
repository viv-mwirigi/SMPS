"""Physics-oriented validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from smps.validation.metrics import ValidationMetrics, compute_metrics


@dataclass(frozen=True)
class StandardPhysicsMetrics:
    """Subset of metrics used by physics validation checks."""

    n_valid: int
    kge: float
    nse: float
    ubrmse: float
    bias: float

    @classmethod
    def from_validation(cls, metrics: ValidationMetrics) -> "StandardPhysicsMetrics":
        return cls(
            n_valid=metrics.n_valid,
            kge=metrics.kge,
            nse=metrics.nse,
            ubrmse=metrics.ubrmse,
            bias=metrics.mbe,
        )


@dataclass(frozen=True)
class PhysicsValidationReport:
    """Container for physics validation outputs."""

    standard_metrics: StandardPhysicsMetrics
    raw_metrics: ValidationMetrics


def run_physics_validation(
    obs: np.ndarray,
    pred: np.ndarray,
) -> PhysicsValidationReport:
    metrics = compute_metrics(obs, pred)
    return PhysicsValidationReport(
        standard_metrics=StandardPhysicsMetrics.from_validation(metrics),
        raw_metrics=metrics,
    )
