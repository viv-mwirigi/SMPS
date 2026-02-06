"""Validation configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SensorDepthMapping:
    """Provide layer weights for mapping sensor depths to model layers."""

    surface_max_m: float = 0.10
    root_max_m: float = 0.50

    def get_layer_weights(self, depth_m: float) -> Dict[str, float]:
        if depth_m <= self.surface_max_m:
            return {"surface": 1.0, "root_zone": 0.0, "deep": 0.0}
        if depth_m <= self.root_max_m:
            return {"surface": 0.2, "root_zone": 0.8, "deep": 0.0}
        return {"surface": 0.0, "root_zone": 0.3, "deep": 0.7}
