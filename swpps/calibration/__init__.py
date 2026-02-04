"""
SWPPS Calibration Module.

Provides parameter calibration for:
- Van Genuchten soil hydraulic parameters
- Water balance model parameters
- Hybrid model corrections
- Tropical soil corrections for African soils
- Residual-based diagnostic analysis
"""

from .calibrate import (
    CalibrationResult,
    VanGenuchtenCalibrator,
    WaterBalanceCalibrator,
    TropicalSoilCalibrator,
    calibrate_van_genuchten,
    calibrate_water_balance,
    calibrate_for_african_soil,
    ResidualDiagnostics,
    ResidualAnalyzer,
)

from .objective import (
    ObjectiveFunction,
    rmse_objective,
    nse_objective,
    kge_objective,
    multi_objective,
)

__all__ = [
    "CalibrationResult",
    "VanGenuchtenCalibrator",
    "WaterBalanceCalibrator",
    "TropicalSoilCalibrator",
    "calibrate_van_genuchten",
    "calibrate_water_balance",
    "calibrate_for_african_soil",
    "ResidualDiagnostics",
    "ResidualAnalyzer",
    "ObjectiveFunction",
    "rmse_objective",
    "nse_objective",
    "kge_objective",
    "multi_objective",
]
