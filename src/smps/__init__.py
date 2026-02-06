"""
SMPS - Soil Moisture Prediction System
======================================

A unified system for soil moisture prediction using matric potential (kPa)
as the primary prediction target, enabling universal irrigation thresholds
that work across all soil types.

Key Features:
- Physics-based water balance model in tension space
- ML residual learning for bias correction
- Multi-horizon forecasting (0h, 24h, 72h, 168h)
- Universal crop-specific irrigation thresholds
- WaziGate IoT integration

The fundamental innovation is predicting MATRIC POTENTIAL instead of
volumetric water content. This eliminates the need for soil-specific
calibration because:
- A sandy soil at -50 kPa has the same plant-available water ENERGY
  as clay at -50 kPa, even though VWC differs dramatically
- Irrigation thresholds become universal across soil types
- Plant stress is directly related to matric potential

Typical thresholds (work for ALL soils):
- Field capacity: -10 to -33 kPa
- Optimal range: -33 to -100 kPa
- Stress onset: -100 to -200 kPa
- Wilting point: -1500 kPa

Usage:
    from smps import create_pipeline

    pipeline = create_pipeline(
        site_id="plot_1",
        latitude=1.234,
        longitude=36.789,
        crop_type="tomato",
    )

    result = pipeline.run_prediction_cycle()

    if result["decision"]["should_irrigate"]:
        print(f"Irrigate {result['decision']['amount_mm']} mm")

Validation:
    from smps.validation import ValidationMetrics, compute_metrics, generate_report

Calibration:
    from smps.calibration import calibrate_van_genuchten, CalibrationResult
"""

from smps.pipeline.harmonizer import Harmonizer
__version__ = "1.0.0"
__author__ = "Waziup Irrigation Team"

# Core types and configuration
from smps.core.config import SWPPSConfig, load_config, save_config
from smps.core.constants import (
    IRRIGATION_THRESHOLDS,
    CROP_THRESHOLDS,
    MatricPotentialRange,
)
from smps.core.types import (
    MatricPotential,
    SoilTensionKPa,
    VanGenuchtenParams,
    SoilMoistureStatus,
    PredictionResult,
    IrrigationDecision,
)

# Physics models
from smps.physics.van_genuchten import (
    water_content_from_potential,
    potential_from_water_content,
    estimate_van_genuchten_params,
)
from smps.physics.water_balance import TensionSpaceWaterBalance
from smps.physics.evapotranspiration import (
    CropCoefficients,
    compute_et_partitioning,
    get_Kcb_from_ndvi,
)
from smps.physics.tropical import (
    TropicalSoilCorrections,
    partition_infiltration,
)

# Data handling
from smps.data.weather import OpenMeteoClient
from smps.data.sensors import WaziGateClient, SensorDataManager
from smps.data.quality import QualityControlPipeline, QCConfig, run_qc_pipeline
from smps.data.site_manager import SiteManager
from smps.data.preprocessor import DataPreprocessor

# Feature engineering
from smps.features.engineer import FeatureEngineer

# ML models
from smps.ml.hybrid_model import HybridTensionModel, ResidualLearner
from smps.ml.residual_model import ResidualModel

# Physics models
from smps.physics.model import PhysicsModel

# Prediction
from smps.prediction.forecaster import SoilWaterForecaster, create_forecaster
from smps.prediction.decision import IrrigationDecisionEngine, create_decision_engine

# Actuation
from smps.actuation.irrigation import IrrigationActuator, create_actuator

# Validation
from smps.validation import (
    ValidationMetrics,
    compute_metrics,
    compute_kge,
    compute_nse,
    ValidationReport,
    generate_report,
    SWPPSPlotter,
)

# Calibration
from smps.calibration import (
    CalibrationResult,
    VanGenuchtenCalibrator,
    calibrate_van_genuchten,
    calibrate_water_balance,
    TropicalSoilCalibrator,
    calibrate_for_african_soil,
)

# Main pipeline
# NOTE: There is also a `swpps/pipeline/` package; load the file-based
# orchestrator pipeline explicitly but in a path-portable way.
import importlib.util
from pathlib import Path

_pipeline_path = Path(__file__).resolve().parent / "pipeline.py"
_spec = importlib.util.spec_from_file_location(
    "swpps._pipeline_file", str(_pipeline_path))
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load SWPPSPipeline from {_pipeline_path}")

_pipeline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline_module)

SWPPSPipeline = _pipeline_module.SWPPSPipeline
PipelineConfig = _pipeline_module.PipelineConfig
create_pipeline = _pipeline_module.create_pipeline

# Data science pipeline

__all__ = [
    # Version
    "__version__",
    # Core config
    "SWPPSConfig",
    "load_config",
    "save_config",
    "IRRIGATION_THRESHOLDS",
    "CROP_THRESHOLDS",
    "MatricPotentialRange",
    # Core types
    "MatricPotential",
    "SoilTensionKPa",
    "VanGenuchtenParams",
    "SoilMoistureStatus",
    "PredictionResult",
    "IrrigationDecision",
    # Physics
    "water_content_from_potential",
    "potential_from_water_content",
    "estimate_van_genuchten_params",
    "TensionSpaceWaterBalance",
    "CropCoefficients",
    "compute_et_partitioning",
    "get_Kcb_from_ndvi",
    "TropicalSoilCorrections",
    "partition_infiltration",
    "PhysicsModel",
    # Data
    "OpenMeteoClient",
    "WaziGateClient",
    "SensorDataManager",
    "QualityControlPipeline",
    "QCConfig",
    "run_qc_pipeline",
    "SiteManager",
    "DataPreprocessor",
    # Features
    "FeatureEngineer",
    # ML
    "HybridTensionModel",
    "ResidualLearner",
    "ResidualModel",
    # Prediction
    "SoilWaterForecaster",
    "create_forecaster",
    "IrrigationDecisionEngine",
    "create_decision_engine",
    # Actuation
    "IrrigationActuator",
    "create_actuator",
    # Validation
    "ValidationMetrics",
    "compute_metrics",
    "compute_kge",
    "compute_nse",
    "ValidationReport",
    "generate_report",
    "SWPPSPlotter",
    # Calibration
    "CalibrationResult",
    "VanGenuchtenCalibrator",
    "calibrate_van_genuchten",
    "calibrate_water_balance",
    "TropicalSoilCalibrator",
    "calibrate_for_african_soil",
    # Pipeline
    "SWPPSPipeline",
    "PipelineConfig",
    "create_pipeline",
    "Harmonizer",
]
