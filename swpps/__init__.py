"""
SWPPS - Soil Water Potential Prediction System
=============================================

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
    from swpps import create_pipeline

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
    from swpps.validation import ValidationMetrics, compute_metrics, generate_report

Calibration:
    from swpps.calibration import calibrate_van_genuchten, CalibrationResult
"""

__version__ = "1.0.0"
__author__ = "Waziup Irrigation Team"

# Core types and configuration
from swpps.core.config import SWPPSConfig, load_config, save_config
from swpps.core.constants import (
    IRRIGATION_THRESHOLDS,
    CROP_THRESHOLDS,
    MatricPotentialRange,
)
from swpps.core.types import (
    MatricPotential,
    SoilTensionKPa,
    VanGenuchtenParams,
    SoilMoistureStatus,
    PredictionResult,
    IrrigationDecision,
)

# Physics models
from swpps.physics.van_genuchten import (
    water_content_from_potential,
    potential_from_water_content,
    estimate_van_genuchten_params,
)
from swpps.physics.water_balance import TensionSpaceWaterBalance
from swpps.physics.evapotranspiration import (
    CropCoefficients,
    compute_et_partitioning,
    get_Kcb_from_ndvi,
)
from swpps.physics.tropical import (
    TropicalSoilCorrections,
    partition_infiltration,
)

# Data handling
from swpps.data.weather import OpenMeteoClient
from swpps.data.sensors import WaziGateClient, SensorDataManager
from swpps.data.quality import QualityControlPipeline, QCConfig, run_qc_pipeline

# Feature engineering
from swpps.features.engineering import FeatureEngineer, create_training_dataset

# ML models
from swpps.ml.hybrid_model import HybridTensionModel, ResidualLearner

# Prediction
from swpps.prediction.forecaster import SoilWaterForecaster, create_forecaster
from swpps.prediction.decision import IrrigationDecisionEngine, create_decision_engine

# Actuation
from swpps.actuation.irrigation import IrrigationActuator, create_actuator

# Validation
from swpps.validation import (
    ValidationMetrics,
    compute_metrics,
    compute_kge,
    compute_nse,
    ValidationReport,
    generate_report,
    SWPPSPlotter,
)

# Calibration
from swpps.calibration import (
    CalibrationResult,
    VanGenuchtenCalibrator,
    calibrate_van_genuchten,
    calibrate_water_balance,
    TropicalSoilCalibrator,
    calibrate_for_african_soil,
)

# Main pipeline
import importlib.util
spec = importlib.util.spec_from_file_location(
    'pipeline', '/home/viv/SMPS/swpps/pipeline.py')
pipeline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_module)
SWPPSPipeline = pipeline_module.SWPPSPipeline
PipelineConfig = pipeline_module.PipelineConfig
create_pipeline = pipeline_module.create_pipeline

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
    # Data
    "OpenMeteoClient",
    "WaziGateClient",
    "SensorDataManager",
    "QualityControlPipeline",
    "QCConfig",
    "run_qc_pipeline",
    # Features
    "FeatureEngineer",
    "create_training_dataset",
    # ML
    "HybridTensionModel",
    "ResidualLearner",
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
]
