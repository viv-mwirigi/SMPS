"""
SMPS Machine Learning Module.

Provides hybrid physics-ML models for soil moisture prediction.

Architecture:
------------
1. Physics Model: Generates physics-based priors at multiple depths
2. Feature Engineering: Climate forcings, static attributes, temporal patterns
3. Residual Learning: ML model learns residuals from physics baseline
4. Stacking Ensemble: LightGBM + XGBoost with meta-learner
5. Uncertainty Quantification: Quantile regression, ensemble uncertainty

Key Components:
- CanonicalDatasetBuilder: Unified feature space construction
- FeatureStore: Feature versioning and caching
- ResidualLearner: Learns physics model residuals
- StackingEnsemble: Multi-model ensemble with meta-learning
- SHAPExplainer: Feature importance and interpretability
- Validation: Temporal, spatial, and spatio-temporal splits
- Uncertainty: Quantile regression, conformal prediction

Usage:
------
>>> from smps.ml import (
...     CanonicalDatasetBuilder,
...     HybridSoilMoistureModel,
...     StackingEnsemble,
...     DataSplitter,
...     HybridUncertaintyQuantifier,
... )
>>>
>>> # Build dataset from SpaceIoTBox + Physics
>>> builder = CanonicalDatasetBuilder()
>>> dataset = builder.build(site_id, start_date, end_date)
>>>
>>> # Split data temporally (train on past, test on future)
>>> splitter = DataSplitter(SplitConfig(train_years=[2020,2021,2022]))
>>> train, val, test = splitter.temporal_split(dataset)
>>>
>>> # Train hybrid model with uncertainty
>>> model = HybridSoilMoistureModel()
>>> model.fit(train)
>>>
>>> # Predict with uncertainty
>>> predictions = model.predict(test, return_uncertainty=True)
"""

from smps.ml.dataset_builder import (
    CanonicalDatasetBuilder,
    FeatureConfig,
    DatasetConfig,
)
from smps.ml.feature_store import (
    FeatureStore,
    FeatureGroup,
    FeatureMetadata,
)
from smps.ml.hybrid_model import (
    HybridSoilMoistureModel,
    ResidualLearner,
    PhysicsResidualTarget,
)
from smps.ml.ensemble import (
    StackingEnsemble,
    HybridStackingEnsemble,
    MultiDepthEnsemble,
    EnsembleConfig,
    BaseModelConfig,
)
from smps.ml.explainer import (
    SHAPExplainer,
    FeatureImportance,
)
from smps.ml.trainer import (
    TrainingOrchestrator,
    TrainingConfig,
    TrainingResults,
    train_site,
)
from smps.ml.validation import (
    DataSplitter,
    SplitConfig,
    MetricsCalculator,
    MetricsResult,
    BaselineModels,
    BaselineComparison,
    ValidationRunner,
    ValidationResult,
)
from smps.ml.uncertainty import (
    UncertaintyConfig,
    PredictionWithUncertainty,
    QuantileRegressor,
    EnsembleUncertainty,
    ConformalPredictor,
    CQRPredictor,
    HybridUncertaintyQuantifier,
    UncertaintyCalibrator,
    UncertaintyQuantifier,  # Alias for HybridUncertaintyQuantifier
)

__all__ = [
    # Dataset building
    "CanonicalDatasetBuilder",
    "FeatureConfig",
    "DatasetConfig",
    # Feature store
    "FeatureStore",
    "FeatureGroup",
    "FeatureMetadata",
    # Hybrid model
    "HybridSoilMoistureModel",
    "ResidualLearner",
    "PhysicsResidualTarget",
    # Ensemble
    "StackingEnsemble",
    "HybridStackingEnsemble",
    "MultiDepthEnsemble",
    "EnsembleConfig",
    "BaseModelConfig",
    # Explainability
    "SHAPExplainer",
    "FeatureImportance",
    # Training
    "TrainingOrchestrator",
    "TrainingConfig",
    "TrainingResults",
    "train_site",
    # Validation
    "DataSplitter",
    "SplitConfig",
    "MetricsCalculator",
    "MetricsResult",
    "BaselineModels",
    "BaselineComparison",
    "ValidationRunner",
    "ValidationResult",
    # Uncertainty
    "UncertaintyConfig",
    "PredictionWithUncertainty",
    "QuantileRegressor",
    "EnsembleUncertainty",
    "ConformalPredictor",
    "CQRPredictor",
    "HybridUncertaintyQuantifier",
    "UncertaintyQuantifier",
    "UncertaintyCalibrator",
]
