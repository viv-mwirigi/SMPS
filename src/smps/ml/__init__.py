"""
Machine Learning module for soil water potential prediction.

This module provides residual learning models that correct
physics model predictions using gradient boosting.

Philosophy:
- Train and evaluate in ψ-space (matric potential) where physics is valid
- Use standard PTF for θ conversion (no site-specific calibration)
- Site-specific PTF calibration = overfitting with zero transferability
"""

from smps.ml.hybrid_model import (
    HybridModelConfig,
    HybridTensionModel,
    ResidualLearner,
)
from smps.ml.training import (
    TrainingConfig,
    ResidualTrainer,
    create_residual_targets,
    create_matric_residual_targets,
    create_prediction_features,
    # New training utilities
    SiteBlockedCV,
    HorizonTrainingResult,
    # Sequential feature engineering
    add_sequential_features,
    prepare_features_with_sequences,
    # Site bias correction
    compute_site_bias_corrections,
    apply_site_bias_correction,
    create_matric_residual_targets_debiased,
)
from smps.ml.retention_learning import (
    # PSI-space evaluation (recommended)
    evaluate_psi_space_metrics,
    evaluate_log_psi_space_metrics,
)
from smps.ml.residual_model import (
    ResidualModel,
    ResidualConfig,
)

__all__ = [
    "HybridModelConfig",
    "HybridTensionModel",
    "ResidualLearner",
    "TrainingConfig",
    "ResidualTrainer",
    "add_sequential_features",
    "prepare_features_with_sequences",
    "ModelEvaluatorExtended",
    "create_residual_targets",
    "create_matric_residual_targets",
    "create_matric_residual_targets_debiased",
    "compute_site_bias_corrections",
    "apply_site_bias_correction",
    "create_prediction_features",
    # New training utilities
    "SiteBlockedCV",
    "HorizonTrainingResult",
    # PSI-space evaluation
    "evaluate_psi_space_metrics",
    "evaluate_log_psi_space_metrics",
    # Residual Model
    "ResidualModel",
    "ResidualConfig",
]
