"""
Uncertainty Quantification for Matric Potential Predictions.

Implements multiple approaches for ψ (matric potential) uncertainty:

1. Quantile Regression: Direct prediction of confidence intervals for ψ
2. Ensemble Uncertainty: Variance from ensemble members for ψ predictions
3. Conformal Prediction: Distribution-free prediction intervals for ψ
4. Physics Uncertainty: Uncertainty from van Genuchten parameter ensembles

Research Background:
- Meinshausen (2006): Quantile Regression Forests
- Tagasovska & Lopez-Paz (2019): Single-model uncertainty
- Romano et al. (2019): Conformal prediction intervals
- Gal & Ghahramani (2016): Dropout as Bayesian approximation

Use Cases:
- Irrigation scheduling under ψ uncertainty
- Risk-aware decision making for ψ-based irrigation
- Model reliability assessment for ψ predictions
- Identifying prediction confidence regions for ψ
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb

logger = logging.getLogger("swpps.ml.uncertainty")


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification."""

    # Quantiles to predict for ψ (matric potential)
    quantiles: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    )

    # Conformal prediction
    conformal_alpha: float = 0.10  # For 90% prediction intervals

    # Ensemble settings
    n_ensemble_members: int = 10
    bootstrap_fraction: float = 0.8

    # Physics uncertainty (van Genuchten parameter ensembles)
    n_parameter_sets: int = 5

    # Model settings
    use_quantile_regression: bool = True
    use_ensemble: bool = True
    use_conformal: bool = False
    use_physics_uncertainty: bool = True


@dataclass
class PsiUncertaintyResult:
    """Uncertainty quantification result for ψ predictions."""

    # Point prediction
    psi_predicted: float

    # Uncertainty measures
    psi_std: float  # Standard deviation
    psi_interval_lower: float  # Lower bound (e.g., 5th percentile)
    psi_interval_upper: float  # Upper bound (e.g., 95th percentile)

    # Quantiles (if available)
    quantiles: Optional[Dict[float, float]] = None

    # Confidence score (0-1, higher = more confident)
    confidence_score: float = 1.0

    # Uncertainty sources
    aleatoric_uncertainty: float = 0.0  # Data noise
    epistemic_uncertainty: float = 0.0  # Model uncertainty
    physics_uncertainty: float = 0.0    # Physics parameter uncertainty

    # Reliability indicators
    is_reliable: bool = True
    uncertainty_category: str = "low"  # low, medium, high


class QuantileRegressor:
    """
    Quantile regression for ψ uncertainty using LightGBM.

    Trains separate models for different quantiles of ψ distribution.
    """

    def __init__(self, config: UncertaintyConfig):
        self.config = config
        self.models: Dict[float, Any] = {}
        self.feature_names: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, quantiles: Optional[List[float]] = None):
        """Fit quantile regression models for ψ."""
        if quantiles is None:
            quantiles = self.config.quantiles

        self.feature_names = list(X.columns)

        for quantile in quantiles:
            logger.info(
                f"Training quantile regression for ψ {quantile:.0%} percentile")

            # LightGBM parameters for quantile regression
            params = {
                'objective': 'quantile',
                'alpha': quantile,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'n_estimators': 1000,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'verbosity': -1
            }

            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)

            self.models[quantile] = model

        logger.info(
            f"Trained quantile regression for {len(quantiles)} ψ quantiles")

    def predict(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """Predict ψ quantiles."""
        if not self.models:
            raise ValueError("Models not trained. Call fit() first.")

        predictions = {}
        for quantile, model in self.models.items():
            pred = model.predict(X)
            predictions[quantile] = pred

        return predictions


class EnsembleUncertainty:
    """
    Ensemble uncertainty for ψ predictions.

    Uses bootstrap aggregation of multiple models to estimate ψ uncertainty.
    """

    def __init__(self, config: UncertaintyConfig):
        self.config = config
        self.models: List[Any] = []
        self.feature_names: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit ensemble of models for ψ uncertainty."""
        self.feature_names = list(X.columns)

        n_samples = int(len(X) * self.config.bootstrap_fraction)

        for i in range(self.config.n_ensemble_members):
            logger.info(
                f"Training ensemble member {i+1}/{self.config.n_ensemble_members}")

            # Bootstrap sample
            indices = np.random.choice(len(X), size=n_samples, replace=True)
            X_boot = X.iloc[indices]
            y_boot = y[indices]

            # Train model (using LightGBM for ψ prediction)
            model = lgb.LGBMRegressor(
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=500,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=i,
                verbosity=-1
            )

            model.fit(X_boot, y_boot)
            self.models.append(model)

        logger.info(
            f"Trained {len(self.models)} ensemble members for ψ uncertainty")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict ψ mean and standard deviation."""
        if not self.models:
            raise ValueError("Models not trained. Call fit() first.")

        # Get predictions from all ensemble members
        predictions = np.array([model.predict(X) for model in self.models])

        # Calculate mean and std across ensemble
        psi_mean = np.mean(predictions, axis=0)
        psi_std = np.std(predictions, axis=0)

        return psi_mean, psi_std


class PsiUncertaintyQuantifier:
    """
    Comprehensive uncertainty quantification for ψ predictions.

    Combines multiple uncertainty sources:
    1. Model uncertainty (ensemble/epistemic)
    2. Data uncertainty (aleatoric)
    3. Physics uncertainty (parameter ensembles)
    """

    def __init__(self, config: Optional[UncertaintyConfig] = None):
        self.config = config or UncertaintyConfig()
        self.quantile_regressor: Optional[QuantileRegressor] = None
        self.ensemble_uncertainty: Optional[EnsembleUncertainty] = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit uncertainty quantification models."""
        logger.info("Fitting ψ uncertainty quantification models")

        # Fit quantile regression for prediction intervals
        if self.config.use_quantile_regression:
            self.quantile_regressor = QuantileRegressor(self.config)
            self.quantile_regressor.fit(X, y)

        # Fit ensemble for epistemic uncertainty
        if self.config.use_ensemble:
            self.ensemble_uncertainty = EnsembleUncertainty(self.config)
            self.ensemble_uncertainty.fit(X, y)

        self.is_fitted = True
        logger.info("ψ uncertainty quantification models fitted")

    def predict_uncertainty(self, X: pd.DataFrame,
                            physics_uncertainty: Optional[np.ndarray] = None) -> List[PsiUncertaintyResult]:
        """
        Predict ψ with comprehensive uncertainty quantification.

        Args:
            X: Feature matrix
            physics_uncertainty: Optional physics uncertainty from parameter ensembles

        Returns:
            List of uncertainty results for each prediction
        """
        if not self.is_fitted:
            raise ValueError(
                "Uncertainty quantifier not fitted. Call fit() first.")

        results = []

        # Get quantile predictions
        quantile_preds = None
        if self.quantile_regressor:
            quantile_preds = self.quantile_regressor.predict(X)

        # Get ensemble predictions
        ensemble_mean = None
        ensemble_std = None
        if self.ensemble_uncertainty:
            ensemble_mean, ensemble_std = self.ensemble_uncertainty.predict(X)

        # Process each sample
        for i in range(len(X)):
            # Base prediction (use median from quantiles or ensemble mean)
            if quantile_preds and 0.5 in quantile_preds:
                psi_pred = quantile_preds[0.5][i]
            elif ensemble_mean is not None:
                psi_pred = ensemble_mean[i]
            else:
                raise ValueError("No prediction method available")

            # Calculate uncertainty intervals
            if quantile_preds and 0.05 in quantile_preds and 0.95 in quantile_preds:
                lower_bound = quantile_preds[0.05][i]
                upper_bound = quantile_preds[0.95][i]
            else:
                # Fallback to ensemble std
                std = ensemble_std[i] if ensemble_std is not None else 0.1
                lower_bound = psi_pred - 1.96 * std
                upper_bound = psi_pred + 1.96 * std

            # Calculate uncertainties
            epistemic_uncertainty = ensemble_std[i] if ensemble_std is not None else 0.0
            aleatoric_uncertainty = abs(
                upper_bound - lower_bound) / 3.92  # Rough estimate
            physics_uncertainty_val = physics_uncertainty[i] if physics_uncertainty is not None else 0.0

            # Total uncertainty
            total_uncertainty = np.sqrt(
                epistemic_uncertainty**2 +
                aleatoric_uncertainty**2 +
                physics_uncertainty_val**2
            )

            # Confidence score (inverse of normalized uncertainty)
            confidence_score = max(
                0.0, min(1.0, 1.0 - (total_uncertainty / 2.0)))

            # Uncertainty category
            if total_uncertainty < 0.5:
                category = "low"
            elif total_uncertainty < 1.0:
                category = "medium"
            else:
                category = "high"

            # Reliability assessment
            is_reliable = confidence_score > 0.7 and abs(
                psi_pred) < 10.0  # Reasonable ψ range

            result = PsiUncertaintyResult(
                psi_predicted=psi_pred,
                psi_std=total_uncertainty,
                psi_interval_lower=lower_bound,
                psi_interval_upper=upper_bound,
                quantiles={q: preds[i] for q, preds in quantile_preds.items(
                )} if quantile_preds else None,
                confidence_score=confidence_score,
                aleatoric_uncertainty=aleatoric_uncertainty,
                epistemic_uncertainty=epistemic_uncertainty,
                physics_uncertainty=physics_uncertainty_val,
                is_reliable=is_reliable,
                uncertainty_category=category
            )

            results.append(result)

        return results

    def get_uncertainty_stats(self, X: pd.DataFrame) -> Dict[str, float]:
        """Get uncertainty statistics for the dataset."""
        results = self.predict_uncertainty(X)

        return {
            'mean_psi_uncertainty': np.mean([r.psi_std for r in results]),
            'median_psi_uncertainty': np.median([r.psi_std for r in results]),
            'mean_confidence': np.mean([r.confidence_score for r in results]),
            'reliable_predictions_pct': np.mean([r.is_reliable for r in results]) * 100,
            'low_uncertainty_pct': np.mean([r.uncertainty_category == 'low' for r in results]) * 100,
            'high_uncertainty_pct': np.mean([r.uncertainty_category == 'high' for r in results]) * 100,
        }
