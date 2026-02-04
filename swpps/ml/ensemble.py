"""
Ensemble Methods for Matric Potential Prediction.

Implements stacking ensemble for ψ (matric potential) prediction that combines
multiple base learners with a meta-learner.

Ensemble Architecture for ψ:
─────────────────────────────────────────────────────────────────
                        META-LEARNER (Ridge/XGB)
                              ▲
                              │ Out-of-fold ψ predictions
                ┌─────────────┼─────────────┐
                │             │             │
           ┌────┴────┐  ┌────┴────┐  ┌────┴────┐
           │ LightGBM │  │  XGBoost │  │   RF    │
           │  Depth1  │  │  Depth1  │  │ Depth1  │
           └─────────┘  └─────────┘  └─────────┘
                ▲             ▲             ▲
                │             │             │
         ┌──────┴──────┴──────┴──────┐
         │     Features + Physics     │
         └────────────────────────────┘
─────────────────────────────────────────────────────────────────

Benefits for ψ prediction:
- Combines strengths of different algorithms for ψ modeling
- Reduces overfitting through out-of-fold predictions
- Handles diverse feature types affecting ψ

Research References:
- Wolpert (1992): Stacked Generalization
- Breiman (1996): Stacked Regressions
"""

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb

logger = logging.getLogger("swpps.ml.ensemble")


@dataclass
class EnsembleConfig:
    """Configuration for ψ ensemble methods."""

    # Base models
    base_models: List[str] = field(default_factory=lambda: [
                                   'lightgbm', 'xgboost', 'rf'])

    # Meta-learner
    meta_model: str = "ridge"  # ridge, xgboost, lightgbm

    # Cross-validation for stacking
    n_folds: int = 5
    cv_random_state: int = 42

    # Base model hyperparameters (ψ-specific tuning)
    lightgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'verbosity': -1
    })

    xgboost_params: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'verbosity': 0
    })

    rf_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 500,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    })

    # Meta-model hyperparameters
    meta_params: Dict[str, Any] = field(default_factory=lambda: {
        'alpha': 1.0,
        'random_state': 42
    })


class PsiEnsembleTrainer:
    """
    Trains base models for ψ ensemble using cross-validation.

    Generates out-of-fold predictions for meta-learner training.
    """

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.base_models: Dict[str, Any] = {}
        self.feature_names: Optional[List[str]] = None

    def _create_base_model(self, model_name: str) -> Any:
        """Create a base model for ψ prediction."""
        if model_name == 'lightgbm':
            return lgb.LGBMRegressor(**self.config.lightgbm_params)
        elif model_name == 'xgboost':
            return xgb.XGBRegressor(**self.config.xgboost_params)
        elif model_name == 'rf':
            return RandomForestRegressor(**self.config.rf_params)
        else:
            raise ValueError(f"Unknown base model: {model_name}")

    def train_base_models_cv(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Train base models using cross-validation and generate OOF predictions.

        Returns:
            Tuple of (trained_models, oof_predictions_df)
        """
        logger.info(
            f"Training {len(self.config.base_models)} base models for ψ ensemble with {self.config.n_folds}-fold CV")

        self.feature_names = list(X.columns)

        # Initialize storage
        trained_models = {}
        oof_predictions = {model_name: np.zeros(
            len(X)) for model_name in self.config.base_models}

        # Cross-validation
        kf = KFold(n_splits=self.config.n_folds, shuffle=True,
                   random_state=self.config.cv_random_state)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.info(f"Training fold {fold + 1}/{self.config.n_folds}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            for model_name in self.config.base_models:
                # Train model on this fold
                model = self._create_base_model(model_name)
                model.fit(X_train, y_train)

                # Store model (keep last fold's model for final ensemble)
                trained_models[model_name] = model

                # Generate OOF predictions
                oof_pred = model.predict(X_val)
                oof_predictions[model_name][val_idx] = oof_pred

        # Create OOF predictions DataFrame
        oof_df = pd.DataFrame(oof_predictions)

        logger.info("Base model training completed for ψ ensemble")
        return trained_models, oof_df


class PsiStackingEnsemble:
    """
    Stacking ensemble for ψ prediction.

    Combines multiple base learners with a meta-learner for improved ψ accuracy.
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.base_models: Dict[str, Any] = {}
        self.meta_model: Optional[Any] = None
        self.trainer: PsiEnsembleTrainer = PsiEnsembleTrainer(self.config)
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False

    def _create_meta_model(self) -> Any:
        """Create the meta-learner for ψ ensemble."""
        if self.config.meta_model == 'ridge':
            return Ridge(**self.config.meta_params)
        elif self.config.meta_model == 'xgboost':
            return xgb.XGBRegressor(**self.config.meta_params)
        elif self.config.meta_model == 'lightgbm':
            return lgb.LGBMRegressor(**self.config.meta_params)
        else:
            raise ValueError(f"Unknown meta model: {self.config.meta_model}")

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit the stacking ensemble for ψ prediction."""
        logger.info("Fitting ψ stacking ensemble")

        # Train base models with CV and get OOF predictions
        self.base_models, oof_predictions = self.trainer.train_base_models_cv(
            X, y)

        # Train meta-model on OOF predictions
        self.meta_model = self._create_meta_model()
        self.meta_model.fit(oof_predictions, y)

        self.feature_names = list(X.columns)
        self.is_fitted = True

        logger.info("ψ stacking ensemble fitted successfully")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ψ predictions using the ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Get base model predictions
        base_predictions = {}
        for model_name, model in self.base_models.items():
            base_predictions[model_name] = model.predict(X)

        # Create prediction matrix for meta-model
        pred_matrix = pd.DataFrame(base_predictions)

        # Meta-model prediction
        return self.meta_model.predict(pred_matrix)

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ψ predictions with ensemble uncertainty.

        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Get predictions from all base models
        base_predictions = np.array([
            model.predict(X) for model in self.base_models.values()
        ])

        # Ensemble prediction (mean of base models)
        ensemble_pred = np.mean(base_predictions, axis=0)

        # Uncertainty as standard deviation across base models
        ensemble_uncertainty = np.std(base_predictions, axis=0)

        return ensemble_pred, ensemble_uncertainty

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Get importance from meta-model if available
        if hasattr(self.meta_model, 'coef_'):
            # Linear meta-model
            importance = np.abs(self.meta_model.coef_)
            return dict(zip(self.config.base_models, importance))

        elif hasattr(self.meta_model, 'feature_importances_'):
            # Tree-based meta-model
            importance = self.meta_model.feature_importances_
            return dict(zip(self.config.base_models, importance))

        else:
            # Fallback: equal importance
            n_models = len(self.config.base_models)
            return {name: 1.0/n_models for name in self.config.base_models}

    def evaluate_ensemble(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble performance on ψ predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        predictions = self.predict(X)

        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - predictions))

        # ψ-specific metrics (reasonable range: -15 to 0 kPa)
        within_range = np.mean((predictions >= -15) & (predictions <= 0))

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'psi_within_reasonable_range': within_range * 100,
        }
