"""
Residual Model for SMPS.

Handles ML model training with proper validation strategies,
baseline comparisons, and temporal cross-validation.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

from smps.ml.training import ResidualTrainer, TrainingConfig
from smps.ml.evaluation import EnhancedEvaluator

logger = logging.getLogger(__name__)


@dataclass
class ResidualConfig:
    """Configuration for residual model training and validation."""
    # Cross-validation
    use_temporal_cv: bool = True
    n_cv_folds: int = 5

    # Baselines to compare against
    baselines: List[str] = None

    # Metrics
    primary_metric: str = 'rmse'
    additional_metrics: List[str] = None

    def __post_init__(self):
        if self.baselines is None:
            self.baselines = ['persistence', 'climatology', 'physics_only']
        if self.additional_metrics is None:
            self.additional_metrics = ['kge', 'nse', 'r2']


@dataclass
class BaselineModel:
    """Simple baseline model for comparison."""
    name: str
    prediction_function: Any
    description: str


class ResidualModel:
    """
    ML model for residual learning with comprehensive validation.

    Supports both θ-space and ψ-space training with proper baseline comparisons.
    """

    def __init__(self, config: Optional[TrainingConfig] = None,
                 val_config: Optional[ResidualConfig] = None):
        self.config = config or TrainingConfig()
        self.val_config = val_config or ResidualConfig()

        self.trainer = ResidualTrainer(self.config)
        self.evaluator = EnhancedEvaluator()

        # Baseline models
        self.baselines = self._create_baselines()

        # Trained models
        self.models: Dict[int, Any] = {}
        self.feature_cols: List[str] = []

    def _create_baselines(self) -> Dict[str, BaselineModel]:
        """Create baseline models for comparison."""
        baselines = {}

        # Persistence: θ(t+h) = θ(t)
        def persistence_predict(df, horizon):
            return df['soil_moisture'].values

        baselines['persistence'] = BaselineModel(
            name='persistence',
            prediction_function=persistence_predict,
            description='Predicts current value persists'
        )

        # Climatology: θ(t+h) = long-term mean for that day/month
        def climatology_predict(df, horizon):
            # Simple monthly climatology
            monthly_means = df.groupby(df['date'].dt.month)[
                'soil_moisture'].transform('mean')
            return monthly_means.values

        baselines['climatology'] = BaselineModel(
            name='climatology',
            prediction_function=climatology_predict,
            description='Predicts long-term average for date'
        )

        # Physics-only: θ(t+h) = physics prediction
        def physics_predict(df, horizon):
            physics_col = f'theta_phys_surface'  # Assume physics model provides this
            if physics_col in df.columns:
                return df[physics_col].values
            else:
                return df['soil_moisture'].values  # Fallback

        baselines['physics_only'] = BaselineModel(
            name='physics_only',
            prediction_function=physics_predict,
            description='Uses physics model prediction only'
        )

        return baselines

    def train_with_validation(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                              horizons: List[int], feature_cols: List[str],
                              target_col: str = 'target') -> Dict[str, Any]:
        """
        Train models with comprehensive validation.

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            horizons: Forecast horizons (hours)
            feature_cols: Feature column names
            target_col: Target column name

        Returns:
            Training results and validation metrics
        """
        logger.info(f"Training models for horizons: {horizons}")

        self.feature_cols = feature_cols
        training_results = {}

        # Train models for each horizon
        for horizon in horizons:
            logger.info(f"Training {horizon}h horizon model...")

            # Prepare target data
            target_data = self._prepare_horizon_targets(
                train_df, horizon, target_col)

            # Train model
            model, fold_results = self.trainer.train_with_site_cv(
                target_data, target_data[f'residual_target_{horizon}h'].values,
                target_data['station_id'].values, feature_cols,
                n_folds=self.val_config.n_cv_folds, horizon_hours=horizon
            )

            self.models[horizon] = model
            training_results[horizon] = {
                'model': model,
                'cv_results': fold_results,
                'feature_importance': fold_results[0].feature_importance if fold_results else {}
            }

        # Comprehensive validation
        validation_results = self.validate_models(val_df, horizons, target_col)

        return {
            'training': training_results,
            'validation': validation_results
        }

    def _prepare_horizon_targets(self, df: pd.DataFrame, horizon: int,
                                 target_col: str) -> pd.DataFrame:
        """Prepare target data for a specific horizon."""
        # This would use the create_residual_targets function from training.py
        # For now, create simple shifted targets
        df = df.copy()

        # Shift target by horizon (simplified - should use proper temporal shifting)
        df[f'target_{horizon}h'] = df.groupby(
            'station_id')[target_col].shift(-horizon)
        df[f'physics_{horizon}h'] = df.groupby(
            'station_id')['theta_phys_surface'].shift(-horizon)
        df[f'residual_target_{horizon}h'] = df[f'target_{horizon}h'] - \
            df[f'physics_{horizon}h']

        return df

    def validate_models(self, val_df: pd.DataFrame, horizons: List[int],
                        target_col: str = 'target') -> Dict[str, Any]:
        """
        Comprehensive model validation with baselines.

        Args:
            val_df: Validation dataframe
            horizons: Forecast horizons
            target_col: Target column name

        Returns:
            Validation results
        """
        logger.info("Running comprehensive model validation...")

        results = {}

        for horizon in horizons:
            logger.info(f"Validating {horizon}h predictions...")

            # Prepare validation data
            val_data = self._prepare_horizon_targets(
                val_df, horizon, target_col)
            val_data = val_data[val_data[f'target_{horizon}h'].notna()].copy()

            if len(val_data) == 0:
                logger.warning(f"No valid validation data for {horizon}h")
                continue

            # Get model predictions
            if horizon in self.models:
                model = self.models[horizon]
                residual_pred, uncertainty = self.trainer.predict_with_uncertainty(
                    model, val_data[self.feature_cols], self.feature_cols
                )

                # Combine with physics
                physics_pred = val_data[f'physics_{horizon}h'].values
                model_pred = physics_pred + residual_pred
                actuals = val_data[f'target_{horizon}h'].values

                # Evaluate model
                model_metrics = self.evaluator.evaluate_predictions(
                    model_pred, actuals, uncertainty
                )

                # Evaluate baselines
                baseline_results = {}
                for baseline_name, baseline in self.baselines.items():
                    baseline_pred = baseline.prediction_function(
                        val_data, horizon)
                    baseline_metrics = self.evaluator.evaluate_predictions(
                        baseline_pred, actuals
                    )
                    baseline_results[baseline_name] = baseline_metrics

                    # Add to evaluator for comparison
                    self.evaluator.add_baseline(
                        baseline_name, baseline_pred, actuals)

                # Compare to baselines
                baseline_comparisons = self.evaluator.compare_to_baselines(
                    model_pred, actuals, uncertainty
                )

                results[horizon] = {
                    'model_metrics': model_metrics,
                    'baseline_metrics': baseline_results,
                    'baseline_comparisons': baseline_comparisons,
                    'n_samples': len(val_data),
                    'feature_importance': self._get_feature_importance(horizon)
                }

        return results

    def _get_feature_importance(self, horizon: int) -> Dict[str, float]:
        """Get feature importance for a horizon."""
        if horizon in self.models:
            model = self.models[horizon]
            try:
                importance = dict(zip(
                    self.feature_cols,
                    model.feature_importance(importance_type='gain')
                ))
                return importance
            except:
                return {}
        return {}

    def predict(self, df: pd.DataFrame, horizons: List[int]) -> Dict[int, np.ndarray]:
        """
        Make predictions for multiple horizons.

        Args:
            df: Input dataframe
            horizons: Forecast horizons

        Returns:
            Dictionary of predictions by horizon
        """
        predictions = {}

        for horizon in horizons:
            if horizon not in self.models:
                logger.warning(f"No model for horizon {horizon}h")
                continue

            model = self.models[horizon]

            # Predict residual
            residual_pred, _ = self.trainer.predict_with_uncertainty(
                model, df[self.feature_cols], self.feature_cols
            )

            # Add physics prior
            physics_prior = df['theta_phys_surface'].values  # Assume available
            predictions[horizon] = physics_prior + residual_pred

        return predictions

    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        summary = {
            'config': {
                'training': self.config.__dict__,
                'validation': self.val_config.__dict__,
            },
            'horizons': list(self.models.keys()),
            'n_features': len(self.feature_cols),
            'feature_cols': self.feature_cols[:10],  # First 10 for brevity
            'baselines': [b.name for b in self.baselines.values()],
        }

        return summary

    def save_models(self, output_dir: Path):
        """Save trained models."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for horizon, model in self.models.items():
            model_path = output_dir / f"residual_model_{horizon}h.txt"
            model.save_model(str(model_path))
            logger.info(f"Saved {horizon}h model to {model_path}")

    def load_models(self, model_dir: Path):
        """Load trained models."""
        for horizon in [24, 72, 168]:  # Standard horizons
            model_path = model_dir / f"residual_model_{horizon}h.txt"
            if model_path.exists():
                model = lgb.Booster(model_file=str(model_path))
                self.models[horizon] = model
                logger.info(f"Loaded {horizon}h model from {model_path}")
