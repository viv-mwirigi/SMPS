"""
Hybrid Physics-ML Model for Matric Potential Prediction.

This module implements residual learning where:
1. Physics model provides baseline matric potential prediction
2. ML model learns the residual (bias correction)
3. Final prediction = Physics + ML residual

The key innovation is predicting directly in matric potential space,
eliminating the need for soil-specific calibration.

Enhancements:
- Ensemble methods for improved robustness
- Advanced uncertainty quantification
- Adaptive model selection per horizon
"""

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger("swpps.ml.hybrid_model")


@dataclass
class HybridModelConfig:
    """Configuration for hybrid physics-ML model."""

    # Model type
    model_type: str = "lightgbm"  # "lightgbm", "xgboost", "gradient_boosting", "ensemble"

    # Training parameters
    n_estimators: int = 2000
    learning_rate: float = 0.015
    max_depth: int = 8
    early_stopping_rounds: int = 100
    validation_fraction: float = 0.15

    # Regularization
    min_samples_leaf: int = 50
    reg_alpha: float = 0.5
    reg_lambda: float = 1.0

    # Feature handling
    max_features: int = 100
    feature_selection: bool = False
    feature_importance_threshold: float = 0.01

    # Physics constraints
    enforce_bounds: bool = True
    min_potential_kpa: float = -2000.0
    max_potential_kpa: float = 0.0

    # Uncertainty quantification
    use_quantile_regression: bool = True
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Ensemble options
    use_ensemble: bool = True
    ensemble_models: List[str] = field(
        default_factory=lambda: ["lightgbm", "xgboost", "catboost"])
    ensemble_weights: Optional[List[float]] = None  # None = equal weights

    # Random state for reproducibility
    random_state: int = 42

    # Cross-validation
    use_time_series_cv: bool = True
    n_cv_splits: int = 5


class ResidualLearner:
    """
    Learns residuals between physics model and observations.

    The residual learner captures patterns the physics model misses:
    - Systematic biases in physics parameterization
    - Local effects not in physics model
    - Complex feature interactions
    """

    def __init__(self, config: Optional[HybridModelConfig] = None):
        self.config = config or HybridModelConfig()
        self.model = None
        self.ensemble_models: Dict[str, Any] = {}  # For ensemble mode
        self.quantile_models: Dict[float, Any] = {}
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_names: List[str] = []
        self.selected_features: List[str] = []
        self.feature_importances: Dict[str, float] = {}
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y_observed: np.ndarray,
        y_physics: np.ndarray,
    ) -> "ResidualLearner":
        """
        Fit residual model.

        Args:
            X: Feature matrix
            y_observed: Observed matric potential (kPa)
            y_physics: Physics model predictions (kPa)

        Returns:
            Self
        """
        # Calculate residuals (what physics got wrong)
        residuals = y_observed - y_physics

        # Store feature names
        self.feature_names = list(X.columns)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(
            X_scaled, columns=self.feature_names, index=X.index)

        # Feature selection if enabled
        if self.config.feature_selection:
            self._select_features(X_scaled_df, residuals)
            X_scaled = X_scaled_df[self.selected_features].values
        else:
            self.selected_features = self.feature_names

        # Use time series CV or simple train/val split
        if self.config.use_time_series_cv:
            X_train, X_val, y_train, y_val = self._time_series_split(
                X_scaled, residuals
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, residuals,
                test_size=self.config.validation_fraction,
                random_state=self.config.random_state,
            )

        # Train model(s)
        if self.config.use_ensemble:
            self._fit_ensemble(X_train, y_train, X_val, y_val)
        else:
            self.model = self._create_model()
            self._fit_model(self.model, X_train, y_train, X_val, y_val)

        # Train quantile models for uncertainty
        if self.config.use_quantile_regression:
            for q in self.config.quantiles:
                if q == 0.5:
                    continue  # Use main model for median

                q_model = self._create_quantile_model(q)
                self._fit_model(q_model, X_train, y_train, X_val, y_val)
                self.quantile_models[q] = q_model

        self.is_fitted = True

        # Log training metrics
        val_pred = self._predict_internal(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        mae = mean_absolute_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)
        logger.info(
            "Residual model trained: RMSE=%.3f kPa, MAE=%.3f kPa, R²=%.3f",
            rmse, mae, r2
        )

        return self

    def _time_series_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data respecting time series order."""
        n_samples = len(X)
        split_point = int(n_samples * (1 - self.config.validation_fraction))

        return (
            X[:split_point],
            X[split_point:],
            y[:split_point],
            y[split_point:],
        )

    def _select_features(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Select important features using preliminary model."""
        logger.info("Running feature selection...")

        # Quick fit for feature importance
        quick_model = self._create_model()
        quick_model.n_estimators = 100  # Quick fit

        X_train, X_val, y_train, y_val = train_test_split(
            X.values, y, test_size=0.2, random_state=self.config.random_state
        )
        self._fit_model(quick_model, X_train, y_train, X_val, y_val)

        # Get feature importances
        try:
            importances = quick_model.feature_importances_
        except AttributeError:
            self.selected_features = self.feature_names
            return

        # Normalize
        importances = importances / importances.sum()
        self.feature_importances = dict(zip(self.feature_names, importances))

        # Select features above threshold
        self.selected_features = [
            f for f, imp in self.feature_importances.items()
            if imp >= self.config.feature_importance_threshold
        ]

        # Ensure minimum features
        if len(self.selected_features) < 10:
            sorted_feats = sorted(
                self.feature_importances.items(),
                key=lambda x: x[1],
                reverse=True
            )
            self.selected_features = [f for f, _ in sorted_feats[:20]]

        logger.info(
            "Selected %d features from %d",
            len(self.selected_features), len(self.feature_names)
        )

    def _fit_ensemble(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray
    ) -> None:
        """Fit ensemble of models."""
        for model_type in self.config.ensemble_models:
            try:
                model = self._create_model(model_type)
                self._fit_model(model, X_train, y_train, X_val, y_val)
                self.ensemble_models[model_type] = model

                # Evaluate individual model
                pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, pred))
                logger.info("Ensemble %s: val RMSE=%.3f", model_type, rmse)
            except Exception as e:
                logger.warning("Failed to fit %s: %s", model_type, e)

        if not self.ensemble_models:
            # Fallback to single model
            self.model = self._create_model()
            self._fit_model(self.model, X_train, y_train, X_val, y_val)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        """Internal prediction using model or ensemble."""
        if self.config.use_ensemble and self.ensemble_models:
            predictions = []
            weights = self.config.ensemble_weights or [
                1.0] * len(self.ensemble_models)

            for (model_type, model), weight in zip(self.ensemble_models.items(), weights):
                pred = model.predict(X)
                predictions.append(pred * weight)

            # Weighted average
            total_weight = sum(weights[:len(predictions)])
            return np.sum(predictions, axis=0) / total_weight

        return self.model.predict(X)

    def predict(
        self,
        X: pd.DataFrame,
        y_physics: np.ndarray,
        return_uncertainty: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Predict matric potential.

        Args:
            X: Feature matrix
            y_physics: Physics model predictions
            return_uncertainty: Whether to return uncertainty bounds

        Returns:
            Dictionary with predictions and uncertainty
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # Scale and select features
        X_scaled = self.scaler.transform(X)
        if self.config.feature_selection:
            feat_indices = [self.feature_names.index(
                f) for f in self.selected_features]
            X_scaled = X_scaled[:, feat_indices]

        # Predict residual
        residual_pred = self._predict_internal(X_scaled)

        # Final prediction = physics + residual
        y_pred = y_physics + residual_pred

        # Apply physical bounds
        if self.config.enforce_bounds:
            y_pred = np.clip(y_pred, self.config.min_potential_kpa,
                             self.config.max_potential_kpa)

        result = {
            "prediction": y_pred,
            "physics": y_physics,
            "residual": residual_pred,
        }

        # Add uncertainty bounds
        if return_uncertainty and self.quantile_models:
            for q, model in self.quantile_models.items():
                q_residual = model.predict(X_scaled)
                q_pred = y_physics + q_residual

                if self.config.enforce_bounds:
                    q_pred = np.clip(q_pred, self.config.min_potential_kpa,
                                     self.config.max_potential_kpa)

                result[f"quantile_{int(q*100)}"] = q_pred

            # Standard deviation estimate
            if 0.1 in self.quantile_models and 0.9 in self.quantile_models:
                lower = result["quantile_10"]
                upper = result["quantile_90"]
                # Approximate std from quantiles
                result["std"] = (upper - lower) / 2.56

        # Add ensemble uncertainty if available
        if self.config.use_ensemble and len(self.ensemble_models) > 1:
            ensemble_preds = []
            for model in self.ensemble_models.values():
                pred = model.predict(X_scaled)
                ensemble_preds.append(y_physics + pred)

            result["ensemble_std"] = np.std(ensemble_preds, axis=0)
            result["ensemble_spread"] = np.max(
                ensemble_preds, axis=0) - np.min(ensemble_preds, axis=0)

        return result

    def _create_model(self, model_type: Optional[str] = None):
        """Create the ML model."""
        model_type = model_type or self.config.model_type

        if model_type == "lightgbm":
            try:
                import lightgbm as lgb
                return lgb.LGBMRegressor(
                    n_estimators=self.config.n_estimators,
                    learning_rate=self.config.learning_rate,
                    max_depth=self.config.max_depth,
                    min_child_samples=self.config.min_samples_leaf,
                    reg_alpha=self.config.reg_alpha,
                    reg_lambda=self.config.reg_lambda,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    verbose=-1,
                )
            except ImportError:
                logger.warning("LightGBM not available")

        if model_type == "xgboost":
            try:
                import xgboost as xgb
                return xgb.XGBRegressor(
                    n_estimators=self.config.n_estimators,
                    learning_rate=self.config.learning_rate,
                    max_depth=self.config.max_depth,
                    min_child_weight=self.config.min_samples_leaf,
                    reg_alpha=self.config.reg_alpha,
                    reg_lambda=self.config.reg_lambda,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    verbosity=0,
                )
            except ImportError:
                logger.warning("XGBoost not available")

        if model_type == "catboost":
            try:
                from catboost import CatBoostRegressor
                return CatBoostRegressor(
                    iterations=self.config.n_estimators,
                    learning_rate=self.config.learning_rate,
                    depth=self.config.max_depth,
                    min_data_in_leaf=self.config.min_samples_leaf,
                    l2_leaf_reg=self.config.reg_lambda,
                    random_seed=self.config.random_state,
                    verbose=False,
                )
            except ImportError:
                logger.warning("CatBoost not available")

        # Fallback to sklearn
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=min(self.config.n_estimators, 500),
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
        )

    def _create_quantile_model(self, quantile: float):
        """Create a quantile regression model."""
        if self.config.model_type == "lightgbm":
            try:
                import lightgbm as lgb
                return lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=quantile,
                    n_estimators=self.config.n_estimators,
                    learning_rate=self.config.learning_rate,
                    max_depth=self.config.max_depth,
                    min_child_samples=self.config.min_samples_leaf,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    verbose=-1,
                )
            except ImportError:
                pass

        # Fallback
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            loss="quantile",
            alpha=quantile,
            n_estimators=min(self.config.n_estimators, 500),
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
        )

    def _fit_model(self, model, X_train, y_train, X_val, y_val):
        """Fit model with early stopping if supported."""
        try:
            # LightGBM with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    __import__('lightgbm').early_stopping(
                        self.config.early_stopping_rounds,
                        verbose=False
                    )
                ],
            )
        except (TypeError, AttributeError, ImportError):
            # Fallback without early stopping
            model.fit(X_train, y_train)

    def save(self, path: Path) -> None:
        """Save model to file."""
        data = {
            "config": self.config,
            "model": self.model,
            "ensemble_models": self.ensemble_models,
            "quantile_models": self.quantile_models,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "selected_features": self.selected_features,
            "feature_importances": self.feature_importances,
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "ResidualLearner":
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        learner = cls(data["config"])
        learner.model = data["model"]
        learner.ensemble_models = data.get("ensemble_models", {})
        learner.quantile_models = data["quantile_models"]
        learner.scaler = data["scaler"]
        learner.feature_names = data["feature_names"]
        learner.selected_features = data.get(
            "selected_features", data["feature_names"])
        learner.feature_importances = data.get("feature_importances", {})
        learner.is_fitted = data["is_fitted"]

        return learner


class HybridTensionModel:
    """
    Complete hybrid physics-ML model for matric potential prediction.

    Combines:
    1. Physics water balance model for baseline
    2. ResidualLearner for bias correction
    3. Multi-horizon forecasting
    """

    def __init__(
        self,
        physics_model,
        config: Optional[HybridModelConfig] = None,
    ):
        self.physics_model = physics_model
        self.config = config or HybridModelConfig()
        self.residual_learners: Dict[int, ResidualLearner] = {}  # By horizon
        self.is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "psi_observed_kpa",
        physics_col: str = "psi_physics_root_kpa",
        horizons: List[int] = [0, 24, 72, 168],
    ) -> "HybridTensionModel":
        """
        Fit hybrid model for multiple forecast horizons.

        Args:
            df: Training data with features, observations, and physics predictions
            target_col: Column name for observed matric potential
            physics_col: Column name for physics model predictions
            horizons: Forecast horizons in hours

        Returns:
            Self
        """
        # Get feature columns (exclude identifiers and targets)
        exclude_cols = {
            target_col, physics_col, "date", "site_id", "timestamp",
            "psi_observed", "observation_quality"
        }
        feature_cols = [c for c in df.columns if c not in exclude_cols
                        and not c.startswith("target_")]

        for horizon in horizons:
            logger.info("Training model for horizon %dh", horizon)

            # Create shifted target for forecasting
            if horizon > 0:
                shift_periods = horizon  # Assuming hourly data
                target = df[target_col].shift(-shift_periods)
            else:
                target = df[target_col]

            # Remove rows with NaN target
            valid_mask = ~target.isna()
            X = df.loc[valid_mask, feature_cols]
            y_obs = target[valid_mask].values
            y_phys = df.loc[valid_mask, physics_col].values

            if len(X) < self.config.min_samples_leaf:
                logger.warning("Insufficient data for horizon %d (%d samples)",
                               horizon, len(X))
                continue

            # Train residual learner
            learner = ResidualLearner(self.config)
            learner.fit(X, y_obs, y_phys)
            self.residual_learners[horizon] = learner

        self.is_fitted = bool(self.residual_learners)
        return self

    def predict(
        self,
        df: pd.DataFrame,
        physics_col: str = "psi_physics_root_kpa",
        horizon: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions for a given horizon.

        Args:
            df: Data with features and physics predictions
            physics_col: Column name for physics predictions
            horizon: Forecast horizon in hours

        Returns:
            Dictionary with predictions and uncertainty
        """
        if horizon not in self.residual_learners:
            # Fall back to physics only
            return {
                "prediction": df[physics_col].values,
                "physics": df[physics_col].values,
                "residual": np.zeros(len(df)),
            }

        learner = self.residual_learners[horizon]

        # Get feature columns (same as training)
        feature_cols = learner.feature_names
        X = df[feature_cols]
        y_physics = df[physics_col].values

        return learner.predict(X, y_physics)

    def save(self, directory: Path) -> None:
        """Save all models."""
        directory.mkdir(parents=True, exist_ok=True)

        for horizon, learner in self.residual_learners.items():
            learner.save(directory / f"residual_model_{horizon}h.pkl")

        # Save config
        import json
        with open(directory / "hybrid_config.json", "w") as f:
            json.dump({
                "horizons": list(self.residual_learners.keys()),
                "config": {
                    "model_type": self.config.model_type,
                    "n_estimators": self.config.n_estimators,
                    "learning_rate": self.config.learning_rate,
                },
            }, f, indent=2)

    @classmethod
    def load(cls, directory: Path, physics_model=None) -> "HybridTensionModel":
        """Load saved models."""
        import json

        with open(directory / "hybrid_config.json") as f:
            meta = json.load(f)

        model = cls(physics_model)

        for horizon in meta["horizons"]:
            path = directory / f"residual_model_{horizon}h.pkl"
            if path.exists():
                model.residual_learners[horizon] = ResidualLearner.load(path)

        model.is_fitted = bool(model.residual_learners)
        return model
