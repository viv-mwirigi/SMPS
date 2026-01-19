"""
Hybrid Physics-ML Model for Soil Moisture Prediction.

This module implements the hybrid approach:
1. Physics model generates baseline predictions at multiple depths
2. ML model (LightGBM/XGBoost) learns the residuals
3. Final prediction = Physics baseline + ML residual correction

Research Background:
- Karpatne et al. (2017): Theory-guided data science
- Reichstein et al. (2019): Deep learning for Earth sciences
- Fang et al. (2019): LSTM for soil moisture with physics constraints
- Kraft et al. (2022): Hybrid modeling in hydrology

Key Benefits:
- Physics provides interpretable baseline with water balance constraints
- ML captures complex nonlinear patterns physics misses
- Better extrapolation than pure ML (physics anchors predictions)
- Reduced data requirements (physics encodes domain knowledge)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger("smps.ml.hybrid_model")


@dataclass
class PhysicsResidualTarget:
    """
    Target configuration for residual learning.

    The ML model predicts:
        residual = observation - physics_prior

    Final prediction:
        prediction = physics_prior + ML_residual
    """
    target_depth: str  # 'surface', 'root', 'deep'
    observation_col: str  # Column name for observations
    physics_col: str  # Column name for physics prior
    residual_col: str  # Column name for residual (computed)

    # Bounds for predictions
    min_value: float = 0.0
    max_value: float = 1.0

    # Weight in multi-target scenarios
    weight: float = 1.0


@dataclass
class ResidualLearnerConfig:
    """Configuration for the residual learning model."""

    # Model type
    model_type: str = "lightgbm"  # 'lightgbm', 'xgboost', 'catboost'

    # LightGBM parameters
    lightgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 8,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    })

    # XGBoost parameters
    xgboost_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
        "random_state": 42,
    })

    # Training parameters
    n_estimators: int = 1000
    early_stopping_rounds: int = 50
    validation_fraction: float = 0.2

    # Feature selection
    max_features: int = 100
    min_feature_importance: float = 0.001

    # Physics constraint
    enforce_bounds: bool = True
    residual_clip_percentile: float = 99.0


class ResidualLearner:
    """
    Learns residuals between physics model and observations.

    The residual learner captures patterns that the physics model misses:
    - Systematic biases in physics parameterization
    - Local effects not in physics (land cover, microclimate)
    - Temporal dynamics beyond physics time constants
    - Complex feature interactions
    """

    def __init__(self, config: Optional[ResidualLearnerConfig] = None):
        """
        Initialize residual learner.

        Args:
            config: Model configuration
        """
        self.config = config or ResidualLearnerConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.feature_importance: Dict[str, float] = {}

        # Training metadata
        self.training_info: Dict[str, Any] = {}
        self.is_fitted = False

    def prepare_residual_target(
        self,
        df: pd.DataFrame,
        target: PhysicsResidualTarget,
    ) -> pd.DataFrame:
        """
        Compute residual target from observations and physics.

        Args:
            df: DataFrame with observations and physics predictions
            target: Target configuration

        Returns:
            DataFrame with residual column added
        """
        result = df.copy()

        # Compute residual where observations exist
        mask = result[target.observation_col].notna()

        result[target.residual_col] = np.nan
        result.loc[mask, target.residual_col] = (
            result.loc[mask, target.observation_col] -
            result.loc[mask, target.physics_col]
        )

        # Clip extreme residuals
        if self.config.residual_clip_percentile < 100:
            valid_residuals = result[target.residual_col].dropna()
            if len(valid_residuals) > 10:
                lower = np.percentile(
                    valid_residuals, 100 - self.config.residual_clip_percentile)
                upper = np.percentile(
                    valid_residuals, self.config.residual_clip_percentile)
                result[target.residual_col] = result[target.residual_col].clip(
                    lower, upper)

        logger.info(
            "Computed residuals for %s: mean=%.4f, std=%.4f",
            target.target_depth,
            float(result[target.residual_col].mean()),
            float(result[target.residual_col].std()),
        )

        return result

    def select_features(
        self,
        df: pd.DataFrame,
        exclude_cols: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Select features for training.

        Args:
            df: DataFrame with all features
            exclude_cols: Columns to exclude

        Returns:
            List of selected feature names
        """
        exclude = set(exclude_cols or [])
        exclude.update(['site_id', 'date', 'quality_flag'])

        # Select numeric columns only
        features = []
        for col in df.columns:
            if col in exclude:
                continue
            if col.startswith('obs_'):  # Exclude observations (data leakage)
                continue
            if col.endswith('_residual'):  # Exclude target residuals
                continue

            if np.issubdtype(df[col].dtype, np.number):
                features.append(col)

        # Limit to max features
        if len(features) > self.config.max_features:
            # Use variance as proxy for informativeness
            variances = df[features].var().sort_values(ascending=False)
            features = variances.head(self.config.max_features).index.tolist()

        return features

    def fit(
        self,
        df: pd.DataFrame,
        target: PhysicsResidualTarget,
        feature_names: Optional[List[str]] = None,
        validation_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Fit the residual learner.

        Args:
            df: Training DataFrame
            target: Target configuration
            feature_names: Optional list of features to use
            validation_df: Optional separate validation set

        Returns:
            Training metrics and info
        """
        # Prepare residual target
        df = self.prepare_residual_target(df, target)

        # Select features
        self.feature_names = feature_names or self.select_features(
            df, exclude_cols=[target.observation_col, target.physics_col]
        )

        # Filter to rows with valid residuals
        train_mask = df[target.residual_col].notna()
        train_df = df[train_mask].copy()

        if len(train_df) < 50:
            raise ValueError(
                f"Insufficient training data: {len(train_df)} rows")

        X = train_df[self.feature_names].values
        y = train_df[target.residual_col].values

        # Handle missing features
        X = np.nan_to_num(X, nan=0.0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split for validation if not provided
        if validation_df is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y,
                test_size=self.config.validation_fraction,
                random_state=42
            )
        else:
            validation_df = self.prepare_residual_target(validation_df, target)
            val_mask = validation_df[target.residual_col].notna()
            X_val = self.scaler.transform(
                np.nan_to_num(
                    validation_df.loc[val_mask, self.feature_names].values)
            )
            y_val = validation_df.loc[val_mask, target.residual_col].values
            X_train, y_train = X_scaled, y

        # Train model
        if self.config.model_type == "lightgbm":
            self._fit_lightgbm(X_train, y_train, X_val, y_val)
        elif self.config.model_type == "xgboost":
            self._fit_xgboost(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        # Compute feature importance
        self._compute_feature_importance()

        # Compute training metrics
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)

        self.training_info = {
            "model_type": self.config.model_type,
            "n_features": len(self.feature_names),
            "n_train_samples": len(y_train),
            "n_val_samples": len(y_val),
            "target_depth": target.target_depth,
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "train_mae": mean_absolute_error(y_train, y_train_pred),
            "train_r2": r2_score(y_train, y_train_pred),
            "val_rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
            "val_mae": mean_absolute_error(y_val, y_val_pred),
            "val_r2": r2_score(y_val, y_val_pred),
            "fitted_at": datetime.now().isoformat(),
        }

        self.is_fitted = True

        logger.info(
            "Trained %s residual learner: val_RMSE=%.4f, val_R²=%.3f",
            self.config.model_type,
            float(self.training_info['val_rmse']),
            float(self.training_info['val_r2']),
        )

        return self.training_info

    def _fit_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        """Fit LightGBM model."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM not installed. Run: pip install lightgbm")

        params = self.config.lightgbm_params.copy()

        train_data = lgb.Dataset(
            X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=self.config.early_stopping_rounds),
                lgb.log_evaluation(period=100),
            ],
        )

    def _fit_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        """Fit XGBoost model."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost not installed. Run: pip install xgboost")

        params = self.config.xgboost_params.copy()

        dtrain = xgb.DMatrix(X_train, label=y_train,
                             feature_names=self.feature_names)
        dval = xgb.DMatrix(X_val, label=y_val,
                           feature_names=self.feature_names)

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.n_estimators,
            evals=[(dtrain, 'train'), (dval, 'valid')],
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose_eval=100,
        )

    def _compute_feature_importance(self):
        """Compute and store feature importance."""
        if self.config.model_type == "lightgbm":
            importance = self.model.feature_importance(importance_type='gain')
        elif self.config.model_type == "xgboost":
            importance = list(self.model.get_score(
                importance_type='gain').values())
            # Align with feature names
            importance_dict = self.model.get_score(importance_type='gain')
            importance = [importance_dict.get(f, 0)
                          for f in self.feature_names]
        else:
            importance = [0] * len(self.feature_names)

        # Normalize
        total = sum(importance) if sum(importance) > 0 else 1
        importance = [i / total for i in importance]

        self.feature_importance = dict(zip(self.feature_names, importance))

    def predict(
        self,
        df: pd.DataFrame,
        target: PhysicsResidualTarget,
    ) -> np.ndarray:
        """
        Predict residuals.

        Args:
            df: DataFrame with features
            target: Target configuration

        Returns:
            Array of predicted residuals
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = df[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)

        if self.config.model_type == "xgboost":
            import xgboost as xgb
            dtest = xgb.DMatrix(X_scaled, feature_names=self.feature_names)
            residuals = self.model.predict(dtest)
        else:
            residuals = self.model.predict(X_scaled)

        return residuals

    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get top N features by importance."""
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]


class HybridSoilMoistureModel:
    """
    Hybrid Physics-ML model for soil moisture prediction.

    Architecture:
    ------------
    1. Physics Model (EnhancedWaterBalance):
       - 5-layer soil model with full physics
       - Green-Ampt infiltration, FAO-56 ET, Darcy flux
       - Generates physics priors at surface, root, deep

    2. ML Residual Learner:
       - Learns residuals (obs - physics)
       - Separate models for each depth
       - Features: weather, physics states, remote sensing, static

    3. Prediction Combination:
       - final_prediction = physics_prior + ml_residual
       - Physics anchors predictions (physical bounds)
       - ML corrects systematic biases

    Benefits:
    - Better extrapolation than pure ML
    - Interpretable physics baseline
    - Reduced data requirements
    - Physical consistency (mass balance)
    """

    def __init__(
        self,
        targets: Optional[List[PhysicsResidualTarget]] = None,
        residual_config: Optional[ResidualLearnerConfig] = None,
    ):
        """
        Initialize hybrid model.

        Args:
            targets: List of targets for multi-depth prediction
            residual_config: Configuration for residual learners
        """
        # Default targets for 3 depths
        self.targets = targets or [
            PhysicsResidualTarget(
                target_depth="surface",
                observation_col="obs_vwc_surface",
                physics_col="physics_theta_surface",
                residual_col="residual_surface",
                weight=1.0,
            ),
            PhysicsResidualTarget(
                target_depth="root",
                observation_col="obs_vwc_root",
                physics_col="physics_theta_root",
                residual_col="residual_root",
                weight=1.5,  # Root zone often most important
            ),
            PhysicsResidualTarget(
                target_depth="deep",
                observation_col="obs_vwc_deep",
                physics_col="physics_theta_deep",
                residual_col="residual_deep",
                weight=0.5,
            ),
        ]

        self.residual_config = residual_config or ResidualLearnerConfig()

        # One residual learner per target
        self.residual_learners: Dict[str, ResidualLearner] = {}

        # Model metadata
        self.model_info: Dict[str, Any] = {}
        self.is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        targets: Optional[List[PhysicsResidualTarget]] = None,
        cv_folds: int = 0,
    ) -> Dict[str, Any]:
        """
        Fit the hybrid model.

        Args:
            df: Training DataFrame with features, observations, and physics
            targets: Override target configuration
            cv_folds: Number of cross-validation folds (0 for no CV)

        Returns:
            Training metrics
        """
        targets = targets or self.targets
        all_metrics = {}

        for target in targets:
            # Check if observation column exists and has data
            if target.observation_col not in df.columns:
                logger.warning(
                    "Observation column %s not found, skipping", target.observation_col)
                continue

            if df[target.observation_col].notna().sum() < 50:
                logger.warning(
                    "Insufficient observations for %s, skipping", target.target_depth)
                continue

            logger.info("Training residual learner for %s",
                        target.target_depth)

            learner = ResidualLearner(config=self.residual_config)

            if cv_folds > 0:
                # Cross-validation
                metrics = self._cross_validate(df, learner, target, cv_folds)
            else:
                # Single train/val split
                metrics = learner.fit(df, target)

            self.residual_learners[target.target_depth] = learner
            all_metrics[target.target_depth] = metrics

        # Store model info
        self.model_info = {
            "n_targets": len(self.residual_learners),
            "targets": [t.target_depth for t in targets if t.target_depth in self.residual_learners],
            "model_type": self.residual_config.model_type,
            "metrics": all_metrics,
            "fitted_at": datetime.now().isoformat(),
        }

        self.is_fitted = True

        return all_metrics

    def _cross_validate(
        self,
        df: pd.DataFrame,
        learner: ResidualLearner,
        target: PhysicsResidualTarget,
        n_folds: int,
    ) -> Dict[str, Any]:
        """Run cross-validation and return aggregated metrics."""

        # Prepare residual
        df = learner.prepare_residual_target(df, target)
        mask = df[target.residual_col].notna()
        valid_df = df[mask].copy()

        # Use TimeSeriesSplit for temporal data
        tscv = TimeSeriesSplit(n_splits=n_folds)

        cv_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(valid_df)):
            train_df = valid_df.iloc[train_idx]
            val_df = valid_df.iloc[val_idx]

            fold_learner = ResidualLearner(config=self.residual_config)
            metrics = fold_learner.fit(train_df, target, validation_df=val_df)
            cv_metrics.append(metrics)

        # Aggregate metrics
        agg_metrics = {
            "cv_folds": n_folds,
            "val_rmse_mean": np.mean([m['val_rmse'] for m in cv_metrics]),
            "val_rmse_std": np.std([m['val_rmse'] for m in cv_metrics]),
            "val_r2_mean": np.mean([m['val_r2'] for m in cv_metrics]),
            "val_r2_std": np.std([m['val_r2'] for m in cv_metrics]),
        }

        # Final fit on all data
        learner.fit(df, target)
        agg_metrics.update(learner.training_info)

        return agg_metrics

    def predict(
        self,
        df: pd.DataFrame,
        return_components: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Make predictions using hybrid model.

        Args:
            df: DataFrame with features and physics priors
            return_components: Whether to return physics and residual separately

        Returns:
            DataFrame with predictions (and optionally components)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        predictions = df[['site_id', 'date']].copy(
        ) if 'site_id' in df.columns else df[['date']].copy()

        components = {
            'physics': {},
            'residual': {},
        }

        for target in self.targets:
            if target.target_depth not in self.residual_learners:
                continue

            learner = self.residual_learners[target.target_depth]

            # Get physics prior
            physics_prior = df[target.physics_col].values

            # Predict residual
            residual = learner.predict(df, target)

            # Combine: final = physics + residual
            final_pred = physics_prior + residual

            # Enforce physical bounds
            if self.residual_config.enforce_bounds:
                final_pred = np.clip(
                    final_pred, target.min_value, target.max_value)

            # Store predictions
            pred_col = f"pred_vwc_{target.target_depth}"
            predictions[pred_col] = final_pred

            components['physics'][target.target_depth] = physics_prior
            components['residual'][target.target_depth] = residual

        if return_components:
            physics_df = pd.DataFrame(components['physics'])
            residual_df = pd.DataFrame(components['residual'])
            return predictions, physics_df, residual_df

        return predictions

    def evaluate(
        self,
        df: pd.DataFrame,
        targets: Optional[List[PhysicsResidualTarget]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance against observations.

        Args:
            df: DataFrame with observations
            targets: Override targets

        Returns:
            Dict of metrics per target depth
        """
        targets = targets or self.targets
        predictions = self.predict(df)

        metrics = {}

        for target in targets:
            if target.target_depth not in self.residual_learners:
                continue

            pred_col = f"pred_vwc_{target.target_depth}"
            obs_col = target.observation_col
            physics_col = target.physics_col

            if obs_col not in df.columns:
                continue

            # Filter to valid observations
            mask = df[obs_col].notna()
            y_true = df.loc[mask, obs_col].values
            y_pred = predictions.loc[mask, pred_col].values
            y_physics = df.loc[mask, physics_col].values

            if len(y_true) < 10:
                continue

            # Compute metrics for hybrid model
            hybrid_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            hybrid_mae = mean_absolute_error(y_true, y_pred)
            hybrid_r2 = r2_score(y_true, y_pred)

            # Compute metrics for physics-only baseline
            physics_rmse = np.sqrt(mean_squared_error(y_true, y_physics))
            physics_mae = mean_absolute_error(y_true, y_physics)
            physics_r2 = r2_score(y_true, y_physics)

            # Skill improvement
            rmse_improvement = (physics_rmse - hybrid_rmse) / \
                physics_rmse * 100

            metrics[target.target_depth] = {
                "hybrid_rmse": hybrid_rmse,
                "hybrid_mae": hybrid_mae,
                "hybrid_r2": hybrid_r2,
                "physics_rmse": physics_rmse,
                "physics_mae": physics_mae,
                "physics_r2": physics_r2,
                "rmse_improvement_pct": rmse_improvement,
                "n_samples": len(y_true),
            }

            logger.info(
                f"{target.target_depth}: Hybrid RMSE={hybrid_rmse:.4f} vs "
                f"Physics RMSE={physics_rmse:.4f} ({rmse_improvement:.1f}% improvement)"
            )

        return metrics

    def get_feature_importance(self, depth: str = "root") -> Dict[str, float]:
        """Get feature importance for a specific depth."""
        if depth not in self.residual_learners:
            return {}
        return self.residual_learners[depth].feature_importance

    def save(self, path: Path):
        """Save model to disk."""
        import pickle

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each learner
        for depth, learner in self.residual_learners.items():
            learner_path = path / f"learner_{depth}.pkl"
            with open(learner_path, 'wb') as f:
                pickle.dump(learner, f)

        # Save metadata
        meta_path = path / "model_info.json"
        import json
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.model_info, f, indent=2)

        logger.info("Saved hybrid model to %s", path)

    def load(self, path: Path):
        """Load model from disk."""
        import pickle
        import json

        path = Path(path)

        # Load metadata
        meta_path = path / "model_info.json"
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.model_info = json.load(f)

        # Load learners
        self.residual_learners = {}
        for target in self.targets:
            learner_path = path / f"learner_{target.target_depth}.pkl"
            if learner_path.exists():
                with open(learner_path, 'rb') as f:
                    self.residual_learners[target.target_depth] = pickle.load(
                        f)

        self.is_fitted = True
        logger.info("Loaded hybrid model from %s", path)
