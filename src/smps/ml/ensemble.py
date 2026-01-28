"""
Stacking Ensemble for Soil Moisture Prediction.

This module implements a multi-layer stacking ensemble that combines
multiple base learners (LightGBM, XGBoost) with a meta-learner.

Stacking Architecture:
─────────────────────────────────────────────────────────────────
                        META-LEARNER (Ridge/XGB)
                              ▲
                              │ Out-of-fold predictions
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

Benefits:
- Combines strengths of different algorithms
- Reduces overfitting through out-of-fold predictions
- Handles diverse feature types (temporal, spatial, physics)

Research References:
- Wolpert (1992): Stacked Generalization
- Breiman (1996): Stacked Regressions

Usage:
------
>>> from smps.ml.ensemble import StackingEnsemble, EnsembleConfig
>>>
>>> config = EnsembleConfig(
...     base_models=['lightgbm', 'xgboost'],
...     meta_model='ridge',
...     n_folds=5
... )
>>> ensemble = StackingEnsemble(config)
>>> ensemble.fit(X_train, y_train)
>>> predictions = ensemble.predict(X_test)
"""

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit

logger = logging.getLogger("smps.ml.ensemble")


@dataclass
class BaseModelConfig:
    """Configuration for a base model in the ensemble."""

    name: str  # Unique identifier
    model_type: str  # 'lightgbm', 'xgboost', 'randomforest', 'ridge'
    params: Dict[str, Any] = field(default_factory=dict)

    # Feature selection
    feature_fraction: float = 1.0  # Fraction of features to use
    feature_include: Optional[List[str]] = None  # Features to always include
    feature_exclude: Optional[List[str]] = None  # Features to exclude

    # Training configuration
    early_stopping_rounds: int = 50
    verbose: bool = False


@dataclass
class EnsembleConfig:
    """Configuration for stacking ensemble."""

    # Base models
    base_models: List[Union[str, BaseModelConfig]] = field(
        default_factory=lambda: ['lightgbm', 'xgboost']
    )

    # Meta-learner
    meta_model: str = 'ridge'  # 'ridge', 'lightgbm', 'xgboost', 'linear'
    meta_params: Dict[str, Any] = field(default_factory=dict)

    # Cross-validation for stacking
    n_folds: int = 5
    cv_type: str = 'timeseries'  # 'kfold', 'timeseries'

    # Feature engineering for meta-learner
    include_original_features: bool = False  # Include raw features in meta
    include_feature_interactions: bool = True  # Add base model interactions

    # Prediction aggregation
    use_soft_voting: bool = False  # Average instead of stack

    # Random state
    random_state: int = 42


class BaseModelWrapper:
    """Wrapper for base models with consistent interface."""

    def __init__(self, config: BaseModelConfig):
        self.config = config
        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.selected_features: Optional[List[str]] = None

    def _create_model(self):
        """Create the underlying model based on type."""
        if self.config.model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'verbosity': -1,
                    'n_estimators': 500,
                    'learning_rate': 0.05,
                    'max_depth': 8,
                    'num_leaves': 31,
                    'min_child_samples': 20,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'random_state': 42,
                    **self.config.params,
                }
                self.model = lgb.LGBMRegressor(**params)
            except ImportError as exc:
                raise ImportError(
                    "LightGBM not installed: pip install lightgbm") from exc

        elif self.config.model_type == 'xgboost':
            try:
                import xgboost as xgb
                params = {
                    'objective': 'reg:squarederror',
                    'n_estimators': 500,
                    'learning_rate': 0.05,
                    'max_depth': 8,
                    'min_child_weight': 3,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'random_state': 42,
                    'verbosity': 0,
                    **self.config.params,
                }
                self.model = xgb.XGBRegressor(**params)
            except ImportError as exc:
                raise ImportError(
                    "XGBoost not installed: pip install xgboost") from exc

        elif self.config.model_type == 'randomforest':
            from sklearn.ensemble import RandomForestRegressor
            params = {
                'n_estimators': 200,
                'max_depth': 12,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1,
                **self.config.params,
            }
            self.model = RandomForestRegressor(**params)

        elif self.config.model_type == 'ridge':
            from sklearn.linear_model import Ridge
            params = {'alpha': 1.0, **self.config.params}
            self.model = Ridge(**params)

        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features based on config."""
        if self.selected_features is not None:
            return X[self.selected_features]

        available = X.columns.tolist()

        # Start with all features
        selected = set(available)

        # Apply exclusions
        if self.config.feature_exclude:
            selected -= set(self.config.feature_exclude)

        # Apply inclusions (these must be present)
        if self.config.feature_include:
            selected |= set(self.config.feature_include) & set(available)

        # Apply feature fraction
        if self.config.feature_fraction < 1.0:
            n_select = max(
                1, int(len(selected) * self.config.feature_fraction))
            rng = np.random.default_rng(42)

            # Keep included features, randomly sample others
            must_include = set(self.config.feature_include or []) & selected
            optional = selected - must_include

            if len(optional) > n_select - len(must_include):
                sampled = set(rng.choice(list(optional),
                              n_select - len(must_include), replace=False))
                selected = must_include | sampled

        self.selected_features = list(selected)
        self.feature_names = self.selected_features

        return X[self.selected_features]

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        """Fit the base model."""
        self._create_model()

        X_selected = self._select_features(X)

        # Models with early stopping
        if self.config.model_type in ['lightgbm', 'xgboost'] and X_val is not None:
            X_val_selected = X_val[self.selected_features]

            if self.config.model_type == 'lightgbm':
                self.model.fit(
                    X_selected, y,
                    eval_set=[(X_val_selected, y_val)],
                    callbacks=[
                        self._lgb_early_stopping(
                            self.config.early_stopping_rounds)
                    ],
                )
            else:
                self.model.fit(
                    X_selected, y,
                    eval_set=[(X_val_selected, y_val)],
                )
        else:
            self.model.fit(X_selected, y)

        return self

    def _lgb_early_stopping(self, rounds):
        """Create LightGBM early stopping callback."""
        try:
            import lightgbm as lgb
            return lgb.early_stopping(rounds, verbose=self.config.verbose)
        except (ImportError, AttributeError):
            return None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_selected = X[self.selected_features]
        return self.model.predict(X_selected)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_names, importance))
        return {}


class StackingEnsemble:
    """
    Multi-layer stacking ensemble for soil moisture prediction.

    Combines multiple base learners with a meta-learner:
    1. Base models make out-of-fold predictions via cross-validation
    2. Meta-learner trains on stacked base predictions
    3. Final prediction is meta-learner output
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        """
        Initialize stacking ensemble.

        Args:
            config: Ensemble configuration
        """
        self.config = config or EnsembleConfig()

        self.base_models: List[BaseModelWrapper] = []
        self.meta_model = None

        self.oof_predictions: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.meta_feature_names: Optional[List[str]] = None

        self._is_fitted = False

        # Initialize base models
        self._init_base_models()

    def _init_base_models(self):
        """Initialize base model wrappers."""
        for i, model_config in enumerate(self.config.base_models):
            if isinstance(model_config, str):
                # Convert string to config
                config = BaseModelConfig(
                    name=f"{model_config}_{i}",
                    model_type=model_config,
                )
            else:
                config = model_config

            self.base_models.append(BaseModelWrapper(config))

    def _create_cv_splitter(self, _n_samples: int):
        """Create cross-validation splitter."""
        if self.config.cv_type == 'timeseries':
            return TimeSeriesSplit(n_splits=self.config.n_folds)
        else:
            return KFold(
                n_splits=self.config.n_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )

    def _create_meta_model(self):
        """Create the meta-learner model."""
        if self.config.meta_model == 'ridge':
            from sklearn.linear_model import Ridge
            params = {'alpha': 1.0, **self.config.meta_params}
            self.meta_model = Ridge(**params)

        elif self.config.meta_model == 'linear':
            from sklearn.linear_model import LinearRegression
            self.meta_model = LinearRegression(**self.config.meta_params)

        elif self.config.meta_model == 'lightgbm':
            try:
                import lightgbm as lgb
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'verbosity': -1,
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'num_leaves': 8,
                    **self.config.meta_params,
                }
                self.meta_model = lgb.LGBMRegressor(**params)
            except ImportError as exc:
                raise ImportError("LightGBM not installed") from exc

        elif self.config.meta_model == 'xgboost':
            try:
                import xgboost as xgb
                params = {
                    'objective': 'reg:squarederror',
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    **self.config.meta_params,
                }
                self.meta_model = xgb.XGBRegressor(**params)
            except ImportError as exc:
                raise ImportError("XGBoost not installed") from exc

        else:
            raise ValueError(f"Unknown meta model: {self.config.meta_model}")

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        _sample_weight: Optional[np.ndarray] = None,
    ) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble.

        Args:
            X: Feature matrix
            y: Target values
            sample_weight: Optional sample weights

        Returns:
            Self
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        if isinstance(y, pd.Series):
            y = y.values

        self.feature_names = X.columns.tolist()
        n_samples = len(X)
        n_base_models = len(self.base_models)

        logger.info(
            "Fitting stacking ensemble with %d base models...", n_base_models)

        # Initialize out-of-fold predictions
        self.oof_predictions = np.zeros((n_samples, n_base_models))

        # Cross-validation splitter
        cv = self._create_cv_splitter(n_samples)

        # Train base models and generate OOF predictions
        for model_idx, base_model in enumerate(self.base_models):
            logger.info(
                "Training base model %d/%d: %s", model_idx + 1, n_base_models, base_model.config.name)

            # Store fold models for later
            fold_models = []

            for _fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y[val_idx]

                # Create fresh model for this fold
                fold_model = deepcopy(base_model)

                # Fit on fold
                fold_model.fit(
                    X_train_fold, y_train_fold,
                    X_val_fold, y_val_fold,
                )

                # Store OOF predictions
                self.oof_predictions[val_idx,
                                     model_idx] = fold_model.predict(X_val_fold)

                fold_models.append(fold_model)

            # Refit on full data for final prediction
            base_model.fit(X, y)

        # Build meta-features
        meta_X = self._build_meta_features(X, self.oof_predictions)
        self.meta_feature_names = list(meta_X.columns)

        # Train meta-learner
        logger.info("Training meta-learner: %s", self.config.meta_model)
        self._create_meta_model()
        self.meta_model.fit(meta_X, y)

        self._is_fitted = True

        # Log OOF performance
        self._log_oof_performance(y)

        return self

    def _build_meta_features(
        self,
        X: pd.DataFrame,
        base_predictions: np.ndarray,
    ) -> pd.DataFrame:
        """Build feature matrix for meta-learner."""
        meta_features = {}

        # Base model predictions
        for i, model in enumerate(self.base_models):
            meta_features[f"pred_{model.config.name}"] = base_predictions[:, i]

        # Feature interactions between base predictions
        if self.config.include_feature_interactions and len(self.base_models) > 1:
            for i in range(len(self.base_models)):
                for j in range(i + 1, len(self.base_models)):
                    name_i = self.base_models[i].config.name
                    name_j = self.base_models[j].config.name

                    # Prediction difference
                    meta_features[f"diff_{name_i}_{name_j}"] = (
                        base_predictions[:, i] - base_predictions[:, j]
                    )

                    # Prediction product
                    meta_features[f"prod_{name_i}_{name_j}"] = (
                        base_predictions[:, i] * base_predictions[:, j]
                    )

        # Optionally include original features
        if self.config.include_original_features:
            # Select top features to avoid dimensionality explosion
            for col in X.columns[:20]:  # Limit to top 20
                meta_features[f"orig_{col}"] = X[col].values

        return pd.DataFrame(meta_features)

    def _log_oof_performance(self, y: np.ndarray):
        """Log out-of-fold performance of base models."""
        from sklearn.metrics import mean_squared_error, r2_score

        logger.info("Out-of-fold performance:")

        for i, model in enumerate(self.base_models):
            oof_pred = self.oof_predictions[:, i]
            rmse = np.sqrt(mean_squared_error(y, oof_pred))
            r2 = r2_score(y, oof_pred)
            logger.info("  %s: RMSE=%.4f, R²=%.4f",
                        model.config.name, rmse, r2)

        # Ensemble mean performance
        mean_pred = self.oof_predictions.mean(axis=1)
        rmse = np.sqrt(mean_squared_error(y, mean_pred))
        r2 = r2_score(y, mean_pred)
        logger.info("  Ensemble (mean): RMSE=%.4f, R²=%.4f", rmse, r2)

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_base_predictions: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions with the ensemble.

        Args:
            X: Feature matrix
            return_base_predictions: Whether to also return base model predictions

        Returns:
            Predictions (and optionally base predictions)
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)

        # Get base model predictions
        base_predictions = np.zeros((len(X), len(self.base_models)))

        for i, model in enumerate(self.base_models):
            base_predictions[:, i] = model.predict(X)

        if self.config.use_soft_voting:
            # Simple averaging
            final_predictions = base_predictions.mean(axis=1)
        else:
            # Stacked prediction via meta-learner
            meta_X = self._build_meta_features(X, base_predictions)
            final_predictions = self.meta_model.predict(meta_X)

        if return_base_predictions:
            return final_predictions, base_predictions

        return final_predictions

    def get_base_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from each base model."""
        importance = {}

        for model in self.base_models:
            importance[model.config.name] = model.get_feature_importance()

        return importance

    def get_aggregated_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance across all base models."""
        all_importance = self.get_base_feature_importance()

        # Aggregate across models
        aggregated = {}

        for _model_name, model_importance in all_importance.items():
            for feature, imp in model_importance.items():
                if feature not in aggregated:
                    aggregated[feature] = []
                aggregated[feature].append(imp)

        # Average
        return {f: np.mean(imps) for f, imps in aggregated.items()}

    def save(self, path: Path):
        """Save ensemble to disk."""
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        import json
        with open(path / "config.json", 'w', encoding='utf-8') as f:
            json.dump(self.config.__dict__, f, default=str, indent=2)

        # Save models
        for i, model in enumerate(self.base_models):
            joblib.dump(model, path / f"base_model_{i}.joblib")

        joblib.dump(self.meta_model, path / "meta_model.joblib")

        # Save metadata
        metadata = {
            "feature_names": self.feature_names,
            "meta_feature_names": self.meta_feature_names,
        }
        with open(path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f)

        logger.info("Saved ensemble to %s", path)

    @classmethod
    def load(cls, path: Path) -> 'StackingEnsemble':
        """Load ensemble from disk."""
        import joblib
        import json

        path = Path(path)

        # Load config
        with open(path / "config.json", encoding='utf-8') as f:
            config_dict = json.load(f)

        # Reconstruct config
        config = EnsembleConfig()
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)

        ensemble = cls(config)

        # Load base models
        ensemble.base_models = []
        i = 0
        while (path / f"base_model_{i}.joblib").exists():
            model = joblib.load(path / f"base_model_{i}.joblib")
            ensemble.base_models.append(model)
            i += 1

        # Load meta model
        ensemble.meta_model = joblib.load(path / "meta_model.joblib")

        # Load metadata
        with open(path / "metadata.json", encoding='utf-8') as f:
            metadata = json.load(f)

        ensemble.feature_names = metadata["feature_names"]
        ensemble.meta_feature_names = metadata["meta_feature_names"]
        ensemble._is_fitted = True

        logger.info("Loaded ensemble from %s", path)

        return ensemble


class HybridStackingEnsemble(StackingEnsemble):
    """
    Stacking ensemble specialized for physics-hybrid residual learning.

    Extends base stacking with:
    - Physics-aware training (residual targets)
    - Bounded predictions (physical constraints)
    - Multi-depth outputs
    """

    def __init__(
        self,
        config: Optional[EnsembleConfig] = None,
        physics_columns: Optional[List[str]] = None,
        target_bounds: Tuple[float, float] = (0.0, 1.0),
    ):
        """
        Initialize hybrid stacking ensemble.

        Args:
            config: Ensemble configuration
            physics_columns: Columns containing physics predictions
            target_bounds: (min, max) bounds for predictions
        """
        super().__init__(config)
        self.physics_columns = physics_columns or []
        self.target_bounds = target_bounds

        self._physics_scale: Dict[str, Tuple[float, float]] = {}

    def fit_residual(
        self,
        X: pd.DataFrame,
        y_obs: np.ndarray,
        physics_col: str,
    ) -> 'HybridStackingEnsemble':
        """
        Fit ensemble on residuals (observation - physics).

        Args:
            X: Feature matrix (must include physics_col)
            y_obs: Observed target values
            physics_col: Column name for physics predictions

        Returns:
            Self
        """
        if physics_col not in X.columns:
            raise ValueError(f"Physics column '{physics_col}' not in X")

        # Compute residuals
        physics_pred = X[physics_col].values
        residuals = y_obs - physics_pred

        # Store physics scale for bounded predictions
        self._physics_scale[physics_col] = (
            physics_pred.min(), physics_pred.max())

        logger.info(
            "Training ensemble on residuals (mean=%.4f, std=%.4f)", residuals.mean(), residuals.std())

        # Remove physics column from features
        X_features = X.drop(columns=[physics_col])

        return self.fit(X_features, residuals)

    def predict_with_physics(
        self,
        X: pd.DataFrame,
        physics_col: str,
    ) -> np.ndarray:
        """
        Make bounded predictions: physics + ML residual.

        Args:
            X: Feature matrix (must include physics_col)
            physics_col: Column name for physics predictions

        Returns:
            Bounded predictions
        """
        if physics_col not in X.columns:
            raise ValueError(f"Physics column '{physics_col}' not in X")

        physics_pred = X[physics_col].values

        # Remove physics column for ML prediction
        X_features = X.drop(columns=[physics_col])

        # Get residual prediction
        residual_pred = self.predict(X_features)

        # Combine: prediction = physics + residual
        final_pred = physics_pred + residual_pred

        # Apply bounds
        final_pred = np.clip(final_pred, *self.target_bounds)

        return final_pred


class MultiDepthEnsemble:
    """
    Multi-output ensemble for predicting soil moisture at multiple depths.

    Trains separate ensembles for each depth level with shared features.
    """

    def __init__(
        self,
        depths: List[str] = None,
        config: Optional[EnsembleConfig] = None,
    ):
        """
        Initialize multi-depth ensemble.

        Args:
            depths: List of depth identifiers (e.g., ['surface', 'root', 'deep'])
            config: Shared ensemble config
        """
        self.depths = depths or ['surface', 'root', 'deep']
        self.config = config or EnsembleConfig()

        self.ensembles: Dict[str, HybridStackingEnsemble] = {}

        for depth in self.depths:
            self.ensembles[depth] = HybridStackingEnsemble(
                config=deepcopy(self.config),
                target_bounds=(0.0, 0.6),  # Typical soil moisture bounds
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: Dict[str, np.ndarray],
        physics_cols: Dict[str, str],
    ) -> 'MultiDepthEnsemble':
        """
        Fit ensembles for all depths.

        Args:
            X: Feature matrix
            y: Dict mapping depth to observed target
            physics_cols: Dict mapping depth to physics column name

        Returns:
            Self
        """
        for depth in self.depths:
            if depth not in y:
                logger.warning("No target for depth '%s', skipping", depth)
                continue

            if depth not in physics_cols:
                logger.warning(
                    "No physics column for depth '%s', skipping", depth)
                continue

            logger.info("Training ensemble for depth: %s", depth)

            self.ensembles[depth].fit_residual(
                X, y[depth], physics_cols[depth]
            )

        return self

    def predict(
        self,
        X: pd.DataFrame,
        physics_cols: Dict[str, str],
    ) -> Dict[str, np.ndarray]:
        """
        Predict for all depths.

        Args:
            X: Feature matrix
            physics_cols: Dict mapping depth to physics column name

        Returns:
            Dict mapping depth to predictions
        """
        predictions = {}

        for depth, ensemble in self.ensembles.items():
            if depth in physics_cols:
                predictions[depth] = ensemble.predict_with_physics(
                    X, physics_cols[depth]
                )

        return predictions

    def get_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for each depth."""
        return {
            depth: ensemble.get_aggregated_importance()
            for depth, ensemble in self.ensembles.items()
        }
