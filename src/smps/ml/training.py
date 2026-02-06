"""
ML Training utilities for SWPPS.

Provides training pipelines with:
- Site-blocked cross-validation
- Residual learning
- Multi-horizon training
- Model evaluation
- Sequential feature engineering for temporal dependencies
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

from smps.ml.experiment_tracking import (
    ExperimentTracker, ExperimentConfig, ModelMetadata,
    HyperparameterOptimizer, ModelVersionManager
)
from smps.ml.uncertainty import PsiUncertaintyQuantifier, UncertaintyConfig
from smps.core.reproducibility import reproducibility_manager

logger = logging.getLogger("swpps.ml.training")


@dataclass
class TrainingConfig:
    """Configuration for ML training."""
    # Model type
    model_type: str = "lightgbm"

    # Cross-validation
    use_site_blocked_cv: bool = True
    n_cv_folds: int = 5
    cv_group_col: str = "station_id"

    # Training parameters
    n_estimators: int = 1000
    learning_rate: float = 0.03
    max_depth: int = 6
    num_leaves: int = 31
    min_child_samples: int = 20

    # Regularization
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0

    # Feature selection
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5

    # Early stopping
    early_stopping_rounds: int = 50

    # Experiment tracking
    enable_experiment_tracking: bool = True
    enable_hyperparameter_optimization: bool = False
    experiment_name: str = "smps_theta_space"
    run_name: Optional[str] = None

    # Uncertainty quantification
    enable_uncertainty: bool = True
    uncertainty_method: str = "ensemble"  # "ensemble", "quantile", "bootstrap"
    n_uncertainty_models: int = 10

    # Random state
    random_state: Optional[int] = None

    def __post_init__(self):
        """Set random state from reproducibility manager if not specified."""
        if self.random_state is None:
            self.random_state = reproducibility_manager.get_seed('sklearn')


@dataclass
class CVFoldResult:
    """Results from a single cross-validation fold."""
    fold: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_score: float
    val_score: float
    feature_importance: Dict[str, float]
    model: Any


class SiteBlockedCV:
    """
    Site-blocked cross-validation splitter.

    Ensures that all data from a site stays together in either train or validation.
    This prevents geographic fingerprinting and provides more realistic evaluation.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state or reproducibility_manager.get_seed(
            'cv_split')

    def split(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into train/validation sets with site blocking.

        Args:
            X: Feature matrix
            y: Target vector
            groups: Group labels (station_ids)

        Yields:
            Train and validation indices for each fold
        """
        unique_groups = groups.unique()
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(unique_groups)

        fold_size = len(unique_groups) // self.n_splits

        for fold in range(self.n_splits):
            start_idx = fold * fold_size
            if fold == self.n_splits - 1:
                end_idx = len(unique_groups)
            else:
                end_idx = (fold + 1) * fold_size

            val_groups = unique_groups[start_idx:end_idx]
            train_groups = np.setdiff1d(unique_groups, val_groups)

            val_mask = groups.isin(val_groups)
            train_mask = groups.isin(train_groups)

            yield train_mask.values, val_mask.values


@dataclass
class HorizonTrainingResult:
    """Results from training a model for a specific forecast horizon."""
    horizon_hours: int
    model: Any
    cv_results: List[CVFoldResult]
    feature_importance: Dict[str, float]
    best_score: float
    training_time_seconds: float


class ResidualTrainer:
    """
    Trainer for residual learning models.

    Learns the difference between physics predictions and observations.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.experiment_tracker = ExperimentTracker(
            ExperimentConfig(experiment_name=self.config.experiment_name)
        ) if self.config.enable_experiment_tracking else None
        self.hyper_optimizer = HyperparameterOptimizer(
            ExperimentConfig()
        ) if self.config.enable_hyperparameter_optimization else None
        self.model_registry = ModelVersionManager()

        # Uncertainty quantification
        self.uncertainty_quantifier = None
        if self.config.enable_uncertainty:
            uncertainty_config = UncertaintyConfig(
                n_ensemble_members=self.config.n_uncertainty_models,
                n_parameter_sets=5,  # For physics uncertainty
            )
            self.uncertainty_quantifier = PsiUncertaintyQuantifier(
                uncertainty_config)

    def train_with_site_cv(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
        feature_cols: List[str],
        n_folds: int = 5,
        horizon_hours: Optional[int] = None,
        data_hash: Optional[str] = None,
    ) -> Tuple[Any, List[CVFoldResult]]:
        """
        Train with site-blocked cross-validation.

        Args:
            X: Feature matrix
            y: Target values
            groups: Group labels for blocking
            feature_cols: Feature column names
            n_folds: Number of CV folds
            horizon_hours: Forecast horizon for logging
            data_hash: Hash of training data for reproducibility

        Returns:
            Best model and fold results
        """
        logger.info(f"Training with {n_folds}-fold site-blocked CV")

        # Start experiment tracking
        run_id = None
        if self.experiment_tracker:
            run_name = f"{self.config.run_name or 'cv_training'}_h{horizon_hours or 'unknown'}"
            run_id = self.experiment_tracker.start_run(run_name)
            self.experiment_tracker.log_params({
                "n_folds": n_folds,
                "n_features": len(feature_cols),
                "n_samples": len(X),
                "horizon_hours": horizon_hours,
                "data_hash": data_hash,
                **self.config.__dict__
            })

        # GroupKFold ensures sites don't leak between train/val
        gkf = GroupKFold(n_splits=n_folds)

        fold_results = []
        best_model = None
        best_val_score = float('inf')

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            logger.info(f"Training fold {fold + 1}/{n_folds}")

            # Verify no site leakage
            train_sites = set(groups[train_idx])
            val_sites = set(groups[val_idx])
            assert len(
                train_sites & val_sites) == 0, f"Site leakage in fold {fold}"

            # Split data
            X_train_fold = X.iloc[train_idx][feature_cols].values
            y_train_fold = y[train_idx]
            X_val_fold = X.iloc[val_idx][feature_cols].values
            y_val_fold = y[val_idx]

            # Train model
            model = self._train_single_model(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, feature_cols
            )

            # Evaluate
            val_pred = model.predict(X_val_fold)
            val_score = np.sqrt(mean_squared_error(y_val_fold, val_pred))

            # Get feature importance
            importance = dict(
                zip(feature_cols, model.feature_importance(importance_type='gain')))

            fold_result = CVFoldResult(
                fold=fold,
                train_indices=train_idx,
                val_indices=val_idx,
                train_score=np.sqrt(mean_squared_error(
                    y_train_fold, model.predict(X_train_fold))),
                val_score=val_score,
                feature_importance=importance,
                model=model,
            )

            fold_results.append(fold_result)

            if val_score < best_val_score:
                best_val_score = val_score
                best_model = model

            logger.info(f"  Fold {fold + 1}: val RMSE = {val_score:.4f}")

        # Log CV summary
        val_scores = [r.val_score for r in fold_results]
        cv_mean = np.mean(val_scores)
        cv_std = np.std(val_scores)
        logger.info(
            f"CV complete: mean RMSE = {cv_mean:.4f} ± {cv_std:.4f}")

        # Log final results to experiment tracker
        if self.experiment_tracker and run_id:
            self.experiment_tracker.log_metrics({
                "cv_mean_rmse": cv_mean,
                "cv_std_rmse": cv_std,
                "best_fold_rmse": best_val_score,
                "n_folds": n_folds,
            })

            # Log best model
            self.experiment_tracker.log_model(
                best_model, "best_model", feature_cols)

            # Create and log metadata
            if horizon_hours and data_hash:
                metadata = ModelMetadata(
                    experiment_id=self.experiment_tracker.config.experiment_name,
                    run_id=run_id,
                    model_name=f"theta_model_{horizon_hours}h",
                    horizon_hours=horizon_hours,
                    feature_columns=feature_cols,
                    training_config=self.config.__dict__,
                    cv_results={
                        "mean_rmse": cv_mean,
                        "std_rmse": cv_std,
                        "best_rmse": best_val_score,
                        "n_folds": n_folds,
                    },
                    # Use first fold as example
                    feature_importance=fold_results[0].feature_importance,
                    data_hash=data_hash,
                )
                self.experiment_tracker.log_metadata(metadata)
                self.model_registry.register_model(metadata)

            self.experiment_tracker.end_run()

        # Fit uncertainty quantifier on full training data
        if self.uncertainty_quantifier:
            logger.info("Fitting uncertainty quantifier...")
            # Convert back to DataFrame for uncertainty fitting
            X_train_full = X[feature_cols]
            self.uncertainty_quantifier.fit(X_train_full, y)

            # Log uncertainty statistics
            uncertainty_stats = self.uncertainty_quantifier.get_uncertainty_stats(
                X_train_full)
            logger.info(f"Uncertainty stats: mean={uncertainty_stats['mean_psi_uncertainty']:.4f}, "
                        f"reliable={uncertainty_stats['reliable_predictions_pct']:.1f}%")

            if self.experiment_tracker and run_id:
                self.experiment_tracker.start_run(f"{run_name}_uncertainty")
                self.experiment_tracker.log_metrics(uncertainty_stats)
                self.experiment_tracker.end_run()

        return best_model, fold_results

    def _train_single_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
    ) -> lgb.Booster:
        """Train a single LightGBM model."""
        # Prepare datasets
        train_data = lgb.Dataset(
            X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'max_depth': self.config.max_depth,
            'num_leaves': self.config.num_leaves,
            'learning_rate': self.config.learning_rate,
            'feature_fraction': self.config.feature_fraction,
            'bagging_fraction': self.config.bagging_fraction,
            'bagging_freq': self.config.bagging_freq,
            'min_child_samples': self.config.min_child_samples,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'verbose': -1,
            'random_state': self.config.random_state,
        }

        # Train
        model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.n_estimators,
            valid_sets=[val_data],
            valid_names=['val'],
            callbacks=[lgb.early_stopping(
                self.config.early_stopping_rounds, verbose=False)],
        )

        return model

    def predict_with_uncertainty(
        self,
        model: Any,
        X: pd.DataFrame,
        feature_cols: List[str],
        physics_uncertainty: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[List]]:
        """
        Make predictions with uncertainty quantification.

        Args:
            model: Trained model
            X: Feature matrix
            feature_cols: Feature column names
            physics_uncertainty: Optional physics parameter uncertainty

        Returns:
            Tuple of (predictions, uncertainty_results)
        """
        # Base predictions
        predictions = model.predict(X[feature_cols])

        # Uncertainty quantification
        uncertainty_results = None
        if self.uncertainty_quantifier:
            uncertainty_results = self.uncertainty_quantifier.predict_uncertainty(
                X[feature_cols], physics_uncertainty
            )

        return predictions, uncertainty_results

    def train_multi_horizon(
        self,
        df: pd.DataFrame,
        horizons: List[int],
        feature_cols: List[str],
        target_prefix: str = "residual_target",
    ) -> Dict[int, Tuple[Any, List[CVFoldResult]]]:
        """
        Train separate models for each forecast horizon.

        Args:
            df: Training dataframe with horizon targets
            horizons: List of forecast horizons (hours)
            feature_cols: Feature column names
            target_prefix: Prefix for target columns

        Returns:
            Dict mapping horizon to (model, fold_results)
        """
        logger.info(f"Training multi-horizon models for horizons: {horizons}")

        horizon_models = {}

        for horizon in horizons:
            logger.info(f"Training horizon {horizon}h")

            target_col = f"{target_prefix}_{horizon}h"

            # Filter valid data
            valid_mask = df[target_col].notna()
            df_valid = df[valid_mask].copy()

            if len(df_valid) < 100:
                logger.warning(
                    f"Insufficient data for horizon {horizon}h: {len(df_valid)} samples")
                continue

            # Prepare data
            X = df_valid[feature_cols]
            y = df_valid[target_col].values
            groups = df_valid[self.config.cv_group_col].values

            # Train with CV
            model, fold_results = self.train_with_site_cv(
                X, y, groups, feature_cols, self.config.n_cv_folds
            )

            horizon_models[horizon] = (model, fold_results)

        return horizon_models


def create_residual_targets(
    df: pd.DataFrame,
    horizons: List[int],
    physics_col: str = "physics_prior",
    observed_col: str = "soil_moisture",
    group_cols: Optional[List[str]] = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Create residual targets for multi-horizon training.

    Args:
        df: DataFrame with physics predictions and observations
        horizons: Forecast horizons in hours
        physics_col: Physics prediction column
        observed_col: Observed soil moisture column

    Returns:
        DataFrame with residual targets
    """
    result = df.copy()

    if group_cols is None:
        group_cols = ["station_id"]

    if date_col in result.columns:
        result[date_col] = pd.to_datetime(result[date_col])

        key_cols = list(group_cols) + [date_col]
        rhs = result[key_cols + [observed_col, physics_col]].copy()

        # Guard against duplicate keys creating cartesian products on merge
        rhs = rhs.drop_duplicates(subset=key_cols)

        for horizon in horizons:
            future_date_col = f"{date_col}_plus_{horizon}h"
            target_col = f"target_{horizon}h"
            physics_future_col = f"physics_{horizon}h"
            residual_col = f"residual_target_{horizon}h"

            result[future_date_col] = result[date_col] + \
                pd.Timedelta(hours=int(horizon))

            merge_left = list(group_cols) + [future_date_col]
            rhs_renamed = rhs.rename(
                columns={
                    date_col: future_date_col,
                    observed_col: target_col,
                    physics_col: physics_future_col,
                }
            )

            result = result.merge(rhs_renamed, how="left", on=merge_left)
            result[residual_col] = result[target_col] - \
                result[physics_future_col]

        return result

    # Fallback: legacy behavior when no date column is available
    for horizon in horizons:
        future_obs = result.groupby(group_cols)[observed_col].shift(-horizon)
        future_physics = result.groupby(
            group_cols)[physics_col].shift(-horizon)
        result[f"target_{horizon}h"] = future_obs
        result[f"physics_{horizon}h"] = future_physics
        result[f"residual_target_{horizon}h"] = future_obs - future_physics

    return result


def create_matric_residual_targets(
    df: pd.DataFrame,
    horizons: List[int],
    physics_col: str = "physics_prior",
    observed_col: str = "soil_moisture",
    observed_matric_col: Optional[str] = None,
    group_cols: Optional[List[str]] = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Create residual targets in matric potential space for multi-horizon training.

    Converts volumetric observations to matric potential space and uses physics predictions
    (already in matric potential space) to create residuals.

    Args:
        df: DataFrame with physics predictions, observations, and soil parameters
        horizons: Forecast horizons in hours
        physics_col: Physics prediction column (matric potential)
        observed_col: Observed soil moisture column (volumetric)

    Returns:
        DataFrame with matric potential residual targets
    """
    result = df.copy()

    if group_cols is None:
        group_cols = ["station_id"]

    if date_col in result.columns:
        result[date_col] = pd.to_datetime(result[date_col])

    # Convert observed volumetric to matric potential (or reuse precomputed)
    if observed_matric_col is not None and observed_matric_col in result.columns:
        result['observed_matric'] = result[observed_matric_col]
    else:
        result['observed_matric'] = _convert_series_to_matric_potential(
            result[observed_col], result['station_id'], result)

    # Physics predictions are already in matric potential space
    result['physics_matric'] = result[physics_col]

    if date_col in result.columns:
        key_cols = list(group_cols) + [date_col]
        rhs = result[key_cols + [observed_col,
                                 "observed_matric", "physics_matric"]].copy()
        rhs = rhs.drop_duplicates(subset=key_cols)

        for horizon in horizons:
            future_date_col = f"{date_col}_plus_{horizon}h"
            target_col = f"target_{horizon}h"
            theta_target_col = f"theta_target_{horizon}h"
            physics_future_col = f"physics_{horizon}h"

            result[future_date_col] = result[date_col] + \
                pd.Timedelta(hours=int(horizon))
            merge_left = list(group_cols) + [future_date_col]
            rhs_renamed = rhs.rename(
                columns={
                    date_col: future_date_col,
                    observed_col: theta_target_col,
                    "observed_matric": target_col,
                    "physics_matric": physics_future_col,
                }
            )
            result = result.merge(rhs_renamed, how="left", on=merge_left)
            result[f"residual_target_{horizon}h"] = result[target_col] - \
                result[physics_future_col]

        return result

    for horizon in horizons:
        future_obs_matric = result.groupby(
            group_cols)["observed_matric"].shift(-horizon)
        future_obs_theta = result.groupby(
            group_cols)[observed_col].shift(-horizon)
        future_physics_matric = result.groupby(
            group_cols)["physics_matric"].shift(-horizon)
        result[f"target_{horizon}h"] = future_obs_matric
        result[f"theta_target_{horizon}h"] = future_obs_theta
        result[f"physics_{horizon}h"] = future_physics_matric
        result[f"residual_target_{horizon}h"] = future_obs_matric - \
            future_physics_matric

    return result


def compute_site_bias_corrections(
    df: pd.DataFrame,
    physics_col: str = "physics_prior_surface",
    observed_col: str = "soil_moisture",
    observed_matric_col: Optional[str] = None,
    clip_bias: float = 500.0
) -> Dict[str, float]:
    """
    Compute site-level bias corrections for matric potential predictions.

    The physics model often has systematic biases that vary by site due to:
    - Local soil property differences not captured by remote sensing
    - Microclimate effects
    - Sensor calibration differences

    This function computes a mean bias correction per site that can be applied
    before ML residual learning, helping the ML focus on temporal patterns
    rather than large systematic biases.

    Args:
        df: DataFrame with physics predictions and observations
        physics_col: Column name for physics prior (matric potential)
        observed_col: Column name for observed soil moisture (volumetric)
        clip_bias: Maximum absolute bias to allow (prevents extreme corrections)

    Returns:
        Dictionary mapping station_id to bias correction (in kPa)
    """
    # Convert observations to matric potential (or reuse precomputed)
    df_work = df.copy()
    if observed_matric_col is not None and observed_matric_col in df_work.columns:
        df_work['observed_matric'] = df_work[observed_matric_col]
    else:
        df_work['observed_matric'] = _convert_series_to_matric_potential(
            df_work[observed_col], df_work['station_id'], df_work
        )

    # Compute per-site mean bias: observed_ψ - physics_ψ
    site_biases = {}
    for station_id in df_work['station_id'].unique():
        mask = df_work['station_id'] == station_id
        site_data = df_work[mask]

        obs_psi = site_data['observed_matric'].dropna()
        phys_psi = site_data[physics_col][obs_psi.index].dropna()

        if len(obs_psi) > 0 and len(phys_psi) > 0:
            # Mean bias correction
            bias = obs_psi.mean() - phys_psi.mean()
            # Clip to prevent extreme corrections
            bias = np.clip(bias, -clip_bias, clip_bias)
            site_biases[station_id] = bias
        else:
            site_biases[station_id] = 0.0

    return site_biases


def apply_site_bias_correction(
    df: pd.DataFrame,
    site_biases: Dict[str, float],
    physics_col: str = "physics_prior_surface"
) -> pd.DataFrame:
    """
    Apply site-level bias corrections to physics predictions.

    Args:
        df: DataFrame with physics predictions
        site_biases: Dictionary mapping station_id to bias (in kPa)
        physics_col: Column name for physics prior

    Returns:
        DataFrame with new 'physics_bias_corrected' column
    """
    result = df.copy()
    result['site_bias'] = result['station_id'].map(site_biases).fillna(0.0)
    result['physics_bias_corrected'] = result[physics_col] + result['site_bias']
    return result


def create_matric_residual_targets_debiased(
    df: pd.DataFrame,
    horizons: List[int],
    physics_col: str = "physics_prior_surface",
    observed_col: str = "soil_moisture",
    observed_matric_col: Optional[str] = None,
    clip_residual: float = 1000.0,
    group_cols: Optional[List[str]] = None,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Create debiased residual targets in matric potential space.

    This function:
    1. Computes site-level bias corrections
    2. Applies bias corrections to physics predictions
    3. Creates residual targets from the bias-corrected physics

    The result is residuals that are centered around zero and more learnable
    by the ML model.

    Args:
        df: DataFrame with physics predictions, observations, and soil parameters
        horizons: Forecast horizons in hours
        physics_col: Physics prediction column (matric potential)
        observed_col: Observed soil moisture column (volumetric)
        clip_residual: Maximum absolute residual to allow

    Returns:
        Tuple of (DataFrame with residual targets, site_biases dict)
    """
    logger = logging.getLogger(__name__)

    # Step 1: Compute site-level biases
    site_biases = compute_site_bias_corrections(
        df,
        physics_col=physics_col,
        observed_col=observed_col,
        observed_matric_col=observed_matric_col,
    )
    logger.info(f"Computed bias corrections for {len(site_biases)} sites")
    logger.info(
        f"  Bias range: [{min(site_biases.values()):.1f}, {max(site_biases.values()):.1f}] kPa")
    logger.info(f"  Bias mean: {np.mean(list(site_biases.values())):.1f} kPa")

    # Step 2: Apply bias corrections
    result = apply_site_bias_correction(df, site_biases, physics_col)

    # Step 3: Convert observations to matric potential (or reuse precomputed)
    if observed_matric_col is not None and observed_matric_col in result.columns:
        result['observed_matric'] = result[observed_matric_col]
    else:
        result['observed_matric'] = _convert_series_to_matric_potential(
            result[observed_col], result['station_id'], result)

    if group_cols is None:
        group_cols = ["station_id"]

    if date_col in result.columns:
        result[date_col] = pd.to_datetime(result[date_col])

    # Step 4: Create debiased residual targets
    if date_col in result.columns:
        key_cols = list(group_cols) + [date_col]
        rhs = result[key_cols + ["observed_matric",
                                 "physics_bias_corrected"]].copy()
        rhs = rhs.drop_duplicates(subset=key_cols)

        for horizon in horizons:
            future_date_col = f"{date_col}_plus_{horizon}h"
            target_col = f"target_{horizon}h"
            physics_future_col = f"physics_{horizon}h"
            residual_col = f"residual_target_{horizon}h"

            result[future_date_col] = result[date_col] + \
                pd.Timedelta(hours=int(horizon))
            merge_left = list(group_cols) + [future_date_col]
            rhs_renamed = rhs.rename(
                columns={
                    date_col: future_date_col,
                    "observed_matric": target_col,
                    "physics_bias_corrected": physics_future_col,
                }
            )
            result = result.merge(rhs_renamed, how="left", on=merge_left)

            raw_residual = result[target_col] - result[physics_future_col]
            result[residual_col] = raw_residual.clip(
                -clip_residual, clip_residual)

            valid_residuals = result[residual_col].dropna()
            logger.info(
                f"  {horizon}h residuals: mean={valid_residuals.mean():.1f}, std={valid_residuals.std():.1f}, "
                f"5%={valid_residuals.quantile(0.05):.1f}, 95%={valid_residuals.quantile(0.95):.1f}"
            )

        return result, site_biases

    for horizon in horizons:
        future_obs_matric = result.groupby(
            group_cols)["observed_matric"].shift(-horizon)
        future_physics_debiased = result.groupby(
            group_cols)["physics_bias_corrected"].shift(-horizon)
        result[f"target_{horizon}h"] = future_obs_matric
        result[f"physics_{horizon}h"] = future_physics_debiased

        raw_residual = future_obs_matric - future_physics_debiased
        result[f"residual_target_{horizon}h"] = raw_residual.clip(
            -clip_residual, clip_residual)

        valid_residuals = result[f"residual_target_{horizon}h"].dropna()
        logger.info(
            f"  {horizon}h residuals: mean={valid_residuals.mean():.1f}, std={valid_residuals.std():.1f}, "
            f"5%={valid_residuals.quantile(0.05):.1f}, 95%={valid_residuals.quantile(0.95):.1f}"
        )

    return result, site_biases


def _convert_series_to_matric_potential(
    theta_series: pd.Series,
    station_ids: pd.Series,
    df: pd.DataFrame
) -> pd.Series:
    """
    Convert a series of volumetric water contents to matric potential.

    Uses deterministic Van Genuchten parameters for each station.
    """
    result = pd.Series(index=theta_series.index, dtype=float)

    # Group by station for parameter estimation
    for station_id in station_ids.unique():
        mask = station_ids == station_id
        station_theta = theta_series[mask]

        # Get soil parameters for this station
        station_data = df[df['station_id'] == station_id]
        if len(station_data) > 0:
            sand_pct = station_data['sand_pct'].iloc[0] if 'sand_pct' in station_data.columns else 50.0
            clay_pct = station_data['clay_pct'].iloc[0] if 'clay_pct' in station_data.columns else 20.0
        else:
            sand_pct, clay_pct = 50.0, 20.0

        if pd.isna(sand_pct) or pd.isna(clay_pct):
            sand_pct, clay_pct = 50.0, 20.0

        # Get single deterministic parameter set
        params = tropical_ptf_van_genuchten(sand_pct, clay_pct, n_sets=1)[0]

        # Convert each value deterministically
        station_psi = []
        for theta in station_theta:
            if pd.isna(theta):
                psi = np.nan
            else:
                try:
                    psi = potential_from_water_content(theta, params)
                    # Clip extreme values
                    if psi < -10000:
                        psi = -10000
                    elif psi > 0:
                        psi = -0.1
                except:
                    psi = np.nan

            station_psi.append(psi)

        result.loc[mask] = station_psi

    return result


def create_prediction_features(
    df: pd.DataFrame,
    horizon: int,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Create features for prediction at a specific horizon.

    Shifts features backward by horizon to predict future values.

    Args:
        df: DataFrame with features
        horizon: Forecast horizon in hours
        feature_cols: Feature column names

    Returns:
        DataFrame with shifted features for prediction
    """
    # For prediction, we use current features to predict future residuals
    # So we don't shift the features, just select them
    return df[feature_cols].copy()


# =============================================================================
# SEQUENTIAL FEATURE ENGINEERING
# =============================================================================

def add_sequential_features(
    df: pd.DataFrame,
    group_col: str = "station_id",
    date_col: str = "date",
    target_cols: Optional[List[str]] = None,
    weather_cols: Optional[List[str]] = None,
    lag_days: List[int] = [1, 2, 3, 7, 14],
    rolling_windows: List[int] = [3, 7, 14, 30],
) -> pd.DataFrame:
    """
    Add sequential features to capture temporal dependencies in soil moisture.

    This is critical for soil moisture modeling because:
    1. Soil moisture has strong autocorrelation (yesterday's value predicts today)
    2. Antecedent moisture conditions affect infiltration and drainage
    3. Weather history affects current soil water balance

    Args:
        df: DataFrame with time series data
        group_col: Column to group by (station_id)
        date_col: Date column name
        target_cols: Columns to create lag features for (soil moisture, physics priors)
        weather_cols: Weather columns for cumulative/rolling features
        lag_days: List of lag days for creating lag features
        rolling_windows: List of window sizes for rolling statistics

    Returns:
        DataFrame with sequential features added
    """
    logger.info("Adding sequential features for temporal dependency modeling...")

    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.sort_values([group_col, date_col])

    # Default columns if not specified
    if target_cols is None:
        target_cols = ['soil_moisture', 'physics_prior_surface']
        target_cols = [c for c in target_cols if c in result.columns]

    if weather_cols is None:
        weather_cols = ['precipitation_mm', 'et0_mm', 'temperature_2m']
        weather_cols = [c for c in weather_cols if c in result.columns]

    new_features = []

    # 1. LAG FEATURES - capture autocorrelation
    logger.info(f"  Creating lag features for {target_cols}")
    for col in target_cols:
        for lag in lag_days:
            lag_col = f"{col}_lag{lag}d"
            result[lag_col] = result.groupby(group_col)[col].shift(lag)
            new_features.append(lag_col)

    # 2. ROLLING STATISTICS - capture recent trends
    logger.info(f"  Creating rolling statistics for windows {rolling_windows}")
    for col in target_cols + weather_cols:
        for window in rolling_windows:
            # Rolling mean
            mean_col = f"{col}_roll{window}d_mean"
            result[mean_col] = result.groupby(group_col)[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            new_features.append(mean_col)

            # Rolling std (variability indicator)
            if window >= 3:
                std_col = f"{col}_roll{window}d_std"
                result[std_col] = result.groupby(group_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=2).std()
                )
                new_features.append(std_col)

    # 3. CUMULATIVE WEATHER - antecedent conditions
    logger.info("  Creating cumulative weather features")
    if 'precipitation_mm' in weather_cols:
        for window in [3, 7, 14, 30]:
            cum_col = f"precip_cum{window}d"
            result[cum_col] = result.groupby(group_col)['precipitation_mm'].transform(
                lambda x: x.rolling(window=window, min_periods=1).sum()
            )
            new_features.append(cum_col)

    if 'et0_mm' in weather_cols:
        for window in [3, 7, 14]:
            cum_col = f"et0_cum{window}d"
            result[cum_col] = result.groupby(group_col)['et0_mm'].transform(
                lambda x: x.rolling(window=window, min_periods=1).sum()
            )
            new_features.append(cum_col)

    # 4. WATER BALANCE INDICATORS
    logger.info("  Creating water balance indicators")
    if 'precipitation_mm' in result.columns and 'et0_mm' in result.columns:
        # P-ET balance over different periods
        result['p_minus_et_today'] = result['precipitation_mm'] - result['et0_mm']

        for window in [7, 14, 30]:
            balance_col = f"p_minus_et_cum{window}d"
            result[balance_col] = result.groupby(group_col)['p_minus_et_today'].transform(
                lambda x: x.rolling(window=window, min_periods=1).sum()
            )
            new_features.append(balance_col)

        # Days since significant rain
        result['days_since_rain_5mm'] = result.groupby(group_col).apply(
            lambda g: _days_since_event(g['precipitation_mm'], threshold=5.0)
        ).reset_index(level=0, drop=True)
        new_features.append('days_since_rain_5mm')

    # 5. TREND FEATURES - direction of change
    logger.info("  Creating trend features")
    for col in target_cols:
        # Difference from previous day (rate of change)
        diff_col = f"{col}_diff1d"
        result[diff_col] = result.groupby(group_col)[col].diff(1)
        new_features.append(diff_col)

        # Trend over last week (slope approximation)
        if f"{col}_lag7d" in result.columns:
            trend_col = f"{col}_trend7d"
            result[trend_col] = result[col] - result[f"{col}_lag7d"]
            new_features.append(trend_col)

    # 6. SEASONAL FEATURES
    logger.info("  Creating seasonal features")
    result['day_of_year'] = result[date_col].dt.dayofyear
    result['day_of_year_sin'] = np.sin(2 * np.pi * result['day_of_year'] / 365)
    result['day_of_year_cos'] = np.cos(2 * np.pi * result['day_of_year'] / 365)
    new_features.extend(['day_of_year_sin', 'day_of_year_cos'])

    # 7. PHYSICS BIAS FEATURES - track residual patterns
    logger.info("  Creating physics bias features")
    if 'physics_prior_surface' in result.columns and 'soil_moisture' in result.columns:
        # Current residual (obs - physics) as a feature for learning systematic biases
        # Note: We need to convert to same units for meaningful residual
        # Physics is in psi (kPa), soil_moisture is in theta (m3/m3)
        # Create a rolling estimate of recent bias
        for window in [7, 14, 30]:
            bias_col = f"physics_residual_roll{window}d"
            # This is a rough proxy - the ML will learn the true relationship
            # Using physics prior directly since it's what we want to correct
            result[bias_col] = result.groupby(group_col)['physics_prior_surface'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            ) - result.groupby(group_col)['physics_prior_surface'].transform(
                lambda x: x.shift(1).rolling(
                    window=window, min_periods=1).mean()
            )
            new_features.append(bias_col)

    logger.info(f"  Added {len(new_features)} sequential features")

    return result, new_features


def _days_since_event(series: pd.Series, threshold: float) -> pd.Series:
    """Calculate days since last event exceeding threshold."""
    result = pd.Series(index=series.index, dtype=float)
    days_count = 0

    for idx, val in series.items():
        if val >= threshold:
            days_count = 0
        else:
            days_count += 1
        result[idx] = days_count

    return result


def prepare_features_with_sequences(
    df: pd.DataFrame,
    static_features: List[str],
    dynamic_features: List[str],
    group_col: str = "station_id",
    date_col: str = "date",
    lag_days: List[int] = [1, 2, 3, 7, 14],
    rolling_windows: List[int] = [3, 7, 14],
    sequential_target_cols: Optional[List[str]] = None,
    sequential_weather_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features combining static site characteristics with dynamic sequential features.

    This addresses the need to "learn interdependency between input and output data
    when the sequence span gets longer" by creating features that capture:
    - Short-term dynamics (1-3 day lags)
    - Medium-term patterns (7-14 day rolling stats)
    - Long-term antecedent conditions (cumulative weather)

    Args:
        df: Raw DataFrame with observations
        static_features: Site-level features (clay, sand, depth, etc.)
        dynamic_features: Time-varying features (weather, physics priors)
        group_col: Station grouping column
        date_col: Date column
        lag_days: Days for lag features
        rolling_windows: Windows for rolling statistics

    Returns:
        Enriched DataFrame and list of all feature column names
    """
    logger.info("Preparing features with sequential dependencies...")

    # Determine which columns to create sequences for
    if sequential_target_cols is None:
        target_cols = [c for c in ['soil_moisture', 'physics_prior_surface']
                       if c in df.columns]
    else:
        target_cols = [c for c in sequential_target_cols if c in df.columns]

    if sequential_weather_cols is None:
        weather_cols = [c for c in ['precipitation_mm', 'et0_mm', 'temperature_2m', 'relative_humidity_2m']
                        if c in df.columns]
    else:
        weather_cols = [c for c in sequential_weather_cols if c in df.columns]

    # Add sequential features
    result, seq_features = add_sequential_features(
        df,
        group_col=group_col,
        date_col=date_col,
        target_cols=target_cols,
        weather_cols=weather_cols,
        lag_days=lag_days,
        rolling_windows=rolling_windows,
    )

    # Combine all features
    all_features = list(static_features) + \
        list(dynamic_features) + seq_features

    # Remove duplicates while preserving order
    seen = set()
    unique_features = []
    for f in all_features:
        if f in result.columns and f not in seen:
            seen.add(f)
            unique_features.append(f)

    logger.info(f"Total features: {len(unique_features)} ({len(static_features)} static, "
                f"{len(dynamic_features)} dynamic, {len(seq_features)} sequential)")

    return result, unique_features
