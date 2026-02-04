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

from swpps.validation.evaluation import ModelEvaluator, HorizonEvaluationResult
from swpps.physics.van_genuchten import potential_from_water_content, tropical_ptf_van_genuchten

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

    # Random state
    random_state: int = 42


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

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

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

    def train_with_site_cv(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
        feature_cols: List[str],
        n_folds: int = 5,
    ) -> Tuple[Any, List[CVFoldResult]]:
        """
        Train with site-blocked cross-validation.

        Args:
            X: Feature matrix
            y: Target values
            groups: Group labels for blocking
            feature_cols: Feature column names
            n_folds: Number of CV folds

        Returns:
            Best model and fold results
        """
        logger.info(f"Training with {n_folds}-fold site-blocked CV")

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
        logger.info(
            f"CV complete: mean RMSE = {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")

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


class ModelEvaluatorExtended(ModelEvaluator):
    """
    Extended model evaluator with training utilities.
    """

    def evaluate_trained_models(
        self,
        models: Dict[int, Any],
        test_df: pd.DataFrame,
        horizons: List[int],
        feature_cols: List[str],
        target_prefix: str = "residual_target",
        physics_col: str = "physics_prior",
    ) -> Dict[int, HorizonEvaluationResult]:
        """
        Evaluate trained models on test data.

        Args:
            models: Dict mapping horizon to trained model
            test_df: Test dataframe
            horizons: Forecast horizons
            feature_cols: Feature columns
            target_prefix: Target column prefix
            physics_col: Physics prediction column

        Returns:
            Horizon evaluation results
        """
        logger.info("Evaluating trained models on test data")

        predictions_by_horizon = {}

        for horizon in horizons:
            if horizon not in models:
                logger.warning(f"No model for horizon {horizon}h")
                continue

            model = models[horizon]
            target_col = f"{target_prefix}_{horizon}h"

            # Filter valid test data
            valid_mask = test_df[target_col].notna()
            test_valid = test_df[valid_mask].copy()

            if len(test_valid) == 0:
                logger.warning(f"No valid test data for horizon {horizon}h")
                continue

            # Prepare features
            X_test = test_valid[feature_cols].values

            # Predict residual
            residual_pred = model.predict(X_test)

            # Combine with physics
            test_valid = test_valid.copy()
            test_valid['residual_pred'] = residual_pred
            test_valid['predicted'] = test_valid[physics_col] + residual_pred

            # Clip to reasonable range
            test_valid['predicted'] = test_valid['predicted'].clip(0, 0.6)

            predictions_by_horizon[horizon] = test_valid

        # Evaluate
        return self.evaluate_multi_horizon(
            predictions_by_horizon,
            observed_col="target_0h",  # Use 0h target as observed
            predicted_col="predicted",
        )


def create_residual_targets(
    df: pd.DataFrame,
    horizons: List[int],
    physics_col: str = "physics_prior",
    observed_col: str = "soil_moisture",
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

    for horizon in horizons:
        # Future observed value
        future_obs = result.groupby('station_id')[observed_col].shift(-horizon)

        # Future physics prediction (assume physics is for same horizon)
        future_physics = result.groupby(
            'station_id')[physics_col].shift(-horizon)

        # Residual target: future_obs - future_physics
        result[f'residual_target_{horizon}h'] = future_obs - future_physics

    return result


def create_matric_residual_targets(
    df: pd.DataFrame,
    horizons: List[int],
    physics_col: str = "physics_prior",
    observed_col: str = "soil_moisture",
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

    # Convert observed volumetric to matric potential
    result['observed_matric'] = _convert_series_to_matric_potential(
        result[observed_col], result['station_id'], result)

    # Physics predictions are already in matric potential space
    result['physics_matric'] = result[physics_col]

    for horizon in horizons:
        # Future observed matric potential
        future_obs_matric = result.groupby(
            'station_id')['observed_matric'].shift(-horizon)

        # Future physics matric potential
        future_physics_matric = result.groupby(
            'station_id')['physics_matric'].shift(-horizon)

        # Residual target in matric potential space: future_obs_ψ - future_physics_ψ
        result[f'residual_target_{horizon}h'] = future_obs_matric - \
            future_physics_matric

    return result


def compute_site_bias_corrections(
    df: pd.DataFrame,
    physics_col: str = "physics_prior_surface",
    observed_col: str = "soil_moisture",
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
    # Convert observations to matric potential
    df_work = df.copy()
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
    clip_residual: float = 1000.0
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
    site_biases = compute_site_bias_corrections(df, physics_col, observed_col)
    logger.info(f"Computed bias corrections for {len(site_biases)} sites")
    logger.info(
        f"  Bias range: [{min(site_biases.values()):.1f}, {max(site_biases.values()):.1f}] kPa")
    logger.info(f"  Bias mean: {np.mean(list(site_biases.values())):.1f} kPa")

    # Step 2: Apply bias corrections
    result = apply_site_bias_correction(df, site_biases, physics_col)

    # Step 3: Convert observations to matric potential
    result['observed_matric'] = _convert_series_to_matric_potential(
        result[observed_col], result['station_id'], result)

    # Step 4: Create debiased residual targets
    for horizon in horizons:
        # Future observed matric potential
        future_obs_matric = result.groupby(
            'station_id')['observed_matric'].shift(-horizon)

        # Future BIAS-CORRECTED physics matric potential
        future_physics_debiased = result.groupby(
            'station_id')['physics_bias_corrected'].shift(-horizon)

        # Debiased residual target
        raw_residual = future_obs_matric - future_physics_debiased

        # Clip extreme residuals to improve ML stability
        result[f'residual_target_{horizon}h'] = raw_residual.clip(
            -clip_residual, clip_residual)

        # Log residual stats
        valid_residuals = result[f'residual_target_{horizon}h'].dropna()
        logger.info(f"  {horizon}h residuals: mean={valid_residuals.mean():.1f}, std={valid_residuals.std():.1f}, "
                    f"5%={valid_residuals.quantile(0.05):.1f}, 95%={valid_residuals.quantile(0.95):.1f}")

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
    target_cols = [c for c in ['soil_moisture', 'physics_prior_surface']
                   if c in df.columns]
    weather_cols = [c for c in ['precipitation_mm', 'et0_mm', 'temperature_2m', 'relative_humidity_2m']
                    if c in df.columns]

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
