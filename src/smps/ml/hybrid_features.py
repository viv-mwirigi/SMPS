"""
Hybrid model feature enhancement utilities.

This module provides feature enhancement specifically for hybrid
physics+ML models, including:
- Physics state variables integration
- Physics-observation bias features
- Confidence indicators
- Residual smoothing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger("smps.ml.hybrid_features")


class HybridFeatureEnhancer:
    """
    Enhance features for hybrid physics+ML models.

    Adds physics-informed features that help the ML model
    learn effective residual corrections.
    """

    def __init__(
        self,
        physics_col_prefix: str = 'physics_sm_',
        obs_col_prefix: str = 'obs_sm_',
        rolling_bias_window: int = 7,
    ):
        """
        Initialize hybrid feature enhancer.

        Args:
            physics_col_prefix: Prefix for physics prediction columns
            obs_col_prefix: Prefix for observation columns
            rolling_bias_window: Window for rolling bias calculation
        """
        self.physics_col_prefix = physics_col_prefix
        self.obs_col_prefix = obs_col_prefix
        self.rolling_bias_window = rolling_bias_window

        # Mapping from observation depth to physics layer
        self.obs_to_physics_mapping = {
            '5cm': 'surface',
            '10cm': 'surface',
            '15cm': 'surface',
            '20cm': 'root',
            '30cm': 'root',
            '50cm': 'deep',
            '100cm': 'deep',
            '200cm': 'deep',
        }

    def enhance_features(
        self,
        X: pd.DataFrame,
        df_orig: pd.DataFrame,
        physics_vals: np.ndarray,
        obs_vals: np.ndarray,
        physics_weight: float,
        horizon_days: int,
    ) -> pd.DataFrame:
        """
        Enhance features with physics-informed variables.

        Args:
            X: Feature DataFrame
            df_orig: Original DataFrame with all columns
            physics_vals: Physics predictions (aligned with X)
            obs_vals: Observations (aligned with X)
            physics_weight: Adaptive physics weight (0-1)
            horizon_days: Forecast horizon in days

        Returns:
            Enhanced feature DataFrame
        """
        X_enhanced = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # 1. Add physics state variables
        X_enhanced = self._add_physics_state_variables(X_enhanced, df_orig)

        # 2. Add physics-observation bias features
        X_enhanced = self._add_bias_features(X_enhanced, df_orig)

        # 3. Add confidence indicators
        X_enhanced['physics_confidence'] = physics_weight
        X_enhanced['horizon_days'] = horizon_days

        # 4. Add physics variance (uncertainty proxy)
        X_enhanced = self._add_physics_variance(X_enhanced, df_orig)

        return X_enhanced

    def enhance_train_val_test(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        physics_train: np.ndarray,
        physics_val: np.ndarray,
        physics_test: np.ndarray,
        obs_train: np.ndarray,
        obs_val: np.ndarray,
        obs_test: np.ndarray,
        physics_weight: float,
        horizon_days: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Enhance features for train/val/test splits.

        Args:
            X_train, X_val, X_test: Feature arrays/DataFrames
            df_train, df_val, df_test: Original DataFrames
            physics_train, physics_val, physics_test: Physics predictions
            obs_train, obs_val, obs_test: Observations
            physics_weight: Adaptive physics weight
            horizon_days: Forecast horizon

        Returns:
            Tuple of enhanced feature arrays (train, val, test)
        """
        # Convert to DataFrames if needed
        columns = X_train.columns if hasattr(X_train, 'columns') else None

        X_train_df = pd.DataFrame(X_train, columns=columns)
        X_val_df = pd.DataFrame(X_val, columns=columns)
        X_test_df = pd.DataFrame(X_test, columns=columns)

        # Enhance each split
        X_train_enhanced = self.enhance_features(
            X_train_df, df_train.iloc[:len(X_train_df)],
            physics_train[:len(X_train_df)], obs_train,
            physics_weight, horizon_days
        )

        X_val_enhanced = self.enhance_features(
            X_val_df, df_val.iloc[:len(X_val_df)],
            physics_val[:len(X_val_df)], obs_val,
            physics_weight, horizon_days
        )

        X_test_enhanced = self.enhance_features(
            X_test_df, df_test.iloc[:len(X_test_df)],
            physics_test[:len(X_test_df)], obs_test,
            physics_weight, horizon_days
        )

        return X_train_enhanced.values, X_val_enhanced.values, X_test_enhanced.values

    def _add_physics_state_variables(
        self,
        X: pd.DataFrame,
        df_orig: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add physics model state variables as features."""
        result = X.copy()

        physics_cols = [
            c for c in df_orig.columns
            if c.startswith('physics_')
        ]

        for col in physics_cols:
            if col in df_orig.columns and len(df_orig) >= len(result):
                result[f'physics_state_{col}'] = df_orig[col].iloc[:len(
                    result)].values

        return result

    def _add_bias_features(
        self,
        X: pd.DataFrame,
        df_orig: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add physics-observation bias features."""
        result = X.copy()

        # Find observation columns
        obs_cols = [
            c for c in df_orig.columns
            if c.startswith(self.obs_col_prefix)
            and not c.endswith('_lag1')
        ]

        for obs_col in obs_cols:
            # Get matching physics column
            depth_suffix = obs_col.replace(self.obs_col_prefix, '')
            phys_layer = self.obs_to_physics_mapping.get(
                depth_suffix, 'surface')
            phys_col = f'{self.physics_col_prefix}{phys_layer}'

            if phys_col not in df_orig.columns or obs_col not in df_orig.columns:
                continue

            if len(df_orig) < len(result):
                continue

            # Current bias
            bias = df_orig[phys_col].iloc[:len(
                result)] - df_orig[obs_col].iloc[:len(result)]
            result[f'physics_bias_{depth_suffix}'] = bias.fillna(0).values

            # Rolling bias
            rolling_bias = bias.rolling(
                self.rolling_bias_window, min_periods=1).mean()
            result[f'physics_bias_{self.rolling_bias_window}d_{depth_suffix}'] = rolling_bias.fillna(
                0).values

        return result

    def _add_physics_variance(
        self,
        X: pd.DataFrame,
        df_orig: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add physics prediction variance (uncertainty proxy)."""
        result = X.copy()

        physics_cols = [
            c for c in df_orig.columns
            if c.startswith('physics_') and 'state' not in c
        ]

        if len(physics_cols) > 1 and len(df_orig) >= len(result):
            physics_variance = df_orig[physics_cols].iloc[:len(
                result)].var(axis=1)
            result['physics_variance'] = physics_variance.fillna(0).values

        return result


def smooth_residuals(
    residuals: np.ndarray,
    method: str = 'exponential',
    alpha: float = 0.3,
    window: int = 7,
) -> np.ndarray:
    """
    Apply temporal smoothing to residuals to reduce noise.

    Args:
        residuals: Array of residual values
        method: Smoothing method ('exponential' or 'rolling')
        alpha: Exponential smoothing factor (0 < alpha < 1)
        window: Rolling window size (for rolling method)

    Returns:
        Smoothed residuals
    """
    if len(residuals) < 2:
        return residuals

    if method == 'exponential':
        smoothed = np.zeros_like(residuals)
        smoothed[0] = residuals[0]

        for i in range(1, len(residuals)):
            if not np.isnan(residuals[i]):
                smoothed[i] = alpha * residuals[i] + \
                    (1 - alpha) * smoothed[i-1]
            else:
                smoothed[i] = smoothed[i-1]

        return smoothed

    elif method == 'rolling':
        series = pd.Series(residuals)
        smoothed = series.rolling(window, min_periods=1, center=True).mean()
        return smoothed.fillna(method='ffill').fillna(method='bfill').values

    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def compute_residual_target(
    observations: np.ndarray,
    physics_predictions: np.ndarray,
    smooth: bool = False,
    smooth_window: int = 7,
) -> np.ndarray:
    """
    Compute residual target for ML model.

    residual = observation - physics_prediction

    Args:
        observations: Ground truth observations
        physics_predictions: Physics model predictions
        smooth: Whether to smooth residuals
        smooth_window: Smoothing window size

    Returns:
        Residual array
    """
    residuals = observations - physics_predictions

    if smooth:
        residuals = smooth_residuals(
            residuals, method='rolling', window=smooth_window)

    return residuals


def combine_physics_ml_predictions(
    physics_predictions: np.ndarray,
    ml_residual_predictions: np.ndarray,
    physics_weight: float = 0.5,
    use_additive: bool = True,
    clip_range: Tuple[float, float] = (0.0, 0.6),
) -> np.ndarray:
    """
    Combine physics model and ML residual predictions.

    Two combination strategies:
    1. Additive: final = physics + ml_residual
    2. Weighted: final = weight * physics + (1 - weight) * (physics + ml_residual)

    Args:
        physics_predictions: Physics model predictions
        ml_residual_predictions: ML-predicted residuals
        physics_weight: Weight for physics in weighted combination
        use_additive: If True, use simple additive; if False, use weighted
        clip_range: Range to clip final predictions

    Returns:
        Combined predictions
    """
    if use_additive or physics_weight > 0.4:
        # Simple additive: physics + ML residual
        hybrid = physics_predictions + ml_residual_predictions
    else:
        # Weighted combination
        hybrid = (
            physics_weight * physics_predictions +
            (1 - physics_weight) * (physics_predictions + ml_residual_predictions)
        )

    # Clip to valid soil moisture range
    hybrid = np.clip(hybrid, clip_range[0], clip_range[1])

    return hybrid


def prepare_hybrid_training_data(
    X: pd.DataFrame,
    observations: np.ndarray,
    physics_predictions: np.ndarray,
    feature_cols: List[str],
    smooth_residuals_flag: bool = False,
    smooth_window: int = 7,
) -> pd.DataFrame:
    """
    Prepare training DataFrame for hybrid residual learner.

    Args:
        X: Feature DataFrame
        observations: Ground truth observations
        physics_predictions: Physics model predictions
        feature_cols: List of feature column names
        smooth_residuals_flag: Whether to smooth residuals
        smooth_window: Smoothing window size

    Returns:
        Training DataFrame with obs, physics, residual, and features
    """
    residuals = compute_residual_target(
        observations, physics_predictions,
        smooth=smooth_residuals_flag,
        smooth_window=smooth_window
    )

    train_df = pd.DataFrame({
        'obs': observations,
        'physics': physics_predictions,
        'residual': residuals,
        'obs_lag1': np.roll(observations, 1),
        'physics_lag1': np.roll(physics_predictions, 1),
    })

    # Set first values to NaN (no valid lag)
    train_df.loc[0, ['obs_lag1', 'physics_lag1']] = np.nan

    # Add feature columns
    X_values = X.values if hasattr(X, 'values') else X
    for i, col in enumerate(feature_cols):
        if i < X_values.shape[1]:
            train_df[col] = X_values[:, i]

    return train_df
