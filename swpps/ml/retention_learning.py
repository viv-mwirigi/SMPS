"""
Learnable Retention Curve Corrections for SWPPS.

The Problem:
- Standard PTFs (Saxton-Rawls, Hodnett-Tomasella) are calibrated on global datasets
- For specific sites, the θ-ψ relationship can be very wrong
- Even perfect ψ predictions become garbage when converted to θ with wrong PTF

Solution:
- Learn site-specific corrections to the retention curve
- Train ML to predict adjustment factors for α, n, θs, θr
- Or directly learn the θ-ψ relationship residual

This module provides:
1. Site-specific retention curve calibration from observation pairs
2. ML-based retention curve correction models
3. Hybrid ψ→θ conversion with learned corrections
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

from swpps.core.types import VanGenuchtenParams
from swpps.physics.van_genuchten import (
    water_content_from_potential,
    potential_from_water_content,
    tropical_ptf_van_genuchten,
)

logger = logging.getLogger("swpps.ml.retention_learning")


@dataclass
class RetentionCorrectionConfig:
    """Configuration for retention curve learning."""
    # Correction mode: 'scaling', 'direct', or 'hybrid'
    correction_mode: str = "scaling"

    # Scaling bounds for parameters
    alpha_scale_bounds: Tuple[float, float] = (0.1, 10.0)
    n_scale_bounds: Tuple[float, float] = (0.5, 2.0)
    theta_s_scale_bounds: Tuple[float, float] = (0.7, 1.3)
    theta_r_scale_bounds: Tuple[float, float] = (0.5, 2.0)

    # ML model params for learning corrections
    n_estimators: int = 500
    learning_rate: float = 0.03
    max_depth: int = 4
    num_leaves: int = 15
    min_child_samples: int = 30


def calibrate_retention_curve_per_site(
    df: pd.DataFrame,
    observed_theta_col: str = "soil_moisture",
    physics_psi_col: str = "physics_prior_surface",
    station_col: str = "station_id",
    sand_col: str = "sand_pct",
    clay_col: str = "clay_pct",
) -> Dict[str, Dict[str, float]]:
    """
    Calibrate retention curve parameters for each site using observed θ-ψ pairs.

    For each site, we:
    1. Start with PTF-estimated VG parameters
    2. Optimize scaling factors to minimize θ prediction error

    Args:
        df: DataFrame with observations and physics predictions
        observed_theta_col: Column with observed volumetric water content
        physics_psi_col: Column with physics-predicted matric potential
        station_col: Station identifier column
        sand_col: Sand percentage column
        clay_col: Clay percentage column

    Returns:
        Dict mapping station_id to calibrated VG parameter scaling factors
    """
    logger.info("Calibrating retention curves per site...")

    calibrated_params = {}

    for station_id, station_df in df.groupby(station_col):
        # Get base PTF parameters
        sand_pct = station_df[sand_col].iloc[0] if sand_col in station_df.columns else 50.0
        clay_pct = station_df[clay_col].iloc[0] if clay_col in station_df.columns else 20.0

        if pd.isna(sand_pct) or pd.isna(clay_pct):
            sand_pct, clay_pct = 50.0, 20.0

        base_params = tropical_ptf_van_genuchten(
            sand_pct, clay_pct, n_sets=1)[0]

        # Get valid observation pairs (θ_obs, ψ_physics)
        valid_mask = (
            station_df[observed_theta_col].notna() &
            station_df[physics_psi_col].notna()
        )
        valid_df = station_df[valid_mask]

        if len(valid_df) < 30:
            # Not enough data to calibrate - use default scaling
            calibrated_params[station_id] = {
                'alpha_scale': 1.0,
                'n_scale': 1.0,
                'theta_s_scale': 1.0,
                'theta_r_scale': 1.0,
                'base_params': base_params,
                'n_samples': len(valid_df),
                'calibrated': False,
            }
            continue

        theta_obs = valid_df[observed_theta_col].values
        psi_physics = valid_df[physics_psi_col].values

        # Objective: minimize RMSE when converting ψ → θ
        def objective(scales):
            alpha_s, n_s, theta_s_s, theta_r_s = scales

            # Apply scaling
            params = VanGenuchtenParams(
                alpha=base_params.alpha * alpha_s,
                n=max(1.05, base_params.n * n_s),  # n must be > 1
                theta_s=min(0.65, base_params.theta_s * theta_s_s),
                theta_r=max(0.001, base_params.theta_r * theta_r_s),
                K_sat=base_params.K_sat,
            )

            # Convert ψ → θ with scaled params
            theta_pred = np.array([
                water_content_from_potential(psi, params)
                for psi in psi_physics
            ])

            # RMSE
            rmse = np.sqrt(np.mean((theta_pred - theta_obs) ** 2))

            return rmse

        # Optimize scaling factors
        initial = [1.0, 1.0, 1.0, 1.0]
        bounds = [
            (0.1, 10.0),   # alpha scale
            (0.5, 2.0),    # n scale
            (0.7, 1.3),    # theta_s scale
            (0.5, 2.0),    # theta_r scale
        ]

        try:
            result = minimize(
                objective,
                initial,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )

            alpha_s, n_s, theta_s_s, theta_r_s = result.x

            # Check improvement
            baseline_rmse = objective([1.0, 1.0, 1.0, 1.0])
            calibrated_rmse = result.fun

            if calibrated_rmse < baseline_rmse * 0.95:  # At least 5% improvement
                calibrated_params[station_id] = {
                    'alpha_scale': float(alpha_s),
                    'n_scale': float(n_s),
                    'theta_s_scale': float(theta_s_s),
                    'theta_r_scale': float(theta_r_s),
                    'base_params': base_params,
                    'baseline_rmse': float(baseline_rmse),
                    'calibrated_rmse': float(calibrated_rmse),
                    'n_samples': len(valid_df),
                    'calibrated': True,
                }
                logger.debug(
                    f"  {station_id}: RMSE {baseline_rmse:.4f} → {calibrated_rmse:.4f} "
                    f"(α×{alpha_s:.2f}, n×{n_s:.2f})"
                )
            else:
                calibrated_params[station_id] = {
                    'alpha_scale': 1.0,
                    'n_scale': 1.0,
                    'theta_s_scale': 1.0,
                    'theta_r_scale': 1.0,
                    'base_params': base_params,
                    'baseline_rmse': float(baseline_rmse),
                    'n_samples': len(valid_df),
                    'calibrated': False,
                }

        except Exception as e:
            logger.warning(f"  Calibration failed for {station_id}: {e}")
            calibrated_params[station_id] = {
                'alpha_scale': 1.0,
                'n_scale': 1.0,
                'theta_s_scale': 1.0,
                'theta_r_scale': 1.0,
                'base_params': base_params,
                'n_samples': len(valid_df),
                'calibrated': False,
            }

    # Summary
    n_calibrated = sum(1 for p in calibrated_params.values()
                       if p.get('calibrated', False))
    logger.info(
        f"Calibrated {n_calibrated}/{len(calibrated_params)} sites successfully")

    return calibrated_params


def get_calibrated_vg_params(
    station_id: str,
    calibrated_params: Dict[str, Dict[str, float]],
    sand_pct: float = 50.0,
    clay_pct: float = 20.0,
) -> VanGenuchtenParams:
    """
    Get calibrated Van Genuchten parameters for a station.

    Args:
        station_id: Station identifier
        calibrated_params: Dict of calibrated parameters per station
        sand_pct: Sand percentage (for uncalibrated stations)
        clay_pct: Clay percentage (for uncalibrated stations)

    Returns:
        VanGenuchtenParams with calibrated scaling applied
    """
    if station_id in calibrated_params:
        site_params = calibrated_params[station_id]
        base = site_params['base_params']

        return VanGenuchtenParams(
            alpha=base.alpha * site_params['alpha_scale'],
            n=max(1.05, base.n * site_params['n_scale']),
            theta_s=min(0.65, base.theta_s * site_params['theta_s_scale']),
            theta_r=max(0.001, base.theta_r * site_params['theta_r_scale']),
            K_sat=base.K_sat,
        )
    else:
        # Use standard PTF
        return tropical_ptf_van_genuchten(sand_pct, clay_pct, n_sets=1)[0]


def convert_psi_to_theta_calibrated(
    psi_series: pd.Series,
    station_ids: pd.Series,
    calibrated_params: Dict[str, Dict[str, float]],
    df: pd.DataFrame,
    sand_col: str = "sand_pct",
    clay_col: str = "clay_pct",
) -> pd.Series:
    """
    Convert matric potential to volumetric water content using calibrated params.

    Args:
        psi_series: Series of matric potentials (kPa)
        station_ids: Series of station identifiers
        calibrated_params: Dict of calibrated parameters per station
        df: DataFrame with soil texture info
        sand_col: Sand percentage column
        clay_col: Clay percentage column

    Returns:
        Series of volumetric water contents (m³/m³)
    """
    result = pd.Series(index=psi_series.index, dtype=float)

    for station_id in station_ids.unique():
        mask = station_ids == station_id
        station_psi = psi_series[mask]

        # Get soil parameters
        station_data = df[df['station_id'] == station_id]
        if len(station_data) > 0:
            sand_pct = station_data[sand_col].iloc[0] if sand_col in station_data.columns else 50.0
            clay_pct = station_data[clay_col].iloc[0] if clay_col in station_data.columns else 20.0
        else:
            sand_pct, clay_pct = 50.0, 20.0

        if pd.isna(sand_pct) or pd.isna(clay_pct):
            sand_pct, clay_pct = 50.0, 20.0

        # Get calibrated VG params
        params = get_calibrated_vg_params(
            station_id, calibrated_params, sand_pct, clay_pct)

        # Convert
        station_theta = []
        for psi in station_psi:
            if pd.isna(psi):
                theta = np.nan
            else:
                try:
                    theta = water_content_from_potential(psi, params)
                    theta = np.clip(theta, 0.0, params.theta_s)
                except:
                    theta = np.nan
            station_theta.append(theta)

        result.loc[mask] = station_theta

    return result


def convert_theta_to_psi_calibrated(
    theta_series: pd.Series,
    station_ids: pd.Series,
    calibrated_params: Dict[str, Dict[str, float]],
    df: pd.DataFrame,
    sand_col: str = "sand_pct",
    clay_col: str = "clay_pct",
) -> pd.Series:
    """
    Convert volumetric water content to matric potential using calibrated params.

    Args:
        theta_series: Series of volumetric water contents (m³/m³)
        station_ids: Series of station identifiers
        calibrated_params: Dict of calibrated parameters per station
        df: DataFrame with soil texture info
        sand_col: Sand percentage column
        clay_col: Clay percentage column

    Returns:
        Series of matric potentials (kPa)
    """
    result = pd.Series(index=theta_series.index, dtype=float)

    for station_id in station_ids.unique():
        mask = station_ids == station_id
        station_theta = theta_series[mask]

        # Get soil parameters
        station_data = df[df['station_id'] == station_id]
        if len(station_data) > 0:
            sand_pct = station_data[sand_col].iloc[0] if sand_col in station_data.columns else 50.0
            clay_pct = station_data[clay_col].iloc[0] if clay_col in station_data.columns else 20.0
        else:
            sand_pct, clay_pct = 50.0, 20.0

        if pd.isna(sand_pct) or pd.isna(clay_pct):
            sand_pct, clay_pct = 50.0, 20.0

        # Get calibrated VG params
        params = get_calibrated_vg_params(
            station_id, calibrated_params, sand_pct, clay_pct)

        # Convert
        station_psi = []
        for theta in station_theta:
            if pd.isna(theta):
                psi = np.nan
            else:
                try:
                    psi = potential_from_water_content(theta, params)
                    psi = np.clip(psi, -10000, -0.1)
                except:
                    psi = np.nan
            station_psi.append(psi)

        result.loc[mask] = station_psi

    return result


# =============================================================================
# ML-BASED RETENTION CORRECTION
# =============================================================================

class RetentionCorrectionModel:
    """
    ML model that learns direct θ corrections for the ψ→θ conversion.

    Instead of predicting VG parameter adjustments, this directly predicts:
        θ_corrected = θ_ptf + Δθ_ml

    where Δθ_ml is learned from features including ψ, soil properties,
    and the PTF-predicted θ.

    This is more flexible than parameter scaling because:
    - Can learn non-VG shaped corrections
    - Can learn different corrections at different moisture levels
    - Works even when VG parameters are fundamentally wrong
    """

    def __init__(self, config: Optional[RetentionCorrectionConfig] = None):
        self.config = config or RetentionCorrectionConfig()
        self.model = None
        self.feature_cols = None

    def create_features(
        self,
        df: pd.DataFrame,
        psi_col: str = "predicted_psi",
        theta_ptf_col: str = "theta_from_ptf",
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create features for the retention correction model.

        Args:
            df: DataFrame with predictions and soil properties
            psi_col: Column with matric potential predictions
            theta_ptf_col: Column with PTF-converted θ

        Returns:
            Tuple of (DataFrame with features, list of feature names)
        """
        result = df.copy()
        features = []

        # Core features
        if psi_col in df.columns:
            result['log_abs_psi'] = np.log1p(np.abs(df[psi_col]))
            features.append('log_abs_psi')
            features.append(psi_col)

        if theta_ptf_col in df.columns:
            features.append(theta_ptf_col)

        # Soil texture features
        for col in ['sand_pct', 'clay_pct', 'silt_pct', 'organic_carbon_pct']:
            if col in df.columns:
                features.append(col)

        # Effective saturation from PTF
        if theta_ptf_col in df.columns and 'theta_s' in df.columns and 'theta_r' in df.columns:
            result['Se_ptf'] = (df[theta_ptf_col] - df['theta_r']
                                ) / (df['theta_s'] - df['theta_r'])
            result['Se_ptf'] = result['Se_ptf'].clip(0, 1)
            features.append('Se_ptf')

        # Derived features
        if psi_col in df.columns and 'sand_pct' in df.columns:
            # Interaction: sandy soils behave differently at given ψ
            result['psi_sand_interaction'] = df[psi_col] * df['sand_pct'] / 100
            features.append('psi_sand_interaction')

        if psi_col in df.columns and 'clay_pct' in df.columns:
            result['psi_clay_interaction'] = df[psi_col] * df['clay_pct'] / 100
            features.append('psi_clay_interaction')

        # Filter to existing features
        features = [f for f in features if f in result.columns]

        return result, features

    def train(
        self,
        df: pd.DataFrame,
        observed_theta_col: str = "soil_moisture",
        psi_col: str = "predicted_psi",
        theta_ptf_col: str = "theta_from_ptf",
        station_col: str = "station_id",
    ) -> Dict[str, float]:
        """
        Train the retention correction model.

        Target: Δθ = θ_observed - θ_ptf

        Args:
            df: Training DataFrame
            observed_theta_col: Column with observed θ
            psi_col: Column with predicted ψ
            theta_ptf_col: Column with PTF-converted θ
            station_col: Station identifier column

        Returns:
            Dict with training metrics
        """
        logger.info("Training retention correction model...")

        # Create features
        train_df, self.feature_cols = self.create_features(
            df, psi_col, theta_ptf_col)

        # Target: correction needed
        train_df['theta_correction'] = train_df[observed_theta_col] - \
            train_df[theta_ptf_col]

        # Filter valid rows
        valid_mask = (
            train_df[self.feature_cols].notna().all(axis=1) &
            train_df['theta_correction'].notna()
        )
        valid_df = train_df[valid_mask]

        if len(valid_df) < 100:
            logger.warning(
                f"Insufficient training data: {len(valid_df)} samples")
            return {'n_samples': len(valid_df), 'trained': False}

        X = valid_df[self.feature_cols]
        y = valid_df['theta_correction'].values
        groups = valid_df[station_col].values

        # Train with site-blocked CV
        cv = GroupKFold(n_splits=5)
        val_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'n_estimators': self.config.n_estimators,
                'learning_rate': self.config.learning_rate,
                'max_depth': self.config.max_depth,
                'num_leaves': self.config.num_leaves,
                'min_child_samples': self.config.min_child_samples,
                'verbosity': -1,
            }

            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=callbacks,
            )

            y_pred = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            val_scores.append(val_rmse)

        # Train final model on all data
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.n_estimators,
        )

        metrics = {
            'n_samples': len(valid_df),
            'cv_rmse_mean': float(np.mean(val_scores)),
            'cv_rmse_std': float(np.std(val_scores)),
            'trained': True,
        }

        logger.info(
            f"  CV RMSE: {metrics['cv_rmse_mean']:.4f} ± {metrics['cv_rmse_std']:.4f} m³/m³")

        return metrics

    def predict(
        self,
        df: pd.DataFrame,
        psi_col: str = "predicted_psi",
        theta_ptf_col: str = "theta_from_ptf",
    ) -> pd.Series:
        """
        Predict θ corrections.

        Args:
            df: DataFrame with features
            psi_col: Column with predicted ψ
            theta_ptf_col: Column with PTF-converted θ

        Returns:
            Series of θ corrections to add to PTF θ
        """
        if self.model is None:
            raise ValueError("Model not trained")

        pred_df, _ = self.create_features(df, psi_col, theta_ptf_col)
        X = pred_df[self.feature_cols]

        # Handle missing features
        valid_mask = X.notna().all(axis=1)
        result = pd.Series(index=df.index, dtype=float)

        if valid_mask.sum() > 0:
            result.loc[valid_mask] = self.model.predict(X[valid_mask])

        return result

    def apply_correction(
        self,
        df: pd.DataFrame,
        psi_col: str = "predicted_psi",
        theta_ptf_col: str = "theta_from_ptf",
    ) -> pd.Series:
        """
        Apply learned correction to get final θ predictions.

        Args:
            df: DataFrame with PTF-converted θ
            psi_col: Column with predicted ψ
            theta_ptf_col: Column with PTF-converted θ

        Returns:
            Series of corrected θ values
        """
        correction = self.predict(df, psi_col, theta_ptf_col)
        theta_corrected = df[theta_ptf_col] + correction

        # Physical constraints
        theta_corrected = theta_corrected.clip(0.01, 0.65)

        return theta_corrected


# =============================================================================
# EVALUATION IN PSI-SPACE
# =============================================================================

def evaluate_psi_space_metrics(
    observed_psi: np.ndarray,
    predicted_psi: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate predictions in matric potential (ψ) space.

    This is the appropriate space for physics model tuning because:
    - ψ relates directly to soil hydraulic processes
    - Infiltration, drainage, ET all depend on ψ gradients
    - PTF errors don't contaminate the evaluation

    Args:
        observed_psi: Observed matric potentials (kPa)
        predicted_psi: Predicted matric potentials (kPa)

    Returns:
        Dict with evaluation metrics
    """
    valid_mask = ~np.isnan(observed_psi) & ~np.isnan(predicted_psi)
    obs = observed_psi[valid_mask]
    pred = predicted_psi[valid_mask]

    if len(obs) < 10:
        return {
            'n_samples': len(obs),
            'rmse_kpa': np.nan,
            'mae_kpa': np.nan,
            'kge': np.nan,
            'nse': np.nan,
            'r2': np.nan,
            'bias_kpa': np.nan,
        }

    # Basic metrics
    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    mae = np.mean(np.abs(pred - obs))
    bias = np.mean(pred - obs)

    # Correlation
    if np.std(obs) > 0 and np.std(pred) > 0:
        r = np.corrcoef(obs, pred)[0, 1]
        r2 = r ** 2
    else:
        r = np.nan
        r2 = np.nan

    # NSE (Nash-Sutcliffe Efficiency)
    ss_res = np.sum((pred - obs) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    if ss_tot > 0:
        nse = 1 - ss_res / ss_tot
    else:
        nse = np.nan

    # KGE (Kling-Gupta Efficiency)
    if np.std(obs) > 0 and np.std(pred) > 0:
        r_kge = r
        alpha = np.std(pred) / np.std(obs)
        beta = np.mean(pred) / np.mean(obs) if np.mean(obs) != 0 else np.nan
        if not np.isnan(beta):
            kge = 1 - np.sqrt((r_kge - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        else:
            kge = np.nan
    else:
        kge = np.nan

    return {
        'n_samples': len(obs),
        'rmse_kpa': float(rmse),
        'mae_kpa': float(mae),
        'kge': float(kge),
        'nse': float(nse),
        'r2': float(r2),
        'bias_kpa': float(bias),
    }


def evaluate_log_psi_space_metrics(
    observed_psi: np.ndarray,
    predicted_psi: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate predictions in log-transformed ψ space.

    Log-transform is often more appropriate because:
    - ψ spans orders of magnitude (-0.1 to -10000 kPa)
    - Errors at dry conditions shouldn't dominate
    - pF scale (log10 of cm H2O) is standard in soil physics

    Args:
        observed_psi: Observed matric potentials (kPa, negative)
        predicted_psi: Predicted matric potentials (kPa, negative)

    Returns:
        Dict with evaluation metrics in log space
    """
    valid_mask = (
        ~np.isnan(observed_psi) & ~np.isnan(predicted_psi) &
        (observed_psi < 0) & (predicted_psi < 0)  # Must be negative
    )
    obs = observed_psi[valid_mask]
    pred = predicted_psi[valid_mask]

    if len(obs) < 10:
        return {
            'n_samples': len(obs),
            'rmse_log_kpa': np.nan,
            'mae_log_kpa': np.nan,
            'kge_log': np.nan,
            'nse_log': np.nan,
            'r2_log': np.nan,
        }

    # Log transform (of absolute values)
    log_obs = np.log10(np.abs(obs))
    log_pred = np.log10(np.abs(pred))

    # Metrics in log space
    rmse = np.sqrt(np.mean((log_pred - log_obs) ** 2))
    mae = np.mean(np.abs(log_pred - log_obs))

    # Correlation
    if np.std(log_obs) > 0 and np.std(log_pred) > 0:
        r = np.corrcoef(log_obs, log_pred)[0, 1]
        r2 = r ** 2
    else:
        r = np.nan
        r2 = np.nan

    # NSE in log space
    ss_res = np.sum((log_pred - log_obs) ** 2)
    ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
    if ss_tot > 0:
        nse = 1 - ss_res / ss_tot
    else:
        nse = np.nan

    # KGE in log space
    if np.std(log_obs) > 0 and np.std(log_pred) > 0 and not np.isnan(r):
        alpha = np.std(log_pred) / np.std(log_obs)
        beta = np.mean(log_pred) / \
            np.mean(log_obs) if np.mean(log_obs) != 0 else np.nan
        if not np.isnan(beta):
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        else:
            kge = np.nan
    else:
        kge = np.nan

    return {
        'n_samples': len(obs),
        'rmse_log_kpa': float(rmse),
        'mae_log_kpa': float(mae),
        'kge_log': float(kge),
        'nse_log': float(nse),
        'r2_log': float(r2),
    }


def evaluate_theta_space_metrics(
    observed_theta: np.ndarray,
    predicted_theta: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate predictions in volumetric water content (θ) space.

    This is the appropriate space for irrigation applications because:
    - θ directly relates to plant-available water
    - Farmers care about water content, not matric potential
    - Irrigation decisions are based on θ thresholds

    Args:
        observed_theta: Observed water contents (m³/m³)
        predicted_theta: Predicted water contents (m³/m³)

    Returns:
        Dict with evaluation metrics
    """
    valid_mask = ~np.isnan(observed_theta) & ~np.isnan(predicted_theta)
    obs = observed_theta[valid_mask]
    pred = predicted_theta[valid_mask]

    if len(obs) < 10:
        return {
            'n_samples': len(obs),
            'rmse': np.nan,
            'mae': np.nan,
            'kge': np.nan,
            'nse': np.nan,
            'r2': np.nan,
            'bias': np.nan,
        }

    # Basic metrics
    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    mae = np.mean(np.abs(pred - obs))
    bias = np.mean(pred - obs)

    # Correlation
    if np.std(obs) > 0 and np.std(pred) > 0:
        r = np.corrcoef(obs, pred)[0, 1]
        r2 = r ** 2
    else:
        r = np.nan
        r2 = np.nan

    # NSE (Nash-Sutcliffe Efficiency)
    ss_res = np.sum((pred - obs) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    if ss_tot > 0:
        nse = 1 - ss_res / ss_tot
    else:
        nse = np.nan

    # KGE (Kling-Gupta Efficiency)
    if np.std(obs) > 0 and np.std(pred) > 0:
        r_kge = r
        alpha = np.std(pred) / np.std(obs)
        beta = np.mean(pred) / np.mean(obs) if np.mean(obs) != 0 else np.nan
        if not np.isnan(beta):
            kge = 1 - np.sqrt((r_kge - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        else:
            kge = np.nan
    else:
        kge = np.nan

    return {
        'n_samples': len(obs),
        'rmse': float(rmse),
        'mae': float(mae),
        'kge': float(kge),
        'nse': float(nse),
        'r2': float(r2),
        'bias': float(bias),
    }
