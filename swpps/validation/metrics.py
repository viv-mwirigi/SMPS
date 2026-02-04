"""
Validation Metrics for SWPPS.

This module provides comprehensive metrics for evaluating soil water
potential predictions, including:
- Standard metrics (RMSE, MAE, R², Bias)
- Physics-aware metrics (KGE, NSE, ubRMSE)
- Uncertainty calibration metrics
- Temporal structure metrics
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("swpps.validation.metrics")


@dataclass
class ValidationMetrics:
    """
    Comprehensive validation metrics for matric potential predictions.

    All metrics are computed in kPa (matric potential units).
    """
    # Sample information
    n_samples: int = 0
    n_valid: int = 0

    # Standard error metrics
    rmse: float = np.nan          # Root Mean Square Error (kPa)
    mae: float = np.nan           # Mean Absolute Error (kPa)
    mbe: float = np.nan           # Mean Bias Error (kPa)

    # Correlation metrics
    r: float = np.nan             # Pearson correlation
    r_squared: float = np.nan     # Coefficient of determination

    # Efficiency metrics
    nse: float = np.nan           # Nash-Sutcliffe Efficiency
    kge: float = np.nan           # Kling-Gupta Efficiency

    # KGE decomposition
    kge_r: float = np.nan         # Correlation component
    kge_alpha: float = np.nan     # Variability ratio
    kge_beta: float = np.nan      # Bias ratio

    # Advanced metrics
    ubrmse: float = np.nan        # Unbiased RMSE
    mape: float = np.nan          # Mean Absolute Percentage Error

    # Temporal structure
    lag1_autocorr_obs: float = np.nan
    lag1_autocorr_pred: float = np.nan
    autocorr_error: float = np.nan

    # Uncertainty calibration
    coverage_90: float = np.nan   # % of obs within 90% prediction interval
    sharpness: float = np.nan     # Average width of prediction intervals

    # Horizon-specific
    horizon_hours: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "N_samples": self.n_samples,
            "N_valid": self.n_valid,
            "RMSE_kPa": self.rmse,
            "MAE_kPa": self.mae,
            "MBE_kPa": self.mbe,
            "R": self.r,
            "R²": self.r_squared,
            "NSE": self.nse,
            "KGE": self.kge,
            "KGE_r": self.kge_r,
            "KGE_α": self.kge_alpha,
            "KGE_β": self.kge_beta,
            "ubRMSE_kPa": self.ubrmse,
            "MAPE_%": self.mape,
            "Lag1_AC_obs": self.lag1_autocorr_obs,
            "Lag1_AC_pred": self.lag1_autocorr_pred,
            "AC_error": self.autocorr_error,
            "Coverage_90%": self.coverage_90,
            "Sharpness_kPa": self.sharpness,
            "Horizon_h": self.horizon_hours,
        }

    def __str__(self) -> str:
        return (
            f"ValidationMetrics(N={self.n_valid}, "
            f"RMSE={self.rmse:.2f} kPa, "
            f"R²={self.r_squared:.3f}, "
            f"KGE={self.kge:.3f})"
        )


def compute_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    horizon_hours: int = 0,
) -> ValidationMetrics:
    """
    Compute all validation metrics.

    Args:
        observed: Observed matric potential (kPa)
        predicted: Predicted matric potential (kPa)
        lower_bound: Lower prediction bound (10th percentile)
        upper_bound: Upper prediction bound (90th percentile)
        horizon_hours: Forecast horizon in hours

    Returns:
        ValidationMetrics dataclass
    """
    # Handle arrays and masks
    obs = np.asarray(observed).flatten()
    pred = np.asarray(predicted).flatten()

    # Valid mask
    valid = np.isfinite(obs) & np.isfinite(pred)
    n_samples = len(obs)
    n_valid = np.sum(valid)

    if n_valid < 2:
        logger.warning("Insufficient valid samples for metrics: %d", n_valid)
        return ValidationMetrics(n_samples=n_samples, n_valid=n_valid)

    obs_v = obs[valid]
    pred_v = pred[valid]

    # Basic metrics
    errors = pred_v - obs_v
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    mbe = np.mean(errors)

    # Correlation
    r = np.corrcoef(obs_v, pred_v)[0, 1] if n_valid > 2 else np.nan
    r_squared = r ** 2 if np.isfinite(r) else np.nan

    # NSE
    nse = compute_nse(obs_v, pred_v)

    # KGE and components
    kge, kge_r, kge_alpha, kge_beta = compute_kge(obs_v, pred_v)

    # Unbiased RMSE
    ubrmse = compute_ubrmse(obs_v, pred_v)

    # MAPE (avoid division by zero, use absolute values)
    abs_obs = np.abs(obs_v)
    mape_mask = abs_obs > 1.0  # Only compute MAPE where obs is significant
    if np.sum(mape_mask) > 0:
        mape = 100 * np.mean(np.abs(errors[mape_mask]) / abs_obs[mape_mask])
    else:
        mape = np.nan

    # Temporal structure
    lag1_ac_obs = _compute_lag1_autocorr(obs_v)
    lag1_ac_pred = _compute_lag1_autocorr(pred_v)
    ac_error = abs(lag1_ac_obs - lag1_ac_pred) if np.isfinite(
        lag1_ac_obs) and np.isfinite(lag1_ac_pred) else np.nan

    # Uncertainty calibration
    coverage_90 = np.nan
    sharpness = np.nan

    if lower_bound is not None and upper_bound is not None:
        lower = np.asarray(lower_bound).flatten()[valid]
        upper = np.asarray(upper_bound).flatten()[valid]

        within = (obs_v >= lower) & (obs_v <= upper)
        coverage_90 = 100 * np.mean(within)
        sharpness = np.mean(upper - lower)

    return ValidationMetrics(
        n_samples=n_samples,
        n_valid=n_valid,
        rmse=rmse,
        mae=mae,
        mbe=mbe,
        r=r,
        r_squared=r_squared,
        nse=nse,
        kge=kge,
        kge_r=kge_r,
        kge_alpha=kge_alpha,
        kge_beta=kge_beta,
        ubrmse=ubrmse,
        mape=mape,
        lag1_autocorr_obs=lag1_ac_obs,
        lag1_autocorr_pred=lag1_ac_pred,
        autocorr_error=ac_error,
        coverage_90=coverage_90,
        sharpness=sharpness,
        horizon_hours=horizon_hours,
    )


def compute_nse(obs: np.ndarray, pred: np.ndarray) -> float:
    """
    Nash-Sutcliffe Efficiency.

    NSE = 1 - Σ(obs - pred)² / Σ(obs - mean(obs))²

    NSE = 1: Perfect prediction
    NSE = 0: Prediction as good as mean
    NSE < 0: Prediction worse than mean
    """
    obs_mean = np.mean(obs)
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs_mean) ** 2)

    if ss_tot < 1e-10:
        return np.nan

    return 1.0 - ss_res / ss_tot


def compute_kge(
    obs: np.ndarray,
    pred: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Kling-Gupta Efficiency with decomposition.

    KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)

    where:
        r = correlation coefficient
        α = σ_pred / σ_obs (variability ratio)
        β = μ_pred / μ_obs (bias ratio)

    Returns:
        Tuple of (KGE, r, alpha, beta)
    """
    # Correlation
    r = np.corrcoef(obs, pred)[0, 1]

    # Variability ratio
    std_obs = np.std(obs)
    std_pred = np.std(pred)
    alpha = std_pred / std_obs if std_obs > 1e-10 else np.nan

    # Bias ratio
    mean_obs = np.mean(obs)
    mean_pred = np.mean(pred)
    beta = mean_pred / mean_obs if abs(mean_obs) > 1e-10 else np.nan

    # KGE
    if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta):
        kge = 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    else:
        kge = np.nan

    return kge, r, alpha, beta


def compute_ubrmse(obs: np.ndarray, pred: np.ndarray) -> float:
    """
    Unbiased Root Mean Square Error.

    ubRMSE = sqrt(RMSE² - bias²)

    This removes systematic bias to show the random error component.
    """
    bias = np.mean(pred - obs)
    mse = np.mean((pred - obs) ** 2)

    variance = mse - bias ** 2

    if variance < 0:
        # Numerical precision issue
        return 0.0

    return np.sqrt(variance)


def _compute_lag1_autocorr(x: np.ndarray) -> float:
    """Compute lag-1 autocorrelation."""
    if len(x) < 3:
        return np.nan

    return np.corrcoef(x[:-1], x[1:])[0, 1]


def compute_per_site_metrics(
    df: pd.DataFrame,
    site_col: str = "site_id",
    obs_col: str = "psi_observed_kpa",
    pred_col: str = "psi_predicted_kpa",
) -> pd.DataFrame:
    """
    Compute metrics separately for each site.

    Args:
        df: DataFrame with predictions
        site_col: Column name for site identifier
        obs_col: Column name for observations
        pred_col: Column name for predictions

    Returns:
        DataFrame with per-site metrics
    """
    results = []

    for site_id, group in df.groupby(site_col):
        obs = group[obs_col].values
        pred = group[pred_col].values

        metrics = compute_metrics(obs, pred)
        metrics_dict = metrics.to_dict()
        metrics_dict["site_id"] = site_id
        results.append(metrics_dict)

    return pd.DataFrame(results)


def compute_horizon_metrics(
    df: pd.DataFrame,
    horizon_col: str = "horizon_hours",
    obs_col: str = "psi_observed_kpa",
    pred_col: str = "psi_predicted_kpa",
) -> pd.DataFrame:
    """
    Compute metrics separately for each forecast horizon.

    Args:
        df: DataFrame with predictions
        horizon_col: Column name for horizon
        obs_col: Column name for observations
        pred_col: Column name for predictions

    Returns:
        DataFrame with per-horizon metrics
    """
    results = []

    for horizon, group in df.groupby(horizon_col):
        obs = group[obs_col].values
        pred = group[pred_col].values

        metrics = compute_metrics(obs, pred, horizon_hours=int(horizon))
        results.append(metrics.to_dict())

    return pd.DataFrame(results)
