"""
Validation metrics for soil moisture predictions.
Provides comprehensive statistical metrics for model evaluation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of validation metrics"""
    ERROR = "error"          # RMSE, MAE, etc.
    CORRELATION = "correlation"  # R², correlation
    EFFICIENCY = "efficiency"    # NSE, KGE
    BIAS = "bias"            # Mean bias, relative bias


@dataclass
class ValidationMetrics:
    """Container for validation metrics"""
    # Error metrics
    rmse: float = np.nan
    mae: float = np.nan
    mse: float = np.nan

    # Bias metrics
    bias: float = np.nan  # Mean bias (predicted - observed)
    relative_bias: float = np.nan  # Bias / mean(observed)
    percent_bias: float = np.nan  # 100 * bias / mean(observed)

    # Correlation metrics
    r_squared: float = np.nan
    pearson_r: float = np.nan
    spearman_r: float = np.nan

    # Efficiency metrics
    nse: float = np.nan  # Nash-Sutcliffe Efficiency
    kge: float = np.nan  # Kling-Gupta Efficiency

    # Sample info
    n_samples: int = 0
    n_valid: int = 0

    # Additional statistics
    obs_mean: float = np.nan
    obs_std: float = np.nan
    pred_mean: float = np.nan
    pred_std: float = np.nan

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'RMSE': self.rmse,
            'MAE': self.mae,
            'MSE': self.mse,
            'Bias': self.bias,
            'Relative_Bias': self.relative_bias,
            'Percent_Bias': self.percent_bias,
            'R²': self.r_squared,
            'Pearson_r': self.pearson_r,
            'Spearman_r': self.spearman_r,
            'NSE': self.nse,
            'KGE': self.kge,
            'N_samples': self.n_samples,
            'N_valid': self.n_valid,
            'Obs_mean': self.obs_mean,
            'Obs_std': self.obs_std,
            'Pred_mean': self.pred_mean,
            'Pred_std': self.pred_std
        }

    def summary(self) -> str:
        """Return formatted summary string"""
        return (
            f"Validation Metrics (n={self.n_valid}):\n"
            f"  RMSE:    {self.rmse:.4f}\n"
            f"  MAE:     {self.mae:.4f}\n"
            f"  Bias:    {self.bias:+.4f} ({self.percent_bias:+.1f}%)\n"
            f"  R²:      {self.r_squared:.4f}\n"
            f"  NSE:     {self.nse:.4f}\n"
            f"  KGE:     {self.kge:.4f}"
        )


class ValidationEngine:
    """
    Engine for computing validation metrics between predictions and observations.
    """

    def __init__(self, min_samples: int = 10):
        """
        Initialize validation engine.

        Args:
            min_samples: Minimum samples required for valid statistics
        """
        self.min_samples = min_samples
        self.logger = logging.getLogger("smps.validation")

    def compute_metrics(self,
                       observed: np.ndarray,
                       predicted: np.ndarray,
                       weights: Optional[np.ndarray] = None) -> ValidationMetrics:
        """
        Compute all validation metrics.

        Args:
            observed: Array of observed values
            predicted: Array of predicted values
            weights: Optional weights for weighted metrics

        Returns:
            ValidationMetrics with all computed metrics
        """
        # Convert to numpy arrays
        obs = np.asarray(observed).flatten()
        pred = np.asarray(predicted).flatten()

        # Create mask for valid pairs (both non-NaN)
        valid_mask = ~(np.isnan(obs) | np.isnan(pred))

        n_samples = len(obs)
        n_valid = np.sum(valid_mask)

        if n_valid < self.min_samples:
            self.logger.warning(
                f"Insufficient valid samples: {n_valid} < {self.min_samples}"
            )
            return ValidationMetrics(n_samples=n_samples, n_valid=n_valid)

        # Filter to valid pairs
        obs_valid = obs[valid_mask]
        pred_valid = pred[valid_mask]

        if weights is not None:
            weights = np.asarray(weights).flatten()[valid_mask]

        # Compute all metrics
        metrics = ValidationMetrics(
            n_samples=n_samples,
            n_valid=n_valid,
            obs_mean=np.mean(obs_valid),
            obs_std=np.std(obs_valid),
            pred_mean=np.mean(pred_valid),
            pred_std=np.std(pred_valid)
        )

        # Error metrics
        metrics.rmse = self._rmse(obs_valid, pred_valid, weights)
        metrics.mae = self._mae(obs_valid, pred_valid, weights)
        metrics.mse = self._mse(obs_valid, pred_valid, weights)

        # Bias metrics
        metrics.bias = self._mean_bias(obs_valid, pred_valid)
        metrics.relative_bias = self._relative_bias(obs_valid, pred_valid)
        metrics.percent_bias = self._percent_bias(obs_valid, pred_valid)

        # Correlation metrics
        metrics.r_squared = self._r_squared(obs_valid, pred_valid)
        metrics.pearson_r = self._pearson_r(obs_valid, pred_valid)
        metrics.spearman_r = self._spearman_r(obs_valid, pred_valid)

        # Efficiency metrics
        metrics.nse = self._nse(obs_valid, pred_valid)
        metrics.kge = self._kge(obs_valid, pred_valid)

        return metrics

    def compute_metrics_by_group(self,
                                 df: pd.DataFrame,
                                 obs_col: str,
                                 pred_col: str,
                                 group_col: str) -> Dict[str, ValidationMetrics]:
        """
        Compute metrics grouped by a categorical column.

        Args:
            df: DataFrame with observations and predictions
            obs_col: Column name for observations
            pred_col: Column name for predictions
            group_col: Column to group by

        Returns:
            Dictionary mapping group values to metrics
        """
        results = {}

        for group_name, group_df in df.groupby(group_col):
            metrics = self.compute_metrics(
                group_df[obs_col].values,
                group_df[pred_col].values
            )
            results[group_name] = metrics

        return results

    def compute_temporal_metrics(self,
                                df: pd.DataFrame,
                                obs_col: str,
                                pred_col: str,
                                date_col: str,
                                freq: str = 'M') -> pd.DataFrame:
        """
        Compute metrics over time periods.

        Args:
            df: DataFrame with observations and predictions
            obs_col: Column name for observations
            pred_col: Column name for predictions
            date_col: Column name for dates
            freq: Frequency for grouping ('D', 'W', 'M', 'Q', 'Y')

        Returns:
            DataFrame with metrics per time period
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df['period'] = df[date_col].dt.to_period(freq)

        results = []
        for period, group in df.groupby('period'):
            metrics = self.compute_metrics(
                group[obs_col].values,
                group[pred_col].values
            )
            results.append({
                'period': period,
                **metrics.to_dict()
            })

        return pd.DataFrame(results)

    # Individual metric implementations

    def _rmse(self, obs: np.ndarray, pred: np.ndarray,
              weights: Optional[np.ndarray] = None) -> float:
        """Root Mean Square Error"""
        errors = pred - obs
        if weights is not None:
            return np.sqrt(np.average(errors**2, weights=weights))
        return np.sqrt(np.mean(errors**2))

    def _mae(self, obs: np.ndarray, pred: np.ndarray,
             weights: Optional[np.ndarray] = None) -> float:
        """Mean Absolute Error"""
        errors = np.abs(pred - obs)
        if weights is not None:
            return np.average(errors, weights=weights)
        return np.mean(errors)

    def _mse(self, obs: np.ndarray, pred: np.ndarray,
             weights: Optional[np.ndarray] = None) -> float:
        """Mean Square Error"""
        errors = pred - obs
        if weights is not None:
            return np.average(errors**2, weights=weights)
        return np.mean(errors**2)

    def _mean_bias(self, obs: np.ndarray, pred: np.ndarray) -> float:
        """Mean Bias (predicted - observed)"""
        return np.mean(pred - obs)

    def _relative_bias(self, obs: np.ndarray, pred: np.ndarray) -> float:
        """Relative Bias (bias / mean observation)"""
        obs_mean = np.mean(obs)
        if abs(obs_mean) < 1e-10:
            return np.nan
        return self._mean_bias(obs, pred) / obs_mean

    def _percent_bias(self, obs: np.ndarray, pred: np.ndarray) -> float:
        """Percent Bias"""
        return self._relative_bias(obs, pred) * 100

    def _r_squared(self, obs: np.ndarray, pred: np.ndarray) -> float:
        """Coefficient of determination (R²)"""
        ss_res = np.sum((obs - pred)**2)
        ss_tot = np.sum((obs - np.mean(obs))**2)
        if ss_tot < 1e-10:
            return np.nan
        return 1 - (ss_res / ss_tot)

    def _pearson_r(self, obs: np.ndarray, pred: np.ndarray) -> float:
        """Pearson correlation coefficient"""
        if len(obs) < 2:
            return np.nan

        obs_centered = obs - np.mean(obs)
        pred_centered = pred - np.mean(pred)

        numerator = np.sum(obs_centered * pred_centered)
        denominator = np.sqrt(np.sum(obs_centered**2) * np.sum(pred_centered**2))

        if denominator < 1e-10:
            return np.nan
        return numerator / denominator

    def _spearman_r(self, obs: np.ndarray, pred: np.ndarray) -> float:
        """Spearman rank correlation coefficient"""
        from scipy import stats
        if len(obs) < 2:
            return np.nan
        corr, _ = stats.spearmanr(obs, pred)
        return corr

    def _nse(self, obs: np.ndarray, pred: np.ndarray) -> float:
        """
        Nash-Sutcliffe Efficiency.

        NSE = 1 - sum((obs - pred)²) / sum((obs - mean(obs))²)

        NSE = 1: Perfect prediction
        NSE = 0: Prediction as good as mean
        NSE < 0: Mean is better predictor
        """
        ss_res = np.sum((obs - pred)**2)
        ss_tot = np.sum((obs - np.mean(obs))**2)

        if ss_tot < 1e-10:
            return np.nan
        return 1 - (ss_res / ss_tot)

    def _kge(self, obs: np.ndarray, pred: np.ndarray) -> float:
        """
        Kling-Gupta Efficiency.

        KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)

        where:
        - r = Pearson correlation
        - α = σ_pred / σ_obs (variability ratio)
        - β = μ_pred / μ_obs (bias ratio)

        KGE = 1: Perfect prediction
        KGE ≥ -0.41: Better than mean flow benchmark
        """
        r = self._pearson_r(obs, pred)

        obs_std = np.std(obs)
        pred_std = np.std(pred)

        if obs_std < 1e-10:
            return np.nan

        alpha = pred_std / obs_std  # Variability ratio

        obs_mean = np.mean(obs)
        if abs(obs_mean) < 1e-10:
            return np.nan

        beta = np.mean(pred) / obs_mean  # Bias ratio

        # KGE formula
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

        return kge


def compute_quick_metrics(observed: np.ndarray,
                         predicted: np.ndarray) -> Dict[str, float]:
    """
    Quick function to compute common metrics.

    Args:
        observed: Array of observed values
        predicted: Array of predicted values

    Returns:
        Dictionary with RMSE, MAE, Bias, R², NSE
    """
    engine = ValidationEngine(min_samples=3)
    metrics = engine.compute_metrics(observed, predicted)

    return {
        'RMSE': metrics.rmse,
        'MAE': metrics.mae,
        'Bias': metrics.bias,
        'R²': metrics.r_squared,
        'NSE': metrics.nse,
        'KGE': metrics.kge
    }


def print_metrics_comparison(metrics_dict: Dict[str, ValidationMetrics],
                            title: str = "Model Comparison"):
    """
    Print formatted comparison of multiple model metrics.

    Args:
        metrics_dict: Dictionary mapping model names to ValidationMetrics
        title: Title for the comparison table
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")

    # Header
    print(f"{'Metric':<15}", end="")
    for name in metrics_dict.keys():
        print(f"{name:>12}", end="")
    print()
    print("-" * 60)

    # Metrics to display
    metric_names = ['RMSE', 'MAE', 'Bias', 'R²', 'NSE', 'KGE']

    for metric in metric_names:
        print(f"{metric:<15}", end="")
        for name, m in metrics_dict.items():
            value = m.to_dict().get(metric, np.nan)
            if np.isnan(value):
                print(f"{'N/A':>12}", end="")
            else:
                print(f"{value:>12.4f}", end="")
        print()

    print(f"{'='*60}\n")
