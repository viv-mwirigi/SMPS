"""
Physics-Based Validation Metrics (Gap 10).

This module extends the basic validation metrics with physics-aware assessment:

1. ADDITIONAL METRICS: ubRMSE, MAPE, KGE decomposition
2. MASS BALANCE VALIDATION: Cumulative error tracking
3. EXTREME EVENT VALIDATION: Drought/flood period metrics
4. TEMPORAL STRUCTURE: Autocorrelation, seasonal phase
5. MULTI-DEPTH CONSISTENCY: Layer coherence validation
6. FLUX VALIDATION: ET and runoff comparisons

References:
- Gupta et al. (2009) Decomposition of KGE
- Entekhabi et al. (2010) SMAP validation protocols
- Kumar et al. (2012) Soil moisture evaluation framework
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
from scipy import signal

logger = logging.getLogger(__name__)


# =============================================================================
# EXTENDED METRICS
# =============================================================================

@dataclass
class ExtendedMetrics:
    """
    Extended validation metrics including physics-based assessments.
    """
    # Standard metrics
    rmse: float = np.nan
    mae: float = np.nan
    bias: float = np.nan
    r_squared: float = np.nan
    nse: float = np.nan
    kge: float = np.nan

    # Additional metrics (Gap 10)
    ubrmse: float = np.nan      # Unbiased RMSE
    mape: float = np.nan        # Mean Absolute Percentage Error

    # KGE decomposition
    kge_r: float = np.nan       # Correlation component
    kge_alpha: float = np.nan   # Variability ratio
    kge_beta: float = np.nan    # Bias ratio

    # Temporal structure
    lag1_autocorr_obs: float = np.nan
    lag1_autocorr_pred: float = np.nan
    autocorr_error: float = np.nan  # Difference in autocorrelation

    # Phase metrics
    seasonal_phase_error_days: float = np.nan

    # Sample info
    n_samples: int = 0
    n_valid: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'RMSE': self.rmse,
            'ubRMSE': self.ubrmse,
            'MAE': self.mae,
            'MAPE': self.mape,
            'Bias': self.bias,
            'R²': self.r_squared,
            'NSE': self.nse,
            'KGE': self.kge,
            'KGE_r': self.kge_r,
            'KGE_alpha': self.kge_alpha,
            'KGE_beta': self.kge_beta,
            'Lag1_AC_obs': self.lag1_autocorr_obs,
            'Lag1_AC_pred': self.lag1_autocorr_pred,
            'AC_error': self.autocorr_error,
            'Phase_error_days': self.seasonal_phase_error_days,
            'N_samples': self.n_samples,
            'N_valid': self.n_valid
        }


def compute_ubrmse(obs: np.ndarray, pred: np.ndarray) -> float:
    """
    Unbiased Root Mean Square Error.

    ubRMSE = sqrt(RMSE² - bias²)

    This removes systematic bias to show random error component.
    """
    bias = np.mean(pred - obs)
    mse = np.mean((pred - obs)**2)
    ubrmse_sq = mse - bias**2
    return np.sqrt(max(0, ubrmse_sq))


def compute_mape(obs: np.ndarray, pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Mean Absolute Percentage Error.

    MAPE = 100 × mean(|pred - obs| / |obs|)

    Epsilon prevents division by zero for small observations.
    """
    abs_obs = np.abs(obs)
    valid = abs_obs > epsilon
    if np.sum(valid) < 2:
        return np.nan

    ape = np.abs(pred[valid] - obs[valid]) / abs_obs[valid]
    return 100 * np.mean(ape)


def compute_kge_decomposition(
    obs: np.ndarray,
    pred: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Compute KGE with full decomposition.

    KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)

    Returns: (KGE, r, alpha, beta)
    """
    # Correlation
    if len(obs) < 2:
        return np.nan, np.nan, np.nan, np.nan

    obs_centered = obs - np.mean(obs)
    pred_centered = pred - np.mean(pred)

    num = np.sum(obs_centered * pred_centered)
    denom = np.sqrt(np.sum(obs_centered**2) * np.sum(pred_centered**2))

    if denom < 1e-10:
        return np.nan, np.nan, np.nan, np.nan

    r = num / denom

    # Variability ratio
    obs_std = np.std(obs)
    pred_std = np.std(pred)

    if obs_std < 1e-10:
        return np.nan, r, np.nan, np.nan

    alpha = pred_std / obs_std

    # Bias ratio
    obs_mean = np.mean(obs)
    if abs(obs_mean) < 1e-10:
        return np.nan, r, alpha, np.nan

    beta = np.mean(pred) / obs_mean

    # KGE
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    return kge, r, alpha, beta


def compute_autocorrelation(x: np.ndarray, lag: int = 1) -> float:
    """Compute autocorrelation at specified lag"""
    # ACF at lag 0 is always 1.0
    if lag == 0:
        return 1.0

    n = len(x)
    # Need at least lag+2 elements to compute meaningful autocorrelation
    # Also handle edge case where lag >= n would produce empty arrays
    if n < lag + 2 or lag >= n:
        return np.nan

    x_centered = x - np.mean(x)
    var = np.var(x)

    if var < 1e-10:
        return np.nan

    acf = np.sum(x_centered[:-lag] * x_centered[lag:]) / ((n - lag) * var)
    return acf


# =============================================================================
# MASS BALANCE VALIDATION
# =============================================================================

@dataclass
class MassBalanceValidation:
    """
    Validate mass balance closure for water balance models.

    Checks that: ΔS = P - ET - R - D
    where:
        ΔS = storage change
        P = precipitation
        ET = evapotranspiration
        R = runoff
        D = deep drainage
    """
    # Cumulative values (mm)
    cumulative_precip: float = 0.0
    cumulative_et: float = 0.0
    cumulative_runoff: float = 0.0
    cumulative_drainage: float = 0.0
    cumulative_storage_change: float = 0.0

    # Error tracking
    cumulative_error: float = 0.0
    max_daily_error: float = 0.0
    n_days: int = 0

    # Thresholds
    seasonal_threshold_mm: float = 10.0  # Max acceptable error per season
    daily_threshold_mm: float = 1.0      # Max acceptable daily error

    def add_timestep(
        self,
        precip: float,
        et: float,
        runoff: float,
        drainage: float,
        storage_change: float
    ):
        """Add one timestep to the running totals"""
        self.cumulative_precip += precip
        self.cumulative_et += et
        self.cumulative_runoff += runoff
        self.cumulative_drainage += drainage
        self.cumulative_storage_change += storage_change

        # Calculate expected vs actual
        expected_change = precip - et - runoff - drainage
        error = storage_change - expected_change

        self.cumulative_error += error
        self.max_daily_error = max(self.max_daily_error, abs(error))
        self.n_days += 1

    def is_valid(self) -> bool:
        """Check if mass balance is within acceptable limits"""
        return abs(self.cumulative_error) < self.seasonal_threshold_mm

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics"""
        return {
            'cumulative_precip_mm': self.cumulative_precip,
            'cumulative_et_mm': self.cumulative_et,
            'cumulative_runoff_mm': self.cumulative_runoff,
            'cumulative_drainage_mm': self.cumulative_drainage,
            'cumulative_storage_change_mm': self.cumulative_storage_change,
            'cumulative_error_mm': self.cumulative_error,
            'mean_daily_error_mm': self.cumulative_error / max(1, self.n_days),
            'max_daily_error_mm': self.max_daily_error,
            'n_days': self.n_days,
            'is_valid': self.is_valid()
        }


def validate_mass_balance_series(
    df: pd.DataFrame,
    precip_col: str = 'precipitation_mm',
    et_col: str = 'et_mm',
    runoff_col: str = 'runoff_mm',
    drainage_col: str = 'drainage_mm',
    storage_cols: Optional[List[str]] = None,
    threshold_mm: float = 10.0
) -> MassBalanceValidation:
    """
    Validate mass balance for a time series.

    Args:
        df: DataFrame with flux columns
        precip_col: Precipitation column name
        et_col: ET column name
        runoff_col: Runoff column name
        drainage_col: Deep drainage column name
        storage_cols: Columns to sum for storage (e.g., ['theta_0', 'theta_1', 'theta_2'])
        threshold_mm: Acceptable seasonal error

    Returns:
        MassBalanceValidation with results
    """
    validator = MassBalanceValidation(seasonal_threshold_mm=threshold_mm)

    # Calculate storage if columns provided
    if storage_cols:
        storage = df[storage_cols].sum(axis=1)
        storage_change = storage.diff().fillna(0)
    else:
        storage_change = pd.Series(0, index=df.index)

    for idx in df.index:
        validator.add_timestep(
            precip=df.loc[idx, precip_col] if precip_col in df else 0,
            et=df.loc[idx, et_col] if et_col in df else 0,
            runoff=df.loc[idx, runoff_col] if runoff_col in df else 0,
            drainage=df.loc[idx, drainage_col] if drainage_col in df else 0,
            storage_change=storage_change.loc[idx]
        )

    return validator


# =============================================================================
# EXTREME EVENT VALIDATION
# =============================================================================

class ExtremeEventType(Enum):
    """Types of extreme events"""
    DROUGHT = "drought"
    FLOOD = "flood"
    NORMAL = "normal"


@dataclass
class ExtremeEventMetrics:
    """
    Metrics for extreme event periods.
    """
    event_type: ExtremeEventType
    n_events: int = 0
    n_days: int = 0

    # Performance during events
    rmse: float = np.nan
    bias: float = np.nan
    r_squared: float = np.nan

    # Detection metrics
    hit_rate: float = np.nan  # True positives / actual events
    false_alarm_rate: float = np.nan

    # Timing
    mean_onset_error_days: float = np.nan
    mean_duration_error_days: float = np.nan


def identify_drought_periods(
    soil_moisture: np.ndarray,
    threshold_percentile: float = 20.0,
    min_duration_days: int = 7
) -> List[Tuple[int, int]]:
    """
    Identify drought periods based on soil moisture percentile.

    Args:
        soil_moisture: Time series of soil moisture
        threshold_percentile: Percentile below which is drought
        min_duration_days: Minimum consecutive days to define drought

    Returns:
        List of (start_idx, end_idx) tuples for drought periods
    """
    threshold = np.nanpercentile(soil_moisture, threshold_percentile)
    is_dry = soil_moisture < threshold

    return _find_consecutive_periods(is_dry, min_duration_days)


def identify_flood_periods(
    soil_moisture: np.ndarray,
    saturation: float,
    threshold_fraction: float = 0.9,
    min_duration_days: int = 2
) -> List[Tuple[int, int]]:
    """
    Identify near-saturation (flood) periods.

    Args:
        soil_moisture: Time series of soil moisture
        saturation: Saturated water content
        threshold_fraction: Fraction of saturation for flood
        min_duration_days: Minimum consecutive days

    Returns:
        List of (start_idx, end_idx) tuples
    """
    threshold = saturation * threshold_fraction
    is_wet = soil_moisture > threshold

    return _find_consecutive_periods(is_wet, min_duration_days)


def _find_consecutive_periods(
    condition: np.ndarray,
    min_duration: int
) -> List[Tuple[int, int]]:
    """Find periods where condition is True for at least min_duration"""
    periods = []
    in_period = False
    start_idx = 0

    for i, val in enumerate(condition):
        if val and not in_period:
            in_period = True
            start_idx = i
        elif not val and in_period:
            if i - start_idx >= min_duration:
                periods.append((start_idx, i - 1))
            in_period = False

    # Handle period at end
    if in_period and len(condition) - start_idx >= min_duration:
        periods.append((start_idx, len(condition) - 1))

    return periods


def compute_extreme_event_metrics(
    obs: np.ndarray,
    pred: np.ndarray,
    event_periods: List[Tuple[int, int]],
    event_type: ExtremeEventType
) -> ExtremeEventMetrics:
    """
    Compute metrics specifically for extreme event periods.

    Args:
        obs: Observed soil moisture
        pred: Predicted soil moisture
        event_periods: List of (start, end) index tuples
        event_type: Type of extreme event

    Returns:
        ExtremeEventMetrics
    """
    if not event_periods:
        return ExtremeEventMetrics(event_type=event_type)

    # Collect all values during events
    obs_event = []
    pred_event = []

    for start, end in event_periods:
        obs_event.extend(obs[start:end+1])
        pred_event.extend(pred[start:end+1])

    obs_arr = np.array(obs_event)
    pred_arr = np.array(pred_event)

    # Remove NaN
    valid = ~(np.isnan(obs_arr) | np.isnan(pred_arr))
    obs_valid = obs_arr[valid]
    pred_valid = pred_arr[valid]

    if len(obs_valid) < 3:
        return ExtremeEventMetrics(
            event_type=event_type,
            n_events=len(event_periods),
            n_days=len(obs_arr)
        )

    # Compute metrics
    rmse = np.sqrt(np.mean((pred_valid - obs_valid)**2))
    bias = np.mean(pred_valid - obs_valid)

    ss_res = np.sum((obs_valid - pred_valid)**2)
    ss_tot = np.sum((obs_valid - np.mean(obs_valid))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else np.nan

    return ExtremeEventMetrics(
        event_type=event_type,
        n_events=len(event_periods),
        n_days=len(obs_arr),
        rmse=rmse,
        bias=bias,
        r_squared=r2
    )


# =============================================================================
# TEMPORAL STRUCTURE VALIDATION
# =============================================================================

@dataclass
class TemporalStructureMetrics:
    """Metrics for temporal structure validation"""
    # Autocorrelation
    acf_obs: np.ndarray = None
    acf_pred: np.ndarray = None
    acf_rmse: float = np.nan

    # Spectral
    dominant_period_obs_days: float = np.nan
    dominant_period_pred_days: float = np.nan
    period_error_days: float = np.nan

    # Cross-correlation
    max_cross_corr: float = np.nan
    lag_at_max_days: int = 0

    # Seasonal phase
    seasonal_amplitude_ratio: float = np.nan
    seasonal_phase_shift_days: float = np.nan


def compute_temporal_structure(
    obs: np.ndarray,
    pred: np.ndarray,
    max_lag: int = 30,
    sampling_days: float = 1.0
) -> TemporalStructureMetrics:
    """
    Validate temporal structure of predictions.

    Compares autocorrelation structure and seasonal patterns.

    Args:
        obs: Observed time series
        pred: Predicted time series
        max_lag: Maximum lag for ACF (days)
        sampling_days: Days per sample

    Returns:
        TemporalStructureMetrics
    """
    # Remove NaN
    valid = ~(np.isnan(obs) | np.isnan(pred))
    obs_valid = obs[valid]
    pred_valid = pred[valid]

    if len(obs_valid) < max_lag + 10:
        return TemporalStructureMetrics()

    metrics = TemporalStructureMetrics()

    # Compute ACF
    acf_obs = np.array([compute_autocorrelation(obs_valid, lag)
                        for lag in range(max_lag)])
    acf_pred = np.array([compute_autocorrelation(pred_valid, lag)
                         for lag in range(max_lag)])

    metrics.acf_obs = acf_obs
    metrics.acf_pred = acf_pred

    # ACF RMSE (how well temporal structure is reproduced)
    valid_acf = ~(np.isnan(acf_obs) | np.isnan(acf_pred))
    if np.sum(valid_acf) > 0:
        metrics.acf_rmse = np.sqrt(
            np.mean((acf_obs[valid_acf] - acf_pred[valid_acf])**2))

    # Cross-correlation to detect phase shift
    cross_corr = np.correlate(
        (obs_valid - np.mean(obs_valid)) / np.std(obs_valid),
        (pred_valid - np.mean(pred_valid)) / np.std(pred_valid),
        mode='full'
    ) / len(obs_valid)

    center = len(cross_corr) // 2
    search_range = min(30, center)

    local_cross = cross_corr[center - search_range:center + search_range + 1]
    max_idx = np.argmax(local_cross)

    metrics.max_cross_corr = local_cross[max_idx]
    metrics.lag_at_max_days = int((max_idx - search_range) * sampling_days)

    # Spectral analysis for seasonal period
    if len(obs_valid) > 60:
        try:
            # Find dominant period
            freqs_obs, psd_obs = signal.periodogram(
                obs_valid, fs=1/sampling_days)
            freqs_pred, psd_pred = signal.periodogram(
                pred_valid, fs=1/sampling_days)

            # Ignore DC component
            psd_obs[0] = 0
            psd_pred[0] = 0

            if np.max(psd_obs) > 0:
                dom_freq_obs = freqs_obs[np.argmax(psd_obs)]
                if dom_freq_obs > 0:
                    metrics.dominant_period_obs_days = 1 / dom_freq_obs

            if np.max(psd_pred) > 0:
                dom_freq_pred = freqs_pred[np.argmax(psd_pred)]
                if dom_freq_pred > 0:
                    metrics.dominant_period_pred_days = 1 / dom_freq_pred

            if not np.isnan(metrics.dominant_period_obs_days) and \
               not np.isnan(metrics.dominant_period_pred_days):
                metrics.period_error_days = (
                    metrics.dominant_period_pred_days -
                    metrics.dominant_period_obs_days
                )
        except (ValueError, RuntimeError) as e:
            logger.warning("Spectral analysis failed: %s", e)

    return metrics


# =============================================================================
# MULTI-DEPTH CONSISTENCY VALIDATION
# =============================================================================

@dataclass
class MultiDepthConsistency:
    """
    Validate consistency across multiple soil depths.

    Physically consistent models should show:
    1. Surface responds faster to rainfall than deeper layers
    2. Deeper layers have smoother (lower frequency) variations
    3. Vertical gradients drive flux between layers
    """
    # Per-layer metrics
    layer_rmse: List[float] = field(default_factory=list)
    layer_bias: List[float] = field(default_factory=list)
    layer_r2: List[float] = field(default_factory=list)

    # Inter-layer consistency
    vertical_gradient_correlation: float = np.nan
    response_time_error_hours: List[float] = field(default_factory=list)

    # Overall score (0-1, higher is better)
    consistency_score: float = np.nan


def compute_multilayer_consistency(
    obs_layers: List[np.ndarray],
    pred_layers: List[np.ndarray],
    layer_depths_m: List[float],
    precip: Optional[np.ndarray] = None
) -> MultiDepthConsistency:
    """
    Compute multi-layer consistency metrics.

    Args:
        obs_layers: List of observed SM for each layer
        pred_layers: List of predicted SM for each layer
        layer_depths_m: Depth of each layer (m)
        precip: Precipitation time series (optional, for response time)

    Returns:
        MultiDepthConsistency metrics
    """
    n_layers = len(obs_layers)

    if n_layers != len(pred_layers):
        raise ValueError("Number of observed and predicted layers must match")

    result = MultiDepthConsistency()

    # Per-layer metrics
    for i in range(n_layers):
        obs = np.asarray(obs_layers[i])
        pred = np.asarray(pred_layers[i])

        valid = ~(np.isnan(obs) | np.isnan(pred))
        if np.sum(valid) < 10:
            result.layer_rmse.append(np.nan)
            result.layer_bias.append(np.nan)
            result.layer_r2.append(np.nan)
            continue

        obs_v = obs[valid]
        pred_v = pred[valid]

        result.layer_rmse.append(np.sqrt(np.mean((pred_v - obs_v)**2)))
        result.layer_bias.append(np.mean(pred_v - obs_v))

        ss_res = np.sum((obs_v - pred_v)**2)
        ss_tot = np.sum((obs_v - np.mean(obs_v))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else np.nan
        result.layer_r2.append(r2)

    # Vertical gradient consistency
    if n_layers >= 2:
        # Check if vertical gradients are consistent
        obs_grad = np.asarray(obs_layers[0]) - np.asarray(obs_layers[1])
        pred_grad = np.asarray(pred_layers[0]) - np.asarray(pred_layers[1])

        valid = ~(np.isnan(obs_grad) | np.isnan(pred_grad))
        if np.sum(valid) > 10:
            obs_g = obs_grad[valid]
            pred_g = pred_grad[valid]

            # Correlation of gradients
            obs_centered = obs_g - np.mean(obs_g)
            pred_centered = pred_g - np.mean(pred_g)

            num = np.sum(obs_centered * pred_centered)
            denom = np.sqrt(np.sum(obs_centered**2) * np.sum(pred_centered**2))

            if denom > 1e-10:
                result.vertical_gradient_correlation = num / denom

    # Response time to precipitation events
    if precip is not None and len(precip) == len(obs_layers[0]):
        for i in range(n_layers):
            try:
                obs_response = _estimate_response_time(precip, obs_layers[i])
                pred_response = _estimate_response_time(precip, pred_layers[i])
                result.response_time_error_hours.append(
                    (pred_response - obs_response) * 24  # Convert days to hours
                )
            except (ValueError, IndexError):
                result.response_time_error_hours.append(np.nan)

    # Overall consistency score
    valid_r2 = [r for r in result.layer_r2 if not np.isnan(r)]
    if valid_r2:
        mean_r2 = np.mean(valid_r2)
        grad_corr = result.vertical_gradient_correlation
        if np.isnan(grad_corr):
            grad_corr = 0

        result.consistency_score = 0.5 * \
            max(0, mean_r2) + 0.5 * max(0, grad_corr)

    return result


def _estimate_response_time(
    precip: np.ndarray,
    soil_moisture: np.ndarray,
    threshold_mm: float = 5.0
) -> float:
    """
    Estimate response time of soil moisture to rainfall events.

    Uses cross-correlation to find typical lag.
    """
    # Find rainfall events
    events = precip > threshold_mm
    if np.sum(events) < 3:
        return np.nan

    # Cross-correlation
    p_norm = (precip - np.nanmean(precip)) / (np.nanstd(precip) + 1e-6)
    sm_norm = (soil_moisture - np.nanmean(soil_moisture)) / \
        (np.nanstd(soil_moisture) + 1e-6)

    valid = ~(np.isnan(p_norm) | np.isnan(sm_norm))
    if np.sum(valid) < 20:
        return np.nan

    cross_corr = np.correlate(sm_norm[valid], p_norm[valid], mode='full')
    center = len(cross_corr) // 2

    # Look for peak in positive lag range (SM responds after precip)
    search_end = min(center + 15, len(cross_corr))
    local = cross_corr[center:search_end]

    if len(local) > 0:
        max_idx = np.argmax(local)
        return max_idx  # Response time in timesteps (days)

    return np.nan


# =============================================================================
# FLUX VALIDATION
# =============================================================================

@dataclass
class FluxValidationMetrics:
    """Metrics for validating water fluxes"""
    # ET validation
    et_rmse_mm_day: float = np.nan
    et_bias_mm_day: float = np.nan
    et_r2: float = np.nan

    # Runoff validation
    runoff_rmse_mm_day: float = np.nan
    runoff_bias_mm_day: float = np.nan
    runoff_r2: float = np.nan
    runoff_peak_error_percent: float = np.nan

    # Volume errors
    et_volume_error_percent: float = np.nan
    runoff_volume_error_percent: float = np.nan


def validate_et_flux(
    et_predicted: np.ndarray,
    et_observed: np.ndarray,
    quality_flags: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Validate ET predictions against observations (e.g., eddy covariance).

    Args:
        et_predicted: Predicted ET (mm/day)
        et_observed: Observed ET (mm/day)
        quality_flags: Optional quality flags (0=good, >0=suspect)

    Returns:
        Dictionary of ET validation metrics
    """
    pred = np.asarray(et_predicted)
    obs = np.asarray(et_observed)

    valid = ~(np.isnan(pred) | np.isnan(obs))
    if quality_flags is not None:
        valid &= (quality_flags == 0)

    if np.sum(valid) < 10:
        return {'et_n_valid': np.sum(valid)}

    pred_v = pred[valid]
    obs_v = obs[valid]

    rmse = np.sqrt(np.mean((pred_v - obs_v)**2))
    bias = np.mean(pred_v - obs_v)

    ss_res = np.sum((obs_v - pred_v)**2)
    ss_tot = np.sum((obs_v - np.mean(obs_v))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else np.nan

    # Volume error
    total_obs = np.sum(obs_v)
    total_pred = np.sum(pred_v)
    vol_error = 100 * (total_pred - total_obs) / \
        total_obs if total_obs > 0 else np.nan

    return {
        'et_rmse_mm_day': rmse,
        'et_bias_mm_day': bias,
        'et_r2': r2,
        'et_volume_error_percent': vol_error,
        'et_n_valid': np.sum(valid)
    }


def validate_runoff_flux(
    runoff_predicted: np.ndarray,
    runoff_observed: np.ndarray,
    catchment_area_km2: Optional[float] = None
) -> Dict[str, float]:
    """
    Validate runoff predictions against streamflow observations.

    Args:
        runoff_predicted: Predicted runoff (mm/day)
        runoff_observed: Observed runoff (mm/day or m³/s if area provided)
        catchment_area_km2: Catchment area for unit conversion

    Returns:
        Dictionary of runoff validation metrics
    """
    pred = np.asarray(runoff_predicted)
    obs = np.asarray(runoff_observed)

    # Convert m³/s to mm/day if area provided
    if catchment_area_km2 is not None:
        # m³/s to mm/day: Q × 86400 / (A × 10⁶) × 1000
        obs = obs * 86400 / (catchment_area_km2 * 1e6) * 1000

    valid = ~(np.isnan(pred) | np.isnan(obs))

    if np.sum(valid) < 10:
        return {'runoff_n_valid': np.sum(valid)}

    pred_v = pred[valid]
    obs_v = obs[valid]

    rmse = np.sqrt(np.mean((pred_v - obs_v)**2))
    bias = np.mean(pred_v - obs_v)

    ss_res = np.sum((obs_v - pred_v)**2)
    ss_tot = np.sum((obs_v - np.mean(obs_v))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else np.nan

    # Peak error
    obs_peak = np.max(obs_v)
    pred_at_obs_peak = pred_v[np.argmax(obs_v)]
    peak_error = 100 * (pred_at_obs_peak - obs_peak) / \
        obs_peak if obs_peak > 0 else np.nan

    # Volume error
    total_obs = np.sum(obs_v)
    total_pred = np.sum(pred_v)
    vol_error = 100 * (total_pred - total_obs) / \
        total_obs if total_obs > 0 else np.nan

    return {
        'runoff_rmse_mm_day': rmse,
        'runoff_bias_mm_day': bias,
        'runoff_r2': r2,
        'runoff_peak_error_percent': peak_error,
        'runoff_volume_error_percent': vol_error,
        'runoff_n_valid': np.sum(valid)
    }


# =============================================================================
# COMPREHENSIVE VALIDATION REPORT
# =============================================================================

@dataclass
class PhysicsValidationReport:
    """
    Comprehensive physics-based validation report.
    """
    # Standard metrics
    standard_metrics: ExtendedMetrics = None

    # Mass balance
    mass_balance: MassBalanceValidation = None

    # Extreme events
    drought_metrics: ExtremeEventMetrics = None
    flood_metrics: ExtremeEventMetrics = None

    # Temporal structure
    temporal_structure: TemporalStructureMetrics = None

    # Multi-depth
    multi_depth: MultiDepthConsistency = None

    # Flux validation
    flux_metrics: FluxValidationMetrics = None

    # Overall assessment
    overall_score: float = np.nan
    passes_validation: bool = False
    issues: List[str] = field(default_factory=list)

    def compute_overall_score(self):
        """Compute weighted overall validation score"""
        scores = []
        weights = []

        # Standard metrics (weight 0.3)
        if self.standard_metrics is not None:
            kge = self.standard_metrics.kge
            if not np.isnan(kge):
                scores.append(max(0, (kge + 0.41) / 1.41))  # Normalize KGE
                weights.append(0.3)

        # Mass balance (weight 0.2)
        if self.mass_balance is not None:
            if self.mass_balance.is_valid():
                scores.append(1.0)
            else:
                # Score based on how close to threshold
                error_ratio = abs(self.mass_balance.cumulative_error) / \
                    self.mass_balance.seasonal_threshold_mm
                scores.append(max(0, 1 - error_ratio))
            weights.append(0.2)

        # Extreme events (weight 0.15)
        if self.drought_metrics is not None and not np.isnan(self.drought_metrics.r_squared):
            scores.append(max(0, self.drought_metrics.r_squared))
            weights.append(0.075)
        if self.flood_metrics is not None and not np.isnan(self.flood_metrics.r_squared):
            scores.append(max(0, self.flood_metrics.r_squared))
            weights.append(0.075)

        # Temporal structure (weight 0.2)
        if self.temporal_structure is not None:
            if not np.isnan(self.temporal_structure.acf_rmse):
                acf_score = max(0, 1 - self.temporal_structure.acf_rmse)
                scores.append(acf_score)
                weights.append(0.2)

        # Multi-depth (weight 0.15)
        if self.multi_depth is not None and not np.isnan(self.multi_depth.consistency_score):
            scores.append(self.multi_depth.consistency_score)
            weights.append(0.15)

        if scores:
            self.overall_score = np.average(scores, weights=weights)

        # Determine pass/fail
        self.passes_validation = (
            self.overall_score >= 0.5 and
            (self.mass_balance is None or self.mass_balance.is_valid())
        )

        # Identify issues
        self.issues = []
        if self.standard_metrics is not None:
            if self.standard_metrics.kge < 0:
                self.issues.append("KGE < 0: Model worse than climatology")
            if abs(self.standard_metrics.bias) > 0.05:
                self.issues.append(
                    f"Large bias: {self.standard_metrics.bias:.3f}")

        if self.mass_balance is not None and not self.mass_balance.is_valid():
            self.issues.append(
                f"Mass balance violation: {self.mass_balance.cumulative_error:.1f} mm"
            )

        if self.temporal_structure is not None:
            if abs(self.temporal_structure.lag_at_max_days) > 3:
                self.issues.append(
                    f"Phase shift: {self.temporal_structure.lag_at_max_days} days"
                )

    def summary(self) -> str:
        """Generate text summary"""
        lines = [
            "=" * 60,
            "PHYSICS-BASED VALIDATION REPORT",
            "=" * 60,
            ""
        ]

        if self.standard_metrics is not None:
            lines.extend([
                "STANDARD METRICS:",
                f"  RMSE:    {self.standard_metrics.rmse:.4f}",
                f"  ubRMSE:  {self.standard_metrics.ubrmse:.4f}",
                f"  Bias:    {self.standard_metrics.bias:+.4f}",
                f"  R²:      {self.standard_metrics.r_squared:.4f}",
                f"  KGE:     {self.standard_metrics.kge:.4f}",
                f"    r:     {self.standard_metrics.kge_r:.4f}",
                f"    α:     {self.standard_metrics.kge_alpha:.4f}",
                f"    β:     {self.standard_metrics.kge_beta:.4f}",
                ""
            ])

        if self.mass_balance is not None:
            status = "✓ PASS" if self.mass_balance.is_valid() else "✗ FAIL"
            lines.extend([
                f"MASS BALANCE: {status}",
                f"  Cumulative error: {self.mass_balance.cumulative_error:.2f} mm",
                f"  Max daily error:  {self.mass_balance.max_daily_error:.3f} mm",
                ""
            ])

        if self.drought_metrics is not None and self.drought_metrics.n_events > 0:
            lines.extend([
                "DROUGHT PERIODS:",
                f"  N events: {self.drought_metrics.n_events}",
                f"  RMSE:     {self.drought_metrics.rmse:.4f}",
                f"  Bias:     {self.drought_metrics.bias:+.4f}",
                ""
            ])

        if self.temporal_structure is not None:
            lines.extend([
                "TEMPORAL STRUCTURE:",
                f"  ACF RMSE:    {self.temporal_structure.acf_rmse:.4f}",
                f"  Phase shift: {self.temporal_structure.lag_at_max_days} days",
                ""
            ])

        lines.extend([
            "-" * 60,
            f"OVERALL SCORE: {self.overall_score:.2f}",
            f"VALIDATION: {'PASS' if self.passes_validation else 'FAIL'}",
        ])

        if self.issues:
            lines.append("")
            lines.append("ISSUES:")
            for issue in self.issues:
                lines.append(f"  • {issue}")

        lines.append("=" * 60)

        return "\n".join(lines)


def run_physics_validation(
    obs: np.ndarray,
    pred: np.ndarray,
    obs_layers: Optional[List[np.ndarray]] = None,
    pred_layers: Optional[List[np.ndarray]] = None,
    layer_depths_m: Optional[List[float]] = None,
    fluxes_df: Optional[pd.DataFrame] = None,
    saturation: float = 0.45
) -> PhysicsValidationReport:
    """
    Run comprehensive physics-based validation.

    Args:
        obs: Observed soil moisture (primary layer)
        pred: Predicted soil moisture (primary layer)
        obs_layers: Optional list of observed SM per layer
        pred_layers: Optional list of predicted SM per layer
        layer_depths_m: Layer depths
        fluxes_df: DataFrame with flux columns for mass balance
        saturation: Saturated water content

    Returns:
        PhysicsValidationReport
    """
    report = PhysicsValidationReport()

    # Standard metrics
    obs_arr = np.asarray(obs)
    pred_arr = np.asarray(pred)

    valid = ~(np.isnan(obs_arr) | np.isnan(pred_arr))
    obs_v = obs_arr[valid]
    pred_v = pred_arr[valid]

    if len(obs_v) >= 10:
        kge, r, alpha, beta = compute_kge_decomposition(obs_v, pred_v)

        report.standard_metrics = ExtendedMetrics(
            rmse=np.sqrt(np.mean((pred_v - obs_v)**2)),
            ubrmse=compute_ubrmse(obs_v, pred_v),
            mae=np.mean(np.abs(pred_v - obs_v)),
            mape=compute_mape(obs_v, pred_v),
            bias=np.mean(pred_v - obs_v),
            r_squared=1 - np.sum((obs_v - pred_v)**2) /
            np.sum((obs_v - np.mean(obs_v))**2),
            nse=1 - np.sum((obs_v - pred_v)**2) /
            np.sum((obs_v - np.mean(obs_v))**2),
            kge=kge,
            kge_r=r,
            kge_alpha=alpha,
            kge_beta=beta,
            lag1_autocorr_obs=compute_autocorrelation(obs_v),
            lag1_autocorr_pred=compute_autocorrelation(pred_v),
            n_samples=len(obs_arr),
            n_valid=len(obs_v)
        )
        report.standard_metrics.autocorr_error = (
            report.standard_metrics.lag1_autocorr_pred -
            report.standard_metrics.lag1_autocorr_obs
        )

    # Mass balance
    if fluxes_df is not None:
        report.mass_balance = validate_mass_balance_series(fluxes_df)

    # Extreme events
    drought_periods = identify_drought_periods(obs_arr)
    if drought_periods:
        report.drought_metrics = compute_extreme_event_metrics(
            obs_arr, pred_arr, drought_periods, ExtremeEventType.DROUGHT
        )

    flood_periods = identify_flood_periods(obs_arr, saturation)
    if flood_periods:
        report.flood_metrics = compute_extreme_event_metrics(
            obs_arr, pred_arr, flood_periods, ExtremeEventType.FLOOD
        )

    # Temporal structure
    report.temporal_structure = compute_temporal_structure(obs_arr, pred_arr)

    # Multi-depth consistency
    if obs_layers is not None and pred_layers is not None:
        report.multi_depth = compute_multilayer_consistency(
            obs_layers, pred_layers, layer_depths_m or [0.1, 0.3, 0.6]
        )

    # Compute overall score
    report.compute_overall_score()

    return report
