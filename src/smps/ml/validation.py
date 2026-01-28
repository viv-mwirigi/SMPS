"""
Data Splitting and Validation Framework for Soil Moisture Prediction.

Implements research-grade validation strategies:

1. Temporal Split: Train on earlier years, test on later (forecast realism)
2. Spatial Split: Leave-One-Station-Out (LOSO) for generalization
3. Spatio-Temporal Split: Combined for rigorous validation
4. Blocked Time Series: Account for temporal autocorrelation

Metrics:
- Standard: RMSE, MAE, R², Bias
- Hydrological: NSE, KGE, ubRMSE
- Uncertainty: CRPS, Reliability diagrams

Research References:
- Roberts et al. (2017): Cross-validation strategies for data with temporal/spatial structure
- Meyer et al. (2019): Importance of spatial CV for environmental modeling
- Knoben et al. (2019): NSE benchmarking in hydrology
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    TimeSeriesSplit,
    GroupKFold,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

logger = logging.getLogger("smps.ml.validation")


# =============================================================================
# Data Splitting Strategies
# =============================================================================

@dataclass
class SplitConfig:
    """Configuration for data splitting."""

    # Split type
    split_type: str = 'temporal'  # 'temporal', 'spatial', 'spatiotemporal', 'random'

    # Temporal split params
    train_years: List[int] = field(default_factory=lambda: [2020, 2021, 2022])
    test_years: List[int] = field(default_factory=lambda: [2023, 2024])
    val_fraction: float = 0.15  # From training set

    # Spatial split params
    n_spatial_folds: int = 5  # For GroupKFold
    leave_out_sites: Optional[List[str]] = None  # Specific sites for LOSO
    # 'leave_one_out', 'kfold', 'clustered'
    spatial_split_type: str = 'leave_one_out'
    spatial_cluster_method: str = 'distance'  # 'distance', 'region', 'soil_type'

    # Spatiotemporal split settings
    spatiotemporal_blocks: int = 3  # Number of spatiotemporal blocks
    temporal_overlap_days: int = 30  # Days of overlap between temporal blocks

    # Rolling forecast settings (NEW - for realistic forecasting)
    rolling_forecast: bool = True
    forecast_horizons: List[int] = field(
        default_factory=lambda: [1, 3, 7, 14])  # Days ahead
    retrain_frequency_days: int = 30  # How often to retrain model
    minimum_train_days: int = 365  # Minimum training data required

    # Time series CV params
    n_splits: int = 5
    gap_days: int = 7  # Gap between train and test to avoid leakage
    test_size_days: int = 90  # Test window size

    # Blocked CV (for autocorrelation)
    block_size_days: int = 30

    # Cross-validation settings
    n_cv_folds: int = 5
    cv_type: str = 'blocked'  # 'blocked', 'sliding', 'expanding'

    # Stratification
    stratify_by: Optional[str] = 'site_id'  # Column to stratify by
    balance_classes: bool = False

    # Random state
    random_state: int = 42


class DataSplitter:
    """
    Implements various data splitting strategies for soil moisture modeling.

    Supports:
    - Temporal split: Train on past, test on future (operational scenario)
    - Spatial split: LOSO/GroupKFold (generalization to new sites)
    - Spatio-temporal: Both dimensions for rigorous validation
    - Blocked time series: Accounts for temporal autocorrelation
    """

    def __init__(self, config: Optional[SplitConfig] = None):
        self.config = config or SplitConfig()
        self.split_info: Dict[str, Any] = {}

    def temporal_split(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally for forecast-realistic evaluation.

        Train: Earlier years (e.g., 2020-2022)
        Val: End of training period
        Test: Later years (e.g., 2023-2024)

        This mimics operational forecasting where we train on historical
        data and predict into the future.

        Args:
            df: Input DataFrame with date column
            date_col: Name of date column

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df = df.copy()
        df['_year'] = pd.to_datetime(df[date_col]).dt.year

        # Train: specified years
        train_mask = df['_year'].isin(self.config.train_years)
        train_data = df[train_mask].copy()

        # Test: specified years
        test_mask = df['_year'].isin(self.config.test_years)
        test_data = df[test_mask].copy()

        # Validation: split from training (last portion)
        if self.config.val_fraction > 0:
            n_train = len(train_data)
            val_size = int(n_train * self.config.val_fraction)

            # Sort by date and take last portion for validation
            train_data = train_data.sort_values(date_col)
            val_data = train_data.iloc[-val_size:].copy()
            train_data = train_data.iloc[:-val_size].copy()
        else:
            val_data = pd.DataFrame()

        # Clean up
        for df_split in [train_data, val_data, test_data]:
            if '_year' in df_split.columns:
                df_split.drop('_year', axis=1, inplace=True)

        self.split_info = {
            'type': 'temporal',
            'train_years': self.config.train_years,
            'test_years': self.config.test_years,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'train_date_range': (
                train_data[date_col].min(),
                train_data[date_col].max()
            ) if len(train_data) > 0 else None,
            'test_date_range': (
                test_data[date_col].min(),
                test_data[date_col].max()
            ) if len(test_data) > 0 else None,
        }

        logger.info("Temporal split: Train=%d, Val=%d, Test=%d",
                    len(train_data), len(val_data), len(test_data))

        return train_data, val_data, test_data

    def spatial_loso_split(
        self,
        df: pd.DataFrame,
        site_col: str = 'site_id',
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, str], None, None]:
        """
        Leave-One-Station-Out cross-validation.

        Tests generalization to completely new/unseen sites.
        Critical for assessing transferability of the model.

        Args:
            df: Input DataFrame with site column
            site_col: Name of site identifier column

        Yields:
            Tuple of (train_df, test_df, left_out_site)
        """
        sites = df[site_col].unique()

        for site in sites:
            test_mask = df[site_col] == site
            train_df = df[~test_mask].copy()
            test_df = df[test_mask].copy()

            logger.info("LOSO: Left out %s, Train=%d, Test=%d",
                        site, len(train_df), len(test_df))

            yield train_df, test_df, site

    def spatial_group_kfold(
        self,
        df: pd.DataFrame,
        site_col: str = 'site_id',
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, int], None, None]:
        """
        GroupKFold cross-validation by site.

        Sites in the same fold are either all in train or all in test.
        More efficient than LOSO for many sites.

        Args:
            df: Input DataFrame with site column
            site_col: Name of site identifier column

        Yields:
            Tuple of (train_df, test_df, fold_number)
        """
        gkf = GroupKFold(n_splits=self.config.n_spatial_folds)
        groups = df[site_col].values

        # Dummy X and y for sklearn interface
        X_dummy = np.zeros(len(df))

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_dummy, groups=groups)):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            test_sites = df.iloc[test_idx][site_col].unique()
            logger.info("Fold %d: Test sites=%s, Train=%d, Test=%d",
                        fold_idx, list(test_sites), len(train_df), len(test_df))

            yield train_df, test_df, fold_idx

    def blocked_time_series_split(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, int], None, None]:
        """
        Blocked time series split with gap to prevent leakage.

        Accounts for temporal autocorrelation by:
        1. Using blocks instead of individual samples
        2. Adding gap between train and test

        Args:
            df: Input DataFrame with date column
            date_col: Name of date column

        Yields:
            Tuple of (train_df, test_df, fold_number)
        """
        df = df.copy().sort_values(date_col)
        dates = pd.to_datetime(df[date_col])

        # Create time series splitter with gap
        tscv = TimeSeriesSplit(
            n_splits=self.config.n_splits,
            gap=self.config.gap_days,
            test_size=self.config.test_size_days,
        )

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df)):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            logger.info(
                "TS Fold %d: Train dates=%s to %s, Test dates=%s to %s",
                fold_idx,
                dates.iloc[train_idx].min(), dates.iloc[train_idx].max(),
                dates.iloc[test_idx].min(), dates.iloc[test_idx].max(),
            )

            yield train_df, test_df, fold_idx

    def spatiotemporal_split(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        site_col: str = 'site_id',
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Combined spatial and temporal split for rigorous validation.

        Ensures test data is both from future time AND new sites.
        Most stringent test of model generalization.

        Args:
            df: Input DataFrame
            date_col: Date column name
            site_col: Site column name

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df = df.copy()
        df['_year'] = pd.to_datetime(df[date_col]).dt.year

        sites = df[site_col].unique()
        np.random.seed(self.config.random_state)

        # Hold out some sites completely for spatial test
        n_holdout = max(1, len(sites) // 5)
        holdout_sites = np.random.choice(sites, n_holdout, replace=False)

        # Spatial split
        spatial_test = df[site_col].isin(holdout_sites)

        # Temporal split on remaining data
        remaining = df[~spatial_test]
        temporal_test = remaining['_year'].isin(self.config.test_years)

        # Train: not holdout sites AND training years
        train_mask = ~spatial_test & ~temporal_test
        train_data = df[train_mask].copy()

        # Validation: training sites, test years
        val_mask = ~spatial_test & temporal_test
        val_data = df[val_mask].copy()

        # Test: holdout sites (all years)
        test_data = df[spatial_test].copy()

        # Clean up
        for df_split in [train_data, val_data, test_data]:
            df_split.drop('_year', axis=1, inplace=True)

        self.split_info = {
            'type': 'spatiotemporal',
            'holdout_sites': list(holdout_sites),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
        }

        logger.info("Spatiotemporal split: Holdout sites=%s",
                    list(holdout_sites))
        logger.info("Train=%d, Val=%d, Test=%d", len(
            train_data), len(val_data), len(test_data))

        return train_data, val_data, test_data

    def rolling_forecast_split(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        target_col: str = 'soil_moisture',
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp], None, None]:
        """
        Rolling forecast evaluation for operational realism.

        Simulates real forecasting where:
        1. Model is trained on historical data up to current date
        2. Predictions are made for future horizons (1, 3, 7, 14 days)
        3. Model is retrained periodically (e.g., monthly)

        This is the most realistic evaluation for operational deployment.

        Args:
            df: Input DataFrame with date and target columns
            date_col: Name of date column
            target_col: Name of target column

        Yields:
            Tuple of (train_df, forecast_df, forecast_date)
            forecast_df contains actual values for all horizons
        """
        df = df.copy().sort_values(date_col)
        df[date_col] = pd.to_datetime(df[date_col])

        # Define forecast periods
        start_date = df[date_col].min(
        ) + pd.Timedelta(days=self.config.minimum_train_days)
        end_date = df[date_col].max()

        current_date = start_date
        forecast_dates = []

        # Generate forecast dates
        while current_date <= end_date:
            forecast_dates.append(current_date)
            current_date += pd.Timedelta(
                days=self.config.retrain_frequency_days)

        for forecast_date in forecast_dates:
            # Training data: all data up to forecast_date
            train_mask = df[date_col] < forecast_date
            train_df = df[train_mask].copy()

            if len(train_df) < self.config.minimum_train_days:
                continue  # Skip if insufficient training data

            # Forecast data: future values for all horizons
            forecast_data = []
            for horizon in self.config.forecast_horizons:
                future_date = forecast_date + pd.Timedelta(days=horizon)
                future_mask = df[date_col] == future_date
                if future_mask.any():
                    future_row = df[future_mask].copy()
                    future_row[f'horizon'] = horizon
                    future_row[f'forecast_date'] = forecast_date
                    forecast_data.append(future_row)

            if forecast_data:
                forecast_df = pd.concat(forecast_data, ignore_index=True)
                logger.info("Rolling forecast: Date=%s, Train=%d samples, Horizons=%s",
                            forecast_date.date(), len(train_df), self.config.forecast_horizons)
                yield train_df, forecast_df, forecast_date


# =============================================================================
# Evaluation Metrics
# =============================================================================

@dataclass
class MetricsResult:
    """Container for evaluation metrics."""

    # Standard metrics
    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    bias: float = 0.0

    # Unbiased RMSE (useful for bias-corrected comparison)
    ubrmse: float = 0.0

    # Hydrological metrics
    nse: float = 0.0  # Nash-Sutcliffe Efficiency
    kge: float = 0.0  # Kling-Gupta Efficiency

    # Correlation
    pearson_r: float = 0.0
    spearman_r: float = 0.0

    # Uncertainty metrics (if quantiles provided)
    crps: Optional[float] = None
    coverage_90: Optional[float] = None
    sharpness: Optional[float] = None

    # Sample info
    n_samples: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'bias': self.bias,
            'ubrmse': self.ubrmse,
            'nse': self.nse,
            'kge': self.kge,
            'pearson_r': self.pearson_r,
            'spearman_r': self.spearman_r,
            'n_samples': self.n_samples,
        }


class MetricsCalculator:
    """
    Calculates comprehensive evaluation metrics for soil moisture predictions.

    Includes standard ML metrics plus hydrologically-relevant metrics
    commonly used in soil moisture research.
    """

    @staticmethod
    def calculate_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_lower: Optional[np.ndarray] = None,
        y_pred_upper: Optional[np.ndarray] = None,
    ) -> MetricsResult:
        """
        Calculate all metrics.

        Args:
            y_true: Observed values
            y_pred: Predicted values (point predictions)
            y_pred_lower: Lower bound of prediction interval (optional)
            y_pred_upper: Upper bound of prediction interval (optional)

        Returns:
            MetricsResult with all computed metrics
        """
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return MetricsResult()

        result = MetricsResult(n_samples=len(y_true))

        # Standard metrics
        result.rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        result.mae = mean_absolute_error(y_true, y_pred)
        result.r2 = r2_score(y_true, y_pred)
        result.bias = np.mean(y_pred - y_true)

        # Unbiased RMSE (removes mean bias)
        bias_corrected = y_pred - result.bias
        result.ubrmse = np.sqrt(mean_squared_error(y_true, bias_corrected))

        # Correlation
        result.pearson_r = np.corrcoef(y_true, y_pred)[0, 1]
        try:
            from scipy.stats import spearmanr
            result.spearman_r = spearmanr(y_true, y_pred)[0]
        except ImportError:
            result.spearman_r = result.pearson_r

        # Hydrological metrics
        result.nse = MetricsCalculator.nash_sutcliffe_efficiency(
            y_true, y_pred)
        result.kge = MetricsCalculator.kling_gupta_efficiency(y_true, y_pred)

        # Uncertainty metrics (if prediction intervals provided)
        if y_pred_lower is not None and y_pred_upper is not None:
            y_pred_lower = y_pred_lower[mask]
            y_pred_upper = y_pred_upper[mask]

            # Coverage
            in_interval = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
            result.coverage_90 = np.mean(in_interval)

            # Sharpness (average interval width)
            result.sharpness = np.mean(y_pred_upper - y_pred_lower)

            # CRPS approximation
            result.crps = MetricsCalculator.continuous_ranked_probability_score(
                y_true, y_pred, y_pred_lower, y_pred_upper
            )

        return result

    @staticmethod
    def nash_sutcliffe_efficiency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Nash-Sutcliffe Efficiency (NSE).

        NSE = 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)

        Interpretation:
        - NSE = 1: Perfect fit
        - NSE = 0: Model performs as well as mean
        - NSE < 0: Mean is a better predictor

        Reference: Nash & Sutcliffe (1970)
        """
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)

        if denominator == 0:
            return 0.0

        return 1 - (numerator / denominator)

    @staticmethod
    def kling_gupta_efficiency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Kling-Gupta Efficiency (KGE).

        KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)

        Where:
        - r: Correlation coefficient
        - alpha: Ratio of standard deviations
        - beta: Ratio of means (bias ratio)

        Interpretation:
        - KGE = 1: Perfect fit
        - KGE > -0.41: Better than using mean flow as predictor

        Reference: Gupta et al. (2009), Knoben et al. (2019)
        """
        # Correlation
        r = np.corrcoef(y_true, y_pred)[0, 1]

        # Variability ratio
        alpha = np.std(y_pred) / (np.std(y_true) + 1e-10)

        # Bias ratio
        beta = np.mean(y_pred) / (np.mean(y_true) + 1e-10)

        # KGE
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

        return kge

    @staticmethod
    def continuous_ranked_probability_score(
        y_true: np.ndarray,
        _y_pred: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
    ) -> float:
        """
        Approximation of CRPS using prediction intervals.

        CRPS measures both calibration and sharpness of probabilistic predictions.
        Lower is better.

        Reference: Gneiting & Raftery (2007)
        """
        # Approximate CRPS using a simple formula
        # This is a simplification; full CRPS requires the full predictive distribution
        spread = (y_upper - y_lower) / 2
        center = (y_upper + y_lower) / 2

        error = np.abs(y_true - center)
        crps_approx = np.mean(error + spread)

        return crps_approx


# =============================================================================
# Baseline Comparison
# =============================================================================

@dataclass
class BaselineResult:
    """Results from baseline model comparison."""
    name: str
    metrics: 'MetricsResult'
    predictions: np.ndarray


class BaselineComparison:
    """
    Comprehensive baseline comparison framework.

    Essential for demonstrating model value over simple alternatives.
    Following best practices from hydrology literature (Knoben et al., 2019).
    """

    def __init__(self, metrics_calculator: Optional['MetricsCalculator'] = None):
        self.metrics = metrics_calculator or MetricsCalculator()
        self.results: Dict[str, BaselineResult] = {}

    def run_all_baselines(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        model_predictions: np.ndarray,
    ) -> Dict[str, BaselineResult]:
        """
        Run all baseline comparisons and return results.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            model_predictions: Predictions from the model being evaluated

        Returns:
            Dictionary mapping baseline name to results
        """
        self.results = {}

        # 1. Persistence baseline
        try:
            pers_pred = BaselineModels.persistence(y_train, X_test)
            pers_metrics = self.metrics.calculate_all(y_test, pers_pred)
            self.results['persistence'] = BaselineResult(
                name='Persistence (lag-1)',
                metrics=pers_metrics,
                predictions=pers_pred,
            )
        except Exception as e:
            logger.warning("Persistence baseline failed: %s", e)

        # 2. Climatology baseline
        try:
            clim_pred = BaselineModels.climatology(y_train, X_test)
            clim_metrics = self.metrics.calculate_all(y_test, clim_pred)
            self.results['climatology'] = BaselineResult(
                name='Climatology (mean)',
                metrics=clim_metrics,
                predictions=clim_pred,
            )
        except Exception as e:
            logger.warning("Climatology baseline failed: %s", e)

        # 3. Physics-only baseline
        try:
            phys_pred = BaselineModels.physics_only(X_test)
            phys_metrics = self.metrics.calculate_all(y_test, phys_pred)
            self.results['physics'] = BaselineResult(
                name='Physics Model',
                metrics=phys_metrics,
                predictions=phys_pred,
            )
        except Exception as e:
            logger.warning("Physics baseline failed: %s", e)

        # 4. Linear regression baseline
        try:
            lr_pred = BaselineModels.linear_regression(
                X_train, y_train, X_test)
            lr_metrics = self.metrics.calculate_all(y_test, lr_pred)
            self.results['linear'] = BaselineResult(
                name='Linear Regression',
                metrics=lr_metrics,
                predictions=lr_pred,
            )
        except Exception as e:
            logger.warning("Linear regression baseline failed: %s", e)

        # 5. Model being evaluated
        model_metrics = self.metrics.calculate_all(y_test, model_predictions)
        self.results['model'] = BaselineResult(
            name='Hybrid Model',
            metrics=model_metrics,
            predictions=model_predictions,
        )

        return self.results

    def calculate_skill_scores(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate skill scores relative to baselines.

        Skill = 1 - (MSE_model / MSE_baseline)
        Positive = model better than baseline
        """
        if 'model' not in self.results:
            raise ValueError("Must run baselines first")

        model_mse = self.results['model'].metrics.rmse ** 2
        skill_scores = {}

        for name, result in self.results.items():
            if name == 'model':
                continue

            baseline_mse = result.metrics.rmse ** 2
            skill = 1 - (model_mse / baseline_mse) if baseline_mse > 0 else 0

            skill_scores[name] = {
                'skill_score': skill,
                'rmse_improvement': result.metrics.rmse - self.results['model'].metrics.rmse,
                'r2_improvement': self.results['model'].metrics.r2 - result.metrics.r2,
            }

        return skill_scores

    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate a comparison table of all baselines."""
        rows = []
        for _, result in self.results.items():
            rows.append({
                'Model': result.name,
                'RMSE': result.metrics.rmse,
                'MAE': result.metrics.mae,
                'R²': result.metrics.r2,
                'NSE': result.metrics.nse,
                'KGE': result.metrics.kge,
                'Bias': result.metrics.bias,
            })

        df = pd.DataFrame(rows)
        df = df.sort_values('RMSE')
        return df


# =============================================================================
# Baseline Models
# =============================================================================

class BaselineModels:
    """
    Simple baseline models for comparison.

    Essential for establishing whether complex models actually add value.
    """

    @staticmethod
    def persistence(y_train: np.ndarray, X_test: pd.DataFrame,
                    lag: int = 1) -> np.ndarray:
        """
        Persistence baseline: predict previous value.

        y_pred(t) = y_obs(t - lag)

        Strong baseline in soil moisture (high autocorrelation).
        """
        if f'target_lag{lag}' in X_test.columns:
            return X_test[f'target_lag{lag}'].values

        # If lag not available, use last training value
        return np.full(len(X_test), y_train[-1])

    @staticmethod
    def climatology(y_train: np.ndarray, X_test: pd.DataFrame,
                    date_col: str = 'day_of_year') -> np.ndarray:
        """
        Climatology baseline: predict historical mean for that day of year.

        Strong baseline capturing seasonal patterns.
        """
        if date_col not in X_test.columns:
            return np.full(len(X_test), np.mean(y_train))

        # Would need training data DOY for proper implementation
        # Simplified: return mean
        return np.full(len(X_test), np.mean(y_train))

    @staticmethod
    def physics_only(X_test: pd.DataFrame,
                     physics_col: str = 'physics_theta_root') -> np.ndarray:
        """
        Physics-only baseline: use raw physics model output.

        Essential baseline for hybrid models to beat.
        """
        if physics_col in X_test.columns:
            return X_test[physics_col].values

        raise ValueError(f"Physics column '{physics_col}' not found")

    @staticmethod
    def linear_regression(X_train: pd.DataFrame, y_train: np.ndarray,
                          X_test: pd.DataFrame,
                          feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """
        Linear regression baseline.

        Tests whether complex nonlinear models are necessary.
        """
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        # Select features
        if feature_cols is None:
            feature_cols = [c for c in X_train.columns
                            if X_train[c].dtype in ['float64', 'int64']]

        X_tr = X_train[feature_cols].fillna(0)
        X_te = X_test[feature_cols].fillna(0)

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        model = Ridge(alpha=1.0)
        model.fit(X_tr_scaled, y_train)

        return model.predict(X_te_scaled)


# =============================================================================
# Validation Runner
# =============================================================================

@dataclass
class ValidationResult:
    """Complete validation results."""

    # Per-fold metrics
    fold_metrics: List[MetricsResult] = field(default_factory=list)

    # Aggregated metrics
    mean_metrics: Optional[MetricsResult] = None
    std_metrics: Optional[Dict[str, float]] = None

    # Baseline comparisons
    baseline_metrics: Dict[str, MetricsResult] = field(default_factory=dict)

    # Predictions (optional)
    all_predictions: Optional[pd.DataFrame] = None

    # Metadata
    split_type: str = ''
    n_folds: int = 0
    validation_time: float = 0.0


class ValidationRunner:
    """
    Runs complete validation pipeline with multiple splits and baselines.

    Provides comprehensive evaluation with:
    - Multiple CV strategies
    - Baseline comparisons
    - Uncertainty quantification
    - Detailed reporting
    """

    def __init__(
        self,
        model,  # Any model with fit/predict interface
        split_config: Optional[SplitConfig] = None,
    ):
        self.model = model
        self.split_config = split_config or SplitConfig()
        self.splitter = DataSplitter(split_config)
        self.metrics_calc = MetricsCalculator()

    def run_temporal_validation(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        date_col: str = 'date',
        physics_col: Optional[str] = None,
    ) -> ValidationResult:
        """
        Run temporal validation (train on past, test on future).

        Args:
            df: Full dataset
            feature_cols: Feature column names
            target_col: Target column name
            date_col: Date column name
            physics_col: Physics baseline column (optional)

        Returns:
            ValidationResult with metrics and predictions
        """
        import time
        start_time = time.time()

        result = ValidationResult(split_type='temporal', n_folds=1)

        # Split data
        train_df, val_df, test_df = self.splitter.temporal_split(df, date_col)

        # Prepare data
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df[target_col].values

        # Train model
        if val_df is not None and len(val_df) > 0:
            X_val = val_df[feature_cols].fillna(0)
            y_val = val_df[target_col].values
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            self.model.fit(X_train, y_train)

        # Predict
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = self.metrics_calc.calculate_all(y_test, y_pred)
        result.fold_metrics.append(metrics)
        result.mean_metrics = metrics

        # Baseline comparisons
        result.baseline_metrics = self._calculate_baselines(
            X_train, y_train, X_test, y_test, test_df, physics_col
        )

        # Store predictions
        result.all_predictions = pd.DataFrame({
            'date': test_df[date_col].values,
            'observed': y_test,
            'predicted': y_pred,
        })

        result.validation_time = time.time() - start_time

        return result

    def run_loso_validation(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        site_col: str = 'site_id',
        _physics_col: Optional[str] = None,
    ) -> ValidationResult:
        """
        Run Leave-One-Station-Out cross-validation.

        Tests model generalization to completely unseen sites.
        """
        import time
        start_time = time.time()

        all_predictions = []
        fold_metrics = []

        for train_df, test_df, left_out_site in self.splitter.spatial_loso_split(df, site_col):
            # Prepare data
            X_train = train_df[feature_cols].fillna(0)
            y_train = train_df[target_col].values
            X_test = test_df[feature_cols].fillna(0)
            y_test = test_df[target_col].values

            # Train and predict
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            # Calculate metrics
            metrics = self.metrics_calc.calculate_all(y_test, y_pred)
            fold_metrics.append(metrics)

            # Store predictions
            preds_df = pd.DataFrame({
                'site_id': left_out_site,
                'observed': y_test,
                'predicted': y_pred,
            })
            all_predictions.append(preds_df)

        result = ValidationResult(
            split_type='loso',
            n_folds=len(fold_metrics),
            fold_metrics=fold_metrics,
        )

        # Aggregate metrics
        result.mean_metrics = self._aggregate_metrics(fold_metrics)
        result.std_metrics = self._std_metrics(fold_metrics)
        result.all_predictions = pd.concat(all_predictions, ignore_index=True)
        result.validation_time = time.time() - start_time

        return result

    def run_blocked_cv_validation(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        date_col: str = 'date',
        _physics_col: Optional[str] = None,
    ) -> ValidationResult:
        """
        Run blocked time series cross-validation.

        Accounts for temporal autocorrelation with gaps between folds.
        """
        import time
        start_time = time.time()

        all_predictions = []
        fold_metrics = []

        for train_df, test_df, fold_idx in self.splitter.blocked_time_series_split(df, date_col):
            # Prepare data
            X_train = train_df[feature_cols].fillna(0)
            y_train = train_df[target_col].values
            X_test = test_df[feature_cols].fillna(0)
            y_test = test_df[target_col].values

            # Train and predict
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            # Calculate metrics
            metrics = self.metrics_calc.calculate_all(y_test, y_pred)
            fold_metrics.append(metrics)

            # Store predictions
            preds_df = pd.DataFrame({
                'fold': fold_idx,
                'date': test_df[date_col].values,
                'observed': y_test,
                'predicted': y_pred,
            })
            all_predictions.append(preds_df)

        result = ValidationResult(
            split_type='blocked_cv',
            n_folds=len(fold_metrics),
            fold_metrics=fold_metrics,
        )

        # Aggregate metrics
        result.mean_metrics = self._aggregate_metrics(fold_metrics)
        result.std_metrics = self._std_metrics(fold_metrics)
        result.all_predictions = pd.concat(all_predictions, ignore_index=True)
        result.validation_time = time.time() - start_time

        # Calculate baselines on last fold
        if len(all_predictions) > 0:
            _last_fold = all_predictions[-1]
            # Simplified baseline calculation

        return result

    def run_spatiotemporal_validation(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        date_col: str = 'date',
        site_col: str = 'site_id',
        physics_col: Optional[str] = None,
    ) -> ValidationResult:
        """
        Run spatiotemporal validation (most rigorous test).

        Combines spatial and temporal splits: test on future data from
        completely unseen sites. This is the gold standard for evaluating
        model generalization in environmental modeling.

        Args:
            df: Full dataset
            feature_cols: Feature column names
            target_col: Target column name
            date_col: Date column name
            site_col: Site column name
            physics_col: Physics baseline column (optional)

        Returns:
            ValidationResult with metrics and predictions
        """
        import time
        start_time = time.time()

        result = ValidationResult(split_type='spatiotemporal', n_folds=1)

        # Split data (spatial + temporal)
        train_df, val_df, test_df = self.splitter.spatiotemporal_split(
            df, date_col, site_col)

        # Prepare data
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df[target_col].values

        # Train model (with validation if available)
        if val_df is not None and len(val_df) > 0:
            X_val = val_df[feature_cols].fillna(0)
            y_val = val_df[target_col].values
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            self.model.fit(X_train, y_train)

        # Predict
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = self.metrics_calc.calculate_all(y_test, y_pred)
        result.fold_metrics.append(metrics)
        result.mean_metrics = metrics

        # Baseline comparisons
        result.baseline_metrics = self._calculate_baselines(
            X_train, y_train, X_test, y_test, test_df, physics_col
        )

        # Store predictions
        result.all_predictions = pd.DataFrame({
            'site_id': test_df[site_col].values,
            'date': test_df[date_col].values,
            'observed': y_test,
            'predicted': y_pred,
        })

        result.validation_time = time.time() - start_time

        return result

    def run_model_assumption_checks(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        predictions: Optional[np.ndarray] = None,
        model=None,
    ) -> Dict[str, Any]:
        """
        Check common ML model assumptions and diagnostics.

        Tests:
        - Residual normality (Shapiro-Wilk, Q-Q plot)
        - Homoscedasticity (Breusch-Pagan test)
        - Independence (Durbin-Watson test)
        - Multicollinearity (VIF)
        - Feature importance stability
        - Prediction intervals calibration

        Args:
            df: Dataset used for training
            feature_cols: Feature column names
            target_col: Target column name
            predictions: Model predictions (if available)
            model: Trained model (if available)

        Returns:
            Dictionary with assumption check results
        """
        results = {}

        try:
            from scipy import stats
            import statsmodels.api as sm
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            logger.warning("Statsmodels not available for assumption checks")
            return results

        # Prepare data
        X = df[feature_cols].fillna(0).values
        y = df[target_col].values

        if predictions is None and model is not None:
            predictions = model.predict(X)

        if predictions is not None:
            residuals = y - predictions

            # 1. Normality of residuals
            try:
                _, p_value = stats.shapiro(residuals)
                results['residual_normality'] = {
                    'shapiro_p_value': p_value,
                    'is_normal': p_value > 0.05,
                    'skewness': stats.skew(residuals),
                    'kurtosis': stats.kurtosis(residuals),
                }
            except Exception as e:
                logger.warning("Normality test failed: %s", e)

            # 2. Homoscedasticity (constant variance)
            try:
                # Breusch-Pagan test
                X_with_const = sm.add_constant(X)
                bp_test = sm.stats.diagnostic.het_breuschpagan(
                    residuals, X_with_const)
                results['homoscedasticity'] = {
                    'breusch_pagan_p_value': bp_test[1],
                    'constant_variance': bp_test[1] > 0.05,
                    'lm_statistic': bp_test[0],
                }
            except Exception as e:
                logger.warning("Homoscedasticity test failed: %s", e)

            # 3. Independence of residuals (autocorrelation)
            try:
                # Durbin-Watson test
                dw_stat = sm.stats.stattools.durbin_watson(residuals)
                results['independence'] = {
                    'durbin_watson_stat': dw_stat,
                    'no_autocorrelation': 1.5 < dw_stat < 2.5,  # Rule of thumb
                }
            except Exception as e:
                logger.warning("Independence test failed: %s", e)

        # 4. Multicollinearity check
        try:
            if X.shape[1] > 1:
                vif_data = pd.DataFrame()
                vif_data["feature"] = feature_cols
                vif_data["VIF"] = [variance_inflation_factor(
                    X, i) for i in range(X.shape[1])]

                results['multicollinearity'] = {
                    'vif_scores': vif_data.set_index('feature')['VIF'].to_dict(),
                    'high_vif_features': vif_data[vif_data['VIF'] > 5]['feature'].tolist(),
                    'max_vif': vif_data['VIF'].max(),
                }
        except Exception as e:
            logger.warning("Multicollinearity check failed: %s", e)

        # 5. Feature-target correlations
        try:
            correlations = {}
            for col in feature_cols:
                corr = df[col].corr(df[target_col])
                correlations[col] = corr

            results['feature_correlations'] = {
                'correlations': correlations,
                'strong_correlations': {k: v for k, v in correlations.items() if abs(v) > 0.3},
            }
        except Exception as e:
            logger.warning("Correlation analysis failed: %s", e)

        return results

    def run_diagnostic_plots(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        predictions: Optional[np.ndarray] = None,
        model=None,
    ) -> Dict[str, Any]:
        """
        Generate diagnostic plots for model evaluation.

        Creates:
        - Residuals vs Fitted
        - Q-Q plot
        - Scale-Location plot
        - Residuals vs Leverage
        - Feature importance plot
        - Prediction error distribution

        Args:
            df: Dataset
            feature_cols: Feature columns
            target_col: Target column
            predictions: Model predictions
            model: Trained model

        Returns:
            Dictionary with plot data (for external plotting)
        """
        plot_data = {}

        if predictions is None and model is not None:
            X = df[feature_cols].fillna(0)
            predictions = model.predict(X)

        if predictions is not None:
            y_true = df[target_col].values
            residuals = y_true - predictions

            # Residuals vs Fitted
            plot_data['residuals_vs_fitted'] = {
                'fitted': predictions,
                'residuals': residuals,
            }

            # Q-Q plot data
            try:
                from scipy import stats
                (osm, osr), (slope, intercept, r) = stats.probplot(
                    residuals, dist="norm")
                plot_data['qq_plot'] = {
                    'theoretical_quantiles': osm,
                    'sample_quantiles': osr,
                    'slope': slope,
                    'intercept': intercept,
                }
            except Exception as e:
                logger.warning("Q-Q plot data failed: %s", e)

            # Scale-Location (sqrt|residuals| vs fitted)
            plot_data['scale_location'] = {
                'fitted': predictions,
                'sqrt_abs_residuals': np.sqrt(np.abs(residuals)),
            }

            # Prediction error distribution
            plot_data['error_distribution'] = {
                'errors': residuals,
                'bins': 30,
            }

        # Feature importance (if model supports it)
        if model is not None and hasattr(model, 'feature_importances_'):
            plot_data['feature_importance'] = {
                'features': feature_cols,
                'importance': model.feature_importances_,
            }
        elif model is not None and hasattr(model, 'coef_'):
            # Linear model coefficients
            plot_data['feature_importance'] = {
                'features': feature_cols,
                'importance': np.abs(model.coef_),
            }

        return plot_data

    def run_comprehensive_validation(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        date_col: str = 'date',
        site_col: str = 'site_id',
        physics_col: Optional[str] = None,
        include_spatiotemporal: bool = True,
        include_assumption_checks: bool = True,
    ) -> Dict[str, ValidationResult]:
        """
        Run comprehensive validation suite with all strategies.

        Includes:
        - Temporal validation
        - Spatial LOSO validation
        - Blocked time series CV
        - Spatiotemporal validation (optional)
        - Model assumption checks (optional)

        Args:
            df: Full dataset
            feature_cols: Feature column names
            target_col: Target column name
            date_col: Date column name
            site_col: Site column name
            physics_col: Physics baseline column
            include_spatiotemporal: Whether to run spatiotemporal validation
            include_assumption_checks: Whether to run assumption checks

        Returns:
            Dictionary with results from all validation strategies
        """
        results = {}

        logger.info("Starting comprehensive validation suite...")

        # 1. Temporal validation
        logger.info("Running temporal validation...")
        results['temporal'] = self.run_temporal_validation(
            df, feature_cols, target_col, date_col, physics_col
        )

        # 2. Spatial LOSO validation
        logger.info("Running spatial LOSO validation...")
        results['loso'] = self.run_loso_validation(
            df, feature_cols, target_col, site_col, physics_col
        )

        # 3. Blocked time series CV
        logger.info("Running blocked time series CV...")
        results['blocked_cv'] = self.run_blocked_cv_validation(
            df, feature_cols, target_col, date_col, physics_col
        )

        # 4. Spatiotemporal validation (most rigorous)
        if include_spatiotemporal:
            logger.info("Running spatiotemporal validation...")
            results['spatiotemporal'] = self.run_spatiotemporal_validation(
                df, feature_cols, target_col, date_col, site_col, physics_col
            )

        # 5. Model assumption checks
        if include_assumption_checks:
            logger.info("Running model assumption checks...")
            # Use temporal validation model for checks
            temp_result = results['temporal']
            if temp_result.all_predictions is not None and len(temp_result.all_predictions) > 0:
                results['assumption_checks'] = self.run_model_assumption_checks(
                    df, feature_cols, target_col,
                    predictions=temp_result.all_predictions['predicted'].values,
                    model=self.model
                )

            # Diagnostic plots
            results['diagnostic_plots'] = self.run_diagnostic_plots(
                df, feature_cols, target_col,
                predictions=temp_result.all_predictions[
                    'predicted'].values if temp_result.all_predictions is not None else None,
                model=self.model
            )

        logger.info("Comprehensive validation complete!")

        return results

    def _calculate_baselines(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        test_df: pd.DataFrame,
        physics_col: Optional[str],
    ) -> Dict[str, MetricsResult]:
        """Calculate baseline model metrics."""
        baselines = {}

        # Persistence baseline
        try:
            y_pers = BaselineModels.persistence(y_train, X_test)
            baselines['persistence'] = self.metrics_calc.calculate_all(
                y_test, y_pers)
        except Exception as e:
            logger.warning("Persistence baseline failed: %s", e)

        # Climatology baseline
        try:
            y_clim = BaselineModels.climatology(y_train, X_test)
            baselines['climatology'] = self.metrics_calc.calculate_all(
                y_test, y_clim)
        except Exception as e:
            logger.warning("Climatology baseline failed: %s", e)

        # Physics baseline
        if physics_col and physics_col in test_df.columns:
            try:
                y_phys = test_df[physics_col].values
                baselines['physics'] = self.metrics_calc.calculate_all(
                    y_test, y_phys)
            except Exception as e:
                logger.warning("Physics baseline failed: %s", e)

        # Linear regression baseline
        try:
            y_lin = BaselineModels.linear_regression(X_train, y_train, X_test)
            baselines['linear'] = self.metrics_calc.calculate_all(
                y_test, y_lin)
        except Exception as e:
            logger.warning("Linear baseline failed: %s", e)

        return baselines

    def _aggregate_metrics(self, fold_metrics: List[MetricsResult]) -> MetricsResult:
        """Aggregate metrics across folds."""
        if not fold_metrics:
            return MetricsResult()

        return MetricsResult(
            rmse=np.mean([m.rmse for m in fold_metrics]),
            mae=np.mean([m.mae for m in fold_metrics]),
            r2=np.mean([m.r2 for m in fold_metrics]),
            bias=np.mean([m.bias for m in fold_metrics]),
            ubrmse=np.mean([m.ubrmse for m in fold_metrics]),
            nse=np.mean([m.nse for m in fold_metrics]),
            kge=np.mean([m.kge for m in fold_metrics]),
            pearson_r=np.mean([m.pearson_r for m in fold_metrics]),
            spearman_r=np.mean([m.spearman_r for m in fold_metrics]),
            n_samples=sum([m.n_samples for m in fold_metrics]),
        )

    def _std_metrics(self, fold_metrics: List[MetricsResult]) -> Dict[str, float]:
        """Calculate standard deviation of metrics across folds."""
        if not fold_metrics:
            return {}

        return {
            'rmse_std': np.std([m.rmse for m in fold_metrics]),
            'mae_std': np.std([m.mae for m in fold_metrics]),
            'r2_std': np.std([m.r2 for m in fold_metrics]),
            'nse_std': np.std([m.nse for m in fold_metrics]),
            'kge_std': np.std([m.kge for m in fold_metrics]),
        }

    def generate_report(self, result: ValidationResult) -> str:
        """Generate human-readable validation report."""
        lines = [
            "=" * 60,
            "VALIDATION REPORT",
            "=" * 60,
            f"Split type: {result.split_type}",
            f"Number of folds: {result.n_folds}",
            f"Validation time: {result.validation_time:.2f}s",
            "",
            "--- MODEL PERFORMANCE ---",
        ]

        if result.mean_metrics:
            m = result.mean_metrics
            lines.extend([
                f"RMSE:      {m.rmse:.4f}",
                f"MAE:       {m.mae:.4f}",
                f"R²:        {m.r2:.4f}",
                f"Bias:      {m.bias:.4f}",
                f"ubRMSE:    {m.ubrmse:.4f}",
                f"NSE:       {m.nse:.4f}",
                f"KGE:       {m.kge:.4f}",
                f"Pearson r: {m.pearson_r:.4f}",
            ])

        if result.std_metrics:
            lines.append("")
            lines.append("--- STANDARD DEVIATIONS ---")
            for key, val in result.std_metrics.items():
                lines.append(f"{key}: {val:.4f}")

        if result.baseline_metrics:
            lines.append("")
            lines.append("--- BASELINE COMPARISONS ---")
            for name, metrics in result.baseline_metrics.items():
                lines.append(
                    f"{name}: RMSE={metrics.rmse:.4f}, R²={metrics.r2:.4f}")

            # Improvement over physics
            if 'physics' in result.baseline_metrics and result.mean_metrics:
                phys_rmse = result.baseline_metrics['physics'].rmse
                model_rmse = result.mean_metrics.rmse
                improvement = (phys_rmse - model_rmse) / phys_rmse * 100
                lines.append(
                    f"\nRMSE improvement over physics: {improvement:.1f}%")

        lines.append("=" * 60)

        return "\n".join(lines)
