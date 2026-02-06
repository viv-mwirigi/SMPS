"""
Data Preprocessor for SMPS.

Handles temporal splits, proper preprocessing, and feature engineering
to prevent data leakage in time series.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


@dataclass
class TemporalSplitConfig:
    """Configuration for temporal data splits."""
    train_end_date: str  # End date for training data (YYYY-MM-DD)
    val_end_date: str    # End date for validation data (YYYY-MM-DD)
    test_end_date: str   # End date for test data (YYYY-MM-DD)
    date_col: str = 'date'

    # Gap between splits (days)
    train_val_gap_days: int = 0
    val_test_gap_days: int = 0

    # Minimum samples per split
    min_train_samples: int = 1000
    min_val_samples: int = 500
    min_test_samples: int = 500


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    # Scaling
    use_robust_scaling: bool = True  # Better for outliers
    scale_features: bool = True

    # Encoding
    # Use lat/lon instead of nominal site encoding
    use_coordinate_features: bool = True

    # Missing data
    max_missing_pct: float = 0.1  # Maximum missing data percentage
    # 'forward_fill', 'interpolate', 'mean'
    imputation_method: str = 'forward_fill'

    # Outlier handling
    handle_outliers: bool = True
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'none'
    outlier_threshold: float = 3.0


class DataPreprocessor:
    """
    Preprocessor for SMPS data with proper temporal splits and leakage prevention.

    Handles:
    - Temporal train/val/test splits
    - Proper feature scaling (fit on train, transform on val/test)
    - Coordinate-based site encoding
    - Missing data imputation
    - Outlier handling
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None,
                 split_config: Optional[TemporalSplitConfig] = None):
        self.config = config or PreprocessingConfig()
        self.split_config = split_config or TemporalSplitConfig(
            train_end_date='2020-12-31',
            val_end_date='2021-12-31',
            test_end_date='2022-12-31'
        )

        # Fitted transformers (fit only on training data)
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}

        # Split dataframes
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    def load_and_split_data(self, data_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data and create temporal splits.

        Args:
            data_path: Path to prepared data file

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Loading data from {data_path}")

        # Load data
        df = pd.read_csv(data_path)
        df[self.split_config.date_col] = pd.to_datetime(
            df[self.split_config.date_col])

        # Sort by date
        df = df.sort_values(self.split_config.date_col).reset_index(drop=True)

        # Create temporal splits
        train_end = pd.to_datetime(self.split_config.train_end_date)
        val_end = pd.to_datetime(self.split_config.val_end_date)
        test_end = pd.to_datetime(self.split_config.test_end_date)

        # Apply gaps if specified
        if self.split_config.train_val_gap_days > 0:
            val_start = train_end + \
                timedelta(days=self.split_config.train_val_gap_days)
        else:
            val_start = train_end + timedelta(days=1)

        if self.split_config.val_test_gap_days > 0:
            test_start = val_end + \
                timedelta(days=self.split_config.val_test_gap_days)
        else:
            test_start = val_end + timedelta(days=1)

        # Create splits
        train_mask = df[self.split_config.date_col] <= train_end
        val_mask = (df[self.split_config.date_col] >= val_start) & (
            df[self.split_config.date_col] <= val_end)
        test_mask = (df[self.split_config.date_col] >= test_start) & (
            df[self.split_config.date_col] <= test_end)

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()

        # Validate split sizes
        self._validate_splits(train_df, val_df, test_df)

        logger.info(
            f"Created splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        # Store splits
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        return train_df, val_df, test_df

    def _validate_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Validate that splits meet minimum size requirements."""
        if len(train_df) < self.split_config.min_train_samples:
            logger.warning(
                f"Train split too small: {len(train_df)} < {self.split_config.min_train_samples}")
        if len(val_df) < self.split_config.min_val_samples:
            logger.warning(
                f"Val split too small: {len(val_df)} < {self.split_config.min_val_samples}")
        if len(test_df) < self.split_config.min_test_samples:
            logger.warning(
                f"Test split too small: {len(test_df)} < {self.split_config.min_test_samples}")

    def fit_transformers(self, train_df: pd.DataFrame, feature_cols: List[str]):
        """
        Fit all transformers on training data only.

        Args:
            train_df: Training dataframe
            feature_cols: Feature column names
        """
        logger.info("Fitting transformers on training data...")

        # Handle missing data
        if self.config.imputation_method == 'forward_fill':
            # Forward fill within groups (by site)
            for col in feature_cols:
                if col in train_df.columns:
                    train_df[col] = train_df.groupby(
                        'station_id')[col].fillna(method='ffill')

        # Handle outliers
        if self.config.handle_outliers and self.config.outlier_method == 'iqr':
            for col in feature_cols:
                if col in train_df.columns and train_df[col].dtype in ['float64', 'int64']:
                    self._handle_outliers_iqr(train_df, col)

        # Fit scalers
        if self.config.scale_features:
            for col in feature_cols:
                if col in train_df.columns and train_df[col].dtype in ['float64', 'int64']:
                    scaler = RobustScaler() if self.config.use_robust_scaling else StandardScaler()
                    scaler.fit(train_df[[col]])
                    self.scalers[col] = scaler

        logger.info(f"Fitted {len(self.scalers)} scalers")

    def transform_data(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Transform data using fitted transformers.

        Args:
            df: Dataframe to transform
            feature_cols: Feature column names

        Returns:
            Transformed dataframe
        """
        df = df.copy()

        # Handle missing data (same method as training)
        if self.config.imputation_method == 'forward_fill':
            for col in feature_cols:
                if col in df.columns:
                    df[col] = df.groupby('station_id')[
                        col].fillna(method='ffill')

        # Apply scaling
        for col in feature_cols:
            if col in df.columns and col in self.scalers:
                df[col] = self.scalers[col].transform(df[[col]]).ravel()

        return df

    def _handle_outliers_iqr(self, df: pd.DataFrame, col: str):
        """Handle outliers using IQR method."""
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.config.outlier_threshold * IQR
        upper_bound = Q3 + self.config.outlier_threshold * IQR

        # Clip outliers
        df[col] = np.clip(df[col], lower_bound, upper_bound)

    def create_temporal_cross_validation_splits(self, df: pd.DataFrame,
                                                n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create temporal cross-validation splits.

        Args:
            df: Dataframe sorted by date
            n_splits: Number of CV folds

        Returns:
            List of (train_indices, val_indices) tuples
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Get date-based indices
        dates = pd.to_datetime(df[self.split_config.date_col])
        unique_dates = dates.sort_values().unique()

        splits = []
        for train_dates, val_dates in tscv.split(unique_dates):
            train_mask = dates.isin(unique_dates[train_dates])
            val_mask = dates.isin(unique_dates[val_dates])
            splits.append((train_mask.values, val_mask.values))

        return splits

    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get feature statistics from training data."""
        if self.train_df is None:
            raise ValueError("Must load data first")

        stats = {}
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in self.train_df.columns:
                series = self.train_df[col].dropna()
                stats[col] = {
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'missing_pct': self.train_df[col].isnull().mean() * 100,
                }

        return stats
