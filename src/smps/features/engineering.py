"""Feature engineering for soil moisture prediction.

This module provides the FeatureEngineer class that computes derived
features from raw weather, soil, and physics data for ML models.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from smps.core.types import SiteID


class FeatureEngineer:
    """
    Engineers features from base canonical data for ML models.

    Computes lag features, rolling statistics, cumulative indices,
    and interaction terms useful for soil moisture prediction.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize feature engineer.

        Args:
            config: Optional configuration dict with feature settings.
                   Defaults provide reasonable feature engineering.
        """
        self.config = config or {}
        self.logger = logging.getLogger("smps.features.engineering")

        # Feature configuration with defaults
        self.lag_days = self.config.get("lag_days", [1, 3, 7, 14])
        self.rolling_windows = self.config.get(
            "rolling_windows", [3, 7, 14, 30])
        self.include_interactions = self.config.get(
            "include_interactions", True)

    def engineer_features(self, df: pd.DataFrame, site_id: SiteID) -> pd.DataFrame:
        """
        Engineer features from base canonical table.

        Args:
            df: Base DataFrame with columns like precipitation_mm, et0_mm,
                physics_theta_surface, physics_theta_root, etc.
            site_id: Site identifier for logging.

        Returns:
            DataFrame with additional engineered features.
        """
        self.logger.debug("Engineering features for site %s", site_id)

        result = df.copy()

        # Ensure date sorting
        if 'date' in result.columns:
            result = result.sort_values('date')

        # Engineer weather lag features
        result = self._add_lag_features(result)

        # Engineer rolling statistics
        result = self._add_rolling_features(result)

        # Engineer cumulative water balance index
        result = self._add_cumulative_features(result)

        # Engineer interaction features
        if self.include_interactions:
            result = self._add_interaction_features(result)

        # Engineer temporal features
        result = self._add_temporal_features(result)

        self.logger.debug(
            "Engineered %d features for %d rows",
            len(result.columns) - len(df.columns), len(result)
        )

        return result

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged versions of key variables."""
        result = df.copy()

        lag_columns = ['precipitation_mm', 'et0_mm', 'physics_theta_root']

        for col in lag_columns:
            if col in result.columns:
                for lag in self.lag_days:
                    result[f'{col}_lag{lag}'] = result[col].shift(lag)

        return result

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics for weather variables."""
        result = df.copy()

        # Rolling precipitation sums
        if 'precipitation_mm' in result.columns:
            for window in self.rolling_windows:
                result[f'precip_sum_{window}d'] = (
                    result['precipitation_mm'].rolling(
                        window, min_periods=1).sum()
                )
                result[f'precip_max_{window}d'] = (
                    result['precipitation_mm'].rolling(
                        window, min_periods=1).max()
                )

        # Rolling ET0 means
        if 'et0_mm' in result.columns:
            for window in self.rolling_windows:
                result[f'et0_mean_{window}d'] = (
                    result['et0_mm'].rolling(window, min_periods=1).mean()
                )

        # Rolling soil moisture statistics
        if 'physics_theta_root' in result.columns:
            for window in [7, 14]:
                result[f'theta_root_std_{window}d'] = (
                    result['physics_theta_root'].rolling(
                        window, min_periods=1).std()
                )

        return result

    def _add_cumulative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cumulative water balance features."""
        result = df.copy()

        # Cumulative water balance (P - ET0)
        if 'precipitation_mm' in result.columns and 'et0_mm' in result.columns:
            result['daily_water_balance'] = result['precipitation_mm'] - \
                result['et0_mm']
            result['cumulative_water_balance'] = result['daily_water_balance'].cumsum()

            # Rolling cumulative indices
            for window in [7, 14, 30]:
                result[f'water_balance_{window}d'] = (
                    result['daily_water_balance'].rolling(
                        window, min_periods=1).sum()
                )

        # Aridity index proxy (ET0/P) over rolling windows
        if 'precipitation_mm' in result.columns and 'et0_mm' in result.columns:
            for window in [7, 30]:
                precip_sum = result['precipitation_mm'].rolling(
                    window, min_periods=1).sum()
                et0_sum = result['et0_mm'].rolling(window, min_periods=1).sum()
                # Avoid division by zero
                result[f'aridity_index_{window}d'] = et0_sum / \
                    (precip_sum + 0.1)

        return result

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between variables."""
        result = df.copy()

        # Precipitation efficiency (considering recent dryness)
        if 'precipitation_mm' in result.columns and 'physics_theta_root' in result.columns:
            # More impact when soil is drier
            result['precip_efficiency'] = (
                result['precipitation_mm'] *
                (1.0 - result['physics_theta_root'])
            )

        # ET demand relative to available water
        if 'et0_mm' in result.columns and 'physics_theta_root' in result.columns:
            result['et_stress_ratio'] = result['et0_mm'] / \
                (result['physics_theta_root'] + 0.01)

        return result

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal/seasonal features."""
        result = df.copy()

        if 'date' in result.columns:
            dates = pd.to_datetime(result['date'])

            # Day of year (cyclic encoding)
            day_of_year = dates.dt.dayofyear
            result['day_sin'] = np.sin(2 * np.pi * day_of_year / 365)
            result['day_cos'] = np.cos(2 * np.pi * day_of_year / 365)

            # Month (for seasonal effects)
            result['month'] = dates.dt.month

            # Days since start (trend feature)
            result['days_since_start'] = (dates - dates.min()).dt.days

        return result
