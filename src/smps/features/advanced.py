"""
Advanced feature engineering for soil moisture prediction.

This module provides sophisticated feature engineering including:
- Antecedent Precipitation Index (API) with multiple decay constants
- Time-since-event features (days since rain, etc.)
- Exponential memory features for soil moisture
- Future weather features (from forecasts)
- Comprehensive lagged observation features

These features are particularly important for multi-horizon forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("smps.features.advanced")


class AdvancedFeatureEngineer:
    """
    Engineers advanced features for soil moisture forecasting.

    Implements memory-based features, API calculations, and
    future weather integration for horizon-specific predictions.
    """

    def __init__(
        self,
        api_decay_factors: List[float] = [0.85, 0.90, 0.95],
        significant_rain_threshold_mm: float = 5.0,
        memory_decay_factors: List[float] = [0.9, 0.95, 0.98],
        forecast_horizons: List[int] = [1, 3, 7],
        gdd_base_temp_c: float = 10.0,
    ):
        """
        Initialize advanced feature engineer.

        Args:
            api_decay_factors: Decay factors for Antecedent Precipitation Index
            significant_rain_threshold_mm: Threshold for significant rain events
            memory_decay_factors: Decay factors for soil moisture memory features
            forecast_horizons: Horizons (days) for future weather features
            gdd_base_temp_c: Base temperature for Growing Degree Days
        """
        self.api_decay_factors = api_decay_factors
        self.significant_rain_threshold_mm = significant_rain_threshold_mm
        self.memory_decay_factors = memory_decay_factors
        self.forecast_horizons = forecast_horizons
        self.gdd_base_temp_c = gdd_base_temp_c

    def engineer_all_advanced_features(
        self,
        df: pd.DataFrame,
        station_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Engineer all advanced features for a DataFrame.

        Args:
            df: Input DataFrame with base features
            station_id: Optional station identifier for logging

        Returns:
            DataFrame with all advanced features added
        """
        result = df.copy()

        logger.debug(
            f"Engineering advanced features for {station_id or 'unknown'}")

        # 1. Basic lag features (precipitation, temperature, ET0)
        result = self.add_basic_lag_features(result)

        # 2. Antecedent Precipitation Index
        result = self.add_api_features(result)

        # 3. Time-since-event features
        result = self.add_time_since_features(result)

        # 4. Physics model lags and memory
        result = self.add_physics_lag_features(result)

        # 5. Soil moisture memory terms
        result = self.add_memory_features(result)

        # 6. Growing Degree Days
        result = self.add_gdd_features(result)

        # 7. Future weather features (for forecasting)
        result = self.add_future_weather_features(result)

        # 8. Water balance indicators
        result = self.add_water_balance_features(result)

        # 9. Lagged observation features (critical for forecasting)
        result = self.add_observation_lag_features(result)

        logger.debug(
            f"Added {len(result.columns) - len(df.columns)} advanced features")

        return result

    def add_basic_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic lag features for weather and ET variables."""
        result = df.copy()

        # Precipitation lags and rolling sums
        precip_lags = [1, 2, 3, 5, 7, 14, 21]
        if 'precipitation_mm' in result.columns:
            for lag in precip_lags:
                result[f'precip_lag{lag}'] = result['precipitation_mm'].shift(
                    lag)
                result[f'precip_sum{lag}d'] = result['precipitation_mm'].rolling(
                    lag, min_periods=1).sum()

        # Temperature lags and statistics
        temp_lags = [1, 3, 7, 14]
        if 'temperature_mean_c' in result.columns:
            for lag in temp_lags:
                result[f'temp_lag{lag}'] = result['temperature_mean_c'].shift(
                    lag)
                result[f'temp_mean{lag}d'] = result['temperature_mean_c'].rolling(
                    lag, min_periods=1).mean()

        # ET0 lags
        et0_lags = [1, 3, 7]
        if 'et0_mm' in result.columns:
            for lag in et0_lags:
                result[f'et0_lag{lag}'] = result['et0_mm'].shift(lag)
                result[f'et0_sum{lag}d'] = result['et0_mm'].rolling(
                    lag, min_periods=1).sum()

        return result

    def add_api_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Antecedent Precipitation Index features.

        API represents cumulative precipitation memory with exponential decay:
        API_t = P_t + decay_factor * API_{t-1}

        Different decay factors capture different memory timescales.
        """
        result = df.copy()

        if 'precipitation_mm' not in result.columns:
            return result

        for decay_factor in self.api_decay_factors:
            api_col = f'api_decay_{int(decay_factor*100)}'
            api_values = np.zeros(len(result))
            precip = result['precipitation_mm'].fillna(0).values

            # Calculate API recursively
            api_values[0] = precip[0]
            for i in range(1, len(result)):
                api_values[i] = precip[i] + decay_factor * api_values[i-1]

            result[api_col] = api_values

        return result

    def add_time_since_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-since-event features (days since rain, etc.)."""
        result = df.copy()

        if 'precipitation_mm' not in result.columns:
            return result

        precip = result['precipitation_mm'].fillna(0)

        # Days since significant rain (>threshold mm)
        significant_rain = precip > self.significant_rain_threshold_mm
        result['days_since_rain'] = self._calculate_days_since_event(
            significant_rain)

        # Days since any rain (>0.1mm)
        any_rain = precip > 0.1
        result['days_since_any_rain'] = self._calculate_days_since_event(
            any_rain)

        # Dry spell indicator (consecutive days with <1mm)
        if 'precipitation_mm' in result.columns:
            result['dry_days'] = (precip < 1).rolling(14, min_periods=1).sum()

        return result

    def _calculate_days_since_event(self, event_mask: pd.Series) -> pd.Series:
        """Calculate days since last True event in a boolean series."""
        # Group by consecutive False periods
        groups = (~event_mask).cumsum()
        days_since = event_mask.groupby(groups).cumcount()

        # Fill forward from last event, default to 30 if no events
        days_since = days_since.where(event_mask, np.nan)
        days_since = days_since.ffill().fillna(30)

        return days_since

    def add_physics_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for physics model outputs."""
        result = df.copy()

        physics_cols = [c for c in result.columns
                        if c.startswith('physics_') and 'lag' not in c.lower()]

        for col in physics_cols:
            # Standard lags
            for lag in [1, 3, 7]:
                result[f'{col}_lag{lag}'] = result[col].shift(lag)

            # Rolling mean
            result[f'{col}_mean7d'] = result[col].rolling(
                7, min_periods=1).mean()

            # Change over time
            result[f'{col}_change1d'] = result[col].diff(1)
            result[f'{col}_change3d'] = result[col].diff(3)

        return result

    def add_memory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add exponential memory features for soil moisture.

        Similar to API but for soil moisture - captures persistence.
        """
        result = df.copy()

        sm_cols = [c for c in result.columns
                   if c.startswith(('physics_', 'soil_moisture'))
                   and 'memory' not in c and 'lag' not in c]

        for col in sm_cols:
            if col not in result.columns:
                continue

            values = result[col].values

            for decay in self.memory_decay_factors:
                memory_col = f'{col}_memory_{int(decay*100)}'
                memory_values = np.zeros(len(result))

                # Calculate memory recursively
                if not np.isnan(values[0]):
                    memory_values[0] = values[0]

                for i in range(1, len(result)):
                    if not np.isnan(values[i]):
                        memory_values[i] = values[i] + \
                            decay * memory_values[i-1]
                    else:
                        memory_values[i] = memory_values[i-1]

                result[memory_col] = memory_values

        return result

    def add_gdd_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Growing Degree Days features.

        GDD proxy for root growth and vegetation development.
        """
        result = df.copy()

        if 'temperature_mean_c' not in result.columns:
            return result

        # Daily GDD (temperatures above base)
        result['gdd_daily'] = (
            result['temperature_mean_c'] - self.gdd_base_temp_c).clip(lower=0)

        # Cumulative GDD - reset each year if 'date' available
        if 'date' in result.columns:
            result['year'] = pd.to_datetime(result['date']).dt.year
            result['cumulative_gdd'] = result.groupby(
                'year')['gdd_daily'].cumsum()
        else:
            result['cumulative_gdd'] = result['gdd_daily'].cumsum()

        return result

    def add_future_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add future weather features for forecasting.

        These represent weather forecasts - in training we use actual values
        shifted backward (perfect forecast), in deployment use forecast API.
        """
        result = df.copy()

        for horizon in self.forecast_horizons:
            # Future precipitation (cumulative over horizon window)
            if 'precipitation_mm' in result.columns:
                result[f'precip_future_{horizon}d'] = (
                    result['precipitation_mm']
                    .shift(-horizon)
                    .rolling(horizon, min_periods=1)
                    .sum()
                )
                result[f'precip_future{horizon}'] = result['precipitation_mm'].shift(
                    -horizon)

            # Future ET0
            if 'et0_mm' in result.columns:
                result[f'et0_future_{horizon}d'] = (
                    result['et0_mm']
                    .shift(-horizon)
                    .rolling(horizon, min_periods=1)
                    .sum()
                )
                result[f'et0_future{horizon}'] = result['et0_mm'].shift(
                    -horizon)

            # Future temperature
            if 'temperature_mean_c' in result.columns:
                result[f'temp_future{horizon}'] = result['temperature_mean_c'].shift(
                    -horizon)
                result[f'temp_future_mean_{horizon}d'] = (
                    result['temperature_mean_c']
                    .shift(-horizon)
                    .rolling(horizon, min_periods=1)
                    .mean()
                )

        return result

    def add_water_balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add water balance indicator features (P - ET)."""
        result = df.copy()

        if 'precipitation_mm' in result.columns and 'et0_mm' in result.columns:
            # Daily water balance
            result['water_balance_1d'] = result['precipitation_mm'] - \
                result['et0_mm']

            # Multi-day water balance
            if 'precip_sum7d' in result.columns and 'et0_sum7d' in result.columns:
                result['water_balance_7d'] = result['precip_sum7d'] - \
                    result['et0_sum7d']

            # Future water balance (for forecasting)
            for horizon in self.forecast_horizons:
                precip_col = f'precip_future_{horizon}d'
                et0_col = f'et0_future_{horizon}d'
                if precip_col in result.columns and et0_col in result.columns:
                    result[f'water_balance_future_{horizon}d'] = (
                        result[precip_col] - result[et0_col]
                    )

        return result

    def add_observation_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lagged observation features.

        Critical for forecasting - soil moisture has very high autocorrelation
        (~0.98 at 1-day lag), so lagged observations are highly predictive.
        """
        result = df.copy()

        # Find observation columns (raw, not already lagged)
        obs_cols = [
            c for c in result.columns
            if c.startswith('obs_sm_')
            and '_lag' not in c
            and '_mean' not in c
            and '_std' not in c
            and '_change' not in c
            and '_memory' not in c
        ]

        for obs_col in obs_cols:
            # Lags matching and beyond forecast horizons
            for lag in [1, 3, 7, 14]:
                result[f'{obs_col}_lag{lag}'] = result[obs_col].shift(lag)

            # Rolling statistics
            result[f'{obs_col}_mean7d'] = result[obs_col].rolling(
                7, min_periods=1).mean()
            result[f'{obs_col}_std7d'] = result[obs_col].rolling(
                7, min_periods=1).std()

            # Recent changes
            result[f'{obs_col}_change1d'] = result[obs_col].diff(1)
            result[f'{obs_col}_change7d'] = result[obs_col].diff(7)

        # Also handle soil_moisture column if present
        if 'soil_moisture' in result.columns:
            for lag in [1, 3, 7, 14]:
                result[f'sm_lag_{lag}d'] = result['soil_moisture'].shift(lag)

            result['sm_mean7d'] = result['soil_moisture'].rolling(
                7, min_periods=1).mean()
            result['sm_std7d'] = result['soil_moisture'].rolling(
                7, min_periods=1).std()
            result['sm_change1d'] = result['soil_moisture'].diff(1)
            result['sm_change7d'] = result['soil_moisture'].diff(7)

        return result


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal/seasonal features.

    Standalone function for quick temporal feature addition.
    """
    result = df.copy()

    if 'date' not in result.columns:
        return result

    dates = pd.to_datetime(result['date'])

    result['day_of_year'] = dates.dt.dayofyear
    result['month'] = dates.dt.month
    result['week'] = dates.dt.isocalendar().week

    # Cyclic encoding
    result['sin_doy'] = np.sin(2 * np.pi * result['day_of_year'] / 365)
    result['cos_doy'] = np.cos(2 * np.pi * result['day_of_year'] / 365)
    result['sin_month'] = np.sin(2 * np.pi * result['month'] / 12)
    result['cos_month'] = np.cos(2 * np.pi * result['month'] / 12)

    return result


def create_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create spatial features from location data.

    Standalone function for quick spatial feature addition.
    """
    result = df.copy()

    if 'latitude' in result.columns:
        result['lat_normalized'] = (
            (result['latitude'] - result['latitude'].mean()) /
            result['latitude'].std()
        )

    if 'longitude' in result.columns:
        result['lon_normalized'] = (
            (result['longitude'] - result['longitude'].mean()) /
            result['longitude'].std()
        )

    if 'depth_cm' in result.columns:
        result['depth_normalized'] = result['depth_cm'] / \
            result['depth_cm'].max()
        result['is_surface'] = (result['depth_cm'] <= 15).astype(int)
        result['is_deep'] = (result['depth_cm'] > 50).astype(int)

    return result
