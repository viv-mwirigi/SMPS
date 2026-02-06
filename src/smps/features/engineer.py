"""
Feature Engineer for SMPS.

Creates the 7 categories of features for soil moisture prediction:

1. Direct Priors: ψ_phys (Mechanistic output)
2. Fluxes: ET_actual and Drainage
3. Plant Status: K_c (derived from NDVI)
4. Soil Texture: Target-encoded categorical data
5. Weather Dynamics: Sequential weather patterns
6. Spatial Features: Coordinate-based encodings
7. Temporal Features: Time-based patterns and lags
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Sequential features
    lag_days: List[int] = None
    rolling_windows: List[int] = None

    # Weather features
    weather_features: List[str] = None

    # Target features (for sequential modeling)
    target_features: List[str] = None

    # Spatial features
    include_coordinate_features: bool = True
    include_spatial_patterns: bool = True

    # Temporal features
    include_seasonal_features: bool = True
    include_trend_features: bool = True

    def __post_init__(self):
        if self.lag_days is None:
            self.lag_days = [1, 2, 3, 7, 14]
        if self.rolling_windows is None:
            self.rolling_windows = [3, 7, 14]
        if self.weather_features is None:
            self.weather_features = [
                'precipitation_mm', 'et0_mm', 'temperature_2m',
                'relative_humidity_2m', 'wind_speed_10m'
            ]
        if self.target_features is None:
            self.target_features = [
                'soil_moisture', 'psi_phys_surface', 'et_actual', 'drainage'
            ]


class FeatureEngineer:
    """
    Feature engineer for the 7 categories of SMPS features.

    Ensures no data leakage by using proper temporal feature engineering.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def create_all_features(self, df: pd.DataFrame,
                            site_manager: Optional[Any] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create all 7 categories of features.

        Args:
            df: Input dataframe with raw data
            site_manager: Site manager for coordinate features

        Returns:
            Tuple of (enriched_dataframe, feature_column_names)
        """
        logger.info("Creating all feature categories...")

        df = df.copy()

        # 1. Direct Priors (already created by PhysicsModel)
        # ψ_phys_surface, ψ_phys_root, ψ_phys_deep

        # 2. Fluxes (already created by PhysicsModel)
        # et_actual, drainage

        # 3. Plant Status
        df = self._create_plant_status_features(df)

        # 4. Soil Texture
        df = self._create_soil_texture_features(df, site_manager)

        # 5. Weather Dynamics
        df = self._create_weather_dynamic_features(df)

        # 6. Spatial Features
        df = self._create_spatial_features(df, site_manager)

        # 7. Temporal Features
        df = self._create_temporal_features(df)

        # Collect all feature columns
        feature_cols = self._collect_feature_columns(df)

        logger.info(
            f"Created {len(feature_cols)} features across 7 categories")

        return df, feature_cols

    def _create_plant_status_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create plant status features from K_c and NDVI."""
        df = df.copy()

        # K_c is already created by PhysicsModel
        # Add derived features
        if 'k_c' in df.columns:
            # Plant stress indicators
            df['plant_stress_index'] = 1.0 - \
                (df['k_c'] - 0.3) / 0.9  # 0-1, higher = more stressed
            df['canopy_cover_fraction'] = (
                df['k_c'] - 0.3) / 0.9  # 0-1 canopy cover

            # Seasonal plant development
            df['plant_growth_rate'] = df.groupby(
                'station_id')['k_c'].diff(7).fillna(0)

        return df

    def _create_soil_texture_features(self, df: pd.DataFrame,
                                      site_manager: Optional[Any]) -> pd.DataFrame:
        """Create soil texture features."""
        df = df.copy()

        # Basic texture features already created by PhysicsModel
        # Add derived hydraulic properties
        if 'sand_percent' in df.columns and 'clay_percent' in df.columns:
            # Hydraulic conductivity proxy (simplified)
            df['k_saturated_proxy'] = np.exp(
                -0.038 * df['clay_percent'] + 0.022 * df['sand_percent']
            ) / 100.0  # cm/day to m/day

            # Porosity proxy
            df['porosity_proxy'] = 0.489 - 0.00126 * df['sand_percent']

            # Texture-based water retention
            df['texture_wetting_front'] = (
                0.5 + 0.3 * (df['clay_percent'] / 100.0) -
                0.2 * (df['sand_percent'] / 100.0)
            )

        return df

    def _create_weather_dynamic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dynamic weather features with proper temporal handling."""
        df = df.copy()

        # Sort by site and date
        df = df.sort_values(['station_id', 'date'])

        # Weather accumulation features
        for col in self.config.weather_features:
            if col in df.columns:
                # Cumulative over different periods
                for window in [1, 3, 7, 14, 30]:
                    df[f'{col}_cum_{window}d'] = (
                        df.groupby('station_id')[col]
                        .rolling(window=window, min_periods=1)
                        .sum().reset_index(0, drop=True)
                    )

                # Rate of change
                df[f'{col}_rate_1d'] = df.groupby(
                    'station_id')[col].diff(1).fillna(0)
                df[f'{col}_rate_3d'] = df.groupby(
                    'station_id')[col].diff(3).fillna(0)

                # Weather stability (coefficient of variation)
                for window in [7, 14]:
                    mean_col = f'{col}_mean_{window}d'
                    std_col = f'{col}_std_{window}d'
                    cv_col = f'{col}_cv_{window}d'

                    df[mean_col] = (
                        df.groupby('station_id')[col]
                        .rolling(window=window, min_periods=1)
                        .mean().reset_index(0, drop=True)
                    )
                    df[std_col] = (
                        df.groupby('station_id')[col]
                        .rolling(window=window, min_periods=1)
                        .std().reset_index(0, drop=True)
                    )
                    df[cv_col] = df[std_col] / (df[mean_col] + 1e-6)

        return df

    def _create_spatial_features(self, df: pd.DataFrame,
                                 site_manager: Optional[Any]) -> pd.DataFrame:
        """Create spatial features using coordinates."""
        df = df.copy()

        if not self.config.include_coordinate_features:
            return df

        # Coordinate features already created by SiteManager
        # Add spatial pattern features
        if self.config.include_spatial_patterns and 'latitude' in df.columns:
            # Distance-based features
            ref_lat, ref_lon = df['latitude'].median(
            ), df['longitude'].median()

            df['distance_from_center'] = np.sqrt(
                (df['latitude'] - ref_lat)**2 + (df['longitude'] - ref_lon)**2
            )

            # Climate zone proxies based on coordinates
            df['latitude_zone'] = pd.cut(df['latitude'],
                                         bins=[-90, 0, 20, 40, 60, 90],
                                         labels=['southern', 'tropical', 'subtropical', 'temperate', 'boreal'])

            # One-hot encode latitude zones
            zone_dummies = pd.get_dummies(
                df['latitude_zone'], prefix='lat_zone')
            df = pd.concat([df, zone_dummies], axis=1)

        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features."""
        df = df.copy()

        # Date features
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['week_of_year'] = df['date'].dt.isocalendar().week

        if self.config.include_seasonal_features:
            # Seasonal features
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        if self.config.include_trend_features:
            # Target-based trends (no leakage - these are calculated properly)
            for target in self.config.target_features:
                if target in df.columns:
                    # Short-term trends
                    for lag in [1, 3, 7]:
                        df[f'{target}_trend_{lag}d'] = (
                            df.groupby('station_id')[
                                target].diff(lag).fillna(0)
                        )

                    # Acceleration (change in trend)
                    df[f'{target}_accel_3d'] = (
                        df.groupby('station_id')[
                            f'{target}_trend_1d'].diff(3).fillna(0)
                    )

        return df

    def create_sequential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sequential features (lags, rolling stats) for target variables.

        This is done separately to ensure proper temporal ordering.
        """
        df = df.copy()

        # Sort by site and date
        df = df.sort_values(['station_id', 'date'])

        # Create lags
        for col in self.config.target_features:
            if col in df.columns:
                for lag in self.config.lag_days:
                    df[f'{col}_lag_{lag}d'] = (
                        df.groupby('station_id')[col].shift(lag)
                    )

                # Rolling statistics
                for window in self.config.rolling_windows:
                    df[f'{col}_roll_{window}d_mean'] = (
                        df.groupby('station_id')[col]
                        .rolling(window=window, min_periods=1)
                        .mean().reset_index(0, drop=True)
                    )
                    df[f'{col}_roll_{window}d_std'] = (
                        df.groupby('station_id')[col]
                        .rolling(window=window, min_periods=1)
                        .std().reset_index(0, drop=True)
                    )
                    df[f'{col}_roll_{window}d_min'] = (
                        df.groupby('station_id')[col]
                        .rolling(window=window, min_periods=1)
                        .min().reset_index(0, drop=True)
                    )
                    df[f'{col}_roll_{window}d_max'] = (
                        df.groupby('station_id')[col]
                        .rolling(window=window, min_periods=1)
                        .max().reset_index(0, drop=True)
                    )

        return df

    def _collect_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Collect all feature column names."""
        # Exclude target and metadata columns
        exclude_cols = {
            'station_id', 'date', 'soil_moisture', 'target', 'date_plus_24h',
            'date_plus_72h', 'date_plus_168h', 'target_24h', 'target_72h', 'target_168h',
            'latitude_zone'  # Intermediate column
        }

        # Include all numeric columns that aren't excluded
        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols and df[col].dtype in ['float64', 'int64', 'bool']:
                # Check if column has enough non-null values
                non_null_pct = df[col].notna().mean()
                if non_null_pct > 0.5:  # At least 50% non-null
                    feature_cols.append(col)

        return sorted(feature_cols)
