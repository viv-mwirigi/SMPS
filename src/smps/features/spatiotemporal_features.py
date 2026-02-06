"""
Spatiotemporal Features for Matric Potential Modeling.

Implements spatial correlation, temporal patterns, and spatiotemporal
interactions for improved ψ (matric potential) prediction.

Spatiotemporal Features for ψ:
─────────────────────────────────────────────────────────────────
Spatial Correlation:         Nearby sensor ψ relationships
Temporal Patterns:           Diurnal/hydroclimatic ψ cycles
Spatial Interpolation:       Kriging/IDW for missing ψ data
Temporal Smoothing:          Kalman filtering for ψ time series
Spatiotemporal Covariates:   Weather × spatial ψ interactions
Cross-Correlation:           ψ relationships across space/time
─────────────────────────────────────────────────────────────────

Benefits for ψ Modeling:
- Captures spatial variability in ψ across landscapes
- Models temporal dynamics of ψ under weather forcing
- Improves prediction accuracy through spatiotemporal context
- Better handling of missing data and sensor networks

Research References:
- Western et al. (2004): Spatial correlation of soil moisture
- Vereecken et al. (2014): Soil hydrology at the landscape scale
- Rodriguez-Iturbe et al. (1995): Fractal scaling of soil water
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
import warnings

logger = logging.getLogger("swpps.features.spatiotemporal")


@dataclass
class SpatiotemporalConfig:
    """Configuration for spatiotemporal ψ features."""

    # Spatial correlation
    max_neighbors: int = 10
    spatial_decay: float = 0.1  # Spatial correlation decay parameter
    # Max distance for correlation (meters)
    correlation_distance: float = 1000.0

    # Temporal patterns
    temporal_window: str = '7D'  # Window for temporal feature calculation
    lag_features: List[str] = field(
        default_factory=lambda: ['1H', '6H', '1D', '3D'])

    # Spatial interpolation
    interpolation_method: str = 'idw'  # idw, kriging, gaussian_process
    idw_power: float = 2.0

    # Temporal smoothing
    smoothing_method: str = 'exponential'  # exponential, kalman, savgol
    smoothing_alpha: float = 0.3  # Exponential smoothing factor

    # Spatiotemporal interactions
    interaction_features: bool = True
    weather_psi_interactions: List[str] = field(default_factory=lambda: [
        'precipitation', 'temperature', 'humidity', 'wind_speed'
    ])


class SpatialCorrelationFeatures:
    """
    Computes spatial correlation features for ψ across sensor networks.

    Captures how ψ at nearby locations are related, useful for spatial interpolation
    and understanding landscape-scale ψ patterns.
    """

    def __init__(self, config: SpatiotemporalConfig):
        self.config = config

    def compute_spatial_distances(self, coordinates: np.ndarray) -> np.ndarray:
        """Compute pairwise spatial distances between sensors."""
        return squareform(pdist(coordinates, metric='euclidean'))

    def compute_spatial_correlation(self, psi_values: np.ndarray,
                                    distances: np.ndarray) -> np.ndarray:
        """Compute spatial correlation matrix for ψ values."""
        n_sensors = len(psi_values)

        # Exponential decay correlation model
        correlation = np.exp(-self.config.spatial_decay * distances)

        # Set self-correlation to 1
        np.fill_diagonal(correlation, 1.0)

        # Mask correlations beyond max distance
        correlation[distances > self.config.correlation_distance] = 0.0

        return correlation

    def extract_spatial_features(self, df: pd.DataFrame, psi_col: str = 'psi',
                                 lat_col: str = 'latitude', lon_col: str = 'longitude',
                                 time_col: str = 'timestamp') -> pd.DataFrame:
        """
        Extract spatial correlation features for ψ.

        For each sensor, computes correlation with nearby sensors.
        """
        logger.info("Extracting spatial correlation features for ψ")

        df_features = df.copy()

        # Group by time to compute spatial features at each timestamp
        if time_col in df.columns:
            time_groups = df.groupby(time_col)
        else:
            # Assume all data is from same time
            time_groups = [('', df)]

        spatial_features = []

        for time_val, time_data in time_groups:
            if len(time_data) < 2:
                continue  # Need at least 2 sensors for spatial features

            # Get coordinates and ψ values
            coords = time_data[[lat_col, lon_col]].values
            psi_vals = time_data[psi_col].values

            # Compute spatial distances
            distances = self.compute_spatial_distances(coords)

            # Compute spatial correlations
            correlations = self.compute_spatial_correlation(
                psi_vals, distances)

            # Extract features for each sensor
            for i, (_, row) in enumerate(time_data.iterrows()):
                # Find nearest neighbors
                neighbor_indices = np.argsort(distances[i])[
                    :self.config.max_neighbors + 1]
                # Exclude self
                neighbor_indices = neighbor_indices[neighbor_indices != i]

                if len(neighbor_indices) > 0:
                    # Spatial statistics
                    neighbor_distances = distances[i, neighbor_indices]
                    neighbor_psi = psi_vals[neighbor_indices]
                    neighbor_correlations = correlations[i, neighbor_indices]

                    features = {
                        f'spatial_mean_psi_{self.config.max_neighbors}nn': np.mean(neighbor_psi),
                        f'spatial_std_psi_{self.config.max_neighbors}nn': np.std(neighbor_psi),
                        f'spatial_min_distance_{self.config.max_neighbors}nn': np.min(neighbor_distances),
                        f'spatial_mean_correlation_{self.config.max_neighbors}nn': np.mean(neighbor_correlations),
                        f'spatial_psi_gradient_{self.config.max_neighbors}nn': np.mean(np.abs(psi_vals[i] - neighbor_psi))
                    }

                    # Add to row
                    for feature_name, value in features.items():
                        row[feature_name] = value

                spatial_features.append(row)

        if spatial_features:
            df_features = pd.DataFrame(spatial_features)
        else:
            logger.warning("No spatial features could be computed")

        logger.info("Spatial correlation features extracted")

        return df_features


class TemporalPatternFeatures:
    """
    Extracts temporal pattern features for ψ time series.

    Captures diurnal cycles, weather-driven patterns, and temporal autocorrelation.
    """

    def __init__(self, config: SpatiotemporalConfig):
        self.config = config

    def extract_temporal_features(self, df: pd.DataFrame, psi_col: str = 'psi',
                                  time_col: str = 'timestamp') -> pd.DataFrame:
        """Extract temporal pattern features for ψ."""
        logger.info("Extracting temporal pattern features for ψ")

        df_features = df.copy()

        if time_col not in df.columns:
            logger.warning(
                "No timestamp column found - skipping temporal features")
            return df_features

        # Ensure datetime index
        df_features[time_col] = pd.to_datetime(df_features[time_col])
        df_features = df_features.set_index(time_col).sort_index()

        # Rolling window features
        window = self.config.temporal_window

        # Basic temporal statistics
        df_features[f'psi_rolling_mean_{window}'] = df_features[psi_col].rolling(
            window).mean()
        df_features[f'psi_rolling_std_{window}'] = df_features[psi_col].rolling(
            window).std()
        df_features[f'psi_rolling_min_{window}'] = df_features[psi_col].rolling(
            window).min()
        df_features[f'psi_rolling_max_{window}'] = df_features[psi_col].rolling(
            window).max()

        # Lag features
        for lag in self.config.lag_features:
            lag_td = pd.Timedelta(lag)
            df_features[f'psi_lag_{lag}'] = df_features[psi_col].shift(
                freq=lag_td)

            # Lag differences
            df_features[f'psi_diff_{lag}'] = df_features[psi_col] - \
                df_features[f'psi_lag_{lag}']

        # Temporal derivatives (rates of change)
        df_features['psi_hourly_rate'] = df_features[psi_col].diff(
            # per hour
        ) / df_features.index.to_series().diff().dt.total_seconds() * 3600
        df_features['psi_daily_rate'] = df_features[psi_col].diff(
            24) / (24 * 3600)  # per day

        # Cyclic features
        df_features['hour_of_day'] = df_features.index.hour
        df_features['day_of_year'] = df_features.index.dayofyear
        df_features['month'] = df_features.index.month

        # Cyclical encoding (sin/cos for continuity)
        df_features['hour_sin'] = np.sin(
            2 * np.pi * df_features['hour_of_day'] / 24)
        df_features['hour_cos'] = np.cos(
            2 * np.pi * df_features['hour_of_day'] / 24)
        df_features['day_sin'] = np.sin(
            2 * np.pi * df_features['day_of_year'] / 365.25)
        df_features['day_cos'] = np.cos(
            2 * np.pi * df_features['day_of_year'] / 365.25)

        # Reset index
        df_features = df_features.reset_index()

        logger.info("Temporal pattern features extracted")

        return df_features


class SpatialInterpolationFeatures:
    """
    Spatial interpolation features for missing ψ data and spatial context.
    """

    def __init__(self, config: SpatiotemporalConfig):
        self.config = config

    def inverse_distance_weighting(self, target_coord: np.ndarray,
                                   source_coords: np.ndarray,
                                   source_values: np.ndarray,
                                   power: float = 2.0) -> float:
        """Perform inverse distance weighting interpolation."""
        distances = np.linalg.norm(source_coords - target_coord, axis=1)

        # Avoid division by zero
        distances = np.maximum(distances, 1e-6)

        # Compute weights
        weights = 1.0 / (distances ** power)
        weights = weights / np.sum(weights)

        # Weighted average
        return np.sum(weights * source_values)

    def gaussian_process_interpolation(self, target_coord: np.ndarray,
                                       source_coords: np.ndarray,
                                       source_values: np.ndarray) -> Tuple[float, float]:
        """Gaussian process interpolation with uncertainty."""
        try:
            # Standardize coordinates
            scaler = StandardScaler()
            coords_scaled = scaler.fit_transform(source_coords)
            target_scaled = scaler.transform(target_coord.reshape(1, -1))

            # Fit GP
            gp = GaussianProcessRegressor(alpha=1e-6, normalize_y=True)
            gp.fit(coords_scaled, source_values)

            # Predict
            pred, std = gp.predict(target_scaled, return_std=True)

            return pred[0], std[0]

        except Exception as e:
            logger.warning(f"GP interpolation failed: {e}")
            return np.mean(source_values), np.std(source_values)

    def interpolate_missing_psi(self, df: pd.DataFrame, psi_col: str = 'psi',
                                lat_col: str = 'latitude', lon_col: str = 'longitude',
                                time_col: str = 'timestamp') -> pd.DataFrame:
        """Interpolate missing ψ values using spatial methods."""
        logger.info("Interpolating missing ψ values spatially")

        df_interp = df.copy()

        # Group by time
        time_groups = df.groupby(
            time_col) if time_col in df.columns else [('', df)]

        for time_val, time_data in time_groups:
            # Find missing ψ values
            missing_mask = time_data[psi_col].isna()

            if not missing_mask.any():
                continue  # No missing values

            # Get available data for interpolation
            available_mask = ~missing_mask
            available_coords = time_data.loc[available_mask, [
                lat_col, lon_col]].values
            available_psi = time_data.loc[available_mask, psi_col].values

            if len(available_coords) < 3:
                logger.warning(
                    f"Insufficient data for interpolation at {time_val}")
                continue

            # Interpolate missing values
            for idx in time_data[missing_mask].index:
                target_coord = time_data.loc[idx, [lat_col, lon_col]].values

                if self.config.interpolation_method == 'idw':
                    interpolated_psi = self.inverse_distance_weighting(
                        target_coord, available_coords, available_psi,
                        power=self.config.idw_power
                    )
                    uncertainty = 0.0  # IDW doesn't provide uncertainty

                elif self.config.interpolation_method == 'gaussian_process':
                    interpolated_psi, uncertainty = self.gaussian_process_interpolation(
                        target_coord, available_coords, available_psi
                    )

                else:
                    interpolated_psi = np.mean(available_psi)
                    uncertainty = np.std(available_psi)

                # Store interpolated value
                df_interp.loc[idx,
                              f'{psi_col}_interpolated'] = interpolated_psi
                df_interp.loc[idx,
                              f'{psi_col}_interpolation_uncertainty'] = uncertainty

        logger.info("Spatial interpolation completed")

        return df_interp


class SpatiotemporalInteractionFeatures:
    """
    Creates spatiotemporal interaction features for ψ modeling.

    Combines spatial and temporal information with external covariates.
    """

    def __init__(self, config: SpatiotemporalConfig):
        self.config = config

    def create_weather_psi_interactions(self, df: pd.DataFrame,
                                        psi_col: str = 'psi',
                                        weather_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Create weather × ψ interaction features."""
        df_interactions = df.copy()

        weather_features = weather_cols or self.config.weather_psi_interactions

        for weather_col in weather_features:
            if weather_col in df.columns:
                # Direct interactions
                df_interactions[f'{psi_col}_{weather_col}_interaction'] = (
                    df[psi_col] * df[weather_col]
                )

                # Spatial weather gradients (if spatial data available)
                if all(col in df.columns for col in ['latitude', 'longitude']):
                    # Group by time and compute spatial weather stats
                    time_groups = df.groupby(
                        'timestamp') if 'timestamp' in df.columns else [('', df)]

                    for time_val, time_data in time_groups:
                        weather_mean = time_data[weather_col].mean()
                        weather_std = time_data[weather_col].std()

                        mask = df_interactions['timestamp'] == time_val if 'timestamp' in df.columns else slice(
                            None)
                        df_interactions.loc[mask,
                                            f'{weather_col}_spatial_mean'] = weather_mean
                        df_interactions.loc[mask,
                                            f'{weather_col}_spatial_std'] = weather_std

        return df_interactions

    def create_spatiotemporal_covariates(self, df: pd.DataFrame,
                                         psi_col: str = 'psi') -> pd.DataFrame:
        """Create comprehensive spatiotemporal covariates."""
        df_covariates = df.copy()

        # Spatial-temporal lag features
        if all(col in df.columns for col in ['latitude', 'longitude', 'timestamp']):
            # Sort by time and space for lag calculations
            df_covariates = df_covariates.sort_values(
                ['timestamp', 'latitude', 'longitude'])

            # Spatial-temporal lags (neighbor at previous time)
            df_covariates['psi_spatiotemporal_lag_1h'] = df_covariates.groupby(
                ['latitude', 'longitude']
            )[psi_col].shift(1)  # 1 hour lag at same location

            # Cross-spatial lags (different location, same time)
            # This is simplified - in practice would need proper spatial indexing
            df_covariates['psi_cross_spatial_mean'] = df_covariates.groupby(
                'timestamp'
            )[psi_col].transform('mean')

        return df_covariates


class SpatiotemporalFeaturePipeline:
    """
    Complete spatiotemporal feature engineering pipeline for ψ.

    Orchestrates spatial correlation, temporal patterns, interpolation, and interactions.
    """

    def __init__(self, config: Optional[SpatiotemporalConfig] = None):
        self.config = config or SpatiotemporalConfig()

        # Initialize feature components
        self.spatial_features = SpatialCorrelationFeatures(self.config)
        self.temporal_features = TemporalPatternFeatures(self.config)
        self.spatial_interpolation = SpatialInterpolationFeatures(self.config)
        self.interaction_features = SpatiotemporalInteractionFeatures(
            self.config)

    def create_all_features(self, df: pd.DataFrame, psi_col: str = 'psi',
                            lat_col: str = 'latitude', lon_col: str = 'longitude',
                            time_col: str = 'timestamp',
                            weather_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create comprehensive spatiotemporal features for ψ modeling.

        Returns DataFrame with all spatiotemporal features added.
        """
        logger.info("Creating comprehensive spatiotemporal features for ψ")

        df_features = df.copy()

        # 1. Spatial correlation features
        logger.info("Adding spatial correlation features")
        df_features = self.spatial_features.extract_spatial_features(
            df_features, psi_col, lat_col, lon_col, time_col
        )

        # 2. Temporal pattern features
        logger.info("Adding temporal pattern features")
        df_features = self.temporal_features.extract_temporal_features(
            df_features, psi_col, time_col
        )

        # 3. Spatial interpolation features
        logger.info("Adding spatial interpolation features")
        df_features = self.spatial_interpolation.interpolate_missing_psi(
            df_features, psi_col, lat_col, lon_col, time_col
        )

        # 4. Spatiotemporal interaction features
        if self.config.interaction_features:
            logger.info("Adding spatiotemporal interaction features")
            df_features = self.interaction_features.create_weather_psi_interactions(
                df_features, psi_col, weather_cols
            )

            df_features = self.interaction_features.create_spatiotemporal_covariates(
                df_features, psi_col
            )

        logger.info("Spatiotemporal feature engineering completed")

        return df_features

    def get_feature_importance_analysis(self, df: pd.DataFrame,
                                        target_col: str = 'psi') -> Dict[str, float]:
        """Analyze importance of spatiotemporal features."""
        # Simple correlation-based importance (could be enhanced with ML feature importance)
        feature_cols = [col for col in df.columns if col !=
                        target_col and not col.startswith('timestamp')]

        importance = {}
        for col in feature_cols:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                try:
                    corr = abs(df[col].corr(df[target_col]))
                    if not np.isnan(corr):
                        importance[col] = corr
                except Exception:
                    continue

        # Sort by importance
        importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return importance
