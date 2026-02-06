"""
Harmonizer for SMPS.

Orchestrates the complete data pipeline for both development (ISMN) and production (IoT + API) modes.

Development Mode (ISMN):
- Uses ISMN data for training/validation
- Coordinate-based fetching from Open-Meteo
- Satellite data from Sentinel-2/MODIS with interpolation

Production Mode (IoT + API):
- Real-time IoT sensor data
- API endpoints for weather/satellite data
- Streaming predictions
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from smps.data.site_manager import SiteManager
from smps.data.preprocessor import DataPreprocessor, TemporalSplitConfig, PreprocessingConfig
from smps.physics.model import PhysicsModel, PhysicsConfig
from smps.features.engineer import FeatureEngineer, FeatureConfig
from smps.ml.residual_model import ResidualModel, ResidualConfig
from smps.data.weather import OpenMeteoClient
from smps.data.sources.stubs import SatelliteDataClient

logger = logging.getLogger(__name__)


@dataclass
class HarmonizerConfig:
    """Configuration for the harmonizer."""
    # Mode
    mode: str = 'development'  # 'development' or 'production'

    # Data sources
    use_ismn_data: bool = True
    use_iot_data: bool = False

    # External data
    fetch_weather: bool = True
    fetch_satellite: bool = True
    satellite_interpolation: str = 'cubic_spline'  # 'cubic_spline', 'linear'

    # Caching
    cache_dir: Path = Path('data/cache')
    max_cache_age_hours: int = 24

    # Processing
    batch_size: int = 1000
    parallel_processing: bool = True


class Harmonizer:
    """
    Main orchestrator for SMPS data pipeline.

    Handles both development and production modes with unified interface.
    """

    def __init__(self, config: Optional[HarmonizerConfig] = None):
        self.config = config or HarmonizerConfig()

        # Initialize components
        self.site_manager = SiteManager()
        self.weather_client = OpenMeteoClient() if self.config.fetch_weather else None
        self.satellite_client = SatelliteDataClient(
        ) if self.config.fetch_satellite else None

        # Processing components
        self.preprocessor = DataPreprocessor()
        self.physics_model = PhysicsModel()
        self.feature_engineer = FeatureEngineer()
        self.residual_model = ResidualModel()

        # Data storage
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None

    def harmonize_data(self, input_data: pd.DataFrame,
                       target_space: str = 'theta') -> Tuple[pd.DataFrame, List[str]]:
        """
        Main harmonization pipeline.

        Args:
            input_data: Raw input data
            target_space: 'theta' or 'psi' for training targets

        Returns:
            Tuple of (processed_dataframe, feature_columns)
        """
        logger.info(f"Starting data harmonization in {self.config.mode} mode")

        # Store raw data
        self.raw_data = input_data.copy()

        # Step 1: Coordinate-based data fetching
        enriched_data = self._fetch_coordinate_data(input_data)

        # Step 2: Satellite data integration
        enriched_data = self._integrate_satellite_data(enriched_data)

        # Step 3: Physics model priors
        enriched_data = self.physics_model.generate_physics_priors(
            enriched_data, self.site_manager)

        # Step 4: Plant status features
        enriched_data = self.physics_model.generate_plant_status_features(
            enriched_data)

        # Step 5: Soil texture features
        enriched_data = self.physics_model.generate_soil_texture_features(
            enriched_data, self.site_manager)

        # Step 6: Feature engineering (7 categories)
        enriched_data, feature_cols = self.feature_engineer.create_all_features(
            enriched_data, self.site_manager
        )

        # Step 7: Sequential features (temporal)
        enriched_data = self.feature_engineer.create_sequential_features(
            enriched_data)

        # Step 8: Target conversion
        enriched_data = self.physics_model.convert_targets_for_training(
            enriched_data, target_space
        )

        # Update feature columns
        all_feature_cols = self.feature_engineer._collect_feature_columns(
            enriched_data)

        self.processed_data = enriched_data

        logger.info(
            f"Harmonization complete: {len(enriched_data)} samples, {len(all_feature_cols)} features")

        return enriched_data, all_feature_cols

    def _fetch_coordinate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch weather data using coordinates."""
        if not self.config.fetch_weather or self.weather_client is None:
            logger.info("Skipping weather data fetch")
            return df

        logger.info("Fetching coordinate-based weather data...")

        df = df.copy()
        unique_sites = df['station_id'].unique()

        # Fetch weather for each site
        for site_id in unique_sites:
            site_data = df[df['station_id'] == site_id]
            coordinates = self.site_manager.get_coordinates(site_id)

            if coordinates is None:
                logger.warning(f"No coordinates for site {site_id}")
                continue

            lat, lon = coordinates
            date_range = (site_data['date'].min(), site_data['date'].max())

            try:
                weather_df = self.weather_client.fetch_historical_weather(
                    latitude=lat,
                    longitude=lon,
                    start_date=date_range[0],
                    end_date=date_range[1]
                )

                # Merge weather data
                site_mask = df['station_id'] == site_id
                df.loc[site_mask, weather_df.columns] = weather_df.values

            except Exception as e:
                logger.error(f"Failed to fetch weather for {site_id}: {e}")

        return df

    def _integrate_satellite_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Integrate satellite data with interpolation."""
        if not self.config.fetch_satellite or self.satellite_client is None:
            logger.info("Skipping satellite data integration")
            return df

        logger.info("Integrating satellite data...")

        df = df.copy()
        unique_sites = df['station_id'].unique()

        for site_id in unique_sites:
            coordinates = self.site_manager.get_coordinates(site_id)
            if coordinates is None:
                continue

            lat, lon = coordinates
            site_data = df[df['station_id'] == site_id].copy()

            try:
                # Fetch satellite data (NDVI, etc.)
                satellite_data = self.satellite_client.fetch_satellite_data(
                    latitude=lat,
                    longitude=lon,
                    date_range=(site_data['date'].min(),
                                site_data['date'].max())
                )

                # Interpolate to hourly
                if self.config.satellite_interpolation == 'cubic_spline':
                    interpolated = self._interpolate_satellite_data(
                        satellite_data, site_data['date']
                    )
                else:
                    interpolated = satellite_data.reindex(
                        site_data['date'], method='nearest'
                    )

                # Merge interpolated data
                site_mask = df['station_id'] == site_id
                for col in interpolated.columns:
                    if col in df.columns:
                        df.loc[site_mask, col] = interpolated[col].values

            except Exception as e:
                logger.error(
                    f"Failed to integrate satellite data for {site_id}: {e}")

        return df

    def _interpolate_satellite_data(self, satellite_df: pd.DataFrame,
                                    target_dates: pd.Series) -> pd.DataFrame:
        """Interpolate satellite data to target dates using cubic splines."""
        from scipy import interpolate

        satellite_df = satellite_df.sort_values('date')
        target_dates = pd.to_datetime(target_dates).sort_values()

        interpolated_data = {}

        for col in satellite_df.columns:
            if col == 'date':
                continue

            # Remove NaN values for interpolation
            valid_data = satellite_df[col].dropna()
            if len(valid_data) < 4:  # Need at least 4 points for cubic spline
                # Fallback to linear interpolation
                interp_func = interpolate.interp1d(
                    satellite_df['date'].astype(np.int64),
                    satellite_df[col].fillna(
                        method='ffill').fillna(method='bfill'),
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
            else:
                # Cubic spline interpolation
                interp_func = interpolate.interp1d(
                    satellite_df['date'].astype(np.int64),
                    satellite_df[col].fillna(
                        method='ffill').fillna(method='bfill'),
                    kind='cubic',
                    bounds_error=False,
                    fill_value='extrapolate'
                )

            interpolated_values = interp_func(target_dates.astype(np.int64))
            interpolated_data[col] = interpolated_values

        return pd.DataFrame(interpolated_data, index=target_dates)

    def prepare_training_data(self, processed_df: pd.DataFrame,
                              feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare temporal splits for training.

        Args:
            processed_df: Processed dataframe from harmonize_data
            feature_cols: Feature column names

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Preparing training data with temporal splits...")

        # Create temporal splits
        train_df, val_df, test_df = self.preprocessor.load_and_split_data(
            processed_df.to_csv(index=False),  # Save to temp CSV for loading
        )

        # Fit transformers on training data
        self.preprocessor.fit_transformers(train_df, feature_cols)

        # Transform all splits
        train_df = self.preprocessor.transform_data(train_df, feature_cols)
        val_df = self.preprocessor.transform_data(val_df, feature_cols)
        test_df = self.preprocessor.transform_data(test_df, feature_cols)

        # Store splits
        self.train_data = train_df
        self.val_data = val_df
        self.test_data = test_df

        logger.info(
            f"Training data prepared: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        return train_df, val_df, test_df

    def train_models(self, horizons: List[int] = [24, 72, 168]) -> Dict[str, Any]:
        """
        Train residual models with comprehensive validation.

        Args:
            horizons: Forecast horizons in hours

        Returns:
            Training and validation results
        """
        if self.train_data is None or self.val_data is None:
            raise ValueError("Must prepare training data first")

        logger.info(f"Training models for horizons: {horizons}")

        # Get feature columns
        feature_cols = [col for col in self.train_data.columns
                        if col not in ['station_id', 'date', 'target', 'soil_moisture']]

        # Train with validation
        results = self.residual_model.train_with_validation(
            self.train_data, self.val_data, horizons, feature_cols
        )

        logger.info("Model training complete")
        return results

    def predict(self, input_df: pd.DataFrame, horizons: List[int] = [24, 72, 168]) -> Dict[int, np.ndarray]:
        """
        Make predictions on new data.

        Args:
            input_df: Input dataframe (must be processed through harmonize_data)
            horizons: Forecast horizons

        Returns:
            Predictions by horizon
        """
        # Ensure data is processed
        if self.processed_data is None:
            input_df, _ = self.harmonize_data(input_df)

        return self.residual_model.predict(input_df, horizons)

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary."""
        summary = {
            'mode': self.config.mode,
            'components': {
                'site_manager': 'active' if self.site_manager else 'inactive',
                'weather_client': 'active' if self.weather_client else 'inactive',
                'satellite_client': 'active' if self.satellite_client else 'inactive',
                'physics_model': 'active',
                'feature_engineer': 'active',
                'residual_model': 'active',
            },
            'data_stats': {
                'raw_samples': len(self.raw_data) if self.raw_data is not None else 0,
                'processed_samples': len(self.processed_data) if self.processed_data is not None else 0,
                'train_samples': len(self.train_data) if self.train_data is not None else 0,
                'val_samples': len(self.val_data) if self.val_data is not None else 0,
                'test_samples': len(self.test_data) if self.test_data is not None else 0,
            }
        }

        if self.residual_model.models:
            summary['models'] = self.residual_model.get_model_summary()

        return summary

    def save_pipeline(self, output_dir: Path):
        """Save the complete pipeline state."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save site metadata
        self.site_manager.save_metadata()

        # Save models
        model_dir = output_dir / "models"
        self.residual_model.save_models(model_dir)

        # Save configuration
        config_path = output_dir / "pipeline_config.json"
        import json
        with open(config_path, 'w') as f:
            json.dump({
                'harmonizer': self.config.__dict__,
                'preprocessor': self.preprocessor.config.__dict__,
                'physics': self.physics_model.config.__dict__,
                'features': self.feature_engineer.config.__dict__,
            }, f, indent=2)

        logger.info(f"Pipeline saved to {output_dir}")

    def load_pipeline(self, pipeline_dir: Path):
        """Load a saved pipeline."""
        # Load models
        model_dir = pipeline_dir / "models"
        if model_dir.exists():
            self.residual_model.load_models(model_dir)

        logger.info(f"Pipeline loaded from {pipeline_dir}")
