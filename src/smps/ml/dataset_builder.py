"""
Canonical Dataset Builder for Soil Moisture ML Pipeline.

This module orchestrates the construction of a unified modeling dataset by:
1. Pulling raw data from SpaceIoTBox API (weather, satellite, agro)
2. Running physics model to generate priors at multiple depths
3. Fetching historical soil moisture observations
4. Adding static site attributes (soil texture, topography, land cover)
5. Engineering temporal and derived features
6. Cleaning, aligning, and standardizing all data

The output is a structured DataFrame ready for ML training.

Research References:
- Reichstein et al. (2019): Deep learning for Earth system science
- Fang et al. (2017): Prolongation of SMAP to LSTM
- Pan et al. (2019): Physics-informed machine learning for hydrology
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from smps.core.config import get_config, DataConfig
from smps.core.exceptions import DataSourceError, DataValidationError
from smps.core.types import SiteID
from smps.data.contracts import (
    DailyWeather,
    SoilProfile,
    RemoteSensingData,
    SoilMoistureObservation,
)
from smps.data.sources.base import DataFetchRequest, DataFetchResult
from smps.physics import create_water_balance_model
from smps.physics.pedotransfer import estimate_soil_parameters_tropical

logger = logging.getLogger("smps.ml.dataset_builder")


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # Lag features for temporal patterns
    lag_days: List[int] = field(default_factory=lambda: [
                                1, 2, 3, 5, 7, 14, 21, 30])

    # Rolling window sizes for aggregations
    rolling_windows: List[int] = field(
        default_factory=lambda: [3, 7, 14, 21, 30, 60])

    # Cumulative windows for water balance
    cumulative_windows: List[int] = field(
        default_factory=lambda: [3, 7, 14, 30, 60, 90])

    # Physics feature options
    include_physics_fluxes: bool = True
    include_physics_states: bool = True
    include_physics_residuals: bool = True
    physics_depths: List[str] = field(
        default_factory=lambda: ["surface", "root", "deep"])

    # Remote sensing features
    include_ndvi_features: bool = True
    include_sar_features: bool = True
    ndvi_lag_days: List[int] = field(default_factory=lambda: [0, 7, 14, 30])

    # Interaction features
    include_interactions: bool = True

    # Temporal encoding
    cyclic_encoding: bool = True
    include_season: bool = True
    include_growing_season: bool = True


@dataclass
class DatasetConfig:
    """Configuration for dataset building."""

    # Data sources
    use_spaceiotbox: bool = True
    use_open_meteo_fallback: bool = True

    # Physics model
    run_physics_model: bool = True
    physics_warmup_days: int = 30
    physics_crop_type: str = "maize"

    # Quality control
    min_data_coverage: float = 0.8
    max_gap_days: int = 7
    outlier_sigma: float = 3.0

    # Target configuration
    target_depths_cm: List[int] = field(default_factory=lambda: [10, 30, 50])
    target_variable: str = "soil_moisture_vwc"

    # Cache settings
    cache_dir: Optional[Path] = None
    cache_ttl_hours: int = 24


@dataclass
class SiteMetadata:
    """Static metadata for a site."""
    site_id: SiteID
    latitude: float
    longitude: float
    elevation_m: float = 0.0
    land_cover: str = "cropland"
    crop_type: Optional[str] = None
    irrigation_type: Optional[str] = None
    soil_texture_class: Optional[str] = None
    climate_zone: Optional[str] = None

    # Derived attributes
    slope_degrees: float = 0.0
    aspect_degrees: float = 0.0
    twi: float = 10.0  # Topographic Wetness Index


class CanonicalDatasetBuilder:
    """
    Builds unified canonical datasets for soil moisture ML models.

    This class orchestrates data collection from multiple sources,
    runs physics simulations, and constructs the feature matrix.

    Architecture:
    ------------
    SpaceIoTBox API ──┐
                      │
    Open-Meteo ───────┼──► Raw Data Alignment ──► Physics Model ──► Feature Engineering
                      │
    Soil Data ────────┘

    Output Features:
    ---------------
    1. Climate Forcings: P, ET0, T, RH, wind, radiation (current + lagged + rolling)
    2. Physics Priors: θ_surface, θ_root, θ_deep, fluxes, residuals
    3. Remote Sensing: NDVI, EVI, LAI, SAR backscatter
    4. Static Attributes: Soil texture, topography, land cover
    5. Temporal Features: DOY cycles, season, growing stage
    6. Derived Features: Water balance, aridity index, stress indicators
    """

    def __init__(
        self,
        config: Optional[DatasetConfig] = None,
        feature_config: Optional[FeatureConfig] = None,
    ):
        """
        Initialize the dataset builder.

        Args:
            config: Dataset configuration
            feature_config: Feature engineering configuration
        """
        self.config = config or DatasetConfig()
        self.feature_config = feature_config or FeatureConfig()
        self.logger = logger

        # Initialize data sources lazily
        self._spaceiotbox_client = None
        self._weather_source = None
        self._soil_source = None

        # Caches
        self._soil_profile_cache: Dict[SiteID, SoilProfile] = {}
        self._site_metadata_cache: Dict[SiteID, SiteMetadata] = {}

        # Build metrics
        self.build_metrics: Dict[str, Any] = {}

    def build(
        self,
        site_id: SiteID,
        start_date: date,
        end_date: date,
        site_metadata: Optional[SiteMetadata] = None,
        observations: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build canonical dataset for a single site.

        Args:
            site_id: Site identifier
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            site_metadata: Optional site metadata (will be fetched if not provided)
            observations: Optional DataFrame of soil moisture observations

        Returns:
            DataFrame with unified feature space
        """
        build_start = datetime.now()
        self.logger.info(
            f"Building canonical dataset for {site_id} "
            f"from {start_date} to {end_date}"
        )

        try:
            # Step 1: Get or create site metadata
            metadata = site_metadata or self._get_site_metadata(site_id)

            # Step 2: Fetch all raw data
            raw_data = self._fetch_all_data(
                site_id, metadata, start_date, end_date
            )

            # Step 3: Get soil profile
            soil_profile = self._get_soil_profile(site_id, metadata)

            # Step 4: Run physics model
            physics_output = self._run_physics_model(
                site_id, raw_data, soil_profile, metadata
            )

            # Step 5: Build base table with aligned data
            base_df = self._build_base_table(
                site_id, metadata, start_date, end_date,
                raw_data, physics_output, soil_profile
            )

            # Step 6: Add observations if available
            if observations is not None:
                base_df = self._merge_observations(base_df, observations)

            # Step 7: Engineer features
            feature_df = self._engineer_features(base_df, metadata)

            # Step 8: Quality control and validation
            final_df = self._quality_control(feature_df)

            # Record metrics
            build_time = (datetime.now() - build_start).total_seconds()
            self.build_metrics = {
                "site_id": site_id,
                "n_rows": len(final_df),
                "n_features": len(final_df.columns),
                "build_time_s": build_time,
                "data_coverage": self._calculate_coverage(final_df),
                "date_range": f"{start_date} to {end_date}",
            }

            self.logger.info(
                "Built dataset: %d rows, %d features in %.1f s",
                len(final_df), len(final_df.columns), build_time,
            )

            return final_df

        except Exception as e:
            self.logger.error("Failed to build dataset for %s: %s", site_id, e)
            raise DataSourceError(f"Dataset build failed: {e}")

    def build_multi_site(
        self,
        sites: List[Tuple[SiteID, SiteMetadata]],
        start_date: date,
        end_date: date,
        observations: Optional[Dict[SiteID, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Build canonical dataset for multiple sites.

        Args:
            sites: List of (site_id, metadata) tuples
            start_date: Start date
            end_date: End date
            observations: Dict mapping site_id to observation DataFrames

        Returns:
            Concatenated DataFrame with site_id column
        """
        all_dfs = []

        for site_id, metadata in sites:
            try:
                site_obs = observations.get(site_id) if observations else None
                df = self.build(site_id, start_date,
                                end_date, metadata, site_obs)
                all_dfs.append(df)
            except Exception as e:
                self.logger.warning("Failed to build for %s: %s", site_id, e)
                continue

        if not all_dfs:
            raise DataSourceError("No sites were successfully processed")

        return pd.concat(all_dfs, ignore_index=True)

    # =========================================================================
    # Data Fetching
    # =========================================================================

    def _fetch_all_data(
        self,
        site_id: SiteID,
        metadata: SiteMetadata,
        start_date: date,
        end_date: date,
    ) -> Dict[str, Any]:
        """Fetch all required data from various sources."""

        raw_data = {
            "weather": [],
            "remote_sensing": [],
            "agro": {},
            "soil": None,
        }

        # Extend date range for warmup
        fetch_start = start_date - \
            timedelta(days=self.config.physics_warmup_days)

        # Try SpaceIoTBox first
        if self.config.use_spaceiotbox:
            try:
                spaceiotbox_data = self._fetch_from_spaceiotbox(
                    site_id, metadata, fetch_start, end_date
                )
                raw_data.update(spaceiotbox_data)
            except Exception as e:
                self.logger.warning("SpaceIoTBox fetch failed: %s", e)

        # Fallback to Open-Meteo for weather
        if not raw_data["weather"] and self.config.use_open_meteo_fallback:
            self.logger.info("Falling back to Open-Meteo for weather data")
            raw_data["weather"] = self._fetch_weather_fallback(
                metadata.latitude, metadata.longitude, fetch_start, end_date
            )

        return raw_data

    def _fetch_from_spaceiotbox(
        self,
        site_id: SiteID,
        metadata: SiteMetadata,
        start_date: date,
        end_date: date,
    ) -> Dict[str, Any]:
        """Fetch data from SpaceIoTBox API."""
        from smps.data.sources.spaceiotbox import (
            SpaceIoTBoxWeatherSource,
            SpaceIoTBoxCopernicusSource,
            SpaceIoTBoxAgroSource,
        )

        result = {
            "weather": [],
            "remote_sensing": [],
            "agro": {},
        }

        # Weather data
        weather_source = SpaceIoTBoxWeatherSource()
        weather_source.register_site(
            site_id, metadata.latitude, metadata.longitude)

        request = DataFetchRequest(
            site_id=site_id,
            start_date=start_date,
            end_date=end_date,
        )

        weather_result = weather_source.fetch(request)
        if weather_result.success:
            result["weather"] = weather_result.data

        # Satellite data
        copernicus_source = SpaceIoTBoxCopernicusSource()
        copernicus_source.register_site(
            site_id, metadata.latitude, metadata.longitude)

        rs_result = copernicus_source.fetch(request)
        if rs_result.success:
            result["remote_sensing"] = rs_result.data

        # Agro data
        agro_source = SpaceIoTBoxAgroSource()
        agro_source.register_site(
            site_id, metadata.latitude, metadata.longitude)

        agro_result = agro_source.fetch(request)
        if agro_result.success:
            result["agro"] = agro_result.data

        return result

    def _fetch_weather_fallback(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> List[DailyWeather]:
        """Fetch weather from Open-Meteo as fallback."""
        from smps.data.sources.weather import OpenMeteoSource

        source = OpenMeteoSource()
        request = DataFetchRequest(
            site_id=f"{latitude}_{longitude}",
            start_date=start_date,
            end_date=end_date,
        )

        result = source.fetch(request)
        return result.data if result.success else []

    def _get_site_metadata(self, site_id: SiteID) -> SiteMetadata:
        """Get or create site metadata."""
        if site_id in self._site_metadata_cache:
            return self._site_metadata_cache[site_id]

        # Try to parse coordinates from site_id
        try:
            parts = str(site_id).split("_")
            if len(parts) >= 2:
                lat, lon = float(parts[0]), float(parts[1])
                metadata = SiteMetadata(
                    site_id=site_id,
                    latitude=lat,
                    longitude=lon,
                )
                self._site_metadata_cache[site_id] = metadata
                return metadata
        except (ValueError, IndexError):
            pass

        # Default metadata (should be provided by caller)
        raise DataValidationError(
            f"Site metadata not found for {site_id}. "
            "Please provide SiteMetadata when calling build()."
        )

    def _get_soil_profile(
        self,
        site_id: SiteID,
        metadata: SiteMetadata,
    ) -> SoilProfile:
        """Get soil profile from cache or fetch."""
        if site_id in self._soil_profile_cache:
            return self._soil_profile_cache[site_id]

        # Try SoilGrids or iSDA
        try:
            from smps.data.sources.soilgrids import SoilGridsGlobalSource

            soil_source = SoilGridsGlobalSource()
            profile = soil_source.fetch_soil_profile(
                site_id,
                latitude=metadata.latitude,
                longitude=metadata.longitude,
            )
            self._soil_profile_cache[site_id] = profile
            return profile
        except Exception as e:
            self.logger.warning("Soil fetch failed: %s, using defaults", e)
            return self._create_default_soil_profile(site_id, metadata)

    def _create_default_soil_profile(
        self,
        site_id: SiteID,
        metadata: SiteMetadata,
    ) -> SoilProfile:
        """Create default soil profile based on texture class."""
        # Default loam properties
        texture_defaults = {
            "sand": {"sand": 85, "silt": 10, "clay": 5},
            "loam": {"sand": 40, "silt": 40, "clay": 20},
            "clay": {"sand": 20, "silt": 30, "clay": 50},
            "sandy_loam": {"sand": 65, "silt": 25, "clay": 10},
            "clay_loam": {"sand": 30, "silt": 35, "clay": 35},
        }

        texture = metadata.soil_texture_class or "loam"
        tex = texture_defaults.get(texture, texture_defaults["loam"])

        # Estimate hydraulic properties
        params = estimate_soil_parameters_tropical(
            sand_percent=tex["sand"],
            clay_percent=tex["clay"],
        )

        return SoilProfile(
            site_id=site_id,
            sand_percent=tex["sand"],
            silt_percent=tex["silt"],
            clay_percent=tex["clay"],
            porosity=params.porosity,
            field_capacity=params.field_capacity,
            wilting_point=params.wilting_point,
            saturated_hydraulic_conductivity_cm_day=params.saturated_hydraulic_conductivity_cm_day,
            profile_depth_cm=100.0,
            source="estimated",
        )

    # =========================================================================
    # Physics Model
    # =========================================================================

    def _run_physics_model(
        self,
        site_id: SiteID,
        raw_data: Dict[str, Any],
        soil_profile: SoilProfile,
        metadata: SiteMetadata,
    ) -> pd.DataFrame:
        """Run physics model to generate priors."""
        if not self.config.run_physics_model:
            return pd.DataFrame()

        weather_records = raw_data.get("weather", [])
        if not weather_records:
            self.logger.warning("No weather data for physics model")
            return pd.DataFrame()

        # Convert weather to DataFrame
        weather_df = pd.DataFrame([
            w.model_dump() if hasattr(w, 'model_dump') else w.dict()
            for w in weather_records
        ])
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        weather_df.set_index('date', inplace=True)
        weather_df.sort_index(inplace=True)

        # Determine soil texture for model
        soil_texture = self._classify_soil_texture(
            soil_profile.sand_percent,
            soil_profile.clay_percent,
        )

        # Create physics model
        physics_model = create_water_balance_model(
            crop_type=metadata.crop_type or self.config.physics_crop_type,
            soil_texture=soil_texture,
            use_full_physics=True,
        )

        # Prepare forcings
        forcings = weather_df[['precipitation_mm', 'et0_mm']].copy()

        # Add NDVI from remote sensing if available
        rs_data = raw_data.get("remote_sensing", [])
        if rs_data:
            rs_df = pd.DataFrame([
                r.model_dump() if hasattr(r, 'model_dump') else r.dict()
                for r in rs_data
            ])
            if 'ndvi' in rs_df.columns:
                rs_df['date'] = pd.to_datetime(rs_df['date'])
                rs_df.set_index('date', inplace=True)
                forcings['ndvi'] = rs_df['ndvi'].reindex(forcings.index)

        if 'ndvi' not in forcings.columns:
            forcings['ndvi'] = 0.5  # Default

        forcings['irrigation_mm'] = 0.0  # No irrigation by default

        # Run model
        try:
            physics_results = physics_model.run_period(
                forcings=forcings,
                warmup_days=min(self.config.physics_warmup_days,
                                len(forcings) // 4),
                return_fluxes=self.feature_config.include_physics_fluxes,
            )
            return physics_results
        except Exception as e:
            self.logger.error("Physics model failed: %s", e)
            return pd.DataFrame()

    def _classify_soil_texture(self, sand: float, clay: float) -> str:
        """Classify soil texture from sand/clay percentages."""
        if sand > 70:
            return "sand"
        elif clay > 40:
            return "clay"
        elif sand > 50 and clay < 20:
            return "sandy_loam"
        elif clay > 25:
            return "clay_loam"
        else:
            return "loam"

    # =========================================================================
    # Table Building
    # =========================================================================

    def _build_base_table(
        self,
        site_id: SiteID,
        metadata: SiteMetadata,
        start_date: date,
        end_date: date,
        raw_data: Dict[str, Any],
        physics_output: pd.DataFrame,
        soil_profile: SoilProfile,
    ) -> pd.DataFrame:
        """Build the base canonical table with all aligned data."""

        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        base_df = pd.DataFrame({
            'date': date_range,
            'site_id': site_id,
        })
        base_df.set_index('date', inplace=True)

        # ----- Add Weather Data -----
        weather_records = raw_data.get("weather", [])
        if weather_records:
            weather_df = pd.DataFrame([
                w.model_dump() if hasattr(w, 'model_dump') else w.dict()
                for w in weather_records
            ])
            weather_df['date'] = pd.to_datetime(weather_df['date'])
            weather_df.set_index('date', inplace=True)

            # Select weather columns
            weather_cols = [
                'precipitation_mm', 'et0_mm',
                'temperature_mean_c', 'temperature_min_c', 'temperature_max_c',
                'solar_radiation_mj_m2', 'relative_humidity_mean',
                'wind_speed_mean_m_s', 'vapor_pressure_deficit_kpa',
            ]
            for col in weather_cols:
                if col in weather_df.columns:
                    base_df[col] = weather_df[col].reindex(base_df.index)

        # ----- Add Physics Output -----
        if not physics_output.empty:
            # Map physics columns
            physics_col_map = {
                'theta_phys_surface': 'physics_theta_surface',
                'theta_phys_root': 'physics_theta_root',
                'theta_phys_deep': 'physics_theta_deep',
                'theta_surface': 'physics_theta_surface',
                'theta_root': 'physics_theta_root',
                'theta_deep': 'physics_theta_deep',
            }

            physics_df = physics_output.copy()
            if not isinstance(physics_df.index, pd.DatetimeIndex):
                physics_df.index = pd.to_datetime(physics_df.index)

            # Rename and add columns
            physics_df = physics_df.rename(columns=physics_col_map)

            for col in physics_df.columns:
                if col.startswith('physics_') or col.startswith('flux_'):
                    base_df[col] = physics_df[col].reindex(base_df.index)

        # ----- Add Remote Sensing -----
        rs_data = raw_data.get("remote_sensing", [])
        if rs_data:
            rs_df = pd.DataFrame([
                r.model_dump() if hasattr(r, 'model_dump') else r.dict()
                for r in rs_data
            ])
            rs_df['date'] = pd.to_datetime(rs_df['date'])
            rs_df.set_index('date', inplace=True)

            rs_cols = ['ndvi', 'evi', 'lai', 'sar_vv_db', 'sar_vh_db']
            for col in rs_cols:
                if col in rs_df.columns:
                    base_df[col] = rs_df[col].reindex(base_df.index)

        # ----- Add Soil Properties (Static) -----
        base_df['sand_percent'] = soil_profile.sand_percent
        base_df['silt_percent'] = soil_profile.silt_percent
        base_df['clay_percent'] = soil_profile.clay_percent
        base_df['porosity'] = soil_profile.porosity
        base_df['field_capacity'] = soil_profile.field_capacity
        base_df['wilting_point'] = soil_profile.wilting_point
        base_df['sat_hydraulic_cond'] = soil_profile.saturated_hydraulic_conductivity_cm_day

        # Derived soil properties
        base_df['available_water_capacity'] = (
            soil_profile.field_capacity - soil_profile.wilting_point
        )
        base_df['plant_available_water'] = (
            soil_profile.porosity - soil_profile.wilting_point
        )

        # ----- Add Site Metadata (Static) -----
        base_df['latitude'] = metadata.latitude
        base_df['longitude'] = metadata.longitude
        base_df['elevation_m'] = metadata.elevation_m
        base_df['slope_degrees'] = metadata.slope_degrees
        base_df['twi'] = metadata.twi

        # Reset index for further processing
        base_df = base_df.reset_index()

        return base_df

    def _merge_observations(
        self,
        base_df: pd.DataFrame,
        observations: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge soil moisture observations into base table."""
        result = base_df.copy()

        if observations.empty:
            return result

        obs = observations.copy()

        # Ensure date column
        if 'timestamp' in obs.columns:
            obs['date'] = pd.to_datetime(obs['timestamp']).dt.date
        elif 'date' in obs.columns:
            obs['date'] = pd.to_datetime(obs['date']).dt.date

        result['date'] = pd.to_datetime(result['date']).dt.date

        # Pivot observations by depth
        if 'depth_cm' in obs.columns and 'vwc' in obs.columns:
            for depth in obs['depth_cm'].unique():
                depth_obs = obs[obs['depth_cm'] == depth][['date', 'vwc']]
                depth_obs = depth_obs.groupby(
                    'date')['vwc'].mean().reset_index()
                depth_obs.columns = ['date', f'obs_vwc_{depth}cm']
                result = result.merge(depth_obs, on='date', how='left')

        # Map to standard depth columns
        depth_mapping = {
            10: 'obs_vwc_surface',
            30: 'obs_vwc_root',
            50: 'obs_vwc_deep',
        }

        for depth, col_name in depth_mapping.items():
            source_col = f'obs_vwc_{depth}cm'
            if source_col in result.columns and col_name not in result.columns:
                result[col_name] = result[source_col]

        result['date'] = pd.to_datetime(result['date'])

        return result

    # =========================================================================
    # Feature Engineering
    # =========================================================================

    def _engineer_features(
        self,
        df: pd.DataFrame,
        metadata: SiteMetadata,
    ) -> pd.DataFrame:
        """Engineer all features for the ML model."""
        result = df.copy()
        result = result.sort_values('date')

        # ----- Temporal Features -----
        result = self._add_temporal_features(result)

        # ----- Lag Features -----
        result = self._add_lag_features(result)

        # ----- Rolling Features -----
        result = self._add_rolling_features(result)

        # ----- Cumulative Features -----
        result = self._add_cumulative_features(result)

        # ----- Physics-Derived Features -----
        result = self._add_physics_features(result)

        # ----- Remote Sensing Features -----
        if self.feature_config.include_ndvi_features:
            result = self._add_ndvi_features(result)

        # ----- Interaction Features -----
        if self.feature_config.include_interactions:
            result = self._add_interaction_features(result)

        # ----- Climate Indices -----
        result = self._add_climate_indices(result)

        return result

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal and seasonal features."""
        result = df.copy()
        dates = pd.to_datetime(result['date'])

        # Day of year (cyclic encoding for continuity)
        doy = dates.dt.dayofyear
        result['day_of_year'] = doy

        if self.feature_config.cyclic_encoding:
            result['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
            result['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)

            # Monthly cycle
            month = dates.dt.month
            result['month_sin'] = np.sin(2 * np.pi * month / 12)
            result['month_cos'] = np.cos(2 * np.pi * month / 12)

        result['month'] = dates.dt.month
        result['week_of_year'] = dates.dt.isocalendar().week.astype(int)

        # Season (meteorological)
        if self.feature_config.include_season:
            result['season'] = dates.dt.month.map({
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'fall', 10: 'fall', 11: 'fall',
            })
            # One-hot encode season
            season_dummies = pd.get_dummies(result['season'], prefix='season')
            result = pd.concat([result, season_dummies], axis=1)

        # Days since start (trend feature)
        result['days_since_start'] = (dates - dates.min()).dt.days

        return result

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features for key variables."""
        result = df.copy()

        # Weather lags
        weather_lag_cols = ['precipitation_mm', 'et0_mm', 'temperature_mean_c']
        for col in weather_lag_cols:
            if col in result.columns:
                for lag in self.feature_config.lag_days:
                    result[f'{col}_lag{lag}'] = result[col].shift(lag)

        # Physics prior lags
        physics_lag_cols = ['physics_theta_surface',
                            'physics_theta_root', 'physics_theta_deep']
        for col in physics_lag_cols:
            if col in result.columns:
                for lag in [1, 3, 7, 14]:
                    result[f'{col}_lag{lag}'] = result[col].shift(lag)

        # Observation lags (if available)
        obs_cols = [c for c in result.columns if c.startswith('obs_vwc_')]
        for col in obs_cols:
            for lag in [1, 3, 7, 14, 30]:
                result[f'{col}_lag{lag}'] = result[col].shift(lag)

        return result

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics."""
        result = df.copy()

        # Precipitation rolling sums
        if 'precipitation_mm' in result.columns:
            for window in self.feature_config.rolling_windows:
                result[f'precip_sum_{window}d'] = (
                    result['precipitation_mm']
                    .rolling(window, min_periods=1).sum()
                )
                result[f'precip_max_{window}d'] = (
                    result['precipitation_mm']
                    .rolling(window, min_periods=1).max()
                )
                result[f'precip_days_{window}d'] = (
                    (result['precipitation_mm'] > 0.1)
                    .rolling(window, min_periods=1).sum()
                )

        # ET0 rolling statistics
        if 'et0_mm' in result.columns:
            for window in self.feature_config.rolling_windows:
                result[f'et0_mean_{window}d'] = (
                    result['et0_mm']
                    .rolling(window, min_periods=1).mean()
                )
                result[f'et0_sum_{window}d'] = (
                    result['et0_mm']
                    .rolling(window, min_periods=1).sum()
                )

        # Temperature rolling stats
        if 'temperature_mean_c' in result.columns:
            for window in [7, 14, 30]:
                result[f'temp_mean_{window}d'] = (
                    result['temperature_mean_c']
                    .rolling(window, min_periods=1).mean()
                )
                result[f'temp_std_{window}d'] = (
                    result['temperature_mean_c']
                    .rolling(window, min_periods=1).std()
                )

        # Physics theta rolling stats
        for theta_col in ['physics_theta_surface', 'physics_theta_root']:
            if theta_col in result.columns:
                for window in [7, 14]:
                    result[f'{theta_col}_mean_{window}d'] = (
                        result[theta_col]
                        .rolling(window, min_periods=1).mean()
                    )
                    result[f'{theta_col}_std_{window}d'] = (
                        result[theta_col]
                        .rolling(window, min_periods=1).std()
                    )
                    result[f'{theta_col}_trend_{window}d'] = (
                        result[theta_col].diff(window) / window
                    )

        return result

    def _add_cumulative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cumulative water balance features."""
        result = df.copy()

        if 'precipitation_mm' in result.columns and 'et0_mm' in result.columns:
            # Daily water balance
            result['daily_water_balance'] = (
                result['precipitation_mm'] - result['et0_mm']
            )

            # Cumulative water balance over windows
            for window in self.feature_config.cumulative_windows:
                result[f'water_balance_{window}d'] = (
                    result['daily_water_balance']
                    .rolling(window, min_periods=1).sum()
                )

            # Antecedent Precipitation Index (API)
            # API = sum(P_i * k^i) where k is decay factor
            decay_factor = 0.85
            result['api_30d'] = 0.0
            for i in range(30):
                result['api_30d'] += (
                    result['precipitation_mm'].shift(i).fillna(0)
                    * (decay_factor ** i)
                )

        return result

    def _add_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add physics-derived features."""
        result = df.copy()

        # Physics residuals (if observations available)
        for depth in ['surface', 'root', 'deep']:
            physics_col = f'physics_theta_{depth}'
            obs_col = f'obs_vwc_{depth}'

            if physics_col in result.columns and obs_col in result.columns:
                result[f'physics_residual_{depth}'] = (
                    result[obs_col] - result[physics_col]
                )

        # Relative saturation
        if 'physics_theta_root' in result.columns and 'porosity' in result.columns:
            result['relative_saturation'] = (
                result['physics_theta_root'] / result['porosity']
            )

        # Soil moisture deficit
        if 'physics_theta_root' in result.columns and 'field_capacity' in result.columns:
            result['soil_moisture_deficit'] = (
                result['field_capacity'] - result['physics_theta_root']
            ).clip(lower=0)

        # Available water fraction
        if all(c in result.columns for c in ['physics_theta_root', 'wilting_point', 'field_capacity']):
            result['available_water_fraction'] = (
                (result['physics_theta_root'] - result['wilting_point']) /
                (result['field_capacity'] - result['wilting_point'] + 0.001)
            ).clip(0, 1)

        # Stress indicator
        if 'available_water_fraction' in result.columns:
            result['water_stress_index'] = 1 - \
                result['available_water_fraction']

        return result

    def _add_ndvi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add NDVI-derived features."""
        result = df.copy()

        if 'ndvi' not in result.columns:
            return result

        # NDVI lags
        for lag in self.feature_config.ndvi_lag_days:
            if lag > 0:
                result[f'ndvi_lag{lag}'] = result['ndvi'].shift(lag)

        # NDVI rolling statistics
        for window in [14, 30]:
            result[f'ndvi_mean_{window}d'] = (
                result['ndvi'].rolling(window, min_periods=1).mean()
            )
            result[f'ndvi_std_{window}d'] = (
                result['ndvi'].rolling(window, min_periods=1).std()
            )

        # NDVI trend
        result['ndvi_trend_14d'] = result['ndvi'].diff(14) / 14

        # NDVI anomaly (relative to rolling mean)
        result['ndvi_anomaly'] = result['ndvi'] - \
            result.get('ndvi_mean_30d', result['ndvi'])

        # Vegetation fraction estimate
        result['vegetation_fraction'] = (
            (result['ndvi'] - 0.1) / 0.8).clip(0, 1)

        return result

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between variables."""
        result = df.copy()

        # Precipitation × soil moisture interactions
        if 'precipitation_mm' in result.columns:
            if 'physics_theta_root' in result.columns:
                # Infiltration potential (more infiltration when drier)
                result['precip_infiltration_potential'] = (
                    result['precipitation_mm'] *
                    (1 - result['physics_theta_root'])
                )

        # ET × soil moisture interactions
        if 'et0_mm' in result.columns and 'physics_theta_root' in result.columns:
            result['et_stress_ratio'] = (
                result['et0_mm'] / (result['physics_theta_root'] + 0.01)
            )

        # Temperature × moisture interactions
        if 'temperature_mean_c' in result.columns and 'physics_theta_root' in result.columns:
            result['temp_moisture_product'] = (
                result['temperature_mean_c'] * result['physics_theta_root']
            )

        # NDVI × soil moisture
        if 'ndvi' in result.columns and 'physics_theta_root' in result.columns:
            result['ndvi_moisture_product'] = (
                result['ndvi'] * result['physics_theta_root']
            )

        # Clay × moisture (drainage indicator)
        if 'clay_percent' in result.columns and 'physics_theta_root' in result.columns:
            result['clay_moisture_product'] = (
                result['clay_percent'] / 100 * result['physics_theta_root']
            )

        return result

    def _add_climate_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add climate-related indices."""
        result = df.copy()

        if 'precipitation_mm' in result.columns and 'et0_mm' in result.columns:
            # Aridity Index (P/ET0)
            for window in [7, 30]:
                precip_sum = result['precipitation_mm'].rolling(
                    window, min_periods=1).sum()
                et0_sum = result['et0_mm'].rolling(window, min_periods=1).sum()
                result[f'aridity_index_{window}d'] = precip_sum / \
                    (et0_sum + 0.1)

            # Moisture Index
            result['moisture_index'] = (
                result['precipitation_mm'] - result['et0_mm']
            ) / (result['et0_mm'] + 0.1)

        # Vapor Pressure Deficit normalized
        if 'vapor_pressure_deficit_kpa' in result.columns:
            result['vpd_normalized'] = (
                result['vapor_pressure_deficit_kpa'] /
                result['vapor_pressure_deficit_kpa'].rolling(30).mean()
            )

        return result

    # =========================================================================
    # Quality Control
    # =========================================================================

    def _quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality control and fill missing values."""
        result = df.copy()

        # Remove duplicate dates
        if 'date' in result.columns:
            result = result.drop_duplicates(
                subset=['date', 'site_id'], keep='first')

        # Handle missing values
        result = self._fill_missing_values(result)

        # Remove outliers
        result = self._remove_outliers(result)

        # Add quality flags
        result = self._add_quality_flags(result)

        return result

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply tiered missing value imputation."""
        result = df.copy()

        # Columns to skip
        skip_cols = {'site_id', 'date', 'season'}

        for col in result.columns:
            if col in skip_cols or result[col].dtype == 'object':
                continue

            null_count = result[col].isna().sum()
            if null_count == 0:
                continue

            # Tier 1: Forward/backward fill for short gaps
            result[col] = result[col].ffill(limit=self.config.max_gap_days)
            result[col] = result[col].bfill(limit=self.config.max_gap_days)

            # Tier 2: Linear interpolation
            remaining = result[col].isna().sum()
            if remaining > 0:
                result[col] = result[col].interpolate(
                    method='linear',
                    limit=self.config.max_gap_days * 2
                )

            # Tier 3: Column mean as last resort
            if result[col].isna().any():
                col_mean = result[col].mean()
                if pd.notna(col_mean):
                    result[col] = result[col].fillna(col_mean)

        return result

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove or cap outliers."""
        result = df.copy()

        # Physical bounds for soil moisture
        theta_cols = [
            c for c in result.columns if 'theta' in c.lower() or 'vwc' in c.lower()]
        for col in theta_cols:
            if col in result.columns:
                result[col] = result[col].clip(0, 1)

        # Precipitation bounds
        if 'precipitation_mm' in result.columns:
            result['precipitation_mm'] = result['precipitation_mm'].clip(
                0, 500)

        # ET0 bounds
        if 'et0_mm' in result.columns:
            result['et0_mm'] = result['et0_mm'].clip(0, 20)

        return result

    def _add_quality_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add data quality flags."""
        result = df.copy()

        # Calculate coverage for each row
        feature_cols = [
            c for c in result.columns if c not in ['site_id', 'date']]
        result['data_coverage'] = result[feature_cols].notna().mean(axis=1)

        # Flag rows below minimum coverage
        result['quality_flag'] = np.where(
            result['data_coverage'] >= self.config.min_data_coverage,
            'good',
            'poor'
        )

        return result

    def _calculate_coverage(self, df: pd.DataFrame) -> float:
        """Calculate overall data coverage."""
        if df.empty:
            return 0.0

        feature_cols = [c for c in df.columns if c not in [
            'site_id', 'date', 'quality_flag']]
        return df[feature_cols].notna().mean().mean()
