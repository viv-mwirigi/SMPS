"""
Data Pipeline for SWPPS.

This module provides comprehensive data enrichment and canonical table building
for soil moisture prediction pipelines. It integrates weather, soil, satellite,
and physics data into a unified dataset.
"""

from smps.core.types import SiteID
from smps.data.sources.stubs import IsdaAfricaAuthenticatedSource, SoilGridsGlobalSource
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

from smps.data.quality import QualityControlPipeline, WeatherGapFiller, QCConfig, run_weather_qc
from smps.data.weather import OpenMeteoClient, WeatherFetchRequest
from smps.physics.water_balance import create_water_balance_model
from smps.physics.van_genuchten import potential_from_water_content

# Import soil data sources from SMPS
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logger = logging.getLogger("swpps.data.pipeline")


@dataclass
class DataPipelineConfig:
    """Configuration for data pipeline."""
    cache_dir: Path = Path("data/cache")
    weather_cache_dir: Optional[Path] = None
    soil_cache_dir: Optional[Path] = None
    satellite_cache_dir: Optional[Path] = None

    skip_weather_fetch: bool = False
    skip_soil_fetch: bool = False
    skip_satellite_fetch: bool = True  # Default to skip for simplicity

    # Weather quality control settings
    enable_weather_qc: bool = True
    fill_weather_gaps: bool = True
    max_weather_gap_days: int = 7
    weather_qc_config: Optional[QCConfig] = None

    max_stations: Optional[int] = None
    min_station_days: int = 30


class DataPipeline:
    """
    Comprehensive data pipeline for soil moisture prediction.

    Orchestrates data fetching, enrichment, and canonical table building.
    """

    def __init__(self, config: Optional[DataPipelineConfig] = None):
        self.config = config or DataPipelineConfig()

        # Setup cache directories
        self.weather_cache = (
            self.config.weather_cache_dir or self.config.cache_dir / "weather"
        )
        self.soil_cache = (
            self.config.soil_cache_dir or self.config.cache_dir / "soil"
        )
        self.satellite_cache = (
            self.config.satellite_cache_dir or self.config.cache_dir / "satellite"
        )

        for cache_dir in [self.weather_cache, self.soil_cache, self.satellite_cache]:
            cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize clients
        self.weather_client = OpenMeteoClient(cache_dir=self.weather_cache)

        # Initialize soil data sources
        self.soil_sources = []
        if not self.config.skip_soil_fetch:
            try:
                # Try iSDA first (Africa only, higher quality)
                isda_source = IsdaAfricaAuthenticatedSource(
                    cache_dir=self.soil_cache)
                self.soil_sources.append(isda_source)
                logger.info("iSDA Africa soil source available")
            except Exception as e:
                logger.warning(f"Failed to initialize iSDA source: {e}")

            try:
                # Add SoilGrids as fallback (global coverage)
                soilgrids_source = SoilGridsGlobalSource(
                    cache_dir=self.soil_cache)
                self.soil_sources.append(soilgrids_source)
                logger.info("SoilGrids global soil source available")
            except Exception as e:
                logger.warning(f"Failed to initialize SoilGrids source: {e}")

            if not self.soil_sources:
                logger.error("No soil data sources available!")
                raise ValueError("No soil data sources could be initialized")

    def build_canonical_table(
        self,
        station_data: pd.DataFrame,
        max_stations: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Build canonical table by enriching station data with external sources.

        Args:
            station_data: DataFrame with station observations
            max_stations: Limit number of stations for testing

        Returns:
            Enriched DataFrame with weather, soil, physics, and satellite data
        """
        logger.info("Building canonical table...")

        # Get unique locations
        locations = self._get_unique_locations(station_data)
        if max_stations:
            locations = locations.head(max_stations)

        logger.info(f"Processing {len(locations)} stations...")

        enriched_stations = []

        for _, loc in tqdm(locations.iterrows(), total=len(locations), desc="Enriching stations"):
            station_df = self._enrich_single_station(loc, station_data)
            if station_df is not None and len(station_df) >= self.config.min_station_days:
                enriched_stations.append(station_df)

        if not enriched_stations:
            raise ValueError("No valid stations after enrichment")

        # Combine all stations
        canonical = pd.concat(enriched_stations, ignore_index=True)

        # Add derived features
        canonical = self._add_derived_features(canonical)

        logger.info(
            f"Canonical table: {len(canonical):,} rows, {len(canonical.columns)} columns")
        return canonical

    def _get_unique_locations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract unique station locations."""
        return df.groupby('station_id').agg({
            'latitude': 'first',
            'longitude': 'first',
            'elevation_m': 'first',
            'clay_pct': 'first',
            'sand_pct': 'first',
            'silt_pct': 'first',
            'organic_carbon_pct': 'first',
        }).reset_index()

    def _enrich_single_station(
        self, location: pd.Series, all_data: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Enrich a single station with external data."""
        station_id = location['station_id']
        lat, lon = location['latitude'], location['longitude']

        # Get station's observation data
        station_obs = all_data[all_data['station_id'] == station_id].copy()
        if len(station_obs) < self.config.min_station_days:
            return None

        # Ensure date column is datetime
        station_obs['date'] = pd.to_datetime(station_obs['date'])
        station_obs = station_obs.sort_values('date')
        start_date = station_obs['date'].min()
        end_date = station_obs['date'].max()

        # 1. Fetch weather data (MANDATORY for physics model)
        weather_df = None
        if not self.config.skip_weather_fetch:
            try:
                weather_df = self._fetch_weather_data(
                    station_id, lat, lon, start_date, end_date)
                if weather_df is None or len(weather_df) == 0:
                    logger.error(
                        f"No weather data available for {station_id} - cannot run physics model")
                    return None
            except Exception as e:
                logger.error(
                    f"Weather fetch failed for {station_id}: {e} - cannot run physics model")
                return None
        else:
            logger.error(
                f"Weather fetch disabled for {station_id} - physics model requires weather data")
            return None

        # Merge weather data (now guaranteed to be available)
        station_obs = station_obs.merge(weather_df, on='date', how='left')

        # 2. Update soil properties
        soil_data = self._get_soil_data(station_id, lat, lon, location)
        for col, value in soil_data.items():
            if value is not None:
                station_obs[col] = value

        # 3. Run physics model
        physics_df = self._run_physics_model(station_obs, location)
        if physics_df is not None:
            station_obs = station_obs.merge(physics_df, on='date', how='left')

        # 4. Add satellite data (placeholder for now)
        if not self.config.skip_satellite_fetch:
            satellite_data = self._get_satellite_data(
                station_id, lat, lon, start_date, end_date)
            for col, value in satellite_data.items():
                if value is not None:
                    station_obs[col] = value

        return station_obs

    def _fetch_weather_data(
        self,
        station_id: str,
        lat: float,
        lon: float,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch weather data for a station."""
        cache_file = self.weather_cache / \
            f"weather_{station_id.replace('/', '_')}.parquet"

        # Check cache
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                # Ensure date column is datetime
                df['date'] = pd.to_datetime(df['date'])

                # Check if cached data covers the required date range
                cached_start = df['date'].min()
                cached_end = df['date'].max()
                required_days = (end_date - start_date).days + 1

                if (cached_start <= start_date and cached_end >= end_date and
                        len(df) >= required_days):  # Require complete coverage
                    logger.info(f"Using cached weather data for {station_id}")
                    return df
                else:
                    logger.info(
                        f"Cached weather data incomplete for {station_id}, fetching fresh data")
            except Exception as e:
                logger.warning(
                    f"Cached weather data corrupted for {station_id}: {e}, fetching fresh data")

        # Fetch new data
        request = WeatherFetchRequest(
            latitude=lat,
            longitude=lon,
            start_date=start_date.date(),
            end_date=end_date.date()
        )

        try:
            weather_data = self.weather_client.fetch_daily_weather(request)

            # Convert to DataFrame
            records = []
            for w in weather_data:
                records.append({
                    'date': w.date,
                    'temperature_2m': w.temperature_mean_c,
                    'precipitation_mm': w.precipitation_mm,
                    'et0_mm': w.et0_mm,
                    'relative_humidity_2m': w.relative_humidity_mean,
                    'wind_speed_10m': w.wind_speed_m_s,
                    'shortwave_radiation': w.solar_radiation_mj_m2,
                })

            weather_df = pd.DataFrame(records)

            # Ensure date column is datetime for merging
            if len(weather_df) > 0:
                weather_df['date'] = pd.to_datetime(weather_df['date'])

                # Apply weather quality control and gap filling
                if self.config.enable_weather_qc:
                    logger.debug(f"Running weather QC for {station_id}")
                    qc_config = self.config.weather_qc_config or QCConfig()

                    # Run QC
                    weather_df, qc_result = run_weather_qc(
                        weather_df,
                        config=qc_config,
                        fill_gaps=self.config.fill_weather_gaps,
                        max_gap_days=self.config.max_weather_gap_days
                    )

                    # Log QC results
                    if qc_result.n_flagged > 0:
                        logger.warning(
                            f"Weather QC for {station_id}: {qc_result.summary()}")

                    # Check if we still have enough valid data after QC
                    valid_fraction = 1.0 - \
                        (qc_result.n_flagged /
                         qc_result.n_total) if qc_result.n_total > 0 else 1.0
                    if valid_fraction < 0.5:  # Less than 50% valid data
                        logger.warning(
                            f"Low weather data quality for {station_id}: {valid_fraction:.1%} valid")
                        if not self.config.fill_weather_gaps:
                            return None  # Reject station if gap filling is disabled

            # Cache
            if len(weather_df) > 0:
                weather_df.to_parquet(cache_file, index=False)

            return weather_df

        except Exception as e:
            logger.warning(f"Weather fetch failed: {e}")
            return None

    def _get_soil_data(self, station_id: str, lat: float, lon: float, location: pd.Series) -> Dict[str, float]:
        """Get soil properties by fetching from available data sources with cache fallback."""
        site_id = SiteID(station_id)
        cache_file = self.soil_cache / \
            f"soil_{station_id.replace('/', '_')}.json"

        # Check cache first
        if cache_file.exists():
            try:
                import json
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                logger.info(f"Using cached soil data for {station_id}")
                return cached_data
            except Exception as e:
                logger.warning(
                    f"Failed to load cached soil data for {station_id}: {e}")

        # Try to fetch fresh data
        fresh_data = None
        for source in self.soil_sources:
            try:
                logger.info(
                    f"Fetching fresh soil data for {station_id} from {source.name}")
                profile = source.fetch_soil_profile(
                    site_id, latitude=lat, longitude=lon, depth="0-20")

                # Convert to the format expected by the pipeline
                fresh_data = {
                    'clay_pct': profile.clay_percent,
                    'sand_pct': profile.sand_percent,
                    'silt_pct': profile.silt_percent,
                    'porosity': profile.porosity,
                    'field_capacity': profile.field_capacity,
                    'wilting_point': profile.wilting_point,
                    'bulk_density': 1.4,  # Estimate from porosity if not available
                    'slope_degrees': 5.0,  # Default, could be fetched from DEM
                    'source': profile.source,
                    'confidence': profile.confidence
                }

                # Estimate bulk density from porosity if possible
                if profile.porosity:
                    # Rough estimate: bulk_density = (1 - porosity) * 2.65 (assuming mineral density)
                    fresh_data['bulk_density'] = (1 - profile.porosity) * 2.65

                # Cache the fresh data
                try:
                    import json
                    with open(cache_file, 'w') as f:
                        json.dump(fresh_data, f, indent=2)
                    logger.info(f"Cached fresh soil data for {station_id}")
                except Exception as e:
                    logger.warning(
                        f"Failed to cache soil data for {station_id}: {e}")

                logger.info(
                    f"Successfully fetched fresh soil data from {profile.source}")
                return fresh_data

            except Exception as e:
                logger.warning(f"Failed to fetch from {source.name}: {e}")
                continue

        # If fresh fetch failed and no cache available, raise error
        raise ValueError(
            f"Could not fetch soil data for station {station_id} from any source and no cache available")

    def _run_physics_model(
        self, station_df: pd.DataFrame, location: pd.Series
    ) -> Optional[pd.DataFrame]:
        """Run physics model to generate priors."""
        station_id = location['station_id']
        # Check required columns (no defaults allowed)
        required_cols = ['date', 'precipitation_mm',
                         'et0_mm', 'clay_pct', 'sand_pct']
        if not all(col in station_df.columns for col in required_cols):
            logger.warning(
                f"Missing required columns for physics model: {required_cols}")
            return None

        # Check for missing values in required data
        if station_df['precipitation_mm'].isna().any() or station_df['et0_mm'].isna().any():
            logger.warning(
                "Missing precipitation or ET0 data - cannot run physics model")
            return None

        # Get soil parameters
        clay_pct = station_df['clay_pct'].iloc[0]
        sand_pct = station_df['sand_pct'].iloc[0]

        if pd.isna(clay_pct) or pd.isna(sand_pct):
            logger.warning(
                "Missing soil texture data - cannot run physics model")
            return None

        # Create physics model
        model = create_water_balance_model(
            sand_percent=sand_pct,
            clay_percent=clay_pct,
            n_layers=3,
            max_depth_m=1.0,
        )

        # Initialize at observed mean if available
        obs_mean = station_df['soil_moisture'].mean()
        if not pd.isna(obs_mean):
            init_theta = np.clip(obs_mean, 0.05, 0.45)
            # Convert to potential for initialization
            init_psi = potential_from_water_content(
                init_theta,
                # Use surface layer params
                model.config.layers[0].van_genuchten
            )
            model.reset(initial_psi_kpa=init_psi)

        # Run model for the period
        dates = station_df['date'].tolist()
        precip = station_df['precipitation_mm'].tolist()
        et0 = station_df['et0_mm'].tolist()

        # Validate weather data completeness
        missing_weather = station_df[[
            'precipitation_mm', 'et0_mm']].isna().any(axis=1).sum()
        if missing_weather > 0:
            logger.warning(
                f"Missing weather data for {missing_weather} days in {station_id} - physics model may be inaccurate")

        try:
            outputs = model.run_period(
                dates=dates,
                precipitation=precip,
                et0=et0,
                warmup_days=10,  # Short warmup for initialization
            )

            results = []
            for i, output in enumerate(outputs):
                # Compute theta from psi for backward compatibility
                output_with_theta = output.compute_theta_from_psi(
                    model.config.layers[0].van_genuchten
                )

                results.append({
                    'date': dates[i],
                    'physics_prior_surface': output.psi_surface_kpa,  # Store ψ directly
                    'physics_prior_root': output.psi_root_kpa,        # Store ψ directly
                    'physics_prior_deep': output.psi_deep_kpa,        # Store ψ directly
                })

        except Exception as e:
            logger.warning(
                f"Physics model failed for station {station_id}: {e}")
            results = []
            for _, row in station_df.iterrows():
                results.append({
                    'date': row['date'],
                    'physics_prior_surface': np.nan,
                    'physics_prior_root': np.nan,
                    'physics_prior_deep': np.nan,
                })

        return pd.DataFrame(results)

    def _get_satellite_data(self, station_id: str, lat: float, lon: float,
                            start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Get satellite vegetation data with cache fallback."""
        cache_file = self.satellite_cache / \
            f"satellite_{station_id.replace('/', '_')}.json"

        # Check cache first
        if cache_file.exists():
            try:
                import json
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                logger.info(f"Using cached satellite data for {station_id}")
                return cached_data
            except Exception as e:
                logger.warning(
                    f"Failed to load cached satellite data for {station_id}: {e}")

        # Try to fetch fresh satellite data (placeholder for future implementation)
        try:
            # TODO: Implement actual satellite data fetching (e.g., from Google Earth Engine, Sentinel Hub, etc.)
            # For now, return placeholder values but mark as fresh fetch attempt
            logger.info(
                f"Attempting to fetch fresh satellite data for {station_id} (placeholder)")

            satellite_data = {
                'ndvi_mean': None,  # Would be calculated from NDVI time series
                'lai_mean': None,   # Would be calculated from LAI time series
                'source': 'placeholder',
                'last_updated': datetime.now().isoformat()
            }

            # Cache the data (even if it's placeholder)
            try:
                import json
                with open(cache_file, 'w') as f:
                    json.dump(satellite_data, f, indent=2)
                logger.info(f"Cached satellite data for {station_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to cache satellite data for {station_id}: {e}")

            return satellite_data

        except Exception as e:
            logger.warning(
                f"Failed to fetch satellite data for {station_id}: {e}")
            # Return minimal data structure if even placeholder fails
            return {
                'ndvi_mean': None,
                'lai_mean': None,
            }

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to canonical table."""
        result = df.copy()

        # Note: physics_prior is now in ψ space (kPa), soil_moisture in θ space
        # Residual calculation is handled separately in training

        # Water balance
        if 'precipitation_mm' in result.columns and 'et0_mm' in result.columns:
            result['p_minus_et'] = result['precipitation_mm'] - result['et0_mm']

        # Depth normalization
        if 'depth_cm' in result.columns:
            result['depth_normalized'] = result['depth_cm'] / \
                result['depth_cm'].max()

        # Hydraulic conductivity estimate (simplified)
        if 'sand_pct' in result.columns and 'clay_pct' in result.columns:
            # Simple Kozeny-Carman approximation for Ksat (cm/day)
            result['ksat_estimate'] = np.exp(7.755 + 0.0352 * result['sand_pct'] -
                                             0.967 * result['clay_pct']**0.5)

        return result
