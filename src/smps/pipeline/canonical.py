"""
Canonical Table Builder for Smps.
Orchestrates creation of the unified daily table from all data sources.
"""
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from smps.core.types import SiteID
from smps.core.config import get_config, DataConfig
from smps.core.exceptions import DataSourceError, DataValidationError
from smps.data.contracts import CanonicalDailyRow, SoilProfile
from smps.data.sources.base import DataSourceRegistry, DataFetchRequest
from smps.data.sources.weather import OpenMeteoSource
from smps.data.sources.soil import SoilGridsSource
from smps.physics import create_water_balance_model
from smps.physics.pedotransfer import estimate_soil_parameters
from smps.features.engineering import FeatureEngineer
from smps.data.quality import QualityControlPipeline


class CanonicalTableBuilder:
    """
    Builds the canonical daily table for soil moisture prediction.
    Orchestrates data fetching, physics simulation, and feature engineering.
    """

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or get_config().data
        self.logger = logging.getLogger("smps.pipeline.canonical")

        # Initialize components
        self._initialize_data_sources()
        self.feature_engineer = FeatureEngineer()
        self.qc_pipeline = QualityControlPipeline()

        # Cache for soil profiles (don't fetch repeatedly)
        self._soil_profile_cache: Dict[SiteID, SoilProfile] = {}

        # Performance tracking
        self.metrics = {
            "fetch_time_ms": 0,
            "physics_time_ms": 0,
            "feature_time_ms": 0,
            "rows_processed": 0,
            "data_coverage": 0.0
        }

    def build_for_site(self, site_id: SiteID,
                       start_date: date,
                       end_date: date,
                       include_forecast: bool = True) -> pd.DataFrame:
        """
        Build canonical table for a single site.

        Args:
            site_id: Site identifier
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            include_forecast: Whether to include weather forecast data

        Returns:
            DataFrame with canonical daily rows
        """
        start_time = datetime.now()
        self.logger.info(
            f"Building canonical table for {site_id} from {start_date} to {end_date}")

        try:
            # Step 1: Fetch all required data
            self.logger.debug("Step 1: Fetching data from sources...")
            raw_data = self._fetch_all_data(
                site_id, start_date, end_date, include_forecast)

            # Step 2: Get or create soil profile
            self.logger.debug("Step 2: Getting soil profile...")
            soil_profile = self._get_soil_profile(site_id)

            # Step 3: Run physics prior model
            self.logger.debug("Step 3: Running physics prior model...")
            physics_data = self._run_physics_prior(
                site_id, start_date, end_date, raw_data, soil_profile
            )

            # Step 4: Build base table
            self.logger.debug("Step 4: Building base table...")
            base_table = self._build_base_table(
                site_id, start_date, end_date, raw_data, physics_data, soil_profile
            )

            # Step 5: Engineer features
            self.logger.debug("Step 5: Engineering features...")
            feature_table = self.feature_engineer.engineer_features(
                base_table, site_id)

            # Step 6: Apply quality control
            self.logger.debug("Step 6: Applying quality control...")
            final_table = self._apply_quality_control(feature_table, site_id)

            # Step 7: Validate and finalize
            self.logger.debug("Step 7: Validating table...")
            validated_table = self._validate_table(final_table, site_id)

            # Calculate metrics
            processing_time = (
                datetime.now() - start_time).total_seconds() * 1000
            self.metrics["total_time_ms"] = processing_time
            self.metrics["rows_processed"] = len(validated_table)
            self.metrics["data_coverage"] = self._calculate_data_coverage(
                validated_table)

            self.logger.info(
                f"Built table with {len(validated_table)} rows in {processing_time:.0f}ms. "
                f"Coverage: {self.metrics['data_coverage']:.1%}"
            )

            return validated_table

        except Exception as e:
            self.logger.error(
                f"Failed to build canonical table for {site_id}: {e}")
            raise DataSourceError(f"Canonical table build failed: {e}")

    def build_for_multiple_sites(self, site_ids: List[SiteID],
                                 start_date: date,
                                 end_date: date,
                                 max_workers: int = 4) -> Dict[SiteID, pd.DataFrame]:
        """
        Build canonical tables for multiple sites in parallel.
        """
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all site builds
            future_to_site = {
                executor.submit(self.build_for_site, site_id, start_date, end_date): site_id
                for site_id in site_ids
            }

            # Collect results
            for future in as_completed(future_to_site):
                site_id = future_to_site[future]
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results[site_id] = result
                    self.logger.info(f"Successfully built table for {site_id}")
                except Exception as e:
                    self.logger.error(
                        f"Failed to build table for {site_id}: {e}")
                    # Empty DataFrame for failed sites
                    results[site_id] = pd.DataFrame()

        return results

    def _initialize_data_sources(self):
        """Initialize data source registry"""
        self.data_registry = DataSourceRegistry()

        # Register available sources
        cache_dir = Path(
            self.config.cache_dir) if self.config.cache_dir else None

        # Weather sources
        if "open_meteo" in self.config.enabled_sources:
            self.data_registry.register(OpenMeteoSource(cache_dir))
        # No fallback - system requires real weather data

        # Soil sources - try ISDA first if credentials available, then SoilGrids, then mock
        from smps.data.sources.isda_authenticated import IsdaAfricaAuthenticatedSource

        isda_source = IsdaAfricaAuthenticatedSource(cache_dir=cache_dir)
        if isda_source.username and isda_source.password:
            self.data_registry.register(isda_source)
            self.logger.info("Registered iSDA Africa soil source")
        else:
            self.logger.info("iSDA credentials not available, skipping")

        if "soilgrids" in self.config.enabled_sources:
            self.data_registry.register(SoilGridsSource(cache_dir))
        # No fallback - system requires real soil data

        # Remote sensing sources
        try:
            from smps.data.sources import GoogleEarthEngineSatelliteSource
            if GoogleEarthEngineSatelliteSource is not None:
                gee_source = GoogleEarthEngineSatelliteSource(
                    cache_dir=cache_dir)
                self.data_registry.register(gee_source)
                self.logger.info(
                    "Registered Google Earth Engine satellite source")
            else:
                self.logger.warning("Google Earth Engine not available")
        except Exception as e:
            self.logger.warning(f"Could not initialize GEE source: {e}")

        # Note: Remote sensing and sensor sources would be added here
        # For now, only configured real data sources are used

        self.logger.info(
            f"Initialized data sources: {list(self.data_registry.get_all().keys())}")

    def _fetch_all_data(self, site_id: SiteID,
                        start_date: date,
                        end_date: date,
                        include_forecast: bool) -> Dict[str, Any]:
        """
        Fetch data from all enabled sources.
        """
        raw_data = {
            "weather": [],
            "soil": None,
            "remote_sensing": [],
            "irrigation": [],
            "sensor": []
        }

        # Create fetch request
        request = DataFetchRequest(
            site_id=site_id,
            start_date=start_date,
            end_date=end_date,
            parameters={"include_forecast": include_forecast}
        )

        # Fetch from all sources
        source_results = self.data_registry.fetch_all(request)

        # Process results
        for source_name, result in source_results.items():
            if not result.success:
                self.logger.warning(
                    f"Source {source_name} failed: {result.errors}")
                continue

            # Categorize data based on source type
            if "weather" in source_name.lower():
                raw_data["weather"].extend(result.data)
            elif "soil" in source_name.lower():
                raw_data["soil"] = result.data
            elif "gee" in source_name.lower() or "satellite" in source_name.lower():
                raw_data["remote_sensing"].extend(result.data)

        # If no weather data, raise error
        if not raw_data["weather"]:
            raise DataSourceError(
                "No weather data fetched from any source. Real data sources must be available."
            )

        # Merge NDVI data into weather data
        raw_data["weather"] = self._merge_ndvi_into_weather(
            raw_data["weather"], raw_data["remote_sensing"])

        return raw_data

    def _merge_ndvi_into_weather(self, weather_data: List, remote_sensing_data: List) -> List:
        """
        Merge NDVI data from remote sensing into weather data.

        Args:
            weather_data: List of DailyWeather objects
            remote_sensing_data: List of RemoteSensingData objects

        Returns:
            Updated weather_data with NDVI values
        """
        if not remote_sensing_data:
            self.logger.warning("No remote sensing data available for NDVI")
            return weather_data

        # Create NDVI lookup by date
        ndvi_by_date = {}
        for rs_data in remote_sensing_data:
            if rs_data.ndvi is not None:
                ndvi_by_date[rs_data.date] = rs_data.ndvi

        if not ndvi_by_date:
            self.logger.warning("No NDVI values found in remote sensing data")
            return weather_data

        # Merge NDVI into weather data
        updated_weather = []
        for weather in weather_data:
            # Find closest NDVI date (within 8 days for 16-day MODIS composite)
            closest_ndvi = None
            min_days_diff = float('inf')

            for ndvi_date, ndvi_value in ndvi_by_date.items():
                days_diff = abs((ndvi_date - weather.date).days)
                if days_diff <= 8 and days_diff < min_days_diff:  # 8 days tolerance
                    min_days_diff = days_diff
                    closest_ndvi = ndvi_value

            if closest_ndvi is not None:
                # Create a new weather object with NDVI
                weather_dict = weather.dict()
                weather_dict['ndvi'] = closest_ndvi
                # Reconstruct the object (this is a bit hacky, but works)
                updated_weather.append(weather.__class__(**weather_dict))
            else:
                updated_weather.append(weather)

        self.logger.info(
            f"Merged NDVI data for {len([w for w in updated_weather if hasattr(w, 'ndvi') and w.ndvi is not None])} weather records")
        return updated_weather

    def _get_soil_profile(self, site_id: SiteID) -> SoilProfile:
        """Get soil profile from cache or fetch fresh"""
        if site_id in self._soil_profile_cache:
            self.logger.debug(f"Using cached soil profile for {site_id}")
            return self._soil_profile_cache[site_id]

        # Try sources in priority order: ISDA, SoilGrids, Mock
        sources_to_try = ["isda_africa", "soilgrids", "mock_soil"]

        for source_name in sources_to_try:
            soil_source = self.data_registry.get(source_name)
            if soil_source:
                try:
                    soil_profile = soil_source.fetch_soil_profile(site_id)
                    self._soil_profile_cache[site_id] = soil_profile
                    self.logger.info(
                        f"Successfully fetched soil profile from {source_name}")
                    return soil_profile
                except Exception as e:
                    self.logger.warning(
                        f"Failed to fetch from {source_name}: {e}")
                    continue

        raise DataValidationError(
            f"No soil profile available for {site_id} from any source")

    def _run_physics_prior(self, site_id: SiteID,
                           start_date: date,
                           end_date: date,
                           raw_data: Dict[str, Any],
                           soil_profile: SoilProfile) -> pd.DataFrame:
        """
        Run two-bucket water balance model to generate physics prior.
        """
        # Extract weather data
        weather_records = raw_data.get("weather", [])
        if not weather_records:
            raise DataValidationError("No weather data for physics model")

        # Convert to DataFrame
        weather_df = pd.DataFrame([w.dict() for w in weather_records])
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        weather_df.set_index('date', inplace=True)

        # Prepare soil parameters for physics model
        soil_params = {
            'sand_percent': soil_profile.sand_percent,
            'clay_percent': soil_profile.clay_percent,
            'porosity': soil_profile.porosity,
            'field_capacity': soil_profile.field_capacity,
            'wilting_point': soil_profile.wilting_point,
            'saturated_hydraulic_conductivity': soil_profile.saturated_hydraulic_conductivity_cm_day
        }

        # Determine soil texture for model factory
        sand = soil_profile.sand_percent
        clay = soil_profile.clay_percent
        if sand > 70:
            soil_texture = "sand"
        elif clay > 40:
            soil_texture = "clay"
        elif sand > 50 and clay < 20:
            soil_texture = "sandy_loam"
        elif clay > 25:
            soil_texture = "clay_loam"
        else:
            soil_texture = "loam"

        # Initialize physics model using factory
        physics_model = create_water_balance_model(
            crop_type="maize",
            soil_texture=soil_texture,
            use_full_physics=True
        )

        # Prepare irrigation data (if available)
        irrigation_df = None
        if raw_data.get("irrigation"):
            irrigation_df = pd.DataFrame(
                [i.dict() for i in raw_data["irrigation"]])
            irrigation_df['timestamp'] = pd.to_datetime(
                irrigation_df['timestamp'])
            irrigation_df.set_index('timestamp', inplace=True)

        # Build forcings DataFrame for run_period
        forcings = weather_df[['precipitation_mm', 'et0_mm']].copy()

        # Check for NDVI data - required for physics model
        if 'ndvi' in weather_df.columns and weather_df['ndvi'].notna().any():
            forcings['ndvi'] = weather_df['ndvi']
            self.logger.info("Using NDVI data from remote sensing")
        else:
            raise DataValidationError(
                "NDVI data is required for physics model but not available. "
                "Ensure Google Earth Engine is properly configured and can fetch satellite data.")

        if irrigation_df is not None:
            forcings['irrigation_mm'] = irrigation_df['volume_mm'].reindex(
                forcings.index).fillna(0)
        else:
            forcings['irrigation_mm'] = 0.0

        # Run model using run_period for efficient batch processing
        try:
            physics_results = physics_model.run_period(
                forcings=forcings,
                warmup_days=min(30, len(forcings) // 4),
                return_fluxes=False
            )

            # Ensure physics_results has the correct date index
            # Physics model returns results only after warmup period
            warmup_days = min(30, len(forcings) // 4)
            result_dates = forcings.index[warmup_days:]
            if len(physics_results) == len(result_dates):
                physics_results.index = result_dates
                physics_results.index.name = 'date'
            else:
                self.logger.warning(
                    f"Physics results length mismatch: {len(physics_results)} vs {len(result_dates)}")
                raise DataValidationError(
                    "Physics model returned wrong number of results")

            # Check if results are valid (not all NaN)
            self.logger.info(
                f"Physics results columns: {list(physics_results.columns)}")
            self.logger.info(f"Physics results shape: {physics_results.shape}")
            if 'theta_phys_surface' in physics_results.columns:
                self.logger.info(
                    f"theta_phys_surface values: {physics_results['theta_phys_surface'].head()}")
                if physics_results['theta_phys_surface'].isna().any():
                    raise DataValidationError(
                        "Physics model returned NaN results")

            if physics_results.empty or (len(physics_results.columns) == 0):
                raise DataValidationError(
                    "Physics model returned invalid results")

        except Exception as e:
            self.logger.error(
                f"Physics model failed: {e}")
            raise DataValidationError(
                f"Physics model execution failed: {e}") from e

        return physics_results

    def _build_base_table(self, site_id: SiteID,
                          start_date: date,
                          end_date: date,
                          raw_data: Dict[str, Any],
                          physics_data: pd.DataFrame,
                          soil_profile: SoilProfile) -> pd.DataFrame:
        """
        Build base table by merging all data sources.
        """
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        base_df = pd.DataFrame({'date': date_range})
        base_df['site_id'] = site_id
        base_df.set_index(['site_id', 'date'], inplace=True)

        # Add weather data
        weather_df = pd.DataFrame([w.dict()
                                  for w in raw_data.get("weather", [])])
        if not weather_df.empty:
            weather_df['date'] = pd.to_datetime(weather_df['date'])
            weather_df.set_index(['site_id', 'date'], inplace=True)
            base_df = base_df.join(weather_df, how='left')

        # Add physics prior data
        physics_data = physics_data.copy()

        # Rename physics columns to canonical names
        physics_column_renames = {
            'theta_phys_surface': 'physics_theta_surface',
            'theta_phys_root': 'physics_theta_root',
            'theta_phys_deep': 'physics_theta_deep',
        }
        physics_data = physics_data.rename(columns=physics_column_renames)

        physics_data.index.name = 'date'
        physics_data = physics_data.reset_index()
        physics_data['site_id'] = site_id
        physics_data.set_index(['site_id', 'date'], inplace=True)

        # Merge physics data
        for col in physics_data.columns:
            if col in base_df.columns:
                # Overwrite with physics data where available
                base_df[col] = physics_data[col].combine_first(base_df[col])
            else:
                base_df[col] = physics_data[col]

        # Add soil properties (static)
        base_df['sand_percent'] = soil_profile.sand_percent
        base_df['silt_percent'] = soil_profile.silt_percent
        base_df['clay_percent'] = soil_profile.clay_percent
        base_df['porosity'] = soil_profile.porosity
        base_df['field_capacity'] = soil_profile.field_capacity
        base_df['wilting_point'] = soil_profile.wilting_point

        # Add site metadata (must come from real data sources)
        # Note: latitude, longitude, elevation_m, crop_type must be provided by data sources

        # Note: NDVI is already merged into weather data in _merge_ndvi_into_weather

        # Add irrigation data
        if raw_data.get("irrigation"):
            # Aggregate irrigation to daily
            irrigation_df = pd.DataFrame(
                [i.dict() for i in raw_data["irrigation"]])
            irrigation_df['date'] = pd.to_datetime(
                irrigation_df['timestamp']).dt.date
            irrigation_df['date'] = pd.to_datetime(irrigation_df['date'])
            daily_irrigation = irrigation_df.groupby(
                ['site_id', 'date'])['volume_mm'].sum()
            base_df['irrigation_mm'] = daily_irrigation
            base_df['irrigation_flag'] = base_df['irrigation_mm'] > 0

        # Fill missing values
        base_df = self._fill_missing_values(base_df)

        return base_df.reset_index()

    def _apply_quality_control(self, df: pd.DataFrame, site_id: SiteID) -> pd.DataFrame:
        """
        Apply quality control to the table.
        """
        result_df = df.copy()

        # Calculate data coverage for each column
        coverage = {}
        for col in result_df.columns:
            if col not in ['site_id', 'date']:
                non_null = result_df[col].notna().sum()
                coverage[col] = non_null / len(result_df)

        # Add coverage information
        result_df.attrs['data_coverage'] = coverage
        result_df.attrs['overall_coverage'] = np.mean(list(coverage.values()))

        # Flag rows with poor data coverage
        required_cols = ['precipitation_mm', 'et0_mm', 'physics_theta_root']
        missing_required = result_df[required_cols].isna().any(axis=1)

        if missing_required.any():
            self.logger.warning(
                f"{missing_required.sum()} rows missing required data")

        return result_df

    def _validate_table(self, df: pd.DataFrame, site_id: SiteID) -> pd.DataFrame:
        """
        Validate canonical table and convert to proper format.
        """
        # Get the expected fields from CanonicalDailyRow model
        from smps.data.contracts import CanonicalDailyRow
        import inspect

        # Get model fields
        model_fields = set(CanonicalDailyRow.model_fields.keys())

        # Filter DataFrame to only include expected columns
        available_columns = [col for col in df.columns if col in model_fields]
        filtered_df = df[available_columns].copy()

        # Check for missing required columns (those without defaults in the model)
        required_fields_without_defaults = {
            'site_id', 'date', 'latitude', 'longitude', 'precipitation_mm', 'et0_mm',
            'temperature_mean_c', 'temperature_min_c', 'temperature_max_c',
            'solar_radiation_mj_m2', 'relative_humidity_mean', 'wind_speed_mean_m_s',
            'precip_cumulative_3d', 'precip_cumulative_7d', 'precip_cumulative_14d', 'precip_cumulative_30d',
            'et0_cumulative_3d', 'et0_cumulative_7d', 'et0_cumulative_14d',
            'sand_percent', 'silt_percent', 'clay_percent', 'porosity', 'field_capacity', 'wilting_point',
            'physics_theta_surface', 'physics_theta_root', 'day_of_year', 'day_of_year_sin', 'day_of_year_cos',
            'month', 'season', 'water_deficit_mm', 'available_water_capacity', 'created_at', 'etl_version', 'data_sources'
        }

        missing_required = required_fields_without_defaults - \
            set(filtered_df.columns)
        if missing_required:
            raise DataValidationError(
                f"Missing required fields without defaults: {missing_required}. "
                "All required data must come from real data sources."
            )

        # For fields with defaults in the model, let Pydantic handle them during validation
        # No hardcoded defaults added here

        # Convert to list of CanonicalDailyRow objects (for validation)
        validated_rows = []

        for _, row in filtered_df.iterrows():
            try:
                # Convert row to dict and create canonical row
                row_dict = row.to_dict()

                # Ensure date is date object
                if isinstance(row_dict['date'], pd.Timestamp):
                    row_dict['date'] = row_dict['date'].date()

                # Create canonical row (validates automatically)
                canonical_row = CanonicalDailyRow(**row_dict)
                validated_rows.append(canonical_row)

            except Exception as e:
                self.logger.warning(
                    f"Failed to validate row {row.get('date', 'unknown')}: {e}")
                # Optionally skip invalid rows or keep with flag

        # Convert back to DataFrame
        validated_df = pd.DataFrame([r.dict() for r in validated_rows])

        # Add validation metadata
        validated_df.attrs['validation_status'] = 'validated'
        validated_df.attrs['valid_rows'] = len(validated_rows)
        validated_df.attrs['total_rows'] = len(df)
        validated_df.attrs['validation_timestamp'] = datetime.now().isoformat()

        return validated_df

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply tiered missing value imputation.
        """
        result_df = df.copy()

        for col in result_df.columns:
            if col in ['site_id', 'date']:
                continue

            null_count = result_df[col].isna().sum()
            if null_count == 0:
                continue

            # Tier 1: Forward/backward fill for short gaps
            max_gap = self.config.max_gap_days
            result_df[col] = result_df[col].fillna(
                method='ffill', limit=max_gap)
            result_df[col] = result_df[col].fillna(
                method='bfill', limit=max_gap)

            # Tier 2: Linear interpolation for medium gaps
            remaining_nulls = result_df[col].isna().sum()
            if remaining_nulls > 0:
                result_df[col] = result_df[col].interpolate(
                    method='linear', limit=max_gap*2)

            # Tier 3: Seasonal average for long gaps
            remaining_nulls = result_df[col].isna().sum()
            if remaining_nulls > 0 and 'month' in result_df.columns:
                # Calculate monthly averages
                monthly_avg = result_df.groupby('month')[col].mean()
                for month, avg_value in monthly_avg.items():
                    month_mask = result_df['month'] == month
                    result_df.loc[month_mask &
                                  result_df[col].isna(), col] = avg_value

            # Tier 4: Global mean as last resort
            if result_df[col].isna().any():
                global_mean = result_df[col].mean()
                if pd.notna(global_mean):
                    result_df[col].fillna(global_mean, inplace=True)

        return result_df

    def _calculate_data_coverage(self, df: pd.DataFrame) -> float:
        """Calculate overall data coverage"""
        if df.empty:
            return 0.0

        # Count non-null values in feature columns
        feature_cols = [
            col for col in df.columns if col not in ['site_id', 'date']]
        total_cells = len(df) * len(feature_cols)

        if total_cells == 0:
            return 0.0

        non_null_cells = df[feature_cols].notna().sum().sum()

        return non_null_cells / total_cells

    def _create_default_soil_profile(self, site_id: SiteID) -> SoilProfile:
        """Create default soil profile when none is available"""
        self.logger.warning(f"Creating default soil profile for {site_id}")

        return SoilProfile(
            site_id=site_id,
            sand_percent=40.0,
            silt_percent=40.0,
            clay_percent=20.0,
            porosity=0.45,
            field_capacity=0.27,
            wilting_point=0.15,
            saturated_hydraulic_conductivity_cm_day=25.0,
            profile_depth_cm=100.0,
            effective_rooting_depth_cm=40.0,
            source="default",
            confidence=0.5
        )


class CanonicalTableManager:
    """
    Manages storage and retrieval of canonical tables.
    Provides versioning and caching.
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.builder = CanonicalTableBuilder()
        self.logger = logging.getLogger("smps.pipeline.manager")

    def get_table(self, site_id: SiteID,
                  start_date: date,
                  end_date: date,
                  force_rebuild: bool = False,
                  cache_ttl_days: int = 7) -> pd.DataFrame:
        """
        Get canonical table, using cache if available and fresh.
        """
        # Generate cache key
        cache_key = self._generate_cache_key(site_id, start_date, end_date)
        cache_path = self.storage_dir / f"{cache_key}.parquet"

        # Check cache
        if not force_rebuild and cache_path.exists():
            cache_age = (
                datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).days

            if cache_age <= cache_ttl_days:
                try:
                    self.logger.info(f"Loading cached table for {site_id}")
                    df = pd.read_parquet(cache_path)

                    # Check if cached data has required metadata
                    if 'etl_version' in df.attrs:
                        return df
                except Exception as e:
                    self.logger.warning(f"Failed to load cached table: {e}")

        # Build fresh table
        self.logger.info(f"Building fresh table for {site_id}")
        df = self.builder.build_for_site(site_id, start_date, end_date)

        # Add metadata
        df.attrs['etl_version'] = '1.0'
        df.attrs['build_timestamp'] = datetime.now().isoformat()
        df.attrs['cache_key'] = cache_key

        # Cache the result
        try:
            df.to_parquet(cache_path)
            self.logger.info(f"Cached table to {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cache table: {e}")

        return df

    def _generate_cache_key(self, site_id: SiteID,
                            start_date: date,
                            end_date: date) -> str:
        """Generate cache key for table"""
        import hashlib
        key_string = f"{site_id}_{start_date}_{end_date}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
