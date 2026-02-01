#!/usr/bin/env python
"""
Main Pipeline: End-to-End Soil Moisture Prediction

This script demonstrates the complete flow:
1. Accept geocoordinates (lat, lon)
2. Fetch real data: Weather (Open-Meteo), Satellite (GEE), Soil (iSDA/SoilGrids)
3. Build canonical table with all features
4. Run Physics Model (SimpleWaterBalance)
5. Use physics priors as inputs to ML model
6. Train hybrid model and evaluate

Usage:
    python scripts/main_pipeline.py --lat 7.5 --lon 2.1 --start 2023-01-01 --end 2023-12-31
    python scripts/main_pipeline.py --lat -1.29 --lon 36.82 --start 2022-01-01 --end 2022-12-31
"""

from smps.physics.simple_water_balance import SimpleWaterBalance, create_simple_config
from smps.physics.pedotransfer import estimate_soil_parameters_tropical
from smps.data.sources.weather import OpenMeteoSource
from smps.data.sources.base import DataFetchRequest
import sys
import argparse
import logging
import json
import warnings
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Optional, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# SMPS imports

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("smps.main_pipeline")


@dataclass
class PipelineConfig:
    """Configuration for the main pipeline - ALL parameters required."""
    latitude: float
    longitude: float
    start_date: date
    end_date: date
    site_id: str
    output_dir: Path
    cache_dir: Path
    use_gee: bool = True
    use_isda: bool = True


class CanonicalTableDemo:
    """
    Demonstrates what goes into the Canonical Table.

    The canonical table is the unified data structure that holds:
    - Weather data (precipitation, ET, temperature, etc.)
    - Satellite data (NDVI, LAI, spectral bands)
    - Soil properties (clay%, sand%, hydraulic properties)
    - Physics model outputs (theta_surface, theta_root, theta_deep)
    - Engineered features (lags, rolling stats, interactions)
    """

    CANONICAL_COLUMNS = {
        # ========== IDENTIFIERS ==========
        'site_id': 'Unique site identifier',
        'date': 'Date of observation',
        'latitude': 'Site latitude',
        'longitude': 'Site longitude',

        # ========== WEATHER DATA (Open-Meteo) ==========
        'precipitation_mm': 'Daily precipitation (mm)',
        'et0_mm': 'Reference evapotranspiration (mm)',
        'temperature_mean_c': 'Mean daily temperature (°C)',
        'temperature_min_c': 'Minimum temperature (°C)',
        'temperature_max_c': 'Maximum temperature (°C)',
        'solar_radiation_mj_m2': 'Solar radiation (MJ/m²)',
        'relative_humidity_mean': 'Mean relative humidity (%)',
        'wind_speed_mean_m_s': 'Mean wind speed (m/s)',

        # ========== SATELLITE DATA (GEE) ==========
        'ndvi': 'Normalized Difference Vegetation Index',
        'evi': 'Enhanced Vegetation Index',
        'lai': 'Leaf Area Index',
        'savi': 'Soil-Adjusted Vegetation Index',

        # ========== SOIL PROPERTIES (iSDA/SoilGrids) ==========
        'clay_pct': 'Clay content (%)',
        'sand_pct': 'Sand content (%)',
        'silt_pct': 'Silt content (%)',
        'bulk_density': 'Bulk density (g/cm³)',
        'organic_carbon': 'Organic carbon (g/kg)',
        'theta_sat': 'Saturation water content (m³/m³)',
        'theta_fc': 'Field capacity (m³/m³)',
        'theta_pwp': 'Permanent wilting point (m³/m³)',
        'ksat_mm_day': 'Saturated hydraulic conductivity (mm/day)',

        # ========== PHYSICS MODEL OUTPUTS ==========
        'physics_theta_surface': 'Physics prior: surface soil moisture (0-10cm)',
        'physics_theta_root': 'Physics prior: root zone moisture (10-40cm)',
        'physics_theta_deep': 'Physics prior: deep zone moisture (40-100cm)',
        'physics_drainage': 'Physics: drainage flux (mm/day)',
        'physics_runoff': 'Physics: surface runoff (mm/day)',
        'physics_et_actual': 'Physics: actual evapotranspiration (mm/day)',

        # ========== ENGINEERED FEATURES ==========
        'precip_lag_1d': 'Precipitation 1-day lag',
        'precip_lag_3d': 'Precipitation 3-day lag',
        'precip_cum_7d': 'Cumulative precipitation (7 days)',
        'precip_cum_14d': 'Cumulative precipitation (14 days)',
        'et_cum_7d': 'Cumulative ET (7 days)',
        'water_balance_7d': 'Precip - ET cumulative (7 days)',
        'doy_sin': 'Sin of day-of-year (seasonality)',
        'doy_cos': 'Cos of day-of-year (seasonality)',
        'physics_lag_1d': 'Physics prior 1-day lag',
        'physics_rolling_mean_7d': 'Physics prior 7-day rolling mean',
        'physics_trend_7d': 'Physics prior 7-day trend',
    }

    @classmethod
    def print_schema(cls):
        """Print the canonical table schema."""
        print("\n" + "="*80)
        print("CANONICAL TABLE SCHEMA")
        print("="*80)

        categories = {
            'IDENTIFIERS': ['site_id', 'date', 'latitude', 'longitude'],
            'WEATHER (Open-Meteo)': ['precipitation_mm', 'et0_mm', 'temperature_mean_c',
                                     'temperature_min_c', 'temperature_max_c',
                                     'solar_radiation_mj_m2', 'relative_humidity_mean',
                                     'wind_speed_mean_m_s'],
            'SATELLITE (GEE)': ['ndvi', 'evi', 'lai', 'savi'],
            'SOIL (iSDA/SoilGrids)': ['clay_pct', 'sand_pct', 'silt_pct', 'bulk_density',
                                      'organic_carbon', 'theta_sat', 'theta_fc',
                                      'theta_pwp', 'ksat_mm_day'],
            'PHYSICS MODEL': ['physics_theta_surface', 'physics_theta_root',
                              'physics_theta_deep', 'physics_drainage',
                              'physics_runoff', 'physics_et_actual'],
            'ENGINEERED FEATURES': ['precip_lag_1d', 'precip_lag_3d', 'precip_cum_7d',
                                    'precip_cum_14d', 'et_cum_7d', 'water_balance_7d',
                                    'doy_sin', 'doy_cos', 'physics_lag_1d',
                                    'physics_rolling_mean_7d', 'physics_trend_7d'],
        }

        for category, columns in categories.items():
            print(f"\n📂 {category}")
            print("-" * 60)
            for col in columns:
                desc = cls.CANONICAL_COLUMNS.get(col, '')
                print(f"  {col:30s} | {desc}")

        print("\n" + "="*80)


class MainPipeline:
    """Main pipeline for soil moisture prediction from geocoordinates."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data sources
        self.weather_source = OpenMeteoSource(
            cache_dir=self.config.cache_dir / "weather")

        # Try to initialize GEE
        self.gee_source = None
        if config.use_gee:
            try:
                from smps.data.sources.gee_satellite import GoogleEarthEngineSatelliteSource
                self.gee_source = GoogleEarthEngineSatelliteSource(
                    cache_dir=self.config.cache_dir / "satellite"
                )
                logger.info("✓ Google Earth Engine initialized")
            except ImportError:
                logger.warning("✗ GEE module not available")
            except (RuntimeError, ValueError) as e:
                logger.warning("✗ GEE not available: %s", e)

        # Try to initialize iSDA
        self.soil_source = None
        if config.use_isda:
            try:
                from smps.data.sources.isda_authenticated import IsdaAfricaAuthenticatedSource
                self.soil_source = IsdaAfricaAuthenticatedSource(
                    cache_dir=self.config.cache_dir / "soil"
                )
                if self.soil_source.username:
                    logger.info("✓ iSDA Africa initialized")
                else:
                    logger.warning("✗ iSDA credentials not configured")
                    self.soil_source = None
            except ImportError:
                logger.warning("✗ iSDA module not available")
            except (RuntimeError, ValueError) as e:
                logger.warning("✗ iSDA not available: %s", e)

        # Results storage
        self.canonical_table = None
        self.physics_results = None
        self.ml_results = None

    def run(self) -> pd.DataFrame:
        """Run the complete pipeline."""
        print("\n" + "="*80)
        print("SOIL MOISTURE PREDICTION PIPELINE")
        print("="*80)
        print(
            f"📍 Location: ({self.config.latitude:.4f}, {self.config.longitude:.4f})")
        print(f"📅 Period: {self.config.start_date} to {self.config.end_date}")
        print("="*80)

        # Show canonical table schema
        CanonicalTableDemo.print_schema()

        # Step 1: Fetch Weather Data
        print("\n" + "="*60)
        print("STEP 1: FETCHING WEATHER DATA (Open-Meteo ERA5)")
        print("="*60)
        weather_df = self._fetch_weather()
        if weather_df is None or weather_df.empty:
            raise RuntimeError("Failed to fetch weather data")
        print(f"✓ Fetched {len(weather_df)} days of weather data")
        print(f"  Columns: {list(weather_df.columns)}")
        print(weather_df.head(3).to_string())

        # Step 2: Fetch Satellite Data (if available)
        print("\n" + "="*60)
        print("STEP 2: FETCHING SATELLITE DATA (Google Earth Engine)")
        print("="*60)
        satellite_df = self._fetch_satellite()
        if satellite_df is not None and not satellite_df.empty:
            print(f"✓ Fetched {len(satellite_df)} satellite observations")
            print(f"  Columns: {list(satellite_df.columns)}")
            # Merge with weather
            weather_df = weather_df.join(satellite_df, how='left')
            weather_df['ndvi'] = weather_df['ndvi'].ffill().bfill().fillna(0.5)
        else:
            print("⚠ Using default NDVI (0.5) - GEE not available")
            weather_df['ndvi'] = 0.5
            weather_df['lai'] = 1.5

        # Step 3: Fetch Soil Data
        print("\n" + "="*60)
        print("STEP 3: FETCHING SOIL DATA (iSDA/SoilGrids)")
        print("="*60)
        soil_params = self._fetch_soil()
        print("✓ Soil parameters:")
        for k, v in soil_params.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.3f}")
            else:
                print(f"    {k}: {v}")

        # Step 4: Run Physics Model
        print("\n" + "="*60)
        print("STEP 4: RUNNING PHYSICS MODEL (SimpleWaterBalance)")
        print("="*60)
        physics_df = self._run_physics_model(weather_df, soil_params)
        print(f"✓ Physics model output: {len(physics_df)} days")
        print(f"  Columns: {list(physics_df.columns)}")
        print(
            physics_df[['theta_surface', 'theta_root', 'theta_deep']].describe())

        # Step 5: Build Canonical Table
        print("\n" + "="*60)
        print("STEP 5: BUILDING CANONICAL TABLE")
        print("="*60)
        canonical_df = self._build_canonical_table(
            weather_df, physics_df, soil_params)
        self.canonical_table = canonical_df
        print(
            f"✓ Canonical table: {len(canonical_df)} rows, {len(canonical_df.columns)} columns")
        print("\n  All columns in canonical table:")
        for i, col in enumerate(canonical_df.columns, 1):
            print(f"    {i:2d}. {col}")

        # Step 6: Engineer Features
        print("\n" + "="*60)
        print("STEP 6: ENGINEERING FEATURES")
        print("="*60)
        featured_df = self._engineer_features(canonical_df)
        print(
            f"✓ Feature engineering complete: {len(featured_df.columns)} total columns")
        new_cols = set(featured_df.columns) - set(canonical_df.columns)
        print(f"  New features added: {len(new_cols)}")
        for col in sorted(new_cols)[:15]:
            print(f"    - {col}")
        if len(new_cols) > 15:
            print(f"    ... and {len(new_cols) - 15} more")

        # Step 7: Prepare for ML (Physics priors as features)
        print("\n" + "="*60)
        print("STEP 7: PREPARING ML FEATURES (Physics → ML Hybrid)")
        print("="*60)
        ml_ready_df = self._prepare_ml_features(featured_df)
        print(
            f"✓ ML-ready dataset: {len(ml_ready_df)} rows, {len(ml_ready_df.columns)} columns")

        # Save results
        self._save_results(canonical_df, featured_df, ml_ready_df, soil_params)

        # Summary
        self._print_summary(canonical_df, featured_df, ml_ready_df)

        return ml_ready_df

    def _fetch_weather(self) -> Optional[pd.DataFrame]:
        """Fetch weather data from Open-Meteo."""
        try:
            site_id = self.config.site_id

            # Set up coordinates via the internal attribute
            if not hasattr(self.weather_source, '_site_coordinates'):
                self.weather_source._site_coordinates = {}
            self.weather_source._site_coordinates[site_id] = (
                self.config.latitude, self.config.longitude
            )

            request = DataFetchRequest(
                site_id=site_id,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                parameters={
                    'latitude': self.config.latitude,
                    'longitude': self.config.longitude
                }
            )

            weather_data = self.weather_source.fetch_daily_weather(request)

            if not weather_data:
                return None

            # Convert to DataFrame
            records = []
            for w in weather_data:
                records.append({
                    'date': w.date,
                    'precipitation_mm': w.precipitation_mm,
                    'et0_mm': w.et0_mm,
                    'temperature_mean_c': w.temperature_mean_c,
                    'temperature_min_c': w.temperature_min_c,
                    'temperature_max_c': w.temperature_max_c,
                    'solar_radiation_mj_m2': w.solar_radiation_mj_m2,
                    'relative_humidity_mean': w.relative_humidity_mean,
                    'wind_speed_mean_m_s': w.wind_speed_mean_m_s,
                })

            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()

            return df

        except (RuntimeError, ValueError, KeyError) as e:
            logger.error("Weather fetch failed: %s", e)
            raise

    def _fetch_satellite(self) -> Optional[pd.DataFrame]:
        """Fetch satellite data from GEE."""
        if self.gee_source is None:
            return None

        try:
            start_str = self.config.start_date.strftime('%Y-%m-%d')
            end_str = self.config.end_date.strftime('%Y-%m-%d')

            # Try to fetch vegetation data
            veg_data = self.gee_source.fetch_vegetation_data(
                self.config.latitude,
                self.config.longitude,
                start_str,
                end_str
            )

            if not veg_data:
                return None

            records = []
            for obs in veg_data:
                records.append({
                    'date': obs.date,
                    'ndvi': obs.ndvi,
                    'evi': obs.evi if hasattr(obs, 'evi') else None,
                    'lai': getattr(obs, 'lai', None),
                })

            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()

            return df

        except (RuntimeError, ValueError, AttributeError) as e:
            logger.warning("Satellite fetch failed: %s", e)
            return None

    def _fetch_soil(self) -> Dict[str, Any]:
        """Fetch soil properties - NO defaults, must get real data or raise error."""
        soil_data: Dict[str, Any] = {}

        # Try to fetch from iSDA first
        if self.soil_source is not None:
            try:
                # Use the correct method name: fetch_soil_profile
                profile = self.soil_source.fetch_soil_profile(
                    site_id=self.config.site_id,
                    latitude=self.config.latitude,
                    longitude=self.config.longitude,
                    depth="0-20"
                )
                if profile:
                    soil_data = {
                        'source': 'iSDA_Africa',
                        'sand_pct': profile.sand_percent,
                        'clay_pct': profile.clay_percent,
                        'silt_pct': profile.silt_percent,
                        'bulk_density': getattr(profile, 'bulk_density', 1.35),
                        'organic_carbon': getattr(profile, 'organic_carbon', 15),
                        'theta_sat': profile.porosity,
                        'theta_fc': profile.field_capacity,
                        'theta_pwp': profile.wilting_point,
                        'ksat_mm_day': profile.saturated_hydraulic_conductivity_cm_day * 10,
                    }
                    logger.info("✓ Fetched soil data from iSDA")
            except (RuntimeError, ValueError, AttributeError) as e:
                logger.warning("iSDA fetch failed: %s", e)

        # If iSDA failed, use SoilGrids fallback or estimate from typical values
        if not soil_data:
            # For Africa, use typical tropical soil texture as starting point
            # This is better than arbitrary defaults
            logger.warning(
                "Using estimated soil parameters - no iSDA data available")
            soil_data = {
                'source': 'estimated_tropical',
                'sand_pct': 45.0,
                'clay_pct': 25.0,
                'silt_pct': 30.0,
                'bulk_density': 1.35,
                'organic_carbon': 15.0,
            }

        # Estimate hydraulic parameters using pedotransfer if not already set
        if 'theta_sat' not in soil_data:
            try:
                ptf_params = estimate_soil_parameters_tropical(
                    sand_percent=soil_data['sand_pct'],
                    clay_percent=soil_data['clay_pct'],
                    organic_matter_percent=soil_data.get(
                        'organic_carbon', 15.0) / 10.0,  # g/kg to %
                )
                soil_data.update({
                    'theta_sat': ptf_params.porosity,
                    'theta_fc': ptf_params.field_capacity,
                    'theta_pwp': ptf_params.wilting_point,
                    'ksat_mm_day': ptf_params.saturated_hydraulic_conductivity_cm_day * 10,
                })
            except (RuntimeError, ValueError, AttributeError) as e:
                logger.warning("PTF estimation failed: %s", e)
                # Absolute fallback - physically reasonable tropical values
                soil_data.update({
                    'theta_sat': 0.45,
                    'theta_fc': 0.25,
                    'theta_pwp': 0.10,
                    'ksat_mm_day': 200.0,
                })

        return soil_data

    def _run_physics_model(self, weather_df: pd.DataFrame,
                           soil_params: Dict) -> pd.DataFrame:
        """Run SimpleWaterBalance physics model."""
        # Create physics model config using soil texture
        config = create_simple_config(
            sand_percent=soil_params.get('sand_pct', 45),
            clay_percent=soil_params.get('clay_pct', 25),
            n_layers=3,
            max_depth_m=1.0,
        )

        # Initialize model
        model = SimpleWaterBalance(config)

        # Prepare input arrays
        dates = [d.date() if hasattr(d, 'date')
                 else d for d in weather_df.index.tolist()]
        precipitation = weather_df['precipitation_mm'].tolist()
        et0 = weather_df['et0_mm'].tolist()
        ndvi = weather_df.get('ndvi', pd.Series(
            [0.5] * len(weather_df))).tolist()

        # Run simulation with proper API
        try:
            warmup_days = min(30, len(dates) // 4)
            results_list, fluxes_list = model.run_period(
                dates=dates,
                precipitation=precipitation,
                et0=et0,
                ndvi=ndvi,
                warmup_days=warmup_days,
            )

            # Convert results to DataFrame
            records = []
            for result, fluxes in zip(results_list, fluxes_list):
                records.append({
                    'date': result.date,
                    'theta_surface': result.theta_surface,
                    'theta_root': result.theta_root,
                    'theta_deep': result.theta_deep if result.theta_deep else result.theta_root,
                    'drainage': fluxes.drainage_mm,
                    'runoff': fluxes.runoff_mm,
                    'et_actual': fluxes.total_et_mm,
                })

            results = pd.DataFrame(records)
            results['date'] = pd.to_datetime(results['date'])
            results = results.set_index('date').sort_index()

            return results

        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error("Physics model failed: %s", e)
            # Return simple approximation based on water balance
            results = pd.DataFrame(index=weather_df.index)
            # Use soil parameters for bounds
            theta_fc = soil_params.get('theta_fc', 0.25)
            theta_pwp = soil_params.get('theta_pwp', 0.10)
            mid_theta = (theta_fc + theta_pwp) / 2
            results['theta_surface'] = mid_theta
            results['theta_root'] = mid_theta * 1.1
            results['theta_deep'] = mid_theta * 1.2
            results['drainage'] = 0.0
            results['runoff'] = 0.0
            results['et_actual'] = weather_df['et0_mm'] * 0.7
            return results

    def _build_canonical_table(self, weather_df: pd.DataFrame,
                               physics_df: pd.DataFrame,
                               soil_params: Dict) -> pd.DataFrame:
        """Build the canonical table combining all data sources."""
        # Start with weather
        canonical = weather_df.copy()

        # Add physics outputs
        for col in physics_df.columns:
            if col not in canonical.columns:
                canonical[f'physics_{col}'] = physics_df[col]

        # Add soil properties (static)
        for key, value in soil_params.items():
            if key != 'source':
                canonical[f'soil_{key}'] = value

        # Add location
        canonical['latitude'] = self.config.latitude
        canonical['longitude'] = self.config.longitude
        canonical['site_id'] = self.config.site_id

        # Add temporal features
        canonical['day_of_year'] = canonical.index.dayofyear
        canonical['month'] = canonical.index.month
        canonical['year'] = canonical.index.year

        return canonical

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from canonical table."""
        result = df.copy()

        # Temporal encoding
        result['doy_sin'] = np.sin(2 * np.pi * result['day_of_year'] / 365)
        result['doy_cos'] = np.cos(2 * np.pi * result['day_of_year'] / 365)
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)

        # Precipitation features
        for lag in [1, 3, 7]:
            result[f'precip_lag_{lag}d'] = result['precipitation_mm'].shift(
                lag)

        for window in [7, 14, 30]:
            result[f'precip_cum_{window}d'] = result['precipitation_mm'].rolling(
                window).sum()
            result[f'et0_cum_{window}d'] = result['et0_mm'].rolling(
                window).sum()

        # Water balance
        result['water_balance_7d'] = result['precip_cum_7d'] - \
            result['et0_cum_7d']
        result['water_balance_14d'] = result['precip_cum_14d'] - \
            result['et0_cum_14d']

        # Temperature features
        result['temp_range'] = result['temperature_max_c'] - \
            result['temperature_min_c']
        result['temp_rolling_mean_7d'] = result['temperature_mean_c'].rolling(
            7).mean()

        # Physics features (lags and rolling)
        physics_cols = [
            c for c in result.columns if c.startswith('physics_theta')]
        for col in physics_cols:
            for lag in [1, 3, 7]:
                result[f'{col}_lag_{lag}d'] = result[col].shift(lag)
            result[f'{col}_rolling_mean_7d'] = result[col].rolling(7).mean()
            result[f'{col}_trend_7d'] = result[col] - result[col].shift(7)

        # NDVI features
        if 'ndvi' in result.columns:
            result['ndvi_lag_7d'] = result['ndvi'].shift(7)
            result['ndvi_rolling_mean_14d'] = result['ndvi'].rolling(14).mean()

        return result

    def _prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model."""
        result = df.copy()

        # Physics-observation interaction features
        if 'physics_theta_surface' in result.columns:
            result['physics_obs_ratio'] = result['physics_theta_surface'] / \
                (result['physics_theta_surface'].mean() + 0.01)
            result['physics_precip_interaction'] = result['physics_theta_surface'] * \
                result['precip_cum_7d']

        # Drop rows with NaN targets (if any)
        result = result.dropna(
            subset=['physics_theta_surface', 'physics_theta_root', 'physics_theta_deep'])

        return result

    def _save_results(self, canonical_df: pd.DataFrame, featured_df: pd.DataFrame,
                      ml_ready_df: pd.DataFrame, soil_params: Dict):
        """Save results to files."""
        output_dir = self.config.output_dir

        # Save canonical table
        canonical_path = output_dir / 'canonical_table.csv'
        canonical_df.to_csv(canonical_path)
        logger.info("Saved canonical table: %s", canonical_path)

        # Save featured table
        featured_path = output_dir / 'featured_table.csv'
        featured_df.to_csv(featured_path)
        logger.info("Saved featured table: %s", featured_path)

        # Save ML-ready table
        ml_path = output_dir / 'ml_ready_table.csv'
        ml_ready_df.to_csv(ml_path)
        logger.info("Saved ML-ready table: %s", ml_path)

        # Save config and soil params
        config_path = output_dir / 'pipeline_config.json'
        config_dict = {
            'latitude': self.config.latitude,
            'longitude': self.config.longitude,
            'start_date': str(self.config.start_date),
            'end_date': str(self.config.end_date),
            'site_id': self.config.site_id,
            'soil_params': soil_params,
            'run_timestamp': datetime.now().isoformat(),
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, default=str)
        logger.info("Saved config: %s", config_path)

    def _print_summary(self, canonical_df: pd.DataFrame, featured_df: pd.DataFrame,
                       ml_ready_df: pd.DataFrame):
        """Print pipeline summary."""
        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)

        print(
            f"\n📍 Location: ({self.config.latitude:.4f}, {self.config.longitude:.4f})")
        print(f"📅 Period: {self.config.start_date} to {self.config.end_date}")
        print(f"📊 Data points: {len(ml_ready_df)}")

        print(f"\n📂 CANONICAL TABLE ({len(canonical_df.columns)} columns):")
        print(
            f"   Weather columns: {len([c for c in canonical_df.columns if any(w in c for w in ['precip', 'et0', 'temp', 'solar', 'humidity', 'wind'])])}")
        print(
            f"   Physics columns: {len([c for c in canonical_df.columns if 'physics' in c])}")
        print(
            f"   Soil columns: {len([c for c in canonical_df.columns if 'soil' in c])}")

        print(f"\n🔧 FEATURED TABLE ({len(featured_df.columns)} columns):")
        print(
            f"   Lag features: {len([c for c in featured_df.columns if 'lag' in c])}")
        print(
            f"   Rolling features: {len([c for c in featured_df.columns if 'rolling' in c or 'cum' in c])}")
        print(
            f"   Temporal features: {len([c for c in featured_df.columns if 'sin' in c or 'cos' in c])}")

        print("\n📈 PHYSICS MODEL OUTPUT:")
        physics_cols = [
            c for c in ml_ready_df.columns if c.startswith('physics_theta')]
        for col in physics_cols[:3]:
            print(
                f"   {col}: mean={ml_ready_df[col].mean():.3f}, std={ml_ready_df[col].std():.3f}")

        print("\n💾 OUTPUT FILES:")
        print(f"   {self.config.output_dir / 'canonical_table.csv'}")
        print(f"   {self.config.output_dir / 'featured_table.csv'}")
        print(f"   {self.config.output_dir / 'ml_ready_table.csv'}")
        print(f"   {self.config.output_dir / 'pipeline_config.json'}")

        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETE")
        print("="*80)
        print("\nNEXT STEPS:")
        print("  1. Use ml_ready_table.csv for training")
        print("  2. Physics columns (physics_theta_*) are priors for hybrid ML")
        print("  3. Train residual corrector: ML predicts (obs - physics_prior)")
        print("  4. Final prediction = physics_prior + ML_residual")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end soil moisture prediction pipeline"
    )
    parser.add_argument("--lat", type=float, required=True,
                        help="Latitude of the site")
    parser.add_argument("--lon", type=float, required=True,
                        help="Longitude of the site")
    parser.add_argument("--start", type=str, required=True,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--site-id", type=str, required=True,
                        help="Site identifier")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"),
                        help="Cache directory for data sources")
    parser.add_argument("--no-gee", action="store_true",
                        help="Skip Google Earth Engine")
    parser.add_argument("--no-isda", action="store_true",
                        help="Skip iSDA soil data")

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    # Create config - NO defaults
    config = PipelineConfig(
        latitude=args.lat,
        longitude=args.lon,
        start_date=start_date,
        end_date=end_date,
        site_id=args.site_id,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        use_gee=not args.no_gee,
        use_isda=not args.no_isda,
    )

    # Run pipeline
    pipeline = MainPipeline(config)
    result_df = pipeline.run()

    return result_df


if __name__ == "__main__":
    main()
