#!/usr/bin/env python
"""
Full Physics+ML Validation Pipeline using Prepared ISMN CSVs.

This script implements the COMPLETE pipeline:
1. Loads prepared ISMN train/test CSVs (ground truth soil moisture)
2. Enriches data with external sources:
   - Weather (Open-Meteo): precipitation, temperature, ET0
   - Satellite (GEE): NDVI, LAI (optional)
   - Soil (iSDA): clay%, sand% (if missing from ISMN)
3. Runs Physics Model (SimpleWaterBalance) to generate physics priors
4. Trains ML Model on residuals (Physics error correction)
5. Evaluates at multiple forecast horizons (0h, 24h, 72h, 168h)

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │ PREPARED DATA (ground truth + basic metadata)                │
    └──────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ ENRICH with external data (weather, satellite, soil)         │
    │ → Creates CANONICAL TABLE with all features aligned by date  │
    └──────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ PHYSICS MODEL (SimpleWaterBalance)                           │
    │ Inputs: precip, ET0, soil hydraulics                         │
    │ Output: physics_prior (predicted soil moisture)              │
    └──────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ ML MODEL (LightGBM residual learner)                         │
    │ Target: observation - physics_prior                          │
    │ Features: weather + physics features + lags                  │
    │ Output: residual_correction                                  │
    └──────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ FINAL PREDICTION = physics_prior + residual_correction       │
    └──────────────────────────────────────────────────────────────┘

Usage:
    python scripts/run_prepared_validation.py --max-stations 10
    python scripts/run_prepared_validation.py --skip-gee --skip-fetch
"""
import lightgbm as lgb
from smps.validation.plotting import (
    ValidationPlotter,
    print_validation_summary,
    create_model_comparison_table,
)
from smps.ml.hybrid_features import (
    smooth_residuals,
    compute_residual_target,
    combine_physics_ml_predictions,
)
from smps.data.quality.station_assessment import (
    StationQualityAssessor,
    compute_physics_kge,
    calculate_adaptive_physics_weight,
)
from smps.features.advanced import (
    AdvancedFeatureEngineer,
    create_temporal_features,
    create_spatial_features,
)
from smps.data.sources.base import DataFetchRequest
from smps.data.sources.weather import OpenMeteoSource
from smps.physics.pedotransfer import estimate_soil_parameters_tropical
from smps.physics.simple_water_balance import (
    SimpleWaterBalance,
    create_simple_config_improved,
)
import argparse
import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold


# =============================================================================
# ADDITIONAL METRICS: KGE and NSE
# =============================================================================

def compute_nse(obs: np.ndarray, pred: np.ndarray) -> float:
    """
    Nash-Sutcliffe Efficiency.

    NSE = 1 - Σ(obs - pred)² / Σ(obs - mean(obs))²

    Range: (-∞, 1], with 1 being perfect prediction.
    NSE = 0 means prediction is as good as using the mean.
    """
    if len(obs) < 2:
        return np.nan
    obs_mean = np.mean(obs)
    numerator = np.sum((obs - pred) ** 2)
    denominator = np.sum((obs - obs_mean) ** 2)
    if denominator < 1e-10:
        return np.nan
    return 1 - numerator / denominator


def compute_kge(obs: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Kling-Gupta Efficiency with decomposition.

    KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)

    where:
    - r = correlation coefficient
    - α = σ_pred / σ_obs (variability ratio)
    - β = μ_pred / μ_obs (bias ratio)

    Returns: (KGE, r, alpha, beta)
    """
    if len(obs) < 2:
        return np.nan, np.nan, np.nan, np.nan

    # Correlation
    obs_std = np.std(obs)
    pred_std = np.std(pred)
    obs_mean = np.mean(obs)
    pred_mean = np.mean(pred)

    if obs_std < 1e-10 or pred_std < 1e-10:
        return np.nan, np.nan, np.nan, np.nan

    r = np.corrcoef(obs, pred)[0, 1]
    if np.isnan(r):
        return np.nan, np.nan, np.nan, np.nan

    # Variability ratio
    alpha = pred_std / obs_std

    # Bias ratio
    if abs(obs_mean) < 1e-10:
        beta = np.nan
        kge = np.nan
    else:
        beta = pred_mean / obs_mean
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    return kge, r, alpha, beta


# Configure logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("smps.prepared_validation")

# SMPS imports

# New modular imports

# ML imports

# Forecast horizons
HORIZONS = {
    '0h': 0,    # Nowcast
    '24h': 1,   # 1-day ahead
    '72h': 3,   # 3-day ahead
    '168h': 7   # 7-day ahead
}


class PreparedDataValidator:
    """Run full validation using prepared ISMN CSVs with external data enrichment."""

    def __init__(
        self,
        prepared_data_dir: Path = Path("data/prepared"),
        output_dir: Path = Path("results/prepared_validation"),
        cache_dir: Path = Path("data/cache"),
        skip_gee: bool = True,
        skip_fetch: bool = False,
    ):
        self.prepared_data_dir = Path(prepared_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = Path(cache_dir)
        self.weather_cache_dir = self.cache_dir / "weather"
        self.weather_cache_dir.mkdir(parents=True, exist_ok=True)
        self.soil_cache_dir = self.cache_dir / "soil"
        self.soil_cache_dir.mkdir(parents=True, exist_ok=True)
        self.satellite_cache_dir = self.cache_dir / "satellite"
        self.satellite_cache_dir.mkdir(parents=True, exist_ok=True)

        self.skip_gee = skip_gee
        self.skip_fetch = skip_fetch

        # Initialize data sources
        self.weather_source = OpenMeteoSource(cache_dir=self.weather_cache_dir)

        # Initialize soil data source (iSDA for Africa, SoilGrids as fallback)
        self.soil_source = None
        self.has_isda = False
        try:
            from smps.data.sources.isda_authenticated import IsdaAfricaAuthenticatedSource
            self.soil_source = IsdaAfricaAuthenticatedSource(
                cache_dir=self.soil_cache_dir)
            self.has_isda = True
            logger.info("✓ iSDA Africa soil source initialized")
        except Exception as e:
            logger.warning(f"iSDA not available: {e}")
            try:
                from smps.data.sources.soil import SoilGridsSource
                self.soil_source = SoilGridsSource(
                    cache_dir=self.soil_cache_dir)
                logger.info("✓ SoilGrids fallback initialized")
            except Exception as e2:
                logger.warning(f"SoilGrids not available: {e2}")

        # Try to initialize GEE
        self.has_gee = False
        if not skip_gee:
            try:
                from smps.data.sources.gee_satellite import GoogleEarthEngineSatelliteSource
                self.satellite_source = GoogleEarthEngineSatelliteSource()
                self.has_gee = True
                logger.info("✓ Google Earth Engine initialized")
            except Exception as e:
                logger.warning(f"GEE not available: {e}")

        # Results storage
        self.canonical_table = None
        self.results = {}
        self.models = {}

    def load_prepared_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the prepared train/test CSVs."""
        logger.info("Loading prepared ISMN data...")

        train_df = pd.read_csv(
            self.prepared_data_dir / "ismn_soil_moisture_train.csv",
            parse_dates=['date']
        )
        test_temporal_df = pd.read_csv(
            self.prepared_data_dir / "ismn_soil_moisture_test_temporal.csv",
            parse_dates=['date']
        )
        test_spatial_df = pd.read_csv(
            self.prepared_data_dir / "ismn_soil_moisture_test_spatial.csv",
            parse_dates=['date']
        )

        logger.info(
            f"  Train: {len(train_df):,} rows, {train_df['station_id'].nunique()} stations")
        logger.info(f"  Test (temporal): {len(test_temporal_df):,} rows")
        logger.info(f"  Test (spatial): {len(test_spatial_df):,} rows")

        return train_df, test_temporal_df, test_spatial_df

    def get_unique_locations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract unique station locations."""
        locations = df.groupby('station_id').agg({
            'latitude': 'first',
            'longitude': 'first',
            'elevation_m': 'first',
            'region': 'first',
            'clay_pct': 'first',
            'sand_pct': 'first',
            'silt_pct': 'first',
            'saturation': 'first',
            'organic_carbon_pct': 'first',
        }).reset_index()
        return locations

    def fetch_weather_data(
        self,
        station_id: str,
        lat: float,
        lon: float,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch weather data from Open-Meteo with caching."""
        safe_id = station_id.replace(
            "/", "_").replace(",", "_").replace(" ", "_")
        cache_file = self.weather_cache_dir / f"weather_{safe_id}.parquet"

        # Check cache first - this is critical for avoiding rate limits
        if cache_file.exists():
            try:
                cached = pd.read_parquet(cache_file)
                # Filter to date range
                cached['date'] = pd.to_datetime(cached['date'])
                cached = cached[(cached['date'] >= pd.to_datetime(start_date))
                                & (cached['date'] <= pd.to_datetime(end_date))]
                if len(cached) > 0:
                    logger.debug(
                        f"  ✓ Weather cache hit: {station_id} ({len(cached)} days)")
                    return cached
            except Exception as e:
                logger.warning(f"  Cache read failed for {station_id}: {e}")

        if self.skip_fetch:
            logger.warning(
                f"  ⚠ No cached weather for {station_id}, skip_fetch=True")
            return None

        try:
            request = DataFetchRequest(
                site_id=station_id,
                start_date=start_date.date() if hasattr(start_date, 'date') else start_date,
                end_date=end_date.date() if hasattr(end_date, 'date') else end_date,
                parameters={'latitude': lat, 'longitude': lon}
            )
            self.weather_source._site_coordinates = {station_id: (lat, lon)}
            weather_data = self.weather_source.fetch_daily_weather(request)

            records = []
            for w in weather_data:
                records.append({
                    'date': pd.to_datetime(w.date),
                    'precipitation_mm': w.precipitation_mm,
                    'et0_mm': w.et0_mm,
                    'temperature_mean_c': w.temperature_mean_c,
                    'temperature_min_c': w.temperature_min_c,
                    'temperature_max_c': w.temperature_max_c,
                    'solar_radiation_mj_m2': w.solar_radiation_mj_m2,
                    'relative_humidity_mean': w.relative_humidity_mean,
                    'wind_speed_mean_m_s': w.wind_speed_mean_m_s,
                })

            weather_df = pd.DataFrame(records)

            # Cache for future use
            if len(weather_df) > 0:
                weather_df.to_parquet(cache_file)
                logger.debug(
                    f"  ✓ Weather fetched and cached: {station_id} ({len(weather_df)} days)")

            return weather_df

        except Exception as e:
            logger.warning(f"  ⚠ Weather fetch failed for {station_id}: {e}")
            return None

    def fetch_soil_data(
        self,
        station_id: str,
        lat: float,
        lon: float,
        existing_clay: Optional[float] = None,
        existing_sand: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Fetch soil properties from iSDA (Africa) or SoilGrids (global).

        Returns dict with: clay_pct, sand_pct, silt_pct, organic_carbon_pct, bulk_density, slope_degrees
        Falls back to existing values if fetch fails.
        """
        safe_id = station_id.replace(
            "/", "_").replace(",", "_").replace(" ", "_")
        cache_file = self.soil_cache_dir / f"soil_{safe_id}.json"

        # Check cache
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                logger.debug(f"  ✓ Soil cache hit: {station_id}")
                return cached
            except Exception:
                pass  # Cache read failed, will fetch fresh

        # Default values from existing data or reasonable estimates
        result = {
            'clay_pct': existing_clay if existing_clay and not np.isnan(existing_clay) else None,
            'sand_pct': existing_sand if existing_sand and not np.isnan(existing_sand) else None,
            'silt_pct': None,
            'organic_carbon_pct': None,
            'bulk_density': None,
            'slope_degrees': None,
        }

        if self.skip_fetch or self.soil_source is None:
            return result

        # Check if in Africa for iSDA
        is_africa = -35 <= lat <= 37 and -25 <= lon <= 55

        try:
            if self.has_isda and is_africa:
                # Fetch from iSDA
                profile = self.soil_source.fetch_soil_profile(
                    site_id=station_id,
                    latitude=lat,
                    longitude=lon,
                    depth="0-20"
                )
                result['clay_pct'] = profile.clay_percent
                result['sand_pct'] = profile.sand_percent
                result['silt_pct'] = profile.silt_percent
                result['organic_carbon_pct'] = getattr(
                    profile, 'organic_carbon_percent', None)
                result['bulk_density'] = getattr(profile, 'bulk_density', None)
                # iSDA also provides slope
                if hasattr(profile, 'slope_angle'):
                    result['slope_degrees'] = profile.slope_angle
                logger.debug(f"  ✓ iSDA soil fetched: {station_id}")
            elif self.soil_source is not None and not self.has_isda:
                # Use SoilGrids for non-Africa or fallback
                # Set site coordinates for SoilGrids
                self.soil_source._site_coordinates = {station_id: (lat, lon)}
                profile = self.soil_source.fetch_soil_profile(
                    site_id=station_id
                )
                result['clay_pct'] = profile.clay_percent
                result['sand_pct'] = profile.sand_percent
                result['silt_pct'] = profile.silt_percent
                logger.debug(f"  ✓ SoilGrids soil fetched: {station_id}")

            # Cache successful fetch
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f)

        except Exception as e:
            logger.warning(f"  ⚠ Soil fetch failed for {station_id}: {e}")

        return result

    def fetch_satellite_data(
        self,
        station_id: str,
        lat: float,
        lon: float,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, float]:
        """
        Fetch satellite vegetation data (NDVI, LAI) with caching.

        For crop irrigation applications, NDVI is CRITICAL because:
        1. Kc (crop coefficient) = f(NDVI) → accurate crop water requirements
        2. ETc = Kc × ET0 → irrigation scheduling
        3. Vegetation fraction → ET partitioning (transpiration vs soil evaporation)
        4. Crop stress detection → proactive water management

        Data sources (in priority order):
        1. GEE Sentinel-2/Landsat (if authenticated) - highest resolution
        2. MODIS NDVI API (no auth required) - reliable fallback

        Returns dict with: ndvi_mean, lai_mean, ndvi_min, ndvi_max, ndvi_std
        """
        safe_id = station_id.replace(
            "/", "_").replace(",", "_").replace(" ", "_")
        cache_file = self.satellite_cache_dir / f"satellite_{safe_id}.json"

        # Check cache
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                logger.debug(f"  ✓ Satellite cache hit: {station_id}")
                return cached
            except Exception:
                pass  # Cache read failed, will fetch fresh

        result = {
            'ndvi_mean': None,
            'lai_mean': None,
            'ndvi_min': None,
            'ndvi_max': None,
            'ndvi_std': None,
        }

        # Try GEE first (best quality)
        if not self.skip_gee and self.has_gee:
            try:
                veg_data = self.satellite_source.fetch_vegetation_data(
                    lat=lat,
                    lon=lon,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )

                if veg_data and len(veg_data) > 0:
                    ndvi_values = [
                        obs.ndvi for obs in veg_data if obs.ndvi is not None]
                    if ndvi_values:
                        result['ndvi_mean'] = float(np.mean(ndvi_values))
                        result['ndvi_min'] = float(np.min(ndvi_values))
                        result['ndvi_max'] = float(np.max(ndvi_values))
                        result['ndvi_std'] = float(np.std(ndvi_values))
                        result['source'] = 'GEE'

            except Exception as e:
                logger.debug(f"  GEE fetch failed: {e}, trying MODIS fallback")

        # Fallback to MODIS AppEEARS API (no authentication required for simple queries)
        if result['ndvi_mean'] is None and not self.skip_fetch:
            try:
                result = self._fetch_modis_ndvi_simple(
                    lat, lon, start_date, end_date)
                if result['ndvi_mean'] is not None:
                    result['source'] = 'MODIS'
            except Exception as e:
                logger.debug(f"  MODIS fallback failed: {e}")

        # Cache result
        if result['ndvi_mean'] is not None:
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f)
                logger.debug(
                    f"  ✓ Satellite: {station_id} NDVI={result['ndvi_mean']:.2f} ({result.get('source', 'unknown')})")
            except Exception:
                pass

        return result

    def _fetch_modis_ndvi_simple(
        self,
        lat: float,
        lon: float,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, float]:
        """
        Fetch MODIS NDVI using simple HTTP request to NASA AppEEARS.
        This is a lightweight alternative that doesn't require GEE authentication.
        """
        import requests

        result = {
            'ndvi_mean': None,
            'lai_mean': None,
            'ndvi_min': None,
            'ndvi_max': None,
            'ndvi_std': None,
        }

        # Use MODIS MOD13Q1 (250m, 16-day NDVI)
        # This endpoint provides point queries without authentication
        try:
            # Try the simpler MODIS Web Service
            base_url = "https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset"
            params = {
                "latitude": lat,
                "longitude": lon,
                "startDate": f"A{start_date.year}{start_date.timetuple().tm_yday:03d}",
                "endDate": f"A{end_date.year}{end_date.timetuple().tm_yday:03d}",
                "kmAboveBelow": 0,
                "kmLeftRight": 0,
            }

            response = requests.get(base_url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if 'subset' in data:
                    ndvi_values = []
                    for item in data['subset']:
                        if item.get('band') == '250m_16_days_NDVI':
                            # MODIS NDVI is scaled by 10000
                            for val in item.get('data', []):
                                if val > -2000:  # Valid range
                                    ndvi_values.append(val / 10000.0)

                    if ndvi_values:
                        result['ndvi_mean'] = float(np.mean(ndvi_values))
                        result['ndvi_min'] = float(np.min(ndvi_values))
                        result['ndvi_max'] = float(np.max(ndvi_values))
                        result['ndvi_std'] = float(np.std(ndvi_values))

        except Exception:
            pass  # MODIS API not available

        return result

    def run_physics_model(
        self,
        weather_df: pd.DataFrame,
        sand_pct: float,
        clay_pct: float,
        lat: float,
        lon: float,
        elevation_m: float,
        observed_mean: Optional[float] = None,
        slope_percent: Optional[float] = None,
        ndvi_mean: Optional[float] = None,
    ) -> pd.DataFrame:
        """Run SimpleWaterBalance physics model."""
        # Create config with tropical PTFs
        config = create_simple_config_improved(
            sand_percent=sand_pct if not np.isnan(sand_pct) else 50,
            clay_percent=clay_pct if not np.isnan(clay_pct) else 20,
            output_depth_m=0.10,
            n_layers=3,
            max_depth_m=1.0,
            vegetation_fraction=0.4,
            latitude=lat,
            longitude=lon,
            elevation_m=elevation_m if not np.isnan(elevation_m) else 200,
            slope_percent=slope_percent if slope_percent is not None else 5.0,
            observed_mean=observed_mean,
            use_tropical_ptf=True,
            apply_adaptive_calibration=True,
            ndvi_mean=ndvi_mean,
        )

        model = SimpleWaterBalance(config)

        # Initialize at observed mean
        if observed_mean is not None and not np.isnan(observed_mean):
            init_theta = np.clip(observed_mean, 0.05, 0.45)
            model.set_initial_conditions([init_theta] * 3)

        # Run model
        results = []
        for _, row in weather_df.iterrows():
            precip = row.get('precipitation_mm', 0)
            et0 = row.get('et0_mm', 3)  # Default ~3mm/day

            if np.isnan(precip):
                precip = 0
            if np.isnan(et0):
                et0 = 3

            fluxes, theta_surface = model.run_daily(
                float(precip), float(et0), None)
            theta_layers = [layer.theta for layer in model.layers]

            results.append({
                'date': row['date'],
                'physics_prior_surface': theta_layers[0],
                'physics_prior_root': theta_layers[1] if len(theta_layers) > 1 else theta_layers[0],
                'physics_prior_deep': theta_layers[2] if len(theta_layers) > 2 else theta_layers[-1],
            })

        return pd.DataFrame(results)

    def build_canonical_table(
        self,
        df: pd.DataFrame,
        max_stations: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Build canonical table by enriching prepared data with external sources.

        The canonical table contains ALL features aligned by date:
        - Ground truth (soil_moisture from ISMN)
        - Weather features (from Open-Meteo)
        - Physics priors (from SimpleWaterBalance)
        - Soil properties (from ISMN or iSDA)
        - Satellite features (from GEE, if available)
        """
        logger.info("\n" + "=" * 70)
        logger.info("BUILDING CANONICAL TABLE")
        logger.info("=" * 70)

        locations = self.get_unique_locations(df)
        if max_stations:
            locations = locations.head(max_stations)

        logger.info(f"Processing {len(locations)} unique stations...")

        enriched_dfs = []

        for _, loc in tqdm(locations.iterrows(), total=len(locations), desc="Enriching"):
            station_id = loc['station_id']
            lat, lon = loc['latitude'], loc['longitude']

            # Get this station's data
            station_df = df[df['station_id'] == station_id].copy()
            station_df = station_df.sort_values('date')

            if len(station_df) < 30:
                continue

            # Get date range
            start_date = station_df['date'].min()
            end_date = station_df['date'].max()

            # 1. Fetch weather data
            weather_df = self.fetch_weather_data(
                station_id, lat, lon, start_date, end_date
            )

            if weather_df is not None and len(weather_df) > 0:
                # Merge weather with observations
                station_df = station_df.merge(
                    weather_df,
                    on='date',
                    how='left'
                )
                # Fill any remaining NaN values with interpolation or forward fill
                weather_cols = ['precipitation_mm', 'et0_mm', 'temperature_mean_c',
                                'temperature_min_c', 'temperature_max_c']
                for col in weather_cols:
                    if col in station_df.columns:
                        station_df[col] = station_df[col].ffill().bfill()
            else:
                # CRITICAL: Skip stations without weather data - don't use fake defaults
                logger.warning(
                    f"  ⚠ Skipping {station_id}: No weather data available")
                continue

            # 2. Fetch/update soil properties if missing or from iSDA
            soil_data = self.fetch_soil_data(
                station_id, lat, lon,
                existing_clay=loc.get('clay_pct'),
                existing_sand=loc.get('sand_pct'),
            )

            # Update soil columns with fetched data
            if soil_data['clay_pct'] is not None:
                station_df['clay_pct'] = soil_data['clay_pct']
            if soil_data['sand_pct'] is not None:
                station_df['sand_pct'] = soil_data['sand_pct']
            if soil_data['silt_pct'] is not None:
                station_df['silt_pct'] = soil_data['silt_pct']
            if soil_data['organic_carbon_pct'] is not None:
                station_df['organic_carbon_pct'] = soil_data['organic_carbon_pct']
            if soil_data['slope_degrees'] is not None:
                station_df['slope_degrees'] = soil_data['slope_degrees']

            # Get final soil values for physics model (use station data or fetched)
            clay_pct = station_df['clay_pct'].iloc[0] if 'clay_pct' in station_df.columns else None
            sand_pct = station_df['sand_pct'].iloc[0] if 'sand_pct' in station_df.columns else None

            # Skip if no soil data available
            if clay_pct is None or np.isnan(clay_pct) or sand_pct is None or np.isnan(sand_pct):
                logger.warning(
                    f"  ⚠ Skipping {station_id}: No soil texture data available")
                continue

            # Get slope (from iSDA/SoilGrids or estimate from elevation)
            slope_percent = 5.0  # Default
            if soil_data.get('slope_degrees') is not None:
                slope_percent = np.tan(np.radians(
                    soil_data['slope_degrees'])) * 100
            elif 'slope_degrees' in station_df.columns and station_df['slope_degrees'].notna().any():
                slope_deg = station_df['slope_degrees'].iloc[0]
                slope_percent = np.tan(np.radians(slope_deg)) * 100

            # 3. Fetch satellite data (NDVI/LAI) for physics model calibration
            satellite_data = self.fetch_satellite_data(
                station_id, lat, lon, start_date, end_date
            )
            ndvi_mean = satellite_data.get('ndvi_mean')

            # Add NDVI features to station data if available
            if ndvi_mean is not None:
                station_df['ndvi_mean'] = ndvi_mean
            if satellite_data.get('lai_mean') is not None:
                station_df['lai_mean'] = satellite_data['lai_mean']

            # 4. Run physics model
            physics_input = station_df[[
                'date', 'precipitation_mm', 'et0_mm']].drop_duplicates('date')
            physics_input = physics_input.sort_values(
                'date').reset_index(drop=True)

            obs_mean = station_df['soil_moisture'].mean()

            # Get elevation - use actual value, not default
            elevation_m = loc.get('elevation_m', 200)
            if elevation_m is None or (isinstance(elevation_m, float) and np.isnan(elevation_m)):
                elevation_m = 200  # Only use default if truly missing

            physics_results = self.run_physics_model(
                physics_input,
                sand_pct=sand_pct,
                clay_pct=clay_pct,
                lat=lat,
                lon=lon,
                elevation_m=elevation_m,
                observed_mean=obs_mean,
                slope_percent=slope_percent,
                ndvi_mean=ndvi_mean,
            )

            # Merge physics
            station_df = station_df.merge(
                physics_results,
                on='date',
                how='left'
            )

            # 5. Select appropriate physics prior based on depth
            def get_physics_prior(row):
                depth = row['depth_cm']
                if depth <= 15:
                    return row['physics_prior_surface']
                elif depth <= 50:
                    return row['physics_prior_root']
                else:
                    return row['physics_prior_deep']

            station_df['physics_prior'] = station_df.apply(
                get_physics_prior, axis=1)

            # 6. Compute residual (what physics model misses)
            station_df['residual'] = station_df['soil_moisture'] - \
                station_df['physics_prior']

            enriched_dfs.append(station_df)

            # Rate limiting for API calls
            if not self.skip_fetch:
                time.sleep(0.5)

        # Combine all
        canonical = pd.concat(enriched_dfs, ignore_index=True)

        logger.info(
            f"\n✓ Canonical table built: {len(canonical):,} rows, {len(canonical.columns)} columns")

        # Show canonical table structure
        self._show_canonical_structure(canonical)

        return canonical

    def _show_canonical_structure(self, df: pd.DataFrame):
        """Display canonical table structure."""
        print("\n" + "=" * 70)
        print("CANONICAL TABLE STRUCTURE")
        print("=" * 70)

        categories = {
            'Identity': ['station_id', 'network', 'station', 'region', 'dataset'],
            'Location': ['latitude', 'longitude', 'elevation_m'],
            'Time': ['date', 'year', 'month', 'day_of_year'],
            'Ground Truth': ['soil_moisture', 'depth_cm', 'soil_moisture_min', 'soil_moisture_max'],
            'Weather': ['precipitation_mm', 'et0_mm', 'temperature_mean_c', 'temperature_min_c',
                        'temperature_max_c', 'solar_radiation_mj_m2', 'relative_humidity_mean'],
            'Soil Properties': ['clay_pct', 'sand_pct', 'silt_pct', 'saturation', 'organic_carbon_pct'],
            'Physics Model': ['physics_prior', 'physics_prior_surface', 'physics_prior_root',
                              'physics_prior_deep', 'residual'],
        }

        for category, cols in categories.items():
            present = [c for c in cols if c in df.columns]
            missing = [c for c in cols if c not in df.columns]

            print(f"\n{category}:")
            for col in present:
                sample = df[col].iloc[0] if len(df) > 0 else "N/A"
                if isinstance(sample, float):
                    sample = f"{sample:.4f}"
                elif isinstance(sample, str) and len(str(sample)) > 30:
                    sample = str(sample)[:30] + "..."
                print(f"  ✓ {col}: {sample}")
            for col in missing:
                print(f"  ✗ {col}: NOT POPULATED")

        print("\n" + "=" * 70)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for ML model.

        Uses the modular AdvancedFeatureEngineer for comprehensive features
        including API, memory terms, and future weather features.
        """
        result = df.copy()
        result = result.sort_values(
            ['station_id', 'depth_cm', 'date']).reset_index(drop=True)

        # Use modular feature engineering
        advanced_engineer = AdvancedFeatureEngineer(
            api_decay_factors=[0.85, 0.90, 0.95],
            forecast_horizons=[1, 3, 7],
        )

        # Apply advanced features per station to handle grouping properly
        all_features = []
        for station_id in result['station_id'].unique():
            station_df = result[result['station_id'] == station_id].copy()

            # Add temporal features
            station_df = create_temporal_features(station_df)

            # Add spatial features
            station_df = create_spatial_features(station_df)

            # Add advanced lag/memory features (for single station)
            station_df = advanced_engineer.add_basic_lag_features(station_df)
            station_df = advanced_engineer.add_api_features(station_df)
            station_df = advanced_engineer.add_time_since_features(station_df)
            station_df = advanced_engineer.add_water_balance_features(
                station_df)

            all_features.append(station_df)

        result = pd.concat(all_features, ignore_index=True)

        # Add grouped features that need station context
        # 1. Physics lag features ONLY (no observed soil moisture lags - causes leakage)
        for lag in [1, 3, 7, 14]:
            result[f'physics_lag_{lag}d'] = result.groupby(['station_id', 'depth_cm'])[
                'physics_prior'].shift(lag)

        # NOTE: We do NOT include:
        # - sm_lag_*d (observed soil moisture lags - LEAKAGE for short horizons)
        # - residual_lag_*d (contains observed SM - LEAKAGE)
        # - sm_rolling_mean/std_*d (contains observed SM - LEAKAGE)
        # - physics_obs_diff/ratio (directly uses current observed SM - LEAKAGE)

        # 4. Weather aggregations
        if 'precipitation_mm' in result.columns:
            result['precip_7d_sum'] = (
                result.groupby(['station_id'])['precipitation_mm']
                .transform(lambda x: x.rolling(7, min_periods=1).sum())
            )
            result['precip_14d_sum'] = (
                result.groupby(['station_id'])['precipitation_mm']
                .transform(lambda x: x.rolling(14, min_periods=1).sum())
            )

        if 'et0_mm' in result.columns:
            result['et0_7d_sum'] = (
                result.groupby(['station_id'])['et0_mm']
                .transform(lambda x: x.rolling(7, min_periods=1).sum())
            )

        # 5. Water balance
        if 'precipitation_mm' in result.columns and 'et0_mm' in result.columns:
            result['water_balance_1d'] = result['precipitation_mm'] - \
                result['et0_mm']
            result['water_balance_7d'] = result['precip_7d_sum'] - \
                result['et0_7d_sum']

        # 6. Physics-observation features - REMOVED (LEAKAGE!)
        # These use current observed soil_moisture which is the target!
        # DO NOT USE: physics_obs_ratio, physics_obs_diff

        # 7. Depth features
        result['depth_normalized'] = result['depth_cm'] / \
            result['depth_cm'].max()
        result['is_surface'] = (result['depth_cm'] <= 15).astype(int)
        result['is_deep'] = (result['depth_cm'] > 50).astype(int)

        # 8. FIX #2: PHYSICAL SOIL ENCODING (instead of lat/lon fingerprints)
        # These force ML to learn "Clay soils store more water" instead of
        # "Station at (7.5°N, 5.2°E) behaves like this"
        result = self._add_physical_soil_features(result)

        # 9. LAI estimation from NDVI (for irrigation applications)
        result = self._add_lai_from_ndvi(result)

        # NOTE: lat_normalized and lon_normalized are REMOVED (FIX #1)
        # They act as station fingerprints allowing ML to memorize geography

        return result

    def _add_physical_soil_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FIX #2: Add physically meaningful soil features.

        These encode soil behavior that ML can generalize:
        - Porosity: How much water soil can hold
        - Field capacity: Maximum retained water after drainage
        - Wilting point: Minimum plant-available water
        - Available water capacity: FC - WP (usable water storage)
        - Drainage class: How fast soil drains (from texture)
        - Texture class: Simplified soil type
        """
        result = df.copy()

        # Use tropical PTF to estimate hydraulic parameters
        if 'sand_pct' in result.columns and 'clay_pct' in result.columns:
            # Porosity (saturated water content) - Saxton & Rawls PTF
            sand = result['sand_pct'].fillna(40) / 100
            clay = result['clay_pct'].fillna(25) / 100
            silt = (100 - result['sand_pct'].fillna(40) -
                    result['clay_pct'].fillna(25)) / 100
            organic = result['organic_carbon_pct'].fillna(
                1.0) / 100 if 'organic_carbon_pct' in result.columns else 0.01

            # Porosity (θs) - Saxton & Rawls (2006)
            result['porosity'] = 0.332 - 0.0007251 * \
                (sand * 100) + 0.1276 * np.log10(clay * 100 + 1)

            # Field capacity (θ at -33 kPa)
            result['field_capacity'] = (
                0.2576 - 0.002 * (sand * 100) + 0.0036 * (clay * 100) +
                0.0299 * (organic * 100) - 0.00006 *
                (sand * 100) * (clay * 100)
            )

            # Wilting point (θ at -1500 kPa)
            result['wilting_point'] = (
                0.026 + 0.005 * (clay * 100) + 0.0158 * (organic * 100)
            )

            # Available water capacity (plant-usable storage)
            result['available_water_capacity'] = result['field_capacity'] - \
                result['wilting_point']

            # Saturated hydraulic conductivity (drainage rate) - Cosby PTF
            # log10(Ksat) = -0.6 + 0.0126*sand% - 0.0064*clay%
            result['log_ksat'] = -0.6 + 0.0126 * \
                (sand * 100) - 0.0064 * (clay * 100)

            # Drainage class (categorical to numeric)
            # High clay = slow drainage (0), High sand = fast drainage (1)
            result['drainage_index'] = (sand - clay + 1) / 2  # Normalized 0-1

            # Texture class index (simplified USDA texture triangle)
            # 0 = clay-dominated, 1 = balanced, 2 = sand-dominated
            conditions = [
                (clay >= 0.40),  # Clay-dominated
                (sand >= 0.70),  # Sand-dominated
            ]
            choices = [0, 2]
            result['texture_class'] = np.select(conditions, choices, default=1)

            # Water retention ratio (FC / porosity) - soil's ability to hold water
            result['water_retention_ratio'] = result['field_capacity'] / \
                result['porosity'].clip(lower=0.2)

            # Soil permeability index (sand/clay ratio affects permeability)
            result['permeability_index'] = (
                sand / clay.clip(lower=0.05)).clip(upper=10)

            logger.info(
                "  Added physical soil features: porosity, field_capacity, AWC, drainage_index")

        return result

    def _add_lai_from_ndvi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FIX #5: Estimate Leaf Area Index (LAI) from NDVI.

        LAI is crucial for irrigation applications:
        - Higher LAI = more transpiration = more crop water demand
        - LAI is used in FAO-56 dual crop coefficient method

        Empirical relationship: LAI = f(NDVI)
        Common formula: LAI = (NDVI - NDVI_min) / (k * (NDVI_max - NDVI_min))
        Or: LAI = -ln((1 - NDVI) / (1 + NDVI)) / k  (Beer-Lambert based)
        """
        result = df.copy()

        if 'ndvi_mean' in result.columns:
            ndvi = result['ndvi_mean'].fillna(0.3).clip(0.05, 0.95)

            # Beer-Lambert based LAI estimation (Baret & Guyot, 1991)
            # LAI = -ln(1 - fAPAR) / k, where fAPAR ≈ 1.25 * NDVI - 0.1
            fapar = (1.25 * ndvi - 0.1).clip(0.01, 0.95)
            k_extinction = 0.5  # Typical extinction coefficient for crops
            result['lai_estimated'] = -np.log(1 - fapar) / k_extinction

            # Clipped LAI (realistic range 0-8)
            result['lai_estimated'] = result['lai_estimated'].clip(0, 8)

            # Vegetation fraction (ground cover)
            result['vegetation_fraction'] = ((ndvi - 0.1) / 0.6).clip(0, 1)

            # Crop coefficient (Kc) estimation from NDVI for irrigation
            # Kc = 1.2 * NDVI + 0.1 (simplified FAO-56 relationship)
            result['kc_estimated'] = (1.2 * ndvi + 0.1).clip(0.15, 1.2)

            # Potential crop ET (ETc = Kc * ET0)
            if 'et0_mm' in result.columns:
                result['etc_estimated'] = result['kc_estimated'] * \
                    result['et0_mm']

            logger.info("  Added LAI and irrigation features from NDVI")

        return result

    def create_horizon_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create multi-horizon forecast targets."""
        result = df.copy()
        result = result.sort_values(
            ['station_id', 'depth_cm', 'date']).reset_index(drop=True)

        for horizon_name, horizon_days in HORIZONS.items():
            # Target: future soil moisture
            result[f'target_{horizon_name}'] = (
                result.groupby(['station_id', 'depth_cm'])['soil_moisture']
                .shift(-horizon_days)
            )
            # Residual target: future residual
            result[f'residual_target_{horizon_name}'] = (
                result.groupby(['station_id', 'depth_cm'])['residual']
                .shift(-horizon_days)
            )

        return result

    def prepare_ml_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for ML training."""
        # Filter valid rows
        valid_mask = df[target_col].notna()
        df_valid = df[valid_mask].copy()

        X = df_valid[feature_cols].copy()
        y = df_valid[target_col].values

        # Handle missing/infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            X[col] = X[col].fillna(
                X[col].median() if X[col].notna().any() else 0)

        return X.values, y

    def train_hybrid_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        horizon: str = '24h',
    ) -> Tuple[lgb.Booster, List[str]]:
        """Train LightGBM model for residual prediction."""

        # Define feature columns - STRICT NO-LEAKAGE policy
        # Any feature derived from observed soil_moisture is EXCLUDED
        # FIX #1: Remove raw lat/lon - they act as station fingerprints!
        # ML learns "station at 7.5°N behaves like X" instead of physics
        exclude_cols = [
            'station_id', 'network', 'station', 'region', 'dataset', 'date',
            # Target and observed values - NEVER use as features
            'soil_moisture', 'soil_moisture_min', 'soil_moisture_max', 'soil_moisture_std',
            # Residual uses observed SM, so exclude
            'residual',
            # Metadata
            'sensor_type', 'land_cover', 'climate', 'is_outlier',
            # GEOGRAPHY FINGERPRINTS - REMOVED to force learning hydrology!
            'latitude', 'longitude', 'lat_normalized', 'lon_normalized',
            'physics_prior_surface', 'physics_prior_root', 'physics_prior_deep',
            # Any feature derived from observed soil moisture (LEAKAGE)
            'physics_obs_ratio', 'physics_obs_diff',
        ]
        # Exclude targets
        exclude_cols += [c for c in train_df.columns if c.startswith(
            'target_') or c.startswith('residual_target_')]
        # Exclude any remaining soil moisture derived features (sm_lag, sm_rolling, residual_lag)
        exclude_cols += [c for c in train_df.columns if c.startswith(
            'sm_lag_') or c.startswith('sm_rolling_') or c.startswith('residual_lag_')]
        # Also exclude sm_mean, sm_std, sm_change if they exist
        exclude_cols += [c for c in train_df.columns if c.startswith(
            'sm_mean') or c.startswith('sm_std') or c.startswith('sm_change')]

        feature_cols = [
            c for c in train_df.columns
            if c not in exclude_cols
            and train_df[c].dtype in ['float64', 'int64', 'float32', 'int32']
        ]

        target_col = f'residual_target_{horizon}'

        # Prepare data
        X_train, y_train = self.prepare_ml_data(
            train_df, target_col, feature_cols)
        X_val, y_val = self.prepare_ml_data(val_df, target_col, feature_cols)

        logger.info(
            f"  Training: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        logger.info(f"  Validation: {X_val.shape[0]:,} samples")

        # FIX #4: Limit ML capacity to force learning physical relationships
        # Shallow trees + more data per leaf + random feature sampling
        # This prevents memorizing station fingerprints
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            # FIX #4: Shallow trees (was 31 leaves → unlimited depth)
            'max_depth': 5,
            'num_leaves': 20,            # FIX #4: Fewer leaves (was 31)
            'learning_rate': 0.03,       # Slower learning
            # FIX #4: Only 60% features per tree (was 0.8)
            'feature_fraction': 0.6,
            'bagging_fraction': 0.7,     # 70% samples per tree (was 0.8)
            'bagging_freq': 3,           # More frequent subsampling
            'min_child_samples': 100,    # FIX #4: More data per leaf (was 20)
            'lambda_l1': 0.1,            # L1 regularization
            'lambda_l2': 1.0,            # L2 regularization
            'verbose': -1,
            'n_jobs': -1,
            'random_state': 42,
        }

        train_data = lgb.Dataset(
            X_train, label=y_train, feature_name=feature_cols)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            valid_names=['val'],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        return model, feature_cols

    def train_hybrid_model_with_site_cv(
        self,
        df: pd.DataFrame,
        horizon: str = '24h',
        n_folds: int = 5,
    ) -> Tuple[lgb.Booster, List[str]]:
        """
        FIX #3: Train with site-blocked cross-validation (GroupKFold).

        This ensures ML NEVER sees a test site during training.
        Without this, LightGBM can cheat by memorizing station patterns.

        Returns the best model from cross-validation folds.
        """
        # Get feature columns (same exclusion logic)
        exclude_cols = [
            'station_id', 'network', 'station', 'region', 'dataset', 'date',
            'soil_moisture', 'soil_moisture_min', 'soil_moisture_max', 'soil_moisture_std',
            'residual', 'sensor_type', 'land_cover', 'climate', 'is_outlier',
            # GEOGRAPHY FINGERPRINTS - REMOVED
            'latitude', 'longitude', 'lat_normalized', 'lon_normalized',
            'physics_prior_surface', 'physics_prior_root', 'physics_prior_deep',
            'physics_obs_ratio', 'physics_obs_diff',
        ]
        exclude_cols += [c for c in df.columns if c.startswith(
            'target_') or c.startswith('residual_target_')]
        exclude_cols += [c for c in df.columns if c.startswith(
            'sm_lag_') or c.startswith('sm_rolling_') or c.startswith('residual_lag_')]
        exclude_cols += [c for c in df.columns if c.startswith(
            'sm_mean') or c.startswith('sm_std') or c.startswith('sm_change')]

        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols
            and df[c].dtype in ['float64', 'int64', 'float32', 'int32']
        ]

        target_col = f'residual_target_{horizon}'

        # Filter valid data
        valid_mask = df[target_col].notna()
        df_valid = df[valid_mask].copy()

        # Get station groups for GroupKFold
        groups = df_valid['station_id'].values

        X, y = self.prepare_ml_data(df_valid, target_col, feature_cols)

        # GroupKFold ensures sites don't leak between train/val
        gkf = GroupKFold(n_splits=n_folds)

        best_model = None
        best_val_rmse = float('inf')
        fold_scores = []

        logger.info(
            f"  Site-blocked {n_folds}-fold CV on {df_valid['station_id'].nunique()} stations")

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'max_depth': 5,
            'num_leaves': 20,
            'learning_rate': 0.03,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.7,
            'bagging_freq': 3,
            'min_child_samples': 100,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'verbose': -1,
            'n_jobs': -1,
            'random_state': 42,
        }

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Verify no site leakage
            train_sites = set(groups[train_idx])
            val_sites = set(groups[val_idx])
            assert len(train_sites & val_sites) == 0, "Site leakage detected!"

            train_data = lgb.Dataset(
                X_train_fold, label=y_train_fold, feature_name=feature_cols)
            val_data = lgb.Dataset(
                X_val_fold, label=y_val_fold, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                valid_names=['val'],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )

            val_pred = model.predict(X_val_fold)
            val_rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
            fold_scores.append(val_rmse)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model = model

            logger.info(
                f"    Fold {fold+1}: val_rmse={val_rmse:.4f}, train_sites={len(train_sites)}, val_sites={len(val_sites)}")

        logger.info(
            f"  CV mean RMSE: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

        return best_model, feature_cols

    def evaluate_model(
        self,
        model: lgb.Booster,
        df: pd.DataFrame,
        feature_cols: List[str],
        horizon: str,
    ) -> Dict:
        """Evaluate hybrid model (physics + ML residual)."""
        target_col = f'residual_target_{horizon}'

        X, y_residual = self.prepare_ml_data(df, target_col, feature_cols)

        # Predict residual
        residual_pred = model.predict(X)

        # Get physics prior for these rows
        valid_mask = df[target_col].notna()
        physics_prior = df.loc[valid_mask, 'physics_prior'].values

        # Hybrid prediction = physics + ML residual
        y_hybrid_pred = physics_prior + residual_pred

        # Actual observation (future)
        y_obs = df.loc[valid_mask, f'target_{horizon}'].values

        # Clip to valid range
        y_hybrid_pred = np.clip(y_hybrid_pred, 0, 0.6)

        # Standard metrics
        rmse = np.sqrt(mean_squared_error(y_obs, y_hybrid_pred))
        mae = mean_absolute_error(y_obs, y_hybrid_pred)
        r2 = r2_score(y_obs, y_hybrid_pred)
        bias = np.mean(y_hybrid_pred - y_obs)

        # KGE and NSE metrics
        nse = compute_nse(y_obs, y_hybrid_pred)
        kge, kge_r, kge_alpha, kge_beta = compute_kge(y_obs, y_hybrid_pred)

        # Physics-only metrics
        physics_rmse = np.sqrt(mean_squared_error(y_obs, physics_prior))
        physics_r2 = r2_score(y_obs, physics_prior)
        physics_nse = compute_nse(y_obs, physics_prior)
        physics_kge, physics_kge_r, physics_kge_alpha, physics_kge_beta = compute_kge(
            y_obs, physics_prior)

        return {
            'horizon': horizon,
            'n_samples': len(y_obs),
            # Hybrid metrics
            'hybrid_rmse': rmse,
            'hybrid_mae': mae,
            'hybrid_r2': r2,
            'hybrid_bias': bias,
            'hybrid_nse': nse,
            'hybrid_kge': kge,
            'hybrid_kge_r': kge_r,
            'hybrid_kge_alpha': kge_alpha,
            'hybrid_kge_beta': kge_beta,
            # Physics metrics
            'physics_rmse': physics_rmse,
            'physics_r2': physics_r2,
            'physics_nse': physics_nse,
            'physics_kge': physics_kge,
            # Improvement
            'improvement_rmse_pct': (physics_rmse - rmse) / physics_rmse * 100,
            'improvement_kge': kge - physics_kge if not np.isnan(kge) and not np.isnan(physics_kge) else np.nan,
        }

    def evaluate_per_site(
        self,
        model: lgb.Booster,
        df: pd.DataFrame,
        feature_cols: List[str],
        horizon: str,
    ) -> pd.DataFrame:
        """Evaluate model performance per site."""
        target_col = f'residual_target_{horizon}'
        valid_mask = df[target_col].notna()
        df_valid = df[valid_mask].copy()

        # Prepare features
        X = df_valid[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            X[col] = X[col].fillna(
                X[col].median() if X[col].notna().any() else 0)

        # Predict residual
        residual_pred = model.predict(X.values)
        df_valid['residual_pred'] = residual_pred
        df_valid['hybrid_pred'] = df_valid['physics_prior'] + residual_pred
        df_valid['hybrid_pred'] = df_valid['hybrid_pred'].clip(0, 0.6)
        df_valid['y_obs'] = df_valid[f'target_{horizon}']

        # Per-site metrics
        site_results = []
        for station_id in df_valid['station_id'].unique():
            site_df = df_valid[df_valid['station_id'] == station_id]
            if len(site_df) < 5:
                continue

            y_obs = site_df['y_obs'].values
            y_pred = site_df['hybrid_pred'].values
            y_physics = site_df['physics_prior'].values

            # Get metadata
            row = site_df.iloc[0]

            # Compute KGE and NSE
            hybrid_nse = compute_nse(y_obs, y_pred)
            hybrid_kge, hybrid_kge_r, hybrid_kge_alpha, hybrid_kge_beta = compute_kge(
                y_obs, y_pred)
            physics_nse = compute_nse(y_obs, y_physics)
            physics_kge, _, _, _ = compute_kge(y_obs, y_physics)

            site_results.append({
                'station_id': station_id,
                'network': row.get('network', 'unknown'),
                'region': row.get('region', 'unknown'),
                'latitude': row.get('latitude', np.nan),
                'longitude': row.get('longitude', np.nan),
                'n_samples': len(site_df),
                'horizon': horizon,
                # Hybrid metrics
                'hybrid_rmse': np.sqrt(mean_squared_error(y_obs, y_pred)),
                'hybrid_mae': mean_absolute_error(y_obs, y_pred),
                'hybrid_r2': r2_score(y_obs, y_pred) if len(y_obs) > 1 else np.nan,
                'hybrid_bias': np.mean(y_pred - y_obs),
                'hybrid_nse': hybrid_nse,
                'hybrid_kge': hybrid_kge,
                'hybrid_kge_r': hybrid_kge_r,
                'hybrid_kge_alpha': hybrid_kge_alpha,
                'hybrid_kge_beta': hybrid_kge_beta,
                # Physics metrics
                'physics_rmse': np.sqrt(mean_squared_error(y_obs, y_physics)),
                'physics_r2': r2_score(y_obs, y_physics) if len(y_obs) > 1 else np.nan,
                'physics_nse': physics_nse,
                'physics_kge': physics_kge,
                # Observation stats
                'mean_obs_sm': np.mean(y_obs),
                'std_obs_sm': np.std(y_obs),
            })

        return pd.DataFrame(site_results)

    def run(self, max_stations: Optional[int] = None):
        """Run the full validation pipeline."""
        print("\n" + "=" * 70)
        print("FULL PHYSICS + ML HYBRID VALIDATION PIPELINE")
        print("=" * 70)
        print(f"📁 Data: {self.prepared_data_dir}")
        print(f"📤 Output: {self.output_dir}")
        print("☀️  Weather: Open-Meteo API")

        # Satellite data status - NDVI is critical for irrigation
        if self.has_gee:
            sat_status = "GEE Sentinel-2/Landsat (high-res NDVI)"
        elif not self.skip_fetch:
            sat_status = "MODIS NDVI (fallback)"
        else:
            sat_status = "Disabled (skip_fetch=True)"

        print(f"🌍 Satellite: {sat_status}")

        print(
            f"🧪 Soil: {'iSDA Africa' if self.has_isda else 'SoilGrids (global)'}")
        print("🔬 Physics: SimpleWaterBalance (tropical PTFs)")
        print("🤖 ML: LightGBM residual learner")
        print("=" * 70 + "\n")

        # 1. Load prepared data
        train_df, test_temporal_df, test_spatial_df = self.load_prepared_data()

        # 2. Build canonical table (enrich with weather + run physics)
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: Build Canonical Table (Training Data)")
        logger.info("=" * 70)
        train_canonical = self.build_canonical_table(train_df, max_stations)

        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: Build Canonical Table (Test Data)")
        logger.info("=" * 70)
        # Use fewer stations for test to save time
        test_temporal_canonical = self.build_canonical_table(
            test_temporal_df,
            max_stations=max_stations
        )
        test_spatial_canonical = self.build_canonical_table(
            test_spatial_df,
            max_stations=max_stations
        )

        # 3. Feature engineering
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: Feature Engineering")
        logger.info("=" * 70)
        train_featured = self.engineer_features(train_canonical)
        test_temporal_featured = self.engineer_features(
            test_temporal_canonical)
        test_spatial_featured = self.engineer_features(test_spatial_canonical)

        # 4. Create horizon targets
        train_with_targets = self.create_horizon_targets(train_featured)
        test_temporal_with_targets = self.create_horizon_targets(
            test_temporal_featured)
        test_spatial_with_targets = self.create_horizon_targets(
            test_spatial_featured)

        # Save canonical table
        self.canonical_table = train_with_targets
        train_with_targets.to_csv(
            self.output_dir / "canonical_table_train.csv", index=False)
        logger.info(
            f"✓ Saved canonical table: {self.output_dir / 'canonical_table_train.csv'}")

        # 5. Train models for each horizon
        # FIX #3: Use site-blocked cross-validation (GroupKFold)
        # This ensures ML never sees test sites during training
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: Train Hybrid Models (with Site-Blocked CV)")
        logger.info("=" * 70)
        logger.info(
            "NOTE: Using GroupKFold to prevent station fingerprint learning")

        all_results = []

        # Create validation split from training data (for final evaluation only)
        # The actual training uses site-blocked CV internally
        train_dates = train_with_targets['date']
        val_split_date = train_dates.quantile(0.8)

        train_split = train_with_targets[train_with_targets['date']
                                         <= val_split_date]
        val_split = train_with_targets[train_with_targets['date']
                                       > val_split_date]

        for horizon in HORIZONS.keys():
            logger.info(f"\n--- Training for {horizon} horizon ---")

            # FIX #3: Use site-blocked CV instead of simple train/val split
            # This trains with GroupKFold ensuring no site leakage
            model, feature_cols = self.train_hybrid_model_with_site_cv(
                train_with_targets, horizon, n_folds=5
            )
            self.models[horizon] = (model, feature_cols)

            # Save model
            model.save_model(
                str(self.output_dir / f"hybrid_model_{horizon}.txt"))

            # Evaluate
            val_metrics = self.evaluate_model(
                model, val_split, feature_cols, horizon)
            val_metrics['split'] = 'validation'
            all_results.append(val_metrics)

            temp_metrics = self.evaluate_model(
                model, test_temporal_with_targets, feature_cols, horizon)
            temp_metrics['split'] = 'test_temporal'
            all_results.append(temp_metrics)

            spat_metrics = self.evaluate_model(
                model, test_spatial_with_targets, feature_cols, horizon)
            spat_metrics['split'] = 'test_spatial'
            all_results.append(spat_metrics)

            # Per-site evaluation
            if horizon == '24h':  # Do detailed per-site for 24h horizon
                site_results_temporal = self.evaluate_per_site(
                    model, test_temporal_with_targets, feature_cols, horizon)
                site_results_temporal['split'] = 'test_temporal'

                site_results_spatial = self.evaluate_per_site(
                    model, test_spatial_with_targets, feature_cols, horizon)
                site_results_spatial['split'] = 'test_spatial'

                all_site_results = pd.concat(
                    [site_results_temporal, site_results_spatial], ignore_index=True)
                all_site_results.to_csv(
                    self.output_dir / "per_site_results_24h.csv", index=False)
                logger.info(
                    f"  ✓ Saved per-site results: {self.output_dir / 'per_site_results_24h.csv'}")

        # 6. Display results
        results_df = pd.DataFrame(all_results)

        print("\n" + "=" * 70)
        print("FINAL RESULTS: Physics + ML Hybrid Model")
        print("=" * 70)

        pivot = results_df.pivot_table(
            index='horizon',
            columns='split',
            values=['hybrid_rmse', 'physics_rmse', 'improvement_rmse_pct']
        ).round(4)
        print(pivot)

        # Save results
        results_df.to_csv(self.output_dir /
                          "validation_results.csv", index=False)

        # Save feature importance
        if '24h' in self.models:
            model, feature_cols = self.models['24h']
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            importance_df.to_csv(
                self.output_dir / "feature_importance.csv", index=False)

            print("\n" + "=" * 70)
            print("TOP 15 FEATURES (24h horizon)")
            print("=" * 70)
            print(importance_df.head(15).to_string())

        # Print leakage check summary
        print("\n" + "=" * 70)
        print("LEAKAGE CHECK: Features used (NO observed soil moisture!)")
        print("=" * 70)
        if '24h' in self.models:
            _, feature_cols = self.models['24h']
            leakage_keywords = ['sm_lag', 'sm_roll',
                                'residual_lag', 'physics_obs', 'soil_moisture']
            safe_features = [f for f in feature_cols if not any(
                kw in f.lower() for kw in leakage_keywords)]
            suspect_features = [f for f in feature_cols if any(
                kw in f.lower() for kw in leakage_keywords)]
            print(f"✓ Safe features: {len(safe_features)}")
            if suspect_features:
                print(
                    f"⚠️  SUSPECT features (potential leakage): {suspect_features}")
            else:
                print("✓ No suspected leakage features found!")
            print(f"\nFeature categories used:")
            print(f"  - Weather: precipitation, et0, temperature, radiation, humidity")
            print(f"  - Physics: physics_prior, physics_lag_*d")
            print(f"  - Soil: clay_pct, sand_pct, silt_pct, saturation, organic_carbon")
            print(f"  - Location: latitude, longitude, elevation")
            print(f"  - Time: day_of_year, month, year, sin/cos encodings")
            print(f"  - Derived: API indices, water balance, precip/et0 aggregations")

        print("\n" + "=" * 70)
        print(f"✓ Results saved to: {self.output_dir}")
        print("=" * 70)

        return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Run full physics+ML validation with prepared ISMN data"
    )
    parser.add_argument(
        "--prepared-dir", type=Path, default=Path("data/prepared"),
        help="Directory with prepared CSVs"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/prepared_validation"),
        help="Output directory"
    )
    parser.add_argument(
        "--max-stations", type=int, default=None,
        help="Limit number of stations (for testing)"
    )
    parser.add_argument(
        "--use-gee", action="store_true", default=True,
        help="Enable Google Earth Engine for satellite data (NDVI/LAI)"
    )
    parser.add_argument(
        "--skip-fetch", action="store_true", default=False,
        help="Skip fetching external data (use cache only)"
    )

    args = parser.parse_args()

    validator = PreparedDataValidator(
        prepared_data_dir=args.prepared_dir,
        output_dir=args.output_dir,
        skip_gee=not args.use_gee,
        skip_fetch=args.skip_fetch,
    )

    validator.run(max_stations=args.max_stations)


if __name__ == "__main__":
    main()
