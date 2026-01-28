#!/usr/bin/env python
"""
Full Hybrid Physics+ML Validation for ISMN Data.

This script:
1. Loads ISMN stations with ground truth soil moisture
2. Fetches features: Weather (Open-Meteo), Satellite (GEE), Soil (ISMN/ISDA)
3. Runs Physics Model (SimpleWaterBalance)
4. Trains and evaluates ML model (HybridSoilMoistureModel)
5. Computes metrics at different depths and forecast horizons (24h, 72h, 168h)
6. Generates scatter plots (single-site and multi-site)

Usage:
    python run_full_validation.py --max-stations 10
"""

import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
import json
import time

import pandas as pd
import numpy as np
from tqdm import tqdm

# SMPS imports
from smps.data.sources.ismn_loader import ISMNStationLoader, ISMNStationData, get_daily_soil_moisture
from smps.physics.simple_water_balance import (
    SimpleWaterBalance,
    create_simple_config_improved,
    create_simple_config
)
from smps.physics.pedotransfer import estimate_soil_parameters_tropical, TropicalSoilCorrections
from smps.validation.physics_metrics import run_physics_validation
from smps.data.sources.weather import OpenMeteoSource
from smps.data.sources.gee_satellite import GoogleEarthEngineSatelliteSource
from smps.data.sources.base import DataFetchRequest

# ML imports
from smps.ml.hybrid_model import (
    HybridSoilMoistureModel,
    ResidualLearner,
    PhysicsResidualTarget,
    ResidualLearnerConfig
)
from smps.ml.enhanced_hybrid_model import (
    UncertaintyAwareHybridModel,
    EnhancedResidualLearnerConfig,
    UncertaintyConfig,
    ResidualQualityAssessment
)
from smps.ml.domain_shift_detection import (
    DomainShiftConfig,
    DomainShiftAwareModel
)
from smps.ml.ensemble import StackingEnsemble, EnsembleConfig, BaseModelConfig
from smps.ml.spatiotemporal_features import SpatialFeatureEngineer, SpatialConfig
from smps.ml.validation import DataSplitter, SplitConfig
from smps.ml.trainer import MLTrainingPipeline

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("smps.validation.full")

# Forecast horizons for evaluation (in days)
# 0h = nowcast (same-day prediction)
HORIZONS = {
    '0h': 0,    # Nowcast (same day)
    '24h': 1,   # 1-day ahead
    '72h': 3,   # 3-day ahead
    '168h': 7   # 7-day ahead (weekly)
}

# Station quality thresholds for filtering
STATION_QUALITY_THRESHOLDS = {
    'max_abs_bias': 0.12,      # Max absolute bias |obs_mean - physics_mean|
    # Min acceptable KGE (allow some negative for correction)
    'min_kge': -0.5,
    # Max deviation from realistic soil moisture (0.1-0.5)
    'max_obs_mean_deviation': 0.05,
}


class FullValidationRunner:
    """Run complete physics + ML validation on ISMN data."""

    def __init__(
        self,
        ismn_data_dir: Path,
        output_dir: Path,
        start_date: str = "2019-01-01",
        end_date: str = "2021-12-31",
    ):
        self.ismn_data_dir = Path(ismn_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        # Weather cache
        self.weather_cache_dir = Path("data/cache/weather")
        self.weather_cache_dir.mkdir(parents=True, exist_ok=True)

        # Data sources
        self.ismn_loader = ISMNStationLoader(ismn_data_dir)
        self.weather_source = OpenMeteoSource(
            cache_dir=self.output_dir / "cache" / "weather")

        try:
            self.satellite_source = GoogleEarthEngineSatelliteSource()
            self.has_gee = True
        except Exception as e:
            logger.warning(f"GEE not available: {e}")
            self.has_gee = False

        # Spatiotemporal feature engineering
        self.spatial_engineer = SpatialFeatureEngineer(
            SpatialConfig(
                max_neighbor_distance_km=50.0,
                min_neighbors=2,  # Relaxed for sparse networks
                regional_radius_km=100.0
            )
        )

        # Results storage
        self.physics_results = []
        self.ml_results = []
        self.hybrid_results = []
        self.paired_data = []
        self.training_history = {}  # Store training curves for overfitting analysis

        # Station quality tracking for filtering
        self.station_quality = {}
        self.excluded_stations = []

        # Quality filtering settings
        self.enable_quality_filtering = True
        self.quality_thresholds = STATION_QUALITY_THRESHOLDS.copy()

    def run(
        self,
        networks: Optional[List[str]] = None,
        max_stations: Optional[int] = None,
    ):
        """Run full validation."""
        logger.info(
            f"Starting full validation from {self.start_date.date()} to {self.end_date.date()}")

        # Load stations
        if networks:
            station_data_list = []
            for network in networks:
                stations_dict = self.ismn_loader.load_network(network)
                station_data_list.extend(stations_dict.values())
        else:
            stations_dict = self.ismn_loader.load_all_stations()
            station_data_list = list(stations_dict.values())

        # Filter to stations with data in date range
        valid_stations = []
        for station in station_data_list:
            if station.daily_data is not None:
                df = station.daily_data
                df_filtered = df[(df['date'] >= self.start_date)
                                 & (df['date'] <= self.end_date)]
                if len(df_filtered) >= 100:  # At least 100 days of data
                    valid_stations.append(station)

        if max_stations:
            valid_stations = valid_stations[:max_stations]

        logger.info(f"Processing {len(valid_stations)} stations")

        # Print overview
        self._print_overview(valid_stations)

        # Process each station
        all_station_features = []

        for station_data in tqdm(valid_stations, desc="Processing stations"):
            try:
                features_df = self._process_station(station_data)
                if features_df is not None and len(features_df) >= 50:
                    all_station_features.append(features_df)
            except Exception as e:
                logger.error(
                    f"Failed to process {station_data.station_id}: {e}")
                import traceback
                traceback.print_exc()

            # Rate limiting
            time.sleep(1)

        if not all_station_features:
            logger.error("No station data collected!")
            return

        # Combine all station data
        combined_df = pd.concat(all_station_features, ignore_index=True)
        logger.info(
            f"Combined dataset: {len(combined_df)} rows, {len(combined_df.columns)} columns")

        # Add spatiotemporal features to capture spatial correlations
        logger.info("Adding spatiotemporal features...")
        try:
            # Build site metadata for spatial feature engineering
            site_metadata = {}
            for station_data in valid_stations:
                site_metadata[station_data.station_id] = {
                    'latitude': station_data.latitude,
                    'longitude': station_data.longitude,
                    'elevation_m': getattr(station_data, 'elevation_m', 200.0),
                    'slope_percent': 5.0,  # Will be updated with real data if available
                    'land_use_code': 1.0,  # Will be updated with real data if available
                }

            # Add spatiotemporal features
            combined_df = self.spatial_engineer.engineer_spatiotemporal_features(
                combined_df, site_metadata
            )
            logger.info(
                f"✓ Spatiotemporal features added: {len(combined_df)} rows, {len(combined_df.columns)} columns")

        except Exception as e:
            logger.warning(f"Spatiotemporal feature engineering failed: {e}")
            # Continue without spatiotemporal features

        # ===== STATION QUALITY FILTERING =====
        if self.enable_quality_filtering:
            logger.info("\n" + "=" * 60)
            logger.info("STATION QUALITY ASSESSMENT")
            logger.info("=" * 60)

            # Assess quality for each station
            station_ids = combined_df['station_id'].unique()
            included_stations = []
            excluded_stations = []

            for station_id in station_ids:
                station_df = combined_df[combined_df['station_id']
                                         == station_id]
                quality = self._assess_station_quality(station_id, station_df)
                self.station_quality[station_id] = quality

                if quality['include']:
                    included_stations.append(station_id)
                else:
                    excluded_stations.append(station_id)
                    logger.warning(
                        f"  Excluding {station_id}: {', '.join(quality['reasons'])}")

            self.excluded_stations = excluded_stations

            logger.info(f"\nQuality filtering results:")
            logger.info(f"  Included: {len(included_stations)} stations")
            logger.info(f"  Excluded: {len(excluded_stations)} stations")

            if excluded_stations:
                logger.info(f"\nExcluded stations:")
                for s in excluded_stations:
                    q = self.station_quality[s]
                    logger.info(f"    {s}: {q['reasons']}")

            # Filter the combined dataframe
            original_len = len(combined_df)
            combined_df = combined_df[combined_df['station_id'].isin(
                included_stations)].copy()
            logger.info(
                f"\nFiltered dataset: {len(combined_df)} rows (removed {original_len - len(combined_df)})")

        if len(combined_df) < 100:
            logger.error("Insufficient data after quality filtering!")
            return

        # Save combined dataset
        combined_df.to_csv(self.output_dir /
                           "combined_features.csv", index=False)

        # Train and evaluate ML models
        self._train_and_evaluate_ml(combined_df)

        # Save results
        self._save_results()

        # Generate plots
        self._generate_plots()

    def _print_overview(self, stations: List[ISMNStationData]):
        """Print data overview."""
        print("\n" + "=" * 80)
        print("FULL HYBRID PHYSICS + ML VALIDATION")
        print("=" * 80)
        print(f"📍 ISMN Ground Truth: {self.ismn_data_dir}")
        print(f"☀️  Weather: Open-Meteo ERA5")
        print(f"🌍 Satellite: Google Earth Engine (MODIS)")
        print(f"🏜️  Soil: ISMN Static / iSDA fallback")
        print(f"📅 Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"📊 Stations: {len(stations)}")
        print("=" * 80)

        print("\nSTATION OVERVIEW:")
        print("-" * 60)
        for s in stations[:10]:  # Show first 10
            depths = [int(d) for d in s.available_depths_cm]
            print(f"  {s.station_id[:40]:<40} | depths: {depths}")
        if len(stations) > 10:
            print(f"  ... and {len(stations) - 10} more")
        print("-" * 60 + "\n")

    def _process_station(self, station_data: ISMNStationData) -> Optional[pd.DataFrame]:
        """Process a single station and return feature DataFrame."""
        station_id = station_data.station_id
        logger.info(f"Processing: {station_id}")

        # Get observation date range
        obs = self._get_observations(station_data)
        if not obs:
            return None

        # Determine fetch window
        all_dates = []
        for depth, series in obs.items():
            all_dates.extend(series.index.tolist())

        if not all_dates:
            return None

        fetch_start = max(self.start_date, pd.Timestamp(min(all_dates)))
        fetch_end = min(self.end_date, pd.Timestamp(max(all_dates)))

        if fetch_end <= fetch_start:
            return None

        logger.info(
            f"  Date range: {fetch_start.date()} to {fetch_end.date()}")

        # Get forcing data (weather + satellite)
        forcings = self._get_forcing_data(
            station_data.latitude,
            station_data.longitude,
            fetch_start,
            fetch_end,
            station_id=station_id
        )

        if forcings is None or len(forcings) < 30:
            logger.warning(f"Insufficient forcing data for {station_id}")
            return None

        # Get soil parameters
        soil_params = self._get_soil_parameters(station_data)
        if soil_params is None:
            return None

        # Calculate observed mean for initialization
        obs_mean = None
        if obs:
            all_obs = pd.concat(obs.values())
            obs_mean = all_obs.mean()
            logger.info(f"  Observed mean θ = {obs_mean:.3f}")

        # Run physics model
        physics_results = self._run_physics_model(
            forcings, soil_params, station_data, obs_mean)
        if physics_results is None:
            return None

        # Build feature DataFrame
        features_df = self._build_features(
            station_data,
            forcings,
            physics_results,
            obs,
            soil_params
        )

        if features_df is not None:
            # Store physics validation results
            self._evaluate_physics(station_data, features_df)

        return features_df

    def _get_observations(self, station_data: ISMNStationData) -> Dict[float, pd.Series]:
        """Get ISMN observations by depth."""
        if station_data.daily_data is None:
            return {}

        observations = {}
        for depth_cm in station_data.available_depths_cm:
            df = get_daily_soil_moisture(station_data, depth_cm=depth_cm)
            if not df.empty:
                series = df.set_index('date')['soil_moisture_mean']
                series = series[(series.index >= self.start_date)
                                & (series.index <= self.end_date)]
                if not series.empty:
                    observations[depth_cm] = series

        return observations

    def _get_forcing_data(
        self,
        lat: float,
        lon: float,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        station_id: str
    ) -> Optional[pd.DataFrame]:
        """Get weather and satellite forcing data."""
        try:
            site_id = station_id
            safe_site_id = site_id.replace(
                "/", "_").replace(",", "_").replace(" ", "_")

            # Check weather cache
            cache_file = self.weather_cache_dir / \
                f"weather_{safe_site_id}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"

            if cache_file.exists():
                logger.info(f"  ✓ Loading cached weather")
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                weather_df = pd.DataFrame(cached['data'])
                weather_df['date'] = pd.to_datetime(weather_df['date'])
                weather_df = weather_df.set_index('date')
            else:
                # Fetch from API
                request = DataFetchRequest(
                    site_id=site_id,
                    start_date=start_date.date(),
                    end_date=end_date.date(),
                    parameters={'latitude': lat, 'longitude': lon}
                )
                self.weather_source._site_coordinates = {site_id: (lat, lon)}
                weather_data = self.weather_source.fetch_daily_weather(request)

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
                        'wind_speed_mean_m_s': w.wind_speed_mean_m_s
                    })

                weather_df = pd.DataFrame(records)
                weather_df['date'] = pd.to_datetime(weather_df['date'])
                weather_df = weather_df.set_index('date')

                # Cache
                try:
                    cache_data = {
                        'site_id': site_id,
                        'latitude': lat,
                        'longitude': lon,
                        'data': weather_df.reset_index().to_dict('records')
                    }
                    for rec in cache_data['data']:
                        if 'date' in rec:
                            rec['date'] = str(rec['date'])
                    with open(cache_file, 'w') as f:
                        json.dump(cache_data, f, default=str)
                    logger.info(f"  ✓ Cached weather")
                except Exception as e:
                    logger.warning(f"Cache write failed: {e}")

            # Get satellite data if available (use robust helper for different GEE API variants)
            if self.has_gee:
                try:
                    ndvi_obs = self._get_satellite_index(
                        'NDVI', lat, lon, start_date, end_date)
                    lai_obs = self._get_satellite_index(
                        'LAI', lat, lon, start_date, end_date)

                    # Normalize NDVI results into a DataFrame
                    satellite_records = []
                    for obs in ndvi_obs:
                        if isinstance(obs, dict):
                            date = obs.get('date')
                            value = obs.get('value')
                        else:
                            date = getattr(obs, 'date', None)
                            value = getattr(obs, 'value', None)
                        satellite_records.append({'date': date, 'ndvi': value})
                    satellite_df = pd.DataFrame(satellite_records)
                    satellite_df['date'] = pd.to_datetime(satellite_df['date'])

                    # Build LAI lookup
                    lai_dict = {}
                    for obs in lai_obs:
                        if isinstance(obs, dict):
                            date = obs.get('date')
                            value = obs.get('value')
                        else:
                            date = getattr(obs, 'date', None)
                            value = getattr(obs, 'value', None)
                        if date is not None:
                            lai_dict[pd.to_datetime(date)] = value

                    satellite_df['lai'] = satellite_df['date'].map(lai_dict)
                    satellite_df = satellite_df.set_index('date')

                    forcings = weather_df.join(satellite_df, how='left')
                    forcings['ndvi'] = forcings['ndvi'].ffill().bfill()
                    forcings['lai'] = forcings['lai'].ffill().bfill()
                except Exception as e:
                    logger.warning(f"Satellite data unavailable: {e}")
                    forcings = weather_df.copy()
                    forcings['ndvi'] = 0.5
                    forcings['lai'] = 1.5
            else:
                forcings = weather_df.copy()
                forcings['ndvi'] = 0.5
                forcings['lai'] = 1.5

            return forcings

        except Exception as e:
            logger.error(f"Failed to get forcing data: {e}")
            return None

    def _get_satellite_index(self, index_name: str, lat: float, lon: float,
                             start_date: pd.Timestamp, end_date: pd.Timestamp):
        """
        Fetch satellite index timeseries (e.g. NDVI, LAI) using multiple possible API methods
        and normalize the result to a list of {'date': ..., 'value': ...} entries.
        """
        if not self.has_gee or not hasattr(self, 'satellite_source') or self.satellite_source is None:
            return []

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Candidate method names commonly found in different GEE wrappers
        candidates = [
            f"fetch_{index_name.lower()}",
            f"get_{index_name.lower()}",
            f"fetch_{index_name.lower()}_timeseries",
            "fetch_index",
            "fetch_timeseries",
            "get_timeseries",
        ]

        for method_name in candidates:
            if hasattr(self.satellite_source, method_name):
                method = getattr(self.satellite_source, method_name)
                # Try common calling conventions
                for call_style in (
                    {'lat': lat, 'lon': lon, 'start_date': start_str,
                        'end_date': end_str, 'index': index_name},
                    {'lat': lat, 'lon': lon, 'start': start_str,
                        'end': end_str, 'index': index_name},
                    (lat, lon, start_str, end_str, index_name),
                    (lat, lon, start_str, end_str),
                ):
                    try:
                        if isinstance(call_style, dict):
                            res = method(
                                **{k: v for k, v in call_style.items() if v is not None})
                        else:
                            res = method(*call_style)
                        # Normalize output
                        if res is None:
                            continue
                        # pandas DataFrame
                        if isinstance(res, pd.DataFrame):
                            records = []
                            # try to find columns
                            date_col = None
                            value_col = None
                            for c in res.columns:
                                lc = c.lower()
                                if 'date' in lc:
                                    date_col = c
                                if index_name.lower() in lc or 'value' in lc:
                                    value_col = c
                            if date_col is None:
                                # assume index is datetime
                                for idx, row in res.iterrows():
                                    val = row[value_col] if value_col in res.columns else row.iloc[0]
                                    records.append({'date': idx, 'value': val})
                            else:
                                for _, row in res.iterrows():
                                    records.append(
                                        {'date': row[date_col], 'value': row[value_col]})
                            return records
                        # pandas Series
                        if isinstance(res, pd.Series):
                            return [{'date': idx, 'value': v} for idx, v in res.items()]
                        # list/tuple of dicts or objects
                        if isinstance(res, (list, tuple)):
                            # If entries are primitives, try to coerce
                            out = []
                            for entry in res:
                                if isinstance(entry, dict):
                                    out.append(
                                        {'date': entry.get('date'), 'value': entry.get('value')})
                                else:
                                    date = getattr(entry, 'date', None)
                                    val = getattr(entry, 'value', None)
                                    if date is None and isinstance(entry, (list, tuple)) and len(entry) >= 2:
                                        date, val = entry[0], entry[1]
                                    out.append({'date': date, 'value': val})
                            return out
                        # single object with attributes
                        date = getattr(res, 'date', None)
                        value = getattr(res, 'value', None)
                        if date is not None:
                            return [{'date': date, 'value': value}]
                    except Exception:
                        # try next call style or method
                        continue

        # If nothing worked, return empty list
        return []

    def _get_soil_parameters(self, station_data: ISMNStationData) -> Optional[Dict]:
        """Get soil parameters from ISMN or ISDA fallback."""
        if station_data.soil_properties:
            layer_keys = sorted(station_data.soil_properties.keys())
            for layer_key in layer_keys:
                soil_props = station_data.soil_properties[layer_key]
                if soil_props.sand_fraction is not None and soil_props.clay_fraction is not None:
                    logger.info(
                        f"  ✓ Using ISMN soil: Sand={soil_props.sand_fraction}%, Clay={soil_props.clay_fraction}%")

                    om_pct = soil_props.organic_carbon if soil_props.organic_carbon else 1.5

                    # Detect East African sites and apply stronger oxide clay corrections
                    is_east_africa = 25 <= station_data.longitude <= 45
                    clay_pct = soil_props.clay_fraction

                    if is_east_africa and clay_pct > 25:
                        # East African oxide clays need very aggressive corrections
                        # Standard PTFs overestimate water retention by 50-200%
                        logger.info(
                            f"  → Applying East African oxide clay corrections (lon={station_data.longitude})")

                        # Create corrections for EA oxide soils
                        # Lower oxide_aggregation = more effective aggregation = less water held
                        # 0 at 25%, 1 at 50%+
                        clay_factor = min(1.0, (clay_pct - 25) / 25.0)
                        tropical_corrections = TropicalSoilCorrections(
                            fc_per_percent_om=0.01,
                            structure_factor=3.0 + 2.0 * clay_factor,  # High structure factor
                            oxide_aggregation=0.3 - 0.2 * clay_factor,  # Very low for high clay
                            macropore_factor=2.0 + 1.0 * clay_factor,
                        )

                        hydraulic = estimate_soil_parameters_tropical(
                            sand_percent=soil_props.sand_fraction,
                            clay_percent=soil_props.clay_fraction,
                            organic_matter_percent=om_pct,
                            bulk_density_g_cm3=1.4,
                            tropical_corrections=tropical_corrections,
                            structure_factor=tropical_corrections.structure_factor
                        )

                        # For extreme clay, compute corrected WP and FC
                        wp_corrected = hydraulic.wilting_point
                        fc_corrected = hydraulic.field_capacity

                        if clay_pct > 40:
                            # Obs data shows these soils can dry to θ < 0.05
                            wp_corrected = max(
                                0.02, hydraulic.wilting_point * 0.3)
                            fc_corrected = max(
                                0.08, hydraulic.field_capacity * 0.5)
                            logger.info(
                                f"  → Extreme clay correction: WP={wp_corrected:.3f}, FC={fc_corrected:.3f}")

                        return {
                            'source': 'ISMN_static_EA_corrected',
                            'sand_pct': soil_props.sand_fraction,
                            'clay_pct': soil_props.clay_fraction,
                            'silt_pct': soil_props.silt_fraction,
                            'om_pct': om_pct,
                            'theta_sat': hydraulic.porosity,
                            'theta_fc': fc_corrected,
                            'theta_pwp': wp_corrected,
                            'ksat_mm_day': hydraulic.saturated_hydraulic_conductivity_cm_day * 10,
                            'alpha': hydraulic.van_genuchten_alpha,
                            'n': hydraulic.van_genuchten_n
                        }
                    else:
                        hydraulic = estimate_soil_parameters_tropical(
                            sand_percent=soil_props.sand_fraction,
                            clay_percent=soil_props.clay_fraction,
                            organic_matter_percent=om_pct,
                            bulk_density_g_cm3=1.4
                        )

                    return {
                        'source': 'ISMN_static',
                        'sand_pct': soil_props.sand_fraction,
                        'clay_pct': soil_props.clay_fraction,
                        'silt_pct': soil_props.silt_fraction,
                        'om_pct': om_pct,
                        'theta_sat': hydraulic.porosity,
                        'theta_fc': hydraulic.field_capacity,
                        'theta_pwp': hydraulic.wilting_point,
                        'ksat_mm_day': hydraulic.saturated_hydraulic_conductivity_cm_day * 10,
                        'alpha': hydraulic.van_genuchten_alpha,
                        'n': hydraulic.van_genuchten_n
                    }

        # Fallback to ISDA
        try:
            from smps.data.sources.isda_authenticated import IsdaAfricaAuthenticatedSource
            isda = IsdaAfricaAuthenticatedSource()
            soil = isda.fetch_soil_profile(
                site_id=station_data.station_id,
                latitude=station_data.latitude,
                longitude=station_data.longitude
            )
            logger.info(f"  ✓ Using ISDA soil data")
            # Convert to our format
            return {
                'source': 'ISDA',
                'sand_pct': soil.sand_pct,
                'clay_pct': soil.clay_pct,
                'theta_sat': 0.45,
                'theta_fc': 0.25,
                'theta_pwp': 0.10,
                'ksat_mm_day': 200,
                'alpha': 0.02,
                'n': 1.4
            }
        except Exception as e:
            logger.warning(f"  ISDA fetch failed: {e}")

        # Use regional defaults
        logger.warning(f"  Using regional default soil parameters")
        return {
            'source': 'regional_default',
            'sand_pct': 50,
            'clay_pct': 20,
            'theta_sat': 0.45,
            'theta_fc': 0.25,
            'theta_pwp': 0.10,
            'ksat_mm_day': 300,
            'alpha': 0.025,
            'n': 1.4
        }

    def _run_physics_model(
        self,
        forcings: pd.DataFrame,
        soil_params: Dict,
        station_data: ISMNStationData,
        obs_mean: Optional[float] = None
    ) -> Optional[pd.DataFrame]:
        """Run SimpleWaterBalance model."""
        try:
            sand_pct = soil_params.get('sand_pct', 50)
            clay_pct = soil_params.get('clay_pct', 20)

            # Get vegetation fraction from NDVI
            if 'ndvi' in forcings.columns:
                ndvi_mean = forcings['ndvi'].mean()
                if np.isnan(ndvi_mean):
                    veg_frac = 0.4
                else:
                    veg_frac = np.clip((ndvi_mean - 0.1) / 0.8, 0.1, 0.9)
            else:
                veg_frac = 0.4

            # Fetch spatial data for adaptive calibration
            try:
                elevation = self.satellite_source.fetch_elevation(
                    station_data.latitude, station_data.longitude)
                slope = self.satellite_source.fetch_slope(
                    station_data.latitude, station_data.longitude)
                logger.info(
                    f"   ✓ Spatial data: elev={elevation:.0f}m, slope={slope:.1f}%")
            except Exception as e:
                logger.warning(
                    f"   ⚠ Failed to fetch spatial data: {e}. Using defaults.")
                elevation = station_data.elevation_m if hasattr(
                    station_data, 'elevation_m') else 200.0
                slope = 5.0

            # Create improved config with tropical PTFs and adaptive calibration
            config = create_simple_config_improved(
                sand_percent=sand_pct,
                clay_percent=clay_pct,
                output_depth_m=0.10,
                n_layers=3,
                max_depth_m=1.0,
                vegetation_fraction=veg_frac,
                latitude=station_data.latitude,
                longitude=station_data.longitude,
                elevation_m=elevation,
                slope_percent=slope,
                observed_mean=obs_mean,
                use_tropical_ptf=True,
                apply_adaptive_calibration=True,
            )

            # Initialize model
            model = SimpleWaterBalance(config)

            # Initialize at observed mean if available
            if obs_mean is not None and not np.isnan(obs_mean):
                init_theta = np.clip(obs_mean, 0.05, 0.45)
                model.set_initial_conditions([init_theta] * 3)
                logger.debug(f"  Initialized at θ={init_theta:.3f}")

            # Get forcing columns
            precip = forcings.get('precipitation_mm', forcings.get(
                'precipitation', pd.Series(0, index=forcings.index)))
            et0 = forcings.get('et0_mm', forcings.get('eto_mm', None))
            if et0 is None:
                tmax = forcings.get('temperature_max_c',
                                    pd.Series(30, index=forcings.index))
                tmin = forcings.get('temperature_min_c',
                                    pd.Series(20, index=forcings.index))
                et0 = 0.0023 * 15 * (tmax + 17.8) * \
                    np.sqrt(np.maximum(0.1, tmax - tmin))
            ndvi = forcings.get('ndvi', None)

            # Run model day by day
            results = []
            dates = forcings.index
            for i, d in enumerate(dates):
                ndvi_val = float(ndvi.iloc[i]) if ndvi is not None and not np.isnan(
                    ndvi.iloc[i]) else None
                fluxes, theta_surface = model.run_daily(
                    float(precip.iloc[i]),
                    float(et0.iloc[i]),
                    ndvi_val
                )
                # Get all layer values
                theta_layers = [layer.theta for layer in model.layers]
                results.append({
                    'date': d,
                    'theta_layer_0': theta_layers[0],
                    'theta_layer_1': theta_layers[1] if len(theta_layers) > 1 else theta_layers[0],
                    'theta_layer_2': theta_layers[2] if len(theta_layers) > 2 else theta_layers[-1],
                })

            results_df = pd.DataFrame(results)
            logger.info(f"  ✓ Physics model complete ({len(results_df)} days)")

            return results_df

        except Exception as e:
            logger.error(f"Physics model failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _build_features(
        self,
        station_data: ISMNStationData,
        forcings: pd.DataFrame,
        physics_results: pd.DataFrame,
        observations: Dict[float, pd.Series],
        soil_params: Dict
    ) -> Optional[pd.DataFrame]:
        """Build feature DataFrame combining all data."""
        try:
            # Start with forcings
            features = forcings.reset_index().copy()
            features.columns = ['date'] + list(features.columns[1:])

            # Add station metadata
            features['station_id'] = station_data.station_id
            features['network'] = station_data.network
            features['latitude'] = station_data.latitude
            features['longitude'] = station_data.longitude

            # Add soil parameters
            for key, value in soil_params.items():
                if key != 'source' and isinstance(value, (int, float)):
                    features[f'soil_{key}'] = value

            # Add spatial features from Google Earth Engine
            try:
                # Fetch real spatial data
                elevation = self.satellite_source.fetch_elevation(
                    station_data.latitude, station_data.longitude)
                slope = self.satellite_source.fetch_slope(
                    station_data.latitude, station_data.longitude)
                land_use = self.satellite_source.fetch_land_use(
                    station_data.latitude, station_data.longitude)

                features['elevation_m'] = elevation
                features['slope_percent'] = slope
                features['land_use_code'] = land_use
                features['distance_to_water_m'] = 1000.0  # Still placeholder

                logger.info(
                    f"   ✓ Spatial features: elev={elevation:.0f}m, slope={slope:.1f}%, land_use={land_use}")

            except Exception as e:
                logger.warning(
                    f"   ⚠ Failed to fetch spatial data: {e}. Using defaults.")
                # Fallback to defaults
                features['elevation_m'] = station_data.elevation_m if hasattr(
                    station_data, 'elevation_m') else 200.0
                features['slope_percent'] = 5.0
                features['land_use_code'] = 1.0
                features['distance_to_water_m'] = 1000.0

            # Add physics predictions
            n_results = len(physics_results)
            features = features.iloc[:n_results].copy()

            layer_cols = [
                c for c in physics_results.columns if c.startswith('theta_layer_')]
            for col in layer_cols:
                features[f'physics_{col}'] = physics_results[col].values

            # Map to standard depth names
            if 'theta_layer_0' in physics_results.columns:
                features['physics_sm_surface'] = physics_results['theta_layer_0'].values
            if 'theta_layer_1' in physics_results.columns:
                features['physics_sm_root'] = physics_results['theta_layer_1'].values
            if 'theta_layer_2' in physics_results.columns:
                features['physics_sm_deep'] = physics_results['theta_layer_2'].values

            # Add observations at different depths
            features = features.set_index('date')
            for depth_cm, series in observations.items():
                col_name = f'obs_sm_{int(depth_cm)}cm'
                features[col_name] = series.reindex(features.index)

            features = features.reset_index()

            # Feature engineering
            features = self._add_temporal_features(features)
            features = self._add_lag_features(features)

            return features

        except Exception as e:
            logger.error(f"Feature building failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features."""
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['week'] = df['date'].dt.isocalendar().week
        df['sin_doy'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['cos_doy'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive lag features for improved forecasting."""
        # Precipitation lags and rolling sums (PAST data - available at forecast time)
        for lag in [1, 2, 3, 5, 7, 14, 21]:
            if 'precipitation_mm' in df.columns:
                df[f'precip_lag{lag}'] = df['precipitation_mm'].shift(lag)
                df[f'precip_sum{lag}d'] = df['precipitation_mm'].rolling(
                    lag, min_periods=1).sum()

        # Temperature lags and statistics (PAST data)
        for lag in [1, 3, 7, 14]:
            if 'temperature_mean_c' in df.columns:
                df[f'temp_lag{lag}'] = df['temperature_mean_c'].shift(lag)
                df[f'temp_mean{lag}d'] = df['temperature_mean_c'].rolling(
                    lag, min_periods=1).mean()

        # ET lags if available (PAST data)
        for lag in [1, 3, 7]:
            if 'et0_mm' in df.columns:
                df[f'et0_lag{lag}'] = df['et0_mm'].shift(lag)
                df[f'et0_sum{lag}d'] = df['et0_mm'].rolling(
                    lag, min_periods=1).sum()

        # Physics model lags (PAST data - very important for forecasting!)
        for physics_col in ['physics_sm_surface', 'physics_sm_root', 'physics_sm_deep']:
            if physics_col in df.columns:
                for lag in [1, 3, 7]:
                    df[f'{physics_col}_lag{lag}'] = df[physics_col].shift(lag)
                # Rolling mean of physics predictions
                df[f'{physics_col}_mean7d'] = df[physics_col].rolling(
                    7, min_periods=1).mean()
                # Change in physics prediction
                df[f'{physics_col}_change1d'] = df[physics_col].diff(1)
                df[f'{physics_col}_change3d'] = df[physics_col].diff(3)

        # ===== ADVANCED MEMORY FEATURES =====
        # ANTECEDENT PRECIPITATION INDEX (API) - Multiple decay constants
        # API represents cumulative precipitation memory with exponential decay
        if 'precipitation_mm' in df.columns:
            # API with different decay factors (0.85, 0.90, 0.95)
            for decay_factor in [0.85, 0.90, 0.95]:
                api_col = f'api_decay_{int(decay_factor*100)}'
                df[api_col] = 0.0

                # Calculate API recursively: API_t = P_t + decay_factor * API_{t-1}
                for i in range(len(df)):
                    if i == 0:
                        df.loc[df.index[i], api_col] = df.loc[df.index[i],
                                                              'precipitation_mm']
                    else:
                        prev_api = df.loc[df.index[i-1], api_col]
                        current_p = df.loc[df.index[i], 'precipitation_mm']
                        df.loc[df.index[i], api_col] = current_p + \
                            decay_factor * prev_api

        # TIME-SINCE EVENT FEATURES
        if 'precipitation_mm' in df.columns:
            # Days since last significant rain (>5mm)
            significant_rain = df['precipitation_mm'] > 5.0
            df['days_since_rain'] = significant_rain.groupby(
                (significant_rain != significant_rain.shift()).cumsum()
            ).cumcount().where(significant_rain, np.nan)
            df['days_since_rain'] = df['days_since_rain'].fillna(
                method='ffill').fillna(30)

            # Days since last rain of any amount
            any_rain = df['precipitation_mm'] > 0.1
            df['days_since_any_rain'] = any_rain.groupby(
                (any_rain != any_rain.shift()).cumsum()
            ).cumcount().where(any_rain, np.nan)
            df['days_since_any_rain'] = df['days_since_any_rain'].fillna(
                method='ffill').fillna(30)

        # PHYSICS-FLUX FEATURES (if available from physics model)
        # These capture the internal state and fluxes of the physics model
        physics_cols = [c for c in df.columns if c.startswith('physics_')]
        for col in physics_cols:
            if col.endswith('_drainage') or 'drainage' in col:
                # Drainage flux - indicates water movement through profile
                df[f'{col}_rate'] = df[col].diff(1).fillna(0)

            if col.endswith('_runoff') or 'runoff' in col:
                # Surface runoff - indicates infiltration capacity exceeded
                df[f'{col}_binary'] = (df[col] > 0.1).astype(int)

        # SOIL MOISTURE MEMORY TERMS (exponential decay lags)
        sm_cols = [c for c in df.columns if c.startswith(
            ('physics_sm_', 'obs_sm_'))]
        for sm_col in sm_cols:
            if sm_col in df.columns:
                # Exponential decay memory (similar to API but for soil moisture)
                for decay in [0.9, 0.95, 0.98]:  # Different memory lengths
                    memory_col = f'{sm_col}_memory_{int(decay*100)}'
                    df[memory_col] = 0.0

                    for i in range(len(df)):
                        if i == 0:
                            df.loc[df.index[i],
                                   memory_col] = df.loc[df.index[i], sm_col]
                        else:
                            prev_memory = df.loc[df.index[i-1], memory_col]
                            current_sm = df.loc[df.index[i], sm_col]
                            if not np.isnan(current_sm):
                                df.loc[df.index[i], memory_col] = current_sm + \
                                    decay * prev_memory
                            else:
                                df.loc[df.index[i], memory_col] = prev_memory

        # CUMULATIVE GDD SINCE START OF SEASON (proxy for root growth)
        if 'temperature_mean_c' in df.columns and 'date' in df.columns:
            # Calculate GDD (simplified - base temp = 10°C)
            base_temp = 10.0
            df['gdd_daily'] = (df['temperature_mean_c'] -
                               base_temp).clip(lower=0)

            # Reset GDD at start of each year (simplified season)
            df['year'] = df['date'].dt.year
            df['cumulative_gdd'] = df.groupby('year')['gdd_daily'].cumsum()

        # ===== FUTURE WEATHER FEATURES (for forecast horizons) =====
        # These represent "weather forecasts" - what weather is expected in the future
        # For a perfect forecast, we use actual weather shifted backward
        # In real deployment, these would come from weather forecast API
        for horizon in [1, 3, 7]:  # 1-day, 3-day, 7-day ahead
            if 'precipitation_mm' in df.columns:
                # Future cumulative precipitation (forecast window)
                df[f'precip_future_{horizon}d'] = df['precipitation_mm'].shift(-horizon).rolling(
                    horizon, min_periods=1).sum()
                # Simple future values
                df[f'precip_future{horizon}'] = df['precipitation_mm'].shift(
                    -horizon)

            if 'et0_mm' in df.columns:
                df[f'et0_future_{horizon}d'] = df['et0_mm'].shift(-horizon).rolling(
                    horizon, min_periods=1).sum()
                df[f'et0_future{horizon}'] = df['et0_mm'].shift(-horizon)

            if 'temperature_mean_c' in df.columns:
                df[f'temp_future{horizon}'] = df['temperature_mean_c'].shift(
                    -horizon)
                df[f'temp_future_mean_{horizon}d'] = df['temperature_mean_c'].shift(-horizon).rolling(
                    horizon, min_periods=1).mean()

        # Future water balance indicator (P - ET for forecast window)
        if 'precipitation_mm' in df.columns and 'et0_mm' in df.columns:
            for horizon in [1, 3, 7]:
                if f'precip_future_{horizon}d' in df.columns and f'et0_future_{horizon}d' in df.columns:
                    df[f'water_balance_future_{horizon}d'] = (
                        df[f'precip_future_{horizon}d'] -
                        df[f'et0_future_{horizon}d']
                    )

        # Water balance indicator: P - ET
        if 'precipitation_mm' in df.columns and 'et0_mm' in df.columns:
            df['water_balance_1d'] = df['precipitation_mm'] - df['et0_mm']
            df['water_balance_7d'] = df['precip_sum7d'] - \
                df['et0_sum7d'] if 'et0_sum7d' in df.columns else None

        # Dry spell indicator
        if 'precipitation_mm' in df.columns:
            df['dry_days'] = (df['precipitation_mm'] < 1).rolling(
                14, min_periods=1).sum()

        # ===== LAGGED OBSERVATION FEATURES (CRITICAL for forecasting!) =====
        # Add lagged observed soil moisture - these are the most predictive features
        # for forecasting because soil moisture has very high autocorrelation (~0.98 at 1d lag)
        obs_cols = [c for c in df.columns if c.startswith(
            'obs_sm_') and '_lag' not in c and '_mean' not in c and '_std' not in c and '_change' not in c]
        for obs_col in obs_cols:
            # Create lags matching and beyond forecast horizons
            for lag in [1, 3, 7, 14]:
                df[f'{obs_col}_lag{lag}'] = df[obs_col].shift(lag)
            # Rolling statistics of observations
            df[f'{obs_col}_mean7d'] = df[obs_col].rolling(
                7, min_periods=1).mean()
            df[f'{obs_col}_std7d'] = df[obs_col].rolling(
                7, min_periods=1).std()
            # Recent changes in observations
            df[f'{obs_col}_change1d'] = df[obs_col].diff(1)
            df[f'{obs_col}_change7d'] = df[obs_col].diff(7)

        return df

    def _assess_station_quality(self, station_id: str, features_df: pd.DataFrame) -> Dict:
        """
        Assess station quality based on physics model performance and data characteristics.
        Returns quality metrics and whether to include the station.
        """
        quality = {
            'station_id': station_id,
            'include': True,
            'reasons': [],
        }

        # Check observation data quality (only raw observations, not lagged)
        obs_cols = [c for c in features_df.columns if c.startswith(
            'obs_sm_') and '_lag' not in c and '_mean' not in c and '_std' not in c and '_change' not in c and '_memory' not in c]
        physics_cols = ['physics_sm_surface',
                        'physics_sm_root', 'physics_sm_deep']

        for obs_col in obs_cols:
            if obs_col not in features_df.columns:
                continue

            obs_data = features_df[obs_col].dropna()
            if len(obs_data) < 30:
                continue

            obs_mean = obs_data.mean()

            # Check for unrealistic soil moisture values (too dry or too wet)
            if obs_mean < 0.05:  # Less than 5% VWC mean is suspicious
                quality['include'] = False
                quality['reasons'].append(
                    f'{obs_col} mean={obs_mean:.3f} too low (likely sensor issue)')
            elif obs_mean > 0.55:  # Greater than 55% is unrealistic
                quality['include'] = False
                quality['reasons'].append(
                    f'{obs_col} mean={obs_mean:.3f} too high')

            # Check physics model bias for this observation depth
            depth_num = int(obs_col.replace('obs_sm_', '').replace('cm', ''))
            if depth_num <= 15:
                phys_col = 'physics_sm_surface'
            elif depth_num <= 40:
                phys_col = 'physics_sm_root'
            else:
                phys_col = 'physics_sm_deep'

            if phys_col in features_df.columns:
                aligned = features_df[[obs_col, phys_col]].dropna()
                if len(aligned) >= 30:
                    obs = aligned[obs_col].values
                    pred = aligned[phys_col].values

                    bias = np.mean(pred - obs)
                    abs_bias = abs(bias)

                    # Calculate correlation
                    if np.std(obs) > 0 and np.std(pred) > 0:
                        corr = np.corrcoef(obs, pred)[0, 1]
                    else:
                        corr = 0

                    # Calculate KGE components
                    r = corr
                    alpha = np.std(pred) / \
                        np.std(obs) if np.std(obs) > 0 else 1
                    beta = np.mean(pred) / \
                        np.mean(obs) if np.mean(obs) > 0 else 1
                    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)
                                      ** 2 + (beta - 1)**2)

                    quality[f'{obs_col}_bias'] = bias
                    quality[f'{obs_col}_kge'] = kge
                    quality[f'{obs_col}_corr'] = corr

                    # Apply quality thresholds
                    if abs_bias > self.quality_thresholds['max_abs_bias']:
                        quality['include'] = False
                        quality['reasons'].append(
                            f'{obs_col} bias={bias:.3f} exceeds threshold')

                    if kge < self.quality_thresholds['min_kge']:
                        quality['include'] = False
                        quality['reasons'].append(
                            f'{obs_col} KGE={kge:.3f} below threshold')

        return quality

    def _evaluate_physics(self, station_data: ISMNStationData, features_df: pd.DataFrame):
        """Evaluate physics model performance at different time horizons."""
        depth_mapping = {
            10: 'physics_sm_surface',
            20: 'physics_sm_root',
            30: 'physics_sm_root',
            60: 'physics_sm_deep',
        }

        for depth_cm in station_data.available_depths_cm:
            obs_col = f'obs_sm_{int(depth_cm)}cm'
            physics_col = depth_mapping.get(
                int(depth_cm), 'physics_sm_surface')

            if obs_col not in features_df.columns or physics_col not in features_df.columns:
                continue

            aligned = features_df[[obs_col, physics_col]].dropna()
            if len(aligned) < 30:
                continue

            # Evaluate at different horizons
            for horizon_name, horizon_days in HORIZONS.items():
                # Shift predictions forward (or equivalently, shift obs backward)
                # pred[t] compared to obs[t + horizon_days]
                obs_shifted = aligned[obs_col].shift(-horizon_days)

                horizon_aligned = pd.DataFrame({
                    obs_col: obs_shifted,
                    physics_col: aligned[physics_col]
                }).dropna()

                if len(horizon_aligned) < 20:
                    continue

                obs = horizon_aligned[obs_col].values
                pred = horizon_aligned[physics_col].values

                metrics = run_physics_validation(obs=obs, pred=pred)

                result = {
                    'station_id': station_data.station_id,
                    'network': station_data.network,
                    'latitude': station_data.latitude,
                    'longitude': station_data.longitude,
                    'depth_cm': depth_cm,
                    'n_days': len(horizon_aligned),
                    'model': 'physics',
                    'horizon': horizon_name,
                    'horizon_days': horizon_days,
                }

                if metrics.standard_metrics:
                    result.update(metrics.standard_metrics.to_dict())

                self.physics_results.append(result)

                # Store paired data for scatter plots
                for i, (idx, row) in enumerate(horizon_aligned.iterrows()):
                    self.paired_data.append({
                        'station_id': station_data.station_id,
                        'depth_cm': depth_cm,
                        'model': f'physics_{horizon_name}',
                        'horizon': horizon_name,
                        'horizon_days': horizon_days,
                        'date': features_df.iloc[idx]['date'] if 'date' in features_df.columns else idx,
                        'obs': row[obs_col],
                        'pred': row[physics_col],
                    })

            # Also evaluate same-day (0h horizon) for comparison
            obs = aligned[obs_col].values
            pred = aligned[physics_col].values

            metrics = run_physics_validation(obs=obs, pred=pred)

            result = {
                'station_id': station_data.station_id,
                'network': station_data.network,
                'latitude': station_data.latitude,
                'longitude': station_data.longitude,
                'depth_cm': depth_cm,
                'n_days': len(aligned),
                'model': 'physics',
                'horizon': '0h',
                'horizon_days': 0,
            }

            if metrics.standard_metrics:
                result.update(metrics.standard_metrics.to_dict())

            self.physics_results.append(result)

            # Store paired data for same-day
            for i, (idx, row) in enumerate(aligned.iterrows()):
                self.paired_data.append({
                    'station_id': station_data.station_id,
                    'depth_cm': depth_cm,
                    'model': 'physics_0h',
                    'horizon': '0h',
                    'horizon_days': 0,
                    'date': features_df.iloc[idx]['date'] if 'date' in features_df.columns else idx,
                    'obs': row[obs_col],
                    'pred': row[physics_col],
                })

    def _train_and_evaluate_ml(self, combined_df: pd.DataFrame):
        """Train and evaluate ML models using comprehensive best practices."""
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING ML MODELS WITH BEST PRACTICES")
        logger.info("=" * 60)

        # Initialize ML training pipeline
        ml_pipeline = MLTrainingPipeline(
            output_dir=self.output_dir,
            experiment_name=f"soil_moisture_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )

        # Find observation columns (raw observations, not lagged)
        obs_cols_raw = [c for c in combined_df.columns if c.startswith(
            'obs_sm_') and '_lag' not in c and '_mean' not in c and '_std' not in c and '_change' not in c and '_memory' not in c]
        obs_cols = [c for c in combined_df.columns if c.startswith('obs_sm_')]
        physics_cols = [
            c for c in combined_df.columns if c.startswith('physics_sm_')]

        if not obs_cols_raw:
            logger.error("No observation columns found!")
            return

        logger.info(f"Observation columns (raw): {obs_cols_raw}")
        logger.info(f"Physics columns: {physics_cols}")

        # Define feature columns (EXCLUDE raw observations to prevent data leakage)
        exclude_cols = ['date', 'station_id', 'network'] + obs_cols_raw
        feature_cols = [c for c in combined_df.columns if c not in exclude_cols and
                        combined_df[c].dtype in ['float64', 'int64']]

        # Log which obs-derived features are included
        obs_derived_features = [c for c in feature_cols if 'obs_sm_' in c]
        logger.info(
            f"Observation-derived features included: {obs_derived_features}")
        logger.info(
            f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")

        # Enhanced temporal split with gap to prevent data leakage
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df = combined_df.sort_values('date').reset_index(drop=True)

        # Calculate split dates with gap
        total_samples = len(combined_df)
        train_end_idx = int(total_samples * 0.7)
        gap_idx = int(total_samples * 0.75)  # 5% gap
        val_end_idx = int(total_samples * 0.85)

        train_end_date = combined_df.iloc[train_end_idx]['date']
        gap_end_date = combined_df.iloc[gap_idx]['date']
        val_end_date = combined_df.iloc[val_end_idx]['date']

        train_df = combined_df[combined_df['date'] <= train_end_date].copy()
        val_df = combined_df[(combined_df['date'] > gap_end_date) &
                             (combined_df['date'] <= val_end_date)].copy()
        test_df = combined_df[combined_df['date'] > val_end_date].copy()

        logger.info(
            f"Train: {len(train_df)} samples (up to {train_end_date.date()})")
        logger.info(
            f"Gap: {gap_idx - train_end_idx} samples ({train_end_date.date()} to {gap_end_date.date()})")
        logger.info(
            f"Val: {len(val_df)} samples ({gap_end_date.date()} to {val_end_date.date()})")
        logger.info(
            f"Test: {len(test_df)} samples (after {val_end_date.date()})")

        if len(train_df) < 100 or len(val_df) < 50 or len(test_df) < 50:
            logger.warning("Insufficient data for ML training")
            return

        # Train for each observation depth
        for obs_col in obs_cols_raw:
            depth_str = obs_col.replace('obs_sm_', '').replace('cm', '')
            logger.info(
                f"\n--- Training for {obs_col} with Best Practices ---")

            # Filter to rows with valid observations
            train_valid = train_df[train_df[obs_col].notna()].copy()
            val_valid = val_df[val_df[obs_col].notna()].copy()
            test_valid = test_df[test_df[obs_col].notna()].copy()

            if len(train_valid) < 50 or len(val_valid) < 20 or len(test_valid) < 20:
                logger.warning(
                    f"Insufficient data for {obs_col}: train={len(train_valid)}, val={len(val_valid)}, test={len(test_valid)}")
                continue

            # Combine train and validation for final preprocessing (but keep separate for evaluation)
            combined_train_val = pd.concat(
                [train_valid, val_valid], ignore_index=True)

            # Preprocess data with comprehensive pipeline
            X_combined, y_combined = ml_pipeline.preprocess_data(
                combined_train_val, feature_cols, obs_col
            )

            # Split back into train/val after preprocessing
            train_size = len(train_valid)
            X_train = X_combined.iloc[:train_size]
            y_train = y_combined.iloc[:train_size]
            X_val = X_combined.iloc[train_size:]
            y_val = y_combined.iloc[train_size:]

            # Preprocess test data
            X_test, y_test = ml_pipeline.preprocess_data(
                test_valid, feature_cols, obs_col)

            # Get matching physics column
            depth_num = int(depth_str)
            if depth_num <= 15:
                physics_col = 'physics_sm_surface'
            elif depth_num <= 40:
                physics_col = 'physics_sm_root'
            else:
                physics_col = 'physics_sm_deep'

            if physics_col not in combined_df.columns:
                physics_col = physics_cols[0] if physics_cols else None

            # ===== HORIZON-SPECIFIC ML TRAINING WITH BEST PRACTICES =====
            horizon_models = {}

            for horizon_name, horizon_days in HORIZONS.items():
                logger.info(
                    f"  Training ML model for {horizon_name} horizon with optimization...")

                # Create target: shift observations forward by horizon_days
                if horizon_days == 0:
                    y_train_h = y_train
                    y_val_h = y_val
                    y_test_h = y_test
                    X_train_h = X_train
                    X_val_h = X_val
                    X_test_h = X_test
                else:
                    # Forecast: predict future observation
                    y_train_shifted = y_train.shift(-horizon_days)
                    y_val_shifted = y_val.shift(-horizon_days)
                    y_test_shifted = y_test.shift(-horizon_days)

                    # Remove NaN targets
                    train_mask = y_train_shifted.notna()
                    val_mask = y_val_shifted.notna()
                    test_mask = y_test_shifted.notna()

                    y_train_h = y_train_shifted[train_mask]
                    y_val_h = y_val_shifted[val_mask]
                    y_test_h = y_test_shifted[test_mask]
                    X_train_h = X_train[train_mask]
                    X_val_h = X_val[val_mask]
                    X_test_h = X_test[test_mask]

                # Skip if insufficient data
                if len(y_train_h) < 50 or len(y_val_h) < 20 or len(y_test_h) < 20:
                    logger.warning(
                        f"    Insufficient data for {horizon_name}: train={len(y_train_h)}, val={len(y_val_h)}, test={len(y_test_h)}")
                    continue

                # Train with comprehensive pipeline
                training_results = ml_pipeline.train_with_cross_validation(
                    X_train_h, y_train_h, horizon_name, depth_str
                )

                model = training_results['model']
                horizon_models[horizon_name] = model

                # Predictions with physical bounds clipping
                y_pred_train_h = np.clip(model.predict(X_train_h), 0.0, 0.6)
                y_pred_val_h = np.clip(model.predict(X_val_h), 0.0, 0.6)
                y_pred_test_h = np.clip(model.predict(X_test_h), 0.0, 0.6)

                # Compute comprehensive metrics
                train_metrics = ml_pipeline._compute_metrics(
                    y_train_h, y_pred_train_h)
                val_metrics = ml_pipeline._compute_metrics(
                    y_val_h, y_pred_val_h)
                test_metrics = ml_pipeline._compute_metrics(
                    y_test_h, y_pred_test_h)

                # Full validation metrics
                test_full_metrics = run_physics_validation(
                    obs=y_test_h, pred=y_pred_test_h)

                ml_result = {
                    'depth': depth_str,
                    'model': 'ml_lightgbm_optimized',
                    'horizon': horizon_name,
                    'horizon_days': horizon_days,
                    'n_train': len(y_train_h),
                    'n_val': len(y_val_h),
                    'n_test': len(y_test_h),
                    'overfitting_detected': training_results['overfitting_detected'],
                    'best_iteration': training_results['best_iteration'],
                }

                # Add all metrics
                for split in ['train', 'val', 'test']:
                    metrics = locals()[f'{split}_metrics']
                    for key, value in metrics.items():
                        ml_result[f'{split}_{key}'] = value

                # Add full validation metrics
                if test_full_metrics.standard_metrics:
                    for key, value in test_full_metrics.standard_metrics.to_dict().items():
                        ml_result[f'test_{key.lower()}'] = value

                self.ml_results.append(ml_result)

                # Add SHAP explainability analysis
                try:
                    from smps.ml.explainer import SHAPExplainer
                    logger.info(
                        f"    Computing SHAP feature importance for {horizon_name}...")

                    # Use validation data for SHAP (smaller, faster)
                    shap_explainer = SHAPExplainer(model)
                    feature_importance_obj = shap_explainer.get_feature_importance(
                        X_val_h)

                    # Store top 10 most important features
                    top_features = feature_importance_obj.get_top_features(10)
                    ml_result['top_shap_features'] = top_features
                    ml_result['shap_feature_importance'] = feature_importance_obj.global_importance

                    logger.info(
                        f"    Top SHAP features: {', '.join([f'{name} ({imp:.3f})' for name, imp in top_features[:5]])}")

                except Exception as e:
                    logger.warning(f"    SHAP analysis failed: {e}")
                    ml_result['shap_error'] = str(e)

                # Store paired data for this horizon
                test_indices = test_valid.index[test_mask] if horizon_days > 0 else test_valid.index
                for i, (obs_val, pred_val) in enumerate(zip(y_test_h, y_pred_test_h)):
                    idx = test_indices[i] if isinstance(
                        test_indices, pd.Index) else test_indices.iloc[i]
                    self.paired_data.append({
                        'station_id': test_valid.loc[idx]['station_id'],
                        'depth_cm': int(depth_str),
                        'model': f'ml_lightgbm_optimized_{horizon_name}',
                        'horizon': horizon_name,
                        'horizon_days': horizon_days,
                        'date': test_valid.loc[idx]['date'],
                        'obs': obs_val,
                        'pred': pred_val,
                    })

            # Store training history for learning curves
            if '0h' in horizon_models:
                model = horizon_models['0h']
                # Simplified training history (would need to capture during training)
                self.training_history[depth_str] = {
                    'best_iteration': model.best_iteration_,
                    'n_features': len(feature_cols),
                    'overfitting_detected': training_results.get('overfitting_detected', False),
                }

            # ===== HORIZON-SPECIFIC HYBRID MODEL TRAINING =====
            # Enhanced hybrid with optimized ML
            if physics_col and physics_col in train_valid.columns:
                try:
                    logger.info(
                        f"  Training Enhanced Hybrid model for {depth_str}cm")

                    # Calculate physics quality for weighting
                    physics_train_all = train_valid[physics_col].values
                    y_train_all = train_valid[obs_col].values
                    valid_mask = ~np.isnan(
                        physics_train_all) & ~np.isnan(y_train_all)

                    physics_kge = 0.0
                    if np.sum(valid_mask) >= 30:
                        phys = physics_train_all[valid_mask]
                        obs = y_train_all[valid_mask]
                        r = np.corrcoef(obs, phys)[0, 1] if np.std(
                            obs) > 0 and np.std(phys) > 0 else 0
                        alpha = np.std(phys) / \
                            np.std(obs) if np.std(obs) > 0 else 1
                        beta = np.mean(phys) / \
                            np.mean(obs) if np.mean(obs) > 0 else 1
                        physics_kge = 1 - \
                            np.sqrt((r - 1)**2 + (alpha - 1)
                                    ** 2 + (beta - 1)**2)

                    # Adaptive weighting
                    physics_weight = max(0.1, min(0.9, physics_kge))

                    for horizon_name, horizon_days in HORIZONS.items():
                        logger.info(
                            f"    Training Hybrid for {horizon_name} horizon...")

                        # Get physics predictions
                        physics_train = train_valid[physics_col].values
                        physics_val = val_valid[physics_col].values
                        physics_test = test_valid[physics_col].values

                        if horizon_days > 0:
                            physics_train = physics_train[:-horizon_days] if len(
                                physics_train) > horizon_days else physics_train
                            physics_val = physics_val[:-horizon_days] if len(
                                physics_val) > horizon_days else physics_val
                            physics_test = physics_test[:-horizon_days] if len(
                                physics_test) > horizon_days else physics_test

                        # Enhanced hybrid features
                        X_train_h, X_val_h, X_test_h = self._enhance_hybrid_features(
                            X_train_h, X_val_h, X_test_h,
                            train_valid.iloc[:len(
                                X_train_h)] if horizon_days > 0 else train_valid,
                            val_valid.iloc[:len(
                                X_val_h)] if horizon_days > 0 else val_valid,
                            test_valid.iloc[:len(
                                X_test_h)] if horizon_days > 0 else test_valid,
                            physics_train, physics_val, physics_test,
                            y_train_h, y_val_h, y_test_h,
                            physics_weight, horizon_days
                        )

                        # Use optimized residual learner
                        learner_config = ResidualLearnerConfig(
                            use_quantile_regression=True,
                            n_estimators=2000,
                            early_stopping_rounds=100,
                            lightgbm_params=ml_pipeline.best_params.get(
                                f"{horizon_name}_{depth_str}", {})
                        )

                        residual_target = PhysicsResidualTarget(
                            target_depth=f"{depth_str}cm",
                            observation_col="obs",
                            physics_col="physics",
                            residual_col="residual",
                            # Enable delta learning for better dynamics
                            use_delta_learning=True,
                            observation_lag_col="obs_lag1",
                            physics_lag_col="physics_lag1"
                        )

                        # Prepare training data
                        train_df_hybrid = pd.DataFrame({
                            'obs': y_train_h,
                            'physics': physics_train[:len(y_train_h)],
                            'residual': y_train_h - physics_train[:len(y_train_h)],
                            # Add lag columns for delta learning
                            'obs_lag1': np.roll(y_train_h, 1),
                            'physics_lag1': np.roll(physics_train[:len(y_train_h)], 1)
                        })
                        # Set first value to NaN (no lag available)
                        train_df_hybrid.loc[0, [
                            'obs_lag1', 'physics_lag1']] = np.nan

                        for i, col in enumerate(feature_cols):
                            train_df_hybrid[col] = X_train_h.iloc[:, i] if hasattr(
                                X_train_h, 'iloc') else X_train_h[:, i]

                        val_df_hybrid = pd.DataFrame({
                            'obs': y_val_h,
                            'physics': physics_val[:len(y_val_h)],
                            'residual': y_val_h - physics_val[:len(y_val_h)],
                            # Add lag columns for delta learning
                            'obs_lag1': np.roll(y_val_h, 1),
                            'physics_lag1': np.roll(physics_val[:len(y_val_h)], 1)
                        })
                        # Set first value to NaN (no lag available)
                        val_df_hybrid.loc[0, [
                            'obs_lag1', 'physics_lag1']] = np.nan

                        for i, col in enumerate(feature_cols):
                            val_df_hybrid[col] = X_val_h.iloc[:, i] if hasattr(
                                X_val_h, 'iloc') else X_val_h[:, i]

                        # Apply residual smoothing
                        if physics_weight > 0.4:
                            train_df_hybrid['residual'] = self._smooth_residuals(
                                train_df_hybrid['residual'].values, window=7)
                            val_df_hybrid['residual'] = self._smooth_residuals(
                                val_df_hybrid['residual'].values, window=7)

                        # Train hybrid model
                        hybrid_model = ResidualLearner(learner_config)
                        training_info = hybrid_model.fit(
                            train_df_hybrid,
                            residual_target,
                            feature_names=feature_cols,
                            validation_df=val_df_hybrid
                        )

                        # Predict and combine
                        test_df_hybrid = pd.DataFrame({
                            'obs': y_test_h,
                            'physics': physics_test[:len(y_test_h)],
                        })
                        for i, col in enumerate(feature_cols):
                            test_df_hybrid[col] = X_test_h.iloc[:, i] if hasattr(
                                X_test_h, 'iloc') else X_test_h[:, i]

                        residual_pred = hybrid_model.predict(
                            test_df_hybrid, residual_target)

                        # Adaptive hybrid prediction
                        if physics_weight > 0.4:
                            y_pred_hybrid = np.clip(
                                physics_test[:len(y_test_h)] + residual_pred, 0.0, 0.6)
                        else:
                            y_pred_hybrid = np.clip(
                                physics_weight * physics_test[:len(y_test_h)] +
                                (1 - physics_weight) *
                                (physics_test[:len(y_test_h)] + residual_pred),
                                0.0, 0.6
                            )

                        # Compute metrics
                        hybrid_metrics = ml_pipeline._compute_metrics(
                            y_test_h, y_pred_hybrid)
                        hybrid_full_metrics = run_physics_validation(
                            obs=y_test_h, pred=y_pred_hybrid)

                        hybrid_result = {
                            'depth': depth_str,
                            'model': 'hybrid_physics_ml_optimized',
                            'horizon': horizon_name,
                            'horizon_days': horizon_days,
                            'n_train': len(y_train_h),
                            'n_test': len(y_test_h),
                            'physics_kge': physics_kge,
                            'physics_weight': physics_weight,
                            'use_residual': physics_weight > 0.4,
                            'quantile_enabled': True,
                            'best_iteration': training_info.get('best_iteration', 0),
                        }

                        # Add metrics
                        for key, value in hybrid_metrics.items():
                            hybrid_result[f'test_{key}'] = value
                        if hybrid_full_metrics.standard_metrics:
                            hybrid_result.update(
                                hybrid_full_metrics.standard_metrics.to_dict())

                        self.hybrid_results.append(hybrid_result)

                        logger.info(f"    {horizon_name}: RMSE={hybrid_metrics['rmse']:.4f}, "
                                    f"KGE={hybrid_metrics['kge']:.3f}, Weight={physics_weight:.3f}")

                        # Store paired data
                        test_indices = test_valid.index[test_mask] if horizon_days > 0 else test_valid.index
                        for i, (obs_val, pred_val) in enumerate(zip(y_test_h, y_pred_hybrid)):
                            idx = test_indices[i] if isinstance(
                                test_indices, pd.Index) else test_indices.iloc[i]
                            self.paired_data.append({
                                'station_id': test_valid.loc[idx]['station_id'],
                                'depth_cm': int(depth_str),
                                'model': f'hybrid_physics_ml_optimized_{horizon_name}',
                                'horizon': horizon_name,
                                'horizon_days': horizon_days,
                                'date': test_valid.loc[idx]['date'],
                                'obs': obs_val,
                                'pred': pred_val,
                            })

                except Exception as e:
                    logger.error(f"Enhanced hybrid training failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

            # ===== UNCERTAINTY-AWARE HYBRID MODEL TRAINING =====
            # Enhanced hybrid with uncertainty quantification
            if physics_col and physics_col in train_valid.columns:
                try:
                    logger.info(
                        f"  Training Uncertainty-Aware Hybrid model for {depth_str}cm")

                    # Calculate physics quality for residual assessment
                    physics_train_all = train_valid[physics_col].values
                    y_train_all = train_valid[obs_col].values
                    valid_mask = ~np.isnan(
                        physics_train_all) & ~np.isnan(y_train_all)

                    physics_kge = 0.0
                    if np.sum(valid_mask) >= 30:
                        phys = physics_train_all[valid_mask]
                        obs = y_train_all[valid_mask]
                        r = np.corrcoef(obs, phys)[0, 1] if np.std(
                            obs) > 0 and np.std(phys) > 0 else 0
                        alpha = np.std(phys) / \
                            np.std(obs) if np.std(obs) > 0 else 1
                        beta = np.mean(phys) / \
                            np.mean(obs) if np.mean(obs) > 0 else 1
                        physics_kge = 1 - \
                            np.sqrt((r - 1)**2 + (alpha - 1)
                                    ** 2 + (beta - 1)**2)

                    for horizon_name, horizon_days in HORIZONS.items():
                        logger.info(
                            f"    Training Uncertainty-Aware Hybrid for {horizon_name} horizon...")

                        # Get physics predictions
                        physics_train = train_valid[physics_col].values
                        physics_val = val_valid[physics_col].values
                        physics_test = test_valid[physics_col].values

                        if horizon_days > 0:
                            physics_train = physics_train[:-horizon_days] if len(
                                physics_train) > horizon_days else physics_train
                            physics_val = physics_val[:-horizon_days] if len(
                                physics_val) > horizon_days else physics_val
                            physics_test = physics_test[:-horizon_days] if len(
                                physics_test) > horizon_days else physics_test

                        # Enhanced hybrid features
                        X_train_h, X_val_h, X_test_h = self._enhance_hybrid_features(
                            X_train_h, X_val_h, X_test_h,
                            train_valid.iloc[:len(
                                X_train_h)] if horizon_days > 0 else train_valid,
                            val_valid.iloc[:len(
                                X_val_h)] if horizon_days > 0 else val_valid,
                            test_valid.iloc[:len(
                                X_test_h)] if horizon_days > 0 else test_valid,
                            physics_train, physics_val, physics_test,
                            y_train_h, y_val_h, y_test_h,
                            0.5, horizon_days  # Use balanced weighting for uncertainty model
                        )

                        # Configure uncertainty-aware hybrid model
                        domain_shift_config = DomainShiftConfig(
                            enable_covariate_shift_detection=True,
                            enable_ood_detection=True,
                            ood_method="isolation_forest",
                            shift_uncertainty_multiplier=2.0,
                            ood_uncertainty_multiplier=3.0
                        )

                        # Construct UncertaintyConfig without passing unexpected kwargs
                        # Some UncertaintyConfig implementations may not accept these parameters
                        # in the constructor, so create a default instance and set attributes
                        # where available.
                        uncertainty_config = UncertaintyConfig()
                        for key, value in {
                            'aleatoric_uncertainty': True,
                            'epistemic_uncertainty': True,
                            'prediction_intervals': True,
                            'quantile_levels': [0.1, 0.5, 0.9],
                            'ensemble_size': 5,
                            'enable_domain_shift_detection': True,
                            'domain_shift_config': domain_shift_config
                        }.items():
                            try:
                                setattr(uncertainty_config, key, value)
                            except Exception as e:
                                logger.debug(
                                    f"UncertaintyConfig has no attribute {key}: {e}")

                        # Create EnhancedResidualLearnerConfig instance and set attributes
                        # conditionally to avoid passing unsupported constructor kwargs.
                        learner_config = EnhancedResidualLearnerConfig()
                        for key, value in {
                            'use_quantile_regression': True,
                            'n_estimators': 2000,
                            'early_stopping_rounds': 100,
                            'lightgbm_params': ml_pipeline.best_params.get(
                                f"{horizon_name}_{depth_str}", {}),
                            'uncertainty_config': uncertainty_config
                        }.items():
                            try:
                                setattr(learner_config, key, value)
                            except Exception as e:
                                logger.debug(
                                    f"EnhancedResidualLearnerConfig has no attribute {key}: {e}")

                        # Create uncertainty-aware hybrid model
                        uncertainty_hybrid = UncertaintyAwareHybridModel(
                            learner_config
                        )

                        # Prepare training data with physics quality assessment
                        train_residuals = y_train_h - \
                            physics_train[:len(y_train_h)]
                        val_residuals = y_val_h - physics_val[:len(y_val_h)]

                        # Assess residual quality with a robust fallback (support multiple API variants)
                        residual_assessment = ResidualQualityAssessment()
                        quality_metrics = {}

                        # Try multiple common method names and both keyword and positional argument styles
                        for method_name in (
                            'assess_residual_quality',
                            'assess',
                            'evaluate_residual_quality',
                            'evaluate',
                            'assess_quality'
                        ):
                            if hasattr(residual_assessment, method_name):
                                method = getattr(
                                    residual_assessment, method_name)
                                try:
                                    quality_metrics = method(
                                        residuals=train_residuals,
                                        physics_predictions=physics_train[:len(
                                            y_train_h)],
                                        observations=y_train_h
                                    )
                                except TypeError:
                                    try:
                                        quality_metrics = method(
                                            train_residuals,
                                            physics_train[:len(y_train_h)],
                                            y_train_h
                                        )
                                    except Exception as e:
                                        logger.warning(
                                            f"Residual quality assessment ({method_name}) failed: {e}")
                                        quality_metrics = {}
                                break
                        else:
                            logger.warning(
                                "ResidualQualityAssessment has no known assessment method; skipping residual quality scoring"
                            )

                        if quality_metrics is None:
                            quality_metrics = {}

                        # Extract feature arrays
                        X_train_array = X_train_h.values if hasattr(
                            X_train_h, 'values') else X_train_h
                        X_val_array = X_val_h.values if hasattr(
                            X_val_h, 'values') else X_val_h

                        # Fit uncertainty-aware model
                        uncertainty_hybrid.fit(
                            X_train_array,
                            physics_train[:len(y_train_h)],
                            y_train_h,
                            feature_names=feature_cols
                        )

                        # Predict with uncertainty and domain shift detection
                        X_test_array = X_test_h.values if hasattr(
                            X_test_h, 'values') else X_test_h
                        uncertainty_result = uncertainty_hybrid.predict_with_uncertainty(
                            X_test_array, physics_test[:len(
                                y_test_h)], feature_names=feature_cols
                        )

                        # Extract predictions and uncertainties with domain shift awareness
                        y_pred_mean = uncertainty_result['prediction']
                        y_pred_lower = uncertainty_result.get(
                            'prediction_interval_lower', y_pred_mean - 0.05)
                        y_pred_upper = uncertainty_result.get(
                            'prediction_interval_upper', y_pred_mean + 0.05)
                        aleatoric_uncertainty = uncertainty_result['aleatoric_uncertainty']
                        epistemic_uncertainty = uncertainty_result['epistemic_uncertainty']
                        total_uncertainty = uncertainty_result.get('total_uncertainty',
                                                                   np.sqrt(aleatoric_uncertainty**2 + epistemic_uncertainty**2))

                        # Domain shift and reliability information
                        domain_shift_detected = uncertainty_result.get(
                            'domain_shift_detected', False)
                        ood_detected = uncertainty_result.get(
                            'ood_detected', False)
                        reliability_score = uncertainty_result.get(
                            'reliability_score', np.ones(len(y_pred_mean)))
                        shift_results = uncertainty_result.get(
                            'shift_results', None)
                        ood_results = uncertainty_result.get(
                            'ood_results', None)

                        # Clip predictions to physical bounds
                        y_pred_mean = np.clip(y_pred_mean, 0.0, 0.6)
                        y_pred_lower = np.clip(y_pred_lower, 0.0, 0.6)
                        y_pred_upper = np.clip(y_pred_upper, 0.0, 0.6)

                        # Compute metrics for mean prediction
                        uncertainty_metrics = ml_pipeline._compute_metrics(
                            y_test_h, y_pred_mean)
                        uncertainty_full_metrics = run_physics_validation(
                            obs=y_test_h, pred=y_pred_mean)

                        # Calculate uncertainty metrics
                        prediction_interval_width = np.mean(
                            y_pred_upper - y_pred_lower)
                        coverage_80 = np.mean(
                            (y_test_h >= y_pred_lower) & (y_test_h <= y_pred_upper))
                        uncertainty_ratio = np.mean(
                            aleatoric_uncertainty + epistemic_uncertainty) / np.std(y_test_h)

                        uncertainty_result_dict = {
                            'depth': depth_str,
                            'model': 'uncertainty_aware_hybrid',
                            'horizon': horizon_name,
                            'horizon_days': horizon_days,
                            'n_train': len(y_train_h),
                            'n_test': len(y_test_h),
                            'physics_kge': physics_kge,
                            'residual_quality_score': quality_metrics.get('overall_quality', 0),
                            'prediction_interval_width': prediction_interval_width,
                            'coverage_80pct': coverage_80,
                            'uncertainty_ratio': uncertainty_ratio,
                            'mean_aleatoric_uncertainty': np.mean(aleatoric_uncertainty),
                            'mean_epistemic_uncertainty': np.mean(epistemic_uncertainty),
                            'mean_total_uncertainty': np.mean(total_uncertainty),
                            'domain_shift_detected': domain_shift_detected,
                            'ood_detected': ood_detected,
                            'mean_reliability_score': np.mean(reliability_score),
                            'shift_detection_enabled': True,
                            'ood_detection_enabled': True,
                        }

                        # Add standard metrics
                        for key, value in uncertainty_metrics.items():
                            uncertainty_result_dict[f'test_{key}'] = value
                        if uncertainty_full_metrics.standard_metrics:
                            uncertainty_result_dict.update(
                                uncertainty_full_metrics.standard_metrics.to_dict())

                        self.hybrid_results.append(uncertainty_result_dict)

                        logger.info(f"    {horizon_name}: RMSE={uncertainty_metrics['rmse']:.4f}, "
                                    f"KGE={uncertainty_metrics['kge']:.3f}, "
                                    f"Coverage={coverage_80:.3f}, "
                                    f"Interval Width={prediction_interval_width:.4f}")

                        # Store paired data with uncertainty bounds and domain shift info
                        test_indices = test_valid.index[test_mask] if horizon_days > 0 else test_valid.index
                        for i, (obs_val, pred_val, lower_val, upper_val, alea_val, epis_val, total_unc, rel_score, shift_det, ood_det) in enumerate(zip(
                                y_test_h, y_pred_mean, y_pred_lower, y_pred_upper,
                                aleatoric_uncertainty, epistemic_uncertainty, total_uncertainty,
                                reliability_score, [domain_shift_detected]*len(y_test_h), [ood_detected]*len(y_test_h))):
                            idx = test_indices[i] if isinstance(
                                test_indices, pd.Index) else test_indices.iloc[i]
                            self.paired_data.append({
                                'station_id': test_valid.loc[idx]['station_id'],
                                'depth_cm': int(depth_str),
                                'model': f'uncertainty_aware_hybrid_{horizon_name}',
                                'horizon': horizon_name,
                                'horizon_days': horizon_days,
                                'date': test_valid.loc[idx]['date'],
                                'obs': obs_val,
                                'pred': pred_val,
                                'pred_lower': lower_val,
                                'pred_upper': upper_val,
                                'aleatoric_uncertainty': alea_val,
                                'epistemic_uncertainty': epis_val,
                                'total_uncertainty': total_unc,
                                'reliability_score': rel_score,
                                'domain_shift_detected': shift_det,
                                'ood_detected': ood_det,
                            })

                except Exception as e:
                    logger.error(
                        f"Uncertainty-aware hybrid training failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

    def _save_results(self):
        """Save all results to files."""
        logger.info("\nSaving results...")

        # Physics results
        if self.physics_results:
            physics_df = pd.DataFrame(self.physics_results)
            physics_df.to_csv(self.output_dir /
                              "physics_validation_results.csv", index=False)
            logger.info(f"  Saved physics results: {len(physics_df)} rows")

        # ML results
        if self.ml_results:
            ml_df = pd.DataFrame(self.ml_results)
            ml_df.to_csv(self.output_dir /
                         "ml_validation_results.csv", index=False)
            logger.info(f"  Saved ML results: {len(ml_df)} rows")

        # Hybrid results
        if self.hybrid_results:
            hybrid_df = pd.DataFrame(self.hybrid_results)
            hybrid_df.to_csv(self.output_dir /
                             "hybrid_validation_results.csv", index=False)
            logger.info(f"  Saved hybrid results: {len(hybrid_df)} rows")

        # Paired data for scatter plots
        if self.paired_data:
            paired_df = pd.DataFrame(self.paired_data)
            paired_df.to_csv(self.output_dir /
                             "paired_obs_pred.csv", index=False)
            logger.info(f"  Saved paired data: {len(paired_df)} rows")

        # Training history for learning curves
        if self.training_history:
            import json
            with open(self.output_dir / "training_history.json", 'w') as f:
                json.dump(self.training_history, f, indent=2)
            logger.info(
                f"  Saved training history: {len(self.training_history)} depths")

        # Summary
        self._print_summary()

    def _print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 80)
        print("VALIDATION RESULTS SUMMARY")
        print("=" * 80)

        # Physics results
        if self.physics_results:
            physics_df = pd.DataFrame(self.physics_results)
            print("\n📊 PHYSICS MODEL PERFORMANCE:")
            print("-" * 50)
            print(f"  Stations: {physics_df['station_id'].nunique()}")
            print(f"  Total observations: {physics_df['n_days'].sum()}")
            print(
                f"  RMSE: {physics_df['RMSE'].mean():.4f} ± {physics_df['RMSE'].std():.4f}")
            print(
                f"  MAE:  {physics_df['MAE'].mean():.4f} ± {physics_df['MAE'].std():.4f}")
            print(
                f"  KGE:  {physics_df['KGE'].mean():.3f} ± {physics_df['KGE'].std():.3f}")
            print(
                f"  R²:   {physics_df['R²'].mean():.3f} ± {physics_df['R²'].std():.3f}")

            print("\n  By Horizon:")
            for horizon in sorted(physics_df['horizon'].unique()):
                horizon_df = physics_df[physics_df['horizon'] == horizon]
                horizon_days = horizon_df['horizon_days'].iloc[0] if len(
                    horizon_df) > 0 else 0
                print(
                    f"    {horizon} ({horizon_days}d): RMSE={horizon_df['RMSE'].mean():.4f}, KGE={horizon_df['KGE'].mean():.3f}, n={len(horizon_df)}")

            print("\n  By Depth:")
            for depth in sorted(physics_df['depth_cm'].unique()):
                depth_df = physics_df[physics_df['depth_cm'] == depth]
                print(
                    f"    {int(depth):>3}cm: RMSE={depth_df['RMSE'].mean():.4f}, KGE={depth_df['KGE'].mean():.3f}, n={len(depth_df)}")

        # ML results
        if self.ml_results:
            ml_df = pd.DataFrame(self.ml_results)
            print("\n🤖 ML MODEL PERFORMANCE (LightGBM):")
            print("-" * 50)
            # Use test_rmse if RMSE not available (horizon-specific training)
            rmse_col = 'test_rmse' if 'test_rmse' in ml_df.columns else 'RMSE'
            r2_col = 'test_r2' if 'test_r2' in ml_df.columns else 'R²'
            kge_col = 'test_kge' if 'test_kge' in ml_df.columns else 'KGE'
            mae_col = 'test_mae' if 'test_mae' in ml_df.columns else 'MAE'

            print(
                f"  RMSE: {ml_df[rmse_col].mean():.4f} ± {ml_df[rmse_col].std():.4f}")
            if mae_col in ml_df.columns:
                print(
                    f"  MAE:  {ml_df[mae_col].mean():.4f} ± {ml_df[mae_col].std():.4f}")
            if kge_col in ml_df.columns:
                print(
                    f"  KGE:  {ml_df[kge_col].mean():.3f} ± {ml_df[kge_col].std():.3f}")
            print(
                f"  R²:   {ml_df[r2_col].mean():.3f} ± {ml_df[r2_col].std():.3f}")

            print("\n  By Horizon:")
            for horizon in sorted(ml_df['horizon'].unique()):
                horizon_df = ml_df[ml_df['horizon'] == horizon]
                horizon_days = horizon_df['horizon_days'].iloc[0] if len(
                    horizon_df) > 0 else 0
                rmse_val = horizon_df[rmse_col].mean()
                r2_val = horizon_df[r2_col].mean()
                kge_val = horizon_df[kge_col].mean(
                ) if kge_col in horizon_df.columns else 0
                print(
                    f"    {horizon} ({horizon_days}d): RMSE={rmse_val:.4f}, R²={r2_val:.3f}, KGE={kge_val:.3f}, n={len(horizon_df)}")

        # Hybrid results
        if self.hybrid_results:
            hybrid_df = pd.DataFrame(self.hybrid_results)
            print("\n🔬 HYBRID MODEL PERFORMANCE (Physics + ML):")
            print("-" * 50)
            print(
                f"  RMSE: {hybrid_df['RMSE'].mean():.4f} ± {hybrid_df['RMSE'].std():.4f}")
            print(
                f"  MAE:  {hybrid_df['MAE'].mean():.4f} ± {hybrid_df['MAE'].std():.4f}")
            print(
                f"  KGE:  {hybrid_df['KGE'].mean():.3f} ± {hybrid_df['KGE'].std():.3f}")
            print(
                f"  R²:   {hybrid_df['R²'].mean():.3f} ± {hybrid_df['R²'].std():.3f}")

            print("\n  By Horizon:")
            for horizon in sorted(hybrid_df['horizon'].unique()):
                horizon_df = hybrid_df[hybrid_df['horizon'] == horizon]
                horizon_days = horizon_df['horizon_days'].iloc[0] if len(
                    horizon_df) > 0 else 0
                print(
                    f"    {horizon} ({horizon_days}d): RMSE={horizon_df['RMSE'].mean():.4f}, KGE={horizon_df['KGE'].mean():.3f}, n={len(horizon_df)}")

        # Comparison
        if self.physics_results and self.ml_results and self.hybrid_results:
            print("\n📈 MODEL COMPARISON:")
            print("-" * 50)
            physics_rmse = pd.DataFrame(self.physics_results)['RMSE'].mean()

            # Handle both old and new column naming
            ml_df_comp = pd.DataFrame(self.ml_results)
            hybrid_df_comp = pd.DataFrame(self.hybrid_results)
            ml_rmse_col = 'test_rmse' if 'test_rmse' in ml_df_comp.columns else 'RMSE'
            hybrid_rmse_col = 'test_rmse' if 'test_rmse' in hybrid_df_comp.columns else 'RMSE'

            ml_rmse = ml_df_comp[ml_rmse_col].mean()
            hybrid_rmse = hybrid_df_comp[hybrid_rmse_col].mean()

            print(f"  {'Model':<20} {'RMSE':<12} {'Improvement':<15}")
            print(f"  {'-'*47}")
            print(f"  {'Physics':<20} {physics_rmse:.4f}       {'(baseline)':<15}")
            print(
                f"  {'ML (LightGBM)':<20} {ml_rmse:.4f}       {(1-ml_rmse/physics_rmse)*100:+.1f}%")
            print(
                f"  {'Hybrid':<20} {hybrid_rmse:.4f}       {(1-hybrid_rmse/physics_rmse)*100:+.1f}%")

            # Horizon comparison
            print("\n  Forecast Skill by Horizon:")
            physics_df = pd.DataFrame(self.physics_results)
            ml_df = pd.DataFrame(self.ml_results)
            hybrid_df = pd.DataFrame(self.hybrid_results)

            horizons = sorted(physics_df['horizon'].unique())
            print(
                f"  {'Horizon':<10} {'Physics':<10} {'ML':<10} {'Hybrid':<10} {'ML Imp':<10} {'Hyb Imp':<10}")
            print(f"  {'-'*66}")
            for horizon in horizons:
                phys_h = physics_df[physics_df['horizon']
                                    == horizon]['RMSE'].mean()
                ml_h = ml_df[ml_df['horizon'] == horizon][ml_rmse_col].mean()
                hyb_h = hybrid_df[hybrid_df['horizon']
                                  == horizon][hybrid_rmse_col].mean()

                ml_imp = (1 - ml_h/phys_h) * 100 if phys_h > 0 else 0
                hyb_imp = (1 - hyb_h/phys_h) * 100 if phys_h > 0 else 0

                print(
                    f"  {horizon:<10} {phys_h:.4f}    {ml_h:.4f}    {hyb_h:.4f}    {ml_imp:+.1f}%      {hyb_imp:+.1f}%")

        print("\n" + "=" * 80)

    def _generate_plots(self):
        """Generate scatter plots by horizon."""
        if not self.paired_data:
            logger.warning("No paired data for plotting")
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            paired_df = pd.DataFrame(self.paired_data)

            # Get unique horizons
            horizons = sorted(paired_df['horizon'].unique())

            # Multi-horizon scatter plot by model type
            for horizon in horizons:
                horizon_data = paired_df[paired_df['horizon'] == horizon]
                horizon_days = horizon_data['horizon_days'].iloc[0] if len(
                    horizon_data) > 0 else 0

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Model names in paired data include horizon suffix
                model_patterns = {
                    'physics': f'physics_{horizon}',
                    'ml_lightgbm': f'ml_lightgbm_{horizon}',
                    'hybrid': f'hybrid_physics_ml_{horizon}'
                }

                for idx, (model_key, model_pattern) in enumerate(model_patterns.items()):
                    ax = axes[idx]
                    model_data = horizon_data[horizon_data['model']
                                              == model_pattern]

                    if len(model_data) == 0:
                        ax.text(
                            0.5, 0.5, f'No data for {model_key}', ha='center', va='center')
                        continue

                    obs = model_data['obs'].values
                    pred = model_data['pred'].values

                    # Scatter plot
                    ax.scatter(pred, obs, alpha=0.3, s=10, c='steelblue')

                    # 1:1 line
                    lim = [min(obs.min(), pred.min()),
                           max(obs.max(), pred.max())]
                    ax.plot(lim, lim, 'r--', lw=2, label='1:1')

                    # Fit line
                    if len(obs) > 10:
                        z = np.polyfit(pred, obs, 1)
                        p = np.poly1d(z)
                        ax.plot(lim, p(lim), 'g-', lw=1.5,
                                label=f'Fit: y={z[0]:.2f}x+{z[1]:.3f}')

                    # Metrics
                    rmse = np.sqrt(np.mean((obs - pred) ** 2))
                    r2 = np.corrcoef(obs, pred)[0, 1] ** 2

                    ax.set_xlabel('Predicted SM (m³/m³)')
                    ax.set_ylabel('Observed SM (m³/m³)')
                    ax.set_title(
                        f'{model_key.upper()} - {horizon} ({horizon_days}d)\nRMSE={rmse:.4f}, R²={r2:.3f}, n={len(obs)}')
                    ax.legend(loc='upper left')
                    ax.set_xlim(lim)
                    ax.set_ylim(lim)
                    ax.set_aspect('equal')

                plt.tight_layout()
                plt.savefig(self.output_dir /
                            f"scatter_multimodel_{horizon}.png", dpi=150)
                plt.close()
                logger.info(
                    f"  Saved scatter plot: scatter_multimodel_{horizon}.png")

            # Scatter by depth (using 0h horizon for comparison)
            depth_data = paired_df[(paired_df['horizon'] == '0h') & (
                paired_df['model'] == 'physics_0h')]
            depths = sorted(depth_data['depth_cm'].unique())
            n_depths = len(depths)
            if n_depths > 0:
                fig, axes = plt.subplots(
                    1, min(n_depths, 4), figsize=(5*min(n_depths, 4), 5))
                if n_depths == 1:
                    axes = [axes]

                for idx, depth in enumerate(depths[:4]):
                    ax = axes[idx]
                    depth_data_plot = paired_df[(paired_df['depth_cm'] == depth) & (
                        paired_df['model'] == 'physics_0h') & (paired_df['horizon'] == '0h')]

                    if len(depth_data_plot) == 0:
                        continue

                    obs = depth_data_plot['obs'].values
                    pred = depth_data_plot['pred'].values

                    ax.scatter(pred, obs, alpha=0.4, s=15, c='steelblue')

                    lim = [min(obs.min(), pred.min()),
                           max(obs.max(), pred.max())]
                    ax.plot(lim, lim, 'r--', lw=2)

                    rmse = np.sqrt(np.mean((obs - pred) ** 2))
                    r2 = np.corrcoef(obs, pred)[0, 1] ** 2

                    ax.set_xlabel('Predicted SM (m³/m³)')
                    ax.set_ylabel('Observed SM (m³/m³)')
                    ax.set_title(
                        f'{int(depth)}cm Depth\nRMSE={rmse:.4f}, R²={r2:.3f}')
                    ax.set_xlim(lim)
                    ax.set_ylim(lim)
                    ax.set_aspect('equal')

                plt.tight_layout()
                plt.savefig(self.output_dir / "scatter_by_depth.png", dpi=150)
                plt.close()
                logger.info(f"  Saved scatter plot: scatter_by_depth.png")

            # Horizon comparison plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            models = ['physics', 'ml_lightgbm', 'hybrid']
            model_names = ['Physics', 'ML (LightGBM)', 'Hybrid']

            for idx, (model, name) in enumerate(zip(models, model_names)):
                ax = axes[idx]

                horizon_stats = []
                horizon_labels = []

                for horizon in horizons:
                    # Filter by model pattern that includes horizon suffix
                    model_pattern = f'{model}_{horizon}'
                    model_data = paired_df[paired_df['model'] == model_pattern]

                    if len(model_data) > 10:
                        rmse = np.sqrt(
                            np.mean((model_data['obs'] - model_data['pred']) ** 2))
                        horizon_stats.append(rmse)
                        horizon_labels.append(horizon)

                if horizon_stats:
                    ax.bar(range(len(horizon_stats)), horizon_stats,
                           color='steelblue', alpha=0.7)
                    ax.set_xticks(range(len(horizon_labels)))
                    ax.set_xticklabels(horizon_labels)
                    ax.set_ylabel('RMSE (m³/m³)')
                    ax.set_title(f'{name} Forecast Skill by Horizon')
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / "horizon_comparison.png", dpi=150)
            plt.close()
            logger.info(
                f"  Saved horizon comparison plot: horizon_comparison.png")

            # Single-site scatter plots (top 5 stations) - using 0h horizon
            station_data_0h = paired_df[paired_df['horizon'] == '0h']
            stations = station_data_0h['station_id'].unique()[:5]
            if len(stations) > 0:
                fig, axes = plt.subplots(
                    2, len(stations), figsize=(4*len(stations), 8))
                if len(stations) == 1:
                    axes = axes.reshape(2, 1)

                for idx, station in enumerate(stations):
                    station_data = station_data_0h[station_data_0h['station_id'] == station]
                    # Physics
                    ax = axes[0, idx]
                    phys_data = station_data[station_data['model']
                                             == 'physics_0h']
                    if len(phys_data) > 0:
                        obs = phys_data['obs'].values
                        pred = phys_data['pred'].values
                        ax.scatter(pred, obs, alpha=0.5, s=20)
                        lim = [min(obs.min(), pred.min()),
                               max(obs.max(), pred.max())]
                        ax.plot(lim, lim, 'r--', lw=2)
                        rmse = np.sqrt(np.mean((obs - pred) ** 2))
                        ax.set_title(
                            f'{station[:20]}\nPhysics RMSE={rmse:.4f}')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Observed')

                    # Hybrid
                    ax = axes[1, idx]
                    hybrid_data = station_data[station_data['model']
                                               == 'hybrid_0h']
                    if len(hybrid_data) > 0:
                        obs = hybrid_data['obs'].values
                        pred = hybrid_data['pred'].values
                        ax.scatter(pred, obs, alpha=0.5, s=20, c='green')
                        lim = [min(obs.min(), pred.min()),
                               max(obs.max(), pred.max())]
                        ax.plot(lim, lim, 'r--', lw=2)
                        rmse = np.sqrt(np.mean((obs - pred) ** 2))
                        ax.set_title(f'Hybrid RMSE={rmse:.4f}')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Observed')

                plt.tight_layout()
                plt.savefig(self.output_dir /
                            "scatter_single_sites.png", dpi=150)
                plt.close()
                logger.info(f"  Saved scatter plot: scatter_single_sites.png")

            # Learning curves
            self._plot_learning_curves()

        except ImportError:
            logger.warning("matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Plotting failed: {e}")
            import traceback
            traceback.print_exc()

    def _plot_learning_curves(self):
        """Plot learning curves for overfitting analysis."""
        if not self.training_history:
            return

        try:
            import matplotlib.pyplot as plt

            depths = list(self.training_history.keys())
            n_depths = len(depths)

            if n_depths == 0:
                return

            fig, axes = plt.subplots(n_depths, 1, figsize=(10, 4*n_depths))
            if n_depths == 1:
                axes = [axes]

            for i, depth in enumerate(depths):
                ax = axes[i]
                history = self.training_history[depth]

                train_rmse = history['train_rmse']
                val_rmse = history['val_rmse']
                test_rmse = history['test_rmse']

                iterations = range(1, len(train_rmse) + 1)

                ax.plot(iterations, train_rmse, 'b-',
                        label='Train RMSE', linewidth=2)
                ax.plot(iterations, val_rmse, 'g-',
                        label='Validation RMSE', linewidth=2)
                ax.plot(iterations, test_rmse, 'r-',
                        label='Test RMSE', linewidth=2)

                # Mark best iteration
                best_iter = history['best_iteration']
                if best_iter < len(train_rmse):
                    ax.axvline(x=best_iter, color='orange', linestyle='--',
                               alpha=0.7, label=f'Best iteration ({best_iter})')

                ax.set_xlabel('Boosting Iteration')
                ax.set_ylabel('RMSE')
                ax.set_title(f'Learning Curve - Depth {depth} cm')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Add final metrics
                final_train = train_rmse[-1]
                final_val = val_rmse[-1]
                final_test = test_rmse[-1]

                ax.text(0.02, 0.98, '.4f',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.savefig(self.output_dir / "learning_curves.png",
                        dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("  Saved learning curves: learning_curves.png")

        except Exception as e:
            logger.error(f"Learning curve plotting failed: {e}")

    def _calculate_adaptive_physics_weight(self, physics_kge, physics_vals, obs_vals, horizon_days, depth):
        """
        Calculate adaptive physics weight based on multiple quality metrics.

        Args:
            physics_kge: Kling-Gupta Efficiency of physics model
            physics_vals: Physics predictions
            obs_vals: Observations
            horizon_days: Forecast horizon
            depth: Soil depth

        Returns:
            Weight between 0-1 (0 = pure ML, 1 = full physics trust)
        """
        # Base weight from KGE (clipped and scaled)
        base_weight = np.clip((physics_kge + 0.5) / 1.0, 0.0, 1.0)

        # Horizon penalty: longer horizons reduce physics trust
        horizon_penalty = max(0, 1.0 - horizon_days / 168.0)  # 168h = 7 days

        # Depth-specific adjustments
        depth_factor = 1.0
        if depth <= 15:  # Surface layer - physics usually good
            depth_factor = 1.1
        elif depth >= 100:  # Deep layer - physics often poor
            depth_factor = 0.8

        # Bias penalty: high bias reduces trust
        bias = np.mean(physics_vals - obs_vals)
        bias_penalty = max(0, 1.0 - abs(bias) / 0.1)  # 0.1 VWC bias threshold

        # Combine factors
        weight = base_weight * horizon_penalty * depth_factor * bias_penalty
        weight = np.clip(weight, 0.0, 1.0)

        return weight

    def _enhance_hybrid_features(self, X_train, X_val, X_test, df_train, df_val, df_test,
                                 physics_train, physics_val, physics_test,
                                 obs_train, obs_val, obs_test, physics_weight, horizon_days):
        """
        Enhance features for hybrid model with physics state variables and temporal smoothing.
        """
        # Convert to DataFrames for easier manipulation
        X_train_df = pd.DataFrame(
            X_train, columns=X_train.columns if hasattr(X_train, 'columns') else None)
        X_val_df = pd.DataFrame(
            X_val, columns=X_val.columns if hasattr(X_val, 'columns') else None)
        X_test_df = pd.DataFrame(
            X_test, columns=X_test.columns if hasattr(X_test, 'columns') else None)

        # Add physics state variables
        physics_cols = [
            c for c in df_train.columns if c.startswith('physics_')]
        for col in physics_cols:
            if col in df_train.columns:
                X_train_df[f'physics_state_{col}'] = df_train[col].values
                X_val_df[f'physics_state_{col}'] = df_val[col].values
                X_test_df[f'physics_state_{col}'] = df_test[col].values

        # Add physics-observation differences (model bias indicators)
        obs_cols = [c for c in df_train.columns if c.startswith(
            'obs_sm_') and not c.endswith('_lag1')]
        for obs_col in obs_cols:
            phys_col = f'physics_sm_surface'  # Default mapping
            if '20' in obs_col:
                phys_col = 'physics_sm_root'
            elif '30' in obs_col:
                phys_col = 'physics_sm_root'
            elif '200' in obs_col:
                phys_col = 'physics_sm_deep'

            if phys_col in df_train.columns and obs_col in df_train.columns:
                # Current bias
                X_train_df[f'physics_bias_{obs_col.replace("obs_sm_", "")}'] = (
                    df_train[phys_col] - df_train[obs_col]
                ).fillna(0)
                X_val_df[f'physics_bias_{obs_col.replace("obs_sm_", "")}'] = (
                    df_val[phys_col] - df_val[obs_col]
                ).fillna(0)
                X_test_df[f'physics_bias_{obs_col.replace("obs_sm_", "")}'] = (
                    df_test[phys_col] - df_test[obs_col]
                ).fillna(0)

                # Rolling bias (7-day average)
                bias_rolling = (
                    df_train[phys_col] - df_train[obs_col]).rolling(7, min_periods=1).mean()
                X_train_df[f'physics_bias_7d_{obs_col.replace("obs_sm_", "")}'] = bias_rolling.fillna(
                    0)

                bias_rolling_val = (
                    df_val[phys_col] - df_val[obs_col]).rolling(7, min_periods=1).mean()
                X_val_df[f'physics_bias_7d_{obs_col.replace("obs_sm_", "")}'] = bias_rolling_val.fillna(
                    0)

                bias_rolling_test = (
                    df_test[phys_col] - df_test[obs_col]).rolling(7, min_periods=1).mean()
                X_test_df[f'physics_bias_7d_{obs_col.replace("obs_sm_", "")}'] = bias_rolling_test.fillna(
                    0)

        # Add physics confidence indicators
        X_train_df['physics_confidence'] = physics_weight
        X_val_df['physics_confidence'] = physics_weight
        X_test_df['physics_confidence'] = physics_weight

        # Add horizon-specific features
        X_train_df['horizon_days'] = horizon_days
        X_val_df['horizon_days'] = horizon_days
        X_test_df['horizon_days'] = horizon_days

        # Add uncertainty indicators (physics prediction variance)
        if len(physics_cols) > 1:
            physics_variances = df_train[physics_cols].var(axis=1)
            X_train_df['physics_variance'] = physics_variances.fillna(0)

            physics_variances_val = df_val[physics_cols].var(axis=1)
            X_val_df['physics_variance'] = physics_variances_val.fillna(0)

            physics_variances_test = df_test[physics_cols].var(axis=1)
            X_test_df['physics_variance'] = physics_variances_test.fillna(0)

        return X_train_df.values, X_val_df.values, X_test_df.values

    def _smooth_residuals(self, residuals, window=7):
        """
        Apply temporal smoothing to residuals to reduce noise.

        Args:
            residuals: Array of residual values
            window: Smoothing window size

        Returns:
            Smoothed residuals
        """
        if len(residuals) < window:
            return residuals

        # Exponential smoothing
        smoothed = np.zeros_like(residuals)
        alpha = 0.3  # Smoothing factor

        smoothed[0] = residuals[0]
        for i in range(1, len(residuals)):
            smoothed[i] = alpha * residuals[i] + (1 - alpha) * smoothed[i-1]

        return smoothed


def main():
    parser = argparse.ArgumentParser(
        description="Full hybrid physics+ML validation")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(
                            "data/ismn/Data_separate_files_header_20170105_20250105_12892_F2PyW_20260105"),
                        help="Path to ISMN data")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/full_validation"),
                        help="Output directory")
    parser.add_argument("--network", type=str, nargs="+",
                        help="Networks to process")
    parser.add_argument("--start-date", type=str, default="2019-01-01")
    parser.add_argument("--end-date", type=str, default="2021-12-31")
    parser.add_argument("--max-stations", type=int, default=None,
                        help="Max stations to process")

    args = parser.parse_args()

    runner = FullValidationRunner(
        ismn_data_dir=args.data_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    runner.run(
        networks=args.network,
        max_stations=args.max_stations,
    )


if __name__ == "__main__":
    main()
