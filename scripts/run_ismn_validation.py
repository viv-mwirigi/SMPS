"""
ISMN Validation Benchmark for SMPS Physics Model.

This script:
1. Loads ISMN stations with multi-depth soil moisture observations (2017-2021)
2. For each station:
   - Fetches weather data (precipitation, temperature, ET)
   - Fetches soil properties (from ISMN static vars or iSDA/SoilGrids)
   - Fetches NDVI/LAI (from MODIS via GEE)
   - Runs EnhancedWaterBalance physics model
   - Compares predictions vs observations at multiple depths
3. Computes physics-based validation metrics:
   - Standard: RMSE, MAE, Bias, R², NSE, KGE
   - Extended: ubRMSE, MAPE, KGE decomposition
   - Multi-depth: Layer consistency, vertical gradients
   - Temporal: Autocorrelation, seasonal phase
   - Extreme events: Drought/flood performance
4. Aggregates results by:
   - Station
   - Depth
   - Time horizon (24h, 72h, 168h)
   - Season
5. Exports metrics to CSV/Parquet for analysis

Usage:
    python run_ismn_validation.py --data-dir /path/to/ismn --network AMMA-CATCH
    python run_ismn_validation.py --all-networks --start-date 2017-01-01 --end-date 2021-12-31
"""

import logging
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
from smps.physics.enhanced_water_balance import EnhancedWaterBalance, EnhancedModelParameters
from smps.physics.pedotransfer import (
    estimate_soil_parameters_tropical,
    TropicalSoilCorrections,
    estimate_van_genuchten_parameters,
)
from smps.core.types import SoilParameters
from smps.validation.physics_metrics import (
    run_physics_validation,
    compute_multilayer_consistency,
    identify_drought_periods,
    identify_flood_periods,
    compute_extreme_event_metrics,
    ExtremeEventType,
    PhysicsValidationReport
)

# Data source imports (weather, satellite, soil)
from smps.data.sources.weather import OpenMeteoSource
from smps.data.sources.gee_satellite import GoogleEarthEngineSatelliteSource
from smps.data.sources.base import DataFetchRequest

logger = logging.getLogger("smps.validation.ismn_benchmark")


class ISMNValidationRunner:
    """
    Orchestrates ISMN validation for SMPS physics model.
    """

    def __init__(
        self,
        ismn_data_dir: Path,
        output_dir: Path,
        run_name: Optional[str] = None,
        start_date: str = "2017-01-01",
        end_date: str = "2021-12-31",
        export_pairs: bool = False,
    ):
        self.ismn_data_dir = Path(ismn_data_dir)
        self.output_root = Path(output_dir)
        self.run_name = (run_name.strip() if isinstance(
            run_name, str) and run_name.strip() else None)
        self.output_dir = self.output_root / \
            self.run_name if self.run_name else self.output_root
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.export_pairs = bool(export_pairs)

        # Weather cache directory
        self.weather_cache_dir = Path("data/cache/weather")
        self.weather_cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data clients
        self.ismn_loader = ISMNStationLoader(ismn_data_dir)
        self.weather_source = OpenMeteoSource(
            cache_dir=self.output_dir / "cache" / "weather")
        # Requires GEE authentication
        self.satellite_source = GoogleEarthEngineSatelliteSource()

        # Results storage
        self.results = []
        self.pairs_records: List[Dict] = []

    def run_validation(
        self,
        networks: Optional[List[str]] = None,
        stations: Optional[List[str]] = None,
        max_stations: Optional[int] = None
    ):
        """
        Run validation for specified networks/stations.

        Args:
            networks: List of network names to process (None = all)
            stations: List of specific station names (None = all in networks)
            max_stations: Maximum number of stations to process (for testing)
        """
        logger.info(
            f"Starting ISMN validation from {self.start_date.date()} to {self.end_date.date()}")

        # Load stations
        if stations:
            # Load specific stations
            station_data_list = self._load_specific_stations(stations)
        elif networks:
            # Load all stations in specified networks
            station_data_list = []
            for network in networks:
                stations_dict = self.ismn_loader.load_network(network)
                station_data_list.extend(stations_dict.values())
        else:
            # Load all stations
            stations_dict = self.ismn_loader.load_all_stations()
            station_data_list = list(stations_dict.values())

        if max_stations:
            station_data_list = station_data_list[:max_stations]

        logger.info(f"Loaded {len(station_data_list)} stations for validation")

        # Show data sources being used
        print("\n" + "=" * 80)
        print("DATA SOURCES")
        print("=" * 80)
        print(f"  📍 ISMN Ground Truth: {self.ismn_data_dir}")
        print(
            f"  ☀️  Weather: Open-Meteo ERA5 (https://archive-api.open-meteo.com/v1/era5)")
        print(f"  🌍 Satellite: Google Earth Engine (MODIS NDVI/LAI)")
        print(f"  🏜️  Soil: iSDA Africa / SoilGrids (250m)")
        print(
            f"  📅 Period: {self.start_date.date()} to {self.end_date.date()}")
        print("=" * 80 + "\n")

        # Show station overview
        print("STATION DATA QUALITY OVERVIEW")
        print("-" * 60)
        for station_data in station_data_list:
            depths = [int(d) for d in station_data.available_depths_cm]
            obs = station_data.daily_data
            n_days = len(obs) if obs is not None else 0
            print(
                f"  {station_data.station_id[:45]:<45} | depths: {depths} | {n_days} days")
        print("-" * 60 + "\n")

        # Process each station
        for station_data in tqdm(station_data_list, desc="Validating stations"):
            try:
                self._validate_station(station_data)
            except Exception as e:
                logger.error(
                    f"Failed to validate {station_data.station_id}: {e}")

            # Rate limiting: wait 2 seconds between stations to avoid API limits
            time.sleep(2)

        # Save results
        self._save_results()

    def prefetch_weather(
        self,
        networks: Optional[List[str]] = None,
        stations: Optional[List[str]] = None,
        max_stations: Optional[int] = None,
        sleep_seconds: float = 0.0,
    ) -> None:
        """Fetch and cache weather for stations (weather only).

        Populates the same on-disk cache used by validation:
        `data/cache/weather/weather_<station>_<YYYYMMDD>_<YYYYMMDD>.json`.
        Skips any station windows that already exist.
        """
        logger.info(
            "Prefetching weather cache from %s to %s",
            self.start_date.date(),
            self.end_date.date(),
        )

        # Load stations
        if stations:
            station_data_list = self._load_specific_stations(stations)
        elif networks:
            station_data_list = []
            for network in networks:
                stations_dict = self.ismn_loader.load_network(network)
                station_data_list.extend(stations_dict.values())
        else:
            all_stations = self.ismn_loader.load_all_stations()
            station_data_list = list(all_stations.values())

        if max_stations:
            station_data_list = station_data_list[:max_stations]

        n_hit = 0
        n_miss = 0
        n_ok = 0
        n_fail = 0

        for station_data in tqdm(station_data_list, desc="Prefetch weather"):
            obs = self._get_observations(station_data)
            if not obs:
                continue

            starts = [s.index.min() for s in obs.values() if len(s) > 0]
            ends = [s.index.max() for s in obs.values() if len(s) > 0]
            if not starts or not ends:
                continue

            obs_start = pd.to_datetime(min(starts))
            obs_end = pd.to_datetime(max(ends))
            fetch_start = max(self.start_date, obs_start)
            fetch_end = min(self.end_date, obs_end)
            if fetch_end < fetch_start:
                continue

            site_id = station_data.station_id
            safe_site_id = site_id.replace(
                "/", "_").replace(",", "_").replace(" ", "_")
            cache_file = (
                self.weather_cache_dir
                / f"weather_{safe_site_id}_{fetch_start.strftime('%Y%m%d')}_{fetch_end.strftime('%Y%m%d')}.json"
            )

            if cache_file.exists():
                n_hit += 1
                continue
            n_miss += 1

            try:
                request = DataFetchRequest(
                    site_id=site_id,
                    start_date=fetch_start.date(),
                    end_date=fetch_end.date(),
                    parameters={"latitude": station_data.latitude,
                                "longitude": station_data.longitude},
                )

                # Provide coordinates mapping for OpenMeteoSource
                self.weather_source._site_coordinates = {
                    site_id: (station_data.latitude, station_data.longitude)
                }

                weather_data = self.weather_source.fetch_daily_weather(request)
                if not weather_data:
                    n_fail += 1
                    continue

                records = []
                for w in weather_data:
                    records.append(
                        {
                            "date": w.date,
                            "precipitation_mm": w.precipitation_mm,
                            "et0_mm": w.et0_mm,
                            "temperature_mean_c": w.temperature_mean_c,
                            "temperature_min_c": w.temperature_min_c,
                            "temperature_max_c": w.temperature_max_c,
                            "solar_radiation_mj_m2": w.solar_radiation_mj_m2,
                            "relative_humidity_mean": w.relative_humidity_mean,
                            "wind_speed_mean_m_s": w.wind_speed_mean_m_s,
                        }
                    )

                weather_df = pd.DataFrame(records)
                weather_df["date"] = pd.to_datetime(weather_df["date"])
                weather_df = weather_df.set_index("date")

                cache_data = {
                    "site_id": site_id,
                    "latitude": station_data.latitude,
                    "longitude": station_data.longitude,
                    "start_date": fetch_start.strftime("%Y-%m-%d"),
                    "end_date": fetch_end.strftime("%Y-%m-%d"),
                    "data": weather_df.reset_index().to_dict("records"),
                }
                for rec in cache_data["data"]:
                    if "date" in rec:
                        rec["date"] = str(rec["date"])

                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, default=str)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                logger.warning(
                    "Weather prefetch failed for %s: %s", site_id, e)
            finally:
                if sleep_seconds and sleep_seconds > 0:
                    time.sleep(float(sleep_seconds))

        logger.info(
            "Prefetch complete. cache_hit=%d cache_miss=%d fetched_ok=%d fetched_fail=%d",
            n_hit,
            n_miss,
            n_ok,
            n_fail,
        )

    def _validate_station(self, station_data: ISMNStationData):
        """Validate a single station."""
        logger.info(f"Processing station: {station_data.station_id}")
        logger.info(
            f"  Location: ({station_data.latitude:.4f}, {station_data.longitude:.4f})")
        logger.info(
            f"  Sensor depths: {[int(d) for d in station_data.available_depths_cm]} cm")

        # Data quality check
        obs = self._get_observations(station_data)
        if obs:
            for depth, series in obs.items():
                n_valid = series.notna().sum()
                date_range = f"{series.index.min().date()} to {series.index.max().date()}" if len(
                    series) > 0 else "N/A"
                logger.info(
                    f"  → {int(depth):3d}cm: {n_valid} valid days ({date_range})")

        # Determine the station-specific observation window for correct forcing fetch.
        # This prevents fetching weather/satellite for years where the station has no observations.
        obs_start = None
        obs_end = None
        if obs:
            starts = [s.index.min() for s in obs.values() if len(s) > 0]
            ends = [s.index.max() for s in obs.values() if len(s) > 0]
            if starts and ends:
                obs_start = min(starts)
                obs_end = max(ends)

        # Clamp station window to the runner's configured bounds
        fetch_start = self.start_date
        fetch_end = self.end_date
        if obs_start is not None:
            fetch_start = max(fetch_start, pd.to_datetime(obs_start))
        if obs_end is not None:
            fetch_end = min(fetch_end, pd.to_datetime(obs_end))

        if fetch_end < fetch_start:
            logger.warning(
                f"Station {station_data.station_id} has no observations within configured period; skipping"
            )
            return

        logger.info(
            f"  Forcing fetch window: {fetch_start.date()} to {fetch_end.date()} (from observations)"
        )

        # 1. Get forcing data (weather + satellite)
        forcings = self._get_forcing_data(
            station_data.latitude,
            station_data.longitude,
            fetch_start,
            fetch_end,
            station_id=station_data.station_id
        )

        if forcings is None or len(forcings) < 30:
            logger.warning(
                f"Insufficient forcing data for {station_data.station_id}")
            return

        # 2. Get soil parameters
        soil_params = self._get_soil_parameters(station_data)

        if soil_params is None:
            logger.warning(
                f"No reliable soil data for {station_data.station_id}, skipping")
            return

        # 3. Configure and run physics model
        model = self._run_physics_model(forcings, soil_params)

        if model is None:
            logger.warning(f"Model failed for {station_data.station_id}")
            return

        # 4. Extract model predictions by depth (interpolated to sensor depths)
        target_depths = list(station_data.available_depths_cm)
        model_predictions = self._extract_model_predictions(
            model, forcings.index, target_depths_cm=target_depths)

        # 5. Get ISMN observations by depth
        observations = self._get_observations(station_data)

        # 6. Align and validate
        self._compute_and_store_metrics(
            station_data,
            observations,
            model_predictions,
            forcings
        )

    def _get_forcing_data(
        self,
        lat: float,
        lon: float,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        station_id: str = None
    ) -> Optional[pd.DataFrame]:
        """Get weather and satellite forcing data with caching."""
        try:
            # Use station_id for cache key if provided
            site_id = station_id or f"site_{lat:.4f}_{lon:.4f}"
            safe_site_id = site_id.replace(
                "/", "_").replace(",", "_").replace(" ", "_")

            # Check weather cache first
            cache_file = self.weather_cache_dir / \
                f"weather_{safe_site_id}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"

            if cache_file.exists():
                logger.info(f"✓ Loading cached weather for {site_id}")
                try:
                    with open(cache_file, 'r') as f:
                        cached = json.load(f)
                    weather_df = pd.DataFrame(cached['data'])
                    weather_df['date'] = pd.to_datetime(weather_df['date'])
                    weather_df = weather_df.set_index('date')
                except Exception as e:
                    logger.warning(f"Cache read failed, fetching fresh: {e}")
                    weather_df = None
            else:
                weather_df = None

            # Fetch from API if not cached
            if weather_df is None:
                request = DataFetchRequest(
                    site_id=site_id,
                    start_date=start_date.date(),
                    end_date=end_date.date(),
                    parameters={'latitude': lat, 'longitude': lon}
                )

                # Store coordinates for the weather source to use
                self.weather_source._site_coordinates = {site_id: (lat, lon)}

                weather_data = self.weather_source.fetch_daily_weather(request)

                # Convert list of DailyWeather to DataFrame
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

                # Cache weather data
                try:
                    cache_data = {
                        'site_id': site_id,
                        'latitude': lat,
                        'longitude': lon,
                        'start_date': start_date.strftime('%Y-%m-%d'),
                        'end_date': end_date.strftime('%Y-%m-%d'),
                        'data': weather_df.reset_index().to_dict('records')
                    }
                    # Convert date objects to strings
                    for rec in cache_data['data']:
                        if 'date' in rec:
                            rec['date'] = str(rec['date'])
                    with open(cache_file, 'w') as f:
                        json.dump(cache_data, f, default=str)
                    logger.info(f"✓ Cached weather for {site_id}")
                except Exception as e:
                    logger.warning(f"Failed to cache weather: {e}")

            # Satellite data (NDVI, LAI) from GEE
            # NOTE: This requires GEE authentication
            try:
                ndvi_data = self.satellite_source.fetch_ndvi(
                    lat=lat,
                    lon=lon,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )

                lai_data = self.satellite_source.fetch_lai(
                    lat=lat,
                    lon=lon,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )

                # Build satellite DataFrame
                satellite_records = []
                for obs in ndvi_data:
                    satellite_records.append({
                        'date': obs.date,
                        'ndvi': obs.value
                    })
                satellite_df = pd.DataFrame(satellite_records)
                satellite_df['date'] = pd.to_datetime(satellite_df['date'])

                # Add LAI
                lai_dict = {obs.date: obs.value for obs in lai_data}
                satellite_df['lai'] = satellite_df['date'].map(lai_dict)
                satellite_df = satellite_df.set_index('date')

                # Merge
                forcings = weather_df.join(satellite_df, how='left')
                # Forward fill satellite data (sparse temporal resolution)
                forcings['ndvi'] = forcings['ndvi'].ffill().bfill()
                forcings['lai'] = forcings['lai'].ffill().bfill()

                # Check if we have any satellite data
                if forcings['ndvi'].isna().all() or forcings['lai'].isna().all():
                    raise ValueError(
                        "No satellite data available for this location")

            except Exception as e:
                logger.warning(
                    f"Satellite data unavailable for {site_id}: {e}")
                logger.info(
                    f"Skipping station {site_id} due to missing satellite data")
                return  # Skip this station

            return forcings

        except Exception as e:
            logger.error(f"Failed to get forcing data: {e}")
            return None

    def _get_soil_parameters(self, station_data: ISMNStationData) -> Dict:
        """
        Extract soil parameters with multi-tier fallback:

        Priority 1: ISMN static variables (station-specific, most accurate)
        Priority 2: SoilGrids/iSDA at coordinates (250m-30m resolution)
        Priority 3: Regional defaults based on climate zone (last resort)

        Args:
            station_data: ISMNStationData with soil_properties

        Returns:
            Dict with soil texture and hydraulic parameters
        """
        station_id = station_data.station_id
        lat = station_data.latitude
        lon = station_data.longitude

        # =========================================================================
        # PRIORITY 1: ISMN Static Variables (Best)
        # =========================================================================
        if station_data.soil_properties:
            # Use surface layer (0-30 cm) if available
            layer_keys = sorted(station_data.soil_properties.keys())

            for layer_key in layer_keys:
                soil_props = station_data.soil_properties[layer_key]

                # Check if we have minimum required data (sand + clay)
                if soil_props.sand_fraction is not None and soil_props.clay_fraction is not None:
                    logger.info(
                        f"✓ Using ISMN soil properties for {station_id} "
                        f"(layer {layer_key}): "
                        f"Sand={soil_props.sand_fraction}%, Clay={soil_props.clay_fraction}%"
                    )

                    # Use tropical PTFs to derive hydraulic parameters
                    om_pct = soil_props.organic_carbon if soil_props.organic_carbon else 1.5
                    hydraulic_params = estimate_soil_parameters_tropical(
                        sand_percent=soil_props.sand_fraction,
                        clay_percent=soil_props.clay_fraction,
                        organic_matter_percent=om_pct,
                        bulk_density_g_cm3=1.4  # Could be improved with actual BD if available
                    )

                    return {
                        'source': 'ISMN_static',
                        'layer': layer_key,
                        'sand_pct': soil_props.sand_fraction,
                        'clay_pct': soil_props.clay_fraction,
                        'silt_pct': soil_props.silt_fraction if soil_props.silt_fraction else None,
                        'om_pct': om_pct,
                        'theta_sat': hydraulic_params.porosity,
                        'theta_fc': hydraulic_params.field_capacity,
                        'theta_pwp': hydraulic_params.wilting_point,
                        # cm/day to mm/day
                        'ksat_mm_day': hydraulic_params.saturated_hydraulic_conductivity_cm_day * 10,
                        'alpha': hydraulic_params.van_genuchten_alpha,
                        'n': hydraulic_params.van_genuchten_n
                    }

            logger.warning(
                f"⚠ ISMN soil properties exist for {station_id} but missing sand/clay. "
                f"Available layers: {list(station_data.soil_properties.keys())}"
            )

        # =========================================================================
        # PRIORITY 2: SoilGrids/iSDA Fallback (Good)
        # =========================================================================
        logger.info(
            f"→ ISMN soil data unavailable, fetching SoilGrids for {station_id}")

        try:
            from smps.data.sources.soilgrids import SoilGridsClient

            soilgrids = SoilGridsClient()
            soil_data = soilgrids.get_soil_properties(
                lat=lat,
                lon=lon,
                depths=['0-5cm', '5-15cm']  # Get surface layers
            )

            if soil_data and '0-5cm' in soil_data:
                layer_data = soil_data['0-5cm']

                # SoilGrids provides sand, clay, silt in g/kg (convert to %)
                sand_pct = layer_data.get('sand', 600) / 10  # g/kg → %
                clay_pct = layer_data.get('clay', 200) / 10
                silt_pct = layer_data.get('silt', 200) / 10
                oc_pct = layer_data.get('soc', 15) / 10  # Soil organic carbon

                logger.info(
                    f"✓ Using SoilGrids data for {station_id}: "
                    f"Sand={sand_pct:.1f}%, Clay={clay_pct:.1f}%"
                )

                # Use tropical PTFs
                hydraulic_params = estimate_soil_parameters_tropical(
                    sand_percent=sand_pct,
                    clay_percent=clay_pct,
                    organic_matter_percent=oc_pct,
                    bulk_density_g_cm3=1.4
                )

                return {
                    'source': 'SoilGrids_250m',
                    'sand_pct': sand_pct,
                    'clay_pct': clay_pct,
                    'silt_pct': silt_pct,
                    'om_pct': oc_pct,
                    'theta_sat': hydraulic_params.porosity,
                    'theta_fc': hydraulic_params.field_capacity,
                    'theta_pwp': hydraulic_params.wilting_point,
                    # cm/day to mm/day
                    'ksat_mm_day': hydraulic_params.saturated_hydraulic_conductivity_cm_day * 10,
                    'alpha': hydraulic_params.van_genuchten_alpha,
                    'n': hydraulic_params.van_genuchten_n
                }

        except ImportError:
            logger.warning("SoilGridsClient not available (import failed)")
        except Exception as e:
            logger.warning(f"SoilGrids fetch failed for {station_id}: {e}")

        # =========================================================================
        # PRIORITY 3: Regional Defaults (Last Resort) - DISABLED
        # =========================================================================
        logger.warning(
            f"⚠ No reliable soil data available for {station_id} at ({lat:.3f}, {lon:.3f}). "
            f"Skipping station due to insufficient data quality."
        )
        return None

    def _regional_soil_defaults(self, lat: float, lon: float) -> Dict:
        """
        Regional soil parameter defaults for Africa.

        Based on FAO soil maps, climate zones, and typical soil types.
        Should only be used when ISMN and SoilGrids data are unavailable.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Dict with soil parameters including source metadata
        """
        region = self._classify_soil_region(lat, lon)

        # Regional defaults based on literature and FAO soil maps
        regional_params = {
            'sahel': {
                'name': 'Sahel - Arenosols/Lixisols',
                'description': 'Sandy soils with low organic matter',
                'countries': 'Mauritania, Mali, Niger, Chad',
                'sand_pct': 75,
                'clay_pct': 10,
                'om_pct': 0.8,
                'theta_sat': 0.41,
                'theta_fc': 0.18,
                'theta_pwp': 0.06,
                'ksat_mm_day': 800,
                'alpha': 0.04,
                'n': 1.6
            },
            'sudanian': {
                'name': 'Sudanian - Lixisols/Luvisols',
                'description': 'Loamy sand to sandy loam, moderate OM',
                'countries': 'Senegal, Ghana, Benin, Nigeria, Sudan',
                'sand_pct': 60,
                'clay_pct': 15,
                'om_pct': 1.5,
                'theta_sat': 0.43,
                'theta_fc': 0.22,
                'theta_pwp': 0.08,
                'ksat_mm_day': 500,
                'alpha': 0.03,
                'n': 1.5
            },
            'guinean': {
                'name': 'Guinean - Acrisols/Ferralsols',
                'description': 'Sandy clay loam, higher OM, weathered',
                'countries': 'Coastal West Africa, Guinea',
                'sand_pct': 45,
                'clay_pct': 25,
                'om_pct': 2.2,
                'theta_sat': 0.46,
                'theta_fc': 0.28,
                'theta_pwp': 0.12,
                'ksat_mm_day': 250,
                'alpha': 0.02,
                'n': 1.4
            },
            'east_highlands': {
                'name': 'East African Highlands - Nitisols/Andosols',
                'description': 'Clay loam to clay, volcanic influence, high OM',
                'countries': 'Kenya, Ethiopia, Rwanda, Tanzania highlands',
                'sand_pct': 35,
                'clay_pct': 32,
                'om_pct': 2.8,
                'theta_sat': 0.48,
                'theta_fc': 0.32,
                'theta_pwp': 0.15,
                'ksat_mm_day': 150,
                'alpha': 0.015,
                'n': 1.3
            },
            'east_lowlands': {
                'name': 'East African Lowlands - Cambisols/Vertisols',
                'description': 'Variable texture, often clay-rich in valleys',
                'countries': 'Kenya lowlands, Tanzania coast',
                'sand_pct': 40,
                'clay_pct': 28,
                'om_pct': 1.8,
                'theta_sat': 0.47,
                'theta_fc': 0.30,
                'theta_pwp': 0.14,
                'ksat_mm_day': 200,
                'alpha': 0.018,
                'n': 1.35
            },
            'humid_forest': {
                'name': 'Humid Tropical Forest - Ferralsols/Acrisols',
                'description': 'Highly weathered clays, high OM in topsoil',
                'countries': 'Congo Basin, Cameroon, Gabon',
                'sand_pct': 25,
                'clay_pct': 45,
                'om_pct': 3.5,
                'theta_sat': 0.52,
                'theta_fc': 0.38,
                'theta_pwp': 0.20,
                'ksat_mm_day': 80,
                'alpha': 0.01,
                'n': 1.2
            },
            'southern_savanna': {
                'name': 'Southern Savanna - Arenosols/Lixisols',
                'description': 'Sandy loam to loam, moderate drainage',
                'countries': 'Zimbabwe, Zambia, Mozambique',
                'sand_pct': 55,
                'clay_pct': 18,
                'om_pct': 1.2,
                'theta_sat': 0.44,
                'theta_fc': 0.24,
                'theta_pwp': 0.10,
                'ksat_mm_day': 400,
                'alpha': 0.025,
                'n': 1.4
            },
            'southern_arid': {
                'name': 'Southern Arid - Arenosols/Calcisols',
                'description': 'Deep sands, very low OM',
                'countries': 'Kalahari (Botswana, Namibia)',
                'sand_pct': 85,
                'clay_pct': 5,
                'om_pct': 0.5,
                'theta_sat': 0.38,
                'theta_fc': 0.14,
                'theta_pwp': 0.04,
                'ksat_mm_day': 1200,
                'alpha': 0.05,
                'n': 1.7
            },
            'mediterranean': {
                'name': 'Mediterranean - Calcisols/Cambisols',
                'description': 'Loam to clay loam, moderate OM',
                'countries': 'North Africa coast, South Africa Cape',
                'sand_pct': 42,
                'clay_pct': 22,
                'om_pct': 1.8,
                'theta_sat': 0.45,
                'theta_fc': 0.26,
                'theta_pwp': 0.11,
                'ksat_mm_day': 300,
                'alpha': 0.022,
                'n': 1.38
            }
        }

        params = regional_params.get(region, regional_params['sudanian'])

        logger.info(
            f"Using regional defaults: {params['name']} - {params['description']}"
        )

        return {
            'source': f'Regional_default_{region}',
            'region': region,
            'region_name': params['name'],
            'description': params['description'],
            'sand_pct': params['sand_pct'],
            'clay_pct': params['clay_pct'],
            'om_pct': params['om_pct'],
            'theta_sat': params['theta_sat'],
            'theta_fc': params['theta_fc'],
            'theta_pwp': params['theta_pwp'],
            'ksat_mm_day': params['ksat_mm_day'],
            'alpha': params['alpha'],
            'n': params['n']
        }

    def _classify_soil_region(self, lat: float, lon: float) -> str:
        """
        Classify soil region based on coordinates and climate zones.

        Uses lat/lon as proxy for climate-soil relationships in Africa.
        Based on FAO soil maps and Köppen climate classification.

        Args:
            lat: Latitude (-35 to 37 for Africa)
            lon: Longitude (-18 to 52 for Africa)

        Returns:
            Region code (e.g., 'sahel', 'east_highlands')
        """
        # Sahel belt: 12-18°N, across West-Central Africa
        if 12 <= lat <= 18 and -18 <= lon <= 35:
            return 'sahel'

        # Sudanian zone: 8-12°N, West-Central Africa
        if 8 <= lat <= 12 and -18 <= lon <= 35:
            return 'sudanian'

        # Guinean zone: 4-8°N, coastal West Africa
        if 4 <= lat <= 8 and -18 <= lon <= 15:
            return 'guinean'

        # East African highlands: Elevated areas, use lat/lon box
        # Kenya highlands: -1 to 2°N, 35-38°E
        # Ethiopian highlands: 5-15°N, 35-42°E
        # Tanzania highlands: -10 to -2°S, 30-37°E
        if ((-1 <= lat <= 2 and 35 <= lon <= 38) or  # Kenya
            (5 <= lat <= 15 and 35 <= lon <= 42) or   # Ethiopia
                (-10 <= lat <= -2 and 30 <= lon <= 37)):  # Tanzania
            return 'east_highlands'

        # East African lowlands
        if -12 <= lat <= 5 and 30 <= lon <= 42:
            return 'east_lowlands'

        # Humid tropical forest: Congo Basin + coastal
        if -5 <= lat <= 5 and 8 <= lon <= 30:
            return 'humid_forest'

        # Southern African savanna: 10-20°S
        if -20 <= lat <= -10 and 20 <= lon <= 35:
            return 'southern_savanna'

        # Southern arid (Kalahari): 20-28°S
        if -28 <= lat <= -20 and 15 <= lon <= 28:
            return 'southern_arid'

        # Mediterranean (North Africa coast, South Africa Cape)
        if ((30 <= lat <= 37 and -10 <= lon <= 35) or  # North
                (-35 <= lat <= -30 and 15 <= lon <= 25)):   # South Cape
            return 'mediterranean'

        # Default to Sudanian (most common, middle-of-road properties)
        return 'sudanian'

    def _run_physics_model(
        self,
        forcings: pd.DataFrame,
        soil_params: Dict
    ) -> Optional[pd.DataFrame]:
        """Run EnhancedWaterBalance model and return results DataFrame."""
        try:
            from smps.physics.soil_hydraulics import VanGenuchtenParameters

            # Create VG parameters from soil properties
            vg_params = VanGenuchtenParameters(
                theta_r=0.05,  # Residual water content
                theta_s=soil_params['theta_sat'],
                alpha=soil_params['alpha'],
                n=soil_params['n'],
                K_sat=soil_params['ksat_mm_day'] /
                1000  # Convert mm/day to m/day
            )

            # Configure model with 5 layers
            config = EnhancedModelParameters(
                n_layers=5,
                # 10, 20, 20, 25, 25 cm
                layer_depths_m=[0.10, 0.20, 0.20, 0.25, 0.25],
                # Same params for all layers
                vg_params=[vg_params, vg_params,
                           vg_params, vg_params, vg_params],
                crop_type="grassland",  # Natural grassland vegetation for ISMN validation
                use_green_ampt=True,
                use_fao56_dual=True,
                use_feddes_uptake=True,
                use_darcy_flux=True
            )

            model = EnhancedWaterBalance(config)

            # Prepare forcings - map column names to what model expects
            run_forcings = forcings.copy()

            # Column mapping from weather data to model inputs
            # EnhancedWaterBalance.run_period expects: precipitation_mm, et0_mm, ndvi, temperature_max_c
            column_mapping = {
                'precipitation': 'precipitation_mm',  # Keep as _mm suffix for run_period
                'temperature_mean_c': 'temperature',
                'temperature_max_c': 'temperature_max_c',
                'relative_humidity_mean': 'humidity',
                'wind_speed_mean_m_s': 'wind_speed',
                'et0': 'et0_mm',  # Keep as _mm suffix for run_period
                'solar_radiation_mj_m2': 'solar_radiation'
            }

            for old_col, new_col in column_mapping.items():
                if old_col in run_forcings.columns:
                    run_forcings[new_col] = run_forcings[old_col]

            # Check required columns
            required_cols = ['precipitation_mm',
                             'temperature', 'ndvi', 'et0_mm']
            missing_cols = [
                col for col in required_cols if col not in run_forcings.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing required forcing columns: {missing_cols}")

            # Run simulation using run_period method
            results = model.run_period(
                forcings=run_forcings,
                warmup_days=0,  # Use all data
                return_fluxes=False
            )

            # Add dates from forcings index
            results['date'] = forcings.index[:len(results)]

            return results

        except Exception as e:
            logger.error(f"Model run failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_model_predictions(
        self,
        model_results: pd.DataFrame,
        dates: pd.DatetimeIndex,
        target_depths_cm: Optional[List[float]] = None
    ) -> Dict[float, pd.Series]:
        """
        Extract model predictions by depth from run results with interpolation.

        Uses linear interpolation to match predictions to actual sensor depths.
        Model layers (default configuration):
        - Layer 0: 0-10 cm (center: 5 cm)
        - Layer 1: 10-30 cm (center: 20 cm)
        - Layer 2: 30-50 cm (center: 40 cm)
        - Layer 3: 50-75 cm (center: 62.5 cm)
        - Layer 4: 75-100 cm (center: 87.5 cm)
        """
        from scipy import interpolate as scipy_interp

        predictions = {}
        n_results = len(model_results)
        valid_dates = dates[:n_results]

        # Model layer configuration (center depths in cm)
        layer_centers = [5, 20, 40, 62.5, 87.5]  # Default 5-layer config

        # Extract layer values
        layer_cols = [
            col for col in model_results.columns if col.startswith('theta_layer_')]
        if not layer_cols:
            # Fallback to theta_phys columns
            if 'theta_phys_surface' in model_results.columns:
                predictions[10] = pd.Series(
                    model_results['theta_phys_surface'].values, index=valid_dates)
            if 'theta_phys_root' in model_results.columns:
                predictions[20] = pd.Series(
                    model_results['theta_phys_root'].values, index=valid_dates)
            return predictions

        # Sort layer columns numerically
        layer_cols = sorted(layer_cols, key=lambda x: int(x.split('_')[-1]))
        n_layers = len(layer_cols)
        layer_centers = layer_centers[:n_layers]

        # Default target depths if not specified
        if target_depths_cm is None:
            target_depths_cm = [5, 10, 20, 25, 30,
                                40, 50, 55, 60, 85, 100, 115, 120]

        # Interpolate to each target depth
        for target_depth in target_depths_cm:
            interp_values = []

            for i in range(n_results):
                layer_values = [model_results[col].iloc[i]
                                for col in layer_cols]

                # Handle edge cases
                if target_depth <= layer_centers[0]:
                    interp_values.append(layer_values[0])
                elif target_depth >= layer_centers[-1]:
                    interp_values.append(layer_values[-1])
                else:
                    # Linear interpolation
                    f = scipy_interp.interp1d(
                        layer_centers, layer_values, kind='linear')
                    interp_values.append(float(f(target_depth)))

            predictions[target_depth] = pd.Series(
                interp_values, index=valid_dates)

        logger.info(
            f"  → Interpolated predictions to depths: {target_depths_cm} cm")
        return predictions

    def _get_observations(
        self,
        station_data: ISMNStationData
    ) -> Dict[float, pd.Series]:
        """Get ISMN observations organized by depth."""
        if station_data.daily_data is None:
            return {}

        observations = {}
        for depth_cm in station_data.available_depths_cm:
            df = get_daily_soil_moisture(station_data, depth_cm=depth_cm)
            if not df.empty:
                series = df.set_index('date')['soil_moisture_mean']
                observations[depth_cm] = series

        return observations

    def _compute_and_store_metrics(
        self,
        station_data: ISMNStationData,
        observations: Dict[float, pd.Series],
        predictions: Dict[float, pd.Series],
        forcings: pd.DataFrame
    ):
        """Compute validation metrics and store results, including seasonal and temporal horizon breakdown."""

        lai_mean = float(pd.to_numeric(forcings.get('lai'), errors='coerce').mean(
        )) if forcings is not None and 'lai' in forcings.columns else float('nan')
        ndvi_mean = float(pd.to_numeric(forcings.get('ndvi'), errors='coerce').mean(
        )) if forcings is not None and 'ndvi' in forcings.columns else float('nan')

        # Observation QC: extremely low-variance series can yield pathological NSE/KGE.
        # This usually indicates a stuck sensor, heavy smoothing, or a depth that is effectively constant.
        LOW_VARIANCE_STD_THRESHOLD = 1e-3  # m3/m3

        # Define forecast horizons (in days)
        HORIZONS = {
            '24h': 1,   # 1-day ahead
            '72h': 3,   # 3-day ahead
            '168h': 7   # 7-day ahead (weekly)
        }

        # For each depth with both obs and pred
        for depth_cm in observations.keys():
            closest_pred_depth = min(
                predictions.keys(), key=lambda x: abs(x - depth_cm))

            obs = observations[depth_cm]
            pred = predictions[closest_pred_depth]

            # Align time series
            aligned = pd.DataFrame({
                'obs': obs,
                'pred': pred
            }).dropna()

            obs_mean = float(aligned['obs'].mean())
            obs_std = float(aligned['obs'].std())
            pred_mean = float(aligned['pred'].mean())
            pred_std = float(aligned['pred'].std())
            obs_min = float(aligned['obs'].min())
            obs_max = float(aligned['obs'].max())
            pred_min = float(aligned['pred'].min())
            pred_max = float(aligned['pred'].max())

            is_low_variance_obs = bool(np.isfinite(
                obs_std) and obs_std < LOW_VARIANCE_STD_THRESHOLD)
            if is_low_variance_obs:
                logger.warning(
                    f"Low-variance observation series for {station_data.station_id} at {depth_cm} cm: "
                    f"obs_std={obs_std:.6f} m3/m3 (metrics may be misleading)"
                )

            if len(aligned) < 30:
                logger.warning(
                    f"Insufficient overlap for {station_data.station_id} at {depth_cm} cm")
                continue

            # Export paired obs/pred for scatter-fit evaluation (same-day + horizons)
            if self.export_pairs:
                # Same-day pairs (0h)
                base_pairs = aligned[['pred', 'obs']].copy()
                base_pairs = base_pairs.dropna()
                for pred_date, row in base_pairs.iterrows():
                    self.pairs_records.append({
                        'station_id': station_data.station_id,
                        'network': station_data.network,
                        'station': station_data.station,
                        'latitude': station_data.latitude,
                        'longitude': station_data.longitude,
                        'depth_cm': float(depth_cm),
                        'pred_depth_cm': float(closest_pred_depth),
                        'horizon_name': '0h',
                        'horizon_days': 0,
                        'pred_date': pred_date,
                        'obs_date': pred_date,
                        'pred': float(row['pred']),
                        'obs': float(row['obs']),
                        'lai_mean': lai_mean,
                        'ndvi_mean': ndvi_mean,
                    })

            # Compute overall metrics (instantaneous / same-day)
            metrics_result = run_physics_validation(
                obs=aligned['obs'].values,
                pred=aligned['pred'].values
            )

            # Get metrics dict from standard_metrics
            metrics_dict = {}
            if metrics_result.standard_metrics is not None:
                metrics_dict = metrics_result.standard_metrics.to_dict()
            metrics_dict['overall_score'] = metrics_result.overall_score
            # Treat low-variance observations as a data-quality failure (avoid declaring "pass" on junk series)
            metrics_dict['passes_validation'] = (
                metrics_result.passes_validation and not is_low_variance_obs
            )

            # =====================================================================
            # TEMPORAL HORIZON ANALYSIS (24h, 72h, 168h)
            # =====================================================================
            # For each horizon, we compare predictions at time t with observations at time t+horizon
            # This simulates forecast skill at different lead times
            horizon_metrics = {}

            for horizon_name, horizon_days in HORIZONS.items():
                # Shift predictions forward (or equivalently, shift obs backward)
                # pred[t] compared to obs[t + horizon_days]
                obs_shifted = aligned['obs'].shift(-horizon_days)

                horizon_aligned = pd.DataFrame({
                    'obs': obs_shifted,
                    'pred': aligned['pred']
                }).dropna()

                if self.export_pairs and len(horizon_aligned) > 0:
                    for pred_date, row in horizon_aligned.iterrows():
                        obs_date = pred_date + \
                            pd.Timedelta(days=int(horizon_days))
                        self.pairs_records.append({
                            'station_id': station_data.station_id,
                            'network': station_data.network,
                            'station': station_data.station,
                            'latitude': station_data.latitude,
                            'longitude': station_data.longitude,
                            'depth_cm': float(depth_cm),
                            'pred_depth_cm': float(closest_pred_depth),
                            'horizon_name': horizon_name,
                            'horizon_days': int(horizon_days),
                            'pred_date': pred_date,
                            'obs_date': obs_date,
                            'pred': float(row['pred']),
                            'obs': float(row['obs']),
                            'lai_mean': lai_mean,
                            'ndvi_mean': ndvi_mean,
                        })

                if len(horizon_aligned) >= 20:
                    horizon_result = run_physics_validation(
                        obs=horizon_aligned['obs'].values,
                        pred=horizon_aligned['pred'].values
                    )
                    if horizon_result.standard_metrics is not None:
                        horizon_metrics[f'RMSE_{horizon_name}'] = horizon_result.standard_metrics.rmse
                        horizon_metrics[f'KGE_{horizon_name}'] = horizon_result.standard_metrics.kge
                        horizon_metrics[f'NSE_{horizon_name}'] = horizon_result.standard_metrics.nse
                        horizon_metrics[f'R2_{horizon_name}'] = horizon_result.standard_metrics.r_squared
                        horizon_metrics[f'n_days_{horizon_name}'] = len(
                            horizon_aligned)

            # =====================================================================
            # SEASONAL ANALYSIS
            # =====================================================================
            # Determine hemisphere for season classification
            lat = station_data.latitude
            is_northern = lat >= 0

            # Classify seasons based on month
            # Northern Hemisphere: Wet = May-Oct (JJAS monsoon), Dry = Nov-Apr
            # Southern Hemisphere: Wet = Nov-Apr, Dry = May-Oct
            # Tropical Africa (10S-10N): Use rainfall patterns - typically bimodal
            aligned_with_season = aligned.copy()
            aligned_with_season['month'] = aligned_with_season.index.month

            if abs(lat) < 10:
                # Tropical: bimodal rainfall - Mar-May (long rains), Oct-Dec (short rains)
                def classify_tropical_season(month):
                    if month in [3, 4, 5, 10, 11]:  # Wet seasons
                        return 'wet'
                    elif month in [1, 2, 6, 7, 8, 9, 12]:  # Dry seasons
                        return 'dry'
                    return 'transition'
                aligned_with_season['season'] = aligned_with_season['month'].apply(
                    classify_tropical_season)
            elif is_northern:
                # Northern: June-Sept wet (monsoon), Dec-Feb dry
                def classify_north_season(month):
                    if month in [6, 7, 8, 9]:
                        return 'wet'
                    elif month in [12, 1, 2, 3]:
                        return 'dry'
                    return 'transition'
                aligned_with_season['season'] = aligned_with_season['month'].apply(
                    classify_north_season)
            else:
                # Southern: Dec-Feb wet, June-Aug dry
                def classify_south_season(month):
                    if month in [11, 12, 1, 2, 3]:
                        return 'wet'
                    elif month in [5, 6, 7, 8]:
                        return 'dry'
                    return 'transition'
                aligned_with_season['season'] = aligned_with_season['month'].apply(
                    classify_south_season)

            # Compute seasonal metrics
            seasonal_metrics = {}
            for season in ['wet', 'dry', 'transition']:
                season_data = aligned_with_season[aligned_with_season['season'] == season]
                if len(season_data) >= 20:  # Minimum samples for reliable metrics
                    season_result = run_physics_validation(
                        obs=season_data['obs'].values,
                        pred=season_data['pred'].values
                    )
                    if season_result.standard_metrics is not None:
                        seasonal_metrics[f'RMSE_{season}'] = season_result.standard_metrics.rmse
                        seasonal_metrics[f'KGE_{season}'] = season_result.standard_metrics.kge
                        seasonal_metrics[f'NSE_{season}'] = season_result.standard_metrics.nse
                        seasonal_metrics[f'n_days_{season}'] = len(season_data)

            # Store result with seasonal and horizon breakdown
            result = {
                'station_id': station_data.station_id,
                'network': station_data.network,
                'station': station_data.station,
                'latitude': station_data.latitude,
                'longitude': station_data.longitude,
                'depth_cm': depth_cm,
                'pred_depth_cm': closest_pred_depth,
                'n_days': len(aligned),
                'start_date': aligned.index.min(),
                'end_date': aligned.index.max(),
                'obs_mean': obs_mean,
                'obs_std': obs_std,
                'obs_min': obs_min,
                'obs_max': obs_max,
                'pred_mean': pred_mean,
                'pred_std': pred_std,
                'pred_min': pred_min,
                'pred_max': pred_max,
                'obs_low_variance_flag': is_low_variance_obs,
                **metrics_dict,
                **horizon_metrics,
                **seasonal_metrics
            }

            self.results.append(result)

            # Log with horizon info
            h24_kge = horizon_metrics.get('KGE_24h', float('nan'))
            h72_kge = horizon_metrics.get('KGE_72h', float('nan'))
            h168_kge = horizon_metrics.get('KGE_168h', float('nan'))
            logger.info(
                f"  {depth_cm} cm: RMSE={result['RMSE']:.4f}, KGE={result['KGE']:.3f} | "
                f"24h={h24_kge:.3f}, 72h={h72_kge:.3f}, 168h={h168_kge:.3f}")

    def _save_results(self):
        """Save validation results to files with detailed station-by-station reporting."""
        if not self.results:
            logger.warning("No results to save")
            return

        df = pd.DataFrame(self.results)

        # If run_name is provided, write a single consistent set of files (no timestamps)
        # into output_dir/<run_name>/ to avoid accumulating many confusing CSVs.
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        use_fixed_filenames = bool(self.run_name)

        # Save detailed results as CSV
        csv_path = (
            self.output_dir / "ismn_validation_results.csv"
            if use_fixed_filenames
            else self.output_dir / f"ismn_validation_results_{timestamp}.csv"
        )
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved results to {csv_path}")

        # Save paired obs/sim records for scatter-fit plots
        if self.export_pairs and self.pairs_records:
            pairs_df = pd.DataFrame(self.pairs_records)
            pairs_path = (
                self.output_dir / "paired_obs_sim.csv"
                if use_fixed_filenames
                else self.output_dir / f"paired_obs_sim_{timestamp}.csv"
            )
            pairs_df.to_csv(pairs_path, index=False)
            logger.info(f"Saved paired obs/sim series to {pairs_path}")

        # Save as Parquet if available (better for large datasets)
        try:
            parquet_path = (
                self.output_dir / "ismn_validation_results.parquet"
                if use_fixed_filenames
                else self.output_dir / f"ismn_validation_results_{timestamp}.parquet"
            )
            df.to_parquet(parquet_path, index=False)
            logger.info(f"Saved results to {parquet_path}")
        except ImportError:
            logger.info("Parquet export skipped (pyarrow not installed)")

        # =====================================================================
        # Station-by-Station Summary (Multi-depth)
        # =====================================================================
        station_summary = []
        for station_id in df['station_id'].unique():
            station_df = df[df['station_id'] == station_id]
            row = station_df.iloc[0]

            n_depths_total = len(station_df)
            n_depths_passed = int(station_df['passes_validation'].sum(
            )) if 'passes_validation' in station_df.columns else 0
            depth_pass_rate = (n_depths_passed /
                               n_depths_total) if n_depths_total else 0.0
            n_low_variance = int(station_df.get('obs_low_variance_flag', False).sum(
            )) if 'obs_low_variance_flag' in station_df.columns else 0

            station_row = {
                'station_id': station_id,
                'network': row['network'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'n_depths': len(station_df),
                'n_depths_passed': n_depths_passed,
                'depth_pass_rate': round(depth_pass_rate, 3),
                'n_low_variance_depths': n_low_variance,
                'depths_cm': ','.join([str(int(d)) for d in sorted(station_df['depth_cm'].unique())]),
                'total_obs_days': station_df['n_days'].sum(),
                'date_range': f"{station_df['start_date'].min()} to {station_df['end_date'].max()}",
            }

            # Aggregate metrics across depths
            for metric in ['RMSE', 'MAE', 'NSE', 'KGE', 'R²']:
                if metric in station_df.columns:
                    station_row[f'{metric}_mean'] = station_df[metric].mean()
                    station_row[f'{metric}_best'] = station_df[metric].max(
                    ) if metric in ['NSE', 'KGE', 'R²'] else station_df[metric].min()

            # Per-depth metrics
            for _, depth_row in station_df.iterrows():
                depth = int(depth_row['depth_cm'])
                station_row[f'RMSE_{depth}cm'] = depth_row.get('RMSE')
                station_row[f'KGE_{depth}cm'] = depth_row.get('KGE')
                station_row[f'NSE_{depth}cm'] = depth_row.get('NSE')

            # For stations with multiple depths, keep these separate so it's clear when only some depths are good.
            station_row['passes_any_depth'] = bool(
                station_df['passes_validation'].any())
            station_row['passes_all_depths'] = bool(
                station_df['passes_validation'].all())
            # Back-compat: keep the original column name, but make it explicit and stricter than "any".
            station_row['passes_validation'] = bool(depth_pass_rate >= 0.5)
            station_summary.append(station_row)

        station_summary_df = pd.DataFrame(station_summary)
        station_summary_path = (
            self.output_dir / "station_summary.csv"
            if use_fixed_filenames
            else self.output_dir / f"station_summary_{timestamp}.csv"
        )
        station_summary_df.to_csv(station_summary_path, index=False)
        logger.info(f"Saved station summary to {station_summary_path}")

        # =====================================================================
        # Depth-wise Summary
        # =====================================================================
        depth_summary = df.groupby('depth_cm').agg({
            'RMSE': ['mean', 'std', 'min', 'max', 'count'],
            'MAE': ['mean', 'std'],
            'NSE': ['mean', 'std', 'min', 'max'],
            'KGE': ['mean', 'std', 'min', 'max'],
            'n_days': 'sum'
        }).round(4)
        depth_summary.columns = ['_'.join(col).strip()
                                 for col in depth_summary.columns.values]
        depth_summary_path = (
            self.output_dir / "depth_summary.csv"
            if use_fixed_filenames
            else self.output_dir / f"depth_summary_{timestamp}.csv"
        )
        depth_summary.to_csv(depth_summary_path)
        logger.info(f"Saved depth summary to {depth_summary_path}")

        # =====================================================================
        # Print Console Summary
        # =====================================================================
        print("\n" + "=" * 80)
        print("ISMN VALIDATION RESULTS - STATION SUMMARY")
        print("=" * 80)

        # Sort by KGE_mean descending
        if 'KGE_mean' in station_summary_df.columns:
            station_summary_df = station_summary_df.sort_values(
                'KGE_mean', ascending=False)

        print(
            f"\n{'Station':<45} {'Network':<12} {'Depths':<12} {'RMSE':<8} {'KGE':<8} {'Pass':<5}")
        print("-" * 90)

        for _, row in station_summary_df.iterrows():
            station_name = row['station_id'][:44]
            rmse = f"{row.get('RMSE_mean', 0):.4f}" if pd.notna(
                row.get('RMSE_mean')) else "N/A"
            kge = f"{row.get('KGE_mean', 0):.3f}" if pd.notna(
                row.get('KGE_mean')) else "N/A"
            passed = "✓" if row.get('passes_validation') else "✗"
            print(
                f"{station_name:<45} {row['network']:<12} {row['depths_cm']:<12} {rmse:<8} {kge:<8} {passed:<5}")

        print("-" * 90)

        # Overall statistics
        n_stations = len(station_summary_df)
        n_passed = station_summary_df['passes_validation'].sum()
        print(f"\nTotal Stations: {n_stations}")
        print(
            f"Stations Passed: {n_passed}/{n_stations} ({100*n_passed/n_stations:.1f}%)")
        print(f"\nOverall Metrics (mean ± std):")
        print(f"  RMSE: {df['RMSE'].mean():.4f} ± {df['RMSE'].std():.4f}")
        print(f"  MAE:  {df['MAE'].mean():.4f} ± {df['MAE'].std():.4f}")
        print(f"  KGE:  {df['KGE'].mean():.3f} ± {df['KGE'].std():.3f}")
        print(f"  NSE:  {df['NSE'].mean():.3f} ± {df['NSE'].std():.3f}")

        # =====================================================================
        # Temporal Horizon Analysis Summary (24h, 72h, 168h)
        # =====================================================================
        print(f"\nBy Forecast Horizon:")
        print(f"  {'Horizon':<10} {'RMSE':<10} {'KGE':<10} {'NSE':<10} {'R²':<10}")
        print(f"  {'-'*50}")

        # Same-day (0h) - baseline
        print(
            f"  {'0h (now)':<10} {df['RMSE'].mean():.4f}     {df['KGE'].mean():.4f}     {df['NSE'].mean():.4f}     {df['R²'].mean():.4f}")

        for horizon in ['24h', '72h', '168h']:
            rmse_col = f'RMSE_{horizon}'
            kge_col = f'KGE_{horizon}'
            nse_col = f'NSE_{horizon}'
            r2_col = f'R2_{horizon}'

            if rmse_col in df.columns:
                horizon_data = df[df[rmse_col].notna()]
                if len(horizon_data) > 0:
                    rmse_mean = horizon_data[rmse_col].mean()
                    kge_mean = horizon_data[kge_col].mean()
                    nse_mean = horizon_data[nse_col].mean(
                    ) if nse_col in horizon_data.columns else float('nan')
                    r2_mean = horizon_data[r2_col].mean(
                    ) if r2_col in horizon_data.columns else float('nan')
                    print(
                        f"  {horizon:<10} {rmse_mean:.4f}     {kge_mean:.4f}     {nse_mean:.4f}     {r2_mean:.4f}")

        # By depth
        print(f"\nBy Depth:")
        for depth in sorted(df['depth_cm'].unique()):
            depth_df = df[df['depth_cm'] == depth]
            print(
                f"  {int(depth):>3} cm: RMSE={depth_df['RMSE'].mean():.4f}, KGE={depth_df['KGE'].mean():.3f}, n={len(depth_df)}")

        # =====================================================================
        # Seasonal Analysis Summary
        # =====================================================================
        print(f"\nBy Season:")
        for season in ['wet', 'dry', 'transition']:
            rmse_col = f'RMSE_{season}'
            kge_col = f'KGE_{season}'
            n_col = f'n_days_{season}'

            if rmse_col in df.columns:
                season_data = df[df[rmse_col].notna()]
                if len(season_data) > 0:
                    rmse_mean = season_data[rmse_col].mean()
                    kge_mean = season_data[kge_col].mean()
                    total_days = season_data[n_col].sum()
                    print(
                        f"  {season.capitalize():>12}: RMSE={rmse_mean:.4f}, KGE={kge_mean:.3f}, n_days={int(total_days)}")

        # Save horizon summary
        horizon_summary = []
        for horizon in ['0h', '24h', '72h', '168h']:
            if horizon == '0h':
                horizon_summary.append({
                    'horizon': horizon,
                    'horizon_days': 0,
                    'RMSE_mean': df['RMSE'].mean(),
                    'RMSE_std': df['RMSE'].std(),
                    'KGE_mean': df['KGE'].mean(),
                    'KGE_std': df['KGE'].std(),
                    'NSE_mean': df['NSE'].mean(),
                    'R2_mean': df['R²'].mean(),
                    'n_observations': len(df)
                })
            else:
                rmse_col = f'RMSE_{horizon}'
                if rmse_col in df.columns:
                    horizon_data = df[df[rmse_col].notna()]
                    if len(horizon_data) > 0:
                        horizon_summary.append({
                            'horizon': horizon,
                            'horizon_days': {'24h': 1, '72h': 3, '168h': 7}[horizon],
                            'RMSE_mean': horizon_data[rmse_col].mean(),
                            'RMSE_std': horizon_data[rmse_col].std(),
                            'KGE_mean': horizon_data[f'KGE_{horizon}'].mean(),
                            'KGE_std': horizon_data[f'KGE_{horizon}'].std(),
                            'NSE_mean': horizon_data[f'NSE_{horizon}'].mean() if f'NSE_{horizon}' in horizon_data.columns else None,
                            'R2_mean': horizon_data[f'R2_{horizon}'].mean() if f'R2_{horizon}' in horizon_data.columns else None,
                            'n_observations': len(horizon_data)
                        })

        if horizon_summary:
            horizon_df = pd.DataFrame(horizon_summary)
            horizon_path = (
                self.output_dir / "horizon_summary.csv"
                if use_fixed_filenames
                else self.output_dir / f"horizon_summary_{timestamp}.csv"
            )
            horizon_df.to_csv(horizon_path, index=False)
            logger.info(f"Saved horizon summary to {horizon_path}")

        # Save seasonal summary
        seasonal_summary = []
        for season in ['wet', 'dry', 'transition']:
            rmse_col = f'RMSE_{season}'
            kge_col = f'KGE_{season}'
            nse_col = f'NSE_{season}'
            n_col = f'n_days_{season}'

            if rmse_col in df.columns:
                season_data = df[df[rmse_col].notna()]
                if len(season_data) > 0:
                    seasonal_summary.append({
                        'season': season,
                        'RMSE_mean': season_data[rmse_col].mean(),
                        'RMSE_std': season_data[rmse_col].std(),
                        'KGE_mean': season_data[kge_col].mean(),
                        'KGE_std': season_data[kge_col].std(),
                        'NSE_mean': season_data[nse_col].mean() if nse_col in season_data.columns else None,
                        'n_observations': len(season_data),
                        'total_days': season_data[n_col].sum()
                    })

        if seasonal_summary:
            seasonal_df = pd.DataFrame(seasonal_summary)
            seasonal_path = (
                self.output_dir / "seasonal_summary.csv"
                if use_fixed_filenames
                else self.output_dir / f"seasonal_summary_{timestamp}.csv"
            )
            seasonal_df.to_csv(seasonal_path, index=False)
            logger.info(f"Saved seasonal summary to {seasonal_path}")

        print("=" * 80 + "\n")

        # Legacy network summary (keep for backwards compatibility)
        summary = df.groupby('network').agg({
            'RMSE': ['mean', 'std', 'min', 'max'],
            'NSE': ['mean', 'std', 'min', 'max'],
            'KGE': ['mean', 'std', 'min', 'max'],
            'MAE': ['mean', 'std', 'min', 'max'],
            'R²': ['mean', 'std', 'min', 'max'],
            'station_id': 'count'
        })
        summary_path = self.output_dir / "validation_summary.csv"
        summary.to_csv(summary_path)
        logger.info(f"Saved network summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ISMN validation benchmark for SMPS")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to ISMN data directory")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("./results/ismn_validation"), help="Output directory")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional subfolder name inside output-dir (e.g., 'latest') to keep outputs consolidated",
    )
    parser.add_argument("--network", type=str, nargs="+",
                        help="Network names to process")
    parser.add_argument("--all-networks", action="store_true",
                        help="Process all networks")
    parser.add_argument("--start-date", type=str,
                        default="2017-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str,
                        default="2021-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-stations", type=int,
                        help="Maximum stations to process (for testing)")
    parser.add_argument(
        "--prefetch-weather",
        action="store_true",
        help="Only fetch and cache weather for stations (skips validation)",
    )
    parser.add_argument(
        "--prefetch-sleep-seconds",
        type=float,
        default=0.0,
        help="Optional sleep between uncached weather fetches (helps avoid 429)",
    )
    parser.add_argument(
        "--export-pairs",
        action="store_true",
        help="Export paired obs/sim records (per depth and horizon) for scatter-fit plots",
    )
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run validation
    runner = ISMNValidationRunner(
        ismn_data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        start_date=args.start_date,
        end_date=args.end_date,
        export_pairs=args.export_pairs,
    )

    networks = args.network if args.network else (
        None if args.all_networks else [])

    if args.prefetch_weather:
        runner.prefetch_weather(
            networks=networks,
            max_stations=args.max_stations,
            sleep_seconds=args.prefetch_sleep_seconds,
        )
        return

    runner.run_validation(networks=networks, max_stations=args.max_stations)


if __name__ == "__main__":
    main()
