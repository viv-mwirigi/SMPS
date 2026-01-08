#!/usr/bin/env python3
"""
Run SMPS model validation against ISMN in-situ soil moisture observations.

This script:
1. Parses ISMN TAHMO network data files to extract site coordinates and observations
2. Selects diverse sites across different regions (Kenya and Ghana)
3. Fetches weather data from Open-Meteo and soil data from iSDA
4. Runs the physics-based water balance model
5. Compares model predictions against in-situ measurements

Usage:
    python run_ismn_validation.py
    python run_ismn_validation.py --start-date 2021-01-01 --end-date 2021-12-31
    python run_ismn_validation.py --min-observations 50
"""

from smps.physics import create_water_balance_model, EnhancedModelParameters
from smps.physics.soil_hydraulics import VanGenuchtenParameters
from smps.physics.pedotransfer import estimate_soil_parameters_saxton, is_tropical_location
from smps.data.sources.weather import OpenMeteoSource
from smps.data.sources.isda_authenticated import IsdaAfricaAuthenticatedSource
from smps.data.sources.soil import MockSoilSource
from smps.data.sources.satellite import MODISNDVISource
from smps.data.sources.base import DataFetchRequest
from smps.core.types import SoilParameters
from smps.validation.physics_metrics import (
    ExtendedMetrics,
    compute_ubrmse,
    compute_mape,
    compute_kge_decomposition,
    compute_autocorrelation,
)
import argparse
import logging
import warnings
import hashlib
import json
import time
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add src to path BEFORE importing smps modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# SMPS imports (after sys.path modification)

warnings.filterwarnings('ignore')

# Set up logging - reduce verbosity for cleaner output
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress excessive mass balance warnings during long runs
logging.getLogger(
    'smps.physics.enhanced_water_balance').setLevel(logging.ERROR)
logging.getLogger('smps.physics.numerical_solver').setLevel(logging.ERROR)

# Cache and rate limiting configuration
CACHE_DIR = Path(__file__).parent.parent / 'data' / 'cache' / 'weather'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Rate limiting: minimum seconds between Open-Meteo API calls
OPENMETEO_RATE_LIMIT_SECONDS = 1.0
_last_openmeteo_call = 0.0

# Import SMPS components


@dataclass
class ISMNSite:
    """ISMN site metadata"""
    site_id: str
    network: str
    station: str
    latitude: float
    longitude: float
    elevation: float
    country: str
    region: str
    depths: List[float]  # Available measurement depths in meters
    data_files: List[str]


def parse_ismn_stm_header(file_path: str) -> Dict[str, Any]:
    """
    Parse the header line of an ISMN .stm file to extract site metadata.

    Format: Network Network StationName Lat Lon Elevation DepthFrom DepthTo Sensor
    """
    with open(file_path, 'r') as f:
        header = f.readline().strip()

    parts = header.split()
    if len(parts) < 9:
        raise ValueError(f"Invalid header format in {file_path}: {header}")

    return {
        'network': parts[0],
        # Station name often has underscores
        'station': parts[2].replace('_', ' '),
        'latitude': float(parts[3]),
        'longitude': float(parts[4]),
        'elevation': float(parts[5]),
        'depth_from': float(parts[6]),
        'depth_to': float(parts[7]),
        'sensor': parts[8]
    }


def load_ismn_observations(file_path: str) -> pd.DataFrame:
    """
    Load soil moisture observations from an ISMN .stm file.

    Data format (starting from line 2):
    YYYY/MM/DD HH:MM soil_moisture quality_flag network_flag
    """
    # Parse header for metadata
    header_info = parse_ismn_stm_header(file_path)

    # Read data (skip header line)
    data = []
    with open(file_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    date_str = parts[0]
                    time_str = parts[1]
                    sm_value = float(parts[2])
                    quality_flag = parts[3] if len(parts) > 3 else 'U'

                    # Skip missing values (-9999 is ISMN missing data flag)
                    if sm_value < -999 or sm_value > 1.0:
                        continue

                    # Only accept valid volumetric water content (0 to 1)
                    if not (0.0 <= sm_value <= 1.0):
                        continue

                    timestamp = datetime.strptime(
                        f"{date_str} {time_str}", "%Y/%m/%d %H:%M")

                    data.append({
                        'timestamp': timestamp,
                        'soil_moisture': sm_value,
                        'quality_flag': quality_flag
                    })
                except (ValueError, IndexError) as e:
                    continue

    df = pd.DataFrame(data)
    if not df.empty:
        df['depth_m'] = header_info['depth_from']
        df['station'] = header_info['station']
        df['latitude'] = header_info['latitude']
        df['longitude'] = header_info['longitude']

    return df, header_info


def discover_ismn_sites(data_dirs: List[str]) -> Dict[str, ISMNSite]:
    """
    Discover all ISMN sites from data directories.

    Args:
        data_dirs: List of paths to ISMN data directories

    Returns:
        Dictionary mapping site_id to ISMNSite metadata
    """
    sites = {}

    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            continue

        # Look for network subdirectory (TAHMO in our case)
        for network_dir in data_path.iterdir():
            if not network_dir.is_dir() or network_dir.name.startswith('.'):
                continue
            if network_dir.name in ['ISMN_network_flags_descriptions.txt',
                                    'ISMN_qualityflags_description.txt',
                                    'Metadata.xml', 'Readme.txt']:
                continue

            network = network_dir.name

            # Each subdirectory is a station
            for station_dir in network_dir.iterdir():
                if not station_dir.is_dir():
                    continue

                station_name = station_dir.name
                stm_files = list(station_dir.glob('*.stm'))

                if not stm_files:
                    continue

                # Parse first file to get site metadata
                try:
                    _, header_info = load_ismn_observations(str(stm_files[0]))

                    # Collect all available depths
                    depths = set()
                    for stm_file in stm_files:
                        try:
                            _, h = load_ismn_observations(str(stm_file))
                            depths.add(h['depth_from'])
                        except:
                            pass

                    # Determine country/region from coordinates
                    lat = header_info['latitude']
                    lon = header_info['longitude']

                    # Kenya: roughly lat -5 to 5, lon 33 to 42
                    # Ghana: roughly lat 4 to 12, lon -3 to 2
                    if 33 <= lon <= 42 and -5 <= lat <= 5:
                        country = 'Kenya'
                        region = 'East Africa'
                    elif -3 <= lon <= 2 and 4 <= lat <= 12:
                        country = 'Ghana'
                        region = 'West Africa'
                    elif -10 <= lon <= 15 and 4 <= lat <= 15:
                        country = 'West Africa'
                        region = 'West Africa'
                    else:
                        country = 'Unknown'
                        region = 'Africa'

                    # Create unique site ID
                    site_id = f"{network}_{station_name}".replace(
                        ' ', '_').replace(',', '')

                    sites[site_id] = ISMNSite(
                        site_id=site_id,
                        network=network,
                        station=station_name,
                        latitude=lat,
                        longitude=lon,
                        elevation=header_info['elevation'],
                        country=country,
                        region=region,
                        depths=sorted(list(depths)),
                        data_files=[str(f) for f in stm_files]
                    )

                    logger.debug(
                        f"Found site: {site_id} at ({lat:.4f}, {lon:.4f})")

                except Exception as e:
                    logger.warning(f"Error parsing {station_dir}: {e}")
                    continue

    return sites


def select_diverse_sites(sites: Dict[str, ISMNSite], n_sites: int = 20) -> List[ISMNSite]:
    """
    Select diverse sites from available ISMN stations.

    Aims for geographical diversity and good data coverage.
    """
    # Group by region
    by_region = {}
    for site in sites.values():
        if site.region not in by_region:
            by_region[site.region] = []
        by_region[site.region].append(site)

    logger.info(
        f"Sites by region: {[(r, len(s)) for r, s in by_region.items()]}")

    # Select sites proportionally from each region
    selected = []
    total_available = sum(len(s) for s in by_region.values())

    for region, region_sites in by_region.items():
        # Number to select from this region
        proportion = len(region_sites) / total_available
        n_from_region = max(1, int(n_sites * proportion))

        # Sort by number of available depths (prefer sites with more data)
        region_sites_sorted = sorted(region_sites,
                                     key=lambda s: len(s.depths),
                                     reverse=True)

        # Select up to n_from_region sites, spread geographically
        selected.extend(region_sites_sorted[:n_from_region])

    # If we have too few, add more
    while len(selected) < n_sites and len(selected) < len(sites):
        remaining = [s for s in sites.values() if s not in selected]
        if remaining:
            selected.append(remaining[0])

    # If we have too many, trim
    selected = selected[:n_sites]

    return selected


def select_sites_by_depth(sites: Dict[str, ISMNSite],
                          depth_counts: Dict[float, int]) -> List[Tuple[ISMNSite, float]]:
    """
    Select sites with specific depths according to requested counts.

    Args:
        sites: Dictionary of all available sites
        depth_counts: Dict mapping depth (m) to number of sites wanted
                      e.g. {0.05: 4, 0.10: 10, 0.20: 6, 0.30: 4, 0.60: 2}

    Returns:
        List of (site, depth) tuples
    """
    selected = []
    used_sites = set()

    # Process depths from deepest to shallowest (prioritize rare deep measurements)
    for depth in sorted(depth_counts.keys(), reverse=True):
        count = depth_counts[depth]

        # Find sites with this depth
        sites_with_depth = [
            s for s in sites.values()
            if depth in s.depths and s.site_id not in used_sites
        ]

        # Sort by number of depths (prefer sites with multiple depths for better comparison)
        sites_with_depth = sorted(sites_with_depth,
                                  key=lambda s: len(s.depths),
                                  reverse=True)

        # Select up to count sites
        for site in sites_with_depth[:count]:
            selected.append((site, depth))
            used_sites.add(site.site_id)

        logger.info(
            f"  Depth {depth*100:.0f}cm: found {len(sites_with_depth)} sites, selected {min(count, len(sites_with_depth))}")

    return selected


def load_site_observations(site: ISMNSite,
                           start_date: date,
                           end_date: date,
                           target_depth: float = 0.10) -> pd.DataFrame:
    """
    Load and aggregate observations for a site.

    Args:
        site: ISMNSite object
        start_date: Start of analysis period
        end_date: End of analysis period
        target_depth: Preferred measurement depth (default 10cm)

    Returns:
        DataFrame with daily soil moisture observations
    """
    # Find best matching depth
    available_depths = site.depths
    if target_depth in available_depths:
        selected_depth = target_depth
    else:
        # Find closest depth
        selected_depth = min(
            available_depths, key=lambda d: abs(d - target_depth))

    logger.debug(f"Using depth {selected_depth}m for site {site.site_id}")

    # Find file with matching depth
    all_obs = []
    for file_path in site.data_files:
        if f"_{selected_depth:.6f}_{selected_depth:.6f}_" in file_path or \
           f"_{selected_depth:.2f}" in file_path:
            try:
                df, _ = load_ismn_observations(file_path)
                all_obs.append(df)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")

    # If no exact match, load first available
    if not all_obs and site.data_files:
        try:
            df, _ = load_ismn_observations(site.data_files[0])
            all_obs.append(df)
        except Exception as e:
            logger.warning(f"Error loading first file for {site.site_id}: {e}")

    if not all_obs:
        return pd.DataFrame()

    # Combine observations
    df = pd.concat(all_obs, ignore_index=True)

    # Filter by date range
    df = df[(df['timestamp'] >= datetime.combine(start_date, datetime.min.time())) &
            (df['timestamp'] <= datetime.combine(end_date, datetime.max.time()))]

    # Filter good quality data (G = good)
    # G=Good, M=Maybe, 1=good flag
    df = df[df['quality_flag'].isin(['G', 'M', '1'])]

    if df.empty:
        return df

    # Aggregate to daily (mean of hourly observations)
    df['date'] = df['timestamp'].dt.date
    daily = df.groupby('date').agg({
        'soil_moisture': 'mean',
        'depth_m': 'first',
        'station': 'first',
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()

    daily['date'] = pd.to_datetime(daily['date'])

    return daily


def fetch_soil_data(site_id: str, lat: float, lon: float) -> Dict[str, Any]:
    """Fetch soil data from iSDA or fallback to defaults."""
    try:
        isda = IsdaAfricaAuthenticatedSource()
        profile = isda.fetch_soil_profile(site_id, latitude=lat, longitude=lon)
        logger.info(
            f"  ✓ Soil data from iSDA: Sand={profile.sand_percent:.0f}%, Clay={profile.clay_percent:.0f}%")
        return {
            'sand': profile.sand_percent,
            'clay': profile.clay_percent,
            'silt': profile.silt_percent,
            'porosity': profile.porosity,
            'field_capacity': profile.field_capacity,
            'wilting_point': profile.wilting_point,
            'organic_carbon': getattr(profile, 'organic_carbon_percent', 1.5),
            'bulk_density': getattr(profile, 'bulk_density_g_cm3', 1.3),
            'source': 'isda'
        }
    except Exception as e:
        logger.warning(f"  iSDA failed: {e}")
        # Fallback defaults based on region (typical tropical soils)
        return {
            'sand': 40,
            'clay': 30,
            'silt': 30,
            'porosity': 0.45,
            'field_capacity': 0.30,
            'wilting_point': 0.12,
            'organic_carbon': 1.5,
            'bulk_density': 1.3,
            'source': 'default'
        }


def _get_weather_cache_path(lat: float, lon: float, start_date: date, end_date: date) -> Path:
    """Generate a unique cache file path for weather data."""
    # Create a hash from the request parameters
    key = f"{lat:.4f}_{lon:.4f}_{start_date}_{end_date}"
    hash_key = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"weather_{hash_key}.json"


def _load_weather_from_cache(cache_path: Path) -> Optional[pd.DataFrame]:
    """Load weather data from cache if available."""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        return df
    except Exception as e:
        logger.debug(f"Cache read failed: {e}")
        return None


def _save_weather_to_cache(cache_path: Path, df: pd.DataFrame) -> None:
    """Save weather data to cache."""
    try:
        # Reset index to include date in the data
        df_to_save = df.reset_index()
        df_to_save['date'] = df_to_save['date'].dt.strftime('%Y-%m-%d')
        with open(cache_path, 'w') as f:
            json.dump(df_to_save.to_dict(orient='records'), f)
    except Exception as e:
        logger.debug(f"Cache write failed: {e}")


def _rate_limit_openmeteo() -> None:
    """Apply rate limiting for Open-Meteo API calls."""
    global _last_openmeteo_call
    now = time.time()
    elapsed = now - _last_openmeteo_call
    if elapsed < OPENMETEO_RATE_LIMIT_SECONDS:
        sleep_time = OPENMETEO_RATE_LIMIT_SECONDS - elapsed
        time.sleep(sleep_time)
    _last_openmeteo_call = time.time()


def fetch_weather_data(site_id: str, lat: float, lon: float,
                       start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch weather data from Open-Meteo with caching and rate limiting."""
    # Check cache first
    cache_path = _get_weather_cache_path(lat, lon, start_date, end_date)
    cached_df = _load_weather_from_cache(cache_path)
    if cached_df is not None:
        logger.info(f"  ✓ Weather data: {len(cached_df)} days from cache")
        return cached_df

    # Apply rate limiting before API call
    _rate_limit_openmeteo()

    weather_source = OpenMeteoSource()

    # Monkey-patch coordinate lookup
    weather_source._get_site_coordinates = lambda s: (lat, lon)

    request = DataFetchRequest(
        site_id=site_id,
        start_date=start_date,
        end_date=end_date,
        parameters={"include_forecast": False}
    )

    try:
        data = weather_source.fetch_daily_weather(request)
        df = pd.DataFrame([d.model_dump() for d in data])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # Save to cache
        _save_weather_to_cache(cache_path, df)

        logger.info(f"  ✓ Weather data: {len(df)} days from Open-Meteo")
        return df
    except Exception as e:
        logger.error(f"  Weather fetch failed: {e}")
        return pd.DataFrame()


def fetch_ndvi_data(site_id: str, lat: float, lon: float,
                    start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch or generate NDVI data."""
    ndvi_source = MODISNDVISource()
    ndvi_source._get_site_coordinates = lambda s: (lat, lon)

    request = DataFetchRequest(
        site_id=site_id,
        start_date=start_date,
        end_date=end_date,
        parameters={}
    )

    try:
        result = ndvi_source.fetch(request)
        if result.data:
            df = pd.DataFrame([d.model_dump() for d in result.data])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            logger.info(f"  ✓ NDVI data: {len(df)} observations from MODIS")
            return df
    except Exception as e:
        logger.debug(f"  NDVI fetch failed: {e}")

    # Fallback: generate seasonal NDVI pattern
    dates = pd.date_range(start_date, end_date, freq='D')
    doy = dates.dayofyear

    if abs(lat) < 15:  # Tropics - less seasonal variation
        ndvi_vals = 0.55 + 0.15 * np.sin(2 * np.pi * (doy - 60) / 365)
    elif lat > 0:  # Northern hemisphere
        ndvi_vals = 0.35 + 0.35 * np.sin(2 * np.pi * (doy - 100) / 365)
    else:  # Southern hemisphere
        ndvi_vals = 0.35 + 0.35 * np.sin(2 * np.pi * (doy + 80) / 365)

    logger.info(f"  ℹ NDVI: Using synthetic seasonal pattern")
    return pd.DataFrame({'ndvi': ndvi_vals}, index=dates)


def run_model_for_site(site: ISMNSite,
                       observations: pd.DataFrame,
                       start_date: date,
                       end_date: date) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Run the water balance model for a site.

    Returns:
        Tuple of (model predictions DataFrame, metadata dict)
    """
    site_id = site.site_id
    lat = site.latitude
    lon = site.longitude
    sensor_depth_m = observations['depth_m'].iloc[0] if len(
        observations) > 0 else 0.10

    metadata = {
        'site_id': site_id,
        'latitude': lat,
        'longitude': lon,
        'country': site.country,
        'region': site.region,
        'sensor_depth_m': sensor_depth_m
    }

    # Fetch soil data
    soil_data = fetch_soil_data(site_id, lat, lon)
    metadata['soil_source'] = soil_data.get('source', 'unknown')
    metadata['sand_pct'] = soil_data['sand']
    metadata['clay_pct'] = soil_data['clay']

    # Determine if this is a tropical location (for PTF corrections)
    is_tropical = is_tropical_location(lat, lon)
    metadata['is_tropical'] = is_tropical

    # Calculate Saxton-Rawls hydraulic parameters with tropical correction
    sr_params = estimate_soil_parameters_saxton(
        sand_percent=soil_data['sand'],
        clay_percent=soil_data['clay'],
        organic_matter_percent=soil_data['organic_carbon'],
        is_tropical_oxide_soil=is_tropical
    )

    metadata['theta_fc'] = sr_params.field_capacity
    metadata['theta_sat'] = sr_params.porosity
    metadata['theta_wp'] = sr_params.wilting_point
    metadata['k_sat'] = sr_params.saturated_hydraulic_conductivity_cm_day

    # Build soil parameters
    soil_params = SoilParameters(
        sand_percent=soil_data['sand'],
        silt_percent=soil_data['silt'],
        clay_percent=soil_data['clay'],
        porosity=sr_params.porosity,
        field_capacity=sr_params.field_capacity,
        wilting_point=sr_params.wilting_point,
        saturated_hydraulic_conductivity_cm_day=sr_params.saturated_hydraulic_conductivity_cm_day,
        bulk_density_g_cm3=soil_data['bulk_density'],
        organic_matter_percent=soil_data['organic_carbon'],
    )

    # Determine soil texture class for the production model
    sand_pct = soil_data['sand']
    clay_pct = soil_data['clay']
    if sand_pct > 70:
        soil_texture = "sand"
    elif clay_pct > 40:
        soil_texture = "clay"
    elif sand_pct > 50 and clay_pct < 20:
        soil_texture = "sandy_loam"
    elif clay_pct > 25:
        soil_texture = "clay_loam"
    else:
        soil_texture = "loam"

    # Create PRODUCTION model with full physics (EnhancedWaterBalance)
    try:
        model = create_water_balance_model(
            crop_type="maize",
            soil_texture=soil_texture,
            use_full_physics=True  # Full physics for best accuracy
        )
        metadata['model_type'] = 'EnhancedWaterBalance'
    except Exception as e:
        logger.error(f"  Failed to create model: {e}")
        return None, metadata

    # Fetch weather data
    weather_df = fetch_weather_data(site_id, lat, lon, start_date, end_date)
    if weather_df.empty:
        logger.error(f"  No weather data for {site_id}")
        return None, metadata

    logger.info(f"  ✓ Weather: {len(weather_df)} days")

    # Fetch NDVI data
    ndvi_df = fetch_ndvi_data(site_id, lat, lon, start_date, end_date)

    # Run daily simulation
    results = []
    errors_count = 0
    first_error = None
    for day_date in pd.date_range(start_date, end_date, freq='D'):
        day_date_py = day_date.date()

        # Get weather for this day
        if day_date not in weather_df.index:
            continue

        weather = weather_df.loc[day_date]
        precip = float(weather.get('precipitation_mm', 0))
        et0 = float(weather.get('et0_mm', 3.0))

        # Get NDVI for this day
        if day_date in ndvi_df.index:
            ndvi = float(ndvi_df.loc[day_date].get('ndvi', 0.5))
        else:
            ndvi = 0.5

        # Run model step with ENHANCED physics model
        try:
            # EnhancedWaterBalance returns (PhysicsPriorResult, EnhancedFluxes)
            result, fluxes = model.run_daily(
                precipitation_mm=precip,
                et0_mm=et0,
                ndvi=ndvi,
                day_of_season=60  # Mid-season default
            )

            # Get soil moisture values from model layers
            # EnhancedWaterBalance has 3 layers: Surface (0-10cm), Root zone (10-40cm), Deep (40-100cm)
            theta_surface = result.theta_surface
            theta_root = result.theta_root
            theta_deep = model.layers[2].theta  # Deep layer from model state

            # Integrated 0-100cm (weighted by layer thickness)
            # Layer 0: 10cm, Layer 1: 30cm, Layer 2: 60cm = 100cm total
            theta_integrated = (0.10 * theta_surface +
                                0.30 * theta_root + 0.60 * theta_deep)

            # Get soil moisture at sensor depth
            if sensor_depth_m <= 0.05:
                # 5cm - pure surface layer
                sm_model = theta_surface
            elif sensor_depth_m <= 0.10:
                # 10cm - surface layer
                sm_model = theta_surface
            elif sensor_depth_m <= 0.20:
                # 20cm - weighted blend of surface and root zone
                sm_model = 0.5 * theta_surface + 0.5 * theta_root
            elif sensor_depth_m <= 0.30:
                # 30cm - mostly root zone
                sm_model = (10/30) * theta_surface + (20/30) * theta_root
            elif sensor_depth_m <= 0.40:
                # 40cm - full root zone integration
                sm_model = (10/40) * theta_surface + (30/40) * theta_root
            elif sensor_depth_m <= 0.60:
                # 60cm - surface + root + part of deep
                sm_model = (10/60) * theta_surface + \
                           (30/60) * theta_root + \
                           (20/60) * theta_deep
            else:
                # Deeper - use integrated 0-100cm
                sm_model = theta_integrated

            results.append({
                'date': day_date,
                'sm_model': sm_model,
                'sm_surface': theta_surface,
                'sm_root': theta_root,
                'sm_deep': theta_deep,
                'sm_integrated': theta_integrated,
                'precip_mm': precip,
                'et0_mm': et0,
                'ndvi': ndvi,
                'infiltration': fluxes.infiltration,
                'runoff': fluxes.runoff,
                'actual_et': fluxes.actual_et,
                'transpiration': fluxes.transpiration,
                'soil_evap': fluxes.soil_evaporation,
                'deep_drainage': fluxes.deep_drainage,
                'water_balance_error': result.water_balance_error,
            })
        except Exception as e:
            errors_count += 1
            if first_error is None:
                first_error = str(e)
            continue

    if errors_count > 0:
        logger.warning(
            f"  Model had {errors_count} errors. First error: {first_error}")

    if not results:
        logger.error(f"  No successful model runs!")
        return None, metadata

    return pd.DataFrame(results), metadata


def calculate_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calculate validation metrics including extended physics-based metrics."""
    # Filter out NaN values
    mask = ~(np.isnan(observed) | np.isnan(predicted))
    obs = observed[mask]
    pred = predicted[mask]

    if len(obs) < 10:
        return {'n': len(obs), 'mae': np.nan, 'rmse': np.nan, 'bias': np.nan, 'r': np.nan}

    mae = np.mean(np.abs(obs - pred))
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    bias = np.mean(pred - obs)

    if np.std(obs) > 0 and np.std(pred) > 0:
        r = np.corrcoef(obs, pred)[0, 1]
    else:
        r = np.nan

    # Extended metrics (Gap 10)
    ubrmse = compute_ubrmse(obs, pred)
    mape = compute_mape(obs, pred)
    kge, kge_r, kge_alpha, kge_beta = compute_kge_decomposition(obs, pred)

    # Temporal structure
    ac_obs = compute_autocorrelation(obs, lag=1)
    ac_pred = compute_autocorrelation(pred, lag=1)
    ac_error = abs(ac_obs - ac_pred) if not (np.isnan(ac_obs)
                                             or np.isnan(ac_pred)) else np.nan

    # Nash-Sutcliffe Efficiency
    nse = 1 - np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2)

    return {
        'n': len(obs),
        'mae': mae,
        'rmse': rmse,
        'ubrmse': ubrmse,
        'bias': bias,
        'r': r,
        'r_squared': r**2 if not np.isnan(r) else np.nan,
        'nse': nse,
        'kge': kge,
        'kge_r': kge_r,
        'kge_alpha': kge_alpha,
        'kge_beta': kge_beta,
        'mape': mape,
        'ac_obs': ac_obs,
        'ac_pred': ac_pred,
        'ac_error': ac_error,
        'obs_mean': np.mean(obs),
        'pred_mean': np.mean(pred),
        'obs_std': np.std(obs),
        'pred_std': np.std(pred)
    }


def calculate_multistep_metrics(
    obs_series: np.ndarray,
    pred_series: np.ndarray,
    horizons: List[int] = [1, 3, 7]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for multi-step predictions at different forecast horizons.

    For each horizon h, we evaluate how well day-t predictions match day-(t+h) observations.
    This tests the model's ability to forecast soil moisture evolution.

    Args:
        obs_series: Observed soil moisture time series
        pred_series: Predicted soil moisture time series
        horizons: List of forecast horizons in days (e.g., [1, 3, 7])

    Returns:
        Dictionary mapping horizon to metrics dict
    """
    results = {}

    for h in horizons:
        if len(obs_series) <= h:
            results[f'{h}-day'] = {'n': 0, 'rmse': np.nan, 'r': np.nan}
            continue

        # For multi-step: compare prediction at day t with obs at day t+h
        # This evaluates model's prognostic skill
        obs_shifted = obs_series[h:]
        pred_base = pred_series[:-h]

        mask = ~(np.isnan(obs_shifted) | np.isnan(pred_base))
        obs_h = obs_shifted[mask]
        pred_h = pred_base[mask]

        if len(obs_h) < 10:
            results[f'{h}-day'] = {'n': len(obs_h),
                                   'rmse': np.nan, 'r': np.nan}
            continue

        rmse = np.sqrt(np.mean((obs_h - pred_h) ** 2))
        r = np.corrcoef(obs_h, pred_h)[0, 1] if np.std(
            obs_h) > 0 and np.std(pred_h) > 0 else np.nan
        mae = np.mean(np.abs(obs_h - pred_h))
        bias = np.mean(pred_h - obs_h)

        results[f'{h}-day'] = {
            'n': len(obs_h),
            'rmse': rmse,
            'mae': mae,
            'bias': bias,
            'r': r,
        }

    return results


# Default configuration values
DEFAULT_START_DATE = date(2020, 1, 1)
DEFAULT_END_DATE = date(2022, 12, 31)
DEFAULT_MIN_OBSERVATIONS = 100
DEFAULT_RATE_LIMIT_SECONDS = 0.5
DEFAULT_FORECAST_HORIZONS = [1, 3, 7]
DEFAULT_DEPTH_COUNTS = {
    0.05: 4,   # 5cm - surface
    0.10: 10,  # 10cm - surface
    0.20: 6,   # 20cm - surface/root blend
    0.30: 4,   # 30cm - root zone
    0.60: 2,   # 60cm - root/deep blend
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run SMPS model validation against ISMN in-situ observations.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--start-date', type=str, default=DEFAULT_START_DATE.isoformat(),
        help='Start date for analysis (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date', type=str, default=DEFAULT_END_DATE.isoformat(),
        help='End date for analysis (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--min-observations', type=int, default=DEFAULT_MIN_OBSERVATIONS,
        help='Minimum number of daily observations required per site'
    )
    parser.add_argument(
        '--rate-limit', type=float, default=DEFAULT_RATE_LIMIT_SECONDS,
        help='Rate limit between API calls (seconds)'
    )
    parser.add_argument(
        '--data-dirs', nargs='+', default=None,
        help='ISMN data directories (defaults to standard paths)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='data/ismn',
        help='Output directory for results'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Main validation workflow."""
    args = parse_args()

    # Configure logging based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger('smps.physics.enhanced_water_balance').setLevel(
            logging.WARNING)
        logging.getLogger('smps.physics.numerical_solver').setLevel(
            logging.WARNING)

    # Parse dates
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    min_observations = args.min_observations
    rate_limit = args.rate_limit
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("ISMN VALIDATION: EnhancedWaterBalance Model vs In-Situ Observations")
    print("Full Physics: Green-Ampt, FAO-56, Feddes, Darcy, Capillary Rise")
    print("=" * 80)
    print()

    # Depth-based site selection
    depth_counts = DEFAULT_DEPTH_COUNTS.copy()
    total_sites = sum(depth_counts.values())

    # ISMN data directories
    if args.data_dirs:
        data_dirs = args.data_dirs
    else:
        data_dirs = [
            '/home/viv/SMPS/data/ismn/Data_separate_files_header_20170105_20250105_12892_F2PyW_20260105',
            '/home/viv/SMPS/data/ismn/Data_separate_files_header_20170105_20250105_12892_gdf6E_20260105'
        ]

    print(f"Analysis period: {start_date} to {end_date}")
    print(f"Target sites by depth:")
    for depth, count in sorted(depth_counts.items()):
        print(f"   {depth*100:.0f}cm: {count} sites")
    print(f"   Total: {total_sites} sites")
    print()

    # 1. Discover all sites
    print("1. DISCOVERING ISMN SITES...")
    print("-" * 40)
    sites = discover_ismn_sites(data_dirs)
    print(f"   Found {len(sites)} sites in total")

    # Show depth distribution
    depth_dist = {}
    for site in sites.values():
        for d in site.depths:
            depth_dist[d] = depth_dist.get(d, 0) + 1
    print(f"   Depth distribution:")
    for d in sorted(depth_dist.keys()):
        print(f"      {d*100:.0f}cm: {depth_dist[d]} sites")
    print()

    # 2. Select sites by depth
    print(f"2. SELECTING SITES BY DEPTH...")
    print("-" * 40)
    selected_sites_with_depths = select_sites_by_depth(sites, depth_counts)

    print(
        f"\n   Selected {len(selected_sites_with_depths)} site-depth combinations:")
    for site, depth in selected_sites_with_depths:
        print(f"   - {site.station} @ {depth*100:.0f}cm ({site.country})")
    print()

    # 3. Run validation
    print("3. RUNNING MODEL VALIDATION (EnhancedWaterBalance - Full Physics)...")
    print("-" * 40)

    all_results = []
    site_summaries = []
    multistep_results = []

    for i, (site, target_depth) in enumerate(selected_sites_with_depths, 1):
        print(
            f"\n[{i}/{len(selected_sites_with_depths)}] {site.station} @ {target_depth*100:.0f}cm ({site.country})")
        print(f"   Location: ({site.latitude:.4f}, {site.longitude:.4f})")

        # Load observations at the specific depth
        obs_df = load_site_observations(
            site, start_date, end_date, target_depth=target_depth)

        if obs_df.empty or len(obs_df) < min_observations:
            print(
                f"   ✗ Insufficient observations ({len(obs_df)} days, need {min_observations})")
            continue

        actual_depth = obs_df['depth_m'].iloc[0]
        print(f"   Observations: {len(obs_df)} days at depth {actual_depth}m")

        # Run model
        model_df, metadata = run_model_for_site(
            site, obs_df, start_date, end_date)

        if model_df is None or model_df.empty:
            print(f"   ✗ Model failed")
            continue

        # Merge and calculate metrics
        obs_df['date'] = pd.to_datetime(obs_df['date'])
        model_cols = ['date', 'sm_model', 'sm_surface', 'sm_root', 'sm_deep',
                      'sm_integrated', 'precip_mm', 'infiltration', 'runoff',
                      'actual_et', 'transpiration', 'soil_evap', 'deep_drainage']
        # Use only columns that exist
        available_cols = [c for c in model_cols if c in model_df.columns]
        merged = pd.merge(
            obs_df[['date', 'soil_moisture']],
            model_df[available_cols],
            on='date',
            how='inner'
        )

        if len(merged) < min_observations:
            print(
                f"   ✗ Insufficient overlap ({len(merged)} days, need {min_observations})")
            continue

        # Calculate 1-day metrics (standard)
        metrics = calculate_metrics(
            merged['soil_moisture'].values,
            merged['sm_model'].values
        )

        # Calculate multi-step metrics
        ms_metrics = calculate_multistep_metrics(
            merged['soil_moisture'].values,
            merged['sm_model'].values,
            horizons=DEFAULT_FORECAST_HORIZONS
        )

        print(
            f"   ✓ Results: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, KGE={metrics['kge']:.3f}, r={metrics['r']:.3f}")

        # Show multi-step forecast skill
        for horizon, h_metrics in ms_metrics.items():
            if not np.isnan(h_metrics.get('rmse', np.nan)):
                print(
                    f"      {horizon} forecast: RMSE={h_metrics['rmse']:.4f}, r={h_metrics['r']:.3f}")

        # Store results
        merged['site_id'] = site.site_id
        merged['depth_m'] = actual_depth
        all_results.append(merged)

        # Store multi-step results
        for horizon, h_metrics in ms_metrics.items():
            multistep_results.append({
                'site_id': site.site_id,
                'station': site.station,
                'depth_m': actual_depth,
                'horizon': horizon,
                **h_metrics
            })

        site_summaries.append({
            'site_id': site.site_id,
            'station': site.station,
            'latitude': site.latitude,
            'longitude': site.longitude,
            'country': site.country,
            'region': site.region,
            'depth_m': actual_depth,
            **metadata,
            **metrics
        })

        # Rate limit
        time.sleep(rate_limit)

    # 4. Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY - EnhancedWaterBalance (Full Physics)")
    print("=" * 80)

    if site_summaries:
        summary_df = pd.DataFrame(site_summaries)

        print(f"\n{'='*60}")
        print("OVERALL PERFORMANCE METRICS")
        print(f"{'='*60}")
        print(f"Sites validated: {len(summary_df)}")
        print(f"\n  Standard Metrics:")
        print(f"    MAE:     {summary_df['mae'].mean():.4f} m³/m³")
        print(f"    RMSE:    {summary_df['rmse'].mean():.4f} m³/m³")
        print(f"    ubRMSE:  {summary_df['ubrmse'].mean():.4f} m³/m³")
        print(f"    Bias:    {summary_df['bias'].mean():.4f} m³/m³")
        print(f"    r:       {summary_df['r'].mean():.3f}")
        print(f"    R²:      {summary_df['r_squared'].mean():.3f}")

        print(f"\n  Advanced Metrics:")
        print(f"    NSE:     {summary_df['nse'].mean():.3f}")
        print(f"    KGE:     {summary_df['kge'].mean():.3f}")
        print(f"    KGE_r:   {summary_df['kge_r'].mean():.3f} (correlation)")
        print(
            f"    KGE_α:   {summary_df['kge_alpha'].mean():.3f} (variability)")
        print(f"    KGE_β:   {summary_df['kge_beta'].mean():.3f} (bias ratio)")

        print(f"\n  Temporal Structure:")
        print(f"    AC(1) obs:  {summary_df['ac_obs'].mean():.3f}")
        print(f"    AC(1) pred: {summary_df['ac_pred'].mean():.3f}")
        print(f"    AC error:   {summary_df['ac_error'].mean():.3f}")

        # Multi-step prediction summary
        if multistep_results:
            print(f"\n{'='*60}")
            print("MULTI-STEP PREDICTION PERFORMANCE")
            print(f"{'='*60}")
            ms_df = pd.DataFrame(multistep_results)
            for horizon in ['1-day', '3-day', '7-day']:
                h_df = ms_df[ms_df['horizon'] == horizon]
                if len(h_df) > 0:
                    print(
                        f"  {horizon}: RMSE={h_df['rmse'].mean():.4f}, MAE={h_df['mae'].mean():.4f}, r={h_df['r'].mean():.3f}")

        print(f"\n{'='*60}")
        print("PERFORMANCE BY DEPTH")
        print(f"{'='*60}")
        for depth in sorted(summary_df['depth_m'].unique()):
            depth_df = summary_df[summary_df['depth_m'] == depth]
            print(f"  {depth*100:.0f}cm: n={len(depth_df)}, MAE={depth_df['mae'].mean():.4f}, "
                  f"RMSE={depth_df['rmse'].mean():.4f}, KGE={depth_df['kge'].mean():.3f}, r={depth_df['r'].mean():.3f}")

        print(f"\n{'='*60}")
        print("PERFORMANCE BY REGION")
        print(f"{'='*60}")
        for region in summary_df['region'].unique():
            region_df = summary_df[summary_df['region'] == region]
            print(f"  {region}: n={len(region_df)}, MAE={region_df['mae'].mean():.4f}, "
                  f"RMSE={region_df['rmse'].mean():.4f}, KGE={region_df['kge'].mean():.3f}, r={region_df['r'].mean():.3f}")

        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_df.to_csv(
            output_dir / 'ismn_validation_summary_enhanced.csv', index=False)
        print(
            f"\n✓ Summary saved to: {output_dir / 'ismn_validation_summary_enhanced.csv'}")

        if multistep_results:
            ms_df.to_csv(
                output_dir / 'ismn_multistep_validation.csv', index=False)
            print(
                f"✓ Multi-step results saved to: {output_dir / 'ismn_multistep_validation.csv'}")

        if all_results:
            results_df = pd.concat(all_results, ignore_index=True)
            results_df.to_csv(
                output_dir / 'ismn_daily_results_enhanced.csv', index=False)
            print(
                f"✓ Daily results saved to: {output_dir / 'ismn_daily_results_enhanced.csv'}")
    else:
        print("\nNo sites were successfully validated!")

    return site_summaries


if __name__ == '__main__':
    main()
