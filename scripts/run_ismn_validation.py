#!/usr/bin/env python3
"""
Run SMPS model validation against ISMN in-situ soil moisture observations.

This script:
1. Parses ISMN TAHMO network data files to extract site coordinates and observations
2. Selects 20 diverse sites across different regions (Kenya and Ghana)
3. Fetches weather data from Open-Meteo and soil data from iSDA
4. Runs the physics-based water balance model
5. Compares model predictions against in-situ measurements

Data period: January 2020 - December 2021
"""

from smps.physics.water_balance import (
    ThreeLayerWaterBalance,
    ThreeLayerParameters,
)
from smps.physics.pedotransfer import estimate_soil_parameters_saxton, is_tropical_location
from smps.data.sources.weather import OpenMeteoSource
from smps.data.sources.isda_authenticated import IsdaAfricaAuthenticatedSource
from smps.data.sources.soil import MockSoilSource
from smps.data.sources.satellite import MODISNDVISource
from smps.data.sources.base import DataFetchRequest
from smps.core.types import SoilParameters
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta, date
import os
import time
import json
import hashlib
import warnings
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

    # Create three-layer model
    params = ThreeLayerParameters.from_soil_parameters(soil_params)
    model = ThreeLayerWaterBalance(params)

    # Fetch weather data
    weather_df = fetch_weather_data(site_id, lat, lon, start_date, end_date)
    if weather_df.empty:
        logger.error(f"  No weather data for {site_id}")
        return None, metadata

    # Fetch NDVI data
    ndvi_df = fetch_ndvi_data(site_id, lat, lon, start_date, end_date)

    # Run daily simulation
    results = []
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

        # Run model step
        try:
            result = model.run_daily(
                precipitation_mm=precip,
                et0_mm=et0,
                ndvi=ndvi
            )

            # Get soil moisture at sensor depth
            # Model layers: Surface (0-10cm), Root zone (10-40cm), Deep (40-100cm)
            if sensor_depth_m <= 0.05:
                # 5cm - pure surface layer
                sm_model = result['theta_surface']
            elif sensor_depth_m <= 0.10:
                # 10cm - surface layer
                sm_model = result['theta_surface']
            elif sensor_depth_m <= 0.20:
                # 20cm - weighted blend of surface and root zone
                # 10cm from surface (0-10), 10cm from root (10-20)
                sm_model = 0.5 * result['theta_surface'] + \
                    0.5 * result['theta_root']
            elif sensor_depth_m <= 0.30:
                # 30cm - mostly root zone
                # Weighted: surface contributes 10cm/30cm, root zone 20cm/30cm
                sm_model = (
                    10/30) * result['theta_surface'] + (20/30) * result['theta_root']
            elif sensor_depth_m <= 0.40:
                # 40cm - full root zone integration
                sm_model = (
                    10/40) * result['theta_surface'] + (30/40) * result['theta_root']
            elif sensor_depth_m <= 0.60:
                # 60cm - surface + root + part of deep
                # Surface: 10cm, Root: 30cm, Deep: 20cm out of 60cm
                sm_model = (10/60) * result['theta_surface'] + \
                           (30/60) * result['theta_root'] + \
                           (20/60) * result['theta_deep']
            else:
                # Deeper - use integrated 0-100cm
                sm_model = result['theta_0_100cm']

            results.append({
                'date': day_date,
                'sm_model': sm_model,
                'sm_surface': result['theta_surface'],
                'sm_root': result['theta_root'],
                'sm_deep': result['theta_deep'],
                'sm_integrated': result['theta_0_100cm'],
                'precip_mm': precip,
                'et0_mm': et0,
                'ndvi': ndvi,
            })
        except Exception as e:
            logger.debug(f"Model step error on {day_date}: {e}")
            continue
        except Exception as e:
            logger.debug(f"Model step error on {day_date}: {e}")
            continue

    if not results:
        return None, metadata

    return pd.DataFrame(results), metadata


def calculate_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calculate validation metrics."""
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

    return {
        'n': len(obs),
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'r': r,
        'obs_mean': np.mean(obs),
        'pred_mean': np.mean(pred),
        'obs_std': np.std(obs),
        'pred_std': np.std(pred)
    }


def main():
    """Main validation workflow."""
    print("=" * 80)
    print("ISMN VALIDATION: SMPS Water Balance Model vs In-Situ Observations")
    print("=" * 80)
    print()

    # Configuration
    START_DATE = date(2020, 1, 1)
    END_DATE = date(2021, 12, 31)

    # Depth-based site selection: 4 sites @5cm, 10 @10cm, 6 @20cm, 4 @30cm, 2 @60cm
    DEPTH_COUNTS = {
        0.05: 4,   # 5cm - surface
        0.10: 10,  # 10cm - surface
        0.20: 6,   # 20cm - surface/root blend
        0.30: 4,   # 30cm - root zone
        0.60: 2,   # 60cm - root/deep blend
    }
    TOTAL_SITES = sum(DEPTH_COUNTS.values())  # 26 sites

    # ISMN data directories
    DATA_DIRS = [
        '/home/viv/SMPS/data/ismn/Data_separate_files_header_20200101_20211231_12892_kDkY_20251215',
        '/home/viv/SMPS/data/ismn/Data_separate_files_header_20200101_20211231_12892_n1Fe_20251215'
    ]

    print(f"Analysis period: {START_DATE} to {END_DATE}")
    print(f"Target sites by depth:")
    for depth, count in sorted(DEPTH_COUNTS.items()):
        print(f"   {depth*100:.0f}cm: {count} sites")
    print(f"   Total: {TOTAL_SITES} sites")
    print()

    # 1. Discover all sites
    print("1. DISCOVERING ISMN SITES...")
    print("-" * 40)
    sites = discover_ismn_sites(DATA_DIRS)
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
    selected_sites_with_depths = select_sites_by_depth(sites, DEPTH_COUNTS)

    print(
        f"\n   Selected {len(selected_sites_with_depths)} site-depth combinations:")
    for site, depth in selected_sites_with_depths:
        print(f"   - {site.station} @ {depth*100:.0f}cm ({site.country})")
    print()

    # 3. Run validation
    print("3. RUNNING MODEL VALIDATION...")
    print("-" * 40)

    all_results = []
    site_summaries = []

    for i, (site, target_depth) in enumerate(selected_sites_with_depths, 1):
        print(
            f"\n[{i}/{len(selected_sites_with_depths)}] {site.station} @ {target_depth*100:.0f}cm ({site.country})")
        print(f"   Location: ({site.latitude:.4f}, {site.longitude:.4f})")

        # Load observations at the specific depth
        obs_df = load_site_observations(
            site, START_DATE, END_DATE, target_depth=target_depth)

        if obs_df.empty or len(obs_df) < 100:
            print(f"   ✗ Insufficient observations ({len(obs_df)} days)")
            continue

        actual_depth = obs_df['depth_m'].iloc[0]
        print(f"   Observations: {len(obs_df)} days at depth {actual_depth}m")

        # Run model
        model_df, metadata = run_model_for_site(
            site, obs_df, START_DATE, END_DATE)

        if model_df is None or model_df.empty:
            print(f"   ✗ Model failed")
            continue

        # Merge and calculate metrics
        obs_df['date'] = pd.to_datetime(obs_df['date'])
        merged = pd.merge(
            obs_df[['date', 'soil_moisture']],
            model_df[['date', 'sm_model', 'sm_surface', 'sm_root',
                      'sm_deep', 'sm_integrated', 'precip_mm']],
            on='date',
            how='inner'
        )

        if len(merged) < 100:
            print(f"   ✗ Insufficient overlap ({len(merged)} days)")
            continue

        metrics = calculate_metrics(
            merged['soil_moisture'].values,
            merged['sm_model'].values
        )

        print(
            f"   ✓ Results: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, r={metrics['r']:.3f}")

        # Store results
        merged['site_id'] = site.site_id
        merged['depth_m'] = actual_depth
        all_results.append(merged)

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
        time.sleep(0.5)

    # 4. Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    if site_summaries:
        summary_df = pd.DataFrame(site_summaries)

        print(f"\nSites validated: {len(summary_df)}")
        print(f"Mean MAE:  {summary_df['mae'].mean():.4f} m³/m³")
        print(f"Mean RMSE: {summary_df['rmse'].mean():.4f} m³/m³")
        print(f"Mean Bias: {summary_df['bias'].mean():.4f} m³/m³")
        print(f"Mean r:    {summary_df['r'].mean():.3f}")

        print("\nBy Depth:")
        for depth in sorted(summary_df['depth_m'].unique()):
            depth_df = summary_df[summary_df['depth_m'] == depth]
            print(
                f"  {depth*100:.0f}cm: n={len(depth_df)}, MAE={depth_df['mae'].mean():.4f}, RMSE={depth_df['rmse'].mean():.4f}, r={depth_df['r'].mean():.3f}")

        print("\nBy Region:")
        for region in summary_df['region'].unique():
            region_df = summary_df[summary_df['region'] == region]
            print(
                f"  {region}: n={len(region_df)}, MAE={region_df['mae'].mean():.4f}, r={region_df['r'].mean():.3f}")

        # Save results
        output_dir = Path('data/ismn')
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_df.to_csv(
            output_dir / 'ismn_validation_summary_multidepth.csv', index=False)
        print(
            f"\nSummary saved to: {output_dir / 'ismn_validation_summary_multidepth.csv'}")

        if all_results:
            results_df = pd.concat(all_results, ignore_index=True)
            results_df.to_csv(
                output_dir / 'ismn_daily_results_multidepth.csv', index=False)
            print(
                f"Daily results saved to: {output_dir / 'ismn_daily_results_multidepth.csv'}")
    else:
        print("\nNo sites were successfully validated!")

    return site_summaries


if __name__ == '__main__':
    main()
