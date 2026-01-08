"""
Shared data fetching utilities for validation scripts.

Provides common functions for fetching soil, weather, and NDVI data
with caching, rate limiting, and fallback logic.
"""
import hashlib
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from smps.data.sources.base import DataFetchRequest
from smps.data.sources.isda_authenticated import IsdaAfricaAuthenticatedSource
from smps.data.sources.satellite import MODISNDVISource
from smps.data.sources.soil import MockSoilSource
from smps.data.sources.weather import OpenMeteoSource

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache'

# Rate limiting
_last_openmeteo_call = 0.0
OPENMETEO_RATE_LIMIT_SECONDS = 1.0


# =============================================================================
# SOIL DATA FETCHING
# =============================================================================

def fetch_soil_data(
    site_id: str,
    lat: float,
    lon: float,
    fallback_defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Fetch soil data from iSDA or fallback to defaults.

    Args:
        site_id: Site identifier for logging
        lat: Latitude
        lon: Longitude
        fallback_defaults: Custom fallback values (optional)

    Returns:
        Dictionary with soil properties
    """
    # Default fallback values (typical tropical soils)
    defaults = fallback_defaults or {
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

    try:
        isda = IsdaAfricaAuthenticatedSource()
        profile = isda.fetch_soil_profile(site_id, latitude=lat, longitude=lon)
        logger.info(
            f"  ✓ Soil data from iSDA: Sand={profile.sand_percent:.0f}%, "
            f"Clay={profile.clay_percent:.0f}%"
        )
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
        logger.warning(f"  iSDA failed for {site_id}: {e}")

        try:
            mock = MockSoilSource()
            profile = mock.fetch_soil_profile(site_id)
            return {
                'sand': profile.sand_percent,
                'clay': profile.clay_percent,
                'silt': profile.silt_percent,
                'porosity': profile.porosity,
                'field_capacity': profile.field_capacity,
                'wilting_point': profile.wilting_point,
                'organic_carbon': 1.5,
                'bulk_density': 1.3,
                'source': 'mock'
            }
        except Exception:
            return defaults


# =============================================================================
# WEATHER DATA FETCHING WITH CACHING
# =============================================================================

def _get_weather_cache_path(
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
    cache_dir: Optional[Path] = None
) -> Path:
    """Generate a unique cache file path for weather data."""
    cache_dir = cache_dir or (DEFAULT_CACHE_DIR / 'weather')
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = f"{lat:.4f}_{lon:.4f}_{start_date}_{end_date}"
    hash_key = hashlib.md5(key.encode()).hexdigest()[:12]
    return cache_dir / f"weather_{hash_key}.json"


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
        df_to_save = df.reset_index()
        df_to_save['date'] = df_to_save['date'].dt.strftime('%Y-%m-%d')
        with open(cache_path, 'w') as f:
            json.dump(df_to_save.to_dict(orient='records'), f)
    except Exception as e:
        logger.debug(f"Cache write failed: {e}")


def fetch_weather_data(
    site_id: str,
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
    rate_limit_seconds: float = OPENMETEO_RATE_LIMIT_SECONDS
) -> Optional[pd.DataFrame]:
    """
    Fetch weather data from Open-Meteo with caching and rate limiting.

    Args:
        site_id: Site identifier
        lat: Latitude
        lon: Longitude
        start_date: Start date
        end_date: End date
        use_cache: Whether to use caching
        cache_dir: Custom cache directory
        rate_limit_seconds: Minimum seconds between API calls

    Returns:
        DataFrame with weather data or None on failure
    """
    global _last_openmeteo_call

    # Check cache first
    if use_cache:
        cache_path = _get_weather_cache_path(
            lat, lon, start_date, end_date, cache_dir)
        cached = _load_weather_from_cache(cache_path)
        if cached is not None:
            logger.debug(f"Using cached weather for {site_id}")
            return cached

    # Rate limiting
    elapsed = time.time() - _last_openmeteo_call
    if elapsed < rate_limit_seconds:
        time.sleep(rate_limit_seconds - elapsed)

    try:
        weather_source = OpenMeteoSource()
        request = DataFetchRequest(
            site_id=site_id,
            start_date=start_date,
            end_date=end_date,
            parameters={"include_forecast": False}
        )

        # Inject coordinates
        weather_source._get_site_coordinates = lambda s: (lat, lon)
        result = weather_source.fetch(request)
        _last_openmeteo_call = time.time()

        if result.success and result.data:
            df = pd.DataFrame([d.model_dump() for d in result.data])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()

            # Save to cache
            if use_cache:
                _save_weather_to_cache(cache_path, df)

            return df
        else:
            logger.warning(f"Weather fetch returned no data for {site_id}")
            return None

    except Exception as e:
        logger.error(f"Weather fetch failed for {site_id}: {e}")
        return None


# =============================================================================
# NDVI DATA FETCHING
# =============================================================================

def fetch_ndvi_data(
    site_id: str,
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
    generate_synthetic_on_failure: bool = True
) -> Optional[pd.DataFrame]:
    """
    Fetch NDVI data from MODIS or generate synthetic data.

    Args:
        site_id: Site identifier
        lat: Latitude
        lon: Longitude
        start_date: Start date
        end_date: End date
        generate_synthetic_on_failure: If True, generate synthetic NDVI on failure

    Returns:
        DataFrame with NDVI data or None
    """
    try:
        ndvi_source = MODISNDVISource()
        request = DataFetchRequest(
            site_id=site_id,
            start_date=start_date,
            end_date=end_date
        )
        ndvi_source._get_site_coordinates = lambda s: (lat, lon)
        result = ndvi_source.fetch(request)

        if result.success and result.data:
            df = pd.DataFrame([d.model_dump() for d in result.data])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            return df

    except Exception as e:
        logger.warning(f"NDVI fetch failed for {site_id}: {e}")

    if generate_synthetic_on_failure:
        return generate_synthetic_ndvi(lat, start_date, end_date)

    return None


def generate_synthetic_ndvi(
    lat: float,
    start_date: date,
    end_date: date
) -> pd.DataFrame:
    """
    Generate synthetic NDVI based on latitude and seasonality.

    Args:
        lat: Latitude (determines seasonal pattern)
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with synthetic NDVI values
    """
    dates = pd.date_range(start_date, end_date, freq='D')
    doy = dates.dayofyear

    # Seasonal patterns based on latitude
    if abs(lat) < 15:
        # Tropics: weak seasonality, higher baseline
        ndvi_vals = 0.55 + 0.15 * np.sin(2 * np.pi * (doy - 60) / 365)
    elif lat > 0:
        # Northern hemisphere: summer peak
        ndvi_vals = 0.35 + 0.35 * np.sin(2 * np.pi * (doy - 100) / 365)
    else:
        # Southern hemisphere: offset by 6 months
        ndvi_vals = 0.35 + 0.35 * np.sin(2 * np.pi * (doy + 80) / 365)

    # Add noise
    noise = np.random.normal(0, 0.03, len(dates))
    ndvi_vals = np.clip(ndvi_vals + noise, 0.1, 0.95)

    return pd.DataFrame({'ndvi': ndvi_vals}, index=dates)


# =============================================================================
# TEXTURE CLASSIFICATION
# =============================================================================

def determine_soil_texture(sand: float, clay: float) -> str:
    """
    Determine soil texture class from sand and clay percentages.

    Args:
        sand: Sand percentage (0-100)
        clay: Clay percentage (0-100)

    Returns:
        Texture class string
    """
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
