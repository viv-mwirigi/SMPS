"""
SWPPS Utility Module.

Provides common utility functions for:
- Data caching
- Coordinate handling
- Time series utilities
- Logging helpers
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


logger = logging.getLogger("swpps.utils")


# =============================================================================
# CACHING UTILITIES
# =============================================================================

class SimpleCache:
    """
    Simple file-based cache for API responses.

    Stores JSON-serializable data with TTL-based expiration.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "./cache",
        default_ttl_hours: float = 24.0,
    ):
        """
        Initialize cache.

        Args:
            cache_dir: Directory for cache files
            default_ttl_hours: Default time-to-live in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = timedelta(hours=default_ttl_hours)

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.json"

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                data = json.load(f)

            # Check expiration
            expires = datetime.fromisoformat(data["expires"])
            if datetime.now() > expires:
                cache_path.unlink()
                return None

            return data["value"]

        except (json.JSONDecodeError, KeyError):
            cache_path.unlink(missing_ok=True)
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None,
    ) -> None:
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl: Time-to-live (defaults to default_ttl)
        """
        ttl = ttl or self.default_ttl
        expires = datetime.now() + ttl

        cache_path = self._get_cache_path(key)

        data = {
            "key": key,
            "value": value,
            "created": datetime.now().isoformat(),
            "expires": expires.isoformat(),
        }

        with open(cache_path, "w") as f:
            json.dump(data, f)

    def clear(self) -> int:
        """
        Clear all cached items.

        Returns:
            Number of items cleared
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count


# =============================================================================
# TIME SERIES UTILITIES
# =============================================================================

def resample_to_daily(
    df: pd.DataFrame,
    time_col: str = "datetime",
    aggregations: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Resample time series to daily frequency.

    Args:
        df: Input DataFrame
        time_col: Name of datetime column
        aggregations: Column -> aggregation method mapping

    Returns:
        Daily aggregated DataFrame
    """
    if time_col not in df.columns:
        if df.index.name == time_col or isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
        else:
            raise ValueError(f"Time column '{time_col}' not found")
    else:
        df = df.set_index(time_col)

    # Default aggregations
    default_agg = {
        "psi_kpa": "mean",
        "precipitation_mm": "sum",
        "et_mm": "sum",
        "temperature_c": "mean",
        "humidity_percent": "mean",
    }

    agg = aggregations or {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in agg:
            agg[col] = default_agg.get(col, "mean")

    return df.resample("D").agg(agg)


def fill_gaps(
    df: pd.DataFrame,
    max_gap_hours: int = 6,
    method: str = "linear",
) -> pd.DataFrame:
    """
    Fill small gaps in time series data.

    Args:
        df: Input DataFrame with DatetimeIndex
        max_gap_hours: Maximum gap size to fill
        method: Interpolation method

    Returns:
        DataFrame with gaps filled
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    # Find gaps
    gaps = df.index.to_series().diff()

    # Only interpolate small gaps
    mask = gaps <= pd.Timedelta(hours=max_gap_hours)

    result = df.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        result[col] = result[col].interpolate(
            method=method, limit=max_gap_hours)

    return result


def compute_rolling_stats(
    series: pd.Series,
    windows: List[int] = [7, 14, 30],
) -> pd.DataFrame:
    """
    Compute rolling statistics for multiple windows.

    Args:
        series: Input time series
        windows: List of window sizes (in periods)

    Returns:
        DataFrame with rolling mean, std, min, max for each window
    """
    results = {}

    for w in windows:
        results[f"rolling_mean_{w}"] = series.rolling(w, min_periods=1).mean()
        results[f"rolling_std_{w}"] = series.rolling(w, min_periods=1).std()
        results[f"rolling_min_{w}"] = series.rolling(w, min_periods=1).min()
        results[f"rolling_max_{w}"] = series.rolling(w, min_periods=1).max()

    return pd.DataFrame(results, index=series.index)


# =============================================================================
# COORDINATE UTILITIES
# =============================================================================

def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """
    Calculate great-circle distance between two points.

    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)

    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def is_tropical(latitude: float) -> bool:
    """Check if location is in tropical zone (±23.5°)."""
    return abs(latitude) <= 23.5


def get_timezone_offset(longitude: float) -> int:
    """
    Estimate timezone offset from longitude.

    Args:
        longitude: Longitude in degrees

    Returns:
        Estimated UTC offset in hours
    """
    return round(longitude / 15)


# =============================================================================
# DATA CONVERSION UTILITIES
# =============================================================================

def kpa_to_cm_water(kpa: float) -> float:
    """Convert kPa to cm water head."""
    return kpa * 10.197


def cm_water_to_kpa(cm: float) -> float:
    """Convert cm water head to kPa."""
    return cm / 10.197


def mm_to_m3_per_ha(mm: float) -> float:
    """Convert mm depth to m³/ha."""
    return mm * 10  # 1 mm = 10 m³/ha


def m3_per_ha_to_mm(m3_ha: float) -> float:
    """Convert m³/ha to mm depth."""
    return m3_ha / 10


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure logging for SWPPS.

    Args:
        level: Logging level
        log_file: Optional file path for log output
        format_string: Custom format string
    """
    format_str = format_string or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers: List[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
    )

    # Set SWPPS logger level
    logging.getLogger("swpps").setLevel(level)


class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time: float = 0
        self.elapsed: float = 0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self.start_time
        if self.name:
            logger.debug(f"{self.name}: {self.elapsed:.3f}s")

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed * 1000
