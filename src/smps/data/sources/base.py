"""
Abstract base classes for data sources.
Provides unified interface for all data fetching.
"""
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import hashlib
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from smps.core.types import SiteID
from smps.core.exceptions import DataSourceError
from smps.data.contracts import (
    DailyWeather, SoilProfile, RemoteSensingData,
    IrrigationRecord, SoilMoistureObservation
)


@dataclass
class DataFetchRequest:
    """Request for data fetching"""
    site_id: SiteID
    start_date: date
    end_date: date
    parameters: Optional[Dict[str, Any]] = None
    priority: int = 1  # 1=high, 2=medium, 3=low

    def cache_key(self) -> str:
        """Generate cache key for this request"""
        data = {
            'site_id': self.site_id,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'parameters': self.parameters or {}
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()


@dataclass
class DataFetchResult:
    """Result of data fetching operation"""
    data: Any  # Could be DataFrame, list of objects, etc.
    metadata: Dict[str, Any]
    quality_score: float = 1.0
    cache_hit: bool = False
    processing_time_ms: float = 0.0
    errors: Optional[List[str]] = None

    @property
    def success(self) -> bool:
        """Whether fetch was successful"""
        return self.errors is None or len(self.errors) == 0


class DataSource(ABC):
    """
    Abstract base class for all data sources.
    Implements common patterns: caching, retries, logging.
    """

    def __init__(self, name: str, cache_dir: Optional[Path] = None):
        self.name = name
        self.cache_dir = cache_dir or Path(f"./data/cache/{name}")
        self.cache_enabled = cache_dir is not None
        self.logger = logging.getLogger(f"smps.data.{name}")

        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch(self, request: DataFetchRequest) -> DataFetchResult:
        """
        Fetch data for given request.
        Must be implemented by concrete sources.
        """
        pass

    def fetch_batch(self, requests: List[DataFetchRequest],
                   max_workers: int = 4) -> Dict[SiteID, DataFetchResult]:
        """
        Fetch data for multiple requests in parallel.
        """
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all fetch tasks
            future_to_request = {
                executor.submit(self.fetch, req): req
                for req in requests
            }

            # Collect results as they complete
            for future in as_completed(future_to_request):
                request = future_to_request[future]
                try:
                    result = future.result(timeout=30)
                    results[request.site_id] = result
                except Exception as e:
                    self.logger.error(f"Failed to fetch for {request.site_id}: {e}")
                    results[request.site_id] = DataFetchResult(
                        data=None,
                        metadata={'error': str(e)},
                        quality_score=0.0,
                        errors=[str(e)]
                    )

        return results

    def _get_cache_path(self, request: DataFetchRequest) -> Path:
        """Get cache file path for request"""
        cache_key = request.cache_key()
        return self.cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, request: DataFetchRequest) -> Optional[Any]:
        """Load data from cache if available and fresh"""
        if not self.cache_enabled:
            return None

        cache_path = self._get_cache_path(request)

        if not cache_path.exists():
            return None

        # Check cache freshness (default: 24 hours for weather, 7 days for static)
        cache_age_hours = (datetime.now().timestamp() - cache_path.stat().st_mtime) / 3600
        max_age = self._get_max_cache_age_hours(request)

        if cache_age_hours > max_age:
            return None

        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            self.logger.debug(f"Cache hit for {request.site_id}")
            return cached_data
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_to_cache(self, request: DataFetchRequest, data: Any):
        """Save data to cache"""
        if not self.cache_enabled:
            return

        cache_path = self._get_cache_path(request)

        try:
            # Ensure data is JSON serializable
            serializable_data = self._make_json_serializable(data)

            with open(cache_path, 'w') as f:
                json.dump(serializable_data, f, default=str)

            self.logger.debug(f"Cached data for {request.site_id}")
        except Exception as e:
            self.logger.warning(f"Failed to cache data: {e}")

    def _make_json_serializable(self, data: Any) -> Any:
        """Convert data to JSON serializable format"""
        if isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif isinstance(data, (list, tuple)):
            return [self._make_json_serializable(item) for item in data]
        elif isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        elif hasattr(data, 'dict'):  # Pydantic models
            return data.dict()
        elif hasattr(data, 'to_dict'):  # Pandas, etc.
            return data.to_dict()
        else:
            return str(data)

    def _get_max_cache_age_hours(self, request: DataFetchRequest) -> float:
        """Get maximum cache age in hours for this request type"""
        # Override in subclasses based on data volatility
        return 24.0  # Default: 24 hours

    def validate_request(self, request: DataFetchRequest) -> List[str]:
        """Validate fetch request. Returns list of errors or empty list if valid."""
        errors = []

        if request.start_date > request.end_date:
            errors.append("start_date must be <= end_date")

        # Check date range limits
        max_days = self._get_max_date_range_days()
        days_diff = (request.end_date - request.start_date).days
        if days_diff > max_days:
            errors.append(f"Date range exceeds maximum of {max_days} days")

        return errors

    def _get_max_date_range_days(self) -> int:
        """Maximum date range allowed by this source"""
        return 365  # Default: 1 year

    def get_metadata(self) -> Dict[str, Any]:
        """Get source metadata (resolution, coverage, availability, etc.)"""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "cache_enabled": self.cache_enabled,
            "cache_dir": str(self.cache_dir) if self.cache_enabled else None,
            "max_date_range_days": self._get_max_date_range_days(),
        }


class WeatherSource(DataSource):
    """Abstract base class for weather data sources"""

    @abstractmethod
    def fetch_daily_weather(self, request: DataFetchRequest) -> List[DailyWeather]:
        """Fetch daily weather data"""
        pass

    def fetch(self, request: DataFetchRequest) -> DataFetchResult:
        """Implementation of base fetch method"""
        start_time = datetime.now()

        try:
            # Validate request
            errors = self.validate_request(request)
            if errors:
                return DataFetchResult(
                    data=None,
                    metadata={'request': request},
                    quality_score=0.0,
                    errors=errors
                )

            # Try cache
            cached_data = self._load_from_cache(request)
            if cached_data:
                # Convert cached data back to DailyWeather objects
                weather_data = [
                    DailyWeather(**item) for item in cached_data.get('data', [])
                ]

                return DataFetchResult(
                    data=weather_data,
                    metadata=cached_data.get('metadata', {}),
                    quality_score=cached_data.get('quality_score', 1.0),
                    cache_hit=True,
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
                )

            # Fetch fresh data
            weather_data = self.fetch_daily_weather(request)

            # Calculate quality score
            quality_score = self._calculate_quality_score(weather_data, request)

            # Prepare result
            result = DataFetchResult(
                data=weather_data,
                metadata={
                    'source': self.name,
                    'site_id': request.site_id,
                    'date_range': f"{request.start_date} to {request.end_date}",
                    'count': len(weather_data),
                    'parameters': request.parameters
                },
                quality_score=quality_score,
                cache_hit=False,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

            # Cache the result
            cache_data = {
                'data': [w.dict() for w in weather_data],
                'metadata': result.metadata,
                'quality_score': quality_score
            }
            self._save_to_cache(request, cache_data)

            return result

        except Exception as e:
            self.logger.error(f"Failed to fetch weather data: {e}")
            return DataFetchResult(
                data=None,
                metadata={'request': request, 'error': str(e)},
                quality_score=0.0,
                errors=[str(e)]
            )

    def _calculate_quality_score(self, data: List[DailyWeather],
                               request: DataFetchRequest) -> float:
        """Calculate quality score for weather data"""
        if not data:
            return 0.0

        # Expected number of days
        expected_days = (request.end_date - request.start_date).days + 1

        # Completeness
        completeness = len(data) / expected_days

        # Data quality metrics
        valid_records = 0
        for record in data:
            # Check for reasonable values
            if (0 <= record.precipitation_mm <= 500 and  # Max 500mm/day
                0 <= record.et0_mm <= 20 and            # Max 20mm/day ET0
                -50 <= record.temperature_min_c <= 60 and
                -50 <= record.temperature_max_c <= 60):
                valid_records += 1

        validity = valid_records / len(data) if data else 0

        # Combined score
        score = 0.6 * completeness + 0.4 * validity

        return round(score, 3)


class SoilSource(DataSource):
    """Abstract base class for soil data sources"""

    @abstractmethod
    def fetch_soil_profile(self, site_id: SiteID,
                         depth_cm: Optional[int] = None) -> SoilProfile:
        """Fetch soil profile data"""
        pass

    def _get_max_cache_age_hours(self, request: DataFetchRequest) -> float:
        """Soil data changes very slowly - cache for 30 days"""
        return 30 * 24  # 30 days


class RemoteSensingSource(DataSource):
    """Abstract base class for remote sensing data sources"""

    @abstractmethod
    def fetch_remote_sensing(self, request: DataFetchRequest) -> List[RemoteSensingData]:
        """Fetch remote sensing data"""
        pass


class IrrigationSource(DataSource):
    """Abstract base class for irrigation data sources"""

    @abstractmethod
    def fetch_irrigation_records(self, request: DataFetchRequest) -> List[IrrigationRecord]:
        """Fetch irrigation records"""
        pass


class SensorSource(DataSource):
    """Abstract base class for in-situ sensor data sources"""

    @abstractmethod
    def fetch_sensor_data(self, request: DataFetchRequest) -> List[SoilMoistureObservation]:
        """Fetch sensor data"""
        pass


class DataSourceRegistry:
    """Registry for managing data sources"""

    def __init__(self):
        self._sources: Dict[str, DataSource] = {}
        self.logger = logging.getLogger("smps.data.registry")

    def register(self, source: DataSource):
        """Register a data source"""
        self._sources[source.name] = source
        self.logger.info(f"Registered data source: {source.name}")

    def get(self, name: str) -> Optional[DataSource]:
        """Get data source by name"""
        return self._sources.get(name)

    def get_all(self) -> Dict[str, DataSource]:
        """Get all registered sources"""
        return self._sources.copy()

    def fetch_all(self, request: DataFetchRequest,
                 source_names: Optional[List[str]] = None) -> Dict[str, DataFetchResult]:
        """
        Fetch data from all (or specified) sources.
        Returns dict mapping source name to fetch result.
        """
        results = {}

        sources_to_fetch = self._sources
        if source_names:
            sources_to_fetch = {
                name: source for name, source in self._sources.items()
                if name in source_names
            }

        for name, source in sources_to_fetch.items():
            try:
                result = source.fetch(request)
                results[name] = result
            except Exception as e:
                self.logger.error(f"Source {name} failed: {e}")
                results[name] = DataFetchResult(
                    data=None,
                    metadata={'error': str(e)},
                    quality_score=0.0,
                    errors=[str(e)]
                )

        return results