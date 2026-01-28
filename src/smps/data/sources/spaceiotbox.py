"""
SpaceIoTBox API Data Source Integration.

Provides access to Copernicus satellite data and various environmental datasets
(Weather, Agro, Datasets) through the SpaceIoTBox API.

API Features:
- Satellite Data: Integration with Copernicus services
- Weather Data: Climate and weather information
- Agro Data: Agricultural data services
- Datasets: General dataset management

Authentication: HTTP Basic Authentication
"""
import os
import requests
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import date, datetime, timedelta
from dataclasses import dataclass
import logging
from dotenv import load_dotenv

from smps.data.sources.base import (
    DataSource,
    WeatherSource,
    RemoteSensingSource,
    DataFetchRequest,
    DataFetchResult,
)
from smps.data.contracts import DailyWeather, RemoteSensingData, SoilProfile
from smps.core.exceptions import DataSourceError
from smps.core.types import SiteID

# Load environment variables
load_dotenv()

logger = logging.getLogger("smps.data.spaceiotbox")


@dataclass
class SpaceIoTBoxConfig:
    """Configuration for SpaceIoTBox API connection."""
    base_url: str = "http://127.0.0.1:8000"
    api_version: str = "v1"
    timeout: int = 30
    max_retries: int = 3
    backoff_base_seconds: float = 2.0
    backoff_max_seconds: float = 60.0

    @property
    def api_base_url(self) -> str:
        """Get versioned API base URL."""
        return f"{self.base_url}/{self.api_version}"


class SpaceIoTBoxClient:
    """
    HTTP client for SpaceIoTBox API with authentication and retry logic.

    Handles:
    - HTTP Basic Authentication
    - Request retries with exponential backoff
    - Error handling and logging
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        config: Optional[SpaceIoTBoxConfig] = None,
    ):
        """
        Initialize SpaceIoTBox client.

        Args:
            username: API username (or set SPACEIOTBOX_USERNAME env var)
            password: API password (or set SPACEIOTBOX_PASSWORD env var)
            config: Optional configuration override
        """
        self.config = config or SpaceIoTBoxConfig()

        self.username = username or os.getenv("SPACEIOTBOX_USERNAME")
        self.password = password or os.getenv("SPACEIOTBOX_PASSWORD")

        if not self.username or not self.password:
            logger.warning(
                "SpaceIoTBox credentials not provided. Set SPACEIOTBOX_USERNAME "
                "and SPACEIOTBOX_PASSWORD environment variables or pass to constructor."
            )

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "SMPS-Soil-Moisture-Prediction/1.0",
            "Accept": "application/json",
        })

        # Set up basic auth if credentials are provided
        if self.username and self.password:
            self.session.auth = (self.username, self.password)

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (relative to api_base_url)
            params: Query parameters
            json_data: JSON request body

        Returns:
            Response JSON data

        Raises:
            DataSourceError: On request failure after retries
        """
        url = f"{self.config.api_base_url}/{endpoint.lstrip('/')}"
        last_exc: Optional[Exception] = None

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    timeout=self.config.timeout,
                )

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and str(retry_after).strip().isdigit():
                        sleep_s = float(retry_after)
                    else:
                        sleep_s = min(
                            self.config.backoff_max_seconds,
                            self.config.backoff_base_seconds * (2 ** attempt)
                        )
                    logger.warning(
                        f"SpaceIoTBox rate limited (429). Retrying in {sleep_s:.1f}s "
                        f"(attempt {attempt + 1}/{self.config.max_retries})."
                    )
                    import time
                    time.sleep(sleep_s)
                    continue

                # Handle transient server errors
                if response.status_code in {500, 502, 503, 504}:
                    sleep_s = min(
                        self.config.backoff_max_seconds,
                        self.config.backoff_base_seconds * (2 ** attempt)
                    )
                    logger.warning(
                        f"SpaceIoTBox transient HTTP {response.status_code}. "
                        f"Retrying in {sleep_s:.1f}s "
                        f"(attempt {attempt + 1}/{self.config.max_retries})."
                    )
                    import time
                    time.sleep(sleep_s)
                    continue

                # Handle authentication error
                if response.status_code == 401:
                    raise DataSourceError(
                        "SpaceIoTBox authentication failed. Check credentials."
                    )

                # Handle other client errors
                if response.status_code >= 400:
                    try:
                        error_detail = response.json().get("detail", response.text)
                    except Exception:
                        error_detail = response.text
                    raise DataSourceError(
                        f"SpaceIoTBox API error ({response.status_code}): {error_detail}"
                    )

                response.raise_for_status()
                data = response.json()

                # Check for mock data indicators
                if self._is_mock_data(data):
                    raise DataSourceError(
                        "Mock data detected, no real data available")

                return data

            except requests.exceptions.RequestException as e:
                last_exc = e
                if attempt >= self.config.max_retries - 1:
                    break

                sleep_s = min(
                    self.config.backoff_max_seconds,
                    self.config.backoff_base_seconds * (2 ** attempt)
                )
                logger.warning(
                    f"SpaceIoTBox request failed: {e}. Retrying in {sleep_s:.1f}s "
                    f"(attempt {attempt + 1}/{self.config.max_retries})."
                )
                import time
                time.sleep(sleep_s)

        raise DataSourceError(
            f"SpaceIoTBox API error after {self.config.max_retries} attempts: {last_exc}"
        )

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make GET request."""
        return self._request("GET", endpoint, params=params)

    def post(
        self,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make POST request."""
        return self._request("POST", endpoint, params=params, json_data=json_data)

    def _is_mock_data(self, data: Dict[str, Any]) -> bool:
        """Check if response contains mock data indicators."""
        # Check for known mock data patterns
        if "data" in data and isinstance(data["data"], dict):
            mock_indicators = ["forecast", "Sunny", "temperature", "humidity"]
            data_str = str(data["data"]).lower()
            if any(indicator.lower() in data_str for indicator in mock_indicators):
                return True
        if "location" in data and data["location"] == "test":
            return True
        return False


class SpaceIoTBoxWeatherSource(WeatherSource):
    """
    Weather data source using SpaceIoTBox API.

    Provides access to climate and weather information through the
    SpaceIoTBox Weather API endpoints.
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize SpaceIoTBox Weather source.

        Args:
            username: API username (or set SPACEIOTBOX_USERNAME env var)
            password: API password (or set SPACEIOTBOX_PASSWORD env var)
            base_url: Override default API base URL
            cache_dir: Directory for caching responses
        """
        super().__init__("spaceiotbox_weather", cache_dir)

        config = SpaceIoTBoxConfig()
        if base_url:
            config.base_url = base_url

        self.client = SpaceIoTBoxClient(
            username=username,
            password=password,
            config=config,
        )

        # Site coordinates mapping (can be extended or loaded from config)
        self._site_coordinates: Dict[SiteID, Tuple[float, float]] = {}

    def register_site(self, site_id: SiteID, latitude: float, longitude: float):
        """Register site coordinates for weather data fetching."""
        self._site_coordinates[site_id] = (latitude, longitude)

    def _get_site_coordinates(self, site_id: SiteID) -> Tuple[float, float]:
        """Get coordinates for a site."""
        if site_id in self._site_coordinates:
            return self._site_coordinates[site_id]

        # Try to parse coordinates from site_id format "lat_lon"
        try:
            parts = str(site_id).split("_")
            if len(parts) >= 2:
                lat = float(parts[0])
                lon = float(parts[1])
                return lat, lon
        except (ValueError, IndexError):
            pass

        raise DataSourceError(
            f"Unknown site_id: {site_id}. Register coordinates using register_site()."
        )

    def fetch_daily_weather(self, request: DataFetchRequest) -> List[DailyWeather]:
        """
        Fetch daily weather data from SpaceIoTBox API.

        Args:
            request: Data fetch request with site_id and date range

        Returns:
            List of DailyWeather objects
        """
        lat, lon = self._get_site_coordinates(request.site_id)

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
        }

        # Add any additional parameters from request
        if request.parameters:
            params.update(request.parameters)

        try:
            response = self.client.get("weather/daily", params=params)
            return self._parse_weather_response(response, request.site_id)
        except DataSourceError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch weather data: {e}")
            raise DataSourceError(f"Weather data fetch failed: {e}")

    def _parse_weather_response(
        self,
        response: Dict[str, Any],
        site_id: SiteID,
    ) -> List[DailyWeather]:
        """Parse API response into DailyWeather objects."""
        weather_data = []

        # Handle different response formats
        daily_data = response.get("daily", response.get("data", []))

        if isinstance(daily_data, dict):
            # Response with separate arrays for each variable
            dates = daily_data.get("time", daily_data.get("date", []))

            for i, date_str in enumerate(dates):
                try:
                    weather_date = datetime.fromisoformat(date_str).date()

                    weather = DailyWeather(
                        date=weather_date,
                        site_id=site_id,
                        precipitation_mm=self._get_value(
                            daily_data, "precipitation", i),
                        et0_mm=self._get_value(
                            daily_data, "et0_fao_evapotranspiration", i),
                        temperature_mean_c=self._get_value(
                            daily_data, "temperature_2m_mean", i),
                        temperature_min_c=self._get_value(
                            daily_data, "temperature_2m_min", i),
                        temperature_max_c=self._get_value(
                            daily_data, "temperature_2m_max", i),
                        solar_radiation_mj_m2=self._get_value(
                            daily_data, "shortwave_radiation_sum", i
                        ),
                        relative_humidity_mean=self._get_value(
                            daily_data, "relative_humidity_2m_mean", i
                        ),
                        wind_speed_mean_m_s=self._get_value(
                            daily_data, "wind_speed_10m_mean", i
                        ),
                        source="spaceiotbox",
                        is_forecast=self._is_forecast_date(weather_date),
                    )
                    weather_data.append(weather)
                except Exception as e:
                    logger.warning(f"Failed to parse weather record {i}: {e}")
                    continue

        elif isinstance(daily_data, list):
            # Response with list of daily records
            for record in daily_data:
                try:
                    weather_date = datetime.fromisoformat(
                        record.get("date", record.get("time"))
                    ).date()

                    weather = DailyWeather(
                        date=weather_date,
                        site_id=site_id,
                        precipitation_mm=record.get(
                            "precipitation_mm", record.get("precipitation", 0.0)),
                        et0_mm=record.get("et0_mm", record.get(
                            "et0_fao_evapotranspiration", 0.0)),
                        temperature_mean_c=record.get(
                            "temperature_mean_c", record.get("temperature_2m_mean", 20.0)),
                        temperature_min_c=record.get(
                            "temperature_min_c", record.get("temperature_2m_min", 15.0)),
                        temperature_max_c=record.get(
                            "temperature_max_c", record.get("temperature_2m_max", 25.0)),
                        solar_radiation_mj_m2=record.get(
                            "solar_radiation_mj_m2", record.get(
                                "shortwave_radiation_sum", 15.0)
                        ),
                        relative_humidity_mean=record.get(
                            "relative_humidity_mean", record.get(
                                "relative_humidity_2m_mean", 60.0)
                        ),
                        wind_speed_mean_m_s=record.get(
                            "wind_speed_mean_m_s", record.get(
                                "wind_speed_10m_mean", 2.0)
                        ),
                        source="spaceiotbox",
                        is_forecast=self._is_forecast_date(weather_date),
                    )
                    weather_data.append(weather)
                except Exception as e:
                    logger.warning(f"Failed to parse weather record: {e}")
                    continue

        return weather_data

    def _get_value(
        self,
        data: Dict[str, Any],
        key: str,
        index: int,
    ) -> float:
        """Safely get value from array-based response."""
        # Try multiple possible key names
        possible_keys = [key, key.replace("_", ""), key.lower()]

        for k in possible_keys:
            if k in data:
                values = data[k]
                if isinstance(values, list) and len(values) > index:
                    val = values[index]
                    if val is not None:
                        return float(val)
        raise DataSourceError(
            f"Required value '{key}' not found in response data")

    def _is_forecast_date(self, weather_date: date) -> bool:
        """Check if date is in the future (forecast data)."""
        return weather_date > date.today()


class SpaceIoTBoxCopernicusSource(RemoteSensingSource):
    """
    Copernicus satellite data source using SpaceIoTBox API.

    Provides access to Copernicus services data including:
    - Sentinel-2 optical data (NDVI, EVI, LAI)
    - Sentinel-1 SAR data
    - Other Copernicus products
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize SpaceIoTBox Copernicus source.

        Args:
            username: API username (or set SPACEIOTBOX_USERNAME env var)
            password: API password (or set SPACEIOTBOX_PASSWORD env var)
            base_url: Override default API base URL
            cache_dir: Directory for caching responses
        """
        super().__init__("spaceiotbox_copernicus", cache_dir)

        config = SpaceIoTBoxConfig()
        if base_url:
            config.base_url = base_url

        self.client = SpaceIoTBoxClient(
            username=username,
            password=password,
            config=config,
        )

        self._site_coordinates: Dict[SiteID, Tuple[float, float]] = {}

    def register_site(self, site_id: SiteID, latitude: float, longitude: float):
        """Register site coordinates for satellite data fetching."""
        self._site_coordinates[site_id] = (latitude, longitude)

    def _get_site_coordinates(self, site_id: SiteID) -> Tuple[float, float]:
        """Get coordinates for a site."""
        if site_id in self._site_coordinates:
            return self._site_coordinates[site_id]

        try:
            parts = str(site_id).split("_")
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            pass

        raise DataSourceError(
            f"Unknown site_id: {site_id}. Register coordinates using register_site()."
        )

    def fetch_remote_sensing(self, request: DataFetchRequest) -> List[RemoteSensingData]:
        """
        Fetch remote sensing data from SpaceIoTBox Copernicus API.

        Args:
            request: Data fetch request with site_id and date range

        Returns:
            List of RemoteSensingData objects
        """
        lat, lon = self._get_site_coordinates(request.site_id)

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
        }

        # Add product filter if specified
        if request.parameters:
            if "product" in request.parameters:
                params["product"] = request.parameters["product"]
            if "collection" in request.parameters:
                params["collection"] = request.parameters["collection"]

        try:
            response = self.client.get("copernicus/data", params=params)
            return self._parse_copernicus_response(response, request.site_id)
        except DataSourceError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch Copernicus data: {e}")
            raise DataSourceError(f"Copernicus data fetch failed: {e}")

    def _parse_copernicus_response(
        self,
        response: Dict[str, Any],
        site_id: SiteID,
    ) -> List[RemoteSensingData]:
        """Parse Copernicus API response into RemoteSensingData objects."""
        rs_data = []
        records = response.get("data", response.get("observations", []))

        for record in records:
            try:
                obs_date = datetime.fromisoformat(
                    record.get("date", record.get("timestamp", ""))
                ).date()

                rs = RemoteSensingData(
                    date=obs_date,
                    site_id=site_id,
                    ndvi=record.get("ndvi"),
                    evi=record.get("evi"),
                    lai=record.get("lai"),
                    sar_vv_db=record.get("sar_vv", record.get("vv_db")),
                    sar_vh_db=record.get("sar_vh", record.get("vh_db")),
                    cloud_cover_percent=record.get(
                        "cloud_cover", record.get("cloud_cover_percent")
                    ),
                )
                rs_data.append(rs)
            except Exception as e:
                logger.warning(f"Failed to parse Copernicus record: {e}")
                continue

        return rs_data

    def fetch(self, request: DataFetchRequest) -> DataFetchResult:
        """Implementation of base fetch method."""
        start_time = datetime.now()

        try:
            errors = self.validate_request(request)
            if errors:
                return DataFetchResult(
                    data=None,
                    metadata={"request": request},
                    quality_score=0.0,
                    errors=errors,
                )

            # Try cache first
            cached_data = self._load_from_cache(request)
            if cached_data:
                rs_data = [
                    RemoteSensingData(**item)
                    for item in cached_data.get("data", [])
                ]
                return DataFetchResult(
                    data=rs_data,
                    metadata=cached_data.get("metadata", {}),
                    quality_score=cached_data.get("quality_score", 1.0),
                    cache_hit=True,
                    processing_time_ms=(
                        datetime.now() - start_time).total_seconds() * 1000,
                )

            # Fetch fresh data
            rs_data = self.fetch_remote_sensing(request)

            # Calculate quality score based on cloud cover
            quality_score = self._calculate_quality_score(rs_data)

            result = DataFetchResult(
                data=rs_data,
                metadata={
                    "source": self.name,
                    "site_id": request.site_id,
                    "date_range": f"{request.start_date} to {request.end_date}",
                    "count": len(rs_data),
                },
                quality_score=quality_score,
                cache_hit=False,
                processing_time_ms=(
                    datetime.now() - start_time).total_seconds() * 1000,
            )

            # Cache the result
            cache_data = {
                "data": [r.model_dump() for r in rs_data],
                "metadata": result.metadata,
                "quality_score": quality_score,
            }
            self._save_to_cache(request, cache_data)

            return result

        except Exception as e:
            logger.error(f"Failed to fetch Copernicus data: {e}")
            return DataFetchResult(
                data=None,
                metadata={"request": request, "error": str(e)},
                quality_score=0.0,
                errors=[str(e)],
            )

    def _calculate_quality_score(self, data: List[RemoteSensingData]) -> float:
        """Calculate quality score based on cloud cover."""
        if not data:
            return 0.0

        valid_count = 0
        total_cloud_cover = 0.0
        cloud_records = 0

        for record in data:
            if record.ndvi is not None or record.sar_vv_db is not None:
                valid_count += 1
            if record.cloud_cover_percent is not None:
                total_cloud_cover += record.cloud_cover_percent
                cloud_records += 1

        validity = valid_count / len(data) if data else 0
        avg_cloud = total_cloud_cover / cloud_records if cloud_records > 0 else 50
        cloud_score = max(0, (100 - avg_cloud) / 100)

        return round(0.6 * validity + 0.4 * cloud_score, 3)


class SpaceIoTBoxAgroSource(DataSource):
    """
    Agricultural data source using SpaceIoTBox API.

    Provides access to agricultural data services including:
    - Crop monitoring data
    - Soil information
    - Agricultural indices
    - Growth stage indicators
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize SpaceIoTBox Agro source.

        Args:
            username: API username (or set SPACEIOTBOX_USERNAME env var)
            password: API password (or set SPACEIOTBOX_PASSWORD env var)
            base_url: Override default API base URL
            cache_dir: Directory for caching responses
        """
        super().__init__("spaceiotbox_agro", cache_dir)

        config = SpaceIoTBoxConfig()
        if base_url:
            config.base_url = base_url

        self.client = SpaceIoTBoxClient(
            username=username,
            password=password,
            config=config,
        )

        self._site_coordinates: Dict[SiteID, Tuple[float, float]] = {}

    def register_site(self, site_id: SiteID, latitude: float, longitude: float):
        """Register site coordinates for agro data fetching."""
        self._site_coordinates[site_id] = (latitude, longitude)

    def _get_site_coordinates(self, site_id: SiteID) -> Tuple[float, float]:
        """Get coordinates for a site."""
        if site_id in self._site_coordinates:
            return self._site_coordinates[site_id]

        try:
            parts = str(site_id).split("_")
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            pass

        raise DataSourceError(
            f"Unknown site_id: {site_id}. Register coordinates using register_site()."
        )

    def fetch(self, request: DataFetchRequest) -> DataFetchResult:
        """
        Fetch agricultural data from SpaceIoTBox API.

        Args:
            request: Data fetch request with site_id and date range

        Returns:
            DataFetchResult with agricultural data
        """
        start_time = datetime.now()

        try:
            errors = self.validate_request(request)
            if errors:
                return DataFetchResult(
                    data=None,
                    metadata={"request": request},
                    quality_score=0.0,
                    errors=errors,
                )

            lat, lon = self._get_site_coordinates(request.site_id)

            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat(),
            }

            if request.parameters:
                params.update(request.parameters)

            response = self.client.get("agro/data", params=params)

            return DataFetchResult(
                data=response.get("data", response),
                metadata={
                    "source": self.name,
                    "site_id": request.site_id,
                    "date_range": f"{request.start_date} to {request.end_date}",
                },
                quality_score=1.0,
                cache_hit=False,
                processing_time_ms=(
                    datetime.now() - start_time).total_seconds() * 1000,
            )

        except Exception as e:
            logger.error(f"Failed to fetch agro data: {e}")
            return DataFetchResult(
                data=None,
                metadata={"request": request, "error": str(e)},
                quality_score=0.0,
                errors=[str(e)],
            )

    def fetch_soil_moisture(
        self,
        site_id: SiteID,
        start_date: date,
        end_date: date,
        depth_cm: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Fetch soil moisture data from Agro API.

        Args:
            site_id: Site identifier
            start_date: Start date for data
            end_date: End date for data
            depth_cm: Optional depth filter

        Returns:
            Soil moisture data dictionary
        """
        lat, lon = self._get_site_coordinates(site_id)

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }

        if depth_cm is not None:
            params["depth_cm"] = depth_cm

        return self.client.get("agro/soil-moisture", params=params)

    def fetch_crop_data(
        self,
        site_id: SiteID,
        crop_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch crop monitoring data.

        Args:
            site_id: Site identifier
            crop_type: Optional crop type filter

        Returns:
            Crop data dictionary
        """
        lat, lon = self._get_site_coordinates(site_id)

        params = {
            "latitude": lat,
            "longitude": lon,
        }

        if crop_type:
            params["crop_type"] = crop_type

        return self.client.get("agro/crop", params=params)


class SpaceIoTBoxDatasetsSource(DataSource):
    """
    General datasets source using SpaceIoTBox API.

    Provides access to various environmental datasets managed
    through the SpaceIoTBox Datasets API.
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize SpaceIoTBox Datasets source.

        Args:
            username: API username (or set SPACEIOTBOX_USERNAME env var)
            password: API password (or set SPACEIOTBOX_PASSWORD env var)
            base_url: Override default API base URL
            cache_dir: Directory for caching responses
        """
        super().__init__("spaceiotbox_datasets", cache_dir)

        config = SpaceIoTBoxConfig()
        if base_url:
            config.base_url = base_url

        self.client = SpaceIoTBoxClient(
            username=username,
            password=password,
            config=config,
        )

    def fetch(self, request: DataFetchRequest) -> DataFetchResult:
        """
        Fetch dataset from SpaceIoTBox API.

        Args:
            request: Data fetch request

        Returns:
            DataFetchResult with dataset data
        """
        start_time = datetime.now()

        try:
            errors = self.validate_request(request)
            if errors:
                return DataFetchResult(
                    data=None,
                    metadata={"request": request},
                    quality_score=0.0,
                    errors=errors,
                )

            params = {
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat(),
            }

            if request.parameters:
                params.update(request.parameters)

            response = self.client.get("datasets", params=params)

            return DataFetchResult(
                data=response.get("data", response),
                metadata={
                    "source": self.name,
                    "site_id": request.site_id,
                },
                quality_score=1.0,
                cache_hit=False,
                processing_time_ms=(
                    datetime.now() - start_time).total_seconds() * 1000,
            )

        except Exception as e:
            logger.error(f"Failed to fetch datasets: {e}")
            return DataFetchResult(
                data=None,
                metadata={"request": request, "error": str(e)},
                quality_score=0.0,
                errors=[str(e)],
            )

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List available datasets.

        Returns:
            List of dataset metadata dictionaries
        """
        response = self.client.get("datasets/list")
        return response.get("datasets", [])

    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get information about a specific dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dataset metadata dictionary
        """
        return self.client.get(f"datasets/{dataset_id}")


class SpaceIoTBoxUnifiedSource(DataSource):
    """
    Unified interface for all SpaceIoTBox API endpoints.

    Provides a single entry point to access Weather, Copernicus,
    Agro, and Datasets APIs with shared authentication.
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize unified SpaceIoTBox source.

        Args:
            username: API username (or set SPACEIOTBOX_USERNAME env var)
            password: API password (or set SPACEIOTBOX_PASSWORD env var)
            base_url: Override default API base URL
            cache_dir: Directory for caching responses
        """
        super().__init__("spaceiotbox", cache_dir)

        # Initialize all sub-sources with shared credentials
        self.weather = SpaceIoTBoxWeatherSource(
            username=username,
            password=password,
            base_url=base_url,
            cache_dir=cache_dir / "weather" if cache_dir else None,
        )

        self.copernicus = SpaceIoTBoxCopernicusSource(
            username=username,
            password=password,
            base_url=base_url,
            cache_dir=cache_dir / "copernicus" if cache_dir else None,
        )

        self.agro = SpaceIoTBoxAgroSource(
            username=username,
            password=password,
            base_url=base_url,
            cache_dir=cache_dir / "agro" if cache_dir else None,
        )

        self.datasets = SpaceIoTBoxDatasetsSource(
            username=username,
            password=password,
            base_url=base_url,
            cache_dir=cache_dir / "datasets" if cache_dir else None,
        )

    def register_site(self, site_id: SiteID, latitude: float, longitude: float):
        """Register site coordinates across all sub-sources."""
        self.weather.register_site(site_id, latitude, longitude)
        self.copernicus.register_site(site_id, latitude, longitude)
        self.agro.register_site(site_id, latitude, longitude)

    def fetch(self, request: DataFetchRequest) -> DataFetchResult:
        """
        Fetch data from appropriate SpaceIoTBox endpoint.

        The endpoint is determined by the 'source_type' parameter:
        - 'weather': Weather data
        - 'copernicus': Satellite data
        - 'agro': Agricultural data
        - 'datasets': General datasets

        Args:
            request: Data fetch request with optional 'source_type' parameter

        Returns:
            DataFetchResult from the appropriate source
        """
        source_type = (request.parameters or {}).get("source_type", "weather")

        if source_type == "weather":
            return self.weather.fetch(request)
        elif source_type == "copernicus":
            return self.copernicus.fetch(request)
        elif source_type == "agro":
            return self.agro.fetch(request)
        elif source_type == "datasets":
            return self.datasets.fetch(request)
        else:
            return DataFetchResult(
                data=None,
                metadata={"error": f"Unknown source_type: {source_type}"},
                quality_score=0.0,
                errors=[f"Unknown source_type: {source_type}"],
            )


# Convenience functions for quick access
def get_spaceiotbox_weather(
    site_id: SiteID,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    username: Optional[str] = None,
    password: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[DailyWeather]:
    """
    Quick function to fetch weather data from SpaceIoTBox.

    Args:
        site_id: Site identifier
        latitude: Site latitude
        longitude: Site longitude
        start_date: Start date
        end_date: End date
        username: Optional API username (uses env var if not provided)
        password: Optional API password (uses env var if not provided)
        base_url: Optional API base URL override

    Returns:
        List of DailyWeather objects
    """
    source = SpaceIoTBoxWeatherSource(
        username=username,
        password=password,
        base_url=base_url,
    )
    source.register_site(site_id, latitude, longitude)

    request = DataFetchRequest(
        site_id=site_id,
        start_date=start_date,
        end_date=end_date,
    )

    result = source.fetch(request)
    if result.success:
        return result.data
    else:
        raise DataSourceError(f"Failed to fetch weather: {result.errors}")


def get_spaceiotbox_satellite(
    site_id: SiteID,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    username: Optional[str] = None,
    password: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[RemoteSensingData]:
    """
    Quick function to fetch satellite data from SpaceIoTBox Copernicus.

    Args:
        site_id: Site identifier
        latitude: Site latitude
        longitude: Site longitude
        start_date: Start date
        end_date: End date
        username: Optional API username (uses env var if not provided)
        password: Optional API password (uses env var if not provided)
        base_url: Optional API base URL override

    Returns:
        List of RemoteSensingData objects
    """
    source = SpaceIoTBoxCopernicusSource(
        username=username,
        password=password,
        base_url=base_url,
    )
    source.register_site(site_id, latitude, longitude)

    request = DataFetchRequest(
        site_id=site_id,
        start_date=start_date,
        end_date=end_date,
    )

    result = source.fetch(request)
    if result.success:
        return result.data
    else:
        raise DataSourceError(
            f"Failed to fetch satellite data: {result.errors}")
