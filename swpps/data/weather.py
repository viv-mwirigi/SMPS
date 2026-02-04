"""
Weather data fetching for SWPPS.

Supports Open-Meteo API for both historical and forecast data.
"""

import requests
import json
import time
import logging
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

from swpps.core.types import DailyWeather
from swpps.core.constants import OPENMETEO_CONFIG
from swpps.core.exceptions import DataFetchError

logger = logging.getLogger("swpps.data.weather")


@dataclass
class WeatherFetchRequest:
    """Request for weather data."""
    latitude: float
    longitude: float
    start_date: date
    end_date: date
    timezone: str = "UTC"


class OpenMeteoClient:
    """
    Client for Open-Meteo weather API.

    Supports both historical (ERA5) and forecast data.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.config = OPENMETEO_CONFIG
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SWPPS-Irrigation-System/1.0'
        })
        self.last_request_time = 0.0  # Track last request time

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_daily_weather(
        self,
        request: WeatherFetchRequest
    ) -> List[DailyWeather]:
        """
        Fetch daily weather data.

        Automatically determines whether to use historical or forecast API.

        Args:
            request: Weather fetch request

        Returns:
            List of DailyWeather objects
        """
        today = date.today()

        weather_data = []

        # Historical data (if needed)
        historical_end = min(request.end_date, today - timedelta(days=2))
        if request.start_date <= historical_end:
            try:
                hist_data = self._fetch_historical(
                    request.latitude,
                    request.longitude,
                    request.start_date,
                    historical_end,
                    request.timezone,
                )
                weather_data.extend(hist_data)
            except Exception as e:
                logger.warning("Historical fetch failed: %s", e)

        # Recent/forecast data
        recent_start = max(request.start_date, today - timedelta(days=1))
        if recent_start <= request.end_date:
            try:
                forecast_data = self._fetch_forecast(
                    request.latitude,
                    request.longitude,
                    recent_start,
                    request.end_date,
                    request.timezone,
                )
                weather_data.extend(forecast_data)
            except Exception as e:
                logger.warning("Forecast fetch failed: %s", e)

        # Deduplicate by date
        seen_dates = set()
        unique_data = []
        for w in weather_data:
            if w.date not in seen_dates:
                unique_data.append(w)
                seen_dates.add(w.date)

        if not unique_data:
            raise DataFetchError(
                f"No weather data available for the requested period {request.start_date} to {request.end_date}")

        return sorted(unique_data, key=lambda w: w.date)

    def _fetch_historical(
        self,
        lat: float,
        lon: float,
        start_date: date,
        end_date: date,
        timezone: str,
    ) -> List[DailyWeather]:
        """Fetch historical data from ERA5 archive."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": ",".join(self.config["daily_variables"]),
            "timezone": timezone,
        }

        data = self._make_request(self.config["historical_url"], params)
        return self._parse_daily_response(data)

    def _fetch_forecast(
        self,
        lat: float,
        lon: float,
        start_date: date,
        end_date: date,
        timezone: str,
    ) -> List[DailyWeather]:
        """Fetch forecast/recent data."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join(self.config["daily_variables"]),
            "timezone": timezone,
            "past_days": 7,
            "forecast_days": 16,
        }

        data = self._make_request(self.config["forecast_url"], params)

        # Filter to requested date range
        all_weather = self._parse_daily_response(data)
        return [w for w in all_weather if start_date <= w.date <= end_date]

    def _make_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request with retries and rate limiting."""
        import time

        max_retries = self.config.get("max_retries", 5)
        backoff = self.config.get("backoff_seconds", 2.0)
        timeout = self.config.get("timeout_seconds", 30)
        min_request_interval = self.config.get(
            "min_request_interval", 1.0)  # Minimum 1 second between requests

        # Enforce minimum interval between requests
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < min_request_interval:
            sleep_time = min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)

        last_error = None

        for attempt in range(max_retries):
            try:
                self.last_request_time = time.time()
                response = self.session.get(
                    url, params=params, timeout=timeout)

                if response.status_code == 429:
                    # Rate limited
                    retry_after = float(response.headers.get(
                        "Retry-After", backoff * (2 ** attempt)))
                    logger.warning("Rate limited, waiting %.1fs", retry_after)
                    time.sleep(retry_after)
                    continue

                if response.status_code >= 500:
                    # Server error, retry
                    time.sleep(backoff * (2 ** attempt))
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                last_error = e
                logger.warning("Request failed (attempt %d/%d): %s",
                               attempt + 1, max_retries, e)
                time.sleep(backoff * (2 ** attempt))

        raise DataFetchError(
            f"Failed to fetch weather after {max_retries} attempts: {last_error}")

    def _parse_daily_response(self, data: Dict[str, Any]) -> List[DailyWeather]:
        """Parse Open-Meteo daily response."""
        daily = data.get("daily", {})

        if not daily or "time" not in daily:
            return []

        times = daily["time"]
        n_days = len(times)

        def get_values(key: str, default: float = 0.0) -> List[float]:
            """Get values with fallback."""
            values = daily.get(key, [])
            if not values:
                return [default] * n_days
            return [v if v is not None else default for v in values]

        # Extract all variables
        temp_max = get_values("temperature_2m_max", 25.0)
        temp_min = get_values("temperature_2m_min", 15.0)
        temp_mean = get_values("temperature_2m_mean", 20.0)

        # Calculate mean if not provided
        if not daily.get("temperature_2m_mean"):
            temp_mean = [(mx + mn) / 2 for mx, mn in zip(temp_max, temp_min)]

        precip = get_values("precipitation_sum", 0.0)
        et0 = get_values("et0_fao_evapotranspiration", 3.0)

        # Handle et0 that might be named differently
        if not daily.get("et0_fao_evapotranspiration"):
            et0 = get_values("et0_fao_evapotranspiration_sum", 3.0)

        radiation = get_values("shortwave_radiation_sum", 15.0)
        humidity = get_values("relative_humidity_2m_mean", 60.0)
        wind = get_values("wind_speed_10m_mean", 2.0)

        # Also try wind_speed_10m_max as fallback
        if not daily.get("wind_speed_10m_mean"):
            wind = get_values("wind_speed_10m_max", 2.0)

        weather_list = []
        for i in range(n_days):
            try:
                d = datetime.fromisoformat(times[i]).date()

                weather_list.append(DailyWeather(
                    date=d,
                    precipitation_mm=max(0, precip[i] or 0),
                    et0_mm=max(0.1, et0[i] or 3.0),
                    temperature_mean_c=temp_mean[i] or 20.0,
                    temperature_min_c=temp_min[i] or 15.0,
                    temperature_max_c=temp_max[i] or 25.0,
                    relative_humidity_mean=humidity[i] or 60.0,
                    solar_radiation_mj_m2=radiation[i] or 15.0,
                    wind_speed_m_s=wind[i] or 2.0,
                ))
            except Exception as e:
                logger.warning("Failed to parse day %s: %s", times[i], e)

        return weather_list


def fetch_weather_for_plot(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    timezone: str = "UTC",
) -> List[DailyWeather]:
    """
    Convenience function to fetch weather for a plot.

    Args:
        latitude: Plot latitude
        longitude: Plot longitude
        start_date: Start date
        end_date: End date
        timezone: Timezone string

    Returns:
        List of DailyWeather objects
    """
    client = OpenMeteoClient()
    request = WeatherFetchRequest(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        timezone=timezone,
    )
    return client.fetch_daily_weather(request)
