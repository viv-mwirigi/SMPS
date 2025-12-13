"""
Concrete weather data source implementations.
Currently supports Open-Meteo (historical and forecast).
"""
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
import requests
import json
from pathlib import Path

from smps.data.sources.base import WeatherSource, DataFetchRequest
from smps.data.contracts import DailyWeather
from smps.core.config import get_config
from smps.core.exceptions import DataSourceError
from smps.core.types import SiteID


class OpenMeteoSource(WeatherSource):
    """
    Weather data source using Open-Meteo API.
    Supports historical data (ERA5) and forecasts.
    """

    # API endpoints
    HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/era5"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    # Available variables
    HOURLY_VARIABLES = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "snowfall",
        "cloud_cover",
        "shortwave_radiation",
        "wind_speed_10m",
        "wind_direction_10m",
        "soil_temperature_0_to_7cm",
        "soil_temperature_7_to_28cm",
        "soil_temperature_28_to_100cm",
        "soil_moisture_0_to_7cm",
        "soil_moisture_7_to_28cm",
        "soil_moisture_28_to_100cm",
        "et0_fao_evapotranspiration"
    ]

    DAILY_VARIABLES = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration_sum",
        "wind_speed_10m_max",
        "wind_direction_10m_dominant"
    ]

    def __init__(self, cache_dir: Optional[Path] = None):
        super().__init__("open_meteo", cache_dir)
        self.config = get_config()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Smps-Soil-Moisture-Prediction/1.0'
        })

    def fetch_daily_weather(self, request: DataFetchRequest) -> List[DailyWeather]:
        """
        Fetch daily weather data from Open-Meteo.
        Automatically decides whether to use historical or forecast API.
        """
        # Get site coordinates (simplified - in real system, this would come from site config)
        site_lat, site_lon = self._get_site_coordinates(request.site_id)

        # Determine if we need forecast data
        today = date.today()
        use_forecast = request.end_date > today

        if use_forecast:
            # Split request into historical + forecast parts
            historical_end = min(request.end_date, today)
            forecast_start = max(request.start_date, today + timedelta(days=1))

            weather_data = []

            # Fetch historical part
            if request.start_date <= historical_end:
                hist_request = DataFetchRequest(
                    site_id=request.site_id,
                    start_date=request.start_date,
                    end_date=historical_end,
                    parameters=request.parameters
                )
                hist_data = self._fetch_historical(hist_request, site_lat, site_lon)
                weather_data.extend(hist_data)

            # Fetch forecast part
            if forecast_start <= request.end_date:
                forecast_request = DataFetchRequest(
                    site_id=request.site_id,
                    start_date=forecast_start,
                    end_date=request.end_date,
                    parameters=request.parameters
                )
                forecast_data = self._fetch_forecast(forecast_request, site_lat, site_lon)
                weather_data.extend(forecast_data)
        else:
            # All historical
            weather_data = self._fetch_historical(request, site_lat, site_lon)

        return weather_data

    def _fetch_historical(self, request: DataFetchRequest,
                         latitude: float, longitude: float) -> List[DailyWeather]:
        """Fetch historical weather data"""
        url = self.HISTORICAL_URL

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
            "daily": ",".join(self.DAILY_VARIABLES),
            "timezone": "auto"
        }

        # Add hourly variables if requested
        if request.parameters and request.parameters.get("include_hourly", False):
            params["hourly"] = ",".join(request.parameters.get("hourly_variables",
                                                              self.HOURLY_VARIABLES[:5]))

        self.logger.info(f"Fetching historical weather: {params}")

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            return self._parse_daily_response(data, is_forecast=False)

        except requests.exceptions.RequestException as e:
            raise DataSourceError(f"Open-Meteo API error: {e}")

    def _fetch_forecast(self, request: DataFetchRequest,
                       latitude: float, longitude: float) -> List[DailyWeather]:
        """Fetch weather forecast data"""
        url = self.FORECAST_URL

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
            "daily": ",".join(self.DAILY_VARIABLES),
            "timezone": "auto"
        }

        self.logger.info(f"Fetching forecast weather: {params}")

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            return self._parse_daily_response(data, is_forecast=True)

        except requests.exceptions.RequestException as e:
            raise DataSourceError(f"Open-Meteo forecast API error: {e}")

    def _parse_daily_response(self, data: Dict[str, Any],
                            is_forecast: bool) -> List[DailyWeather]:
        """Parse Open-Meteo API response into DailyWeather objects"""
        daily_data = data.get("daily", {})

        if not daily_data:
            raise DataSourceError("No daily data in response")

        # Extract arrays
        dates = daily_data.get("time", [])
        if not dates:
            return []

        # Map Open-Meteo field names to our field names
        field_mapping = {
            "temperature_2m_max": "temperature_max_c",
            "temperature_2m_min": "temperature_min_c",
            "temperature_2m_mean": "temperature_mean_c",
            "precipitation_sum": "precipitation_mm",
            "shortwave_radiation_sum": "solar_radiation_mj_m2",
            "et0_fao_evapotranspiration_sum": "et0_mm",
            "wind_speed_10m_max": "wind_speed_mean_m_s"
        }

        weather_records = []

        for i, date_str in enumerate(dates):
            try:
                record_date = date.fromisoformat(date_str)

                # Build record data
                record_data = {
                    "date": record_date,
                    "site_id": "unknown",  # Will be set by caller
                    "source": "open_meteo",
                    "is_forecast": is_forecast,
                    "forecast_horizon_days": (record_date - date.today()).days if is_forecast else None
                }

                # Map fields
                for api_field, our_field in field_mapping.items():
                    value = daily_data.get(api_field, [])
                    if i < len(value):
                        record_data[our_field] = float(value[i])
                    else:
                        record_data[our_field] = None

                # Set defaults for missing fields
                if record_data.get("relative_humidity_mean") is None:
                    record_data["relative_humidity_mean"] = 60.0  # Default

                # Calculate VPD if we have temperature and humidity
                if (record_data.get("temperature_mean_c") is not None and
                    record_data.get("relative_humidity_mean") is not None):
                    vpd = self._calculate_vpd(
                        record_data["temperature_mean_c"],
                        record_data["relative_humidity_mean"]
                    )
                    record_data["vapor_pressure_deficit_kpa"] = vpd

                # Create validated record
                weather_record = DailyWeather(**record_data)
                weather_records.append(weather_record)

            except Exception as e:
                self.logger.warning(f"Failed to parse record for {date_str}: {e}")
                continue

        return weather_records

    def _calculate_vpd(self, temperature_c: float,
                      relative_humidity: float) -> float:
        """Calculate Vapor Pressure Deficit (kPa)"""
        # Tetens formula for saturation vapor pressure
        svp = 0.6108 * np.exp(17.27 * temperature_c / (temperature_c + 237.3))

        # Actual vapor pressure
        avp = svp * relative_humidity / 100

        # VPD
        vpd = svp - avp

        return round(vpd, 3)

    def _get_site_coordinates(self, site_id: SiteID) -> Tuple[float, float]:
        """
        Get coordinates for a site.
        In a real system, this would query a site database.
        """
        # Mock implementation - would come from site configuration
        site_coords = {
            "test_site_001": (35.222866, 9.090245),
            "test_site_002": (34.0, 8.0),
        }

        if site_id in site_coords:
            return site_coords[site_id]
        else:
            # Default coordinates (center of Tunisia)
            return (34.0, 9.0)

    def _get_max_date_range_days(self) -> int:
        """Open-Meteo limits (historical: years, forecast: 16 days)"""
        return 365 * 5  # 5 years for historical

    def get_metadata(self) -> Dict[str, Any]:
        """Get source metadata"""
        base_metadata = super().get_metadata()
        base_metadata.update({
            "api_provider": "Open-Meteo",
            "data_types": ["historical", "forecast"],
            "historical_depth": "1940-present",
            "forecast_horizon": "16 days",
            "spatial_resolution": "11km",
            "update_frequency": "daily",
            "variables_available": self.HOURLY_VARIABLES + self.DAILY_VARIABLES
        })
        return base_metadata


class MockWeatherSource(WeatherSource):
    """
    Mock weather source for testing and development.
    Generates realistic synthetic weather data.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        super().__init__("mock_weather", cache_dir)
        self.seasonal_patterns = self._initialize_seasonal_patterns()

    def fetch_daily_weather(self, request: DataFetchRequest) -> List[DailyWeather]:
        """Generate synthetic weather data"""

        weather_records = []

        current_date = request.start_date
        while current_date <= request.end_date:
            # Generate seasonal weather
            day_of_year = current_date.timetuple().tm_yday
            season_factor = np.sin(2 * np.pi * day_of_year / 365)

            # Base values with seasonal variation
            base_temp = 15 + 10 * season_factor  # 5-25°C range
            base_precip = max(0, 2 + 3 * np.sin(2 * np.pi * day_of_year / 30))  # Monthly cycle

            # Add randomness
            temp_mean = base_temp + np.random.normal(0, 3)
            temp_min = temp_mean - 5 + np.random.normal(0, 2)
            temp_max = temp_mean + 5 + np.random.normal(0, 2)

            # Precipitation: sometimes heavy rain
            if np.random.random() < 0.2:  # 20% chance of rain
                precip = np.random.exponential(5)  # Exponential distribution
                if precip > 50:  # Cap extreme values
                    precip = 50
            else:
                precip = 0

            # ET0 based on temperature and season
            et0 = max(0.5, 3 + 2 * season_factor + np.random.normal(0, 1))

            # Solar radiation
            solar = max(5, 15 + 5 * season_factor + np.random.normal(0, 3))

            # Humidity inversely related to temperature
            humidity = max(30, min(90, 70 - 0.5 * (temp_mean - 15) + np.random.normal(0, 10)))

            # Wind speed
            wind = max(0.5, 2 + np.random.exponential(1))

            # Calculate VPD
            svp = 0.6108 * np.exp(17.27 * temp_mean / (temp_mean + 237.3))
            avp = svp * humidity / 100
            vpd = svp - avp

            # Create record
            record_data = {
                "date": current_date,
                "site_id": request.site_id,
                "precipitation_mm": round(precip, 1),
                "et0_mm": round(et0, 1),
                "temperature_mean_c": round(temp_mean, 1),
                "temperature_min_c": round(temp_min, 1),
                "temperature_max_c": round(temp_max, 1),
                "solar_radiation_mj_m2": round(solar, 1),
                "relative_humidity_mean": round(humidity, 1),
                "wind_speed_mean_m_s": round(wind, 1),
                "vapor_pressure_deficit_kpa": round(vpd, 3),
                "source": "mock",
                "is_forecast": False,
                "quality_score": 1.0
            }

            try:
                weather_record = DailyWeather(**record_data)
                weather_records.append(weather_record)
            except Exception as e:
                self.logger.warning(f"Failed to create mock record: {e}")

            current_date += timedelta(days=1)

        return weather_records

    def _initialize_seasonal_patterns(self) -> Dict:
        """Initialize realistic seasonal patterns"""
        return {
            "winter": {"temp_mean": 10, "precip_freq": 0.3},
            "spring": {"temp_mean": 15, "precip_freq": 0.2},
            "summer": {"temp_mean": 25, "precip_freq": 0.1},
            "fall": {"temp_mean": 18, "precip_freq": 0.25}
        }

    def _get_max_date_range_days(self) -> int:
        """Mock source can generate any date range"""
        return 365 * 100  # 100 years