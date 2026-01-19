"""
Tests for SpaceIoTBox API data source integration.
"""
import pytest
from datetime import date, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from smps.data.sources.spaceiotbox import (
    SpaceIoTBoxClient,
    SpaceIoTBoxConfig,
    SpaceIoTBoxWeatherSource,
    SpaceIoTBoxCopernicusSource,
    SpaceIoTBoxAgroSource,
    SpaceIoTBoxDatasetsSource,
    SpaceIoTBoxUnifiedSource,
    get_spaceiotbox_weather,
    get_spaceiotbox_satellite,
)
from smps.data.sources.base import DataFetchRequest
from smps.data.contracts import DailyWeather, RemoteSensingData
from smps.core.exceptions import DataSourceError


class TestSpaceIoTBoxConfig:
    """Tests for SpaceIoTBox configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SpaceIoTBoxConfig()

        assert config.base_url == "http://127.0.0.1:8000"
        assert config.api_version == "v1"
        assert config.timeout == 30
        assert config.max_retries == 3

    def test_api_base_url(self):
        """Test API base URL construction."""
        config = SpaceIoTBoxConfig(
            base_url="http://example.com", api_version="v2")

        assert config.api_base_url == "http://example.com/api/v2"

    def test_custom_config(self):
        """Test custom configuration."""
        config = SpaceIoTBoxConfig(
            base_url="http://custom.api",
            timeout=60,
            max_retries=5,
        )

        assert config.base_url == "http://custom.api"
        assert config.timeout == 60
        assert config.max_retries == 5


class TestSpaceIoTBoxClient:
    """Tests for SpaceIoTBox HTTP client."""

    def test_client_initialization(self):
        """Test client initialization with credentials."""
        client = SpaceIoTBoxClient(
            username="test_user",
            password="test_pass",
        )

        assert client.username == "test_user"
        assert client.password == "test_pass"
        assert client.session.auth == ("test_user", "test_pass")

    def test_client_without_credentials(self):
        """Test client initialization without credentials (uses env vars)."""
        with patch.dict("os.environ", {}, clear=True):
            client = SpaceIoTBoxClient()
            assert client.username is None
            assert client.password is None

    @patch("smps.data.sources.spaceiotbox.requests.Session")
    def test_successful_get_request(self, mock_session_class):
        """Test successful GET request."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_session.request.return_value = mock_response

        client = SpaceIoTBoxClient(username="user", password="pass")
        result = client.get("test/endpoint")

        assert result == {"data": "test"}

    @patch("smps.data.sources.spaceiotbox.requests.Session")
    def test_authentication_error(self, mock_session_class):
        """Test authentication failure handling."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 401
        mock_session.request.return_value = mock_response

        client = SpaceIoTBoxClient(username="user", password="wrong_pass")

        with pytest.raises(DataSourceError, match="authentication failed"):
            client.get("test/endpoint")


class TestSpaceIoTBoxWeatherSource:
    """Tests for SpaceIoTBox weather data source."""

    @pytest.fixture
    def weather_source(self):
        """Create weather source for testing."""
        source = SpaceIoTBoxWeatherSource(
            username="test_user",
            password="test_pass",
        )
        source.register_site("site1", 36.8, 10.2)
        return source

    def test_source_initialization(self, weather_source):
        """Test weather source initialization."""
        assert weather_source.name == "spaceiotbox_weather"
        assert "site1" in weather_source._site_coordinates

    def test_register_site(self, weather_source):
        """Test site registration."""
        weather_source.register_site("new_site", 1.0, 2.0)

        assert "new_site" in weather_source._site_coordinates
        assert weather_source._site_coordinates["new_site"] == (1.0, 2.0)

    def test_get_site_coordinates_registered(self, weather_source):
        """Test getting coordinates for registered site."""
        lat, lon = weather_source._get_site_coordinates("site1")

        assert lat == 36.8
        assert lon == 10.2

    def test_get_site_coordinates_from_id(self, weather_source):
        """Test parsing coordinates from site ID format."""
        lat, lon = weather_source._get_site_coordinates("5.5_10.5")

        assert lat == 5.5
        assert lon == 10.5

    def test_get_site_coordinates_unknown(self, weather_source):
        """Test error for unknown site."""
        with pytest.raises(DataSourceError, match="Unknown site_id"):
            weather_source._get_site_coordinates("unknown_site")

    @patch.object(SpaceIoTBoxClient, "get")
    def test_fetch_daily_weather(self, mock_get, weather_source):
        """Test fetching daily weather data."""
        mock_get.return_value = {
            "daily": {
                "time": ["2025-01-01", "2025-01-02"],
                "precipitation": [5.0, 0.0],
                "et0_fao_evapotranspiration": [3.5, 4.0],
                "temperature_2m_mean": [22.0, 23.0],
                "temperature_2m_min": [18.0, 19.0],
                "temperature_2m_max": [26.0, 27.0],
                "shortwave_radiation_sum": [18.0, 20.0],
                "relative_humidity_2m_mean": [65.0, 60.0],
                "wind_speed_10m_mean": [2.5, 3.0],
            }
        }

        request = DataFetchRequest(
            site_id="site1",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 2),
        )

        result = weather_source.fetch_daily_weather(request)

        assert len(result) == 2
        assert isinstance(result[0], DailyWeather)
        assert result[0].precipitation_mm == 5.0
        assert result[0].temperature_mean_c == 22.0
        assert result[0].source == "spaceiotbox"

    @patch.object(SpaceIoTBoxClient, "get")
    def test_fetch_daily_weather_list_format(self, mock_get, weather_source):
        """Test fetching daily weather with list response format."""
        mock_get.return_value = {
            "data": [
                {
                    "date": "2025-01-01",
                    "precipitation_mm": 5.0,
                    "et0_mm": 3.5,
                    "temperature_mean_c": 22.0,
                    "temperature_min_c": 18.0,
                    "temperature_max_c": 26.0,
                    "solar_radiation_mj_m2": 18.0,
                    "relative_humidity_mean": 65.0,
                    "wind_speed_mean_m_s": 2.5,
                }
            ]
        }

        request = DataFetchRequest(
            site_id="site1",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 1),
        )

        result = weather_source.fetch_daily_weather(request)

        assert len(result) == 1
        assert result[0].precipitation_mm == 5.0


class TestSpaceIoTBoxCopernicusSource:
    """Tests for SpaceIoTBox Copernicus satellite data source."""

    @pytest.fixture
    def copernicus_source(self):
        """Create Copernicus source for testing."""
        source = SpaceIoTBoxCopernicusSource(
            username="test_user",
            password="test_pass",
        )
        source.register_site("site1", 36.8, 10.2)
        return source

    def test_source_initialization(self, copernicus_source):
        """Test Copernicus source initialization."""
        assert copernicus_source.name == "spaceiotbox_copernicus"

    @patch.object(SpaceIoTBoxClient, "get")
    def test_fetch_remote_sensing(self, mock_get, copernicus_source):
        """Test fetching remote sensing data."""
        mock_get.return_value = {
            "data": [
                {
                    "date": "2025-01-01",
                    "ndvi": 0.65,
                    "evi": 0.45,
                    "lai": 2.5,
                    "cloud_cover": 10.0,
                },
                {
                    "date": "2025-01-05",
                    "ndvi": 0.70,
                    "sar_vv": -12.5,
                    "sar_vh": -18.3,
                    "cloud_cover": 5.0,
                }
            ]
        }

        request = DataFetchRequest(
            site_id="site1",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 10),
        )

        result = copernicus_source.fetch_remote_sensing(request)

        assert len(result) == 2
        assert isinstance(result[0], RemoteSensingData)
        assert result[0].ndvi == 0.65
        assert result[1].sar_vv_db == -12.5

    def test_calculate_quality_score(self, copernicus_source):
        """Test quality score calculation."""
        data = [
            RemoteSensingData(
                date=date(2025, 1, 1),
                site_id="site1",
                ndvi=0.65,
                cloud_cover_percent=10.0,
            ),
            RemoteSensingData(
                date=date(2025, 1, 2),
                site_id="site1",
                ndvi=0.70,
                cloud_cover_percent=20.0,
            ),
        ]

        score = copernicus_source._calculate_quality_score(data)

        assert 0 <= score <= 1
        assert score > 0.5  # Good data with low cloud cover

    def test_calculate_quality_score_empty(self, copernicus_source):
        """Test quality score for empty data."""
        score = copernicus_source._calculate_quality_score([])
        assert score == 0.0


class TestSpaceIoTBoxAgroSource:
    """Tests for SpaceIoTBox agricultural data source."""

    @pytest.fixture
    def agro_source(self):
        """Create Agro source for testing."""
        source = SpaceIoTBoxAgroSource(
            username="test_user",
            password="test_pass",
        )
        source.register_site("site1", 36.8, 10.2)
        return source

    def test_source_initialization(self, agro_source):
        """Test Agro source initialization."""
        assert agro_source.name == "spaceiotbox_agro"

    @patch.object(SpaceIoTBoxClient, "get")
    def test_fetch_agro_data(self, mock_get, agro_source):
        """Test fetching agricultural data."""
        mock_get.return_value = {
            "data": {
                "soil_moisture": 0.35,
                "crop_stage": "vegetative",
            }
        }

        request = DataFetchRequest(
            site_id="site1",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 10),
        )

        result = agro_source.fetch(request)

        assert result.success
        assert result.data["soil_moisture"] == 0.35

    @patch.object(SpaceIoTBoxClient, "get")
    def test_fetch_soil_moisture(self, mock_get, agro_source):
        """Test fetching soil moisture data."""
        mock_get.return_value = {
            "soil_moisture": [
                {"date": "2025-01-01", "value": 0.30, "depth_cm": 10},
                {"date": "2025-01-02", "value": 0.28, "depth_cm": 10},
            ]
        }

        result = agro_source.fetch_soil_moisture(
            "site1",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 2),
            depth_cm=10,
        )

        mock_get.assert_called_once()
        assert "soil_moisture" in result


class TestSpaceIoTBoxDatasetsSource:
    """Tests for SpaceIoTBox datasets source."""

    @pytest.fixture
    def datasets_source(self):
        """Create Datasets source for testing."""
        return SpaceIoTBoxDatasetsSource(
            username="test_user",
            password="test_pass",
        )

    def test_source_initialization(self, datasets_source):
        """Test Datasets source initialization."""
        assert datasets_source.name == "spaceiotbox_datasets"

    @patch.object(SpaceIoTBoxClient, "get")
    def test_list_datasets(self, mock_get, datasets_source):
        """Test listing available datasets."""
        mock_get.return_value = {
            "datasets": [
                {"id": "dataset1", "name": "Weather Data"},
                {"id": "dataset2", "name": "Soil Data"},
            ]
        }

        result = datasets_source.list_datasets()

        assert len(result) == 2
        assert result[0]["id"] == "dataset1"

    @patch.object(SpaceIoTBoxClient, "get")
    def test_get_dataset_info(self, mock_get, datasets_source):
        """Test getting dataset information."""
        mock_get.return_value = {
            "id": "dataset1",
            "name": "Weather Data",
            "description": "Daily weather observations",
        }

        result = datasets_source.get_dataset_info("dataset1")

        assert result["id"] == "dataset1"
        mock_get.assert_called_with("datasets/dataset1")


class TestSpaceIoTBoxUnifiedSource:
    """Tests for SpaceIoTBox unified data source."""

    @pytest.fixture
    def unified_source(self):
        """Create unified source for testing."""
        return SpaceIoTBoxUnifiedSource(
            username="test_user",
            password="test_pass",
        )

    def test_source_initialization(self, unified_source):
        """Test unified source initialization with all sub-sources."""
        assert unified_source.name == "spaceiotbox"
        assert unified_source.weather is not None
        assert unified_source.copernicus is not None
        assert unified_source.agro is not None
        assert unified_source.datasets is not None

    def test_register_site_propagates(self, unified_source):
        """Test that site registration propagates to all sub-sources."""
        unified_source.register_site("test_site", 1.0, 2.0)

        assert "test_site" in unified_source.weather._site_coordinates
        assert "test_site" in unified_source.copernicus._site_coordinates
        assert "test_site" in unified_source.agro._site_coordinates

    @patch.object(SpaceIoTBoxWeatherSource, "fetch")
    def test_fetch_weather_routing(self, mock_fetch, unified_source):
        """Test that fetch routes to weather source."""
        request = DataFetchRequest(
            site_id="site1",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 10),
            parameters={"source_type": "weather"},
        )

        unified_source.fetch(request)

        mock_fetch.assert_called_once_with(request)

    @patch.object(SpaceIoTBoxCopernicusSource, "fetch")
    def test_fetch_copernicus_routing(self, mock_fetch, unified_source):
        """Test that fetch routes to Copernicus source."""
        request = DataFetchRequest(
            site_id="site1",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 10),
            parameters={"source_type": "copernicus"},
        )

        unified_source.fetch(request)

        mock_fetch.assert_called_once_with(request)

    def test_fetch_unknown_source_type(self, unified_source):
        """Test handling unknown source type."""
        request = DataFetchRequest(
            site_id="site1",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 10),
            parameters={"source_type": "unknown"},
        )

        result = unified_source.fetch(request)

        assert not result.success
        assert "Unknown source_type" in result.errors[0]


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @patch.object(SpaceIoTBoxWeatherSource, "fetch")
    def test_get_spaceiotbox_weather(self, mock_fetch):
        """Test get_spaceiotbox_weather convenience function."""
        from smps.data.sources.base import DataFetchResult

        mock_weather = [
            DailyWeather(
                date=date(2025, 1, 1),
                site_id="site1",
                precipitation_mm=5.0,
                et0_mm=3.5,
                temperature_mean_c=22.0,
                temperature_min_c=18.0,
                temperature_max_c=26.0,
                solar_radiation_mj_m2=18.0,
                relative_humidity_mean=65.0,
                wind_speed_mean_m_s=2.5,
                source="spaceiotbox",
            )
        ]

        mock_fetch.return_value = DataFetchResult(
            data=mock_weather,
            metadata={},
            quality_score=1.0,
        )

        result = get_spaceiotbox_weather(
            site_id="site1",
            latitude=36.8,
            longitude=10.2,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 1),
        )

        assert len(result) == 1
        assert isinstance(result[0], DailyWeather)

    @patch.object(SpaceIoTBoxWeatherSource, "fetch")
    def test_get_spaceiotbox_weather_error(self, mock_fetch):
        """Test get_spaceiotbox_weather error handling."""
        from smps.data.sources.base import DataFetchResult

        mock_fetch.return_value = DataFetchResult(
            data=None,
            metadata={},
            quality_score=0.0,
            errors=["API error"],
        )

        with pytest.raises(DataSourceError, match="Failed to fetch weather"):
            get_spaceiotbox_weather(
                site_id="site1",
                latitude=36.8,
                longitude=10.2,
                start_date=date(2025, 1, 1),
                end_date=date(2025, 1, 1),
            )


class TestIntegration:
    """Integration tests (marked for skip if no API available)."""

    @pytest.mark.skip(reason="Requires running SpaceIoTBox API")
    def test_real_weather_fetch(self):
        """Test actual weather fetch from running API."""
        source = SpaceIoTBoxWeatherSource()
        source.register_site("test_site", 36.8, 10.2)

        request = DataFetchRequest(
            site_id="test_site",
            start_date=date.today() - timedelta(days=7),
            end_date=date.today(),
        )

        result = source.fetch(request)

        assert result.success
        assert len(result.data) > 0
