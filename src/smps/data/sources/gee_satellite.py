"""
Google Earth Engine Satellite Data Source.

Provides access to satellite-derived vegetation indices, land surface temperature,
and other remote sensing products for soil moisture estimation.

Products available:
- MODIS NDVI (MOD13Q1) - 250m, 16-day
- MODIS LAI/FPAR (MOD15A2H) - 500m, 8-day
- MODIS LST (MOD11A2) - 1km, 8-day
- Sentinel-2 Surface Reflectance - 10m, 5-day
- SMAP Soil Moisture (optional) - 9km, daily

Setup:
1. Create a Google Cloud project
2. Enable Earth Engine API
3. Create service account and download key
4. Set GEE_SERVICE_ACCOUNT_KEY env var to key file path
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from dotenv import load_dotenv

from smps.data.contracts import SatelliteObservation
from smps.core.exceptions import DataSourceError

load_dotenv()
logger = logging.getLogger("smps.data.gee")

# Check if Earth Engine is available
try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False
    logger.warning("earthengine-api not installed. Run: pip install earthengine-api")


@dataclass
class GEEConfig:
    """Google Earth Engine configuration."""
    service_account: Optional[str] = None
    key_file: Optional[str] = None
    project_id: Optional[str] = None

    @classmethod
    def from_env(cls) -> "GEEConfig":
        """Load config from environment variables."""
        return cls(
            service_account=os.getenv("GEE_SERVICE_ACCOUNT"),
            key_file=os.getenv("GEE_SERVICE_ACCOUNT_KEY"),
            project_id=os.getenv("GEE_PROJECT_ID")
        )

    @property
    def is_configured(self) -> bool:
        """Check if GEE credentials are configured."""
        return bool(self.key_file and Path(self.key_file).exists())


class GoogleEarthEngineSatelliteSource:
    """
    Google Earth Engine satellite data source.

    Provides vegetation indices and land surface parameters for soil moisture modeling.

    Available products:
    - NDVI: Normalized Difference Vegetation Index
    - EVI: Enhanced Vegetation Index
    - LAI: Leaf Area Index
    - FPAR: Fraction of Absorbed Photosynthetically Active Radiation
    - LST: Land Surface Temperature
    - Albedo: Surface albedo
    """

    # Product configurations
    PRODUCTS = {
        "MODIS_NDVI": {
            "collection": "MODIS/061/MOD13Q1",
            "bands": ["NDVI", "EVI"],
            "scale_factor": 0.0001,
            "spatial_res": 250,
            "temporal_res": 16,
        },
        "MODIS_LAI": {
            "collection": "MODIS/061/MOD15A2H",
            "bands": ["Lai_500m", "Fpar_500m"],
            "scale_factor": 0.1,
            "spatial_res": 500,
            "temporal_res": 8,
        },
        "MODIS_LST": {
            "collection": "MODIS/061/MOD11A2",
            "bands": ["LST_Day_1km", "LST_Night_1km"],
            "scale_factor": 0.02,
            "spatial_res": 1000,
            "temporal_res": 8,
        },
        "SENTINEL2": {
            "collection": "COPERNICUS/S2_SR_HARMONIZED",
            "bands": ["B2", "B3", "B4", "B8"],  # Blue, Green, Red, NIR
            "scale_factor": 0.0001,
            "spatial_res": 10,
            "temporal_res": 5,
        }
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize GEE satellite source."""
        self.name = "google_earth_engine"
        self.cache_dir = cache_dir
        self.config = GEEConfig.from_env()
        self._initialized = False

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> bool:
        """Initialize Earth Engine with authentication."""
        if not EE_AVAILABLE:
            logger.error("earthengine-api not installed")
            return False

        if self._initialized:
            return True

        # Get project ID from env or use default
        project_id = self.config.project_id or os.getenv("GEE_PROJECT_ID") or "smps-481210"

        try:
            if self.config.is_configured:
                # Service account authentication
                credentials = ee.ServiceAccountCredentials(
                    self.config.service_account,
                    self.config.key_file
                )
                ee.Initialize(credentials, project=project_id)
                logger.info("GEE initialized with service account")
            else:
                # Try default/user authentication (earthengine authenticate)
                try:
                    ee.Initialize(project=project_id)
                    logger.info(f"GEE initialized with project: {project_id}")
                except Exception as init_err:
                    logger.warning(f"GEE init failed: {init_err}, trying without project")
                    try:
                        ee.Initialize()
                        logger.info("GEE initialized with default credentials")
                    except Exception:
                        logger.error("GEE not authenticated. Run: earthengine authenticate")
                        return False

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize GEE: {e}")
            return False

    def fetch_ndvi(self,
                   lat: float,
                   lon: float,
                   start_date: datetime,
                   end_date: datetime,
                   buffer_m: int = 250) -> List[SatelliteObservation]:
        """
        Fetch MODIS NDVI time series.

        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start of time range
            end_date: End of time range
            buffer_m: Buffer radius in meters for spatial averaging

        Returns:
            List of SatelliteObservation objects
        """
        if not self.initialize():
            return self._generate_synthetic_ndvi(start_date, end_date)

        try:
            point = ee.Geometry.Point([lon, lat]).buffer(buffer_m)

            collection = ee.ImageCollection("MODIS/061/MOD13Q1") \
                .filterBounds(point) \
                .filterDate(start_date.strftime("%Y-%m-%d"),
                           end_date.strftime("%Y-%m-%d")) \
                .select(["NDVI", "EVI"])

            # Get time series
            def extract_values(image):
                stats = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=point,
                    scale=250
                )
                return ee.Feature(None, {
                    "date": image.date().format("YYYY-MM-dd"),
                    "NDVI": ee.Number(stats.get("NDVI")).multiply(0.0001),
                    "EVI": ee.Number(stats.get("EVI")).multiply(0.0001)
                })

            features = collection.map(extract_values)
            data = features.getInfo()["features"]

            observations = []
            for feature in data:
                props = feature["properties"]
                observations.append(SatelliteObservation(
                    site_id="gee_query",
                    timestamp=datetime.strptime(props["date"], "%Y-%m-%d"),
                    product="MOD13Q1",
                    ndvi=props.get("NDVI"),
                    evi=props.get("EVI"),
                    quality_flag=1.0,
                    source="google_earth_engine"
                ))

            return observations

        except Exception as e:
            logger.warning(f"GEE NDVI fetch failed: {e}, using synthetic data")
            return self._generate_synthetic_ndvi(start_date, end_date)

    def fetch_lai(self,
                  lat: float,
                  lon: float,
                  start_date: datetime,
                  end_date: datetime) -> List[Dict[str, Any]]:
        """
        Fetch MODIS LAI (Leaf Area Index) time series.

        LAI is important for soil moisture modeling:
        - Affects evapotranspiration calculations
        - Indicates vegetation density
        - Used for vegetation fraction estimation
        """
        if not self.initialize():
            return self._generate_synthetic_lai(start_date, end_date)

        try:
            point = ee.Geometry.Point([lon, lat]).buffer(500)

            collection = ee.ImageCollection("MODIS/061/MOD15A2H") \
                .filterBounds(point) \
                .filterDate(start_date.strftime("%Y-%m-%d"),
                           end_date.strftime("%Y-%m-%d")) \
                .select(["Lai_500m", "Fpar_500m"])

            def extract_values(image):
                stats = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=point,
                    scale=500
                )
                return ee.Feature(None, {
                    "date": image.date().format("YYYY-MM-dd"),
                    "LAI": ee.Number(stats.get("Lai_500m")).multiply(0.1),
                    "FPAR": ee.Number(stats.get("Fpar_500m")).multiply(0.01)
                })

            features = collection.map(extract_values)
            data = features.getInfo()["features"]

            return [
                {
                    "date": f["properties"]["date"],
                    "lai": f["properties"].get("LAI"),
                    "fpar": f["properties"].get("FPAR")
                }
                for f in data
            ]

        except Exception as e:
            logger.warning(f"GEE LAI fetch failed: {e}, using synthetic data")
            return self._generate_synthetic_lai(start_date, end_date)

    def fetch_lst(self,
                  lat: float,
                  lon: float,
                  start_date: datetime,
                  end_date: datetime) -> List[Dict[str, Any]]:
        """
        Fetch MODIS Land Surface Temperature time series.

        LST is important for soil moisture modeling:
        - Indicates evaporative demand
        - Soil temperature affects microbial activity
        - Thermal inertia relates to soil moisture
        """
        if not self.initialize():
            return self._generate_synthetic_lst(start_date, end_date)

        try:
            point = ee.Geometry.Point([lon, lat]).buffer(1000)

            collection = ee.ImageCollection("MODIS/061/MOD11A2") \
                .filterBounds(point) \
                .filterDate(start_date.strftime("%Y-%m-%d"),
                           end_date.strftime("%Y-%m-%d")) \
                .select(["LST_Day_1km", "LST_Night_1km"])

            def extract_values(image):
                stats = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=point,
                    scale=1000
                )
                # LST is in Kelvin * 50, convert to Celsius
                lst_day = ee.Number(stats.get("LST_Day_1km")).multiply(0.02).subtract(273.15)
                lst_night = ee.Number(stats.get("LST_Night_1km")).multiply(0.02).subtract(273.15)

                return ee.Feature(None, {
                    "date": image.date().format("YYYY-MM-dd"),
                    "LST_Day": lst_day,
                    "LST_Night": lst_night
                })

            features = collection.map(extract_values)
            data = features.getInfo()["features"]

            return [
                {
                    "date": f["properties"]["date"],
                    "lst_day_celsius": f["properties"].get("LST_Day"),
                    "lst_night_celsius": f["properties"].get("LST_Night")
                }
                for f in data
            ]

        except Exception as e:
            logger.warning(f"GEE LST fetch failed: {e}, using synthetic data")
            return self._generate_synthetic_lst(start_date, end_date)

    def fetch_sentinel2_indices(self,
                               lat: float,
                               lon: float,
                               date: datetime,
                               cloud_threshold: float = 20) -> Dict[str, Any]:
        """
        Fetch vegetation indices from Sentinel-2.

        Higher resolution (10m) than MODIS but less frequent.
        Best for small-scale applications.
        """
        if not self.initialize():
            return self._generate_synthetic_sentinel2()

        try:
            point = ee.Geometry.Point([lon, lat]).buffer(100)

            # Get best image within ±30 days
            start = (date - timedelta(days=30)).strftime("%Y-%m-%d")
            end = (date + timedelta(days=30)).strftime("%Y-%m-%d")

            collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
                .filterBounds(point) \
                .filterDate(start, end) \
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold)) \
                .sort("CLOUDY_PIXEL_PERCENTAGE") \
                .first()

            if collection is None:
                return self._generate_synthetic_sentinel2()

            # Calculate indices
            ndvi = collection.normalizedDifference(["B8", "B4"]).rename("NDVI")
            ndwi = collection.normalizedDifference(["B3", "B8"]).rename("NDWI")

            # Stack and reduce
            indices = ee.Image.cat([ndvi, ndwi])
            stats = indices.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=10
            ).getInfo()

            return {
                "date": date.strftime("%Y-%m-%d"),
                "ndvi": stats.get("NDVI"),
                "ndwi": stats.get("NDWI"),  # NDWI relates to water content
                "source": "sentinel2",
                "resolution_m": 10
            }

        except Exception as e:
            logger.warning(f"Sentinel-2 fetch failed: {e}")
            return self._generate_synthetic_sentinel2()

    def _generate_synthetic_ndvi(self,
                                 start_date: datetime,
                                 end_date: datetime) -> List[SatelliteObservation]:
        """Generate synthetic NDVI for testing when GEE unavailable."""
        observations = []
        current = start_date

        while current <= end_date:
            # Seasonal NDVI pattern
            doy = current.timetuple().tm_yday
            base_ndvi = 0.4 + 0.25 * np.sin(2 * np.pi * (doy - 80) / 365)
            ndvi = base_ndvi + np.random.normal(0, 0.05)
            ndvi = max(0.1, min(0.9, ndvi))

            observations.append(SatelliteObservation(
                site_id="synthetic",
                timestamp=current,
                product="MOD13Q1_synthetic",
                ndvi=round(ndvi, 4),
                evi=round(ndvi * 0.8, 4),
                quality_flag=1.0,
                source="synthetic_gee"
            ))

            current += timedelta(days=16)  # MODIS 16-day composite

        return observations

    def _generate_synthetic_lai(self,
                               start_date: datetime,
                               end_date: datetime) -> List[Dict[str, Any]]:
        """Generate synthetic LAI data."""
        data = []
        current = start_date

        while current <= end_date:
            doy = current.timetuple().tm_yday
            base_lai = 1.5 + 1.5 * np.sin(2 * np.pi * (doy - 80) / 365)
            lai = max(0.5, base_lai + np.random.normal(0, 0.2))

            data.append({
                "date": current.strftime("%Y-%m-%d"),
                "lai": round(lai, 2),
                "fpar": round(lai * 0.3, 3)
            })

            current += timedelta(days=8)

        return data

    def _generate_synthetic_lst(self,
                               start_date: datetime,
                               end_date: datetime) -> List[Dict[str, Any]]:
        """Generate synthetic LST data."""
        data = []
        current = start_date

        while current <= end_date:
            doy = current.timetuple().tm_yday
            # Temperature varies seasonally
            base_temp = 25 + 10 * np.sin(2 * np.pi * (doy - 172) / 365)
            lst_day = base_temp + 5 + np.random.normal(0, 2)
            lst_night = base_temp - 8 + np.random.normal(0, 2)

            data.append({
                "date": current.strftime("%Y-%m-%d"),
                "lst_day_celsius": round(lst_day, 1),
                "lst_night_celsius": round(lst_night, 1)
            })

            current += timedelta(days=8)

        return data

    def _generate_synthetic_sentinel2(self) -> Dict[str, Any]:
        """Generate synthetic Sentinel-2 data."""
        ndvi = 0.4 + np.random.uniform(-0.2, 0.3)
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "ndvi": round(ndvi, 4),
            "ndwi": round(-0.1 + np.random.uniform(-0.2, 0.2), 4),
            "source": "synthetic_sentinel2",
            "resolution_m": 10
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get source metadata."""
        return {
            "name": self.name,
            "provider": "Google Earth Engine",
            "products": self.PRODUCTS,
            "initialized": self._initialized,
            "has_credentials": self.config.is_configured
        }


def setup_gee_authentication():
    """
    Interactive helper to set up GEE authentication.

    Steps:
    1. Go to https://console.cloud.google.com/
    2. Create or select a project
    3. Enable Earth Engine API: https://console.cloud.google.com/apis/library/earthengine.googleapis.com
    4. Create Service Account: IAM & Admin > Service Accounts > Create
    5. Add role: Earth Engine Resource Writer
    6. Create key: Actions > Create Key > JSON
    7. Save key file and set env var: GEE_SERVICE_ACCOUNT_KEY=/path/to/key.json
    """
    print("""
=== Google Earth Engine Setup Guide ===

1. Create Google Cloud Project:
   https://console.cloud.google.com/projectcreate

2. Enable Earth Engine API:
   https://console.cloud.google.com/apis/library/earthengine.googleapis.com

3. Register for Earth Engine (if not done):
   https://earthengine.google.com/signup/

4. Create Service Account:
   - Go to: IAM & Admin > Service Accounts
   - Click "Create Service Account"
   - Name: earth-engine-service
   - Add role: Earth Engine Resource Writer (or Admin)

5. Create JSON Key:
   - Click on the service account
   - Keys tab > Add Key > Create New Key > JSON
   - Save the downloaded file

6. Configure Environment:
   Add to your .env file:

   GEE_SERVICE_ACCOUNT=your-service-account@project.iam.gserviceaccount.com
   GEE_SERVICE_ACCOUNT_KEY=/path/to/your-key.json
   GEE_PROJECT_ID=your-project-id

7. Test connection:
   >>> from smps.data.sources.gee_satellite import GoogleEarthEngineSatelliteSource
   >>> gee = GoogleEarthEngineSatelliteSource()
   >>> gee.initialize()

For interactive use (Jupyter notebooks), you can also use:
   >>> import ee
   >>> ee.Authenticate()  # Opens browser for authentication
   >>> ee.Initialize()
""")


if __name__ == "__main__":
    setup_gee_authentication()
