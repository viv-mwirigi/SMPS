"""
Google Earth Engine Satellite Data Source

Provides satellite-derived vegetation indices (NDVI, LAI) using Google Earth Engine.
"""

import ee
from typing import List, NamedTuple, Tuple
from datetime import datetime
import pandas as pd
import subprocess
import sys

from smps.data.sources.base import DataSource, DataFetchRequest, DataFetchResult
from smps.data.contracts import RemoteSensingData
from smps.core.exceptions import DataSourceError
from smps.core.types import SiteID


class SatelliteObservation(NamedTuple):
    """Satellite observation data point with spectral bands."""
    date: datetime
    blue: float
    green: float
    red: float
    nir: float
    swir1: float
    swir2: float
    ndvi: float
    evi: float
    savi: float
    arvi: float
    gndvi: float
    cvi: float


def setup_gee_authentication():
    """
    Set up Google Earth Engine authentication.

    This function runs the earthengine authenticate command to set up
    credentials for accessing Google Earth Engine.
    """
    try:
        print("Setting up Google Earth Engine authentication...")
        print("This will open a browser window for authentication.")
        print("If no browser opens, visit the URL shown below.")

        result = subprocess.run(
            [sys.executable, "-m", "earthengine", "authenticate"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("Google Earth Engine authentication successful!")
            return True
        else:
            print(f"Authentication failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"Error during authentication: {e}")
        return False


class GoogleEarthEngineSatelliteSource(DataSource):
    """
    Satellite data source using Google Earth Engine.

    Provides NDVI and LAI data from various satellite datasets.
    """

    def __init__(self, project: str = None, cache_dir=None):
        """Initialize GEE API.

        Args:
            project: Google Cloud project ID with Earth Engine enabled.
                    If None, will try to load from .env file or GOOGLE_CLOUD_PROJECT env var.
        """
        super().__init__("gee_satellite", cache_dir)
        import os
        from pathlib import Path

        # Try to load from .env file if dotenv is available
        try:
            from dotenv import load_dotenv
            # Look for .env in project root
            env_path = Path(__file__).parents[4] / '.env'
            if env_path.exists():
                load_dotenv(env_path)
        except ImportError:
            pass

        try:
            import ee
            project_id = project or os.environ.get('GOOGLE_CLOUD_PROJECT')
            if project_id:
                ee.Initialize(project=project_id)
            else:
                ee.Initialize()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Earth Engine: {e}. "
                               "Make sure you have authenticated with 'earthengine authenticate' "
                               "and have a Google Cloud project with Earth Engine enabled.")

    def fetch(self, request: DataFetchRequest) -> DataFetchResult:
        """
        Fetch remote sensing data for the given request.
        """
        try:
            # Get site coordinates
            lat, lon = self._get_site_coordinates(request.site_id)

            # Fetch comprehensive vegetation data
            vegetation_data = self.fetch_vegetation_data(lat, lon,
                                                         request.start_date.isoformat(),
                                                         request.end_date.isoformat())

            # Convert to RemoteSensingData objects
            remote_sensing_data = []
            for obs in vegetation_data:
                remote_sensing_data.append(RemoteSensingData(
                    date=obs.date.date(),
                    site_id=request.site_id,
                    ndvi=obs.ndvi,
                    evi=obs.evi,
                    savi=obs.savi,
                    arvi=obs.arvi,
                    gndvi=obs.gndvi,
                    cvi=obs.cvi,
                    blue=obs.blue,
                    green=obs.green,
                    red=obs.red,
                    nir=obs.nir,
                    swir1=obs.swir1,
                    swir2=obs.swir2,
                    source="GEE_Sentinel2_Landsat8"
                ))

            return DataFetchResult(
                data=remote_sensing_data,
                metadata={"source": "Google Earth Engine",
                          "product": "Sentinel-2/Landsat-8 Vegetation Indices"},
                quality_score=0.9
            )

        except Exception as e:
            self.logger.error(f"GEE fetch failed: {e}")
            return DataFetchResult(
                data=None,
                metadata={"error": str(e)},
                quality_score=0.0,
                errors=[str(e)]
            )

    def _get_site_coordinates(self, site_id: SiteID) -> Tuple[float, float]:
        """Get latitude and longitude for a site ID."""
        # For now, use a simple mapping. In production, this would query a database
        # or use a site registry
        site_coords = {
            "test_site_001": (35.222866, 9.090245),  # Test location in Tunisia
            # Add more sites as needed
        }

        if site_id in site_coords:
            return site_coords[site_id]
        else:
            # Default to Nairobi, Kenya for unknown sites
            return (-1.2921, 36.8219)

    def fetch_vegetation_data(self, lat: float, lon: float, start_date: str, end_date: str) -> List[SatelliteObservation]:
        """
        Fetch comprehensive vegetation data including multiple spectral bands.

        Fetches bands needed for computing NDVI, EVI, SAVI, ARVI, GNDVI, CVI, etc.

        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of dictionaries with spectral bands and vegetation indices
        """
        try:
            # Use Sentinel-2 for higher resolution and more bands
            # Fallback to Landsat 8 if Sentinel-2 not available
            try:
                # Sentinel-2 MSI: Surface Reflectance
                collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

                # Filter for less cloudy images
                filtered = collection.filterDate(start_date, end_date)\
                    .filterBounds(ee.Geometry.Point([lon, lat]))\
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
                    .sort('CLOUDY_PIXEL_PERCENTAGE')

                # Select relevant bands
                # Blue, Green, Red, NIR, SWIR1, SWIR2
                bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']

                def extract_bands(image):
                    date = image.date().format('YYYY-MM-dd')

                    # Extract band values
                    values = image.select(bands).reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=ee.Geometry.Point([lon, lat]),
                        scale=10  # 10m resolution for Sentinel-2
                    )

                    # Calculate vegetation indices
                    blue = image.select('B2').divide(
                        10000)  # Convert to reflectance
                    green = image.select('B3').divide(10000)
                    red = image.select('B4').divide(10000)
                    nir = image.select('B8').divide(10000)
                    swir1 = image.select('B11').divide(10000)
                    swir2 = image.select('B12').divide(10000)

                    # NDVI
                    ndvi = nir.subtract(red).divide(nir.add(red))

                    # EVI (Enhanced Vegetation Index)
                    evi = nir.subtract(red).multiply(2.5).divide(
                        nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))

                    # SAVI (Soil-Adjusted Vegetation Index)
                    savi = nir.subtract(red).multiply(
                        1.5).divide(nir.add(red).add(0.5))

                    # ARVI (Atmospherically Resistant Vegetation Index)
                    rb = red.subtract(blue.multiply(0.1)).add(
                        red.multiply(0.9))  # gamma = 1
                    arvi = nir.subtract(rb).divide(nir.add(rb))

                    # GNDVI (Green NDVI)
                    gndvi = nir.subtract(green).divide(nir.add(green))

                    # CVI (Chlorophyll Vegetation Index)
                    cvi = nir.multiply(green).divide(red.pow(2))

                    # Get final values
                    final_values = values.combine({
                        'ndvi': ndvi.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([lon, lat]), 10).get('B8'),
                        'evi': evi.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([lon, lat]), 10).get('B8'),
                        'savi': savi.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([lon, lat]), 10).get('B8'),
                        'arvi': arvi.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([lon, lat]), 10).get('B8'),
                        'gndvi': gndvi.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([lon, lat]), 10).get('B8'),
                        'cvi': cvi.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([lon, lat]), 10).get('B8')
                    })

                    return ee.Feature(None, final_values.set('date', date))

                # Map the extractor over the filtered Sentinel-2 collection and fetch results
                features = filtered.map(extract_bands).getInfo()

            except Exception as e:
                self.logger.warning(
                    f"Sentinel-2 failed, falling back to Landsat 8: {e}")
                # Fallback to Landsat 8
                collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')

                filtered = collection.filterDate(start_date, end_date)\
                    .filterBounds(ee.Geometry.Point([lon, lat]))\
                    .filter(ee.Filter.lt('CLOUD_COVER', 20))\
                    .sort('CLOUD_COVER')

                # Landsat 8 bands: B2(Blue), B3(Green), B4(Red), B5(NIR), B6(SWIR1), B7(SWIR2)
                bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']

                def extract_bands_landsat(image):
                    date = image.date().format('YYYY-MM-dd')

                    # Extract and scale reflectance values (multiply by 0.0000275, add -0.2)
                    blue = image.select('SR_B2').multiply(0.0000275).add(-0.2)
                    green = image.select('SR_B3').multiply(0.0000275).add(-0.2)
                    red = image.select('SR_B4').multiply(0.0000275).add(-0.2)
                    nir = image.select('SR_B5').multiply(0.0000275).add(-0.2)
                    swir1 = image.select('SR_B6').multiply(0.0000275).add(-0.2)
                    swir2 = image.select('SR_B7').multiply(0.0000275).add(-0.2)

                    # Calculate vegetation indices
                    ndvi = nir.subtract(red).divide(nir.add(red))
                    evi = nir.subtract(red).multiply(2.5).divide(
                        nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
                    savi = nir.subtract(red).multiply(
                        1.5).divide(nir.add(red).add(0.5))
                    rb = red.subtract(blue.multiply(
                        0.1)).add(red.multiply(0.9))
                    arvi = nir.subtract(rb).divide(nir.add(rb))
                    gndvi = nir.subtract(green).divide(nir.add(green))
                    cvi = nir.multiply(green).divide(red.pow(2))

                    # Extract values
                    values = image.select(bands).reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=ee.Geometry.Point([lon, lat]),
                        scale=30  # 30m resolution for Landsat
                    )

                    final_values = {
                        'blue': blue.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([lon, lat]), 30).get('SR_B2'),
                        'green': green.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([lon, lat]), 30).get('SR_B3'),
                        'red': red.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([lon, lat]), 30).get('SR_B4'),
                        'nir': nir.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([lon, lat]), 30).get('SR_B5'),
                        'swir1': swir1.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([lon, lat]), 30).get('SR_B6'),
                        'swir2': swir2.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([lon, lat]), 30).get('SR_B7'),
                        'ndvi': ndvi.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([lon, lat]), 30).get('SR_B5'),
                        'evi': evi.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([lon, lat]), 30).get('SR_B5'),
                        'savi': savi.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([lon, lat]), 30).get('SR_B5'),
                        'arvi': arvi.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([lon, lat]), 30).get('SR_B5'),
                        'gndvi': gndvi.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([lon, lat]), 30).get('SR_B5'),
                        'cvi': cvi.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([lon, lat]), 30).get('SR_B5')
                    }

                    return ee.Feature(None, ee.Dictionary(final_values).set('date', date))

                features = filtered.map(extract_bands_landsat).getInfo()

            # Process results
            vegetation_data = []
            for feature in features['features']:
                props = feature['properties']
                if props.get('ndvi') is not None:  # At least NDVI should be available
                    data_point = SatelliteObservation(
                        date=datetime.strptime(props['date'], '%Y-%m-%d'),
                        blue=float(props.get('blue', 0)),
                        green=float(props.get('green', 0)),
                        red=float(props.get('red', 0)),
                        nir=float(props.get('nir', 0)),
                        swir1=float(props.get('swir1', 0)),
                        swir2=float(props.get('swir2', 0)),
                        ndvi=float(props['ndvi']),
                        evi=float(props.get('evi', 0)),
                        savi=float(props.get('savi', 0)),
                        arvi=float(props.get('arvi', 0)),
                        gndvi=float(props.get('gndvi', 0)),
                        cvi=float(props.get('cvi', 0))
                    )
                    vegetation_data.append(data_point)

            return vegetation_data

        except Exception as e:
            self.logger.error(f"Failed to fetch vegetation data: {e}")
            return []

    def fetch_lai(self, lat: float, lon: float, start_date: str, end_date: str) -> List[SatelliteObservation]:
        """
        Fetch LAI data for a location and time period.

        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of SatelliteObservation objects with LAI values
        """
        try:
            # Use MODIS LAI (MCD15A3H) - 4-day composite, 500m resolution
            collection = ee.ImageCollection('MODIS/061/MCD15A3H')

            # Filter by date and location
            point = ee.Geometry.Point([lon, lat])
            filtered = collection.filterDate(
                start_date, end_date).filterBounds(point)

            # Get LAI band (scaled by 0.1)
            def extract_lai(image):
                lai = image.select('Lai').multiply(0.1)
                date = image.date().format('YYYY-MM-dd')
                value = lai.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=point,
                    scale=500
                ).get('Lai')
                return ee.Feature(None, {'date': date, 'lai': value})

            features = filtered.map(extract_lai).getInfo()

            observations = []
            for feature in features['features']:
                props = feature['properties']
                if 'lai' in props and props['lai'] is not None:
                    observations.append(SatelliteObservation(
                        date=datetime.strptime(props['date'], '%Y-%m-%d'),
                        value=float(props['lai'])
                    ))

            return observations

        except Exception as e:
            raise RuntimeError(f"Failed to fetch LAI data: {e}")

    def fetch_elevation(self, lat: float, lon: float) -> float:
        """
        Fetch elevation data for a location using SRTM DEM.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Elevation in meters
        """
        try:
            # Use SRTM 30m Digital Elevation Model
            collection = ee.Image('USGS/SRTMGL1_003')

            # Create point
            point = ee.Geometry.Point([lon, lat])

            # Get elevation value
            elevation = collection.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=30
            ).get('elevation')

            # Get the value
            elevation_value = elevation.getInfo()

            if elevation_value is not None:
                return float(elevation_value)
            else:
                # Fallback to default elevation
                return 200.0

        except Exception as e:
            self.logger.warning(
                f"Failed to fetch elevation data: {e}. Using default.")
            return 200.0

    def fetch_slope(self, lat: float, lon: float) -> float:
        """
        Fetch slope data for a location using SRTM DEM.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Slope in percent
        """
        try:
            # Use SRTM 30m Digital Elevation Model
            dem = ee.Image('USGS/SRTMGL1_003')

            # Calculate slope
            slope = ee.Terrain.slope(dem)

            # Create point
            point = ee.Geometry.Point([lon, lat])

            # Get slope value
            slope_value = slope.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=30
            ).get('slope')

            # Get the value
            slope_degrees = slope_value.getInfo()

            if slope_degrees is not None:
                # Convert degrees to percent (rise/run * 100)
                slope_percent = float(slope_degrees) * \
                    100.0 / 45.0  # Approximation
                return slope_percent
            else:
                # Fallback to default slope
                return 5.0

        except Exception as e:
            self.logger.warning(
                f"Failed to fetch slope data: {e}. Using default.")
            return 5.0

    def fetch_land_use(self, lat: float, lon: float) -> int:
        """
        Fetch land use/land cover data for a location using ESA WorldCover.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Land use code (ESA WorldCover classification)
        """
        try:
            # Use ESA WorldCover 2020
            collection = ee.ImageCollection('ESA/WorldCover/v100').first()

            # Create point
            point = ee.Geometry.Point([lon, lat])

            # Get land cover value
            landcover = collection.reduceRegion(
                reducer=ee.Reducer.mode(),
                geometry=point,
                scale=10
            ).get('Map')

            # Get the value
            landcover_value = landcover.getInfo()

            if landcover_value is not None:
                return int(landcover_value)
            else:
                # Fallback to default (savanna/grassland)
                return 40

        except Exception as e:
            self.logger.warning(
                f"Failed to fetch land use data: {e}. Using default.")
            return 40
