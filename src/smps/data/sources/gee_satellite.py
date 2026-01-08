"""
Google Earth Engine Satellite Data Source

Provides satellite-derived vegetation indices (NDVI, LAI) using Google Earth Engine.
"""

import ee
from typing import List, NamedTuple
from datetime import datetime
import pandas as pd
import subprocess
import sys


class SatelliteObservation(NamedTuple):
    """Satellite observation data point."""
    date: datetime
    value: float


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


class GoogleEarthEngineSatelliteSource:
    """
    Satellite data source using Google Earth Engine.

    Provides NDVI and LAI data from various satellite datasets.
    """

    def __init__(self, project: str = None):
        """Initialize GEE API.

        Args:
            project: Google Cloud project ID with Earth Engine enabled.
                    If None, will try to load from .env file or GOOGLE_CLOUD_PROJECT env var.
        """
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
            project_id = project or os.environ.get('GOOGLE_CLOUD_PROJECT')
            if project_id:
                ee.Initialize(project=project_id)
            else:
                ee.Initialize()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Earth Engine: {e}. "
                               "Make sure you have authenticated with 'earthengine authenticate' "
                               "and have a Google Cloud project with Earth Engine enabled.")

    def fetch_ndvi(self, lat: float, lon: float, start_date: str, end_date: str) -> List[SatelliteObservation]:
        """
        Fetch NDVI data for a location and time period.

        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of SatelliteObservation objects with NDVI values
        """
        try:
            # Use MODIS NDVI (MOD13Q1) - 16-day composite, 250m resolution
            collection = ee.ImageCollection('MODIS/061/MOD13Q1')

            # Filter by date and location
            point = ee.Geometry.Point([lon, lat])
            filtered = collection.filterDate(
                start_date, end_date).filterBounds(point)

            # Get NDVI band (scaled by 0.0001)
            def extract_ndvi(image):
                ndvi = image.select('NDVI').multiply(0.0001)
                date = image.date().format('YYYY-MM-dd')
                value = ndvi.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=point,
                    scale=250
                ).get('NDVI')
                return ee.Feature(None, {'date': date, 'ndvi': value})

            features = filtered.map(extract_ndvi).getInfo()

            observations = []
            for feature in features['features']:
                props = feature['properties']
                if 'ndvi' in props and props['ndvi'] is not None:
                    observations.append(SatelliteObservation(
                        date=datetime.strptime(props['date'], '%Y-%m-%d'),
                        value=float(props['ndvi'])
                    ))

            return observations

        except Exception as e:
            raise RuntimeError(f"Failed to fetch NDVI data: {e}")

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
