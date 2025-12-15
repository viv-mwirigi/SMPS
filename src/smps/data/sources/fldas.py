"""
FLDAS (Famine Early Warning Systems Network Land Data Assimilation System)
Soil Moisture Data Source

This module provides access to FLDAS soil moisture data from GeoTIFF files.
FLDAS provides monthly averaged soil moisture at 0-100cm depth.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import date, datetime
from dataclasses import dataclass

import numpy as np

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

logger = logging.getLogger(__name__)


@dataclass
class FLDASObservation:
    """Single FLDAS soil moisture observation"""
    site_id: str
    latitude: float
    longitude: float
    date: date
    soil_moisture_vwc: float  # m³/m³ (0-100cm integrated)
    depth_cm: str = "0-100"
    source: str = "FLDAS"
    quality_flag: str = "OK"


class FLDASSource:
    """
    FLDAS Soil Moisture Data Source.

    Reads monthly soil moisture data from FLDAS GeoTIFF files.
    Data represents 0-100cm depth-integrated volumetric water content.

    File naming convention:
        {region}_monthly_fldas_soilmoi00_100cm_wavg_{YYMM}/
            {region}_monthly_fldas_soilmoi00_100cm_wavg_{YYMM}.tif

    Where:
        - region: ea (East Africa), wa (West Africa), sa (Southern Africa)
        - YYMM: Year-month code (e.g., 2412 = December 2024)

    Scale factor: Values are stored as integers, divide by 10000 for m³/m³
    """

    # Regional coverage bounds
    REGIONS = {
        'ea': {
            'name': 'East Africa',
            'bounds': {'left': 21.0, 'bottom': -12.5, 'right': 52.0, 'top': 23.0}
        },
        'wa': {
            'name': 'West Africa',
            'bounds': {'left': -19.1, 'bottom': 2.0, 'right': 27.4, 'top': 21.0}
        },
        'sa': {
            'name': 'Southern Africa',
            'bounds': {'left': 4.1, 'bottom': -35.5, 'right': 52.0, 'top': 5.5}
        }
    }

    SCALE_FACTOR = 10000.0  # Divide raw values by this to get m³/m³

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize FLDAS data source.

        Args:
            data_dir: Path to directory containing FLDAS data folders.
                     Defaults to {project_root}/data/soil_moisture
        """
        if not HAS_RASTERIO:
            raise ImportError(
                "rasterio is required for FLDAS data. Install with: pip install rasterio")

        if data_dir is None:
            # Default to project data directory
            data_dir = Path(
                __file__).parent.parent.parent.parent.parent / 'data' / 'soil_moisture'

        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}")

        # Cache for loaded rasters
        self._raster_cache: Dict[str, rasterio.DatasetReader] = {}

        self.logger.info(
            f"FLDAS source initialized with data_dir: {self.data_dir}")

    def get_region_for_location(self, lat: float, lon: float) -> Optional[str]:
        """
        Determine which FLDAS region contains the given coordinates.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Region code ('ea', 'wa', 'sa') or None if outside coverage
        """
        for region_code, region_info in self.REGIONS.items():
            bounds = region_info['bounds']
            if (bounds['left'] <= lon <= bounds['right'] and
                    bounds['bottom'] <= lat <= bounds['top']):
                return region_code
        return None

    def _get_tif_path(self, region: str, year: int, month: int) -> Optional[Path]:
        """
        Get path to FLDAS GeoTIFF file for given region and date.

        Args:
            region: Region code ('ea', 'wa', 'sa')
            year: Year (e.g., 2024)
            month: Month (1-12)

        Returns:
            Path to .tif file or None if not found
        """
        # Format: YYMM
        yymm = f"{year % 100:02d}{month:02d}"
        folder_name = f"{region}_monthly_fldas_soilmoi00_100cm_wavg_{yymm}"
        tif_name = f"{folder_name}.tif"

        tif_path = self.data_dir / folder_name / tif_name

        if tif_path.exists():
            return tif_path

        self.logger.warning(f"FLDAS file not found: {tif_path}")
        return None

    def fetch_observation(
        self,
        site_id: str,
        lat: float,
        lon: float,
        year: int,
        month: int
    ) -> Optional[FLDASObservation]:
        """
        Fetch FLDAS soil moisture for a specific location and month.

        Args:
            site_id: Site identifier
            lat: Latitude
            lon: Longitude
            year: Year
            month: Month (1-12)

        Returns:
            FLDASObservation or None if data unavailable
        """
        # Determine region
        region = self.get_region_for_location(lat, lon)
        if region is None:
            self.logger.warning(
                f"Location ({lat}, {lon}) is outside FLDAS coverage")
            return None

        # Get file path
        tif_path = self._get_tif_path(region, year, month)
        if tif_path is None:
            return None

        try:
            with rasterio.open(tif_path) as src:
                # Get pixel coordinates
                row, col = src.index(lon, lat)

                # Check bounds
                if not (0 <= row < src.shape[0] and 0 <= col < src.shape[1]):
                    self.logger.warning(
                        f"Location ({lat}, {lon}) is outside raster bounds")
                    return None

                # Read value
                raw_value = src.read(1)[row, col]

                # Check for nodata
                if src.nodata is not None and raw_value == src.nodata:
                    self.logger.warning(f"NoData value at ({lat}, {lon})")
                    return None

                # Scale to m³/m³
                sm_vwc = raw_value / self.SCALE_FACTOR

                # Validate physical range
                if not (0.0 <= sm_vwc <= 1.0):
                    self.logger.warning(
                        f"Soil moisture {sm_vwc} outside physical range at ({lat}, {lon})")
                    return None

                return FLDASObservation(
                    site_id=site_id,
                    latitude=lat,
                    longitude=lon,
                    date=date(year, month, 15),  # Mid-month date
                    soil_moisture_vwc=sm_vwc,
                    depth_cm="0-100",
                    source="FLDAS",
                    quality_flag="OK"
                )

        except Exception as e:
            self.logger.error(f"Error reading FLDAS data: {e}")
            return None

    def fetch_observations_for_sites(
        self,
        sites: Dict[str, Dict],
        year: int,
        month: int
    ) -> List[FLDASObservation]:
        """
        Fetch FLDAS observations for multiple sites.

        Args:
            sites: Dictionary of site_id -> {'latitude': lat, 'longitude': lon}
            year: Year
            month: Month

        Returns:
            List of FLDASObservation objects
        """
        observations = []

        for site_id, site_info in sites.items():
            obs = self.fetch_observation(
                site_id=site_id,
                lat=site_info['latitude'],
                lon=site_info['longitude'],
                year=year,
                month=month
            )
            if obs is not None:
                observations.append(obs)
                self.logger.info(
                    f"✓ {site_id}: SM = {obs.soil_moisture_vwc:.4f} m³/m³")
            else:
                self.logger.warning(f"✗ {site_id}: No data available")

        return observations

    def list_available_data(self) -> List[Dict]:
        """
        List all available FLDAS data files.

        Returns:
            List of dicts with region, year, month info
        """
        available = []

        for folder in self.data_dir.iterdir():
            if folder.is_dir() and 'fldas' in folder.name:
                # Parse folder name
                parts = folder.name.split('_')
                if len(parts) >= 6:
                    region = parts[0]
                    yymm = parts[-1]
                    if len(yymm) == 4:
                        try:
                            year = 2000 + int(yymm[:2])
                            month = int(yymm[2:])
                            available.append({
                                'region': region,
                                'region_name': self.REGIONS.get(region, {}).get('name', 'Unknown'),
                                'year': year,
                                'month': month,
                                'path': folder
                            })
                        except ValueError:
                            pass

        return sorted(available, key=lambda x: (x['year'], x['month'], x['region']))


# Convenience function
def load_fldas_observation(lat: float, lon: float, year: int = 2024, month: int = 12) -> Optional[float]:
    """
    Quick function to load FLDAS soil moisture at a point.

    Args:
        lat: Latitude
        lon: Longitude
        year: Year (default 2024)
        month: Month (default 12)

    Returns:
        Soil moisture in m³/m³ or None if unavailable
    """
    source = FLDASSource()
    obs = source.fetch_observation("point", lat, lon, year, month)
    return obs.soil_moisture_vwc if obs else None
