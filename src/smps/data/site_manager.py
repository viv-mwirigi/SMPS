"""
Site Manager for Coordinate-Based Data Fetching.

Maps site_ids to coordinates and manages site metadata for both
development (ISMN) and production (IoT) modes.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SiteMetadata:
    """Metadata for a single site."""
    site_id: str
    latitude: float
    longitude: float
    elevation_m: Optional[float] = None
    soil_texture: Optional[str] = None
    sand_percent: Optional[float] = None
    clay_percent: Optional[float] = None
    organic_matter_percent: Optional[float] = None
    land_cover: Optional[str] = None

    @property
    def coordinates(self) -> Tuple[float, float]:
        """Return (lat, lon) tuple."""
        return (self.latitude, self.longitude)


class SiteManager:
    """
    Manages site metadata and coordinate mappings.

    Supports both ISMN development mode and IoT production mode.
    """

    def __init__(self, metadata_path: Optional[Path] = None):
        self.metadata_path = metadata_path or Path("data/site_metadata.csv")
        self.sites: Dict[str, SiteMetadata] = {}
        self._load_metadata()

    def _load_metadata(self):
        """Load site metadata from file or create default mappings."""
        if self.metadata_path.exists():
            df = pd.read_csv(self.metadata_path)
            for _, row in df.iterrows():
                site = SiteMetadata(
                    site_id=str(row['site_id']),
                    latitude=row['latitude'],
                    longitude=row['longitude'],
                    elevation_m=row.get('elevation_m'),
                    soil_texture=row.get('soil_texture'),
                    sand_percent=row.get('sand_percent'),
                    clay_percent=row.get('clay_percent'),
                    organic_matter_percent=row.get('organic_matter_percent'),
                    land_cover=row.get('land_cover'),
                )
                self.sites[site.site_id] = site
        else:
            logger.warning(
                f"Site metadata file {self.metadata_path} not found. Using empty metadata.")

    def add_site(self, site: SiteMetadata):
        """Add or update site metadata."""
        self.sites[site.site_id] = site

    def get_site(self, site_id: str) -> Optional[SiteMetadata]:
        """Get metadata for a site."""
        return self.sites.get(site_id)

    def get_coordinates(self, site_id: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a site."""
        site = self.get_site(site_id)
        return site.coordinates if site else None

    def get_all_sites(self) -> List[SiteMetadata]:
        """Get all registered sites."""
        return list(self.sites.values())

    def get_sites_in_bbox(self, min_lat: float, max_lat: float,
                          min_lon: float, max_lon: float) -> List[SiteMetadata]:
        """Get sites within bounding box."""
        return [
            site for site in self.sites.values()
            if (min_lat <= site.latitude <= max_lat and
                min_lon <= site.longitude <= max_lon)
        ]

    def save_metadata(self):
        """Save site metadata to file."""
        if self.sites:
            data = []
            for site in self.sites.values():
                data.append({
                    'site_id': site.site_id,
                    'latitude': site.latitude,
                    'longitude': site.longitude,
                    'elevation_m': site.elevation_m,
                    'soil_texture': site.soil_texture,
                    'sand_percent': site.sand_percent,
                    'clay_percent': site.clay_percent,
                    'organic_matter_percent': site.organic_matter_percent,
                    'land_cover': site.land_cover,
                })

            df = pd.DataFrame(data)
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.metadata_path, index=False)
            logger.info(
                f"Saved metadata for {len(self.sites)} sites to {self.metadata_path}")

    def create_coordinate_features(self, df: pd.DataFrame,
                                   site_col: str = 'station_id') -> pd.DataFrame:
        """
        Add coordinate-based features to dataframe.

        Replaces nominal site encoding with continuous lat/lon features.
        """
        df = df.copy()

        # Add coordinate features
        df['latitude'] = df[site_col].map(
            lambda x: self.sites.get(x, SiteMetadata(x, 0, 0)).latitude)
        df['longitude'] = df[site_col].map(
            lambda x: self.sites.get(x, SiteMetadata(x, 0, 0)).longitude)

        # Add coordinate-based features
        df['lat_sin'] = np.sin(np.radians(df['latitude']))
        df['lat_cos'] = np.cos(np.radians(df['latitude']))
        df['lon_sin'] = np.sin(np.radians(df['longitude']))
        df['lon_cos'] = np.cos(np.radians(df['longitude']))

        # Distance from reference point (can help with spatial patterns)
        ref_lat, ref_lon = 30.0, 0.0  # Tunisian reference
        df['lat_offset'] = df['latitude'] - ref_lat
        df['lon_offset'] = df['longitude'] - ref_lon

        return df
