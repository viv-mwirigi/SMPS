"""
iSDA Africa Soil Data API - Authenticated Implementation.
Uses the official iSDA API with token authentication.

API Documentation: https://api.isda-africa.com/docs
"""
import os
import requests
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

from smps.data.sources.base import SoilSource, DataFetchRequest, DataFetchResult
from smps.data.contracts import SoilProfile
from smps.physics.pedotransfer import (
    estimate_soil_parameters_saxton,
    classify_soil_texture,
)
from smps.core.exceptions import DataSourceError
from smps.core.types import SiteID

# Load environment variables
load_dotenv()

logger = logging.getLogger("smps.data.isda")


@dataclass
class IsdaToken:
    """iSDA API authentication token"""
    access_token: str
    expires_at: datetime

    @property
    def is_expired(self) -> bool:
        return datetime.now() >= self.expires_at


class IsdaAfricaAuthenticatedSource(SoilSource):
    """
    iSDA Africa Soil Data API with proper authentication.

    Provides soil properties for Africa at 30m resolution.

    Available Properties:
    - Texture: clay, sand, silt (g/kg)
    - Chemical: pH, CEC, nitrogen, carbon, phosphorus
    - Physical: bulk density, stone content
    - Derived: texture class, fertility class

    Depths: 0-20cm, 20-50cm

    Reference: https://www.isda-africa.com/isdasoil/
    """

    BASE_URL = "https://api.isda-africa.com"

    # Available properties and their units
    PROPERTIES = {
        # Texture (g/kg) - Note: API uses clay_content, sand_content, silt_content
        "clay": {"endpoint": "clay_content", "unit": "g/kg", "scale": 0.1},  # to %
        "sand": {"endpoint": "sand_content", "unit": "g/kg", "scale": 0.1},
        "silt": {"endpoint": "silt_content", "unit": "g/kg", "scale": 0.1},

        # Physical
        "bulk_density": {"endpoint": "bulk_density", "unit": "g/cm³", "scale": 1.0},
        "stone_content": {"endpoint": "stone_content", "unit": "%", "scale": 1.0},
        "bedrock_depth": {"endpoint": "bedrock_depth", "unit": "cm", "scale": 1.0},

        # Chemical
        "ph": {"endpoint": "ph", "unit": "pH", "scale": 1.0},
        "cec": {"endpoint": "cation_exchange_capacity", "unit": "cmol(+)/kg", "scale": 1.0},
        "nitrogen": {"endpoint": "nitrogen_total", "unit": "g/kg", "scale": 1.0},
        "carbon_organic": {"endpoint": "carbon_organic", "unit": "g/kg", "scale": 1.0},
        "carbon_total": {"endpoint": "carbon_total", "unit": "g/kg", "scale": 1.0},
        "phosphorus": {"endpoint": "phosphorous_extractable", "unit": "mg/kg", "scale": 1.0},
        "potassium": {"endpoint": "potassium_extractable", "unit": "mg/kg", "scale": 1.0},
        "calcium": {"endpoint": "calcium_extractable", "unit": "mg/kg", "scale": 1.0},
        "magnesium": {"endpoint": "magnesium_extractable", "unit": "mg/kg", "scale": 1.0},
        "iron": {"endpoint": "iron_extractable", "unit": "mg/kg", "scale": 1.0},
        "zinc": {"endpoint": "zinc_extractable", "unit": "mg/kg", "scale": 1.0},
        "sulphur": {"endpoint": "sulphur_extractable", "unit": "mg/kg", "scale": 1.0},
        "aluminium": {"endpoint": "aluminium_extractable", "unit": "mg/kg", "scale": 1.0},

        # Classification
        "texture_class": {"endpoint": "texture_class", "unit": "class", "scale": 1.0},
        "fcc": {"endpoint": "fcc", "unit": "class", "scale": 1.0},  # Fertility capability

        # Topography
        "slope_angle": {"endpoint": "slope_angle", "unit": "degrees", "scale": 1.0},
    }

    # Depth options
    DEPTHS = ["0-20", "20-50"]

    def __init__(self,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 cache_dir: Optional[Path] = None):
        """
        Initialize iSDA authenticated source.

        Args:
            username: iSDA API username (or set ISDA_USERNAME env var)
            password: iSDA API password (or set ISDA_PASSWORD env var)
            cache_dir: Directory for caching responses
        """
        super().__init__("isda_africa", cache_dir)

        self.username = username or os.getenv("ISDA_USERNAME")
        self.password = password or os.getenv("ISDA_PASSWORD")

        if not self.username or not self.password:
            logger.warning(
                "iSDA credentials not provided. Set ISDA_USERNAME and ISDA_PASSWORD "
                "environment variables or pass to constructor."
            )

        self.session = requests.Session()
        self._token: Optional[IsdaToken] = None

    def _authenticate(self) -> bool:
        """Authenticate with iSDA API and get access token."""
        if not self.username or not self.password:
            raise DataSourceError("iSDA credentials not configured")

        try:
            response = self.session.post(
                f"{self.BASE_URL}/login",
                data={
                    "username": self.username,
                    "password": self.password
                },
                timeout=30
            )

            if response.status_code != 200:
                logger.error(f"iSDA login failed: {response.status_code}")
                return False

            data = response.json()

            if "access_token" not in data:
                logger.error(f"iSDA login failed: {data}")
                return False

            # Token expires in 60 minutes
            self._token = IsdaToken(
                access_token=data["access_token"],
                expires_at=datetime.now() + timedelta(minutes=55)  # 5 min buffer
            )

            logger.info("iSDA authentication successful")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"iSDA authentication error: {e}")
            return False

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers, refreshing token if needed."""
        if self._token is None or self._token.is_expired:
            if not self._authenticate():
                raise DataSourceError("Failed to authenticate with iSDA API")

        return {"Authorization": f"Bearer {self._token.access_token}"}

    def fetch_soil_profile(self,
                          site_id: SiteID,
                          latitude: Optional[float] = None,
                          longitude: Optional[float] = None,
                          depth: str = "0-20") -> SoilProfile:
        """
        Fetch complete soil profile from iSDA.

        Args:
            site_id: Site identifier
            latitude: Site latitude
            longitude: Site longitude
            depth: Depth layer ("0-20" or "20-50")

        Returns:
            SoilProfile with all available properties
        """
        # Get coordinates
        if latitude is None or longitude is None:
            latitude, longitude = self._get_site_coordinates(site_id)

        # Validate location is in Africa
        if not self._is_within_africa(latitude, longitude):
            raise DataSourceError(
                f"Location ({latitude}, {longitude}) is outside iSDA Africa coverage"
            )

        # Fetch all properties
        properties = self._fetch_all_properties(latitude, longitude, depth)

        # Create soil profile
        return self._create_profile(site_id, latitude, longitude, properties, depth)

    def _fetch_all_properties(self, lat: float, lon: float, depth: str) -> Dict[str, Any]:
        """Fetch all soil properties for a location."""
        headers = self._get_auth_headers()
        properties = {}

        for prop_name, prop_info in self.PROPERTIES.items():
            try:
                endpoint = prop_info["endpoint"]
                url = f"{self.BASE_URL}/isdasoil/v2/soilproperty"

                params = {
                    "lat": lat,
                    "lon": lon,
                    "property": endpoint,
                    "depth": depth
                }

                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()

                    # Extract value from nested structure
                    if "property" in data and endpoint in data["property"]:
                        prop_data = data["property"][endpoint][0]
                        value = prop_data.get("value", {}).get("value")

                        if value is not None:
                            # Apply scale factor
                            properties[prop_name] = value * prop_info["scale"]

                            # Also store uncertainty if available
                            uncertainty = prop_data.get("uncertainty", [])
                            if uncertainty:
                                properties[f"{prop_name}_uncertainty"] = {
                                    "lower": uncertainty[0].get("lower_bound"),
                                    "upper": uncertainty[0].get("upper_bound")
                                }
                else:
                    logger.warning(f"Failed to fetch {prop_name}: {response.status_code}")

            except Exception as e:
                logger.warning(f"Error fetching {prop_name}: {e}")

        return properties

    def fetch_single_property(self,
                             lat: float,
                             lon: float,
                             property_name: str,
                             depth: str = "0-20") -> Optional[float]:
        """
        Fetch a single soil property.

        Args:
            lat: Latitude
            lon: Longitude
            property_name: Property name (e.g., "clay", "ph", "nitrogen")
            depth: Depth layer

        Returns:
            Property value or None if not available
        """
        if property_name not in self.PROPERTIES:
            raise ValueError(f"Unknown property: {property_name}. "
                           f"Available: {list(self.PROPERTIES.keys())}")

        headers = self._get_auth_headers()
        prop_info = self.PROPERTIES[property_name]

        try:
            response = self.session.get(
                f"{self.BASE_URL}/isdasoil/v2/soilproperty",
                params={
                    "lat": lat,
                    "lon": lon,
                    "property": prop_info["endpoint"],
                    "depth": depth
                },
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                endpoint = prop_info["endpoint"]

                if "property" in data and endpoint in data["property"]:
                    value = data["property"][endpoint][0].get("value", {}).get("value")
                    if value is not None:
                        return value * prop_info["scale"]

        except Exception as e:
            logger.error(f"Error fetching {property_name}: {e}")

        return None

    def _create_profile(self, site_id: SiteID, lat: float, lon: float,
                       properties: Dict[str, Any], depth: str) -> SoilProfile:
        """Create SoilProfile from fetched properties."""

        # Extract texture
        sand = properties.get("sand", 40.0)
        clay = properties.get("clay", 20.0)
        silt = properties.get("silt", 40.0)

        # Normalize to 100%
        total = sand + clay + silt
        if total > 0 and abs(total - 100) > 1:
            sand = (sand / total) * 100
            clay = (clay / total) * 100
            silt = (silt / total) * 100

        # Get bulk density
        bulk_density = properties.get("bulk_density", 1.4)

        # Estimate hydraulic properties using pedotransfer
        estimated = estimate_soil_parameters_saxton(sand, clay)

        # Calculate porosity from bulk density
        # porosity = 1 - (bulk_density / particle_density)
        # particle density typically 2.65 g/cm³
        porosity_from_bd = 1 - (bulk_density / 2.65)
        porosity = estimated.porosity if hasattr(estimated, 'porosity') else porosity_from_bd

        field_capacity = estimated.field_capacity if hasattr(estimated, 'field_capacity') else 0.27
        wilting_point = estimated.wilting_point if hasattr(estimated, 'wilting_point') else 0.12
        ksat = estimated.saturated_hydraulic_conductivity_cm_day if hasattr(estimated, 'saturated_hydraulic_conductivity_cm_day') else 10.0

        # Determine profile depth from depth parameter
        profile_depth = 50.0 if depth == "20-50" else 20.0

        return SoilProfile(
            site_id=site_id,
            sand_percent=round(sand, 1),
            silt_percent=round(silt, 1),
            clay_percent=round(clay, 1),
            porosity=round(porosity, 3),
            field_capacity=round(field_capacity, 3),
            wilting_point=round(wilting_point, 3),
            saturated_hydraulic_conductivity_cm_day=round(ksat, 2),
            profile_depth_cm=profile_depth,
            effective_rooting_depth_cm=40.0,
            source="isda_africa_api",
            confidence=0.90
        )

    def _is_within_africa(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within Africa bounds."""
        return -35 <= lat <= 37 and -25 <= lon <= 55

    def _get_site_coordinates(self, site_id: SiteID) -> Tuple[float, float]:
        """Get coordinates for site from registry."""
        site_coords = {
            "tunisia_sfax": (34.740, 10.760),
            "kenya_eldoret": (0.5143, 35.2698),
            "kenya_nairobi": (-1.2921, 36.8219),
            "ghana_kumasi": (6.6885, -1.6244),
            "ghana_accra": (5.6037, -0.1870),
            "ethiopia_addis": (9.0320, 38.7497),
            "nigeria_kano": (12.0022, 8.5919),
            "nigeria_ibadan": (7.3775, 3.9470),
            "southafrica_cape": (-33.9249, 18.4241),
            "morocco_marrakech": (31.6295, -7.9811),
            "egypt_cairo": (30.0444, 31.2357),
            "senegal_dakar": (14.7167, -17.4677),
        }
        return site_coords.get(site_id, (0.0, 35.0))

    def get_available_properties(self) -> List[str]:
        """Get list of available properties."""
        return list(self.PROPERTIES.keys())

    def get_metadata(self) -> Dict[str, Any]:
        """Get source metadata."""
        return {
            "name": self.name,
            "provider": "iSDA Africa (Innovative Solutions for Decision Agriculture)",
            "spatial_resolution": "30m",
            "temporal_resolution": "static (2001-2017 composites)",
            "coverage": "Africa continent",
            "available_properties": list(self.PROPERTIES.keys()),
            "available_depths": self.DEPTHS,
            "data_license": "Creative Commons Attribution 4.0",
            "api_docs": "https://api.isda-africa.com/docs",
            "reference": "Hengl et al. (2021) African Soil Properties"
        }


# Convenience function
def get_isda_soil_data(lat: float, lon: float, depth: str = "0-20") -> Dict[str, Any]:
    """
    Quick function to get soil data from iSDA.

    Example:
        >>> data = get_isda_soil_data(-0.7196, 35.2400)
        >>> print(f"Clay: {data['clay']}%")
    """
    source = IsdaAfricaAuthenticatedSource()
    profile = source.fetch_soil_profile("query", latitude=lat, longitude=lon, depth=depth)

    return {
        "sand_percent": profile.sand_percent,
        "clay_percent": profile.clay_percent,
        "silt_percent": profile.silt_percent,
        "porosity": profile.porosity,
        "field_capacity": profile.field_capacity,
        "wilting_point": profile.wilting_point,
        "ksat_cm_day": profile.saturated_hydraulic_conductivity_cm_day,
        "source": profile.source,
        "confidence": profile.confidence
    }
