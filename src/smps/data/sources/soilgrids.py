"""
SoilGrids Global Soil Data Source.

SoilGrids provides global soil property maps at 250m resolution.
Unlike iSDA (Africa-only), SoilGrids covers the entire world.

API Documentation: https://rest.isric.org/soilgrids/v2.0/docs
"""
import requests
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import logging
from functools import lru_cache

from smps.data.sources.base import SoilSource
from smps.data.contracts import SoilProfile
from smps.physics.pedotransfer import estimate_soil_parameters_saxton
from smps.core.exceptions import DataSourceError
from smps.core.types import SiteID

logger = logging.getLogger("smps.data.soilgrids")


class SoilGridsGlobalSource(SoilSource):
    """
    SoilGrids 2.0 - Global Soil Information System.

    Provides soil properties globally at 250m resolution.
    Free access, no authentication required for REST API.

    Available Properties:
    - bdod: Bulk density (cg/cm³)
    - cec: Cation Exchange Capacity (mmol(c)/kg)
    - cfvo: Coarse fragments volume (cm³/dm³)
    - clay: Clay content (g/kg)
    - nitrogen: Nitrogen content (cg/kg)
    - ocd: Organic carbon density (hg/m³)
    - ocs: Organic carbon stock (t/ha)
    - phh2o: pH in H2O (pH*10)
    - sand: Sand content (g/kg)
    - silt: Silt content (g/kg)
    - soc: Soil organic carbon (dg/kg)
    - wv0010: Volumetric water content at 10 kPa (0.1 bar) (cm³/dm³)
    - wv0033: Volumetric water content at 33 kPa (0.3 bar, field capacity) (cm³/dm³)
    - wv1500: Volumetric water content at 1500 kPa (15 bar, wilting point) (cm³/dm³)

    Depths: 0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm

    Reference: https://www.isric.org/explore/soilgrids
    """

    BASE_URL = "https://rest.isric.org/soilgrids/v2.0"

    # Property specifications with conversion factors
    PROPERTIES = {
        "clay": {"unit": "g/kg", "to_percent": 0.1},
        "sand": {"unit": "g/kg", "to_percent": 0.1},
        "silt": {"unit": "g/kg", "to_percent": 0.1},
        "bdod": {"unit": "cg/cm³", "to_gcm3": 0.01},  # Bulk density
        "phh2o": {"unit": "pH*10", "to_ph": 0.1},
        "cec": {"unit": "mmol(c)/kg", "scale": 1.0},
        "nitrogen": {"unit": "cg/kg", "scale": 0.01},
        "soc": {"unit": "dg/kg", "scale": 0.1},
        "cfvo": {"unit": "cm³/dm³", "scale": 0.1},  # Coarse fragments
        "wv0033": {"unit": "cm³/dm³", "to_m3m3": 0.001},  # Field capacity
        "wv1500": {"unit": "cm³/dm³", "to_m3m3": 0.001},  # Wilting point
    }

    # Depth mappings
    DEPTHS = {
        "0-5": "0-5cm",
        "5-15": "5-15cm",
        "15-30": "15-30cm",
        "30-60": "30-60cm",
        "60-100": "60-100cm",
        "100-200": "100-200cm"
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize SoilGrids source."""
        super().__init__("soilgrids", cache_dir)
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "SMPS-SoilMoisture/1.0"
        })

    def fetch_soil_profile(self,
                          site_id: SiteID,
                          latitude: float,
                          longitude: float,
                          depth: str = "0-5cm") -> SoilProfile:
        """
        Fetch soil profile from SoilGrids.

        Args:
            site_id: Site identifier
            latitude: Site latitude (-90 to 90)
            longitude: Site longitude (-180 to 180)
            depth: Depth layer (default "0-5cm")

        Returns:
            SoilProfile with soil properties
        """
        # Fetch all properties at once
        properties = self._fetch_point_properties(latitude, longitude, depth)

        # Create profile from properties
        return self._create_profile(site_id, latitude, longitude, properties, depth)

    def _fetch_point_properties(self,
                                lat: float,
                                lon: float,
                                depth: str) -> Dict[str, Any]:
        """Fetch all properties for a point."""
        # Validate coordinates
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise DataSourceError(f"Invalid coordinates: ({lat}, {lon})")

        properties = {}

        # Convert depth format if needed
        depth_param = depth.replace("cm", "").strip()

        # Properties we need for soil moisture modeling
        needed = ["clay", "sand", "silt", "bdod", "phh2o", "wv0033", "wv1500"]

        for prop in needed:
            try:
                url = f"{self.BASE_URL}/properties/query"
                params = {
                    "lat": lat,
                    "lon": lon,
                    "property": prop,
                    "depth": f"{depth_param}cm",
                    "value": "mean"  # Use mean prediction
                }

                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()

                    # Extract value from nested response
                    if "properties" in data:
                        layers = data["properties"].get("layers", [])
                        if layers:
                            depths = layers[0].get("depths", [])
                            if depths:
                                values = depths[0].get("values", {})
                                mean_val = values.get("mean")
                                if mean_val is not None:
                                    properties[prop] = mean_val

                elif response.status_code == 404:
                    logger.warning(f"SoilGrids: No data for {prop} at ({lat}, {lon})")
                else:
                    logger.warning(f"SoilGrids error for {prop}: {response.status_code}")

            except Exception as e:
                logger.warning(f"Error fetching {prop}: {e}")

        return properties

    @lru_cache(maxsize=100)
    def fetch_properties_cached(self, lat: float, lon: float, depth: str) -> Dict[str, float]:
        """Cached version of property fetch."""
        lat_r = round(lat, 4)
        lon_r = round(lon, 4)
        return self._fetch_point_properties(lat_r, lon_r, depth)

    def _create_profile(self,
                       site_id: SiteID,
                       lat: float,
                       lon: float,
                       properties: Dict[str, Any],
                       depth: str) -> SoilProfile:
        """Create SoilProfile from SoilGrids properties."""

        # Extract and convert texture (g/kg to %)
        sand = properties.get("sand", 400) * 0.1  # Default 40%
        clay = properties.get("clay", 200) * 0.1  # Default 20%
        silt = properties.get("silt", 400) * 0.1  # Default 40%

        # Normalize texture to 100%
        total = sand + clay + silt
        if total > 0 and abs(total - 100) > 1:
            sand = (sand / total) * 100
            clay = (clay / total) * 100
            silt = (silt / total) * 100

        # Bulk density (cg/cm³ to g/cm³)
        bulk_density = properties.get("bdod", 140) * 0.01

        # Use SoilGrids water retention if available
        field_capacity = properties.get("wv0033")
        wilting_point = properties.get("wv1500")

        if field_capacity is not None:
            field_capacity *= 0.001  # cm³/dm³ to m³/m³
        if wilting_point is not None:
            wilting_point *= 0.001

        # Otherwise use pedotransfer functions
        estimated = estimate_soil_parameters_saxton(sand, clay)

        if field_capacity is None:
            field_capacity = estimated.field_capacity if hasattr(estimated, 'field_capacity') else 0.27
        if wilting_point is None:
            wilting_point = estimated.wilting_point if hasattr(estimated, 'wilting_point') else 0.12

        # Porosity from bulk density
        porosity = 1 - (bulk_density / 2.65)
        porosity = max(0.3, min(0.7, porosity))  # Reasonable bounds

        # Saturated hydraulic conductivity from pedotransfer
        ksat = estimated.saturated_hydraulic_conductivity_cm_day if hasattr(estimated, 'saturated_hydraulic_conductivity_cm_day') else 10.0

        # Depth from layer
        depth_map = {
            "0-5cm": 5, "5-15cm": 15, "15-30cm": 30,
            "30-60cm": 60, "60-100cm": 100, "100-200cm": 200
        }
        profile_depth = depth_map.get(depth, 30)

        return SoilProfile(
            site_id=site_id,
            sand_percent=round(sand, 1),
            silt_percent=round(silt, 1),
            clay_percent=round(clay, 1),
            porosity=round(porosity, 3),
            field_capacity=round(field_capacity, 3),
            wilting_point=round(wilting_point, 3),
            saturated_hydraulic_conductivity_cm_day=round(ksat, 2),
            profile_depth_cm=float(profile_depth),
            effective_rooting_depth_cm=40.0,
            source="soilgrids_v2",
            confidence=0.80  # Slightly lower than iSDA for Africa
        )

    def get_available_properties(self) -> List[str]:
        """Get list of available properties."""
        return list(self.PROPERTIES.keys())

    def check_coverage(self, lat: float, lon: float) -> bool:
        """Check if coordinates have SoilGrids coverage."""
        try:
            response = self.session.get(
                f"{self.BASE_URL}/properties/query",
                params={
                    "lat": lat,
                    "lon": lon,
                    "property": "clay",
                    "depth": "0-5cm",
                    "value": "mean"
                },
                timeout=10
            )
            return response.status_code == 200
        except:
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """Get source metadata."""
        return {
            "name": self.name,
            "provider": "ISRIC - World Soil Information",
            "spatial_resolution": "250m",
            "temporal_resolution": "static (2017 harmonization)",
            "coverage": "Global (except Antarctica)",
            "available_properties": list(self.PROPERTIES.keys()),
            "available_depths": list(self.DEPTHS.values()),
            "data_license": "Open Data Commons Open Database License (ODbL)",
            "api_docs": "https://rest.isric.org/soilgrids/v2.0/docs",
            "reference": "Poggio et al. (2021) SoilGrids 2.0"
        }


# Convenience function
def get_soilgrids_profile(lat: float, lon: float, depth: str = "0-5cm") -> Dict[str, Any]:
    """
    Quick function to get soil profile from SoilGrids.

    Example:
        >>> profile = get_soilgrids_profile(52.1, 5.2)
        >>> print(f"Clay: {profile['clay_percent']}%")
    """
    source = SoilGridsGlobalSource()
    profile = source.fetch_soil_profile("query", lat, lon, depth)

    return {
        "sand_percent": profile.sand_percent,
        "clay_percent": profile.clay_percent,
        "silt_percent": profile.silt_percent,
        "porosity": profile.porosity,
        "field_capacity": profile.field_capacity,
        "wilting_point": profile.wilting_point,
        "ksat_cm_day": profile.saturated_hydraulic_conductivity_cm_day,
        "source": profile.source
    }


if __name__ == "__main__":
    # Test query
    import sys

    lat = float(sys.argv[1]) if len(sys.argv) > 1 else 52.1
    lon = float(sys.argv[2]) if len(sys.argv) > 2 else 5.2

    print(f"Querying SoilGrids for ({lat}, {lon})...")
    profile = get_soilgrids_profile(lat, lon)

    for k, v in profile.items():
        print(f"  {k}: {v}")
