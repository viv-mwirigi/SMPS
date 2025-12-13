"""
Soil data source implementations.
Currently supports SoilGrids API and mock data for testing.
"""
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

from smps.data.sources.base import SoilSource, DataFetchRequest
from smps.data.contracts import SoilProfile
from smps.physics.pedotransfer import (
    estimate_soil_properties_from_texture,
    classify_soil_texture,
    estimate_hydraulic_conductivity
)
from smps.core.exceptions import DataSourceError
from smps.core.types import SiteID


class SoilGridsSource(SoilSource):
    """
    Soil data source using SoilGrids API (ISRIC).
    Provides global soil property maps at 250m resolution.
    """

    BASE_URL = "https://rest.isric.org/soilgrids/v2.0/properties"

    # Available soil properties
    PROPERTIES = [
        "bdod",  # Bulk density
        "cec",   # Cation exchange capacity
        "cfvo",  # Coarse fragments
        "clay",  # Clay content
        "nitrogen",  # Nitrogen
        "ocd",   # Organic carbon density
        "ocs",   # Organic carbon stocks
        "phh2o", # pH
        "sand",  # Sand content
        "silt",  # Silt content
        "soc",   # Soil organic carbon
        "wv0010", # Water content at 10kPa
        "wv0033", # Water content at 33kPa (field capacity)
        "wv1500"  # Water content at 1500kPa (wilting point)
    ]

    DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]

    def __init__(self, cache_dir: Optional[Path] = None):
        super().__init__("soilgrids", cache_dir)
        self.session = requests.Session()

    def fetch_soil_profile(self, site_id: SiteID,
                         depth_cm: Optional[int] = None) -> SoilProfile:
        """
        Fetch soil profile from SoilGrids.
        depth_cm: If provided, returns properties at that depth
                  If None, returns weighted average for top 100cm
        """
        # Get site coordinates
        lat, lon = self._get_site_coordinates(site_id)

        # Prepare request
        properties = ["clay", "sand", "silt", "wv0033", "wv1500"]
        depths = self.DEPTHS

        params = {
            "lat": lat,
            "lon": lon,
            "properties": ",".join(properties),
            "depths": ",".join(depths),
            "value": "mean"  # Can be "mean", "Q0.5", "uncertainty"
        }

        self.logger.info(f"Fetching SoilGrids data: {params}")

        try:
            response = self.session.get(
                f"{self.BASE_URL}/query",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            return self._parse_response(data, site_id, depth_cm)

        except requests.exceptions.RequestException as e:
            raise DataSourceError(f"SoilGrids API error: {e}")

    def _parse_response(self, data: Dict[str, Any],
                       site_id: SiteID,
                       target_depth_cm: Optional[int]) -> SoilProfile:
        """Parse SoilGrids response into SoilProfile"""

        # Extract property data
        properties_data = {}
        for prop in data.get("properties", []):
            prop_name = prop["name"]
            depths_data = prop["depths"]

            for depth in depths_data:
                depth_label = depth["label"]
                depth_min, depth_max = self._parse_depth_label(depth_label)
                value = depth["values"]["mean"]

                if depth_label not in properties_data:
                    properties_data[depth_label] = {}
                properties_data[depth_label][prop_name] = value

        # Calculate weighted averages or get specific depth
        if target_depth_cm:
            # Find the depth layer containing target_depth_cm
            target_layer = None
            for depth_label in properties_data:
                depth_min, depth_max = self._parse_depth_label(depth_label)
                if depth_min <= target_depth_cm < depth_max:
                    target_layer = depth_label
                    break

            if not target_layer:
                # Use shallowest layer as fallback
                target_layer = sorted(properties_data.keys())[0]

            layer_data = properties_data[target_layer]
        else:
            # Calculate weighted average for top 100cm
            layer_data = self._calculate_weighted_average(properties_data, max_depth=100)

        # Extract values
        sand = layer_data.get("sand", 40)  # g/kg to %
        clay = layer_data.get("clay", 30)
        silt = layer_data.get("silt", 30)

        # Convert from g/kg to percentage
        sand_percent = sand / 10.0
        clay_percent = clay / 10.0
        silt_percent = silt / 10.0

        # Field capacity and wilting point (cm³/cm³)
        # SoilGrids provides water content at 33kPa and 1500kPa in cm³/dm³
        # Convert to m³/m³ (same numeric value)
        field_capacity = layer_data.get("wv0033", 0.25) / 100.0  # Convert from %
        wilting_point = layer_data.get("wv1500", 0.10) / 100.0

        # Estimate other properties using pedotransfer functions
        texture_class = classify_soil_texture(sand_percent, clay_percent)

        estimated_props = estimate_soil_properties_from_texture(
            sand_percent, clay_percent, method="saxton_rawls"
        )

        # Use SoilGrids values if available, otherwise use estimates
        porosity = estimated_props.get("porosity", 0.45)
        sat_hyd_cond = estimate_hydraulic_conductivity(
            sand_percent, clay_percent, porosity
        )

        # Create soil profile
        profile = SoilProfile(
            site_id=site_id,
            sand_percent=sand_percent,
            silt_percent=silt_percent,
            clay_percent=clay_percent,
            porosity=porosity,
            field_capacity=field_capacity,
            wilting_point=wilting_point,
            saturated_hydraulic_conductivity_cm_day=sat_hyd_cond,
            profile_depth_cm=100.0,  # Standard depth
            effective_rooting_depth_cm=40.0,  # Default
            source="soilgrids",
            confidence=0.8  # SoilGrids has uncertainty
        )

        return profile

    def _parse_depth_label(self, label: str) -> Tuple[float, float]:
        """Parse depth label like '0-5cm' into (min, max) in cm"""
        # Remove 'cm' and split
        parts = label.replace("cm", "").split("-")
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
        else:
            return 0.0, 5.0  # Default

    def _calculate_weighted_average(self, properties_data: Dict[str, Dict],
                                  max_depth: float = 100.0) -> Dict[str, float]:
        """Calculate depth-weighted average of soil properties"""
        weighted_sum = {}
        total_weight = 0.0

        for depth_label, props in properties_data.items():
            depth_min, depth_max = self._parse_depth_label(depth_label)
            depth_thickness = min(depth_max, max_depth) - max(depth_min, 0)

            if depth_thickness <= 0:
                continue

            for prop_name, value in props.items():
                if prop_name not in weighted_sum:
                    weighted_sum[prop_name] = 0.0
                weighted_sum[prop_name] += value * depth_thickness

            total_weight += depth_thickness

        # Calculate averages
        averages = {}
        for prop_name, weighted_value in weighted_sum.items():
            averages[prop_name] = weighted_value / total_weight if total_weight > 0 else 0.0

        return averages

    def _get_site_coordinates(self, site_id: SiteID) -> Tuple[float, float]:
        """Get coordinates for site (mock implementation)"""
        # Same as in weather source - would come from site config
        site_coords = {
            "test_site_001": (35.222866, 9.090245),
            "test_site_002": (34.0, 8.0),
        }

        return site_coords.get(site_id, (34.0, 9.0))

    def get_metadata(self) -> Dict[str, Any]:
        """Get source metadata"""
        base_metadata = super().get_metadata()
        base_metadata.update({
            "provider": "ISRIC SoilGrids",
            "spatial_resolution": "250m",
            "temporal_resolution": "static",
            "update_frequency": "periodic",
            "available_properties": self.PROPERTIES,
            "available_depths": self.DEPTHS,
            "data_license": "Creative Commons Attribution 4.0 International"
        })
        return base_metadata


class MockSoilSource(SoilSource):
    """
    Mock soil source for testing and development.
    Generates realistic synthetic soil profiles.
    """

    # Predefined soil texture classes with typical properties
    SOIL_CLASSES = {
        "sand": {
            "sand": 85, "clay": 10, "silt": 5,
            "porosity": 0.43, "fc": 0.10, "wp": 0.03,
            "k_sat": 100.0, "description": "Coarse-textured"
        },
        "loam": {
            "sand": 40, "clay": 20, "silt": 40,
            "porosity": 0.47, "fc": 0.27, "wp": 0.15,
            "k_sat": 25.0, "description": "Medium-textured"
        },
        "clay_loam": {
            "sand": 30, "clay": 35, "silt": 35,
            "porosity": 0.48, "fc": 0.32, "wp": 0.24,
            "k_sat": 10.0, "description": "Fine-textured"
        },
        "clay": {
            "sand": 20, "clay": 60, "silt": 20,
            "porosity": 0.46, "fc": 0.38, "wp": 0.27,
            "k_sat": 5.0, "description": "Very fine-textured"
        }
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        super().__init__("mock_soil", cache_dir)

    def fetch_soil_profile(self, site_id: SiteID,
                         depth_cm: Optional[int] = None) -> SoilProfile:
        """Generate synthetic soil profile"""

        # Determine soil class based on site_id (deterministic hash)
        soil_classes = list(self.SOIL_CLASSES.keys())
        class_idx = hash(site_id) % len(soil_classes)
        soil_class = soil_classes[class_idx]
        class_props = self.SOIL_CLASSES[soil_class]

        # Add some randomness
        import random
        random.seed(hash(site_id))

        sand = class_props["sand"] + random.uniform(-5, 5)
        clay = class_props["clay"] + random.uniform(-5, 5)
        silt = 100 - sand - clay

        # Ensure valid ranges
        sand = max(0, min(100, sand))
        clay = max(0, min(100, clay))
        silt = max(0, min(100, silt))

        # Normalize to sum to 100
        total = sand + clay + silt
        sand = (sand / total) * 100
        clay = (clay / total) * 100
        silt = (silt / total) * 100

        # Other properties with some variation
        porosity = class_props["porosity"] + random.uniform(-0.02, 0.02)
        field_capacity = class_props["fc"] + random.uniform(-0.03, 0.03)
        wilting_point = class_props["wp"] + random.uniform(-0.02, 0.02)
        k_sat = class_props["k_sat"] * random.uniform(0.8, 1.2)

        # Create profile
        profile = SoilProfile(
            site_id=site_id,
            sand_percent=round(sand, 1),
            silt_percent=round(silt, 1),
            clay_percent=round(clay, 1),
            porosity=round(porosity, 3),
            field_capacity=round(field_capacity, 3),
            wilting_point=round(wilting_point, 3),
            saturated_hydraulic_conductivity_cm_day=round(k_sat, 1),
            profile_depth_cm=100.0,
            effective_rooting_depth_cm=40.0,
            source="mock",
            confidence=0.9
        )

        return profile

    def get_metadata(self) -> Dict[str, Any]:
        """Get source metadata"""
        base_metadata = super().get_metadata()
        base_metadata.update({
            "provider": "Mock Soil Data",
            "available_classes": list(self.SOIL_CLASSES.keys()),
            "description": "Synthetic soil data for testing"
        })
        return base_metadata