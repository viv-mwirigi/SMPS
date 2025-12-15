"""
GRAFS - Global Reference Agricultural Field Stations.

Provides validation/reference data for soil moisture and agricultural parameters.

Note: GRAFS is primarily a validation data source with in-situ measurements.
For soil moisture validation, we also support:
- ISMN (International Soil Moisture Network)
- FLUXNET (eddy covariance flux measurements)
"""
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from smps.core.exceptions import DataSourceError

load_dotenv()
logger = logging.getLogger("smps.data.validation")


@dataclass
class ValidationObservation:
    """A single validation measurement."""
    site_id: str
    timestamp: datetime
    soil_moisture: Optional[float] = None  # m³/m³
    soil_moisture_depth_cm: Optional[float] = None
    soil_temperature: Optional[float] = None  # °C
    surface_temperature: Optional[float] = None
    precipitation: Optional[float] = None  # mm
    evapotranspiration: Optional[float] = None  # mm/day
    quality_flag: float = 1.0
    source: str = "unknown"


class ISMNDataSource:
    """
    International Soil Moisture Network (ISMN) data source.

    ISMN provides harmonized in-situ soil moisture data from
    global observation networks.

    Registration required: https://ismn.geo.tuwien.ac.at/en/

    Data covers:
    - 3000+ stations globally
    - Various depths (typically 5, 10, 20, 50 cm)
    - Hourly to daily frequency
    - Historical data from 1950s onwards
    """

    BASE_URL = "https://ismn.geo.tuwien.ac.at"

    def __init__(self):
        self.username = os.getenv("ISMN_USERNAME")
        self.password = os.getenv("ISMN_PASSWORD")
        self.session = requests.Session()

    def fetch_station_data(self,
                          network: str,
                          station: str,
                          start_date: datetime,
                          end_date: datetime,
                          depth_cm: float = 5.0) -> List[ValidationObservation]:
        """
        Fetch soil moisture observations from an ISMN station.

        Args:
            network: Network name (e.g., "COSMOS", "USCRN")
            station: Station name
            start_date: Start of period
            end_date: End of period
            depth_cm: Measurement depth in cm

        Returns:
            List of ValidationObservation objects
        """
        if not self.username or not self.password:
            logger.warning("ISMN credentials not configured. Using synthetic data.")
            return self._generate_synthetic_data(station, start_date, end_date, depth_cm)

        # Note: ISMN typically requires downloading full datasets
        # Their API access may vary - check their documentation
        try:
            # This is a placeholder for the actual ISMN API call
            # The real implementation would use their specific API
            logger.info(f"Fetching ISMN data for {network}/{station}")

            # For now, return synthetic data as ISMN requires bulk download
            return self._generate_synthetic_data(station, start_date, end_date, depth_cm)

        except Exception as e:
            logger.error(f"ISMN fetch error: {e}")
            return self._generate_synthetic_data(station, start_date, end_date, depth_cm)

    def _generate_synthetic_data(self,
                                 station: str,
                                 start_date: datetime,
                                 end_date: datetime,
                                 depth_cm: float) -> List[ValidationObservation]:
        """Generate synthetic validation data for testing."""
        observations = []
        current = start_date

        # Parameters for synthetic generation
        base_sm = 0.25  # Base soil moisture
        seasonal_amp = 0.08
        noise_std = 0.02

        while current <= end_date:
            doy = current.timetuple().tm_yday

            # Seasonal pattern (wetter in winter/spring)
            seasonal = seasonal_amp * np.sin(2 * np.pi * (doy + 80) / 365)

            # Add noise
            sm = base_sm + seasonal + np.random.normal(0, noise_std)
            sm = max(0.05, min(0.55, sm))  # Bounds

            observations.append(ValidationObservation(
                site_id=station,
                timestamp=current,
                soil_moisture=round(sm, 4),
                soil_moisture_depth_cm=depth_cm,
                quality_flag=1.0,
                source="ismn_synthetic"
            ))

            current += timedelta(days=1)

        return observations

    def get_available_networks(self) -> List[str]:
        """Get list of available ISMN networks."""
        return [
            "AMMA-CATCH", "ARM", "BNZ-LTER", "COSMOS",
            "CTP_SMTMN", "FMI", "FR_Aqui", "GTK",
            "HOBE", "REMEDHUS", "RSMN", "SCAN",
            "SMOSMANIA", "TERENO", "USCRN", "WSMN"
        ]


class FluxnetDataSource:
    """
    FLUXNET data source for eddy covariance measurements.

    Provides:
    - Evapotranspiration (latent heat flux)
    - Sensible heat flux
    - Net radiation
    - Soil temperature and moisture

    Useful for validating water balance calculations.
    """

    def __init__(self):
        self.name = "fluxnet"

    def fetch_site_data(self,
                       site_id: str,
                       start_date: datetime,
                       end_date: datetime) -> List[Dict[str, Any]]:
        """
        Fetch FLUXNET data for a site.

        FLUXNET data typically requires registration and download.
        This method returns synthetic data for demonstration.
        """
        logger.info(f"FLUXNET data for {site_id} would be fetched here")

        # Generate synthetic flux data
        data = []
        current = start_date

        while current <= end_date:
            doy = current.timetuple().tm_yday

            # ET varies seasonally (mm/day)
            base_et = 3.0 + 2.0 * np.sin(2 * np.pi * (doy - 172) / 365)
            et = max(0.5, base_et + np.random.normal(0, 0.5))

            data.append({
                "date": current.strftime("%Y-%m-%d"),
                "ET_mm_day": round(et, 2),
                "soil_moisture": round(0.25 + np.random.normal(0, 0.03), 3),
                "source": "fluxnet_synthetic"
            })

            current += timedelta(days=1)

        return data


class ValidationDataManager:
    """
    Manager class for accessing multiple validation data sources.

    Aggregates data from:
    - ISMN (soil moisture)
    - FLUXNET (evapotranspiration)
    - GRAFS (agricultural reference)
    """

    def __init__(self):
        self.ismn = ISMNDataSource()
        self.fluxnet = FluxnetDataSource()

    def get_validation_data(self,
                           lat: float,
                           lon: float,
                           start_date: datetime,
                           end_date: datetime,
                           search_radius_km: float = 50.0) -> Dict[str, Any]:
        """
        Find and fetch validation data near a location.

        Args:
            lat: Target latitude
            lon: Target longitude
            start_date: Start of period
            end_date: End of period
            search_radius_km: Search radius in kilometers

        Returns:
            Dictionary with validation data and metadata
        """
        # Find nearby stations (would query actual databases)
        nearby_stations = self._find_nearby_stations(lat, lon, search_radius_km)

        result = {
            "location": {"latitude": lat, "longitude": lon},
            "period": {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d")
            },
            "stations": [],
            "observations": []
        }

        for station in nearby_stations:
            obs = self.ismn.fetch_station_data(
                station["network"],
                station["name"],
                start_date,
                end_date
            )

            result["stations"].append(station)
            result["observations"].extend(obs)

        return result

    def _find_nearby_stations(self,
                             lat: float,
                             lon: float,
                             radius_km: float) -> List[Dict[str, Any]]:
        """Find validation stations near a location."""
        # In practice, this would query a station database
        # For now, return sample African stations

        sample_stations = [
            {"network": "AMMA-CATCH", "name": "Benin_Upper_Oueme",
             "lat": 9.74, "lon": 1.60, "country": "Benin"},
            {"network": "AMMA-CATCH", "name": "Niger_Wankama",
             "lat": 13.65, "lon": 2.63, "country": "Niger"},
            {"network": "REMEDHUS", "name": "Spain_Salamanca",
             "lat": 41.29, "lon": -5.57, "country": "Spain"},
        ]

        # Filter by distance (simple approximation)
        nearby = []
        for station in sample_stations:
            dist = np.sqrt((station["lat"] - lat)**2 + (station["lon"] - lon)**2) * 111
            if dist <= radius_km:
                station["distance_km"] = round(dist, 1)
                nearby.append(station)

        return nearby

    def compute_validation_metrics(self,
                                   predicted: np.ndarray,
                                   observed: np.ndarray) -> Dict[str, float]:
        """
        Compute validation metrics between predictions and observations.

        Args:
            predicted: Array of predicted values
            observed: Array of observed values

        Returns:
            Dictionary of metrics (RMSE, MAE, Bias, R², NSE, KGE)
        """
        # Remove NaNs
        mask = ~(np.isnan(predicted) | np.isnan(observed))
        pred = predicted[mask]
        obs = observed[mask]

        if len(pred) < 3:
            return {"error": "Insufficient data points"}

        # RMSE
        rmse = np.sqrt(np.mean((pred - obs) ** 2))

        # MAE
        mae = np.mean(np.abs(pred - obs))

        # Bias
        bias = np.mean(pred - obs)

        # R² (coefficient of determination)
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - np.mean(obs)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # NSE (Nash-Sutcliffe Efficiency)
        nse = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # KGE (Kling-Gupta Efficiency)
        r = np.corrcoef(pred, obs)[0, 1] if len(pred) > 1 else 0
        alpha = np.std(pred) / np.std(obs) if np.std(obs) > 0 else 1
        beta = np.mean(pred) / np.mean(obs) if np.mean(obs) > 0 else 1
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "bias": round(bias, 4),
            "r2": round(r2, 4),
            "nse": round(nse, 4),
            "kge": round(kge, 4),
            "n_samples": len(pred)
        }


# =============================================================================
# Additional Soil Moisture Prediction Attributes Research
# =============================================================================

SOIL_MOISTURE_PREDICTION_ATTRIBUTES = """
=== Attributes Needed for Soil Moisture Prediction ===

1. SOIL PROPERTIES (from iSDA/SoilGrids)
   Required:
   - Texture: clay%, sand%, silt%
   - Bulk density (g/cm³)
   - Porosity (derived)

   Highly Beneficial:
   - Organic matter/carbon content
   - Cation exchange capacity (CEC)
   - Soil depth to bedrock
   - Stone/gravel content

   Derived Hydraulic Properties:
   - Field capacity (θ at -33 kPa)
   - Wilting point (θ at -1500 kPa)
   - Saturated hydraulic conductivity (Ksat)
   - van Genuchten parameters (α, n)

2. VEGETATION INDICES (from GEE/Satellite)
   Essential:
   - NDVI (vegetation cover)
   - LAI (leaf area index)

   Beneficial:
   - EVI (enhanced vegetation index)
   - FPAR (photosynthetically active radiation)
   - Vegetation fraction cover
   - Crop type/land cover class

3. WEATHER DATA (from Open-Meteo or similar)
   Essential:
   - Precipitation (mm)
   - Temperature (min, max, mean)
   - Reference evapotranspiration (ET₀)

   Beneficial:
   - Solar radiation
   - Wind speed
   - Relative humidity
   - Vapor pressure deficit

4. TOPOGRAPHY (from SRTM/Copernicus DEM)
   Beneficial:
   - Elevation
   - Slope
   - Aspect
   - Topographic wetness index (TWI)
   - Flow accumulation

5. LAND SURFACE TEMPERATURE (from MODIS/Sentinel-3)
   Highly Beneficial:
   - Daytime LST
   - Nighttime LST
   - Diurnal temperature range
   - Thermal inertia (derived)

6. MICROWAVE DATA (from Sentinel-1/SMAP/SMOS)
   Optional but powerful:
   - SAR backscatter
   - SMAP/SMOS soil moisture (for calibration)
   - Radar vegetation index (RVI)

=== Data Source Priority for Africa ===

1. Soil: iSDA Africa (30m, Africa) > SoilGrids (250m, global)
2. Vegetation: MODIS (250m, 16-day) + Sentinel-2 (10m, 5-day)
3. Weather: Open-Meteo (free, global) or NASA POWER
4. LST: MODIS MOD11A2 (1km, 8-day)
5. Topography: SRTM 30m or Copernicus DEM
6. Validation: ISMN stations or SMAP 9km

=== Recommended Minimum Attribute Set ===

For basic soil moisture modeling:
- Sand%, Clay%, Bulk density → Hydraulic properties
- NDVI → Vegetation fraction, root depth
- Precipitation, Temperature, ET₀ → Water balance
- Elevation, Slope → Lateral flow (if needed)

For enhanced accuracy:
- Add: LAI, LST, organic carbon, TWI, land cover
"""


def print_attribute_guide():
    """Print the guide for soil moisture prediction attributes."""
    print(SOIL_MOISTURE_PREDICTION_ATTRIBUTES)


if __name__ == "__main__":
    print_attribute_guide()
