"""
Physical constants, default values, and system-wide constants.
"""
import numpy as np
from typing import Dict, Final, Tuple, List

# Physical constants
WATER_DENSITY: Final[float] = 1000.0  # kg/m³
GRAVITY: Final[float] = 9.81  # m/s²
LATENT_HEAT_VAPORIZATION: Final[float] = 2.45e6  # J/kg

#
# System constants
MAX_SOIL_MOISTURE: Final[float] = 0.5  # m³/m³ (saturation)
MIN_SOIL_MOISTURE: Final[float] = 0.01  # m³/m³ (residual)
MAX_SOIL_TENSION: Final[float] = 1500.0  # kPa (oven dry)
MIN_SOIL_TENSION: Final[float] = 0.0  # kPa (saturation)

# Numerical stability
EPSILON: Final[float] = 1e-10


# Data source parameters
DATA_SOURCE_CONFIGS = {
    "isdasoil": {
        "base_url": "https://api.isda-africa.com",
        "properties": ["clay", "sand", "silt", "bdod", "carbon_organic"],
        "depths": ["0-20cm", "20-50cm"],
        "rate_limit_per_hour": 600,
    },
    "era5": {
        "variables": ["total_precipitation", "2m_temperature", "surface_solar_radiation_downwards"],
        "resolution_km": 31,
        "temporal_resolution": "hourly",
    },
    "grafs": {
        "resolution_km": 10,
        "depths": ["0-5cm", "0-100cm"],
        "temporal_resolution": "daily",
    }
}

# Default model parameters
DEFAULT_LIGHTGBM_PARAMS: Final[Dict] = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 8,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "verbosity": -1,
    "seed": 42
}

# Uncertainty quantification constants
UNCERTAINTY_CONSTANTS: Final[Dict[str, float]] = {
    "default_coverage": 0.9,
    "min_ensemble_members": 3,
    "max_ensemble_members": 20,
    "calibration_min_samples": 100,
    "prediction_interval_tolerance": 0.05,  # Allowed deviation from nominal coverage
    "drift_detection_window": 30  # Days for detecting prediction drift
}

# Error thresholds for quality flags
QUALITY_THRESHOLDS: Final[Dict[str, float]] = {
    "rmse_good": 0.05,  # m³/m³
    "rmse_acceptable": 0.08,
    "rmse_poor": 0.12,
    "bias_good": 0.02,
    "bias_acceptable": 0.05,
    "correlation_good": 0.8,
    "correlation_acceptable": 0.6
}

# Remote sensing constants
SENTINEL_BANDS: Final[Dict[str, Tuple[float, float]]] = {
    "B2": (0.458, 0.523),  # Blue
    "B3": (0.543, 0.578),  # Green
    "B4": (0.650, 0.680),  # Red
    "B8": (0.785, 0.900),  # NIR
    "B11": (1.565, 1.655),  # SWIR1
    "B12": (2.100, 2.280)   # SWIR2
}

# Vegetation indices formulas
VEGETATION_INDICES: Final[Dict[str, str]] = {
    "NDVI": "(B8 - B4) / (B8 + B4)",
    "EVI": "2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1)",
    "NDWI": "(B3 - B8) / (B3 + B8)",
    "MSAVI": "(2 * B8 + 1 - sqrt((2 * B8 + 1)^2 - 8 * (B8 - B4))) / 2"
}

# SAR backscatter ranges (dB)
SENTINEL1_BACKSCATTER_RANGES: Final[Dict[str, Tuple[float, float]]] = {
    "VV_dry": (-20, -10),
    "VV_moist": (-15, -5),
    "VV_wet": (-10, 0),
    "VH_dry": (-25, -15),
    "VH_moist": (-20, -10),
    "VH_wet": (-15, -5)
}

# Unit conversion factors
UNIT_CONVERSIONS: Final[Dict[str, float]] = {
    "mm_to_m": 0.001,
    "m_to_mm": 1000.0,
    "celsius_to_kelvin": 273.15,
    "kPa_to_bar": 0.01,
    "bar_to_kPa": 100.0,
    "joule_to_calorie": 0.239006,
    "hectare_to_sqm": 10000.0,
    "inch_to_mm": 25.4,
    "mm_to_inch": 0.0393701,
    "acre_to_hectare": 0.404686
}
# Feature names (for consistency)
FEATURE_GROUPS: Final[Dict[str, List[str]]] = {
    "physics": ["theta_phys_surface", "theta_phys_root", "physics_residual"],
    "temporal": ["soil_moisture_lag_1d", "soil_moisture_lag_7d", 
                 "soil_moisture_rolling_mean_7d", "soil_moisture_rolling_std_7d"],
    "meteorological": ["precip_1d_sum", "precip_7d_sum", "et0_1d_sum", "et0_7d_sum",
                       "temperature_mean_7d", "vapor_pressure_deficit_mean_7d"],
    "remote_sensing": ["ndvi", "ndvi_anomaly", "sar_vv_mean", "sar_vh_mean"],
    "static": ["sand_percent", "silt_percent", "clay_percent", 
               "field_capacity", "wilting_point", "elevation", "slope"]
}