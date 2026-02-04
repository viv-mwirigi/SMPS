"""
Universal constants and irrigation thresholds for SWPPS.

The key advantage of using matric potential is that these thresholds
are UNIVERSAL - they work for any soil type without calibration.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


# =============================================================================
# IRRIGATION ACTION TYPES
# =============================================================================

class IrrigationAction(Enum):
    """Possible irrigation actions."""

    NO_ACTION = "no_action"       # No irrigation needed
    MONITOR = "monitor"           # Continue monitoring, getting close
    SCHEDULE = "schedule"         # Schedule irrigation for future
    IRRIGATE_NOW = "irrigate_now"  # Irrigate immediately


# =============================================================================
# MATRIC POTENTIAL REFERENCE SCALE (kPa)
# =============================================================================
# These are universal thresholds that apply to ALL soil types!
# This is the fundamental advantage over VWC-based systems.

@dataclass(frozen=True)
class MatricPotentialRange:
    """Universal matric potential ranges for plant-water relations."""

    # Saturation and field capacity
    SATURATION: float = 0.0              # Fully saturated
    AIR_ENTRY: float = -2.0              # Air begins entering pores
    VERY_WET: float = -5.0               # Very wet, may cause root issues
    FIELD_CAPACITY: float = -33.0        # Maximum held against gravity

    # Optimal range for most crops
    OPTIMAL_UPPER: float = -33.0         # Upper optimal bound
    OPTIMAL_MIDDLE: float = -60.0        # Middle of optimal range
    OPTIMAL_LOWER: float = -100.0        # Lower optimal bound

    # Stress thresholds
    STRESS_ONSET: float = -200.0         # Beginning of stress
    MODERATE_STRESS: float = -500.0      # Significant stress
    SEVERE_STRESS: float = -1000.0       # Severe stress
    PERMANENT_WILTING: float = -1500.0   # Plants cannot extract water

    # Extreme values
    AIR_DRY: float = -10000.0           # Air-dry soil
    OVEN_DRY: float = -1000000.0        # Oven-dry soil (theoretical)


# =============================================================================
# UNIVERSAL IRRIGATION THRESHOLDS
# =============================================================================

IRRIGATION_THRESHOLDS: Dict[str, float] = {
    # Status boundaries (kPa)
    "saturation": 0.0,
    "field_capacity_kpa": -33.0,
    "optimal_upper_kpa": -33.0,
    "optimal_lower_kpa": -100.0,
    "stress_threshold_kpa": -200.0,
    "moderate_stress_kpa": -500.0,
    "severe_stress_kpa": -1000.0,
    "wilting_point_kpa": -1500.0,

    # Decision thresholds
    "too_wet_kpa": -10.0,            # Below this, don't irrigate
    "refill_target_kpa": -25.0,      # Target after irrigation
    "irrigate_below_kpa": -80.0,     # Default irrigation trigger
}


# =============================================================================
# CROP-SPECIFIC THRESHOLDS
# =============================================================================
# Different crops have different tolerance to water stress.
# These thresholds determine when to irrigate for each crop.
# Keys: irrigate_below_kpa, refill_to_kpa, stress_threshold_kpa

CROP_THRESHOLDS: Dict[str, Dict[str, float]] = {
    # High water requirement crops
    "lettuce": {
        "irrigate_below_kpa": -40.0,
        "refill_to_kpa": -15.0,
        "stress_threshold_kpa": -80.0,
        "description": "Shallow roots, sensitive to water stress",
    },
    "spinach": {
        "irrigate_below_kpa": -40.0,
        "refill_to_kpa": -15.0,
        "stress_threshold_kpa": -80.0,
        "description": "Leafy green, needs consistent moisture",
    },
    "celery": {
        "irrigate_below_kpa": -30.0,
        "refill_to_kpa": -10.0,
        "stress_threshold_kpa": -60.0,
        "description": "Very sensitive to water stress",
    },

    # Moderate water requirement
    "tomato": {
        "irrigate_below_kpa": -80.0,
        "refill_to_kpa": -25.0,
        "stress_threshold_kpa": -150.0,
        "description": "Moderate tolerance, deeper roots",
    },
    "pepper": {
        "irrigate_below_kpa": -70.0,
        "refill_to_kpa": -25.0,
        "stress_threshold_kpa": -120.0,
        "description": "Moderate water needs",
    },
    "eggplant": {
        "irrigate_below_kpa": -80.0,
        "refill_to_kpa": -25.0,
        "stress_threshold_kpa": -150.0,
        "description": "Similar to tomato",
    },
    "cucumber": {
        "irrigate_below_kpa": -50.0,
        "refill_to_kpa": -20.0,
        "stress_threshold_kpa": -100.0,
        "description": "Shallow roots, moderate needs",
    },

    # Lower water requirement / drought tolerant
    "maize": {
        "irrigate_below_kpa": -100.0,
        "refill_to_kpa": -30.0,
        "stress_threshold_kpa": -300.0,
        "description": "Deeper roots, tolerates some stress",
    },
    "sorghum": {
        "irrigate_below_kpa": -150.0,
        "refill_to_kpa": -40.0,
        "stress_threshold_kpa": -500.0,
        "description": "Drought tolerant",
    },
    "wheat": {
        "irrigate_below_kpa": -100.0,
        "refill_to_kpa": -30.0,
        "stress_threshold_kpa": -300.0,
        "description": "Moderate drought tolerance",
    },
    "rice": {
        "irrigate_below_kpa": -20.0,
        "refill_to_kpa": -5.0,
        "stress_threshold_kpa": -40.0,
        "description": "Requires flooded or saturated conditions",
    },

    # Root crops
    "potato": {
        "irrigate_below_kpa": -60.0,
        "refill_to_kpa": -20.0,
        "stress_threshold_kpa": -100.0,
        "description": "Sensitive to stress during tuber formation",
    },
    "carrot": {
        "irrigate_below_kpa": -70.0,
        "refill_to_kpa": -25.0,
        "stress_threshold_kpa": -120.0,
        "description": "Moderate needs, consistent moisture",
    },
    "onion": {
        "irrigate_below_kpa": -60.0,
        "refill_to_kpa": -20.0,
        "stress_threshold_kpa": -100.0,
        "description": "Shallow roots",
    },

    # Tree crops
    "citrus": {
        "irrigate_below_kpa": -80.0,
        "refill_to_kpa": -25.0,
        "stress_threshold_kpa": -200.0,
        "description": "Deep roots, moderate drought tolerance",
    },
    "mango": {
        "irrigate_below_kpa": -100.0,
        "refill_to_kpa": -30.0,
        "stress_threshold_kpa": -300.0,
        "description": "Tolerates some drought",
    },
    "avocado": {
        "irrigate_below_kpa": -50.0,
        "refill_to_kpa": -20.0,
        "stress_threshold_kpa": -100.0,
        "description": "Shallow roots, sensitive",
    },

    # Legumes
    "bean": {
        "irrigate_below_kpa": -70.0,
        "refill_to_kpa": -25.0,
        "stress_threshold_kpa": -150.0,
        "description": "Moderate water needs",
    },
    "groundnut": {
        "irrigate_below_kpa": -100.0,
        "refill_to_kpa": -30.0,
        "stress_threshold_kpa": -300.0,
        "description": "Some drought tolerance",
    },
    "cowpea": {
        "irrigate_below_kpa": -150.0,
        "refill_to_kpa": -40.0,
        "stress_threshold_kpa": -400.0,
        "description": "Good drought tolerance",
    },

    # Generic/default for unknown crops
    "generic": {
        "irrigate_below_kpa": -80.0,
        "refill_to_kpa": -25.0,
        "stress_threshold_kpa": -200.0,
        "description": "Conservative default values",
    },
}


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

PHYSICAL_CONSTANTS = {
    # Water properties
    "water_density_kg_m3": 1000.0,
    "water_specific_heat_j_kg_k": 4186.0,
    "latent_heat_vaporization_mj_kg": 2.45,

    # Air properties
    "psychrometric_constant_kpa_c": 0.0665,  # At sea level

    # Soil-water constants
    "gravity_m_s2": 9.81,

    # Conversions
    "kpa_to_m_water": 0.102,  # 1 kPa = 0.102 m water head
    "cbar_to_kpa": 1.0,       # 1 cbar = 1 kPa
    "bar_to_kpa": 100.0,      # 1 bar = 100 kPa
    "psi_to_kpa": 6.895,      # 1 psi = 6.895 kPa
}


# =============================================================================
# WEATHER API CONFIGURATION
# =============================================================================

OPENMETEO_CONFIG = {
    "historical_url": "https://archive-api.open-meteo.com/v1/era5",
    "forecast_url": "https://api.open-meteo.com/v1/forecast",
    "max_retries": 3,  # Reduced from 5
    "backoff_seconds": 5.0,  # Increased from 2.0
    "timeout_seconds": 30,
    "min_request_interval": 3.0,  # Increased from 2.0

    # Variables to fetch
    "daily_variables": [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "et0_fao_evapotranspiration",
        "shortwave_radiation_sum",
        "relative_humidity_2m_mean",
        "wind_speed_10m_mean",
    ],
    "hourly_variables": [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "et0_fao_evapotranspiration",
        "soil_moisture_0_to_7cm",
        "soil_temperature_0_to_7cm",
    ],
}


# =============================================================================
# MODEL CONFIGURATION DEFAULTS
# =============================================================================

MODEL_DEFAULTS = {
    # Physics model
    "n_layers": 3,
    "max_depth_m": 1.0,
    "warmup_days": 30,

    # ML model
    "ml_model_type": "lightgbm",
    "n_estimators": 2000,
    "early_stopping_rounds": 100,
    "learning_rate": 0.015,
    "max_depth": 8,

    # Forecast horizons (hours)
    "forecast_horizons": [0, 6, 24, 72, 168],

    # Feature engineering
    "lag_days": [1, 3, 7, 14],
    "rolling_windows": [7, 14, 30],

    # Training
    "validation_fraction": 0.15,
    "min_training_samples": 30,
    "retrain_interval_days": 7,
}


# =============================================================================
# SENSOR CONFIGURATION
# =============================================================================

SENSOR_TYPES = {
    "watermark": {
        "unit": "cbar",
        "range": (0, 200),
        "accuracy_cbar": 2,
        "description": "Granular matrix sensor (Irrometer)",
    },
    "tensiometer": {
        "unit": "cbar",
        "range": (0, 85),
        "accuracy_cbar": 1,
        "description": "Water-filled tube with ceramic tip",
    },
    "mps6": {
        "unit": "kPa",
        "range": (-100000, 0),
        "accuracy_kpa": 5,
        "description": "METER MPS-6 dielectric water potential sensor",
    },
    "teros21": {
        "unit": "kPa",
        "range": (-100000, 0),
        "accuracy_kpa": 5,
        "description": "METER TEROS 21 water potential sensor",
    },
    "capacitive": {
        "unit": "percent",
        "range": (0, 100),
        "description": "Volumetric water content sensor (needs conversion)",
        "needs_conversion": True,
    },
}
