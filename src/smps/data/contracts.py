"""
Data contracts and schemas for the Smps system.
Ensures data consistency and provides validation.
"""
from datetime import date, datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import pandas as pd
import numpy as np

from smps.core.types import (
    SiteID, SoilMoistureVWC, SoilTensionKPa, 
    PrecipitationMm, ET0Mm, DataQualityFlag
)


class TimePeriod(str, Enum):
    """Time periods for data aggregation"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    SEASON = "season"


class SensorType(str, Enum):
    """Types of soil moisture sensors"""
    TENSION = "tension"  # Watermark, etc.
    VOLUMETRIC = "volumetric"  # TDR, FDR
    CAPACITANCE = "capacitance"
    RESISTANCE = "resistance"


class IrrigationMethod(str, Enum):
    """Irrigation methods"""
    DRIP = "drip"
    SPRINKLER = "sprinkler"
    FLOOD = "flood"
    PIVOT = "pivot"
    MANUAL = "manual"


class RawObservation(BaseModel):
    """Raw observation from sensor or API"""
    timestamp: datetime
    value: float
    site_id: SiteID
    sensor_id: str
    sensor_type: Optional[SensorType] = None
    depth_cm: Optional[int] = None
    unit: str
    quality_flag: DataQualityFlag = DataQualityFlag.OK
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "forbid"  # Strict validation


class DailyWeather(BaseModel):
    """Daily weather data for canonical table"""
    date: date
    site_id: SiteID
    
    # Core measurements
    precipitation_mm: PrecipitationMm = Field(ge=0)
    et0_mm: ET0Mm = Field(ge=0)
    temperature_mean_c: float
    temperature_min_c: float
    temperature_max_c: float
    solar_radiation_mj_m2: float = Field(ge=0)
    relative_humidity_mean: float = Field(ge=0, le=100)
    wind_speed_mean_m_s: float = Field(ge=0)
    vapor_pressure_deficit_kpa: Optional[float] = Field(ge=0)
    
    # Derived/optional
    dewpoint_c: Optional[float] = None
    atmospheric_pressure_hpa: Optional[float] = None
    soil_temperature_c: Optional[float] = None
    growing_degree_days: Optional[float] = None
    
    # Quality and metadata
    source: str
    is_forecast: bool = False
    forecast_horizon_days: Optional[int] = None
    quality_score: float = Field(default=1.0, ge=0, le=1)
    
    @validator('temperature_max_c')
    def validate_temperature_range(cls, v, values):
        """Ensure temperature max >= min"""
        if 'temperature_min_c' in values and v < values['temperature_min_c']:
            raise ValueError('temperature_max must be >= temperature_min')
        return v


class SoilProfile(BaseModel):
    """Soil profile information"""
    site_id: SiteID
    
    # Texture (USDA triangle)
    sand_percent: float = Field(ge=0, le=100)
    silt_percent: float = Field(ge=0, le=100)
    clay_percent: float = Field(ge=0, le=100)
    
    # Hydraulic properties
    porosity: float = Field(ge=0, le=1)
    field_capacity: float = Field(ge=0, le=1)
    wilting_point: float = Field(ge=0, le=1)
    saturated_hydraulic_conductivity_cm_day: float = Field(ge=0)
    
    # Depth information
    profile_depth_cm: float = Field(gt=0)
    effective_rooting_depth_cm: Optional[float] = None
    
    # Optional: Van Genuchten parameters
    van_genuchten_alpha: Optional[float] = Field(gt=0)
    van_genuchten_n: Optional[float] = Field(gt=1)
    
    # Source and quality
    source: str
    confidence: float = Field(default=1.0, ge=0, le=1)
    
    @root_validator
    def validate_texture_sum(cls, values):
        """Ensure texture percentages sum to 100% (± tolerance)"""
        sand = values.get('sand_percent', 0)
        silt = values.get('silt_percent', 0)
        clay = values.get('clay_percent', 0)
        
        total = sand + silt + clay
        if abs(total - 100) > 1:  # 1% tolerance
            raise ValueError(f'Texture percentages must sum to 100% (got {total})')
        
        return values
    
    @validator('field_capacity')
    def validate_field_capacity_range(cls, v, values):
        """Ensure field capacity is between wilting point and porosity"""
        if 'wilting_point' in values and v <= values['wilting_point']:
            raise ValueError('field_capacity must be > wilting_point')
        if 'porosity' in values and v >= values['porosity']:
            raise ValueError('field_capacity must be < porosity')
        return v


class RemoteSensingData(BaseModel):
    """Remote sensing data for a site"""
    date: date
    site_id: SiteID
    
    # Optical (Sentinel-2)
    ndvi: Optional[float] = Field(ge=-1, le=1)
    evi: Optional[float] = None
    lai: Optional[float] = Field(ge=0)
    
    # SAR (Sentinel-1)
    sar_vv_db: Optional[float] = None
    sar_vh_db: Optional[float] = None
    sar_vv_vh_ratio: Optional[float] = None
    
    # Derived metrics
    ndvi_anomaly: Optional[float] = None  # vs site climatology
    ndvi_slope_14d: Optional[float] = None  # trend
    
    # Quality
    cloud_cover_percent: Optional[float] = Field(ge=0, le=100)
    data_mask_percent: Optional[float] = Field(ge=0, le=100)
    quality_flag: DataQualityFlag = DataQualityFlag.OK
    
    class Config:
        extra = "allow"  # Allow additional bands if needed


class IrrigationRecord(BaseModel):
    """Record of irrigation event"""
    timestamp: datetime
    site_id: SiteID
    volume_mm: float = Field(ge=0)
    duration_minutes: Optional[float] = Field(gt=0)
    method: Optional[IrrigationMethod] = None
    efficiency_factor: float = Field(default=0.85, ge=0, le=1)
    source: str  # 'sensor', 'farmer_log', 'inferred'
    confidence: float = Field(default=1.0, ge=0, le=1)
    
    @validator('timestamp')
    def validate_not_future(cls, v):
        """Ensure timestamp is not in the future"""
        if v > datetime.now():
            raise ValueError('Irrigation timestamp cannot be in the future')
        return v


class PhysicsPriorRecord(BaseModel):
    """Output from the two-bucket physics model"""
    date: date
    site_id: SiteID
    
    # Soil moisture states
    theta_surface: SoilMoistureVWC = Field(ge=0, le=1)
    theta_root: SoilMoistureVWC = Field(ge=0, le=1)
    
    # Water fluxes (mm/day)
    precipitation: PrecipitationMm = Field(ge=0)
    irrigation: PrecipitationMm = Field(ge=0)
    evaporation: PrecipitationMm = Field(ge=0)
    transpiration: PrecipitationMm = Field(ge=0)
    drainage: PrecipitationMm = Field(ge=0)
    runoff: PrecipitationMm = Field(ge=0)
    infiltration: PrecipitationMm = Field(ge=0)
    
    # Balance check
    water_balance_error_mm: float
    water_balance_closure_percent: float = Field(ge=0, le=100)
    
    # Model state
    converged: bool = True
    iterations: Optional[int] = None
    soil_water_storage_mm: float = Field(ge=0)
    
    # Metadata
    physics_model_version: str
    parameter_set_hash: str
    
    @root_validator
    def validate_water_balance(cls, values):
        """Validate water balance closure"""
        error = values.get('water_balance_error_mm', 0)
        if abs(error) > 1.0:  # More than 1 mm error is suspicious
            closure = values.get('water_balance_closure_percent', 0)
            if closure < 95:  # Less than 95% closure
                raise ValueError(f'Poor water balance closure: {closure}%')
        return values


class SoilMoistureObservation(BaseModel):
    """In-situ soil moisture observation"""
    timestamp: datetime
    site_id: SiteID
    depth_cm: int = Field(gt=0)
    
    # Measurements
    vwc: Optional[SoilMoistureVWC] = Field(ge=0, le=1)
    tension_kpa: Optional[SoilTensionKPa] = Field(ge=0)
    temperature_c: Optional[float] = None
    
    # Derived
    available_water_capacity: Optional[float] = Field(ge=0, le=1)
    saturation_percent: Optional[float] = Field(ge=0, le=100)
    
    # Quality
    sensor_type: SensorType
    calibration_date: Optional[date] = None
    quality_flag: DataQualityFlag = DataQualityFlag.OK
    confidence: float = Field(default=1.0, ge=0, le=1)
    
    # Metadata
    sensor_id: str
    installation_depth_cm: Optional[float] = None
    
    @root_validator
    def validate_measurement_present(cls, values):
        """Ensure at least one measurement type is present"""
        vwc = values.get('vwc')
        tension = values.get('tension_kpa')
        
        if vwc is None and tension is None:
            raise ValueError('Either vwc or tension must be provided')
        
        return values
    
    @validator('vwc', 'tension_kpa')
    def validate_not_both_present(cls, v, values, field):
        """Don't allow both vwc and tension to be provided"""
        if field.name == 'vwc' and v is not None:
            if values.get('tension_kpa') is not None:
                raise ValueError('Provide either vwc or tension, not both')
        return v


class CanonicalDailyRow(BaseModel):
    """
    Single row of the canonical daily table.
    This is the unified data representation for modeling.
    """
    # Primary keys
    site_id: SiteID
    date: date
    
    # Site metadata
    latitude: float
    longitude: float
    elevation_m: Optional[float]
    crop_type: Optional[str]
    growing_season_day: Optional[int] = Field(ge=0)
    
    # Weather - current day
    precipitation_mm: PrecipitationMm = Field(ge=0)
    et0_mm: ET0Mm = Field(ge=0)
    temperature_mean_c: float
    temperature_min_c: float
    temperature_max_c: float
    solar_radiation_mj_m2: float = Field(ge=0)
    relative_humidity_mean: float = Field(ge=0, le=100)
    wind_speed_mean_m_s: float = Field(ge=0)
    vapor_pressure_deficit_kpa: Optional[float]
    
    # Weather - cumulative
    precip_cumulative_3d: PrecipitationMm = Field(ge=0)
    precip_cumulative_7d: PrecipitationMm = Field(ge=0)
    precip_cumulative_14d: PrecipitationMm = Field(ge=0)
    precip_cumulative_30d: PrecipitationMm = Field(ge=0)
    
    et0_cumulative_3d: ET0Mm = Field(ge=0)
    et0_cumulative_7d: ET0Mm = Field(ge=0)
    et0_cumulative_14d: ET0Mm = Field(ge=0)
    
    # Soil properties
    sand_percent: float = Field(ge=0, le=100)
    silt_percent: float = Field(ge=0, le=100)
    clay_percent: float = Field(ge=0, le=100)
    porosity: float = Field(ge=0, le=1)
    field_capacity: float = Field(ge=0, le=1)
    wilting_point: float = Field(ge=0, le=1)
    
    # Remote sensing
    ndvi: Optional[float] = Field(ge=-1, le=1)
    ndvi_anomaly: Optional[float]
    sar_vv_db: Optional[float]
    
    # Observations (if available)
    obs_vwc_surface: Optional[SoilMoistureVWC] = Field(ge=0, le=1)
    obs_vwc_root: Optional[SoilMoistureVWC] = Field(ge=0, le=1)
    obs_quality_surface: DataQualityFlag = DataQualityFlag.OK
    obs_quality_root: DataQualityFlag = DataQualityFlag.OK
    
    # Irrigation
    irrigation_mm: PrecipitationMm = Field(default=0, ge=0)
    days_since_irrigation: Optional[int] = Field(ge=0)
    irrigation_flag: bool = False
    
    # Physics prior
    physics_theta_surface: SoilMoistureVWC = Field(ge=0, le=1)
    physics_theta_root: SoilMoistureVWC = Field(ge=0, le=1)
    physics_residual_surface: Optional[float]
    physics_residual_root: Optional[float]
    
    # Temporal features
    day_of_year: int = Field(ge=1, le=366)
    day_of_year_sin: float = Field(ge=-1, le=1)
    day_of_year_cos: float = Field(ge=-1, le=1)
    month: int = Field(ge=1, le=12)
    season: str  # 'winter', 'spring', 'summer', 'fall'
    
    # Lagged soil moisture (most important features)
    vwc_surface_lag_1d: Optional[SoilMoistureVWC] = Field(ge=0, le=1)
    vwc_surface_lag_3d: Optional[SoilMoistureVWC] = Field(ge=0, le=1)
    vwc_surface_lag_7d: Optional[SoilMoistureVWC] = Field(ge=0, le=1)
    vwc_surface_lag_14d: Optional[SoilMoistureVWC] = Field(ge=0, le=1)
    vwc_surface_lag_30d: Optional[SoilMoistureVWC] = Field(ge=0, le=1)
    
    vwc_root_lag_1d: Optional[SoilMoistureVWC] = Field(ge=0, le=1)
    vwc_root_lag_3d: Optional[SoilMoistureVWC] = Field(ge=0, le=1)
    vwc_root_lag_7d: Optional[SoilMoistureVWC] = Field(ge=0, le=1)
    vwc_root_lag_14d: Optional[SoilMoistureVWC] = Field(ge=0, le=1)
    vwc_root_lag_30d: Optional[SoilMoistureVWC] = Field(ge=0, le=1)
    
    # Rolling statistics
    vwc_surface_rolling_mean_7d: Optional[SoilMoistureVWC] = Field(ge=0, le=1)
    vwc_surface_rolling_std_7d: Optional[float] = Field(ge=0)
    vwc_root_rolling_mean_7d: Optional[SoilMoistureVWC] = Field(ge=0, le=1)
    vwc_root_rolling_std_7d: Optional[float] = Field(ge=0)
    
    # Water balance features
    water_deficit_mm: float
    available_water_capacity: float = Field(ge=0, le=1)
    
    # Quality flags
    data_coverage_percent: float = Field(default=1.0, ge=0, le=1)
    weather_quality_score: float = Field(default=1.0, ge=0, le=1)
    remote_sensing_quality_score: float = Field(default=1.0, ge=0, le=1)
    physics_prior_quality_score: float = Field(default=1.0, ge=0, le=1)
    
    # Metadata
    created_at: datetime
    etl_version: str
    data_sources: List[str]
    
    class Config:
        extra = "forbid"  # Strict validation - no extra fields
    
    @validator('vwc_surface_lag_*', 'vwc_root_lag_*', pre=True)
    def validate_lagged_values(cls, v):
        """Handle missing lagged values gracefully"""
        if v is None or pd.isna(v):
            return None
        return float(v)
    
    @root_validator
    def calculate_water_deficit(cls, values):
        """Calculate water deficit based on field capacity and current moisture"""
        fc = values.get('field_capacity', 0.3)
        wp = values.get('wilting_point', 0.1)
        current = values.get('physics_theta_root', 0.2)
        
        if current < wp:
            deficit = 0  # Below wilting point
        elif current > fc:
            deficit = 0  # Above field capacity
        else:
            # Linear between wp and fc
            deficit = (fc - current) / (fc - wp)
        
        values['water_deficit_mm'] = deficit
        values['available_water_capacity'] = max(0, current - wp) / (fc - wp)
        
        return values
    
    @validator('season')
    def validate_season(cls, v, values):
        """Validate season based on month"""
        month = values.get('month')
        if month:
            season_map = {
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'fall', 10: 'fall', 11: 'fall'
            }
            expected = season_map.get(month)
            if v != expected:
                raise ValueError(f'Season {v} does not match month {month}')
        return v


# Utility functions for working with data contracts
def dataframe_to_canonical(df: pd.DataFrame) -> List[CanonicalDailyRow]:
    """Convert DataFrame to list of validated canonical rows"""
    rows = []
    errors = []
    
    for idx, row in df.iterrows():
        try:
            # Convert row to dict and validate
            row_dict = row.to_dict()
            canonical_row = CanonicalDailyRow(**row_dict)
            rows.append(canonical_row)
        except Exception as e:
            errors.append({
                'index': idx,
                'row': row.to_dict(),
                'error': str(e)
            })
    
    if errors:
        error_msg = f"Validation failed for {len(errors)} rows"
        if len(errors) < 5:
            error_msg += f": {errors}"
        
        # Log errors but continue (partial data is better than no data)
        import logging
        logging.warning(error_msg)
    
    return rows


def canonical_to_dataframe(rows: List[CanonicalDailyRow]) -> pd.DataFrame:
    """Convert canonical rows to DataFrame"""
    data = [row.dict() for row in rows]
    return pd.DataFrame(data)


class DataContractError(Exception):
    """Raised when data contract validation fails"""
    pass