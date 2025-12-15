"""
SMPS Data Package.

Provides data contracts, sources, and quality assessment for soil moisture prediction.
"""

from smps.data.contracts import (
    DailyWeather,
    SoilProfile,
    RemoteSensingData,
    SatelliteObservation,
    SoilMoistureObservation,
    CanonicalDailyRow,
    PhysicsPriorRecord,
)

__all__ = [
    "DailyWeather",
    "SoilProfile",
    "RemoteSensingData",
    "SatelliteObservation",
    "SoilMoistureObservation",
    "CanonicalDailyRow",
    "PhysicsPriorRecord",
]
