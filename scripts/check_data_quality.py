#!/usr/bin/env python3
"""
ISMN Station Data Quality Check and Data Source Demonstration.

This script:
1. Checks all ISMN stations for data quality
2. Shows the sensor depths at each station
3. Demonstrates data fetching from Open-Meteo, GEE, and soil sources
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import date

from smps.data.sources.ismn_loader import ISMNStationLoader
from smps.data.sources.weather import OpenMeteoSource
from smps.data.sources.base import DataFetchRequest
from smps.core.types import SiteID


def check_station_quality():
    """Check data quality for all ISMN stations."""
    print("=" * 80)
    print("ISMN STATION DATA QUALITY CHECK")
    print("=" * 80)

    loader = ISMNStationLoader(Path("data/ismn"))
    all_stations = loader.load_all_stations()

    results = []
    for station_id, station in all_stations.items():
        obs_data = station.daily_data
        if obs_data is None or len(obs_data) == 0:
            continue

        dates = pd.to_datetime(obs_data["date"])

        for depth in station.available_depths_cm:
            depth_data = obs_data[obs_data["depth_cm"] == depth]
            if len(depth_data) == 0:
                continue

            sm = depth_data["soil_moisture_mean"]
            valid_sm = sm.dropna()

            # Quality assessment
            quality = "GOOD"
            if len(valid_sm) < 100:
                quality = "LOW_DATA"
            elif valid_sm.min() < 0:
                quality = "NEGATIVE"
            elif valid_sm.max() > 0.65:
                quality = "TOO_HIGH"

            results.append({
                "network": station.network,
                "station": station.station,
                "lat": round(station.latitude, 4),
                "lon": round(station.longitude, 4),
                "depth_cm": depth,
                "start": dates.min().strftime("%Y-%m-%d"),
                "end": dates.max().strftime("%Y-%m-%d"),
                "n_days": len(depth_data),
                "valid_pct": round(100 * len(valid_sm) / max(1, len(depth_data)), 1),
                "sm_min": round(valid_sm.min(), 3) if len(valid_sm) > 0 else np.nan,
                "sm_max": round(valid_sm.max(), 3) if len(valid_sm) > 0 else np.nan,
                "sm_mean": round(valid_sm.mean(), 3) if len(valid_sm) > 0 else np.nan,
                "quality": quality
            })

    df = pd.DataFrame(results)
    print(f"\nTotal stations: {df['station'].nunique()}")
    print(f"Total sensor-depths: {len(df)}")

    # Summary by network
    print("\n" + "=" * 80)
    print("SUMMARY BY NETWORK")
    print("=" * 80)
    for network in sorted(df["network"].unique()):
        net_df = df[df["network"] == network]
        depths = sorted(net_df["depth_cm"].unique())
        print(f"\n{network}:")
        print(f"  Stations: {net_df['station'].nunique()}")
        print(f"  Sensor depths: {depths} cm")
        print(
            f"  Date coverage: {net_df['start'].min()} to {net_df['end'].max()}")
        print(f"  Total days: {net_df['n_days'].sum()}")
        print(f"  Valid data: {net_df['valid_pct'].mean():.1f}%")

    # Detailed table
    print("\n" + "=" * 80)
    print("STATION DETAILS")
    print("=" * 80)
    cols = ["network", "station", "depth_cm",
            "n_days", "valid_pct", "sm_mean", "quality"]
    print(df[cols].to_string(index=False))

    # Quality issues
    print("\n" + "=" * 80)
    print("DATA QUALITY FLAGS")
    print("=" * 80)
    issues = df[df["quality"] != "GOOD"]
    if len(issues) > 0:
        print(f"\nSensors with quality issues: {len(issues)}")
        print(issues[["station", "depth_cm", "n_days", "sm_min",
              "sm_max", "quality"]].to_string(index=False))
    else:
        print("\nAll sensors pass basic quality checks!")

    return df


def demonstrate_data_sources():
    """Demonstrate fetching data from all sources."""
    print("\n" + "=" * 80)
    print("DATA SOURCE DEMONSTRATION")
    print("=" * 80)

    # Load a sample station
    loader = ISMNStationLoader(Path("data/ismn"))
    all_stations = loader.load_all_stations()

    # Find a multi-depth station
    station = None
    for s in all_stations.values():
        if len(s.available_depths_cm) >= 4:
            station = s
            break

    if station is None:
        station = list(all_stations.values())[0]

    print(f"\nExample Station: {station.station_id}")
    print(f"Location: ({station.latitude:.4f}°N, {station.longitude:.4f}°E)")
    print(f"Depths: {station.available_depths_cm} cm")

    # 1. Open-Meteo Weather
    print("\n" + "-" * 40)
    print("1. OPEN-METEO WEATHER DATA")
    print("-" * 40)
    print("API Endpoints:")
    print("  Historical: https://archive-api.open-meteo.com/v1/era5")
    print("  Forecast:   https://api.open-meteo.com/v1/forecast")
    print("\nVariables fetched:")
    print("  - precipitation_sum (mm/day)")
    print("  - temperature_2m_max/min/mean (°C)")
    print("  - et0_fao_evapotranspiration_sum (mm/day)")
    print("  - shortwave_radiation_sum (MJ/m²)")
    print("  - wind_speed_10m_max (m/s)")

    try:
        weather_source = OpenMeteoSource(cache_dir=Path("data/cache/weather"))
        site_id = SiteID(station.station_id)
        request = DataFetchRequest(
            site_id=site_id,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 10),
            parameters={}
        )
        weather = weather_source._fetch_historical(
            request, station.latitude, station.longitude)
        print(f"\nSample fetch: {len(weather)} days")
        for w in weather[:3]:
            print(
                f"  {w.date}: P={w.precipitation_mm:.1f}mm, T={w.temperature_mean_c:.1f}°C, ET0={w.et0_mm:.1f}mm")
    except Exception as e:
        print(f"\nWeather fetch error: {e}")

    # 2. Google Earth Engine
    print("\n" + "-" * 40)
    print("2. GOOGLE EARTH ENGINE (VEGETATION)")
    print("-" * 40)
    print("Products:")
    print("  MODIS MOD13Q1 (NDVI): 250m, 16-day composite")
    print("  MODIS MCD15A3H (LAI): 500m, 4-day composite")
    print("\nMethods:")
    print("  GEESatelliteSource.fetch_ndvi(lat, lon, start, end)")
    print("  GEESatelliteSource.fetch_lai(lat, lon, start, end)")
    print("\nAuthentication: Service account required")
    print("  See: gee_satellite.py for implementation")

    # 3. Soil Data
    print("\n" + "-" * 40)
    print("3. SOIL DATA SOURCES")
    print("-" * 40)
    print("\niSDA Africa (isda_authenticated.py):")
    print("  URL: https://www.isda-africa.com/isdasoil/")
    print("  Coverage: African continent")
    print("  Depths: 0-20cm, 20-50cm")
    print("  Properties: Sand, Clay, Silt, SOC, Bulk Density")

    print("\nSoilGrids (soilgrids.py):")
    print("  URL: https://soilgrids.org/")
    print("  Coverage: Global 250m")
    print("  Depths: 0-5, 5-15, 15-30, 30-60, 60-100, 100-200cm")
    print("  Properties: Sand, Clay, Silt, SOC, Bulk Density")

    # 4. Data flow summary
    print("\n" + "=" * 80)
    print("SMPS DATA PIPELINE FLOW")
    print("=" * 80)
    print("""
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        INPUT DATA SOURCES                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  WEATHER (Open-Meteo)                                                   │
    │  ├─ ERA5 historical reanalysis (1940 - present)                        │
    │  ├─ GFS/ICON weather forecast (up to 16 days ahead)                    │
    │  └─ Variables: Precipitation, Temperature, ET0, Solar Radiation        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  VEGETATION (Google Earth Engine)                                       │
    │  ├─ MODIS NDVI: Vegetation greenness (16-day, 250m)                    │
    │  └─ MODIS LAI: Leaf Area Index (4-day, 500m)                           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  SOIL PROPERTIES (iSDA / SoilGrids)                                     │
    │  ├─ Texture: Sand, Clay, Silt fractions                                │
    │  ├─ Organic carbon content                                              │
    │  └─ Bulk density → Hydraulic properties                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  GROUND TRUTH (ISMN)                                                    │
    │  ├─ In-situ soil moisture observations                                  │
    │  ├─ Multiple sensor depths per station                                  │
    │  └─ Networks: TAHMO (Africa), AMMA-CATCH (West Africa)                  │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    PHYSICS MODEL (EnhancedWaterBalance)                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Layers: [0-10cm, 10-30cm, 30-50cm, 50-75cm, 75-100cm]                 │
    │  Daily timestep with hourly-equivalent physics                          │
    │  Processes: Infiltration, ET, Drainage, Root uptake                    │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         OUTPUT PREDICTIONS                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Soil moisture at each layer (m³/m³)                                    │
    │  Time horizons: Same-day, 24h, 72h, 168h ahead                         │
    │  Validation metrics: RMSE, KGE, NSE, R² by depth                       │
    └─────────────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    # Run quality check
    quality_df = check_station_quality()

    # Demonstrate data sources
    demonstrate_data_sources()

    # Save quality report
    output_dir = Path("results/ismn_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_df.to_csv(output_dir / "station_quality_report.csv", index=False)
    print(
        f"\nQuality report saved to: {output_dir / 'station_quality_report.csv'}")
