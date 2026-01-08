#!/usr/bin/env python3
"""
Validate SMPS model against ISMN in-situ soil moisture observations.

Uses TAHMO network stations in Kenya with data from Jan 2020 - Dec 2021.
"""

from smps.core.types import SoilParameters
from smps.data.sources.base import DataFetchRequest
from smps.data.sources.satellite import MODISNDVISource
from smps.data.sources.soil import MockSoilSource
from smps.data.sources.isda_authenticated import IsdaAfricaAuthenticatedSource
from smps.data.sources.weather import OpenMeteoSource
from smps.physics.pedotransfer import estimate_soil_parameters_saxton
from smps.physics import (
    EnhancedWaterBalance, EnhancedModelParameters,
    create_water_balance_model,
)
import logging
import warnings
import time
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

warnings.filterwarnings('ignore')


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def load_ismn_data():
    """Load ISMN observations and site metadata."""
    obs_df = pd.read_csv('data/ismn/ismn_daily_observations.csv')
    obs_df['date'] = pd.to_datetime(obs_df['date'])

    meta_df = pd.read_csv('data/ismn/ismn_site_metadata.csv')
    return obs_df, meta_df


def fetch_soil_data(site_id: str, lat: float, lon: float):
    """Fetch soil data from iSDA or fallback to mock"""
    try:
        isda = IsdaAfricaAuthenticatedSource()
        profile = isda.fetch_soil_profile(site_id, latitude=lat, longitude=lon)
        # Convert SoilProfile object to dictionary
        return {
            'sand': profile.sand_percent,
            'clay': profile.clay_percent,
            'silt': profile.silt_percent,
            'porosity': profile.porosity,
            'field_capacity': profile.field_capacity,
            'wilting_point': profile.wilting_point,
            'organic_carbon': 1.5,  # Default value
            'bulk_density': 1.3,  # Default value
        }
    except Exception as e:
        logger.warning(f"iSDA failed for {site_id}: {e}")
        try:
            mock = MockSoilSource()
            profile = mock.fetch_soil_profile(site_id)
            return {
                'sand': profile.sand_percent,
                'clay': profile.clay_percent,
                'silt': profile.silt_percent,
                'porosity': profile.porosity,
                'field_capacity': profile.field_capacity,
                'wilting_point': profile.wilting_point,
                'organic_carbon': 1.5,
                'bulk_density': 1.3,
            }
        except:
            # Fallback defaults for Kenya (loamy soils)
            return {
                'sand': 40,
                'clay': 30,
                'silt': 30,
                'porosity': 0.45,
                'field_capacity': 0.30,
                'wilting_point': 0.12,
                'organic_carbon': 1.5,
                'bulk_density': 1.3,
            }


def fetch_weather_data(site_id: str, lat: float, lon: float,
                       start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch weather data from Open-Meteo"""
    weather_source = OpenMeteoSource()

    request = DataFetchRequest(
        site_id=site_id,
        start_date=start_date,
        end_date=end_date,
        parameters={"include_forecast": False}
    )

    weather_source._get_site_coordinates = lambda s: (lat, lon)

    try:
        data = weather_source.fetch_daily_weather(request)
        df = pd.DataFrame([d.model_dump() for d in data])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        return df
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
        return pd.DataFrame()


def fetch_ndvi_data(site_id: str, lat: float, lon: float,
                    start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch NDVI data from MODIS/Sentinel source"""
    ndvi_source = MODISNDVISource()
    ndvi_source._get_site_coordinates = lambda s: (lat, lon)

    request = DataFetchRequest(
        site_id=site_id,
        start_date=start_date,
        end_date=end_date,
        parameters={}
    )

    try:
        result = ndvi_source.fetch(request)
        if result.data:
            df = pd.DataFrame([d.model_dump() for d in result.data])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            return df
    except Exception as e:
        logger.warning(f"NDVI fetch failed for {site_id}: {e}")

    # Fallback: generate simple seasonal NDVI
    dates = pd.date_range(start_date, end_date, freq='D')
    doy = dates.dayofyear
    if abs(lat) < 15:  # Tropics
        ndvi_vals = 0.55 + 0.15 * np.sin(2 * np.pi * (doy - 60) / 365)
    elif lat > 0:
        ndvi_vals = 0.35 + 0.35 * np.sin(2 * np.pi * (doy - 100) / 365)
    else:
        ndvi_vals = 0.35 + 0.35 * np.sin(2 * np.pi * (doy + 80) / 365)
    return pd.DataFrame({'ndvi': ndvi_vals}, index=dates)


def run_model_for_site(site_id: str, lat: float, lon: float,
                       start_date: date, end_date: date,
                       sensor_depth_m: float = 0.10):
    """
    Run the water balance model for a site and return daily soil moisture.

    Args:
        site_id: Site identifier
        lat: Latitude
        lon: Longitude
        start_date: Start date
        end_date: End date
        sensor_depth_m: Depth of sensor for comparison (default 0.10m)

    Returns:
        Tuple of (DataFrame with daily model predictions, sr_results, soil_data)
    """
    # Fetch soil data
    soil_data = fetch_soil_data(site_id, lat, lon)
    if soil_data is None:
        return None

    # Determine soil texture for model
    sand = soil_data.get('sand', 40)
    clay = soil_data.get('clay', 30)
    if sand > 70:
        soil_texture = "sand"
    elif clay > 40:
        soil_texture = "clay"
    elif sand > 50 and clay < 20:
        soil_texture = "sandy_loam"
    elif clay > 25:
        soil_texture = "clay_loam"
    else:
        soil_texture = "loam"

    # Create model using factory (uses EnhancedWaterBalance)
    model = create_water_balance_model(
        crop_type="maize",
        soil_texture=soil_texture,
        use_full_physics=True
    )

    # Calculate Saxton-Rawls hydraulic properties for reporting
    om = soil_data.get('organic_carbon', 1.5)
    sr_params = estimate_soil_parameters_saxton(
        sand_percent=sand,
        clay_percent=clay,
        organic_matter_percent=om
    )
    sr_results = {
        'theta_fc': sr_params.field_capacity,
        'theta_sat': sr_params.porosity,
        'theta_wp': sr_params.wilting_point,
        'k_sat': sr_params.saturated_hydraulic_conductivity_cm_day,
    }

    # Fetch weather data
    weather_df = fetch_weather_data(site_id, lat, lon, start_date, end_date)
    if weather_df is None or len(weather_df) == 0:
        return None

    # Fetch NDVI data
    ndvi_df = fetch_ndvi_data(site_id, lat, lon, start_date, end_date)

    # Run daily simulation
    results = []
    for day_date, row in weather_df.iterrows():
        # Get NDVI for this day
        ndvi = 0.4
        if ndvi_df is not None and len(ndvi_df) > 0:
            try:
                ndvi = ndvi_df.loc[day_date,
                                   'ndvi'] if day_date in ndvi_df.index else ndvi_df['ndvi'].mean()
            except:
                ndvi = 0.4

        # Get precipitation and ET0
        precip = row.get('precipitation_sum', row.get('precipitation', 0)) or 0
        et0 = row.get('et0_fao_evapotranspiration_sum',
                      row.get('et0', 3.5)) or 3.5

        # Run model for one day (EnhancedWaterBalance returns tuple)
        result, fluxes = model.run_daily(
            precipitation_mm=precip,
            et0_mm=et0,
            ndvi=ndvi
        )

        # Get soil moisture at sensor depth
        # PhysicsPriorResult has theta_surface, theta_root attributes
        theta_surface = result.theta_surface
        theta_root = result.theta_root
        theta_deep = theta_root * 0.9  # Approximate deep layer

        if sensor_depth_m <= 0.10:
            sm_model = theta_surface
        elif sensor_depth_m <= 0.20:
            # Weighted average: 50% surface, 50% root zone (upper part)
            sm_model = 0.5 * theta_surface + 0.5 * theta_root
        else:
            # Integrated value or deep layer
            sm_model = theta_deep

        results.append({
            'date': day_date,
            'sm_model': sm_model,
            'sm_surface': theta_surface,
            'sm_root': theta_root,
            'sm_deep': theta_deep,
            'sm_integrated': (theta_surface + theta_root + theta_deep) / 3,
            'precip_mm': precip,
            'et0_mm': et0,
            'ndvi': ndvi,
        })

    return pd.DataFrame(results), sr_results, soil_data


def main():
    print("=" * 70)
    print("ISMN VALIDATION: SMPS Model vs In-Situ Observations")
    print("=" * 70)
    print()

    # Load ISMN data
    obs_df, meta_df = load_ismn_data()
    print(f"Loaded {len(obs_df)} observations from {len(meta_df)} stations")

    # Select sites with good data coverage (>500 days)
    good_sites = meta_df[meta_df['n_days'] > 500].copy()
    print(f"Sites with >500 days of data: {len(good_sites)}")
    print()

    # Run validation for each site
    all_results = []
    site_summaries = []

    for idx, site in good_sites.iterrows():
        station = site['station']
        lat = site['latitude']
        lon = site['longitude']
        depth = site['depth_m']

        print(
            f"Processing {station} ({lat:.4f}, {lon:.4f}, depth={depth}m)...", end=" ")

        # Get observation data for this site
        site_obs = obs_df[obs_df['station'] == station].copy()
        start_dt = site_obs['date'].min()
        end_dt = site_obs['date'].max()

        try:
            result = run_model_for_site(
                site_id=station,
                lat=lat,
                lon=lon,
                start_date=date(start_dt.year, start_dt.month, start_dt.day),
                end_date=date(end_dt.year, end_dt.month, end_dt.day),
                sensor_depth_m=depth
            )

            if result is None:
                print("FAILED (no data)")
                continue

            model_df, sr_results, soil_data = result

            # Merge model and observations
            model_df['date'] = pd.to_datetime(model_df['date'])
            merged = pd.merge(
                site_obs[['date', 'sm_observed']],
                model_df[['date', 'sm_model', 'sm_surface',
                          'sm_root', 'sm_integrated', 'precip_mm']],
                on='date',
                how='inner'
            )

            if len(merged) < 100:
                print(f"SKIPPED (only {len(merged)} matching days)")
                continue

            # Calculate metrics
            merged['bias'] = merged['sm_model'] - merged['sm_observed']
            merged['abs_error'] = merged['bias'].abs()

            mae = merged['abs_error'].mean()
            bias = merged['bias'].mean()
            rmse = np.sqrt((merged['bias'] ** 2).mean())
            corr = merged['sm_model'].corr(merged['sm_observed'])

            print(f"OK - MAE={mae:.3f}, Bias={bias:+.3f}, r={corr:.2f}")

            # Store results
            merged['station'] = station
            merged['latitude'] = lat
            merged['longitude'] = lon
            merged['sensor_depth_m'] = depth
            all_results.append(merged)

            site_summaries.append({
                'station': station,
                'latitude': lat,
                'longitude': lon,
                'sensor_depth_m': depth,
                'n_days': len(merged),
                'obs_mean': merged['sm_observed'].mean(),
                'model_mean': merged['sm_model'].mean(),
                'mae': mae,
                'bias': bias,
                'rmse': rmse,
                'correlation': corr,
                'sand_pct': soil_data.get('sand', np.nan),
                'clay_pct': soil_data.get('clay', np.nan),
                'fc_saxton_rawls': sr_results.get('theta_fc', np.nan),
                'porosity_saxton_rawls': sr_results.get('theta_sat', np.nan),
            })

            # Rate limit API calls
            time.sleep(0.5)

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Combine all results
    if len(all_results) > 0:
        results_df = pd.concat(all_results, ignore_index=True)
        summary_df = pd.DataFrame(site_summaries)

        # Save results
        results_df.to_csv('data/ismn/ismn_validation_results.csv', index=False)
        summary_df.to_csv('data/ismn/ismn_validation_summary.csv', index=False)

        print()
        print("=" * 70)
        print("OVERALL VALIDATION RESULTS")
        print("=" * 70)
        print()
        print(f"Sites validated: {len(summary_df)}")
        print(f"Total observations: {len(results_df)}")
        print()
        print("Aggregate Metrics:")
        print(
            f"  Mean Absolute Error: {results_df['abs_error'].mean():.3f} m³/m³")
        print(f"  Mean Bias:           {results_df['bias'].mean():+.3f} m³/m³")
        print(
            f"  RMSE:                {np.sqrt((results_df['bias']**2).mean()):.3f} m³/m³")
        print()
        print("Observed vs Model Means:")
        print(
            f"  ISMN observed mean:  {results_df['sm_observed'].mean():.3f} m³/m³")
        print(
            f"  Model predicted mean: {results_df['sm_model'].mean():.3f} m³/m³")
        print()
        print("By-Site Summary:")
        print(summary_df[['station', 'obs_mean', 'model_mean',
              'bias', 'mae', 'correlation']].to_string())
        print()
        print("=" * 70)
        print("SAXTON-RAWLS VS OBSERVED COMPARISON")
        print("=" * 70)
        print()
        print("Key Question: Is the FC from Saxton-Rawls correct?")
        print()
        print(summary_df[['station', 'obs_mean',
              'fc_saxton_rawls', 'sand_pct', 'clay_pct']].to_string())
        print()
        print(f"Average observed SM: {summary_df['obs_mean'].mean():.3f}")
        print(
            f"Average Saxton-Rawls FC: {summary_df['fc_saxton_rawls'].mean():.3f}")

    else:
        print("No sites successfully validated!")


if __name__ == '__main__':
    main()
