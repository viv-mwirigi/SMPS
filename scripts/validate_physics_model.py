#!/usr/bin/env python
"""
Physics Model Validation with FLDAS Soil Moisture Data

This script validates the two-bucket water balance model using:
- Real FLDAS soil moisture observations (Dec 2024) as ground truth
- Soil data from iSDA/SoilGrids APIs
- Weather data from Open-Meteo API
- Satellite NDVI from Google Earth Engine

The FLDAS data provides monthly averaged 0-100cm soil moisture which serves
as reference data to validate our physics model predictions.

Run from the project root with:
    python scripts/validate_physics_model.py
"""

from smps.physics import (
    EnhancedWaterBalance, EnhancedModelParameters,
    create_water_balance_model,
)
from smps.physics.pedotransfer import estimate_soil_parameters_saxton
from smps.data.sources.weather import OpenMeteoSource
from smps.data.sources.soil import MockSoilSource
from smps.data.sources.isda_authenticated import IsdaAfricaAuthenticatedSource
from smps.data.sources.gee_satellite import GoogleEarthEngineSatelliteSource
from smps.data.sources.base import DataFetchRequest
from smps.core.types import SoilParameters
import logging
from datetime import date, timedelta
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# SMPS imports

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Test sites configuration - Updated to fall within FLDAS coverage for Dec 2024
# All coordinates verified against FLDAS regional bounds
TEST_SITES = {
    # East Africa (EA): Bounds: left=21.0, bottom=-12.5, right=52.0, top=23.0
    "kenya_nairobi": {
        "latitude": -1.286,
        "longitude": 36.817,
        "elevation_m": 1795,
        "region": "ea",
        "description": "Kenya highlands - agricultural zone",
        "crop_type": "maize",
    },
    "ethiopia_addis": {
        "latitude": 9.032,
        "longitude": 38.750,
        "elevation_m": 2355,
        "region": "ea",
        "description": "Ethiopian highlands - teff cultivation",
        "crop_type": "teff",
    },
    "tanzania_arusha": {
        "latitude": -3.386,
        "longitude": 36.683,
        "elevation_m": 1400,
        "region": "ea",
        "description": "Northern Tanzania - coffee/banana zone",
        "crop_type": "coffee",
    },

    # West Africa (WA): Bounds: left=-19.1, bottom=2.0, right=27.4, top=21.0
    "ghana_kumasi": {
        "latitude": 6.688,
        "longitude": -1.624,
        "elevation_m": 270,
        "region": "wa",
        "description": "Humid tropical zone - cocoa belt",
        "crop_type": "cocoa",
    },
    "nigeria_kano": {
        "latitude": 12.002,
        "longitude": 8.592,
        "elevation_m": 476,
        "region": "wa",
        "description": "Sahel zone - sorghum/millet",
        "crop_type": "sorghum",
    },
    "senegal_thies": {
        "latitude": 14.790,
        "longitude": -16.926,
        "elevation_m": 70,
        "region": "wa",
        "description": "Groundnut basin - semi-arid",
        "crop_type": "groundnut",
    },

    # Southern Africa (SA): Bounds: left=4.1, bottom=-35.5, right=52.0, top=5.5
    "zambia_lusaka": {
        "latitude": -15.387,
        "longitude": 28.323,
        "elevation_m": 1280,
        "region": "sa",
        "description": "Central plateau - maize zone",
        "crop_type": "maize",
    },
    "mozambique_beira": {
        "latitude": -19.844,
        "longitude": 34.839,
        "elevation_m": 10,
        "region": "sa",
        "description": "Coastal lowland - rice/sugarcane",
        "crop_type": "rice",
    },
    "zimbabwe_harare": {
        "latitude": -17.829,
        "longitude": 31.054,
        "elevation_m": 1490,
        "region": "sa",
        "description": "Highveld - tobacco/maize",
        "crop_type": "maize",
    },
}


def fetch_fldas_observations(sites: dict, year: int = 2024, month: int = 12) -> dict:
    """Fetch FLDAS soil moisture observations for all sites"""
    fldas = FLDASSource()
    observations = {}

    print(f"\n📡 Fetching FLDAS observations for {year}-{month:02d}")
    print("-" * 50)

    for site_id, site_info in sites.items():
        obs = fldas.fetch_observation(
            site_id=site_id,
            lat=site_info['latitude'],
            lon=site_info['longitude'],
            year=year,
            month=month
        )

        if obs is not None:
            observations[site_id] = obs
            print(f"  ✓ {site_id}: SM = {obs.soil_moisture_vwc:.4f} m³/m³")
        else:
            print(f"  ✗ {site_id}: No FLDAS data available")

    return observations


def fetch_soil_data(site_id: str, lat: float, lon: float):
    """Fetch soil data from iSDA (authenticated API)"""
    try:
        isda = IsdaAfricaAuthenticatedSource()
        profile = isda.fetch_soil_profile(site_id, latitude=lat, longitude=lon)
        logger.info(f"✓ Fetched real soil data from iSDA for {site_id}")
        return profile
    except Exception as e:
        logger.warning(f"iSDA failed for {site_id}: {e}")

    # Fallback to mock
    mock = MockSoilSource()
    profile = mock.fetch_soil_profile(site_id)
    logger.info(f"Using mock soil data for {site_id}")
    return profile


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

    # Override coordinates
    weather_source._get_site_coordinates = lambda s: (lat, lon)

    try:
        data = weather_source.fetch_daily_weather(request)
        df = pd.DataFrame([d.dict() for d in data])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        logger.info(f"Fetched {len(df)} days of weather for {site_id}")
        return df
    except Exception as e:
        logger.error(f"Weather fetch failed for {site_id}: {e}")
        return pd.DataFrame()


def fetch_ndvi_data(site_id: str, lat: float, lon: float,
                    start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch NDVI data from Google Earth Engine"""
    from datetime import datetime

    gee_source = GoogleEarthEngineSatelliteSource()

    try:
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.min.time())

        observations = gee_source.fetch_ndvi(lat, lon, start_dt, end_dt)

        if observations:
            data = []
            for obs in observations:
                data.append({
                    'date': obs.timestamp,
                    'ndvi': obs.ndvi,
                    'evi': obs.evi,
                    'source': obs.source
                })
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            logger.info(
                f"✓ Fetched {len(df)} NDVI observations from GEE for {site_id}")
            return df
    except Exception as e:
        logger.warning(f"GEE NDVI fetch failed for {site_id}: {e}")

    return pd.DataFrame()


def run_physics_model(weather_df: pd.DataFrame, ndvi_df: pd.DataFrame,
                      soil_profile, use_three_layer: bool = True) -> pd.DataFrame:
    """
    Run the water balance model.

    Args:
        weather_df: Weather forcings
        ndvi_df: NDVI data
        soil_profile: Soil profile data
        use_three_layer: If True, use 3-layer model (0-100cm) for FLDAS comparison
    """

    # Create soil parameters
    soil_params = SoilParameters(
        sand_percent=soil_profile.sand_percent,
        silt_percent=soil_profile.silt_percent,
        clay_percent=soil_profile.clay_percent,
        porosity=soil_profile.porosity,
        field_capacity=soil_profile.field_capacity,
        wilting_point=soil_profile.wilting_point,
        saturated_hydraulic_conductivity_cm_day=soil_profile.saturated_hydraulic_conductivity_cm_day
    )

    # Determine soil texture for model
    sand = soil_profile.sand_percent
    clay = soil_profile.clay_percent
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

    # Create model using factory
    model = create_water_balance_model(
        crop_type="maize",
        soil_texture=soil_texture,
        use_full_physics=use_three_layer  # Full physics for 3-layer
    )

    # Merge NDVI with weather
    if not ndvi_df.empty and 'ndvi' in ndvi_df.columns:
        weather_df = weather_df.join(ndvi_df[['ndvi']], how='left')
        weather_df['ndvi'] = weather_df['ndvi'].ffill().bfill()
    else:
        weather_df['ndvi'] = 0.5

    # Run simulation
    results = []
    for idx, row in weather_df.iterrows():
        precip = row.get('precipitation_mm', 0.0) or 0.0
        et0 = row.get('et0_mm', 3.0) or 3.0
        ndvi = row.get('ndvi', 0.5) or 0.5

        try:
            # EnhancedWaterBalance always returns (result, fluxes) tuple
            result, fluxes = model.run_daily(
                precipitation_mm=precip,
                et0_mm=et0,
                ndvi=ndvi
            )

            theta_surface = result.theta_surface
            theta_root = result.theta_root
            theta_deep = result.theta_deep if result.theta_deep is not None else theta_root * 0.9

            results.append({
                'date': idx,
                'theta_surface': theta_surface,
                'theta_root': theta_root,
                'theta_deep': theta_deep,
                'theta_0_100cm': (theta_surface + theta_root + theta_deep) / 3,
                'et': fluxes.actual_et,
                'runoff': fluxes.runoff,
                'drainage': fluxes.deep_drainage,
                'water_balance_error': result.water_balance_error
            })
        except Exception as e:
            logger.warning(f"Model error on {idx}: {e}")
            results.append({
                'date': idx, 'theta_surface': np.nan, 'theta_root': np.nan,
                'theta_deep': np.nan, 'theta_0_100cm': np.nan
            })

    return pd.DataFrame(results).set_index('date')


def calculate_validation_metrics(model_sm: float, fldas_sm: float, site_id: str) -> dict:
    """
    Calculate validation metrics comparing model to FLDAS.

    Since FLDAS provides 0-100cm integrated moisture, we compare against
    a weighted average of surface (0-10cm) and root zone (10-40cm).
    """
    # Calculate absolute error
    abs_error = abs(model_sm - fldas_sm)

    # Calculate relative error
    rel_error = abs_error / fldas_sm if fldas_sm > 0 else np.nan

    # Bias (model - observation)
    bias = model_sm - fldas_sm

    return {
        'site': site_id,
        'model_sm': model_sm,
        'fldas_sm': fldas_sm,
        'abs_error': abs_error,
        'rel_error_pct': rel_error * 100,
        'bias': bias,
    }


def main():
    """Main validation workflow with FLDAS data"""
    print("=" * 70)
    print("PHYSICS MODEL VALIDATION WITH FLDAS SOIL MOISTURE DATA")
    print("=" * 70)

    # Analysis period - November to December 2024 (warm-up + validation)
    # FLDAS data available for December 2024
    WARMUP_START = date(2024, 11, 1)
    VALIDATION_MONTH_START = date(2024, 12, 1)
    VALIDATION_MONTH_END = date(2024, 12, 31)

    print(
        f"\nWarm-up period: {WARMUP_START} to {VALIDATION_MONTH_START - timedelta(days=1)}")
    print(
        f"Validation period: {VALIDATION_MONTH_START} to {VALIDATION_MONTH_END}")
    print(f"Reference data: FLDAS monthly soil moisture (0-100cm)")

    # Output directory
    output_dir = Path(__file__).parent.parent / 'data' / 'features'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch FLDAS observations (ground truth)
    fldas_observations = fetch_fldas_observations(
        TEST_SITES, year=2024, month=12)

    if not fldas_observations:
        print("\n❌ ERROR: No FLDAS observations available. Cannot validate.")
        return

    print(f"\n✓ Retrieved FLDAS data for {len(fldas_observations)} sites")

    # Collect results
    validation_results = []
    detailed_results = {}

    for site_id, site_info in TEST_SITES.items():
        # Skip sites without FLDAS data
        if site_id not in fldas_observations:
            print(f"\n⚠ Skipping {site_id} - no FLDAS reference data")
            continue

        fldas_obs = fldas_observations[site_id]

        print(f"\n{'='*60}")
        print(f"Processing: {site_id}")
        print(
            f"Location: ({site_info['latitude']:.3f}, {site_info['longitude']:.3f})")
        print(
            f"Region: {site_info['region'].upper()} - {site_info['description']}")
        print(f"FLDAS reference SM: {fldas_obs.soil_moisture_vwc:.4f} m³/m³")
        print("=" * 60)

        # 1. Fetch soil data
        soil_profile = fetch_soil_data(
            site_id, site_info['latitude'], site_info['longitude'])
        print(
            f"  Soil: Sand={soil_profile.sand_percent:.1f}%, Clay={soil_profile.clay_percent:.1f}%")

        # 2. Fetch weather (Nov-Dec 2024)
        weather_df = fetch_weather_data(
            site_id, site_info['latitude'], site_info['longitude'],
            WARMUP_START, VALIDATION_MONTH_END
        )
        if weather_df.empty:
            print(f"  ⚠ Skipping {site_id} - no weather data")
            continue

        # 3. Fetch NDVI
        ndvi_df = fetch_ndvi_data(
            site_id, site_info['latitude'], site_info['longitude'],
            WARMUP_START, VALIDATION_MONTH_END
        )

        # 4. Run physics model
        physics_df = run_physics_model(weather_df, ndvi_df, soil_profile)

        # 5. Calculate December monthly average from model
        dec_data = physics_df.loc[physics_df.index >=
                                  str(VALIDATION_MONTH_START)]
        if dec_data.empty:
            print(f"  ⚠ No December model output for {site_id}")
            continue

        # FLDAS is 0-100cm integrated - use theta_0_100cm from 3-layer model
        model_sm_surface = dec_data['theta_surface'].mean()
        model_sm_root = dec_data['theta_root'].mean()

        # Get integrated 0-100cm (properly calculated by 3-layer model)
        if 'theta_0_100cm' in dec_data.columns and not dec_data['theta_0_100cm'].isna().all():
            model_sm_integrated = dec_data['theta_0_100cm'].mean()
            model_sm_deep = dec_data['theta_deep'].mean(
            ) if 'theta_deep' in dec_data.columns else np.nan
        else:
            # Fallback for 2-layer model
            model_sm_integrated = 0.10 * model_sm_surface + 0.90 * model_sm_root
            model_sm_deep = np.nan

        print(
            f"  Model Dec mean: Surface={model_sm_surface:.4f}, Root={model_sm_root:.4f}")
        if not np.isnan(model_sm_deep):
            print(f"  Model Dec mean: Deep (40-100cm)={model_sm_deep:.4f}")
        print(f"  Model 0-100cm integrated: {model_sm_integrated:.4f} m³/m³")
        print(
            f"  FLDAS 0-100cm observed: {fldas_obs.soil_moisture_vwc:.4f} m³/m³")

        # 6. Calculate validation metrics
        metrics = calculate_validation_metrics(
            model_sm=model_sm_integrated,
            fldas_sm=fldas_obs.soil_moisture_vwc,
            site_id=site_id
        )

        metrics['region'] = site_info['region']
        metrics['crop_type'] = site_info['crop_type']
        metrics['sand_pct'] = soil_profile.sand_percent
        metrics['clay_pct'] = soil_profile.clay_percent

        validation_results.append(metrics)
        detailed_results[site_id] = {
            'physics_df': physics_df,
            'weather_df': weather_df,
            'fldas_obs': fldas_obs,
            'metrics': metrics
        }

        print(f"  Absolute Error: {metrics['abs_error']:.4f} m³/m³")
        print(f"  Relative Error: {metrics['rel_error_pct']:.1f}%")
        print(f"  Bias: {metrics['bias']:+.4f} m³/m³")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if validation_results:
        results_df = pd.DataFrame(validation_results)

        print(f"\nSites validated: {len(results_df)}")
        print(
            f"Mean Absolute Error: {results_df['abs_error'].mean():.4f} m³/m³")
        print(
            f"Mean Relative Error: {results_df['rel_error_pct'].mean():.1f}%")
        print(f"Mean Bias: {results_df['bias'].mean():+.4f} m³/m³")
        print(f"RMSE: {np.sqrt((results_df['bias']**2).mean()):.4f} m³/m³")

        # By region
        print("\nBy Region:")
        for region in results_df['region'].unique():
            region_data = results_df[results_df['region'] == region]
            print(f"  {region.upper()}: MAE={region_data['abs_error'].mean():.4f}, "
                  f"Bias={region_data['bias'].mean():+.4f} ({len(region_data)} sites)")

        # Save results
        results_df.to_csv(
            output_dir / 'fldas_validation_results.csv', index=False)
        print(
            f"\n✓ Results saved to {output_dir / 'fldas_validation_results.csv'}")

        # Detailed per-site table
        print("\n" + "-" * 90)
        print(
            f"{'Site':<20} {'Model SM':>10} {'FLDAS SM':>10} {'Error':>10} {'Bias':>10} {'Region':>8}")
        print("-" * 90)
        for _, row in results_df.iterrows():
            print(f"{row['site']:<20} {row['model_sm']:>10.4f} {row['fldas_sm']:>10.4f} "
                  f"{row['abs_error']:>10.4f} {row['bias']:>+10.4f} {row['region']:>8}")
        print("-" * 90)

    else:
        print("\n❌ No validation results generated")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
