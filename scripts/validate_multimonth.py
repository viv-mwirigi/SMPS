#!/usr/bin/env python
"""
Multi-Month Physics Model Validation with FLDAS Soil Moisture Data

Validates the three-layer water balance model across 12 months of FLDAS data:
- October 2024 to September 2025
- Tests seasonal dynamics and model performance across wet/dry seasons

Run from the project root with:
    python scripts/validate_multimonth.py
"""

from smps.core.types import SoilParameters
from smps.data.sources.base import DataFetchRequest
from smps.data.sources.isda_authenticated import IsdaAfricaAuthenticatedSource
from smps.data.sources.soil import MockSoilSource
from smps.data.sources.weather import OpenMeteoSource
from smps.data.sources.satellite import MODISNDVISource
from smps.physics import create_water_balance_model
import logging
from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Track data source usage
DATA_SOURCE_STATS = {
    'weather_fetched': 0,
    'weather_failed': 0,
    'ndvi_fetched': 0,
    'ndvi_synthetic': 0,
    'soil_isda': 0,
    'soil_mock': 0
}

# Available months (Oct 2024 - Sep 2025)
AVAILABLE_MONTHS = [
    (2024, 10), (2024, 11), (2024, 12),
    (2025, 1), (2025, 2), (2025, 3),
    (2025, 4), (2025, 5), (2025, 6),
    (2025, 7), (2025, 8), (2025, 9)
]

# Test sites - 7+ sites per region (EA, WA, SA)
TEST_SITES = {
    # ============================================================
    # EAST AFRICA (7 sites)
    # ============================================================
    "kenya_nairobi": {
        "latitude": -1.286, "longitude": 36.817,
        "region": "ea", "description": "Kenya highlands"
    },
    "ethiopia_addis": {
        "latitude": 9.032, "longitude": 38.750,
        "region": "ea", "description": "Ethiopian highlands"
    },
    "tanzania_arusha": {
        "latitude": -3.386, "longitude": 36.683,
        "region": "ea", "description": "Northern Tanzania"
    },
    "uganda_kampala": {
        "latitude": 0.347, "longitude": 32.582,
        "region": "ea", "description": "Lake Victoria region"
    },
    "rwanda_kigali": {
        "latitude": -1.944, "longitude": 30.059,
        "region": "ea", "description": "Rwanda highlands"
    },
    "kenya_mombasa": {
        "latitude": -4.043, "longitude": 39.668,
        "region": "ea", "description": "Coastal Kenya"
    },
    "tanzania_dodoma": {
        "latitude": -6.162, "longitude": 35.752,
        "region": "ea", "description": "Central Tanzania"
    },

    # ============================================================
    # WEST AFRICA (7 sites)
    # ============================================================
    "ghana_kumasi": {
        "latitude": 6.688, "longitude": -1.624,
        "region": "wa", "description": "Humid tropical Ghana"
    },
    "nigeria_kano": {
        "latitude": 12.002, "longitude": 8.592,
        "region": "wa", "description": "Sahel zone Nigeria"
    },
    "senegal_thies": {
        "latitude": 14.790, "longitude": -16.926,
        "region": "wa", "description": "Semi-arid Senegal"
    },
    "nigeria_ibadan": {
        "latitude": 7.378, "longitude": 3.947,
        "region": "wa", "description": "Humid Nigeria"
    },
    "mali_bamako": {
        "latitude": 12.639, "longitude": -8.003,
        "region": "wa", "description": "Sahel Mali"
    },
    "burkina_ouaga": {
        "latitude": 12.364, "longitude": -1.534,
        "region": "wa", "description": "Burkina Faso savanna"
    },
    "cote_divoire_abidjan": {
        "latitude": 5.316, "longitude": -4.028,
        "region": "wa", "description": "Coastal Ivory Coast"
    },

    # ============================================================
    # SOUTHERN AFRICA (7 sites)
    # ============================================================
    "zambia_lusaka": {
        "latitude": -15.387, "longitude": 28.323,
        "region": "sa", "description": "Central Zambia plateau"
    },
    "zimbabwe_harare": {
        "latitude": -17.829, "longitude": 31.054,
        "region": "sa", "description": "Zimbabwe highveld"
    },
    "mozambique_maputo": {
        "latitude": -25.891, "longitude": 32.605,
        "region": "sa", "description": "Coastal Mozambique"
    },
    "malawi_lilongwe": {
        "latitude": -13.983, "longitude": 33.774,
        "region": "sa", "description": "Central Malawi"
    },
    "botswana_gaborone": {
        "latitude": -24.628, "longitude": 25.923,
        "region": "sa", "description": "Semi-arid Botswana"
    },
    "south_africa_pretoria": {
        "latitude": -25.747, "longitude": 28.229,
        "region": "sa", "description": "South Africa highveld"
    },
    "namibia_windhoek": {
        "latitude": -22.560, "longitude": 17.083,
        "region": "sa", "description": "Semi-arid Namibia"
    },
}


def fetch_soil_data(site_id: str, lat: float, lon: float):
    """Fetch soil data from iSDA or fallback to mock"""
    global DATA_SOURCE_STATS
    try:
        isda = IsdaAfricaAuthenticatedSource()
        profile = isda.fetch_soil_profile(site_id, latitude=lat, longitude=lon)
        DATA_SOURCE_STATS['soil_isda'] += 1
        return profile
    except Exception as e:
        logger.warning(f"iSDA failed for {site_id}: {e}")
        mock = MockSoilSource()
        DATA_SOURCE_STATS['soil_mock'] += 1
        return mock.fetch_soil_profile(site_id)


def fetch_weather_data(site_id: str, lat: float, lon: float,
                       start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch weather data from Open-Meteo"""
    global DATA_SOURCE_STATS
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
        DATA_SOURCE_STATS['weather_fetched'] += 1
        return df
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
        DATA_SOURCE_STATS['weather_failed'] += 1
        return pd.DataFrame()


def fetch_ndvi_data(site_id: str, lat: float, lon: float,
                    start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch NDVI data from MODIS/Sentinel source"""
    global DATA_SOURCE_STATS

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

            # Track if synthetic
            if result.metadata.get('source') == 'synthetic':
                DATA_SOURCE_STATS['ndvi_synthetic'] += 1
            else:
                DATA_SOURCE_STATS['ndvi_fetched'] += 1

            return df

    except Exception as e:
        logger.warning(f"NDVI fetch failed for {site_id}: {e}")

    # Fallback: generate simple seasonal NDVI
    DATA_SOURCE_STATS['ndvi_synthetic'] += 1
    dates = pd.date_range(start_date, end_date, freq='D')

    # Simple seasonal pattern
    doy = dates.dayofyear
    if abs(lat) < 15:  # Tropics
        ndvi_vals = 0.55 + 0.15 * np.sin(2 * np.pi * (doy - 60) / 365)
    elif lat > 0:  # Northern hemisphere
        ndvi_vals = 0.35 + 0.35 * np.sin(2 * np.pi * (doy - 100) / 365)
    else:  # Southern hemisphere
        ndvi_vals = 0.35 + 0.35 * np.sin(2 * np.pi * (doy + 80) / 365)

    return pd.DataFrame({'ndvi': ndvi_vals}, index=dates)


def run_continuous_simulation(site_id: str, site_info: dict,
                              start_date: date, end_date: date) -> pd.DataFrame:
    """Run continuous model simulation for entire period"""

    # Fetch soil data once
    soil_profile = fetch_soil_data(
        site_id, site_info['latitude'], site_info['longitude'])

    # Fetch weather for entire period
    weather_df = fetch_weather_data(
        site_id, site_info['latitude'], site_info['longitude'],
        start_date, end_date
    )

    if weather_df.empty:
        logger.error(f"No weather data for {site_id}")
        return pd.DataFrame()

    # Fetch NDVI data for vegetation dynamics
    ndvi_df = fetch_ndvi_data(
        site_id, site_info['latitude'], site_info['longitude'],
        start_date, end_date
    )

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

    # Determine soil texture
    sand = soil_params.sand_percent
    clay = soil_params.clay_percent
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
        use_full_physics=True
    )

    # Run simulation with dynamic NDVI
    results = []
    for idx, row in weather_df.iterrows():
        precip = row.get('precipitation_mm', 0.0) or 0.0
        et0 = row.get('et0_mm', 3.0) or 3.0

        # Get NDVI for this date (or interpolate)
        if not ndvi_df.empty and idx in ndvi_df.index:
            ndvi = ndvi_df.loc[idx, 'ndvi']
        elif not ndvi_df.empty:
            # Find nearest date
            nearest_idx = ndvi_df.index.get_indexer([idx], method='nearest')[0]
            if nearest_idx >= 0 and nearest_idx < len(ndvi_df):
                ndvi = ndvi_df.iloc[nearest_idx]['ndvi']
            else:
                ndvi = 0.5
        else:
            ndvi = 0.5  # Fallback

        # EnhancedWaterBalance returns (result, fluxes) tuple
        # result is a PhysicsPriorResult dataclass with theta_surface, theta_root, theta_deep
        result, fluxes = model.run_daily(
            precipitation_mm=precip,
            et0_mm=et0,
            ndvi=ndvi
        )

        # PhysicsPriorResult now has theta_deep attribute for 3-layer models
        theta_surface = result.theta_surface
        theta_root = result.theta_root
        theta_deep = result.theta_deep if result.theta_deep is not None else theta_root * 0.9

        theta_integrated = (theta_surface + theta_root + theta_deep) / 3

        results.append({
            'date': idx,
            'theta_surface': theta_surface,
            'theta_root': theta_root,
            'theta_deep': theta_deep,
            'theta_0_100cm': theta_integrated,
            'precipitation_mm': precip,
            'et0_mm': et0,
            'ndvi': ndvi
        })

    return pd.DataFrame(results).set_index('date')


def main():
    """Main multi-month validation workflow"""
    print("=" * 70)
    print("MULTI-MONTH PHYSICS MODEL VALIDATION")
    print("FLDAS Data: October 2024 - September 2025 (12 months)")
    print(f"Test Sites: {len(TEST_SITES)} sites across 3 African regions")
    print("=" * 70)

    # Show sites by region
    regions = {'ea': 'East Africa',
               'wa': 'West Africa', 'sa': 'Southern Africa'}
    for region_code, region_name in regions.items():
        sites = [s for s, info in TEST_SITES.items() if info['region']
                 == region_code]
        print(f"\n{region_name} ({len(sites)} sites):")
        for site in sites:
            info = TEST_SITES[site]
            print(
                f"  - {site}: {info['description']} ({info['latitude']:.2f}, {info['longitude']:.2f})")

    # Output directory
    output_dir = Path(__file__).parent.parent / 'data' / 'features'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize FLDAS source
    fldas = FLDASSource()

    # Simulation period (with 1-month warmup before Oct 2024)
    SIM_START = date(2024, 9, 1)  # Warmup from Sep 2024
    SIM_END = date(2025, 9, 30)

    all_results = []
    monthly_results = []

    for site_id, site_info in TEST_SITES.items():
        print(f"\n{'='*60}")
        print(f"Processing: {site_id} ({site_info['description']})")
        print(f"Region: {site_info['region'].upper()}")
        print("=" * 60)

        # Run continuous simulation
        sim_df = run_continuous_simulation(
            site_id, site_info, SIM_START, SIM_END)

        if sim_df.empty:
            print(f"  ⚠ Skipping {site_id} - simulation failed")
            continue

        print(f"  ✓ Simulated {len(sim_df)} days")

        # Compare with FLDAS for each month
        for year, month in AVAILABLE_MONTHS:
            # Get FLDAS observation
            fldas_obs = fldas.fetch_observation(
                site_id=site_id,
                lat=site_info['latitude'],
                lon=site_info['longitude'],
                year=year,
                month=month
            )

            if fldas_obs is None:
                continue

            # Get monthly average from simulation
            month_start = date(year, month, 1)
            if month == 12:
                month_end = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                month_end = date(year, month + 1, 1) - timedelta(days=1)

            month_mask = (sim_df.index >= str(month_start)) & (
                sim_df.index <= str(month_end))
            month_data = sim_df[month_mask]

            if month_data.empty:
                continue

            model_sm = month_data['theta_0_100cm'].mean()
            fldas_sm = fldas_obs.soil_moisture_vwc

            result = {
                'site': site_id,
                'region': site_info['region'],
                'year': year,
                'month': month,
                'model_sm': model_sm,
                'fldas_sm': fldas_sm,
                'bias': model_sm - fldas_sm,
                'abs_error': abs(model_sm - fldas_sm),
                'rel_error_pct': abs(model_sm - fldas_sm) / fldas_sm * 100 if fldas_sm > 0 else np.nan
            }

            monthly_results.append(result)

            print(f"  {year}-{month:02d}: Model={model_sm:.3f}, FLDAS={fldas_sm:.3f}, "
                  f"Bias={model_sm - fldas_sm:+.3f}")

    # Create results DataFrame
    results_df = pd.DataFrame(monthly_results)

    if results_df.empty:
        print("\n❌ No results generated")
        return

    # Summary statistics
    print("\n" + "=" * 70)
    print("MULTI-MONTH VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\nTotal observations: {len(results_df)}")
    print(f"Sites: {results_df['site'].nunique()}")
    print(f"Months: {len(AVAILABLE_MONTHS)}")

    print(f"\n{'='*60}")
    print("OVERALL METRICS")
    print("="*60)
    print(f"Mean Absolute Error: {results_df['abs_error'].mean():.4f} m³/m³")
    print(f"Mean Bias: {results_df['bias'].mean():+.4f} m³/m³")
    print(f"RMSE: {np.sqrt((results_df['bias']**2).mean()):.4f} m³/m³")
    print(f"Mean Relative Error: {results_df['rel_error_pct'].mean():.1f}%")

    # By month
    print(f"\n{'='*60}")
    print("BY MONTH")
    print("="*60)
    monthly_stats = results_df.groupby(['year', 'month']).agg({
        'bias': 'mean',
        'abs_error': 'mean',
        'model_sm': 'mean',
        'fldas_sm': 'mean'
    }).round(4)
    print(monthly_stats.to_string())

    # By region
    print(f"\n{'='*60}")
    print("BY REGION")
    print("="*60)
    region_stats = results_df.groupby('region').agg({
        'bias': ['mean', 'std'],
        'abs_error': 'mean',
        'rel_error_pct': 'mean'
    }).round(4)
    print(region_stats.to_string())

    # By site
    print(f"\n{'='*60}")
    print("BY SITE")
    print("="*60)
    site_stats = results_df.groupby('site').agg({
        'bias': 'mean',
        'abs_error': 'mean',
        'model_sm': 'mean',
        'fldas_sm': 'mean'
    }).round(4)
    print(site_stats.to_string())

    # Seasonal analysis
    print(f"\n{'='*60}")
    print("SEASONAL ANALYSIS")
    print("="*60)

    # Define seasons (for East/Southern Africa)
    def get_season(row):
        month = row['month']
        if month in [10, 11, 12, 1, 2, 3]:
            return 'Wet (Oct-Mar)'
        else:
            return 'Dry (Apr-Sep)'

    results_df['season'] = results_df.apply(get_season, axis=1)
    season_stats = results_df.groupby('season').agg({
        'bias': ['mean', 'std'],
        'abs_error': 'mean',
        'model_sm': 'mean',
        'fldas_sm': 'mean'
    }).round(4)
    print(season_stats.to_string())

    # Save results
    results_df.to_csv(
        output_dir / 'multimonth_validation_results.csv', index=False)
    print(
        f"\n✓ Results saved to {output_dir / 'multimonth_validation_results.csv'}")

    # Save summary
    summary = {
        'n_observations': len(results_df),
        'n_sites': results_df['site'].nunique(),
        'n_months': len(AVAILABLE_MONTHS),
        'mae': results_df['abs_error'].mean(),
        'rmse': np.sqrt((results_df['bias']**2).mean()),
        'mean_bias': results_df['bias'].mean(),
        'mean_rel_error_pct': results_df['rel_error_pct'].mean()
    }
    pd.DataFrame([summary]).to_csv(
        output_dir / 'validation_metrics_summary.csv', index=False)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    # Print data source statistics
    print(f"\n{'='*60}")
    print("DATA SOURCE STATISTICS")
    print("="*60)
    print(f"Weather Data:")
    print(
        f"  - Fetched from Open-Meteo: {DATA_SOURCE_STATS['weather_fetched']} sites")
    print(
        f"  - Failed (fallback): {DATA_SOURCE_STATS['weather_failed']} sites")
    print(f"\nNDVI/Vegetation Data:")
    print(f"  - Fetched from MODIS: {DATA_SOURCE_STATS['ndvi_fetched']} sites")
    print(
        f"  - Synthetic (seasonal model): {DATA_SOURCE_STATS['ndvi_synthetic']} sites")
    print(f"\nSoil Data:")
    print(f"  - Fetched from iSDA: {DATA_SOURCE_STATS['soil_isda']} sites")
    print(f"  - Mock data: {DATA_SOURCE_STATS['soil_mock']} sites")


if __name__ == "__main__":
    main()
