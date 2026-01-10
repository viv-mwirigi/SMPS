#!/usr/bin/env python3
"""
ISMN Validation with Depth-Matched Predictions.

This script:
1. Loads ISMN stations with their actual sensor depths
2. Runs the physics model with interpolation to sensor depths
3. Validates at different time horizons (24h, 72h, 168h)
4. Produces metrics by depth, network, and horizon
"""

import sys
import logging
from pathlib import Path
from datetime import date, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import interpolate

# SMPS imports
from smps.data.sources.ismn_loader import ISMNStationLoader, ISMNStationData
from smps.data.sources.weather import OpenMeteoSource
from smps.data.sources.base import DataFetchRequest
from smps.core.types import SiteID, SoilParameters
from smps.physics.enhanced_water_balance import (
    EnhancedWaterBalance,
    EnhancedModelParameters,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results for one station-depth combination."""
    network: str
    station: str
    depth_cm: float
    n_obs: int
    rmse: float
    mae: float
    r2: float
    kge: float
    nse: float
    bias: float
    horizon: str = "0h"


def interpolate_to_depth(layer_centers: np.ndarray,
                         layer_values: np.ndarray,
                         target_depth_cm: float) -> float:
    """
    Interpolate model layer values to a specific target depth.

    Args:
        layer_centers: Center depths of model layers in cm
        layer_values: Soil moisture values for each layer
        target_depth_cm: Target depth to interpolate to

    Returns:
        Interpolated soil moisture value
    """
    # Handle edge cases
    if target_depth_cm <= layer_centers[0]:
        return layer_values[0]
    if target_depth_cm >= layer_centers[-1]:
        return layer_values[-1]

    # Linear interpolation
    f = interpolate.interp1d(layer_centers, layer_values, kind='linear',
                             bounds_error=False, fill_value='extrapolate')
    return float(f(target_depth_cm))


def get_model_layer_centers(layer_depths_m: List[float]) -> np.ndarray:
    """
    Calculate center depths of model layers in cm.

    Args:
        layer_depths_m: Layer thicknesses in meters

    Returns:
        Array of layer center depths in cm
    """
    centers = []
    cum_depth = 0.0
    for thickness in layer_depths_m:
        center = (cum_depth + thickness / 2) * 100  # Convert to cm
        centers.append(center)
        cum_depth += thickness
    return np.array(centers)


def calculate_metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Calculate validation metrics."""
    valid = ~(np.isnan(obs) | np.isnan(pred))
    if valid.sum() < 10:
        return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan,
                'kge': np.nan, 'nse': np.nan, 'bias': np.nan, 'n': 0}

    o, p = obs[valid], pred[valid]

    # RMSE
    rmse = np.sqrt(np.mean((o - p) ** 2))

    # MAE
    mae = np.mean(np.abs(o - p))

    # Bias
    bias = np.mean(p - o)

    # R²
    if np.std(o) > 0:
        r2 = np.corrcoef(o, p)[0, 1] ** 2
    else:
        r2 = np.nan

    # NSE
    if np.var(o) > 0:
        nse = 1 - np.sum((o - p) ** 2) / np.sum((o - np.mean(o)) ** 2)
    else:
        nse = np.nan

    # KGE
    if np.std(o) > 0 and np.std(p) > 0:
        r = np.corrcoef(o, p)[0, 1]
        alpha = np.std(p) / np.std(o)
        beta = np.mean(p) / np.mean(o)
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    else:
        kge = np.nan

    return {'rmse': rmse, 'mae': mae, 'r2': r2,
            'kge': kge, 'nse': nse, 'bias': bias, 'n': len(o)}


def run_model_for_station(station: ISMNStationData,
                          weather_source: OpenMeteoSource,
                          target_depths_cm: List[float]) -> Optional[pd.DataFrame]:
    """
    Run model for a station and interpolate to target depths.

    Returns:
        DataFrame with columns: date, pred_10cm, pred_20cm, etc.
    """
    # Get observation date range
    obs_data = station.daily_data
    if obs_data is None or len(obs_data) == 0:
        return None

    dates = pd.to_datetime(obs_data['date'])
    start_date = dates.min().date()
    end_date = dates.max().date()

    # Create model with default layer configuration
    # 0-10, 10-30, 30-50, 50-75, 75-100 cm
    layer_depths_m = [0.10, 0.20, 0.20, 0.25, 0.25]
    layer_centers = get_model_layer_centers(layer_depths_m)

    # Default soil parameters (can be improved with actual soil data)
    soil_params = SoilParameters(
        sand_percent=40.0,
        clay_percent=25.0,
        silt_percent=35.0,
        porosity=0.45,
        field_capacity=0.25,
        wilting_point=0.10,
        saturated_hydraulic_conductivity_cm_day=20.0,
        organic_matter_percent=2.0,
        bulk_density_g_cm3=1.35
    )

    try:
        # Create model parameters
        model_params = EnhancedModelParameters.from_soil_parameters(
            soil_params=soil_params,
            layer_depths_m=layer_depths_m,
            crop_type='grassland',
            use_depth_dependent_properties=True
        )

        # Initialize model
        model = EnhancedWaterBalance(model_params)

        # Fetch weather
        site_id = SiteID(station.station_id)
        weather_request = DataFetchRequest(
            site_id=site_id,
            start_date=start_date,
            end_date=end_date,
            parameters={}
        )
        weather = weather_source._fetch_historical(
            weather_request, station.latitude, station.longitude
        )

        if not weather:
            logger.warning(f"No weather data for {station.station_id}")
            return None

        # Run model day by day
        predictions = []
        for w in weather:
            result = model.step(
                date=w.date,
                precipitation_mm=w.precipitation_mm,
                et0_mm=w.et0_mm or 4.0,
                temperature_mean_c=w.temperature_mean_c or 25.0,
                ndvi=0.4  # Default NDVI
            )

            # Get layer soil moisture
            layer_sm = np.array([layer.theta for layer in model.layers])

            # Interpolate to target depths
            row = {'date': w.date}
            for depth in target_depths_cm:
                sm_at_depth = interpolate_to_depth(
                    layer_centers, layer_sm, depth)
                row[f'pred_{int(depth)}cm'] = sm_at_depth

            predictions.append(row)

        return pd.DataFrame(predictions)

    except Exception as e:
        logger.warning(f"Model error for {station.station_id}: {e}")
        return None


def validate_station(station: ISMNStationData,
                     predictions: pd.DataFrame,
                     horizons: Dict[str, int]) -> List[ValidationResult]:
    """
    Validate predictions against observations at all depths and horizons.
    """
    results = []
    obs_data = station.daily_data

    if obs_data is None or predictions is None:
        return results

    # Merge observations with predictions
    obs_data = obs_data.copy()
    obs_data['date'] = pd.to_datetime(obs_data['date'])
    predictions['date'] = pd.to_datetime(predictions['date'])

    for depth in station.available_depths_cm:
        depth_int = int(round(depth))
        pred_col = f'pred_{depth_int}cm'

        if pred_col not in predictions.columns:
            # Find nearest available prediction depth
            avail_depths = [int(c.split('_')[1].replace('cm', ''))
                            for c in predictions.columns if c.startswith('pred_')]
            nearest = min(avail_depths, key=lambda x: abs(x - depth_int))
            pred_col = f'pred_{nearest}cm'

        # Get observations for this depth
        depth_obs = obs_data[obs_data['depth_cm'] == depth].copy()
        if len(depth_obs) == 0:
            continue

        # Merge with predictions
        merged = pd.merge(
            depth_obs[['date', 'soil_moisture_mean']],
            predictions[['date', pred_col]],
            on='date',
            how='inner'
        )

        if len(merged) < 30:
            continue

        # Validate at each horizon
        for horizon_name, horizon_days in horizons.items():
            if horizon_days == 0:
                obs = merged['soil_moisture_mean'].values
                pred = merged[pred_col].values
            else:
                # Shift observations to simulate forecast validation
                obs = merged['soil_moisture_mean'].shift(-horizon_days).values
                pred = merged[pred_col].values

            metrics = calculate_metrics(obs, pred)

            if metrics['n'] >= 30:
                results.append(ValidationResult(
                    network=station.network,
                    station=station.station,
                    depth_cm=depth,
                    n_obs=metrics['n'],
                    rmse=metrics['rmse'],
                    mae=metrics['mae'],
                    r2=metrics['r2'],
                    kge=metrics['kge'],
                    nse=metrics['nse'],
                    bias=metrics['bias'],
                    horizon=horizon_name
                ))

    return results


def main():
    print("=" * 80)
    print("ISMN VALIDATION WITH DEPTH-MATCHED PREDICTIONS")
    print("=" * 80)

    # Configuration
    horizons = {'0h': 0, '24h': 1, '72h': 3, '168h': 7}

    # Load stations
    loader = ISMNStationLoader(Path("data/ismn"))
    all_stations = loader.load_all_stations()
    print(f"\nLoaded {len(all_stations)} stations")

    # Get all unique sensor depths
    all_depths = set()
    for station in all_stations.values():
        all_depths.update(station.available_depths_cm)
    all_depths = sorted(all_depths)
    print(f"Sensor depths to match: {[int(d) for d in all_depths]} cm")

    # Weather source
    weather_source = OpenMeteoSource(cache_dir=Path("data/cache/weather"))

    # Run validation
    all_results = []

    for i, (station_id, station) in enumerate(all_stations.items()):
        print(f"\n[{i+1}/{len(all_stations)}] {station.station_id}")
        print(f"  Location: ({station.latitude:.4f}, {station.longitude:.4f})")
        print(f"  Depths: {station.available_depths_cm} cm")

        # Run model with all target depths
        predictions = run_model_for_station(
            station, weather_source, target_depths_cm=all_depths
        )

        if predictions is None:
            print("  SKIPPED: No predictions")
            continue

        # Validate
        results = validate_station(station, predictions, horizons)
        all_results.extend(results)

        if results:
            # Show sample result for 0h horizon
            r0 = [r for r in results if r.horizon == '0h']
            if r0:
                print(f"  Results ({len(r0)} depths):")
                for r in r0[:3]:
                    print(
                        f"    {int(r.depth_cm):3d}cm: RMSE={r.rmse:.3f}, KGE={r.kge:.3f}")

    # Save detailed results
    df = pd.DataFrame([{
        'network': r.network,
        'station': r.station,
        'depth_cm': r.depth_cm,
        'horizon': r.horizon,
        'n_obs': r.n_obs,
        'rmse': r.rmse,
        'mae': r.mae,
        'r2': r.r2,
        'kge': r.kge,
        'nse': r.nse,
        'bias': r.bias
    } for r in all_results])

    output_dir = Path("results/ismn_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "depth_matched_validation.csv", index=False)
    print(
        f"\n\nSaved detailed results to: {output_dir / 'depth_matched_validation.csv'}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY BY DEPTH AND HORIZON")
    print("=" * 80)

    for horizon in horizons.keys():
        print(f"\n--- Horizon: {horizon} ---")
        h_df = df[df['horizon'] == horizon]

        if len(h_df) == 0:
            print("  No results")
            continue

        # By depth
        print(f"\n  By Depth:")
        for depth in sorted(h_df['depth_cm'].unique()):
            d_df = h_df[h_df['depth_cm'] == depth]
            print(f"    {int(depth):3d}cm: n={len(d_df):2d}, RMSE={d_df['rmse'].mean():.3f}, "
                  f"KGE={d_df['kge'].mean():.3f}, R²={d_df['r2'].mean():.3f}")

    # Overall by horizon
    print("\n" + "=" * 80)
    print("OVERALL BY HORIZON")
    print("=" * 80)
    for horizon in horizons.keys():
        h_df = df[df['horizon'] == horizon]
        if len(h_df) > 0:
            print(f"\n{horizon}: RMSE={h_df['rmse'].mean():.3f}, "
                  f"KGE={h_df['kge'].mean():.3f}, "
                  f"R²={h_df['r2'].mean():.3f}, "
                  f"Bias={h_df['bias'].mean():.4f}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
