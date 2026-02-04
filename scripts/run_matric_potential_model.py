#!/usr/bin/env python
"""
Run and Validate the New Matric Potential Hybrid Model.

This script:
1. Loads prepared ISMN data (soil_moisture in m³/m³)
2. Converts to matric potential (kPa) using Van Genuchten
3. Trains a simple model to demonstrate the concept
4. Validates performance
"""

from smps.physics.adaptive_calibration import tropical_ptf_van_genuchten
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Import the validated tropical pedotransfer function
sys.path.append(str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_soil_type(latitude: float, longitude: float, clay_percent: float) -> str:
    """
    Detect soil type from coordinates and clay content.

    Simplified classification for tropical soils.
    """
    # Tropical soil classification based on location and clay content
    if -15 <= latitude <= 15:  # Tropical belt
        if clay_percent > 35:
            return "vertisol"  # High clay, shrink-swell
        elif clay_percent > 25:
            return "nitisol"   # Moderate clay, well-weathered
        else:
            return "ferralsol"  # Lower clay, oxide-rich
    else:
        return "ferralsol"  # Default for tropical regions


def estimate_van_genuchten_params(sand_percent: float, clay_percent: float,
                                  organic_matter_percent: float = 2.0,
                                  latitude: float = 0.0, longitude: float = 0.0):
    """
    Use validated tropical pedotransfer function with soil type detection.
    """
    # Detect soil type
    soil_type = detect_soil_type(latitude, longitude, clay_percent)

    # Get parameters using validated tropical PTF
    params = tropical_ptf_van_genuchten(
        sand_percent=sand_percent,
        clay_percent=clay_percent,
        organic_matter_percent=organic_matter_percent,
        soil_type=soil_type
    )

    # Convert to the format expected by the rest of the script
    return {
        'theta_s': params['theta_s'],
        'theta_r': params['theta_r'],
        'alpha': params['alpha'] / 10,  # Convert from 1/m to 1/kPa (approx)
        'n': params['n'],
        'k_sat': params['K_sat']
    }


def water_content_from_potential(psi_kpa: float, params: dict) -> float:
    """
    Calculate volumetric water content from matric potential.
    """
    if psi_kpa >= 0:
        return params['theta_s']

    h = abs(psi_kpa)
    m = 1.0 - 1.0 / params['n']
    se = (1 + (params['alpha'] * h) ** params['n']) ** (-m)
    theta = params['theta_r'] + (params['theta_s'] - params['theta_r']) * se

    return theta


def convert_soil_moisture_to_matric_potential(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert soil_moisture (m³/m³) to matric_potential (kPa).

    Uses pedotransfer functions to estimate VG parameters from soil texture.
    """
    df = df.copy()

    # Group by station to avoid recomputing VG params
    stations = df['station_id'].unique()
    logger.info(f"Converting {len(stations)} stations...")

    for station in stations:
        station_mask = df['station_id'] == station
        station_data = df[station_mask]

        if station_data.empty:
            continue

        # Get soil properties (use median values)
        clay_pct = station_data['clay_pct'].median()
        sand_pct = station_data['sand_pct'].median()
        om_pct = station_data['organic_carbon_pct'].median()
        lat = station_data['latitude'].median()
        lon = station_data['longitude'].median()

        if pd.isna(clay_pct) or pd.isna(sand_pct):
            logger.warning(f"Missing soil data for {station}, skipping")
            continue

        # Estimate VG parameters using validated tropical PTF
        try:
            soil_type = detect_soil_type(lat, lon, clay_pct)
            logger.info(
                f"Station {station}: lat={lat:.2f}, lon={lon:.2f}, clay={clay_pct:.1f}%, detected soil type: {soil_type}")

            vg_params = estimate_van_genuchten_params(
                sand_percent=sand_pct,
                clay_percent=clay_pct,
                organic_matter_percent=om_pct,
                latitude=lat,
                longitude=lon
            )

            # Convert each soil_moisture to matric potential
            theta_values = station_data['soil_moisture'].values
            psi_values = []

            for theta in theta_values:
                if pd.isna(theta):
                    psi_values.append(np.nan)
                else:
                    # Convert m³/m³ to kPa (note: function expects kPa, returns m³/m³)
                    # We need the inverse: theta -> psi
                    # Use numerical inversion since analytical inverse is complex
                    psi = invert_van_genuchten_theta_to_psi(theta, vg_params)
                    psi_values.append(psi)

            df.loc[station_mask, 'matric_potential_kpa'] = psi_values

        except Exception as e:
            logger.error(f"Failed to convert {station}: {e}")
            df.loc[station_mask, 'matric_potential_kpa'] = np.nan

    return df


def invert_van_genuchten_theta_to_psi(theta: float, vg_params) -> float:
    """
    Numerically invert Van Genuchten to get psi from theta.

    Since the analytical inverse is complex, use bisection method.
    """
    if theta >= vg_params['theta_s']:
        return 0.0  # Saturated
    if theta <= vg_params['theta_r']:
        return -10000.0  # Very dry

    # Bisection method
    psi_min = -10000.0  # kPa
    psi_max = 0.0       # kPa

    for _ in range(50):
        psi_mid = (psi_min + psi_max) / 2
        theta_mid = water_content_from_potential(psi_mid, vg_params)

        if theta_mid > theta:
            psi_min = psi_mid
        else:
            psi_max = psi_mid

    return (psi_min + psi_max) / 2


def prepare_features_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data with features and matric potential targets.
    """
    # Basic feature engineering (simplified)
    df = df.copy()

    # Convert date
    df['date'] = pd.to_datetime(df['date'])

    # Sort by station and date
    df = df.sort_values(['station_id', 'date'])

    # Add basic weather features (placeholder - would need actual weather data)
    # For now, just use some dummy features
    df['temp_c'] = 25.0  # Dummy temperature
    df['precip_mm'] = 0.0  # Dummy precip
    df['et0_mm'] = 3.0    # Dummy ET0

    # Add lag features
    df['psi_lag_1d'] = df.groupby('station_id')[
        'matric_potential_kpa'].shift(1)
    df['psi_lag_7d'] = df.groupby('station_id')[
        'matric_potential_kpa'].shift(7)

    return df


def train_and_validate_model(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Train a simple model and validate.
    """
    logger.info("Training simple RandomForest model...")

    # Prepare data
    train_prepared = prepare_features_and_targets(train_df)
    test_prepared = prepare_features_and_targets(test_df)

    # Simple features
    feature_cols = ['temp_c', 'precip_mm',
                    'et0_mm', 'psi_lag_1d', 'psi_lag_7d']
    target_col = 'matric_potential_kpa'

    # Prepare training data
    train_valid = train_prepared.dropna(subset=[target_col] + feature_cols)
    X_train = train_valid[feature_cols].values
    y_train = train_valid[target_col].values

    # Prepare test data
    test_valid = test_prepared.dropna(subset=[target_col] + feature_cols)
    X_test = test_valid[feature_cols].values
    y_test = test_valid[target_col].values

    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("No valid training or test data")
        return

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics in matric potential space
    rmse_psi = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_psi = mean_absolute_error(y_test, y_pred)
    r2_psi = r2_score(y_test, y_pred)

    logger.info("Validation Results (Matric Potential Space):")
    logger.info(f"  Samples: {len(y_test)}")
    logger.info(f"  RMSE: {rmse_psi:.3f} kPa")
    logger.info(f"  MAE: {mae_psi:.3f} kPa")
    logger.info(f"  R²: {r2_psi:.3f}")

    # Convert predictions back to water content for practical validation
    logger.info("Converting predictions back to water content space...")

    # Get VG parameters for each test sample to convert back
    test_valid_copy = test_valid.copy()
    test_valid_copy['psi_pred'] = y_pred
    test_valid_copy['psi_obs'] = y_test

    # Group by station for conversion
    theta_pred_list = []
    theta_obs_list = []

    for station in test_valid_copy['station_id'].unique():
        station_mask = test_valid_copy['station_id'] == station
        station_data = test_valid_copy[station_mask]

        # Get soil properties for this station
        clay_pct = station_data['clay_pct'].iloc[0]
        sand_pct = station_data['sand_pct'].iloc[0]
        om_pct = station_data['organic_carbon_pct'].iloc[0]
        lat = station_data['latitude'].iloc[0]
        lon = station_data['longitude'].iloc[0]

        if pd.isna(clay_pct) or pd.isna(sand_pct):
            # Use default conversion if soil data missing
            theta_pred_list.extend([np.nan] * len(station_data))
            theta_obs_list.extend([np.nan] * len(station_data))
            continue

        # Get VG parameters
        vg_params = estimate_van_genuchten_params(
            sand_percent=sand_pct, clay_percent=clay_pct,
            organic_matter_percent=om_pct, latitude=lat, longitude=lon
        )

        # Convert predictions and observations back to water content
        for psi_pred, psi_obs in zip(station_data['psi_pred'], station_data['psi_obs']):
            theta_pred = water_content_from_potential(psi_pred, vg_params)
            theta_obs_val = water_content_from_potential(psi_obs, vg_params)
            theta_pred_list.append(theta_pred)
            theta_obs_list.append(theta_obs_val)

    # Calculate metrics in water content space
    theta_pred_array = np.array(theta_pred_list)
    theta_obs_array = np.array(theta_obs_list)

    valid_mask = ~(np.isnan(theta_pred_array) | np.isnan(theta_obs_array))
    theta_pred_valid = theta_pred_array[valid_mask]
    theta_obs_valid = theta_obs_array[valid_mask]

    if len(theta_pred_valid) > 0:
        rmse_theta = np.sqrt(mean_squared_error(
            theta_obs_valid, theta_pred_valid))
        mae_theta = mean_absolute_error(theta_obs_valid, theta_pred_valid)
        r2_theta = r2_score(theta_obs_valid, theta_pred_valid)

        logger.info("Validation Results (Water Content Space):")
        logger.info(f"  Samples: {len(theta_pred_valid)}")
        logger.info(f"  RMSE: {rmse_theta:.3f} m³/m³")
        logger.info(f"  MAE: {mae_theta:.3f} m³/m³")
        logger.info(f"  R²: {r2_theta:.3f}")
    else:
        logger.warning("No valid water content conversions for comparison")
        rmse_theta = mae_theta = r2_theta = np.nan

    return {
        'rmse_psi': rmse_psi,
        'mae_psi': mae_psi,
        'r2_psi': r2_psi,
        'rmse_theta': rmse_theta,
        'mae_theta': mae_theta,
        'r2_theta': r2_theta,
        'n_samples': len(y_test)
    }


def main():
    """Main execution."""
    # Load data
    data_dir = Path("data/prepared")
    train_file = data_dir / "ismn_soil_moisture_train.csv"
    test_file = data_dir / "ismn_soil_moisture_test.csv"

    if not train_file.exists() or not test_file.exists():
        logger.error("Data files not found. Run data preparation first.")
        return

    logger.info("Loading training data...")
    train_df = pd.read_csv(train_file)
    logger.info(f"Training data: {len(train_df)} rows")

    logger.info("Loading test data...")
    test_df = pd.read_csv(test_file)
    logger.info(f"Test data: {len(test_df)} rows")

    # Convert to matric potential
    logger.info("Converting soil moisture to matric potential...")
    train_df = convert_soil_moisture_to_matric_potential(train_df)
    test_df = convert_soil_moisture_to_matric_potential(test_df)

    # Check conversion
    valid_train = train_df['matric_potential_kpa'].notna().sum()
    valid_test = test_df['matric_potential_kpa'].notna().sum()
    logger.info(
        f"Valid conversions - Train: {valid_train}/{len(train_df)}, Test: {valid_test}/{len(test_df)}")

    # Train and validate
    results = train_and_validate_model(train_df, test_df)

    if results:
        logger.info("Model validation complete!")
        logger.info("Matric Potential Space: RMSE={:.3f} kPa, MAE={:.3f} kPa, R²={:.3f}".format(
            results['rmse_psi'], results['mae_psi'], results['r2_psi']))
        if not np.isnan(results['rmse_theta']):
            logger.info("Water Content Space: RMSE={:.3f} m³/m³, MAE={:.3f} m³/m³, R²={:.3f}".format(
                results['rmse_theta'], results['mae_theta'], results['r2_theta']))


if __name__ == "__main__":
    main()
