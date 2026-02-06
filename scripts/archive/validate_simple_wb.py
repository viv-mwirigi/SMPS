#!/usr/bin/env python
"""
Validate simple water balance model on ISMN data.
"""

import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse
import time

import pandas as pd
import numpy as np
from tqdm import tqdm

from smps.data.sources.ismn_loader import (
    ISMNStationLoader,
    ISMNStationData,
    get_daily_soil_moisture
)
from smps.data.sources.weather import OpenMeteoSource
from smps.physics.simple_water_balance import (
    SimpleWaterBalance,
    create_simple_config,
)

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("smps.validation.simple_wb")


def calculate_metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Calculate validation metrics."""
    valid = ~(np.isnan(obs) | np.isnan(pred))
    obs_v, pred_v = obs[valid], pred[valid]
    n = len(obs_v)
    if n < 10:
        return {'N': n, 'error': 'Insufficient'}

    errors = pred_v - obs_v
    bias = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    r = float(np.corrcoef(obs_v, pred_v)[0, 1]) if np.std(
        obs_v) > 1e-6 else 0.0

    ss_res = np.sum((obs_v - pred_v) ** 2)
    ss_tot = np.sum((obs_v - np.mean(obs_v)) ** 2)
    nse = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else -999

    alpha = np.std(pred_v) / np.std(obs_v) if np.std(obs_v) > 1e-6 else 1.0
    beta = np.mean(pred_v) / \
        np.mean(obs_v) if abs(np.mean(obs_v)) > 1e-6 else 1.0
    kge = float(1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2))

    return {
        'N': n, 'Bias': bias, 'RMSE': rmse, 'R': r, 'NSE': nse, 'KGE': kge,
        'obs_mean': float(np.mean(obs_v)), 'pred_mean': float(np.mean(pred_v)),
    }


class SimpleWBValidator:
    def __init__(self, ismn_data_dir, output_dir, start_date="2019-01-01", end_date="2021-12-31"):
        self.ismn_data_dir = Path(ismn_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.ismn_loader = ISMNStationLoader(ismn_data_dir)
        self.weather_source = OpenMeteoSource(
            cache_dir=self.output_dir / "cache")
        self.results = []

    def run(self, networks=None, max_stations=None):
        logger.info(
            f"Starting simple WB validation: {self.start_date.date()} to {self.end_date.date()}")

        if networks:
            station_data_list = []
            for network in networks:
                stations_dict = self.ismn_loader.load_network(network)
                station_data_list.extend(stations_dict.values())
        else:
            stations_dict = self.ismn_loader.load_all_stations()
            station_data_list = list(stations_dict.values())

        valid_stations = [s for s in station_data_list if s.daily_data is not None
                          and len(s.daily_data[(s.daily_data['date'] >= self.start_date) &
                                               (s.daily_data['date'] <= self.end_date)]) >= 100]
        if max_stations:
            valid_stations = valid_stations[:max_stations]

        logger.info(f"Processing {len(valid_stations)} stations")

        for station in tqdm(valid_stations, desc="Validating"):
            try:
                self._process_station(station)
            except Exception as e:
                logger.error(f"Failed {station.station_id}: {e}")
            time.sleep(0.3)

        self._save_and_print()

    def _process_station(self, station_data):
        station_id = station_data.station_id
        obs_dict = get_daily_soil_moisture(
            station_data, self.start_date, self.end_date)
        if not obs_dict:
            return

        all_dates = [d for series in obs_dict.values() for d in series.index]
        start = max(self.start_date, pd.Timestamp(min(all_dates)))
        end = min(self.end_date, pd.Timestamp(max(all_dates)))

        forcings = self.weather_source.get_daily_weather(
            station_data.latitude, station_data.longitude, start.to_pydatetime(), end.to_pydatetime()
        )
        if forcings is None or len(forcings) < 50:
            return

        sand, clay = self._get_texture(station_data)

        for depth_cm, obs_series in obs_dict.items():
            if len(obs_series) < 50:
                continue

            obs_mean = obs_series.mean()
            logger.info(
                f"  {station_id} @ {depth_cm}cm ({len(obs_series)} obs, mean={obs_mean:.3f})")

            pred = self._run_model(
                forcings, sand, clay, depth_cm, station_data.latitude, obs_mean=obs_mean)
            if pred is None:
                continue

            merged = obs_series.to_frame('observed').join(
                pred.to_frame('predicted'), how='inner').dropna()
            if len(merged) < 30:
                continue

            metrics = calculate_metrics(
                merged['observed'].values, merged['predicted'].values)
            self.results.append({
                'station_id': station_id, 'depth_cm': depth_cm,
                'lat': station_data.latitude, 'lon': station_data.longitude,
                'sand': sand, 'clay': clay, **metrics
            })

    def _get_texture(self, station_data):
        sand, clay = None, None
        if station_data.soil_properties:
            for props in station_data.soil_properties.values():
                if props.sand_fraction is not None:
                    sand = props.sand_fraction * 100
                if props.clay_fraction is not None:
                    clay = props.clay_fraction * 100

        # Use regional defaults for tropical Africa if no data
        # These are conservative loamy sand values
        if sand is None:
            sand = 55.0  # Slightly sandy for tropical soils
            logger.warning(f"  Using default sand={sand}%")
        if clay is None:
            clay = 20.0  # Moderate clay
            logger.warning(f"  Using default clay={clay}%")
        return sand, clay

    def _run_model(self, forcings, sand, clay, depth_cm, lat, obs_mean=None):
        try:
            # Get vegetation fraction from NDVI if available, else skip station
            if 'ndvi' in forcings.columns:
                ndvi_mean = forcings['ndvi'].mean()
                if np.isnan(ndvi_mean):
                    logger.warning(f"    No valid NDVI data")
                    return None
                veg_frac = np.clip((ndvi_mean - 0.1) / 0.8, 0.1, 0.9)
            else:
                # No NDVI - cannot determine vegetation, use conservative estimate
                logger.warning(f"    No NDVI data, using lat-based estimate")
                veg_frac = 0.5 if abs(lat) < 15 else 0.4

            config = create_simple_config(
                sand, clay, depth_cm/100, 3, 1.0, veg_frac)
            model = SimpleWaterBalance(config)

            # Initialize at observed mean if provided, otherwise use a conservative value
            if obs_mean is not None and not np.isnan(obs_mean):
                init_theta = np.clip(obs_mean, 0.05, 0.45)
                model.set_initial_conditions([init_theta] * 3)
                logger.debug(f"    Initialized at θ={init_theta:.3f}")

            dates = pd.to_datetime(forcings.index)
            precip = forcings.get('precipitation_mm', forcings.get(
                'precipitation', pd.Series(0, index=dates)))
            et0 = forcings.get('et0_mm', forcings.get('eto_mm', None))
            if et0 is None:
                tmax = forcings.get('temperature_max_c',
                                    pd.Series(30, index=dates))
                tmin = forcings.get('temperature_min_c',
                                    pd.Series(20, index=dates))
                et0 = 0.0023 * 15 * (tmax + 17.8) * \
                    np.sqrt(np.maximum(0.1, tmax - tmin))
            ndvi = forcings.get('ndvi', None)

            preds = []
            for i, d in enumerate(dates):
                ndvi_val = float(ndvi.iloc[i]) if ndvi is not None else None
                _, theta = model.run_daily(
                    float(precip.iloc[i]), float(et0.iloc[i]), ndvi_val)
                preds.append({'date': d, 'theta': theta})
            return pd.Series([p['theta'] for p in preds], index=[p['date'] for p in preds])
        except Exception as e:
            logger.warning(f"    Model error: {e}")
            return None

    def _save_and_print(self):
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(self.output_dir / 'validation_results.csv', index=False)

            print("\n" + "="*60)
            print("SIMPLE WATER BALANCE VALIDATION")
            print("="*60)
            print(f"Stations: {df['station_id'].nunique()}, Pairs: {len(df)}")
            for m in ['RMSE', 'Bias', 'R', 'NSE', 'KGE']:
                if m in df:
                    v = df[m].dropna()
                    print(f"{m:>8}: {v.mean():.4f} ± {v.std():.4f}")
            print(
                f"Obs mean: {df['obs_mean'].mean():.3f}, Pred mean: {df['pred_mean'].mean():.3f}")
            print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir', default='data/ismn/Data_separate_files_header_20170105_20250105_12892_F2PyW_20260105')
    parser.add_argument('--network', default='TAHMO')
    parser.add_argument('--max-stations', type=int)
    parser.add_argument('--output-dir', default='results/simple_wb_validation')
    args = parser.parse_args()

    validator = SimpleWBValidator(args.data_dir, args.output_dir)
    validator.run([args.network] if args.network else None, args.max_stations)


if __name__ == '__main__':
    main()
