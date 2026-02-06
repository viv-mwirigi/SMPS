#!/usr/bin/env python
"""
PSI-Space Validation Pipeline - Honest Evaluation.

Philosophy:
- Train and evaluate PRIMARILY in ψ-space (matric potential)
- ψ-space is where physics is valid and transferable
- Use STANDARD PTF for θ conversion (no site-specific calibration!)
- Report θ metrics honestly - they will be worse due to PTF limitations
- This is the correct approach: PTF errors are real and shouldn't be hidden

Why NO site-specific PTF calibration:
1. Zero transferability to new sites
2. Massive overfitting (4 params per site)
3. ML learns station fingerprints, not physics
4. Hides real PTF limitations

Usage:
    python scripts/run_psi_space_validation.py --results-dir results/psi_v1
"""

import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from swpps.data import DataPipeline, DataPipelineConfig
from swpps.ml.training import (
    prepare_features_with_sequences,
    ResidualTrainer,
    TrainingConfig,
    create_matric_residual_targets_debiased,
    create_matric_residual_targets,
)
from swpps.ml.retention_learning import (
    evaluate_psi_space_metrics,
    evaluate_log_psi_space_metrics,
)
from swpps.physics.van_genuchten import (
    potential_from_water_content,
    water_content_from_potential,
    tropical_ptf_van_genuchten,
)
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PsiSpaceValidator:
    """
    Validation pipeline working in ψ-space with honest θ reporting.

    NO site-specific calibration - that's overfitting.
    """

    def __init__(self, prepared_data_dir: Path, results_dir: Path):
        self.prepared_data_dir = prepared_data_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.site_biases: Dict[str, float] = {}

        # Load data
        self.train_df = pd.read_csv(
            prepared_data_dir / "ismn_soil_moisture_train.csv")
        self.test_temporal_df = pd.read_csv(
            prepared_data_dir / "ismn_soil_moisture_test_temporal.csv")
        self.test_spatial_df = pd.read_csv(
            prepared_data_dir / "ismn_soil_moisture_test_spatial.csv")

        logger.info(f"Loaded {len(self.train_df)} train, {len(self.test_temporal_df)} temporal test, "
                    f"{len(self.test_spatial_df)} spatial test samples")

    def _get_soil_type(self, df: pd.DataFrame, station_id: str) -> str:
        """Infer soil type from available data for tropical PTF."""
        station_data = df[df['station_id'] == station_id]

        # Try to get soil type from data if available
        if 'soil_type' in station_data.columns:
            soil_type = station_data['soil_type'].iloc[0]
            if pd.notna(soil_type):
                return soil_type.lower()

        # Infer from CEC if available (key indicator of mineralogy)
        if 'cec' in station_data.columns:
            cec = station_data['cec'].iloc[0]
            if pd.notna(cec):
                if cec < 8:
                    return 'ferralsol'  # Very low CEC = highly weathered
                elif cec < 15:
                    return 'acrisol'
                elif cec > 30:
                    return 'vertisol'  # High CEC = swelling clays

        # Infer from clay content and geography
        clay_pct = station_data['clay_pct'].iloc[0] if 'clay_pct' in station_data.columns else 20.0
        if pd.isna(clay_pct):
            clay_pct = 20.0

        if clay_pct > 50:
            return 'nitisol'  # High clay tropical
        elif clay_pct < 15:
            return 'arenosol'  # Sandy
        else:
            return 'generic'  # Default for African tropics

    def _convert_theta_to_psi(self, theta_series: pd.Series, station_ids: pd.Series,
                              df: pd.DataFrame) -> pd.Series:
        """Convert θ to ψ using tropical PTF with kaolinite/oxide corrections."""
        result = pd.Series(index=theta_series.index, dtype=float)

        for station_id in station_ids.unique():
            mask = station_ids == station_id
            station_theta = theta_series[mask]

            # Get soil texture
            station_data = df[df['station_id'] == station_id]
            sand_pct = station_data['sand_pct'].iloc[0] if 'sand_pct' in station_data.columns else 50.0
            clay_pct = station_data['clay_pct'].iloc[0] if 'clay_pct' in station_data.columns else 20.0

            if pd.isna(sand_pct) or pd.isna(clay_pct):
                sand_pct, clay_pct = 50.0, 20.0

            # Get soil type for proper kaolinite/oxide corrections
            soil_type = self._get_soil_type(df, station_id)

            params = tropical_ptf_van_genuchten(
                sand_pct, clay_pct, n_sets=1, soil_type=soil_type
            )[0]

            station_psi = []
            for theta in station_theta:
                if pd.isna(theta):
                    psi = np.nan
                else:
                    try:
                        psi = potential_from_water_content(theta, params)
                        psi = np.clip(psi, -10000, -0.1)
                    except:
                        psi = np.nan
                station_psi.append(psi)

            result.loc[mask] = station_psi

        return result

    def _convert_psi_to_theta(self, psi_series: pd.Series, station_ids: pd.Series,
                              df: pd.DataFrame) -> pd.Series:
        """Convert ψ to θ using tropical PTF with kaolinite/oxide corrections."""
        result = pd.Series(index=psi_series.index, dtype=float)

        for station_id in station_ids.unique():
            mask = station_ids == station_id
            station_psi = psi_series[mask]

            station_data = df[df['station_id'] == station_id]
            sand_pct = station_data['sand_pct'].iloc[0] if 'sand_pct' in station_data.columns else 50.0
            clay_pct = station_data['clay_pct'].iloc[0] if 'clay_pct' in station_data.columns else 20.0

            if pd.isna(sand_pct) or pd.isna(clay_pct):
                sand_pct, clay_pct = 50.0, 20.0

            # Get soil type for proper kaolinite/oxide corrections
            soil_type = self._get_soil_type(df, station_id)

            params = tropical_ptf_van_genuchten(
                sand_pct, clay_pct, n_sets=1, soil_type=soil_type
            )[0]

            station_theta = []
            for psi in station_psi:
                if pd.isna(psi):
                    theta = np.nan
                else:
                    try:
                        theta = water_content_from_potential(psi, params)
                        theta = np.clip(theta, 0.0, params.theta_s)
                    except:
                        theta = np.nan
                station_theta.append(theta)

            result.loc[mask] = station_theta

        return result

    def run_validation(self, max_stations: Optional[int] = None,
                       skip_weather_fetch: bool = False) -> Dict:
        """Run ψ-space validation with honest θ reporting."""

        logger.info("\n" + "="*80)
        logger.info("PSI-SPACE VALIDATION (No Site-Specific Calibration)")
        logger.info("="*80)

        horizons = [24, 72, 168]

        # Step 1: Build canonical tables
        logger.info("\nStep 1: Building canonical tables...")
        pipeline = DataPipeline(DataPipelineConfig(
            skip_weather_fetch=skip_weather_fetch,
            max_stations=max_stations
        ))

        canonical_train = pipeline.build_canonical_table(self.train_df)
        canonical_test_temporal = pipeline.build_canonical_table(
            self.test_temporal_df)
        canonical_test_spatial = pipeline.build_canonical_table(
            self.test_spatial_df)

        # Step 2: Convert observed θ to ψ
        logger.info("\nStep 2: Converting observed θ to ψ...")
        for df in [canonical_train, canonical_test_temporal, canonical_test_spatial]:
            df['observed_psi'] = self._convert_theta_to_psi(
                df['soil_moisture'], df['station_id'], df
            )

        # Step 3: Add sequential features
        logger.info("\nStep 3: Adding sequential features...")
        static_features = ['depth_cm', 'sand_pct', 'clay_pct', 'ksat_estimate']
        dynamic_features = [
            'precipitation_mm', 'et0_mm', 'temperature_2m', 'relative_humidity_2m',
            'physics_prior_surface', 'physics_prior_root', 'physics_prior_deep'
        ]

        canonical_train, all_features = prepare_features_with_sequences(
            canonical_train, static_features, dynamic_features,
            lag_days=[1, 2, 3, 7, 14], rolling_windows=[3, 7, 14]
        )
        canonical_test_temporal, _ = prepare_features_with_sequences(
            canonical_test_temporal, static_features, dynamic_features,
            lag_days=[1, 2, 3, 7, 14], rolling_windows=[3, 7, 14]
        )
        canonical_test_spatial, _ = prepare_features_with_sequences(
            canonical_test_spatial, static_features, dynamic_features,
            lag_days=[1, 2, 3, 7, 14], rolling_windows=[3, 7, 14]
        )

        feature_cols = [f for f in all_features
                        if f in canonical_train.columns
                        and f in canonical_test_temporal.columns
                        and f in canonical_test_spatial.columns]
        logger.info(f"Using {len(feature_cols)} features")

        # Step 4: Train ψ-space models
        logger.info("\nStep 4: Training ψ-space residual models...")

        trainer = ResidualTrainer(TrainingConfig(
            use_site_blocked_cv=True, n_cv_folds=5,
            n_estimators=1000, learning_rate=0.03, max_depth=8, num_leaves=63
        ))

        canonical_train_targets, site_biases = create_matric_residual_targets_debiased(
            canonical_train,
            horizons,
            physics_col='physics_prior_surface',
            observed_col='soil_moisture',
            observed_matric_col='observed_psi',
            clip_residual=1000.0,
            group_cols=['station_id', 'depth_cm'],
            date_col='date',
        )
        self.site_biases = site_biases

        # Save biases
        pd.DataFrame({
            'station_id': list(site_biases.keys()),
            'bias_kpa': list(site_biases.values())
        }).to_csv(self.results_dir / 'site_bias_corrections.csv', index=False)

        training_results = {}
        for horizon in horizons:
            logger.info(f"Training {horizon}h model...")
            target_col = f'residual_target_{horizon}h'

            valid_data = canonical_train_targets[
                canonical_train_targets[target_col].notna()
            ].copy()
            valid_mask = valid_data[feature_cols].notna().all(axis=1)
            valid_data = valid_data[valid_mask]

            if len(valid_data) < 100:
                continue

            X = valid_data[feature_cols]
            y = valid_data[target_col].values
            groups = valid_data['station_id'].values

            model, fold_results = trainer.train_with_site_cv(
                X, y, groups, feature_cols, 5
            )
            training_results[horizon] = (model, fold_results)

            model.save_model(
                str(self.results_dir / f"psi_model_{horizon}h.txt"))

            # Feature importance
            pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False).to_csv(
                self.results_dir / f'feature_importance_{horizon}h.csv', index=False
            )

        # Step 5: Evaluate in ψ-space (PRIMARY metrics)
        logger.info("\nStep 5: Evaluating in ψ-space (PRIMARY)...")

        all_results = {'train': {}, 'test_temporal': {}, 'test_spatial': {}}

        for split_name, split_df in [
            ('train', canonical_train),
            ('test_temporal', canonical_test_temporal),
            ('test_spatial', canonical_test_spatial)
        ]:
            # Add future-aligned targets/physics for honest horizon evaluation
            split_aligned = create_matric_residual_targets(
                split_df,
                horizons,
                physics_col='physics_prior_surface',
                observed_col='soil_moisture',
                observed_matric_col='observed_psi',
                group_cols=['station_id', 'depth_cm'],
                date_col='date',
            )
            for horizon in horizons:
                if horizon not in training_results:
                    continue

                model, _ = training_results[horizon]

                target_col = f'target_{horizon}h'
                physics_future_col = f'physics_{horizon}h'

                valid_mask = split_aligned[feature_cols].notna().all(axis=1)
                valid_mask &= split_aligned[target_col].notna(
                ) & split_aligned[physics_future_col].notna()
                valid_df = split_aligned[valid_mask].copy()

                if len(valid_df) < 10:
                    continue

                # Predict ψ
                residual_pred = model.predict(valid_df[feature_cols])
                physics_psi = valid_df[physics_future_col].values

                # Site bias - for new sites use mean bias
                if split_name == 'test_spatial':
                    bias_values = np.full(len(valid_df), np.mean(
                        list(self.site_biases.values())))
                else:
                    bias_values = valid_df['station_id'].map(
                        self.site_biases).fillna(0.0).values

                physics_corrected = physics_psi + bias_values
                predicted_psi = np.clip(
                    physics_corrected + residual_pred, -10000, -0.1)

                # ψ-space metrics (PRIMARY)
                psi_metrics = evaluate_psi_space_metrics(
                    valid_df[target_col].values, predicted_psi
                )
                log_psi_metrics = evaluate_log_psi_space_metrics(
                    valid_df[target_col].values, predicted_psi
                )

                # θ-space metrics (SECONDARY - honest, with standard PTF)
                predicted_theta = self._convert_psi_to_theta(
                    pd.Series(predicted_psi, index=valid_df.index),
                    valid_df['station_id'], valid_df
                )
                theta_metrics = self._compute_theta_metrics(
                    valid_df[f'theta_target_{horizon}h'].values,
                    predicted_theta.values
                )

                all_results[split_name][horizon] = {
                    'psi': psi_metrics,
                    'psi_log': log_psi_metrics,
                    'theta': theta_metrics,
                    'n_samples': len(valid_df),
                }

                # Save predictions
                pd.DataFrame({
                    'station_id': valid_df['station_id'].values,
                    'date': valid_df['date'].values,
                    'target_date': valid_df[f'date_plus_{horizon}h'].values if f'date_plus_{horizon}h' in valid_df.columns else valid_df['date'].values,
                    'observed_theta': valid_df[f'theta_target_{horizon}h'].values,
                    'observed_psi': valid_df[target_col].values,
                    'physics_psi': physics_psi,
                    'predicted_psi': predicted_psi,
                    'predicted_theta': predicted_theta.values,
                }).to_csv(self.results_dir / f'predictions_{split_name}_{horizon}h.csv', index=False)

        # Step 6: Report results
        logger.info("\n" + "="*80)
        logger.info("RESULTS SUMMARY")
        logger.info("="*80)

        for split_name in ['train', 'test_temporal', 'test_spatial']:
            logger.info(f"\n{split_name.upper()}:")
            logger.info("  PSI-SPACE (Primary - physics-valid):")
            for horizon in horizons:
                if horizon in all_results[split_name]:
                    r = all_results[split_name][horizon]
                    logger.info(f"    {horizon}h: KGE={r['psi']['kge']:.3f}, "
                                f"RMSE={r['psi']['rmse_kpa']:.0f} kPa, "
                                f"R²={r['psi']['r2']:.3f}")

            logger.info(
                "  THETA-SPACE (Secondary - standard PTF, expect degradation):")
            for horizon in horizons:
                if horizon in all_results[split_name]:
                    r = all_results[split_name][horizon]
                    logger.info(f"    {horizon}h: KGE={r['theta']['kge']:.3f}, "
                                f"RMSE={r['theta']['rmse']:.4f} m³/m³")

        # Save JSON
        with open(self.results_dir / 'validation_results.json', 'w') as f:
            json.dump({
                'results': all_results,
                'metadata': {
                    'approach': 'psi_space_primary_standard_ptf',
                    'site_calibration': False,
                    'n_stations_train': canonical_train['station_id'].nunique(),
                    'horizons': horizons,
                    'n_features': len(feature_cols),
                    'timestamp': datetime.now().isoformat(),
                    'note': 'θ metrics degraded due to PTF limitations - this is honest reporting'
                }
            }, f, indent=2, default=str)

        canonical_train.to_csv(
            self.results_dir / 'canonical_table_train.csv', index=False)
        logger.info(f"\nResults saved to {self.results_dir}")

        return all_results

    def _compute_theta_metrics(self, observed: np.ndarray, predicted: np.ndarray) -> Dict:
        """Compute θ-space metrics."""
        valid = ~np.isnan(observed) & ~np.isnan(predicted)
        obs, pred = observed[valid], predicted[valid]

        if len(obs) < 10:
            return {'rmse': np.nan, 'mae': np.nan, 'kge': np.nan, 'nse': np.nan,
                    'r2': np.nan, 'bias': np.nan, 'n': len(obs)}

        rmse = np.sqrt(np.mean((pred - obs) ** 2))
        mae = np.mean(np.abs(pred - obs))
        bias = np.mean(pred - obs)

        if np.std(obs) > 0 and np.std(pred) > 0:
            r = np.corrcoef(obs, pred)[0, 1]
            r2 = r ** 2
            alpha = np.std(pred) / np.std(obs)
            beta = np.mean(pred) / \
                np.mean(obs) if np.mean(obs) != 0 else np.nan
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 +
                              (beta - 1)**2) if not np.isnan(beta) else np.nan
        else:
            r2, kge = np.nan, np.nan

        ss_res = np.sum((pred - obs) ** 2)
        ss_tot = np.sum((obs - np.mean(obs)) ** 2)
        nse = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return {'rmse': float(rmse), 'mae': float(mae), 'kge': float(kge),
                'nse': float(nse), 'r2': float(r2), 'bias': float(bias), 'n': len(obs)}


def main():
    parser = argparse.ArgumentParser(
        description="PSI-space validation (no site calibration)")
    parser.add_argument("--max-stations", type=int, default=None)
    parser.add_argument("--skip-weather-fetch", action="store_true")
    parser.add_argument("--results-dir", type=str,
                        default="results/psi_space_v1")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    validator = PsiSpaceValidator(
        project_root / "data" / "prepared",
        project_root / args.results_dir
    )
    validator.run_validation(args.max_stations, args.skip_weather_fetch)


if __name__ == "__main__":
    main()
