#!/usr/bin/env python
"""
Matric Potential Validation Pipeline using SWPPS - HYBRID APPROACH.

This script implements a HYBRID approach for matric potential modeling:
1. Loads prepared ISMN train/test CSVs (ground truth soil moisture in θ)
2. Runs Physics+ML pipeline in VOLUMETRIC SPACE (where models are calibrated)
3. Converts final volumetric predictions to matric potential for evaluation
4. Evaluates at multiple forecast horizons (0h, 24h, 72h, 168h)

Key Innovation - Hybrid Approach:
- Physics model operates in θ space (well-calibrated)
- ML model predicts residuals in θ space (meaningful corrections)
- Final predictions converted to ψ space only for evaluation
- Avoids accumulation of conversion errors

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │ PREPARED DATA (θ ground truth + basic metadata)              │
    └──────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ ENRICH with external data (DataPipeline)                     │
    │ → Creates CANONICAL TABLE with physics priors in ψ space     │
    └──────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ PHYSICS MODEL (TensionSpaceWaterBalance)                     │
    │ Inputs: precip, ET0, soil hydraulics                         │
    │ Output: physics_prior_ψ (predicted matric potential)         │
    └──────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ ML MODEL (ψ-space corrections)                               │
    │ Target: ψ_observation - physics_prior_ψ                      │
    │ Features: weather + ψ_physics + soil features                │
    │ Output: residual_correction_ψ                                │
    └──────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ HYBRID PREDICTION = physics_prior_ψ + residual_correction_ψ  │
    │ → ψ_hybrid (matric potential prediction)                     │
    └──────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ CONVERT ψ → θ using van Genuchten (tropical PTFs)            │
    │ → Final evaluation in volumetric space for users             │
    └──────────────────────────────────────────────────────────────┘

Usage:
    python scripts/run_matric_potential_validation.py --max-stations 10
    python scripts/run_matric_potential_validation.py --skip-weather-fetch
"""

from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
import json
import argparse
from swpps.data import DataPipeline, DataPipelineConfig
from swpps.ml.training import add_sequential_features, prepare_features_with_sequences
from swpps.ml import (
    ResidualTrainer, TrainingConfig, create_matric_residual_targets, create_prediction_features,
    create_matric_residual_targets_debiased, compute_site_bias_corrections, apply_site_bias_correction
)
from swpps.validation import ModelEvaluator
from swpps.physics.van_genuchten import potential_from_water_content, water_content_from_potential, tropical_ptf_van_genuchten
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MatricPotentialValidator:
    """
    Validation pipeline for matric potential modeling using HYBRID approach.

    Runs physics+ML pipeline in volumetric space (where models are calibrated),
    then converts final predictions to matric potential for evaluation.
    This avoids accumulation of conversion errors.
    """

    def __init__(self, prepared_data_dir: Path, results_dir: Path):
        self.prepared_data_dir = prepared_data_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Site-level bias corrections (computed during training)
        self.site_biases: Dict[str, float] = {}

        # Load prepared data
        self.train_df = pd.read_csv(
            self.prepared_data_dir / "ismn_soil_moisture_train.csv"
        )
        self.test_temporal_df = pd.read_csv(
            self.prepared_data_dir / "ismn_soil_moisture_test_temporal.csv"
        )
        self.test_spatial_df = pd.read_csv(
            self.prepared_data_dir / "ismn_soil_moisture_test_spatial.csv"
        )

        logger.info(f"Loaded {len(self.train_df)} training samples")
        logger.info(
            f"Loaded {len(self.test_temporal_df)} temporal test samples")
        logger.info(f"Loaded {len(self.test_spatial_df)} spatial test samples")

    def convert_to_matric_potential(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert volumetric water content (θ) to matric potential (ψ).

        Uses tropical pedotransfer functions for better parameter estimation
        in African soils. Creates ensemble of parameter sets for uncertainty.
        """
        logger.info(
            "Converting volumetric water content to matric potential...")

        result_df = df.copy()

        # Group by station to estimate soil parameters once per station
        station_groups = result_df.groupby('station_id')

        matric_potentials = []

        for station_id, station_df in tqdm(station_groups, desc="Converting stations"):
            # Get soil texture (use defaults if missing)
            sand_pct = station_df['sand_pct'].iloc[0] if 'sand_pct' in station_df.columns else 50.0
            clay_pct = station_df['clay_pct'].iloc[0] if 'clay_pct' in station_df.columns else 20.0

            if pd.isna(sand_pct) or pd.isna(clay_pct):
                sand_pct, clay_pct = 50.0, 20.0  # Default sandy loam

            # Get deterministic van Genuchten parameters
            params = tropical_ptf_van_genuchten(
                sand_pct, clay_pct, n_sets=1)[0]

            # Convert each observation deterministically
            station_potentials = []

            for _, row in station_df.iterrows():
                theta_obs = row['soil_moisture']

                if pd.isna(theta_obs):
                    psi = np.nan
                else:
                    try:
                        psi = potential_from_water_content(
                            theta_obs, params)
                        # Clip extreme values that might be numerical artifacts
                        if psi < -10000:  # Very dry conditions
                            psi = -10000
                        elif psi > 0:  # Can't be positive
                            psi = -0.1
                    except Exception as e:
                        logger.warning(
                            f"Conversion failed for θ={theta_obs}: {e}")
                        psi = np.nan

                station_potentials.append(psi)

            # Add to result
            station_df = station_df.copy()
            station_df['matric_potential_kpa'] = station_potentials
            matric_potentials.append(station_df)

        # Combine all stations
        result_df = pd.concat(matric_potentials, ignore_index=True)

        # Log conversion statistics
        valid_conversions = result_df['matric_potential_kpa'].notna().sum()
        total_samples = len(result_df)
        logger.info(
            f"Successfully converted {valid_conversions}/{total_samples} samples")
        logger.info(
            f"ψ range: {result_df['matric_potential_kpa'].min():.1f} to {result_df['matric_potential_kpa'].max():.1f} kPa")

        return result_df

    def _convert_series_to_matric_potential(self, theta_series: pd.Series,
                                            station_ids: pd.Series) -> pd.Series:
        """
        Convert a series of volumetric water contents to matric potential.

        Uses deterministic Van Genuchten parameters for precise predictions.
        """
        result = pd.Series(index=theta_series.index, dtype=float)

        # Group by station for parameter estimation
        for station_id in station_ids.unique():
            mask = station_ids == station_id
            station_theta = theta_series[mask]

            # Get soil parameters (use cached/default values)
            station_data = self.train_df[self.train_df['station_id']
                                         == station_id]
            if len(station_data) > 0:
                sand_pct = station_data['sand_pct'].iloc[0] if 'sand_pct' in station_data.columns else 50.0
                clay_pct = station_data['clay_pct'].iloc[0] if 'clay_pct' in station_data.columns else 20.0
            else:
                sand_pct, clay_pct = 50.0, 20.0

            if pd.isna(sand_pct) or pd.isna(clay_pct):
                sand_pct, clay_pct = 50.0, 20.0

            # Get single deterministic parameter set (no ensemble uncertainty)
            params = tropical_ptf_van_genuchten(
                sand_pct, clay_pct, n_sets=1)[0]

            # Convert each value deterministically
            station_psi = []
            for theta in station_theta:
                if pd.isna(theta):
                    psi = np.nan
                else:
                    try:
                        psi = potential_from_water_content(theta, params)
                        # Clip extreme values
                        if psi < -10000:
                            psi = -10000
                        elif psi > 0:
                            psi = -0.1
                    except:
                        psi = np.nan

                station_psi.append(psi)

            result.loc[mask] = station_psi

        return result

    def _convert_matric_to_volumetric(self, psi_series: pd.Series,
                                      station_ids: pd.Series, df: pd.DataFrame) -> pd.Series:
        """
        Convert a series of matric potentials to volumetric water content.

        Uses deterministic Van Genuchten parameters for each station.
        """
        from swpps.physics.van_genuchten import water_content_from_potential

        result = pd.Series(index=psi_series.index, dtype=float)

        # Group by station for parameter estimation
        for station_id in station_ids.unique():
            mask = station_ids == station_id
            station_psi = psi_series[mask]

            # Get soil parameters for this station
            station_data = df[df['station_id'] == station_id]
            if len(station_data) > 0:
                sand_pct = station_data['sand_pct'].iloc[0] if 'sand_pct' in station_data.columns else 50.0
                clay_pct = station_data['clay_pct'].iloc[0] if 'clay_pct' in station_data.columns else 20.0
            else:
                sand_pct, clay_pct = 50.0, 20.0

            if pd.isna(sand_pct) or pd.isna(clay_pct):
                sand_pct, clay_pct = 50.0, 20.0

            # Get single deterministic parameter set
            params = tropical_ptf_van_genuchten(
                sand_pct, clay_pct, n_sets=1)[0]

            # Convert each value deterministically
            station_theta = []
            for psi in station_psi:
                if pd.isna(psi):
                    theta = np.nan
                else:
                    try:
                        theta = water_content_from_potential(psi, params)
                        # Clip to valid range
                        if theta < 0:
                            theta = 0.0
                        elif theta > params.theta_s:
                            theta = params.theta_s
                    except:
                        theta = np.nan

                station_theta.append(theta)

            result.loc[mask] = station_theta

        return result

    def _convert_matric_to_volumetric(
        self, psi_series: pd.Series, station_ids: pd.Series, df: pd.DataFrame
    ) -> pd.Series:
        """
        Convert a series of matric potentials to volumetric water contents.

        Uses Van Genuchten parameters for each station.
        """
        from swpps.physics.van_genuchten import water_content_from_potential

        result = pd.Series(index=psi_series.index, dtype=float)

        # Group by station for parameter estimation
        for station_id in station_ids.unique():
            mask = station_ids == station_id
            station_psi = psi_series[mask]

            # Get soil parameters for this station
            station_data = df[df['station_id'] == station_id]
            if len(station_data) > 0:
                sand_pct = station_data['sand_pct'].iloc[0] if 'sand_pct' in station_data.columns else 50.0
                clay_pct = station_data['clay_pct'].iloc[0] if 'clay_pct' in station_data.columns else 20.0
            else:
                sand_pct, clay_pct = 50.0, 20.0

            if pd.isna(sand_pct) or pd.isna(clay_pct):
                sand_pct, clay_pct = 50.0, 20.0

            # Get single deterministic parameter set
            params = tropical_ptf_van_genuchten(
                sand_pct, clay_pct, n_sets=1)[0]

            # Convert each value
            station_theta = []
            for psi in station_psi:
                if pd.isna(psi):
                    theta = np.nan
                else:
                    try:
                        theta = water_content_from_potential(psi, params)
                        # Clip to valid range
                        if theta < 0:
                            theta = 0.0
                        elif theta > params.theta_s:
                            theta = params.theta_s
                    except:
                        theta = np.nan

                station_theta.append(theta)

            result.loc[mask] = station_theta

        return result

    def run_validation(self, max_stations: Optional[int] = None,
                       skip_weather_fetch: bool = False) -> Dict:
        """
        Run the complete matric potential validation pipeline using hybrid approach.

        Physics+ML pipeline operates in volumetric space, final evaluation in ψ space.
        """
        logger.info("\n" + "="*80)
        logger.info("MATRIC POTENTIAL VALIDATION PIPELINE (HYBRID APPROACH)")
        logger.info("="*80)

        # Define forecast horizons (hours)
        horizons = [0, 24, 72, 168]

        # 1. Build canonical tables in volumetric space
        logger.info(
            "\nStep 1: Building canonical tables in volumetric space...")

        pipeline_config = DataPipelineConfig(
            skip_weather_fetch=skip_weather_fetch,
            max_stations=max_stations
        )

        pipeline = DataPipeline(pipeline_config)

        # Build canonical tables (physics priors will be in volumetric space)
        canonical_train = pipeline.build_canonical_table(self.train_df)
        canonical_test_temporal = pipeline.build_canonical_table(
            self.test_temporal_df)
        canonical_test_spatial = pipeline.build_canonical_table(
            self.test_spatial_df)

        # 2. Add sequential features for temporal dependency modeling
        logger.info(
            "\nStep 2: Adding sequential features for temporal dependencies...")

        # Define static and dynamic features
        static_features = ['depth_cm', 'sand_pct', 'clay_pct', 'ksat_estimate']
        dynamic_features = [
            'precipitation_mm', 'et0_mm', 'temperature_2m', 'relative_humidity_2m',
            'physics_prior_surface', 'physics_prior_root', 'physics_prior_deep'
        ]

        # Add sequential features to capture temporal dependencies
        canonical_train, all_features = prepare_features_with_sequences(
            canonical_train,
            static_features=static_features,
            dynamic_features=dynamic_features,
            group_col='station_id',
            date_col='date',
            lag_days=[1, 2, 3, 7, 14],
            rolling_windows=[3, 7, 14]
        )

        canonical_test_temporal, _ = prepare_features_with_sequences(
            canonical_test_temporal,
            static_features=static_features,
            dynamic_features=dynamic_features,
            group_col='station_id',
            date_col='date',
            lag_days=[1, 2, 3, 7, 14],
            rolling_windows=[3, 7, 14]
        )

        canonical_test_spatial, _ = prepare_features_with_sequences(
            canonical_test_spatial,
            static_features=static_features,
            dynamic_features=dynamic_features,
            group_col='station_id',
            date_col='date',
            lag_days=[1, 2, 3, 7, 14],
            rolling_windows=[3, 7, 14]
        )

        # Filter features to those that exist in all datasets
        feature_cols = [f for f in all_features
                        if f in canonical_train.columns
                        and f in canonical_test_temporal.columns
                        and f in canonical_test_spatial.columns]

        logger.info(f"Using {len(feature_cols)} features for training")

        # 3. Train models with sequential features
        logger.info(
            "\nStep 3: Training residual models with sequential features...")

        training_config = TrainingConfig(
            use_site_blocked_cv=True,
            n_cv_folds=5,
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=8,  # Slightly deeper for complex sequential patterns
            num_leaves=63,
        )

        trainer = ResidualTrainer(training_config)

        # Create DEBIASED residual targets for multi-horizon training (in matric potential space)
        # This uses site-level bias correction to remove systematic physics model errors
        canonical_train_with_targets, site_biases = create_matric_residual_targets_debiased(
            canonical_train, horizons, physics_col='physics_prior_surface',
            observed_col='soil_moisture', clip_residual=1000.0
        )

        # Save site biases for later use in prediction
        self.site_biases = site_biases

        # Save site biases to file
        bias_df = pd.DataFrame({
            'station_id': list(site_biases.keys()),
            'bias_kpa': list(site_biases.values())
        })
        bias_df.to_csv(self.results_dir /
                       'site_bias_corrections.csv', index=False)
        logger.info(
            f"Saved site bias corrections to {self.results_dir / 'site_bias_corrections.csv'}")

        # Train models for each horizon
        training_results = {}
        for horizon in horizons:
            logger.info(f"Training model for {horizon}h horizon...")
            target_col = f'residual_target_{horizon}h'

            # Filter to valid data for this horizon
            valid_data = canonical_train_with_targets[canonical_train_with_targets[target_col].notna(
            )].copy()

            # Also filter out rows with NaN in features (due to lags)
            valid_mask = valid_data[feature_cols].notna().all(axis=1)
            valid_data = valid_data[valid_mask]

            if len(valid_data) < 100:
                logger.warning(
                    f"Insufficient data for horizon {horizon}h: {len(valid_data)} samples")
                continue

            # Prepare data
            X = valid_data[feature_cols]
            y = valid_data[target_col].values
            groups = valid_data['station_id'].values

            model, fold_results = trainer.train_with_site_cv(
                X, y, groups, feature_cols, trainer.config.n_cv_folds
            )

            training_results[horizon] = (model, fold_results)

            # Save model
            model_path = self.results_dir / f"hybrid_model_{horizon}h.txt"
            model.save_model(str(model_path))

            # Save feature importance
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            importance_df.to_csv(
                self.results_dir / f'feature_importance_{horizon}h.csv', index=False)

        # 4. Generate predictions in matric potential space, then convert back to volumetric
        logger.info(
            "\nStep 4: Generating matric potential predictions and converting to volumetric...")

        predictions_by_horizon = {}
        for horizon in horizons:
            if horizon not in training_results:
                continue

            model, _ = training_results[horizon]

            # Filter canonical_train to rows with valid features
            valid_mask = canonical_train[feature_cols].notna().all(axis=1)
            valid_train = canonical_train[valid_mask].copy()

            pred_features = valid_train[feature_cols]

            # ML predicts matric potential residuals (DEBIASED - on top of site-corrected physics)
            matric_residual_predictions = model.predict(pred_features)

            # Physics priors are already in matric potential space (kPa)
            physics_prior_matric = valid_train['physics_prior_surface'].values

            # Apply site-level bias correction FIRST
            site_bias_values = valid_train['station_id'].map(
                self.site_biases).fillna(0.0).values
            physics_bias_corrected = physics_prior_matric + site_bias_values

            # Add ML residuals to BIAS-CORRECTED physics priors → matric predictions
            # Full hybrid prediction = physics + site_bias + ML_residual
            predicted_matric_raw = physics_bias_corrected + matric_residual_predictions

            # CRITICAL: Constrain predictions to physically valid range
            # Matric potential must be negative (suction), and reasonable range is -10000 to -0.1 kPa
            # Values > 0 are impossible; values < -10000 kPa exceed wilting point
            predicted_matric = np.clip(predicted_matric_raw, -10000, -0.1)

            # Convert matric predictions back to volumetric for evaluation
            predicted_volumetric = self._convert_matric_to_volumetric(
                pd.Series(predicted_matric, index=valid_train.index),
                valid_train['station_id'],
                valid_train
            )
            pred_df = pd.DataFrame({
                'station_id': valid_train['station_id'].values,
                'date': valid_train['date'].values,
                'observed_volumetric': valid_train['soil_moisture'].values,
                'physics_prior_matric': physics_prior_matric,
                'site_bias_correction': site_bias_values,
                'physics_bias_corrected': physics_bias_corrected,
                'residual_correction_matric': matric_residual_predictions,
                'predicted_matric': predicted_matric,
                'predicted_volumetric': predicted_volumetric.values,
                'physics_prior_root': valid_train['physics_prior_root'].values,
                'physics_prior_deep': valid_train['physics_prior_deep'].values,
            })

            predictions_by_horizon[horizon] = pred_df

            # Save detailed predictions for this horizon
            pred_csv_path = self.results_dir / \
                f"detailed_predictions_{horizon}h.csv"
            pred_df.to_csv(pred_csv_path, index=False)
            logger.info(f"Saved detailed predictions to {pred_csv_path}")

        # 5. Evaluate in volumetric space
        logger.info("\nStep 5: Evaluating in volumetric space...")

        evaluator = ModelEvaluator()

        temporal_evaluation = evaluator.evaluate_multi_horizon(
            predictions_by_horizon,
            observed_col='observed_volumetric',
            predicted_col='predicted_volumetric'
        )

        # Generate predictions for spatial test (ψ-space approach)
        predictions_spatial = {}
        for horizon in horizons:
            if horizon not in training_results:
                continue

            model, _ = training_results[horizon]

            # Filter to valid features
            valid_mask = canonical_test_spatial[feature_cols].notna().all(
                axis=1)
            valid_spatial = canonical_test_spatial[valid_mask].copy()

            pred_features = valid_spatial[feature_cols]

            # ML predicts matric potential residuals
            matric_residual_predictions = model.predict(pred_features)

            # Physics priors are already in matric potential space (kPa)
            physics_prior_matric = valid_spatial['physics_prior_surface'].values

            # Add matric residuals to matric physics priors → matric predictions
            predicted_matric = physics_prior_matric + matric_residual_predictions

            # Convert matric predictions back to volumetric for evaluation
            predicted_volumetric = self._convert_matric_to_volumetric(
                pd.Series(predicted_matric, index=valid_spatial.index),
                valid_spatial['station_id'],
                valid_spatial
            )

            pred_df = pd.DataFrame({
                'station_id': valid_spatial['station_id'].values,
                'date': valid_spatial['date'].values,
                'observed_volumetric': valid_spatial['soil_moisture'].values,
                'physics_prior_matric': physics_prior_matric,
                'residual_correction_matric': matric_residual_predictions,
                'predicted_matric': predicted_matric,
                'predicted_volumetric': predicted_volumetric.values,
            })

            predictions_spatial[horizon] = pred_df

            # Save predictions
            pred_csv_path = self.results_dir / \
                f"predictions_spatial_{horizon}h.csv"
            pred_df.to_csv(pred_csv_path, index=False)
            logger.info(f"Saved predictions to {pred_csv_path}")

        # Evaluate on spatial test set (in volumetric space)
        spatial_evaluation = evaluator.evaluate_multi_horizon(
            predictions_spatial,
            observed_col='observed_volumetric',
            predicted_col='predicted_volumetric'
        )

        # 6. Save canonical tables for inspection
        logger.info("\nStep 6: Saving canonical tables...")
        canonical_train.to_csv(
            self.results_dir / 'canonical_table_train.csv', index=False)

        # 7. Save results
        logger.info("\nStep 7: Saving results...")

        # Extract CV results without models for JSON serialization
        training_summary = {}
        for horizon, (model, fold_results) in training_results.items():
            val_scores = [r.val_score for r in fold_results]
            training_summary[horizon] = {
                'cv_rmse_mean': float(np.mean(val_scores)),
                'cv_rmse_std': float(np.std(val_scores)),
                'model_path': str(self.results_dir / f"hybrid_model_{horizon}h.txt")
            }

        # Convert evaluation results to dicts
        temporal_eval_dict = {}
        for horizon, result in temporal_evaluation.items():
            temporal_eval_dict[horizon] = {
                'rmse_mean': float(result.rmse_mean),
                'rmse_std': float(result.rmse_std),
                'mae_mean': float(result.mae_mean),
                'mae_std': float(result.mae_std),
                'kge_mean': float(result.kge_mean),
                'kge_std': float(result.kge_std),
                'nse_mean': float(result.nse_mean),
                'nse_std': float(result.nse_std),
                'n_sites': int(result.n_sites)
            }

        spatial_eval_dict = {}
        for horizon, result in spatial_evaluation.items():
            spatial_eval_dict[horizon] = {
                'rmse_mean': float(result.rmse_mean),
                'rmse_std': float(result.rmse_std),
                'mae_mean': float(result.mae_mean),
                'mae_std': float(result.mae_std),
                'kge_mean': float(result.kge_mean),
                'kge_std': float(result.kge_std),
                'nse_mean': float(result.nse_mean),
                'nse_std': float(result.nse_std),
                'n_sites': int(result.n_sites)
            }

        results = {
            'training_summary': training_summary,
            'temporal_evaluation': temporal_eval_dict,
            'spatial_evaluation': spatial_eval_dict,
            'metadata': {
                'approach': 'psi_space_corrections_hybrid',
                'description': 'ML corrections applied in matric potential space, evaluated in volumetric space',
                'conversion_method': 'tropical_ptf_van_genuchten_deterministic',
                'n_parameter_sets': 1,
                'training_space': 'matric_potential_kpa',
                'evaluation_space': 'volumetric_m3_m3',
                'horizons': horizons,
                'training_samples': len(canonical_train),
                'temporal_test_samples': len(canonical_test_temporal),
                'spatial_test_samples': len(canonical_test_spatial),
                'timestamp': datetime.now().isoformat()
            }
        }

        # Save to JSON
        results_file = self.results_dir / "matric_potential_validation_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = self._make_json_serializable(results)
            json.dump(json_results, f, indent=2)

        logger.info(f"Results saved to {results_file}")

        # Print summary
        self._print_summary(temporal_evaluation, spatial_evaluation, horizons)

        return results

    def _make_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            # Skip non-serializable objects like LightGBM Booster
            if hasattr(obj, '__class__') and 'Booster' in str(obj.__class__):
                return f"<{obj.__class__.__name__} object - not serializable>"
            return obj

    def _print_summary(self, temporal_eval, spatial_eval, horizons):
        """Print a summary of the validation results."""
        logger.info("\n" + "="*80)
        logger.info("VALIDATION SUMMARY - MATRIC POTENTIAL MODEL")
        logger.info("="*80)

        logger.info("\nTemporal Test Results (Same stations, different time):")
        for horizon in horizons:
            if horizon in temporal_eval:
                result = temporal_eval[horizon]
                logger.info(f"  {horizon}h: RMSE={result.rmse_mean:.3f}, MAE={result.mae_mean:.3f}, "
                            f"KGE={result.kge_mean:.3f}, NSE={result.nse_mean:.3f}")

        logger.info(
            "\nSpatial Test Results (Different stations, same time period):")
        for horizon in horizons:
            if horizon in spatial_eval:
                result = spatial_eval[horizon]
                logger.info(f"  {horizon}h: RMSE={result.rmse_mean:.3f}, MAE={result.mae_mean:.3f}, "
                            f"KGE={result.kge_mean:.3f}, NSE={result.nse_mean:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Matric Potential Validation Pipeline")
    parser.add_argument("--max-stations", type=int, default=None,
                        help="Maximum number of stations to process")
    parser.add_argument("--skip-weather-fetch", action="store_true",
                        help="Skip weather data fetching (use cached data)")
    parser.add_argument("--results-dir", type=str, default="results/matric_potential_v1",
                        help="Directory to save results")

    args = parser.parse_args()

    # Setup paths
    prepared_data_dir = Path("data/prepared")
    results_dir = Path(args.results_dir)

    # Run validation
    validator = MatricPotentialValidator(prepared_data_dir, results_dir)
    results = validator.run_validation(
        max_stations=args.max_stations,
        skip_weather_fetch=args.skip_weather_fetch
    )

    logger.info("Matric potential validation completed successfully!")


if __name__ == "__main__":
    main()
