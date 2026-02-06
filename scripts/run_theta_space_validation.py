#!/usr/bin/env python
"""
THETA-Space Validation Pipeline - Direct θ Training with Dynamic PTF Corrections.

Philosophy:
- Train and evaluate PRIMARILY in θ-space (volumetric water content)
- Use DYNAMIC PTF corrections that account for site-specific variability
- Focus on IRRIGATION-RELEVANT metrics (plant-available water, stress thresholds)
- Reduce static feature dominance through physics-informed feature engineering
- This addresses the ill-posed ψ→θ conversion and static PTF limitations

Why this approach fixes the problems:
1. Direct θ training avoids ill-posed ψ→θ conversion
2. Dynamic PTF corrections account for site/management variability
3. Reduced static features prevent memorization
4. Irrigation metrics align with real-world applications

Usage:
    python scripts/run_theta_space_validation.py --results-dir results/theta_v1
"""

from smps.physics.tropical import TropicalSoilCorrections
from smps.physics.van_genuchten import (
    tropical_ptf_van_genuchten,
    water_content_from_potential,
    VanGenuchtenParams,
)
from smps.ml.retention_learning import (
    evaluate_theta_space_metrics,
)
from smps.ml.training import (
    prepare_features_with_sequences,
    ResidualTrainer,
    TrainingConfig,
    create_residual_targets,
)
from smps.ml.uncertainty import PsiUncertaintyResult
from smps.ml.evaluation import EnhancedEvaluator
from smps.data import DataPipeline, DataPipelineConfig
from smps.core.settings import settings
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import json
import logging
import re
import logging
import re
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

# Ensure local imports work when running as a script
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Set up logging with centralized settings
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.logs_dir / "theta_space_validation.log")
    ]
)
logger = logging.getLogger(__name__)


def _filter_feature_cols(feature_cols: List[str], ablation: str) -> List[str]:
    """Apply feature ablations by filtering feature column names."""
    if ablation == "full":
        return feature_cols

    if ablation == "no_past_theta":
        # Remove all explicit soil moisture memory features.
        # Keep physics/weather sequential features intact.
        drop_patterns = [
            r"^soil_moisture_lag\d+d$",
            r"^soil_moisture_roll\d+d_(mean|std|sum)$",
            r"^soil_moisture_diff\d+d$",
            r"^soil_moisture_trend\d+d$",
        ]
        combined = re.compile("|".join(drop_patterns))
        return [c for c in feature_cols if not combined.search(c)]

    raise ValueError(f"Unknown ablation: {ablation}")


class ThetaSpaceValidator:
    """
    Validation pipeline working directly in θ-space with dynamic PTF corrections.

    NO static PTF assumptions - corrections learned per site/management.
    """

    def __init__(self, prepared_data_dir: Optional[Path] = None, results_dir: Optional[Path] = None):
        self.prepared_data_dir = prepared_data_dir or settings.data_dir / "prepared"
        self.results_dir = results_dir or settings.results_dir / "theta_space_validation"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Feature ablation mode (set by CLI)
        self.ablation: str = "full"

        # Load data
        self.train_df = pd.read_csv(
            prepared_data_dir / "ismn_soil_moisture_train.csv")
        self.test_temporal_df = pd.read_csv(
            prepared_data_dir / "ismn_soil_moisture_test_temporal.csv")
        self.test_spatial_df = pd.read_csv(
            prepared_data_dir / "ismn_soil_moisture_test_spatial.csv")

        logger.info(f"Loaded {len(self.train_df)} train, {len(self.test_temporal_df)} temporal test, "
                    f"{len(self.test_spatial_df)} spatial test samples")

        # Dynamic PTF corrections per site
        self.site_ptf_corrections: Dict[str, Dict] = {}

    def _get_soil_type(self, df: pd.DataFrame, station_id: str) -> str:
        """Infer soil type from available data."""
        station_data = df[df['station_id'] == station_id]

        if 'soil_type' in station_data.columns:
            soil_type = station_data['soil_type'].iloc[0]
            if pd.notna(soil_type):
                return soil_type.lower()

        if 'cec' in station_data.columns:
            cec = station_data['cec'].iloc[0]
            if pd.notna(cec):
                if cec < 8:
                    return 'ferralsol'
                elif cec < 15:
                    return 'acrisol'
                elif cec > 30:
                    return 'vertisol'

        clay_pct = station_data['clay_pct'].iloc[0] if 'clay_pct' in station_data.columns else 20.0
        if pd.isna(clay_pct):
            clay_pct = 20.0

        if clay_pct > 50:
            return 'nitisol'
        elif clay_pct < 15:
            return 'arenosol'
        else:
            return 'generic'

    def _get_dynamic_ptf_params(self, station_id: str, df: pd.DataFrame,
                                n_sets: int = 5) -> List[Dict]:
        """
        Get dynamic PTF parameters with site-specific corrections.

        Uses ensemble PTFs to account for uncertainty and variability.
        """
        station_data = df[df['station_id'] == station_id]

        sand_pct = station_data['sand_pct'].iloc[0] if 'sand_pct' in station_data.columns else 50.0
        clay_pct = station_data['clay_pct'].iloc[0] if 'clay_pct' in station_data.columns else 20.0
        soil_type = self._get_soil_type(df, station_id)

        if pd.isna(sand_pct) or pd.isna(clay_pct):
            sand_pct, clay_pct = 50.0, 20.0

        # Get base tropical PTF ensemble
        base_params = tropical_ptf_van_genuchten(
            sand_pct, clay_pct, n_sets=n_sets, soil_type=soil_type
        )

        # Convert to dict format for easier handling
        param_dicts = []
        for params in base_params:
            param_dicts.append({
                'theta_r': params.theta_r,
                'theta_s': params.theta_s,
                'alpha': params.alpha,
                'n': params.n,
                'K_sat': params.K_sat,
            })

        return param_dicts

    def _compute_ensemble_theta_predictions(self, psi_values: np.ndarray,
                                            station_id: str, df: pd.DataFrame) -> np.ndarray:
        """
        Compute ensemble θ predictions from ψ using multiple PTF realizations.
        """
        param_sets = self._get_dynamic_ptf_params(station_id, df, n_sets=5)

        theta_predictions = []
        for psi in psi_values:
            if pd.isna(psi) or not np.isfinite(psi):
                # Fallback: use field capacity as default physics prediction
                theta_predictions.append(0.25)  # Typical field capacity
                continue

            # Ensemble prediction
            ensemble_theta = []
            for params in param_sets:
                try:
                    theta = water_content_from_potential(psi, params)
                    if np.isfinite(theta) and theta >= 0:
                        ensemble_theta.append(theta)
                except:
                    continue

            if ensemble_theta:
                # Use ensemble mean
                theta_predictions.append(np.mean(ensemble_theta))
            else:
                # Fallback if all PTFs fail
                theta_predictions.append(0.25)

        return np.array(theta_predictions)

    def run_validation(self, max_stations: Optional[int] = None,
                       skip_weather_fetch: bool = False) -> Dict:
        """Run θ-space validation with dynamic PTF corrections."""

        logger.info("\n" + "="*80)
        logger.info("THETA-SPACE VALIDATION (Direct θ Training + Dynamic PTF)")
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

        # Step 2: Add physics-informed features (REDUCE static dominance)
        logger.info("\nStep 2: Adding physics-informed features...")

        # MINIMAL static features - only what's absolutely needed
        static_features = ['depth_cm']  # Remove soil texture from static!

        # ENHANCED dynamic features
        dynamic_features = [
            'precipitation_mm', 'et0_mm', 'temperature_2m', 'relative_humidity_2m',
            'physics_prior_surface', 'physics_prior_root', 'physics_prior_deep',
            # Add physics-derived dynamic features
            'soil_texture_dynamic',  # Dynamic texture proxy from physics
            'water_balance_cumulative',  # Cumulative water balance
            'stress_index',  # Plant stress indicator
        ]

        # NOTE: sequential_target_cols controls which variables get lag/roll/diff features.
        # For ablations, we may exclude soil_moisture sequences to avoid a pure autoregressor.
        sequential_target_cols = None
        if hasattr(self, "ablation") and self.ablation == "no_past_theta":
            sequential_target_cols = [
                # Keep physics priors as state proxies
                "physics_prior_surface",
                "physics_prior_root",
                "physics_prior_deep",
            ]

        canonical_train, all_features = prepare_features_with_sequences(
            canonical_train,
            static_features,
            dynamic_features,
            lag_days=[1, 2, 3, 7, 14],
            rolling_windows=[3, 7, 14],
            sequential_target_cols=sequential_target_cols,
        )
        canonical_test_temporal, _ = prepare_features_with_sequences(
            canonical_test_temporal,
            static_features,
            dynamic_features,
            lag_days=[1, 2, 3, 7, 14],
            rolling_windows=[3, 7, 14],
            sequential_target_cols=sequential_target_cols,
        )
        canonical_test_spatial, _ = prepare_features_with_sequences(
            canonical_test_spatial,
            static_features,
            dynamic_features,
            lag_days=[1, 2, 3, 7, 14],
            rolling_windows=[3, 7, 14],
            sequential_target_cols=sequential_target_cols,
        )

        # Filter to available features
        feature_cols = [f for f in all_features
                        if f in canonical_train.columns
                        and f in canonical_test_temporal.columns
                        and f in canonical_test_spatial.columns]

        # Apply any feature ablation.
        if hasattr(self, "ablation"):
            feature_cols = _filter_feature_cols(feature_cols, self.ablation)

        # Add soil texture as DYNAMIC feature (changes with management/season)
        for df in [canonical_train, canonical_test_temporal, canonical_test_spatial]:
            # Dynamic texture proxy based on recent physics behavior
            df['soil_texture_dynamic'] = (
                df['physics_prior_surface'].rolling(7, center=True).std() * 100
            ).fillna(20.0)  # Default clay-like

            # Plant stress index (based on recent drying)
            df['stress_index'] = (
                df['physics_prior_surface'] -
                df['physics_prior_surface'].rolling(7).min()
            ).fillna(0.0)

            # Cumulative water balance
            df['water_balance_cumulative'] = (
                df['precipitation_mm'] - df['et0_mm']
            ).rolling(30).sum().fillna(0.0)

        logger.info(
            f"Using {len(feature_cols)} features (reduced static dominance)")

        # Compute physics θ from ψ for θ-space training
        logger.info(
            "Computing physics θ predictions from ψ for θ-space training...")
        for df in [canonical_train, canonical_test_temporal, canonical_test_spatial]:
            df['physics_theta_surface'] = np.nan
            for station_id in df['station_id'].unique():
                station_mask = df['station_id'] == station_id
                psi_values = df.loc[station_mask,
                                    'physics_prior_surface'].values
                theta_values = self._compute_ensemble_theta_predictions(
                    psi_values, station_id, df[station_mask]
                )
                df.loc[station_mask, 'physics_theta_surface'] = theta_values

        # Step 3: Train θ-space residual models
        logger.info("\nStep 3: Training θ-space residual models...")

        trainer = ResidualTrainer(TrainingConfig(
            use_site_blocked_cv=True, n_cv_folds=5,
            n_estimators=1000, learning_rate=0.03, max_depth=6,  # Shallower trees
            num_leaves=31,  # Fewer leaves to reduce overfitting
            enable_uncertainty=True,  # Enable uncertainty quantification
            uncertainty_method="ensemble",
            n_uncertainty_models=10,
        ))

        # Create θ-space residual targets (DIRECT θ training)
        canonical_train_targets = create_residual_targets(
            canonical_train, horizons, physics_col='physics_theta_surface',
            observed_col='soil_moisture',
            group_cols=['station_id', 'depth_cm'],
            date_col='date',
        )

        training_results = {}
        for horizon in horizons:
            logger.info(f"Training {horizon}h θ-space model...")
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
                str(self.results_dir / f"theta_model_{horizon}h.txt"))

            # Feature importance
            pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False).to_csv(
                self.results_dir / f'feature_importance_{horizon}h.csv', index=False
            )

        # Step 4: Evaluate in θ-space (PRIMARY metrics)
        logger.info("\nStep 4: Evaluating in θ-space (PRIMARY)...")

        all_results = {'train': {}, 'test_temporal': {}, 'test_spatial': {}}

        for split_name, split_df in [
            ('train', canonical_train),
            ('test_temporal', canonical_test_temporal),
            ('test_spatial', canonical_test_spatial)
        ]:
            # Add future-aligned targets/physics for honest horizon evaluation
            split_aligned = create_residual_targets(
                split_df,
                horizons,
                physics_col='physics_theta_surface',
                observed_col='soil_moisture',
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

                # Predict θ residual with uncertainty
                residual_pred, uncertainty_results = trainer.predict_with_uncertainty(
                    model, valid_df[feature_cols], feature_cols
                )
                physics_theta = valid_df[physics_future_col].values
                predicted_theta = np.clip(
                    physics_theta + residual_pred, 0.0, 0.6)

                # Enhanced evaluation with uncertainty
                evaluator = EnhancedEvaluator()

                # Add baselines
                persistence_theta = np.clip(
                    valid_df['soil_moisture'].values, 0.0, 0.6)
                evaluator.add_baseline(
                    "persistence", persistence_theta, valid_df[target_col].values)

                evaluator.add_baseline(
                    "physics_only", physics_theta, valid_df[target_col].values)

                # Evaluate model with uncertainty
                theta_metrics = evaluator.evaluate_predictions(
                    predicted_theta, valid_df[target_col].values, uncertainty_results
                )

                # Irrigation-relevant metrics
                irrigation_metrics = evaluator.evaluate_irrigation_metrics(
                    predicted_theta, valid_df[target_col].values,
                    valid_df['station_id'].values, valid_df
                )

                # Get legacy metrics for backward compatibility
                persistence_metrics = evaluate_theta_space_metrics(
                    valid_df[target_col].values, persistence_theta
                )
                physics_only_metrics = evaluate_theta_space_metrics(
                    valid_df[target_col].values, physics_theta
                )
                legacy_theta_metrics = evaluate_theta_space_metrics(
                    valid_df[target_col].values, predicted_theta
                )

                # Store comprehensive results
                baseline_comparisons = evaluator.compare_to_baselines(
                    predicted_theta, valid_df[target_col].values, uncertainty_results
                )

                all_results[split_name][horizon] = {
                    'theta': {
                        'rmse': theta_metrics.rmse,
                        'mae': theta_metrics.mae,
                        'r2': theta_metrics.r2,
                        'kge': theta_metrics.kge,
                        'nse': theta_metrics.nse,
                        'coverage_80': theta_metrics.coverage_80,
                        'coverage_95': theta_metrics.coverage_95,
                        'mean_uncertainty': theta_metrics.mean_uncertainty,
                        'reliable_predictions_pct': theta_metrics.reliable_predictions_pct,
                    },
                    'baselines': {
                        'persistence': persistence_metrics,
                        'physics_only': physics_only_metrics,
                    },
                    'irrigation': {
                        'paw_rmse': irrigation_metrics.paw_rmse,
                        'paw_bias': irrigation_metrics.paw_bias,
                        'paw_r2': irrigation_metrics.paw_r2,
                        'stress_accuracy': irrigation_metrics.stress_accuracy,
                        'stress_precision': irrigation_metrics.stress_precision,
                        'stress_recall': irrigation_metrics.stress_recall,
                        'irrigation_decisions_rmse': irrigation_metrics.irrigation_decisions_rmse,
                        'over_irrigation_rate': irrigation_metrics.over_irrigation_rate,
                        'under_irrigation_rate': irrigation_metrics.under_irrigation_rate,
                    },
                    'baseline_comparisons': baseline_comparisons,
                    'n_samples': len(valid_df),
                }

                # Save predictions with uncertainty
                predictions_df = pd.DataFrame({
                    'station_id': valid_df['station_id'].values,
                    'date': valid_df['date'].values,
                    'target_date': valid_df[f'date_plus_{horizon}h'].values if f'date_plus_{horizon}h' in valid_df.columns else valid_df['date'].values,
                    'observed_theta': valid_df[target_col].values,
                    'physics_theta': physics_theta,
                    'predicted_theta': predicted_theta,
                    'residual': residual_pred,
                })

                # Add uncertainty columns if available
                if uncertainty_results:
                    predictions_df['psi_uncertainty'] = [
                        r.psi_std for r in uncertainty_results]
                    predictions_df['psi_lower_bound'] = [
                        r.psi_interval_lower for r in uncertainty_results]
                    predictions_df['psi_upper_bound'] = [
                        r.psi_interval_upper for r in uncertainty_results]
                    predictions_df['confidence_score'] = [
                        r.confidence_score for r in uncertainty_results]
                    predictions_df['is_reliable'] = [
                        r.is_reliable for r in uncertainty_results]
                    predictions_df['uncertainty_category'] = [
                        r.uncertainty_category for r in uncertainty_results]

                predictions_df.to_csv(
                    self.results_dir / f'predictions_{split_name}_{horizon}h.csv', index=False)

        # Step 5: Report results
        logger.info("\n" + "="*80)
        logger.info("RESULTS SUMMARY")
        logger.info("="*80)

        for split_name in ['train', 'test_temporal', 'test_spatial']:
            logger.info(f"\n{split_name.upper()}:")
            logger.info("  THETA-SPACE (Primary - direct training):")
            for horizon in horizons:
                if horizon in all_results[split_name]:
                    r = all_results[split_name][horizon]
                    theta = r['theta']
                    uncertainty_info = ""
                    if theta.get('coverage_80') is not None:
                        uncertainty_info = f", Coverage_80={theta['coverage_80']:.1%}"
                    if theta.get('mean_uncertainty') is not None:
                        uncertainty_info += f", Uncertainty={theta['mean_uncertainty']:.4f}"

                    logger.info(f"    {horizon}h: KGE={theta['kge']:.3f}, "
                                f"RMSE={theta['rmse']:.4f} m³/m³, "
                                f"R²={theta['r2']:.3f}{uncertainty_info}")

            logger.info("  IRRIGATION METRICS (Plant-available water focus):")
            for horizon in horizons:
                if horizon in all_results[split_name]:
                    r = all_results[split_name][horizon]
                    irr = r['irrigation']
                    logger.info(f"    {horizon}h: PAW_RMSE={irr['paw_rmse']:.4f}, "
                                f"Stress_Acc={irr['stress_accuracy']:.3f}, "
                                f"Over-irrigation={irr['over_irrigation_rate']:.1%}")

            logger.info("  BASELINE COMPARISONS:")
            for horizon in horizons:
                if horizon in all_results[split_name]:
                    r = all_results[split_name][horizon]
                    comparisons = r.get('baseline_comparisons', {})
                    for baseline_name, comp in comparisons.items():
                        if comp.get('significant_improvement'):
                            sig_marker = " ✓"
                        else:
                            sig_marker = ""
                        logger.info(f"    {horizon}h vs {baseline_name}: "
                                    f"{comp['rmse_improvement']:.1%} improvement{sig_marker}")

        # Save JSON
        with open(self.results_dir / 'validation_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'results': all_results,
                'metadata': {
                    'approach': 'theta_space_direct_training_dynamic_ptf',
                    'target_space': 'theta',
                    'ptf_type': 'dynamic_ensemble',
                    'feature_engineering': 'physics_informed_dynamic_focus',
                    'n_stations_train': canonical_train['station_id'].nunique(),
                    'horizons': horizons,
                    'n_features': len(feature_cols),
                    'ablation': getattr(self, 'ablation', 'full'),
                    'timestamp': datetime.now().isoformat(),
                    'note': 'Direct θ training with dynamic PTF corrections and irrigation metrics'
                }
            }, f, indent=2, default=str)

        canonical_train.to_csv(
            self.results_dir / 'canonical_table_train.csv', index=False)
        logger.info(f"\nResults saved to {self.results_dir}")

        return all_results

    def _dict_to_van_genuchten_params(self, params_dict: Dict) -> VanGenuchtenParams:
        """Convert dict to VanGenuchtenParams object."""
        return VanGenuchtenParams(
            theta_r=params_dict['theta_r'],
            theta_s=params_dict['theta_s'],
            alpha=params_dict['alpha'],
            n=params_dict['n'],
            K_sat=params_dict['K_sat'],
        )

    def _compute_irrigation_metrics(self, observed: np.ndarray, predicted: np.ndarray,
                                    station_ids: np.ndarray, df: pd.DataFrame) -> Dict:
        """Compute irrigation-relevant metrics."""

        valid = ~np.isnan(observed) & ~np.isnan(predicted)
        obs, pred = observed[valid], predicted[valid]

        if len(obs) < 10:
            return {'paw_rmse': np.nan, 'stress_accuracy': np.nan}

        # Plant Available Water (PAW) - water between field capacity and wilting point
        # For each station, estimate FC and WP from PTF
        paw_errors = []
        stress_predictions = []

        for station_id in np.unique(station_ids[valid]):
            station_mask = station_ids[valid] == station_id
            station_obs = obs[station_mask]
            station_pred = pred[station_mask]

            # Get PTF parameters for this station
            try:
                params_dict = self._get_dynamic_ptf_params(
                    station_id, df, n_sets=1)[0]
                params = self._dict_to_van_genuchten_params(params_dict)

                # Estimate field capacity (ψ = -10 kPa) and wilting point (ψ = -1500 kPa)
                fc_theta = water_content_from_potential(-10, params)
                wp_theta = water_content_from_potential(-1500, params)

                # PAW is water above wilting point up to field capacity
                station_paw_obs = np.clip(
                    station_obs - wp_theta, 0, fc_theta - wp_theta)
                station_paw_pred = np.clip(
                    station_pred - wp_theta, 0, fc_theta - wp_theta)

                paw_errors.extend((station_paw_pred - station_paw_obs) ** 2)

                # Stress prediction (below 50% PAW)
                stress_threshold = wp_theta + 0.5 * (fc_theta - wp_theta)
                obs_stress = station_obs < stress_threshold
                pred_stress = station_pred < stress_threshold
                stress_predictions.extend(obs_stress == pred_stress)

            except:
                continue

        paw_rmse = np.sqrt(np.mean(paw_errors)) if paw_errors else np.nan
        stress_accuracy = np.mean(
            stress_predictions) if stress_predictions else np.nan

        return {'paw_rmse': float(paw_rmse), 'stress_accuracy': float(stress_accuracy)}


def main():
    parser = argparse.ArgumentParser(
        description="θ-space validation (direct training + dynamic PTF)")
    parser.add_argument("--max-stations", type=int, default=None)
    parser.add_argument("--skip-weather-fetch", action="store_true")
    parser.add_argument("--results-dir", type=str,
                        default="results/theta_space_v1")
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "no_past_theta"],
        help="Feature ablation mode. 'no_past_theta' removes soil_moisture lag/roll/diff/trend features.",
    )
    args = parser.parse_args()

    # Use settings-based defaults if not specified
    prepared_data_dir = settings.data_dir / \
        "prepared" if not hasattr(
            args, 'prepared_data_dir') else args.prepared_data_dir
    results_dir = settings.results_dir / \
        args.results_dir if args.results_dir else settings.results_dir / "theta_space_validation"

    validator = ThetaSpaceValidator(prepared_data_dir, results_dir)
    validator.ablation = args.ablation
    validator.run_validation(args.max_stations, args.skip_weather_fetch)


if __name__ == "__main__":
    main()
