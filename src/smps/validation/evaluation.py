"""
Evaluation utilities for SWPPS models.

Provides comprehensive evaluation including:
- Per-site performance metrics
- Multi-horizon evaluation
- Cross-validation results
- Model comparison
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from smps.validation.metrics import ValidationMetrics, compute_metrics

logger = logging.getLogger("swpps.validation.evaluation")


@dataclass
class SiteEvaluationResult:
    """Evaluation results for a single site."""
    station_id: str
    n_samples: int
    n_valid: int

    # Standard metrics
    rmse: float
    mae: float
    r2: float
    bias: float

    # Efficiency metrics
    nse: float
    kge: float

    # KGE components
    kge_r: float
    kge_alpha: float
    kge_beta: float

    # Sample info
    depth_cm: Optional[float] = None
    clay_pct: Optional[float] = None
    sand_pct: Optional[float] = None

    @classmethod
    def from_metrics(
        cls,
        station_id: str,
        metrics: ValidationMetrics,
        depth_cm: Optional[float] = None,
        clay_pct: Optional[float] = None,
        sand_pct: Optional[float] = None
    ) -> "SiteEvaluationResult":
        """Create from ValidationMetrics object."""
        return cls(
            station_id=station_id,
            n_samples=metrics.n_samples,
            n_valid=metrics.n_valid,
            rmse=metrics.rmse,
            mae=metrics.mae,
            r2=metrics.r_squared,
            bias=metrics.mbe,
            nse=metrics.nse,
            kge=metrics.kge,
            kge_r=metrics.kge_r,
            kge_alpha=metrics.kge_alpha,
            kge_beta=metrics.kge_beta,
            depth_cm=depth_cm,
            clay_pct=clay_pct,
            sand_pct=sand_pct,
        )


@dataclass
class HorizonEvaluationResult:
    """Evaluation results for a forecast horizon."""
    horizon_hours: int
    n_sites: int
    n_samples_total: int

    # Aggregated metrics (mean across sites)
    rmse_mean: float
    rmse_std: float
    mae_mean: float
    mae_std: float
    r2_mean: float
    r2_std: float
    bias_mean: float
    bias_std: float

    # Efficiency metrics
    nse_mean: float
    nse_std: float
    kge_mean: float
    kge_std: float

    # Per-site results
    site_results: List[SiteEvaluationResult]


class ModelEvaluator:
    """
    Comprehensive model evaluation for soil moisture prediction.

    Supports:
    - Per-site evaluation
    - Multi-horizon evaluation
    - Cross-validation results
    - Model comparison
    """

    def __init__(self):
        self.results: Dict[str, Any] = {}

    def evaluate_per_site(
        self,
        model_predictions: pd.DataFrame,
        observed_col: str = "soil_moisture",
        predicted_col: str = "predicted",
        groupby_cols: Optional[List[str]] = None,
    ) -> List[SiteEvaluationResult]:
        """
        Evaluate model performance per site.

        Args:
            model_predictions: DataFrame with predictions and observations
            observed_col: Column name for observed values
            predicted_col: Column name for predicted values
            groupby_cols: Columns to group by (default: ['station_id'])

        Returns:
            List of SiteEvaluationResult for each site
        """
        if groupby_cols is None:
            groupby_cols = ['station_id']

        site_results = []

        for group_values, group_df in model_predictions.groupby(groupby_cols):
            if isinstance(group_values, str):
                station_id = group_values
            else:
                station_id = "_".join(str(v) for v in group_values)

            # Get valid data
            valid_mask = (
                group_df[observed_col].notna() &
                group_df[predicted_col].notna()
            )
            obs = group_df.loc[valid_mask, observed_col].values
            pred = group_df.loc[valid_mask, predicted_col].values

            if len(obs) < 5:  # Minimum samples for meaningful evaluation
                logger.warning(
                    f"Insufficient data for {station_id}: {len(obs)} samples")
                continue

            # Compute metrics
            metrics = compute_metrics(obs, pred)

            # Get site metadata
            depth_cm = group_df['depth_cm'].iloc[0] if 'depth_cm' in group_df.columns else None
            clay_pct = group_df['clay_pct'].iloc[0] if 'clay_pct' in group_df.columns else None
            sand_pct = group_df['sand_pct'].iloc[0] if 'sand_pct' in group_df.columns else None

            site_result = SiteEvaluationResult.from_metrics(
                station_id=station_id,
                metrics=metrics,
                depth_cm=depth_cm,
                clay_pct=clay_pct,
                sand_pct=sand_pct,
            )

            site_results.append(site_result)

        logger.info(f"Evaluated {len(site_results)} sites")
        return site_results

    def evaluate_multi_horizon(
        self,
        predictions_by_horizon: Dict[int, pd.DataFrame],
        observed_col: str = "target_soil_moisture",
        predicted_col: str = "predicted",
    ) -> Dict[int, HorizonEvaluationResult]:
        """
        Evaluate model across multiple forecast horizons.

        Args:
            predictions_by_horizon: Dict mapping horizon (hours) to predictions DataFrame
            observed_col: Column name for observed values
            predicted_col: Column name for predicted values

        Returns:
            Dict mapping horizon to HorizonEvaluationResult
        """
        horizon_results = {}

        for horizon, df in predictions_by_horizon.items():
            # Evaluate per site
            site_results = self.evaluate_per_site(
                df,
                observed_col=observed_col,
                predicted_col=predicted_col,
            )

            if not site_results:
                logger.warning(f"No valid results for horizon {horizon}h")
                continue

            # Aggregate across sites
            rmse_values = [
                r.rmse for r in site_results if not np.isnan(r.rmse)]
            mae_values = [r.mae for r in site_results if not np.isnan(r.mae)]
            r2_values = [r.r2 for r in site_results if not np.isnan(r.r2)]
            bias_values = [
                r.bias for r in site_results if not np.isnan(r.bias)]
            nse_values = [r.nse for r in site_results if not np.isnan(r.nse)]
            kge_values = [r.kge for r in site_results if not np.isnan(r.kge)]

            horizon_result = HorizonEvaluationResult(
                horizon_hours=horizon,
                n_sites=len(site_results),
                n_samples_total=sum(r.n_valid for r in site_results),
                rmse_mean=np.mean(rmse_values) if rmse_values else np.nan,
                rmse_std=np.std(rmse_values) if rmse_values else np.nan,
                mae_mean=np.mean(mae_values) if mae_values else np.nan,
                mae_std=np.std(mae_values) if mae_values else np.nan,
                r2_mean=np.mean(r2_values) if r2_values else np.nan,
                r2_std=np.std(r2_values) if r2_values else np.nan,
                bias_mean=np.mean(bias_values) if bias_values else np.nan,
                bias_std=np.std(bias_values) if bias_values else np.nan,
                nse_mean=np.mean(nse_values) if nse_values else np.nan,
                nse_std=np.std(nse_values) if nse_values else np.nan,
                kge_mean=np.mean(kge_values) if kge_values else np.nan,
                kge_std=np.std(kge_values) if kge_values else np.nan,
                site_results=site_results,
            )

            horizon_results[horizon] = horizon_result

            logger.info(
                f"Horizon {horizon}h: {len(site_results)} sites, "
                f"RMSE={horizon_result.rmse_mean:.3f}±{horizon_result.rmse_std:.3f}, "
                f"KGE={horizon_result.kge_mean:.3f}±{horizon_result.kge_std:.3f}"
            )

        return horizon_results

    def compare_models(
        self,
        model_results: Dict[str, Dict[int, HorizonEvaluationResult]],
        metrics: List[str] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple models across horizons.

        Args:
            model_results: Dict mapping model names to horizon results
            metrics: List of metrics to compare (default: ['rmse', 'kge', 'nse'])

        Returns:
            DataFrame with model comparison
        """
        if metrics is None:
            metrics = ['rmse_mean', 'kge_mean', 'nse_mean']

        comparison_data = []

        for model_name, horizon_results in model_results.items():
            for horizon, result in horizon_results.items():
                row = {
                    'model': model_name,
                    'horizon_h': horizon,
                    'n_sites': result.n_sites,
                }

                for metric in metrics:
                    if hasattr(result, metric):
                        row[metric] = getattr(result, metric)

                comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def save_evaluation_results(
        self,
        results: Dict[int, HorizonEvaluationResult],
        output_dir: Path,
        prefix: str = "evaluation",
    ) -> None:
        """
        Save evaluation results to files.

        Args:
            results: Horizon evaluation results
            output_dir: Output directory
            prefix: File prefix
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Summary table
        summary_data = []
        for horizon, result in results.items():
            summary_data.append({
                'horizon_h': horizon,
                'n_sites': result.n_sites,
                'n_samples': result.n_samples_total,
                'rmse_mean': result.rmse_mean,
                'rmse_std': result.rmse_std,
                'mae_mean': result.mae_mean,
                'mae_std': result.mae_std,
                'r2_mean': result.r2_mean,
                'r2_std': result.r2_std,
                'bias_mean': result.bias_mean,
                'bias_std': result.bias_std,
                'nse_mean': result.nse_mean,
                'nse_std': result.nse_std,
                'kge_mean': result.kge_mean,
                'kge_std': result.kge_std,
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / f"{prefix}_summary.csv", index=False)

        # Per-site results
        all_site_results = []
        for horizon, result in results.items():
            for site_result in result.site_results:
                row = {
                    'horizon_h': horizon,
                    'station_id': site_result.station_id,
                    'n_samples': site_result.n_samples,
                    'n_valid': site_result.n_valid,
                    'rmse': site_result.rmse,
                    'mae': site_result.mae,
                    'r2': site_result.r2,
                    'bias': site_result.bias,
                    'nse': site_result.nse,
                    'kge': site_result.kge,
                    'kge_r': site_result.kge_r,
                    'kge_alpha': site_result.kge_alpha,
                    'kge_beta': site_result.kge_beta,
                    'depth_cm': site_result.depth_cm,
                    'clay_pct': site_result.clay_pct,
                    'sand_pct': site_result.sand_pct,
                }
                all_site_results.append(row)

        site_df = pd.DataFrame(all_site_results)
        site_df.to_csv(output_dir / f"{prefix}_per_site.csv", index=False)

        logger.info(f"Saved evaluation results to {output_dir}")


def print_evaluation_summary(results: Dict[int, HorizonEvaluationResult]) -> None:
    """
    Print a formatted summary of evaluation results.

    Args:
        results: Horizon evaluation results
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 80)

    for horizon in sorted(results.keys()):
        result = results[horizon]
        print(f"\nHorizon: {horizon} hours")
        print(f"  Sites: {result.n_sites}")
        print(f"  Total samples: {result.n_samples_total:,}")
        print(f"  RMSE: {result.rmse_mean:.3f} ± {result.rmse_std:.3f}")
        print(f"  MAE:  {result.mae_mean:.3f} ± {result.mae_std:.3f}")
        print(f"  R²:   {result.r2_mean:.3f} ± {result.r2_std:.3f}")
        print(f"  Bias: {result.bias_mean:.3f} ± {result.bias_std:.3f}")
        print(f"  NSE:  {result.nse_mean:.3f} ± {result.nse_std:.3f}")
        print(f"  KGE:  {result.kge_mean:.3f} ± {result.kge_std:.3f}")

    print("\n" + "=" * 80)
