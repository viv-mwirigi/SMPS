"""
Enhanced Evaluation with Uncertainty Quantification.

Provides comprehensive evaluation metrics including:
- Traditional ML metrics (RMSE, R², KGE)
- Uncertainty-aware metrics (coverage, sharpness)
- Irrigation-relevant metrics (plant-available water, stress detection)
- Model reliability assessment
- Baseline comparisons with statistical significance
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    # Traditional metrics
    rmse: float
    mae: float
    r2: float
    kge: float  # Kling-Gupta Efficiency
    nse: float  # Nash-Sutcliffe Efficiency

    # Uncertainty metrics
    coverage_80: Optional[float] = None
    coverage_95: Optional[float] = None
    mean_uncertainty: Optional[float] = None
    uncertainty_sharpness: Optional[float] = None

    # Reliability metrics
    reliable_predictions_pct: Optional[float] = None
    high_confidence_accuracy: Optional[float] = None

    # Sample statistics
    n_samples: int = 0
    mean_observed: float = 0.0
    mean_predicted: float = 0.0


@dataclass
class IrrigationMetrics:
    """Irrigation-relevant evaluation metrics."""
    # Plant-available water metrics
    paw_rmse: float  # RMSE in plant-available water range
    paw_bias: float  # Bias in PAW range
    paw_r2: float    # R² in PAW range

    # Stress detection metrics
    stress_accuracy: float     # Accuracy in detecting water stress
    stress_precision: float    # Precision for stress detection
    stress_recall: float       # Recall for stress detection

    # Irrigation decision metrics
    irrigation_decisions_rmse: float  # RMSE for irrigation-relevant decisions
    over_irrigation_rate: float       # Rate of over-irrigation recommendations
    under_irrigation_rate: float      # Rate of under-irrigation recommendations


class EnhancedEvaluator:
    """Enhanced evaluator with uncertainty quantification and irrigation metrics."""

    def __init__(self):
        self.baselines = {}

    def add_baseline(self, name: str, predictions: np.ndarray, actuals: np.ndarray,
                     uncertainties: Optional[List] = None):
        """Add a baseline model for comparison."""
        metrics = self.evaluate_predictions(
            predictions, actuals, uncertainties)
        self.baselines[name] = {
            "metrics": metrics,
            "predictions": predictions,
            "actuals": actuals,
        }

    def evaluate_predictions(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        uncertainties: Optional[List] = None,
        threshold_stress: float = 0.15,  # θ threshold for water stress
        threshold_paw: Tuple[float, float] = (0.10, 0.35),  # PAW range
    ) -> EvaluationMetrics:
        """
        Evaluate predictions with comprehensive metrics.

        Args:
            predictions: Model predictions
            actuals: Observed values
            uncertainties: Optional uncertainty results
            threshold_stress: θ threshold below which plants experience stress
            threshold_paw: (min, max) θ range for plant-available water
        """

        # Basic metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = np.mean(np.abs(actuals - predictions))
        r2 = r2_score(actuals, predictions)

        # Kling-Gupta Efficiency
        kge = self._calculate_kge(predictions, actuals)

        # Nash-Sutcliffe Efficiency
        nse = 1 - np.sum((actuals - predictions)**2) / \
            np.sum((actuals - np.mean(actuals))**2)

        # Uncertainty metrics
        coverage_80 = None
        coverage_95 = None
        mean_uncertainty = None
        uncertainty_sharpness = None
        reliable_predictions_pct = None
        high_confidence_accuracy = None

        if uncertainties:
            # Coverage metrics
            if hasattr(uncertainties[0], 'psi_interval_lower') and hasattr(uncertainties[0], 'psi_interval_upper'):
                lower_bounds = np.array(
                    [u.psi_interval_lower for u in uncertainties])
                upper_bounds = np.array(
                    [u.psi_interval_upper for u in uncertainties])
                coverage_80 = np.mean(
                    (actuals >= lower_bounds) & (actuals <= upper_bounds))

                # 95% coverage using 2-sigma rule
                wider_lower = predictions - 2 * \
                    np.array([u.psi_std for u in uncertainties])
                wider_upper = predictions + 2 * \
                    np.array([u.psi_std for u in uncertainties])
                coverage_95 = np.mean(
                    (actuals >= wider_lower) & (actuals <= wider_upper))

            # Uncertainty statistics
            uncertainties_vals = np.array([u.psi_std for u in uncertainties])
            mean_uncertainty = np.mean(uncertainties_vals)
            uncertainty_sharpness = np.mean(
                uncertainties_vals)  # Lower is sharper

            # Reliability metrics
            if hasattr(uncertainties[0], 'is_reliable'):
                reliable_mask = np.array(
                    [u.is_reliable for u in uncertainties])
                reliable_predictions_pct = np.mean(reliable_mask) * 100

                if np.any(reliable_mask):
                    reliable_errors = np.abs(
                        actuals[reliable_mask] - predictions[reliable_mask])
                    high_confidence_accuracy = np.mean(
                        reliable_errors < rmse)  # Better than mean error

        return EvaluationMetrics(
            rmse=rmse,
            mae=mae,
            r2=r2,
            kge=kge,
            nse=nse,
            coverage_80=coverage_80,
            coverage_95=coverage_95,
            mean_uncertainty=mean_uncertainty,
            uncertainty_sharpness=uncertainty_sharpness,
            reliable_predictions_pct=reliable_predictions_pct,
            high_confidence_accuracy=high_confidence_accuracy,
            n_samples=len(actuals),
            mean_observed=np.mean(actuals),
            mean_predicted=np.mean(predictions),
        )

    def evaluate_irrigation_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        station_ids: Optional[np.ndarray] = None,
        df: Optional[pd.DataFrame] = None,
        threshold_stress: float = 0.15,
        threshold_paw: Tuple[float, float] = (0.10, 0.35),
        threshold_irrigation: float = 0.20,  # θ threshold for irrigation decision
    ) -> IrrigationMetrics:
        """
        Evaluate irrigation-relevant metrics.

        Args:
            predictions: Model predictions (θ)
            actuals: Observed θ values
            station_ids: Station identifiers for grouping
            df: Original dataframe with additional context
            threshold_stress: θ below which plants are stressed
            threshold_paw: (min, max) θ for plant-available water
            threshold_irrigation: θ threshold for irrigation decisions
        """

        # Plant-available water metrics (focus on 0.10-0.35 m³/m³ range)
        paw_min, paw_max = threshold_paw
        paw_mask = (actuals >= paw_min) & (actuals <= paw_max)

        if np.any(paw_mask):
            paw_rmse = np.sqrt(mean_squared_error(
                actuals[paw_mask], predictions[paw_mask]))
            paw_bias = np.mean(predictions[paw_mask] - actuals[paw_mask])
            paw_r2 = r2_score(actuals[paw_mask], predictions[paw_mask])
        else:
            paw_rmse = np.sqrt(mean_squared_error(actuals, predictions))
            paw_bias = np.mean(predictions - actuals)
            paw_r2 = r2_score(actuals, predictions)

        # Stress detection metrics
        actual_stress = actuals < threshold_stress
        predicted_stress = predictions < threshold_stress

        stress_accuracy = np.mean(actual_stress == predicted_stress)
        if np.any(predicted_stress):
            stress_precision = np.mean(actual_stress[predicted_stress])
        else:
            stress_precision = 0.0
        if np.any(actual_stress):
            stress_recall = np.mean(predicted_stress[actual_stress])
        else:
            stress_recall = 0.0

        # Irrigation decision metrics
        actual_needs_irrigation = actuals < threshold_irrigation
        predicted_needs_irrigation = predictions < threshold_irrigation

        irrigation_errors = np.abs(actuals - predictions)
        irrigation_decisions_rmse = np.sqrt(np.mean(irrigation_errors**2))

        # Over/under irrigation rates
        over_irrigation = np.sum(
            (predicted_needs_irrigation) & (~actual_needs_irrigation))
        under_irrigation = np.sum(
            (~predicted_needs_irrigation) & (actual_needs_irrigation))
        total_decisions = len(actuals)

        over_irrigation_rate = over_irrigation / \
            total_decisions if total_decisions > 0 else 0.0
        under_irrigation_rate = under_irrigation / \
            total_decisions if total_decisions > 0 else 0.0

        return IrrigationMetrics(
            paw_rmse=paw_rmse,
            paw_bias=paw_bias,
            paw_r2=paw_r2,
            stress_accuracy=stress_accuracy,
            stress_precision=stress_precision,
            stress_recall=stress_recall,
            irrigation_decisions_rmse=irrigation_decisions_rmse,
            over_irrigation_rate=over_irrigation_rate,
            under_irrigation_rate=under_irrigation_rate,
        )

    def compare_to_baselines(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        uncertainties: Optional[List] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare model performance to baselines with statistical significance."""

        model_metrics = self.evaluate_predictions(
            predictions, actuals, uncertainties)

        comparisons = {}
        for name, baseline in self.baselines.items():
            baseline_metrics = baseline["metrics"]

            # Statistical significance tests
            rmse_improvement = (baseline_metrics.rmse -
                                model_metrics.rmse) / baseline_metrics.rmse

            # Paired t-test for RMSE differences
            model_errors = actuals - predictions
            baseline_errors = baseline["actuals"] - baseline["predictions"]

            if len(model_errors) == len(baseline_errors):
                t_stat, p_value = stats.ttest_rel(
                    np.abs(model_errors), np.abs(baseline_errors)
                )
                significant_improvement = p_value < 0.05 and rmse_improvement > 0
            else:
                p_value = None
                significant_improvement = None

            comparisons[name] = {
                "rmse_improvement": rmse_improvement,
                "p_value": p_value,
                "significant_improvement": significant_improvement,
                "model_rmse": model_metrics.rmse,
                "baseline_rmse": baseline_metrics.rmse,
            }

        return comparisons

    def _calculate_kge(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate Kling-Gupta Efficiency."""
        mean_obs = np.mean(actuals)
        mean_pred = np.mean(predictions)

        r = np.corrcoef(actuals, predictions)[0, 1]
        alpha = np.std(predictions) / np.std(actuals)
        beta = mean_pred / mean_obs

        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        return kge

    def generate_evaluation_report(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        uncertainties: Optional[List] = None,
        station_ids: Optional[np.ndarray] = None,
        df: Optional[pd.DataFrame] = None,
        model_name: str = "Model",
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""

        # Core metrics
        metrics = self.evaluate_predictions(
            predictions, actuals, uncertainties)

        # Irrigation metrics
        irrigation_metrics = self.evaluate_irrigation_metrics(
            predictions, actuals, station_ids, df
        )

        # Baseline comparisons
        baseline_comparisons = self.compare_to_baselines(
            predictions, actuals, uncertainties)

        # Station-wise performance (if station_ids provided)
        station_performance = None
        if station_ids is not None:
            unique_stations = np.unique(station_ids)
            station_metrics = []

            for station in unique_stations:
                mask = station_ids == station
                if np.sum(mask) >= 10:  # Minimum samples for reliable metrics
                    station_preds = predictions[mask]
                    station_actuals = actuals[mask]
                    station_met = self.evaluate_predictions(
                        station_preds, station_actuals)
                    station_irr = self.evaluate_irrigation_metrics(
                        station_preds, station_actuals)

                    station_metrics.append({
                        "station_id": station,
                        "n_samples": station_met.n_samples,
                        "rmse": station_met.rmse,
                        "r2": station_met.r2,
                        "paw_rmse": station_irr.paw_rmse,
                        "stress_accuracy": station_irr.stress_accuracy,
                    })

            if station_metrics:
                station_performance = sorted(
                    station_metrics, key=lambda x: x["rmse"])

        return {
            "model_name": model_name,
            "metrics": {
                "rmse": metrics.rmse,
                "mae": metrics.mae,
                "r2": metrics.r2,
                "kge": metrics.kge,
                "nse": metrics.nse,
                "coverage_80": metrics.coverage_80,
                "coverage_95": metrics.coverage_95,
                "mean_uncertainty": metrics.mean_uncertainty,
                "reliable_predictions_pct": metrics.reliable_predictions_pct,
            },
            "irrigation_metrics": {
                "paw_rmse": irrigation_metrics.paw_rmse,
                "paw_bias": irrigation_metrics.paw_bias,
                "paw_r2": irrigation_metrics.paw_r2,
                "stress_accuracy": irrigation_metrics.stress_accuracy,
                "stress_precision": irrigation_metrics.stress_precision,
                "stress_recall": irrigation_metrics.stress_recall,
                "irrigation_decisions_rmse": irrigation_metrics.irrigation_decisions_rmse,
                "over_irrigation_rate": irrigation_metrics.over_irrigation_rate,
                "under_irrigation_rate": irrigation_metrics.under_irrigation_rate,
            },
            "baseline_comparisons": baseline_comparisons,
            "station_performance": station_performance,
            "summary": {
                "n_samples": metrics.n_samples,
                "mean_observed": metrics.mean_observed,
                "mean_predicted": metrics.mean_predicted,
                "best_station": station_performance[0] if station_performance else None,
                "worst_station": station_performance[-1] if station_performance else None,
            }
        }
