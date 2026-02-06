"""
Sensitivity Analysis for Matric Potential Models.

Implements comprehensive sensitivity analysis to understand how ψ (matric potential)
predictions respond to changes in input parameters and model assumptions.

Sensitivity Analysis Methods for ψ:
─────────────────────────────────────────────────────────────────
Local Sensitivity:           ∂ψ/∂θ derivatives at specific points
Global Sensitivity:          Variance-based (Sobol) sensitivity indices
Morris Screening:            Qualitative parameter importance ranking
Parameter Importance:        ML feature importance for ψ predictions
Uncertainty Propagation:     Monte Carlo analysis for ψ uncertainty
Scenario Analysis:           ψ response to extreme parameter changes
─────────────────────────────────────────────────────────────────

Benefits for ψ Modeling:
- Identifies most influential parameters on ψ predictions
- Guides model calibration and feature selection
- Quantifies uncertainty in ψ from parameter uncertainty
- Supports model interpretation and validation

Research References:
- Sobol (1993): Global sensitivity indices
- Morris (1991): Screening method for sensitivity analysis
- Saltelli et al. (2008): Global Sensitivity Analysis
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings

logger = logging.getLogger("swpps.analysis.sensitivity")


@dataclass
class SensitivityConfig:
    """Configuration for ψ sensitivity analysis."""

    # Local sensitivity
    derivative_method: str = 'finite_difference'  # finite_difference, automatic
    finite_diff_step: float = 1e-6

    # Global sensitivity
    n_samples: int = 1000  # Number of Monte Carlo samples
    confidence_level: float = 0.95

    # Morris screening
    morris_levels: int = 4
    morris_trajectories: int = 10

    # Uncertainty propagation
    mc_samples: int = 10000
    parameter_distributions: Dict[str, Tuple[str, Any]] = field(default_factory=lambda: {
        'theta_r': ('uniform', (0.0, 0.2)),      # Residual water content
        'theta_s': ('uniform', (0.3, 0.6)),      # Saturated water content
        'alpha': ('lognormal', (0.01, 0.5)),     # Scale parameter (1/kPa)
        'n': ('uniform', (1.1, 5.0)),            # Shape parameter
        'ks': ('lognormal', (0.001, 1.0)),       # Hydraulic conductivity
    })

    # Scenario analysis
    scenario_percentiles: List[float] = field(
        default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95])


class LocalSensitivityAnalysis:
    """
    Local sensitivity analysis using derivatives.

    Computes ∂ψ/∂θ at specific parameter values to understand local behavior.
    """

    def __init__(self, config: SensitivityConfig):
        self.config = config

    def finite_difference_derivative(self, func: callable, x: np.ndarray,
                                     param_idx: int) -> float:
        """Compute derivative using finite differences."""
        h = self.config.finite_diff_step

        # Forward difference
        x_plus = x.copy()
        x_plus[param_idx] += h

        x_minus = x.copy()
        x_minus[param_idx] -= h

        derivative = (func(x_plus) - func(x_minus)) / (2 * h)

        return derivative

    def compute_local_sensitivity(self, model_func: callable, baseline_params: np.ndarray,
                                  param_names: List[str]) -> Dict[str, float]:
        """
        Compute local sensitivity indices for ψ model.

        Args:
            model_func: Function that takes parameters and returns ψ prediction
            baseline_params: Baseline parameter values
            param_names: Names of parameters
        """
        logger.info("Computing local sensitivity analysis")

        sensitivities = {}

        baseline_psi = model_func(baseline_params)

        for i, param_name in enumerate(param_names):
            derivative = self.finite_difference_derivative(
                model_func, baseline_params, i)

            # Normalized sensitivity (dimensionless)
            param_value = baseline_params[i]
            if abs(param_value) > 1e-10:
                normalized_sensitivity = abs(
                    derivative * param_value / baseline_psi)
            else:
                normalized_sensitivity = abs(derivative)

            sensitivities[param_name] = {
                'derivative': derivative,
                'normalized_sensitivity': normalized_sensitivity,
                'absolute_sensitivity': abs(derivative)
            }

        # Sort by importance
        sorted_sensitivities = dict(sorted(
            sensitivities.items(),
            key=lambda x: x[1]['normalized_sensitivity'],
            reverse=True
        ))

        logger.info("Local sensitivity analysis completed")

        return sorted_sensitivities


class GlobalSensitivityAnalysis:
    """
    Global sensitivity analysis using variance-based methods.

    Computes Sobol sensitivity indices to understand parameter importance across
    the entire parameter space.
    """

    def __init__(self, config: SensitivityConfig):
        self.config = config

    def generate_parameter_samples(self, param_ranges: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """Generate parameter samples for sensitivity analysis."""
        samples = {}

        for param_name, (dist_type, params) in self.config.parameter_distributions.items():
            if param_name in param_ranges:
                min_val, max_val = param_ranges[param_name]

                if dist_type == 'uniform':
                    samples[param_name] = np.random.uniform(
                        min_val, max_val, self.config.n_samples)
                elif dist_type == 'normal':
                    mean, std = params
                    samples[param_name] = np.random.normal(
                        mean, std, self.config.n_samples)
                    samples[param_name] = np.clip(
                        samples[param_name], min_val, max_val)
                elif dist_type == 'lognormal':
                    mean, std = params
                    samples[param_name] = np.random.lognormal(
                        mean, std, self.config.n_samples)
                    samples[param_name] = np.clip(
                        samples[param_name], min_val, max_val)

        return pd.DataFrame(samples)

    def compute_sobol_indices(self, model_func: callable, param_samples: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Compute Sobol sensitivity indices.

        Simplified implementation - full Sobol requires specialized sampling.
        """
        logger.info("Computing Sobol sensitivity indices")

        param_names = list(param_samples.columns)
        n_params = len(param_names)

        # Evaluate model at all sample points
        psi_predictions = []
        for _, params in param_samples.iterrows():
            psi = model_func(params.values)
            psi_predictions.append(psi)

        psi_predictions = np.array(psi_predictions)
        total_variance = np.var(psi_predictions)

        sobol_indices = {}

        # First-order effects (simplified)
        for i, param_name in enumerate(param_names):
            # Group by parameter and compute variance
            param_values = param_samples[param_name].values
            unique_values = np.unique(param_values)

            if len(unique_values) > 10:  # Continuous parameter
                # Bin the parameter and compute ANOVA-like decomposition
                bins = np.linspace(np.min(param_values),
                                   np.max(param_values), 11)
                bin_indices = np.digitize(param_values, bins)

                group_variances = []
                group_weights = []

                for bin_idx in range(1, len(bins)):
                    mask = bin_indices == bin_idx
                    if np.sum(mask) > 5:
                        group_psi = psi_predictions[mask]
                        group_var = np.var(group_psi)
                        group_variances.append(group_var)
                        group_weights.append(np.sum(mask))

                if group_variances:
                    # Weighted average of within-group variances
                    within_group_var = np.average(
                        group_variances, weights=group_weights)
                    first_order_effect = total_variance - within_group_var
                    sobol_index = first_order_effect / total_variance if total_variance > 0 else 0
                else:
                    sobol_index = 0.0
            else:
                # Discrete parameter - use ANOVA
                groups = [psi_predictions[param_values == val]
                          for val in unique_values]
                if all(len(g) > 1 for g in groups):
                    f_stat, p_val = stats.f_oneway(*groups)
                    # Convert F-statistic to approximate sensitivity index
                    sobol_index = min(
                        f_stat / (f_stat + len(psi_predictions) - len(unique_values)), 1.0)
                else:
                    sobol_index = 0.0

            # Confidence interval (simplified bootstrap)
            bootstrap_indices = np.random.choice(
                len(psi_predictions), (100, len(psi_predictions)))
            bootstrap_sobol = []

            for boot_idx in bootstrap_indices:
                boot_psi = psi_predictions[boot_idx]
                boot_var = np.var(boot_psi)
                if total_variance > 0:
                    boot_sobol = 1 - np.var([np.mean(boot_psi[param_values[boot_idx] == val])
                                             for val in unique_values if np.sum(param_values[boot_idx] == val) > 0]) / boot_var
                    bootstrap_sobol.append(max(0, min(1, boot_sobol)))

            conf_interval = np.percentile(bootstrap_sobol,
                                          [(1-self.config.confidence_level)/2 * 100,
                                           (1+self.config.confidence_level)/2 * 100])

            sobol_indices[param_name] = {
                'sobol_index': sobol_index,
                'confidence_interval': conf_interval,
                'rank': 0  # Will be set after sorting
            }

        # Rank parameters
        sorted_params = sorted(sobol_indices.items(),
                               key=lambda x: x[1]['sobol_index'], reverse=True)
        for rank, (param_name, _) in enumerate(sorted_params):
            sobol_indices[param_name]['rank'] = rank + 1

        logger.info("Sobol sensitivity indices computed")

        return sobol_indices


class MorrisScreeningAnalysis:
    """
    Morris screening method for qualitative parameter importance ranking.

    Efficient method for identifying important parameters when full global
    sensitivity analysis is too expensive.
    """

    def __init__(self, config: SensitivityConfig):
        self.config = config

    def generate_morris_trajectory(self, param_ranges: Dict[str, Tuple[float, float]]) -> List[np.ndarray]:
        """Generate a Morris screening trajectory."""
        param_names = list(param_ranges.keys())
        n_params = len(param_names)

        # Initialize at random point
        trajectory = []
        current_point = np.array([np.random.uniform(low, high)
                                 for low, high in param_ranges.values()])
        trajectory.append(current_point.copy())

        # Generate trajectory by changing one parameter at a time
        for _ in range(n_params):
            # Choose random parameter to change
            param_idx = np.random.randint(n_params)

            # Choose direction (±1) and step size
            direction = np.random.choice([-1, 1])
            delta = direction * (param_ranges[param_names[param_idx]][1] -
                                 param_ranges[param_names[param_idx]][0]) / (self.config.morris_levels - 1)

            # Update parameter
            current_point[param_idx] += delta
            current_point[param_idx] = np.clip(current_point[param_idx],
                                               param_ranges[param_names[param_idx]][0],
                                               param_ranges[param_names[param_idx]][1])

            trajectory.append(current_point.copy())

        return trajectory

    def compute_morris_indices(self, model_func: callable,
                               param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Dict[str, float]]:
        """
        Compute Morris screening indices (μ and σ).

        μ: Mean effect (overall importance)
        σ: Standard deviation of effects (non-linear/interaction effects)
        """
        logger.info("Computing Morris screening indices")

        param_names = list(param_ranges.keys())
        n_params = len(param_names)

        # Storage for elementary effects
        elementary_effects = {param: [] for param in param_names}

        # Generate trajectories
        for traj_idx in range(self.config.morris_trajectories):
            trajectory = self.generate_morris_trajectory(param_ranges)

            # Evaluate model at trajectory points
            psi_values = [model_func(point) for point in trajectory]

            # Compute elementary effects
            for i in range(len(trajectory) - 1):
                for j, param in enumerate(param_names):
                    if trajectory[i+1][j] != trajectory[i][j]:
                        # Parameter j was changed
                        delta_psi = psi_values[i+1] - psi_values[i]
                        delta_param = trajectory[i+1][j] - trajectory[i][j]

                        # Normalized elementary effect
                        if abs(delta_param) > 1e-10:
                            ee = delta_psi / abs(delta_param)
                            elementary_effects[param].append(ee)
                        break  # Only one parameter changes per step

        # Compute Morris indices
        morris_indices = {}
        for param in param_names:
            effects = np.array(elementary_effects[param])

            if len(effects) > 0:
                mu = np.mean(effects)  # Mean effect
                sigma = np.std(effects)  # Standard deviation

                # Normalize by parameter range for comparability
                param_range = param_ranges[param][1] - param_ranges[param][0]
                mu_star = mu * param_range  # Absolute effect

                morris_indices[param] = {
                    'mu': mu,
                    'mu_star': mu_star,
                    'sigma': sigma,
                    'n_effects': len(effects)
                }
            else:
                morris_indices[param] = {
                    'mu': 0.0,
                    'mu_star': 0.0,
                    'sigma': 0.0,
                    'n_effects': 0
                }

        # Rank parameters by mu_star
        sorted_params = sorted(morris_indices.items(
        ), key=lambda x: abs(x[1]['mu_star']), reverse=True)
        for rank, (param_name, _) in enumerate(sorted_params):
            morris_indices[param_name]['rank'] = rank + 1

        logger.info("Morris screening indices computed")

        return morris_indices


class UncertaintyPropagationAnalysis:
    """
    Uncertainty propagation analysis for ψ predictions.

    Uses Monte Carlo methods to propagate parameter uncertainty to ψ uncertainty.
    """

    def __init__(self, config: SensitivityConfig):
        self.config = config

    def monte_carlo_propagation(self, model_func: callable,
                                param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Perform Monte Carlo uncertainty propagation for ψ.

        Returns statistics of ψ distribution under parameter uncertainty.
        """
        logger.info("Performing Monte Carlo uncertainty propagation")

        param_names = list(param_ranges.keys())

        # Generate Monte Carlo samples
        mc_samples = self.generate_parameter_samples(param_ranges)

        # Evaluate model
        psi_samples = []
        for _, params in mc_samples.iterrows():
            try:
                psi = model_func(params.values)
                psi_samples.append(psi)
            except Exception:
                continue  # Skip invalid parameter combinations

        psi_samples = np.array(psi_samples)

        if len(psi_samples) == 0:
            logger.error("No valid ψ samples generated")
            return {}

        # Compute statistics
        psi_stats = {
            'mean': np.mean(psi_samples),
            'std': np.std(psi_samples),
            'median': np.median(psi_samples),
            'percentiles': {
                p: np.percentile(psi_samples, p*100) for p in self.config.scenario_percentiles
            },
            'cv': np.std(psi_samples) / abs(np.mean(psi_samples)) if np.mean(psi_samples) != 0 else np.inf,
            'n_samples': len(psi_samples)
        }

        # Parameter-ψ correlations
        correlations = {}
        for param_name in param_names:
            param_values = mc_samples[param_name].values[:len(psi_samples)]
            try:
                pearson_corr, _ = pearsonr(param_values, psi_samples)
                spearman_corr, _ = spearmanr(param_values, psi_samples)
                correlations[param_name] = {
                    'pearson': pearson_corr,
                    'spearman': spearman_corr,
                    'abs_pearson': abs(pearson_corr)
                }
            except Exception:
                correlations[param_name] = {
                    'pearson': 0.0,
                    'spearman': 0.0,
                    'abs_pearson': 0.0
                }

        # Sort by correlation strength
        sorted_correlations = dict(sorted(
            correlations.items(),
            key=lambda x: x[1]['abs_pearson'],
            reverse=True
        ))

        result = {
            'psi_statistics': psi_stats,
            'parameter_correlations': sorted_correlations,
            'psi_samples': psi_samples[:1000]  # Store subset for plotting
        }

        logger.info("Monte Carlo uncertainty propagation completed")

        return result

    def generate_parameter_samples(self, param_ranges: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """Generate Monte Carlo parameter samples."""
        samples = {}

        for param_name, (min_val, max_val) in param_ranges.items():
            if param_name in self.config.parameter_distributions:
                dist_type, params = self.config.parameter_distributions[param_name]

                if dist_type == 'uniform':
                    samples[param_name] = np.random.uniform(
                        min_val, max_val, self.config.mc_samples)
                elif dist_type == 'normal':
                    mean, std = params
                    samples[param_name] = np.random.normal(
                        mean, std, self.config.mc_samples)
                    samples[param_name] = np.clip(
                        samples[param_name], min_val, max_val)
                elif dist_type == 'lognormal':
                    mean, std = params
                    samples[param_name] = np.random.lognormal(
                        mean, std, self.config.mc_samples)
                    samples[param_name] = np.clip(
                        samples[param_name], min_val, max_val)
            else:
                # Default to uniform
                samples[param_name] = np.random.uniform(
                    min_val, max_val, self.config.mc_samples)

        return pd.DataFrame(samples)


class SensitivityAnalysisPipeline:
    """
    Complete sensitivity analysis pipeline for ψ models.

    Orchestrates local, global, Morris, and uncertainty propagation analyses.
    """

    def __init__(self, config: Optional[SensitivityConfig] = None):
        self.config = config or SensitivityConfig()

        # Initialize analysis components
        self.local_analysis = LocalSensitivityAnalysis(self.config)
        self.global_analysis = GlobalSensitivityAnalysis(self.config)
        self.morris_analysis = MorrisScreeningAnalysis(self.config)
        self.uncertainty_analysis = UncertaintyPropagationAnalysis(self.config)

    def run_full_sensitivity_analysis(self, model_func: callable,
                                      baseline_params: np.ndarray,
                                      param_names: List[str],
                                      param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Run complete sensitivity analysis suite for ψ model.

        Returns comprehensive sensitivity analysis results.
        """
        logger.info("Running full sensitivity analysis for ψ model")

        results = {}

        # 1. Local sensitivity analysis
        logger.info("Performing local sensitivity analysis")
        results['local_sensitivity'] = self.local_analysis.compute_local_sensitivity(
            model_func, baseline_params, param_names
        )

        # 2. Global sensitivity analysis (Sobol indices)
        logger.info("Performing global sensitivity analysis")
        results['global_sensitivity'] = self.global_analysis.compute_sobol_indices(
            model_func, self.global_analysis.generate_parameter_samples(
                param_ranges)
        )

        # 3. Morris screening
        logger.info("Performing Morris screening analysis")
        results['morris_screening'] = self.morris_analysis.compute_morris_indices(
            model_func, param_ranges
        )

        # 4. Uncertainty propagation
        logger.info("Performing uncertainty propagation analysis")
        results['uncertainty_propagation'] = self.uncertainty_analysis.monte_carlo_propagation(
            model_func, param_ranges
        )

        # 5. Overall parameter ranking
        results['parameter_ranking'] = self._compute_overall_ranking(results)

        logger.info("Full sensitivity analysis completed")

        return results

    def _compute_overall_ranking(self, results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Compute overall parameter importance ranking across all methods."""
        param_scores = {}

        # Collect scores from different methods
        methods = ['local_sensitivity',
                   'global_sensitivity', 'morris_screening']

        for method in methods:
            if method in results:
                method_results = results[method]

                for param_name, param_data in method_results.items():
                    if param_name not in param_scores:
                        param_scores[param_name] = {}

                    # Extract importance score based on method
                    if method == 'local_sensitivity':
                        score = param_data.get('normalized_sensitivity', 0)
                    elif method == 'global_sensitivity':
                        score = param_data.get('sobol_index', 0)
                    elif method == 'morris_screening':
                        score = abs(param_data.get('mu_star', 0))

                    param_scores[param_name][method] = score

        # Compute composite score
        for param_name in param_scores:
            scores = [param_scores[param_name].get(
                method, 0) for method in methods]
            param_scores[param_name]['composite_score'] = np.mean(scores)
            param_scores[param_name]['methods_used'] = len(
                [s for s in scores if s > 0])

        # Rank by composite score
        ranked_params = sorted(param_scores.items(),
                               key=lambda x: x[1]['composite_score'],
                               reverse=True)

        # Add ranking
        overall_ranking = {}
        for rank, (param_name, scores) in enumerate(ranked_params):
            overall_ranking[param_name] = {
                **scores,
                'overall_rank': rank + 1
            }

        return overall_ranking

    def get_sensitivity_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable sensitivity analysis summary."""
        summary = "ψ Model Sensitivity Analysis Summary\n"
        summary += "=" * 45 + "\n\n"

        if 'parameter_ranking' in results:
            ranking = results['parameter_ranking']

            summary += "Parameter Importance Ranking:\n"
            summary += "-" * 30 + "\n"

            for param_name, data in ranking.items():
                rank = data['overall_rank']
                score = data['composite_score']
                methods = data['methods_used']
                summary += f"{rank}. {param_name}: {score:.3f} (methods: {methods})\n"

            summary += "\n"

        if 'uncertainty_propagation' in results:
            up = results['uncertainty_propagation']
            if 'psi_statistics' in up:
                stats = up['psi_statistics']
                summary += "ψ Uncertainty Propagation:\n"
                summary += f"  Mean: {stats['mean']:.3f} kPa\n"
                summary += f"  Std: {stats['std']:.3f} kPa\n"
                summary += f"  CV: {stats['cv']:.3f}\n"
                summary += f"  Range: [{stats['percentiles'][0.05]:.3f}, {stats['percentiles'][0.95]:.3f}] kPa\n\n"

        summary += "Key Insights:\n"
        summary += "- Parameters ranked by overall sensitivity across multiple methods\n"
        summary += "- Uncertainty ranges show prediction confidence under parameter uncertainty\n"
        summary += "- Focus calibration efforts on highest-ranked parameters\n"

        return summary
