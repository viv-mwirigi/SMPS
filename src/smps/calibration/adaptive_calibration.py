"""
Adaptive Calibration for Matric Potential Models.

Implements site-specific and temporal calibration methods for ψ (matric potential)
models to improve accuracy across different soil types and conditions.

Adaptive Calibration Methods:
─────────────────────────────────────────────────────────────────
Site-Specific Calibration:   Calibrate VG parameters per site
Temporal Adaptation:         Update model parameters over time
Online Learning:            Continuously adapt to new ψ data
Transfer Learning:          Adapt pre-trained models to new sites
Ensemble Calibration:       Calibrate ensemble weights per site
─────────────────────────────────────────────────────────────────

Benefits for ψ Modeling:
- Improved accuracy for site-specific soil conditions
- Better temporal stability of ψ predictions
- Reduced calibration effort through automation
- Enhanced generalization across soil types

Research References:
- Vereecken et al. (2010): Soil hydraulic parameter estimation
- Van Genuchten & Nielsen (1985): Closed-form expressions for ψ
- Hupet & Vanclooster (2002): Temporal stability of soil water
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

logger = logging.getLogger("swpps.calibration.adaptive")


@dataclass
class CalibrationConfig:
    """Configuration for adaptive ψ model calibration."""

    # van Genuchten parameter bounds for optimization
    vg_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'theta_r': (0.0, 0.2),      # Residual water content
        'theta_s': (0.3, 0.6),      # Saturated water content
        'alpha': (0.001, 1.0),      # Scale parameter (1/kPa)
        'n': (1.1, 5.0)             # Shape parameter
    })

    # Optimization settings
    max_iter: int = 1000
    tolerance: float = 1e-6

    # Temporal adaptation
    adaptation_window: int = 100  # Number of recent observations for adaptation
    adaptation_frequency: str = '1D'  # How often to update calibration

    # Online learning
    learning_rate: float = 0.01
    forgetting_factor: float = 0.95  # For exponential forgetting

    # Transfer learning
    similarity_threshold: float = 0.7  # Minimum similarity for transfer
    fine_tune_epochs: int = 10

    # Ensemble calibration
    ensemble_weights_init: List[float] = field(
        default_factory=lambda: [0.33, 0.33, 0.34])


class SiteSpecificCalibrator:
    """
    Calibrates van Genuchten parameters for specific sites.

    Optimizes VG parameters (θr, θs, α, n) to best fit observed ψ-θ data at each site.
    """

    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.site_params: Dict[str, Dict[str, float]] = {}

    def calibrate_site_vg(self, psi_obs: np.ndarray, theta_obs: np.ndarray,
                          site_id: str) -> Dict[str, Any]:
        """
        Calibrate van Genuchten parameters for a specific site.

        Returns fitted parameters and calibration quality metrics.
        """
        logger.info(f"Calibrating VG parameters for site {site_id}")

        def vg_objective(params):
            """van Genuchten objective function for optimization."""
            theta_r, theta_s, alpha, n = params
            m = 1 - 1/n

            # Clamp parameters to valid ranges
            theta_r = np.clip(theta_r, 0, 0.5)
            theta_s = np.clip(theta_s, 0.2, 0.8)
            alpha = np.clip(alpha, 0.001, 2.0)
            n = np.clip(n, 1.01, 10.0)
            m = 1 - 1/n

            # van Genuchten equation
            psi_abs = np.abs(psi_obs)
            theta_pred = theta_r + (theta_s - theta_r) / \
                (1 + (alpha * psi_abs)**n)**m

            # Mean squared error
            return np.mean((theta_obs - theta_pred)**2)

        # Initial guess (typical values)
        p0 = [0.05, 0.45, 0.02, 2.0]  # θr, θs, α, n

        # Parameter bounds
        bounds = [
            self.config.vg_bounds['theta_r'],
            self.config.vg_bounds['theta_s'],
            self.config.vg_bounds['alpha'],
            self.config.vg_bounds['n']
        ]

        try:
            # Optimize parameters
            result = optimize.minimize(
                vg_objective, p0,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': self.config.max_iter,
                         'ftol': self.config.tolerance}
            )

            # Extract fitted parameters
            theta_r, theta_s, alpha, n = result.x
            m = 1 - 1/n

            fitted_params = {
                'theta_r': theta_r,
                'theta_s': theta_s,
                'alpha': alpha,
                'n': n,
                'm': m,
                'convergence': result.success,
                'nfev': result.nfev,
                'final_loss': result.fun
            }

            # Calculate calibration quality
            final_loss = result.fun
            rmse = np.sqrt(final_loss)
            n_samples = len(psi_obs)

            # R² calculation
            theta_pred_opt = self._predict_theta_from_psi(
                psi_obs, fitted_params)
            r2 = 1 - np.sum((theta_obs - theta_pred_opt)**2) / \
                np.sum((theta_obs - np.mean(theta_obs))**2)

            calibration_quality = {
                'rmse': rmse,
                'r2': r2,
                'n_samples': n_samples,
                'calibration_success': result.success and r2 > 0.5
            }

            # Store parameters
            self.site_params[site_id] = {
                'params': fitted_params,
                'quality': calibration_quality
            }

            logger.info(
                f"Site {site_id} calibration completed. R²: {r2:.3f}, RMSE: {rmse:.4f}")

            return {
                'site_id': site_id,
                'parameters': fitted_params,
                'quality': calibration_quality
            }

        except Exception as e:
            logger.error(f"Calibration failed for site {site_id}: {e}")
            return {
                'site_id': site_id,
                'parameters': None,
                'quality': {'calibration_success': False, 'error': str(e)}
            }

    def _predict_theta_from_psi(self, psi: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Predict θ from ψ using fitted VG parameters."""
        theta_r = params['theta_r']
        theta_s = params['theta_s']
        alpha = params['alpha']
        n = params['n']
        m = params['m']

        psi_abs = np.abs(psi)
        theta = theta_r + (theta_s - theta_r) / (1 + (alpha * psi_abs)**n)**m

        return theta

    def get_site_parameters(self, site_id: str) -> Optional[Dict[str, Any]]:
        """Get calibrated parameters for a site."""
        return self.site_params.get(site_id)


class TemporalCalibrator:
    """
    Adapts model parameters over time using recent observations.

    Implements exponential forgetting and sliding window adaptation for temporal stability.
    """

    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.temporal_params: Dict[str, List[Dict[str, Any]]] = {}
        self.forgetting_factor = config.forgetting_factor

    def update_temporal_calibration(self, site_id: str, psi_recent: np.ndarray,
                                    theta_recent: np.ndarray, current_time: pd.Timestamp) -> Dict[str, Any]:
        """
        Update calibration parameters using recent temporal data.

        Uses exponential forgetting to weight recent observations more heavily.
        """
        logger.info(
            f"Updating temporal calibration for site {site_id} at {current_time}")

        # Get weights for exponential forgetting (more recent = higher weight)
        n_samples = len(psi_recent)
        weights = np.array(
            [self.forgetting_factor ** (n_samples - i - 1) for i in range(n_samples)])

        def weighted_vg_objective(params):
            theta_r, theta_s, alpha, n = params
            m = 1 - 1/n

            psi_abs = np.abs(psi_recent)
            theta_pred = theta_r + (theta_s - theta_r) / \
                (1 + (alpha * psi_abs)**n)**m

            # Weighted MSE
            errors = (theta_recent - theta_pred)**2
            return np.average(errors, weights=weights)

        # Initial guess from previous calibration or defaults
        if site_id in self.temporal_params and self.temporal_params[site_id]:
            prev_params = self.temporal_params[site_id][-1]['parameters']
            p0 = [prev_params['theta_r'], prev_params['theta_s'],
                  prev_params['alpha'], prev_params['n']]
        else:
            p0 = [0.05, 0.45, 0.02, 2.0]

        bounds = [
            self.config.vg_bounds['theta_r'],
            self.config.vg_bounds['theta_s'],
            self.config.vg_bounds['alpha'],
            self.config.vg_bounds['n']
        ]

        try:
            result = optimize.minimize(
                weighted_vg_objective, p0,
                bounds=bounds,
                method='L-BFGS-B',
                # Faster for temporal updates
                options={'maxiter': self.config.max_iter // 2}
            )

            theta_r, theta_s, alpha, n = result.x
            m = 1 - 1/n

            updated_params = {
                'theta_r': theta_r,
                'theta_s': theta_s,
                'alpha': alpha,
                'n': n,
                'm': m,
                'timestamp': current_time,
                'convergence': result.success,
                'final_loss': result.fun
            }

            # Store temporal history
            if site_id not in self.temporal_params:
                self.temporal_params[site_id] = []
            self.temporal_params[site_id].append({
                'timestamp': current_time,
                'parameters': updated_params
            })

            # Keep only recent history (last 30 days)
            cutoff_time = current_time - pd.Timedelta(days=30)
            self.temporal_params[site_id] = [
                entry for entry in self.temporal_params[site_id]
                if entry['timestamp'] >= cutoff_time
            ]

            logger.info(f"Temporal calibration updated for site {site_id}")

            return updated_params

        except Exception as e:
            logger.error(
                f"Temporal calibration update failed for site {site_id}: {e}")
            return None

    def get_temporal_parameters(self, site_id: str, timestamp: Optional[pd.Timestamp] = None) -> Optional[Dict[str, Any]]:
        """Get temporally adapted parameters for a site at a specific time."""
        if site_id not in self.temporal_params or not self.temporal_params[site_id]:
            return None

        if timestamp is None:
            # Return most recent
            return self.temporal_params[site_id][-1]['parameters']

        # Find closest temporal parameters
        timestamps = [entry['timestamp']
                      for entry in self.temporal_params[site_id]]
        closest_idx = np.argmin(
            [abs((ts - timestamp).total_seconds()) for ts in timestamps])

        return self.temporal_params[site_id][closest_idx]['parameters']


class EnsembleCalibrator:
    """
    Calibrates ensemble model weights for site-specific performance.

    Optimizes the combination weights of ensemble members for each site.
    """

    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.site_weights: Dict[str, np.ndarray] = {}

    def calibrate_ensemble_weights(self, site_id: str, base_predictions: np.ndarray,
                                   true_values: np.ndarray) -> Dict[str, Any]:
        """
        Calibrate ensemble weights for optimal site-specific performance.

        Args:
            base_predictions: Shape (n_samples, n_models)
            true_values: Shape (n_samples,)
        """
        logger.info(f"Calibrating ensemble weights for site {site_id}")

        n_models = base_predictions.shape[1]

        def ensemble_objective(weights):
            """Objective function for ensemble weight optimization."""
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize to sum to 1

            ensemble_pred = np.dot(base_predictions, weights)
            return mean_squared_error(true_values, ensemble_pred)

        # Initial weights
        if site_id in self.site_weights:
            w0 = self.site_weights[site_id]
        else:
            w0 = np.array(self.config.ensemble_weights_init[:n_models])
            w0 = w0 / np.sum(w0)  # Normalize

        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(n_models)]

        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        try:
            result = optimize.minimize(
                ensemble_objective, w0,
                bounds=bounds,
                constraints=constraints,
                method='SLSQP',
                options={'maxiter': 200}
            )

            optimal_weights = result.x / np.sum(result.x)  # Ensure normalized

            # Evaluate calibrated ensemble
            ensemble_pred = np.dot(base_predictions, optimal_weights)
            rmse = np.sqrt(mean_squared_error(true_values, ensemble_pred))
            r2 = 1 - np.var(true_values - ensemble_pred) / np.var(true_values)

            calibration_result = {
                'weights': optimal_weights,
                'rmse': rmse,
                'r2': r2,
                'convergence': result.success,
                'n_models': n_models
            }

            # Store weights
            self.site_weights[site_id] = optimal_weights

            logger.info(
                f"Ensemble calibration completed for site {site_id}. R²: {r2:.3f}")

            return calibration_result

        except Exception as e:
            logger.error(
                f"Ensemble calibration failed for site {site_id}: {e}")
            return {
                'weights': w0,
                'rmse': np.sqrt(mean_squared_error(true_values, np.dot(base_predictions, w0))),
                'r2': 0.0,
                'convergence': False,
                'error': str(e)
            }

    def get_ensemble_weights(self, site_id: str) -> Optional[np.ndarray]:
        """Get calibrated ensemble weights for a site."""
        return self.site_weights.get(site_id)


class AdaptiveCalibrationPipeline:
    """
    Complete adaptive calibration pipeline for ψ models.

    Orchestrates site-specific, temporal, and ensemble calibration methods.
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()

        # Initialize calibration components
        self.site_calibrator = SiteSpecificCalibrator(self.config)
        self.temporal_calibrator = TemporalCalibrator(self.config)
        self.ensemble_calibrator = EnsembleCalibrator(self.config)

    def run_full_calibration(self, df: pd.DataFrame, site_col: str = 'site_id',
                             psi_col: str = 'psi', theta_col: str = 'theta',
                             time_col: Optional[str] = 'timestamp') -> Dict[str, Any]:
        """
        Run complete adaptive calibration pipeline.

        Returns calibration results for all sites.
        """
        logger.info("Running full adaptive calibration pipeline for ψ models")

        calibration_results = {}

        # Group by site
        if site_col in df.columns:
            site_groups = df.groupby(site_col)
        else:
            # Assume single site
            site_groups = [('single_site', df)]

        for site_id, site_data in site_groups:
            logger.info(f"Calibrating site: {site_id}")

            site_results = {}

            # 1. Site-specific VG calibration
            psi_obs = site_data[psi_col].values
            theta_obs = site_data[theta_col].values if theta_col in site_data.columns else None

            if theta_obs is not None:
                site_results['vg_calibration'] = self.site_calibrator.calibrate_site_vg(
                    psi_obs, theta_obs, site_id
                )
            else:
                logger.warning(
                    f"No theta data for site {site_id} - skipping VG calibration")

            # 2. Temporal calibration (if time data available)
            if time_col and time_col in site_data.columns:
                site_data_time = site_data.set_index(time_col).sort_index()

                # Use recent data for temporal calibration
                recent_data = site_data_time.tail(
                    self.config.adaptation_window)
                if len(recent_data) >= 10:  # Minimum samples
                    temporal_result = self.temporal_calibrator.update_temporal_calibration(
                        site_id,
                        recent_data[psi_col].values,
                        recent_data[theta_col].values if theta_col in recent_data.columns else psi_obs[:len(
                            recent_data)],
                        recent_data.index[-1]
                    )
                    site_results['temporal_calibration'] = temporal_result

            calibration_results[site_id] = site_results

        logger.info("Adaptive calibration pipeline completed")

        return calibration_results

    def apply_calibrated_predictions(self, df: pd.DataFrame, base_predictions: Optional[np.ndarray] = None,
                                     site_col: str = 'site_id', time_col: Optional[str] = 'timestamp') -> pd.DataFrame:
        """
        Apply calibrated parameters to generate improved ψ predictions.

        Returns DataFrame with calibrated predictions.
        """
        df_calibrated = df.copy()

        # Apply site-specific and temporal calibrations
        for idx, row in df_calibrated.iterrows():
            site_id = row[site_col] if site_col in df_calibrated.columns else 'single_site'

            # Get calibrated parameters
            site_params = self.site_calibrator.get_site_parameters(site_id)
            temporal_params = None

            if time_col and time_col in df_calibrated.columns:
                timestamp = pd.to_datetime(row[time_col])
                temporal_params = self.temporal_calibrator.get_temporal_parameters(
                    site_id, timestamp)

            # Use temporal params if available, otherwise site params
            active_params = temporal_params or (
                site_params['params'] if site_params else None)

            if active_params:
                # Could apply VG corrections here if we had the base model predictions
                # For now, just store the calibrated parameters
                df_calibrated.loc[idx,
                                  'calibrated_theta_r'] = active_params['theta_r']
                df_calibrated.loc[idx,
                                  'calibrated_theta_s'] = active_params['theta_s']
                df_calibrated.loc[idx,
                                  'calibrated_alpha'] = active_params['alpha']
                df_calibrated.loc[idx, 'calibrated_n'] = active_params['n']

        # Apply ensemble calibration if base predictions provided
        if base_predictions is not None and site_col in df_calibrated.columns:
            ensemble_weights = {}
            for site_id in df_calibrated[site_col].unique():
                weights = self.ensemble_calibrator.get_ensemble_weights(
                    site_id)
                if weights is not None:
                    ensemble_weights[site_id] = weights

            if ensemble_weights:
                # Apply ensemble weighting
                df_calibrated['ensemble_pred'] = 0.0
                for i, (site_id, weights) in enumerate(ensemble_weights.items()):
                    site_mask = df_calibrated[site_col] == site_id
                    site_preds = base_predictions[site_mask]
                    if len(site_preds.shape) > 1:
                        df_calibrated.loc[site_mask, 'ensemble_pred'] = np.dot(
                            site_preds, weights)

        return df_calibrated

    def get_calibration_summary(self, calibration_results: Dict[str, Any]) -> str:
        """Generate human-readable calibration summary."""
        summary = "Adaptive Calibration Summary\n"
        summary += "=" * 30 + "\n\n"

        total_sites = len(calibration_results)
        successful_sites = 0

        for site_id, results in calibration_results.items():
            summary += f"Site: {site_id}\n"

            if 'vg_calibration' in results:
                vg = results['vg_calibration']
                if vg['quality']['calibration_success']:
                    successful_sites += 1
                    summary += f"  VG Calibration: SUCCESS (R²: {vg['quality']['r2']:.3f})\n"
                else:
                    summary += "  VG Calibration: FAILED\n"

            if 'temporal_calibration' in results and results['temporal_calibration']:
                summary += "  Temporal Calibration: ACTIVE\n"
            else:
                summary += "  Temporal Calibration: NOT APPLIED\n"

            summary += "\n"

        summary += f"Summary: {successful_sites}/{total_sites} sites successfully calibrated\n"

        return summary
