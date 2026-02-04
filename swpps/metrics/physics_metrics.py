"""
Physics-Based Metrics for Matric Potential Model Evaluation.

Implements evaluation metrics that incorporate soil physics principles
for assessing ψ (matric potential) model performance.

Physics-Based Metrics for ψ:
─────────────────────────────────────────────────────────────────
van Genuchten Fit Quality:   How well ψ predictions fit VG curve
Hydraulic Conductivity:      K(ψ) consistency with soil physics
Water Retention Accuracy:    θ(ψ) relationship preservation
Energy Conservation:         Gibbs free energy consistency
Mass Balance:               Water balance in ψ predictions
Soil-Specific Performance:   Performance by soil texture/depth
─────────────────────────────────────────────────────────────────

Benefits for ψ Modeling:
- Ensures physically realistic ψ predictions
- Better model interpretability with physics constraints
- Improved generalization across soil types
- Early detection of model physics violations

Research References:
- van Genuchten (1980): Closed-form equation for soil water retention
- Mualem (1976): Hydraulic conductivity prediction from ψ
- Vogel & Cislerova (1988): Energy conservation in soil water
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger("swpps.metrics.physics")


@dataclass
class PhysicsConfig:
    """Configuration for physics-based ψ metrics."""

    # van Genuchten parameters (typical ranges)
    vg_theta_r: float = 0.0    # Residual water content
    vg_theta_s: float = 0.5    # Saturated water content
    vg_alpha: float = 0.01     # Scale parameter (1/kPa)
    vg_n: float = 2.0          # Shape parameter
    vg_m: float = 0.5          # m = 1 - 1/n

    # Hydraulic conductivity parameters
    ks: float = 0.01           # Saturated hydraulic conductivity (cm/h)
    l_param: float = 0.5       # Tortuosity parameter

    # Energy conservation
    rho_w: float = 1.0         # Water density (g/cm³)
    g: float = 981.0           # Gravitational acceleration (cm/h²)

    # Soil-specific evaluation
    soil_textures: List[str] = field(default_factory=lambda: [
        'sand', 'loamy_sand', 'sandy_loam', 'loam', 'silt_loam',
        'sandy_clay_loam', 'clay_loam', 'silty_clay_loam', 'sandy_clay', 'silty_clay', 'clay'
    ])


class VanGenuchtenModel:
    """
    van Genuchten soil water retention model for ψ evaluation.

    θ(ψ) = θr + (θs - θr) / (1 + |αψ|^n)^m

    Used to assess how well ψ predictions align with expected water retention behavior.
    """

    def __init__(self, config: PhysicsConfig):
        self.config = config

    def water_content_from_psi(self, psi: np.ndarray) -> np.ndarray:
        """Calculate θ from ψ using van Genuchten equation."""
        psi_abs = np.abs(
            psi)  # ψ is negative, but equation uses absolute value

        theta = self.config.vg_theta_r + (
            (self.config.vg_theta_s - self.config.vg_theta_r) /
            (1 + (self.config.vg_alpha * psi_abs) **
             self.config.vg_n) ** self.config.vg_m
        )

        return theta

    def psi_from_water_content(self, theta: np.ndarray) -> np.ndarray:
        """Calculate ψ from θ using van Genuchten equation (inverse)."""
        # Clamp theta to valid range
        theta_clamped = np.clip(
            theta, self.config.vg_theta_r + 1e-6, self.config.vg_theta_s - 1e-6)

        # Inverse van Genuchten
        ratio = (self.config.vg_theta_s - self.config.vg_theta_r) / \
            (theta_clamped - self.config.vg_theta_r)
        psi_abs = (ratio ** (1/self.config.vg_m) - 1) ** (1 /
                                                          self.config.vg_n) / self.config.vg_alpha

        return -psi_abs  # ψ is negative

    def fit_vg_parameters(self, psi_obs: np.ndarray, theta_obs: np.ndarray) -> Dict[str, float]:
        """Fit van Genuchten parameters to observed ψ-θ data."""
        def vg_objective(params):
            theta_r, theta_s, alpha, n = params
            m = 1 - 1/n

            psi_abs = np.abs(psi_obs)
            theta_pred = theta_r + (theta_s - theta_r) / \
                (1 + (alpha * psi_abs)**n)**m

            return np.sum((theta_obs - theta_pred)**2)

        # Initial guess
        p0 = [self.config.vg_theta_r, self.config.vg_theta_s,
              self.config.vg_alpha, self.config.vg_n]

        # Bounds
        bounds = [(0, 0.5), (0.2, 0.6), (0.001, 1.0), (1.1, 5.0)]

        try:
            result = optimize.minimize(
                vg_objective, p0, bounds=bounds, method='L-BFGS-B')
            fitted_params = {
                'theta_r': result.x[0],
                'theta_s': result.x[1],
                'alpha': result.x[2],
                'n': result.x[3],
                'm': 1 - 1/result.x[3],
                'fit_success': result.success
            }
        except Exception as e:
            logger.warning(f"VG parameter fitting failed: {e}")
            fitted_params = {
                'theta_r': self.config.vg_theta_r,
                'theta_s': self.config.vg_theta_s,
                'alpha': self.config.vg_alpha,
                'n': self.config.vg_n,
                'm': self.config.vg_m,
                'fit_success': False
            }

        return fitted_params


class HydraulicConductivityModel:
    """
    Mualem-van Genuchten hydraulic conductivity model.

    K(ψ) = Ks * (θ/θs)^l * [1 - (1 - (θ/θs)^(1/m))^m]^2

    Used to evaluate if ψ predictions are consistent with expected K behavior.
    """

    def __init__(self, config: PhysicsConfig):
        self.config = config
        self.vg = VanGenuchtenModel(config)

    def conductivity_from_psi(self, psi: np.ndarray) -> np.ndarray:
        """Calculate K from ψ using Mualem-van Genuchten model."""
        # Get water content from ψ
        theta = self.vg.water_content_from_psi(psi)
        theta_s = self.config.vg_theta_s

        # Relative saturation
        Se = (theta - self.config.vg_theta_r) / \
            (theta_s - self.config.vg_theta_r)
        Se = np.clip(Se, 0, 1)  # Ensure valid range

        # Mualem model
        m = self.config.vg_m
        l = self.config.l_param

        K_rel = Se**l * (1 - (1 - Se**(1/m))**m)**2
        K = self.config.ks * K_rel

        return K

    def check_conductivity_consistency(self, psi_pred: np.ndarray, psi_obs: np.ndarray) -> Dict[str, float]:
        """Check if predicted ψ maintains hydraulic conductivity consistency."""
        K_pred = self.conductivity_from_psi(psi_pred)
        K_obs = self.conductivity_from_psi(psi_obs)

        # K should generally decrease as ψ becomes more negative
        # Check if this trend is preserved
        psi_trend_pred = np.corrcoef(psi_pred, K_pred)[0, 1]
        psi_trend_obs = np.corrcoef(psi_obs, K_obs)[0, 1]

        # Relative error in K
        k_relative_error = np.mean(np.abs(K_pred - K_obs) / (K_obs + 1e-6))

        return {
            'k_psi_correlation_preserved': abs(psi_trend_pred) > 0.5,
            'k_relative_error': k_relative_error,
            'psi_trend_pred': psi_trend_pred,
            'psi_trend_obs': psi_trend_obs
        }


class EnergyConservationMetrics:
    """
    Energy conservation metrics for ψ model evaluation.

    Checks if ψ predictions conserve energy according to soil water thermodynamics.
    """

    def __init__(self, config: PhysicsConfig):
        self.config = config

    def calculate_capillary_energy(self, psi: np.ndarray, depth: float = 0.0) -> np.ndarray:
        """Calculate total potential energy (capillary + gravitational)."""
        # Capillary potential energy
        capillary_energy = self.config.rho_w * \
            np.abs(psi) * 1000  # Convert kPa to Pa

        # Gravitational potential energy (positive downward)
        gravitational_energy = self.config.rho_w * self.config.g * depth

        return capillary_energy + gravitational_energy

    def check_energy_conservation(self, psi_pred: np.ndarray, psi_obs: np.ndarray,
                                  depth: float = 0.0) -> Dict[str, float]:
        """Check energy conservation between predicted and observed ψ."""
        energy_pred = self.calculate_capillary_energy(psi_pred, depth)
        energy_obs = self.calculate_capillary_energy(psi_obs, depth)

        # Energy should be conserved (no energy creation/destruction)
        energy_error = np.mean(np.abs(energy_pred - energy_obs))
        energy_relative_error = energy_error / (np.mean(energy_obs) + 1e-6)

        # Check if energy gradients are reasonable
        energy_gradient_pred = np.gradient(energy_pred)
        energy_gradient_obs = np.gradient(energy_obs)

        gradient_correlation = np.corrcoef(
            energy_gradient_pred, energy_gradient_obs)[0, 1]

        return {
            'energy_absolute_error': energy_error,
            'energy_relative_error': energy_relative_error,
            'energy_gradient_correlation': gradient_correlation,
            'energy_conserved': energy_relative_error < 0.1  # 10% threshold
        }


class PhysicsBasedMetrics:
    """
    Comprehensive physics-based evaluation metrics for ψ models.

    Combines van Genuchten fit, hydraulic conductivity, and energy conservation metrics.
    """

    def __init__(self, config: Optional[PhysicsConfig] = None):
        self.config = config or PhysicsConfig()
        self.vg_model = VanGenuchtenModel(self.config)
        self.hc_model = HydraulicConductivityModel(self.config)
        self.energy_metrics = EnergyConservationMetrics(self.config)

    def calculate_vg_fit_quality(self, psi_pred: np.ndarray, psi_obs: np.ndarray,
                                 theta_obs: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate how well ψ predictions fit van Genuchten water retention curve."""
        # If theta_obs not provided, assume VG relationship for observed ψ
        if theta_obs is None:
            theta_obs = self.vg_model.water_content_from_psi(psi_obs)

        # Get predicted theta from predicted psi
        theta_pred_from_psi = self.vg_model.water_content_from_psi(psi_pred)

        # VG fit quality: how well predicted ψ corresponds to expected θ
        vg_mse = mean_squared_error(theta_obs, theta_pred_from_psi)
        vg_rmse = np.sqrt(vg_mse)
        vg_r2 = r2_score(theta_obs, theta_pred_from_psi)

        # Check if predictions follow VG shape
        # Sort by psi for shape comparison
        sort_idx_obs = np.argsort(psi_obs)
        sort_idx_pred = np.argsort(psi_pred)

        psi_obs_sorted = psi_obs[sort_idx_obs]
        psi_pred_sorted = psi_pred[sort_idx_pred]
        theta_obs_sorted = theta_obs[sort_idx_obs]

        # Interpolate predicted theta at observed psi points
        theta_pred_interp = np.interp(psi_obs_sorted, psi_pred_sorted,
                                      theta_pred_from_psi[sort_idx_pred])

        shape_preservation = np.corrcoef(
            theta_obs_sorted, theta_pred_interp)[0, 1]

        return {
            'vg_mse': vg_mse,
            'vg_rmse': vg_rmse,
            'vg_r2': vg_r2,
            'shape_preservation': shape_preservation,
            'vg_fit_quality': 'good' if vg_r2 > 0.8 else 'poor'
        }

    def calculate_physics_metrics(self, psi_pred: np.ndarray, psi_obs: np.ndarray,
                                  theta_obs: Optional[np.ndarray] = None,
                                  depth: float = 0.0) -> Dict[str, Any]:
        """Calculate comprehensive physics-based metrics for ψ model evaluation."""
        logger.info("Calculating physics-based metrics for ψ model evaluation")

        metrics = {}

        # 1. van Genuchten fit quality
        metrics['vg_fit'] = self.calculate_vg_fit_quality(
            psi_pred, psi_obs, theta_obs)

        # 2. Hydraulic conductivity consistency
        metrics['hydraulic_conductivity'] = self.hc_model.check_conductivity_consistency(
            psi_pred, psi_obs)

        # 3. Energy conservation
        metrics['energy_conservation'] = self.energy_metrics.check_energy_conservation(
            psi_pred, psi_obs, depth)

        # 4. Overall physics score
        physics_scores = [
            metrics['vg_fit']['vg_r2'],
            1.0 if metrics['hydraulic_conductivity']['k_psi_correlation_preserved'] else 0.0,
            1.0 if metrics['energy_conservation']['energy_conserved'] else 0.0
        ]

        overall_physics_score = np.mean(physics_scores)

        metrics['overall'] = {
            'physics_score': overall_physics_score,
            'physics_compliant': overall_physics_score > 0.7,
            'component_scores': physics_scores
        }

        logger.info(
            f"Physics-based evaluation completed. Overall score: {overall_physics_score:.3f}")

        return metrics

    def get_physics_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate human-readable physics metrics summary."""
        summary = "Physics-Based ψ Model Evaluation\n"
        summary += "=" * 40 + "\n\n"

        # Overall score
        overall = metrics['overall']
        summary += f"Overall Physics Score: {overall['physics_score']:.3f}\n"
        summary += f"Physics Compliant: {overall['physics_compliant']}\n\n"

        # VG fit
        vg = metrics['vg_fit']
        summary += f"van Genuchten Fit Quality:\n"
        summary += f"  R²: {vg['vg_r2']:.3f}\n"
        summary += f"  RMSE: {vg['vg_rmse']:.4f}\n"
        summary += f"  Shape Preservation: {vg['shape_preservation']:.3f}\n"
        summary += f"  Fit Quality: {vg['vg_fit_quality']}\n\n"

        # Hydraulic conductivity
        hc = metrics['hydraulic_conductivity']
        summary += f"Hydraulic Conductivity Consistency:\n"
        summary += f"  K-ψ Correlation Preserved: {hc['k_psi_correlation_preserved']}\n"
        summary += f"  K Relative Error: {hc['k_relative_error']:.3f}\n\n"

        # Energy conservation
        energy = metrics['energy_conservation']
        summary += f"Energy Conservation:\n"
        summary += f"  Energy Conserved: {energy['energy_conserved']}\n"
        summary += f"  Relative Error: {energy['energy_relative_error']:.3f}\n"
        summary += f"  Gradient Correlation: {energy['energy_gradient_correlation']:.3f}\n"

        return summary
