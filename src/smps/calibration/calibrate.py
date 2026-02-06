"""
Parameter Calibration for SWPPS.

Implements calibration of:
- Van Genuchten parameters from measured data
- Water balance parameters using optimization
- Physics model corrections
- Tropical soil corrections for African soils
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize, Bounds
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import logging

from ..core.types import VanGenuchtenParams
from ..physics.van_genuchten import (
    water_content_from_potential,
    potential_from_water_content,
)
from ..physics.tropical import TropicalSoilCorrections
from ..validation.metrics import compute_kge, compute_nse

logger = logging.getLogger("swpps.calibration.calibrate")


@dataclass
class CalibrationResult:
    """Result of parameter calibration."""

    success: bool = False
    message: str = ""

    # Optimized parameters
    parameters: Dict[str, float] = field(default_factory=dict)

    # Performance metrics
    objective_value: float = np.inf
    n_iterations: int = 0
    n_function_evals: int = 0

    # Validation
    rmse: float = np.nan
    r_squared: float = np.nan
    kge: float = np.nan
    nse: float = np.nan

    def __str__(self) -> str:
        status = "✓ Success" if self.success else "✗ Failed"
        return (
            f"CalibrationResult({status}, "
            f"obj={self.objective_value:.4f}, "
            f"KGE={self.kge:.3f}, "
            f"iterations={self.n_iterations})"
        )


class VanGenuchtenCalibrator:
    """
    Calibrate Van Genuchten parameters from retention curve data.

    Fits the van Genuchten equation:
        θ(ψ) = θr + (θs - θr) / [1 + (α|ψ|)^n]^m

    to measured water content and matric potential pairs.
    """

    # Parameter bounds: (alpha, n, theta_r, theta_s)
    DEFAULT_BOUNDS = {
        "alpha": (0.001, 0.5),    # 1/kPa
        "n": (1.05, 5.0),         # dimensionless
        "theta_r": (0.01, 0.15),  # m³/m³
        "theta_s": (0.25, 0.60),  # m³/m³
    }

    def __init__(
        self,
        fix_theta_s: Optional[float] = None,
        fix_theta_r: Optional[float] = None,
        m_constrained: bool = True,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Initialize calibrator.

        Args:
            fix_theta_s: Fix saturated water content (if known)
            fix_theta_r: Fix residual water content (if known)
            m_constrained: Use m = 1 - 1/n constraint
            bounds: Custom parameter bounds
        """
        self.fix_theta_s = fix_theta_s
        self.fix_theta_r = fix_theta_r
        self.m_constrained = m_constrained
        self.bounds = bounds or self.DEFAULT_BOUNDS.copy()

    def calibrate(
        self,
        psi_observed: np.ndarray,
        theta_observed: np.ndarray,
        method: str = "differential_evolution",
        maxiter: int = 1000,
    ) -> CalibrationResult:
        """
        Calibrate VG parameters.

        Args:
            psi_observed: Matric potential measurements (kPa, negative)
            theta_observed: Volumetric water content measurements (m³/m³)
            method: Optimization method
            maxiter: Maximum iterations

        Returns:
            CalibrationResult with fitted parameters
        """
        psi = np.asarray(psi_observed).flatten()
        theta = np.asarray(theta_observed).flatten()

        # Remove invalid
        valid = np.isfinite(psi) & np.isfinite(theta)
        psi = psi[valid]
        theta = theta[valid]

        if len(psi) < 5:
            return CalibrationResult(
                success=False,
                message="Insufficient data points for calibration"
            )

        # Determine which parameters to fit
        fit_params = []
        bounds_list = []

        fit_params.extend(["alpha", "n"])
        bounds_list.extend([self.bounds["alpha"], self.bounds["n"]])

        if self.fix_theta_r is None:
            fit_params.append("theta_r")
            bounds_list.append(self.bounds["theta_r"])

        if self.fix_theta_s is None:
            fit_params.append("theta_s")
            # Constrain theta_s to be at least max observed theta
            theta_s_min = max(self.bounds["theta_s"][0], np.max(theta) * 1.01)
            bounds_list.append((theta_s_min, self.bounds["theta_s"][1]))

        def objective(x):
            params = self._unpack_params(x, fit_params)
            theta_pred = self._predict_theta(psi, params)
            return np.mean((theta - theta_pred) ** 2)

        # Run optimization
        if method == "differential_evolution":
            result = differential_evolution(
                objective,
                bounds_list,
                maxiter=maxiter,
                seed=42,
                polish=True,
            )
        else:
            x0 = np.array([(b[0] + b[1]) / 2 for b in bounds_list])
            result = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds_list,
                options={"maxiter": maxiter},
            )

        # Extract results
        params = self._unpack_params(result.x, fit_params)
        theta_pred = self._predict_theta(psi, params)

        # Compute validation metrics
        errors = theta - theta_pred
        rmse = np.sqrt(np.mean(errors ** 2))
        r_squared = 1 - np.sum(errors ** 2) / \
            np.sum((theta - np.mean(theta)) ** 2)
        kge, _, _, _ = compute_kge(theta, theta_pred)
        nse = compute_nse(theta, theta_pred)

        return CalibrationResult(
            success=result.success if hasattr(result, "success") else True,
            message=result.message if hasattr(
                result, "message") else "Optimization completed",
            parameters=params,
            objective_value=result.fun,
            n_iterations=result.nit if hasattr(result, "nit") else 0,
            n_function_evals=result.nfev if hasattr(result, "nfev") else 0,
            rmse=rmse,
            r_squared=r_squared,
            kge=kge,
            nse=nse,
        )

    def _unpack_params(
        self,
        x: np.ndarray,
        fit_params: List[str],
    ) -> Dict[str, float]:
        """Unpack optimization vector to parameter dict."""
        params = {}

        idx = 0
        for name in fit_params:
            params[name] = x[idx]
            idx += 1

        # Add fixed values
        if self.fix_theta_r is not None:
            params["theta_r"] = self.fix_theta_r
        if self.fix_theta_s is not None:
            params["theta_s"] = self.fix_theta_s

        # Compute m
        if self.m_constrained:
            params["m"] = 1 - 1 / params["n"]
        else:
            params["m"] = params.get("m", 1 - 1 / params["n"])

        return params

    def _predict_theta(
        self,
        psi: np.ndarray,
        params: Dict[str, float],
    ) -> np.ndarray:
        """Predict water content from potential using VG equation."""
        alpha = params["alpha"]
        n = params["n"]
        m = params["m"]
        theta_r = params["theta_r"]
        theta_s = params["theta_s"]

        psi_abs = np.abs(psi)
        Se = (1 + (alpha * psi_abs) ** n) ** (-m)
        theta = theta_r + (theta_s - theta_r) * Se

        return theta


class WaterBalanceCalibrator:
    """
    Calibrate water balance model parameters.

    Optimizes:
    - Curve number (CN) for runoff
    - Drainage coefficients
    - Root zone distribution
    """

    DEFAULT_BOUNDS = {
        "cn": (50, 95),              # Curve number
        "ksat_mult": (0.5, 2.0),     # K_sat multiplier
        "root_depth_cm": (20, 60),   # Effective root depth
        "et_mult": (0.7, 1.3),       # ET multiplier
    }

    def __init__(
        self,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """Initialize calibrator."""
        self.bounds = bounds or self.DEFAULT_BOUNDS.copy()

    def calibrate(
        self,
        observed_psi: np.ndarray,
        times: np.ndarray,
        precip: np.ndarray,
        et: np.ndarray,
        vg_params: VanGenuchtenParams,
        run_model: Callable,
        maxiter: int = 500,
    ) -> CalibrationResult:
        """
        Calibrate water balance parameters.

        Args:
            observed_psi: Observed matric potential timeseries (kPa)
            times: Timestamps (datetime or numeric)
            precip: Precipitation timeseries (mm)
            et: ET timeseries (mm)
            vg_params: Van Genuchten parameters
            run_model: Callable that runs the model with given params
            maxiter: Maximum iterations

        Returns:
            CalibrationResult
        """
        bounds_list = [
            self.bounds["cn"],
            self.bounds["ksat_mult"],
            self.bounds["root_depth_cm"],
            self.bounds["et_mult"],
        ]

        def objective(x):
            cn, ksat_mult, root_depth, et_mult = x

            # Run model
            try:
                predicted_psi = run_model(
                    times=times,
                    precip=precip,
                    et=et * et_mult,
                    vg_params=vg_params,
                    cn=cn,
                    ksat_mult=ksat_mult,
                    root_depth_cm=root_depth,
                )

                # Compute KGE (maximize -> minimize negative)
                kge, _, _, _ = compute_kge(observed_psi, predicted_psi)
                return -kge if np.isfinite(kge) else 1e6

            except Exception as e:
                logger.warning("Model run failed: %s", e)
                return 1e6

        result = differential_evolution(
            objective,
            bounds_list,
            maxiter=maxiter,
            seed=42,
            polish=True,
            workers=1,
        )

        # Extract results
        cn, ksat_mult, root_depth, et_mult = result.x

        params = {
            "cn": cn,
            "ksat_mult": ksat_mult,
            "root_depth_cm": root_depth,
            "et_mult": et_mult,
        }

        # Run final prediction
        try:
            predicted_psi = run_model(
                times=times,
                precip=precip,
                et=et * et_mult,
                vg_params=vg_params,
                cn=cn,
                ksat_mult=ksat_mult,
                root_depth_cm=root_depth,
            )

            errors = observed_psi - predicted_psi
            rmse = np.sqrt(np.nanmean(errors ** 2))
            kge, _, _, _ = compute_kge(observed_psi, predicted_psi)
            nse = compute_nse(observed_psi, predicted_psi)

            valid = np.isfinite(observed_psi) & np.isfinite(predicted_psi)
            r_squared = np.corrcoef(
                observed_psi[valid], predicted_psi[valid]
            )[0, 1] ** 2

        except Exception:
            rmse, r_squared, kge, nse = np.nan, np.nan, np.nan, np.nan

        return CalibrationResult(
            success=result.success,
            message=result.message,
            parameters=params,
            objective_value=result.fun,
            n_iterations=result.nit,
            n_function_evals=result.nfev,
            rmse=rmse,
            r_squared=r_squared,
            kge=kge,
            nse=nse,
        )


def calibrate_van_genuchten(
    psi: np.ndarray,
    theta: np.ndarray,
    fix_theta_s: Optional[float] = None,
    fix_theta_r: Optional[float] = None,
) -> CalibrationResult:
    """
    Convenience function to calibrate Van Genuchten parameters.

    Args:
        psi: Matric potential measurements (kPa)
        theta: Volumetric water content (m³/m³)
        fix_theta_s: Fixed saturated water content
        fix_theta_r: Fixed residual water content

    Returns:
        CalibrationResult with fitted VG parameters
    """
    calibrator = VanGenuchtenCalibrator(
        fix_theta_s=fix_theta_s,
        fix_theta_r=fix_theta_r,
    )
    return calibrator.calibrate(psi, theta)


def calibrate_water_balance(
    observed_psi: np.ndarray,
    times: np.ndarray,
    precip: np.ndarray,
    et: np.ndarray,
    vg_params: VanGenuchtenParams,
    run_model: Callable,
) -> CalibrationResult:
    """
    Convenience function to calibrate water balance parameters.

    Args:
        observed_psi: Observed matric potential (kPa)
        times: Timestamps
        precip: Precipitation (mm)
        et: Evapotranspiration (mm)
        vg_params: Van Genuchten parameters
        run_model: Model runner function

    Returns:
        CalibrationResult with calibrated parameters
    """
    calibrator = WaterBalanceCalibrator()
    return calibrator.calibrate(
        observed_psi, times, precip, et, vg_params, run_model
    )


class TropicalSoilCalibrator:
    """
    Calibrate tropical soil correction parameters.

    Optimizes oxide aggregation and macropore factors based on
    observed soil moisture dynamics.
    """

    DEFAULT_BOUNDS = {
        "oxide_content": (0.05, 0.50),       # Oxide content factor
        "macropore_mult": (0.5, 3.0),        # Macropore multiplier
        "agg_exponent": (0.5, 1.5),          # Aggregation exponent
        "Ksat_mult": (0.5, 5.0),             # Ksat multiplier
    }

    def __init__(
        self,
        clay_fraction: float,
        sand_fraction: float,
        soil_type: str = "generic",
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Initialize tropical soil calibrator.

        Args:
            clay_fraction: Clay fraction (0-1)
            sand_fraction: Sand fraction (0-1)
            soil_type: Soil type for initial estimates
            bounds: Custom parameter bounds
        """
        self.clay_fraction = clay_fraction
        self.sand_fraction = sand_fraction
        self.soil_type = soil_type
        self.bounds = bounds or self.DEFAULT_BOUNDS.copy()

        # Get initial tropical corrections
        self.base_corrections = TropicalSoilCorrections.for_african_soil(
            soil_type=soil_type,
            clay_fraction=clay_fraction,
            sand_fraction=sand_fraction,
        )

    def calibrate(
        self,
        observed_psi: np.ndarray,
        predicted_psi_standard: np.ndarray,
        precip: np.ndarray,
        Ksat_standard_mm_day: float,
        vg_params: VanGenuchtenParams,
        run_model_with_corrections: Callable,
        maxiter: int = 300,
    ) -> CalibrationResult:
        """
        Calibrate tropical corrections.

        Args:
            observed_psi: Observed matric potential (kPa)
            predicted_psi_standard: Standard PTF predictions (kPa)
            precip: Precipitation (mm)
            Ksat_standard_mm_day: Standard Ksat estimate
            vg_params: Base VG parameters
            run_model_with_corrections: Model callable
            maxiter: Maximum iterations

        Returns:
            CalibrationResult with optimized tropical corrections
        """
        bounds_list = [
            self.bounds["oxide_content"],
            self.bounds["macropore_mult"],
            self.bounds["Ksat_mult"],
        ]

        def objective(x):
            oxide_content, macro_mult, Ksat_mult = x

            # Create adjusted corrections
            corrections = TropicalSoilCorrections(
                clay_fraction=self.clay_fraction,
                sand_fraction=self.sand_fraction,
                oxide_content=oxide_content,
            )

            # Apply corrections
            corrected = corrections.apply_all_corrections(
                theta_sat=vg_params.theta_s,
                theta_res=vg_params.theta_r,
                alpha=vg_params.alpha,
                n=vg_params.n,
                Ksat_mm_day=Ksat_standard_mm_day * Ksat_mult,
            )

            # Run model with corrections
            try:
                predicted_psi = run_model_with_corrections(
                    corrections=corrections,
                    corrected_params=corrected,
                )

                # Compute objective (minimize negative KGE)
                kge, _, _, _ = compute_kge(observed_psi, predicted_psi)

                # Penalize extreme corrections
                penalty = 0.0
                if corrected["aggregation_factor"] > 2.0:
                    penalty += 0.1 * (corrected["aggregation_factor"] - 2.0)
                if corrected["macropore_factor"] > 3.0:
                    penalty += 0.1 * (corrected["macropore_factor"] - 3.0)

                return -kge + penalty if np.isfinite(kge) else 1e6

            except Exception as e:
                logger.warning("Tropical calibration failed: %s", e)
                return 1e6

        # Run optimization
        result = differential_evolution(
            objective,
            bounds_list,
            maxiter=maxiter,
            seed=42,
            polish=True,
        )

        # Extract results
        oxide_content, macro_mult, Ksat_mult = result.x

        params = {
            "oxide_content": oxide_content,
            "macropore_mult": macro_mult,
            "Ksat_mult": Ksat_mult,
        }

        # Create final corrections
        final_corrections = TropicalSoilCorrections(
            clay_fraction=self.clay_fraction,
            sand_fraction=self.sand_fraction,
            oxide_content=oxide_content,
        )
        corrected_params = final_corrections.apply_all_corrections(
            theta_sat=vg_params.theta_s,
            theta_res=vg_params.theta_r,
            alpha=vg_params.alpha,
            n=vg_params.n,
            Ksat_mm_day=Ksat_standard_mm_day * Ksat_mult,
        )

        params.update({
            "theta_sat_corrected": corrected_params["theta_sat"],
            "alpha_corrected": corrected_params["alpha"],
            "n_corrected": corrected_params["n"],
            "Ksat_corrected_mm_day": corrected_params["Ksat_mm_day"],
            "aggregation_factor": corrected_params["aggregation_factor"],
            "macropore_factor": corrected_params["macropore_factor"],
        })

        return CalibrationResult(
            success=result.success,
            message=result.message,
            parameters=params,
            objective_value=result.fun,
            n_iterations=result.nit,
            n_function_evals=result.nfev,
        )


def calibrate_for_african_soil(
    observed_psi: np.ndarray,
    precip: np.ndarray,
    clay_pct: float,
    sand_pct: float,
    soil_type: str,
    vg_params: VanGenuchtenParams,
    Ksat_mm_day: float,
    run_model: Callable,
) -> CalibrationResult:
    """
    Convenience function to calibrate tropical corrections.

    Args:
        observed_psi: Observed matric potential
        precip: Precipitation timeseries
        clay_pct: Clay percentage
        sand_pct: Sand percentage
        soil_type: African soil type
        vg_params: Base VG parameters
        Ksat_mm_day: Base Ksat estimate
        run_model: Model runner

    Returns:
        CalibrationResult with tropical corrections
    """
    calibrator = TropicalSoilCalibrator(
        clay_fraction=clay_pct / 100,
        sand_fraction=sand_pct / 100,
        soil_type=soil_type,
    )

    # Create wrapper for model
    def run_with_corrections(corrections, corrected_params):
        return run_model(
            precip=precip,
            Ksat=corrected_params["Ksat_mm_day"],
            theta_s=corrected_params["theta_sat"],
            alpha=corrected_params["alpha"],
            n=corrected_params["n"],
        )

    return calibrator.calibrate(
        observed_psi=observed_psi,
        predicted_psi_standard=None,  # Not used in current impl
        precip=precip,
        Ksat_standard_mm_day=Ksat_mm_day,
        vg_params=vg_params,
        run_model_with_corrections=run_with_corrections,
    )


@dataclass
class ResidualDiagnostics:
    """
    Diagnostic analysis of physics model residuals.

    Analyzes patterns in ψ_obs - ψ_phys to identify calibration issues:
    - Positive residual: Physics predicts too dry (ψ too negative)
    - Negative residual: Physics predicts too wet (ψ too high)

    This informs which parameters to adjust:
    - Ksat: Affects infiltration rate
    - ET stress coefficient: Affects drying rate
    - Root depth: Affects response depth to conditions
    """
    mean_residual: float = 0.0
    std_residual: float = 0.0

    # Systematic biases
    dry_bias_fraction: float = 0.0  # Fraction where physics too dry
    wet_bias_fraction: float = 0.0  # Fraction where physics too wet

    # Conditional residuals
    residual_during_rain: float = 0.0
    residual_dry_periods: float = 0.0
    residual_high_et: float = 0.0

    # Suggested adjustments
    ksat_adjustment: str = "none"  # "increase", "decrease", "none"
    et_stress_adjustment: str = "none"
    root_depth_adjustment: str = "none"

    # Confidence
    n_samples: int = 0
    analysis_reliable: bool = False

    def __str__(self) -> str:
        return (
            f"ResidualDiagnostics(\n"
            f"  mean={self.mean_residual:.1f} kPa, std={self.std_residual:.1f},\n"
            f"  dry_bias={self.dry_bias_fraction:.1%}, wet_bias={self.wet_bias_fraction:.1%},\n"
            f"  suggested: Ksat {self.ksat_adjustment}, ET stress {self.et_stress_adjustment}\n"
            f")"
        )


class ResidualAnalyzer:
    """
    Analyze physics model residuals to diagnose calibration issues.

    Uses ψ residuals to identify what the physics model is getting wrong
    and suggest parameter adjustments.
    """

    def __init__(
        self,
        dry_threshold_kpa: float = 10.0,
        wet_threshold_kpa: float = -10.0,
    ):
        """
        Initialize analyzer.

        Args:
            dry_threshold_kpa: Residual above this = physics too dry
            wet_threshold_kpa: Residual below this = physics too wet
        """
        self.dry_threshold = dry_threshold_kpa
        self.wet_threshold = wet_threshold_kpa

    def analyze(
        self,
        psi_observed: np.ndarray,
        psi_physics: np.ndarray,
        precipitation: Optional[np.ndarray] = None,
        et: Optional[np.ndarray] = None,
    ) -> ResidualDiagnostics:
        """
        Analyze residuals and diagnose calibration issues.

        Args:
            psi_observed: Observed matric potential (kPa)
            psi_physics: Physics model prediction (kPa)
            precipitation: Precipitation timeseries (optional)
            et: Evapotranspiration timeseries (optional)

        Returns:
            ResidualDiagnostics with analysis and suggestions
        """
        obs = np.asarray(psi_observed)
        phys = np.asarray(psi_physics)

        # Basic residual
        residual = obs - phys  # Positive = physics too dry

        valid = np.isfinite(residual)
        residual = residual[valid]

        if len(residual) < 10:
            return ResidualDiagnostics(
                n_samples=len(residual),
                analysis_reliable=False,
            )

        # Basic statistics
        diag = ResidualDiagnostics(
            mean_residual=float(np.mean(residual)),
            std_residual=float(np.std(residual)),
            dry_bias_fraction=float(np.mean(residual > self.dry_threshold)),
            wet_bias_fraction=float(np.mean(residual < self.wet_threshold)),
            n_samples=len(residual),
            analysis_reliable=len(residual) >= 50,
        )

        # Conditional residuals
        if precipitation is not None:
            precip = np.asarray(precipitation)[valid]
            rain_mask = precip > 1
            dry_mask = precip < 0.1

            if np.sum(rain_mask) > 5:
                diag.residual_during_rain = float(np.mean(residual[rain_mask]))
            if np.sum(dry_mask) > 5:
                diag.residual_dry_periods = float(np.mean(residual[dry_mask]))

        if et is not None:
            et_arr = np.asarray(et)[valid]
            high_et_mask = et_arr > np.percentile(et_arr, 75)
            if np.sum(high_et_mask) > 5:
                diag.residual_high_et = float(np.mean(residual[high_et_mask]))

        # Diagnose and suggest adjustments
        self._diagnose_ksat(diag)
        self._diagnose_et_stress(diag)
        self._diagnose_root_depth(diag)

        return diag

    def _diagnose_ksat(self, diag: ResidualDiagnostics) -> None:
        """Diagnose Ksat calibration from infiltration behavior."""
        # During rain: positive residual = water not infiltrating = Ksat too low
        #              negative residual = too much infiltrating = Ksat too high
        if diag.residual_during_rain > 15:
            diag.ksat_adjustment = "increase"
        elif diag.residual_during_rain < -15:
            diag.ksat_adjustment = "decrease"
        else:
            diag.ksat_adjustment = "none"

    def _diagnose_et_stress(self, diag: ResidualDiagnostics) -> None:
        """Diagnose ET stress coefficient from dry period behavior."""
        # During high ET: positive residual = drying too fast = stress too low
        #                 negative residual = not drying enough = stress too high
        if diag.residual_high_et > 15:
            diag.et_stress_adjustment = "increase"  # Increase stress = reduce ET
        elif diag.residual_high_et < -15:
            diag.et_stress_adjustment = "decrease"
        else:
            diag.et_stress_adjustment = "none"

    def _diagnose_root_depth(self, diag: ResidualDiagnostics) -> None:
        """Diagnose root depth from response patterns."""
        # Systematic dry bias in dry periods = roots accessing too deep
        # Systematic wet bias in dry periods = roots too shallow
        if diag.residual_dry_periods > 20:
            diag.root_depth_adjustment = "decrease"
        elif diag.residual_dry_periods < -20:
            diag.root_depth_adjustment = "increase"
        else:
            diag.root_depth_adjustment = "none"

    def suggest_parameter_changes(
        self,
        diag: ResidualDiagnostics,
        current_ksat: float,
        current_stress_p: float,
        current_root_depth: float,
    ) -> Dict[str, float]:
        """
        Suggest specific parameter changes based on diagnostics.

        Args:
            diag: ResidualDiagnostics from analyze()
            current_ksat: Current Ksat (mm/day)
            current_stress_p: Current stress depletion fraction
            current_root_depth: Current root depth (m)

        Returns:
            Dict with suggested new parameter values
        """
        suggestions = {
            "Ksat_mm_day": current_ksat,
            "p_stress": current_stress_p,
            "root_depth_m": current_root_depth,
        }

        # Ksat adjustment
        if diag.ksat_adjustment == "increase":
            suggestions["Ksat_mm_day"] = current_ksat * 1.5
        elif diag.ksat_adjustment == "decrease":
            suggestions["Ksat_mm_day"] = current_ksat * 0.67

        # ET stress adjustment
        if diag.et_stress_adjustment == "increase":
            suggestions["p_stress"] = min(0.8, current_stress_p + 0.1)
        elif diag.et_stress_adjustment == "decrease":
            suggestions["p_stress"] = max(0.3, current_stress_p - 0.1)

        # Root depth adjustment
        if diag.root_depth_adjustment == "increase":
            suggestions["root_depth_m"] = current_root_depth * 1.3
        elif diag.root_depth_adjustment == "decrease":
            suggestions["root_depth_m"] = current_root_depth * 0.7

        return suggestions
