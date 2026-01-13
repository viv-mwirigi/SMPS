from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize

from smps.calibration.objective import evaluate_parameters
from smps.calibration.problem import CalibrationConfig, CalibrationDataset, CalibrationResult
from smps.physics.enhanced_water_balance import EnhancedWaterBalance


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float
    transform: str = "identity"  # identity | log10


def _vector_to_params(x: np.ndarray, specs: List[ParameterSpec]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for value, spec in zip(x, specs):
        if spec.transform == "log10":
            out[spec.name] = float(10 ** float(value))
        else:
            out[spec.name] = float(value)
    return out


def default_parameter_space() -> List[ParameterSpec]:
    """A compact, high-impact search space (global multipliers + macro/crop)."""
    return [
        ParameterSpec("Ks_adj", -1.0, 1.0, transform="log10"),
        ParameterSpec("alpha_adj", -1.0, 1.0, transform="log10"),
        ParameterSpec("theta_s_adj", 0.85, 1.15, transform="identity"),
        ParameterSpec("n_adj", 0.90, 1.10, transform="identity"),
        ParameterSpec("theta_r_adj", 0.80, 1.20, transform="identity"),
        ParameterSpec("kcb_scale", 0.80, 1.20, transform="identity"),
        ParameterSpec("feddes_dryness_scale", 0.80,
                      1.25, transform="identity"),
        ParameterSpec("infiltration_macropore_max_fraction",
                      0.0, 0.6, transform="identity"),
        ParameterSpec("macropore_conductivity_factor", 0.0,
                      2.0, transform="log10"),  # 1..100
    ]


def calibrate(
    base_model: EnhancedWaterBalance,
    dataset: CalibrationDataset,
    config: CalibrationConfig,
    specs: Optional[List[ParameterSpec]] = None,
    seed: int = 42,
    global_maxiter: int = 40,
    global_popsize: int = 12,
    refine: bool = True,
    refine_method: str = "L-BFGS-B",
) -> CalibrationResult:
    """Calibrate key parameters using global + local optimization.

    Global stage uses differential evolution (robust global search).
    Local stage optionally refines with L-BFGS-B or Nelder-Mead.
    """

    specs = specs or default_parameter_space()

    bounds = [(s.lower, s.upper) for s in specs]

    def obj_fn(x: np.ndarray) -> float:
        p = _vector_to_params(x, specs)
        value, _diag = evaluate_parameters(base_model, dataset, config, p)
        return float(value)

    result_de = differential_evolution(
        obj_fn,
        bounds=bounds,
        seed=seed,
        maxiter=global_maxiter,
        popsize=global_popsize,
        polish=False,
        updating="deferred",
        workers=1,
    )

    x_best = np.asarray(result_de.x, dtype=float)

    if refine:
        if refine_method.upper() == "NELDER-MEAD":
            res_local = minimize(
                obj_fn, x_best, method="Nelder-Mead", options={"maxiter": 200})
        else:
            # L-BFGS-B supports bounds
            res_local = minimize(
                obj_fn, x_best, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
        if res_local.success and np.isfinite(res_local.fun) and res_local.fun < result_de.fun:
            x_best = np.asarray(res_local.x, dtype=float)

    best_params = _vector_to_params(x_best, specs)
    best_obj, diag = evaluate_parameters(
        base_model, dataset, config, best_params)

    diag_out = {**diag, "global_fun": float(result_de.fun)}

    return CalibrationResult(
        best_parameters=best_params,
        best_objective=float(best_obj),
        diagnostics=diag_out,
    )


def save_result_json(path: str, result: CalibrationResult) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_objective": result.best_objective,
                "best_parameters": result.best_parameters,
                "diagnostics": result.diagnostics,
            },
            f,
            indent=2,
            sort_keys=True,
        )
