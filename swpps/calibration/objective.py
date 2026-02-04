"""
Objective Functions for Calibration.

Provides various objective functions for parameter optimization:
- RMSE (minimize error)
- NSE (maximize efficiency)
- KGE (maximize efficiency)
- Multi-objective combinations
"""

import numpy as np
from typing import Callable, List, Optional, Tuple


class ObjectiveFunction:
    """
    Wrapper for objective functions used in calibration.

    Handles sign conventions (minimize vs maximize) and
    combines multiple objectives.
    """

    def __init__(
        self,
        name: str,
        func: Callable[[np.ndarray, np.ndarray], float],
        minimize: bool = True,
        weight: float = 1.0,
    ):
        """
        Initialize objective function.

        Args:
            name: Name of the objective
            func: Function taking (observed, predicted) -> score
            minimize: True if lower is better
            weight: Weight for multi-objective optimization
        """
        self.name = name
        self.func = func
        self.minimize = minimize
        self.weight = weight

    def __call__(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
    ) -> float:
        """
        Evaluate objective.

        Returns value suitable for minimization (multiply by -1 if needed).
        """
        score = self.func(observed, predicted)

        if not self.minimize:
            # Convert maximize to minimize
            score = -score

        return score * self.weight


def rmse_objective(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Root Mean Square Error objective.

    RMSE = sqrt(mean((obs - pred)²))

    Lower is better.
    """
    obs = np.asarray(observed).flatten()
    pred = np.asarray(predicted).flatten()

    valid = np.isfinite(obs) & np.isfinite(pred)
    if np.sum(valid) < 2:
        return 1e6

    return np.sqrt(np.mean((obs[valid] - pred[valid]) ** 2))


def mae_objective(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Mean Absolute Error objective.

    MAE = mean(|obs - pred|)

    Lower is better.
    """
    obs = np.asarray(observed).flatten()
    pred = np.asarray(predicted).flatten()

    valid = np.isfinite(obs) & np.isfinite(pred)
    if np.sum(valid) < 2:
        return 1e6

    return np.mean(np.abs(obs[valid] - pred[valid]))


def nse_objective(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Nash-Sutcliffe Efficiency objective.

    NSE = 1 - Σ(obs - pred)² / Σ(obs - mean(obs))²

    Higher is better (max = 1).
    """
    obs = np.asarray(observed).flatten()
    pred = np.asarray(predicted).flatten()

    valid = np.isfinite(obs) & np.isfinite(pred)
    if np.sum(valid) < 2:
        return -1e6

    obs_v = obs[valid]
    pred_v = pred[valid]

    obs_mean = np.mean(obs_v)
    ss_res = np.sum((obs_v - pred_v) ** 2)
    ss_tot = np.sum((obs_v - obs_mean) ** 2)

    if ss_tot < 1e-10:
        return -1e6

    return 1.0 - ss_res / ss_tot


def kge_objective(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Kling-Gupta Efficiency objective.

    KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)

    Higher is better (max = 1).
    """
    obs = np.asarray(observed).flatten()
    pred = np.asarray(predicted).flatten()

    valid = np.isfinite(obs) & np.isfinite(pred)
    if np.sum(valid) < 2:
        return -1e6

    obs_v = obs[valid]
    pred_v = pred[valid]

    # Correlation
    r = np.corrcoef(obs_v, pred_v)[0, 1]

    # Variability ratio
    std_obs = np.std(obs_v)
    std_pred = np.std(pred_v)
    alpha = std_pred / std_obs if std_obs > 1e-10 else 1.0

    # Bias ratio
    mean_obs = np.mean(obs_v)
    mean_pred = np.mean(pred_v)
    beta = mean_pred / mean_obs if abs(mean_obs) > 1e-10 else 1.0

    if not np.isfinite(r):
        return -1e6

    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def kge_np_objective(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Non-parametric KGE (KGE') objective.

    Uses Spearman correlation instead of Pearson for
    robustness to outliers.

    Higher is better (max = 1).
    """
    from scipy.stats import spearmanr

    obs = np.asarray(observed).flatten()
    pred = np.asarray(predicted).flatten()

    valid = np.isfinite(obs) & np.isfinite(pred)
    if np.sum(valid) < 2:
        return -1e6

    obs_v = obs[valid]
    pred_v = pred[valid]

    # Spearman correlation
    r_sp, _ = spearmanr(obs_v, pred_v)

    # Variability ratio (using CV)
    cv_obs = np.std(obs_v) / np.mean(obs_v) if np.mean(obs_v) != 0 else 0
    cv_pred = np.std(pred_v) / np.mean(pred_v) if np.mean(pred_v) != 0 else 0
    alpha = cv_pred / cv_obs if cv_obs > 1e-10 else 1.0

    # Bias ratio
    beta = np.mean(pred_v) / \
        np.mean(obs_v) if abs(np.mean(obs_v)) > 1e-10 else 1.0

    if not np.isfinite(r_sp):
        return -1e6

    return 1.0 - np.sqrt((r_sp - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def log_nse_objective(
    observed: np.ndarray,
    predicted: np.ndarray,
    eps: float = 1.0,
) -> float:
    """
    Log-transformed NSE objective.

    Emphasizes low flows / dry conditions.

    Higher is better.
    """
    obs = np.asarray(observed).flatten()
    pred = np.asarray(predicted).flatten()

    # For matric potential, work with absolute values
    obs_abs = np.abs(obs) + eps
    pred_abs = np.abs(pred) + eps

    valid = np.isfinite(obs_abs) & np.isfinite(pred_abs)
    if np.sum(valid) < 2:
        return -1e6

    log_obs = np.log(obs_abs[valid])
    log_pred = np.log(pred_abs[valid])

    mean_log_obs = np.mean(log_obs)
    ss_res = np.sum((log_obs - log_pred) ** 2)
    ss_tot = np.sum((log_obs - mean_log_obs) ** 2)

    if ss_tot < 1e-10:
        return -1e6

    return 1.0 - ss_res / ss_tot


def multi_objective(
    observed: np.ndarray,
    predicted: np.ndarray,
    objectives: List[ObjectiveFunction],
) -> float:
    """
    Combine multiple objectives into weighted sum.

    Args:
        observed: Observed values
        predicted: Predicted values
        objectives: List of ObjectiveFunction instances

    Returns:
        Weighted sum of objectives (for minimization)
    """
    total = 0.0

    for obj in objectives:
        total += obj(observed, predicted)

    return total


def create_default_multi_objective() -> List[ObjectiveFunction]:
    """
    Create default multi-objective setup.

    Combines KGE and RMSE with appropriate weights.
    """
    return [
        ObjectiveFunction("KGE", kge_objective, minimize=False, weight=1.0),
        ObjectiveFunction("RMSE", rmse_objective, minimize=True, weight=0.01),
    ]


def compute_all_objectives(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    """
    Compute all available objective values.

    Returns:
        Dictionary of objective name -> value
    """
    return {
        "RMSE": rmse_objective(observed, predicted),
        "MAE": mae_objective(observed, predicted),
        "NSE": nse_objective(observed, predicted),
        "KGE": kge_objective(observed, predicted),
        "logNSE": log_nse_objective(observed, predicted),
    }
