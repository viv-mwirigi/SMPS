from __future__ import annotations

import numpy as np


def rmse(sim: np.ndarray, obs: np.ndarray) -> float:
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = np.isfinite(sim) & np.isfinite(obs)
    if not np.any(mask):
        return float("nan")
    e = sim[mask] - obs[mask]
    return float(np.sqrt(np.mean(e * e)))


def ubrmse(sim: np.ndarray, obs: np.ndarray) -> float:
    """Unbiased RMSE (RMSE of anomalies)."""
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = np.isfinite(sim) & np.isfinite(obs)
    if not np.any(mask):
        return float("nan")
    s = sim[mask]
    o = obs[mask]
    e = (s - np.mean(s)) - (o - np.mean(o))
    return float(np.sqrt(np.mean(e * e)))


def kge(sim: np.ndarray, obs: np.ndarray, min_obs_std: float = 1e-3) -> float:
    """Kling-Gupta efficiency (Gupta et al., 2009).

    Args:
        sim: Simulated values
        obs: Observed values
        min_obs_std: Minimum observation standard deviation threshold.
            If obs std is below this, return NaN to avoid pathological metrics
            from near-constant observation series (e.g., stuck sensors).
    """
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = np.isfinite(sim) & np.isfinite(obs)
    if not np.any(mask):
        return float("nan")

    s = sim[mask]
    o = obs[mask]

    if s.size < 2 or o.size < 2:
        return float("nan")

    mean_s = float(np.mean(s))
    mean_o = float(np.mean(o))
    std_s = float(np.std(s, ddof=1))
    std_o = float(np.std(o, ddof=1))

    # Return NaN for near-constant observation series to avoid pathological metrics
    if std_o < min_obs_std:
        return float("nan")

    if std_s <= 0:
        return float("nan")

    r = float(np.corrcoef(s, o)[0, 1])
    alpha = std_s / std_o
    beta = mean_s / mean_o if mean_o != 0 else float("nan")

    if not np.isfinite(r) or not np.isfinite(alpha) or not np.isfinite(beta):
        return float("nan")

    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
