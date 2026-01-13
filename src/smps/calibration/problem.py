from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationConfig:
    """Config for calibrating the physics model.

    The objective is a weighted sum over depths and sites:

        obj = w_ubrmse * mean(ubRMSE_norm) + w_kge * mean(1 - KGE) + w_mb * penalty

    Where ubRMSE_norm is ubRMSE normalized by obs std (per depth).
    """

    # Observation columns to use (each is a depth/layer time series).
    # If None, will infer columns that start with "theta_obs_".
    obs_theta_columns: Optional[List[str]] = None

    # Warmup to discard (days)
    warmup_days: int = 30

    # Weights
    w_ubrmse: float = 1.0
    w_kge: float = 1.0
    w_mass_balance: float = 0.1

    # Mass balance tolerance for penalty (mm)
    mass_balance_tolerance_mm: float = 0.5

    # Optional depth weights aligned with obs_theta_columns
    depth_weights: Optional[List[float]] = None


@dataclass(frozen=True)
class CalibrationDataset:
    """Holds forcing + observation data for one or more sites."""

    df: pd.DataFrame
    site_column: str = "site_id"
    date_column: str = "date"
    group_columns: Optional[List[str]] = None

    def group_keys(self) -> List[str]:
        """Return unique group keys for calibration aggregation.

        By default, groups by site_id if present; otherwise a single group.
        If group_columns is provided (e.g., ["site_id", "window_id"]) then
        each unique combination becomes its own group.
        """
        if self.group_columns:
            for c in self.group_columns:
                if c not in self.df.columns:
                    raise ValueError(f"Missing group column '{c}'")
            keys = (
                self.df[self.group_columns]
                .astype(str)
                .fillna("__nan__")
                .agg("|".join, axis=1)
                .dropna()
                .unique()
            )
            return sorted([str(k) for k in keys])

        if self.site_column in self.df.columns:
            return sorted([str(x) for x in self.df[self.site_column].dropna().unique()])

        return ["__single_site__"]

    def for_group(self, group_key: str) -> pd.DataFrame:
        if self.group_columns:
            parts = str(group_key).split("|")
            if len(parts) != len(self.group_columns):
                raise ValueError("group_key does not match group_columns")
            mask = np.ones(len(self.df), dtype=bool)
            for c, v in zip(self.group_columns, parts):
                mask &= self.df[c].astype(str) == v
            return self.df.loc[mask].copy()

        if self.site_column not in self.df.columns:
            out = self.df.copy()
            out[self.site_column] = "__single_site__"
            return out
        return self.df[self.df[self.site_column].astype(str) == str(group_key)].copy()


@dataclass(frozen=True)
class CalibrationResult:
    best_parameters: Dict[str, float]
    best_objective: float
    diagnostics: Dict[str, float]


def infer_obs_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("theta_obs_")]
    return sorted(cols)


def normalize_weights(weights: Optional[List[float]], n: int) -> np.ndarray:
    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.size != n:
            raise ValueError(f"depth_weights length {w.size} must match n={n}")
    s = float(np.sum(w))
    if s <= 0:
        return np.ones(n, dtype=float) / float(n)
    return w / s


def get_forcing_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = ["precipitation_mm", "et0_mm"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required forcing column '{col}'")

    forcings = df[["precipitation_mm", "et0_mm"]].copy()

    # Optional columns used by run_period
    for opt in ["ndvi", "irrigation_mm", "temperature_max_c", "day_of_season"]:
        if opt in df.columns:
            forcings[opt] = df[opt]

    # Ensure index is datetime
    if "date" in df.columns:
        forcings.index = pd.to_datetime(df["date"])
    elif isinstance(df.index, pd.DatetimeIndex):
        pass
    else:
        raise ValueError(
            "CalibrationDataset must include a 'date' column or datetime index")

    return forcings
