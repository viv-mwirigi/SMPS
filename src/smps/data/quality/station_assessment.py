"""
Station quality assessment for soil moisture prediction.

This module provides quality control for station/sensor data,
assessing physics model performance and filtering problematic data.

Quality checks include:
- Observation data validity (realistic soil moisture values)
- Physics model bias assessment
- KGE and correlation analysis
- Automatic station filtering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("smps.data.quality.station_assessment")


@dataclass
class StationQualityThresholds:
    """Thresholds for station quality filtering."""

    max_abs_bias: float = 0.12  # Max absolute bias |obs_mean - physics_mean|
    # Min acceptable KGE (allow some negative for correction)
    min_kge: float = -0.5
    max_obs_mean_deviation: float = 0.05  # Max deviation from realistic range
    min_obs_mean: float = 0.05  # Min acceptable mean VWC (below is suspicious)
    # Max acceptable mean VWC (above is unrealistic)
    max_obs_mean: float = 0.55
    min_samples: int = 30  # Minimum samples for quality assessment


@dataclass
class StationQualityResult:
    """Result of station quality assessment."""

    station_id: str
    include: bool = True
    reasons: List[str] = field(default_factory=list)

    # Optional quality metrics by observation column
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            'station_id': self.station_id,
            'include': self.include,
            'reasons': self.reasons,
        }
        result.update(self.metrics)
        return result


class StationQualityAssessor:
    """
    Assess station quality based on physics model performance and data characteristics.

    Used to filter out problematic stations before ML training to improve
    model quality and avoid fitting to bad data.
    """

    def __init__(
        self,
        thresholds: Optional[StationQualityThresholds] = None,
        depth_physics_mapping: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize quality assessor.

        Args:
            thresholds: Quality thresholds for filtering
            depth_physics_mapping: Mapping from depth (cm) to physics column name
        """
        self.thresholds = thresholds or StationQualityThresholds()
        self.depth_physics_mapping = depth_physics_mapping or {
            10: 'physics_sm_surface',
            15: 'physics_sm_surface',
            20: 'physics_sm_root',
            30: 'physics_sm_root',
            60: 'physics_sm_deep',
            100: 'physics_sm_deep',
        }

    def assess_station(
        self,
        station_id: str,
        features_df: pd.DataFrame,
    ) -> StationQualityResult:
        """
        Assess quality of a single station.

        Args:
            station_id: Station identifier
            features_df: Feature DataFrame for this station

        Returns:
            StationQualityResult with include decision and reasons
        """
        result = StationQualityResult(station_id=station_id)

        # Find observation columns (raw, not lagged)
        obs_cols = self._get_observation_columns(features_df)

        for obs_col in obs_cols:
            if obs_col not in features_df.columns:
                continue

            obs_data = features_df[obs_col].dropna()
            if len(obs_data) < self.thresholds.min_samples:
                continue

            obs_mean = obs_data.mean()

            # Check 1: Unrealistic soil moisture values
            if obs_mean < self.thresholds.min_obs_mean:
                result.include = False
                result.reasons.append(
                    f'{obs_col} mean={obs_mean:.3f} too low (likely sensor issue)'
                )
            elif obs_mean > self.thresholds.max_obs_mean:
                result.include = False
                result.reasons.append(
                    f'{obs_col} mean={obs_mean:.3f} too high'
                )

            # Check 2: Physics model bias
            phys_col = self._get_physics_column_for_obs(obs_col)
            if phys_col and phys_col in features_df.columns:
                metrics = self._assess_physics_quality(
                    features_df, obs_col, phys_col
                )
                result.metrics[obs_col] = metrics

                # Apply thresholds
                if abs(metrics.get('bias', 0)) > self.thresholds.max_abs_bias:
                    result.include = False
                    result.reasons.append(
                        f'{obs_col} bias={metrics["bias"]:.3f} exceeds threshold'
                    )

                if metrics.get('kge', 0) < self.thresholds.min_kge:
                    result.include = False
                    result.reasons.append(
                        f'{obs_col} KGE={metrics["kge"]:.3f} below threshold'
                    )

        return result

    def assess_multiple_stations(
        self,
        combined_df: pd.DataFrame,
        station_col: str = 'station_id',
    ) -> Tuple[List[str], List[str], Dict[str, StationQualityResult]]:
        """
        Assess quality for all stations in a combined DataFrame.

        Args:
            combined_df: Combined DataFrame with all stations
            station_col: Column name for station identifier

        Returns:
            Tuple of (included_stations, excluded_stations, all_results)
        """
        station_ids = combined_df[station_col].unique()

        included = []
        excluded = []
        all_results = {}

        for station_id in station_ids:
            station_df = combined_df[combined_df[station_col] == station_id]
            result = self.assess_station(station_id, station_df)
            all_results[station_id] = result

            if result.include:
                included.append(station_id)
            else:
                excluded.append(station_id)
                logger.warning(
                    f"Excluding {station_id}: {', '.join(result.reasons)}"
                )

        logger.info(
            f"Quality assessment: {len(included)} included, {len(excluded)} excluded")

        return included, excluded, all_results

    def filter_dataframe(
        self,
        combined_df: pd.DataFrame,
        station_col: str = 'station_id',
    ) -> pd.DataFrame:
        """
        Filter a combined DataFrame to only include quality stations.

        Args:
            combined_df: Combined DataFrame with all stations
            station_col: Column name for station identifier

        Returns:
            Filtered DataFrame with only quality stations
        """
        included, excluded, _ = self.assess_multiple_stations(
            combined_df, station_col)

        original_len = len(combined_df)
        filtered_df = combined_df[combined_df[station_col].isin(
            included)].copy()

        logger.info(
            f"Filtered dataset: {len(filtered_df)} rows "
            f"(removed {original_len - len(filtered_df)})"
        )

        return filtered_df

    def _get_observation_columns(self, df: pd.DataFrame) -> List[str]:
        """Get raw observation columns (not lagged or derived)."""
        return [
            c for c in df.columns
            if c.startswith('obs_sm_')
            and '_lag' not in c
            and '_mean' not in c
            and '_std' not in c
            and '_change' not in c
            and '_memory' not in c
        ]

    def _get_physics_column_for_obs(self, obs_col: str) -> Optional[str]:
        """Get matching physics column for an observation column."""
        # Extract depth from obs column (e.g., 'obs_sm_10cm' -> 10)
        try:
            depth_str = obs_col.replace('obs_sm_', '').replace('cm', '')
            depth = int(depth_str)

            # Find closest mapping
            closest_depth = min(
                self.depth_physics_mapping.keys(),
                key=lambda x: abs(x - depth)
            )
            return self.depth_physics_mapping[closest_depth]
        except (ValueError, KeyError):
            return None

    def _assess_physics_quality(
        self,
        df: pd.DataFrame,
        obs_col: str,
        phys_col: str,
    ) -> Dict[str, float]:
        """Assess physics model quality against observations."""
        aligned = df[[obs_col, phys_col]].dropna()

        if len(aligned) < self.thresholds.min_samples:
            return {}

        obs = aligned[obs_col].values
        pred = aligned[phys_col].values

        # Calculate metrics
        bias = float(np.mean(pred - obs))

        # Correlation
        if np.std(obs) > 0 and np.std(pred) > 0:
            corr = float(np.corrcoef(obs, pred)[0, 1])
        else:
            corr = 0.0

        # KGE components
        r = corr
        alpha = float(np.std(pred) / np.std(obs)) if np.std(obs) > 0 else 1.0
        beta = float(np.mean(pred) / np.mean(obs)) if np.mean(obs) > 0 else 1.0
        kge = float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))

        return {
            'bias': bias,
            'correlation': corr,
            'kge': kge,
            'kge_r': r,
            'kge_alpha': alpha,
            'kge_beta': beta,
            'n_samples': len(aligned),
        }


def compute_physics_kge(
    physics_vals: np.ndarray,
    obs_vals: np.ndarray,
) -> float:
    """
    Compute KGE (Kling-Gupta Efficiency) for physics model.

    Standalone function for quick KGE calculation.

    Args:
        physics_vals: Physics model predictions
        obs_vals: Observations

    Returns:
        KGE value
    """
    valid_mask = ~np.isnan(physics_vals) & ~np.isnan(obs_vals)

    if np.sum(valid_mask) < 30:
        return 0.0

    phys = physics_vals[valid_mask]
    obs = obs_vals[valid_mask]

    r = np.corrcoef(obs, phys)[0, 1] if np.std(
        obs) > 0 and np.std(phys) > 0 else 0
    alpha = np.std(phys) / np.std(obs) if np.std(obs) > 0 else 1
    beta = np.mean(phys) / np.mean(obs) if np.mean(obs) > 0 else 1

    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    return float(kge)


def calculate_adaptive_physics_weight(
    physics_kge: float,
    physics_vals: np.ndarray,
    obs_vals: np.ndarray,
    horizon_days: int,
    depth_cm: int,
) -> float:
    """
    Calculate adaptive physics weight based on multiple quality metrics.

    Used in hybrid models to balance physics vs ML contributions.

    Args:
        physics_kge: Kling-Gupta Efficiency of physics model
        physics_vals: Physics predictions
        obs_vals: Observations
        horizon_days: Forecast horizon
        depth_cm: Soil depth in cm

    Returns:
        Weight between 0-1 (0 = pure ML, 1 = full physics trust)
    """
    # Base weight from KGE (clipped and scaled)
    base_weight = np.clip((physics_kge + 0.5) / 1.0, 0.0, 1.0)

    # Horizon penalty: longer horizons reduce physics trust
    horizon_penalty = max(0, 1.0 - horizon_days / 168.0)  # 168h = 7 days

    # Depth-specific adjustments
    depth_factor = 1.0
    if depth_cm <= 15:  # Surface layer - physics usually good
        depth_factor = 1.1
    elif depth_cm >= 100:  # Deep layer - physics often poor
        depth_factor = 0.8

    # Bias penalty: high bias reduces trust
    valid_mask = ~np.isnan(physics_vals) & ~np.isnan(obs_vals)
    if np.sum(valid_mask) > 0:
        bias = np.mean(physics_vals[valid_mask] - obs_vals[valid_mask])
        bias_penalty = max(0, 1.0 - abs(bias) / 0.1)  # 0.1 VWC bias threshold
    else:
        bias_penalty = 1.0

    # Combine factors
    weight = base_weight * horizon_penalty * depth_factor * bias_penalty
    weight = np.clip(weight, 0.0, 1.0)

    return float(weight)
