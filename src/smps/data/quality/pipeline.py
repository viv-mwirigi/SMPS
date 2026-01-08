"""Quality control pipeline for soil moisture data.

This module provides the QualityControlPipeline class for flagging
and filtering problematic data in soil moisture datasets.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any
import logging

from smps.core.types import SiteID


class QualityControlPipeline:
    """
    Quality control pipeline for soil moisture data.

    Applies a series of checks to flag or remove suspicious data points
    including range checks, spike detection, and temporal consistency.
    """

    # Physical bounds for soil moisture (volumetric water content)
    DEFAULT_THETA_MIN = 0.0
    DEFAULT_THETA_MAX = 0.6  # Saturated porosity rarely exceeds 0.6 m³/m³

    # Weather bounds
    DEFAULT_PRECIP_MAX = 500  # mm/day (extreme but possible)
    DEFAULT_ET0_MIN = 0.0
    DEFAULT_ET0_MAX = 20.0   # mm/day

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize quality control pipeline.

        Args:
            config: Optional configuration dict with QC thresholds.
        """
        self.config = config or {}
        self.logger = logging.getLogger("smps.data.quality")

        # Configurable thresholds
        self.theta_min = self.config.get("theta_min", self.DEFAULT_THETA_MIN)
        self.theta_max = self.config.get("theta_max", self.DEFAULT_THETA_MAX)
        self.precip_max = self.config.get(
            "precip_max", self.DEFAULT_PRECIP_MAX)
        self.et0_min = self.config.get("et0_min", self.DEFAULT_ET0_MIN)
        self.et0_max = self.config.get("et0_max", self.DEFAULT_ET0_MAX)

        # Spike detection sensitivity
        self.spike_threshold = self.config.get(
            "spike_threshold", 3.0)  # std deviations

        # Minimum data coverage required
        self.min_coverage = self.config.get("min_coverage", 0.7)

    def run(self, df: pd.DataFrame, site_id: Optional[SiteID] = None) -> pd.DataFrame:
        """
        Run full QC pipeline on DataFrame.

        Args:
            df: DataFrame with soil moisture and weather data.
            site_id: Optional site identifier for logging.

        Returns:
            DataFrame with QC flags added.
        """
        self.logger.debug("Running QC pipeline for site %s", site_id)

        result = df.copy()

        # Initialize QC flag column
        result['qc_flag'] = 0

        # Apply range checks
        result = self._check_physical_ranges(result)

        # Apply spike detection on soil moisture
        result = self._detect_spikes(result)

        # Apply temporal consistency checks
        result = self._check_temporal_consistency(result)

        # Calculate overall data coverage
        result = self._calculate_coverage(result)

        # Log QC summary
        flagged_count = (result['qc_flag'] > 0).sum()
        self.logger.debug(
            "QC complete: %d/%d rows flagged (%.1f%%)",
            flagged_count, len(result), 100 * flagged_count /
            len(result) if len(result) > 0 else 0
        )

        return result

    def filter_flagged(self, df: pd.DataFrame,
                       max_flag_level: int = 1) -> pd.DataFrame:
        """
        Remove rows exceeding the flag threshold.

        Args:
            df: DataFrame with qc_flag column.
            max_flag_level: Maximum acceptable flag level.

        Returns:
            DataFrame with flagged rows removed.
        """
        if 'qc_flag' not in df.columns:
            return df

        return df[df['qc_flag'] <= max_flag_level].copy()

    def _check_physical_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check variables are within physical bounds."""
        result = df.copy()

        # Check soil moisture columns
        theta_cols = [c for c in result.columns if c.startswith('physics_theta') or
                      c.startswith('theta')]

        for col in theta_cols:
            out_of_range = (result[col] < self.theta_min) | (
                result[col] > self.theta_max)
            result.loc[out_of_range,
                       'qc_flag'] = result.loc[out_of_range, 'qc_flag'] + 1

            if out_of_range.any():
                self.logger.debug(
                    "%d rows with %s outside [%.2f, %.2f]",
                    out_of_range.sum(), col, self.theta_min, self.theta_max
                )

        # Check precipitation
        if 'precipitation_mm' in result.columns:
            invalid_precip = (result['precipitation_mm'] < 0) | (
                result['precipitation_mm'] > self.precip_max)
            result.loc[invalid_precip,
                       'qc_flag'] = result.loc[invalid_precip, 'qc_flag'] + 1

        # Check ET0
        if 'et0_mm' in result.columns:
            invalid_et0 = (result['et0_mm'] < self.et0_min) | (
                result['et0_mm'] > self.et0_max)
            result.loc[invalid_et0,
                       'qc_flag'] = result.loc[invalid_et0, 'qc_flag'] + 1

        return result

    def _detect_spikes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect unrealistic spikes in soil moisture time series."""
        result = df.copy()

        theta_cols = [c for c in result.columns if 'theta' in c.lower() and
                      result[c].dtype in [np.float64, np.float32]]

        for col in theta_cols:
            if result[col].notna().sum() < 3:
                continue

            # Calculate day-to-day differences
            diff = result[col].diff().abs()

            # Flag spikes > threshold standard deviations from mean change
            threshold = diff.mean() + self.spike_threshold * diff.std()
            spikes = diff > threshold

            result.loc[spikes, 'qc_flag'] = result.loc[spikes, 'qc_flag'] + 2

            if spikes.any():
                self.logger.debug(
                    "%d potential spikes detected in %s",
                    spikes.sum(), col
                )

        return result

    def _check_temporal_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for temporal consistency in the data."""
        result = df.copy()

        if 'date' not in result.columns:
            return result

        # Sort by date
        result = result.sort_values('date')

        # Check for duplicate dates
        if result['date'].duplicated().any():
            self.logger.warning("Duplicate dates found in data")
            result.loc[result['date'].duplicated(keep='first'), 'qc_flag'] += 4

        # Check for large gaps (might indicate sensor issues)
        dates = pd.to_datetime(result['date'])
        gaps = dates.diff().dt.days
        large_gaps = gaps > 7  # Flag if gap > 7 days

        if large_gaps.any():
            self.logger.debug(
                "%d large time gaps (>7 days) found", large_gaps.sum())

        return result

    def _calculate_coverage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate and store data coverage metrics."""
        result = df.copy()

        # Key columns to check coverage
        key_cols = ['precipitation_mm', 'et0_mm',
                    'physics_theta_root', 'physics_theta_surface']
        key_cols = [c for c in key_cols if c in result.columns]

        coverage_dict = {}
        for col in key_cols:
            coverage_dict[col] = result[col].notna().mean()

        result.attrs['column_coverage'] = coverage_dict
        result.attrs['overall_coverage'] = np.mean(
            list(coverage_dict.values())) if coverage_dict else 1.0

        return result

    def get_qc_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate QC summary statistics.

        Args:
            df: DataFrame after QC pipeline.

        Returns:
            Dict with QC summary statistics.
        """
        summary = {
            'total_rows': len(df),
            'flagged_rows': (df['qc_flag'] > 0).sum() if 'qc_flag' in df.columns else 0,
            'coverage': df.attrs.get('overall_coverage', None),
            'column_coverage': df.attrs.get('column_coverage', {})
        }

        if 'qc_flag' in df.columns:
            summary['flag_distribution'] = df['qc_flag'].value_counts().to_dict()

        return summary
