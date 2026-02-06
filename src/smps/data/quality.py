"""
Data Quality Control for SWPPS.

Provides quality control pipeline for:
- Matric potential measurements
- Sensor data validation
- Weather data QC
- Temporal consistency checks
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger("swpps.data.quality")


@dataclass
class QCConfig:
    """Configuration for quality control."""

    # Matric potential bounds (kPa)
    psi_min: float = -2000.0  # Below wilting point
    psi_max: float = 10.0     # Slightly positive allowed

    # Water content bounds (m³/m³)
    theta_min: float = 0.0
    theta_max: float = 0.65   # Very high porosity soils

    # Weather bounds
    precip_max_mm: float = 500.0     # Max daily precip
    et_min_mm: float = 0.0
    et_max_mm: float = 20.0          # Max daily ET
    temp_min_c: float = -40.0
    temp_max_c: float = 60.0
    humidity_min: float = 0.0
    humidity_max: float = 100.0
    wind_max_m_s: float = 50.0
    radiation_max_mj_m2: float = 40.0

    # Spike detection
    spike_threshold_std: float = 3.0  # Standard deviations

    # Temporal consistency
    max_gap_hours: float = 48.0       # Flag gaps larger than this
    min_coverage: float = 0.7         # Minimum valid data fraction

    # Weather consistency checks
    max_consecutive_same_temp: int = 7  # Max days with identical temperature
    max_consecutive_zero_precip: int = 30  # Max days with zero precipitation
    # Minimum ET variability (coefficient of variation)
    min_et_variability: float = 0.1


@dataclass
class QCResult:
    """Result of quality control."""

    n_total: int = 0
    n_flagged: int = 0
    n_valid: int = 0

    flag_counts: Dict[str, int] = field(default_factory=dict)
    coverage: float = 1.0
    column_coverage: Dict[str, float] = field(default_factory=dict)

    quality_score: float = 1.0  # 0-1 score

    issues: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"QC Result: {self.n_valid}/{self.n_total} valid ({self.quality_score:.1%})",
            f"Flagged: {self.n_flagged}",
        ]

        if self.flag_counts:
            for flag, count in sorted(self.flag_counts.items()):
                lines.append(f"  - {flag}: {count}")

        if self.issues:
            lines.append("Issues:")
            for issue in self.issues[:5]:
                lines.append(f"  ⚠ {issue}")

        return "\n".join(lines)


class QCFlags:
    """Quality control flag definitions."""

    GOOD = 0
    RANGE_WARNING = 1       # Soft range violation
    RANGE_ERROR = 2         # Hard range violation
    SPIKE_DETECTED = 4      # Rapid unrealistic change
    TEMPORAL_GAP = 8        # Large time gap
    MISSING_DATA = 16       # Required field missing
    SENSOR_SUSPECT = 32     # Sensor behavior suspicious
    DUPLICATE = 64          # Duplicate timestamp


class QualityControlPipeline:
    """
    Quality control pipeline for SWPPS data.

    Applies comprehensive checks including:
    - Physical range validation
    - Spike detection
    - Temporal consistency
    - Sensor plausibility
    """

    def __init__(self, config: Optional[QCConfig] = None):
        """
        Initialize QC pipeline.

        Args:
            config: QC configuration
        """
        self.config = config or QCConfig()

    def run(
        self,
        df: pd.DataFrame,
        psi_col: str = "psi_kpa",
        time_col: str = "datetime",
    ) -> Tuple[pd.DataFrame, QCResult]:
        """
        Run full QC pipeline.

        Args:
            df: Input DataFrame
            psi_col: Column name for matric potential
            time_col: Column name for timestamps

        Returns:
            Tuple of (flagged DataFrame, QCResult)
        """
        result = df.copy()

        # Initialize flag column
        if "qc_flag" not in result.columns:
            result["qc_flag"] = QCFlags.GOOD

        # Apply checks
        result = self._check_missing_data(result, psi_col)
        result = self._check_physical_ranges(result, psi_col)
        result = self._detect_spikes(result, psi_col)
        result = self._check_temporal_consistency(result, time_col)
        result = self._check_duplicates(result, time_col)

        # Compute result summary
        qc_result = self._compute_summary(result, psi_col)

        return result, qc_result

    def _check_missing_data(
        self,
        df: pd.DataFrame,
        psi_col: str,
    ) -> pd.DataFrame:
        """Flag rows with missing critical data."""
        result = df.copy()

        if psi_col in result.columns:
            missing = result[psi_col].isna()
            result.loc[missing, "qc_flag"] |= QCFlags.MISSING_DATA

            n_missing = missing.sum()
            if n_missing > 0:
                logger.debug("Missing %s: %d rows", psi_col, n_missing)

        return result

    def _check_physical_ranges(
        self,
        df: pd.DataFrame,
        psi_col: str,
    ) -> pd.DataFrame:
        """Check values are within physical bounds."""
        result = df.copy()

        # Check matric potential
        if psi_col in result.columns:
            values = result[psi_col]

            # Soft range (warning)
            soft_low = values < self.config.psi_min * 0.9
            soft_high = values > self.config.psi_max * 1.1
            result.loc[soft_low | soft_high,
                       "qc_flag"] |= QCFlags.RANGE_WARNING

            # Hard range (error)
            hard_low = values < self.config.psi_min * 1.5
            hard_high = values > self.config.psi_max * 2
            result.loc[hard_low | hard_high, "qc_flag"] |= QCFlags.RANGE_ERROR

        # Check water content columns
        theta_cols = [c for c in result.columns if "theta" in c.lower()]
        for col in theta_cols:
            values = result[col]
            out_of_range = (values < self.config.theta_min) | (
                values > self.config.theta_max)
            result.loc[out_of_range, "qc_flag"] |= QCFlags.RANGE_WARNING

        # Check weather columns
        weather_cols = ["precipitation_mm", "et0_mm", "temperature_2m",
                        "relative_humidity_2m", "wind_speed_10m", "shortwave_radiation"]
        for col in weather_cols:
            if col in result.columns:
                values = result[col]
                if col == "precipitation_mm":
                    invalid = (values < 0) | (
                        values > self.config.precip_max_mm)
                    result.loc[invalid, "qc_flag"] |= QCFlags.RANGE_ERROR
                elif col == "et0_mm":
                    invalid = (values < self.config.et_min_mm) | (
                        values > self.config.et_max_mm)
                    result.loc[invalid, "qc_flag"] |= QCFlags.RANGE_WARNING
                elif col == "temperature_2m":
                    invalid = (values < self.config.temp_min_c) | (
                        values > self.config.temp_max_c)
                    result.loc[invalid, "qc_flag"] |= QCFlags.RANGE_ERROR
                elif col == "relative_humidity_2m":
                    invalid = (values < self.config.humidity_min) | (
                        values > self.config.humidity_max)
                    result.loc[invalid, "qc_flag"] |= QCFlags.RANGE_WARNING
                elif col == "wind_speed_10m":
                    invalid = (values < 0) | (
                        values > self.config.wind_max_m_s)
                    result.loc[invalid, "qc_flag"] |= QCFlags.RANGE_WARNING
                elif col == "shortwave_radiation":
                    invalid = (values < 0) | (
                        values > self.config.radiation_max_mj_m2)
                    result.loc[invalid, "qc_flag"] |= QCFlags.RANGE_WARNING

        # Weather consistency checks
        result = self._check_weather_consistency(result)

        return result

    def _check_weather_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check weather data for consistency and plausibility."""
        result = df.copy()

        # Check for consecutive identical temperatures (possible sensor freeze)
        if "temperature_2m" in result.columns:
            temp_series = result["temperature_2m"]
            # Find runs of identical values
            temp_diff = temp_series.diff().fillna(0)
            identical_runs = (temp_diff == 0).groupby(
                (temp_diff != 0).cumsum()).cumsum()

            suspicious_temp = identical_runs > self.config.max_consecutive_same_temp
            result.loc[suspicious_temp, "qc_flag"] |= QCFlags.SENSOR_SUSPECT

            n_suspicious = suspicious_temp.sum()
            if n_suspicious > 0:
                logger.debug("Found %d suspicious temperature runs (> %d consecutive identical values)",
                             n_suspicious, self.config.max_consecutive_same_temp)

        # Check for excessively long dry periods
        if "precipitation_mm" in result.columns:
            precip_series = result["precipitation_mm"]
            zero_precip = (precip_series == 0) | precip_series.isna()
            zero_runs = zero_precip.groupby(
                (~zero_precip).cumsum()).cumsum()

            excessive_dry = zero_runs > self.config.max_consecutive_zero_precip
            result.loc[excessive_dry, "qc_flag"] |= QCFlags.SENSOR_SUSPECT

            n_excessive = excessive_dry.sum()
            if n_excessive > 0:
                logger.debug("Found %d days in excessive dry periods (> %d consecutive zero precip)",
                             n_excessive, self.config.max_consecutive_zero_precip)

        # Check ET variability (should vary with weather conditions)
        if "et0_mm" in result.columns:
            et_series = result["et0_mm"]
            if et_series.notna().sum() > 10:  # Need enough data
                et_cv = et_series.std() / et_series.mean() if et_series.mean() > 0 else 0
                if et_cv < self.config.min_et_variability:
                    # Flag all ET values as suspicious if too constant
                    result.loc[et_series.notna(
                    ), "qc_flag"] |= QCFlags.SENSOR_SUSPECT
                    logger.debug("ET variability too low (CV=%.3f < %.3f), possible sensor issue",
                                 et_cv, self.config.min_et_variability)

        # Check temperature-humidity consistency (very low humidity with high temps is suspicious)
        if "temperature_2m" in result.columns and "relative_humidity_2m" in result.columns:
            temp = result["temperature_2m"]
            humidity = result["relative_humidity_2m"]

            # Very hot and very dry is suspicious
            hot_and_dry = (temp > 35) & (humidity < 10)
            result.loc[hot_and_dry, "qc_flag"] |= QCFlags.RANGE_WARNING

            n_hot_dry = hot_and_dry.sum()
            if n_hot_dry > 0:
                logger.debug(
                    "Found %d hot-and-dry conditions (T>35°C & RH<10%%)", n_hot_dry)

        return result

    def _detect_spikes(
        self,
        df: pd.DataFrame,
        psi_col: str,
    ) -> pd.DataFrame:
        """Detect unrealistic rapid changes."""
        result = df.copy()

        if psi_col not in result.columns:
            return result

        values = result[psi_col]

        if values.notna().sum() < 3:
            return result

        # Compute first difference
        diff = values.diff().abs()

        # Compute threshold
        threshold = diff.mean() + self.config.spike_threshold_std * diff.std()

        # Flag spikes
        spikes = diff > threshold
        result.loc[spikes, "qc_flag"] |= QCFlags.SPIKE_DETECTED

        n_spikes = spikes.sum()
        if n_spikes > 0:
            logger.debug("Detected %d potential spikes in %s",
                         n_spikes, psi_col)

        return result

    def _check_temporal_consistency(
        self,
        df: pd.DataFrame,
        time_col: str,
    ) -> pd.DataFrame:
        """Check for temporal gaps and consistency."""
        result = df.copy()

        if time_col not in result.columns:
            return result

        # Sort by time
        result = result.sort_values(time_col)

        # Compute time gaps
        times = pd.to_datetime(result[time_col])
        gaps_hours = times.diff().dt.total_seconds() / 3600

        # Flag large gaps
        large_gaps = gaps_hours > self.config.max_gap_hours
        result.loc[large_gaps, "qc_flag"] |= QCFlags.TEMPORAL_GAP

        n_gaps = large_gaps.sum()
        if n_gaps > 0:
            logger.debug("Found %d temporal gaps > %d hours",
                         n_gaps, self.config.max_gap_hours)

        return result

    def _check_duplicates(
        self,
        df: pd.DataFrame,
        time_col: str,
    ) -> pd.DataFrame:
        """Check for duplicate timestamps."""
        result = df.copy()

        if time_col not in result.columns:
            return result

        duplicates = result[time_col].duplicated(keep="first")
        result.loc[duplicates, "qc_flag"] |= QCFlags.DUPLICATE

        n_dups = duplicates.sum()
        if n_dups > 0:
            logger.warning("Found %d duplicate timestamps", n_dups)

        return result

    def _compute_summary(
        self,
        df: pd.DataFrame,
        psi_col: str,
    ) -> QCResult:
        """Compute QC summary statistics."""
        n_total = len(df)
        n_flagged = (df["qc_flag"] > QCFlags.GOOD).sum()
        n_valid = n_total - n_flagged

        # Count by flag type
        flag_counts = {}
        for flag_name, flag_value in [
            ("missing", QCFlags.MISSING_DATA),
            ("range_warning", QCFlags.RANGE_WARNING),
            ("range_error", QCFlags.RANGE_ERROR),
            ("spike", QCFlags.SPIKE_DETECTED),
            ("temporal_gap", QCFlags.TEMPORAL_GAP),
            ("duplicate", QCFlags.DUPLICATE),
        ]:
            count = ((df["qc_flag"] & flag_value) > 0).sum()
            if count > 0:
                flag_counts[flag_name] = count

        # Compute coverage
        column_coverage = {}
        for col in df.columns:
            if col != "qc_flag":
                column_coverage[col] = df[col].notna().mean()

        coverage = df[psi_col].notna().mean() if psi_col in df.columns else 1.0

        # Quality score (simple heuristic)
        quality_score = max(0, 1.0 - (n_flagged / n_total)
                            if n_total > 0 else 1.0)

        # Generate issues list
        issues = []
        if n_flagged > n_total * 0.2:
            issues.append(
                f"High flagged rate: {n_flagged}/{n_total} ({100*n_flagged/n_total:.1f}%)")
        if coverage < self.config.min_coverage:
            issues.append(f"Low data coverage: {coverage:.1%}")

        return QCResult(
            n_total=n_total,
            n_flagged=n_flagged,
            n_valid=n_valid,
            flag_counts=flag_counts,
            coverage=coverage,
            column_coverage=column_coverage,
            quality_score=quality_score,
            issues=issues,
        )

    def filter_valid(
        self,
        df: pd.DataFrame,
        max_flag: int = QCFlags.RANGE_WARNING,
    ) -> pd.DataFrame:
        """
        Filter to valid rows only.

        Args:
            df: DataFrame with qc_flag column
            max_flag: Maximum acceptable flag value

        Returns:
            Filtered DataFrame
        """
        if "qc_flag" not in df.columns:
            return df

        return df[df["qc_flag"] <= max_flag].copy()


def run_qc_pipeline(
    df: pd.DataFrame,
    config: Optional[QCConfig] = None,
    psi_col: str = "psi_kpa",
) -> Tuple[pd.DataFrame, QCResult]:
    """
    Convenience function to run QC pipeline.

    Args:
        df: Input DataFrame
        config: QC configuration
        psi_col: Column name for matric potential

    Returns:
        Tuple of (flagged DataFrame, QCResult)
    """
    pipeline = QualityControlPipeline(config)
    return pipeline.run(df, psi_col=psi_col)


class WeatherGapFiller:
    """
    Gap-filling strategies for weather data.

    Supports multiple interpolation and statistical methods.
    """

    def __init__(self, config: Optional[QCConfig] = None):
        self.config = config or QCConfig()

    def fill_gaps(
        self,
        df: pd.DataFrame,
        max_gap_days: int = 7,
        method: str = "auto"
    ) -> pd.DataFrame:
        """
        Fill gaps in weather data.

        Args:
            df: DataFrame with weather columns
            max_gap_days: Maximum gap size to fill
            method: Filling method ('linear', 'spline', 'climatology', 'auto')

        Returns:
            DataFrame with gaps filled
        """
        result = df.copy()

        # Ensure datetime index
        if 'date' in result.columns:
            result = result.set_index('date').sort_index()

        weather_cols = ["precipitation_mm", "et0_mm", "temperature_2m",
                        "relative_humidity_2m", "wind_speed_10m", "shortwave_radiation"]

        for col in weather_cols:
            if col not in result.columns:
                continue

            series = result[col]
            n_missing = series.isna().sum()

            if n_missing == 0:
                continue

            logger.debug("Filling %d gaps in %s", n_missing, col)

            # Choose filling method
            if method == "auto":
                fill_method = self._choose_method(col, series)
            else:
                fill_method = method

            # Apply filling
            if fill_method == "linear":
                result[col] = self._linear_interpolation(series, max_gap_days)
            elif fill_method == "spline":
                result[col] = self._spline_interpolation(series, max_gap_days)
            elif fill_method == "climatology":
                result[col] = self._climatological_fill(series)
            elif fill_method == "forward_fill":
                result[col] = series.ffill(limit=max_gap_days)
            elif fill_method == "backward_fill":
                result[col] = series.bfill(limit=max_gap_days)

        return result.reset_index() if 'date' in result.index.names else result

    def _choose_method(self, col: str, series: pd.Series) -> str:
        """Choose appropriate filling method based on variable type."""
        if col == "precipitation_mm":
            # Precipitation is event-based, use climatology or forward-fill short gaps
            return "climatology" if series.isna().sum() > 10 else "forward_fill"
        elif col in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]:
            # Smooth variables, use interpolation
            return "linear"
        elif col in ["et0_mm", "shortwave_radiation"]:
            # Weather-dependent, use spline interpolation
            return "spline"
        else:
            return "linear"

    def _linear_interpolation(self, series: pd.Series, max_gap: int) -> pd.Series:
        """Linear interpolation with gap limit."""
        return series.interpolate(method='linear', limit=max_gap, limit_direction='both')

    def _spline_interpolation(self, series: pd.Series, max_gap: int) -> pd.Series:
        """Spline interpolation for smooth variables."""
        try:
            return series.interpolate(method='spline', order=3, limit=max_gap, limit_direction='both')
        except:
            # Fallback to linear if spline fails
            return self._linear_interpolation(series, max_gap)

    def _climatological_fill(self, series: pd.Series) -> pd.Series:
        """
        Fill using climatological averages.

        Uses day-of-year averages from available data.
        """
        if series.isna().all():
            return series

        # Compute day-of-year averages
        valid_data = series.dropna()
        if len(valid_data) == 0:
            return series

        # Group by day of year
        doy_avg = valid_data.groupby(valid_data.index.dayofyear).mean()

        # Fill missing values
        filled = series.copy()
        for idx in series[series.isna()].index:
            doy = idx.dayofyear
            if doy in doy_avg.index:
                filled.loc[idx] = doy_avg.loc[doy]

        return filled


def run_weather_qc(
    df: pd.DataFrame,
    config: Optional[QCConfig] = None,
    fill_gaps: bool = False,
    max_gap_days: int = 7
) -> Tuple[pd.DataFrame, QCResult]:
    """
    Run weather-specific quality control and optional gap filling.

    Args:
        df: DataFrame with weather columns
        config: QC configuration
        fill_gaps: Whether to fill gaps after QC
        max_gap_days: Maximum gap size for filling

    Returns:
        Tuple of (processed DataFrame, QCResult)
    """
    # Run QC
    qc_config = config or QCConfig()
    pipeline = QualityControlPipeline(qc_config)
    # No psi column for weather-only QC
    flagged_df, qc_result = pipeline.run(df, psi_col="dummy")

    # Optional gap filling
    if fill_gaps:
        filler = WeatherGapFiller(qc_config)
        flagged_df = filler.fill_gaps(flagged_df, max_gap_days=max_gap_days)

    return flagged_df, qc_result
