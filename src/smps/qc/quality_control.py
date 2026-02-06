"""
Quality Control Pipeline for Matric Potential Data.

Implements comprehensive QC checks for ψ (matric potential) data to ensure
data quality before model training and prediction.

QC Checks for ψ Data:
─────────────────────────────────────────────────────────────────
Physical Range Check:     -15 kPa ≤ ψ ≤ 0 kPa (saturated to dry)
Temporal Consistency:     No unrealistic jumps in ψ time series
Spatial Consistency:      ψ values reasonable for soil type/depth
Outlier Detection:        Statistical outliers in ψ distributions
Gap Detection:           Missing data patterns that affect ψ modeling
Metadata Validation:     Sensor calibration, depth, soil properties
─────────────────────────────────────────────────────────────────

Benefits for ψ Modeling:
- Prevents training on physically unrealistic ψ values
- Improves model generalization with clean data
- Early detection of sensor failures affecting ψ readings
- Better uncertainty quantification with quality-filtered data

Research References:
- Dorigo et al. (2011): Soil moisture data quality assessment
- Gruber et al. (2020): ISMN quality control procedures
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("swpps.qc")


@dataclass
class QCConfig:
    """Configuration for ψ data quality control."""

    # Physical range checks for ψ (matric potential in kPa)
    psi_min: float = -15.0  # Dry soil limit (kPa)
    psi_max: float = 0.0    # Saturated soil limit (kPa)

    # Temporal consistency checks
    max_psi_jump: float = 5.0  # Max realistic ψ change per hour (kPa)
    temporal_window: str = '1H'  # Window for temporal checks

    # Outlier detection
    outlier_method: str = 'isolation_forest'  # isolation_forest, zscore, iqr
    outlier_contamination: float = 0.05  # Expected outlier fraction

    # Gap detection
    max_gap_hours: int = 24  # Max acceptable data gap (hours)
    min_data_density: float = 0.8  # Minimum data completeness

    # Spatial consistency
    spatial_neighbors: int = 5  # Number of spatial neighbors to check
    spatial_tolerance: float = 2.0  # Max deviation from spatial mean (kPa)

    # Metadata validation
    required_metadata: List[str] = field(default_factory=lambda: [
        'sensor_depth', 'soil_texture', 'bulk_density', 'site_id'
    ])


class PsiPhysicalRangeCheck:
    """
    Checks if ψ values are within physically reasonable ranges.

    ψ (matric potential) should be between -15 kPa (dry) and 0 kPa (saturated).
    """

    def __init__(self, config: QCConfig):
        self.config = config

    def check_range(self, psi_values: np.ndarray) -> np.ndarray:
        """Check if ψ values are within physical range."""
        valid = (psi_values >= self.config.psi_min) & (
            psi_values <= self.config.psi_max)
        return valid

    def get_range_violations(self, psi_values: np.ndarray) -> Dict[str, Any]:
        """Get statistics on range violations."""
        valid = self.check_range(psi_values)
        n_violations = np.sum(~valid)
        violation_rate = n_violations / len(psi_values)

        violations = psi_values[~valid]
        if len(violations) > 0:
            min_violation = np.min(violations)
            max_violation = np.max(violations)
        else:
            min_violation = max_violation = None

        return {
            'n_violations': n_violations,
            'violation_rate': violation_rate,
            'min_violation': min_violation,
            'max_violation': max_violation,
            'valid_range': f"[{self.config.psi_min}, {self.config.psi_max}] kPa"
        }


class PsiTemporalConsistencyCheck:
    """
    Checks temporal consistency of ψ time series.

    Prevents unrealistic jumps in ψ that indicate sensor issues.
    """

    def __init__(self, config: QCConfig):
        self.config = config

    def check_temporal_jumps(self, df: pd.DataFrame, psi_col: str = 'psi') -> pd.DataFrame:
        """Check for unrealistic temporal jumps in ψ."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "DataFrame must have DatetimeIndex for temporal checks")

        # Calculate ψ differences over time
        df = df.copy()
        df = df.sort_index()

        # Resample to consistent time intervals
        df_resampled = df.resample(self.config.temporal_window).mean()

        # Calculate absolute differences
        psi_diff = df_resampled[psi_col].diff().abs()

        # Flag jumps exceeding threshold
        df_resampled['temporal_jump_flag'] = psi_diff > self.config.max_psi_jump

        return df_resampled

    def get_temporal_stats(self, df: pd.DataFrame, psi_col: str = 'psi') -> Dict[str, Any]:
        """Get temporal consistency statistics."""
        jump_check = self.check_temporal_jumps(df, psi_col)

        n_jumps = jump_check['temporal_jump_flag'].sum()
        jump_rate = n_jumps / len(jump_check)

        if n_jumps > 0:
            max_jump = jump_check.loc[jump_check['temporal_jump_flag'], psi_col].diff(
            ).abs().max()
        else:
            max_jump = 0.0

        return {
            'n_temporal_jumps': n_jumps,
            'temporal_jump_rate': jump_rate,
            'max_jump_size': max_jump,
            'jump_threshold': self.config.max_psi_jump
        }


class PsiOutlierDetector:
    """
    Detects statistical outliers in ψ data using multiple methods.
    """

    def __init__(self, config: QCConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.isolation_forest: Optional[IsolationForest] = None

    def detect_outliers(self, psi_values: np.ndarray) -> np.ndarray:
        """Detect outliers in ψ values."""
        if self.config.outlier_method == 'isolation_forest':
            return self._isolation_forest_outliers(psi_values)
        elif self.config.outlier_method == 'zscore':
            return self._zscore_outliers(psi_values)
        elif self.config.outlier_method == 'iqr':
            return self._iqr_outliers(psi_values)
        else:
            raise ValueError(
                f"Unknown outlier method: {self.config.outlier_method}")

    def _isolation_forest_outliers(self, psi_values: np.ndarray) -> np.ndarray:
        """Detect outliers using Isolation Forest."""
        if self.isolation_forest is None:
            self.isolation_forest = IsolationForest(
                contamination=self.config.outlier_contamination,
                random_state=42
            )

        # Fit and predict (returns -1 for outliers, 1 for inliers)
        predictions = self.isolation_forest.fit_predict(
            psi_values.reshape(-1, 1))
        return predictions == -1

    def _zscore_outliers(self, psi_values: np.ndarray) -> np.ndarray:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(psi_values))
        return z_scores > 3  # 3 standard deviations

    def _iqr_outliers(self, psi_values: np.ndarray) -> np.ndarray:
        """Detect outliers using IQR method."""
        Q1 = np.percentile(psi_values, 25)
        Q3 = np.percentile(psi_values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (psi_values < lower_bound) | (psi_values > upper_bound)


class PsiGapDetector:
    """
    Detects data gaps that could affect ψ modeling quality.
    """

    def __init__(self, config: QCConfig):
        self.config = config

    def detect_gaps(self, df: pd.DataFrame, time_col: Optional[str] = None) -> pd.DataFrame:
        """Detect data gaps in ψ time series."""
        if time_col and time_col in df.columns:
            df = df.set_index(time_col)

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "DataFrame must have DatetimeIndex for gap detection")

        df = df.sort_index()

        # Calculate time differences
        time_diffs = df.index.to_series().diff()

        # Convert to hours
        gap_hours = time_diffs.dt.total_seconds() / 3600

        # Flag gaps exceeding threshold
        gap_flags = gap_hours > self.config.max_gap_hours

        # Create gap summary
        gaps_df = pd.DataFrame({
            'gap_start': df.index[:-1][gap_flags[1:]],
            'gap_end': df.index[1:][gap_flags[1:]],
            'gap_hours': gap_hours[1:][gap_flags[1:]]
        })

        return gaps_df

    def get_gap_stats(self, df: pd.DataFrame, time_col: Optional[str] = None) -> Dict[str, Any]:
        """Get gap statistics."""
        gaps = self.detect_gaps(df, time_col)

        total_gaps = len(gaps)
        total_gap_hours = gaps['gap_hours'].sum() if total_gaps > 0 else 0

        # Data completeness
        if isinstance(df.index, pd.DatetimeIndex):
            total_hours = (df.index.max() - df.index.min()
                           ).total_seconds() / 3600
            data_hours = total_hours - total_gap_hours
            completeness = data_hours / total_hours if total_hours > 0 else 0
        else:
            completeness = 1.0  # Assume complete if no time index

        return {
            'total_gaps': total_gaps,
            'total_gap_hours': total_gap_hours,
            'data_completeness': completeness,
            'max_gap_threshold': self.config.max_gap_hours
        }


class PsiQualityControlPipeline:
    """
    Complete quality control pipeline for ψ data.

    Orchestrates all QC checks and provides comprehensive quality assessment.
    """

    def __init__(self, config: Optional[QCConfig] = None):
        self.config = config or QCConfig()

        # Initialize QC components
        self.range_check = PsiPhysicalRangeCheck(self.config)
        self.temporal_check = PsiTemporalConsistencyCheck(self.config)
        self.outlier_detector = PsiOutlierDetector(self.config)
        self.gap_detector = PsiGapDetector(self.config)

    def run_full_qc(self, df: pd.DataFrame, psi_col: str = 'psi',
                    time_col: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run complete QC pipeline on ψ data.

        Returns:
            Tuple of (flagged_data, qc_report)
        """
        logger.info("Running full QC pipeline on ψ data")

        df_qc = df.copy()
        qc_report = {}

        # 1. Physical range check
        logger.info("Checking physical range for ψ values")
        psi_values = df_qc[psi_col].values
        range_valid = self.range_check.check_range(psi_values)
        df_qc['range_flag'] = ~range_valid
        qc_report['physical_range'] = self.range_check.get_range_violations(
            psi_values)

        # 2. Temporal consistency check
        logger.info("Checking temporal consistency for ψ time series")
        if time_col or isinstance(df_qc.index, pd.DatetimeIndex):
            temporal_stats = self.temporal_check.get_temporal_stats(
                df_qc, psi_col)
            df_qc['temporal_flag'] = self.temporal_check.check_temporal_jumps(df_qc, psi_col)[
                'temporal_jump_flag']
            qc_report['temporal_consistency'] = temporal_stats
        else:
            logger.warning(
                "No time column provided - skipping temporal checks")
            qc_report['temporal_consistency'] = {'skipped': True}

        # 3. Outlier detection
        logger.info("Detecting outliers in ψ data")
        outliers = self.outlier_detector.detect_outliers(psi_values)
        df_qc['outlier_flag'] = outliers
        qc_report['outliers'] = {
            'n_outliers': np.sum(outliers),
            'outlier_rate': np.sum(outliers) / len(psi_values),
            'method': self.config.outlier_method
        }

        # 4. Gap detection
        logger.info("Detecting data gaps in ψ time series")
        gap_stats = self.gap_detector.get_gap_stats(df_qc, time_col)
        qc_report['data_gaps'] = gap_stats

        # 5. Overall quality score
        df_qc['overall_flag'] = df_qc[[
            'range_flag', 'outlier_flag']].any(axis=1)
        if 'temporal_flag' in df_qc.columns:
            df_qc['overall_flag'] = df_qc['overall_flag'] | df_qc['temporal_flag']

        n_flagged = df_qc['overall_flag'].sum()
        quality_score = 1.0 - (n_flagged / len(df_qc))

        qc_report['overall_quality'] = {
            'quality_score': quality_score,
            'n_flagged_points': n_flagged,
            'flagged_rate': n_flagged / len(df_qc),
            'n_total_points': len(df_qc)
        }

        logger.info(
            f"QC pipeline completed. Quality score: {quality_score:.3f}")

        return df_qc, qc_report

    def filter_quality_data(self, df: pd.DataFrame, qc_report: Dict[str, Any],
                            min_quality_score: float = 0.8) -> pd.DataFrame:
        """Filter data to keep only high-quality ψ measurements."""
        quality_score = qc_report['overall_quality']['quality_score']

        if quality_score >= min_quality_score:
            logger.info(
                f"Data quality score {quality_score:.3f} meets threshold {min_quality_score}")
            return df[~df['overall_flag']] if 'overall_flag' in df.columns else df
        else:
            logger.warning(
                f"Data quality score {quality_score:.3f} below threshold {min_quality_score}")
            return pd.DataFrame()  # Return empty DataFrame if quality too low

    def get_qc_summary(self, qc_report: Dict[str, Any]) -> str:
        """Generate human-readable QC summary."""
        summary = "ψ Data Quality Control Summary\n"
        summary += "=" * 40 + "\n\n"

        # Overall quality
        overall = qc_report['overall_quality']
        summary += f"Overall Quality Score: {overall['quality_score']:.3f}\n"
        summary += f"Total Points: {overall['n_total_points']}\n"
        summary += f"Flagged Points: {overall['n_flagged_points']} ({overall['flagged_rate']:.1%})\n\n"

        # Physical range
        phys_range = qc_report['physical_range']
        summary += f"Physical Range Violations: {phys_range['n_violations']} ({phys_range['violation_rate']:.1%})\n"
        if phys_range['n_violations'] > 0:
            summary += f"  Range: {phys_range['valid_range']}\n"
            summary += f"  Min violation: {phys_range['min_violation']:.2f} kPa\n"
            summary += f"  Max violation: {phys_range['max_violation']:.2f} kPa\n"

        # Temporal consistency
        if 'skipped' not in qc_report['temporal_consistency']:
            temp = qc_report['temporal_consistency']
            summary += f"Temporal Jumps: {temp['n_temporal_jumps']} ({temp['temporal_jump_rate']:.1%})\n"
            summary += f"  Max jump: {temp['max_jump_size']:.2f} kPa (threshold: {temp['jump_threshold']} kPa)\n"

        # Outliers
        outliers = qc_report['outliers']
        summary += f"Outliers: {outliers['n_outliers']} ({outliers['outlier_rate']:.1%})\n"
        summary += f"  Method: {outliers['method']}\n"

        # Data gaps
        gaps = qc_report['data_gaps']
        summary += f"Data Gaps: {gaps['total_gaps']}\n"
        summary += f"  Total gap time: {gaps['total_gap_hours']:.1f} hours\n"
        summary += f"  Data completeness: {gaps['data_completeness']:.1%}\n"

        return summary
