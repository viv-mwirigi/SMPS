"""
Validation Report Generation for SWPPS.

Generates comprehensive validation reports including:
- Summary statistics
- Per-site and per-horizon breakdowns
- Quality flags and recommendations
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import json
import logging

from .metrics import ValidationMetrics, compute_metrics, compute_per_site_metrics

logger = logging.getLogger("swpps.validation.report")


@dataclass
class ValidationReport:
    """
    Comprehensive validation report for model evaluation.
    """
    # Report metadata
    report_id: str = ""
    generated_at: str = ""
    model_version: str = ""

    # Overall metrics
    overall_metrics: Optional[ValidationMetrics] = None

    # Breakdown by horizon
    horizon_metrics: List[ValidationMetrics] = field(default_factory=list)

    # Breakdown by site
    site_metrics: Dict[str, ValidationMetrics] = field(default_factory=dict)

    # Sample counts
    total_samples: int = 0
    valid_samples: int = 0
    excluded_samples: int = 0

    # Data quality
    quality_flags: List[str] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=" * 60,
            "SWPPS Validation Report",
            "=" * 60,
            f"Report ID: {self.report_id}",
            f"Generated: {self.generated_at}",
            f"Model: {self.model_version}",
            "",
            "--- Sample Statistics ---",
            f"Total samples: {self.total_samples:,}",
            f"Valid samples: {self.valid_samples:,}",
            f"Excluded: {self.excluded_samples:,}",
            "",
            "--- Overall Performance ---",
        ]

        if self.overall_metrics:
            m = self.overall_metrics
            lines.extend([
                f"RMSE: {m.rmse:.2f} kPa",
                f"MAE: {m.mae:.2f} kPa",
                f"Bias: {m.mbe:+.2f} kPa",
                f"R²: {m.r_squared:.3f}",
                f"KGE: {m.kge:.3f}",
                f"NSE: {m.nse:.3f}",
                f"ubRMSE: {m.ubrmse:.2f} kPa",
            ])

        if self.horizon_metrics:
            lines.extend(["", "--- Performance by Horizon ---"])
            for hm in self.horizon_metrics:
                lines.append(
                    f"  {hm.horizon_hours:3d}h: RMSE={hm.rmse:6.2f} kPa, "
                    f"R²={hm.r_squared:.3f}, KGE={hm.kge:+.3f}"
                )

        if self.quality_flags:
            lines.extend(["", "--- Quality Flags ---"])
            for flag in self.quality_flags:
                lines.append(f"  ⚠ {flag}")

        if self.recommendations:
            lines.extend(["", "--- Recommendations ---"])
            for rec in self.recommendations:
                lines.append(f"  → {rec}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "model_version": self.model_version,
            "sample_counts": {
                "total": self.total_samples,
                "valid": self.valid_samples,
                "excluded": self.excluded_samples,
            },
            "overall_metrics": self.overall_metrics.to_dict() if self.overall_metrics else None,
            "horizon_metrics": [m.to_dict() for m in self.horizon_metrics],
            "site_metrics": {k: v.to_dict() for k, v in self.site_metrics.items()},
            "quality_flags": self.quality_flags,
            "recommendations": self.recommendations,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, filepath: str) -> None:
        """Save report to file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("Saved validation report to %s", filepath)


def generate_report(
    observed: np.ndarray,
    predicted: np.ndarray,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    horizons: Optional[np.ndarray] = None,
    site_ids: Optional[np.ndarray] = None,
    model_version: str = "unknown",
) -> ValidationReport:
    """
    Generate comprehensive validation report.

    Args:
        observed: Observed matric potential (kPa)
        predicted: Predicted matric potential (kPa)
        lower_bound: Lower prediction bound (optional)
        upper_bound: Upper prediction bound (optional)
        horizons: Forecast horizons in hours (optional)
        site_ids: Site identifiers (optional)
        model_version: Model version string

    Returns:
        ValidationReport instance
    """
    report = ValidationReport(
        report_id=f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        generated_at=datetime.now().isoformat(),
        model_version=model_version,
    )

    # Convert to arrays
    obs = np.asarray(observed).flatten()
    pred = np.asarray(predicted).flatten()

    report.total_samples = len(obs)

    # Valid mask
    valid = np.isfinite(obs) & np.isfinite(pred)
    report.valid_samples = int(np.sum(valid))
    report.excluded_samples = report.total_samples - report.valid_samples

    # Overall metrics
    report.overall_metrics = compute_metrics(
        obs, pred, lower_bound, upper_bound
    )

    # Per-horizon metrics
    if horizons is not None:
        hz = np.asarray(horizons).flatten()
        unique_horizons = np.unique(hz[np.isfinite(hz)])

        for h in sorted(unique_horizons):
            mask = hz == h
            h_metrics = compute_metrics(
                obs[mask], pred[mask],
                lower_bound[mask] if lower_bound is not None else None,
                upper_bound[mask] if upper_bound is not None else None,
                horizon_hours=int(h),
            )
            report.horizon_metrics.append(h_metrics)

    # Per-site metrics
    if site_ids is not None:
        sites = np.asarray(site_ids).flatten()
        unique_sites = np.unique(sites)

        for site in unique_sites:
            if pd.isna(site):
                continue
            mask = sites == site
            s_metrics = compute_metrics(
                obs[mask], pred[mask],
                lower_bound[mask] if lower_bound is not None else None,
                upper_bound[mask] if upper_bound is not None else None,
            )
            report.site_metrics[str(site)] = s_metrics

    # Quality assessment
    _assess_quality(report)

    # Generate recommendations
    _generate_recommendations(report)

    return report


def _assess_quality(report: ValidationReport) -> None:
    """Assess data and model quality, add flags."""
    m = report.overall_metrics

    if m is None:
        return

    # Check sample size
    if report.valid_samples < 100:
        report.quality_flags.append(
            f"Small sample size ({report.valid_samples}). Results may be unreliable."
        )

    # Check exclusion rate
    if report.excluded_samples > 0:
        excl_rate = 100 * report.excluded_samples / report.total_samples
        if excl_rate > 10:
            report.quality_flags.append(
                f"High exclusion rate ({excl_rate:.1f}%). Check data quality."
            )

    # Check bias
    if abs(m.mbe) > 10:
        report.quality_flags.append(
            f"Large systematic bias ({m.mbe:+.1f} kPa). Consider bias correction."
        )

    # Check temporal structure
    if np.isfinite(m.autocorr_error) and m.autocorr_error > 0.2:
        report.quality_flags.append(
            f"Poor temporal structure preservation (AC error={m.autocorr_error:.2f})."
        )

    # Check uncertainty calibration
    if np.isfinite(m.coverage_90):
        if m.coverage_90 < 80:
            report.quality_flags.append(
                f"Under-confident predictions (90% coverage={m.coverage_90:.0f}%)."
            )
        elif m.coverage_90 > 98:
            report.quality_flags.append(
                f"Over-confident predictions (90% coverage={m.coverage_90:.0f}%)."
            )

    # Check skill vs climatology
    if np.isfinite(m.nse) and m.nse < 0:
        report.quality_flags.append(
            f"Model worse than climatology (NSE={m.nse:.2f})."
        )


def _generate_recommendations(report: ValidationReport) -> None:
    """Generate actionable recommendations."""
    m = report.overall_metrics

    if m is None:
        return

    # KGE component analysis
    if np.isfinite(m.kge_alpha) and np.isfinite(m.kge_beta):
        if m.kge_alpha < 0.7:
            report.recommendations.append(
                "Variability underestimated. Consider training with more extreme events."
            )
        elif m.kge_alpha > 1.3:
            report.recommendations.append(
                "Variability overestimated. Consider regularization or ensemble smoothing."
            )

        if m.kge_beta < 0.9:
            report.recommendations.append(
                "Systematic wet bias. Review ET estimation or drainage parameters."
            )
        elif m.kge_beta > 1.1:
            report.recommendations.append(
                "Systematic dry bias. Check infiltration or precipitation inputs."
            )

    # Horizon degradation
    if len(report.horizon_metrics) >= 2:
        rmse_0 = report.horizon_metrics[0].rmse
        rmse_last = report.horizon_metrics[-1].rmse

        if np.isfinite(rmse_0) and np.isfinite(rmse_last):
            degradation = (rmse_last - rmse_0) / rmse_0 * 100

            if degradation > 100:
                report.recommendations.append(
                    f"High forecast degradation ({degradation:.0f}%). "
                    "Consider shorter forecast horizons or ensemble methods."
                )

    # Site-specific issues
    if report.site_metrics:
        kges = [s.kge for s in report.site_metrics.values()
                if np.isfinite(s.kge)]
        if kges:
            min_kge = min(kges)
            max_kge = max(kges)

            if max_kge - min_kge > 0.4:
                report.recommendations.append(
                    "Large performance variation across sites. "
                    "Consider site-specific calibration or additional features."
                )
