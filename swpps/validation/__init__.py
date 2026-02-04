"""
SWPPS Validation Module.

Provides comprehensive model validation tools including:
- Performance metrics (RMSE, MAE, KGE, NSE, ubRMSE)
- Validation reports
- Visualization utilities
- Model evaluation and comparison
"""

from swpps.validation.metrics import (
    ValidationMetrics,
    compute_metrics,
    compute_kge,
    compute_nse,
    compute_ubrmse,
    compute_per_site_metrics,
    compute_horizon_metrics,
)
from swpps.validation.report import (
    ValidationReport,
    generate_report,
)
from swpps.validation.plotting import (
    SWPPSPlotter,
    is_plotting_available,
    create_validation_plots,
)
from swpps.validation.evaluation import (
    ModelEvaluator,
    SiteEvaluationResult,
    HorizonEvaluationResult,
    print_evaluation_summary,
)

__all__ = [
    # Metrics
    "ValidationMetrics",
    "compute_metrics",
    "compute_kge",
    "compute_nse",
    "compute_ubrmse",
    "compute_per_site_metrics",
    "compute_horizon_metrics",
    # Reports
    "ValidationReport",
    "generate_report",
    # Plotting
    "SWPPSPlotter",
    "is_plotting_available",
    "create_validation_plots",
    # Evaluation
    "ModelEvaluator",
    "SiteEvaluationResult",
    "HorizonEvaluationResult",
    "print_evaluation_summary",
]
