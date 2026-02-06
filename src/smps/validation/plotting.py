"""
Visualization Utilities for SWPPS.

Provides plotting functions for:
- Matric potential time series
- Scatter plots (observed vs predicted)
- Forecast degradation analysis
- Irrigation decision visualization
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging

logger = logging.getLogger("swpps.validation.plotting")

# Handle matplotlib availability
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available - plotting disabled")


def is_plotting_available() -> bool:
    """Check if plotting is available."""
    return HAS_MATPLOTLIB


class SWPPSPlotter:
    """
    Visualization toolkit for SWPPS predictions.

    All plots use matric potential (kPa) as primary y-axis.
    """

    # Color scheme
    COLORS = {
        "observed": "#2C3E50",
        "predicted": "#3498DB",
        "physics": "#27AE60",
        "hybrid": "#9B59B6",
        "irrigation": "#E74C3C",
        "stress_zone": "#FADBD8",
        "optimal_zone": "#D5F5E3",
    }

    # Threshold lines
    THRESHOLDS = {
        "saturation": 0.0,
        "field_capacity": -33.0,
        "stress_onset": -100.0,
        "wilting_point": -1500.0,
    }

    def __init__(
        self,
        output_dir: Union[str, Path],
        dpi: int = 150,
        figsize: Tuple[int, int] = (12, 6),
        style: str = "seaborn-v0_8-whitegrid",
    ):
        """
        Initialize plotter.

        Args:
            output_dir: Directory to save plots
            dpi: Resolution for saved figures
            figsize: Default figure size
            style: Matplotlib style
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize

        if HAS_MATPLOTLIB:
            try:
                plt.style.use(style)
            except Exception:
                pass  # Use default style

    def plot_timeseries(
        self,
        dates: np.ndarray,
        observed: np.ndarray,
        predicted: Optional[np.ndarray] = None,
        lower_bound: Optional[np.ndarray] = None,
        upper_bound: Optional[np.ndarray] = None,
        irrigation_times: Optional[List[datetime]] = None,
        title: str = "Matric Potential Time Series",
        filename: str = "timeseries.png",
    ) -> Optional[Path]:
        """
        Plot matric potential time series.

        Args:
            dates: Datetime array
            observed: Observed psi (kPa)
            predicted: Predicted psi (kPa)
            lower_bound: Lower prediction bound
            upper_bound: Upper prediction bound
            irrigation_times: List of irrigation event times
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved file
        """
        if not HAS_MATPLOTLIB:
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot thresholds as horizontal zones
        ax.axhspan(self.THRESHOLDS["saturation"], 100,
                   alpha=0.1, color='blue', label='Saturated')
        ax.axhspan(self.THRESHOLDS["stress_onset"], self.THRESHOLDS["field_capacity"],
                   alpha=0.1, color='green', label='Optimal')
        ax.axhspan(self.THRESHOLDS["wilting_point"], self.THRESHOLDS["stress_onset"],
                   alpha=0.1, color='red', label='Stress Zone')

        # Plot threshold lines
        for name, value in self.THRESHOLDS.items():
            if value > -500:  # Only show relevant thresholds
                ax.axhline(value, color='gray', linestyle='--',
                           alpha=0.5, linewidth=0.8)
                ax.text(dates[-1], value, f' {name}',
                        va='center', fontsize=8, color='gray')

        # Plot observed
        ax.plot(dates, observed, 'o-', color=self.COLORS["observed"],
                markersize=3, linewidth=1, label='Observed', alpha=0.8)

        # Plot predicted with uncertainty
        if predicted is not None:
            ax.plot(dates, predicted, '-', color=self.COLORS["predicted"],
                    linewidth=2, label='Predicted')

            if lower_bound is not None and upper_bound is not None:
                ax.fill_between(dates, lower_bound, upper_bound,
                                alpha=0.2, color=self.COLORS["predicted"],
                                label='90% CI')

        # Mark irrigation events
        if irrigation_times:
            for irr_time in irrigation_times:
                ax.axvline(irr_time, color=self.COLORS["irrigation"],
                           linestyle=':', alpha=0.7, linewidth=1.5)
            ax.axvline(irr_time, color=self.COLORS["irrigation"],
                       linestyle=':', alpha=0.7, linewidth=1.5, label='Irrigation')

        # Format axes
        ax.set_xlabel('Date')
        ax.set_ylabel('Matric Potential (kPa)')
        ax.set_title(title)
        ax.legend(loc='lower left', fontsize=9)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)

        # Set y-axis to show negative values properly
        y_min = min(np.nanmin(observed), self.THRESHOLDS["stress_onset"])
        ax.set_ylim(y_min * 1.1, 20)
        ax.invert_yaxis()  # More negative at top for intuitive reading

        plt.tight_layout()

        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info("Saved timeseries plot: %s", save_path.name)
        return save_path

    def plot_scatter(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        title: str = "Observed vs Predicted",
        filename: str = "scatter.png",
        show_metrics: bool = True,
    ) -> Optional[Path]:
        """
        Create scatter plot of observed vs predicted.

        Args:
            observed: Observed values (kPa)
            predicted: Predicted values (kPa)
            title: Plot title
            filename: Output filename
            show_metrics: Whether to show performance metrics

        Returns:
            Path to saved file
        """
        if not HAS_MATPLOTLIB:
            return None

        # Filter valid
        valid = np.isfinite(observed) & np.isfinite(predicted)
        obs = observed[valid]
        pred = predicted[valid]

        if len(obs) < 2:
            logger.warning("Insufficient data for scatter plot")
            return None

        fig, ax = plt.subplots(figsize=(8, 8))

        # Scatter plot
        ax.scatter(pred, obs, alpha=0.4, s=20, c=self.COLORS["predicted"],
                   edgecolors='none')

        # 1:1 line
        lim_min = min(obs.min(), pred.min())
        lim_max = max(obs.max(), pred.max())
        margin = (lim_max - lim_min) * 0.05
        lim = [lim_min - margin, lim_max + margin]

        ax.plot(lim, lim, 'r--', lw=2, label='1:1 line')

        # Regression line
        if len(obs) > 10:
            z = np.polyfit(pred, obs, 1)
            p = np.poly1d(z)
            ax.plot(lim, p(lim), 'g-', lw=1.5,
                    label=f'Fit: y={z[0]:.2f}x + {z[1]:.1f}')

        # Compute metrics
        if show_metrics:
            errors = obs - pred
            rmse = np.sqrt(np.mean(errors ** 2))
            mae = np.mean(np.abs(errors))
            r = np.corrcoef(obs, pred)[0, 1]
            bias = np.mean(errors)

            metrics_text = (
                f'N = {len(obs)}\n'
                f'RMSE = {rmse:.1f} kPa\n'
                f'MAE = {mae:.1f} kPa\n'
                f'R² = {r**2:.3f}\n'
                f'Bias = {bias:+.1f} kPa'
            )

            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Predicted ψ (kPa)')
        ax.set_ylabel('Observed ψ (kPa)')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info("Saved scatter plot: %s", save_path.name)
        return save_path

    def plot_horizon_degradation(
        self,
        horizon_metrics: pd.DataFrame,
        filename: str = "horizon_degradation.png",
    ) -> Optional[Path]:
        """
        Plot forecast skill degradation with horizon.

        Args:
            horizon_metrics: DataFrame with horizon_hours, RMSE, KGE columns
            filename: Output filename

        Returns:
            Path to saved file
        """
        if not HAS_MATPLOTLIB:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        horizons = horizon_metrics["Horizon_h"].values
        rmse = horizon_metrics["RMSE_kPa"].values
        kge = horizon_metrics["KGE"].values

        # RMSE plot
        ax1 = axes[0]
        ax1.bar(range(len(horizons)), rmse,
                color=self.COLORS["predicted"], alpha=0.7)
        ax1.set_xticks(range(len(horizons)))
        ax1.set_xticklabels([f'{h}h' for h in horizons])
        ax1.set_xlabel('Forecast Horizon')
        ax1.set_ylabel('RMSE (kPa)')
        ax1.set_title('Forecast RMSE by Horizon')
        ax1.grid(True, alpha=0.3, axis='y')

        # KGE plot
        ax2 = axes[1]
        colors = ['green' if k > 0 else 'red' for k in kge]
        ax2.bar(range(len(horizons)), kge, color=colors, alpha=0.7)
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.set_xticks(range(len(horizons)))
        ax2.set_xticklabels([f'{h}h' for h in horizons])
        ax2.set_xlabel('Forecast Horizon')
        ax2.set_ylabel('KGE')
        ax2.set_title('Kling-Gupta Efficiency by Horizon')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info("Saved horizon degradation plot: %s", save_path.name)
        return save_path

    def plot_irrigation_schedule(
        self,
        dates: np.ndarray,
        psi_predicted: np.ndarray,
        irrigation_decisions: List[dict],
        filename: str = "irrigation_schedule.png",
    ) -> Optional[Path]:
        """
        Plot irrigation schedule with predicted soil status.

        Args:
            dates: Datetime array
            psi_predicted: Predicted matric potential
            irrigation_decisions: List of decision dicts with 'time', 'amount_mm'
            filename: Output filename

        Returns:
            Path to saved file
        """
        if not HAS_MATPLOTLIB:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                       gridspec_kw={'height_ratios': [2, 1]})

        # Top: Matric potential
        ax1.fill_between(dates, self.THRESHOLDS["stress_onset"], 0,
                         alpha=0.1, color='green', label='Optimal Zone')
        ax1.fill_between(dates, -200, self.THRESHOLDS["stress_onset"],
                         alpha=0.1, color='orange', label='Mild Stress')

        ax1.plot(dates, psi_predicted, '-', color=self.COLORS["predicted"],
                 linewidth=2, label='Predicted ψ')
        ax1.axhline(self.THRESHOLDS["field_capacity"], color='blue',
                    linestyle='--', alpha=0.5, label='Field Capacity')
        ax1.axhline(self.THRESHOLDS["stress_onset"], color='red',
                    linestyle='--', alpha=0.5, label='Stress Threshold')

        ax1.set_ylabel('Matric Potential (kPa)')
        ax1.set_ylim(-200, 20)
        ax1.invert_yaxis()
        ax1.legend(loc='lower left', fontsize=9)
        ax1.set_title('Soil Water Status and Irrigation Schedule')
        ax1.grid(True, alpha=0.3)

        # Bottom: Irrigation amounts
        irr_times = [d['time']
                     for d in irrigation_decisions if d.get('should_irrigate', False)]
        irr_amounts = [d['amount_mm']
                       for d in irrigation_decisions if d.get('should_irrigate', False)]

        if irr_times:
            ax2.bar(irr_times, irr_amounts, width=0.3, color=self.COLORS["irrigation"],
                    alpha=0.8, label='Irrigation')

        ax2.set_xlabel('Date')
        ax2.set_ylabel('Irrigation (mm)')
        ax2.set_ylim(0, max(irr_amounts) * 1.2 if irr_amounts else 20)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')

        # Format dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)

        plt.tight_layout()

        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info("Saved irrigation schedule plot: %s", save_path.name)
        return save_path

    def plot_model_comparison(
        self,
        results: Dict[str, pd.DataFrame],
        metric: str = "RMSE_kPa",
        filename: str = "model_comparison.png",
    ) -> Optional[Path]:
        """
        Compare multiple models.

        Args:
            results: Dict of model_name -> metrics DataFrame
            metric: Metric to compare
            filename: Output filename

        Returns:
            Path to saved file
        """
        if not HAS_MATPLOTLIB:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        models = list(results.keys())
        values = [results[m][metric].mean() for m in models]

        bars = ax.bar(range(len(models)), values, alpha=0.7)

        # Color by performance
        max_val = max(values)
        for bar, val in zip(bars, values):
            bar.set_color(plt.cm.RdYlGn(1 - val / max_val))

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(f'Model Comparison: {metric}')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info("Saved model comparison plot: %s", save_path.name)
        return save_path


def create_validation_plots(
    output_dir: Union[str, Path],
    observed: np.ndarray,
    predicted: np.ndarray,
    dates: Optional[np.ndarray] = None,
    prefix: str = "",
) -> List[Path]:
    """
    Convenience function to create standard validation plots.

    Args:
        output_dir: Directory to save plots
        observed: Observed values
        predicted: Predicted values
        dates: Optional datetime array
        prefix: Filename prefix

    Returns:
        List of saved file paths
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available")
        return []

    plotter = SWPPSPlotter(output_dir)
    saved = []

    # Scatter plot
    path = plotter.plot_scatter(
        observed, predicted,
        filename=f"{prefix}scatter.png" if prefix else "scatter.png"
    )
    if path:
        saved.append(path)

    # Time series if dates provided
    if dates is not None:
        path = plotter.plot_timeseries(
            dates, observed, predicted,
            filename=f"{prefix}timeseries.png" if prefix else "timeseries.png"
        )
        if path:
            saved.append(path)

    return saved
