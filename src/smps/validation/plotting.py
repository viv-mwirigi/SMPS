"""
Visualization utilities for soil moisture prediction validation.

This module provides plotting functions for:
- Scatter plots (single-site and multi-site)
- Horizon comparison plots
- Learning curves for overfitting analysis
- Model comparison plots
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger("smps.validation.plotting")

# Try to import matplotlib, handle if not available
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning(
        "matplotlib not available - plotting functions will be disabled")


def check_matplotlib_available() -> bool:
    """Check if matplotlib is available."""
    return HAS_MATPLOTLIB


class ValidationPlotter:
    """
    Create validation plots for soil moisture prediction models.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        dpi: int = 150,
        figsize: Tuple[int, int] = (10, 6),
    ):
        """
        Initialize plotter.

        Args:
            output_dir: Directory to save plots
            dpi: Resolution for saved figures
            figsize: Default figure size
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize

    def plot_scatter_by_horizon(
        self,
        paired_df: pd.DataFrame,
        horizons: List[str],
        model_patterns: Dict[str, str],
        filename: str = "scatter_multimodel_{horizon}.png",
    ) -> List[Path]:
        """
        Create multi-model scatter plots for each horizon.

        Args:
            paired_df: DataFrame with 'obs', 'pred', 'model', 'horizon' columns
            horizons: List of horizon names to plot
            model_patterns: Dict mapping model_key to model pattern in data
            filename: Filename template (use {horizon} placeholder)

        Returns:
            List of saved file paths
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available for plotting")
            return []

        saved_files = []

        for horizon in horizons:
            horizon_data = paired_df[paired_df['horizon'] == horizon]

            if len(horizon_data) == 0:
                continue

            horizon_days = horizon_data['horizon_days'].iloc[0] if 'horizon_days' in horizon_data.columns else 0

            n_models = len(model_patterns)
            fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))

            if n_models == 1:
                axes = [axes]

            for idx, (model_key, model_pattern) in enumerate(model_patterns.items()):
                ax = axes[idx]
                model_data = horizon_data[horizon_data['model']
                                          == model_pattern]

                if len(model_data) == 0:
                    ax.text(0.5, 0.5, f'No data for {model_key}',
                            ha='center', va='center', transform=ax.transAxes)
                    continue

                obs = model_data['obs'].values
                pred = model_data['pred'].values

                # Scatter plot
                ax.scatter(pred, obs, alpha=0.3, s=10, c='steelblue')

                # 1:1 line
                lim = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
                ax.plot(lim, lim, 'r--', lw=2, label='1:1')

                # Fit line
                if len(obs) > 10:
                    z = np.polyfit(pred, obs, 1)
                    p = np.poly1d(z)
                    ax.plot(lim, p(lim), 'g-', lw=1.5,
                            label=f'Fit: y={z[0]:.2f}x+{z[1]:.3f}')

                # Metrics
                rmse = np.sqrt(np.mean((obs - pred) ** 2))
                r2 = np.corrcoef(obs, pred)[0, 1] ** 2 if len(obs) > 2 else 0

                ax.set_xlabel('Predicted SM (m³/m³)')
                ax.set_ylabel('Observed SM (m³/m³)')
                ax.set_title(f'{model_key.upper()} - {horizon} ({horizon_days}d)\n'
                             f'RMSE={rmse:.4f}, R²={r2:.3f}, n={len(obs)}')
                ax.legend(loc='upper left')
                ax.set_xlim(lim)
                ax.set_ylim(lim)
                ax.set_aspect('equal')

            plt.tight_layout()

            save_path = self.output_dir / filename.format(horizon=horizon)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            saved_files.append(save_path)
            logger.info(f"Saved scatter plot: {save_path.name}")

        return saved_files

    def plot_scatter_by_depth(
        self,
        paired_df: pd.DataFrame,
        horizon: str = '0h',
        model: str = 'physics_0h',
        max_depths: int = 4,
        filename: str = "scatter_by_depth.png",
    ) -> Optional[Path]:
        """
        Create scatter plots by soil depth.

        Args:
            paired_df: DataFrame with paired observations and predictions
            horizon: Horizon to filter by
            model: Model to filter by
            max_depths: Maximum number of depths to plot
            filename: Output filename

        Returns:
            Path to saved file, or None if plotting failed
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available for plotting")
            return None

        depth_data = paired_df[
            (paired_df['horizon'] == horizon) &
            (paired_df['model'] == model)
        ]

        depths = sorted(depth_data['depth_cm'].unique())[:max_depths]
        n_depths = len(depths)

        if n_depths == 0:
            logger.warning("No data for depth scatter plot")
            return None

        fig, axes = plt.subplots(1, n_depths, figsize=(5 * n_depths, 5))
        if n_depths == 1:
            axes = [axes]

        for idx, depth in enumerate(depths):
            ax = axes[idx]
            depth_subset = depth_data[depth_data['depth_cm'] == depth]

            if len(depth_subset) == 0:
                continue

            obs = depth_subset['obs'].values
            pred = depth_subset['pred'].values

            ax.scatter(pred, obs, alpha=0.4, s=15, c='steelblue')

            lim = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
            ax.plot(lim, lim, 'r--', lw=2)

            rmse = np.sqrt(np.mean((obs - pred) ** 2))
            r2 = np.corrcoef(obs, pred)[0, 1] ** 2 if len(obs) > 2 else 0

            ax.set_xlabel('Predicted SM (m³/m³)')
            ax.set_ylabel('Observed SM (m³/m³)')
            ax.set_title(f'{int(depth)}cm Depth\nRMSE={rmse:.4f}, R²={r2:.3f}')
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect('equal')

        plt.tight_layout()

        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved scatter plot: {save_path.name}")
        return save_path

    def plot_horizon_comparison(
        self,
        paired_df: pd.DataFrame,
        horizons: List[str],
        models: List[str],
        model_names: List[str],
        filename: str = "horizon_comparison.png",
    ) -> Optional[Path]:
        """
        Create bar chart comparing RMSE across horizons for each model.

        Args:
            paired_df: DataFrame with paired data
            horizons: List of horizons to compare
            models: List of model patterns
            model_names: Display names for models
            filename: Output filename

        Returns:
            Path to saved file
        """
        if not HAS_MATPLOTLIB:
            return None

        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))

        if n_models == 1:
            axes = [axes]

        for idx, (model, name) in enumerate(zip(models, model_names)):
            ax = axes[idx]

            horizon_stats = []
            horizon_labels = []

            for horizon in horizons:
                model_pattern = f'{model}_{horizon}'
                model_data = paired_df[paired_df['model'] == model_pattern]

                if len(model_data) > 10:
                    rmse = np.sqrt(np.mean(
                        (model_data['obs'] - model_data['pred']) ** 2
                    ))
                    horizon_stats.append(rmse)
                    horizon_labels.append(horizon)

            if horizon_stats:
                ax.bar(range(len(horizon_stats)), horizon_stats,
                       color='steelblue', alpha=0.7)
                ax.set_xticks(range(len(horizon_labels)))
                ax.set_xticklabels(horizon_labels)
                ax.set_ylabel('RMSE (m³/m³)')
                ax.set_title(f'{name} Forecast Skill by Horizon')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved horizon comparison: {save_path.name}")
        return save_path

    def plot_single_site_comparison(
        self,
        paired_df: pd.DataFrame,
        horizon: str = '0h',
        max_stations: int = 5,
        filename: str = "scatter_single_sites.png",
    ) -> Optional[Path]:
        """
        Create scatter plots comparing physics vs hybrid for individual stations.

        Args:
            paired_df: DataFrame with paired data
            horizon: Horizon to plot
            max_stations: Maximum number of stations to include
            filename: Output filename

        Returns:
            Path to saved file
        """
        if not HAS_MATPLOTLIB:
            return None

        station_data = paired_df[paired_df['horizon'] == horizon]
        stations = station_data['station_id'].unique()[:max_stations]

        if len(stations) == 0:
            return None

        fig, axes = plt.subplots(
            2, len(stations), figsize=(4 * len(stations), 8))

        if len(stations) == 1:
            axes = axes.reshape(2, 1)

        for idx, station in enumerate(stations):
            station_subset = station_data[station_data['station_id'] == station]

            # Physics row
            ax = axes[0, idx]
            phys_data = station_subset[
                station_subset['model'].str.contains('physics', case=False)
            ]

            if len(phys_data) > 0:
                obs = phys_data['obs'].values
                pred = phys_data['pred'].values
                ax.scatter(pred, obs, alpha=0.5, s=20)

                lim = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
                ax.plot(lim, lim, 'r--', lw=2)

                rmse = np.sqrt(np.mean((obs - pred) ** 2))
                ax.set_title(f'{str(station)[:20]}\nPhysics RMSE={rmse:.4f}')

            ax.set_xlabel('Predicted')
            ax.set_ylabel('Observed')

            # Hybrid row
            ax = axes[1, idx]
            hybrid_data = station_subset[
                station_subset['model'].str.contains('hybrid', case=False)
            ]

            if len(hybrid_data) > 0:
                obs = hybrid_data['obs'].values
                pred = hybrid_data['pred'].values
                ax.scatter(pred, obs, alpha=0.5, s=20, c='green')

                lim = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
                ax.plot(lim, lim, 'r--', lw=2)

                rmse = np.sqrt(np.mean((obs - pred) ** 2))
                ax.set_title(f'Hybrid RMSE={rmse:.4f}')

            ax.set_xlabel('Predicted')
            ax.set_ylabel('Observed')

        plt.tight_layout()

        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved single site comparison: {save_path.name}")
        return save_path

    def plot_learning_curves(
        self,
        training_history: Dict[str, Dict],
        filename: str = "learning_curves.png",
    ) -> Optional[Path]:
        """
        Plot learning curves for overfitting analysis.

        Args:
            training_history: Dict with depth -> {train_rmse, val_rmse, test_rmse, best_iteration}
            filename: Output filename

        Returns:
            Path to saved file
        """
        if not HAS_MATPLOTLIB:
            return None

        if not training_history:
            return None

        depths = list(training_history.keys())
        n_depths = len(depths)

        if n_depths == 0:
            return None

        fig, axes = plt.subplots(n_depths, 1, figsize=(10, 4 * n_depths))

        if n_depths == 1:
            axes = [axes]

        for i, depth in enumerate(depths):
            ax = axes[i]
            history = training_history[depth]

            # Check if we have iteration-level history
            if 'train_rmse' in history and isinstance(history['train_rmse'], (list, np.ndarray)):
                train_rmse = history['train_rmse']
                val_rmse = history['val_rmse']
                test_rmse = history.get('test_rmse', val_rmse)

                iterations = range(1, len(train_rmse) + 1)

                ax.plot(iterations, train_rmse, 'b-',
                        label='Train RMSE', linewidth=2)
                ax.plot(iterations, val_rmse, 'g-',
                        label='Validation RMSE', linewidth=2)
                ax.plot(iterations, test_rmse, 'r-',
                        label='Test RMSE', linewidth=2)

                # Mark best iteration
                best_iter = history.get('best_iteration', len(train_rmse))
                if best_iter < len(train_rmse):
                    ax.axvline(x=best_iter, color='orange', linestyle='--',
                               alpha=0.7, label=f'Best iteration ({best_iter})')
            else:
                # Just show summary statistics
                ax.text(0.5, 0.5,
                        f"Best iteration: {history.get('best_iteration', 'N/A')}\n"
                        f"Features: {history.get('n_features', 'N/A')}\n"
                        f"Overfitting: {history.get('overfitting_detected', 'N/A')}",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=12)

            ax.set_xlabel('Boosting Iteration')
            ax.set_ylabel('RMSE')
            ax.set_title(f'Learning Curve - Depth {depth}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved learning curves: {save_path.name}")
        return save_path


def create_model_comparison_table(
    physics_results: List[Dict],
    ml_results: List[Dict],
    hybrid_results: List[Dict],
) -> pd.DataFrame:
    """
    Create a comparison table summarizing model performance.

    Args:
        physics_results: List of physics model result dicts
        ml_results: List of ML model result dicts
        hybrid_results: List of hybrid model result dicts

    Returns:
        DataFrame with comparison metrics
    """
    results = []

    if physics_results:
        physics_df = pd.DataFrame(physics_results)
        results.append({
            'Model': 'Physics',
            'RMSE': physics_df['RMSE'].mean() if 'RMSE' in physics_df else np.nan,
            'MAE': physics_df['MAE'].mean() if 'MAE' in physics_df else np.nan,
            'KGE': physics_df['KGE'].mean() if 'KGE' in physics_df else np.nan,
            'R²': physics_df['R²'].mean() if 'R²' in physics_df else np.nan,
            'N_stations': physics_df['station_id'].nunique() if 'station_id' in physics_df else 0,
        })

    if ml_results:
        ml_df = pd.DataFrame(ml_results)
        rmse_col = 'test_rmse' if 'test_rmse' in ml_df else 'RMSE'
        results.append({
            'Model': 'ML (LightGBM)',
            'RMSE': ml_df[rmse_col].mean() if rmse_col in ml_df else np.nan,
            'MAE': ml_df.get('test_mae', ml_df.get('MAE', pd.Series())).mean(),
            'KGE': ml_df.get('test_kge', ml_df.get('KGE', pd.Series())).mean(),
            'R²': ml_df.get('test_r2', ml_df.get('R²', pd.Series())).mean(),
            'N_stations': ml_df['station_id'].nunique() if 'station_id' in ml_df else 0,
        })

    if hybrid_results:
        hybrid_df = pd.DataFrame(hybrid_results)
        rmse_col = 'test_rmse' if 'test_rmse' in hybrid_df else 'RMSE'
        results.append({
            'Model': 'Hybrid (Physics+ML)',
            'RMSE': hybrid_df[rmse_col].mean() if rmse_col in hybrid_df else np.nan,
            'MAE': hybrid_df.get('test_mae', hybrid_df.get('MAE', pd.Series())).mean(),
            'KGE': hybrid_df.get('test_kge', hybrid_df.get('KGE', pd.Series())).mean(),
            'R²': hybrid_df.get('test_r2', hybrid_df.get('R²', pd.Series())).mean(),
            'N_stations': hybrid_df['station_id'].nunique() if 'station_id' in hybrid_df else 0,
        })

    return pd.DataFrame(results)


def print_validation_summary(
    physics_results: List[Dict],
    ml_results: List[Dict],
    hybrid_results: List[Dict],
    horizons: Optional[List[str]] = None,
):
    """
    Print a formatted validation summary to console.

    Args:
        physics_results: Physics model results
        ml_results: ML model results
        hybrid_results: Hybrid model results
        horizons: Optional list of horizons to summarize
    """
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 80)

    if physics_results:
        physics_df = pd.DataFrame(physics_results)
        print("\n📊 PHYSICS MODEL PERFORMANCE:")
        print("-" * 50)
        print(
            f"  Stations: {physics_df['station_id'].nunique() if 'station_id' in physics_df else 'N/A'}")

        if 'RMSE' in physics_df:
            print(
                f"  RMSE: {physics_df['RMSE'].mean():.4f} ± {physics_df['RMSE'].std():.4f}")
        if 'KGE' in physics_df:
            print(
                f"  KGE:  {physics_df['KGE'].mean():.3f} ± {physics_df['KGE'].std():.3f}")

    if ml_results:
        ml_df = pd.DataFrame(ml_results)
        rmse_col = 'test_rmse' if 'test_rmse' in ml_df else 'RMSE'
        print("\n🤖 ML MODEL PERFORMANCE:")
        print("-" * 50)
        print(
            f"  RMSE: {ml_df[rmse_col].mean():.4f} ± {ml_df[rmse_col].std():.4f}")

    if hybrid_results:
        hybrid_df = pd.DataFrame(hybrid_results)
        rmse_col = 'test_rmse' if 'test_rmse' in hybrid_df else 'RMSE'
        print("\n🔬 HYBRID MODEL PERFORMANCE:")
        print("-" * 50)
        print(
            f"  RMSE: {hybrid_df[rmse_col].mean():.4f} ± {hybrid_df[rmse_col].std():.4f}")

    # Comparison
    if physics_results and ml_results and hybrid_results:
        physics_rmse = pd.DataFrame(physics_results)['RMSE'].mean()
        ml_rmse = pd.DataFrame(ml_results)[rmse_col].mean()
        hybrid_rmse = pd.DataFrame(hybrid_results)[rmse_col].mean()

        print("\n📈 MODEL COMPARISON:")
        print("-" * 50)
        print(f"  {'Model':<20} {'RMSE':<12} {'Improvement':<15}")
        print(f"  {'-'*47}")
        print(f"  {'Physics':<20} {physics_rmse:.4f}       {'(baseline)':<15}")
        print(
            f"  {'ML':<20} {ml_rmse:.4f}       {(1-ml_rmse/physics_rmse)*100:+.1f}%")
        print(
            f"  {'Hybrid':<20} {hybrid_rmse:.4f}       {(1-hybrid_rmse/physics_rmse)*100:+.1f}%")

    print("\n" + "=" * 80)
