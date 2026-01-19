"""
SHAP-based Model Explainability for Soil Moisture Prediction.

This module provides SHAP (SHapley Additive exPlanations) analysis for
understanding feature contributions to soil moisture predictions.

SHAP Benefits:
- Consistent and locally accurate attributions
- Model-agnostic (works with any ML model)
- Global and local interpretations
- Interaction effect detection

Research References:
- Lundberg & Lee (2017): A unified approach to interpreting model predictions
- Lundberg et al. (2020): From local explanations to global understanding

Usage:
------
>>> from smps.ml.explainer import SHAPExplainer
>>>
>>> explainer = SHAPExplainer(model)
>>> importance = explainer.get_feature_importance(X)
>>> explainer.plot_summary(X)
>>> explainer.explain_prediction(X.iloc[0])
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("smps.ml.explainer")


@dataclass
class FeatureImportance:
    """Container for feature importance information."""

    # Feature name to mean absolute SHAP value
    global_importance: Dict[str, float] = field(default_factory=dict)

    # Feature importance ranking
    ranking: List[Tuple[str, float]] = field(default_factory=list)

    # Feature interactions (feature_pair -> interaction strength)
    interactions: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Category-level importance
    category_importance: Dict[str, float] = field(default_factory=dict)

    # Statistics
    n_samples: int = 0
    n_features: int = 0

    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get top N features by importance."""
        return self.ranking[:n]

    def get_features_by_category(self, category: str) -> List[Tuple[str, float]]:
        """Get features in a category ranked by importance."""
        # Requires feature -> category mapping
        return [(f, imp) for f, imp in self.ranking if category.lower() in f.lower()]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        return pd.DataFrame(
            self.ranking,
            columns=['feature', 'importance']
        )


class SHAPExplainer:
    """
    SHAP-based model explainability.

    Provides:
    - Global feature importance (mean |SHAP|)
    - Local explanations (per-prediction attributions)
    - Feature interaction analysis
    - Visualization support

    Optimized for tree-based models (LightGBM, XGBoost) using TreeSHAP.
    """

    def __init__(
        self,
        model: Any = None,
        feature_names: Optional[List[str]] = None,
        background_size: int = 100,
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model (LightGBM, XGBoost, or sklearn-compatible)
            feature_names: List of feature names
            background_size: Number of samples for background dataset
        """
        self.model = model
        self.feature_names = feature_names
        self.background_size = background_size

        self._explainer = None
        self._shap_values = None
        self._expected_value = None

        # Feature grouping for category importance
        self._feature_categories: Dict[str, str] = {}

    def set_feature_categories(self, categories: Dict[str, str]):
        """
        Set feature to category mapping.

        Args:
            categories: Dict mapping feature name to category
        """
        self._feature_categories = categories

    def _create_explainer(self, X: Union[pd.DataFrame, np.ndarray]):
        """Create SHAP explainer based on model type."""
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP not installed. Run: pip install shap"
            )

        # Determine model type and create appropriate explainer
        model_type = type(self.model).__name__.lower()

        if hasattr(self.model, 'booster_'):
            # LightGBM Booster
            self._explainer = shap.TreeExplainer(self.model)
        elif hasattr(self.model, 'get_booster'):
            # XGBoost
            self._explainer = shap.TreeExplainer(self.model)
        elif 'lightgbm' in model_type or 'booster' in model_type:
            # LightGBM native
            self._explainer = shap.TreeExplainer(self.model)
        elif 'xgb' in model_type:
            # XGBoost native
            self._explainer = shap.TreeExplainer(self.model)
        else:
            # Generic explainer with background
            if isinstance(X, pd.DataFrame):
                background = X.sample(min(self.background_size, len(X)))
            else:
                idx = np.random.choice(len(X), min(
                    self.background_size, len(X)), replace=False)
                background = X[idx]

            self._explainer = shap.KernelExplainer(
                self.model.predict, background
            )

        logger.info("Created SHAP explainer: %s",
                    type(self._explainer).__name__)

    def compute_shap_values(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        check_additivity: bool = False,
    ) -> np.ndarray:
        """
        Compute SHAP values for dataset.

        Args:
            X: Feature matrix
            check_additivity: Whether to check SHAP additivity (slower)

        Returns:
            Array of SHAP values (n_samples x n_features)
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP not installed. Run: pip install shap")

        if self._explainer is None:
            self._create_explainer(X)

        # Get feature names
        if self.feature_names is None and isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()

        # Compute SHAP values
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X

        logger.info("Computing SHAP values for %d samples...", len(X_values))

        # Use optimized computation for tree models
        if hasattr(self._explainer, 'shap_values'):
            self._shap_values = self._explainer.shap_values(
                X_values,
                check_additivity=check_additivity
            )
        else:
            shap_explanation = self._explainer(X_values)
            self._shap_values = shap_explanation.values

        # Handle multi-output models
        if isinstance(self._shap_values, list):
            self._shap_values = self._shap_values[0]

        # Store expected value
        if hasattr(self._explainer, 'expected_value'):
            self._expected_value = self._explainer.expected_value
            if isinstance(self._expected_value, np.ndarray):
                self._expected_value = self._expected_value[0]

        logger.info("Computed SHAP values shape: %s",
                    str(self._shap_values.shape))

        return self._shap_values

    def get_feature_importance(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        method: str = "mean_abs",
    ) -> FeatureImportance:
        """
        Compute global feature importance from SHAP values.

        Args:
            X: Feature matrix (required if SHAP values not yet computed)
            method: Aggregation method ('mean_abs', 'max_abs', 'std')

        Returns:
            FeatureImportance object
        """
        if self._shap_values is None:
            if X is None:
                raise ValueError("X required when SHAP values not computed")
            self.compute_shap_values(X)

        # Compute importance based on method
        if method == "mean_abs":
            importance = np.abs(self._shap_values).mean(axis=0)
        elif method == "max_abs":
            importance = np.abs(self._shap_values).max(axis=0)
        elif method == "std":
            importance = self._shap_values.std(axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Normalize to sum to 1
        total = importance.sum()
        if total > 0:
            importance = importance / total

        # Create importance dict
        feature_names = self.feature_names or [
            f"f{i}" for i in range(len(importance))]
        global_importance = dict(zip(feature_names, importance))

        # Sort by importance
        ranking = sorted(global_importance.items(),
                         key=lambda x: x[1], reverse=True)

        # Category importance
        category_importance = {}
        for feature, imp in global_importance.items():
            cat = self._feature_categories.get(feature, "other")
            category_importance[cat] = category_importance.get(cat, 0) + imp

        return FeatureImportance(
            global_importance=global_importance,
            ranking=ranking,
            category_importance=category_importance,
            n_samples=self._shap_values.shape[0],
            n_features=self._shap_values.shape[1],
        )

    def explain_prediction(
        self,
        x: Union[pd.Series, np.ndarray],
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.

        Args:
            x: Single sample features
            top_n: Number of top contributing features to return

        Returns:
            Dict with prediction explanation
        """
        if self._explainer is None:
            raise RuntimeError(
                "Explainer not initialized. Call compute_shap_values first.")

        # Reshape for single sample
        if isinstance(x, pd.Series):
            x_values = x.values.reshape(1, -1)
        else:
            x_values = x.reshape(1, -1)

        # Compute SHAP for single sample
        shap_values = self._explainer.shap_values(x_values)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = shap_values.flatten()

        # Feature names
        feature_names = self.feature_names or [
            f"f{i}" for i in range(len(shap_values))]

        # Get feature values
        feature_values = x_values.flatten()

        # Create contribution list
        contributions = [
            {
                "feature": name,
                "value": float(feature_values[i]),
                "shap": float(shap_values[i]),
                "direction": "positive" if shap_values[i] > 0 else "negative",
            }
            for i, name in enumerate(feature_names)
        ]

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x["shap"]), reverse=True)

        # Expected value (baseline)
        base_value = self._expected_value if self._expected_value is not None else 0

        return {
            "base_value": float(base_value),
            "prediction": float(base_value + shap_values.sum()),
            "top_contributions": contributions[:top_n],
            "all_contributions": contributions,
        }

    def compute_interactions(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        top_n_features: int = 10,
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute SHAP interaction values.

        Note: This is computationally expensive for large datasets.

        Args:
            X: Feature matrix
            top_n_features: Consider only top N features for interactions

        Returns:
            Dict mapping feature pairs to interaction strength
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP not installed")

        if self._explainer is None:
            self._create_explainer(X)

        # Get top features
        importance = self.get_feature_importance(X)
        top_features = [
            f for f, _ in importance.get_top_features(top_n_features)]

        # Get indices of top features
        feature_names = self.feature_names or [
            f"f{i}" for i in range(X.shape[1])]
        top_indices = [feature_names.index(
            f) for f in top_features if f in feature_names]

        # Subset data for efficiency
        if isinstance(X, pd.DataFrame):
            X_subset = X[top_features].values
        else:
            X_subset = X[:, top_indices]

        # Compute interaction values (if supported)
        if hasattr(self._explainer, 'shap_interaction_values'):
            logger.info("Computing SHAP interaction values...")
            interaction_values = self._explainer.shap_interaction_values(
                X_subset)

            # Handle multi-output
            if isinstance(interaction_values, list):
                interaction_values = interaction_values[0]

            # Aggregate to get interaction strength
            mean_interactions = np.abs(interaction_values).mean(axis=0)

            interactions = {}
            for i, f1 in enumerate(top_features):
                for j, f2 in enumerate(top_features):
                    if i < j:  # Only upper triangle
                        interactions[(f1, f2)] = float(mean_interactions[i, j])

            return interactions
        else:
            logger.warning(
                "Interaction values not supported for this model type")
            return {}

    def plot_summary(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        max_display: int = 20,
        plot_type: str = "bar",
        show: bool = True,
        save_path: Optional[Path] = None,
    ):
        """
        Create SHAP summary plot.

        Args:
            X: Feature matrix
            max_display: Maximum features to display
            plot_type: 'bar', 'dot', or 'violin'
            show: Whether to display plot
            save_path: Optional path to save figure
        """
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("SHAP and matplotlib required")

        if self._shap_values is None:
            self.compute_shap_values(X)

        feature_names = self.feature_names or [
            f"f{i}" for i in range(self._shap_values.shape[1])]

        plt.figure(figsize=(10, 8))

        if plot_type == "bar":
            shap.summary_plot(
                self._shap_values,
                X if isinstance(X, pd.DataFrame) else X,
                feature_names=feature_names,
                plot_type="bar",
                max_display=max_display,
                show=False,
            )
        elif plot_type == "dot":
            shap.summary_plot(
                self._shap_values,
                X if isinstance(X, pd.DataFrame) else X,
                feature_names=feature_names,
                max_display=max_display,
                show=False,
            )
        elif plot_type == "violin":
            shap.summary_plot(
                self._shap_values,
                X if isinstance(X, pd.DataFrame) else X,
                feature_names=feature_names,
                plot_type="violin",
                max_display=max_display,
                show=False,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info("Saved SHAP summary plot to %s", save_path)

        if show:
            plt.show()
        else:
            plt.close()

    def plot_waterfall(
        self,
        x: Union[pd.Series, np.ndarray],
        max_display: int = 15,
        show: bool = True,
        save_path: Optional[Path] = None,
    ):
        """
        Create waterfall plot for single prediction.

        Args:
            x: Single sample features
            max_display: Maximum features to display
            show: Whether to display plot
            save_path: Optional path to save figure
        """
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("SHAP and matplotlib required")

        if self._explainer is None:
            raise RuntimeError("Explainer not initialized")

        # Reshape for single sample
        if isinstance(x, pd.Series):
            x_values = x.values.reshape(1, -1)
            feature_names = x.index.tolist()
        else:
            x_values = x.reshape(1, -1)
            feature_names = self.feature_names or [
                f"f{i}" for i in range(len(x))]

        # Create explanation
        explanation = self._explainer(x_values)

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=explanation.values[0],
                base_values=explanation.base_values[0] if hasattr(
                    explanation.base_values, '__len__') else explanation.base_values,
                data=x_values[0],
                feature_names=feature_names,
            ),
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_dependence(
        self,
        feature: str,
        X: Union[pd.DataFrame, np.ndarray],
        interaction_feature: Optional[str] = None,
        show: bool = True,
        save_path: Optional[Path] = None,
    ):
        """
        Create SHAP dependence plot.

        Args:
            feature: Feature to plot
            X: Feature matrix
            interaction_feature: Optional feature for interaction coloring
            show: Whether to display plot
            save_path: Optional path to save figure
        """
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("SHAP and matplotlib required")

        if self._shap_values is None:
            self.compute_shap_values(X)

        feature_names = self.feature_names or (
            X.columns.tolist() if isinstance(X, pd.DataFrame)
            else [f"f{i}" for i in range(X.shape[1])]
        )

        feature_idx = feature_names.index(
            feature) if feature in feature_names else 0

        plt.figure(figsize=(8, 6))

        interaction_idx = None
        if interaction_feature and interaction_feature in feature_names:
            interaction_idx = feature_names.index(interaction_feature)

        shap.dependence_plot(
            feature_idx,
            self._shap_values,
            X if isinstance(X, pd.DataFrame) else X,
            feature_names=feature_names,
            interaction_index=interaction_idx,
            show=False,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def get_category_importance(
        self,
        category_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """
        Get importance aggregated by feature category.

        Args:
            category_mapping: Dict mapping feature name to category

        Returns:
            Dict of category to total importance
        """
        if category_mapping:
            self._feature_categories = category_mapping

        importance = self.get_feature_importance()
        return importance.category_importance

    def save_explanations(
        self,
        path: Path,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ):
        """
        Save SHAP values and importance to disk.

        Args:
            path: Directory to save files
            X: Optional feature matrix
        """
        import json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save SHAP values
        if self._shap_values is not None:
            np.save(path / "shap_values.npy", self._shap_values)

        # Save feature importance
        if X is not None or self._shap_values is not None:
            importance = self.get_feature_importance(X)

            with open(path / "feature_importance.json", 'w', encoding='utf-8') as f:
                json.dump({
                    "global_importance": importance.global_importance,
                    "ranking": importance.ranking,
                    "category_importance": importance.category_importance,
                    "n_samples": importance.n_samples,
                    "n_features": importance.n_features,
                }, f, indent=2)

        # Save feature names
        if self.feature_names:
            with open(path / "feature_names.json", 'w', encoding='utf-8') as f:
                json.dump(self.feature_names, f)

        logger.info("Saved SHAP explanations to %s", path)
