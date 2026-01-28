"""
Feature Store for Soil Moisture ML Pipeline.

Provides feature versioning, caching, and metadata management for
reproducible ML experiments.

Features:
- Feature group management
- Versioned feature storage
- Feature metadata tracking
- Efficient caching with TTL
- Feature lineage tracking
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import numpy as np

logger = logging.getLogger("smps.ml.feature_store")


@dataclass
class FeatureMetadata:
    """Metadata for a feature."""
    name: str
    description: str
    dtype: str
    source: str  # 'raw', 'physics', 'engineered', 'remote_sensing', 'static'
    category: str  # 'weather', 'soil', 'vegetation', 'temporal', 'interaction'

    # Statistics
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    null_fraction: float = 0.0

    # Engineering info
    depends_on: List[str] = field(default_factory=list)
    lag_days: Optional[int] = None
    window_days: Optional[int] = None
    # 'log', 'sqrt', 'normalize', 'standardize'
    transform: Optional[str] = None

    # Importance
    importance_score: Optional[float] = None
    correlation_with_target: Optional[float] = None

    # Versioning
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "dtype": self.dtype,
            "source": self.source,
            "category": self.category,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_value": self.mean_value,
            "std_value": self.std_value,
            "null_fraction": self.null_fraction,
            "depends_on": self.depends_on,
            "lag_days": self.lag_days,
            "window_days": self.window_days,
            "transform": self.transform,
            "importance_score": self.importance_score,
            "correlation_with_target": self.correlation_with_target,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class FeatureGroup:
    """A logical grouping of related features."""
    name: str
    description: str
    features: List[str]
    category: str

    # Group metadata
    source: str = "engineered"
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "features": self.features,
            "category": self.category,
            "source": self.source,
            "version": self.version,
        }


# Pre-defined feature groups for soil moisture prediction
FEATURE_GROUPS = {
    "weather_current": FeatureGroup(
        name="weather_current",
        description="Current day weather variables",
        features=[
            "precipitation_mm", "et0_mm",
            "temperature_mean_c", "temperature_min_c", "temperature_max_c",
            "solar_radiation_mj_m2", "relative_humidity_mean",
            "wind_speed_mean_m_s", "vapor_pressure_deficit_kpa",
        ],
        category="weather",
        source="raw",
    ),
    "weather_lag": FeatureGroup(
        name="weather_lag",
        description="Lagged weather features",
        features=[
            "precipitation_mm_lag1", "precipitation_mm_lag3", "precipitation_mm_lag7",
            "et0_mm_lag1", "et0_mm_lag3", "et0_mm_lag7",
            "temperature_mean_c_lag1", "temperature_mean_c_lag7",
        ],
        category="weather",
        source="engineered",
    ),
    "weather_rolling": FeatureGroup(
        name="weather_rolling",
        description="Rolling aggregations of weather",
        features=[
            "precip_sum_3d", "precip_sum_7d", "precip_sum_14d", "precip_sum_30d",
            "precip_max_7d", "precip_days_7d", "precip_days_14d",
            "et0_mean_7d", "et0_sum_7d", "et0_sum_14d", "et0_sum_30d",
            "temp_mean_7d", "temp_std_7d",
        ],
        category="weather",
        source="engineered",
    ),
    "water_balance": FeatureGroup(
        name="water_balance",
        description="Cumulative water balance features",
        features=[
            "daily_water_balance",
            "water_balance_3d", "water_balance_7d", "water_balance_14d", "water_balance_30d",
            "api_30d", "aridity_index_7d", "aridity_index_30d", "moisture_index",
        ],
        category="hydrology",
        source="engineered",
    ),
    "physics_states": FeatureGroup(
        name="physics_states",
        description="Physics model soil moisture states",
        features=[
            "physics_theta_surface", "physics_theta_root", "physics_theta_deep",
            "physics_theta_surface_lag1", "physics_theta_surface_lag7",
            "physics_theta_root_lag1", "physics_theta_root_lag7",
        ],
        category="physics",
        source="physics",
    ),
    "physics_derived": FeatureGroup(
        name="physics_derived",
        description="Features derived from physics model",
        features=[
            "relative_saturation", "soil_moisture_deficit",
            "available_water_fraction", "water_stress_index",
            "physics_theta_surface_trend_14d", "physics_theta_root_trend_14d",
        ],
        category="physics",
        source="engineered",
    ),
    "physics_residuals": FeatureGroup(
        name="physics_residuals",
        description="Physics model residuals (observation - physics)",
        features=[
            "physics_residual_surface", "physics_residual_root", "physics_residual_deep",
        ],
        category="physics",
        source="engineered",
    ),
    "soil_properties": FeatureGroup(
        name="soil_properties",
        description="Static soil texture and hydraulic properties",
        features=[
            "sand_percent", "silt_percent", "clay_percent",
            "porosity", "field_capacity", "wilting_point",
            "sat_hydraulic_cond", "available_water_capacity", "plant_available_water",
        ],
        category="soil",
        source="raw",
    ),
    "site_attributes": FeatureGroup(
        name="site_attributes",
        description="Static site location and topography",
        features=[
            "latitude", "longitude", "elevation_m",
            "slope_degrees", "twi",
        ],
        category="site",
        source="raw",
    ),
    "vegetation_indices": FeatureGroup(
        name="vegetation_indices",
        description="Remote sensing vegetation features",
        features=[
            "ndvi", "evi", "lai",
            "ndvi_lag7", "ndvi_lag14", "ndvi_lag30",
            "ndvi_mean_14d", "ndvi_mean_30d", "ndvi_std_14d",
            "ndvi_trend_14d", "ndvi_anomaly", "vegetation_fraction",
        ],
        category="vegetation",
        source="remote_sensing",
    ),
    "sar_features": FeatureGroup(
        name="sar_features",
        description="SAR backscatter features",
        features=[
            "sar_vv_db", "sar_vh_db",
        ],
        category="remote_sensing",
        source="raw",
    ),
    "temporal": FeatureGroup(
        name="temporal",
        description="Temporal and seasonal features",
        features=[
            "day_of_year", "doy_sin", "doy_cos",
            "month", "month_sin", "month_cos",
            "week_of_year", "days_since_start",
            "season_fall", "season_spring", "season_summer", "season_winter",
        ],
        category="temporal",
        source="engineered",
    ),
    "interactions": FeatureGroup(
        name="interactions",
        description="Feature interaction terms",
        features=[
            "precip_infiltration_potential", "et_stress_ratio",
            "temp_moisture_product", "ndvi_moisture_product",
            "clay_moisture_product",
        ],
        category="interaction",
        source="engineered",
    ),
    "observations": FeatureGroup(
        name="observations",
        description="Historical soil moisture observations",
        features=[
            "obs_vwc_surface", "obs_vwc_root", "obs_vwc_deep",
            "obs_vwc_surface_lag1", "obs_vwc_surface_lag7", "obs_vwc_surface_lag14",
            "obs_vwc_root_lag1", "obs_vwc_root_lag7", "obs_vwc_root_lag14",
        ],
        category="observations",
        source="raw",
    ),
}


class FeatureStore:
    """
    Feature store for managing ML features.

    Provides:
    - Feature versioning and caching
    - Feature metadata management
    - Feature group organization
    - Statistics computation
    """

    def __init__(
        self,
        store_path: Optional[Path] = None,
        cache_enabled: bool = True,
    ):
        """
        Initialize feature store.

        Args:
            store_path: Path for persisting features
            cache_enabled: Whether to enable caching
        """
        self.store_path = store_path or Path("./data/feature_store")
        self.cache_enabled = cache_enabled

        # In-memory cache
        self._feature_cache: Dict[str, pd.DataFrame] = {}
        self._metadata_cache: Dict[str, FeatureMetadata] = {}

        # Feature groups
        self._feature_groups: Dict[str, FeatureGroup] = FEATURE_GROUPS.copy()

        # Create store directory
        if self.store_path:
            self.store_path.mkdir(parents=True, exist_ok=True)

    def register_feature_group(self, group: FeatureGroup):
        """Register a feature group."""
        self._feature_groups[group.name] = group
        logger.info("Registered feature group: %s", group.name)

    def register_feature(self, name: str, group_name: str, metadata: FeatureMetadata):
        """Register a feature with metadata."""
        self._metadata_cache[name] = metadata
        if group_name in self._feature_groups:
            if name not in self._feature_groups[group_name].features:
                self._feature_groups[group_name].features.append(name)
        logger.info("Registered feature: %s in group %s", name, group_name)

    def get_feature_group(self, name: str) -> Optional[FeatureGroup]:
        """Get a feature group by name."""
        return self._feature_groups.get(name)

    def get_features_by_category(self, category: str) -> List[str]:
        """Get all features in a category."""
        features = []
        for group in self._feature_groups.values():
            if group.category == category:
                features.extend(group.features)
        return list(set(features))

    def get_all_feature_names(self) -> List[str]:
        """Get all registered feature names."""
        all_features = []
        for group in self._feature_groups.values():
            all_features.extend(group.features)
        return list(set(all_features))

    def compute_feature_metadata(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> Dict[str, FeatureMetadata]:
        """
        Compute metadata for all features in DataFrame.

        Args:
            df: DataFrame with features
            target_col: Optional target column for correlation

        Returns:
            Dict mapping feature name to metadata
        """
        metadata = {}

        for col in df.columns:
            if col in ['site_id', 'date', 'quality_flag']:
                continue

            # Determine source and category
            source = "raw"
            category = "other"
            for group in self._feature_groups.values():
                if col in group.features:
                    source = group.source
                    category = group.category
                    break

            # Compute statistics
            series = df[col]

            meta = FeatureMetadata(
                name=col,
                description=f"Feature: {col}",
                dtype=str(series.dtype),
                source=source,
                category=category,
                null_fraction=series.isna().mean(),
            )

            # Numeric statistics
            if np.issubdtype(series.dtype, np.number):
                meta.min_value = float(
                    series.min()) if series.notna().any() else None
                meta.max_value = float(
                    series.max()) if series.notna().any() else None
                meta.mean_value = float(
                    series.mean()) if series.notna().any() else None
                meta.std_value = float(
                    series.std()) if series.notna().any() else None

                # Correlation with target
                if target_col and target_col in df.columns:
                    target = df[target_col]
                    if np.issubdtype(target.dtype, np.number):
                        try:
                            corr = series.corr(target)
                            meta.correlation_with_target = float(
                                corr) if pd.notna(corr) else None
                        except Exception:
                            pass

            # Detect engineering parameters
            if '_lag' in col:
                try:
                    lag = int(col.split('_lag')[-1].replace('d', ''))
                    meta.lag_days = lag
                except (ValueError, IndexError):
                    pass

            if '_sum_' in col or '_mean_' in col or '_std_' in col:
                try:
                    window = int(col.split('_')[-1].replace('d', ''))
                    meta.window_days = window
                except (ValueError, IndexError):
                    pass

            metadata[col] = meta
            self._metadata_cache[col] = meta

        return metadata

    def save_features(
        self,
        df: pd.DataFrame,
        name: str,
        version: str = "1.0",
    ):
        """
        Save features to store.

        Args:
            df: DataFrame with features
            name: Feature set name
            version: Version string
        """
        if not self.store_path:
            return

        # Create versioned path
        version_path = self.store_path / name / version
        version_path.mkdir(parents=True, exist_ok=True)

        # Save data
        data_path = version_path / "features.parquet"
        df.to_parquet(data_path, index=False)

        # Save metadata
        metadata = self.compute_feature_metadata(df)
        meta_path = version_path / "metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(
                {k: v.to_dict() for k, v in metadata.items()},
                f, indent=2
            )

        # Update cache
        cache_key = f"{name}_{version}"
        self._feature_cache[cache_key] = df

        logger.info("Saved features to %s", version_path)

    def load_features(
        self,
        name: str,
        version: str = "1.0",
    ) -> Optional[pd.DataFrame]:
        """
        Load features from store.

        Args:
            name: Feature set name
            version: Version string

        Returns:
            DataFrame or None if not found
        """
        # Check cache first
        cache_key = f"{name}_{version}"
        if self.cache_enabled and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        # Load from disk
        if self.store_path:
            data_path = self.store_path / name / version / "features.parquet"
            if data_path.exists():
                df = pd.read_parquet(data_path)
                self._feature_cache[cache_key] = df
                return df

        return None

    def get_feature_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of feature DataFrame for versioning."""
        # Hash column names and dtypes
        col_str = str(sorted(df.columns.tolist()))
        dtype_str = str([(c, str(df[c].dtype)) for c in sorted(df.columns)])
        shape_str = str(df.shape)

        combined = f"{col_str}_{dtype_str}_{shape_str}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]

    def select_features(
        self,
        df: pd.DataFrame,
        groups: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Select subset of features.

        Args:
            df: Full feature DataFrame
            groups: List of feature group names to include
            categories: List of categories to include
            exclude_features: List of features to exclude

        Returns:
            DataFrame with selected features
        """
        selected_features = set()

        # By groups
        if groups:
            for group_name in groups:
                group = self._feature_groups.get(group_name)
                if group:
                    selected_features.update(group.features)

        # By categories
        if categories:
            for cat in categories:
                selected_features.update(self.get_features_by_category(cat))

        # Default: all features
        if not groups and not categories:
            selected_features = set(df.columns)

        # Exclude specified features
        if exclude_features:
            selected_features -= set(exclude_features)

        # Filter to available columns
        available = selected_features.intersection(set(df.columns))

        # Always include identifiers
        for col in ['site_id', 'date']:
            if col in df.columns:
                available.add(col)

        return df[list(available)]

    def get_high_importance_features(
        self,
        n_features: int = 50,
        min_importance: float = 0.001,
    ) -> List[str]:
        """
        Get features ranked by importance.

        Args:
            n_features: Maximum number of features to return
            min_importance: Minimum importance score

        Returns:
            List of feature names
        """
        scored_features = [
            (meta.name, meta.importance_score)
            for meta in self._metadata_cache.values()
            if meta.importance_score is not None and meta.importance_score >= min_importance
        ]

        # Sort by importance
        scored_features.sort(key=lambda x: x[1], reverse=True)

        return [f[0] for f in scored_features[:n_features]]

    def update_feature_importance(
        self,
        importance_scores: Dict[str, float],
    ):
        """
        Update feature importance scores from model.

        Args:
            importance_scores: Dict mapping feature name to importance
        """
        for name, score in importance_scores.items():
            if name in self._metadata_cache:
                self._metadata_cache[name].importance_score = score

    def get_feature_summary(self) -> pd.DataFrame:
        """Get summary of all registered features."""
        rows = []

        for group in self._feature_groups.values():
            for feature in group.features:
                meta = self._metadata_cache.get(feature)
                rows.append({
                    'feature': feature,
                    'group': group.name,
                    'category': group.category,
                    'source': group.source,
                    'importance': meta.importance_score if meta else None,
                    'correlation': meta.correlation_with_target if meta else None,
                    'null_fraction': meta.null_fraction if meta else None,
                })

        return pd.DataFrame(rows)
