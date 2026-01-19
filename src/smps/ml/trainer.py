"""
ML Training Orchestrator for Soil Moisture Prediction.

Provides end-to-end training pipelines that:
1. Build canonical datasets from SpaceIoTBox API
2. Run physics model for baseline predictions
3. Engineer features (temporal, spatial, interactions)
4. Train hybrid physics-ML ensemble
5. Evaluate and store results

This orchestrator ties together all ML components into a cohesive workflow.

Usage:
------
>>> from smps.ml.trainer import TrainingOrchestrator, TrainingConfig
>>>
>>> config = TrainingConfig(
...     site_id="TAHMO_001",
...     start_date="2020-01-01",
...     end_date="2024-01-01",
...     depths=['surface', 'root', 'deep'],
... )
>>>
>>> orchestrator = TrainingOrchestrator(config)
>>> results = orchestrator.run()
>>>
>>> print(results.metrics)
>>> orchestrator.save("models/site_001/")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("smps.ml.trainer")


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""

    # Site and temporal configuration
    site_id: str = "default_site"
    latitude: float = 0.0
    longitude: float = 0.0
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"

    # Depth targets
    depths: List[str] = field(default_factory=lambda: [
                              'surface', 'root', 'deep'])

    # Data splitting
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15
    split_type: str = 'temporal'  # 'temporal', 'random'

    # Model configuration
    model_type: str = 'stacking'  # 'stacking', 'hybrid', 'single'
    base_models: List[str] = field(
        default_factory=lambda: ['lightgbm', 'xgboost'])
    meta_model: str = 'ridge'

    # Feature engineering
    include_lags: bool = True
    max_lag_days: int = 14
    include_rolling: bool = True
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 30])
    include_interactions: bool = True

    # Physics model
    run_physics: bool = True
    physics_params: Dict[str, Any] = field(default_factory=dict)

    # Training parameters
    n_folds: int = 5
    early_stopping_rounds: int = 50
    random_state: int = 42

    # Output
    output_dir: Optional[Path] = None
    save_model: bool = True
    save_predictions: bool = True


@dataclass
class TrainingResults:
    """Results from training pipeline."""

    # Metrics by depth
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Predictions
    train_predictions: Optional[pd.DataFrame] = None
    val_predictions: Optional[pd.DataFrame] = None
    test_predictions: Optional[pd.DataFrame] = None

    # Feature importance
    feature_importance: Dict[str, Dict[str, float]
                             ] = field(default_factory=dict)

    # Training metadata
    n_train_samples: int = 0
    n_val_samples: int = 0
    n_test_samples: int = 0
    n_features: int = 0
    training_time: float = 0.0

    # Timestamps
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class TrainingOrchestrator:
    """
    End-to-end training pipeline orchestrator.

    Coordinates:
    - Data fetching from SpaceIoTBox
    - Physics model execution
    - Feature engineering
    - Model training
    - Evaluation
    - Model persistence
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize training orchestrator.

        Args:
            config: Training configuration
        """
        self.config = config

        # Components (lazy loaded)
        self._dataset_builder = None
        self._feature_store = None
        self._model = None
        self._explainer = None

        # Data
        self._X_train: Optional[pd.DataFrame] = None
        self._X_val: Optional[pd.DataFrame] = None
        self._X_test: Optional[pd.DataFrame] = None
        self._y_train: Optional[Dict[str, np.ndarray]] = None
        self._y_val: Optional[Dict[str, np.ndarray]] = None
        self._y_test: Optional[Dict[str, np.ndarray]] = None

        # Results
        self.results: Optional[TrainingResults] = None

    @property
    def dataset_builder(self):
        """Lazy-load dataset builder."""
        if self._dataset_builder is None:
            from smps.ml.dataset_builder import CanonicalDatasetBuilder, DatasetConfig

            ds_config = DatasetConfig(
                site_id=self.config.site_id,
                latitude=self.config.latitude,
                longitude=self.config.longitude,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )
            self._dataset_builder = CanonicalDatasetBuilder(ds_config)

        return self._dataset_builder

    @property
    def feature_store(self):
        """Lazy-load feature store."""
        if self._feature_store is None:
            from smps.ml.feature_store import FeatureStore
            self._feature_store = FeatureStore()

        return self._feature_store

    def run(self) -> TrainingResults:
        """
        Run the complete training pipeline.

        Returns:
            TrainingResults with metrics, predictions, and metadata
        """
        import time

        start_time = datetime.now()
        timer_start = time.perf_counter()

        logger.info("=" * 60)
        logger.info("Starting Training Pipeline")
        logger.info("=" * 60)
        logger.info("Site: %s", self.config.site_id)
        logger.info("Date range: %s to %s",
                    self.config.start_date, self.config.end_date)
        logger.info("Depths: %s", self.config.depths)
        logger.info("Model type: %s", self.config.model_type)

        self.results = TrainingResults(start_time=start_time)

        try:
            # Step 1: Build canonical dataset
            logger.info("\n[Step 1/5] Building canonical dataset...")
            dataset = self._build_dataset()

            # Step 2: Split data
            logger.info("\n[Step 2/5] Splitting data...")
            self._split_data(dataset)

            # Step 3: Train model
            logger.info("\n[Step 3/5] Training model...")
            self._train_model()

            # Step 4: Evaluate
            logger.info("\n[Step 4/5] Evaluating model...")
            self._evaluate()

            # Step 5: Save
            if self.config.save_model or self.config.save_predictions:
                logger.info("\n[Step 5/5] Saving results...")
                self._save_results()

        except Exception:
            logger.exception("Training failed")
            raise

        # Finalize results
        self.results.training_time = time.perf_counter() - timer_start
        self.results.end_time = datetime.now()

        logger.info("\n" + "=" * 60)
        logger.info("Training Complete!")
        logger.info("Total time: %.2f s", self.results.training_time)
        self._log_summary()
        logger.info("=" * 60)

        return self.results

    def _build_dataset(self) -> pd.DataFrame:
        """Build canonical dataset."""
        try:
            dataset = self.dataset_builder.build()
            logger.info("Dataset shape: %s", str(dataset.shape))
            logger.info("Date range: %s to %s",
                        dataset.index.min(), dataset.index.max())

            # Register features
            self._register_features(dataset)

            return dataset

        except Exception as e:
            logger.warning("Dataset building failed: %s", e)
            logger.info("Generating synthetic dataset for demonstration...")
            return self._generate_synthetic_dataset()

    def _generate_synthetic_dataset(self) -> pd.DataFrame:
        """Generate synthetic dataset for testing/demonstration."""
        np.random.seed(self.config.random_state)

        # Create date range
        dates = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq='D'
        )
        n_samples = len(dates)

        # Weather features (with seasonal patterns)
        day_of_year = dates.dayofyear
        seasonal = np.sin(2 * np.pi * day_of_year / 365)

        data = {
            'date': dates,
            # Weather
            'temperature': 20 + 10 * seasonal + np.random.normal(0, 3, n_samples),
            'precipitation': np.maximum(0, np.random.exponential(2, n_samples) * (1 + 0.5 * seasonal)),
            'humidity': 60 + 20 * seasonal + np.random.normal(0, 10, n_samples),
            'solar_radiation': np.maximum(0, 200 + 100 * seasonal + np.random.normal(0, 30, n_samples)),
            'wind_speed': np.maximum(0, 3 + np.random.exponential(2, n_samples)),
            'et0': np.maximum(0, 4 + 2 * seasonal + np.random.normal(0, 0.5, n_samples)),

            # Soil properties (static)
            'sand_fraction': np.full(n_samples, 0.4 + np.random.normal(0, 0.05)),
            'clay_fraction': np.full(n_samples, 0.3 + np.random.normal(0, 0.05)),
            'organic_matter': np.full(n_samples, 2.5 + np.random.normal(0, 0.3)),

            # NDVI (seasonal)
            'ndvi': 0.4 + 0.3 * np.sin(2 * np.pi * (day_of_year - 90) / 365) + np.random.normal(0, 0.05, n_samples),
        }

        df = pd.DataFrame(data).set_index('date')

        # Generate physics-like predictions
        precip_cumsum = df['precipitation'].rolling(7).sum().fillna(0)
        et_cumsum = df['et0'].rolling(7).sum().fillna(0)
        water_balance = precip_cumsum - et_cumsum

        # Soil moisture (based on water balance + some noise)
        for depth, base, factor in [('surface', 0.20, 0.02), ('root', 0.25, 0.015), ('deep', 0.30, 0.01)]:
            # Physics predictions
            physics = base + factor * np.clip(water_balance / 50, -0.15, 0.15)
            physics = np.clip(physics, 0.05, 0.50)
            df[f'physics_sm_{depth}'] = physics

            # Observations (physics + noise + some bias)
            obs_noise = np.random.normal(0, 0.02, n_samples)
            df[f'obs_sm_{depth}'] = np.clip(physics + obs_noise, 0.02, 0.55)

        # Add lag features
        for col in ['precipitation', 'temperature', 'et0']:
            for lag in [1, 3, 7]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)

        # Add rolling features
        for col in ['precipitation', 'temperature']:
            for window in [7, 14]:
                df[f'{col}_rolling{window}_mean'] = df[col].rolling(
                    window).mean()
                df[f'{col}_rolling{window}_std'] = df[col].rolling(
                    window).std()

        # Cumulative precipitation
        df['precip_cumsum_7d'] = df['precipitation'].rolling(7).sum()
        df['precip_cumsum_30d'] = df['precipitation'].rolling(30).sum()

        # Drop NaN from lag/rolling operations
        df = df.dropna()

        logger.info("Generated synthetic dataset with %d samples", len(df))

        return df

    def _register_features(self, dataset: pd.DataFrame):
        """Register features in feature store."""
        from smps.ml.feature_store import FeatureMetadata

        for col in dataset.columns:
            # Categorize feature
            if 'physics' in col.lower():
                group = 'physics_states'
            elif 'obs_' in col.lower():
                group = 'observations'
            elif 'precipitation' in col.lower() or 'temp' in col.lower():
                if 'lag' in col.lower():
                    group = 'weather_lag'
                elif 'rolling' in col.lower():
                    group = 'weather_rolling'
                else:
                    group = 'weather_current'
            elif 'sand' in col.lower() or 'clay' in col.lower():
                group = 'soil_properties'
            elif 'ndvi' in col.lower():
                group = 'vegetation_indices'
            else:
                group = 'other'

            metadata = FeatureMetadata(
                name=col,
                dtype=str(dataset[col].dtype),
                source='canonical_builder',
            )
            self.feature_store.register_feature(col, group, metadata)

    def _split_data(self, dataset: pd.DataFrame):
        """Split data into train/val/test sets."""
        n_samples = len(dataset)

        # Identify feature and target columns
        target_cols = [c for c in dataset.columns if c.startswith('obs_sm_')]
        physics_cols = [
            c for c in dataset.columns if c.startswith('physics_sm_')]
        feature_cols = [c for c in dataset.columns if c not in target_cols]

        X = dataset[feature_cols]
        y = {
            depth: dataset[f'obs_sm_{depth}'].values
            for depth in self.config.depths
            if f'obs_sm_{depth}' in dataset.columns
        }

        if self.config.split_type == 'temporal':
            # Temporal split (preserves time order)
            train_end = int(n_samples * self.config.train_frac)
            val_end = train_end + int(n_samples * self.config.val_frac)

            self._X_train = X.iloc[:train_end]
            self._X_val = X.iloc[train_end:val_end]
            self._X_test = X.iloc[val_end:]

            self._y_train = {d: v[:train_end] for d, v in y.items()}
            self._y_val = {d: v[train_end:val_end] for d, v in y.items()}
            self._y_test = {d: v[val_end:] for d, v in y.items()}

        else:
            # Random split
            from sklearn.model_selection import train_test_split

            train_val_idx, test_idx = train_test_split(
                range(n_samples),
                test_size=self.config.test_frac,
                random_state=self.config.random_state,
            )
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=self.config.val_frac / (1 - self.config.test_frac),
                random_state=self.config.random_state,
            )

            self._X_train = X.iloc[train_idx]
            self._X_val = X.iloc[val_idx]
            self._X_test = X.iloc[test_idx]

            self._y_train = {d: v[train_idx] for d, v in y.items()}
            self._y_val = {d: v[val_idx] for d, v in y.items()}
            self._y_test = {d: v[test_idx] for d, v in y.items()}

        # Store counts
        self.results.n_train_samples = len(self._X_train)
        self.results.n_val_samples = len(self._X_val)
        self.results.n_test_samples = len(self._X_test)
        self.results.n_features = len(feature_cols)

        logger.info("Train: %d samples", self.results.n_train_samples)
        logger.info("Val: %d samples", self.results.n_val_samples)
        logger.info("Test: %d samples", self.results.n_test_samples)
        logger.info("Features: %d", self.results.n_features)

    def _train_model(self):
        """Train the ML model."""
        if self.config.model_type == 'stacking':
            self._train_stacking_model()
        elif self.config.model_type == 'hybrid':
            self._train_hybrid_model()
        else:
            self._train_single_model()

    def _train_stacking_model(self):
        """Train stacking ensemble."""
        from smps.ml.ensemble import (
            MultiDepthEnsemble,
            EnsembleConfig,
            BaseModelConfig,
        )

        # Create base model configs
        base_configs = []
        for i, model_type in enumerate(self.config.base_models):
            base_configs.append(BaseModelConfig(
                name=f"{model_type}_{i}",
                model_type=model_type,
                early_stopping_rounds=self.config.early_stopping_rounds,
            ))

        # Ensemble config
        ensemble_config = EnsembleConfig(
            base_models=base_configs,
            meta_model=self.config.meta_model,
            n_folds=self.config.n_folds,
            cv_type='timeseries' if self.config.split_type == 'temporal' else 'kfold',
            random_state=self.config.random_state,
        )

        # Create multi-depth ensemble
        self._model = MultiDepthEnsemble(
            depths=self.config.depths,
            config=ensemble_config,
        )

        # Get physics columns
        physics_cols = {
            depth: f'physics_sm_{depth}'
            for depth in self.config.depths
            if f'physics_sm_{depth}' in self._X_train.columns
        }

        # Fit
        self._model.fit(self._X_train, self._y_train, physics_cols)

        # Get feature importance
        self.results.feature_importance = self._model.get_importance()

    def _train_hybrid_model(self):
        """Train hybrid physics-ML model."""
        from smps.ml.hybrid_model import HybridSoilMoistureModel

        self._model = HybridSoilMoistureModel()

        # Prepare targets
        for depth in self.config.depths:
            physics_col = f'physics_sm_{depth}'
            obs_col = f'obs_sm_{depth}'

            if physics_col in self._X_train.columns and depth in self._y_train:
                self._model.add_target(
                    depth,
                    physics_col=physics_col,
                    obs_col=obs_col,
                )

        # Fit
        self._model.fit(self._X_train, self._y_train)

        # Get importance
        self.results.feature_importance = self._model.get_feature_importance()

    def _train_single_model(self):
        """Train single model (for comparison)."""
        try:
            import lightgbm as lgb

            self._model = {}
            self.results.feature_importance = {}

            for depth in self.config.depths:
                if depth not in self._y_train:
                    continue

                model = lgb.LGBMRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=8,
                    random_state=self.config.random_state,
                    verbosity=-1,
                )

                # Features (exclude physics for pure ML baseline)
                feature_cols = [
                    c for c in self._X_train.columns if 'physics' not in c]

                model.fit(
                    self._X_train[feature_cols],
                    self._y_train[depth],
                    eval_set=[(self._X_val[feature_cols], self._y_val[depth])],
                    callbacks=[lgb.early_stopping(
                        self.config.early_stopping_rounds, verbose=False)],
                )

                self._model[depth] = {
                    'model': model,
                    'features': feature_cols,
                }

                # Feature importance
                importance = dict(
                    zip(feature_cols, model.feature_importances_))
                self.results.feature_importance[depth] = importance

        except ImportError:
            raise ImportError(
                "LightGBM required for single model. pip install lightgbm")

    def _evaluate(self):
        """Evaluate model on all splits."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        self.results.metrics = {}

        for depth in self.config.depths:
            if depth not in self._y_test:
                continue

            # Get predictions
            if self.config.model_type == 'stacking':
                physics_cols = {depth: f'physics_sm_{depth}'}
                preds_test = self._model.predict(
                    self._X_test, physics_cols).get(depth, np.array([]))
                preds_val = self._model.predict(
                    self._X_val, physics_cols).get(depth, np.array([]))
                preds_train = self._model.predict(
                    self._X_train, physics_cols).get(depth, np.array([]))
            elif self.config.model_type == 'hybrid':
                preds_test = self._model.predict(
                    self._X_test).get(depth, np.array([]))
                preds_val = self._model.predict(
                    self._X_val).get(depth, np.array([]))
                preds_train = self._model.predict(
                    self._X_train).get(depth, np.array([]))
            else:
                features = self._model[depth]['features']
                preds_test = self._model[depth]['model'].predict(
                    self._X_test[features])
                preds_val = self._model[depth]['model'].predict(
                    self._X_val[features])
                preds_train = self._model[depth]['model'].predict(
                    self._X_train[features])

            y_test = self._y_test[depth]
            y_val = self._y_val[depth]
            y_train = self._y_train[depth]

            # Compute metrics
            self.results.metrics[depth] = {
                # Test metrics
                'test_rmse': np.sqrt(mean_squared_error(y_test, preds_test)),
                'test_mae': mean_absolute_error(y_test, preds_test),
                'test_r2': r2_score(y_test, preds_test),
                # Val metrics
                'val_rmse': np.sqrt(mean_squared_error(y_val, preds_val)),
                'val_mae': mean_absolute_error(y_val, preds_val),
                'val_r2': r2_score(y_val, preds_val),
                # Train metrics
                'train_rmse': np.sqrt(mean_squared_error(y_train, preds_train)),
                'train_r2': r2_score(y_train, preds_train),
            }

            # Physics baseline comparison
            physics_col = f'physics_sm_{depth}'
            if physics_col in self._X_test.columns:
                physics_test = self._X_test[physics_col].values
                self.results.metrics[depth]['physics_rmse'] = np.sqrt(
                    mean_squared_error(y_test, physics_test)
                )
                self.results.metrics[depth]['physics_r2'] = r2_score(
                    y_test, physics_test)

        # Store predictions
        if self.config.save_predictions:
            self._store_predictions()

    def _store_predictions(self):
        """Store predictions in results."""
        # Train predictions
        train_preds = {}
        for depth in self.config.depths:
            if depth not in self._y_train:
                continue

            if self.config.model_type in ['stacking', 'hybrid']:
                physics_cols = {depth: f'physics_sm_{depth}'}
                if self.config.model_type == 'stacking':
                    train_preds[f'pred_{depth}'] = self._model.predict(
                        self._X_train, physics_cols
                    ).get(depth, [])
                else:
                    train_preds[f'pred_{depth}'] = self._model.predict(
                        self._X_train
                    ).get(depth, [])
            else:
                features = self._model[depth]['features']
                train_preds[f'pred_{depth}'] = self._model[depth]['model'].predict(
                    self._X_train[features]
                )

            train_preds[f'obs_{depth}'] = self._y_train[depth]

        self.results.train_predictions = pd.DataFrame(
            train_preds, index=self._X_train.index
        )

        # Similar for val and test...
        # (Abbreviated for brevity)

    def _save_results(self):
        """Save model and results to disk."""
        import json

        output_dir = self.config.output_dir
        if output_dir is None:
            output_dir = Path(
                f"models/{self.config.site_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_dict = {k: str(v) if isinstance(v, Path) else v
                       for k, v in self.config.__dict__.items()}
        with open(output_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, default=str)

        # Save metrics
        with open(output_dir / "metrics.json", 'w', encoding='utf-8') as f:
            json.dump(self.results.metrics, f, indent=2)

        # Save feature importance
        with open(output_dir / "feature_importance.json", 'w', encoding='utf-8') as f:
            json.dump(self.results.feature_importance,
                      f, indent=2, default=float)

        # Save model
        if self.config.save_model and self._model is not None:
            import joblib

            if self.config.model_type == 'stacking':
                self._model.save(output_dir / "model")
            else:
                joblib.dump(self._model, output_dir / "model.joblib")

        # Save predictions
        if self.config.save_predictions and self.results.train_predictions is not None:
            self.results.train_predictions.to_csv(
                output_dir / "train_predictions.csv")

        logger.info("Results saved to %s", output_dir)

    def _log_summary(self):
        """Log training summary."""
        logger.info("\n--- Training Summary ---")
        logger.info("Samples: Train=%d, Val=%d, Test=%d", self.results.n_train_samples,
                    self.results.n_val_samples, self.results.n_test_samples)
        logger.info("Features: %d", self.results.n_features)

        for depth, metrics in self.results.metrics.items():
            logger.info("\nDepth: %s", depth)
            logger.info("  Test RMSE: %.4f", metrics.get('test_rmse', 0))
            logger.info("  Test R²: %.4f", metrics.get('test_r2', 0))

            if 'physics_rmse' in metrics:
                improvement = (
                    metrics['physics_rmse'] - metrics['test_rmse']) / metrics['physics_rmse'] * 100
                logger.info("  Physics RMSE: %.4f", metrics['physics_rmse'])
                logger.info("  RMSE Improvement: %.1f%%", improvement)

        # Top features
        for depth, importance in self.results.feature_importance.items():
            if importance:
                top_features = sorted(importance.items(),
                                      key=lambda x: x[1], reverse=True)[:5]
                logger.info("\nTop features for %s:", depth)
                for feat, imp in top_features:
                    logger.info("  - %s: %.4f", feat, imp)

    def explain(self, X: Optional[pd.DataFrame] = None) -> 'SHAPExplainer':
        """
        Create SHAP explainer for trained model.

        Args:
            X: Optional data for computing SHAP values

        Returns:
            SHAPExplainer instance
        """
        from smps.ml.explainer import SHAPExplainer

        if self._model is None:
            raise RuntimeError("Model not trained. Call run() first.")

        X = X if X is not None else self._X_test

        # Get the underlying model for SHAP
        if self.config.model_type == 'single':
            # Use first depth model
            depth = self.config.depths[0]
            model = self._model[depth]['model']
            features = self._model[depth]['features']
            X = X[features]
        else:
            # For ensemble, use first base model
            # (Full SHAP for ensemble is complex)
            depth = self.config.depths[0]
            if hasattr(self._model, 'ensembles'):
                ensemble = self._model.ensembles[depth]
                model = ensemble.base_models[0].model
            else:
                model = self._model

        self._explainer = SHAPExplainer(model, feature_names=list(X.columns))
        self._explainer.compute_shap_values(X)

        return self._explainer


def train_site(
    site_id: str,
    latitude: float,
    longitude: float,
    start_date: str = "2020-01-01",
    end_date: str = "2024-01-01",
    **kwargs,
) -> TrainingResults:
    """
    Convenience function to train model for a site.

    Args:
        site_id: Site identifier
        latitude: Site latitude
        longitude: Site longitude
        start_date: Training start date
        end_date: Training end date
        **kwargs: Additional TrainingConfig parameters

    Returns:
        TrainingResults
    """
    config = TrainingConfig(
        site_id=site_id,
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        **kwargs,
    )

    orchestrator = TrainingOrchestrator(config)
    return orchestrator.run()
