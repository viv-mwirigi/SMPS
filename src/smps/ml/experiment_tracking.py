"""
Experiment tracking and model registry for SMPS.

Provides:
- Experiment tracking with MLflow
- Model versioning and lineage
- Hyperparameter optimization with Optuna
- Model registry with metadata
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib

import mlflow
import mlflow.lightgbm
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from smps.core.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    experiment_name: str = "smps_theta_space"
    tracking_uri: Optional[str] = None
    artifact_location: Optional[Path] = None

    # Hyperparameter optimization
    enable_optuna: bool = True
    n_optuna_trials: int = 50
    optuna_timeout_minutes: int = 30

    # Model registry
    register_best_models: bool = True
    model_stage: str = "Development"

    def __post_init__(self):
        if self.artifact_location is None:
            self.artifact_location = settings.results_dir / "mlruns"


@dataclass
class ModelMetadata:
    """Metadata for trained models."""
    experiment_id: str
    run_id: str
    model_name: str
    horizon_hours: int
    feature_columns: List[str]
    training_config: Dict[str, Any]
    cv_results: Dict[str, Any]
    feature_importance: Dict[str, float]
    data_hash: str
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "model_name": self.model_name,
            "horizon_hours": self.horizon_hours,
            "feature_columns": self.feature_columns,
            "training_config": self.training_config,
            "cv_results": self.cv_results,
            "feature_importance": self.feature_importance,
            "data_hash": self.data_hash,
            "created_at": self.created_at.isoformat(),
        }


class ExperimentTracker:
    """ML experiment tracking and model registry."""

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Set up MLflow tracking."""
        if self.config.tracking_uri:
            mlflow.set_tracking_uri(str(self.config.tracking_uri))
        else:
            # Use local tracking
            mlflow.set_tracking_uri(str(self.config.artifact_location))

        mlflow.set_experiment(self.config.experiment_name)

    def _calculate_data_hash(self, X: pd.DataFrame, y: np.ndarray) -> str:
        """Calculate hash of training data for reproducibility."""
        data_str = f"{X.shape}_{hash(str(X.values.tobytes())[:1000])}_{hash(str(y)[:1000])}"
        return hashlib.md5(data_str.encode()).hexdigest()[:8]

    def start_run(self, run_name: Optional[str] = None) -> str:
        """Start a new MLflow run."""
        run = mlflow.start_run(run_name=run_name)
        return run.info.run_id

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, model_name: str, feature_columns: List[str]):
        """Log model to MLflow."""
        mlflow.lightgbm.log_model(model, model_name, input_example=None)

        # Log feature columns as artifact
        feature_info = {"features": feature_columns}
        feature_path = Path("feature_info.json")
        feature_path.write_text(json.dumps(feature_info, indent=2))
        mlflow.log_artifact(str(feature_path))
        feature_path.unlink()

    def log_metadata(self, metadata: ModelMetadata):
        """Log model metadata as artifact."""
        metadata_path = Path("model_metadata.json")
        metadata_path.write_text(json.dumps(
            metadata.to_dict(), indent=2, default=str))
        mlflow.log_artifact(str(metadata_path))
        metadata_path.unlink()

    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()

    def register_model(self, run_id: str, model_name: str, model_version: str):
        """Register model in MLflow Model Registry."""
        if self.config.register_best_models:
            try:
                model_uri = f"runs:/{run_id}/{model_name}"
                mlflow.register_model(
                    model_uri, f"{model_name}_v{model_version}")
                logger.info(f"Registered model {model_name}_v{model_version}")
            except Exception as e:
                logger.warning(f"Failed to register model: {e}")


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def optimize_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          feature_names: List[str]) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters using Optuna."""

        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            }

            model = lgb.LGBMRegressor(**params, random_state=42)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      eval_metric='rmse',
                      callbacks=[lgb.early_stopping(50, verbose=False)])

            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            return rmse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.config.n_optuna_trials,
                       timeout=self.config.optuna_timeout_minutes * 60)

        logger.info(f"Best trial: {study.best_trial.value}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params


class ModelVersionManager:
    """Model versioning and lineage tracking."""

    def __init__(self, registry_dir: Optional[Path] = None):
        self.registry_dir = registry_dir or settings.models_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "model_registry.json"

        # Load existing registry
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, List[ModelMetadata]]:
        """Load model registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    # Convert back to ModelMetadata objects
                    registry = {}
                    for model_name, versions in data.items():
                        registry[model_name] = [
                            ModelMetadata(**v) for v in versions
                        ]
                    return registry
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")

        return {}

    def _save_registry(self):
        """Save model registry to disk."""
        data = {}
        for model_name, versions in self.registry.items():
            data[model_name] = [v.to_dict() for v in versions]

        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def register_model(self, metadata: ModelMetadata):
        """Register a new model version."""
        if metadata.model_name not in self.registry:
            self.registry[metadata.model_name] = []

        self.registry[metadata.model_name].append(metadata)
        self._save_registry()

        logger.info(
            f"Registered model {metadata.model_name} version {len(self.registry[metadata.model_name])}")

    def get_latest_version(self, model_name: str) -> Optional[ModelMetadata]:
        """Get the latest version of a model."""
        if model_name in self.registry and self.registry[model_name]:
            return max(self.registry[model_name], key=lambda x: x.created_at)
        return None

    def list_versions(self, model_name: str) -> List[ModelMetadata]:
        """List all versions of a model."""
        return self.registry.get(model_name, [])

    def get_model_path(self, metadata: ModelMetadata) -> Path:
        """Get the file path for a model's artifacts."""
        return self.registry_dir / metadata.experiment_id / metadata.run_id / "artifacts"


# Global instances
experiment_tracker = ExperimentTracker()
model_registry = ModelVersionManager()
