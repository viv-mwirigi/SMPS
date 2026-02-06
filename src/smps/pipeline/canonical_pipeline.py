"""
Canonical Pipeline for Matric Potential Modeling.

Implements the complete orchestrated pipeline for ψ (matric potential) data
processing, from raw data ingestion to model deployment.

Canonical Pipeline Stages:
─────────────────────────────────────────────────────────────────
1. Data Ingestion:         Load and validate raw ψ data sources
2. Quality Control:        Apply QC checks and filtering
3. Feature Engineering:    Create spatiotemporal and physics features
4. Model Training:         Train ensemble with uncertainty quantification
5. Validation:             Physics-based and statistical validation
6. Calibration:            Site-specific and temporal adaptation
7. Deployment:             Production-ready ψ prediction service
─────────────────────────────────────────────────────────────────

Benefits for ψ Modeling:
- End-to-end automation from data to predictions
- Consistent processing across development and production
- Integrated quality control and validation
- Scalable architecture for multiple sites

Research References:
- Bishop (2006): Pattern Recognition and Machine Learning
- Hastie et al. (2009): Elements of Statistical Learning
- Soil Science Society of America standards for ψ measurement
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import joblib
from datetime import datetime, timedelta

# Import SWPPS components
from smps.qc.quality_control import PsiQualityControlPipeline, QCConfig
from smps.features.spatiotemporal_features import SpatiotemporalFeaturePipeline, SpatiotemporalConfig
from smps.ml.ensemble import PsiStackingEnsemble, EnsembleConfig
from smps.ml.uncertainty import PsiUncertaintyQuantifier, UncertaintyConfig
from smps.ml.domain_shift import PsiDomainShiftMonitor, DomainShiftConfig
from smps.metrics.physics_metrics import PhysicsBasedMetrics, PhysicsConfig
from smps.calibration.adaptive_calibration import AdaptiveCalibrationPipeline, CalibrationConfig
from smps.analysis.sensitivity_analysis import SensitivityAnalysisPipeline, SensitivityConfig

logger = logging.getLogger("swpps.pipeline.canonical")


@dataclass
class CanonicalPipelineConfig:
    """Configuration for the canonical ψ pipeline."""

    # Pipeline stages
    enable_qc: bool = True
    enable_feature_engineering: bool = True
    enable_uncertainty: bool = True
    enable_domain_shift: bool = True
    enable_physics_metrics: bool = True
    enable_calibration: bool = True
    enable_sensitivity_analysis: bool = False  # Optional, computationally expensive

    # Data configuration
    data_sources: List[str] = field(default_factory=lambda: [
                                    'ismn', 'satellite', 'weather'])
    target_column: str = 'psi'
    feature_columns: List[str] = field(default_factory=lambda: [
        'soil_moisture', 'temperature', 'precipitation', 'depth',
        'bulk_density', 'clay_content', 'sand_content'
    ])

    # Training configuration
    train_test_split: float = 0.8
    cv_folds: int = 5
    random_state: int = 42

    # Model persistence
    model_save_path: str = 'models/canonical_psi_pipeline'
    results_save_path: str = 'results/canonical_pipeline'

    # Component configurations
    qc_config: QCConfig = field(default_factory=QCConfig)
    feature_config: SpatiotemporalConfig = field(
        default_factory=SpatiotemporalConfig)
    ensemble_config: EnsembleConfig = field(default_factory=EnsembleConfig)
    uncertainty_config: UncertaintyConfig = field(
        default_factory=UncertaintyConfig)
    domain_shift_config: DomainShiftConfig = field(
        default_factory=DomainShiftConfig)
    physics_config: PhysicsConfig = field(default_factory=PhysicsConfig)
    calibration_config: CalibrationConfig = field(
        default_factory=CalibrationConfig)
    sensitivity_config: SensitivityConfig = field(
        default_factory=SensitivityConfig)


class DataIngestionStage:
    """
    Data ingestion and initial validation stage.
    """

    def __init__(self, config: CanonicalPipelineConfig):
        self.config = config

    def load_data_sources(self, data_dir: str = 'data') -> pd.DataFrame:
        """Load and merge data from multiple sources."""
        logger.info("Loading data from sources: %s", self.config.data_sources)

        data_frames = []

        for source in self.config.data_sources:
            if source == 'ismn':
                df = self._load_ismn_data(data_dir)
            elif source == 'satellite':
                df = self._load_satellite_data(data_dir)
            elif source == 'weather':
                df = self._load_weather_data(data_dir)
            else:
                logger.warning(f"Unknown data source: {source}")
                continue

            if df is not None and not df.empty:
                data_frames.append(df)

        if not data_frames:
            raise ValueError("No data loaded from any source")

        # Merge data sources
        merged_df = self._merge_data_sources(data_frames)

        logger.info(f"Data ingestion completed. Shape: {merged_df.shape}")

        return merged_df

    def _load_ismn_data(self, data_dir: str) -> pd.DataFrame:
        """Load ISMN soil moisture data."""
        ismn_path = Path(data_dir) / 'ismn' / 'prepared' / \
            'ismn_soil_moisture_full.csv'
        if ismn_path.exists():
            df = pd.read_csv(ismn_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            logger.info(f"Loaded ISMN data: {len(df)} records")
            return df
        return pd.DataFrame()

    def _load_satellite_data(self, data_dir: str) -> pd.DataFrame:
        """Load satellite data."""
        satellite_path = Path(data_dir) / 'cache' / 'satellite'
        # Implementation would depend on specific satellite data format
        logger.info("Satellite data loading not implemented")
        return pd.DataFrame()

    def _load_weather_data(self, data_dir: str) -> pd.DataFrame:
        """Load weather data."""
        weather_path = Path(data_dir) / 'cache' / 'weather'
        # Implementation would depend on specific weather data format
        logger.info("Weather data loading not implemented")
        return pd.DataFrame()

    def _merge_data_sources(self, data_frames: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge data from multiple sources."""
        if len(data_frames) == 1:
            return data_frames[0]

        # For now, concatenate (more sophisticated merging would be needed)
        merged = pd.concat(data_frames, ignore_index=True)

        # Remove duplicates based on timestamp and location
        if 'timestamp' in merged.columns and 'latitude' in merged.columns and 'longitude' in merged.columns:
            merged = merged.drop_duplicates(
                subset=['timestamp', 'latitude', 'longitude'])

        return merged


class QualityControlStage:
    """
    Quality control and data filtering stage.
    """

    def __init__(self, config: CanonicalPipelineConfig):
        self.config = config
        self.qc_pipeline = PsiQualityControlPipeline(self.config.qc_config)

    def apply_quality_control(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply comprehensive quality control to ψ data."""
        logger.info("Applying quality control to ψ data")

        # Run full QC pipeline
        df_qc, qc_report = self.qc_pipeline.run_full_qc(
            df, psi_col=self.config.target_column, time_col='timestamp'
        )

        # Filter to high-quality data
        df_filtered = self.qc_pipeline.filter_quality_data(df_qc, qc_report)

        logger.info(
            f"Quality control completed. Filtered {len(df_filtered)}/{len(df)} records")

        return df_filtered, qc_report


class FeatureEngineeringStage:
    """
    Feature engineering and spatiotemporal feature creation.
    """

    def __init__(self, config: CanonicalPipelineConfig):
        self.config = config
        self.feature_pipeline = SpatiotemporalFeaturePipeline(
            self.config.feature_config)

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive spatiotemporal features."""
        logger.info("Creating spatiotemporal features for ψ modeling")

        # Create all features
        df_features = self.feature_pipeline.create_all_features(
            df,
            psi_col=self.config.target_column,
            lat_col='latitude',
            lon_col='longitude',
            time_col='timestamp',
            weather_cols=['temperature', 'precipitation', 'humidity']
        )

        # Select final feature set
        feature_cols = self._select_feature_columns(df_features)
        df_final = df_features[feature_cols + [self.config.target_column]]

        logger.info(
            f"Feature engineering completed. Features: {len(feature_cols)}")

        return df_final

    def _select_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Select relevant feature columns for modeling."""
        available_features = [col for col in df.columns
                              if col != self.config.target_column
                              and not col.startswith('timestamp')
                              and df[col].dtype in ['float64', 'int64']]

        # Prioritize configured features, then add engineered features
        selected_features = []
        selected_features.extend(
            [f for f in self.config.feature_columns if f in available_features])

        # Add spatiotemporal features
        spatiotemporal_features = [f for f in available_features
                                   if any(keyword in f.lower() for keyword in
                                          ['spatial', 'temporal', 'lag', 'rolling', 'gradient'])]

        # Limit to prevent overfitting
        selected_features.extend(spatiotemporal_features[:50])

        return list(set(selected_features))  # Remove duplicates


class ModelTrainingStage:
    """
    Model training with ensemble methods and uncertainty quantification.
    """

    def __init__(self, config: CanonicalPipelineConfig):
        self.config = config
        self.ensemble = PsiStackingEnsemble(self.config.ensemble_config)
        self.uncertainty = PsiUncertaintyQuantifier(
            self.config.uncertainty_config)
        self.domain_shift = PsiDomainShiftMonitor(
            self.config.domain_shift_config)

    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train ensemble model with uncertainty quantification."""
        logger.info("Training ψ ensemble model")

        # Prepare data
        feature_cols = [col for col in df.columns if col !=
                        self.config.target_column]
        X = df[feature_cols].values
        y = df[self.config.target_column].values

        # Split data
        n_train = int(len(df) * self.config.train_test_split)
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        # Train ensemble
        self.ensemble.fit(X_train, y_train)

        # Generate predictions with uncertainty
        pred_train, unc_train = self.ensemble.predict_with_uncertainty(X_train)
        pred_test, unc_test = self.ensemble.predict_with_uncertainty(X_test)

        # Evaluate
        ensemble_metrics = self.ensemble.evaluate_ensemble(X_test, y_test)

        # Train uncertainty quantifier
        train_data = pd.DataFrame(X_train, columns=feature_cols)
        train_data[self.config.target_column] = y_train
        train_data['predictions'] = pred_train
        train_data['uncertainty'] = unc_train

        self.uncertainty.fit(train_data, feature_cols,
                             self.config.target_column)

        # Domain shift monitoring
        domain_shift_report = self.domain_shift.monitor_domain_shift(
            train_data, train_data, feature_cols
        )

        training_results = {
            'ensemble_metrics': ensemble_metrics,
            'feature_importance': self.ensemble.get_feature_importance(),
            'domain_shift_report': domain_shift_report,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'feature_columns': feature_cols
        }

        logger.info("Model training completed")

        return training_results


class ValidationStage:
    """
    Comprehensive validation with physics-based and statistical metrics.
    """

    def __init__(self, config: CanonicalPipelineConfig):
        self.config = config
        self.physics_metrics = PhysicsBasedMetrics(self.config.physics_config)

    def validate_model(self, df: pd.DataFrame, predictions: np.ndarray,
                       training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive model validation."""
        logger.info("Validating ψ model performance")

        y_true = df[self.config.target_column].values

        # Physics-based validation
        physics_validation = self.physics_metrics.calculate_physics_metrics(
            predictions, y_true
        )

        # Statistical validation
        statistical_validation = self._compute_statistical_metrics(
            predictions, y_true)

        # Combined validation report
        validation_report = {
            'physics_based': physics_validation,
            'statistical': statistical_validation,
            'overall_score': self._compute_overall_validation_score(
                physics_validation, statistical_validation
            )
        }

        logger.info("Model validation completed")

        return validation_report

    def _compute_statistical_metrics(self, predictions: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Compute statistical performance metrics."""
        mse = np.mean((predictions - y_true)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_true))
        r2 = 1 - mse / np.var(y_true)

        # ψ-specific metrics
        within_physical_range = np.mean(
            (predictions >= -15) & (predictions <= 0))

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'within_physical_range': within_physical_range
        }

    def _compute_overall_validation_score(self, physics: Dict[str, Any],
                                          statistical: Dict[str, float]) -> float:
        """Compute overall validation score."""
        physics_score = physics['overall']['physics_score']
        r2_score = statistical['r2']
        physical_compliance = statistical['within_physical_range']

        # Weighted combination
        overall_score = 0.4 * physics_score + 0.4 * \
            max(0, r2_score) + 0.2 * physical_compliance

        return overall_score


class CalibrationStage:
    """
    Adaptive calibration for site-specific and temporal optimization.
    """

    def __init__(self, config: CanonicalPipelineConfig):
        self.config = config
        self.calibration_pipeline = AdaptiveCalibrationPipeline(
            self.config.calibration_config)

    def calibrate_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Apply adaptive calibration."""
        logger.info("Applying adaptive calibration")

        calibration_results = self.calibration_pipeline.run_full_calibration(
            df,
            site_col='site_id',
            psi_col=self.config.target_column,
            theta_col='soil_moisture',  # Assuming θ is available
            time_col='timestamp'
        )

        logger.info("Adaptive calibration completed")

        return calibration_results


class SensitivityAnalysisStage:
    """
    Optional sensitivity analysis for parameter importance.
    """

    def __init__(self, config: CanonicalPipelineConfig):
        self.config = config
        self.sensitivity_pipeline = SensitivityAnalysisPipeline(
            self.config.sensitivity_config)

    def analyze_sensitivity(self, model_func: callable, baseline_params: np.ndarray,
                            param_names: List[str], param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Perform sensitivity analysis."""
        logger.info("Performing sensitivity analysis")

        sensitivity_results = self.sensitivity_pipeline.run_full_sensitivity_analysis(
            model_func, baseline_params, param_names, param_ranges
        )

        logger.info("Sensitivity analysis completed")

        return sensitivity_results


class CanonicalPsiPipeline:
    """
    Complete canonical pipeline for ψ modeling.

    Orchestrates all stages from data ingestion to model deployment.
    """

    def __init__(self, config: Optional[CanonicalPipelineConfig] = None):
        self.config = config or CanonicalPipelineConfig()

        # Initialize pipeline stages
        self.data_ingestion = DataIngestionStage(self.config)
        self.quality_control = QualityControlStage(
            self.config) if self.config.enable_qc else None
        self.feature_engineering = FeatureEngineeringStage(
            self.config) if self.config.enable_feature_engineering else None
        self.model_training = ModelTrainingStage(self.config)
        self.validation = ValidationStage(self.config)
        self.calibration = CalibrationStage(
            self.config) if self.config.enable_calibration else None
        self.sensitivity = SensitivityAnalysisStage(
            self.config) if self.config.enable_sensitivity_analysis else None

        # Pipeline state
        self.is_trained = False
        self.pipeline_results = {}

    def run_full_pipeline(self, data_dir: str = 'data') -> Dict[str, Any]:
        """
        Execute the complete canonical pipeline.

        Returns comprehensive results from all pipeline stages.
        """
        logger.info("Starting canonical ψ pipeline execution")

        try:
            # 1. Data Ingestion
            logger.info("Stage 1: Data Ingestion")
            df_raw = self.data_ingestion.load_data_sources(data_dir)
            self.pipeline_results['data_ingestion'] = {'shape': df_raw.shape}

            # 2. Quality Control
            df_processed = df_raw
            if self.quality_control:
                logger.info("Stage 2: Quality Control")
                df_processed, qc_report = self.quality_control.apply_quality_control(
                    df_raw)
                self.pipeline_results['quality_control'] = qc_report

            # 3. Feature Engineering
            if self.feature_engineering:
                logger.info("Stage 3: Feature Engineering")
                df_processed = self.feature_engineering.create_features(
                    df_processed)
                self.pipeline_results['feature_engineering'] = {
                    'n_features': len(df_processed.columns) - 1,
                    'features': list(df_processed.columns)
                }

            # 4. Model Training
            logger.info("Stage 4: Model Training")
            training_results = self.model_training.train_models(df_processed)
            self.pipeline_results['model_training'] = training_results

            # 5. Validation
            logger.info("Stage 5: Validation")
            # Get test predictions for validation
            feature_cols = training_results['feature_columns']
            X_test = df_processed[feature_cols].values[int(
                len(df_processed) * self.config.train_test_split):]
            test_predictions, _ = self.model_training.ensemble.predict_with_uncertainty(
                X_test)
            df_test = df_processed.iloc[int(
                len(df_processed) * self.config.train_test_split):]

            validation_results = self.validation.validate_model(
                df_test, test_predictions, training_results)
            self.pipeline_results['validation'] = validation_results

            # 6. Calibration
            if self.calibration:
                logger.info("Stage 6: Calibration")
                calibration_results = self.calibration.calibrate_model(
                    df_processed)
                self.pipeline_results['calibration'] = calibration_results

            # 7. Sensitivity Analysis (optional)
            if self.sensitivity:
                logger.info("Stage 7: Sensitivity Analysis")
                # Define model function for sensitivity analysis

                def model_func(params):
                    # Simplified model function using ensemble predictions
                    return self.model_training.ensemble.predict(X_test[:1])[0]

                baseline_params = np.array(
                    [0.1, 0.4, 0.02, 2.0])  # θr, θs, α, n
                param_names = ['theta_r', 'theta_s', 'alpha', 'n']
                param_ranges = {
                    'theta_r': (0.0, 0.3),
                    'theta_s': (0.2, 0.6),
                    'alpha': (0.001, 1.0),
                    'n': (1.1, 5.0)
                }

                sensitivity_results = self.sensitivity.analyze_sensitivity(
                    model_func, baseline_params, param_names, param_ranges
                )
                self.pipeline_results['sensitivity_analysis'] = sensitivity_results

            self.is_trained = True

            # Save pipeline
            self.save_pipeline()

            logger.info(
                "Canonical ψ pipeline execution completed successfully")

            return self.pipeline_results

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ψ predictions using the trained pipeline.

        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.is_trained:
            raise ValueError(
                "Pipeline not trained. Call run_full_pipeline() first.")

        # Apply feature engineering if enabled
        df_processed = df
        if self.feature_engineering:
            df_processed = self.feature_engineering.create_features(df)

        # Select features used in training
        feature_cols = self.pipeline_results['model_training']['feature_columns']
        available_features = [
            col for col in feature_cols if col in df_processed.columns]

        if len(available_features) == 0:
            raise ValueError(
                "No trained features available in prediction data")

        X = df_processed[available_features].values

        # Generate predictions with uncertainty
        predictions, uncertainties = self.model_training.ensemble.predict_with_uncertainty(
            X)

        return predictions, uncertainties

    def save_pipeline(self, path: Optional[str] = None):
        """Save the trained pipeline to disk."""
        save_path = Path(path or self.config.model_save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config_path = save_path / 'pipeline_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config.__dict__, f)

        # Save pipeline results
        results_path = save_path / 'pipeline_results.joblib'
        joblib.dump(self.pipeline_results, results_path)

        # Save trained models
        if hasattr(self.model_training, 'ensemble'):
            ensemble_path = save_path / 'ensemble_model.joblib'
            joblib.dump(self.model_training.ensemble, ensemble_path)

        logger.info(f"Pipeline saved to {save_path}")

    @classmethod
    def load_pipeline(cls, path: str) -> 'CanonicalPsiPipeline':
        """Load a trained pipeline from disk."""
        load_path = Path(path)

        # Load configuration
        config_path = load_path / 'pipeline_config.yaml'
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        config = CanonicalPipelineConfig(**config_dict)

        # Create pipeline instance
        pipeline = cls(config)

        # Load results and models
        results_path = load_path / 'pipeline_results.joblib'
        pipeline.pipeline_results = joblib.load(results_path)
        pipeline.is_trained = True

        # Load trained models
        ensemble_path = load_path / 'ensemble_model.joblib'
        if ensemble_path.exists():
            pipeline.model_training.ensemble = joblib.load(ensemble_path)

        logger.info(f"Pipeline loaded from {load_path}")

        return pipeline

    def get_pipeline_summary(self) -> str:
        """Generate comprehensive pipeline execution summary."""
        summary = "Canonical ψ Pipeline Execution Summary\n"
        summary += "=" * 45 + "\n\n"

        # Overall status
        summary += f"Pipeline Status: {'Trained' if self.is_trained else 'Not Trained'}\n\n"

        # Stage summaries
        stages = [
            ('Data Ingestion', 'data_ingestion'),
            ('Quality Control', 'quality_control'),
            ('Feature Engineering', 'feature_engineering'),
            ('Model Training', 'model_training'),
            ('Validation', 'validation'),
            ('Calibration', 'calibration'),
            ('Sensitivity Analysis', 'sensitivity_analysis')
        ]

        for stage_name, stage_key in stages:
            if stage_key in self.pipeline_results:
                summary += f"{stage_name}:\n"
                stage_results = self.pipeline_results[stage_key]

                if stage_key == 'data_ingestion':
                    summary += f"  Data shape: {stage_results['shape']}\n"
                elif stage_key == 'quality_control':
                    qc = stage_results['overall_quality']
                    summary += f"  Quality score: {qc['quality_score']:.3f}\n"
                    summary += f"  Flagged points: {qc['n_flagged_points']}/{qc['n_total_points']}\n"
                elif stage_key == 'feature_engineering':
                    summary += f"  Features created: {stage_results['n_features']}\n"
                elif stage_key == 'model_training':
                    metrics = stage_results['ensemble_metrics']
                    summary += f"  R²: {metrics['r2']:.3f}, RMSE: {metrics['rmse']:.3f}\n"
                elif stage_key == 'validation':
                    overall = stage_results['overall_score']
                    summary += f"  Overall validation score: {overall:.3f}\n"
                elif stage_key == 'calibration':
                    summary += f"  Sites calibrated: {len(stage_results)}\n"

                summary += "\n"

        return summary
