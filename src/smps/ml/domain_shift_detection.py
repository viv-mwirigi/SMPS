"""
Domain Shift Detection and Adaptation for Soil Moisture Prediction.

Addresses critical deployment issue:
- Training on ISMN stations, deployment on unknown farms
- Distributions differ in soil, management, irrigation, crops, sensors
- No domain adaptation, covariate shift detection, or OOD flagging

This module provides:
- Covariate shift detection using statistical tests
- Out-of-distribution detection using isolation methods
- Domain adaptation through feature normalization
- Uncertainty quantification for domain shift
- Confidence scores for deployment reliability

Key Components:
- Statistical shift detection (KS-test, JS-divergence)
- Isolation-based OOD detection
- Feature distribution monitoring
- Domain-invariant feature engineering
- Shift-aware uncertainty calibration
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)


@dataclass
class DomainShiftConfig:
    """Configuration for domain shift detection and adaptation."""

    # Detection methods
    enable_covariate_shift_detection: bool = True
    enable_ood_detection: bool = True

    # Statistical tests
    ks_test_threshold: float = 0.05  # p-value threshold for rejecting null hypothesis
    js_divergence_threshold: float = 0.1  # Jensen-Shannon divergence threshold

    # OOD detection
    # 'isolation_forest', 'one_class_svm', 'mahalanobis'
    ood_method: str = "isolation_forest"
    ood_contamination: float = 0.1  # Expected proportion of outliers
    ood_threshold: float = -0.5  # Decision threshold for isolation forest

    # Domain adaptation
    enable_domain_adaptation: bool = True
    # 'feature_scaling', 'domain_mapping'
    adaptation_method: str = "feature_scaling"

    # Uncertainty calibration
    # Multiply uncertainty by this when shift detected
    shift_uncertainty_multiplier: float = 2.0
    # Multiply uncertainty by this for OOD samples
    ood_uncertainty_multiplier: float = 3.0

    # Monitoring
    enable_distribution_monitoring: bool = True
    monitoring_window_size: int = 1000  # Samples to keep for distribution monitoring


@dataclass
class ShiftDetectionResult:
    """Results from domain shift detection."""

    # Overall shift indicators
    is_shift_detected: bool = False
    shift_confidence: float = 0.0  # 0-1, higher = more confident shift detected

    # Covariate shift results
    covariate_shift_detected: bool = False
    ks_test_p_values: Dict[str, float] = field(default_factory=dict)
    js_divergences: Dict[str, float] = field(default_factory=dict)

    # OOD detection results
    is_ood: bool = False
    ood_score: float = 0.0  # Higher = more likely OOD
    ood_confidence: float = 0.0

    # Feature-level shift indicators
    shifted_features: List[str] = field(default_factory=list)
    feature_shift_magnitudes: Dict[str, float] = field(default_factory=dict)

    # Recommendations
    adaptation_needed: bool = False
    uncertainty_multiplier: float = 1.0


class CovariateShiftDetector:
    """
    Detects covariate shift between training and deployment distributions.

    Uses statistical tests to identify when input distributions have changed.
    """

    def __init__(self, config: DomainShiftConfig):
        self.config = config
        self.training_distributions = {}
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit detector on training data.

        Args:
            X_train: Training features
            feature_names: Optional feature names for interpretable results
        """
        logger.info("Fitting covariate shift detector on training data...")

        X_train = np.asarray(X_train)
        n_features = X_train.shape[1]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        elif len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length {len(feature_names)} != n_features {n_features}")

        # Store training distributions for each feature
        self.training_distributions = {}
        for i, feature_name in enumerate(feature_names):
            feature_data = X_train[:, i]

            # Remove NaN values for distribution fitting
            clean_data = feature_data[~np.isnan(feature_data)]
            if len(clean_data) == 0:
                logger.warning(f"No valid data for feature {feature_name}")
                continue

            # Store distribution parameters
            self.training_distributions[feature_name] = {
                'mean': np.mean(clean_data),
                'std': np.std(clean_data),
                'median': np.median(clean_data),
                'q25': np.percentile(clean_data, 25),
                'q75': np.percentile(clean_data, 75),
                'min': np.min(clean_data),
                'max': np.max(clean_data),
                'data': clean_data.copy()  # Keep sample for KS test
            }

        self.is_fitted = True
        logger.info(
            f"Covariate shift detector fitted on {len(self.training_distributions)} features")

    def detect_shift(self, X_test: np.ndarray, feature_names: Optional[List[str]] = None) -> ShiftDetectionResult:
        """
        Detect covariate shift in test data.

        Args:
            X_test: Test/deployment features
            feature_names: Optional feature names

        Returns:
            ShiftDetectionResult with detailed shift analysis
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detecting shift")

        X_test = np.asarray(X_test)
        n_features = X_test.shape[1]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        result = ShiftDetectionResult()

        # Feature-level shift detection
        ks_p_values = {}
        js_divergences = {}
        shifted_features = []
        feature_shift_magnitudes = {}

        for i, feature_name in enumerate(feature_names):
            if feature_name not in self.training_distributions:
                continue

            train_dist = self.training_distributions[feature_name]
            test_data = X_test[:, i]

            # Remove NaN values
            test_clean = test_data[~np.isnan(test_data)]
            if len(test_clean) == 0:
                continue

            # Kolmogorov-Smirnov test
            try:
                ks_stat, ks_p = stats.ks_2samp(train_dist['data'], test_clean)
                ks_p_values[feature_name] = ks_p
            except Exception as e:
                logger.warning(f"KS test failed for {feature_name}: {e}")
                ks_p_values[feature_name] = 1.0  # No shift detected

            # Jensen-Shannon divergence
            try:
                # Create histograms for JS divergence
                train_hist, bin_edges = np.histogram(
                    train_dist['data'], bins=50, density=True)
                test_hist, _ = np.histogram(
                    test_clean, bins=bin_edges, density=True)

                # Ensure same length and normalize
                min_len = min(len(train_hist), len(test_hist))
                train_hist = train_hist[:min_len] / \
                    np.sum(train_hist[:min_len])
                test_hist = test_hist[:min_len] / np.sum(test_hist[:min_len])

                js_div = jensenshannon(train_hist, test_hist)
                js_divergences[feature_name] = js_div
            except Exception as e:
                logger.warning(f"JS divergence failed for {feature_name}: {e}")
                js_divergences[feature_name] = 0.0

            # Determine if feature has shifted
            ks_significant = ks_p_values[feature_name] < self.config.ks_test_threshold
            js_significant = js_divergences[feature_name] > self.config.js_divergence_threshold

            if ks_significant or js_significant:
                shifted_features.append(feature_name)
                # Magnitude as average of normalized KS stat and JS divergence
                magnitude = (1 - ks_p_values[feature_name]) * \
                    0.5 + js_divergences[feature_name] * 0.5
                feature_shift_magnitudes[feature_name] = magnitude

        # Overall shift assessment
        result.ks_test_p_values = ks_p_values
        result.js_divergences = js_divergences
        result.shifted_features = shifted_features
        result.feature_shift_magnitudes = feature_shift_magnitudes

        # Determine overall shift
        n_shifted_features = len(shifted_features)
        total_features = len(
            [f for f in feature_names if f in self.training_distributions])

        if total_features > 0:
            shift_ratio = n_shifted_features / total_features
            result.covariate_shift_detected = shift_ratio > 0.3  # >30% features shifted
            result.shift_confidence = shift_ratio
            result.is_shift_detected = result.covariate_shift_detected

        return result


class OutOfDistributionDetector:
    """
    Detects out-of-distribution samples using unsupervised methods.
    """

    def __init__(self, config: DomainShiftConfig):
        self.config = config
        self.detector = None
        self.scaler = None
        self.is_fitted = False

    def fit(self, X_train: np.ndarray):
        """Fit OOD detector on training data."""
        logger.info(f"Fitting OOD detector using {self.config.ood_method}...")

        X_train = np.asarray(X_train)

        # Remove rows with NaN values
        valid_mask = ~np.any(np.isnan(X_train), axis=1)
        X_train_clean = X_train[valid_mask]

        if len(X_train_clean) == 0:
            raise ValueError("No valid training data for OOD detection")

        # Feature scaling for distance-based methods
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_clean)

        # Initialize detector based on method
        if self.config.ood_method == "isolation_forest":
            self.detector = IsolationForest(
                contamination=self.config.ood_contamination,
                random_state=42,
                n_estimators=100
            )
        elif self.config.ood_method == "one_class_svm":
            self.detector = OneClassSVM(
                nu=self.config.ood_contamination,
                kernel='rbf',
                gamma='scale'
            )
        elif self.config.ood_method == "mahalanobis":
            # For Mahalanobis, we just need the training distribution
            self.detector = {
                'mean': np.mean(X_train_scaled, axis=0),
                'cov': np.cov(X_train_scaled.T)
            }
        else:
            raise ValueError(f"Unknown OOD method: {self.config.ood_method}")

        # Fit detector (except for Mahalanobis which is just statistics)
        if self.config.ood_method != "mahalanobis":
            self.detector.fit(X_train_scaled)

        self.is_fitted = True
        logger.info(f"OOD detector fitted on {len(X_train_clean)} samples")

    def detect_ood(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect OOD samples.

        Returns:
            Tuple of (is_ood_array, ood_score_array)
            - is_ood: boolean array, True for OOD samples
            - ood_score: continuous score, higher = more likely OOD
        """
        if not self.is_fitted:
            raise ValueError("OOD detector must be fitted before detection")

        X_test = np.asarray(X_test)

        # Handle NaN values
        nan_mask = np.any(np.isnan(X_test), axis=1)
        X_test_clean = X_test.copy()
        X_test_clean[nan_mask] = 0  # Temporary fill for scaling

        # Scale features
        X_test_scaled = self.scaler.transform(X_test_clean)

        # Initialize outputs to safe defaults so they are always defined
        n_samples = X_test.shape[0]
        is_ood = np.zeros(n_samples, dtype=bool)
        ood_scores = np.zeros(n_samples, dtype=float)

        if self.config.ood_method == "isolation_forest":
            # Isolation forest returns -1 for outliers, 1 for inliers
            scores = self.detector.decision_function(X_test_scaled)
            is_ood = scores < self.config.ood_threshold
            ood_scores = -scores  # Convert to positive OOD scores

        elif self.config.ood_method == "one_class_svm":
            # One-class SVM returns -1 for outliers, 1 for inliers
            predictions = self.detector.predict(X_test_scaled)
            scores = self.detector.decision_function(X_test_scaled)
            is_ood = predictions == -1
            ood_scores = -scores  # Convert to positive OOD scores

        elif self.config.ood_method == "mahalanobis":
            # Mahalanobis distance
            diff = X_test_scaled - self.detector['mean']
            inv_cov = np.linalg.inv(self.detector['cov'])
            distances = np.sum(diff * (diff @ inv_cov), axis=1)
            ood_scores = distances
            # Threshold based on chi-squared distribution
            threshold = stats.chi2.ppf(0.95, X_test_scaled.shape[1])
            is_ood = distances > threshold

        # Set OOD for samples with NaN values
        is_ood = np.asarray(is_ood, dtype=bool)
        is_ood[nan_mask] = True
        # Provide a fallback value if ood_scores are all zero to avoid assigning zeros unintentionally
        max_score = np.max(ood_scores) if ood_scores.size > 0 else 0.0
        fallback = (max_score * 2) if max_score != 0 else 1.0
        ood_scores[nan_mask] = fallback

        return is_ood, ood_scores


class DomainShiftAwareModel:
    """
    Wrapper that adds domain shift detection and adaptation to any ML model.

    Integrates with uncertainty-aware hybrid models to provide:
    - Shift detection during inference
    - Uncertainty inflation for shifted domains
    - OOD flagging for unreliable predictions
    - Domain adaptation recommendations
    """

    def __init__(self, base_model: Any, config: DomainShiftConfig):
        self.base_model = base_model
        self.config = config

        # Initialize detectors
        self.shift_detector = CovariateShiftDetector(
            config) if config.enable_covariate_shift_detection else None
        self.ood_detector = OutOfDistributionDetector(
            config) if config.enable_ood_detection else None

        # Monitoring
        self.monitoring_buffer = []
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit the domain shift aware model.

        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: Optional feature names
        """
        logger.info("Fitting domain shift aware model...")

        # Fit base model
        self.base_model.fit(X_train, y_train)

        # Fit shift detectors
        if self.shift_detector:
            self.shift_detector.fit(X_train, feature_names)

        if self.ood_detector:
            self.ood_detector.fit(X_train)

        self.is_fitted = True
        logger.info("Domain shift aware model fitted")

    def predict_with_shift_detection(self, X_test: np.ndarray,
                                     feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Make predictions with domain shift detection and uncertainty adjustment.

        Returns:
            dict with:
            - 'predictions': Base model predictions
            - 'shift_results': ShiftDetectionResult
            - 'ood_results': OOD detection results
            - 'adjusted_uncertainty': Uncertainty adjusted for domain shift
            - 'reliability_score': Overall prediction reliability (0-1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_test = np.asarray(X_test)

        # Get base model predictions
        if hasattr(self.base_model, 'predict_with_uncertainty'):
            base_results = self.base_model.predict_with_uncertainty(X_test)
            predictions = base_results['prediction']
            base_uncertainty = base_results.get('total_uncertainty',
                                                base_results.get('aleatoric_uncertainty', np.full(len(predictions), 0.05)))
        else:
            predictions = self.base_model.predict(X_test)
            base_uncertainty = np.full(
                len(predictions), 0.05)  # Default uncertainty

        # Initialize results
        results = {
            'predictions': predictions,
            'base_uncertainty': base_uncertainty,
            'shift_results': None,
            'ood_results': None,
            'adjusted_uncertainty': base_uncertainty.copy(),
            # Start with full reliability
            'reliability_score': np.ones(len(predictions)),
            'shift_detected': False,
            'ood_detected': False
        }

        # Covariate shift detection
        if self.shift_detector:
            shift_result = self.shift_detector.detect_shift(
                X_test, feature_names)
            results['shift_results'] = shift_result

            if shift_result.is_shift_detected:
                results['shift_detected'] = True
                # Inflate uncertainty for shifted samples
                shift_multiplier = 1 + \
                    (shift_result.shift_confidence *
                     (self.config.shift_uncertainty_multiplier - 1))
                results['adjusted_uncertainty'] *= shift_multiplier
                results['reliability_score'] *= (1 -
                                                 shift_result.shift_confidence * 0.5)

        # OOD detection
        if self.ood_detector:
            is_ood, ood_scores = self.ood_detector.detect_ood(X_test)
            results['ood_results'] = {
                'is_ood': is_ood,
                'ood_scores': ood_scores
            }

            if np.any(is_ood):
                results['ood_detected'] = True
                # Inflate uncertainty for OOD samples
                ood_mask = is_ood
                results['adjusted_uncertainty'][ood_mask] *= self.config.ood_uncertainty_multiplier
                # Reduce reliability for OOD samples
                ood_penalty = np.clip(ood_scores / np.max(ood_scores), 0, 1)
                results['reliability_score'] *= (1 - ood_penalty * 0.7)

        # Update monitoring buffer
        if self.config.enable_distribution_monitoring:
            for i, x in enumerate(X_test):
                self.monitoring_buffer.append({
                    'features': x,
                    'prediction': predictions[i],
                    'uncertainty': results['adjusted_uncertainty'][i],
                    'reliability': results['reliability_score'][i],
                    'shift_detected': results['shift_detected'],
                    'ood_detected': results['ood_detected']
                })

            # Keep buffer size limited
            if len(self.monitoring_buffer) > self.config.monitoring_window_size:
                self.monitoring_buffer = self.monitoring_buffer[-self.config.monitoring_window_size:]

        return results

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get statistics from monitoring buffer."""
        if not self.monitoring_buffer:
            return {}

        buffer_df = pd.DataFrame(self.monitoring_buffer)

        stats = {
            'total_predictions': len(buffer_df),
            'shift_detected_ratio': buffer_df['shift_detected'].mean(),
            'ood_detected_ratio': buffer_df['ood_detected'].mean(),
            'mean_reliability': buffer_df['reliability'].mean(),
            'mean_uncertainty': buffer_df['uncertainty'].mean(),
            'reliability_distribution': {
                'q25': buffer_df['reliability'].quantile(0.25),
                'median': buffer_df['reliability'].median(),
                'q75': buffer_df['reliability'].quantile(0.75)
            }
        }

        return stats

    def save(self, path: Path):
        """Save the domain shift aware model."""
        save_dict = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'monitoring_buffer': self.monitoring_buffer
        }

        # Save base model if it has save method
        if hasattr(self.base_model, 'save'):
            base_path = path / 'base_model'
            base_path.mkdir(exist_ok=True)
            self.base_model.save(base_path)
            save_dict['base_model_path'] = str(base_path)
        else:
            save_dict['base_model'] = self.base_model

        # Save detectors
        if self.shift_detector:
            shift_path = path / 'shift_detector.pkl'
            joblib.dump(self.shift_detector, shift_path)
            save_dict['shift_detector_path'] = str(shift_path)

        if self.ood_detector:
            ood_path = path / 'ood_detector.pkl'
            joblib.dump(self.ood_detector, ood_path)
            save_dict['ood_detector_path'] = str(ood_path)

        # Save main config
        config_path = path / 'domain_shift_model.pkl'
        joblib.dump(save_dict, config_path)

    @classmethod
    def load(cls, path: Path) -> 'DomainShiftAwareModel':
        """Load a saved domain shift aware model."""
        config_path = path / 'domain_shift_model.pkl'
        save_dict = joblib.load(config_path)

        # Load base model
        if 'base_model_path' in save_dict:
            # Assume base model has load method
            base_model_path = Path(save_dict['base_model_path'])
            if hasattr(save_dict.get('base_model', None), 'load'):
                base_model = save_dict['base_model'].load(base_model_path)
            else:
                raise ValueError("Base model doesn't have load method")
        else:
            base_model = save_dict['base_model']

        # Create instance
        instance = cls(base_model, save_dict['config'])
        instance.is_fitted = save_dict['is_fitted']
        instance.monitoring_buffer = save_dict['monitoring_buffer']

        # Load detectors
        if 'shift_detector_path' in save_dict:
            instance.shift_detector = joblib.load(
                save_dict['shift_detector_path'])

        if 'ood_detector_path' in save_dict:
            instance.ood_detector = joblib.load(save_dict['ood_detector_path'])

        return instance
