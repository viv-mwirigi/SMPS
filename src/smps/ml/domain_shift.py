"""
Domain Shift Detection and Adaptation for Matric Potential Prediction.

Addresses critical deployment issue for ψ (matric potential) modeling:
- Training on ISMN stations, deployment on unknown farms
- Distributions differ in soil, management, irrigation, crops, sensors
- No domain adaptation, covariate shift detection, or OOD flagging for ψ

This module provides:
- Covariate shift detection using statistical tests for ψ features
- Out-of-distribution detection using isolation methods for ψ predictions
- Domain adaptation through feature normalization for ψ modeling
- Uncertainty quantification for domain shift in ψ space
- Confidence scores for deployment reliability of ψ models

Key Components:
- Statistical shift detection (KS-test, JS-divergence) for ψ-related features
- Isolation-based OOD detection for ψ predictions
- Feature distribution monitoring for ψ modeling
- Domain-invariant feature engineering for ψ
- Shift-aware uncertainty calibration for ψ
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

logger = logging.getLogger("swpps.ml.domain_shift")


@dataclass
class DomainShiftConfig:
    """Configuration for domain shift detection and adaptation."""

    # Detection methods
    enable_covariate_shift_detection: bool = True
    enable_ood_detection: bool = True

    # Statistical tests for ψ features
    statistical_tests: List[str] = field(
        default_factory=lambda: ['ks', 'js_divergence']
    )

    # OOD detection
    ood_methods: List[str] = field(
        default_factory=lambda: ['isolation_forest', 'one_class_svm']
    )

    # Thresholds for ψ domain shift
    ks_p_value_threshold: float = 0.05  # Below this = significant shift
    js_divergence_threshold: float = 0.1  # Above this = significant shift
    ood_contamination: float = 0.1  # Expected proportion of outliers

    # Adaptation
    enable_feature_normalization: bool = True
    normalization_method: str = "robust"  # standard, robust, quantile

    # Monitoring
    enable_drift_monitoring: bool = True
    drift_window_size: int = 100  # Samples for drift detection


@dataclass
class PsiDomainShiftResult:
    """Domain shift detection result for ψ predictions."""

    # Overall assessment
    has_significant_shift: bool = False
    shift_confidence: float = 0.0  # 0-1, higher = more confident in shift detection
    shift_severity: str = "none"  # none, low, medium, high

    # Statistical tests results
    ks_test_p_values: Dict[str, float] = field(default_factory=dict)
    js_divergences: Dict[str, float] = field(default_factory=dict)

    # OOD detection results
    ood_scores: Dict[str, float] = field(
        default_factory=dict)  # Higher = more OOD
    is_outlier: bool = False

    # Feature-level shifts (for ψ-relevant features)
    shifted_features: List[str] = field(default_factory=list)

    # Recommendations
    adaptation_needed: bool = False
    recommended_actions: List[str] = field(default_factory=list)

    # Uncertainty adjustment
    uncertainty_multiplier: float = 1.0  # Multiply prediction uncertainty by this


class PsiCovariateShiftDetector:
    """
    Detect covariate shift in ψ (matric potential) modeling features.

    Monitors distribution changes in features that affect ψ predictions:
    - Weather patterns (precipitation, ET0, temperature)
    - Soil properties (texture, hydraulic parameters)
    - Physics priors (ψ predictions from water balance model)
    """

    def __init__(self, config: DomainShiftConfig):
        self.config = config
        self.reference_distributions: Dict[str, np.ndarray] = {}
        self.feature_scalers: Dict[str, Any] = {}
        self.is_fitted = False

    def fit(self, X_reference: pd.DataFrame, feature_columns: Optional[List[str]] = None):
        """Fit reference distributions for ψ features."""
        logger.info(
            "Fitting reference distributions for ψ domain shift detection")

        if feature_columns is None:
            # Default ψ-relevant features
            feature_columns = [
                'precipitation_mm', 'et0_mm', 'temperature_2m', 'relative_humidity_2m',
                'physics_prior_surface', 'physics_prior_root', 'physics_prior_deep',
                'clay_pct', 'sand_pct', 'bulk_density'
            ]
            # Filter to available columns
            feature_columns = [
                col for col in feature_columns if col in X_reference.columns]

        for feature in feature_columns:
            values = X_reference[feature].dropna().values

            if len(values) < 10:
                logger.warning(f"Insufficient data for {feature}, skipping")
                continue

            # Store reference distribution
            self.reference_distributions[feature] = values.copy()

            # Fit scaler for normalization
            if self.config.enable_feature_normalization:
                if self.config.normalization_method == "robust":
                    scaler = RobustScaler()
                elif self.config.normalization_method == "quantile":
                    # Could implement quantile normalization
                    scaler = StandardScaler()
                else:
                    scaler = StandardScaler()

                scaler.fit(values.reshape(-1, 1))
                self.feature_scalers[feature] = scaler

        self.is_fitted = True
        logger.info(
            f"Fitted domain shift detection for {len(self.reference_distributions)} ψ features")

    def detect_shift(self, X_test: pd.DataFrame) -> PsiDomainShiftResult:
        """Detect domain shift in test data for ψ predictions."""
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")

        result = PsiDomainShiftResult()

        # Statistical tests
        if self.config.enable_covariate_shift_detection:
            ks_p_values = {}
            js_divergences = {}

            for feature, ref_values in self.reference_distributions.items():
                if feature not in X_test.columns:
                    continue

                test_values = X_test[feature].dropna().values

                if len(test_values) < 5:
                    continue

                # KS test
                try:
                    ks_stat, ks_p = stats.ks_2samp(ref_values, test_values)
                    ks_p_values[feature] = ks_p
                except Exception as e:
                    logger.warning(f"KS test failed for {feature}: {e}")
                    ks_p_values[feature] = 1.0  # No significant difference

                # JS divergence
                try:
                    # Create histograms
                    hist_ref, bins = np.histogram(
                        ref_values, bins=20, density=True)
                    hist_test, _ = np.histogram(
                        test_values, bins=bins, density=True)

                    # Add small epsilon to avoid log(0)
                    hist_ref = hist_ref + 1e-10
                    hist_test = hist_test + 1e-10

                    js_div = jensenshannon(hist_ref, hist_test)
                    js_divergences[feature] = js_div
                except Exception as e:
                    logger.warning(f"JS divergence failed for {feature}: {e}")
                    js_divergences[feature] = 0.0

            result.ks_test_p_values = ks_p_values
            result.js_divergences = js_divergences

            # Identify shifted features
            shifted_features = []
            for feature in ks_p_values:
                if ks_p_values[feature] < self.config.ks_p_value_threshold:
                    shifted_features.append(feature)

            for feature in js_divergences:
                if js_divergences[feature] > self.config.js_divergence_threshold:
                    if feature not in shifted_features:
                        shifted_features.append(feature)

            result.shifted_features = shifted_features

        # Overall shift assessment
        n_significant_shifts = len(result.shifted_features)
        total_features = len(self.reference_distributions)

        if total_features > 0:
            shift_ratio = n_significant_shifts / total_features

            if shift_ratio > 0.5:
                result.shift_severity = "high"
                result.has_significant_shift = True
                result.shift_confidence = min(1.0, shift_ratio)
                result.adaptation_needed = True
                result.recommended_actions = [
                    "High domain shift detected - consider retraining model",
                    "Use uncertainty inflation for predictions",
                    "Collect more data from this domain"
                ]
                result.uncertainty_multiplier = 2.0

            elif shift_ratio > 0.2:
                result.shift_severity = "medium"
                result.has_significant_shift = True
                result.shift_confidence = shift_ratio
                result.adaptation_needed = True
                result.recommended_actions = [
                    "Moderate domain shift - use caution with predictions",
                    "Consider feature normalization",
                    "Monitor prediction performance"
                ]
                result.uncertainty_multiplier = 1.5

            elif shift_ratio > 0.1:
                result.shift_severity = "low"
                result.has_significant_shift = False
                result.shift_confidence = shift_ratio
                result.recommended_actions = [
                    "Minor shift detected - predictions should be reliable",
                    "Optional: apply feature normalization"
                ]
                result.uncertainty_multiplier = 1.2

        return result


class PsiOODDetector:
    """
    Out-of-distribution detection for ψ predictions.

    Uses unsupervised methods to detect when ψ predictions are unreliable
    due to operating in unseen regions of feature space.
    """

    def __init__(self, config: DomainShiftConfig):
        self.config = config
        self.ood_models: Dict[str, Any] = {}
        self.is_fitted = False

    def fit(self, X_reference: pd.DataFrame):
        """Fit OOD detection models on reference ψ data."""
        logger.info("Fitting OOD detection models for ψ predictions")

        # Isolation Forest
        if 'isolation_forest' in self.config.ood_methods:
            iso_forest = IsolationForest(
                contamination=self.config.ood_contamination,
                random_state=42,
                n_estimators=100
            )
            iso_forest.fit(X_reference)
            self.ood_models['isolation_forest'] = iso_forest

        # One-Class SVM
        if 'one_class_svm' in self.config.ood_methods:
            oc_svm = OneClassSVM(
                nu=self.config.ood_contamination,
                kernel='rbf',
                gamma='scale'
            )
            oc_svm.fit(X_reference)
            self.ood_models['one_class_svm'] = oc_svm

        self.is_fitted = True
        logger.info(
            f"Fitted {len(self.ood_models)} OOD detection models for ψ")

    def detect_ood(self, X_test: pd.DataFrame) -> Dict[str, float]:
        """Detect OOD samples in test data."""
        if not self.is_fitted:
            raise ValueError("OOD detector not fitted. Call fit() first.")

        ood_scores = {}

        # Isolation Forest score (negative = more anomalous)
        if 'isolation_forest' in self.ood_models:
            iso_scores = self.ood_models['isolation_forest'].decision_function(
                X_test)
            # Convert to 0-1 scale (higher = more OOD)
            ood_scores['isolation_forest'] = 1 / (1 + np.exp(iso_scores))

        # One-Class SVM score
        if 'one_class_svm' in self.ood_models:
            oc_scores = self.ood_models['one_class_svm'].decision_function(
                X_test)
            # Convert to 0-1 scale
            ood_scores['one_class_svm'] = 1 / (1 + np.exp(-oc_scores))

        return ood_scores


class PsiDomainShiftMonitor:
    """
    Comprehensive domain shift monitoring for ψ predictions.

    Combines covariate shift detection and OOD detection for robust
    deployment monitoring of ψ models.
    """

    def __init__(self, config: Optional[DomainShiftConfig] = None):
        self.config = config or DomainShiftConfig()
        self.covariate_detector: Optional[PsiCovariateShiftDetector] = None
        self.ood_detector: Optional[PsiOODDetector] = None
        self.is_fitted = False

    def fit(self, X_reference: pd.DataFrame, y_reference: Optional[np.ndarray] = None):
        """Fit domain shift monitoring on reference ψ data."""
        logger.info("Fitting comprehensive domain shift monitoring for ψ")

        # Fit covariate shift detector
        if self.config.enable_covariate_shift_detection:
            self.covariate_detector = PsiCovariateShiftDetector(self.config)
            self.covariate_detector.fit(X_reference)

        # Fit OOD detector
        if self.config.enable_ood_detection:
            self.ood_detector = PsiOODDetector(self.config)
            self.ood_detector.fit(X_reference)

        self.is_fitted = True
        logger.info("Domain shift monitoring fitted for ψ predictions")

    def monitor_sample(self, X_sample: pd.DataFrame) -> PsiDomainShiftResult:
        """Monitor a single sample or batch for domain shift."""
        if not self.is_fitted:
            raise ValueError("Monitor not fitted. Call fit() first.")

        result = PsiDomainShiftResult()

        # Covariate shift detection
        if self.covariate_detector:
            shift_result = self.covariate_detector.detect_shift(X_sample)
            result.has_significant_shift = shift_result.has_significant_shift
            result.shift_confidence = shift_result.shift_confidence
            result.shift_severity = shift_result.shift_severity
            result.ks_test_p_values = shift_result.ks_test_p_values
            result.js_divergences = shift_result.js_divergences
            result.shifted_features = shift_result.shifted_features
            result.adaptation_needed = shift_result.adaptation_needed
            result.recommended_actions = shift_result.recommended_actions
            result.uncertainty_multiplier = shift_result.uncertainty_multiplier

        # OOD detection
        if self.ood_detector:
            ood_scores = self.ood_detector.detect_ood(X_sample)
            result.ood_scores = ood_scores

            # Determine if outlier based on ensemble of OOD methods
            if ood_scores:
                mean_ood_score = np.mean(list(ood_scores.values()))
                result.is_outlier = mean_ood_score > 0.7  # Threshold for outlier

                if result.is_outlier:
                    result.recommended_actions.append(
                        "Sample detected as outlier - high uncertainty")
                    result.uncertainty_multiplier *= 1.5

        return result

    def get_monitoring_stats(self, X_test: pd.DataFrame) -> Dict[str, float]:
        """Get monitoring statistics for a test dataset."""
        results = [self.monitor_sample(X_test.iloc[i:i+1])
                   for i in range(len(X_test))]

        return {
            'shift_detected_pct': np.mean([r.has_significant_shift for r in results]) * 100,
            'outlier_detected_pct': np.mean([r.is_outlier for r in results]) * 100,
            'high_shift_pct': np.mean([r.shift_severity == 'high' for r in results]) * 100,
            'mean_uncertainty_multiplier': np.mean([r.uncertainty_multiplier for r in results]),
            'mean_shift_confidence': np.mean([r.shift_confidence for r in results]),
        }
