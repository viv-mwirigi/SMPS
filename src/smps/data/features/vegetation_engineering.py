"""
Advanced Vegetation Feature Engineering for Soil Moisture Prediction.

Addresses critical NDVI limitations:
- NDVI saturation at high LAI (can't distinguish dense vegetation)
- Noise in cloudy conditions (unreliable observations)
- Confounding crops vs grass (no vegetation type distinction)
- Overloaded for multiple purposes (phenology, transpiration, growth stage)

This module provides:
- Multiple vegetation indices (EVI, SAVI, ARVI, etc.) for different purposes
- Cloud filtering and quality assessment
- Vegetation type classification (crops vs grass vs trees)
- Phenology-specific features beyond simple NDVI
- Transpiration proxies using vegetation structure
- LAI saturation handling with index combinations
- Temporal vegetation dynamics and trends

Key Innovations:
- Vegetation index ensemble for robust vegetation modeling
- Cloud-aware temporal interpolation
- Vegetation type-specific features
- Phenological stage detection
- Transpiration potential estimation
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import warnings

logger = logging.getLogger(__name__)


@dataclass
class VegetationConfig:
    """Configuration for advanced vegetation feature engineering."""

    # Index selection
    use_ndvi: bool = True
    # Enhanced Vegetation Index (better for dense vegetation)
    use_evi: bool = True
    # Soil-Adjusted Vegetation Index (better for sparse vegetation)
    use_savi: bool = True
    # Atmospherically Resistant Vegetation Index (better in haze)
    use_arvi: bool = True
    use_gndvi: bool = True  # Green NDVI (better for chlorophyll content)
    use_cvi: bool = True   # Chlorophyll Vegetation Index

    # Cloud and quality filtering
    enable_cloud_filtering: bool = True
    cloud_threshold_ndvi: float = 0.1  # NDVI values below this may indicate clouds
    cloud_threshold_variability: float = 0.3  # High variability indicates clouds
    quality_filter_enabled: bool = True

    # Vegetation type classification
    enable_vegetation_classification: bool = True
    n_vegetation_types: int = 4  # crops, grass, trees, mixed
    classification_method: str = "clustering"  # 'clustering', 'thresholds', 'ml'

    # Phenology features
    enable_phenology_features: bool = True
    phenology_smoothing_window: int = 15  # days for temporal smoothing
    growth_stage_detection: bool = True

    # Transpiration features
    enable_transpiration_features: bool = True
    # 'vegetation_structure', 'temperature_based'
    transpiration_method: str = "vegetation_structure"

    # LAI saturation handling
    handle_lai_saturation: bool = True
    saturation_threshold_ndvi: float = 0.8
    multi_index_fusion: bool = True

    # Temporal features
    enable_temporal_features: bool = True
    temporal_lags: List[int] = field(
        default_factory=lambda: [7, 14, 30])  # days
    trend_windows: List[int] = field(default_factory=lambda: [
                                     7, 30])  # days for trend calculation


@dataclass
class VegetationFeatures:
    """Container for computed vegetation features."""

    # Base vegetation indices
    ndvi: Optional[np.ndarray] = None
    evi: Optional[np.ndarray] = None
    savi: Optional[np.ndarray] = None
    arvi: Optional[np.ndarray] = None
    gndvi: Optional[np.ndarray] = None
    cvi: Optional[np.ndarray] = None

    # Quality and filtering
    cloud_mask: Optional[np.ndarray] = None
    quality_score: Optional[np.ndarray] = None
    valid_observations: Optional[np.ndarray] = None

    # Vegetation classification
    vegetation_type: Optional[np.ndarray] = None
    vegetation_type_confidence: Optional[np.ndarray] = None

    # Phenology features
    phenology_stage: Optional[np.ndarray] = None
    growing_season_progress: Optional[np.ndarray] = None
    senescence_index: Optional[np.ndarray] = None

    # Transpiration features
    transpiration_potential: Optional[np.ndarray] = None
    canopy_conductance_proxy: Optional[np.ndarray] = None

    # LAI and structure
    lai_proxy: Optional[np.ndarray] = None
    canopy_cover: Optional[np.ndarray] = None
    vegetation_height_proxy: Optional[np.ndarray] = None

    # Temporal features
    temporal_lags: Dict[str, np.ndarray] = field(default_factory=dict)
    trends: Dict[str, np.ndarray] = field(default_factory=dict)
    seasonality: Optional[np.ndarray] = None

    # Derived features
    vegetation_dynamics: Optional[np.ndarray] = None
    stress_indicators: Optional[np.ndarray] = None


class VegetationIndexCalculator:
    """
    Calculates multiple vegetation indices from spectral bands.

    Addresses NDVI limitations by providing specialized indices for different purposes.
    """

    def __init__(self, config: VegetationConfig):
        self.config = config

    def calculate_indices(self, spectral_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate vegetation indices from spectral bands.

        Args:
            spectral_data: DataFrame with spectral bands (red, nir, blue, green, swir1, swir2)

        Returns:
            Dict of vegetation indices
        """
        indices = {}

        # Extract bands (handle missing bands gracefully)
        red = spectral_data.get('red', spectral_data.get(
            'B4', spectral_data.get('red_band')))
        nir = spectral_data.get('nir', spectral_data.get(
            'B8', spectral_data.get('nir_band')))
        blue = spectral_data.get('blue', spectral_data.get(
            'B2', spectral_data.get('blue_band')))
        green = spectral_data.get('green', spectral_data.get(
            'B3', spectral_data.get('green_band')))
        swir1 = spectral_data.get('swir1', spectral_data.get(
            'B11', spectral_data.get('swir1_band')))
        swir2 = spectral_data.get('swir2', spectral_data.get(
            'B12', spectral_data.get('swir2_band')))

        # Convert to numpy arrays
        red = self._to_numpy(red)
        nir = self._to_numpy(nir)
        blue = self._to_numpy(blue)
        green = self._to_numpy(green)
        swir1 = self._to_numpy(swir1)
        swir2 = self._to_numpy(swir2)

        # NDVI - Normalized Difference Vegetation Index
        if self.config.use_ndvi and red is not None and nir is not None:
            indices['ndvi'] = self._calculate_ndvi(red, nir)

        # EVI - Enhanced Vegetation Index (better for dense vegetation, reduces saturation)
        if self.config.use_evi and red is not None and nir is not None and blue is not None:
            indices['evi'] = self._calculate_evi(red, nir, blue)

        # SAVI - Soil-Adjusted Vegetation Index (better for sparse vegetation)
        if self.config.use_savi and red is not None and nir is not None:
            indices['savi'] = self._calculate_savi(red, nir)

        # ARVI - Atmospherically Resistant Vegetation Index (better in haze/aerosols)
        if self.config.use_arvi and red is not None and nir is not None and blue is not None:
            indices['arvi'] = self._calculate_arvi(red, nir, blue)

        # GNDVI - Green NDVI (better for chlorophyll content)
        if self.config.use_gndvi and green is not None and nir is not None:
            indices['gndvi'] = self._calculate_gndvi(green, nir)

        # CVI - Chlorophyll Vegetation Index
        if self.config.use_cvi and nir is not None and green is not None and red is not None:
            indices['cvi'] = self._calculate_cvi(nir, green, red)

        return indices

    def _to_numpy(self, data) -> Optional[np.ndarray]:
        """Convert data to numpy array, handling None values."""
        if data is None:
            return None
        return np.asarray(data, dtype=np.float32)

    def _calculate_ndvi(self, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Calculate NDVI: (NIR - Red) / (NIR + Red)"""
        numerator = nir - red
        denominator = nir + red
        # Avoid division by zero
        ndvi = np.divide(numerator, denominator,
                         out=np.full_like(numerator, np.nan),
                         where=denominator != 0)
        return np.clip(ndvi, -1, 1)

    def _calculate_evi(self, red: np.ndarray, nir: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """Calculate EVI: 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)"""
        numerator = 2.5 * (nir - red)
        denominator = nir + 6 * red - 7.5 * blue + 1
        evi = np.divide(numerator, denominator,
                        out=np.full_like(numerator, np.nan),
                        where=denominator != 0)
        return np.clip(evi, -1, 1)

    def _calculate_savi(self, red: np.ndarray, nir: np.ndarray, L: float = 0.5) -> np.ndarray:
        """Calculate SAVI: (NIR - Red) / (NIR + Red + L) * (1 + L)"""
        numerator = nir - red
        denominator = nir + red + L
        savi = np.divide(numerator, denominator,
                         out=np.full_like(numerator, np.nan),
                         where=denominator != 0)
        return np.clip(savi * (1 + L), -1, 1)

    def _calculate_arvi(self, red: np.ndarray, nir: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """Calculate ARVI: (NIR - RB) / (NIR + RB) where RB = Red - gamma*(Blue - Red)"""
        gamma = 1.0  # Atmospheric resistance parameter
        rb = red - gamma * (blue - red)
        numerator = nir - rb
        denominator = nir + rb
        arvi = np.divide(numerator, denominator,
                         out=np.full_like(numerator, np.nan),
                         where=denominator != 0)
        return np.clip(arvi, -1, 1)

    def _calculate_gndvi(self, green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Calculate GNDVI: (NIR - Green) / (NIR + Green)"""
        numerator = nir - green
        denominator = nir + green
        gndvi = np.divide(numerator, denominator,
                          out=np.full_like(numerator, np.nan),
                          where=denominator != 0)
        return np.clip(gndvi, -1, 1)

    def _calculate_cvi(self, nir: np.ndarray, green: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Calculate CVI: NIR * (Green / Red^2)"""
        cvi = nir * (green / (red ** 2 + 1e-10)
                     )  # Add small epsilon to avoid division by zero
        return cvi


class CloudFilter:
    """
    Filters out cloudy or low-quality vegetation observations.

    Addresses NDVI noise in cloudy conditions.
    """

    def __init__(self, config: VegetationConfig):
        self.config = config

    def filter_clouds(self, vegetation_indices: Dict[str, np.ndarray],
                      temporal_data: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter out cloudy observations based on multiple criteria.

        Args:
            vegetation_indices: Dict of vegetation indices
            temporal_data: Optional temporal context data

        Returns:
            Tuple of (cloud_mask, quality_score)
            - cloud_mask: True for cloudy/invalid observations
            - quality_score: 0-1 quality score
        """
        n_samples = len(next(iter(vegetation_indices.values())))
        cloud_mask = np.zeros(n_samples, dtype=bool)
        quality_scores = np.ones(n_samples, dtype=float)

        # NDVI-based cloud detection
        if 'ndvi' in vegetation_indices and self.config.enable_cloud_filtering:
            ndvi = vegetation_indices['ndvi']
            # Low NDVI may indicate clouds
            cloud_mask |= (ndvi < self.config.cloud_threshold_ndvi)
            # Adjust quality scores
            quality_scores *= np.clip(ndvi /
                                      self.config.cloud_threshold_ndvi, 0, 1)

        # Variability-based cloud detection (temporal)
        if temporal_data is not None and len(temporal_data) > 5:
            for index_name, index_values in vegetation_indices.items():
                if len(index_values) > 5:
                    # Rolling standard deviation
                    rolling_std = pd.Series(index_values).rolling(
                        5, center=True).std()
                    high_variability = rolling_std > self.config.cloud_threshold_variability
                    cloud_mask |= high_variability.fillna(False).values
                    # Reduce quality for variable observations
                    variability_penalty = np.clip(
                        rolling_std / self.config.cloud_threshold_variability, 0, 1)
                    quality_scores *= (1 - 0.3 *
                                       variability_penalty.fillna(0).values)

        # Multi-index consistency check
        if len(vegetation_indices) > 1:
            index_values = list(vegetation_indices.values())
            # Check correlation between indices
            correlations = []
            for i in range(len(index_values)):
                for j in range(i+1, len(index_values)):
                    corr = np.corrcoef(index_values[i], index_values[j])[0, 1]
                    correlations.append(abs(corr) if not np.isnan(corr) else 0)

            mean_correlation = np.mean(correlations) if correlations else 1.0
            # Low correlation may indicate inconsistent measurements (clouds/artifacts)
            if mean_correlation < 0.5:
                cloud_mask |= True  # Mark as potentially cloudy
                quality_scores *= mean_correlation

        return cloud_mask, quality_scores


class VegetationClassifier:
    """
    Classifies vegetation types to distinguish crops from grass, trees, etc.

    Addresses NDVI confounding of different vegetation types.
    """

    def __init__(self, config: VegetationConfig):
        self.config = config
        self.classifier = None
        self.is_fitted = False

    def classify_vegetation(self, vegetation_indices: Dict[str, np.ndarray],
                            temporal_features: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify vegetation types using multiple vegetation indices and temporal patterns.

        Args:
            vegetation_indices: Dict of vegetation indices
            temporal_features: Optional temporal context features

        Returns:
            Tuple of (vegetation_types, confidence_scores)
        """
        if not self.config.enable_vegetation_classification:
            n_samples = len(next(iter(vegetation_indices.values())))
            return np.full(n_samples, -1, dtype=int), np.full(n_samples, 0.5, dtype=float)

        # Prepare features for classification
        features = self._prepare_classification_features(
            vegetation_indices, temporal_features)

        if self.config.classification_method == "clustering":
            return self._classify_by_clustering(features)
        elif self.config.classification_method == "thresholds":
            return self._classify_by_thresholds(vegetation_indices)
        else:
            raise ValueError(
                f"Unknown classification method: {self.config.classification_method}")

    def _prepare_classification_features(self, vegetation_indices: Dict[str, np.ndarray],
                                         temporal_features: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Prepare feature matrix for vegetation classification."""
        feature_list = []

        # Add vegetation indices
        for index_name, index_values in vegetation_indices.items():
            feature_list.append(index_values.reshape(-1, 1))

        # Add temporal features if available
        if temporal_features:
            for feature_name, feature_values in temporal_features.items():
                if feature_values.ndim == 1:
                    feature_list.append(feature_values.reshape(-1, 1))
                else:
                    feature_list.extend(
                        [feature_values[:, i].reshape(-1, 1) for i in range(feature_values.shape[1])])

        # Combine features
        if feature_list:
            features = np.concatenate(feature_list, axis=1)
            # Handle NaN values
            features = np.nan_to_num(features, nan=0.0)
            return features
        else:
            # Fallback: use NDVI only
            ndvi = vegetation_indices.get('ndvi', np.zeros(100))
            return ndvi.reshape(-1, 1)

    def _classify_by_clustering(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Classify vegetation using unsupervised clustering."""
        # Remove rows with all zeros or NaN
        valid_mask = ~np.all(features == 0, axis=1)
        features_valid = features[valid_mask]

        if len(features_valid) < self.config.n_vegetation_types * 2:
            # Not enough data for clustering
            return np.full(len(features), 0, dtype=int), np.full(len(features), 0.5, dtype=float)

        # Use Gaussian Mixture Model for soft clustering
        try:
            gmm = GaussianMixture(n_components=self.config.n_vegetation_types,
                                  random_state=42, covariance_type='full')
            gmm.fit(features_valid)

            # Predict on all data
            predictions = gmm.predict(features)
            probabilities = gmm.predict_proba(features)
            confidence = np.max(probabilities, axis=1)

            return predictions, confidence

        except Exception as e:
            logger.warning(f"Clustering classification failed: {e}")
            return np.full(len(features), 0, dtype=int), np.full(len(features), 0.5, dtype=float)

    def _classify_by_thresholds(self, vegetation_indices: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Classify vegetation using rule-based thresholds."""
        n_samples = len(next(iter(vegetation_indices.values())))
        vegetation_types = np.zeros(n_samples, dtype=int)
        # Default high confidence for rules
        confidence = np.full(n_samples, 0.8, dtype=float)

        ndvi = vegetation_indices.get('ndvi', np.full(n_samples, 0.5))
        evi = vegetation_indices.get('evi', np.full(n_samples, 0.5))

        # Simple rule-based classification
        # 0: Bare soil / sparse vegetation
        # 1: Grass / herbaceous
        # 2: Crops (seasonal pattern)
        # 3: Trees / dense vegetation

        # Bare soil / sparse
        mask_sparse = (ndvi < 0.2) & (evi < 0.2)
        vegetation_types[mask_sparse] = 0

        # Grass / herbaceous (moderate NDVI, lower EVI)
        mask_grass = (ndvi >= 0.2) & (ndvi < 0.6) & (evi < 0.4)
        vegetation_types[mask_grass] = 1

        # Crops (higher NDVI, seasonal pattern would be detected temporally)
        mask_crops = (ndvi >= 0.4) & (ndvi < 0.8) & (evi >= 0.3)
        vegetation_types[mask_crops] = 2

        # Trees / dense vegetation (high values, EVI helps distinguish from saturation)
        mask_trees = (ndvi >= 0.6) & (evi >= 0.4)
        vegetation_types[mask_trees] = 3

        return vegetation_types, confidence


class PhenologyExtractor:
    """
    Extracts phenological features beyond simple NDVI.

    Addresses phenology modeling limitations of NDVI.
    """

    def __init__(self, config: VegetationConfig):
        self.config = config

    def extract_phenology(self, vegetation_indices: Dict[str, np.ndarray],
                          dates: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Extract phenological features from vegetation time series.

        Args:
            vegetation_indices: Dict of vegetation indices over time
            dates: Optional datetime array for seasonal analysis

        Returns:
            Dict of phenology features
        """
        features = {}

        if not self.config.enable_phenology_features:
            return features

        # Use EVI or NDVI as primary vegetation signal (EVI preferred for phenology)
        primary_index = vegetation_indices.get(
            'evi', vegetation_indices.get('ndvi'))
        if primary_index is None:
            return features

        # Smooth the time series to reduce noise
        smoothed = self._smooth_timeseries(primary_index)

        # Detect growing season
        if dates is not None:
            growing_season = self._detect_growing_season(smoothed, dates)
            features['growing_season_progress'] = growing_season
        else:
            features['growing_season_progress'] = np.full(len(smoothed), 0.5)

        # Phenology stages (simplified)
        phenology_stages = self._classify_phenology_stages(smoothed)
        features['phenology_stage'] = phenology_stages

        # Senescence detection
        senescence = self._detect_senescence(smoothed)
        features['senescence_index'] = senescence

        return features

    def _smooth_timeseries(self, timeseries: np.ndarray) -> np.ndarray:
        """Smooth time series to reduce noise."""
        if len(timeseries) < self.config.phenology_smoothing_window:
            return timeseries

        # Gaussian smoothing
        smoothed = gaussian_filter1d(
            timeseries, sigma=self.config.phenology_smoothing_window/3)
        return smoothed

    def _detect_growing_season(self, smoothed: np.ndarray, dates: np.ndarray) -> np.ndarray:
        """Detect progress through growing season."""
        # Simple approach: normalize by annual range
        if len(smoothed) < 30:
            return np.full(len(smoothed), 0.5)

        # Rolling min/max over longer window
        window = min(90, len(smoothed))  # ~3 months
        rolling_min = pd.Series(smoothed).rolling(
            window, center=True).min().fillna(method='bfill').fillna(method='ffill')
        rolling_max = pd.Series(smoothed).rolling(
            window, center=True).max().fillna(method='bfill').fillna(method='ffill')

        # Progress as normalized position between min and max
        progress = (smoothed - rolling_min) / \
            (rolling_max - rolling_min + 1e-10)
        progress = np.clip(progress, 0, 1)

        return progress.values

    def _classify_phenology_stages(self, smoothed: np.ndarray) -> np.ndarray:
        """Classify phenological stages."""
        stages = np.full(len(smoothed), 0, dtype=int)  # 0: dormant

        if len(smoothed) < 10:
            return stages

        # Simple threshold-based stages
        # 1: green-up, 2: peak growth, 3: senescence, 4: dormancy
        mean_val = np.mean(smoothed)
        std_val = np.std(smoothed)

        # Green-up (increasing trend)
        diff = np.diff(smoothed, prepend=smoothed[0])
        increasing = diff > 0.01
        stages[increasing & (smoothed < mean_val)] = 1

        # Peak growth (high values, stable)
        high_values = smoothed > (mean_val + 0.5 * std_val)
        stable = np.abs(diff) < 0.005
        stages[high_values & stable] = 2

        # Senescence (decreasing trend)
        decreasing = diff < -0.01
        stages[decreasing] = 3

        return stages

    def _detect_senescence(self, smoothed: np.ndarray) -> np.ndarray:
        """Detect senescence (yellowing/dying vegetation)."""
        if len(smoothed) < 5:
            return np.full(len(smoothed), 0.0)

        # Senescence indicated by decreasing trend after peak
        senescence = np.zeros(len(smoothed))

        # Find local maxima
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(smoothed, distance=10)

        for peak_idx in peaks:
            # Look at period after peak
            end_idx = min(peak_idx + 30, len(smoothed))
            post_peak = smoothed[peak_idx:end_idx]

            if len(post_peak) > 5:
                # Rate of decline
                decline_rate = (post_peak[0] - post_peak[-1]) / len(post_peak)
                senescence[peak_idx:end_idx] = np.clip(decline_rate * 10, 0, 1)

        return senescence


class TranspirationEstimator:
    """
    Estimates transpiration potential from vegetation characteristics.

    Addresses transpiration modeling limitations of NDVI.
    """

    def __init__(self, config: VegetationConfig):
        self.config = config

    def estimate_transpiration(self, vegetation_indices: Dict[str, np.ndarray],
                               meteorological_data: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """
        Estimate transpiration potential from vegetation structure.

        Args:
            vegetation_indices: Dict of vegetation indices
            meteorological_data: Optional weather data (temperature, humidity, etc.)

        Returns:
            Dict of transpiration-related features
        """
        features = {}

        if not self.config.enable_transpiration_features:
            return features

        # Primary vegetation signal
        primary_vi = vegetation_indices.get(
            'evi', vegetation_indices.get('ndvi'))
        if primary_vi is None:
            return features

        # Canopy conductance proxy (related to transpiration)
        # Higher vegetation density = higher conductance = higher transpiration potential
        canopy_conductance = self._estimate_canopy_conductance(primary_vi)
        features['canopy_conductance_proxy'] = canopy_conductance

        # Transpiration potential (combines vegetation and environmental factors)
        transpiration_potential = self._estimate_transpiration_potential(
            primary_vi, meteorological_data)
        features['transpiration_potential'] = transpiration_potential

        return features

    def _estimate_canopy_conductance(self, vegetation_index: np.ndarray) -> np.ndarray:
        """Estimate canopy conductance from vegetation index."""
        # Simplified relationship: higher VI = higher conductance
        # In reality, this would be calibrated with field measurements
        conductance = np.clip(vegetation_index * 2.0, 0,
                              1)  # Scale to 0-1 range
        return conductance

    def _estimate_transpiration_potential(self, vegetation_index: np.ndarray,
                                          meteorological_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Estimate transpiration potential combining vegetation and weather."""
        # Base transpiration from vegetation
        base_transpiration = np.clip(vegetation_index, 0, 1)

        # Modify by environmental factors if available
        if meteorological_data is not None:
            # Temperature effect (optimal around 20-25°C)
            temp = meteorological_data.get(
                'temperature', np.full(len(vegetation_index), 20))
            # Gaussian around 22°C
            temp_factor = np.exp(-((temp - 22) / 10)**2)

            # Humidity effect (higher humidity = lower transpiration)
            humidity = meteorological_data.get(
                'humidity', np.full(len(vegetation_index), 50))
            humidity_factor = 1 - (humidity / 100) * 0.5  # Reduce by up to 50%

            # Combine factors
            environmental_factor = temp_factor * humidity_factor
            transpiration = base_transpiration * environmental_factor
        else:
            transpiration = base_transpiration

        return np.clip(transpiration, 0, 1)


class VegetationFeatureEngineer:
    """
    Main class for advanced vegetation feature engineering.

    Orchestrates all vegetation-related feature extraction to address NDVI limitations.
    """

    def __init__(self, config: VegetationConfig):
        self.config = config

        # Initialize components
        self.index_calculator = VegetationIndexCalculator(config)
        self.cloud_filter = CloudFilter(config)
        self.vegetation_classifier = VegetationClassifier(config)
        self.phenology_extractor = PhenologyExtractor(config)
        self.transpiration_estimator = TranspirationEstimator(config)

    def engineer_features(self, spectral_data: pd.DataFrame,
                          temporal_data: Optional[pd.DataFrame] = None) -> VegetationFeatures:
        """
        Engineer comprehensive vegetation features from spectral and temporal data.

        Args:
            spectral_data: DataFrame with spectral bands and dates
            temporal_data: Optional additional temporal context

        Returns:
            VegetationFeatures object with all computed features
        """
        logger.info("Engineering advanced vegetation features...")

        features = VegetationFeatures()

        # 1. Calculate multiple vegetation indices
        vegetation_indices = self.index_calculator.calculate_indices(
            spectral_data)

        # Store base indices
        for index_name, index_values in vegetation_indices.items():
            setattr(features, index_name, index_values)

        # 2. Cloud filtering and quality assessment
        if self.config.enable_cloud_filtering:
            cloud_mask, quality_scores = self.cloud_filter.filter_clouds(
                vegetation_indices, temporal_data)
            features.cloud_mask = cloud_mask
            features.quality_score = quality_scores
            features.valid_observations = ~cloud_mask

        # 3. Vegetation type classification
        if self.config.enable_vegetation_classification:
            veg_types, veg_confidence = self.vegetation_classifier.classify_vegetation(
                vegetation_indices)
            features.vegetation_type = veg_types
            features.vegetation_type_confidence = veg_confidence

        # 4. Phenology features
        dates = spectral_data.get(
            'date') if 'date' in spectral_data.columns else None
        phenology_features = self.phenology_extractor.extract_phenology(
            vegetation_indices, dates)
        for feature_name, feature_values in phenology_features.items():
            setattr(features, feature_name, feature_values)

        # 5. Transpiration features
        transpiration_features = self.transpiration_estimator.estimate_transpiration(
            vegetation_indices, temporal_data)
        for feature_name, feature_values in transpiration_features.items():
            setattr(features, feature_name, feature_values)

        # 6. LAI and canopy structure estimation
        if self.config.handle_lai_saturation:
            lai_features = self._estimate_lai_and_structure(vegetation_indices)
            for feature_name, feature_values in lai_features.items():
                setattr(features, feature_name, feature_values)

        # 7. Temporal features
        if self.config.enable_temporal_features and len(spectral_data) > max(self.config.temporal_lags):
            temporal_features = self._extract_temporal_features(
                vegetation_indices, spectral_data)
            features.temporal_lags = temporal_features.get('lags', {})
            features.trends = temporal_features.get('trends', {})
            features.seasonality = temporal_features.get('seasonality')

        # 8. Derived vegetation dynamics
        dynamics = self._compute_vegetation_dynamics(vegetation_indices)
        features.vegetation_dynamics = dynamics.get('dynamics')
        features.stress_indicators = dynamics.get('stress')

        logger.info(
            f"Engineered {len([f for f in dir(features) if not f.startswith('_') and getattr(features, f) is not None])} vegetation features")

        return features

    def _estimate_lai_and_structure(self, vegetation_indices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Estimate LAI and canopy structure, handling saturation."""
        features = {}

        # Use multiple indices to handle saturation
        ndvi = vegetation_indices.get('ndvi')
        evi = vegetation_indices.get('evi')
        savi = vegetation_indices.get('savi')

        if ndvi is not None:
            # LAI proxy using multiple indices
            lai_proxy = ndvi.copy()

            # Use EVI for high LAI where NDVI saturates
            if evi is not None and self.config.multi_index_fusion:
                # Fuse indices: use EVI where NDVI > saturation threshold
                saturation_mask = ndvi > self.config.saturation_threshold_ndvi
                lai_proxy[saturation_mask] = evi[saturation_mask] * \
                    1.2  # Scale EVI to match NDVI range

            # Use SAVI for sparse vegetation
            if savi is not None and self.config.multi_index_fusion:
                sparse_mask = ndvi < 0.3
                lai_proxy[sparse_mask] = savi[sparse_mask]

            features['lai_proxy'] = np.clip(lai_proxy, 0, 1)

        # Canopy cover estimation
        if evi is not None:
            # Simplified canopy cover from EVI
            canopy_cover = np.clip(evi * 1.5, 0, 1)
            features['canopy_cover'] = canopy_cover

        # Vegetation height proxy (rough estimate from density)
        if ndvi is not None and evi is not None:
            # Combine indices for height proxy
            height_proxy = (ndvi + evi) / 2
            features['vegetation_height_proxy'] = np.clip(height_proxy, 0, 1)

        return features

    def _extract_temporal_features(self, vegetation_indices: Dict[str, np.ndarray],
                                   spectral_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract temporal vegetation features."""
        features = {'lags': {}, 'trends': {}}

        # Use primary vegetation index
        primary_vi = vegetation_indices.get(
            'evi', vegetation_indices.get('ndvi'))
        if primary_vi is None:
            return features

        # Temporal lags
        for lag in self.config.temporal_lags:
            if len(primary_vi) > lag:
                lagged = np.roll(primary_vi, lag)
                lagged[:lag] = np.nan  # No valid lag data at start
                features['lags'][f'vi_lag_{lag}d'] = lagged

        # Trends over different windows
        for window in self.config.trend_windows:
            if len(primary_vi) > window:
                trends = np.full(len(primary_vi), np.nan)
                for i in range(window, len(primary_vi)):
                    # Linear trend over window
                    y = primary_vi[i-window:i]
                    x = np.arange(window)
                    if len(y) > 1:
                        slope, _ = np.polyfit(x, y, 1)
                        trends[i] = slope
                features['trends'][f'vi_trend_{window}d'] = trends

        # Seasonality (simplified)
        if len(primary_vi) > 365:  # Need at least a year
            # Detrend and find seasonal component
            from scipy import signal
            detrended = signal.detrend(primary_vi)
            seasonality = primary_vi - detrended
            features['seasonality'] = seasonality
        else:
            features['seasonality'] = np.full(len(primary_vi), 0.0)

        return features

    def _compute_vegetation_dynamics(self, vegetation_indices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute vegetation dynamics and stress indicators."""
        dynamics = {}

        primary_vi = vegetation_indices.get(
            'evi', vegetation_indices.get('ndvi'))
        if primary_vi is None or len(primary_vi) < 5:
            return {'dynamics': None, 'stress': None}

        # Vegetation dynamics (rate of change)
        dynamics_smooth = gaussian_filter1d(primary_vi, sigma=3)
        vegetation_dynamics = np.gradient(dynamics_smooth)
        dynamics['dynamics'] = vegetation_dynamics

        # Stress indicators (deviations from expected)
        if len(primary_vi) > 30:
            # Rolling mean as expected value
            rolling_mean = pd.Series(primary_vi).rolling(
                30, center=True).mean()
            stress = primary_vi - \
                rolling_mean.fillna(method='bfill').fillna(method='ffill')
            # Normalize stress
            stress = stress / (np.std(primary_vi) + 1e-10)
            dynamics['stress'] = np.clip(np.abs(stress), 0, 3)  # Cap at 3 SD
        else:
            dynamics['stress'] = np.full(len(primary_vi), 0.0)

        return dynamics
