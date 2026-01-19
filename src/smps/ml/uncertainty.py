"""
Uncertainty Quantification for Soil Moisture Predictions.

Implements multiple approaches:

1. Quantile Regression: Direct prediction of confidence intervals
2. Ensemble Uncertainty: Variance from ensemble members
3. Conformal Prediction: Distribution-free prediction intervals
4. Bayesian Approximation: Dropout-based uncertainty (for neural nets)

Research Background:
- Meinshausen (2006): Quantile Regression Forests
- Tagasovska & Lopez-Paz (2019): Single-model uncertainty
- Romano et al. (2019): Conformal prediction intervals
- Gal & Ghahramani (2016): Dropout as Bayesian approximation

Use Cases:
- Irrigation scheduling under uncertainty
- Risk-aware decision making
- Model reliability assessment
- Identifying prediction confidence regions
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("smps.ml.uncertainty")


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification."""

    # Quantiles to predict
    quantiles: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    )

    # Conformal prediction
    conformal_alpha: float = 0.10  # For 90% prediction intervals

    # Ensemble settings
    n_ensemble_members: int = 10
    bootstrap_fraction: float = 0.8

    # Model settings
    model_type: str = 'lightgbm'  # 'lightgbm', 'xgboost', 'quantile_forest'


@dataclass
class PredictionWithUncertainty:
    """Container for predictions with uncertainty estimates."""

    # Point prediction
    mean: np.ndarray

    # Prediction intervals
    lower_90: Optional[np.ndarray] = None
    upper_90: Optional[np.ndarray] = None
    lower_50: Optional[np.ndarray] = None
    upper_50: Optional[np.ndarray] = None

    # Full quantiles (if computed)
    quantiles: Optional[Dict[float, np.ndarray]] = None

    # Uncertainty measures
    std: Optional[np.ndarray] = None
    epistemic_uncertainty: Optional[np.ndarray] = None
    aleatoric_uncertainty: Optional[np.ndarray] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        data = {'prediction': self.mean}

        if self.lower_90 is not None:
            data['lower_90'] = self.lower_90
        if self.upper_90 is not None:
            data['upper_90'] = self.upper_90
        if self.lower_50 is not None:
            data['lower_50'] = self.lower_50
        if self.upper_50 is not None:
            data['upper_50'] = self.upper_50
        if self.std is not None:
            data['std'] = self.std

        return pd.DataFrame(data)


class QuantileRegressor:
    """
    Quantile regression for direct uncertainty estimation.

    Trains separate models for each quantile to produce
    prediction intervals that adapt to input features.

    Benefits:
    - Non-parametric: No distributional assumptions
    - Adaptive: Uncertainty varies with inputs
    - Interpretable: Direct prediction of percentiles
    """

    def __init__(self, config: Optional[UncertaintyConfig] = None):
        self.config = config or UncertaintyConfig()
        self.models: Dict[float, Any] = {}
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        eval_set: Optional[Tuple[pd.DataFrame, np.ndarray]] = None,
    ):
        """
        Fit quantile regression models.

        Args:
            X: Features
            y: Target values
            eval_set: Optional validation set for early stopping
        """
        logger.info(
            f"Fitting quantile regression for {len(self.config.quantiles)} quantiles")

        for q in self.config.quantiles:
            logger.debug(f"Fitting quantile {q}")

            if self.config.model_type == 'lightgbm':
                self.models[q] = self._fit_lightgbm_quantile(X, y, q, eval_set)
            elif self.config.model_type == 'xgboost':
                self.models[q] = self._fit_xgboost_quantile(X, y, q, eval_set)
            else:
                raise ValueError(
                    f"Unknown model type: {self.config.model_type}")

        self.is_fitted = True
        logger.info("Quantile regression fitting complete")

    def _fit_lightgbm_quantile(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        quantile: float,
        eval_set: Optional[Tuple[pd.DataFrame, np.ndarray]] = None,
    ):
        """Fit LightGBM model for specific quantile."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM required: pip install lightgbm")

        params = {
            'objective': 'quantile',
            'alpha': quantile,
            'metric': 'quantile',
            'verbosity': -1,
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 6,
            'num_leaves': 31,
            'min_child_samples': 20,
            'random_state': 42,
        }

        model = lgb.LGBMRegressor(**params)

        if eval_set:
            model.fit(
                X, y,
                eval_set=[eval_set],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
        else:
            model.fit(X, y)

        return model

    def _fit_xgboost_quantile(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        quantile: float,
        eval_set: Optional[Tuple[pd.DataFrame, np.ndarray]] = None,
    ):
        """Fit XGBoost model for specific quantile."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost required: pip install xgboost")

        # XGBoost quantile loss
        def quantile_loss(y_true, y_pred):
            residual = y_true - y_pred
            grad = np.where(residual > 0, -quantile, -(quantile - 1))
            hess = np.ones_like(grad)
            return grad, hess

        params = {
            'objective': 'reg:quantileerror',
            'quantile_alpha': quantile,
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 6,
            'random_state': 42,
            'verbosity': 0,
        }

        model = xgb.XGBRegressor(**params)

        if eval_set:
            model.fit(X, y, eval_set=[eval_set], verbose=False)
        else:
            model.fit(X, y)

        return model

    def predict(self, X: pd.DataFrame) -> PredictionWithUncertainty:
        """
        Predict with uncertainty.

        Args:
            X: Features

        Returns:
            PredictionWithUncertainty with quantile predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Predict all quantiles
        quantile_preds = {}
        for q, model in self.models.items():
            quantile_preds[q] = model.predict(X)

        # Extract key quantiles
        median = quantile_preds.get(0.5, quantile_preds.get(0.50))
        if median is None:
            median = np.mean([p for p in quantile_preds.values()], axis=0)

        result = PredictionWithUncertainty(
            mean=median,
            quantiles=quantile_preds,
        )

        # 90% interval
        if 0.05 in quantile_preds and 0.95 in quantile_preds:
            result.lower_90 = quantile_preds[0.05]
            result.upper_90 = quantile_preds[0.95]
        elif 0.10 in quantile_preds and 0.90 in quantile_preds:
            result.lower_90 = quantile_preds[0.10]
            result.upper_90 = quantile_preds[0.90]

        # 50% interval
        if 0.25 in quantile_preds and 0.75 in quantile_preds:
            result.lower_50 = quantile_preds[0.25]
            result.upper_50 = quantile_preds[0.75]

        # Estimate std from IQR
        if 0.25 in quantile_preds and 0.75 in quantile_preds:
            iqr = quantile_preds[0.75] - quantile_preds[0.25]
            result.std = iqr / 1.35  # IQR to std approximation for normal

        return result


class EnsembleUncertainty:
    """
    Uncertainty from ensemble of models.

    Trains multiple models with bootstrap sampling and
    uses prediction variance as uncertainty estimate.

    Captures both:
    - Epistemic uncertainty: Model uncertainty (reducible with more data)
    - Aleatoric uncertainty: Data uncertainty (irreducible noise)
    """

    def __init__(self, config: Optional[UncertaintyConfig] = None):
        self.config = config or UncertaintyConfig()
        self.models: List[Any] = []
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        eval_set: Optional[Tuple[pd.DataFrame, np.ndarray]] = None,
    ):
        """
        Fit ensemble of bootstrap models.

        Args:
            X: Features
            y: Target values
            eval_set: Optional validation set
        """
        n_samples = len(X)
        np.random.seed(42)

        logger.info(
            f"Fitting ensemble of {self.config.n_ensemble_members} models")

        for i in range(self.config.n_ensemble_members):
            # Bootstrap sample
            boot_idx = np.random.choice(
                n_samples,
                size=int(n_samples * self.config.bootstrap_fraction),
                replace=True
            )

            X_boot = X.iloc[boot_idx]
            y_boot = y[boot_idx]

            # Train model
            if self.config.model_type == 'lightgbm':
                model = self._create_lightgbm()
            elif self.config.model_type == 'xgboost':
                model = self._create_xgboost()
            else:
                raise ValueError(
                    f"Unknown model type: {self.config.model_type}")

            if eval_set:
                model.fit(X_boot, y_boot, eval_set=[eval_set])
            else:
                model.fit(X_boot, y_boot)

            self.models.append(model)

        self.is_fitted = True
        logger.info("Ensemble fitting complete")

    def _create_lightgbm(self):
        """Create LightGBM model with some randomness."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM required")

        return lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=np.random.randint(10000),
            verbosity=-1,
        )

    def _create_xgboost(self):
        """Create XGBoost model with some randomness."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost required")

        return xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=np.random.randint(10000),
            verbosity=0,
        )

    def predict(self, X: pd.DataFrame) -> PredictionWithUncertainty:
        """
        Predict with uncertainty from ensemble.

        Args:
            X: Features

        Returns:
            PredictionWithUncertainty with ensemble statistics
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get predictions from all models
        all_preds = np.array([m.predict(X) for m in self.models])

        # Statistics
        mean_pred = np.mean(all_preds, axis=0)
        std_pred = np.std(all_preds, axis=0)

        # Percentiles
        lower_90 = np.percentile(all_preds, 5, axis=0)
        upper_90 = np.percentile(all_preds, 95, axis=0)
        lower_50 = np.percentile(all_preds, 25, axis=0)
        upper_50 = np.percentile(all_preds, 75, axis=0)

        return PredictionWithUncertainty(
            mean=mean_pred,
            std=std_pred,
            lower_90=lower_90,
            upper_90=upper_90,
            lower_50=lower_50,
            upper_50=upper_50,
            epistemic_uncertainty=std_pred,  # Ensemble spread = epistemic
        )


class ConformalPredictor:
    """
    Conformal prediction for distribution-free prediction intervals.

    Provides valid coverage guarantees under mild assumptions
    (exchangeability of calibration and test data).

    Benefits:
    - Distribution-free: No parametric assumptions
    - Valid coverage: Provable coverage guarantees
    - Adaptive: Interval width varies with difficulty

    Reference: Romano et al. (2019) - Conformalized Quantile Regression
    """

    def __init__(
        self,
        base_model,
        alpha: float = 0.10,  # For 90% intervals
    ):
        """
        Initialize conformal predictor.

        Args:
            base_model: Any model with fit/predict interface
            alpha: Miscoverage rate (1 - alpha = coverage)
        """
        self.base_model = base_model
        self.alpha = alpha
        self.calibration_scores: Optional[np.ndarray] = None
        self.is_calibrated = False

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """
        Fit the base model.

        Args:
            X_train: Training features
            y_train: Training targets
        """
        self.base_model.fit(X_train, y_train)

    def calibrate(
        self,
        X_cal: pd.DataFrame,
        y_cal: np.ndarray,
    ):
        """
        Calibrate conformal predictor using held-out calibration set.

        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
        """
        # Get predictions on calibration set
        y_pred = self.base_model.predict(X_cal)

        # Compute nonconformity scores (absolute residuals)
        self.calibration_scores = np.abs(y_cal - y_pred)
        self.is_calibrated = True

        logger.info("Calibrated with %d samples", len(y_cal))

    def predict(self, X: pd.DataFrame) -> PredictionWithUncertainty:
        """
        Predict with conformal intervals.

        Args:
            X: Features

        Returns:
            PredictionWithUncertainty with conformal intervals
        """
        if not self.is_calibrated:
            raise RuntimeError("Model not calibrated. Call calibrate() first.")

        # Point predictions
        y_pred = self.base_model.predict(X)

        # Conformal quantile
        n_cal = len(self.calibration_scores)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        q = np.quantile(self.calibration_scores, min(q_level, 1.0))

        # Prediction intervals
        lower = y_pred - q
        upper = y_pred + q

        return PredictionWithUncertainty(
            mean=y_pred,
            lower_90=lower if self.alpha == 0.10 else None,
            upper_90=upper if self.alpha == 0.10 else None,
        )


class CQRPredictor:
    """
    Conformalized Quantile Regression (CQR).

    Combines quantile regression with conformal prediction
    for adaptive, distribution-free prediction intervals.

    Benefits over standard conformal:
    - Adaptive interval width (narrows for easy predictions)
    - Heteroscedastic: Uncertainty varies with input
    - Still has valid coverage guarantees

    Reference: Romano, Patterson, Candès (2019)
    """

    def __init__(
        self,
        alpha: float = 0.10,
        model_type: str = 'lightgbm',
    ):
        """
        Initialize CQR predictor.

        Args:
            alpha: Miscoverage rate
            model_type: Base model type
        """
        self.alpha = alpha
        self.model_type = model_type

        # Train models for lower and upper quantiles
        self.lower_quantile = alpha / 2
        self.upper_quantile = 1 - alpha / 2

        self.model_lower = None
        self.model_upper = None
        self.model_median = None

        self.calibration_scores: Optional[np.ndarray] = None
        self.is_calibrated = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
    ):
        """
        Fit quantile regression models.

        Args:
            X_train: Training features
            y_train: Training targets
        """
        logger.info("Fitting CQR models for alpha=%s", str(self.alpha))

        # Fit quantile models
        qr_config = UncertaintyConfig(
            quantiles=[self.lower_quantile, 0.5, self.upper_quantile],
            model_type=self.model_type,
        )

        self._qr = QuantileRegressor(qr_config)
        self._qr.fit(X_train, y_train)

        self.model_lower = self._qr.models[self.lower_quantile]
        self.model_upper = self._qr.models[self.upper_quantile]
        self.model_median = self._qr.models[0.5]

    def calibrate(
        self,
        X_cal: pd.DataFrame,
        y_cal: np.ndarray,
    ):
        """
        Calibrate CQR using held-out data.

        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
        """
        # Get quantile predictions
        q_lower = self.model_lower.predict(X_cal)
        q_upper = self.model_upper.predict(X_cal)

        # Compute conformity scores
        # Score = max(q_lower - y, y - q_upper)
        scores = np.maximum(q_lower - y_cal, y_cal - q_upper)
        self.calibration_scores = scores
        self.is_calibrated = True

        logger.info("CQR calibrated with %d samples", len(y_cal))

    def predict(self, X: pd.DataFrame) -> PredictionWithUncertainty:
        """
        Predict with CQR intervals.

        Args:
            X: Features

        Returns:
            PredictionWithUncertainty with adaptive conformal intervals
        """
        if not self.is_calibrated:
            raise RuntimeError("CQR not calibrated. Call calibrate() first.")

        # Get quantile predictions
        q_lower = self.model_lower.predict(X)
        q_upper = self.model_upper.predict(X)
        q_median = self.model_median.predict(X)

        # Conformal adjustment
        n_cal = len(self.calibration_scores)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        q_correction = np.quantile(self.calibration_scores, min(q_level, 1.0))

        # Adjust intervals
        lower = q_lower - q_correction
        upper = q_upper + q_correction

        # Estimate std from interval width
        std = (upper - lower) / (2 * 1.96)  # Approximate for 95%

        return PredictionWithUncertainty(
            mean=q_median,
            lower_90=lower,
            upper_90=upper,
            std=std,
        )


class UncertaintyCalibrator:
    """
    Calibration assessment and adjustment for uncertainty estimates.

    Ensures that predicted confidence intervals have
    the correct empirical coverage.
    """

    @staticmethod
    def reliability_diagram(
        y_true: np.ndarray,
        predictions: PredictionWithUncertainty,
        n_bins: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute reliability diagram data.

        Compares expected vs observed coverage at different quantiles.

        Args:
            y_true: True values
            predictions: Predictions with quantiles
            n_bins: Number of probability bins

        Returns:
            Tuple of (expected_coverage, observed_coverage, bin_counts)
        """
        if predictions.quantiles is None:
            raise ValueError("Predictions must have quantiles")

        # Get available quantiles
        quantiles = sorted(predictions.quantiles.keys())

        expected = []
        observed = []
        counts = []

        for q in quantiles:
            q_pred = predictions.quantiles[q]

            # Observed fraction below quantile
            below = y_true <= q_pred
            obs_coverage = np.mean(below)

            expected.append(q)
            observed.append(obs_coverage)
            counts.append(len(y_true))

        return np.array(expected), np.array(observed), np.array(counts)

    @staticmethod
    def calibration_error(
        y_true: np.ndarray,
        predictions: PredictionWithUncertainty,
    ) -> float:
        """
        Compute average calibration error.

        Measures how well predicted intervals match observed coverage.

        Args:
            y_true: True values
            predictions: Predictions with quantiles

        Returns:
            Mean absolute calibration error
        """
        expected, observed, _ = UncertaintyCalibrator.reliability_diagram(
            y_true, predictions
        )

        return np.mean(np.abs(expected - observed))

    @staticmethod
    def interval_coverage(
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> float:
        """
        Compute empirical coverage of prediction interval.

        Args:
            y_true: True values
            lower: Lower bounds
            upper: Upper bounds

        Returns:
            Fraction of true values within interval
        """
        in_interval = (y_true >= lower) & (y_true <= upper)
        return np.mean(in_interval)

    @staticmethod
    def interval_sharpness(
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> float:
        """
        Compute sharpness (average interval width).

        Narrower intervals are sharper (better if coverage is maintained).

        Args:
            lower: Lower bounds
            upper: Upper bounds

        Returns:
            Mean interval width
        """
        return np.mean(upper - lower)

    @staticmethod
    def winkler_score(
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        alpha: float = 0.10,
    ) -> float:
        """
        Compute Winkler score for interval forecasts.

        Rewards narrow intervals, penalizes non-coverage.

        Args:
            y_true: True values
            lower: Lower bounds
            upper: Upper bounds
            alpha: Nominal miscoverage rate

        Returns:
            Winkler score (lower is better)
        """
        width = upper - lower

        # Penalty for under-coverage
        below = y_true < lower
        above = y_true > upper

        penalty_below = (2 / alpha) * (lower - y_true) * below
        penalty_above = (2 / alpha) * (y_true - upper) * above

        score = width + penalty_below + penalty_above

        return np.mean(score)


# =============================================================================
# Combined Uncertainty Quantifier
# =============================================================================

class HybridUncertaintyQuantifier:
    """
    Combines multiple uncertainty methods for robust estimates.

    Approach:
    1. Quantile regression for adaptive intervals
    2. Ensemble for epistemic uncertainty
    3. Conformal calibration for valid coverage

    Provides comprehensive uncertainty estimates for
    decision-making under uncertainty.
    """

    def __init__(
        self,
        config: Optional[UncertaintyConfig] = None,
        use_quantile: bool = True,
        use_ensemble: bool = True,
        use_conformal: bool = True,
    ):
        self.config = config or UncertaintyConfig()

        self.use_quantile = use_quantile
        self.use_ensemble = use_ensemble
        self.use_conformal = use_conformal

        self.quantile_model: Optional[QuantileRegressor] = None
        self.ensemble_model: Optional[EnsembleUncertainty] = None
        self.conformal_model: Optional[CQRPredictor] = None

        self.is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_cal: Optional[pd.DataFrame] = None,
        y_cal: Optional[np.ndarray] = None,
    ):
        """
        Fit all uncertainty models.

        Args:
            X_train: Training features
            y_train: Training targets
            X_cal: Calibration features (for conformal)
            y_cal: Calibration targets
        """
        logger.info("Fitting hybrid uncertainty quantifier")

        # Split for calibration if not provided
        if X_cal is None or y_cal is None:
            n = len(X_train)
            cal_size = int(n * 0.2)

            X_cal = X_train.iloc[-cal_size:]
            y_cal = y_train[-cal_size:]
            X_train = X_train.iloc[:-cal_size]
            y_train = y_train[:-cal_size]

        # Fit quantile regression
        if self.use_quantile:
            logger.info("Fitting quantile regression...")
            self.quantile_model = QuantileRegressor(self.config)
            self.quantile_model.fit(X_train, y_train)

        # Fit ensemble
        if self.use_ensemble:
            logger.info("Fitting ensemble...")
            self.ensemble_model = EnsembleUncertainty(self.config)
            self.ensemble_model.fit(X_train, y_train)

        # Fit and calibrate CQR
        if self.use_conformal:
            logger.info("Fitting and calibrating CQR...")
            self.conformal_model = CQRPredictor(
                alpha=self.config.conformal_alpha,
                model_type=self.config.model_type,
            )
            self.conformal_model.fit(X_train, y_train)
            self.conformal_model.calibrate(X_cal, y_cal)

        self.is_fitted = True
        logger.info("Hybrid uncertainty quantifier ready")

    def predict(self, X: pd.DataFrame) -> PredictionWithUncertainty:
        """
        Predict with combined uncertainty estimates.

        Combines:
        - Point prediction from median
        - Intervals from CQR (calibrated)
        - Epistemic uncertainty from ensemble

        Args:
            X: Features

        Returns:
            PredictionWithUncertainty with comprehensive estimates
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get predictions from each method
        results = {}

        if self.use_quantile:
            results['quantile'] = self.quantile_model.predict(X)

        if self.use_ensemble:
            results['ensemble'] = self.ensemble_model.predict(X)

        if self.use_conformal:
            results['conformal'] = self.conformal_model.predict(X)

        # Combine results
        # Use conformal for calibrated intervals
        # Use ensemble for epistemic uncertainty
        # Use quantile for additional percentiles

        if 'conformal' in results:
            primary = results['conformal']
        elif 'quantile' in results:
            primary = results['quantile']
        else:
            primary = results['ensemble']

        combined = PredictionWithUncertainty(
            mean=primary.mean,
            lower_90=primary.lower_90,
            upper_90=primary.upper_90,
        )

        # Add epistemic from ensemble
        if 'ensemble' in results:
            combined.epistemic_uncertainty = results['ensemble'].std

        # Add quantiles
        if 'quantile' in results:
            combined.quantiles = results['quantile'].quantiles
            combined.lower_50 = results['quantile'].lower_50
            combined.upper_50 = results['quantile'].upper_50

        # Combined std
        if combined.lower_90 is not None and combined.upper_90 is not None:
            combined.std = (combined.upper_90 -
                            combined.lower_90) / (2 * 1.645)

        return combined


# Alias for backward compatibility
UncertaintyQuantifier = HybridUncertaintyQuantifier
