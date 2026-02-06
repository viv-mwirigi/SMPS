"""
Multi-Horizon Soil Water Potential Forecaster.

This module orchestrates physics and ML models to produce
matric potential forecasts at multiple horizons (0h, 24h, 72h, 168h).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from swpps.core.types import MatricPotential, PredictionResult, SoilMoistureStatus, VanGenuchtenParams
from swpps.physics.van_genuchten import water_content_from_potential
from swpps.physics.water_balance import LayerConfig, TensionSpaceWaterBalance, WaterBalanceConfig
from swpps.ml.hybrid_model import HybridTensionModel, HybridModelConfig
from swpps.features.engineering import FeatureEngineer, FeatureConfig

logger = logging.getLogger("swpps.prediction.forecaster")


@dataclass
class ForecastConfig:
    """Configuration for multi-horizon forecasting."""

    # Forecast horizons in hours
    horizons: List[int] = field(default_factory=lambda: [0, 6, 24, 72, 168])

    # Physics model configuration
    physics_dt_hours: float = 1.0
    root_depth_m: float = 0.30

    # Feature configuration
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)

    # ML model configuration
    ml_config: HybridModelConfig = field(default_factory=HybridModelConfig)

    # Whether to use hybrid (physics + ML) or physics only
    use_hybrid: bool = True

    # Number of ensemble members for uncertainty
    n_ensemble: int = 10


class SoilWaterForecaster:
    """
    Produces matric potential forecasts for multiple horizons.

    The forecaster combines:
    1. Physics-based water balance model
    2. ML residual correction model
    3. Uncertainty quantification through quantile regression
    """

    def __init__(
        self,
        vg_params: VanGenuchtenParams,
        config: Optional[ForecastConfig] = None,
    ):
        self.config = config or ForecastConfig()
        self.vg_params = vg_params

        # Initialize physics model (daily timestep water balance)
        # Use a small 3-layer profile to preserve surface/root/deep dynamics.
        layer_thickness = 1.0 / 3.0
        layers = [
            LayerConfig(
                depth_top_m=i * layer_thickness,
                depth_bottom_m=(i + 1) * layer_thickness,
                van_genuchten=vg_params,
            )
            for i in range(3)
        ]
        wb_config = WaterBalanceConfig(
            layers=layers,
            initial_psi_kpa=-50.0,
        )
        self.physics_model = TensionSpaceWaterBalance(wb_config)

        # ML model (will be set after training)
        self.hybrid_model: Optional[HybridTensionModel] = None

        # Feature engineer
        self.feature_engineer = FeatureEngineer(
            self.config.feature_config,
            vg_params=self.vg_params,
        )

        # Track state
        self.is_fitted = False

    def train(
        self,
        training_data: pd.DataFrame,
        psi_obs_col: str = "psi_observed_kpa",
    ) -> "SoilWaterForecaster":
        """
        Train the hybrid model on historical data.

        Args:
            training_data: DataFrame with observations, weather, and datetime index
            psi_obs_col: Column name for observed matric potential

        Returns:
            Self
        """
        logger.info("Training forecaster with %d samples", len(training_data))

        # Step 1: Run physics model over training period
        physics_output = self._run_physics_simulation(training_data)

        # Merge physics output with training data
        training_data = training_data.join(physics_output, rsuffix="_physics")

        # Step 2: Engineer features
        training_data = self.feature_engineer.create_features(
            training_data, psi_col=psi_obs_col
        )

        # Step 3: Train hybrid ML model
        if self.config.use_hybrid:
            self.hybrid_model = HybridTensionModel(
                physics_model=self.physics_model,
                config=self.config.ml_config,
            )

            self.hybrid_model.fit(
                training_data,
                target_col=psi_obs_col,
                physics_col="psi_root_kpa",
                horizons=self.config.horizons,
            )

        self.is_fitted = True
        logger.info("Forecaster training complete")

        return self

    def forecast(
        self,
        current_state: Dict[str, float],
        weather_forecast: pd.DataFrame,
        irrigation_schedule: Optional[pd.DataFrame] = None,
    ) -> Dict[int, PredictionResult]:
        """
        Generate matric potential forecasts for all horizons.

        Args:
            current_state: Current conditions (psi_kpa, temperature, etc.)
            weather_forecast: Weather forecast with datetime index
            irrigation_schedule: Planned irrigation (datetime, amount_mm)

        Returns:
            Dictionary mapping horizon (hours) to PredictionResult
        """
        results = {}

        # Run physics forecast
        physics_forecast = self._run_physics_forecast(
            current_state, weather_forecast, irrigation_schedule
        )

        # Prepare features for ML model
        features_df = self._prepare_forecast_features(
            current_state, weather_forecast, physics_forecast
        )

        for horizon in self.config.horizons:
            result = self._forecast_horizon(
                horizon, features_df, physics_forecast
            )
            results[horizon] = result

        return results

    def _run_physics_simulation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run physics model over historical data.

        The physics model is daily. If the input is sub-daily, we aggregate
        to daily totals/means for forcing.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("training_data must have a DatetimeIndex")

        # Aggregate forcing to daily
        daily = pd.DataFrame(index=df.resample("1D").mean().index)
        daily["precipitation_mm"] = df.get("precipitation", df.get(
            "precipitation_mm", 0.0)).resample("1D").sum()
        daily["et0_mm"] = df.get("evapotranspiration", df.get(
            "et0_mm", 0.0)).resample("1D").sum()
        daily["irrigation_mm"] = df.get("irrigation", df.get(
            "irrigation_mm", 0.0)).resample("1D").sum()
        daily["ndvi"] = df.get("ndvi", np.nan).resample("1D").mean()

        psi_init = float(df.iloc[0].get("psi_observed_kpa", -50.0))
        self.physics_model.reset(psi_init)

        outputs = []
        for day, row in daily.iterrows():
            out = self.physics_model.step(
                current_date=day.date(),
                precipitation_mm=float(row.get("precipitation_mm", 0.0)),
                et0_mm=float(row.get("et0_mm", 0.0)),
                ndvi=(None if not np.isfinite(row.get("ndvi", np.nan))
                      else float(row.get("ndvi"))),
                irrigation_mm=float(row.get("irrigation_mm", 0.0)),
            )
            outputs.append({
                "psi_surface_kpa": out.psi_surface_kpa,
                "psi_root_kpa": out.psi_root_kpa,
                "drainage_physics_mm": out.drainage_mm,
                "runoff_physics_mm": out.runoff_mm,
            })

        daily_out = pd.DataFrame(outputs, index=daily.index)
        # Expand daily outputs back to the original index for feature creation
        return daily_out.reindex(df.index, method="ffill")

    def _run_physics_forecast(
        self,
        current_state: Dict[str, float],
        weather_forecast: pd.DataFrame,
        irrigation_schedule: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Run physics model forward with forecast weather.

        Expects a DatetimeIndex; aggregates to daily forcings.
        """
        if not isinstance(weather_forecast.index, pd.DatetimeIndex):
            raise ValueError("weather_forecast must have a DatetimeIndex")

        # Aggregate to daily
        daily = pd.DataFrame(
            index=weather_forecast.resample("1D").mean().index)
        daily["precipitation_mm"] = weather_forecast.get(
            "precipitation", weather_forecast.get("precipitation_mm", 0.0)).resample("1D").sum()
        daily["et0_mm"] = weather_forecast.get(
            "evapotranspiration", weather_forecast.get("et0_mm", 0.0)).resample("1D").sum()
        daily["ndvi"] = weather_forecast.get(
            "ndvi", np.nan).resample("1D").mean()

        # Daily irrigation schedule (optional)
        daily_irrig = None
        if irrigation_schedule is not None and isinstance(irrigation_schedule.index, pd.DatetimeIndex):
            daily_irrig = irrigation_schedule["amount_mm"].resample("1D").sum()

        psi_current = float(current_state.get("psi_kpa", -50.0))
        self.physics_model.reset(psi_current)

        rows = []
        for day, row in daily.iterrows():
            irrig_mm = float(
                daily_irrig.loc[day]) if daily_irrig is not None and day in daily_irrig.index else 0.0
            out = self.physics_model.step(
                current_date=day.date(),
                precipitation_mm=float(row.get("precipitation_mm", 0.0)),
                et0_mm=float(row.get("et0_mm", 0.0)),
                ndvi=(None if not np.isfinite(row.get("ndvi", np.nan))
                      else float(row.get("ndvi"))),
                irrigation_mm=irrig_mm,
            )
            rows.append({
                "datetime": day,
                "psi_physics_kpa": out.psi_root_kpa,
                "psi_surface_kpa": out.psi_surface_kpa,
            })

        return pd.DataFrame(rows).set_index("datetime")

    def _prepare_forecast_features(
        self,
        current_state: Dict[str, float],
        weather_forecast: pd.DataFrame,
        physics_forecast: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare feature matrix for ML predictions."""
        # Start with weather forecast
        features = weather_forecast.copy()

        # Add physics forecast
        for col in physics_forecast.columns:
            features[col] = physics_forecast[col]

        # Add current state as lag features
        features["psi_lag_0h"] = current_state.get("psi_kpa", -50.0)

        # Engineer additional features
        features = self.feature_engineer.create_features(
            features, psi_col="psi_physics_kpa"
        )

        return features

    def _forecast_horizon(
        self,
        horizon: int,
        features_df: pd.DataFrame,
        physics_forecast: pd.DataFrame,
    ) -> PredictionResult:
        """Generate forecast for a specific horizon."""
        # Find the forecast timestamp
        if horizon == 0:
            # Current state (nowcast)
            target_time = features_df.index[0]
        else:
            target_time = features_df.index[0] + timedelta(hours=horizon)

        # Get the row closest to target time
        if target_time not in features_df.index:
            # Find nearest
            time_diffs = abs(features_df.index - target_time)
            nearest_idx = time_diffs.argmin()
            target_time = features_df.index[nearest_idx]

        row = features_df.loc[[target_time]]
        physics_pred = physics_forecast.loc[target_time, "psi_physics_kpa"]

        if self.hybrid_model is not None and self.config.use_hybrid:
            # Use hybrid model
            ml_output = self.hybrid_model.predict(
                row, physics_col="psi_physics_kpa", horizon=horizon
            )

            psi_pred = ml_output["prediction"][0]
            psi_std = ml_output.get("std", [5.0])[0]  # Default uncertainty
            psi_lower = ml_output.get("quantile_10", [psi_pred - 2*psi_std])[0]
            psi_upper = ml_output.get("quantile_90", [psi_pred + 2*psi_std])[0]
        else:
            # Physics only
            psi_pred = physics_pred
            psi_std = 10.0  # Higher uncertainty for physics only
            psi_lower = psi_pred - 20
            psi_upper = psi_pred + 20

        psi_pred_f = float(psi_pred)
        psi_phys_f = float(physics_pred)
        psi_std_f = float(max(0.0, psi_std))
        conf = float(np.clip(1.0 - (psi_std_f / 200.0), 0.0, 1.0))

        result = PredictionResult(
            timestamp=datetime.now(),
            horizon_hours=int(horizon),
            psi_predicted_kpa=MatricPotential(psi_pred_f),
            psi_lower_bound_kpa=MatricPotential(float(psi_lower)),
            psi_upper_bound_kpa=MatricPotential(float(psi_upper)),
            psi_physics_kpa=MatricPotential(psi_phys_f),
            psi_ml_residual_kpa=float(psi_pred_f - psi_phys_f),
            status=SoilMoistureStatus.from_potential(psi_pred_f),
            confidence=conf,
            uncertainty_kpa=psi_std_f,
            model_version="swpps-forecaster-1.0",
            theta_predicted=water_content_from_potential(
                psi_pred_f, self.vg_params),
        )
        return result


def create_forecaster(
    soil_texture: str,
    root_depth_m: float = 0.30,
    latitude: float = 0.0,
    use_hybrid: bool = True,
) -> SoilWaterForecaster:
    """
    Factory function to create a forecaster with appropriate parameters.

    Args:
        soil_texture: Soil texture class (e.g., "loam", "clay", "sand")
        root_depth_m: Root zone depth in meters
        latitude: Site latitude (for temporal features)
        use_hybrid: Whether to use hybrid physics-ML model

    Returns:
        Configured SoilWaterForecaster instance
    """
    from swpps.physics.van_genuchten import estimate_van_genuchten_params

    texture = (soil_texture or "loam").lower()
    texture_map = {
        "sand": (90.0, 5.0),
        "loamy_sand": (80.0, 8.0),
        "sandy_loam": (65.0, 12.0),
        "loam": (40.0, 20.0),
        "silt_loam": (20.0, 15.0),
        "clay_loam": (30.0, 35.0),
        "clay": (20.0, 50.0),
    }
    sand_pct, clay_pct = texture_map.get(texture, (40.0, 20.0))
    vg_params = estimate_van_genuchten_params(
        sand_percent=sand_pct,
        clay_percent=clay_pct,
        organic_matter_percent=2.0,
    )

    # Create configuration
    config = ForecastConfig(
        root_depth_m=root_depth_m,
        use_hybrid=use_hybrid,
    )

    return SoilWaterForecaster(vg_params, config)
