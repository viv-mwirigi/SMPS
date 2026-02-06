"""
Main Pipeline for Soil Water Potential Prediction System (SWPPS).

This module provides the complete end-to-end pipeline for:
1. Data collection from sensors and weather APIs
2. Physics-based water balance modeling
3. ML residual learning
4. Multi-horizon forecasting
5. Irrigation decision making
6. Actuation control

The key innovation is using matric potential (kPa) as the primary
prediction target, enabling universal irrigation thresholds that
work across all soil types without soil-specific calibration.
"""

import logging
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from swpps.core.types import (
    IrrigationDecision,
    MatricPotential,
    PredictionResult,
    SoilMoistureStatus,
    VanGenuchtenParams,
)
from swpps.core.config import SWPPSConfig, load_config, save_config
from swpps.core.exceptions import ConfigurationError, SWPPSError
from swpps.physics.van_genuchten import estimate_van_genuchten_params
from swpps.physics.water_balance import (
    TensionSpaceWaterBalance,
    WaterBalanceConfig,
    LayerConfig,
)
from swpps.data.weather import OpenMeteoClient, WeatherFetchRequest
from swpps.data.sensors import SensorDataManager
from swpps.core.config import PlotConfig, SensorConfig
from swpps.features.engineering import FeatureEngineer, FeatureConfig
from swpps.ml.hybrid_model import HybridTensionModel, HybridModelConfig
from swpps.prediction.forecaster import SoilWaterForecaster, ForecastConfig
from swpps.prediction.decision import IrrigationDecisionEngine, DecisionConfig
from swpps.actuation.irrigation import IrrigationActuator, ActuatorConfig

logger = logging.getLogger("swpps.pipeline")


@dataclass
class PipelineConfig:
    """Complete configuration for SWPPS pipeline."""

    # Site configuration
    site_id: str = "plot_1"
    latitude: float = 0.0
    longitude: float = 0.0

    # Soil configuration
    soil_texture: str = "loam"
    sand_fraction: Optional[float] = None
    clay_fraction: Optional[float] = None
    organic_matter: float = 2.0

    # Crop configuration
    crop_type: str = "generic"
    root_depth_m: float = 0.30

    # WaziGate configuration
    gateway_url: str = "http://localhost"
    gateway_port: int = 880
    device_id: str = ""
    tension_sensor_ids: List[str] = field(default_factory=lambda: ["tension"])

    # Operation mode
    training_enabled: bool = True
    prediction_enabled: bool = True
    actuation_enabled: bool = False  # Disabled by default for safety

    # Timing
    # Spec: update predictions every 6 hours as new data arrives
    prediction_interval_minutes: int = 360
    # Spec: dynamic retraining every 2–10 days
    retrain_min_days: int = 2
    retrain_max_days: int = 10
    retrain_base_days: int = 7

    # Paths
    model_dir: Path = field(default_factory=lambda: Path("./models"))
    data_dir: Path = field(default_factory=lambda: Path("./data"))

    # Forecast horizons
    horizons: List[int] = field(default_factory=lambda: [0, 6, 24, 72, 168])


class SWPPSPipeline:
    """
    Complete Soil Water Potential Prediction System pipeline.

    This is the main orchestrator that:
    1. Collects data from sensors and weather APIs
    2. Runs the physics water balance model
    3. Applies ML bias correction
    4. Generates multi-horizon forecasts
    5. Makes irrigation decisions
    6. Triggers actuation (if enabled)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Initialize components
        self._init_soil_parameters()
        self._init_data_clients()
        self._init_physics_model()
        self._init_ml_model()
        self._init_forecaster()
        self._init_decision_engine()
        self._init_actuator()

        # State tracking
        self.last_prediction_time: Optional[datetime] = None
        self.last_training_time: Optional[datetime] = None
        self.current_predictions: Dict[int, PredictionResult] = {}
        self.current_decision: Optional[IrrigationDecision] = None

        # Thread control
        self._stop_event = threading.Event()
        self._prediction_thread: Optional[threading.Thread] = None
        self._training_thread: Optional[threading.Thread] = None

        logger.info("SWPPS Pipeline initialized for site '%s'", config.site_id)

    def _init_soil_parameters(self) -> None:
        """Initialize Van Genuchten parameters for soil."""
        def _to_percent(x: float) -> float:
            # Accept either 0–1 fractions or 0–100 percentages.
            return float(x * 100.0) if 0.0 <= x <= 1.0 else float(x)

        if self.config.sand_fraction is not None and self.config.clay_fraction is not None:
            sand_pct = _to_percent(self.config.sand_fraction)
            clay_pct = _to_percent(self.config.clay_fraction)
        else:
            texture = (self.config.soil_texture or "loam").lower().replace(
                " ", "_")
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

        self.vg_params = estimate_van_genuchten_params(
            sand_percent=float(sand_pct),
            clay_percent=float(clay_pct),
            organic_matter_percent=float(self.config.organic_matter),
        )

        logger.info("Soil parameters: θs=%.3f, θr=%.3f, α=%.4f, n=%.3f",
                    self.vg_params.theta_s, self.vg_params.theta_r,
                    self.vg_params.alpha, self.vg_params.n)

    def _init_data_clients(self) -> None:
        """Initialize data collection clients."""
        # Weather client (OpenMeteoClient takes a cache_dir, location is in request)
        self.weather_client = OpenMeteoClient(
            cache_dir=self.config.data_dir / "cache" / "weather"
        )

        # Sensor manager
        if self.config.device_id and self.config.tension_sensor_ids:
            plot_config = PlotConfig(
                plot_id=1,
                name=self.config.site_id,
                latitude=self.config.latitude,
                longitude=self.config.longitude,
                crop_type=self.config.crop_type,
                soil_type=self.config.soil_texture,
                moisture_sensors=[
                    SensorConfig(
                        device_id=self.config.device_id,
                        sensor_id=sensor_id,
                        sensor_type="tension",
                        unit="cbar",
                    )
                    for sensor_id in self.config.tension_sensor_ids
                ],
            )

            api_url = str(self.config.gateway_url).rstrip("/")
            if self.config.gateway_port and ":" not in api_url.split("//")[-1]:
                api_url = f"{api_url}:{self.config.gateway_port}"
            if not api_url.startswith("http"):
                api_url = f"http://{api_url}"

            self.sensor_manager = SensorDataManager(
                plot_config=plot_config,
                api_url=api_url,
                token=None,
            )
        else:
            self.sensor_manager = None
            logger.warning(
                "Sensor manager not configured (device_id or sensor_ids missing)")

    def _init_physics_model(self) -> None:
        """Initialize physics water balance model."""
        # Build a minimal 1-layer root-zone model (0 .. root_depth_m).
        layers = [
            LayerConfig(
                depth_top_m=0.0,
                depth_bottom_m=float(self.config.root_depth_m),
                van_genuchten=self.vg_params,
            )
        ]
        wb_config = WaterBalanceConfig(
            layers=layers,
            initial_psi_kpa=-50.0,
        )
        self.physics_model = TensionSpaceWaterBalance(wb_config)

    def _init_ml_model(self) -> None:
        """Initialize or load ML model."""
        self.ml_config = HybridModelConfig()

        model_path = self.config.model_dir / \
            f"{self.config.site_id}_hybrid_model"
        if model_path.exists():
            try:
                self.hybrid_model = HybridTensionModel.load(
                    model_path,
                    physics_model=self.physics_model
                )
                logger.info("Loaded existing ML model from %s", model_path)
            except Exception as e:
                logger.warning("Could not load ML model: %s", str(e))
                self.hybrid_model = None
        else:
            self.hybrid_model = None

    def _init_forecaster(self) -> None:
        """Initialize forecaster."""
        forecast_config = ForecastConfig(
            horizons=self.config.horizons,
            root_depth_m=self.config.root_depth_m,
        )
        self.forecaster = SoilWaterForecaster(self.vg_params, forecast_config)

    def _init_decision_engine(self) -> None:
        """Initialize irrigation decision engine."""
        decision_config = DecisionConfig(
            crop_type=self.config.crop_type,
            irrigation_enabled=self.config.actuation_enabled,
            vg_params=self.vg_params,
            root_depth_m=self.config.root_depth_m,
        )
        self.decision_engine = IrrigationDecisionEngine(decision_config)

    def _init_actuator(self) -> None:
        """Initialize irrigation actuator."""
        if self.config.actuation_enabled and self.config.device_id:
            actuator_config = ActuatorConfig(
                gateway_url=self.config.gateway_url,
                gateway_port=self.config.gateway_port,
                device_id=self.config.device_id,
            )
            self.actuator = IrrigationActuator(actuator_config)
        else:
            self.actuator = None

    # -------------------------------------------------------------------------
    # Main Operations
    # -------------------------------------------------------------------------

    def run_prediction_cycle(self) -> Dict[str, Any]:
        """
        Run a single prediction cycle.

        This is the core operation that:
        1. Gets current sensor readings
        2. Fetches weather data/forecasts
        3. Runs physics model
        4. Applies ML correction
        5. Generates forecasts
        6. Makes irrigation decision

        Returns:
            Dictionary with predictions and decision
        """
        logger.info("Running prediction cycle")
        cycle_start = datetime.now()

        result = {
            "timestamp": cycle_start.isoformat(),
            "site_id": self.config.site_id,
            "success": False,
        }

        try:
            current_state = self._get_current_state()
            result["current_state"] = current_state

            # Step 2: Fetch weather
            weather_data = self._fetch_weather()

            # Step 3: Generate forecasts
            forecasts = self._generate_forecasts(current_state, weather_data)
            result["forecasts"] = {
                h: {
                    "prediction_kpa": float(p.psi_predicted_kpa),
                    "uncertainty_kpa": float(p.uncertainty_kpa),
                    "lower_kpa": float(p.psi_lower_bound_kpa),
                    "upper_kpa": float(p.psi_upper_bound_kpa),
                }
                for h, p in forecasts.items()
            }
            self.current_predictions = forecasts

            # Step 4: Make irrigation decision
            current_psi = MatricPotential(current_state.get("psi_kpa", -50.0))
            decision = self.decision_engine.evaluate(
                current_psi=current_psi,
                forecasts=forecasts,
                current_time=cycle_start,
            )
            result["decision"] = {
                "should_irrigate": decision.should_irrigate,
                "action": decision.action,
                "amount_mm": decision.amount_mm,
                "reason": decision.reason,
                "status": decision.status,
            }
            self.current_decision = decision

            # Step 5: Execute actuation if needed
            if decision.should_irrigate and self.actuator:
                actuation_result = self.actuator.trigger_irrigation(decision)
                result["actuation"] = actuation_result

            result["success"] = True
            self.last_prediction_time = cycle_start

        except Exception as e:
            logger.error("Prediction cycle failed: %s", str(e), exc_info=True)
            result["error"] = str(e)

        result["duration_seconds"] = (
            datetime.now() - cycle_start).total_seconds()
        return result

    def _get_current_state(self) -> Dict[str, float]:
        """Get current state from sensors."""
        state = {}

        if self.sensor_manager:
            try:
                psi = self.sensor_manager.get_current_potential()
                if psi is not None:
                    state["psi_kpa"] = float(psi)
                    state["status"] = SoilMoistureStatus.from_potential(
                        psi).value
            except Exception as e:
                logger.warning("Could not read sensors: %s", str(e))

        # Default if no sensor data
        if "psi_kpa" not in state:
            state["psi_kpa"] = -50.0  # Reasonable default
            state["status"] = "optimal"

        return state

    def _fetch_weather(self) -> pd.DataFrame:
        """Fetch weather data and forecasts."""
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=7)  # recent history for features

        req = WeatherFetchRequest(
            latitude=self.config.latitude,
            longitude=self.config.longitude,
            start_date=start_dt.date(),
            end_date=(end_dt + timedelta(days=7)).date(),
            timezone="UTC",
        )
        daily_list = self.weather_client.fetch_daily_weather(req)
        df = pd.DataFrame([
            {
                "date": w.date,
                "precipitation_sum": w.precipitation_mm,
                "et0_fao_evapotranspiration": w.et0_mm,
                "temperature_2m_mean": w.temperature_mean_c,
                "relative_humidity_2m_mean": w.relative_humidity_mean,
                "shortwave_radiation_sum": w.solar_radiation_mj_m2,
                "wind_speed_10m_mean": w.wind_speed_m_s,
            }
            for w in daily_list
        ])
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    def _generate_forecasts(
        self,
        current_state: Dict[str, float],
        weather_data: pd.DataFrame,
    ) -> Dict[int, PredictionResult]:
        """Generate forecasts for all horizons."""
        # Prepare data for physics model
        physics_forecast = self._run_physics_forecast(
            current_state, weather_data)

        # If hybrid model available, apply ML correction
        if self.hybrid_model and self.hybrid_model.is_fitted:
            return self._generate_hybrid_forecasts(
                current_state, weather_data, physics_forecast
            )

        # Physics-only forecasts
        return self._generate_physics_forecasts(current_state, physics_forecast)

    def _run_physics_forecast(
        self,
        current_state: Dict[str, float],
        weather_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run physics model forward."""
        psi_init = float(current_state.get("psi_kpa", -50.0))
        self.physics_model.reset(psi_init)

        rows = []
        for idx, row in weather_data.iterrows():
            out = self.physics_model.step(
                current_date=idx.date(),
                precipitation_mm=float(row.get("precipitation_sum", 0.0)),
                et0_mm=float(row.get("et0_fao_evapotranspiration", 0.0)),
                ndvi=float(row.get("ndvi")) if "ndvi" in weather_data.columns and pd.notna(
                    row.get("ndvi")) else None,
                irrigation_mm=0.0,
            )
            rows.append({
                "date": idx,
                "psi_physics_kpa": out.psi_root_kpa,
                "psi_surface_kpa": out.psi_surface_kpa,
            })

        return pd.DataFrame(rows).set_index("date")

    def _generate_physics_forecasts(
        self,
        current_state: Dict[str, float],
        physics_df: pd.DataFrame,
    ) -> Dict[int, PredictionResult]:
        """Generate forecasts from physics model only."""
        forecasts = {}
        now = datetime.now()

        for horizon in self.config.horizons:
            if horizon == 0:
                psi_pred = float(current_state.get("psi_kpa", -50.0))
                psi_phys = psi_pred
            else:
                days_ahead = max(0, int(horizon // 24))
                if len(physics_df) == 0:
                    psi_phys = float(current_state.get("psi_kpa", -50.0))
                elif days_ahead < len(physics_df):
                    psi_phys = float(
                        physics_df.iloc[days_ahead]["psi_physics_kpa"])
                else:
                    psi_phys = float(physics_df.iloc[-1]["psi_physics_kpa"])
                psi_pred = psi_phys

            std = float(10.0 + horizon * 0.1)
            lower = float(psi_pred - 2 * std)
            upper = float(psi_pred + 2 * std)

            forecasts[horizon] = PredictionResult(
                timestamp=now,
                horizon_hours=int(horizon),
                psi_predicted_kpa=MatricPotential(psi_pred),
                psi_lower_bound_kpa=MatricPotential(lower),
                psi_upper_bound_kpa=MatricPotential(upper),
                psi_physics_kpa=MatricPotential(psi_phys),
                psi_ml_residual_kpa=0.0,
                status=SoilMoistureStatus.from_potential(psi_pred),
                confidence=float(np.clip(1.0 - (std / 200.0), 0.0, 1.0)),
                uncertainty_kpa=std,
                model_version="swpps-physics-1.0",
            )

        return forecasts

    def get_prediction_curve(self) -> pd.DataFrame:
        """Return a time-based prediction curve for UI/API consumption.

        Columns include the forecasted ψ curve plus bounds and the trigger threshold.
        """
        if not self.current_predictions:
            return pd.DataFrame()

        now = datetime.now()
        rows = []
        for h, p in sorted(self.current_predictions.items(), key=lambda kv: kv[0]):
            rows.append({
                "timestamp": now + timedelta(hours=int(h)),
                "horizon_hours": int(h),
                "psi_predicted_kpa": float(p.psi_predicted_kpa),
                "psi_lower_kpa": float(p.psi_lower_bound_kpa),
                "psi_upper_kpa": float(p.psi_upper_bound_kpa),
                "trigger_threshold_kpa": float(self.decision_engine.trigger_threshold),
            })
        return pd.DataFrame(rows).set_index("timestamp")

    def _generate_hybrid_forecasts(
        self,
        current_state: Dict[str, float],
        weather_data: pd.DataFrame,
        physics_df: pd.DataFrame,
    ) -> Dict[int, PredictionResult]:
        """Generate forecasts using hybrid physics-ML model."""
        # Use forecaster which handles hybrid predictions
        return self.forecaster.forecast(
            current_state=current_state,
            weather_forecast=weather_data,
        )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train_model(
        self,
        training_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Train or retrain the ML model.

        Args:
            training_data: Optional training data. If not provided,
                          will fetch historical data.

        Returns:
            Training result summary
        """
        logger.info("Starting model training")
        start_time = datetime.now()

        try:
            # Get training data if not provided
            if training_data is None:
                training_data = self._prepare_training_data()

            if training_data is None or len(training_data) < 100:
                return {
                    "success": False,
                    "error": "Insufficient training data",
                }

            # Train forecaster (which includes hybrid model)
            self.forecaster.train(
                training_data,
                psi_obs_col="psi_observed_kpa",
            )

            # Save model
            model_path = self.config.model_dir / \
                f"{self.config.site_id}_hybrid_model"
            model_path.mkdir(parents=True, exist_ok=True)
            if self.forecaster.hybrid_model:
                self.forecaster.hybrid_model.save(model_path)

            self.last_training_time = datetime.now()

            return {
                "success": True,
                "samples": len(training_data),
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "model_path": str(model_path),
            }

        except Exception as e:
            logger.error("Training failed: %s", str(e), exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    def _prepare_training_data(self) -> Optional[pd.DataFrame]:
        """Prepare training data from historical observations."""
        # This would fetch historical data from sensors and weather
        # For now, return None to indicate no data available
        logger.warning("Historical data preparation not implemented")
        return None

    # -------------------------------------------------------------------------
    # Continuous Operation
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Start continuous prediction and training loops."""
        logger.info("Starting SWPPS pipeline")
        self._stop_event.clear()

        if self.config.prediction_enabled:
            self._prediction_thread = threading.Thread(
                target=self._prediction_loop,
                name="swpps-prediction",
                daemon=True,
            )
            self._prediction_thread.start()

        if self.config.training_enabled:
            self._training_thread = threading.Thread(
                target=self._training_loop,
                name="swpps-training",
                daemon=True,
            )
            self._training_thread.start()

    def stop(self) -> None:
        """Stop all pipeline threads."""
        logger.info("Stopping SWPPS pipeline")
        self._stop_event.set()

        if self._prediction_thread:
            self._prediction_thread.join(timeout=30)
        if self._training_thread:
            self._training_thread.join(timeout=30)

    def _prediction_loop(self) -> None:
        """Continuous prediction loop."""
        interval = self.config.prediction_interval_minutes * 60

        while not self._stop_event.is_set():
            try:
                self.run_prediction_cycle()
            except Exception as e:
                logger.error("Prediction loop error: %s", str(e))

            self._stop_event.wait(interval)

    def _training_loop(self) -> None:
        """Continuous training loop."""
        # Initial delay
        self._stop_event.wait(60)

        last_success = True

        while not self._stop_event.is_set():
            try:
                result = self.train_model()
                last_success = bool(result.get("success"))
            except Exception as e:
                logger.error("Training loop error: %s", str(e))
                last_success = False

            # Dynamic interval in [min, max] days.
            base_days = int(self.config.retrain_base_days)
            if not last_success:
                base_days = int(self.config.retrain_min_days)

            days = int(np.clip(base_days, self.config.retrain_min_days,
                       self.config.retrain_max_days))
            self._stop_event.wait(days * 24 * 3600)

    # -------------------------------------------------------------------------
    # Status and Configuration
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "site_id": self.config.site_id,
            "crop_type": self.config.crop_type,
            "prediction_enabled": self.config.prediction_enabled,
            "training_enabled": self.config.training_enabled,
            "actuation_enabled": self.config.actuation_enabled,
            "last_prediction": self.last_prediction_time.isoformat()
            if self.last_prediction_time else None,
            "last_training": self.last_training_time.isoformat()
            if self.last_training_time else None,
            "current_predictions": {
                h: float(p.prediction_kpa)
                for h, p in self.current_predictions.items()
            } if self.current_predictions else {},
            "current_decision": {
                "should_irrigate": self.current_decision.should_irrigate,
                "action": self.current_decision.action,
                "status": self.current_decision.status,
            } if self.current_decision else None,
            "model_fitted": self.forecaster.is_fitted if self.forecaster else False,
        }

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update pipeline configuration."""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Re-initialize affected components
        if "crop_type" in updates:
            self.decision_engine.set_crop(updates["crop_type"])

        if any(k in updates for k in ["soil_texture", "sand_fraction", "clay_fraction"]):
            self._init_soil_parameters()
            self._init_physics_model()


def create_pipeline_from_config(config_path: Path) -> SWPPSPipeline:
    """
    Create pipeline from configuration file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Configured SWPPSPipeline instance
    """
    with open(config_path) as f:
        config_dict = json.load(f)

    config = PipelineConfig(**config_dict)
    return SWPPSPipeline(config)


def create_pipeline(
    site_id: str,
    latitude: float,
    longitude: float,
    device_id: str = "",
    crop_type: str = "generic",
    soil_texture: str = "loam",
    actuation_enabled: bool = False,
) -> SWPPSPipeline:
    """
    Factory function to create a pipeline with minimal configuration.

    Args:
        site_id: Unique identifier for the site/plot
        latitude: Site latitude
        longitude: Site longitude
        device_id: WaziGate device ID
        crop_type: Crop type for thresholds
        soil_texture: Soil texture class
        actuation_enabled: Whether to enable irrigation actuation

    Returns:
        Configured SWPPSPipeline instance
    """
    config = PipelineConfig(
        site_id=site_id,
        latitude=latitude,
        longitude=longitude,
        device_id=device_id,
        crop_type=crop_type,
        soil_texture=soil_texture,
        actuation_enabled=actuation_enabled,
    )
    return SWPPSPipeline(config)
