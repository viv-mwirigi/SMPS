"""
Feature Engineering for Matric Potential Prediction.

This module creates features from sensor data and weather forecasts
that are meaningful for predicting soil water tension dynamics.

Key feature categories:
1. Lag features - Past soil water status
2. Weather aggregates - Recent and forecast conditions
3. Water balance indices - Physics-derived features
4. Temporal features - Seasonality and timing
5. Stress indices - Plant water stress indicators
6. Gradient features - Rate of change and acceleration
7. Physics-informed features - ψ, θ, K(θ), drainage, infiltration from physics

Enhanced for African/tropical agriculture context.

The key innovation is giving ML the physics model's perspective:
"Physics thinks ψ should change like this — should it?"
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from swpps.core.types import VanGenuchtenParams
from swpps.physics.van_genuchten import (
    potential_from_water_content,
    specific_water_capacity,
    water_content_from_potential,
)

logger = logging.getLogger("swpps.features.engineering")


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # Lag features
    lag_hours: List[int] = None  # e.g., [6, 12, 24, 48, 72]

    # Rolling window features
    rolling_windows: List[int] = None  # e.g., [24, 72, 168]

    # Weather forecast horizons
    forecast_horizons: List[int] = None  # e.g., [24, 48, 72]

    # Physics features to include
    include_physics: bool = True

    # Full physics-informed features (ψ, θ, K(θ), drainage, ET stress)
    include_physics_informed: bool = True

    # Temporal features
    include_temporal: bool = True

    # Advanced features
    include_stress_indices: bool = True
    include_gradient_features: bool = True
    include_cross_features: bool = True

    # Hemisphere for seasonal features (True = Northern, False = Southern)
    northern_hemisphere: bool = True

    # Crop-specific thresholds
    critical_psi_kpa: float = -50.0  # MAD threshold
    stress_psi_kpa: float = -100.0   # Stress onset
    wilting_psi_kpa: float = -1500.0  # Permanent wilting

    # Soil hydraulic parameters (for physics features)
    # These can be overridden with site-specific values
    vg_params: Optional[VanGenuchtenParams] = None
    Ksat_mm_day: float = 100.0  # Saturated hydraulic conductivity
    theta_sat: float = 0.45     # Saturated water content
    theta_res: float = 0.05     # Residual water content
    alpha_vg: float = 0.05      # Van Genuchten α (1/kPa)
    n_vg: float = 1.5           # Van Genuchten n
    lambda_bc: float = 0.5      # Brooks-Corey λ (pore size distribution)
    psi_fc_kpa: float = -33.0   # Field capacity potential
    root_depth_m: float = 0.30  # Root zone depth

    def __post_init__(self):
        if self.lag_hours is None:
            self.lag_hours = [6, 12, 24, 48, 72, 168]
        if self.rolling_windows is None:
            self.rolling_windows = [24, 72, 168]
        if self.forecast_horizons is None:
            self.forecast_horizons = [24, 48, 72]


class FeatureEngineer:
    """
    Creates features for soil water potential prediction.

    Features are designed around the water balance concept:
    - Inputs: Precipitation, irrigation
    - Outputs: Evapotranspiration, drainage
    - State: Current soil water status
    """

    def __init__(self, config: Optional[FeatureConfig] = None, *, vg_params: Optional[VanGenuchtenParams] = None):
        self.config = config or FeatureConfig()
        if vg_params is not None:
            self.config.vg_params = vg_params
        self.feature_names: List[str] = []

    def _get_vg_params(self) -> VanGenuchtenParams:
        """Return the Van Genuchten params to use for θ↔ψ conversions.

        Preference order:
        1) Explicit `FeatureConfig.vg_params` (keeps train/infer consistent)
        2) Constructed from scalar config fields (legacy/default behavior)
        """
        if self.config.vg_params is not None:
            return self.config.vg_params

        return VanGenuchtenParams(
            theta_r=float(self.config.theta_res),
            theta_s=float(self.config.theta_sat),
            alpha=float(self.config.alpha_vg),
            n=float(self.config.n_vg),
            K_sat=float(self.config.Ksat_mm_day),
        )

    def create_features(
        self,
        df: pd.DataFrame,
        psi_col: str = "psi_kpa",
        weather_cols: Optional[List[str]] = None,
        physics_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Create all features from input data.

        Args:
            df: Input dataframe with timestamps as index
            psi_col: Column name for matric potential observations
            weather_cols: Weather variable column names
            physics_cols: Physics model output column names

        Returns:
            DataFrame with engineered features
        """
        features = df.copy()

        # Default column sets
        if weather_cols is None:
            weather_cols = [
                "temperature_2m", "relative_humidity_2m",
                "precipitation", "evapotranspiration",
                "wind_speed_10m", "shortwave_radiation",
            ]

        if physics_cols is None:
            physics_cols = [
                "psi_physics_kpa", "psi_physics_root_kpa",
                "et_physics", "drainage_physics", "runoff_physics",
            ]

        # Create feature groups
        self._add_lag_features(features, psi_col)
        self._add_weather_rolling_features(features, weather_cols)
        self._add_water_balance_indices(features, weather_cols)

        if self.config.include_physics:
            self._add_physics_features(features, physics_cols)
            # Add physics-informed features for ML learning
            self._add_infiltration_physics_features(features, psi_col)
            self._add_drainage_physics_features(features, psi_col)

        if self.config.include_physics_informed:
            # Comprehensive physics-informed features
            self._add_psi_physics_state_features(
                features, psi_col, physics_cols)
            self._add_hydraulic_conductivity_features(features, psi_col)
            self._add_et_stress_features(features, psi_col, weather_cols)
            self._add_mass_balance_diagnostics(features, psi_col, weather_cols)

        if self.config.include_temporal:
            self._add_temporal_features(features)

        if self.config.include_stress_indices:
            self._add_stress_indices(features, psi_col, weather_cols)

        if self.config.include_gradient_features:
            self._add_gradient_features(features, psi_col)

        if self.config.include_cross_features:
            self._add_cross_features(features, psi_col, weather_cols)

        # Add physical soil features (prevents geographic fingerprinting)
        self._add_physical_soil_features(features)

        # Add LAI from NDVI for irrigation applications
        self._add_lai_from_ndvi(features)

        # Store feature names
        self.feature_names = [
            c for c in features.columns if c not in df.columns]

        return features

    def _add_lag_features(self, df: pd.DataFrame, psi_col: str) -> None:
        """Add lagged soil water potential features."""
        if psi_col not in df.columns:
            logger.warning(
                "Potential column '%s' not found, skipping lag features", psi_col)
            return

        for lag in self.config.lag_hours:
            df[f"psi_lag_{lag}h"] = df[psi_col].shift(lag)

        # Change features (trend detection)
        for window in [6, 24, 72]:
            df[f"psi_change_{window}h"] = df[psi_col] - \
                df[psi_col].shift(window)
            df[f"psi_pct_change_{window}h"] = df[f"psi_change_{window}h"] / (
                df[psi_col].shift(window).abs() + 1  # Avoid division by zero
            )

        # Rolling statistics on psi
        for window in self.config.rolling_windows:
            df[f"psi_rolling_mean_{window}h"] = df[psi_col].rolling(
                window).mean()
            df[f"psi_rolling_std_{window}h"] = df[psi_col].rolling(
                window).std()
            df[f"psi_rolling_min_{window}h"] = df[psi_col].rolling(
                window).min()
            df[f"psi_rolling_max_{window}h"] = df[psi_col].rolling(
                window).max()

    def _add_weather_rolling_features(
        self, df: pd.DataFrame, weather_cols: List[str]
    ) -> None:
        """Add rolling aggregates of weather variables."""
        for col in weather_cols:
            if col not in df.columns:
                continue

            for window in self.config.rolling_windows:
                # Sum for flux variables
                if col in ["precipitation", "evapotranspiration", "irrigation"]:
                    df[f"{col}_sum_{window}h"] = df[col].rolling(window).sum()

                # Mean and extremes for state variables
                df[f"{col}_mean_{window}h"] = df[col].rolling(window).mean()

                if col in ["temperature_2m", "relative_humidity_2m"]:
                    df[f"{col}_max_{window}h"] = df[col].rolling(window).max()
                    df[f"{col}_min_{window}h"] = df[col].rolling(window).min()

    def _add_water_balance_indices(
        self, df: pd.DataFrame, weather_cols: List[str]
    ) -> None:
        """Add water balance derived indices."""
        # Precipitation minus ET (simple water balance)
        if "precipitation" in df.columns and "evapotranspiration" in df.columns:
            df["p_minus_et"] = df["precipitation"] - df["evapotranspiration"]

            for window in self.config.rolling_windows:
                df[f"p_minus_et_sum_{window}h"] = df["p_minus_et"].rolling(
                    window).sum()

        # Antecedent precipitation index (exponential decay)
        if "precipitation" in df.columns:
            for tau in [24, 72, 168]:
                alpha = 1 / tau
                df[f"api_{tau}h"] = df["precipitation"].ewm(alpha=alpha).mean()

        # Vapor pressure deficit (atmospheric demand indicator)
        if "temperature_2m" in df.columns and "relative_humidity_2m" in df.columns:
            # Simplified VPD calculation
            T = df["temperature_2m"]
            RH = df["relative_humidity_2m"]
            # Saturated vapor pressure
            es = 0.6108 * np.exp(17.27 * T / (T + 237.3))
            df["vpd_kpa"] = es * (1 - RH / 100)

            # Rolling VPD
            for window in self.config.rolling_windows:
                df[f"vpd_mean_{window}h"] = df["vpd_kpa"].rolling(
                    window).mean()

        # Atmospheric demand stress indicator
        if "shortwave_radiation" in df.columns:
            # High radiation + high VPD = high stress potential
            if "vpd_kpa" in df.columns:
                df["atm_demand_index"] = (
                    df["shortwave_radiation"] / 1000 * df["vpd_kpa"]
                )

        # Days since significant rain
        if "precipitation" in df.columns:
            significant_rain = df["precipitation"] > 2  # mm/h
            groups = (~significant_rain).cumsum()
            hours_since = df.groupby(groups).cumcount()
            df["hours_since_rain"] = hours_since.where(~significant_rain, 0)
            df["days_since_rain"] = df["hours_since_rain"] / 24

    def _add_stress_indices(
        self, df: pd.DataFrame, psi_col: str, weather_cols: List[str]
    ) -> None:
        """Add plant water stress indices."""
        # Relative extractable water proxy
        if psi_col in df.columns:
            psi = df[psi_col].abs()  # Make positive for calculations
            # Normalize between FC (-10 kPa) and PWP (-1500 kPa)
            df["relative_soil_moisture"] = 1 - np.clip(
                (psi - 10) / (1500 - 10), 0, 1
            )

            # Stress flags
            df["stress_flag"] = (
                df[psi_col] < self.config.stress_psi_kpa).astype(float)
            df["critical_flag"] = (
                df[psi_col] < self.config.critical_psi_kpa).astype(float)

            # Cumulative stress hours
            df["stress_hours_cumsum"] = df["stress_flag"].cumsum()
            df["stress_hours_7d"] = df["stress_flag"].rolling(168).sum()

            # Distance from irrigation threshold
            df["psi_to_threshold"] = df[psi_col] - self.config.critical_psi_kpa

        # Crop Water Stress Index (CWSI) proxy
        if "temperature_2m" in df.columns and "vpd_kpa" in df.columns:
            # High temp + high VPD when soil is dry = stress
            if psi_col in df.columns:
                soil_dryness = np.clip(df[psi_col].abs() / 100, 0, 1)
                atm_demand = np.clip(df["vpd_kpa"] / 3, 0, 1)
                df["cwsi_proxy"] = soil_dryness * atm_demand

        # Evaporative Fraction proxy
        if "temperature_2m" in df.columns and "relative_humidity_2m" in df.columns:
            # Higher humidity suggests more evaporation occurring
            df["evap_fraction"] = df["relative_humidity_2m"] / 100

    def _add_gradient_features(self, df: pd.DataFrame, psi_col: str) -> None:
        """Add rate of change and acceleration features."""
        if psi_col not in df.columns:
            return

        psi = df[psi_col]

        # First derivative (rate of change)
        df["psi_rate_1h"] = psi.diff(1)
        df["psi_rate_3h"] = psi.diff(3) / 3
        df["psi_rate_6h"] = psi.diff(6) / 6
        df["psi_rate_12h"] = psi.diff(12) / 12
        df["psi_rate_24h"] = psi.diff(24) / 24

        # Second derivative (acceleration - is drying speeding up or slowing?)
        df["psi_accel_6h"] = df["psi_rate_6h"].diff(6) / 6
        df["psi_accel_24h"] = df["psi_rate_24h"].diff(24) / 24

        # Rate statistics over windows
        for window in [24, 72]:
            df[f"psi_rate_max_{window}h"] = df["psi_rate_1h"].rolling(
                window).max()
            df[f"psi_rate_min_{window}h"] = df["psi_rate_1h"].rolling(
                window).min()
            df[f"psi_rate_std_{window}h"] = df["psi_rate_1h"].rolling(
                window).std()

        # Trend indicators
        df["drying_trend_24h"] = (df["psi_rate_24h"] < -0.5).astype(float)
        df["wetting_trend_24h"] = (df["psi_rate_24h"] > 0.5).astype(float)
        df["stable_trend_24h"] = (
            df["psi_rate_24h"].abs() <= 0.5).astype(float)

    def _add_cross_features(
        self, df: pd.DataFrame, psi_col: str, weather_cols: List[str]
    ) -> None:
        """Add interaction features between variables."""
        # VPD × soil dryness interaction
        if "vpd_kpa" in df.columns and psi_col in df.columns:
            soil_dryness = np.clip(df[psi_col].abs() / 100, 0, 2)
            df["vpd_soil_interaction"] = df["vpd_kpa"] * soil_dryness

        # Radiation × temperature interaction
        if "shortwave_radiation" in df.columns and "temperature_2m" in df.columns:
            df["rad_temp_interaction"] = (
                df["shortwave_radiation"] / 1000 * df["temperature_2m"] / 30
            )

        # Precipitation effectiveness (rain when soil is dry is more effective)
        if "precipitation" in df.columns and psi_col in df.columns:
            # More negative = drier
            soil_deficit = np.clip(-df[psi_col] / 100, 0, 10)
            df["precip_effectiveness"] = df["precipitation"] * \
                np.sqrt(soil_deficit + 0.1)

        # Wind × VPD (wind increases ET when VPD is high)
        if "wind_speed_10m" in df.columns and "vpd_kpa" in df.columns:
            df["wind_vpd_interaction"] = df["wind_speed_10m"] * df["vpd_kpa"]

    def _add_physics_features(
        self, df: pd.DataFrame, physics_cols: List[str]
    ) -> None:
        """Add features from physics model outputs."""
        for col in physics_cols:
            if col not in df.columns:
                continue

            # Lags of physics outputs
            for lag in [24, 72]:
                df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

            # Changes
            df[f"{col}_change_24h"] = df[col] - df[col].shift(24)

        # Physics residual diagnostics (if both physics and observed available)
        psi_physics_col = None
        psi_obs_col = None
        for col in physics_cols:
            if 'physics' in col.lower() and 'psi' in col.lower():
                psi_physics_col = col
        for col in df.columns:
            if 'observed' in col.lower() and 'psi' in col.lower():
                psi_obs_col = col

        if psi_physics_col and psi_physics_col in df.columns:
            if psi_obs_col and psi_obs_col in df.columns:
                self._add_residual_diagnostics(
                    df, psi_obs_col, psi_physics_col)

    def _add_residual_diagnostics(
        self, df: pd.DataFrame, psi_obs_col: str, psi_physics_col: str
    ) -> None:
        """
        Add residual diagnostic features for physics model calibration.

        These features help ML learn what physics is getting wrong:
        - Positive residual (ψ_obs > ψ_phys): Physics predicts too dry
        - Negative residual (ψ_obs < ψ_phys): Physics predicts too wet

        This informs calibration of:
        - Ksat (if infiltration is wrong)
        - ET stress (if drying rate is wrong)
        - Root depth (if response depth is wrong)
        """
        # Basic residual
        df["psi_residual"] = df[psi_obs_col] - df[psi_physics_col]

        # Residual statistics
        for window in [24, 72, 168]:
            df[f"psi_residual_mean_{window}h"] = df["psi_residual"].rolling(
                window).mean()
            df[f"psi_residual_std_{window}h"] = df["psi_residual"].rolling(
                window).std()

        # Systematic bias detection
        df["physics_too_dry"] = (df["psi_residual"] > 10).astype(
            float)  # Physics ψ too negative
        df["physics_too_wet"] = (
            df["psi_residual"] < -10).astype(float)  # Physics ψ too high

        # Bias persistence (indicates systematic calibration error)
        df["dry_bias_streak"] = df["physics_too_dry"].rolling(24).sum()
        df["wet_bias_streak"] = df["physics_too_wet"].rolling(24).sum()

        # Conditional residuals for calibration insights
        if "precipitation" in df.columns:
            # Residual during/after rain → infiltration physics error
            rain_mask = df["precipitation"] > 1
            df["residual_during_rain"] = df["psi_residual"].where(rain_mask)
            df["residual_during_rain"] = df["residual_during_rain"].ffill(
                limit=6)

            # Positive after rain = Ksat too low (water not infiltrating)
            # Negative after rain = Ksat too high (too much infiltration)

        if "evapotranspiration" in df.columns:
            # Residual during high ET → ET stress calibration error
            high_et_mask = df["evapotranspiration"] > df["evapotranspiration"].quantile(
                0.75)
            df["residual_high_et"] = df["psi_residual"].where(high_et_mask)
            df["residual_high_et"] = df["residual_high_et"].ffill(limit=6)

            # Positive during high ET = ET stress coefficient too high
            # Negative during high ET = ET stress coefficient too low

    def _add_infiltration_physics_features(
        self, df: pd.DataFrame, psi_col: str
    ) -> None:
        """
        Add features that help ML learn infiltration physics.

        Key relationships:
        - Infiltration rate depends on initial soil wetness
        - Wetter soil → lower infiltration capacity
        - This creates ψ-dependent infiltration behavior
        """
        if psi_col not in df.columns or "precipitation" not in df.columns:
            return

        psi = df[psi_col]
        precip = df["precipitation"]

        # Soil's infiltration capacity indicator
        # More negative ψ = drier = higher infiltration capacity
        df["infiltration_capacity_index"] = np.clip(-psi / 100, 0, 10)

        # Expected infiltration given soil state
        df["expected_infiltration"] = precip * (1 - np.exp(psi / 50))
        df["expected_infiltration"] = df["expected_infiltration"].clip(lower=0)

        # Infiltration efficiency (how much of rain infiltrates)
        # Higher when soil is dry (negative ψ), lower when wet
        df["infiltration_efficiency"] = 1 / (1 + np.exp(psi / 20 + 1))

        # Ponding potential (when infiltration < rainfall rate)
        # This indicates when runoff might occur
        if precip.max() > 0:
            df["ponding_potential"] = (precip > 5) & (psi > -20)
            df["ponding_potential"] = df["ponding_potential"].astype(float)

    def _add_drainage_physics_features(
        self, df: pd.DataFrame, psi_col: str
    ) -> None:
        """
        Add features that help ML learn drainage physics.

        Key relationships:
        - Drainage occurs when ψ > field capacity (~-10 to -33 kPa)
        - Drainage rate depends on hydraulic conductivity
        - K(ψ) is highly nonlinear near saturation
        """
        if psi_col not in df.columns:
            return

        psi = df[psi_col]

        # Drainage potential (increases as soil gets wetter)
        # Field capacity typically at -10 to -33 kPa
        df["above_fc"] = (psi > -33).astype(float)
        df["drainage_potential"] = np.clip((psi + 33) / 33, 0, 1)

        # Hours above field capacity (drainage duration)
        df["hours_above_fc"] = df["above_fc"].rolling(24).sum()

        # Relative hydraulic conductivity proxy (highly nonlinear)
        # K/Ksat ≈ Se^0.5 * [1-(1-Se^(1/m))^m]^2 where Se = f(ψ)
        # Simplified: K drops rapidly as soil dries
        df["rel_conductivity_proxy"] = np.exp(
            psi / 10)  # Simplified exponential
        df["rel_conductivity_proxy"] = df["rel_conductivity_proxy"].clip(0, 1)

        # Expected drainage rate (higher when wet)
        df["expected_drainage"] = df["drainage_potential"] * \
            df["rel_conductivity_proxy"]

    # =========================================================================
    # COMPREHENSIVE PHYSICS-INFORMED FEATURES
    # These let ML see what physics thinks and learn to correct it
    # =========================================================================

    def _add_psi_physics_state_features(
        self,
        df: pd.DataFrame,
        psi_col: str,
        physics_cols: List[str],
    ) -> None:
        """
        Add core physics state features: ψ_phys, θ_phys, ψ_obs, dψ/dθ.

        These are the fundamental features that let ML understand
        what the physics model is predicting and compare to observations.
        """
        # Find physics psi column
        psi_phys_col = None
        for col in physics_cols:
            if 'psi' in col.lower() and 'physics' in col.lower():
                psi_phys_col = col
                break

        # ψ_phys - matric potential from physics
        if psi_phys_col and psi_phys_col in df.columns:
            df["psi_phys"] = df[psi_phys_col]

            # ψ_phys lags
            for lag in [1, 6, 24]:
                df[f"psi_phys_lag_{lag}h"] = df["psi_phys"].shift(lag)

            # ψ_phys rate of change
            df["psi_phys_rate_1h"] = df["psi_phys"].diff(1)
            df["psi_phys_rate_24h"] = df["psi_phys"].diff(24) / 24

            # θ_phys - volumetric water content from physics (via Van Genuchten)
            df["theta_phys"] = self._psi_to_theta(df["psi_phys"])

            # θ_phys lags
            for lag in [1, 6, 24]:
                df[f"theta_phys_lag_{lag}h"] = df["theta_phys"].shift(lag)

        # ψ_obs (if available) - derived from observed data
        if psi_col in df.columns:
            df["psi_obs"] = df[psi_col]

            # θ_obs - volumetric water content from observations
            df["theta_obs"] = self._psi_to_theta(df["psi_obs"])

            # dψ/dθ - retention curve slope (capacity term)
            # This is crucial for understanding sensitivity
            df["dpsi_dtheta"] = self._retention_curve_slope(df["psi_obs"])

            # Difference between physics and observations
            if "psi_phys" in df.columns:
                df["psi_phys_minus_obs"] = df["psi_phys"] - df["psi_obs"]
                df["theta_phys_minus_obs"] = df["theta_phys"] - df["theta_obs"]

                # Physics vs obs rate comparison
                psi_obs_rate = df["psi_obs"].diff(24) / 24
                df["rate_phys_minus_obs"] = df["psi_phys_rate_24h"] - psi_obs_rate

        # Root zone water - integrated ψ over depth (approximation)
        if "psi_phys" in df.columns:
            # Integrate using trapezoidal assumption over root zone
            root_depth_mm = self.config.root_depth_m * 1000
            df["root_zone_water_mm"] = (
                df["theta_phys"] * root_depth_mm
            )

    def _add_hydraulic_conductivity_features(
        self, df: pd.DataFrame, psi_col: str
    ) -> None:
        """
        Add hydraulic conductivity features K(θ) and K(ψ).

        K is highly nonlinear and drives both infiltration and drainage.
        ML needs to see K to learn drainage physics.

        K(θ) = Ksat × Se^(2+3λ)  [Brooks-Corey]
        where Se = (θ - θr) / (θs - θr) is effective saturation
        """
        Ksat = self.config.Ksat_mm_day
        theta_s = self.config.theta_sat
        theta_r = self.config.theta_res
        lambda_bc = self.config.lambda_bc

        # Compute from observed ψ if available
        if psi_col in df.columns:
            theta = self._psi_to_theta(df[psi_col])

            # Effective saturation Se
            Se = np.clip((theta - theta_r) / (theta_s - theta_r), 0.001, 1.0)
            df["Se_obs"] = Se

            # Hydraulic conductivity K(θ) using Brooks-Corey
            # K = Ksat × Se^(2 + 3λ)
            exponent = 2 + 3 * lambda_bc
            df["K_theta_obs"] = Ksat * (Se ** exponent)

            # Log of K (spans orders of magnitude)
            df["log_K_theta_obs"] = np.log10(
                df["K_theta_obs"].clip(lower=1e-6))

        # Compute from physics θ if available
        if "theta_phys" in df.columns:
            theta = df["theta_phys"]
            Se = np.clip((theta - theta_r) / (theta_s - theta_r), 0.001, 1.0)
            df["Se_phys"] = Se

            exponent = 2 + 3 * lambda_bc
            df["K_theta_phys"] = Ksat * (Se ** exponent)
            df["log_K_theta_phys"] = np.log10(
                df["K_theta_phys"].clip(lower=1e-6))

        # Provide Ksat as a feature (soil property)
        df["Ksat_mm_day"] = Ksat

    def _add_infiltration_comprehensive_features(
        self, df: pd.DataFrame, psi_col: str
    ) -> None:
        """
        Add comprehensive infiltration physics features.

        Infiltration capacity f_infil = min(rain, Ks × (1 + |ψ|))

        This lets ML learn:
        - "Physics overestimates infiltration in clay"
        - "Physics underestimates it in sand"
        """
        Ksat = self.config.Ksat_mm_day
        Ksat_mm_h = Ksat / 24  # Convert to mm/hour

        if psi_col not in df.columns:
            return

        psi = df[psi_col]

        # Provide rain if available
        if "precipitation" in df.columns:
            rain = df["precipitation"]
            df["rain"] = rain

            # Infiltration capacity: increases with drier soil
            # f_infil = Ks × (1 + |ψ|/100)  - normalized
            psi_factor = 1 + np.abs(psi) / 100
            df["infiltration_capacity_mm_h"] = Ksat_mm_h * psi_factor

            # Actual infiltration (limited by rain rate)
            df["infiltration_actual"] = np.minimum(
                rain, df["infiltration_capacity_mm_h"])

            # Infiltration excess (potential runoff)
            df["infiltration_excess"] = np.maximum(
                0, rain - df["infiltration_capacity_mm_h"])

            # Infiltration deficit (how much capacity remains)
            df["infiltration_deficit"] = df["infiltration_capacity_mm_h"] - \
                df["infiltration_actual"]

            # Cumulative rain over windows
            for window in [6, 24, 72]:
                df[f"rain_sum_{window}h"] = rain.rolling(window).sum()

        # Provide Ksat for ML to learn soil-specific corrections
        df["Ksat_mm_h"] = Ksat_mm_h

        # Physics psi if available
        if "psi_phys" in df.columns:
            psi_phys = df["psi_phys"]
            df["psi_phys_for_infil"] = psi_phys

            # Infiltration capacity from physics psi
            psi_phys_factor = 1 + np.abs(psi_phys) / 100
            df["infiltration_capacity_phys"] = Ksat_mm_h * psi_phys_factor

    def _add_drainage_comprehensive_features(
        self, df: pd.DataFrame, psi_col: str
    ) -> None:
        """
        Add comprehensive drainage physics features.

        Gravity drainage flux = K(θ) when θ > θ_fc

        This lets ML learn:
        - "This soil drains too fast"
        - "This soil holds water longer than physics predicts"
        """
        psi_fc = self.config.psi_fc_kpa

        if psi_col not in df.columns:
            return

        psi = df[psi_col]

        # Gravity drainage occurs above field capacity
        above_fc = psi > psi_fc
        df["draining"] = above_fc.astype(float)

        # Drainage flux = K(θ) when above FC, 0 otherwise
        if "K_theta_obs" in df.columns:
            df["gravity_drainage_flux"] = df["K_theta_obs"].where(above_fc, 0)

        # Physics drainage flux
        if "K_theta_phys" in df.columns:
            df["gravity_drainage_flux_phys"] = df["K_theta_phys"].where(
                df["psi_phys"] > psi_fc if "psi_phys" in df.columns else above_fc, 0
            )

        # Time above field capacity (drainage duration)
        df["hours_draining"] = df["draining"].rolling(48).sum()

        # Cumulative drainage potential
        if "gravity_drainage_flux" in df.columns:
            df["cum_drainage_24h"] = df["gravity_drainage_flux"].rolling(
                24).sum()

    def _add_et_stress_features(
        self, df: pd.DataFrame, psi_col: str, weather_cols: List[str]
    ) -> None:
        """
        Add ψ-driven ET stress features.

        ET = ET0 × f(ψ)

        where f(ψ) is the stress function:
        - f(ψ) = 1 when ψ > ψ_critical (no stress)
        - f(ψ) → 0 as ψ → ψ_wilting

        This lets ML learn:
        - "Physics dries too fast at ψ < -100 kPa"
        """
        psi_critical = self.config.stress_psi_kpa  # -100 kPa
        psi_wilting = self.config.wilting_psi_kpa  # -1500 kPa

        # Reference ET
        if "evapotranspiration" in df.columns:
            df["ET_0"] = df["evapotranspiration"]
        elif "et0" in df.columns:
            df["ET_0"] = df["et0"]
        else:
            # Estimate from weather if available
            if "shortwave_radiation" in df.columns and "temperature_2m" in df.columns:
                # Simple Hargreaves-like approximation
                Rn = df["shortwave_radiation"] * 0.0036  # Convert to MJ/m2/h
                T = df["temperature_2m"]
                df["ET_0"] = 0.0023 * Rn * (T + 17.8) * 0.408
                df["ET_0"] = df["ET_0"].clip(lower=0)

        if psi_col in df.columns:
            psi = df[psi_col]

            # Stress function f(ψ) - linear in log space
            # f = 1 at ψ_critical, f = 0 at ψ_wilting
            log_psi = np.log10(np.abs(psi).clip(lower=1))
            log_crit = np.log10(abs(psi_critical))
            log_wilt = np.log10(abs(psi_wilting))

            f_psi = np.clip((log_wilt - log_psi) / (log_wilt - log_crit), 0, 1)
            # Correct for when psi > psi_critical (no stress)
            f_psi = np.where(psi > psi_critical, 1.0, f_psi)

            df["f_psi_stress"] = f_psi

            # Actual ET (stressed)
            if "ET_0" in df.columns:
                df["ET_actual"] = df["ET_0"] * f_psi

                # ET reduction due to stress
                df["ET_stress_reduction"] = df["ET_0"] - df["ET_actual"]

                # Cumulative ET
                for window in [24, 72]:
                    df[f"ET_actual_sum_{window}h"] = df["ET_actual"].rolling(
                        window).sum()

        # Physics stress if available
        if "psi_phys" in df.columns:
            psi_phys = df["psi_phys"]

            log_psi_phys = np.log10(np.abs(psi_phys).clip(lower=1))
            f_psi_phys = np.clip((log_wilt - log_psi_phys) /
                                 (log_wilt - log_crit), 0, 1)
            f_psi_phys = np.where(psi_phys > psi_critical, 1.0, f_psi_phys)

            df["f_psi_stress_phys"] = f_psi_phys

            if "ET_0" in df.columns:
                df["ET_actual_phys"] = df["ET_0"] * f_psi_phys

        # Root zone depth as feature
        df["root_zone_depth_m"] = self.config.root_depth_m

    def _add_mass_balance_diagnostics(
        self, df: pd.DataFrame, psi_col: str, weather_cols: List[str]
    ) -> None:
        """
        Add residual diagnostic features for mass balance errors.

        These let ML detect:
        - Storage errors
        - Mass balance errors
        - Systematic biases

        Features:
        - θ_phys - θ_lag (storage change from physics)
        - ψ_phys - ψ_lag (potential change from physics)
        - cumulative rain - cumulative ET (mass balance)
        """
        # Storage change diagnostics
        if "theta_phys" in df.columns:
            for lag in [1, 6, 24]:
                df[f"theta_phys_change_{lag}h"] = df["theta_phys"] - \
                    df["theta_phys"].shift(lag)

        if "psi_phys" in df.columns:
            for lag in [1, 6, 24]:
                df[f"psi_phys_change_{lag}h"] = df["psi_phys"] - \
                    df["psi_phys"].shift(lag)

        # Mass balance: cumulative rain - cumulative ET
        if "precipitation" in df.columns:
            df["cum_rain_24h"] = df["precipitation"].rolling(24).sum()
            df["cum_rain_72h"] = df["precipitation"].rolling(72).sum()
            df["cum_rain_168h"] = df["precipitation"].rolling(168).sum()

        if "ET_actual" in df.columns:
            df["cum_et_24h"] = df["ET_actual"].rolling(24).sum()
            df["cum_et_72h"] = df["ET_actual"].rolling(72).sum()
            df["cum_et_168h"] = df["ET_actual"].rolling(168).sum()

            if "cum_rain_24h" in df.columns:
                df["water_balance_24h"] = df["cum_rain_24h"] - df["cum_et_24h"]
                df["water_balance_72h"] = df["cum_rain_72h"] - df["cum_et_72h"]
                df["water_balance_168h"] = df["cum_rain_168h"] - df["cum_et_168h"]

        elif "evapotranspiration" in df.columns:
            df["cum_et_24h"] = df["evapotranspiration"].rolling(24).sum()
            df["cum_et_72h"] = df["evapotranspiration"].rolling(72).sum()

            if "cum_rain_24h" in df.columns:
                df["water_balance_24h"] = df["cum_rain_24h"] - df["cum_et_24h"]
                df["water_balance_72h"] = df["cum_rain_72h"] - df["cum_et_72h"]

        # Observed vs physics storage change comparison
        if psi_col in df.columns and "psi_phys" in df.columns:
            psi_obs_change = df[psi_col] - df[psi_col].shift(24)
            psi_phys_change = df["psi_phys"] - df["psi_phys"].shift(24)

            df["storage_change_discrepancy"] = psi_obs_change - psi_phys_change

            # Rolling bias detection
            df["storage_bias_24h"] = df["storage_change_discrepancy"].rolling(
                24).mean()
            df["storage_bias_72h"] = df["storage_change_discrepancy"].rolling(
                72).mean()

    # =========================================================================
    # HELPER FUNCTIONS FOR PHYSICS CALCULATIONS
    # =========================================================================

    def _psi_to_theta(self, psi: pd.Series) -> pd.Series:
        """
        Convert matric potential (ψ) to volumetric water content (θ).

        Uses Van Genuchten equation:
        θ = θr + (θs - θr) / [1 + (α|ψ|)^n]^m
        where m = 1 - 1/n
        """
        params = self._get_vg_params()
        psi_values = pd.to_numeric(psi, errors="coerce").to_numpy(dtype=float)
        theta_values = np.array([
            water_content_from_potential(
                p, params) if np.isfinite(p) else np.nan
            for p in psi_values
        ], dtype=float)
        return pd.Series(theta_values, index=psi.index)

    def _theta_to_psi(self, theta: pd.Series) -> pd.Series:
        """
        Convert volumetric water content (θ) to matric potential (ψ).

        Inverse Van Genuchten equation.
        """
        params = self._get_vg_params()
        theta_values = pd.to_numeric(
            theta, errors="coerce").to_numpy(dtype=float)
        psi_values = np.array([
            potential_from_water_content(
                t, params) if np.isfinite(t) else np.nan
            for t in theta_values
        ], dtype=float)
        return pd.Series(psi_values, index=theta.index)

    def _retention_curve_slope(self, psi: pd.Series) -> pd.Series:
        """
        Compute dψ/dθ - the slope of the retention curve.

        This is the soil water capacity, indicating sensitivity:
        - Large dψ/dθ: small θ change → large ψ change (sensitive)
        - Small dψ/dθ: large θ change → small ψ change (buffered)

        dψ/dθ = dψ/dSe × dSe/dθ
        """
        params = self._get_vg_params()
        psi_values = pd.to_numeric(psi, errors="coerce").to_numpy(dtype=float)

        # specific_water_capacity returns dθ/dψ (1/kPa); we want dψ/dθ
        dtheta_dpsi = np.array([
            specific_water_capacity(p, params) if np.isfinite(p) else np.nan
            for p in psi_values
        ], dtype=float)
        dpsi_dtheta = 1.0 / (dtheta_dpsi + 1e-12)
        return pd.Series(dpsi_dtheta, index=psi.index)

    def _add_temporal_features(self, df: pd.DataFrame) -> None:
        """Add time-based features."""
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(
                "Index is not DatetimeIndex, skipping temporal features")
            return

        # Hour of day (cyclical encoding)
        hour = df.index.hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # Day of year (seasonality)
        doy = df.index.dayofyear
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

        # Growing season indicator (Northern hemisphere default)
        month = df.index.month
        df["growing_season"] = ((month >= 4) & (month <= 10)).astype(float)

        # Day of week (irrigation schedules often weekly)
        dow = df.index.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    def _add_physical_soil_features(self, df: pd.DataFrame) -> None:
        """
        Add physically meaningful soil features to prevent geographic fingerprinting.

        These encode soil behavior that ML can generalize:
        - Porosity: How much water soil can hold
        - Field capacity: Maximum retained water after drainage
        - Wilting point: Minimum plant-available water
        - Available water capacity: FC - WP (usable water storage)
        - Drainage class: How fast soil drains (from texture)
        - Texture class: Simplified soil type
        """
        result = df.copy()

        # Use tropical PTF to estimate hydraulic parameters
        if 'sand_pct' in result.columns and 'clay_pct' in result.columns:
            # Porosity (saturated water content) - Saxton & Rawls PTF
            sand = result['sand_pct'].fillna(40) / 100
            clay = result['clay_pct'].fillna(25) / 100
            silt = (100 - result['sand_pct'].fillna(40) -
                    result['clay_pct'].fillna(25)) / 100
            organic = result['organic_carbon_pct'].fillna(
                1.0) / 100 if 'organic_carbon_pct' in result.columns else 0.01

            # Porosity (θs) - Saxton & Rawls (2006)
            result['porosity'] = 0.332 - 0.0007251 * \
                (sand * 100) + 0.1276 * np.log10(clay * 100 + 1)

            # Field capacity (θ at -33 kPa)
            result['field_capacity'] = (
                0.2576 - 0.002 * (sand * 100) + 0.0036 * (clay * 100) +
                0.0299 * (organic * 100) - 0.00006 *
                (sand * 100) * (clay * 100)
            )

            # Wilting point (θ at -1500 kPa)
            result['wilting_point'] = (
                0.026 + 0.005 * (clay * 100) + 0.0158 * (organic * 100)
            )

            # Available water capacity (plant-usable storage)
            result['available_water_capacity'] = result['field_capacity'] - \
                result['wilting_point']

            # Saturated hydraulic conductivity (drainage rate) - Cosby PTF
            # log10(Ksat) = -0.6 + 0.0126*sand% - 0.0064*clay%
            result['log_ksat'] = -0.6 + 0.0126 * \
                (sand * 100) - 0.0064 * (clay * 100)

            # Drainage class (categorical to numeric)
            # High clay = slow drainage (0), High sand = fast drainage (1)
            result['drainage_index'] = (sand - clay + 1) / 2  # Normalized 0-1

            # Texture class index (simplified USDA texture triangle)
            # 0 = clay-dominated, 1 = balanced, 2 = sand-dominated
            conditions = [
                (clay >= 0.40),  # Clay
                (sand >= 0.70),  # Sand
            ]
            choices = [0, 2]
            result['texture_class'] = np.select(conditions, choices, default=1)

            # Water retention ratio (FC / porosity) - soil's ability to hold water
            result['water_retention_ratio'] = result['field_capacity'] / \
                result['porosity'].clip(lower=0.2)

            # Soil permeability index (sand/clay ratio affects permeability)
            result['permeability_index'] = (
                sand / clay.clip(lower=0.05)).clip(upper=10)

            logger.info(
                "  Added physical soil features: porosity, field_capacity, AWC, drainage_index")

    def _add_lai_from_ndvi(self, df: pd.DataFrame) -> None:
        """
        Estimate Leaf Area Index (LAI) from NDVI for irrigation applications.

        LAI is crucial for irrigation:
        - Higher LAI = more transpiration = more crop water demand
        - LAI is used in FAO-56 dual crop coefficient method
        """
        result = df.copy()

        if 'ndvi_mean' in result.columns:
            ndvi = result['ndvi_mean'].fillna(0.3).clip(0.05, 0.95)

            # Beer-Lambert based LAI estimation (Baret & Guyot, 1991)
            # LAI = -ln(1 - fAPAR) / k, where fAPAR ≈ 1.25 * NDVI - 0.1
            fapar = (1.25 * ndvi - 0.1).clip(0.01, 0.95)
            k_extinction = 0.5  # Typical extinction coefficient for crops
            result['lai_estimated'] = -np.log(1 - fapar) / k_extinction

            # Clipped LAI (realistic range 0-8)
            result['lai_estimated'] = result['lai_estimated'].clip(0, 8)

            # Vegetation fraction (ground cover)
            result['vegetation_fraction'] = ((ndvi - 0.1) / 0.6).clip(0, 1)

            # Crop coefficient (Kc) estimation from NDVI for irrigation
            # Kc = 1.2 * NDVI + 0.1 (simplified FAO-56 relationship)
            result['kc_estimated'] = (1.2 * ndvi + 0.1).clip(0.15, 1.2)

            # Potential crop ET (ETc = Kc * ET0)
            if 'et0_mm' in result.columns:
                result['etc_mm'] = result['kc_estimated'] * result['et0_mm']

            logger.info("  Added LAI and irrigation features from NDVI")


def create_training_dataset(
    observations_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    physics_df: pd.DataFrame,
    psi_obs_col: str = "psi_observed_kpa",
    config: Optional[FeatureConfig] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create complete training dataset by merging observations, weather, and physics.

    Args:
        observations_df: Sensor observations with datetime index
        weather_df: Weather data with datetime index
        physics_df: Physics model outputs with datetime index
        psi_obs_col: Column name for observed matric potential
        config: Feature engineering configuration

    Returns:
        Tuple of (training dataframe, list of feature names)
    """
    # Merge all data sources on datetime index
    df = observations_df.copy()

    # Merge weather
    weather_cols = weather_df.columns.difference(df.columns)
    df = df.join(weather_df[weather_cols], how="left")

    # Merge physics
    physics_cols = physics_df.columns.difference(df.columns)
    df = df.join(physics_df[physics_cols], how="left")

    # Engineer features
    engineer = FeatureEngineer(config)
    df = engineer.create_features(df, psi_col=psi_obs_col)

    # Remove rows with NaN features (from lags, rolling windows)
    max_lag = max(config.lag_hours) if config else 168
    df = df.iloc[max_lag:]

    return df, engineer.feature_names


def create_forecast_features(
    current_state: pd.DataFrame,
    weather_forecast: pd.DataFrame,
    physics_forecast: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """
    Create features for forecasting from current state and forecasts.

    This function handles the case where we're making predictions
    into the future using weather and physics forecasts.

    Args:
        current_state: Recent observations up to now
        weather_forecast: Forecast weather data
        physics_forecast: Forecast physics model outputs
        config: Feature configuration

    Returns:
        Feature matrix for forecast horizons
    """
    # Combine historical and forecast
    df = pd.concat([current_state, weather_forecast], axis=0)
    df = df.sort_index()

    # Add physics forecast if available
    if physics_forecast is not None:
        for col in physics_forecast.columns:
            if col not in df.columns:
                df[col] = physics_forecast[col]
            else:
                # Fill future values from forecast
                df.loc[physics_forecast.index, col] = physics_forecast[col]

    # Engineer features
    engineer = FeatureEngineer(config)

    # Use last observed psi value
    psi_col = "psi_observed_kpa" if "psi_observed_kpa" in df.columns else "psi_kpa"
    if psi_col in df.columns:
        # Forward fill for forecast period
        df[psi_col] = df[psi_col].ffill()

    df = engineer.create_features(df, psi_col=psi_col)

    # Return only forecast period
    return df.loc[weather_forecast.index]
