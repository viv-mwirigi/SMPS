"""
Irrigation Decision Engine.

This module converts matric potential forecasts into actionable
irrigation decisions. It uses universal tension thresholds that
work across all soil types.

The key advantage of using matric potential for decisions:
- No need for soil-specific calibration
- Universal crop stress thresholds
- Direct translation to plant water status
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from swpps.core.types import (
    IrrigationDecision,
    MatricPotential,
    PredictionResult,
    SoilMoistureStatus,
    VanGenuchtenParams,
)
from swpps.core.constants import (
    CROP_THRESHOLDS,
    IRRIGATION_THRESHOLDS,
    IrrigationAction,
)

logger = logging.getLogger("swpps.prediction.decision")


@dataclass
class DecisionConfig:
    """Configuration for irrigation decision making."""

    # Crop type for threshold selection
    crop_type: str = "generic"

    # Decision lookahead horizon (hours)
    forecast_horizon: int = 72

    # Minimum time between irrigations (hours)
    min_irrigation_interval: float = 24.0

    # Risk tolerance (0-1, higher = more conservative/earlier irrigation)
    risk_tolerance: float = 0.3

    # Whether to use uncertainty in decisions
    use_uncertainty: bool = True

    # Irrigation system efficiency (0-1)
    irrigation_efficiency: float = 0.85

    # Maximum irrigation amount per event (mm)
    max_irrigation_mm: float = 30.0

    # Optional physics parameters for physically-based refill amount
    vg_params: Optional[VanGenuchtenParams] = None
    root_depth_m: float = 0.30

    # Whether irrigation is allowed (can be disabled for manual control)
    irrigation_enabled: bool = True


class IrrigationDecisionEngine:
    """
    Makes irrigation decisions based on matric potential forecasts.

    Decision logic:
    1. If current ψ < stress_threshold: Irrigate immediately
    2. If forecast ψ will cross threshold: Schedule irrigation
    3. If ψ > field_capacity: No irrigation needed

    The amount of irrigation is calculated to bring the soil
    back to field capacity, accounting for efficiency losses.
    """

    def __init__(self, config: Optional[DecisionConfig] = None):
        self.config = config or DecisionConfig()
        self.last_irrigation_time: Optional[datetime] = None

        # Get thresholds for crop type
        self._load_crop_thresholds()

    def _load_crop_thresholds(self) -> None:
        """Load thresholds for configured crop type."""
        crop = self.config.crop_type.lower()

        if crop in CROP_THRESHOLDS:
            thresholds = CROP_THRESHOLDS[crop]
            self.trigger_threshold = thresholds["irrigate_below_kpa"]
            self.refill_target = thresholds["refill_to_kpa"]
            self.stress_threshold = thresholds.get("stress_threshold_kpa",
                                                   self.trigger_threshold * 1.5)
        else:
            # Use generic thresholds
            self.trigger_threshold = IRRIGATION_THRESHOLDS["irrigate_below_kpa"]
            self.refill_target = IRRIGATION_THRESHOLDS["optimal_upper_kpa"]
            self.stress_threshold = IRRIGATION_THRESHOLDS["stress_threshold_kpa"]

        logger.info(
            "Crop '%s' thresholds: trigger=%.0f kPa, refill=%.0f kPa",
            crop, self.trigger_threshold, self.refill_target
        )

    def evaluate(
        self,
        current_psi: MatricPotential,
        forecasts: Dict[int, PredictionResult],
        current_time: Optional[datetime] = None,
    ) -> IrrigationDecision:
        """
        Evaluate whether irrigation is needed.

        Args:
            current_psi: Current soil matric potential
            forecasts: Forecast predictions by horizon
            current_time: Current timestamp

        Returns:
            IrrigationDecision with action and details
        """
        current_time = current_time or datetime.now()

        # Check if irrigation recently occurred
        if self._recently_irrigated(current_time):
            return self._create_decision(
                IrrigationAction.NO_ACTION,
                current_psi,
                reason="Minimum irrigation interval not elapsed",
            )

        # Check if irrigation is disabled
        if not self.config.irrigation_enabled:
            status = SoilMoistureStatus.from_potential(current_psi)
            return self._create_decision(
                IrrigationAction.NO_ACTION,
                current_psi,
                reason="Irrigation disabled",
                status=status,
            )

        # Evaluate current state
        current_status = SoilMoistureStatus.from_potential(current_psi)

        # Immediate irrigation check
        if current_psi < self.trigger_threshold:
            amount = self._calculate_irrigation_amount(current_psi)
            return self._create_decision(
                IrrigationAction.IRRIGATE_NOW,
                current_psi,
                amount_mm=amount,
                reason=f"Soil water potential ({current_psi:.0f} kPa) below trigger ({self.trigger_threshold:.0f} kPa)",
                status=current_status,
            )

        # Check if already at field capacity
        if current_psi > self.refill_target:
            return self._create_decision(
                IrrigationAction.NO_ACTION,
                current_psi,
                reason="Soil moisture adequate",
                status=current_status,
            )

        # Evaluate forecasts for preemptive irrigation
        should_irrigate, trigger_horizon, forecast_psi = self._evaluate_forecasts(
            forecasts
        )

        if should_irrigate:
            amount = self._calculate_irrigation_amount(forecast_psi)
            return self._create_decision(
                IrrigationAction.SCHEDULE,
                current_psi,
                amount_mm=amount,
                # Irrigate before stress
                scheduled_hours=max(trigger_horizon - 12, 0),
                reason=f"Forecast: soil will reach {forecast_psi:.0f} kPa in {trigger_horizon}h",
                status=current_status,
            )

        return self._create_decision(
            IrrigationAction.MONITOR,
            current_psi,
            reason="Conditions adequate, continue monitoring",
            status=current_status,
        )

    def _evaluate_forecasts(
        self,
        forecasts: Dict[int, PredictionResult],
    ) -> Tuple[bool, int, float]:
        """
        Evaluate forecasts to determine if preemptive irrigation needed.

        Returns:
            Tuple of (should_irrigate, trigger_horizon, predicted_psi)
        """
        for horizon in sorted(forecasts.keys()):
            if horizon > self.config.forecast_horizon:
                continue

            pred = forecasts[horizon]

            # Consider uncertainty if enabled
            if self.config.use_uncertainty:
                # Use lower confidence bound for conservative estimate
                risk_factor = 1 - self.config.risk_tolerance
                effective_psi = (
                    pred.prediction_kpa * risk_factor +
                    pred.confidence_lower_kpa * (1 - risk_factor)
                )
            else:
                effective_psi = pred.prediction_kpa

            # Check if threshold will be crossed
            if effective_psi < self.trigger_threshold:
                return True, horizon, pred.prediction_kpa

        return False, 0, 0.0

    def _calculate_irrigation_amount(
        self,
        current_psi: MatricPotential,
    ) -> float:
        """
        Calculate irrigation amount to reach refill target.

        This uses a simplified approach based on the water needed
        to change matric potential from current to target.
        """
        # Preferred: use Van Genuchten to compute the θ deficit to refill target,
        # then convert to an equivalent mm over root depth.
        if self.config.vg_params is not None:
            from swpps.physics.van_genuchten import water_content_from_potential

            vg = self.config.vg_params
            theta_now = water_content_from_potential(float(current_psi), vg)
            theta_target = water_content_from_potential(
                float(self.refill_target), vg)

            # If already wetter than target, no refill.
            dtheta = max(0.0, float(theta_target - theta_now))
            gross_amount = dtheta * \
                float(self.config.root_depth_m) * 1000.0  # m -> mm
        else:
            # Fallback: simple linear approximation
            psi_deficit = self.refill_target - current_psi  # Positive value
            mm_per_kpa = 0.15
            gross_amount = abs(psi_deficit) * mm_per_kpa

        net_amount = gross_amount / self.config.irrigation_efficiency
        return min(float(net_amount), float(self.config.max_irrigation_mm))

    def _recently_irrigated(self, current_time: datetime) -> bool:
        """Check if irrigation occurred recently."""
        if self.last_irrigation_time is None:
            return False

        elapsed = (current_time -
                   self.last_irrigation_time).total_seconds() / 3600
        return elapsed < self.config.min_irrigation_interval

    def _create_decision(
        self,
        action: IrrigationAction,
        current_psi: MatricPotential,
        amount_mm: float = 0.0,
        scheduled_hours: float = 0.0,
        reason: str = "",
        status: Optional[SoilMoistureStatus] = None,
    ) -> IrrigationDecision:
        """Create irrigation decision object."""
        if status is None:
            status = SoilMoistureStatus.from_potential(current_psi)

        if action == IrrigationAction.IRRIGATE_NOW:
            urgency = "immediate"
        elif action == IrrigationAction.SCHEDULE:
            urgency = "scheduled"
        elif action == IrrigationAction.MONITOR:
            urgency = "soon"
        elif action == IrrigationAction.NO_ACTION:
            urgency = "none"
        else:
            urgency = "none"

        recommended_time = None
        time_until_critical_hours = None
        if action == IrrigationAction.SCHEDULE:
            recommended_time = datetime.now() + timedelta(hours=float(scheduled_hours))
            time_until_critical_hours = float(scheduled_hours)

        return IrrigationDecision(
            should_irrigate=action in (IrrigationAction.IRRIGATE_NOW,
                                       IrrigationAction.SCHEDULE),
            urgency=urgency,
            action=action.value,
            amount_mm=float(amount_mm),
            reason=reason,
            current_psi_kpa=current_psi,
            status=status.value,
            scheduled_time_hours=scheduled_hours if action == IrrigationAction.SCHEDULE else None,
            recommended_time=recommended_time,
            time_until_critical_hours=time_until_critical_hours,
            recommended_amount_mm=float(amount_mm) if action in (
                IrrigationAction.IRRIGATE_NOW, IrrigationAction.SCHEDULE) else None,
        )

    def confirm_irrigation(self, irrigation_time: datetime, amount_mm: float) -> None:
        """
        Confirm that irrigation occurred.

        Call this after irrigation is triggered to update internal state.
        """
        self.last_irrigation_time = irrigation_time
        logger.info("Irrigation confirmed: %.1f mm at %s",
                    amount_mm, irrigation_time)

    def set_crop(self, crop_type: str) -> None:
        """Change the crop type and reload thresholds."""
        self.config.crop_type = crop_type
        self._load_crop_thresholds()


def create_decision_engine(
    crop_type: str = "generic",
    risk_tolerance: float = 0.3,
    irrigation_enabled: bool = True,
) -> IrrigationDecisionEngine:
    """
    Factory function to create a configured decision engine.

    Args:
        crop_type: Type of crop (e.g., "tomato", "maize", "lettuce")
        risk_tolerance: Risk tolerance (0-1, higher = more conservative)
        irrigation_enabled: Whether irrigation actuation is enabled

    Returns:
        Configured IrrigationDecisionEngine instance
    """
    config = DecisionConfig(
        crop_type=crop_type,
        risk_tolerance=risk_tolerance,
        irrigation_enabled=irrigation_enabled,
    )
    return IrrigationDecisionEngine(config)
