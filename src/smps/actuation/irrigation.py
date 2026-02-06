"""
Irrigation Actuation Controller.

This module handles the physical triggering of irrigation through
the WaziGate IoT platform. It converts irrigation decisions into
actuator commands and monitors irrigation events.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

from smps.core.types import IrrigationDecision
from smps.core.exceptions import ActuatorError, ConfigurationError

logger = logging.getLogger("swpps.actuation.irrigation")


@dataclass
class ActuatorConfig:
    """Configuration for irrigation actuator."""

    # WaziGate connection
    gateway_url: str = "http://localhost"
    gateway_port: int = 880

    # Device identification
    device_id: str = ""
    actuator_id: str = "irrigation_valve"

    # Flow sensor for monitoring
    flow_sensor_id: str = "waterflow"

    # Irrigation parameters
    flow_rate_mm_per_min: float = 0.5  # Typical drip system
    max_duration_minutes: float = 60.0
    min_duration_minutes: float = 5.0

    # Safety limits
    max_daily_irrigation_mm: float = 50.0
    max_irrigations_per_day: int = 3

    # Timing
    blackout_start_hour: int = 22  # No irrigation after 10 PM
    blackout_end_hour: int = 6    # Or before 6 AM

    # Dry run mode (for testing)
    dry_run: bool = False


class IrrigationActuator:
    """
    Controls irrigation hardware through WaziGate.

    Features:
    - Valve control via WaziGate actuators
    - Flow monitoring for actual water delivery
    - Safety limits (max duration, blackout periods)
    - Event logging for water accounting
    """

    def __init__(self, config: Optional[ActuatorConfig] = None):
        self.config = config or ActuatorConfig()

        if not self.config.device_id:
            raise ConfigurationError("Device ID is required for actuation")

        self.base_url = f"{self.config.gateway_url}:{self.config.gateway_port}"

        # Track daily irrigation
        self.daily_irrigation_mm: float = 0.0
        self.daily_irrigation_count: int = 0
        self.last_reset_date: datetime = datetime.now().date()

        # Event log
        self.irrigation_events: List[Dict] = []

    def trigger_irrigation(
        self,
        decision: IrrigationDecision,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Trigger irrigation based on decision.

        Args:
            decision: Irrigation decision with amount
            force: Override safety checks

        Returns:
            Result dictionary with status and details
        """
        if not decision.should_irrigate:
            return {"status": "skipped", "reason": "Decision indicates no irrigation"}

        # Reset daily counter if needed
        self._check_daily_reset()

        # Safety checks
        if not force:
            safety_check = self._safety_check(decision.amount_mm)
            if not safety_check["passed"]:
                return {
                    "status": "blocked",
                    "reason": safety_check["reason"],
                }

        # Calculate duration
        duration_min = self._calculate_duration(decision.amount_mm)

        logger.info(
            "Triggering irrigation: %.1f mm over %.1f minutes",
            decision.amount_mm, duration_min
        )

        # Execute irrigation
        result = self._execute_irrigation(duration_min, decision.amount_mm)

        # Update tracking
        if result["status"] == "success":
            self.daily_irrigation_mm += decision.amount_mm
            self.daily_irrigation_count += 1
            self._log_event(decision, result)

        return result

    def _safety_check(self, amount_mm: float) -> Dict[str, Any]:
        """Perform safety checks before irrigation."""
        now = datetime.now()

        # Blackout period check
        if self._in_blackout_period(now):
            return {
                "passed": False,
                "reason": f"Blackout period ({self.config.blackout_start_hour}:00 - {self.config.blackout_end_hour}:00)",
            }

        # Daily limit check
        if self.daily_irrigation_mm + amount_mm > self.config.max_daily_irrigation_mm:
            return {
                "passed": False,
                "reason": f"Daily limit exceeded ({self.daily_irrigation_mm:.1f}/{self.config.max_daily_irrigation_mm:.1f} mm)",
            }

        # Event count check
        if self.daily_irrigation_count >= self.config.max_irrigations_per_day:
            return {
                "passed": False,
                "reason": f"Max daily irrigations reached ({self.config.max_irrigations_per_day})",
            }

        return {"passed": True, "reason": ""}

    def _in_blackout_period(self, dt: datetime) -> bool:
        """Check if current time is in blackout period."""
        hour = dt.hour

        if self.config.blackout_start_hour > self.config.blackout_end_hour:
            # Blackout spans midnight
            return hour >= self.config.blackout_start_hour or hour < self.config.blackout_end_hour
        else:
            return self.config.blackout_start_hour <= hour < self.config.blackout_end_hour

    def _calculate_duration(self, amount_mm: float) -> float:
        """Calculate irrigation duration from amount."""
        duration = amount_mm / self.config.flow_rate_mm_per_min

        # Apply limits
        duration = max(duration, self.config.min_duration_minutes)
        duration = min(duration, self.config.max_duration_minutes)

        return duration

    def _execute_irrigation(
        self,
        duration_min: float,
        target_mm: float,
    ) -> Dict[str, Any]:
        """Execute the irrigation event."""
        start_time = datetime.now()

        if self.config.dry_run:
            logger.info(
                "DRY RUN: Would irrigate for %.1f minutes", duration_min)
            return {
                "status": "success",
                "dry_run": True,
                "duration_minutes": duration_min,
                "target_mm": target_mm,
                "actual_mm": target_mm,
                "start_time": start_time.isoformat(),
            }

        try:
            # Turn on valve
            self._set_actuator_value(True)
            logger.info("Irrigation valve OPENED")

            # Monitor flow during irrigation
            initial_flow = self._read_flow_sensor()

            # Wait for duration (with monitoring)
            actual_duration = self._wait_with_monitoring(
                duration_min, target_mm)

            # Turn off valve
            self._set_actuator_value(False)
            logger.info("Irrigation valve CLOSED")

            # Calculate actual amount delivered
            final_flow = self._read_flow_sensor()
            actual_mm = final_flow - initial_flow if final_flow and initial_flow else target_mm

            end_time = datetime.now()

            return {
                "status": "success",
                "duration_minutes": actual_duration,
                "target_mm": target_mm,
                "actual_mm": actual_mm,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            }

        except Exception as e:
            # Ensure valve is closed on error
            try:
                self._set_actuator_value(False)
            except Exception:
                pass

            logger.error("Irrigation failed: %s", str(e))
            raise ActuatorError(f"Irrigation execution failed: {str(e)}")

    def _set_actuator_value(self, on: bool) -> None:
        """Set actuator state via WaziGate API."""
        url = f"{self.base_url}/devices/{self.config.device_id}/actuators/{self.config.actuator_id}/value"

        try:
            response = requests.post(
                url,
                json=on,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ActuatorError(f"Failed to set actuator: {str(e)}")

    def _read_flow_sensor(self) -> Optional[float]:
        """Read cumulative flow from sensor."""
        url = f"{self.base_url}/devices/{self.config.device_id}/sensors/{self.config.flow_sensor_id}/value"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return float(response.json())
        except Exception as e:
            logger.warning("Could not read flow sensor: %s", str(e))
            return None

    def _wait_with_monitoring(
        self,
        duration_min: float,
        target_mm: float,
    ) -> float:
        """Wait for irrigation duration while monitoring flow."""
        start = time.time()
        target_seconds = duration_min * 60
        check_interval = 30  # Check every 30 seconds

        elapsed = 0
        while elapsed < target_seconds:
            time.sleep(min(check_interval, target_seconds - elapsed))
            elapsed = time.time() - start

            # Could add flow-based cutoff here
            # If actual_mm >= target_mm, stop early

        return elapsed / 60  # Return actual duration in minutes

    def _check_daily_reset(self) -> None:
        """Reset daily counters if new day."""
        today = datetime.now().date()
        if today > self.last_reset_date:
            logger.info("Resetting daily irrigation counters")
            self.daily_irrigation_mm = 0.0
            self.daily_irrigation_count = 0
            self.last_reset_date = today

    def _log_event(
        self,
        decision: IrrigationDecision,
        result: Dict[str, Any],
    ) -> None:
        """Log irrigation event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "target_mm": decision.amount_mm,
            "actual_mm": result.get("actual_mm", 0),
            "duration_min": result.get("duration_minutes", 0),
            "trigger_reason": decision.reason,
            "current_psi_kpa": decision.current_psi_kpa,
        }
        self.irrigation_events.append(event)

        # Keep only last 100 events
        if len(self.irrigation_events) > 100:
            self.irrigation_events = self.irrigation_events[-100:]

    def get_status(self) -> Dict[str, Any]:
        """Get current actuator status."""
        return {
            "daily_irrigation_mm": self.daily_irrigation_mm,
            "daily_irrigation_count": self.daily_irrigation_count,
            "max_daily_mm": self.config.max_daily_irrigation_mm,
            "max_daily_count": self.config.max_irrigations_per_day,
            "in_blackout": self._in_blackout_period(datetime.now()),
            "dry_run_mode": self.config.dry_run,
            "recent_events": self.irrigation_events[-5:],
        }

    def emergency_stop(self) -> bool:
        """Emergency stop - close valve immediately."""
        logger.warning("EMERGENCY STOP triggered")
        try:
            self._set_actuator_value(False)
            return True
        except Exception as e:
            logger.error("Emergency stop failed: %s", str(e))
            return False


class IrrigationScheduler:
    """
    Schedules irrigation based on predictions and decisions.

    Works with the decision engine to plan irrigation events
    and manages the queue of scheduled irrigations.
    """

    def __init__(self, actuator: IrrigationActuator):
        self.actuator = actuator
        self.scheduled_events: List[Dict] = []

    def schedule(
        self,
        decision: IrrigationDecision,
        execute_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Schedule an irrigation event.

        Args:
            decision: Irrigation decision
            execute_at: When to execute (None = immediately)

        Returns:
            Scheduling result
        """
        if execute_at is None or execute_at <= datetime.now():
            # Execute immediately
            return self.actuator.trigger_irrigation(decision)

        # Add to schedule
        event = {
            "scheduled_time": execute_at,
            "decision": decision,
            "status": "pending",
        }
        self.scheduled_events.append(event)

        logger.info("Irrigation scheduled for %s", execute_at.isoformat())

        return {
            "status": "scheduled",
            "execute_at": execute_at.isoformat(),
            "amount_mm": decision.amount_mm,
        }

    def check_schedule(self) -> Optional[Dict[str, Any]]:
        """
        Check and execute any due scheduled irrigations.

        Should be called periodically (e.g., every minute).
        """
        now = datetime.now()

        for event in self.scheduled_events:
            if event["status"] == "pending" and event["scheduled_time"] <= now:
                # Execute scheduled event
                result = self.actuator.trigger_irrigation(event["decision"])
                event["status"] = "executed"
                event["result"] = result
                return result

        # Clean up old events
        self.scheduled_events = [
            e for e in self.scheduled_events
            if e["status"] == "pending" or
            (datetime.fromisoformat(e.get("result", {}).get("end_time", "2000-01-01"))
             > now - timedelta(hours=24))
        ]

        return None


def create_actuator(
    device_id: str,
    gateway_url: str = "http://localhost",
    gateway_port: int = 880,
    flow_rate_mm_per_min: float = 0.5,
    dry_run: bool = False,
) -> IrrigationActuator:
    """
    Factory function to create an irrigation actuator.

    Args:
        device_id: WaziGate device ID
        gateway_url: WaziGate URL
        gateway_port: WaziGate port
        flow_rate_mm_per_min: System flow rate
        dry_run: If True, don't actually trigger irrigation

    Returns:
        Configured IrrigationActuator instance
    """
    config = ActuatorConfig(
        device_id=device_id,
        gateway_url=gateway_url,
        gateway_port=gateway_port,
        flow_rate_mm_per_min=flow_rate_mm_per_min,
        dry_run=dry_run,
    )
    return IrrigationActuator(config)
