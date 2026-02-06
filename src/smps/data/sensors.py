"""
WaziGate sensor integration for SWPPS.

Handles reading soil tension and temperature data from WaziGate IoT sensors.
"""

import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pytz

from smps.core.types import SensorReading, MatricPotential
from smps.core.config import PlotConfig, SensorConfig
from smps.core.exceptions import SensorError

logger = logging.getLogger("swpps.data.sensors")


class WaziGateClient:
    """
    Client for WaziGate API.

    Reads sensor data from local or remote WaziGate instances.
    """

    def __init__(self, api_url: str = "http://wazigate/", token: Optional[str] = None):
        self.api_url = api_url.rstrip('/')
        self.token = token
        self.session = requests.Session()

        if token:
            self.session.headers.update({
                'Authorization': f'Bearer {token}'
            })

    def get_latest_value(
        self,
        device_id: str,
        sensor_id: str,
        sensor_type: str = "sensors",
    ) -> Optional[SensorReading]:
        """
        Get latest value from a sensor.

        Args:
            device_id: Device identifier
            sensor_id: Sensor identifier
            sensor_type: Type ("sensors" or "actuators")

        Returns:
            SensorReading or None if unavailable
        """
        url = f"{self.api_url}/devices/{device_id}/{sensor_type}/{sensor_id}/value"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            return SensorReading(
                timestamp=datetime.now(pytz.UTC),
                sensor_id=sensor_id,
                device_id=device_id,
                value=float(data.get("value", 0)),
                unit=data.get("unit", "cbar"),
            )

        except requests.exceptions.RequestException as e:
            logger.error("Failed to read sensor %s/%s: %s",
                         device_id, sensor_id, e)
            return None

    def get_historical_values(
        self,
        device_id: str,
        sensor_id: str,
        start_date: datetime,
        sensor_type: str = "sensors",
    ) -> List[SensorReading]:
        """
        Get historical values from a sensor.

        Args:
            device_id: Device identifier
            sensor_id: Sensor identifier
            start_date: Start datetime for data
            sensor_type: Type ("sensors" or "actuators")

        Returns:
            List of SensorReading objects
        """
        url = f"{self.api_url}/devices/{device_id}/{sensor_type}/{sensor_id}/values"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            readings = []
            for item in data:
                try:
                    timestamp = datetime.fromisoformat(
                        item["time"].replace("Z", "+00:00")
                    )

                    if timestamp >= start_date:
                        readings.append(SensorReading(
                            timestamp=timestamp,
                            sensor_id=sensor_id,
                            device_id=device_id,
                            value=float(item.get("value", 0)),
                            unit=item.get("unit", "cbar"),
                        ))
                except (KeyError, ValueError) as e:
                    logger.warning("Failed to parse sensor value: %s", e)

            return sorted(readings, key=lambda r: r.timestamp)

        except requests.exceptions.RequestException as e:
            logger.error("Failed to fetch history for %s/%s: %s",
                         device_id, sensor_id, e)
            return []

    def get_token(self, username: str = "admin", password: str = "loragateway") -> str:
        """
        Get authentication token.

        Args:
            username: WaziGate username
            password: WaziGate password

        Returns:
            Authentication token
        """
        if self.api_url.startswith("http://wazigate/"):
            # Local gateway doesn't need auth
            return ""

        url = f"{self.api_url}/auth/token"

        try:
            response = self.session.post(
                url,
                json={"username": username, "password": password},
                timeout=10,
            )
            response.raise_for_status()
            token = response.json()

            self.token = token
            self.session.headers.update({
                'Authorization': f'Bearer {token}'
            })

            return token

        except requests.exceptions.RequestException as e:
            logger.error("Failed to get token: %s", e)
            raise SensorError(f"Authentication failed: {e}")


class SensorDataManager:
    """
    Manages sensor data collection for a plot.

    Aggregates readings from multiple sensors into matric potential values.
    """

    def __init__(self, plot_config: PlotConfig, api_url: str, token: Optional[str] = None):
        self.plot_config = plot_config
        self.client = WaziGateClient(api_url, token)

    def get_current_potential(self) -> Optional[MatricPotential]:
        """
        Get current matric potential from soil moisture sensors.

        Averages readings from multiple sensors if configured.

        Returns:
            Average matric potential in kPa (negative) or None
        """
        potentials = []

        for sensor in self.plot_config.moisture_sensors:
            reading = self.client.get_latest_value(
                sensor.device_id,
                sensor.sensor_id,
            )

            if reading and reading.matric_potential_kpa is not None:
                potentials.append(reading.matric_potential_kpa)

        if not potentials:
            return None

        return sum(potentials) / len(potentials)

    def get_current_temperature(self) -> Optional[float]:
        """
        Get current soil temperature.

        Returns:
            Average temperature in Celsius or None
        """
        temperatures = []

        for sensor in self.plot_config.temperature_sensors:
            reading = self.client.get_latest_value(
                sensor.device_id,
                sensor.sensor_id,
            )

            if reading:
                temperatures.append(reading.value)

        if not temperatures:
            return None

        return sum(temperatures) / len(temperatures)

    def get_historical_potential(
        self,
        start_date: datetime,
    ) -> List[tuple[datetime, MatricPotential]]:
        """
        Get historical matric potential data.

        Args:
            start_date: Start datetime

        Returns:
            List of (timestamp, potential) tuples
        """
        # Collect readings from all moisture sensors
        all_readings: Dict[datetime, List[float]] = {}

        for sensor in self.plot_config.moisture_sensors:
            readings = self.client.get_historical_values(
                sensor.device_id,
                sensor.sensor_id,
                start_date,
            )

            for reading in readings:
                psi = reading.matric_potential_kpa
                if psi is not None:
                    # Round timestamp to nearest hour for aggregation
                    ts = reading.timestamp.replace(
                        minute=0, second=0, microsecond=0)

                    if ts not in all_readings:
                        all_readings[ts] = []
                    all_readings[ts].append(psi)

        # Average readings at each timestamp
        result = []
        for ts in sorted(all_readings.keys()):
            values = all_readings[ts]
            avg_psi = sum(values) / len(values)
            result.append((ts, avg_psi))

        return result

    def get_flow_meter_reading(self) -> Optional[float]:
        """
        Get latest flow meter reading (for irrigation verification).

        Returns:
            Flow reading or None
        """
        if not self.plot_config.flow_sensors:
            return None

        sensor = self.plot_config.flow_sensors[0]
        reading = self.client.get_latest_value(
            sensor.device_id,
            sensor.sensor_id,
            sensor_type="actuators",
        )

        return reading.value if reading else None
