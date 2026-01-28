"""
Irrigation Data Source

Collects irrigation data from multiple sources:
1. Farmer logs and manual records
2. Smart irrigation sensors
3. Inferred from soil moisture patterns
4. Satellite-based irrigation detection
"""

import logging
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
import json

from smps.core.types import SiteID
from smps.data.sources.base import DataSource, DataFetchRequest, DataFetchResult
from smps.data.contracts import IrrigationRecord, IrrigationMethod

logger = logging.getLogger(__name__)


class IrrigationDataSource(DataSource):
    """
    Data source for irrigation events.

    Collects irrigation data from:
    - Farmer logs (manual records)
    - Smart irrigation systems
    - Soil moisture pattern analysis (inferred)
    - Satellite irrigation detection
    """

    def __init__(self, name: str = "irrigation", cache_dir: Optional[Path] = None):
        super().__init__(name, cache_dir)
        # In production, this would connect to irrigation databases/APIs
        self._mock_irrigation_data = self._load_mock_data()

    def fetch(self, request: DataFetchRequest) -> DataFetchResult:
        """
        Fetch irrigation records for the given request.
        """
        try:
            # Get irrigation records for this site and date range
            irrigation_records = self._get_irrigation_records(
                request.site_id,
                request.start_date,
                request.end_date
            )

            return DataFetchResult(
                data=irrigation_records,
                metadata={
                    'source': self.name,
                    'site_id': request.site_id,
                    'date_range': f"{request.start_date} to {request.end_date}",
                    'count': len(irrigation_records),
                    'total_volume_mm': sum(r.volume_mm for r in irrigation_records)
                },
                quality_score=self._calculate_quality_score(
                    irrigation_records),
                cache_hit=False
            )

        except Exception as e:
            self.logger.error(f"Failed to fetch irrigation data: {e}")
            return DataFetchResult(
                data=None,
                metadata={'error': str(e)},
                quality_score=0.0,
                errors=[str(e)]
            )

    def _get_irrigation_records(self, site_id: SiteID,
                                start_date: date, end_date: date) -> List[IrrigationRecord]:
        """
        Get irrigation records for a site and date range.

        In production, this would query:
        1. Irrigation management systems
        2. Farmer databases
        3. Sensor networks
        4. Satellite irrigation detection algorithms
        """
        # For now, return mock data based on site patterns
        records = []

        # Mock irrigation patterns based on site ID
        site_hash = hash(str(site_id)) % 1000

        # Different irrigation frequencies based on site
        if site_hash < 300:  # Drip irrigation sites
            irrigation_days = self._generate_drip_schedule(
                start_date, end_date, site_hash)
        elif site_hash < 600:  # Sprinkler sites
            irrigation_days = self._generate_sprinkler_schedule(
                start_date, end_date, site_hash)
        else:  # Rain-fed or minimal irrigation
            irrigation_days = self._generate_minimal_schedule(
                start_date, end_date, site_hash)

        for irrig_date in irrigation_days:
            # Volume based on crop type and site conditions
            volume_mm = self._estimate_irrigation_volume(
                site_id, irrig_date, site_hash)

            records.append(IrrigationRecord(
                timestamp=datetime.combine(irrig_date, datetime.min.time()),
                site_id=site_id,
                volume_mm=volume_mm,
                duration_minutes=self._estimate_duration(volume_mm),
                method=self._infer_irrigation_method(site_hash),
                efficiency_factor=0.85,  # Typical irrigation efficiency
                source="inferred",  # Could be 'sensor', 'farmer_log', 'satellite'
                confidence=0.7  # Confidence in inferred data
            ))

        return records

    def _generate_drip_schedule(self, start_date: date, end_date: date, site_hash: int) -> List[date]:
        """Generate drip irrigation schedule (frequent, small amounts)."""
        schedule = []
        current = start_date

        # Drip irrigation every 2-3 days
        interval = 2 + (site_hash % 2)  # 2 or 3 days

        while current <= end_date:
            # Skip winter months (Dec-Feb) for some crops
            if not (current.month in [12, 1, 2] and site_hash % 3 == 0):
                schedule.append(current)
            current += timedelta(days=interval)

        return schedule

    def _generate_sprinkler_schedule(self, start_date: date, end_date: date, site_hash: int) -> List[date]:
        """Generate sprinkler irrigation schedule (weekly)."""
        schedule = []
        current = start_date

        # Sprinkler irrigation weekly
        while current <= end_date:
            # Every 7 days, but vary by ±1 day
            day_offset = (site_hash % 3) - 1  # -1, 0, or 1
            irrig_date = current + timedelta(days=day_offset)

            if start_date <= irrig_date <= end_date:
                schedule.append(irrig_date)

            current += timedelta(days=7)

        return schedule

    def _generate_minimal_schedule(self, start_date: date, end_date: date, site_hash: int) -> List[date]:
        """Generate minimal irrigation schedule (supplemental only)."""
        schedule = []

        # Only irrigate during dry spells
        current = start_date
        dry_spell_count = 0

        while current <= end_date:
            # Simulate dry spells (random but based on site)
            is_dry_day = (hash(f"{site_hash}_{current}") %
                          10) < 2  # 20% chance

            if is_dry_day:
                dry_spell_count += 1
                # Irrigate after 5+ dry days
                # Only some sites
                if dry_spell_count >= 5 and (site_hash % 5) == 0:
                    schedule.append(current)
                    dry_spell_count = 0
            else:
                dry_spell_count = 0

            current += timedelta(days=1)

        return schedule

    def _estimate_irrigation_volume(self, site_id: SiteID, irrig_date: date, site_hash: int) -> float:
        """Estimate irrigation volume based on crop, weather, and site conditions."""
        # Base volume depends on irrigation method
        if site_hash < 300:  # Drip
            base_volume = 10.0 + (site_hash % 10)  # 10-20mm
        elif site_hash < 600:  # Sprinkler
            base_volume = 20.0 + (site_hash % 15)  # 20-35mm
        else:  # Minimal
            base_volume = 5.0 + (site_hash % 10)  # 5-15mm

        # Seasonal adjustment (more in summer)
        month = irrig_date.month
        if month in [6, 7, 8]:  # Summer
            seasonal_factor = 1.2
        elif month in [5, 9]:  # Spring/Fall
            seasonal_factor = 1.0
        else:  # Winter
            seasonal_factor = 0.8

        return base_volume * seasonal_factor

    def _estimate_duration(self, volume_mm: float) -> float:
        """Estimate irrigation duration in minutes."""
        # Typical application rate: 5-10mm per hour
        rate_mm_per_hour = 7.5  # 7.5mm/hour = 2.08mm/minute
        return volume_mm / (rate_mm_per_hour / 60)  # Convert to minutes

    def _infer_irrigation_method(self, site_hash: int) -> IrrigationMethod:
        """Infer irrigation method from site characteristics."""
        if site_hash < 300:
            return IrrigationMethod.DRIP
        elif site_hash < 600:
            return IrrigationMethod.SPRINKLER
        else:
            return IrrigationMethod.MANUAL

    def _calculate_quality_score(self, records: List[IrrigationRecord]) -> float:
        """Calculate quality score for irrigation data."""
        if not records:
            return 0.0

        # Base score on data completeness and source reliability
        total_confidence = sum(r.confidence for r in records)
        avg_confidence = total_confidence / len(records)

        # Bonus for sensor data vs inferred
        sensor_records = sum(1 for r in records if r.source == 'sensor')
        sensor_ratio = sensor_records / len(records)

        return min(1.0, avg_confidence * (0.7 + 0.3 * sensor_ratio))

    def _load_mock_data(self) -> Dict[str, Any]:
        """Load mock irrigation data for development."""
        # In production, this would load from actual data sources
        return {
            'patterns': {
                'drip_sites': ['site_001', 'site_002', 'site_005'],
                'sprinkler_sites': ['site_003', 'site_004'],
                'rainfed_sites': ['site_006', 'site_007']
            }
        }
