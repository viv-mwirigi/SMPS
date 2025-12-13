"""
Quality control system for soil moisture data.
Implements multi-stage QC pipeline with configurable rules.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from smps.core.types import DataQualityFlag
from smps.data.contracts import SoilMoistureObservation


class QCLevel(str, Enum):
    """QC processing levels"""
    LEVEL_0 = "raw"           # No QC
    LEVEL_1 = "basic"         # Range checks, spike detection
    LEVEL_2 = "intermediate"  # Temporal consistency, sensor comparison
    LEVEL_3 = "advanced"      # Physical plausibility, model-based
    LEVEL_4 = "expert"        # Manual review flags


@dataclass
class QCResult:
    """Result of quality control check"""
    passed: bool
    flag: DataQualityFlag
    original_value: Optional[float] = None
    corrected_value: Optional[float] = None
    confidence: float = 1.0
    rule_name: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "passed": self.passed,
            "flag": self.flag.value,
            "original_value": self.original_value,
            "corrected_value": self.corrected_value,
            "confidence": self.confidence,
            "rule_name": self.rule_name,
            "details": self.details
        }


@dataclass
class QCRule:
    """Quality control rule definition"""
    name: str
    description: str
    check_function: Callable
    correction_function: Optional[Callable] = None
    severity: str = "warning"  # "info", "warning", "error"
    enabled: bool = True

    def apply(self, value: float, context: Dict[str, Any]) -> QCResult:
        """Apply rule to value"""
        try:
            result = self.check_function(value, context)

            if isinstance(result, bool):
                # Simple pass/fail
                passed = result
                details = {}
            else:
                # Detailed result
                passed = result.get("passed", False)
                details = result.get("details", {})

            # Determine flag
            if passed:
                flag = DataQualityFlag.OK
            else:
                flag = self._get_flag_for_severity()

            # Apply correction if available and failed
            corrected_value = value
            if not passed and self.correction_function:
                try:
                    corrected_value = self.correction_function(value, context)
                except Exception as e:
                    logging.warning(f"Correction failed for rule {self.name}: {e}")

            return QCResult(
                passed=passed,
                flag=flag,
                original_value=value,
                corrected_value=corrected_value,
                confidence=1.0,
                rule_name=self.name,
                details=details
            )

        except Exception as e:
            logging.error(f"Rule {self.name} failed: {e}")
            return QCResult(
                passed=False,
                flag=DataQualityFlag.FAILED_QC,
                original_value=value,
                confidence=0.0,
                rule_name=self.name,
                details={"error": str(e)}
            )

    def _get_flag_for_severity(self) -> DataQualityFlag:
        """Map severity to quality flag"""
        severity_map = {
            "info": DataQualityFlag.OK,
            "warning": DataQualityFlag.UNCERTAIN,
            "error": DataQualityFlag.FAILED_QC
        }
        return severity_map.get(self.severity, DataQualityFlag.UNCERTAIN)


class QualityControlPipeline:
    """
    Multi-stage quality control pipeline for soil moisture data.
    Applies rules in sequence with configurable thresholds.
    """

    def __init__(self, level: QCLevel = QCLevel.LEVEL_2):
        self.level = level
        self.logger = logging.getLogger("smps.qc")
        self.rules = self._initialize_rules()

        # Thresholds (configurable)
        self.thresholds = {
            "range_min_vwc": 0.01,      # Minimum VWC (m³/m³)
            "range_max_vwc": 0.50,      # Maximum VWC
            "range_min_tension": 0.0,   # Minimum tension (kPa)
            "range_max_tension": 1500.0,# Maximum tension
            "spike_threshold_std": 3.0, # Standard deviations for spike detection
            "rate_of_change_max": 0.1,  # Max daily change (VWC/day)
            "freezing_temp": 0.0,       # Freezing temperature (°C)
        }

    def process_observation(self, observation: SoilMoistureObservation,
                           context: Dict[str, Any]) -> SoilMoistureObservation:
        """
        Process a single observation through QC pipeline.
        Returns observation with updated quality flag.
        """
        # Build context
        full_context = {
            "observation": observation,
            "thresholds": self.thresholds,
            **context
        }

        # Initial result list
        results = []

        # Apply rules based on level
        for rule in self.rules:
            if not rule.enabled:
                continue

            # Get value to check (VWC or tension)
            value = None
            if observation.vwc is not None:
                value = observation.vwc
                value_type = "vwc"
            elif observation.tension_kpa is not None:
                value = observation.tension_kpa
                value_type = "tension"
            else:
                # No measurement value
                continue

            # Apply rule
            full_context["value_type"] = value_type
            result = rule.apply(value, full_context)
            results.append(result)

            # Update observation if correction applied
            if result.corrected_value is not None and result.corrected_value != value:
                if value_type == "vwc":
                    observation.vwc = result.corrected_value
                else:
                    observation.tension_kpa = result.corrected_value

        # Determine overall quality flag
        overall_flag = self._determine_overall_flag(results)

        # Update observation
        observation.quality_flag = overall_flag
        observation.confidence = self._calculate_confidence(results)

        # Add QC metadata
        qc_metadata = {
            "qc_level": self.level.value,
            "rule_results": [r.to_dict() for r in results],
            "overall_flag": overall_flag.value,
            "confidence": observation.confidence
        }

        if hasattr(observation, 'metadata'):
            observation.metadata['qc'] = qc_metadata
        else:
            observation.metadata = {'qc': qc_metadata}

        return observation

    def process_dataframe(self, df: pd.DataFrame,
                         sensor_type: SensorType,
                         context: Dict[str, Any]) -> pd.DataFrame:
        """
        Process DataFrame of observations.
        Assumes columns: ['timestamp', 'vwc' or 'tension_kpa', 'site_id', ...]
        """
        results_df = df.copy()

        # Initialize QC columns
        results_df['quality_flag'] = DataQualityFlag.OK.value
        results_df['qc_confidence'] = 1.0
        results_df['qc_corrected'] = False

        # Get value column name
        value_col = 'vwc' if 'vwc' in df.columns else 'tension_kpa'

        # Apply rules to each row
        for idx, row in df.iterrows():
            value = row[value_col]

            if pd.isna(value):
                results_df.at[idx, 'quality_flag'] = DataQualityFlag.MISSING.value
                results_df.at[idx, 'qc_confidence'] = 0.0
                continue

            # Create observation for processing
            observation = SoilMoistureObservation(
                timestamp=row['timestamp'],
                site_id=row['site_id'],
                depth_cm=row.get('depth_cm', 10),
                vwc=row.get('vwc'),
                tension_kpa=row.get('tension_kpa'),
                sensor_type=sensor_type
            )

            # Process through pipeline
            processed = self.process_observation(observation, context)

            # Update results
            results_df.at[idx, 'quality_flag'] = processed.quality_flag.value
            results_df.at[idx, 'qc_confidence'] = processed.confidence

            # Update corrected value if changed
            if processed.vwc is not None and processed.vwc != row.get('vwc'):
                results_df.at[idx, 'vwc'] = processed.vwc
                results_df.at[idx, 'qc_corrected'] = True
            elif (processed.tension_kpa is not None and
                  processed.tension_kpa != row.get('tension_kpa')):
                results_df.at[idx, 'tension_kpa'] = processed.tension_kpa
                results_df.at[idx, 'qc_corrected'] = True

        return results_df

    def _determine_overall_flag(self, results: List[QCResult]) -> DataQualityFlag:
        """Determine overall quality flag from rule results"""
        if not results:
            return DataQualityFlag.OK

        # Count flags
        flag_counts = {}
        for result in results:
            flag = result.flag
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

        # Hierarchy: FAILED_QC > UNCERTAIN > INTERPOLATED > OK
        if DataQualityFlag.FAILED_QC in flag_counts:
            return DataQualityFlag.FAILED_QC
        elif DataQualityFlag.UNCERTAIN in flag_counts:
            return DataQualityFlag.UNCERTAIN
        elif DataQualityFlag.INTERPOLATED in flag_counts:
            return DataQualityFlag.INTERPOLATED
        else:
            return DataQualityFlag.OK

    def _calculate_confidence(self, results: List[QCResult]) -> float:
        """Calculate overall confidence from rule results"""
        if not results:
            return 1.0

        # Weight by rule severity
        weights = {
            DataQualityFlag.OK: 1.0,
            DataQualityFlag.INTERPOLATED: 0.8,
            DataQualityFlag.UNCERTAIN: 0.5,
            DataQualityFlag.FAILED_QC: 0.0
        }

        total_weight = 0
        weighted_sum = 0

        for result in results:
            weight = weights.get(result.flag, 0.5)
            total_weight += weight
            weighted_sum += result.confidence * weight

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.5  # Default medium confidence

    def _initialize_rules(self) -> List[QCRule]:
        """Initialize QC rules based on level"""
        rules = []

        # Level 1: Basic range checks
        rules.extend([
            QCRule(
                name="range_vwc",
                description="Check VWC within physical limits",
                check_function=self._check_range_vwc,
                correction_function=self._correct_range_vwc,
                severity="error"
            ),
            QCRule(
                name="range_tension",
                description="Check tension within physical limits",
                check_function=self._check_range_tension,
                correction_function=self._correct_range_tension,
                severity="error"
            ),
            QCRule(
                name="spike_detection",
                description="Detect spikes in time series",
                check_function=self._check_spike,
                correction_function=self._correct_spike,
                severity="warning"
            ),
        ])

        if self.level in [QCLevel.LEVEL_2, QCLevel.LEVEL_3, QCLevel.LEVEL_4]:
            rules.extend([
                QCRule(
                    name="rate_of_change",
                    description="Check rate of change",
                    check_function=self._check_rate_of_change,
                    correction_function=self._correct_rate_of_change,
                    severity="warning"
                ),
                QCRule(
                    name="frozen_soil",
                    description="Check for frozen soil conditions",
                    check_function=self._check_frozen_soil,
                    severity="info"
                ),
            ])

        if self.level in [QCLevel.LEVEL_3, QCLevel.LEVEL_4]:
            rules.extend([
                QCRule(
                    name="sensor_consistency",
                    description="Check consistency between nearby sensors",
                    check_function=self._check_sensor_consistency,
                    severity="warning"
                ),
                QCRule(
                    name="physical_plausibility",
                    description="Check physical plausibility with weather",
                    check_function=self._check_physical_plausibility,
                    severity="warning"
                ),
            ])

        # Enable/disable based on level
        for rule in rules:
            if self.level == QCLevel.LEVEL_1 and rule.name not in ["range_vwc", "range_tension", "spike_detection"]:
                rule.enabled = False

        return rules

    # Rule implementations
    def _check_range_vwc(self, value: float, context: Dict) -> Dict[str, Any]:
        """Check VWC within physical range"""
        min_vwc = context.get("thresholds", {}).get("range_min_vwc", 0.01)
        max_vwc = context.get("thresholds", {}).get("range_max_vwc", 0.50)

        passed = min_vwc <= value <= max_vwc

        return {
            "passed": passed,
            "details": {
                "min": min_vwc,
                "max": max_vwc,
                "value": value
            }
        }

    def _correct_range_vwc(self, value: float, context: Dict) -> float:
        """Correct out-of-range VWC values"""
        min_vwc = context.get("thresholds", {}).get("range_min_vwc", 0.01)
        max_vwc = context.get("thresholds", {}).get("range_max_vwc", 0.50)

        if value < min_vwc:
            return min_vwc
        elif value > max_vwc:
            return max_vwc
        else:
            return value

    def _check_range_tension(self, value: float, context: Dict) -> Dict[str, Any]:
        """Check tension within physical range"""
        min_tension = context.get("thresholds", {}).get("range_min_tension", 0.0)
        max_tension = context.get("thresholds", {}).get("range_max_tension", 1500.0)

        passed = min_tension <= value <= max_tension

        return {
            "passed": passed,
            "details": {
                "min": min_tension,
                "max": max_tension,
                "value": value
            }
        }

    def _correct_range_tension(self, value: float, context: Dict) -> float:
        """Correct out-of-range tension values"""
        min_tension = context.get("thresholds", {}).get("range_min_tension", 0.0)
        max_tension = context.get("thresholds", {}).get("range_max_tension", 1500.0)

        if value < min_tension:
            return min_tension
        elif value > max_tension:
            return max_tension
        else:
            return value

    def _check_spike(self, value: float, context: Dict) -> Dict[str, Any]:
        """Detect spikes relative to historical data"""
        observation = context.get("observation")
        if not observation:
            return {"passed": True}

        # Get historical values from context
        historical = context.get("historical_values", [])
        if len(historical) < 3:
            return {"passed": True}

        # Calculate statistics
        hist_mean = np.mean(historical)
        hist_std = np.std(historical)

        if hist_std == 0:
            return {"passed": True}

        # Check if value is spike
        threshold = context.get("thresholds", {}).get("spike_threshold_std", 3.0)
        z_score = abs(value - hist_mean) / hist_std

        passed = z_score <= threshold

        return {
            "passed": passed,
            "details": {
                "z_score": z_score,
                "threshold": threshold,
                "hist_mean": hist_mean,
                "hist_std": hist_std
            }
        }

    def _correct_spike(self, value: float, context: Dict) -> float:
        """Correct spike by replacing with historical median"""
        historical = context.get("historical_values", [])
        if len(historical) > 0:
            return float(np.median(historical))
        else:
            return value

    def _check_rate_of_change(self, value: float, context: Dict) -> Dict[str, Any]:
        """Check rate of change compared to previous value"""
        observation = context.get("observation")
        if not observation:
            return {"passed": True}

        # Get previous value from context
        previous_value = context.get("previous_value")
        if previous_value is None:
            return {"passed": True}

        # Calculate rate of change
        rate = abs(value - previous_value)
        max_rate = context.get("thresholds", {}).get("rate_of_change_max", 0.1)

        passed = rate <= max_rate

        return {
            "passed": passed,
            "details": {
                "rate": rate,
                "max_rate": max_rate,
                "previous": previous_value
            }
        }

    def _correct_rate_of_change(self, value: float, context: Dict) -> float:
        """Correct excessive rate of change"""
        previous_value = context.get("previous_value")
        max_rate = context.get("thresholds", {}).get("rate_of_change_max", 0.1)

        if previous_value is None:
            return value

        if abs(value - previous_value) > max_rate:
            # Cap the change
            if value > previous_value:
                return previous_value + max_rate
            else:
                return previous_value - max_rate
        else:
            return value

    def _check_frozen_soil(self, value: float, context: Dict) -> Dict[str, Any]:
        """Check if soil might be frozen"""
        observation = context.get("observation")
        if not observation:
            return {"passed": True}

        # Get soil temperature from context
        soil_temp = context.get("soil_temperature")
        freezing_temp = context.get("thresholds", {}).get("freezing_temp", 0.0)

        if soil_temp is None:
            return {"passed": True}

        # Soil might be frozen if temperature below freezing
        # and moisture is not changing much (indicative of frozen state)
        passed = soil_temp > freezing_temp

        return {
            "passed": passed,
            "details": {
                "soil_temp": soil_temp,
                "freezing_temp": freezing_temp
            }
        }

    def _check_sensor_consistency(self, value: float, context: Dict) -> Dict[str, Any]:
        """Check consistency with nearby sensors"""
        # This would compare with other sensors at same site/depth
        # For now, return passed (implementation depends on sensor network)
        return {"passed": True}

    def _check_physical_plausibility(self, value: float, context: Dict) -> Dict[str, Any]:
        """Check if moisture change is physically plausible given weather"""
        # This would compare expected change (from weather) with observed change
        # For now, return passed
        return {"passed": True}