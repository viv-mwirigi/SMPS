"""
Physical constraints and validators for soil moisture modeling.
Ensures predictions remain physically plausible.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

from smps.core.types import SoilMoistureVWC, SoilParameters
from smps.core.exceptions import PhysicalConstraintError


@dataclass
class PhysicalConstraint:
    """Definition of a physical constraint"""
    name: str
    description: str
    check_function: Callable[[Dict], bool]
    correction_function: Optional[Callable[[Dict], Dict]] = None
    severity: str = "warning"  # "warning", "error", "adjust"
    tolerance: float = 1e-6


class PhysicalConstraintEnforcer:
    """
    Enforces physical constraints on soil moisture predictions.
    Can either raise errors or apply corrections.
    """
    
    def __init__(self, soil_params: SoilParameters):
        self.soil_params = soil_params
        self.constraints = self._initialize_constraints()
        self.violation_history = []
    
    def _initialize_constraints(self) -> List[PhysicalConstraint]:
        """Initialize all physical constraints"""
        return [
            PhysicalConstraint(
                name="porosity_limit",
                description="Soil moisture cannot exceed porosity",
                check_function=self._check_porosity_limit,
                correction_function=self._correct_porosity_limit,
                severity="adjust"
            ),
            PhysicalConstraint(
                name="wilting_point_limit",
                description="Soil moisture cannot be below wilting point (except during extreme drying)",
                check_function=self._check_wilting_point_limit,
                correction_function=self._correct_wilting_point_limit,
                severity="warning"
            ),
            PhysicalConstraint(
                name="depth_coherence",
                description="Surface soil moisture should not be significantly higher than root zone (except during infiltration)",
                check_function=self._check_depth_coherence,
                correction_function=self._correct_depth_coherence,
                severity="adjust"
            ),
            PhysicalConstraint(
                name="water_balance",
                description="Water balance must be conserved",
                check_function=self._check_water_balance,
                severity="error"
            ),
            PhysicalConstraint(
                name="monotonic_drying",
                description="Soil moisture should not increase without water input",
                check_function=self._check_monotonic_drying,
                severity="warning"
            ),
        ]
    
    def enforce(
        self,
        predictions: Dict[str, SoilMoistureVWC],
        fluxes: Optional[Dict[str, float]] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, SoilMoistureVWC]:
        """
        Enforce all physical constraints on predictions.
        
        Args:
            predictions: Dictionary with soil moisture predictions
            fluxes: Water fluxes for context
            context: Additional context information
            
        Returns:
            Corrected predictions
        """
        context = context or {}
        context.update({
            "predictions": predictions.copy(),
            "fluxes": fluxes or {},
            "soil_params": self.soil_params
        })
        
        corrected_predictions = predictions.copy()
        
        for constraint in self.constraints:
            try:
                # Check constraint
                if not constraint.check_function(context):
                    self._log_violation(constraint, context)
                    
                    # Apply correction if available
                    if constraint.correction_function is not None:
                        if constraint.severity in ["adjust", "warning"]:
                            corrected = constraint.correction_function(context)
                            corrected_predictions.update(corrected)
                            
                            # Update context with corrected values
                            context["predictions"] = corrected_predictions.copy()
                            
                    # Raise error for severe violations
                    if constraint.severity == "error":
                        raise PhysicalConstraintError(
                            f"Violated constraint: {constraint.name} - {constraint.description}",
                            context=context
                        )
                        
            except Exception as e:
                if constraint.severity == "error":
                    raise
                else:
                    # Log but continue for warnings
                    import logging
                    logging.warning(f"Constraint {constraint.name} failed: {e}")
        
        return corrected_predictions
    
    def _check_porosity_limit(self, context: Dict) -> bool:
        """Check if soil moisture exceeds porosity"""
        predictions = context["predictions"]
        porosity = self.soil_params.porosity
        
        for layer, value in predictions.items():
            if value > porosity + context.get("tolerance", 1e-6):
                return False
        
        return True
    
    def _correct_porosity_limit(self, context: Dict) -> Dict:
        """Correct soil moisture that exceeds porosity"""
        predictions = context["predictions"]
        porosity = self.soil_params.porosity
        
        corrected = {}
        for layer, value in predictions.items():
            if value > porosity:
                corrected[layer] = porosity - 1e-6  # Slightly below porosity
        
        return corrected
    
    def _check_wilting_point_limit(self, context: Dict) -> bool:
        """Check if soil moisture is below wilting point"""
        predictions = context["predictions"]
        wilting_point = self.soil_params.wilting_point
        
        for layer, value in predictions.items():
            if value < wilting_point - context.get("tolerance", 1e-6):
                # Check if there's been extreme drying (low fluxes)
                fluxes = context.get("fluxes", {})
                et = fluxes.get("evapotranspiration", 0)
                if et < 1.0:  # Not much ET
                    return False
        
        return True
    
    def _correct_wilting_point_limit(self, context: Dict) -> Dict:
        """Correct soil moisture below wilting point"""
        predictions = context["predictions"]
        wilting_point = self.soil_params.wilting_point
        
        corrected = {}
        for layer, value in predictions.items():
            if value < wilting_point:
                corrected[layer] = wilting_point + 1e-6
        
        return corrected

    def _check_depth_coherence(self, context: Dict) -> bool:
        """Ensure surface layer is not unrealistically wetter than root zone."""
        predictions = context["predictions"]
        if "surface" not in predictions or "root_zone" not in predictions:
            return True
        diff = predictions["surface"] - predictions["root_zone"]
        return diff <= 0.05 + context.get("tolerance", 1e-6)
    
    def _check_depth_coherence_with_infiltration(self, context: Dict) -> bool:
        """
        Advanced depth coherence check with infiltration modeling.
        Considers:
        1. Infiltration front propagation
        2. Time since last infiltration
        3. Soil hydraulic properties
        """
        predictions = context["predictions"]
        fluxes = context.get("fluxes", {})

        if "surface" not in predictions or "root_zone" not in predictions:
            return True

        surface_value = predictions["surface"]
        root_value = predictions["root_zone"]

        hours_since_infiltration = context.get("hours_since_infiltration", 24)
        cumulative_infiltration = context.get("cumulative_infiltration_7d", 0)

        if cumulative_infiltration > 0:
            delta_theta = self.soil_params.porosity - context.get("initial_moisture", 0.2)
            if delta_theta > 0:
                front_depth_m = cumulative_infiltration / 1000 / delta_theta  # mm → m
            else:
                front_depth_m = 0.1
        else:
            front_depth_m = 0.0

        if front_depth_m < 0.15:
            max_diff = 0.15
        elif hours_since_infiltration < 6:
            max_diff = 0.1
        elif hours_since_infiltration < 24:
            max_diff = 0.05
        else:
            max_diff = 0.02

        return surface_value - root_value <= max_diff

    def _correct_depth_coherence(self, context: Dict) -> Dict:
        """Correct depth coherence violations"""
        predictions = context["predictions"]
        fluxes = context.get("fluxes", {})
        
        if "surface" not in predictions or "root_zone" not in predictions:
            return {}
        
        surface_value = predictions["surface"]
        root_value = predictions["root_zone"]
        infiltration = fluxes.get("infiltration", 0)
        
        corrected = {}
        
        if infiltration > 0.1:
            # After infiltration, allow some difference
            max_diff = 0.1
        else:
            # Normally, limit difference
            max_diff = 0.05
        
        if surface_value - root_value > max_diff:
            # Adjust both values minimally toward mean
            mean_value = (surface_value + root_value) / 2
            corrected["surface"] = mean_value + max_diff / 2
            corrected["root_zone"] = mean_value - max_diff / 2
        
        return corrected
    
    def _check_water_balance(self, context: Dict) -> bool:
        """Check water balance closure"""
        fluxes = context.get("fluxes", {})
        tolerance = context.get("water_balance_tolerance", 1.0)  # 1 mm
        
        # Simplified water balance: inputs - outputs = ΔS
        inputs = fluxes.get("precipitation", 0) + fluxes.get("irrigation", 0)
        outputs = (
            fluxes.get("runoff", 0) +
            fluxes.get("evaporation", 0) +
            fluxes.get("transpiration", 0) +
            fluxes.get("drainage", 0)
        )
        
        # ΔS should be in context
        delta_storage = context.get("delta_storage", 0)
        
        balance_error = abs((inputs - outputs) - delta_storage)
        return balance_error <= tolerance
    
    def _check_monotonic_drying(self, context: Dict) -> bool:
        """Check that soil moisture doesn't increase without water input"""
        predictions = context["predictions"]
        previous_predictions = context.get("previous_predictions", {})
        fluxes = context.get("fluxes", {})
        
        if not previous_predictions:
            return True
        
        water_input = fluxes.get("precipitation", 0) + fluxes.get("irrigation", 0)
        
        for layer in predictions:
            if layer in previous_predictions:
                current = predictions[layer]
                previous = previous_predictions[layer]
                
                # Soil moisture increased
                if current > previous + 1e-6:
                    # Check if there was water input
                    if water_input < 0.1:  # Less than 0.1 mm input
                        return False
        
        return True
    
    def _log_violation(self, constraint: PhysicalConstraint, context: Dict):
        """Log constraint violation"""
        violation = {
            "constraint": constraint.name,
            "description": constraint.description,
            "severity": constraint.severity,
            "timestamp": context.get("timestamp"),
            "predictions": context.get("predictions", {}),
            "fluxes": context.get("fluxes", {}),
        }
        self.violation_history.append(violation)
    
    def get_violation_summary(self) -> Dict:
        """Get summary of constraint violations"""
        if not self.violation_history:
            return {"total_violations": 0, "by_severity": {}}
        
        by_severity = {}
        for violation in self.violation_history:
            severity = violation["severity"]
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            "total_violations": len(self.violation_history),
            "by_severity": by_severity,
            "recent_violations": self.violation_history[-10:]  # Last 10
        }


def calculate_water_storage(
    soil_moisture_vwc: float,
    layer_depth_m: float,
    area_m2: float = 1.0
) -> float:
    """
    Calculate water storage from volumetric water content.
    
    Args:
        soil_moisture_vwc: Volumetric water content (m³/m³)
        layer_depth_m: Soil layer depth (m)
        area_m2: Area (m²), default 1 m² for depth-equivalent
        
    Returns:
        Water storage in mm (depth equivalent)
    """
    # m³/m³ * m * 1000 = mm
    return soil_moisture_vwc * layer_depth_m * 1000


def convert_soil_tension_to_vwc(
    tension_kpa: float,
    soil_params: SoilParameters,
    method: str = "van_genuchten"
) -> SoilMoistureVWC:
    """
    Convert soil tension to volumetric water content.
    
    Args:
        tension_kpa: Soil water tension (kPa)
        soil_params: Soil parameters
        method: Conversion method ("van_genuchten", "campbell", "simplified")
        
    Returns:
        Volumetric water content (m³/m³)
    """
    if method == "van_genuchten":
        return _convert_tension_vwc_van_genuchten(tension_kpa, soil_params)
    elif method == "campbell":
        return _convert_tension_vwc_campbell(tension_kpa, soil_params)
    else:  # simplified
        return _convert_tension_vwc_simplified(tension_kpa, soil_params)


def _convert_tension_vwc_van_genuchten(
    tension_kpa: float,
    soil_params: SoilParameters
) -> SoilMoistureVWC:
    """
    Convert tension to VWC using van Genuchten (1980) equation.
    
    θ(h) = θ_r + (θ_s - θ_r) / [1 + (α|h|)^n]^m
    
    Note: m = 1 - 1/n for Mualem (1976) model, but can be independent.
    """
    if soil_params.van_genuchten_alpha is None or soil_params.van_genuchten_n is None:
        raise ValueError("Van Genuchten parameters not available")
    
    # Use residual water content if available, otherwise wilting point
    theta_r = getattr(soil_params, 'residual_water_content', soil_params.wilting_point)
    theta_s = soil_params.porosity
    alpha = soil_params.van_genuchten_alpha  # 1/kPa
    n = soil_params.van_genuchten_n
    
    # m parameter - check if available, otherwise use Mualem assumption
    m = getattr(soil_params, 'van_genuchten_m', None)
    if m is None:
        m = 1 - 1/n  # Mualem assumption
    
    # Handle tension (negative pressure head in van Genuchten)
    # In soil physics, tension is positive, pressure head is negative
    h = abs(tension_kpa)  # kPa, positive
    
    # Avoid division by zero or negative
    h = max(0.001, h)
    
    # Van Genuchten equation
    if (alpha * h) ** n < 1e-10:  # Very wet soil
        theta = theta_s
    else:
        denominator = (1 + (alpha * h) ** n) ** m
        theta = theta_r + (theta_s - theta_r) / denominator
    
    # Ensure bounds
    theta = np.clip(theta, theta_r, theta_s)
    
    return theta


def _convert_tension_vwc_campbell(
    tension_kpa: float,
    soil_params: SoilParameters
) -> SoilMoistureVWC:
    """
    Convert using Campbell equation.
    
    θ(h) = θ_s * (h_b/h)^(1/b) for h > h_b
    θ(h) = θ_s for h ≤ h_b
    """
    theta_s = soil_params.porosity
    
    # Campbell parameters (simplified)
    h_b = 1.0  # Air entry pressure (kPa)
    b = 5.0  # Pore size distribution parameter
    
    h = max(h_b, tension_kpa)
    
    if h <= h_b:
        return theta_s
    else:
        theta = theta_s * (h_b / h) ** (1/b)
        return np.clip(theta, soil_params.wilting_point, theta_s)


def _convert_tension_vwc_simplified(
    tension_kpa: float,
    soil_params: SoilParameters
) -> SoilMoistureVWC:
    """
    Simplified conversion using logarithmic relationship.
    """
    theta_s = soil_params.porosity
    theta_r = soil_params.wilting_point
    
    # Log-linear relationship between log(tension) and water content
    # Tension range: 10 kPa (field capacity) to 1500 kPa (wilting point)
    h_fc = 10.0  # kPa at field capacity
    h_wp = 1500.0  # kPa at wilting point
    
    if tension_kpa <= h_fc:
        return soil_params.field_capacity
    elif tension_kpa >= h_wp:
        return theta_r
    else:
        # Linear in log space
        log_h = np.log(tension_kpa)
        log_h_fc = np.log(h_fc)
        log_h_wp = np.log(h_wp)
        
        # Interpolate
        fraction = (log_h - log_h_fc) / (log_h_wp - log_h_fc)
        theta = soil_params.field_capacity + fraction * (theta_r - soil_params.field_capacity)
        
        return np.clip(theta, theta_r, theta_s)


def calculate_plant_available_water(
    soil_moisture_vwc: float,
    soil_params: SoilParameters,
    layer_depth_m: float = 0.3
) -> float:
    """
    Calculate plant available water in a soil layer.
    
    Args:
        soil_moisture_vwc: Current soil moisture (m³/m³)
        soil_params: Soil parameters
        layer_depth_m: Soil layer depth (m)
        
    Returns:
        Plant available water in mm
    """
    # Available water is between wilting point and field capacity
    available_vwc = max(0, soil_moisture_vwc - soil_params.wilting_point)
    
    # Convert to depth equivalent (mm)
    return available_vwc * layer_depth_m * 1000

class HysteresisCorrector:
    """
    Account for soil moisture hysteresis (different wetting/drying curves).
    
    Soil has different moisture-tension relationships during
    wetting vs drying due to air entrapment and contact angle effects.
    """
    
    def __init__(self, soil_params: SoilParameters):
        self.soil_params = soil_params
        self.wetting_curve = {}  # Store wetting curve parameters
        self.drying_curve = {}   # Store drying curve parameters
        self.state = "drying"    # Current state: "wetting" or "drying"
        
        # Initialize from soil parameters
        self._initialize_curves()
    
    def _initialize_curves(self):
        """Initialize wetting and drying curves from soil parameters."""
        # Simplified: drying curve is main curve, wetting curve has higher air entry
        alpha_dry = self.soil_params.van_genuchten_alpha
        
        # Wetting curve has smaller alpha (higher air entry)
        alpha_wet = alpha_dry * 0.7
        
        self.drying_curve = {
            "alpha": alpha_dry,
            "n": self.soil_params.van_genuchten_n,
            "theta_r": getattr(self.soil_params, 'residual_water_content', 
                             self.soil_params.wilting_point),
            "theta_s": self.soil_params.porosity
        }
        
        self.wetting_curve = {
            "alpha": alpha_wet,
            "n": self.soil_params.van_genuchten_n,
            "theta_r": getattr(self.soil_params, 'residual_water_content',
                             self.soil_params.wilting_point),
            "theta_s": self.soil_params.porosity * 0.95  # Lower saturation due to air entrapment
        }
    
    def update_state(self, current_theta: float, previous_theta: float):
        """Update hysteresis state based on moisture trend."""
        if current_theta > previous_theta + 0.001:
            self.state = "wetting"
        elif current_theta < previous_theta - 0.001:
            self.state = "drying"
        # else: keep current state
    
    def convert_tension_to_vwc(self, tension_kpa: float) -> float:
        """Convert tension to VWC using appropriate curve."""
        if self.state == "wetting":
            curve = self.wetting_curve
        else:
            curve = self.drying_curve
        
        # Van Genuchten equation
        theta_r = curve["theta_r"]
        theta_s = curve["theta_s"]
        alpha = curve["alpha"]
        n = curve["n"]
        m = 1 - 1/n
        
        h = max(0.001, abs(tension_kpa))
        
        if (alpha * h) ** n < 1e-10:
            theta = theta_s
        else:
            denominator = (1 + (alpha * h) ** n) ** m
            theta = theta_r + (theta_s - theta_r) / denominator
        
        return np.clip(theta, theta_r, theta_s)

def adjust_for_salinity(
    soil_moisture_vwc: float,
    salinity_ds_m: float,  # dS/m
    soil_params: SoilParameters
) -> float:
    """
    Adjust soil moisture for osmotic effects of salinity.
    
    Salinity reduces plant-available water by increasing osmotic potential.
    """
    # Osmotic potential (kPa) = -36 * EC (dS/m)
    osmotic_potential_kpa = -36 * salinity_ds_m
    
    # Equivalent tension increase due to salinity
    effective_tension = osmotic_potential_kpa
    
    # Convert current VWC to tension
    current_tension = convert_vwc_to_tension(
        soil_moisture_vwc,
        soil_params,
        method="van_genuchten"
    )
    
    # Add osmotic effect
    total_tension = current_tension + effective_tension
    
    # Convert back to VWC with total tension
    adjusted_vwc = convert_soil_tension_to_vwc(
        total_tension,
        soil_params,
        method="van_genuchten"
    )
    
    return adjusted_vwc


def convert_vwc_to_tension(
    vwc: float,
    soil_params: SoilParameters,
    method: str = "van_genuchten"
) -> float:
    """
    Inverse of van Genuchten: convert VWC to tension.
    
    h = (1/α) * [((θ_s - θ_r)/(θ - θ_r))^(1/m) - 1]^(1/n)
    """
    if method != "van_genuchten":
        raise NotImplementedError("Only van Genuchten inverse is implemented")
    
    theta_r = getattr(soil_params, 'residual_water_content', soil_params.wilting_point)
    theta_s = soil_params.porosity
    alpha = soil_params.van_genuchten_alpha
    n = soil_params.van_genuchten_n
    m = getattr(soil_params, 'van_genuchten_m', 1 - 1/n)
    
    # Ensure within bounds
    vwc = np.clip(vwc, theta_r + 1e-6, theta_s - 1e-6)
    
    # Inverse van Genuchten
    term = ((theta_s - theta_r) / (vwc - theta_r)) ** (1/m) - 1
    
    if term <= 0:
        # Very wet, tension near 0
        return 0.1  # kPa, small positive
    
    h = (1/alpha) * term ** (1/n)
    
    return h

def estimate_soil_moisture_deficit(
    current_vwc: float,
    target_vwc: float,
    soil_params: SoilParameters,
    layer_depth_m: float = 0.3
) -> float:
    """
    Estimate soil moisture deficit for irrigation scheduling.
    
    Args:
        current_vwc: Current soil moisture (m³/m³)
        target_vwc: Target soil moisture (typically field capacity)
        soil_params: Soil parameters
        layer_depth_m: Soil layer depth (m)
        
    Returns:
        Water deficit in mm
    """
    deficit_vwc = max(0, target_vwc - current_vwc)
    return deficit_vwc * layer_depth_m * 1000