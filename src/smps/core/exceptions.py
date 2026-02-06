"""
Custom exceptions for SWPPS.
"""


class SWPPSError(Exception):
    """Base exception for SWPPS system."""
    pass


class ConfigurationError(SWPPSError):
    """Error in system configuration."""
    pass


class SensorError(SWPPSError):
    """Error reading from or communicating with sensors."""
    pass


class DataFetchError(SWPPSError):
    """Error fetching external data (weather, soil, etc.)."""
    pass


class PhysicsModelError(SWPPSError):
    """Error in physics model computation."""
    pass


class MLModelError(SWPPSError):
    """Error in ML model training or prediction."""
    pass


class InsufficientDataError(SWPPSError):
    """Not enough data for training or prediction."""
    pass


class CalibrationError(SWPPSError):
    """Error in sensor or model calibration."""
    pass


class ActuationError(SWPPSError):
    """Error triggering irrigation actuation."""
    pass


# Alias for compatibility
ActuatorError = ActuationError


class ValidationError(SWPPSError):
    """Data validation error."""
    pass
