"""
Custom exception hierarchy for the Smps system.
Provides clear error categories and rich error information.
"""
from typing import Optional, Any, Dict
from dataclasses import dataclass


@dataclass
class ErrorContext:
    """Context information for errors"""
    site_id: Optional[str] = None
    date: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class SmpsError(Exception):
    """Base exception for all Smps errors"""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.message = message
        self.context = context or ErrorContext()
    
    def __str__(self) -> str:
        context_str = ""
        if self.context.site_id:
            context_str += f" [Site: {self.context.site_id}]"
        if self.context.date:
            context_str += f" [Date: {self.context.date}]"
        if self.context.component:
            context_str += f" [Component: {self.context.component}]"
        
        return f"{self.__class__.__name__}: {self.message}{context_str}"


# Data-related errors
class DataError(SmpsError):
    """Base class for data-related errors"""
    pass


class DataSourceError(DataError):
    """Error fetching data from source"""
    pass


class DataQualityError(DataError):
    """Data quality issue"""
    pass


class MissingDataError(DataError):
    """Required data is missing"""
    pass


class DataValidationError(DataError):
    """Data validation failed"""
    pass


# Physics model errors
class PhysicsModelError(SmpsError):
    """Base class for physics model errors"""
    pass


class WaterBalanceError(PhysicsModelError):
    """Water balance violation"""
    pass


class ParameterError(PhysicsModelError):
    """Invalid model parameters"""
    pass


class ConvergenceError(PhysicsModelError):
    """Model failed to converge"""
    pass


# Model training errors
class ModelError(SmpsError):
    """Base class for model-related errors"""
    pass


class TrainingError(ModelError):
    """Error during model training"""
    pass


class ValidationError(ModelError):
    """Error during model validation"""
    pass


class InferenceError(ModelError):
    """Error during model inference"""
    pass


# Configuration errors
class ConfigurationError(SmpsError):
    """Configuration error"""
    pass


class SiteConfigurationError(ConfigurationError):
    """Site-specific configuration error"""
    pass


# Feature engineering errors
class FeatureError(SmpsError):
    """Feature engineering error"""
    pass


class FeatureValidationError(FeatureError):
    """Feature validation failed"""
    pass


# Uncertainty quantification errors
class UncertaintyError(SmpsError):
    """Uncertainty quantification error"""
    pass


class CalibrationError(UncertaintyError):
    """Calibration failed"""
    pass


# Post-processing errors
class PostProcessingError(SmpsError):
    """Post-processing error"""
    pass


class BiasCorrectionError(PostProcessingError):
    """Bias correction error"""
    pass


class PhysicalConstraintError(PostProcessingError):
    """Physical constraint violation"""
    pass


# API errors
class APIError(SmpsError):
    """API error"""
    pass


class AuthenticationError(APIError):
    """Authentication error"""
    pass


class RateLimitError(APIError):
    """Rate limit exceeded"""
    pass


class NotFoundError(APIError):
    """Resource not found"""
    pass


def handle_exception(exc: Exception, context: Optional[ErrorContext] = None) -> SmpsError:
    """
    Wrap generic exceptions in SmpsError hierarchy.
    Useful for catching and categorizing third-party exceptions.
    """
    if isinstance(exc, SmpsError):
        return exc
    
    # Map common third-party exceptions
    error_map = {
        FileNotFoundError: DataSourceError,
        ConnectionError: DataSourceError,
        TimeoutError: DataSourceError,
        ValueError: DataValidationError,
        KeyError: DataValidationError,
        RuntimeError: ModelError,
    }
    
    for exc_type, smps_exc_type in error_map.items():
        if isinstance(exc, exc_type):
            return smps_exc_type(str(exc), context)
    
    # Default to generic SmpsError
    return SmpsError(str(exc), context)