"""
Base classes for data sources in SWPPS.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Dict, Any, Optional


@dataclass
class DataFetchRequest:
    """Base request for data fetching."""
    site_id: str
    start_date: date
    end_date: date
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class DataSourceConfig:
    """Configuration for data sources."""
    name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: int = 30
    retry_attempts: int = 3


class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def fetch_data(self, request: DataFetchRequest) -> Any:
        """Fetch data for the given request."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the data source is available."""
        pass
