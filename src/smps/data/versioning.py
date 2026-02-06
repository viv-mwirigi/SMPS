"""
Data Versioning and Provenance Tracking for SMPS.

Provides complete data lineage tracking using DVC and custom provenance logging.
Ensures all data transformations are tracked and reproducible.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import json
import hashlib
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataProvenance:
    """Provenance information for a dataset."""

    # Dataset identification
    dataset_name: str
    version: str
    created_at: datetime

    # Source information
    source_files: List[str]
    source_hashes: Dict[str, str]

    # Transformation information
    transformation_script: str
    transformation_params: Dict[str, Any]
    transformation_hash: str

    # Data characteristics
    n_rows: int
    n_cols: int
    column_names: List[str]
    dtypes: Dict[str, str]

    # Quality metrics
    completeness_score: float
    uniqueness_score: float

    # Dependencies
    dependencies: List[str]  # Other datasets this depends on

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'dataset_name': self.dataset_name,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'source_files': self.source_files,
            'source_hashes': self.source_hashes,
            'transformation_script': self.transformation_script,
            'transformation_params': self.transformation_params,
            'transformation_hash': self.transformation_hash,
            'n_rows': self.n_rows,
            'n_cols': self.n_cols,
            'column_names': self.column_names,
            'dtypes': {k: str(v) for k, v in self.dtypes.items()},
            'completeness_score': self.completeness_score,
            'uniqueness_score': self.uniqueness_score,
            'dependencies': self.dependencies
        }


class DataVersionManager:
    """
    Manages data versioning and provenance tracking for SMPS.

    Provides:
    - Automatic dataset versioning
    - Provenance tracking for all transformations
    - Data quality monitoring
    - Dependency management
    """

    def __init__(self, provenance_dir: Path = None):
        self.provenance_dir = provenance_dir or Path("data/provenance")
        self.provenance_dir.mkdir(parents=True, exist_ok=True)

    def track_dataset(self,
                      df: pd.DataFrame,
                      dataset_name: str,
                      source_files: List[Union[str, Path]],
                      transformation_script: str,
                      transformation_params: Dict[str, Any] = None,
                      dependencies: List[str] = None) -> DataProvenance:
        """
        Track a dataset with full provenance information.

        Args:
            df: The dataset to track
            dataset_name: Name of the dataset
            source_files: List of source files used to create this dataset
            transformation_script: Script/module that created this dataset
            transformation_params: Parameters used in transformation
            dependencies: Other datasets this depends on

        Returns:
            DataProvenance object with complete tracking information
        """

        # Generate version hash
        version = self._generate_version_hash(df, transformation_params or {})

        # Calculate source file hashes
        source_hashes = {}
        for source_file in source_files:
            source_path = Path(source_file)
            if source_path.exists():
                source_hashes[str(source_file)] = self._calculate_file_hash(
                    source_path)

        # Calculate transformation hash
        transformation_hash = self._calculate_transformation_hash(
            transformation_script, transformation_params or {}
        )

        # Calculate data quality metrics
        completeness_score = self._calculate_completeness_score(df)
        uniqueness_score = self._calculate_uniqueness_score(df)

        # Create provenance record
        provenance = DataProvenance(
            dataset_name=dataset_name,
            version=version,
            created_at=datetime.now(),
            source_files=[str(f) for f in source_files],
            source_hashes=source_hashes,
            transformation_script=transformation_script,
            transformation_params=transformation_params or {},
            transformation_hash=transformation_hash,
            n_rows=len(df),
            n_cols=len(df.columns),
            column_names=list(df.columns),
            dtypes=df.dtypes.to_dict(),
            completeness_score=completeness_score,
            uniqueness_score=uniqueness_score,
            dependencies=dependencies or []
        )

        # Save provenance record
        self._save_provenance(provenance)

        logger.info(f"Tracked dataset '{dataset_name}' with version {version}")
        return provenance

    def get_provenance(self, dataset_name: str, version: str = None) -> Optional[DataProvenance]:
        """Retrieve provenance information for a dataset."""
        if version is None:
            # Get latest version
            provenance_files = list(
                self.provenance_dir.glob(f"{dataset_name}_*.json"))
            if not provenance_files:
                return None
            # Sort by creation time (filename includes timestamp)
            latest_file = max(provenance_files,
                              key=lambda x: x.stat().st_mtime)
        else:
            provenance_file = self.provenance_dir / \
                f"{dataset_name}_{version}.json"
            if not provenance_file.exists():
                return None
            latest_file = provenance_file

        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)

            # Convert back to DataProvenance
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            # Keep as strings
            data['dtypes'] = {k: v for k, v in data['dtypes'].items()}

            return DataProvenance(**data)

        except Exception as e:
            logger.error(f"Error loading provenance for {dataset_name}: {e}")
            return None

    def list_versions(self, dataset_name: str) -> List[str]:
        """List all versions of a dataset."""
        provenance_files = list(
            self.provenance_dir.glob(f"{dataset_name}_*.json"))
        versions = []
        for file_path in provenance_files:
            # Extract version from filename
            filename = file_path.stem
            version = filename.split(
                '_', 1)[1] if '_' in filename else filename
            versions.append(version)

        return sorted(versions, reverse=True)  # Most recent first

    def verify_dataset_integrity(self, df: pd.DataFrame, dataset_name: str,
                                 version: str = None) -> bool:
        """Verify that a dataset matches its provenance record."""
        provenance = self.get_provenance(dataset_name, version)
        if not provenance:
            logger.warning(f"No provenance record found for {dataset_name}")
            return False

        # Check basic characteristics
        checks = [
            len(df) == provenance.n_rows,
            len(df.columns) == provenance.n_cols,
            list(df.columns) == provenance.column_names,
        ]

        if not all(checks):
            logger.error(f"Dataset {dataset_name} integrity check failed")
            return False

        # Recalculate hash to verify content
        current_version = self._generate_version_hash(
            df, provenance.transformation_params)
        if current_version != provenance.version:
            logger.error(f"Dataset {dataset_name} content hash mismatch")
            return False

        logger.info(f"Dataset {dataset_name} integrity verified")
        return True

    def _generate_version_hash(self, df: pd.DataFrame, params: Dict[str, Any]) -> str:
        """Generate a version hash based on data content and parameters."""
        # Create a string representation of the data and parameters
        content_str = str(df.values.tobytes()) + str(sorted(params.items()))
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _calculate_transformation_hash(self, script: str, params: Dict[str, Any]) -> str:
        """Calculate hash of transformation script and parameters."""
        content = script + str(sorted(params.items()))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _calculate_completeness_score(self, df: pd.DataFrame) -> float:
        """Calculate data completeness score (0-1, higher is better)."""
        total_cells = df.shape[0] * df.shape[1]
        if total_cells == 0:
            return 1.0

        missing_cells = df.isnull().sum().sum()
        return 1.0 - (missing_cells / total_cells)

    def _calculate_uniqueness_score(self, df: pd.DataFrame) -> float:
        """Calculate data uniqueness score (0-1, higher is better)."""
        if len(df) == 0:
            return 1.0

        # Calculate uniqueness for each column
        uniqueness_scores = []
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                # For categorical/text columns
                unique_ratio = df[col].nunique() / len(df)
                uniqueness_scores.append(unique_ratio)
            else:
                # For numeric columns, use coefficient of variation
                if df[col].std() != 0:
                    cv = abs(df[col].std() / df[col].mean()
                             ) if df[col].mean() != 0 else 0
                    uniqueness_scores.append(min(cv, 1.0))  # Cap at 1.0
                else:
                    uniqueness_scores.append(0.0)

        return np.mean(uniqueness_scores) if uniqueness_scores else 0.0

    def _save_provenance(self, provenance: DataProvenance) -> None:
        """Save provenance record to disk."""
        filename = f"{provenance.dataset_name}_{provenance.version}.json"
        filepath = self.provenance_dir / filename

        with open(filepath, 'w') as f:
            json.dump(provenance.to_dict(), f, indent=2, default=str)

        logger.debug(f"Saved provenance record to {filepath}")


# Global data version manager instance
data_version_manager = DataVersionManager()


def track_dataset(df: pd.DataFrame, dataset_name: str, source_files: List[Union[str, Path]],
                  transformation_script: str, **kwargs) -> DataProvenance:
    """Convenience function to track a dataset."""
    return data_version_manager.track_dataset(df, dataset_name, source_files,
                                              transformation_script, **kwargs)


def verify_dataset(df: pd.DataFrame, dataset_name: str, version: str = None) -> bool:
    """Convenience function to verify dataset integrity."""
    return data_version_manager.verify_dataset_integrity(df, dataset_name, version)
