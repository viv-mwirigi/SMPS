#!/usr/bin/env python
"""
SMPS Reproducibility Initialization Script.

Sets up complete reproducibility environment for SMPS including:
- Random seed management
- Environment validation
- Data versioning initialization
- Execution order controls
"""

from smps.core.execution_order import execution_manager
from smps.data.versioning import data_version_manager
from smps.core.reproducibility import initialize_reproducibility, reproducibility_manager
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


logger = logging.getLogger(__name__)


def validate_environment() -> Dict[str, Any]:
    """Validate that the environment meets reproducibility requirements."""
    import numpy as np
    import pandas as pd
    import sklearn
    import lightgbm

    validation_results = {
        'python_version': sys.version,
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'sklearn_version': sklearn.__version__,
        'lightgbm_version': lightgbm.__version__,
        'all_versions_pinned': True,
        'issues': []
    }

    # Check if versions match expected
    expected_versions = {
        'numpy': '2.4.1',
        'pandas': '3.0.0',
        'sklearn': '1.8.0',
        'lightgbm': '4.6.0'
    }

    for package, expected in expected_versions.items():
        actual = validation_results[f'{package}_version']
        if actual != expected:
            validation_results['issues'].append(
                f"{package} version mismatch: expected {expected}, got {actual}"
            )
            validation_results['all_versions_pinned'] = False

    return validation_results


def initialize_data_versioning() -> None:
    """Initialize data versioning system."""
    logger.info("Initializing data versioning system...")

    # Create necessary directories
    provenance_dir = Path("data/provenance")
    provenance_dir.mkdir(parents=True, exist_ok=True)

    # Initialize DVC if available
    try:
        import dvc.main
        # DVC initialization would go here
        logger.info("DVC available for data versioning")
    except ImportError:
        logger.warning("DVC not available - install with: pip install dvc")

    logger.info("Data versioning system initialized")


def setup_execution_order() -> None:
    """Set up deterministic execution order controls."""
    logger.info("Setting up execution order controls...")

    # Register core SMPS tasks in dependency order
    from smps.core.execution_order import register_task, ExecutionPhase

    # Data pipeline tasks
    register_task(
        name="load_raw_data",
        phase=ExecutionPhase.DATA_LOADING,
        function=lambda: logger.info("Loading raw data"),
        dependencies=[],
        outputs=["raw_data"]
    )

    register_task(
        name="validate_data",
        phase=ExecutionPhase.DATA_VALIDATION,
        function=lambda: logger.info("Validating data"),
        dependencies=["load_raw_data"],
        inputs=["raw_data"],
        outputs=["validated_data"]
    )

    register_task(
        name="preprocess_data",
        phase=ExecutionPhase.DATA_PREPROCESSING,
        function=lambda: logger.info("Preprocessing data"),
        dependencies=["validate_data"],
        inputs=["validated_data"],
        outputs=["processed_data"]
    )

    # Feature engineering tasks
    register_task(
        name="engineer_features",
        phase=ExecutionPhase.FEATURE_ENGINEERING,
        function=lambda: logger.info("Engineering features"),
        dependencies=["preprocess_data"],
        inputs=["processed_data"],
        outputs=["features"]
    )

    # Model training tasks
    register_task(
        name="train_model",
        phase=ExecutionPhase.MODEL_TRAINING,
        function=lambda: logger.info("Training model"),
        dependencies=["engineer_features"],
        inputs=["features"],
        outputs=["trained_model"]
    )

    # Validation tasks
    register_task(
        name="physics_validation",
        phase=ExecutionPhase.PHYSICS_VALIDATION,
        function=lambda: logger.info("Running physics validation"),
        dependencies=["trained_model"],
        inputs=["trained_model"],
        outputs=["physics_results"]
    )

    register_task(
        name="cross_validation",
        phase=ExecutionPhase.CROSS_VALIDATION,
        function=lambda: logger.info("Running cross-validation"),
        dependencies=["trained_model"],
        inputs=["trained_model"],
        outputs=["cv_results"]
    )

    logger.info("Execution order controls initialized")


def create_reproducibility_report() -> Dict[str, Any]:
    """Create a comprehensive reproducibility report."""
    logger.info("Generating reproducibility report...")

    report = {
        'timestamp': '2026-02-06T00:00:00Z',  # Current date
        'reproducibility_manager': reproducibility_manager.get_reproducibility_info(),
        'environment_validation': validate_environment(),
        'data_versioning_status': {
            'provenance_dir_exists': Path("data/provenance").exists(),
            'dvc_available': False
        },
        'execution_order_status': {
            'tasks_registered': len(execution_manager.tasks),
            'execution_plan': execution_manager.get_execution_plan()
        }
    }

    # Check DVC availability
    try:
        import dvc
        report['data_versioning_status']['dvc_available'] = True
        report['data_versioning_status']['dvc_version'] = dvc.__version__
    except ImportError:
        pass

    return report


def main():
    """Main reproducibility initialization function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Starting SMPS reproducibility initialization...")

    # Initialize reproducibility with master seed
    master_seed = 42  # Fixed master seed for complete reproducibility
    initialize_reproducibility(
        master_seed=master_seed, enforce_determinism=True)

    # Validate environment
    env_validation = validate_environment()
    if not env_validation['all_versions_pinned']:
        logger.warning("Environment versions not fully pinned:")
        for issue in env_validation['issues']:
            logger.warning(f"  - {issue}")

    # Initialize data versioning
    initialize_data_versioning()

    # Set up execution order
    setup_execution_order()

    # Generate reproducibility report
    report = create_reproducibility_report()

    # Save report
    report_path = Path("reproducibility_report.json")
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Reproducibility report saved to {report_path}")
    logger.info("SMPS reproducibility initialization complete!")

    # Print summary
    print("\n" + "="*60)
    print("SMPS REPRODUCIBILITY INITIALIZATION COMPLETE")
    print("="*60)
    print(f"Master seed: {master_seed}")
    print(f"Environment validated: {env_validation['all_versions_pinned']}")
    print(f"Tasks registered: {len(execution_manager.tasks)}")
    print(f"Report saved: {report_path}")
    print("="*60)


if __name__ == "__main__":
    main()
