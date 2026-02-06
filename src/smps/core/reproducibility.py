"""
Reproducibility Configuration for SMPS.

Centralized random seed management and reproducibility controls.
Ensures all stochastic processes are deterministic and reproducible.
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any
import numpy as np
import random

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducibility controls."""

    # Master seed - all other seeds derive from this
    master_seed: int = 42

    # Component-specific seeds (derived from master)
    seeds: Dict[str, int] = None

    # Whether to enforce deterministic operations
    enforce_determinism: bool = True

    # Random number generator state tracking
    track_rng_state: bool = True

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = self._derive_component_seeds()

    def _derive_component_seeds(self) -> Dict[str, int]:
        """Derive deterministic seeds for all components from master seed."""
        rng = np.random.RandomState(self.master_seed)

        return {
            # ML training
            'sklearn': rng.randint(0, 2**32),
            'lightgbm': rng.randint(0, 2**32),
            'torch': rng.randint(0, 2**32),
            'numpy': rng.randint(0, 2**32),
            'python_random': rng.randint(0, 2**32),

            # Cross-validation
            'cv_split': rng.randint(0, 2**32),

            # Data processing
            'data_split': rng.randint(0, 2**32),
            'feature_engineering': rng.randint(0, 2**32),

            # Physics validation
            'physics_validation': rng.randint(0, 2**32),
            'sensitivity_analysis': rng.randint(0, 2**32),

            # Ensemble methods
            'ensemble_generation': rng.randint(0, 2**32),
            'bootstrap': rng.randint(0, 2**32),

            # Hyperparameter optimization
            'hyperopt': rng.randint(0, 2**32),
        }


class ReproducibilityManager:
    """
    Centralized manager for all random seed and reproducibility controls.

    Ensures that all stochastic operations in SMPS are deterministic and
    reproducible across different runs and environments.
    """

    def __init__(self, config: Optional[ReproducibilityConfig] = None):
        self.config = config or ReproducibilityConfig()
        self._original_states = {}
        self._is_initialized = False

    def initialize_reproducibility(self) -> None:
        """Initialize all random number generators with deterministic seeds."""
        if self._is_initialized:
            logger.warning("Reproducibility already initialized")
            return

        logger.info(
            f"Initializing reproducibility with master seed: {self.config.master_seed}")

        # Store original states for potential restoration
        if self.config.track_rng_state:
            self._store_original_states()

        # Set Python random
        random.seed(self.config.seeds['python_random'])

        # Set NumPy
        np.random.seed(self.config.seeds['numpy'])

        # Set environment variables for additional determinism
        os.environ['PYTHONHASHSEED'] = str(self.config.seeds['python_random'])
        # For CUDA determinism
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        # Configure PyTorch if available
        try:
            import torch
            torch.manual_seed(self.config.seeds['torch'])
            torch.cuda.manual_seed_all(self.config.seeds['torch'])
            if self.config.enforce_determinism:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

        # Configure scikit-learn
        try:
            import sklearn
            sklearn.utils.check_random_state(self.config.seeds['sklearn'])
        except ImportError:
            pass

        # Configure LightGBM
        os.environ['LIGHTGBM_SEED'] = str(self.config.seeds['lightgbm'])

        self._is_initialized = True
        logger.info("Reproducibility initialization complete")

    def get_seed(self, component: str) -> int:
        """Get the deterministic seed for a specific component."""
        if component not in self.config.seeds:
            # Generate a new seed deterministically if component not predefined
            rng = np.random.RandomState(self.config.master_seed)
            for _ in range(len(self.config.seeds) + 1):  # +1 for new component
                rng.randint(0, 2**32)
            new_seed = rng.randint(0, 2**32)
            self.config.seeds[component] = new_seed
            logger.info(
                f"Generated new seed for component '{component}': {new_seed}")

        return self.config.seeds[component]

    def create_rng(self, component: str) -> np.random.RandomState:
        """Create a RandomState instance for a specific component."""
        return np.random.RandomState(self.get_seed(component))

    def reset_component_rng(self, component: str) -> None:
        """Reset the RNG state for a specific component."""
        if component == 'numpy':
            np.random.seed(self.config.seeds['numpy'])
        elif component == 'python_random':
            random.seed(self.config.seeds['python_random'])
        elif component == 'torch':
            try:
                import torch
                torch.manual_seed(self.config.seeds['torch'])
            except ImportError:
                pass
        else:
            logger.warning(
                f"Cannot reset RNG for unknown component: {component}")

    def _store_original_states(self) -> None:
        """Store original RNG states for potential restoration."""
        try:
            self._original_states = {
                'python_random': random.getstate(),
                'numpy': np.random.get_state(),
            }

            try:
                import torch
                self._original_states['torch'] = torch.get_rng_state()
            except ImportError:
                pass

        except Exception as e:
            logger.warning(f"Could not store original RNG states: {e}")

    def restore_original_states(self) -> None:
        """Restore original RNG states (if stored)."""
        if not self._original_states:
            logger.warning("No original states stored to restore")
            return

        try:
            if 'python_random' in self._original_states:
                random.setstate(self._original_states['python_random'])
            if 'numpy' in self._original_states:
                np.random.set_state(self._original_states['numpy'])
            if 'torch' in self._original_states:
                try:
                    import torch
                    torch.set_rng_state(self._original_states['torch'])
                except ImportError:
                    pass

            logger.info("Original RNG states restored")

        except Exception as e:
            logger.error(f"Could not restore original RNG states: {e}")

    def get_reproducibility_info(self) -> Dict[str, Any]:
        """Get information about current reproducibility configuration."""
        return {
            'master_seed': self.config.master_seed,
            'component_seeds': self.config.seeds.copy(),
            'enforce_determinism': self.config.enforce_determinism,
            'initialized': self._is_initialized,
            'python_version': os.sys.version,
            'numpy_version': np.__version__,
            'environment_variables': {
                'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
                'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                'LIGHTGBM_SEED': os.environ.get('LIGHTGBM_SEED'),
            }
        }


# Global reproducibility manager instance
reproducibility_manager = ReproducibilityManager()


def initialize_reproducibility(master_seed: int = 42,
                               enforce_determinism: bool = True) -> None:
    """Convenience function to initialize reproducibility globally."""
    config = ReproducibilityConfig(
        master_seed=master_seed,
        enforce_determinism=enforce_determinism
    )
    global reproducibility_manager
    reproducibility_manager = ReproducibilityManager(config)
    reproducibility_manager.initialize_reproducibility()


def get_seed(component: str) -> int:
    """Convenience function to get a component-specific seed."""
    return reproducibility_manager.get_seed(component)


def create_rng(component: str) -> np.random.RandomState:
    """Convenience function to create a component-specific RNG."""
    return reproducibility_manager.create_rng(component)
