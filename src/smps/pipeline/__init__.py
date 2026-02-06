"""SWPPS Pipeline Package.

This package contains the canonical pipeline implementations under
`swpps/pipeline/`, but the main orchestrator class historically lives in the
file-based module `swpps/pipeline.py`.

Re-export `SWPPSPipeline` here so user code can reliably do:
    from smps.pipeline import SWPPSPipeline
"""

from .canonical_pipeline import CanonicalPsiPipeline, CanonicalPipelineConfig
from .harmonizer import Harmonizer, HarmonizerConfig

# Re-export the file-based orchestrator pipeline.
# Using a relative import from the parent package avoids re-loading it.
try:
    from .. import SWPPSPipeline, PipelineConfig, create_pipeline
except (ImportError, AttributeError):  # pragma: no cover
    SWPPSPipeline = None  # type: ignore[assignment]
    "PipelineConfig",
    "create_pipeline",
    create_pipeline = None  # type: ignore[assignment]


__all__ = [
    "CanonicalPsiPipeline",
    "CanonicalPipelineConfig",
    "Harmonizer",
    "HarmonizerConfig",
    "SWPPSPipeline",
    "PipelineConfig",
    "create_pipeline",
]
