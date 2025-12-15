"""
SMPS Pipeline Module.

Provides data pipeline components for building canonical tables
and processing soil moisture data.
"""

# Defer imports to avoid circular dependencies and missing modules
# Import directly from smps.pipeline.canonical when needed

__all__ = [
    "CanonicalTableBuilder",
    "CanonicalTableManager",
]


def __getattr__(name):
    """Lazy import to handle missing dependencies gracefully."""
    if name in ("CanonicalTableBuilder", "CanonicalTableManager"):
        from smps.pipeline.canonical import CanonicalTableBuilder, CanonicalTableManager
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
