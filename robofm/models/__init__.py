"""RoboFM task-agnostic sequence models.

Legacy benchmark models are no longer imported from the package namespace.
Keeping this module narrow prevents optional benchmark dependencies from being
loaded when the unified runtime is used.
"""

from .unified_sequence import (
    RawImageEncoder,
    UnifiedModelConfig,
    UnifiedModelOutput,
    UnifiedSequenceModel,
)

__all__ = [
    "RawImageEncoder",
    "UnifiedModelConfig",
    "UnifiedModelOutput",
    "UnifiedSequenceModel",
]
