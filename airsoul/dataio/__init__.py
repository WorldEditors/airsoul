"""Unified language-token/image sequence data I/O."""

from .schema import (
    AtomType,
    ChunkView,
    ImageCodec,
    RecordView,
    TargetEntry,
    UnifiedBatch,
)
from .reader import UnifiedMMapDataset
from .writer import StreamingRecordWriter, UnifiedDatasetWriter
from .collate import collate_chunks
from .lane import ChunkRequest, LaneCursor, LaneScheduler

# Torch conversion remains optional so pure Data I/O tools only require NumPy.
try:
    from .torch_batch import TorchTargetBatch, TorchUnifiedBatch, to_torch_batch
except ImportError:  # pragma: no cover - exercised in NumPy-only environments
    TorchTargetBatch = None
    TorchUnifiedBatch = None
    to_torch_batch = None

__all__ = [
    "AtomType",
    "ChunkView",
    "ImageCodec",
    "RecordView",
    "TargetEntry",
    "UnifiedBatch",
    "UnifiedDatasetWriter",
    "StreamingRecordWriter",
    "UnifiedMMapDataset",
    "ChunkRequest",
    "LaneCursor",
    "LaneScheduler",
    "collate_chunks",
    "TorchTargetBatch",
    "TorchUnifiedBatch",
    "to_torch_batch",
]
