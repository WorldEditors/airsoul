"""Unified language-token/image sequence data I/O."""

from .schema import (
    AtomType,
    ChunkView,
    ImageCodec,
    RecordView,
    TargetEntry,
    UnifiedBatch,
)
from .reader import UnifiedDatasetCollection, UnifiedMMapDataset, open_unified_dataset
from .writer import StreamingRecordWriter, UnifiedDatasetWriter
from .collate import collate_chunks
from .lane import ChunkRequest, LaneCursor, LaneScheduler
from .producer import (
    trajectory_messages,
    write_unified_messages,
    write_unified_record,
)

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
    "UnifiedDatasetCollection",
    "open_unified_dataset",
    "ChunkRequest",
    "LaneCursor",
    "LaneScheduler",
    "collate_chunks",
    "trajectory_messages",
    "write_unified_messages",
    "write_unified_record",
    "TorchTargetBatch",
    "TorchUnifiedBatch",
    "to_torch_batch",
]
