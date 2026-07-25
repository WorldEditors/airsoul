"""On-disk schema and public data containers for Data I/O V1."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Mapping, Sequence
import zlib

import numpy as np


FORMAT_NAME = "robofm-unified-sequence"
FORMAT_VERSION = 1
ENDIANNESS = "little"


class AtomType(IntEnum):
    """Persistent atom types. Padding exists only in an in-memory batch."""

    LANGUAGE_TOKEN = 1
    IMAGE = 2


class ImageCodec(IntEnum):
    RAW = 0


class ImageDType(IntEnum):
    UINT8 = 1
    INT8 = 2
    INT16 = 3
    INT32 = 4
    FLOAT16 = 5
    FLOAT32 = 6
    FLOAT64 = 7


NUMPY_TO_IMAGE_DTYPE = {
    np.dtype("uint8"): ImageDType.UINT8,
    np.dtype("int8"): ImageDType.INT8,
    np.dtype("int16"): ImageDType.INT16,
    np.dtype("int32"): ImageDType.INT32,
    np.dtype("float16"): ImageDType.FLOAT16,
    np.dtype("float32"): ImageDType.FLOAT32,
    np.dtype("float64"): ImageDType.FLOAT64,
}
IMAGE_DTYPE_TO_NUMPY = {value: key for key, value in NUMPY_TO_IMAGE_DTYPE.items()}


RECORD_DTYPE = np.dtype([
    ("atom_offset", "<u8"),
    ("atom_length", "<u8"),
    ("target_offset", "<u8"),
    ("target_count", "<u8"),
    ("loss_mask_offset", "<u8"),
    ("flags", "<u4"),
    ("reserved", "<u4"),
    ("checksum", "<u8"),
])

TARGET_DTYPE = np.dtype([
    ("source_position", "<u8"),
    ("atom_type", "u1"),
    ("valid", "u1"),
    ("reserved", "<u2"),
    ("target_value", "<u8"),
    ("weight", "<f4"),
])

TARGET_INDEX_DTYPE = np.dtype([
    ("target_offset", "<u8"),
    ("target_count", "<u8"),
])

IMAGE_INDEX_DTYPE = np.dtype([
    ("byte_offset", "<u8"),
    ("byte_length", "<u8"),
    ("shape0", "<u4"),
    ("shape1", "<u4"),
    ("shape2", "<u4"),
    ("shape3", "<u4"),
    ("ndim", "u1"),
    ("dtype_code", "u1"),
    ("codec", "u1"),
    ("flags", "u1"),
])


@dataclass(frozen=True)
class TargetEntry:
    source_position: int
    atom_type: AtomType
    target_value: int
    weight: float = 1.0
    valid: bool = True


@dataclass(frozen=True)
class RecordView:
    record_index: int
    shard_index: int
    shard_record_index: int
    atom_values: np.ndarray
    atom_types: np.ndarray
    loss_mask: np.ndarray
    memory_update_mask: np.ndarray
    reset_mask: np.ndarray
    targets: np.ndarray
    flags: int


@dataclass(frozen=True)
class ChunkView:
    record_index: int
    shard_index: int
    shard_record_index: int
    atom_offset: int
    atom_values: np.ndarray
    atom_types: np.ndarray
    loss_mask: np.ndarray
    memory_update_mask: np.ndarray
    reset_mask: np.ndarray
    targets: np.ndarray
    end_of_record: bool

    @property
    def length(self) -> int:
        return int(self.atom_values.shape[0])


@dataclass(frozen=True)
class UnifiedBatch:
    atom_values: np.ndarray
    atom_types: np.ndarray
    image_payloads: Sequence[Mapping[int, np.ndarray]]
    positions: np.ndarray
    padding_mask: np.ndarray
    valid_atom_mask: np.ndarray
    attention_mask: np.ndarray
    loss_mask: np.ndarray
    memory_update_mask: np.ndarray
    reset_mask: np.ndarray
    lengths: np.ndarray
    targets: Sequence[np.ndarray]
    target_image_payloads: Sequence[Mapping[int, np.ndarray]]
    cursors: Sequence[Mapping[str, int]]


def normalize_atom_arrays(atom_values: Any, atom_types: Any) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(atom_values, dtype="<u8")
    types = np.asarray(atom_types, dtype="u1")
    if values.ndim != 1 or types.ndim != 1:
        raise ValueError("atom_values and atom_types must be one-dimensional")
    if values.shape != types.shape:
        raise ValueError(f"atom value/type shape mismatch: {values.shape} != {types.shape}")
    valid_types = np.isin(types, [int(AtomType.LANGUAGE_TOKEN), int(AtomType.IMAGE)])
    if not bool(np.all(valid_types)):
        invalid = np.unique(types[~valid_types]).tolist()
        raise ValueError(f"unsupported persistent atom types: {invalid}")
    return np.ascontiguousarray(values), np.ascontiguousarray(types)


def normalize_binary_mask(mask: Any, length: int, name: str, default: int = 0) -> np.ndarray:
    if mask is None:
        return np.full(length, default, dtype="u1")
    result = np.asarray(mask, dtype="u1")
    if result.ndim != 1 or result.shape[0] != length:
        raise ValueError(f"{name} must have shape ({length},), got {result.shape}")
    if not bool(np.all((result == 0) | (result == 1))):
        raise ValueError(f"{name} must contain only 0 or 1")
    return np.ascontiguousarray(result)


def targets_to_array(targets: Sequence[TargetEntry] | np.ndarray | None) -> np.ndarray:
    if targets is None:
        return np.empty(0, dtype=TARGET_DTYPE)
    if isinstance(targets, np.ndarray):
        result = np.ascontiguousarray(targets, dtype=TARGET_DTYPE)
    else:
        result = np.empty(len(targets), dtype=TARGET_DTYPE)
        for index, target in enumerate(targets):
            result[index] = (
                int(target.source_position),
                int(target.atom_type),
                int(bool(target.valid)),
                0,
                int(target.target_value),
                float(target.weight),
            )
    valid_types = np.isin(result["atom_type"], [
        int(AtomType.LANGUAGE_TOKEN), int(AtomType.IMAGE)
    ])
    if not bool(np.all(valid_types)):
        raise ValueError("targets contain an unsupported atom type")
    if not bool(np.all(np.isfinite(result["weight"]))) or bool(np.any(result["weight"] < 0)):
        raise ValueError("target weights must be finite and non-negative")
    return result


def combine_field_checksums(checksums: Sequence[int]) -> int:
    result = 1469598103934665603
    for checksum in checksums:
        result ^= int(checksum) & 0xFFFFFFFF
        result = (result * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return result


def record_checksum(atom_values: np.ndarray, atom_types: np.ndarray,
                    loss_mask: np.ndarray, memory_update_mask: np.ndarray,
                    reset_mask: np.ndarray,
                    targets: np.ndarray) -> int:
    checksums = []
    for array in (atom_values, atom_types, loss_mask, memory_update_mask, reset_mask, targets):
        checksums.append(zlib.crc32(memoryview(np.ascontiguousarray(array)).cast("B")))
    return combine_field_checksums(checksums)


def image_shape_from_index(entry: np.void) -> tuple[int, ...]:
    ndim = int(entry["ndim"])
    if ndim < 1 or ndim > 4:
        raise ValueError(f"invalid image ndim: {ndim}")
    return tuple(int(entry[f"shape{axis}"]) for axis in range(ndim))
