"""Atomic writer for RoboFM unified sequence datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid
import zlib

import numpy as np

from .schema import (
    AtomType,
    ENDIANNESS,
    FORMAT_NAME,
    FORMAT_VERSION,
    IMAGE_INDEX_DTYPE,
    NUMPY_TO_IMAGE_DTYPE,
    RECORD_DTYPE,
    TARGET_DTYPE,
    TARGET_INDEX_DTYPE,
    ImageCodec,
    TargetEntry,
    combine_field_checksums,
    normalize_atom_arrays,
    normalize_binary_mask,
    record_checksum,
    targets_to_array,
)


_WRITE_BLOCK_BYTES = 64 * 1024 * 1024


def _write_buffer(handle, payload: memoryview) -> None:
    payload = payload.cast("B")
    for offset in range(0, len(payload), _WRITE_BLOCK_BYTES):
        block = payload[offset:offset + _WRITE_BLOCK_BYTES]
        written = handle.write(block)
        if written != len(block):
            raise OSError(f"short write: {written} != {len(block)}")


def _write_array(handle, array: np.ndarray) -> None:
    _write_buffer(handle, memoryview(np.ascontiguousarray(array)))


@dataclass
class _ShardStats:
    name: str
    records: int
    atoms: int
    targets: int
    images: int
    image_bytes: int


class _ShardWriter:
    def __init__(self, path: Path, shard_index: int):
        self.path = path
        self.shard_index = shard_index
        self.path.mkdir(parents=True, exist_ok=False)
        self._files = {
            "tokens": open(path / "tokens.bin", "wb"),
            "types": open(path / "atom_types.bin", "wb"),
            "records": open(path / "records.idx", "wb"),
            "targets": open(path / "targets.bin", "wb"),
            "target_index": open(path / "targets.idx", "wb"),
            "loss_mask": open(path / "loss_mask.bin", "wb"),
            "memory_update_mask": open(path / "memory_update_mask.bin", "wb"),
            "reset_mask": open(path / "reset_mask.bin", "wb"),
            "images": open(path / "images.bin", "wb"),
            "image_index": open(path / "images.idx", "wb"),
        }
        self.record_count = 0
        self.atom_count = 0
        self.target_count = 0
        self.image_count = 0
        self.image_bytes = 0
        self.closed = False
        self._stream: dict[str, Any] | None = None

    def _append_images(self, images: Sequence[np.ndarray]) -> int:
        image_base = self.image_count
        for image in images:
            array = np.asarray(image)
            if array.ndim < 1 or array.ndim > 4:
                raise ValueError(f"images must have 1-4 dimensions, got {array.shape}")
            dtype = NUMPY_TO_IMAGE_DTYPE.get(array.dtype)
            if dtype is None:
                raise ValueError(f"unsupported raw image dtype: {array.dtype}")
            array = np.ascontiguousarray(array)
            payload = memoryview(array).cast("B")
            shape = list(array.shape) + [0] * (4 - array.ndim)
            index_entry = np.array([(
                self.image_bytes,
                len(payload),
                shape[0], shape[1], shape[2], shape[3],
                array.ndim,
                int(dtype),
                int(ImageCodec.RAW),
                0,
            )], dtype=IMAGE_INDEX_DTYPE)
            _write_buffer(self._files["images"], payload)
            _write_array(self._files["image_index"], index_entry)
            self.image_count += 1
            self.image_bytes += len(payload)
        return image_base

    def begin_record(self, *, flags: int = 0, expected_atoms: int | None = None) -> None:
        if self.closed or self._stream is not None:
            raise RuntimeError("cannot begin a record in the current shard state")
        if expected_atoms is not None and expected_atoms < 1:
            raise ValueError("expected_atoms must be positive")
        self._stream = {
            "atom_start": self.atom_count,
            "target_start": self.target_count,
            "atom_length": 0,
            "target_count": 0,
            "flags": int(flags),
            "expected_atoms": expected_atoms,
            "checksums": [0] * 6,
        }

    def append_record_chunk(self, atom_values: Any, atom_types: Any, *,
                            images: Sequence[np.ndarray] = (), loss_mask: Any = None,
                            memory_update_mask: Any = None, reset_mask: Any = None,
                            targets: Sequence[TargetEntry] | np.ndarray | None = None) -> None:
        if self._stream is None:
            raise RuntimeError("begin_record must be called before appending chunks")
        values, types = normalize_atom_arrays(atom_values, atom_types)
        if len(values) == 0:
            return
        loss = normalize_binary_mask(loss_mask, len(values), "loss_mask", default=1)
        memory_update = normalize_binary_mask(
            memory_update_mask, len(values), "memory_update_mask", default=1
        )
        reset = normalize_binary_mask(reset_mask, len(values), "reset_mask", default=0)
        target_array = targets_to_array(targets)
        if len(target_array) and int(target_array["source_position"].max()) >= len(values):
            raise ValueError("chunk target source_position is outside the chunk")

        image_base = self._append_images(images)
        image_positions = types == int(AtomType.IMAGE)
        if bool(np.any(image_positions)):
            local_refs = values[image_positions]
            if not images or int(local_refs.max()) >= len(images):
                raise ValueError("IMAGE atom references an unavailable chunk-local image")
            values = values.copy()
            values[image_positions] += image_base
        image_targets = target_array["atom_type"] == int(AtomType.IMAGE)
        if bool(np.any(image_targets)):
            local_refs = target_array["target_value"][image_targets]
            if not images or int(local_refs.max()) >= len(images):
                raise ValueError("IMAGE target references an unavailable chunk-local image")
            target_array = target_array.copy()
            target_array["target_value"][image_targets] += image_base
        if len(target_array):
            target_array = target_array.copy()
            target_array["source_position"] += self._stream["atom_length"]

        arrays = (values, types, loss, memory_update, reset, target_array)
        file_names = ("tokens", "types", "loss_mask", "memory_update_mask", "reset_mask", "targets")
        for index, (array, file_name) in enumerate(zip(arrays, file_names)):
            payload = memoryview(np.ascontiguousarray(array)).cast("B")
            self._stream["checksums"][index] = zlib.crc32(
                payload, self._stream["checksums"][index]
            )
            _write_buffer(self._files[file_name], payload)
        self._stream["atom_length"] += len(values)
        self._stream["target_count"] += len(target_array)
        self.atom_count += len(values)
        self.target_count += len(target_array)

    def end_record(self) -> int:
        if self._stream is None:
            raise RuntimeError("no streaming record is active")
        stream = self._stream
        if stream["atom_length"] < 1:
            raise ValueError("records must contain at least one atom")
        if (stream["expected_atoms"] is not None and
                stream["expected_atoms"] != stream["atom_length"]):
            raise ValueError(
                f"streamed atom count {stream['atom_length']} does not match "
                f"expected_atoms={stream['expected_atoms']}"
            )
        checksum = combine_field_checksums(stream["checksums"])
        record_entry = np.array([(
            stream["atom_start"], stream["atom_length"], stream["target_start"],
            stream["target_count"], stream["atom_start"], stream["flags"], 0, checksum,
        )], dtype=RECORD_DTYPE)
        target_index = np.array([(
            stream["target_start"], stream["target_count"]
        )], dtype=TARGET_INDEX_DTYPE)
        _write_array(self._files["records"], record_entry)
        _write_array(self._files["target_index"], target_index)
        record_id = self.record_count
        self.record_count += 1
        self._stream = None
        return record_id

    def append_record(self, atom_values: Any, atom_types: Any,
                      images: Sequence[np.ndarray] = (),
                      loss_mask: Any = None, memory_update_mask: Any = None,
                      reset_mask: Any = None,
                      targets: Sequence[TargetEntry] | np.ndarray | None = None,
                      flags: int = 0) -> int:
        if self.closed:
            raise RuntimeError("cannot append to a closed shard")
        values, types = normalize_atom_arrays(atom_values, atom_types)
        if len(values) == 0:
            raise ValueError("records must contain at least one atom")
        loss = normalize_binary_mask(loss_mask, len(values), "loss_mask", default=1)
        memory_update = normalize_binary_mask(
            memory_update_mask, len(values), "memory_update_mask", default=1
        )
        reset = normalize_binary_mask(reset_mask, len(values), "reset_mask", default=0)
        target_array = targets_to_array(targets)
        if len(target_array) and int(target_array["source_position"].max()) >= len(values):
            raise ValueError("target source_position is outside the record")

        image_base = self._append_images(images)
        image_positions = types == int(AtomType.IMAGE)
        if bool(np.any(image_positions)):
            local_refs = values[image_positions]
            if not images or int(local_refs.max()) >= len(images):
                raise ValueError("IMAGE atom references an unavailable record-local image")
            values = values.copy()
            values[image_positions] += image_base

        image_targets = target_array["atom_type"] == int(AtomType.IMAGE)
        if bool(np.any(image_targets)):
            local_refs = target_array["target_value"][image_targets]
            if not images or int(local_refs.max()) >= len(images):
                raise ValueError("IMAGE target references an unavailable record-local image")
            target_array = target_array.copy()
            target_array["target_value"][image_targets] += image_base

        checksum = record_checksum(values, types, loss, memory_update, reset, target_array)
        record_entry = np.array([(
            self.atom_count,
            len(values),
            self.target_count,
            len(target_array),
            self.atom_count,
            int(flags),
            0,
            checksum,
        )], dtype=RECORD_DTYPE)
        target_index = np.array([(self.target_count, len(target_array))], dtype=TARGET_INDEX_DTYPE)

        _write_array(self._files["tokens"], values)
        _write_array(self._files["types"], types)
        _write_array(self._files["loss_mask"], loss)
        _write_array(self._files["memory_update_mask"], memory_update)
        _write_array(self._files["reset_mask"], reset)
        _write_array(self._files["targets"], target_array)
        _write_array(self._files["target_index"], target_index)
        _write_array(self._files["records"], record_entry)

        record_id = self.record_count
        self.record_count += 1
        self.atom_count += len(values)
        self.target_count += len(target_array)
        return record_id

    def close(self) -> _ShardStats:
        if self.closed:
            raise RuntimeError("shard is already closed")
        if self._stream is not None:
            raise RuntimeError("cannot close a shard with an unfinished streaming record")
        for handle in self._files.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        (self.path / "COMMITTED").write_text("committed\n", encoding="ascii")
        self.closed = True
        return _ShardStats(
            name=self.path.name,
            records=self.record_count,
            atoms=self.atom_count,
            targets=self.target_count,
            images=self.image_count,
            image_bytes=self.image_bytes,
        )

    def abort(self) -> None:
        for handle in self._files.values():
            if not handle.closed:
                handle.close()
        self.closed = True


class UnifiedDatasetWriter:
    """Build an immutable dataset and publish it with an atomic rename."""

    def __init__(self, output_path: str | os.PathLike[str], *,
                 tokenizer: Mapping[str, Any], special_tokens: Mapping[str, int],
                 producer: Mapping[str, Any] | None = None,
                 max_atoms_per_shard: int = 10_000_000):
        self.output_path = Path(output_path).resolve()
        if self.output_path.exists():
            raise FileExistsError(f"dataset output already exists: {self.output_path}")
        if max_atoms_per_shard < 1:
            raise ValueError("max_atoms_per_shard must be positive")
        self.dataset_uuid = str(uuid.uuid4())
        self.temp_path = self.output_path.with_name(f".{self.output_path.name}.tmp-{self.dataset_uuid}")
        self.temp_path.mkdir(parents=True, exist_ok=False)
        (self.temp_path / "shards").mkdir()
        self.tokenizer = dict(tokenizer)
        self.special_tokens = {str(key): int(value) for key, value in special_tokens.items()}
        self.producer = dict(producer or {})
        self.max_atoms_per_shard = int(max_atoms_per_shard)
        self.shard_stats: list[_ShardStats] = []
        self.current: _ShardWriter | None = None
        self.closed = False
        self._active_stream = False

    def _new_shard(self) -> _ShardWriter:
        shard_index = len(self.shard_stats)
        path = self.temp_path / "shards" / f"shard-{shard_index:06d}"
        self.current = _ShardWriter(path, shard_index)
        return self.current

    def append_record(self, atom_values: Any, atom_types: Any, **kwargs: Any) -> tuple[int, int]:
        if self._active_stream:
            raise RuntimeError("cannot append while a streaming record is active")
        values = np.asarray(atom_values)
        if values.ndim != 1:
            raise ValueError("atom_values must be one-dimensional")
        if self.current is None:
            self._new_shard()
        elif self.current.record_count and self.current.atom_count + len(values) > self.max_atoms_per_shard:
            self.shard_stats.append(self.current.close())
            self._new_shard()
        assert self.current is not None
        local_record = self.current.append_record(atom_values, atom_types, **kwargs)
        return self.current.shard_index, local_record

    def stream_record(self, *, flags: int = 0,
                      expected_atoms: int | None = None) -> "StreamingRecordWriter":
        if self.closed or self._active_stream:
            raise RuntimeError("writer is closed or already streaming a record")
        if self.current is None:
            self._new_shard()
        elif (self.current.record_count and expected_atoms is not None and
              self.current.atom_count + expected_atoms > self.max_atoms_per_shard):
            self.shard_stats.append(self.current.close())
            self._new_shard()
        assert self.current is not None
        self.current.begin_record(flags=flags, expected_atoms=expected_atoms)
        self._active_stream = True
        return StreamingRecordWriter(self, self.current)

    def finalize(self) -> Path:
        if self.closed:
            raise RuntimeError("writer is already finalized or aborted")
        if self._active_stream:
            raise RuntimeError("cannot finalize with an unfinished streaming record")
        if self.current is not None:
            self.shard_stats.append(self.current.close())
            self.current = None
        if not self.shard_stats:
            raise ValueError("cannot finalize an empty dataset")
        manifest = {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "endianness": ENDIANNESS,
            "dataset_uuid": self.dataset_uuid,
            "commit_generation": self.dataset_uuid,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tokenizer": self.tokenizer,
            "special_tokens": self.special_tokens,
            "producer": self.producer,
            "atom_types": {
                "language_token": int(AtomType.LANGUAGE_TOKEN),
                "image": int(AtomType.IMAGE),
            },
            "streams": {
                "tokens.bin": "uint64-le",
                "atom_types.bin": "uint8",
                "loss_mask.bin": "uint8-binary",
                "memory_update_mask.bin": "uint8-binary",
                "reset_mask.bin": "uint8-binary",
                "records.idx": "record-index-v1-le",
                "targets.bin": "target-v1-le",
                "targets.idx": "target-index-v1-le",
            },
            "images": {
                "payload_file": "images.bin",
                "index_file": "images.idx",
                "codecs": {"raw": int(ImageCodec.RAW)},
                "index_schema": "image-index-v1-le",
            },
            "checksum": {
                "scope": "record-fields",
                "algorithm": "crc32-per-field-fnv1a64-combine",
            },
            "shards": [stats.__dict__ for stats in self.shard_stats],
            "totals": {
                "records": sum(item.records for item in self.shard_stats),
                "atoms": sum(item.atoms for item in self.shard_stats),
                "targets": sum(item.targets for item in self.shard_stats),
                "images": sum(item.images for item in self.shard_stats),
                "image_bytes": sum(item.image_bytes for item in self.shard_stats),
            },
        }
        manifest_tmp = self.temp_path / "manifest.json.tmp"
        manifest_tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(manifest_tmp, self.temp_path / "manifest.json")
        (self.temp_path / "COMMITTED").write_text(self.dataset_uuid + "\n", encoding="ascii")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.temp_path, self.output_path)
        self.closed = True
        return self.output_path

    def abort(self) -> None:
        if self.closed:
            return
        if self.current is not None:
            self.current.abort()
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)
        self.closed = True

    def __enter__(self) -> "UnifiedDatasetWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.finalize()
        else:
            self.abort()


class StreamingRecordWriter:
    """Bounded-memory writer for records containing arbitrarily many atoms."""

    def __init__(self, owner: UnifiedDatasetWriter, shard: _ShardWriter):
        self.owner = owner
        self.shard = shard
        self.closed = False

    def append(self, atom_values: Any, atom_types: Any, **kwargs: Any) -> None:
        if self.closed:
            raise RuntimeError("streaming record is closed")
        self.shard.append_record_chunk(atom_values, atom_types, **kwargs)

    def close(self) -> tuple[int, int]:
        if self.closed:
            raise RuntimeError("streaming record is closed")
        local_record = self.shard.end_record()
        self.owner._active_stream = False
        self.closed = True
        return self.shard.shard_index, local_record

    def __enter__(self) -> "StreamingRecordWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.owner.abort()
            self.closed = True
