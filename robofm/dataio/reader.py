"""Memory-mapped reader for RoboFM unified sequence datasets."""

from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
import os
from pathlib import Path
from dataclasses import replace
from typing import Iterator

import numpy as np

from .schema import (
    AtomType,
    ChunkView,
    FORMAT_NAME,
    FORMAT_VERSION,
    IMAGE_DTYPE_TO_NUMPY,
    IMAGE_INDEX_DTYPE,
    RECORD_DTYPE,
    TARGET_DTYPE,
    TARGET_INDEX_DTYPE,
    ImageCodec,
    RecordView,
    image_shape_from_index,
)


def _mmap_or_empty(path: Path, dtype: np.dtype) -> np.ndarray:
    size = path.stat().st_size
    if size == 0:
        return np.empty(0, dtype=dtype)
    if size % dtype.itemsize:
        raise ValueError(f"file size is not aligned to dtype {dtype}: {path}")
    return np.memmap(path, dtype=dtype, mode="r")


class _MMapShard:
    def __init__(self, root: Path, shard_index: int, expected: dict):
        if not (root / "COMMITTED").is_file():
            raise ValueError(f"uncommitted shard: {root}")
        self.root = root
        self.shard_index = shard_index
        self.tokens = _mmap_or_empty(root / "tokens.bin", np.dtype("<u8"))
        self.atom_types = _mmap_or_empty(root / "atom_types.bin", np.dtype("u1"))
        self.records = _mmap_or_empty(root / "records.idx", RECORD_DTYPE)
        self.targets = _mmap_or_empty(root / "targets.bin", TARGET_DTYPE)
        self.target_index = _mmap_or_empty(root / "targets.idx", TARGET_INDEX_DTYPE)
        self.loss_mask = _mmap_or_empty(root / "loss_mask.bin", np.dtype("u1"))
        self.memory_update_mask = _mmap_or_empty(
            root / "memory_update_mask.bin", np.dtype("u1")
        )
        self.reset_mask = _mmap_or_empty(root / "reset_mask.bin", np.dtype("u1"))
        self.image_index = _mmap_or_empty(root / "images.idx", IMAGE_INDEX_DTYPE)
        self.image_bytes = _mmap_or_empty(root / "images.bin", np.dtype("u1"))
        actual = {
            "records": len(self.records),
            "atoms": len(self.tokens),
            "targets": len(self.targets),
            "images": len(self.image_index),
            "image_bytes": len(self.image_bytes),
        }
        for key, value in actual.items():
            if int(expected[key]) != int(value):
                raise ValueError(f"manifest mismatch for {root.name}.{key}: {expected[key]} != {value}")
        if len(self.atom_types) != len(self.tokens):
            raise ValueError(f"token/type length mismatch in {root}")
        if (len(self.loss_mask) != len(self.tokens) or
                len(self.memory_update_mask) != len(self.tokens) or
                len(self.reset_mask) != len(self.tokens)):
            raise ValueError(f"mask/token length mismatch in {root}")
        if len(self.target_index) != len(self.records):
            raise ValueError(f"target index/record length mismatch in {root}")

    def record(self, local_index: int, global_index: int) -> RecordView:
        entry = self.records[local_index]
        atom_start = int(entry["atom_offset"])
        atom_end = atom_start + int(entry["atom_length"])
        target_start = int(entry["target_offset"])
        target_end = target_start + int(entry["target_count"])
        mask_start = int(entry["loss_mask_offset"])
        mask_end = mask_start + int(entry["atom_length"])
        return RecordView(
            record_index=global_index,
            shard_index=self.shard_index,
            shard_record_index=local_index,
            atom_values=self.tokens[atom_start:atom_end],
            atom_types=self.atom_types[atom_start:atom_end],
            loss_mask=self.loss_mask[mask_start:mask_end],
            memory_update_mask=self.memory_update_mask[atom_start:atom_end],
            reset_mask=self.reset_mask[atom_start:atom_end],
            targets=self.targets[target_start:target_end],
            flags=int(entry["flags"]),
        )

    def image(self, image_index: int) -> np.ndarray:
        entry = self.image_index[image_index]
        codec = ImageCodec(int(entry["codec"]))
        if codec is not ImageCodec.RAW:
            raise NotImplementedError(f"image codec is not supported by the V1 reader: {codec}")
        dtype_code = int(entry["dtype_code"])
        try:
            dtype = IMAGE_DTYPE_TO_NUMPY[dtype_code]
        except KeyError as error:
            raise ValueError(f"unknown image dtype code: {dtype_code}") from error
        shape = image_shape_from_index(entry)
        offset = int(entry["byte_offset"])
        length = int(entry["byte_length"])
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if length != expected:
            raise ValueError(f"raw image byte length mismatch: {length} != {expected}")
        if offset + length > len(self.image_bytes):
            raise ValueError("image payload extends beyond images.bin")
        return np.ndarray(shape=shape, dtype=dtype, buffer=self.image_bytes, offset=offset)

    def close(self) -> None:
        for value in vars(self).values():
            if isinstance(value, np.memmap) and getattr(value, "_mmap", None) is not None:
                value._mmap.close()


class UnifiedMMapDataset:
    """Immutable, random-seekable view over one unified dataset."""

    def __init__(self, root: str | os.PathLike[str], *, validate_manifest: bool = True):
        self.root = Path(root).resolve()
        if not (self.root / "COMMITTED").is_file():
            raise ValueError(f"dataset is not committed: {self.root}")
        with open(self.root / "manifest.json", "r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        canonical_manifest = json.dumps(
            self.manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        self.manifest_hash = hashlib.sha256(canonical_manifest).hexdigest()
        if validate_manifest:
            if self.manifest.get("format") != FORMAT_NAME:
                raise ValueError(f"unsupported dataset format: {self.manifest.get('format')}")
            if int(self.manifest.get("version", -1)) != FORMAT_VERSION:
                raise ValueError(f"unsupported dataset version: {self.manifest.get('version')}")
            if self.manifest.get("endianness") != "little":
                raise ValueError("only little-endian unified datasets are supported")
            required_streams = {
                "tokens.bin", "atom_types.bin", "loss_mask.bin",
                "memory_update_mask.bin", "reset_mask.bin", "records.idx",
                "targets.bin", "targets.idx",
            }
            if not required_streams.issubset(self.manifest.get("streams", {})):
                raise ValueError("manifest does not declare every required V1 stream")
        self.dataset_uuid = str(self.manifest["dataset_uuid"])
        committed_uuid = (self.root / "COMMITTED").read_text(encoding="ascii").strip()
        if committed_uuid != self.dataset_uuid:
            raise ValueError("dataset COMMITTED marker does not match manifest UUID")
        self.shards: list[_MMapShard] = []
        self._record_ends: list[int] = []
        record_total = 0
        for shard_index, shard_info in enumerate(self.manifest["shards"]):
            shard = _MMapShard(self.root / "shards" / shard_info["name"], shard_index, shard_info)
            self.shards.append(shard)
            record_total += len(shard.records)
            self._record_ends.append(record_total)
        if record_total != int(self.manifest["totals"]["records"]):
            raise ValueError("dataset record total does not match manifest")

    def __len__(self) -> int:
        return self._record_ends[-1] if self._record_ends else 0

    def _locate_record(self, record_index: int) -> tuple[_MMapShard, int]:
        if record_index < 0:
            record_index += len(self)
        if record_index < 0 or record_index >= len(self):
            raise IndexError(record_index)
        shard_index = bisect_right(self._record_ends, record_index)
        previous_end = self._record_ends[shard_index - 1] if shard_index else 0
        return self.shards[shard_index], record_index - previous_end

    def __getitem__(self, record_index: int) -> RecordView:
        normalized_index = record_index if record_index >= 0 else len(self) + record_index
        shard, local_index = self._locate_record(record_index)
        return shard.record(local_index, normalized_index)

    def record_length(self, record_index: int) -> int:
        shard, local_index = self._locate_record(record_index)
        return int(shard.records[local_index]["atom_length"])

    def read_chunk(self, record_index: int, atom_offset: int, chunk_length: int) -> ChunkView:
        if atom_offset < 0 or chunk_length < 1:
            raise ValueError("atom_offset must be non-negative and chunk_length must be positive")
        record = self[record_index]
        if atom_offset > len(record.atom_values):
            raise ValueError("atom_offset is beyond the end of the record")
        atom_end = min(atom_offset + chunk_length, len(record.atom_values))
        targets = record.targets
        if len(targets):
            selected = (
                (targets["source_position"] >= atom_offset) &
                (targets["source_position"] < atom_end)
            )
            chunk_targets = np.array(targets[selected], copy=True)
            chunk_targets["source_position"] -= atom_offset
        else:
            chunk_targets = np.empty(0, dtype=TARGET_DTYPE)
        return ChunkView(
            record_index=record.record_index,
            shard_index=record.shard_index,
            shard_record_index=record.shard_record_index,
            atom_offset=atom_offset,
            atom_values=record.atom_values[atom_offset:atom_end],
            atom_types=record.atom_types[atom_offset:atom_end],
            loss_mask=record.loss_mask[atom_offset:atom_end],
            memory_update_mask=record.memory_update_mask[atom_offset:atom_end],
            reset_mask=record.reset_mask[atom_offset:atom_end],
            targets=chunk_targets,
            end_of_record=atom_end == len(record.atom_values),
        )

    def get_image(self, shard_index: int, image_index: int) -> np.ndarray:
        return self.shards[shard_index].image(image_index)

    def iter_record_chunks(self, record_index: int, chunk_length: int) -> Iterator[ChunkView]:
        offset = 0
        length = self.record_length(record_index)
        while offset < length:
            chunk = self.read_chunk(record_index, offset, chunk_length)
            yield chunk
            offset += chunk.length

    def close(self) -> None:
        for shard in self.shards:
            shard.close()

    def __enter__(self) -> "UnifiedMMapDataset":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class UnifiedDatasetCollection:
    """Read a directory containing independently committed V1 datasets."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).resolve()
        children = sorted(
            path for path in self.root.iterdir()
            if path.is_dir() and (path / "COMMITTED").is_file()
        )
        if not children:
            raise ValueError(f"no committed V1 datasets found under: {self.root}")
        self.datasets = [UnifiedMMapDataset(path) for path in children]
        self._record_ends: list[int] = []
        record_total = 0
        for dataset in self.datasets:
            record_total += len(dataset)
            self._record_ends.append(record_total)
        identity = "\n".join(dataset.dataset_uuid for dataset in self.datasets).encode("ascii")
        manifests = "\n".join(dataset.manifest_hash for dataset in self.datasets).encode("ascii")
        self.dataset_uuid = hashlib.sha256(identity).hexdigest()
        self.manifest_hash = hashlib.sha256(manifests).hexdigest()
        self.manifest = {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "collection": True,
            "members": [str(path.relative_to(self.root)) for path in children],
            "totals": {"records": record_total},
        }

    def __len__(self) -> int:
        return self._record_ends[-1]

    def _locate_record(self, record_index: int) -> tuple[int, UnifiedMMapDataset, int, int]:
        normalized = record_index if record_index >= 0 else len(self) + record_index
        if normalized < 0 or normalized >= len(self):
            raise IndexError(record_index)
        dataset_index = bisect_right(self._record_ends, normalized)
        previous_end = self._record_ends[dataset_index - 1] if dataset_index else 0
        return dataset_index, self.datasets[dataset_index], normalized - previous_end, normalized

    @staticmethod
    def _global_shard(dataset_index: int, shard_index: int) -> int:
        return (dataset_index << 32) | shard_index

    def __getitem__(self, record_index: int) -> RecordView:
        dataset_index, dataset, local_index, normalized = self._locate_record(record_index)
        record = dataset[local_index]
        return replace(
            record,
            record_index=normalized,
            shard_index=self._global_shard(dataset_index, record.shard_index),
        )

    def record_length(self, record_index: int) -> int:
        _, dataset, local_index, _ = self._locate_record(record_index)
        return dataset.record_length(local_index)

    def read_chunk(self, record_index: int, atom_offset: int, chunk_length: int) -> ChunkView:
        dataset_index, dataset, local_index, normalized = self._locate_record(record_index)
        chunk = dataset.read_chunk(local_index, atom_offset, chunk_length)
        return replace(
            chunk,
            record_index=normalized,
            shard_index=self._global_shard(dataset_index, chunk.shard_index),
        )

    def get_image(self, shard_index: int, image_index: int) -> np.ndarray:
        dataset_index = shard_index >> 32
        local_shard = shard_index & 0xFFFFFFFF
        return self.datasets[dataset_index].get_image(local_shard, image_index)

    def iter_record_chunks(self, record_index: int, chunk_length: int) -> Iterator[ChunkView]:
        offset = 0
        length = self.record_length(record_index)
        while offset < length:
            chunk = self.read_chunk(record_index, offset, chunk_length)
            yield chunk
            offset += chunk.length

    def close(self) -> None:
        for dataset in self.datasets:
            dataset.close()

    def __enter__(self) -> "UnifiedDatasetCollection":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def open_unified_dataset(root: str | os.PathLike[str]) -> UnifiedMMapDataset | UnifiedDatasetCollection:
    path = Path(root).resolve()
    if (path / "COMMITTED").is_file():
        return UnifiedMMapDataset(path)
    return UnifiedDatasetCollection(path)
