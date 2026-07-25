"""Padding and mask collation for lane chunks."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .reader import UnifiedDatasetCollection, UnifiedMMapDataset
from .schema import AtomType, ChunkView, TARGET_DTYPE, UnifiedBatch


def collate_chunks(chunks: Sequence[ChunkView | None], *, pad_token_id: int,
                   pad_to_multiple: int = 1,
                   pad_to_length: int | None = None,
                   dataset: UnifiedMMapDataset | UnifiedDatasetCollection | None = None,
                   load_images: bool = False) -> UnifiedBatch:
    if not chunks:
        raise ValueError("cannot collate an empty lane list")
    if pad_to_multiple < 1:
        raise ValueError("pad_to_multiple must be positive")
    lengths = np.asarray([0 if chunk is None else chunk.length for chunk in chunks], dtype="<u8")
    max_length = int(lengths.max(initial=0))
    if pad_to_length is not None:
        if pad_to_length < max_length:
            raise ValueError("pad_to_length is smaller than an input chunk")
        max_length = int(pad_to_length)
    elif max_length:
        max_length = ((max_length + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    batch_size = len(chunks)
    atom_values = np.full((batch_size, max_length), int(pad_token_id), dtype="<u8")
    atom_types = np.zeros((batch_size, max_length), dtype="u1")
    positions = np.zeros((batch_size, max_length), dtype="<u8")
    valid = np.zeros((batch_size, max_length), dtype=np.bool_)
    loss_mask = np.zeros((batch_size, max_length), dtype=np.bool_)
    reset_mask = np.zeros((batch_size, max_length), dtype=np.bool_)
    memory_update_mask = np.zeros((batch_size, max_length), dtype=np.bool_)
    image_payloads: list[dict[int, np.ndarray]] = []
    target_image_payloads: list[dict[int, np.ndarray]] = []
    targets: list[np.ndarray] = []
    cursors: list[dict[str, int]] = []

    for lane_id, chunk in enumerate(chunks):
        lane_images: dict[int, np.ndarray] = {}
        lane_target_images: dict[int, np.ndarray] = {}
        if chunk is None:
            image_payloads.append(lane_images)
            target_image_payloads.append(lane_target_images)
            targets.append(np.empty(0, dtype=TARGET_DTYPE))
            cursors.append({"lane_id": lane_id, "record_index": -1, "atom_offset": 0})
            continue
        length = chunk.length
        atom_values[lane_id, :length] = chunk.atom_values
        atom_types[lane_id, :length] = chunk.atom_types
        positions[lane_id, :length] = chunk.atom_offset + np.arange(length, dtype="<u8")
        valid[lane_id, :length] = True
        loss_mask[lane_id, :length] = chunk.loss_mask.astype(np.bool_, copy=False)
        reset_mask[lane_id, :length] = chunk.reset_mask.astype(np.bool_, copy=False)
        memory_update_mask[lane_id, :length] = chunk.memory_update_mask.astype(np.bool_, copy=False)
        if length and chunk.atom_offset == 0:
            # Record boundaries always begin a new memory stream, independent
            # of producer-provided special tokens.
            reset_mask[lane_id, 0] = True
        if load_images:
            if dataset is None:
                raise ValueError("dataset is required when load_images=True")
            image_positions = np.flatnonzero(chunk.atom_types == int(AtomType.IMAGE))
            for position in image_positions:
                lane_images[int(position)] = dataset.get_image(
                    chunk.shard_index, int(chunk.atom_values[position])
                )
            for target_index, target in enumerate(chunk.targets):
                if int(target["atom_type"]) == int(AtomType.IMAGE) and bool(target["valid"]):
                    lane_target_images[target_index] = dataset.get_image(
                        chunk.shard_index, int(target["target_value"])
                    )
        image_payloads.append(lane_images)
        target_image_payloads.append(lane_target_images)
        targets.append(chunk.targets)
        cursors.append({
            "lane_id": lane_id,
            "record_index": chunk.record_index,
            "atom_offset": chunk.atom_offset,
        })

    return UnifiedBatch(
        atom_values=atom_values,
        atom_types=atom_types,
        image_payloads=tuple(image_payloads),
        positions=positions,
        padding_mask=~valid,
        valid_atom_mask=valid,
        attention_mask=valid.copy(),
        loss_mask=loss_mask & valid,
        memory_update_mask=memory_update_mask & valid,
        reset_mask=reset_mask & valid,
        lengths=lengths,
        targets=tuple(targets),
        target_image_payloads=tuple(target_image_payloads),
        cursors=tuple(cursors),
    )
