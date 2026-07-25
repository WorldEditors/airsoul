"""Torch boundary types for NumPy-backed unified sequence batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .schema import UnifiedBatch


@dataclass(frozen=True)
class TorchTargetBatch:
    lane_indices: torch.Tensor
    source_positions: torch.Tensor
    atom_types: torch.Tensor
    target_values: torch.Tensor
    weights: torch.Tensor
    valid: torch.Tensor
    image_payloads: Mapping[int, torch.Tensor]


@dataclass(frozen=True)
class TorchUnifiedBatch:
    atom_values: torch.Tensor
    atom_types: torch.Tensor
    positions: torch.Tensor
    padding_mask: torch.Tensor
    valid_atom_mask: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    memory_update_mask: torch.Tensor
    reset_mask: torch.Tensor
    lengths: torch.Tensor
    targets: TorchTargetBatch
    image_payloads: Sequence[Mapping[int, torch.Tensor]]
    cursors: Sequence[Mapping[str, int]]

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "TorchUnifiedBatch":
        def move(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(device=device, non_blocking=non_blocking)

        images = tuple({position: move(image) for position, image in lane.items()}
                       for lane in self.image_payloads)
        return TorchUnifiedBatch(
            atom_values=move(self.atom_values),
            atom_types=move(self.atom_types),
            positions=move(self.positions),
            padding_mask=move(self.padding_mask),
            valid_atom_mask=move(self.valid_atom_mask),
            attention_mask=move(self.attention_mask),
            loss_mask=move(self.loss_mask),
            memory_update_mask=move(self.memory_update_mask),
            reset_mask=move(self.reset_mask),
            lengths=move(self.lengths),
            targets=TorchTargetBatch(**{
                "lane_indices": move(self.targets.lane_indices),
                "source_positions": move(self.targets.source_positions),
                "atom_types": move(self.targets.atom_types),
                "target_values": move(self.targets.target_values),
                "weights": move(self.targets.weights),
                "valid": move(self.targets.valid),
                "image_payloads": {
                    index: move(image) for index, image in self.targets.image_payloads.items()
                },
            }),
            image_payloads=images,
            cursors=self.cursors,
        )


def _flatten_targets(targets: Sequence[np.ndarray],
                     target_images: Sequence[Mapping[int, np.ndarray]]) -> TorchTargetBatch:
    lane_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    flat_images: dict[int, torch.Tensor] = {}
    target_offset = 0
    for lane, (entries, lane_images) in enumerate(zip(targets, target_images)):
        if len(entries):
            lane_parts.append(np.full(len(entries), lane, dtype=np.int64))
            target_parts.append(entries)
            for local_index, image in lane_images.items():
                flat_images[target_offset + int(local_index)] = torch.from_numpy(
                    np.asarray(image).copy()
                )
            target_offset += len(entries)
    if not target_parts:
        empty_long = torch.empty(0, dtype=torch.long)
        return TorchTargetBatch(
            lane_indices=empty_long,
            source_positions=empty_long.clone(),
            atom_types=empty_long.clone(),
            target_values=empty_long.clone(),
            weights=torch.empty(0, dtype=torch.float32),
            valid=torch.empty(0, dtype=torch.bool),
            image_payloads={},
        )
    entries = np.concatenate(target_parts)
    return TorchTargetBatch(
        lane_indices=torch.from_numpy(np.concatenate(lane_parts)),
        source_positions=torch.from_numpy(entries["source_position"].astype(np.int64)),
        atom_types=torch.from_numpy(entries["atom_type"].astype(np.int64)),
        target_values=torch.from_numpy(entries["target_value"].astype(np.int64)),
        weights=torch.from_numpy(entries["weight"].astype(np.float32)),
        valid=torch.from_numpy(entries["valid"].astype(np.bool_)),
        image_payloads=flat_images,
    )


def to_torch_batch(batch: UnifiedBatch, *, pin_memory: bool = False) -> TorchUnifiedBatch:
    """Convert the small collated chunk, never the underlying mmap dataset."""
    def tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        value = torch.from_numpy(np.asarray(array).astype({
            torch.long: np.int64,
            torch.bool: np.bool_,
        }[dtype], copy=False))
        return value.pin_memory() if pin_memory else value

    images = tuple({
        position: torch.from_numpy(np.asarray(image).copy())
        for position, image in lane.items()
    } for lane in batch.image_payloads)
    targets = _flatten_targets(batch.targets, batch.target_image_payloads)
    result = TorchUnifiedBatch(
        atom_values=tensor(batch.atom_values, torch.long),
        atom_types=tensor(batch.atom_types, torch.long),
        positions=tensor(batch.positions, torch.long),
        padding_mask=tensor(batch.padding_mask, torch.bool),
        valid_atom_mask=tensor(batch.valid_atom_mask, torch.bool),
        attention_mask=tensor(batch.attention_mask, torch.bool),
        loss_mask=tensor(batch.loss_mask, torch.bool),
        memory_update_mask=tensor(batch.memory_update_mask, torch.bool),
        reset_mask=tensor(batch.reset_mask, torch.bool),
        lengths=tensor(batch.lengths, torch.long),
        targets=targets,
        image_payloads=images,
        cursors=batch.cursors,
    )
    if pin_memory:
        targets = TorchTargetBatch(
            lane_indices=result.targets.lane_indices.pin_memory(),
            source_positions=result.targets.source_positions.pin_memory(),
            atom_types=result.targets.atom_types.pin_memory(),
            target_values=result.targets.target_values.pin_memory(),
            weights=result.targets.weights.pin_memory(),
            valid=result.targets.valid.pin_memory(),
            image_payloads={
                index: image.pin_memory()
                for index, image in result.targets.image_payloads.items()
            },
        )
        images = tuple({position: image.pin_memory() for position, image in lane.items()}
                       for lane in result.image_payloads)
        result = TorchUnifiedBatch(**{**vars(result), "targets": targets, "image_payloads": images})
    return result
