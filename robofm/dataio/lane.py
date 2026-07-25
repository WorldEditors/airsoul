"""Serializable two-phase lane scheduler for continuous chunk streaming."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from .reader import UnifiedDatasetCollection, UnifiedMMapDataset


@dataclass
class LaneCursor:
    lane_id: int
    record_index: int | None
    atom_offset: int = 0
    chunk_index: int = 0
    record_generation: int = 0


@dataclass(frozen=True)
class ChunkRequest:
    lane_id: int
    record_index: int
    atom_offset: int
    chunk_length: int
    chunk_index: int
    record_generation: int


class LaneScheduler:
    """Assign records to stable lanes and advance them only after commit."""

    def __init__(self, dataset: UnifiedMMapDataset | UnifiedDatasetCollection, *, lane_count: int,
                 chunk_length: int, rank: int = 0, world_size: int = 1,
                 seed: int = 0, shuffle: bool = True, repeat: bool = True):
        if lane_count < 1 or chunk_length < 1:
            raise ValueError("lane_count and chunk_length must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"invalid rank/world_size: {rank}/{world_size}")
        self.dataset = dataset
        self.lane_count = int(lane_count)
        self.chunk_length = int(chunk_length)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.repeat = bool(repeat)
        self.generation = 0
        self._rank_records: np.ndarray = np.empty(0, dtype=np.int64)
        self._next_record_index = 0
        self._pending: list[ChunkRequest | None] | None = None
        self._start_generation(0)
        self.lanes = [LaneCursor(lane_id=index, record_index=None) for index in range(self.lane_count)]
        for lane in self.lanes:
            self._refill_lane(lane)

    def _start_generation(self, generation: int) -> None:
        order = np.arange(len(self.dataset), dtype=np.int64)
        if self.shuffle and len(order):
            rng = np.random.Generator(np.random.PCG64(self.seed + generation))
            rng.shuffle(order)
        self.generation = int(generation)
        self._rank_records = order[self.rank::self.world_size]
        self._next_record_index = 0

    def _next_record(self) -> tuple[int, int] | None:
        if self._next_record_index >= len(self._rank_records):
            if not self.repeat or len(self._rank_records) == 0:
                return None
            self._start_generation(self.generation + 1)
        record = int(self._rank_records[self._next_record_index])
        self._next_record_index += 1
        return record, self.generation

    def _refill_lane(self, lane: LaneCursor) -> None:
        next_record = self._next_record()
        if next_record is None:
            lane.record_index = None
            lane.atom_offset = 0
            lane.chunk_index = 0
            return
        lane.record_index, lane.record_generation = next_record
        lane.atom_offset = 0
        lane.chunk_index = 0

    def peek(self) -> tuple[ChunkRequest | None, ...]:
        if self._pending is not None:
            return tuple(self._pending)
        pending: list[ChunkRequest | None] = []
        for lane in self.lanes:
            if lane.record_index is None:
                pending.append(None)
                continue
            remaining = self.dataset.record_length(lane.record_index) - lane.atom_offset
            length = min(self.chunk_length, remaining)
            pending.append(ChunkRequest(
                lane_id=lane.lane_id,
                record_index=lane.record_index,
                atom_offset=lane.atom_offset,
                chunk_length=length,
                chunk_index=lane.chunk_index,
                record_generation=lane.record_generation,
            ))
        self._pending = pending
        return tuple(pending)

    def read(self):
        return tuple(
            None if request is None else self.dataset.read_chunk(
                request.record_index, request.atom_offset, request.chunk_length
            )
            for request in self.peek()
        )

    def commit(self, consumed_lengths: Sequence[int] | None = None) -> None:
        pending = self.peek()
        if consumed_lengths is None:
            consumed_lengths = [0 if request is None else request.chunk_length for request in pending]
        if len(consumed_lengths) != self.lane_count:
            raise ValueError("consumed_lengths must have one entry per lane")
        for lane, request, consumed in zip(self.lanes, pending, consumed_lengths):
            consumed = int(consumed)
            if request is None:
                if consumed != 0:
                    raise ValueError("cannot consume atoms from an inactive lane")
                continue
            if consumed < 0 or consumed > request.chunk_length:
                raise ValueError(f"invalid consumed length for lane {lane.lane_id}: {consumed}")
            lane.atom_offset += consumed
            if consumed:
                lane.chunk_index += 1
            if lane.atom_offset == self.dataset.record_length(request.record_index):
                self._refill_lane(lane)
        self._pending = None

    def rollback(self) -> None:
        self._pending = None

    def state_dict(self) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("cannot checkpoint a lane scheduler with an uncommitted peek")
        return {
            "version": 1,
            "dataset_uuid": self.dataset.dataset_uuid,
            "lane_count": self.lane_count,
            "chunk_length": self.chunk_length,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "repeat": self.repeat,
            "generation": self.generation,
            "next_record_index": self._next_record_index,
            "lanes": [asdict(lane) for lane in self.lanes],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "dataset_uuid": self.dataset.dataset_uuid,
            "lane_count": self.lane_count,
            "chunk_length": self.chunk_length,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "repeat": self.repeat,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"lane scheduler state mismatch for {key}: {state.get(key)!r} != {value!r}")
        self._start_generation(int(state["generation"]))
        next_record_index = int(state["next_record_index"])
        if next_record_index < 0 or next_record_index > len(self._rank_records):
            raise ValueError("invalid next_record_index in lane scheduler state")
        self._next_record_index = next_record_index
        lane_states = state["lanes"]
        if len(lane_states) != self.lane_count:
            raise ValueError("lane state count mismatch")
        self.lanes = [LaneCursor(**lane_state) for lane_state in lane_states]
        for lane in self.lanes:
            if lane.record_index is not None:
                length = self.dataset.record_length(lane.record_index)
                if lane.atom_offset < 0 or lane.atom_offset >= length:
                    raise ValueError(f"invalid atom_offset in lane {lane.lane_id}")
        self._pending = None
