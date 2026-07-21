"""Validated configuration for the unified training runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TBPTTConfig:
    chunk_length: int = 1024
    lane_count: int = 1
    tbptt_chunks: int = 10
    optimizer_step_chunks: int = 10
    max_atoms_per_step: int | None = None

    def __post_init__(self) -> None:
        values = (self.chunk_length, self.lane_count, self.tbptt_chunks,
                  self.optimizer_step_chunks)
        if any(value < 1 for value in values):
            raise ValueError("TBPTT lengths/counts must be positive")
        if self.optimizer_step_chunks % self.tbptt_chunks:
            raise ValueError("optimizer_step_chunks must be divisible by tbptt_chunks")
        if self.max_atoms_per_step is not None and self.max_atoms_per_step < 1:
            raise ValueError("max_atoms_per_step must be positive")


@dataclass(frozen=True)
class RuntimeConfig:
    dtype: str = "bfloat16"
    grad_clip_norm: float | None = 1.0
    seed: int = 0
    fsdp2: bool = True
    fsdp_reshard_after_forward: bool = False
    activation_checkpointing: bool = True
    compile_model: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("runtime dtype must be float32 or bfloat16")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive")


@dataclass(frozen=True)
class CheckpointConfig:
    directory: str = "./checkpoints-unified"
    every_optimizer_steps: int = 1000

    def __post_init__(self) -> None:
        if self.every_optimizer_steps < 1:
            raise ValueError("checkpoint interval must be positive")


@dataclass(frozen=True)
class LoggingConfig:
    directory: str = "./logs-unified"
    every_optimizer_steps: int = 10
    tensorboard: bool = True

    def __post_init__(self) -> None:
        if self.every_optimizer_steps < 1:
            raise ValueError("logging interval must be positive")
