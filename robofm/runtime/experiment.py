"""Strict, versioned experiment configuration for the unified runtime."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping, TypeVar

from robofm.backbones import BackboneConfig
from robofm.models.unified_sequence import UnifiedModelConfig

from .config import CheckpointConfig, LoggingConfig, RuntimeConfig, TBPTTConfig


T = TypeVar("T")


def _strict_dataclass(cls: type[T], values: Mapping[str, Any]) -> T:
    known = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {unknown}")
    return cls(**dict(values))


@dataclass(frozen=True)
class DataConfig:
    dataset_root: str
    shuffle: bool = True
    repeat: bool = True
    seed: int = 0


@dataclass(frozen=True)
class TokenizerConfig:
    pad_token_id: int


@dataclass(frozen=True)
class ImageConfig:
    channels: int = 3
    encoder_width: int = 64
    target_weight: float = 1.0
    detach_targets: bool = True


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.name != "adamw":
            raise ValueError("only the public torch.optim.AdamW optimizer is currently supported")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.eps <= 0:
            raise ValueError("invalid optimizer scalar")
        object.__setattr__(self, "betas", tuple(self.betas))
        if len(self.betas) != 2 or not all(0 <= value < 1 for value in self.betas):
            raise ValueError("optimizer betas must contain two values in [0, 1)")


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    data: DataConfig
    tokenizer: TokenizerConfig
    image: ImageConfig
    model: UnifiedModelConfig
    backbone: BackboneConfig
    runtime: RuntimeConfig
    tbptt: TBPTTConfig
    optimizer: OptimizerConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig
    max_optimizer_steps: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported experiment schema version: {self.schema_version}")
        if self.max_optimizer_steps is not None and self.max_optimizer_steps < 1:
            raise ValueError("max_optimizer_steps must be positive")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        required_sections = {
            "schema_version", "data", "tokenizer", "image", "model", "backbone",
            "runtime", "tbptt", "optimizer", "checkpoint", "logging",
        }
        allowed = required_sections | {"max_optimizer_steps"}
        unknown = sorted(set(raw) - allowed)
        missing = sorted(required_sections - set(raw))
        if unknown or missing:
            raise ValueError(f"experiment config unknown={unknown}, missing={missing}")
        backbone = BackboneConfig.from_mapping(raw["backbone"])
        image = _strict_dataclass(ImageConfig, raw["image"])
        model_values = dict(raw["model"])
        model_values.update({
            "backbone": backbone,
            "image_channels": image.channels,
            "image_width": image.encoder_width,
            "image_target_weight": image.target_weight,
            "detach_image_targets": image.detach_targets,
        })
        model = _strict_dataclass(UnifiedModelConfig, model_values)
        return cls(
            schema_version=int(raw["schema_version"]),
            data=_strict_dataclass(DataConfig, raw["data"]),
            tokenizer=_strict_dataclass(TokenizerConfig, raw["tokenizer"]),
            image=image,
            model=model,
            backbone=backbone,
            runtime=_strict_dataclass(RuntimeConfig, raw["runtime"]),
            tbptt=_strict_dataclass(TBPTTConfig, raw["tbptt"]),
            optimizer=_strict_dataclass(OptimizerConfig, raw["optimizer"]),
            checkpoint=_strict_dataclass(CheckpointConfig, raw["checkpoint"]),
            logging=_strict_dataclass(LoggingConfig, raw["logging"]),
            max_optimizer_steps=raw.get("max_optimizer_steps"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))
