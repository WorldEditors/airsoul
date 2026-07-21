"""Configuration and lazy construction for unified sequence backbones."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

from .base import StatefulBackbone


_BACKBONE_NAMES = frozenset({"kda", "gdn", "gdn2", "transformer_swa"})


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int | None = None
    dropout: float = 0.0
    window_size: int = 4096
    sink_tokens: int = 0
    expand_v: float = 1.0
    conv_size: int = 4
    mode: str = "chunk"

    def __post_init__(self) -> None:
        normalized_name = self.name.lower().replace("-", "_")
        object.__setattr__(self, "name", normalized_name)
        if normalized_name not in _BACKBONE_NAMES:
            raise ValueError(
                f"unknown backbone {self.name!r}; expected one of {sorted(_BACKBONE_NAMES)}"
            )
        if self.hidden_size < 1 or self.num_layers < 1 or self.num_heads < 1:
            raise ValueError("hidden_size, num_layers, and num_heads must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.intermediate_size is not None and self.intermediate_size < 1:
            raise ValueError("intermediate_size must be positive")
        if self.window_size < 1 or self.sink_tokens < 0:
            raise ValueError("window_size must be positive and sink_tokens non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.mode not in {"chunk", "fused_recurrent"}:
            raise ValueError("mode must be 'chunk' or 'fused_recurrent'")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "BackboneConfig":
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(config) - known)
        if unknown:
            raise ValueError(f"unknown backbone config fields: {unknown}")
        return cls(**dict(config))


def build_backbone(config: BackboneConfig | Mapping[str, Any]) -> StatefulBackbone:
    """Build a backbone without importing optional FLA modules eagerly."""
    if not isinstance(config, BackboneConfig):
        config = BackboneConfig.from_mapping(config)
    intermediate_size = config.intermediate_size or 4 * config.hidden_size
    if config.name == "transformer_swa":
        from .transformer_swa import TransformerSWABackbone

        return TransformerSWABackbone(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            intermediate_size=intermediate_size,
            window_size=config.window_size,
            sink_tokens=config.sink_tokens,
            dropout=config.dropout,
        )

    from .fla import FLABackbone

    return FLABackbone(
        architecture=config.name,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        intermediate_size=intermediate_size,
        expand_v=config.expand_v,
        conv_size=config.conv_size,
        mode=config.mode,
    )
