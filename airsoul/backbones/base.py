"""Common interface for explicit-state sequence backbones."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _flatten_tensor_tree(value: Any) -> tuple[list[torch.Tensor], Any]:
    if isinstance(value, torch.Tensor):
        return [value], ("tensor",)
    if isinstance(value, dict):
        leaves: list[torch.Tensor] = []
        entries = []
        for key, item in value.items():
            item_leaves, item_spec = _flatten_tensor_tree(item)
            leaves.extend(item_leaves)
            entries.append((key, item_spec))
        return leaves, ("dict", entries)
    if isinstance(value, list):
        leaves = []
        specs = []
        for item in value:
            item_leaves, item_spec = _flatten_tensor_tree(item)
            leaves.extend(item_leaves)
            specs.append(item_spec)
        return leaves, ("list", specs)
    if isinstance(value, tuple):
        leaves = []
        specs = []
        for item in value:
            item_leaves, item_spec = _flatten_tensor_tree(item)
            leaves.extend(item_leaves)
            specs.append(item_spec)
        return leaves, ("tuple", specs)
    if is_dataclass(value):
        leaves = []
        specs = []
        for item in fields(value):
            item_leaves, item_spec = _flatten_tensor_tree(getattr(value, item.name))
            leaves.extend(item_leaves)
            specs.append((item.name, item_spec))
        return leaves, ("dataclass", value, specs)
    return [], ("constant", value)


def _unflatten_tensor_tree(leaves: list[torch.Tensor], spec: Any) -> Any:
    iterator = iter(leaves)

    def build(node: Any) -> Any:
        kind = node[0]
        if kind == "tensor":
            return next(iterator)
        if kind == "constant":
            return node[1]
        if kind == "dict":
            return {key: build(child) for key, child in node[1]}
        if kind == "list":
            return [build(child) for child in node[1]]
        if kind == "tuple":
            return tuple(build(child) for child in node[1])
        if kind == "dataclass":
            return replace(node[1], **{name: build(child) for name, child in node[2]})
        raise ValueError(f"unknown tensor-tree node: {kind}")

    result = build(spec)
    try:
        next(iterator)
    except StopIteration:
        return result
    raise ValueError("tensor-tree contains more leaves than its specification")


@dataclass(frozen=True)
class BackboneCapabilities:
    state_kind: str
    supports_padding_mask: bool = True
    supports_memory_update_mask: bool = True
    supports_compile: bool = False
    supports_activation_checkpointing: bool = True
    fsdp_layer_classes: tuple[type[nn.Module], ...] = ()


@dataclass
class BackboneOutput:
    hidden_states: torch.Tensor
    state: Any
    metrics: Mapping[str, Any] = field(default_factory=dict)


class StatefulBackbone(nn.Module, ABC):
    capabilities: BackboneCapabilities

    def __init__(self) -> None:
        super().__init__()
        self.activation_checkpointing = False

    def set_activation_checkpointing(self, enabled: bool) -> None:
        if enabled and not self.capabilities.supports_activation_checkpointing:
            raise ValueError("the selected backbone does not support activation checkpointing")
        self.activation_checkpointing = bool(enabled)

    def _call_checkpointed_layer(self, layer: nn.Module, hidden_states: torch.Tensor,
                                 state: Any, **kwargs: Any) -> tuple[torch.Tensor, Any]:
        if not self.activation_checkpointing or not self.training:
            return layer(hidden_states, state, **kwargs)
        flat_state, state_spec = _flatten_tensor_tree(state)
        output_spec: list[Any] = []

        def run(value: torch.Tensor, *state_leaves: Any):
            restored_state = _unflatten_tensor_tree(list(state_leaves), state_spec)
            output, next_state = layer(value, restored_state, **kwargs)
            flat_next_state, next_spec = _flatten_tensor_tree(next_state)
            output_spec[:] = [next_spec]
            return (output, *flat_next_state)

        result = checkpoint(run, hidden_states, *flat_state, use_reentrant=False)
        if not isinstance(result, tuple):
            raise RuntimeError("checkpointed stateful layer returned an invalid output")
        return result[0], _unflatten_tensor_tree(list(result[1:]), output_spec[0])

    @abstractmethod
    def forward_chunk(self, hidden_states: torch.Tensor, *, state: Any = None,
                      attention_mask: torch.Tensor | None = None,
                      memory_update_mask: torch.Tensor | None = None,
                      reset_mask: torch.Tensor | None = None,
                      position_ids: torch.Tensor | None = None,
                      return_state: bool = True) -> BackboneOutput:
        raise NotImplementedError

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> BackboneOutput:
        return self.forward_chunk(hidden_states, **kwargs)
