"""Operations over nested explicit backbone state trees."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any, Callable

import torch


def _tree_map(function: Callable[[torch.Tensor], torch.Tensor], state: Any) -> Any:
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return function(state)
    if isinstance(state, dict):
        return {key: _tree_map(function, value) for key, value in state.items()}
    if isinstance(state, list):
        return [_tree_map(function, value) for value in state]
    if isinstance(state, tuple):
        return tuple(_tree_map(function, value) for value in state)
    if is_dataclass(state):
        return replace(state, **{
            item.name: _tree_map(function, getattr(state, item.name)) for item in fields(state)
        })
    return state


def detach_state(state: Any) -> Any:
    return _tree_map(lambda tensor: tensor.detach(), state)


def clone_state(state: Any, *, detach: bool = False) -> Any:
    if detach:
        return _tree_map(lambda tensor: tensor.detach().clone(), state)
    return _tree_map(lambda tensor: tensor.clone(), state)


def state_to(state: Any, *args: Any, **kwargs: Any) -> Any:
    return _tree_map(lambda tensor: tensor.to(*args, **kwargs), state)


def reset_state_lanes(state: Any, reset_mask: torch.Tensor | None) -> Any:
    if state is None or reset_mask is None or not bool(torch.any(reset_mask)):
        return state
    reset_mask = reset_mask.to(dtype=torch.bool)

    def reset_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0 or tensor.shape[0] != reset_mask.shape[0]:
            return tensor
        view_shape = (reset_mask.shape[0],) + (1,) * (tensor.ndim - 1)
        return torch.where(reset_mask.view(view_shape), torch.zeros_like(tensor), tensor)

    return _tree_map(reset_tensor, state)


def select_state_lanes(state: Any, lane_indices: torch.Tensor, *, batch_size: int) -> Any:
    """Select batch-leading state leaves while preserving non-batched metadata."""
    lane_indices = lane_indices.to(dtype=torch.long)

    def select(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim and tensor.shape[0] == batch_size:
            return tensor.index_select(0, lane_indices.to(tensor.device))
        return tensor

    return _tree_map(select, state)


def merge_state_lanes(base: Any, update: Any, update_mask: torch.Tensor) -> Any:
    """Take candidate state for selected lanes and preserve the other lanes."""
    if update is None:
        return base
    update_mask = update_mask.to(dtype=torch.bool)
    if base is None:
        base = _tree_map(torch.zeros_like, update)

    def merge(old: Any, new: Any) -> Any:
        if isinstance(new, torch.Tensor):
            if not isinstance(old, torch.Tensor) or old.shape != new.shape:
                raise ValueError("state tensor structure changed while merging lanes")
            if new.ndim and new.shape[0] == update_mask.shape[0]:
                shape = (update_mask.shape[0],) + (1,) * (new.ndim - 1)
                return torch.where(update_mask.to(new.device).view(shape), new, old)
            return new
        if isinstance(new, dict):
            return {key: merge(old[key], value) for key, value in new.items()}
        if isinstance(new, list):
            return [merge(old[index], value) for index, value in enumerate(new)]
        if isinstance(new, tuple):
            return tuple(merge(old[index], value) for index, value in enumerate(new))
        if is_dataclass(new):
            return replace(new, **{
                item.name: merge(getattr(old, item.name), getattr(new, item.name))
                for item in fields(new)
            })
        return new

    return merge(base, update)


def scatter_state_lanes(base: Any, update: Any, lane_indices: torch.Tensor,
                        *, batch_size: int) -> Any:
    """Scatter a smaller state tree into an existing batch-leading state tree."""
    if base is None:
        raise ValueError("base state is required for lane scatter")
    lane_indices = lane_indices.to(dtype=torch.long)

    def scatter(old: Any, new: Any) -> Any:
        if isinstance(new, torch.Tensor):
            if isinstance(old, torch.Tensor) and old.ndim and old.shape[0] == batch_size:
                result = old.clone()
                result.index_copy_(0, lane_indices.to(result.device), new.to(result.device))
                return result
            return new
        if isinstance(new, dict):
            return {key: scatter(old[key], value) for key, value in new.items()}
        if isinstance(new, list):
            return [scatter(old[index], value) for index, value in enumerate(new)]
        if isinstance(new, tuple):
            return tuple(scatter(old[index], value) for index, value in enumerate(new))
        if is_dataclass(new):
            return replace(new, **{
                item.name: scatter(getattr(old, item.name), getattr(new, item.name))
                for item in fields(new)
            })
        return new

    return scatter(base, update)


def state_nbytes(state: Any) -> int:
    total = 0

    def count(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal total
        total += tensor.numel() * tensor.element_size()
        return tensor

    _tree_map(count, state)
    return total
