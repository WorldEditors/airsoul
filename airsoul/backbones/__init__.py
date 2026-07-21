"""Stateful causal backbones for unified sequence training."""

from .base import BackboneCapabilities, BackboneOutput, StatefulBackbone
from .registry import BackboneConfig, build_backbone
from .state import (
    clone_state,
    detach_state,
    merge_state_lanes,
    reset_state_lanes,
    scatter_state_lanes,
    select_state_lanes,
    state_nbytes,
    state_to,
)

__all__ = [
    "BackboneCapabilities",
    "BackboneConfig",
    "BackboneOutput",
    "StatefulBackbone",
    "build_backbone",
    "clone_state",
    "detach_state",
    "merge_state_lanes",
    "reset_state_lanes",
    "scatter_state_lanes",
    "select_state_lanes",
    "state_nbytes",
    "state_to",
]
