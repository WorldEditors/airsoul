"""Native PyTorch causal sliding-window Transformer with explicit state."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from .base import BackboneCapabilities, BackboneOutput, StatefulBackbone
from .state import reset_state_lanes


def _apply_rotary(tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    head_dim = tensor.shape[-1]
    rotary_dim = head_dim - head_dim % 2
    if rotary_dim == 0:
        return tensor
    frequencies = torch.arange(0, rotary_dim, 2, device=tensor.device, dtype=torch.float32)
    frequencies = torch.exp(-math.log(10_000.0) * frequencies / rotary_dim)
    angles = positions.to(torch.float32).unsqueeze(1).unsqueeze(-1) * frequencies
    cos = torch.cos(angles).to(tensor.dtype)
    sin = torch.sin(angles).to(tensor.dtype)
    rotary = tensor[..., :rotary_dim]
    even, odd = rotary[..., 0::2], rotary[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    rotated = rotated.flatten(-2)
    if rotary_dim == head_dim:
        return rotated
    return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)


class TransformerSWALayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int,
                 dropout: float, window_size: int, sink_tokens: int):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.sink_tokens = sink_tokens
        self.state_size = window_size + sink_tokens
        self.norm1 = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.dropout = dropout

    def _append_state(self, old: dict[str, torch.Tensor] | None,
                      current_hidden: torch.Tensor, current_positions: torch.Tensor,
                      update_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, _, hidden = current_hidden.shape
        rows_hidden: list[torch.Tensor] = []
        rows_position: list[torch.Tensor] = []
        rows_valid: list[torch.Tensor] = []
        rows_sink: list[torch.Tensor] = []
        for lane in range(batch):
            parts_hidden = []
            parts_position = []
            parts_sink = []
            old_sink_count = 0
            if old is not None:
                old_valid = old["valid"][lane]
                parts_hidden.append(old["hidden"][lane][old_valid])
                parts_position.append(old["positions"][lane][old_valid])
                old_sink = old["sink"][lane][old_valid]
                parts_sink.append(old_sink)
                old_sink_count = int(old_sink.sum().item())
            selected_hidden = current_hidden[lane][update_mask[lane]]
            selected_positions = current_positions[lane][update_mask[lane]]
            current_sink_count = min(
                max(self.sink_tokens - old_sink_count, 0), selected_hidden.shape[0]
            )
            current_sink = torch.arange(
                selected_hidden.shape[0], device=current_hidden.device
            ) < current_sink_count
            parts_hidden.append(selected_hidden)
            parts_position.append(selected_positions)
            parts_sink.append(current_sink)
            all_hidden = torch.cat(parts_hidden, dim=0)
            all_position = torch.cat(parts_position, dim=0)
            all_sink = torch.cat(parts_sink, dim=0)
            keep = all_sink.clone()
            non_sink_indices = torch.nonzero(~all_sink, as_tuple=False).flatten()[-self.window_size:]
            keep[non_sink_indices] = True
            lane_hidden = all_hidden[keep]
            lane_position = all_position[keep]
            lane_sink = all_sink[keep]
            count = lane_hidden.shape[0]
            padded_hidden = F.pad(lane_hidden, (0, 0, self.state_size - count, 0))
            padded_position = F.pad(lane_position, (self.state_size - count, 0))
            padded_sink = F.pad(lane_sink, (self.state_size - count, 0))
            valid = torch.arange(self.state_size, device=current_hidden.device) >= self.state_size - count
            rows_hidden.append(padded_hidden)
            rows_position.append(padded_position)
            rows_valid.append(valid)
            rows_sink.append(padded_sink)
        return {
            "hidden": torch.stack(rows_hidden),
            "positions": torch.stack(rows_position),
            "valid": torch.stack(rows_valid),
            "sink": torch.stack(rows_sink),
        }

    def _current_sink_mask(self, old: dict[str, torch.Tensor] | None,
                           update_mask: torch.Tensor) -> torch.Tensor:
        rows = []
        for lane in range(update_mask.shape[0]):
            old_count = 0 if old is None else int((old["sink"][lane] & old["valid"][lane]).sum().item())
            available = max(self.sink_tokens - old_count, 0)
            update_order = torch.cumsum(update_mask[lane].to(torch.long), dim=0)
            rows.append(update_mask[lane] & (update_order <= available))
        return torch.stack(rows)

    def forward(self, hidden_states: torch.Tensor, layer_state: Any = None, *,
                valid_mask: torch.Tensor, update_mask: torch.Tensor,
                position_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        layer_input = hidden_states
        normalized = self.norm1(hidden_states)
        current_sink = self._current_sink_mask(layer_state, update_mask)
        if layer_state is None:
            kv_hidden = normalized
            kv_positions = position_ids
            kv_valid = update_mask
            kv_sink = current_sink
        else:
            cached = self.norm1(layer_state["hidden"])
            kv_hidden = torch.cat((cached, normalized), dim=1)
            kv_positions = torch.cat((layer_state["positions"], position_ids), dim=1)
            kv_valid = torch.cat((layer_state["valid"], update_mask), dim=1)
            kv_sink = torch.cat((layer_state["sink"], current_sink), dim=1)

        batch, query_length, _ = normalized.shape
        key_length = kv_hidden.shape[1]
        q = self.q_proj(normalized).view(batch, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_hidden).view(batch, key_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_hidden).view(batch, key_length, self.num_heads, self.head_dim).transpose(1, 2)
        q = _apply_rotary(q, position_ids)
        k = _apply_rotary(k, kv_positions)
        # A fallback key is used for all-masked rows. Zeroing invalid K/V keeps
        # that numerical safeguard from leaking a memory-masked atom.
        kv_scale = kv_valid.unsqueeze(1).unsqueeze(-1)
        k = k * kv_scale
        v = v * kv_scale

        query_positions = position_ids.unsqueeze(-1)
        key_positions = kv_positions.unsqueeze(-2)
        allowed = (
            kv_valid.unsqueeze(-2) &
            (key_positions <= query_positions) &
            ((key_positions > query_positions - self.window_size) | kv_sink.unsqueeze(-2))
        )
        # Every all-masked query needs one harmless key to avoid NaNs in older SDPA kernels.
        if key_length:
            fallback = torch.zeros_like(allowed)
            fallback[..., -1] = True
            allowed = torch.where(allowed.any(dim=-1, keepdim=True), allowed, fallback)
        attention = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=allowed.unsqueeze(1),
            dropout_p=self.dropout if self.training else 0.0,
        )
        attention = attention.transpose(1, 2).reshape(batch, query_length, self.hidden_size)
        attention = attention * valid_mask.unsqueeze(-1)
        hidden_states = layer_input + self.o_proj(attention)
        hidden_states = hidden_states + self.ffn(self.norm2(hidden_states)) * valid_mask.unsqueeze(-1)
        next_state = self._append_state(layer_state, layer_input, position_ids, update_mask)
        return hidden_states, next_state

class TransformerSWABackbone(StatefulBackbone):
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int,
                 intermediate_size: int, window_size: int, dropout: float = 0.0,
                 sink_tokens: int = 0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerSWALayer(
                hidden_size, num_heads, intermediate_size, dropout, window_size, sink_tokens
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)
        self.capabilities = BackboneCapabilities(
            state_kind="bounded_kv",
            # Per-lane reset/sink state currently contains data-dependent Python
            # control flow, so compile stays capability-gated off until verified.
            supports_compile=False,
            fsdp_layer_classes=(TransformerSWALayer,),
        )

    def _forward_segment(self, hidden_states: torch.Tensor, *, state: Any,
                         valid: torch.Tensor, update: torch.Tensor,
                         position_ids: torch.Tensor,
                         return_state: bool) -> BackboneOutput:
        layer_states = [None] * len(self.layers) if state is None else state["layers"]
        if len(layer_states) != len(self.layers):
            raise ValueError("Transformer-SWA state layer count does not match the model")
        next_layers = []
        for layer, layer_state in zip(self.layers, layer_states):
            hidden_states, next_state = self._call_checkpointed_layer(
                layer,
                hidden_states,
                layer_state,
                valid_mask=valid,
                update_mask=update,
                position_ids=position_ids,
            )
            next_layers.append(next_state)
        hidden_states = self.final_norm(hidden_states) * valid.unsqueeze(-1)
        return BackboneOutput(
            hidden_states=hidden_states,
            state={"layers": next_layers} if return_state else None,
        )

    def forward_chunk(self, hidden_states: torch.Tensor, *, state: Any = None,
                      attention_mask: torch.Tensor | None = None,
                      memory_update_mask: torch.Tensor | None = None,
                      reset_mask: torch.Tensor | None = None,
                      position_ids: torch.Tensor | None = None,
                      return_state: bool = True) -> BackboneOutput:
        batch, length, _ = hidden_states.shape
        device = hidden_states.device
        if length == 0:
            return BackboneOutput(
                hidden_states=hidden_states,
                state=state if return_state else None,
                metrics={"valid_atoms": hidden_states.new_zeros((), dtype=torch.long),
                         "state_window": self.layers[0].window_size,
                         "state_sink_tokens": self.layers[0].sink_tokens},
            )
        valid = torch.ones((batch, length), dtype=torch.bool, device=device)
        if attention_mask is not None:
            valid &= attention_mask.to(device=device, dtype=torch.bool)
        update = valid if memory_update_mask is None else (
            valid & memory_update_mask.to(device=device, dtype=torch.bool)
        )
        if position_ids is None:
            position_ids = torch.arange(length, device=device).expand(batch, -1)
        else:
            position_ids = position_ids.to(device=device, dtype=torch.long)
        reset = torch.zeros((batch, length), dtype=torch.bool, device=device)
        if reset_mask is not None:
            reset_mask = reset_mask.to(device=device, dtype=torch.bool)
            if reset_mask.ndim == 1:
                if reset_mask.shape != (batch,):
                    raise ValueError("one-dimensional reset_mask must have shape [batch]")
                reset[:, 0] = reset_mask
            elif reset_mask.shape == (batch, length):
                reset = reset_mask & valid
            else:
                raise ValueError("reset_mask must have shape [batch] or [batch, length]")

        # Split at the union of reset positions. This preserves exact per-lane
        # reset semantics without turning the normal path into token-wise calls.
        reset_boundaries = reset.any(dim=0)
        if dist.is_initialized():
            min_length = torch.tensor(length, dtype=torch.long, device=device)
            max_length = min_length.clone()
            dist.all_reduce(min_length, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_length, op=dist.ReduceOp.MAX)
            if min_length.item() != max_length.item():
                raise ValueError("FSDP2 Transformer-SWA requires equal chunk lengths on all ranks")
            global_boundaries = reset_boundaries.to(dtype=torch.uint8)
            dist.all_reduce(global_boundaries, op=dist.ReduceOp.MAX)
            reset_boundaries = global_boundaries.to(dtype=torch.bool)
        reset_positions = torch.nonzero(reset_boundaries, as_tuple=False).flatten().tolist()
        boundaries = sorted(set([0, length, *[int(value) for value in reset_positions]]))
        outputs: list[torch.Tensor] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if start == end:
                continue
            state = reset_state_lanes(state, reset[:, start])
            segment = self._forward_segment(
                hidden_states[:, start:end],
                state=state,
                valid=valid[:, start:end],
                update=update[:, start:end],
                position_ids=position_ids[:, start:end],
                return_state=True,
            )
            outputs.append(segment.hidden_states)
            state = segment.state
        output_hidden = torch.cat(outputs, dim=1) if outputs else hidden_states[:, :0]
        return BackboneOutput(
            hidden_states=output_hidden,
            state=state if return_state else None,
            metrics={
                "valid_atoms": valid.sum(),
                "state_window": self.layers[0].window_size,
                "state_sink_tokens": self.layers[0].sink_tokens,
            },
        )
