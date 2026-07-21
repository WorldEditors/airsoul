"""FLA public-layer adapters with AirSoul-owned explicit recurrent state."""

from __future__ import annotations

import importlib
import importlib.machinery
import inspect
import os
import sys
import types
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from .base import BackboneCapabilities, BackboneOutput, StatefulBackbone
from .state import clone_state, merge_state_lanes, reset_state_lanes


def _prepare_fla_import() -> None:
    # Some FLA releases import TileLang before consulting FLA_TILELANG. Triton
    # remains the supported default, so an unavailable/broken optional backend
    # must not make importing a public FLA layer fail.
    os.environ.setdefault("FLA_TILELANG", "0")
    if os.environ.get("FLA_TILELANG") != "1" and "tilelang" not in sys.modules:
        stub = types.ModuleType("tilelang")
        stub.__spec__ = importlib.machinery.ModuleSpec("tilelang", loader=None)
        sys.modules["tilelang"] = stub


def _load_public_layer(architecture: str) -> type[nn.Module]:
    _prepare_fla_import()
    candidates = {
        "kda": [("fla.layers.kda", "KimiDeltaAttention")],
        "gdn": [("fla.layers.gated_deltanet", "GatedDeltaNet")],
        "gdn2": [
            ("fla.layers.gated_deltanet2", "GatedDeltaNet2"),
            ("fla.layers.gated_deltanet2", "GatedDeltaNetV2"),
            ("fla.layers.gdn2", "GatedDeltaNet2"),
        ],
    }[architecture]
    errors: list[str] = []
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            layer = getattr(module, class_name)
        except (ImportError, AttributeError) as error:
            errors.append(f"{module_name}.{class_name}: {error}")
            continue
        if not issubclass(layer, nn.Module):
            errors.append(f"{module_name}.{class_name}: not a torch.nn.Module")
            continue
        return layer
    detail = "; ".join(errors)
    raise RuntimeError(
        f"FLA public {architecture.upper()} layer is unavailable in this installation. "
        f"Install a supported flash-linear-attention release. Probes: {detail}"
    )


class _ExplicitFLACache:
    """Small public cache protocol used by FLA layers, without FLA Cache classes."""

    def __init__(self, state: Any = None):
        self.states = [] if state is None else [clone_state(state)]
        self.seen_tokens = 0

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, layer_idx: int) -> Any:
        return self.states[layer_idx]

    def update(self, recurrent_state=None, attn_state=None, conv_state=None,
               ffn_state=None, layer_idx=0, offset=1, **kwargs):
        while len(self.states) <= layer_idx:
            self.states.append({
                "recurrent_state": None,
                "attn_state": None,
                "conv_state": None,
                "ffn_state": None,
            })
        state = self.states[layer_idx]
        for name, value in (
            ("recurrent_state", recurrent_state),
            ("attn_state", attn_state),
            ("conv_state", conv_state),
            ("ffn_state", ffn_state),
        ):
            if value is not None:
                state[name] = value
        self.seen_tokens += int(offset or 0)
        return state

    def get_seq_length(self, layer_idx=0, **kwargs) -> int:
        return self.seen_tokens if layer_idx < len(self.states) else 0


def _construct_layer(layer_type: type[nn.Module], **available: Any) -> nn.Module:
    signature = inspect.signature(layer_type.__init__)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = available if accepts_kwargs else {
        name: value for name, value in available.items() if name in signature.parameters
    }
    return layer_type(**kwargs)


class FLASequenceLayer(nn.Module):
    def __init__(self, architecture: str, hidden_size: int, num_heads: int,
                 intermediate_size: int, expand_v: float, conv_size: int,
                 mode: str):
        super().__init__()
        layer_type = _load_public_layer(architecture)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.temporal = _construct_layer(
            layer_type,
            mode=mode,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=hidden_size // num_heads,
            expand_v=expand_v,
            conv_size=conv_size,
            d_conv=conv_size,
            layer_idx=0,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor, state: Any) -> tuple[torch.Tensor, Any]:
        cache = _ExplicitFLACache(state)
        result = self.temporal(
            hidden_states=self.norm1(hidden_states),
            past_key_values=cache,
            use_cache=True,
        )
        if not isinstance(result, tuple) or not result:
            raise RuntimeError("FLA layer did not return its documented tuple output")
        temporal_output = result[0]
        returned_cache = result[2] if len(result) > 2 else cache
        if returned_cache is None:
            returned_cache = cache
        new_state = returned_cache[0] if len(returned_cache) else None
        output = hidden_states + temporal_output
        output = output + self.ffn(self.norm2(output))
        return output, new_state


class FLABackbone(StatefulBackbone):
    def __init__(self, architecture: str, hidden_size: int, num_layers: int,
                 num_heads: int, intermediate_size: int, expand_v: float,
                 conv_size: int, mode: str = "chunk"):
        super().__init__()
        if architecture not in {"kda", "gdn", "gdn2"}:
            raise ValueError(f"unsupported FLA architecture: {architecture}")
        self.architecture = architecture
        self.layers = nn.ModuleList([
            FLASequenceLayer(
                architecture, hidden_size, num_heads, intermediate_size,
                expand_v, conv_size, mode,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)
        self.capabilities = BackboneCapabilities(
            state_kind="recurrent",
            supports_compile=False,
            fsdp_layer_classes=(FLASequenceLayer,),
        )

    def _run(self, hidden_states: torch.Tensor, state: Any) -> tuple[torch.Tensor, Any]:
        layer_states = [None] * len(self.layers) if state is None else state["layers"]
        if len(layer_states) != len(self.layers):
            raise ValueError("FLA state layer count does not match the model")
        next_states = []
        for layer, layer_state in zip(self.layers, layer_states):
            hidden_states, layer_state = self._call_checkpointed_layer(
                layer, hidden_states, layer_state
            )
            next_states.append(layer_state)
        return self.final_norm(hidden_states), {"layers": next_states}

    def forward_chunk(self, hidden_states: torch.Tensor, *, state: Any = None,
                      attention_mask: torch.Tensor | None = None,
                      memory_update_mask: torch.Tensor | None = None,
                      reset_mask: torch.Tensor | None = None,
                      position_ids: torch.Tensor | None = None,
                      return_state: bool = True) -> BackboneOutput:
        del position_ids  # Current public FLA recurrent layers own their positions.
        batch, length, _ = hidden_states.shape
        device = hidden_states.device
        if length == 0:
            return BackboneOutput(
                hidden_states=hidden_states,
                state=state if return_state else None,
                metrics={"valid_atoms": hidden_states.new_zeros((), dtype=torch.long),
                         "masked_recurrent_fallback": torch.zeros((), dtype=torch.bool, device=device)},
            )
        valid = torch.ones((batch, length), dtype=torch.bool, device=device)
        if attention_mask is not None:
            valid &= attention_mask.to(device=device, dtype=torch.bool)
        update = valid if memory_update_mask is None else (
            valid & memory_update_mask.to(device=device, dtype=torch.bool)
        )
        reset = torch.zeros_like(valid)
        if reset_mask is not None:
            raw_reset = reset_mask.to(device=device, dtype=torch.bool)
            if raw_reset.ndim == 1 and raw_reset.shape == (batch,):
                reset[:, 0] = raw_reset
            elif raw_reset.shape == (batch, length):
                reset = raw_reset & valid
            else:
                raise ValueError("reset_mask must have shape [batch] or [batch, length]")

        regular = bool(torch.all(update)) and not bool(torch.any(reset[:, 1:]))
        if dist.is_initialized():
            min_length = torch.tensor(length, dtype=torch.long, device=device)
            max_length = min_length.clone()
            dist.all_reduce(min_length, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_length, op=dist.ReduceOp.MAX)
            if min_length.item() != max_length.item():
                raise ValueError("FSDP2 FLA backbones require equal chunk lengths on all ranks")
            regular_tensor = torch.tensor(regular, dtype=torch.uint8, device=device)
            dist.all_reduce(regular_tensor, op=dist.ReduceOp.MIN)
            regular = bool(regular_tensor.item())
        if regular:
            state = reset_state_lanes(state, reset[:, 0])
            output, state = self._run(hidden_states, state)
            output = output * valid.unsqueeze(-1)
        else:
            pieces: list[torch.Tensor] = []
            for position in range(length):
                state = reset_state_lanes(state, reset[:, position])
                candidate_output, candidate_state = self._run(
                    hidden_states[:, position:position + 1], state
                )
                state = merge_state_lanes(state, candidate_state, update[:, position])
                pieces.append(candidate_output * valid[:, position, None, None])
            output = torch.cat(pieces, dim=1) if pieces else hidden_states[:, :0]
        return BackboneOutput(
            hidden_states=output,
            state=state if return_state else None,
            metrics={
                "valid_atoms": valid.sum(),
                "masked_recurrent_fallback": torch.as_tensor(not regular, device=device),
            },
        )
