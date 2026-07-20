"""FLA Kimi Delta Attention adapter.

The rest of AirSoul expects temporal modules to return ``(output, cache)``
and to accept the legacy per-layer cache format.  FLA's KDA layer uses its
``Cache`` object internally, so this adapter converts at the boundary.
"""

import importlib.machinery
import os
import sys
import types

# FLA versions that eagerly register optional backends import ``tilelang``
# before checking FLA_TILELANG.  A broken TileLang/TVM installation therefore
# prevents even Triton-only layers from importing.  AirSoul's KDA backend uses
# Triton, so hide TileLang from FLA before importing it.  Supplying a module
# spec also keeps newer FLA versions' find_spec-based detection happy.
os.environ["FLA_TILELANG"] = "0"
if "tilelang" not in sys.modules:
    _tilelang_stub = types.ModuleType("tilelang")
    _tilelang_stub.__spec__ = importlib.machinery.ModuleSpec("tilelang", loader=None)
    sys.modules["tilelang"] = _tilelang_stub

import torch.nn as nn
from fla.layers.kda import KimiDeltaAttention
from airsoul.utils import memory_cpy


class _KDACache:
    """Minimal cache protocol consumed by FLA attention layers."""

    def __init__(self, state=None):
        self.states = [] if state is None else [memory_cpy(state)]
        self.seen_tokens = 0

    def __len__(self):
        return len(self.states)

    def __getitem__(self, layer_idx):
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
        for key, value in (
            ("recurrent_state", recurrent_state),
            ("attn_state", attn_state),
            ("conv_state", conv_state),
            ("ffn_state", ffn_state),
        ):
            if value is not None:
                state[key] = value
        self.seen_tokens += offset or 0
        return state

    def get_seq_length(self, layer_idx=0, **kwargs):
        return self.seen_tokens if layer_idx < len(self.states) else 0


class KDABlock(nn.Module):
    def __init__(self, io_size=512, num_heads=4, expand_v=1.0,
                 d_conv=4, layer_idx=0, is_generate=False, **kwargs):
        super().__init__()
        if io_size % num_heads != 0:
            raise ValueError(f"KDA hidden size ({io_size}) must be divisible by num_heads ({num_heads})")
        self.hidden_size = io_size
        self.layer_idx = layer_idx
        self.encoder = KimiDeltaAttention(
            hidden_size=io_size,
            expand_v=expand_v,
            head_dim=io_size // num_heads,
            num_heads=num_heads,
            mode="fused_recurrent" if is_generate else "chunk",
            conv_size=d_conv,
            layer_idx=0,
        )

    def forward(self, x, cache=None, need_cache=False):
        past_key_values = _KDACache(cache) if need_cache else None

        out, _, new_cache = self.encoder(
            hidden_states=x,
            past_key_values=past_key_values,
            use_cache=need_cache,
        )
        if not need_cache or new_cache is None:
            return out, None
        return out, memory_cpy(new_cache[0])
