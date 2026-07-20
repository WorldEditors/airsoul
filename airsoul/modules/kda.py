"""FLA Kimi Delta Attention adapter.

The rest of AirSoul expects temporal modules to return ``(output, cache)``
and to accept the legacy per-layer cache format.  FLA's KDA layer uses its
``Cache`` object internally, so this adapter converts at the boundary.
"""

import torch.nn as nn
from fla.layers.kda import KimiDeltaAttention
from fla.models.utils import Cache
from airsoul.utils import memory_cpy


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
        if need_cache:
            past_key_values = Cache.from_legacy_cache(
                [memory_cpy(cache)] if cache is not None else None
            )
        else:
            past_key_values = None

        out, _, new_cache = self.encoder(
            hidden_states=x,
            past_key_values=past_key_values,
            use_cache=need_cache,
        )
        if not need_cache or new_cache is None:
            return out, None
        return out, memory_cpy(new_cache[0])
