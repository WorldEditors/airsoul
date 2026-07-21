"""Public PyTorch distributed/FSDP2 setup, isolated from model and data code."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized_here: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def close(self) -> None:
        if self.initialized_here and dist.is_initialized():
            dist.destroy_process_group()

    def __enter__(self) -> "DistributedContext":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def init_distributed(backend: str | None = None) -> DistributedContext:
    """Initialize from torchrun environment variables; never spawn children."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = backend or "nccl"
    else:
        device = torch.device("cpu")
        backend = backend or "gloo"
    initialized_here = False
    if not dist.is_initialized() and (world_size > 1 or "RANK" in os.environ):
        dist.init_process_group(backend=backend, init_method="env://")
        initialized_here = True
    return DistributedContext(rank, local_rank, world_size, device, initialized_here)


def _mixed_precision_policy(dtype: torch.dtype):
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy
    except ImportError:
        return None
    return MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)


def apply_fsdp2(model: torch.nn.Module, context: DistributedContext, *,
                dtype: torch.dtype = torch.bfloat16,
                reshard_after_forward: bool = False) -> torch.nn.Module:
    """Shard layer blocks bottom-up with the composable FSDP2 public API."""
    if not dist.is_initialized():
        # A single-process CPU/debug run does not need sharding.
        return model.to(context.device)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
    except ImportError as error:
        raise RuntimeError("this PyTorch build does not provide the FSDP2 public API") from error
    mesh = init_device_mesh(context.device.type, (context.world_size,), mesh_dim_names=("dp",))
    policy = _mixed_precision_policy(dtype)
    kwargs: dict[str, Any] = {
        "mesh": mesh,
        "reshard_after_forward": reshard_after_forward,
    }
    if policy is not None:
        kwargs["mp_policy"] = policy
    capabilities = getattr(getattr(model, "backbone", None), "capabilities", None)
    layer_types = () if capabilities is None else capabilities.fsdp_layer_classes
    if layer_types:
        for module in model.modules():
            if isinstance(module, layer_types):
                fully_shard(module, **kwargs)
    fully_shard(model, **kwargs)
    return model.to(context.device)
