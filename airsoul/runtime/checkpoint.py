"""Distributed checkpoints including exact lane cursors and explicit memory."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist

from airsoul.backbones import state_to


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


class CheckpointManager:
    def __init__(self, directory: str, *, dataset_uuid: str, manifest_hash: str,
                 resolved_config: Mapping[str, Any] | None = None):
        self.root = Path(directory).resolve()
        self.dataset_uuid = dataset_uuid
        self.manifest_hash = manifest_hash
        self.resolved_config = dict(resolved_config or {})

    def _checkpoint_path(self, optimizer_step: int) -> Path:
        return self.root / f"checkpoint-{optimizer_step:012d}"

    def latest(self) -> Path | None:
        marker = self.root / "LATEST"
        if marker.is_file():
            candidate = self.root / marker.read_text(encoding="ascii").strip()
            if (candidate / "COMMITTED").is_file():
                return candidate
        if not self.root.is_dir():
            return None
        candidates = sorted(
            path for path in self.root.glob("checkpoint-*")
            if (path / "COMMITTED").is_file()
        )
        return candidates[-1] if candidates else None

    def save(self, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             lr_scheduler: Any, lane_scheduler: Any, model_state: Any,
             progress: Mapping[str, Any]) -> Path:
        """Save at a flushed optimizer boundary; callers must have no pending grads."""
        from torch.distributed.checkpoint import save
        from torch.distributed.checkpoint.state_dict import get_state_dict

        optimizer_step = int(progress["optimizer_step"])
        final = self._checkpoint_path(optimizer_step)
        temporary = self.root / f".{final.name}.incomplete"
        self.root.mkdir(parents=True, exist_ok=True)
        conflict = torch.tensor(
            final.exists() or temporary.exists(), dtype=torch.uint8,
            device=(torch.device("cuda", torch.cuda.current_device())
                    if torch.cuda.is_available() else torch.device("cpu")),
        )
        if dist.is_initialized():
            dist.all_reduce(conflict, op=dist.ReduceOp.MAX)
        if conflict.item():
            raise FileExistsError(f"checkpoint generation already exists: {final}")
        if _rank() == 0:
            temporary.mkdir()
        _barrier()

        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
        save(
            {"model": model_state_dict, "optimizer": optimizer_state_dict},
            checkpoint_id=str(temporary / "distributed"),
        )
        local_extra = {
            "version": 1,
            "rank": _rank(),
            "world_size": _world_size(),
            "dataset_uuid": self.dataset_uuid,
            "manifest_hash": self.manifest_hash,
            "lane_scheduler": lane_scheduler.state_dict(),
            "model_state": state_to(model_state, "cpu"),
            "rng": capture_rng_state(),
            "progress": dict(progress),
            "lr_scheduler": None if lr_scheduler is None else lr_scheduler.state_dict(),
        }
        torch.save(local_extra, temporary / f"rank-{_rank():05d}-extra.pt")
        _barrier()
        if _rank() == 0:
            metadata = {
                "version": 1,
                "dataset_uuid": self.dataset_uuid,
                "manifest_hash": self.manifest_hash,
                "world_size": _world_size(),
                "optimizer_step": optimizer_step,
                "resolved_config": self.resolved_config,
            }
            with open(temporary / "metadata.json", "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, sort_keys=True, indent=2, ensure_ascii=True)
                handle.flush()
                os.fsync(handle.fileno())
            (temporary / "COMMITTED").write_text("ok\n", encoding="ascii")
            os.replace(temporary, final)
            latest_temp = self.root / ".LATEST.tmp"
            latest_temp.write_text(final.name + "\n", encoding="ascii")
            os.replace(latest_temp, self.root / "LATEST")
        _barrier()
        return final

    def load(self, checkpoint: str | os.PathLike[str] | None, *,
             model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             lr_scheduler: Any, lane_scheduler: Any,
             device: torch.device | str) -> tuple[Any, dict[str, Any]]:
        from torch.distributed.checkpoint import load
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

        path = self.latest() if checkpoint is None else Path(checkpoint).resolve()
        if path is None or not (path / "COMMITTED").is_file():
            raise FileNotFoundError(f"no committed checkpoint found at {path or self.root}")
        with open(path / "metadata.json", "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata["dataset_uuid"] != self.dataset_uuid:
            raise ValueError("checkpoint dataset UUID does not match the open dataset")
        if metadata.get("manifest_hash") != self.manifest_hash:
            raise ValueError("checkpoint manifest hash does not match the open dataset")
        if int(metadata["world_size"]) != _world_size():
            raise ValueError("exact resume requires the checkpoint world size")

        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
        distributed_state = {"model": model_state_dict, "optimizer": optimizer_state_dict}
        load(distributed_state, checkpoint_id=str(path / "distributed"))
        set_state_dict(
            model,
            optimizer,
            model_state_dict=distributed_state["model"],
            optim_state_dict=distributed_state["optimizer"],
        )
        extra = torch.load(
            path / f"rank-{_rank():05d}-extra.pt", map_location="cpu", weights_only=False
        )
        if (extra["dataset_uuid"] != self.dataset_uuid or
                extra.get("manifest_hash") != self.manifest_hash or
                extra["world_size"] != _world_size()):
            raise ValueError("rank checkpoint metadata mismatch")
        lane_scheduler.load_state_dict(extra["lane_scheduler"])
        if lr_scheduler is not None and extra["lr_scheduler"] is not None:
            lr_scheduler.load_state_dict(extra["lr_scheduler"])
        restore_rng_state(extra["rng"])
        return state_to(extra["model_state"], device), dict(extra["progress"])
