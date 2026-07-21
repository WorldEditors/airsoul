"""Rank-zero JSONL and optional TensorBoard metrics sink."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping


class MetricLogger:
    def __init__(self, directory: str, *, enabled: bool, tensorboard: bool):
        self.enabled = enabled
        self.root = Path(directory)
        self.writer = None
        self.handle = None
        if not enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.root / "metrics.jsonl", "a", encoding="utf-8", buffering=1)
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                SummaryWriter = None
            if SummaryWriter is not None:
                self.writer = SummaryWriter(log_dir=str(self.root / "tensorboard"))

    def log(self, step: int, metrics: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        record = {"timestamp_unix_s": time.time(), "optimizer_step": int(step)}
        for name, value in metrics.items():
            if hasattr(value, "item"):
                value = value.item()
            record[name] = value
        assert self.handle is not None
        self.handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
        if self.writer is not None:
            for name, value in record.items():
                if name != "optimizer_step" and isinstance(value, (int, float)):
                    self.writer.add_scalar(name, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.handle is not None:
            self.handle.close()

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
