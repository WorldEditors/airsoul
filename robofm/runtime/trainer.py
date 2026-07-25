"""Lane-aware TBPTT loop with continuous explicit model memory."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import time
from typing import Any

import torch
import torch.distributed as dist

from robofm.backbones import detach_state, state_nbytes
from robofm.dataio import LaneScheduler, collate_chunks
from robofm.dataio.torch_batch import to_torch_batch

from .checkpoint import CheckpointManager
from .config import CheckpointConfig, LoggingConfig, RuntimeConfig, TBPTTConfig
from .distributed import DistributedContext
from .logging import MetricLogger


@dataclass
class TrainProgress:
    chunk_step: int = 0
    consumed_chunks: int = 0
    consumed_atoms: int = 0
    consumed_language_tokens: int = 0
    consumed_images: int = 0
    consumed_records: int = 0
    optimizer_step: int = 0


class UnifiedTrainer:
    def __init__(self, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 lane_scheduler: LaneScheduler, context: DistributedContext,
                 tbptt: TBPTTConfig, runtime: RuntimeConfig,
                 logging: LoggingConfig, checkpoint: CheckpointConfig,
                 lr_scheduler: Any = None, pad_token_id: int = 0):
        if lane_scheduler.chunk_length != tbptt.chunk_length:
            raise ValueError("lane scheduler and TBPTT chunk_length must match")
        if lane_scheduler.lane_count != tbptt.lane_count:
            raise ValueError("lane scheduler and TBPTT lane_count must match")
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lane_scheduler = lane_scheduler
        self.context = context
        self.tbptt = tbptt
        self.runtime = runtime
        self.pad_token_id = int(pad_token_id)
        self.progress = TrainProgress()
        self.model_state = None
        self.logger = MetricLogger(
            logging.directory, enabled=context.is_main, tensorboard=logging.tensorboard
        )
        self.logging_config = logging
        self.checkpoint_config = checkpoint
        self.checkpoints = CheckpointManager(
            checkpoint.directory,
            dataset_uuid=lane_scheduler.dataset.dataset_uuid,
            manifest_hash=lane_scheduler.dataset.manifest_hash,
            resolved_config={
                "tbptt": asdict(tbptt),
                "runtime": asdict(runtime),
                "logging": asdict(logging),
                "checkpoint": asdict(checkpoint),
            },
        )

    def resume(self, checkpoint: str | None = None) -> None:
        self.model_state, progress = self.checkpoints.load(
            checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            lane_scheduler=self.lane_scheduler,
            device=self.context.device,
        )
        self.progress = TrainProgress(**progress)
        self.optimizer.zero_grad(set_to_none=True)

    def _autocast(self):
        if self.runtime.dtype == "bfloat16":
            return torch.autocast(device_type=self.context.device.type, dtype=torch.bfloat16)
        return nullcontext()

    def _global_sum(self, value: torch.Tensor) -> torch.Tensor:
        result = value.detach().to(device=self.context.device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result

    def _global_any(self, value: bool) -> bool:
        flag = torch.tensor(value, dtype=torch.uint8, device=self.context.device)
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def _normalize_and_step(self, accumulated_weight: torch.Tensor) -> tuple[float, float]:
        global_weight = self._global_sum(accumulated_weight)
        if global_weight.item() == 0:
            self.optimizer.zero_grad(set_to_none=True)
            return 0.0, 0.0
        scale = self.context.world_size / global_weight.item()
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        grad_norm = 0.0
        if self.runtime.grad_clip_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.runtime.grad_clip_norm, foreach=False
            )
            grad_norm = float(norm.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.progress.optimizer_step += 1
        return float(global_weight.item()), grad_norm

    def _save(self) -> None:
        self.checkpoints.save(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            lane_scheduler=self.lane_scheduler,
            model_state=self.model_state,
            progress=asdict(self.progress),
        )

    def run(self, *, max_optimizer_steps: int | None = None) -> TrainProgress:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        graph_loss: torch.Tensor | None = None
        chunks_since_backward = 0
        chunks_since_step = 0
        atoms_since_step = 0
        target_weight_since_step = torch.zeros((), device=self.context.device)
        language_loss_since_step = torch.zeros((), device=self.context.device)
        image_loss_since_step = torch.zeros((), device=self.context.device)
        language_atoms_since_step = 0
        image_atoms_since_step = 0
        records_since_step = 0
        reset_atoms_since_step = 0
        padded_slots_since_step = 0
        total_slots_since_step = 0
        data_seconds_since_step = 0.0
        forward_seconds_since_step = 0.0
        backward_seconds_since_step = 0.0
        step_started = time.perf_counter()

        while max_optimizer_steps is None or self.progress.optimizer_step < max_optimizer_steps:
            requests = self.lane_scheduler.peek()
            local_active = any(request is not None for request in requests)
            if not self._global_any(local_active):
                self.lane_scheduler.rollback()
                break
            data_started = time.perf_counter()
            chunks = self.lane_scheduler.read()
            numpy_batch = collate_chunks(
                chunks,
                pad_token_id=self.pad_token_id,
                pad_to_length=self.tbptt.chunk_length,
                dataset=self.lane_scheduler.dataset,
                load_images=True,
            )
            batch = to_torch_batch(
                numpy_batch, pin_memory=self.context.device.type == "cuda"
            ).to(self.context.device, non_blocking=True)
            data_seconds = time.perf_counter() - data_started

            forward_started = time.perf_counter()
            with self._autocast():
                output = self.model(batch, state=self.model_state, return_state=True)
            forward_seconds = time.perf_counter() - forward_started
            self.model_state = output.state
            graph_loss = output.loss_sum if graph_loss is None else graph_loss + output.loss_sum
            target_weight_since_step += output.loss_weight.detach()
            language_loss_since_step += output.language_loss_sum.detach()
            image_loss_since_step += output.image_loss_sum.detach()

            active_chunks = sum(chunk is not None for chunk in chunks)
            consumed_atoms = int(batch.lengths.sum().item())
            valid_types = batch.atom_types[batch.valid_atom_mask]
            language_atoms = int((valid_types == 1).sum().item())
            image_atoms = int((valid_types == 2).sum().item())
            completed_records = sum(
                chunk is not None and chunk.end_of_record for chunk in chunks
            )
            self.lane_scheduler.commit(batch.lengths.tolist())
            self.progress.chunk_step += 1
            self.progress.consumed_chunks += active_chunks
            self.progress.consumed_atoms += consumed_atoms
            self.progress.consumed_language_tokens += language_atoms
            self.progress.consumed_images += image_atoms
            self.progress.consumed_records += completed_records
            language_atoms_since_step += language_atoms
            image_atoms_since_step += image_atoms
            records_since_step += completed_records
            reset_atoms_since_step += int(batch.reset_mask.sum().item())
            padded_slots_since_step += int(batch.padding_mask.sum().item())
            total_slots_since_step += batch.padding_mask.numel()
            data_seconds_since_step += data_seconds
            forward_seconds_since_step += forward_seconds
            chunks_since_backward += 1
            chunks_since_step += 1
            atoms_since_step += consumed_atoms

            backward_seconds = 0.0
            backward_due = chunks_since_backward >= self.tbptt.tbptt_chunks
            local_next_active = any(
                request is not None for request in self.lane_scheduler.peek()
            )
            data_exhausted = not self._global_any(local_next_active)
            self.lane_scheduler.rollback()
            if backward_due or data_exhausted:
                assert graph_loss is not None
                backward_started = time.perf_counter()
                graph_loss.backward()
                backward_seconds = time.perf_counter() - backward_started
                backward_seconds_since_step += backward_seconds
                # Truncate only autograd history. Tensor values and lane positions survive.
                self.model_state = detach_state(self.model_state)
                graph_loss = None
                chunks_since_backward = 0

            atoms_due = (
                self.tbptt.max_atoms_per_step is not None and
                atoms_since_step >= self.tbptt.max_atoms_per_step
            )
            optimizer_due = (
                chunks_since_step >= self.tbptt.optimizer_step_chunks or atoms_due or data_exhausted
            )
            if optimizer_due:
                # An atom cap may force an early boundary, but never cuts through a TBPTT graph.
                if graph_loss is not None:
                    continue
                optimizer_started = time.perf_counter()
                global_weight, grad_norm = self._normalize_and_step(target_weight_since_step)
                optimizer_seconds = time.perf_counter() - optimizer_started
                elapsed = time.perf_counter() - step_started
                if self.progress.optimizer_step and (
                    self.progress.optimizer_step % self.logging_config.every_optimizer_steps == 0
                ):
                    global_language_loss = self._global_sum(language_loss_since_step).item()
                    global_image_loss = self._global_sum(image_loss_since_step).item()
                    global_atoms = self._global_sum(torch.tensor(
                        atoms_since_step, device=self.context.device
                    )).item()
                    global_language_atoms = self._global_sum(torch.tensor(
                        language_atoms_since_step, device=self.context.device
                    )).item()
                    global_images = self._global_sum(torch.tensor(
                        image_atoms_since_step, device=self.context.device
                    )).item()
                    global_records = self._global_sum(torch.tensor(
                        records_since_step, device=self.context.device
                    )).item()
                    global_resets = self._global_sum(torch.tensor(
                        reset_atoms_since_step, device=self.context.device
                    )).item()
                    global_padded = self._global_sum(torch.tensor(
                        padded_slots_since_step, device=self.context.device
                    )).item()
                    global_slots = self._global_sum(torch.tensor(
                        total_slots_since_step, device=self.context.device
                    )).item()
                    gpu_allocated = 0
                    gpu_reserved = 0
                    gpu_peak = 0
                    if self.context.device.type == "cuda":
                        gpu_allocated = torch.cuda.memory_allocated(self.context.device)
                        gpu_reserved = torch.cuda.memory_reserved(self.context.device)
                        gpu_peak = torch.cuda.max_memory_allocated(self.context.device)
                    self.logger.log(self.progress.optimizer_step, {
                        "loss/global_language_sum": global_language_loss,
                        "loss/global_image_sum": global_image_loss,
                        "loss/global_weighted_mean": (
                            (global_language_loss + global_image_loss) / max(global_weight, 1.0)
                        ),
                        "targets/global_weight": global_weight,
                        "train/rank0_consumed_atoms_total": self.progress.consumed_atoms,
                        "train/rank0_consumed_chunks_total": self.progress.consumed_chunks,
                        "train/global_atoms_this_step": global_atoms,
                        "train/global_language_tokens_this_step": global_language_atoms,
                        "train/global_images_this_step": global_images,
                        "train/global_records_this_step": global_records,
                        "train/global_lane_resets_this_step": global_resets,
                        "throughput/global_atoms_per_second": global_atoms / max(elapsed, 1e-9),
                        "data/global_padding_ratio": global_padded / max(global_slots, 1.0),
                        "optimizer/global_grad_norm_before_clip": grad_norm,
                        "optimizer/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "time/rank0_data_seconds": data_seconds_since_step,
                        "time/rank0_forward_seconds": forward_seconds_since_step,
                        "time/rank0_backward_seconds": backward_seconds_since_step,
                        "time/rank0_optimizer_seconds": optimizer_seconds,
                        "memory/rank0_state_bytes": state_nbytes(self.model_state),
                        "memory/rank0_gpu_allocated_bytes": gpu_allocated,
                        "memory/rank0_gpu_reserved_bytes": gpu_reserved,
                        "memory/rank0_gpu_peak_allocated_bytes": gpu_peak,
                        "runtime/backbone": getattr(self.model.backbone, "architecture", "transformer_swa"),
                        "runtime/dtype": self.runtime.dtype,
                    })
                if self.progress.optimizer_step and (
                    self.progress.optimizer_step % self.checkpoint_config.every_optimizer_steps == 0
                ):
                    self._save()
                chunks_since_step = 0
                atoms_since_step = 0
                target_weight_since_step.zero_()
                language_loss_since_step.zero_()
                image_loss_since_step.zero_()
                language_atoms_since_step = 0
                image_atoms_since_step = 0
                records_since_step = 0
                reset_atoms_since_step = 0
                padded_slots_since_step = 0
                total_slots_since_step = 0
                data_seconds_since_step = 0.0
                forward_seconds_since_step = 0.0
                backward_seconds_since_step = 0.0
                step_started = time.perf_counter()
                if global_weight == 0 and data_exhausted:
                    break

        return self.progress

    def close(self) -> None:
        self.logger.close()

    def __enter__(self) -> "UnifiedTrainer":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
