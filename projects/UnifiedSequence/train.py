"""torchrun entry point for RoboFM unified language/image sequence training."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from robofm.dataio import LaneScheduler, open_unified_dataset
from robofm.models.unified_sequence import UnifiedSequenceModel
from robofm.runtime import UnifiedTrainer, apply_fsdp2, init_distributed
from robofm.runtime.experiment import ExperimentConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="versioned JSON experiment config")
    parser.add_argument("--resume", nargs="?", const="latest", default=None)
    args = parser.parse_args(argv)
    config = ExperimentConfig.from_json(args.config)

    context = init_distributed()
    dataset = None
    trainer = None
    try:
        seed = config.runtime.seed + context.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        dataset = open_unified_dataset(config.data.dataset_root)
        lanes = LaneScheduler(
            dataset,
            lane_count=config.tbptt.lane_count,
            chunk_length=config.tbptt.chunk_length,
            rank=context.rank,
            world_size=context.world_size,
            seed=config.data.seed,
            shuffle=config.data.shuffle,
            repeat=config.data.repeat,
        )
        model = UnifiedSequenceModel(config.model)
        model.backbone.set_activation_checkpointing(
            config.runtime.activation_checkpointing
        )
        dtype = torch.bfloat16 if config.runtime.dtype == "bfloat16" else torch.float32
        if config.runtime.fsdp2:
            if (config.tbptt.tbptt_chunks > 1 and
                    config.runtime.fsdp_reshard_after_forward):
                raise ValueError(
                    "TBPTT with multiple forwards before backward requires "
                    "fsdp_reshard_after_forward=false"
                )
            model = apply_fsdp2(
                model, context, dtype=dtype,
                reshard_after_forward=config.runtime.fsdp_reshard_after_forward,
            )
        elif context.world_size > 1:
            raise ValueError("multi-rank training requires fsdp2=true")
        else:
            model = model.to(context.device)
        if config.runtime.compile_model:
            if not model.backbone.capabilities.supports_compile:
                raise ValueError("the selected backbone does not support torch.compile")
            model.compile()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )
        trainer = UnifiedTrainer(
            model=model,
            optimizer=optimizer,
            lane_scheduler=lanes,
            context=context,
            tbptt=config.tbptt,
            runtime=config.runtime,
            logging=config.logging,
            checkpoint=config.checkpoint,
            pad_token_id=config.tokenizer.pad_token_id,
        )
        if args.resume is not None:
            trainer.resume(None if args.resume == "latest" else args.resume)
        trainer.run(max_optimizer_steps=config.max_optimizer_steps)
        return 0
    finally:
        if trainer is not None:
            trainer.close()
        if dataset is not None:
            dataset.close()
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
