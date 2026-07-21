"""CUDA smoke tests for unified backbones and explicit state continuity."""

from __future__ import annotations

import argparse
import sys

import torch

from airsoul.backbones import BackboneConfig, build_backbone, detach_state


def run_backbone(name: str, hidden_size: int, num_heads: int,
                 sequence_length: int, chunk_length: int) -> None:
    config = BackboneConfig(
        name=name,
        hidden_size=hidden_size,
        num_layers=2,
        num_heads=num_heads,
        intermediate_size=hidden_size * 4,
        window_size=max(chunk_length * 2, 16),
        mode="chunk",
    )
    model = build_backbone(config).cuda().to(torch.bfloat16).train()
    inputs = torch.randn(
        1, sequence_length, hidden_size, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    positions = torch.arange(sequence_length, device="cuda").unsqueeze(0)
    mask = torch.ones(1, sequence_length, dtype=torch.bool, device="cuda")
    state = None
    outputs = []
    for start in range(0, sequence_length, chunk_length):
        end = min(start + chunk_length, sequence_length)
        result = model.forward_chunk(
            inputs[:, start:end],
            state=state,
            attention_mask=mask[:, start:end],
            memory_update_mask=mask[:, start:end],
            reset_mask=torch.zeros_like(mask[:, start:end]),
            position_ids=positions[:, start:end],
        )
        outputs.append(result.hidden_states)
        state = result.state
    output = torch.cat(outputs, dim=1)
    output.float().square().mean().backward()
    if not torch.isfinite(output).all() or inputs.grad is None:
        raise RuntimeError(f"{name} produced invalid output or no input gradient")
    detached = detach_state(state)
    if any(parameter.grad is None for parameter in model.parameters() if parameter.requires_grad):
        raise RuntimeError(f"{name} produced missing parameter gradients")
    print(f"[PASS] {name}: output={tuple(output.shape)}, state={type(detached).__name__}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbones", nargs="+", default=["kda", "gdn", "gdn2", "transformer_swa"])
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--chunk-length", type=int, default=32)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    for name in args.backbones:
        run_backbone(
            name, args.hidden_size, args.num_heads,
            args.sequence_length, args.chunk_length,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
