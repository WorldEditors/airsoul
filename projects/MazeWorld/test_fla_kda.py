#!/usr/bin/env python
"""Minimal CUDA smoke test for AirSoul's FLA KDA adapter."""

import argparse
import platform
import sys

import torch

from airsoul.modules.kda import KDABlock


def assert_finite(value, name):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{name} contains NaN or Inf")
        return 1
    if isinstance(value, dict):
        return sum(assert_finite(item, f"{name}.{key}") for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return sum(assert_finite(item, f"{name}[{index}]") for index, item in enumerate(value))
    if value is None:
        return 0
    raise TypeError(f"Unexpected value in {name}: {type(value).__name__}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the FLA Triton KDA kernels")
    if args.hidden_size % args.num_heads:
        raise ValueError("--hidden-size must be divisible by --num-heads")

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"dtype={dtype}, shape=({args.batch_size}, {args.seq_len}, {args.hidden_size})")

    # Training path: chunk KDA forward and backward.
    train_model = KDABlock(
        io_size=args.hidden_size,
        num_heads=args.num_heads,
        expand_v=1.0,
        d_conv=4,
        is_generate=False,
    ).to(device=device, dtype=dtype).train()
    x = torch.randn(
        args.batch_size, args.seq_len, args.hidden_size,
        device=device, dtype=dtype, requires_grad=True,
    )
    output, _ = train_model(x, need_cache=False)
    assert output.shape == x.shape
    assert_finite(output, "training output")
    output.float().square().mean().backward()
    assert_finite(x.grad, "input gradient")
    parameter_gradients = [p.grad for p in train_model.parameters() if p.grad is not None]
    if not parameter_gradients:
        raise RuntimeError("KDA produced no parameter gradients")
    assert_finite(parameter_gradients, "parameter gradients")
    torch.cuda.synchronize()
    print("[PASS] chunk forward/backward")

    # Inference path: recurrent cache produced by one chunk and consumed by the next.
    inference_model = KDABlock(
        io_size=args.hidden_size,
        num_heads=args.num_heads,
        expand_v=1.0,
        d_conv=4,
        is_generate=True,
    ).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        first = torch.randn(args.batch_size, 16, args.hidden_size, device=device, dtype=dtype)
        first_output, cache = inference_model(first, need_cache=True)
        cache_tensor_count = assert_finite(cache, "first cache")
        if cache_tensor_count == 0:
            raise RuntimeError("KDA returned an empty recurrent cache")

        second = torch.randn(args.batch_size, 1, args.hidden_size, device=device, dtype=dtype)
        second_output, next_cache = inference_model(second, cache=cache, need_cache=True)
        assert first_output.shape == first.shape
        assert second_output.shape == second.shape
        assert_finite(first_output, "first inference output")
        assert_finite(second_output, "second inference output")
        assert_finite(next_cache, "next cache")
    torch.cuda.synchronize()
    print(f"[PASS] fused recurrent cache ({cache_tensor_count} tensors)")
    print("FLA KDA smoke test passed")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FLA KDA smoke test failed: {error}", file=sys.stderr)
        raise
