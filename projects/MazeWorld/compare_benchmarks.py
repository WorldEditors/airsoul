#!/usr/bin/env python3
"""Compare two AIRSoul benchmark JSON reports."""

import argparse
import json
from pathlib import Path


def load_report(path):
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    required = {"world_size", "samples_per_second", "tokens_per_second", "step_time_ms"}
    missing = required.difference(report)
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(sorted(missing))}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Print distributed-training speedup")
    parser.add_argument("baseline", type=Path, help="single-GPU JSON report")
    parser.add_argument("candidate", type=Path, help="multi-GPU JSON report")
    parser.add_argument("--output", type=Path, help="optional comparison JSON output")
    args = parser.parse_args()

    baseline = load_report(args.baseline)
    candidate = load_report(args.candidate)
    if candidate["world_size"] <= baseline["world_size"]:
        raise ValueError("candidate world_size must be larger than baseline world_size")
    for field in ("epoch_type", "local_batch_size", "sequence_length"):
        if baseline.get(field) != candidate.get(field):
            raise ValueError(
                f"reports are not comparable: {field} differs "
                f"({baseline.get(field)!r} != {candidate.get(field)!r})"
            )

    speedup = candidate["samples_per_second"] / baseline["samples_per_second"]
    scale = candidate["world_size"] / baseline["world_size"]
    result = {
        "baseline_world_size": baseline["world_size"],
        "candidate_world_size": candidate["world_size"],
        "baseline_samples_per_second": baseline["samples_per_second"],
        "candidate_samples_per_second": candidate["samples_per_second"],
        "speedup": speedup,
        "scaling_efficiency_percent": speedup / scale * 100.0,
        "candidate_step_time_ms": candidate["step_time_ms"],
    }

    print(
        f"[SPEEDUP] {baseline['world_size']} -> {candidate['world_size']} GPU(s): "
        f"{speedup:.3f}x, scaling efficiency={result['scaling_efficiency_percent']:.2f}%"
    )
    print(
        f"[THROUGHPUT] {baseline['samples_per_second']:.3f} -> "
        f"{candidate['samples_per_second']:.3f} samples/s; "
        f"{baseline['tokens_per_second']:.3f} -> "
        f"{candidate['tokens_per_second']:.3f} tokens/s"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
