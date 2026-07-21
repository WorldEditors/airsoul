# AIRSoul

AIRSoul is a research codebase for long-context, in-context learning in
interactive and embodied environments. The project studies how one causal
model can learn from different interaction and supervision patterns without
gradient updates at inference time.

The repository is currently migrating from benchmark-specific sequence
layouts and training loops to a task-agnostic language/image sequence stack.
Both paths are present on `dev_test` while the migration is in progress.

## Current Status

### Unified sequence path

The new path represents persistent data using only two atom types:

- `LANGUAGE_TOKEN`
- `IMAGE`

Task semantics such as observation, action, reward, prompt, policy target, or
world-model target are not fixed data fields. Producers may express them with
ordinary language special tokens such as `<observation>` and `</observation>`.

The current implementation includes:

- immutable `tokens.bin` datasets with memory-mapped token, mask, target, and
  raw-image streams;
- a bounded-memory streaming writer for records containing at least 1B atoms;
- padding, attention, loss, memory-update, and per-lane reset masks;
- serializable lane schedulers with record-local `uint64` cursors;
- causal next-atom training and explicit language/image target mappings;
- continuous recurrent memory across chunks with configurable TBPTT;
- KDA, GDN, GDN-2 capability probing, and Transformer sliding-window
  attention with optional sink tokens;
- composable PyTorch FSDP2, activation checkpointing, distributed checkpoint,
  JSONL metrics, and optional TensorBoard output;
- exact same-topology resume of model, optimizer, RNG, lane cursor, and
  explicit per-lane memory.

See [the refactor requirements](./REFACTOR_REQUIREMENTS.md) for the complete
contract and [the unified training guide](./projects/UnifiedSequence/README.md)
for the new entry point.

### Legacy benchmark path

The existing YAML-based benchmark projects and their task-specific datasets
remain available during migration. They still use the legacy model, epoch, and
checkpoint abstractions. Gym integration has been migrated to Gymnasium, and
Mamba/Mamba2 are no longer supported.

The legacy and unified formats are not interchangeable. Existing data
generators under `data/` still produce their project-specific formats unless
they explicitly use `airsoul.dataio.UnifiedDatasetWriter`.

## Repository Layout

- [`airsoul/dataio`](./airsoul/dataio): unified V1 schema, mmap reader, atomic
  and streaming writers, validator, collator, and lane scheduler.
- [`airsoul/backbones`](./airsoul/backbones): explicit-state KDA, GDN, GDN-2,
  and Transformer-SWA adapters.
- [`airsoul/runtime`](./airsoul/runtime): FSDP2 setup, lane-aware TBPTT,
  structured logging, and distributed checkpoints.
- [`airsoul/models`](./airsoul/models): unified language/image model plus
  legacy benchmark models.
- [`airsoul/modules`](./airsoul/modules): legacy and shared neural-network
  building blocks.
- [`projects/UnifiedSequence`](./projects/UnifiedSequence): strict JSON config,
  `torchrun` training entry point, and backbone smoke test.
- [`projects/MazeWorld`](./projects/MazeWorld): MazeWorld training and
  validation.
- [`projects/OmniRL`](./projects/OmniRL) and
  [`projects/OmniRLQuad`](./projects/OmniRLQuad): AnyMDP/OmniRL variants.
- [`projects/MetaLM`](./projects/MetaLM),
  [`projects/MetaControl`](./projects/MetaControl),
  [`projects/MultiAgentHVAC`](./projects/MultiAgentHVAC): additional legacy
  research projects.
- [`data`](./data): current benchmark-specific data generation and evaluation
  scripts.
- [`tests`](./tests): unified data, state, TBPTT, and resume tests.

## Installation

Install the current checkout for development:

```bash
python -m pip install -e .
```

Install FLA backbones and TensorBoard support when needed:

```bash
python -m pip install -e ".[fla,logging]"
```

Core requirements currently include PyTorch 2.4 or newer, NumPy, Gymnasium,
and `restools`. KDA/GDN kernels require a compatible CUDA, Triton, and
`flash-linear-attention` installation. FLA is imported lazily, so the native
Transformer-SWA and data tools do not require it.

The new runtime uses public PyTorch FSDP2 and distributed-checkpoint APIs. It
does not require a TorchTitan checkout. Pin and validate PyTorch, CUDA, Triton,
and FLA together on the target training server before a large run.

## Unified Data

A V1 dataset has the following top-level shape:

```text
dataset_root/
  manifest.json
  COMMITTED
  shards/
    shard-000000/
      tokens.bin
      atom_types.bin
      records.idx
      targets.bin
      targets.idx
      loss_mask.bin
      memory_update_mask.bin
      reset_mask.bin
      images.bin
      images.idx
      COMMITTED
```

Validate a committed dataset before training:

```bash
python -m airsoul.dataio.validate /path/to/dataset
```

For normal records use `UnifiedDatasetWriter.append_record`. For records that
cannot be materialized in RAM, use `UnifiedDatasetWriter.stream_record` and
append producer-sized chunks. The writer publishes the dataset only after all
shards and the manifest are committed.

Data synthesis policy and the existing benchmark generators have not yet been
systematically migrated. New producers should write the unified atom streams
directly; legacy producers should continue using their existing readers until
their converter or native writer is available.

## Unified Training

Start from
[`projects/UnifiedSequence/config.example.json`](./projects/UnifiedSequence/config.example.json)
and point `data.dataset_root` to a committed V1 dataset.

Single GPU:

```bash
torchrun --standalone --nproc-per-node=1 \
  projects/UnifiedSequence/train.py projects/UnifiedSequence/config.example.json
```

Multiple GPUs:

```bash
torchrun --standalone --nproc-per-node=8 \
  projects/UnifiedSequence/train.py /path/to/resolved-config.json
```

Resume the latest committed checkpoint:

```bash
torchrun --standalone --nproc-per-node=8 \
  projects/UnifiedSequence/train.py /path/to/resolved-config.json --resume
```

Exact resume currently requires the same world size, dataset UUID, and
manifest hash. Checkpoints are written only after a flushed optimizer boundary.

Run the CUDA backbone smoke test before training:

```bash
CUDA_VISIBLE_DEVICES=0 python projects/UnifiedSequence/smoke_test.py \
  --backbones kda gdn transformer_swa
```

Test GDN-2 separately. The adapter reports a clear unsupported error when the
installed FLA release has no public GDN-2 layer.

## Legacy Training and Validation

Legacy projects continue to use their local YAML configuration and entry
points. For example:

```bash
cd projects/MazeWorld
CUDA_VISIBLE_DEVICES=0 python train.py config.yaml
CUDA_VISIBLE_DEVICES=0 python validate.py config.yaml
```

Most legacy runners also accept project-specific overrides through
`--configs`. Consult the project directory and its config before running it;
the exact fields differ by benchmark.

## Tests

Run the focused unified-path tests with:

```bash
python -m unittest tests.test_dataio tests.test_unified_model -v
```

The pure Data I/O tests require NumPy. Backbone, TBPTT, and checkpoint tests
require PyTorch; CUDA/FLA kernel coverage is provided by the server smoke test.

## Migration Boundaries

- The unified runtime is implemented but still requires CUDA/FSDP2 validation
  on each target server stack.
- Existing data synthesis logic is intentionally not rewritten in this phase.
- Legacy observation/action/reward fields remain only in legacy projects and
  offline migration code; the unified schema does not define them.
- GDN-2 support depends on a public API in the installed FLA release.
- World-size-changing checkpoint resume is not yet bitwise or exact.

## License

AIRSoul is distributed under the [Apache License 2.0](./LICENSE).
