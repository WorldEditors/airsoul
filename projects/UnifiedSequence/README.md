# Unified Sequence Training

This is the new task-agnostic training path. It consumes only language-token
and image atoms from an immutable AIRSoul V1 dataset. Legacy MazeWorld and
POTAR/PODAR entry points remain separate during migration.

Launch one GPU:

```bash
torchrun --standalone --nproc-per-node=1 \
  projects/UnifiedSequence/train.py projects/UnifiedSequence/config.example.json
```

Launch multiple GPUs:

```bash
torchrun --standalone --nproc-per-node=8 \
  projects/UnifiedSequence/train.py /path/to/resolved-config.json
```

Resume the latest committed checkpoint:

```bash
torchrun --standalone --nproc-per-node=8 \
  projects/UnifiedSequence/train.py /path/to/resolved-config.json --resume
```

The config is strict and versioned. `backbone.name` accepts `kda`, `gdn`,
`gdn2`, or `transformer_swa`. FLA is imported only for the first three. GDN-2
fails with an explicit capability error when the installed FLA release has no
public GDN-2 layer.

For records that cannot be materialized in memory, use the streaming writer:

```python
from airsoul.dataio import UnifiedDatasetWriter

with UnifiedDatasetWriter(output, tokenizer=tokenizer, special_tokens=special) as writer:
    with writer.stream_record(expected_atoms=1_000_000_000) as record:
        for chunk in producer:
            record.append(
                chunk.atom_values,
                chunk.atom_types,
                images=chunk.images,
                loss_mask=chunk.loss_mask,
                memory_update_mask=chunk.memory_update_mask,
                reset_mask=chunk.reset_mask,
                targets=chunk.targets,
            )
```

Image references supplied to each streamed append are local to that append's
`images` argument. Explicit target source positions are local to the append and
are converted to record positions by the writer.

Checkpoints are only emitted after backward and optimizer flush. Each committed
generation contains DCP model/optimizer shards plus rank-local RNG, lane cursor,
and explicit recurrent state. Exact resume currently requires the same world
size and the same dataset UUID and manifest hash.
