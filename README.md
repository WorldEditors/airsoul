# RoboFM

RoboFM is a task-agnostic language/image sequence framework for long-context,
in-context learning in interactive and embodied environments.

The repository has one runtime and one persistent data contract. Benchmark
semantics such as observation, action, reward, prompt, or policy target are
represented by special-token-wrapped spans rather than fixed Dataset fields.

## Architecture

- `robofm/dataio`: V1 writer, mmap reader, collection reader, validator,
  collator, lane scheduler, and benchmark producer adapter.
- `robofm/backbones`: explicit-state KDA, GDN, GDN-2, and Transformer-SWA.
- `robofm/models`: the unified language/image causal model.
- `robofm/runtime`: FSDP2, lane-aware TBPTT, logging, and checkpoints.
- `projects/UnifiedSequence`: the only training and smoke-test entry point.
- `data`: producer-only environment logic. It may sample tasks or train an
  offline coach, but it never owns a model-training loop for RoboFM.

The former `.npy` dataloaders, PODAR/POTAR models, project-specific epoch
managers, and legacy benchmark runners have been removed.

## Installation

```bash
python -m pip install -e .
```

Optional FLA backbones and TensorBoard support:

```bash
python -m pip install -e ".[fla,logging]"
```

Common Gymnasium producer dependencies:

```bash
python -m pip install -e ".[data]"
```

Core RoboFM requires NumPy and PyTorch 2.4 or newer. Data producers additionally
require their environment packages, including Gymnasium, Xenoverse, and the
policy packages imported by the selected producer.

## Unified Data

A V1 dataset contains `manifest.json`, `COMMITTED`, and immutable shard files
for atom values/types, masks, targets, and raw images. Environment-specific
producers use `robofm.dataio.write_unified_record`; they never publish `.npy`
trajectory directories.

Multi-process producers create one committed V1 dataset per trajectory:

```text
dataset_root/
  record-000000/
    manifest.json
    COMMITTED
    shards/...
  record-000001/
    manifest.json
    COMMITTED
    shards/...
```

`open_unified_dataset` and the training entry point treat this layout as one
record collection. A single committed V1 dataset root remains supported.

Validate either layout before training:

```bash
python -m robofm.dataio.validate /path/to/dataset_root
```

For custom producers:

```python
from robofm.dataio import write_unified_record

write_unified_record(
    "dataset/record-000000",
    {"observations": observations, "actions": actions, "rewards": rewards},
    producer={"name": "my-environment", "version": "1"},
)
```

Image-like fields become `IMAGE` atoms. Other values are encoded as deterministic
byte-tokenized language spans. The producer protocol follows standard message
roles (`system`, `user`, `assistant`, `tool`) and supports VLM content parts
(`text`, `image`) plus OpenAI-style `tool_calls` / `function` arguments. The
training model only consumes the resulting language/image atom stream and does
not branch on MazeWorld, AnyMDP, Gym, or MetaLang.

For a custom VLM or function-calling producer:

```python
from robofm.dataio import write_unified_messages

write_unified_messages(
    "dataset/record-000000",
    [
        {"role": "user", "content": [
            {"type": "text", "text": "inspect"},
            {"type": "image", "image": frame},
        ]},
        {"role": "assistant", "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "set_action", "arguments": {"action": 2}},
        }]},
    ],
    producer={"name": "my-producer", "protocol": "messages-v1"},
)
```

## Training

Set `data.dataset_root` in
`projects/UnifiedSequence/config.example.json` to a committed dataset or record
collection.

```bash
torchrun --standalone --nproc-per-node=1 \
  projects/UnifiedSequence/train.py projects/UnifiedSequence/config.example.json
```

Resume the latest checkpoint:

```bash
torchrun --standalone --nproc-per-node=1 \
  projects/UnifiedSequence/train.py /path/to/config.json --resume
```

Exact resume requires the same world size and the same dataset collection
identity. See `projects/UnifiedSequence/README.md` for runtime details.

## Tests

```bash
python -m unittest tests.test_dataio tests.test_unified_model -v
```

CUDA/FLA kernels should also be checked on the target server:

```bash
python projects/UnifiedSequence/smoke_test.py --backbones kda gdn transformer_swa
```

## License

RoboFM is distributed under the [Apache License 2.0](./LICENSE).
