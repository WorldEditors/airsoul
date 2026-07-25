# RoboFM Data Producers

This directory contains producer-only environment logic. MazeWorld, AnyMDP,
Gymnasium, MetaControl, and MetaLang do not have independent RoboFM training
entries here. They only sample tasks, optionally prepare offline coach/policy
artifacts, and emit standard VLM/function-call messages through the RoboFM V1
data contract.

Each worker writes an independent committed dataset under
`OUTPUT_PATH/record-NNNNNN`. The parent output directory can be passed directly
to `projects/UnifiedSequence/train.py` or `python -m robofm.dataio.validate`.

Available producers:

- `anymdp/gen_anymdp_record.py`
- `anymdp/gen_gym_record.py`
- `anymdpv2/gen_anymdp_record_v2.py`
- `mazeworld/gen_maze_record.py`
- `metacontrol/gen_cartpole.py`
- `metalang/gen_metalang.py` and `gen_metalang_v3.py`

Task definitions and trained coach files may remain pickle artifacts because
they configure sampling rather than store training trajectories. MetaLang uses
a private temporary NumPy file only to bridge Xenoverse's generator API; that
file is removed immediately and is never a published dataset.

All training, validation, checkpoint, and resume behavior lives under
`projects/UnifiedSequence`. The producer directories must not import the
RoboFM trainer or define task-specific model losses.

Run a producer with `--help` for its environment-specific dependencies and
arguments. Existing output directories are rejected to preserve immutable,
atomic dataset publication.
