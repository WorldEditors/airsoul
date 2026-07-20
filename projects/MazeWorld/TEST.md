# MazeWorld 云侧性能测试全流程

本文覆盖数据生产、单卡基线、4 机 32 卡 DDP 测试、加速比输出和结果检查。单卡与 32 卡必须使用相同代码、模型、序列长度、单卡 batch size、精度和计时步数；只有 GPU 数量不同。

## 1. 测试口径

基准测试默认针对 `MazeEpochCausal` 训练阶段，计入稳态数据读取、前向、反向、DDP 梯度通信、梯度裁剪和优化器更新，不计模型初始化、首批数据等待、预热、训练日志和 checkpoint 保存。每个 rank 的本地 batch size 为 1，序列长度为 1000，因此 32 卡全局 batch size 为 32。

输出指标如下：

- `samples_per_second`：整个集群每秒处理的轨迹样本数，是加速比的计算依据。
- `tokens_per_second`：`samples_per_second × sequence_length`，此处 token 指时间步。
- `step_time_ms`：一个训练 step 的集群墙钟时间。
- `peak_memory_bytes_per_gpu`：所有 GPU 中最大的 PyTorch 峰值已分配显存。
- `speedup`：32 卡吞吐 / 单卡吞吐。
- `scaling_efficiency_percent`：`speedup / 32 × 100%`。

## 2. 云资源和软件准备

4 台机器应使用相同 GPU 型号、驱动、CUDA、PyTorch 和代码版本，每台 8 卡；机器间建议使用 RDMA/IB。训练数据、代码和结果目录需要位于各节点相同的绝对路径。以下示例约定：

```bash
export REPO=/shared/airsoul
export TRAIN_DATA=/shared/data/mazeworld/train
export TEST_DATA=/shared/data/mazeworld/test
export BENCH_DIR=/shared/results/mazeworld
```

在每台机器执行：

```bash
cd "$REPO"
python -m pip install -e .
python -m pip install pyyaml opencv-python-headless matplotlib gymnasium
# 数据生成依赖 Xenoverse；按云环境的源码/镜像安装方式安装后检查：
python -c "import torch, xenoverse.mazeworld; print(torch.__version__, torch.cuda.device_count())"
mkdir -p "$TRAIN_DATA" "$TEST_DATA" "$BENCH_DIR"
```

确认以下条件：

```bash
nvidia-smi
ulimit -n
```

建议锁定 GPU 时钟/功耗策略，并确保测试期间没有其他 GPU 作业。若使用 NCCL，先用集群自带的 `nccl-tests` 验证 4 机网络；性能异常时设置 `NCCL_DEBUG=INFO` 排查网卡选择和 RDMA 回退。

## 3. 生产训练与测试数据

MazeWorld 数据应通过仓库的 `data/mazeworld/` 生产。若仓库检出目录名为 `airsoul`，其完整路径就是 `airsoul/data/mazeworld/`；它与训练入口 `airsoul/projects/MazeWorld/` 是两个不同目录。本文的 `$REPO` 指向仓库根目录，因此命令统一写成 `$REPO/data/mazeworld/...`。

该目录中的文件职责如下：

```text
airsoul/
├── data/mazeworld/
│   ├── gen_maze_task.py       # 随机生成 MazeWorld 任务定义，输出 .pkl
│   ├── gen_maze_record.py     # 在任务中运行行为/标签策略，输出训练轨迹
│   ├── maze_behavior_solver.py# 轨迹生产使用的带噪专家策略
│   └── dump_maze.sh           # 历史批处理示例，以 Python 脚本参数为准
└── projects/MazeWorld/
    ├── train.py               # 消费轨迹并训练/测速
    └── TEST.md
```

当前版本建议直接调用 `gen_maze_task.py` 和 `gen_maze_record.py`。`dump_maze.sh` 保留了旧版参数，仅作为历史示例，不应直接用于本次测试。

数据只需在一台能够写共享存储的机器上生产，4 个训练节点随后读取同一份数据。先生成可复现的任务集合：

```bash
cd "$REPO"
python data/mazeworld/gen_maze_task.py \
  --scale 15,16 --landmarks 6,10 --task_number 1024 \
  --output_path /shared/data/mazeworld/tasks.pkl
```

也可以先进入数据工具目录再运行；两种写法等价：

```bash
cd "$REPO/data/mazeworld"
python gen_maze_task.py \
  --scale 15,16 --landmarks 6,10 --task_number 1024 \
  --output_path /shared/data/mazeworld/tasks.pkl
cd "$REPO"
```

生产至少 1024 条训练轨迹。默认基准为 5 个预热 step 加 20 个计时 step，32 卡、单卡 batch size 1 至少需要 769 条轨迹；使用 1024 条可留出余量。

```bash
python data/mazeworld/gen_maze_record.py \
  --output_path "$TRAIN_DATA" \
  --task_source FILE \
  --task_file /shared/data/mazeworld/tasks.pkl \
  --max_steps 1200 --n_range 15,16 \
  --epochs 1024 --start_index 0 --workers 32
```

其中 `--task_source FILE` 会从 `tasks.pkl` 读取任务并重新采样起点与命令，适合让单卡和 32 卡使用完全相同的数据；`--task_source NEW` 会在每条轨迹生产时随机创建任务，适合生成独立测试集。`--epochs` 是轨迹条数，`--workers` 是并行的数据生产进程数，不是训练 GPU 数量。

再生成独立测试集：

```bash
python data/mazeworld/gen_maze_record.py \
  --output_path "$TEST_DATA" \
  --task_source NEW \
  --max_steps 1200 --n_range 15,16 \
  --epochs 64 --start_index 0 --workers 16
```

每条轨迹会形成一个独立目录，最终结构应类似：

```text
/shared/data/mazeworld/train/
├── record-000000/
│   ├── observations.npy
│   ├── commands.npy
│   ├── actions_behavior_id.npy
│   ├── actions_behavior_val.npy
│   ├── actions_behavior_prior.npy
│   ├── actions_label_id.npy
│   ├── actions_label_val.npy
│   ├── BEVs.npy
│   └── rewards.npy
├── record-000001/
└── ...
```

训练参数 `train_config.data_path` 必须指向包含这些 `record-*` 子目录的父目录，即示例中的 `/shared/data/mazeworld/train`，不能指向某一个 `.npy` 文件或单条 record 目录。

检查记录数量和单条记录的关键字段：

```bash
find "$TRAIN_DATA" -mindepth 1 -maxdepth 1 -type d | wc -l
python - <<'PY'
import os, numpy as np
root = os.environ["TRAIN_DATA"]
record = sorted(os.path.join(root, x) for x in os.listdir(root))[0]
for name in ("observations", "commands", "actions_behavior_id", "actions_label_id", "rewards"):
    value = np.load(os.path.join(record, name + ".npy"), mmap_mode="r")
    print(name, value.shape, value.dtype)
PY
```

`observations` 长度应比 action/reward 多 1。建议先确认大多数轨迹不少于 1000 步；若不足，应增大 `--max_steps` 或把两次性能测试的 `seq_len_causal` 同步调小。

## 4. 单卡冒烟与基线测试

先在任意一台机器执行 1 个预热、2 个计时 step 的冒烟测试：

```bash
cd "$REPO/projects/mazeworld"
CUDA_VISIBLE_DEVICES=0 python train.py config.yaml --configs \
  train_config.data_path="$TRAIN_DATA" \
  train_config.epoch_vae_stop=0 \
  train_config.epoch_causal_start=0 \
  train_config.benchmark.enabled=True \
  train_config.benchmark.warmup_steps=1 \
  train_config.benchmark.measure_steps=2 \
  train_config.benchmark.output="$BENCH_DIR/smoke.json"
```

冒烟通过后运行正式单卡基线：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py config.yaml --configs \
  run_name=mazeworld-1gpu \
  train_config.data_path="$TRAIN_DATA" \
  train_config.epoch_vae_stop=0 \
  train_config.epoch_causal_start=0 \
  train_config.benchmark.enabled=True \
  train_config.benchmark.warmup_steps=5 \
  train_config.benchmark.measure_steps=20 \
  train_config.benchmark.output="$BENCH_DIR/1gpu.json"
```

结束时终端会打印类似：

```text
[BENCHMARK] 1 GPU(s), 0.123 samples/s, 123.000 tokens/s, 8123.000 ms/step
[BENCHMARK] report: /shared/results/mazeworld/1gpu.json
```

正式记录建议连续运行 3 次，以 `samples_per_second` 中位数作为基线；每次需使用不同的输出文件名。

## 5. 4 机 32 卡测试

选定主节点可达 IP，例如 `10.0.0.10`，并为四台机器分别设置 `NODE_RANK=0,1,2,3`。在所有节点执行相同命令；先启动 rank 0，再在 rendezvous 超时前启动其余节点。

```bash
cd "$REPO/projects/mazeworld"
export NNODES=4
export NODE_RANK=0                 # 其余三台依次改为 1、2、3
export MASTER_ADDR=10.0.0.10
export MASTER_PORT=12402
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun \
  --nnodes="$NNODES" \
  --nproc-per-node=8 \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  train.py config.yaml --configs \
    run_name=mazeworld-32gpu \
    train_config.data_path="$TRAIN_DATA" \
    train_config.epoch_vae_stop=0 \
    train_config.epoch_causal_start=0 \
    train_config.benchmark.enabled=True \
    train_config.benchmark.warmup_steps=5 \
    train_config.benchmark.measure_steps=20 \
    train_config.benchmark.output="$BENCH_DIR/32gpu.json" \
    train_config.benchmark.baseline_report="$BENCH_DIR/1gpu.json"
```

rank 0 会直接打印 32 卡吞吐、相对单卡的加速比和并行效率，例如：

```text
[BENCHMARK] 32 GPU(s), 3.200 samples/s, 3200.000 tokens/s, 10000.000 ms/step, speedup=26.016x, scaling_efficiency=81.30%
```

也可以在任务完成后独立复算并保存比较结果：

```bash
python compare_benchmarks.py \
  "$BENCH_DIR/1gpu.json" "$BENCH_DIR/32gpu.json" \
  --output "$BENCH_DIR/comparison.json"
```

## 6. 功能测试与结果验收

性能测试完成后，可使用独立测试集做 VAE 静态验证。先把 `config.yaml` 中 `test_config.datasets[0].data_path` 改为测试集路径，并把 `load_model_path` 指向训练产生的 checkpoint，再执行：

```bash
CUDA_VISIBLE_DEVICES=0 python validate.py config.yaml
```

基准验收至少检查：

- `1gpu.json` 的 `world_size` 为 1，`32gpu.json` 为 32。
- 两份报告的 `local_batch_size`、`sequence_length`、`warmup_steps`、`measured_steps` 一致。
- 32 卡运行没有 NCCL timeout、OOM、NaN 或数据读取超时。
- 三次重复测试的吞吐波动建议不超过 5%；超出时先排查共享存储、GPU 降频和其他租户负载。
- 结合目标设置加速比门槛；例如要求 `speedup >= 24` 等价于 32 卡并行效率至少 75%。

## 7. 常见问题

- 报告 “dataloader only has ...”：32 卡每个 step 消耗 32 条不同轨迹，增加训练记录，或同时降低两组测试的预热/计时 step。
- 多机启动后挂住：检查四机 `MASTER_ADDR/PORT`、`NNODES`、`NODE_RANK` 是否一致且 rank 唯一，并检查安全组和防火墙。
- 32 卡反而很慢：确认 `train_config.manual_sync=False`。DDP 已在 backward 中同步梯度，重复手工 all-reduce 会增加通信。
- OOM：先降低 `seq_len_causal`/`seg_len_causal`，且单卡与 32 卡必须使用同一值；不要只修改其中一组。
- 结果不可比：不要在 32 卡上增大单卡 batch size、打开 AMP 或更换模型配置，除非单卡基线也做同样修改。
