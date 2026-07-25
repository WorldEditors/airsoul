# RoboFM 统一序列与训练底座重构需求

状态：统一架构已落地，旧训练与 `.npy` 数据路径已移除；任务 producer 已统一消息协议
适用分支：`dev_test` 及后续重构分支
最后更新：2026-07-22

## 1. 背景

重构前的系统将 observation、prompt、tag、behavior action、label action、reward 等定义为固定字段，并通过 Dataset tuple、`PODAR/POTAR` 字符串和项目专用 loss 约定组合顺序。这导致数据格式、任务语义、模型输入和训练代码相互耦合。

新架构不再在底层定义 observation、action、reward、policy、world model 等领域对象。所有数据统一表示为由两类 atom 构成的序列：

1. `LANGUAGE_TOKEN`；
2. `IMAGE`。

领域语义由 language special token 表达。例如 `<observation>`、`</observation>`、`<action>`、`</action>` 都只是词表中的 token。数据 I/O、Dataset、主干模型和训练循环不解释这些标签的业务含义。

producer-facing protocol follows standard multimodal/function-calling messages:
`system`, `user`, `assistant`, and `tool` roles; `text` and `image` content
parts; and OpenAI-compatible `tool_calls` with function name and arguments.
The protocol is encoded into the same language/image atom stream, so the
unified trainer does not contain task-specific branches.

## 2. 本阶段目标

1. 建立以 `tokens.bin` 为核心的统一 mmap 序列格式；
2. 数据底层只保留 language token 和 image 两种 atom；
3. 支持 special token 包裹的任意结构，不再使用固定 PODAR/POTAR 顺序；
4. 支持输入输出长度不同、稀疏监督、任意 span 监督和 image/token 混合目标；
5. 支持至少 1B atom 的单条超长序列；
6. 支持 padding、attention mask、loss mask、memory update mask 和 reset mask；
7. 支持 KDA、GDN、GDN-2、Transformer-SWA 四类主干；
8. 建立 FSDP2、lane-aware TBPTT 和连续 memory 驱动的训练循环；
9. 建立包含 per-lane memory 与数据 cursor 的精确断点续训；
10. 建立结构化训练日志、性能指标、错误诊断和 profiling；
11. 尽可能复用 PyTorch 和 TorchTitan 的公开工具。

## 3. 本阶段非目标

- 不重新设计 MazeWorld、AnyMDP 等数据生成算法和采样策略；
- 不在数据底层建立 observation/action/reward 等固定 schema；
- 不要求本阶段迁移全部历史数据生产脚本；
- 不规定 `<observation>` 等领域 special token 的最终词表；
- 不进行大规模效果调参；
- 不建设在线推理或服务化框架。

本阶段提供统一 writer、reader、converter 和训练接口。数据合成逻辑将在接口稳定后单独迁移。

## 4. 核心原则

- **两类 atom**：数据核心只认识 language token 和 image。
- **语义 token 化**：任务语义由 special token 和序列结构表达。
- **序列即协议**：数据排列由记录内容决定，不由模型配置字符串决定。
- **目标显式化**：监督位置由 target alignment 和 loss mask 表达。
- **状态显式化**：memory/state 作为模型输入输出，不保存在隐式全局变量中。
- **memory 连续、梯度截断**：超长序列前向状态连续，反向图只覆盖配置窗口。
- **mmap 优先**：读取任意 chunk 不加载完整序列。
- **可恢复优先**：数据 cursor、memory、RNG 和梯度累积状态均属于 checkpoint。
- **公共 API 优先**：优先复用 PyTorch/TorchTitan 公开能力。
- **失败显式化**：损坏数据和 worker 错误必须快速失败。

## 5. 统一 Atom 模型

### 5.1 Atom 类型

底层只定义：

```text
LANGUAGE_TOKEN(token_id)
IMAGE(image_payload_id)
```

不得在核心数据格式中增加以下固定类型：

- observation；
- action；
- reward；
- prompt；
- policy label；
- world-model label；
- state；
- agent event。

上述概念如有需要，均通过 language special token 包裹对应 span。

### 5.2 示例

```text
<bos>
<observation>
<image> IMAGE(1042) </image>
<text> turn left at the red door </text>
</observation>
<action> turn_left </action>
<observation>
<image> IMAGE(1043) </image>
</observation>
<eos>
```

对数据层而言，`<observation>`、`turn_left`、`<action>` 都只是 language token ID。只有 `IMAGE(1042)` 和 `IMAGE(1043)` 使用 image payload。

另一个 producer 可以采用完全不同的结构：

```text
<input> ... </input>
<prediction> ... </prediction>
```

新 Dataset 和主干不需要为此增加字段或分支。

### 5.3 结构性 special token

词表可以定义任意 special token。训练运行时只允许少量与执行有关的结构 token 被配置识别：

- padding token；
- begin/end-of-record token；
- 可选 memory-reset token；
- image placeholder/open/close token。

`<observation>`、`<action>`、`<reward>` 等均不属于运行时强制结构 token，除非具体训练配置显式赋予其 mask 或 reset 行为。

## 6. Data I/O V1

### 6.1 目录布局

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
```

### 6.2 tokens.bin

- `tokens.bin` 是 canonical atom stream。
- 默认使用 little-endian `uint64`。
- 对 `LANGUAGE_TOKEN`，value 表示词表 token ID。
- 对 `IMAGE`，value 表示 shard 内 image payload ID。
- `atom_types.bin` 使用紧凑整数或 bit-packed enum，区分 language/image。
- 允许后续将 type 和 value bit-pack 到同一 `uint64`，但 V1 reader API 不暴露物理编码差异。
- 单条 record 和整个 dataset 均不得受 32-bit offset 限制。

### 6.3 Image payload

- `images.bin` 保存原始或编码后的 image payload。
- `images.idx` 保存 `uint64` offset、length、shape、dtype/codec ID。
- manifest 声明支持的 image codec、颜色空间、布局和归一化方式。
- 默认支持 mmap 读取定长 raw image。
- 变长压缩图片允许按需解码，但必须显式记录 copy/decode 成本。
- image 可以在模型侧编码为一个或多个 hidden token。
- 若 producer 使用 VQ/image tokenizer，也可以把图像直接写成 language token 序列，此时不需要 `IMAGE` atom。

### 6.4 records.idx

每条记录至少包含：

- dataset/shard UUID；
- atom offset；
- atom length；
- target offset/count；
- loss-mask offset；
- flags；
- checksum 或完整性信息。

所有 offset、length、record ID、chunk ID 和 consumed atom 计数使用至少 `uint64` 语义。

### 6.5 Manifest

manifest 必须包含：

- 格式版本和端序；
- tokenizer/vocabulary ID 与 hash；
- special token 表；
- image codec 定义；
- shard 列表、记录数、atom 数和 image 数；
- producer 名称与版本；
- 数据集 UUID；
- checksum/commit generation；
- 可选的默认 target 和 mask 规则。

manifest 不得定义 observation/action/reward 等固定字段。

### 6.6 原子写入

- writer 先写临时 shard；
- 完成所有 bin/index/checksum 后再原子提交；
- 未出现 `COMMITTED` 的 shard 不得被 reader 加载；
- 多进程 producer 必须各写独立临时 shard，禁止并发 append 同一 mmap 文件；
- finalize 工具负责排序、校验和生成 manifest。

## 7. 输入、输出与监督

### 7.1 默认因果训练

默认情况下，目标是预测下一个 atom：

```text
input  = atoms[0:N-1]
target = atoms[1:N]
```

language target 由 language decoder 处理，image target 由 image prediction/reconstruction head 处理。

### 7.2 Loss mask

- `loss_mask.bin` 与 atom stream 对齐，决定哪些 target 参与训练。
- prompt/input span 可以保留在上下文中但 mask 为 0。
- output span 的 mask 可以为 1。
- padding、无效 target 和不训练的 image span 必须 mask 为 0。
- loss normalization 只统计有效 target 数，不统计 padding 或 mask=0 的位置。
- 可选 weight stream 可以为每个 target 指定浮点权重。

例如：

```text
<input> ... </input> <output> ... </output>
mask:  0 0 0 0 0 0 0 0 1 1 1 1
```

### 7.3 输入输出不一致

必须支持：

- 输入 span 与输出 span 长度不同；
- 输入中包含 image，输出只有 language；
- 输入只有 language，输出为 image；
- 输出只监督部分 token；
- 一个输入对应多个输出 span；
- 跨多个 chunk 的输出；
- 输入输出由不同数据过程产生。

默认使用单一 causal stream + loss mask 表达。对无法通过 shift 对齐的情况，使用统一 target mapping：

```text
TargetEntry(
    source_position,
    target_atom_type,
    target_value_or_payload_ref,
    weight,
    valid,
)
```

TargetEntry 只区分 language/image，不包含 policy/world-model/reward 等 objective 类型。

### 7.4 任务表达

策略、世界模型、模仿学习、自监督等不再是数据格式中的固定任务类型。它们由 producer 写出的序列和 mask 决定。

例如策略数据可以写成：

```text
<history> ... </history> <next_action> ... </next_action>
```

世界预测数据可以写成：

```text
<history> ... </history> <next_image> IMAGE(...) </next_image>
```

训练核心只执行 language/image prediction，不理解 `next_action` 或 `next_image` 的业务语义。

## 8. Padding 与 Mask

Batch 必须提供：

- `padding_mask`；
- `valid_atom_mask`；
- `attention_mask`；
- `loss_mask`；
- `memory_update_mask`；
- `reset_mask`；
- 每条 lane 的真实长度。

要求：

- padding 不推进 lane cursor、逻辑 position、memory 或 consumed-atom 统计；
- padding 不参与 attention、target 或 loss normalization；
- `loss_mask=0` 的有效输入仍可更新 memory；
- `memory_update_mask=0` 的 atom 不得改变 recurrent/KV state；
- `reset_mask` 只重置对应 lane；
- target mask、attention mask、padding mask 和 memory mask 不得混用；
- collator 可以将不同长度 chunk pad 到统一长度；
- 支持 partial final chunk 和全 padding lane；
- 全 padding batch 不得执行无意义 optimizer step。

Transformer-SWA 必须组合 causal、sliding-window 和 padding mask，且不构造与 1B 序列长度相关的 dense mask。KDA/GDN/GDN-2 必须按 lane 应用 memory update/reset mask。

## 9. mmap Dataset 与 Lane 调度

### 9.1 Dataset API

Dataset 统一返回：

```python
UnifiedSequenceBatch(
    atom_types=...,
    atom_values=...,
    images=...,
    image_refs=...,
    positions=...,
    targets=...,
    padding_mask=...,
    attention_mask=...,
    loss_mask=...,
    memory_update_mask=...,
    reset_mask=...,
    cursor=...,
)
```

Dataset 不返回 observation/action/reward 等具名字段。

### 9.2 mmap 要求

- 读取一个 chunk 不加载完整 record 或 shard；
- worker 初始化时打开 mmap，样本读取不得重复打开文件；
- 支持多个 shard 和多个数据根目录；
- 支持按 record/chunk 随机 seek；
- 支持 map-style 验证和 streaming/lane-style 训练；
- 提供 validator 检查 offset、边界、atom type、image ref、target 和 checksum；
- 数据损坏必须快速失败，不得无限跳过。

### 9.3 Lane 调度

- 每个 rank 维护固定数量 local lanes；
- 每条 lane 持有 dataset UUID、shard ID、record ID、atom offset、chunk index 和 permutation generation；
- 一个 lane 在同一 record 内持续读取后续 chunk；
- record 结束后只 refill/reset 对应 lane；
- lane scheduler 可序列化和恢复；
- rank 之间不得重复消费数据，除非显式配置 replacement；
- 1B atom record 必须可从任意 checkpoint cursor 直接恢复，不重放前缀。

## 10. 超长序列与 TBPTT

### 10.1 长度要求

- 支持至少 1B atom 的单条 record；
- 记录长度、offset、chunk count 使用 `uint64`；
- 序列不得整体 materialize 到 RAM/GPU；
- Dataset、logger 和 checkpoint 不得使用 32-bit step/chunk 计数；
- 允许一条序列跨越进程生命周期和多次 checkpoint。

### 10.2 Memory continuity

必须严格区分：

- **前向 memory continuity**：同一序列内始终连续；
- **反向 gradient continuity**：只覆盖配置的 TBPTT 窗口。

同一序列只有在 record 结束或显式 memory-reset token 生效时才能清空 memory。optimizer update、日志、checkpoint 和 chunk 边界均不得隐式 reset memory。

### 10.3 周期反向

配置至少包含：

- `chunk_length`；
- `tbptt_chunks` 或 `tbptt_tokens`；
- `optimizer_step_chunks`；
- `lane_count`；
- `max_atoms_per_step`。

标准行为：

```text
1B atoms / 1K atoms per chunk = 1M chunks
tbptt_chunks = 10
optimizer_step_chunks = 10

chunk 0000001 ... 0000010:
  memory 连续
  保留最近 10 chunks 的计算图
  累积有效 loss

backward
optimizer step
detach memory，但保留 memory 数值

chunk 0000011 ... 0000020:
  从上一个 detached memory 继续
  建立新的计算图

...

chunk 1000000:
  完成 record 后才 reset
```

要求：

- backward 后 detach/clone state，不清空 state；
- detach 不改变 state 数值和 logical position；
- 峰值显存近似 `O(tbptt_chunks * chunk_length + state_size)`；
- 显存不得随 1B 总长度线性增长；
- 不允许每个 chunk 调用 `model.reset()`；
- loss 按 accumulation 窗口内有效 language/image targets 归一化；
- optimizer update 间隔可以与 TBPTT detach 间隔不同，但语义必须显式定义。

## 11. 统一模型结构

### 11.1 模型顶层

新模型只包含：

1. language token embedding；
2. image encoder/image tokenizer adapter；
3. atom/type/position embedding；
4. 统一 causal backbone；
5. language output head；
6. image output/reconstruction head；
7. 可选 image-to-token 或 token-to-image adapter。

核心模型不包含固定 policy head、reward head 或 world-model head。若后续需要专用 head，必须作为可选扩展，不能改变 Data I/O 基础类型。

### 11.2 Backbone API

```python
BackboneOutput = backbone.forward_chunk(
    hidden_states,
    state=state,
    attention_mask=attention_mask,
    memory_update_mask=memory_update_mask,
    reset_mask=reset_mask,
    position_ids=position_ids,
    return_state=True,
)
```

要求：

- state tree 支持 detach、CPU offload、序列化、lane select/scatter/reset；
- backbone 不依赖 batch 间内部可变 memory；
- 完整序列与等价 chunk forward 在容差内一致；
- state API 不暴露 Hugging Face/FLA 私有 Cache 类型；
- 提供 FSDP2 和 activation-checkpoint capability metadata。

### 11.3 KDA

- 基于 FLA `KimiDeltaAttention`；
- 训练使用 chunk kernel，递归推理使用 fused recurrent；
- 使用 RoboFM-managed state；
- 支持 BF16、conv/recurrent state、per-lane mask/reset；
- 默认使用 Triton，不因可选 TileLang 损坏而阻塞。

### 11.4 GDN

- 基于 FLA 官方 Gated DeltaNet；
- 使用统一显式 state adapter；
- 支持 head、head dim、value expansion、短卷积和训练/inference kernel 配置；
- 不使用旧 `fla.models.utils.Cache` wrapper。

### 11.5 GDN-2

- 使用 FLA 对应的公开第二代实现；
- 阶段 A 固定支持版本和公开 API；
- 若稳定版没有公开 API，明确报告 unsupported，不依赖私有模块临时实现；
- state/mask/FSDP2 语义与 KDA/GDN 一致。

### 11.6 Transformer-SWA

- 使用 PyTorch 原生 SDPA/FlexAttention 公开 API；
- 支持 causal sliding-window attention；
- 支持 padding mask、memory update mask 和 bounded KV state；
- window size、sink/global token 和 position encoding 显式配置；
- 不构造全序列 dense mask；
- 支持 FSDP2、activation checkpoint 和 capability-gated `torch.compile`。

## 12. FSDP2 与训练运行时

### 12.1 TorchTitan 复用

实施前完成 TorchTitan capability spike，验证：

- 自定义 image/language batch；
- stateful lane dataloader；
- FSDP2 fully-shard plan；
- distributed checkpoint extra state；
- metrics、profiling 和 fault-tolerance 扩展点；
- 当前 Python/PyTorch/CUDA/FLA 组合。

原则：

- 优先使用 TorchTitan/PyTorch 公开 FSDP2、DCP、DeviceMesh、metrics 和 profiler；
- TorchTitan 依赖集中在单独 runtime adapter；
- Dataset、模型和项目代码不直接导入其内部模块；
- 固定验证过的 release/commit；
- 若公开扩展点不能保存 lane memory/cursor，使用 PyTorch 原生薄训练循环，不 fork 私有 trainer。

### 12.2 FSDP2

- 使用 composable FSDP2 `fully_shard`；
- 单卡和多卡使用同一循环；
- 支持 DeviceMesh、BF16、梯度裁剪、activation checkpoint 和梯度累积；
- 项目代码不得访问 `model.module`；
- 使用 `torchrun`/TorchTitan launcher，不嵌套 `mp.spawn`；
- process group 必须在 `finally` 中销毁。

### 12.3 通用训练循环

训练循环负责：

- 获取每条 lane 的下一个 chunk；
- padding/collation；
- reset 和 memory update mask；
- 显式传入和收回 per-lane state；
- language/image target routing；
- loss mask 和有效 target 归一化；
- TBPTT graph 管理；
- backward、clip、optimizer 和 scheduler；
- state detach 但保持 memory continuity；
- cursor、consumed atoms 和日志更新；
- 一致 checkpoint。

项目代码不再实现独立 `Epoch.compute(*tuple)` 或 `segment_iterator()`。

## 13. 断点续训

checkpoint 必须包含：

- FSDP2 sharded model；
- optimizer、scheduler、scaler；
- global step、optimizer step、consumed atoms；
- Python/NumPy/Torch/CUDA RNG；
- sampler/permutation state；
- 每个 rank 的 lane cursor；
- 每个 lane 的 explicit memory state；
- record 内 atom offset 和 chunk index；
- accumulation 窗口 chunk 数、有效 target 数和 normalization state；
- 可选未清空梯度；
- manifest UUID/hash；
- 最终配置、代码版本和 checkpoint generation。

要求：

- 使用 `torch.distributed.checkpoint` 或 TorchTitan 公开封装；
- 临时目录写入，所有 rank 成功后原子提交；
- 半写 checkpoint 不可见；
- 相同 topology 精确恢复到下一个 chunk；
- 恢复 1B 序列时不重放前缀；
- 默认只在 optimizer/gradient flush 后保存；
- 若允许 accumulation 窗口内保存，必须恢复未完成梯度；
- world-size 改变仅保证安全边界 elastic resume，并明确标记非 bitwise。

## 14. 训练日志

输出：

- TensorBoard；
- rank-0 JSONL；
- 控制台摘要；
- 可选 TorchTitan metrics sink；
- 可选 `torch.profiler` trace。

至少记录：

- global/optimizer step；
- consumed atoms、language tokens、images、records 和 chunks；
- language loss、image loss、有效 target 数；
- learning rate、grad norm、overflow/clip；
- data wait、H2D、forward、backward、optimizer、checkpoint 时间；
- atoms/s、language tokens/s、images/s；
- padding 比例和有效监督比例；
- lane refill/reset 和平均连续长度；
- memory/state 大小；
- GPU allocated/reserved/peak memory；
- FSDP communication 和 checkpoint 吞吐；
- backbone、kernel、dtype 和 backend。

所有指标必须声明单位和分布式聚合语义。

## 15. 配置

配置至少分为：

- `data`；
- `tokenizer`；
- `image`；
- `model`；
- `backbone`；
- `runtime`；
- `tbptt`；
- `checkpoint`；
- `logging`。

要求：

- 配置 schema 版本化；
- 启动前校验未知字段、缺失字段和类型；
- backbone 枚举为 `kda`、`gdn`、`gdn2`、`transformer_swa`；
- special token ID 和结构行为来自 tokenizer/config；
- 配置不得包含固定 PODAR/POTAR 顺序；
- 最终解析配置写入运行目录和 checkpoint。

## 16. 兼容与迁移

- 所有 trajectory producer 直接写 special-token-wrapped V1 atom stream；
- producer 决定 observation/action/reward 如何编码，Data I/O 不保留固定领域字段；
- reader/trainer 只读取 V1，且支持单 dataset 与多进程 record collection；
- 旧 `.npy` Dataset、DDP/EpochManager、PrefetchDataLoader 和项目专用训练入口已移除；
- 不保留 `LegacyPOTARAdapter` 或旧 PODAR/POTAR 运行时兼容层；
- task/coach 文件仍可作为采样器输入，但训练轨迹只能输出为 V1。
- `data/` 只保留 producer 逻辑；MazeWorld、AnyMDP、Gym、MetaControl 和 MetaLang 不得提供独立 RoboFM 训练入口；
- `projects/UnifiedSequence` 是唯一的训练、验证、checkpoint 和 resume 入口。

## 17. 测试与验收

### 17.1 Data I/O

- token/image atom 编解码；
- uint64 offset 和大于 32-bit 的逻辑位置；
- mmap random seek 和跨 shard；
- image index、shape、codec 和损坏 payload；
- special token 嵌套不由 Dataset 解释；
- causal shift、loss mask 和 explicit target mapping；
- input/output 不同长度；
- padding、partial chunk、全 padding lane；
- 1B 逻辑长度虚拟 record，不加载完整序列。

### 17.2 Backbone

- 四类 backbone forward/backward；
- language-only、image-only、image-language 混合序列；
- 完整序列与 chunked state 一致性；
- padding/memory/reset mask；
- lane state select/scatter/detach/serialize；
- BF16；
- FSDP2 1/2/8 GPU smoke test。

### 17.3 TBPTT

- 1K chunk、10 chunks/update；
- 跨多个 optimizer update 的 memory continuity；
- 反向图只覆盖配置窗口；
- update 后 state detached 但数值连续；
- 1M chunk counter/cursor 不溢出；
- 峰值显存不随总序列长度增长。

### 17.4 Resume

- 在读取、forward、backward、optimizer 和 checkpoint 阶段注入中断；
- accumulation 边界和窗口内部恢复；
- 下一 chunk cursor、memory、RNG、loss、optimizer 和参数一致；
- lane refill/reset 边界不重复、不跳数据；
- 半写 checkpoint 不可加载。

### 17.5 性能

- 与当前 `.npy + DDP` 基线比较 data wait、吞吐、显存和 checkpoint 时间；
- benchmark 记录硬件、软件、配置、tokenizer hash 和 manifest hash；
- 性能门槛在服务器基线建立后确定。

## 18. 分阶段交付

### 阶段 A：契约与能力验证

- unified atom/target/mask/state schema；
- tokenizer 和 image payload 契约；
- TorchTitan capability spike；
- 四类 backbone capability matrix；
- gold streams 和测试规范。

### 阶段 B：Data I/O

- writer、reader、validator；
- mmap Dataset；
- lane scheduler；
- legacy converter；
- I/O 正确性与性能测试。

### 阶段 C：统一模型

- language/image embedding；
- unified causal model；
- language/image heads；
- KDA/GDN/GDN-2/Transformer-SWA；
- state 和 mask 一致性测试。

### 阶段 D：训练与恢复

- FSDP2/TBPTT runtime；
- TorchTitan/PyTorch logging；
- distributed checkpoint；
- 1B sequence、周期 backward 和故障恢复测试。

### 阶段 E：MazeWorld 样板

- legacy 数据转换；
- special-token 序列约定；
- 单卡、多卡、resume 和性能验证；
- 输出其他 producer 的迁移模板。

### 阶段 F：旧路径收敛

- 删除新路径中的位置 tuple、`model.module` 和项目内 segment 循环；
- 停用 PODAR/POTAR；
- 删除旧 DDP/EpochManager/PrefetchDataLoader；
- 固化后续数据合成器直接写 unified stream 的 API。

## 19. 风险与待确认项

1. Image target 是使用原始像素重建、latent prediction 还是离散 image token，需要在阶段 A 固定默认实现。
2. Special token 词表和 producer 约定需要版本化，但不应进入 Data I/O 类型系统。
3. TorchTitan 接口变化较快，必须固定版本并隔离依赖。
4. GDN-2 的 FLA 公开 API 和稳定性需要验证。
5. Python 3.14、CUDA 13、Triton、FLA 和 TorchTitan 需要兼容矩阵。
6. 1B 序列的 checkpoint memory 可能很大，需要评估 CPU offload、异步写入和压缩。
7. 对 image atom 的动态编码可能成为吞吐瓶颈，应支持离线 image latent/token。
8. world-size-change resume 不保证 bitwise 一致。

## 20. 实施前默认建议

- 数据底层严格限制为 language token 和 image 两类 atom；
- 领域语义全部由 special token 表达；
- 默认使用单 causal stream + loss mask；
- 非 shift 对齐场景使用只含 language/image target 的显式 mapping；
- 新模型只有 language/image 输入输出能力，不内建 policy/reward/world-model 类型；
- 新训练核心使用 FSDP2；
- 优先复用 TorchTitan 公开组件；
- 相同 topology 提供精确 resume；
- MazeWorld 是首个样板；
- 数据生成策略的系统性迁移延后。

确认本文档后，实施从阶段 A 开始，不直接重写现有数据生成逻辑。
