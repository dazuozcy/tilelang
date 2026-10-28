# Ascend TileLang 当前现象分析与候选优化点指南

## 目录

- 用途
- 输入
- Step 1：确认 Profile 口径
- Step 2：建立本轮诊断上下文
- Step 3：生成候选优化点
- Step 4：匹配瓶颈模式
- Step 5：分别创建实验分支
- Step 6：评估本轮 winner
- Step 7：本轮输出模板
- 注意事项

## 用途

当 `tilelang-op-optimize` 进入 Phase 2 优化闭环的每一轮“本轮优化点分析”时，读取本文件。

本文件不把 kernel 强行归到某个固定瓶颈大类，而是把 do_bench(msprof) 数据、msprof_detail/npusim 深挖数据、kernel 代码结构和历史迭代结果转换成：

```text
本轮诊断上下文 -> 当前现象 -> 候选优化点 -> 独立实验分支 -> 本轮 winner
```

核心原则：

```text
先记录事实和上下文，再描述当前现象。
一轮可以生成多个候选优化点，并分别尝试。
每个实验分支都从同一个 current best 派生，只改一个主要优化点。
实验后先选择本轮实测最好的 valid/improved 分支作为候选 winner。
完成目标 workload 的迭代后才尝试核内合并；合并版通过本 kernel 全部 tune workload 的非回退检查与 smoke 精度检查后，才能成为新的已合并 current best。
下一轮基于新的 current best 测量数据重新分析当前现象。
```

---

## 输入

诊断前必须具备：

- `{op}.py`、current best 或当前实验分支 `perf_opt/{op}_round{iter}_{opt_id}.py`。
- `DESIGN.md` 中的性能目标章节。
- [hardware-context.md](hardware-context.md) 中的硬件上下文。
- 按 [profile-collection.md](profile-collection.md) 从 benchmark 的非 smoke workload 逐 kernel 采集到的测量结果（do_bench(msprof) 是唯一时延口径；msprof_detail/npusim 为深挖与归因口径）。
- `perf_opt/opt_log.md` 中的 `Performance Test Data` 最小表。
- 已尝试过的优化点、实验分支结果、keep/rollback 结论和最新 current best。

诊断只允许使用 `profile_status=valid` 的记录。若某个 kernel/workload 的测量被标记为 `profile_invalid`，或被测 callable 无法追溯到目标 TileLang kernel 与候选版本，不能用该数据判断硬件瓶颈。

---

## Step 1：确认 Profile 口径

先确认：

```text
Performance Test Data 已存在
profile_status == valid
kernel_id 与 workload_id 已记录，且 inventory 分类为 tune
被测 callable 来自当前要优化的 TileLang kernel（实现文件与候选版本可追溯）
duration_us 来自 do_bench(fn, backend="msprof")（毫秒 ×1000），多次采样取中位
采样数与独立运行次数已记录，同 workload 重复运行方差 ≤1%
采到的不是 cache flush 辅助 kernel 或 Cast / Mul / OnesLike / Random 等框架小算子
profiler_output（msprof_detail 深挖 / npusim trace）存在时能追溯到本次测量
同一轮采集未出现并发 profiling
```

按 `kernel_id + workload_id` 分别诊断，不把不同 workload 的时延合成平均数后下结论。smoke 不进入性能诊断，只用于合并版精度回归。

---

## Step 2：建立本轮诊断上下文

每轮先记录事实和上下文，不要急着下结论。

### Profile 事实

```text
do_bench(msprof) 时延（us）、采样数、复测方差
msprof_detail（如有）：dur_ns、AIC/AIV total cycles、pipe ratios
  - AIC: mad / scalar / mte1 / mte2 / mte3 / fixpipe
  - AIV: vec / scalar / mte2 / mte3
npusim 周期数据（如有）：VF / 搬运 / Cube / 同步交接区间的周期
grid 大小（T.Kernel(N) 的 N）与核数实查值（torch.npu.get_device_properties()）之比
派生指标：TFLOPS = 2*M*N*K / t、有效 GB/s
```

pipe ratios 相对各自 AIC/AIV cycles，不可相加；重叠 pipe 占比不必合计 100%。

### 理论估算

尽量从代码和 workload 推导：

```text
logical tile 数与 grid 划分（一维 grid 映射，如 row = tile_idx // N_TILES; col = tile_idx % N_TILES）
候选 grid 的 min/max iters、imbalance（若做持久化/grid 划分实验）
单 block 元素数 / 字节数
GM 读写总字节数
理论 memory-bound 周期（CONST-mte-bandwidth-model，cycle 域相对比较）
时延与理论差距（绝对换算须 do_bench 实测频率基线，禁止编造频率）
```

### 代码结构观察

记录当前 kernel 的结构事实，重点看这些结构是否能解释当前现象：

- dispatch 入口：是否有 dtype / axis / layout 等真实分支（仅 `@T.prim_func` 内）。
- kernel 边界：一次测试是否调用多个 TileLang kernel，或夹杂框架生成的小算子。
- 并行粒度：`T.Kernel(N)` 一维 grid 大小、每个 block 处理的数据量、`T.Persistent` / `tile_schedule` 持久化遍历是否使用。
- 编程模式：SimtVF threads、SimdVF 掩码与向量宽度、mixed 的 `sids` 与 AIC/AIV 分工。
- 调度：AutoSchedule 默认还是 `T.Stage` / `T.Task(latency=, ii=)` / `T.PerCoreTask` 提示；`T.Pipelined` num_stages；buffer versions（`T.annotate_buffer_versions`）。
- 搬运路径：`T.copy` 粒度 / 连续性 / 对齐 / `l2_cache_ctrl`；ND2NZ 分形转换（pass 自动插入）；`T.dual_copy` 双 AIV 分担是否使用。
- 搬运/计算组织：是否按 tile 串行执行“搬入、计算、写回”，是否具备 pipeline / multibuffering 的空间。
- 片上 buffer：UB/L1/L0 buffer 数量、大小、生命周期和 last-use；输入 staging、cast buffer、临时 buffer、输出 staging 是否能在 last-use 后复用。
- 尾块与分支：是否存在明显 tail block、OOB padding、mask、if branch 或空 block。
- 冗余访问：是否有中间结果写回 GM 后又读回，或重复读取同一批 GM 数据。

### 已知迭代结论

每轮必须读 `opt_log.md` 中已有结论：

```text
哪些优化点已验证有效
哪些优化点已被当前 workload 证明无收益
哪些修改导致精度失败 / 编译失败 / 测量 invalid
current best 是哪个版本
本轮测量数据相比上一轮哪些指标发生变化
```

---

## Step 3：生成候选优化点

候选优化点不是互斥分类。一个 profile 可以同时生成多个优化点。

每个候选优化点必须是可执行实验，而不是抽象判断。它必须写清：

```text
想解决哪个当前现象
为什么认为这个优化点可能有效
具体怎么改
看什么指标验证
什么情况回滚
```

候选表格式：

```markdown
| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP1 | grid 过大且单 block 工作量小 | 持久化遍历（T.Persistent + tile_schedule） | ... | 从 current best 派生 round{iter}_op1，只改 grid 分配 | do_bench 时延 / grid 大小 / correctness | candidate |
```

状态含义：

```text
candidate：本轮候选
running：实验中
improved：实测有提升
config_no_gain：当前配置无收益，只否定该配置
family_no_gain：整类方向无收益，必须有足够配置覆盖或明确机制证据
invalid：精度失败或测量无效
blocked：编译、环境或合法性问题阻塞
defer：证据不足，暂缓
```

---

## Step 4：匹配瓶颈模式

如果当前现象需要模式参考，读取 [bottleneck-patterns.md](bottleneck-patterns.md)，用本轮诊断上下文辅助生成候选优化点。

匹配时注意：

- 不要把瓶颈模式当成互斥分类树。
- 一个 profile 可以同时匹配多个模式。
- 尽可能把有证据支撑、能落成代码实验的优化点放入候选表。
- 如果现有模式覆盖不了当前现象，可以自己寻找优化点，但必须写清触发信号、假设和验证方法。

---

## Step 5：分别创建实验分支

本轮可以尝试多个候选优化点，但每个实验分支必须满足：

```text
base = 本轮开始时的 current best
每个分支都从 base 派生
一个分支只改一个主要优化点
不要在同一个分支里同时叠加多个候选优化点
```

推荐命名：

```text
perf_opt/{op}_round{iter}_{opt_id}.py
```

例如：

```text
perf_opt/mish_round2_op1.py
perf_opt/mish_round2_op2.py
perf_opt/mish_round2_op3.py
```

如果某个优化点本质上是参数选择，可以在该分支内做小范围参数实验；但搜索空间必须服务同一个优化点，不能混入其它结构性改动。

结构性优化分支通过正确性且方向有效后，不能只保留单个手写配置。必须围绕该结构暴露的关键参数做一轮 coarse autotune 或等价手动粗搜；例如 `T.Pipelined` 分支至少搜索 tile size × num_stages，持久化/grid 划分分支至少搜索 grid 大小曲线。autotune winner 不是最终结论，还要检查 top-k 配置和 winner 邻域，必要时手动/脚本精搜，再用最终 winner 进入 do_bench(msprof) 复测。

若优化点涉及 grid 划分 / 持久化遍历，不能只测一个配置。先计算 `num_logical=ceildiv(N, tile)`，再构造候选：

- 核数实查值附近和若干倍核数（任务并发锚点）。
- `num_logical` 附近可整除或低 imbalance 的候选，记录 `min_iters / max_iters`。
- 当前 flat-grid 端点只作对照，不默认作为 winner。

每个配置都记录 do_bench(msprof) 时延和 imbalance。若甜点区内时延差异小于噪声阈值，先用更多采样与独立运行复测（方差 ≤1%）；复测后仍打平时用 imbalance、整除性和资源占用决胜，不要用单次测量结果对平区内候选排序。

---

## Step 6：评估本轮 winner

每个实验分支都要独立跑：

```text
L0 精度回归
do_bench(msprof) 性能测量
测量有效性校验
```

评估规则：

- 只比较同一 `kernel_id + workload_id` 下的 valid 测量。
- 有一个或多个分支提升时，先按本轮主指标（do_bench(msprof) 时延，µs）选择候选 winner；时延打平时，用 imbalance、UB/L1/L0 余量和代码复杂度决胜。
- 每个 workload 完成迭代后，把其 winner 尝试合入同一个 kernel；先尝试无条件优化，若本 kernel 其他 tune workload 有可信回退，再尝试核内 shape/dtype/参数分支并重新编译、验证。
- 只有合并后的实际 kernel 通过本 kernel 全部 tune workload 的精度和非回退检查、以及 smoke 精度检查，才能更新已合并 current best。
- 小 kernel（时延 <20µs）或逼近带宽/搬运地板的 workload，<5% 量级的候选差异须先经**交错 A/B/A/B 多 run 协议**裁决：≥3 次独立 do_bench(msprof) run、候选与基线交替次序、合并中位数比较——同 kernel 跨 run 存在 ±3–5% 快/慢双态（BP_run_state_bimodality），单 run 中位数可能把双态膨胀误判为增益或回退。协议在首次出现 <5% 差异时即前置使用，而非采纳后复核。工具化执行：`python3 .agents/tools/ab_test.py`（do_bench(msprof) 交错多 run + 合并中位差 + 配对方向一致性判定，参数以其 `--help` 为准）。
- 如果合并版只在目标 workload 提升，但本 kernel 其他 tune workload 明显回退，且核内条件分支不能安全解决，记录 rollback 或 defer，不更新已合并 current best；不得转移到 Python factory、op 层或 wrapper 分派。
- 如果所有分支都无提升、无效或阻塞，current best 保持不变。
- 其它分支记录为 `config_no_gain / family_no_gain / invalid / blocked / defer`。
- 单个配置无收益只能记为 `config_no_gain`；不能据此否定 `T.Persistent / T.Pipelined / grid 划分` 这类整类方向。
- 只有在关键配置已覆盖，或有明确机制证据时，才能记录为 `family_no_gain`。
- 已被证明无收益的配置，下一轮不要重复尝试；整类方向只有在 `family_no_gain` 后才停止尝试，除非新的 current best 测量形态已经变化。
- autotune 只在结构稳定或结构性分支已经通过正确性后使用，不作为逃避现象分析的默认动作。

---

## Step 7：本轮输出模板

```markdown
## Iteration {iter} Diagnosis

### Diagnostic Context

- kernel_id: {kernel}
- workload_id: {workload}
- current_best: {file}

#### Profile Facts

- do_bench(msprof) latency: {duration_us} us（samples={n}，variance={v}%）
- grid size / core count: {grid} / {core_count}
- AIC pipes (msprof_detail): mad={v} scalar={v} mte1={v} mte2={v} mte3={v} fixpipe={v}
- AIV pipes (msprof_detail): vec={v} scalar={v} mte2={v} mte3={v}
- npusim cycle attribution（如有）: {region -> cycles}
- derived: TFLOPS={v} / effective GB/s={v}

#### Theory Estimate

- estimated GM bytes: {bytes}
- estimated memory-bound cycles（CONST-mte-bandwidth-model）: {theory_cycles}
- latency gap: {brief}
- grid / iters / imbalance: {if_applicable}

#### Code Structure

- existing entry routing: {仅说明 workload 如何到达目标 kernel，不作为新增性能分派}
- kernel boundary: {single TileLang kernel / multiple kernels / extra framework ops}
- parallel granularity: {T.Kernel grid, per-block workload, T.Persistent / tile_schedule usage}
- programming mode: {simt / simd / mixed；threads、掩码、sids}
- scheduling: {AutoSchedule default / T.Stage / T.Task(latency=,ii=) / T.Pipelined num_stages / buffer versions}
- transfer path: {T.copy / dual_copy granularity, continuity, l2_cache_ctrl, fractal conversion}
- transfer/compute organization: {tile-by-tile serial transfer/compute/store or pipeline-ready structure}
- on-chip buffers: {UB/L1/L0 buffer count, size, lifetime}
- tail and branches: {tail block, OOB padding, mask, if branch, empty block}
- redundant access: {GM round-trip or repeated GM reads}

#### Known Iteration Conclusions

- previous effective/no-gain optimization points: {summary}
- current best profile changes: {summary}

### Current Phenomena

- P1: {phenomenon + metrics}
- P2: {phenomenon + metrics}

### Candidate Optimization Points

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP1 | {phenomenon} | {optimization_point} | {evidence} | {change} | {metrics} | candidate |

### Experiment Branches

| branch | base | opt_id | major_change | correctness | do_bench_us | profile_status | result |
|---|---|---|---|---|---:|---|---|
| {file} | {current_best} | OP1 | {change} | pass/fail | {v} | valid/invalid | improved/config_no_gain/family_no_gain/invalid/blocked |

### 当前 Kernel 全部 Tune Workload 合并检查

| merged_candidate | kernel_id | workload_id | current_best_us | merged_us | precision | status |
|---|---|---|---:|---:|---|---|
| {file} | {kernel} | {workload} | {v} | {v} | pass/fail | pass/regressed |

smoke workload 只列精度结果，不填性能时延或参与性能门禁。

### Iteration Winner

- winner: {file_or_current_best}
- reason: {本 workload 的 do_bench(msprof) 时延最优；完成该 workload 后另验证核内合并版在本 kernel 全部 tune workload 上无可信回退}
- new_current_best: {file}
- rollback_branches: {list}
```

---

## 注意事项

- 一轮可以尝试多个候选优化点。
- 每个实验分支只改一个主要优化点。
- 同一轮所有实验分支都必须从同一个 current best 派生。
- 组合优化只在单点优化已经证明有效后再做。
- 每个实验分支都必须跑 L0 精度回归。
- 每个有效实验分支都必须用 `do_bench(fn, backend="msprof")` 重新测量目标 kernel；候选差异落入噪声阈值时先用更多采样与独立运行复测，不要用单次测量结果对平区内候选排序。
- 更新已合并 current best 前，必须确认本 kernel 全部 tune workload 不发生可信性能回退，并完成 smoke 精度回归。
- 区分 `config_no_gain` 和 `family_no_gain`；一个配置失败不能否定整类优化方向。
- 如果所有候选优化点都不够直接，先补充测量或检查测量口径，不要盲目扩大搜索空间。
