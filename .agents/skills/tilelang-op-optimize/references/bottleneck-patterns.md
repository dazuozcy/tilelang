# Ascend A5 瓶颈模式参考（冷启动版）

## 用途

本文件是 `tilelang-op-optimize` 的可迭代瓶颈模式参考。每轮诊断时，先按 [iteration-diagnosis.md](iteration-diagnosis.md) 建立本轮诊断上下文，再从本文件匹配可能的瓶颈模式。

这些模式不是固定分类树。一个 profile 可以同时匹配多个模式；每个匹配到的模式都可以生成一个或多个候选优化点，并在同一轮分别创建实验分支验证。

> **A5 冷启动说明（2026-10-16）**：本文件自旧栈复制后按 ascend（A5/950，dav-3510）调优手段**从简重建**——每条只写「触发信号 → 假设 → 验证方法」，不写未经实证的收益数字；实测收益、代价常数与反例由调优任务回写 pattern-library（[INDEX.md](pattern-library/INDEX.md) §1）后在此引用。时延主指标 `tilelang.profiler.do_bench(fn, backend="msprof")`（毫秒 ×1000 记 µs）；周期级归因用 npusim；pipe ratios 用 `backend="msprof_detail"`（权威：`tilelang-ascend` skill §12 + `references/profiling.md`）。

## 目录

- `BP_autoschedule_suboptimal`：AutoSchedule 调度次优（task 顺序/重叠不足）。
- `BP_cost_model_misestimate`：VF/Task 代价估计失准（latency=/ii= 注解缺口）。
- `BP_memory_bandwidth_or_mte`：GM/MTE 搬运效率不足。
- `BP_l2_cache_policy`：L2 cache_ctrl 策略不当。
- `BP_fractal_conversion_overhead`：fractal/ND2NZ 转换开销占比高（布局选择）。
- `BP_dual_copy_utilization`：dual_copy / 双 AIV 利用率不足。
- `BP_pipeline_overlap`：搬运和计算未重叠（num_stages / multibuffering）。
- `BP_onchip_capacity`：UB/L1/L0 容量压力限制 tile/stage。
- `BP_scalar_ratio`：标量执行占比高 / 向量化不足（SimdVF 向量宽度/掩码模式）。
- `BP_grid_balance_or_tail`：grid 划分不均、尾块或持久化分界不当。
- `BP_redundant_work_or_roundtrip`：重复计算或多余 GM 往返。
- `BP_compute_granularity`：Cube/Vector 计算粒度或分形对齐不当。
- `BP_parameter_uncertain`：结构稳定但参数不确定。
- `BP_measurement_resolution_limited`：测量噪声内候选无法可靠排序；这是测量质量模式，不代表 kernel 本身瓶颈。
- `BP_run_state_bimodality`：同配置跨 run 快/慢双态，小差异裁决需交错多 run 协议。

新增模式时保持同一结构：

```markdown
## BP_xxx：一句话描述

触发信号：

- ...

假设：

- ...

验证方法：

- ...
```

跨模式提醒：

- **solved II / 编译器估计 latency / npusim 周期是调度模型与归因输入，不是设备时延**；winner 判定只看 do_bench(msprof)。
- pipe ratios 相对各自 AIC/AIV cycles，不可相加，也不必合计 100%（重叠 pipe 各自占比）。
- 一个配置失败只记为 `config_no_gain`；只有覆盖关键配置或有明确机制证据时，才能把整类方向记为 `family_no_gain`。
- 某个模式在当前环境无收益，只能记录为当前 workload / tilelang-fork / CANN / 编程模式下的实测结论，不要泛化为永久无效。
- 候选时延曲线可能出现平区；平区内不要用单次测量结果排序（见 `BP_measurement_resolution_limited`）。

---

## BP_autoschedule_suboptimal：AutoSchedule 调度次优（task 顺序/重叠不足）

触发信号：

- msprof_detail 显示 mte 与 mad/vec pipe 交替忙闲（阶段性空洞），或搬运明显晚于计算开始、写出阻塞 buffer 复用。
- npusim trace 显示生产者完成后消费者长时间等待，且等待并非必需依赖。
- 生成源码（`kernel.get_kernel_source()`）中 task 顺序与依赖允许重排（独立同 PIPE task、可跨 Stage 的生产者-消费者）。

假设：

- AutoSchedule 在估计代价下得到的 task 顺序/重叠不是最优；显式调度约束（`T.Stage` 划分、独立同 PIPE task 源码重排、`T.PerCoreTask` 候选归并）可改善重叠。

验证方法：

- 按 `tilelang-ascend` skill `references/synchronization.md`「Tune performance with T.Stage」流程执行：保存 baseline（生成源码 + do_bench 时延），固定 tile/核划分/VF 代码只改调度约束；检查物化调度（生成顺序、跨迭代重叠、prologue/epilogue、wait、buffer versions）确实变化；do_bench(msprof) 复测 + 精度回归（含尾块/短循环/重复发射）。无实测收益则回退自动调度。
- 注意：`T.Stage` 须包住完整 task（如 `with T.Stage(1), T.SimtVF(...)`），不能放进 Task/VF/parallel loop 内部；非零 Stage 需 `enable_offset=True`。

## BP_cost_model_misestimate：VF/Task 代价估计失准（latency=/ii= 注解缺口）

触发信号：

- npusim 实测某 VF/搬运/Cube task 的周期与编译器估计显著不符（调度形态上该 task 后出现异常等待）。
- msprof_detail pipe ratios 与预期占用形态矛盾（某 pipe 占比异常低但数据依赖它）。

假设：

- 未注解的估计代价失准导致 AutoSchedule 次优排程；用实测周期回填 `T.Task(latency=, ii=)` / `T.SimtVF(latency=)` / `T.SimdVF(latency=)` 可改善。

验证方法：

- 按 `tilelang-ascend` skill §12.4 + `references/profiling.md`「Override automatic task estimates」：npusim 隔离测量该操作（保持 shape/dtype/布局/掩码/线程维度/依赖一致），取**完整任务体**的 latency（输出就绪延迟，输入就绪且无关队列排空）与独立流的 ii（持续发射间隔），勿用单指令值或 burst 均值充当；约束 `latency >= ii > 0`。
- 注解后重编译 → 正确性验证 → do_bench(msprof) 复测完整 kernel；task 体/shape/布局/工具链变化后重测。

## BP_memory_bandwidth_or_mte：GM/MTE 搬运效率不足

触发信号：

- GM 读写量大、计算 pipe 占比不高；有效 GB/s（派生指标）远低于带宽模型档位。
- msprof_detail mte2/mte3 占比高；copy 粒度小、跨步、不连续（N-strided 按档位折算，见 CONST-mte-bandwidth-model）。

假设：

- tile shape 使 GM↔L1/UB 搬运不连续或粒度不足；合并搬运/调整布局可提升有效带宽。

验证方法：

- 调整 tile shape 使 GM→L1/UB 与 UB→GM 连续且粒度足够；合并写回或融合后处理减少 GM round-trip；对照 do_bench 时延与有效 GB/s 改善。
- npusim 定位 copy 区间周期，对照 CONST-mte-bandwidth-model 的带宽/延迟档位（含 N-stried 档），确认是否落在预期档位。

## BP_l2_cache_policy：L2 cache_ctrl 策略不当

触发信号：

- 同一 GM 数据被多 block 重复读（权重/查找表），未利用 L2 驻留。
- msprof_detail 显示 mte2 等待长、有效带宽低于模型，且访问模式跨 block 复用明显。

假设：

- 默认 L2 策略（`T.copy` GM→L1/UB 默认 `NORMAL_FV`、UB→GM 默认 `NOTALLOC_CI`，`T.dual_copy` UB→GM 显式 `NORMAL_FV`——见 `tilelang-ascend` skill §6.1）不适合该访问模式；显式 `l2_cache_ctrl` 可改善。

验证方法：

- 对重复读/流写的 copy 显式设置 `l2_cache_ctrl`（合法策略名与默认值映射以 `tilelang/ascend/language/copy_op.py` 为准，大小写不敏感）；单点实验 + do_bench(msprof) 对比；记录策略与 copy 路径的合法组合。

## BP_fractal_conversion_overhead：fractal/ND2NZ 转换开销占比高（布局选择）

触发信号：

- mixed/GEMM 类 kernel 中 ND→NZ 分形转换（InsertNd2Nz pass 自动插入的搬运）占比高（npusim/pipe ratios 归因）。
- GM 端布局上游可控，当前布局对分形转换不友好。

假设：

- 布局选择使分形转换开销成为主要成本——输入以分形友好布局存放、或 `T.copy` GM→L1 `transpose=True` 选 dn2nz 路径，可减少转换开销（ND2NZ base cycles 见 CONST-mte-bandwidth-model）。

验证方法：

- 对照 `tilelang-ascend` skill §6.1 transpose 语义（GM→L1 `transpose=True` 选 dn2nz 并反转逻辑源/目的轴；**非通用转置操作**）；评估 GM 布局变体的转换开销收益。
- 布局契约变化须与调用方对齐（I/O layout 是契约）、重跑精度与全部 tune workload 非回退检查。

## BP_dual_copy_utilization：dual_copy / 双 AIV 利用率不足

触发信号：

- mixed kernel 的 L0C→UB 输出搬运只占单 AIV；AIV pipe ratios 显示两个 sub-core 负载不均；输出 epilogue 成为瓶颈。

假设：

- 未用 `T.dual_copy`（L0C→UB 双 AIV 分担，M/N-split）或双 AIV 角色划分（sid 分派）不当。

验证方法：

- 对照 `tilelang-ascend` skill §6.2（L0C→UB 需控制流跨 sub-core 一致；M-split/N-split 形状约束、N-split 须完整 N 被 32 整除）与 §3.2（`T.MixedKernel(sids=2)`、sid 分派不同工作）。
- 改 dual_copy 或调整 sid 分工后 do_bench(msprof) 复测 + 精度回归；独立选择目标地址的场景改 guarded `T.copy`（independent AIV execution，见 `references/synchronization.md`）。

## BP_pipeline_overlap：搬运和计算未重叠（num_stages / multibuffering）

触发信号：

- 代码按 tile 串行执行「搬入 → 计算 → 写回」。
- msprof_detail 显示 MTE 与计算 pipe 交替忙闲、串行感明显；`T.Pipelined` 未用或 num_stages 不足。

假设：

- 增大 `T.Pipelined(num_stages=k)` 或依赖 AutoSchedule multibuffering（buffer versions）可重叠搬运与计算。

验证方法：

- 小步尝试 `T.Pipelined` num_stages 提升；检查 buffer versions 与容量（AutoSchedule 自动加版本维度，UB 膨胀对照 CONST-capacity-a5——含 SimtVF 时上限更低；必要时 `T.annotate_buffer_versions` 固定版本数，`{buf: 1}` 退出多缓冲资格）。
- do_bench(msprof) 复测 + 精度；stage 增加导致容量溢出或无收益时记录证据回退，不反复尝试。
- 注意区分：内层分块流水解决 overlap，持久化分界（`T.Persistent`）解决派发/驻留——一条失败不否定另一条（见 BP_grid_balance_or_tail）。

## BP_onchip_capacity：UB/L1/L0 容量压力限制 tile/stage

触发信号：

- tile/stage/buffer 增大后编译失败（容量约束）或性能回退；buffer 数量多、接近容量上限（CONST-capacity-a5）。

假设：

- 容量压力限制可行 tile/stage；buffer 生命周期复用可腾出空间。

验证方法：

- buffer 生命周期分析：标出每个 buffer 的定义点、last-use、dtype/shape；last-use 后复用（不只临时中间 buffer，也含输入 staging、cast buffer、输出 staging）。
- 缩 tile / 减 stage / 删不必要 buffer；容量检查计入 layout padding、buffer versions、编译器保留与对齐（`tilelang-ascend` skill §4.3）。
- `T.annotate_unlimit_memory` 只作调度器解约束的排查诊断（不扩容），结论须真机复测。
- 区分 UB staging 复用与 GM 输入/输出 alias；默认不假设 GM 输入可原地写成输出。

## BP_scalar_ratio：标量执行占比高 / 向量化不足

触发信号：

- msprof_detail AIV scalar pipe ratio 高占比（如 >50%），热点段以标量/地址计算为主。
- SimtVF 自动向量化未覆盖当前计算模式，或循环最内轴不连续。

假设：

- 向量化不足——换向量宽度/掩码模式（SimdVF 显式 2048-bit）或重排布局使最内轴连续可向量化，可把工作从 scalar pipe 移到 vec pipe。

验证方法：

- 强制评估换轴/布局/编程模式候选（SKILL.md「证伪协议」第 4 条）：`T.SimdVF()` + `T.simd.*`（掩码 `pset`/`pld`、2048-bit 寄存器整除性见 CONST-valu-model），或核内布局重排（pattern-library/layout.md）。
- 对照 do_bench 时延与 aiv_vec/aiv_scalar ratio 变化；精度回归（掩码/尾块路径重点覆盖）。

## BP_grid_balance_or_tail：grid 划分不均、尾块或持久化分界不当

触发信号：

- shape 不能整除 tile，尾块/OOB padding 分支明显；部分 block 空转或极小。
- `T.Kernel(N)` 一维 grid 远大于或远小于核数实查值（CONST-arch-a5：运行时 `torch.npu.get_device_properties()` 实查，惯例初值 32 待标定）；flat-grid 派发开销高。

假设：

- grid 划分或持久化分界存在更优候选——持久化遍历（`T.Persistent` + `tile_schedule` 核-tile 映射）或不同 grid 大小可降低派发开销/imbalance。

验证方法：

- 计算 `num_logical`、候选 grid 的 `min/max iters` 与 `imbalance`；grid 候选覆盖：核数附近、整数倍核数、`num_logical` 附近低 imbalance 候选、flat-grid 端点（只作对照）。
- 持久化分界用 `T.Persistent`（参考 `examples/ascend/deepgemm/` 持久化遍历与 `tilelang/ascend/language/tile_schedule.py`）；逐候选记录 do_bench 时延曲线；平区按 BP_measurement_resolution_limited 交错复测。

## BP_redundant_work_or_roundtrip：重复计算或多余 GM 往返

触发信号：

- 多个小 kernel 通过 GM 传递中间结果；中间结果写回 GM 后又读回。
- 多个 reduce 或多 pass 遍历同一批数据；循环内重复计算地址、scale、mask、index 或公共表达式。

假设：

- 融合/复用可减少 GM 读写与重复计算。

验证方法：

- epilogue fusion、reduce 合并 / single-pass、公共子表达式复用、streaming / online 递推公式。
- 对照 GM 读写次数（有效 GB/s）、向量/标量操作数与 do_bench 时延；融合后 UB 压力须单独验证（见 BP_onchip_capacity）。

## BP_compute_granularity：Cube/Vector 计算粒度或分形对齐不当

触发信号：

- Cube/Vector pipe 占比低；GEMM 类 kernel 的 tile 不匹配 C0 分形（CONST-gemm-dim-constraints）。
- 大量小向量操作碎片化（未填满 2048-bit 向量寄存器）。

假设：

- tile 粒度/分形对齐不当使硬件管线使用不足。

验证方法：

- 调整 `block_M / block_N / block_K`（对照 CONST-gemm-dim-constraints：M/N 16 倍数、`K % TILE_K_SUB == 0` 且 `TILE_K_SUB % C0 == 0`、NT 约定权重 [N,K]）。
- 向量 tile 对齐 2048-bit（CONST-valu-model）；合并小循环增加每 block 有效计算量。
- 对照 do_bench 时延与 pipe ratios；GM/MTE 或容量压力没有明显恶化。

## BP_parameter_uncertain：结构稳定但参数不确定

触发信号：

- 已经确定结构方向，但 tile/grid/num_stages/threads 多个配置都合理。
- 同一 kernel 内不同 shape/dtype workload 的最优参数可能不同；手动推理无法可靠选择 winner。

假设：

- 参数空间内存在更优配置；受控搜索比手动推理可靠。

验证方法：

- 按 [autotune.md](autotune.md) 定义小而合法的搜索空间（预先过滤违反容量/分形对齐/一维 grid/shape 整除约束的配置）。
- winner 必须单独用 do_bench(msprof) 复测，不能只用 autotune latency；平区内记录 imbalance/整除性决胜依据。

## BP_measurement_resolution_limited：测量噪声内候选无法可靠排序

触发信号：

- do_bench(msprof) 时延曲线大面积平坦，候选差异小于噪声阈值（默认按 10% 判断）。
- 同一配置跨独立运行漂移明显，或 session 早期出现系统性膨胀。
- 参考实现、历史 best 的时延与当前量级明显不一致。

假设：

- 这是测量质量模式：候选差异落入测量噪声，当前数据不足以可靠排序（不代表 kernel 真实瓶颈）。

验证方法：

- 采集前先预热（do_bench 自带 warmup；首 JIT/模块加载/profiler 初始化与稳态分离），再记录有效数据。
- 候选差异小于噪声阈值时，用更多采样与独立运行复测（同 workload 重复运行方差 ≤1% 为对账阈值）；复测后仍打平的平区候选，用 imbalance、整除性、资源余量和代码复杂度决胜，不要用单次测量结果排序。
- 记录每次采集的采样数与独立运行次数；作废的噪声运行不进入最终排序。

## BP_run_state_bimodality：同配置跨 run 快/慢双态，小差异裁决需交错多 run 协议

> 溯源：旧栈两算子 optimize 任务首证/二证（A5 适配内容清零，协议保留；A5 实证待积累，回写 pattern-library 后更新本条）。

触发信号：

- 同一 kernel、同一配置，跨独立 do_bench(msprof) run 的时延差 ±3–5% 量级，且差异与配置无关（疑似 per-process 快/慢态）。
- 小 kernel（时延 <20µs）或逼近带宽/搬运地板的 workload，候选间差异落在噪声阈值（3%）附近。
- 复测翻转：首轮判回退复测变 tie；或单轮增益复测收缩。

假设：

- 存在与配置无关的跨 run 双态；单 run 中位数可能把双态膨胀误判为增益或回退。

验证方法：

- 首次出现 <5% 量级候选差异即前置启用交错 A/B/A/B 协议：≥3 次独立 do_bench(msprof) run、候选与基线交替次序、合并中位数比较——勿等采纳后复核。
- 过 3% 门槛的采纳判断必须基于合并中位；配对次序稳定（同一方向跨 run 成立）才判定真实差异。
- 差异随采样数增大而消失 → 采集噪声，改判 BP_measurement_resolution_limited。
- 工具化执行：`python3 .agents/tools/ab_test.py`（do_bench(msprof) 交错多 run 封装，参数以其 `--help` 为准）；opt_log 记录各 run 的次序与中位数，A/B 配对结论次序一致才判定差异真实。
