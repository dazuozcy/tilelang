# 分核与调度策略标准（一维 grid + AIC/AIV 分工 + 同步策略）⭐

> **单一事实源**：本文件是「分核三要素」（grid 大小 / AIC-AIV 分工 / 同步策略）的权威版本。conductor / designer / reviewer / developer / optimizer / README 只引用本文件，不复制本文文本（可留一句话摘要）。
>
> **背景**（tilelang-fork ascend 后端，权威依据 `.agents/skills/tilelang-ascend/SKILL.md` §2/§3/§7）：Ascend A5 上 kernel launch 是**一维 grid**（`T.Kernel(N)`，无 threads 参数）；AIC（Cube）与 AIV（Vector）两类核，物理编组 1 AIC : 2 AIV（`__mix__(1,2)`），AIC/AIV 指派与跨核同步**默认由 AutoSchedule 自动完成**（AssignCore/InsertSync pass）。核数无编译期静态常数——运行时经 `torch.npu.get_device_properties()` 的 `cube_core_num` / `vector_core_num` 实查（examples/ascend 惯例初值 32 blocks，属待标定假设，见 `pattern-library/constants.md` CONST-arch-a5）。因此 Tiling 策略是**双重维度**：Block 尺寸（片上容量预算 UB 248KB〔SimtVF 216KB〕/ L1 512KB / L0C 256KB，对照 `CONST-capacity-a5`，assumption 条目按 90% 折扣）+ **grid 划分与核分工（block 数与 AIC/AIV 职责适配）**。

## 1. 分核三要素（Stage 1 门禁核对项）

DESIGN.md 的 Tiling 策略章节（§5）必须同时包含：

| 要素 | 内容 | 缺失判定 |
|------|------|---------|
| ① grid 划分（含核数依据） | `num_blocks = ceil(M/block_M) × ceil(N/block_N)`（按算子实际输出网格，多维分解自行计算一维映射）+ 核数依据：`torch.npu.get_device_properties().cube_core_num` / `vector_core_num` 实查值（记录查询代码与实际返回值；无真机时标明"惯例初值 32 blocks，assumption 待标定"并给实查命令），说明 block 数与核数的关系（单波/多波、persistent 化与否） | 无 grid 计算、无核数依据或纯假设值未标注 → 门禁失败 |
| ② AIC/AIV 分工 | 说明各计算落位：Cube 工作（`T.gemm` 等）与 Vector 工作（elementwise/reduction/搬运）各自所在（pure kernel 由 codegen 自动判定 AIC/AICV；mixed 用 `T.MixedKernel(sids=1\|2)` 显式控制 AIV 数量与 sid 分工——须写明 sid guard 划分与 dual_copy 所有权）；纯 Vector 算子注明仅占 AIV | 无分工说明 → 门禁失败 |
| ③ 同步策略（auto\|manual 声明） | 声明 `auto_schedule: yes`（默认，依赖 AutoSchedule 自动插 flag/同步，须说明依赖的数据依赖边界）或 `no`（完全手动：显式 `T.ascend_set_flag`/`T.ascend_wait_flag`/`T.ascend_pipe_barrier` 配对方案 + flag id 预算〔核内 8 / 跨核 16〕+ `TL_ENABLE_AUTO_SCHEDULE: False`）；`T.Stage`/`T.Pipelined`/`T.Task` 调度提示的使用位置 | 无声明或 manual 无配对方案 → 门禁失败 |

机械核对由 `statectl gate 1`（`gate_lint.py` 规则 `S1-CORES-LOGICAL` / `S1-CORES-PHYSICAL` / `S1-CORES-PLAN`）执行存在性检查；数值自洽与正确性由 Stage 2 维度 3 检视。

## 2. 各角色要求文本（调度 prompt 透传 / 自检用）

### 2.1 Stage 1 — designer 设计要求（透传文本）

> Tiling 策略必须含分核三要素：grid 划分（一维 `T.Kernel(N)`，含核数实查依据——`torch.npu.get_device_properties().cube_core_num`/`vector_core_num`，记录查询代码与返回值；无真机时标明惯例初值 32 blocks 为 assumption 待标定）、AIC/AIV 分工（Cube/Vector 工作落位；mixed 核写明 sid guard 划分与 dual_copy 所有权）、同步策略（`auto_schedule: yes|no` 声明；manual 时给 set_flag/wait_flag 配对方案与 flag id 预算）；核内串行循环边界必须为静态值。权威依据：`.agents/skills/tilelang-ascend/SKILL.md` §2/§3/§7。

### 2.2 Stage 2 — reviewer 检视要点（透传文本）

> 维度 3（Tiling 策略）须核对分核三要素齐全（grid 划分含核数实查/标注、AIC/AIV 分工、同步策略声明）、grid 大小与 block 取值自洽（片上容量对照 constants.md，assumption 条目按 90% 折扣）、manual 同步的 flag 配对完整、核内串行边界静态。

### 2.3 Stage 3 — developer 提醒

按 DESIGN.md §5 分核方案实现（grid 划分 + AIC/AIV 分工 + 同步策略），不得擅自改回逻辑核超发或绕过声明的同步方案。

### 2.4 Stage 4 — optimizer 调优提示（透传文本）

> 分核策略调优是可选优化策略：调整 block 数对齐物理核数整数倍（消除负载不均）/ persistent 化摊薄 launch 开销（`T.Persistent`）/ `T.Stage` 与 `T.Task(latency=, ii=)` 调度提示对照 AutoSchedule baseline 检验任务顺序与流水线重叠假设（`tilelang-ascend` skill §7/§8.3）。

另：DESIGN.md §1.6 含实验裁决三件套（主选+备选+裁决计划）时，A/B 实测为调优必做项——按裁决计划对代表 shape 实测主选 vs 备选（perf_opt/ 下产出备选变体，基准不动），实测出裁决所依赖的未知常数，按判定阈值裁决；备选胜出（全局或按 shape 分片）则采纳备选变体，并按 [perf-feedback.md](perf-feedback.md) 产出 `perf_feedback.md` + `[DESIGN_LIMIT]` 信号交 conductor 受控路由（附录补记回写 DESIGN.md 或设计修订路径 C）；主选胜出则以实测数字经附录补记固化 §1.6.3 判定依据。

## 3. 分核相关失败信号路由（conductor）

| 信号 | 来源 | 路由 |
|------|------|------|
| DESIGN.md 缺分核策略要素 | Stage 1 门禁（statectl gate） | 门禁失败 → `fail_stage(1)` 重试（计 `stage_retry_count[1]`） |
| 维度 3 分核策略 fail | Stage 2 `REVIEW.md` | 设计修订路径 A（计 `retry_count`） |
| `[DESIGN_ERROR]` 分核类（核内串行边界依赖动态值 / grid 远超核数致串行调度开销剧增 / manual 同步 flag 不配对） | Stage 3 Subagent | 设计修订路径 B（计 `retry_count`） |
| 分核参数性能不达标 | Stage 4 | Stage 4 内迭代（参数级不逆向反馈），不回退 Stage 1/3；判定为设计层天花板时按 [perf-feedback.md](perf-feedback.md) 受控路由 |
