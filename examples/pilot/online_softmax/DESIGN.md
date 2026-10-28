# online_softmax 算子设计文档（GPU TileLang → Ascend A5 迁移）

> 迁移任务（migration-plain）。源算子：`examples/online_softmax/online_softmax.py`（GPU TileLang，device="cuda"）。目标：tilelang-fork ascend 后端（A5/950，`--npu-arch=dav-3510`），`import tilelang.ascend.language as T`。工件目录：`examples/pilot/online_softmax/`。
>
> 源算子关键特征（Phase M0 亲读源码确认）：`T.Kernel(ceildiv(M, BLOCK_M), threads=128)`（GPU block×thread 两级并行）、`T.alloc_fragment` 全中间量寄存器驻留、`T.reduce_max/reduce_sum(dim=1, clear=True)` warp 归约、`T.exp2` 换底（scale=log2e）、`T.Pipelined` 块循环、行级 online lse 流式更新、X 二遍 GM 读取。

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 `.agents/skills/tilelang-op-design/references/migration-analysis.md`）：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§10 的所有设计决策。

### 0.1 源算子语义（做什么）

- **数学语义**：行级 softmax（规约轴 = dim=1，行内）：`Y[i,j] = exp(X[i,j]) / Σ_k exp(X[i,k])`，输出与输入同 shape 同布局。数值稳定化：减行最大值 + log2 域 lse 形式——源实现以 `y = exp2(x·s − lse)` 输出，其中 `s = log2(e) = 1.44269504`（源码字面量为截断值，完整 log2e = 1.4426950408889634…，偏差 ~8.9e-10 相对量级），`lse = m·s + log2(Σ exp2(x·s − m·s))`，m 为行内（分块）最大值。
- **规约语义**：规约轴 N（dim=1），无 keepdim 问题（输出同 shape）。累加顺序：块内 `T.reduce_sum` 硬件树形归约 + 跨块 online 合并（`lse ← max_x·s + log2(exp2(lse − max_x·s) + sum_exp_x)`）——浮点累加为非固定顺序分组，中间累加 dtype = fp32（`accum_dtype = T.float32`）。
- **dtype 语义**：输入/输出 fp16；中间 `exp_x`/`sum_exp_x`/`lse` 均 fp32；`max_x` 为 **fp16**（fp16 值的 max 仍是 fp16 可精确表示，无损）；表达式提升：`x(fp16)·scale(fp32 常量)` → fp32 参与后续运算，`exp2` 在 fp32 求值后赋给 fp16 的 `y` 时隐式降精度。
- **边界语义**：NaN 输入 → 行内逐元素传播（max/exp 链上 NaN 污染整行输出 NaN）；+inf 参与：`x − m = inf − inf = NaN` → 整行 NaN（与 torch softmax 对 +inf 行的 NaN 行为一致，以 golden 实测为准）；全 −inf 行：`0/0` 语义 → NaN；N=1 行 → y=1；源 host 侧无显式 padding，整除性由默认 shape（M=N=8192，BLOCK_M=1，BLOCK_N=8192）天然满足。
- **语义保持基线**：§8.1 golden 以本节语义为唯一依据（源仓自身测试基准 `X.softmax(dim=1)`，源文件 L53）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| X | (M, N)，M/N 动态可参数化（主配置 8192×8192） | float16 | 输入，行主连续（row-major contiguous），kernel 内 `T.Tensor([M, N], dtype)` |
| Y | (M, N)（= 输入同 shape；主配置 (8192, 8192)） | float16 | 输出，同布局无转置，kernel 内 `T.empty([M, N], dtype)` 新分配，非 in-place |

host 侧无预处理/后处理（无 im2col/reshape/cast），仅构造张量、调用、对比、计时。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每个计算语句归入某步骤，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `T.fill(lse, -T.infinity(accum_dtype))` | lse fragment | — | online 累积器初始化（−inf 幾元） |
| 2 | `T.copy(X[i_m*BLOCK_M, i_n*BLOCK_N], x)`（pass-1 循环内） | GM[X] | x fragment (fp16) | 行块载入 |
| 3 | `T.reduce_max(x, max_x, dim=1, clear=True)` | x | max_x (fp16) | 行内块最大值 m |
| 4 | `exp_x[i,j] = T.exp2(x[i,j]*scale - max_x[i]*scale)`（`T.Parallel(BLOCK_M, BLOCK_N)`） | x, max_x, scale | exp_x (fp32) | 稳定化指数 e = 2^(x·s − m·s) |
| 5 | `T.reduce_sum(exp_x, sum_exp_x, dim=1, clear=True)` | exp_x | sum_exp_x (fp32) | 行内块指数和 S_c |
| 6 | `lse[i] = max_x[i]*scale + T.log2(T.exp2(lse[i] - max_x[i]*scale) + sum_exp_x[i])`（`T.Parallel(BLOCK_M)`） | lse, max_x, sum_exp_x | lse (fp32) | online lse 跨块合并 |
| 7 | `T.copy(X[i_m*BLOCK_M, i_n*BLOCK_N], x)`（pass-2 循环内，**重读 GM**） | GM[X] | x | 二遍重载行块 |
| 8 | `y[i,j] = T.exp2(x[i,j]*scale - lse[i])`（`T.Parallel(BLOCK_M, BLOCK_N)`） | x, lse | y (fp16，赋值时降精度) | 归一化输出（分母并入指数，无除法） |
| 9 | `T.copy(y, Y[i_m*BLOCK_M, i_n*BLOCK_N])` | y | GM[Y] | 写回输出 |

**数据流与内存访问模式**（源 GPU 视角）：

```
GM[X] --T.copy--> register x[BLOCK_M, BLOCK_N] (fp16 fragment)
  ├─ warp reduce_max(dim=1) --> max_x[BLOCK_M] (fp16, register)
  ├─ exp2(x·s − m·s) --> exp_x[BLOCK_M, BLOCK_N] (fp32, register)
  │    └─ warp reduce_sum(dim=1) --> sum_exp_x[BLOCK_M] (fp32, register)
  │         └─ lse 标量更新 (fp32, register)
  └─ pass-2: GM[X] --T.copy(重读)--> x --> exp2(x·s − lse) --> y (fp16)
       --T.copy--> GM[Y]
```

全程寄存器驻留（`T.alloc_fragment`，无 SMEM），中间结果不落 GM；**X 被从 GM 读取两遍**（pass-1 与 pass-2 各一次），Y 写一遍。

**循环与并行结构**：grid = `ceildiv(M, BLOCK_M)` = 8192 个 block（BLOCK_M=1，每 block 负责 1 行）；每 block `threads=128`（行内 8192 元素 → 64 元素/线程）；两个 `T.Pipelined(ceildiv(N, BLOCK_N))` 块循环（主 shape 下 BLOCK_N=N=8192 → 各退化为 1 次迭代，软件流水不生效）；内层 `T.Parallel(BLOCK_M, BLOCK_N)` 逐元素并行。BLOCK_N=8192 是源码默认参数，结构上支持 N > BLOCK_N 的分块流式（online lse 即为此设计）。

**host 侧逻辑**：构造 `X = torch.randn(8192, 8192, fp16, device="cuda")` → 调用 kernel → `X.softmax(dim=1)` 精度对比（rtol/atol 1e-2）→ `do_bench` 计时对比 torch。无 pass_configs 特殊配置（GPU 默认调度）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | exp→exp2 换底（scale=log2e 常量折入） | 使用 GPU SFU 的原生 exp2 快路径 | `e^t = 2^(t·log2e)`，乘常数换底 | GPU SFU（exp2 原生指令，exp 需软件换底） | 硬件强相关（A5 需复核 exp/exp2 代价关系） |
| 2 | 除法消除（lse 并入指数） | 消 N 次除法 | `y = exp2(x·s − lse)`，分母 S 以 `log2 S` 并入指数 | 无（纯数学恒等变形） | 可移植 |
| 3 | online lse 流式更新 | N > BLOCK_N 时无需驻留整行/O(N) 中间缓冲 | 分块 running (max, sum) + 指数域合并公式 | 无（纯算法层，flash-attention 同构） | 可移植 |
| 4 | 寄存器驻留（全部中间量 `T.alloc_fragment`） | 中间结果零 SMEM/GM 往返 | fragment = 每线程寄存器切片 | GPU 大寄存器堆（128 threads × 高水位寄存器） | 硬件强相关 |
| 5 | `T.Pipelined` 块循环 | 拷贝/计算重叠（软件流水） | 编译器多缓冲流水 | GPU 异步拷贝引擎 | 有等价能力（需按 Ascend 容量重算） |
| 6 | `T.reduce_max/reduce_sum(clear=True)` 硬件归约 | 低开销跨线程行内归约 | warp shuffle 树形归约 | warp（32 线程）shuffle | 硬件强相关 |
| 7 | BLOCK_M=1 行级 grid + threads=128 行内协作 | 8192 行并行度 + 行内 64 elem/线程 ILP | block×thread 两级并行 | CUDA 线程模型/SM 调度 | 硬件强相关 |
| 8 | fp32 累加 + max 存 fp16 | 数值安全（exp 溢出防护 + 归约累加精度） | dtype 分层 | 无（语义层） | 可移植 |
| 9 | pass-2 二遍重读 X | 规避双倍寄存器压力（x + exp_x fp32 同驻留两遍超寄存器预算） | 以一次 GM 重读换寄存器 | GPU 寄存器容量/带宽权衡 | 硬件相关（A5 上 UB 驻留可消除重读 → 重新设计） |

> 识别不出来 ≠ 不存在。未列出的优化会在迁移中被静默丢弃，导致性能回退无法追溯。上表 9 项在 §0.5 逐项处置，无静默丢弃。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（三层模型：语义层/算法层/优化层；四态处置：保留/等价替换/重新设计/舍弃）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（row softmax、fp32 累加、NaN/inf 行为） | 语义 | — | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| 减 max 稳定化 | 算法 | 无 | — | **保留** | 纯数学（`.agents/skills/tilelang-op-design/references/algorithm-candidates.md` ALG-softmax R1 必选项） |
| online lse 流式结构 | 算法 | 无 | — | **保留**（作为 N > UB 驻留上限时的泛化路径，§0.6 R2） | 纯算法层（migration-analysis.md §5.3 online 算法行：保留） |
| #1 exp2 换底 | 优化 | GPU SFU exp2 原生 | A5 上 exp 与 exp2 同为注册 intrinsic 且同 special_func 代价档 | **舍弃（A5 上）** | `src/ascend/codegen/intrin_rule_ascend.cc` L34-49：exp/exp2 均注册（fp16 走 h 前缀派发）；`src/ascend/transform/estimate_latency.cc` L1267：exp2/log2/exp 同档 special_func——换底在 A5 无单位代价收益，反多 1 次乘法/elem。**优化意图（廉价 exp 求值）由原生 exp intrinsic 承接**（§1.6.1 O3），非静默丢弃 |
| #2 除法消除（lse 并入指数） | 优化 | 无 | 有 | **保留（形态更新）** | 意图（无除法）由「e 驻留复用 + 每行一次倒数」更廉价承接：归一化 0 次 special/elem（§1.6.1 O1+O2） |
| #4 寄存器 fragment 驻留 | 优化 | GPU 寄存器堆 | 有（SimtVF 内 `T.alloc_fragment`，寄存器切片语义一致） | **等价替换** | `examples/ascend/example_rmsnorm.py` L55（fragment 向量化装载 + 跨循环复用）；tilelang-ascend skill §4.1 |
| #6 warp shuffle 归约 | 优化 | warp 32 线程 | 有（`T.reduce_max/reduce_sum` fragment 维归约，fp16/fp32 硬件直连） | **等价替换** | `examples/ascend/example_per_token_cast_to_fp8.py` L68（SimtVF 内 `T.reduce_max(frag, out, dim=1)` 同构先例）；`testing/ascend/language/test_tilelang_ascend_reduce.py`（sum/max/min 回归）；skill §10.11（硬件直连 dtype：fp16/fp32/int32/uint32——本算子归约走 fp32，直连） |
| #5 `T.Pipelined` 块循环 | 优化 | GPU 异步拷贝 | 有（`T.Persistent(num_stages=)` + AutoSchedule multibuffering） | **等价替换** | `examples/ascend/example_per_token_cast_to_fp8.py` L45（T.Persistent + num_stages 流水）；skill §7 |
| #7 block×thread 两级并行（threads=128） | 优化 | CUDA 线程模型 | 无直接等价（`T.Kernel` 一维、无 threads 参数） | **重新设计** | → §0.6 R1：一维 `T.Kernel(56)` + `T.SimtVF(threads=1024)` 线程域 + persistent 行块调度（ascend-constraints §1；skill §2.1/§2.2） |
| #9 二遍 GM 重读 X | 优化 | GPU 寄存器容量权衡 | UB 216KB〔SimtVF〕可驻留整行（N=8192 fp16 行 16KB） | **重新设计** | → §0.6 R2：UB/VRF 行块驻留单遍读（GM 流量 −33%、special −50%；调研结论 §1.6.0 驱动；`CONST-capacity-a5`） |
| BLOCK_M=1 行级 grid（8192 blocks） | 优化 | GPU SM 调度假设 | 56 AIV persistent 行批更契合 | **重新设计** | → §0.6 R1（分核三要素见 §5.5） |
| host device="cuda" 环境 | 环境 | CUDA | 有（device="npu"） | **等价替换** | AGENTS.md API Convention（torch on NPU + do_bench(backend="msprof")） |

**判定统计**：保留 4 项（语义/稳定化/online 结构/除法消除意图）+ 等价替换 5 项（fragment 驻留/归约/流水/host 环境）+ 重新设计 3 项（R1 并行结构、R2 数据驻留、R3 调度）+ 舍弃 1 项（exp2 换底，A5 同档证据 + 意图由 O3 承接）。

### 0.6 NPU 算法重设计

**重设计项 R1: 并行结构（block×thread 两级并行 → 一维 grid + SimtVF + persistent 行批）**

- **源方案**：grid = 8192 blocks（BLOCK_M=1 行/block）× threads=128（行内 64 元素/线程），`T.Pipelined` 块循环；性能意图 = 行级大规模并行 + 行内线程协作 ILP。
- **NPU 新算法**：一维 `T.Kernel(num_blocks)`，num_blocks = 56（= vector_core_num 实查，见 §5.5）；`for t in T.Persistent([num_tiles], 56, core_id, num_stages=2)` 行块调度，每 tile 处理 BM 行（BM ∈ {4,2,1} host 自适应、整除 M，主 shape BM=4 → num_tiles=2048）；元素并行在 `T.SimtVF(threads=1024)` 线程域内以 `T.Parallel(BM, TILE)` 表达，行内归约用 `T.reduce_max/reduce_sum(dim=1)`（fragment 维归约）。
- **语义保持论证**：softmax 行间完全独立 → 行批大小/行到核的映射变化不影响数值；行内归约由 warp shuffle 树变为 asc_reduce 硬件树——浮点累加分组变化：max 精确无舍入（fp16 值域内 max 可精确表示），sum 为 fp32 树形归约（误差 ≤ 树深×ulp，量级 13×2^-24 远小于 fp16 输出容差 1e-2）；NaN/inf 逐元素传播语义不变（§8.2 L2 边界测试覆盖）。
- **依据**：`.agents/skills/tilelang-ascend/SKILL.md` §2.1/§2.2；`examples/ascend/example_per_token_cast_to_fp8.py` L44-51（T.Kernel(N_CORES) + T.Persistent + SimtVF(threads=1024) + reduce_max 完整同构形态）；`examples/ascend/example_vecadd.py` L21-31（SimtVF threads 与 Parallel 分布）。

**重设计项 R2: 数据驻留（二遍 GM 重读 → UB 行块驻留单遍读 + exp 单次求值复用）**

- **源方案**：pass-1 读 X 求 lse（exp 求值第一次），pass-2 **重读 X** 以最终 lse 重算 exp（exp 求值第二次）；性能意图 = 以 6 B/elem GM 流量 + 2 次 special/elem 为代价规避 GPU 寄存器压力（x fp16 + exp_x fp32 双驻留两遍超预算）。
- **NPU 新算法**：x_ub（BM×N fp16 = 64KB @BM4/N8192，双版本 128KB）UB 驻留行块，行内三段式：`reduce_max` → `e_frag = exp(x − m)`（fp32，**寄存器驻留**）+ `reduce_sum` → `y = cast(e·inv_s, fp16)` 原位覆写 x_ub → 单次 GM 写。X 只从 GM 读一遍：GM 流量 4 B/elem（源 6，−33%），special_func 1 次/elem（源 2，−50%）。e_frag（BM×TILE fp32 = 128KB）分布在线程域寄存器（1024 threads → 48-64 regs/thread，rmsnorm 同量级）。
- **语义保持论证**：`y = e·(1/S) ≡ e/S ≡ exp(x − m − ln S) ≡ 源式 exp2(x·s − lse)`（倒数分解 + 换底恒等变形）；差异仅为 fp32 中间舍入路径（≤ 数 ulp），fp16 输出容差内——`examples/pilot/online_softmax/verify_equiv.py` V1/V2/V3 机器验证全部通过（§1.6.1 结果表）；边界语义：masked 载入的 pad lane 取 −inf（max 幺元）→ e_pad = exp(−inf − m) = 0（sum 幺元）→ 存储侧掩码不写 pad lane；NaN/±inf 行为与源一致（全 −inf 行 → m=−inf → arg=NaN → y=NaN，同 torch）。
- **泛化路径（N > UB 驻留上限，TILE 上限 32768）**：保留源 online 两遍流式结构（BLOCK_N=8192 分块 + online lse 更新 + pass-2 重算 exp），并按 §1.6.1 O6 将合并公式稳健化（flash 形式 max 基重缩放 + fp16 有限哨兵）——源 online 优化手段（§0.4 #3）**未被丢弃**，在驻留不可行的 shape 区间接管。
- **依据**：调研结论 §1.6.0（R3/R4：驻留候选流量与 special 双占优）；`CONST-capacity-a5`（UB 248KB/SimtVF 216KB，90% 折扣 194.4KB ≥ 128KB）；`examples/ascend/example_rmsnorm.py` L72-78（x_ub 原位覆写 reuse_input 先例）。

**重设计项 R3: 调度（T.Pipelined 块循环 → T.Persistent + AutoSchedule）**

- **源方案**：`T.Pipelined(ceildiv(N, BLOCK_N))` 软件流水（主 shape 退化 1 次迭代，不生效）。
- **NPU 新算法**：外层 `T.Persistent([num_tiles], 56, core_id, num_stages=2)`（tile 间流水：tile t+1 的 MTE2 预取与 tile t 的 VALU 计算重叠），x_ub 双版本由 `T.annotate_buffer_versions` 显式固定；MTE2→Vector→MTE3 的跨引擎依赖由 AutoSchedule 自动插 flag 对（§7）。
- **语义保持论证**：纯调度变换（buffer 版本交替 + 同步边），不改任何数值路径。
- **依据**：`examples/ascend/example_per_token_cast_to_fp8.py` L45-47（T.Persistent + num_stages + annotate_buffer_versions 三件套）；skill §7（AutoSchedule 三模式）。

### 0.7 标杆实现

- 源算子路径：`examples/online_softmax/online_softmax.py`（golden 实现与语义解读的唯一来源）。
- 源仓参考实现/测试基准：`X.softmax(dim=1)`（torch，源文件 L53/L55，rtol/atol 1e-2）——§8.1 golden 直接采用（torch on NPU），保证验证独立性（不复刻 §0.6 NPU 算法）。

## 1. 概述

### 1.1 算子名称

online_softmax（迁移自 GPU TileLang `examples/online_softmax/online_softmax.py`）

### 1.2 功能描述

行级数值稳定 softmax：对 [M, N] fp16 输入沿 dim=1 做归一化指数变换，输出同 shape fp16；支持 M/N 参数化（主配置 8192×8192），N 超出片上驻留上限时自动切换 online 流式结构（源算子泛化能力保留）。

### 1.3 数学公式

$$
Y[i,j] = \frac{\exp(X[i,j] - m_i)}{\sum_{k=0}^{N-1} \exp(X[i,k] - m_i)}, \quad m_i = \max_j X[i,j]
$$

（与源式 `Y = exp2(X·s − lse)`、`lse = m·s + log2(Σ exp2(X·s − m·s))`、`s = log2e` 数学恒等——换底 + 倒数分解，等价性机器验证见 §1.6.1。）

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

**主路径（N ≤ 驻留上限 TILE_MAX=32768，含主 shape 8192）——UB/VRF 行块驻留三段式**（§0.6 R1/R2/R3 落地）：

1. `T.copy` 将 X 的 BM×N 行块 GM→UB（x_ub，fp16，双版本流水）；
2. SimtVF 线程域内 masked 载入 x_frag（BM, TILE）（TILE = next_pow2(N)，pad lane = −inf，fp32）；
3. `T.reduce_max(x_frag, m_frag, dim=1)` 行最大（fp32，精确）；
4. `e_frag = T.exp(x_frag − m_frag)`（fp32，寄存器驻留，**单次求值**）+ `T.reduce_sum(e_frag, s_frag, dim=1)` 行和（fp32）；
5. `inv_frag = 1/s_frag`（每行一次倒数）；
6. 原位归一化：`x_ub[r,i] = T.cast(e_frag[r,i]·inv_frag[r], fp16)`（i < N 掩码）；
7. `T.copy` x_ub → Y 行块（单次 GM 写）。

**与源算法的结构差异**（逐条注明来源）：① X 单遍 GM 读（源二遍）——§0.6 R2 / 调研 §1.6.0；② exp 单次求值 e 驻留复用（源两遍各一次）——§1.6.1 O1；③ e-base 直接 `exp(x−m)`，无 log2e 换底（源 exp2+scale）——§1.6.1 O3；④ 归一化 = e·倒数（源 lse 并入指数）——§1.6.1 O2；⑤ persistent 56 核行批 + SimtVF（源 8192 blocks × threads=128）——§0.6 R1；⑥ max 归约 fp32 化（源 max_x fp16——fp16 值的 max 在 fp32 仍精确，且 reduce 回归先例覆盖 fp32）——§3.2。

**泛化路径（N > 32768）——online 两遍流式**（源结构保留 + 稳健化）：pass-1 按 BLOCK_N=8192 分块流式维护 running (m_run, s_run)（flash 形式 max 基合并 + fp16 哨兵，§1.6.1 O6），得 lse = m_run + log(s_run)；pass-2 重读分块 `y = exp(x − lse)`。此路径仅特大 N 触发，主 shape 不执行。

### 1.5 数据流图

```
GM[X] --T.copy (MTE2)--> UB[x_ub (BM,N) fp16, 2 版本]
  --SimtVF masked T.Parallel 载入--> VRF[x_frag (BM,TILE) fp32]
  → VRF reduce_max → m_frag (BM,) fp32
  → VRF e_frag = exp(x_frag − m_frag) (BM,TILE) fp32   [寄存器驻留]
  → VRF reduce_sum → s_frag (BM,) fp32 → inv_frag (BM,) fp32
  → VRF y = cast(e_frag · inv_frag, fp16)
  --T.Parallel 原位写--> UB[x_ub 同版本覆写]
  --T.copy (MTE3)--> GM[Y]
```

纯 Vector 算子：无 L1/L0A/L0B/L0C 参与段（无 Cube/T.gemm），NT GEMM 布局不适用。

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先调研（1.6.0，调研四问——输入公式/源算法只是候选之一，迁移任务在源算子解读后、耦合性判定前完成）；再做数学等价的公式优化（1.6.1）；再做循环/标量计算的向量化替代（1.6.2）；最后决策向量化轴与核内布局（1.6.3）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。1.6.1–1.6.3 的分析对象 = §1.6.0 调研选定、经 §0.6 重设计落地的 NPU 算法，不止步于源公式直译。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：同一数学语义（row softmax）的算法族。调研深度：**完整调研**（规约/统计类算子）。执行位置：Phase M0 之后、Phase M1 之前（调研结论是 §0.5/§0.6 的判定输入）。
> 信息源：`.agents/skills/tilelang-op-design/references/algorithm-candidates.md`（ALG-softmax 命中行 + ALG-reduction/ALG-layernorm 的 row-reduction 形态）、`examples/ascend/` 同类实现（example_rmsnorm / example_per_token_cast_to_fp8 / flash_attention）、pattern-library constants（`CONST-valu-model` / `CONST-mte-bandwidth-model` / `CONST-capacity-a5`，A5 冷启动期全部 assumption）、源算法（§0.3/§0.4）、`kb_search "softmax online reduction"`（命中 ALG-attention / ALG-softmax / BP_redundant_work_or_roundtrip）。互联网检索：未使用——本地源（候选表 + examples + 源码结构分析）已覆盖 softmax 族全部已知变体（三遍朴素/两遍/单遍 online/驻留式），结构判据完备，无新增候选空间。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线 = 源算法（两遍流式 exp2-lse） | `y=exp2(x·s−lse)`，lse online 更新，pass-2 重读重算 | — | — | ✅（基线） |
| 1 | 减 max 稳定化 | `e^(x−m)/Σe^(x−m)` | 数学恒等 | 消 fp16 溢出（必选；基线已含） | 基线已含 |
| 2 | exp→exp2·log2e 换底 | `e^t = 2^(t·log2e)` | 数学恒等 | GPU SFU 快路径；**A5 上 exp 与 exp2 同档** | ❌ A5 上反向采纳（去换底，§1.6.1 O3）：`src/ascend/codegen/intrin_rule_ascend.cc` L34-49 两者均注册、`src/ascend/transform/estimate_latency.cc` L1267 同 special_func 档——换底无单位收益且多 1 乘/elem |
| 3 | 除法→乘倒数（每行一次） | `e/S → e·(1/S)` | 容差内等价（1 ulp 级） | N 次除法 → 1 次倒数 + N 次乘 | ✅（O2） |
| 4 | e 驻留单次求值 | pass-2 重算 exp → 复用 e_frag（寄存器驻留） | 容差内等价（舍入路径差，机器验证） | special −50%/elem；GM 读 −33% | ✅（O1，A5 UB 驻留使能——这是对基线最大的结构性差异） |
| 5 | e-base 无换底 | `exp2(x·s−m·s) → exp(x−m)` | 数学恒等（换底逆变换） | −1 乘/elem；消除源 scale 字面量截断误差 | ✅（O3） |
| 6 | 行内单遍 online（running max + 分块校正因子） | `y = e_k·exp(M_k−M_f)/S_f`（K 个行内块） | 容差内等价（多一次重舍入） | −1 遍 cmp 扫描/elem，+K 次 special/行 | ❌ 纳入对比后弃选：收益 ~0.02 cycle/elem（1 cmp @51 档）vs special 0.125 cycle/elem 的主项，~8% 且引入行内块序依赖与额外精度重舍入；列为 Stage 4 观察项不进主选 |
| 7 | log-softmax 直出 | 输出 log 域 | **语义不同**（输出契约是 fp16 概率，非 log） | — | ❌ 语义不符 |

**R2 在线算法**：**有**，三个变体。① 源算法本身即 online 变体（§0.4 #3 是「存在在线算法」的直接证据）——分块 running (max, sum) + 指数域合并，扫描遍数 2（GM），中间缓冲 O(BLOCK_N)；② 行内单遍 online（候选 6：running max + 每块校正因子，扫描遍数 1 GM + 2 UB）；③ 结构判据：softmax 分母可分解为 running 统计量（max/sum）→ 原则上存在在线变体，与 ①② 一致。**收益口径**（UB 驻留可行性）：N=8192 fp16 行 16KB，驻留工作集 x_ub 双版本 128KB + e_frag 寄存器 128KB ≤ 216KB〔SimtVF〕预算（`CONST-capacity-a5`，90% 折扣 194.4KB）——**行块驻留可行，online 结构在主 shape 区间不是必需**；N > 32768（驻留上限）时 online 是唯一结构，予以保留（§1.4 泛化路径）。

**R3 复杂度对比**（逐候选四口径；口径说明：FLOPs/elem 中 mul/sub/add/cmp/cast 各计 1 op、exp/log 计 1 special（折算系数按 `CONST-valu-model` special=8 ops/cycle/AIV 标注为 assumption）、除法计 1 op（div=13 档）；GM Bytes = 输入读 + 输出写 + 中间 GM 往返；扫描遍数 = 同一 GM 数据的读写次数合计；中间缓冲 = 行级驻存需求（UB+VRF））：

| 算法候选 | FLOPs/elem | GM Bytes/elem | 扫描遍数 | 中间缓冲/行 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线（源两遍流式） | ~9 op + 2 special | 6（读 2×2B + 写 2B） | 3（读×2 + 写×1） | O(BLOCK_N) 寄存器 | M 行独立并行；无跨核归约 |
| 驻留三段式 fp32-e（**选定**） | ~6 op + 1 special | 4（读 2B + 写 2B） | 2 | x_ub 2·N·2B（UB）+ e_frag N·4B（VRF） | 同上 |
| 驻留 fp16-e（备选，§1.6.4） | ~6 op + 1 special(fp16) | 4 | 2 | x_ub 2·N·2B + e_frag N·2B | 同上 |
| 行内单遍 online（候选 6） | ~4 op + 1 special + K/N special | 4 | 2 | 同驻留式 | 同上 + 行内块序依赖（削弱 N 向流水） |
| 朴素三遍（max / exp+sum / div，X 驻留） | ~6 op + 1 special + 1 div | 4 | 2 | 同驻留式 | 同上 |

**R4 硬件亲和性评估**（逐候选对照检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水融合 / 跨核结构；负向淘汰依据已附）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（源结构直译） | 纯 Vector ✓（无 Cube 段；但 special 2/elem 撞最窄吞吐档 `CONST-valu-model` special=8 assumption） | 需按 A5 重设驻留层（源寄存器假设不成立 → §0.6 R2） | TILE=8192 pow2 ✓（fp16 128 lane 整除） | ceildiv 静态 ✓ | Pipelined→Persistent 等价替换 ✓ | ⚠️ 可行但 GM 流量（6B）与 special（2/elem）双劣——降级为泛化路径（N > 驻留上限） |
| 驻留 fp32-e | Vector ✓；special 1/elem；div 仅 BM 次/tile | x_ub 128KB ≤ 194.4KB（90% 折扣）✓；e_frag 128KB VRF / 1024 threads = 48-64 reg ✓（rmsnorm 同量级先例） | TILE=8192 % 128 lane = 0 ✓；GM 行 16KB 32B 对齐 ✓ | num_tiles/TILE 编译期常量 ✓ | T.Persistent(num_stages=2) + x_ub 双版本 ✓ | ✅ **选定（主选）** |
| 驻留 fp16-e | 同上；fp16 exp 走 h 前缀 intrinsic（`src/ascend/codegen/intrin_rule_ascend.cc` L27-32），潜在 2× special 吞吐（**未文档化假设**：`CONST-valu-model` 未区分 dtype 粒度） | e_frag 64KB VRF 更省 | 同上 ✓ | ✓ | ✓ | ✅ **备选（实验裁决，§1.6.4）** |
| 行内单遍 online | 同驻留式 | 同驻留式 | ✓ | ✓ | 行内块序依赖削弱流水重叠 | ❌ 弃选：收益 ~8% 且换复杂度与精度重舍入（R1 候选 6） |
| 朴素三遍 | Vector ✓；div N 次/elem（13 档，劣于乘法 51 档） | 同驻留式 | ✓ | ✓ | ✓ | ❌ 弃选：op 数与除法代价均劣于选定式（R3 表） |

**调研结论**: 选定 **「UB/VRF 行块驻留三段式 softmax（fp32-e 主选）」**，关键依据：R3 表——GM 流量 4 vs 6 B/elem（−33%）、special 1 vs 2 /elem（−50%；special_func = 8 ops/cycle/AIV 是 A5 最窄吞吐档，`CONST-valu-model` assumption），两项同时是本算子的主导代价项；R4——纯 AIV 负载、容量 128KB ≤ 194.4KB、pow2 整除全满足、persistent 流水可融合。与基线的结构差异一句话：**二遍 GM 流式（源）→ 单遍 GM 读 + 行块 UB 驻留 + e 寄存器复用（选定）**；N > 32768 时回退 online 流式（基线结构 + O6 稳健化）。源优化手段意图承接：换底（舍弃，A5 同档证据 + O3 承接意图）/ 除法消除（O1+O2 更廉价承接）/ online 结构（泛化路径保留）/ 寄存器驻留（VRF fragment 承接）/ Pipelined（Persistent 承接）/ warp 归约（T.reduce_* 承接）——逐项见 §0.5，无静默丢弃。备选 fp16-e 进实验裁决（§1.6.4）。调研范围：ALG-softmax/ALG-reduction/ALG-layernorm 参考表条目、examples/ascend 四个同类实现、pattern-library constants.md、源码结构分析、kb_search 命中条目。

**设计期估算（D-2）**: 估算下界(相对): **计算项（VALU special）主导，约为流量项的 6.5–12 倍**（代表 workload 8192×8192 fp16：流量 268MB → 读 134.2MB/(56×100B/cycle)≈24.0K cycle + 写 134.2MB/(56×115B/cycle)≈20.9K cycle〔`CONST-mte-bandwidth-model` assumption，MTE2/MTE3 可重叠取 max≈24.0K、串行 44.9K〕；计算 67.1M×(6/51+1/8)/56 AIV ≈ 290K cycle〔`CONST-valu-model` assumption，special=8 档〕；容量项可忽略——UB 驻留无 L0C 往返、无分块重扫）。主选 vs 基线相对比：VALU 时间 ≈ 0.57×（special 2→1/elem 主导），GM 流量 0.67×。**口径不确定披露**：special_func「8 ops/cycle」的 op 粒度（element 级 vs 2048-bit 向量指令级）在 `src/ascend/transform/estimate_latency.h` L58-70 未区分——若为向量指令粒度则瓶颈反转为 MTE 流量主导（~24K cycle）；两种解释下主选结构（同时最小化 special 次数与 GM 流量）均为最优，该常数列 §9 风险点，Stage 4 以 npusim/msprof 实测归因标定。A5 冷启动期 assumption 基估算，常数待标定，容量按 90% 折扣。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定、经 §0.6 R2 落地的驻留三段式算法。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| O1 | exp 单次求值 + 驻留复用 | 源：pass-1 `e1=exp2(x·s−m·s)`（求和用）+ pass-2 `y=exp2(x·s−lse)`（归一化**重算**） | `e=exp(x−m)` 写入 e_frag 寄存器驻留；归一化直接复用 `y=e·inv` | 数学：`exp(x−m)/S = exp(x−m−lnS)` 恒等；数值：复用路径少一次 exp 舍入与一次减法舍入，fp64→fp16 对照机器验证（V3：max_rel 6.16e-08，fp16 违反率 0） | special 2→1 /elem（−0.125 cycle/elem @special=8 档 assumption）；pass-2 的 mul+sub 2 op/elem → 0；GM 读 −1 遍 |
| O2 | 归一化除法→乘倒数 | `y = e / S`（或源的 lse 并指数式，pass-2 需 1 special/elem） | `inv = 1/S`（每行 1 次）；`y = e · inv` | 数学恒等（除法分解）；数值：`e·inv` vs `e/S` 差 ≤1 ulp fp32（V2：max_ulp=1，fp16 违反率 0） | N 次除法（div=13 ops/cycle 档）→ BM 次倒数/tile + N 次乘（51 档）；驻留路径下同时消除源 pass-2 的 1 special/elem |
| O3 | 去 log2e 换底（e-base） | `e = exp2(x·s − m·s)`，s=1.44269504 | `e = exp(x−m)` | 数学恒等：`2^(s·(x−m)) = e^(x−m)`；A5 上 exp 与 exp2 同为注册 intrinsic 且同 special_func 代价档（`src/ascend/codegen/intrin_rule_ascend.cc` L34-49、`src/ascend/transform/estimate_latency.cc` L1267）→ 换底无单位代价收益、纯增 1 乘/elem；且源 scale 字面量截断（1.44269504 vs log2e）引入 ~8.9e-10 相对误差，e-base 无此项（V1：max_rel 6.16e-08，fp16 违反率 0） | −1 乘/elem（−N op/tile）；消除 scale 截断误差源 |
| O4 | 减 max 稳定化（必选确认） | `e^{x}/Σe^{x}` | `e^{x−m}/Σe^{x−m}`，m=max(x) | 指数平移恒等；消 fp16 溢出（e^11 > 65504 上限） | 计算量不变，数值稳定（源已含，语义层保留项） |
| O5 | 尾块标量判定 → 中性值掩码 | 尾块逐元素边界分支 | masked 载入：pad=−inf（max 幺元）；`e_pad = exp(−inf−m) = 0`（sum 幺元）；存储侧 i<N 掩码 | `max(−inf,·)=·`、`+0` 恒等；NaN 语义保持（全 −inf 行 → m=−inf → arg NaN → y NaN，同 torch）；V3 角点行覆盖 | 尾块标量分支 → 纯向量化 masked load（`examples/ascend/example_rmsnorm.py` L56-57 同构） |
| O6 | online 合并稳健化（泛化路径） | 源合并式 `lse ← m_c + log(exp(lse−m_c) + S_c)`（m_c 基） | flash 形式：`m_run = max(m_run, m_c)`；`s_run = s_run·exp(m_run_old−m_run) + S_c·exp(m_c−m_run)`；`lse = m_run + log(s_run)`；块 max 钳位 `max(m_c, −65504)`（fp16 有限哨兵） | 数学：两种合并式对良性数据恒等（均为 log-sum-exp 递推的代数变形）；稳健性：源式在 ① 块间 max 间隙 > 88.7（fp16 数据域可达 131008）时 `exp(lse−m_c)` fp32 溢出→inf→全 0 输出、② 块内全 −inf 时 `exp(−inf−(−inf))=NaN`——机器验证诊断（V4b：源式角点 fp16 违反率 13.6%〔fp64〕/ 18.2%〔fp32〕）；稳健式全部 exp 参数 ≤ 0 无溢出、哨兵钳位消 NaN（V4：max_rel 1.32e-12，fp16 违反率 0） | 正确性收益（角点与 torch/golden 一致）；计算量同级（每块每行 ~4 标量 op）；属「log-sum-exp 改写」稳定化变形 |

**优化结论**: **采纳 5 项（O1 驻留复用、O2 倒数乘、O3 去换底、O5 中性值掩码、O6 online 合并稳健化）+ 1 项必选确认（O4 稳定化）**。优化后主路径公式（§3.1 拆解的唯一输入）：

```
m_i   = max_j x[i,j]                     # fp32，精确
e_ij  = exp(x[i,j] − m_i)                # fp32，单次求值，寄存器驻留
S_i   = Σ_j e_ij                         # fp32 树形归约
Y_ij  = fp16(e_ij · (1/S_i))             # 倒数乘 + 降精度输出
```

泛化路径（N > 32768）：`m_run/s_run` flash 形式 online 递推（含哨兵钳位）→ `lse = m_run + log(s_run)` → `Y = fp16(exp(x − lse))`。

**等价性机器验证（D-1）**: 已生成并执行 `examples/pilot/online_softmax/verify_equiv.py`（fp64 参照 + 角点值：全等行/单主导/±65504/subnormal/深度下溢/±inf/NaN/gap>88/非 pow2 长度，torch CPU；两式均 fp64 计算后按 fp16 舍入对照，atol=rtol=1e-2）。执行结果（2026-10-16，tilelang-fork 工具链）：

| 验证项 | 对照（候选 vs 基线/精确） | 最大 ulp 差（fp64 位级） | 最大相对差 | fp16 违反率（@1e-2） | 结论 |
|--------|--------------------------|--------------------------|------------|---------------------|------|
| V1（O3） | exp(x−m) vs exp2(x·s−m·s) | 4.6e8（角点 subnormal 区，绝对差 ≤1e-24 量级） | 6.16e-08 | 0.000000 | EQUIV_PASS |
| V2（O2） | e·(1/S) vs e/S 全链 | 1 | 2.22e-16 | 0.000000 | EQUIV_PASS |
| V3（O1） | 驻留链 fp16 输出 vs 源链 fp16 输出 vs 精确 softmax | 4.6e8（同 V1 subnormal 注记） | 6.16e-08 | 0.000000 | EQUIV_PASS |
| V4（O6） | 稳健 online 合并 vs 全行直接 | 11605（深度下溢角点，绝对差 subnormal 量级） | 1.32e-12 | 0.000000 | EQUIV_PASS |
| V4b（诊断，不采纳） | 源朴素合并 vs 精确 | — | 发散 | 0.136（fp64）/ 0.182（fp32） | 角点发散——O6 稳健化必要性证据（非采纳项） |
| V5（备选预估，不采纳） | fp16-e 模拟链 vs 精确 | — | 2.26e-03（有效域） | 0.000000 | §1.6.4 备选精度守卫内（非采纳项） |

> ulp 注记：V1/V3/V4 的大 ulp 数字全部来自深度下溢角点行（exp 值进入 subnormal 区，ulp 间距极小，绝对差 ≤1e-24），判据以最大相对差与 fp16 违反率为准（均 0）。**全部采纳项 EQUIV_PASS；诊断项 V4b 证实源朴素合并在角点发散，O6 稳健化必要。**

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | x_frag masked 载入（含 fp16→fp32 提升） | 逐元素边界分支载入 | `T.Parallel(BM, TILE)` + `T.if_then_else(i<N, T.cast(x_ub[r,i],"float32"), −T.infinity(T.float32))`（`examples/ascend/example_rmsnorm.py` L56-57 同构掩码载入） | ✅ | — |
| 2 | 行内 max | （源 warp shuffle） | `T.reduce_max(x_frag, m_frag, dim=1, clear=True)`（`examples/ascend/example_per_token_cast_to_fp8.py` L68 SimtVF 内 fragment 维归约同构） | ✅ | — |
| 3 | 逐元素 e = exp(x−m) | 标量循环 exp | `T.Parallel(BM, TILE)` 向量化（编译器自动向量化；rmsnorm L66 rsqrt 同构；exp 为注册 intrinsic） | ✅ | — |
| 4 | 行内 sum | 标量累加循环 | `T.reduce_sum(e_frag, s_frag, dim=1, clear=True)`（`testing/ascend/language/test_tilelang_ascend_reduce.py` sum 回归覆盖 fp32 fragment） | ✅ | — |
| 5 | 每行 inv = 1/S | 每行标量除法 | `T.Parallel(BM)`（BM ≤ 4 行维向量化） | ✅（行维） | 依赖 reduce 终值（sum 定稿后才能求倒），无法并入元素循环——tile 级标量（BM 个/tile vs BM·TILE 元素，占比 1/TILE），非逐元素热点 |
| 6 | y 原位写回 | 标量写 | `T.Parallel(BM, TILE)` 向量化 store（`T.cast(e·inv, "float16")` 内联） | ✅ | — |
| 7 | tile 调度循环 | — | `T.Persistent([num_tiles], 56, core_id, num_stages=2)` | 结构循环 | 块级任务映射/调度（非计算循环） |
| 8 | 泛化路径 chunk 循环 ×2 | — | `T.serial(N_CHUNKS)` | ❌（保留串行） | online 递推的循环携带依赖（running 统计量跨块串行）——在线算法本质结构；仅当行块无法整体驻留（N > 32768）才执行；主 shape 不走此路径 |
| 9 | 泛化路径 lse/合并标量更新（每行每块 ~6 标量 op，含 2 special） | 逐行标量 | 无逐元素等价向量 API | ❌（保留） | 依赖前块递推终值（tile 级顺序依赖）；量级 BM 个/块 vs BM·BLOCK_N 元素/块（占比 1/8192），非热点 |
| 10 | host 侧 BM/TILE/路径选择 | Python 算术 | — | n/a | host 元数据计算，不在 kernel 内（决策树 §1 维度自推导） |

**向量化结论**: 逐元素计算已 100% 向量化（#1/#2/#3/#4/#6 全部 `T.Parallel`/`T.reduce_*`，API 均有 examples 与 testing 佐证）；保留 3 处非向量化点——#5 tile 级每行倒数（依赖归约终值，占比 1/TILE）、#8/#9 泛化路径 online 顺序依赖（算法本质结构，主 shape 不执行），理由类别 = tile 级标量 + 顺序依赖，均非「实现简单」类借口；与 §6 循环结构一致（§6 无逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（[M, N] 行主进/行主出），核内布局与 lane 映射是设计变量。规约类算子必选候选：水平归约（lane 映射到归约轴 N）vs 垂直扫描（lane 映射到独立行 M、串行步进 N），另加分块/跨核变体。

**轴质量评分**（对选定轴 N）：

| 评分项 | 判定 |
|--------|------|
| 整除性 | TILE=8192 % 128（fp16 lane）= 0 ✓ / % 64（fp32 lane）= 0 ✓（主 shape 零尾 lane；非 pow2 N 经 TILE 填充 + masked 载入，尾 lane 浪费 < 2×） |
| 尾 lane 浪费率 | 主 shape 0%；N=8191 → TILE=8192 浪费 1/8192 ≈ 0.01% |
| 累加链形态 | 归约维 N = stride-1 连续向量维（`T.reduce_*(dim=1)` on (BM, TILE) fragment）；无跨步系数 |
| repack 代价 | 无（I/O 原生行主连续，核内布局与 I/O 同构；x_ub 紧凑 (BM,N)，fragment pow2 填充——`examples/ascend/example_rmsnorm.py` L42 注释先例：仅 fragment 需 pow2，DMA 紧凑） |
| UB 容量影响 | 行块驻留 2·BM·N·2B（x_ub 双版本）= 128KB ≤ 194.4KB（§4.5 预算表） |

**候选矩阵**（规约类候选全枚举，弃选给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | 行主 [M,N] 原生 + **水平归约**（lane→N 连续轴，BM 行批驻留） | N | 无 | 连续 GM 行 16KB（32B 对齐 ✓）；归约维 = stride-1 向量维；rmsnorm + per_token 双先例 | ✅ **主选** |
| 2 | 垂直扫描（lane→M 行，N 向串行步进） | M | 无 | N=8192 串行步进/行——向量宽度利用率 BM/128 < 4%，行内并行度归零；吞吐劣化 >25×（估）；repo 无先例 | ❌ 串行化 N 维，量化劣化不可接受 |
| 3 | 2D 分块 + 跨核行归约（M/BM × N/BN grid，partial 落 GM + 跨核 flag 合并） | N（块内） | partial 归约中间量 GM 往返 | 行已可驻留（16KB/行），跨核归约新增 2 遍 partial GM 读写 + 跨核同步（R3 流量 +50% 起步）；UB 容量足够无需分块 | ❌ 违反流量最小化（R3）；容量不构成约束 |
| 4 | 泛化路径：行主 + BLOCK_N=8192 分块水平归约 | N（块内） | 无 | 方案 1 在 N > 驻留上限时的退化形态（online 结构保留源泛化能力） | ✅（泛化路径专用） |

**源码轴选择对照**（迁移任务必做）：源 GPU 也是 N 轴并行（threads=128 → 64 elem/线程沿 N）+ BLOCK_M=1 行级 grid——轴结论一致（N 是唯一高带宽连续轴，两代硬件共识），但**承载结构完全重设计**：GPU thread 域/warp → SimtVF 线程域 + `T.reduce_*`；GPU 行级 grid → persistent 56 核行批（§0.6 R1）。GPU 轴选择是输入而非结论，已独立复核。

**布局决策结论**: 选定方案 #1：核内布局 = x_ub (BM, N) fp16 紧凑 + x_frag/e_frag (BM, TILE) pow2 填充（fp32）；向量化轴 = **N（行内连续轴）**；repack = 无。该结论同步落入 §3.3 伪代码与 §6 循环结构（Parallel 内层维 = TILE/N 轴，buffer 形状 = §4.3）；弃选方案量化理由见候选矩阵（#2 串行化劣化 >25×、#3 流量 +50%）。

#### 1.6.4 实验裁决三件套（fp16-e 备选）

> 主选（fp32-e）与备选（fp16-e）的取舍判定裕度落在未实证常数区间（special_func 吞吐的 dtype 粒度——`CONST-valu-model` 未区分 fp16/fp32，A5 冷启动 assumption），按 skill Phase 2 第 3 项强制产出三件套：

- **主选方案**：fp32-e 驻留三段式（本 DESIGN §3–§7 全部按此展开）——e_frag fp32，`T.exp` fp32 求值，与源精度链一致（§1.6.1 V3）。
- **备选方案**（1 个，结构完整可实现）：fp16-e 变体——`e_frag` 改 fp16：arg 以 fp16 舍入（`x_frag 保持 fp32 载入、arg = x−m 计算 fp32 后 cast fp16` 或 x_frag 直接 fp16 载入），`T.exp` 于 fp16 参数走 h 前缀 intrinsic（`src/ascend/codegen/intrin_rule_ascend.cc` L27-32 dtype 派发）；S 由 `T.reduce_sum` fp16 树形归约（asc_reduce 硬件树，树深 13 → 相对误差 ~0.6%）后升 fp32 求倒数；y = `T.cast(e_frag·inv, fp16)`。buffer 形状/UB 预算不变（e 不入 UB）；循环结构不变；分核三要素不变（56 AIV persistent，BM=4）；精度预估：模拟链最大相对误差 2.26e-03（verify_equiv.py V5，|y|≥1e-3 有效域），fp16 违反率 0——在 5e-3 守卫内。
- **实验裁决计划**：Stage 4 于代表 shape [8192, 8192]（驻留路径）与 [128, 65536]（泛化路径）实测两变体 `do_bench(backend="msprof")` 时延 + 精度（vs golden 最大相对误差）；**判定阈值**：备选时延 < 主选 × 0.90 且 max_rel_err ≤ 5e-3 → 采纳备选（lru_cache 按 shape 分派主选/备选，达成总体最优）；否则主选固化，实测数字经附录补记回写本节；实测反解 special_func fp16 吞吐常数 → 回写 `CONST-valu-model`（Tier 0，含 repro 三件套）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: simt（auto_schedule: yes）

### 2.2 选型理由

1. **迁移默认 + 算子特征**：纯 Vector 算子（elementwise + 行规约 + 超越函数，无 matmul/Cube 段）——决策树「含归约 → simt + reduce/reducer」路径；编程模式偏好由调度方指定 simt，重设计未产生 mixed 需求（无 Cube 工作落位，`T.MixedKernel` 不适用）。
2. **repo 先例**：同族（行规约-归一化）`examples/ascend/example_rmsnorm.py`（simt + fragment + reducer + 行块流水）与 `examples/ascend/example_per_token_cast_to_fp8.py`（simt + `T.reduce_max` + `T.Persistent` + SimtVF(threads=1024)）均为 simt 形态，API 佐证链最完整。
3. **从 simt 起步原则**（决策树 §2 注）：无证实热点前不降级 simd。降级触发条件（Stage 4）：msprof/npusim 证实 special_func 段为瓶颈且 simt 自动向量化不足时，可评估 SimdVF + `T.simd.vexpdif`（融合 exp-sub intrinsic，`tilelang/ascend/language/simd.py` L1182 定义 dst=exp(src0−src1)，`examples/ascend/flash_attention/core.py` L130-133 在用——fp32 同宽形态）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| VF 入口 | `with T.SimtVF(threads=1024)` + `T.Parallel(BM, TILE)` 逐元素段 + `T.reduce_max/reduce_sum(dim=1)` 行归约 |
| 内存分配 | `T.alloc_shared`（UB：x_ub）+ `T.alloc_fragment`（VRF：x_frag/e_frag/m/s/inv） |
| 同步 | auto_schedule: yes——AutoSchedule 管理跨引擎 flag 与 multibuffering（§7） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 优化后公式（主路径）；每步优先向量化映射，与 §1.6.2 结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `m_i = max_j x[i,j]` | 行最大（fp32，fp16 值域内精确） |
| 2 | `e_ij = exp(x_ij − m_i)` | fp32 稳定化指数，单次求值，寄存器驻留 |
| 3 | `S_i = Σ_j e_ij` | fp32 树形归约 |
| 4 | `Y_ij = fp16(e_ij · (1/S_i))` | 倒数乘归一化 + 降精度输出 |

泛化路径替换步骤 3/4：3' `m_run/s_run` flash 形式 online 递推（含 −65504 哨兵钳位）→ `lse = m_run + log(s_run)`；4' `Y_ij = fp16(exp(x_ij − lse))`。

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 搬入 | X 行块 GM→UB | `T.copy(X[base : base+BM, 0:N], x_ub)` | 2D 行块区域拷贝（per_token L49 同构） | simt（MTE2） |
| 掩码载入 | x_ub → x_frag（pow2 填充 + fp32 提升） | `for r, i in T.Parallel(BM, TILE): x_frag[r,i] = T.if_then_else(i < N, T.cast(x_ub[r,i], "float32"), -T.infinity(T.float32))` | pad = −inf（max 幺元） | simt |
| 行最大 | m | `T.reduce_max(x_frag, m_frag, dim=1, clear=True)` | fragment 维归约（per_token L68 同构） | simt |
| 指数 | e = exp(x−m) | `for r, i in T.Parallel(BM, TILE): e_frag[r,i] = T.exp(x_frag[r,i] - m_frag[r])` | fp32 全程 | simt |
| 行和 | S | `T.reduce_sum(e_frag, s_frag, dim=1, clear=True)` | fp32 fragment | simt |
| 倒数 | inv = 1/S | `for r in T.Parallel(BM): inv_frag[r] = 1.0 / s_frag[r]` | 每行一次（div 档，量级 BM/tile） | simt |
| 原位归一化 | y = fp16(e·inv) | `for r, i in T.Parallel(BM, TILE): if i < N: x_ub[r,i] = T.cast(e_frag[r,i] * inv_frag[r], "float16")` | 原位覆写（rmsnorm L72-74 reuse_input 同构） | simt |
| 搬出 | UB→GM | `T.copy(x_ub, Y[base : base+BM, 0:N])` | 2D 行块 | simt（MTE3） |
| tile 调度 | — | `with T.Kernel(56) as core_id: for t in T.Persistent([num_tiles], 56, core_id, num_stages=2):` | persistent 行块（per_token L45 同构） | — |
| 版本管理 | — | `T.annotate_buffer_versions({x_ub: 2})` | 双版本流水（rmsnorm L47 同构） | — |

### 3.3 计算伪代码

> 迁移注意（模板 §3.3）：`T.Tensor` 的 dtype 作为第二个位置参数传入（`T.Tensor((M, N), "float16")`），不用 `dtype=` 关键字——与 examples/ascend 形态一致，避免方言前端解析问题。

```python
import tilelang
import tilelang.ascend.language as T

def online_softmax_fwd(M, N):
    # host 侧静态化（lru_cache 按 shape 特化；决策树 §1 维度自推导，无全局变量）
    TILE = tilelang.next_power_of_2(N)
    BM = 4 if (N <= 8192 and M % 4 == 0) else (2 if (N <= 16384 and M % 2 == 0) else 1)
    resident = N <= 32768                       # 驻留上限（UB 2 版本 + VRF 预算，§5.2）
    if not resident:
        BM, BLOCK_N = (4 if M % 4 == 0 else 1), 8192
        num_chunks = T.ceildiv(N, BLOCK_N)
    num_tiles = T.ceildiv(M, BM)
    threads = min(1024, BM * TILE)

    @T.prim_func
    def main(X: T.Tensor((M, N), "float16"), Y: T.Tensor((M, N), "float16")):
        with T.Kernel(56) as core_id:                     # 一维 grid，无 threads 参数
            x_ub = T.alloc_shared((BM, N), "float16")     # UB，DMA 紧凑（无 pow2 填充）
            T.annotate_buffer_versions({x_ub: 2})         # 双版本：MTE2 预取与计算重叠
            for t in T.Persistent([num_tiles], 56, core_id, num_stages=2):
                base = t * BM
                T.copy(X[base : base + BM, 0:N], x_ub)    # GM → UB（MTE2）
                with T.SimtVF(threads=threads):
                    x_frag = T.alloc_fragment((BM, TILE), "float32")
                    e_frag = T.alloc_fragment((BM, TILE), "float32")
                    m_frag = T.alloc_fragment((BM,), "float32")
                    s_frag = T.alloc_fragment((BM,), "float32")
                    inv_frag = T.alloc_fragment((BM,), "float32")
                    # ① 掩码载入（pad = −inf：max 幺元；后续 e_pad = exp(−inf−m) = 0：sum 幺元）
                    for r, i in T.Parallel(BM, TILE):
                        x_frag[r, i] = T.if_then_else(
                            i < N, T.cast(x_ub[r, i], "float32"), -T.infinity(T.float32))
                    # ② 行最大（fp32 精确；fp16 输入的 max 在 fp32 无舍入）
                    T.reduce_max(x_frag, m_frag, dim=1, clear=True)
                    # ③ 单次求值 e = exp(x − m)，寄存器驻留
                    for r, i in T.Parallel(BM, TILE):
                        e_frag[r, i] = T.exp(x_frag[r, i] - m_frag[r])
                    # ④ 行和（fp32 树形归约）
                    T.reduce_sum(e_frag, s_frag, dim=1, clear=True)
                    # ⑤ 每行一次倒数（tile 级标量，占比 1/TILE）
                    for r in T.Parallel(BM):
                        inv_frag[r] = 1.0 / s_frag[r]
                    # ⑥ 原位归一化（读 e_frag 寄存器 → 写 x_ub，rmsnorm reuse_input 同构）
                    for r, i in T.Parallel(BM, TILE):
                        if i < N:
                            x_ub[r, i] = T.cast(e_frag[r, i] * inv_frag[r], "float16")
                T.copy(x_ub, Y[base : base + BM, 0:N])    # UB → GM（MTE3）
    # 泛化路径（N > 32768）：两遍 T.serial(num_chunks) 流式——
    #   pass-1: 逐块 reduce_max（哨兵钳位 max(m_c, −65504)）→ e/exp → reduce_sum
    #           → flash 形式 m_run/s_run 递推（§1.6.1 O6，全部 exp 参数 ≤ 0）
    #   pass-2: 重读分块 → y = T.cast(T.exp(x − lse), "float16") 原位 → 搬出
    return main
```

（泛化路径与主路径共用 x_ub/SimtVF 骨架，仅循环结构与步骤 ③–⑥ 替换为递推/重算形态，Stage 3 按 §1.4 泛化描述实现。）

### 3.4 API 可行性确认

| API | 佐证（亲读源码/执行 hasattr 实查） | 状态 |
|-----|-----------------------------------|------|
| `T.Kernel(N)`（一维，无 threads） | `.agents/skills/tilelang-ascend/SKILL.md` §2.1；`examples/ascend/example_vecadd.py` L21 | ✅ 已验证 |
| `T.Persistent(domain, wave_size, index, num_stages)` | `tilelang/language/loop.py` L82 定义；`examples/ascend/example_per_token_cast_to_fp8.py` L45/L118 用例 | ✅ 已验证 |
| `T.SimtVF(threads=)` | `examples/ascend/example_rmsnorm.py` L54；per_token L51（threads=1024） | ✅ 已验证 |
| `T.alloc_shared / T.alloc_fragment` | `examples/ascend/example_rmsnorm.py` L41/L55 | ✅ 已验证 |
| `T.copy`（2D 行块 GM↔UB） | `examples/ascend/example_per_token_cast_to_fp8.py` L49（x_groups 行块 → y_ub） | ✅ 已验证 |
| `T.Parallel`（多维 + if 掩码） | `examples/ascend/example_rmsnorm.py` L56/L60/L70 | ✅ 已验证 |
| `T.if_then_else` / `T.infinity` | `examples/ascend/example_rmsnorm.py` L57（if_then_else）；源文件 L23（−T.infinity 同款，hasattr 实查 True） | ✅ 已验证 |
| `T.reduce_max(frag, out, dim=1, clear=)` | `tilelang/ascend/language/reduce_op.py` L95 签名；`examples/ascend/example_per_token_cast_to_fp8.py` L68 用例 | ✅ 已验证 |
| `T.reduce_sum(frag, out, dim=1, clear=)` | `tilelang/ascend/language/reduce_op.py` L111 签名；`testing/ascend/language/test_tilelang_ascend_reduce.py`（fp32 sum 回归） | ✅ 已验证 |
| `T.exp` | `src/ascend/codegen/intrin_rule_ascend.cc` L35 注册（fp16 h 前缀派发 L27-32）；hasattr 实查 True | ✅ 已验证 |
| `T.cast(v, "float32"/"float16")` | `testing/ascend/language/test_tilelang_ascend_simtvf_cast.py` L96 | ✅ 已验证 |
| `T.annotate_buffer_versions` | `examples/ascend/example_rmsnorm.py` L47 | ✅ 已验证 |
| 标量除法 `1.0 / s` | TIR 标量表达式（div=13 ops/cycle 档，`CONST-valu-model`） | ✅ 常规 |

---

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 一维 Kernel / threads= 禁用 | **Yes**（源用 `T.Kernel(grid, threads=128)`） | `T.Kernel(56)` 一维 + `T.SimtVF(threads=1024)` 线程域（§0.6 R1；skill §10 陷阱 2） |
| GPU 专用 API（TMA/cp.async/warp shuffle/cluster） | 部分（源 warp 归约/Pipelined 语义） | warp 归约 → `T.reduce_*`；`T.Pipelined` → `T.Persistent(num_stages=)`（§0.5 等价替换行） |
| GEMM C0 分形对齐（M/N 16 倍数、K 约束） | **No** | 纯 Vector 算子，无 `T.gemm`/Cube 段——C0 分形不适用；行主连续访问，NT GEMM 布局不适用本算子（调度方已确认） |
| L0C 容量上限 | **No** | 无 L0C 分配 |
| UB/L1 容量上限 | **Yes** | §4.5 预算表：x_ub 双版本 128KB + 标量暂存 < 1KB ≤ 194.4KB（216KB×90%，`CONST-capacity-a5`）✓；VRF 256KB/1024 threads = 64 reg/thread（rmsnorm 80 reg 先例内）✓ |
| OOB padding（pad_value=/data_select= 互斥、仅 GM→UB） | **Yes**（非 pow2 N 尾 lane） | 计算侧 masked 载入/存储（−inf/0 中性值，§1.6.1 O5），**不使用** GM→UB DMA padding——DMA 拷贝的行完整存在（X 行宽即 N），无需 32B 行填充；skill §6.1 padding 是可选 opt-in |
| 静态循环边界 | **Yes** | num_tiles/TILE/num_chunks 均由 host 闭包编译期常量化；核内无动态边界循环 |
| 核数无编译期静态常数 | **Yes** | 56 = vector_core_num 运行时实查记录（§5.5）；JIT 按 num_blocks=56 编译 |
| flag id 预算（核内 8/跨核 16） | **No** | auto_schedule: yes，无手动 flag（§7） |
| bf16 扩展的 warp-reduce dtype 限制 | 潜在（本设计 fp16/fp32 均直连） | 本设计归约全 fp32（max/sum），属 `asc_reduce_*` 硬件直连 dtype 集（skill §10.11）；若未来扩 bf16 输入，归约前升 fp32 即可 |

### 3.5.2 参考实现差异说明

**重要**：外部参考实现（GPU 版 TileLang）差异汇总（完整分析见 §0.5/§0.6，此处只列影响 API 选型的关键差异）：

| 差异项 | 参考实现（GPU 源） | 本项目（Ascend） | 转换方案 |
|--------|-------------------|------------------|----------|
| Kernel 维度 | `T.Kernel(ceildiv(M,BLOCK_M), threads=128)`（grid + threads 两级） | `T.Kernel(56)` 一维，无 threads 参数 | persistent 行块 + SimtVF 线程域（skill §10 陷阱 2/3） |
| fragment 语义 | GPU 寄存器堆任意水位 | SimtVF 内 VRF 寄存器切片，需按 threads 预算 | BM×TILE×4B×2 / threads ≤ ~256B/thread 预算（§5.2） |
| 归约 | warp shuffle 树 | `T.reduce_max/reduce_sum`（fragment 维，fp32 硬件直连） | dtype 保持 fp32（skill §10.11） |
| 超越函数 | `T.exp2`（SFU） | `T.exp`（注册 intrinsic，与 exp2 同 special 档） | e-base 直算（§1.6.1 O3） |
| 流水 | `T.Pipelined` 块循环 | `T.Persistent(num_stages=2)` + AutoSchedule multibuffering | §0.6 R3 |
| host 环境 | device="cuda"、do_bench 默认 | device="npu"、`do_bench(backend="msprof")` | AGENTS.md API Convention |

### 3.5.3 本项目同类实现参考

**必须列出**：本项目 `examples/ascend/` 中最相似实现：

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/ascend/example_rmsnorm.py` | 高度相似（行规约-归一化同族） | 行批驻留 UB 预算化、fragment 掩码载入（L56-57）、reduce/reducer 用法、原位覆写 reuse_input（L72-78）、`T.annotate_buffer_versions`（L47）、行块流水（L49） |
| `examples/ascend/example_per_token_cast_to_fp8.py` | 高度相似（行内 reduce_max + 逐元素变换） | `T.Kernel(N_CORES)` + `T.Persistent` + `SimtVF(threads=1024)` + `T.reduce_max(frag, out, dim=1)` 完整同构（L44-68） |
| `examples/ascend/example_vecadd.py` | 结构参考 | SimtVF/SimdVF 双模式对照、`T.Pipelined` 版本注解（L25-31） |
| `examples/ascend/flash_attention/core.py` | 算法参考 | online softmax 的 Ascend 形态（vexpdif 重缩放 L130-133/L169-171；e-base 先例）；Stage 4 simd 降级参考 |
| `testing/ascend/language/test_tilelang_ascend_reduce.py` | 回归佐证 | `T.reduce_sum/max/min` fragment 维归约覆盖（fp32） |
| `testing/ascend/language/test_tilelang_ascend_simtvf_cast.py` | 回归佐证 | `T.cast` 向量化形态（L96） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| X | (M, N)，M/N 动态（主配置 8192×8192） | float16 | 输入矩阵，行主连续（假定 caller 传入连续张量；host 侧仅允许 stride/shape 视图操作） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| Y | (M, N) | float16 | 输出矩阵，与输入同 shape 同布局（无转置），kernel 显式输出参数（out_idx） |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| x_ub | (BM, N)（主 shape (4, 8192)） | float16 | UB（`T.alloc_shared`，双版本） | 行块 DMA 暂存 + y 原位覆写（reuse_input 模式） |
| x_frag | (BM, TILE)（(4, 8192)） | float32 | VRF（`T.alloc_fragment`） | 掩码载入 + fp32 提升的行数据（pad=−inf） |
| e_frag | (BM, TILE) | float32 | VRF | 单次求值的 exp 值，寄存器驻留复用 |
| m_frag / s_frag / inv_frag | (BM,) | float32 | VRF | 行统计量（max/sum/倒数） |

（泛化路径追加：m_run/s_run/lse_frag (BM,) fp32 VRF——running 统计量，无新增 UB buffer。）

### 4.4 内存搬运路径

```
GM[X] --T.copy (MTE2, 2D 行块)--> UB[x_ub (BM,N) fp16 ×2 版本]
  --SimtVF masked T.Parallel--> VRF[x_frag fp32]
  → VRF[e_frag = exp(x−m)] → VRF[m/s/inv]
  → VRF[y = cast(e·inv, fp16)] --T.Parallel 原位--> UB[x_ub 同版本]
  --T.copy (MTE3, 2D 行块)--> GM[Y]
```

无 L1/L0A/L0B/L0C 段（纯 Vector）；无中间量落 GM；X 单遍 GM 读、Y 单遍 GM 写。

### 4.5 UB 内存预算

| Buffer | Shape | dtype | 大小 (Bytes) | 版本 | 小计 |
|--------|-------|-------|-------------|------|------|
| x_ub | (4, 8192) | float16 | 65536 | 2 | 131072 |
| 归约/标量暂存余量 | — | — | ~1024（编译器保留裕量） | 1 | ~1024 |
| **总计** | | | | | **≈132KB / 194.4KB**（UB SimtVF 216KB × 90%，`CONST-capacity-a5` assumption）✓ |

VRF（寄存器，非 UB）：x_frag 128KB + e_frag 128KB = 256KB / 1024 threads = 256B/thread = 64 fp32 reg（rmsnorm 80 reg/thread 先例内 ✓）；溢出时降级 BM=2/threads=512（同 64 reg，§9.2）。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| M | host 闭包参数（`T.Tensor((M, N), ...)` 符号维度），JIT per-shape 编译 + lru_cache | ≥ 1（BM 自适应整除：M%4→BM=4，M%2→BM=2，否则 BM=1） |
| N | 同上 | 1 ~ 任意（N ≤ 32768 驻留路径；N > 32768 泛化流式路径，BLOCK_N=8192） |

### 4.7 JIT 配置

```python
import functools

@functools.lru_cache(maxsize=None)
def get_kernel(M, N):
    prog = online_softmax_fwd(M, N)          # §3.3 闭包，host 侧静态化 BM/TILE/路径
    return tilelang.compile(prog, target="ascend", out_idx=-1)   # Y 为最后参数

# host 调用（维度自推导，无全局变量）：
def online_softmax(X: torch.Tensor) -> torch.Tensor:
    M, N = X.shape
    Y = torch.empty_like(X)
    get_kernel(M, N)(X, Y)
    return Y
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 算子仅含逐元素运算（exp/mul/sub/cast）与行内规约（max/sum），无 matmul——无 Cube 段，全部计算与搬运落 AIV（56 个）。

### 5.2 Block 划分

```python
TILE = next_pow2(N)                 # 8192（主 shape；fragment pow2 填充，rmsnorm L42 先例）
BM = 4                              # 主 shape（M=8192 % 4 == 0）；DMA 行批 64KB/tile 合并拷贝
num_tiles = ceildiv(M, BM)          # 2048 行块（T.Persistent 域）
num_blocks = 56                     # = vector_core_num（persistent grid，§5.5）
threads = min(1024, BM * TILE)      # 1024（per_token 先例；64 reg/thread 预算内）
```

- **BM=4 依据**：① DMA 合并——每 tile 单次 2D 拷贝 4×16KB=64KB 连续区间，摊薄行级搬运启动；② VRF 预算内最大行批（64 reg/thread）；③ 整除 M=8192（无行尾块）。
- **BM 自适应**（host 侧）：BM = 4（N ≤ 8192 且 M%4==0）→ 2（N ≤ 16384 且 M%2==0）→ 1（兜底，任何 M）——保证 BM | M，**无行尾 OOB**（§5.4）。
- **驻留上限推导**：UB 约束 2·BM·N·2B ≤ 194.4KB + VRF 约束 BM·TILE·6B ≤ 256KB（64 reg × 1024 threads）→ BM=1 时 N ≤ 32768；N > 32768 切泛化路径（BLOCK_N=8192，x_ub (BM, 8192) 双版本 128KB 不变）。

### 5.3 约束分析

- **C0 分形对齐**: 不适用（无 GEMM/Cube 段；`CONST-gemm-dim-constraints` 仅约束 `T.gemm` 路径）。
- **UB 容量**: 总 buffer ≈132KB ≤ 194.4KB（`CONST-capacity-a5` assumption 按 90% 折扣）✓。
- **VRF 容量**: 64 reg/thread（1024 threads）≤ rmsnorm 先例 80 reg ✓。
- **GM→UB 对齐**: fp16 行 8192×2B = 16KB，32B 对齐 ✓；非 16 倍数 N（如 N=1000 → 行 2000B 非 32B 倍）——DMA 支持字节级行长（skill §6.1），正确性不受影响，带宽档降级风险列 §9.3 观察项。
- **向量宽度整除**: TILE=8192 % 128（fp16 lane）= 0、% 64（fp32 lane）= 0 ✓；非 pow2 N 由 masked 载入吸收（尾 lane < 2×浪费）。

### 5.4 注意事项

- **行尾（M % BM ≠ 0）**：BM 自适应整除消除（BM ∈ {4,2,1} 必有除数 1），无 OOB DMA、无动态拷贝范围。
- **列尾（N < TILE）**：masked 载入（−inf 中性值）+ masked 存储（i < N），数值中性（§1.6.1 O5）。
- **泛化路径 N 尾块（N % BLOCK_N ≠ 0）**：末 chunk 同 masked 机制；哨兵钳位保护全 −inf 块（§1.6.1 O6）。
- **56 核尾波**：num_tiles=2048 = 56×36 + 32 → 末波 32/56 核占用（~2.4% 不均衡）；Stage 4 可调 BM/num_tiles 对齐（core-split-strategy §2.4 调优提示）。

### 5.5 分核策略（grid 划分 + AIC/AIV 分工 + 同步策略）⭐

> 三要素判定标准与核数实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（依据 `.agents/skills/tilelang-ascend/SKILL.md` §2/§3/§7）。

- **grid 划分（含核数依据）**：`num_blocks = 56`（一维 `T.Kernel(56)`，persistent）；`num_tiles = ceil(M/BM) = 2048` 行块由 `T.Persistent([num_tiles], 56, core_id, num_stages=2)` 调度器分配（多波：2048/56 ≈ 36.6 tiles/核，含尾波 32 tiles）。**核数实查记录**（2026-10-16，本设计执行期亲测）：
  ```python
  import torch
  props = torch.npu.get_device_properties()
  # name: Ascend950PR_9579, cube_core_num: 28, vector_core_num: 56
  ```
  纯 Vector 算子按 `vector_core_num = 56` 取 num_blocks（persistent 单波常驻，无超发串行）；cube_core_num=28 记录备查（本算子不占 AIC）。无真机环境的复现命令即上述查询代码；本仓 examples 惯例初值 32 blocks 在此被实测值 56 取代（`CONST-arch-a5` 的「无静态常数、须运行时实查」条款）。
- **AIC/AIV 分工**：纯 Vector 算子——逐元素（exp/mul/sub/cast）、行归约（`T.reduce_max/reduce_sum`）、数据搬运（`T.copy` MTE2/MTE3）全部落 **AIV**（56 个）；**无 Cube 工作**（无 `T.gemm`/L1/L0 段）→ 不占 AIC（28 个闲置）。pure kernel 的核型由 codegen 自动判定（core-split-strategy §1 ②），无需 `T.MixedKernel`/sid 声明；无 dual_copy 所有权问题（无 L0C→UB 段）。
- **同步策略（auto|manual 声明）**：**auto_schedule: yes**（默认）——依赖 AutoSchedule 自动插 flag/同步，数据依赖边界：① x_ub 版本 v 的 MTE2 写 → SimtVF 读（跨引擎，自动 flag 对）；② x_ub 版本 v 的 VALU 原位写（y）→ MTE3 读（跨引擎，自动 flag 对）；③ 版本 v 与 v⊕1 交替（multibuffering，编译器管理）；④ SimtVF 内跨线程 UB 读写 hazard → 自动 `asc_syncthreads`（skill §2.2/§10.8，回归 `testing/ascend/transform/test_ascend_thread_sync.py`）；⑤ reduce 终值 → inv/y 消费的 VALU 内顺序依赖（同引擎顺序语义）。无手动 `T.ascend_set_flag/wait_flag`（核内 8/跨核 16 flag 预算不占用）；调度提示使用位置：`T.Persistent(num_stages=2)`（tile 间流水）——不使用 `T.Stage`/`T.Task`（AutoSchedule 足以覆盖本算子依赖结构）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表仅 block 级/tile 级调度循环；无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 行块（M 向） | persistent 调度 | `T.Persistent([num_tiles], 56, core_id, num_stages=2)` | 2048 行块在 56 AIV 间持久调度 + tile 间流水 |
| 元素级（N 向） | 向量化 | `T.Parallel(BM, TILE)`（×4 处：载入/e/y 写）+ `T.Parallel(BM)`（inv） | §1.6.2 #1/#3/#5/#6 全向量化 |
| 行内归约 | 硬件归约 | `T.reduce_max / T.reduce_sum`（dim=1） | §1.6.2 #2/#4 |
| 泛化路径 chunk | 串行迭代 | `T.serial(num_chunks)` ×2（pass-1/pass-2） | online 递推循环携带依赖（§1.6.2 #8）；主 shape 不执行 |

### 6.2 循环伪代码

```python
# Block 级并行（一维 grid，56 persistent 核）
with T.Kernel(56) as core_id:
    x_ub = T.alloc_shared((BM, N), "float16")
    T.annotate_buffer_versions({x_ub: 2})
    for t in T.Persistent([num_tiles], 56, core_id, num_stages=2):
        base = t * BM
        T.copy(X[base : base + BM, 0:N], x_ub)          # MTE2（版本 v）
        with T.SimtVF(threads=1024):
            ...（§3.3 ①–⑥：masked 载入 → reduce_max → e=exp → reduce_sum → inv → 原位 y）
        T.copy(x_ub, Y[base : base + BM, 0:N])          # MTE3（版本 v）
# 泛化路径：外层同 T.Persistent；内部先 T.serial(num_chunks) 统计递推，再 T.serial(num_chunks) 重读归一化
```

### 6.3 流水线优化

`T.Persistent(num_stages=2)` + `T.annotate_buffer_versions({x_ub: 2})`：tile t+1 的 GM→UB 拷贝（MTE2，版本 v⊕1）与 tile t 的 SimtVF 计算（版本 v）重叠；跨引擎依赖（MTE2→VALU→MTE3）由 AutoSchedule 自动插 flag 对；不手动管理双缓冲（skill §4.2：优先编译器 multibuffering）。

### 6.4 尾块处理

- 行尾（M % BM）：BM 自适应整除（§5.2），不存在行尾块。
- 列尾（N % TILE）：masked 载入（pad=−inf）/masked 存储（i < N），数值中性（§1.6.1 O5）。
- 泛化路径块尾（N % BLOCK_N）：末 chunk masked + 哨兵钳位（§1.6.1 O6）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: auto_schedule: yes（AutoSchedule 自动同步；未设 `TL_ENABLE_AUTO_SCHEDULE: False`）

### 7.2 同步点说明

auto 模式下依赖边界说明（无手动同步点表；AutoSchedule 自动插入）：

| 依赖边界 | 涉及引擎/buffer | 自动机制 |
|----------|-----------------|----------|
| x_ub 写（MTE2）→ SimtVF 读 | MTE2 → VALU，版本 v | AutoSchedule flag 对 |
| x_ub 原位写 y（VALU）→ MTE3 读 | VALU → MTE3，版本 v | AutoSchedule flag 对 |
| 版本交替 v / v⊕1 | multibuffering | `T.annotate_buffer_versions({x_ub: 2})` + 编译器 |
| SimtVF 跨线程 UB hazard | VALU 线程域 | 自动 `asc_syncthreads`（skill §2.2） |
| reduce 终值 → inv/y | VALU 内 | 同引擎顺序语义 |

### 7.3 pass_configs 配置

```python
# 默认自动调度，不传 TL_ENABLE_AUTO_SCHEDULE（仅完全手动核需要 False）
kernel = tilelang.compile(prog, target="ascend", out_idx=-1)
```

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（源仓自身测试基准 `X.softmax(dim=1)`，源文件 L53/L55），**不复刻 §0.6 的 NPU 算法**——保证验证独立性。

```python
def golden_online_softmax(X: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax, torch on NPU（源仓测试基准直移植；fp16 in/out，
    torch 内部 fp32 累加路径与 §0.1 dtype 语义一致）。"""
    assert X.device.type == "npu" and X.dtype == torch.float16
    return X.softmax(dim=1)
```

```python
# L0 精度对比语句（主 shape 门槛；NaN 行以 equal_nan 对照 golden 实测行为）
torch.testing.assert_close(Y, golden_online_softmax(X), rtol=1e-2, atol=1e-2)
```

### 8.2 精度标准

> golden 以 torch 在 NPU（device="npu"）上计算参考，`torch.testing.assert_close` 对比；容差按 dtype 分档（AGENTS.md API Convention）：

| dtype | atol | rtol |
|-------|------|------|
| float16（本算子） | 1e-2 | 1e-2 |
| float32（如扩展） | 1e-3 | 1e-3（累加重算子放宽至 5e-3） |

**分层测试计划**（L0 门槛在本节冻结；L1/L2/Boundary 完整套件由 Stage 3 `tilelang-op-develop` 按本计划实现）：

| 层级 | 范围 | 用例 |
|------|------|------|
| **L0（门槛）** | 主 shape 正确性 | ① [8192, 8192] fp16 `randn` vs golden（assert_close 1e-2/1e-2）；② 小 shape smoke [128, 4096]；③ 源仓对照：与 `examples/online_softmax/online_softmax.py` 相同的对比方式（golden 同源） |
| **L1（shape/配置扫描）** | 参数化覆盖 | M ∈ {1, 3, 56, 57, 128, 4096, 8192} × N ∈ {1, 16, 1000, 4096, 8191, 8192}（覆盖 BM 自适应、非 pow2 掩码、非 32B 行）；[128, 65536]（泛化流式路径：N > 32768）；[1, 1] 最小 shape；全部 fp16，每 shape 独立 JIT（lru_cache 生效验证） |
| **L2（数值边界/压力）** | 角点与鲁棒性 | 大幅值行（±65504）、全等行、单主导值行、subnormal（6e-8）、深度下溢行（≤ −745）、±inf 混合行（golden equal_nan=True 对照）、NaN 行、N=1（y=1）、M=1；重复调用确定性（同输入 3 次 bitwise 一致——AutoSchedule 调度确定性检查）；驻留/泛化路径切换边界（N=32768/32769） |
| **Boundary（与 L2 合并枚举）** | 契约边界 | 非连续输入视图（host 侧 stride 断言——本算子契约要求连续行主，非连续由 caller `.contiguous()` 后传入）；非 32B 行对齐 N ∈ {1000, 8191} 的正确性（性能观察项见 §9.3） |

L2 的 ±inf/NaN 行为预期（依 §0.1 语义与 verify_equiv V3 角点实证）：含 +inf 行 → 全 NaN（与 torch 一致）；全 −inf 行 → 全 NaN；含 −inf 正常行 → −inf 位置输出 0；NaN 行 → 全 NaN。

---

## 9. 风险点与注意事项

### 9.1 已知约束（本算子相关）

1. **special_func 吞吐常数口径未定**（设计期最大不确定性）：`CONST-valu-model` 的 special=8 ops/cycle/AIV 未标注 op 粒度（element 级 vs 2048-bit 向量指令级，`src/ascend/transform/estimate_latency.h` L58-70）——瓶颈判定（VALU-special 主导 vs MTE 流量主导）存在两种解释。**缓解**：主选结构对两种解释均最优（同时最小化 special 次数与 GM 流量，§1.6.0 估算行）；Stage 4 以 npusim cycle 归因 + msprof 实测标定，常数回写 pattern-library。
2. **源朴素 online 合并的角点缺陷**（verify_equiv V4b 实证：fp64 角点违反率 13.6% / fp32 18.2%）：块间 max 间隙 > 88.7 时 fp32 溢出、块内全 −inf 时 NaN。**已处置**：泛化路径强制采用 O6 稳健合并（flash 形式 + 哨兵），主路径（驻留式）无合并操作不受影响；L2 用例覆盖 gap>88 与 −inf 角点。
3. **VRF 寄存器压力**（BM=4/TILE=8192 → 64 reg/thread）：接近 rmsnorm 先例水位（80 reg）；编译器 spill 时性能劣化。**缓解**：Stage 3 若见 spill 降 BM=2/threads=512（同 64 reg）或启用 fp16-e 备选（e_frag 减半，§1.6.4）。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | BM×N 超预算（大 N 仍走驻留路径） | 编译失败 | host 侧驻留上限判定（N ≤ 32768）强制切泛化路径（§5.2） |
| VRF spill | BM=4/TILE=8192 寄存器超水位 | 性能劣化 | BM=2/threads=512；或 fp16-e 备选（§1.6.4） |
| 原位覆写依赖误判 | x_ub reuse_input 的 WAR 分析 | 编译报错/结果错误 | fallback：独立 y_ub（+64KB，总 192KB ≤ 194.4KB 边缘通过；仍超则 BM=2） |
| reduce dtype 陷阱 | 归约 buffer 非 fp16/fp32（如误用 bf16） | 无法硬件直连（skill §10.11） | 本设计归约全 fp32（§3.2）；bf16 扩展时先升 fp32 |
| threads= 误用 | 照抄源 `T.Kernel(grid, threads=128)` | TypeError | `T.SimtVF(threads=)` 承接线程域（skill §10 陷阱 2） |
| 动态 shape 重编译延迟 | M/N 参数化首次调用 | JIT 编译耗时 | lru_cache per-shape（§4.7）；测试顺序调用同 shape 复用 |
| 56 核尾波不均 | num_tiles % 56 ≠ 0（2048%56=32） | ~2.4% 末波空转 | Stage 4 调 BM 或 num_tiles 对齐（core-split-strategy §2.4） |
| fp16-e 备选精度吃紧 | hexp fp16 arg 舍入 + fp16 树形和 | 相对误差 ~2.3e-03（模拟值） | 裁决阈值 5e-3 守卫 + L0 精度门禁拦截（§1.6.4） |

### 9.3 特殊场景处理

- **非 pow2 N**：TILE 填充 + masked 载入/存储（−inf/0 中性值）——正确性等价（§1.6.1 O5），L1 用例 N ∈ {1000, 8191} 覆盖。
- **N 非 16 倍数（非 32B 行）**：DMA 字节级行支持（skill §6.1），正确性不受影响；带宽档可能降级（N-strided/宽行档，`CONST-mte-bandwidth-model`）——性能观察项，Stage 4 对 N=1000 类 shape 实测带宽归因。
- **特大 N（> 32768）**：泛化流式路径（BLOCK_N=8192 + online 递推 + O6 稳健合并），主 shape 不执行；L1 用例 [128, 65536] 覆盖。
- **±inf/NaN 输入**：行为与 torch 对齐（§8.2 L2 预期表）；golden equal_nan=True 对照。
- **M=1/N=1 极小 shape**：BM=1/threads=min(1024, TILE) 自适应；N=1 时 m=x、S=1、y=1（语义正确）。
- **bf16/其它 dtype 扩展**（超出本契约）：归约前升 fp32（skill §10.11 dtype 白名单），e_frag 维持 fp32，仅 x_ub/输出 dtype 变化——设计结构不变。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/pilot/online_softmax/
├── online_softmax.py       # 算子实现（kernel + golden + L0 测试）—— Stage 3 产出
├── DESIGN.md               # 本设计文档（Stage 1 产出）
├── verify_equiv.py         # §1.6.1 等价性机器验证脚本（D-1，已执行全 PASS）
└── RETROSPECTIVE.md        # Stage 1 复盘（自进化钩子）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 设计文档（本文件） |
| `verify_equiv.py` | 已完成（已执行，V1–V4 全部 EQUIV_PASS） | 等价性机器验证：V1 去换底 / V2 倒数乘 / V3 驻留链 / V4 稳健 online 合并 + V4b/V5 诊断项 |
| `online_softmax.py` | 待实现（Stage 3） | 算子实现：§3.3 伪代码骨架 + §4.7 JIT 封装 + §8.1 golden + L0 测试 |
| `test_online_softmax.py` | 待实现（Stage 3，L1/L2/Boundary 分层套件） | 按 §8.2 分层计划实现 |

### 10.3 命名规范

- 项目目录名: `pilot`（conductor 指定）
- 算子目录名: `online_softmax`（snake_case）
- 实现文件: `online_softmax.py`
- 测试文件: `test_online_softmax.py`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）
2. ⬜ Golden 函数（§8.1，torch on NPU softmax）
3. ⬜ 算子实现（online_softmax.py，按 §3.3 伪代码 + §1.6.4 主选方案）+ L0 精度门禁（§8.2 L0 用例）
4. ⬜ L1/L2/Boundary 分层测试（§8.2 计划，Stage 3）
5. ⬜ Stage 4 性能调优 + §1.6.4 实验裁决（fp16-e 备选 A/B）
