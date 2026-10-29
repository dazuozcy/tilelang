# silu 算子设计文档

> 任务类型：非迁移（first_design）。输入需求：`examples/pilot/silu/`，shape `[128, 1024]` fp16 全静态，编程模式 simt，auto_schedule: yes。
> 真机环境：Ascend950PR_9579（实查 `cube_core_num=28` / `vector_core_num=56`，见 §5.5）。

## 1. 概述

### 1.1 算子名称

silu（SiLU / Swish 激活函数）

### 1.2 功能描述

逐元素激活映射：对输入张量每个元素独立计算 `silu(x) = x · sigmoid(x)`，输出与输入同 shape 同 dtype。常用于 LLM Feed-Forward / Attention 块的激活层。

### 1.3 数学公式

$$
\text{silu}(x) = x \cdot \text{sigmoid}(x) = \frac{x}{1 + e^{-x}}
$$

逐元素定义域行为（fp16 语义下的边界，详见 §9.3）：

- $x \to +\infty$：$e^{-x} \to 0$（下溢），$y \to x$；
- $x \to -\infty$（有限大负数）：$e^{-x} \to +\inf$（fp16 上溢），$y = x/\inf \to -0.0$，与真值 $x \cdot \text{sigmoid}(x) \to 0^-$ 的偏差 ≤ 1.7e-4（§8.3 边界分析）；
- $x = \pm\inf$：$-\inf/(1+\inf) = \text{NaN}$、$+\inf/1 = +\inf$，与 PyTorch `silu` 的 `sigmoid(±inf)` 语义一致；
- NaN 输入逐元素传播。

### 1.4 算法描述

单步逐元素算法（与 §1.6.0 调研选定算法一致——基线公式族即最优，无结构差异）：

1. 输入 `[128, 1024]` fp16 按行主序扁平化为 131072 个连续元素（host 侧纯视图操作）；
2. 每个元素独立执行 4 个向量操作：`t = -x` → `e = exp(t)` → `d = 1 + e` → `y = x / d`；
3. 结果按同一扁平布局写回输出张量，host 侧 reshape 回 `[128, 1024]`。

无跨元素依赖、无规约、无中间 GM 往返：输入 tile 一次 GM→UB 搬入，UB 内完成全部计算，一次 UB→GM 搬出。

### 1.5 数据流图

```
GM[A (131072,) fp16 扁平视图]
  --T.copy (MTE2, 2048 el/tile)--> UB[a_ub]
  --SimtVF T.Parallel(2048): y = x/(1+exp(-x))--> UB[b_ub]
  --T.copy (MTE3, 2048 el/tile)--> GM[B (131072,) fp16] --host reshape--> [128, 1024]
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先调研（1.6.0，调研四问）→ 再公式级优化（1.6.1）→ 再向量化替代（1.6.2）→ 再向量化轴与布局决策（1.6.3）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研深度判定（Phase 1）：单步逐元素 / 无规约 / 无窗口 / 无矩阵 → **轻量调研**（四问各一行结论 + 依据，algorithm-research.md §2）。调研对象：同一数学语义（silu 逐元素映射）的算法族，输入公式 `x/(1+exp(-x))` 只是基线候选之一。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线：输入公式 | `y = x / (1 + exp(-x))` | — | — | ✅（基线） |
| 1 | sigmoid 乘法形式 | `y = x·exp(x) / (1 + exp(x))` | 数学恒等 | op 数不变（exp+add+mul+div 仍 4 个向量 op），无净收益 | ❌ |
| 2 | 倒数乘形式 | `r = 1/(1+exp(-x)); y = x·r` | 数学恒等 | rcp（div 档 13 ops/cyc）+ mul（51 ops/cyc）替代 div（13 ops/cyc）：**多 1 个 op**，VALU 周期 +0.02 cyc/el（CONST-valu-model 折算） | ❌ |
| 3 | tanh 形式 | `y = x/2 · (1 + tanh(x/2))` | 数学恒等 | A5 无 tanh 直连 SFU intrinsic（`tilelang/ascend/language/simd.py` 仅有 vexp/vln/vsqrt/vexpdif，无 vtanh），tanh 须由 exp 合成 ≥2 个 SFU op + 除 2，op 数上升 | ❌ |
| 4 | exp2 换底缩放 | `exp(x) = exp2(x·log2e)` | 数学恒等 | A5 SFU 直连自然指数 `vexp`（simd.py:860，softmax 族的 exp2 缩放收益不成立），换底反增 1 次 mul | ❌ |
| 5 | vexpdif 融合取负 | `exp(-x) = vexpdif(0, x)`（融合 exp 与减法） | 数学恒等 | 仅省 1 次 vneg（0.02/0.24 cyc/el ≈ 8% 向量管线周期）；但 `T.simd.vexpdif` **仅 SimdVF 域且仅 fp32**（simd.py:1182-1195 "requires matching float32 vectors"），fp16 负载需双向 cast，净亏 | ❌ |
| 6 | 公共子表达式消除 | — | — | `exp` 实参 `-x` 仅使用一次；`x` 寄存器驻留被分子与取负复用，无可消除的重复子式 | ❌（无空间） |

**R2 在线算法**：**有——基线本身即单遍**。结构依据：silu 是逐元素映射，输出 `y_i` 只依赖同下标 `x_i`，不存在 running 统计量（和/平方和/max）需求，无跨元素依赖 → 单遍扫描天然在线（algorithm-research.md §3 R2 ③ 结构判据），无需也无法进一步"在线化"。

**R3 复杂度对比**（四口径；R1 仅基线存活，其余候选 op 数只增不减已淘汰；口径：FLOPs 按向量 op 计，SFU/div 按吞吐折算常规 op 当量 [CONST-valu-model：exp ≈ 51/8 ≈ 6.4 当量、div ≈ 51/13 ≈ 3.9 当量]；访存 = GM 读 + GM 写 + 中间 GM 往返）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线 `x/(1+exp(-x))` | 131072 el × 4 向量 op（1 SFU exp + 1 div + 2 常规）≈ 0.52M 向量 op，折算 ≈ 1.61M 常规 op 当量 | 512KB（读 256KB + 写 256KB，无中间 GM 往返） | 1（单遍） | O(tile) = 8KB UB 驻留（2048 el × 2B × 2 buffer） | 完全并行（56 AIV 可用），无跨核归约、无跨核同步 |

**R4 硬件亲和性评估**（基线候选逐项对照检查清单，常数均为 A5 assumption 状态）：

| 检查项 | 判定 | 依据 |
|--------|------|------|
| 计算单元匹配 | ✅ 逐元素 + SFU exp + div → Vector（AIV）；无 MAC 密集段，Cube 不参与 | 决策树「纯 element-wise → simt」；CONST-valu-model |
| 片上容量 | ✅ 工作集 2 tile × 4KB × 2 版本 = 16KB ≪ 194.4KB（SimtVF UB 216KB × 90% 折扣，CONST-capacity-a5） | §4.5 预算表 |
| 对齐 / 整除 | ✅ 扁平连续访问，tile 2048 el × 2B = 4KB（32B 行对齐 ✓）；T.Parallel 区域 2048 el = 16 × 128 el（fp16 2048-bit 向量宽），尾 lane 浪费 0% | CONST-valu-model（向量宽度）；§1.6.3 |
| 静态边界 | ✅ shape 全静态，TILE/iterations 均编译期常量 | 输入规格 dynamic_axes=[] |
| 流水可融合性 | ✅ 单 kernel 内 load→compute→store 三段，`T.Pipelined(num_stages=2)` 双缓冲流水 | example_vecadd.py:26 同构 |
| 跨核结构 | ✅ 各 block 输出区间互不相交，无跨 block 数据依赖；AutoSchedule 处理核内 MTE→VF 同步 | core-split-strategy.md §1 ③ |

**调研结论**：选定**基线公式族 `y = x/(1+exp(-x))`**（每元素 4 向量 op：neg + SFU exp + add + div，已是该语义下最小 op 组合）。与基线结构差异：无（基线即最优）。**「无更优替代」调研范围**：① `algorithm-candidates.md` 参考表（`kb_search "silu sigmoid elementwise"` 无 silu/sigmoid 专条目，命中 ALG-elementwise 族行——其"除法乘倒数 / CSE / cast 链合并"候选已逐项评估见 R1 #2/#6；ALG-softmax 的 exp→exp2 缩放思路对本算子不适用，R1 #4）；② `examples/ascend/`（vecadd / rmsnorm / per_token_cast_to_fp8 均无 silu/sigmoid 实现）；③ `pattern-library/constants.md`（CONST-valu-model / CONST-mte-bandwidth-model 代价口径）；④ 恒等变形自查（R1 表 6 候选全覆盖：代数重排、降代换、SFU 融合指令、换底、CSE）。本结论是 §1.4 与 §1.6.1–§1.6.3 的分析对象。

**设计期估算（D-2）**：估算下界(相对)：**计算项(SFU) ≈ 1.4× 流量项**（每 block 计算项 ≈ 988 cyc〔4096 el × (exp 1/8 + div 1/13 + 2 常规 op / 51)，`CONST-valu-model`，assumption〕vs 流量项 ≈ 694 cyc〔2 iter × (4096B 读 / 100 B/cyc + 90 base) + 2 iter × (4096B 写 / 115 B/cyc + 180 base)，`CONST-mte-bandwidth-model`，assumption〕；容量项豁免——轻量调研（单步逐元素类），UB 驻留 16KB ≪ 194.4KB〔216KB × 90%，`CONST-capacity-a5`，assumption〕）——**SFU exp/div 主导的向量计算受限算子（非纯带宽受限）**。A5 冷启动期为 assumption 基估算，常数待标定；只做候选间相对比较，绝对时延以 do_bench(msprof) 实测为准（协议：`_shared/standards/hardware-cost-model.md` §2）。

#### 1.6.1 数学等价优化（公式级）

**结论：无优化空间（0 项采纳）**。依据（候选逐项量化否决，承接 §1.6.0 R1 表）：

1. **op 组合已最小**：基线每元素 4 向量 op（neg + exp + add + div）。全部 6 个等价变形候选均不减少 op 数：sigmoid 乘法形式 op 数不变（R1 #1）；倒数乘形式多 1 op（R1 #2）；tanh 形式需 exp 合成（R1 #3）；exp2 换底反增 mul（R1 #4）；vexpdif 融合省 1 vneg（≈8% 向量管线周期）但 API 仅 SimdVF 域 + fp32，fp16 负载双向 cast 净亏（R1 #5，`tilelang/ascend/language/simd.py:1182-1195`）；无公共子表达式可消除（R1 #6——`-x` 用一次、`x` 寄存器驻留复用）。
2. **数值稳定化不适用且不需要**：softmax 式"减 max"稳定化针对 `exp(正大数)` 溢出；sigmoid 形式 `1/(1+exp(-x))` 双向饱和自稳定——`x` 大正 → `exp(-x)` 下溢为 0 → `y = x`（正确极限）；`x` 大负 → `exp(-x)` 上溢为 inf → `y = x/inf = -0.0`（真值趋 0，最大偏差 1.7e-4 ≪ fp16 容差 1e-2，见 §8.3）；NaN 逐元素传播。无需任何稳定化变形。
3. **访存已最小**：单遍、无中间 GM 往返（R3：512KB = 不可消除的 I/O 下界），输入 tile 单次搬入 UB 驻留复用（分子 `x` 与取负实参共享同一 UB 数据），无重复读取。

**优化后公式（供 §3.1 使用）= 基线公式**：`y = x / (1 + exp(-x))`。
**等价性机器验证（D-1）**：无采纳优化项，按调度指令豁免 `verify_equiv.py`；补充证据——设计期探针（`history_version/design_probe_simt_exp.py`）已实测该公式 kernel 与 fp32 opmath golden 的最大绝对偏差 3.906e-3 < 1e-2（2026-10-16，详见 §3.4 / §8.2）。
**优化结论**：采纳 **0 项——无优化空间**（单步逐元素映射：op 组合已最小、数值双向饱和自稳定、访存已达单遍 I/O 下界，逐项否决见上 1–3 条与 §1.6.0 R1 表）；优化后公式（供 §3.1 公式拆解使用）= 基线公式 `y = x / (1 + exp(-x))`；`verify_equiv.py` 按调度指令豁免（无采纳优化项，等价性证据为设计期探针实测，见上行 D-1）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

实现方案（§3.3）中全部循环与标量计算点盘点：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 逐元素 silu 映射（neg + exp + add + div，131072 元素） | 逐元素标量循环 | SimtVF 内 `for i in T.Parallel(TILE)` 向量化——编译器自动把 `tir.exp` lower 为 2048-bit `vexp`、div lower 为 `vdiv`（`src/ascend/transform/ascend_simdvf_lower_parallel.cc:409-412`：`tir.exp` → `simd_vexp()`；探针实测编译通过且精度达标） | ✅ | — |
| 2 | tile 起始偏移 `begin = (step·num_blocks + bx)·TILE` | 每 block 每迭代 ≤3 次标量乘加 | 无逐元素等价 API | ❌ | block 级索引 / 任务映射计算：每 block 每迭代仅 ≤3 次标量 op（全 kernel 合计 ≤192 次），非 131072 量级的逐元素热点，向量 API 无意义 |
| 3 | GM↔UB 数据搬运（2048 el × 2 方向 × 2 iter） | — | `T.copy`（MTE2/MTE3 DMA 引擎搬运，AutoSchedule 编排，非 VF 标量工作） | ✅ | — |

**向量化结论**：逐元素计算 100% 向量化（替代方案 API 均有源码 + 探针佐证）；保留 1 处标量计算点（#2，理由类别：block 级索引 / 任务映射，每 block ≤3 op 非热点）。与 §6 循环结构一致：§6 无任何逐元素标量循环。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（`[128, 1024]` fp16 行主序进/出），核内布局与向量化轴是设计变量。本算子为纯逐元素单轴算子（无广播、无窗口、无规约轴选择），候选矩阵聚焦"同一连续轴的不同切法 + 重排变体"：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | 扁平 1D 连续轴（I/O 原生行主序的 flat view） | 扁平元素轴（tile=2048 跨行连续） | 无 repack | 2048 el ÷ 128 el/向量宽（fp16 2048-bit，CONST-valu-model）= 16 向量宽整除，尾 lane 0%；MTE 连续 4KB 事务（峰值档，CONST-mte-bandwidth-model）；索引最简（线性偏移） | ✅ |
| 2 | 行块映射（block ↔ 行组，沿 N=1024 向量化） | N 轴（行内连续） | 无 repack（行连续） | 1024 ÷ 128 = 8 向量宽整除，尾 lane 0%；但需 2D 索引分解（行号/列号乘加），128 行 ÷ 32 block = 4 行/block；与方案 1 是**同一连续轴的等价切分**，访存与向量效率完全相同，仅增加索引计算 | ❌（无净收益，纯增索引复杂度） |
| 3 | 核内转置 / 列主序（沿 M=128 跨步 1024 向量化） | M 轴（跨步访问） | VF 内交错/解交错重排（`T.simd.vintlv/vdintlv`）或 `T.copy` 特定转置路径 | 逐元素算子对布局无语义偏好；跨步 1024 破坏连续向量装载 + 重排指令开销，纯负收益 | ❌（净亏：重排代价 > 0 收益） |
| 4 | host permute 进出（host 侧转置） | 任意 | 两次 host permute | host 数据搬运量 ≥ 算子本体流量（512KB），且违反 Host 侧输入操作约束（禁止改动输入张量内容/物理排布，ascend-constraints.md §4） | ❌（净亏 + 违反约束） |

**布局决策结论**：选定**方案 1——扁平 1D 连续轴**。`[128, 1024]` 行主序连续，flatten 后为 131072 连续元素段；silu 逐元素映射无行语义，tile 跨行边界切分合法。向量化轴 = 扁平元素轴（TILE=2048 = 16 × 128 el 向量宽，整除无尾 lane）。本算子属"纯单轴逐元素"类（方案 2/3/4 分别为等价切分、负收益重排、违规 host 操作，弃选理由见矩阵）。该结论同步落入 §3.3 伪代码（线性偏移索引）与 §6 循环结构（`T.Parallel(TILE)` 内层向量维 = 扁平元素轴）、§4 内存规划（一维 buffer 形状）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：simt（auto_schedule: yes）

### 2.2 选型理由

1. **算子特征**：纯 element-wise、无 matmul（→ 无需 mixed）、无显式向量指令控制需求（无掩码 / 舍入模式 / `latency=` 注解 / 求值顺序控制）→ 决策树「纯 element-wise 常规 → simt（默认推荐）」。
2. **编译器自动向量化已实证**：SimtVF 的 `T.Parallel` 循环内 `T.exp` 自动 lower 为 2048-bit `vexp` 向量指令（`src/ascend/transform/ascend_simdvf_lower_parallel.cc:409-412`），探针实测编译通过、精度达标（max diff 3.9e-3）——显式 SimdVF 无吞吐增益（同一 SFU 硬件、同 8 ops/cyc）。
3. **从 simt 起步原则**（AGENTS.md API Convention / 决策树 §2）：仅对 do_bench 证实的热点降级 simd。SimdVF 的唯一候选收益 `vexpdif` 融合已量化否决（§1.6.0 R1 #5：fp32-only + cast 往返净亏）。用户指定 simt，与决策树结论一致。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| VF 入口 | `T.SimtVF(threads=512)` + `for i in T.Parallel(2048)`（4 el/thread，与 vecadd 的 8192/2048 同比例） |
| 内存分配 | `T.alloc_shared`（UB）；无 Cube 段，不用 L1/L0 |
| 同步 | auto_schedule: yes——AutoSchedule 自动插 MTE→VF→MTE flag 对（§7） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式 = §1.6.1 优化后公式（= 基线公式）`y = x/(1+exp(-x))`；全部步骤在单个 `T.Parallel` 循环内逐元素融合执行（与 §1.6.2 向量化结论一致，无中间循环）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `t = -x` | 取负（vneg） |
| 2 | `e = exp(t)` | SFU 指数（vexp，8 ops/cyc/AIV 档） |
| 3 | `d = 1 + e` | 加 1（vadd） |
| 4 | `y = x / d` | 除法（vdiv，13 ops/cyc/AIV 档） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| grid | — | `T.Kernel(32)` | 一维 32 blocks（无 threads 参数） | simt |
| UB 分配 | — | `a_ub = T.alloc_shared((2048,), "float16")`；`b_ub = T.alloc_shared((2048,), "float16")` | UB 输入/输出 tile | simt |
| 多缓冲 | — | `T.annotate_buffer_versions({a_ub: 2, b_ub: 2})` | 双版本支撑 2 级流水 | simt |
| 装载 | `x tile` | `T.copy(A[begin:begin+2048], a_ub)` | GM→UB（MTE2，连续 4KB） | simt |
| 计算全式 | `y = x/(1+exp(-x))` | `with T.SimtVF(threads=512):` 内 `for i in T.Parallel(2048): b_ub[i] = a_ub[i] / (T.float16(1.0) + T.exp(-a_ub[i]))` | 编译器自动向量化（exp→vexp、/→vdiv） | simt |
| 写回 | `y tile` | `T.copy(b_ub, B[begin:begin+2048])` | UB→GM（MTE3） | simt |
| 流水 | — | `for step in T.Pipelined(2, num_stages=2):` | 2 迭代 × 2 级流水 | simt |

### 3.3 计算伪代码

> `T.Tensor` 的 dtype 以第二位置参数传递（与 `examples/ascend/` 形态一致，不用 `dtype=` 关键字）。

```python
import tilelang
import tilelang.ascend.language as T

M, N = 128, 1024
TOTAL = M * N                    # 131072 = 2^17
NUM_BLOCKS = 32                  # grid（§5.5：单波 ≤ 56 AIV，整除 2^17）
TILE = 2048                      # 每 block 每迭代元素数（16 × 128 el 向量宽）
THREADS = 512                    # SimtVF 线程域（4 el/thread）
ITERATIONS = TOTAL // (NUM_BLOCKS * TILE)   # = 2，静态常量

@T.prim_func
def silu_kernel(A: T.Tensor((TOTAL,), "float16"), B: T.Tensor((TOTAL,), "float16")):
    with T.Kernel(NUM_BLOCKS) as bx:            # 一维 grid，无 threads 参数
        a_ub = T.alloc_shared((TILE,), "float16")   # UB 输入 tile（4KB）
        b_ub = T.alloc_shared((TILE,), "float16")   # UB 输出 tile（4KB）
        T.annotate_buffer_versions({a_ub: 2, b_ub: 2})
        for step in T.Pipelined(ITERATIONS, num_stages=2):   # 双缓冲流水
            begin = (step * NUM_BLOCKS + bx) * TILE          # block 级标量索引（§1.6.2 #2）
            T.copy(A[begin : begin + TILE], a_ub)            # GM→UB（MTE2）
            with T.SimtVF(threads=THREADS):
                for i in T.Parallel(TILE):                   # 逐元素全向量化（§1.6.2 #1）
                    b_ub[i] = a_ub[i] / (T.float16(1.0) + T.exp(-a_ub[i]))
            T.copy(b_ub, B[begin : begin + TILE])            # UB→GM（MTE3）
```

host 侧（kernel 外）仅做元数据视图：输入 `x.flatten()`（连续张量的纯视图，不改内容/物理排布）、输出 `.reshape(128, 1024)`——符合 Host 侧输入操作约束（ascend-constraints.md §4）。

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|----------|----------|
| `T.Kernel(32)`（一维，无 threads=） | example_vecadd.py:21；ascend-constraints.md §1 | ✅ 探针编译+运行通过 |
| `T.alloc_shared((2048,), "float16")`（UB） | example_vecadd.py:22-24 | ✅ 探针通过 |
| `T.annotate_buffer_versions({..: 2})` | example_vecadd.py:25 | ✅ 探针通过 |
| `T.Pipelined(2, num_stages=2)` | example_vecadd.py:26 | ✅ 探针通过 |
| `T.copy(GM↔UB)` | example_vecadd.py:28,42 | ✅ 探针通过 |
| `T.SimtVF(threads=512)` + `T.Parallel` | example_vecadd.py:31-33；example_rmsnorm.py:54 | ✅ 探针通过 |
| `T.exp`（TIR `tir.exp` intrinsic） | 升降低径：`src/ascend/transform/ascend_simdvf_lower_parallel.cc:409-412`（`tir.exp` → `simd_vexp()` 2048-bit 向量指令） | ✅ 探针通过（fp16 输入，max diff 3.9e-3） |
| `T.float16(1.0)` 常量提升 | 探针同款写法 | ✅ 探针通过 |
| 编译入口 `tilelang.compile(prog, out_idx=-1)` / `@tilelang.jit(target="ascend")` | AGENTS.md API Convention；op-develop skill Phase 2 | ✅ 探针用 `tilelang.compile(silu_kernel, out_idx=-1)` 实测 |

**探针**：`examples/pilot/silu/history_version/design_probe_simt_exp.py`（2026-10-16 执行，32 blocks × 2048 × 2 iter 配置全量通过：编译 ✓ / vs fp32-upcast golden 1e-2 ✓ / vs direct-fp16 golden 1e-2 ✓ / max abs diff 3.906e-3）。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 一维 Kernel / `threads=` 禁用 | Yes（已遵守） | `T.Kernel(32)` 一维；线程域在 `T.SimtVF(threads=512)` 内声明 |
| C0 分形对齐（GEMM） | No | 无 matmul，不适用 |
| L0C/L0A/L0B 容量 | No | 无 Cube 段，不适用 |
| UB 容量 | Yes（已核） | 16KB ≪ 194.4KB（216KB × 90% 折扣），§4.5 |
| OOB padding（`pad_value`/`data_select` 互斥） | No | 精确整除（131072 = 32 × 2048 × 2），无尾块 |
| 静态循环边界 | Yes（已核） | ITERATIONS/TILE/NUM_BLOCKS 均编译期常量 |
| grid 整除性（无静态核数常数） | Yes（已核） | §5.5：grid 必须整除元素总量，选 32（2^5 | 2^17）；含 7 因子的 28/56 网格不可整除 |
| Host 侧输入操作约束 | Yes（已遵守） | host 仅 flatten/reshape 元数据视图 |

#### 3.5.2 参考实现差异说明

非迁移任务，无外部 GPU 参考实现。本项目同类实现参考见 §3.5.3。

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/ascend/example_vecadd.py` | 高度相似（同构模板） | elementwise + `T.Pipelined` 双缓冲 + `T.copy` GM↔UB + SimtVF/SimdVF 双模式对照（本设计取 simt 分支结构） |
| `examples/ascend/example_rmsnorm.py` | 相似 | SimtVF 线程域配比（threads 与 tile 比例）、UB 预算规划写法、`T.Tensor` 扁平化签名 + host reshape 模式 |
| `examples/pilot/silu/history_version/design_probe_simt_exp.py` | 本算子探针 | `T.exp` SimtVF 升降低径 + fp16 精度 + grid 整除性的设计期实证 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| A（x） | `(128, 1024)` → kernel 签名扁平视图 `(131072,)` | float16 | 行主序连续；host 侧 `flatten()` 纯视图 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| B（y） | `(131072,)` → host 侧 `reshape(128, 1024)` | float16 | 与输入同 shape/dtype（same_as_input） |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| a_ub | `(2048,)` | float16 | UB | 输入 tile（4KB × 2 版本） |
| b_ub | `(2048,)` | float16 | UB | 输出 tile（4KB × 2 版本） |

无 fragment 级显式分配（SimtVF 寄存器分配由编译器处理）；无 GM 中间缓冲。

### 4.4 内存搬运路径

```
GM[A] --T.copy (MTE2, 2048 el/tile)--> UB[a_ub]
  --SimtVF T.Parallel(2048): y = x/(1+exp(-x))（UB→VRF 装载，寄存器计算，VRF→UB 写回）--> UB[b_ub]
UB[b_ub] --T.copy (MTE3, 2048 el/tile)--> GM[B]
```

无 L1 / L0 路径（纯 Vector 算子，Cube 不参与）。

### 4.5 UB 内存预算

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| a_ub × 2 版本 | `(2048,)` × 2 | float16 | 8192 |
| b_ub × 2 版本 | `(2048,)` × 2 | float16 | 8192 |
| **总计** | | | **16384（16KB）** / 上限 194.4KB（SimtVF UB 216KB × 90% 折扣，`CONST-capacity-a5` assumption）——占用 8.2% ✓ |

（AutoSchedule multibuffering 版本数已计入：`annotate_buffer_versions` 声明的 2 版本即上表 ×2。）

### 4.6 动态轴定义

无（输入规格 `dynamic_axes: []`，全静态；shape/tiling 常量编译期确定）。

### 4.7 JIT 配置

```python
@tilelang.jit(
    target="ascend",          # 规范入口（别名 asc）
    out_idx=[1],              # 输出为第 2 个张量（B）；tilelang.compile(prog, out_idx=-1) 亦合法
)
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector

**判定依据**：算子仅含逐元素运算（neg/exp/add/div），无 matmul、无规约——决策树「纯 element-wise → simt / 仅需 UB」。

### 5.2 Block 划分

```python
# 扁平元素轴划分（2D shape 无行语义，按 §1.6.3 方案 1 扁平化）：
TILE = 2048                       # 每 block 每迭代元素数：16 × 128 el 向量宽整除、4KB 连续 MTE 事务
NUM_BLOCKS = 32                   # grid：单波 ≤ 56 AIV（§5.5）；32 = 2^5 整除 2^17
ITERATIONS = TOTAL // (NUM_BLOCKS * TILE)   # = 131072 // 65536 = 2（静态常量）
# 每 block 处理 NUM_BLOCKS 均分的 4096 元素（2 迭代 × 2048），block 间输出区间互不相交
```

TILE=2048 依据：① 向量宽整除（2048 = 16 × 128 el，fp16 2048-bit 向量，尾 lane 0%）；② MTE 连续事务 4KB（32B 对齐 ✓，峰值档带宽）；③ 双 tile × 双版本 UB 驻留仅 16KB，流水深度 2 足够覆盖 MTE base latency（90/180 cyc）。

### 5.3 约束分析

- **C0 分形对齐**：不适用（无 GEMM、无 Cube 段）。
- **UB 容量**：16KB ≪ 194.4KB（SimtVF 216KB × 90%，`CONST-capacity-a5` assumption）✓（§4.5）。
- **L0 容量**：不适用（无 Cube 计算）。
- **grid 整除性**（本算子关键约束）：元素总量 `TOTAL = 131072 = 2^17`，必须满足 `TOTAL % (NUM_BLOCKS × TILE) == 0`，否则尾部元素静默漏写（探针 v1 用 28 blocks 实证：12.5% 输出未初始化，max abs diff 4.8）。`2^17` 不含因子 7 → 含 7 因子的网格（28 = 4×7、56 = 8×7）**无法整除**，此为 grid 选 2 的幂（32）的硬依据。host 侧须 `assert TOTAL % (NUM_BLOCKS * TILE) == 0` 显式防御。

### 5.4 注意事项

- **精确整除，无尾块**：契约 shape `[128, 1024]` 下 `131072 = 32 × 2048 × 2`，无需 padding / OOB 处理 / 动态边界。
- **泛化约束**：其他 shape 复用本 kernel 时须满足同一整除约束（host assert 显式失败优于静默漏写）；不满足时需改用 GM→UB `pad_value=` 右 padding + 写回 crop 或按 shape 特化 grid（超出本契约范围，列为 Stage 3/4 泛化项）。
- **参数级 A/B（非结构备选，交 Stage 4 do_bench 裁决）**：grid/tile 组合在 assumption 级代价模型下差异 ≤ ±10%（SFU 计算项与 MTE 流量项的相对主导随常数标定可能翻转）：① 64 blocks × 1024 × 2 iter（vecadd 惯例 64 网格，2 波形态：56+8）；② 128 blocks × 1024 × 1 iter（3 波，更细粒度摊薄 per-block 计算）；③ TILE ∈ {1024, 4096} 变体。均满足整除约束，Stage 4 按 core-split-strategy.md §2.4（block 数对齐物理核数 / 消除负载不均）A/B 实测。

### 5.5 分核策略（grid 划分 + AIC/AIV 分工 + 同步策略）⭐

> 三要素判定标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 tilelang-ascend skill §2/§3/§7）。

- **grid 划分（含核数依据）**：`num_blocks = 32`（扁平元素轴均分，等价于 `ceil(M/block_M) × ceil(N/block_N)` 的扁平形式：每 block 4096 元素 = 2 iter × 2048）。**核数实查**（2026-10-16，真机执行）：

  ```python
  import torch
  props = torch.npu.get_device_properties(0)
  # 实测返回：name = Ascend950PR_9579, cube_core_num = 28, vector_core_num = 56
  ```

  block 数与核数关系：32 blocks ≤ 56 AIV → **单波**（每 block 驻留一个 AIV，无多波串行调度），AIV 占用率 32/56 ≈ 57.1%。未选满 56 AIV 的原因：56 = 8×7 不整除 2^17（§5.3 整除性硬约束，需尾块掩码/OOB 守卫破坏静态流水结构）；估算行显示计算项（SFU）≈ 1.4× 流量项，更多 AIV 的边际收益受 wave 量化与 MTE 聚合上限约束（assumption 级常数，不纸面拍板）——64/128 网格变体列入 §5.4 Stage 4 A/B。不 persistent 化（工作总量 2 迭代，launch 摊薄无意义）。

- **AIC/AIV 分工**：**纯 Vector 算子，仅占 AIV**——全部计算（SimtVF 向量段：neg/vexp/vadd/vdiv）与全部数据搬运（MTE2 GM→UB、MTE3 UB→GM）落位 AIV；AIC（Cube）无 `T.gemm` 类工作，不参与本 kernel；pure kernel 核型指派由 codegen 自动判定（core-split-strategy.md §1 ②）。无 mixed 核、无 sid 划分、无 dual_copy 所有权问题。

- **同步策略（auto|manual）**：**`auto_schedule: yes`**（默认）。依赖的数据依赖边界：① 核内序列 `T.copy 装载 → SimtVF 计算 → T.copy 写回`（MTE2→VF→MTE3 管线依赖）+ `annotate_buffer_versions` 双版本间的读写冲突——由 AutoSchedule 自动插 flag 对（AssignCore/InsertSync pass）；② 跨 block：各 block 输出区间互不相交，**无跨核数据依赖、无跨核同步需求**。调度提示使用位置：`T.Pipelined(ITERATIONS, num_stages=2)`（tile 级双缓冲流水）；无手动 `T.ascend_set_flag`/`T.ascend_wait_flag`，无 flag id 预算问题。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化（`T.Parallel`）——本表循环仅指 block 级 / tile 级调度循环，无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| block 级并行（32 blocks） | 隐式（grid 维度） | `T.Kernel(32)` | 每 block 独立处理均分的 2×2048 元素，输出区间不相交 |
| tile 迭代（2 次） | 流水迭代 | `T.Pipelined(2, num_stages=2)` | 双缓冲：iter i+1 的 MTE2 装载与 iter i 的计算/写回重叠 |
| 元素级（2048 el/tile） | 向量化并行 | `T.SimtVF(threads=512)` 内 `T.Parallel(2048)` | 逐元素全向量化（§1.6.2 #1）；4 el/thread，编译器发射 2048-bit vneg/vexp/vadd/vdiv |

### 6.2 循环伪代码

```python
with T.Kernel(NUM_BLOCKS) as bx:                       # block 级并行（一维 grid）
    for step in T.Pipelined(ITERATIONS, num_stages=2): # tile 级流水（静态边界 2）
        begin = (step * NUM_BLOCKS + bx) * TILE        # block 级标量索引（§1.6.2 #2 保留理由）
        T.copy(A[begin:begin+TILE], a_ub)              # MTE2 装载
        with T.SimtVF(threads=512):
            for i in T.Parallel(TILE):                 # 元素级向量化（无标量循环）
                b_ub[i] = a_ub[i] / (T.float16(1.0) + T.exp(-a_ub[i]))
        T.copy(b_ub, B[begin:begin+TILE])              # MTE3 写回
```

### 6.3 流水线优化

使用 `T.Pipelined(num_stages=2)`：`a_ub`/`b_ub` 各 2 版本（`T.annotate_buffer_versions`），iter i+1 的 GM→UB 装载与 iter i 的 VF 计算 + UB→GM 写回重叠。迭代数仅 2，流水收益有限（~15% 量级，按估算行 MTE/计算比重折算）；选择流水而非单迭代大 tile 的原因：保持 TILE=2048 的向量宽/MTE 事务最优形态，且结构与 vecadd 模板一致、Stage 4 可平滑扫参。

### 6.4 尾块处理

无尾块：契约 shape 下精确整除（`131072 = 32 × 2048 × 2`）。泛化 shape 时的防御策略见 §5.4（host assert 显式失败 + padding 方案）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：auto_schedule: yes（AutoSchedule 自动同步）

### 7.2 同步点说明

auto_schedule: yes 下无需手动同步点。AutoSchedule 自动处理的依赖边界：

| 依赖边界 | 处理方式 |
|----------|----------|
| MTE2 装载 → SimtVF 计算（a_ub 读前写完成） | AutoSchedule 自动插 flag 对（InsertSync pass） |
| SimtVF 计算 → MTE3 写回（b_ub 写后读完成） | 同上 |
| `annotate_buffer_versions` 双版本间流水冲突 | AutoSchedule multibuffering 编排 |
| 跨 block | 无数据依赖（输出区间不相交），无需跨核同步 |

### 7.3 pass_configs 配置

```python
# 默认自动调度，不传 TL_ENABLE_AUTO_SCHEDULE（完全手动核才需要 False）
pass_configs = {}   # 默认配置
```

---

## 8. 验证方案

### 8.1 Golden 函数

> 非迁移任务：golden 以 §1.3 数学语义（silu 标准定义）为依据，用 PyTorch 内置算子独立实现（fp32 opmath），不复刻本设计的 NPU kernel 算法（kernel 为 fp16 直接计算路径，golden 为 torch fp32 路径，实现相互独立）。

```python
def golden_silu(x):
    """PyTorch 参考实现：fp32 opmath 计算，回 cast 输入 dtype（device="npu" 上执行）。"""
    return torch.nn.functional.silu(x.to(torch.float32)).to(x.dtype)
```

探针已同时验证 direct-fp16 golden（`F.silu(x)` fp16 直算）也在 1e-2 容差内（max diff 3.906e-3）——两种 golden 口径均可用，取 fp32-opmath 版为准（与 torch 语义一致，且对 kernel 的 fp16 舍入更宽容）。

### 8.2 精度标准

golden 以 torch 在 NPU（`device="npu"`）上计算，`torch.testing.assert_close` 对比（AGENTS.md API Convention 分档）：

| dtype | atol | rtol |
|-------|------|------|
| float16（本算子） | 1e-2 | 1e-2 |

设计期实证：探针 max abs diff = 3.906e-3 < 1e-2（fp16 vexp "hw" 精度 + fp16 vdiv，含 ±3σ 随机全量 131072 元素）。

### 8.3 L0 门槛测试计划（分层测试交由 tilelang-op-develop，此处定 L0 门槛）

按 op-develop skill 口径（`python silu.py --level L0`）：

| 用例 | shape | dtype | 判定 |
|------|-------|-------|------|
| L0-1（契约） | `[128, 1024]` | fp16 | `assert_close(kernel(x), golden_silu(x), rtol=1e-2, atol=1e-2)` |
| L0-2（同整除类异形态） | `[512, 512]`（TOTAL=2^18，ITERATIONS=4） | fp16 | 同上（验证 grid 算术对形态泛化正确） |

L1/L2/Boundary 由开发者按 op-develop 模板扩展，本设计给定关键边界值与期望（Boundary 层）：

| 输入 x | 期望输出 | 说明 |
|--------|----------|------|
| `0` | `0` | 原点 |
| `±6e-8`（fp16 subnormal） | `≈ ±3e-8`（≈ x/2） | 次正规区 |
| `+11.0 / -11.0`（exp 上溢边界 exp(11.09)≈65504） | `≈ 11.0 / ≈ -1.66e-4` | 饱和过渡区 |
| `+30 / -30`（深饱和） | `≈ 30 / -0.0`（真值 -6.9e-12，flush 偏差 < 1.7e-4） | fp16 exp 上溢 → x/inf → -0.0 |
| `±65504`（fp16 极值） | `≈ +65504 / -0.0` | 同上 |
| `+inf` | `+inf`（inf/(1+0)） | 与 torch `sigmoid(inf)=1 → inf×1` 一致 |
| `-inf` | `NaN`（-inf/(1+inf) = -inf/inf） | 与 torch `sigmoid(-inf)=0 → -inf×0=NaN` 一致 |
| `NaN` | `NaN` | 逐元素传播 |

边界语义一致性说明：kernel 公式 `x/(1+exp(-x))` 与 torch `x·sigmoid(x)` 在 ±inf 处同为 NaN/inf（不同路径、相同结果）；有限大负数处 kernel flush 为 -0.0 而 golden 给出微小负值，偏差 ≤ 1.7e-4 ≪ atol 1e-2，容差覆盖。

---

## 9. 风险点与注意事项

### 9.1 已知约束

- **一维 Kernel / `threads=` 禁用**：已遵守（`T.Kernel(32)`；线程域在 `T.SimtVF(threads=512)` 内）。
- **静态循环边界**：ITERATIONS/TILE/NUM_BLOCKS 全编译期常量 ✓。
- **UB 容量**：16KB / 194.4KB（90% 折扣），余量充足 ✓。
- **C0 分形 / OOB padding**：不适用（无 GEMM、精确整除无尾块）。
- **Host 侧输入操作约束**：host 仅 flatten/reshape 元数据视图，不改输入内容/物理排布 ✓。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| grid 整除违例（尾部静默漏写） | `TOTAL % (NUM_BLOCKS×TILE) != 0`（探针 v1 用 28 blocks：12.5% 输出未初始化，max abs diff 4.8） | 精度失败且表现为随机垃圾值 | host 侧 `assert` 显式防御；grid 只取 2 的幂（2^17 不含因子 7） |
| fp16 exp 精度超差 | 硬件 vexp "hw" 精度（非精确舍入） | 理论风险 | 已实证：探针 max diff 3.906e-3 < 1e-2（全量 131072 随机元素 + 边界值待 Stage 3 Boundary 层覆盖） |
| 饱和区 flush-to-zero | x < -11.09 时 exp(-x) 上溢 inf → y = -0.0 | 与 golden 微小偏差（≤ 1.7e-4） | 容差 1e-2 覆盖；Boundary 层显式用例 |
| `T.Kernel(threads=...)` 误用 | 照 GPU 习惯传 threads | TypeError | 线程域在 `T.SimtVF(threads=512)` 内声明 |
| 环境阴影导入 | 按脚本路径直跑时 `import tilelang` 解析到其他仓库（无 `tilelang.ascend`） | ModuleNotFoundError | 以本仓为 cwd 或显式 `PYTHONPATH=/home/z00875089/tilelang-fork` |

### 9.3 特殊场景处理

- **NaN/inf 传播**：见 §8.3 边界表——`-inf → NaN`、`+inf → +inf`、`NaN → NaN`，与 PyTorch 语义逐项一致。
- **深饱和区**（x < -11.09）：fp16 exp 上溢使输出 flush 为 -0.0，最大偏差 1.7e-4（在 x = -11.09 处，$|x \cdot e^{x}|$ 的峰值），远小于容差。
- **shape 泛化**：本设计面向静态契约 `[128, 1024]`；其他 shape 需满足整除约束（§5.4），否则 Stage 3/4 需引入 padding 或 shape 特化工厂（`lru_cache` 按 shape 分派 grid）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/pilot/silu/
├── DESIGN.md               # 本设计文档
├── silu.py                 # 算子实现 + 分层测试（待实现，Stage 3）
├── RETROSPECTIVE.md        # Stage 1 复盘
└── history_version/
    └── design_probe_simt_exp.py   # 设计期探针（T.exp 升降低径 + 精度 + grid 整除性实证）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 本设计文档 |
| `silu.py` | 待实现 | 算子实现（kernel + golden + run_L0/L1/L2/boundary 分层测试，op-develop 模板） |
| `history_version/design_probe_simt_exp.py` | 已完成 | 设计期探针（2026-10-16 全量通过） |
| `verify_equiv.py` | 不适用 | §1.6.1 无采纳优化项，豁免（调度指令确认） |

### 10.3 命名规范

- 项目目录名：`pilot`；算子目录名：`silu`；实现文件：`silu.py`；测试入口：`silu.py --level {L0,L1,L2,boundary,all}`。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）
2. ⬜ Golden 函数（§8.1，验证基准）
3. ⬜ 算子实现（`silu.py`：§3.3 结构 + L0 门槛用例）+ 与 golden 精度比对
4. ⬜ L1/L2/Boundary 扩展（op-develop）
