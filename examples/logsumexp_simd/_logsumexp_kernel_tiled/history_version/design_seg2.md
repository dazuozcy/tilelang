## 1. 概述

### 1.1 算子名称

`_logsumexp_kernel_tiled`（op 名 `LogSumExpFwdOpSimd` / manifest id `logsumexp_simd` 的**分块流式成员**；SIMD 家族 = 本仓 `examples/logsumexp_simd/` 独立变体谱系）。产出文件：`examples/logsumexp_simd/_logsumexp_kernel_tiled/_logsumexp_kernel_tiled.py`。

### 1.2 功能描述

行级数值稳定 logsumexp 的 **N 向分块流式（online）变体**：当 N 超出单 tile 驻留上限（`MAX_SINGLE_TILE_COLS = 32512`）或驻留预算超限时，将整行归约改写为 chunk 串行 + running (max, sum) flash 形式合并——**任意大 N 单遍 GM 读完成归约，片上驻存 O(BLOCK_N) 而非 O(N)**。适用场景：超长上下文注意力权重归约（wl2 `attn-weights-32k` (1024, 32768) bf16）与 lm-head 超长 logits（wl3 `lm-head-logits` (4, 102400) fp16/bf16）。与 single 变体（`examples/logsumexp_simd/_logsumexp_kernel_single/`，已完成）构成 SIMD 家族的算法分工双变体——wrapper 按 `tile_n != 0` 与驻留预算 dispatch（§0.2），本 kernel 承接大 N / 超驻留路径。

### 1.3 数学公式

解析式与 single 变体同一（§0.1）：

$$
y_i \;=\; \mathrm{logsumexp}_j(x_{ij}) \;=\; m_i + \log\Big(\sum_{j=0}^{N-1} \exp(x_{ij} - m_i)\Big), \qquad m_i = \max_j x_{ij}
$$

tiled 变体的执行形态为 **chunk 串行 online 递推**（chunk 宽 BLOCK_N，c = 0..num_chunks−1）：

$$
\text{init: } m^{\text{run}} = -\infty,\; s^{\text{run}} = 0; \qquad
\text{chunk } c:\; \begin{cases} p = m^{\text{run}} \\ m^{\text{run}} \leftarrow \max(m^{\text{run}},\; \max_j x^{c}_j) \\ e^c_j = \exp(x^{c}_j - m^{\text{run}}) \\ s^{\text{run}} \leftarrow s^{\text{run}} \cdot \exp(p - m^{\text{run}}) + \sum_j e^c_j \end{cases}; \qquad
y = m^{\text{run}} + \log s^{\text{run}}
$$

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

§1.6.0 调研选定 **「online flash 递推（chunk 粒度）」** 算法族（在线算法是本变体的存在理由——任意大 N 单遍扫描），经 §0.6 R1–R6 重设计后的 NPU 侧算法（B14 架构）：

1. **任务划分**：一维 `T.Kernel(56)`（= `vector_core_num` 实查）；`T.Persistent([num_tiles], 56, core_id, num_stages=2)` 行块持久调度，num_tiles = ceil(M / block_m)；任务内 **行串行**（`for r in range(rows_this)`，rows_this ∈ {block_m, rem_m} 静态），每行独立完成其全部 N 向 chunk 的 online 流式。
2. **chunk 流式**（每行）：`for c in T.serial(num_full)` full chunk 串行 + Python 静态尾 chunk 分支（rem = N % BLOCK_N > 0 时）；每 chunk 一条 `T.copy` (1, BLOCK_N) GM→UB 单行切片 DMA（x_ub 双版本，chunk c+1 预取与 chunk c 计算重叠）。
3. **per-chunk SimdVF 指令流**（每行每 chunk 一个 `with T.SimdVF():` 块，两遍扫描）：pass1 逐 64 元素组 `vld(UNPK_B16)`+`vcvt` → `vmax` 累加 → 遍末单次 `vcmax`+`vdupv` 得 chunk max 广播 m_c_brc；**max 合并**：`prev_brc = m_run`（SSA 快照）→ `m_run = vmax(m_run, m_c_brc)`；pass2 重装载 → `e = vexpdif(c, m_run)` 融合指数 → `vadd` 累加 → 单次 `vcadd` 得 s_c（lane-0 形态）；**sum flash 重缩放合并**：`s_run = vadd(vmul(s_run, vexpdif(prev_brc, m_run)), s_c)`。
4. **running 统计量驻留**：`m_run` / `s_run` 为 `T.simd.alloc_var("float32")`（local.var 可变寄存器），行迭代作用域声明（SimdVF 块外）、per-chunk SimdVF 块内读写、行尾 epilogue 块内消费（§0.6 R5；跨块访问无单一先例 = 风险 1 + fallback）。
5. **尾 chunk**：静态 rem 切片 `T.copy` + 组循环 G_tail 满组 + 尾组 `vsel(c, vdup(−inf,f32), update_mask(k_tail,32))` fp32 域掩码（−inf 同为 max 幂元与 exp 后 sum 幂元）。
6. **行尾 epilogue**：`out = vadd(m_run, vln(s_run))` → `vcvt(out, dtype)`（fp32 免）→ `vsts(lse_ub[r], ·, pset(16,"PAT_VL1"), "ONEPT_B16")` 单点存储；任务尾 `T.copy(lse_ub[0:rows_this], y[base:+rows_this])` 静态切片写回。
7. **BLOCK_N = tile_n 直保留**（≤ UB clamp 上界：16-bit 55,040 / fp32 27,520，§4.5）——SimdVF 无 fragment，非 pow2 tile_n（16640/25600）不重划，chunk 划分与源一致（§0.6 R2）。

### 1.5 数据流图

```
GM[x (M,N) dtype]
  │ per-row per-chunk T.copy (1,BLOCK_N) 单行切片 ──MTE2──▶ UB[x_ub (1,BLOCK_N) ×2 版本]
  │                                        （chunk c+1 预取 ∥ chunk c 计算，AutoSchedule flag 对）
  ▼ per-chunk SimdVF 块（每行每 chunk）
UB[x_ub] ──vld(UNPK_B16)+vcvt──▶ VRF[c (64-lane fp32)]
  pass1: acc=vdup(−inf) ─vmax×G─▶ acc ──vcmax──▶ vdupv ─▶ m_c_brc (broadcast)
  merge: prev_brc = m_run ──────▶ m_run = vmax(m_run, m_c_brc)      ┐
  pass2: 重装载 c ─vexpdif(c,m_run)─▶ e ─vadd×G─▶ accs ──vcadd──▶ s_c (lane-0)
  merge: s_run = vadd(vmul(s_run, vexpdif(prev_brc, m_run)), s_c)   ┘ ← m_run/s_run
        （alloc_var 寄存器，行作用域，跨 chunk SimdVF 块驻留——R5）
  [chunk 循环 num_full + 尾 chunk，串行（running stats 顺序依赖）]
  ▼ 行尾 epilogue SimdVF 块
out = vadd(m_run, vln(s_run)) ─vcvt─▶ vsts(lse_ub[r], ONEPT) ──▶ UB[lse_ub (block_m,)]
  │ 任务尾 T.copy(lse_ub[0:rows_this], y[base:]) ──MTE3──▶ GM[y (M,)]
```

VRF 全程流式（~13 个 2048-bit 寄存器 ≈ 3.3KB，零行数据驻留）；行数据驻留在 UB（x_ub 单 chunk 粒度双版本）。

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先调研（1.6.0，调研四问——输入公式/源算法只是候选之一，迁移任务在源算子解读后、耦合性判定前完成）；再做数学等价的公式优化（1.6.1）；再做循环/标量计算的向量化替代（1.6.2）；最后决策向量化轴与核内布局（1.6.3）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。1.6.1–1.6.3 的分析对象 = §1.6.0 调研选定、经 §0.6 重设计落地的 NPU 算法，不止步于源公式直译。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：同一数学语义（行 logsumexp，**大 N 超驻留语境**）的算法族。调研深度：**完整调研**（规约/统计类算子）。执行位置：Phase M0 之后、Phase M1 之前（调研结论是 §0.5/§0.6 的判定输入）。**本变体的调研特有维度**：① chunk 划分粒度（BLOCK_N = tile_n 直保留 vs pow2 重划——SIMT 兄弟变体被迫后者的约束在 SimdVF 是否存在）；② running 统计量的驻留层级与更新粒度（寄存器 per-chunk vs 组粒度 online vs UB 往返）；③ 在线 vs 两遍重读 vs 多 kernel 部分合并（大 N 语境下的扫描遍数权衡——single 变体调研中「驻留可行则 online 非必需」的结论在本变体反转：驻留不可行是本变体的前提）。
> 信息源：`.agents/skills/tilelang-op-design/references/algorithm-candidates.md`（**ALG-softmax** 命中行〔family 含 logsumexp，R2 online softmax 行〕+ **ALG-reduction** row-reduction 形态）、pattern-library（`TRAP-NAIVE-ONLINE-MERGE` measured / `P-LAYOUT-ROW-RESIDENCY` measured / `P-LAYOUT-TAIL-STATIC-EXTENT` measured / `D-LOGSUMEXP-MIGRATION-FIDELITY` measured（fp16-e 50–60× 实测回退）/ `CONST-valu-model` / `CONST-mte-bandwidth-model` / `CONST-capacity-a5` / `CONST-vrf-capacity-a5` / `CONST-arch-a5` measured）、`examples/ascend/` SimdVF 同类（flash_attention core.py / deepgemm transform_sf / per_token_cast_to_fp8）、`testing/ascend/language/`（simdvf_cast UNPK+vcvt 位级回归 / test_simdvf_arithmetic serial_vreg_accum 循环携带先例）、`examples/pilot/online_softmax/online_softmax.py`（streaming 路径完整同构先例，measured 63/63 PASS）、SIMD single 姊妹 DESIGN（同族先例完整调研）+ SIMT tiled 兄弟 DESIGN 及其 Stage 4 perf 记录（pow2 重划代价 / fp16-e 50–60× 回退 / M≪56 并行度天花板 `[DESIGN_LIMIT]`）、源算法（§0.3/§0.4——源 tiled 已用 online flash 递推）、知识预注入（VP-2026-0014 vexpdif fp32-only·exp2 无快档 / CASE-logsumexp-harness）。互联网检索：未使用——本地源（候选表命中行 + streaming/flash/cast/arithmetic 四类先例 + 源仓双变体 + 两个兄弟变体完整调研与 Stage 4 实测记录 + pattern-library 十条目）已覆盖大 N logsumexp 在 SIMD 承载下的全部已知变体（online 单遍 / 两遍重读 / 多 kernel 部分合并 / Cube GEMV / 组粒度 online / pow2 重划 / fp16-e 域），无新增候选空间。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线 = 源算法（online flash 递推，N_padded 对齐 + 快慢双路径） | chunk 宽 tile_n（256 倍数任意值）；`N_padded = align_up(N,256)`；仅最后 chunk masked（`_needs_mask`） | — | — | ✅（基线） |
| 1 | 减 max 稳定化（flash 形式：合并基 = 合并后 m_run） | 同 §1.3 递推式（源已含） | 数学恒等 | 消溢出 + 合并式全部 exp 参数 ≤ 0 | 基线已含（必选） |
| 2 | exp→exp2·log2e 换底 | `e^t = 2^(t·log2e)` | 数学恒等 | GPU SFU 快路径 | ❌ **A5 弃选**：`CONST-valu-model` special_func 档位成员注记（measured，2026-10-16 三 pilot 实证）：exp/exp2/log2 同入 8 ops/cycle 档（`src/ascend/codegen/intrin_rule_ascend.cc` L34–49 均注册）——换底无单位收益、反多 1 乘/elem；`tilelang/ascend/language/simd.py` 无 exp2 快档内在；VP-2026-0014 交叉印证（与 single 姊妹调研同源结论） |
| 3 | log 域拆分 / log2 换底输出 | `y = (m·log2e + log2(S))/log2e` | 数学恒等 | 无 | ❌ 每行仅 1 次 special（占比 1/chunk-instr ≈ 0.03%）；同上档位证据 |
| 4 | **BLOCK_N = tile_n 直保留**（非 pow2 重划） | chunk 宽 = wrapper tile_n（16640/25600 等非 pow2 256 倍数），64 倍数自动满足组宽整除 | 划分满足结合律（每个元素恰计入一个 chunk，flash 合并代数恒等；浮点下仅重缩放乘法分组舍入差） | 非 pow2 tile_n 下 chunk 数与源一致（wl3 4 vs 兄弟 pow2 7；(33,33000) 2 vs 3）——merge 指令 / DMA 迭代 / 常量重绑定同比例↓ | ✅（O2，本变体核心增量） |
| 5 | **vexpdif 融合**（SIMD 特有） | `e = vexpdif(c, m_run)` 单指令 = exp(c−m)，替代 `vsub`+`vexp` 两指令 | 公式恒等（同式融合）；舍入差 ≤ 减法 0.5 ulp 量级（§1.6.1 O3） | −1 VALU op/64 元素（vsub @51 档）；pass2 与 rescale 因子两处受益 | ✅（O3，flash core.py L130–133 实测在用） |
| 6 | **延迟跨 lane 归约**（SIMD 特有） | 逐组 `vmax(acc, c)` 累加 + 遍末**单次** `vcmax`/`vcadd`，替代逐组跨 lane 归约+合并 | max 精确（分组无关）；sum 为 fp32 分组重排（ulp 级） | 每 chunk −(G−1)×2 指令（wl2 G=256：−510/chunk ≈ chunk 指令 −28%） | ✅（O4） |
| 7 | VRF 整 chunk 驻留单装载 | pass1 装载转换后的 fp32 值驻留 `alloc_local` 寄存器数组，pass2 免重读重转 | 数值同链 | −G×(vld+vcvt)/chunk（pass2 免）≈ VALU −10% | ❌ **弃选**（R3/R4：wl2 需 256 个 / wl3 400 个数据寄存器（fp32 chunk 数据 64–100KB），`CONST-vrf-capacity-a5`（256KB assumption）25–39% 水位 + alloc_local 数组索引复杂度 + 与 m_run/s_run 跨块驻留（R5）叠加无先例风险；UB 重读 4.68K cyc/core 仅计算项 3.8%（D-2）非瓶颈） |
| 8 | 组粒度 online（每 64 元素组更新 running stats） | 逐组 `new_m = vmax(m, vcmax(c))`；`acc = vmul(acc, vexpdif(m, new_m)) + Σ vexpdif(c, new_m)` | 容差内等价——**但角点保真有坑** | −pass2 重读（UB 域） | ❌ **弃选**：① 每 64 元素组 +3 指令（vcmax + vexpdif + vmul）×G vs chunk 粒度 7 指令/chunk——wl2 每行 +512 指令 vs +14（37×）；② **角点保真分歧**——m 初始化 −inf 时首组全 −inf 触发 `vexpdif(−inf,−inf)=NaN`（部分 −inf 行误判）；初始化 −MAX_FLOAT 则全 −inf 行输出 ≠ 源 NaN 语义（§0.1 ②④；`TRAP-NAIVE-ONLINE-MERGE` measured 同类角点）；③ chunk 数据已在 UB，online 的 GM 流量收益为零（online 的价值在 chunk 间而非组间） |
| 9 | fp16/bf16 域 max 遍（省 pass1 vcvt） | pass1 `vmax(acc_h, x_h)` 128-lane 16-bit → `vcmax(acc_h)` → 单值转 fp32 | max 为精确选择，16-bit 域 max 值 == fp32 域（无舍入）→ 数学恒等 | max 遍 −2 vcvt −1 vmax / 128 元素 | ❌ **弃选**（与 single 姊妹同源结论）：① bf16 `vcmax` dtype 支持未文档化（`simd_inst.h` L706 无 dtype 限制声明，CCE 硬件支持无佐证——未文档化假设）；② NaN 行为依赖硬件 reduce 的 NaN 语义（未文档化）——fp32 域链不依赖（NaN 经 exp 路径传播，V4 位级锚点）；③ 收益 ~4% 不抵双域 max 链复杂度。Stage 4 可重估 |
| 10 | **fp16-e 域计算**（exp/累加在 16-bit 域，省全部 vcvt） | `e_h = vexp_h(x_h − m_h)` 16-bit | **不等价**（16-bit exp 精度崩塌 + 定义域 e^11 溢出） | −2 vcvt/64 元素 | ❌ **弃选（实测否决）**：`D-LOGSUMEXP-MIGRATION-FIDELITY`（measured）：SIMT 兄弟变体 Stage 4 实测 fp16-e 路径 **50–60× 精度回退**（bf16 不可用、fp16 违反率 100%）；`VP-2026-0014`：`vexpdif` fp32-only（强校验）——16-bit 装载必须 vcvt 提升，融合内在不可用 |
| 11 | Cube GEMV 归并（sum = e·1 内积） | exp 后 `T.gemm(e_l1, ones_l1, c_l0c)` | 数学等价 | Cube MAC 替代 Vector 归约 | ❌ **弃选**：exp 主导段仍在 Vector（逐元素 special），Cube 只接管 sum——多出 UB→L1 ND2NZ + L0C→UB dual_copy 往返（`CONST-mte-bandwidth-model` ND2NZ 62 cycles base）+ mixed 核复杂度；`vcadd` 已是单指令全归约（flash core.py L134 实测），归约非瓶颈（D-2：special 61% 主导）。违反流量最小化 |
| 12 | 多 kernel 部分合并（N-split 跨核 + partial (m,s) 落 GM + 合并 kernel） | 各核算行部分 (m_p, s_p) → GM → 合并 kernel flash 合并 | 数学等价（划分结合律） | M≪56 时并行度 M→M×nsplit | ❌ **弃选**：① wl3 (M=4) 理论受益（4→56 核），但 (max,sum) 合并非 additive——须第二 kernel 或跨核 flag 定序合并（partial GM 往返 + 合并段串行化 + 浮点合并顺序变化使 V1/V4 位级保真论证失效）；② wl2（M=1024）56 核满载零收益；③ D-2 wl3 计算项 20.9K cyc/行 与 SIMT 兄弟 Stage 4 采纳值（17.6µs）同量级，M=4 是 wrapper 契约边缘 shape；④ 同族两变体（single SIMD 实证 + SIMT tiled 实证 + 兄弟 Stage 4 `[DESIGN_LIMIT]` M≪56 天花板记录）均为行并行单 kernel 形态。列 Stage 4 观察项（§9.1 风险 3） |
| 13 | 两遍重读（先全行 max 扫描，再 exp+sum 扫描，均 chunk 流式） | pass A：流式求全局 m；pass B：流式求 Σexp(x−m) | 数学等价（解析式直译） | 免 online 合并（7 指令/chunk = 0.39%） | ❌ **弃选**：① GM 扫描 2 遍（流量 +100%——online 单遍是本变体存在理由；D-2 wl2：MTE 带宽需求 12.0K→24.0K cyc/core，带宽余量 10.2×→5.1×——计算主导域下端到端增量被流水掩盖，**故流量非决定性**）；② **角点保真破坏（决定性）**——两遍式 = single 形态语义：首 chunk 全 −inf 且后续含有限值的行给出**有限值**，源 tiled online 递推给出 **NaN**（§0.1 ④、verify_equiv V5 实测分歧 9.70 vs NaN）——违反迁移保真；③ 省下的 merge 指令仅 0.39%（wl2） |
| 14 | 哨兵钳位（m_run 初始化 −MAX_FLOAT 防 −inf−−inf NaN） | init `m_run = vdup(−3.4e38)` | **不等价**（边界语义改变） | 消 NaN 角点 | ❌ **弃选**：全 −inf 行 y = −MAX_FLOAT + log(0) = −inf ≠ 源 NaN（§0.1 ②）；首 chunk 全 −inf 角点同理变为有限值（§0.1 ④）——源以 NaN 传播为语义而非缺陷（与 pilot online_softmax 的 O6 哨兵不同语境：该处是无 −inf 数据域的防御，本处 −inf 是契约内数据）。V5 位级锚点保持源行为 |

**R2 在线算法**：**有，且源即在线**——本变体与 single 的调研问题方向相反（single 问「驻留可行时 online 是否必要」，本变体问「驻留不可行时的在线算法族选哪种」）。证据三层齐全：① 源仓自身：`_logsumexp_kernel_tiled`（L105–194）即 online 变体（分块 running (max,sum) + flash 重缩放——「存在在线算法」的直接证据）；② 参考表：ALG-softmax R2「online softmax（分块 running max/sum + 重缩放）」命中行；③ 结构判据：logsumexp 的 (max, sum) 统计量均可增量合并 → 在线变体存在。**粒度轴调研**（本变体特有）：online 更新粒度 ∈ {chunk 粒度（源/选定）、组粒度（R1 #8）、行批粒度（= chunk 粒度 × block_m，与粒度无关）}——组粒度弃选（指令净增 37× + 角点保真分歧 + 零流量收益），**chunk 粒度选定**（merge 开销 O(rows·chunks) = 7 指令/chunk/行 = wl2 0.39% 指令占比，D-2）；**两遍重读与多 kernel（R1 #12/#13）作为「在线的替代」同台比较后弃选**（R3 表：扫描遍数 2/4 vs 1）。

**R3 算法复杂度对比**（逐候选四口径；口径说明：FLOPs/elem 中 vcvt/vmax/vadd 各计 1 op、vexpdif/vln 计 1 special（折算按 `CONST-valu-model` special = 8 ops/cycle/AIV，assumption）；GM Bytes/elem = 输入读 + 输出写（输出 M 个 << M·N，忽略）+ 中间 GM 往返；**UB 重读量**（SIMD 特有口径补充，计入访存权衡不计 GM）：pass2 重读 elem B/elem；扫描遍数 = 同一 GM 数据读写次数；中间缓冲 = 行级驻存需求峰值（UB + VRF）；代表 workload = wl2 (M=1024, N=32768, bf16, tile_n=16384, block_m=1)）：

| 算法候选 | FLOPs/elem | GM Bytes/elem | 扫描遍数 | 中间缓冲/行 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线直译（N_padded 256 对齐 + 快慢双路径 + fragment 线程直译形态） | ~5 op + 1 special | 2（读 2B；对齐 N）至 2.08（非对齐 N=33000：N_padded=33280，+0.85% pad 列） | 1 | x_ub block_m·tile_n·elem（+ N_padded pad 列）+ fragment 寄存器压力（无线程域承载参考）+ running stats 须 UB 往返（SSA 不能跨 SimdVF 域——§0.5 R5 判定输入） | M 行独立；无跨核归约；masked 路径逐元素谓词开销 |
| **选定：online flash（chunk 粒度）+ BLOCK_N=tile_n + UB 两遍重读 + 延迟跨 lane 归约 + vexpdif 融合 + alloc_var stats 驻留** | ~4 op + 1 special（vcvt×2 + vmax×1 + vadd×1 + special×1；vsub 被融合消除）+ 7 merge 指令/chunk/行（0.39% 摊薄） | 2（任意 N 紧凑——无 pad 列）+ UB 重读 2B（16-bit）/4B（fp32）——VRF↔UB 256B/cycle，wl2 4.68K cyc/core = 计算项 3.8% 非瓶颈 | **1**（单遍 GM——online 的定义性收益） | x_ub 2·BLOCK_N·elem（双版本 ≤220KB 上界，wl2 65.5KB）+ VRF ~13 向量寄存器 ≈ 3.3KB（含 m_run/s_run）+ lse_ub ≤64B | M 行独立（M 任务 / 56 核 persistent）；无跨核归约 |
| pow2 BLOCK_N 重划（SIMT 兄弟被迫形态） | 同选定 | 同选定 | 1 | 同选定（x_ub 同 BLOCK_N 档） | 同选定；**非 pow2 tile_n 下 chunk 数虚增**（wl3 7 vs 4、(33,33000) 3 vs 2）——merge/DMA 迭代/常量重绑定 +75%/+50%；SimdVF 无 fragment，重划无存在理由 |
| 组粒度 online（R1 #8） | ~4 op + 1 special + 3 op/64 元素组（+37× merge 指令） | 2（UB 重读 0） | 1 | 同选定 | 行内组序依赖（削弱双缓冲流水）；角点保真分歧（R1 #8） |
| 两遍重读（R1 #13） | ~4 op + 1 special（与选定同量；仅省 7 merge 指令/chunk = 0.39%） | **4**（读 2×2B） | **2** | x_ub 同选定（两遍共用） | 同选定；**角点保真破坏**（首 chunk 全 −inf 行为分歧，V5）——决定性弃选依据 |
| 多 kernel 部分合并（R1 #12） | ~4 op + 1 special + 合并段 | 2 + partial 往返（2×4B×M×nsplit）+ 启动 ×2 | 1（X 单遍）+ partial 2 | 同选定 + partial GM 缓冲 | M×nsplit 任务 / 56 核（wl3 4→56）；(max,sum) 非 additive 合并段串行 + 跨 kernel 同步 + 合并顺序位级保真失效 |
| Cube GEMV（R1 #11） | ~4 op + 1 special + Cube MAC | 2 + ND2NZ/dual_copy 往返 | 1 | 追加 L1 staging (e_l1) + L0C | mixed 核协调开销；exp 主项仍 Vector |
| fp16-e 域（R1 #10） | ~2 op（省 vcvt） | 2 | 1 | 同选定 | **实测 50–60× 精度回退**（measured 否决，非估算弃选） |

**R4 硬件亲和性评估**（逐候选对照检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水融合 / 跨核结构；负向淘汰依据已附）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线直译 | 纯 Vector ✓；special 1/elem 撞最窄吞吐档（`CONST-valu-model` special=8） | N_padded pad 列驻留浪费；fragment 形态在 SimdVF 无承载 | N_padded 256 列对齐（GPU 约束直译，A5 DMA 字节级行长无此需求）；逐元素掩码无线程域承载 | ✓（JIT 常量） | 双路径分支运行期化（`_needs_mask`） | ⚠️ 可行但约束直译无益——**降级为结构参照** |
| **online flash + BLOCK_N=tile_n（选定）** | Vector ✓；special 1/elem（公式层已最小）；vcmax/vcadd 单指令全归约（`simd_inst.h` L688–710）；vexpdif 融合 −1 op；merge 7 指令/chunk/行寄存器合并 | x_ub 双版本 ≤220KB 上界（SimdVF UB 248KB × 90%，`CONST-capacity-a5`）✓；**VRF ~3.3KB 零压力**（vs SIMT 128KB 封顶）✓；alloc_var stats 驻留零 UB 占用 | tile_n 256 倍数自动 64 整除（组宽）✓ 零尾组；尾 chunk rem 任意（组级掩码吸收）；vld UNPK_B16 组宽 64（cast 回归位级） | BLOCK_N/num_full/rem/full_m/rem_m 全 JIT 常量 ✓；尾 chunk Python 静态分支 ✓；尾批 TIR if ✓（先例实证） | x_ub 双版本（num_stages=2）chunk 间 MTE2/VALU 重叠 ✓；行边界共享双缓冲链 ✓；persistent 多波 ✓ | ✅ **选定（主选）** |
| pow2 重划 | 同选定 | 同选定 | **人为引入非整除**：BLOCK_N < tile_n 时 chunk 数虚增（wl3 7 vs 4） | ✓ | ✓ | ❌ 弃选（SimdVF 无 fragment 约束——重划是 SIMT 兄弟的约束残留，纯负优化） |
| 组粒度 online | 同选定 + 3 指令/组 | 同选定 | ✓ | ✓ | 行内组序依赖削弱双缓冲 | ❌ 弃选（R1 #8：37× merge 指令 + 角点保真 + 零流量收益） |
| 两遍重读 | ✓ | 同选定（两遍共用 x_ub） | ✓ | ✓ | 两遍串行（pass B 依赖 pass A 全行终值——跨 chunk 全局依赖，无双缓冲重叠） | ❌ 弃选（R1 #13：GM 2 遍 + 角点保真破坏） |
| 多 kernel 合并 | ✓ | + partial GM | ✓ | ✓ | 跨 kernel 间隙 + 合并段串行 | ❌ 弃选（R1 #12：复杂度 + 位级保真失效；wl3 边缘 shape 不值） |
| Cube GEMV | exp 段 Vector + sum 段 Cube 混置 | + L1/L0C staging | 须凑 C0 分形（`CONST-gemm-dim-constraints`） | ✓ | mixed 协调 | ❌ 弃选（R1 #11：归约非瓶颈，转换开销纯增） |

**调研结论**：选定 **「online flash 递推（chunk 粒度，BLOCK_N = tile_n 直保留）+ UB 两遍重读 + 延迟跨 lane 归约 + vexpdif 融合 + alloc_var 寄存器驻留 running stats + 尾 chunk 组级 −inf 掩码」**。关键依据：① R2/R3——超驻留大 N 语境下 online 单遍是唯一单遍扫描族（两遍重读 GM +100% 且破坏角点保真、多 kernel 加合并段与同步、Cube GEMV 归约段非瓶颈徒增往返）；② R3——chunk 粒度 merge 摊薄 0.39%（wl2），BLOCK_N=tile_n 使非 pow2 tile_n 下 chunk 数保持源划分（vs pow2 重划 +75% chunk 数 @wl3）；③ R4——纯 AIV 负载、容量全档 ✓（x_ub 双版本 65.5KB@wl2 / 102.4KB@wl3 ≤ 220KB 上界、VRF 3.3KB 零压力）、组对齐整除（256 倍数 → 64 整除零尾组）、persistent 双缓冲流水可融合、无跨核归约；④ 弃选项中两项为**实测否决**（fp16-e 50–60× 回退 `D-LOGSUMEXP-MIGRATION-FIDELITY`；M≪56 天花板兄弟 Stage 4 记录），其余为量级差或语义保真弃选。与基线的结构差异一句话：**N_padded 对齐 + 快慢双路径 + fragment 线程直译（源）→ BLOCK_N=tile_n 紧凑 + full/tail 静态二分支 + 显式 SIMD 指令流 + alloc_var 跨块驻留（选定）**，递推公式逐式恒等。源优化意图承接：#1 online 递推（保留）/ #2 chunk 分块（保留，粒度重算 R2）/ #3 快慢双路径（更优形态承接：静态二分支）/ #4 SMEM staging（UB 单跳 + UNPK 装载承接）/ #5 稳定化（保留）/ #6 树形归约（vcmax/vcadd 等价替换 + 延迟归约更优）/ #7 合并标量最小化（寄存器合并承接）/ #8 行批（persistent + 行串行承接）——逐项见 §0.5，无静默丢弃。调研范围：ALG-softmax/ALG-reduction 命中行、examples 四个 SimdVF 同类 + pilot online_softmax streaming 先例、pattern-library 十条目（四 measured：TRAP-NAIVE-ONLINE-MERGE / P-LAYOUT-ROW-RESIDENCY / P-LAYOUT-TAIL-STATIC-EXTENT / D-LOGSUMEXP-MIGRATION-FIDELITY；CONST-arch-a5 measured；其余 assumption：CONST-valu-model / CONST-mte-bandwidth-model / CONST-capacity-a5 / CONST-vrf-capacity-a5）、源仓双变体结构、两个兄弟变体完整调研 + Stage 4 实测记录、知识预注入两条目。

**设计期估算（D-2）**：估算下界(相对)：**计算项（VALU special_func）主导，约为流量项（带宽项）的 ~10 倍**（代表 workload wl2 (1024, 32768) bf16：流量 67.1MB → GM 读 67.1MB/(56×100 B/cycle) ≈ **12.0K cyc/core**〔`CONST-mte-bandwidth-model` assumption，AIV MTE2 宽行档；+ MTE base 项 3.3K + MTE3 3.3K → 含 base 18.6K〕；计算 1024 行×2 chunk×(1,030 ALU 指令 + 257 special 指令) → **122.7K cyc/core**（均值；最重核 19 行 127.5K）〔`CONST-valu-model` assumption：ALU 51 lane-ops/cycle、special 8 lane-ops/cycle——**special 占计算项 61%**；单源脚本 `history_version/design_budget_calc.py` 复算可重现〕；**UB 重读流量**：pass2 重读 2B/elem ÷ VRF↔UB 256B/cycle = **4.68K cyc/core**——计算项的 3.8%，不构成瓶颈；容量项 = 0——单遍 GM 读、无 L0C 往返、无分块重扫、VRF 3.3KB）。计算/流量比（带宽项）10.2×（含 base 6.6×）；merge 指令占比 0.39%（wl2）。主选 vs 弃选相对比：vs 两遍重读 GM 流量 2×（带宽余量 10.2×→5.1×——计算主导域下端到端被流水掩盖，其弃选决定性依据为角点保真破坏〔V5〕而非流量）；vs pow2 重划 wl3 merge+DMA 迭代 +75%；vs 组粒度 online merge 指令 +37×。**口径不确定披露**：special_func「8 ops/cycle」的 op 粒度（element vs 向量指令级）在 `src/ascend/transform/estimate_latency.h` L58–70 未区分——若为向量指令粒度则瓶颈反转为 MTE 主导（与 SIMT 兄弟变体/single 姊妹风险同源）；vcvt 转换吞吐按 VALU 51 档折算（档位未单列）。两种解释下主选结构（special 已最小化 + 流量已单遍最小化 + UB 重读 26× 余量）均为最优；该常数列 §9.1 风险 4，Stage 4 npusim/msprof 实测归因标定。A5 冷启动期 assumption 基估算，常数待标定，容量按 90% 折扣。

---
