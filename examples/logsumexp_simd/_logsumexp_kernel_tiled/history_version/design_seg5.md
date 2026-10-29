## 4. 数据规格与内存规划

### 4.1 输入张量

`x: (M, N)` 行主连续（wrapper 已归约/contiguous 到 2D——§0.2，kernel 契约不含维度归一化）；dtype ∈ {float16, bfloat16, float32}；GM。M ≥ 1、N ≥ 1 任意（**无 single 变体的 32,512 上限——任意大 N 是本变体存在理由**）。

### 4.2 输出张量

`y: (M,)` 行主连续，dtype 同输入；GM。

### 4.3 中间缓冲区

| buffer | 形状 | dtype | scope | 版本 | 生命周期 | 说明 |
|--------|------|-------|-------|------|---------|------|
| x_ub | (1, BLOCK_N) | 输入 dtype | UB（`T.alloc_shared`） | 2（双缓冲流水） | chunk 粒度 | 单行 chunk 数据；BLOCK_N = tile_n（clamp 后，§5.2）；**与 block_m 解耦**（行数不进容量式——无 single 的 SUB 封顶机制） |
| lse_ub | (block_m,) | 输入 dtype | UB | 1 | 任务粒度 | 行 lse 结果暂存（epilogue ONEPT 写入 → 任务尾切片写回 GM） |
| m_run / s_run | 标量级（2048-bit 向量寄存器） | float32 | VRF（`T.simd.alloc_var` local.var） | — | 行粒度 | running 统计量：m_run broadcast 形态 / s_run lane-0 形态；行作用域声明、per-chunk SimdVF 块内读写（风险 1 + fallback：`stats_ub (2,) fp32` UB 往返形态） |
| acc / accs / cv / e / m_c_brc / prev_brc / 常量（neg_inf/zero_f/one_pt/tail_mask） | 向量寄存器 | float32 / 输入 dtype | VRF | — | 组/chunk 粒度（流式） | VRF 合计 ~13 个 2048-bit 寄存器 ≈ 3.3KB，零行数据驻留 |

### 4.4 内存搬运路径

```
GM[x] ──MTE2 T.copy (1, BLOCK_N) 单行切片（full chunk；尾 chunk rem 列切片）──▶ UB[x_ub v0/v1 交替]
UB[x_ub] ──vld(UNPK_B16)+vcvt（VALU 装载，VRF↔UB 256 B/cycle）──▶ VRF[cv 64-lane fp32]
VRF[统计量链：vcmax/vcadd → m_run/s_run alloc_var 跨 chunk 驻留]（不出 VRF）
VRF[out] ──vsts(ONEPT)──▶ UB[lse_ub[r]]
UB[lse_ub[0:rows_this]] ──MTE3 T.copy 静态切片──▶ GM[y[base:+rows_this]]
```

无 L1/L0A/L0B/L0C 路径（纯 Vector 归约）；无 ND2NZ/fractal 转换；无 GM 中间往返（online 单遍——统计量全程寄存器）。

### 4.5 UB / VRF 内存预算

**单一来源**：`history_version/design_budget_calc.py`（确定性复算，嵌入数字与脚本输出一致；常数引用 `CONST-capacity-a5` status=assumption——UB 248KB 物理值、90% 折扣；`CONST-arch-a5` status=measured——vector_core_num=56 本机实查）。

- **UB_BUDGET_BYTES = int(248×1024×0.9) = 228,556B**（SimdVF 无 SimtVF 32KB 线程上下文扣减）。
- **静态预留 8,256B** = lse_ub（block_m × elem ≤ 16×4 = 64B，wrapper block_m ≤ 16；显式 config 更大 block_m 时 lse_ub 仍 ≪ 预留）+ 对齐取整/调度杂项保守余量（**非膨胀系数**——constants.md「90% 折扣之外勿叠加未证实膨胀系数」的口径下仅取缓冲取整级余量）。
- **BLOCK_N clamp 上界**（BLOCK_N = 最大 64 倍数 s.t. BLOCK_N ≤ tile_n ∧ 2·BLOCK_N·elem ≤ 228,556 − 8,256）：**16-bit 55,040 / fp32 27,520**。wrapper tile_n ≤ 32,512（`compute_tile_n` 预算封顶）→ 16-bit **恒不触发**（BLOCK_N = tile_n 直保留）；fp32 或显式超大 tile_n 由 clamp 兜底（chunk 划分变化语义中性——V6 划分不变性）。

| workload / 用例 | (M, N) dtype | tile_n | BLOCK_N | chunks | x_ub（2 版本） | 总占用（含预留） | 占预算 |
|----------------|-------------|--------|---------|--------|----------------|------------------|--------|
| wl2 attn-weights-32k | (1024, 32768) bf16 | 16384 | 16384 | 2 full | 65,536B | 73,792B | 32.3% |
| wl3 lm-head-logits | (4, 102400) fp16 / bf16 | 25600 | 25600 | 4 full | 102,400B | 110,656B | 48.4% |
| test `(4,32768)` | (4, 32768) fp16/bf16 | 16384 | 16384 | 2 full | 65,536B | 73,792B | 32.3% |
| test `(33,33000)` | (33, 33000) fp16 | 16640 | 16640 | 1 full + 尾（G_tail=255, k_tail=40） | 66,560B | 74,816B | 32.7% |
| test `(33,32768)` | (33, 32768) fp16 | 16384 | 16384 | 2 full | 65,536B | 73,792B | 32.3% |
| fp32 显式路径 @clamp | (1024, 32768) fp32, tn=32512 | 32512 | **27520（clamp 触发）** | 1 full + 尾（G_tail=82, k_tail=0） | 220,160B | 228,416B | 99.9% |

- **VRF**：~13 个 2048-bit 寄存器 × 256B = **3,328B**（常量 4 + 累加器 2 + 装载/中间 4 + stats 2 + epilogue 1，峰值同时活跃 ~10–13）；零行数据驻留（vs SIMT 兄弟 fragment 封顶 128KB）——`CONST-vrf-capacity-a5`（256KB assumption）压力可忽略。
- lse_ub 计入 8,256B 预留（≤64B）。

### 4.6 动态轴定义

本设计 **无动态轴**（`T.dynamic_axis` 不使用）：M / N / dtype / tile_n / block_m 全部为 JIT 编译期常量（per-shape JIT + lru_cache，源仓同款）；kernel 内全部循环边界（num_tiles、rows_this、num_full、G、G_tail、k_tail）为编译期标量——`T.Persistent` 的任务维 t 与 `T.serial` 的 c 为唯一运行期循环变量。

### 4.7 JIT 配置与调用面契约

```python
_logsumexp_kernel_tiled(M, N, dtype, tile_n)   # builder：host 静态化 + 契约防御（raise）
    → _func(block_m)                           # @tilelang.jit(target="ascend", out_idx=[1]) 懒工厂
    → kernel(x) → y                            # (x) 执行，out_idx=[1] 返回 y
```

- GPU `threads` 参数按 K3 移除（SimdVF 无线程概念，连内部推导也不存在）；`tile_n` 保留在调用面（接口不变，内部粒度按 SimdVF 约束重解释——§0.5 R2「保留（粒度重解释）」）。
- **无 N 上限 assert**（对照 single 变体的 `N > MAX_SINGLE_TILE_COLS` raise——本变体承接超限路径）；assert：M ≥ 1、N ≥ 1、tile_n ≥ 1、block_m ≥ 1。
- wrapper 对接（SIMD 家族 `logsumexp_simd.py`，Stage 5 集成）：`tile_n` 由 `compute_tile_n` 启发式传入（§0.2 落点复算：wl2 (1,16384) / wl3 (1,25600) / (33,33000) (1,16640)）。

## 5. Tiling 策略

### 5.1 计算类型

规约类（reduction）：行级 logsumexp，N 向 chunk 流式 online 归约；无 GEMM（C0 分形对齐 n/a）。

### 5.2 Block 划分

- **N 维（chunk 粒度）**：BLOCK_N = 最大 64 倍数 s.t. BLOCK_N ≤ tile_n ∧ 2·BLOCK_N·elem ≤ 228,556 − 8,256（**= tile_n 直保留**，wrapper 256 倍数 tile_n 自动满足 64 整除——full chunk 零尾组零掩码；clamp 上界 16-bit 55,040 / fp32 27,520，§4.5）。`num_full = N // BLOCK_N`、`rem = N % BLOCK_N`（仅尾 chunk 可能越界——与源 `_needs_mask` 同构但静态化）。**与 SIMT 兄弟变体的关键差异**：兄弟因 fragment pow2 约束被迫 BLOCK_N = pow2 ≤ tile_n（16640→16384、25600→16384，chunk 数虚增 wl3 7 vs 4、(33,33000) 3 vs 2）；SimdVF 无 fragment——chunk 划分与源一致，V6 划分不变性（含非 pow2 档）保证语义中性。
- **M 维（任务粒度）**：任务 = block_m 行（num_tiles = ceil(M/block_m)）；任务内**行串行**（`for r in range(rows_this)`，rows_this ∈ {block_m, rem_m} 静态）；**x_ub (1, BLOCK_N) 与 block_m 解耦**——无 single 变体的 SUB 封顶机制（容量式不含行数），任意 block_m 零容量影响。主流落点 block_m=1（wrapper tiled 路径偏好，§0.2）→ 行循环单迭代退化、任务 = 单行。
- **组粒度**：64 元素/组（2048-bit fp32 lane 宽；16-bit 输入 UNPK_B16 装载后同 64-lane fp32 域）。

### 5.3 约束分析

| 约束 | 分析 |
|------|------|
| UB 容量 | §4.5 表：最大档（fp32 clamp）228,416B ≤ 228,556B（99.9%）；主流 wl2 32.3% / wl3 48.4% |
| VRF 容量 | 3.3KB 流式，`CONST-vrf-capacity-a5`（256KB）零压力 |
| 组宽整除 | tile_n 256 倍数 → BLOCK_N 64 倍数 → full chunk 零尾组；尾 chunk rem 任意（组级 vsel 掩码吸收）；显式 config 任意 tile_n → clamp 取 64 倍数仍零尾组 |
| M 尾批 | TIR if + rows_this 静态 + 静态切片（R4，不依赖 T.copy OOB 行为） |
| N < BLOCK_N | num_full = 0 → 纯尾 chunk 路径（G_tail/k_tail 由 rem=N 推导，同一尾 chunk 代码路径承接） |

### 5.4 注意事项（非整除 / 边界处理策略）

1. **N % BLOCK_N ≠ 0（尾 chunk）**：静态 rem 切片 DMA（只搬 rem 列，无 pad 搬运）+ G_tail 满组直载 + 尾组 `vsel(−inf)` fp32 域掩码（−inf 双幂元中性，O2；Python 静态分支 `if rem > 0` / `if k_tail > 0`，trace 期消除）。
2. **M % block_m ≠ 0（尾批）**：TIR if 尾批分支 + rem_m 静态行数 + `row_task(M − rem_m, rem_m)` 宏第二实例化（兄弟 `sub_batch` 同构）。
3. **rem = 0**：尾 chunk 分支不实例化（零开销）；**k_tail = 0 但 rem > 0**（如 fp32 clamp 档 rem=5248=82×64）：尾组 vsel 分支不实例化，仅满组。
4. **block_m > M**：full_m = 0 → 全部任务走尾批分支（rows=rem_m=M）；block_m=1 且 M=1：单任务单行。
5. **tile_n > clamp 上界**：BLOCK_N = clamp 值（chunk 数多于源意图，语义中性 V6；fp32 显式路径 99.9% 容量档，§4.5）。

### 5.5 分核策略（grid 划分 + AIC/AIV 分工 + 同步策略）⭐

权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（三要素）/ §2.1（设计要求）。

1. **grid 划分（核数依据实查）**：`T.Kernel(56)`——**56 = `vector_core_num` 本机实查值**。查询代码与返回值（本设计执行期复测，2026-10-17）：

   ```python
   import torch

   p = torch.npu.get_device_properties()  # name / cube_core_num / vector_core_num
   # → name: 'Ascend950PR_9579', cube_core_num: 28, vector_core_num: 56
   ```

   依据：`CONST-arch-a5`（status=**measured**，2026-10-16 三 pilot 实查转正 + 本任务复测一致；28/56 非 2 幂，「惯例 32 blocks」假设在本机不成立——grid 不做整除性假设，persistent 调度器消化任意任务数）。任务维 = num_tiles（M/block_m 行块任务），多波由 `T.Persistent` 调度器分配（wl2：1024 任务/56 核 ≈ 18.3 波内摊）。

2. **AIC-AIV 分工**：**纯 Vector 负载**——全部计算（装载提升/max/exp/sum/合并/epilogue）在 AIV VALU，数据搬运 MTE2/MTE3 亦为 AIV 侧引擎；**无 Cube 段**（无 `T.gemm`）→ 不使用 `T.MixedKernel`/sid 划分/dual_copy（mixed 仅当单 kernel 含 Cube + Vector 段）；本 kernel 运行期 AIC（28 核）空闲。任务到核映射由 `T.Persistent` 一维调度（无跨核通信/归约——行间独立）。

3. **同步策略**：**`auto_schedule: yes`**——依赖 AutoSchedule 自动插 flag 对（MTE2→VALU per chunk x_ub 版本交替、VALU→MTE3 lse_ub 依赖、行/任务边界）；无手动 `T.ascend_set_flag`/`T.ascend_wait_flag`（flag 预算由 AutoSchedule 管理，不占用手动配对预算）；alloc_var 为引擎私有寄存器（`insert_sync.cc` L1308 local.var 依赖跳过跨引擎同步——无 VALU 内部同步需求）。详见 §7。

## 6. 循环与调度结构

### 6.1 循环结构总结

| 层级 | 循环/结构 | 形态 | 选择理由 |
|------|-----------|------|---------|
| 任务调度 | `for t in T.Persistent([num_tiles], 56, core_id, num_stages=2)` | Persistent | 56 核持久行块调度 + 多波摊薄；num_stages=2 使 x_ub 双版本流水可推断（single/兄弟同构） |
| 尾批分支 | `if t < full_m` | TIR if（运行期） | M 尾批边界语义；分支体 = 宏两次静态实例化（rows 常量） |
| 行循环 | `for r in range(rows_this)` | Python 展开（静态行数） | 行串行（R1/R5：alloc_var 单寄存器语义不可行索引，§1.6.2 #8）；主流 bm=1 单迭代退化；显式 config 大 block_m 的展开体积见 §9.1 风险 8 |
| chunk 循环 | `for c in T.serial(num_full)` | T.serial 串行 | running stats 循环携带依赖（online 递推顺序本质）；保持 x_ub 双版本流水顺序（`P-LAYOUT-ROW-RESIDENCY`） |
| 组循环 ×2 | `for g in range(G)` / `range(G_tail)`（pass1/pass2 各一） | Python 展开 | 体内全向量化（3–4 向量指令/组）；G 静态 ≤ 430（single 姊妹 G ≤ 508 实证同形态）；两遍扫描数据流本质 |
| 尾 chunk / 尾组 | `if rem > 0` / `if k_tail > 0` | Python 静态分支 | trace 期消除（非运行期分支）；边界全静态（gate 要求 ✓） |

**不使用**的结构：`T.Pipelined`（显式软件流水由 `num_stages=2` + `annotate_buffer_versions` 承接，同族两变体实证形态）；`T.Parallel`（SimdVF 无线程域——元素并行在指令流内）；动态轴（§4.6）。

### 6.2 循环伪代码

见 §3.3（完整 kernel 伪代码——Block 级结构 = `T.Kernel(56) > T.Persistent > TIR if > range(rows) > T.serial(num_full) > SimdVF{range(G)×2} > 尾分支 > epilogue SimdVF > MTE3`）。

### 6.3 流水线优化

- **x_ub 双版本流水**（`annotate_buffer_versions({x_ub: 2})` + `num_stages=2`）：chunk c+1 的 MTE2 预取与 chunk c 的 SimdVF 计算（VALU）重叠——跨引擎重叠由 AutoSchedule 插 flag 对（§7）。
- **行边界共享双缓冲链**：行 r+1 的 chunk 0 预取可重叠行 r 的末 chunk 计算/epilogue（比 SIMT 兄弟的 (SUB, BLOCK_N) 整批 DMA 更细的流水粒度——行粒度任务切换无流水断流）。
- **重叠非并行**：chunk 间计算保持串行（running stats 依赖，§1.6.2 #9）——流水只隐藏 MTE 延迟；D-2 下 MTE 占计算 10%（带宽口径），流水收益上界即该比值。

### 6.4 尾块处理

M 尾批：TIR if + rem_m 静态（§5.4-2）；N 尾 chunk：静态 rem 切片 + 组级掩码（§5.4-1）；双尾（M、N 同时非整除，如 (33,33000)）：尾批分支内行循环 × 每行尾 chunk——两条边界处理正交组合，测试用例 (33,33000) 覆盖。

## 7. 同步策略

### 7.1 同步模式

**`auto_schedule: yes`**（默认自动调度）——依赖 AutoSchedule 自动插同步；**无手动 `T.ascend_set_flag`/`T.ascend_wait_flag`/`T.ascend_pipe_barrier`**；不传 `PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False`（仅完全手动核需要）。

### 7.2 同步点说明

| 依赖边 | 引擎对 | 处理 |
|--------|--------|------|
| x_ub 写（MTE2 chunk c+1 预取）→ x_ub 读（VALU chunk c 计算） | MTE2→VALU | AutoSchedule 按版本交替自动插 flag 对（`num_stages=2` + 双版本声明——同族两变体实证形态） |
| lse_ub 写（VALU epilogue vsts）→ lse_ub 读（MTE3 任务尾 copy） | VALU→MTE3 | AutoSchedule 自动（版本 1，任务粒度） |
| m_run/s_run 跨 chunk 读写 | VALU→VALU（同核） | **无需同步**——local.var 引擎私有寄存器（`insert_sync.cc` L1308 对 local.var 依赖跳过同步插入） |
| 行间/任务间/核间 | — | 无数据依赖（行独立、无跨核归约）——无同步需求 |
| fallback 路径（风险 1 触发时） | VALU→VALU 经 UB | stats UB 往返需 `mem_bar("VST_VLD")` 排序 vsts/vld（flash core.py L120–126 形态）——届时按 fallback 方案补配对说明 |

### 7.3 pass_configs 配置

```python
# 默认自动调度，不传 TL_ENABLE_AUTO_SCHEDULE（仅完全手动核需要 False）；
# JIT 入口即 §4.7 的 @tilelang.jit(target="ascend", out_idx=[1]) 懒工厂。
```

---
