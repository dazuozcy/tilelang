# 硬件常数表（设计期 roofline 与调优定量口径）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)）。**本表是硬件常数的唯一事实源**（D-2：evolver 蒸馏新 D 类常数时追加进本表，Tier 0）；设计期 roofline 估算协议（如何用本表算下界）见 `_shared/standards/hardware-cost-model.md`。条目带 front-matter（schema 见 INDEX.md §5），`source` 指向证据出处（provenance，允许失效——常数数值与适用条件已自包含于本表）。
>
> **使用纪律**：① 引用常数须带版本戳（工具链变更后自动待重验，`kb_stale_check.py` 检测）；② 常数绑定设备（当前全部为 Ascend A5/950，npu-arch dav-3510）；③ 外推前核对适用条件。
>
> **A5 冷启动状态（2026-10-16，P0 标定）**：当前条目全部为 `status: assumption`——证据为 tilelang-fork 源码/文档（无 910B2C 式实测）。**设计期使用须打折扣**：容量按 90% 预算，并在 DESIGN.md 中披露所引条目与折扣；首次真机标定任务（vecadd copy-floor、GEMM roofline、flag/launch 开销）后转 `measured` 并附 repro。频率/绝对 TFLOPS/GBs：仓库无静态值（代价模型纯 cycle 域），**换算绝对时延/吞吐必须用 do_bench 实测**，禁止编造频率。

---
id: CONST-capacity-a5
kind: constant
family: [general]
apis: []
dtype: []
device: A5/950 (dav-3510)
status: assumption
origin_task: a5-adaptation-P0-20261016
toolchain: tilelang-fork fc32f47c；CANN 9.x（bicheng --npu-arch=dav-3510）
source: src/ascend/transform/auto_schedule.cc:106-140（GetSharedMemoryLimit/GetL1MemoryLimit/GetL0C/L0A/L0BMemoryLimit）
repro: repro-missing
---

### 片上容量（AutoSchedule 容量约束，A5）

- **UB（"shared"）= 248KB**（`kPhysicalUnifiedBufferSize = 248*1024`）；**含 `T.SimtVF` 时可用上限 = 248KB − 32KB（SIMT_VF 线程上下文）= 216KB**。注意非整值 256KB——勿按 256KB 预算。**SimdVF（`T.SimdVF`）模式无 32KB 线程上下文扣减**——设计预算按 248KB 计：90% 折扣口径 SimdVF `int(248×1024×0.9)=228,556B` vs SimtVF `int(216×1024×0.9)=199,065B`（logsumexp_simd single DESIGN §4.5 实算口径，2026-10-17）；SimdVF 行串行 + VRF 流式形态下 SUB 封顶仅剩 UB 单约束（无 SimtVF 的 x_frag ≤128KB VRF 项——非 pow2 N 的 TILE 膨胀压 SUB 效应不存在）。注意 TileOPs wrapper 侧 `device_smem_budget()` 固定返回 216KiB、与编程模式无关（wrapper 契约落点偏保守，kernel 内预算不受此限）。
- **L1（CBuf，"shared.l1"）= 512KB**（auto_schedule.cc 注释 "512 KB L1 cache on Ascend"）。
- **L0C（累加器，"shared.l0c"）= 256KB**。
- **L0A = L0B = 64KB**（GEMM K-sub 切分参考：`tilelang/ascend/op/gemm/gemm_mad.py` 取 `l0_capacity = 65536`、`num_l0_stages = 2`）。
- 缓冲对齐（merge UB）：L0C 1024B；L1/L0A/L0B 512B（`src/ascend/transform/merge_ub_common.h` `AscendScopeAlignBytes`）。
- 容量检查须计入 layout padding、buffer versions（AutoSchedule multibuffering）、编译器保留（`tilelang-ascend` skill §4.3）；`T.annotate_unlimit_memory("shared","shared.l1")` 只解除调度器约束、不扩容。
- GM/L2/HBM 容量：运行时查询（`torch.npu.get_device_properties()`），无静态值。
- **待实测转正项**：AutoSchedule multibuffering 下的 UB 实际膨胀系数（旧 910 系栈曾实测 ×1.10–1.12~1.7〔历史注记〕，A5 未知——设计期在 90% 折扣之外勿再叠加未证实膨胀系数，超大 block 探索以编译器报错为准并回写本条目）。

---
id: CONST-arch-a5
kind: constant
family: [general]
apis: []
dtype: []
device: A5/950 (dav-3510)
status: measured
origin_task: pilot-silu/gemm_silu/online_softmax-20261016（核数实查项转正；其余子项仍 assumption）
toolchain: tilelang-fork 48345cd7；CANN 9.2.0 + dav-3510；Ascend950PR_9579 实机
source: tilelang/carver/arch/driver/ascend_driver.py:27-40（查询接口）；examples/pilot/{silu,gemm_silu,online_softmax} 三任务实查记录
repro: repro-missing
---

### 核组织与工具链（A5）

- **AIC:AIV 物理编组 = 1:2**（`__mix__(1, 2)`，dav-3510 直连 AIC→AIV 数据通路强制）；编译器可见具体核 = {Vector0, Cube, Vector1}。
- **AIV sub-core（sid）仅可 1 或 2**（`T.MixedKernel(..., sids=1|2)`；默认 2）；`sids=1` 只激活 AIV0。
- **整芯片 AIC/AIV 总数：无静态常数**——运行时 `torch.npu.get_device_properties()` 的 `cube_core_num` / `vector_core_num` 实查；examples/ascend 惯例默认 32 blocks（`examples/ascend/deepgemm/config.py` `num_cores=32`）。设计文档核数声明须写"实查命令 + 实测值"。**〔2026-10-16 试点实测，P2〕本机 `Ascend950PR_9579`（126GB HBM、L2 128MB）实查 cube_core_num=28 / vector_core_num=56**——「惯例 32」假设在本机不成立（28=4×7 非 2 幂，grid 整除性陷阱见 silu 试点 RETROSPECTIVE）；3 个试点任务（silu/gemm_silu/online_softmax）均按 28/56 实查设计并一次通过。
- npu-arch = `dav-3510`（默认，`ASCEND_NPU_ARCH` 可覆盖）；bisheng 公共选项 `-O2 -fPIC -std=c++20 --npu-arch=<arch>`。
- flag/event 预算：核内 flag 8 个、跨核 flag 16 个（溢出经 RewriteFlagToBuf 压缩/spill）——手动同步方案须做 flag 配对预算。

---
id: CONST-gemm-dim-constraints
kind: constant
family: [gemm, cube]
apis: [T.gemm]
dtype: []
device: A5/950 (dav-3510)
status: assumption
origin_task: a5-adaptation-P0-20261016
toolchain: tilelang-fork fc32f47c；CANN 9.x
source: src/tl_templates/ascend/gemm.h:34-53,145-151；tilelang/layout/ascend.py:25-27,83-92,136-139；tilelang/ascend/op/gemm/gemm_mad.py:161-163,210-221
repro: repro-missing
---

### GEMM 维度约束（A5）

- **C0（Cube fractal 宽度）= 32B / 元素大小**：bf16/fp16=16、fp32=8、int8=32、fp4=64。
- **FRAC_N（行分形）= 16**（dtype 无关）；L0C 累加器固定 16×16 fractal（物理 `[N/16, M/16, 16, 16]`）。
- **M、N 必须是 16 的倍数**（static_assert）；**K % TILE_K_SUB == 0 且 TILE_K_SUB % C0 == 0**。
- L1-input GEMM 仅支持 **NT：`trans_A=False, trans_B=True`**（dav-3510 要求）；权重 GM 布局 **[N, K]**。L0-input GEMM 有独立布局处理，勿把 L1 限制推广。
- Blockscaled(MX)：K 必须 64 整除（一个 SF pair 覆盖 64 个 K 元素）；仅 bf16/fp8e4m3fn/fp4。
- GEMM 输入 dtype：fp16/bf16/fp32/fp8e4m3fn/fp8e5m2/fp4e2m1fn/int8；**INT8 MMAD 不支持（A5）**。
- L0C→UB dual_copy N-split：完整 N 须被 32 整除；GM→UB 右 padding 到 32B 边界；ND2NZ scatter rows 16 整除、行 32B 对齐。

---
id: CONST-cube-throughput-model
kind: constant
family: [gemm, cube]
apis: [T.gemm]
dtype: [fp16, bf16, fp32, fp8, fp4]
device: A5/950 (dav-3510)
status: assumption
origin_task: a5-adaptation-P0-20261016
toolchain: tilelang-fork fc32f47c；CANN 9.x
source: src/ascend/transform/estimate_latency.h:27-56（AscendLatencyParams；单位：ops/cycle/AIC，一次乘加=2 ops）
repro: repro-missing
---

### Cube 吞吐代价模型（AutoSchedule estimate_latency，A5）

- fp16 / bf16：**8192 ops/cycle/AIC**；fp8：16384；mxfp4：32768；HF32：4096（fp32 base latency 67）；fp32：512；fallback 8192。
- **INT8 MMAD 不支持（吞吐 0）**——int8 GEMM 需另寻路径。
- cube_base_latency=8 cycles；cube_unit_m/k/n=16；cube_min_ii=1、pipeline_depth=1。
- 交叉校准参考：`testing/ascend/auto_schedule/test_estimate_latency.py:70-82`（CANN 9.2 / npusim Ascend950 实测校准值，blockscaled 场景）。
- 绝对 TFLOPS 换算需频率（无静态值）——设计期比较用相对比值，绝对性能以 do_bench(msprof) 实测为准。

---
id: CONST-mte-bandwidth-model
kind: constant
family: [general, gemm, attention]
apis: [T.copy, T.dual_copy]
dtype: []
device: A5/950 (dav-3510)
status: assumption
origin_task: a5-adaptation-P0-20261016
toolchain: tilelang-fork fc32f47c；CANN 9.x
source: src/ascend/transform/estimate_latency.h:83-158（MTE/FixPipe/dual_copy/ND2NZ 参数；B/cycle + cycles）
repro: repro-missing
---

### MTE / FixPipe 带宽与延迟模型（A5）

- **AIC MTE2（GM→L1）= 100 B/cycle**（base 190、min_ii 64）；**AIV MTE2（GM→UB）= 100 B/cycle**（base 90、min_ii 13）；**MTE3（UB→GM）= 115 B/cycle**（base 180、min_ii 10）；**MTE3（UB→L1）= 256 B/cycle**（base 43）；L1→L0A/L0B = 256 B/cycle；FixPipe = 128 B/cycle（base 58）；**dual_copy：带宽 256 B/cycle、width_scale 2、base 55**。
- N-strided 事务 128B；峰值/平台/宽行/未知带宽（GM→UB）= 102/50/100/48 B/cycle，（UB→GM）= 86/40/110/40——**跨步行访问按相应档位折算，勿按峰值**。
- UB→L1 raw N-split：<160B/行按 3/4 速率，≥160B 满速率、上限 256 B/cycle。
- ND2NZ 转换开销（same-dtype 62 cycles base 等）——fractal 转换非免费，布局选择时计入。
- **待实测转正项**：以上为编译器代价模型假设，A5 真机 copy-floor 标定（vecadd 探针）后转 measured 并记录退化曲线。

---
id: CONST-valu-model
kind: constant
family: [elementwise, reduction]
apis: []
dtype: []
device: A5/950 (dav-3510)
status: assumption
origin_task: a5-adaptation-P0-20261016
toolchain: tilelang-fork fc32f47c；CANN 9.x
source: src/ascend/transform/estimate_latency.h:58-70（VALU 吞吐；ops/cycle/AIV）
repro: repro-missing
---

### VALU 吞吐与 VRF 带宽（A5）

- VALU（每 AIV）：add/sub/mul/min_max/cmp/logic = **51 ops/cycle**；div/mod = 13；special_func = 8。
- **special_func 档位成员注记（2026-10-16 实证，pilot 三任务）**：exp / exp2 / log2 同入 special_func 8 ops/cycle 档（`src/ascend/transform/estimate_latency.cc:1267` 同列；`src/ascend/codegen/intrin_rule_ascend.cc:34-49` 均注册）——GPU 式 exp2 换底在 A5 无单位代价收益，反多 1 次乘/elem 与 scale 字面量截断误差。
- **吞吐档位粒度口径 = lane-op 粒度（设计期复算锚点，未上机标定）**：`src/ascend/transform/estimate_latency.h:66` `special_func_throughput{8}` 未声明按指令还是按 lane-op 计——D-2 roofline 复算按 **lane-op 粒度**（指令数 × 64 lane ÷ 8 lane-ops/cycle）才与设计自算值吻合：logsumexp_simd tiled wl2 `(1030×64/51 + 257×64/8) × 2 chunk × (1024/56 行)` = 122,588 ≈ 设计 122.7K cyc/core（special 占比 61.3% ≈ 设计 61%）；按指令粒度折算差 64×（~1.9K）。VALU 51 档同口径按 lane-op 折算。npusim 实测标定前，设计期估算按此口径并在 DESIGN 披露（origin_task: logsumexp_simd tiled S2 REVIEW 机械复核，2026-10-17）。
- **标量 tanh 非 special_func 档位（实测反例）**：silu tanh 恒等变形实测 **1.80×** 劣化于直除式（388.42 vs 215.67µs，探针口径见 pattern-library/elementwise.md D-ELEM-ACT-FORMULA-PROBE），推断为软件例程——激活公式弃用 tanh 形式的实测依据（origin_task: pilot-gemm_silu-20261016T034510Z；截至 2026-10-16，CANN 9.2）。
- **VRF↔UB 带宽 = 256 B/cycle**（每 AIV）。
- SIMD 向量寄存器 256B（2048-bit = 64×fp32 / 128×fp16）；fill/copy/reduce/Parallel 区域须整除。
- 硬件 warp-reduce dtype：仅 fp16/fp32/int32/uint32 直接 lower 到 `asc_reduce_*` intrinsics（`tilelang-ascend` skill §10 陷阱 11）。

---

id: CONST-vrf-capacity-a5
kind: constant
family: [general, reduction]
apis: []
dtype: [fp32]
device: A5/950 (dav-3510)
status: assumption
origin_task: pilot-online_softmax-20261016T044814Z（溯源：该任务 REVIEW 维度 2 缺口提案 + Stage 3 get_kernel_source 实测样本）；logsumexp-harness 2026-10-16 追加（两函数 `--show-source` 寄存器样本 + S2 fragment 层级归属冲突消解——溯源允许失效）
toolchain: tilelang-fork dev build + CANN 9.x + bisheng dav-3510 + Ascend950PR_9579，2026-10-16
repro: repro-missing
---

### VRF（向量寄存器文件）容量与双口径预算（A5）

- **容量总量无文档化/源码化常数**：pattern-library 与 tilelang-ascend skill 均无条目（grep SKILL.md + references 零命中）——设计期按假设 **256KB** 起步，属 assumption，roofline 使用须打折扣并在 DESIGN.md 披露。
- **双口径预算规则**：per-thread reg 数不构成总量佐证——rmsnorm 先例 80 reg/thread 是 256 threads 配置（x_frag 64KB + w_frag 16KB ≈ 80KB 总量）；1024 threads × 64 reg/thread = 256KB 总量即达假设上限 100%。寄存器预算检视须分别核 per-thread 与总量两口径。
- **实测样本（2026-10-16，下界证据）**：online_softmax 主 shape [8192,8192]（BM=4/TILE=8192、threads=1024）生成的 AscendC 为 `float x_frag[32]; float e_frag[32];` + 12 标量 fragment = **76 fp32 reg/thread**，`__launch_bounds__(1024)`，**spill 计数 0**（在 rmsnorm 80-reg 先例水位内，未触假设上限）；x_frag 在步骤③后的死寄存器复用未被编译器回收（两个 (BM,TILE) fp32 fragment 各 32 reg 并置）。
- 转正方法：Stage 3 编译后首查 spill 计数（`get_kernel_source()` / 编译日志）或 npusim 反解；容量上限值待实测后翻转本条 status。
- **fragment 层级归属（2026-10-16，logsumexp S2 知识冲突消解）**：A5 SimtVF `T.alloc_fragment` 落 VRF per-thread 寄存器数组、**不占 UB 216KB 预算**——反证：online_softmax 主 shape UB 128KB + x_frag/e_frag 2×128KB fp32 fragment 并存真机编译运行 63/63 PASS（若 fragment 计入 UB 216KB 则该先例不可能编译）；TileOPs `_primitives.py` 的 "planned into UB" 注释为规划口径，与实测冲突时按「实测生成代码 > harness 成本模型注释」消解（生成 AscendC 为 `float x_frag[32]` 寄存器数组，非 UB 数组）。
- **追加实测样本（2026-10-16，logsumexp 双变体；SUB/行子批封顶把 VRF 从贴顶压到舒适水位）**：single `float x_frag[32]`（SUB=8×TILE=4096 / 1024 threads）+ m/s/out 标量 ≈ 48 regs/thread 无 spill；tiled `float x_frag[16]`（SUB=1×BLOCK_N=16384 / 1024 threads）+ 合并标量无 spill——均低于 76 reg 实测水位；x_frag ≤ 128KB 封顶 + SUB 联合封顶（SUB·BLOCK_N ≤ 32768）是 VRF 预算的主动管理手段（结构见 layout.md P-LAYOUT-ROW-RESIDENCY）。
