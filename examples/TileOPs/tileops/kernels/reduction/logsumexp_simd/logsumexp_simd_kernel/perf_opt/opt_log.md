# LogSumExpFwdOpSimd（SIMD 变体，双 kernel）Stage 4 调优日志

- operator: `LogSumExpFwdOpSimd`（`tileops/kernels/reduction/logsumexp_simd/logsumexp_simd_kernel/`）
- programming_mode: **simd**（SimdVF，显式 2048-bit `T.simd.*`），auto_schedule: yes
- benchmark 入口: `examples/TileOPs/benchmarks/ops/bench_logsumexp_simd.py::test_logsumexp_simd_bench`
- 时延口径: `tilelang.profiler.do_bench(fn, backend="msprof")`（ms×1000 记 µs）；深挖 `backend="msprof_detail"`；周期归因 npusim
- 环境: Ascend950PR_9579（28 AIC / 56 AIV 实查），CANN 9.2.0，bisheng dav-3510，tilelang dev build（fc32f47c 后，tilelang 0.1.15+cuda.git73a4b92d 同代）
- 基准 kernel（只读）: `_logsumexp_kernel_single.py` / `_logsumexp_kernel_tiled.py`

## Phase 0：上下文加载

### 输入与映射（实证）

- workload→kernel 映射经 `logs/phase0/probe_mapping.py` 实证（`probe_mapping.json`）：benchmark 参数化展开 6 案例，无 smoke/skip 标记 → 全部 tune；wrapper `default_config`（`device_smem_budget()`=216KiB 与编程模式无关）选出与 SIMT 兄弟完全相同的 (block_m, tile_n) 落点（4k: bm=16/tn=0 single；3d: bm=4/tn=0 single；32k: bm=1/tn=16384 tiled；lm-head: bm=1/tn=25600 tiled）。
- 设计参考（只读）: `_logsumexp_kernel_single_DESIGN.md` / `_logsumexp_kernel_tiled_DESIGN.md`；**tiled §8.3 预设 Stage 4 A/B 观察项**（BLOCK_N = tile_n 直保留 25600/32512 vs 甜点候选 16384，偏离档劣于 16384 档 >5% 则换档；V6 划分不变性保证数值中性）。
- pattern-library 检索: `P-LAYOUT-ROW-RESIDENCY`（其 BLOCK_N 甜点 16384 为 **SIMT 结构实测**，条目自身声明不可跨结构外推 → 本任务为 SimdVF 结构新实测）；`P-LAYOUT-TAIL-STATIC-EXTENT`（尾块静态 extent 范式，分支实现沿用）；CASE-logsumexp-harness；kb_stale_check stale_count=0。
- SIMT 兄弟 Stage 4（`logsumexp/logsumexp_kernel/perf_opt/`，同 workload 集）作为参数空间参考（非结论）：其 lm-head（M=4）产出 [DESIGN_LIMIT]（4/56 核激活，~2.5x 结构天花板）；**本 SIMD 变体 baseline 已实测 6.7µs，远快于 SIMT final 17.7µs**——SIMT 的结构天花板数值在 SIMD 结构下不直接适用，须重新判定。

### 生成源码结构观察（get_kernel_source，归档 logs/phase0/*_source.txt）

- tiled（lm-head 落点）：组循环 `for g < 400` 未展开，每迭代带 `(version_counter & 1) * 25600 + g*64` 地址算术 + 循环控制标量；每 chunk 5 次 scalar sync + `mem_bar(VST_VLD)` stats 往返；lse 写回 `asc_copy_ub2gm_align` 每 tile 2 字节（M=1024 的 32k workload → 1024 次微写，msprof_detail aiv_mte3 64.3% 印证）。
- single（4k 落点）：SUB 行子批 + 双缓冲（version counter），同样的未展开组循环；56 核 round-robin。
- API 能力核实（AGENTS.md 证伪协议）：`asc_reduce_max` 有 `vector_half` 重载、**无 bfloat16 重载**（`reg_reduce.h` L28-38）→ bf16 域 pass1 max 须 EVEN/ODD 拆分到 fp32 归约（精确）；`vmax`/`vlds NORM` 支持 half/bf16 128-lane 向量（`asc_max_impl.h`）。

## Workload Inventory（Phase 1）

| kernel_id | workload_id | (M, N) | dtype | wrapper 落点 | kind |
|---|---|---|---|---|---|
| _logsumexp_kernel_single | attn-weights-4k-float16 | (1024, 4096) | fp16 | block_m=16, tile_n=0 | tune |
| _logsumexp_kernel_single | attn-weights-4k-bfloat16 | (1024, 4096) | bf16 | block_m=16, tile_n=0 | tune |
| _logsumexp_kernel_single | 3d-multidim-reduce-float16 | (128, 16384) | fp16 | block_m=4, tile_n=0 | tune |
| _logsumexp_kernel_tiled | attn-weights-32k-bfloat16 | (1024, 32768) | bf16 | block_m=1, tile_n=16384 | tune |
| _logsumexp_kernel_tiled | lm-head-logits-float16 | (4, 102400) | fp16 | block_m=1, tile_n=25600 | tune |
| _logsumexp_kernel_tiled | lm-head-logits-bfloat16 | (4, 102400) | bf16 | block_m=1, tile_n=25600 | tune |

（完整 JSON 见 `workload_inventory.json`；3d-multidim 的 op 层 permute+contiguous 在 kernel 边界之外，kernel 级测量与 SIMT 兄弟同口径。）

## Baseline Performance Test Data（Phase 1，round=0，candidate=baseline）

每项 = 2 独立批次 × 5 do_bench(msprof) 采样，pooled 中位数；批次间方差 ≤0.42%（≤1% 对账阈值）；msprof_detail 归档 `profiles/baseline/`。

| kernel_id | workload_id | candidate_id | duration_us | samples | variance_pct | bottleneck（pipe ratios） | profiler_output |
|---|---|---|---:|---:|---:|---|---|
| _logsumexp_kernel_single | attn-weights-4k-float16 | baseline | 10.862 | 10 | 0.03% | aiv_mte2 57.2%（vec 44.0/scalar 8.1/mte3 14.0） | profiles/baseline/_logsumexp_kernel_single/attn-weights-4k-float16_baseline |
| _logsumexp_kernel_single | attn-weights-4k-bfloat16 | baseline | 10.879 | 10 | 0.03% | aiv_mte2 57.7%（vec 44.3/scalar 8.7/mte3 11.1） | profiles/baseline/_logsumexp_kernel_single/attn-weights-4k-bfloat16_baseline |
| _logsumexp_kernel_single | 3d-multidim-reduce-float16 | baseline | 11.308 | 10 | 0.14% | aiv_mte3 37.2%（mte2 30.9/vec 23.9/scalar 8.0） | profiles/baseline/_logsumexp_kernel_single/3d-multidim-reduce-float16_baseline |
| _logsumexp_kernel_tiled | attn-weights-32k-bfloat16 | baseline | 72.439 | 10 | 0.01% | aiv_mte3 64.3%（vec 52.9/mte2 50.2/scalar 2.4） | profiles/baseline/_logsumexp_kernel_tiled/attn-weights-32k-bfloat16_baseline |
| _logsumexp_kernel_tiled | lm-head-logits-float16 | baseline | 6.728 | 10 | 0.09% | aiv_scalar 57.8%（vec 31.6/mte2 22.7） | profiles/baseline/_logsumexp_kernel_tiled/lm-head-logits-float16_baseline |
| _logsumexp_kernel_tiled | lm-head-logits-bfloat16 | baseline | 6.803 | 10 | 0.42% | aiv_scalar 62.0%（vec 29.3/mte2 20.3） | profiles/baseline/_logsumexp_kernel_tiled/lm-head-logits-bfloat16_baseline |

（对照 SIMT 兄弟 final：4k 21.5µs / 3d 13.8µs / 32k 100.2µs / lm-head 17.7µs——SIMD baseline 已全面大幅领先，分别为 2.0x/1.2x/1.38x/2.6x。）

### 设计估算 vs 实测偏差（D-2 回填）

tiled DESIGN §1.6.0 估算（wl2 代表 workload）为 cycle 域相对口径（无绝对时延假设），无「设计期估算时延行」可对账；不适用偏差行。以实测有效带宽替代口径：32k = 1024×32768×2B / 72.44µs = **925GB/s**（读侧），4k = 8MB/10.86µs = **765GB/s**，lm-head = 0.8MB/6.73µs = **119GB/s**（4 核激活）。

### Phase 1 排序（优先级 + 理由）

1. **attn-weights-32k**（72.4µs，总时延最大）：mte3 64.3% = 1024 次 2 字节 lse 微写；结构性候选（ROWS 批量化）+ BLOCK_N 32768（2 chunk→1 chunk/行）。
2. **lm-head-logits**（6.73/6.80µs，双 workload）：scalar 57.8-62.0% 触发 BP_scalar_ratio 强制换轴/向量化分析——per-iteration 循环控制 + 地址算术；候选 = 组循环 unroll、pass1 16-bit 域 max（vec 指令 ÷3）、BLOCK_N 51200（4 chunk→2）；§8.3 A/B（25600 vs 16384）必测。
3. **attn-weights-4k**（10.86/10.88µs，双 workload）：mte2 57.2% + SUB=8 波次不均（128 tiles/56 核 = 3 波、40 核空转第 3 波）；候选 = SUB sweep。
4. **3d-multidim-reduce**（11.31µs）：SUB=2 → 64 tiles/56 核（8 核 2 批、48 核 1 批，makespan 2×2=4 行 vs 理想 2.3 行）；候选 = SUB=1。

## Iteration 1 Diagnosis（Round 1：双 kernel 结构性候选 + §8.3 A/B 强制裁决）

### Diagnostic Context

- base: baseline（两 kernel），测量见上表；生成源码归档 `logs/phase0/*_source.txt`。
- 关键源码观察（get_kernel_source）：组循环未展开（每迭代带 `(counter&1)*BLOCK_N + g*64` 地址算术 + 循环控制标量）；tiled 每 tile 经 MTE3 写 2 字节 lse（block_m=1 × M 行）；每 chunk 5 次 scalar sync + `mem_bar(VST_VLD)`。
- 首轮必查项：编程模式重估——本算子即 SimdVF 显式向量结构（vs SIMT 兄弟 +2.0~2.6x 实测），无需再换轴；标量占比诊断——lm-head aiv_scalar 57.8/62.0% >50% 触发强制向量化候选分析（→ OP-unroll/OP-pair/OP-f16max）；pattern-library 版本戳核对 stale_count=0。

### Current Phenomena

- P1（32k，72.4µs）：aiv_mte3 64.3% —— 1024 次 2 字节 lse 微写 + 1024 次持久循环迭代；vec 52.9%/mte2 50.2% 未饱和。
- P2（lm-head，6.73µs）：aiv_scalar 57.8-62.0% —— 每行 4 chunk × 800 组迭代的循环控制/地址算术 + 每 chunk stats 往返；M=4 → 4 tile（4/56 核实做，AIV 速率 14.5GHz vs 32k 满载 74.6GHz）。
- P3（4k，10.86µs）：aiv_mte2 57.2% + SUB=8 → 128 批/56 核 = 3 波（16 核 3 批/40 核 2 批，makespan 24 行 vs 理想 19）。
- P4（3d，11.31µs）：SUB=2 → 64 批/56 核（8 核 2 批，makespan 4 行 vs 理想 2.3）。

### Candidate Optimization Points（Round 1）

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP-ROWS | P1（mte3 微写） | 持久 tile 行数 ROWS（lse 批量写回） | mte3 64.3% 由 M 次 2B 微写贡献 | ROWS∈{2,4,8}（env 旋钮），lse_ub (ROWS,) 单次写回 | do_bench + mte3 ratio | candidate |
| OP-BLOCKN | P2/P1 + §8.3 强制 A/B | BLOCK_N 重推导（chunk 宽 sweep） | §8.3 预设：直保留 25600/32512 vs 甜点 16384 | lm-head {16384,20480,25600,51200}；32k {16384,32768} | do_bench（>5% 判换档） | candidate |
| OP-F16MAX | P2/P3 + 全 workload vec 占比 | pass1 在 16-bit 输入域做 max | max 无舍入 → 与 fp32 detour 位级等价；3 ops/64 → 2 ops/128 | vld NORM 128-lane + vmax(h/bf16) + winner 单次 vcvt；bf16 无 asc_reduce_max 重载 → EVEN/ODD 拆分 | do_bench + max_abs=0 | candidate |
| OP-UNROLL4 | P2（scalar 58%） | 组循环 4 路步进展开 | 每迭代 ~3-5 标量（循环控制+地址） | `for g in range(0, G_u, 4)` 内联 4 组 | do_bench + scalar ratio | candidate |
| OP-SUB | P3/P4（波次不均） | SUB 行子批 sweep | makespan 模型 24→20 行（4k） | SUB∈{2,4}(4k)、{1}(3d) | do_bench | candidate |

### 实验分支与结果（run_experiments.py，A/B 交错 3 轮 × 3 采样）

| branch | base | opt_id | 主要改动 | L0 | do_bench 结果（vs 同轮 baseline A 侧） | verdict |
|---|---|---|---|---|---|---|
| _tiled_round1_knobs.py (ROWS=2/4/8) | baseline | OP-ROWS | 32k ROWS sweep | pass | 60.52/53.76/**53.23**µs（+16.4/+25.8/+26.6%；mte3 64.3→6.2%，mte2→83.7%） | improved（ROWS=8/4） |
| _tiled_round1_knobs.py (BLOCKN=32768) | baseline | OP-BLOCKN | 32k 单 chunk | pass | 72.28µs（+0.1% tie） | config_no_gain |
| _tiled_round1_knobs.py (BLOCKN=16384/20480/51200) | baseline | OP-BLOCKN | lm-head §8.3 A/B | pass | 16384: 10.17µs（**−52.0%**）；20480: 6.74µs（+1.0% tie）；51200: 7.18µs（−6.9%） | **直保留 25600 胜出**（§8.3 裁决：不换档） |
| _tiled_round1_f16max.py | baseline | OP-F16MAX | tiled 双 workload | pass | 32k +4.0%；lm-head fp16 **+15.5%**/bf16 **+14.8%** | improved |
| _tiled_round1_unroll4.py | baseline | OP-UNROLL4 | 组循环步进 4 展开 | pass | 32k −4.5%；lm-head −13.6/−14.1% | **refuted** |
| _single_round1_knobs.py (SUB=2/4) | baseline | OP-SUB | 4k SUB 下探 | pass（SUB=4 因 fp32 L0 形状校验中断，未测） | SUB=2: 18.57µs（**−69.8%**） | **refuted**（下探方向） |
| _single_round1_knobs.py (SUB=1) | baseline | OP-SUB | 3d SUB 下探 | pass | 17.88µs（**−60.0%**） | **refuted** |
| _single_round1_f16max.py | baseline | OP-F16MAX | single 3 workload | pass | 4k fp16 +5.6%/bf16 +6.3%；3d +4.6%（2/3 配对——落 <5% 平区） | improved（3d 走 ab_test） |

**3d f16max ab_test 交错裁决**（`.agents/tools/ab_test.py`，5 对，logs/round1/ab_3d_f16max_run3.log）：A(baseline) 合并中位 11.273µs vs B(f16max) 10.702µs，**−5.06%，5/5 方向一致 → B_better**（前两次运行受 run-state 双态污染——BP_run_state_bimodality 实证，第 3 次 5 对裁决采信）。

### Round 1 收束

- single winner：**f16max**（3/3 workload 提升，max_abs=0 位级）；SUB 下探 family 级否决（per-tile 固定开销主导，makespan-only 模型再次失准——与 SIMT ROWS 实测同教训）。
- tiled winner：**ROWS（32k）+ f16max（lm-head）**；§8.3 A/B 实测裁决：**BLOCK_N = tile_n 直保留（25600）显著最优**（甜点候选 16384 在 SimdVF 结构 + 该 workload 上是反甜点：chunk 数 4→7，per-chunk 开销主导）；unroll4 refuted。

## Iteration 2 Diagnosis（Round 2：winner 组合 + scalar 归因假设检验）

### Diagnostic Context（每轮必读 iteration-diagnosis.md / bottleneck-patterns.md；先看表）

**候选 vs current best 对比表**（do_bench(msprof) µs / msprof_detail pipe ratios / L0）：

| 候选 | 32k-bf16 | lm-head-fp16 | lm-head-bf16 | 4k-fp16 | 4k-bf16 | 3d-fp16 | L0 |
|---|---|---|---|---|---|---|---|
| baseline | 72.40（mte3 64.3/vec 52.9/mte2 50.2/sca 2.4） | 6.73（sca 57.8/vec 31.6） | 6.80（sca 62.0） | 10.86（mte2 57.2） | 10.88（mte2 57.7） | 11.31（mte3 37.2） | pass |
| r1_rows8 | **53.23**（mte3 6.2/mte2 83.7/vec 65.7） | — | — | — | — | — | pass |
| r1_f16max（tiled） | 69.47（mte3 68.9） | **5.67**（sca 62.7） | **5.73**（sca 63.4） | — | — | — | pass |
| r1_f16max（single） | — | — | — | 10.49 | 10.39 | 10.67（ab −5.06%） | pass |
| r1_unroll4 | 75.79（refuted） | 7.62（refuted） | 7.65（refuted） | — | — | — | pass |

### Candidate Optimization Points（Round 2，base = Round 1 winner）

- OP-COMBO（32k）：ROWS 策略化（`_pick_rows`：M≤56→1；否则 makespan ≤ min+1 的最大 pow2、cap 8）+ f16max 合并（两单点已证）。
- OP-PAIR（P2 余量）：pass2 EVEN/ODD 配对（1 次 128-lane NORM 载入 + 2 次 vcvt —— 迭代数与 vld 减半；fp32 求和顺序重结合，ulp 级）。
- OP-MASKHOIST（P2 scalar 假设）：显式提升 b16 谓词掩码（生成源码中 `asc_create_mask_b16(PAT_ALL)` 每迭代重建）。

### 实验分支与结果

| branch | base | opt_id | 结果 | verdict |
|---|---|---|---|---|
| _tiled_round2_combo.py（ROWS 策略=4） | r1_rows8 | OP-COMBO（32k） | **50.58µs（+4.7% vs rows8；累计 +30.1% vs baseline）**，mte2 85.7% | improved |
| _tiled_round2_combo.py（ROWS=8 env） | r1_rows8 | OP-COMBO 邻域 | 50.59µs（与 ROWS=4 tie 0.03%） | 策略取 4（一致） |
| _tiled_round2_combo.py | r1_f16max | OP-COMBO（lm-head ROWS=1） | 5.710/5.784µs（±0.0%——lm-head 路径与 f16max 结构等价，无回退） | pass |
| _tiled_round2_pair.py | r1_f16max | OP-PAIR | 32k −4.7%；lm-head **−17.1/−16.7%** | **refuted**（单载入双转换的深依赖链劣化 ILP） |
| _tiled_round2_maskhoist.py | r1_f16max | OP-MASKHOIST | 32k +0.0%；lm-head +0.3/−0.0% | config_no_gain（掩码非 scalar 主项） |
| _single_round2_pair.py / _single_round2_maskhoist.py | r1_f16max_s | （single 侧同型） | 未单独测（tiled 侧同型已 family 级否决/无收益） | family_no_gain（PAIR）/no_gain（MASKHOIST） |

### Round 2 收束

- tiled current best = **combo**（f16max + ROWS 策略）：32k 50.58µs（+30.1%），lm-head 5.71µs（+15%）。
- PAIR family 否决、MASKHOIST 无收益：lm-head 的 scalar 62% 不是循环控制/掩码类（unroll4/maskhoist 双证）——归因收窄到载入地址算术 + 每 chunk sync/stats 往返 + vexpdif special 档（8 ops/cycle）混叠；post_inc 指针（deepgemm transform_sf 先例）因 x_ub 双缓冲版本计数交互的正确性风险 defer。

## Iteration 3 Diagnosis（Round 3：single 波次再平衡 + 32k 邻域收尾）

### 候选 vs current best 对比表

| 候选 | 32k-bf16 | 4k-fp16 | 4k-bf16 | 3d-fp16 | L0 |
|---|---|---|---|---|---|
| current best（combo/single f16max） | 50.58 | 10.36 | 10.33 | 10.68 | pass |
| r3_slab | — | 12.76（−23.1%） | 12.74（−23.5%） | 11.42（−5.8%） | pass |
| r3_stages3 | 51.46（−0.0%） | — | — | — | pass |
| r3_bn32768 | 51.75（−0.1%） | — | — | — | pass |

- OP-SLAB（P3/P4 波次不均）：均衡波调度——满波批量取 56 整数倍 + 单次小额补尾波（4k makespan 24→19 行；3d 4→3 行；三分支 TIR if + 静态 extent）。**实测 refuted**：4k −23%、3d −5.8%——机理：异构分支破坏 copy/compute 双缓冲重叠（serialized 估算 12.4µs ≈ 实测 12.7µs），重叠损失 > 波次收益。**波次再平衡 family 否决**（SUB 下探 + slab 两证：本结构内 uniform batch + 双缓冲重叠 > 理论均衡）。
- OP-STAGES3 / OP-BN32768（32k 邻域）：均 tie（−0.0/−0.1%）——32k 已达 MTE2 墙（64MB/50.6µs = **1.27TB/s** 有效读带宽，mte2 85.7% busy）。

### lm-head 4/56 核激活与 [DESIGN_LIMIT] 判定（Phase 3 item 4）

- 实测证据：lm-head f16max AIV 速率 14.5-16.2GHz vs 32k 满载 74.6GHz（≈56 核）——M=4 → 4 行块 tile，与 SIMT 兄弟同构（其 [DESIGN_LIMIT]：4/56 核、~2.5x 天花板）。
- **结构性加速估算（关键差异）**：跨核 N-split（4 行 × 7 split = 28 核）后计算段 ≈ 1.5-2µs（exp special 档 4×102400 elems/8/28 核 ≈ 1830 cyc/核）+ (max,sum) 定序合并段（partial GM 往返 + 跨核 flag 定序，SIMT 兄弟同协议估算 2-3µs）≈ **合计 4-5µs vs 当前 5.71µs → ~1.2-1.4x**。SIMT 的 ~2.5x 估算建立在 17.7µs 串行链上（合并开销占比小）；SIMD 变体每 chunk 效率已 2.6x 于 SIMT（5.71 vs 17.7µs），合并段开销吞掉并行收益。
- 判定：① 设计层归因成立（DESIGN tiled R1 #12/#3 弃选 N-split + §0.2 冻结签名 (x,y) 无 workspace——设计已预判并列为 §9.1 风险 3 Stage 4 观察项）；② **结构性加速估计 ~1.2-1.4x < 2x 门槛不满足**；实测亦未推翻设计假设（R1 #12 的弃选理由与本实测一致）。**→ [DESIGN_LIMIT] 不触发，不产出 perf_feedback.md**（4 核激活为事实，但参数/结构空间内已收束且逆向反馈不达门槛）。
- 调优内已穷尽：BLOCK_N 双向 sweep（16384 −52%/20480 tie/51200 −6.9%/32768 tie）、unroll4/pair/maskhoist（scalar 假设三连否决）、ROWS（M=4 域无并行度可造）、stages（tie）。post_inc 因版本计数正确性风险 defer（flash_attention core.py L98 先例为非版本化 buffer 的 store 指针）。

## 合并与最终收束（Phase 2 item 10 → Phase 3）

### 合并（merged current best）

- **single** = f16max → `perf_opt/_logsumexp_kernel_single.py`（`--level all` 84 用例 PASS，max_diff 全 0）。
- **tiled** = f16max + `_pick_rows` 策略 → `perf_opt/_logsumexp_kernel_tiled.py`（`--level all` 69 用例 PASS，max_diff 全 0；L2 WARN 均为 kernel_cache/torch.tensor 既有噪声）。

### 当前 Kernel 全部 Tune Workload 合并检查（merged vs baseline，A/B 交错 3×3）

| merged_candidate | kernel_id | workload_id | baseline_us | merged_us | precision | status |
|---|---|---|---:|---:|---|---|
| _logsumexp_kernel_single.py | single | attn-weights-4k-float16 | 10.976 | 10.376 | max_abs=0 | pass |
| _logsumexp_kernel_single.py | single | attn-weights-4k-bfloat16 | 11.062 | 10.329 | max_abs=0 | pass |
| _logsumexp_kernel_single.py | single | 3d-multidim-reduce-float16 | 11.205 | 10.591 | max_abs=0 | pass |
| _logsumexp_kernel_tiled.py | tiled | attn-weights-32k-bfloat16 | 72.229 | 51.657 | max_abs=0 | pass |
| _logsumexp_kernel_tiled.py | tiled | lm-head-logits-float16 | 6.629 | 5.598 | max_abs=0 | pass |
| _logsumexp_kernel_tiled.py | tiled | lm-head-logits-bfloat16 | 6.708 | 5.710 | max_abs=0 | pass |

（smoke：本 benchmark 无 smoke 案例——6 案例全部 tune；合并后全量 `--level all` 覆盖精度。）

### Phase 3：final 记录 + 复测方差对账 + 端到端验收

- final 记录（candidate_id=final，同一最终候选，`perf_records.jsonl` phase=final 6 行）：原 final（per-function 文件）vs merged 逐 workload 方差 0.05–0.61%（≤1% 对账阈值 ✓）；2026-10-19 gate 补齐时 final 记录改测 consolidated 工件 `perf_opt/LogSumExpFwdOpSimd.py`（kernel 代码字节一致，B_max_abs=0 全 6 workload），重测方差与 superseded 数值审计见「Gate 4 Remediation」节。
- **TileOPs 端到端验收**（临时翻转 wrapper 双 import 块 → `python -m tileops.reporting.cli run --op LogSumExpFwdOpSimd --prof-mode msprof` → 恢复原状，sha256 校验一致）：**status=passed，正确性 45/45（avg max abs err 9.17e-08）**，6 shape 全测（report 归档 `examples/TileOPs/reports/tileops/20261017_204656_958420_LogSumExpFwdOpSimd/`）。端到端口径：attn-4k 10.28µs（0.816TB/s）、attn-32k **51.60µs（1.30TB/s，roofline 75.2%）**、lm-head 5.68/5.71µs、3d-multidim 16.13µs（含 op 层 permute+contiguous 前处理 ~5.5µs，kernel 级 10.66µs——非 kernel 调优域）。首次运行遇设备子进程启动超时（E39007 瞬态环境故障），npu-smi 确认健康后重跑通过。
- wrapper 采纳状态：按约束保持 baseline 激活，翻转由 conductor 在回归通过后执行。

### Final Performance Test Data

（final 候选统一为 consolidated 最终工件 `perf_opt/LogSumExpFwdOpSimd.py`——Stage 4 gate 契约的单文件汇总视图，kernel 代码与 per-function 文件逐字一致；时延全部来自同一最终候选的 phase=final 记录，2026-10-19 在 consolidated 工件上按 A/B 交错 3×3 协议实测。）

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| _logsumexp_kernel_single | attn-weights-4k-float16 | 10.862 | 10.217 | final |
| _logsumexp_kernel_single | attn-weights-4k-bfloat16 | 10.879 | 10.119 | final |
| _logsumexp_kernel_single | 3d-multidim-reduce-float16 | 11.308 | 10.703 | final |
| _logsumexp_kernel_tiled | attn-weights-32k-bfloat16 | 72.439 | 51.636 | final |
| _logsumexp_kernel_tiled | lm-head-logits-float16 | 6.728 | 5.632 | final |
| _logsumexp_kernel_tiled | lm-head-logits-bfloat16 | 6.803 | 5.669 | final |

（baseline 列为 Phase 1 round=0 记录；improvement = (baseline−final)/baseline：4k fp16 +5.9% / 4k bf16 +7.0% / 3d +5.4% / 32k bf16 **+28.7%** / lm-head fp16 **+16.3%** / lm-head bf16 **+16.7%**；各 final 记录 profiler_output 指向 profiles/round3/{kernel}/{workload}_final msprof_detail 归档。）

### Final 深挖画像（msprof_detail，profiles/round3/*_merged/）

| workload | dur_us | 主瓶颈 pipe | vec | scalar | mte2 | mte3 |
|---|---:|---|---:|---:|---:|---:|
| attn-weights-4k-fp16 | 10.68 | aiv_mte2 60.2% | ~0.40 | ~0.08 | 0.602 | ~0.10 |
| attn-weights-32k-bf16 | 52.32 | aiv_mte2 75.2% | 0.531 | 0.030 | 0.752* | 0.079 |
| lm-head-fp16 | 5.94 | aiv_scalar 65.2% | ~0.26 | 0.652 | 0.229 | 0.020 |
（*merged 档 mte2 75.2%/rows8 档 83.7%——采样批内波动，两者均印证 MTE2 主导。）

### 中止条件与结论

- **stop_reason = plateau**（Round 3 无 valid 增益；Round 2 增益 +4.7% 后 Round 3 slab/stages/bn 全部 refuted/tie；主要结构候选已充分验证——§8.3 A/B 双向 sweep、scalar 三假设（unroll/maskhoist/pair）、波次再平衡双路（SUB/slab）、32k 邻域（stages/bn））。参照结构 diff 检查（morph ladder 轻量版）：同族最优参照 = SIMT 兄弟（本 SIMD baseline 已全面 1.2-2.6x 领先其 final）+ flash_attention（stats-UB-roundtrip 同构，已采用）；无未试结构差异。
- 全部 6 tune workload 有结论（winner_merged）、最终全量记录完成、精度 45/45 + 84+69 分层用例通过。

## Skill Retrospective（Phase 4）

### Skill 流程自检（对照 skill-retrospective.md）

- Phase 0 首轮必查项完整执行（编程模式重估/标量占比诊断/陷阱版本戳）；§8.3 实验裁决项作为必做项完成并回写（下节）。
- run-state 双态（BP_run_state_bimodality）在本任务两度实证：ab_test 前两次运行结论翻转（inconclusive→B_better 19%→第 3 次稳定 −5.06%），交错多 run 协议是必要的；另 msprof 首调偶发「No profiled kernels remain」竞态（bench_common 加重试根治，建议 skill 模板吸收）。
- 证伪协议执行：BLOCK_N 否决基于合法 64 倍数全档 sweep（非单点）；post_inc defer 而非否决（版本计数交互未实证，标为未实证假设）。

### 价值点（vp_type + 证据三件套；已回写 pattern-library 的注明回写位置防蒸馏双写）

| id | vp_type | 内容 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-S4-SIMD-BLOCKN-AB | D | SimdVF tiled 结构 BLOCK_N 甜点实测：lm-head (4,102400) 直保留 25600 最优，16384 档 −52%（chunk 数 4→7，per-chunk 开销主导）；SIMT 甜点不可跨结构外推的实证补全（§8.3 裁决） | 本任务 opt_log Round 1 + perf_records.jsonl r1_blockn16384/20480/51200 | 回写 P-LAYOUT-ROW-RESIDENCY（layout.md 注记） | tilelang dev fc32f47c 后 + CANN 9.2.0 + Ascend950PR_9579, 2026-10-17 |
| VP-S4-SIMD-F16MAX | P | 16-bit 输入域 pass-1 max（vld NORM 128-lane + vmax + winner 单次 vcvt；bf16 需 EVEN/ODD 拆分——asc_reduce_max 无 bf16 重载）：位级等价（max 无舍入），5 workload +4~+15.5% | 本任务 Round 1/2 + max_abs=0 | 回写 pattern-library elementwise.md 新条目 D-LOGSUMEXP-SIMD-F16MAX + repro | 同上 |
| VP-S4-SIMD-ROWS-LSE | P | 持久 tile ROWS 批量 lse 写回（M≤56→1 保并行；否则 makespan ≤min+1 最大 pow2 cap 8）：32k +26.6%（mte3 64.3→6.2%） | 本任务 Round 1/2 | 回写 P-LAYOUT-ROW-RESIDENCY 注记 + repro | 同上 |
| VP-S4-SIMD-OVERLAP-BEATS-BALANCE | D | 均衡波 slab（理论 makespan −21%）实测 −23%：异构分支破坏 copy/compute 双缓冲重叠；uniform batch + 双缓冲重叠 > 理论均衡（makespan-only 模型缺重叠项——P-LAYOUT-ROW-RESIDENCY per-tile 开销教训的姊妹篇） | 本任务 Round 3 r3_slab | 回写 P-LAYOUT-ROW-RESIDENCY 注记 | 同上 |
| VP-S4-SIMD-PAIR-UNROLL-REFUTED | D | SimdVF 反模式实证：组循环步进展开（−14%）、EVEN/ODD 单载入双转换配对（−17%）均劣化——深依赖链破坏软件流水；掩码提升无效果（非 scalar 主项） | 本任务 Round 1/2 | 回写 pattern-library（traps-compiler.md 或 elementwise.md 反模式条目） | 同上 |
| VP-S4-SIMD-LMHEAD-NO-DESIGN-LIMIT | D | SIMD 变体 lm-head 4/56 核激活但 N-split 结构收益 ~1.2-1.4x（合并段开销吞掉并行收益）<2x 门槛——与 SIMT 兄弟 [DESIGN_LIMIT]（~2.5x）同现象不同结论：结构天花板结论不可跨结构外推 | 本任务 Phase 3 判定 + AIV 速率 14.5 vs 74.6GHz | opt_log 本节 | 同上 |

### Skill 流程问题与建议

- **msprof 首调竞态**（"No profiled kernels remain"偶发 ~30-40%）：do_bench(msprof) 在进程内首次调用时 profiler 会话偶发丢失。建议 profile-collection.md / bench 模板加「重试 3-4 次」标准包装（本任务 bench_common._with_msprof_retry）。
- **run_experiments 模板对「L0 全形状 × env 旋钮」的校验冲突**：env 旋钮全局作用于 L0 全形状集时，个别形状（如 fp32 大 N）触发容量校验使整个 L0 失败。建议模板支持非致命回退 + 告警（本任务 BLOCKN/SUB 旋钮均遇）。
- **ab_test 工具需 PYTHONPATH 显式传递**（`No module named tilelang.ascend`）——建议工具内部 sys.path 注入 repo root。

## Gate 4 Remediation（2026-10-19：consolidated 工件补齐 + final 记录重测）

首轮 `statectl gate 4` 机械门禁 14 条失败（3 类：S4-EXISTS 缺与 operator_name 同名的 consolidated 工件；S4-PERF-RECORDS-RECON final 记录未指向 consolidated 文件；S4-OPTLOG-COMPTABLE 缺 Task Duration 对比表 + Final 表缺 gate 5 列格式）。本节为 artifact 完整性补齐记录，**非调优重做**（per-function 最优内核代码体字节冻结未动，调优结论不变）。

### 修复 1：consolidated 工件 `perf_opt/LogSumExpFwdOpSimd.py`

- 按 SIMT 兄弟 `LogSumExpFwdOp.py` 已验证模板生成（`logs/gate4/assemble_consolidated.py` 装配：逐行区间切拷 + 首行断言）：单文件含两个 tuned kernel + tuned 常量 + 双 variant 分层测试（combined `run_L0/run_L1/run_L2/run_boundary` 入口）；`--level all` **153 用例全过**（84 single + 69 tiled，max_diff 全 0，logs/gate4/consolidated_level_all.log）。
- 逐字一致校验（`logs/gate4/verify_verbatim.py`，AST 行区间比对）：`_logsumexp_kernel_single`（179 行）/`_logsumexp_kernel_tiled`（215 行）及 `_load_group`/`_pick_rows`/`_env_int`/`_wrapper_tile_n`/golden/双 source-sim/tolerance/elem/dtype 共 **16 个函数 byte-identical**；测试脚手架 rename-only（`_run_case`→`_run_single_case`/`_run_tiled_case`、tiled `_call`→`_call_tiled`，与兄弟模板同款）；7 个 tuned 常量（NUM_BLOCKS/MAX_SINGLE_TILE_COLS/UB_BUDGET_BYTES/UB_RESERVE_BYTES/WRAPPER_BUDGET_BYTES/GROUP/HMAX_ENABLE）值一致。

### 修复 2：final 记录改测 consolidated 工件（perf_records.jsonl）

- 原 6 条 phase=final 记录指向 per-function 文件——过早 final 标记（gate 契约要求 final 记录 artifact_path = consolidated `perf_opt/{op}.py` 且每 (kernel, workload) 恰一条；纯追加会形成双 final 记录，无法过门禁）。处置：**移除该 6 条过早记录并追加 6 条 consolidated 工件上的新实测记录**（`logs/gate4/rewrite_final_records.py`；移除前全量备份 `logs/gate4/perf_records.before_remediation.jsonl`，superseded 数值完整留痕于下表）；baseline/candidate/merged 历史 49 行逐字节未动。
- 新记录全部字段实测：A/B 交错 3×3（run_experiments.py 同款协议；L0 B 双 variant pass；B_max_abs=0 全 6 workload），duration_us = 9 样本中位数，artifact_sha256 = consolidated 文件实际 SHA256，profiler_output = profiles/round3/{kernel}/{workload}_final msprof_detail 归档（32k aiv_mte2 76.5% / lm-head aiv_scalar 62.6~65.2%，与 Round 3 画像一致）。

### superseded final 数值审计留痕（2026-10-17 per-function 文件实测 vs 2026-10-19 consolidated 重测，kernel 代码字节一致）

| kernel_id | workload_id | superseded_final_us | consolidated_final_us | Δ | 同会话 baseline 锚点 Δ |
|---|---|---:|---:|---:|---:|
| _logsumexp_kernel_single | attn-weights-4k-float16 | 10.381 | 10.217 | −1.58% | −0.47%（10.913 vs 10.862） |
| _logsumexp_kernel_single | attn-weights-4k-bfloat16 | 10.340 | 10.119 | −2.14% | −1.13%（10.756 vs 10.879） |
| _logsumexp_kernel_single | 3d-multidim-reduce-float16 | 10.656 | 10.703 | +0.44% | −1.65%（11.121 vs 11.308） |
| _logsumexp_kernel_tiled | attn-weights-32k-bfloat16 | 51.617 | 51.636 | +0.04% | −0.37%（72.173 vs 72.439） |
| _logsumexp_kernel_tiled | lm-head-logits-float16 | 5.594 | 5.632 | +0.68% | −1.07%（6.656 vs 6.728） |
| _logsumexp_kernel_tiled | lm-head-logits-bfloat16 | 5.707 | 5.669 | −0.67% | −2.00%（6.667 vs 6.803） |

（4k 两 workload 跨会话差 −1.6~−2.1% 略超 1% 复测方差阈值，但同会话 A 侧 baseline 锚点同向漂移 −0.5~−2.0%（设备/会话级漂移，非代码差异——kernel 字节一致且 B_max_abs=0）；A/B 配对相对提升稳定：本会话 +6.4/+5.9/+3.8/+28.5/+15.4/+15.0% vs 首测 +5.4/+7.1/+5.4/+28.8/+15.3/+14.7%，逐 workload 方向一致，paired_B_faster 6/6 workload = 3/3。）

### 候选 vs current best 的 Task Duration 对比表（汇总，do_bench(msprof) µs）

Task Duration = `do_bench(fn, backend="msprof")` 端到端 kernel 时延（µs，交错采样中位数）。下表汇总各轮 winner 候选 vs 该轮开始时 current best 的 Task Duration（数据源：perf_records.jsonl）：

| round | kernel_id | workload_id | current best Task Duration (µs) | 候选 Task Duration (µs) | Δ vs current best | 结论 |
|---|---|---|---:|---:|---:|---|
| 1 | tiled | attn-weights-32k-bfloat16 | 72.439（baseline） | 53.227（r1_rows8） | −26.5% | improved（winner） |
| 1 | tiled | lm-head-logits-float16 | 6.728（baseline） | 5.673（r1_f16max） | −15.7% | improved（winner） |
| 1 | tiled | lm-head-logits-bfloat16 | 6.803（baseline） | 5.734（r1_f16max） | −15.7% | improved（winner） |
| 1 | single | attn-weights-4k-float16 | 10.862（baseline） | 10.490（r1_f16max_s） | −3.4% | improved（winner） |
| 1 | single | attn-weights-4k-bfloat16 | 10.879（baseline） | 10.386（r1_f16max_s） | −4.5% | improved（winner） |
| 1 | single | 3d-multidim-reduce-float16 | 11.308（baseline） | 10.668（r1_f16max_s） | −5.7% | improved（winner） |
| 2 | tiled | attn-weights-32k-bfloat16 | 53.227（r1_rows8） | 50.581（r2_combo_r4） | −5.0% | improved（winner） |
| 2 | tiled | lm-head-logits-float16 | 5.673（r1_f16max） | 5.710（r2_combo） | +0.7% | tie（32k 主导采纳） |
| 2 | tiled | lm-head-logits-bfloat16 | 5.734（r1_f16max） | 5.784（r2_combo） | +0.9% | tie |
| 3 | single | attn-weights-4k-float16 | 10.490（r1_f16max_s） | 12.755（r3_slab） | +21.6% | refuted |
| 3 | single | 3d-multidim-reduce-float16 | 10.668（r1_f16max_s） | 11.419（r3_slab） | +7.1% | refuted |
| 3 | tiled | attn-weights-32k-bfloat16 | 50.581（r2_combo_r4） | 51.463（r3_stages3） | +1.7% | refuted（tie） |
| merged | 6 workload 全部 | vs baseline | 10.862/10.879/11.308/72.439/6.728/6.803 | 10.376/10.329/10.591/51.657/5.598/5.710 | −4.5~−28.6% | winner_merged |
| final | 6 workload 全部 | vs baseline（consolidated 重测） | 同上 | 10.217/10.119/10.703/51.636/5.632/5.669 | −5.4~−28.7% | final（见 Final Performance Test Data） |

（逐分支明细见各轮「候选 vs current best 对比表」与「实验分支与结果」节；Final Performance Test Data 表已改为 gate 5 列格式，数值与 perf_records.jsonl baseline/final 记录逐项一致。）

### 补齐后自检（logs/gate4/gate_selfcheck.py，只读直调 gate_lint.perf_lint）

- consolidated 文件 SHA256 `5aa7f39e…fb1168` = perf_records.jsonl 全部 6 条 final 记录 artifact_sha256 ✓；final 记录单一候选（均 candidate_id=final）、6 个 (kernel, workload) 键恰各一条 ✓。
- `python LogSumExpFwdOpSimd.py --level all` 153 用例全过（max_diff 全 0）✓；两 per-function 文件 SHA 与 merged 记录哈希一致（字节冻结未动）✓；16 个 kernel/辅助函数 byte-identical + 脚手架 rename-only ✓。
- **gate_lint.perf_lint 直调（statectl gate 4 同规则）：0 failures / 0 warnings** —— 14 条门禁失败全部解除（S4-EXISTS / S4-PERF-RECORDS-RECON×12 / S4-OPTLOG-COMPTABLE）。
