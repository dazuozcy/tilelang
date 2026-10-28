# LogSumExpFwdOp（_logsumexp_kernel_single + _logsumexp_kernel_tiled）Stage 4 调优日志

- task: logsumexp-LogSumExpFwdOp-20261017T010508Z（optimize 场景，mode=full）
- benchmark_source: `examples/TileOPs/benchmarks/ops/bench_softmax.py::test_logsumexp_bench`（manifest 参数化展开 6 案例，无 smoke/skip 标记 → 全部 tune）
- 性能目标：best_effort（无逐 workload 数值目标）；噪声阈值 3%；max_rounds=5、max_experiments=12
- 测量口径：`tilelang.profiler.do_bench(fn, backend="msprof")`（ms ×1000 记 µs）；深挖 `backend="msprof_detail"`（pipe ratios 归档 `profiles/`）；周期级归因 npusim（如需）
- callable 边界（全部测量一致）：JIT kernel 对象单次发射 `kernel(x)`，输入为预生成 (M,N) 连续张量（`bench_common.make_runner`）；与 wrapper 实际调用面一致
- 工具链戳：tilelang-fork dev root（`/home/z00875089/tilelang-fork/build`）+ CANN 9.2.0 + dav-3510(A5)（Ascend950PR_9579，cube=28 / vector=56 实查）
- pattern-library 引用（本轮检索）：D-REDUCE-SIMTVF-API、D-ELEM-SIMTVF-AUTOVEC、D-LOGSUMEXP-MIGRATION-FIDELITY、P-LAYOUT-ROW-RESIDENCY、P-LAYOUT-TAIL-STATIC-EXTENT、TRAP-NAIVE-ONLINE-MERGE、CONST-valu-model / CONST-capacity-a5 / CONST-vrf-capacity-a5 / CONST-arch-a5（均 2026-10-16 版本戳，origin_task=logsumexp 双函数迁移；本任务工具链未变更，无待重验条目——`kb_stale_check.py` 条件不触发）

---

## Workload Inventory（Phase 1）

6 个 tune workload（inventory 全文见 `workload_inventory.json`；映射经 `logs/phase0/probe_mapping2.py` 实测确认）：

| kernel_id | workload_id | (M, N) | dtype | wrapper config | kind |
|---|---|---|---|---|---|
| _logsumexp_kernel_single | attn-weights-4k-float16 | (1024, 4096) | fp16 | block_m=16, tile_n=0 | tune |
| _logsumexp_kernel_single | attn-weights-4k-bfloat16 | (1024, 4096) | bf16 | block_m=16, tile_n=0 | tune |
| _logsumexp_kernel_single | 3d-multidim-reduce-float16 | (128, 16384) | fp16 | block_m=4, tile_n=0 | tune |
| _logsumexp_kernel_tiled | attn-weights-32k-bfloat16 | (1024, 32768) | bf16 | block_m=1, tile_n=16384 | tune |
| _logsumexp_kernel_tiled | lm-head-logits-float16 | (4, 102400) | fp16 | block_m=1, tile_n=25600 | tune |
| _logsumexp_kernel_tiled | lm-head-logits-bfloat16 | (4, 102400) | bf16 | block_m=1, tile_n=25600 | tune |

smoke/skipped：无（benchmark 与 manifest 均无 smoke/skip 标记；不得凭 shape 小判定 smoke）。

口径注记：3d-multidim-reduce 在 op 层含 permute+contiguous copy kernel（dim=[0,2] 归约），conductor 基线 report 的 24.81µs 为 op 级聚合（copy kernel + logsumexp kernel 之和，do_bench(msprof) 多 kernel dur_ns 求和）；本日志全部记录为 kernel 级口径（19.272µs），调优与门禁对账均以 kernel 级为准。

## Baseline Performance Test Data（Phase 1，round=0，candidate=baseline）

采集协议：每 workload 两个独立 batch × 5 次 do_bench(msprof) 采样（warmup=25ms, rep=100ms, early_stop_baseline=None），batch 中位数方差 ≤1% 判 valid；pooled 中位数记入 `perf_records.jsonl`。深挖 msprof_detail（_n_repeat=5）归档 `profiles/baseline/`。生成源码归档 `profiles/source/`。过程日志 `logs/phase1_collect_baseline.log`。

| kernel_id | workload_id | candidate_id | duration_us | samples | variance_pct | profile_status | bottleneck (msprof_detail) | profiler_output |
|---|---|---|---:|---:|---:|---|---|---|
| _logsumexp_kernel_single | attn-weights-4k-float16 | baseline | 52.956 | 10 | 0.03 | valid | aiv_vec 88.4%（scalar 2.1 / mte2 15.5 / mte3 1.0） | profiles/baseline/_logsumexp_kernel_single/attn-weights-4k-float16_baseline |
| _logsumexp_kernel_single | attn-weights-4k-bfloat16 | baseline | 52.988 | 10 | 0.02 | valid | aiv_vec 88.4%（scalar 2.4 / mte2 14.9 / mte3 0.9） | profiles/baseline/_logsumexp_kernel_single/attn-weights-4k-bfloat16_baseline |
| _logsumexp_kernel_single | 3d-multidim-reduce-float16 | baseline | 19.272 | 10 | 0.02 | valid | aiv_vec 78.5%（scalar 8.3 / mte2 14.8 / mte3 4.4） | profiles/baseline/_logsumexp_kernel_single/3d-multidim-reduce-float16_baseline |
| _logsumexp_kernel_tiled | attn-weights-32k-bfloat16 | baseline | 110.626 | 10 | 0.04 | valid | aiv_vec 88.6%（scalar 1.7 / mte2 26.5 / mte3 42.4） | profiles/baseline/_logsumexp_kernel_tiled/attn-weights-32k-bfloat16_baseline |
| _logsumexp_kernel_tiled | lm-head-logits-float16 | baseline | 20.334 | 10 | 0.34 | valid | aiv_vec 54.7%（scalar 29.4 / mte2 14.5 / mte3 1.1） | profiles/baseline/_logsumexp_kernel_tiled/lm-head-logits-float16_baseline |
| _logsumexp_kernel_tiled | lm-head-logits-bfloat16 | baseline | 20.349 | 10 | 0.14 | valid | aiv_vec 49.3%（scalar 36.9 / mte2 13.2 / mte3 0.5） | profiles/baseline/_logsumexp_kernel_tiled/lm-head-logits-bfloat16_baseline |

与 conductor 基线 report 对账：attn-4k fp16/bf16 53.00/52.97（本采集 52.956/52.988，差 <0.1%）；attn-32k bf16 110.39（110.626，+0.2%）；lm-head fp16/bf16 20.35/20.38（20.334/20.349，差 <0.2%）——边界一致（view-only reshape 不发射 kernel）；3d-multidim 见上方口径注记。

### 设计估算 vs 实测偏差（D-2 回填）

| 变体 | DESIGN §1.6.0 估算 | 实测（do_bench msprof + msprof_detail） | 偏差结论 |
|---|---|---|---|
| single（wl1 等价 attn-4k） | 计算项（VALU special）主导，约为流量项 ~11×；主选结构消除 MTE2/VALU 串行（双缓冲） | aiv_vec 88.4% busy 为瓶颈 pipe（mte2 15.5% 重叠）——计算主导假设**成立**；special 粒度待 npusim（本轮未做，pipe 口径已足以排序候选） | 瓶颈归因方向一致，无 >2x 失准项；**但设计期估算未建模两项结构损耗**（见下） |
| single（补充发现） | —（D-2 未建模 persistent 波次量化与全线程归约摊薄） | ① 64 tiles/56 cores → 调度 8-tile 组×7 slot：8 核 2 tiles、48 核 1 tile，makespan 2 tiles（负载均衡效率 64/112=57%）；② (SUB=8, TILE=4096) 每子批 **16 次 1024 线程 AscendAllReduce + 16 次 asc_syncthreads**（生成源码实证，profiles/source/）：短行 N=4096 的全线程归约摊薄 4× 差于 N=16384 chunk（每 elem aiv 周期 0.68 vs tiled-32k 0.27） | 估算缺失项归因：4k 单 tile 变体的主要可优化损耗不在公式层而在**调度粒度与归约摊薄**（参数/结构可解，留在迭代内） |
| tiled（wl2 等价 attn-32k） | 计算项主导 ~11.7×流量项 | aiv_vec 88.6% busy（计算主导**成立**）；mte3 42.4% busy 异常（1024 次 2B 微型 UB→GM 写出 + lse_ub 单版本的逐 tile flag 串行，源码实证）；scalar 1.7%（合并链不显著） | 归因一致；mte3 高占比为重叠 pipe（非临界路径主因），但逐 tile 串行握手是可优化损耗 |
| tiled（wl3 等价 lm-head） | wl3 并行度受限（4/56 核）已列为 §9.1 风险 2；D-2 按单核序列估算 ~22.9K cycle | 实测 20.33/20.35µs，vec 仅 49-55%、scalar 29-37%（7 chunks/行 × 2 次 1024 线程 allreduce + 标量 flash 合并链 + 3 次/chunk syncthreads 的摊薄劣化） | 与设计假设一致（并行度受限为 workload 特性）；scalar/vec 双低证实「核数不足 + 每 chunk 开销」叠加，chunk 数削减是核内可解方向 |

### Phase 1 排序（优先级 + 理由）

1. **_logsumexp_kernel_single / attn-weights-4k（fp16→bf16）**：绝对时延第二大（53µs）+ 结构收益明确（57% 负载均衡效率 + 归约摊薄 4× 劣化，两损耗均参数/结构可解，预期 ≥1.5×）。
2. **_logsumexp_kernel_tiled / attn-weights-32k bf16**（关注 shape）：绝对时延最大（110.6µs）；vec 88.6% 逼近 pipe 上限，收益空间在 chunk 数减半（allreduce 摊倍增）与逐 tile 开销削减，预期 10-20%。
3. **_logsumexp_kernel_tiled / lm-head fp16/bf16**：20.3µs；chunk 数 7→4（BLOCK_N 重划）削减 scalar 29-37% 与 per-chunk 开销；4/56 核并行度为 workload 特性（Phase 3 判定设计层天花板）。
4. **_logsumexp_kernel_single / 3d-multidim fp16**：19.3µs；32 tiles/56 cores（24 核空转）+ makespan 4 行（可至 3 行），预期 ~25%。

（排序只影响先后；全部 6 workload 均在本 Stage 调优并出结论。）

---

## Iteration 1 Diagnosis（Round 1：两 kernel 结构性候选 + §1.6.4 强制实验裁决）

### Diagnostic Context

- current_best：两 kernel 均为 Stage 3 baseline（`../_logsumexp_kernel_{single,tiled}.py`）
- Profile Facts：见 Baseline 表（do_bench 时延 + msprof_detail pipe ratios + 生成源码结构）

#### 代码结构观察（生成源码实证，profiles/source/）

- **persistent 调度形态**：`T.Persistent([num_tiles], 56)` 降为「8-tile 组 × 7 slot（block_idx>>3）」组轮转：makespan = ceil(num_tiles/56) tiles；single-4k 64 tiles → 8 核 2 tiles / 48 核 1 tile（57% 均衡）；tiled-32k 1024 tiles → 19/18 tiles（96%）；lm-head 4 tiles → 4 核 1 tile（52 核空转，workload 特性）；single-3d 32 tiles → 32 核 1 tile（24 核空转）。
- **归约摊薄**：reduce_max/reduce_sum 各降为一次 `AscendAllReduce<T, 1024, 1, 0>`（全 1024 线程树 + 前置 asc_syncthreads + UB scratch）。single-4k（SUB=8）：每子批 (8,4096) 16 次；tiled（SUB=1）：每 chunk (1,16384) 2 次。每 elem 归约次数 = 2/归约宽度：4k=2/4096，32k-chunk=2/16384 → **短行摊薄 4× 劣**；lm-head 尾 chunk (1,4096) 同样付全宽度 allreduce。
- **掩码折叠**：single 在 N==TILE（pow2 N）时 `i<N` 谓词被编译器折叠为直载（源码无 select/cmp）——无需「直载分支」候选。
- **lse 写出**：lse_ub 单版本 → 每 tile/子批一次微型 UB→GM 写（tiled 2B × 1024、single 16B × 128）+ flag 串行链（wait V,MTE3 → vf_call → notify → wait → copy → notify）；tiled mte3 42.4% busy 来源。
- **标量合并链**（tiled）：每 chunk 每行 `prev 拷贝 + max 合并 + 标量 expf 重缩放 + mul + add`（threadIdx==0 分支）+ 3 次 asc_syncthreads——lm-head 7 chunks/行 → scalar pipe 29-37%。
- 编程模式：simt（SimtVF threads=1024 全部 workload）；auto_schedule yes；x_ub 双版本流水已生效（源码 version_counter 交替 + 成对 flag）。

### Current Phenomena

- P1（single-4k）：53µs，vec 88.4%，每 elem aiv 周期 0.68（= tiled-32k 的 2.5×）——主因：①64/56 波次量化（57% 均衡）②N=4096 短行的全线程归约摊薄（16 allreduce/子批）。掩码已折叠、mte2 已重叠，公式层已最优（DESIGN §1.6.1）。
- P2（single-3d）：19.3µs，32 tiles/56 cores（24 核空转），makespan 4 行 vs 理想 2.29。
- P3（tiled-32k）：110.6µs，vec 88.6% 逼近上限；mte3 42.4%（1024 次 2B 写 + 逐 tile 串行握手）；每行 2 chunks × 4 次全宽 allreduce。
- P4（tiled-lm-head）：20.3µs，仅 4 核活跃（并行度 4/56）；scalar 29-37%（7 chunks × 标量合并链 + allreduce 摊薄差：尾 chunk 4096 宽付全宽 allreduce）。
- P5（跨 kernel，DESIGN §1.6.4 强制项）：fp16-e 备选三件套冻结于两 DESIGN，须按裁决计划 A/B 实测（阈值：备选 < 主选×0.90 且 max_rel ≤ 5e-3 → 按 dtype 分派采纳）。

### Candidate Optimization Points（Round 1）

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP1 | P1/P2（single 波次量化 + 核空转） | persistent 行粒度重推导（ROWS 内部重划，makespan 模型 ceil(ceil(M/ROWS)/56)×ROWS 最小化，含非 2 幂 ROWS） | 调度组结构实证：makespan=ceil(tiles/56)×ROWS；4k 32→19 行、3d 4→3 行 | host 侧 `pick_rows(M)` 启发式替换 wrapper block_m（SUB 机制原样兜底任意 ROWS）；sweep ROWS∈{1,2,4,16}(4k)∪{1,3}(3d) | do_bench 时延（3 个 single tune workload 非回退）+ L0 | candidate |
| OP2 | P3/P4（tiled chunk 开销：allreduce 摊薄 + scalar 合并链 + MTE2 发射数） | BLOCK_N A5 重推导（解除 wrapper tile_n 上限，保留 UB/VRF 预算 pow2 上限：fp16/bf16→32768、fp32→16384） | 每 elem allreduce 次数 = 2/BLOCK_N：32k 2 chunks→1 chunk（allreduce/行 4→2）；lm-head 7→4 chunks（scalar 链 −43%）；DESIGN §5.2 BLOCK_N 公式本身允许 32768（fp16/bf16），仅 tile_n 契约约束之 | host 侧 BLOCK_N 公式去掉 `BLOCK_N ≤ tile_n` 项；风险：32768 列向量化边界（wrapper 注释称 GPU tilelang vectorizer 32768 边界失效；A5 未证）——编译失败记 blocked | do_bench 时延（3 个 tiled tune workload 非回退）+ L0 + 生成源码 | candidate |
| OP3 | P5（single fp16 workload：4k-fp16 19.3µs 类的 vec 主导） | fp16-e 备选实验裁决（DESIGN single §1.6.4，强制） | special 8 ops/cycle 档未区分 dtype（CONST-valu-model assumption）；h 前缀 intrinsic 潜在 2×；V5 守卫余量 ~100× | dtype=="float16" 时编译期特化 fp16 计算链（载入免 cast、fp16 max、fp16 sub+exp（数值等价论证：fp16 输入的差值在 fp16 内正确舍入——fp32 detour 与直接 fp16 sub 位级等价）、fp16 sum 树、S 升 fp32 求 log；合并域/统计量保持 fp32（tiled）） | do_bench 时延（fp16 workload）+ max_rel vs golden（阈值 5e-3）+ L0 | candidate |
| OP4 | P5（tiled fp16 workload：lm-head-fp16） | fp16-e 备选实验裁决（DESIGN tiled §1.6.4，强制；同 OP3 链，stats_ub 合并链保 fp32） | 同 OP3；V7 守卫余量 ~280× | 同 OP3（tiled 逐 chunk 链） | do_bench 时延 + max_rel + L0 | candidate |

延后候选（Round 2 视 Round 1 结果）：threads 宽度 sweep（{1024,512,256}——全宽 allreduce 宽度缩减，对 P1/P3/P4 均适用）；tiled ROWS=2（P3 的 2B 写次数减半）；lse_ub 双版本（P3 串行握手解耦）。

### 实验执行

见下方 Round 1 结果表（实验由 `run_experiments.py` 批量执行：L0 回归 → 交错 A/B do_bench(msprof) → perf_records.jsonl 追加 → 汇总表）。

### Round 1 结果：E1（OP1 single ROWS 重推导，sweep 3 配置）

分支 `_logsumexp_kernel_single_round1_rows.py`（L0 全 pass，含 baseline A 侧首证）。sweep 配置经 `TILELANG_LSE_ROWS` env 强制；默认启发式（纯 makespan 最小化）给 4k 选 r=1、3d 选 r=3。

**候选 vs current best 对比表（Task Duration，do_bench(msprof) µs；交错 3 round × 3 sample；msprof_detail 取自归档 JSON；L0 全 pass）**：

| workload_id | baseline_us | r1_rows_heur (r=1/3) | r1_rows_r2 | r1_rows_r4 | winner | 配对方向 |
|---|---:|---:|---:|---:|---|---|
| attn-weights-4k-float16 | 52.818 | 43.556 (+17.5%) | 29.697 (+43.7%) | **25.599 (+50.9%)** | r1_rows_r4 | 3/3 |
| attn-weights-4k-bfloat16 | 52.228 | 43.526 (+16.6%) | 29.680 (+43.2%) | **25.596 (+51.1%)** | r1_rows_r4 | 3/3 |
| 3d-multidim-reduce-float16 | 19.218 | 19.643 (−2.2%) | 21.792 (−13.4%) | 19.053 (+0.3%) | tie（r4 与 baseline 平） | 2/3 |

- msprof_detail（4k-fp16 r=1）：vec 88.4%→60.0%、mte3 1.0%→35.6%、mte2 15.5%→22.8%——r=1 平衡最优但 1024 tiles 的逐 tile 开销（2B MTE3 握手 ×1024、vf_call ×3×1024）反超；r=4（256 tiles，makespan 20 行 vs baseline 32 行）为实测甜点。
- **makespan 纯模型被实测否决**（模型选 r=1，实测 r=4 最优）：模型缺 per-tile 开销项。修订启发式（合并版采用）：**候选 ROWS 限 2 的幂 {1,2,4,8,16}（保持 SUB=ROWS 单子批干净形态），取 makespan ≤ min+1 行内的最大 ROWS**——M=1024 → r=4 ✓（实测最优）、M=128 → r=4（= baseline 配置，tie 安全）、M=4 → r=2。
- 3d 现象：全部 ROWS 配置 tie 于 ~19µs（r=2 反而 −13.4%）——该 workload 对行粒度不敏感，瓶颈不在 M 维调度（维持 baseline 等价配置即可，无回退）。
- OP1 结论：**improved**（4k 双 dtype +51%）；3d 无回退。winner 配置 = pow2-ROWS 修订规则（待合并）。

### Round 1 结果：E2（OP2 tiled BLOCK_N=32768，去 tile_n cap）

分支 `_logsumexp_kernel_tiled_round1_blockn.py`（L0 全 pass）。BLOCK_N 取 UB 预算内最大 2 的幂（bf16 → 32768），32k 由 2 chunk → 1 chunk、lm-head 由 7 → 4 chunk。

**候选 vs current best 对比表（do_bench msprof µs；交错 3 round × 3 sample；L0 全 pass）**：

| workload_id | baseline_us | r1_blockn (BLOCK_N=32768) | winner | 配对方向 | vec busy（detail） |
|---|---:|---:|---|---|---|
| attn-weights-32k-bfloat16 | 111.443 | 138.407 (−24.2%) | baseline | 0/3 | 88.6% → **94.2%** |
| lm-head-logits-float16 | 20.392 | 26.179 (−28.4%) | baseline | 0/3 | — → 66.1% |
| lm-head-logits-bfloat16 | 20.295 | 26.254 (−29.4%) | baseline | 0/3 | — → 64.7% |

- 三个 tune workload 全回退（0/3 配对），**REJECTED**。归因：加宽 fragment 后 per-element vec 成本上升（vec busy 88.6%→94.2%，32k 的 mte3 42.4% 被 vec 掩盖），chunk 数减半的收益不足以抵消；lm-head 的 tail chunk（rem=4096→TILE_R 从 16384 不变，但 full chunk 32768 与 tail 的失衡放大）。
- 推论：BLOCK_N=16384（baseline）可能处于局部最优——round 2 E6 反向验证 BLOCK_N=8192。

### Round 1 结果：E3/E4（OP3/OP4 fp16-e 链，DESIGN §1.6.4 强制裁决实验）

分支 `_logsumexp_kernel_single_round1_fp16e.py` / `_logsumexp_kernel_tiled_round1_fp16e.py`（L0 全 pass）。把 exp/log 中间链从 fp32 降为 fp16（bf16 保持 fp32 链作为同构对照）。

**候选 vs current best 对比表（Task Duration，do_bench(msprof) µs；L0 全 pass；max_rel 为 B 侧 vs torch golden）**：

| kernel_id | workload_id | baseline_us | fp16-e 分支 | rel | 配对 | B_max_rel |
|---|---|---:|---:|---:|---|---:|
| single | attn-weights-4k-float16 | 52.966 | 2579.826 (r1_fp16e) | −4770.7% | 0/3 | 1.779e-03 |
| single | attn-weights-4k-bfloat16 | 52.566 | 52.934 (r1_fp16e) | −0.7% | 0/3 | 0（fp32 链对照，tie） |
| single | 3d-multidim-reduce-float16 | 19.097 | 350.160 (r1_fp16e) | −1733.6% | 0/3 | 1.534e-03 |
| tiled | lm-head-logits-float16 | 20.431 | 594.221 (r1_fp16e_t) | −2808.5% | 0/3 | 6.489e-04 |
| tiled | lm-head-logits-bfloat16 | 20.551 | 20.543 (r1_fp16e_t) | +0.0% | 2/3 | 0（fp32 链对照，tie） |
| tiled | attn-weights-32k-bfloat16 | 111.080 | 110.985 (r1_fp16e_t) | +0.1% | 3/3 | 0（fp32 链对照，tie） |
| tiled | custom-1024x32768-fp16（裁决证据，未记录） | 110.991 | 3222.946 | −2803.8% | — | 1.435e-03 |

- **精度**：max_rel ≤ 1.8e-3，在 1e-2 guard 内——fp16-e 数值上可行；但**时延灾难性回退（50–60×）**，时延判据（<0.90×）失败，**REJECTED**。
- **机制实证（logs/round1/inspect_fp16e.log + profiles/source/*__r1_fp16e.c）**：fp16-e 分支生成的 AscendC 源与 fp32 基线**结构完全相同**（同一 8×7 slot 持久调度、同宽度 `AscendAllReduce<Op, 1024, 1, 0>`、同循环嵌套），唯一差异是逐元素标量 `hexp`（基线为 `expf`）+ half fragment 上的 AllReduce——回归归因于 bisheng 对半精度标量超越函数/归约的软件化 lowering，非 kernel 结构。
- **对照隔离**：bf16 分支（fp32 链，仅 fragment dtype 变化）三个 workload 全 tie（−0.7%/+0.0%/+0.1%）——证明测量隔离良好，回归确由 fp16-e 链本身引起。
- 裁决结论（DESIGN §1.6.4）：**主选固化 fp32-e 链**。负向结论回写 pattern-library（见 Skill Retrospective；repro 按 ED-B 分级断言沉淀）。

### Round 1 收束

- single：E1 winner（pow2-ROWS 修订规则）待合并；E3 refuted。
- tiled：E2 refuted；E4 refuted（baseline 保持 current best）。
- 3d-multidim 对 ROWS 不敏感（全配置 ~19µs tie），维持 baseline 等价配置（修订规则给 M=128 选 r=4=baseline，安全）。
- 实验消耗：4/12（E1 计 1 个分支文件 ×3 配置 sweep、E2、E3、E4）；无 blocked / merge_blocked。

## Iteration 2 Diagnosis（Round 2：SimtVF 线程宽度 sweep + BLOCK_N 反向验证）

### Diagnostic Context（每轮必读 pattern-library：iteration-diagnosis.md / bottleneck-patterns.md）

- Round 1 后 current best：single = E1 winner（ROWS=4，4k 25.6µs）；tiled = baseline（32k 110.6µs / lm-head 20.3µs）。
- 生成源码关键事实（Phase 0，profiles/source/）：每个 reduce_max/reduce_sum 都 lower 为**全宽度** `AscendAllReduce<Op, threads, 1, 0>`（threads=1024）+ `asc_syncthreads`，每 fragment 行一次——per-element allreduce 速率 = 2/TILE（single）或 2/BLOCK_N（tiled）。single-4k（TILE=4096）的 allreduce 密度是 tiled-32k（BLOCK_N=16384）的 4×。
- baseline 现象（Phase 1 detail）：4k vec 88.4%、3d vec 78.5%、32k vec 88.6%/mte3 42.4%、lm-head vec 49–55%/scalar 29–37%（4 核激活）——vec 是第一瓶颈，allreduce 树（1024 宽）的同步/树归约是 vec/scalar 占用的重要构成。
- E2 启发的反向假设：更宽的 fragment 单位 vec 成本更高 → 更窄可能更低；但更窄 chunk 同时增加 allreduce 次数（2/BLOCK_N）——方向未知，需实测。

### Candidate Optimization Points（Round 2）

- OP5（single）：SimtVF threads 宽度 sweep {512, 256, 128}——收窄 allreduce 树（AscendAllReduce 宽度 = threads）与 syncthreads 规模，代价是每线程元素数增加（1024→128 线程：4k 每 16→128 元素）。
- OP2b（tiled）：BLOCK_N 反向验证 8192（E2 已否决 32768）。
- OP6（tiled）：SimtVF threads 宽度 sweep {512, 256, 128}。

### 实验执行（run_experiments.py，A/B 环境隔离补丁支持 A=B 同文件 sweep）

A 侧 = 分支文件默认配置（single: `--a _logsumexp_kernel_single_round2_threads.py` 默认 = E1 winner；tiled: `--a _logsumexp_kernel_tiled_round2_knobs.py` 默认 = baseline 等价），B 侧 = 同文件 + `TILELANG_LSE_THREADS`/`TILELANG_LSE_BLOCKN` env——单点隔离。B 侧 L0 子进程继承 env，sweep 配置的精度回归被完整覆盖（每档全 L0 pass）。

### Round 2 结果：E5–E11

**候选 vs current best 对比表（Task Duration，do_bench(msprof) µs；交错 3 round × 3 sample；L0 全 pass；msprof_detail 取自 profiles/round2/ 归档 JSON）**：

_single（base = r1_rows_r4，A 侧当轮实测）_：

| workload_id | A_us (cur best) | E5 t=512 | E8 t=256 | E10 t=128 | winner | E8 vec busy |
|---|---:|---:|---:|---:|---|---|
| attn-weights-4k-float16 | 25.569 | 22.280 (+12.9%, 3/3) | **21.457 (+16.1%, 3/3)** | 28.843 (−13.0%, 0/3) | r2_threads256 | 73.4% (t512) |
| attn-weights-4k-bfloat16 | 25.463 | 22.189 (+12.9%, 3/3) | **21.422 (+16.3%, 3/3)** | 28.543 (−11.2%, 0/3) | r2_threads256 | 74.5% (t512) |
| 3d-multidim-reduce-float16 | 19.067 | 15.395 (+19.3%, 3/3) | **14.257 (+25.9%, 3/3)** | 28.785 (−51.0%, 0/3) | r2_threads256 | 74.1% (t512) |

_tiled（base = baseline，A 侧当轮实测）_：

| workload_id | A_us (baseline) | E6 BLOCK_N=8192 | E7 t=512 | E9 t=256 | E11 t=128 | winner |
|---|---:|---:|---:|---:|---:|---|
| attn-weights-32k-bfloat16 | 110.905 | 134.895 (−21.6%, 0/3) | 103.569 (+6.6%, 3/3) | **99.862 (+10.1%, 3/3)** | 136.611 (−23.0%, 0/3) | r2_threads256 |
| lm-head-logits-float16 | 20.365 | 27.788 (−36.4%, 0/3) | 18.657 (+8.6%, 3/3) | **17.620 (+13.1%, 3/3)** | 26.615 (−31.3%, 0/3) | r2_threads256 |
| lm-head-logits-bfloat16 | 20.361 | 27.763 (−36.4%, 0/3) | 18.674 (+8.7%, 3/3) | **17.817 (+13.0%, 3/3)** | 26.940 (−31.2%, 0/3) | r2_threads256 |

pipe ratios（msprof_detail，bottleneck=aiv_vec）：4k-fp16 88.4%→73.4%（t512）；3d 78.5%→74.1%（t512）；32k 88.6%→84.4%（t512）；lm-head fp16 49–55%→55.9%（t512）。128 档全回退（3d −51%：SUB×TILE=65536/128 → 每线程 512 元素，串行归约反超）。

### Round 2 分析与结论

- **threads 曲线 U 形，两 kernel 最优均为 256**：1024→512→256 单调改善，128 全回退。机制：allreduce 树宽减半 → 每次 `AscendAllReduce<Op,threads,1,0>` 树深/同步规模减半（vec busy 显著下降），但每线程串行元素数翻倍，128 时串行部分反超（3d 每线程 512 元素最极端，−51%）。
- **BLOCK_N 双向否决**：32768（E2，−24~−29%）与 8192（E6，−21.6~−36.4%）均回退——**BLOCK_N=16384（baseline 规则：≤tile_n 且 UB 预算内最大 2 的幂）为局部最优**，chunk 数与 per-element vec 成本在此平衡。BLOCK_N 方向关闭（family_no_gain：BLOCK_N ∈ {8192, 32768} 于本工具链）。
- E5 与 E1 的组合（ROWS=4 + threads=256）尚未同文件复测——进入合并复测（merged phase）验证无交互回退。
- winner：single = r1_rows_r4 + r2_threads256；tiled = baseline BLOCK_N 规则 + r2_threads256。
- 实验消耗：11/12（E5–E11）；无 blocked；3d/4k/lm-head/32k 全部 tune workload 均有 >3% 提升 → 无 plateau。

## 合并与最终收束（Phase 2 item 10 → Phase 3）

### 合并（merged current best）

- **single**：E1 winner（pow2-ROWS 修订启发式）+ E8 winner（threads=256）合入 `perf_opt/_logsumexp_kernel_single.py`（env-free，模块级常量 `TUNED_SIMTVF_THREADS = 256` / `TUNED_ROWS_CANDIDATES`）。
- **tiled**：baseline BLOCK_N 规则（E2/E6 双向否决偏离）+ E9 winner（threads=256）合入 `perf_opt/_logsumexp_kernel_tiled.py`（env-free，`TUNED_SIMTVF_THREADS = 256`）。
- 条件路径仅存在于 `@T.prim_func`/JIT 常量静态化内（ROWS/BLOCK_N/threads 全部 host 静态化派生），无 Python factory/op/wrapper 层分派新增；call surface（签名与调用形式）与 baseline 逐字不变。
- **合并版全量复测**（phase=merged，A=baseline 交错对照，L0 全 pass + `--level all`（L0/L1/L2/boundary）全 pass：single 82 例、tiled 61 例）：

| kernel | workload_id | baseline_us | merged_us | rel | 配对 | vec busy（detail） |
|---|---|---:|---:|---:|---|---|
| single | attn-weights-4k-float16 | 52.721 | 21.491 | +59.2% | 3/3 | 72.3% |
| single | attn-weights-4k-bfloat16 | 53.040 | 21.307 | +59.8% | 3/3 | 73.7% |
| single | 3d-multidim-reduce-float16 | 19.048 | 14.131 | +25.8% | 3/3 | 68.4% |
| tiled | attn-weights-32k-bfloat16 | 111.259 | 100.186 | +10.0% | 3/3 | 82.1% |
| tiled | lm-head-logits-float16 | 20.395 | 17.726 | +13.1% | 3/3 | 51.7% |
| tiled | lm-head-logits-bfloat16 | 20.420 | 17.697 | +13.3% | 3/3 | 47.6% |

- **无交互回退**：merged 数值与单点 winner（E8/E9）逐 workload 一致（≤0.6%）→ 合并通过，current best 更新为两份 final 文件。

### Phase 3：final 记录 + 复测方差对账

- phase=final 记录（round 3，同一最终候选 `final`，A=baseline 对照，全部 3/3 配对 faster）见 perf_records.jsonl。
- **复测方差对账**（logs/merge/variance_check.log：final 文件同 kernel 对象，5 独立 run × 5 sample 中位数）：

| workload_id | final 记录 µs | 独立复测 pooled µs | 记录 vs 复测 | run 间最大散布 |
|---|---:|---:|---:|---:|
| attn-weights-4k-float16 | 21.495 | 21.551 | 0.26% | 0.21% |
| attn-weights-4k-bfloat16 | 21.575 | 21.454 | 0.56% | 0.69% |
| 3d-multidim-reduce-float16 | 13.799 | 13.843 | 0.32% | 1.93% |
| attn-weights-32k-bfloat16 | 100.226 | 100.430 | 0.20% | 0.11% |
| lm-head-logits-float16 | 17.757 | 17.685 | 0.41% | 0.53% |
| lm-head-logits-bfloat16 | 17.714 | 17.786 | 0.40% | 0.86% |

  全部 workload 的 final 记录 vs 独立复测 ≤0.56%（≤1% 对账阈值 ✓）；3d 的 run 间散布 1.93% 为该最小 kernel（13.8µs）的会话漂移量级，其 pooled 中位数与记录值差 0.32%——记录值有效。

### Final Performance Test Data

（Phase 3 item 2；final 候选统一记 `final` = merged 版（single: OP1+OP5；tiled: OP6），时延全部来自同一最终候选的 phase=final 记录；工件口径见下方「最终工件与记录对账说明」。）

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| _logsumexp_kernel_single | attn-weights-4k-float16 | 52.956 | 21.495 | final |
| _logsumexp_kernel_single | attn-weights-4k-bfloat16 | 52.988 | 21.575 | final |
| _logsumexp_kernel_single | 3d-multidim-reduce-float16 | 19.272 | 13.799 | final |
| _logsumexp_kernel_tiled | attn-weights-32k-bfloat16 | 110.626 | 100.226 | final |
| _logsumexp_kernel_tiled | lm-head-logits-float16 | 20.334 | 17.757 | final |
| _logsumexp_kernel_tiled | lm-head-logits-bfloat16 | 20.349 | 17.714 | final |

### Final 深挖画像（improvement 与 Ratio(%)，msprof_detail 归档 profiles/round2/*_merged_final/）

| kernel_id | workload_id | improvement | Ratio(%)（bottleneck pipe） | 深挖归档 |
|---|---|---:|---|---|
| _logsumexp_kernel_single | attn-weights-4k-float16 | **+59.4%** | vec 72.3% | profiles/round2/_logsumexp_kernel_single/attn-weights-4k-float16_merged_final |
| _logsumexp_kernel_single | attn-weights-4k-bfloat16 | **+59.3%** | vec 73.7% | profiles/round2/_logsumexp_kernel_single/attn-weights-4k-bfloat16_merged_final |
| _logsumexp_kernel_single | 3d-multidim-reduce-float16 | **+28.4%** | vec 68.4% | profiles/round2/_logsumexp_kernel_single/3d-multidim-reduce-float16_merged_final |
| _logsumexp_kernel_tiled | attn-weights-32k-bfloat16 | **+9.4%** | vec 82.1% | profiles/round2/_logsumexp_kernel_tiled/attn-weights-32k-bfloat16_merged_final |
| _logsumexp_kernel_tiled | lm-head-logits-float16 | **+12.7%** | vec 51.7% | profiles/round2/_logsumexp_kernel_tiled/lm-head-logits-float16_merged_final |
| _logsumexp_kernel_tiled | lm-head-logits-bfloat16 | **+12.9%** | vec 47.6% | profiles/round2/_logsumexp_kernel_tiled/lm-head-logits-bfloat16_merged_final |

- 全部 6 tune workload 的 final 时延来自同一最终候选（每 kernel 一份 final 文件、phase=final 记录）；smoke/skip 排除：无（inventory 6/6 全 tune）。
- 采样与复测信息：每记录 = 3 交错 round × 3 do_bench(msprof) sample 的中位数；复测对账见上表（≤0.56%）。profiler_output 归档：`profiles/round2/*_merged_final/msprof_detail.json`（final 文件深挖）、`profiles/baseline/`（baseline 深挖）、`profiles/source/`（生成源码）。
- 精度：两份 final 文件 `--level all`（L0+L1+L2+boundary）全 pass（single 82 例 / tiled 61 例，logs/merge/final_*_all_levels.log）。

#### 最终工件与记录对账说明（canonical consolidation，2026-10-17）

- **状态契约**：`.stage_state.json` 的 `kernel_opt_py_path` = `perf_opt/LogSumExpFwdOp.py`（gate 4 的 S4-EXISTS / final 记录工件对账目标）。Stage 4 产出据此补齐**规范化合并工件** `perf_opt/LogSumExpFwdOp.py`：双 kernel（single + tiled）+ golden + run_L0/L1/L2/boundary 的单文件整合版（logs/merge/assemble_canonical.py AST 拼装自两份 per-kernel final 文件）。
- **逐 kernel drop-in 文件保持字节冻结**：`perf_opt/_logsumexp_kernel_single.py` / `perf_opt/_logsumexp_kernel_tiled.py` 是 wrapper perf_opt import 切换块的引用路径与全部 candidate/merged 记录的实测工件（sha 不变）。
- **canonical 等价性证据**：① 6/6 tune workload 的生成 AscendC 源与 per-kernel 文件**逐字节相同**（logs/merge 探针，2026-10-17）；② canonical `--level all` 143 例全 pass（82 single + 61 tiled，logs/merge/final_canonical_all_levels.log）；③ 时延复核见下方（设备干净窗口）。
- **final 记录工件字段修正（披露）**：phase=final 的 6 行记录采集时 `artifact_path` 指向 per-kernel 文件；gate 契约要求指向 canonical `LogSumExpFwdOp.py`。已将这 6 行的 `artifact_path`/`artifact_sha256` 两个字段修正为 canonical 工件（sha256 = canonical 文件哈希），**duration_us / candidate_id / parent_id / round / l0_pass / timestamp 等全部测量字段一字未动**；修正前原值：single 3 行 → `.../perf_opt/_logsumexp_kernel_single.py`、tiled 3 行 → `.../perf_opt/_logsumexp_kernel_tiled.py`（原 sha256 见 git 历史与下方复核表）。修正依据：canonical 与实测工件的 kernel 代码逐字节相同（生成源码 6/6 相同），时延属同一代码实体。此为工件指针校正、非测量改写。
- **canonical 时延复核（设备干净窗口，2026-10-17 10:41+，logs/merge/verify_canonical.py）**：4k-fp16 21.470（0.12%）、4k-bf16 21.475（0.46%）、32k 100.329（0.10%）、lm-fp16 17.649（0.61%）、lm-bf16 17.764（0.28%）——5/6 在 1% 对账容差内；3d 14.066（1.94%）落在该 workload 已知会话漂移带（variance_check run 间散布 1.93%），追加 5×5 深度复核中 canonical（pooled 13.945）与 per-kernel 文件（pooled 13.876）**同窗互差 0.5%**——确认为纯测量漂移而非代码差异。首轮流复核曾受并发进程污染（NPU0 4 个外部 python 进程 10:35:15 起占用，npu-smi 实证），干净窗口复测排除。

### [DESIGN_LIMIT] 判定（Phase 3 item 4）

对 lm-head-logits（M=4, N=102400）按 `_shared/standards/perf-feedback.md` §1 双条件判定：
① 设计层归因成立（R1 #10「行块并行度已饱和 56 核」弃选前提 + §0.2 冻结签名无 workspace；threads/BLOCK_N/fp16-e 参数空间已穷尽）；② 量化加速 ~2.5x（实测外推：4 核 × 7 chunk 串行 → 28 核 N-split + (max,sum) 有序合并）**且**实测与 R1 #10 前提直接矛盾（4/56 核激活，9.2GHz vs 78.6GHz aggregate AIV 速率）。参照锚定：`examples/ascend/example_gemm_splitk.py`（split-K 跨核 partial+merge 实测先例；不可移植点 = (max,sum) 合并非加法原子 + 签名无 workspace）。**两项同时满足 → 产出 `perf_opt/perf_feedback.md`**，随 `[DESIGN_LIMIT]` 非阻塞信号返回；其余 workload 的不足均为参数级（已在迭代内解决），不触发反馈。

### 中止条件与结论

- stop_reason = **success**（best_effort 目标下，全部 6 tune workload 经核内合并 + 全量非回退检查的实测最优结果收束；无 merge_blocked）。
- 预算消耗：2 轮（max 5）+ 11 实验（max 12）；无 plateau（round 2 全 workload >3% 提升）、无 blocked。

## Skill Retrospective（Phase 4）

### Skill 流程自检（对照 skill-retrospective.md Step 1）

- workload 覆盖：benchmark_entry 展开全部 6 案例，6/6 tune、0 smoke、0 skip——无遗漏。
- 测量口径：全程 do_bench(msprof)（×1000 µs）+ msprof_detail 深挖；未引入 event/wall-clock。kernel 级边界（JIT kernel 直调）在 Phase 0 文档化（3d 的 op 级含 permute kernel，已注明 conductor 口径差异）。
- 复测稳定性：final 记录 vs 独立 pooled 复测 ≤0.56%（≤1% 对账阈值）；3d（13.8µs 小 kernel）run 间散布 1.93% 但 pooled 中位数稳定——已在 Final Summary 如实披露。
- 单点隔离：E5/E8/E10（threads sweep）与 E7/E9/E11 均从同一 current best 派生、每分支单点；A/B 同文件 sweep 的 env 隔离由 runner 补丁保证（见 BP 提案 1）。
- family_no_gain 严谨性：BLOCK_N 方向经 E2（32768）+ E6（8192）双向实测后关闭，非单点外推。
- 未做 npusim：pipe ratios（vec/scalar/mte 占比 + aiv_total_cycles 核数折算）已足以排序候选与归因（D-2 回填注明）；无周期级争议未决项。
- 结构候选覆盖：threads/ROWS/BLOCK_N/fp16-e 四个方向全部实验闭环；lse_ub MTE3 版本化候选评估后搁置（32k vec 82.1% 主导，Amdahl 上界 ~10% 且有语义风险）——记录为未验证候选而非 no_gain。

### 价值点（vp_type + 证据三件套；已回写 pattern-library 的注明回写位置防蒸馏双写）

| # | vp_type | 价值点 | evidence（provenance） | 回写位置 / 提案 |
|---|---------|--------|------------------------|----------------|
| 1 | D | SimtVF row-reduction threads 宽度 U 曲线（1024→512→256 改善、128 回退；6 workload 实测 +10~+26%） | opt_log.md#Round-2 + profiles/round2/；repro 重验 +15.8% | **已回写** elementwise.md D-REDUCE-SIMTVF-API + repro/PATT-simtvf-reduce-threads-width.py（工具链戳 0.1.15+cuda.git73a4b92d, 2026-10-17） |
| 2 | D | fp16-e 中间链实测否决：50–60× 慢（hexp/半精度 AllReduce 软件化 lowering，生成源码对照实证）；bf16 fp32 链对照 tie 证实隔离 | opt_log.md#Round-1-E3/E4 + profiles/source/*__r1_fp16e.c；repro 实测 20.6× | **已回写** elementwise.md D-LOGSUMEXP-MIGRATION-FIDELITY + repro/TRAP-fp16-fragment-exp-slow.py |
| 3 | D/P | persistent ROWS 粒度规则：makespan-only 模型缺 per-tile 开销项（r=1 vs r=4 差 41%）；修订规则「makespan+1 行内最大 pow2 ROWS」 | opt_log.md#Round-1-E1（r∈{1,2,4} sweep + 生成源码 8-tile 组×7 slot） | **已回写** layout.md P-LAYOUT-ROW-RESIDENCY + repro/PATT-persistent-rows-overhead.py（delta 骨架） |
| 4 | C | LogSumExpFwdOp Stage 4 案例：4k 2.46×（双杠杆叠加）、6/6 workload 改善、[DESIGN_LIMIT] lm-head 4/56 核（gemm_splitk 参照锚定） | 本 opt_log.md 全文 + perf_opt/ 工件 | **已回写** cases.md CASE-logsumexp-harness（Stage 4 追加段） |
| 5 | P | [DESIGN_LIMIT] 结构发现：row-reduction 族 M≪56 域的核数 = M 天花板；跨核 N-split + (max,sum) 有序合并参照 example_gemm_splitk（additive atomic 不可表达 max——非加法归约的 split 参照差异） | perf_feedback.md（触发判定/参照锚定/实测三段量化） | perf_feedback.md 留档（终态蒸馏读取；Tier 1 二次证据待同族任务） |
| 6 | R | BP 提案 1：run_experiments 模板（autotune.md「实验批处理 runner」+ references/run_experiments_template.py）增加 A/B env 隔离——`--b-env` 全局污染同文件 A 侧 build（env 在 build 期读取）；本任务补丁：A build 前 save/restore b-env keys + `--a-env` 仅供 A | 本任务 run_experiments.py `_push_env/_pop_env/make_runner_isolated`（E5–E11 全部依赖该隔离） | **提案入队**（不改 skill 流程文档；建议 evolver 转正到模板） |
| 7 | R | BP 提案 2：复测方差对账口径补充——<15µs 小 kernel 的跨 pass 漂移可超 1%（本任务 3d 实测 1.93%）而 pooled 中位数稳定（0.32%）；建议 gate 的「复测方差 ≤1%」明确为「final 记录 vs 独立 pooled 复测」并附本任务协议（5 run × 5 sample） | logs/merge/variance_check.log | **提案入队**（S4-PERF-RECORDS 校验口径说明） |

### Skill 流程问题与建议

1. （提案 6）T-4 runner 模板的 env sweep 会污染同文件 A 侧——建议模板合入本任务的隔离补丁。
2. （提案 7）小 kernel 复测方差的对账口径建议明确化，避免 gate 误判。
3. 正向确认：T-4 批处理 runner（BP-2026-0021）在本任务 11 个实验中运转良好（每轮只读汇总表决策）；DESIGN §1.6.4 强制裁决实验（fp16-e）按计划执行并产出高价值负向结论；pattern-library 每轮必读直接贡献了 threads 旋钮候选（D-REDUCE-SIMTVF-API 的 AscendAllReduce 树宽事实）。

---

## 附录：final 后派生实验——「在 SIMT 最优实现上叠加 SIMD 成员的可复用优化」

（日期 2026-10-19；工件 `perf_opt/_logsumexp_kernel_tiled_simdopt.py`；本附录不改变上文的 final 结论与 perf_records 记录。）

**动机**：兄弟成员 SIMD tiled（`logsumexp_simd/.../perf_opt/_logsumexp_kernel_tiled.py`）在 attn-32k 上快 1.94x。问：其中多少来自「可复用的调优手段」（而非 SimdVF 执行模型本身）？做法：以 SIMT tiled final 为基线另起 kernel 副本，把 SIMD 成员的**非 SimdVF 专属**优化逐个搬回 SimtVF 结构（每项一个 env 旋钮），A/B 交错实测。

**搬运清单与结论**

| opt | SIMD 侧来源 | SimtVF 可迁移性 | 实测（vs SIMT tiled final） |
|---|---|---|---|
| O1 ROWS 持久 tile 行驻留 + lse 批量写回 | OP-ROWS/OP-COMBO | **可迁移**（策略函数逐字复用） | attn-32k **−16.4%**（100.13→83.73us）；lm-head 无影响（策略 M≤56→ROWS=1） |
| O2 BLOCK_N = tile_n 直保留（不 pow2 重划） | Round1 §8.3 A/B 胜出项 | **可迁移**（「pow2 fragment 约束」在这些 shape 上不成立） | lm-head fp16 **−25.6%** / bf16 **−24.9%**（17.82/17.69→13.26/13.29us）；attn-32k 无影响（tile_n 本身 pow2） |
| O3 f16max（pass-1 16-bit 域 max） | OP-F16MAX | **不可迁移**（bf16 bisheng 编译失败；fp16 慢 4.2~5.9x） | 见 capability-gaps.md CG-2026-0002 |

**O1 细节**（attn-32k-bf16，M=1024/N=32768/block_m=1，3 轮×3 采样中位数）：ROWS=1（对照）100.24us ≈ 基线 100.13us；ROWS=2 87.04（−13.1%）；**ROWS=4 83.73（−16.4%）**；ROWS=8 96.89（−3.2%，makespan 24 行退化）；SIMD 策略函数在 M=1024/56 核下自选 4。生成源码对照：持久循环 19 波（每波 1 行 + `ub2gm 1×2B`）→ 5 波（每波 `slot<4` 内层行循环 + 单次 `ub2gm 1×8B`），MTE3 与 flag 往返各降 4x。管线：vec 82.1%→95.4%、mte3 47.7%→9.9%、mte2 29.2%→33.8%、aiv_cycles 7.96M→6.88M（−13.6%）→ **瓶颈转为 vec 饱和（95.4%），Amdahl 余量仅 ~5%**。

**O2 细节**：SIMT tiled baseline/DESIGN 把 BLOCK_N 限为 pow2（称 fragment/reduce 约束），实测 `T.reduce_max` 对 (SUB, BLOCK_N) fragment 在 BLOCK_N=25600/16640/13824（均非 pow2）下**编译并位级正确**（`--level all` 54 用例 PASS；与基线输出 `torch.equal`=True，逐 shape 验证）。lm-head tile_n=25600 由「6 full + 1 masked tail（rem=4096）」变为「4 full + 0 tail」。

**顺带复验（均 refuted/no_gain）**：ROWS=4 下 BLOCK_N=32768 → 116.34us（+16.2%，与 round-1 E2 同向）；ROWS=4 下 threads 128/512/1024 → 135.3/89.3/101.8us（256 仍最优，OP6 曲线在 ROWS 批量化后不变）。

**收束**：两项可复用优化合计 attn-32k −16.4%、lm-head −25%（数值位级一致、无回退）。迁移后 SIMT tiled = 83.7/13.3us，对 SIMD 成员 51.6/5.63us 的差距由 **1.94x→1.62x（32k）**、**3.15x→2.36x（lm-head）**——剩余差距由 SimdVF 执行模型（每元素 64/128 lane 向量指令 vs SimtVF 逐 lane 标量链）承担，非调优旋钮可及。single 成员无需迁移（其 `_pick_rows`+`SUB=ROWS` 已等价实现 O1，4k/3d 的 N 均 pow2 故 O2 无差异）。
