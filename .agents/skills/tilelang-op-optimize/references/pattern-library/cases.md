# 案例索引与参考实现集（同类问题的正/反例参考）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)，front-matter schema 见其 §5）。
>
> **A5 冷启动（2026-10-16）**：本文件自 tilelang-mlir-fork（npuir/910B2C 实测）复制后**内容清零**——原条目绑定 npuir 工具链实测，不适用于 tilelang-fork（ascend/A5，dav-3510）。schema 保留，条目由 A5 任务蒸馏积累。

---

id: CASE-pilot-silu
kind: case
family: [elementwise, activation]
apis: [T.SimtVF, T.Parallel, T.copy, T.Pipelined]
dtype: [fp16]
device: A5/950 (dav-3510)
status: measured
origin_task: pilot-silu-20261016T025344Z（溯源：任务交付件目录，允许失效）
toolchain: tilelang-fork dev build 48345cd7 + CANN 9.2 + bisheng dav-3510 + Ascend950PR_9579，2026-10-16
repro: repro-missing
---

### examples/pilot/silu/ — simt 纯 Vector 扁平 elementwise 模板（正例；溯源：任务工作区目录，允许失效）

- 触发条件：逐元素激活类算子开发（simt 模式）；grid 整除性/尾部静默漏写排查；fp16 激活边界语义确认。
- 要点：`@tilelang.jit` 参数化工厂（一个 kernel 覆盖分层测试多 shape）；扁平 1D 连续轴（TILE=2048 = 16×128 el 向量宽整除，尾 lane 0%）；SimtVF `T.Parallel` 直写公式自动向量化（elementwise.md D-ELEM-SIMTVF-AUTOVEC）；host `assert TOTAL % (NUM_BLOCKS*TILE) == 0` 整除防御（traps-runtime.md TRAP-GRID-DIVISIBILITY）。Stage 2/3 一次通过；Stage 1 首 attempt 因探针 PYTHONPATH 错仓重试 1 次（1079s→290s）。

---

id: CASE-pilot-gemm-silu
kind: case
family: [gemm, mixed, elementwise]
apis: [T.gemm, T.dual_copy, T.SimtVF, T.exp]
dtype: [fp16]
device: A5/950 (dav-3510)
status: measured
origin_task: pilot-gemm_silu-20261016T034510Z（溯源：任务交付件目录，允许失效）
toolchain: tilelang-fork fc32f47c / CANN 9.2 / bisheng dav-3510 / Ascend950PR_9579，2026-10-16
repro: repro-missing
---

### examples/pilot/gemm_silu/ — mixed GEMM+非线性 epilogue 融合案例（正例；溯源：任务工作区目录，允许失效）

- 触发条件：GEMM + 逐元素 epilogue 融合算子设计（CV 融合结构）；激活公式选型；dual_copy 间插 Vector 段可行性确认。
- 要点：融合 vs 分离 GM 流量 −40%（2.5MB→1.5MB @512³，R3 分析口径）；dual_copy 间插 SimtVF 元素段（含 T.exp）AutoSchedule 直接支持（attention.md D-MIXED-EPILOGUE-DUALCOPY）；设计期探针 ~3 分钟否决 tanh 形式（1.80×）并裁决直除式（0.902×，elementwise.md D-ELEM-ACT-FORMULA-PROBE）；fp32 累加器 fp32 落 UB + cast 后置；静态契约 shape（M%TILE_M 校验，L1 只测 block 整除子集）。三 Stage 全一次通过。

---

id: CASE-pilot-online-softmax
kind: case
family: [softmax, reduction]
apis: [T.reduce_max, T.reduce_sum, T.Persistent, T.exp, T.infinity, T.serial]
dtype: [fp16]
device: A5/950 (dav-3510)
status: measured
origin_task: pilot-online_softmax-20261016T044814Z（溯源：任务交付件目录，允许失效）
toolchain: tilelang-fork dev build + CANN 9.x + bisheng dav-3510 + Ascend950PR_9579，2026-10-16
repro: repro-missing
---

### examples/pilot/online_softmax/ — GPU TileLang→A5 迁移 row-reduction 案例（正例，含门禁教训；溯源：任务工作区目录，允许失效）

- 触发条件：softmax/layernorm/rmsnorm 等 row-reduction 族迁移；流式 online vs 行块驻留结构抉择；换底类 GPU 优化复核；流式尾块处理。
- 要点：换底类优化（exp2·log2e）先核对目标后端超越函数代价关系（A5 exp/exp2 同档，constants.md CONST-valu-model 注记）——被舍弃优化写明「意图由什么承接」；UB 216KB 驻留判定（N=8192 fp16 行 16KB 驻留，online 结构降级为 N>32768 泛化路径）；朴素 online 合并角点发散→flash 形式+哨兵钳位（traps-runtime.md TRAP-NAIVE-ONLINE-MERGE）；尾块静态 rem 元编程分支；`@tilelang.jit` 懒模式工厂（工厂返回 PrimFunc，`_is_lazy_style` fallback 识别）。63 用例全 PASS；Stage 3 首 attempt 因 S3-JIT 门禁（tilelang.compile 形态被拒）重试 1 次（1541s→325s）。

---
id: CASE-logsumexp-harness
kind: case
family: [reduction, logsumexp, softmax]
apis: [T.Persistent, T.SimtVF, T.reduce_max, T.reduce_sum, T.copy, T.macro, T.if_then_else, T.And]
dtype: [fp16, bf16, fp32]
device: A5/950 (dav-3510)
status: measured
origin_task: logsumexp-harness-scaffold+devloop-20261016T083428Z（溯源：examples/logsumexp/ 任务工件 + examples/TileOPs/tileops/kernels/reduction/logsumexp/ 集成包 + reports/tileops/20261016_201213_480829_LogSumExpFwdOp/——允许失效）
toolchain: tilelang 0.1.15+cuda.git40c9d668 + Ascend950PR_9579 / CANN 9.2.0，2026-10-16
repro: repro-missing
---

### examples/logsumexp/ — migration-harness 全链首例（Stage 0→5；row-reduction 驻留+流式双变体；正例；溯源：任务工件目录，允许失效）

- 触发条件：harness 迁移任务参考（GPU TileOPs 同构仓多函数迁移全链：脚手架→逐函数设计/检视/实现→集成 report）；logsumexp/softmax 族双变体结构；`@T.macro` 因子化子批体；尾 chunk extent<threads reduce；TileOPs 集成链路验证。
- 要点：双函数 S3 均首过 `[PRECISION_PASS]`（fp16/bf16 位级一致，见 elementwise.md D-LOGSUMEXP-MIGRATION-FIDELITY）；wrapper dispatch 双变体分界（N_padded ≤ 32512 ∧ bm·elem·N_padded ≤ 216KB → single；UB 常量单一事实源 `examples/TileOPs/tileops/device.py` UB_SIMTVF_BUDGET_BYTES = 248KB 物理 − 32KB SimtVF 上下文）；tiled 用 `@T.macro` 消除子批体 4× 重复（traps-compiler.md TRAP-MACRO-SCOPE-PLACEMENT）；reviewer 真机探针闭合尾 chunk reduce 先例空隙（review_probe_tail_reduce.py PROBE_PASS，~70 行最小 kernel 精确复刻设计配置）；Stage 5 零 attempt 通过（45/45 正确性 + 6/6 benchmark，profiler_fallback_count=0；bench_kernel_msprof = `tilelang.profiler.do_bench(backend="msprof")` 封装链路真机实测有效）。
- 基线性能（do_bench msprof，Stage 3 产物，供同族 Stage 4 对照）：[32,32,4096] fp16/bf16（single，bm=16）53.0µs / 0.158 TB/s；[32,32,32768] bf16（tiled，bm=1 tile_n=16384）110.4µs / 0.608 TB/s；[4,102400] fp16/bf16（tiled，bm=1 tile_n=25600）20.4µs / 0.040 TB/s（M=4 行数下带宽明显受限，Stage 4 调优候选）；[4,128,4096] dim=[0,2] fp16（single，bm=4）24.8µs / 0.169 TB/s。
- 效率画像（SESSION_TIMING_ANALYSIS，13261s 总耗时）：S2 检视占 44.3%（design_calc_check 解析伪影人工消解是主因——工具修复提案见 `.agents/evolution/queue.md`）；S3 合计仅 16.1%（前序函数教训逐字注入机制实证有效：函数 2 S1 零门禁重试、per-chunk 计算体复用直接落到 tiled 实现）；Stage 5 极快（235s）。
- **Stage 4 调优追加（2026-10-17，tilelang 0.1.15+cuda.git73a4b92d）**：双 kernel 6 tune workload 全部改善——single 4k fp16/bf16 53.0→21.5µs（**2.46×**，ROWS 修订规则 + threads=256）、3d 19.3→13.8µs（+28%）；tiled 32k 110.6→100.2µs（+9.4%）、lm-head 20.3→17.7µs（+12.8%）。两个有效杠杆（threads 宽度 U 曲线、persistent ROWS 粒度）与两个实测否决（fp16-e 链 50×、BLOCK_N 双向偏离）已回写 pattern-library（elementwise.md D-REDUCE-SIMTVF-API / D-LOGSUMEXP-MIGRATION-FIDELITY、layout.md P-LAYOUT-ROW-RESIDENCY + 3 个 repro）。lm-head（M=4）4/56 核并行度天花板 → `[DESIGN_LIMIT]`（perf_opt/perf_feedback.md：R1 #10 弃选前提在 M≪56 域失效，~2.5x 结构空间被冻结签名锁死）。调优工件：`examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp_kernel/perf_opt/`（opt_log.md + perf_records.jsonl + 双 final kernel）。

---
id: CASE-logsumexp-simd
kind: case
family: [reduction, logsumexp, softmax]
apis: [T.SimdVF, T.simd.vld, T.simd.vmax, T.simd.vcvt, T.simd.vcmax, T.simd.vdupv, T.simd.vexpdif, T.simd.alloc_var, T.Persistent, T.copy]
dtype: [fp16, bf16, fp32]
device: A5/950 (dav-3510)
status: measured
origin_task: logsumexp_simd（migration-harness 双函数，2026-10-17；溯源：examples/logsumexp_simd/ 任务工件——允许失效）+ logsumexp_simd-LogSumExpFwdOpSimd-20261017T113205Z（optimize，2026-10-17/19；溯源：examples/TileOPs/tileops/kernels/reduction/logsumexp_simd/logsumexp_simd_kernel/perf_opt/opt_log.md——允许失效）
toolchain: tilelang dev build（fc32f47c 后，0.1.15+cuda.git73a4b92d 同代）/ CANN 9.2.0 + bisheng dav-3510 + Ascend950PR_9579（28 AIC/56 AIV 实查），2026-10-17/19
repro: repro-missing
---

### examples/logsumexp_simd/ — SimdVF 双变体迁移 + Stage 4 调优案例（与 SIMT 兄弟变体共存的 SIMD 重设计；正例含两处流程教训；溯源：任务工件目录，允许失效）

- 触发条件：SimdVF（simd 编程模式）row-reduction/logsumexp 族开发或迁移；与既有 SIMT 变体同进程共存（variant 算子二次迁移）；16-bit 输入域归约 / bf16 EVEN-ODD 拆分；跨 VF 块在线统计量设计；GPU TileOPs 算子的 SIMD 变体全链（脚手架→设计→检视→实现→集成→调优）。
- 要点：① **variant 共存**：custom_op 注册独立 op 名（`ascend::logsumexp_simd_fwd` vs SIMT 的 `ascend::logsumexp_fwd`）即同进程零冲突；wrapper 双 import 切换块（baseline 激活），分派函数保留在 wrapper 内；Stage 5 零 attempt（45/45 正确性 + 6/6 msprof）。② SimdVF 对 SimtVF 设计移植的系统性差异四项：无 `threads=`（连内部推导也无）、无 syncthreads 行、SUB 封顶仅 UB 单约束（VRF 流式零驻留，constants.md CONST-capacity-a5 SimdVF 注记）、块内统计量寄存器驻留免 `mem_bar`。③ pass1 16-bit 域 max + bf16 EVEN/ODD（elementwise.md D-LOGSUMEXP-SIMD-F16MAX，含宽化装载位级补注）。④ 跨 VF 块 running stats 被 VFChecker 禁止 → stats-UB 往返 fallback（`examples/ascend/flash_attention/core.py` 同构先例；条目待 VP-2026-0035 转正）。⑤ wrapper 落点防再犯机制：L0 assert import 仓内镜像 `examples/TileOPs/tileops/kernels/reduction/_primitives.py::compute_tile_n` 复算期望落点（v0 手算 16640 错误被检视拦截后 v1 机制化，Stage 3 首跑验证生效）。
- Stage 4 终局（6 tune workload，do_bench(msprof) baseline→final µs）：single 4k-f16 10.862→10.217（+5.9%）/ 4k-bf16 10.879→10.119（+7.0%）/ 3d-f16 11.308→10.703（+5.4%）；tiled 32k-bf16 72.439→51.636（**+28.7%**，f16max + `_pick_rows` ROWS 策略；mte2 85.7% 搬运墙，有效读带宽 1.27–1.30TB/s、roofline 75.2%）/ lm-head-f16 6.728→5.632（+16.3%）/ lm-head-bf16 6.803→5.669（+16.7%）。**SIMD baseline 全面 1.2–2.6x 快于 SIMT 兄弟 final**（4k 10.9 vs 21.5µs、lm-head 6.7 vs 17.7µs——结构优势，跨结构外推禁止反向成立）；§8.3 A/B 实测裁决 BLOCK_N=tile_n(25600) 胜出 16384 档 −52%（SimdVF 反甜点，layout.md P-LAYOUT-ROW-RESIDENCY SimdVF 补充段）；lm-head M=4 4/56 核激活但 N-split 估算 ~1.2–1.4x < 2x 门槛 → [DESIGN_LIMIT] 不触发（与 SIMT 兄弟 ~2.5x 结论同现象不同结构判定，algorithm-candidates.md ALG-reduction R2 注记）。3 轮 stop=plateau；实测否决集：unroll4/EVEN-ODD pair/maskhoist（scalar 三假设）、slab/SUB 下探（波次再平衡 family）、BLOCKN 16384/51200。
- 流程教训（详见 queue）：检视 Subagent 上下文超长触发 2 次 compaction 基础设施失败（2873s = 会话 12.5%，VP-2026-0036）；tiled v0 wrapper 落点手算错误跨任务复发（VP-2026-0025 第二族证据——知识库条目被检索返回但子句未被设计期消费）。数据卫生：迁移期 report 的 lm-head-bf16 0.66µs 与 Stage 4 双批次×5 采样基线 6.803µs 不符（report 侧测量伪影，同结构 fp16 双胞胎 6.65µs 佐证；以 Stage 4 协议为准）。
- 效率画像（SESSION_TIMING_ANALYSIS，迁移会话 23029s ≈ 6.4h）：检视环节 37.9%（tiled 复检 4 dispatch，其中 compaction 12.5%）；设计修订 ~22%（v0 落点错误返工链）；S3 双函数合计 12.2%（均首过，84+69 用例 max_diff 全 0）；Stage 5 1.7%。optimize 侧首轮 gate 4 工件完整性 14 条失败 → 重派 remediation（consolidated 工件契约，VP-2026-0039）。
