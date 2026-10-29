# 算法候选库（algorithm-candidates）— 常见算子族替代算法参考表

> **版本化结构化候选库**（D-4）：由 `algorithm-research.md` §5 参考表升级而来——该文件保留调研方法论，**候选表本体以本文件为唯一事实源**。R1/R2 的命中行是**必查清单**（下界不是上界）：命中行的候选必须评估，但调研不得止步于本表——结构判据（algorithm-research.md §3 R2）与外部已知算法仍须过一遍。表内条目涉及 API 时仍须本地佐证（`examples/ascend/` / `testing/ascend/` / tilelang-ascend skill）——**算法思路与 API 存在性是两回事**。
>
> **A5 冷启动状态（2026-10-16）**：`known_impl` 指向 tilelang-fork 的 `examples/ascend/` 与 `testing/ascend/`；`kb_links` 目前仅可指向 pattern-library constants.md 的 `CONST-*` 条目（PL/TRAP/CASE 类实测条目随数据层清零重建，待 A5 任务蒸馏积累）——引用前先确认条目存在，无条目时按「未实证常数」处理并进实验裁决流程。
>
> **维护**：`tilelang-skill-evolver` 蒸馏 Stage 4 发现的"算法族 X 的候选 Y 实测收益 Z"时同步更新本表（P 类，Tier 1，target_doc 指向本文件）；任务内 optimizer 回写 pattern-library 的算法级发现同样在复盘表标注本表候选行。条目带 front-matter：`known_impl`（本仓已验证实现指针）/ `kb_links`（pattern-library 条目 ID——候选的实测代价/反例证据）。检索：`kb_search.py`（K-3 统一入口）。

---
id: ALG-softmax
family: [softmax, log-softmax, logsumexp]
known_impl: [examples/ascend/flash_attention/（attention 内在线 softmax 结构，core.py 共享 builder）]
kb_links: []
---

R1 等价化简候选：减 max 稳定化（必选）；log-softmax 直接由 logsumexp 表示；exp → exp2·log2e 缩放——**A5 上此候选无收益**（exp/exp2 同为注册 intrinsic 且同 special_func 代价档，换底反多 1 次乘/elem，见 pattern-library constants.md CONST-valu-model 注记；GPU 迁移任务须复核目标后端超越函数代价关系）。**online 合并稳健化（双任务实测：online_softmax + logsumexp）**：朴素 m_c 基 lse 合并角点发散（间隙>88.7 溢出全 0、块内全 −inf NaN，违反率 13.6%/18.2%，见 pattern-library traps-runtime.md TRAP-NAIVE-ONLINE-MERGE）→ 泛化路径用 flash 形式 max 基合并；**哨兵钳位按族语义取舍**——softmax 保留 fp16 哨兵钳位（exp 之前）以保 torch NaN 传播语义，logsumexp 族弃用（源链全 −inf 行固有 NaN，钳位唯一作用是改变该行为 = 语义变更；flash 式 exp 参数 ≤ 0 角点本已安全，fp16 跨 chunk max-gap 131008 无溢出）；flash 合并对 chunk 划分满足结合律——tile_n 任意 256 倍数划分（源 GPU）与 BLOCK_N pow2 重划分（A5 fragment 约束）7 组 shape×精度 fp64/fp32 位级一致（≤1 ulp；BLOCK_N 256…16384 × tile_n 16640/25000/32512 扫描），非 pow2 tile_n 直译使 fragment next_pow2 翻倍、有效 lane 50.8%–78.1%。**驻留优先判定**：A5 UB 216KB〔SimtVF〕可驻留整行时行块驻留单遍结构优于流式 online（GM 流量 −33%、special_func −50%，模型推导口径；TileOPs logsumexp wrapper 以 N_padded ≤ 32512 ∧ bm·elem·N_padded ≤ 216KB 分界双变体、双变体位级保真实证）——online 为 N 超驻留上限（对照 CONST-capacity-a5）时的泛化路径而非默认选择（结构细则见 pattern-library layout.md P-LAYOUT-ROW-RESIDENCY）。R2 在线变体：online softmax（分块 running max/sum + 重缩放）。R3 复杂度要点：三遍（max/sum/normalize）vs 两遍 vs 单遍 online；行缓冲 O(row) vs O(tile)。R4 亲和要点：行内水平归约 vs 垂直扫描（与 §1.6.3 交互）；整行 UB 驻留；attention 场景 mixed 融合——softmax 概率直写 NZ 布局供第二 GEMM 消费可省一次 ND↔NZ 转换（flash_attention/core.py 头注）。

---
id: ALG-layernorm
family: [layer_norm, batch_norm]
known_impl: []（无直接例子；近族参考 examples/ascend/example_rmsnorm.py）
kb_links: [CONST-valu-model（special_func 吞吐档位——超越函数折算）]
---

R1：`x/sqrt(var+ε)` → `x·rsqrt(...)`；单遍式 `var = E[x²]−(E[x])²`（须评估 fp32 累加与 catastrophic cancellation）。R2：Welford 增量 mean/var；分块 running 统计。R3：两遍扫描 vs 单遍；统计量缓冲。R4：规约轴 lane 映射（见 §1.6.3）；fp32 累加路径。**row-reduction 常见形态（设计期估算参照）**：全 N 驻留 UB（GM 流量最小）+ 按 M 行块分核（persistent 化视 grid/核数关系定）+ 尾块处理；N 超驻留上限（对照 `CONST-capacity-a5`，含 multibuffering 版本膨胀）才需 N-tile 流式（引入重扫流量，须估算对比）。

---
id: ALG-rmsnorm
family: [rmsnorm]
known_impl: [examples/ascend/example_rmsnorm.py（SIMT fragment 向量化加载 + reducer epoch 归约 + T.Pipelined 行块流水 + UB 预算化的行批处理）]
kb_links: [CONST-valu-model]
---

R1：无 mean 减法（相对 layer_norm 少一遍扫描）；rsqrt。R2：平方和单遍天然在线。R3：平方和单遍；无均值遍。R4：同 layer_norm；参考 example_rmsnorm.py 的形态——`T.alloc_fragment` 2 的幂填充 + 谓词掩尾 + `T.alloc_reducer`/`reducer_init/update/finalize_reducer` + `T.copy` 行块流水。

---
id: ALG-attention
family: [attention]
known_impl: [examples/ascend/flash_attention/example_mha.py、example_gqa.py、example_gqa_manual_schedule.py（显式 T.Stage 版）]
kb_links: [CONST-capacity-a5（tile 上限）、CONST-gemm-dim-constraints（C0 分形对齐）]
---

R1：scale 融入 Q/K；softmax 分母倒数乘法化。R2：flash 结构——online softmax + 分块 KV，O(N²) 中间矩阵 → O(N)。R3：materialize O(N²) 访存/缓冲 vs flash O(N) 缓冲（FLOPs 同阶）。R4：Cube GEMM 分块 + Vector online softmax 流水（mixed）；L0C/UB 容量定 tile；QK^T 与 PV 两次 GEMM 同路径。**形态参考**：examples/ascend/flash_attention/（MHA/GQA/手动调度三版对照；`core.py`/`core_bwd.py` 共享 builder）。

---
id: ALG-conv
family: [conv]
known_impl: []
kb_links: []
---

R1：im2col + GEMM；implicit GEMM；Winograd（小核乘法次数↓加法次数↑，须按单元吞吐比〔`CONST-cube-throughput-model` vs `CONST-valu-model`〕评估）。R2：滑窗天然流式。R3：direct vs im2col——FLOPs 同阶、访存与单元利用率不同；Winograd 乘法按窗口比例下降。R4：Cube 路径 C0 分形对齐；Vector 直接窗口的 C 轴整除性；im2col 展开缓冲。

---
id: ALG-pooling
family: [pooling]
known_impl: []
kb_links: []
---

R1：avg pool = 常数权重卷积；sum pool × 常数 = avg pool。R2：滑窗天然流式。R3：窗口重叠数据的重复读。R4：C 轴向量轴；跨步系数（见 §1.6.3）。**代价提示**：核内重排 vs host permute 的取舍须实测或标注「未实证假设」（pattern-library 积累前走实验裁决流程，勿凭 GPU 直觉拍板）。

---
id: ALG-reduction
family: [reduction, statistics]
known_impl: [examples/ascend/example_rmsnorm.py（reducer epoch 用例）；回归 testing/ascend/language/test_tilelang_ascend_reduce.py、test_tilelang_ascend_warp_reduce.py]
kb_links: [CONST-arch-a5（跨核 flag 预算）]
---

R1：max + sum 合并扫描；和与平方和一次扫描。R2：分块归约 + 合并（需要保证在同一次 device kernel launch、同一个 `@T.prim_func` 内）；Welford。**M≪56 域行块并行天花板 + 跨核 N-split 候选（A5 双结构实测，2026-10-17/19）**：row-reduction 族单 kernel 行块并行结构在 M≪56 契约 shape 上激活核数 = M（lm-head M=4/N=102400：SIMT 实测 4/56 核、aggregate AIV 速率 9.2GHz vs 满载 78.6GHz；SimdVF 变体同现象，14.5–16.2GHz vs 74.6GHz——msprof_detail aiv_total_cycles/duration 折算）——该域结构候选是跨核 N-split partial+merge（参照 `examples/ascend/example_gemm_splitk.py` 实测先例：deterministic 有序归并 + atomic 双模式），但非加法归约的 (max,sum) flash 合并不可用 additive atomic 表达（`tilelang/ascend/language/__init__.py` atomic 原语仅 add 类），且需 GM workspace 形参（源签名冻结时锁死）。**结构收益倍率不可跨编程模式外推**：SIMT 结构 ~2.5x 估计建立在 17.7µs 低效串行链上；SimdVF 变体每 chunk 效率 2.6x 于 SIMT（5.7 vs 17.7µs）时同一 N-split 方案估算仅 ~1.2–1.4x（跨核 4×7 split 计算段 1.5–2µs + (max,sum) 定序合并段 2–3µs vs 当前 5.7µs，合并段开销吞掉并行收益）< [DESIGN_LIMIT] 2x 触发门槛——高效结构下 N-split 收益由「合并段开销 / 当前时延」比率决定（数字与判定见 pattern-library cases.md CASE-logsumexp-harness [DESIGN_LIMIT] 段 / CASE-logsumexp-simd Stage 4 段）。R3：扫描遍数；跨核归并代价。R4：水平 vs 垂直 lane 映射；跨核同步代价（AutoSchedule 自动 flag / 手动配对）；`T.warp_reduce_*` 硬件直连 dtype 仅 fp16/fp32/int32/uint32。row-reduction 族常见形态见 ALG-layernorm R4。

---
id: ALG-topk
family: [top-k, sort]
known_impl: []
kb_links: []
---

R1：部分选择 vs 全排序；分块 top-k + 归并。R2：分块 + 归并。R3：O(N·k) vs O(N log N)。R4：归并链的核内结构与缓冲。

---
id: ALG-gemv
family: [gemv, matvec]
known_impl: []
kb_links: [CONST-cube-throughput-model、CONST-valu-model（Cube 利用率 vs 归约代价对比口径）]
---

R1：GEMM 分形路径（Cube，`T.copy` 自动 ND2NZ + `T.gemm`）vs Vector 规约累加。R2：split-K 分块累加。R3：Cube 利用率 vs 归约代价（按两吞吐模型折算）。R4：K 维分块与 L0 容量。

---
id: ALG-triangular
family: [triangular-solve, inverse]
known_impl: []
kb_links: []
---

R1：只需解不需逆时——直接求解替代显式求逆（同阶但常数更小、数值更稳）；对角/三角特例降阶。R2：—。R3：求逆 + 乘法 vs 单次求解；特例 O(n²)/O(n)。R4：Cube 三角 GEMM 支持须按 L1-input NT 约定验证（无先例时标注「未实证假设 + 估算依据」）。

---
id: ALG-elementwise
family: [elementwise, activation]
known_impl: [examples/ascend/example_vecadd.py（simt/simd 双模式对照 + T.Pipelined 流水）、example_per_token_cast_to_fp8.py（SimdVF 掩码/分组量化）]
kb_links: [CONST-mte-bandwidth-model（搬运下界口径）]
---

R1：公共子表达式消除；除法乘倒数；cast 链合并。R2：逐元素映射天然单遍。R3：融合消除中间 GM 往返（访存收益主导）。R4：单遍 Vector 流；dtype 路径。**代价提示**：逐元素算子通常 MTE 带宽主导——计算链优化的收益须对照搬运下界估算（`CONST-mte-bandwidth-model`，勿按峰值带宽），勿只优化计算。

---
id: ALG-transpose
family: [transpose, repack]
known_impl: []
kb_links: []
---

R1：核内重排（VF 内交错/解交错 `T.simd.vintlv/vdintlv`、`T.copy` 特定路径 transpose）vs host permute（通常净亏）。R2：分块转置。R3：GM 流量不变，代价在向量管线开销与 UB 容量。R4：`T.copy` 的 `transpose=` 仅特定路径（GM→L1 选 dn2nz、L1→L0A/B 请求转置装载，tilelang-ascend skill §6.1）——非通用转置；通用转置 API 存在性须先在 `tilelang/ascend/language/` 检索佐证，未命中即标注「未文档化假设」。**注意**：epilogue 侧布局变换对照 L0C→UB `T.dual_copy` 路径评估（dtype 转换须另行 copy 或 VF cast，skill §6.2）。
