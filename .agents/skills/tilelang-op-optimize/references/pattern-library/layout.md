# 向量化轴与布局模式（优先级最高的检索项）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)，front-matter schema 见其 §5）。
>
> **A5 冷启动（2026-10-16）**：本文件自 tilelang-mlir-fork（npuir/910B2C 实测）复制后**内容清零**——原条目绑定 npuir 工具链实测，不适用于 tilelang-fork（ascend/A5，dav-3510）。schema 保留，条目由 A5 任务蒸馏积累。

（以下两条 2026-10-16 合入：Tier 1 达 2/2 双任务独立证据——pilot-online_softmax + logsumexp-harness）

---
id: P-LAYOUT-ROW-RESIDENCY
kind: pattern
family: [reduction, softmax, logsumexp]
apis: [T.Persistent, T.SimtVF, T.alloc_shared, T.alloc_fragment, T.reduce_max, T.reduce_sum, T.annotate_buffer_versions]
dtype: [fp16, bf16, fp32]
device: A5/950 (dav-3510)
status: measured
origin_task: pilot-online_softmax-20261016T044814Z + logsumexp-harness（_logsumexp_kernel_single/_tiled，2026-10-16；溯源：online_softmax DESIGN.md §1.6.0 R3/R4 + §0.6 R2、logsumexp DESIGN.md §0.2/§5.2/§5.4 与两函数 kernel——允许失效）；Stage 4 行块粒度证据：TileOPs LogSumExpFwdOp Stage 4（opt_log.md Round 1 E1 + Round 2 E2/E6，2026-10-17——允许失效）
toolchain: tilelang-fork dev build（fc32f47c 后）/ tilelang 0.1.15+cuda.git40c9d668 + CANN 9.2 + bisheng dav-3510 + Ascend950PR_9579，2026-10-16；Stage 4 行块粒度证据：tilelang 0.1.15+cuda.git73a4b92d + CANN 9.2.0 + Ascend950PR_9579，2026-10-17
repro: repro/PATT-persistent-rows-overhead.py
---

### row-reduction 族结构判定：行块驻留 vs N 向流式 online（双任务实测）

- 判定法四步：① 计算行工作集（输入 tile 双版本 + 中间量，如 N=8192 fp16 行 16KB×2 版本）；② 对照 CONST-capacity-a5 驻留预算（SimtVF 216KB × 90% 折扣，含 multibuffering 版本膨胀——`T.annotate_buffer_versions({x_ub: 2})` 类显式版本数计入）；③ 驻留可行 → 行块驻留单遍结构（GM 单遍读 + e 单次求值寄存器复用 + 原位覆写），online 递推降级为 N 超限时的泛化路径；④ 超限 → N-tile 流式 online（chunk 串行 + running (max,sum) flash 合并，引入重扫流量须估算对比）。
- 驻留变体内的容量/边界消化（block_m 类 wrapper 契约参数不可整除 M、不可内部改时）：行子批 SUB = min(block_m, UB 双版本约束, VRF 约束) 的最大 2 幂 + 尾块 TIR if（full/rem 静态分解）双机制可同时消化任意 block_m × 任意 M；SUB 封顶同时把 VRF 压到舒适水位（x_frag ≤ 128KB 时 ≈48 regs/thread 无 spill，样本见 CONST-vrf-capacity-a5）。
- 流式变体的 running stats 驻留：A5 SimtVF 域 per-chunk 开闭（域间有 MTE2）——stats 须驻留 UB（`stats_ub (SUB,2)` + 域内读写），不能照抄源 GPU 的跨循环 register fragment。
- 双任务实测样本：online_softmax [8192,8192] 驻留路径主 shape max_diff 1.9e-6、流式 [128,65536] 2.4e-7，双路径 63 用例全 PASS（含尾块 rem=1 与 ±inf/NaN/subnormal 边界）；logsumexp 双变体（single 驻留 / tiled 流式 flash 合并）fp16/bf16 输出与 `torch.logsumexp(x.float()).to(dtype)` 位级一致（max_diff=0），86+61 用例全 PASS（含 M%block_m≠0、M<block_m、非 pow2 N、尾 chunk extent<threads）。
- wrapper 驱动的变体分界（TileOPs logsumexp 实测口径，Ascend950PR_9579）：N_padded ≤ 32512（MAX_SINGLE_TILE_COLS）∧ block_m·elem·N_padded ≤ 216KB（`examples/TileOPs/tileops/device.py` UB_SIMTVF_BUDGET_BYTES 单一事实源）→ single 驻留变体，否则 tiled 流式变体（tile_n 由 wrapper 启发式给出、不保证 pow2，须 BLOCK_N pow2 重划——见 P-LAYOUT-TAIL-STATIC-EXTENT；effective_budget ∝ block_m 使小 bm 得大 tile_n，落点偏好与 single 相反，契约落点复算须逐 block_m 展开 wrapper 启发式）。
- **驻留变体 persistent 行块粒度：makespan-only 模型缺 per-tile 开销项（Stage 4 实测，2026-10-17）**：T.Persistent 56 核调度按 ceil(num_tiles/56) 波次分派（生成源码实证 8-tile 组 × 7 slot 形态）；纯 makespan 模型 min ceil(ceil(M/ROWS)/56)·ROWS 对 M=1024 选 ROWS=1（1024 tiles、makespan 19 行）实测 43.6µs，远慢于 ROWS=4（256 tiles、makespan 20 行）的 25.6µs（+41%）——每 tile 固定开销（lse 微型 MTE3 写出 flag 握手、vf_call 发射、MTE2 issue）随 tile 数线性放大。**修订规则（实测甜点）**：候选限 pow2 ROWS（保持 SUB=ROWS 单子批干净形态），取 makespan ≤ min+1 行内的最大 ROWS——M=1024→4（+50.9% vs baseline bm=16）、M=128→4（= baseline 配置 tie）、M=4→2；3d-multidim（M=128）对 ROWS 全配置不敏感（~19µs tie，瓶颈不在 M 维调度）。慢→快关键更改 = `_pick_rows` 规则替换（host 静态化参数重推导，kernel 体不变；repro/PATT-persistent-rows-overhead.py delta 骨架）。同轮实测佐证：BLOCK_N（流式变体 chunk 宽）双向偏离 16384 均回退（8192 −21.6~−36.4%、32768 −24~−29%）——「tile/chunk 粒度甜点」须实测扫描，不能只靠 makespan/摊薄模型外推。

---
id: P-LAYOUT-TAIL-STATIC-EXTENT
kind: pattern
family: [reduction, softmax, elementwise]
apis: [T.copy, T.Persistent, T.if_then_else, T.And]
dtype: [fp16, bf16, fp32]
device: A5/950 (dav-3510)
status: measured
origin_task: pilot-online_softmax-20261016T044814Z + logsumexp-harness（两函数 kernel；溯源：online_softmax.py `_build_streaming` num_full/rem 推导 + [128,32769] rem=1 用例；logsumexp `_logsumexp_kernel_{single,tiled}.py` 尾块/尾 chunk 路径 + 各自 RETROSPECTIVE Stage 3——允许失效）
toolchain: tilelang-fork dev build（fc32f47c 后）/ tilelang 0.1.15+cuda.git40c9d668 + CANN 9.2 + bisheng dav-3510 + Ascend950PR_9579，2026-10-16
repro: repro-missing
---

### 流式/分块 kernel 尾块处理范式：host 静态分解 + 元编程分支 + 静态 extent 切片

- host 侧算 `rem = N - (N // BLOCK_N) * BLOCK_N` 为 Python 常量 → 元编程 `if rem > 0:` 分支（parse 期决定，rmsnorm `if reuse_input:` 先例）→ 尾块 `T.copy` 两侧写显式静态 region（`X[base:base+BM, col:N]` ↔ `x_ub[0:BM, 0:rem]`）。依据：GM↔UB 的 T.copy 区域 extent 须为编译期常量，动态 extent（`T.min` 形式）落到非预期 DMA 搬运形态。
- 尾块 copy 目的切片形状必须匹配：`x_ub[0:rows, 0:N]`（源区域 (rows,N) 写入 (SUB,N) UB 的前 rows 行），形状错位编译期报错；online_softmax L142 `x_ub[0:BM, 0:rem]` 与 logsumexp 两 kernel 同构。
- 尾块 TIR if（`if t < full`）可直接写在 `T.Persistent` 循环体内（t 为循环变量、full 为 host 常量），无需 `T.macro` 包装，比源 GPU `with T.If/T.Then/T.Else` 旧语法简洁；`full/rem/tsub/tsub_rem/SUB` 全部 host JIT 常量才能让切片静态化。行子批串行循环用 `T.serial`（保持 x_ub 双版本流水顺序依赖），不要用 Python range 展开（TVMScript 把 range 也解析成串行循环而非 IR 构建期展开）。
- 行方向 M 尾块：`T.copy` 的 `pad_value=`/`data_select=` 是列向行尾 padding（补到 32B 边界），不能用于行向 OOB——用「静态 full/rem 分解 + fragment 行谓词」规避，不依赖未文档化的 DMA 行向越界行为；被丢弃行（r ≥ rows）内部链产生 NaN 但不写出，reduce_* 沿 dim=1 逐行独立不污染有效行。
- fragment 粒度重划（wrapper tile 参数非 pow2 时，如 tile_n=25600/16640）：BLOCK_N = 满足「BLOCK_N ≤ tile_n 且 2·BLOCK_N·elem ≤ UB 90% 折扣预算」的最大 2 幂，tile 参数降级为预算上限（签名不变、clamp 兜底）——与 K3 移除 GPU threads 参数同类：签名保留、语义按目标硬件约束重解释；尾 chunk TILE_R = next_pow2(rem) masked 载入，`T.if_then_else(T.And(r < rows, i < rem), cast, -inf)` 单一 masked 载入同时消化行尾+列尾（−inf 同时是 max/exp/sum 三幂元）。
- 实测：online_softmax [128,32769]（rem=1）全 PASS；logsumexp single L0 ④ (33,256) M 尾块、tiled L0 ⑤ (33,33000)（rem=232、TILE_R=256、extent<threads 复制布局 reduce——见 elementwise.md D-REDUCE-SIMTVF-API）、⑥ (33,32768, bm=4) M 尾块+子批双触发全 PASS；full chunk 无谓词直载 + 尾 chunk masked 二分支比源 GPU `_needs_mask` 快慢双路径更简洁且省 select。

**SimdVF 结构补充实测（TileOPs LogSumExpFwdOpSimd Stage 4，2026-10-17；origin_task: logsumexp-simd-harness；toolchain: tilelang dev fc32f47c 后 / CANN 9.2.0 + bisheng dav-3510 + Ascend950PR_9579；evidence: 该任务 perf_opt/opt_log.md Round 1-3 + perf_records.jsonl）**：

- **BLOCK_N 甜点（SimdVF 流式变体，lm-head (4,102400) fp16/bf16）**：`BLOCK_N = tile_n 直保留（25600）` 最优；甜点候选 16384 **−52%**（chunk 数 4→7，per-chunk stats 往返/sync 开销主导）、20480 +1.0%（tie）、51200（2 chunk）−6.9%；32k (1024,32768) 16384 vs 32768 tie（±0.1%）。**与 SimtVF 实测甜点 16384 方向相反——chunk 粒度甜点由每 chunk 固定开销与搬运效率的乘积决定，跨编程模式不可外推**（回应上文明示的「不可跨结构外推」缺口；DESIGN tiled §8.3 A/B 观察项闭环）。
- **流式变体 lse 微写批量化（ROWS）**：block_m=1 × M 行 → M 次 2 字节 UB→GM lse 写（msprof aiv_mte3 64.3% busy）。kernel 内 ROWS 行/tile（lse_ub (ROWS,) 单次写回）：M=1024 ROWS 2/4/8 → +16.4/+25.8/+26.6%（mte3 64.3→6.2%）。**策略：M ≤ 56 → ROWS=1（单波域保核并行）；否则 makespan ceil(ceil(M/ROWS)/56)·ROWS ≤ min+1 内最大 pow2、cap 8**（4 与 8 实测 tie 0.03%）。
- **均衡波调度反例（overlap > balance）**：理论 makespan −21% 的「满波 56 整数倍 + 小额补尾波」均衡调度（三分支 TIR if + 静态 extent）实测 **−23%**——异构分支破坏 copy/compute 双缓冲重叠（serialized 估算 12.4µs ≈ 实测 12.7µs）。**makespan-only 模型缺重叠项**：uniform batch + 双缓冲重叠 > 理论均衡（与上文「makespan-only 模型缺 per-tile 开销项」同为该模型失准族；SUB 下探 −60~−70% 为同族证据）。
