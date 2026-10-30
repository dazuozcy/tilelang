# _indexed_expert_gemm 算子设计文档（IndexedExpertMLPFwdOp 路由索引专家 GEMM kernel，TileOPs 迁移）

> **任务标识**：project = `indexed_expert_mlp`，op = `_indexed_expert_gemm`（`@tilelang.jit` 工厂函数名，保持不变），scenario = migration（TileOPs harness，family=moe）。上层算子 `IndexedExpertMLPFwdOp`（indexed/grouped 融合 MoE 专家 MLP 前向，DeepSeek-V3 / Qwen3-235B 规模）。整体算子编程模式 mixed；本函数是 **GEMM 主导 kernel（route 索引/分组专家 GEMM 模板）**，按用户确认指令以 **mixed**（Cube `T.gemm` 主路径 + Vector epilogue 融合）设计，`auto_schedule: yes`（默认）。
>
> **源码版本**：GPU 仓 `/home/l00970450/z00875089/TileOPs` @ `3979278e2da6ac4fa84d8c7c800dec64ea9c0425`（clean）。Stage 0 转换记录：`examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/conversion_record.md`；提取参考（逐字）：`examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/_indexed_expert_m_l_p_fwd_kernels.py`。前序函数 `_route_stats`（本批迁移，已完成 Stage 1–3）：`examples/indexed_expert_mlp/_route_stats/`——其 metadata 生产契约（counts 精确 / 每专家 rank 双射 / lists 配对 / rank<T 守卫 / 稳定序规范实现）是本设计的消费方输入。
>
> **修订标识**：revision_index = **1**（revision，2026-10-29；v0 备份 `history_version/design_v0.md`——Stage 2 REVIEW.md 不通过〔阻塞 ×1 + 建议 ×3〕后的**增量修订**，B0 选定不变）。
>
> **相对上一版（v0）的关键调整**（检视维度 8 阻塞 + 建议，逐条修复）：
> 1. **〔阻塞〕§1.6.0 R1 补评 C7 split-K 分块累加 + C8 Vector 规约累加**（候选库 ALG-gemv 命中行 R1/R2 必查项，v0 漏评）：R3/R4 补定量淘汰论证（R4 四条理由：瓶颈不变 / 并行度已足 / 接口冻结+精度契约 / epilogue 耦合；本地先例 `examples/ascend/example_gemm_splitk.py` 已亲读核对）。v0 L292「穷举结构空间」声明被推翻（枚举轴缺 K 维拆分）——本次枚举轴补「K 维拆分 + 计算单元归属」两轴、调研范围补 `example_gemm_splitk.py`，并在 §1.6.0 注明修订增量。
> 2. **〔建议〕D-2 no-op 行口径修正**：v0 结论「≈0.2–0.4M cyc（≈2–4%）」只匹配 gate/up 单实例 8,192 块，与供给列「8192+14336 块 / 28 核」（两实例合计 22,528 块 = 804.6 块/核 × 0.7–1.5K cyc）不自洽——修正为 **≈0.56–1.2M cyc（≈7–15%）**并给出可复现推导；§5.2 与调研结论依据 (3) 同步。
> 3. **〔建议〕verify_equiv.py E3c 修正为 NaN 金丝雀 off-grid 对偶**：v0 表描述「pad 行 NaN 金丝雀 / gated / off-grid」与脚本实参（`pad_fill=0.0`，与 E4a 同参）不符——取修复方向 (b)，脚本 E3c 改 `pad_fill=float("nan")`（E3a 的 off-grid 对偶：NaN pad 行 × off-grid 钳制/零填/掩码组合），**全部 12 项重跑**，12/12 EQUIV_PASS（RNG 消耗序列不变，其余项数字与 v0 逐位一致；E3c max_diff=0.000e+00）。
> 4. **〔建议〕字面修正（行号漂移家族全量归一 + API 名 + 容量数字）**：① 源行号 -14 漂移（提取参考文件 `_indexed_expert_m_l_p_fwd_kernels.py` 行号体系混入「源 L」引用——该文件 kernel body 恰比 GPU 源提前 14 行，已逐行对照证实）：REVIEW 标记的 OPT-1 L180/L195→**L194/L209**、§3.2 cast L183/L198→**L197/L212**，以及同家族另 6 处——OPT-2 L134–141/L185–192→**L148–155/L199–206**、OPT-3 L126–132/L142–150→**L140–146/L156–164**、§3.2 L97–113→**L111–127** / L111–113→**L125–127** / L102/L106→**L116/L120**、§3.3 L106–113→**L120–127**、§3.4 L97–99→**L111–113**、§9.4 L97–113→**L111–127**——全部归一为 GPU 源行号（与 §0.1/§0.3 一致）；② §2.3 wrapper API `device_smem_budget()`→**`get_device_backend().shared_memory_budget()`**（A5 wrapper L213 实调）；③ R4 C5 行「L1 峰值 160 KB」→**144 KB**（复算 (8+64)×2，与 §4.5/§3.5.1 一致）。
>
> **为何不会再犯同一错误**：R1 枚举轴集合已扩为六轴（分组 × M 粒度 × **K 维拆分** × **计算单元** × 数据通路 × 调度）并写入调研结论，后续修订按轴集合完整性自查（对照候选库命中行 R1/R2 全部条目 + `examples/ascend/` 文件清单，而非只核轴内候选）；D-2 表逐行「供给 × 单价 = 结论」复算纪律已落实（no-op 行给出逐步推导）；表-脚本一致性以脚本实参为唯一事实源，重跑后回填。

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 tilelang-op-design skill 的 `references/migration-analysis.md`）。解读对象为源文件 `src/tileops/_kernels/moe/indexed_expert_gemm.py`（GPU 仓）的 `_indexed_expert_gemm` 工厂（L66–216）、`_route_metadata_size` helper（L30–32）、`IndexedExpertGEMMTemplate` wrapper（L303–427）及其两个特化 `IndexedExpertGateUpKernel`（L504–536）/ `IndexedExpertDownKernel`（L539–566），host 侧调用点 `src/tileops/_ops/moe/routed_expert/indexed_routed_expert.py` 的 `IndexedExpertMLPFwdOp.forward`（L129–134），以及 A5 wrapper `examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/indexed_expert_mlp.py`（`IndexedExpertGEMMTemplate` L145–266，冻结契约）。全部计算语句已逐行归入 §0.3 步骤表。

### 0.1 源算子语义（做什么）

**数学语义**：route 索引的分组专家 GEMM 模板。给定路由表 `expert_ids[T, K]`（int32，调用方契约取值 ∈ [0, E)——manifest `requires: all(0 <= topk_ids[t,k] < E)`，kernel 不做值校验，与源一致），对每条 route `r = (t, s)` 取其专家 `e = expert_ids[t, s]`，计算

$$
y_{t,s,\cdot} = a_{\text{row}(t,s)} \cdot W_e^{\top},\qquad W_e = \text{weights}[e] \in \mathbb{R}^{N \times k}
$$

其中 `row(t,s)`：`route_input=False` 时为 `a[t, :]`（gate/up 实例，`a=[T,H]`）；`route_input=True` 时为 `a[t, s, :]`（down 实例，`a=[T,K,F]`，每 route 一行）。`N = weight_n = 2n`（gated，`activation="silu_and_mul"`）或 `n`（`activation="none"`）：

- **gate/up 实例**（`IndexedExpertGateUpKernel`）：`activation="silu_and_mul"`、`route_input=False`、`n=F`、`k=H`、`weights=[E,2F,H]`、`a=[T,H]`、`out=[T,K,F]`。gate/up 两半拼接为单个 GEMM：`W_e` 前 `F` 行为 gate 权重、后 `F` 行为 up 权重（与 golden `ref_routed_experts` 的 `gate_up.chunk(2, dim=-1)` 列序一致——`examples/TileOPs/workloads/moe.py` L50–52），epilogue 融合门控激活：
  $$
  \text{out}[t,s,j] = \text{silu}(y_{t,s,j}) \cdot y_{t,s,F+j},\quad j \in [0,F),\qquad \text{silu}(z) = z \cdot \sigma(z)
  $$
- **down 实例**（`IndexedExpertDownKernel`）：`activation="none"`、`route_input=True`、`n=H`、`k=F`、`weights=[E,H,F]`、`a=[T,K,F]`、`out=[T,K,H]`，epilogue 仅为直通（cast 到 `dtype`）。

**分派语义（dispatch_mode）**：`"grouped"` iff `T > 1`（上层分支谓词 `grouped_dispatch = num_tokens > 1`，conversion_record.md §3）。grouped 模式消费 `_route_stats` 产出的 `route_metadata`（三段契约：counts[E] 精确 / ranks[E+R] 每专家双射 / lists[E·T] 配对，`rank < T` 守卫），把同专家 route 按 **block_m = 16** 对齐分组，**leader 块计算整组**：`rank % 16 == 0` 的 route 是其组 leader；组行来自 `list[e·T + group_start + mi]`（`group_start = rank − rank % 16`，`route_count = min(16, min(counts[e], T) − group_start)`）。**非 leader 的 route（rank % 16 ≠ 0）对应块直接退出**；**rank ≥ T 的重复 route（duplicate-fallback）不进列表、独立计算自身单行**。direct 模式（T = 1）不读 metadata（wrapper 传入 `expert_ids.reshape(-1)[:1]` 占位 1-int 张量），每 route 独立计算单行。

**分组不变性（本迁移的核心语义结论，承接 `_route_stats` DESIGN §0.1 / E3）**：分组只改变「哪个 block 计算哪批行」，不改变每条 route 的数学内容——每行结果 = 该行激活向量与该 route 专家权重的 K 维点积，K 累加序由 K 循环（`ceildiv(k, block_k)` 分块）决定、与 route 落在 16 行组的哪个槽位无关。因此**任意自洽 metadata（任意双射）下最终输出逐位一致**；golden 无需消费 metadata（§8.1）。本设计 `verify_equiv.py` E2 对 stable 序与 random 双射两组 metadata 独立复核（7 组路由模式 × 4 shape 组合，全部一致）。

**规约语义**：GEMM K 维 fp32 累加（`acc` fragment/`float32`，源 L131）；K 分块顺序累加（`ceildiv(k, block_k)` 串行 k 步）；**无跨行规约**（行间独立）。silu 在 fp32 域施加、仅在写回时 cast 到 `dtype`（源 L194–197：`value = gate * T.sigmoid(gate) * acc_out[...]` 于 fp32 acc_out 上计算，`T.Cast(dtype, value)` 落存）。
**dtype 语义**：输入/输出 `dtype ∈ {fp16, bf16}`（manifest 契约，fp32 被 wrapper 拒绝）；累加与激活域 fp32。GEMM 权重布局 `[E, N, k]` = `[batch, N, K]`——**恰好是 A5 L1-input NT 约定的 `[N, K]` 逐专家堆叠**（tilelang-ascend skill §5.2），无需转置。
**边界语义**：(a) N/K 非整除 tile（off-grid，测试 `test_dims_off_the_tile_grid` H=100/F=96）：源以 `col < n` / `offset < k` 掩码 + A 侧零填（`offset ≥ k` 写 0）处理；**W 侧切片不加守卫**（源 L165–183 的 `T.copy` 区域可越界读——GPU 上靠 tensor pool 越界读不崩 + 掩码不可观察兜底；A5 上该行为不可移植，见 §0.5 条目 6 与 §0.6 R3）；(b) `mi ≥ route_count` 的 A 行源不写（SMEM 陈旧值），输出行被掩码；(c) 直接模式下仅 row 0 有效（行 1–15 陈旧，掩码）；(d) `expert_ids` 越界为未定义行为（调用方契约禁止，与源一致）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| a | (T, K_route, k) 或 (T, k) | fp16/bf16 | 激活输入；route_input=True（down）为 3-D 每 route 一行，False（gate/up）为 2-D 每 token 一行 |
| weights | (E, 2n 或 n, k) | fp16/bf16 | 逐专家权重 `[N, K]` 行主（NT 约定原生）；gated 时前 n 行 gate、后 n 行 up |
| expert_ids | (T, K_route) | int32 | 路由表（值 ∈ [0, E)，调用方契约） |
| route_metadata | (E+R+E·T,) grouped / (1,) direct | int32 | grouped：`_route_stats` 输出三段布局（counts/ranks/lists）；direct：占位（不读） |
| out | (T, K_route, n) | fp16/bf16 | **原位写回**（`out_idx=[]`，wrapper 返回传入 out）；输出 shape 依据：wrapper `forward` 校验 `(num_tokens, top_k, n)`（A5 wrapper L260–264，逐字保留源 L417–421） |

- 契约来源：A5 wrapper `IndexedExpertGEMMTemplate.forward`（`examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/indexed_expert_mlp.py` L224–266，校验逻辑逐字保留）；工厂签名 `_indexed_expert_gemm(num_tokens, top_k, num_experts, n, k, dtype, activation, route_input, dispatch_mode)` → `_kernel(block_n, block_k, num_stages)`（**threads 已按 K3/K9 移除**，config 三键保留）→ program `(a, weights, expert_ids, route_metadata, out)`——**接口冻结，不可改变**。
- 两个实例化（conversion_record.md §3）：gate/up `（n=F, k=H, silu_and_mul, route_input=False)`；down `（n=H, k=F, none, route_input=True)`；`dispatch_mode="grouped"` iff `T>1`。`R = T·K_route`。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码 L107–212 每个计算语句归入某步骤，无遗漏、无臆造；block_m = 16 硬编码）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义部分 |
|------|------|------|------|-------------|
| 1 | pid 分解：`route = pid // n_blocks; n_block = pid % n_blocks; token = route // K; slot = route % K`（L108–115） | pid | (route, n_block, token, slot) | 任务映射 |
| 2 | `expert = expert_ids[token, slot]`（L116，GM 标量读） | expert_ids | e | 路由读取 |
| 3 | grouped 首部：`rank = route_metadata[rank_base + route]`；`if rank < T: group_start = rank − rank%16; if rank ≠ group_start: is_leader=0; route_count = min(16, min(counts[e], T) − group_start)`（L117–127，GM 标量读 ×2） | metadata | leader/route_count/group_start | leader 判定与组界 |
| 4 | `if is_leader != 0:` 守卫全部计算（L134）；`T.clear(acc)`（L135） | — | acc 清零 | 累加器初始化 |
| 5 | K 流水：`for ko in T.Pipelined(ceildiv(k, block_k), num_stages)`（L136） | — | — | K 分块流 |
| 5a | A tile 装载：direct/单行路径 `a_shared[0, ki] = a[token, (slot,) offset]`（L138–146，逐元素 + `offset < k` 零填守卫）；grouped 路径 `for mi, ki in T.Parallel(16, block_k)`：`selected_route = route_metadata[list_base + e·T + group_start + mi]` → `selected_token/slot = div/mod` → `a_shared[mi, ki] = a[st, (ss,) offset]`，`mi < route_count` 与 `offset < k` 守卫、越界写 0（L148–164） | a (GM), metadata | a_shared[16, block_k] | 16 行 gather（组行/单行） |
| 5b | W tile 装载：`T.copy(weights[e, nb·bn:(nb+1)·bn, ko·bk:(ko+1)·bk], weight_shared[0:bn])` + gated 追加 up 半 `weights[e, n+nb·bn:...] → weight_shared[bn:2bn]`（L165–183，**切片无 OOB 守卫**，`eviction_policy="evict_first"`） | weights (GM) | weight_shared[(2·)bn, block_k] | 权重 tile 双段拷贝 |
| 5c | `T.sync_threads()`（L184）+ `T.gemm(a_shared, weight_shared, acc, transpose_B=True)`（L185，SMEM 输入 NT GEMM，跨 k 步累加） | a_shared, weight_shared | acc[16, (2·)bn] fp32 | MAC 主体 |
| 6 | `T.copy(acc, acc_out)`（L187，fragment→SMEM fp32） | acc | acc_out | L0 累加器落地 |
| 7 | epilogue：direct/单行 `for j in T.Parallel(bn)`：`col < n` 守卫，gated `value = gate·σ(gate)·up`（fp32），`out[token, slot, col] = Cast(dtype, value)`（L188–197）；grouped `for mi, j in T.Parallel(16, bn)`：重读 lists 得 (st, ss)，`mi < route_count and col < n` 守卫，`out[st, ss, col] = Cast(...)`（L198–212） | acc_out, metadata | out (GM) | 门控激活 + 掩码 scatter 写回 |

**数据流（源硬件 = CUDA SM 视角）**：

```
GM[expert_ids] --标量 load--> 寄存器（每 block 1 次）
GM[route_metadata] --标量 load（ranks/counts/lists 逐次）--> 寄存器
GM[a] --T.Parallel 逐元素 gather（组行/单行，div/mod 寻址）--> SMEM[a_shared 16×bk]
GM[weights] --T.copy 切片 DMA（gate/up 两段）--> SMEM[weight_shared (2bn)×bk]
SMEM×2 --T.gemm(wmma NT)--> fragment[acc 16×(2bn) fp32]（跨 ko 累加）
fragment --T.copy--> SMEM[acc_out]
GM[out] <--T.Parallel 掩码逐元素 scatter store（div/mod 行寻址 + col<n 掩码）--
```

**循环与并行结构**：grid = `num_routes × n_blocks` 个 block × `threads` 线程（default 128）；pid 二维分解（route-major：同 route 的 n_blocks 块相邻——A 行 L2 局部性）；K 维 `T.Pipelined(num_stages=2–3)` 流水（SMEM 双/三缓冲）；block_m = 16 固定（wmma 最小 M tile）；核内全部边界（T, K_route, E, n, k, bn, bk）为 `lru_cache` 工厂烘焙的编译期常量；`pass_configs={TL_DISABLE_WARP_SPECIALIZED: True}`（源 L94 注释：A tile 在 K 循环内重写，producer/consumer WS 会竞争）。

**host 侧逻辑**：`functools.lru_cache(maxsize=64)` 按 9 元组特化缓存工厂；`@tilelang.jit(out_idx=[], compile_flags=["-O3", "-DENABLE_BF16"])`；wrapper `default_config` 启发式：`block_n = 128 if route_input and (T==1 or T≥16) else 64`；`block_k = 128`，`num_stages = 3 if 1<T<16 else 2`；`route_input and (T==1 or (grouped and 4≤T≤8))` 时 `block_k=256, num_stages=2`；随后按 SMEM 预算（A5：UB 216 KiB via device backend，K6）递减 num_stages/block_k（L354–379 GPU / L197–222 A5）。Op 侧调用链：`route_stats → expert_gate_up(hidden, w_gate_up, ids, metadata) → expert_down(hidden_rows, w_down, ids, metadata) → weighted_reduce`（indexed_routed_expert.py L122–134）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | route 索引原地 gather（免 host permute） | 免 MoEPrePermute/PostPermute 两个额外 kernel 与 GM 往返 | 核内按 metadata lists gather A 行 | 无（算法层） | 可移植（保留） |
| 2 | 同专家 16 行分组 + leader 块整组计算 | 每 (专家, n_block) 权重 tile 只读一次（组内 16 route 摊薄）；非 leader 块立即退出省调度 | leader 判定（rank%16==0）+ route_count 组界 | 无（算法层；依赖 metadata 契约） | 可移植（保留） |
| 3 | gate/up 拼接单 GEMM（[2·bn, bk] 权重 tile 一次装载） | A tile 只读一次、一次 wmma 调用覆盖两半 | 权重 [2F, k] 前后半拼接 + 输出列 [0,bn)/[bn,2bn) 分读 | 无（算法层） | 可移植（保留） |
| 4 | silu_and_mul 融合进 GEMM epilogue | 免中间 [T,K,2F] 矩阵的 GM 写+读两遍 | fp32 acc 上直接施加激活、cast 后置 | 无（算法层） | 可移植（保留） |
| 5 | block_m=16 最小 M tile | 稀疏路由（每专家 route ≪ 16）下最小化 padding 浪费 | wmma M 维最小粒度 16 | GPU Tensor Core M 粒度 | 硬件相关（A5 Cube M 粒度同为 16，`CONST-gemm-dim-constraints`——恰好等价，保留） |
| 6 | `T.Pipelined(num_stages)` K 维软流水 + SMEM 多缓冲 | 隐藏 GM→SMEM 拷贝延迟 behind wmma | cp.async 式双/三缓冲 | GPU 异步拷贝 / wmma | 硬件强相关（A5 等价机制 = MTE2+AutoSchedule multibuffering，§0.5 条目 5） |
| 7 | SMEM 输入 `T.gemm(transpose_B=True)` NT wmma | Tensor Core 利用率 | ldmatrix/wmma NT | GPU Tensor Core | 硬件强相关（A5 L1-input NT 路径等价，§0.5 条目 4） |
| 8 | `eviction_policy="evict_first"` 权重流式读 | 权重单读不占 L2 | cache hint | GPU L2 | 硬件相关（A5 等价 `l2_cache_ctrl`，§0.5 条目 8） |
| 9 | `TL_DISABLE_WARP_SPECIALIZED: True` | A tile 在 K 循环内重写，避免 producer/consumer warp 竞争 | 关闭 warp specialization | GPU warp 特化流水 | 硬件强相关（A5 无 warp specialization 概念，§0.5 条目 9 舍弃） |
| 10 | route-major pid 排序 | 同 route 的 n_blocks 块相邻 → A 行 L2 复用 | grid 顺序 | GPU L2 | 硬件相关（A5 L2 128MB 同性质，保留 + §1.6.3 复核） |
| 11 | 每块标量 prologue 只读 2–3 个 GM 标量（expert_ids/rank/counts） | 非 leader 块最小退出代价 | 标量 GM 读 | GPU 标量 load | 可移植（保留；A5 上 lists 逐元素重读另见 §1.6.1 OPT-2） |

> 注：源平台该 kernel 为 **权重流量主导**（算术强度 = 2·M·N·K FLOP / (N·K·2B 权重 + M·K·2B A) ≈ 16–32 FLOP/B，远低于 Tensor Core 平衡点），手段 1–4 的意图都是**最小化权重 GM 流量**（每活跃专家权重恰读一次）＋减少中间量；手段 5–8 是让拷贝/wmma 流水不被打断。A5 迁移承接判断以 §1.6.0 R3/R4 定量为准。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定输入含 §1.6.0 调研结论——源算法与调研候选同台比较，选定算法族 = 基线结构保留 + A5 三核数据通路重设计，见 §1.6.0 调研结论。）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 1. 分组不变性语义 + metadata 三段消费契约（§0.1） | 语义 | 无 | — | 保留 | 语义层无条件保留；消费 `_route_stats`（本批已迁移）按同一契约（counts 精确/双射/配对/rank<T 守卫/稳定序）读取 |
| 2. route 索引原地 gather 免 permute（§0.4 #1） | 算法 | 无 | — | 保留 | 算法层直接保留（§1.4 步骤 3；`AscendMGroupedTileScheduler` 的 contiguous permute 布局是替代路线，R3 淘汰——接口契约不符） |
| 3. leader 分组 + 16 行组计算 + 非 leader 早退（§0.4 #2） | 算法 | 无 | — | 保留 | 逻辑逐字保留（§3.3 伪代码步骤 3）；block_m=16 恰为 A5 Cube M 粒度（`CONST-gemm-dim-constraints`：M 须 16 倍数，16 = 最小值） |
| 4. gate/up 拼接单 GEMM（§0.4 #3） | 算法 | 无 | — | 保留 | `[2F, k]` 权重前后半 + 输出列分读逐字保留；E6 int64 精确验证块恒等式 |
| 5. silu_and_mul epilogue 融合（§0.4 #4） | 算法 | 无 | 有：mixed CV epilogue 实证拓扑 | 保留（通路重设计） | `T.dual_copy`/guarded `T.copy` L0C→UB（fp32 落 UB，dual_copy 禁类型转换）→ SimtVF 逐元素 `g/(1+exp(−g))·u` + cast——pattern-library `D-MIXED-EPILOGUE-DUALCOPY`（measured）+ `CASE-pilot-gemm-silu`（`examples/pilot/gemm_silu/`，L0/L1/L2/Boundary 全 PASS） |
| 6. W 切片 `T.copy` 无 OOB 守卫（§0.1 边界 (a)） | 优化 | GPU tensor pool 越界读不崩 + 掩码不可观察 | **不可移植**：A5 `T.copy` GM→L1 走 ND2NZ 分形装载（行 16 整除、行 32B 对齐，`CONST-gemm-dim-constraints`），越界区域既违反对齐也无定义 | **重新设计** | → §0.6 R3：off-grid 尾区改为「钳制 in-bounds 拷贝 + 未写区零填」；on-grid（全部 manifest workload）走直拷快速路径。E4 + E4-probe 机器验证（零填承重：NaN 金丝雀泄漏 rel=inf） |
| 7. `T.Parallel` 两级并行 + `threads` launch 参数（§0.3） | 优化 | CUDA SIMT | 无：A5 `T.Kernel` 一维无 `threads=`（TypeError） | **重新设计** | → §0.6 R1：`T.MixedKernel(grid, sids=2)` 一维 grid + sid 分工（AIV0 = A 侧 gather 生产者，AIV1 = epilogue，AIC = 权重装载 + Cube）；`testing/ascend/auto_schedule/test_schedule_integration.py` L201–247 "fixed" 模式为逐形回归先例 |
| 8. SMEM 输入 NT `T.gemm`（§0.4 #7） | 优化 | GPU wmma + SMEM | 有：A5 L1-input 路径 **恰好同为 NT 约定**（`trans_A=False, trans_B=True`，权重 `[N, K]`——源权重布局原生兼容，零转换） | **等价替换** | `T.alloc_l1` 输入 + `T.alloc_l0c` fp32 累加器 + `T.gemm(a_l1, w_l1, res, transpose_B=True, clear_accum=(ko==0))`（tilelang-ascend skill §5.1；gemm_silu L94 同形） |
| 9. `T.Pipelined` K 软流水 + SMEM 多缓冲（§0.4 #6） | 优化 | GPU cp.async | 有：`T.Pipelined(num_stages)` + AutoSchedule 编译器托管 multibuffering（tilelang-ascend skill §7） | **等价替换** | K 循环保持 `T.Pipelined(k_blocks, num_stages)`（flash_attention `core.py` L388 先例：Pipelined 体内含 SimdVF Vector 段 + T.gemm 混合流水）；`TL_DISABLE_WARP_SPECIALIZED` 随 warp-specialization 概念一并**舍弃**（A5 无此机制；A tile 重写的版本管理由 AutoSchedule multibuffering 承接） |
| 10. `eviction_policy="evict_first"`（§0.4 #8） | 优化 | GPU L2 hint | 有：`T.copy(..., l2_cache_ctrl=...)`（tilelang-ascend skill §6.1，GM→L1 默认 NORMAL_FV） | **等价替换（降级为调优项）** | 主路径用默认策略（A5 L2 128MB 容纳 ~2 专家全权重〔ds-v3 58.7MB/专家〕，保不保留各有理）；`l2_cache_ctrl` 列入 Stage 4 调优面 |
| 11. `T.sync_threads()`（§0.3 步骤 5c） | 优化 | CUDA block barrier | 不需要：跨 pipe/跨核依赖由 AutoSchedule 自动插 flag | **舍弃（被结构消除）** | auto_schedule: yes（§7）；a_l1/w_l1→gemm 的 RAW 由 AssignCore/InsertSync pass 承接 |
| 12. `T.clear(acc)` + 跨步累加 | 优化 | 无 | 有：`T.gemm(..., clear_accum=(ko==0))` | **等价替换** | 首步清零、后续累加（gemm_silu L94 同形；语义等价——E2 覆盖） |
| 13. host：lru_cache 9 元组特化 + wrapper 校验 + `out_idx=[]` + default_config 启发式（含 UB 预算 fit 循环） | host | 无（预算查询已 K6 适配） | — | 保留 | A5 wrapper 已脚手架化（L145–266）；`block_n/block_k/num_stages` 三键调优面保留，threads 移除（K3/K9） |
| 14. route-major pid 排序（§0.4 #10） | 优化 | GPU L2 | A5 L2 128MB（`CONST-arch-a5` 实测注记）同性质 | 保留 | pid 分解顺序逐字保留（§3.3 步骤 1）；§1.6.3 复核 |

> 每项处置依据：tilelang-ascend skill §5/§6/§7 + `CONST-gemm-dim-constraints`（条目 3/6/8）、pattern-library `D-MIXED-EPILOGUE-DUALCOPY`（条目 5）、`testing/ascend/auto_schedule/test_schedule_integration.py`（条目 7）、gemm_silu/flash_attention 先例（条目 5/9）。舍弃项（9 后半/11）理由已写明；无静默丢弃——§0.4 全部 11 项手段的意图承接见 §1.6.0 调研结论末表。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（判定输入含 §1.6.0 调研结论：选定算法族 = **基线分组结构保留 + A5 三核数据通路重设计**——MTE2 权重直装 L1 / AIV gather 生产 A / Cube NT GEMM / AIV epilogue，源算法的算法层结构〔分组/拼接/融合/最小 M tile〕全部保留）：

**重设计项 R1: block×thread 两级并行 → `T.MixedKernel(sids=2)` 三核角色分工（sid guard 划分 + 拷贝所有权）**

- **源方案**：`T.Kernel(num_routes*n_blocks, threads=threads)` 二维 pid 分解 + 128 线程 `T.Parallel` 逐元素 gather/scatter；producer/consumer 经 `T.sync_threads` 同步。
- **NPU 新算法**（一维 grid + 三核角色，依据 core-split-strategy.md §1 要素 ② 与 tilelang-ascend references/synchronization.md「Independent AIV execution」）：
  1. `with T.MixedKernel(num_routes * n_blocks, sids=2) as (bx, sid):`——一维 grid，block 数与源相同（route-major pid 分解逐字保留）。
  2. **公共标量 prologue**（三核同构执行，各读各的 GM 标量）：pid 分解 → `expert = expert_ids[t, s]` → grouped 首部（rank/leader/route_count/group_start，2 次 GM 标量读）。`if is_leader != 0:` 守卫后续全部工作（**子核一致**的运行时守卫——三核对同一数据求值相同，满足 hardware dual_copy「control flow uniform across sub-cores」约束与独立 AIV 退出规则；is_leader 为 block 级循环不变量）。
  3. **sid guard 划分**：`if sid == 0:` 拥有 **A 侧生产**（行元数据 staging → SimtVF gather GM→a_ub → `T.copy(a_ub, a_l1)` UB→L1）；`if sid == 1:` 拥有 **epilogue**（`T.copy(res, acc_ub)` L0C→UB → SimtVF silu/cast → 掩码 scatter store）；**公共 Cube 段**（W `T.copy` GM→L1 + `T.gemm`）置于 sid guard 之外（「Keep common Cube work outside sid guards」）。AIV1 在 K 循环向量阶段空闲（gather 墙钟时间被 AIC 权重 MTE2 完全掩盖，§1.6.0 D-2：gather ≈150 cyc vs W 拷贝 ≈330 cyc/k-step；两 AIV 分摊 gather 的墙钟收益 ≈ 0，仅省 AIV1 空闲周期）。
  4. **拷贝/缓冲所有权**：`a_ub`（AIV0 私有 UB；`T.alloc_shared` 在 mixed kernel 内为 per-AIV 私有——synchronization.md「UB and sub-core-dependent scalar state are local to each AIV; L1, L0, and GM storage remain common」）；`a_l1/w_l1/res` 为 block 级公共（L1/L0C）；`acc_ub`（AIV1 私有）。L0C→UB 用 **guarded `T.copy`**（「implicit T.copy routing targets the participating AIVs」——guard 把 Fixpipe 路由投影到 AIV1；deepgemm epilogue.py L279–286 同机制先例）；`T.dual_copy` M-split 备选形态（fallback A，§1.6.3 候选 2，E5 已验证其行映射）。
- **语义保持论证**：并行组织是纯实现层（migration-analysis.md §5.3「block × thread 两级并行 → 重新设计」）；三核分工不改变任何计算内容——gather 行集、GEMM 累加序（K 分块串行）、epilogue 掩码与 scatter 目标均与源逐项相同（§3.3 伪代码对照表）；E2（分组算法模拟 vs per-route golden，stable/random 双射 × 7 路由模式 × 4 shape）+ E5（epilogue 半分/整分行映射一致性）机器验证。源 `T.sync_threads`（条目 11）的意图（A/W 装载与 wmma 的 SMEM 竞争屏障）由 AutoSchedule 对 a_l1/w_l1→gemm 的版本化 RAW 同步承接（auto_schedule: yes）。

**重设计项 R2: SMEM 输入 wmma + threads 并行 gather/scatter → L1/L0C Cube 路径 + VF 域向量化（数据通路整体迁移）**

- **源方案**：`T.alloc_shared` SMEM tile（a_shared/weight_shared/acc_out）+ `T.alloc_fragment` acc + SMEM 输入 `T.gemm`；gather/scatter 为 128 线程逐元素。
- **NPU 新算法**（AGENTS.md Cube 路径基线 + gemm_silu 实测拓扑）：
  1. `a_l1 = T.alloc_l1((16, block_k), dtype)`、`w_l1 = T.alloc_l1((N_gemm, block_k), dtype)`（N_gemm = 2·block_n gated / block_n）、`res = T.alloc_l0c((16, N_gemm), "float32")`。
  2. W 装载（on-grid）：`T.copy(weights[expert, nb·bn:(nb+1)·bn, ko·bk:(ko+1)·bk], w_l1[0:bn, 0:bk])`（+ gated up 半段）——**GM→L1 直装，ND2NZ 分形转换由 InsertNd2Nz pass 自动插入**（AGENTS.md「never hand-write NZ staging」）；运行时首轴/行基址（expert、n_block 派生）有 deepgemm `matrix_region` 先例（`examples/ascend/deepgemm/kernels/copy.py` L13–27 + `common.py` L34–38：runtime mn_idx/batch_idx 的 GM→L1 拷贝）。
  3. A 装载：`with T.SimtVF(threads=128): for mi, ki in T.Parallel(16, block_k):` 掩码 gather → `a_ub`；`T.copy(a_ub, a_l1)`（UB→L1，MTE3 256 B/cycle，`CONST-mte-bandwidth-model`）。gather 必须经 UB（VF 写 UB，无 L1 直达向量通路）；逐元素形态保留源的行级 gather 语义（route_input 2-D/3-D 寻址、`mi < route_count`、`offset < k` 零填）。
  4. `T.gemm(a_l1, w_l1, res, transpose_B=True, clear_accum=(ko == 0))`——L1-input NT 路径（`tilelang/ascend/op/gemm/gemm_mad.py` L162 强制 NT；源 `transpose_B=True` 语义直译；K=block_k 须 C0 整除——block_k ∈ {64,128,256} 均满足，`_compute_tile_k_sub` L206–222 自动 K 子切分）。
  5. epilogue：`T.copy(res, acc_ub)`（Fixpipe L0C→UB，fp32 无转换）→ `with T.SimtVF(threads=128): for mi, j in T.Parallel(16, block_n):` 掩码 silu_and_mul（直除式）/cast + 逐元素 scatter store。
- **语义保持论证**：数学内容逐项同源（NT 转置约定同形、fp32 累加、K 分块序、掩码集相同）；差异仅在数据通路（SMEM→L1/L0C/UB 与 VF 域）。E2 机器验证（模拟含 A5 尾区契约）；精度路径与源一致（fp32 累加 + fp32 域激活 + 落存 cast——同时是 `D-MIXED-EPILOGUE-DUALCOPY` 记录的精度最优路径：消除 z 的 16-bit 预舍入）。源手段 6（K 软流水）由 `T.Pipelined(num_stages)` + 编译器托管 multibuffering 等价承接（条目 9）；手段 8（evict_first）降级为 `l2_cache_ctrl` 调优项（条目 10）。

**重设计项 R3: W 切片无守卫越界拷贝 → on-grid 直拷 / off-grid 钳制拷贝 + 零填（尾区契约）**

- **源方案**：`T.copy(weights[e, rows, cols], weight_shared[...])` 切片可越界（N/K 尾区：up 半段行基址 `n + nb·bn` 可超 `weight_n`；K 列可超 `k`）——GPU 靠 pool 越界读不崩 + `col < n` 掩码不可观察；A 侧尾区零填（`offset < k` else 写 0）。
- **NPU 新算法**（工厂静态双路径——`(n, k, block_n, block_k)` 全为 `lru_cache` 编译期常量，尾区性静态判定）：
  1. **on-grid**（`n % block_n == 0 and k % block_k == 0`——4 条 manifest workload 全部命中；测试 `test_manifest_shaped_call` 与 small-route 用例亦命中）：W 直拷 GM→L1（全静态 extent，无守卫、无零填——零开销快速路径）；A gather 的 `offset < k` 守卫与 epilogue 的 `col < n` 掩码被 Python 级静态消除。
  2. **off-grid**（`test_dims_off_the_tile_grid` H=100/F=96 等测试形态）：W 经 UB staging——`T.fill(w_ub, 0)`（静态整 tile 零填，SimdVF 域）→ `T.copy(weights[e, r0 : r0+rb, kc : kc+kb], w_ub[0:rb, 0:kb])`（**钳制 in-bounds 拷贝**：`rb = clamp(weight_n − r0, 0, block_n)`、`kb = min(k − kc, block_k)` 为运行时表达式〔block 内循环不变量，n_block 固定〕；GM→UB 字节粒度行支持任意 rb/kb——tilelang-ascend skill §6.1「byte-granular rows with independent source/destination pitches」，无 ND2NZ 的 16 整除/32B 对齐约束）→ `T.copy(w_ub, w_l1)`（整 tile UB→L1，行 16 整除 ✓ 行 32B 对齐 ✓）；gated up 半段 `rb ≤ 0` 时运行时守卫跳过拷贝。K 尾（`kb < block_k`）：W 尾列零填 + A 尾列零填（gather else 分支）→ GEMM 仍以 K=block_k（C0 整除）计算，尾列贡献恰为 0。A gather 与 epilogue 掩码按源保留（运行时）。
- **语义保持论证**：(a) in-bounds 部分（`[0, rb) × [0, kb)`）与源读取的有效数据逐元素相同；(b) 源越界读产生的垃圾只落在 `col ≥ n`（掩码不可观察）或与 A 零列相乘（贡献 0）——A5 零填产生确定的 0，同属不可观察/零贡献，但消除未定义行为与 `0 × NaN = NaN` 污染有效列的风险；(c) `mi ≥ route_count` 行（源 SMEM 陈旧值）→ A5 零填（确定性合约，输出行掩码不变）。**E3**（NaN 金丝雀于 pad 行：7 模式 × 3 shape 组合输出逐位不变——掩码充分性）+ **E4/E4-probe**（零填下 off-grid 全对；**未零填的 NaN 金丝雀泄漏 rel=inf——零填是承重结构而非装饰**）机器验证。off-grid 仅测试形态使用（4/4 manifest 行 + 5/7 移植测试 on-grid），性能面以 on-grid 快速路径为准。

（§0.5 无其他「重新设计」条目；条目 8/9/10/12 的等价替换已并入 R2，条目 11 的屏障随 R1 结构消除。）

### 0.7 标杆实现

- **源算子定义**（GPU 仓，绝对路径）：`/home/l00970450/z00875089/TileOPs/src/tileops/_kernels/moe/indexed_expert_gemm.py`（`_indexed_expert_gemm` L66–216、`_route_metadata_size` L30–32、`IndexedExpertGEMMTemplate` L303–427、`IndexedExpertGateUpKernel` L504–536、`IndexedExpertDownKernel` L539–566）；host 调用点 `/home/l00970450/z00875089/TileOPs/src/tileops/_ops/moe/routed_expert/indexed_routed_expert.py`（`IndexedExpertMLPFwdOp.forward` L82–134）。
- **提取参考**（逐字，A5 仓）：`examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/_indexed_expert_m_l_p_fwd_kernels.py`。
- **A5 wrapper（冻结契约）**：`examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/indexed_expert_mlp.py`（`IndexedExpertGEMMTemplate` L145–266；调用面 `self.kernel(**{"block_n","block_k","num_stages"})(a, weights, expert_ids, route_metadata, out)`）。
- **golden 依据**：op 级参考 `ref_routed_experts`（`examples/TileOPs/workloads/moe.py` L58–91，移植自源仓，fp32 逐专家 gathered matmul + fp32 silu_and_mul + fp32 加权累加）；kernel 级 golden 由本设计按 §0.1 语义给出（per-route 独立计算，分组不变性使其不消费 metadata——E2 机器验证，§8.1）。
- **A5 结构先例**：`examples/pilot/gemm_silu/gemm_silu.py`（mixed GEMM+silu 实测案例，L0/L1/L2/Boundary 全 PASS）；`examples/ascend/deepgemm/`（m-grouped MoE GEMM、动态区域拷贝、sid guard epilogue）；`examples/ascend/flash_attention/core.py`（Pipelined 体内 SimdVF+gemm 混合流水）；`testing/ascend/auto_schedule/test_schedule_integration.py` L196–249（"fixed" 模式 = 本设计主选结构逐形回归）。
- **规格**：`examples/TileOPs/src/tileops/manifest/spec/moe.yaml` L30–61（IndexedExpertMLPFwdOp workloads：ds-v3-t1/t32/t64、qwen3-235b-t32 × {fp16, bf16}）；测试 `examples/TileOPs/tests/ops/test_indexed_expert_mlp.py`；bench `examples/TileOPs/benchmarks/ops/bench_indexed_expert_mlp.py`。

## 1. 概述

### 1.1 算子名称

- **工厂函数**：`_indexed_expert_gemm`（`@tilelang.jit(target="ascend")`，名称保持不变，迁移契约）
- **算子族**：moe / route 索引分组专家 GEMM（GEMM 主导 + 门控激活 epilogue）
- **上层算子**：`IndexedExpertMLPFwdOp`（indexed expert MLP 前向，`IndexedExpertGateUpKernel` + `IndexedExpertDownKernel` 两个特化）
- **来源**：TileOPs GPU 仓 `src/tileops/_kernels/moe/indexed_expert_gemm.py`（迁移，见 §0）

### 1.2 功能描述

对每条 route `(t, s)`（token t 的第 s 个专家选择），取出其专家权重 `weights[e]`，计算该 route 激活向量与 `weights[e]^T` 的 GEMM，并（gated 实例）在 epilogue 施加 `silu(gate)·up` 门控融合、cast 后按 route 散写回 `out[t, s, :]`。grouped 模式下消费 `_route_stats` 的路由 metadata，把同专家 route 按 16 行组对齐分组、leader 块整组计算（每 (专家, n_block) 权重 tile 只读一次）；direct 模式（T=1）逐 route 独立计算。两个实例化：gate/up（`n=F, k=H, silu_and_mul, route_input=False`）与 down（`n=H, k=F, none, route_input=True`）；`dispatch_mode="grouped"` iff `T>1`。

### 1.3 数学公式

逐 route 语义（`e = expert_ids[t,s]`，`R = T·K_route`）：

$$
y^{(t,s)} = a_{\text{row}(t,s)} \cdot W_e^{\top} \in \mathbb{R}^{N},\qquad W_e = \text{weights}[e] \in \mathbb{R}^{N \times k}
$$

- `row(t,s)`：`route_input=False` → `a[t,:]`（2-D `[T,k]`）；`route_input=True` → `a[t,s,:]`（3-D `[T,K_route,k]`）
- `N = 2n`（gated，`silu_and_mul`）或 `n`（`none`）

输出（gated 实例，`j ∈ [0,n)`）：

$$
\text{out}[t,s,j] = \text{Cast}\Big(\underbrace{\text{silu}\big(y^{(t,s)}_j\big)}_{\text{gate}} \cdot \underbrace{y^{(t,s)}_{n+j}}_{\text{up}}\Big),\qquad \text{silu}(z) = \frac{z}{1+e^{-z}}
$$

输出（`none` 实例）：`out[t,s,j] = Cast(y_j)`。K 维 fp32 累加（K 分块串行），激活在 fp32 域施加、落存时 cast 到 `dtype`（fp16/bf16）。分组不变性：任意自洽 metadata（任意双射）下逐位一致（§0.1，E2 机器验证）。

### 1.4 算法描述（NPU 侧，§0.6 重设计后）

1. **grid**：`T.MixedKernel(R × n_blocks, sids=2) as (pid, sid)`，一维 grid；`n_blocks = ceildiv(n, block_n)`（gated 时按半宽计：每块覆盖 gate 列 `[nb·bn, (nb+1)·bn)` 与 up 列 `[n+nb·bn, ...)`，N tile 宽 = `2·bn`）。route-major pid 分解：`route = pid // n_blocks; nb = pid % n_blocks; token = route // K; slot = route % K`。
2. **公共标量 prologue**（三核同构）：`expert = expert_ids[token, slot]`；grouped 时 `rank = route_metadata[E + route]`，`if rank < T: group_start = rank − rank%16; is_leader = (rank == group_start); route_count = min(16, min(counts[expert], T) − group_start)`（`rank ≥ T` 重复 route：is_leader=1、route_count=1、独立计算自身行）。
3. **`if is_leader != 0:` 守卫全部后续工作**（子核一致运行时守卫）。
4. **sid==0（AIV0，A 侧生产）**：staging——`use_lists = (grouped and rank < T)` 为真时 `for mi in T.serial(16): if mi < route_count: sel = route_metadata[E+R+e·T+group_start+mi]; st_ub[mi] = sel // K; ss_ub[mi] = sel % K`；否则 `st_ub[0]=token; ss_ub[0]=slot`（route_count=1）。随后 K 流水内逐 k 步：SimtVF `for mi in T.serial(16): for ki in T.Parallel(block_k):` 掩码 gather（`mi < route_count` 与 `offset < k` 越界写 0，2-D/3-D 寻址同源）→ `a_ub`；`T.copy(a_ub, a_l1)` UB→L1。
5. **公共 Cube 段（AIC，sid guard 之外）**：逐 k 步 W 装载——on-grid：`T.copy(weights[expert, rows, cols], w_l1[seg])` GM→L1 直装（ND2NZ 自动插入，gated 双段：gate 半 `[nb·bn,(nb+1)·bn)` + up 半 `[n+nb·bn, ...)`）；off-grid：`T.fill(w_ub, 0)` → 钳制 GM→UB 拷贝（`rb/kb` 运行时表达式）→ `T.copy(w_ub, w_l1)`。
6. **公共 Cube 段**：`T.gemm(a_l1, w_l1, res, transpose_B=True, clear_accum=(ko==0))`——L1-input NT 路径，K=block_k（C0 整除），fp32 L0C 累加。
7. **sid==1（AIV1，epilogue）**：`T.copy(res, acc_ub)` L0C→UB（fp32）；SimtVF `for mi in T.serial(16): for j in T.Parallel(block_n):` 掩码（`mi < route_count and col < n`）silu（直除式）/cast，按 `out[st_ub'[mi], ss_ub'[mi], col]` 散写（AIV1 自有 staging 副本 st_ub'/ss_ub'——UB 为 per-AIV 私有）。
8. **K 循环**：`for ko in T.Pipelined(k_blocks, num_stages)`，步骤 4 的 gather/copy 与步骤 5 的 W copy/gemm 均在循环体内（mixed 流水，"fixed" 形态）。
9. **工厂静态双路径**：`(n, k, block_n, block_k)` 全为编译期常量，`on_grid = (n % block_n == 0 and k % block_k == 0)` Python 级静态判定——on-grid 消除 `offset < k` / `col < n` 运行时掩码与 W 尾区处理（零开销直拷）；off-grid 保留运行时掩码 + 钳制拷贝 + 零填。`route_count` 相关掩码恒为运行时（metadata 依赖）。
10. **direct 模式**：Python 工厂级 `if dispatch_mode == "direct"` 直接不发射 metadata 代码（staging else 分支、无 rank/counts 读）；`route_metadata` 为占位 (1,) 张量，不读。

### 1.5 数据流图（A5 三核视角）

```
GM[expert_ids]  ──标量 load（公共 prologue，三核各读）────────────► 寄存器（e）
GM[route_metadata] ──标量 load×2（rank/counts）──────────────────► 寄存器（leader/route_count）
GM[route_metadata] ──16×守卫标量 load + div/mod（每 AIV 私有 staging）─► UB[st_ub/ss_ub]（AIV0/AIV1 各一份）
GM[a]           ──SimtVF 掩码 gather（行散集，AIV0 MTE2+VALU）──► UB[a_ub 16×bk]        ──T.copy(MTE3 256B/cyc)──► L1[a_l1]
GM[weights]     ──T.copy（AIC MTE2 100B/cyc，ND2NZ 自动分形）────► L1[w_l1 (2·)bn×bk]     （off-grid: 经 UB[w_ub] 钳制+零填）
L1[a_l1] × L1[w_l1] ──T.gemm(NT, fp32 累加)──► L0C[res 16×(2·)bn fp32]
L0C[res]        ──T.copy(Fixpipe，guard: sid==1 路由到 AIV1）──► UB[acc_ub fp32]
UB[acc_ub]      ──SimtVF silu(直除)+cast+掩码散写（AIV1）──────► GM[out[t,s,col]]
```

- 依赖边（AutoSchedule 自动插 flag，§7）：`a_ub→a_l1→gemm`（RAW）、`w_l1→gemm`（RAW）、`res→acc_ub`（RAW，Fixpipe）、`staging→gather/epilogue`（RAW）。
- 瓶颈资源：W 的 AIC MTE2（GM→L1）装载，≈22.5 GB / 8.05M cycles（ds-v3-t64 两实例合计，§1.6.0 D-2）；AIV0/AIV1 全部工作 ≪ 瓶颈周期，被 Cube 流水完全掩盖。

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：**同一数学语义（route 索引分组专家 GEMM + 可选门控 epilogue）的算法族**；源算法（§0.3）只是基线候选之一。四问：R1 等价化简公式候选、R2 在线算法、R3 复杂度四口径、R4 硬件亲和。执行标准见 skill `references/algorithm-research.md`。

**R1 等价化简公式 / 结构候选**

| 候选 | 形态 | 与基线关系 | 来源 |
|------|------|-----------|------|
| **B0 基线（源算法）** | 分组 leader + 16 行组 + 核内 gather/scatter + gate/up 拼接单 GEMM + 融合 silu epilogue | — | GPU 源码（§0.3） |
| C1 direct-dispatch-all | 取消分组：每 route 独立 16 行 tile（1 有效行），不消费 metadata | 结构特化（route_count≡1） | 极简形态（源 direct 分支的一般化） |
| C2 staged permute + contiguous m-grouped | 先把激活按专家连续化排列（额外 permute kernel），再跑 contiguous m-grouped GEMM | 数学等价、布局重排 | `examples/ascend/deepgemm/`（m-grouped MoE GEMM） |
| C3 grid-restructure（grid=组数，n 维块内串行，res 全 N 常驻 L0C） | 每 (专家组) 一个 block，块内串行 n_blocks，输出整行 [16, 2n] fp32 常驻 L0C | 结构重排（省 block 数） | 自提 |
| C4 block_m=32 | 32 行组（其余同基线） | 参数化变体 | 源参数空间 |
| C5 L0-input GEMM | 权重经 L0B 直入 Cube（绕开 L1-input 路径） | 数据通路变体 | `CONST-gemm-dim-constraints`「L0-input 有独立布局处理」 |
| C6 persistent + `AscendMGroupedTileScheduler` | 常驻核 + 调度器产 contiguous m-tile 索引 | 结构重排 | `examples/ascend/deepgemm/` 调度形态 |
| C7 split-K 分块累加 | K 维按 split_k 拆分跨核并行部分累加 + 跨核归并（atomic add 或 deterministic ordered reduction——`T.set_atomic`/`T.PerCoreTask` + `T.ascend_sync_inter_arrive/wait`） | 结构变体（K 维并行化；输出 tile 由 split_k 块共算） | `examples/ascend/example_gemm_splitk.py`（atomic/ordered 双模式实测先例，已亲读核对）+ `examples/ascend/test_gemm_splitk.py`；候选库 ALG-gemv 行 R2「split-K 分块累加」（命中行必查项） |
| C8 Vector 规约累加 | GEMM 经 VALU lane-op 点积（每输出元素 = A 行 × W 列逐元素乘加归约，不经 Cube 分形路径） | 计算单元替换 | 候选库 ALG-gemv 行 R1「GEMM 分形路径（Cube）vs Vector 规约累加」（命中行必查项）；`CONST-cube-throughput-model` vs `CONST-valu-model` 对比口径 |
| 公式级：silu 形式 | 直除式 `z/(1+e^{-z})` vs `z·σ(z)` vs tanh 恒等式 | 代数恒等 | §1.6.1 OPT-1（探针裁决） |

公式级化简空间调研结论：GEMM 本体已是点积最简形（无更低复杂度代数形式——两个 `[M,K]×[N,K]` 矩阵乘无结构可利用：权重稠密无稀疏性、K 维无条件结构）；唯一公式级自由度是 silu 的代数形式（→ §1.6.1 OPT-1 以实测探针裁决）与分组/布局/并行的结构等价变体（上表 C1–C8，R3/R4 定量淘汰）。

**R2 在线算法：无在线变体（明确结论）**

结构性依据：(a) 本算子唯一的规约是 K 维点积累加，其在线（流式单遍）形态**就是** GEMM 的 fp32 分块累加本身（状态 = 1 个 `[16, N]` fp32 累加器，不存在更小的充分统计量——点积的任意部分和都不可替代完整和）；(b) `silu_and_mul` 耦合**同一 GEMM 输出的两半列**（gate 列 j 与 up 列 n+j），任何增量/部分状态形式都需要完整的两半向量才能施加，状态无法低于完整行；(c) 行间独立、无跨行规约（无 softmax/归一化类耦合，不需要 online-softmax/Welford 类技术）。在线算法（单遍流式聚合）适用前提「规约中间统计量 ≪ 完整数据」在本算子全部三个维度上不成立 → **无在线算法变体**（调研范围：online softmax/Welford/流式前缀类算法族的适用条件，见 `references/algorithm-research.md` 参考表）。注：候选库 ALG-gemv 行 R2 所列「split-K 分块累加」属 **K 维并行化结构候选**（部分和跨核归并）而非在线统计量算法——已作 **C7** 入 R1/R3/R4 评估并淘汰（见 R4 ①–④），命中行 R2 覆盖完毕；其「部分和不可先行施加 epilogue」的特性与本条结论 (b) 同源。

**R3 复杂度四口径对比**（ds-v3-t64：T=64, K=8, R=512, E=256, H=7168, F=2048，均衡路由 active_E=256；两实例合计；口径 = FLOPs / 访存 / 扫描遍数 / 中间缓冲）

| 候选 | FLOPs（padded） | GM 访存量 | 扫描遍数 | GM 中间缓冲 |
|------|----------------|-----------|----------|-------------|
| **B0 基线** | **361 GFLOP**（gate/up 240 + down 120） | **W 22.5 GB**（每活跃专家权重恰读 1 次）+ A 首读 9.4 MB（L2 常驻后核侧重读 2.8 GB，L2 命中）+ out 写 9.4 MB | W 1×；A 逻辑 n_blocks×（物理经 L2） | **0**（epilogue 融合，无 [T,K,2F] 中间矩阵） |
| C1 direct-all | 722 GFLOP（**2×**：512 route 块 vs 256 组块） | W 45 GB（**2×**：每 (route,n_block) 重读权重 tile） | W 2× | 0 |
| C2 permute + contiguous | ≈361 GFLOP×(128/16 稀疏放大 ≤8×)——block_m=128 下每专家 2 行 pad 到 128 行 | W 22.5 GB（持平）+ permute/un-permute 额外 ≈38 MB（A 2 遍 + out 2 遍，≪W） | W 1×；A +2 遍 | +2（permuted A、permuted out） |
| C3 grid-restructure | fallback 形态 361 GFLOP（持平） | 同基线 | 同基线 | 0；**常驻形态不可行**：res[16,2F] fp32 = 16×4096×4B = 256 KB = L0C 全容量（`CONST-capacity-a5`），无 multibuffering 余量 |
| C4 block_m=32 | 722 GFLOP（**2×**：c_e≤16 的全部 manifest 形态（c_e≤2）下 32 行组仍 1 组/专家，行 padding 16→32） | W 22.5 GB（持平）+ A 5.6 GB（**2×**） | 同基线 | 0 |
| C5 L0-input | 361 GFLOP（持平） | 同基线（通路差异在 L1 以下） | 同 | 0 |
| C6 persistent 调度器 | 361 GFLOP（持平） | W 22.5 GB + 调度器索引（若 host 预生成） | 同 | +索引张量 |
| C7 split-K | 361 GFLOP（持平：各 split 计算 disjoint K 段） | W 22.5 GB **持平**（各 split 读 disjoint K 段，W 每元素仍恰读一次）+ **新增** partial 部分和 GM 往返（split_k × grid × [16, N_gemm] fp32 写+读） | W 1×；partial **+2 遍**（落存 + 归并回读） | **+GM workspace**（gated：[T,K,2n] fp32——partial 含 gate/up 两半列，out [·,n] 列宽减半放不下）；或 out 原子多写（见 R4 ③） |
| C8 Vector 规约 | **361 G lane-ops**（同 FLOPs 由 VALU 执行）→ **≈126M cyc**（56 AIV × 51 lane-ops/cyc）= Cube 路径 1.57M cyc 的 **80×**、W 瓶颈 8.05M cyc 的 **15.7×** | 同基线（W 22.5 GB 不减——Vector 读同一 W 字节） | 同基线 | 0（[16, N] fp32 驻 UB）——但计算自身成为新瓶颈 |

基线在瓶颈口径（W 访存）上全局最优（22.5 GB：与 C2/C3/C5/C6/C7 持平、C1/C4 翻倍；C7 持平但**新增** partial GM 往返与串行归并段），并保持「0 GM 中间缓冲 + 每活跃专家权重恰读一次」形态（C8 同形但计算单元错配 80×，见 R4）。

**R4 硬件亲和逐候选评估（A5）**

| 候选 | A5 亲和评估 | 裁决 |
|------|-------------|------|
| **B0** | 瓶颈资源恰为 A5 最粗管道：W GM→L1 走 **AIC MTE2 100 B/cyc**（`CONST-mte-bandwidth-model`）；权重 `[E,N,k]` = `[N,K]` 行主 = **L1-input NT 路径原生布局，零转换**（§0.5 条目 8）；M=16 = Cube 最小 M 粒度（`CONST-gemm-dim-constraints`）；epilogue Fixpipe 128 B/cyc + VALU 51/8 lane-ops 全离临界路径（D-2）；mixed 三核拓扑有逐形回归先例（`test_schedule_integration.py` "fixed"） | **选定** |
| C1 | 瓶颈资源（W MTE2）×2 → 墙钟近似 ×2 | 淘汰 |
| C2 | deepgemm 形态本身 A5 实证可行（亲和好），但其 contiguous m-indices 布局契约 ≠ 冻结接口（签名无 permute 缓冲；out 为 route 序 `[T,K,n]`），需接口外 host permute kernel + 索引预生成 → 违反迁移契约（§0.2） | 淘汰 |
| C3 | L0C 256 KB 硬容量封死常驻形态；fallback 形态（每 n 重建 res tile）与基线逐项同量、纯重构零收益，且块粒度变粗（585→18 waves）劣化负载均衡 | 淘汰 |
| C4 | W-bound 下无墙钟收益（Cube 仍 5× 余量）；稀疏专家 padding 严格 ×2，AIV0 gather 流量 ×2 | 淘汰 |
| C5 | L1 压力不存在（本设计 L1 峰值 144 KB ≪ 512 KB〔(8+64)×2 复算，§4.5/§3.5.1 同口径〕）；L0-input 需独立 NZ 分形布局管理（`CONST-gemm-dim-constraints` 明示「勿把 L1 限制推广」），收益为零、布局风险为正 | 淘汰 |
| C6 | 调度器 m-tile 契约（contiguous m_indices）≠ `_route_stats` 三段 metadata 契约；接口冻结无法预生成索引；persistent 化收益（block 头尾开销）在本 grid 规模（≥448 blocks / 28 核 ≥16 waves）不显著 | 淘汰 |
| C7 split-K | 四条独立理由（依 revision 1 增补，先例 `example_gemm_splitk.py` 已亲读核对）：**① 瓶颈不变，收益为零**——split-K 不减 W 流量（各 split 读 disjoint K 段，W 每元素仍恰读一次），D-2 瓶颈 ≈8.05M cyc 不降；反而新增 partial GM 往返 + 归并段纯开销。**② 并行度已足**——split-K 适用前提是「输出 tile 数 ≪ 核数」的低并行困境；本算子最小 grid 256 blocks（ds-v3-t1 gate/up）= 9.1 waves（§5.2 全表 waves ≥ 9），前提不成立。**③ 接口冻结 + 精度契约**——ordered 归并需 [T,K,2n] fp32 GM workspace（gated partial 含 gate/up 两半列，out [·,n] 列宽减半放不下），五参签名 `(a, weights, expert_ids, route_metadata, out)` 无 workspace 形参（§0.2 接口冻结，同族淘汰机制对照候选库 ALG-reduction 行「需 GM workspace 形参（源签名冻结时锁死）」）；atomic 路线：`T.set_atomic` 原语存在且 fp16/bf16 在 dtype 表内（`tilelang/ascend/language/mode.py` L56–60 实查；K 维求和仅 add 类适用），但 out 为 fp16/bf16 时 16-bit 原子累加把「fp32 累加 → 落存单次 cast」契约（§0.1 规约语义）改为每 split 部分和 16-bit 舍入 + 16-bit 加法（bf16 8-bit 尾数下 ~split_k·2⁻⁸ 相对误差量级危及 1e-2 容差）——语义变更非容差内等价；且 publish-then-add 协议（example L88–98：split 0 先写 + per-tile `T.ascend_sync_inter_arrive/wait`）要求 out 预置/预零 + 跨核 barrier，违反 `out_idx=[]` 原位写回与 §7.2「每 out 行恰被一个 block 写一次」的无跨块 WAW 结构。**④ epilogue 耦合**——silu_and_mul 需 gate/up 两半列的完整 K 和（与 R2 结论 (b) 同源），部分和无法先行施加非线性（silu(partial) ≠ 对完整和的贡献），归并段必须插在 GEMM 与 epilogue 之间——新增基线不存在的串行段 | 淘汰 |
| C8 Vector 规约 | 计算单元错配：GEMM 主体经 VALU = 361 G lane-ops / (56 AIV × 51 lane-ops/cyc) ≈ **126M cyc**，为 Cube 路径 1.57M cyc 的 **80×**、W 装载瓶颈 8.05M cyc 的 **15.7×**——Vector 规约自身成为新瓶颈（D-2 既有口径折算：`CONST-cube-throughput-model` vs `CONST-valu-model`，候选库 ALG-gemv kb_links 对比口径）；本算子是 MAC 密集算子，Cube 分形路径是唯一匹配单元 | 淘汰 |

**D-2 设计期 roofline 估算**（ds-v3-t64，两实例合计；常数全部 `status: assumption`〔A5 冷启动期〕，按 D-2 协议**只做相对量级比较**；绝对值以 Stage 4 `do_bench(msprof)` 实测为准。常数来源：`CONST-mte-bandwidth-model` / `CONST-cube-throughput-model` / `CONST-valu-model` / `CONST-arch-a5`〔核数 28/56 为本机实测转正项〕）

| 资源 | 需求 | 供给（assumption 口径） | 周期估算 | 备注 |
|------|------|------------------------|----------|------|
| **W 装载（AIC MTE2 GM→L1）** | **22.5 GB** | 28 核 × 100 B/cyc = 2800 B/cyc | **≈8.05M cyc** | **瓶颈**；E-active 数据依赖（W 流量 ∝ 活跃专家数） |
| Cube MAC（padded） | 361 GFLOP | 28 × 8192 ops/cyc = 229K ops/cyc | ≈1.57M cyc | 5.1× 余量（相对瓶颈） |
| A gather（AIV0，GM→UB，行散集） | 2.8 GB | 56 AIV × 100 B/cyc（N-strided 档折减已容 ~2× 余量） | ≈0.50M cyc | 与 AIC MTE2 并行重叠，隐藏（逐 k 步：gather ≈150 cyc vs W ≈320 cyc） |
| A UB→L1（MTE3） | 2.8 GB | 256 B/cyc/AIV | ≈0.20M cyc | 隐藏 |
| VALU epilogue（silu+cast+散写寻址） | ≈0.3–0.5 G lane-ops | 56 × 51（special_func 档 exp=8） | ≈0.1–0.2M cyc | 隐藏（AIV1 K 循环期空闲正是为此留出） |
| out 写（MTE3 UB→GM） | 9.4 MB | 115 B/cyc/AIV | <0.01M cyc | — |
| no-op 块开销（非 leader 块：标量 prologue + 退出） | t64 约 50% 网格块（均衡 c_e=2 时），单块 ≈0.7–1.5K cyc（2 次依赖 GM 标量读，L2 命中假设） | 8192+14336 块 / 28 核 | ≈0.56–1.2M cyc（**≈7–15%**） | 接受（t32 ds-v3 为 0%：c_e=1 全 leader；t1 direct 无分组）。**复算（两实例合计口径，与表头一致）**：22,528 块 / 28 核 = 804.6 块/核 × 0.7–1.5K cyc = 0.56–1.2M cyc；单实例拆分：gate/up 8,192 块 → ≈0.21–0.44M（其自身 W 瓶颈 5.36M 的 3.9–8.2%）、down 14,336 块 → ≈0.36–0.77M（2.68M 的 13–29%）——v0 误将 gate/up 单实例数字标入合计行（revision 1 修正）。no-op 开销不随算法选型内的分组参数消长（C1 可消除 no-op 块但 W 瓶颈翻倍，R3/R4 已淘汰）；assumption 区间估计，Stage 4 msprof 可实测校准 |

**调研结论**：**选定 B0 基线算法族**（源算法的分组结构：leader 分组 + 16 行组 + gate/up 拼接 + 融合 epilogue + in-kernel gather/scatter），叠加 §0.6 R1/R2/R3 的 **A5 数据通路重设计**（MixedKernel 三核分工 / L1-L0C Cube 路径 / off-grid 尾区零填契约）。依据：(1) R3 瓶颈口径（W 访存）全局最优、0 GM 中间缓冲（C8 同为 0 缓冲但计算单元错配 80×，R4 淘汰）；(2) R4 瓶颈资源恰为 A5 最粗管道 AIC MTE2、布局 NT 原生、M=16 恰为 Cube 最小粒度——**硬件亲和最大化**；(3) D-2 显示权重 MTE2 带宽主导（≈8.1M cyc），Cube 5.1×、AIV 一个数量级余量——「最小化 W 流量」是唯一优化目标；no-op 块开销 ≈0.56–1.2M cyc（≈7–15%，两实例合计，t32/t1 形态为 0%）与 padding/block 数同为瓶颈之下的次级项，不改变算法选型（修正口径见 D-2 行注）；(4) C2/C6 布局契约与冻结接口冲突、C1/C4 瓶颈翻倍、C3 硬容量封死、C5 零收益正风险、**C7 split-K 零收益且接口/精度/epilogue 三重冲突（R4 ①–④）、C8 Vector 规约计算侧 80× 劣化**。**调研范围**：本地 `examples/ascend/`（deepgemm / gemm_silu / flash_attention / per_token_cast / **gemm_splitk〔含 `test_gemm_splitk.py` 回归〕**）+ `examples/pilot/` + pattern-library（constants/elementwise/attention/cases）+ GPU 源仓 + AGENTS.md/docs 路由；未做互联网检索——R1 候选表按「分组维度（不分组/分组/连续布局）× M 粒度 × **K 维拆分（串行分块 / split-K 并行归并）** × **计算单元（Cube 分形 / Vector 规约）** × 数据通路（L1/L0 输入）× 调度（block/persistent）」**六轴枚举结构空间**，本地源覆盖充分。

> **修订增量（revision 1）**：按 Stage 2 REVIEW.md 阻塞问题 1 补评 **C7 split-K**（ALG-gemv R2 命中行）与 **C8 Vector 规约累加**（ALG-gemv R1 命中行后半）——v0 枚举轴缺「K 维拆分」「计算单元」两轴致漏评，本次补齐两候选与 R3/R4 定量淘汰论证，**B0 选定不变**；同步修正 D-2 no-op 行口径（两实例合计 ≈0.56–1.2M cyc ≈7–15%；v0 的 ≈0.2–0.4M 为 gate/up 单实例误标）、调研范围补 `example_gemm_splitk.py`。v0 的「穷举结构空间」声明按四轴口径成立、按完整结构维度不成立——本次以六轴枚举替代，防再漏轴。

**源优化意图承接表**（§0.4 全部 11 项手段 → 新设计落点，无静默丢弃）：

| §0.4 # | 源手段 | 新设计落点 |
|--------|--------|-----------|
| 1 | route 索引原地 gather 免 permute | §1.4 步骤 4（AIV0 SimtVF gather，逐语义保留）——R4 C2/C6 淘汰印证其必要性 |
| 2 | 同专家 16 行组 + leader 整组计算 | §1.4 步骤 2–3（逐字保留；R3 证实其是 W 流量最优的关键） |
| 3 | gate/up 拼接单 GEMM | §1.4 步骤 5–6（`[2·bn, bk]` W tile 双段装载 + 单 `T.gemm`；E6 验证块恒等式） |
| 4 | silu 融合 epilogue | §1.4 步骤 7（AIV1；fp32 域 + cast 后置精度路径保留） |
| 5 | block_m=16 最小 M tile | 保留（恰为 A5 Cube M 最小粒度，`CONST-gemm-dim-constraints`） |
| 6 | K 软流水 + SMEM 多缓冲 | `T.Pipelined(num_stages)` + AutoSchedule multibuffering（§0.5 条目 9 等价替换） |
| 7 | SMEM 输入 NT wmma | L1-input NT `T.gemm`（§0.5 条目 8，布局原生零转换） |
| 8 | evict_first L2 hint | 降级为 `l2_cache_ctrl` Stage 4 调优项（§0.5 条目 10；A5 L2 128 MB 容纳 ~2 专家全权重，策略不明确故不默认） |
| 9 | TL_DISABLE_WARP_SPECIALIZED | 舍弃（warp specialization 概念在 A5 不存在；A tile 重写的版本管理由 AutoSchedule multibuffering 承接，§0.5 条目 9） |
| 10 | route-major pid 排序 | 逐字保留（§1.4 步骤 1）；A 行重读经 L2 128 MB（组行集 229 KB × 组数 ≪ 容量） |
| 11 | 标量 prologue 最小退出代价 | 保留 + 强化（OPT-2 staging 把 per-element lists 重读收敛为每块 16 次守卫标量读） |

#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 选定算法（B0 + §0.6 通路重设计）。每项含「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素；**等价性经 `verify_equiv.py` 机器验证**（torch CPU fp64 参照 + 角点值；NaN 感知比较——`max(0.0, nan)` 吞 NaN 的比较缺陷已修复为 NaN-aware）。脚本：`examples/indexed_expert_mlp/_indexed_expert_gemm/verify_equiv.py`（2026-10-29 运行，**12/12 EQUIV_PASS**；revision 1 将 E3c 由零填改为 NaN 金丝雀 off-grid 对偶后**全量重跑**，12/12 EQUIV_PASS——RNG 消耗序列不变，除 E3c 外各项数字与 v0 逐位一致）。

**OPT-1：silu 直除式选型（公式形式）**

- 原式：`value = gate * T.sigmoid(gate) * up`（源 L194/L209 的 sigmoid 形式；候选还有 tanh 恒等式 `z·(tanh(z/2)+1)/2` 与 exp2 换底形式）
- 优化后：`value = gate / (1.0 + T.exp(-gate)) * up`（fp32 域计算，落存 cast）
- 等价性论证：代数恒等 `σ(z) = 1/(1+e^{−z})`；数值上 exp(−z) 上溢时（z 极负）→ `1/inf = 0` → 结果 = `z·0·up... ` 精确极限为 0 ✓；下溢（z 极正）→ `e^{−z}=0` → `z/1·u` ✓；±0.0 与 NaN 传播与 sigmoid 形式一致（E1 角点验证）
- 收益量化：实测探针 `D-ELEM-ACT-FORMULA-PROBE`（pattern-library/elementwise.md，2026-10-16，CANN 9.2）：**直除式比 tanh 恒等式快 1.80×**（215.67µs vs 388.42µs，标量 tanh 为软件例程）；exp2 换底无收益（special_func 档位成员，`CONST-valu-model` 注记：exp/exp2 同档 8 ops/cyc，换底反多 1 乘 + scale 截断误差）
- 机器验证：**E1**（fp64 相对差 + fp32→fp16 ulp 双口径，含 ±0/±inf/NaN 角点）

**OPT-2：行元数据 staging（每 AIV 私有 UB 数组，prologue 一次计算）**

- 原式：源在 gather（L148–155）与 epilogue（L199–206）**各**以 `T.Parallel(16, block_k)` / `T.Parallel(16, block_n)` 逐元素读 `route_metadata[list_base + ...]` 并重算 `sel // top_k`、`sel % top_k`（每元素 1 load + 2 整除/取模，依赖 GPU L2/编译器 CSE 摊薄）
- 优化后：每 AIV 在 prologue 各自 staging 私有 UB 数组：`for mi in T.serial(16): if mi < route_count: sel = metadata[E+R+e·T+group_start+mi]; st_ub[mi] = sel // K; ss_ub[mi] = sel % K`；gather / epilogue 主体改为 `for mi in T.serial(16): for ki/j in T.Parallel(...)` 嵌套循环，行寻址 `a[st_ub[mi], (ss_ub[mi],) offset]` / `out[st_ub[mi], ss_ub[mi], col]`（mi 维外提 → div/mod 与 metadata 读自动每行一次）
- 等价性论证：staging 值 = 原逐元素值的确定性整数算术（同一 `sel`、同一 div/mod 语义，int32 精确无舍入）；UB per-AIV 私有两份副本值相同（synchronization.md「UB 与 sub-core 相关标量状态对每个 AIV 局部」——AIC 不写 UB，各 AIV 自读同一 GM）。单行路径（`route_count==1` 或 direct/重复 route）统一为 `st_ub[0]=token, ss_ub[0]=slot`：与源单行分支数学相同（leader 且 rank<T 时 `lists[e·T+group_start+0]` = 自身 route，E2d 覆盖）
- 收益量化：消除每块 `2 × 16 × bk`（gather）+ `2 × 16 × bn`（epilogue）次逐元素 div/mod（VALU div/mod = 13 lane-ops，`CONST-valu-model`）与 metadata 重读。不做 staging 时 gather 寻址开销 ≈ 26 lane-ops × 16 × 128 × 56 k-steps ≈ **2.98M lane-ops ≈ 58K cyc/AIV**，超过块预算（W 装载 ≈18.4K cyc，D-2）→ **AIV0 将成为落后核**；staging 后 ≈ 32 整除/取模 + 32 守卫标量读（L2 热）≈ 数百 cyc，AIV0 回到 ≈3.2K cyc/块（5.7× 余量）——**本项是把 AIV0 挡在临界路径外的承重结构**
- 机器验证：**E2a–d**（分组算法模拟〔含 staging 寻址〕vs per-route golden：7 路由模式 × stable/random 双射 metadata × 4 shape 组合〔含 off-grid n=96/k=100、route_input 两形态、gated/none〕，最大差 6.127e-05 = fp32 求和序噪声，≤1e-3 门限内）

**OPT-3：A-tile 越界行/列确定性零填**

- 原式：源对 `offset ≥ k` 的 A 列写 0（L140–146/L156–164），但 `mi ≥ route_count` 的 A 行**不写**（SMEM 陈旧值参与 wmma，输出行被 epilogue 掩码掩盖——不可观察但不确定）
- 优化后：统一 gather 循环对 `mi ≥ route_count` 行与 `offset ≥ k` 列**都**写 `Cast(dtype, 0)`
- 等价性论证：被掩码行的输出不写（`mi < route_count` 掩码保留）→ 0 填只改变「不可观察的垃圾」为「确定的 0」；`0 × W = 0` 不污染 L0C 有效行（行独立，GEMM 行间无串扰）；K 尾零列贡献恰为 0（与源 A 零填同）
- 收益量化：确定性合约（同输入同输出，消除 SMEM/UB 陈旧值依赖——源依赖 GPU tensor pool 复用语义，A5 不可移植，§0.5 条目 6）；统一代码路径（单行/分组共用一个 tile 循环，§1.6.2 #8）
- 机器验证：**E3a–c**（pad 行注入 NaN 金丝雀后输出逐位不变——掩码充分性；7 路由模式 × 3 shape 组合）

**OPT-4：W off-grid 尾区「钳制拷贝 + 零填」契约（替代源无守卫越界读）**

- 原式：源 `T.copy(weights[e, n_block·bn:(n_block+1)·bn, ko·bk:(ko+1)·bk], ...)` 切片**可越界**（N 尾：up 半行基址 `n + nb·bn` 可超 `weight_n`；K 尾列可超 `k`）——GPU 靠 pool 越界读不崩 + 掩码不可观察兜底（§0.1 边界 (a)）
- 优化后（off-grid 仅测试形态，on-grid 走零开销直拷）：`T.fill(w_ub, 0)`（整 tile 零填）→ `T.copy(weights[e, r0:r0+rb, kc:kc+kb], w_ub[0:rb, 0:kb])`，`rb = clamp(n − nb·bn, 0, bn)`（gated 两半同式：两半各 n 行、同偏移 nb·bn）、`kb = min(k − ko·bk, block_k)`（运行时表达式，块内循环不变量）→ `T.copy(w_ub, w_l1)` 整 tile UB→L1（行 16 整除 ✓ 行 32B 对齐 ✓）；gated up 半 `rb ≤ 0` 时运行时守卫跳过该半拷贝；GEMM 恒以 K=block_k（C0 整除）计算，零填尾贡献 0
- 等价性论证：(a) in-bounds 区 `[0,rb)×[0,kb)` 与源有效数据逐元素相同；(b) 源越界读垃圾落在 `col ≥ n`（掩码不可观察）或与 A 零列相乘（贡献 0）——零填产生确定的 0，同属不可观察/零贡献，且消除 `0×NaN=NaN` 污染有效列的风险与未定义行为；(c) GM→UB 字节粒度行 + 独立 pitch（skill §6.1）支持任意 rb/kb，绕开 ND2NZ 的 16 整除/32B 对齐约束（该约束由整 tile UB→L1 满足）
- 收益量化：把「未定义但被掩盖」变为「确定的零」——E4-probe 反证其承重性：**去掉零填后 NaN 金丝雀泄漏 rel=inf**（未初始化 UB 值进入有效列）
- 机器验证：**E4a/E4b**（off-grid 全 shape 对比 golden：3.492e-04 / 1.526e-06，fp32 噪声级）+ **E4-probe**（反例对照）

**结构保留项（非优化，语义基线，一并机器验证）**

| 项 | 内容 | 验证 |
|----|------|------|
| S1 | gate/up 拼接单 GEMM：`[16, 2bn] × [2bn, bk]` 块恒等于 gate 半 + up 半两个独立 GEMM（列序 [0,bn)=gate、[bn,2bn)=up ↔ golden `chunk(2, dim=-1)` 列序） | **E6**（int64 精确恒等；fp64 求值序噪声 0 记录在案） |
| S2 | `T.gemm(..., clear_accum=(ko==0))` 替代源 `T.clear(acc)` + 跨步累加（首步清零、后续累加） | **E2a–d**（K 分块串行累加序同源） |
| S3 | fp32 累加域 + fp32 域激活 + 落存 cast（消除 z 的 16-bit 预舍入——`D-MIXED-EPILOGUE-DUALCOPY` 记录的精度最优路径） | **E1/E2**（精度路径与源一致） |
| S4 | 备选形态 epilogue 半分（fallback A：`T.dual_copy` M-split，`mi = sid·8+i` 行映射） | **E5**（半分 vs 整分行映射 fp32 模拟逐位一致，为 §5/§9 fallback 提供语义担保） |

**机器验证结果表**（verify_equiv.py，2026-10-29；revision 1 E3c 修正为 NaN 金丝雀 off-grid 对偶后全量重跑；门限：fp32 序噪声 ≤1e-3、ulp 违反率 = 0；NaN-aware 比较）：

| 检查 | 内容 | 最大差 | 违反率 | 结论 |
|------|------|--------|--------|------|
| E1 | silu 直除式 == g·sigmoid(g) == torch.silu（fp64 rel；fp32→fp16 ulp；含 ±0/±inf/NaN 角点） | 3.969e-16 | 0.00% | EQUIV_PASS |
| E2a | silu_and_mul / route_input=False / n=256,k=128,bn=64,bk=128（7 路由模式 × stable/random metadata） | 6.127e-05 | 0.00% | EQUIV_PASS |
| E2b | 同上 off-grid n=96,k=100 | 0.000e+00 | 0.00% | EQUIV_PASS |
| E2c | none / route_input=True / n=128,k=256,bn=64,bk=128 | 1.426e-05 | 0.00% | EQUIV_PASS |
| E2d | none / route_input=True / off-grid n=100,k=96 | 0.000e+00 | 0.00% | EQUIV_PASS |
| E3a | pad 行 NaN 金丝雀 / gated / on-grid（输出逐位不变） | 0.000e+00 | 0.00% | EQUIV_PASS |
| E3b | pad 行 NaN 金丝雀 / none / on-grid | 0.000e+00 | 0.00% | EQUIV_PASS |
| E3c | pad 行 NaN 金丝雀 / gated / off-grid（E3a 的 off-grid 对偶：NaN pad 行 × 钳制/零填/col<n 掩码组合；revision 1 由零填改为 NaN，表-脚本一致） | 0.000e+00 | 0.00% | EQUIV_PASS |
| E4a | off-grid 尾区（钳制+零填）/ gated / n=96,k=100 | 3.492e-04 | 0.00% | EQUIV_PASS |
| E4b | off-grid 尾区 / none / n=100,k=96 | 1.526e-06 | 0.00% | EQUIV_PASS |
| E4-probe | **反例对照**：未零填 W 尾的 NaN 泄漏 | rel=inf | — | 零填承重性证明 |
| E5 | 半分 epilogue（mi=sid·8+i）== 整分 epilogue（fp32 模拟） | 0.000e+00 | 0.00% | EQUIV_PASS |
| E6 | gate/up 拼接块恒等式（int64 精确；fp64 求值序噪声 0.000e+00 记录为注记） | 0.000e+00 | 0.00% | EQUIV_PASS |

> E2a/E2c 的 1e-5~1e-4 量级差 = 分组算法（K 分块 + 组内行序）与 per-route golden 的 fp32 求和序差异（门限 1e-3 内）；E2b/E2d 为 0（单行组退化时序完全一致）。**全部 12 项 EQUIV_PASS，无 FAIL 项。**
>
> **优化结论**：采纳 OPT-1（silu 直除式）/ OPT-2（行元数据 per-AIV 私有 staging + mi 外提）/ OPT-3（A 行/列确定性零填）/ OPT-4（W off-grid 钳制+零填契约）共 4 项，全部经 `verify_equiv.py` 机器验证 EQUIV_PASS；S1–S4 为结构保留基线（一并验证）。无更多公式级优化空间——依据 §1.6.0 R1：GEMM 本体无更低复杂度代数形式（权重稠密、K 无条件结构），silu 形式自由度已由实测探针裁决（OPT-1），分组/布局结构变体已由 R3/R4 定量淘汰。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点实现方案全部循环与标量计算点；能向量化的全部向量化，不可替代的逐项给出充分理由（「实现简单」不构成理由）。

| # | 计算点 | 形态 | 向量化处置 | 依据 / 不可替代理由 |
|---|--------|------|-----------|---------------------|
| 1 | A gather 主体（`16 × block_k` 元素，行散集 + K 尾零填） | 元素级 | ✅ **向量化**：`T.SimtVF(threads=128)` 内 `for mi in T.serial(16): for ki in T.Parallel(block_k):`——mi 外提（OPT-2），ki 维 128 元素 = 128 线程 × 1 元素，连续 256 B 行段 | SimtVF 线程域编译器自动向量化（AGENTS.md 默认模式）；`example_vecadd.py` / `example_per_token_cast_to_fp8.py` 同形先例 |
| 2 | epilogue silu/cast/散写（`16 × block_n` 元素） | 元素级 | ✅ **向量化**：同上嵌套形态，`T.Parallel(block_n)` 内 `T.exp` / 直除 / `T.Cast` | 同上；exp 为 special_func 档向量化 intrinsic |
| 3 | W 装载（on-grid：`(2·)bn × block_k`；off-grid：钳制行段） | 块传输 | ✅ **DMA**：`T.copy` GM→L1（MTE2）——块传输本身即最大粒度搬运，非标量循环 | AGENTS.md Cube 路径基线；ND2NZ 自动插入 |
| 4 | A UB→L1 / res L0C→UB / off-grid W UB→L1 | 块传输 | ✅ **DMA**：`T.copy`（MTE3 / Fixpipe） | "fixed" 形态 L221–240 逐形先例 |
| 5 | off-grid W 零填（整 tile `(2·)bn × block_k`） | 元素级 | ✅ **向量化**：`T.fill(w_ub, 0)`（SimdVF 域向量填充，off-grid 路径才发射） | `T.fill` 存在性已验证（import 实查 True） |
| 6 | **metadata staging**（16 守卫标量读 + 32 div/mod，每 AIV prologue） | 标量 | ⛔ **保留标量**：16 元素、每块一次、块级 prologue；32 次 div/mod + 32 次 L2 热标量读 ≈ 数百 cyc，摊薄到 ≈18.4K cyc 块预算 <3%。**16-int 批量 `T.copy` 装载已评估并拒绝**：lists 段按专家 T 槽布局，`list_base + e·T + group_start` 起 16 连续 int 在 T 非 16 倍数时跨界（如 T=17 时末专家段读越界 ≤15 int）——批量拷贝 OOB 不安全；逐元素 `mi < route_count` 守卫装载天然 in-bounds（route_count ≤ T − group_start） | 动态边界 + host 元数据类；OOB 不可安全性是硬约束 |
| 7 | **标量 prologue**（pid 分解 / rank / leader / route_count / group_start / off-grid rb、kb 钳制） | 标量 | ⛔ **保留标量**：block 索引类（pid 派生）+ 块级循环不变量（is_leader/route_count 决定整块分支，为**子核一致**运行时守卫——向量化的前提是被替代对象是数据并行计算，而这是控制流标量）；rb/kb 为 `T.copy` 区域 extent 的运行时表达式（每块 2–4 次 min/max，deepgemm 动态区域先例 `copy.py` L13–27） | block 索引 / 控制流类；每块 ≈10 整数运算 |
| 8 | 单行路径（direct / route_count==1 / 重复 route） | 元素级 | ✅ **统一向量化**：不特化 row-0 标量路径，统一走 #1/#2 的 16 行 tile 循环（st_ub[0]=token，`mi < route_count=1` 掩码，行 1–15 零填）——与 Cube 16 行 tile 对齐 | 消除分支重复代码；OPT-3 确定性；E2d/E3 验证 |

> **向量化结论**：元素级计算（gather / epilogue / 零填）全部向量化（SimtVF/SimdVF）或 DMA 化（T.copy）；保留标量的仅 #6/#7 两类——block 索引 / 控制流 / host 元数据 / 动态边界，且 #6 的批量向量化替代方案（16-int T.copy）因 OOB 不安全性被明确拒绝（T=17 反例）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

| # | 决策点 | 决策 | 依据 |
|---|--------|------|------|
| 1 | 权重布局 | **NT 原生零转换**：源 `[E, N, k]`（行主 [N, K]）恰为 A5 L1-input NT 路径约定（`CONST-gemm-dim-constraints`：「L1-input 仅支持 NT，权重 GM 布局 [N, K]」）——`T.gemm(transpose_B=True)` 直用，无转置、无 staging | 源布局与 A5 约定逐字重合（§0.5 条目 8）；`gemm_mad.py` L161–163 强制 NT |
| 2 | gather / scatter 向量化轴 | **k / col 轴**：gather 内层 `T.Parallel(block_k)`（行内连续 128 元素 = 256 B）；scatter 内层 `T.Parallel(block_n)`（out 行内连续 64/128 元素）——行间散集（st/ss 寻址）在 mi 外提后为每行一次 | 行内连续性 = 向量宽度最大化；行散集本质（route 索引）不可向量化但已被 OPT-2 摊薄 |
| 3 | L0C→UB epilogue 通路 | **主选 guarded `T.copy(res, acc_ub)`（`if sid == 1:` 内）**：Fixpipe 路由投影到 AIV1（「implicit T.copy routing targets the participating AIVs」，synchronization.md）；**备选 fallback A = `T.dual_copy` M-split 半分 epilogue**（两 AIV 各取 `mi = sid·8 + i` 半块，N_split 需 32 整除：N_gemm ∈ {64, 128, 256} ✓） | 主选 = "fixed" 形态逐形回归先例（`test_schedule_integration.py` L234–240：`if sid == 1: T.copy(accum, epi)` + SimdVF + store）+ deepgemm `epilogue.py` L279–286 同机制；E5 已验证 fallback 的行映射语义。选主选理由：epilogue 仅 ≈0.1–0.2M cyc（D-2）≪ 瓶颈 8M，半分收益 ≈0 墙钟，且 guarded 形态是被测 lowering |
| 4 | L0C→GM 直写 | ❌ **拒绝**：scatter 目标行散集（16 行 → 16 个 (st,ss) 行），Fixpipe L0C→GM 仅支持连续 2D 区域写出——需连续 out 布局，不符 | skill §6 数据搬运约束；散写必须经 UB→VF 域 |
| 5 | A 装载通路 | **gather → `a_ub`（UB）→ `T.copy(a_ub, a_l1)`**：VF 只写 UB（无 VF→L1 直达向量通路），UB→L1 走 MTE3 256 B/cyc | "fixed" 形态 L221–225 逐形先例（`T.copy(dq, a_l1)`） |
| 6 | pid 排序（L2 局部性） | **route-major 保留**（`route = pid // n_blocks` 先除）：同 route 组的 n_blocks 块相邻 → 组 A 行集（16×K×2B ≈ 229 KB）跨 32 块 L2 复用；A5 L2 128 MB（`CONST-arch-a5` 实测注记）≫ 全部活跃 A 行（9.4 MB） | §0.4 #10 意图承接；L2 容量核算 |
| 7 | off-grid W staging 布局 | **UB 中转**：`w_ub` 整 tile（[N_gemm, block_k]）+ 零填 + 钳制 GM→UB（字节粒度行 + 独立 pitch）→ 整 tile UB→L1（对齐满足 ND2NZ 约束） | skill §6.1「byte-granular rows with independent source/destination pitches」；GM→L1 直拷需行 16 整除/32B 对齐（钳制 rb 不满足，如 n=100/bn=64 时 rb=36） |
| 8 | `l2_cache_ctrl`（evict_first 承接） | **默认 NORMAL_FV，不入主路径**：A5 L2 128 MB 可容 ~2 个 ds-v3 专家全权重（58.7 MB/专家）——逐块流式读下单专家 tile 集工作集 ≪ L2，保不保留各有理；列为 Stage 4 调优面 | §0.5 条目 10；`CONST-arch-a5` L2 实测注记 |

> **布局决策结论**：8 项决策定案——① 权重 NT 原生零转换（源布局与 A5 L1-input 约定逐字重合）；② gather/scatter 向量化轴 = k/col（行内连续）；③ L0C→UB 主选 guarded `T.copy`（"fixed" 形态先例）、fallback `T.dual_copy` M-split（E5 验证）；④ L0C→GM 直写因散集行拒绝；⑤ A 装载经 UB 中转（VF 无 L1 直达）；⑥ route-major pid 保留（L2 局部性）；⑦ off-grid W 经 UB 零填中转（对齐约束）；⑧ `l2_cache_ctrl` 降级为调优面。全部决策有一致先例（"fixed"/gemm_silu/deepgemm）或 E 系机器验证支撑，无「实现简单」类理由。

## 2. 编程模式选型

### 2.1 模式结论

- **编程模式：mixed**（`T.MixedKernel(grid, sids=2) as (pid, sid)`；核内 Vector 区域 `with T.SimtVF(threads=128):`）
- **正交字段：`auto_schedule: yes`**（默认，不设 `PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False`，不写手动 flag）
- 一维 grid：`T.MixedKernel(R × n_blocks, sids=2)`（kernel docstring 声明 `programming_mode: mixed`）

### 2.2 选型理由

1. **GEMM 主导 → Cube 路径强制**：本 kernel 计算主体是 `[16, (2·)bn] × [(2·)bn, block_k]` 的 K 分块 GEMM（fp32 累加），AGENTS.md 规定 GEMM 主导算子用 mixed（`T.gemm` L1-input NT + `T.alloc_l0c` 累加器 + `clear_accum`）。
2. **A 侧 gather 与 epilogue 是 VF 域工作**：A 行散集（metadata 索引、掩码、零填）与 silu/cast/散写均为逐元素向量计算（`T.Parallel` 域），VF 无 L1 直达通路——必须经 UB（§1.6.3 #5）→ 天然的三核分工。
3. **sids=2（而非 1）**：AIV0 = A 侧生产者（staging + gather + `T.copy(a_ub, a_l1)`），AIV1 = epilogue（L0C→UB + silu + 散写）。两角色互补且各持私有 UB（`st_ub/ss_ub` 每 AIV 一份副本，OPT-2）；sids=1 会把两段 VF 工作串行压到 AIV0——虽然 D-2 显示两者都在 AIC MTE2 瓶颈掩盖之下（墙钟收益 ≈0），但 sids=2 是「fixed」形态的**已测拓扑**（`test_schedule_integration.py` L219–240 逐形回归）且避免单 AIV 的 UB 版本管理压力。
4. **auto_schedule: yes**：数据依赖边全部跨 pipe / 跨核（`a_ub→a_l1→gemm`、`w_l1→gemm`、`res→acc_ub` Fixpipe、staging→消费），AutoSchedule 自动插 flag + multibuffering 版本管理（tilelang-ascend skill §7）；手动 flag 配对（核内 8 / 跨核 16 预算）在 ~56 k-steps × 依赖边数下收益为负且易错。gemm_silu / deepgemm 均默认 auto_schedule。
5. **不选 SimdVF 主路径**：gather 含运行时分歧掩码（`mi < route_count` / `offset < k`）与行散集——SIMT 线程域天然处理分歧；SimdVF 2048-bit intrinsic 形态留给 Stage 4 热点（备选形态：bk=128 fp16 恰为一行 2048-bit，`T.simd.vld` 按行装载可行，§9）。

### 2.3 模式影响

- **UB 预算**：SimtVF 线程上下文 32KB → 每 AIV 可用 UB = 216KB（`CONST-capacity-a5`；wrapper 侧 `get_device_backend().shared_memory_budget(device_index)`〔A5 wrapper `indexed_expert_mlp.py` L213 实调〕恒按 216KiB 保守口径，K6）。本设计 UB 峰值 ≈16KB（on-grid）/ ≈80KB（off-grid，§4.5）——余量 ≥2.7×。
- **kernel launch**：一维 `T.MixedKernel(grid, sids=2)`；`threads=` 已按 K3/K9 从 config 与 `_kernel` 签名移除（`_kernel(block_n, block_k, num_stages)` 三键调优面保留）。
- **无 warp specialization / `TL_DISABLE_WARP_SPECIALIZED`**：A5 无此机制，pass_config 不携带（§0.5 条目 9 舍弃）。

## 3. API 映射设计

### 3.1 公式拆解

$y^{(t,s)} = a_{\text{row}(t,s)} \cdot W_e^{\top}$ 拆解为：(1) 行映射 `row(t,s)`（grouped：metadata lists 经 staging 展开 16 行组；direct：自身行）；(2) K 分块点积累加（Cube GEMM，fp32）；(3) gated 融合 `silu(gate)·up`（直除式，fp32 域）；(4) cast + 掩码散写。分组语义 = 纯任务映射（哪块算哪批行），不改变每行数学内容（E2）。

### 3.2 TileLang API 映射

| 计算逻辑 | TileLang Ascend API（含参数） | 依据 |
|----------|-------------------------------|------|
| 一维 grid + 三核分工 | `with T.MixedKernel(num_routes * n_blocks, sids=2) as (pid, sid):` | skill §3；"fixed" 形态 L201 |
| AIV0/AIV1 角色守卫 | `if sid == 0:` / `if sid == 1:`（块级 `if is_leader != 0:` 守卫在 sid 守卫之外） | synchronization.md「Independent AIV execution」；deepgemm `epilogue.py` L279 |
| 块级可变标量 | `route_count = T.alloc_var("int32", init=1)`（is_leader/group_start 同式，`if` 内赋值） | 源 L111–127 逐字模式 |
| A tile（AIV0 私有 UB） | `a_ub = T.alloc_shared((16, block_k), dtype)` | skill §4；"fixed" L205 `dq` |
| W tile / A tile（L1 公共） | `a_l1 = T.alloc_l1((16, block_k), dtype)`、`w_l1 = T.alloc_l1((N_gemm, block_k), dtype)` | gemm_silu L78–79 |
| fp32 累加器（L0C） | `res = T.alloc_l0c((16, N_gemm), "float32")` | gemm_silu L77 |
| A 行 gather（SimtVF） | `with T.SimtVF(threads=128): for mi in T.serial(16): for ki in T.Parallel(block_k): a_ub[mi, ki] = a[st_ub[mi], (ss_ub[mi],) offset]`（掩码 + 零填） | skill §2 SimtVF；test_schedule_integration L986–988（Pipelined 内 SimtVF） |
| UB→L1 | `T.copy(a_ub, a_l1)` | "fixed" L225；MTE3 256 B/cyc |
| W GM→L1 直装（on-grid） | `T.copy(weights[expert, rows, cols], w_l1[seg, :])`（ND2NZ 自动插入） | AGENTS.md Cube 路径；gemm_silu L90–93 |
| W off-grid staging | `T.fill(w_ub, 0)` + `T.copy(weights[expert, r0:r0+rb, kc:kc+kb], w_ub[0:rb, 0:kb])`（运行时 extent）+ `T.copy(w_ub, w_l1)` | skill §6.1 字节粒度行；deepgemm `copy.py` L13–27 动态区域 |
| Cube MAC | `T.gemm(a_l1, w_l1, res, transpose_B=True, clear_accum=(ko == 0))` | gemm_silu L94 逐字同形；`gemm_mad.py` L161–163 NT 强制 |
| K 软流水 | `for ko in T.Pipelined(k_blocks, num_stages=num_stages):` | gemm_silu L85；flash_attention `core.py`（Pipelined 内 VF+gemm 混流） |
| L0C→UB（epilogue 首步） | `if sid == 1: T.copy(res, acc_ub)`（fp32 无转换） | "fixed" L236 `T.copy(accum, epi)`；deepgemm epilogue.py L279–286「编译器把拷贝与私有 UB epoch 投影到该 AIV」 |
| silu（直除式） | `value = gate / (1.0 + T.exp(-gate)) * up`（fp32 域） | gemm_silu L101 逐字同形；OPT-1 探针裁决 |
| cast | `out[...] = T.Cast(dtype, value)` | 源 L197/L212 |
| 散写 | `out[st_ub[mi], ss_ub[mi], col] = ...`（SimtVF 内逐元素 store） | VF→GM 直写（vecadd/per_token_cast 形态） |
| 钳制/极值 | `rb = T.max(T.min(n - nb * block_n, block_n), 0)`、`kb = T.min(k - ko * block_k, block_k)`、`route_count = T.min(16, T.min(metadata[expert], num_tokens) - group_start)` | 源 L125–127 同式 |
| 标量 GM 读 | `expert_ids[token, slot]`、`route_metadata[idx]`（TIR 标量索引） | 源 L116/L120 同式 |

### 3.3 计算伪代码

```python
import tilelang
import tilelang.ascend.language as T


@tilelang.jit(out_idx=[], compile_flags=["-O3", "-DENABLE_BF16"])
def _indexed_expert_gemm(num_tokens, top_k, num_experts, n, k, dtype,
                         activation, route_input, dispatch_mode):
    """programming_mode: mixed, auto_schedule: yes. 接口冻结（K3/K9：无 threads）。"""
    # —— 工厂期常量（lru_cache 按 9 元组特化，全部编译期烘焙）——
    num_routes = num_tokens * top_k
    gated = activation != "none"                 # "silu_and_mul" / "none"
    weight_n = 2 * n if gated else n
    a_shape = (num_tokens, top_k, k) if route_input else (num_tokens, k)
    grouped = dispatch_mode == "grouped"         # iff T > 1（上层分支谓词）
    metadata_size = (num_experts + num_routes + num_experts * num_tokens) if grouped else 1
    rank_base, list_base = num_experts, num_experts + num_routes
    block_m = 16

    def _kernel(block_n: int, block_k: int, num_stages: int):
        N_gemm = 2 * block_n if gated else block_n      # W tile N 宽（gated 拼接两半）
        n_blocks = (n + block_n - 1) // block_n          # gated 按半宽计数，覆盖全部 weight_n 列
        k_blocks = (k + block_k - 1) // block_k
        on_grid = (n % block_n == 0) and (k % block_k == 0)   # 静态尾区判定

        @T.prim_func
        def main(
            a: T.Tensor(a_shape, dtype),
            weights: T.Tensor((num_experts, weight_n, k), dtype),
            expert_ids: T.Tensor((num_tokens, top_k), "int32"),
            route_metadata: T.Tensor((metadata_size,), "int32"),
            out: T.Tensor((num_tokens, top_k, n), dtype),
        ):
            with T.MixedKernel(num_routes * n_blocks, sids=2) as (pid, sid):
                # —— 缓冲区（UB=alloc_shared 每 AIV 私有；L1/L0C 公共）——
                a_ub   = T.alloc_shared((block_m, block_k), dtype)      # AIV0 私有
                acc_ub = T.alloc_shared((block_m, N_gemm), "float32")   # AIV1 私有
                st_ub  = T.alloc_shared((block_m,), "int32")            # 每 AIV 私有 staging
                ss_ub  = T.alloc_shared((block_m,), "int32")
                a_l1 = T.alloc_l1((block_m, block_k), dtype)
                w_l1 = T.alloc_l1((N_gemm, block_k), dtype)
                res  = T.alloc_l0c((block_m, N_gemm), "float32")
                if not on_grid:                                         # Python 级：off-grid 才发射
                    w_ub = T.alloc_shared((N_gemm, block_k), dtype)     # W 中转（零填域）

                # —— 公共标量 prologue（三核同构执行；子核一致的运行时守卫源）——
                route = pid // n_blocks
                nb = pid % n_blocks
                token = route // top_k
                slot = route % top_k
                expert = expert_ids[token, slot]
                route_count = T.alloc_var("int32", init=1)
                is_leader = T.alloc_var("int32", init=1)
                group_start = T.alloc_var("int32", init=0)
                if grouped:                                             # Python 级：direct 不发射
                    rank = route_metadata[rank_base + route]
                    if rank < num_tokens:                               # rank >= T：重复 route，独立算自身
                        group_start = rank - rank % block_m
                        if rank != group_start:
                            is_leader = 0
                        route_count = T.min(
                            block_m, T.min(route_metadata[expert], num_tokens) - group_start)

                if is_leader != 0:                                       # 非 leader 块：prologue 后整块退出
                    # —— AIV0：行元数据 staging（prologue，OPT-2；Pipelined 之前）——
                    if sid == 0:
                        if grouped:
                            for mi in T.serial(block_m):
                                if mi < route_count:
                                    if route_count == 1:               # 孤立 leader / 重复 route → 自身
                                        sel = route
                                    else:                               # 组行 → lists（in-bounds：mi < route_count）
                                        sel = route_metadata[
                                            list_base + expert * num_tokens + group_start + mi]
                                    st_ub[mi] = sel // top_k
                                    ss_ub[mi] = sel % top_k
                        else:                                           # direct：无 metadata 代码
                            st_ub[0] = token
                            ss_ub[0] = slot

                    # —— K 软流水（AIV0 产 A → AIC 装 W + MAC；epilogue 后置）——
                    for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                        if sid == 0:                                    # AIV0：gather → a_ub → a_l1
                            with T.SimtVF(threads=128):
                                for mi in T.serial(block_m):
                                    for ki in T.Parallel(block_k):
                                        offset = ko * block_k + ki
                                        if mi < route_count and (on_grid or offset < k):
                                            if route_input:             # down：3-D 每 route 一行
                                                a_ub[mi, ki] = a[st_ub[mi], ss_ub[mi], offset]
                                            else:                       # gate/up：2-D 每 token 一行
                                                a_ub[mi, ki] = a[st_ub[mi], offset]
                                        else:                           # OPT-3：行/列越界确定性零填
                                            a_ub[mi, ki] = T.Cast(dtype, 0)
                            T.copy(a_ub, a_l1)                          # UB→L1（MTE3）
                            if not on_grid:                             # off-grid：W 零填 + 钳制拷贝 + 上 L1
                                T.fill(w_ub, 0)                         # E4-probe：零填承重
                                rb = T.max(T.min(n - nb * block_n, block_n), 0)   # 两半同式（各 n 行、同偏移）
                                kb = T.min(k - ko * block_k, block_k)
                                T.copy(weights[expert, nb * block_n : nb * block_n + rb,
                                               ko * block_k : ko * block_k + kb],
                                       w_ub[0:rb, 0:kb])                 # 字节粒度行 + 独立 pitch
                                if gated:
                                    T.copy(weights[expert, n + nb * block_n : n + nb * block_n + rb,
                                                   ko * block_k : ko * block_k + kb],
                                           w_ub[block_n : block_n + rb, 0:kb])
                                T.copy(w_ub, w_l1)                       # 整 tile UB→L1（行 16 整除、行 32B 对齐）
                        # —— 公共 Cube 段（AIC，sid guard 之外）——
                        if on_grid:                                     # 静态直拷快速路径（零守卫零零填）
                            T.copy(weights[expert, nb * block_n : (nb + 1) * block_n,
                                           ko * block_k : (ko + 1) * block_k],
                                   w_l1[0:block_n, :])                   # gate 半（ND2NZ 自动）
                            if gated:
                                T.copy(weights[expert, n + nb * block_n : n + (nb + 1) * block_n,
                                               ko * block_k : (ko + 1) * block_k],
                                       w_l1[block_n : 2 * block_n, :])   # up 半
                        T.gemm(a_l1, w_l1, res, transpose_B=True, clear_accum=(ko == 0))

                    # —— AIV1：epilogue（K 循环之后，res 累加完成）——
                    if sid == 1:
                        if grouped:                                     # AIV1 私有 staging 副本（同构）
                            for mi in T.serial(block_m):
                                if mi < route_count:
                                    if route_count == 1:
                                        sel = route
                                    else:
                                        sel = route_metadata[
                                            list_base + expert * num_tokens + group_start + mi]
                                    st_ub[mi] = sel // top_k
                                    ss_ub[mi] = sel % top_k
                        else:
                            st_ub[0] = token
                            ss_ub[0] = slot
                        T.copy(res, acc_ub)                              # L0C→UB（guard 路由到 AIV1，fp32）
                        with T.SimtVF(threads=128):
                            for mi in T.serial(block_m):
                                for j in T.Parallel(block_n):
                                    col = nb * block_n + j
                                    if mi < route_count and (on_grid or col < n):
                                        if gated:
                                            gate = acc_ub[mi, j]         # gate 半：输出列 col ↔ acc 列 j
                                            value = gate / (1.0 + T.exp(-gate)) \
                                                    * acc_ub[mi, block_n + j]   # up 半：acc 列 bn+j（E6 列序）
                                        else:
                                            value = acc_ub[mi, j]
                                        out[st_ub[mi], ss_ub[mi], col] = T.Cast(dtype, value)

        return main

    return _kernel
```

**与源语义的逐点对照**（§0.3 步骤表 → 本伪代码）：

| 源步骤 | 本伪代码 | 等价性 |
|--------|----------|--------|
| 1 pid 分解 | prologue `route/nb/token/slot` | 逐字 |
| 2 expert 读 | prologue `expert_ids[token, slot]` | 逐字 |
| 3 grouped 首部 | `if grouped:` rank/leader/route_count/group_start | 逐字（L120–127 同式） |
| 4 is_leader 守卫 | `if is_leader != 0:` | 逐字；**子核一致**（三核对同一 GM 数据求值相同） |
| 5a A 装载 | sid==0 gather（unified 16 行 tile；单行路径 = route_count 1 特例，行 1–15 零填） | E2（分支合一后逐位一致）+ E3（零填掩码充分） |
| 5b W 装载 | on-grid 直拷 / off-grid fill+钳制（源无守卫越界读的 A5 承接，§0.6 R3） | E4a/b + E4-probe |
| 5c 同步 + wmma | （`T.sync_threads` 舍弃）+ `T.gemm` NT L1-input | §0.5 条目 11/8；E2 |
| 6 acc→SMEM | `T.copy(res, acc_ub)`（Fixpipe，sid==1 guard） | §0.5 条目 5；E5 |
| 7 epilogue | sid==1 SimtVF silu/cast/散写（staging 复用免重读） | E1/E2；OPT-2 |

### 3.4 API 可行性确认

| API | 存在性验证 | 参考 |
|-----|-----------|------|
| `T.MixedKernel(grid, sids=2)` | python import 实查 True；`testing/ascend/auto_schedule/test_schedule_integration.py` L201 逐形回归 | skill §3 |
| `T.SimtVF(threads=128)` / `T.Parallel` / `T.serial` | import 实查 True；test L986（Pipelined 内）；ss 测试 L161（MixedKernel 内） | skill §2 |
| `T.Pipelined(n, num_stages)` | import 实查 True；gemm_silu L85 | skill §7 |
| `T.copy`（GM↔UB/L1、UB→L1、L0C→UB、运行时 extent 区域） | import 实查 True；"fixed" L221/225/236；deepgemm `copy.py` L13–27（动态区域）、`epilogue.py` L282 | skill §6 |
| `T.gemm(a, b, c, transpose_B=True, clear_accum=)` | `tilelang/ascend/op/gemm/gemm_mad.py` L161–163（NT 强制）、L206–222（`_compute_tile_k_sub` K 子切分）；gemm_silu L94 | skill §5 |
| `T.dual_copy`（fallback） | import 实查 True；gemm_silu L97/L102 | skill §6 |
| `T.alloc_l1/l0c/shared` | gemm_silu L77–81 | skill §4 |
| `T.alloc_var("int32", init=)` | import 实查 True；源 L111–113 逐字模式 | language_basics |
| `T.exp` / `T.Cast` / `T.min` / `T.max` / `T.fill` / `T.ceildiv` | import 实查全 True（`T.select` 不存在——已规避，用 if/else 表达） | `tilelang/ascend/language/` |
| `T.simd.vld/vsts`（备选 SimdVF 形态） | "fixed" L222–224 | skill §10 |

> 注：`T.select` 缺失（import 实查 False）——所有逐元素条件一律用 `if/else` 块表达（源同形）。负向断言依据：`tilelang/ascend/language/__init__.py` 导出面 + import 实测（2026-10-29）。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 影响 | 本设计处置 |
|------|------|-----------|
| Kernel launch 一维、`threads=` 禁用（TypeError） | 源二维语义（grid×threads）需重组 | §0.6 R1：`T.MixedKernel` 一维 grid + sids=2；wrapper 已移除 threads（K3/K9） |
| C0 分形对齐（bf16/fp16 C0=16；M、N 须 16 倍数；ND2NZ 行 16 整除、行 32B 对齐；K % TILE_K_SUB == 0 且 TILE_K_SUB % C0 == 0） | tile 形态受限 | M=16 ✓ 最小粒度；N_gemm ∈ {64,128,256}（=block_n×2 gated）均 16 倍数 ✓；K=block_k ∈ {64,128,256} 均 C0 整除 ✓；off-grid 尾区经 UB 零填 + 整 tile 上 L1 满足对齐（§0.6 R3）；`gemm_mad.py` tile_k_sub 自动子切分兜底大 bk |
| L0C 256KB / L1 512KB / UB 248KB（SimtVF 216KB） | 缓冲布局 | res 8KB、L1 峰值 144KB、UB 峰值 ≈80KB（§4.5）——全部 ≤64% 容量 |
| `T.copy` GM→L1 无越界语义（源 OOB 行为不可移植） | off-grid 尾区 | §0.6 R3：钳制 + 零填（E4/E4-probe） |
| `T.Kernel` grid 须为编译期可求值 | grid = R × n_blocks | R/n_blocks 全为工厂期 Python int ✓ |
| 双精度/INT8 MMAD 不支持 | — | 本算子 fp16/bf16 输入、fp32 累加 ✓ 不涉及 |
| `l2_cache_ctrl` 等调优面 | — | 默认 NORMAL_FV；Stage 4 面（§1.6.3 #8） |

#### 3.5.2 参考实现差异说明

- **vs `gemm_silu.py`**：同为 mixed GEMM+silu，但 gemm_silu 是 plain `T.Kernel` + `T.Persistent` + `T.dual_copy` M-split 半分（两 AIV 同体执行）+ SimtVF 非守卫；本设计需要 **AIV0/AIV1 异构角色**（gather 生产者 vs epilogue）→ `T.MixedKernel(sids=2)` + guarded `T.copy(res, acc_ub)`。gemm_silu 的 `T.gemm(..., clear_accum=(kt==0))`、SimtVF 直除式 silu（L101）、`T.alloc_l1/l0c` 布局逐字沿用；其 dual_copy 形态保留为本设计 **fallback A**（E5 验证行映射）。
- **vs `deepgemm`**：其 m-grouped GEMM 以 contiguous m-indices + host 排布为契约（R4 淘汰 C2/C6 的对象）；本设计仅继承其 **动态区域拷贝**（`copy.py`）、**sid 守卫 epilogue**（`epilogue.py` L279「A complete MAD tile belongs to one AIV」同机制）、运行时 batch/行基址先例（`common.py` `matrix_region`）。
- **vs GPU 源**：语义逐点对照见 §3.3 表；结构性差异（threads→sids、SMEM→L1/L0C/UB 三级、sync_threads→AutoSchedule、OOB→钳制零填）全部经 §0.5/§0.6 处置 + E 系机器验证。

#### 3.5.3 本项目同类实现参考

| 参考文件 | 借鉴点 |
|----------|--------|
| `testing/ascend/auto_schedule/test_schedule_integration.py` L194–249 | **主选结构逐形回归**："fixed" 模式（sid==0 生产者 + 公共 gemm + sid==1 epilogue，Pipelined 内） |
| `examples/pilot/gemm_silu/gemm_silu.py` | mixed GEMM+silu 实测案例（L0–L2/Boundary 全 PASS）：L1/l0c 布局、`clear_accum`、直除式 silu、dual_copy fallback 拓扑 |
| `examples/ascend/deepgemm/kernels/{copy,common,epilogue}.py` | 动态区域 GM→L1/UB 拷贝、`matrix_region` 运行时基址、sid 守卫 L0C→UB |
| `examples/ascend/flash_attention/core.py` | `T.Pipelined` 体内 VF（SimdVF）+ `T.gemm` 混合流水的先例 |
| `examples/ascend/example_per_token_cast_to_fp8.py` / `examples/ascend/example_vecadd.py` | SimtVF 逐元素 + GM 直写形态（epilogue/scatter 同形） |
| `examples/indexed_expert_mlp/_route_stats/`（本批前序） | metadata 生产契约（counts/ranks/lists 三段布局、`rank_base=E`、`list_base=E+R`——本设计消费方） |

## 4. 数据规格与内存规划

### 4.1 输入张量

| 张量 | Shape | dtype | stride 假设 | 说明 |
|------|-------|-------|-------------|------|
| a | (T, K_route, k) 或 (T, k) | fp16/bf16 | contiguous（manifest 契约） | route_input=True（down）3-D 每 route 一行；False（gate/up）2-D 每 token 一行 |
| weights | (E, 2n 或 n, k) | fp16/bf16 | contiguous | `[N, K]` 行主 = L1-input NT 原生布局（零转换，§1.6.3 #1）；gated 前 n 行 gate、后 n 行 up |
| expert_ids | (T, K_route) | int32 | contiguous | 值 ∈ [0, E)（调用方契约，kernel 不校验，与源一致） |
| route_metadata | (E+R+E·T,) grouped / (1,) direct | int32 | contiguous | 三段：counts[E] @0 / ranks[R] @E / lists[E·T] @E+R（`rank_base=E`、`list_base=E+R`）；direct 为占位不读 |

### 4.2 输出张量

| 张量 | Shape | dtype | 说明 |
|------|-------|-------|------|
| out | (T, K_route, n) | fp16/bf16 | **原位写回**（`out_idx=[]`，wrapper 返回传入 out）；每 route 行恰被一个 block 写一次（leader 写组行 / 重复 route 写自身行 / 非 leader 退出——无跨块 WAW，§7.2） |

### 4.3 中间缓冲区（片上，无 GM 中间量——epilogue 融合）

| 缓冲 | Shape | dtype | scope | 写者 → 读者 | versions（ns=num_stages） |
|------|-------|-------|-------|-------------|--------------------------|
| a_ub | (16, block_k) | fp16/bf16 | UB（AIV0 私有） | AIV0 SimtVF gather → AIV0 `T.copy(a_ub, a_l1)` | ns |
| w_ub | (N_gemm, block_k) | fp16/bf16 | UB（AIV0 私有，**off-grid 才发射**） | AIV0 fill+钳制 copy → AIV0 `T.copy(w_ub, w_l1)` | ns |
| st_ub / ss_ub | (16,) | int32 | UB（每 AIV 私有一份） | 各 AIV staging → 各 AIV gather/epilogue | 1 |
| acc_ub | (16, N_gemm) | float32 | UB（AIV1 私有） | AIV1 `T.copy(res, acc_ub)` → AIV1 SimtVF epilogue | 1–2 |
| a_l1 | (16, block_k) | fp16/bf16 | L1（公共） | AIV0 UB→L1 → AIC `T.gemm` | ns |
| w_l1 | (N_gemm, block_k) | fp16/bf16 | L1（公共） | AIC GM→L1（on-grid）/ AIV0 UB→L1（off-grid）→ AIC `T.gemm` | ns |
| res | (16, N_gemm) | float32 | L0C（公共） | AIC `T.gemm` 累加 → AIV1 Fixpipe 读 | 1–2 |

`N_gemm = 2·block_n`（gated）/ `block_n`（none）。versions 由 AutoSchedule multibuffering 托管（auto_schedule: yes）。

### 4.4 内存搬运路径

```
GM[a]          ──AIV0 MTE2（GM→UB，行散集 ~50–100 B/cyc）──► UB[a_ub] ──AIV0 MTE3（UB→L1 256 B/cyc）──► L1[a_l1]
GM[weights]    ──AIC MTE2（GM→L1 100 B/cyc，ND2NZ 自动分形）───────────────────────────────────────► L1[w_l1]   （on-grid）
GM[weights]    ──AIV0 MTE2（GM→UB，字节粒度行）──► UB[w_ub]（零填域）──AIV0 MTE3──► L1[w_l1]                  （off-grid）
L1[a_l1]×L1[w_l1] ──AIC Cube（NT MAC，fp32）──► L0C[res]（跨 ko 累加）
L0C[res]       ──AIV1 Fixpipe（L0C→UB 128 B/cyc，fp32 无转换，sid==1 guard 路由）──► UB[acc_ub]
UB[acc_ub]     ──AIV1 VF/向量 store（掩码散写，行内连续 128–256 B）──► GM[out]
GM[expert_ids/route_metadata] ──标量 load（公共 prologue / staging，L2 热）──► 寄存器/UB[st_ub,ss_ub]
```

### 4.5 UB 内存预算（SimtVF 口径 216KB / AIV；`CONST-capacity-a5`）

| 配置（实例, bn, bk, ns） | AIV0 UB | AIV1 UB | 峰值占比 | 判定 |
|--------------------------|---------|---------|----------|------|
| gate/up manifest（64,128,2） | a_ub 4KB×2 + st/ss 0.125KB ≈ 8.3KB | acc_ub 8KB×2 + st/ss ≈ 16.3KB | 7.5% | ✓ |
| down manifest（128,128,2） | ≈ 8.3KB | ≈ 16.3KB | 7.5% | ✓ |
| down t1（128,256,2） | a_ub 8KB×2 + 0.125 ≈ 16.3KB | ≈ 16.3KB | 7.5% | ✓ |
| off-grid gate/up（64,128,3，测试形态） | a_ub 4×3 + w_ub 32KB×2 + 0.125 ≈ 76.4KB | ≈ 16.3KB | 35.4% | ✓ |
| off-grid down（64,256,2，测试形态） | a_ub 8×2 + w_ub 32KB×2 ≈ 80.4KB | ≈ 16.3KB | 37.2% | ✓ |

- **L1 预算**：峰值 = down t1（128,256,2）：(a_l1 8KB + w_l1 64KB)×2 = **144KB ≤ 512KB**（28%）✓；manifest 主路径 (4+32)×2 = 72KB。
- **L0C 预算**：res 16×128×4B = 8KB ×2v = 16KB ≤ 256KB ✓。
- **wrapper 预算 fit 循环复核**（`rows = 16 + (2 if gated else 1)·bn`，限 216KiB）：(64,128,2)→73,728B；(128,128,2)→73,728B；(128,256,2)→147,456B；(64,128,3)→110,592B——**全部 ≤ 221,184B，manifest 与 off-grid 测试默认配置均无需降级**（§5.2 引用）。
- 注：AutoSchedule multibuffering 的 UB 实际膨胀系数未知（`CONST-capacity-a5` 待实测转正项）——本设计峰值 ≤37%（off-grid）/ ≤8%（manifest），按 D-2 协议不叠加未证实膨胀系数，余量充分。

### 4.6 动态轴定义

- **张量 shape 全静态**（工厂 9 元组烘焙；无动态 shape 轴）。
- **运行时标量**（全部块级不变量，无元素级动态）：`pid → route/nb/token/slot`；`expert`（GM 读）；grouped 时 `rank/group_start/route_count/is_leader`（GM 读 ×2）；off-grid 时 `rb/kb`（min/max 表达式，n_block/ko 派生）。
- **数据依赖性**：W 流量 ∝ 活跃专家数（≤ min(R, E)，D-2）；块内工作量 ∝ route_count ∈ [1,16]——均为**性能**数据依赖，语义不变（任意自洽 metadata 输出一致，E2）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[], compile_flags=["-O3", "-DENABLE_BF16"])
```

- `out_idx=[]`：原位写回 out（wrapper 契约）。
- **pass_configs：无**——auto_schedule 默认开启；不携带 `TL_DISABLE_WARP_SPECIALIZED`（A5 无 warp specialization，§0.5 条目 9）。
- host 侧 `functools.lru_cache(maxsize=64)` 按 9 元组特化 + wrapper `default_config` 启发式 + UB 预算 fit 循环——**A5 wrapper 已脚手架化**（L145–266，冻结契约），Stage 3 仅替换 kernel 主体。

## 5. Tiling 策略

### 5.1 计算类型

route 索引分组专家 GEMM（GEMM 主导 + Vector epilogue）：M = 16（行组，Cube 最小 M 粒度）；N = `(2·)block_n`（gated 拼接半 ×2）；K = `block_k` 分块流（fp32 L0C 累加）。

### 5.2 Block 划分

**grid = R × n_blocks**（`num_blocks = n_blocks = ceildiv(n, block_n)`，一维 `T.MixedKernel(N, sids=2)`；route-major：`route = pid // n_blocks`；同 route 的 n_blocks 块相邻 → 组 A 行 L2 复用，§1.6.3 #6）。n_blocks 按半宽计（gated 每块覆盖 gate 半段 `[nb·bn,(nb+1)·bn)` + up 半段 `[n+nb·bn, ...)`，N tile = 2·bn）。

| workload（实例） | bn, bk, ns（wrapper 默认） | R | n_blocks | k_blocks | grid | waves（28 AIC） | on-grid |
|------------------|---------------------------|-----|----------|----------|------|----------------|---------|
| ds-v3-t1（gate/up） | 64, 128, 2 | 8 | 32（F=2048/64） | 56（H=7168/128） | 256 | 9.1 | ✓ |
| ds-v3-t1（down） | 128, 256, 2 | 8 | 56（H=7168/128） | 8（F=2048/256） | 448 | 16 | ✓ |
| ds-v3-t32（gate/up） | 64, 128, 2 | 256 | 32 | 56 | 8,192 | 293 | ✓ |
| ds-v3-t32（down） | 128, 128, 2 | 256 | 56 | 16（F=2048/128） | 14,336 | 512 | ✓ |
| ds-v3-t64（gate/up） | 64, 128, 2 | 512 | 32 | 56 | 16,384 | 585 | ✓ |
| ds-v3-t64（down） | 128, 128, 2 | 512 | 56 | 16 | 28,672 | 1,024 | ✓ |
| qwen3-235b-t32（gate/up） | 64, 128, 2 | 256 | 24（F=1536/64） | 32（H=4096/128） | 6,144 | 219 | ✓ |
| qwen3-235b-t32（down） | 128, 128, 2 | 256 | 32（H=4096/128） | 12（F=1536/128） | 8,192 | 293 | ✓ |

- **全部 4 条 manifest workload × 2 实例均 on-grid**（`n % bn == 0 and k % bk == 0`）——走零开销直拷快速路径；off-grid 仅测试形态（`test_dims_off_the_tile_grid` H=100/F=96 等，§0.6 R3）。
- **tile 形态**：A tile `[16, bk]`（4–8KB）；W tile `[2·bn, bk]`（32–64KB）；res `[16, 2·bn]` fp32（8KB）。block_m=16 固定（§0.5 条目 3：Cube 最小 M 粒度 + 稀疏路由最小 padding，R4 淘汰 block_m=32）。
- **leader 块（工作块）**：活跃 (专家, n_block) 对。均衡路由下 t64：256×32 = 8,192 / 16,384（50% 网格）；t32 ds-v3：100%（c_e=1 全 leader）；t1：100%（direct 无分组）。非 leader 块 = 标量 prologue + 退出 ≈0.56–1.2M cyc ≈**7–15%** 墙钟开销（两实例合计口径：22,528 块 / 28 核 = 804.6 块/核 × 0.7–1.5K cyc，D-2 no-op 行——单实例拆分 gate/up ≈3.9–8.2% / down ≈13–29%；t32/t1 为 0%，接受——瓶颈之下的次级项，C1 消除 no-op 的代价是 W 瓶颈翻倍，R3/R4 已淘汰）。

### 5.3 约束分析

1. **C0 分形对齐**（`CONST-gemm-dim-constraints`）：M=16 ✓（最小 16 倍数）；N_gemm ∈ {64,128,256} ✓ 16 倍数；K=block_k ∈ {64,128,256} ✓ C0=16 整除；ND2NZ 行 16 整除 + 行 32B 对齐——on-grid 直拷天然满足（整 tile），off-grid 经 UB 中转满足（整 tile 上 L1，§0.6 R3）；大 bk（256）由 `gemm_mad.py` `_compute_tile_k_sub`（L206–222）自动 K 子切分兜底。
2. **L0C 256KB / L1 512KB / UB 216KB**：见 §4.5（峰值 6.3% / 28% / 37%）。
3. **wrapper 预算 fit**：全部默认配置无需降级（§4.5 复核表）。
4. **非整除（off-grid）**：`rb = clamp(n − nb·bn, 0, bn)`（ceildiv 性质保证 nb·bn < n → rb ≥ 1，gated 两半同式）；`kb = min(k − ko·bk, bk)`；A/W 尾零填 + GEMM 恒 K=block_k（E4）；epilogue `col < n` 掩码。

### 5.4 注意事项

- grid 与 n_blocks 为工厂期 Python int（非 TIR 表达式）——`T.MixedKernel` grid 参数直接可求值。
- route-major pid 顺序不可改为 n_block-major（L2 局部性，§1.6.3 #6）。
- `weights` 首轴 `expert` 为运行时索引（GM 基址）——deepgemm `matrix_region` 先例（`common.py` L34–38 运行时 batch_idx 的 GM→L1）。

### 5.5 分核策略（grid 划分 + AIC/AIV 分工 + 同步策略）⭐

**① grid 划分（含核数依据）**

- grid = `R × n_blocks`（§5.2 表；一维 `T.MixedKernel(grid, sids=2)`）。
- **核数实查**（`CONST-arch-a5`，measured，2026-10-16，本机 `Ascend950PR_9579`）：

```python
import torch
props = torch.npu.get_device_properties(torch.npu.current_device())
props.cube_core_num, props.vector_core_num   # → (28, 56)
```

- waves ≥ 9（§5.2 表，最小 = ds-v3-t1 gate/up 9.1 waves）——并行度充足；28 = 4×7 非 2 幂，本设计 grid 无整除性要求（硬件 block 分发）。设计期核数按实测 28/56；部署到其他 A5 卡时核数由运行时查询决定（grid 不依赖核数——block-per-tile 映射，无 persistent 化）。
- 每 mixed block = 1 AIC + 2 AIV（AIC:AIV = 1:2 物理编组，`CONST-arch-a5`）→ 并发 block 数 = 28。

**② AIC/AIV 分工（sid guard 划分 + 拷贝所有权）**

| 子核 | 执行域 | 工作 | 拥有的拷贝/缓冲 |
|------|--------|------|-----------------|
| **AIC**（公共段，sid guard **外**） | MTE2 + Cube | on-grid W `T.copy` GM→L1；`T.gemm`（NT，fp32 L0C 累加） | w_l1 写者（on-grid）、res 写者 |
| **AIV0**（`if sid == 0:`） | MTE2/MTE3 + SimtVF | 行元数据 staging（私有副本）；K 循环内 gather → `a_ub` → `T.copy(a_ub, a_l1)`；off-grid W staging（`T.fill` + 钳制 GM→UB + `T.copy(w_ub, w_l1)`） | a_ub、w_ub、st_ub/ss_ub（副本 1）、a_l1 写者（off-grid w_l1 写者） |
| **AIV1**（`if sid == 1:`） | Fixpipe + SimtVF | 私有 staging（副本 2）；`T.copy(res, acc_ub)`（L0C→UB，guard 路由到 AIV1）；silu/cast/掩码散写 | acc_ub、st_ub/ss_ub（副本 2） |

- **sid 划分原则**：公共 Cube 段置于 sid guard 之外（「Keep common Cube work outside sid guards」）；`is_leader` 块级守卫在 sid 守卫之外（三核同构求值——子核一致，满足独立 AIV 退出规则）。
- **UB 所有权**：`alloc_shared` 在 mixed kernel 内为 **per-AIV 私有**（synchronization.md：「UB and sub-core-dependent scalar state are local to each AIV; L1, L0, and GM storage remain common」）——st/ss 双副本即由此而来（AIV1 不可读 AIV0 的 a_ub/st_ub）；AIC 不写任何 UB。
- **AIV1 K 循环期空闲**：有意设计（epilogue 依赖 res 累加完成，物理上只能在 K 循环后执行；gather 分摊到 AIV1 的墙钟收益 ≈ 0——D-2：epilogue ≈0.1–0.2M cyc vs 瓶颈 8M）。
- **fallback A（若 guarded L0C→UB 或 SimtVF-under-guard lowering 受阻）**：`T.dual_copy` M-split 半分 epilogue（gemm_silu L97–105 拓扑）：公共段 `T.dual_copy(res, acc_ub_half)`，两 AIV 各持 `[8, N_gemm]` 半块、行映射 `mi = sid·8 + i`（N_split 需 32 整除：N_gemm ∈ {64,128,256} ✓）——E5 已验证行映射语义；此时 gather 可改由两 AIV 各采 8 行（M-split 同理）。

**③ 同步策略**

- **`auto_schedule: yes`**（声明；不设 `TL_ENABLE_AUTO_SCHEDULE: False`，无手动 `T.ascend_set_flag`/`T.ascend_wait_flag`）。
- 数据依赖边界（AutoSchedule 自动插 flag + multibuffering，§7.2 逐边表）；手动 flag 预算 = **0**（不适用）。

## 6. 循环与调度结构

### 6.1 循环结构总结

| 循环 | 结构 | 域 | 说明 |
|------|------|-----|------|
| K 主循环 | `for ko in T.Pipelined(k_blocks, num_stages)` | 公共（AIC + AIV0 体侧） | k_blocks ∈ [8, 56]；soft-pipeline 承接源 K 流水（§0.5 条目 9） |
| 行循环（gather/staging/epilogue） | `for mi in T.serial(16)` | AIV0 / AIV1 | mi 外提 → div/mod 与 metadata 读每行一次（OPT-2） |
| 元素循环（gather） | `for ki in T.Parallel(block_k)` | AIV0 SimtVF(threads=128) | 行内连续 128 元素（256B fp16）|
| 元素循环（epilogue） | `for j in T.Parallel(block_n)` | AIV1 SimtVF(threads=128) | out 行内连续 64/128 元素 |
| 无 `T.Persistent` | — | — | grid-driven（waves ≥ 9，§5.5①）；persistent 化在 R4 已淘汰（C6） |

### 6.2 循环伪代码

```text
with T.MixedKernel(R * n_blocks, sids=2) as (pid, sid):
    <公共标量 prologue>                                  # T.serial 语义（标量）
    if is_leader != 0:
        if sid == 0: <staging: T.serial(16) × 守卫标量读>
        for ko in T.Pipelined(k_blocks, num_stages):     # K 软流水
            if sid == 0:                                 # AIV0
                with T.SimtVF(threads=128):
                    for mi in T.serial(16):
                        for ki in T.Parallel(block_k):   # gather（掩码 + 零填）
                T.copy(a_ub, a_l1)
                [off-grid: T.fill(w_ub,0); 钳制 copy; T.copy(w_ub, w_l1)]
            <on-grid: T.copy W GM→L1（×2 gated 段）>       # AIC（公共）
            T.gemm(a_l1, w_l1, res, transpose_B=True, clear_accum=(ko==0))
        if sid == 1:                                     # AIV1（K 循环后）
            <staging 副本>; T.copy(res, acc_ub)
            with T.SimtVF(threads=128):
                for mi in T.serial(16):
                    for j in T.Parallel(block_n):        # silu/cast/掩码散写
```

### 6.3 流水线优化

- **逐 k 步三引擎重叠**（on-grid，ds-v3-t64 gate/up 量级，D-2）：AIC W 装载 ≈320 cyc（32KB @ 100 B/cyc，**节拍器**）；AIV0 gather ≈150 cyc + UB→L1 ≈16 cyc（隐藏）；Cube MAC m16n128k128 ≈64 cyc（隐藏）。`num_stages=2/3` 让 AutoSchedule 对 a_l1/w_l1/a_ub multibuffering，装载与计算跨 k 步重叠。
- **epilogue 与 K 循环**：物理上串行（res 需累加完成）；其 ≈84 cyc/块 的成本相对块预算（≈18.4K cyc）可忽略（D-2）；跨块并行由多 block 覆盖（585+ waves）。
- **源意图承接**：`TL_DISABLE_WARP_SPECIALIZED`（源 L94 注释：A tile 在 K 循环内重写）→ A5 由 AutoSchedule multibuffering 版本管理承接（§0.5 条目 9），无需手动干预。

### 6.4 尾块处理

| 尾区 | on-grid（manifest 全命中） | off-grid（测试形态） |
|------|---------------------------|----------------------|
| N 尾（n % bn ≠ 0） | 不存在 | `rb = clamp(n − nb·bn, 0, bn)` 钳制拷贝 + W 零填；epilogue `col < n` 掩码 |
| K 尾（k % bk ≠ 0） | 不存在 | `kb = min(k − ko·bk, bk)`；W 尾列零填 + A 尾列零填（gather else）→ GEMM 恒 K=block_k，尾贡献 0 |
| M 尾（route_count < 16） | 运行时（metadata 依赖） | 同左：gather 行零填（OPT-3）+ epilogue `mi < route_count` 掩码（E3） |
| gated up 半段 N 尾 | 同 N 尾（两半各 n 行、同偏移 → rb 同式） | 同左 |

## 7. 同步策略

### 7.1 同步模式

**`auto_schedule: yes`**——依赖 AutoSchedule 自动同步（AGENTS.md 默认；gemm_silu / deepgemm 同模式）。无手动 `T.ascend_set_flag`/`T.ascend_wait_flag`/`T.ascend_pipe_barrier`；`pass_configs` 不携带 `TL_ENABLE_AUTO_SCHEDULE: False`。

### 7.2 同步点说明（AutoSchedule 自动插 flag 的数据依赖边）

| # | 依赖边 | 类型 | 机制 |
|---|--------|------|------|
| 1 | `a_ub`（AIV0 gather 写）→ `T.copy(a_ub, a_l1)`（AIV0） | 同 AIV RAW | 编译器顺序（同子核内程序序） |
| 2 | `a_l1`（AIV0 写）→ `T.gemm`（AIC 读） | **跨子核 RAW**（L1 公共） | AutoSchedule flag（"fixed" 形态被测核心边） |
| 3 | `w_l1`（AIC on-grid 写 / AIV0 off-grid 写）→ `T.gemm`（AIC 读） | RAW（L1） | AutoSchedule flag + Pipelined 版本管理 |
| 4 | `res`（AIC gemm 跨 ko 累加链） | L0C 原地累加序 | `clear_accum=(ko==0)` + AutoSchedule（gemm_mad 同形） |
| 5 | `res`（AIC 写）→ `T.copy(res, acc_ub)`（AIV1 读） | **跨子核 RAW**（L0C→UB Fixpipe） | AutoSchedule flag（deepgemm epilogue 同机制，`unit_flag_ctrl` 由编译器注入） |
| 6 | staging（AIV1 写 st/ss）→ epilogue 读 | 同 AIV RAW | 编译器顺序 |
| 7 | Pipelined multibuffering（a_ub/w_ub/a_l1/w_l1 各 ns 版本） | 版本计数 | AutoSchedule counter/epoch（skill §7） |

- **无跨块依赖**：每 out 行恰被一个 block 写（leader 写组行 / 重复 route 写自身 / 非 leader 退出，§4.2）→ 无 GM 级跨块同步需求。
- **sub-core 一致性**：`is_leader` 守卫对三核求值相同（同 GM 数据、确定性整数算术）→ 独立 AIV 退出规则满足（synchronization.md）；sid guard 内部无跨 AIV UB 读写（st/ss 双副本设计正是为此）。

### 7.3 pass_configs 配置

```python
@tilelang.jit(out_idx=[], compile_flags=["-O3", "-DENABLE_BF16"])   # 无 pass_configs
```

- 源的 `TL_DISABLE_WARP_SPECIALIZED: True` **不迁移**（A5 无 warp specialization 概念；A tile 重写的版本管理由 AutoSchedule multibuffering 承接，§0.5 条目 9）。
- auto_schedule 保持默认开启；若 Stage 3/4 需要诊断性关闭（定位同步问题），须另行评估——本设计无此计划。

## 8. 验证方案

### 8.1 Golden 函数

**kernel 级 golden（独立于 NPU 实现——不消费 metadata、不复刻分组/直除式/tile 结构）**，语义依据 §0.1（逐 route 独立计算）：

```python
def golden_indexed_expert_gemm(a, weights, expert_ids, out, *, activation, route_input):
    """§0.1 语义的 per-route 参考实现：fp32 域、逐 route 独立、不读 route_metadata。
    分组不变性（E2）保证 golden 无需消费 metadata——任意自洽双射下逐位一致。"""
    T, K = expert_ids.shape
    af, wf = a.float(), weights.float()
    y = torch.empty((T, K, wf.shape[1]), dtype=torch.float32, device=a.device)
    for t in range(T):
        for s in range(K):
            row = af[t, s] if route_input else af[t]          # route_input 两形态
            y[t, s] = row @ wf[int(expert_ids[t, s])].t()      # [N,k] 行主 → y·Wᵀ
    if activation == "silu_and_mul":
        gate, up = y.chunk(2, dim=-1)                          # 列序 = golden chunk(2)（E6）
        o = gate * torch.sigmoid(gate) * up                    # torch.sigmoid（独立于 kernel 直除式，E1 等价）
    else:
        o = y
    out.copy_(o.to(out.dtype))                                 # fp32 域计算、落存 cast（§0.1）
```

- **独立性论证**：golden 按源语义逐 route 循环（无分组、无 leader、无 tile、无 staging、sigmoid 形式），与 NPU 实现零结构共享；`route_metadata` 形参不读（分组不变性，E2 机器验证 + `_route_stats` DESIGN §0.1 契约）。
- **op 级集成 golden**：`ref_routed_experts`（`examples/TileOPs/workloads/moe.py` L58–91，移植自源仓：fp32 逐专家 gathered matmul + silu_and_mul + 加权累加）——覆盖 `route_stats → gate/up → down → weighted_reduce` 全链。
- **metadata 生成**（测试侧，满足 `_route_stats` 三段契约）：L0/L1 用 synthetic builder（counts 精确 / 每专家 rank 双射 / lists 配对 / rank<T 守卫 / stable 序）；L2 op 级用已迁移的 `_route_stats` kernel 真实产出。

### 8.2 精度标准

- `torch.testing.assert_close`，golden 在 NPU 上以 fp32 计算（`device="npu"`）。
- 容差按输出 dtype（AGENTS.md 约定）：**fp16/bf16 → rtol=atol=1e-2**（K ≤ 7168 的 fp32 累加 + fp32 域激活后单次 cast——非 accumulation-heavy 输出路径；gemm_silu 同口径 1e-2）。
- NaN/Inf 语义：合法输入（有限值）下输出应有限；`expert_ids` 越界为未定义行为（调用方契约，不测）。
- 比较须 **NaN-aware**（`torch.testing.assert_close` 默认 equal_nan=False 已满足；自定义比较禁止 `max(0.0, nan)` 形式——verify_equiv.py 开发中已修复该缺陷）。

### 8.3 L0 门槛测试计划（Stage 3 首跑必过）

| # | 用例 | 配置 | 判定 |
|---|------|------|------|
| L0-1 | **编译冒烟**：16 个 manifest 特化全部 lower+compile | 4 workloads × {fp16, bf16} × {gate/up, down}（§5.2 配置表） | 编译通过、无容量报错 |
| L0-2 | **主路径正确性**：ds-v3-t32 fp16 两实例 + 均衡路由 + synthetic metadata | grouped、on-grid | assert_close 1e-2 |
| L0-3 | **direct 模式**：ds-v3-t1 两实例（占位 metadata） | direct、on-grid | assert_close 1e-2 |
| L0-4 | **off-grid**：H=100/F=96（`test_dims_off_the_tile_grid` 形态）两实例 | 钳制+零填路径 | assert_close 1e-2（E4 担保） |
| L0-5 | **bf16 冒烟**：ds-v3-t32 bf16 gate/up | `-DENABLE_BF16` | assert_close 1e-2 |

**L1（全路由模式矩阵，7 模式 × 4 shape 组合，与 verify_equiv E2/E3 同源）**：均衡 / 单专家垄断 / Zipf 偏斜 / 重复 route（rank≥T duplicate-fallback）/ 稀疏（多数专家 counts=0）/ 随机合法 / 退化（E=1、K=1）；每模式 × stable / random 双射 metadata——验证分组不变性 + 掩码充分性（pad 行 NaN 金丝雀注入）。

**L2（op 级集成）**：移植 `examples/TileOPs/tests/ops/test_indexed_expert_mlp.py` 用例集（`test_manifest_shaped_call`、`test_dims_off_the_tile_grid` 等）——`_route_stats` 真实 metadata + `ref_routed_experts` op 级 golden + 全链（gate/up → down → weighted_reduce）。

## 9. 风险点与注意事项

### 9.1 已知约束（技术约束检测结论）

| 约束 | 处置 | 状态 |
|------|------|------|
| 一维 Kernel、`threads=` 禁用（TypeError） | `T.MixedKernel(grid, sids=2)`；wrapper 已移除 threads（K3/K9） | 已消解（§2.3） |
| C0 分形对齐（M/N 16 倍数、K C0 整除、ND2NZ 行 16 整除/32B 对齐） | M=16/N_gemm∈{64,128,256}/bk∈{64,128,256}；off-grid 经 UB 中转 + 整 tile 上 L1 | 已消解（§5.3） |
| L0C/L1/UB 容量 | 峰值 6.3%/28%/37%（§4.5） | 已消解 |
| GM→L1 `T.copy` 无越界语义（源 OOB 不可移植） | on/off-grid 双路径 + 钳制零填（§0.6 R3，E4-probe 承重证明） | 已消解 |
| `T.select` 不存在（import 实查） | 全部用 if/else 块 | 已消解（§3.4） |
| INT8 MMAD / fp64 不支持 | 不涉及 | n/a |

### 9.2 常见错误（skill 陷阱对照）

| 陷阱 | 本设计防御 |
|------|-----------|
| 手写 NZ 分形 staging（禁止——InsertNd2Nz 自动插入） | 全部 `T.copy` GM→L1 直拷，无手写分形（§3.3） |
| 标量 tanh 形式 silu（实测 1.80× 劣化） | 直除式 `g/(1+exp(−g))`（OPT-1 探针裁决） |
| 假设 32 核（惯例值） | 实查 28 AIC/56 AIV（`CONST-arch-a5` measured；grid 不依赖核数） |
| UB 按 256KB 预算 | 216KB（SimtVF 32KB 扣减）口径（§4.5） |
| `max(0.0, nan)` 吞 NaN 的比较缺陷 | NaN-aware 比较（§8.2） |
| 跨 AIV UB 读（UB per-AIV 私有） | st/ss 双副本设计（§5.5②） |
| 手动 flag 不配对 | auto_schedule: yes，0 手动 flag（§7） |

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| T=1（direct） | Python 工厂级不发 metadata 代码；占位 (1,) 张量不读；route_count≡1 统一 tile 路径 |
| 重复 route（rank ≥ T） | duplicate-fallback：is_leader=1、route_count=1、自身行（源语义逐字，E2d） |
| 空专家（counts=0） | 无 block 引用（该专家无 route → 无 leader 块）——天然不触 |
| 奇 T（如 T=17，lists 段尾非 16 对齐） | 逐元素守卫 staging 天然 in-bounds；16-int 批量拷贝已拒绝（§1.6.2 #6） |
| 极端偏斜（单专家垄断） | 语义不变（E2 单专家模式）；性能退化 ∝ 活跃专家数下降（W 流量减少、并行度下降——数据依赖性，§4.6） |
| exp 溢出/±0.0/NaN 传播 | E1 角点验证：`exp(−g)` 上溢 → `g/inf = 0` ✓、下溢 → `g/1` ✓、±0.0 与 sigmoid 形式一致、NaN 按源传播 |

### 9.4 残余风险（Stage 3 首跑关注点，按序）

| # | 风险 | 等级 | 依据与缓解 | fallback |
|---|------|------|-----------|----------|
| 1 | **SimtVF 区域位于 sid guard 内** 的 lowering 组合无直接正向测试（"fixed" 用 SimdVF；test L986 的 SimtVF 在无 guard 的 Pipelined 内） | 中 | 结构上 = 子核投影 + VF 线程域正交（ss 测试 L161 证明 SimtVF+MixedKernel 结构合法）；L0-2 首跑即验证 | (a) SimdVF 按行 `T.simd.vld`（bk=128 fp16 = 恰一行 2048-bit）；(b) dual_copy M-split 重构（gemm_silu 拓扑，VF 无 guard） |
| 2 | guarded `T.copy(res, acc_ub)`（L0C→UB 路由到 AIV1） | 低 | "fixed" L236 + deepgemm `epilogue.py` L279–286 双先例（E5 已验行映射） | `T.dual_copy` M-split（fallback A） |
| 3 | M=16 的 L1-input `T.gemm`（examples 多用 M≥64 tile） | 低 | `CONST-gemm-dim-constraints`：M 须 16 倍数——16 = 最小合法值；`gemm_mad.py` 无 M 下限特判 | A tile pad 到 M=32（零行）——W-bound 下墙钟中性（D-2） |
| 4 | `T.alloc_var` 可变标量在 MixedKernel 下的赋值 lowering | 低 | 源 L111–127 逐字模式（GPU 实证）；标量在公共 prologue（guard 外） | 嵌套 if 重构（消除可变量） |
| 5 | 运行时 extent `T.copy`（rb/kb 钳制 GM→UB） | 低 | deepgemm 动态区域先例 + skill §6.1 字节粒度行；L0-4 首跑即验证 | GM→UB 改逐元素 SimtVF 装载（慢但正确，仅 off-grid 测试形态） |
| 6 | AutoSchedule multibuffering UB 膨胀系数未知（`CONST-capacity-a5` 待实测项） | 低 | manifest 峰值 7.5%（8–13× 余量）；L0-1 编译冒烟把关 | 降 num_stages（wrapper fit 循环既有机制） |
| 7 | AIV1 K 循环期空闲（设计如此） | 信息 | D-2：epilogue 0.1–0.2M cyc vs 瓶颈 8M，分摊收益 ≈0 | 无需（Stage 4 可复核 msprof_detail） |

## 10. 交付清单

### 10.1 目录结构

```
examples/indexed_expert_mlp/_indexed_expert_gemm/
├── DESIGN.md            # 本文档（Stage 1 交付）
├── verify_equiv.py      # D-1 等价性机器验证脚本（12/12 EQUIV_PASS，2026-10-29）
├── _indexed_expert_gemm.py   # Stage 3：kernel 实现（工厂名逐字保留）+ golden
└── RETROSPECTIVE.md     # Stage 1 复盘（本文件，随阶段追加）
```

### 10.2 文件清单

| 文件 | 阶段 | 内容 |
|------|------|------|
| `DESIGN.md` | 1 | 本设计文档（§0 迁移分析 + §1–§10 设计） |
| `verify_equiv.py` | 1 | E1–E6 等价性验证（torch CPU fp64 参照 + 角点 + NaN-aware） |
| `_indexed_expert_gemm.py` | 3 | `@tilelang.jit(target="ascend")` 工厂（§3.3 伪代码蓝图）+ golden（§8.1）+ L0–L2 测试（§8.3） |
| `RETROSPECTIVE.md` | 1+ | 复盘（价值点蒸馏输入） |

### 10.3 命名规范

- 工厂函数名 `_indexed_expert_gemm` **逐字保留**（`@tilelang.jit` 工厂名 = 迁移契约）；`_kernel(block_n, block_k, num_stages)` 三键 config 面（threads 已移除，K3/K9）。
- kernel docstring 声明 `programming_mode: mixed, auto_schedule: yes`。
- 两个实例化名保留：`IndexedExpertGateUpKernel` / `IndexedExpertDownKernel`（wrapper 层，A5 已脚手架化）。

### 10.4 实现顺序（Stage 3 指引）

1. on-grid grouped 主路径（gate/up 实例，ds-v3-t32 fp16）→ L0-2；
2. direct 模式 + down 实例 → L0-3 / L0-5；
3. off-grid 双路径（钳制 + 零填）→ L0-4；
4. 16 特化编译冒烟 → L0-1；
5. L1 路由模式矩阵 + NaN 金丝雀；
6. L2 op 级集成（`test_indexed_expert_mlp.py` 移植 + `_route_stats` 真实 metadata + `ref_routed_experts`）。
