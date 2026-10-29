# _logsumexp_kernel_tiled 算子设计文档（GPU TileOPs → Ascend A5 迁移，migration-harness）

> **任务上下文**：迁移 TileOPs `LogSumExpFwdOp` 的多 tile 变体 `_logsumexp_kernel_tiled`（GPU/CUDA 目标 TileLang → tilelang-fork ascend 后端，A5/950，dav-3510）。源算子三问解读对象 = `examples/TileOPs/tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py` 中的 `_logsumexp_kernel_tiled`（Stage 0 抽取件，自包含，L101–189）；wrapper 侧 GPU 上下文（Kernel class / dispatch / tile_n 启发式）= 同目录 `logsumexp.py`。
> **职责边界**：本设计只覆盖 **tiled 变体**（N 超驻留上限的 online 流式路径：N 向 chunk 串行 + running (max, sum) flash 合并）。单 tile 变体 `_logsumexp_kernel_single`（N 一次驻留片上）已由独立任务完成全流程（`examples/logsumexp/_logsumexp_kernel_single/`，DESIGN.md 已冻结、Stage 3 PRECISION_PASS）——per-chunk 计算体（masked 载入 / reduce_max / exp 原位 / reduce_sum / lse+cast / 两段写）按其 Stage 3 复盘结论**原样复用**，本设计的增量在「N 向 chunk 串行 + flash 合并 + running stats 驻留」。
> **调用面契约（Stage 5 集成对接）**：wrapper `LogSumExpKernel` 经 `_logsumexp_kernel(M, N, dtype, tile_n)` 选中本函数（tile_n ≠ 0）后以 `_logsumexp_kernel_tiled(M, N, dtype, tile_n)(block_m)(x)` 调用——A5 重实现保持该签名，`threads` 参数按 K3 适配规则移除（见 §4.7）。

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

**数学语义**（逐行 log-sum-exp，数值稳定形式——与 single 变体同一公式）：

$$
y_i \;=\; \mathrm{logsumexp}_j(x_{ij}) \;=\; m_i + \log\Big(\sum_{j=0}^{N-1} \exp(x_{ij} - m_i)\Big), \qquad m_i = \max_j x_{ij}
$$

tiled 变体将整行求和改写为 **online 递推**（分块 running 统计量 + flash 形式重缩放合并，源码 L135–178）：

$$
\text{init: } m^{(0)}_i = -\infty,\; s^{(0)}_i = 0; \qquad
\text{chunk } c:\; \begin{cases} p = m^{\text{run}}_i \\ m^{\text{run}}_i \leftarrow \max(m^{\text{run}}_i,\; m^{c}_i) \\ e^c_{ij} = \exp(x^c_{ij} - m^{\text{run}}_i) \\ s^c_i = \sum_j e^c_{ij} \\ s^{\text{run}}_i \leftarrow s^{\text{run}}_i \cdot \exp(p - m^{\text{run}}_i) + s^c_i \end{cases}; \qquad
y_i = m^{\text{run}}_i + \log s^{\text{run}}_i
$$

与解析式数学恒等（log-sum-exp 递推的代数变形；每次合并的重缩放因子 exp(p − m_run) 与块内 exp(x − m_run) 的指数参数均 ≤ 0，无溢出——flash 形式特征，见边界语义行）。

**语义要素表**（迁移必须原样保持的部分）：

| 要素 | 结论 | 源码证据 |
|------|------|---------|
| 数学映射 | 行级 logsumexp：`y[i] = row_max[i] + log(row_sum[i])`，row_sum 经 N 向 chunk 的 running (max, sum) 合并得到 | `_log_sum_exp_fwd_kernels.py` L180–185 |
| I/O 契约 | 输入 `x: (M, N)` 行主连续、dtype ∈ {float16, bfloat16, float32}（op 层 `_validate` 允许三者；manifest workload 为 fp16/bf16）；输出 `y: (M,)` 同 dtype | kernel 签名 L120–123；`logsumexp.py` `_validate` |
| 规约语义 | 沿最后一维（N）整行归约；**累加顺序 = chunk 内 `T.reduce_sum` 树形 + chunk 间按 chunk 序的 flash 重缩放顺序合并**（与 single 变体的单遍全行树形不同——fp32 域下差 ulp 级，verify_equiv.py V1 实测位级 / V3 ≤1 ulp）；**中间计算全程 fp32**（`tile_f32` fragment、fp32 的 row/tile 统计量），输出 cast 回输入 dtype | L127–133、L147–178 |
| dtype 语义 | 输入 dtype → fp32 中间（`T.cast(x[...], "float32")`）→ 输出 dtype（`out_local` 为输入 dtype fragment，`T.copy` 写回） | L147–162、L181–185 |
| 边界语义 | ① **NaN 逐元素传播**：任一 chunk 含 NaN → m_c=NaN → m_run=NaN → e=NaN → s_run=NaN → y=NaN；② **全 −inf 行 → y=NaN**（每个 chunk 的 tile_max=−inf，m_run 恒 −inf → 首个 chunk 即 `exp(−inf−(−inf))=NaN` 污染 s_run——源算法固有行为，与 single 变体一致、与 `torch.logsumexp` 的 −inf **分歧**，见 §9.3）；③ **含 +inf 行 → y=NaN**（inf 位置 `exp(inf−inf)=NaN` 污染 s_c → s_run；torch 返回 +inf，同上分歧）；④ **部分元素 −inf 的正常行：完全中性**（chunk 全 −inf 时 tile_max=−inf、m_run 保持有限 → `exp(−inf−finite)=0` → s_c=0，s_run 不变；−inf 散布时 exp(−inf−m)=0 中性——两种分布均验证，verify_equiv.py V5）；⑤ **跨 chunk max 间隙无溢出**：合并式全部 exp 参数 ≤ 0（p ≤ m_run、x^c ≤ m_c ≤ m_run），fp16 域极端间隙 131008（=2×65504）下朴素 m_c 基合并会溢出（`TRAP-NAIVE-ONLINE-MERGE`：间隙 > 88.7 即 fp32 溢出），源式 flash 形式安全——V5 corner `max-gap 131008` 实测位级正确；⑥ M 尾块（M % block_m ≠ 0）行 OOB 以 −inf 载入、不写出（masked 分支行谓词 L152；快路径依赖 GPU `T.copy` OOB 保护——见 §0.5） | L150–157、L164–178；V5 实测 |
| 尾块/非整除 | **N 维**：`N_padded = align_up(N, 256)`、`num_tiles = ceil(N_padded/tile_n)`、`_needs_mask = num_tiles·tile_n > N`——仅最后一个 chunk 可能越界（前面 chunk 列范围 ≤ N_padded − tile_n < N 恒有效）；`_needs_mask` 时最后 chunk 走逐元素 masked 载入（列 j 有效、行 <M，越界填 −inf），其余 chunk 走 `T.copy` 快路径；`_needs_mask=False`（tile_n 整除 N_padded 且 N=N_padded）时全部走快路径。**M 维**：M % block_m ≠ 0 的尾块行为与 single 相同（masked 分支有行谓词；快路径无行掩码） | L111–114、L139–162 |
| host 侧语义 | `N_padded`/`num_tiles`/`total_cols`/`_needs_mask` 常量折叠在 builder 闭包内（JIT per-shape 编译的一部分）；`tile_n` 由 wrapper 启发式传入（`compute_tile_n`：256 倍数、预算封顶、整除偏好——契约参数，kernel 侧不可改）；wrapper 层负责 dim 归一化/transpose/contiguous/reshape 到 (M, N) 与输出 reshape——**不属于本 kernel 的契约** | L111–115；`logsumexp.py` L176–198、L235–251 |

**语义解读完成标准自检**：依据上表可直接写出与源算子数值一致的 golden（§8.1 的 `torch.logsumexp(x.float(), dim=-1).to(x.dtype)`，即源仓 `examples/TileOPs/tests/ops/test_softmax.py` L99 `ref_program` 直移植——与 single 变体共用算子族语义）；±inf 角点的源继承分歧与 max 间隙安全性已显式列出（§9.3 / V5）。

### 0.2 源算子输入输出（含 tiled 变体的契约 shape 空间）

**Kernel 签名**：`_logsumexp_kernel_tiled(M: int, N: int, dtype: str, tile_n: int)` → `_func(block_m)` → `(x) → y`（`@tilelang.jit(out_idx=[1])`；GPU 版 `_func(block_m, threads)`，A5 按 K3 移除 `threads`）。

**tiled 变体被 wrapper 选中的条件**（`examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp.py` L63–68 dispatch `tile_n != 0` + `_tile_n_for_block_m` L176–198，决定本 kernel 的契约 shape 空间）：

1. **大 N 路径**：`N_padded = align_up(N, 256) > MAX_SINGLE_TILE_COLS = 32512`（列数上限，`examples/TileOPs/tileops/kernels/reduction/_primitives.py` L51）——single 变体无条件的硬边界；或
2. **驻留超限路径**：`N_padded ≤ 32512` 但 `block_m × elem_bytes × N_padded > device_smem_budget()`（A5 侧 = **216KB**，`tileops.device` NPUBackend 返回 UB SimtVF 预算，与 `CONST-capacity-a5` 一致）——中 N × 大 block_m 组合；或
3. **显式 config 路径**：caller 显式传 `config["tile_n"]`（`logsumexp.py` L161–165 直接采纳，不重新校验预算）——任意 (M, N) 组合均可被显式路由到 tiled。

**tile_n 的产生方式**（`compute_tile_n`，`_primitives.py` L116–203）：256 倍数；预算封顶 `tile_n ≤ effective_budget // (block_m · elem)`（effective = min(216KB, 32512·block_m·elem)）；**优先返回整除 N_padded 的最大候选**（消尾 chunk 浪费），否则返回预算上限值（尾 chunk masked 由 kernel 兜底）。因此 wrapper 产生的 tile_n 是 256 的倍数，典型值 16384/25600/16640/13824/6912——**不保证 2 的幂**。

**default_config 行为**（`logsumexp.py` L200–233）：block_m ∈ {1,2,4,8,16} 遍历，tiled 路径下**优先最小化 N 向 chunk 数**（最大化 tile_n）——由于 `effective_budget ∝ block_m`，小 block_m 得到大 tile_n，故 tiled 落点偏好 **block_m=1**。实测全部 manifest/测试落点均为 block_m=1（下表）。

**manifest workload 落点**（`examples/TileOPs/tileops/manifest/reduction.yaml` L93–99，归约到 (M, N) 后；tile_n/block_m 按 wrapper 启发式逐 block_m 复算）：

| workload | 归约后 (M, N) | dtype | wrapper 启发式 (block_m, tile_n) | N 向 chunk 数 | 落入变体 |
|----------|--------------|-------|-------------------------------|--------------|---------|
| attn-weights-4k `[32,32,4096]` dim=-1 | (1024, 4096) | fp16 / bf16 | (16, 0=single) | — | single（已完成） |
| attn-weights-32k `[32,32,32768]` | (1024, 32768) | bf16 | **(1, 16384)** | 2（整除） | **tiled（本设计）** |
| lm-head-logits `[4,102400]` | (4, 102400) | fp16 / bf16 | **(1, 25600)** | 4（25600×4=102400 整除） | **tiled（本设计）** |
| 3d-multidim-reduce `[4,128,4096]` dim=[0,2] | (128, 16384) | fp16 | (4, 0=single) | — | single（已完成） |

> wl2 tile_n=16384 推导：bm=1 时 effective=min(216KB, 32512×1×2B)=65024B → tile_n_max=32512，整除 32768 的最大 ≤32512 的 256 倍数 = 16384（2 chunks ≤ ceil(32768/32512)=2，整除偏好接受）；bm∈{2,4} 同样得 16384；bm=8 得 13824（3 chunks）、bm=16 得 6912（5 chunks）——block_m=1 的 2 chunks 最优。wl3 tile_n=25600 推导：bm=1 tile_n_max=32512，整除 102400 的最大 ≤32512 的 256 倍数 = 25600（102400=400×256，400 的 ≤127 最大因子为 100）；bm∈{2,4} 同；bm=8 得 12800（8 chunks）——block_m=1 的 4 chunks 最优。

**精度用例落点**（`examples/TileOPs/tests/ops/test_softmax.py`）：`(4, 32768)` fp16/bf16（L76–77，"large-N triggers N-tiling path"）→ (1, 16384)；`(33, 33000)` fp16（L81，"multi-tile, masked loads"）→ (1, 16640)（N_padded=33280=130×256，130 的 ≤127 最大因子为 65 → 16640，2 chunks）；`(33, 32768)` fp16（L83，非对齐 M + 大 N）→ (1, 16384)。其余用例（N ≤ 300/4096 等）走 single。性能用例（`examples/TileOPs/benchmarks/ops/bench_softmax.py`）manifest 驱动四 workload，tiled 覆盖 wl2/wl3。

**契约空间总结**：M ≥ 1 任意（含 M < block_m、M % block_m ≠ 0）；N ≥ 1 任意（tiled 无 single 的 32512 上限——**任意大 N** 是本变体的存在理由；显式 config 也可路由小 N）；dtype ∈ {fp16, bf16, fp32}；block_m ∈ [1, 16] 常见（wrapper 启发式 tiled 落点偏好 1；显式 config 理论上任意 ≥1，SUB 机制兜底）；tile_n ≥ 1 任意（wrapper 产 256 倍数；显式 config 可任意——BLOCK_N clamp 机制兜底，§5.2）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解表**（源 GPU kernel 每条计算语句归入一步，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式 | 源码行 |
|------|------|------|------|-------------|--------|
| S0 | host 静态化：`N_padded = align_up(N, 256)`；`num_tiles = ceil(N_padded/tile_n)`；`total_cols = num_tiles·tile_n`；`_needs_mask = total_cols > N` | N, tile_n | 编译期常量 | chunk 划分与边界处理选择 | L111–115 |
| S1a | 快路径载入（t < num_tiles−1 或无 mask）：`T.copy(x[pid_m·block_m, t·tile_n] → shared_buf → tile_local)` 再 cast fp32 | x (GM) | tile_f32 (block_m, tile_n) fp32 | chunk 数据就位（全列有效） | L143–148、L158–162 |
| S1b | masked 载入（仅最后 chunk，`_needs_mask` 时）：`tile_f32[i,j] = if(行<M ∧ t·tile_n+j<N, cast(x), −inf)` | x (GM) | tile_f32 | chunk 数据就位 + 列/行边界中性化 | L149–157 |
| S2 | chunk 内行最大：`T.fill(tile_max, −inf)` + `T.reduce_max(tile_f32, tile_max, dim=1, clear=False)` | tile_f32 | tile_max (block_m,) fp32 | m^c_i | L164–165 |
| S3 | running max 合并：`prev_max = row_max; row_max = max(row_max, tile_max)` | row_max, tile_max | prev_max, row_max | m_run ← max(m_run, m_c) | L167–169 |
| S4 | 稳定化指数（**以合并后 max 为基**）：`tile_f32[i,j] = exp(tile_f32[i,j] − row_max[i])`（原位覆写） | tile_f32, row_max | tile_f32 ≡ e^c | e^c_ij = exp(x^c − m_run) | L171–172 |
| S5 | chunk 内行和：`T.reduce_sum(tile_f32, tile_sum, dim=1)` | tile_f32(e) | tile_sum (block_m,) fp32 | s^c_i | L173 |
| S6 | running sum 重缩放合并：`row_sum = row_sum·exp(prev_max − row_max) + tile_sum` | row_sum, prev_max, row_max, tile_sum | row_sum | s_run ← s_run·exp(p−m_run) + s_c | L175–178 |
| S7 | 循环出口 lse：`out_local[i] = row_max[i] + log(row_sum[i])`（fp32 算式，fragment 为输入 dtype） | row_max, row_sum | out_local (block_m,) | y_i = m_run + log(s_run) | L180–183 |
| S8 | 写回：`T.copy(out_local, y[pid_m·block_m])` | out_local | y (GM) (M,) | 输出 | L185 |
| S-init | 循环前初始化：`T.fill(row_max, −inf)`、`T.fill(row_sum, 0.0)`（每行块每 kernel 调用一次） | — | row_max, row_sum | m^(0)=−inf, s^(0)=0 | L135–136 |

**数据流图**（源 GPU 硬件视角，单 chunk 迭代）：

```
GM[x (M,N) dtype]
  --T.copy / masked load（chunk c 列窗口 [c·tile_n, +tile_n)）--> SMEM[shared_buf (block_m, tile_n)]
  --> register[tile_local → tile_f32 (block_m, tile_n) fp32]
  --> reduce_max(dim=1) --> tile_max (block_m,) fp32 ──┐
  --> register[row_max/prev_max (block_m,) fp32 跨 chunk 驻留] <--┘ max 合并（S3）
  --> 原位 exp(x − row_max) --> tile_f32 ≡ e^c --> reduce_sum(dim=1) --> tile_sum
  --> register[row_sum (block_m,) fp32 跨 chunk 驻留] <-- rescale 合并（S6）
  [chunk 循环 num_tiles 次，S1–S6 串行]
  --> out_local = cast(row_max + log(row_sum), dtype) (block_m,)
  --T.copy--> GM[y (M,)]
```

**循环与并行结构**（源）：grid = `T.ceildiv(M, block_m)` 个 block（`T.Kernel(grid, threads=threads)`，**二维参数 + threads**——A5 不支持，见 §0.5）；每 block 处理 block_m 行的**全部 N 向 chunk**（`for t in T.Serial(num_tiles)` chunk 串行——行块间独立、chunk 间顺序依赖 running stats）；chunk 内 `T.Parallel(block_m, tile_n)` 线程级并行；running 统计量（row_max/row_sum/prev_max）为 fragment，跨 chunk 循环迭代存活；无块间通信。

**host 侧逻辑**：`align_up`/`num_tiles`/`_needs_mask` 常量折叠（编译期）；`_neg_inf` 字面量；`threads` 由 wrapper 传入（GPU 侧契约）；JIT per-(M, N, dtype, tile_n, block_m, threads) 编译 + `lru_cache`。wrapper 侧 `tile_n` 在 Kernel 构造时烘焙（`self._tile_n`），`forward` 经 `_logsumexp_fwd_wrapped(...)(block_m)(x)` 调用。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 代码特征 | 目的 | 依赖的源硬件特性 | 层级 |
|---|----------|---------|------|-----------------|------|
| 1 | **online softmax 递推（flash 形式合并）** | running (row_max, row_sum) + `s_run·exp(prev−new) + s_c` 重缩放（S3/S6） | X 单遍 GM 读完成整行归约（vs 两遍法重读 X）；中间缓冲 O(tile)（vs O(N) 落 GM） | **无（纯算法层）** | 算法层 |
| 2 | N 向 chunk 分块流式 | `T.Serial(num_tiles)` chunk 串行 + per-chunk (block_m, tile_n) staging | 突破片上驻留上限处理任意大 N | SMEM 容量数值（块大小硬件相关，分块策略本身可移植） | 算法层（驻留策略） |
| 3 | chunk 级快慢双路径 | `_needs_mask` 时仅最后 chunk masked、其余 `T.copy` 快路径 | 全有效 chunk 免逐元素掩码开销 | GPU `T.copy` shared 256 元素对齐约束 + masked load 代价 | 优化层 |
| 4 | per-chunk SMEM staging | `shared_buf (block_m, tile_n)` 每 chunk 重填充 | GM→寄存器两级缓冲（coalesced bulk copy） | GPU SMEM + 异步拷贝 | 优化层 |
| 5 | 减 max 稳定化 + fp32 中间精度 | `exp(x − m)`、tile_f32/统计量 fp32 | 消 fp16 溢出；累加/重缩放精度 | 无（纯数学） | 算法层 |
| 6 | 树形归约 `T.reduce_max/reduce_sum` | dim=1 fragment 归约（tile_max/tile_sum） | chunk 内行统计量低延迟 | warp shuffle 树（TVM lower） | 优化层 |
| 7 | 合并式标量最小化 | S3/S6 的 per-row 标量合并（每行每 chunk 1 次 max + 1 次 exp + 1 mul + 1 add） | 重缩放开销 O(rows·chunks) 而非 O(N) | GPU 标量管线 | 优化层 |
| 8 | 行批 block_m + 非持久 grid | `block_m` 参数化；`ceildiv(M, block_m)` blocks | 摊薄 block 启动开销；行块直接映射 | GPU SM 调度器 | 优化层 |

**未识别更优手段的风险自检**：源码无 warp shuffle 手写、无 swizzle、无 cp.async/Pipelined、无 persistent、无哨兵钳位（全 −inf 行靠 NaN 传播语义而非钳位防护——与 pilot online_softmax 的 O6 哨兵不同，见 §1.6.0 R1 候选 11 的显式评估）；识别已覆盖全部显式结构。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（三层模型：语义层/算法层/优化层；四态处置：保留/等价替换/重新设计/舍弃。判定输入含 §1.6.0 调研结论——源算法与调研候选同台比较。）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（行 logsumexp、online 合并顺序、fp32 中间、NaN/±inf 边界行为） | 语义 | — | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1）；合并顺序保持 chunk 序 flash 形式（§0.1 规约语义） |
| online softmax 递推（#1） | 算法 | 无 | — | **保留** | 纯算法层（migration-analysis.md §5.3「online 算法/分块累积 → 保留」）；flash 形式已规避 `TRAP-NAIVE-ONLINE-MERGE` 的溢出发散角点（max 间隙；全 −inf chunk 的 NaN 在本算子是源固有语义而非缺陷）；调研确认 online 是超驻留大 N 语境的正确算法族（§1.6.0 R2/R3） |
| N 向 chunk 分块（#2） | 算法 | SMEM 容量数值 | 有（UB 216KB〔SimtVF〕重算 chunk 粒度） | **保留（粒度重算）** | chunk 粒度由 A5 UB/VRF 双预算重新推导（BLOCK_N pow2 机制，§0.6 R2）；任意大 N 的流式能力是本变体语义 |
| 减 max 稳定化 + fp32 中间（#5） | 算法 | 无 | — | **保留** | 纯数学（`ALG-softmax` R1 必选项）；fp32 合并链精度（V3 ≤1 ulp vs golden） |
| 树形归约 `T.reduce_max/reduce_sum`（#6） | 优化 | warp shuffle 树 | 有（`T.reduce_*` 自动降级 AscendAllReduce 跨线程树，fp32 硬件直连） | **等价替换** | pattern-library `D-REDUCE-SIMTVF-API`（measured，get_kernel_source 实见 `tl::AscendAllReduce<MaxOp/SumOp>`）；`examples/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py` L156/L159 同族实证（PRECISION_PASS） |
| 合并式标量最小化（#7） | 优化 | GPU 标量管线 | 有（SimtVF `T.Parallel(SUB)` 行维向量化） | **等价替换** | 源 `for i in T.Parallel(block_m)` 合并体 → A5 同构（`T.Parallel(SUB)`）；§1.6.2 #3/#6 |
| per-chunk SMEM staging（#4） | 优化 | GPU SMEM 两跳 | 有（`T.copy` GM→UB 单跳 + fragment 载入） | **等价替换** | `examples/pilot/online_softmax/online_softmax.py` L115（chunk 级 GM→UB `T.copy`）+ L123（`T.cast` fragment 载入）——streaming 路径完整同构先例 |
| running 统计量跨 chunk 驻留（register fragment row_max/row_sum/prev_max） | 优化 | GPU fragment 生命周期覆盖整个 kernel | **无直接等价**（A5 SimtVF 域 per-chunk 开闭——chunk 间有 MTE2 `T.copy`，fragment 不能跨域存活） | **重新设计** | → §0.6 R5：`stats_ub (SUB, 2)` fp32 UB 驻留 + SimtVF 域内读写（online_softmax streaming L104–111 `stats_ub` 同构先例，measured 63/63 PASS） |
| chunk 级快慢双路径（#3） | 优化 | GPU `T.copy`→SMEM 256 元素对齐 | **无此约束**（A5 GM→UB DMA 字节级行长；fragment 才需 pow2） | **重新设计** | → §0.6 R2：统一为「full chunk 无谓词直载 + 尾 chunk 静态 rem 切片 masked」——online_softmax streaming L113–152 同构（full chunk 直载 L123 / tail chunk masked L150–152）；single 变体 R2 同款决策（消除双路径维护） |
| `tile_n` chunk 粒度（256 倍数任意值） | 优化/契约 | GPU SMEM 对齐（256 元素） | **无此约束**（A5 fragment pow2 约束——非 pow2 tile_n 使 fragment TILE_C=next_pow2(tile_n) 翻倍、masked lane 白算 exp：tile_n=16640 → 32768 列 fragment 仅 50.8% 有效） | **重新设计** | → §0.6 R2：**BLOCK_N pow2 重划**（满足以下约束的最大 2 的幂：BLOCK_N ≤ tile_n 且 2·BLOCK_N·elem ≤ 194.4KB）+ 尾 chunk 静态 rem 切片；flash 合并对 chunk 划分满足结合律（V1 全 shape 位级 / V6 划分扫描 ≤1 ulp）——签名保留 tile_n（接口不变），内部粒度重解释为预算上限（后端实现相关参数适配，与 K3 移除 threads 同类） |
| `T.Kernel(grid, threads=threads)` block×thread 两级并行 | 优化 | CUDA 线程模型 | **无直接等价**（`T.Kernel` 一维、无 `threads=` 参数——TypeError） | **重新设计** | → §0.6 R1：一维 `T.Kernel(56)` + `T.SimtVF(threads=)` 线程域 + persistent 行块调度（ascend-constraints §1 第 1 条；single R1 同款） |
| 非持久 grid（ceildiv(M, block_m) blocks，#8） | 优化 | GPU 硬件调度器 | 有更契合形态（persistent + 多波摊薄） | **重新设计** | → §0.6 R3：`T.Persistent([num_tiles], 56, core_id, num_stages=2)` 行块持久调度（online_softmax L60 / single L138 先例）；#8 行批摊薄意图由 persistent 域 + 行子批承接 |
| M 尾块越界行为（快路径 `T.copy` 无行掩码，S1a/S8） | 实现 | GPU `T.copy` 隐式 OOB 保护（源仓 CI 通过的既成事实） | **未文档化**（A5 DMA 越界行为无保障） | **重新设计** | → §0.6 R4：静态 full/rem 分解 + TIR if 尾块分支 + 静态切片 copy + 行谓词（single R4 同款，Stage 3 已实证该嵌套组合编译运行通过） |
| host `threads` 契约 | 环境 | CUDA | 无（K3：A5 移除） | **舍弃（形态变更）** | wrapper 侧 threads 已在 Stage 0 适配中移除（`logsumexp.py` 模块头 L26–28）；kernel 内 `threads = min(1024, SUB·BLOCK_N)` 自推导（§5.2）。**意图承接**：GPU threads 表达的行内并行度由 SimtVF 线程域承接 |

**判定统计**：保留 4 项（语义 / online 递推 / chunk 分块策略〔粒度重算〕/ 稳定化）+ 等价替换 4 项（树形归约 / 合并式标量 / SMEM staging / 单跳搬运合并计）+ 重新设计 5 项（R1 并行结构、R2 chunk 粒度与双路径、R3 调度、R4 M 尾块、R5 running stats 驻留）+ 舍弃 1 项（host threads 契约，K3 形态变更，意图由 SimtVF 承接）。**无静默丢弃**：每个源优化意图均有承接结构或舍弃理由。

### 0.6 NPU 算法重设计

**重设计项 R1: 并行结构（block×thread 两级并行 → 一维 grid + SimtVF + persistent 行块调度）**

- **源方案**：`T.Kernel(ceildiv(M, block_m), threads=threads)`——block_m 行/block × threads 线程，非持久 grid；chunk 循环 `T.Serial(num_tiles)` 在 block 内。
- **NPU 新算法**：一维 `T.Kernel(NUM_BLOCKS=56)`（= `vector_core_num` 实查值，§5.5）；`for t in T.Persistent([num_tiles], 56, core_id, num_stages=2)` 行块持久调度（num_tiles = ceildiv(M, block_m)，多波由调度器分配）；元素并行在 `with T.SimtVF(threads=min(1024, SUB·BLOCK_N))` 线程域内以 `T.Parallel(SUB, BLOCK_N)` 表达；chunk 内行归约 `T.reduce_max/reduce_sum(dim=1, clear=True)`（AscendAllReduce 硬件树）。
- **语义保持论证**：行间完全独立（chunk 串行只发生在行块内部的 N 维）→ 行块映射与核数变化不影响数值；行内归约 warp shuffle 树 → asc_reduce 硬件树，fp32 累加分组变化引入树深级舍入差（V3 vs golden ≤1 ulp / max_rel 6.5e-8，远小于容差）；NaN/±inf 逐元素传播与 chunk 合并语义不变（V5 NaN 模式逐 corner 一致）。
- **依据**：`examples/pilot/online_softmax/online_softmax.py` L57–63（streaming 路径同构形态）；`examples/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py` L134–138（同族 PRECISION_PASS 实证）；`CONST-arch-a5`（measured，本机实查 28/56，2026-10-16 本设计执行期复测确认）。

**重设计项 R2: chunk 粒度与驻留布局（tile_n 256 倍数 chunk + N_padded 对齐双路径 → BLOCK_N pow2 chunk + UB 紧凑 + 尾 chunk 静态切片）⭐（本变体核心增量）**

- **源方案**：chunk 宽 = tile_n（wrapper 启发式的任意 256 倍数，如 16640/25600）；列尾以 N_padded = align_up(N, 256) 对齐（GPU SMEM 约束），仅最后 chunk masked；非 pow2 tile_n 下 A5 直译将使 fragment 形状 (block_m, next_pow2(tile_n)) 翻倍（16640 → 32768），masked lane 的 `exp(−inf−m)` 仍占 special_func 管线——有效 lane 仅 50.8%（25600 → 32768 为 78.1%），special 吞吐折半（special 是计算主导项，§1.6.0 D-2）。
- **NPU 新算法**：① host 静态化重划 chunk 序列：`BLOCK_N = 满足以下约束的最大 2 的幂：BLOCK_N ≤ tile_n 且 2·BLOCK_N·elem ≤ 194.4KB`（tile_n 降级为预算上限，fp16/bf16 上界 32768 / fp32 上界 16384——wrapper 预算下实际 ≤ 16384；显式 config 超大 tile_n 由该 clamp 兜底）；`num_full = N // BLOCK_N`、`rem = N % BLOCK_N`、`TILE_R = next_pow2(rem)`（rem>0 时）。② full chunk 全列有效：UB 紧凑 `x_ub (SUB, BLOCK_N)` + fragment **无谓词直载**（`T.cast(x_ub[r,i], "float32")`——谓词恒真干脆不写，省 select op/elem）；③ 尾 chunk（rem > 0，Python 静态分支）：静态 rem 切片 `T.copy(x[row_base:+rows, col:N] → x_ub[0:rows, 0:rem])` + fragment `(SUB, TILE_R)` masked 载入（`T.And(r < rows, i < rem)`，pad=−inf 双幂元）；④ 行子批 SUB 同 single 机制：`SUB = 满足以下约束的最大 2 的幂：SUB ≤ block_m ∧ 2·SUB·BLOCK_N·elem ≤ 194.4KB ∧ SUB·BLOCK_N·4B ≤ 128KB`——任意契约 block_m 下 UB 双版本与 VRF 同时达标；⑤ exp 原位覆写保持源结构（S4）。
- **语义保持论证**：flash 合并（max 基重缩放）对 chunk 划分满足**结合律**——每个元素恰计入一个 chunk、合并代数恒等，任意划分（含单 chunk 退化）数学等价；浮点下仅重缩放乘法的分组舍入差。**机器验证 V1**：7 组 (M, N, tile_n, BLOCK_N) 组合（含 tile_n=16640 vs BLOCK_N=16384、tile_n=25600 vs 16384、整除/非整除/tiny rem=4/N=32513）fp64 与 fp32 **位级一致（ulp=0）**；**V6** 划分敏感性扫描（BLOCK_N ∈ 256…16384 × tile_n ∈ 16640/25000/32512）全部 ≤1 ulp（max_rel 6.3e-8）；**V4** 源链 vs A5 链 fp16 输出位级一致 + NaN 模式一致。−inf pad 幂元中性同 single（V1 隐含覆盖，`D-REDUCE-SIMTVF-API` measured 佐证）。
- **收益**：非 pow2 tile_n（16640/25600）下 fragment 有效 lane 50.8%/78.1% → ~100%（full chunk 无谓词 + 尾 chunk pow2 TILE_R），special 计算项恢复满吞吐；GM 读消除 N_padded pad 列搬运（33000→33280：+0.85%；对齐场景无变化）；消除 `_needs_mask` 双路径（统一 full/tail 二分支静态化）。
- **依据**：`examples/pilot/online_softmax/online_softmax.py` `_build_streaming`（full chunk 直载 L123 + tail chunk 静态 rem 切片 L142 + masked L150–152——chunk 序列 pow2 重划 + 尾 chunk 静态化的完整同构先例，measured 63/63 PASS）；`D-REDUCE-SIMTVF-API`（掩码载入与 DMA 紧凑拷贝正交）；`TRAP-NAIVE-ONLINE-MERGE`（flash 形式的角点安全性——源式继承）；`CONST-capacity-a5` / `CONST-vrf-capacity-a5`。

**重设计项 R3: 调度（非持久 grid → persistent + chunk 串行 + 子批/chunk 双缓冲流水 + AutoSchedule）**

- **源方案**：每 block 一次性处理 block_m 行 × 全部 chunk 后退出（无块间流水概念）。
- **NPU 新算法**：外层 `T.Persistent([num_tiles], 56, core_id, num_stages=2)`；核内两级串行：行子批 `T.serial(num_sub)`（外）× N 向 chunk `T.serial(num_full)`（内）+ Python 静态尾 chunk / 末子批分支；`x_ub` 双版本（`T.annotate_buffer_versions({x_ub: 2})`）——chunk c+1 的 MTE2 预取与 chunk c 的 VALU 计算+合并重叠（**流水主体在 chunk 维**——N/BLOCK_N 次迭代是 tiled 的主循环维）；MTE2→VALU→MTE3 跨引擎依赖由 AutoSchedule 自动插 flag 对（§7）。
- **语义保持论证**：纯调度变换（buffer 版本交替 + 同步边），不改任何数值路径；chunk 串行顺序保持（running stats 的循环携带依赖）。
- **依据**：online_softmax streaming L106–115（Persistent + T.serial(chunk) + 双版本 x_ub + stats_ub 三件套）；single L137–144（Persistent + 子批 T.serial + SimtVF 嵌套——Stage 3 实证）；tilelang-ascend skill §4.2/§7。
- **嵌套风险与退化形态**：`Persistent > TIR if（尾块）> T.serial（子批）> T.serial（chunk）> SimtVF` 的完整嵌套无单一先例，但**契约主流落点 block_m=1 → SUB=1 → num_sub=1**，子批循环退化为单迭代——嵌套结构与 online_softmax streaming（measured）完全同构；显式 block_m>1 的深层嵌套由 single 已实证的「Persistent + TIR if + T.serial + SimtVF」四层组合覆盖（其 Stage 3 RETROSPECTIVE：四层组合不再是排查点）。fallback 预案：若 chunk 层嵌套导致 AutoSchedule multibuffering 推断异常，改用「chunk 外层 × 子批内层 + stats_ub 扩容 (block_m, 2)」（语义等价、x_ub 流水粒度变粗）——§9.1 风险 3。

**重设计项 R4: M 尾块（隐式 OOB 依赖 → 静态 full/rem 分解 + TIR if + 静态切片）**

- **源方案**：快路径 `T.copy`（S1a/S8）在尾 tile（M % block_m ≠ 0）依赖 GPU `T.copy` 越界保护（未文档化）；masked 分支有行谓词（L152）但仅 `_needs_mask` 时激活。
- **NPU 新算法**：与 single R4 完全同款——host 静态分解 `full = M // block_m`、`rem_m = M % block_m`、`num_tiles = full + (rem_m>0)`；persistent 体内 TIR if（`if t < full:`）；尾 tile 内满子批 `T.serial(tsub)` + Python 静态末子批（`tsub_rem` 行）；全部 `T.copy` 切片静态；fragment 载入行谓词 `r < rows`（越界行 −inf 载入、结果不写出）；输出 `lse_ub[0:rows] → y[row_base:+rows]` 静态切片。
- **语义保持论证**：与 single R4 相同（chunk 维与 M 尾块维正交——行谓词作用于载入、chunk 合并逐行独立、越界行不写出）；不依赖任何 `T.copy` 越界行为。
- **依据**：single `_logsumexp_kernel_single.py` L140–238（尾块路径 Stage 3 全量 PASS）；`examples/ascend/example_gemm.py` L47（循环变量 TIR if 先例）；`examples/pilot/online_softmax/online_softmax.py` L141–142（静态 rem 切片）。

**重设计项 R5: running 统计量驻留（register fragment 跨 chunk → stats_ub UB 驻留 + SimtVF 域内读写）**

- **源方案**：`row_max/row_sum/prev_max` 为 GPU register fragment，生命周期覆盖整个 chunk 循环（block 内无域边界）。
- **NPU 新算法**：`stats_ub = T.alloc_shared((SUB, 2), "float32")`——`stats_ub[r, 0]` = m_run、`stats_ub[r, 1]` = s_run；每行块子批开始时在 SimtVF 域内初始化（−inf / 0.0）；每个 chunk 的 SimtVF 计算域内直接读写 `stats_ub`（max 合并 / sum 重缩放合并）；chunk 循环结束后同域消费（lse = m_run + log(s_run)）。`T.annotate_buffer_versions({x_ub: 2, stats_ub: 1})` 声明版本语义。
- **语义保持论证**：驻留位置（寄存器 → UB）与读写域结构变化不改数值链——m_run/s_run 的更新公式逐行与源 S3/S6 相同（fp32）；V1/V4 位级一致覆盖全部 chunk 合并路径。UB 代价 (SUB,2)×4B ≤ 128B 可忽略。
- **依据**：online_softmax streaming L102–111（`stats_ub (BM, 2)` + `annotate_buffer_versions({x_ub: 2, stats_ub: 1})` + SimtVF 域内读写——**跨 chunk running stats 的完整同构先例**，measured）；SimtVF 域内 fragment/UB 混合访问（`D-REDUCE-SIMTVF-API`；single L162 `T.copy(out_frag, lse_ub)` 同族）。

### 0.7 标杆实现

- **源算子路径**：`examples/TileOPs/tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py`（`_logsumexp_kernel_tiled` L101–189，语义/算法/优化手段解读的唯一来源）。
- **源仓参考实现/测试基准**：`torch.logsumexp(x.float(), dim).to(x.dtype)`（源仓 `examples/TileOPs/tests/ops/test_softmax.py` L99 `ref_program`，fp32 计算域 golden）——§8.1 golden 直接采用（torch on NPU），保证验证独立性（不复刻 §0.6 NPU 算法）。
- **wrapper 对接面**：`examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp.py`（`_logsumexp_kernel` dispatch L63–68、`_tile_n_for_block_m` L176–198、`default_config` L200–233、`LogSumExpKernel.forward` L235–251）。
- **同族先例**：`examples/logsumexp/_logsumexp_kernel_single/`（per-chunk 计算体 + 尾块机制的 PRECISION_PASS 实证）；`examples/pilot/online_softmax/online_softmax.py`（streaming 路径 = 本设计 chunk 循环 + flash 合并 + stats_ub 驻留的直接同构，measured）。

## 1. 概述

### 1.1 算子名称

`_logsumexp_kernel_tiled`（迁移自 GPU TileOPs `tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py`；算子族 `LogSumExpFwdOp` 的多 tile / online 流式变体）

### 1.2 功能描述

多 tile 行级 log-sum-exp：对 (M, N) 输入沿最后一维做数值稳定的 logsumexp 归约，输出 (M,)。与 single 变体的分工：**N 超出片上驻留上限（N_padded > 32512 或 block_m·elem·N_padded > 216KB，由 wrapper dispatch 判定）或显式 config 指定 tile_n 时选中本变体**——整行无法一次驻留，改用 N 向 chunk 串行流式 + running (max, sum) flash 合并（online softmax 递推），X 单遍 GM 读、任意大 N 可处理。支持 fp16/bf16/fp32 输入（fp32 中间计算）、任意 M（含尾块）、非 pow2 N（尾 chunk masked）、任意 tile_n（BLOCK_N pow2 clamp 兜底）。

### 1.3 数学公式

$$
y_i \;=\; m_i + \log\Big(\sum_{j} \exp(x_{ij} - m_i)\Big), \qquad m_i = \max_j x_{ij}, \qquad i \in [0, M)
$$

online chunk 递推形式（§0.1）：chunk 序列上 $m^{\text{run}}_i \leftarrow \max(m^{\text{run}}_i, m^c_i)$、$s^{\text{run}}_i \leftarrow s^{\text{run}}_i \cdot \exp(p - m^{\text{run}}_i) + s^c_i$，出口 $y_i = m^{\text{run}}_i + \log s^{\text{run}}_i$（与解析式数学恒等——log-sum-exp 递推代数变形；等价性机器验证见 §1.6.1）。

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

**主路径（契约空间全域）——BLOCK_N pow2 chunk 流式 + flash 合并三段式**（§0.6 R1–R5 落地）：

1. host 静态化（per-shape JIT 闭包）：`BLOCK_N = 满足以下约束的最大 2 的幂：BLOCK_N ≤ tile_n 且 2·BLOCK_N·elem ≤ 194.4KB`；`SUB = 满足以下约束的最大 2 的幂：SUB ≤ block_m 且 2·SUB·BLOCK_N·elem ≤ 194.4KB 且 SUB·BLOCK_N·4B ≤ 128KB`；`num_full = N // BLOCK_N`，`rem = N % BLOCK_N`，`TILE_R = next_pow2(rem)`；`full = M // block_m`，`rem_m = M % block_m`，`num_tiles = full + (rem_m>0)`；`num_sub = block_m // SUB`，`last_sub_rows = block_m % SUB`；`tsub = rem_m // SUB`，`tsub_rem = rem_m % SUB`；`threads = min(1024, SUB·BLOCK_N)`；`NUM_BLOCKS = 56`。
2. persistent 行块调度（`T.Persistent([num_tiles], 56, core_id, num_stages=2)`）；全块 tile 走 TIR if 正分支、尾 tile（rem_m 行）走 else 分支（静态切片，§0.6 R4）。
3. 行子批循环 `T.serial(num_sub)`（SUB=1 时单迭代退化）内：
   a. `stats_ub (SUB, 2)` 初始化（m_run=−inf、s_run=0，SimtVF 域内，§0.6 R5）；
   b. **N 向 chunk 串行** `T.serial(num_full)`：`T.copy` 将 chunk 列窗口 `X[row_base:+rows, col:+BLOCK_N]` GM→UB（x_ub (SUB, BLOCK_N) 紧凑、双版本流水）；SimtVF 域内 fragment **无谓词直载** fp32 → `T.reduce_max(dim=1)` 得 m_c → `T.Parallel(SUB)` 合并 `prev=m_run; m_run=max(m_run, m_c)` → 原位 `exp(x − m_run)` → `T.reduce_sum(dim=1)` 得 s_c → `T.Parallel(SUB)` 合并 `s_run = s_run·exp(prev − m_run) + s_c`；
   c. 尾 chunk（rem > 0，Python 静态分支）：静态 rem 切片 copy + fragment `(SUB, TILE_R)` masked 载入（pad=−inf）→ 同款合并体；
   d. lse 收尾：`out_frag[r] = cast(m_run + log(s_run), dtype)`（每行 1 次 log）→ `T.copy(out_frag, lse_ub)` → `T.copy(lse_ub[0:rows], y[row_base:+rows])` UB→GM 静态切片（两段写）。
4. 行谓词（尾子批 / 尾 tile 末子批 `r < rows`）+ 全部 copy 切片静态化——不依赖任何 DMA 越界行为。

**与源算法的结构差异**（逐条注明来源）：① chunk 宽 tile_n（任意 256 倍数）+ N_padded 对齐 → **BLOCK_N pow2 重划** + 尾 chunk 静态 rem 切片（§0.6 R2 / §1.6.1 O2——fragment lane 利用率 50.8%→~100%）；② chunk 级快慢双路径（`_needs_mask`）→ full chunk 无谓词直载 + 尾 chunk masked 二分支静态化（§0.6 R2）；③ 非持久 grid × threads → persistent 56 核 + SimtVF(threads=min(1024, SUB·BLOCK_N))（§0.6 R1，K3）；④ running 统计量 register fragment 跨 chunk → `stats_ub` UB 驻留 + SimtVF 域内读写（§0.6 R5）；⑤ M 尾块隐式 OOB → 静态 full/rem 分解 + TIR if + 静态切片（§0.6 R4）；⑥ `T.reduce_*` warp 树 → AscendAllReduce 硬件树（§0.5 等价替换）；⑦ block_m 整批 → 行子批 SUB 双缓冲（§0.6 R2/R3——VRF/UB 双预算兜底任意 block_m）。**公式层与源恒等**（online 递推逐式同构：max 基重缩放合并、exp 以合并后 max 为基、log 每行 1 次——V1/V4 位级验证）；per-chunk 计算体（masked 载入/reduce_max/exp 原位/reduce_sum/lse+cast/两段写）与 single 变体 Stage 3 实证代码同构复用。

### 1.5 数据流图

```
GM[X (M,N) dtype]
  --T.copy (MTE2, chunk 列窗口 [col:+BLOCK_N] × 行子批 [row_base:+rows], x_ub 双版本)--> UB[x_ub (SUB,BLOCK_N) dtype]
  --SimtVF T.Parallel (full chunk 无谓词直载 / 尾 chunk 行∧列谓词 pad=-inf, cast fp32)--> VRF[x_frag (SUB,BLOCK_N 或 SUB,TILE_R) fp32]
  → VRF reduce_max (AscendAllReduce 硬件树) → m_c (SUB,) fp32
  → UB[stats_ub[r,0]] ← max 合并 (prev 驻留 prev_frag)            [跨 chunk running, UB 驻留]
  → VRF x_frag 原位 = exp(x_frag − m_run)                          [寄存器驻留，单 fragment]
  → VRF reduce_sum → s_c (SUB,) fp32
  → UB[stats_ub[r,1]] ← s_run·exp(prev−m_run) + s_c                [flash 重缩放合并]
  [chunk 循环 num_full 次 + 尾 chunk 1 次]
  → VRF out_frag = cast(m_run + log(s_run), dtype) (SUB,)
  --T.copy (fragment→UB)--> UB[lse_ub (SUB,) dtype]
  --T.copy (MTE3, 1D 静态切片 [row_base:+rows])--> GM[y (M,)]
```

纯 Vector 算子：无 L1/L0A/L0B/L0C 参与段（无 Cube/`T.gemm`），NT GEMM 布局不适用；无中间量落 GM（running stats 驻留 UB）；**X 单遍 GM 读**（chunk 流式恰读一次）、y 单遍 GM 写。

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先调研（1.6.0，调研四问——输入公式/源算法只是候选之一，迁移任务在源算子解读后、耦合性判定前完成）；再做数学等价的公式优化（1.6.1）；再做循环/标量计算的向量化替代（1.6.2）；最后决策向量化轴与核内布局（1.6.3）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。1.6.1–1.6.3 的分析对象 = §1.6.0 调研选定、经 §0.6 重设计落地的 NPU 算法，不止步于源公式的直译。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：同一数学语义（行 logsumexp）在**超驻留大 N 语境**下的算法族——single 变体的调研结论（驻留可行时三段式最优、online 非必需）不适用于本变体的契约空间（N 超驻留是本变体的选中条件），须独立调研。调研深度：**完整调研**（规约/流式类）。执行位置：Phase M0 之后、Phase M1 之前。
> 信息源：`.agents/skills/tilelang-op-design/references/algorithm-candidates.md`（**ALG-softmax** 命中行〔family 含 logsumexp〕：R2「online softmax 分块 running max/sum + 重缩放」即本变体算法；R1 减 max 稳定化 / exp2 换底同 single 结论）、pattern-library（`D-REDUCE-SIMTVF-API` measured / `TRAP-NAIVE-ONLINE-MERGE` measured / `D-ELEM-SIMTVF-AUTOVEC` measured / `CONST-valu-model` / `CONST-mte-bandwidth-model` / `CONST-capacity-a5` / `CONST-vrf-capacity-a5`）、`examples/pilot/online_softmax/`（streaming 路径 = online 流式 + flash 合并 + stats_ub 的 measured 正例，63/63 PASS）、`examples/logsumexp/_logsumexp_kernel_single/`（同族驻留变体全流程实证 + 其 DESIGN §1.6.0 对同一算法族的驻留语境调研）、源算法（§0.3/§0.4——**源即 online 变体**，chunk 序 flash 合并的算法证据）、`kb_search "logsumexp online streaming reduction chunk merge"`（命中 ALG-softmax / TRAP-NAIVE-ONLINE-MERGE / D-REDUCE-SIMTVF-API，无新增候选）。互联网检索：未使用——本地源（候选表命中行 + streaming 先例 + 源仓双变体结构 + single 同族调研）已覆盖超驻留 logsumexp 的全部已知算法结构（online 单遍 / 两遍重读 / 多 kernel partial+merge / Cube 归并），无新增候选空间。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线 = 源算法（online chunk + flash 合并，tile_n 粒度） | §0.1 递推式；chunk=tile_n（256 倍数），尾 chunk masked 至 N_padded | — | — | ✅（基线） |
| 1 | 减 max 稳定化 | `e=exp(x−m)`、`s·exp(p−m_run)` 重缩放（基线内嵌） | 数学恒等（平移不变） | 消 fp16 溢出（必选） | 基线已含 |
| 2 | exp→exp2·log2e 换底 | `e^t = 2^(t·log2e)` | 数学恒等 | GPU SFU 快路径 | ❌ **弃选**：`CONST-valu-model` special_func 档位成员注记（measured，2026-10-16 三 pilot 实证）：exp/exp2/log2 同入 8 ops/cycle 档（`src/ascend/codegen/intrin_rule_ascend.cc` L34–49 均注册）——换底无单位收益、反多 1 乘/elem 与字面量截断误差（single 同结论） |
| 3 | log 域拆分 / log2 换底输出 | `y = (m·log2e + log2(S))/log2e` 类 | 数学恒等 | 无（log 每行仅 1 次，1/N 占比） | ❌ 同上档位证据；每行 1 次 special 非热点 |
| 4 | **BLOCK_N pow2 chunk 重划**（布局级化简） | chunk 宽 tile_n（任意 256 倍数）→ BLOCK_N pow2 ≤ tile_n + 尾 chunk 静态 rem 切片 | 数学恒等（flash 合并结合律；每元素恰计一次） | 非 pow2 tile_n 下 fragment 有效 lane 50.8%→~100%（special 恢复满吞吐）；GM 消 N_padded pad 列 | ✅（O2，A5 结构差异——§0.6 R2） |
| 5 | e 原位覆写（寄存器复用） | 分离 e_frag → 原位写回 x_frag | 数值同链（同舍入路径） | VRF −50%（单 (SUB,BLOCK_N) fp32 fragment） | ✅（O3，源已含此结构——S4 原位，A5 承接确认） |
| 6 | fp16-e 精简链（fp16 输入限定） | chunk 内 exp/sum fp16（s_c fp16 树形）、stats 合并保持 fp32 | 容差内等价（预估 max_rel ~1e-5–4e-4，V7 实测 3.6e-5） | VRF −50%；special 若 fp16 双倍吞吐（**未文档化假设**）则计算项 −50% | ✅ 纳入 §1.6.4 实验裁决（不进 R3 主对比：dtype 限定 + 合并链精度依赖，不构成全局结构候选） |
| 7 | chunk 内行内 online 微变体（单遍 cmp+sum 合并） | chunk 内再分块 K 段 running 合并（消 reduce_max 前的独立扫描） | 容差内等价（多次重舍入） | −1 遍 cmp 扫/chunk（fragment 域），+2K special/行/chunk | ❌ **弃选**：chunk 数据已在 VRF，「遍数」是向量 pass 数而非 GM 读——收益 ~1 cmp @51 档（0.02 cycle/elem）vs 每 extra special 0.125 cycle/elem（K=2 净亏）；引入块内序依赖削弱 chunk 间流水（single R1 候选 7 同款弃选逻辑）；且须哨兵钳位防角点（`TRAP-NAIVE-ONLINE-MERGE`）——复杂度代价明确 |
| 8 | Cube GEMV 归并（s_c = e·1 向量内积） | chunk exp 后 `T.gemm(e_l1, ones_l1, c_l0c)` | 数学等价 | Cube MAC 替代 Vector 归约 | ❌ **弃选**：exp 主导段仍在 Vector（逐元素 special），Cube 只接管 s_c——多出 UB→L1 ND2NZ 转换 + L0C→UB dual_copy 往返（`CONST-mte-bandwidth-model` ND2NZ 62 cycles base）+ 每 chunk 一次（chunk 数 × 转换开销，比 single 场景更频）+ mixed 核复杂度；`T.reduce_sum` 已是 AscendAllReduce 硬件直连（`D-REDUCE-SIMTVF-API` measured）。**违反流量最小化** |
| 9 | 两遍法（pass1 全行 max 落 GM，pass2 exp+sum） | kernel 内两遍 chunk 扫描（或两 kernel） | 数学等价 | 消合并标量（每行每 chunk ~6 op） | ❌ **弃选**：**X 读 2 遍 GM（流量 +100%）**——大 N 语境 X 读是流量主体（wl2 64MB）；合并标量占比 ~0.02%（D-2），以 +100% 流量换 0.02% 计算是数量级误判 |
| 10 | 多 kernel partial+merge（partial (m,s) 落 GM + merge kernel） | 每 chunk 每 partial 行写 GM，merge kernel 归并 | 数学等价（合并律） | chunk 间去串行化（理论并行度↑） | ❌ **弃选**：partial 读写 M×chunks×8B 额外 GM 往返 + 跨 kernel 同步；且行块内 chunk 串行不是并行度瓶颈（M 行块已提供 M/block_m 路独立任务，wl2 = 1024 tiles >> 56 核）——串行只存在于单行块内部，无需跨 chunk 并行化 |
| 11 | 哨兵钳位 m_c ≥ −65504（pilot online_softmax O6 同款） | `m_c = max(reduce_max(x_c), −65504)` 在 exp 之前 | **容差内等价不成立（语义变更）**：全 −inf 行 y 从 NaN → −65504+log(0) = −inf | 防 exp(−inf−(−inf))=NaN 角点 | ❌ **弃选（语义优先）**：源 GPU tiled 无钳位，全 −inf 行 → NaN 是源固有语义（与 single 变体一致——迁移保真）；本算子的 flash 合并式在 ±inf 角点已数值安全（exp 参数 ≤ 0，无溢出路径——V5 `max-gap 131008` 实测位级正确），唯一 NaN 路径 = 全行全 −inf = 源语义本身；引入钳位反而制造 single(NaN)/tiled(−inf) 两变体语义分裂（single DESIGN §9.3 已拒绝同款变更）。若未来需要与 torch 全角点对齐，属算子族级变更回归源仓统一处理 |

**R2 在线算法**：**有——源算法本身即 online 变体**（本变体的存在理由；`ALG-softmax` R2 命中行「online softmax 分块 running max/sum + 重缩放」；结构判据：logsumexp 可分解为 running (max, sum) 统计量，合并满足结合律）。**收益口径（超驻留语境）**：① GM 扫描遍数 = 1（两遍法 2、materialize 不可能——O(N) 中间缓冲超驻留即本变体前提）；② 中间缓冲 O(BLOCK_N)（vs 两遍法的全行 max 落 GM 或 O(N) 驻留）；③ 合并开销 O(rows·chunks) 标量（D-2：~0.02% 占比）。**结论**：online 是超驻留大 N 语境下的唯一单遍算法族，两遍/多 kernel 候选见 R1 #9/#10 弃选——**算法族选型确认为源算法族（online flash 合并），本设计在此基础上做 chunk 粒度与驻留布局的 A5 适配（R1 #4）**。

**R3 算法复杂度对比**（逐候选四口径；口径说明：FLOPs/elem 中 cast/select/sub/add/cmp 各计 1 op、exp/log 计 1 special（折算系数按 `CONST-valu-model` special = 8 ops/cycle/AIV，assumption）；log 每行 1 次忽略；GM Bytes/elem = 输入读 + 输出写（M << M·N 忽略）+ 中间 GM 往返；扫描遍数 = 同一 GM 数据读取次数；中间缓冲 = 行级驻存需求峰值（UB+VRF）；代表 workload = wl2 (M=1024, N=32768, bf16, block_m=1, tile_n=16384)）：

| 算法候选 | FLOPs/elem | GM Bytes/elem | 扫描遍数 | 中间缓冲/行 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线直译（tile_n chunk + N_padded + fragment pow2 翻译形态：tile_n=16640 时 TILE_C=32768） | ~5 op + 1 special × **(TILE_C/tile_n) 有效率折算**（16640 场景 special 实耗 ×1.97；16384 场景 ×1.0） | 2（读 2B；对齐 N）至 2.02（N=33000：N_padded 33280 pad 列 +0.85%） | 2（读 1 + 写 ~0） | x_ub 2·SUB·tile_n·elem（双版本）+ x_frag SUB·next_pow2(tile_n)·4B（wl2 = 64KB UB + 64KB VRF；16640 场景 VRF 128KB 贴上限） | M 行块独立；无跨核归约 |
| **选定：BLOCK_N pow2 chunk 流式 + flash 合并（原位覆写 + 无谓词 full chunk）** | ~5 op + 1 special（full chunk 利用率 ~100%；尾 chunk 1 个摊薄） | 2（任意 N 紧凑，无 pad 列搬运） | 2 | x_ub 2·SUB·BLOCK_N·elem（双版本 ≤194.4KB）+ x_frag SUB·BLOCK_N·4B（≤128KB VRF）+ stats_ub 8B | 同上；chunk 串行在行块内（循环携带依赖，M/block_m 路独立任务 ≥ 56 核饱和） |
| 两遍法（max 落 GM + exp/sum 重读） | ~5 op + 1 special + max 写读 4B/行往返 | 4（读 2×2B + max 写读 ~0）→ **读 2 遍 X** | 4 | O(1) 行统计量 + 全行 max 落 GM（M×4B） | 行块独立；跨 pass 数据依赖 |
| 多 kernel partial+merge | ~5 op + 1 special（+partial 写读 M×chunks×8B） | 2 + partial 往返（chunks×8B/行 ≈ 0.0005·chunks/elem） | 2 + partial 2 | partial GM（M×chunks×8B） | chunk 并行化理论↑；跨 kernel 同步 + launch 开销 |
| fp16-e 精简链（备选 §1.6.4，fp16 限定） | ~3 op + 1 special(fp16)（省载入 cast 1 + select 1） | 2 | 2 | x_frag 减半（VRF −50%） | 同选定 |
| Cube GEMV 归并 | ~5 op + 1 special + Cube MAC + 每 chunk ND2NZ/dual_copy 往返 | 2 + 每 chunk 转换流量 | ≥2 | 追加 L1/L0C staging（每 chunk） | mixed 核协调开销 × chunks |

**R4 硬件亲和性评估**（逐候选对照检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水融合 / 跨核结构；负向淘汰依据已附）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线直译 | 纯 Vector ✓；special 1/elem 撞最窄吞吐档且非 pow2 tile_n 下**实耗 ×1.97**（`CONST-valu-model` special=8，assumption） | 16640 场景 x_frag 128KB 贴 `CONST-vrf-capacity-a5` 假设上限（实测水位 76 reg/thread 无 spill 同段——无余量）；25600 场景 128KB 同贴 | tile_n % 128（fp16 lane）非 0（16640 % 128 = 0 ✓ 但 TILE_C=32768 时尾 lane 49.2% 浪费） | ✓（JIT 常量） | chunk 间双缓冲 ✓ | ⚠️ 可行但非 pow2 tile_n 下 special 折半 + VRF 贴顶——**降级为结构参照** |
| **BLOCK_N pow2 chunk 流式（选定）** | Vector ✓；special 1/elem 满有效（full chunk 无谓词直载还省 select 1 op/elem）；reduce_* AscendAllReduce 直连（`D-REDUCE-SIMTVF-API` measured） | x_ub 双版本 ≤194.4KB（90% 折扣预算）✓；x_frag ≤128KB VRF ✓；stats_ub 8B ✓ | BLOCK_N pow2 % 128 lane = 0 ✓；UB 紧凑（不搬 pad 列）；GM chunk 列窗口 32B 对齐（wl2/wl3 行内 32KB/2048B 对齐 ✓）；尾 chunk TILE_R pow2 masked | BLOCK_N/num_full/rem/TILE_R/SUB/full/rem_m 全 JIT 常量 ✓；尾 chunk/末子批 Python 静态分支 ✓ | chunk 间双缓冲（num_stages=2）MTE2/VALU 重叠 ✓；persistent 多波 ✓ | ✅ **选定（主选）** |
| fp16-e 备选 | 同上；exp fp16 走 h 前缀 intrinsic（`intrin_rule_ascend.cc` L27–32 dtype 派发），潜在 2× special 吞吐（**未文档化假设**：`CONST-valu-model` 未区分 dtype 粒度） | VRF 再 −50% ✓ | 同上 ✓ | ✓ | ✓ | ✅ **备选（实验裁决 §1.6.4）**——判定裕度落在 special dtype 粒度这一未实证常数内（V7 守卫余量 ~280×） |
| 两遍法 | ✓（同计算主体） | O(1) ✓ | ✓ | ✓ | pass 间无重叠（stats 依赖） | ❌ 弃选（R1 #9：X 读 2 遍，流量 +100% vs 合并标量 0.02%） |
| 多 kernel partial+merge | ✓ | partial 落 GM ✓ | ✓ | ✓ | 跨 kernel 间隙 + launch 开销 ×2 | ❌ 弃选（R1 #10：partial 往返 + 同步；行块并行度已饱和 56 核，chunk 串行非瓶颈） |
| Cube GEMV | exp 段 Vector + s_c 段 Cube——单元混置且转换 × chunks | 追加 L1 + L0C staging（每 chunk） | 须凑 C0 分形（M/N 16 倍数——block_m=1 契约下须 padding） | ✓ | mixed 核协调 × chunks | ❌ 弃选（R1 #8：转换开销随 chunk 数放大，归约非瓶颈） |

**调研结论**：选定 **「BLOCK_N pow2 chunk 流式 online logsumexp（flash 合并 + UB 紧凑 + 无谓词 full chunk 直载 + exp 原位覆写 + chunk 间双缓冲流水 + stats_ub 驻留）」**。关键依据：① R2/R3——online flash 合并是超驻留大 N 的唯一单遍算法族（两遍法 X 读 +100%、多 kernel partial 往返 + 同步，均数量级劣于合并标量 0.02% 的节省），算法族与源一致；② R3/R4——结构层 BLOCK_N pow2 重划对非 pow2 tile_n（16640/25600，真实 workload wl3 与测试 (33,33000) 落点）恢复 special 满吞吐（×1.97→×1.0）并消除 pad 列搬运，VRF 从贴顶（128KB）回落可控；③ R4——纯 AIV 负载、容量全档 ✓（194.4KB/128KB/8B）、pow2 整除、chunk 间双缓冲、无跨核归约（M/block_m ≥ 56 行块饱和核）；④ R1 #11 哨兵钳位显式弃选（语义优先——源 NaN 行为保持，flash 形式角点本已安全）。与基线的结构差异一句话：**tile_n 粒度 chunk + N_padded 对齐 + register stats（源）→ BLOCK_N pow2 chunk + 紧凑驻留 + stats_ub UB 驻留（选定）**，递推公式恒等（V1/V4 位级）。源优化手段意图承接：#1 online 递推（保留）/ #2 chunk 分块（保留，粒度重算）/ #3 快慢双路径（R2 统一为更优二分支，意图「全有效 chunk 免掩码」由「full chunk 无谓词直载」承接）/ #4 SMEM staging（UB 单跳等价替换）/ #5 稳定化（保留）/ #6 树形归约（T.reduce_* 等价替换）/ #7 合并式标量（T.Parallel(SUB) 等价替换）/ #8 行批+grid（persistent + 子批承接）——逐项见 §0.5，无静默丢弃。调研范围：ALG-softmax 命中行、examples/pilot/online_softmax（streaming measured）+ single 全流程实证 + examples/ascend 四同类、pattern-library 七条目（三 measured 四 assumption）、源仓双变体结构分析、kb_search 命中条目。

**设计期估算（D-2）**：估算下界(相对)：**计算项（VALU special）主导，约为流量项的 ~11.7 倍**（代表 workload wl2 (1024, 32768) bf16：流量 64MB → 每核 1.14MB/(100 B/cycle) ≈ 11.7K cycle〔`CONST-mte-bandwidth-model` assumption，AIV MTE2 档〕；计算 33.55M elem×(5/51 + 1/8)/56 核 ≈ 133.6K cycle/核〔`CONST-valu-model` assumption，special=8 档——special 0.125 cycle/elem 占计算项 56%〕；合并标量 1024 行×2 chunks×~6 op/56 ≈ 220 op/核可忽略；容量项可忽略——chunk 流式单遍 GM 读、无 L0C 往返）。主选 vs 基线直译相对比：VALU 计算项——wl2（tile_n=16384 pow2）持平 1.0×；**非 pow2 tile_n 场景（wl3 tile_n=25600 → 基线 TILE_C=32768 实耗 ×1.27、测试 (33,33000) tile_n=16640 → ×1.97）主选恢复 ×1.0**；GM 流量——N=33000 基线搬 33280 列（+0.85%），主选紧凑；流水——两者均有 chunk 双缓冲。fp16-e 备选（若 special fp16 2× 吞吐成立）：计算项 0.72×。**wl3 并行度注记**：M=4 → num_tiles=4 < 56（仅 4 核活跃）——端到端 ≈ 单核序列 102400×0.223 ≈ 22.9K cycle，并行度受限是 workload 特性（M 小）而非算法结构问题，Stage 4 记录。**口径不确定披露**：special_func「8 ops/cycle」的 op 粒度（element 级 vs 向量指令级）在 `src/ascend/transform/estimate_latency.h` L58–70 未区分——若为向量指令粒度则瓶颈反转 MTE 主导；两种解释下主选结构（special 已最小化 + 流量已最小化 + chunk 流水）均为最优，该常数列 §9 风险点，Stage 4 npusim/msprof 实测归因标定。A5 冷启动期 assumption 基估算，常数待标定，容量按 90% 折扣。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定、经 §0.6 R2/R5 落地的 BLOCK_N pow2 chunk 流式算法。递推公式层源已是该族最优形态（max 基重缩放合并、exp 以合并后 max 为基、log 1/行——R1 候选 2/3/7/11 弃选），本节优化集中在**chunk 划分与访存层**，每项四要素齐备。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| O1 | 减 max 稳定化 + flash 重缩放（必选确认） | `y = log(Σ e^{x})` | `m_run` 递推 + `s_run·exp(p−m_run)+s_c` 合并 + `y = m_run+log(s_run)` | 指数平移恒等 + log-sum-exp 递推代数变形；消 fp16 溢出；合并式全部 exp 参数 ≤ 0（p ≤ m_run、x ≤ m_c ≤ m_run）——**无 `TRAP-NAIVE-ONLINE-MERGE` 的溢出角点**（朴素 m_c 基 lse 式在间隙 > 88.7 即 fp32 溢出；V5 corner `max-gap 131008` 实测位级正确） | 计算量不变，数值稳定（源已含，语义层保留项） |
| O2 | **BLOCK_N pow2 chunk 重划 + 紧凑驻留**（布局级，本变体核心） | 源：chunk 宽 tile_n（任意 256 倍数），列尾对齐 N_padded = align_up(N, 256)，尾 chunk masked；A5 直译 fragment 形状 (block_m, next_pow2(tile_n))——tile_n=16640 → 32768 列仅 50.8% 有效、25600 → 78.1% | `BLOCK_N = 满足以下约束的最大 2 的幂：BLOCK_N ≤ tile_n 且 2·BLOCK_N·elem ≤ 194.4KB`；full chunk（全列有效，fragment 无谓词直载）+ 尾 chunk 静态 rem 切片（`TILE_R = next_pow2(rem)`，masked pad=−inf） | flash 合并对 chunk 划分满足**结合律**（每元素恰计入一个 chunk、max 基重缩放代数恒等）→ 数学恒等；−inf pad 为 max/exp/sum 三幂元（single V1 位级佐证 + 本任务 V1 覆盖）；**机器验证 V1**：7 组 (tile_n, BLOCK_N) 组合 fp64 与 fp32 **位级一致（ulp=0）**；**V6** 划分扫描（BLOCK_N 256…16384 × tile_n 16640/25000/32512）≤1 ulp；**V4** 源链 vs A5 链 fp16 输出位级一致 + NaN 模式一致 | 非 pow2 tile_n：special 实耗 ×1.97（16640）/×1.27（25600）→ ×1.0（计算主导项恢复满吞吐——wl3 场景 D-2 计算项 ~−21%）；GM 消 pad 列（N=33000：33280→33000 列，−0.85%）；full chunk 省载入 select 1 op/elem |
| O3 | e 原位覆写（寄存器复用确认） | 分离缓冲：`e_frag = exp(x_frag − m_run)`（双 (SUB,BLOCK_N) fp32 fragment 并存） | `x_frag ← exp(x_frag − m_run[r])` 原位（单 fragment；源 S4 已用此结构） | 同公式同舍入链（读-算-写同一位置）；**机器验证 V2**：全组合 fp32 **位级一致（bitwise=True）**——源已含（L171–172），A5 承接确认 | VRF −50%：(SUB,BLOCK_N) fp32 单 fragment ≤128KB（SUB 封顶 32 reg/thread 内），spill 风险消除；UB 无变化 |
| O4 | 合并式标量最小化（确认项，源式 vs pilot 双 exp 式） | pilot 式（m_c 基）：`s_run = s_run·exp(m_run_old−mn) + s_c·exp(m_c−mn)`——每行每 chunk 2 次 exp 标量 | 源式（m_run 基）：chunk 的 e 直接以合并后 m_run 为基（S4），旧 sum 重缩放仅 1 次 exp 标量（S6） | log-sum-exp 递推的代数变形（`TRAP-NAIVE-ONLINE-MERGE` 修复实证注记：两式良性数据恒等，max_rel 1.32e-12）；本任务 V1/V4 的 A5 链即按源式模拟——位级一致覆盖 | 每行每 chunk 省 1 次标量 special + 1 次乘（D-2 合并项 ~0.02% 占比——绝对量小，但源式同时是语义保真形态，无需引入 pilot 式再论证等价） |
| O5 | cast 链最小化（确认项） | — | 输入 cast 1/elem（full chunk 融合于直载；尾 chunk 与 masked select 融合于同一 `T.if_then_else`）+ 输出 cast 1/row | fp32 中间是源语义（§0.1 dtype 语义），不可省 | 已最小（无冗余 cast；stats 全程 fp32 无 cast） |

**优化结论**：**采纳 2 项（O2 BLOCK_N pow2 重划+紧凑驻留、O3 原位覆写）+ 3 项必选/确认（O1 稳定化+flash 重缩放、O4 源式合并、O5 cast 链）**。优化后主路径公式（§3.1 拆解的唯一输入）：

```
init（每行块子批）:  m_run = −inf;  s_run = 0                     # fp32, stats_ub
chunk c（full，无谓词直载）:
    x_c   = cast_fp32(x_ub[0:SUB, 0:BLOCK_N])                     # T.copy GM→UB 后 SimtVF 直载
    m_c   = max_j x_c                                              # T.reduce_max(dim=1), fp32
    prev  = m_run;  m_run = max(m_run, m_c)                       # T.Parallel(SUB) 行维
    x_c   = exp(x_c − m_run)                                       # 原位覆写, fp32
    s_c   = Σ_j x_c                                                # T.reduce_sum(dim=1), fp32
    s_run = s_run·exp(prev − m_run) + s_c                          # flash 重缩放, T.Parallel(SUB)
尾 chunk（rem 列）:  同上，载入 masked（行∧列谓词, pad=−inf）, TILE_R = next_pow2(rem)
出口:  y_i = cast(m_run + log(s_run), dtype)                       # 每行 1 次 special + 1 次 cast
```

**等价性机器验证（D-1）**：已生成并执行 `examples/logsumexp/_logsumexp_kernel_tiled/verify_equiv.py`（fp64 参照 + 角点值：全零/全等 3.5/单主导/±65504 极值/subnormal ±6e-8/深度下溢 −745/部分 −inf/N 非 pow2（33000/5000/32513）/tiny rem=4/整除与非整除 chunk/跨 chunk 角点行〔全 −inf、+inf、NaN、−inf 分布两种顺序、max-gap 131008、主导值在 chunk0/chunk1〕；torch CPU〔NPU 无 fp64——single RETROSPECTIVE 实证 `.double()` 在 NPU 静默降级〕；候选式与基线式均按目标精度链计算对照）。执行结果（2026-10-16，tilelang-fork 工作区 + torch CPU）：

| 验证项 | 对照（候选 vs 基线/精确） | 最大 ulp 差（fp32 位级） | 最大相对差 | 违反率（@容差） | 结论 |
|--------|--------------------------|--------------------------|------------|---------------------|------|
| V1（O2，7 组 (M,N,tile_n,BLOCK_N) × 2 精度） | BLOCK_N pow2 chunk 链 vs tile_n chunk 链（源划分） | **0**（fp64 与 fp32 均位级一致） | 0 | 0.000000 | **EQUIV_PASS** |
| V2（O3，7 组） | 原位覆写链 vs 分离 e 链（fp32） | **0**（位级一致） | 0 | 0.000000 | **EQUIV_PASS** |
| V4（源链 vs A5 链） | 源 GPU tile_n 链 vs A5 BLOCK_N 链，fp64→fp16 输出，含特殊行 | **0**（位级一致）；NaN 模式逐 shape 一致 | 0 | 0.000000 | **EQUIV_PASS** |
| V3（A5 链 vs golden） | A5 fp32 链 vs `torch.logsumexp`（fp32） | 1（树形分组差） | 6.47e-08 | 0.000000（@1e-3） | **EQUIV_PASS** |
| V5（角点专项，信息项+采纳项佐证） | 全 −inf 行 → 链 NaN vs torch −inf；含 +inf → 链 NaN vs torch +inf；含 NaN → 双方 NaN；部分 −inf（chunk0 散布 / chunk1 全 −inf 两种顺序）→ 与精确一致（中性 ✓）；max-gap 131008 → **无溢出、位级正确**；主导值 chunk0/chunk1 → 位级正确 | 0（有限角点） | 0 | 0.000000 | **EQUIV_PASS**（±inf 分歧为源继承，文档化 §9.3） |
| V6（划分敏感性，信息项） | BLOCK_N ∈ 256/1024/4096/8192/16384 × tile_n ∈ 16640/25000/32512 交叉对照 | ≤1 | 6.32e-08 | 0.000000（@1e-3） | **EQUIV_PASS** |
| V7（备选预估，不采纳） | fp16-e tiled 模拟链 vs 精确（1024×32768 / 4×102400） | — | 3.645e-05 / 1.402e-05 | 0.000000（@1e-2） | GUARD_PASS（§1.6.4 备选精度守卫内，余量 ~280×） |

> **全部采纳项 EQUIV_PASS**（V1/V2/V4 位级一致；V3 与 golden ≤1 ulp 树形差）。V5 证实 flash 合并式在 fp16 极端 max 间隙（131008）下无溢出（`TRAP-NAIVE-ONLINE-MERGE` 的朴素式对照）且 ±inf 角点行为为源固有（链与源位级一致即迁移保真达成）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | full chunk 载入（dtype→fp32 提升） | 逐元素载入 | `T.Parallel(SUB, BLOCK_N)` + `T.cast(x_ub[r,i], "float32")` **无谓词直载**（全列有效——谓词恒真不写） | ✅ | —（online_softmax streaming L123 同构直载先例；`D-ELEM-SIMTVF-AUTOVEC` measured） |
| 2 | 尾 chunk masked 载入（含 dtype→fp32） | 逐元素边界分支载入（源 S1b 同构） | `T.Parallel(SUB, TILE_R)` + `T.if_then_else(T.And(r < rows, i < rem), T.cast(x_ub[r,i], "float32"), −T.infinity(T.float32))` | ✅ | —（online_softmax L150–152 / single L150–155 同构掩码载入先例；`D-REDUCE-SIMTVF-API` measured） |
| 3 | running max 合并（`prev=m_run; m_run=max(m_run,m_c)`） | 每行标量 max/copy | `T.Parallel(SUB)` 行维向量化（每行 1 max + 1 copy；prev 驻留 prev_frag） | ✅ | —（源 L167–169 `T.Parallel(block_m)` 同构；行维向量化同 single lse 行） |
| 4 | chunk 内行 max | 树形归约（源 warp shuffle） | `T.reduce_max(x_frag, m_c, dim=1, clear=True)` | ✅ | —（`D-REDUCE-SIMTVF-API` measured：SimtVF 内自动降级 `tl::AscendAllReduce<MaxOp>` 硬件树；single L156 同族实证） |
| 5 | 逐元素 e = exp(x − m_run)（原位） | 标量循环 exp | `T.Parallel(SUB, BLOCK_N)` 向量化（`ascend_simdvf_lower_parallel` 自动向量化，`tir.exp → simd_vexp`） | ✅ | —（`D-ELEM-SIMTVF-AUTOVEC` measured，`src/ascend/transform/ascend_simdvf_lower_parallel.cc:409-412`；single L157–158 同族） |
| 6 | running sum 重缩放合并（`s_run·exp(prev−m_run)+s_c`） | 每行标量算术 + exp | `T.Parallel(SUB)` 行维向量化（每行 1 exp + 1 mul + 1 add） | ✅ | —（源 L175–178 `T.Parallel(block_m)` 同构） |
| 7 | chunk 内行和 | 标量累加循环 | `T.reduce_sum(x_frag, s_c, dim=1, clear=True)` | ✅ | —（`D-REDUCE-SIMTVF-API` measured：`SumOp` 硬件树；single L159 同族） |
| 8 | 出口 lse = m_run + log(s_run) + cast | 每行标量超越函数 | `T.Parallel(SUB)` 行维向量化（log 为注册 intrinsic，`intrin_rule_ascend.cc` L34–49；online_softmax L169 `T.log` 在用） | ✅（行维） | 依赖 reduce 终值（s_run 定稿后才能求 log）——tile 级标量（SUB 个/子批 vs SUB·BLOCK_N 元素，占比 1/BLOCK_N ≈ 0.006%@16384），无法并入元素循环；非逐元素热点 |
| 9 | lse 写出（fragment→UB→GM） | 标量散写 | `T.copy(out_frag, lse_ub)` + `T.copy(lse_ub[0:rows], y[切片])` 两段写 | ✅ | —（single L162–163 / `example_per_token_cast_to_fp8.py` L81–83 一维输出两段写同构） |
| 10 | tile 调度循环 | — | `T.Persistent([num_tiles], 56, core_id, num_stages=2)` | 结构循环 | 块级任务映射/调度（非计算循环） |
| 11 | 行子批循环 | — | `T.serial(num_sub)` | ❌（保留串行） | 子批间 x_ub 版本交替的流水顺序依赖（§1.6.2 #8 同 single）；契约主流落点 block_m=1 → num_sub=1 退化单迭代；子批数 ≤ ceildiv(block_m, SUB) ≤ 16，非逐元素热点 |
| 12 | **N 向 chunk 串行循环** | — | `T.serial(num_full)` | ❌（保留串行） | **running stats 的循环携带依赖**（m_run/s_run 按 chunk 序 flash 合并——顺序语义，§0.1 规约语义；chunk 间数学上可交换〔结合律 V6〕但并行化须 partial (m,s) 数组驻留 + 终局合并链，行块内并行化收益为 0：M/block_m 路行块任务已 ≥ 56 核饱和〔wl2 1024 tiles〕，chunk 维并行只增加驻留与合并复杂度——R1 #10 弃选同款）；收益由 chunk 间双缓冲流水体现（§6.3） |
| 13 | 尾块 TIR if 分支（`if t < full`） | — | 无向量等价 | ❌（保留分支） | 依赖循环变量 t 的运行时控制流（M 尾块边界语义）；分支体两侧均为静态切片——`example_gemm.py` L47 先例 + single Stage 3 实证；每 tile 1 次分支判断，非热点 |
| 14 | host 侧 BLOCK_N/SUB/num_full/rem/TILE_R/threads 计算 | Python 算术 | — | n/a | host 元数据计算，不在 kernel 内（per-shape JIT 静态化） |

**向量化结论**：逐元素计算 100% 向量化（#1–#9 全部 `T.Parallel`/`T.reduce_*`，API 均有 examples + pattern-library measured 佐证）；保留 4 处非向量化点——#8 tile 级每行 lse（依赖归约终值，占比 1/BLOCK_N）、#11 子批串行（双缓冲流水本质 + 主流落点退化）、#12 chunk 串行（running stats 循环携带依赖 + 行块并行度已饱和）、#13 尾块分支（运行时边界语义，先例佐证）——理由类别 = tile 级标量 + 顺序依赖 + 边界处理 + 调度结构，均非「实现简单」类借口；与 §6 循环结构一致（§6 无逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（[M, N] 行主进 / (M,) 行主出），核内布局与 lane 映射是设计变量。规约类算子必选候选：水平归约（lane 映射到归约轴 N）vs 垂直扫描（lane 映射到独立行 M、串行步进归约维）——正交决策分别枚举；本算子另有**chunk 划分维度**（BLOCK_N 粒度与尾 chunk 形态已在 §1.6.1 O2 定案，本节聚焦 lane/轴映射）。

**轴质量评分**（对选定轴 N〔chunk 内〕）：

| 评分项 | 判定 |
|--------|------|
| 整除性 | BLOCK_N pow2 % 128（fp16/bf16 lane）= 0 ✓ / % 64（fp32 lane）= 0 ✓（full chunk 零尾 lane）；尾 chunk TILE_R pow2 同整除，尾 lane 浪费 (TILE_R−rem)/TILE_R ≤ 50% 仅 1 chunk 摊薄（(TILE_R−rem)/N：wl3 rem=4096 → 0；测试 rem=232 → 24/33000 ≈ 0.07%） |
| 尾 lane 浪费率 | 主 workload（wl2/wl3 BLOCK_N=16384 整除或 pow2 rem）full chunk 0%；基线直译的**每 chunk** 49.2%（tile_n=16640→TILE_C 32768）浪费由 O2 消除 |
| 累加链形态 | 归约维 = chunk 内 stride-1 连续向量维（`T.reduce_*(dim=1)` on (SUB, BLOCK_N) fragment）；chunk 间 flash 合并为 (SUB,) 行维标量链（O4 已最小化：1 exp/行/chunk）；无跨步系数 |
| repack 代价 | 无（I/O 原生行主连续，核内布局与 I/O 同构；x_ub 紧凑 (SUB, BLOCK_N)，fragment pow2 同形——full chunk 直载零重排；`example_rmsnorm.py` L42 先例：仅 fragment 需 pow2，DMA 紧凑） |
| UB/VRF 容量影响 | chunk 驻留：x_ub 双版本 ≤194.4KB + x_frag ≤128KB VRF + stats_ub 8B（§4.5 预算表，SUB/BLOCK_N 联合封顶） |

**候选矩阵**（规约类候选全枚举，弃选给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | 行主 [M,N] 原生 + **chunk 内水平归约**（lane→N 连续轴，SUB 行子批 × BLOCK_N chunk 驻留，persistent 行块，chunk 串行 flash 合并） | N（chunk 内） | 无 | 连续 GM chunk 列窗口（wl2 每行每 chunk 32KB，32B 对齐 ✓）；归约维 = stride-1 向量维；online_softmax streaming + single 双先例；reduce 硬件树直连 | ✅ **主选** |
| 2 | 垂直扫描（lane→M 行，N 向串行步进） | M | 无 | N=32768 串行步进/行——向量宽度利用率 SUB/128 ≤ 12.5%（SUB≤16 且主流落点 SUB=1 → <1%），行内并行度归零；吞吐劣化 >8×（按 128 lane 满载比估算）；repo 无先例 | ❌ 串行化 N 维（本变体 N 为最大维），量化劣化不可接受 |
| 3 | 2D 分块 + 跨核 N 向归约（M/SUB × N/BLOCK_N grid，partial (m,s) 落 GM + 跨核合并） | N（块内） | partial 统计量 GM 往返 | 行块已提供 M/block_m 路独立任务（wl2 = 1024 >> 56 核饱和）——N 向再并行只增加 partial GM 往返（M×chunks×8B）+ 跨核同步/第二 kernel；chunk 串行依赖在行块内部、非并行度瓶颈（R1 #10 弃选同款） | ❌ 违反流量最小化（R3）；并行度非瓶颈 |
| 4 | 转置布局（列主驻留 + M 向归约） | M（转置后） | 核内转置链（`T.copy` transpose 仅特定路径——非通用，`tilelang/ascend/language/copy_op.py`；通用转置 API 存在性未命中→「未文档化假设」） | 转置 repack 开销 + UB 翻倍；收益仅在于把归约轴换到 M——M 维短（wl2 1024 vs N 32768）无收益；chunk 流式下转置链 × chunks 放大 | ❌ repack 纯开销、无轴收益（N 已是最长连续轴） |

**源码轴选择对照**（迁移任务必做）：源 GPU 也是 N 轴并行（chunk 内 `T.Parallel(block_m, tile_n)` 行内元素线程化 + warp 归约 dim=1）+ 行级 grid + chunk 串行——**轴结论一致**（N 是唯一高带宽连续轴，两代硬件共识），但承载结构完全重设计：GPU block×threads → SimtVF 线程域 + `T.reduce_*` 硬件树；GPU register stats → stats_ub；GPU tile_n 粒度 → BLOCK_N pow2（§0.6 R1/R2/R5）。GPU 轴选择是输入而非结论，已独立复核。

**布局决策结论**：选定方案 #1：核内布局 = `x_ub (SUB, BLOCK_N)` dtype 紧凑（双版本）+ `x_frag (SUB, BLOCK_N)` pow2 fp32（full chunk 直载；尾 chunk `(SUB, TILE_R)` masked）+ `stats_ub (SUB, 2)` fp32 驻留；向量化轴 = **N（chunk 内连续轴）**；repack = 无。该结论同步落入 §3.3 伪代码与 §6 循环结构（Parallel 内层维 = BLOCK_N/TILE_R 轴，buffer 形状 = §4.3）；弃选方案量化理由见候选矩阵（#2 串行化 >8×、#3 流量+往返非瓶颈、#4 repack 无收益）。

#### 1.6.4 实验裁决三件套（fp16-e 备选）

> 主选（fp32-e chunk 流式）与备选（fp16-e 精简链，仅 fp16 输入）的取舍判定裕度落在未实证常数区间（special_func 吞吐的 dtype 粒度——`CONST-valu-model` 未区分 fp16/fp32，A5 冷启动 assumption；fp16 h 前缀 intrinsic 潜在 2× 吞吐为「未文档化假设」，估算依据 = `src/ascend/codegen/intrin_rule_ascend.cc` L27–32 的 dtype 派发存在性），按 skill Phase 2 第 3 项强制产出三件套（与 single §1.6.4 同款判定框架，独立按 tiled workload 裁决）：

- **主选方案**：fp32-e BLOCK_N pow2 chunk 流式（本 DESIGN §3–§7 全部按此展开）——`x_frag (SUB, BLOCK_N)` fp32 直载/masked 载入，`T.exp` fp32 求值，`T.reduce_sum` fp32 硬件树，stats 合并全程 fp32；与源精度链一致（§0.1 fp32 中间语义；V3 vs golden ≤1 ulp）。
- **备选方案**（1 个，结构完整可实现，**仅 fp16 输入 dtype 分派**）：fp16-e 变体——chunk 内链改为：`x_frag` 载入输入 dtype（full chunk 直载省 cast 1 op/elem）、`m_c` 仍 fp32（`T.reduce_max` fp32——max 对 fp16 值域精确），减法 `x − m_run` 在 fp32 域完成后 **arg cast fp16** 再 `T.exp`（h 前缀 intrinsic），`T.reduce_sum` fp16 硬件树（`CONST-valu-model`：fp16 硬件直连 ✓），`s_c` 升 fp32 进合并（**stats_ub 合并链保持 fp32——rescale 精度不降**，这是与 single 备选的差异点：tiled 的跨 chunk 重缩放链次数多，合并域必须保 fp32）；`y = cast(m_run + log(fp32(s_run)), fp16)`。buffer 形状：x_frag (SUB, BLOCK_N) **fp16**（VRF 再 −50%）；UB 预算不变（x_ub 仍输入 dtype）；循环结构/分核三要素/persistent/chunk 流水均不变（§5.5 同款）；精度预估：verify_equiv.py **V7 模拟链 max_rel 3.645e-05**（1024×32768）/ **1.402e-05**（4×102400），@1e-2 违反率 0——守卫余量 ~280×（实测优于最坏树深估计，因 torch fp16 求和的分组舍入优于最坏估计；硬件树行为以 Stage 3 分层测试为准）。**bf16 输入不适用本备选**（bf16 尾数 8 位，s_c 树形相对误差预估超容差）——分派条件 `dtype == "float16"`。
- **实验裁决计划**：Stage 4 于代表 shape（bench 主对象）**wl2 (1024, 32768) bf16 不适用备选**（bf16 排除）→ 以 **wl2 的 fp16 等价 shape (1024, 32768) fp16** 与 **wl3 (4, 102400) fp16** 实测两变体 `do_bench(backend="msprof")` 时延 + 精度（vs golden 最大相对误差，容差 1e-2）；**判定阈值**：备选时延 < 主选 × 0.90 **且** max_rel_err ≤ 5e-3 → 采纳备选（工厂按 dtype 分派：fp16 → 备选，bf16/fp32 → 主选——per-(M,N,dtype,tile_n) JIT 缓存天然支持分派，达成总体最优而非全局二选一）；否则主选固化，实测数字经附录补记回写本节；实测反解 special_func fp16 吞吐常数 → 回写 `CONST-valu-model`（Tier 0，含 repro 三件套）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: simt（auto_schedule: yes）

### 2.2 选型理由

1. **迁移默认 + 算子特征**：纯 Vector 算子（elementwise cast/exp + 行规约 max/sum + 超越函数 + chunk 间标量合并，无 matmul/Cube 段）——决策树「含归约 → simt + reduce」路径；调度 prompt 指定默认 simt，重设计未产生 mixed 需求（无 Cube 工作落位，`T.MixedKernel` 不适用）。
2. **repo 先例**：`examples/pilot/online_softmax/online_softmax.py`（streaming 路径与本设计直接同构：simt + chunk T.serial + SimtVF + reduce_max/reduce_sum + stats_ub，63 用例全 PASS 的 measured 正例）；`examples/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py`（同族 per-chunk 计算体 PRECISION_PASS 实证）；`examples/ascend/example_rmsnorm.py` / `example_per_token_cast_to_fp8.py`（行规约 + persistent + SimtVF）——API 佐证链完整。
3. **从 simt 起步原则**（决策树 §2 注）：无证实热点前不降级 simd。降级触发条件（Stage 4 观察项）：msprof/npusim 证实 special_func 段为瓶颈且 simt 自动向量化不足时，可评估 SimdVF + `T.simd.vexpdif`（融合 exp-sub intrinsic，`tilelang/ascend/language/simd.py` L1182–1195，`examples/ascend/flash_attention/core.py` L130–133 在用）——收益预估仅省 1 sub @51 档（0.02 cycle/elem vs special 0.125 主项），优先级低于 fp16-e 备选（§1.6.4）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| VF 入口 | `with T.SimtVF(threads=min(1024, SUB·BLOCK_N))` + `T.Parallel(SUB, BLOCK_N)` 逐元素段 + `T.reduce_max/reduce_sum(dim=1, clear=True)` 行归约（硬件树直连）+ `T.Parallel(SUB)` 合并/lse 行维段 |
| 内存分配 | `T.alloc_shared`（UB：x_ub/stats_ub/lse_ub）+ `T.alloc_fragment`（VRF：x_frag/m_c/s_c/prev_frag/out_frag） |
| 同步 | auto_schedule: yes——AutoSchedule 管理跨引擎 flag 与 multibuffering（§7） |
| 模块声明 | kernel 模块 docstring 须声明 `programming_mode: simt`（迁移 prompt 契约，§10.3） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 优化后公式；每步优先向量化映射，与 §1.6.2 结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `m_c = max_j x_c[j]` | chunk 内行最大（fp32；full chunk 直载、尾 chunk masked pad=−inf 幂元） |
| 2 | `prev = m_run; m_run = max(m_run, m_c)` | running max 合并（flash 第一步） |
| 3 | `e_c = exp(x_c − m_run)` | 以合并后 max 为基的稳定化指数（fp32，原位覆写） |
| 4 | `s_c = Σ_j e_c[j]` | chunk 内行和（fp32 硬件树） |
| 5 | `s_run = s_run·exp(prev − m_run) + s_c` | running sum 重缩放合并（flash 第二步；exp 参数 ≤ 0 无溢出） |
| 6 | `y = cast(m_run + log(s_run), dtype)` | 循环出口 lse（每行 1 次 special + 1 次 cast），写出 (M,) |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| chunk 搬入 | X chunk 列窗口 GM→UB | `T.copy(X[row_base:row_base+rows, col:col+BLOCK_N], x_ub)` | 2D 连续 chunk 区域（full）；rows ∈ {SUB, last_sub_rows, tsub_rem} 静态 | simt（MTE2） |
| 尾 chunk 搬入 | 静态 rem 切片 | `T.copy(X[row_base:row_base+rows, col:N], x_ub[0:rows, 0:rem])` | 源 (rows, rem) 与目的切片形状匹配（online_softmax L142 同构） | simt（MTE2） |
| full chunk 直载 | x_ub → x_frag（fp32 提升） | `for r, i in T.Parallel(SUB, BLOCK_N): x_frag[r,i] = T.cast(x_ub[r,i], "float32")` | 无谓词（全列有效） | simt |
| 尾 chunk masked 载入 | x_ub → x_frag（pow2 填充 + fp32 提升） | `for r, i in T.Parallel(SUB, TILE_R): x_frag[r,i] = T.if_then_else(T.And(r < rows, i < rem), T.cast(x_ub[r,i], "float32"), -T.infinity(T.float32))` | pad = −inf（max/exp/sum 三幂元） | simt |
| chunk 行最大 | m_c | `T.reduce_max(x_frag, m_c, dim=1, clear=True)` | fragment 维归约 → AscendAllReduce 硬件树 | simt |
| max 合并 | prev/m_run | `for r in T.Parallel(SUB): prev_frag[r] = stats_ub[r,0]; stats_ub[r,0] = T.max(stats_ub[r,0], m_c[r])` | 行维向量化；stats_ub UB 驻留 | simt |
| 指数（原位） | e_c = exp(x−m_run) | `for r, i in T.Parallel(SUB, BLOCK_N): x_frag[r,i] = T.exp(x_frag[r,i] - stats_ub[r,0])` | fp32 全程；`tir.exp` 自动向量化 | simt |
| chunk 行和 | s_c | `T.reduce_sum(x_frag, s_c, dim=1, clear=True)` | fp32 fragment 硬件树 | simt |
| sum 合并 | s_run 重缩放 | `for r in T.Parallel(SUB): stats_ub[r,1] = stats_ub[r,1] * T.exp(prev_frag[r] - stats_ub[r,0]) + s_c[r]` | flash 重缩放；exp 参数 ≤ 0 | simt |
| 行输出 | y = cast(m_run+log(s_run), dtype) | `for r in T.Parallel(SUB): out_frag[r] = T.cast(stats_ub[r,0] + T.log(stats_ub[r,1]), dtype)` | 每行 1 次 log | simt（行维） |
| lse 暂存 | fragment→UB | `T.copy(out_frag, lse_ub)` | (SUB,) 一维整块 | simt |
| 搬出 | UB→GM | `T.copy(lse_ub[0:rows], y[row_base:row_base+rows])` | 1D 静态切片 | simt（MTE3） |
| stats 初始化 | m_run=−inf, s_run=0 | `with T.SimtVF(threads): for r in T.Parallel(SUB): stats_ub[r,0] = -T.infinity(T.float32); stats_ub[r,1] = T.float32(0.0)` | 每行块子批一次（online_softmax L108–111 同构） | simt |
| tile 调度 | — | `with T.Kernel(56) as core_id: for t in T.Persistent([num_tiles], 56, core_id, num_stages=2):` | persistent 行块 | — |
| 版本管理 | — | `T.annotate_buffer_versions({x_ub: 2, stats_ub: 1})` | chunk 间双缓冲 + stats 单版本 | — |
| 尾块分支 | — | `if t < full: … else: …`（TIR if，循环变量条件） | full/rem_m JIT 常量 | — |

### 3.3 计算伪代码

```python
# host 静态化（builder 闭包，per-(M, N, dtype, tile_n, block_m) JIT + lru_cache）：
#   BLOCK_N = 满足以下约束的最大 2 的幂：BLOCK_N <= tile_n 且 2*BLOCK_N*elem <= 199065   # 194.4KB
#   SUB     = 满足以下约束的最大 2 的幂：SUB <= block_m 且 2*SUB*BLOCK_N*elem <= 199065
#                                            且 SUB*BLOCK_N*4 <= 131072                    # 128KB VRF
#   num_full = N // BLOCK_N;  rem = N % BLOCK_N;  TILE_R = next_pow2(rem) if rem else 0
#   full = M // block_m;  rem_m = M % block_m;  num_tiles = full + (1 if rem_m else 0)
#   num_sub = block_m // SUB;  last_sub_rows = block_m % SUB
#   tsub = rem_m // SUB;  tsub_rem = rem_m % SUB
#   threads = min(1024, SUB * BLOCK_N)

@T.prim_func
def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M,), dtype)):
    with T.Kernel(56) as core_id:
        x_ub = T.alloc_shared((SUB, BLOCK_N), dtype)        # chunk 驻留（紧凑，双版本流水）
        stats_ub = T.alloc_shared((SUB, 2), "float32")      # running (m_run, s_run) 跨 chunk 驻留
        lse_ub = T.alloc_shared((SUB,), dtype)              # 行输出暂存（单版本）
        T.annotate_buffer_versions({x_ub: 2, stats_ub: 1})
        for t in T.Persistent([num_tiles], 56, core_id, num_stages=2):
            base = t * block_m
            if t < full:                                     # ---- 全块 tile（TIR if）----
                for s in T.serial(num_sub):
                    row_base = base + s * SUB
                    <子批体 rows=SUB>                        # 行谓词恒真
                if last_sub_rows > 0:                        # Python 静态分支（block_m % SUB）
                    row_base = base + num_sub * SUB
                    <子批体 rows=last_sub_rows>              # 行谓词 r < last_sub_rows 生效
            else:                                            # ---- 尾块 tile（rem_m 行）----
                for s in T.serial(tsub):
                    row_base = base + s * SUB
                    <子批体 rows=SUB>
                if tsub_rem > 0:                             # Python 静态分支
                    row_base = M - tsub_rem
                    <子批体 rows=tsub_rem>                   # 静态 rem 切片（同 single）

# ---- 子批体（rows 行 × N 列 online 流式；行谓词 pred_r = (r < rows) 于载入处生效）----
#   with T.SimtVF(threads=threads):                         # stats 初始化（每子批）
#       for r in T.Parallel(SUB):
#           stats_ub[r, 0] = -T.infinity(T.float32)
#           stats_ub[r, 1] = T.float32(0.0)
#   for c in T.serial(num_full):                            # full chunks（全列有效）
#       col = c * BLOCK_N
#       T.copy(x[row_base:row_base+rows, col:col+BLOCK_N], x_ub[0:rows, 0:BLOCK_N])
#       with T.SimtVF(threads=threads):
#           x_frag = T.alloc_fragment((SUB, BLOCK_N), "float32")
#           m_c = T.alloc_fragment((SUB,), "float32")
#           s_c = T.alloc_fragment((SUB,), "float32")
#           prev_frag = T.alloc_fragment((SUB,), "float32")
#           for r, i in T.Parallel(SUB, BLOCK_N):           # ① 无谓词直载
#               x_frag[r, i] = T.cast(x_ub[r, i], "float32")
#           T.reduce_max(x_frag, m_c, dim=1, clear=True)    # ② chunk 行最大
#           for r in T.Parallel(SUB):                       # ③ max 合并（源式）
#               prev_frag[r] = stats_ub[r, 0]
#               stats_ub[r, 0] = T.max(stats_ub[r, 0], m_c[r])
#           for r, i in T.Parallel(SUB, BLOCK_N):           # ④ 原位 exp(x − m_run)
#               x_frag[r, i] = T.exp(x_frag[r, i] - stats_ub[r, 0])
#           T.reduce_sum(x_frag, s_c, dim=1, clear=True)    # ⑤ chunk 行和
#           for r in T.Parallel(SUB):                       # ⑥ flash 重缩放合并
#               stats_ub[r, 1] = stats_ub[r, 1] * T.exp(prev_frag[r] - stats_ub[r, 0]) + s_c[r]
#   if rem > 0:                                             # Python 静态：尾 chunk（rem 列）
#       col = num_full * BLOCK_N
#       T.copy(x[row_base:row_base+rows, col:N], x_ub[0:rows, 0:rem])
#       with T.SimtVF(threads=threads):
#           x_frag = T.alloc_fragment((SUB, TILE_R), "float32")   # pow2 填充
#           <同 ②–⑥，载入改 masked：
#            for r, i in T.Parallel(SUB, TILE_R):
#                x_frag[r, i] = T.if_then_else(T.And(r < rows, i < rem),
#                    T.cast(x_ub[r, i], "float32"), -T.infinity(T.float32))>
#   with T.SimtVF(threads=threads):                         # lse 收尾 + 两段写
#       out_frag = T.alloc_fragment((SUB,), dtype)
#       for r in T.Parallel(SUB):
#           out_frag[r] = T.cast(stats_ub[r, 0] + T.log(stats_ub[r, 1]), dtype)
#       T.copy(out_frag, lse_ub)
#   T.copy(lse_ub[0:rows], y[row_base:row_base+rows])
```

> 尾 chunk 载入行谓词 `r < rows` 与源 GPU masked 分支 L152 的 `pid_m·block_m + i < M` 语义一致（行 OOB 以 −inf 载入、计算结果不写出）；full chunk 全列有效（N ≥ col+BLOCK_N 保证）无谓词。被丢弃行（r ≥ rows）内部链产生 NaN 但不写出；reduce_* 沿 dim=1 逐行独立、合并沿行维逐行独立，故不污染有效行。伪代码与 §1.6.3 布局决策、§4.3 buffer 形状、§6 循环结构三方一致。

### 3.4 API 可行性确认

| API | 佐证（examples / testing / 源码） |
|-----|----------------------------------|
| `T.Kernel(56)` 一维 + 无 threads | `examples/pilot/online_softmax/online_softmax.py` L57；`examples/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py` L134（PRECISION_PASS）；ascend-constraints §1 条 1 |
| `T.Persistent([num_tiles], 56, core_id, num_stages=2)` | online_softmax L60（streaming：chunk 循环外层）；single L138 |
| `T.serial(num_full)` chunk 串行（Persistent 体内） | online_softmax L113/171（streaming 两段 T.serial(chunk)）；`D-REDUCE-SIMTVF-API` 注记（`T.serial(n)` 单参等价 range） |
| `T.SimtVF(threads=)` + `T.Parallel` + 直载/掩码载入 | online_softmax L116–123（full chunk 直载）/ L143–152（tail chunk masked）；single L145–155 |
| `T.reduce_max / T.reduce_sum(dim=1, clear=True)` | pattern-library `D-REDUCE-SIMTVF-API`（measured：AscendAllReduce 硬件树实见）；single L156/L159；`testing/ascend/language/test_tilelang_ascend_reduce.py` |
| `stats_ub` UB 驻留 + SimtVF 域内读写 + `annotate_buffer_versions({x_ub: 2, stats_ub: 1})` | online_softmax streaming L102–111（**完整同构先例**，measured 63/63 PASS）；tilelang-ascend skill §4.2 |
| `T.exp`（fp32 向量化）/ `T.log` | `D-ELEM-SIMTVF-AUTOVEC`（measured：`ascend_simdvf_lower_parallel.cc:409-412`）；online_softmax L169（T.log）；`intrin_rule_ascend.cc` L34–49 注册 |
| `T.cast(dtype→"float32"/dtype)`（含 bfloat16） | online_softmax L73/L83；`examples/ascend/flash_attention/core.py` L78–100（bf16 cast 链）；single L153 |
| `T.copy` GM→UB 2D chunk 列窗口（紧凑） | online_softmax L115（`X[base:base+BM, col:col+BLOCK_N]` chunk 拷贝先例）；single L144 |
| `T.copy` 静态 rem 切片（源/目的形状匹配） | online_softmax L142（`x_ub[0:BM, 0:rem]`）；single L219（`x_ub[0:tsub_rem, 0:N]`——**目的切片写法**：源区域 (rows, rem) 与 x_ub 切片 (rows, rem) 匹配） |
| `T.copy` fragment→UB + UB→GM 1D 切片 | single L162–163；`example_per_token_cast_to_fp8.py` L81–83（(block_groups,) 两段写同构） |
| `T.infinity(T.float32)` / `T.float32(0.0)` / `T.max` | online_softmax L110–111/`sentinel`；`D-REDUCE-SIMTVF-API` 注记（接受 dtype 对象）；`T.max` single L169 同款在用 |
| `T.And(a, b)` 联合谓词 | single masked 载入段（Stage 3 实证：`tilelang/language/ast/ir.py` L57/L2209 导出） |
| 核内 TIR if（循环变量条件） | `examples/ascend/example_gemm.py` L43–56（Persistent 体内 `if row_idx < MAIN_ROW:`）；single L140（Stage 3 实证含尾块路径） |
| Python 级常量 if（prim_func 体内） | online_softmax L140（`if rem > 0:`）；single L164/217 |
| `tilelang.next_power_of_2` | single L102；online_softmax L92（`TILE_C`） |
| `@tilelang.jit(target="ascend", out_idx=[1])` 懒工厂 | online_softmax L215–223（`_is_lazy_style` 形态，S3-JIT 门禁要求）；single L105 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（ascend-constraints §2 强制检测）

| 检测项 | 本算子结论 |
|--------|-----------|
| 一维 Kernel / threads= 误用 | ✅ 源含 `T.Kernel(grid, threads=threads)`（二维参数 + threads）——**已重设计**：一维 `T.Kernel(56)` + `T.SimtVF(threads=)`（§0.6 R1）；wrapper 调用面按 K3 移除 threads（§4.7） |
| GPU 专用 API | ✅ 无 TMA/cp.async/warp shuffle/cluster 依赖（源用 `T.copy`/`T.reduce_*`/`T.Parallel`/`T.Serial`，表层共享）；SMEM 两跳 staging 等价替换为 UB 单跳（§0.5） |
| C0 分形对齐 | ✅ 不适用（无 GEMM/Cube 段；`CONST-gemm-dim-constraints` 仅约束 `T.gemm` 路径） |
| L0C/UB 溋出 | ✅ 无 L0C；UB 预算 §4.5（x_ub 双版本 + stats_ub + lse_ub ≤ 194.4KB 90% 折扣内）；VRF §4.5（≤128KB，SUB·BLOCK_N ≤ 32768 封顶） |
| OOB padding 误用 | ✅ 不使用 `pad_value=`/`data_select=`（列向 padding 语义不适用于行 OOB——tilelang-ascend skill §6.1；行尾用 fragment 谓词 + 静态切片规避，§0.6 R4） |
| 分核策略缺失 | ✅ §5.5 三要素齐备（grid 56 实查 / 纯 AIV 分工 / auto_schedule 声明） |
| 静态循环边界 | ✅ 全部 JIT 常量（num_tiles/num_sub/num_full/rem/TILE_R/BLOCK_N/SUB/threads/tsub/tsub_rem/last_sub_rows）；核内串行循环边界静态 |
| flag id 预算 | ✅ auto_schedule yes，无手动 flag（§7） |
| Host 侧输入操作 | ✅ host 仅做 shape/元数据推导（BLOCK_N/SUB/full/rem）与 JIT 缓存；不改输入张量内容（op 层已保证 contiguous/reshape，本 kernel 无 host 预处理） |

#### 3.5.2 参考实现差异说明

源 GPU 实现与本设计的差异全部源于 §0.5/§0.6 迁移决策（tile_n→BLOCK_N pow2、N_padded 对齐→紧凑、双路径→full/tail 二分支、grid→persistent+子批、threads→SimtVF、register stats→stats_ub、隐式 OOB→静态分解），递推公式与数值语义恒等（verify_equiv.py V4 位级一致）。**不复用源码任何 GPU 形态结构**（`T.Kernel(grid, threads)`、SMEM staging、`_needs_mask` 分支、register 跨循环 fragment）。

#### 3.5.3 本项目同类实现参考

| 参考文件 | 复用内容 |
|---------|---------|
| `examples/pilot/online_softmax/online_softmax.py` | streaming 路径直接同构：Persistent + T.serial(chunk) + stats_ub (BM,2) UB 驻留 + flash 合并 + full chunk 直载 / tail chunk 静态 rem 切片 masked + 双版本 x_ub——本设计 chunk 流水与 running stats 结构的 measured 正例 |
| `examples/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py` | 同族 per-chunk 计算体（masked 载入 / reduce_max / exp 原位 / reduce_sum / lse+cast / 两段写）+ 尾块 TIR if + 子批 SUB 机制 + host 静态化模式（PRECISION_PASS 实证；其 Stage 3 复盘明示本变体复用该计算体） |
| `examples/ascend/example_per_token_cast_to_fp8.py` | (M,) 一维输出两段写；SimtVF 内 reduce_max；`T.Persistent` + `annotate_buffer_versions` 三件套 |
| `examples/ascend/example_rmsnorm.py` | 紧凑 UB + pow2 fragment masked 先例（L42 注释）；UB 预算化行批处理思路 |
| `examples/ascend/example_gemm.py` | 核内循环变量 TIR if 先例（aswt_swizzle L47） |
| `examples/ascend/flash_attention/core.py` | bf16 cast 链先例（L78–100） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| x | (M, N)（per-shape JIT 常量；wl2 = (1024, 32768)、wl3 = (4, 102400)） | float16 / bfloat16 / float32 | 输入矩阵，行主连续（op 层 `movedim + contiguous + reshape` 保证；契约空间见 §0.2——N 超驻留或显式 tile_n）；host 侧仅元数据视图操作 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| y | (M,) | 同输入 dtype | 行级 logsumexp 结果；kernel 显式输出参数（`out_idx=[1]`）；写出量 M×elem_bytes（wl2 = 2KB，可忽略 vs 读 64MB） |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| x_ub | (SUB, BLOCK_N)（wl2/wl3 = (1, 16384)；显式 bm=16/tile_n=6912 场景 = (8, 4096)） | 输入 dtype | UB（`T.alloc_shared`，双版本） | chunk DMA 暂存（紧凑，不搬 pad 列；chunk 间流水） |
| stats_ub | (SUB, 2) | float32 | UB（单版本） | running (m_run, s_run) 跨 chunk 驻留（SimtVF 域内读写，§0.6 R5） |
| x_frag | (SUB, BLOCK_N)（尾 chunk 为 (SUB, TILE_R)） | float32 | VRF（`T.alloc_fragment`） | chunk 数据 fp32 提升（full 直载 / 尾 masked pad=−inf）；**原位覆写为 e**（单 fragment） |
| m_c / s_c / prev_frag | (SUB,) | float32 | VRF | chunk 统计量 / 合并暂存（标量级，≤128B） |
| out_frag | (SUB,) | 输入 dtype | VRF | 每行 lse（cast 后） |
| lse_ub | (SUB,) | 输入 dtype | UB（单版本） | lse 写出暂存（fragment→UB→GM 两段写） |

### 4.4 内存搬运路径

```
GM[x (M,N) dtype]
  --T.copy (MTE2, 2D chunk 列窗口 [row_base:+rows, col:+BLOCK_N], x_ub 双版本 v/v⊕1)--> UB[x_ub (SUB,BLOCK_N)]
  --SimtVF T.Parallel (full chunk 无谓词直载 / 尾 chunk 行∧列谓词 pad=-inf, cast fp32)--> VRF[x_frag (SUB,BLOCK_N 或 SUB,TILE_R) fp32]
  → VRF reduce_max → m_c → UB[stats_ub[r,0]] ← max 合并（prev 驻留 prev_frag）
  → VRF x_frag 原位 = exp(x_frag − m_run) → VRF reduce_sum → s_c
  → UB[stats_ub[r,1]] ← s_run·exp(prev−m_run) + s_c（flash 重缩放）
  [chunk 循环 num_full 次 + 尾 chunk 1 次，x_ub 版本交替]
  → VRF out_frag = cast(m_run + log(s_run), dtype)
  --T.copy (fragment→UB)--> UB[lse_ub (SUB,) dtype]
  --T.copy (MTE3, 1D 静态切片 [row_base:+rows])--> GM[y (M,)]
```

无 L1/L0A/L0B/L0C 段（纯 Vector）；无中间量落 GM（running stats 驻留 UB）；x 单遍 GM 读（chunk 流式恰读一次——R3 口径：扫描遍数 2 = 读 1 + 写 1）、y 单遍 GM 写。

### 4.5 UB / VRF 内存预算

**UB 预算**（上限 216KB〔SimtVF〕，`CONST-capacity-a5` assumption 按 90% 折扣 = 194.4KB = 199065B；以下按单 workload 口径分列——两表各自独立求和，勿跨 workload 叠加）：

wl2 (M=1024, N=32768, bf16, block_m=1, tile_n=16384 → BLOCK_N=16384, SUB=1)：

| Buffer | Shape × dtype | 大小 | 版本 | 说明 |
|--------|--------------|------|------|------|
| x_ub | (1, 16384) × 2B × 2 版本 | 64KB | 2 | chunk 双缓冲 |
| stats_ub | (1, 2) × 4B | 8B | 1 | running stats |
| lse_ub | (1,) × 2B | 2B | 1 | 写出暂存 |
| 编译器保留裕量 | — | ~8KB | — | 小 fragment/标量 planned-into-UB 开销（`UB_SAFETY_RESERVE_BYTES` 同量级） |
| **wl2 总计** | | **≈72.1KB / 194.4KB ✓** | | |

wl3 (M=4, N=102400, fp16, block_m=1, tile_n=25600 → BLOCK_N=16384, SUB=1) 与 fp32 档 (M=4, N=32768, fp32, tile_n=16384 → BLOCK_N=16384, SUB=1)：

| Buffer | wl3 (fp16) | fp32 档 | 版本 | 说明 |
|--------|-----------|---------|------|------|
| x_ub | (1, 16384) × 2B × 2 = 64KB | (1, 16384) × 4B × 2 = 128KB | 2 | |
| stats_ub + lse_ub | 10B | 12B | 1 | |
| 编译器保留裕量 | ~8KB | ~8KB | — | |
| **单 workload 总计** | **≈72.1KB / 194.4KB ✓** | **≈136.1KB / 194.4KB ✓** | | |

显式 config 大 block_m 场景（bm=16, tile_n=6912, N=32768, fp16 → BLOCK_N=4096, SUB=8）：x_ub (8, 4096)×2B×2 = 128KB + stats 64B + lse 16B + ~8KB ≈ **136.1KB / 194.4KB ✓**。

**BLOCK_N / SUB 联合推导**（host 静态化公式，§3.3 注）：`BLOCK_N = 满足以下约束的最大 2 的幂：BLOCK_N ≤ tile_n 且 2·BLOCK_N·elem ≤ 199065`（fp16/bf16 上界 32768、fp32 上界 16384——显式 config 超大 tile_n 由该 clamp 兜底；wrapper 预算下实际 ≤ 16384）；`SUB = 满足以下约束的最大 2 的幂：SUB ≤ block_m 且 2·SUB·BLOCK_N·elem ≤ 199065 且 SUB·BLOCK_N·4 ≤ 131072`。该式保证**契约空间全域**（任意 N、任意 block_m ≥ 1、任意 tile_n ≥ 1、三 dtype）UB 双版本与 VRF 同时达标且 SUB ≥ 1 恒有解（BLOCK_N ≤ 16384 时 fp32 亦满足 2·1·16384·4 = 128KB ≤ 194.4KB ∧ 65536 ≤ 131072）。

**VRF 预算**（假设上限 256KB，`CONST-vrf-capacity-a5` assumption；**VRF 为 per-thread 寄存器数组，不计入 UB 216KB 预算**——实测消解口径见该条目：online_softmax UB 128KB + 双 fragment 256KB 并存 76 reg/thread spill 0）：

| Buffer | wl2/wl3（SUB=1, BLOCK_N=16384） | 显式 bm=16（SUB=8, BLOCK_N=4096） | 说明 |
|--------|-------------------------------|-----------------------------------|------|
| x_frag (SUB, BLOCK_N) fp32 | 64KB | 128KB | ≤128KB 封顶（SUB·BLOCK_N ≤ 32768 约束第二条） |
| m_c/s_c/prev/out_frag | ~32B | ~128B | 标量 |
| **合计 / thread** | 64KB/1024 threads ≈ **16 fp32 reg/thread** | 128KB/1024 ≈ **32 reg/thread** | **远低于 76 reg 实测水位**（`CONST-vrf-capacity-a5`）；尾 chunk x_frag (SUB, TILE_R) ≤ (SUB, BLOCK_N) 同域更小 |

**对照基线（无 BLOCK_N pow2 重划的 tile_n 直译）**：tile_n=16640 场景 x_frag (1, 32768) fp32 = 128KB **贴 VRF 封顶**且有效 lane 50.8%；tile_n=25600 场景同 128KB 贴顶、78.1%。BLOCK_N 机制使 VRF 回落 64KB（主流落点）并恢复满有效 lane（§1.6.1 O2 收益）。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| M | JIT 闭包参数（`T.Tensor((M, N), ...)` 符号维度），per-shape 编译 + `lru_cache` | ≥ 1 任意（含 M < block_m：full=0 全尾块路径；M % block_m ≠ 0 尾 tile；wl3 M=4 并行度受限记录 §9.1 风险 2） |
| N | 同上 | ≥ 1 任意（**无 single 的 32512 上限**——任意大 N 是本变体语义；N < BLOCK_N 时 num_full=0 全尾 chunk 路径） |
| block_m | `_func(block_m)` 调用参数（per-config 编译） | wrapper 启发式 tiled 落点 1（主流）+ 显式 config 任意 ≥1（SUB 机制兜底） |
| tile_n | builder 参数（契约输入） | ≥ 1 任意（wrapper 产 256 倍数 ≤ 32512；显式 config 任意——BLOCK_N clamp 兜底） |
| dtype | JIT 闭包参数 | float16 / bfloat16 / float32 |

### 4.7 JIT 配置与调用面契约

```python
import tilelang
import tilelang.ascend.language as T


def _logsumexp_kernel_tiled(M: int, N: int, dtype: str, tile_n: int):
    """Build a multi-tile (online streaming) logsumexp kernel — A5 re-implementation.

    Keeps the wrapper call surface `_logsumexp_kernel_tiled(M, N, dtype, tile_n)
    (block_m)(x)`; the GPU `threads` argument is removed per K3 (threads derived
    inside: min(1024, SUB * BLOCK_N)).  tile_n is honoured as a budget cap:
    the internal chunk width BLOCK_N is the largest power of two with
    BLOCK_N <= tile_n and 2*BLOCK_N*elem <= 194.4KB (flash-merge associativity
    makes the re-partition exact — verify_equiv.py V1/V6).
    """
    # host 静态化：BLOCK_N / SUB / num_full / rem / TILE_R / num_sub /
    # last_sub_rows / full / rem_m / num_tiles / tsub / tsub_rem / threads
    # + 契约防御：assert M >= 1, N >= 1, tile_n >= 1, block_m >= 1
    @tilelang.jit(target="ascend", out_idx=[1])
    def _func(block_m):
        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M,), dtype)):
            ...  # §3.3 伪代码
        return main
    return _func


# wrapper 对接（logsumexp.py L63–68 / L83 现行调用面，Stage 5 集成不改 wrapper）：
#   _logsumexp_kernel(M, N, dtype_str, tile_n)  ->  _logsumexp_kernel_tiled(M, N, dtype, tile_n)
#   kernel = _logsumexp_kernel_tiled(M, N, dtype, tile_n)(block_m)   # (block_m) JIT 编译
#   y = kernel(x)                                                    # (x) 执行，out_idx=[1] 返回 y
```

- **JIT 形态**：`@tilelang.jit(target="ascend", out_idx=[1])` 懒工厂（工厂返回 PrimFunc，`_is_lazy_style` 识别）——S3-JIT 门禁要求（CASE-pilot-online-softmax 教训：`tilelang.compile` 形态被拒）；per-(M, N, dtype, tile_n, block_m) 编译由 jit 装饰器内建缓存承接（wrapper 侧另有 `lru_cache` dispatch）。
- **K3 落地**：`_func(block_m)` 单参数（GPU 版为 `(block_m, threads)`）；wrapper `LogSumExpKernel.forward` → `_logsumexp_fwd_wrapped(...)(block_m)(x)` 调用面不变。
- **tile_n 适配（后端实现相关参数）**：签名与接口不变；内部语义 = chunk 预算上限（BLOCK_N pow2 ≤ tile_n）——与 K3 移除 threads 同类的后端参数适配（GPU tile_n 由 SMEM 256 元素对齐驱动，A5 无该约束而有 fragment pow2 约束；§0.5 判定 + §0.6 R2 语义保持论证 + V1/V6 机器验证）。
- **host 防御**（TRAP-GRID-DIVISIBILITY 纪律）：builder 内 `assert` 契约（M ≥ 1、N ≥ 1、tile_n ≥ 1、block_m ≥ 1）——persistent 调度器保证 tile 全覆盖（无手动 grid-stride 漏写风险），assert 仅拦契约外误用。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 算子仅含逐元素运算（cast/exp/sub/log）与行内规约（max/sum）及行维标量合并（flash 重缩放），无 matmul——无 Cube 段，全部计算与搬运落 AIV（56 个）。

### 5.2 Block 划分

```python
BLOCK_N   = 满足以下约束的最大 2 的幂：BLOCK_N <= tile_n 且 2*BLOCK_N*elem <= 199065
            # chunk 列宽（fragment 同形 pow2）：wl2/wl3 = 16384；显式 tile_n=6912 → 4096
SUB       = 满足以下约束的最大 2 的幂：SUB <= block_m 且 2*SUB*BLOCK_N*elem <= 199065
            且 SUB*BLOCK_N*4 <= 131072
            # wl2/wl3 (block_m=1): SUB=1；显式 (bm=16, BLOCK_N=4096, fp16): SUB=8
num_full  = N // BLOCK_N        # wl2: 2；wl3: 6；测试 (33,33000): 2
rem       = N % BLOCK_N         # wl2: 0；wl3: 4096；测试 (33,33000): 232
TILE_R    = next_pow2(rem)      # 尾 chunk fragment 宽（rem=0 时无尾 chunk）：wl3 4096；232→256
num_sub   = block_m // SUB      # wl2/wl3: 1（单迭代退化）；显式 bm=16/SUB=8: 2
full      = M // block_m        # wl2: 1024；wl3: 4
rem_m     = M % block_m         # wl2: 0；wl3: 0（bm=1 恒整除）；测试 (33,32768,bm=4): 1
num_tiles = full + (1 if rem_m else 0)   # persistent 域：wl2 1024；wl3 4
threads   = min(1024, SUB * BLOCK_N)     # wl2/wl3: 1024；小 BLOCK_N 显式场景自动收缩
NUM_BLOCKS = 56                 # = vector_core_num（实查，§5.5）
```

- **BLOCK_N 依据**（§0.6 R2 / §4.5）：① ≤ tile_n（尊重 wrapper 预算意图——tile_n 已编码 216KB UB 预算）；② 2·BLOCK_N·elem ≤ 194.4KB（x_ub 双版本下 SUB=1 也达标）；③ pow2（fragment 形状约束 + lane 满有效——非 pow2 tile_n 直译的 50.8%/78.1% 有效 lane 消除）；④ 最大化（chunk 数最少 → 合并标量与 DMA 迭代最少）。
- **SUB 依据**（同 single 机制）：① x_ub 双版本 ≤ 194.4KB（chunk 间 MTE2/VALU 流水）；② x_frag ≤ 128KB VRF（≤32 reg/thread，spill 风险消除）；③ SUB ≤ block_m（契约参数不放大）。**契约主流落点 block_m=1 → SUB=1 → num_sub=1**（子批循环单迭代退化——嵌套简化为 online_softmax streaming 同构）；显式 config bm>1 由 SUB 封顶兜底。
- **block_m/tile_n 契约来源**：wrapper `default_config`（tiled 落点 block_m=1：wl2 tile_n=16384、wl3 tile_n=25600，§0.2 推导）或 caller 显式 config——kernel 侧不重选、不假设整除（M 尾块 §5.4；tile_n 任意值 BLOCK_N clamp 兜底）。

### 5.3 约束分析

- **C0 分形对齐**: 不适用（无 GEMM/Cube 段；`CONST-gemm-dim-constraints` 仅约束 `T.gemm` 路径）。
- **UB 容量**: §4.5 预算表（wl2 ≈72.1KB / wl3 ≈72.1KB / fp32 档 ≈136.1KB / 显式大 bm ≈136.1KB，均 ≤ 194.4KB，`CONST-capacity-a5` assumption 90% 折扣）✓。
- **VRF 容量**: §4.5（≤128KB ≈ ≤32 reg/thread，远低于 76 reg 实测水位，`CONST-vrf-capacity-a5`）；Stage 3 编译后首查 spill 计数（`get_kernel_source()`）复核。
- **GM→UB 对齐**: chunk 列窗口 [col:+BLOCK_N] × 行 [row_base:+rows]——wl2 每行每 chunk 32KB / wl3 16KB，32B 对齐 ✓；fp32 档 64KB ✓。非 32B 行长（测试 (33,33000) fp16：行 66000B，chunk 起点 col×2B 对齐性随 col 变化——33000×2=66000 非 32 倍数）：DMA 字节级行长支持（tilelang-ascend skill §6.1 "byte-granular rows"——正确性不受影响），带宽档降级列 §9.1 观察项（同 single 风险 2）。
- **向量宽度整除**: BLOCK_N=16384/4096 % 128（fp16 lane）= 0、% 64（fp32 lane）= 0 ✓；尾 chunk TILE_R pow2 同整除（尾 lane 由 masked 吸收，§1.6.3）。
- **静态循环边界**: num_tiles / num_sub / num_full / rem / TILE_R / BLOCK_N / SUB / threads / tsub / tsub_rem / last_sub_rows 全 JIT 常量 ✓（gate 约束）。

### 5.4 注意事项（非整除 / 边界处理策略）

- **行尾（M % block_m ≠ 0）**：静态 full/rem_m 分解 + 尾 tile TIR if 分支（`if t < full`）——尾 tile 满子批 `T.serial(tsub)` + Python 静态 `if tsub_rem > 0` 末子批；全部 `T.copy` 切片静态；fragment 载入行谓词 `r < rows`（越界行 −inf 载入、不写出）——**不依赖 `T.copy` 越界行为**（§0.6 R4；single Stage 3 实证同款路径）。
- **列尾（N % BLOCK_N ≠ 0）**：尾 chunk 静态 rem 切片 + `(SUB, TILE_R)` masked 载入（pad=−inf 三幂元中性，V1 位级验证）——数值无影响；TILE_R 相对 rem 的翻倍浪费摊薄到全行 ≤ (TILE_R−rem)/N（wl3 rem=4096 恰 pow2 零浪费；(33,33000) rem=232 → 24/33000 ≈ 0.07%）。
- **M < block_m**（含 M=1）：full=0、rem_m=M → 单尾 tile 路径全覆盖。
- **N < BLOCK_N**（小 N 显式路由）：num_full=0、rem=N → 单尾 chunk 路径全覆盖（TILE_R = next_pow2(N)）。
- **56 核尾波**：wl2 num_tiles=1024 = 56×18+16（末波 16/56 核，波次量化 ~8.7% 于 18 波尾）；wl3 num_tiles=4 < 56（52 核空转——M=4 负载特性，§9.1 风险 2 记录）。Stage 4 可调 block_m（wrapper config 参数，不改 kernel）。
- **grid 整除性**（TRAP-GRID-DIVISIBILITY）：persistent 调度器保证 tile 全覆盖——无手动 grid-stride 的静默漏写风险；host assert 防御契约误用（§4.7）。

### 5.5 分核策略（grid 划分 + AIC/AIV 分工 + 同步策略）⭐

> 三要素判定标准与核数实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（依据 `.agents/skills/tilelang-ascend/SKILL.md` §2/§3/§7）。

- **grid 划分（含核数依据）**：`num_blocks = NUM_BLOCKS = 56`（一维 `T.Kernel(56)`，persistent 单波常驻）；行块任务 `num_tiles = ceil(M/block_m)`（wl2: 1024、wl3: 4）由 `T.Persistent([num_tiles], 56, core_id, num_stages=2)` 调度器分配（多波：wl2 1024/56 ≈ 18.3 tiles/核 + 尾波 16 tiles；wl3 单波 4/56）。**核数实查记录**（2026-10-16，本设计执行期亲测——与 single 任务同日记录一致，本次复测确认）：
  ```python
  import torch
  props = torch.npu.get_device_properties()
  # name: Ascend950PR_9579, cube_core_num: 28, vector_core_num: 56
  #       total_memory: 132338679808 (~126GB), L2_cache_size: 134217728 (128MB)
  ```
  纯 Vector 算子按 `vector_core_num = 56` 取 num_blocks（persistent 单波常驻，无超发串行）；cube_core_num=28 记录备查（本算子不占 AIC）。无真机环境的复现命令即上述查询代码；本仓 examples 惯例初值 32 blocks 被实测值 56 取代（`CONST-arch-a5` measured 条目：28 = 4×7 非 2 幂——grid 整除性陷阱见 TRAP-GRID-DIVISIBILITY，本设计 persistent 形态免疫）。
- **AIC/AIV 分工**：纯 Vector 算子——逐元素（cast/exp/sub/log）、行归约（`T.reduce_max/reduce_sum` 硬件树）、行维标量合并（flash 重缩放 `T.Parallel(SUB)`）、数据搬运（`T.copy` MTE2/MTE3）全部落 **AIV**（56 个）；**无 Cube 工作**（无 `T.gemm`/L1/L0 段）→ 不占 AIC（28 个闲置）。pure kernel 的核型由 codegen 自动判定（core-split-strategy §1 ②），无需 `T.MixedKernel`/sid 声明；无 dual_copy 所有权问题（无 L0C→UB 段）。
- **同步策略（auto|manual 声明）**：**auto_schedule: yes**（默认）——依赖 AutoSchedule 自动插 flag/同步，数据依赖边界：① x_ub 版本 v 的 MTE2 写 → SimtVF 读（跨引擎，自动 flag 对）；② stats_ub 的 VALU 写（chunk c 合并）→ 下一次 chunk 的 VALU 读（同引擎顺序语义）+ lse 收尾 VALU 读（同引擎）；③ lse_ub 的 VALU 写（`T.copy(out_frag, lse_ub)`）→ MTE3 读（跨引擎，自动 flag 对）；④ x_ub 版本 v/v⊕1 交替（multibuffering，`T.annotate_buffer_versions({x_ub: 2})` + 编译器管理）；⑤ SimtVF 内跨线程 UB 读写 hazard → 自动 `asc_syncthreads`（tilelang-ascend skill §2.2/§10.8，回归 `testing/ascend/transform/test_ascend_thread_sync.py`）；⑥ reduce 终值 → 合并消费的 VALU 内顺序依赖。无手动 `T.ascend_set_flag/wait_flag`（核内 8/跨核 16 flag 预算不占用）；调度提示使用位置：`T.Persistent(num_stages=2)`（chunk 间流水）——不使用 `T.Stage`/`T.Task`（AutoSchedule 足以覆盖本算子依赖结构）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表仅 persistent / tile / 子批 / chunk 级调度循环；无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 行块（M 向） | persistent 调度 | `T.Persistent([num_tiles], 56, core_id, num_stages=2)` | 行块在 56 AIV 间持久调度 + chunk 间流水 |
| 尾块分支 | 运行时控制流 | `if t < full: … else: …`（TIR if） | M 尾块边界语义（§1.6.2 #13；example_gemm L47 + single Stage 3 先例） |
| 行子批（block_m 内） | 串行迭代 | `T.serial(num_sub)`（全块）/ `T.serial(tsub)` + 静态末子批（尾块） | x_ub 版本交替的流水顺序依赖；主流落点 num_sub=1 退化（§1.6.2 #11） |
| **N 向 chunk** | **串行迭代** | `T.serial(num_full)` + Python 静态尾 chunk 分支 | **running stats 循环携带依赖**（flash 合并 chunk 序语义；行块并行度已饱和 56 核，chunk 维并行化零收益——§1.6.2 #12 / R1 #10）；chunk 间双缓冲流水 |
| 元素级（chunk 内 N 向） | 向量化 | `T.Parallel(SUB, BLOCK_N)` ×2（直载/masked 载入 + exp 原位）+ `T.Parallel(SUB)` ×3（max 合并/sum 合并/lse） | §1.6.2 #1–#8 全向量化 |
| 行内归约 | 硬件归约 | `T.reduce_max` / `T.reduce_sum`（dim=1, clear=True） | §1.6.2 #4/#7（AscendAllReduce 硬件树） |

### 6.2 循环伪代码

```python
# Block 级并行（一维 grid，56 persistent 核；核内完整结构见 §3.3）
with T.Kernel(56) as core_id:
    x_ub = T.alloc_shared((SUB, BLOCK_N), dtype)
    stats_ub = T.alloc_shared((SUB, 2), "float32")
    lse_ub = T.alloc_shared((SUB,), dtype)
    T.annotate_buffer_versions({x_ub: 2, stats_ub: 1})
    for t in T.Persistent([num_tiles], 56, core_id, num_stages=2):
        base = t * block_m
        if t < full:
            for s in T.serial(num_sub):
                <子批体：stats init → T.serial(num_full) chunk 流水
                 {T.copy GM→UB → SimtVF{直载 → reduce_max → max 合并 →
                  exp 原位 → reduce_sum → flash 合并}} → 尾 chunk → lse 收尾
                  → T.copy frag→UB → T.copy UB→GM>
        else:
            for s in T.serial(tsub):
                <子批体（rows=SUB）>
            if tsub_rem > 0:                     # Python 静态分支
                <子批体（rows=tsub_rem，行谓词 masked，静态 rem 切片 copy）>
```

### 6.3 流水线优化

`T.Persistent(num_stages=2)` + `T.annotate_buffer_versions({x_ub: 2})`：**chunk c+1 的 GM→UB 拷贝（MTE2，版本 v⊕1）与 chunk c 的 SimtVF 计算+合并（版本 v）重叠**——chunk 维（N/BLOCK_N 次迭代）是 tiled 的主循环维，流水主体在此；子批间由 persistent 调度器连续供给（主流落点 num_sub=1 时子批边界消失）。跨引擎依赖（MTE2→VALU→MTE3）由 AutoSchedule 自动插 flag 对（§7.2）；不手动管理双缓冲（tilelang-ascend skill §4.2：优先编译器 multibuffering）。**对照基线（源直译单版本 x_ub）**：MTE2 与 VALU 串行——流水收益 ≈ 流量项隐藏（wl2：−11.7K cycle/核 ≈ −8%，D-2 估算口径）。

### 6.4 尾块处理

- 行尾（M % block_m ≠ 0）：静态 full/rem_m 分解 + TIR if + 静态切片 + 行谓词（§5.4；§0.6 R4 语义保持论证）。
- 列尾（N % BLOCK_N ≠ 0）：尾 chunk 静态 rem 切片 + (SUB, TILE_R) masked（pad=−inf 三幂元中性，V1 位级验证）——无数值影响、无 OOB。
- 子批尾（block_m % SUB ≠ 0 或 rem_m % SUB ≠ 0）：全块路径末子批 last_sub_rows（静态）；尾块路径末子批 tsub_rem（静态）——均由 host 静态化推导切片与谓词（single 同款双分支，Stage 3 实证）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: auto_schedule: yes（AutoSchedule 自动同步；未设 `TL_ENABLE_AUTO_SCHEDULE: False`）

### 7.2 同步点说明

auto 模式下依赖边界说明（无手动同步点表；AutoSchedule 自动插入）：

| 依赖边界 | 涉及引擎/buffer | 自动机制 |
|----------|-----------------|----------|
| x_ub 写（MTE2，版本 v）→ SimtVF 读 | MTE2 → VALU，x_ub | AutoSchedule flag 对 |
| stats_ub 写（VALU，chunk c 合并）→ 读（chunk c+1 合并 / lse 收尾） | VALU 内，stats_ub | 同引擎顺序语义（`stats_ub: 1` 单版本——合并链串行） |
| lse_ub 写（VALU，`T.copy(out_frag, lse_ub)`）→ MTE3 读 | VALU → MTE3，lse_ub | AutoSchedule flag 对 |
| x_ub 版本交替 v / v⊕1 | multibuffering | `T.annotate_buffer_versions({x_ub: 2})` + 编译器 |
| SimtVF 跨线程 UB hazard（x_ub/stats_ub 共享读写） | VALU 线程域 | 自动 `asc_syncthreads`（skill §2.2；回归 test_ascend_thread_sync.py） |
| reduce 终值（m_c/s_c）→ 合并消费 | VALU 内 | 同引擎顺序语义 |

**风险提示**（§9.1 风险 3）：`Persistent + TIR if + T.serial（子批）+ T.serial（chunk）+ SimtVF` 五层嵌套无单一完整先例——但契约主流落点 block_m=1 → num_sub=1 使子批层退化，实际嵌套 = online_softmax streaming 四层同构（measured）；显式 bm>1 的五层嵌套由 single 已实证四层（含 TIR if + T.serial）+ online_softmax 已实证 T.serial(chunk) 双先例拼接覆盖。fallback：拆 chunk 外层 × 子批内层 + stats_ub 扩容 (block_m, 2)（语义等价、流水粒度变粗）。

### 7.3 pass_configs 配置

```python
# 默认自动调度，不传 TL_ENABLE_AUTO_SCHEDULE（仅完全手动核需要 False）；
# JIT 入口即 §4.7 的 @tilelang.jit(target="ascend", out_idx=[1]) 懒工厂。
```

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（源仓测试基准 `torch.logsumexp(x.float(), dim).to(x.dtype)`——`examples/TileOPs/tests/ops/test_softmax.py` L99 `ref_program` 直移植，与 single 变体共用算子族语义），**不复刻 §0.6 的 NPU 算法**（不复刻 BLOCK_N chunk 划分与 flash 合并链）——保证验证独立性。

```python
def golden_logsumexp(x: torch.Tensor) -> torch.Tensor:
    """Row-wise logsumexp, torch on NPU（源仓 ref_program 直移植；fp32 计算域，
    输出 cast 回输入 dtype——与 §0.1 dtype 语义一致）。"""
    assert x.device.type == "npu" and x.ndim == 2
    return torch.logsumexp(x.float(), dim=-1).to(x.dtype)
```

```python
# L0 精度对比语句（主 shape 门槛；randn 数据域——±inf 分歧角点不在 L0 域）：
y = _logsumexp_kernel_tiled(M, N, dtype, tile_n)(block_m)(x)
torch.npu.synchronize()
golden = golden_logsumexp(x)
atol = rtol = 1e-2 if dtype in (fp16, bf16) else 1e-3
torch.testing.assert_close(y, golden, rtol=rtol, atol=atol)
```

### 8.2 精度标准与分层测试计划（L0 门槛；L1/L2/Boundary 完整套件由 tilelang-op-develop 落地）

**容差分档**（AGENTS.md / `examples/TileOPs/docs/gpu_to_ascend_adaptation.md` §1.8）：fp16/bf16 → atol=rtol=1e-2；fp32 → 1e-3（fp32 中间累加 + chunk 合并重缩放，非 accumulation-heavy 主链，取 1e-3 档——V3 实测 ≤1 ulp）。

| 层级 | 用例设计（依据：manifest workloads + test_softmax.py 落点分析 §0.2） |
|------|----------------------------------------------------------------------|
| **L0（门槛）** | ① wl2 等价 (1024, 32768) bf16 block_m=1 tile_n=16384（bench 主对象，2 full chunks 整除）；② wl3 等价 (4, 102400) fp16 block_m=1 tile_n=25600（**尾 chunk rem=4096** 静态切片路径）；③ (4, 102400) bf16（wl3 另一 dtype）；④ (4, 32768) fp16 tile_n=16384（测试用例 L76 落点）；⑤ (33, 33000) fp16 tile_n=16640（**非 pow2 tile_n → BLOCK_N=16384 重划** + rem=232 尾 chunk masked）；⑥ (33, 32768) fp16 block_m=4 tile_n=16384（**M 尾块 rem_m=1** TIR if 路径 + 显式 bm>1 子批）；⑦ (4, 32768) fp32 tile_n=16384（fp32 档 UB 128KB 场景） |
| **L1（shape/config 扫描）** | M ∈ {1, 3, 33, 56, 57} × N ∈ {32513, 32768, 33000, 65540, 102400} 三 dtype（block_m=1）；tile_n 扫描 {16384, 25600, 16640, 13824, 8192, 4096, 256}（pow2 / 非 pow2 / 极小值——BLOCK_N clamp 与 chunk 数变化）；block_m ∈ {1, 2, 4, 8, 16}（wrapper 启发式全集 + SUB 分裂点：bm>1 与 SUB 封顶交互）；M=1（单行全流式）；N < BLOCK_N 显式小 N（num_full=0 全尾 chunk）；多维 dim=[0,2] flatten 落点复核（wl4 走 single——负向记录） |
| **L2（稳健性/压力）** | 大 M 多波（M=8192, N=32768 → num_tiles=8192，146 波持久调度）；超大 N（N=1024000，63 chunks 长合并链——fp32 累加重缩放精度压力，容差内验证）；极小 tile_n=256（显式，BLOCK_N=256 → 128 chunks）；契约边界邻域（N=32512 vs 32513——single/tiled 切换边界两侧记录） |
| **Boundary（角点）** | ① 有限域角点（对照 golden）：全零行 / 全等行 3.5 / 单主导 100 / ±65504 极值（fp16）/ subnormal ±6e-8 / 深度下溢 −745 / N=1 / M=1 / 确定性（3× 位级一致）；② 部分 −inf 正常行（golden 一致 ✓ V5——含 chunk 全 −inf 中性两种顺序）；③ NaN 行（golden 双方 NaN，equal_nan=True ✓ V5）；④ **源继承分歧角点**（对照**源算法 fp64 语义模拟**而非 golden，迁移保真验证）：全 −inf 行 → NaN（torch 为 −inf）、含 +inf 行 → NaN（torch 为 +inf）——V5 已证链与源位级一致；⑤ **tiled 特有合并角点**（对照源语义模拟 + V5 先证）：跨 chunk max-gap 131008（fp16 极值对分置 chunk0/chunk1——flash 形式无溢出）、主导值分别在 chunk0/chunk1（合并顺序两向）、chunk 边界跨界的等值组（合并律） |

**已知边界分歧处理**（§9.3 详述）：全 −inf / 含 +inf 行为源算法固有（NaN 链），与 `torch.logsumexp` 分歧；源仓测试域（randn 生成）不触发（`examples/TileOPs/tests/ops/test_softmax.py` 全部 `torch.randn` / `torch.zeros` 基底），本设计保持源语义（迁移保真 + 与 single 变体行为一致）并在 Boundary 用源语义模拟对照证明行为一致。

### 8.3 性能验证（Stage 4，设计期仅记录口径）

- 测量权威：`tilelang.profiler.do_bench(fn, backend="msprof")`；代表 shape = wl2 (1024, 32768) bf16 / wl3 (4, 102400) fp16（bench_softmax.py manifest 驱动）。
- 基线对照：`torch.logsumexp(x, dim=-1)`（native op，bench_softmax.py 的 baseline_fn 同款）。
- A/B 必做项：§1.6.4 fp16-e 备选裁决（wl2 fp16 等价 shape + wl3 fp16，阈值与回写路径见该节）。
- 归因观察项：special_func 8 ops/cycle 粒度（§9.1 风险 5——npusim/msprof 反解）；wl3 并行度受限（4/56 核，风险 2——对比 wl2 每核 cycle 数口径）。

---

## 9. 风险点与注意事项

### 9.1 已知约束（本算子相关）

| # | 风险 | 影响 | 缓解 / 验证 |
|---|------|------|------------|
| 1 | **VRF 容量假设**（`CONST-vrf-capacity-a5` assumption：256KB 起步，无文档化常数） | SUB·BLOCK_N ≤ 32768 封顶后 x_frag ≤128KB（≤32 reg/thread）远低于 76 reg 实测水位——理论安全，但假设本身未标定 | Stage 3 编译后首查 spill 计数（`get_kernel_source()` / 编译日志）；若 spill > 0 降 SUB 一档（host 静态化参数，kernel 结构不变） |
| 2 | **wl3 并行度受限**（M=4 → num_tiles=4 < 56，52 核空转） | wl3 端到端 ≈ 单核序列（D-2：~22.9K cycle）——workload 特性（M 小 N 大）而非算法结构缺陷 | Stage 4 记录实测（wl2 每核口径对照）；wrapper 层 block_m 无法再小（已 1）；不作为设计缺陷处理（logsumexp 行独立性下 M=4 无并行空间可挖——除非 N 向跨核 partial+merge，R1 #10 已弃选） |
| 3 | **五层嵌套组合**（Persistent + TIR if + 子批 T.serial + chunk T.serial + SimtVF）无单一完整先例 | AutoSchedule 对双层串行循环嵌套的 multibuffering 版本推断异常的可能性（未证实） | 双先例拼接覆盖：主流落点 block_m=1 → num_sub=1 子批层退化（= online_softmax streaming 四层 measured 同构）；显式 bm>1 = single 四层实证 + chunk 层 online_softmax 实证；fallback = chunk 外层 × 子批内层 + stats_ub 扩容 (block_m, 2)（§0.6 R3）；Stage 3 首个 L0 用例即触发各路径（⑥ bm=4 触发子批层） |
| 4 | **±inf 边界与 torch golden 分歧**（全 −inf 行 → NaN vs torch −inf；含 +inf 行 → NaN vs torch +inf） | 源算法固有语义（迁移保真项 + 与 single 变体一致）；若未来集成测试加入此类角点将对照 fail | §8.2 Boundary ④ 以源语义模拟对照（迁移保真）；§9.3 文档化；源仓测试域（randn）不触发——Stage 5 集成跑 test_softmax.py 无此角点 |
| 5 | **special_func 8 ops/cycle 的 op 粒度未区分**（element vs 向量指令级，`estimate_latency.h` L58–70 未注明） | D-2 估算的瓶颈归因（计算主导 ~11.7×）可能反转为 MTE 主导 | 两种解释下主选结构均为最优（§1.6.0 披露）；Stage 4 npusim/msprof 实测归因，回写 `CONST-valu-model` |
| 6 | **bf16 精度链** | bf16 输入 fp32 中间（主选）无风险；fp16-e 备选**不适用 bf16**（s_c fp16 树形尾数不足） | §1.6.4 分派条件 `dtype == "float16"` 已排除 bf16；主选链 bf16 走 fp32 全程 |
| 7 | **block_m/tile_n 契约外值**（显式 config 传 >16 / 非 2 幂 / 极小 tile_n=256 / 超大 tile_n） | SUB 封顶 + BLOCK_N clamp 双机制兜底（§5.2），但极端小 BLOCK_N 使 chunk 数激增（N=32768/256 = 128 chunks——合并标量 ×128 仍 ~0.5% 占比） | 性能项非性能正确性项；wrapper default_config 不产生此类值；host assert 记录契约（§4.7）；L2 极小 tile_n 用例覆盖正确性 |
| 8 | **wrapper 预算口径变化**（`device_smem_budget` 返回值若从 216KB 改为 248KB 物理） | tiled/single 契约边界移动（更大 N/block_m 组合变化） | kernel 侧 BLOCK_N/SUB 公式不依赖 wrapper 预算数值（独立 194.4KB 折扣预算）；契约边界以 wrapper 实际判定为准 |
| 9 | **N 非 32B 行对齐的 DMA 带宽档**（(33,33000) fp16 行 66000B） | 正确性 ✓（字节级行长，skill §6.1）；带宽档降级（`CONST-mte-bandwidth-model` 平台/未知档 50/48 B/cycle vs 峰值 102） | 流量项非主导（计算 ~11.7×，D-2）；Stage 4 msprof 归因确认；非对齐 workload 不在 bench 主对象（wl2/wl3 行内 chunk 列窗口 32B 对齐） |
| 10 | **fp32 累加重缩放长链**（超大 N 下 chunk 数多，s_run 经多次 exp 重缩放） | 每 chunk 1 次重缩放乘法的舍入累积（N=1024000 → 63 次）——fp32 域 ulp 级累积，预估 << 容差 | V1/V6 已覆盖 6-8 chunks 链（≤1 ulp）；L2 超大 N 用例实测验证；若超容差（未预期）回退论证 §1.6.1 O1 的重缩放链精度分析 |

### 9.2 常见错误（实现期检查清单）

- ❌ `T.Kernel(grid, threads=…)`（二维/threads 参数）——TypeError（ascend-constraints §1 条 1）。
- ❌ `tilelang.compile(prog, target="ascend")` 直接形态——S3-JIT 门禁拒；用 `@tilelang.jit(target="ascend", out_idx=[1])` 懒工厂。
- ❌ fragment 非 pow2 形状（如 (SUB, rem) 直接用 rem、或 (SUB, tile_n) 直接用非 pow2 tile_n）——alloc_fragment 需 pow2（`example_rmsnorm.py` L42）；尾 chunk 用 TILE_R = next_pow2(rem)，full chunk 用 BLOCK_N pow2。
- ❌ 尾 chunk / 尾块 `T.copy` 越界切片——A5 DMA 越界行为未文档化；必须静态 rem 切片（`x_ub[0:rows, 0:rem]` 目的切片与源区域形状匹配——single Stage 3 教训：目的切片写成 `x_ub[0:rows, 0:N]` 会编译期形状错）。
- ❌ `pad_value=`/`data_select=` 用于行 OOB——二者是**列向**行尾 padding（skill §6.1），行 OOB 用谓词 + 静态切片。
- ❌ running stats 放 fragment 跨 SimtVF 域——fragment 不能跨域存活（chunk 间有 MTE2）；必须 stats_ub UB 驻留（online_softmax streaming 先例）。
- ❌ 全局 PYTHONPATH 污染——所有 python 命令前置 `PYTHONPATH=/home/z00875089/tilelang-fork`。
- ❌ grid=32 惯例值——本机实查 56（`CONST-arch-a5` measured；persistent 形态对整除性免疫）。
- ❌ 逐元素标量循环（载入/exp/合并/lse）——§1.6.2 已全向量化，§6 不得回退。
- ❌ 哨兵钳位 m_c ≥ −65504（照抄 pilot online_softmax O6）——**改变源语义**（全 −inf 行 NaN → −inf），制造 single/tiled 语义分裂（§1.6.0 R1 #11 弃选依据）。
- ❌ NPU 侧 fp64 参照（`.double()`）——NPU 不支持 double，静默降级 fp32（single Stage 3 实证）；fp64 模拟一律 CPU。

### 9.3 特殊场景处理（边界语义文档）

**与 torch golden 的两处分歧（源算法固有，迁移保真保持——与 single 变体 §9.3 同款约定）**：

| 场景 | 本 kernel（=源 GPU 算法） | `torch.logsumexp` | 数学注记 | 测试域 |
|------|---------------------------|-------------------|---------|--------|
| 全 −inf 行 | **NaN**（每 chunk m_c=−inf → m_run 恒 −inf → 首个 chunk `exp(−inf−(−inf))=NaN` 污染 s_run） | −inf | 解析极限为 −inf；torch 特判，源算法按 IEEE 直传 NaN | randn 域不触发；Boundary ④ 以源语义模拟对照（V5 位级一致） |
| 行含 +inf | **NaN**（inf 位置 `exp(inf−inf)=NaN` 污染 s_c → s_run） | +inf | 解析主值为 +inf；同上 | 同上 |
| 行含 NaN | NaN（逐元素传播 → 合并链污染） | NaN | 一致 ✓ | equal_nan 对照 |
| 部分 −inf 正常行（含 chunk 全 −inf） | m_run 有限 → `exp(−inf−finite)=0` → s_c=0 中性 | 一致 | 一致 ✓（V5 两种 −inf 分布顺序实测） | 直接对照 |
| 跨 chunk max 间隙 131008（fp16 极值对分置） | flash 形式 exp 参数 ≤ 0——**无溢出**，位级正确 | 一致 | 朴素 m_c 基合并间隙 > 88.7 即 fp32 溢出（`TRAP-NAIVE-ONLINE-MERGE`）；源式安全 | V5 corner 实测 |
| fp16 深度下溢（−745） | s_run 下溢至最小可表示 → log(s_run) 有限/−inf 边缘 | fp32 域不受影响 | 输出 cast fp16 后差异在容差内 | Boundary ① |

**设计取舍说明**：选择保持源语义而非「对齐 torch」（如哨兵钳位 m ≥ −65504 使全 −inf 行输出 −inf）——理由：① 迁移保真优先（single 与 tiled 两变体行为一致性由源仓定义——源仓 tiled 无钳位）；② randn 测试域不触发；③ 引入钳位反而制造两变体间的语义分裂（single=NaN、tiled 钳位版=−inf）。flash 合并式在数值安全角点（max 间隙）上本已无溢出，钳位唯一改变的是全 −inf 行为——属语义变更而非稳定性需求。若 Stage 5 集成后需要与 torch 全角点对齐，属算子族级语义变更，应回归源仓统一处理（本设计记录不在单变体内擅自变更）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/logsumexp/_logsumexp_kernel_tiled/
├── DESIGN.md                  # 本设计文档
├── verify_equiv.py            # D-1 等价性机器验证（已执行，全 EQUIV_PASS）
├── _logsumexp_kernel_tiled.py # Stage 3：kernel 实现（@tilelang.jit 懒工厂 + golden + 分层测试）
├── REVIEW.md                  # Stage 2：设计检视报告（另生成）
└── history_version/           # 设计期分段草稿 / 历史版本备份
```

### 10.2 文件清单（Stage 3 实现输入）

| 文件 | 内容 | 契约要点 |
|------|------|---------|
| `_logsumexp_kernel_tiled.py` | kernel + golden + L0/L1/L2/Boundary 分层测试 | 调用面 `_logsumexp_kernel_tiled(M, N, dtype, tile_n)(block_m)(x)`；docstring 声明 `programming_mode: simt` + `auto_schedule: yes`；`@tilelang.jit(target="ascend", out_idx=[1])` 懒工厂；golden = `torch.logsumexp(x.float(), dim=-1).to(x.dtype)`（NPU 上计算）；host 静态化 BLOCK_N/SUB/num_full/rem/TILE_R/full/rem_m/num_tiles/num_sub/last_sub_rows/tsub/tsub_rem/threads + 契约 assert |

### 10.3 命名与声明规范

- 模块 docstring 必含：`programming_mode: simt`、`auto_schedule: yes`（迁移 prompt 契约）。
- 函数名/参数名与源保持（`_logsumexp_kernel_tiled`、`M/N/dtype/tile_n/block_m`）；`threads` 参数按 K3 移除（内部 `min(1024, SUB·BLOCK_N)` 自推导）。
- host 静态化变量命名：`BLOCK_N/SUB/num_full/rem/TILE_R/full/rem_m/num_tiles/num_sub/last_sub_rows/tsub/tsub_rem/threads/NUM_BLOCKS`（§3.3 注；与 single 的 TILE/SUB 体系对齐，BLOCK_N 区分 chunk 列宽）。

### 10.4 实现顺序建议（Stage 3）

1. 懒工厂骨架 + host 静态化（BLOCK_N/SUB 双 clamp + 契约 assert）→ 2. 主流落点路径（block_m=1：L0 ①②③④⑤——子批层退化，online_softmax streaming 同构）→ 3. 尾块路径（L0 ⑥ bm=4：TIR if + 子批 + 静态切片）→ 4. `get_kernel_source()` 首查 spill 计数（风险 1）→ 5. L1 扫描（tile_n/bm 全集 + BLOCK_N clamp 边界）→ 6. L2/Boundary（源语义对照项 + tiled 特有合并角点）→ 7. 四出口判定。

### 10.5 本设计引用的外部工件

| 工件 | 用途 |
|------|------|
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py` | 源算子（三问解读对象，L101–189） |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp.py` | wrapper dispatch / tile_n 启发式 / Kernel class（契约 shape 空间推导） |
| `examples/TileOPs/tileops/kernels/reduction/_primitives.py` | `MAX_SINGLE_TILE_COLS` / `compute_tile_n` / `device_smem_budget`（tiled 选中条件与 tile_n 产生方式） |
| `examples/TileOPs/tileops/ops/reduction/logsumexp.py` | op 层（dim 归一化 / flatten / golden 基准 ref_program） |
| `examples/TileOPs/tileops/manifest/reduction.yaml` | workloads / roofline 口径（4·M·N FLOPs、(M·N+M)·elem bytes） |
| `examples/TileOPs/tests/ops/test_softmax.py` / `examples/TileOPs/benchmarks/ops/bench_softmax.py` | 精度 / 性能用例（tiled 落点分析 §0.2） |
| `examples/pilot/online_softmax/online_softmax.py` | streaming 路径直接同构先例（chunk 流水 / flash 合并 / stats_ub / 尾 chunk 静态切片，measured） |
| `examples/logsumexp/_logsumexp_kernel_single/`（DESIGN.md / 实现.py / verify_equiv.py / RETROSPECTIVE.md） | 同族驻留变体全流程实证（per-chunk 计算体 / 尾块机制 / 教训承接） |
| `examples/ascend/example_{rmsnorm, per_token_cast_to_fp8, gemm}.py`、`examples/ascend/flash_attention/core.py` | API 佐证（§3.4 表） |
| pattern-library（`elementwise.md` D-REDUCE-SIMTVF-API / D-ELEM-SIMTVF-AUTOVEC、`traps-runtime.md` TRAP-GRID-DIVISIBILITY / TRAP-NAIVE-ONLINE-MERGE、`constants.md` CONST-* 七条目） | 实测模式与硬件常数（版本戳 2026-10-16） |
