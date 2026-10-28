# _logsumexp_kernel_tiled 算子设计文档（GPU TileOPs → Ascend A5 迁移，**SimdVF 独立变体**）

> **任务上下文**：迁移 TileOPs `LogSumExpFwdOp` 的 **SIMD 模式独立变体**的**分块（tiled）成员**（新算子 `LogSumExpFwdOpSimd` / `logsumexp_simd`，GPU 源仓 `/home/z00875089/TileOPs-fork`）。源算子三问解读对象 = `tileops/kernels/reduction/logsumexp.py` 中的 `_logsumexp_kernel_tiled`（L105–194，**online softmax 递推跨 N-tile 的流式变体**：N 超驻留上限时 chunk 串行 + running (max, sum) flash 合并）及 wrapper 侧 `_tile_n_for_block_m` / `compute_tile_n` / `MAX_SINGLE_TILE_COLS` 契约机制（L338–360 + `_primitives.py`）。单 tile 变体 `_logsumexp_kernel_single` 的 SIMD 版已设计并实现通过（`examples/logsumexp_simd/_logsumexp_kernel_single/`，86 用例全 PASS，2026-10-17）——本设计是其**同族流式姊妹**，但算法结构不同（online 递推 vs 两遍驻留），不复用其"整行驻留"前提。
> **编程模式契约**：**simd（SimdVF）——用户显式指定**，非默认 simt。本设计全部计算以显式 `T.simd.*` 2048-bit 向量内在表达（`tilelang/ascend/language/simd.py`），kernel 模块 docstring 须声明 `programming_mode: simd`、`auto_schedule: yes`。
> **职责边界（严禁改动项）**：现有 SIMT 版本 `examples/TileOPs/tileops/kernels/reduction/logsumexp/`（含 perf_opt/）与 `ascend::logsumexp_fwd` custom_op **不属于本任务产物范围**；SIMD 家族的单 tile 变体 `examples/logsumexp_simd/_logsumexp_kernel_single/` 同样严禁改动。本设计产物只落在 `examples/logsumexp_simd/_logsumexp_kernel_tiled/`。SIMT tiled 版（`logsumexp_kernel/_logsumexp_kernel_tiled.py` + DESIGN）仅作同族先例引用（先例 ≠ 源算子——三问解读对象始终是 GPU 源码）。
> **调用面契约**：A5 侧保持 `_logsumexp_kernel_tiled(M, N, dtype, tile_n)(block_m)(x) -> y` 调用面（与 SIMT 兄弟变体一致；GPU `threads` 参数按 K3 已移除，SimdVF 模式下连内部 threads 推导也不存在——无线程概念）。

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

**数学语义**（逐行 log-sum-exp，数值稳定形式——与 single 变体同一解析式）：

$$
y_i \;=\; \mathrm{logsumexp}_j(x_{ij}) \;=\; m_i + \log\Big(\sum_{j=0}^{N-1} \exp(x_{ij} - m_i)\Big), \qquad m_i = \max_j x_{ij}
$$

tiled 变体将整行归约改写为 **online 递推**（分块 running 统计量 + flash 形式重缩放合并，源码 L135–183）：

$$
\text{init: } m^{(0)}_i = -\infty,\; s^{(0)}_i = 0; \qquad
\text{chunk } c:\; \begin{cases} p = m^{\text{run}}_i \\ m^{\text{run}}_i \leftarrow \max(m^{\text{run}}_i,\; m^{c}_i) \\ e^c_{ij} = \exp(x^c_{ij} - m^{\text{run}}_i) \\ s^c_i = \sum_j e^c_{ij} \\ s^{\text{run}}_i \leftarrow s^{\text{run}}_i \cdot \exp(p - m^{\text{run}}_i) + s^c_i \end{cases}; \qquad
y_i = m^{\text{run}}_i + \log s^{\text{run}}_i
$$

与解析式数学恒等（log-sum-exp 递推的代数变形；合并式全部 exp 参数 ≤ 0〔p ≤ m_run、x^c ≤ m_c ≤ m_run〕，无溢出——flash 形式特征，见边界语义行⑤）。

**语义要素表**（迁移必须原样保持的部分）：

| 要素 | 结论 | 源码证据（`tileops/kernels/reduction/logsumexp.py`） |
|------|------|---------|
| 数学映射 | 行级 logsumexp：`y[i] = row_max[i] + log(row_sum[i])`，row_sum 经 N 向 chunk 的 running (max, sum) flash 合并得到 | L180–188 |
| I/O 契约 | 输入 `x: (M, N)` 行主连续、dtype ∈ {float16, bfloat16, float32}（op 层允许三者；manifest workload 为 fp16/bf16）；输出 `y: (M,)` 同 dtype | kernel 签名 L125–128；`LogSumExpKernel` docstring L259–263 |
| 规约语义 | 沿最后一维（N）整行归约；**累加顺序 = chunk 内 `T.reduce_sum` 树形 + chunk 间按 chunk 序的 flash 重缩放顺序合并**（与 single 变体的单遍全行树形不同——fp32 域下差 ulp 级）；**中间计算全程 fp32**（`tile_f32` fragment、fp32 的 row/tile 统计量），输出 cast 回输入 dtype | L131–138、L147–183 |
| dtype 语义 | 输入 dtype → fp32 中间（`T.cast(x[...], "float32")`）→ 输出 dtype（`out_local` 为输入 dtype fragment，`T.copy` 写回） | L152–167、L180–185 |
| 边界语义 | ① **NaN 逐元素传播**：任一 chunk 含 NaN → m_c 或 e 任一 NaN → s_run=NaN → y=NaN；② **全 −inf 行 → y=NaN**（每 chunk m_c=−inf，m_run 恒 −inf → 首个 chunk 即 `exp(−inf−(−inf))=NaN` 污染 s_run——源固有行为，与 single 变体一致、与 `torch.logsumexp` 的 −inf **分歧**，§9.3）；③ **含 +inf 行 → y=NaN**（inf 位置 `exp(inf−inf)=NaN` 污染；torch 返回 +inf，同上分歧）；④ **部分元素 −inf 的正常行：分布决定行为**——−inf 散布或全 −inf chunk 出现在**首个含有限值的 chunk 之后**时完全中性（m_run 有限 → `exp(−inf−finite)=0` → s_c=0）；**但「首个 chunk 全 −inf 且后续 chunk 含有限值」→ y=NaN**（该 chunk 处理时 m_run 仍 −inf → `exp(−inf−(−inf))=NaN` 污染 s_run，此后 `s_run·exp(−inf−m)+s_c = NaN·0+s_c = NaN` 不可恢复）——**tiled 特有角点：single 变体（全行 max 先行）对该输入给出有限值、torch 给出有限值**，源 tiled 给 NaN，属源 online 递推的固有语义（本设计迁移保真保持，§9.3 文档化 + verify_equiv V5 位级验证）；⑤ **跨 chunk max 间隙无溢出**：合并式全部 exp 参数 ≤ 0，fp16 域极端间隙 131008（=2×65504）下朴素 m_c 基合并会溢出（`TRAP-NAIVE-ONLINE-MERGE`：间隙 > 88.7 即 fp32 溢出），源式 flash 形式安全——V4/V5 corner `max-gap 131008` 实测位级正确；⑥ M 尾块（M % block_m ≠ 0）行 OOB：masked 分支以 −inf 载入、不写出（行谓词 L155–157）；快路径依赖 GPU `T.copy` OOB 保护（未文档化，§0.5 重设计） | L150–162、L169–183；V4/V5 实测 |
| 尾块/非整除 | **N 维**：`N_padded = align_up(N, 256)`、`num_tiles = ceil(N_padded/tile_n)`、`total_cols = num_tiles·tile_n`、`_needs_mask = total_cols > N`——仅最后一个 chunk 可能越界（前面 chunk 列范围 ≤ N_padded − tile_n < N 恒有效）；`_needs_mask` 时最后 chunk 走逐元素 masked 载入（列 j 有效、行 <M，越界填 −inf），其余 chunk 走 `T.copy` 快路径；`_needs_mask=False` 时全部走快路径。**M 维**：M % block_m ≠ 0 的尾块行为同上（masked 分支有行谓词；快路径无行掩码） | L116–120、L143–167 |
| host 侧语义 | `N_padded`/`num_tiles`/`total_cols`/`_needs_mask` 常量折叠在 builder 闭包内（JIT per-shape 编译的一部分）；`tile_n` 由 wrapper 启发式传入（`compute_tile_n`：256 倍数、预算封顶、整除偏好——契约参数，kernel 侧不可改）；wrapper 层负责 dim 归一化/transpose/contiguous/reshape 到 (M, N) 与输出 reshape——**不属于本 kernel 的契约** | L116–121；`logsumexp.py` L338–360、L540–557 |

**语义解读完成标准自检**：依据上表可直接写出与源算子数值一致的 golden（§8.1 的 `torch.logsumexp(x.float(), dim=-1).to(x.dtype)`，源仓测试基准 `ref_program` 直移植）；±inf/NaN 角点的源继承分歧（含 tiled 特有的「首 chunk 全 −inf → NaN」）已显式列出（§9.3 + V4/V5）。

### 0.2 源算子输入输出（含 tiled 变体的契约 shape 空间）

**Kernel 签名**：`_logsumexp_kernel_tiled(M: int, N: int, dtype: str, tile_n: int)` → `_func(block_m)` → `(x) → y`（`@tilelang.jit(out_idx=[1])`；GPU 版 `_func(block_m, threads)`，A5 按 K3 移除 `threads`）。

**tiled 变体被 wrapper 选中的条件**（`logsumexp.py` `_logsumexp_kernel` dispatch L200–205 `tile_n != 0` + `_tile_n_for_block_m` L338–360，决定本 kernel 的契约 shape 空间）：

1. **大 N 路径**：`N_padded = align_up(N, 256) > MAX_SINGLE_TILE_COLS = 32512`（列数上限，`_primitives.py` L39——GPU 侧为 TileLang 向量化器 32768 边界的安全回退值）——single 变体无条件的硬边界；或
2. **驻留超限路径**：`N_padded ≤ 32512` 但 `block_m × elem_bytes × N_padded > device_smem_budget()`（GPU 侧 SMEM opt-in 预算）——中 N × 大 block_m 组合；或
3. **显式 config 路径**：caller 显式传 `config["tile_n"]`（`logsumexp.py` L319–336 直接采纳）——任意 (M, N) 组合均可被显式路由到 tiled。

**tile_n 的产生方式**（`compute_tile_n`，`_primitives.py` L109–194）：256 倍数；预算封顶 `tile_n ≤ effective_budget // (block_m · elem)`（effective = min(smem_budget, 32512·block_m·elem)）；**优先返回整除 N_padded 的最大候选**（消尾 chunk 浪费），否则返回预算上限值（尾 chunk masked 由 kernel 兜底）。因此 wrapper 产生的 tile_n 是 256 的倍数（**256 倍数自动是 64 倍数**——本设计 §0.6 R2 的组宽整除前提），典型值 16384/25600/16640——**不保证 2 的幂**。

**default_config 行为**（`logsumexp.py` L362–395）：block_m ∈ {1,2,4,8,16} 遍历，tiled 路径下**优先最小化 N 向 chunk 数**（最大化 tile_n）——由于 `effective_budget ∝ block_m`，小 block_m 得大 tile_n，故 tiled 落点偏好 **block_m=1**（与 single 相反）。

**manifest workload 落点**（`examples/TileOPs/tileops/manifest/reduction.yaml` L146–152 `LogSumExpFwdOpSimd`，归约到 (M, N) 后；tile_n/block_m 按 wrapper 启发式逐 block_m 复算，与 SIMT 家族共用同一契约口径）：

| workload | 归约后 (M, N) | dtype | wrapper 启发式 (block_m, tile_n) | 落入变体 |
|----------|--------------|-------|-------------------------------|---------|
| attn-weights-4k `[32,32,4096]` dim=-1 | (1024, 4096) | fp16 / bf16 | (16, 0=single) | single（已完成，另任务） |
| attn-weights-32k `[32,32,32768]` | (1024, 32768) | bf16 | **(1, 16384)** | **tiled（本设计，wl2）** |
| lm-head-logits `[4,102400]` | (4, 102400) | fp16 / bf16 | **(1, 25600)** | **tiled（本设计，wl3）** |
| 3d-multidim-reduce `[4,128,4096]` dim=[0,2] | (128, 16384) | fp16 | (4, 0=single) | single（已完成） |

> wl2 tile_n=16384 推导：bm=1 时 effective=min(216KB 级 smem opt-in, 32512×1×2B)=65,024B → max_cols=32,512 → tile_n_max=32,512；整除 32,768 的最大 ≤32,512 的 256 倍数 = 16,384（2 chunks）。wl3 tile_n=25600 推导：tile_n_max=32,512，整除 102,400 的最大 256 倍数因子 = 25,600（102400=400×256，400 的 ≤127 最大因子为 100 → 4 chunks 整除）。

**精度用例落点**（`examples/TileOPs/tests/ops/test_logsumexp_simd.py`，SIMD 家族专属测试）：`(4, 32768)` fp16/bf16（L75–76 "large-N triggers N-tiling path"）→ (1, 16384)；`(33, 33000)` fp16（L80 "multi-tile, masked loads"）→ (1, 16640)（N_padded=33,280=130×256，130 的 ≤127 最大 256 倍数因子为 65 → 16,640，2 chunks）；`(33, 32768)` fp16（L82 非对齐 M + 大 N）→ (1, 16384)。性能用例 `examples/TileOPs/benchmarks/ops/bench_logsumexp_simd.py`（manifest 驱动，wl2/wl3 为 tiled 覆盖对象）。

**契约空间总结**：M ≥ 1 任意（含 M < block_m：full=0 全尾批路径；M % block_m ≠ 0 尾批）；**N ≥ 1 任意（无 single 的 32512 上限——任意大 N 是本变体的存在理由**；N < BLOCK_N 时 num_full=0 全尾 chunk 路径；显式 config 也可路由小 N）；dtype ∈ {fp16, bf16, fp32}；block_m ∈ [1, 16] 常见（wrapper 启发式 tiled 落点偏好 1；显式 config 理论上任意 ≥1，本设计行串行结构天然兜底任意 block_m，§0.6 R1）；tile_n ≥ 1 任意（wrapper 产 256 倍数 ≤ 32,512；显式 config 任意——BLOCK_N clamp 机制兜底，§5.2）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解表**（源 GPU kernel 每条计算语句归入一步，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式 | 源码行 |
|------|------|------|------|-------------|--------|
| S0 | host 静态化：`N_padded = align_up(N, 256)`；`num_tiles = ceil(N_padded/tile_n)`；`total_cols = num_tiles·tile_n`；`_needs_mask = total_cols > N` | N, tile_n | 编译期常量 | chunk 划分与边界处理选择 | L116–120 |
| S-init | 循环前初始化：`T.fill(row_max, −inf)`、`T.fill(row_sum, 0.0)`（每行块每 kernel 调用一次） | — | row_max, row_sum | m^(0)=−inf, s^(0)=0 | L140–141 |
| S1a | 快路径载入（t < num_tiles−1 或无 mask）：`T.copy(x[pid_m·block_m, t·tile_n] → shared_buf → tile_local)` 再 cast fp32 | x (GM) | tile_f32 (block_m, tile_n) fp32 | chunk 数据就位（全列有效） | L143–148、L158–167 |
| S1b | masked 载入（仅最后 chunk，`_needs_mask` 时）：`tile_f32[i,j] = if(行<M ∧ t·tile_n+j<N, cast(x), −inf)` | x (GM) | tile_f32 | chunk 数据就位 + 列/行边界中性化 | L149–157 |
| S2 | chunk 内行最大：`T.fill(tile_max, −inf)` + `T.reduce_max(tile_f32, tile_max, dim=1, clear=False)` | tile_f32 | tile_max (block_m,) fp32 | m^c_i | L169–170 |
| S3 | running max 合并：`prev_max = row_max; row_max = max(row_max, tile_max)` | row_max, tile_max | prev_max, row_max | m_run ← max(m_run, m_c) | L172–174 |
| S4 | 稳定化指数（**以合并后 max 为基**，原位覆写）：`tile_f32[i,j] = exp(tile_f32[i,j] − row_max[i])` | tile_f32, row_max | tile_f32 ≡ e^c | e^c_ij = exp(x^c − m_run) | L176–177 |
| S5 | chunk 内行和：`T.reduce_sum(tile_f32, tile_sum, dim=1)` | tile_f32(e) | tile_sum (block_m,) fp32 | s^c_i | L178 |
| S6 | running sum 重缩放合并：`row_sum = row_sum·exp(prev_max − row_max) + tile_sum` | row_sum, prev_max, row_max, tile_sum | row_sum | s_run ← s_run·exp(p−m_run) + s_c | L180–183 |
| S7 | 循环出口 lse：`out_local[i] = row_max[i] + log(row_sum[i])`（fp32 算式，fragment 为输入 dtype） | row_max, row_sum | out_local (block_m,) | y_i = m_run + log(s_run) | L185–188 |
| S8 | 写回：`T.copy(out_local, y[pid_m·block_m])` | out_local | y (GM) (M,) | 输出 | L190 |

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

**循环与并行结构**（源）：grid = `T.ceildiv(M, block_m)` 个 block（`T.Kernel(grid, threads=threads)`，**二维参数 + threads**——A5 不支持，§0.5）；每 block 处理 block_m 行的**全部 N 向 chunk**（`for t in T.Serial(num_tiles)` chunk 串行——行块间独立、chunk 间顺序依赖 running stats）；chunk 内 `T.Parallel(block_m, tile_n)` 线程级并行（threads=256 默认）；running 统计量（row_max/row_sum/prev_max）为 register fragment，跨 chunk 循环迭代存活；无块间通信。

**host 侧逻辑**：`align_up`/`num_tiles`/`_needs_mask` 常量折叠（编译期）；`_neg_inf` 字面量；`threads` 由 wrapper 传入（GPU 侧契约）；JIT per-(M, N, dtype, tile_n) 编译 + `lru_cache`；wrapper 侧 `tile_n` 在 Kernel 构造时烘焙（`self._tile_n`），`forward` 经 `_logsumexp_fwd_wrapped(...)(block_m, threads, tile_n)(x)` 调用。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 代码特征 | 目的 | 依赖的源硬件特性 | 层级 |
|---|----------|---------|------|-----------------|------|
| 1 | **online softmax 递推（flash 形式合并）** | running (row_max, row_sum) + `s_run·exp(prev−new) + s_c` 重缩放（S3/S6） | X 单遍 GM 读完成整行归约（vs 两遍法重读 X）；中间缓冲 O(tile)（vs O(N) 落 GM） | **无（纯算法层）** | 算法层 |
| 2 | N 向 chunk 分块流式 | `T.Serial(num_tiles)` chunk 串行 + per-chunk (block_m, tile_n) staging | 突破片上驻留上限处理任意大 N | SMEM 容量数值（块大小硬件相关，分块策略本身可移植） | 算法层（驻留策略） |
| 3 | chunk 级快慢双路径 | `_needs_mask` 时仅最后 chunk masked、其余 `T.copy` 快路径 | 全有效 chunk 免逐元素掩码开销 | GPU `T.copy` shared 256 元素对齐约束 + masked load 代价 | 优化层 |
| 4 | per-chunk SMEM staging | `shared_buf (block_m, tile_n)` 每 chunk 重填充 | GM→寄存器两级缓冲（coalesced bulk copy） | GPU SMEM + 异步拷贝 | 优化层 |
| 5 | 减 max 稳定化 + fp32 中间精度 | `exp(x − m)`、tile_f32/统计量 fp32 | 消 fp16 溢出；累加/重缩放精度 | 无（纯数学） | 算法层 |
| 6 | 树形归约 `T.reduce_max/reduce_sum` | dim=1 fragment 归约（tile_max/tile_sum） | chunk 内行统计量低延迟 | warp shuffle 树（TVM lower） | 优化层 |
| 7 | 合并式标量最小化 | S3/S6 的 per-row 标量合并（每行每 chunk 1 max + 1 exp + 1 mul + 1 add） | 重缩放开销 O(rows·chunks) 而非 O(N) | GPU 标量管线 | 优化层 |
| 8 | 行批 block_m + 非持久 grid | `block_m` 参数化；`ceildiv(M, block_m)` blocks | 摊薄 block 启动开销；行块直接映射 | GPU SM 调度器 | 优化层 |

**未识别更优手段的风险自检**：源码无 warp shuffle 手写、无 swizzle、无 cp.async/Pipelined、无 persistent、无哨兵钳位（全 −inf 行靠 NaN 传播语义而非钳位防护——与 pilot online_softmax 的 O6 哨兵不同，§1.6.0 R1 候选 11 显式评估）；识别已覆盖全部显式结构。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（三层模型：语义层/算法层/优化层；四态处置：保留/等价替换/重新设计/舍弃。判定输入含 §1.6.0 调研结论——源算法与调研候选同台比较；本设计的承载模式为 **SimdVF**（用户显式指定），故「源线程级并行如何落到显式 2048-bit 指令流」与「running 统计量如何跨 SimdVF 域驻留」是本变体特有的判定维度。）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（行 logsumexp、online 合并顺序、fp32 中间、NaN/±inf 边界行为含「首 chunk 全 −inf → NaN」） | 语义 | — | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1）；合并顺序保持 chunk 序 flash 形式（§0.1 规约语义；V1/V4 位级、V5 角点位级验证） |
| online softmax 递推（#1） | 算法 | 无 | — | **保留** | 纯算法层；flash 形式已规避 `TRAP-NAIVE-ONLINE-MERGE` 的溢出发散角点（max 间隙；全 −inf chunk 的 NaN 在本算子是源固有语义而非缺陷）；调研确认 online 是超驻留大 N 语境的唯一单遍算法族（§1.6.0 R2/R3） |
| N 向 chunk 分块（#2） | 算法 | SMEM 容量数值 | 有（SimdVF UB 248KB〔无 SimtVF 32KB 线程上下文扣减，`CONST-capacity-a5`〕重算 chunk 粒度） | **保留（粒度重算）** | chunk 粒度由 A5 UB 预算重新推导（BLOCK_N = tile_n 直保留 + clamp，§0.6 R2）；任意大 N 的流式能力是本变体语义 |
| 减 max 稳定化 + fp32 中间（#5） | 算法 | 无 | — | **保留** | 纯数学（`ALG-softmax` R1 必选项）；fp32 合并链精度（V3 vs golden ≤1 ulp） |
| 树形归约 `T.reduce_max/reduce_sum`（#6） | 优化 | warp shuffle 树 | 有（SimdVF 跨 lane 全归约内在 `vcmax`/`vcadd`，`src/tl_templates/ascend/simd_inst.h` L688–710 `asc_reduce_max/asc_reduce_sum`） | **等价替换** | `examples/ascend/flash_attention/core.py` L115–118/L134–137（vcmax/vcadd + ONEPT 存储实测在用）；**SIMD 单 tile 姊妹 Stage 3 实证**（`examples/logsumexp_simd/_logsumexp_kernel_single/` 86 用例全 PASS，2026-10-17）；fp32 累加分组变化 → verify_equiv V2 实测 ≤1 ulp |
| 合并式标量最小化（#7） | 优化 | GPU 标量管线 | 有（源式 per-row 向量寄存器合并——7 向量指令/chunk/行，§1.6.1 O4） | **等价替换** | 源 `for i in T.Parallel(block_m)` 合并体 → SimdVF 域内 alloc_var 寄存器合并（broadcast 形态 vmax / lane-0 形态 vadd·vmul）；占比 wl2 0.39%（D-2），结构同源 |
| per-chunk SMEM staging（#4） | 优化 | GPU SMEM 两跳 | 有（`T.copy` GM→UB 单跳 + `vld(UNPK_B16)`+`vcvt` 向量装载） | **等价替换** | `testing/ascend/language/test_tilelang_ascend_simdvf_cast.py` L64–80（UNPK_B16 + vcvt(part=0) 位级回归）；SIMT tiled `T.copy` chunk 拷贝同构（`_logsumexp_kernel_tiled.py` L156–159） |
| running 统计量跨 chunk 驻留（register fragment row_max/row_sum/prev_max 生命周期覆盖 chunk 循环） | 优化 | GPU fragment 跨循环存活 | **部分**（SimdVF 域 per-chunk 开闭——chunk 间有 MTE2 `T.copy`；SSA 值不能跨 SimdVF 域〔single 变体 Stage 3 教训：`Immutable variable used outside its defining region`〕；**`T.simd.alloc_var`（local.var 可变寄存器）可承载循环携带**——`testing/ascend/language/simdvf/test_simdvf_arithmetic.py` L86–110 `serial_vreg_accum` 实证 SimdVF 域内跨 `T.serial` 循环存活） | **重新设计** | → §0.6 R5：**行串行结构**（行循环外置）使 running stats 退化为**每行一对 alloc_var**（m_run/s_run，行迭代作用域声明、per-chunk SimdVF 块内读写）——避免跨行数组索引；无单一先例覆盖「alloc_var 在 SimdVF 块外声明、块内访问」（风险 1 + fallback 预案：flash core.py L103/L117/L120–126 的 stats UB 往返形态 `vsts ONEPT_B32` + `vld BRC_B32` + `mem_bar VST_VLD`） |
| chunk 级快慢双路径（#3） | 优化 | GPU `T.copy`→SMEM 256 元素对齐 | **无此约束**（A5 GM→UB DMA 字节级行长；SimdVF 组装载只需 64 元素组边界） | **重新设计** | → §0.6 R2：统一为「full chunk 无谓词直载 + 尾 chunk 静态 rem 切片 + 组级 fp32 域 vsel(−inf) 掩码」二分支静态化（single 变体 R2 同款决策；`P-LAYOUT-TAIL-STATIC-EXTENT` measured） |
| `tile_n` chunk 粒度（256 倍数任意值） | 优化/契约 | GPU SMEM 对齐（256 元素） | **无 pow2 约束**（SimdVF 无 fragment——SIMT 兄弟变体被迫 pow2 重划〔16640→16384，fragment 翻倍 50.8% 有效 lane〕的约束在本模式不存在；256 倍数自动是 64 倍数 → 组循环零尾组） | **保留（粒度重解释）** | → §0.6 R2：**BLOCK_N = tile_n 直保留**（≤ UB clamp 上限）——chunk 划分与源一致（预算内），非 pow2 tile_n（16640/25600）下 chunk 数少于 SIMT 兄弟的 pow2 重划（(33,33000): 2 vs 3；wl3: 4 vs 7）；flash 合并对划分满足结合律（V1/V6 位级/≤1 ulp）；签名保留 tile_n（接口不变），内部粒度按 SimdVF 约束重解释（与 K3 移除 threads 同类） |
| `T.Kernel(grid, threads=threads)` block×thread 两级并行 | 优化 | CUDA 线程模型 | **无直接等价**（`T.Kernel` 一维、无 `threads=` 参数——TypeError；SimdVF 无线程域，连内部推导也不存在） | **重新设计** | → §0.6 R1：一维 `T.Kernel(56)` + persistent 行块调度 + SimdVF 显式指令流（ascend-constraints §1 条 1；tilelang-ascend skill §2.1/§2.3）；K3：调用面无 threads |
| 非持久 grid（ceildiv(M, block_m) blocks，#8） | 优化 | GPU 硬件调度器 | 有更契合形态（persistent + 多波摊薄） | **重新设计** | → §0.6 R3：`T.Persistent([num_tiles], 56, core_id, num_stages=2)` 行块持久调度（single SIMD L137 / SIMT tiled L231 同构先例）；#8 行批摊薄意图由 persistent 域 + block_m 行/任务承接 |
| M 尾块越界行为（快路径 `T.copy` 无行掩码，S1a/S8） | 实现 | GPU `T.copy` 隐式 OOB 保护（源仓 CI 通过的既成事实） | **未文档化**（A5 DMA 越界行为无保障） | **重新设计** | → §0.6 R4：静态 full/rem_m 分解 + TIR if 尾批分支 + 静态切片 copy（不依赖 A5 `T.copy` 越界行为；`example_gemm.py` L47 先例 + single SIMD Stage 3 实证） |
| host `threads` 契约 | 环境 | CUDA | 无（K3：A5 移除） | **舍弃（形态变更）** | GPU threads 表达的行内并行度由 SimdVF 2048-bit 向量内在承接（每指令 64 fp32 / 128 fp16 元素）；调用面 `(block_m)` 单参数与 SIMT 兄弟变体一致 |

**判定统计**（表内 13 条目）：保留 4 项（计算语义 / online 递推 / chunk 分块策略〔粒度重算〕/ 稳定化）+ 等价替换 3 项（树形归约→vcmax/vcadd 跨 lane 全归约、合并式标量→寄存器向量合并、SMEM staging→UB 单跳 + UNPK/vcvt 装载）+ 重新设计 5 项（R1 并行结构、R2 chunk 粒度与双路径统一、R3 调度、R4 M 尾块、R5 running stats 驻留）+ 舍弃 1 项（host threads 契约——K3 形态变更，意图由 SimdVF 指令流承接）。**无静默丢弃**：每个源优化意图均有承接结构或舍弃理由。

### 0.6 NPU 算法重设计

**重设计项 R1: 并行结构（block×thread 两级并行 + 行块整批 → 一维 grid + persistent 行块 + 行串行 SimdVF 指令流）**

- **源方案**：`T.Kernel(ceildiv(M, block_m), threads=threads)`——block_m 行/block × threads 线程（chunk 内 `T.Parallel(block_m, tile_n)` 行×列线程化），非持久 grid；行块内 block_m 行经线程域并行处理同一 chunk。
- **NPU 新算法**：一维 `T.Kernel(NUM_BLOCKS=56)`（= `vector_core_num` 实查值，§5.5）；`for t in T.Persistent([num_tiles], 56, core_id, num_stages=2)` 行块持久调度（num_tiles = ceil(M/block_m)，多波由调度器分配）；任务内**行串行**（`for r in range(rows_this)`，rows_this ∈ {block_m, rem_m} 静态），每行独立完成其全部 N 向 chunk 的 online 流式；元素并行在每行每 chunk 的 `with T.SimdVF():` 块内以显式 2048-bit 指令流表达（§0.6 R6）。**与 SIMT 兄弟变体的关键结构差异**：兄弟变体为「chunk 外层 × 子批 SUB 内层 × SimtVF 行并行」三层（stats 驻留 UB）；本设计**扁平化为「行外层 × chunk 内层」两层**——行串行使 running stats 退化为每行一对寄存器（R5），代价是 chunk DMA 从 (SUB, BLOCK_N) 2D 变 (1, BLOCK_N) 1D（bm=1 主流落点两者逐字节相同；bm>1 时 DMA 指令数 ×bm、字节数不变——wl2 bm=4 估算 ≤1% 端到端，§9.1 风险 7）。
- **语义保持论证**：行间完全独立（chunk 串行只发生在行内部的 N 维）→ 行处理顺序（并行线程域 → 串行行循环）与行到核的映射变化不影响数值；行内 chunk 序 flash 合并逐式保持（§0.1 递推式）；fp32 累加分组变化（warp 树 → lane 累加 + 单次跨 lane 归约）引入树深级舍入差（verify_equiv V1 ≤1 ulp / V2 ≤1 ulp，违反率 0）；NaN/±inf 逐元素传播与 chunk 合并语义不变（V4 位级 + NaN 模式一致；V5 首 chunk 全 −inf 角点与源位级一致）。
- **依据**：`examples/logsumexp_simd/_logsumexp_kernel_single/_logsumexp_kernel_single.py` L213–219（T.Kernel(56) + Persistent + SimdVF 同族实证）；SIMT 兄弟 `_logsumexp_kernel_tiled.py` L226–250（tiled 契约的 Persistent + TIR if 结构实证）；`CONST-arch-a5`（measured，本机实查 28/56，2026-10-17 本设计执行期复测确认）。

**重设计项 R2: chunk 粒度与驻留布局（tile_n 256 倍数 chunk + N_padded 对齐双路径 → BLOCK_N = tile_n 直保留 + UB 紧凑 + 尾 chunk 组级掩码）⭐（本变体核心增量）**

- **源方案**：chunk 宽 = tile_n（wrapper 启发式的任意 256 倍数，如 16640/25600）；列尾以 N_padded = align_up(N, 256) 对齐（GPU SMEM 约束），仅最后 chunk masked（`_needs_mask` 双路径）。
- **NPU 新算法**：① host 静态化：`BLOCK_N = 满足以下约束的最大 64 倍数：BLOCK_N ≤ tile_n 且 2·BLOCK_N·elem ≤ (228,556B − 8,256B 预留)`（fp16/bf16 clamp 上界 55,040、fp32 上界 27,520——wrapper 产 tile_n ≤ 32,512，fp16/bf16 下 clamp 恒不触发、BLOCK_N = tile_n；fp32 或显式超大 tile_n 由 clamp 兜底；§4.5 单一来源推导）；`num_full = N // BLOCK_N`、`rem = N % BLOCK_N`。② full chunk 全列有效：UB 紧凑 `x_ub (1, BLOCK_N)` + 组循环**无谓词直载**（`vld(UNPK_B16)`+`vcvt`，BLOCK_N 为 64 倍数 → 零尾组、零掩码开销）；③ 尾 chunk（rem > 0，Python 静态分支）：静态 rem 切片 `T.copy(x[row:+1, col:N], x_ub[0:1, 0:rem])` + 组循环 `G_tail = rem // 64` 满组 + 尾组 `k_tail = rem % 64` 以 `vsel(c, vdup(−inf,f32), update_mask(k_tail,32))` fp32 域掩码（−inf 同时是 max 幂元与 exp 后 sum 幂元）；④ **与 SIMT 兄弟的粒度差异**：兄弟变体因 fragment pow2 约束被迫重划 BLOCK_N = pow2 ≤ tile_n（16640→16384、25600→16384，非 pow2 tile_n 下有效 lane 50.8%/78.1% + chunk 数增加）；SimdVF 无 fragment——**BLOCK_N = tile_n 直保留**，chunk 划分与源一致（预算内），wl3 4 chunks（vs 兄弟 7）、(33,33000) 2 chunks（vs 兄弟 3）——merge 指令与 DMA 迭代同比例减少。
- **语义保持论证**：flash 合并（max 基重缩放）对 chunk 划分满足**结合律**——每个元素恰计入一个 chunk、合并代数恒等，任意划分数学等价；浮点下仅重缩放乘法的分组舍入差。**机器验证 V1**：8 组 (M, N, tile_n)（含 tile_n=25600 非 pow2 4-chunk、16640 masked-tail、tiny、63-chunk 长链）候选链 vs 源链 **fp64 位级一致（ulp=0）、fp32 ≤1 ulp、违反率 0**；**V6** 划分敏感性扫描（BLOCK_N ∈ {4096, 8192, 16384, 16640, 25600, 32768} × 3 shape）全部 ≤1 ulp——**非 pow2 BLOCK_N（16640/25600）首次纳入扫描**（兄弟变体 V6 仅 pow2）；**V4** 源链 vs 本链 fp64→fp16/bf16 输出位级一致 + NaN 模式一致。−inf 尾组掩码幂元中性同 single（V1 隐含覆盖）。
- **收益**：非 pow2 tile_n 下 chunk 数恢复到源划分水平（wl3 7→4、(33,33000) 3→2——merge 标量/DMA 迭代/vdup 常量重绑定同比例↓）；GM 读消除 N_padded pad 列搬运（33000→33280：−0.85%）；消除 `_needs_mask` 双路径（统一 full/tail 二分支静态化）。
- **依据**：`examples/pilot/online_softmax/online_softmax.py` `_build_streaming`（full chunk 直载 + 尾 chunk 静态 rem 切片——chunk 流式完整同构先例，measured 63/63 PASS）；`D-REDUCE-SIMTVF-API`（掩码载入与 DMA 紧凑拷贝正交、−inf pad 幂元位级中性）；`TRAP-NAIVE-ONLINE-MERGE`（flash 形式角点安全性）；`CONST-capacity-a5`；§1.6.0 R3/R4。

**重设计项 R3: 调度（非持久 grid → persistent + 行串行 + chunk 双缓冲流水 + AutoSchedule）**

- **源方案**：每 block 一次性处理 block_m 行 × 全部 chunk 后退出（无块间流水概念）。
- **NPU 新算法**：外层 `T.Persistent([num_tiles], 56, core_id, num_stages=2)`；任务内两级串行：行循环 `for r in range(rows_this)`（外，TIR serial）× N 向 chunk 循环 `for c in T.serial(num_full)`（内）+ Python 静态尾 chunk 分支；`x_ub (1, BLOCK_N)` 双版本（`T.annotate_buffer_versions({x_ub: 2})`）——chunk c+1 的 MTE2 预取与 chunk c 的 SimdVF 计算（版本 v）重叠；**行边界共享同一双缓冲链**（行 r+1 的 chunk 0 预取可重叠行 r 的末 chunk 计算/epilogue——比兄弟变体的 (SUB, BLOCK_N) 整批 DMA 更细的流水粒度）；MTE2→VALU→MTE3 跨引擎依赖由 AutoSchedule 自动插 flag 对（§7）。
- **语义保持论证**：纯调度变换（buffer 版本交替 + 同步边），不改任何数值路径；chunk 串行顺序保持（running stats 的循环携带依赖）。
- **依据**：single SIMD `_logsumexp_kernel_single.py` L213–231（Persistent + SimdVF + annotate_buffer_versions 三件套同族实证）；SIMT tiled `_logsumexp_kernel_tiled.py` L231（Persistent + chunk T.serial + 双版本）；tilelang-ascend skill §4.2/§7。
- **嵌套风险与退化形态**：`Persistent > TIR if（尾批）> 行 serial > chunk T.serial > SimdVF` 五层嵌套——**主流落点 block_m=1 → 行循环单迭代退化**（实际嵌套 = single SIMD 已实证的四层 + chunk 层 online_softmax 实证，双先例拼接）；fallback 预案：若 AutoSchedule 对深层嵌套的 multibuffering 推断异常，改用「chunk 外层 × 行内层 + stats UB 往返」（= 兄弟变体结构 + flash 访问形态，语义等价、x_ub 流水粒度变粗）——§9.1 风险 3。

**重设计项 R4: M 尾块（隐式 OOB 依赖 → 静态 full/rem_m 分解 + TIR if + 静态切片）**

- **源方案**：快路径 `T.copy`（S1a/S8）在尾 tile（M % block_m ≠ 0）依赖 GPU `T.copy` 越界保护（未文档化）；masked 分支有行谓词（L155–157）但仅 `_needs_mask` 时激活。
- **NPU 新算法**：host 静态分解 `full = M // block_m`、`rem_m = M % block_m`、`num_tiles = full + (rem_m > 0)`；persistent 体内 TIR if（`if t < full:` 全块 rows_this=block_m；`else:` 尾块 rows_this=rem_m）；全部 `T.copy` 切片静态（行切片 `[row : row+1]` 单行粒度 + 尾块行循环 bound=rem_m 静态）；**行串行结构下行级 OOB 自然消解**——行循环 `for r in range(rows_this)` 只迭代有效行（无行谓词、无越界拷贝；尾块 UB 行概念不存在——每行用同一 x_ub (1, BLOCK_N) 槽位）；输出 `lse_ub[r]`（r < rows_this）→ `T.copy(lse_ub[0:rows_this], y[base:+rows_this])` 静态切片。
- **语义保持论证**：尾块有效行的公式与全块完全一致（不存在的行不装载不计算不写出——源语义「行 OOB 以 −inf 载入、不写出」保持）；不依赖任何 `T.copy` 越界行为。
- **依据**：single SIMD `_logsumexp_kernel_single.py` Stage 3 全量 PASS 实证（同款 TIR if + 静态切片）；`examples/ascend/example_gemm.py` L43–56（循环变量 TIR if 先例）；SIMT tiled L233–250（tiled 契约的尾块分支实证）。

**重设计项 R5: running 统计量驻留（register fragment 跨 chunk → 每行一对 alloc_var 寄存器跨 chunk SimdVF 块）⭐（本变体特有核心风险项）**

- **源方案**：`row_max/row_sum/prev_max` 为 GPU register fragment，生命周期覆盖整个 chunk 循环（block 内无域边界）。
- **NPU 新算法**：**行串行结构（R1）使 running stats 退化为标量级**——每行迭代作用域声明 `m_run = T.simd.alloc_var("float32")`（broadcast 形态：vdupv 产出，全 lane 同值）与 `s_run = T.simd.alloc_var("float32")`（lane-0 形态：vcadd 产出，值在 lane 0，其余 lane 不消费）；行首 init SimdVF 块内 `m_run = vdup(−inf)`、`s_run = vdup(0.0)`；每个 chunk 的 SimdVF 块内读写（pass1 产出 m_c → `prev_brc = m_run`（SSA 快照，eager builder 立即求值）→ `m_run = vmax(m_run, m_c_brc)`；pass2 后 `s_run = vadd(vmul(s_run, vexpdif(prev_brc, m_run)), s_c)`）；行尾 epilogue SimdVF 块内消费（`out = vadd(m_run, vln(s_run))` → vcvt → `vsts(lse_ub[r], ·, ONEPT)`）。`T.annotate_buffer_versions({x_ub: 2, lse_ub: 1})`。
- **语义保持论证**：驻留位置（GPU 寄存器 → local.var 寄存器）与读写域结构变化不改数值链——m_run/s_run 的更新公式逐行与源 S3/S6 相同（fp32；broadcast/lane-0 形态仅是向量寄存器内的值布局，算术语义逐 lane 独立）；V1/V4 位级一致覆盖全部 chunk 合并路径。
- **机制依据与残余风险**：① `alloc_var` 是 local.var 可变寄存器（`tilelang/language/allocate.py` L103–168 `sblock_alloc_buffer`，无 SimdVF 上下文检查——声明位置自由）；② SimdVF 块内跨 `T.serial` 循环携带有实测回归（`test_simdvf_arithmetic.py` L86–110 `serial_vreg_accum`：alloc_var 声明于 SimdVF 块内、`T.serial` 循环内 `vmula` 累加、循环外 `vsts` 读出，NPU PASS）；③ SIMD_VF 域"keeps the enclosing scope"（`src/ascend/transform/vf_regions.h` L50–53——域体按外围作用域访问，外围作用域分配的 buffer 在域内可寻址）；④ `insert_sync.cc` L1308 对 local/local.var 依赖跳过同步插入（寄存器为引擎私有，无跨引擎同步需求）。**无单一先例覆盖「simd-typed alloc_var 在 SimdVF 块外声明、多块内访问」的组合**——列为 §9.1 风险 1，Stage 3 首编即触发验证；**fallback 预案**（若报 buffer 作用域/VFChecker 错误）：stats UB 往返形态——`stats_ub (2,) fp32` + flash core.py L103/L117/L120–126 同构访问（chunk 末 `vsts(m_ub, ·, ONEPT_B32)` / chunk 首 `vld(m_ub, "BRC_B32")` 广播装载 + `mem_bar("VST_VLD")` 排序），语义等价、API 全先例，代价 = 每 chunk 2 vsts + 2 vld + 1 barrier ≈ 5 指令/chunk/行（wl2 占比 0.14%）。
- **意图承接**：源 #7 合并式标量最小化的「重缩放开销 O(rows·chunks)」意图由寄存器驻留直接保持（7 向量指令/chunk/行，wl2 占比 0.39%——D-2）；源 fragment 跨 chunk 存活意图由 local.var 生命周期承接。

**重设计项 R6: 线程级逐元素并行 → 显式 2048-bit SIMD 指令流（本变体特有）**

- **源方案**：`T.Parallel(block_m, tile_n)` 线程域逐元素并行（threads=256），`T.reduce_max/reduce_sum` 经 warp shuffle 树归约。
- **NPU 新算法**：`with T.SimdVF():` 块内**显式 `T.simd.*` 指令流**（每指令 64 fp32 / 128 fp16 lane），每行每 chunk 两遍扫描：
  - **pass 1（chunk max）**：`acc = alloc_var(f32)` ← `vdup(−inf)`；逐 64 元素组 `xh = vld(x_ub[0, g·64], dist="UNPK_B16")` → `c = vcvt(xh, "float32", part=0)`（fp32 输入直连 `vld` NORM）→ `acc = vmax(acc, c)`；遍末 `m_c_brc = vdupv(vcmax(acc))`（单次跨 lane 全归约 + lane 0 广播）；
  - **max 合并**：`prev_brc = m_run`（快照）→ `m_run = vmax(m_run, m_c_brc)`（broadcast 形态）；
  - **pass 2（chunk sum of exp）**：`accs = alloc_var(f32)` ← `vdup(0.0)`；逐组重装载 → `e = vexpdif(c, m_run)`（**融合 exp(c−m) 单指令**，fp32 同宽强校验）→ `accs = vadd(accs, e)`；遍末 `s_c = vcadd(accs)`（单次跨 lane 全归约，lane-0 形态）；
  - **sum 合并（flash 重缩放）**：`s_run = vadd(vmul(s_run, vexpdif(prev_brc, m_run)), s_c)`——rescale 因子 `vexpdif(prev_brc, m_run)` = exp(prev − m_run) 全 ≤ 0 无溢出；
  - **行尾 epilogue**：`out = vadd(m_run, vln(s_run))` → `vcvt(out, dtype)`（fp32 输出免）→ `vsts(lse_ub[r], out_h, pset(16,"PAT_VL1"), "ONEPT_B16")` 单点存储（fp32 为 ONEPT_B32 + pset(32)）。
  - 尾 chunk（rem 列）：同上，组循环 bound = G_tail、尾组先 `vsel(c, vdup(−inf,f32), update_mask(k_tail,32))` 掩码再进算术。
- **语义保持论证**：公式逐项恒等（max 精确选择无舍入；vexpdif = exp(src0−src1) 同式融合，硬件舍入差 ≤ 减法 0.5 ulp 量级、经 flash 形式保证的 |x−m| ≤ 0 上界（exp 参数 ≤ 0）下相对误差 ≪ 容差——single 变体 §1.6.1 O3 同款论证）；累加分组（lane 内跨组累加 + 单次跨 lane 全归约）vs 源 chunk 内树形 → fp32 树深级差（V1/V2 ≤1 ulp、违反率 0）；NaN 传播与 ±inf 角点行为逐元素等价（V4 位级一致 + NaN 模式一致；V5 首 chunk 全 −inf 角点位级一致——`vexpdif(−inf,−inf)=exp(NaN)=NaN` 由 single 变体 Stage 3 Boundary-all-neg-inf NPU 实证）。
- **依据**：`examples/ascend/flash_attention/core.py` L94–137（SimdVF 行规约完整同构：vcmax/vcadd 全归约 + ONEPT 存储 + vexpdif 融合 + BRC 广播）；single SIMD `_logsumexp_kernel_single.py` L135–211（**同族 SimdVF 行规约指令流 Stage 3 全量 PASS 实证**——本设计的 pass1/pass2/epilogue 逐指令复用其验证过的链）；`tilelang/ascend/language/simd.py`（API 签名权威，§3.4 逐条行号）。
- **意图承接**：源 #6 树形归约的低延迟意图由跨 lane 全归约内在（单指令全 64-lane 收敛）承接；源 #4 SMEM staging 意图由 UB 单跳 + UNPK 装载承接；源 #3 快慢双路径意图由「full chunk 无谓词 + 尾 chunk 组级掩码」更优形态承接。

### 0.7 标杆实现

- **源算子路径**：`/home/z00875089/TileOPs-fork/tileops/kernels/reduction/logsumexp.py`（`_logsumexp_kernel_tiled` L105–194——语义/算法/优化手段解读的唯一来源；`LogSumExpKernel` 类 L246–557——wrapper 契约与 tile_n 启发式；`_primitives.py`——`DEFAULT_ALIGNMENT=256` L30 / `MAX_SINGLE_TILE_COLS=32512` L39 / `align_up` / `compute_tile_n` / `device_smem_budget`）。
- **源仓参考实现/测试基准**：`torch.logsumexp(x.float(), dim).to(x.dtype)`（源仓 `tests/ops/test_softmax.py` `ref_program` 直移植，fp32 计算域 golden）——§8.1 golden 直接采用（torch on NPU），保证验证独立性（不复刻 §0.6 的 NPU 算法）。
- **wrapper 对接面（SIMD 家族）**：manifest `source.kernel = tileops/kernels/reduction/logsumexp_simd/logsumexp_simd.py`（`LogSumExpSimdKernel`，Stage 5 集成对象）；契约口径与 SIMT 家族 wrapper（`examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp.py` dispatch）一致。
- **SIMD 单 tile 姊妹（先例，非源）**：`examples/logsumexp_simd/_logsumexp_kernel_single/`（DESIGN.md + 实现 86 用例全 PASS + RETROSPECTIVE）——pass1/pass2/epilogue 指令链、Persistent+TIR if 嵌套、契约 assert、golden + `_source_lse_sim`、bf16 宽化装载实证（max_diff=0）、alloc_var 惯用法（Stage 3 教训）的参照。
- **SIMT tiled 兄弟变体（先例，非源）**：`examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp_kernel/_logsumexp_kernel_tiled.py` + `_logsumexp_kernel_tiled_DESIGN.md`（61 用例全 PASS + Stage 4 perf_opt 采纳版：wl2 100.2µs / wl3 17.6µs）——tiled 契约、wrapper 落点复算、chunk 流水结构、测试组织（含 L0 ⑤ 首用例顺序注记）、fp16-e 备选实测否决（50–60× 回退）的参照。
- **其他先例**：`examples/pilot/online_softmax/online_softmax.py`（streaming 路径 = chunk 流式 + flash 合并 + stats UB 驻留的 measured 正例）；`examples/ascend/flash_attention/core.py`（stats UB 往返 fallback 形态：vsts ONEPT + vld BRC_B32 + mem_bar VST_VLD）。

---
## 1. 概述

### 1.1 算子名称

`_logsumexp_kernel_tiled`（op 名 `LogSumExpFwdOpSimd` / manifest id `logsumexp_simd` 的**分块流式成员**；SIMD 家族 = 本仓 `examples/logsumexp_simd/` 独立变体谱系）。产出文件：本算子目录 `_logsumexp_kernel_tiled.py`（Stage 3 交付，§10.2——设计期尚未生成）。

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
| 13 | 两遍重读（先全行 max 扫描，再 exp+sum 扫描，均 chunk 流式） | pass A：流式求全局 m；pass B：流式求 Σexp(x−m) | 数学等价（解析式直译） | 免 online 合并（7 指令/chunk = 0.39%） | ❌ **弃选**：① GM 扫描 2 遍（流量 +100%——online 单遍是本变体存在理由；D-2 wl2：MTE 带宽需求 12.0K→24.0K cyc/core，带宽余量 10.2×→5.1×——计算主导域下端到端增量被流水掩盖，**故流量非决定性**）；② **角点保真破坏（决定性）**——两遍式 = single 形态语义：首 chunk 全 −inf 且后续含有限值的行给出**有限值**，源 tiled online 递推给出 **NaN**（§0.1 ④、verify_equiv V5 实测分歧 9.70 vs NaN）——违反迁移保真；③ 省下的 merge 指令仅 0.39%（wl2），远不抵语义分歧 |
| 14 | 哨兵钳位（m_run 初始化 −MAX_FLOAT 防 −inf−−inf NaN） | init `m_run = vdup(−3.4e38)` | **不等价**（边界语义改变） | 消 NaN 角点 | ❌ **弃选**：全 −inf 行 y = −MAX_FLOAT + log(0) = −inf ≠ 源 NaN（§0.1 ②）；首 chunk 全 −inf 角点同理变为有限值（§0.1 ④）——源以 NaN 传播为语义而非缺陷（与 pilot online_softmax 的 O6 哨兵不同语境：该处是无 −inf 数据域的防御，本处 −inf 是契约内数据）。V5 位级锚点保持源行为 |

**R2 在线算法**：**有，且源即在线**——本变体与 single 的调研问题方向相反（single 问「驻留可行时 online 是否必要」，本变体问「驻留不可行时的在线算法族选哪种」）。证据三层齐全：① 源仓自身：`_logsumexp_kernel_tiled`（L105–194）即 online 变体（分块 running (max,sum) + flash 重缩放——「存在在线算法」的直接证据）；② 参考表：ALG-softmax R2「online softmax（分块 running max/sum + 重缩放）」命中行；③ 结构判据：logsumexp 的 (max, sum) 统计量均可增量合并 → 在线变体存在。**粒度轴调研**（本变体特有）：online 更新粒度有三档——chunk 粒度（源/选定）、组粒度（R1 #8）、行批粒度（= chunk 粒度 × block_m，与粒度无关）——组粒度弃选（指令净增 37× + 角点保真分歧 + 零流量收益），**chunk 粒度选定**（merge 开销 O(rows·chunks) = 7 指令/chunk/行 = wl2 0.39% 指令占比，D-2）；**两遍重读与多 kernel（R1 #12/#13）作为「在线的替代」同台比较后弃选**（R3 表：扫描遍数 2/4 vs 1）。

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
#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定、经 §0.6 R1/R2/R5/R6 落地的 online flash（chunk 粒度）算法。公式层源已是该族最优形态（e-base 减 max flash 合并、log 每行 1 次——R1 候选 2/3 弃选），本节优化集中在**划分 / 指令 / 归约时机 / 驻留层级**，每项四要素齐备。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| O1 | 减 max 稳定化 + flash 形式合并基（必选确认） | `y = log(Σ e^x)` | §1.3 递推式：合并基 = 合并后 m_run（非朴素 m_c 基） | 指数平移恒等；合并式全部 exp 参数 ≤ 0（p ≤ m_run、x^c ≤ m_c ≤ m_run）——**朴素 m_c 基在 fp16 域 max 间隙 131008 下溢出**（`TRAP-NAIVE-ONLINE-MERGE` measured：间隙 > 88.7 即 fp32 溢出），flash 形式安全 | 消溢出（源已含，语义层保留）；V4/V5 corner `max-gap 131008` 实测位级正确 |
| O2 | **BLOCK_N = tile_n 直保留 + 尾 chunk 组级 −inf 掩码 + 消 N_padded**（布局级）⭐ | 源：chunk 宽 tile_n 但 N_padded = align_up(N,256) 对齐（(33,33000) 搬 33,280 列，+280 pad 列）；`_needs_mask` 运行期双路径（最后 chunk 逐元素 masked）；〔对照〕SIMT 兄弟变体 fragment pow2 约束下 BLOCK_N = pow2 ≤ tile_n（16640→16384、25600→16384） | ① BLOCK_N = 最大 64 倍数 s.t. BLOCK_N ≤ tile_n ∧ 2·BLOCK_N·elem ≤ UB 预算（§4.5）——SimdVF 无 fragment，**不重划**；② full chunk 无谓词直载（tile_n 256 倍数 → 64 整除零尾组）；③ 尾 chunk 静态 rem 切片 DMA（只搬 rem 列）+ 满组直载 + 尾组 `vsel(c, vdup(−inf,f32), update_mask(k_tail,32))` | flash 合并对 chunk 划分满足**结合律**（每元素恰计入一个 chunk、合并代数恒等；浮点下仅重缩放乘法分组舍入差）——**V1**：8 组 (M,N,tile_n)（含 25600 非 pow2 4-chunk、16640 masked-tail、tiny、63-chunk 长链 1024000/16384）候选链 vs 源链 **fp64 位级一致（ulp=0）、fp32 ≤1 ulp、违反率 0**；**V6** 划分敏感性扫描（BLOCK_N ∈ {4096, 8192, 16384, 16640, 25600, 32768} × 3 shape，**非 pow2 BLOCK_N 首次纳入**）全部 ≤1 ulp；−inf 掩码双幂元中性（max(−inf,·)=· ∧ exp(−inf−m)=0，V1 尾 shape 覆盖） | 非 pow2 tile_n 下 chunk 数保持源划分：wl3 4 chunks（vs 兄弟 pow2 7，merge/DMA 迭代 −43%）、(33,33000) 2（vs 3）；GM 读消 pad 列（33000：−0.85%）；消 `_needs_mask` 双路径（统一 full/tail 静态二分支）；全 chunk 零尾组零掩码开销 |
| O3 | vexpdif 融合（指令级，SIMD 特有；两处受益） | `d = vsub(c, m); e = vexp(d)`（两指令）；rescale 因子 `vsub(prev, m); vexp` | `e = vexpdif(c, m_run)`（单指令，dst = exp(src0−src1)）；rescale `vexpdif(prev_brc, m_run)` | 公式恒等（同式融合——verify_equiv 候选链按融合式建模，V1 已含）；硬件差异仅为融合内部减法舍入时机 ≤0.5 ulp（fp32 减法）→ 经 flash 形式保证的参数 ≤ 0 上界下相对误差 ≤ 11.1·2⁻²⁴ ≈ 6.6e-7 ≪ 16-bit 容差 1e-2（4 个数量级余量；fp32 容差 1e-3 同 3 个数量级余量）——single 姊妹 O3 同款论证 + 本变体 V1/V3 复验 | −1 VALU op/64 元素（vsub @51 档）×pass2 + −1/chunk rescale：wl2 每 chunk −257 ALU 指令 ≈ 计算项 −9.6%（单源脚本复算）；`VP-2026-0014`：vexpdif fp32-only（16-bit 装载须先 vcvt——R1 #10 弃选的 API 侧佐证） |
| O4 | 延迟跨 lane 归约（归约时机，SIMD 特有） | 逐组跨 lane 归约再合并（`T.reduce` 在 SimdVF 的自动 lower 形态：每组 `vcmax(c)` + `vmax(acc, reduced)`，`src/ascend/transform/ascend_simdvf_lower_parallel.cc` L818–841） | 逐组元素级累加 `acc = vmax(acc, c)` / `accs = vadd(accs, e)` + 每 chunk 遍末**单次** `vcmax`/`vcadd` | max：结合/交换律精确成立（任何分组同值，无舍入）；sum：fp32 累加分组重排（lane 内跨组序贯 + 单次 64-lane 全归约 vs 逐组全归约+标量合并）→ 树深级 ulp 差；**V2**：3 shape（含 25600/16640 非 pow2 chunk）≤1 ulp、违反率 0 @1e-3 | wl2（G=256）每 chunk −(G−1)×2 = −510 指令 ≈ chunk 指令 −28%；计算项 −15.7%（vs 逐组形态，单源脚本复算）；比自动 lower 形态更优——显式内在的价值点 |
| O5 | 合并式标量最小化保持（寄存器向量合并） | 源：`for i in T.Parallel(block_m)` per-row 标量合并体（GPU 标量管线）；〔对照〕组粒度 online 每 64 元素组合并（R1 #8） | 每 chunk 每行 7 条向量指令完成 (max,sum) 双合并：`vcmax`+`vdupv`（chunk max 广播）+ `vmax`（max 合并）+ `vcadd`（chunk sum）+ `vexpdif`+`vmul`+`vadd`（flash 重缩放）——m_run broadcast 形态 / s_run lane-0 形态的 alloc_var 寄存器运算 | 合并代数逐式与源 S3/S6 恒等（fp32；broadcast/lane-0 仅是 64-lane 寄存器内的值布局，算术语义逐 lane 独立，lane-0 链自洽）——V1 fp64 位级（ulp=0）+ V4 位级复验 | merge 指令占比 wl2 0.39%（7/1,799 per chunk）、(33,33000) 0.19%（chunk 更大摊薄更薄）；vs 组粒度 online 每行 −512 指令（37×）；承接源 #7「重缩放开销 O(rows·chunks)」意图 |
| O6 | UB 两遍重读（驻留层级确认项） | 〔对照〕VRF 整 chunk 驻留单装载（pass2 免重读重转，R1 #7） | pass1/pass2 各自从 x_ub 重装载（`vld UNPK_B16`+`vcvt` ×2 遍） | 数值同链（两遍读同一 UB 版本，V1 已含该形态） | UB 重读 4.68K cyc/core（wl2，VRF↔UB 256B/cycle）= 计算项 3.8%（26× 余量）非瓶颈；换来 VRF ~3.3KB 零行数据驻留（vs R1 #7 的 256–400 数据寄存器 = fp32 chunk 64–100KB）+ 结构简单（无 alloc_local 数组索引）+ 不与 m_run/s_run 跨块驻留（R5）叠加无先例风险 |
| O7 | pass-1 累加器跨 chunk 折叠 max-merge（拒绝项） | 〔候选〕acc 不按 chunk 复位，chunk 末直接 `m_run = vdupv(vcmax(acc))`，省 per-chunk `vmax` 合并 + acc 复位 | —（不采纳，保留源式显式 m_c 合并） | 等价（max 精确、结合律）但收益微小 | 省约 2 指令/chunk/行（128 lane-ops/51 ≈ 2.5 cyc of 3,348 cyc/chunk）= **0.07% 计算项**；代价：pass1 acc 生命周期跨 chunk + 失去与源 S2/S3 逐式同构的保真锚点（V1 fp64 ulp=0 链按源式建模）——不值得 |
| O8 | 输出饱和角点封闭（确认项，非优化；tiled 契约 N 无上限） | — | `vcvt(out, fp16/bf16)` 默认 sat=True | **契约空间内数学封闭**：out = m + ln(Σ) ≤ 65504 + ln(N) 且 Σ ≥ 1（max 元素贡献 exp(0)=1）→ 饱和阈值 65520（fp16 IEEE 舍入到 inf 的中点）⟺ **N ≤ e^16 ≈ 8.86M**；**V7**：全 65504 行 N=102400 → y=65515.54 < 65520 ✓、N=1,024,000 → y=65517.84 < 65520 ✓（0 分歧 lane）；N > 8.86M 时 sat 钳位与 IEEE 溢出可能分歧——**源同样经 `T.cast` 共享该行为**（迁移保真），§9.3 文档化 | 契约空间（wrapper tile_n ≤ 32,512、显式 config 任意大 N 但 < 8.86M 实际不可达）内免额外钳位指令 |

**优化结论**：**采纳 4 项（O2 布局级 / O3 指令融合 / O4 归约时机 / O5 寄存器合并保持）+ 3 项必选/确认（O1 稳定化、O6 驻留层级、O8 饱和封闭）+ 1 项拒绝（O7，0.07% 不值得）**。优化后主路径公式（§3.1 拆解的唯一输入）：

```
per row:  m_run = −inf; s_run = 0                                    # alloc_var 寄存器
per chunk c (BLOCK_N = tile_n, 64 整除):
  pass1:  acc = vmax* 逐组累加 → m_c = vcmax(acc)（广播 vdupv）       # 延迟跨 lane 归约
  merge:  prev = m_run; m_run = max(m_run, m_c)                       # 源 S3
  pass2:  accs = vadd* 逐组累加 vexpdif(c, m_run)                     # 融合指数，UB 重读
  merge:  s_run = s_run·vexpdif(prev, m_run) + vcadd(accs)            # 源 S6 flash 重缩放
  tail:   尾组 vsel(−inf) 掩码后同上（rem 列静态切片）
epilogue: y = cast(m_run + vln(s_run), dtype)（ONEPT 单点存储）
```

**等价性机器验证（D-1）**：已生成并执行 `examples/logsumexp_simd/_logsumexp_kernel_tiled/verify_equiv.py`（fp64 参照 + 角点值：全零/全等/单主导/±65504 极值/subnormal/深度下溢 −745/部分 −inf/全 −inf/含 +inf/含 NaN/**首 chunk 全 −inf**/**尾 chunk 全 −inf**/N=1/非 64 倍数 N（257、33000 的 rem=16360）/超长链 N=1024000（63 chunks）/非 pow2 tile_n（16640、25600）；torch CPU；候选链按「BLOCK_N=tile_n 划分 + 64 元素组分桶 + lane 内跨组累加 + 单次跨 lane 归约 + vexpdif 同式融合 + 尾组 −inf + flash 逐 chunk 合并」逐要素模拟，基线链按源 tiled 算法（tile_n chunk + fp32 中间 + flash 合并序）模拟）。执行结果（2026-10-17，tilelang-fork 工作区 + torch CPU，**全部 EQUIV_PASS**）：

| 验证项 | 对照（候选 vs 基线/精确） | 最大 ulp 差（fp32 位级） | 最大相对差 | 违反率（@容差） | 结论 |
|--------|--------------------------|--------------------------|------------|---------------------|------|
| V1（O1+O2+O3+O4，8 shape × 2 参照 = 16 行） | SIMD online 链（BLOCK_N=tile_n、含融合/延迟归约/尾组掩码建模）vs 源 tile_n 链（fp32 链对照 + fp64 参照）：(8,32768) tn=16384、(4,102400) tn=25600、(33,33000) tn=16640、(4,32768) tn=32768、(8,100) tn=64、(3,257) tn=256、(2,1024000) tn=16384（63-chunk 长链）、(57,32768) tn=16384 | fp64 参照 **0（全部 8 shape）**；fp32 链 **≤1**（5 shape 为 0） | ≤6.6e-08 | 0.000000（@1e-2 / @1e-12 fp64） | **EQUIV_PASS** |
| V2（O4，3 shape） | 延迟跨 lane 归约 vs 逐组跨 lane 归约（两种 fp32 分组） | **≤1**（(4,102400) tn=25600 为 0） | ≤6.5e-08 | 0.000000（@1e-3） | **EQUIV_PASS** |
| V3（A5 链 vs golden，8 shape） | SIMD online 链 vs `torch.logsumexp`（fp32） | ≤1 | ≤6.5e-08 | 0.000000（@1e-3） | **EQUIV_PASS** |
| V4（源链 vs 候选链，16-bit 输出，3 N × 2 dtype = 6 行） | 18 类特殊行（含 NaN/±inf/部分 −inf/全 −inf/极值/subnormal/深下溢/max-gap 131008）fp64→fp16/bf16 输出对照 + NaN/±inf 模式逐位比对（N=32768 tn=16384、N=33000 tn=16640、N=102400 tn=25600） | **0（位级一致）**；NaN/±inf 模式全部一致 | 0 | 0.000000（@1e-2） | **EQUIV_PASS** |
| V5（tiled 特有角点，信息项） | ① **首 chunk 全 −inf 且后续含有限值**行：候选 vs 源 tiled（NaN 位级比对，int64 位模式——torch.equal 对 NaN 恒 false 须位级比较）vs single 形态 vs torch；② **尾 chunk 全 −inf**（首个含有限值 chunk 之后）：候选 vs torch | ① 候选 = 源 = **NaN（位级一致）**；single 形态 = torch = 9.704（有限）——分歧为**源固有**（§0.1 ④、§9.3），非采纳项失败；② 候选 = 9.704 = torch（**中性**，m_run 有限 → exp(−inf−m)=0） | — | — | **EQUIV_PASS**（迁移保真达成） |
| V6（O2 划分敏感性，12 行） | BLOCK_N ∈ {4096, 8192, 16384, 16640, 25600, 32768} × {(4,102400), (33,33000), (8,32768)}（BLOCK_N ≤ N 取值域内）vs 源链 | **≤1**（(4,102400) 全档 0；(33,33000) 全档 1；含非 pow2 档 16640/25600——**首次纳入非 pow2 扫描**，SIMT 兄弟 V6 仅 pow2） | ≤6.5e-08 | 0.000000（@1e-3） | **EQUIV_PASS** |
| V7（O8，信息项） | sat=True 钳位 vs IEEE 溢出（全 65504 行 × N ∈ {102400, 1024000}） | — | — | — | **0 分歧 lane**（y ≤ 65504+ln(N) < 65520；封闭域 N ≤ e^16 ≈ 8.86M） |

> **全部采纳项 EQUIV_PASS**（V1 fp64 位级 + fp32 ≤1 ulp、V2/V3/V6 ≤1 ulp、V4 位级一致 + NaN 模式一致、V5 角点位级保真）。V5 同时证实：首 chunk 全 −inf → NaN 是源 tiled online 递推的固有语义（single 形态与 torch 给有限值），本设计迁移保真保持（§9.3 文档化 + §8 Boundary ④ 用源语义模拟器验证，不用 golden）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 本变体的模式本身即「显式向量化」——逐元素计算 100% 以 `T.simd.*` 2048-bit 向量内在表达（每指令 64 fp32 / 128 fp16 元素）。本节盘点全部循环与标量计算点，确认无逐元素标量残留、逐项说明保留的非向量化点（与 §6 循环结构一致）。

| # | 计算点 | 实现形态 | 向量化状态 | 不可替代理由（非向量化点必填） |
|---|--------|-----------|-----------|------------------------------|
| 1 | 输入装载 + fp32 提升（逐元素） | `vld(x_ub[0, g·64], dist="UNPK_B16")` + `vcvt(part=0)` 每 64 元素组 1+1 指令（fp32 输入 `vld` NORM 直连） | ✅ 2048-bit 显式 | —（`testing/ascend/language/test_tilelang_ascend_simdvf_cast.py` L64–80 位级回归；single 姊妹 L154–162 Stage 3 实证） |
| 2 | 尾 chunk 边界掩码（逐元素条件） | 满组直载零掩码；尾组 `vsel(c, vdup(−inf,f32), update_mask(k_tail,32))`——向量谓词选择，仅尾 chunk 最后 1 组 × 2 遍 | ✅ 谓词化 | —（`examples/ascend/deepgemm/kernels/transform_sf.py` L205/L215/L249 vsel 先例；`test_simdvf_integer_predicate.py` L261 update_mask 先例；比源逐元素 `T.if_then_else` 载入更优——掩码只付 1 指令/遍而非 1 select/元素） |
| 3 | chunk 内行 max（逐元素比较） | `vmax(acc, c)` 逐组 + 遍末单次 `vcmax`（+`vdupv` 广播） | ✅ | —（flash core.py L115–118 同构） |
| 4 | 稳定化指数（逐元素 exp） | `vexpdif(c, m_run)` 融合单指令/组 | ✅ | —（flash core.py L130–133 同构；O3） |
| 5 | chunk 内行和（逐元素累加） | `vadd(accs, e)` 逐组 + 遍末单次 `vcadd` | ✅ | —（flash core.py L134–137 同构） |
| 6 | (max, sum) flash 合并（跨 chunk 统计量） | 7 条向量指令/chunk/行（O5）：`vmax` 合并 + `vexpdif`/`vmul`/`vadd` 重缩放，m_run broadcast / s_run lane-0 形态 alloc_var 寄存器运算 | ✅（寄存器向量合并） | —（vs 源 GPU 标量管线合并：等价向量形态；vs 组粒度 online：37× 指令差，R1 #8 弃选） |
| 7 | 每行 epilogue：y = m + log(S) + cast + 写出 | `vadd(m_run, vln(s_run))` → `vcvt` → `vsts(ONEPT)`——4–5 向量指令/行，值在 lane 0 有效（其余 lane 垃圾不逃逸——ONEPT 只取 lane 0） | ✅（行粒度向量指令） | 依赖归约终值（s_run 定稿后才能求 log）——tile 级标量（4–5 指令/行 vs 1,799 指令/chunk/行元素段，wl2 占比 ≈ 0.07%），无法并入元素循环 |
| 8 | 行循环（任务内 rows_this 行串行） | `for r in range(rows_this)` Python 展开（rows_this 静态 ∈ {block_m, rem_m}；主流 bm=1 → 单迭代退化） | ❌（保留串行） | **alloc_var 是单寄存器语义（一个 2048-bit 向量寄存器），无法按行索引为数组**——行批在线 running stats 只能落 UB（= fallback 形态，§0.6 R5）；行串行使 stats 退化为每行一对寄存器（主选形态）；行间独立但 SimdVF 无行级并行原语（无线程域，行到核映射由 persistent 调度承接——M 维并行在任务层已消耗） |
| 9 | chunk 循环（行内 num_full chunk 串行） | `for c in T.serial(num_full)` + 尾 chunk Python 静态分支 | ❌（保留串行） | **running stats 循环携带依赖**（s_run 更新依赖前一 chunk 终值——online 递推的算法本质，顺序不可交换）；chunk 间的并行性已由 x_ub 双版本流水**重叠**利用（MTE2 c+1 ∥ VALU c，重叠非并行）；`P-LAYOUT-ROW-RESIDENCY` measured：串行循环用 T.serial 保持双版本流水顺序依赖 |
| 10 | 组循环（chunk 内 G 组 × 2 遍） | `for g in range(G)` Python 展开（G ≤ 430）+ 尾组静态分支 | ❌（循环载体，体内全向量化） | 组循环体 = 3–4 条向量指令（#1–#5 已全向量化）；两遍扫描数据流本质（pass2 依赖 pass1 的 m_run 终值）；G 静态、展开形态与 single 姊妹（G ≤ 508 Stage 3 实证）一致 |
| 11 | 尾批 TIR if 分支（`if t < full_m`） | 运行时控制流，分支体 = `row_task` 宏两次实例化（rows = block_m / rem_m 静态） | ❌（保留分支） | 依赖循环变量 t 的 M 尾批边界语义；分支体两侧均为静态切片 + 静态行数（SIMT tiled 兄弟 L233–250 同款实证）；每任务 1 次分支判断，非热点 |
| 12 | host 侧 BLOCK_N/num_full/rem/full_m/rem_m/G/k_tail 计算 | Python 算术（builder 闭包静态化） | n/a | host 元数据计算，不在 kernel 内（per-shape JIT）；`_tile_n_for_block_m` 的 wrapper 侧启发式不属于本 kernel 契约（§0.2） |
| 13 | lse 写回（任务尾 rows_this 元素） | `T.copy(lse_ub[0:rows_this], y[row_base:+rows_this])` MTE3 DMA | ✅（DMA 块搬运） | 数据搬运非 VALU 计算；静态切片（兄弟 L222 同款） |

**向量化结论**：逐元素计算 100% 显式向量化（#1–#6 全部 2048-bit 内在，API 均有 `examples/ascend` / `testing/ascend` 位级或实测佐证）；保留 4 处非向量化点——#7 行级 epilogue（4–5 向量指令/行，依赖归约终值，占比 0.07%）、#8 行循环（alloc_var 单寄存器语义不可行索引 + 主流单迭代退化）、#9 chunk 循环（online 递推顺序依赖，已由双缓冲流水重叠利用）、#10/#11 组循环载体与尾批分支（数据流本质 + 运行时边界语义，先例佐证）——理由类别 = tile 级标量 + 算法顺序依赖 + 寄存器语义限制 + 边界处理，均非「实现简单」类借口；与 §6 循环结构一致（§6 无逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（[M, N] 行主进 / (M,) 行主出），核内布局与 lane 映射是设计变量。规约类算子必选候选：水平归约（lane 映射到归约轴 N）vs 垂直扫描（lane 映射到独立行 M）；另评估 tiled 特有轴决策（chunk 划分粒度 / running stats 驻留层级——§1.6.0 已裁，此处收口布局面）。

**轴质量评分**（对选定轴 N）：

| 评分项 | 判定 |
|--------|------|
| 整除性 | wrapper tile_n 为 256 倍数 → 64 整除 ✓ full chunk **零尾组**（组宽 64）；尾 chunk rem 任意（组级掩码吸收）；显式 config 任意 tile_n → BLOCK_N 取 64 倍数 clamp（§4.5）仍零尾组 |
| 尾 lane 浪费率 | full chunk 0%；尾 chunk 仅最后 1 组（k_tail/64 < 100%，wl3 类整除 rem=0 完全无尾 chunk；(33,33000) 尾 chunk k_tail=40/64 = 62.5% 有效——pad lane 的 exp(−inf−m)=0 不污染 sum） |
| 累加链形态 | 归约维 N = chunk 内 stride-1 连续向量维（组内 64 lane 连续装载）；无跨步系数；行切片 (1, BLOCK_N) DMA 连续 |
| repack 代价 | 无（I/O 原生行主连续，核内布局与 I/O 同构；UNPK_B16 装载即提升，无交错重排——规约序不敏感，flash 合并对划分结合律〔V6〕） |
| UB/VRF 容量影响 | chunk 驻留：x_ub (1, BLOCK_N) 双版本 65.5KB@wl2 / 102.4KB@wl3 / ≤220KB 上界（fp32 clamp）+ lse_ub ≤64B + VRF 流式 ~3.3KB（§4.5 预算表）——**block_m 与 UB 解耦**（x_ub 单行粒度，行数不进容量式；无 single 变体的 SUB 封顶机制） |

**候选矩阵**（规约类候选全枚举 + tiled 特有轴，弃选给量化理由）：

| # | 布局方案 | 向量化轴 / 驻留层级 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | 行主 [M,N] 原生 + chunk 内**水平归约**（lane→N 连续轴，64 元素组，UB 两遍重读 + alloc_var stats + BLOCK_N=tile_n，persistent 行块任务） | N / UB 两遍 + 寄存器 stats | 无 | 连续 GM 行切片（wl2 行 64KB/chunk）；归约维 = stride-1 组内连续；flash/cast/arithmetic 三先例 + single 姊妹 Stage 3 实证 + SIMT tiled 兄弟（tiled 契约）实证；vcmax/vcadd 单指令全归约；任意大 N（chunk 流式） | ✅ **主选** |
| 2 | 垂直扫描（lane→M 行，N 向串行步进） | M | 无 | M 维短（wl3 M=4 → 4/64 lane 利用率 6%；wl2 1024 行可行但 N 串行 32768 步/行）；行内并行度归零；repo 无先例 | ❌ 串行化 N 维（32768 步 vs 256 组×64 lane），量化劣化 >10×；且 vertical 形态下 running stats 须跨行 维护（更复杂） |
| 3 | 跨核 N-split + partial (m,s) 合并（2D 分块 grid） | N（块内）+ 跨核 M×Nsplit | partial 统计量 GM 往返 | wl3 (M=4) 并行度 4→56 理论受益；(max,sum) 合并非 additive（须定序合并段）；partial GM 往返 + 跨核 flag / 第二 kernel；**浮点合并顺序变化破坏 V1/V4 位级保真论证** | ❌ 弃选（R1 #12 同源：复杂度 + 保真失效；wl2 满载零收益；兄弟 Stage 4 `[DESIGN_LIMIT]` M≪56 天花板同族记录；列 §9.1 风险 3 Stage 4 观察项） |
| 4 | 转置布局（列主驻留 + M 向归约） | M（转置后） | 核内转置链（通用转置 API 存在性未命中——未文档化假设） | repack 纯开销 + UB 翻倍；收益仅换归约轴到 M（M 短无收益） | ❌ repack 无收益（N 已是最长连续轴） |
| 5 | pow2 BLOCK_N 重划（SIMT 兄弟被迫形态） | N（同主选） | 无 | 非 pow2 tile_n 下 chunk 数虚增（wl3 7 vs 4：merge/DMA 迭代 +75%）；有效 lane 浪费（fragment 50.8%/78.1%——SimdVF 无此约束） | ❌ 弃选（fragment 约束在 SimdVF 不存在——重划是约束残留纯负优化，O2 反向论证 + V6 非 pow2 位级/≤1 ulp） |
| 6 | 逐组跨 lane 归约（自动 lower 形态直译） | N / 归约时机=逐组 | 无 | 每组 +1 vcmax +1 vmax（R1 #6 弃选形态） | ❌ 被 O4 延迟归约取代（−28% chunk 指令） |
| 7 | VRF 整 chunk 驻留单装载（SIMT 形态直译） | N / VRF 驻留 | 无 | −10% VALU（免 pass2 重读重转）vs **VRF 64–100KB 行数据驻留**（256–400 寄存器）+ alloc_local 数组复杂度 + 与 alloc_var 跨块驻留叠加无先例风险（R1 #7） | ❌ 弃选（D-2：UB 重读 26× 余量非瓶颈；VRF 压力 + 风险叠加得不偿失） |
| 8 | 组粒度 online 更新（每 64 组合并 stats） | N / 更新粒度=组 | 无 | 每 64 元素 +3 指令 ×G（37× merge 指令）+ 角点保真分歧（R1 #8） | ❌ 弃选（chunk 粒度已最优：7 指令/chunk vs 3 指令/组×G） |

**源码轴选择对照**（迁移任务必做）：源 GPU 也是 N 轴并行（`T.Parallel(block_m, tile_n)` 行内元素线程化 + `T.reduce_max/sum` dim=1）+ 行级 grid + chunk 串行——**轴结论一致**（N 是唯一高带宽连续轴、M 是任务轴，两代硬件共识；chunk 串行顺序依赖亦一致），GPU 轴选择是输入而非结论，已独立复核。承载结构完全重设计：GPU block×threads=256 → SimdVF 2048-bit 指令流；GPU fragment 跨 chunk 驻留 → alloc_var 寄存器；GPU N_padded 对齐 + 快慢双路径 → BLOCK_N=tile_n 紧凑 + 静态二分支（§0.6 R1/R2/R5）。

**布局决策结论**：选定方案 #1：核内布局 = `x_ub (1, BLOCK_N=tile_n)` dtype 紧凑（双版本）+ `lse_ub (block_m,)` + **VRF 流式零行数据驻留**（仅 stats 2 寄存器 + 累加器/常量 ~13 个）；向量化轴 = **N（chunk 内连续轴，64 元素组粒度）**；跨 lane 归约时机 = **延迟单次**（每 chunk 1 vcmax + 1 vcadd）；running stats 驻留 = **行作用域 alloc_var 寄存器**（跨 chunk SimdVF 块，风险 1 + fallback）；chunk 粒度 = **BLOCK_N = tile_n**（64 倍数 clamp 兜底）。该结论同步落入 §3.3 伪代码与 §6 循环结构；弃选方案量化理由见候选矩阵（#2 >10×、#3 保真失效 + 复杂度、#4 repack 无收益、#5 +75% chunk 数、#6 −28%、#7 VRF 压力、#8 37× merge）。**无实验裁决三件套**：主选与全部弃选候选的判定裕度均不依赖未实证常数（#5/#6/#8 为结构性指令计数差；#7 为容量水位与 26× 余量的量级差；#2/#4 为轴量级差；#3 为语义保真 + 复杂度弃选），不触发 skill Phase 2 第 3 项的强制裁决条件。

---
## 2. 编程模式选型

### 2.1 模式结论

**simd（SimdVF）**——用户显式指定（非默认 simt）；`auto_schedule: yes`（AutoSchedule 自动同步，无手动 flag）。kernel 模块 docstring 声明 `programming_mode: simd`、`auto_schedule: yes`（§10.3）。

### 2.2 选型理由

1. **用户显式指定**：本变体是 SIMD 家族（`examples/logsumexp_simd/`）的成员，用户在任务契约中指定 simd 承载——编程模式为输入约束，非选型变量（与 single 姊妹同族口径）。
2. **算子特征匹配**：纯 Vector 归约负载（无 GEMM/矩阵乘段）——special_func 主导（D-2 61%）+ 跨 lane 全归约（`vcmax`/`vcadd` 单指令）+ 逐元素向量算术，全部落在 AIV VALU；无 Cube 段 → 不需要 MixedKernel（mixed 仅当单 kernel 含 Cube `T.gemm` + Vector 段时使用，tilelang-ascend skill §3）。
3. **先例实证**：同族 single 姊妹以 SimdVF 实现 Stage 3 全量 PASS（86 用例，2026-10-17）——pass1/pass2/epilogue 指令链逐指令复用其验证过的形态；flash core.py / online_softmax streaming 为 chunk 流式 + flash 合并的 SimdVF/同构先例。
4. **SimdVF vs SimtVF 对本算子的差异**：显式 2048-bit 指令流（无线程域、编译器不自动向量化）换 per-chunk 指令级确定性（延迟归约/融合/掩码形态全部显式可控，O3/O4 的收益即来自显式形态优于自动 lower）。

### 2.3 模式影响

- **无 `threads=`**：`T.Kernel` 一维、无 threads 参数（TypeError）；行内并行度由 2048-bit 指令承载（每指令 64 fp32 / 128 fp16 lane），GPU `threads` 契约按 K3 舍弃（§0.5）。
- **`with T.SimdVF():` 域结构**：元素计算必须在域内以 `T.simd.*` 表达；`T.copy`（MTE2/MTE3）在域外；跨域值传递 = UB buffer 或 `T.simd.alloc_var`（local.var 寄存器）——本设计 m_run/s_run 跨 chunk 域驻留是后者（风险 1）。
- **SSA 限制**：域内 SSA 值不能跨域引用（`Immutable variable used outside its defining region`——single 姊妹 Stage 3 教训）；循环携带累加器必须 `alloc_var`（`test_simdvf_arithmetic.py` L86–110 先例）。
- **`T.simd.*` API 面**：装载（vld/vsts）、算术（vadd/vmax/vmul）、归约（vcmax/vcadd/vdupv）、超越（vexpdif/vln）、转换（vcvt）、谓词（vsel/update_mask）、常量（vdup/pset）——全部有先例（§3.4）。
- **AutoSchedule**：MTE2→VALU→MTE3 跨引擎依赖由编译器自动插 flag 对（§7）；完全手动同步不需要（无 `TL_ENABLE_AUTO_SCHEDULE: False`）。

## 3. API 映射设计

### 3.1 公式拆解

§1.6.1 优化后主链（逐式 → §3.2 API）：

```
per row:  (m_run, s_run) = (−inf, 0)                                   # alloc_var + vdup
per chunk c:
  A  载入:    x_ub ← T.copy(x[row, c·BLOCK_N : +BLOCK_N])              # MTE2 单行切片
  B  pass1:   acc = vmax(acc, vcvt(vld(x_ub[0, g·64]))) ×G → m_c = vdupv(vcmax(acc))
  C  max 合并: prev = m_run;  m_run = vmax(m_run, m_c)                  # 源 S3
  D  pass2:   e = vexpdif(vcvt(vld(x_ub[0, g·64])), m_run) ×G;  accs = vadd(accs, e) ×G
  E  sum 合并: s_run = vadd(vmul(s_run, vexpdif(prev, m_run)), vcadd(accs))   # 源 S6
  A′ 尾 chunk: T.copy rem 列切片; B–E 同上（组循环 G_tail + 尾组 vsel(−inf)）
epilogue:  y[r] = vsts(ONEPT, vcvt(vadd(m_run, vln(s_run))))           # lane 0
任务尾:    y[base:+rows] ← T.copy(lse_ub[0:rows])                      # MTE3
```

### 3.2 TileLang API 映射

| 计算步骤 | TileLang Ascend API（`import tilelang.ascend.language as T`） | 参数 | 先例 / 依据 |
|---------|------------------------------------------------|------|------------|
| kernel 入口 | `with T.Kernel(NUM_BLOCKS)` | NUM_BLOCKS=56（§5.5 实查） | single L215；一维强制（ascend-constraints §1 条 1） |
| 任务调度 | `for t in T.Persistent([num_tiles], NUM_BLOCKS, core_id, num_stages=2)` | num_tiles=ceil(M/block_m) | single L219 / SIMT tiled L231 同构 |
| UB 分配 | `T.alloc_shared((1, BLOCK_N), dtype)`；`T.alloc_shared((block_m,), dtype)` | x_ub / lse_ub | single L216–217（形状不同：单行 chunk 粒度） |
| 双缓冲声明 | `T.annotate_buffer_versions({x_ub: 2, lse_ub: 1})` | — | single L218 / SIMT tiled L230 |
| GM→UB chunk 载入 | `T.copy(x[row:row+1, c·BLOCK_N:(c+1)·BLOCK_N], x_ub[0:1, 0:BLOCK_N])` | (1, BLOCK_N) 单行切片（row 为 TIR expr） | SIMT tiled L156–159（rows=1 特例同构） |
| UB→VRF 装载 + 提升 | `T.simd.vld(x_ub[0, g·64], dist="UNPK_B16")` + `T.simd.vcvt(·, "float32", part=0)`；fp32 输入 `T.simd.vld(x_ub[0, g·64])` NORM 直连 | 组宽 64 | cast 回归 L64–80 位级；single L154–162 |
| 行 max 累加 | `acc = T.simd.vmax(acc, cv)` | — | single L163 |
| 跨 lane 全归约 + 广播 | `T.simd.vcmax(acc)` + `T.simd.vdupv(·)` | — | flash core.py L115–118；single L175 |
| max 合并（寄存器） | `m_run = T.simd.vmax(m_run, m_c_brc)` | broadcast 形态 | vmax 同上；alloc_var 承载（下行） |
| running stats 驻留 | `m_run = T.simd.alloc_var("float32")`；`s_run = T.simd.alloc_var("float32")`（行作用域声明、域内读写） | local.var 可变寄存器 | `tilelang/language/allocate.py` L103–168；`test_simdvf_arithmetic.py` L86–110 循环携带先例；**跨块声明+访问 = 本设计新组合（风险 1 + fallback）** |
| 融合稳定化指数 | `e = T.simd.vexpdif(cv, m_run)` | fp32-only 强校验 | flash core.py L130–133；VP-2026-0014 |
| 行 sum 累加 + 全归约 | `accs = T.simd.vadd(accs, e)`；`s_c = T.simd.vcadd(accs)` | lane-0 形态 | flash core.py L134–137；single L190/L205 |
| flash 重缩放合并 | `s_run = T.simd.vadd(T.simd.vmul(s_run, T.simd.vexpdif(prev_brc, m_run)), s_c)` | — | vmul/vadd 单指令；代数 = 源 S6 |
| 尾组掩码 | `cv = T.simd.vsel(cv, neg_inf, T.simd.update_mask(k_tail, 32))`（neg_inf = `T.simd.vdup(−inf, "float32")`） | fp32 域 | deepgemm transform_sf L205/L215/L249；single L173 |
| epilogue | `out = T.simd.vadd(m_run, T.simd.vln(s_run))`；16-bit 输出 `out_h = T.simd.vcvt(out, dtype)` | — | single L206/L210 |
| 单点存储 | `T.simd.vsts(lse_ub[r], out_h, T.simd.pset(16, "PAT_VL1"), "ONEPT_B16")`（fp32：`pset(32,…)` + `"ONEPT_B32"`） | lane 0 | single L208/L211 |
| lse 写回 | `T.copy(lse_ub[0:rows], y[row_base:row_base+rows])` | 静态切片 | SIMT tiled L222 |
| JIT 入口 | `@tilelang.jit(target="ascend", out_idx=[1])` 懒工厂：`_logsumexp_kernel_tiled(M, N, dtype, tile_n)(block_m)(x) -> y` | — | single L115 同款（§4.7 调用面契约） |

### 3.3 计算伪代码

```python
# host 静态化（builder 闭包 `_logsumexp_kernel_tiled(M, N, dtype, tile_n)`，
#   per-(M, N, dtype, tile_n) JIT + lru_cache；`_func(block_m)` 懒编译，`(x) -> y`，§4.7）：
#   elem     = 4 if dtype == "float32" else 2
#   BLOCK_N  = 最大 64 倍数 s.t. BLOCK_N <= tile_n 且 2*BLOCK_N*elem <= 228556 - 8256
#              （clamp 上界：16-bit 55,040 / fp32 27,520；wrapper tile_n <= 32,512
#               → 16-bit 恒不触发；fp32 或显式超大 tile_n 由 clamp 兜底，§4.5）
#   G        = BLOCK_N // 64                     # full chunk 组数
#   num_full = N // BLOCK_N                      # full chunk 数（N < BLOCK_N 时 0）
#   rem      = N % BLOCK_N                       # 尾 chunk 列数（0 = 无尾 chunk）
#   G_tail   = rem // 64;  k_tail = rem % 64     # 尾 chunk 满组数 / 尾组有效元素
#   full_m   = M // block_m;  rem_m = M % block_m
#   num_tiles = full_m + (1 if rem_m > 0 else 0)
#   NUM_BLOCKS = 56                              # = vector_core_num（实查，§5.5）
# 契约防御（builder 内 raise）：M >= 1; N >= 1; tile_n >= 1; block_m >= 1
#   （无 N 上限——任意大 N 是本变体契约；无 BLOCK_N <= N 假设——N < BLOCK_N 走纯尾 chunk 路径）

@T.prim_func
def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M,), dtype)):
    with T.Kernel(NUM_BLOCKS) as core_id:            # 一维 grid，无 threads=（K3）
        x_ub = T.alloc_shared((1, BLOCK_N), dtype)   # UB：单行 chunk 粒度，双版本流水
        lse_ub = T.alloc_shared((block_m,), dtype)
        T.annotate_buffer_versions({x_ub: 2, lse_ub: 1})

        @T.macro
        def row_task(row_base, rows):                # row_base: TIR expr; rows: Python 常量
            for r in range(rows):                    # 行串行（主流 bm=1 → 单迭代退化）
                row = row_base + r
                # —— running stats：行作用域 alloc_var（SimdVF 块外声明，风险 1）——
                m_run = T.simd.alloc_var("float32")  # broadcast 形态（vdupv 产出/更新）
                s_run = T.simd.alloc_var("float32")  # lane-0 形态（vcadd 产出，仅 lane 0 消费）
                with T.SimdVF():                     # init（源 S-init）
                    m_run = T.simd.vdup(-T.infinity(T.float32), "float32")
                    s_run = T.simd.vdup(T.float32(0.0), "float32")
                # —— full chunks：online 流式（源 S1a→S6）——
                for c in T.serial(num_full):
                    T.copy(x[row:row+1, c*BLOCK_N:(c+1)*BLOCK_N],
                           x_ub[0:1, 0:BLOCK_N])     # (1, BLOCK_N) 单行 MTE2
                    with T.SimdVF():                 # per-chunk 两遍 + flash 合并
                        neg_inf = T.simd.vdup(-T.infinity(T.float32), "float32")
                        zero_f  = T.simd.vdup(T.float32(0.0), "float32")
                        acc = T.simd.alloc_var("float32"); acc = neg_inf
                        for g in range(G):           # pass 1: chunk max（延迟归约）
                            cv = vld+vcvt(x_ub[0, g*64])      # fp32 输入: vld NORM 直连
                            acc = T.simd.vmax(acc, cv)
                        m_c_brc = T.simd.vdupv(T.simd.vcmax(acc))
                        prev_brc = m_run             # SSA 快照（源 S3 的 prev_max）
                        m_run = T.simd.vmax(m_run, m_c_brc)
                        accs = T.simd.alloc_var("float32"); accs = zero_f
                        for g in range(G):           # pass 2: sum of exp（UB 重读 + 融合）
                            cv = vld+vcvt(x_ub[0, g*64])
                            e = T.simd.vexpdif(cv, m_run)
                            accs = T.simd.vadd(accs, e)
                        s_c = T.simd.vcadd(accs)     # lane-0 形态
                        s_run = T.simd.vadd(         # flash 重缩放（源 S6）
                            T.simd.vmul(s_run, T.simd.vexpdif(prev_brc, m_run)), s_c)
                # —— 尾 chunk（rem > 0 时 Python 静态实例化；源 S1b 的静态化形态）——
                if rem > 0:
                    T.copy(x[row:row+1, num_full*BLOCK_N:N], x_ub[0:1, 0:rem])
                    with T.SimdVF():
                        neg_inf = ...; zero_f = ...  # 同上（域内常量重物化）
                        if k_tail > 0:
                            tail_mask = T.simd.update_mask(k_tail, 32)
                        acc = ...; acc[初始化 neg_inf]
                        for g in range(G_tail):      # 满组同 full
                            cv = vld+vcvt(x_ub[0, g*64]); acc = T.simd.vmax(acc, cv)
                        if k_tail > 0:               # 尾组：−inf 掩码后进算术
                            cv = vld+vcvt(x_ub[0, G_tail*64])
                            cv = T.simd.vsel(cv, neg_inf, tail_mask)
                            acc = T.simd.vmax(acc, cv)
                        m_c_brc = ...; prev_brc = m_run; m_run = ...   # 同 full
                        accs = ...[初始化 zero_f]
                        for g in range(G_tail):
                            cv = vld+vcvt(x_ub[0, g*64])
                            accs = T.simd.vadd(accs, T.simd.vexpdif(cv, m_run))
                        if k_tail > 0:
                            cv = vld+vcvt(x_ub[0, G_tail*64])
                            cv = T.simd.vsel(cv, neg_inf, tail_mask)
                            accs = T.simd.vadd(accs, T.simd.vexpdif(cv, m_run))
                        s_c = T.simd.vcadd(accs)
                        s_run = ...                  # 同 full flash 重缩放
                # —— 行尾 epilogue（源 S7/S8）——
                with T.SimdVF():
                    out = T.simd.vadd(m_run, T.simd.vln(s_run))
                    if dtype == "float32":
                        one_pt = T.simd.pset(32, "PAT_VL1")
                        T.simd.vsts(lse_ub[r], out, one_pt, "ONEPT_B32")
                    else:
                        one_pt = T.simd.pset(16, "PAT_VL1")
                        out_h = T.simd.vcvt(out, dtype)
                        T.simd.vsts(lse_ub[r], out_h, one_pt, "ONEPT_B16")
            T.copy(lse_ub[0:rows], y[row_base:row_base+rows])   # 任务尾 MTE3（静态切片）

        for t in T.Persistent([num_tiles], NUM_BLOCKS, core_id, num_stages=2):
            if t < full_m:                           # TIR if：M 尾批边界（源 R4）
                row_task(t * block_m, block_m)       # 全块（rows=block_m 静态）
            else:
                row_task(M - rem_m, rem_m)           # 尾块（rows=rem_m 静态）
```

注：`vld+vcvt(x_ub[0, g*64])` 展开为 `T.simd.vcvt(T.simd.vld(x_ub[0, g*64], dist="UNPK_B16"), "float32", part=0)`（fp32 输入为 `T.simd.vld(x_ub[0, g*64])` 单指令，无 vcvt）；`acc = ...[初始化 neg_inf]` 等省略号处 = 与 full chunk 完全相同的语句。宏两次实例化（rows=block_m / rem_m）与 SIMT tiled 兄弟 `sub_batch(row_base, rows)` 同构（L145–250 Stage 3 实证）；`if rem > 0` / `if k_tail > 0` 为 Python 静态分支（trace 期消除，非 TIR 运行时分支）。

### 3.4 API 可行性确认

**逐 API 先例锚点**（指令链整体 = single 姊妹 Stage 3 全量 PASS 的复用 + tiled 契约结构 = SIMT tiled 兄弟 Stage 3 实证的复用；`tilelang/ascend/language/simd.py` 为签名权威）：

| API | 证据 | 状态 |
|-----|------|------|
| vdup / vmax / vadd / vmul / vcvt(part=0) / vld(UNPK_B16) / vsts(ONEPT_B16/B32) / pset(PAT_VL1) / vsel / update_mask / vexpdif / vln / vcmax / vcadd / vdupv | single 姊妹 `_logsumexp_kernel_single.py` L135–211 同族指令链（86 用例 Stage 3 全量 PASS，2026-10-17）；cast 位级回归 `test_tilelang_ascend_simdvf_cast.py` L64–80；flash core.py L94–137（vcmax/vcadd/vexpdif/BRC 实测）；deepgemm transform_sf（vsel/update_mask） | ✅ 实证 |
| `T.copy` (1, BLOCK_N) 单行切片 + 静态 rem 切片 + lse_ub 切片 | SIMT tiled 兄弟 L156–159/L191/L222（tiled 契约同款切片，61 用例 PASS）；`D-REDUCE-SIMTVF-API`：掩码载入与 DMA 紧凑拷贝正交 | ✅ 实证 |
| `T.Persistent` + TIR if 尾批 + `T.macro(row_base: TIR, rows: int)` | single L219–229 + SIMT tiled L231–250（宏带 TIR row_base 实证） | ✅ 实证 |
| `T.simd.alloc_var` 域内声明 + `T.serial` 循环携带 | `test_simdvf_arithmetic.py` L86–110 serial_vreg_accum（NPU PASS） | ✅ 实证（域内形态） |
| **`T.simd.alloc_var` 域外声明 + 多个 per-chunk SimdVF 块内读写（m_run/s_run）** | 机制链：`allocate.py` L103–168（声明无 SimdVF 上下文检查）+ `vf_regions.h` L50–53（SIMD_VF keeps the enclosing scope——域体可访问外围作用域分配的 buffer）+ `insert_sync.cc` L1308（local.var 依赖跳过跨引擎同步插入）；**无单一先例覆盖该组合** | ⚠️ **风险 1**：Stage 3 首编验证；fallback = stats UB 往返（`stats_ub (2,) fp32` + flash core.py L103/L117/L120–126 形态：chunk 末 `vsts(m_ub, ·, ONEPT_B32)` / chunk 首 `vld(m_ub, "BRC_B32")` 广播装载 + `mem_bar("VST_VLD")`，~5 指令/chunk/行 = wl2 0.14%，API 全先例、语义等价） |

### 3.5 技术约束确认

**3.5.1 本项目已知限制检查（ascend-constraints §2 强制检测）**：

| 约束 | 本设计处置 | 状态 |
|------|-----------|------|
| `T.Kernel` 一维、无 `threads=` | `T.Kernel(56)` 一维；GPU `threads` 契约按 K3 舍弃（§0.5）；调用面 `(block_m)` 单参数 | ✅ |
| C0 分形对齐（GEMM 类 16 倍数） | n/a——纯 Vector 归约，无 `T.gemm`/L0 路径 | ✅ |
| UB 容量（248KB 物理；SimdVF 无 32KB 线程上下文扣减） | x_ub 双版本 + lse_ub + 8,256B 预留 ≤ 228,556B（90% 折扣预算，§4.5 表：最大 fp32 clamp 档 99.9%） | ✅ |
| `T.copy` OOB 行为未文档化 | 全部切片静态（full/rem/rows_this 推导自编译期常量；行循环只迭代有效行）——不依赖任何越界保护（§0.6 R4） | ✅ |
| SimdVF 域 SSA 限制 | 循环携带/跨块值一律 alloc_var；域内 SSA（prev_brc、m_c_brc、e、cv）仅域内消费 | ✅ |
| `vexpdif` fp32-only 强校验 | 装载即 vcvt 提升 fp32（fp16/bf16）；fp32 输入直连——全链 fp32 算术 | ✅ |
| flag/event 预算（核内 8 / 跨核 16） | auto_schedule: yes——AutoSchedule 管理，无手动 flag（§7） | ✅ |
| L1/L0A/L0B/L0C | 不使用（纯 UB+VRF 路径） | ✅ |

**3.5.2 参考实现差异说明**：与 GPU 源的结构差异（并行结构 / chunk 粒度 / 双路径 / 驻留 / OOB）全部经 §0.5 四态判定 + §0.6 重设计论证，数值链经 V1–V7 机器验证——非逐 API 语法翻译（§0 迁移决策驱动 §1–§7）。

**3.5.3 本项目同类实现参考**：`examples/logsumexp_simd/_logsumexp_kernel_single/`（指令链 + Persistent 嵌套 + 契约防御 + golden 组织）；`examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp_kernel/_logsumexp_kernel_tiled.py`（tiled 契约的切片/宏/尾批结构）；`examples/pilot/online_softmax/online_softmax.py`（streaming 路径）；`examples/ascend/flash_attention/core.py`（stats 访问形态 + fallback）。

---
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

1. **grid 划分（含 num_blocks 计算与核数依据实查）**：`num_blocks = NUM_BLOCKS = 56`（**一维 `T.Kernel(56)`**，persistent 单波常驻）——56 = `vector_core_num` 本机实查值，grid 维度即核数（一维 N=56 形式，无二级 block 维、无 `threads=`）。行块任务 `num_tiles = ceil(M / block_m)`（wl2 (bm=1): 1024、wl3: 4、(33,33000): 33）由 `T.Persistent([num_tiles], 56, core_id, num_stages=2)` 调度器分配（wl2: 1024/56 ≈ 18.29 任务/核 + 尾波 16 任务）。查询代码与返回值（本设计执行期复测，2026-10-17）：

   ```python
   import torch

   p = torch.npu.get_device_properties()  # name / cube_core_num / vector_core_num
   # → name: 'Ascend950PR_9579', cube_core_num: 28, vector_core_num: 56
   ```

   依据：`CONST-arch-a5`（status=**measured**，2026-10-16 三 pilot 实查转正 + 本任务复测一致；28/56 非 2 幂，「惯例 32 blocks」假设在本机不成立——grid 不做整除性假设，persistent 调度器消化任意任务数）。

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

见 §3.3（完整 kernel 伪代码——Block 级结构 = `T.Kernel(56) > T.Persistent > TIR if > range(rows) > T.serial(num_full) > SimdVF 域（range(G)×2）> 尾分支 > epilogue SimdVF > MTE3`）。

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
## 8. 验证方案

### 8.1 Golden 函数

**主 golden（精度对照基准）**——源仓测试基准 `ref_program` 直移植，torch on NPU（fp32 计算域，保证验证独立性——不复刻 §0.6 的 NPU 算法）：

```python
def golden_logsumexp(x: torch.Tensor) -> torch.Tensor:
    """Row-wise logsumexp, torch on NPU (source test baseline ref_program)."""
    assert x.device.type == "npu" and x.ndim == 2
    return torch.logsumexp(x.float(), dim=-1).to(x.dtype)
```

**源语义模拟器（迁移保真基准）**——tiled 源算法的 fp32 逐要素复刻（BLOCK_N=tile_n chunk 划分 + fp32 中间 + flash 合并序 + 尾列 −inf），用于 golden 与源语义**分歧**的角点（±inf / 全 −inf / 首 chunk 全 −inf——torch 在这些角点与源 tiled 不同，§9.3）：

```python
def _source_lse_tiled_sim(x: torch.Tensor, tile_n: int) -> torch.Tensor:
    """fp32 re-implementation of the SOURCE tiled online algorithm semantics.
    Replicates chunk-serial flash merge order (m_run/s_run recursion, pad=-inf
    tail columns) so source-inherent divergence corners (all -inf row -> NaN,
    +inf row -> NaN, first-chunk-all -inf -> NaN) validate against THIS, not
    against torch.logsumexp (DESIGN 9.3 / verify_equiv V5)."""
    M, N = x.shape
    out = torch.empty(M, dtype=x.dtype)
    for i in range(M):  # 行独立（与 kernel 调度无关）
        m_run, s_run = float("-inf"), 0.0  # fp32 链（源 dtype 语义）
        for c0 in range(0, N, tile_n):
            cols = x[i, c0 : c0 + tile_n].float()
            cols = torch.cat([cols, torch.full((tile_n - cols.numel(),), float("-inf"))])  # 越界列 = -inf（源 masked 语义）
            m_c = cols.max()
            prev = m_run
            m_run = max(m_run, m_c)
            e = torch.exp(cols - m_run)
            s_c = e.sum()
            s_run = s_run * torch.exp(torch.tensor(prev - m_run)) + s_c
        out[i] = torch.tensor(m_run + torch.log(torch.tensor(s_run))).to(x.dtype)
    return out
```

> 两基准分工：randn 数据域（L0 主门槛）golden 与源语义模拟器一致（V3：≤1 ulp）；±inf/NaN 角点（Boundary ④）**只对源语义模拟器**（V4/V5：候选链与源位级一致 = 迁移保真达成；与 golden 的分歧为源固有，§9.3 文档化）。模拟器自身以 verify_equiv.py 的源链（V1/V4/V5 基线式）交叉锚定。

### 8.2 精度标准与分层测试计划（L0 门槛；L1/L2/Boundary 完整套件由 tilelang-op-develop 落地）

**容差**（AGENTS.md 口径）：fp16/bf16 → `torch.testing.assert_close` rtol=1e-2；fp32 → 1e-3（fp32 累加链 102,400 元素归约属 accumulation-heavy，若 L0 抖动超限按 5e-3 档复核并在 RETROSPECTIVE 记录）。

**L0（本设计规定的门槛用例，全过才算实现成功）**：

| # | shape (M, N) | dtype | tile_n（wrapper 复算） | block_m | 覆盖点 |
|---|-------------|-------|------------------------|---------|--------|
| 1 | (1024, 32768) | bf16 | 16384 | 1 | wl2 主 workload；2 full chunk；rem=0；**首用例 = 最简主流形态（bm=1/rem=0，风险 1 首编隔离）** |
| 2 | (4, 102400) | fp16 | 25600 | 1 | wl3；4 full chunk；非 pow2 BLOCK_N |
| 3 | (4, 102400) | bf16 | 25600 | 1 | wl3 dtype 变体 |
| 4 | (33, 33000) | fp16 | 16640 | 1 | 尾 chunk（G_tail=255 + k_tail=40）+ M 非整除尾批 + 非 pow2 |
| 5 | (4, 32768) | fp16 | 16384 | 1 | test 形态；2 full chunk |
| 6 | (33, 32768) | fp16 | 16384 | 1 | M 非整除 + 大 N |
| 7 | (8, 100) | fp32 | 64（显式 config） | 1 | fp32 直连装载 + 纯尾 chunk（N < tile_n → num_full=0 路径）+ k_tail=36 |
| 8 | (1024, 32768) | bf16 | 16384 | 4 | bm>1 行串行（wrapper default_config 遍历形态）；M 整除 |

L0 断言：`torch.testing.assert_close(y, golden(x), rtol=tol, atol=tol)`（randn 域）+ wrapper 落点复算 assert（tile_n/block_m 与 §0.2 表一致）。

**L1/L2/Boundary（Stage 3 完整套件，设计规定覆盖面）**：

- **形状/粒度扫描**：BLOCK_N ∈ {64, 256, 4096, 16384, 16640, 25600}（含非 pow2）× shape ∈ {(3,257), (2,1024000), (57,32768), (1,1), (5,63), (7,65)}——num_full=0 / k_tail∈{0,1,63} / 单 chunk / 63-chunk 长链；block_m ∈ {1,2,4,8,16} × M 整除 / M 非整除 / M<block_m 三态；dtype × {fp16, bf16, fp32}。
- **Boundary（角点行，对 `_source_lse_tiled_sim`）**：18 类特殊行复用 verify_equiv V4/V5 集——全零/全等/单主导/±65504 极值/subnormal ±6e-8/深度下溢 −745/部分 −inf 散布/**全 −inf**/**含 +inf**/**含 NaN**/max-gap 131008/**首 chunk 全 −inf 且后续有限**（→ NaN，tiled 特有）/尾 chunk 全 −inf（→ 中性）/N=1/末元素最大（跨 chunk max 边界）/首元素最大；每类 × {tn=16384/N=32768, tn=16640/N=33000, tn=25600/N=102400} × {fp16, bf16}。
- **golden 分歧回归**（对 golden，预期分歧并 assert 分歧形态）：全 −inf 行 → 源 NaN vs torch −inf；+inf 行 → 源 NaN vs torch +inf；首 chunk 全 −inf → 源 NaN vs torch 有限值（**tiled 特有**，single 变体无此分歧）。
- **L2（鲁棒性）**：lru_cache 复用（同 shape 二次调用位级一致）；连续调用不同 shape；M=1/N=1；显式 config tile_n > clamp 上界（fp32：BLOCK_N=27,520 clamp 路径 + rem=5248/k_tail=0）。

### 8.3 性能验证（Stage 4，设计期仅记录口径）

- 入口：`tilelang.profiler.do_bench(fn, backend="msprof")`（端到端时延唯一权威）；`examples/TileOPs/benchmarks/ops/bench_logsumexp_simd.py`（manifest 驱动，wl2/wl3 为 tiled 覆盖对象）。
- 对照基线：SIMT tiled 兄弟 Stage 4 采纳值（wl2 100.2µs / wl3 17.6µs）——同 workload 家族跨模式参考（非门禁，记录口径）。
- 归因：npusim 周期定位（special_func vs MTE 瓶颈裁决——§9.1 风险 4 的实测标定入口）；设计估算 vs 实测偏差行按 hardware-cost-model.md §3 回填 `perf_opt/opt_log.md`。

## 9. 风险点与注意事项

### 9.1 已知约束（本算子相关，按风险度排序）

| # | 风险 | 影响 | 缓解 / 裁决计划 |
|---|------|------|----------------|
| 1 | **alloc_var 跨 SimdVF 块声明+访问（m_run/s_run）无单一先例**——行作用域声明、init/chunk/epilogue 多块读写；若 VFChecker/buffer 作用域报错则设计主形态受阻 | 首编失败 → 阻塞 | 机制链三证据（`allocate.py` L103–168 声明无域检查 / `vf_regions.h` L50–53 enclosing-scope / `insert_sync.cc` L1308 local.var 跳同步）；Stage 3 首用例（L0 #1 最简形态）首编即触发验证；**fallback 全先例**：stats UB 往返（`stats_ub (2,) fp32` + flash core.py L103/L117/L120–126 形态：chunk 末 `vsts(ONEPT_B32)` / chunk 首 `vld(BRC_B32)` + `mem_bar("VST_VLD")`），语义等价（V1 覆盖该链形态——模拟器按值链验证与驻留位置无关），代价 ~5 指令/chunk/行 = wl2 0.14% 计算项 |
| 2 | **首 chunk 全 −inf → NaN 角点（源固有 tiled 分歧）**——golden（torch）在该角点给有限值，误用 golden 对照会把源语义判为 bug | Boundary ④ 误判 | 已 V5 位级锚定（候选=源=NaN）；Boundary ④ 只对 `_source_lse_tiled_sim`（§8.1 分工）；golden 分歧回归用例显式 assert 分歧形态（§8.2）；§9.3 文档化 |
| 3 | **wl3 M=4 并行度天花板**——行任务粒度下仅 4/56 核活跃（D-2：20.9K cyc/行 vs wl2 满载 122.7K/核均值） | wl3 时延受限于 M | 跨核 N-split + partial 合并已评估弃选（R1 #12：(max,sum) 非 additive + 位级保真失效 + 复杂度；兄弟 Stage 4 `[DESIGN_LIMIT]` 同族记录）；列 Stage 4 观察项——若 wl3 实测成瓶颈且用户接受合并序容差语义，再评估 N-split + 定序合并变体 |
| 4 | **special_func 吞吐常数粒度不确定**——`estimate_latency.h` L58–70「8 ops/cycle」未区分 element vs 向量指令粒度；若为指令级则 D-2 瓶颈反转为 MTE 主导 | 设计期 D-2 归因失准（结构选型不受影响——两种解释下主选均最优，§1.6.0 末段） | Stage 4 npusim 周期归因实测标定；偏差 >2× 时按 hardware-cost-model.md §3 回填偏差行并回写 pattern-library 常数条目 |
| 5 | **block_m>1 时 per-row (1, BLOCK_N) DMA**——行串行使 DMA 描述符数 ∝ rows×chunks（兄弟为 (SUB, BLOCK_N) 整批，描述符 ∝ chunks）；字节数不变 | bm>1 显式 config 下 MTE base 项增长（wl2 bm=4 估算 +3.3K cyc/核 ≈ 2.7%——描述符数与 bm=1 相同的总量级，因任务内行循环本就逐行拷贝） | wrapper 主流落点 bm=1（tiled 路径偏好，§0.2）不受影响；bm>1 为显式 config 边缘路径；Stage 4 若 bm>1 shape 进 bench 再评估批拷贝形态（语义等价、x_ub 需 (bm, BLOCK_N)——容量 ×bm 须重算 §4.5） |
| 6 | **饱和封闭域 N ≤ e^16 ≈ 8.86M**——超出后 `vcvt` sat 钳位与 IEEE 溢出可能分歧（y > 65520 时） | 超大 N 显式 config 的 fp16 输出 | V7 实测 102400/1024000 两档 0 分歧；源经 `T.cast` 共享同行为（迁移保真非缺陷）；§9.3-⑧ 文档化；契约空间（wrapper tile_n ≤ 32,512）实际不可达 |
| 7 | **fp32 / 超大 tile_n clamp 路径的 UB 99.9% 档**——BLOCK_N=27,520 时总占用 228,416/228,556B；AutoSchedule multibuffering 膨胀系数未实测（constants.md 待标定项） | 编译期容量报错 | clamp 公式保证显式算术不超预算；若编译报 UB 溢出（膨胀系数 >1.0 首次实测）→ 降 BLOCK_N（64 的倍数递减，语义中性 V6）并回写 constants.md 膨胀系数条目 |
| 8 | **Python 展开体积**——`range(G)`（≤430 × 2 遍）+ `range(rows)`（block_m）× 任务体宏双实例化；大 block_m 显式 config（>64）展开膨胀 | 编译时间增长 | G 展开与 single 姊妹（≤508 Stage 3 实证）同量级；block_m > 64 非契约主流（wrapper ≤16）——builder 不加人为上限，Stage 3 遇大 block_m 显式 config 再评估 `T.serial` 行循环形态（语义等价） |

### 9.2 常见错误（实现期检查清单）

1. **循环携带 / 跨块值必须 `T.simd.alloc_var`**——SSA 重赋值报 `Immutable variable used outside its defining region`（single 姊妹 Stage 3 教训）；m_run/s_run/acc/accs 全部 alloc_var。
2. **掩码只在 fp32 域**：`vsel(cv, vdup(−inf,"float32"), update_mask(k_tail, 32))`——不在 16-bit 域掩码（NaN/−inf 编码差异）。
3. **`vexpdif` fp32-only**：16-bit 装载先 `vcvt(part=0)` 提升；fp32 输入 `vld` NORM 直连无 vcvt。
4. **尾批用 TIR if**：`if t < full_m`（t 为运行期变量，Python if 会 trace 期求值报错/误消除）；尾 chunk/尾组用 Python if（rem/k_tail 编译期常量）。
5. **ONEPT 存储按 dtype 配对**：16-bit `pset(16,"PAT_VL1")` + `"ONEPT_B16"`；fp32 `pset(32,…)` + `"ONEPT_B32"`。
6. **`T.Kernel(56)` 一维、无 threads=**（TypeError）；`T.annotate_buffer_versions` 在 `T.Persistent` 前。
7. **切片全静态推导**：行/列切片边界全部由编译期常量表达式给出（rem/rem_m/rows_this）——不依赖 `T.copy` OOB 行为（§0.6 R4）。
8. **alloc_var 声明位置**：行作用域（chunk 循环外、SimdVF 块外）——若首编报 buffer 作用域/VFChecker 错误，按风险 1 fallback 切 stats UB 往返（先备份后修改）。
9. **域内常量重物化**：neg_inf/zero_f/one_pt/tail_mask 在每个 SimdVF 块内声明（SSA 不跨域）；不要在域外引用域内 SSA 值。
10. **模拟器交叉锚定**：`_source_lse_tiled_sim` 必须先对 verify_equiv 源链自测（torch CPU 对照）再用于 Boundary——防止模拟器自身 bug 污染保真判定。

### 9.3 特殊场景处理（边界语义文档）

| # | 场景 | 源 tiled 行为（= 本设计行为，迁移保真） | torch（golden）行为 | 依据 |
|---|------|--------------------------------------|---------------------|------|
| ① | 任一元素 NaN | y = NaN（逐元素传播：m_c 或 e 任一 NaN → s_run=NaN 不可恢复） | NaN | V4 位级 + NaN 模式 |
| ② | 全 −inf 行 | **NaN**（每 chunk m_c=−inf → m_run 恒 −inf → 首 chunk `exp(−inf−(−inf))=NaN` 污染） | −inf（**分歧**） | V4；single 姊妹同源分歧 |
| ③ | 含 +inf 行 | **NaN**（inf 位置 `exp(inf−inf)=NaN` 污染） | +inf（**分歧**） | V4 |
| ④ | **首 chunk 全 −inf 且后续 chunk 含有限值** | **NaN**（该 chunk 处理时 m_run 仍 −inf → NaN 污染 s_run，此后 `NaN·0+s_c=NaN` 不可恢复）——**tiled 特有**：single 形态（全行 max 先行）与 torch 均给有限值 9.70（V5 实测） | 有限值（**tiled 特有分歧**） | **V5 位级**（本设计新发现角点：兄弟变体 DESIGN V5 与其测试均未覆盖——首 chunk 覆盖率依赖 tile_n） |
| ⑤ | 尾 chunk 全 −inf（首个含有限值 chunk 之后） | 中性（m_run 有限 → `exp(−inf−m)=0` → s_c=0） | 有限值（一致） | V5 |
| ⑥ | 部分元素 −inf 散布（首个有限 chunk 之后） | 中性（同⑤逐元素级） | 一致 | V4 |
| ⑦ | 跨 chunk max 间隙 131008（±65504 极值对，fp16 域最大间隙） | flash 形式安全（合并式全部 exp 参数 ≤ 0，无溢出——朴素 m_c 基会溢出，`TRAP-NAIVE-ONLINE-MERGE`） | —（fp32 域恒安全） | V4 corner 位级 |
| ⑧ | 输出饱和（y > 65520，需 N > e^16 ≈ 8.86M） | `vcvt` sat 钳位（与源 `T.cast` 同行为）；契约空间内不可达（V7 封闭证明：N=102400 → y=65515.54 < 65520） | — | V7 |
| ⑨ | 退化 shape：N=1 / M=1 / N < BLOCK_N（num_full=0 纯尾路径）/ M < block_m（全尾批） | 正常路径承接（静态分支组合，§5.4） | 一致 | V1 tiny shape + L2 |

## 10. 交付清单

### 10.1 目录结构

```
examples/logsumexp_simd/_logsumexp_kernel_tiled/
├── _logsumexp_kernel_tiled.py    # Stage 3 产出：kernel + golden + 模拟器 + 分层测试
├── DESIGN.md                     # 本设计文档
├── RETROSPECTIVE.md              # 阶段复盘（Stage 1 章节随本设计落盘）
├── verify_equiv.py               # D-1 等价性机器验证（已执行，ALL EQUIV_PASS）
└── history_version/              # 设计分段稿 + 预算复算脚本（design_budget_calc.py）
```

### 10.2 文件清单（Stage 3 实现输入）

- `_logsumexp_kernel_tiled.py`：kernel builder（§3.3 伪代码 → 实现，docstring 声明 §10.3）+ `golden_logsumexp` / `_source_lse_tiled_sim`（§8.1）+ L0/L1/L2/Boundary 测试入口（§8.2，文件组织 mirror single 姊妹：单文件 kernel + golden + `run_L0`/`run_boundary` + `__main__` argparse）。
- 输入依赖：`DESIGN.md`（本文件，冻结基线）、`verify_equiv.py`（等价性锚点）、`history_version/design_budget_calc.py`（容量/复杂度数字复算源）。

### 10.3 命名与声明规范

- 模块 docstring 首部声明（mandatory）：

  ```python
  programming_mode: simd
  auto_schedule: yes
  ```

- 调用面（与 SIMT 兄弟变体一致，GPU threads 按 K3 移除）：`_logsumexp_kernel_tiled(M, N, dtype, tile_n)(block_m)(x) -> y`。
- `import tilelang.ascend.language as T`（Ascend 方言，AGENTS.md API 约定）；NUM_BLOCKS=56 常量注释注明实查来源（§5.5）。

### 10.4 实现顺序建议（Stage 3）

1. **Step 0**：golden + `_source_lse_tiled_sim` 落地，torch CPU 上对 verify_equiv 源链自测（§9.2-10）。
2. **Step 1**：builder 静态化 + L0 #1（(1024,32768) bf16 tn=16384 bm=1，rem=0 最简主流形态）——**首编即触发风险 1 验证**（alloc_var 跨块）；报错则按 fallback 切 stats UB 往返（备份后修改，§9.1-1）。
3. **Step 2**：尾 chunk 路径（L0 #4 (33,33000)：G_tail/k_tail/静态 rem 切片）+ L0 #2/#3（非 pow2 4-chunk）。
4. **Step 3**：M 尾批 + bm>1（L0 #8）+ 纯尾路径（L0 #7 fp32）+ L2 形态扫描。
5. **Step 4**：Boundary 18 类角点行（对模拟器）+ golden 分歧回归；L0 全量门槛。
6. **Step 5**：交 Stage 4（do_bench/msprof + npusim 归因，风险 4 标定）。

### 10.5 本设计引用的外部工件

- 源算子：`/home/z00875089/TileOPs-fork/tileops/kernels/reduction/logsumexp.py`（`_logsumexp_kernel_tiled` L105–194 + wrapper L338–395）+ `_primitives.py`。
- 同族先例：`examples/logsumexp_simd/_logsumexp_kernel_single/`（DESIGN.md / 实现 86 用例 PASS / RETROSPECTIVE）；`examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp_kernel/_logsumexp_kernel_tiled.py` + `_logsumexp_kernel_tiled_DESIGN.md`（SIMT 兄弟，61 用例 PASS + Stage 4）。
- API/结构先例：`examples/ascend/flash_attention/core.py`；`examples/pilot/online_softmax/online_softmax.py`；`examples/ascend/example_gemm.py`；`testing/ascend/language/test_tilelang_ascend_simdvf_cast.py`；`testing/ascend/language/simdvf/test_simdvf_arithmetic.py`；`testing/ascend/language/test_tilelang_ascend_simdvf_integer_predicate.py`；`examples/ascend/deepgemm/kernels/transform_sf.py`。
- manifest/测试：`examples/TileOPs/tileops/manifest/reduction.yaml`（LogSumExpFwdOpSimd L124–172）；`examples/TileOPs/tests/ops/test_logsumexp_simd.py`；`examples/TileOPs/benchmarks/ops/bench_logsumexp_simd.py`。
- pattern-library / 标准：`CONST-capacity-a5` / `CONST-arch-a5` / `CONST-valu-model` / `CONST-mte-bandwidth-model` / `CONST-vrf-capacity-a5` / `CONST-gemm-dim-constraints`；`TRAP-NAIVE-ONLINE-MERGE` / `P-LAYOUT-ROW-RESIDENCY` / `P-LAYOUT-TAIL-STATIC-EXTENT` / `D-LOGSUMEXP-MIGRATION-FIDELITY` / `D-REDUCE-SIMTVF-API`；`VP-2026-0014`；`_shared/standards/core-split-strategy.md` / `algorithm-research.md` / `hardware-cost-model.md` / `negative-claim-evidence.md` / `gate-and-retry.md`。
- 源码机制锚点：`tilelang/language/allocate.py` L103–168；`tilelang/ascend/language/simd.py`；`src/ascend/transform/vf_regions.h` L50–53；`src/ascend/transform/insert_sync.cc` L1308；`src/ascend/transform/ascend_simdvf_lower_parallel.cc` L818–841；`src/tl_templates/ascend/simd_inst.h` L688–710；`src/ascend/transform/estimate_latency.h` L58–70。
