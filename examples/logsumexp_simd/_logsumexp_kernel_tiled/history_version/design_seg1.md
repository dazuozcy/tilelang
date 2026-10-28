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

**精度用例落点**（`examples/TileOPs/tests/ops/test_logsumexp_simd.py`，SIMD 家族专属测试）：`(4, 32768)` fp16/bf16（L75–76 "large-N triggers N-tiling path"）→ (1, 16384)；`(33, 33000)` fp16（L80 "multi-tile, masked loads"）→ (1, 16640)（N_padded=33,280=130×256，130 的 ≤127 最大 256 倍数因子为 65 → 16,640，2 chunks）；`(33, 32768)` fp16（L82 非对齐 M + 大 N）→ (1, 16384)。性能用例 `benchmarks/ops/bench_logsumexp_simd.py`（manifest 驱动，wl2/wl3 为 tiled 覆盖对象）。

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
