# _logsumexp_kernel_single 算子设计文档（GPU TileOPs → Ascend A5 迁移，migration-harness）

> **任务上下文**：迁移 TileOPs `LogSumExpFwdOp` 的单 tile 变体 `_logsumexp_kernel_single`（GPU/CUDA 目标 TileLang → tilelang-fork ascend 后端，A5/950，dav-3510）。源算子三问解读对象 = `examples/TileOPs/tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py` 中的 `_logsumexp_kernel_single`（Stage 0 抽取件，自包含）；wrapper 侧 GPU 上下文（Kernel class / dispatch）= 同目录 `logsumexp.py`。
> **职责边界**：本设计只覆盖 **single 变体**（N 在片上一次装下）。多 tile 变体 `_logsumexp_kernel_tiled`（N 超驻留上限的 online 流式路径）由独立任务设计，不在本文范围。
> **调用面契约（Stage 5 集成对接）**：wrapper `LogSumExpKernel` 经 `_logsumexp_kernel(M, N, dtype, tile_n=0)` 选中本函数后以 `_logsumexp_kernel_single(M, N, dtype)(block_m)(x)` 调用——A5 重实现保持该调用面，`threads` 参数按 K3 适配规则移除（见 §4.7）。

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

**数学语义**（逐行 log-sum-exp，数值稳定形式）：

$$
y_i \;=\; \mathrm{logsumexp}_j(x_{ij}) \;=\; m_i + \log\Big(\sum_{j=0}^{N-1} \exp(x_{ij} - m_i)\Big), \qquad m_i = \max_j x_{ij}
$$

与解析式 $\log\sum_j e^{x_{ij}}$ 数学恒等（减 max 平移不变性），是数值稳定化标准形态（algorithm-candidates.md `ALG-softmax` R1 必选项）。

**语义要素表**（迁移必须原样保持的部分）：

| 要素 | 结论 | 源码证据 |
|------|------|---------|
| 数学映射 | 行级 logsumexp：`y[i] = row_max[i] + log(row_sum[i])`，`row_sum = Σ exp(x − row_max)` | `_log_sum_exp_fwd_kernels.py` L85–94 |
| I/O 契约 | 输入 `x: (M, N)` 行主连续、dtype ∈ {float16, bfloat16, float32}（op 层 `_validate` 允许三者；manifest workload 为 fp16/bf16）；输出 `y: (M,)` 同 dtype | kernel 签名 L58–61；`logsumexp.py` `_validate` L111 |
| 规约语义 | 沿最后一维（N）整行归约；**中间计算全程 fp32**（`x_f32` fragment、"float32" 的 row_max/row_sum），输出 cast 回输入 dtype；累加顺序 = 树形归约（`T.reduce_sum`），并行分组变化只引入 fp32 树深级舍入差 | L65–67、L89–94 |
| dtype 语义 | 输入 dtype → fp32 中间（`T.cast(x[...], "float32")`）→ 输出 dtype（`out_local` 为输入 dtype fragment，`T.copy` 写回） | L74–77、L92–94 |
| 边界语义 | ① **NaN 逐元素传播**：行含 NaN → m=NaN → e=NaN → S=NaN → y=NaN；② **全 −inf 行 → y=NaN**（m=−inf → x−m=NaN——源算法固有行为，与 `torch.logsumexp` 的 −inf **分歧**，见 §9.3）；③ **含 +inf 行 → y=NaN**（inf−inf=NaN 污染；torch 返回 +inf，同上分歧）；④ 部分元素 −inf 的正常行：−inf−m=−inf → exp=0，和中性 ✓；⑤ M 尾块（M % block_m ≠ 0）行 OOB 以 −inf 载入、不写出 | L73–78（`T.And(pid_m*block_m+i < M, j < N)` 掩码）；L85–94 |
| 尾块/非整除 | N 非 256 倍数时 `_needs_pad=True` 走逐元素 masked 载入（列 j<N、行 <M，越界填 −inf）；N 对齐时走 `T.copy` 快路径。列 pad 与行 pad 的 −inf 同时是 max 幂元与 exp 后 sum 幂元（数值中性） | L51–52、L69–83 |
| host 侧语义 | `N_padded = align_up(N, 256)`、`_needs_pad` 判定、`threads` 选择均在 builder 闭包内（JIT per-shape 编译的一部分）；wrapper 层负责 dim 归一化/transpose/contiguous/reshape 到 (M, N) 与输出 reshape——**不属于本 kernel 的契约** | L51–56；`logsumexp.py` L193–207 |

**语义解读完成标准自检**：依据上表可直接写出与源算子数值一致的 golden（§8.1 的 `torch.logsumexp(x.float(), dim=-1).to(x.dtype)`，即源仓 `test_softmax.py` L99 的 `ref_program` 直移植）；±inf 角点的源继承分歧已显式列出（§9.3）。

### 0.2 源算子输入输出（含 single 变体的契约 shape 空间）

**Kernel 签名**：`_logsumexp_kernel_single(M: int, N: int, dtype: str)` → `_func(block_m)` → `(x) → y`（`@tilelang.jit(out_idx=[1])`）。

**single 变体被 wrapper 选中的条件**（`logsumexp.py` `_tile_n_for_block_m`，决定本 kernel 的契约 shape 空间）：

1. `N_padded = align_up(N, 256) ≤ MAX_SINGLE_TILE_COLS = 32512`（列数上限，`_primitives.py` L51）；且
2. `block_m × elem_bytes × N_padded ≤ device_smem_budget()`——A5 侧该预算 = **216KB**（`tileops/device.py` `NPUBackend.shared_memory_budget` 返回 `UB_SIMTVF_BUDGET_BYTES`，与 `CONST-capacity-a5` 的 SimtVF 上限一致）。

`default_config` 在 block_m ∈ {1,2,4,8,16} 中**优先选满足条件的最大值**（single 优于 tiled；caller 亦可用 config 显式覆盖 block_m）。由此得到契约空间：**M ≥ 1 任意（含 M < block_m、M % block_m ≠ 0）；N 满足上两条；dtype ∈ {fp16, bf16, fp32}；block_m ∈ [1, 16] 常见（显式 config 理论上可更大，本设计以行子批机制兜底任意 block_m，§5.2）**。

**manifest workload 落点**（`examples/TileOPs/tileops/manifest/reduction.yaml` L93–99，归约到 (M, N) 后）：

| workload | 归约后 (M, N) | dtype | block_m（wrapper 启发式） | 落入变体 |
|----------|--------------|-------|--------------------------|---------|
| attn-weights-4k `[32,32,4096]` dim=-1 | (1024, 4096) | fp16 / bf16 | 16（16×2×4096=128KB ≤ 216KB） | **single（本设计）** |
| attn-weights-32k `[32,32,32768]` | (1024, 32768) | bf16 | —（N_padded=32768 > 32512） | tiled（另任务） |
| lm-head-logits `[4,102400]` | (4, 102400) | fp16 / bf16 | —（N_padded > 32512） | tiled（另任务） |
| 3d-multidim-reduce `[4,128,4096]` dim=[0,2] | (128, 16384) | fp16 | 4（4×2×16384=128KB ≤ 216KB；8 行超预算） | **single（本设计）** |

精度用例（`tests/ops/test_softmax.py`）中除 `(4,32768)`、`(33,33000)`、`(33,32768)`（N_padded > 32512 或 tiled 路径）外全部走 single，含 fp32 三档 dtype、非 pow2 N（300）、M 尾块（33 % 16 ≠ 0）、1D 输入（M=1）、多维 dim 归约（M=1, N=32）、非连续输入（op 层 contiguous 化后进 kernel）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解表**（源 GPU kernel 每条计算语句归入一步，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式 | 源码行 |
|------|------|------|------|-------------|--------|
| S0 | `N_padded = align_up(N, 256)`；`_needs_pad = N_padded != N` | N | 编译期常量 | 边界处理选择 | L51–52 |
| S1a | masked 载入：`x_f32[i,j] = if(行<M ∧ j<N, cast(x), −inf)` | x (GM) | x_f32 fragment (block_m, N_padded) fp32 | 数据就位 + 边界中性化 | L73–78 |
| S1b | 对齐快路径：`T.copy(x → shared_buf → x_local)` 再 cast fp32 | x (GM) | x_f32 | 同上（N 对齐时） | L80–83 |
| S2 | 行最大：`T.fill(row_max, −inf)` + `T.reduce_max(x_f32, row_max, dim=1, clear=False)` | x_f32 | row_max (block_m,) fp32 | m_i | L85–86 |
| S3 | 稳定化指数：`x_f32[i,j] = exp(x_f32[i,j] − row_max[i])`（**原位覆写**） | x_f32, row_max | x_f32（语义上为 e_ij） | e_ij | L88–89 |
| S4 | 行求和：`T.reduce_sum(x_f32, row_sum, dim=1)` | x_f32(e) | row_sum (block_m,) fp32 | S_i | L90 |
| S5 | 输出：`out_local[i] = row_max[i] + log(row_sum[i])`（fp32 算式，fragment 为输入 dtype） | row_max, row_sum | out_local (block_m,) | y_i | L92–94 |
| S6 | 写回：`T.copy(out_local, y[pid_m*block_m])` | out_local | y (GM) (M,) | 输出 | L96 |

**数据流图**（源 GPU 硬件视角）：

```
GM[x (M,N) dtype] --T.copy / masked load--> SMEM[shared_buf (block_m, N_padded)]   (仅对齐路径；masked 路径直达寄存器)
  --> register[x_local → x_f32 (block_m, N_padded) fp32]
  --> reduce_max(dim=1) --> row_max (block_m,) fp32
  --> 原位 exp(x − m) --> x_f32 ≡ e
  --> reduce_sum(dim=1) --> row_sum (block_m,) fp32
  --> out_local = cast(row_max + log(row_sum), dtype) (block_m,) 
  --T.copy--> GM[y (M,)]
```

**循环与并行结构**（源）：grid = `T.ceildiv(M, block_m)` 个 block（`T.Kernel(grid, threads=threads)`，**二维参数 + threads**——A5 不支持，见 §0.5），每 block 处理 block_m 行；block 内 `T.Parallel(block_m, N_padded)` 线程级并行（threads=128，行内 64 元素/线程量级）；无块间通信（行独立）。

**host 侧逻辑**：`align_up` 常量折叠（编译期）；`_neg_inf` 字面量；`threads` 由 wrapper 传入（GPU 侧契约）；JIT per-(M, N, dtype, block_m, threads) 编译 + `lru_cache`。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 代码特征 | 目的 | 依赖的源硬件特性 | 层级 |
|---|----------|---------|------|-----------------|------|
| 1 | 单 tile 全驻留（SMEM tiling 特例） | `shared_buf (block_m, N_padded)` 一次装下整行块 | x 只读一遍 GM，max/exp/sum 全在片上 | GPU SMEM 容量 | 算法层（驻留策略） |
| 2 | N 256 元素对齐快慢双路径 | `_needs_pad` 分支：对齐 `T.copy` / 非对齐 masked 载入 | 对齐时走 bulk copy 免逐元素掩码 | GPU `T.copy` shared 256 元素对齐约束 | 优化层 |
| 3 | 减 max 稳定化 + fp32 中间精度 | `exp(x − m)`、`x_f32` fragment | 消 fp16 溢出；累加精度 | 无（纯数学） | 算法层 |
| 4 | exp 原位覆写（寄存器复用） | S3 直接写回 `x_f32` | 省 (block_m, N_padded) fp32 第二 fragment 寄存器 | GPU 寄存器堆 | 优化层 |
| 5 | 树形归约 `T.reduce_max/reduce_sum` | dim=1 fragment 归约 | 行统计量低延迟 | warp shuffle 树（TVM lower） | 优化层 |
| 6 | 行批 block_m（多行/block） | `block_m` 参数化 | 摊薄 block 启动与调度开销 | GPU SM 调度 | 优化层 |
| 7 | 非持久 grid（`ceildiv(M, block_m)` blocks） | `T.Kernel(grid, threads)` | 行块到 block 的直接映射 | GPU 硬件调度器 | 优化层 |

**未识别更优手段的风险自检**：源码无 warp shuffle 手写、无 swizzle、无 cp.async/Pipelined、无 persistent——识别已覆盖全部显式结构；`threads=128` 的行内并行组织归入 #5/#7 的实现细节。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（三层模型：语义层/算法层/优化层；四态处置：保留/等价替换/重新设计/舍弃。判定输入含 §1.6.0 调研结论——源算法与调研候选同台比较。）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（行 logsumexp、fp32 中间、NaN/±inf 边界行为） | 语义 | — | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| 减 max 稳定化（#3） | 算法 | 无 | — | **保留** | 纯数学（`ALG-softmax` R1 必选项）；调研确认 e-base 直写即 A5 最优形态（`CONST-valu-model` 注记：exp/exp2 同档） |
| 单 tile 全驻留（#1） | 算法 | 无（容量数值硬件相关） | 有（UB 216KB〔SimtVF〕重算预算，`CONST-capacity-a5`） | **保留（驻留层重算）** | 契约空间由 wrapper 按 216KB 判定（§0.2）；驻留层 SMEM→UB 等价替换见下行；行子批机制保持任意 block_m 下 UB/VRF 双预算成立（§0.6 R2） |
| exp 原位覆写（#4） | 优化 | GPU 寄存器堆 | 有（SimtVF `T.alloc_fragment` 寄存器切片，语义一致） | **等价替换** | `examples/ascend/example_rmsnorm.py` L55–57（fragment 向量化装载 + 原位运算）；等价性机器验证 V2 位级一致（§1.6.1） |
| 树形归约 `T.reduce_max/reduce_sum`（#5） | 优化 | warp shuffle 树 | 有（`T.reduce_*` 自动降级 AscendAllReduce 跨线程树归约，fp32 硬件直连） | **等价替换** | pattern-library `D-REDUCE-SIMTVF-API`（measured，get_kernel_source 实见 `tl::AscendAllReduce<MaxOp/SumOp>`）；`examples/ascend/example_per_token_cast_to_fp8.py` L73 SimtVF 内 `T.reduce_max(dim=1)` 同构；`testing/ascend/language/test_tilelang_ascend_reduce.py` 回归 |
| N 256 对齐快慢双路径（#2） | 优化 | GPU `T.copy`→SMEM 256 元素对齐 | **无此约束**（A5 GM→UB DMA 字节级行长；fragment 才需 pow2） | **重新设计** | → §0.6 R2：统一为「UB 紧凑 (SUB, N) + fragment pow2 masked（pad=−inf）」单路径——D-REDUCE-SIMTVF-API 实证「掩码载入与 DMA 紧凑拷贝正交」；`examples/ascend/example_rmsnorm.py` L42 注释先例（"Only fragments need power-of-two padding; DMA tiles stay compact"） |
| `T.copy` SMEM staging（x→shared→local→fp32，#1/#2 实现细节） | 优化 | GPU SMEM 两跳 | 有（`T.copy` GM→UB 单跳 + fragment 载入） | **等价替换** | `examples/ascend/example_per_token_cast_to_fp8.py` L50（GM→UB 2D 区域拷贝）+ L69（`T.copy(ub, frag)`）；A5 侧 x_ub 即驻留层，fragment 直接从 UB masked 载入 |
| block×thread 两级并行（`T.Kernel(grid, threads=threads)`，#7） | 优化 | CUDA 线程模型 | **无直接等价**（`T.Kernel` 一维、无 `threads=` 参数——TypeError） | **重新设计** | → §0.6 R1：一维 `T.Kernel(NUM_BLOCKS)` + `T.SimtVF(threads=)` 线程域 + persistent 行块调度（ascend-constraints §1 第 1 条；tilelang-ascend skill §2.1/§2.2）；K3 适配：wrapper 调用面移除 `threads` |
| 非持久 grid（ceildiv(M, block_m) blocks，#6/#7） | 优化 | GPU 硬件调度器 | 有更契合形态（persistent + 多波摊薄） | **重新设计** | → §0.6 R3：`T.Persistent([num_tiles], 56, core_id, num_stages=2)` 行块持久调度 + 行子批双缓冲流水（`examples/ascend/example_per_token_cast_to_fp8.py` L45 / `examples/pilot/online_softmax/online_softmax.py` L60 先例）；#6 的行批摊薄意图由 persistent 域 + 行子批承接 |
| M 尾块越界行为（对齐路径 `T.copy` 无行掩码，S1b/S6） | 实现 | GPU `T.copy` 隐式 OOB 保护（源仓 CI 通过的既成事实） | **未文档化**（A5 DMA 越界行为无保障） | **重新设计** | → §0.6 R4：静态 `full/rem` 分解 + TIR if 尾块分支 + 静态切片 copy + 行谓词 masked（不依赖 A5 `T.copy` 的越界行为；`examples/ascend/example_gemm.py` L47 循环变量 if 先例；`pad_value=` 仅列向 padding 不适用——tilelang-ascend skill §6.1） |
| host `threads` 契约 | 环境 | CUDA | 无（K3：A5 移除） | **舍弃（形态变更）** | wrapper 侧 threads 已在 Stage 0 适配中移除（`logsumexp.py` 模块头 L26–28）；kernel 内 `threads = min(1024, SUB×TILE)` 自推导（§5.2）。**意图承接**：GPU threads 表达的行内并行度由 SimtVF 线程域承接 |

**判定统计**：保留 3 项（语义 / 稳定化 / 驻留策略）+ 等价替换 4 项（原位覆写 / 树形归约 / SMEM staging / —— 合并计 3 项实现层 + host 环境项）+ 重新设计 4 项（R1 并行结构、R2 驻留布局与双路径统一、R3 调度、R4 M 尾块）+ 舍弃 1 项（host threads 契约，K3 形态变更，意图由 SimtVF 承接）。**无静默丢弃**：每个源优化意图均有承接结构或舍弃理由。

### 0.6 NPU 算法重设计

**重设计项 R1: 并行结构（block×thread 两级并行 → 一维 grid + SimtVF + persistent 行块调度）**

- **源方案**：`T.Kernel(ceildiv(M, block_m), threads=threads)`——block_m 行/block × threads 线程（行内元素并行），非持久 grid。
- **NPU 新算法**：一维 `T.Kernel(NUM_BLOCKS)`，`NUM_BLOCKS = 56`（= `vector_core_num` 实查值，见 §5.5）；`for t in T.Persistent([num_tiles], 56, core_id, num_stages=2)` 行块持久调度（num_tiles = ceildiv(M, block_m)，多波由调度器分配）；元素并行在 `with T.SimtVF(threads=min(1024, SUB·TILE))` 线程域内以 `T.Parallel(SUB, TILE)` 表达；行内归约 `T.reduce_max/reduce_sum(dim=1, clear=True)`（AscendAllReduce 硬件树）。
- **语义保持论证**：行间完全独立 → 行块/行子批大小与行到核的映射变化不影响数值；行内归约 warp shuffle 树 → asc_reduce 硬件树，fp32 累加分组变化引入树深级舍入差（≤13×2⁻²⁴ 相对量级，远小于 fp16/bf16 输出容差 1e-2 与 fp32 容差 1e-3——verify_equiv.py V3 实测 max_rel 9.5e-8，violation 0）；NaN/±inf 逐元素传播语义不变（V3b/V4 NaN 模式一致）。
- **依据**：`examples/pilot/online_softmax/online_softmax.py` L57–63（同族 row-reduction 完整同构形态：T.Kernel(56) + T.Persistent + SimtVF + reduce_max/reduce_sum）；`examples/ascend/example_per_token_cast_to_fp8.py` L44–51；tilelang-ascend skill §2.1/§2.2；`CONST-arch-a5`（measured，本机实查 28/56）。

**重设计项 R2: 驻留布局与双路径统一（N_padded 对齐双路径 → UB 紧凑 + fragment pow2 masked 单路径 + 行子批）**

- **源方案**：`shared_buf/x_local (block_m, N_padded)`，N_padded = align_up(N, 256)（GPU `T.copy`→SMEM 对齐约束）；对齐走 bulk `T.copy`、非对齐走逐元素 masked 载入（双路径）。
- **NPU 新算法**：① 驻留层 `x_ub (SUB, N)` **紧凑**（不搬 padding 列，dtype = 输入 dtype，双版本流水）；② 计算层 `x_frag (SUB, TILE)` fp32，`TILE = next_pow2(N)`，masked 载入 `if_then_else(行∧列谓词, cast(x_ub), −inf)`——单路径统一（对齐与否同构）；③ **行子批 SUB**：`SUB = 满足以下约束的最大 2 的幂：2·SUB·N·elem ≤ 194.4KB ∧ SUB·TILE·4B ≤ 128KB ∧ SUB ≤ block_m`，block_m 行批拆为 `ceildiv(block_m, SUB)` 个子批串行（`T.serial`）——同时锁定 UB 双版本预算与 VRF 舒适水位，使**任意契约 block_m**（含 wrapper 启发式的 16 与显式 config 更大值）均安全；④ exp 原位覆写保持源结构（单 fragment）。
- **语义保持论证**：pad 列从「N_padded − N 个 −inf」变为「TILE − N 个 −inf」——−inf 同为 max 幂元与 exp 后 sum 幂元，数学恒等；verify_equiv.py **V1 全 shape fp64/fp32 位级一致（bitwise=True, ulp=0）**；V4（源链 vs A5 链 fp16 输出，含特殊行）位级一致 + NaN 模式一致。行子批不改变行独立性与行内公式。
- **收益**：非对齐 N（如 300 → N_padded 512）GM 读与 UB 驻留 −41%（512→300 列）；对齐 N 驻留不变但消除双路径维护；VRF 需求从 block_m·TILE·4B（wl1 = 256KB 贴 `CONST-vrf-capacity-a5` 假设上限）降到 SUB·TILE·4B ≤ 128KB（≤32 reg/thread x_frag，spill 风险消除）。
- **依据**：pattern-library `D-REDUCE-SIMTVF-API`（measured：「掩码载入（pad −inf 幂元）与 DMA 紧凑拷贝正交」）；`example_rmsnorm.py` L42–57（紧凑 UB + pow2 fragment masked 同构）；`CONST-capacity-a5` / `CONST-vrf-capacity-a5`；调研结论 §1.6.0（R3/R4 驻留候选流量与 special 双占优）。

**重设计项 R3: 调度（非持久 grid → persistent + 行子批双缓冲流水 + AutoSchedule）**

- **源方案**：每 block 一次性处理 block_m 行后退出（无块间流水概念；GPU 硬件调度器消化 launch）。
- **NPU 新算法**：外层 `T.Persistent([num_tiles], 56, core_id, num_stages=2)`；核内子批循环 `T.serial(num_sub)`，`x_ub` 双版本（`T.annotate_buffer_versions({x_ub: 2})`）——子批 s+1 的 MTE2 预取与子批 s 的 VALU 计算重叠；MTE2→VALU→MTE3 跨引擎依赖由 AutoSchedule 自动插 flag 对（§7）。
- **语义保持论证**：纯调度变换（buffer 版本交替 + 同步边），不改任何数值路径。
- **依据**：`examples/pilot/online_softmax/online_softmax.py` L58–62（`T.annotate_buffer_versions({x_ub: 2})` + `T.Persistent(num_stages=2)` 三件套）；`example_per_token_cast_to_fp8.py` L45–49；tilelang-ascend skill §7。
- **意图承接**：源 #6 行批摊薄 → persistent 域行块 + 行子批 DMA 合并（每子批单次 2D 连续拷贝 SUB×N）；源 #7 grid 映射 → persistent 多波调度。

**重设计项 R4: M 尾块（隐式 OOB 依赖 → 静态 full/rem 分解 + TIR if + 静态切片）**

- **源方案**：对齐路径 `T.copy(x[pid_m·block_m, 0], shared_buf)` 与 `T.copy(out_local, y[pid_m·block_m])` 在尾 tile（M % block_m ≠ 0）时依赖 GPU `T.copy` 的越界保护行为（未文档化）。
- **NPU 新算法**：host 侧静态分解 `full = M // block_m`、`rem = M % block_m`、`num_tiles = full + (1 if rem > 0 else 0)`（编译期常量，per-shape JIT）；persistent 循环体内 TIR if：`if t < full:` 全块路径（子批均 SUB 行，切片 `[base+s·SUB : +SUB]`）；`else:` 尾块路径（满子批 `T.serial(rem // SUB)` + Python 级静态 `if rem % SUB > 0:` 末子批，切片 `[M − rem_sub : M]`，fragment 载入行谓词 `r < rem_sub` 等价 `row_base + r < M`；输出 copy 切片 `lse_ub[0:rows] → y[row_base : row_base+rows]` 静态）。
- **语义保持论证**：尾块有效行的公式与全块完全一致（行谓词只作用于越界行，越界行不写出——源语义「行 OOB 以 −inf 载入、不写出」保持）；不依赖任何 `T.copy` 越界行为。
- **依据**：TIR if（循环变量条件）——`examples/ascend/example_gemm.py` L43–56 `aswt_swizzle` macro 内 `if row_idx < MAIN_ROW:`（`T.Persistent` 循环体内调用，A5 编译运行通过）；Python 级常量 if——`examples/pilot/online_softmax/online_softmax.py` L141 `if rem > 0:`（prim_func 体内闭包常量分支）；静态切片 copy——online_softmax L142（`X[base:base+BM, col:N]` 静态 rem 切片）；fragment→UB→GM 两段写——`example_per_token_cast_to_fp8.py` L81–83（`T.copy(y_s_local, y_s_ub)` + `T.copy(y_s_ub, scales[...])`）、`example_rmsnorm.py` L82。
- **fallback 预案**（若 AutoSchedule 对「Persistent + TIR if + 子批 serial + SimtVF」组合的调度出现问题，该组合无直接先例——见 §9.1 风险 3）：拆双 persistent 域（全块域 [full] + 尾块域 [1]，`core_id` 谓词单核处理尾块），语义等价、API 全先例。

### 0.7 标杆实现

- **源算子路径**：`examples/TileOPs/tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py`（`_logsumexp_kernel_single`，语义/算法/优化手段解读的唯一来源）。
- **源仓参考实现/测试基准**：`torch.logsumexp(x.float(), dim).to(x.dtype)`（源仓 `tests/ops/test_softmax.py` L99 `ref_program`，fp32 计算域 golden）——§8.1 golden 直接采用（torch on NPU），保证验证独立性（不复刻 §0.6 NPU 算法）。
- **wrapper 对接面**：`examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp.py`（`_logsumexp_kernel` dispatch L63–68、`LogSumExpKernel.forward` L235–251、`default_config` L200–233）。

## 1. 概述

### 1.1 算子名称

`_logsumexp_kernel_single`（迁移自 GPU TileOPs `tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py`；算子族 `LogSumExpFwdOp` 的单 tile 变体）

### 1.2 功能描述

单 tile 行级 log-sum-exp：对 (M, N) 输入沿最后一维做数值稳定的 logsumexp 归约，输出 (M,)。契约 shape 空间 = N 一次驻留片上（§0.2：N_padded ≤ 32512 且 block_m×elem×N_padded ≤ 216KB）；N 超出该空间的 online 多 tile 路径由 `_logsumexp_kernel_tiled`（另任务）承接。支持 fp16/bf16/fp32 输入（fp32 中间计算）、任意 M（含尾块）、非 pow2 N（masked 载入）。

### 1.3 数学公式

$$
y_i \;=\; m_i + \log\Big(\sum_{j} \exp(x_{ij} - m_i)\Big), \qquad m_i = \max_j x_{ij}, \qquad i \in [0, M)
$$

（与源式完全一致——源已是 e-base 减 max 稳定化形态；等价性机器验证见 §1.6.1。）

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

**主路径（契约空间全域）——UB/VRF 行子批驻留三段式**（§0.6 R1/R2/R3/R4 落地）：

1. host 静态化（per-shape JIT 闭包）：`TILE = next_pow2(N)`；`SUB = 满足以下约束的最大 2 的幂：2·SUB·N·elem ≤ 194.4KB ∧ SUB·TILE·4B ≤ 128KB ∧ SUB ≤ block_m`；`num_sub = ceildiv(block_m, SUB)`；`full = M // block_m`，`rem = M % block_m`，`num_tiles = full + (rem>0)`；`threads = min(1024, SUB·TILE)`；`NUM_BLOCKS = 56`。
2. `T.copy` 将子批行块 `X[base+s·SUB : +SUB, 0:N]` GM→UB（x_ub (SUB, N) 紧凑、双版本流水；尾块路径切片静态化）。
3. SimtVF 线程域内：masked 载入 `x_frag (SUB, TILE) fp32`（行∧列谓词，pad = −inf）。
4. `T.reduce_max(x_frag, m_frag, dim=1, clear=True)` 行最大（fp32 精确）。
5. `x_frag = T.exp(x_frag − m_frag)`（fp32，**原位覆写**，寄存器驻留）。
6. `T.reduce_sum(x_frag, s_frag, dim=1, clear=True)` 行和（fp32 硬件树）。
7. `out_frag[r] = T.cast(m_frag[r] + T.log(s_frag[r]), dtype)`（每行 1 次 log）；`T.copy(out_frag, lse_ub)`（fragment→UB）。
8. `T.copy(lse_ub[0:rows], y[row_base : row_base+rows])` UB→GM（静态切片）。

**与源算法的结构差异**（逐条注明来源）：① 驻留层 N_padded 256 对齐 → UB 紧凑 N + fragment pow2 masked 单路径（§0.6 R2 / §1.6.1 O2）；② 双路径（对齐 copy / 非对齐 masked）→ 统一 masked 单路径（§0.6 R2）；③ 非持久 grid × threads=128 → persistent 56 核 + SimtVF(threads=min(1024, SUB·TILE))（§0.6 R1，K3）；④ block_m 整批 → 行子批 SUB 双缓冲流水（§0.6 R2/R3——UB 双版本 + VRF 舒适水位）；⑤ M 尾块隐式 OOB → 静态 full/rem 分解 + TIR if + 静态切片（§0.6 R4）；⑥ `T.reduce_*` warp 树 → AscendAllReduce 硬件树（§0.5 等价替换）。公式层与源恒等（e-base、减 max、log 每行 1 次——源已是调研确认的 A5 最优形态，§1.6.0 R1 候选 2 弃选证据）。

### 1.5 数据流图

```
GM[X (M,N) dtype]
  --T.copy (MTE2, 2D 行子批连续区域, x_ub 双版本)--> UB[x_ub (SUB,N) dtype]
  --SimtVF masked T.Parallel (行∧列谓词, pad=-inf, cast fp32)--> VRF[x_frag (SUB,TILE) fp32]
  → VRF reduce_max (AscendAllReduce 硬件树) → m_frag (SUB,) fp32
  → VRF x_frag 原位 = exp(x_frag − m_frag)            [寄存器驻留，单 fragment]
  → VRF reduce_sum → s_frag (SUB,) fp32
  → VRF out_frag = cast(m_frag + log(s_frag), dtype) (SUB,)
  --T.copy (fragment→UB)--> UB[lse_ub (SUB,) dtype]
  --T.copy (MTE3, 1D 静态切片)--> GM[y (M,)]
```

纯 Vector 算子：无 L1/L0A/L0B/L0C 参与段（无 Cube/`T.gemm`），NT GEMM 布局不适用；无中间量落 GM；X 单遍 GM 读、y 单遍 GM 写。

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先调研（1.6.0，调研四问——输入公式/源算法只是候选之一，迁移任务在源算子解读后、耦合性判定前完成）；再做数学等价的公式优化（1.6.1）；再做循环/标量计算的向量化替代（1.6.2）；最后决策向量化轴与核内布局（1.6.3）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。1.6.1–1.6.3 的分析对象 = §1.6.0 调研选定、经 §0.6 重设计落地的 NPU 算法，不止步于源公式直译。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：同一数学语义（行 logsumexp）的算法族。调研深度：**完整调研**（规约/统计类算子）。执行位置：Phase M0 之后、Phase M1 之前（调研结论是 §0.5/§0.6 的判定输入）。
> 信息源：`.agents/skills/tilelang-op-design/references/algorithm-candidates.md`（**ALG-softmax** 命中行〔family 含 logsumexp〕+ **ALG-reduction** / **ALG-layernorm** 的 row-reduction 形态）、pattern-library（`D-REDUCE-SIMTVF-API` measured / `TRAP-NAIVE-ONLINE-MERGE` measured / `D-ELEM-SIMTVF-AUTOVEC` measured / `CONST-valu-model` / `CONST-mte-bandwidth-model` / `CONST-capacity-a5` / `CONST-vrf-capacity-a5`）、`examples/ascend/` 同类实现（example_rmsnorm / example_per_token_cast_to_fp8 / flash_attention）、`examples/pilot/online_softmax/`（同族 row-reduction 迁移正例，CASE-pilot-online-softmax）、源算法（§0.3/§0.4——**源仓的 tiled 变体 `_logsumexp_kernel_tiled` 已用 online softmax 递推，是「存在在线算法」的直接证据**）、`kb_search "logsumexp reduction softmax max sum"`（命中 ALG-softmax / D-REDUCE-SIMTVF-API / TRAP-NAIVE-ONLINE-MERGE / ALG-reduction / VP-2026-0017〔pending，内容与本调研结论一致：exp2 换底 A5 无益 + 行块驻留优先判定〕）。互联网检索：未使用——本地源（候选表命中行 + 三个 pilot/例证 + 源仓自身双变体结构 + 结构判据）已覆盖 logsumexp 族全部已知变体（两遍驻留 / 三遍朴素 / online 流式 / Cube 归并），无新增候选空间。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线 = 源算法（驻留三段式，N_padded 对齐） | `m=max(x); e=exp(x−m); y=m+log(Σe)`，SMEM 驻留 N_padded 列 | — | — | ✅（基线） |
| 1 | 减 max 稳定化 | 同上式（平移不变） | 数学恒等 | 消 fp16 溢出（必选） | 基线已含 |
| 2 | exp→exp2·log2e 换底 | `e^t = 2^(t·log2e)` | 数学恒等 | GPU SFU 快路径 | ❌ **A5 上弃选**：`CONST-valu-model` special_func 档位成员注记（measured 补注，2026-10-16 三 pilot 实证）：exp/exp2/log2 同入 8 ops/cycle 档（`src/ascend/codegen/intrin_rule_ascend.cc` L34–49 均注册、`estimate_latency.cc` L1267 同列）——换底无单位收益、反多 1 乘/elem 与 scale 字面量截断误差；源本身即 e-base，无需「去换底」反向优化（与 online_softmax 任务的 O3 不同源态） |
| 3 | log 域拆分 / log2 换底输出 | `y = (m·log2e + log2(S))/log2e` 类 | 数学恒等 | 无（log 每行仅 1 次，非热点） | ❌ 每行 1 次 special（1/N 占比），换底反而增加 2 次每行乘法；CONST-valu-model 同档证据同上 |
| 4 | 紧凑驻留 + pow2 masked（布局级化简） | N_padded 列搬运 → N 列紧凑 + TILE pow2 fragment masked（−inf 幂元） | 数学恒等（pad 幂元中性） | 非对齐 N：GM 读与 UB 驻留 −(N_padded−N)/N_padded；对齐 N：消除双路径分支 | ✅（O2，A5 结构差异） |
| 5 | e 原位覆写（寄存器复用） | 分离 e_frag → 原位写回 x_frag | 数值同链（同舍入路径） | VRF −50%（单 (SUB,TILE) fp32 fragment） | ✅（O3，源已含此结构，A5 承接确认） |
| 6 | fp16-e 精简链（fp16 输入限定） | x_frag fp16 载入（省 cast）、exp fp16（h 前缀 intrinsic）、S fp16 树形和 | 容差内等价（预估 max_rel ~2e-3–6e-3） | VRF 再 −50%；special 若 fp16 双倍吞吐（**未文档化假设**）则计算项 −50% | ✅ 纳入 §1.6.4 实验裁决（不进 R3 主对比：精度依赖 dtype 限定，不构成全局结构候选） |
| 7 | 行内单遍 online（running max + 分块校正） | K 个行内块流式合并 `S ← S·exp(m_old−m_new) + S_c·exp(m_c−m_new)` | 容差内等价（多次重舍入） | −1 遍 cmp 扫描/行（fragment 域），+2K special/行 | ❌ **弃选**：驻留场景数据已在 VRF，"遍数"是向量 pass 数而非 GM 读——收益 ~1 cmp @51 档（0.02 cycle/elem）vs 每 extra special 0.125 cycle/elem 的主项（K=2 时净亏）；引入行内块序依赖削弱流水（同 online_softmax R1 候选 6 的弃选逻辑）；且 online 合并式须 flash 形式 + 哨兵钳位防角点发散（`TRAP-NAIVE-ONLINE-MERGE` measured：朴素 m_c 基合并 fp16 违反率 13.6%）——复杂度代价明确 |
| 8 | Cube GEMV 归并（sum = e·1 向量内积） | exp 后 `T.gemm(e_l1, ones_l1, c_l0c)` | 数学等价 | Cube MAC 替代 Vector 归约 | ❌ **弃选**：exp 主导段仍在 Vector（逐元素 special），Cube 只接管 sum——多出 UB→L1 ND2NZ 转换 + L0C→UB dual_copy 往返（`CONST-mte-bandwidth-model` ND2NZ 62 cycles base 非免费）+ mixed 核复杂度；而 `T.reduce_sum` 已是 AscendAllReduce 硬件直连（`D-REDUCE-SIMTVF-API` measured），归约非瓶颈（R3 表：计算项 special 主导）。**违反流量最小化** |
| 9 | 两 kernel 分离（先全行 max 落 GM，再 exp+sum） | kernel1 max → GM；kernel2 exp+sum | 数学等价 | 无（每阶段行内依赖不变） | ❌ **弃选**：X 读 2 遍 GM（流量 +100%）+ 跨 kernel 同步；驻留可行时纯负优化 |

**R2 在线算法**：**有**，且证据三层齐全。① 源仓自身：`_log_sum_exp_fwd_kernels.py` 的 `_logsumexp_kernel_tiled`（L101–189）即 online 变体——分块 running (max, sum) + flash 形式重缩放合并（running 结构是「存在在线算法」的直接证据）；② 参考表：ALG-softmax R2「online softmax（分块 running max/sum + 重缩放）」命中行；③ 结构判据：logsumexp 可分解为 running 统计量（max 与 sum 可增量合并：`S ← S·exp(m_old−m_new) + S_c·exp(m_c−m_new)`）→ 原则上存在在线变体。**收益口径与结论**：online 的价值 = 中间缓冲 O(N)→O(tile) + GM 扫描遍数（大 N 无法驻留时）。**本 single 变体的契约空间（§0.2）已由 wrapper 保证 N_padded ≤ 32512 且驻留预算内**——行块可整体驻留 UB/VRF（wl1：x_ub 双版本 ≤128KB + x_frag 128KB，§4.5），**online 结构在契约空间内非必需**；N 超驻留上限的 shape 由 tiled 变体（online 路径，另任务）承接——**算法族分工即源仓双变体结构，本设计不吞并**。行内 online 微变体（R1 候选 7）在驻留域内净亏，弃选（见上）。

**R3 算法复杂度对比**（逐候选四口径；口径说明：FLOPs/elem 中 cast/select/sub/add/cmp 各计 1 op、exp/log 计 1 special（折算系数按 `CONST-valu-model` special = 8 ops/cycle/AIV，assumption）、log 每行 1 次忽略；GM Bytes/elem = 输入读 + 输出写（输出 M 个 << M·N，忽略）+ 中间 GM 往返；扫描遍数 = 同一 GM 数据读写次数；中间缓冲 = 行级驻存需求峰值（UB+VRF）；代表 workload = wl1 (M=1024, N=4096, fp16, block_m=16)）：

| 算法候选 | FLOPs/elem | GM Bytes/elem | 扫描遍数 | 中间缓冲/行 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线（源驻留三段式，A5 直译形态：N_padded 驻留、单版本 x_ub、block_m 整批 VRF） | ~5 op + 1 special | 2（读 2B；对齐 N）至 3.4（非对齐 N=300：N_padded=512 列） | 2（读 1 + 写 ~0） | x_ub block_m·N_padded·elem + x_frag block_m·TILE·4B（wl1 = 128KB UB 单版本 + 256KB VRF **贴上限**） | M 行独立；无跨核归约 |
| **选定：紧凑驻留 + 行子批三段式（原位覆写）** | ~5 op + 1 special（同基线——公式层源已最优） | 2（任意 N 紧凑；非对齐 N 较基线 −41%） | 2 | x_ub 2·SUB·N·elem（双版本 ≤194.4KB）+ x_frag SUB·TILE·4B（≤128KB VRF）| 同上；子批串行在核内（流水重叠） |
| fp16-e 精简链（备选 §1.6.4，fp16 限定） | ~3 op + 1 special(fp16)（省载入 cast 1 + select 1） | 2 | 2 | x_frag SUB·TILE·2B（VRF 再 −50%） | 同上 |
| 行内单遍 online（R1 候选 7） | ~4 op + 1 special + 2K/N special | 2 | 2 | 同选定 | 行内块序依赖（削弱 N 向流水重叠） |
| Cube GEMV 归并（R1 候选 8） | ~5 op + 1 special + Cube MAC | 2 + ND2NZ/dual_copy 往返（+UB→L1→L0C→UB 中转） | ≥2 | 追加 L1/L0C staging | mixed 核协调开销 |
| 两 kernel 分离（R1 候选 9） | ~5 op + 1 special（+max 落 GM 读回） | 6（读 2×2B + max 写读 4B + 写 ~2B） | 4 | O(1) 行统计量 | 跨 kernel 全局同步 |

**R4 硬件亲和性评估**（逐候选对照检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水融合 / 跨核结构；负向淘汰依据已附）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线直译 | 纯 Vector ✓；special 1/elem 撞最窄吞吐档（`CONST-valu-model` special=8，assumption） | x_frag 256KB **贴 `CONST-vrf-capacity-a5` 假设上限**（wl1 block_m=16；该条目实测下界 76 reg/thread 无 spill 恰在同水位——无余量）；x_ub 单版本 128KB 无流水重叠 | TILE=4096 % 128（fp16 lane）= 0 ✓；N_padded 256 列对齐 UB 驻留浪费（非对齐 N +71% @N=300） | ✓（JIT 常量） | 无双缓冲 → MTE2/VALU 串行 | ⚠️ 可行但 VRF 贴顶 + 非对齐 N 浪费 + 无流水——**降级为结构参照** |
| **紧凑驻留 + 行子批（选定）** | Vector ✓；special 1/elem（公式层已最小——log 每行 1 次）；reduce_* AscendAllReduce 直连（`D-REDUCE-SIMTVF-API` measured） | x_ub 双版本 ≤194.4KB（90% 折扣预算）✓；x_frag ≤128KB VRF（≤32 reg/thread，远低于 76 reg 实测水位）✓ | TILE pow2 % 128 lane = 0 ✓；UB 紧凑（不搬 pad 列）；GM 行子批连续区间 32B 对齐（wl1 行 8KB）✓；非 pow2 N 尾 lane < 2× 浪费（masked） | full/rem/num_sub/TILE 全 JIT 常量 ✓；尾块 TIR if 静态谓词 ✓ | 子批双缓冲（num_stages=2）MTE2/VALU 重叠 ✓；persistent 多波 ✓ | ✅ **选定（主选）** |
| fp16-e 备选 | 同上；exp fp16 走 h 前缀 intrinsic（`intrin_rule_ascend.cc` L27–32 dtype 派发），潜在 2× special 吞吐（**未文档化假设**：`CONST-valu-model` 未区分 dtype 粒度） | VRF 再 −50% ✓ | 同上 ✓ | ✓ | ✓ | ✅ **备选（实验裁决 §1.6.4）**——判定裕度落在 special dtype 粒度这一未实证常数内 |
| 行内 online | 同选定 | 同选定 | ✓ | ✓ | 行内块序依赖削弱流水重叠 | ❌ 弃选（R1 #7：驻留域净亏 + `TRAP-NAIVE-ONLINE-MERGE` 角点风险须额外防护） |
| Cube GEMV | exp 段 Vector + sum 段 Cube——单元混置但 exp（主项）仍 Vector | 追加 L1 staging + L0C 累加器 | 须凑 C0 分形（M/N 16 倍数——M 任意契约下须 padding） | ✓ | mixed 核协调 | ❌ 弃选（R1 #8：转换开销 + 复杂度，归约非瓶颈） |
| 两 kernel 分离 | ✓ | O(1) | ✓ | ✓ | 跨 kernel 间隙 | ❌ 弃选（R1 #9：GM 流量 +200%） |

**调研结论**：选定 **「UB/VRF 行子批驻留三段式 logsumexp（紧凑 N + pow2 masked + exp 原位覆写 + 子批双缓冲流水）」**。关键依据：① R3——公式层源已是该族最优形态（e-base 减 max、log 每行 1 次；R1 候选 2/3 的换底在 A5 同档无益），结构层紧凑驻留对非对齐 N 流量 −41%、行子批使 VRF 从贴上限（256KB）降至 128KB 舒适水位并解锁 x_ub 双版本流水；② R4——纯 AIV 负载、容量全档 ✓（194.4KB/128KB 预算）、pow2 整除、persistent 流水可融合、无跨核归约；③ R2——online 变体存在（源仓 tiled 即证据）但契约空间内驻留可行，非必需（ALG-layernorm R4 row-reduction 形态：全 N 驻留 UB + 按 M 行块分核 + 尾块处理；N 超驻留上限才 N-tile 流式——tiled 变体职责）。与基线的结构差异一句话：**N_padded 对齐驻留 + block_m 整批 + 非持久 grid（源）→ 紧凑驻留 + masked 单路径 + 行子批双缓冲 + persistent 56 核（选定）**，公式恒等。源优化手段意图承接：#1 驻留（R2 承接）/#2 双路径（R2 统一为更优单路径，意图「对齐快路径免掩码」由「pow2 整除时 masked 谓词恒真、编译器可优化」承接）/#3 稳定化（保留）/#4 原位覆写（保留）/#5 树形归约（T.reduce_* 等价替换）/#6 行批（persistent+子批承接）/#7 grid（persistent 承接）——逐项见 §0.5，无静默丢弃。调研范围：ALG-softmax/ALG-reduction/ALG-layernorm 命中行、examples/ascend 四个同类实现 + pilot/online_softmax 正例、pattern-library 五条目（三 measured 二 assumption）、源仓双变体结构分析、kb_search 命中条目（含 pending VP-2026-0017 交叉印证）。

**设计期估算（D-2）**：估算下界(相对)：**计算项（VALU special）主导，约为流量项的 ~11 倍**（代表 workload wl1 (1024, 4096) fp16：流量 8.39MB → GM 读 8.39MB/(56×100 B/cycle)≈1.5K cycle〔`CONST-mte-bandwidth-model` assumption，MTE2 档〕；计算 4.19M×(5/51 + 1/8)/56 AIV ≈ 16.7K cycle〔`CONST-valu-model` assumption，special=8 档——special 0.125 cycle/elem 占计算项 56%〕；容量项可忽略——驻留单遍 GM 读、无 L0C 往返、无分块重扫）。主选 vs 基线直译相对比：VALU 计算项持平（公式同），GM 流量非对齐 N 0.59×（N=300：512→300 列）、对齐 N 1.0×，端到端 ≈ max(16.7K, 1.5K) vs max(16.7K, 1.5K)+流水损失（基线无双缓冲 MTE2/VALU 串行 → +1.5K ≈ +9%）。fp16-e 备选（若 special fp16 2× 吞吐成立）：计算项 0.72×。**口径不确定披露**：special_func「8 ops/cycle」的 op 粒度（element 级 vs 向量指令级）在 `src/ascend/transform/estimate_latency.h` L58–70 未区分——若为向量指令粒度则瓶颈反转 MTE 主导；两种解释下主选结构（special 已最小化 + 流量已最小化）均为最优，该常数列 §9 风险点，Stage 4 npusim/msprof 实测归因标定。A5 冷启动期 assumption 基估算，常数待标定，容量按 90% 折扣。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定、经 §0.6 R2 落地的紧凑驻留三段式算法。源公式层已是 A5 最优形态（e-base、减 max、log 1/行——R1 候选 2/3 弃选），本节优化集中在**结构与访存层**（布局级化简 + 寄存器复用），每项四要素齐备。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| O1 | 减 max 稳定化（必选确认） | `y = log(Σ e^{x})` | `y = m + log(Σ e^{x−m})`，m = max(x) | 指数平移恒等；消 fp16/bf16 数据域溢出（e^11 > 65504） | 计算量不变，数值稳定（源已含，语义层保留项） |
| O2 | 紧凑驻留 + pow2 masked（布局级） | 源：`x_f32 (block_m, N_padded)`，N_padded = align_up(N, 256)，padding 列以 −inf 显式载入（`_needs_pad` 路径）或隐式搬运（对齐路径） | `x_ub (SUB, N)` 紧凑 DMA + `x_frag (SUB, TILE)` pow2 填充，`TILE = next_pow2(N)`，masked 载入 pad = −inf | pad 列两种宽度（N_padded−N vs TILE−N）均为 −inf：max 幂元（max(−inf,·)=·）∧ exp 后 sum 幂元（exp(−inf−m)=0，含 m=−inf 时 exp(NaN) 的 NaN 传播路径两链一致）→ 数学恒等；**机器验证 V1**：全 shape（33×300/32×256/4×4096/128×16384/3×7/1×1）fp64 与 fp32 **位级一致**（bitwise=True，ulp=0） | 非对齐 N（300）：GM 读 −41%（512→300 列 = 1024B→600B/行）与 UB 驻留同减；对齐 N（4096）：驻留不变，消除 `_needs_pad` 双路径维护；任意 N 尾 lane < 2× 浪费由 masked 吸收 |
| O3 | e 原位覆写（寄存器复用确认） | 分离缓冲：`e_frag = exp(x_frag − m)`（双 (SUB,TILE) fp32 fragment 并存） | `x_frag ← exp(x_frag − m_frag[r])` 原位（单 fragment） | 同公式同舍入链（读-算-写同一位置，无额外舍入步骤）；**机器验证 V2**：全 shape fp32 **位级一致**（bitwise=True）——源已用此结构（L89），A5 承接确认 | VRF −50%：wl1 从 256KB（贴 `CONST-vrf-capacity-a5` 假设上限）降至（经行子批）128KB ≤ 32 reg/thread x_frag——spill 风险消除；UB 无变化 |
| O4 | cast 链最小化（确认项） | — | 输入 cast 1/elem（dtype→fp32，载入时融合）+ 输出 cast 1/row（lse 写出前） | fp32 中间是源语义（§0.1 dtype 语义），不可省 | 已最小（无冗余 cast：载入 cast 与 masked select 融合于同一 `T.if_then_else`；输出每行仅 1 次） |

**优化结论**：**采纳 2 项（O2 紧凑驻留+masked、O3 原位覆写）+ 2 项必选/确认（O1 稳定化、O4 cast 链确认）**。优化后主路径公式（§3.1 拆解的唯一输入）：

```
m_i   = max_j x[i,j]                        # fp32（masked 载入 pad=-inf，精确）
e_ij  = exp(x[i,j] − m_i)                   # fp32，原位覆写 x_frag，寄存器驻留
S_i   = Σ_j e_ij                            # fp32 硬件树归约（AscendAllReduce）
y_i   = cast(m_i + log(S_i), dtype)         # 每行 1 次 special + 1 次 cast
```

**等价性机器验证（D-1）**：已生成并执行 `examples/logsumexp/_logsumexp_kernel_single/verify_equiv.py`（fp64 参照 + 角点值：全零行/全等行/单主导/±65504 极值/subnormal ±6e-8/深度下溢 −745/部分 −inf/全 −inf/+inf/NaN/N=1/N 非 pow2（300）/M 尾（33）；torch CPU；候选式与基线式均按目标精度链计算对照）。执行结果（2026-10-16，tilelang-fork 工作区 + torch CPU）：

| 验证项 | 对照（候选 vs 基线/精确） | 最大 ulp 差（fp32 位级） | 最大相对差 | 违反率（@容差） | 结论 |
|--------|--------------------------|--------------------------|------------|---------------------|------|
| V1（O2，6 shape × 2 精度） | pow2-masked 链 vs N_padded 链 | **0**（fp64 与 fp32 均位级一致） | 0 | 0.000000 | **EQUIV_PASS** |
| V2（O3，6 shape） | 原位覆写链 vs 分离 e 链（fp32） | **0**（位级一致） | 0 | 0.000000 | **EQUIV_PASS** |
| V4（源链 vs A5 链） | 源 GPU N_padded 链 vs A5 紧凑链，fp64→fp16 输出，含特殊行 | **0**（位级一致）；NaN 模式逐 shape 一致 | 0 | 0.000000 | **EQUIV_PASS** |
| V3（A5 链 vs golden） | A5 fp32 链 vs `torch.logsumexp`（fp32） | 1（N=300——树形分组差） | 9.526e-08 | 0.000000（@1e-3） | **EQUIV_PASS** |
| V3b（边界语义，信息项） | 特殊行：全 −inf → 链 NaN vs torch −inf；含 +inf → 链 NaN vs torch +inf；含 NaN → 双方 NaN；部分 −inf → 双方一致 | — | — | — | **源继承分歧，文档化**（§9.3；randn 测试域不触发；链与源位级一致——迁移保真） |
| V5（备选预估，不采纳） | fp16-e 模拟链 vs 精确（1024×4096 / 128×16384） | — | 4.211e-05 / 3.589e-05 | 0.000000（@1e-2） | GUARD_PASS（§1.6.4 备选精度守卫内，余量 ~100×） |

> **全部采纳项 EQUIV_PASS**（V1/V2/V4 位级一致；V3 与 golden 1 ulp 树形差）。V3b 证实 ±inf 角点行为为源算法固有（链与源位级一致即迁移保真达成），与 torch 的分歧不构成采纳项失败。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | x_frag masked 载入（含 dtype→fp32 提升） | 逐元素边界分支载入（源 `_needs_pad` 路径同构） | `T.Parallel(SUB, TILE)` + `T.if_then_else(行∧列谓词, T.cast(x_ub[r,i], "float32"), −T.infinity(T.float32))`——谓词含 `r < rows`（尾块行）与 `i < N`（列） | ✅ | —（`example_rmsnorm.py` L56–57 / `online_softmax.py` L71–74 同构掩码载入先例；`D-REDUCE-SIMTVF-AUTOVEC` 实证 Parallel 内 cast/if_then_else 向量化） |
| 2 | 行内 max | 树形归约（源 warp shuffle） | `T.reduce_max(x_frag, m_frag, dim=1, clear=True)` | ✅ | —（`example_per_token_cast_to_fp8.py` L73 同构；`D-REDUCE-SIMTVF-API` measured：SimtVF 内自动降级 `tl::AscendAllReduce<MaxOp>` 硬件树，fp32 直连） |
| 3 | 逐元素 e = exp(x − m)（原位） | 标量循环 exp | `T.Parallel(SUB, TILE)` 向量化（编译器 `ascend_simdvf_lower_parallel` 自动向量化，`tir.exp → simd_vexp`） | ✅ | —（`D-ELEM-SIMTVF-AUTOVEC` measured，源码 `src/ascend/transform/ascend_simdvf_lower_parallel.cc:409-412`） |
| 4 | 行内 sum | 标量累加循环 | `T.reduce_sum(x_frag, s_frag, dim=1, clear=True)` | ✅ | —（`D-REDUCE-SIMTVF-API` measured：`SumOp` 硬件树直连；`testing/ascend/language/test_tilelang_ascend_reduce.py` fp32 回归） |
| 5 | 每行 lse = m + log(S) + cast | 每行标量超越函数 | `T.Parallel(SUB)` 行维向量化（log 为注册 intrinsic，`intrin_rule_ascend.cc` L34–49；`online_softmax.py` L169 `T.log` 在用） | ✅（行维） | 依赖 reduce 终值（S 定稿后才能求 log）——tile 级标量（SUB 个/子批 vs SUB·TILE 元素，占比 1/TILE = 0.02%@4096），无法并入元素循环；非逐元素热点 |
| 6 | lse 写出（fragment→UB→GM） | 标量散写 | `T.copy(out_frag, lse_ub)`（fragment→UB 整块）+ `T.copy(lse_ub[0:rows], y[切片])`（UB→GM DMA） | ✅ | —（`example_per_token_cast_to_fp8.py` L81–83 `(block_groups,)` fragment→UB→GM 一维输出两段写完整同构；`example_rmsnorm.py` L82 切片写先例） |
| 7 | tile 调度循环 | — | `T.Persistent([num_tiles], 56, core_id, num_stages=2)` | 结构循环 | 块级任务映射/调度（非计算循环） |
| 8 | 行子批循环 | — | `T.serial(num_sub)` | ❌（保留串行） | 子批间 x_ub 版本交替的流水顺序依赖（子批 s+1 MTE2 预取 ⊥ 子批 s VALU——双缓冲结构本质）；子批数 ≤ ceildiv(block_m, SUB) ≤ 16，非逐元素热点；收益由流水重叠体现（§6.3） |
| 9 | 尾块 TIR if 分支（`if t < full`） | — | 无向量等价 | ❌（保留分支） | 依赖循环变量 t 的运行时控制流（M 尾块边界语义）；分支体两侧均为静态切片——`example_gemm.py` L47 循环变量 if 先例；每 tile 1 次分支判断，非热点 |
| 10 | host 侧 SUB/TILE/threads/full/rem 计算 | Python 算术 | — | n/a | host 元数据计算，不在 kernel 内（per-shape JIT 静态化；决策树 §1 维度自推导） |

**向量化结论**：逐元素计算 100% 向量化（#1/#2/#3/#4/#6 全部 `T.Parallel`/`T.reduce_*`，API 均有 examples + pattern-library measured 佐证）；保留 3 处非向量化点——#5 tile 级每行 lse（依赖归约终值，占比 1/TILE）、#8 子批串行（双缓冲流水本质）、#9 尾块分支（运行时边界语义，先例佐证）——理由类别 = tile 级标量 + 顺序依赖 + 边界处理，均非「实现简单」类借口；与 §6 循环结构一致（§6 无逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（[M, N] 行主进 / (M,) 行主出），核内布局与 lane 映射是设计变量。规约类算子必选候选：水平归约（lane 映射到归约轴 N）vs 垂直扫描（lane 映射到独立行 M、串行步进归约维）——正交决策分别枚举；另评估分块/跨核变体。

**轴质量评分**（对选定轴 N）：

| 评分项 | 判定 |
|--------|------|
| 整除性 | 主 workload TILE=4096/16384 % 128（fp16 lane）= 0 ✓ / % 64（fp32 lane）= 0 ✓（零尾 lane）；非 pow2 N（300→TILE 512）尾 lane 浪费 (512−300)/512 ≈ 41%（masked 吸收，仅 fragment 域——DMA 侧紧凑无浪费） |
| 尾 lane 浪费率 | 主 workload 0%；N=300 → fragment 域 41%（一次性载入开销，归约/计算仍全量有效——pad lane 的 exp(−inf−m)=0 不污染 sum） |
| 累加链形态 | 归约维 N = stride-1 连续向量维（`T.reduce_*(dim=1)` on (SUB, TILE) fragment）；无跨步系数 |
| repack 代价 | 无（I/O 原生行主连续，核内布局与 I/O 同构；x_ub 紧凑 (SUB,N)，fragment pow2 填充——`example_rmsnorm.py` L42 先例：仅 fragment 需 pow2，DMA 紧凑） |
| UB/VRF 容量影响 | 行子批驻留：x_ub 双版本 ≤194.4KB + x_frag ≤128KB VRF（§4.5 预算表，SUB 封顶） |

**候选矩阵**（规约类候选全枚举，弃选给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | 行主 [M,N] 原生 + **水平归约**（lane→N 连续轴，SUB 行子批驻留，persistent 行块） | N | 无 | 连续 GM 行（wl1 行 8KB，32B 对齐 ✓）；归约维 = stride-1 向量维；rmsnorm + per_token + online_softmax 三先例；reduce 硬件树直连 | ✅ **主选** |
| 2 | 垂直扫描（lane→M 行，N 向串行步进） | M | 无 | N=4096 串行步进/行——向量宽度利用率 SUB/128 ≤ 12.5%（SUB≤16），行内并行度归零；吞吐劣化 >8×（按 128 lane 满载比估算）；repo 无先例 | ❌ 串行化 N 维，量化劣化不可接受 |
| 3 | 2D 分块 + 跨核行归约（M/SUB × N/BN grid，partial 落 GM + 跨核 flag 合并） | N（块内） | partial 统计量 GM 往返 | 行块已可驻留（wl1 双版本 128KB + fragment 128KB），跨核归约新增 2 遍 partial GM 读写 + 跨核同步（流量 +50% 起步）；UB 容量足够无需分块 | ❌ 违反流量最小化（R3）；容量不构成约束 |
| 4 | 转置布局（列主驻留 + M 向归约） | M（转置后） | 核内转置链（`T.copy` transpose 仅特定路径——非通用，`tilelang/ascend/language/copy_op.py`；通用转置 API 存在性未命中→「未文档化假设」） | 转置 repack 开销 + UB 翻倍（双布局驻留）；收益仅在于把归约轴换到 M——M 维短（wl1 1024 vs N 4096）无收益 | ❌ repack 纯开销、无轴收益（N 已是最长连续轴） |

**源码轴选择对照**（迁移任务必做）：源 GPU 也是 N 轴并行（`T.Parallel(block_m, N_padded)` 行内元素线程化 + warp 归约 dim=1）+ 行级 grid——**轴结论一致**（N 是唯一高带宽连续轴，两代硬件共识），但承载结构完全重设计：GPU block×threads=128 → SimtVF 线程域 + `T.reduce_*` 硬件树；GPU 非持久 grid → persistent 56 核行块 + 行子批（§0.6 R1/R2）。GPU 轴选择是输入而非结论，已独立复核。

**布局决策结论**：选定方案 #1：核内布局 = `x_ub (SUB, N)` dtype 紧凑（双版本）+ `x_frag (SUB, TILE)` pow2 填充（fp32，masked）；向量化轴 = **N（行内连续轴）**；repack = 无。该结论同步落入 §3.3 伪代码与 §6 循环结构（Parallel 内层维 = TILE/N 轴，buffer 形状 = §4.3）；弃选方案量化理由见候选矩阵（#2 串行化 >8×、#3 流量 +50%、#4 repack 无收益）。

#### 1.6.4 实验裁决三件套（fp16-e 备选）

> 主选（fp32-e 三段式）与备选（fp16-e 精简链，仅 fp16 输入）的取舍判定裕度落在未实证常数区间（special_func 吞吐的 dtype 粒度——`CONST-valu-model` 未区分 fp16/fp32，A5 冷启动 assumption；fp16 h 前缀 intrinsic 潜在 2× 吞吐为「未文档化假设」，估算依据 = `src/ascend/codegen/intrin_rule_ascend.cc` L27–32 的 dtype 派发存在性），按 skill Phase 2 第 3 项强制产出三件套：

- **主选方案**：fp32-e 紧凑驻留三段式（本 DESIGN §3–§7 全部按此展开）——`x_frag (SUB, TILE)` fp32 masked 载入，`T.exp` fp32 求值，`T.reduce_sum` fp32 硬件树；与源精度链一致（§0.1 fp32 中间语义；V3 vs golden max_rel 9.5e-8）。
- **备选方案**（1 个，结构完整可实现，**仅 fp16 输入 dtype 分派**）：fp16-e 变体——`x_frag` 载入输入 dtype（省载入 cast 1 op/elem 与 select 1 op/elem：`if_then_else(i<N, x_ub[r,i], fp16(−inf))`），`m_frag` 仍 fp32（`T.reduce_max` fp32——max 对 fp16 值域精确，减法 `x − m` 在 fp32 域完成后 **arg cast fp16** 再 `T.exp`（h 前缀 intrinsic），`T.reduce_sum` fp16 硬件树（`CONST-valu-model`：fp16 硬件直连 ✓），`S` 升 fp32 求 log：`y = cast(m + log(fp32(S)), fp16)`。buffer 形状：x_frag (SUB, TILE) **fp16**（VRF 再 −50% → 64KB）；UB 预算不变（x_ub 仍输入 dtype）；循环结构/分核三要素/persistent 流水均不变（§5.5 同款）；精度预估：verify_equiv.py **V5 模拟链 max_rel 4.211e-05**（1024×4096）/ 3.589e-05（128×16384），@1e-2 违反率 0——守卫余量 ~100×（远优于预估的 2e-3–6e-3，因 torch fp16 求和的分组舍入优于最坏树深估计；实测硬件树行为以 Stage 3 分层测试为准）。**bf16 输入不适用本备选**（bf16 尾数 8 位，S 树形相对误差预估 ~1.4e-2 超容差；fp32 输入无吞吐差）——分派条件 `dtype == "float16"`。
- **实验裁决计划**：Stage 4 于代表 shape（bench 主对象）**wl1 (1024, 4096) fp16** 与 **wl4 (128, 16384) fp16** 实测两变体 `do_bench(backend="msprof")` 时延 + 精度（vs golden 最大相对误差，容差 1e-2）；**判定阈值**：备选时延 < 主选 × 0.90 **且** max_rel_err ≤ 5e-3 → 采纳备选（工厂按 dtype 分派：fp16 → 备选，bf16/fp32 → 主选——`lru_cache` 键含 dtype，达成总体最优而非全局二选一）；否则主选固化，实测数字经附录补记回写本节；实测反解 special_func fp16 吞吐常数 → 回写 `CONST-valu-model`（Tier 0，含 repro 三件套）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: simt（auto_schedule: yes）

### 2.2 选型理由

1. **迁移默认 + 算子特征**：纯 Vector 算子（elementwise cast/exp + 行规约 max/sum + 超越函数，无 matmul/Cube 段）——决策树「含归约 → simt + reduce/reducer」路径；调度 prompt 指定默认 simt，重设计未产生 mixed 需求（无 Cube 工作落位，`T.MixedKernel` 不适用）。
2. **repo 先例**：同族（行规约）`examples/pilot/online_softmax/online_softmax.py`（simt + `T.reduce_max/reduce_sum` + `T.Persistent` + SimtVF，63 用例全 PASS 的 measured 正例）、`examples/ascend/example_rmsnorm.py`（simt + fragment + 行块流水）、`examples/ascend/example_per_token_cast_to_fp8.py`（simt + `T.reduce_max` + `T.Persistent` + (M,) 一维输出写出）——API 佐证链最完整。
3. **从 simt 起步原则**（决策树 §2 注）：无证实热点前不降级 simd。降级触发条件（Stage 4 观察项）：msprof/npusim 证实 special_func 段为瓶颈且 simt 自动向量化不足时，可评估 SimdVF + `T.simd.vexpdif`（融合 exp-sub intrinsic，`tilelang/ascend/language/simd.py` L1182–1195 定义 dst=exp(src0−src1)，`examples/ascend/flash_attention/core.py` L130–133 在用——fp32 同宽形态）——收益预估仅省 1 sub @51 档（0.02 cycle/elem vs special 0.125 主项），优先级低于 fp16-e 备选（§1.6.4）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| VF 入口 | `with T.SimtVF(threads=min(1024, SUB·TILE))` + `T.Parallel(SUB, TILE)` 逐元素段 + `T.reduce_max/reduce_sum(dim=1, clear=True)` 行归约（硬件树直连） |
| 内存分配 | `T.alloc_shared`（UB：x_ub/lse_ub）+ `T.alloc_fragment`（VRF：x_frag/m_frag/s_frag/out_frag） |
| 同步 | auto_schedule: yes——AutoSchedule 管理跨引擎 flag 与 multibuffering（§7） |
| 模块声明 | kernel 模块 docstring 须声明 `programming_mode: simt`（迁移 prompt 契约，§10.3） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 优化后公式；每步优先向量化映射，与 §1.6.2 结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `m_i = max_j x[i,j]` | 行最大（fp32，masked pad=−inf 幂元，精确） |
| 2 | `e_ij = exp(x[i,j] − m_i)` | fp32 稳定化指数，原位覆写，寄存器驻留 |
| 3 | `S_i = Σ_j e_ij` | fp32 硬件树归约 |
| 4 | `y_i = cast(m_i + log(S_i), dtype)` | 每行 1 次 special + 1 次 cast，写出 (M,) |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 行子批搬入 | X 子批 GM→UB | `T.copy(X[row_base : row_base+rows, 0:N], x_ub)` | 2D 连续行块区域拷贝（列 0:N 全覆盖 → GM 连续区间；rows ∈ {SUB, rem_sub} 静态） | simt（MTE2） |
| 掩码载入 | x_ub → x_frag（pow2 填充 + fp32 提升） | `for r, i in T.Parallel(SUB, TILE): x_frag[r,i] = T.if_then_else(T.And(r < rows, i < N), T.cast(x_ub[r,i], "float32"), -T.infinity(T.float32))` | pad = −inf（max 幂元）；`T.infinity` 接受 dtype 对象（`D-REDUCE-SIMTVF-API` measured 注记） | simt |
| 行最大 | m | `T.reduce_max(x_frag, m_frag, dim=1, clear=True)` | fragment 维归约 → AscendAllReduce 硬件树 | simt |
| 指数（原位） | e = exp(x−m) | `for r, i in T.Parallel(SUB, TILE): x_frag[r,i] = T.exp(x_frag[r,i] - m_frag[r])` | fp32 全程；`tir.exp` 自动向量化（`D-ELEM-SIMTVF-AUTOVEC` measured） | simt |
| 行和 | S | `T.reduce_sum(x_frag, s_frag, dim=1, clear=True)` | fp32 fragment 硬件树 | simt |
| 行输出 | y = cast(m + log(S), dtype) | `for r in T.Parallel(SUB): out_frag[r] = T.cast(m_frag[r] + T.log(s_frag[r]), dtype)` | 每行 1 次 log（`intrin_rule_ascend.cc` 注册；`online_softmax.py` L169 在用） | simt（行维） |
| lse 暂存 | fragment→UB | `T.copy(out_frag, lse_ub)` | (SUB,) 一维 fragment→UB 整块 | simt |
| 搬出 | UB→GM | `T.copy(lse_ub[0:rows], y[row_base : row_base+rows])` | 1D 静态切片 | simt（MTE3） |
| tile 调度 | — | `with T.Kernel(56) as core_id: for t in T.Persistent([num_tiles], 56, core_id, num_stages=2):` | persistent 行块 | — |
| 版本管理 | — | `T.annotate_buffer_versions({x_ub: 2})` | 子批双版本流水 | — |
| 尾块分支 | — | `if t < full: … else: …`（TIR if，循环变量条件） | full/rem JIT 常量 | — |

### 3.3 计算伪代码

```python
# host 静态化（builder 闭包，per-(M, N, dtype, block_m) JIT + lru_cache）：
#   TILE     = next_pow2(N)
#   SUB      = max 2^k s.t. 2*SUB*N*elem <= 194.4KB and SUB*TILE*4B <= 128KB and SUB <= block_m
#   num_sub  = ceildiv(block_m, SUB)          # 全块 tile 的子批数
#   full     = M // block_m;  rem = M % block_m;  num_tiles = full + (1 if rem else 0)
#   tsub     = rem // SUB;  tsub_rem = rem % SUB   # 尾 tile 的满子批数 / 末子批行数
#   threads  = min(1024, SUB * TILE)

@T.prim_func
def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M,), dtype)):
    with T.Kernel(56) as core_id:
        x_ub = T.alloc_shared((SUB, N), dtype)          # 紧凑驻留（双版本流水）
        lse_ub = T.alloc_shared((SUB,), dtype)          # 行输出暂存（单版本）
        T.annotate_buffer_versions({x_ub: 2})
        for t in T.Persistent([num_tiles], 56, core_id, num_stages=2):
            base = t * block_m
            if t < full:                                 # ---- 全块 tile（TIR if）----
                for s in T.serial(num_sub):
                    row_base = base + s * SUB
                    T.copy(x[row_base : row_base + SUB, 0:N], x_ub)      # MTE2，版本 v
                    with T.SimtVF(threads=threads):
                        x_frag = T.alloc_fragment((SUB, TILE), "float32")
                        m_frag = T.alloc_fragment((SUB,), "float32")
                        s_frag = T.alloc_fragment((SUB,), "float32")
                        out_frag = T.alloc_fragment((SUB,), dtype)
                        # ① masked 载入（行谓词恒真，列谓词 i<N；pad=-inf 双幂元）
                        for r, i in T.Parallel(SUB, TILE):
                            x_frag[r, i] = T.if_then_else(
                                i < N, T.cast(x_ub[r, i], "float32"),
                                -T.infinity(T.float32))
                        # ② 行最大（硬件树）
                        T.reduce_max(x_frag, m_frag, dim=1, clear=True)
                        # ③ 原位稳定化指数（自动向量化）
                        for r, i in T.Parallel(SUB, TILE):
                            x_frag[r, i] = T.exp(x_frag[r, i] - m_frag[r])
                        # ④ 行和（硬件树）
                        T.reduce_sum(x_frag, s_frag, dim=1, clear=True)
                        # ⑤ 每行 lse + cast
                        for r in T.Parallel(SUB):
                            out_frag[r] = T.cast(m_frag[r] + T.log(s_frag[r]), dtype)
                        T.copy(out_frag, lse_ub)                          # fragment→UB
                    T.copy(lse_ub, y[row_base : row_base + SUB])          # MTE3
            else:                                        # ---- 尾块 tile（rem 行）----
                for s in T.serial(tsub):                 # 满 SUB 子批
                    row_base = base + s * SUB
                    <同上 ①–⑤，rows=SUB（行谓词恒真）>
                if tsub_rem > 0:                         # Python 级静态分支（闭包常量）
                    row_base = M - tsub_rem
                    T.copy(x[row_base : M, 0:N], x_ub)                    # 静态 rem 切片
                    with T.SimtVF(threads=threads):
                        <同上 ①–⑤，rows=tsub_rem（行谓词 r < tsub_rem 生效，
                          越界行载入 -inf、计算结果不写出）>
                    T.copy(lse_ub[0:tsub_rem], y[row_base : M])           # 静态 rem 切片
```

> 尾块路径的 ① 展开后行谓词为 `T.And(r < rows_total, i < N)`（rows_total = tsub_rem），与源 GPU kernel L75 的 `T.And(pid_m * block_m + i < M, j < N)` 语义一致（行 OOB 以 −inf 载入、不写出）；全块路径 rows_total = SUB 谓词恒真，编译器可优化。伪代码与 §1.6.3 布局决策、§4.3 buffer 形状、§6 循环结构三方一致。

### 3.4 API 可行性确认

| API | 佐证（examples / testing / 源码） |
|-----|----------------------------------|
| `T.Kernel(56)` 一维 + 无 threads | `examples/pilot/online_softmax/online_softmax.py` L57；`example_per_token_cast_to_fp8.py` L44；ascend-constraints §1 条 1 |
| `T.Persistent([num_tiles], N, core_id, num_stages=2)` | `online_softmax.py` L60；`example_per_token_cast_to_fp8.py` L45；`example_rmsnorm.py` L49（T.Pipelined 变体） |
| `T.SimtVF(threads=)` + `T.Parallel` + `T.if_then_else` 掩码 | `online_softmax.py` L63/L71–74（行∧列谓词掩码同构）；`example_rmsnorm.py` L54–57 |
| `T.reduce_max / T.reduce_sum(dim=1, clear=True)` | pattern-library `D-REDUCE-SIMTVF-API`（measured：AscendAllReduce 硬件树实见）；`example_per_token_cast_to_fp8.py` L73；`testing/ascend/language/test_tilelang_ascend_reduce.py` |
| `T.exp`（fp32 向量化）/ `T.log` | `D-ELEM-SIMTVF-AUTOVEC`（measured：`ascend_simdvf_lower_parallel.cc:409-412`）；`online_softmax.py` L169（T.log）；`intrin_rule_ascend.cc` L34–49 注册 |
| `T.cast(dtype→"float32"/dtype)`（含 bfloat16） | `online_softmax.py` L73/L83；`examples/ascend/flash_attention/core.py` L78–100（bf16 cast 链） |
| `T.copy` GM→UB 2D 静态切片（紧凑 N） | `online_softmax.py` L62/L142（静态 rem 切片先例）；`example_per_token_cast_to_fp8.py` L50；`example_rmsnorm.py` L42 注释（DMA 紧凑） |
| `T.copy` fragment→UB + UB→GM 1D 切片 | `example_per_token_cast_to_fp8.py` L81–83（(block_groups,) 两段写同构）；`example_rmsnorm.py` L76/L82（切片写 RSTD） |
| `T.annotate_buffer_versions` | `online_softmax.py` L59；`example_per_token_cast_to_fp8.py` L49 |
| `T.infinity(T.float32)` | `online_softmax.py` L73；`D-REDUCE-SIMTVF-API` 注记（接受 dtype 对象） |
| 核内 TIR if（循环变量条件） | `example_gemm.py` L43–56（`aswt_swizzle` macro，Persistent 循环体内 `if row_idx < MAIN_ROW:`，A5 编译运行通过） |
| Python 级常量 if（prim_func 体内） | `online_softmax.py` L141（`if rem > 0:`） |
| `@tilelang.jit(target="ascend", out_idx=[1])` 懒工厂 | `online_softmax.py` L215–223（`_is_lazy_style` 形态，S3-JIT 门禁要求——CASE-pilot-online-softmax 教训） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（ascend-constraints §2 强制检测）

| 检测项 | 本算子结论 |
|--------|-----------|
| 一维 Kernel / threads= 误用 | ✅ 源含 `T.Kernel(grid, threads=threads)`（二维参数 + threads）——**已重设计**：一维 `T.Kernel(56)` + `T.SimtVF(threads=)`（§0.6 R1）；wrapper 调用面按 K3 移除 threads（§4.7） |
| GPU 专用 API | ✅ 无 TMA/cp.async/warp shuffle/cluster 依赖（源用 `T.copy`/`T.reduce_*`/`T.Parallel`，表层共享）；SMEM 两跳 staging 等价替换为 UB 单跳（§0.5） |
| C0 分形对齐 | ✅ 不适用（无 GEMM/Cube 段；`CONST-gemm-dim-constraints` 仅约束 `T.gemm` 路径） |
| L0C/UB 溢出 | ✅ 无 L0C；UB 预算 §4.5（x_ub 双版本 + lse_ub ≤ 194.4KB 90% 折扣内）；VRF §4.5（≤128KB，SUB 封顶） |
| OOB padding 误用 | ✅ 不使用 `pad_value=`/`data_select=`（列向 padding 语义不适用于行 OOB——tilelang-ascend skill §6.1；行尾用 fragment 谓词 + 静态切片规避，§0.6 R4） |
| 分核策略缺失 | ✅ §5.5 三要素齐备（grid 56 实查 / 纯 AIV 分工 / auto_schedule 声明） |
| 静态循环边界 | ✅ 全部 JIT 常量（num_tiles/num_sub/tsub/tsub_rem/TILE/SUB/threads）；核内串行循环边界静态 |
| flag id 预算 | ✅ auto_schedule yes，无手动 flag（§7） |
| Host 侧输入操作 | ✅ host 仅做 shape/元数据推导（TILE/SUB/full/rem）与 JIT 缓存；不改输入张量内容（op 层已保证 contiguous/reshape，本 kernel 无 host 预处理） |

#### 3.5.2 参考实现差异说明

源 GPU 实现与本设计的差异全部源于 §0.5/§0.6 迁移决策（N_padded→紧凑、双路径→单路径、grid→persistent+子批、threads→SimtVF、隐式 OOB→静态分解），公式与数值语义恒等（verify_equiv.py V4 位级一致）。**不复用源码任何 GPU 形态结构**（`T.Kernel(grid, threads)`、SMEM staging、`_needs_pad` 分支）。

#### 3.5.3 本项目同类实现参考

| 参考文件 | 复用内容 |
|---------|---------|
| `examples/pilot/online_softmax/online_softmax.py` | 同族 row-reduction 迁移正例：resident 路径整体同构（persistent 56 + SimtVF + masked 载入 + reduce_max/exp/reduce_sum 链 + 双版本流水）；JIT 懒工厂形态；分层测试组织 |
| `examples/ascend/example_per_token_cast_to_fp8.py` | (M,) 一维输出两段写（fragment→UB→GM 切片）；SimtVF 内 reduce_max；`T.Persistent` + `annotate_buffer_versions` 三件套 |
| `examples/ascend/example_rmsnorm.py` | 紧凑 UB + pow2 fragment masked 先例（L42 注释）；UB 216KB 预算化行批处理思路 |
| `examples/ascend/example_gemm.py` | 核内循环变量 TIR if 先例（aswt_swizzle L47） |
| `examples/ascend/flash_attention/core.py` | bf16 cast 链先例（L78–100） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| x | (M, N)（per-shape JIT 常量；wl1 = (1024, 4096)、wl4 = (128, 16384)） | float16 / bfloat16 / float32 | 输入矩阵，行主连续（op 层 `movedim + contiguous + reshape` 保证；契约空间见 §0.2）；host 侧仅元数据视图操作 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| y | (M,) | 同输入 dtype | 行级 logsumexp 结果；kernel 显式输出参数（`out_idx=[1]`）；写出量 M×elem_bytes（wl1 = 2KB，可忽略 vs 读 8MB） |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| x_ub | (SUB, N)（wl1 = (8, 4096)、wl4 = (2, 16384)） | 输入 dtype | UB（`T.alloc_shared`，双版本） | 行子批 DMA 暂存（紧凑 N，不搬 pad 列） |
| x_frag | (SUB, TILE)（wl1 = (8, 4096)、wl4 = (2, 16384)） | float32 | VRF（`T.alloc_fragment`） | masked 载入 + fp32 提升的行数据（pad=−inf）；**原位覆写为 e**（单 fragment） |
| m_frag / s_frag | (SUB,) | float32 | VRF | 行统计量（max / sum） |
| out_frag | (SUB,) | 输入 dtype | VRF | 每行 lse（cast 后） |
| lse_ub | (SUB,) | 输入 dtype | UB（单版本） | lse 写出暂存（fragment→UB→GM 两段写） |

### 4.4 内存搬运路径

```
GM[x (M,N) dtype]
  --T.copy (MTE2, 2D 连续行子批 [row_base:+rows, 0:N], x_ub 双版本 v/v⊕1)--> UB[x_ub (SUB,N)]
  --SimtVF masked T.Parallel (行∧列谓词, pad=-inf, cast fp32)--> VRF[x_frag (SUB,TILE) fp32]
  → VRF reduce_max → m_frag → VRF x_frag 原位 = exp(x_frag − m_frag) → VRF reduce_sum → s_frag
  → VRF out_frag = cast(m_frag + log(s_frag), dtype)
  --T.copy (fragment→UB)--> UB[lse_ub (SUB,) dtype]
  --T.copy (MTE3, 1D 静态切片 [row_base:+rows])--> GM[y (M,)]
```

无 L1/L0A/L0B/L0C 段（纯 Vector）；无中间量落 GM；x 单遍 GM 读、y 单遍 GM 写（R3 口径：扫描遍数 2 = 读 1 + 写 1）。

### 4.5 UB / VRF 内存预算

**UB 预算**（上限 216KB〔SimtVF〕，`CONST-capacity-a5` assumption 按 90% 折扣 = 194.4KB）：

| Buffer | wl1 (BM=16, N=4096, fp16) | wl4 (BM=4, N=16384, fp16) | 版本 | 说明 |
|--------|---------------------------|---------------------------|------|------|
| x_ub | (8, 4096)×2B×2 版本 = 128KB | (2, 16384)×2B×2 版本 = 128KB | 2 | SUB=8 / SUB=2（§5.2 推导） |
| lse_ub | (8,)×2B = 16B | (2,)×2B = 4B | 1 | 写出暂存 |
| 编译器保留裕量 | ~8KB（`UB_SAFETY_RESERVE_BYTES` 同量级） | 同左 | — | 小 fragment/标量 planned-into-UB 开销 |
| **总计** | **≈136KB / 194.4KB ✓** | **≈136KB / 194.4KB ✓** | | |

**SUB 上界推导**（host 静态化公式，见 §3.3 注）：`SUB = 满足以下约束的最大 2 的幂：2·SUB·N·elem ≤ 194.4KB ∧ SUB·TILE·4B ≤ 128KB ∧ SUB ≤ block_m`。该式保证**契约空间全域**（任意 N ≤ 32512、任意 block_m、三 dtype）UB 双版本与 VRF 同时达标：fp32 时 elem=4B 使 UB 约束更紧（N=4096 → SUB ≤ 5.9 → SUB=4 → x_ub 2·4·4096·4 = 128KB ✓）；N=32512 fp16 → TILE=32768 → SUB ≤ min(194.4K/(2·32512·2)=1.49, 128K/(32768·4)=0.97) → SUB=1（x_frag 128KB ✓）。

**VRF 预算**（假设上限 256KB，`CONST-vrf-capacity-a5` assumption；实测下界样本 76 reg/thread @1024 threads 无 spill——online_softmax 主 shape）：

| Buffer | wl1 | wl4 | 说明 |
|--------|-----|-----|------|
| x_frag (SUB, TILE) fp32 | 8×4096×4B = 128KB | 2×16384×4B = 128KB | ≤128KB 封顶（SUB 约束第二条） |
| m/s/out_frag (SUB,) | ~96B | ~24B | 标量 |
| **合计 / thread** | ≈128KB+ε / 1024 threads ≈ **32 fp32 reg/thread + ~12 标量 ≈ 44 reg** | 同量级 | **远低于 76 reg 实测水位**——spill 风险由 SUB 封顶消除（对照：无子批的 block_m=16 整批 = 64 reg + 标量 ≈ 76 reg 贴水位） |

**对照基线（无行子批）**：x_frag (16, 4096) fp32 = 256KB = `CONST-vrf-capacity-a5` 假设上限 100%（76 reg/thread 贴实测水位，无余量）；x_ub (16, 4096) 双版本 = 256KB **> 194.4KB 超预算**（单版本 128KB 无流水）。行子批同时解决两者（§1.6.0 R4 容量列）。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| M | JIT 闭包参数（`T.Tensor((M, N), ...)` 符号维度），per-shape 编译 + `lru_cache` | ≥ 1 任意（含 M < block_m：full=0 全尾块路径；M % block_m ≠ 0 尾 tile） |
| N | 同上 | 契约空间：N_padded ≤ 32512 且 block_m·elem·N_padded ≤ 216KB（wrapper 判定，§0.2）；kernel 侧 assert 防御 |
| block_m | `_func(block_m)` 调用参数（per-config 编译） | wrapper 启发式 {1,2,4,8,16} + 显式 config 任意 ≥1（SUB 机制兜底） |
| dtype | JIT 闭包参数 | float16 / bfloat16 / float32 |

### 4.7 JIT 配置与调用面契约

```python
import functools
import tilelang


def _logsumexp_kernel_single(M: int, N: int, dtype: str):
    """Build a single-tile logsumexp kernel (N fits on-chip) — A5 re-implementation.

    Keeps the wrapper call surface `_logsumexp_kernel_single(M, N, dtype)(block_m)(x)`;
    the GPU `threads` argument is removed per K3 (threads derived inside: min(1024, SUB*TILE)).
    """
    @tilelang.jit(target="ascend", out_idx=[1])
    def _func(block_m):
        # host 静态化：TILE / SUB / num_sub / full / rem / num_tiles / tsub / tsub_rem / threads
        # + 契约防御：assert N <= MAX_SINGLE_TILE_COLS（32512）, block_m >= 1
        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M,), dtype)):
            ...  # §3.3 伪代码
        return main
    return _func


# wrapper 对接（logsumexp.py L64–68 / L83 现行调用面，Stage 5 集成不改 wrapper）：
#   _logsumexp_kernel(M, N, dtype_str, tile_n=0)  ->  _logsumexp_kernel_single(M, N, dtype)
#   kernel = _logsumexp_kernel_single(M, N, dtype)(block_m)     # (block_m) JIT 编译
#   y = kernel(x)                                               # (x) 执行，out_idx=[1] 返回 y
```

- **JIT 形态**：`@tilelang.jit(target="ascend", out_idx=[1])` 懒工厂（工厂返回 PrimFunc，`_is_lazy_style` 识别）——S3-JIT 门禁要求（CASE-pilot-online-softmax 教训：`tilelang.compile` 形态被拒）；per-(M, N, dtype, block_m) 编译由 jit 装饰器内建缓存承接（wrapper 侧另有 `lru_cache` dispatch）。
- **K3 落地**：`_func(block_m)` 单参数（GPU 版为 `(block_m, threads)`）；wrapper `LogSumExpKernel.forward` → `_logsumexp_fwd_wrapped(...)(block_m)(x)` 调用面不变。
- **host 防御**（TRAP-GRID-DIVISIBILITY 纪律）：builder 内 `assert` 契约（N ≤ 32512 单 tile 契约、block_m ≥ 1、M ≥ 1、N ≥ 1）——persistent 调度器保证 tile 全覆盖（无手动 grid-stride 漏写风险），assert 仅拦契约外误用。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 算子仅含逐元素运算（cast/exp/sub/log）与行内规约（max/sum），无 matmul——无 Cube 段，全部计算与搬运落 AIV（56 个）。

### 5.2 Block 划分

```python
TILE     = next_pow2(N)                  # fragment pow2 填充（wl1 4096 / wl4 16384）
SUB      = max 2^k s.t. 2*SUB*N*elem <= 194.4KB and SUB*TILE*4B <= 128KB and SUB <= block_m
                                          # wl1 (fp16): SUB=8；wl4 (fp16): SUB=2；(32,256,fp32): SUB=16=block_m（无子批）
num_sub  = ceildiv(block_m, SUB)         # wl1: 2；wl4: 2
full     = M // block_m                  # wl1: 64；wl4: 32
rem      = M % block_m                   # wl1: 0；wl4: 0；test (33,256,bm=16): rem=1
num_tiles = full + (1 if rem else 0)     # persistent 域
NUM_BLOCKS = 56                          # = vector_core_num（实查，§5.5）
threads  = min(1024, SUB * TILE)         # wl1/wl4: 1024；小 shape 自动收缩（如 (1,1): 1）
```

- **SUB 依据**（双约束封顶，§4.5）：① x_ub 双版本 ≤ 194.4KB（解锁子批间 MTE2/VALU 流水）；② x_frag ≤ 128KB VRF（≤32 reg/thread，spill 风险消除）；③ SUB ≤ block_m（契约参数不被放大）。**任意契约 block_m**（wrapper 启发式 1–16 或显式 config 更大值）均由 SUB 封顶兜底——block_m 只决定 tile 数与子批数，不触碰容量约束。
- **block_m 契约来源**：wrapper `default_config`（single 优先最大 block_m：wl1 → 16、wl4 → 4）或 caller 显式 config——kernel 侧不重选、不假设其整除 M（尾块路径 §5.4）。
- **驻留上限（契约边界）**：wrapper 判定 N_padded ≤ 32512 ∧ block_m·elem·N_padded ≤ 216KB 进 single；N 超界走 tiled 变体（另任务）。kernel 侧无 N 维分块（单 tile 语义）。

### 5.3 约束分析

- **C0 分形对齐**: 不适用（无 GEMM/Cube 段；`CONST-gemm-dim-constraints` 仅约束 `T.gemm` 路径）。
- **UB 容量**: §4.5 预算表（wl1/wl4 ≈136KB ≤ 194.4KB，`CONST-capacity-a5` assumption 90% 折扣）✓。
- **VRF 容量**: §4.5（≤128KB ≈ 44 reg/thread，远低于 76 reg 实测水位，`CONST-vrf-capacity-a5`）✓；Stage 3 编译后首查 spill 计数（`get_kernel_source()`）复核。
- **GM→UB 对齐**: wl1 行 8KB / wl4 行 32KB，32B 对齐 ✓；子批 2D 切片列 0:N 全覆盖 → GM 连续区间（单段 DMA）。非 32B 行长（N=300 fp16 = 600B）：DMA 字节级行长支持（tilelang-ascend skill §6.1 "byte-granular rows"——正确性不受影响），带宽档降级列 §9.3 观察项。
- **向量宽度整除**: TILE=4096/16384 % 128（fp16 lane）= 0、% 64（fp32 lane）= 0 ✓；非 pow2 N 由 masked 载入吸收（fragment 域尾 lane < 2× 浪费，§1.6.3）。
- **静态循环边界**: num_tiles / num_sub / tsub / tsub_rem / TILE / SUB / threads 全 JIT 常量 ✓（gate 约束）。

### 5.4 注意事项（非整除 / 边界处理策略）

- **行尾（M % block_m ≠ 0）**：静态 full/rem 分解 + 尾 tile TIR if 分支（`if t < full`）——尾 tile 满子批 `T.serial(rem // SUB)` + Python 静态 `if rem % SUB > 0` 末子批；全部 `T.copy` 切片静态（`[M − tsub_rem : M]`）；fragment 载入行谓词 `r < rows_total`（越界行 −inf 载入、不写出）——**不依赖 `T.copy` 越界行为**（§0.6 R4；对照源 GPU 对齐路径的隐式 OOB 依赖）。
- **列尾（N < TILE）**：masked 载入（pad=−inf，max/exp/sum 三幂元中性，V1 位级验证）+ lse 写出与列无关——无数值影响。
- **M < block_m**（含 M=1）：full=0、rem=M → 单尾 tile 路径全覆盖。
- **56 核尾波**：wl1 num_tiles=64 = 56+8（末波 8/56 核，~1.6% 不均衡）；wl4 num_tiles=32 < 56（24 核空转一轮——M=128 负载小，绝对影响可忽略）。Stage 4 可调 block_m 对齐（core-split-strategy §2.4 提示；block_m 是 wrapper config 可调参数，不改 kernel）。
- **grid 整除性**（TRAP-GRID-DIVISIBILITY）：persistent 调度器保证 tile 全覆盖——无手动 grid-stride 的静默漏写风险；host assert 防御契约误用（§4.7）。

### 5.5 分核策略（grid 划分 + AIC/AIV 分工 + 同步策略）⭐

> 三要素判定标准与核数实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（依据 `.agents/skills/tilelang-ascend/SKILL.md` §2/§3/§7）。

- **grid 划分（含核数依据）**：`num_blocks = NUM_BLOCKS = 56`（一维 `T.Kernel(56)`，persistent 单波常驻）；行块任务 `num_tiles = ceil(M/block_m)`（wl1: 64、wl4: 32）由 `T.Persistent([num_tiles], 56, core_id, num_stages=2)` 调度器分配（多波：64/56 ≈ 1.14 tiles/核 + 尾波 8 tiles）。**核数实查记录**（2026-10-16，本设计执行期亲测）：
  ```python
  import torch
  props = torch.npu.get_device_properties()
  # name: Ascend950PR_9579, cube_core_num: 28, vector_core_num: 56
  #       total_memory: 132338679808 (~126GB), L2_cache_size: 134217728 (128MB)
  ```
  纯 Vector 算子按 `vector_core_num = 56` 取 NUM_BLOCKS（persistent 单波常驻，无超发串行）；cube_core_num=28 记录备查（本算子不占 AIC）。无真机环境的复现命令即上述查询代码；本仓 examples 惯例初值 32 blocks 被实测值 56 取代（`CONST-arch-a5` measured 条目：28 = 4×7 非 2 幂——grid 整除性陷阱见 TRAP-GRID-DIVISIBILITY，本设计 persistent 形态免疫）。
- **AIC/AIV 分工**：纯 Vector 算子——逐元素（cast/exp/sub/log）、行归约（`T.reduce_max/reduce_sum` 硬件树）、数据搬运（`T.copy` MTE2/MTE3）全部落 **AIV**（56 个）；**无 Cube 工作**（无 `T.gemm`/L1/L0 段）→ 不占 AIC（28 个闲置）。pure kernel 的核型由 codegen 自动判定（core-split-strategy §1 ②），无需 `T.MixedKernel`/sid 声明；无 dual_copy 所有权问题（无 L0C→UB 段）。
- **同步策略（auto|manual 声明）**：**auto_schedule: yes**（默认）——依赖 AutoSchedule 自动插 flag/同步，数据依赖边界：① x_ub 版本 v 的 MTE2 写 → SimtVF 读（跨引擎，自动 flag 对）；② lse_ub 的 VALU 写（fragment→UB `T.copy`）→ MTE3 读（跨引擎，自动 flag 对）；③ x_ub 版本 v/v⊕1 交替（multibuffering，`T.annotate_buffer_versions({x_ub: 2})` + 编译器管理）；④ SimtVF 内跨线程 UB 读写 hazard → 自动 `asc_syncthreads`（tilelang-ascend skill §2.2/§10.8，回归 `testing/ascend/transform/test_ascend_thread_sync.py`）；⑤ reduce 终值 → lse 消费的 VALU 内顺序依赖（同引擎顺序语义）。无手动 `T.ascend_set_flag/wait_flag`（核内 8/跨核 16 flag 预算不占用）；调度提示使用位置：`T.Persistent(num_stages=2)`（子批间流水）——不使用 `T.Stage`/`T.Task`（AutoSchedule 足以覆盖本算子依赖结构）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表仅 block 级/tile 级/子批级调度循环；无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 行块（M 向） | persistent 调度 | `T.Persistent([num_tiles], 56, core_id, num_stages=2)` | 行块在 56 AIV 间持久调度 + 子批间流水 |
| 行子批（block_m 内） | 串行迭代 | `T.serial(num_sub)`（全块）/ `T.serial(tsub)` + 静态末子批（尾块） | x_ub 双版本交替的流水顺序依赖（§1.6.2 #8） |
| 尾块分支 | 运行时控制流 | `if t < full: … else: …`（TIR if） | M 尾块边界语义（§1.6.2 #9；example_gemm L47 先例） |
| 元素级（N 向） | 向量化 | `T.Parallel(SUB, TILE)` ×2（载入 / exp 原位）+ `T.Parallel(SUB)`（lse+cast） | §1.6.2 #1/#3/#5 全向量化 |
| 行内归约 | 硬件归约 | `T.reduce_max` / `T.reduce_sum`（dim=1, clear=True） | §1.6.2 #2/#4（AscendAllReduce 硬件树） |

### 6.2 循环伪代码

```python
# Block 级并行（一维 grid，56 persistent 核；核内结构见 §3.3 完整伪代码）
with T.Kernel(56) as core_id:
    x_ub = T.alloc_shared((SUB, N), dtype)
    lse_ub = T.alloc_shared((SUB,), dtype)
    T.annotate_buffer_versions({x_ub: 2})
    for t in T.Persistent([num_tiles], 56, core_id, num_stages=2):
        base = t * block_m
        if t < full:
            for s in T.serial(num_sub):          # 子批双缓冲：s+1 预取 ⊥ s 计算
                <子批体：T.copy GM→UB → SimtVF{masked 载入 → reduce_max →
                 exp 原位 → reduce_sum → lse+cast → T.copy frag→UB} → T.copy UB→GM>
        else:
            for s in T.serial(tsub):
                <子批体（rows=SUB）>
            if tsub_rem > 0:                     # Python 静态分支
                <子批体（rows=tsub_rem，行谓词 masked，静态 rem 切片 copy）>
```

### 6.3 流水线优化

`T.Persistent(num_stages=2)` + `T.annotate_buffer_versions({x_ub: 2})`：子批 s+1 的 GM→UB 拷贝（MTE2，版本 v⊕1）与子批 s 的 SimtVF 计算（版本 v）重叠；tile 间由 persistent 调度器连续供给。跨引擎依赖（MTE2→VALU→MTE3）由 AutoSchedule 自动插 flag 对（§7.2）；不手动管理双缓冲（tilelang-ascend skill §4.2：优先编译器 multibuffering）。**对照基线（源直译单版本 x_ub）**：MTE2 与 VALU 串行——流水收益 ≈ 流量项隐藏（wl1：−1.5K cycle ≈ −8%，D-2 估算）。

### 6.4 尾块处理

- 行尾（M % block_m ≠ 0）：静态 full/rem 分解 + TIR if + 静态切片 + 行谓词（§5.4；§0.6 R4 语义保持论证）。
- 列尾（N % TILE ≠ 0）：masked 载入（pad=−inf 三幂元中性，V1 位级验证）——无数值影响、无 OOB。
- 子批尾（block_m % SUB ≠ 0 或 rem % SUB ≠ 0）：全块路径末子批行数 = block_m − (num_sub−1)·SUB（静态）；尾块路径末子批 tsub_rem（静态）——两者均由 host 静态化推导切片与谓词。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: auto_schedule: yes（AutoSchedule 自动同步；未设 `TL_ENABLE_AUTO_SCHEDULE: False`）

### 7.2 同步点说明

auto 模式下依赖边界说明（无手动同步点表；AutoSchedule 自动插入）：

| 依赖边界 | 涉及引擎/buffer | 自动机制 |
|----------|-----------------|----------|
| x_ub 写（MTE2，版本 v）→ SimtVF 读 | MTE2 → VALU，x_ub | AutoSchedule flag 对 |
| lse_ub 写（VALU，`T.copy(out_frag, lse_ub)`）→ MTE3 读 | VALU → MTE3，lse_ub | AutoSchedule flag 对 |
| x_ub 版本交替 v / v⊕1 | multibuffering | `T.annotate_buffer_versions({x_ub: 2})` + 编译器 |
| SimtVF 跨线程 UB hazard（x_ub 共享读） | VALU 线程域 | 自动 `asc_syncthreads`（skill §2.2；回归 test_ascend_thread_sync.py） |
| reduce 终值 → lse 消费 | VALU 内 | 同引擎顺序语义 |

**风险提示**（§9.1 风险 3）：`T.Persistent + TIR if（尾块分支）+ T.serial（子批）+ SimtVF` 的嵌套组合无单一先例覆盖全部嵌套层级（各层分别有先例：online_softmax 的 Persistent+serial+SimtVF；example_gemm 的 Persistent+TIR if）——若 AutoSchedule 对 if 分支两侧的 multibuffering 版本推断异常，fallback = 拆双 persistent 域（全块域 + 尾块域 [1] + `core_id` 谓词），语义等价、API 全先例（§0.6 R4 预案）。

### 7.3 pass_configs 配置

```python
# 默认自动调度，不传 TL_ENABLE_AUTO_SCHEDULE（仅完全手动核需要 False）；
# JIT 入口即 §4.7 的 @tilelang.jit(target="ascend", out_idx=[1]) 懒工厂。
```

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（源仓测试基准 `torch.logsumexp(x.float(), dim).to(x.dtype)`——`tests/ops/test_softmax.py` L99 `ref_program` 直移植），**不复刻 §0.6 的 NPU 算法**——保证验证独立性。

```python
def golden_logsumexp(x: torch.Tensor) -> torch.Tensor:
    """Row-wise logsumexp, torch on NPU（源仓 ref_program 直移植；fp32 计算域，
    输出 cast 回输入 dtype——与 §0.1 dtype 语义一致）。"""
    assert x.device.type == "npu" and x.ndim == 2
    return torch.logsumexp(x.float(), dim=-1).to(x.dtype)
```

```python
# L0 精度对比语句（主 shape 门槛；randn 数据域——±inf 分歧角点不在 L0 域）：
y = _logsumexp_kernel_single(M, N, dtype)(block_m)(x)
torch.npu.synchronize()
golden = golden_logsumexp(x)
atol = rtol = 1e-2 if dtype in (fp16, bf16) else 1e-3
torch.testing.assert_close(y, golden, rtol=rtol, atol=atol)
```

### 8.2 精度标准与分层测试计划（L0 门槛；L1/L2/Boundary 完整套件由 tilelang-op-develop 落地）

**容差分档**（AGENTS.md / examples/TileOPs/docs/gpu_to_ascend_adaptation.md §1.8）：fp16/bf16 → atol=rtol=1e-2；fp32 → 1e-3（本算子 fp32 中间累加 + 单次归约，非 accumulation-heavy，取 1e-3 档）。

| 层级 | 用例设计（依据：manifest workloads + test_softmax.py 落点分析 §0.2） |
|------|----------------------------------------------------------------------|
| **L0（门槛）** | ① wl1 等价 (1024, 4096) fp16 block_m=16（bench 主对象）；② wl4 等价 (128, 16384) fp16 block_m=4；③ smoke (32, 256) 三 dtype（fp32/fp16/bf16）；④ (33, 256) fp16（M 尾块 rem=1，bm=16）；⑤ (32, 300) fp16（非 pow2 N，TILE=512 masked） |
| **L1（shape/config 扫描）** | M ∈ {1, 3, 33, 56, 57, 128, 1024} × N ∈ {1, 7, 256, 300, 4096, 16384} 三 dtype；block_m ∈ {1, 2, 4, 8, 16}（wrapper 启发式全集 + 极端 1）；SUB 分裂点验证（block_m > SUB 与 block_m = SUB 两形态）；M=1（全尾块）；1D 输入 flatten (1, 256)；多维 dim=[0,2] flatten (128, 16384) |
| **L2（稳健性/压力）** | 大 M 多波（M=8192, N=4096，num_tiles=512 > 56 多波调度）；契约边界 N 邻域（N=32512 fp16 SUB=1 退化形态，若构造 block_m 使驻留成立：block_m=2 → 2·2·32512·2 = 128KB ✓）；切换边界记录（N_padded > 32512 属 tiled 变体——不在本 kernel 测试域，负向 assert 验证） |
| **Boundary（角点）** | ① 有限域角点（对照 golden）：全零行 / 全等行 3.5 / 单主导 100 / ±65504 极值（fp16）/ subnormal ±6e-8 / 深度下溢 −745 / N=1 / M=1 / 确定性（3× 位级一致）；② 部分 −inf 正常行（golden 一致 ✓ V3b）；③ NaN 行（golden 双方 NaN，equal_nan=True 对照 ✓ V3b）；④ **源继承分歧角点**（对照**源算法 fp64 语义模拟**而非 golden，迁移保真验证）：全 −inf 行 → NaN（torch 为 −inf）、含 +inf 行 → NaN（torch 为 +inf）——V3b/V4 已证链与源位级一致；⑤ 非连续输入（op 层 contiguous 化后等价） |

**已知边界分歧处理**（§9.3 详述）：全 −inf / 含 +inf 行为源算法固有（`x − m` 产生 NaN 链），与 `torch.logsumexp` 分歧；源仓测试域（randn 生成）不触发（`test_softmax.py` 全部 `torch.randn` / `torch.zeros` 基底），本设计保持源语义（迁移保真）并在 Boundary 用源语义模拟对照证明行为一致。

### 8.3 性能验证（Stage 4，设计期仅记录口径）

- 测量权威：`tilelang.profiler.do_bench(fn, backend="msprof")`；代表 shape = wl1 (1024, 4096) fp16 / wl4 (128, 16384) fp16（bench_softmax.py manifest 驱动）。
- 基线对照：`torch.logsumexp(x, dim=-1)`（native op，bench_softmax.py 的 baseline_fn 同款）。
- A/B 必做项：§1.6.4 fp16-e 备选裁决（阈值与回写路径见该节）。

---

## 9. 风险点与注意事项

### 9.1 已知约束（本算子相关）

| # | 风险 | 影响 | 缓解 / 验证 |
|---|------|------|------------|
| 1 | **VRF 容量假设**（`CONST-vrf-capacity-a5` assumption：256KB 起步，无文档化常数） | SUB 封顶后 x_frag ≤128KB（≈44 reg/thread）远低于 76 reg 实测水位——理论安全，但假设本身未标定 | Stage 3 编译后首查 spill 计数（`get_kernel_source()` / 编译日志）；若 spill > 0 降 SUB 一档（host 静态化参数，kernel 结构不变） |
| 2 | **N 非 32B 行对齐的 DMA 带宽档**（N=300 fp16 = 600B/行） | 正确性 ✓（字节级行长，skill §6.1）；带宽档降级（`CONST-mte-bandwidth-model` 平台/未知档 50/48 B/cycle vs 峰值 102） | 流量项本就非主导（计算项 ~11×，D-2）；Stage 4 msprof 归因确认；非对齐 workload 不在 bench 主对象（wl1/wl4 均对齐） |
| 3 | **Persistent + TIR if + T.serial + SimtVF 嵌套组合无单一先例**（各层分别有先例：online_softmax 的 Persistent+serial+SimtVF；example_gemm 的 Persistent+TIR if） | AutoSchedule 对 if 分支两侧 multibuffering 版本推断异常的可能性（未证实） | §0.6 R4 fallback 预案：拆双 persistent 域（全块域 [full] + 尾块域 [1] + core_id 谓词），语义等价、API 全先例；Stage 3 首个尾块用例（L0 ④ (33,256)）即触发该路径验证 |
| 4 | **±inf 边界与 torch golden 分歧**（全 −inf 行 → NaN vs torch −inf；含 +inf 行 → NaN vs torch +inf） | 源算法固有语义（迁移保真项）；若未来集成测试加入此类角点将对照 fail | §8.2 Boundary ④ 以源语义模拟对照（迁移保真）；§9.3 文档化；源仓测试域（randn）不触发——Stage 5 集成跑 test_softmax.py 无此角点 |
| 5 | **special_func 8 ops/cycle 的 op 粒度未区分**（element vs 向量指令级，`estimate_latency.h` L58–70 未注明） | D-2 估算的瓶颈归因（计算主导 ~11×）可能反转为 MTE 主导 | 两种解释下主选结构均为最优（§1.6.0 披露）；Stage 4 npusim/msprof 实测归因，回写 `CONST-valu-model` |
| 6 | **bf16 精度链** | bf16 输入 fp32 中间（主选）无风险；fp16-e 备选**不适用 bf16**（S 树形预估 ~1.4e-2 超容差） | §1.6.4 分派条件 `dtype == "float16"` 已排除 bf16；主选链 bf16 走 fp32 全程 |
| 7 | **block_m 契约外值**（显式 config 传 >16 或非 2 幂） | SUB 封顶机制兜底任意 block_m（§5.2），但极端大 block_m 使 num_sub 增大（子批串行长度↑） | 性能项非性能正确性项；wrapper default_config 不产生此类值；host assert 记录契约（§4.7） |
| 8 | **wrapper 预算口径变化**（`device_smem_budget` 返回值若从 216KB 改为 248KB 物理） | single 变体契约边界移动（更大 N/block_m 组合进入） | kernel 侧 assert（N ≤ 32512）+ SUB 双约束自适应（公式不依赖 wrapper 预算数值）；契约边界以 wrapper 实际判定为准 |

### 9.2 常见错误（实现期检查清单）

- ❌ `T.Kernel(grid, threads=…)`（二维/threads 参数）——TypeError（ascend-constraints §1 条 1）。
- ❌ `tilelang.compile(prog, target="ascend")` 直接形态——S3-JIT 门禁拒（CASE 教训）；用 `@tilelang.jit(target="ascend", out_idx=[1])` 懒工厂。
- ❌ fragment 非 pow2 形状（如 (rem, TILE) 直接用 rem）——alloc_fragment 需 pow2（`example_rmsnorm.py` L42）；尾块行数用谓词而非变长 fragment。
- ❌ 尾块 `T.copy` 越界切片（`[base : base+SUB]` 超 M）——A5 DMA 越界行为未文档化；必须静态 rem 切片（§0.6 R4）。
- ❌ `pad_value=`/`data_select=` 用于行 OOB——二者是**列向**行尾 padding（skill §6.1），行 OOB 用谓词 + 静态切片。
- ❌ 全局 PYTHONPATH 污染——所有 python 命令前置 `PYTHONPATH=/home/z00875089/tilelang-fork`。
- ❌ grid=32 惯例值——本机实查 56（`CONST-arch-a5` measured；28=4×7 整除性陷阱，persistent 形态免疫）。
- ❌ 逐元素标量循环（载入/exp/lse）——§1.6.2 已全向量化，§6 不得回退。

### 9.3 特殊场景处理（边界语义文档）

**与 torch golden 的两处分歧（源算法固有，迁移保真保持）**：

| 场景 | 本 kernel（=源 GPU 算法） | `torch.logsumexp` | 数学注记 | 测试域 |
|------|---------------------------|-------------------|---------|--------|
| 全 −inf 行 | **NaN**（m=−inf → x−m=NaN → e=NaN → S=NaN） | −inf | 解析极限为 −inf；torch 特判，源算法按 IEEE 直传 NaN | randn 域不触发；Boundary 以源语义模拟对照（V4 位级一致） |
| 行含 +inf | **NaN**（inf−inf=NaN 污染求和） | +inf | 解析主值为 +inf；同上 | 同上 |
| 行含 NaN | NaN（逐元素传播） | NaN | 一致 ✓ | equal_nan 对照 |
| 部分 −inf 正常行 | m 有限 → exp(−inf−m)=0 中性 | 一致 | 一致 ✓ | 直接对照 |
| fp16 深度下溢（−745） | S 下溢至最小可表示 → log(S) 有限/−inf 边缘 | fp32 域不受影响 | 输出 cast fp16 后差异在容差内（V3 角点覆盖） | Boundary ① |

**设计取舍说明**：选择保持源语义而非「对齐 torch」（如哨兵钳位 m ≥ −65504 使全 −inf 行输出 −inf——tiled 变体的 flash 形式已有此机制）——理由：① 迁移保真优先（single 与 tiled 两变体行为一致性由源仓定义）；② randn 测试域不触发；③ 引入钳位反而制造两变体间的语义分裂（single=−inf、tiled 原版=NaN——源仓 tiled 无钳位）。若 Stage 5 集成后需要与 torch 全角点对齐，属算子族级语义变更，应回归源仓统一处理（本设计记录不在单变体内擅自变更）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/logsumexp/_logsumexp_kernel_single/
├── DESIGN.md                  # 本设计文档
├── verify_equiv.py            # D-1 等价性机器验证（已执行，全 EQUIV_PASS）
├── _logsumexp_kernel_single.py  # Stage 3：kernel 实现（@tilelang.jit 懒工厂 + golden + 分层测试）
├── REVIEW.md                  # Stage 2：设计检视报告（另生成）
└── history_version/           # 设计期探针（如有）/ 历史版本备份
```

### 10.2 文件清单（Stage 3 实现输入）

| 文件 | 内容 | 契约要点 |
|------|------|---------|
| `_logsumexp_kernel_single.py` | kernel + golden + L0/L1/L2/Boundary 分层测试 | 调用面 `_logsumexp_kernel_single(M, N, dtype)(block_m)(x)`；docstring 声明 `programming_mode: simt` + `auto_schedule: yes`；`@tilelang.jit(target="ascend", out_idx=[1])` 懒工厂；golden = `torch.logsumexp(x.float(), dim=-1).to(x.dtype)`（NPU 上计算） |

### 10.3 命名与声明规范

- 模块 docstring 必含：`programming_mode: simt`、`auto_schedule: yes`（迁移 prompt 契约）。
- 函数名/参数名与源保持（`_logsumexp_kernel_single`、`M/N/dtype/block_m`）；`threads` 参数按 K3 移除（内部 `min(1024, SUB·TILE)` 自推导）。
- host 静态化变量命名：`TILE/SUB/num_sub/full/rem/num_tiles/tsub/tsub_rem/threads/NUM_BLOCKS`（§3.3 注）。

### 10.4 实现顺序建议（Stage 3）

1. 懒工厂骨架 + host 静态化（含契约 assert）→ 2. 全块路径（无尾块 shape 先通：L0 ①②③）→ 3. 尾块路径（L0 ④⑤：TIR if + 静态切片 + 行谓词）→ 4. `get_kernel_source()` 首查 spill 计数（风险 1）→ 5. L1 扫描 → 6. L2/Boundary（源语义对照项）→ 7. 四出口判定。

### 10.5 本设计引用的外部工件

| 工件 | 用途 |
|------|------|
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py` | 源算子（三问解读对象） |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp.py` | wrapper dispatch / Kernel class（契约 shape 空间推导） |
| `examples/TileOPs/tileops/kernels/reduction/_primitives.py` | `MAX_SINGLE_TILE_COLS` / `compute_tile_n` / `device_smem_budget`（single 选中条件） |
| `examples/TileOPs/tileops/ops/reduction/logsumexp.py` | op 层（dim 归一化 / flatten / golden 基准 ref_program） |
| `examples/TileOPs/tileops/manifest/reduction.yaml` | workloads / roofline 口径（4·M·N FLOPs、(M·N+M)·elem bytes） |
| `examples/TileOPs/tests/ops/test_softmax.py` / `benchmarks/ops/bench_softmax.py` | 精度 / 性能用例（single 落点分析 §0.2） |
| `examples/pilot/online_softmax/`（DESIGN.md / online_softmax.py / verify_equiv.py） | 同族 row-reduction 迁移正例（结构 / JIT 形态 / 验证组织参照） |
| `examples/ascend/example_{rmsnorm, per_token_cast_to_fp8, gemm}.py`、`flash_attention/core.py` | API 佐证（§3.4 表） |
| pattern-library（`elementwise.md` D-REDUCE-SIMTVF-API / D-ELEM-SIMTVF-AUTOVEC、`traps-runtime.md` TRAP-GRID-DIVISIBILITY / TRAP-NAIVE-ONLINE-MERGE、`constants.md` CONST-* 五条目） | 实测模式与硬件常数（版本戳 2026-10-16） |
