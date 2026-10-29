# _logsumexp_kernel_single 算子设计文档（GPU TileOPs → Ascend A5 迁移，**SimdVF 独立变体**）

> **任务上下文**：迁移 TileOPs `LogSumExpFwdOp` 的 **SIMD 模式独立变体**（新算子 `LogSumExpFwdOpSimd` / `logsumexp_simd`，GPU 源仓 `/home/z00875089/TileOPs-fork`）。源算子三问解读对象 = `tileops/kernels/reduction/logsumexp.py` 中的 `_logsumexp_kernel_single`（单 tile 变体：N 可整体驻留片上；GPU 源 `N_padded = align_up(N, 256)`，masked load 处理非对齐 N 边界）。多 tile 变体 `_logsumexp_kernel_tiled`（N 超驻留上限的 online 流式路径）由独立任务设计，不在本文范围。
> **编程模式契约**：**simd（SimdVF）——用户显式指定**，非默认 simt。本设计全部计算以显式 `T.simd.*` 2048-bit 向量内在表达（`tilelang/ascend/language/simd.py`），kernel 模块 docstring 须声明 `programming_mode: simd`、`auto_schedule: yes`。
> **职责边界（严禁改动项）**：现有 SIMT 版本 `examples/TileOPs/tileops/kernels/reduction/logsumexp/`（含 perf_opt/）与 `ascend::logsumexp_fwd` custom_op **不属于本任务产物范围**；本设计产物只落在 `examples/logsumexp_simd/_logsumexp_kernel_single/`。SIMT 版仅作为同族先例引用（先例 ≠ 源算子——三问解读对象始终是 GPU 源码）。
> **调用面契约**：A5 侧保持 `_logsumexp_kernel_single(M, N, dtype)(block_m)(x) -> y` 调用面（与 SIMT 兄弟变体一致；GPU `threads` 参数按 K3 已移除，SimdVF 模式下连内部 threads 推导也不存在——无线程概念）。

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

**数学语义**（逐行 log-sum-exp，数值稳定形式）：

$$
y_i \;=\; \mathrm{logsumexp}_j(x_{ij}) \;=\; m_i + \log\Big(\sum_{j=0}^{N-1} \exp(x_{ij} - m_i)\Big), \qquad m_i = \max_j x_{ij}
$$

与解析式 $\log\sum_j e^{x_{ij}}$ 数学恒等（减 max 平移不变性），是数值稳定化标准形态（algorithm-candidates.md `ALG-softmax` R1 必选项）。

**语义要素表**（迁移必须原样保持的部分）：

| 要素 | 结论 | 源码证据（`tileops/kernels/reduction/logsumexp.py`） |
|------|------|---------|
| 数学映射 | 行级 logsumexp：`y[i] = row_max[i] + log(row_sum[i])`，`row_sum = Σ exp(x − row_max)` | L85–94 |
| I/O 契约 | 输入 `x: (M, N)` 行主连续、dtype ∈ {float16, bfloat16, float32}（op 层允许三者；manifest workload 为 fp16/bf16）；输出 `y: (M,)` 同 dtype | kernel 签名 L58–61；`_logsumexp_kernel_single` docstring L44 |
| 规约语义 | 沿最后一维（N）整行归约；**中间计算全程 fp32**（`x_f32` fragment、"float32" 的 row_max/row_sum），输出 cast 回输入 dtype；累加顺序 = 树形归约（`T.reduce_sum`），并行分组变化只引入 fp32 树深级舍入差 | L64–66、L85–94 |
| dtype 语义 | 输入 dtype → fp32 中间（`T.cast(x[...], "float32")`）→ 输出 dtype（`out_local` 为输入 dtype fragment，`T.copy` 写回） | L73–77、L91–93 |
| 边界语义 | ① **NaN 逐元素传播**：行含 NaN → m 或 e 任一 NaN → S=NaN → y=NaN；② **全 −inf 行 → y=NaN**（m=−inf → x−m=NaN——源算法固有行为，与 `torch.logsumexp` 的 −inf **分歧**，见 §9.3）；③ **含 +inf 行 → y=NaN**（inf−inf=NaN 污染；torch 返回 +inf，同上分歧）；④ 部分元素 −inf 的正常行：−inf−m=−inf → exp=0，和中性 ✓；⑤ M 尾块（M % block_m ≠ 0）行 OOB 以 −inf 载入、不写出 | L72–77（`T.And(pid_m*block_m+i < M, j < N)` 掩码）；L85–94 |
| 尾块/非整除 | N 非 256 倍数时 `_needs_pad=True` 走逐元素 masked 载入（列 j<N、行 <M，越界填 −inf）；N 对齐时走 `T.copy` 快路径。列 pad 与行 pad 的 −inf 同时是 max 幂元与 exp 后 sum 幂元（数值中性） | L50–52、L68–83 |
| host 侧语义 | `N_padded = align_up(N, 256)`、`_needs_pad` 判定、`_neg_inf` 字面量均在 builder 闭包内（JIT per-shape 编译的一部分）；wrapper 层负责 dim 归一化/transpose/contiguous/reshape 到 (M, N) 与输出 reshape——**不属于本 kernel 的契约** | L50–56；`logsumexp.py` wrapper L193–207 |

**语义解读完成标准自检**：依据上表可直接写出与源算子数值一致的 golden（§8.1 的 `torch.logsumexp(x.float(), dim=-1).to(x.dtype)`，即源仓测试基准 `ref_program` 直移植）；±inf 角点的源继承分歧已显式列出（§9.3）。

### 0.2 源算子输入输出（含 single 变体的契约 shape 空间）

**Kernel 签名**：`_logsumexp_kernel_single(M: int, N: int, dtype: str)` → `_func(block_m)` → `(x) → y`（`@tilelang.jit(out_idx=[1])`）。

**single 变体被 wrapper 选中的条件**（`logsumexp.py` `_tile_n_for_block_m` L338–360 + `_primitives.py` L30/L39，决定本 kernel 的契约 shape 空间）：

1. `N_padded = align_up(N, DEFAULT_ALIGNMENT=256) ≤ MAX_SINGLE_TILE_COLS = 32512`（列数上限；GPU 侧为 TileLang 向量化器 32768 边界的安全回退值）；
2. `block_m × elem_bytes × N_padded ≤ device_smem_budget()`（GPU 侧 SMEM opt-in 预算）。

**manifest workload 落点**（`examples/TileOPs/tileops/manifest/reduction.yaml` L124–172，`LogSumExpFwdOpSimd`，归约到 (M, N) 后；本 SIMD 家族的变体分界沿用同一契约）：

| workload | 归约后 (M, N) | dtype | 落入变体 |
|----------|--------------|-------|---------|
| attn-weights-4k `[32,32,4096]` dim=-1 | (1024, 4096) | fp16 / bf16 | **single（本设计）** |
| attn-weights-32k `[32,32,32768]` dim=-1 | (1024, 32768) | bf16 | N_padded=32768 > 32512 → **tiled（另任务，不在本文范围）** |
| lm-head-logits `[4,102400]` | (4, 102400) | fp16 / bf16 | N_padded > 32512 → **tiled（另任务）** |
| 3d-multidim-reduce `[4,128,4096]` dim=[0,2] | (128, 16384) | fp16 | **single（本设计）** |

精度用例 `tests/ops/test_logsumexp_simd.py` / 性能用例 `benchmarks/ops/bench_logsumexp_simd.py`（manifest `source` 段）；wrapper 侧 GPU 上下文（`LogSumExpKernel` class / dispatch 启发式 `default_config` block_m ∈ {1,2,4,8,16} 偏好 single 最大值）与 SIMT 家族共用契约口径。**契约空间**：M ≥ 1 任意（含 M < block_m、M % block_m ≠ 0）；N 满足上两条；dtype ∈ {fp16, bf16, fp32}；block_m ∈ [1, 16] 常见（显式 config 理论上可更大，本设计以 SUB 封顶机制兜底任意 block_m，§5.2）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解表**（源 GPU kernel 每条计算语句归入一步，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式 | 源码行 |
|------|------|------|------|-------------|--------|
| S0 | `N_padded = align_up(N, 256)`；`_needs_pad = N_padded != N` | N | 编译期常量 | 边界处理选择 | L50–52 |
| S1a | masked 载入：`x_f32[i,j] = if(行<M ∧ j<N, cast(x), −inf)` | x (GM) | x_f32 fragment (block_m, N_padded) fp32 | 数据就位 + 边界中性化 | L72–77 |
| S1b | 对齐快路径：`T.copy(x → shared_buf → x_local)` 再 cast fp32 | x (GM) | x_f32 | 同上（N 对齐时） | L79–82 |
| S2 | 行最大：`T.fill(row_max, −inf)` + `T.reduce_max(x_f32, row_max, dim=1, clear=False)` | x_f32 | row_max (block_m,) fp32 | m_i | L84–85 |
| S3 | 稳定化指数：`x_f32[i,j] = exp(x_f32[i,j] − row_max[i])`（**原位覆写**） | x_f32, row_max | x_f32（语义上为 e_ij） | e_ij | L87–88 |
| S4 | 行求和：`T.reduce_sum(x_f32, row_sum, dim=1)` | x_f32(e) | row_sum (block_m,) fp32 | S_i | L89 |
| S5 | 输出：`out_local[i] = row_max[i] + log(row_sum[i])`（fp32 算式，fragment 为输入 dtype） | row_max, row_sum | out_local (block_m,) | y_i | L91–93 |
| S6 | 写回：`T.copy(out_local, y[pid_m*block_m])` | out_local | y (GM) (M,) | 输出 | L95 |

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

**循环与并行结构**（源）：grid = `T.ceildiv(M, block_m)` 个 block（`T.Kernel(grid, threads=threads)`，**二维参数 + threads**——A5 不支持，见 §0.5），每 block 处理 block_m 行；block 内 `T.Parallel(block_m, N_padded)` 线程级并行（threads=256 默认）；无块间通信（行独立）。

**host 侧逻辑**：`align_up` 常量折叠（编译期）；`_neg_inf` 字面量；`threads` 由 wrapper 传入（GPU 侧契约）；JIT per-(M, N, dtype) 编译 + `lru_cache`。

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

**未识别更优手段的风险自检**：源码无 warp shuffle 手写、无 swizzle、无 cp.async/Pipelined、无 persistent——识别已覆盖全部显式结构；`threads` 的行内并行组织归入 #5/#7 的实现细节。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（三层模型：语义层/算法层/优化层；四态处置：保留/等价替换/重新设计/舍弃。判定输入含 §1.6.0 调研结论——源算法与调研候选同台比较；本设计的承载模式为 **SimdVF**（用户显式指定），故「源线程级并行如何落到 VF」是本变体特有的判定维度。）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（行 logsumexp、fp32 中间、NaN/±inf 边界行为） | 语义 | — | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| 减 max 稳定化（#3） | 算法 | 无 | — | **保留** | 纯数学（`ALG-softmax` R1 必选项）；调研确认 e-base 直写即 A5 最优形态（`CONST-valu-model` 注记：exp/exp2 同档 8 ops/cycle） |
| 单 tile 全驻留（#1） | 算法 | 无（容量数值硬件相关） | 有（SimdVF UB 248KB 重算预算——**无 SimtVF 32KB 线程上下文扣减**，`CONST-capacity-a5`） | **保留（驻留层重算）** | 契约空间由 wrapper 判定（§0.2）；驻留层 SMEM→UB 等价替换 + **VRF 流式化**（无 fragment 驻留，§0.6 R2） |
| exp 原位覆写（#4） | 优化 | GPU 寄存器堆 | **SimdVF 形态下无对应需求**（本变体 x 不驻留 VRF，两遍经 UB 重读，§0.6 R2） | **舍弃（形态变更，意图有承接）** | 意图「省一半寄存器驻留」由「零行数据寄存器驻留」（流式累加器 ~10 向量寄存器）超量承接；代价 = pass 2 重读 UB（D-2 量化：+0.016 cycle/elem，非瓶颈） |
| 树形归约 `T.reduce_max/reduce_sum`（#5） | 优化 | warp shuffle 树 | 有（SimdVF 跨 lane 全归约内在 `vcmax`/`vcadd`，`simd_inst.h` L688–710 `asc_reduce_max/asc_reduce_sum`） | **等价替换** | `examples/ascend/flash_attention/core.py` L115–118/L134–137（vcmax/vcadd + ONEPT 存储实测在用）；`examples/ascend/example_per_token_cast_to_fp8.py` L142–146（vcmax + vdupv）；fp32 累加分组变化 → verify_equiv.py V1/V2 实测 ≤2 ulp |
| 线程级逐元素并行（`T.Parallel(block_m, N_padded)` + threads=256，#5/#7 实现细节） | 优化 | CUDA 线程模型 | **无线程概念等价**（SimdVF = 无线程 2048-bit 向量指令流，tilelang-ascend skill §2.3） | **重新设计** | → §0.6 R5：显式 `T.simd.*` 指令流（`vld`/`vcvt`/`vmax`/`vexpdif`/`vadd`/`vcmax`/`vcadd`/`vln`/`vsts`），64 fp32 lane / 128 fp16 lane 粒度 |
| N 256 对齐快慢双路径（#2） | 优化 | GPU `T.copy`→SMEM 256 元素对齐 | **无此约束**（A5 GM→UB DMA 字节级行长；向量内在只需 64 元素组边界） | **重新设计** | → §0.6 R2：统一为「UB (SUB, N_pad=align64) + fp32 域组尾 −inf 掩码（`vsel`+`update_mask`）」单路径——满组直载 + 尾组掩码二分支（`P-LAYOUT-TAIL-STATIC-EXTENT` measured 同构） |
| `T.copy` SMEM staging（x→shared→local→fp32，#1/#2 实现细节） | 优化 | GPU SMEM 两跳 | 有（`T.copy` GM→UB 单跳 + `vld(UNPK_B16)`+`vcvt` 向量装载） | **等价替换** | `testing/ascend/language/test_tilelang_ascend_simdvf_cast.py` L64–80（UNPK_B16 + vcvt(part=0) 位级回归）；`examples/ascend/example_per_token_cast_to_fp8.py` L129/L138（GM→UB + SimdVF vld） |
| block×thread 两级并行（`T.Kernel(grid, threads=threads)`，#7） | 优化 | CUDA 线程模型 | **无直接等价**（`T.Kernel` 一维、无 `threads=` 参数——TypeError） | **重新设计** | → §0.6 R1：一维 `T.Kernel(56)` + persistent 行块调度 + SimdVF 指令流（ascend-constraints §1 条 1；tilelang-ascend skill §2.1/§2.3）；K3：调用面无 threads（SimdVF 连内部推导也无） |
| 非持久 grid（ceildiv(M, block_m) blocks，#6/#7） | 优化 | GPU 硬件调度器 | 有更契合形态（persistent + 多波摊薄） | **重新设计** | → §0.6 R3：`T.Persistent([num_batches], 56, core_id, num_stages=2)` 子批粒度持久调度 + `x_ub` 双缓冲（`example_per_token_cast_to_fp8.py` L118–128 先例）；#6 行批摊薄意图由 persistent 域 + SUB 行子批承接 |
| M 尾块越界行为（对齐路径 `T.copy` 无行掩码，S1b/S6） | 实现 | GPU `T.copy` 隐式 OOB 保护（源仓 CI 通过的既成事实） | **未文档化**（A5 DMA 越界行为无保障） | **重新设计** | → §0.6 R4：静态 `full/rem` 分解 + TIR if 尾批分支 + 静态切片 copy（不依赖 A5 `T.copy` 越界行为；`example_gemm.py` L47 循环变量 if 先例；`pad_value=` 仅列向 padding 不适用——tilelang-ascend skill §6.1） |
| host `threads` 契约 | 环境 | CUDA | 无（K3：A5 移除） | **舍弃（形态变更）** | GPU threads 表达的行内并行度由 SimdVF 2048-bit 向量内在承接（每指令 64 fp32 / 128 fp16 元素）；调用面 `(block_m)` 单参数与 SIMT 兄弟变体一致 |

**判定统计**（表内 12 条目）：保留 3 项（计算语义 / 减 max 稳定化 / 单 tile 全驻留〔驻留层重算〕）+ 等价替换 2 项（树形归约→vcmax/vcadd 跨 lane 全归约、`T.copy` SMEM staging→UB 单跳 + UNPK/vcvt 装载）+ 重新设计 5 项（R1 并行结构、R2 驻留布局与双路径统一、R3 调度、R4 M 尾块、R5 线程级并行→显式 SIMD 指令流）+ 舍弃 2 项（exp 原位覆写——SimdVF 流式形态下无需求且意图被超量承接；host threads 契约——K3 形态变更）。**无静默丢弃**：每个源优化意图均有承接结构或舍弃理由。

### 0.6 NPU 算法重设计

**重设计项 R1: 并行结构（block×thread 两级并行 → 一维 grid + persistent 子批调度）**

- **源方案**：`T.Kernel(ceildiv(M, block_m), threads=threads)`——block_m 行/block × threads 线程（行内元素并行），非持久 grid。
- **NPU 新算法**：一维 `T.Kernel(NUM_BLOCKS)`，`NUM_BLOCKS = 56`（= `vector_core_num` 实查值，§5.5）；`for t in T.Persistent([num_batches], 56, core_id, num_stages=2)` **行子批粒度**持久调度（num_batches = ceildiv(M, SUB)，SUB = min(block_m, UB 预算) 的最大 2 幂，§5.2）——每 persistent 迭代 = 一次 (SUB, N) DMA + 一个 SimdVF 计算块。
- **语义保持论证**：行间完全独立 → 子批大小与行到核的映射变化不影响数值；fp32 累加分组变化引入树深级舍入差（verify_equiv.py V1 实测 ≤1 ulp、违反率 0）；NaN/±inf 逐元素传播语义不变（V4 NaN 模式逐 shape 一致）。
- **依据**：`examples/ascend/example_per_token_cast_to_fp8.py` L117–129（T.Kernel(N_CORES) + T.Persistent + SimdVF 完整同构）；`examples/pilot/online_softmax/online_softmax.py` L57–63；tilelang-ascend skill §2.1/§2.3；`CONST-arch-a5`（measured，本机实查 28/56）。
- **与 SIMT 兄弟变体的结构差异**：SIMT 采用 block_m tile × SUB 子批两层（4 分支尾块）；本设计**扁平化到子批粒度**（2 分支尾块）——每 SUB 行批一次迭代，MTE2/MTE3/flag 链条数两形态相同（均为 M/SUB 条），仅去掉 tile 层包装；该形态与 pattern-library `P-LAYOUT-ROW-RESIDENCY` Stage 4 实测甜点「SUB=ROWS 单子批干净形态」一致（M=1024→ROWS=4 实测 +50.9% vs bm=16，2026-10-17）。

**重设计项 R2: 驻留布局与双路径统一（N_padded 256 对齐双路径 → UB (SUB, align64) + fp32 域组尾掩码 + VRF 流式）**

- **源方案**：`shared_buf/x_local (block_m, N_padded)`，N_padded = align_up(N, 256)（GPU `T.copy`→SMEM 对齐约束）；对齐走 bulk `T.copy`、非对齐走逐元素 masked 载入（双路径）；x_f32 fragment 全行驻留寄存器。
- **NPU 新算法**：① 驻留层 `x_ub (SUB, N_pad)` dtype = 输入 dtype、**N_pad = align_up(N, 64)**（64 元素组 = fp32 向量宽 256B / fp16-bf16 经 UNPK_B16 装载的组宽 128B——行内每组起始地址 128B/256B 对齐），双版本流水；② 计算层**无行数据寄存器驻留**：每行两遍扫描（max 遍 + exp/sum 遍）均从 UB 流式装载——16-bit 输入经 `vld(dist="UNPK_B16")` + `vcvt("float32", part=0)` 得 64-lane fp32 向量（fp32 输入直连 `vld` NORM），**尾组（N % 64 ≠ 0 时仅最后一组）在 fp32 域 `vsel(c, vdup(−inf, f32), update_mask(k, 32))` 掩码**；满组无谓词直载（单路径统一）；③ 行统计量经跨 lane 全归约内在收敛：`m_brc = vdupv(vcmax(acc_max))`、`s = vcadd(acc_sum)`。
- **语义保持论证**：pad 列从「N_padded−N 个 −inf」变为「align64(N)−N 个 −inf」——−inf 同为 max 幂元与 exp 后 sum 幂元（exp(−inf−m)=0，含 m=−inf 时 exp(NaN) 的 NaN 传播路径两链一致）→ 数学恒等；verify_equiv.py **V1 全 shape fp32 ≤1 ulp、fp64 参照一致、违反率 0**；V4（源链 vs SIMD 链 fp16/bf16 输出，含特殊行）位级一致 + NaN 模式一致。两遍重读不改变行内公式与行独立性。
- **收益**：非对齐 N（300 → N_pad 320 vs 源 512）UB 驻留 −37.5%、GM 读不变（DMA 只搬 N 列）；对齐 N 消除双路径维护；**VRF 需求从 SIMT 的 x_frag ≤128KB 降到 ~10 活跃向量寄存器 ≈ 2.5KB**（零驻留压力，SUB 封顶机制仅由 UB 预算决定）；代价 = pass 2 重读 UB（D-2：+4B/elem ÷ 256B/cycle = +0.016 cycle/elem，vs 计算项 0.203 cycle/elem——非瓶颈，13× 余量）。
- **依据**：`testing/ascend/language/test_tilelang_ascend_simdvf_cast.py` L64–80（UNPK_B16 + vcvt(part=0) 位级回归：f16→f32 / bf16→f32（经 e4m3 组合）atol=0）；`examples/ascend/deepgemm/kernels/transform_sf.py` L205/L215/L249（vsel 掩码装载先例）；`testing/ascend/language/test_tilelang_ascend_simdvf_integer_predicate.py` L261（update_mask）；`CONST-capacity-a5`（SimdVF UB 248KB）、`CONST-vrf-capacity-a5`；调研结论 §1.6.0（R3/R4）。

**重设计项 R3: 调度（非持久 grid → persistent + 双缓冲流水 + AutoSchedule）**

- **源方案**：每 block 一次性处理 block_m 行后退出（GPU 硬件调度器消化 launch）。
- **NPU 新算法**：外层 `T.Persistent([num_batches], 56, core_id, num_stages=2)`；`x_ub` 双版本（`T.annotate_buffer_versions({x_ub: 2})`）——批 t+1 的 MTE2 预取与批 t 的 VALU（SimdVF 指令流）重叠；MTE2→VALU→MTE3 跨引擎依赖由 AutoSchedule 自动插 flag 对（§7）。
- **语义保持论证**：纯调度变换（buffer 版本交替 + 同步边），不改任何数值路径。
- **依据**：`example_per_token_cast_to_fp8.py` L118–128（T.Persistent + annotate_buffer_versions + SimdVF + T.copy 三件套完整同构）；`online_softmax.py` L58–62；tilelang-ascend skill §7。
- **意图承接**：源 #6 行批摊薄 → persistent 域 SUB 行子批 DMA 合并（每批单次 2D 连续拷贝 SUB×N）；源 #7 grid 映射 → persistent 多波调度。

**重设计项 R4: M 尾块（隐式 OOB 依赖 → 静态 full/rem 分解 + TIR if 二分支）**

- **源方案**：对齐路径 `T.copy(x[pid_m·block_m, 0], shared_buf)` 与 `T.copy(out_local, y[pid_m·block_m])` 在尾 tile（M % block_m ≠ 0）时依赖 GPU `T.copy` 的越界保护行为（未文档化）。
- **NPU 新算法**：host 侧静态分解 `full = M // SUB`、`rem = M % SUB`、`num_batches = full + (rem > 0 ? 1 : 0)`（编译期常量，per-shape JIT）；persistent 循环体内 TIR if：`if t < full:` 全批路径（rows=SUB，切片 `[t·SUB : +SUB]`）；`else:` 尾批路径（rows=rem，`T.copy(x[M−rem : M, 0:N], x_ub[0:rem, 0:N])` 静态切片 + 行循环 `for r in range(rem)` + 输出 `lse_ub[0:rem] → y[M−rem : M]`）。**行内无需行谓词**——尾批只装载/计算/写出 rem 行（UB 行 [rem, SUB) 不触碰），比 SIMT 的 fragment 行谓词更简洁（无 pow2 fragment 约束）。
- **语义保持论证**：尾批有效行的公式与全批完全一致（未被装载的行不参与计算也不写出——源语义「行 OOB 以 −inf 载入、不写出」保持）；不依赖任何 `T.copy` 越界行为。
- **依据**：TIR if（循环变量条件）——`examples/ascend/example_gemm.py` L43–56（`aswt_swizzle` 内 `if row_idx < MAIN_ROW:`，A5 编译运行通过）；**SIMT 兄弟变体 `_logsumexp_kernel_single.py` L138–238 本身即 Persistent + TIR if + serial + VF 嵌套的编译运行实证**（86+61 用例全 PASS，2026-10-16）——该组合风险已从「无单一先例」降级为「同族已验证」；静态切片 copy——`online_softmax.py` L142；尾批切片形状匹配——`P-LAYOUT-TAIL-STATIC-EXTENT` measured。

**重设计项 R5: 线程级逐元素并行 → 显式 2048-bit SIMD 指令流（本变体特有）**

- **源方案**：`T.Parallel(block_m, N_padded)` 线程域逐元素并行（threads=256），`T.reduce_max/reduce_sum` 经 warp shuffle 树归约。
- **NPU 新算法**：`with T.SimdVF():` 块内**显式 `T.simd.*` 指令流**，全部计算以 2048-bit 向量内在表达（每指令 64 fp32 / 128 fp16 lane）：
  - 装载/提升：`vld(x_ub[r, g·64], dist="UNPK_B16")` + `vcvt(·, "float32", part=0)`（16-bit）/ `vld(·)`（fp32）；
  - max 遍：`acc_max = vmax(acc_max, c_g)` 逐组累加 → `m_brc = vdupv(vcmax(acc_max))`（单次跨 lane 全归约 + lane 0 广播）；
  - exp/sum 遍：`e = vexpdif(c_g, m_brc)`（**融合 exp(c−m) 单指令**，fp32 同宽）→ `acc_sum = vadd(acc_sum, e)` 逐组累加 → `s = vcadd(acc_sum)`；
  - 行尾：`out = vadd(m_brc, vln(s))` → `vcvt(·, dtype)` → `vsts(lse_ub[r], ·, pset(16,"PAT_VL1"), "ONEPT_B16")` 单点存储（fp32 输出免 cast、ONEPT_B32）。
- **语义保持论证**：公式逐项恒等（max 为精确选择无舍入；vexpdif = exp(src0−src1) 同式融合，硬件舍入差 ≤ 减法 0.5 ulp 量级、经 |x−m| ≤ ~11.1（fp16 域）放大后相对误差 ≤ 6.6e-7 ≪ 1e-2 容差——§1.6.1 O3 论证）；累加分组（lane 内跨组累加 + 单次跨 lane 全归约）vs 源全行树形 → fp32 树深级差（verify_equiv.py V1 ≤1 ulp / V2 ≤2 ulp、违反率 0）；NaN 传播与 ±inf 角点行为逐元素等价（V4 位级一致）。
- **依据**：`examples/ascend/flash_attention/core.py` L94–137（SimdVF 行规约完整同构：vcmax/vcadd 全归约 + ONEPT_B32 存储 + vexpdif 融合 + BRC 广播，NPU 实测在用）；`example_per_token_cast_to_fp8.py` L132–153（SimdVF 内 vld/vabs/vmax/vcmax/vdupv/vdiv/vcvt/vsts 全链先例）；`tilelang/ascend/language/simd.py`（API 签名权威，§3.4 逐条行号）；`src/tl_templates/ascend/simd_inst.h` L688–710（vcmax→asc_reduce_max / vcadd→asc_reduce_sum 全归约语义）。
- **意图承接**：源 #5 树形归约的低延迟意图由跨 lane 全归约内在（单指令全 64-lane 收敛）承接；源 #4 寄存器复用意图由流式累加器（~10 向量寄存器常驻）承接。

### 0.7 标杆实现

- **源算子路径**：`/home/z00875089/TileOPs-fork/tileops/kernels/reduction/logsumexp.py`（`_logsumexp_kernel_single` L40–99——语义/算法/优化手段解读的唯一来源；`LogSumExpKernel` 类 L246–557——wrapper 契约与 tiling 启发式；`_primitives.py`——`DEFAULT_ALIGNMENT=256` L30 / `MAX_SINGLE_TILE_COLS=32512` L39 / `align_up` / `compute_tile_n` / `device_smem_budget`）。
- **源仓参考实现/测试基准**：`torch.logsumexp(x.float(), dim).to(x.dtype)`（源仓 `tests/ops/test_softmax.py` `ref_program` 直移植，fp32 计算域 golden）——§8.1 golden 直接采用（torch on NPU），保证验证独立性（不复刻 §0.6 的 NPU 算法）。
- **wrapper 对接面（SIMD 家族）**：manifest `source.kernel = tileops/kernels/reduction/logsumexp_simd/logsumexp_simd.py`（`LogSumExpSimdKernel`，Stage 5 集成对象）；契约口径与 SIMT 家族 wrapper（`examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp.py` dispatch）一致。
- **SIMT 兄弟变体（先例，非源）**：`examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp_kernel/_logsumexp_kernel_single.py` + `_logsumexp_kernel_single_DESIGN.md`——同族 SimtVF 实现（调用面 / 契约 assert / 边界测试组织 / Persistent+TIR if 嵌套可行性实证的参照）。

---

## 1. 概述

### 1.1 算子名称

`_logsumexp_kernel_single`（`LogSumExpFwdOpSimd` / `logsumexp_simd` 家族的**单 tile 变体**；迁移自 GPU TileOPs `tileops/kernels/reduction/logsumexp.py` 的 `_logsumexp_kernel_single`——与 SIMT 兄弟变体同源入口、独立实现）

### 1.2 功能描述

单 tile 行级 log-sum-exp：对 (M, N) 输入沿最后一维做数值稳定的 logsumexp 归约，输出 (M,)。契约 shape 空间 = N 一次驻留片上（§0.2：N_padded ≤ 32512 且 wrapper 预算内）；N 超出该空间的 online 多 tile 路径由 SIMD 家族的 tiled 变体（另任务）承接。支持 fp16/bf16/fp32 输入（fp32 中间计算）、任意 M（含尾批）、非 64 倍数 N（尾组 −inf 掩码）。**本变体的差异化定位：SimdVF 显式 2048-bit 向量内在实现**（用户显式指定编程模式，与 SimtVF 兄弟变体构成同族双模式对照）。

### 1.3 数学公式

$$
y_i \;=\; m_i + \log\Big(\sum_{j} \exp(x_{ij} - m_i)\Big), \qquad m_i = \max_j x_{ij}, \qquad i \in [0, M)
$$

（与源式完全一致——源已是 e-base 减 max 稳定化形态；等价性机器验证见 §1.6.1。）

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

**主路径（契约空间全域）——UB 驻留两遍式 + SimdVF 显式指令流**（§0.6 R1/R2/R3/R4/R5 落地）：

1. host 静态化（per-shape JIT 闭包）：`N_pad = align_up(N, 64)`；`G_full = N // 64`、`k_tail = N % 64`；`SUB = 满足 2·SUB·N_pad·elem ≤ 228,556B 的最大 2 幂 ∧ SUB ≤ block_m`；`full = M // SUB`、`rem = M % SUB`、`num_batches = full + (rem>0)`；`NUM_BLOCKS = 56`。
2. `T.copy` 将子批行块 `x[base : base+rows, 0:N]` GM→UB（x_ub (SUB, N_pad) 双版本流水，DMA 只搬 N 列；尾批路径静态切片 `[M−rem : M]`）。
3. SimdVF 块内**逐行两遍扫描**（行循环 `for r in range(rows)`，rows ∈ {SUB, rem} 静态）：
   - **max 遍**：`acc_max = vdup(−inf, f32)`；逐 64 元素组 `c = vcvt(vld(x_ub[r, g·64], "UNPK_B16"), "float32", part=0)`（fp32 输入直连 `vld`）→ `acc_max = vmax(acc_max, c)`；尾组先 `vsel(c, vdup(−inf), update_mask(k_tail, 32))`；遍末 `m_brc = vdupv(vcmax(acc_max))`（单次跨 lane 全归约 + 广播）。
   - **exp/sum 遍**：`acc_sum = vdup(0.0, f32)`；逐组重装载 → `e = vexpdif(c, m_brc)`（融合 exp(c−m)）→ `acc_sum = vadd(acc_sum, e)`；遍末 `s = vcadd(acc_sum)`（单次跨 lane 全归约）。
   - **行尾**：`out = vadd(m_brc, vln(s))` → `vcvt(out, dtype)`（fp32 输出免 cast）→ `vsts(lse_ub[r], out_h, pset(16,"PAT_VL1"), "ONEPT_B16")` 单点存储。
4. `T.copy(lse_ub[0:rows], y[base : base+rows])` UB→GM（静态切片）。

**与源算法的结构差异**（逐条注明来源）：① 驻留层 N_padded 256 对齐双路径 → UB align64 单路径 + fp32 域尾组掩码（§0.6 R2）；② x_f32 fragment 全行寄存器驻留 + exp 原位覆写 → **VRF 流式两遍重读**（§0.6 R2/#4 舍弃承接——零行数据寄存器驻留）；③ 线程级 `T.Parallel` + warp 树归约 → **显式 2048-bit SIMD 指令流 + vcmax/vcadd 跨 lane 全归约 + vexpdif 融合**（§0.6 R5）；④ 非持久 grid × threads=256 → persistent 56 核子批粒度调度（§0.6 R1，K3——无线程参数）；⑤ block_m 整批两层 4 分支 → 扁平化子批 2 分支（§0.6 R1/R4）；⑥ M 尾块隐式 OOB → 静态 full/rem + TIR if + 静态切片（§0.6 R4）。公式层与源恒等（e-base、减 max、log 每行 1 次——源已是调研确认的 A5 最优形态，§1.6.0 R1 候选 5 弃选证据）。

### 1.5 数据流图

```
GM[x (M,N) dtype]
  --T.copy (MTE2, 2D 子批连续区域 [base:+rows, 0:N], x_ub 双版本)--> UB[x_ub (SUB,N_pad) dtype]
  ==SimdVF 显式指令流（每行两遍）==
  pass1: UB --vld(UNPK_B16)--> VRF[x_h 128-lane] --vcvt(part=0)--> VRF[c 64-lane fp32]
         → VRF acc_max = vmax(acc_max, c) 逐组累加（尾组 vsel −inf 掩码）
         → VRF m_brc = vdupv(vcmax(acc_max))            [跨 lane 全归约 + 广播，寄存器驻留]
  pass2: UB --vld(UNPK_B16)--> VRF[x_h] --vcvt--> VRF[c]
         → VRF e = vexpdif(c, m_brc)（融合 exp(c−m)）
         → VRF acc_sum = vadd(acc_sum, e) 逐组累加
  epilogue: VRF s = vcadd(acc_sum) → out = vadd(m_brc, vln(s)) → vcvt(dtype)
  --vsts (ONEPT_B16, VALU)--> UB[lse_ub (SUB,) dtype]
  --T.copy (MTE3, 1D 静态切片 [base:+rows])--> GM[y (M,)]
```

纯 Vector 算子：无 L1/L0A/L0B/L0C 参与段（无 Cube/`T.gemm`），NT GEMM 布局不适用；无中间量落 GM；x 单遍 GM 读（两遍重读发生在 UB↔VRF 之间，不触 GM）、y 单遍 GM 写；m/s 统计量全程寄存器驻留（无 UB 往返、无 `mem_bar` 需求）。

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先调研（1.6.0，调研四问——输入公式/源算法只是候选之一，迁移任务在源算子解读后、耦合性判定前完成）；再做数学等价的公式优化（1.6.1）；再做循环/标量计算的向量化替代（1.6.2）；最后决策向量化轴与核内布局（1.6.3）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。1.6.1–1.6.3 的分析对象 = §1.6.0 调研选定、经 §0.6 重设计落地的 NPU 算法，不止步于源公式直译。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：同一数学语义（行 logsumexp）的算法族。调研深度：**完整调研**（规约/统计类算子）。执行位置：Phase M0 之后、Phase M1 之前（调研结论是 §0.5/§0.6 的判定输入）。**本变体的调研特有维度**：SIMD 承载模式下「行数据的驻留层级」（VRF 寄存器数组 vs UB 两遍重读）与「跨 lane 归约时机」（逐组 vs 延迟单次）是 SimtVF 调研中不存在的候选轴。
> 信息源：`.agents/skills/tilelang-op-design/references/algorithm-candidates.md`（**ALG-softmax** 命中行〔family 含 logsumexp〕+ **ALG-reduction** / **ALG-layernorm** 的 row-reduction 形态）、pattern-library（`P-LAYOUT-ROW-RESIDENCY` measured / `P-LAYOUT-TAIL-STATIC-EXTENT` measured / `TRAP-NAIVE-ONLINE-MERGE` measured / `CONST-valu-model` / `CONST-mte-bandwidth-model` / `CONST-capacity-a5` / `CONST-vrf-capacity-a5` / `CONST-arch-a5`）、`examples/ascend/` SimdVF 同类实现（example_per_token_cast_to_fp8 / flash_attention core.py / deepgemm transform_sf）、`testing/ascend/language/test_tilelang_ascend_simdvf_cast.py`（UNPK+vcvt 位级回归）、SIMT 兄弟变体 DESIGN（同族先例调研结论）、源算法（§0.3/§0.4——源仓 tiled 变体已用 online softmax 递推，是「存在在线算法」的直接证据）、知识预注入（CASE-logsumexp-harness / VP-2026-0014 vtanh 不存在·vexpdif fp32-only·exp2 无快档 / D-ELEM-ACT-FORMULA-PROBE）。互联网检索：未使用——本地源（候选表命中行 + 四个 SimdVF 先例 + 源仓双变体结构 + SIMT 兄弟变体完整调研 + 结构判据）已覆盖 logsumexp 族在 SIMD 承载下的全部已知变体（两遍驻留 / 三遍朴素 / online 流式 / Cube 归并 / 寄存器驻留 / UB 重读），无新增候选空间。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线 = 源算法（驻留三段式，N_padded 对齐） | `m=max(x); e=exp(x−m); y=m+log(Σe)`，SMEM 驻留 N_padded 列 | — | — | ✅（基线） |
| 1 | 减 max 稳定化 | 同上式（平移不变） | 数学恒等 | 消 fp16 溢出（必选） | 基线已含 |
| 2 | exp→exp2·log2e 换底 | `e^t = 2^(t·log2e)` | 数学恒等 | GPU SFU 快路径 | ❌ **A5 上弃选**：`CONST-valu-model` special_func 档位成员注记（measured 补注，2026-10-16 三 pilot 实证）：exp/exp2/log2 同入 8 ops/cycle 档（`src/ascend/codegen/intrin_rule_ascend.cc` L34–49 均注册）——换底无单位收益、反多 1 乘/elem；SIMD 路径同档证据（`tilelang/ascend/language/simd.py` L860–878 vexp/vln 注册，无 exp2 快档内在）；VP-2026-0014 交叉印证 |
| 3 | log 域拆分 / log2 换底输出 | `y = (m·log2e + log2(S))/log2e` 类 | 数学恒等 | 无（log 每行仅 1 次，非热点） | ❌ 每行 1 次 special（1/N 占比）；`CONST-valu-model` 同档证据同上 |
| 4 | UB 驻留两遍式 + fp32 域尾组掩码（布局级化简） | N_padded 256 列搬运+双路径 → N_pad=align64 紧凑 + 尾组 vsel(−inf) | 数学恒等（−inf 双幂元中性） | 非对齐 N UB 驻留 −37.5%；消除双路径；VRF 零驻留 | ✅（O2，A5 结构差异） |
| 5 | **vexpdif 融合**（SIMD 特有） | `e = vexpdif(c, m_brc)` 单指令 = exp(c−m)，替代 `vsub`+`vexp` 两指令 | 公式恒等（同式融合）；硬件舍入差 ≤ 减法 0.5 ulp 量级（§1.6.1 O3 论证） | −1 VALU op/64 元素（vsub @51 档） | ✅（O3，flash core.py L130–133 实测在用） |
| 6 | **延迟跨 lane 归约**（SIMD 特有） | 逐组 `vmax(acc, c)` 累加 + 遍末**单次** `vcmax`/`vcadd`，替代逐组 `vcmax`+`vmax` 合并（高层 `T.reduce` 在 SimdVF 内的自动 lower 形态，`src/ascend/transform/ascend_simdvf_lower_parallel.cc` L818–841） | max 精确（分组无关）；sum 为 fp32 分组重排（ulp 级） | −(G−1) 次 vcmax + −(G−1) 次 vmax/行（wl1 G=64：−126 指令/行 ≈ max 遍指令 −50%） | ✅（O4） |
| 7 | 寄存器驻留单装载（SIMT 形态直译到 SimdVF） | `T.simd.alloc_local(G, "float32")` 数组 + 两遍均读寄存器 | 数值同链 | −1 vld −1 vcvt / 64 元素（pass2 免重读重转）≈ −10% VALU | ✅ 纳入 R3/R4 对比后弃选（VRF 压力回归，见 R4 行） |
| 8 | 行内单遍 online（running max + 分块校正，SIMD 组粒度） | 逐组 `new_m = vmax(m, vcmax(c))`；`acc = vmul(acc, vexpdif(m, new_m)) + Σ vexpdif(c, new_m)` | 容差内等价（多次重舍入）——**但角点保真有坑**（见 R2） | −1 遍 vld/vcvt（UB 域），+1 vexpdif +1 vcmax +1 vmul / 组 | ❌ **弃选**：① 指令数净增（每 64 元素组 +3 指令 vs 省下的重读 2 指令）；② **角点保真分歧**——`m` 初始化为 −inf 时首组全 −inf 行使 `vexpdif(−inf,−inf)=exp(NaN)=NaN` 污染累加器（部分 −inf 行误判 NaN，违反 §0.1 ④）；初始化为 −MAX_FLOAT（`ascend_simdvf_lower_parallel.cc` L799–801 的做法）则全 −inf 行输出 −inf ≠ 源语义 NaN（违反 §0.1 ②）——两种初始化各破坏一处源边界语义，须额外钳位防护（`TRAP-NAIVE-ONLINE-MERGE` measured 同类角点发散：朴素合并 fp16 违反率 13.6%）；③ 驻留场景数据已在 UB，online 的 GM 流量收益为零 |
| 9 | fp16 域 max 遍（max 直接在 16-bit 向量上做，省 max 遍的 vcvt） | pass1 `vmax(acc_h, x_h)`（128-lane fp16）→ 遍末 `vcmax(acc_h)` → 单值转 fp32 | max 为精确选择，fp16 域 max 值 == fp32 域 max 值（选择无舍入）→ 数学恒等 | max 遍 −2 vcvt + −1 vmax / 128 元素 ≈ −4% 端到端 | ❌ **弃选**：① bf16 的 `vcmax` dtype 支持未文档化（`simd_inst.h` L706 模板无 dtype 限制声明，但 CCE `asc_reduce_max` 对 bf16 的硬件支持无文档佐证——「未文档化假设」）；② 收益 ~4% 不抵双域 max 链的复杂度与语义保真偏好（fp32 域 max 与源链位级一致，V1 ulp=0 的组成部分）；③ NaN 行为依赖硬件 reduce 的 NaN 语义（未文档化）——fp32 域链不依赖该语义（NaN 经 exp 路径传播，§9.3）。Stage 4 可作探索项重估 |
| 10 | Cube GEMV 归并（sum = e·1 向量内积） | exp 后 `T.gemm(e_l1, ones_l1, c_l0c)` | 数学等价 | Cube MAC 替代 Vector 归约 | ❌ **弃选**：exp 主导段仍在 Vector（逐元素 special），Cube 只接管 sum——多出 UB→L1 ND2NZ 转换 + L0C→UB dual_copy 往返（`CONST-mte-bandwidth-model` ND2NZ 62 cycles base 非免费）+ mixed 核复杂度；而 `vcadd` 已是单指令跨 lane 全归约（flash core.py L134 实测在用），归约非瓶颈（R3 表：计算项 special 主导）。**违反流量最小化** |
| 11 | 两 kernel 分离（先全行 max 落 GM，再 exp+sum） | kernel1 max → GM；kernel2 exp+sum | 数学等价 | 无（每阶段行内依赖不变） | ❌ **弃选**：X 读 2 遍 GM（流量 +100%）+ 跨 kernel 同步；驻留可行时纯负优化 |

**R2 在线算法**：**有**，且证据三层齐全。① 源仓自身：`logsumexp.py` 的 `_logsumexp_kernel_tiled`（L105–194）即 online 变体——分块 running (max, sum) + flash 形式重缩放合并（running 结构是「存在在线算法」的直接证据）；② 参考表：ALG-softmax R2「online softmax（分块 running max/sum + 重缩放）」命中行；③ 结构判据：logsumexp 可分解为 running 统计量（max 与 sum 可增量合并）→ 原则上存在在线变体。**收益口径与结论**：online 的价值 = 中间缓冲 O(N)→O(tile) + GM 扫描遍数（大 N 无法驻留时）。**本 single 变体的契约空间（§0.2）已由 wrapper 保证 N_padded ≤ 32512 且驻留预算内**——行块可整体驻留 UB（wl1：x_ub 双版本 131KB ≤ 223.2KB，§4.5），**online 结构在契约空间内非必需**；N 超驻留上限的 shape 由 tiled 变体（online 路径，另任务）承接——**算法族分工即源仓双变体结构，本设计不吞并**。行内组粒度 online 微变体（R1 候选 8）在驻留域内**角点保真分歧 + 指令净增**，弃选（见上——这是本调研相对 SIMT 兄弟变体的新增否定性证据：SIMD 组粒度 online 的两种 m 初始化方式分别破坏 §0.1 ②/④ 语义）。

**R3 算法复杂度对比**（逐候选四口径；口径说明：FLOPs/elem 中 vcvt/vmax/vadd 各计 1 op、vexpdif/vln 计 1 special（折算系数按 `CONST-valu-model` special = 8 ops/cycle/AIV，assumption）、vln 每行 1 次忽略；GM Bytes/elem = 输入读 + 输出写（输出 M 个 << M·N，忽略）+ 中间 GM 往返；**UB 重读量**（本变体特有口径补充，计入访存权衡不计 GM）：pass2 重读 2B/elem（16-bit）/4B/elem（fp32）；扫描遍数 = 同一 GM 数据读写次数；中间缓冲 = 行级驻存需求峰值（UB + VRF）；代表 workload = wl1 (M=1024, N=4096, fp16, block_m=16)）：

| 算法候选 | FLOPs/elem | GM Bytes/elem | 扫描遍数 | 中间缓冲/行 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线直译（N_padded 驻留 + 线程级直译形态：单版本 x_ub、block_m 整批、逐元素 if_then_else 掩码） | ~5 op + 1 special | 2（读 2B；对齐 N）至 3.4（非对齐 N=300：N_padded=512 列） | 2 | x_ub block_m·N_padded·elem（wl1 = 128KB 单版本无流水）+ 掩码载入寄存器压力（无线程域承载参考） | M 行独立；无跨核归约 |
| **选定：UB 驻留两遍式 + 延迟跨 lane 归约 + vexpdif 融合** | ~4 op + 1 special（vcvt×2 + vmax×1 + vadd×1 + special×1；vsub 被融合消除） | 2（任意 N 紧凑；非对齐 N 较基线 UB 驻留 −37.5%）+ UB 重读 2B（16-bit）/4B（fp32）——VRF↔UB 256B/cycle 非瓶颈（D-2） | 2 | x_ub 2·SUB·N_pad·elem（双版本 ≤223.2KB）+ VRF ~10 向量寄存器 ≈ 2.5KB | 同上；子批串行在核内（流水重叠） |
| 寄存器驻留单装载（R1 候选 7，SIMT 形态直译） | ~3 op + 1 special（省 pass2 的 vld+vcvt） | 2（UB 重读 0） | 2 | x_ub 双版本 + **VRF 4·N·SUB B（wl1 = 128KB——回到 SIMT x_frag 贴顶水位，`CONST-vrf-capacity-a5`）** + alloc_local 数组索引开销 | 同上 |
| 行内组粒度 online（R1 候选 8） | ~4 op + 1 special + 3/64 组 special+乘 | 2（UB 重读 0） | 2 | 同选定 | 行内组序依赖（削弱流水重叠）+ 角点保真分歧（R2） |
| Cube GEMV 归并（R1 候选 10） | ~4 op + 1 special + Cube MAC | 2 + ND2NZ/dual_copy 往返 | ≥2 | 追加 L1/L0C staging | mixed 核协调开销 |
| 两 kernel 分离（R1 候选 11） | ~4 op + 1 special（+max 落 GM 读回） | 6（读 2×2B + max 写读 4B + 写 ~2B） | 4 | O(1) 行统计量 | 跨 kernel 全局同步 |

**R4 硬件亲和性评估**（逐候选对照检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水融合 / 跨核结构；负向淘汰依据已附）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线直译 | 纯 Vector ✓；special 1/elem 撞最窄吞吐档（`CONST-valu-model` special=8） | x_ub 单版本 128KB 无流水重叠；N_padded 256 列驻留浪费（非对齐 N +71% @N=300） | N_padded 256 列对齐（GPU 约束直译，A5 无此需求）；逐元素掩码无线程域承载 | ✓（JIT 常量） | 无双缓冲 → MTE2/VALU 串行 | ⚠️ 可行但无流水 + 驻留浪费——**降级为结构参照** |
| **UB 驻留两遍式（选定）** | Vector ✓；special 1/elem（公式层已最小）；vcmax/vcadd 单指令全归约（`simd_inst.h` L688–710）；vexpdif 融合减 1 op | x_ub 双版本 ≤223.2KB（SimdVF UB 248KB × 90%，`CONST-capacity-a5`）✓；**VRF ~2.5KB 零压力**（vs SIMT 128KB 封顶）✓ | N_pad=align64：组起始 128B/256B 对齐 ✓；vld UNPK_B16 组宽 64 元素 ✓（cast 回归位级验证）；尾组 update_mask(k,32) 任意 k ✓ | G_full/k_tail/full/rem/SUB 全 JIT 常量 ✓；尾组 Python 静态分支（源 `_needs_pad` L68 同法先例）✓ | 子批双缓冲（num_stages=2）MTE2/VALU 重叠 ✓；persistent 多波 ✓ | ✅ **选定（主选）** |
| 寄存器驻留单装载 | 同上 | **VRF 4·N·SUB = 128KB @wl 工作点——回到 SIMT x_frag 贴顶水位**（`CONST-vrf-capacity-a5` assumption 256KB 的 50%，实测样本 76 reg/thread 无 spill 恰在同水位——无余量）；另需 alloc_local 寄存器数组 + 展开循环 | ✓ | ✓（G 静态） | ✓ | ❌ 弃选（R3/R4：省 10% VALU 换 VRF 压力回归 + 结构复杂度；UB 重读本就非瓶颈 13× 余量） |
| 行内组粒度 online | 同选定 + 每组 +3 指令 | 同选定 | ✓ | ✓ | 行内组序依赖削弱流水 | ❌ 弃选（R1 #8/R2：角点保真分歧 + 指令净增 + `TRAP-NAIVE-ONLINE-MERGE` 角点风险） |
| Cube GEMV | exp 段 Vector + sum 段 Cube——单元混置但 exp（主项）仍 Vector | 追加 L1 staging + L0C 累加器 | 须凑 C0 分形（M/N 16 倍数——M 任意契约下须 padding，`CONST-gemm-dim-constraints`） | ✓ | mixed 核协调 | ❌ 弃选（R1 #10：转换开销 + 复杂度，归约非瓶颈） |
| 两 kernel 分离 | ✓ | O(1) | ✓ | ✓ | 跨 kernel 间隙 | ❌ 弃选（R1 #11：GM 流量 +200%） |

**调研结论**：选定 **「UB 驻留两遍式 logsumexp（align64 单路径 + fp32 域尾组 −inf 掩码 + 延迟跨 lane 归约 + vexpdif 融合 + VRF 流式零驻留 + 子批双缓冲流水）」**。关键依据：① R3——公式层源已是该族最优形态（e-base 减 max、log 每行 1 次；R1 候选 2/3 的换底在 A5 同档无益），结构层 UB 驻留两遍式对非对齐 N 驻留 −37.5%、VRF 从 SIMT 的 128KB 封顶降到 2.5KB 零压力（SUB 封顶机制仅剩 UB 单约束）；② R4——纯 AIV 负载、容量全档 ✓（223.2KB/2.5KB 预算）、组对齐整除、persistent 流水可融合、无跨核归约；③ R2——online 变体存在（源仓 tiled 即证据）但契约空间内驻留可行，非必需；组粒度 online 在 SIMD 承载下有**角点保真分歧**（−inf 初始化污染部分 −inf 行 / 有限初始化改变全 −inf 行语义），弃选证据为本调研新增（SIMT 兄弟变体调研中该候选仅因指令数与流水净亏弃选）。与基线的结构差异一句话：**N_padded 对齐驻留 + 线程级并行 + warp 树归约（源）→ align64 UB 驻留 + 显式 SIMD 指令流 + 单指令跨 lane 全归约 + VRF 流式（选定）**，公式恒等。源优化手段意图承接：#1 驻留（R2 承接）/#2 双路径（R2 统一为更优单路径）/#3 稳定化（保留）/#4 原位覆写（舍弃——SimdVF 流式形态下意图被「零寄存器驻留」超量承接，§0.5）/#5 树形归约（vcmax/vcadd 等价替换）/#6 行批（persistent+子批承接）/#7 grid（persistent 承接）——逐项见 §0.5，无静默丢弃。调研范围：ALG-softmax/ALG-reduction/ALG-layernorm 命中行、examples/ascend 四个 SimdVF 同类实现（per_token/flash/deepgemm/cast 回归）+ SIMT 兄弟变体完整调研、pattern-library 八条目（四 measured：P-LAYOUT-ROW-RESIDENCY / P-LAYOUT-TAIL-STATIC-EXTENT / TRAP-NAIVE-ONLINE-MERGE / CONST-arch-a5；四 assumption：CONST-valu-model / CONST-mte-bandwidth-model / CONST-capacity-a5 / CONST-vrf-capacity-a5）、源仓双变体结构分析、知识预注入三条目（CASE-logsumexp-harness / VP-2026-0014 / D-ELEM-ACT-FORMULA-PROBE）。

**设计期估算（D-2）**：估算下界(相对)：**计算项（VALU special_func）主导，约为流量项的 ~10 倍**（代表 workload wl1 (1024, 4096) fp16：流量 8.39MB → GM 读 8.39MB/(56×100 B/cycle)≈1.5K cycle〔`CONST-mte-bandwidth-model` assumption，AIV MTE2 档〕；计算 4.19M×(4/51 + 1/8)/56 AIV ≈ 15.2K cycle〔`CONST-valu-model` assumption，special=8 档——special 0.125 cycle/elem 占计算项 61%〕；**UB 重读流量**：pass2 重读 2B/elem ÷ VRF↔UB 256B/cycle = 0.0078 cycle/elem ≈ 0.6K cycle——远低于计算项（13×），不构成瓶颈；容量项可忽略——驻留单遍 GM 读、无 L0C 往返、无分块重扫、VRF 2.5KB）。主选 vs 基线直译相对比：VALU 计算项 0.93×（vexpdif 融合 −1 op + 延迟归约 −(G−1)·2 指令/行摊薄）、GM 流量非对齐 N 0.625× UB 驻留（N=300：512→320 列）、对齐 N 1.0×；端到端 ≈ max(15.2K, 1.5K) vs max(16.4K, 1.5K)+无流水损失。**口径不确定披露**：special_func「8 ops/cycle」的 op 粒度（element vs 向量指令级）在 `src/ascend/transform/estimate_latency.h` L58–70 未区分——若为向量指令粒度则瓶颈反转 MTE 主导（与 SIMT 兄弟变体风险 5 同源）；vcvt 转换吞吐按 VALU 51 档折算为估算（档位未单列）。两种解释下主选结构（special 已最小化 + 流量已最小化 + UB 重读余量 13×）均为最优；该常数列 §9 风险点，Stage 4 npusim/msprof 实测归因标定。A5 冷启动期 assumption 基估算，常数待标定，容量按 90% 折扣。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定、经 §0.6 R2/R5 落地的 UB 驻留两遍式算法。源公式层已是 A5 最优形态（e-base、减 max、log 1/行——R1 候选 2/3 弃选），本节优化集中在**结构 / 指令 / 访存层**（布局级化简 + 指令融合 + 归约时机），每项四要素齐备。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| O1 | 减 max 稳定化（必选确认） | `y = log(Σ e^{x})` | `y = m + log(Σ e^{x−m})`，m = max(x) | 指数平移恒等；消 fp16/bf16 数据域溢出（e^11 > 65504） | 计算量不变，数值稳定（源已含，语义层保留项） |
| O2 | UB 驻留两遍式 + fp32 域尾组 −inf 掩码（布局级） | 源：`x_f32 (block_m, N_padded)`，N_padded = align_up(N, 256)，padding 列以 −inf 显式载入（`_needs_pad` 路径）或隐式搬运（对齐路径） | `x_ub (SUB, N_pad)` 紧凑 DMA（N_pad = align64，只搬 N 列）+ 逐组流式装载，尾组 `c = vsel(c, vdup(−inf,f32), update_mask(k,32))` | 尾组掩码的 −inf 同为 max 幂元（max(−inf,·)=·）∧ exp 后 sum 幂元（exp(−inf−m)=0，含 m=−inf 时 exp(NaN) 的 NaN 传播路径两链一致）→ 数学恒等；**机器验证 V1**：全 shape（33×300/32×256/64×4096/128×16384/3×7/1×1/2×32512/57×64）× fp16/bf16 域 fp32 链 **≤1 ulp、fp64 参照一致、违反率 0** | 非对齐 N（300）：UB 驻留 −37.5%（512→320 列）且 GM 读不变；对齐 N：消除 `_needs_pad` 双路径；**VRF 零行数据驻留**（vs SIMT 128KB x_frag） |
| O3 | vexpdif 融合（指令级，SIMD 特有） | `d = vsub(c, m_brc); e = vexp(d)`（两指令） | `e = vexpdif(c, m_brc)`（单指令，dst = exp(src0−src1)） | 公式恒等（同式融合——CPU 模拟两式位级同链，V1 已含）；硬件差异仅为融合内部减法的舍入时机：≤0.5 ulp（fp32 减法）→ 经 exp 放大 |x−m| ≤ 11.1（fp16 域 max-gap）/ ln 下溢侧无影响 → 相对误差 ≤ 11.1·2⁻²⁴ ≈ 6.6e-7 ≪ fp16/bf16 容差 1e-2（4 个数量级余量）；fp32 输入容差 1e-3 同样 3 个数量级余量 | −1 VALU op/64 元素（vsub @51 档）：wl1 −4096 op ≈ VALU 计算项 −4.8%；指令数 −1/组 |
| O4 | 延迟跨 lane 归约（归约时机，SIMD 特有） | 逐组跨 lane 归约再合并（高层 `T.reduce` 的 SimdVF 自动 lower 形态：每组 `vcmax(c)` + `vmax(acc, reduced)`，`ascend_simdvf_lower_parallel.cc` L818–841） | 逐组元素级累加 `acc = vmax(acc, c)` + 遍末**单次** `vcmax(acc)` / `vcadd(acc_sum)` | max：结合/交换律精确成立（任何分组同值，无舍入）；sum：fp32 累加分组重排（lane 内跨组序贯 + 单次 64-lane 全归约 vs 逐组全归约+标量合并）→ 树深级 ulp 差；**机器验证 V2**：全 shape ≤2 ulp（128×16384）、违反率 0 @1e-3 | wl1（G=64）每行 −126 指令（63 vcmax + 63 vmax）≈ max 遍指令 −50%；比自动 lower 形态更优——显式内在的价值点 |
| O5 | cast 链最小化（确认项） | — | 输入提升 1 vcvt/64 元素（UNPK_B16 装载 + vcvt(part=0) 单次到位，两遍共 2 次）；输出 cast 1/行（vcvt + ONEPT 存储） | fp32 中间是源语义（§0.1 dtype 语义），不可省 | 已最小（无冗余 cast：装载提升与尾组掩码解耦于 vsel，输出每行仅 1 次）；`test_tilelang_ascend_simdvf_cast.py` 位级回归锚点 |
| O6 | 输出饱和角点分析（确认项，非优化） | — | `vcvt(out, fp16/bf16)` 默认 sat=True | **契约空间内数学封闭**：out = m + ln(Σ) ≤ m + ln(N) ≤ 65504 + ln(32512) ≈ 65514.4 **< 65520**（fp16 IEEE 舍入到 inf 的阈值）且 Σ ≥ 1（max 元素贡献 exp(0)=1）→ sat 钳位与 IEEE 溢出在契约空间内不可能分歧；**机器验证 V5**：最恶劣契约行（全 65504 × N=32512）sat vs IEEE 0 分歧 lane | 无数值风险；免额外钳位指令（对照：若上界可达则需 sat=False + 显式 inf 处理） |

**优化结论**：**采纳 3 项（O2 布局级 / O3 指令融合 / O4 归约时机）+ 3 项必选/确认（O1 稳定化、O5 cast 链、O6 饱和角点封闭）**。优化后主路径公式（§3.1 拆解的唯一输入）：

```
m_i   = max_j x[i,j]                        # fp32（尾组 vsel pad=-inf，精确；lane 累加 + 单次 vcmax）
e_ij  = exp(x[i,j] − m_i)                   # fp32，vexpdif 融合单指令，UB 流式重装载
S_i   = Σ_j e_ij                            # fp32（lane 累加 + 单次 vcadd 全归约）
y_i   = cast(m_i + log(S_i), dtype)         # 每行 1 次 special + 1 次 cast + ONEPT 单点存储
```

**等价性机器验证（D-1）**：已生成并执行 `examples/logsumexp_simd/_logsumexp_kernel_single/verify_equiv.py`（fp64 参照 + 角点值：全零行/全等行/单主导/±65504 极值/subnormal ±6e-8/深度下溢 −745/部分 −inf/全 −inf/含 +inf/含 NaN/N=1/非 64 倍数 N（300）/M 尾（33/57）；torch CPU；候选式与基线式均按目标精度链计算对照——SIMD 链以「64 元素组分桶 + lane 内跨组累加 + 单次跨 lane 归约 + 尾组 −inf」逐要素模拟）。执行结果（2026-10-17，tilelang-fork 工作区 + torch CPU）：

| 验证项 | 对照（候选 vs 基线/精确） | 最大 ulp 差（fp32 位级） | 最大相对差 | 违反率（@容差） | 结论 |
|--------|--------------------------|--------------------------|------------|---------------------|------|
| V1（O2+O3，8 shape × 2 dtype × 2 参照 = 32 行） | SIMD 两遍式链（含 vexpdif 同式融合建模）vs 源 N_padded 链（fp32 链对照 + fp64 参照） | **≤1**（64×4096 / 128×16384 等；33×300、3×7、1×1、2×32512、57×64 为 0） | ≤1.086e-07 | 0.000000（@1e-2） | **EQUIV_PASS** |
| V2（O4，8 shape） | 延迟跨 lane 归约 vs 逐组跨 lane 归约（两种 fp32 分组） | **≤2**（128×16384） | ≤1.869e-07 | 0.000000（@1e-3） | **EQUIV_PASS** |
| V3（A5 链 vs golden，8 shape） | SIMD 链 vs `torch.logsumexp`（fp32） | ≤1 | ≤1.086e-07 | 0.000000（@1e-3） | **EQUIV_PASS** |
| V4（源链 vs SIMD 链，fp16/bf16 输出 × 2 N 值 = 4 行） | 含特殊行（全 −inf/含 +inf/含 NaN/部分 −inf/全等/极值/subnormal/深下溢）的 fp64→16-bit 输出对照 + NaN/±inf 模式逐位比对 | **0**（位级一致）；NaN/±inf 模式全部一致（无 NANPATTERN 失败行） | 0 | 0.000000（@1e-2） | **EQUIV_PASS** |
| V5（O6，信息项） | sat=True 钳位 vs IEEE 溢出（最恶劣契约行：全 65504 × N=32512） | — | — | — | **0 分歧 lane**（上界 65514.4 < 65520，契约空间内数学封闭） |

> **全部采纳项 EQUIV_PASS**（V1 ≤1 ulp、V2 ≤2 ulp、V4 位级一致 + NaN 模式一致；52 门控行 0 失败）。V4 证实 ±inf 角点行为为源算法固有（链与源位级一致即迁移保真达成），与 torch 的分歧不构成采纳项失败（§9.3 文档化）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 本变体的模式本身即「显式向量化」——逐元素计算 100% 以 `T.simd.*` 2048-bit 向量内在表达（每指令 64 fp32 / 128 fp16 元素）。本节盘点全部循环与标量计算点，确认无逐元素标量残留、逐项说明保留的非向量化点。

| # | 计算点 | 实现形态 | 向量化状态 | 不可替代理由（非向量化点必填） |
|---|--------|-----------|-----------|------------------------------|
| 1 | 输入装载 + fp32 提升（逐元素） | `vld(UNPK_B16)` + `vcvt(part=0)` 每 64 元素组 1+1 指令 | ✅ 2048-bit 显式 | —（`test_tilelang_ascend_simdvf_cast.py` L64–80 位级回归；fp32 输入 `vld` NORM 直连） |
| 2 | 尾组边界掩码（逐元素条件） | `vsel(c, vdup(−inf), update_mask(k,32))`——向量谓词选择，仅尾组 1 次 | ✅ 谓词化 | —（`deepgemm/transform_sf.py` L205/L215/L249 vsel 掩码先例；`testing/ascend/language/test_tilelang_ascend_simdvf_integer_predicate.py` L261 update_mask 先例；比源逐元素 `T.if_then_else` 载入更优——掩码只付 1 指令/行而非 1 select/元素） |
| 3 | 行内 max（逐元素比较） | `vmax(acc, c)` 逐组 + `vcmax` 单次全归约 | ✅ | —（flash core.py L115–116 同构） |
| 4 | 稳定化指数（逐元素 exp） | `vexpdif(c, m_brc)` 融合单指令/组 | ✅ | —（flash core.py L130–133 同构） |
| 5 | 行内 sum（逐元素累加） | `vadd(acc, e)` 逐组 + `vcadd` 单次全归约 | ✅ | —（flash core.py L134–135 同构） |
| 6 | 每行 lse = m + log(S) + cast + 写出 | `vadd(m_brc, vln(s))` → `vcvt` → `vsts(ONEPT)` —— 4 向量指令/行，值在 lane 0 有效 | ✅（行粒度向量指令） | 依赖归约终值（S 定稿后才能求 log）——tile 级标量（4 指令/行 vs G×7 指令/行元素段，占比 wl1 = 4/(64×7) ≈ 0.9%），无法并入元素循环；非逐元素热点（SIMT 兄弟变体 #5 同性质） |
| 7 | 行循环（子批内 rows 行串行） | `for r in range(rows)` TIR 串行 | ❌（保留串行） | 行间经 `m_brc`/`acc` 寄存器状态无依赖（每行独立重初始化），但 UB 装载与行计算共享同一指令流——行间交错（r+1 的 pass1 与 r 的 pass2 重叠）属指令调度层（编译器/Stage 4 观察项），语言层无行级并行原语可表达（SimdVF 无线程域）；rows ≤ SUB ≤ 16 非热点 |
| 8 | 组循环（行内 G 组串行 × 2 遍） | `for g in range(G_full)` + 尾组 Python 静态分支 | ❌（保留串行） | 两遍扫描的数据流本质（pass2 依赖 pass1 的 m 终值——行内顺序依赖）；组间流式依赖 UB 装载顺序；串行边界全静态（gate 要求 ✓）；`P-LAYOUT-TAIL-STATIC-EXTENT` measured：「行子批串行循环用 T.serial/range（保持 x_ub 双版本流水顺序依赖），不要展开」 |
| 9 | 尾批 TIR if 分支（`if t < full`） | 运行时控制流 | ❌（保留分支） | 依赖循环变量 t 的 M 尾批边界语义；分支体两侧均为静态切片（`example_gemm.py` L47 先例 + SIMT 兄弟变体实证）；每批 1 次分支判断，非热点 |
| 10 | host 侧 N_pad/SUB/G_full/k_tail/full/rem 计算 | Python 算术 | n/a | host 元数据计算，不在 kernel 内（per-shape JIT 静态化） |

**向量化结论**：逐元素计算 100% 显式向量化（#1–#5 全部 2048-bit 内在，API 均有 `examples/ascend` / `testing/ascend` 位级或实测佐证）；保留 3 处非向量化点——#6 行级 epilogue（4 向量指令/行，依赖归约终值，占比 <1%）、#7/#8 串行循环（行内两遍顺序依赖 + 流水顺序依赖，静态边界）、#9 尾批分支（运行时边界语义，先例佐证）——理由类别 = tile 级标量 + 顺序依赖 + 边界处理，均非「实现简单」类借口；与 §6 循环结构一致（§6 无逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（[M, N] 行主进 / (M,) 行主出），核内布局与 lane 映射是设计变量。规约类算子必选候选：水平归约（lane 映射到归约轴 N）vs 垂直扫描（lane 映射到独立行 M、串行步进归约维）——正交决策分别枚举；另评估 SIMD 特有轴决策（跨 lane 归约时机 / 行数据驻留层级）。

**轴质量评分**（对选定轴 N）：

| 评分项 | 判定 |
|--------|------|
| 整除性 | 主 workload N=4096/16384 % 64（组宽）= 0 ✓（零尾组）；非 64 倍数 N（300→N_pad 320）尾组浪费 (64−44)/64 ≈ 31%（单组掩码吸收，仅 UB 域——DMA 侧紧凑无浪费） |
| 尾 lane 浪费率 | 主 workload 0%；N=300 → 尾组 31%（该组 vmax/vexpdif 仍全量有效计算——pad lane 的 exp(−inf−m)=0 不污染 sum） |
| 累加链形态 | 归约维 N = 组序贯 stride-1 连续向量维（组内 64 lane 连续装载）；无跨步系数 |
| repack 代价 | 无（I/O 原生行主连续，核内布局与 I/O 同构；UNPK_B16 装载即提升，无交错重排需求——规约序不敏感） |
| UB/VRF 容量影响 | 行子批驻留：x_ub 双版本 ≤223.2KB + VRF 流式 ~2.5KB（§4.5 预算表，SUB 封顶） |

**候选矩阵**（规约类候选全枚举 + SIMD 特有轴，弃选给量化理由）：

| # | 布局方案 | 向量化轴 / 驻留层级 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | 行主 [M,N] 原生 + **水平归约**（lane→N 连续轴，64 元素组，UB 驻留 + VRF 流式，persistent 子批） | N / UB 两遍 | 无 | 连续 GM 行（wl1 行 8KB，32B 对齐 ✓）；归约维 = stride-1 组内连续；per_token + flash + cast 回归三先例；vcmax/vcadd 单指令全归约 | ✅ **主选** |
| 2 | 垂直扫描（lane→M 行，N 向串行步进） | M | 无 | N=4096 串行步进/行——向量宽度利用率 rows/64 ≤ 25%（rows≤16），行内并行度归零；吞吐劣化 >4×（按 64 lane 满载比估算）；repo 无先例 | ❌ 串行化 N 维，量化劣化不可接受 |
| 3 | 2D 分块 + 跨核行归约（M/SUB × N/BN grid，partial 落 GM + 跨核 flag 合并） | N（块内） | partial 统计量 GM 往返 | 行块已可驻留（wl1 双版本 131KB ≤ 223.2KB），跨核归约新增 2 遍 partial GM 读写 + 跨核同步（流量 +50% 起步）；UB 容量足够无需分块；且跨核 (max,sum) 合并语义超出 additive atomic 表达力（SIMT 兄弟变体 Stage 4 `[DESIGN_LIMIT]` 实测结论：M≪56 域并行度天花板 = M 核，跨核 N-split 需 (max,sum) 合并语义、additive atomic 不可表达） | ❌ 违反流量最小化（R3）；容量不构成约束；合并语义无 API 支撑 |
| 4 | 转置布局（列主驻留 + M 向归约） | M（转置后） | 核内转置链（`T.copy` transpose 仅特定路径——非通用，`tilelang/ascend/language/copy_op.py`；通用转置 API 存在性未命中→「未文档化假设」） | 转置 repack 开销 + UB 翻倍（双布局驻留）；收益仅在于把归约轴换到 M——M 维短（wl1 1024 vs N 4096）无收益 | ❌ repack 纯开销、无轴收益（N 已是最长连续轴） |
| 5 | SIMD 特有：逐组跨 lane 归约（自动 lower 形态直译） | N / 归约时机=逐组 | 无 | 每组 +1 vcmax +1 vmax（R1 候选 6 弃选形态） | ❌ 被 O4 延迟归约取代（−50% max 遍指令） |
| 6 | SIMD 特有：寄存器驻留单装载（SIMT 形态直译） | N / VRF 驻留 | 无 | −10% VALU（免 pass2 重读重转）vs **VRF 4·N·SUB = 128KB 贴顶回归** + alloc_local 数组复杂度（R3/R4 表） | ❌ 弃选（D-2：UB 重读 13× 余量非瓶颈；VRF 压力回归得不偿失） |

**源码轴选择对照**（迁移任务必做）：源 GPU 也是 N 轴并行（`T.Parallel(block_m, N_padded)` 行内元素线程化 + warp 归约 dim=1）+ 行级 grid——**轴结论一致**（N 是唯一高带宽连续轴，两代硬件共识），但承载结构完全重设计：GPU block×threads=256 → SimdVF 2048-bit 指令流（64 fp32/组）；GPU warp shuffle 树 → vcmax/vcadd 单指令全归约；GPU 非持久 grid → persistent 56 核子批（§0.6 R1/R5）。GPU 轴选择是输入而非结论，已独立复核。

**布局决策结论**：选定方案 #1：核内布局 = `x_ub (SUB, N_pad=align64)` dtype 紧凑（双版本）+ **VRF 流式零驻留**；向量化轴 = **N（行内连续轴，64 元素组粒度）**；跨 lane 归约时机 = **延迟单次**（每行 1 vcmax + 1 vcadd）；行数据驻留层级 = **UB 两遍重读**（VRF 仅 ~10 累加器寄存器）。该结论同步落入 §3.3 伪代码与 §6 循环结构（组循环的内层向量维 = 64 元素组，buffer 形状 = §4.3）；弃选方案量化理由见候选矩阵（#2 串行化 >4×、#3 流量 +50% 且合并语义无支撑、#4 repack 无收益、#5 被 O4 取代、#6 VRF 压力回归）。**无实验裁决三件套**：主选与全部弃选候选的判定裕度均不依赖未实证常数（#6 的 10% VALU 差是结构性指令计数差而非未知常数；#2/#3/#4 为量级差弃选），不触发 skill Phase 2 第 3 项的强制裁决条件。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: **simd（SimdVF，显式 `T.simd.*` 2048-bit 向量内在）**（auto_schedule: yes）

### 2.2 选型理由

1. **用户显式指定（契约约束）**：调度 prompt 明确 `programming_mode: simd`——本变体的任务定位即 LogSumExpFwdOp 的 SIMD 模式独立变体（`LogSumExpFwdOpSimd`），与 SimtVF 兄弟变体构成同族双模式对照。决策树「从 simt 起步原则」被用户显式指令覆盖（该原则适用于无指定时的默认推导，不适用于变体任务契约）。
2. **算子特征与 SimdVF 适配度**：纯 Vector 算子（逐元素 cast/exp + 行规约 max/sum），无 matmul/Cube 段——无 mixed 需求；行规约在 SimdVF 下有完整的内在支撑（`vcmax`/`vcadd` 单指令跨 lane 全归约 + `vdupv` 广播 + `vexpdif` 融合——flash attention 的 softmax 段即此形态，`examples/ascend/flash_attention/core.py` L94–137）；显式内在还可实现自动 lower 形态拿不到的优化（O4 延迟归约：−50% max 遍指令，§1.6.1）。
3. **repo 先例**：`examples/ascend/example_per_token_cast_to_fp8.py`（SimdVF 逐行规约 + persistent + ONEPT 存储 + 输出 cast 全链）、`examples/ascend/flash_attention/core.py`（SimdVF 行 max/sum 归约 + vexpdif + BRC 广播）、`examples/ascend/deepgemm/kernels/transform_sf.py`（SimdVF vsel 掩码装载）、`testing/ascend/language/test_tilelang_ascend_simdvf_cast.py`（UNPK+vcvt 位级回归）——API 佐证链完整。
4. **对比 SimtVF 兄弟变体的结构性得失**（变体差异化定位，非优劣裁决）：得——VRF 零驻留（x_frag ≤128KB → ~2.5KB，SUB 封顶仅剩 UB 单约束）、UB 预算更大（SimdVF 248KB vs SimtVF 216KB，`CONST-capacity-a5`）、延迟归约优化空间（O4）；失——两遍重读 UB（+0.016 cycle/elem，非瓶颈）与编程复杂度（显式指令流 vs `T.Parallel`/`T.reduce_*` 自动形态）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| VF 入口 | `with T.SimdVF():`（无线程参数——SimdVF 无线程概念，tilelang-ascend skill §2.3）；latency 标注设计期不设（编译器估算；Stage 4 可按 npusim 实测补 `latency=`，skill §8.3） |
| 内存分配 | `T.alloc_shared`（UB：x_ub/lse_ub）+ VRF 向量寄存器（SSA 值绑定，无 `alloc_fragment`/`alloc_local` 需求——零行数据驻留） |
| 同步 | auto_schedule: yes——AutoSchedule 管理跨引擎 flag 与 multibuffering（§7） |
| 模块声明 | kernel 模块 docstring 须声明 `programming_mode: simd`（迁移 prompt 契约，§10.3） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 优化后公式；每步均为显式向量内在（与 §1.6.2 结论一致）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `m_i = max_j x[i,j]` | 行最大（fp32；lane 累加 + 单次 vcmax 全归约；尾组 −inf 幂元） |
| 2 | `e_ij = exp(x[i,j] − m_i)` | fp32 稳定化指数（vexpdif 融合单指令，UB 流式重装载） |
| 3 | `S_i = Σ_j e_ij` | fp32（lane 累加 + 单次 vcadd 全归约） |
| 4 | `y_i = cast(m_i + log(S_i), dtype)` | 每行 1 次 special + 1 次 cast + ONEPT 单点存储 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 层级 |
|------|----------|-------------|------|------|
| 子批搬入 | X 子批 GM→UB | `T.copy(x[base : base+rows, 0:N], x_ub[0:rows, 0:N])` | 2D 连续行块区域拷贝（列 0:N 全覆盖 → GM 连续区间；rows ∈ {SUB, rem} 静态；dst 行距 N_pad、src 行距 N——独立 pitch，skill §6.1） | MTE2（AutoSchedule） |
| 输入装载 + 提升 | x_ub → 64-lane fp32 | 16-bit：`T.simd.vld(x_ub[r, g*64], dist="UNPK_B16")` → `T.simd.vcvt(xh, "float32", part=0)`；fp32：`T.simd.vld(x_ub[r, g*64])` 直连 | UNPK_B16 组宽 64 元素（footprint 128B）；vcvt part=0 单次到位（位级回归锚点 §3.4） | SimdVF（VALU） |
| 尾组掩码 | 边界 −inf 中性化 | `c = T.simd.vsel(c, neg_inf, T.simd.update_mask(k_tail, 32))`（仅尾组，Python 静态分支） | neg_inf = `T.simd.vdup(-T.infinity(T.float32), "float32")`（块首绑定一次） | SimdVF |
| 行最大（max 遍） | m | `acc = T.simd.vmax(acc, c)` 逐组 → `m_brc = T.simd.vdupv(T.simd.vcmax(acc))` | acc 初始化 `vdup(−inf, f32)`；vcmax 单指令全归约；vdupv POS_LOWEST 广播 | SimdVF |
| 指数（融合） | e = exp(x−m) | `e = T.simd.vexpdif(c, m_brc)` | fp32 同宽（`simd.py` L1182–1195 强校验）；**融合单指令** | SimdVF |
| 行和（sum 遍） | S | `accs = T.simd.vadd(accs, e)` 逐组 → `s = T.simd.vcadd(accs)` | accs 初始化 `vdup(0.0, f32)`；vcadd 单指令全归约 | SimdVF |
| 行输出 | y = m + log(S) + cast | `out = T.simd.vadd(m_brc, T.simd.vln(s))` → `out_h = T.simd.vcvt(out, dtype)`（fp32 免） | 每行 1 次 vln（值在 lane 0 有效——归约内在结果低 lane 布局，ONEPT 恰取 lane 0） | SimdVF |
| lse 暂存 | VRF→UB 单点 | `T.simd.vsts(lse_ub[r], out_h, pset_16_vl1, "ONEPT_B16")`（fp32 输出 `"ONEPT_B32"` + `pset(32,"PAT_VL1")`） | ONEPT_B16 → `vsts_1st`（`codegen_ascend.cc` L2403–2404 dist 表）；pset_16_vl1 = `T.simd.pset(16, "PAT_VL1")` | SimdVF（VALU→UB） |
| 搬出 | UB→GM | `T.copy(lse_ub[0:rows], y[base : base+rows])` | 1D 静态切片 | MTE3（AutoSchedule） |
| 子批调度 | — | `with T.Kernel(56) as core_id: for t in T.Persistent([num_batches], 56, core_id, num_stages=2):` | persistent 子批 | — |
| 版本管理 | — | `T.annotate_buffer_versions({x_ub: 2})` | 子批双版本流水 | — |
| 尾批分支 | — | `if t < full: … else: …`（TIR if，循环变量条件） | full/rem JIT 常量 | — |

### 3.3 计算伪代码

```python
# host 静态化（builder 闭包，per-(M, N, dtype, block_m) JIT + 缓存）：
#   N_pad     = align_up(N, 64)                 # 组宽对齐（fp16/bf16: 128B 行距；fp32: 256B）
#   G_full    = N // 64                         # 满组数
#   k_tail    = N % 64                          # 尾组有效元素数（0 = 无尾组）
#   SUB       = max 2^k s.t. 2*SUB*N_pad*elem <= 228,556B and SUB <= block_m
#   full      = M // SUB;  rem = M % SUB;  num_batches = full + (1 if rem else 0)
#   NUM_BLOCKS = 56                             # = vector_core_num（实查，§5.5）
# 契约防御（builder 内 assert）：N <= 32512（single 契约）；M, N, block_m >= 1

@T.prim_func
def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M,), dtype)):
    with T.Kernel(56) as core_id:
        x_ub = T.alloc_shared((SUB, N_pad), dtype)      # 紧凑驻留（双版本流水）
        lse_ub = T.alloc_shared((SUB,), dtype)          # 行输出暂存（单版本）
        T.annotate_buffer_versions({x_ub: 2})
        for t in T.Persistent([num_batches], 56, core_id, num_stages=2):
            if t < full:                                 # ---- 全批（TIR if）----
                base = t * SUB
                T.copy(x[base : base + SUB, 0:N], x_ub[0:SUB, 0:N])
                with T.SimdVF():
                    <行循环体 rows=SUB（下）>
                T.copy(lse_ub[0:SUB], y[base : base + SUB])
            else:                                        # ---- 尾批（rem 行）----
                base = M - rem
                T.copy(x[base : M, 0:N], x_ub[0:rem, 0:N])   # 静态 rem 切片
                with T.SimdVF():
                    <行循环体 rows=rem（下）>
                T.copy(lse_ub[0:rem], y[base : M])

# —— 行循环体（SimdVF 块内，rows 为闭包常量；fp16/bf16 形态，fp32 见注）——
    neg_inf = T.simd.vdup(-T.infinity(T.float32), "float32")   # 块首绑定
    zero_f  = T.simd.vdup(0.0, "float32")
    one_pt  = T.simd.pset(16, "PAT_VL1")                # fp32 输出: pset(32, "PAT_VL1")
    for r in range(rows):
        # ---- pass 1: 行最大（延迟跨 lane 归约）----
        acc = neg_inf                                    # vdup(−inf) 初始化
        for g in range(G_full):                          # 满组直载（无谓词）
            xh = T.simd.vld(x_ub[r, g * 64], dist="UNPK_B16")
            c  = T.simd.vcvt(xh, "float32", part=0)
            acc = T.simd.vmax(acc, c)
        if k_tail > 0:                                   # Python 静态分支（闭包常量）
            xh = T.simd.vld(x_ub[r, G_full * 64], dist="UNPK_B16")
            c  = T.simd.vcvt(xh, "float32", part=0)
            c  = T.simd.vsel(c, neg_inf, T.simd.update_mask(k_tail, 32))
            acc = T.simd.vmax(acc, c)
        m_brc = T.simd.vdupv(T.simd.vcmax(acc))          # 单次全归约 + lane0 广播
        # ---- pass 2: 稳定化指数 + 行和（UB 重读）----
        accs = zero_f
        for g in range(G_full):
            xh = T.simd.vld(x_ub[r, g * 64], dist="UNPK_B16")
            c  = T.simd.vcvt(xh, "float32", part=0)
            e  = T.simd.vexpdif(c, m_brc)                # 融合 exp(c − m)
            accs = T.simd.vadd(accs, e)
        if k_tail > 0:
            xh = T.simd.vld(x_ub[r, G_full * 64], dist="UNPK_B16")
            c  = T.simd.vcvt(xh, "float32", part=0)
            c  = T.simd.vsel(c, neg_inf, T.simd.update_mask(k_tail, 32))
            e  = T.simd.vexpdif(c, m_brc)
            accs = T.simd.vadd(accs, e)
        # ---- 行尾 epilogue ----
        s   = T.simd.vcadd(accs)                         # 单次全归约（值在低 lane）
        out = T.simd.vadd(m_brc, T.simd.vln(s))
        out_h = T.simd.vcvt(out, dtype)                  # fp32 输出: 免 cast
        T.simd.vsts(lse_ub[r], out_h, one_pt, "ONEPT_B16")   # fp32: "ONEPT_B32"
# 注（fp32 输入形态）：装载步替换为 c = T.simd.vld(x_ub[r, g*64])（NORM 直连 64-lane
# fp32），无 vcvt；其余全同。注（行循环体在全批/尾批两分支中各实例化一次，rows 不同）。
```

> 伪代码与 §1.6.3 布局决策、§4.3 buffer 形状、§6 循环结构三方一致（组循环内层向量维 = 64 元素组；x_ub (SUB, N_pad)；两分支 rows ∈ {SUB, rem}）。尾批的 UB 行 [rem, SUB) 不装载不计算不写出——行级 OOB 由静态切片规避（无行谓词，§0.6 R4）。

### 3.4 API 可行性确认

| API | 佐证（examples / testing / 源码） |
|-----|----------------------------------|
| `T.Kernel(56)` 一维 + 无 threads | `examples/ascend/example_per_token_cast_to_fp8.py` L117；ascend-constraints §1 条 1；tilelang-ascend skill §2.1 |
| `T.Persistent([n], 56, core_id, num_stages=2)` | `example_per_token_cast_to_fp8.py` L118–124（SimdVF 族同构）；`examples/pilot/online_softmax/online_softmax.py` L60 |
| `T.SimdVF()` 块（无线程参数） | `example_per_token_cast_to_fp8.py` L132；`flash_attention/core.py` L94；tilelang-ascend skill §2.3 |
| `T.simd.vld(addr, dist="UNPK_B16")`（16→32 位提升装载） | `testing/ascend/language/test_tilelang_ascend_simdvf_cast.py` L67/L78（f16→f32、int16→int32 位级回归 atol=0）；`tilelang/ascend/language/simd.py` L302–321（dist 定义：footprint = lanes//2 = 64 元素）+ L324（签名） |
| `T.simd.vcvt(src, "float32", part=0)`（widening 单次到位） | `test_tilelang_ascend_simdvf_cast.py` L68/L79（与 UNPK_B16 成对，位级验证）；`simd.py` L1209（签名）+ L1340–1349（f16/bf16→f32 part_x2 路径） |
| `T.simd.vcvt(out, "float16"/"bfloat16")`（narrowing 输出） | `test_tilelang_ascend_simdvf_cast.py` L50–53（f32→f16 + PK_B32 存储位级）；`flash_attention/core.py` L139–142（f32→bf16，sat=False 场景） |
| `T.simd.vmax / vadd` | `simd.py` L815 / L554；`flash_attention/core.py` L106–113 / L134–135 |
| `T.simd.vcmax / vcadd`（**单指令跨 lane 全归约**） | `src/tl_templates/ascend/simd_inst.h` L688–710（→ `asc_reduce_max` / `asc_reduce_sum`）；`flash_attention/core.py` L115–116/L134–135（`vcmax(vmax(s00,s01))` → ONEPT_B32 存 lane 0 = 行 max，NPU 实测在用）；`example_per_token_cast_to_fp8.py` L142–143 |
| `T.simd.vdupv`（lane 0 广播） | `simd.py` L933（POS_LOWEST）；`example_per_token_cast_to_fp8.py` L146（`scale_brc = vdupv(scale)` 同构） |
| `T.simd.vdup(scalar, dtype)` | `simd.py` L922；`example_per_token_cast_to_fp8.py` L133–134（vdup(0.0001, "float32")）；bf16 vdup 存在性：`simd.py` 模块头注记（"Scalar BF16 vdup retains software merging"，CANN 9.2 适配在案） |
| `T.simd.vexpdif(c, m_brc)`（融合 exp(a−b)，fp32-only） | `simd.py` L1182–1195（fp32 同宽强校验——非 fp32 显式 TypeError）；`flash_attention/core.py` L130–133（softmax 行 exp(x−max) 实测在用）；VP-2026-0014 三步查证交叉印证 |
| `T.simd.vln` | `simd.py` L867（precision 语义同 vexp）；本设计 fp32 域使用 |
| `T.simd.vsel(src0, src1, mask)`（谓词选择） | `simd.py` L1429（Bitwise select: mask ? src0 : src1）；`examples/ascend/deepgemm/kernels/transform_sf.py` L205/L215/L249（vld 后 vsel 常量掩码——本设计尾组掩码同构） |
| `T.simd.update_mask(k, 32)`（运行时尾谓词，常量折叠） | `simd.py` L125；`testing/ascend/language/test_tilelang_ascend_simdvf_integer_predicate.py` L261（update_mask(cnt, 32) 在用） |
| `T.simd.pset(16, "PAT_VL1")` | `simd.py` L111（PAT_VL1..VL128）；`flash_attention/core.py` L96（`pset(32, "PAT_VL1")` softmax_one 同构） |
| `T.simd.vsts(addr, src, mask, "ONEPT_B16")`（单点存储） | `src/ascend/codegen/codegen_ascend.cc` L2403–2404（dist 表：ONEPT_B8/B16/B32 → `vsts_1st`）；`flash_attention/core.py` L117（`vsts(m_ub[r], max0, softmax_one, "ONEPT_B32")` 同构）；`example_per_token_cast_to_fp8.py` L147（ONEPT_B32 无 mask 形态） |
| `T.copy` GM→UB 2D 静态切片（紧凑 N + dst 行距 N_pad） | `example_per_token_cast_to_fp8.py` L129；`online_softmax.py` L62/L142（静态切片先例）；tilelang-ascend skill §6.1（字节级行长 + 独立 src/dst pitch） |
| `T.copy` UB→GM 1D 切片 | `example_per_token_cast_to_fp8.py` L154；`example_rmsnorm.py` L82 |
| `T.annotate_buffer_versions` | `example_per_token_cast_to_fp8.py` L128；`online_softmax.py` L59 |
| 核内 TIR if（循环变量条件） | `example_gemm.py` L43–56；**SIMT 兄弟变体 `_logsumexp_kernel_single.py` L138–238（Persistent + TIR if + serial + VF 嵌套，86+61 用例全 PASS 实证）** |
| Python 级常量 if（prim_func 体内，含嵌套作用域） | **源 GPU kernel 自身**（`logsumexp.py` L68 `if _needs_pad:` / L144 `if _needs_mask:`——闭包常量分支在 prim_func 体内的先例即源码本身）；`online_softmax.py` L141 |
| `for r in range(rows)` / `for g in range(G_full)`（串行 TIR 循环） | `example_per_token_cast_to_fp8.py` L135–136（`for i in range(blk_m): for j in range(group_block):` SimdVF 内 range 循环先例）；`P-LAYOUT-TAIL-STATIC-EXTENT`（range 解析为串行循环，勿展开） |
| `@tilelang.jit(target="ascend", out_idx=[1])` 懒工厂 | `online_softmax.py` L215–223（`_is_lazy_style` 形态，S3-JIT 门禁要求——CASE-pilot-online-softmax 教训） |
| `-T.infinity(T.float32)`（TIR 标量 −inf） | `online_softmax.py` L73（SimtVF 掩码载入在用）；TIR FloatImm 支持 ±inf |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（ascend-constraints §2 强制检测）

| 检测项 | 本算子结论 |
|--------|-----------|
| 一维 Kernel / threads= 误用 | ✅ 源含 `T.Kernel(grid, threads=threads)`（二维参数 + threads）——**已重设计**：一维 `T.Kernel(56)` + SimdVF 无线程域（§0.6 R1/R5）；调用面无 threads（K3，SimdVF 连内部推导也无） |
| GPU 专用 API | ✅ 无 TMA/cp.async/warp shuffle/cluster 依赖；SMEM staging 等价替换为 UB 单跳 + UNPK/vcvt 装载（§0.5）；warp 树归约等价替换为 vcmax/vcadd 跨 lane 全归约 |
| C0 分形对齐 | ✅ 不适用（无 GEMM/Cube 段；`CONST-gemm-dim-constraints` 仅约束 `T.gemm` 路径） |
| L0C/UB 溢出 | ✅ 无 L0C；UB 预算 §4.5（x_ub 双版本 + lse_ub ≤ 223.2KB——SimdVF 248KB × 90% 折扣内，`CONST-capacity-a5`）；VRF ~2.5KB 零压力 |
| OOB padding 误用 | ✅ 不使用 `pad_value=`/`data_select=`（列向 32B 行尾 padding 语义——本设计组尾掩码用 vsel/update_mask 在 fp32 域完成，不依赖 DMA padding；行向 OOB 用静态切片规避——skill §6.1） |
| 分核策略缺失 | ✅ §5.5 三要素齐备（grid 56 实查 / 纯 AIV 分工 / auto_schedule 声明） |
| 静态循环边界 | ✅ 全部 JIT 常量（num_batches/G_full/k_tail/SUB/rows∈{SUB,rem}）；核内串行循环边界静态 |
| flag id 预算 | ✅ auto_schedule yes，无手动 flag（§7） |
| Host 侧输入操作 | ✅ host 仅做 shape/元数据推导（N_pad/SUB/G_full/k_tail/full/rem）与 JIT 缓存；不改输入张量内容 |

#### 3.5.2 参考实现差异说明

源 GPU 实现与本设计的差异全部源于 §0.5/§0.6 迁移决策（N_padded→align64 紧凑、双路径→单路径掩码、grid→persistent 子批、threads→SimdVF 指令流、fragment 驻留→VRF 流式、隐式 OOB→静态分解），公式与数值语义恒等（verify_equiv.py V1/V4）。**不复用源码任何 GPU 形态结构**（`T.Kernel(grid, threads)`、SMEM staging、`_needs_pad` 逐元素分支、warp 树归约）。与 SIMT 兄弟变体的差异均为承载模式差异（§2.2），调用面与契约空间一致。

#### 3.5.3 本项目同类实现参考

| 参考文件 | 复用内容 |
|---------|---------|
| `examples/ascend/example_per_token_cast_to_fp8.py` | **SimdVF 逐行规约完整骨架**：persistent + annotate_buffer_versions + GM→UB copy + SimdVF 块（vld/vmax/vcmax/vdupv/vcvt/vsts）+ UB→GM copy + (M,) 一维输出两段写 |
| `examples/ascend/flash_attention/core.py` | SimdVF 行 max/sum 归约指令族（`vcmax(vmax(s00,s01))` / `vcadd(vadd(e00,e01))` / ONEPT 存储 / `vexpdif` 融合 / BRC 广播 / pset 常量掩码）——L94–137 softmax 段与本设计行处理同构 |
| `testing/ascend/language/test_tilelang_ascend_simdvf_cast.py` | UNPK_B16 + vcvt(part=0) 16→32 位提升装载的位级回归锚点（f16/bf16 路径） |
| `examples/ascend/deepgemm/kernels/transform_sf.py` | vld 后 vsel 常量向量掩码（尾组 −inf 掩码同构，L205/L215/L249） |
| `examples/ascend/example_gemm.py` | 核内循环变量 TIR if 先例（L47） |
| `examples/pilot/online_softmax/online_softmax.py` | JIT 懒工厂形态 / 静态 rem 切片 / 分层测试组织 |
| SIMT 兄弟变体 `_logsumexp_kernel_single.py`（`examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp_kernel/`） | 调用面 / 契约 assert / golden + 源语义模拟器 / 边界测试组织 / Persistent+TIR if 嵌套实证（只读参照，严禁改动） |

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
| x_ub | (SUB, N_pad)（wl1 = (8, 4096)、wl4 = (2, 16384)；N_pad = align_up(N, 64)） | 输入 dtype | UB（`T.alloc_shared`，双版本） | 行子批 DMA 暂存（紧凑 N 列 + 组尾对齐 pad；pad 列 [N, N_pad) 不初始化——被尾组 vsel −inf 掩码中性化） |
| lse_ub | (SUB,)（wl1 = (8,)） | 输入 dtype | UB（单版本） | 行输出暂存（vsts ONEPT 逐行写入 → T.copy UB→GM） |
| VRF 向量寄存器（SSA 绑定，无 alloc 调用） | ~10 × 256B | fp32x64 / dtypex128 | VRF | acc/accs（累加器）、c/e（组数据）、m_brc/s/out（统计量）、neg_inf/zero_f（常量广播）、one_pt（谓词）——**零行数据驻留** |

### 4.4 内存搬运路径

```
GM[x (M,N) dtype]
  --T.copy (MTE2, 2D 连续行子批 [base:+rows, 0:N], x_ub 双版本 v/v⊕1)--> UB[x_ub (SUB,N_pad)]
  ==SimdVF（每行两遍，VRF↔UB 流式）==
  pass1: --vld(UNPK_B16)--> VRF[x_h] --vcvt--> VRF[c fp32] → acc_max = vmax(acc_max, c) 逐组
         （尾组先 vsel(−inf, update_mask)）→ m_brc = vdupv(vcmax(acc_max))
  pass2: --vld(UNPK_B16)--> VRF[x_h] --vcvt--> VRF[c] → e = vexpdif(c, m_brc)
         → acc_sum = vadd(acc_sum, e) 逐组（尾组同掩码）→ s = vcadd(acc_sum)
  epilogue: out = vadd(m_brc, vln(s)) → vcvt(dtype)
  --vsts (ONEPT_B16, VALU 单点)--> UB[lse_ub (SUB,) dtype]
  --T.copy (MTE3, 1D 静态切片 [base:+rows])--> GM[y (M,)]
```

无 L1/L0A/L0B/L0C 段（纯 Vector）；无中间量落 GM；x 单遍 GM 读（两遍重读在 UB↔VRF 间）、y 单遍 GM 写；m/s 统计量寄存器驻留（无 UB 往返 → 无 `mem_bar` 需求——对照 flash core.py L120 的 VST_VLD barrier 仅在其 m_ub UB 往返形态下需要，本设计不采用该形态）。

### 4.5 UB / VRF 内存预算

**UB 预算**（上限 **248KB**〔SimdVF——无 SimtVF 32KB 线程上下文扣减，`CONST-capacity-a5`〕，assumption 按 90% 折扣 = **228,556B**〔= int(248×1024×0.9)，即 223.2KB——与 SIMT 兄弟变体 `int(216×1024×0.9)=199,065B` 同一口径〕）：

| Buffer | wl1 (N=4096, fp16, SUB=8) | wl4 (N=16384, fp16, SUB=2) | 版本 | 说明 |
|--------|---------------------------|---------------------------|------|------|
| x_ub | (8, 4096)×2B×2 版本 = 131,072B | (2, 16384)×2B×2 版本 = 131,072B | 2 | SUB=8 / SUB=2（§5.2 推导） |
| lse_ub | (8,)×2B = 16B | (2,)×2B = 4B | 1 | 写出暂存 |
| 编译器保留裕量 | ~8KB 量级 | 同左 | — | 调度器保留（skill §4.3） |
| **总计** | **≈139KB / 223.2KB（61%）✓** | **≈139KB / 223.2KB（61%）✓** | | |

**SUB 上界推导**（host 静态化公式，§3.3 注）：`SUB = 满足以下约束的最大 2 的幂：2·SUB·N_pad·elem ≤ 228,556B ∧ SUB ≤ block_m`。该式保证**契约空间全域**（任意 N ≤ 32512、任意 block_m、三 dtype）UB 双版本达标：fp32 时 elem=4B 使约束更紧（N=4096 → SUB ≤ 6.98 → 2 幂封顶 SUB=4 → 131KB ✓）；N=32512 fp16 → SUB ≤ 1.76 → SUB=1（65KB ✓）；非对齐 N=300 fp16 → N_pad=320 → SUB ≤ 178 → SUB=min(block_m,128)（20.5KB ✓）。**与 SIMT 兄弟变体对照**：SIMT 的 SUB 受双约束（UB 194.4KB + VRF x_frag ≤128KB）；本变体仅 UB 单约束（VRF 零驻留）——主要工作点 SUB 落点相同（wl1=8 / wl4=2，均由 UB 双版本 131KB 决定），差异显现在大 TILE 小 N 场景（SIMT 的 VRF 约束 SUB·TILE·4B 在非 pow2 N 时 TILE 膨胀压 SUB，本变体无此效应——N=300：SIMT SUB ≤ min(199,065/(2·300·2)=165.9, 131,072/(512·4)=64) = 64 → 实取 min(block_m,64)；本变体 SUB=min(block_m,128) 无 VRF 项）。

**VRF 预算**（假设上限 256KB，`CONST-vrf-capacity-a5` assumption；本变体为**流式零驻留**形态）：

| 项 | 数量 | 字节 | 说明 |
|-----|------|------|------|
| 活跃向量寄存器（acc/accs/c/e/m_brc/s/out/neg_inf/zero_f/out_h） | ~10 | ~2.5KB | SSA 绑定，编译器寄存器分配；**无 alloc_local 数组、无 x 数据驻留** |
| 对照 SIMT 兄弟变体 | x_frag (SUB, TILE) fp32 ≤128KB 封顶 | 128KB | SIMT 需 SUB 封顶压 VRF；本变体该约束消失（§0.6 R2 收益） |

**对照基线（寄存器驻留单装载候选，§1.6.3 #6 弃选）**：VRF 4·N·SUB = wl1 128KB——回到 SIMT x_frag 贴顶水位（`CONST-vrf-capacity-a5` 实测样本 76 reg/thread 无 spill 恰在同水位，无余量）；UB 重读代价 0.016 cycle/elem（D-2，13× 余量非瓶颈）——弃选依据。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| M | JIT 闭包参数（`T.Tensor((M, N), ...)` 符号维度），per-shape 编译 + 缓存 | ≥ 1 任意（含 M < SUB：full=0 全尾批路径；M % SUB ≠ 0 尾批） |
| N | 同上 | 契约空间：N_padded ≤ 32512 且 wrapper 预算内（§0.2）；kernel 侧 assert 防御 |
| block_m | `_func(block_m)` 调用参数（per-config 编译） | wrapper 启发式 {1,2,4,8,16} + 显式 config 任意 ≥1（SUB 封顶兜底） |
| dtype | JIT 闭包参数 | float16 / bfloat16 / float32 |

### 4.7 JIT 配置与调用面契约

```python
import tilelang


def _logsumexp_kernel_single(M: int, N: int, dtype: str):
    """Build a single-tile logsumexp kernel (N fits on-chip) — A5 SimdVF re-implementation.

    Keeps the wrapper call surface `_logsumexp_kernel_single(M, N, dtype)(block_m)(x)`;
    no `threads` argument exists in any form (SimdVF has no thread concept, K3).
    """

    # 契约防御：assert N <= 32512（single 契约）, M >= 1, N >= 1, block_m >= 1
    # host 静态化：N_pad / G_full / k_tail / SUB / full / rem / num_batches / NUM_BLOCKS
    @tilelang.jit(target="ascend", out_idx=[1])
    def _func(block_m):
        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M,), dtype)): ...  # §3.3 伪代码

        return main

    return _func


# wrapper 对接（SIMD 家族 logsumexp_simd.py，Stage 5 集成）：
#   kernel = _logsumexp_kernel_single(M, N, dtype_str)(block_m)   # (block_m) JIT 编译
#   y = kernel(x)                                                 # (x) 执行，out_idx=[1] 返回 y
```

- **JIT 形态**：`@tilelang.jit(target="ascend", out_idx=[1])` 懒工厂（工厂返回 PrimFunc，`_is_lazy_style` 识别）——S3-JIT 门禁要求（CASE-pilot-online-softmax 教训）；per-(M, N, dtype, block_m) 编译由 jit 装饰器内建缓存承接（wrapper 侧另有 dispatch 缓存）。
- **K3 落地**：`_func(block_m)` 单参数；SimdVF 模式下无 threads 概念（连 SIMT 版的内部 `threads = min(1024, SUB·TILE)` 推导也不存在）。
- **host 防御**（TRAP-GRID-DIVISIBILITY 纪律）：builder 内 `assert` 契约（N ≤ 32512 单 tile 契约、block_m ≥ 1、M ≥ 1、N ≥ 1）——persistent 调度器保证批全覆盖（无手动 grid-stride 漏写风险），assert 仅拦契约外误用。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 算子仅含逐元素运算（cast/exp）与行内规约（max/sum），无 matmul——无 Cube 段；全部计算以 SimdVF 显式 2048-bit 指令流落 AIV（56 个），搬运落 MTE2/MTE3（AIV 附属引擎）。与 SIMT 兄弟变体判定一致（§1.6.3 候选 10 已排除 Cube GEMV 归并——转换开销 + 归约非瓶颈）。

### 5.2 Block 划分

```python
N_pad      = align_up(N, 64)             # 组宽对齐（wl1 4096 / wl4 16384；N=300 → 320）
G_full     = N // 64                     # 满组数（wl1 64 / wl4 256；N=300 → 4）
k_tail     = N % 64                      # 尾组有效元素（wl1/wl4 0；N=300 → 44）
SUB        = max 2^k s.t. 2*SUB*N_pad*elem <= 228,556B and SUB <= block_m
                                           # wl1 (fp16): SUB=8；wl4 (fp16): SUB=2；
                                           # (32,256,fp32,bm=16): SUB=16=block_m（无批分裂）
full       = M // SUB                    # wl1: 128；wl4: 64
rem        = M % SUB                     # wl1/wl4: 0；(33,256,bm=16): rem=1
num_batches = full + (1 if rem else 0)   # persistent 域（wl1: 128 ≈ 2.29 批/核；wl4: 64）
NUM_BLOCKS = 56                          # = vector_core_num（实查，§5.5）
```

- **SUB 依据**（单约束封顶，§4.5）：① x_ub 双版本 `2·SUB·N_pad·elem ≤ 228,556B`（SimdVF UB 248KB × 90%，解锁批间 MTE2/VALU 流水）；② SUB ≤ block_m（契约参数不被放大）。**与 SIMT 兄弟变体的关键差异：无 VRF 第二约束**——SIMT 需 `SUB·TILE·4B ≤ 128KB` 压 x_fragment，本变体 VRF 流式零驻留（~2.5KB），SUB 封顶仅剩 UB 单约束（§0.6 R2 收益的 tiling 侧体现）。主要工作点两变体落点相同（wl1=8 / wl4=2，均由 UB 双版本 131KB 决定）；差异显现在非 pow2 N（SIMT 的 TILE 膨胀压 SUB，本变体无此效应，§4.5 对照）。
- **block_m 契约来源**：wrapper `default_config`（single 优先最大 block_m：wl1 → 16、wl4 → 4）或 caller 显式 config——kernel 侧不重选、不假设其整除 M（尾批路径 §5.4）。
- **驻留上限（契约边界）**：wrapper 判定 N_padded ≤ 32512 ∧ block_m·elem·N_padded ≤ 预算 进 single（§0.2）；N 超界走 SIMD 家族 tiled 变体（另任务，承接 attn-weights-32k / lm-head-logits 两个 workload）。kernel 侧无 N 维分块（单 tile 语义）。
- **Stage 4 校准指引**（P-LAYOUT-ROW-RESIDENCY measured，SIMT 侧 Stage 4 实测回写）：persistent 批粒度 = SUB 行，**makespan-only 模型缺 per-tile 固定开销项**（每批的微型 MTE3 写出 flag 握手 + vf_call 发射 + MTE2 issue 随批数线性放大）——修订规则「候选限 pow2 SUB，取 makespan ≤ min+1 行内的最大 SUB」：wl1（M=1024）甜点预期 **SUB=4（block_m=4，256 批 / makespan 20 行）**，SIMT 侧同粒度实测 +50.9% vs bm=16 基线；wl4（M=128）对粒度不敏感（tie）。block_m 是 wrapper config 旋钮（不改 kernel 结构、不触容量约束——SUB 封顶兜底），Stage 4 扫 {4, 8, 16} 实测裁决（§8.3）。

### 5.3 约束分析

- **C0 分形对齐**: 不适用（无 GEMM/Cube 段；`CONST-gemm-dim-constraints` 仅约束 `T.gemm` 路径）。
- **UB 容量**: §4.5 预算表（wl1/wl4 ≈139KB ≤ 223.2KB，`CONST-capacity-a5` assumption 90% 折扣）✓。
- **VRF 容量**: §4.5（~2.5KB 流式零驻留，vs SIMT 128KB x_frag 封顶——本变体无 VRF 约束层）✓；Stage 3 编译后仍首查 spill 计数（`get_kernel_source()`）复核。
- **GM→UB 对齐**: wl1 行 8KB / wl4 行 32KB，32B 对齐 ✓；批 2D 切片列 0:N 全覆盖 → GM 连续区间（单段 DMA）；dst 行距 N_pad×elem 与 src 行距 N×elem 独立 pitch（skill §6.1）。非 32B 行长（N=300 fp16 = 600B/行，dst 640B/行）：DMA 字节级行长支持（正确性不受影响），带宽档降级列 §9.1 风险 2。
- **向量宽度整除**: 组宽 64 元素（fp16/bf16 UNPK_B16 footprint 128B；fp32 NORM 256B）；wl1/wl4 N % 64 = 0（零尾组）✓；非 64 倍数 N（300 → G_full=4 + k_tail=44）由尾组 vsel 掩码吸收（fp32 域，§1.6.1 O2）；组起始偏移 g·64·elem 恒 128B/256B 对齐 ✓。
- **静态循环边界**: num_batches / G_full / k_tail / SUB / rows ∈ {SUB, rem} / full / rem 全 JIT 常量 ✓（gate 约束；尾组分支为 Python 静态 `if k_tail > 0`，源 `_needs_pad` L68 同法先例）。

### 5.4 注意事项（非整除 / 边界处理策略）

- **行尾（M % SUB ≠ 0）**：静态 full/rem 分解 + 尾批 TIR if 分支（`if t < full`）——尾批静态切片 `[M−rem : M]`（GM 与 UB 两侧），UB 行 [rem, SUB) 不装载不计算不写出——**不依赖 `T.copy` 越界行为**（§0.6 R4；对照源 GPU 对齐路径的隐式 OOB 依赖）。**与 SIMT 兄弟变体对照**：SIMT 为 tile×子批两层 4 分支（full/rem × tsub/tsub_rem），本变体扁平化批粒度后仅 **2 分支**（批 full/rem）——子批层消失（persistent 直接以 SUB 行批为域元素），分支复杂度减半。
- **列尾（N % 64 ≠ 0）**：尾组 `vsel(c, vdup(−inf,f32), update_mask(k_tail,32))` fp32 域掩码（max/exp/sum 三幂元中性，V1 位级验证）——pad 列 [N, N_pad) DMA 不搬（紧凑 UB 驻留 −37.5% @N=300）且被掩码中性化；**不依赖 `T.copy` 的 pad_value/`data_select` DMA 填充语义**（列向行尾 padding 语义，行为文档化程度不足以承载数值路径——§3.5.1）。
- **M < SUB**（含 M=1）：full=0、rem=M → 单尾批路径全覆盖。
- **N < 64**（含 N=1）：G_full=0、k_tail=N → 纯尾组单掩码路径（V1 已覆盖 3×7 / 1×1）。
- **56 核尾波**：wl1 num_batches=128 = 2×56+16（末波 16/56 核，~1.8% 不均衡）；wl4 64 = 56+8（末波 8/56——M=128 负载小，绝对影响可忽略）。Stage 4 可调 block_m/SUB 对齐（core-split-strategy §2.4；均为 wrapper/host 旋钮，不改 kernel）。
- **grid 整除性**（TRAP-GRID-DIVISIBILITY）：persistent 调度器保证批全覆盖——无手动 grid-stride 的静默漏写风险；host assert 防御契约误用（§4.7）。

### 5.5 分核策略（grid 划分 + AIC/AIV 分工 + 同步策略）⭐

> 三要素判定标准与核数实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（设计要求见其 §2.1；依据 `.agents/skills/tilelang-ascend/SKILL.md` §2/§3/§7）。

- **grid 划分（含核数依据）**：`num_blocks = NUM_BLOCKS = 56`（一维 `T.Kernel(56)`，persistent 单波常驻）；行批任务 `num_batches = ceil(M/SUB)`（wl1: 128、wl4: 64）由 `T.Persistent([num_batches], 56, core_id, num_stages=2)` 调度器分配（wl1: 128/56 ≈ 2.29 批/核 + 尾波 16 批）。**核数实查记录**（2026-10-17，本设计执行期亲测）：
  ```python
  import torch

  props = torch.npu.get_device_properties()
  # name: Ascend950PR_9579, cube_core_num: 28, vector_core_num: 56
  ```
  纯 Vector 算子按 `vector_core_num = 56` 取 NUM_BLOCKS（persistent 单波常驻，无超发串行）；cube_core_num=28 记录备查（本算子不占 AIC）。无真机环境的复现命令即上述查询代码；本仓 examples 惯例初值 32 blocks 被实测值 56 取代（`CONST-arch-a5` measured 条目：28 = 4×7 非 2 幂——grid 整除性陷阱见 TRAP-GRID-DIVISIBILITY，本设计 persistent 形态免疫）。
- **AIC/AIV 分工**：纯 Vector 算子——SimdVF 显式指令流（`vld/vcvt/vmax/vcmax/vexpdif/vadd/vcadd/vln/vsel/vdup/vdupv/vsts`）与数据搬运（`T.copy` MTE2/MTE3）全部落 **AIV**（56 个）；**无 Cube 工作**（无 `T.gemm`/L1/L0 段）→ 不占 AIC（28 个闲置）。pure kernel 的核型由 codegen 自动判定（core-split-strategy §1 ②），无需 `T.MixedKernel`/sid 声明；无 dual_copy 所有权问题（无 L0C→UB 段）。
- **同步策略（auto|manual 声明）**：**auto_schedule: yes**（默认）——依赖 AutoSchedule 自动插 flag/同步，数据依赖边界：① x_ub 版本 v 的 MTE2 写 → SimdVF `vld` 读（跨引擎，自动 flag 对）；② lse_ub 的 VALU 写（`vsts`）→ MTE3 读（`T.copy` UB→GM，跨引擎，自动 flag 对）；③ x_ub 版本 v/v⊕1 交替（multibuffering，`T.annotate_buffer_versions({x_ub: 2})` + 编译器管理）；④ lse_ub 跨批复用（批 t 的 MTE3 读 → 批 t+1 的 `vsts` 写，WAR——AutoSchedule 顺序化/版本化）；⑤ 行内两遍与 epilogue 的顺序依赖（pass1 m 终值 → pass2 `vexpdif` 消费；S 终值 → `vln` 消费——VALU 内同引擎顺序语义）。**无 SimtVF 线程域 → 无 `asc_syncthreads` 需求**（SIMT 兄弟变体 §7.2 的跨线程 UB hazard 行在本变体不存在——SimdVF 单指令流无线程概念）。无手动 `T.ascend_set_flag/wait_flag`（核内 8/跨核 16 flag 预算不占用）；调度提示使用位置：`T.Persistent(num_stages=2)`（批间流水）——不使用 `T.Stage`/`T.Task`（AutoSchedule 足以覆盖本算子依赖结构）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部以 `T.simd.*` 2048-bit 显式内在向量化——本表仅批级/行级/组级调度循环；无逐元素标量循环。**`T.Parallel` 不使用**（SimdVF 无线程域，逐元素向量化由显式内在承担——与 SimtVF 兄弟变体的结构性差异）；**`T.Pipelined` 不使用**（批间流水由 `T.Persistent(num_stages=2)` 承接）；**`T.serial` 无域外用武之地**（扁平化后 persistent 即子批层；SimdVF 域内串行循环用 `range`——per_token L135–136 / flash core.py 域内先例，TVMScript 将 range 解析为串行 TIR 循环而非构建期展开，P-LAYOUT-TAIL-STATIC-EXTENT 注）。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 行子批（M 向） | persistent 调度 | `T.Persistent([num_batches], 56, core_id, num_stages=2)` | SUB 行批在 56 AIV 间持久调度 + 批间双缓冲流水 |
| 尾批分支 | 运行时控制流 | `if t < full: … else: …`（TIR if，循环变量条件） | M 尾批边界语义（§1.6.2 #9；example_gemm L47 + SIMT 兄弟变体 86+61 用例实证） |
| 行（批内） | 串行迭代 | `for r in range(rows)`（rows ∈ {SUB, rem} 静态） | §1.6.2 #7（行间寄存器状态独立；SimdVF 无行级并行原语；rows ≤ SUB ≤ 16） |
| 组（行内 × 2 遍） | 串行迭代 | `for g in range(G_full)` + Python 静态 `if k_tail > 0` 尾组分支 | §1.6.2 #8（两遍扫描顺序依赖：pass2 需 pass1 的 m 终值；组间流式依赖 UB 装载顺序；静态边界 ✓） |
| 元素级（N 向） | 2048-bit 显式向量化 | `vld/vcvt/vmax/vexpdif/vadd` 每组 1+1+1 指令（64 fp32 lane） | §1.6.2 #1–#5 全显式向量化 |
| 行内归约 | 跨 lane 全归约内在 | `vcmax`/`vcadd` 遍末单次（延迟归约 O4） | §1.6.2 #3/#5（单指令全 64-lane 收敛） |

### 6.2 循环伪代码

```python
# Block 级并行（一维 grid，56 persistent 核；核内完整结构见 §3.3）
with T.Kernel(56) as core_id:
    x_ub = T.alloc_shared((SUB, N_pad), dtype)
    lse_ub = T.alloc_shared((SUB,), dtype)
    T.annotate_buffer_versions({x_ub: 2})
    for t in T.Persistent([num_batches], 56, core_id, num_stages=2):
        if t < full:                                  # 全批（TIR if）
            T.copy(x[t*SUB : t*SUB+SUB, 0:N], x_ub[0:SUB, 0:N])
            with T.SimdVF():
                <行循环体 rows=SUB（§3.3：pass1 max 遍 → pass2 exp/sum 遍 → epilogue）>
            T.copy(lse_ub[0:SUB], y[t*SUB : t*SUB+SUB])
        else:                                         # 尾批（rem 行，静态切片）
            T.copy(x[M-rem : M, 0:N], x_ub[0:rem, 0:N])
            with T.SimdVF():
                <行循环体 rows=rem>
            T.copy(lse_ub[0:rem], y[M-rem : M])
```

### 6.3 流水线优化

`T.Persistent(num_stages=2)` + `T.annotate_buffer_versions({x_ub: 2})`：批 t+1 的 GM→UB 拷贝（MTE2，版本 v⊕1）与批 t 的 SimdVF 计算（版本 v）重叠；批间由 persistent 调度器连续供给。跨引擎依赖（MTE2→VALU→MTE3）由 AutoSchedule 自动插 flag 对（§7.2）；不手动管理双缓冲（tilelang-ascend skill §4.2：优先编译器 multibuffering）。**对照基线（单版本 x_ub 直译）**：MTE2 与 VALU 串行——流水收益 ≈ 流量项隐藏（D-2：wl1 流量项 1.5K cycle vs 计算项 15.2K，隐藏后端到端收益以计算项封顶）。**行内 UB 重读不占 MTE**（VRF↔UB 装载口 256B/cycle 独立于 MTE2/MTE3——批间流水不受两遍扫描影响）；行间指令级交错（r+1 的 pass1 与 r 的 pass2）属编译器调度层，语言层不表达（§1.6.2 #7，Stage 4 npusim 观察项）。

### 6.4 尾块处理

- 行尾（M % SUB ≠ 0）：静态 full/rem 分解 + TIR if + 两侧静态切片（§5.4；§0.6 R4 语义保持论证）。
- 列尾（N % 64 ≠ 0）：尾组 fp32 域 `vsel` −inf 掩码（三幂元中性，V1 位级验证）——无 OOB（DMA 紧凑搬运，pad 列被掩码替换后才进入算术）。
- 组尾（k_tail ∈ (0,64)）：`update_mask(k_tail, 32)` 运行时尾谓词（常量折叠），仅尾组 1 次、每行 2 次（两遍）。
- 批内行尾（尾批 rows=rem < SUB）：UB 行 [rem, SUB) 三不原则（不装载/不计算/不写出）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: auto_schedule: yes（AutoSchedule 自动同步；未设 `TL_ENABLE_AUTO_SCHEDULE: False`）

### 7.2 同步点说明

auto 模式下依赖边界说明（无手动同步点表；AutoSchedule 自动插入）：

| 依赖边界 | 涉及引擎/buffer | 自动机制 |
|----------|-----------------|----------|
| x_ub 写（MTE2，版本 v）→ SimdVF `vld` 读 | MTE2 → VALU，x_ub | AutoSchedule flag 对 |
| lse_ub 写（VALU `vsts`）→ MTE3 读（`T.copy` UB→GM） | VALU → MTE3，lse_ub | AutoSchedule flag 对 |
| x_ub 版本交替 v / v⊕1（批间双缓冲） | multibuffering | `T.annotate_buffer_versions({x_ub: 2})` + 编译器 |
| lse_ub 跨批复用（批 t MTE3 读 → 批 t+1 `vsts` 写，WAR） | MTE3 → VALU，lse_ub | AutoSchedule 顺序化/版本化 |
| 行内两遍顺序（pass1 m 终值 → pass2 `vexpdif`；S 终值 → `vln`） | VALU 内 | 同引擎顺序语义 |
| （无跨线程 UB hazard 行——SimdVF 单指令流，无 `asc_syncthreads` 需求） | — | — |

**m/s 统计量寄存器驻留 → 无 `mem_bar` 需求**：flash core.py L120 的 VST_VLD barrier 服务于其 m_ub UB 往返形态；本设计 m_brc/s 全程 VRF SSA 绑定（`vdupv` 广播后直接消费），不采用该形态（§4.4）。

**风险提示**（§9.1 风险 3）：`T.Persistent + TIR if（尾批分支）+ SimdVF` 嵌套组合无单一先例覆盖全部层级（各层分别有先例：SIMT 兄弟变体的 Persistent+TIR if+serial+SimtVF 86+61 用例全 PASS——AutoSchedule 对 if 分支两侧 multibuffering 的处理已被同构结构实证；per_token 的 Persistent+SimdVF+range+copy 全链）——若 AutoSchedule 对 SimdVF 域内 if 分支两侧的版本推断异常，fallback = 拆双 persistent 域（全批域 [full] + 尾批域 [1] + `core_id` 谓词），语义等价、API 全先例（§0.6 R4 预案）。

### 7.3 pass_configs 配置

```python
# 默认自动调度，不传 TL_ENABLE_AUTO_SCHEDULE（仅完全手动核需要 False）；
# JIT 入口即 §4.7 的 @tilelang.jit(target="ascend", out_idx=[1]) 懒工厂。
```

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（源仓测试基准 `torch.logsumexp(x.float(), dim).to(x.dtype)`——`tests/ops/test_softmax.py` L99 `ref_program` 直移植，SIMD 家族测试 `test_logsumexp_simd.py`〔manifest `source.test`，Stage 5 落地〕沿用同一基准模式），**不复刻 §0.6 的 NPU 算法**——保证验证独立性。

```python
def golden_logsumexp(x: torch.Tensor) -> torch.Tensor:
    """Row-wise logsumexp, torch on NPU（源仓 ref_program 直移植；fp32 计算域，
    输出 cast 回输入 dtype——与 §0.1 dtype 语义一致）。"""
    assert x.device.type == "npu" and x.ndim == 2
    return torch.logsumexp(x.float(), dim=-1).to(x.dtype)


def _source_lse_sim(x: torch.Tensor) -> torch.Tensor:
    """fp32 源语义模拟器（SIMT 兄弟变体 L66–77 同构直移植）：按 §0.1 源链
    m = max(x); y = m + log(sum(exp(x - m))) 以 fp32 中间量复算——自然复现源固有的
    ±inf 角点行为（全 −inf 行 → NaN、含 +inf 行 → NaN），用于 Boundary ④ 迁移保真
    对照（golden 对这些角点与源分歧，不能作对照基准）。"""
    x32 = x.float()
    m = x32.max(dim=-1, keepdim=True).values
    e = torch.exp(x32 - m)
    s = e.sum(dim=-1, keepdim=True)
    return (m + torch.log(s)).squeeze(-1).to(x.dtype)
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

**容差分档**（AGENTS.md / examples/TileOPs/docs/gpu_to_ascend_adaptation.md §1.8）：fp16/bf16 → atol=rtol=1e-2；fp32 → 1e-3（fp32 中间累加 + 单次跨 lane 归约，非 accumulation-heavy，取 1e-3 档）。

| 层级 | 用例设计（依据：manifest workloads + §0.2 落点分析） |
|------|----------------------------------------------------------------------|
| **L0（门槛）** | ① wl1 等价 (1024, 4096) fp16 block_m=16（bench 主对象；SUB=8/128 批/G=64）；② wl4 等价 (128, 16384) fp16 block_m=4（SUB=2/64 批/G=256）；③ smoke (32, 256) 三 dtype（fp32 路径 NORM 直连 + bf16 宽化装载首验——风险 1；SUB=16=block_m 无批分裂形态）；④ (33, 256) fp16 bm=16（M 尾批 rem=1——TIR if 尾批路径 + 风险 3 首触）；⑤ (32, 300) fp16 bm=16（非 64 倍 N：N_pad=320 / G_full=4 / k_tail=44——尾组 vsel 路径 + 风险 6 首触） |
| **L1（shape/config 扫描）** | M ∈ {1, 3, 33, 56, 57, 128, 1024} × N ∈ {1, 7, 256, 300, 4096, 16384} 三 dtype；block_m ∈ {1, 2, 4, 8, 16}（wrapper 启发式全集 + 极端 1）；SUB 分裂点验证（block_m > SUB〔wl1 bm=16→SUB=8〕与 block_m = SUB〔(32,256) bm=16→SUB=16〕两形态）；N < 64（G_full=0 纯尾组：N=1/N=7）；M=1（全尾批）；1D 输入 flatten (1, 256)；多维 dim=[0,2] flatten (4, 128, 4096) → (128, 16384) |
| **L2（稳健性/压力）** | 大 M 多波（M=8192, N=4096, bm=16 → num_batches=1024 ≫ 56，19 波 persistent 调度）；契约边界 N 邻域（(2, 32512) fp16：SUB=1 退化形态、G_full=508、2·1·32512·2=130,048B ✓）；N=32513 越界负向 assert 验证；fp32 大 N（(128, 16384) fp32 → SUB=1、128 批——elem=4B 紧约束形态） |
| **Boundary（角点）** | ① 有限域角点（对照 golden）：全零行 / 全等行 3.5 / 单主导 100 / ±65504 极值（fp16）/ subnormal ±6e-8 / 深度下溢 −745 / N=1 / M=1 / 确定性（3× 位级一致）；② 部分 −inf 正常行（golden 一致 ✓ V1/V4）；③ NaN 行（golden 双方 NaN，equal_nan=True 对照 ✓ V4）；④ **源继承分歧角点**（对照 `_source_lse_sim` 而非 golden，迁移保真验证）：全 −inf 行 → NaN（torch 为 −inf）、含 +inf 行 → NaN（torch 为 +inf）——V4 已证设计链与源位级一致（fp64→16-bit 输出 + NaN 模式）；⑤ 非连续输入（op 层 contiguous 化后等价）；⑥ 尾组 + 尾批双触发（(33, 300) fp16 bm=16：rem=1 ∧ k_tail=44） |

**已知边界分歧处理**（§9.3 详述）：全 −inf / 含 +inf 行为源算法固有（`x − m` 产生 NaN 链），与 `torch.logsumexp` 分歧；源仓测试域（randn 生成）不触发，本设计保持源语义（迁移保真——与 SIMT 兄弟变体行为一致是双模式对照的硬约束）并在 Boundary ④ 用源语义模拟对照证明行为一致。

### 8.3 性能验证（Stage 4，设计期仅记录口径）

- 测量权威：`tilelang.profiler.do_bench(fn, backend="msprof")`；代表 shape = wl1 (1024, 4096) fp16 / wl4 (128, 16384) fp16（bench_logsumexp_simd.py manifest 驱动，bench_manifest_driven）。
- 基线对照：`torch.logsumexp(x, dim=-1)`（native op，baseline_fn 同款）。
- **A/B 必做项（本变体特有）**：与 SIMT 兄弟变体同 shape 对照（`examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp_kernel/_logsumexp_kernel_single.py`，双模式对照的任务定位——同契约 shape、同 golden、同 bench 口径）；SIMT 侧 Stage 4 基线 attn-weights-4k 21.4µs / 3d-multidim ~19µs（perf_opt 采纳版，2026-10-17）。
- block_m 扫描 {4, 8, 16}（P-LAYOUT-ROW-RESIDENCY 校准指引：wl1 甜点预期 SUB=4 → bm=4；§5.2）。

---

## 9. 风险点与注意事项

### 9.1 已知约束（本算子相关）

| # | 风险 | 影响 | 缓解 / 验证 |
|---|------|------|------------|
| 1 | **bf16→f32 宽化装载（UNPK_B16 + vcvt part=0）无独立位级回归**——cast 回归覆盖 f16→f32 / int16→int32（`test_tilelang_ascend_simdvf_cast.py` L64–80），bf16 仅组合证据（UNPK 机制 dtype 无关的 16-bit 解包 + `simd.py` L1340–1349 bf16 part_x2 路径注册） | bf16 输入路径的正确性依赖组合推理而非单测锚点 | L0 ③ (32,256) 三 dtype smoke 即 bf16 首 gate；若失败用 `get_kernel_source()` 查 `asc_cvt` 调用形态（tilelang-ascend-debug）；必要时按 Compiler Capability Gap Rule 记 `.agents/evolution/capability-gaps.md` |
| 2 | **N 非 32B 行对齐的 DMA 带宽档**（N=300 fp16 = 600B/行，dst 640B/行） | 正确性 ✓（字节级行长 + 独立 pitch，skill §6.1）；带宽档降级（`CONST-mte-bandwidth-model` 平台/未知档 50/48 B/cycle vs 峰值 102） | 流量项本就非主导（计算项 ~10×，D-2）；Stage 4 msprof 归因确认；非对齐 workload 不在 bench 主对象（wl1/wl4 均对齐） |
| 3 | **Persistent + TIR if + SimdVF 嵌套组合无单一先例**（各层分别有先例：SIMT 兄弟变体 Persistent+TIR if+serial+SimtVF 86+61 用例全 PASS；per_token Persistent+SimdVF+range） | AutoSchedule 对 SimdVF 域内 if 分支两侧 multibuffering 版本推断异常的可能性（未证实——同构 SimtVF 结构已实证降低该风险） | §0.6 R4 fallback 预案：拆双 persistent 域（全批域 [full] + 尾批域 [1] + core_id 谓词），语义等价、API 全先例；Stage 3 首个尾批用例（L0 ④ (33,256)）即触发该路径验证 |
| 4 | **±inf 边界与 torch golden 分歧**（全 −inf 行 → NaN vs torch −inf；含 +inf 行 → NaN vs torch +inf） | 源算法固有语义（迁移保真项 + 双变体行为一致性硬约束）；若集成测试加入此类角点将对照 fail | §8.2 Boundary ④ 以源语义模拟对照（`_source_lse_sim`）；§9.3 文档化；源仓测试域（randn）不触发 |
| 5 | **special_func 8 ops/cycle 的 op 粒度未区分**（element vs 向量指令级，`estimate_latency.h` L58–70 未注明——与 SIMT 兄弟变体风险同源） | D-2 估算的瓶颈归因（计算主导 ~10×）可能反转为 MTE 主导 | 两种解释下主选结构均为最优（special 已压至 1/elem 公式下限——vexpdif 融合 + 流量最小化 + UB 重读 13× 余量）；Stage 4 npusim/msprof 实测归因，回写 `CONST-valu-model` |
| 6 | **x_ub pad 列 [N, N_pad) 未初始化即被尾组 `vld` 读**（读后 `vsel` −inf 中性化——deepgemm transform_sf L205/L215/L249 vsel 同构先例：vld 后掩码是 repo 内已验证形态） | 硬件层：UB SRAM 读未初始化数据无故障（on-chip，值被丢弃）；语义层：buffer 初始化检查可能告警 | 尾组 lane 在任何算术前被 vsel 替换（掩码先于 vmax/vexpdif）；L0 ⑤ (32,300) 首 gate；若语义检查拒绝：fallback = `T.copy` 带列向 `pad_value=−inf` 预填 pad 列（skill §6.1 列向行尾 padding 正是此语义），vsel 保留作双保险（免费） |
| 7 | **block_m 契约外值**（显式 config 传 >16 或非 2 幂） | SUB 封顶机制兜底任意 block_m（§5.2），但极端大 block_m 使 num_batches 增大（批串行长度↑——per-tile 开销项放大，P-LAYOUT-ROW-RESIDENCY） | 性能项非性能正确性项；wrapper default_config 不产生此类值；host assert 记录契约（§4.7） |
| 8 | **wrapper 预算口径**（SIMT 家族 device.py `UB_SIMTVF_BUDGET_BYTES` = 216KB SimtVF 口径；SIMD 家族 Stage 5 移植时 single/tiled 分界预算未定） | 若照抄 216KB 口径 → 分界偏保守（安全方向，损失部分可驻留 shape）；若按 248KB 口径 → 分界放宽 | kernel 侧 assert（N ≤ 32512）+ SUB 公式自适应（不依赖 wrapper 预算数值）；契约边界以 wrapper 实际判定为准；Stage 5 集成时按 SimdVF 248KB 口径重判并记录 |

### 9.2 常见错误（实现期检查清单）

- ❌ `T.Kernel(grid, threads=…)`（二维/threads 参数）——TypeError（ascend-constraints §1 条 1）；SimdVF 形态连 SIMT 版的内部 `threads = min(1024, SUB·TILE)` 推导也不存在（K3）。
- ❌ `tilelang.compile(prog, target="ascend")` 直接形态——S3-JIT 门禁拒（CASE 教训）；用 `@tilelang.jit(target="ascend", out_idx=[1])` 懒工厂。
- ❌ SimdVF 块内使用 `T.Parallel` / `T.reduce_max` / `T.reduce_sum`（高层线程域/归约原语混入显式指令流）——域内循环用 `range`（per_token/flash 域内先例）；归约用 `vcmax`/`vcadd` 显式内在。
- ❌ `vexpdif` 传非 fp32 操作数（`simd.py` L1182–1195 强校验 TypeError）——16-bit 装载必须先 `vcvt` 提升；`m_brc` 由 `vdupv` 保持 fp32。
- ❌ 尾组掩码在 16-bit 域做——掩码统一 fp32 域（`vsel` + fp32 `vdup(−inf)`；16-bit 域 −inf 掩码路径未验证且 bf16 表示需另证）。
- ❌ `vsts` 的 dist 与 pset 宽度错配——ONEPT_B16 配 `pset(16,"PAT_VL1")`、ONEPT_B32 配 `pset(32,"PAT_VL1")`（flash core.py L96/L117 同构）。
- ❌ 尾批 `T.copy` 越界切片（`[base : base+SUB]` 超 M）——A5 DMA 越界行为未文档化；必须静态 rem 切片（§0.6 R4）。
- ❌ 依赖 `T.copy` 的 `pad_value=`/`data_select=` 做列尾数值路径——二者是列向行尾 padding（skill §6.1），本设计列尾用 fp32 域 vsel（不依赖 DMA 填充语义承载算术）；行向 OOB 二者不适用（静态切片规避）。
- ❌ 照抄 flash 形态的 m_ub UB 往返 + `mem_bar`（VST_VLD）——本设计统计量寄存器驻留（`vdupv` 广播直消费），无 UB 往返即无 barrier 需求（§4.4）。
- ❌ 全局 PYTHONPATH 污染——所有 python 命令前置 `PYTHONPATH=/home/z00875089/tilelang-fork`。
- ❌ grid=32 惯例值——本机实查 56（`CONST-arch-a5` measured；28=4×7 整除性陷阱，persistent 形态免疫）。
- ❌ 逐元素标量循环（载入/exp/lse）——§1.6.2 已全显式向量化，§6 不得回退。

### 9.3 特殊场景处理（边界语义文档）

**与 torch golden 的两处分歧（源算法固有，迁移保真保持——与 SIMT 兄弟变体逐项一致）**：

| 场景 | 本 kernel（=源 GPU 算法） | `torch.logsumexp` | 数学注记 | 测试域 |
|------|---------------------------|-------------------|---------|--------|
| 全 −inf 行 | **NaN**（m=−inf → `vexpdif(−inf,−inf)`=exp(NaN)=NaN → S=NaN） | −inf | 解析极限为 −inf；torch 特判，源算法按 IEEE 直传 NaN | randn 域不触发；Boundary ④ 以源语义模拟对照（V4 位级一致） |
| 行含 +inf | **NaN**（inf−inf=NaN 污染求和） | +inf | 解析主值为 +inf；同上 | 同上 |
| 行含 NaN | NaN（逐元素传播） | NaN | 一致 ✓ | equal_nan 对照 |
| 部分 −inf 正常行 | m 有限 → exp(−inf−m)=0 中性 | 一致 | 一致 ✓ | 直接对照 |
| fp16 深度下溢（−745） | S 下溢至最小可表示 → log(S) 有限/−inf 边缘 | fp32 域不受影响 | 输出 cast fp16 后差异在容差内（V3 角点覆盖） | Boundary ① |
| fp16 输出饱和（全 65504 行） | sat 钳位（O6：上界 65514.4 < 65520 IEEE inf 阈值——契约空间内数学封闭，V5 0 分歧） | 同为 fp16 舍入域 | 一致 ✓ | Boundary ①（V5 已验） |

**设计取舍说明**：选择保持源语义而非「对齐 torch」（哨兵钳位 m ≥ −65504 使全 −inf 行输出 −inf）——理由：① 迁移保真优先；② **双模式对照硬约束**：SIMT 兄弟变体保持源语义（NaN），本 SIMD 变体若钳位将制造同族两变体语义分裂；③ randn 测试域不触发。若 Stage 5 集成后需要与 torch 全角点对齐，属算子族级语义变更，应回归源仓统一处理（本设计记录不在单变体内擅自变更）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/logsumexp_simd/_logsumexp_kernel_single/
├── DESIGN.md                    # 本设计文档
├── verify_equiv.py              # D-1 等价性机器验证（已执行，52 门控行全 EQUIV_PASS）
├── _logsumexp_kernel_single.py  # Stage 3：kernel 实现（@tilelang.jit 懒工厂 + golden + 分层测试）
├── RETROSPECTIVE.md             # Stage 1 复盘（本文件同批产出）
├── REVIEW.md                    # Stage 2：设计检视报告（另生成）
└── history_version/             # 历史版本备份（回退时启用）
```

### 10.2 文件清单（Stage 3 实现输入）

| 文件 | 内容 | 契约要点 |
|------|------|---------|
| `_logsumexp_kernel_single.py` | kernel + golden + `_source_lse_sim` + L0/L1/L2/Boundary 分层测试 | 调用面 `_logsumexp_kernel_single(M, N, dtype)(block_m)(x)`；模块 docstring 声明 `programming_mode: simd` + `auto_schedule: yes`；`@tilelang.jit(target="ascend", out_idx=[1])` 懒工厂；golden = `torch.logsumexp(x.float(), dim=-1).to(x.dtype)`（NPU 上计算）；host 静态化 `N_pad/G_full/k_tail/SUB/full/rem/num_batches/NUM_BLOCKS` + 契约 assert |

### 10.3 命名与声明规范

- 模块 docstring 必含：`programming_mode: simd`、`auto_schedule: yes`（迁移 prompt 契约——**simd 不是 simt**，两变体靠此声明区分）。
- 函数名/参数名与源保持（`_logsumexp_kernel_single`、`M/N/dtype/block_m`）；`threads` 参数任何形态不存在（K3——SimdVF 无线程概念）。
- host 静态化变量命名：`N_pad/G_full/k_tail/SUB/full/rem/num_batches/NUM_BLOCKS`（§3.3 注）；UB 预算常量 `UB_BUDGET_BYTES = int(248*1024*0.9)`（228,556——与 SIMT 兄弟变体 `int(216*1024*0.9)` 同口径异值，§4.5）。

### 10.4 实现顺序建议（Stage 3）

1. 懒工厂骨架 + host 静态化（含契约 assert）→ 2. 全批路径 fp16（L0 ①② 的 fp16 部分 + ③ fp16）→ 3. dtype 路径（③ bf16——风险 1 首验；③ fp32 NORM 直连 + ONEPT_B32/pset(32)）→ 4. 尾组路径（L0 ⑤ (32,300)：k_tail=44 vsel + 风险 6 首验）→ 5. 尾批路径（L0 ④ (33,256)：TIR if + 静态切片 + 风险 3 首验）→ 6. `get_kernel_source()` 首查（spill 计数 / `asc_cvt` 宽化形态 / `vsts_1st` 调用）→ 7. L1 扫描 → 8. L2/Boundary（源语义对照项）→ 9. 四出口判定。

### 10.5 本设计引用的外部工件

| 工件 | 用途 |
|------|------|
| `/home/z00875089/TileOPs-fork/tileops/kernels/reduction/logsumexp.py` | 源算子（三问解读对象：`_logsumexp_kernel_single` L40–99；`LogSumExpKernel` L246–557 wrapper 契约） |
| `/home/z00875089/TileOPs-fork/tileops/kernels/reduction/_primitives.py` | `DEFAULT_ALIGNMENT=256` L30 / `MAX_SINGLE_TILE_COLS=32512` L39 / `align_up` / `compute_tile_n` / `device_smem_budget`（single 选中条件） |
| `examples/TileOPs/tileops/manifest/reduction.yaml` L124–172 | `LogSumExpFwdOpSimd` workloads（single 落点 2/4）+ roofline 口径（4·M·N FLOPs、(M·N+M)·elem bytes）+ source 路径（Stage 5 对接面） |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp.py` | SIMT 家族 wrapper dispatch / `default_config` 启发式（block_m 契约口径）+ `examples/TileOPs/tileops/device.py` `UB_SIMTVF_BUDGET_BYTES`（预算口径参照，风险 8） |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp_kernel/_logsumexp_kernel_single.py` + `_DESIGN.md` | SIMT 兄弟变体（只读参照，严禁改动）：调用面 / 契约 assert / golden + `_source_lse_sim` / 分层测试组织 / Persistent+TIR if 嵌套实证 / Stage 4 性能基线（§8.3 A/B） |
| `examples/TileOPs/tests/ops/test_softmax.py`（源仓 `tests/ops/` 同款） | 精度用例基准（`ref_program` = golden 直移植源）；SIMD 家族 `test_logsumexp_simd.py` Stage 5 落地 |
| `examples/ascend/example_per_token_cast_to_fp8.py` | SimdVF 逐行规约完整骨架（persistent + annotate_buffer_versions + copy + SimdVF 域内 range 循环 + vsts + (M,) 输出，L117–155） |
| `examples/ascend/flash_attention/core.py` | SimdVF 行 max/sum 归约指令族（vcmax/vcadd/vexpdif/ONEPT/BRC/pset，L94–137 softmax 段同构） |
| `examples/ascend/deepgemm/kernels/transform_sf.py` | vld 后 vsel 掩码先例（尾组 −inf 掩码同构，L205/L215/L249） |
| `examples/ascend/example_gemm.py` / `example_rmsnorm.py` | 核内循环变量 TIR if（L47）/ UB→GM 1D 切片（L82）先例 |
| `testing/ascend/language/test_tilelang_ascend_simdvf_cast.py` / `test_tilelang_ascend_simdvf_integer_predicate.py` | UNPK_B16+vcvt 位级回归（L64–80）/ `update_mask(cnt,32)` 先例（L261） |
| `examples/pilot/online_softmax/` | JIT 懒工厂形态 / 静态 rem 切片 / 分层测试组织参照 |
| pattern-library（`layout.md` P-LAYOUT-ROW-RESIDENCY / P-LAYOUT-TAIL-STATIC-EXTENT、`traps-runtime.md` TRAP-GRID-DIVISIBILITY / TRAP-NAIVE-ONLINE-MERGE、`constants.md` CONST-capacity-a5 / CONST-vrf-capacity-a5 / CONST-valu-model / CONST-mte-bandwidth-model / CONST-arch-a5） | 实测模式与硬件常数（版本戳 2026-10-16/17；A5 冷启动 assumption 条目按 90% 折扣使用） |
| `.agents/skills/_shared/standards/`（core-split-strategy.md §1/§2.1、algorithm-research.md、hardware-cost-model.md、negative-claim-evidence.md、gate-and-retry.md §1） | 分核三要素权威标准 / 调研四问执行标准 / D-2 估算协议 / 负向断言证据纪律 / Stage 1 门禁 |
