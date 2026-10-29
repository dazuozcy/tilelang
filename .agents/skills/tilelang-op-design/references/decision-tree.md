# 算子特征分析决策树（Ascend 版：simt/simd/mixed + T.copy/T.gemm/dual_copy 搬运路径）

## 目录

- [算子特征分析决策树（Ascend 版：simt/simd/mixed + T.copy/T.gemm/dual_copy 搬运路径）](#算子特征分析决策树ascend-版simtsimdmixed--tcopytgemmdual_copy-搬运路径)
  - [目录](#目录)
  - [1. 函数设计原则](#1-函数设计原则)
  - [2. 决策树](#2-决策树)
  - [3. NPU 硬件约束](#3-npu-硬件约束)
  - [4. API 映射规则](#4-api-映射规则)

> 本项目使用 TileLang Ascend 方言（`import tilelang.ascend.language as T`），与 GPU 版 TileLang 在 Kernel 维度、threads、循环边界、GEMM 约定、内存分配等方面有显著差异。完整对比与约束清单见 SKILL 子目录索引下的 `ascend-constraints.md`。外部参考实现仅用于理解数学逻辑，API 映射必须查阅本项目（`examples/ascend/` + `testing/ascend/` + tilelang-ascend skill）。算子族级导航（族 → 模式 → 参考例子 → 搬运套路）另见 tilelang-ascend-opdev skill。

---

## 1. 函数设计原则

1. **维度参数自推导**：算子调用函数（如 `gemm`）应从输入 tensor shape 提取 M/N/K 等维度，不依赖模块级全局变量。这保证多场景顺序测试时不发生变量污染。
2. **Host 预处理显式声明**：若计算的一部分在 Python 侧完成（如 im2col），必须在 §1 算法描述和 §4 数据流中明确标注。
3. **编译入口统一**：规范入口 `@tilelang.jit(target="ascend")` / `tilelang.compile(program, target="ascend")`（别名 `asc`；`pto` AOT 线不在 conductor 范围）。

## 2. 决策树

**重要**：`T.reduce_*` 与 reducer 归约在 simt 和 simd 模式下**都可使用**。模式选择取决于是否需要显式控制 2048-bit 向量指令（特定 intrinsic / 掩码 / 舍入模式 / 求值顺序 / `latency=` 注解），而非使用了哪个 API。**从 simt 起步**，精度通过后以 `do_bench(backend="msprof")` 实测，仅对证实的热点降级 simd 或引入 mixed（选型原则见 tilelang-ascend-opdev skill）。

```
算子数学公式
├─ 含 matmul / @ / 矩阵乘
│   ├─ 仅 matmul → 纯 Cube（AIC）
│   │   模式: mixed（T.Kernel 自动识别 __cube__，无需显式 MixedKernel）
│   │   API: T.copy（GM→L1，ND2NZ 由 InsertNd2Nz pass 自动插入）
│   │        + T.gemm(A_l1, B_l1, C_l0c, transpose_B=True)   # L1-input NT 路径，权重 [N, K]
│   │        + T.copy（L0C→GM）
│   │   内存: GM→L1→L0C→GM（L0A/L0B staging 为可选路径）
│   │   Kernel: T.Kernel(NUM_BLOCKS) as bx   # 一维，无 threads 参数
│   │   参考: examples/ascend/example_gemm.py（persistent + swizzle + HF32）
│   │
│   └─ matmul + element-wise 前处理/后处理 → mixed 混合核
│       ├─ T.Kernel 自动识别（Cube + Vector 工作共存时自动 mixed + dual_copy 降级）
│       ├─ T.MixedKernel(NUM_BLOCKS, sids=2) as (bx, sid)   # 需要 sid 或显式 AIV 数量时
│       │   API: T.gemm（Cube）+ T.dual_copy（L0C→UB，双 AIV 分担）+ SimdVF/SimtVF 后处理
│       │   内存: GM→L1→L0C --T.dual_copy--> UB --Vector 后处理--> GM
│       │   写法: examples/ascend/deepgemm/kernels/epilogue.py（mixed epilogue）
│       └─ 完全手动（例外路径）: T.Cube() + T.Vector() 显式域
│           + PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False
│           同步: 手动 T.ascend_set_flag / T.ascend_wait_flag 配对 + flag id 预算
│    典型算子: Flash Attention（examples/ascend/flash_attention/）
│
├─ 纯 element-wise（逐元素运算）
│   参考: examples/ascend/example_vecadd.py（simt/simd 双模式对照）
│   ├─ 常规 → simt 模式（默认推荐）
│   │   API: T.SimtVF(threads=N) + T.Parallel 循环（编译器自动向量化）
│   │   内存: T.alloc_shared（UB）
│   ├─ 证实的热点 → simd 模式
│   │   API: T.SimdVF() + T.simd.*（2048-bit intrinsic + 谓词掩码，可带 latency=）
│   │   写法: example_vecadd.py 的 simd 分支 / example_per_token_cast_to_fp8.py
│   └─ 多步运算（如 norm 族）→ simt + T.Pipelined 行块流水
│       参考: examples/ascend/example_rmsnorm.py
│
├─ 含归约（reduce_sum / reduce_max / reduce_min / 行统计）
│   参考: examples/ascend/example_rmsnorm.py；回归 testing/ascend/language/test_tilelang_ascend_reduce.py
│   API: T.alloc_reducer + T.reducer_init/update/finalize_reducer（SimtVF 内）
│        或 T.warp_reduce_sum/max/min（硬件直连仅 fp16/fp32/int32/uint32）
│   内存: T.alloc_shared（UB）+ T.alloc_fragment（寄存器级向量化加载）
│
└─ 其他复杂算子
    强制步骤: 先搜索 examples/ascend/ + testing/ascend/，再查 tilelang-ascend skill
```

## 3. NPU 硬件约束

**⚠️ NPU 硬件约束（必查）**：

设计 Tiling 策略时，必须考虑（常数条目见 `tilelang-op-optimize` skill pattern-library/constants.md，A5 冷启动期全部 `status: assumption`，容量按 90% 折扣预算）：

1. **Kernel 维度**：仅一维 `T.Kernel(N)`；无 `threads=` 参数（TypeError）；线程域在 `T.SimtVF(threads=N)` 内声明。
2. **GEMM C0 分形约束**（`CONST-gemm-dim-constraints`）：
   - M、N 必须是 16 的倍数；`K % TILE_K_SUB == 0` 且 `TILE_K_SUB % C0 == 0`；
   - C0 = 32B/元素：bf16/fp16=16、fp32=8、int8=32、fp4=64；FRAC_N = 16（L0C 累加器固定 16×16 fractal）；
   - blockscaled(MX)：K 必须 64 整除；仅 bf16/fp8e4m3fn/fp4。
3. **对齐要求**（`CONST-capacity-a5`）：
   - L0C：1024B；L1 / L0A / L0B：512B（merge UB 对齐）；
   - GM→UB copy 行右 padding 到 32B 边界（`pad_value=` / `data_select=`，二者互斥）；
   - L0C→UB dual_copy N-split：完整 N 须被 32 整除。
4. **存储容量上限**（`CONST-capacity-a5`，assumption 按 90% 折扣）：
   - UB：248KB（含 `T.SimtVF` 时 216KB）；L1（CBuf）：512KB；L0C：256KB；L0A/L0B：64KB。
5. **静态循环边界**：核内串行循环边界必须为静态值。
6. **flag id 预算**（手动同步时）：核内 8 个、跨核 16 个（`CONST-arch-a5`）。

违反约束会导致编译错误或运行时错误。

## 4. API 映射规则

| 类别 | Ascend 方言 API（推荐） | GPU 形态（本项目不支持/须改写） |
|------|------------------------|-------------------------------|
| GEMM | `T.gemm(A_l1, B_l1, C_l0c, transpose_B=True, clear_accum=...)`（L1-input NT 路径；权重 [N, K]） | `T.gemm` 通用转置组合（L1 路径仅 NT）❌ |
| 内存分配 | `T.alloc_shared`（UB）/ `T.alloc_l1`（CBuf）/ `T.alloc_l0a/l0b/l0c` / `T.alloc_fragment` | GPU `T.alloc_shared`（SMEM 语义）须按 UB 容量重算 |
| 数据搬运 | `T.copy`（GM↔L1/UB/L0；ND2NZ 由 pass 自动插入）/ `T.dual_copy`（L0C→UB 双 AIV、GM↔UB、UB→L1） | TMA / cp.async ❌（无对应 API，走 T.copy + T.Pipelined） |
| Kernel | `T.Kernel(N)`（一维）/ `T.MixedKernel(N, sids=1\|2)` | `T.Kernel(m, n, b)` 三维 / `threads=` ❌ |
| VF 编程 | `T.SimtVF(threads=N)` + `T.Parallel`；`T.SimdVF()` + `T.simd.*`（可带 `latency=`） | warp shuffle / `__shfl_sync` ❌（用 `T.warp_reduce_*` / reducer） |
| 归约 | `T.alloc_reducer` + `reducer_init/update/finalize_reducer`；`T.warp_reduce_sum/max/min` | 跨线程 shuffle 树形归约 ❌ |
| 同步 | 默认 AutoSchedule（自动插 flag 对）；手动 `T.ascend_set_flag` / `T.ascend_wait_flag` / `T.ascend_pipe_barrier("PIPE_*")`；完全手动核 `PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False` | `__syncthreads()` / mbarrier ❌ |
| 调度提示 | `T.Pipelined(num_stages=)` / `T.Persistent` / `T.Stage` / `T.Task(latency=, ii=)` / `T.PerCoreTask` | — |
