# GPU TileLang → Ascend A5 适配点目录

> **定位**：本目录是 Stage 0 移植（`add-ascend-op` skill，7 文件结构脚手架）与
> Stage 1 设计（`tilelang-op-conductor`，A5 kernel 端到端生成）共用的适配点速查。
> 深入语义以 `.agents/skills/tilelang-ascend/SKILL.md`（仓库根）为权威；
> API 签名以 `tilelang/ascend/language/` 源码为实现真相。

**核心前提**：GPU TileOPs 的 kernel 是 CUDA 方言 TileLang（`@tilelang.jit` 默认
CUDA target、`T.Kernel(grid, threads=)`、`T.alloc_shared` = CUDA shared memory）。
A5 侧**不逐行改写** GPU kernel——Stage 0 只抽取 GPU kernel 函数作算法参考，
A5 kernel 由 conductor Stage 1-3 以 Ascend 方言重实现。本文档列出两者之间的
全部机制差异，供 Stage 0 结构移植（wrapper/Op/test/bench）与 Stage 1 kernel
设计时对照。

---

## 一、通用机制适配点

公共机制是指框架级基础设施——所有算子共享的设备相关代码。适配集中在
`tileops/device.py`（单一适配面）与 TileLang 方言层。

### 1.1 规范入口与方言

| # | GPU 原始代码 | A5 适配 | 说明 |
|---|---|---|---|
| A1 | `import tilelang.language as T`（CUDA 方言） | `import tilelang.ascend.language as T` | Ascend 方言 shadow 常用 API 面并暴露 Ascend 专有 API（`T.MixedKernel`、`T.SimdVF` 等） |
| A2 | `@tilelang.jit`（默认 CUDA target） | `@tilelang.jit(target="ascend")` | 唯一规范入口；`--npu-arch=dav-3510`（Ascend950PR_9579，CANN 9.2.0） |

### 1.2 Kernel 启动模型（一维 grid，无 threads）

| # | GPU 原始代码 | A5 适配 | 说明 |
|---|---|---|---|
| A3 | `T.Kernel(grid, threads=threads)` | 一维 `T.Kernel(N)` | Ascend 只支持一维 grid；`threads=` 参数在 ascend target 上抛 TypeError |
| A4 | 2-D / 3-D grid | 展平为一维 grid，block 内自行分解索引 | 多维循环结构在 kernel body 内表达 |
| A5 | `T.Parallel(threads, npt)`（线程×元素二维分解） | `T.SimtVF(threads=N)` 线程域 + `T.Parallel` 一维循环 | 编译器对 `T.Parallel` 循环自动向量化；GPU 的 `threads * npt` 折叠为单一 `block_size`（elementwise 家族，见 SKILL.md K11） |

### 1.3 编程模式

| # | GPU 原始代码 | A5 适配 | 说明 |
|---|---|---|---|
| A6 | （无此概念；CUDA 只有一种执行模型） | 三种编程模式 `simt \| simd \| mixed` + 正交 `auto_schedule: yes \| no` | 默认 **SimtVF**（simt，auto_schedule yes）；**SimdVF** 为 2048-bit `T.simd.*` 显式内在函数（可加 `latency=`）；GEMM 主导算子用 **MixedKernel**（`T.MixedKernel(sids=1\|2)`）。kernel 模块 docstring 须声明 `programming_mode: simt\|simd\|mixed`（S5 集成门禁 S5-PROG-MODE 检查此声明；旧编程模式开关机制已整体删除） |

### 1.4 Cube 路径（T.copy / T.gemm / T.dual_copy）

| # | GPU 原始代码 | A5 适配 | 说明 |
|---|---|---|---|
| A7 | `T.gemm(A, B, C)`（register/shared 输入） | Cube 路径固定套路：`T.copy` GM↔L1（ND2NZ 分形转换由 InsertNd2Nz pass **自动插入**，禁止手写 NZ staging）→ `T.gemm(A_l1, B_l1, C_l0c, trans_A=False, trans_B=True)` → `T.dual_copy`/`T.copy` L0C→UB | L1 输入路径要求 NT 约定（`trans_A=False, trans_B=True`）；GM 权重布局 `[N, K]` |
| A8 | （CUDA 无累加器缓冲概念） | 累加器 `T.alloc_l0c` + `clear_accum=(kt == 0)` | K-tile 循环首个 tile 清零累加器 |

### 1.5 内存层级与容量

| # | GPU 原始代码 | A5 适配 | 说明 |
|---|---|---|---|
| A9 | `T.alloc_shared` = CUDA shared memory（~48-228KB/SM） | `T.alloc_shared` = **UB**（Unified Buffer） | A5 物理 248KB；**SimtVF 模式可用 216KB** —— per-block workspace 预算按此检查（`device_smem_budget` 的 A5 语义） |
| A10 | L1 cache（隐式） | `T.alloc_l1` = **CBuf**，512KB | Cube 侧数据 staging |
| A11 | register fragments（`T.alloc_fragment`） | `T.alloc_fragment` 保留；另有 Cube 缓冲 `T.alloc_l0a/l0b/l0c` | 分块策略以最小层级容量为准 |

### 1.6 同步

| # | GPU 原始代码 | A5 适配 | 说明 |
|---|---|---|---|
| A12 | `__syncthreads()` / mbarrier / `cp.async` 手动管理 | **默认靠 AutoSchedule** 自动插入同步（SimtVF 区域 `asc_syncthreads()` 自动插入） | 仅全手动 kernel（`PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False`）使用 `T.ascend_set_flag`/`T.ascend_wait_flag`/`T.ascend_pipe_barrier`，且必须成对出现并注释说明 |

### 1.7 Grid 整除性

| # | GPU 原始代码 | A5 适配 | 说明 |
|---|---|---|---|
| A13 | GPU SM 数量通常幂次友好 | A5 `cube_core_num=28` / `vector_core_num=56`（**非 2 幂**） | block 数须整除核数或显式处理余块；不整除会留空核或丢 tile —— Stage 1 设计与 Tier 2 排错的常见坑 |

### 1.8 精度容差

| # | GPU 原始代码 | A5 适配 | 说明 |
|---|---|---|---|
| A14 | 测试容差随算子自定义 | 按 dtype 分档：fp8→1e-1、bf16/fp16→1e-2、fp32→1e-3（累积密集型 5e-3） | golden 用 torch 在 NPU（`device="npu"`）上计算，`torch.testing.assert_close` 断言；fp16/bf16 精度不达标时在 kernel 中加 fp32 中间量（upcast 计算、downcast 写回） |

### 1.9 性能测量口径

| # | GPU 原始代码 | A5 适配 | 说明 |
|---|---|---|---|
| A15 | `torch.cuda.Event` / CUPTI 计时 | `tilelang.profiler.do_bench(fn, backend="msprof")` 是**唯一**端到端时延口径（ms） | 周期级归因唯一在 npusim；`msprof_detail` 用于归因。禁止臆造时钟频率（代价模型是周期域的）。Tier 2 验证命令 `PYTHONPATH=src:.:../.. python -m tileops.reporting.cli run --op {op_name} --prof-mode msprof`（内部为 do_bench 封装） |

### 1.10 tileops 框架层（设备抽象）

| # | GPU 原始代码 | A5 适配 | 说明 |
|---|---|---|---|
| A16 | `torch.cuda.is_available()` / `device_count()` / `current_device()` / `get_device_properties()` / `Event` / `synchronize()` / `empty_cache()` / `manual_seed_all()` 等直接调用 | 全部经 `tileops.backend.device.get_device_backend()` 单一适配面 | 框架代码从不直接调用 `torch.cuda.*` 或 `torch.npu.*` |
| A17 | `supported_archs = [80, 86, 89, 90]`（CUDA SM 整数） | `supported_archs = ["Ascend950"]` | dav-3510 架构标识 |
| A18 | `@torch.library.custom_op("top::...")` | `@torch.library.custom_op("ascend::...")` | 避免与 GPU 原工程命名空间冲突 |
| A19 | autotune（`tilelang.autotuner`）/ `threads` / `npt` 执行参数 | autotune **全部移除**（启发式 config 选择）；`threads` 移除（reduction）或与 `npt` 合并为 `block_size`（elementwise） | 详见 `add-ascend-op` SKILL.md K8/K9/K11；GPU 的 `device_smem_budget` 语义改为 UB 预算（K6/A9） |

---

## 二、按算子族的适配要点索引

本节不逐条罗列 910B 历史算子经验，而是按**算子族**给出 GPU → A5 的适配要点；
每族指向 `.agents/skills/tilelang-ascend/SKILL.md` 相应章节与本仓
`examples/ascend/` 参考例子。Stage 1 设计时按族检索，再回到权威 skill 核对
API 语义与约束。

### 2.1 elementwise（vecadd / lerp / cast 类）

1. 默认 **SimtVF**（`T.SimtVF(threads=N)` + `T.Parallel` 循环，auto_schedule
   yes）——编译器自动向量化，无需手写 SIMD。
2. GPU `threads * npt` 折叠为单一 `block_size`（SKILL.md K11）；per-block
   元素量受 SimtVF 可用 UB（216KB）约束。
3. 仅实测热点才降级 **SimdVF**（`T.SimdVF()` + 2048-bit `T.simd.*` 显式
   内在函数，可加 `latency=` 标注）。
4. grid = `ceildiv(N_total, block_size)`，注意 56 vector 核的整除性/余块
   处理（A13）。
5. 低精度 cast 参考 `examples/ascend/example_per_token_cast_to_fp8.py`；
   基础模式参考 `examples/ascend/example_vecadd.py`。

参考：tilelang-ascend skill §2（线程层级）/§6（SIMD MicroAPI）。

### 2.2 reduction / norm（softmax / rmsnorm / logsumexp 类）

1. 优先**在线算法**（online softmax / Welford）替代两遍扫描，中间量驻留
   UB，减少 GM 往返。
2. 跨线程归约用 `T.alloc_reducer` / `T.warp_reduce_sum` 等 reducer 原语
   （对应 GPU warp-level ops）。
3. 归约轴 tile 大小同时受 SimtVF UB（216KB）与向量化列上限约束——GPU 侧
   `MAX_SINGLE_TILE_COLS` 类常量须按 A5 重新标定，不可照抄。
4. 多维归约的 flatten/restore 逻辑在 Op 层完成（设备无关，Stage 0 直接
   保留 GPU 实现）。
5. 基础模式参考 `examples/ascend/example_rmsnorm.py`；回归用例见
   `testing/ascend/`。

参考：tilelang-ascend skill §4（内存层级）/§7（同步与 Stage 调度）。

### 2.3 gemm / epilogue（matmul / linear / deepgemm 类）

1. GEMM 主导算子用 **mixed kernel**：`T.MixedKernel(sids=1|2)`，Cube 侧
   `T.gemm` + Vector 侧 epilogue（A6）。
2. Cube 路径固定套路：`T.copy` GM↔L1（ND2NZ 自动插入，禁止手写 NZ
   staging）→ `T.gemm(NT: trans_A=False, trans_B=True; GM 权重 [N,K])` →
   `T.dual_copy` L0C→UB（A7）。
3. 累加器 `T.alloc_l0c` + `clear_accum=(kt == 0)`（A8）；split-K 参考
   `examples/ascend/example_gemm_splitk.py`。
4. 容量分层：CBuf（L1）512KB、L0A/L0B/L0C 更小——分块策略以最小层级
   为准（A10/A11）。
5. 基础模式参考 `examples/ascend/example_gemm.py`、
   `examples/ascend/deepgemm/`；GEMM 约束权威在
   `tilelang/ascend/op/gemm/gemm_mad.py`。

参考：tilelang-ascend skill §3（mixed kernel）/§5（GEMM）。

### 2.4 attention（flash attention 类）

1. QKV 布局与 fractal 转换交给 `T.copy` 自动 ND2NZ，不手写 NZ staging。
2. **MixedKernel** 结构：Cube 侧批量 GEMM（QK^T、PV），Vector 侧
   softmax / 掩码 / 缩放。
3. online softmax（flash 式）中间状态驻留 UB；KV 分块大小受 SimtVF
   216KB 约束。
4. 累加器生命周期跨 KV tile，用 `T.alloc_l0c` + `clear_accum` 控制首 tile
   清零。
5. 端到端参考 `examples/ascend/flash_attention/`。

参考：tilelang-ascend skill §3（mixed kernel）/§5（GEMM）/§7（同步）。

---

## 三、不适配项（设备无关，Stage 0 直接保留）

以下部分从 GPU 版本**完整保留**（除第一节列出的点外）：

- Manifest YAML 格式与加载逻辑
- Workload 参数化（`FixtureBase` / `FixtureMeta`）
- Op 的 `dim` 规范化、多维 flatten/restore、`eval_roofline()` 公式
  （roofline 模型设备无关）
- Kernel 类主体流程（config 选择、tiling 启发式、`forward` 分发）——
  除 K5-K11 列出的适配点
- 测试结构（`TestBase` / 断言逻辑；容差按 1.8 分档）
- Benchmark 参数化与 `BenchmarkReport` 报告生成
- GPU TileLang kernel 函数抽取件——作为 Stage 1 算法参考**原样保留**，
  A5 重实现由 conductor 生成并经 `integrate_kernel.py` 集成
