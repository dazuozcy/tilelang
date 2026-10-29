# Backend Glossary（后端词汇映射单一事实源）

> **用途**：tilelang-op-conductor 自 tilelang-mlir-fork（npuir/A2A3）适配到 tilelang-fork（ascend/A5，dav-3510）的概念映射表。所有 agent / skill / 工具文案引用本表而非各自内联映射，降低未来第三后端（如 pto）适配成本。本文件是**禁用词检查的唯一豁免文件**（历史对照必须出现旧词）。
>
> 维护规则：映射变更须同步改本表 + 受影响文案；禁用词清单见文末。

## 1. 概念映射表

| mlir-fork 概念（npuir/A2A3） | tilelang-fork 对应（ascend/A5） | 备注 |
|---|---|---|
| `target="npuir"` | `target="ascend"` | 规范入口统一为 `@tilelang.jit(target="ascend")`；`pto` AOT 线不在 conductor 范围 |
| `import tilelang.language as T` + v-prefix | `import tilelang.ascend.language as T`；显式向量 = `T.simd.*` | 方言 shadowing common surface |
| v-prefix 优先原则 | **SimtVF 优先**（可读性/自动调度）；性能热点降级 SimdVF | 权威：`.agents/skills/tilelang-ascend/SKILL.md` §2/§8 |
| `T.load_nd2nz` | `T.copy`（ND2NZ 由 InsertNd2Nz pass 自动插入） | 禁止手写 NZ 分形搬运 |
| `T.store_fixpipe` | `T.dual_copy`（L0C→UB 双 AIV）/ `T.copy`（L0C→GM） | layout 由 AscendLayoutInference 推理 |
| `T.Scope("Cube"/"Vector")` | `T.Cube()` / `T.Vector()` + `T.MixedKernel(sids=1\|2)` | AIC/AIV 指派默认 AutoSchedule；完全手动核需 `TL_ENABLE_AUTO_SCHEDULE: False` |
| `sync_block_set/wait` 手写同步 | 默认 AutoSchedule 自动插 flag 对；手动档 `T.ascend_set_flag/wait_flag`、`T.ascend_pipe_barrier` | 手动 flag 必须配对并注释 |
| `NPUUtils.get().get_aicore_num()` 实查分核 | `T.Kernel(N)` 一维 grid + AutoSchedule；核数 = `torch.npu.get_device_properties().cube_core_num`/`vector_core_num` 实查（惯例初值 32 待标定） | 分核三要素语义 → grid 大小 / AIC-AIV 分工 / 同步策略（auto\|manual） |
| `TILELANG_ASCEND_MODE=developer\|expert` + `Kernel.ascend_mode` | **已删除**。编程模式枚举：`simt` / `simd` / `mixed` + 正交字段 `auto_schedule: yes\|no` | 见 signal-registry.md §3 |
| GM/L1/UB/L0(A/B/C) 层级 | 同名，alloc API：`T.alloc_shared`(UB) / `T.alloc_l1`(CBuf) / `T.alloc_l0a/l0b/l0c` | 容量常数：pattern-library/constants.md CONST-capacity-a5 |
| `msprof op` CLI（Task Duration 唯一口径） | `do_bench(fn, backend="msprof")`（端到端时延唯一口径）+ `msprof_detail`（pipe ratios）+ npusim（周期级归因唯一口径） | 派生 TFLOPS=`2*M*N*K/t`、有效 GB/s |
| TileLangIR/BishengIR/tladapter 调试 | `kernel.get_kernel_source()`（AscendC 源码）、IR Lower Trace（`docs/tools/lower_trace.md`）、Pass Diff、compile_only | 调试 skill：tilelang-ascend-debug |
| `source set_env.sh` 环境前提 | CANN set_env + `BISHENG_HOME` + `torch.npu.is_available()`；无 npuir set_env.sh，ascendcl stub 惰性 dlopen | |
| toolchain_stamp "tilelang X + CANN 8.5 + 910B2C" | "tilelang X + CANN 9.x + dav-3510(A5)" | kb_stale_check 组装口径 |
| Capability gap layer 枚举（Frontend API/TileLangIR pass/BishengIR/tladapter/runtime） | Frontend API（ascend 方言）/ Ascend pass / codegen / bisheng 编译器 / ACL runtime | capability-gaps.md schema |
| 910B2C 容量（UB 192KB / L1 512KB / L0C 128KB / 24 AIC / 48 AIV） | A5 假设值：UB 248KB（SIMT 216KB）/ L1 512KB / L0C 256KB / L0A、L0B 64KB / AIC:AIV=1:2 / 核数运行时实查 | 全部 `status: assumption`，见 constants.md |
| CUDA/GPU 概念迁移锚（TMA/cp.async/shared/warp shuffle/cluster） | TMA→`T.copy`；cp.async→`T.Pipelined` num_stages；shared→UB；warp shuffle→reduce intrinsic（`T.warp_reduce_sum/max/min`，仅 fp16/fp32/int32/uint32 直连硬件）；cluster→无对应 | 迁移分析用 |
| 错误分类（前端/TileLangIR pass/BishengIR/tladapter/运行时） | ① 方言前端（tilelang-semantic 校验）② Ascend pass 断言（布局推理/VFChecker/AutoSchedule）③ bisheng 编译报错（`--npu-arch=dav-3510`）④ ACL 运行时/精度 | tilelang-ascend-error-fixer |

## 2. 编程模式枚举（signal-registry.md §3 权威，此处速查）

| 枚举 | 语义 | 对应机制 |
|---|---|---|
| `simt` | SIMT 线程域编程，`T.Parallel` 循环 + 编译器自动向量化/自动调度（默认推荐） | `T.SimtVF(threads=N)` |
| `simd` | 显式向量编程，2048-bit intrinsic + 谓词掩码 + `latency=` 注解（性能热点） | `T.SimdVF()` + `T.simd.*` |
| `mixed` | Cube+Vector 混合核（GEMM 主导算子） | `T.MixedKernel(sids=1\|2)` + `T.gemm` + SimdVF 后处理 |

正交字段 `auto_schedule: yes|no`（默认 yes）；`no` 时设计文档必须写明手动同步方案（set_flag/wait_flag 配对），reviewer 增设对应检视点。

## 3. 禁用词清单（验收门）

以下旧词在 `.opencode/` + `.agents/` 规则文件中**零命中**（本文件与显式「清零说明/历史说明」除外）：

```
npuir
v-prefix / vadd / vmul / vexp / vcast / vbrc / vrsqrt / vselect / vcmp（作为 API 引用时）
load_nd2nz / store_fixpipe
TileLangIR / BishengIR / tladapter
910B2C
NPUUtils
set_env.sh（npuir 专属语境）
ASCEND_MODE / TILELANG_ASCEND_MODE / ascend_mode
developer 模式 / expert 模式 / Developer|Expert|hybrid（编程模式语境）
```

机械检查：`python3 .agents/tools/gate_lint.py --forbidden-words`（P1 提供）。

## 4. 路由与文档锚点速查

- 编程权威：`.agents/skills/tilelang-ascend/SKILL.md` §1–§12（§2 线程层级 / §4 内存 / §5 GEMM / §6 搬运 / §7 同步 / §8 SIMD / §10 陷阱 / §11 编译执行 / §12 性能测量）
- 示例入口：`examples/ascend/`（vecadd / rmsnorm / gemm / gemm_splitk / per_token_cast_to_fp8 / deepgemm / flash_attention）+ `testing/ascend/`
- 工具链：CANN 9.x、bisheng `--npu-arch=dav-3510`、`BISHENG_HOME`；运行时 `aclrtBinaryLoadFromData` + `aclrtLaunchKernelWithHostArgs`（ascendcl stub 惰性 dlopen）
