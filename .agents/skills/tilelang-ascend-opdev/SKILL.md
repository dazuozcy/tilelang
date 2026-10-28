---
name: tilelang-ascend-opdev
description: >
  Ascend 算子族开发导航技能。用户要求开发新算子、选择编程模式（simt/simd/mixed）、
  定位参考例子、了解 elementwise/gemm/attention/reduction-norm 某一算子族的
  布局与数据搬运套路时使用本技能。本技能是导航层：API 语义与约束的权威在
  tilelang-ascend skill，此处只做「族 → 模式 → 例子 → 套路」的路由。
---

# TileLang Ascend 算子族开发导航（opdev）

## 定位与边界（必读）

- 本 skill 是**导航层**：回答"这个算子该用什么编程模式、抄哪个例子、走哪条搬运路径"。
- **API 语义与约束的权威是 [tilelang-ascend skill](../tilelang-ascend/SKILL.md)**（§1–§12）。
  本文件只引用其章节号，不复制内容；遇到任何约束细节冲突，以该 skill 与
  `tilelang/ascend/language/` 源码为准。
- `T.simd.*` 完整 API 面见 `tilelang/ascend/language/simd.py`（约 1500 行）。
  本 skill 只索引高频子集（见下文），**禁止凭记忆猜测未索引的 intrinsic**。
- 实现新算子时，遵循 AGENTS.md「Operator Implementation Baseline」：
  优先改 `examples/ascend/` 与 `testing/ascend/` 中最接近的既有案例，
  而不是从零写全新 kernel。

## 编程模式速查

| 模式 | 入口 | 适用 | 权威章节 |
|---|---|---|---|
| `simt`（默认推荐） | `T.SimtVF(threads=N)` + `T.Parallel` 循环 | 可读性优先；编译器自动向量化与自动调度 | tilelang-ascend skill §2.2 |
| `simd`（热点降级） | `T.SimdVF()` + 显式 `T.simd.*`（2048-bit intrinsic，可带 `latency=`） | 已被 profiling 证实的热点；需要特定指令/掩码/舍入模式 | tilelang-ascend skill §2.3、§8 |
| `mixed`（GEMM 主导） | `T.MixedKernel(sids=1\|2)` 或 `T.Kernel` 自动识别 | 同一 kernel 内含 Cube 侧 `T.gemm` 与 Vector 侧工作 | tilelang-ascend skill §3 |

正交字段 `auto_schedule`（yes/no，默认 yes）：默认依赖 AutoSchedule 自动插同步；
`no` 仅用于完全手动核（`TL_ENABLE_AUTO_SCHEDULE: False` + 手动 flag 配对），
必须先读 tilelang-ascend skill §3.3 与其 references/synchronization.md。

选型原则：**从 simt 起步**，精度通过后用 `do_bench(backend="msprof")` 实测，
仅对证实的热点降级 simd 或引入 mixed。禁止跳过 simt 基线直接手写 intrinsic。

## 算子族导航表

| 算子族 | 推荐模式 | 参考实现 | 布局/搬运套路（一句话） | 权威章节 |
|---|---|---|---|---|
| elementwise（逐元素/量化/cast） | simt，热点降级 simd | `examples/ascend/example_vecadd.py`（simt+simd 双模式对照）、`example_per_token_cast_to_fp8.py`；回归：`testing/ascend/language/simtvf/`、`simdvf/` | GM→UB 用 `T.copy` → VF 计算 → UB→GM 用 `T.copy`；`T.Pipelined` + buffer versions 流水 | §6.1、§2.2/§2.3 |
| gemm（含 split-K、量化 GEMM） | mixed（`T.Kernel` 自动识别或 `T.MixedKernel(sids=2)`） | `examples/ascend/example_gemm.py`（persistent + swizzle + mixed epilogue）、`example_gemm_splitk.py`、`deepgemm/`（`kernels/kmn`、`kernels/mnk`、`epilogue.py`、`copy.py`）；回归：`testing/ascend/language/test_tilelang_ascend_l0_gemm_effective_k.py` | GM→L1 用 `T.copy`（ND2NZ 分形由 pass 自动插入，禁止手写）→ `T.gemm(A_l1, B_l1, C_l0c, transpose_B=True)` → `T.dual_copy` L0C→UB（双 AIV）/ `T.copy` L0C→GM；权重存 `[N, K]` | §5、§6.2、§3.2 |
| attention（MHA/GQA） | mixed（两次 GEMM + Vector 在线 softmax） | `examples/ascend/flash_attention/`：`example_mha.py`、`example_gqa.py`、`example_gqa_manual_schedule.py`（显式 `T.Stage` 版）、`core.py`/`core_bwd.py`（共享 builder） | 与 gemm 同路径 ×2（QK^T、PV）；softmax 概率**直接写 NZ 布局**供第二 GEMM 消费，省一次 ND↔NZ 转换（见 `core.py` 头注） | §5、§8、references/synchronization.md |
| reduction-norm（归约/归一化） | simt（reducer epoch API / fragment 归约） | `examples/ascend/example_rmsnorm.py`（fragment + reducer + 行批处理）；回归：`testing/ascend/language/test_tilelang_ascend_reduce.py`、`test_tilelang_ascend_warp_reduce.py` | GM→UB 用 `T.copy` → `T.alloc_fragment` 向量化加载（2 的幂填充 + 谓词掩尾）→ `T.alloc_reducer` + `reducer_init/update/finalize_reducer` → 结果写回 UB/GM | §4.1、§10.7/§10.11 |

> 各族的模式决策依据、结构骨架伪代码与易错点映射见
> [references/family-guide.md](references/family-guide.md)。

## `T.simd.*` 高频子集索引

仅索引高频子集；命名、掩码/谓词语义、dtype 支持以
`tilelang/ascend/language/simd.py` 与 tilelang-ascend skill §8 为准。

| API | 用途 |
|---|---|
| `T.simd.pset` | 设置谓词/掩码（按元素位宽声明有效 lane） |
| `T.simd.vld` / `T.simd.vsts` | 2048-bit 向量 load / store（偏移写进地址表达式） |
| `T.simd.pld` / `T.simd.pst` | 谓词 load / store（配合 `dist="US"/"PK"` 等） |
| `T.simd.vadd` / `T.simd.vsub` / `T.simd.vmul` / `T.simd.vmax` | 向量算术 |
| `T.simd.vaddc` | 带进位加（int32/uint32） |
| `T.simd.vexp` / `T.simd.vln` | 向量数学（指数/对数） |
| `T.simd.vcvt` | 向量类型转换（如 `"float32"`/`"bfloat16"`） |
| `T.simd.vintlv` / `T.simd.vdintlv` | 交错 / 解交错（偶奇 lane 合并/还原） |

lane 宽度、舍入、饱和与 dtype 支持需查安装的 SDK 头文件与官方 Ascend 文档
（tilelang-ascend skill §8 开头的说明）。

## 新算子通用路线

1. **定族选模式**：查导航表定算子族与初始模式（一律 simt 起步，gemm/attention 用 mixed）。
2. **找最近例子**：先 `examples/ascend/`，再 `testing/ascend/`（按
   `language/layout/analysis/auto_schedule/transform/target/runtime` 边界找）；
   优先改既有案例，无接近模板时才建最小新 kernel 并显式说明。
3. **设计搬运路径**：按族套路确定 `T.copy`/`T.dual_copy`/`T.gemm` 的层级链路
   与 UB/L1/L0 容量预算（tilelang-ascend skill §4；容量常数见
   pattern-library `constants.md` 的 CONST-capacity-a5 条目）。
4. **编译与源码检查**：`tilelang.compile(program, target="ascend")` 后
   `kernel.get_kernel_source()` 检查生成的 AscendC 源码是否符合预期。
5. **精度与性能**：golden 用 torch 在 NPU（`device="npu"`）上算，
   `torch.testing.assert_close` 按 dtype 分档容差（fp8→1e-1，bf16/fp16→1e-2，
   fp32→1e-3，累加重→5e-3）；性能以 `do_bench(fn, backend="msprof")` 为唯一
   端到端口径（tilelang-ascend skill §12）。

## References

- [references/family-guide.md](references/family-guide.md)：每算子族一节——
  模式决策、参考实现锚点、结构骨架伪代码（≤30 行/族）、易错点映射到
  tilelang-ascend skill §10 对应条目。

## Related skills

- tilelang-ascend（编程权威）
- tilelang-op-design / tilelang-op-develop / tilelang-op-optimize（conductor 流程）
- tilelang-ascend-debug（调试）/ tilelang-ascend-error-fixer（错误诊断）
