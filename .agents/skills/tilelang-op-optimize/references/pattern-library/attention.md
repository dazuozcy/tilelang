# Attention / Expert persistent 模式与硬上限（attention 族实测）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)，front-matter schema 见其 §5）。
>
> **A5 冷启动（2026-10-16）**：本文件自 tilelang-mlir-fork（npuir/910B2C 实测）复制后**内容清零**——原条目绑定 npuir 工具链实测，不适用于 tilelang-fork（ascend/A5，dav-3510）。schema 保留，条目由 A5 任务蒸馏积累。

---

id: D-MIXED-EPILOGUE-DUALCOPY
kind: pattern
family: [gemm, mixed, elementwise]
apis: [T.dual_copy, T.gemm, T.SimtVF, T.exp]
dtype: [fp16, fp32]
device: A5/950 (dav-3510)
status: measured
origin_task: pilot-gemm_silu-20261016T034510Z（溯源：该任务 gemm_silu.py §3.3 主选路径 L0/L1/L2/Boundary 全 PASS + DESIGN.md §1.6.4 风险项回填为「已证实可行」）
toolchain: tilelang-fork fc32f47c / CANN 9.2 / bisheng dav-3510 / Ascend950PR_9579，2026-10-16
repro: repro-missing
---

### mixed CV epilogue 拓扑实证：dual_copy 间插 SimtVF 元素段（dav-3510 AutoSchedule 直接支持）

- 拓扑：`T.dual_copy` L0C→UB（**fp32 累加器以 fp32 落 UB——dual_copy 禁止类型转换**）→ `with T.SimtVF(threads=128): for i, j in T.Parallel(TILE_M//2, TILE_N)` 逐元素变换（含 `T.exp`、直除、fp32→fp16 cast）→ `T.dual_copy` UB→GM。
- 实证（2026-10-16）：该拓扑在 dav-3510 上直接编译运行通过、L0/L1/L2/Boundary 全 PASS——AutoSchedule 正确处理 dual_copy 间插 SimtVF 元素段的所有权与同步，**无需** deepgemm 式 `for _ in range(1) + T.assume_no_conflict` 隔离，也无需 MixedKernel sid guard。此前仓内无直接先例（example_gemm 的 MIXED epilogue 为纯搬运、deepgemm 元素变换走 plain `T.copy` 链）。
- cast 后置（silu 于 fp32 累加器上施加、fp32 表达式赋值 fp16 buffer 时隐式完成）同时是精度最优路径：消除 z 的 fp16 预舍入（z 可达 ~10² 量级，预舍入误差被放大）。
- 结构收益（分析口径，512³ fp16 融合 GEMM+silu，R3 表）：融合单 kernel vs 分离两 kernel 的 GM 流量 **1.5MB vs 2.5MB（−40%**，消除中间量 Z 的 GM 写+读两遍）+ 省 1 次 kernel launch；FLOPs 口径两者相同（融合不省计算）；CV 跨 tile 流水重叠使向量段（2136 cyc）与 Cube 段（2048 cyc）接近 max 而非 sum。
