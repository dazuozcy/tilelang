# 已知编译器/解析器陷阱（工具链版本绑定 ⚠️）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)，front-matter schema 见其 §5）。
>
> **A5 冷启动（2026-10-16）**：本文件自 tilelang-mlir-fork（npuir/910B2C 实测）复制后**内容清零**——原条目绑定 npuir 工具链实测，不适用于 tilelang-fork（ascend/A5，dav-3510）。schema 保留，条目由 A5 任务蒸馏积累。

（2026-10-16 首条合入，Tier 0）

---
id: TRAP-MACRO-SCOPE-PLACEMENT
kind: trap
family: [general]
apis: [T.macro, T.prim_func, T.serial, T.SimtVF, T.Parallel]
dtype: []
device: A5/950 (dav-3510)
status: measured
origin_task: logsumexp-harness（_logsumexp_kernel_tiled，2026-10-16；溯源：该任务 Stage 3 首次尝试把 macro 定义在 prim_func 内报 `NameError: name 'sub_batch' is not defined` 的修复记录 + RETROSPECTIVE Stage 3 + L0 ⑤⑥⑦ 全 PASS——允许失效）
toolchain: tilelang 0.1.15+cuda.git40c9d668 + Ascend950PR_9579 / CANN 9.2.0，2026-10-16
repro: repro-missing
---

### `@T.macro` 定义层级陷阱：须在 `@T.prim_func` 外层（builder 函数层）；体内 SimtVF/T.serial/Python if 均可用

- 定义在 `@T.prim_func` 内部会 `NameError`（AST 解析器不内联非 macro 的嵌套 def）；合法形态：macro 定义在 builder 函数体内、prim_func 之前——编译期常量（threads/num_full/rem/TILE_R/SUB/BLOCK_N/dtype）走闭包捕获，运行期 buffer（x_ub/stats_ub/x/y）与行基址（TIR 表达式）走 macro 参数传入。
- macro 并非「仅 SimdVF」专用（flash_attention 先例全是 SimdVF）：macro 体内 `T.serial`/`T.SimtVF`/`T.Parallel`/`T.reduce_*`/`T.And` 与 Python `if`（编译期常量条件，如 `if num_full > 0` / `if rows == SUB`）组合在 A5 端到端编译运行通过（logsumexp tiled L0 ⑤⑥⑦ 全 PASS）。
- 用途实证：子批计算体 4× 重复（full/partial × 全块/尾块）用 macro 干净消除（消除 ~240 行内联重复）；不知此约束会退化为大段内联或误用 helper 函数（嵌套 def 同样 NameError）。
