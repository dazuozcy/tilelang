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

---

id: CONST-PrepareMultiBuffer-bool-cast-255
kind: trap
family: [gemm, mixed, moe]
apis: [T.MixedKernel, T.gemm, T.Pipelined, T.alloc_l0c, T.copy, T.alloc_var]
dtype: [fp16, bf16, fp32]
device: A5/950 (dav-3510)
status: measured
origin_task: indexed_expert_mlp（_indexed_expert_gemm，2026-10-29；溯源：该函数 RETROSPECTIVE.md Stage 3「运行时守卫包流水化 gemm」条 + probe_bisect.log 8 变体二分 + repro 转正记录——允许失效）
toolchain: tilelang-fork ascend dev build（report git_commit a2a92cf9 同代工作区）+ CANN 9.2.0 + bisheng 15.0.5 + Ascend950PR_9579，2026-10-29
repro: repro/CONST-PrepareMultiBuffer-bool-cast-255.py
---

### 运行时守卫包流水化 `T.gemm` → PrepareMultiBuffer 崩溃 `255 exceeds range of bool`（守卫只能包 epilogue 写回）

- 触发条件：L0C accumulator 的 write（`T.gemm`）与 read（`T.copy(res, acc_ub)`）处于**不同运行时守卫**（如 `if is_leader != 0:` 包住整段 Pipelined gemm）。PrepareMultiBuffer → AnalyzeStorageEpoch → `arith::Analyzer::CanProveEqual` 对守卫做 canonical simplify，把 `cast(bool, sub(const, const))` 折叠成 `make_const(bool, 255)`，触发 `ValueError: Check failed: (value == 0 || value == 1) is false: 255 exceeds range of bool`，编译中止。守卫比较形式（`!=0`/`==1`/`>0`/`>=1`）、守卫读 `T.alloc_var` 可变标量还是纯 let 标量均无关；`clear_accum=(kt==0)` 不触发（8 变体 bisect 实证：唯二 FAIL 均含「运行时守卫包 gemm」）。
- 报错文本严重误导根因定位：bool 值域检查与真实根因（L0C write/read 守卫不配平）完全无关，按报错文本排查无任何指向守卫配对的线索。
- 规避范式（真机 40/40 用例验证，per-route leader 判定类分组 MoE epilogue 直接复用）：**守卫只包 AIV1 epilogue 写回，Cube 流水线无条件对全块执行**（`if sid == 1: if is_leader != 0:`）——非 leader 块计算冗余但逐位相同的 tile 并丢弃，不破坏 L0C accumulator 单守卫配平、不引入额外同步。
- 上游根因与补齐跟踪：capability-gaps.md `CG-2026-0003`（cast() 工厂对 bool 域未掩码的 const-folding bug，3rdparty/tvm `rewrite_simplify.cc:543` + `op.cc:439`；工具链升级后本条自动待重验，repro 会报 `REPRO_MISS` 提示已修复）。

---
id: TRAP-SIMD-SCAN-FORMS
kind: trap
family: [simd, scan, metadata]
apis: [T.simd.vscatter, T.simd.vselr, T.simd.vci, T.simd.alloc_local, T.copy, T.fill, T.SimdVF]
dtype: [int32]
device: A5/950 (dav-3510)
status: measured
origin_task: indexed_expert_mlp Stage 4（_route_stats_round2_batchscan 开发，2026-10-30；溯源：该任务 perf_opt/logs/round2/batchscan_L0.log 三次迭代报错 + probe_runtime_scalar.py / probe_masked_store_scatter.py D-3 探针——允许失效）
toolchain: tilelang-fork ascend dev build + CANN 9.2.0 + bisheng dav-3510 + Ascend950PR_9579，2026-10-30
repro: repro/TRAP-simd-scan-forms.py
---

### SimdVF 批式 scan 四形态陷阱组（vscatter 索引 uint32 / vreg 循环内重绑定 / 2-D→1-D copy / fill 64 整除）

批式向量 scan（vselr 位移链 + vscatter 三段写）开发中实测的四个编译期形态约束，每个均有确定报错文本与合法替代形态（repro 六断言全 PASS）：

1. **vscatter 索引必须无符号向量**：int32x64 索引在 `ub_scatter.h` 无匹配 `asc_scatter` 重载 → bisheng「no matching function for call to 'asc_scatter'」（dst int32 的候选要求 `vector_uint32_t index`）。**修复**：`tirx.reinterpret("uint32x64", idx)` 前置转换（vselr 内部自动做，vscatter 不做——调用方责任）。32-bit 元素 → uint32x64；8/16-bit → uint16。
2. **TIR script 循环/链内 vreg 重绑定非法**：Python 变量重赋值（`cnt = vadd(cnt, x)`）在 unrolled/serial region 内触发「Immutable variable `cnt` is used outside its defining region!」——重绑定的 Bind 落在链的 region 内、链后引用即越界。**修复**：`T.simd.alloc_local((1,), "int32")` cell + `cnt[0] = vadd(cnt[0], x, full)`（mutable buffer cell，test_simdvf_mutable_reads 先例）。
3. **2-D src → 1-D 切片 T.copy 被拒**：`T.copy(expert_ids, flat_ub[0:R])`（[T,K]→[R]）触发「PlanMTECopy: source and destination regions must contain the same number of elements」（报错数字与实际元素数无关，误导性強）。**修复**：同形 [T,K]→[T,K] 拷贝 + 消费侧用 `(lo // K, lo % K)` 二维下标寻址（线性地址等价）。
4. **SimdVF 域内 T.fill 须 64 整除**：region 元素数非 VReg 64 倍数触发「CanLowerFillRegion ... requires a contiguous region whose element count is divisible by VReg size (64)」。小 shape 尾区零填**不可用 fill**——若尾区垃圾值的全消费路径已被掩码覆盖（eq 折入 valid / gather/scatter 带 mask），直接不填更优。

附注（未证实形态，非断言）：`vsts(addr, src, mask=)` 的 mask 掩蔽有效性在本任务探针中未获确证（v2 探针 store 路径被自身 vsel 参数序 bug 覆盖；v1 探针直接标量 update_mask 作 vsts mask 未生效）——三段写全部改走已实证的 `vscatter(mask=)`（ranks 段用连续 route ramp 作索引）。使用 vsts 掩蔽写前须自测。
