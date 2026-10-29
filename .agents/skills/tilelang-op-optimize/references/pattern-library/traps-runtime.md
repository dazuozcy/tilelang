# 已知运行时/数值/语义陷阱（工具链版本绑定 ⚠️）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)，front-matter schema 见其 §5）。
>
> **A5 冷启动（2026-10-16）**：本文件自 tilelang-mlir-fork（npuir/910B2C 实测）复制后**内容清零**——原条目绑定 npuir 工具链实测，不适用于 tilelang-fork（ascend/A5，dav-3510）。schema 保留，条目由 A5 任务蒸馏积累。

---

id: TRAP-GRID-DIVISIBILITY
kind: trap
family: [elementwise, general]
apis: [T.Kernel, T.Pipelined, T.copy]
dtype: []
device: A5/950 (dav-3510)
status: measured
origin_task: pilot-silu-20261016T025344Z（溯源：该任务 Stage 1 探针 v1 运行记录〔28-blocks 配置〕+ DESIGN.md §5.3/§9.2）
toolchain: tilelang-fork dev build + CANN 9.2 + bisheng dav-3510 + Ascend950PR_9579，2026-10-16
repro: repro-missing
---

### grid 整除性陷阱：不整除时尾部静默漏写、无编译错误（伪装成精度失败）

- 触发条件：元素总量不被 `NUM_BLOCKS × TILE × 迭代数` 整除。实证配置：TOTAL=131072（=2^17，无因子 7）、28 blocks × TILE 4096、`T.Pipelined(1)`——循环只覆盖 114688/131072 元素，**12.5% 输出未初始化**（实测 max abs diff 4.8 的随机垃圾值），症状是精度失败而非任何报错。
- 核数耦合（本机 Ascend950PR_9579 实查 cube=28 / vector=56，见 constants.md CONST-arch-a5）：28 = 4×7 含因子 7，与 2^n 元素总量天然互斥——「惯例 32 blocks」（2 幂，examples/ascend/deepgemm/config.py `num_cores=32`）整除 2^17 但与实查核数不匹配；grid 取舍须同时满足整除性与核占用两约束。
- 防御事实（该任务实测通过的做法）：host 侧 `assert TOTAL % (NUM_BLOCKS * TILE) == 0` 显式防御；grid 取 2 的幂（32 × 2048 × 2 精确整除 2^17）或核数倍数且保证整除。

---

id: TRAP-NAIVE-ONLINE-MERGE
kind: trap
family: [softmax, reduction]
apis: [T.exp, T.reduce_max, T.log]
dtype: [fp16, fp32]
device: A5/950 (dav-3510)
status: measured
origin_task: pilot-online_softmax-20261016T044814Z（溯源：该任务 verify_equiv.py V4b 诊断臂 + DESIGN.md §1.6.1 O6；修复形态见该任务 online_softmax.py `_build_streaming`）
toolchain: tilelang-fork 工作区（fc32f47c 后）+ torch CPU fp64 对照，2026-10-16
repro: repro-missing
---

### 朴素 online softmax 合并（m_c 基 lse 式）角点发散（数值陷阱）

- 源式 `lse ← m_c + log(exp(lse−m_c) + S_c)` 的两个发散角点：① 块间 max 间隙 **> 88.7**（=128/log2e，fp32 exp 上界）时 `exp(lse−m_c)` fp32 溢出 inf → 全 0 输出；② 块内全 −inf 时 `exp(−inf−(−inf))=NaN`。fp16 数据域 max 间隙可达 **131008**（=2×65504），远超 88.7——fp16 softmax 迁移/设计必中。
- 实测角点违反率（verify_equiv V4b 诊断臂，源式 vs 精确参照）：fp64 口径 **0.136**、fp32 口径 **0.182**。
- 修复实证（同任务 63 用例全 PASS，fp16 max_diff ≤ 6.1e-5，容差 1e-2）：flash 形式 max 基合并 `m_run = max(m_run, m_c)`、`s_run = s_run·exp(m_run_old−m_run) + S_c·exp(m_c−m_run)`（全部 exp 参数 ≤ 0，无溢出）+ fp16 哨兵钳位 `m_c = max(reduce_max(x), −65504)`。**钳位须在 exp 之前**——先 exp 再钳位则 `exp(−inf−(−inf))=NaN` 污染 s_run。良性数据上两式恒等（log-sum-exp 递推的代数变形，V4 实测 max_rel 1.32e-12）。
- **logsumexp 变体的钳位弃选判据（2026-10-16，logsumexp tiled verify_equiv V5 8 角点行实测）**：哨兵钳位在 logsumexp 族的唯一作用是改变全 −inf 行为（源链固有 NaN → 钳位后 −inf）= 语义变更，且 flash 形式角点本已安全（exp 参数 ≤ 0，fp16 跨 chunk max-gap 131008 无溢出；部分 −inf 含整 chunk 全 −inf 两种到达顺序完全中性；主导值跨 chunk 位级正确）——弃用。**钳位取舍按目标语义（源/torch 参考在退化行的行为）定：softmax 保留（保 torch NaN 传播），logsumexp 弃用（源固有 NaN）；flash 形式本身两族通用**。

---
id: TRAP-NPU-DOUBLE-SILENT-FP32
kind: trap
family: [general]
apis: []
dtype: [fp64, fp32]
device: A5/950 (dav-3510)
status: measured
origin_task: logsumexp-harness（_logsumexp_kernel_single，2026-10-16；溯源：该任务 Stage 3 `_source_lse_sim` 由 `.double()` 改 `.float()` 的修复记录 + 运行 warning 原文 `ToKernelNpu.cpp:40 Device do not support double dtype`——允许失效）
toolchain: tilelang 0.1.15+cuda.git40c9d668 + Ascend950PR_9579 / CANN 9.2.0，2026-10-16
repro: repro-missing
---

### A5 NPU 无 double dtype：torch `.double()` 静默降级 fp32（伪 fp64 参照陷阱）

- NPU 侧 torch 张量 `.double()` 静默降级为 float32 并打 warning（`ToKernelNpu.cpp:40 Device do not support double dtype`）——不报错，极易误以为拿到 fp64 参照。
- 影响：fp64 参照/模拟器（等价性验证、精度上限估计）须在 CPU 侧跑；NPU 侧 "fp64" 是伪 fp64。logsumexp verify_equiv 的 fp64 对照链即因此把 NPU `.double()` 改为 CPU 计算或显式 `.float()`。
- 复现口径：`PYTHONPATH=<repo> python -c "import torch,torch_npu; x=torch.ones(2,2,device='npu'); y=x.double(); print(y.dtype, (y@y).dtype)"`（观察 warning 输出）。

---
id: TRAP-SOURCE-VS-TORCH-GOLDEN
kind: trap
family: [reduction, logsumexp, softmax]
apis: []
dtype: [fp16, bf16, fp32]
device: A5/950 (dav-3510)
status: measured
origin_task: logsumexp-harness（两函数，2026-10-16；溯源：verify_equiv.py V3b/V4〔single〕+ V5 8 角点行〔tiled〕——允许失效）
toolchain: tilelang-fork 工作区（fc32f47c 后）+ torch CPU，2026-10-16
repro: repro-missing
---

### 源算法固有角点与 torch 参考分歧：迁移保真的 golden 语义选择（logsumexp 全 −inf 行）

- logsumexp 源算法链（m+log(Σexp(x−m))）在全 −inf 行输出 NaN、含 +inf 行输出 +inf，与 `torch.logsumexp`（−inf / +inf）**固有分歧**——randn 测试域不触发，Boundary 用例才暴露。
- 迁移保真（语义保持）应保持源语义：kernel 复刻源链则角点自然与源位级一致（V4 实测链与源位级一致）；此时 golden 不能盲用 torch 参考判 PASS——Boundary 用源语义模拟对照（或 `assert_close(equal_nan=True)` + 显式记录分歧清单），与 elementwise.md D-LOGSUMEXP-MIGRATION-FIDELITY 的 randn 域位级结论（golden 可直用 torch.logsumexp）适用域互补：randn 域用 torch、退化行按源语义。
- 流式变体补充（tiled V5）：部分 −inf（含整 chunk 全 −inf 两种到达顺序）对源式 flash 合并完全中性、fp16 跨 chunk max-gap 131008 无溢出——源式 flash 合并的角点谱本身安全（对照上方朴素式的两个发散角点）。
