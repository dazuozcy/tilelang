# Ascend A5 性能模式库索引（pattern-library/INDEX.md）

> **本目录是 `pattern-library` 的唯一入口**（K-2 渐进披露：INDEX 必读 + `kb_search` 命中条目精读）。收录**已实测验证**的性能模式、代价数据、编译器/运行时陷阱与实验方法学规则。
>
> **冷启动说明（2026-10-16，A5 适配 P0）**：本库自 tilelang-mlir-fork（npuir/A2A3，910B2C 实测）清零重建——历史条目不适用于 tilelang-fork（ascend/A5，dav-3510）。调优手段种子知识源：`tilelang-ascend` skill §12（profiling recipes）与 `maint/scripts/ascend_perf_regression.py`。

## 1. 维护规则（由 optimizer 任务内回写 + `tilelang-skill-evolver` 终态蒸馏共同维护）

- 每次调优任务产出的新模式/代价数据/证伪更正，由调优 Agent 在任务结束前追加到对应**主题文件**（含溯源与版本戳）；evolver 是第二授权写入者（五种 delta：add/update/consolidate/negate/deprecate；含其他 Stage 的 D/C 类价值点）；两次写入互补防双写（evolver 跳过已回写条目）。
- **条目溯源（D2）**：新增条目须带 `origin_task` 与工具链版本戳；**下游引用时须标注其 origin_task**（optimizer Phase 0 / designer 步骤 0.5 / 复盘引用），据此判别可信度与重验需求。
- **预算与生命周期**（evolver 维护，K-2 字节预算）：INDEX ≤12KB（超限优先压缩"一句话"列而非删条目）；主题文件 ≤16KB；`repro/` 每文件 ≤8KB/100 行。超限触发 consolidate（负面条目只可 deprecate）；`update` 优先于 `add`（同主题已有条目禁新开）；**工具链变更后条目自动"待重验"**（`kb_stale_check.py` 机械检测）。
- **条目自包含（ED-A）**：关键数字、适用条件（shape/dtype/workload 上下文）、结论必须内嵌条目正文——provenance（origin_task 与出处）允许失效，条目仍完整可用；**代码证据唯一合法形态是 `repro/` 内自包含脚本**（`repro: repro-missing` 表示待回填）。**经验需要代码时，用最少的代码表达**（ED-B）：优化点条目（kind: pattern）的代码证据 = 慢→快关键代码更改的最小形态（完整对照或 delta 骨架）+ 优化见解摘要（机制归因到 CONST-\*/TRAP-\* 条目）。
- **assumption 折扣纪律**：`constants.md` 中 `status: assumption` 条目在设计期 roofline 估算中须打折（容量按 90% 预算）并在 DESIGN.md 披露所用条目与折扣；转 `measured` 后移除折扣；假设与实测冲突时以实测为准并 update 条目。
- 机械校验：`python3 .agents/tools/kb_lint.py`（三件套齐备、front-matter 合法、字节预算、repro 存在性与解耦、状态枚举）；检索：`python3 .agents/tools/kb_search.py "<query>"`。

## 2. 证伪协议（canonical，否定一个 API/模式前必须遵守）

1. **必须用文档合法形态测试**——否定任何 API/模式前，先按 AGENTS.md「Docs Auto Routing」查权威文档（`.agents/skills/tilelang-ascend/SKILL.md` 及其 references、`docs/programming_guides/`、`tilelang/ascend/` 源码）确认其合法参数/形式，穷举代表性写法后再下结论。
2. **一切编译器/运行时结论必须盖工具链版本戳**（tilelang commit/build 时间 + CANN/bisheng 版本 + 设备型号），工具链变更（源码修改/重编译）后**自动视为待重验**，不得直接引用旧结论。
3. **证伪更正须留痕**：推翻旧结论时在 opt_log 写明"误判根因 + 合法形态 + 新数据"。

## 3. 检索规则（调优 Agent 首轮必读，内容或瓶颈变化时刷新）

1. **首轮必查项**：① 编程模式重估（SimtVF/SimdVF/mixed 对照 `tilelang-ascend` skill §2/§8——设计期结论可能基于旧工具链）；② 标量占比诊断（msprof_detail pipe ratios：`aiv_scalar` 高占比触发换轴）；③ 陷阱条目版本戳与 origin_task 核对（工具链变更则重验，`kb_stale_check.py`）。
2. 新模式/新代价数据/证伪更正 → 任务结束前追加进对应主题文件（含溯源）。
3. 本库模式是**起点不是终点**：每算子的最优 tiling/grid 仍须实测扫描。

## 4. 条目索引（ID + 一句话 + 状态 + 所在文件）

> 完整 front-matter（family/apis/dtype/device/status/origin_task/toolchain/repro）见各条目；`kb_search.py` 按本索引与条目 facets 检索。

### constants.md — 硬件常数表（D-2，设计期 roofline 口径）

| ID | 一句话 | 状态 |
|----|--------|------|
| CONST-capacity-a5 | 片上容量：UB 248KB（SIMT 216KB）/ L1 512KB / L0C 256KB / L0A/L0B 64KB（auto_schedule.cc 源码） | assumption |
| CONST-arch-a5 | A5/950：AIC:AIV=1:2 编组、sids 1/2、npu-arch dav-3510、CANN 9.x；实查 cube=28/vector=56 | measured |
| CONST-gemm-dim-constraints | GEMM 维度约束：M/N 16 倍数、K%TILE_K_SUB==0 且 %C0、C0=32B/elem、FRAC_N=16、NT 约定 | assumption |
| CONST-cube-throughput-model | Cube 吞吐代价模型（fp16/bf16 8192、fp8 16384、HF32 4096、fp32 512、INT8 MMAD 不支持，ops/cycle/AIC） | assumption |
| CONST-mte-bandwidth-model | MTE/FixPipe/dual_copy 带宽与延迟模型（B/cycle + base cycles，estimate_latency.h） | assumption |
| CONST-valu-model | VALU 吞吐与 VRF↔UB 带宽（51/13/8 ops/cycle、256 B/cyc/AIV）；exp/exp2 同档；标量 tanh 软件例程反例 1.80×<!-- codespell:ignore --> | assumption |
| CONST-vrf-capacity-a5 | VRF 容量假设 256KB（无文档常数）+ 双口径预算 + spill=0 实测（76/48/16 reg/thread）；fragment 落 VRF 不占 UB | assumption |
| CONST-mte-nd2nz-short-seg | ND2NZ 短跨步行段 MTE2 实测 ~2 B/cyc/AIC（模型低 50×；通道/段长无关，流量唯一杠杆；去冗余后段长再生效） | measured |
| CONST-scalar-ub-scan-latency | 标量 UB 扫描链 c_scan ≈700-800 cyc/iter（假设低 50×；向量链 ~12 cyc/op；向量化收益按周期常数比估） | measured |

### layout.md — 布局与搬运模式

| ID | 一句话 | 状态 |
|----|--------|------|
| P-LAYOUT-ROW-RESIDENCY | row-reduction 驻留 vs 流式判定四步法 + SUB 行子批 + wrapper 双变体分界 + persistent ROWS 粒度规则（双任务实测） | measured |
| P-LAYOUT-TAIL-STATIC-EXTENT | 尾块处理范式：host 静态分解 + 元编程 if + 静态 extent 切片 + BLOCK_N pow2 重划（tile 参数=预算上限） | measured |

### elementwise.md — 多输入搬运效率

| ID | 一句话 | 状态 |
|----|--------|------|
| D-ELEM-SIMTVF-AUTOVEC | SimtVF T.Parallel 内 tir.* intrinsic 自动向量化 2048-bit（vexp/vdiv），fp16 可用（3.906e-3 内） | measured |
| D-ELEM-ACT-FORMULA-PROBE | silu/sigmoid 公式选型实测：直除 0.902× 最优、tanh 形式 1.80× 劣化（软件例程）、vexpdif 仅 SimdVF+fp32 | measured |
| D-REDUCE-SIMTVF-API | reduce_max/sum → AscendAllReduce 树归约（树宽=threads，U 曲线 256 最优）；extent<threads 复制布局合法；−inf pad 中性；flash chunk 不变；T.And | measured |
| D-LOGSUMEXP-MIGRATION-FIDELITY | logsumexp 迁移保真：golden 直用 torch.logsumexp（16-bit 位级、fp32 ≤1 ulp）；fp16-e 中间链 50–60x 实测否决——保 fp32 fragment | measured |
| D-LOGSUMEXP-SIMD-F16MAX | SimdVF pass-1 16-bit 域 max（NORM+vmax，bf16 需 EVEN/ODD 拆分——asc_reduce_max 无 bf16 重载）：位级等价 +4~+15.5%；反模式（步进/配对/掩码提升）否决 | measured |
| D-SIMD-PERSISTENT-ROWS-LSE-BATCH | SimdVF 流式变体 ROWS：lse 微写批量化（mte3 64.3→6.2%），M≤56 保 1 / 否则 makespan ≤min+1 pow2 cap 8——32k +26.6% | measured |

### attention.md — attention / 混合核族

| ID | 一句话 | 状态 |
|----|--------|------|
| D-MIXED-EPILOGUE-DUALCOPY | mixed CV epilogue：dual_copy L0C→UB → SimtVF 元素段（含 T.exp）→ dual_copy UB→GM（AutoSchedule 支持）；融合 vs 分离 GM −40% | measured |

### traps-compiler.md — 编译器/解析器陷阱

| ID | 一句话 | 状态 |
|----|--------|------|
| TRAP-MACRO-SCOPE-PLACEMENT | `@T.macro` 须定义在 prim_func 外层（内层 NameError）；常量闭包捕获、buffer 参数传入；体内 SimtVF/T.serial/Python if 可用 | measured |
| CONST-PrepareMultiBuffer-bool-cast-255 | 运行时守卫包流水化 `T.gemm`（L0C 守卫不配平）→ PrepareMultiBuffer 崩溃 `255 exceeds range of bool`；规避 = 守卫只包 epilogue、Cube 无条件 | measured |
| TRAP-SIMD-SCAN-FORMS | 批式 scan 四形态陷阱：vscatter 索引须 uint32 / vreg 链内重绑定非法（alloc_local cell）/ 2-D→1-D copy 被拒（同形+二维下标）/ fill 须 64 整除 | measured |

### traps-runtime.md — 运行时/数值/语义陷阱

| ID | 一句话 | 状态 |
|----|--------|------|
| TRAP-GRID-DIVISIBILITY | grid 不整除元素总量（2^17 无因子 7 vs 28/56 核）→ 无编译错误、尾部静默漏写 12.5% 伪装精度失败 | measured |
| TRAP-NAIVE-ONLINE-MERGE | 朴素 online 合并角点发散（max-gap>88.7 溢出/全 −inf NaN，违反率 13.6%/18.2%）——flash 修复；钳位按族取舍（softmax 留/logsumexp 弃） | measured |
| TRAP-NPU-DOUBLE-SILENT-FP32 | A5 NPU 无 double：torch `.double()` 静默降级 fp32 + warning（ToKernelNpu.cpp:40）——fp64 参照须 CPU 侧 | measured |
| TRAP-SOURCE-VS-TORCH-GOLDEN | 源算法固有角点与 torch 参考分歧（logsumexp 全 −inf 行 NaN vs torch −inf）——randn 域不触发，Boundary 按源语义对照 | measured |

### cases.md — 案例索引与参考实现集

| ID | 一句话 | 状态 |
|----|--------|------|
| CASE-pilot-silu | simt 纯 Vector 扁平 elementwise 模板（参数化工厂+整除防御） | measured |
| CASE-pilot-gemm-silu | mixed GEMM+激活 epilogue 融合案例（CV 融合 −40% GM + 探针否决 tanh） | measured |
| CASE-pilot-online-softmax | GPU→A5 迁移 row-reduction 案例（换底复核+驻留判定+flash 哨兵+S3-JIT 教训） | measured |
| CASE-logsumexp-harness | migration-harness 全链首例（Stage 0→5）：双变体位级保真、@T.macro 因子化、检视探针闭合先例空隙、Stage 5 零 attempt | measured |
| CASE-logsumexp-simd | SimdVF 双变体迁移+Stage 4（SIMT 变体共存、16-bit 域 max、stats-UB 往返；32k +28.7%、SIMD 1.2–2.6x 于 SIMT） | measured |
| P-GEMM-LEADER-SKIP-LOADS | 分组 GEMM 非 leader 块 W 装载跳过（守卫只包 W copy 不包 gemm）：gate/up +21%/down +52%（t32，叠 config）；流量线性杠杆 | measured |
| CASE-indexed-expert-mlp-s4 | MoE 三函数 Stage 4：OP-D 装载跳过 + 裁决翻转批式 scan（-68~-71%）+ 微调清单 no_gain 判定 | measured |
| CASE-indexed-expert-mlp | MoE 三函数混合迁移（simd 元数据+mixed 分组 GEMM+simd 归约）：W-traffic roofline 80–98%、偶奇尾声直测、修订三范式（断言换真/缺轴/自证式）、多 Kernel 类异构集成 | measured |

### repro/ — 最小可复现代码（ED-B）

规范与索引见 [repro/README.md](repro/README.md)；批量执行 `python3 .agents/tools/repro_runner.py`（知识回归测试，ED-D）。

## 5. front-matter schema（K-1）

`id`（稳定 ID）/ `kind`（pattern / trap / case / constant）/ `family`（算子族）/ `apis` / `dtype` / `device` / `status`（verified / measured / assumption / stale / overturned）/ `origin_task`（provenance，允许失效）/ `toolchain`（tilelang commit/build + CANN/bisheng 版本 + 设备）/ `repro`（`repro/<file>.py` 或 `repro-missing`；须存在且可执行，`kb_lint.py` 校验）。

- `update` delta 可对 front-matter 字段做机械更新（status 翻转、repro 登记等）；正文保持 markdown。
- **kind: pattern（优化点）的代码证据要求**：正文含优化见解摘要（机制归因），repro 为关键代码更改最小形态——完整 before/after 对照（效应可小规模复现）或 delta 骨架（效应只在完整 kernel 规模显现，py_compile）+ 见解摘要入 repro 头部（分级断言形态见 [repro/README.md](repro/README.md)）。
- `status: assumption` 仅用于 constants 类冷启动条目（源码/文档证据 + 未实测转正）；实测转正后翻 `measured`/`verified` 并补 repro。
