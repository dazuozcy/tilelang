# Ascend A5 性能模式库索引（pattern-library/INDEX.md）

> **本目录是 `pattern-library` 的唯一入口**（K-2 渐进披露：INDEX 必读 + `kb_search` 命中条目精读）。收录**已实测验证**的性能模式、代价数据、编译器/运行时陷阱与实验方法学规则。
>
> **冷启动说明（2026-10-16，A5 适配 P0）**：本库自 tilelang-mlir-fork（npuir/A2A3，Ascend910B2C 实测）整套复制后**内容清零**——原条目绑定 npuir 工具链与 910B2C 实测，不适用于 tilelang-fork（ascend/A5，dav-3510）。仅 `constants.md` 以 assumption 级 A5 条目起步（源码/文档证据标注，见各条目），其余主题文件保留 schema 空骨架，由 A5 任务蒸馏积累。调优手段的种子知识源：`tilelang-ascend` skill §12（profiling recipes）与 `maint/scripts/ascend_perf_regression.py`。

## 1. 维护规则（由 optimizer 任务内回写 + `tilelang-skill-evolver` 终态蒸馏共同维护）

- 每次调优任务产出的新模式/代价数据/证伪更正，由调优 Agent 在任务结束前追加到对应**主题文件**（含任务溯源与工具链版本戳）；evolver 是第二个授权写入者（覆盖各主题文件的增量条目，五种 delta：add/update/consolidate/negate/deprecate；含其他 Stage 产出的 D/C 类价值点）；两次写入互补防双写（evolver 跳过已回写条目）。
- **条目溯源（D2）**：新增条目须带 `origin_task`（来源任务 id）与工具链版本戳；**下游引用条目时须标注其 origin_task**（optimizer Phase 0 检索注入、designer 步骤 0.5、复盘引用），据此判别可信度与是否需按版本戳重验。
- **预算与生命周期**（由 evolver 维护，K-2 字节预算）：INDEX ≤12KB（承载全条目路由表；超限优先压缩"一句话"列而非删条目）；每主题文件 ≤16KB；`repro/` 每文件 ≤8KB/100 行。超限触发 consolidate（合并、压缩措辞；负面条目只可 deprecate 不可删除）；`update` 优先于 `add`（同主题已有条目时禁止新开）；**工具链变更后相关条目自动降级"待重验"**（`kb_stale_check.py` 机械检测）。
- **条目自包含（ED-A）**：关键数字、适用条件（shape/dtype/dispatch/workload 上下文）、结论必须内嵌条目正文——provenance（origin_task 与过程文件出处）允许失效，条目仍完整可用；**代码证据唯一合法形态是 `repro/` 内自包含脚本**（`repro: repro-missing` 表示待回填）。**经验需要代码时，用最少的代码表达**（ED-B）：优化点条目（kind: pattern）的代码证据 = 慢→快关键代码更改的最小形态（完整对照或 delta 骨架）+ 优化见解摘要（机制归因到 CONST-\*/TRAP-\* 条目）。
- **assumption 折扣纪律（A5 冷启动特有）**：`constants.md` 中 `status: assumption` 的条目在设计期 roofline 估算中须打折使用（容量按 90% 预算），并在 DESIGN.md 中披露所用条目与折扣；转为 `measured` 后移除折扣。假设值与实测冲突时以实测为准并 update 条目。
- 机械校验：`python3 .agents/tools/kb_lint.py`（三件套齐备、front-matter 合法、字节预算、repro 存在性与解耦、状态枚举）；检索：`python3 .agents/tools/kb_search.py "<query>"`。

## 2. 证伪协议（canonical，否定一个 API/模式前必须遵守）

1. **必须用文档合法形态测试**——否定任何 API/模式前，先按 AGENTS.md「Docs Auto Routing」查权威文档（`.agents/skills/tilelang-ascend/SKILL.md` 及其 references、`docs/programming_guides/`、`tilelang/ascend/` 源码）确认其合法参数/形式，穷举代表性写法后再下结论。
2. **一切编译器/运行时结论必须盖工具链版本戳**（tilelang commit/build 时间 + CANN/bisheng 版本 + 设备型号），工具链变更（源码修改/重编译）后**自动视为待重验**，不得直接引用旧结论。
3. **证伪更正须留痕**：推翻旧结论时在 opt_log 写明"误判根因 + 合法形态 + 新数据"。

## 3. 检索规则（调优 Agent 每轮必读）

1. **首轮必查项**：① 编程模式重估（SimtVF/SimdVF/mixed 对照 `tilelang-ascend` skill §2/§8，即使上游设计已定——设计期结论可能基于旧工具链）；② 标量执行占比诊断（msprof_detail pipe ratios：`aiv_scalar` 高占比触发换轴）；③ 陷阱条目版本戳与 origin_task 核对（工具链变更则重验，`kb_stale_check.py`）。
2. 新模式/新代价数据/证伪更正 → 任务结束前追加进对应主题文件（含溯源）。
3. 本库模式是**起点不是终点**：每算子的最优 tiling/grid 仍须实测扫描。

## 4. 条目索引（ID + 一句话 + 状态 + 所在文件）

> 完整 front-matter（family/apis/dtype/device/status/origin_task/toolchain/repro）见各条目；`kb_search.py` 按本索引与条目 facets 检索。

### constants.md — 硬件常数表（D-2，设计期 roofline 口径）

| ID | 一句话 | 状态 |
|----|--------|------|
| CONST-capacity-a5 | 片上容量：UB 248KB（SIMT 216KB）/ L1 512KB / L0C 256KB / L0A/L0B 64KB（auto_schedule.cc 源码） | assumption |
| CONST-arch-a5 | A5/950 组织：AIC:AIV=1:2 编组、sids 1/2、npu-arch dav-3510、CANN 9.x；本机实查 cube=28/vector=56 | measured |
| CONST-gemm-dim-constraints | GEMM 维度约束：M/N 16 倍数、K%TILE_K_SUB==0 且 %C0、C0=32B/elem、FRAC_N=16、NT 约定 | assumption |
| CONST-cube-throughput-model | Cube 吞吐代价模型（fp16/bf16 8192、fp8 16384、HF32 4096、fp32 512、INT8 MMAD 不支持，ops/cycle/AIC） | assumption |
| CONST-mte-bandwidth-model | MTE/FixPipe/dual_copy 带宽与延迟模型（B/cycle + base cycles，estimate_latency.h） | assumption |
| CONST-valu-model | VALU 吞吐与 VRF↔UB 带宽（51/13/8 ops/cycle、256 B/cycle/AIV）；exp/exp2 同档注记；标量 tanh 软件例程反例（1.80×） | assumption |
| CONST-vrf-capacity-a5 | VRF 容量假设 256KB（无文档化常数）+ 双口径预算规则 + spill=0 实测样本（76/48/16 reg/thread）；fragment 落 VRF 不占 UB 216KB 预算 | assumption |

### layout.md — 布局与搬运模式

| ID | 一句话 | 状态 |
|----|--------|------|
| P-LAYOUT-ROW-RESIDENCY | row-reduction 族结构判定：驻留 vs 流式 online 四步判定法 + SUB 行子批双消化 + wrapper 双变体分界 + **persistent ROWS 粒度规则（makespan+1 内最大 pow2；per-tile 开销项实测）**（双任务实测） | measured |
| P-LAYOUT-TAIL-STATIC-EXTENT | 尾块处理范式：host 静态分解 + 元编程 if + 静态 extent 切片 + BLOCK_N pow2 重划（tile 参数降级为预算上限） | measured |

### elementwise.md — 多输入搬运效率

| ID | 一句话 | 状态 |
|----|--------|------|
| D-ELEM-SIMTVF-AUTOVEC | SimtVF T.Parallel 内 tir.* 数学 intrinsic 自动向量化为 2048-bit（vexp/vdiv），fp16 可用（3.906e-3 容差内） | measured |
| D-ELEM-ACT-FORMULA-PROBE | silu/sigmoid 公式选型实测：直除 0.902× 最优、tanh 形式 1.80× 劣化（软件例程）、vexpdif 仅 SimdVF+fp32 | measured |
| D-REDUCE-SIMTVF-API | row-reduction 形态实证：reduce_max/sum → AscendAllReduce 硬件树归约（树宽=SimtVF threads，**U 曲线 256 最优**：512 改善/128 回退，6 workload 实测）；extent<threads 复制布局合法；−inf pad 幂元中性；flash 合并 chunk 划分不变；T.And 联合谓词 | measured |
| D-LOGSUMEXP-MIGRATION-FIDELITY | logsumexp 族迁移保真（双变体）：golden 直用 torch.logsumexp（fp16/bf16 位级、fp32 ≤1 ulp）；fp16-e 备选守卫余量 ~100× 但 **Stage 4 时延裁决 50–60x 慢（hexp/半精度归约软件化）实测否决**——中间链保 fp32 fragment | measured |
| D-LOGSUMEXP-SIMD-F16MAX | SimdVF 两遍结构 pass-1 在 16-bit 输入域做 max（NORM 128-lane + vmax，winner 单次 vcvt；bf16 需 EVEN/ODD 拆分——asc_reduce_max 无 bf16 重载）：位级等价，5 workload +4~+15.5%；同轮反模式（步进展开/EVEN-ODD 配对/掩码提升）实证否决 | measured |
| D-SIMD-PERSISTENT-ROWS-LSE-BATCH | SimdVF 流式变体每 tile 行数 ROWS：lse 2 字节微写批量化（mte3 64.3→6.2%），M≤56 保 ROWS=1 / 否则 makespan ≤min+1 最大 pow2 cap 8——32k +26.6% | measured |

### attention.md — attention / 混合核族

| ID | 一句话 | 状态 |
|----|--------|------|
| D-MIXED-EPILOGUE-DUALCOPY | mixed CV epilogue：dual_copy L0C→UB → SimtVF 元素段（含 T.exp）→ dual_copy UB→GM 在 dav-3510 AutoSchedule 直接支持（无需隔离/guard）；融合 vs 分离 GM −40% | measured |

### traps-compiler.md — 编译器/解析器陷阱

| ID | 一句话 | 状态 |
|----|--------|------|
| TRAP-MACRO-SCOPE-PLACEMENT | `@T.macro` 须定义在 prim_func 外层（内层 NameError）；常量闭包捕获、buffer 参数传入；体内 SimtVF/T.serial/Python if 可用 | measured |

### traps-runtime.md — 运行时/数值/语义陷阱

| ID | 一句话 | 状态 |
|----|--------|------|
| TRAP-GRID-DIVISIBILITY | grid 不整除元素总量（2^17 无因子 7 vs 28/56 核）→ 无编译错误、尾部静默漏写 12.5% 伪装精度失败 | measured |
| TRAP-NAIVE-ONLINE-MERGE | 朴素 online softmax 合并（m_c 基 lse 式）角点发散（间隙>88.7 溢出/全 −inf NaN；违反率 13.6%/18.2%）——flash 形式修复；哨兵钳位按族语义取舍（softmax 保留 / logsumexp 弃用） | measured |
| TRAP-NPU-DOUBLE-SILENT-FP32 | A5 NPU 无 double：torch `.double()` 静默降级 fp32 + warning（ToKernelNpu.cpp:40）——fp64 参照须 CPU 侧 | measured |
| TRAP-SOURCE-VS-TORCH-GOLDEN | 源算法固有角点与 torch 参考分歧（logsumexp 全 −inf 行 NaN vs torch −inf）——randn 域不触发，Boundary 按源语义对照 | measured |

### cases.md — 案例索引与参考实现集

| ID | 一句话 | 状态 |
|----|--------|------|
| CASE-pilot-silu | simt 纯 Vector 扁平 elementwise 模板（参数化工厂+整除防御） | measured |
| CASE-pilot-gemm-silu | mixed GEMM+激活 epilogue 融合案例（CV 融合 −40% GM + 探针否决 tanh） | measured |
| CASE-pilot-online-softmax | GPU→A5 迁移 row-reduction 案例（换底复核+驻留判定+flash 哨兵+S3-JIT 教训） | measured |
| CASE-logsumexp-harness | migration-harness 全链首例（Stage 0→5）：驻留+流式双变体位级保真、@T.macro 因子化、探针闭合先例空隙、Stage 5 零 attempt + msprof 基线 | measured |
| CASE-logsumexp-simd | SimdVF 双变体迁移+Stage 4 案例（SIMT 变体共存、16-bit 域 max、stats-UB 往返；32k +28.7%、SIMD 1.2–2.6x 于 SIMT final） | measured |

### repro/ — 最小可复现代码（ED-B）

规范与索引见 [repro/README.md](repro/README.md)；批量执行 `python3 .agents/tools/repro_runner.py`（知识回归测试，ED-D）。

## 5. front-matter schema（K-1）

`id`（稳定 ID，引用与命中统计锚点）/ `kind`（pattern / trap / case / constant）/ `family`（算子族）/ `apis` / `dtype` / `device` / `status`（verified / measured / assumption / stale / overturned）/ `origin_task`（provenance，允许失效，不做存在性核验）/ `toolchain`（tilelang commit/build + CANN/bisheng 版本 + 设备）/ `repro`（`repro/<file>.py` 或 `repro-missing`；repro 必须存在且可执行，`kb_lint.py` 校验）。

- `update` delta 可对 front-matter 字段做机械更新（status 翻转、repro 登记等）；正文保持 markdown。
- **kind: pattern（优化点）的代码证据要求**：正文含优化见解摘要（机制归因），repro 为关键代码更改最小形态——完整 before/after 对照（效应可小规模复现）或 delta 骨架（效应只在完整 kernel 规模显现，py_compile）+ 见解摘要入 repro 头部（分级断言形态见 [repro/README.md](repro/README.md)）。
- `status: assumption` 仅用于 constants 类冷启动条目（源码/文档证据 + 未实测转正）；实测转正后翻 `measured`/`verified` 并补 repro。
