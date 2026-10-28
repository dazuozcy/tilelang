---
name: tilelang-design-review
description: "对 Stage 1 产出的算子设计文档（DESIGN.md）进行风险优先的检视，生成 REVIEW.md，必须给出明确结论（通过/不通过）与不通过时的具体修改建议。含算法优化检视维度（算法调研四问的完整性与结论正确性独立复核、数学等价优化的等价性论证与收益、循环/标量计算的向量化替代完整性）。迁移任务额外检视维度 0：源算子理解（语义/算法/优化手段）、硬件耦合性判定与 NPU 重设计。触发：review 设计文档、检视设计、生成 review.md。"
---

# TileLang-Ascend 算子设计文档检视

迁移来源与阶段衔接遵循 [migration-source.md](../_shared/standards/migration-source.md)。harness 检视须读取 `source_context`，独立核对 Stage 0 转换及当前函数与 public Op 的边界。

## 1. 目标

对 Stage 1 产出的 `DESIGN.md` 进行**风险优先**的检视，生成一份 `REVIEW.md`。`REVIEW.md` 必须包含明确的 `结论: 通过` 或 `结论: 不通过`，以及不通过时的具体修改建议，供 conductor 决定是否进入 Stage 3 或回退 Stage 1 重新设计。

**迁移任务**（`DESIGN.md` 含 §0 源算子解读与迁移分析）必须额外执行**维度 0：源算子理解与迁移分析**——检视 Stage 1 是否真正读懂了源算子（语义 / 算法 / 优化手段）、硬件耦合性判定是否合理、NPU 重设计是否可行且语义保持。

**所有任务**必须执行**维度 8：算法优化分析**——检视 Stage 1 是否先完成了**算法调研**（§1.6.0 调研四问齐全：等价化简公式 / 在线算法 / 复杂度 / 硬件亲和；调研结论经独立复核成立，且选定算法实际驱动 §1.4 与 §1.6.1–§1.6.3），是否先数学等价地优化了公式（更少计算量 / 访存量，且等价性论证成立），循环 / 标量计算是否已完成向量化替代分析（替代不了的必须有充分理由，且与 §3/§6 一致），以及是否完成了向量化轴与核内数据布局决策（候选矩阵 + 量化取舍；I/O layout 是契约、核内布局是设计变量，弃选方案必须有量化理由）。

---

## 2. 输入

| 字段 | 说明 |
|------|------|
| `design_md_path` | 待检视的 `DESIGN.md` 路径（由 conductor 传入） |
| `source_op_path` | **迁移任务必传**：源算子文件路径，用于核对 §0 解读是否与源码一致 |
| 算子目录 | `examples/{project}/{op}/`，用于核对同类实现引用是否真实存在 |

检视前必须用 Read 完整读取 `DESIGN.md`；迁移任务还必须用 Read 读取 `source_op_path` 指向的源算子代码；并用 Glob 核对 `DESIGN.md` 中引用的 `examples/ascend/` 等文件是否真实存在。

---

## 3. 检视维度（风险优先，按顺序）

> 优先识别会直接导致 Stage 3 编译/运行/精度失败的**阻塞级**问题，其次识别**建议级**问题。迁移任务检视维度 0–8（共 9 项），非迁移任务检视维度 1–8（共 8 项，维度 0 标记 n/a 不输出）。

### 维度 0：源算子理解与迁移分析（阻塞级，仅迁移任务）

> 检视方式：**亲自 Read 源算子代码**，与 `DESIGN.md` §0 逐项核对。不读源码的检视无效。

| 检查项 | 通过标准 | 不通过示例 |
|--------|----------|-----------|
| 语义理解正确性 | §0.1 的数学语义、规约语义（累加顺序）、dtype 语义、边界语义与源码实际行为一致；§0.2 I/O（含输出 shape / 转置布局）与源码签名一致 | 源码 fp32 累加，§0.1 写成 fp16 计算；输出 shape 与源码不一致 |
| 算法解读完整性 | §0.3 计算步骤分解覆盖源码全部计算语句（无遗漏步骤、无臆造步骤）；数据流图覆盖全部 buffer；host 侧逻辑已列出 | 源码 host 侧有 im2col，§0.3 未提及 |
| 优化手段覆盖完整性 | §0.4 列出源码中的主要优化手段且逐项标注硬件依赖；未把实现层优化误判为算法层（或反之） | 源码含 warp shuffle 归约 / T.Pipelined 流水，§0.4 未识别 |
| 耦合性判定合理性 | §0.5 每个条目有四态处置（保留/等价替换/重新设计/舍弃）+ 依据；处置与 [backend-glossary.md](../_shared/standards/backend-glossary.md) §1 概念迁移锚（TMA→`T.copy`、cp.async→`T.Pipelined`、shared→UB、warp shuffle→reduce intrinsic、cluster→无）及 ascend-constraints 不矛盾；舍弃项写明理由 | warp shuffle 判"保留"；源码多维 grid 无一维展开方案；舍弃优化项无理由 |
| NPU 重设计可行性 | §0.6 每个重设计项有「源方案 → NPU 新算法 → 语义保持论证」三要素；新算法满足 Ascend 硬件约束（一维 grid `T.Kernel(N)`、内存层级 alloc API、C0 分形对齐、容量〔`CONST-capacity-a5` 按 90% 折扣〕） | 重设计仍用多维 grid（Ascend 仅支持一维）；重设计未论证累加顺序变化对精度的影响 |
| 设计与决策一致性 | §1–§7（概述/API 映射/内存规划/Tiling/循环/同步）描述的是 §0.5/§0.6 决策后的 NPU 算法；无"§0 判重设计、§3 照抄源方案"的脱节 | §0.5 判定舍弃 SMEM swizzle，§3 仍保留布局重排步骤 |
| golden 独立性 | §8.1 golden 以 §0.1 源算子语义为依据实现（优先移植源仓参考实现），未复刻 §0.6 的 NPU 算法 | golden 直接翻译 NPU 数据流，精度对比沦为自证 |

### 维度 1：API 可行性（阻塞级）

| 检查项 | 通过标准 | 不通过示例 |
|--------|----------|-----------|
| API 存在性 | DESIGN.md §3.2 列出的每条 Ascend 方言 API 能在 `examples/ascend/`、`testing/ascend/` 或 `tilelang/ascend/language/` 源码中找到使用佐证（AGENTS.md Docs Auto Routing：API 疑问先枚举 `tilelang/ascend/language/` 模块，再下"不支持"结论） | 设计用了 `T.flash_attention` 但方言无此 API 且无导出佐证 |
| 编程模式匹配 | simt/simd/mixed 与 API 一致：默认依赖 AutoSchedule 自动同步；仅 `auto_schedule: no`（完全手动）才写 `T.ascend_set_flag`/`T.ascend_wait_flag`/`T.ascend_pipe_barrier`（须配对 + 注释 + `TL_ENABLE_AUTO_SCHEDULE: False`） | 声明 `auto_schedule: yes` 却混写手动 flag；mixed 核未用 `T.MixedKernel(sids=)` 控制 sid 分工 |
| SimtVF 优先 | 新设计默认 `T.SimtVF(threads=N)` + `T.Parallel`（编译器自动向量化/调度）；仅性能热点降级 `T.SimdVF()` + `T.simd.*` 并注明理由；GEMM 主导算子选 mixed 模式（权威：`tilelang-ascend` skill §2/§3/§8） | 整段 SimdVF 无热点论证；Cube 路径手写 NZ 分形搬运 |

### 维度 2：内存层级规划（阻塞级；可机检子集：`design_calc_check.py` ub/l1/l0c_budget 复算——VP-2026-0009 混合字节口径类错误的机械拦截位）

| 检查项 | 通过标准 |
|--------|----------|
| 搬运路径完整 | §4.4 搬运路径在 `tilelang-ascend` skill §6 支持集内（GM↔L1/UB、L1→L0A/L0B、L0C→UB〔`T.dual_copy`〕/GM 等 `T.copy`/`T.dual_copy` 路径）；ND2NZ fractal 转换由 pass 自动插入，无手写 NZ 暂存、无未支持的越级路径 |
| UB 预算 | §4.5 中间缓冲区总和对照 pattern-library/constants.md `CONST-capacity-a5`（UB 248KB，含 `T.SimtVF` 时 216KB，status: assumption）**按 90% 折扣**复算；DESIGN.md 未披露所用条目与折扣口径即 fail |
| L1 / L0A/L0B / L0C 容量 | Cube 类算子 `block_M × block_N × sizeof(accum)` ≤ L0C 256KB、L1 驻留 ≤ 512KB、L0A/L0B ≤ 64KB（同 `CONST-capacity-a5`，assumption 条目均按 90% 折扣；AutoSchedule multibuffering 的 buffer versions 与 layout padding 须计入——`tilelang-ascend` skill §4.3） |
| Buffer 层级标注 | 每个 buffer 标注存储层级与 alloc API（`T.alloc_shared`=UB / `T.alloc_l1`=CBuf / `T.alloc_l0a/l0b/l0c`） |

### 维度 3：Tiling 策略（阻塞级；可机检子集：`design_calc_check.py` core_split 一维 grid 分核三要素算术复算）

| 检查项 | 通过标准 | 不通过示例 |
|--------|----------|-----------|
| Block 划分与 grid 映射 | §5.2 给出 block_M/block_N 具体值及理由；grid 为一维 `T.Kernel(N)`，多维分解自行计算一维映射（`row = idx // N_TILES` 等） | — |
| 对齐约束 | GEMM 类满足分形对齐（M、N 为 16 倍数，K % TILE_K_SUB == 0 且 TILE_K_SUB % C0 == 0，`CONST-gemm-dim-constraints`）；尾轴对齐满足 | — |
| 非整除处理 | GEMM 类必须含 padding+crop 或动态 block 策略；纯 Vector 若 shape 不被 block 整除也需说明 | — |
| 分核策略三要素 | §5 含分核三要素（① grid 划分〔含核数实查依据——`torch.npu.get_device_properties()` 的 `cube_core_num`/`vector_core_num`，记录查询代码与返回值；无真机时标明"惯例初值 32 blocks，assumption 待标定"〕/ ② AIC-AIV 分工〔Cube/Vector 工作落位；mixed 核写明 sid guard 划分与 dual_copy 所有权；纯 Vector 算子注明仅占 AIV〕/ ③ 同步策略声明〔`auto_schedule: yes\|no`；manual 须 set_flag/wait_flag 配对方案 + flag id 预算〔核内 8 / 跨核 16〕+ `TL_ENABLE_AUTO_SCHEDULE: False`〕）；grid 大小与 block 取值自洽、核内串行边界为静态值——**核对标准**：`.agents/skills/_shared/standards/core-split-strategy.md` §1（检视要点见其 §2.2） | grid 远超物理核数被串行调度（核启动风暴）却无 persistent 设计；核数非实查且未标 assumption；manual 同步 flag 不配对或超 flag id 预算；核内串行边界依赖动态 shape |

### 维度 4：技术约束检测（阻塞级）

| 检查项 | 通过标准 |
|--------|----------|
| 一维 grid 约束 | 多维分解必须展开为一维 `T.Kernel(N)`；`T.Kernel(N, threads=...)` 直接 fail（TypeError——线程域在 `T.SimtVF(threads=N)` 内声明，`tilelang-ascend` skill §2） |
| GEMM 分形对齐与 NT 约定 | M/N 16 倍数与 K 整除已在 Tiling 维度处理；L1-input GEMM 须 NT（`trans_A=False, trans_B=True`）、权重 GM 布局 [N, K]、`clear_accum=(kt == 0)` 首 k 步清零（`CONST-gemm-dim-constraints`；`tilelang-ascend` skill §5） |
| L0C 溢出 | 已在内存层级维度处理 |

### 维度 5：循环与同步（建议级）

| 检查项 | 通过标准 |
|--------|----------|
| 循环结构 | §6 明确 T.Parallel / T.serial / T.Pipelined / T.Persistent 选择 |
| 同步与模式匹配 | §7 同步策略与 `auto_schedule` 声明一致：默认依赖 AutoSchedule 自动插 flag/同步；手动 `T.ascend_set_flag`/`T.ascend_wait_flag` 须配对并注释；`T.Stage`/`T.Task`/`T.Pipelined` 调度提示使用位置明确（`tilelang-ascend` skill §7） |
| 尾块处理 | §6.4 说明 shape 不整除时的尾块逻辑 |

### 维度 6：验证方案（阻塞级）

| 检查项 | 通过标准 |
|--------|----------|
| Golden 函数 | §8.1 含 PyTorch 参考实现草案，golden 在 NPU 上计算（torch `device="npu"`） |
| L0 测试计划 | §8 含 L0 门槛测试计划：具体规则 shape（block 整除）、dtype、按算子类别的精度标准（atol/rtol） |
| 精度标准 | 按 dtype 分档容差：fp8→1e-1，bf16/fp16→1e-2，fp32→1e-3（累积密集型 5e-3）；用 `torch.testing.assert_close`（AGENTS.md Precision validation） |

### 维度 7：完整性与一致性（建议级）

| 检查项 | 通过标准 |
|--------|----------|
| 章节齐全 | 模板 11 章节齐全（迁移类含 §0 且 0.1–0.7 小节齐全） |
| 无占位符 | 不含 `{placeholder}`、`TODO`、`待补充` |
| 同类引用真实 | §3.5.3 引用的 `examples/ascend/`、`testing/ascend/` 路径真实存在 |
| 内部一致 | API 映射、内存规划、Tiling 三处描述相互一致，无矛盾 |

### 维度 8：算法优化分析（阻塞级，所有任务；可机检子集：`verify_equiv.py` 重跑〔等价性〕+ `design_calc_check.py` r3_metrics〔四口径存在性〕——机械结果与独立推演互补）⭐

> 检视方式：以 §1.6 为对象逐项核对，并交叉核对 §3.1 / §6 与 §1.6 的双向一致性。数学等价性由本维度独立推演核对，不轻信设计文档自述；**调研结论（§1.6.0）同样由本维度独立复核**——等价性推演、复杂度算术复算、负向断言对照设计 skill 的 algorithm-research.md 参考表复核。**弃选/否决类论断（API 不支持 / 代价高 / 无先例）的前提同样必须亲自查证**——打开其引用的 API 文档核对代价机制（如 repack 是核内 UB 级还是 GM 级），不得只复算数字而继承设计的前提。

| 检查项 | 通过标准 | 不通过示例 |
|--------|----------|-----------|
| §1.6 章节存在 | §1.6 含 1.6.0 算法调研、1.6.1 数学等价优化、1.6.2 向量化替代分析与 1.6.3 向量化轴与数据布局决策四个子节（无优化空间时 1.6.1 仍须写明结论与依据；单轴算子 1.6.3 须有豁免说明；轻量调研算子 1.6.0 四问仍须逐问有结论） | 全文无 §1.6；1.6 只有标题无分析；缺 1.6.0；多轴算子无 1.6.3 |
| **算法调研存在性与完整性** | §1.6.0 调研四问齐全：R1 等价化简公式候选表**含基线候选**（输入公式/源算法）且参考表命中行已评估；R2 在线算法有/无**明确结论**（"无"须有结构依据）；R3 复杂度对比表覆盖**四口径**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值，基线在表，口径写清）；R4 硬件亲和逐候选评估（单元匹配/容量/对齐/静态边界/流水融合/跨核）；调研结论含选定算法与关键依据；「无更优替代」写明调研范围；调研深度与算子复杂度匹配（单步逐元素类四问各一行即可） | 四问缺问（如有 R1 无 R2）；候选表无基线；复杂度表只比 FLOPs 无访存/遍数/缓冲；结论空白或「无更优替代」无调研范围 |
| **调研结论正确性（独立复核）** | ① 等价性：选定算法/采纳候选与原语义的等价性独立推演成立（与 1.6.1 同口径，容差内等价须评估 fp16/bf16 舍入影响与数值稳定性，如 E[x²]−E[x]² 单遍式的 cancellation 风险）；② 复杂度算术独立复算（FLOPs/Bytes/遍数/缓冲，与设计数字一致或差异可解释——重点核对中间缓冲是否漏计 GM 往返）；③ 亲和性判断与 ascend-constraints、pattern-library 实测条目不矛盾，负向淘汰有佐证；④ **负向断言（"无在线变体/无化简公式/无更低复杂度算法"）主动复核**：对照 `tilelang-op-design` skill references/algorithm-research.md §5 参考表命中行 + 源码证据（迁移任务源算法已用 online 手段却断言"无在线变体" → 矛盾）+ 同类 examples/pattern-library §4 案例，发现明显漏掉的已知候选（如归约类漏 Welford/分块合并、softmax 族漏 online 变体、attention 漏 flash 结构）→ 本行 fail；⑤ R3 并列候选（判定裕度小）不得纸面拍板，须指向 §1.6.3 布局分析或实验裁决 | 等价性推演不成立；复杂度漏算中间缓冲访存；"无在线变体"与参考表/源码证据矛盾；已知在线/化简候选被漏掉；并列候选无裁决路径直接拍板 |
| **调研→设计一致性** | §1.6.0 选定算法实际驱动 §1.4 算法描述、§3.1 公式拆解、§6 循环结构（与 1.6.1–1.6.3 同口径交叉核对）；迁移任务：§0.5/§0.6 处置与调研选族一致，源算法被替换时 §0.4 优化手段的意图承接有逐项说明（静默丢弃 → fail） | §1.4/§3.1 描述的算法与 §1.6.0 选定不符（调研选 A 设计做 B）；调研选定单遍算法但 §6 仍是两遍扫描结构；迁移任务源算法优化手段被替换后无承接说明 |
| 数学等价优化完整性 | §1.6.1 对**§1.6.0 选定算法**做了系统优化审视（恒等变形降代价 / 稳定化 / 公共子表达式消除 / 算子降代换 / 访存削减 / 归约合并等），采纳项有四要素「原式 → 优化后公式 → 等价性论证 → 收益量化」；无优化空间有结论与依据 | 明显可优化项未审视（如逐元素除法未考虑乘倒数）；收益栏空白 |
| 等价性论证正确性 | 逐项独立推演：恒等式数学成立；容差内等价须评估 fp16/bf16 舍入影响且与 §8.2 精度标准一致 | 变形实际改变语义（如改动累加顺序/精度而无论证）；等价论证为空话（"显然等价"） |
| 向量化替代完整性 | §1.6.2 覆盖方案中**全部**循环 / 标量计算点（与 §3.3 伪代码、§6 循环结构交叉核对，无遗漏）；能替代的均已替代或说明取舍 | §6 仍存在逐元素标量循环而 §1.6.2 未涵盖该点；标量累加未考虑向量归约 |
| 不可替代理由充分性 | 不可替代项逐项给出充分且具体到本算子的理由（block 索引 / host 元数据 / tile 级顺序依赖 / 动态边界 / API 缺失佐证）；"实现简单"、"照源码写"不构成理由 | 理由栏空白或泛泛而谈；存在向量 API 可替代却未替代且无说明 |
| 替代 API 可行性 | 替代方案 API 能在 `examples/ascend/`、`docs/programming_guides/`、`docs/tools/` 或 `tilelang/ascend/language/` 源码找到佐证（与维度 1 核对口径一致）；向量替代优先 SimtVF（`T.Parallel` 自动向量化），降级 SimdVF 须有热点理由 | 使用不存在的向量 API；无佐证即断言 API 不可用 |
| **向量化轴与布局决策完整性** | §1.6.3 候选矩阵**按算子类别从设计 skill 的类别候选清单枚举、类别内必选候选齐全**（逐元素/广播类：广播轴×向量轴对齐；窗口/池化类：「I/O 原生布局+最内连续轴」与「核内重排布局+高整除性轴（如 C 轴）」两类必选；规约类：水平归约 vs 垂直扫描两种 lane 映射（与"选哪条轴"正交，须分别枚举）+两遍/online 交互；卷积类：直接跨步窗口/UB im2col/implicit GEMM；Cube/mixed：ND2NZ 自动转换（InsertNd2Nz pass）+ L0C fractal 布局 vs 显式转置/epilogue 归属侧；gather/scatter 类：连续批量 gather 粒度与向量轴对齐）；逐候选有量化评分（整除性/尾 lane 浪费/累加链形态/repack 代价/UB 影响）；**弃选方案有量化理由，禁止"未考虑"或仅"I/O 是 NCHW 所以内部也 NCHW"式论证**（I/O layout 是契约、核内布局是设计变量）；迁移任务对照了源码隐式轴选择（GPU 轴是输入非结论）；**弃选论证前提核对：每个弃选候选的 repack/代价机制前提必须与所引 API 文档亲自核对一致（转置/重排语义见 `tilelang-ascend` skill §6.1——transpose 为路径特定：GM→L1 选 dn2nz 并反转逻辑源/目的轴、L1→L0 为请求与 fractal-major 失配的 XOR，非任意路径的通用转置），并与 pattern-library（`tilelang-op-optimize` skill `references/pattern-library/`：layout/attention.md、traps-*.md、constants.md）已实测条目对照——负向论断（不支持/代价高/无先例）无 `docs/`、`testing/ascend/`、`examples/ascend/`、`.agents/skills/tilelang-ascend/` 或 pattern-library 实测佐证，或与文档/实测矛盾 → 本行直接 fail** | 多轴算子无候选矩阵；类别内必选候选缺失（如规约类只评估了轴选择未评估水平/垂直 lane 映射）；只有单一候选无豁免论证；弃选理由为空或非量化（"重排有开销"无数量级）；把 I/O 契约当作内部布局依据；把核内重排写成 GM 级 transpose 等与 API 文档矛盾的代价机制；负向论断与 pattern-library 实测条目矛盾或无任何佐证 |
| **实验裁决模式完整性**（判定裕度依赖未实证常数时） | §1.6.3 判定若依赖未文档化/未实证常数（吞吐比/跨步访问代价/转置效率/水平归约指令效率/gather 逐元素代价等，含标注「未文档化假设」的量；**pattern-library constants.md/layout.md/traps-*.md 已实测的量不算未实证——先核对设计是否漏查了 pattern-library**），须有**主选 + 备选 + 实验裁决计划**三件套：备选方案结构完整可实现（buffer 形状/UB 预算/循环结构/转置链位置/dtype 路径/分核三要素按新任务粒度重算，达到 Stage 4 可直接实现深度——仅有弃选论证不达标）；裁决计划含代表 shape、测量指标与未知常数实测/反解方法、**明确判定阈值**、回写路径；shape 特化工厂算子须评估按 shape 分派两方案的可行性 | 纸面单选但裕度落在未知量不确定区间内（判定依赖未实测常数却无备选）；代价类论断未查 pattern-library 已有实测数据；备选只有代价论证无结构设计；无判定阈值导致 A/B 无法收敛；未评估按 shape 分派可行性 |
| 布局决策与实现一致 | §1.6.3 选定的布局/轴与 §3.3 伪代码、§4 内存规划（buffer 形状与 UB 预算）、§6 循环结构三方一致：累加循环内层向量维 = 选定轴，buffer 形状 = 选定布局；重排路径（融合转置链/host permute）在 §3.3 有对应实现 | §3.3/§6 的 buffer 布局与 1.6.3 决策矛盾；§4 UB 预算未按重排布局计算 |
| 与下游章节一致 | §3.1 公式拆解以 §1.6.1 优化后公式为输入；§6 循环结构与 §1.6.2 结论一致（判定向量化的点不得再出现对应逐元素标量循环） | §3.1 仍用优化前公式；§6 出现 §1.6.2 未论证的标量循环 |

---

## 4. 工作流程
### Phase 1：读取与核对

1. Read `DESIGN.md` 全文。
2. **迁移任务**（`DESIGN.md` 含 §0）：Read `source_op_path` 指向的源算子代码全文；§0 与源码不一致的项在维度 0 中逐条记录证据（源码行/语句 + DESIGN.md 章节）。
3. Glob 核对 §3.5.3 引用的 `examples/ascend/`、`testing/ascend/` 路径是否存在。
4. 必要时 Grep `tilelang/ascend/language/`（方言）与 `tilelang/language/`（common surface）确认 API 是否有导出佐证（仅静态文本核对，不执行）。
5. **弃选论证查证（强制）**：对 §1.6.1 否决项与 §1.6.3 弃选行涉及的每个 API——设计已附 `docs/` 佐证的，亲自打开该文档核对限制条款与代价机制；未附佐证的，按 AGENTS.md Docs Auto Routing 顺序 Glob/Grep 检索该 API：`.agents/skills/tilelang-ascend/`（SKILL.md + references/）→ `docs/programming_guides/`、`docs/tools/` → `tilelang/ascend/` 源码（含 `tilelang/ascend/language/` 各模块：`simd.py`、`copy_op.py`、`gemm_op.py`、`mode.py`、`schedule_hint.py`、`tile_schedule.py`）→ `examples/ascend/`、`testing/ascend/`。检索不到且设计声称"不存在"的，在维度 8 记录查证过程；检索得到但与设计论断矛盾的，维度 8 fail。**另须对照 pattern-library**（`tilelang-op-optimize` skill `references/pattern-library/`：layout/elementwise/attention.md 模式与实测代价、traps-*.md 陷阱、constants.md 硬件常数）：设计的代价类负向论断与实测条目矛盾（如"转置慢"vs 实测 µs 级、"跨步向量化无解"vs 已验证换轴模式）→ 维度 8 fail；代价类论断未引用 pattern-library 已有实测数据的，要求补证（该量已有实测则不再是"未实证假设"）。
6. **算法调研独立复核（强制）**：Read `tilelang-op-design` skill 的 [references/algorithm-candidates.md](../tilelang-op-design/references/algorithm-candidates.md)（R1/R2 候选表本体，条目含 `known_impl`/`kb_links` 实测指针），按算子族命中行核对 §1.6.0——① 候选覆盖（命中行候选是否全部评估、基线是否在表）；② 负向断言（"无在线变体/无化简公式/无更低复杂度算法"）逐条复核（对照候选表、源码证据、同类案例）；③ 复杂度表独立复算（重点：中间缓冲的 GM 往返是否漏计、口径是否自洽）；④ 亲和性淘汰依据核对（与弃选论证查证同口径；容量/带宽淘汰引用 constants.md 条目）；⑤ 互联网来源核对（若 §1.6.0 引用了互联网候选）：来源记录（URL/论文/仓库 + 访问日期）齐全，「有候选无来源记录」按证据缺失处理（同弃选论证查证口径）；不要求重新访问验证内容真实性。复核发现矛盾或漏候选 → 维度 8 fail 并附证据（候选表条目 / 源码语句 / 案例路径）。
7. **机械复核（D-1/D-5，强制——可机检部分不由 LLM 复核替代）**：
   - **重跑等价性验证**（`DESIGN.md` §1.6.1 含采纳优化项时）：`python examples/{project}/{op}/verify_equiv.py`——执行结果须与 §1.6.1 内嵌结果表一致且全部 `EQUIV_PASS`；脚本缺失、执行失败或与内嵌表不一致 → 维度 8 fail（等价性机器验证失效）；
   - **算术复算**：`python3 .agents/tools/design_calc_check.py --design <path>`——逐项消费其 JSON 输出：`ub/l1/l0c_budget` fail → 维度 2 fail（VP-2026-0009 混合字节口径类错误由此拦截）；`core_split` fail → 维度 3 fail；`r3_metrics` fail → 维度 8 fail；`skip` 项（无法机械解析）转为人工复核并在维度内注明"机械跳过，人工已核"；
   - 两项机械结果摘要（含 skip 项）附入 REVIEW.md「机械复核」段——与维度 8 的"独立推演"互补而非替代。

### Phase 2：逐维度检视
按 §3 的维度逐项检查（迁移任务 0–8，非迁移任务 1–8），每项标记 `pass / warn / fail` 并记录证据（DESIGN.md 章节号 + 源码证据 + 引用文件）。

### Phase 3：结论判定

| 条件 | 结论 |
|------|------|
| 所有**阻塞级**维度均为 pass（warn 不阻塞） | `结论: 通过` |
| 任一**阻塞级**维度为 fail | `结论: 不通过`，必须给出具体修改建议 |

### Phase 4：生成 REVIEW.md
按 §5 模板写入 `examples/{project}/{op}/REVIEW.md`。

### Phase 5：任务复盘（Retrospective，自进化钩子）⭐

REVIEW.md 结论给出后（通过与不通过均写），向 `examples/{project}/{op}/RETROSPECTIVE.md` **追加**复盘章节（先 Read 既有内容，整文件写回，只追加不覆盖历史章节）：

- 章节模板与字段规范（canonical）：`tilelang-skill-evolution` skill 的 [references/retrospective-schema.md](../tilelang-skill-evolution/references/retrospective-schema.md)。两个标准表（Skill Flow Issues / Value Point Proposals）+ Transferable Lessons 小节。
- 复盘对象是**检视流程自身**，不是设计内容（设计问题已随 REVIEW.md 流转给 Stage 1，不得重复抄入复盘）：
  - 本次检视中发现自己**无法核验**的点（如某 API 文档缺失、某代价机制无实测数据可对照、pattern-library 未覆盖的关键常数）→ Skill Flow Issues；
  - 现有维度表无法覆盖的**检查项缺口**（如新的弃选论证错误形态、新的迁移源码陷阱）→ Value Point Proposals（P/R 类，target_doc 指向本 skill）；
  - 检视过程中亲手查证到的新 API 行为/代价实证（打开文档核对时发现的限制条款与实测矛盾）→ D 类（须带三件套）。
- 质量红线：无则如实写 `none`；单任务偶然现象不得写成通用规则。

---

## 5. REVIEW.md 模板

```markdown
# {op} 设计文档检视报告

## 检视结论

结论: 通过
<!-- 或：结论: 不通过 -->

## 检视元信息
- 文档: examples/{project}/{op}/DESIGN.md
- 任务类型: 迁移 / 新算子开发
- 源算子: {source_op_path，仅迁移任务}
- 检视维度: {迁移任务：9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）；非迁移任务：8 项}
- 阻塞级问题数: {N}
- 建议级问题数: {N}

## 机械复核（D-1/D-5）

- verify_equiv 重跑: {PASS/EQUIV_FAIL/脚本缺失/不适用（无采纳优化项）——执行结果与 §1.6.1 内嵌表一致性结论}
- design_calc_check: {逐项粘贴 JSON checks 摘要（ub/l1/l0c_budget、core_split、r3_metrics 各 pass/fail/skip + 一句 detail；skip 项注明人工复核结论）}

## 检视详情

### 0. 源算子理解与迁移分析 — pass / warn / fail / n/a
{仅迁移任务输出本节；n/a 用于非迁移任务。说明与证据须包含：语义核对结论、算法步骤覆盖核对结论、优化手段识别核对结论、耦合性判定抽检结论、重设计可行性结论、§1–§7 与迁移决策一致性结论、golden 独立性结论}

### 1. API 可行性 — pass / warn / fail
{说明与证据}

### 2. 内存层级规划 — pass / warn / fail
{说明与证据}

### 3. Tiling 策略 — pass / warn / fail
{说明与证据}

### 4. 技术约束检测 — pass / warn / fail
{说明与证据}

### 5. 循环与同步 — pass / warn / fail
{说明与证据}

### 6. 验证方案 — pass / warn / fail
{说明与证据}

### 7. 完整性与一致性 — pass / warn / fail
{说明与证据}

### 8. 算法优化分析 — pass / warn / fail
{说明与证据须包含：§1.6 章节存在性、算法调研存在性与完整性（四问覆盖、基线候选、复杂度四口径核对结论）、调研结论正确性独立复核结论（等价性推演、复杂度复算、负向断言对照参考表复核、并列候选裁决路径）、调研→设计一致性（§1.4/§3.1/§6 与选定算法、迁移任务 §0 处置一致性与优化意图承接）、数学等价优化完整性（四要素逐项核对）、等价性论证独立推演结论、向量化替代覆盖完整性（与 §3.3/§6 交叉核对）、不可替代理由充分性结论、替代 API 佐证结论、向量化轴与布局决策完整性（候选矩阵/量化评分/弃选理由核对结论）、布局决策与 §3.3/§4/§6 一致性结论、与 §3.1/§6 一致性结论}

## 检视问题列表
<!-- 仅当「结论: 不通过」时出现本章节；通过时删除本章节 -->

### 问题 1: {问题标题}
- 严重程度: 阻塞 / 建议
- 位置: DESIGN.md §{X.Y}
- 问题描述: {具体描述；迁移类问题附源码证据（源码语句/行）}
- 修改建议: {可执行的修改建议，供 Stage 1 revision 使用}

### 问题 2: ...
```

---

## 6. 输出规则

1. `REVIEW.md` 必须写入 `examples/{project}/{op}/REVIEW.md`。
2. **结论行必须是字面量** `结论: 通过` 或 `结论: 不通过`（conductor 据此判定路由），不得用"基本通过"、"建议通过"等模糊表述。
3. 不通过时，每个阻塞级问题必须给出**可执行的修改建议**，作为 Stage 1 `revision` 模式的 `design_error_summary` 输入。
4. 通过时不得出现"检视问题列表"章节。
5. 不得修改 `DESIGN.md`。

---

## 7. 错误处理

| 场景 | 处理 |
|------|------|
| `DESIGN.md` 不存在或为空 | 结论: 不通过，问题：文档缺失 |
| 迁移任务但 `source_op_path` 缺失 / 源码不可读 | 本 Agent 交付失败，返回 fail + `source_missing`（不得在未读源码时放水通过维度 0） |
| 章节大面积缺失 | 结论: 不通过，逐项列出缺失章节 |
| 迁移任务但 `DESIGN.md` 缺 §0（或 0.1–0.7 小节不全） | 维度 0 fail + 维度 7 fail：迁移分析缺失/不完整 |
| 引用的 examples 路径不存在 | 标记 fail（维度 7），建议 Stage 1 修正引用或换真实参考 |
| API 无法在源码中找到佐证 | 标记 warn（若 examples 有用法）或 fail（若全无佐证） |
| 源码与 §0 解读存在多处不一致 | 维度 0 fail，逐条列出语义/算法/优化手段差异，作为修订输入 |
| `DESIGN.md` 缺 §1.6（或 1.6.0/1.6.1/1.6.2/1.6.3 子节缺失） | 维度 8 fail：算法优化分析缺失/不完整（阻塞级），建议 Stage 1 补做 Phase 2 后重写 §1.6 |
| `DESIGN.md` 缺 §1.6.0 或调研四问缺失 / 结论无依据 | 维度 8 fail：算法调研缺失/不完整（阻塞级），建议 Stage 1 补做 skill Phase R 后重写 §1.6.0，并同步 §1.4 与 §1.6.1–§1.6.3 |
| 调研负向断言被复核推翻（存在已知在线/化简/低复杂度候选而设计称"无"，或与源码 online 证据矛盾） | 维度 8 fail：调研结论错误（阻塞级），附参考表条目/源码证据/案例路径，建议 Stage 1 重新执行 Phase R 并评估被漏掉的候选（含等价性与复杂度重算） |
| 调研选定算法与 §1.4/§3.1/§6 脱节（调研选 A 设计做 B） | 维度 8 fail：调研→设计不一致（阻塞级），建议 Stage 1 以 §1.6.0 结论为准修订 §1.4/§3.1/§6，或回退调研结论并写明依据 |
| 优化项等价性论证不成立（独立推演矛盾） | 维度 8 fail，附推演证据，建议放弃该项优化或补正论证 |
| 循环/标量点缺向量化替代分析或保留理由不充分 | 维度 8 fail，逐点列出位置（§3.3/§6）与可尝试的替代方向 |
| 多轴算子缺向量化轴/布局候选矩阵，或弃选方案无量化理由（含把 I/O 契约当内部布局依据） | 维度 8 fail：向量化轴与布局决策缺失（阻塞级），建议 Stage 1 补做 Phase 2 第 3 项后补写 §1.6.3，并同步 §3.3/§4/§6 |
| 弃选/否决论断前提与 API 文档矛盾（如核内重排被写成 GM 级 transpose），或负向论断（不支持/代价高/无先例）无 `docs/`、`testing/`、`examples/` 佐证 | 维度 8 fail：弃选论证证据缺失/前提错误（阻塞级），附亲自打开文档核对的证据，建议 Stage 1 按设计 skill「弃选论证证据规则」重查后重写受影响候选 |
| 判定裕度依赖未实证常数却纸面单选（无备选方案结构设计与实验裁决计划），或备选仅有弃选论证、无判定阈值 | 维度 8 fail：实验裁决模式缺失（阻塞级），建议 Stage 1 按设计 skill「实验裁决模式」补齐三件套（主选+备选+裁决计划） |
