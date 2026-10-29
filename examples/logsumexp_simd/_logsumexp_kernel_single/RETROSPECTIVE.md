# RETROSPECTIVE.md — `_logsumexp_kernel_single`（logsumexp_simd 家族，SimdVF 独立变体）

> 追加式日志：每个 Stage 终态追加一个章节（schema：`.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`）。

## Stage 1 (design) Retrospective — first_design — 2026-10-17

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 长文档分段产出（4 段 scratch 拼接）导致**数值推导跨段漂移**：UB 预算初稿误写 228,864B（正确 = int(248×1024×0.9) = 228,556B）、fp32 SUB 边界误写 4.4（正确 6.98）、§0.5 判定统计行与表内 12 条目计数不符（等价替换 3→2）——三处均为**结论不变但中间值/计数错**，交付前统一重算 sweep 才清零 | `DESIGN.md` §4.5/§5.2/§0.5（修正后终稿）；修正 diff 见本 session 工作记录 | `tilelang-op-design` skill Phase 8（交付前自检）增加一步「数值一致性抽检」：预算/阈值/统计行三类数值以单一来源节（§4.5 类）重算 + 跨节 grep 同名数值 | R |
| Info-source | bf16→f32 宽化装载（UNPK_B16 + vcvt part=0）在 cast 回归中**仅有组合覆盖**（`bf16->e4m3` 用例的第一级，L64–70），无直接 bf16→f32 输出用例（直接用例仅 `half->fp32` L76–80 / param L142）——设计期只能列为风险 1（L0 ③ 首 gate），无法提前关闭 | `testing/ascend/language/test_tilelang_ascend_simdvf_cast.py` L64–80/L133–142；`DESIGN.md` §9.1 风险 1 | 在 `test_tilelang_ascend_simdvf_cast.py` param 表补 `("bf16->fp32", BF16, F32, ...)` 直接用例（一行改动，复用 L76–80 分支加 BF16 分支） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| SimdVF 行规约指令流模板：UB 驻留两遍式（max 遍 + exp/sum 遍）+ `vld(UNPK_B16)+vcvt(part=0)` 宽化 + **延迟跨 lane 归约**（逐组 `vmax/vadd` lane 累加 + 遍末单次 `vcmax/vcadd`，比自动 lower 形态 max 遍指令 −50%）+ `vexpdif` 融合（−1 op/64 元素）+ `vdupv` 广播 + `vsts ONEPT` 单点存储；统计量寄存器驻留（无 UB 往返 → 无 mem_bar） | P | `examples/logsumexp_simd/_logsumexp_kernel_single/DESIGN.md` §1.6.1 O3/O4、§3.2–3.3；先例组合：`examples/ascend/flash_attention/core.py` L94–137 + `example_per_token_cast_to_fp8.py` L117–155 | 参照 DESIGN.md §3.3 伪代码复刻到同族算子（row-max/row-sum/softmax 族） | tilelang-fork dev build（fc32f47c 后）+ CANN 9.2 + dav-3510 | pattern-library `elementwise.md`（SimdVF 行规约条目，与 D-REDUCE-SIMTVF-API 并列） |
| **SIMD 组粒度 online 归约的角点保真分歧**（新增否定性证据，SIMT 调研中不存在）：m 初始化 −inf → 首组全 −inf 行 `vexpdif(−inf,−inf)=NaN` 污染累加器（部分 −inf 行误判 NaN）；初始化 −MAX_FLOAT → 全 −inf 行输出 −inf ≠ 源语义 NaN——两种初始化各破坏一处源边界语义，驻留域内 online 微变体因此弃选（角点保真 + 指令净增双重否决） | P | `DESIGN.md` §1.6.0 R1 候选 8 / R2；对照 `pattern-library/traps-runtime.md` TRAP-NAIVE-ONLINE-MERGE（fp16 朴素合并 13.6% 违反率，measured） | 按 §1.6.0 R1 #8 公式在 torch CPU 复现两种初始化的角点行为 | torch CPU（语义推演）+ 设计期论证 | pattern-library `traps-runtime.md` TRAP-NAIVE-ONLINE-MERGE 条目增补「SIMD 寄存器域初始化两难」子项 |
| 显式 SIMD 设计的 D-1 等价性验证组织：候选链以「64 元素组分桶 + lane 内跨组累加 + 单次跨 lane 归约 + 尾组 −inf 掩码」逐要素 CPU 模拟，对照源链（N_padded 256 对齐）与 fp64 参照双锚——52 门控行（V1 ulp/V2 归约分组/V3 golden/V4 位级+NaN 模式/V5 饱和封闭）全 EQUIV_PASS | P | `examples/logsumexp_simd/_logsumexp_kernel_single/verify_equiv.py`（已执行 2 次，结果稳定）；`DESIGN.md` §1.6.1 结果表 | `PYTHONPATH=/home/z00875089/tilelang-fork python3 examples/logsumexp_simd/_logsumexp_kernel_single/verify_equiv.py`（torch CPU，~1min） | tilelang-fork dev build；torch CPU | `tilelang-op-design` skill references（等价性验证组织案例索引） |
| bf16 SimdVF 宽化装载的覆盖现状：UNPK_B16+vcvt(part=0) 对 bf16 仅有组合证据（bf16→e4m3 中间级），直接位级回归缺位——同族 bf16 设计应在 L0 首批安排 bf16 用例并预备 `get_kernel_source()` 排查路径 | C | `testing/ascend/language/test_tilelang_ascend_simdvf_cast.py` L64–80/L137；`DESIGN.md` §9.1 风险 1 | none（覆盖缺口本身待 Stage 3 实证关闭） | tilelang-fork dev build + CANN 9.2 + dav-3510 | 本条随 Stage 3 结果升级（PASS → 转正证据；FAIL → capability-gap） |
| SimdVF 变体的预算/结构差异常数：UB 预算 248KB（无 SimtVF 32KB 线程上下文扣减）→ `int(248×1024×0.9)=228,556B`（对照 SIMT `int(216×1024×0.9)=199,065B`）；VRF 流式零驻留使 SUB 封顶**仅剩 UB 单约束**（SIMT 双约束含 x_frag ≤128KB）——主要工作点 SUB 落点两模式相同（wl1=8/wl4=2），差异显现在非 pow2 N（SIMT TILE 膨胀压 SUB，SimdVF 无此效应） | D | `DESIGN.md` §4.5/§5.2；`pattern-library/constants.md` CONST-capacity-a5 / CONST-vrf-capacity-a5 | 按 §4.5 公式对任意 (N, dtype, block_m) 复算 SUB | tilelang-fork dev build（fc32f47c 后）+ CANN 9.2 + dav-3510 / Ascend950PR_9579 | pattern-library `constants.md`（SimdVF/SimtVF 预算口径对照注记，可并入 CONST-capacity-a5） |
| P-LAYOUT-ROW-RESIDENCY 的 per-tile 开销模型向 SimdVF 变体的映射：persistent 批粒度 = SUB 行（扁平化后无 tile 层），修订规则「pow2 SUB 中取 makespan ≤ min+1 的最大值」在 SIMD 形态下同样适用（wl1 甜点 SUB=4 → block_m=4；wl4 不敏感）——Stage 4 扫 {4,8,16} 裁决 | C | `DESIGN.md` §5.2 Stage 4 校准指引 / §8.3；`pattern-library/layout.md` P-LAYOUT-ROW-RESIDENCY（measured，SIMT 侧 +50.9% 实测） | Stage 4 do_bench(msprof) block_m 扫描 | 同上 | 随 Stage 4 实测回写 P-LAYOUT-ROW-RESIDENCY（SimdVF 侧样本） |

### Transferable Lessons（可迁移教训）

- **预算数值单一来源 + 交付前重算**：分段撰写 DESIGN 时，所有容量/阈值数值必须从 §4.5 类单一来源节派生，交付前对全文同名数值做 grep + 重算 sweep（本次抓出 3 处中间值错误，结论均幸免）。
- **SimdVF 无线程域的连带简化**：无 `threads=`（连内部推导也无）、无 `asc_syncthreads` 行、无 VRF fragment 约束（SUB 封顶仅 UB 单约束）、统计量寄存器驻留免 `mem_bar`——从 SimtVF 设计移植结构时这四项是系统性差异点，逐项核对可避免「惯性照抄兄弟变体」。
- **掩码统一 fp32 域**：尾组/边界掩码用 `vsel(c, vdup(−inf,f32), update_mask(k,32))` 在 fp32 域做（deepgemm transform_sf 同构先例）；16-bit 域掩码未验证且 bf16 −inf 表示需另证——「实现简单」不构成换域理由。
- **`vexpdif` fp32-only 是强校验**（simd.py L1182–1195 非 fp32 显式 TypeError）：16-bit 装载必须先 `vcvt` 提升；max-gap 放大后的相对误差上界（|x−m| ≤ 11.1 → ≤6.6e-7）可作为融合指令等价性论证的通用范式。
- **读未初始化 pad 列后掩码**是 repo 内已验证形态（vld 后 vsel，deepgemm 先例），但需预备语义检查告警的 fallback（`T.copy` 列向 `pad_value=−inf` 预填——skill §6.1 该参数正是列向行尾语义）。
- **迁移双变体语义一致性是硬约束**：SIMT 兄弟变体保持源 ±inf 角点语义（NaN），SIMD 变体若「对齐 torch」将制造同族分裂——迁移任务中多变体共享源语义时，角点行为变更须算子族级统一处理，不在单变体内擅自变更。

## Stage 2 (design-review) Retrospective — review — 2026-10-17

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-tooling | `design_calc_check.py` `ub_budget` 检查产生**解析假阳性 fail**：① 层级桶切换正则按 `\*\*(UB\|L1\|...)` 标记匹配，"**VRF 预算**"标题不命中任何层级标记 → §4.5 VRF 对照表两行（"对照 SIMT 兄弟变体 x_frag ≤128KB / 128KB"）被误计入 UB 桶，2×128KB 即占假阳性总量 256/258.7KB；② SIZE_RE 不识别千分位逗号，"131,072B" 被解析为 72B（真实 x_ub 项几乎未计入）。人工复算真实 UB=139,280B ≤ 228,556B（60.9%）通过 | `.agents/tools/design_calc_check.py` 源码（已读定位）；`REVIEW.md` 机械复核段；`DESIGN.md` §4.5 | 工具增补：层级桶正则加入 `VRF` 关键词或显式排除"对照/兄弟变体"行；SIZE_RE 支持千分位逗号（`[\d,]+`→去逗号）；docstring 已自述 "advisory... NOT a gate"，检视侧应在 skill 中固化"机械 fail 须读工具源码归因后再裁定"流程 | R |
| Review-tooling | `design_calc_check.py` `core_split` 检查产生**模型不适用假阳性 fail**：工具按非持久 grid 模型（grid=输出 tile 数）复算 `ceil(1024/16)×ceil(300/1)=19200 vs declared grid 56`，而本设计为 persistent 形态——`T.Kernel(56)`（56=vector_core_num 实查）+ `T.Persistent([num_batches],56,...)`，grid=物理核数恰为 core-split-strategy §1 ① 合规形态；且工具解析数字跨 workload 混杂（M=1024 取自 wl1、N=300 取自非对齐示例、bn 缺省 1） | `.agents/tools/design_calc_check.py` 源码；`REVIEW.md` 机械复核段 + 维度 3；`core-split-strategy.md` §1/§2.2 | 工具增补：检测 `T.Persistent` 存在时切换 persistent grid 模型（grid=核数，任务域由调度器分派），或对此类设计输出 skip+n/a 而非 fail | R |
| Review-tooling | `design_calc_check.py` `l0c_budget` 检查产生**否定语境假阳性 warn**：CUBE_PATH_RE 被 §5.1 否定句"无 `T.gemm`/L1/L0 段"触发（正则无否定语境识别），16x4 为无关文本数字；本设计纯 Vector 无 Cube 段，L0C n/a | `.agents/tools/design_calc_check.py` 源码；`REVIEW.md` 机械复核段 | 工具增补：CUBE_PATH_RE 匹配前过滤否定前缀（"无/不含/没有"），或要求 `T.gemm`/`T.alloc_l0c` 正向 API 命中才触发 | R |
| Review-flow | 维度 8 负向断言复核若**继承设计文档的行号引用**效率低且不可靠——设计引"L34–49"实际 log2 注册在 L51（主张不受影响但行号偏移）；改为对每条负向断言直接 `grep` 源码自验证（如 `grep exp2\|tanh tilelang/ascend/language/simd.py` 零命中）更快且证据独立 | `REVIEW.md` 维度 8 负向断言逐条亲证段；`tilelang/ascend/language/simd.py`、`intrin_rule_ascend.cc` L45–52 | `tilelang-design-review` skill 维度 8 检视清单增补一条：「负向断言（API 不存在/不支持/无先例）须 grep 源码自验证，不继承设计行号引用；行号偏移记建议级、主张错误记阻塞级」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `design_calc_check.py` 三类解析假阳性模式清单（ub_budget 层级桶误计+千分位逗号 / core_split 非持久模型不适用 persistent / l0c_budget 否定语境误触发）+ 归因方法（读工具源码定位正则→人工复算真实值→对照 docstring "advisory NOT a gate" 裁定）——机械 fail ≠ 设计 fail，检视侧须源码级归因后再裁定，不可直接采信工具 fail 判设计不通过 | R | `.agents/tools/design_calc_check.py` 源码；`REVIEW.md` 机械复核段（三项假阳性逐条归因+人工复算值） | 对任意含 VRF 对照表/persistent grid/纯 Vector 的 DESIGN.md 运行工具即可复现假阳性 | tilelang-fork dev build（fc32f47c 后） | `tilelang-design-review` skill 机械复核步骤增补「假阳性归因清单」参考表；或 `.agents/tools/design_calc_check.py` 直接修复三处正则/模型 |
| 迁移设计检视维度 0 的**源码逐项核对方法**：亲自 Read 源算子全文（logsumexp.py 557 行 + _primitives.py 445 行）→ 按 §0.1 语义要素/§0.3 步骤表/§0.4 优化手段/§0.5 耦合性处置/§0.6 重设计三要素逐行映射源码语句（无遗漏无臆造）→ 边界语义（±inf/NaN/masked 载入）独立推演核对 → golden 独立性（源仓 ref_program 直移植 vs NPU 算法翻译）实文件比对。不读源码的维度 0 检视无效 | P | `REVIEW.md` 维度 0（七检查项各附源码行号证据）；源算子 `/home/z00875089/TileOPs-fork/tileops/kernels/reduction/logsumexp.py` + `_primitives.py`（已全文读取） | 对任意迁移 DESIGN.md 按 §0 小节逐项 Read 源码核对 | tilelang-fork dev build + TileOPs-fork 源仓 | `tilelang-design-review` skill 维度 0 检视清单（已含"亲自 Read 源算子"要求，本条为方法细化：逐行映射+边界推演+golden 实文件比对三步骤） |
| 维度 8 等价性论证的**独立推演核对范式**：不轻信设计自述——O2 幂元中性推演、O3 融合舍入最坏界独立计算（δlnS ≤ N·max(e^d·\|d\|)·2⁻²⁴ ≈ 7.2e-4 ≪ 容差）、O4 归约重排树深级差、O6 饱和上界复算（65504+ln(32512)=65514.39<65520）逐条独立推演；复杂度算术独立复算（D-2 流量/计算/special 占比/UB 重读四项）；并列候选裁决路径核对（判定裕度是否依赖未实证常数）。机器验证（verify_equiv 重跑）与推演互补 | P | `REVIEW.md` 维度 8 等价性论证正确性段 + 复杂度算术独立复算段；`verify_equiv.py` 重跑 52/52 EQUIV_PASS | 对任意含 §1.6.1 优化项的 DESIGN.md 逐条独立推演+重跑 verify_equiv | tilelang-fork dev build；torch CPU（verify_equiv） | `tilelang-design-review` skill 维度 8 检视清单（已含"独立推演"要求，本条为范式细化：幂等/舍入界/归约重排/饱和封闭四类推演模板 + 复杂度四口径复算 + 裁决裕度核对） |

### Transferable Lessons（可迁移教训）

- **机械 fail ≠ 设计 fail**：`design_calc_check.py` 三项 fail/warn 经读工具源码归因均为解析假阳性（层级桶正则不匹配 VRF 标题、千分位逗号误解析、非持久 grid 模型不适用 persistent、否定语境误触发 Cube 正则）。检视侧遇到机械检查 fail 时，必须读工具源码定位解析口径 → 人工复算真实值 → 对照工具 docstring 的 gate 属性（"advisory NOT a gate"）再裁定，不可直接采信工具 fail 判设计不通过。本次若跳过归因将误判阻塞级 fail。
- **负向断言自验证优于继承引用**：维度 8 复核设计文档的负向断言（"无 exp2 快档"/"vtanh 不存在"/"transpose 仅 GM→L1"）时，直接 `grep` 源码自验证（零命中即证据）比继承设计给出的行号引用更快更可靠——设计行号可能偏移（如 log2 注册实际 L51 而设计引 L34–49），但 grep 结果独立于设计文档。行号偏移记建议级，主张错误才记阻塞级。
- **维度 0 必须亲自读源码**：迁移任务的维度 0（源算子理解）检视结论必须附源码核对证据（源码语句/位置 + DESIGN.md 章节号），未读源码不得给维度 0 结论。本次亲自 Read logsumexp.py 557 行 + _primitives.py 445 行，按 §0.1–§0.6 逐行映射，边界语义（全 −inf 行→NaN、含 +inf→NaN、部分 −inf→中性）独立推演核对，golden 独立性以源仓 ref_program 直移植 vs SIMT 兄弟变体实文件同构比对确认。语义理解偏差是最危险的阻塞级问题——即使 API/内存/Tiling 全部 pass 也必须不通过。
- **通过时建议级备注嵌入维度详情而非单列问题列表**：REVIEW.md 结论为通过时不得出现"检视问题列表"章节（门禁 redundant_issue_list）。本次 6 条建议级备注（行号引用偏移 ×1、估算口径小疵 ×3、表述措辞 ×2）全部嵌入对应维度详情段（维度 0 备注 1、维度 8 备注 2–6），既满足门禁又保留可执行修改建议供 Stage 3 顺手修正。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-17

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | SimdVF 行规约累加器若写成不可变 SSA 绑定（`acc = neg_inf` 后在嵌套 `for g` 循环内 `acc = T.simd.vmax(acc, c)` 重赋），eager builder 把重赋绑定到已关闭的内层 ForFrame，外层 `m_brc = vdupv(vcmax(acc))` 读即报 `Immutable variable \`acc\` is used outside its defining region!`——正确形态是`acc = T.simd.alloc_var("float32")`（`local.var` 可变寄存器）后再重赋（走 `buffer_store` 路径）。L0 首编即命中，一次修复即绿 | `_logsumexp_kernel_single.py`（初版 L150/L160 vs 终版`acc = T.simd.alloc_var("float32")`）；`tilelang/language/eager/builder.py`L655–683（`is_var`→`buffer_store`分支）、L716–719（`binding_expired`）；先例`examples/ascend/deepgemm/kernels/transform_sf.py`L197–205（`valid_group = S.alloc_var("bool")` + 循环内重赋） | `tilelang-ascend` skill §8（SimdVF MicroAPI）或 `tilelang-ascend-error-fixer` 的语义校验失败表增补一条：SimdVF 循环携带累加器必须 `T.simd.alloc_var(dtype)` 声明为 mutable register，不可用不可变绑定重赋（报错特征 `Immutable variable ... used outside its defining region`） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **SimdVF 循环携带累加器可变态惯用法**（`T.simd.alloc_var(dtype)` + `buffer_store` 重赋）：`acc = T.simd.alloc_var("float32"); acc = vdup(...); for g: acc = T.simd.vmax(acc, c)` 是唯一合规形态；不可变 SSA 重赋会在嵌套循环后报 `Immutable variable used outside its defining region`（外层读已关闭内层帧绑定）。机制：`alloc_var` 返回 `local.var` Buffer（`is_var` 判真），`=` 重赋走 `tirx.buffer_store(..., 0)` 不改绑定，读取走 `unwrap_expr→BufferLoad[0]` | P | `_logsumexp_kernel_single.py` compute_rows 宏（`acc`/`accs` 两处）；`tilelang/language/eager/builder.py` L655–683 / L716–719；`tilelang/language/allocate.py` L103–168；先例 `transform_sf.py` L197–205 | 写一个含 `for` 嵌套循环的 SimdVF 归约，先用 `acc = vdup(...)` 再循环内 `acc = vmax(acc, c)` → 复现报错；改 `alloc_var` 即绿 | tilelang-fork dev build（fc32f47c 后）+ CANN 9.2 + dav-3510 / Ascend950PR_9579 | pattern-library `traps-runtime.md` 新增 TRAP-SIMDVF-ACC-IMMUTABLE 条目；或 `tilelang-ascend` skill §8 累加器惯用法 |
| **bf16 `vld(UNPK_B16)+vcvt(part=0)` 宽化装载实证关闭（DESIGN 风险 1）**：生成源码为 `simd_inst::vcvt<float>(simd_inst::vlds_unpack<half>(...), 0)`，fp16/bf16 全部用例 max_diff=0.000e+00（含 L0 3-dtype、L1 42 组 fp16/bf16 扫形、Boundary 特值/角点）；fp32 用例 max_diff ≤ 9.537e-07（容差 1e-3） | D | `_logsumexp_kernel_single.py`（`--show-source` 生成的 vlds_unpack<half>/vcvt/vsts_1st 语句）；`DESIGN.md` §9.1 风险 1 | `PYTHONPATH=/home/z00875089/tilelang-fork python examples/logsumexp_simd/_logsumexp_kernel_single/_logsumexp_kernel_single.py --level all`（全绿） | tilelang-fork dev build（fc32f47c 后）+ CANN 9.2 + dav-3510 / Ascend950PR_9579 | pattern-library `elementwise.md`（SimdVF 行规约条目，DESIGN §1.6.1 P 类价值的实现侧回写） |
| **SimdVF 行规约指令流端到端实证**（DESIGN §3.3 落地验证）：UB 驻留两遍式 + 延迟跨 lane 归约（`vcmax→vdupv` / `vcadd`）+ `vexpdif` 融合 + `vln` + `vsts_1st`(ONEPT_B16/B32) 全链生成正确，无 spill；`T.Persistent` 尾批 + 尾组 `vsel(c,−inf,update_mask(k_tail,32))` + `T.Kernel(56)` 一维 grid 组合在 M=1/33/57、N=1/7/300/32512 边界全部通过（含 all −inf/+inf 源语义角点、3×bitwise determinism） | C | `_logsumexp_kernel_single.py`；L0 7/7、L1 57/57、L2 4/4、Boundary 13/13 全 PASS | `--level all`（全绿，见上） | 同上 | pattern-library `elementwise.md` / `layout.md`（SimdVF 行驻留实现侧样本索引） |

### Transferable Lessons（可迁移教训）

- **SimdVF 累加器必须是 `T.simd.alloc_var(dtype)` 可变寄存器**：任何在嵌套循环内重赋的归约量（max/sum 累加）若写成普通 `=` 绑定，会在循环关闭后读时报 `Immutable variable ... used outside its defining region`。声明 `alloc_var` 后 `=` 重赋走 buffer_store、读取走 BufferLoad，才是循环携带的正确形态。同族 SimdVF 归约（row-max/row-sum/softmax）照此写。
- **bf16/fp16 宽化装载用 `vld(dist="UNPK_B16")+vcvt("float32",part=0)` 直通**：生成 `vlds_unpack<half>`+`vcvt<float>`，无 UNPK 陷阱；fp32 直用 `vld()` NORM。16-bit 单点回存用 `vsts(..., ONEPT_B16)`、fp32 用 `ONEPT_B32`（mask 分别 `pset(16/32,"PAT_VL1")`）。尾组掩码统一在 fp32 域 `vsel(c, vdup(−inf,f32), update_mask(k_tail,32))`。
- **Persistent 尾批用 TIR 尾分支 `if t < full` 而非运行时整除**：`num_batches = full + (1 if rem else 0)`、`if t < full: 整批 else: 尾批(rem 行静态切片)`——`full/rem` 均为 JIT 常数，避免 `t == full` 动态边界依赖；`T.Kernel(56)` 一维（无 `threads=`），persistent 任务域由 `T.Persistent([num_batches], 56, core_id, num_stages=2)` 分派。
