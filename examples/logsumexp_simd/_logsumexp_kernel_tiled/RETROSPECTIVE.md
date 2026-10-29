# RETROSPECTIVE — _logsumexp_kernel_tiled（SIMD tiled 变体）

> 阶段复盘（追加式：每阶段完成后追加章节；本文档随任务推进累积，供 skill-evolver 蒸馏）。

## Stage 1：设计（2026-10-17，first_design，迁移任务）

**产出**：DESIGN.md（955 行）+ verify_equiv.py（V1–V7 全 EQUIV_PASS）+ history_version/design_budget_calc.py（容量/D-2 单源复算脚本）。一次成型，无 revision。

### 结论摘要

- **架构（B14）**：`T.Kernel(56)` + `T.Persistent([num_tiles], 56, num_stages=2)` 行块持久调度 × 任务内**行串行**（行外层 × chunk 内层两层）× per-chunk SimdVF 指令流；running stats（m_run/s_run）为行作用域 `T.simd.alloc_var`（SimdVF 块外声明、块内读写）——使在线统计量退化为每行一对寄存器，避免跨行数组索引。
- **BLOCK_N = tile_n 直保留**（SimdVF 无 fragment——SIMT 兄弟被迫 pow2 重划的约束不存在）：wl3 4 chunks（vs 兄弟 7）、(33,33000) 2（vs 3）。
- **调研四问**：R1 15 候选（2 项实测否决：fp16-e 50–60× 回退、M≪56 天花板）；R2 在线存在且源即在线（chunk 粒度选定，组粒度弃选）；R3 四口径表（选定单遍 GM/计算 ~4op+1special/驻留 x_ub+3.3KB VRF）；R4 全档 ✓。D-2：wl2 计算项 122.7K cyc/core（special 61%）vs MTE 12.0K（10.2×，计算主导）。
- **耦合性判定**：13 条目 = 保留 4 + 等价替换 3 + 重新设计 5（R1–R6）+ 舍弃 1（threads，K3）。

### 价值点（供蒸馏）

**D（实测数据）**：
- D-1 等价性：V1 fp64 位级（ulp=0，8 shape 含 63-chunk 长链与非 pow2 tile_n）/ fp32 ≤1 ulp；V6 划分扫描**首次纳入非 pow2 BLOCK_N**（16640/25600 全 ≤1 ulp）——BLOCK_N=tile_n 直保留的语义中性证据。
- D-2 roofline（assumption 常数基）：wl2 计算项 122.7K cyc/core（均值）/ 127.5K（最重核 19 行）；special_func 占计算项 61%；MTE 带宽项 12.0K（ratio 10.2×，含 base 6.6×）；UB 重读 4.68K（3.8%，26× 余量）；merge 指令占比 0.39%。复算脚本 `history_version/design_budget_calc.py` 确定性可重现——**容量/D-2 数字全部单源化**（对 single 姊妹 Stage 1「数值多源漂移」教训的直接治理）。

**P（模式方法）**：
- P-TILED-CORNER-DISCOVERY：**迁移源码语义解读须按「数据分区」枚举角点**——online/tiled 类算法的边界行为依赖分区覆盖率（首 chunk 全 −inf → NaN 是 tiled 特有：chunk 覆盖率由 tile_n 决定，single 形态与 torch 均给有限值）。两个兄弟变体的设计与测试均未覆盖该角点；本设计经「逐 chunk 递推状态推演」发现并 V5 位级锚定。方法论：对分块递推算法，角点 = {分区边界} × {递推初值} 的组合枚举，不止 {元素值} × {行}。
- P-ALLOCVAR-CROSSBLOCK：`T.simd.alloc_var` 跨 SimdVF 块驻留（块外声明、多块读写）的机制论证三件套（声明无域检查 `allocate.py` L103–168 / enclosing-scope `vf_regions.h` L50–53 / local.var 跳同步 `insert_sync.cc` L1308）+ **fallback 全先例**（stats UB 往返，flash core.py 形态，0.14% 代价）——「主选无先例但机制可证 + fallback 有先例」的两手准备模式，Stage 3 首用例即触发裁决。
- P-BUDGET-UNCOUPLING：行串行 + 单行 chunk 粒度 x_ub (1, BLOCK_N) 使 **block_m 与 UB 容量解耦**（容量式不含行数）——消除 single 变体的 SUB 封顶机制一类派生参数；派生参数越少，契约空间越干净。
- P-REJECT-BY-FIDELITY：弃选候选项时区分「性能弃选」与「语义保真弃选」——两遍重读在计算主导域（10.2× 余量）下流量代价被流水掩盖（端到端 ~0），真正的决定性依据是角点保真破坏（V5）。**性能型论证须按瓶颈域自查其端到端有效性**（本设计 D-2 行初稿曾写「+9.8% 端到端」，自查推翻——教训见 R）。

**R（流程规则）**：
- R-SEGMENTED-WRITE：大文档分段写作（6 段 × 100–300 行）+ cat 合并 + 门禁自检（占位符 grep / 章节清单 / 数值一致性扫描）——单次 Write 长文有截断风险，分段 + 合并 + 事后 grep 是可复现的安全路径。
- R-NUMERIC-SWEEP：交付前对关键数字做 grep 计数扫描（同名值跨章节引用一致性）——本次扫描发现 0 处漂移（单源脚本治理生效）；姊妹任务曾有数值多源漂移教训。
- R-D2-SELF-AUDIT：D-2 估算行中的「弃选对比」数字须与瓶颈域模型自洽（计算主导域下流量增量不直接等于端到端增量——须过一遍流水掩盖检查再落笔）。

**C（案例索引）**：
- logsumexp 家族（SIMD 双变体 + SIMT tiled）全部设计完成：`examples/logsumexp_simd/_logsumexp_kernel_single/`（86 用例 PASS）+ 本目录（Stage 3 待实现）+ `examples/TileOPs/tileops/kernels/reduction/logsumexp/`（61 用例 PASS + Stage 4）。跨变体复用资产：指令链（single 实证）、tiled 契约结构（SIMT 实证）、verify_equiv 框架（V1–V7 模式可移植）、预算复算脚本模式。

### 风险交接（Stage 3 输入）

1. **风险 1（top）**：alloc_var 跨块声明+访问无单一先例——首用例（L0 #1 最简形态）首编即裁决；fallback = stats UB 往返（flash 形态，全先例，0.14% wl2）。
2. **风险 2**：首 chunk 全 −inf → NaN 角点——Boundary ④ 必须对 `_source_lse_tiled_sim`（勿用 golden 对照误判）。
3. **风险 4**：special_func 吞吐常数粒度（element vs 指令级）未区分——Stage 4 npusim 归因标定后回写 pattern-library。
4. 若 Stage 3 首编确认 alloc_var 跨块**不可行**（fallback 落地），须按 Compiler Capability Gap Rule 记录 `.agents/evolution/capability-gaps.md`（设计期未结论为 gap——机制证据链支持可行，仅无先例）。

## Stage 2 (review) Retrospective — review — 2026-10-17

**产出**：REVIEW.md（结论: 不通过；阻塞 1 / 建议 2）+ 机械复核重跑（verify_equiv ALL EQUIV_PASS、design_calc_check JSON、design_budget_calc 单源复现、D-2 roofline 独立手工复算、核数本机复测 28/56）。检视对象：DESIGN.md（955 行，迁移任务 9 维度）。维度判定：0 fail / 1 pass / 2 warn / 3 pass / 4 pass / 5 pass / 6 fail / 7 warn / 8 warn。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | **迁移文档的 host 启发式落点无机械复核位**：本次唯一阻塞级问题（`align_up(33000,256)` 手算 33,280〔实际 33,024〕→ `compute_tile_n` 落点 16,640〔实际 32,512〕）无法被现有机械检查捕获——`design_calc_check.py` 只校验文档内已写数字的算术自洽性，`design_budget_calc.py` 以错误 tile_n 为**硬编码输入**（garbage-in-garbage-out），Stage 1 的 R-NUMERIC-SWEEP grep 扫描因错误值跨章节一致而静默通过。检视方靠「照抄源仓 `align_up`/`compute_tile_n` 逻辑独立复算全部 7 个落点」才发现 | REVIEW.md#机械复核（ub_budget skip 段）+ #问题 1；`history_version/design_budget_calc.py#L130`；源仓 `_primitives.py#L91-194` | `.agents/skills/tilelang-design-review/SKILL.md` Phase 1 增一步「迁移落点复算（强制）」：对 §0.2 每个 (M,N,dtype) 用源仓启发式函数脚本化复算 (block_m,tile_n,num_tiles)，禁止采信文档手算值；`.agents/tools/design_calc_check.py` 可增 `landing_recompute` 检查（入参：源仓函数路径 + shape 表） | R |
| Review-flow | `design_calc_check.py` 两项适配缺口：① `l0c_budget` 对**纯 Vector（无 Cube）算子误报 warn**（"block_M/block_N not multiples of 16" C0 分形对齐），须人工判定误报；② `ub_budget` 对本设计 §4.5 表列式（x_ub（N 版本）/总占用/占预算）解析失败 → skip，人工复核成本高 | REVIEW.md#机械复核（design_calc_check 表 l0c_budget / ub_budget 两行）；DESIGN.md#§4.3/§4.4（无 `T.gemm`/`T.alloc_l0c`） | `.agents/tools/design_calc_check.py`：文档无 `T.alloc_l0c`/`T.gemm` 时 `l0c_budget` 返回 skip + reason="pure-vector kernel"；`ub_budget` 增补该列式解析 | R |
| Info-source | **同源文档族的共享数值会伪装成"已交叉验证"**：错误落点从 SIMT 兄弟 DESIGN 逐字继承，两文档一致 → 一致性检查与 grep 扫描全部通过，只有回溯源仓函数才暴露。检视 skill 现无「继承链怀疑」条款 | REVIEW.md#问题 1（同源错误溯源段）；`examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp_kernel/_logsumexp_kernel_tiled_DESIGN.md#L66` vs 本 DESIGN.md#L67 | `.agents/skills/tilelang-design-review/SKILL.md` 维度 0 加注：兄弟/姊妹变体文档**不作为**落点/常数的独立证据源，须回溯源仓函数或本机实测 | R |
| Review-flow | 等价性扫描脚本的**扫描域以被验证的推导值为上界**（`bn ≤ (tn//64)*64`）→ 落点错误同时缩小验证域，形成自遮蔽（输入错 → 全部 PASS）；修正落点后生产值 32,512 未被任何档覆盖，须人工判断「可按划分不变性外推但仍须补档」 | `verify_equiv.py#L357-359`；REVIEW.md#机械复核（关键补充段）+ #问题 1 修改建议第 6 条 | `.agents/skills/tilelang-op-design/SKILL.md`（等价性机器验证段）加注：扫描域上界须独立于被验证的落点推导（取 clamp 上界或宽档枚举） | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 迁移落点复算必须脚本化照抄源仓启发式函数（`align_up`/`compute_tile_n`），禁止手算或继承兄弟文档——一次 `ceil` off-by-one 引发跨 6 章节传播并写入 L0 门槛 assert | R | REVIEW.md#问题 1（逐步复算 1–4 + 影响面清单 + 7 条修改建议）；源仓 `_primitives.py#L107/L165-192` | `python3 -c "print(((33000+255)//256)*256)"` → 33024（文档写 33280）；`compute_tile_n(1,2,33024,65024)` → 32512（文档写 16640） | tilelang-fork 工作区 2026-10-17 + TileOPs-fork 源仓（DEFAULT_ALIGNMENT=256 / MAX_SINGLE_TILE_COLS=32512）；设备 Ascend950PR_9579；本项为静态复算，未上机 | `.agents/skills/tilelang-design-review/SKILL.md`（维度 0）+ `.agents/skills/tilelang-op-design/SKILL.md`（§0.2 落点表规范） |
| 单源复算脚本只治理「转录漂移」，不治理「输入推导错误」——单源化边界须上推到派生量的源头函数（tile_n 应在脚本内复算而非硬编码常数） | R | `history_version/design_budget_calc.py#L130`（硬编码 16640）；REVIEW.md#机械复核 ub_budget 行 | 将 L130 常数改为脚本内复算 `compute_tile_n(1,2,align_up(33000,256),65024)` → 产出 32512，与文档嵌入值不符即暴露错误 | 同上 | `.agents/skills/tilelang-op-design/SKILL.md`（预算单源脚本规范）+ pattern-library（治理模式条目） |
| 检视期独立复算 D-2 roofline 的粒度口径锚点：special_func 项须按 **lane-op 粒度**折算（`instr × 64 lane ÷ 8 lane-ops/cycle`）才复现设计值 122.7K cyc/core；按指令粒度折算得 ~1.9K（差 64×）——该 64× 正是设计自披露 §9.1 风险 4（`estimate_latency.h` 未声明 op 粒度）的量化边界 | D | REVIEW.md#机械复核（D-2 独立复算段）；DESIGN.md#§1.6.0 末段 L327；`src/ascend/transform/estimate_latency.h#L58-70`（special_func_throughput=8、valu_bandwidth=256，无粒度声明） | `(1030×64/51 + 257×64/8) × 2 chunk × (1024/56 行)` = 122,588 ≈ 设计 122.7K；special 占比 2056/3351.9 = 61.3% ≈ 设计 61%；GM 项 67,108,864/(56×100) = 11,983 ≈ 12.0K | tilelang-fork 2026-10-17 当前工作区 `estimate_latency.h`；**静态复算，未上机标定**（真实粒度须 Stage 4 npusim 归因裁决） | pattern-library/constants.md（`CONST-valu-model` 补粒度口径注记）+ `.agents/skills/tilelang-op-design/references/hardware-cost-model.md` |
| 引用 pattern-library 条目支撑决策时须**全量消费其注记**：设计引 `P-LAYOUT-ROW-RESIDENCY` 支撑行驻留布局，却未对账同条目内的 BLOCK_N 甜点实测回退数据（双向偏离 16384：−21.6~−36.4% / −24~−29%），而设计取值（25,600 / 32,512）正落在回退区间 | P | REVIEW.md#问题 2；`.agents/skills/tilelang-op-optimize/references/pattern-library/layout.md`（P-LAYOUT-ROW-RESIDENCY 甜点注记）；DESIGN.md#§1.6.3（甜点/sweet/−24 关键词 0 命中） | `grep -c "甜点\|sweet" DESIGN.md` → 0 | 同上（该条目实测数据来自 SIMT 结构，不可跨结构外推——本项教训是「须对账并声明结构差异」，非「结论必错」） | `.agents/skills/tilelang-design-review/SKILL.md`（维度 8 布局决策行）+ `.agents/skills/tilelang-op-design/SKILL.md`（§1.6.3 选定/弃选论证规则） |

### Transferable Lessons（可迁移教训）

- 迁移设计的 host 启发式落点（block_m / tile_n / num_tiles）必须用**照抄源仓函数的脚本**逐 shape 复算并留档：手算 `ceil` 类算式（如 `align_up(N,256)`）极易 off-by-one，错误会沿「落点表 → 预算表 → L0 用例 → 门槛 assert」链一路传播，到 Stage 3 首跑才爆。
- 兄弟/姊妹变体文档中的共享数值**不构成独立证据**：继承链上的错误在多文档间表现为「一致」，一致性 grep 扫描必然静默通过；核对基准只能是源仓函数或本机实测。
- 单源复算脚本（全部嵌入数字由脚本产出）消除的是转录漂移；若脚本输入常数本身来自手算，治理无效——单源化边界必须上推到派生量的**源头函数**。
- 等价性/敏感性扫描的扫描域若以「被验证的推导值」为上界，会形成自遮蔽（输入错 → 验证域跟着缩小 → 全档 PASS）；扫描域应取独立上界（clamp 上界或宽档枚举）。
- 阻塞级检视结论须区分「推翻设计」与「修订文档」两类并在报告中写明：本次落点错误修正后 kernel 结构 / 数值语义 / 容量合规性均不变（clamp 接受 32,512、划分不变性覆盖该类、修正后 UB 占用 60.5%）→ conductor 可走 Stage 1 revision 而非重做设计。
- 纯 Vector（SimdVF）算子送机械检查器时，Cube 相关项（L0C 容量 / C0 分形对齐）会误报 warn；检视方须显式判定误报并在报告留证，不得直接采纳。

## Stage 1 修订（v1，2026-10-17，revision_index=1）

**触发**：Stage 2 检视 `REVIEW.md` 不通过（1 阻塞 + 2 建议）；v0 备份 `history_version/design_v0.md`。修订性质 = 文档级修正 + 验证域补齐（kernel 结构 / 数值语义 / 选型结论 / 容量合规性不变），明细见 DESIGN.md §11。

### 价值点（供蒸馏）

**R（流程规则）**：
- R-RECOMPUTE-OR-IMPORT：**wrapper 启发式落点必须脚本化复算（或直接 import 仓内镜像的 `compute_tile_n`），禁止手算**——v0 手算 align_up(33000,256) 出错（130×256=33,280，正确 129×256=33,024）且漏读 divisor 拒绝分支（`div_tiles=3 > max_tiles=2` → 返回 tile_n_max=32,512 而非 11,008），落点 (1,16640) 应为 (1,32512)。`P-LAYOUT-ROW-RESIDENCY` 条目早已警示「契约落点复算须逐 block_m 展开 wrapper 启发式」、VP-2026-0025（pending Tier 2）同方向预警——**知识库已有防错规则但设计时未检索命中**。防再犯机制：L0 落点 assert 改为 import `examples/TileOPs/tileops/kernels/reduction/_primitives.py` 的 `compute_tile_n` 复算期望值（§8.2），落点漂移在 Stage 3 首跑暴露。
- R-REVIEW-VERIFY-FIRST：**检视修订须先复核再落笔**——对 REVIEW.md 每条结论先读源码/条目原文核实（本次逐一确认：compute_tile_n L176–183 divisor 规则、vf_regions.h L39–40〔检视指出的行号偏移属实〕、P-LAYOUT-ROW-RESIDENCY 甜点注记原文〔−21.6~−36.4% / −24~−29%〕、GPU 源 S1a/S1b 行号），全部属实后落笔——「按检视意见抄数字」会引入二手错误风险。
- R-PATTERNLIB-RECON：**选定方案与 pattern-library measured 注记存在同量纲数据时必须对账**——v0 选 BLOCK_N=tile_n 直保留（25600/32512）未与 SimtVF 甜点实测（16384，双向偏离回退）对账；v1 补对账（§1.6.3）+ Stage 4 A/B 观察项（>5% 阈值切换，§8.3）+ 风险 9。跨结构（SimtVF→SimdVF）实测不可外推是保留直保留的理由，但张力必须显式文档化并由实测裁决。

**P（模式方法）**：
- P-SINGLE-SOURCE-LOOP：单源约束的修订闭环 = 「脚本先行（改输入/口径）→ 重跑 → 回填文档」——本次 design_budget_calc.py（tile_n 输入 + merge 计数修 full+尾各一次）与 verify_equiv.py（V1 档改 32512、V6 加 32512）均先改后跑，DESIGN.md 内嵌数字零手写。v0 的 O5 merge 0.19% 漏计尾 chunk 正是「文档数字脱离脚本」的口径错误——merge 计数归属脚本后此类错误不可再发生。
- P-MERGE-COUNT-TAIL：分块归约的 merge 开销计数须按「每个实际执行的 chunk（full + tail）各计一次」——尾 chunk 同样执行完整 flash 合并链（7 指令），只计 num_full 会低估（v0 (33,33000) 低估 2×）。

**D（实测数据）**：
- (33,33000) 修正后：BLOCK_N=32,512（=127×256，非 pow2 直保留结论不变）、num_full=1/rem=488/G=508/G_tail=7/k_tail=40、x_ub 130,048B、总占用 138,304B（60.5% ≤ 228,556B 合规）；per-row 指令数与 v0 巧合相同（总组数 516 不变：508+7+1 = 260+255+1）。
- verify_equiv 重跑（V1 (33,33000,tn=32512) fp64 ulp=0 / fp32 ≤1；V6 扩 14 行含 32512 档全 ≤1 ulp）——**划分不变性在修正落点档再次成立**；V1 fp32 4 shape ulp=0（v0 误记 5）。
- 落点复算可执行命令（仓内镜像）：`compute_tile_n(1, elem, align_up(N,256), budget=min(221184, MAX_SINGLE_TILE_COLS*1*elem))` → (33,33000) 32,512 / wl3 25,600 / wl2 16,384（本设计执行期实测）。

**C（案例索引）**：
- 同源错误传播链：GPU 源 `_primitives.compute_tile_n`（divisor 拒绝规则）→ v0 手算错 → 落点表/§4.5/L0 assert 多处继承 → Stage 2 检视拦截。SIMT 兄弟设计存在同款 (33,33000) 落点错误（不在本任务范围，其 Stage 4 复算可同法根治）。
- logsumexp 家族修正后状态：SIMD single（86 PASS）/ SIMD tiled v1（本设计，待 Stage 3）/ SIMT tiled（61 PASS + Stage 4，含同源落点错误待其任务修正）。

## Stage 2 (review, 复检 revision_index=1) Retrospective — review — 2026-10-17

**产出**：REVIEW.md（**结论: 通过**；阻塞 0 / 建议 0，覆盖上轮「不通过」报告）+ 机械复核三件重跑（verify_equiv ALL EQUIV_PASS 且与内嵌表全一致 / design_calc_check JSON / design_budget_calc 单源逐位复现）+ 落点 7 个独立脚本复算（ast 提取镜像纯函数）+ D-2 独立复算 + 核数本机复测（28/56）+ 上轮 3 问题修复验证 3/3。维度判定：0–8 **全 pass**（上轮 0 fail / 2 warn / 6 fail / 7 warn / 8 warn 全部转 pass）。检视对象：DESIGN.md v1（987 行）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | **复检（revision_index≥1）在检视 skill 无专属流程段**：SKILL.md Phase 1–4 按首检全维度设计；复检实际需要「全维度独立重验 + 上轮问题逐条修复验证 + 旧值残留语境判定」三个额外动作——修复验证节结构（问题 → 修正位置 → 验证方式 → 结论）与残留判定（16640/33280 grep 逐处命中判定：泛化档/可达性注记/修订对照 = 合法语境）均为本次自创格式，无模板可依 | REVIEW.md#上轮问题修复验证（3 节）；DESIGN.md#§11 | `.agents/skills/tilelang-design-review/SKILL.md` 增「复检模式」节：REVIEW.md 模板加「上轮问题修复验证」章节；数值错误修正后要求旧值 grep 全文扫描 + 逐处命中语境判定（残留错误 vs 合法语境），合法语境注记的内容本身也须独立复算 | R |
| Tooling | **上轮报告的工具适配缺口本次复检原样复现（未修复）**：`design_calc_check.py` ① `l0c_budget` 对纯 Vector 算子误报 warn（C0 分形对齐不适用）、② `ub_budget` 对 §4.5 表列式解析失败 → skip（人工复核成本转嫁给单源脚本重跑）。两轮复现证明非偶然现象，误报/skip 判定结论可跨版本沿用但须每轮重新留证 | REVIEW.md#机械复核（design_calc_check 表两行）；上轮 RETROSPECTIVE Stage 2 Skill Flow Issues 第 2 行 | 同上轮建议：`.agents/tools/design_calc_check.py`——文档无 `T.alloc_l0c`/`T.gemm` 时 `l0c_budget` 返回 skip + reason="pure-vector kernel"；`ub_budget` 增补该表列式解析 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 检视期迁移落点复算的依赖规避模式：仓内镜像 `_primitives.py` 直接 import 因依赖链触发 `tilelang.ascend` ModuleNotFoundError 时，用 `ast` 提取纯函数（`align_up`/`compute_tile_n`）参数化复算（BUDGET 显式传入）——不依赖包可导入性、不改动镜像源码、与 Stage 3 测试内 import 路径（设计 §8.2 assert）互不干扰 | P | REVIEW.md#机械复核（wrapper 落点独立脚本复算节，方法说明） | `ast.parse` 镜像 `examples/TileOPs/tileops/kernels/reduction/_primitives.py` → 提取两函数 exec → `BUDGET=221184` 逐 block_m 复算 → 7 落点与 §0.2 表全一致（(33,33000)→(1,32512)） | tilelang-fork 工作区 2026-10-17；设备 Ascend950PR_9579；静态复算未上机 | `.agents/skills/tilelang-design-review/SKILL.md`（Phase 1 迁移落点复算步骤，若上轮 R 提案合入则并入其执行方法） |
| 建议级问题（pattern-library 对账类）的修复验证判据三件：① 对账数字与知识库条目原文逐字一致；② 裁决三件套齐（备选结构可实现 / 明确判定阈值 / 结果回写路径）；③ 风险表登记——仅「补一段讨论」不构成修复 | R | REVIEW.md#上轮问题修复验证（问题 2）；DESIGN.md#§1.6.3 L425 + §8.3 L870–873 + §9.1 L889 | grep `layout.md` L29 甜点注记数字（−21.6~−36.4% / −24~−29%）与 DESIGN §1.6.3 对账段逐字比对 → 一致；§8.3 含 >5% 阈值 + opt_log/pattern-library 回写路径 | 同上 | `.agents/skills/tilelang-design-review/SKILL.md`（Phase 2 复检修复验证判据） |

### Transferable Lessons（可迁移教训）

- 复检不得退化为「对上轮问题打勾」：全维度独立重验（机械复核全部重跑 + 落点全量复算 + 维度 0 源码重核）是确认修订未引入新漂移的前提——v1 新增内容（32512 档、G 分档三处、merge 计数修正、甜点对账段）本身也是检视对象，本次逐一重验通过。
- 数值错误修复的验证方法 =「旧值 grep 全扫 + 逐处命中语境判定」：旧值残留 ≠ 未修复（泛化档/可达性注记/修订对照为合法语境），且合法语境注记的内容正确性也须独立复算（§0.2 L54 的 16640 可达性注记经本轮复算确认成立）。
- 机械复核的「误报/skip → 人工判定」结论建立在结构事实（如无 Cube 路径）之上时可跨版本沿用，但每轮报告须重新声明判定依据留证，不得默认继承上轮报告。

## Stage 3（实现，first_impl，attempt_index=1）— 2026-10-17

**产出**：`_logsumexp_kernel_tiled.py`（kernel + golden + L0/L1/L2/Boundary 分层测试 + main `--level` 入口）。**判定：[PRECISION_PASS]**（L0 + L1 全过，L2/Boundary 全过 0 WARN）。

**路径**：首次实现即命中风险 1 —— primary `alloc_var` 跨 SimdVF 块形式在 L0 #1 首编触发 `ValueError: [TileLang Semantic Check] local.var buffer 'm_run' is modified inside a SIMD_VF/SIMT_VF block but is defined outside the VF block ... pointer-type capture`（与 allocate.py L103-168 + vf_regions.h 机制论证、vf_checker.py L73 `_check_vf_local_var_write` 完全吻合）。按 DESIGN §9.1-1 fallback 切换为 stats-UB 往返形式（`stats_ub (2,) fp32`：vsts ONEPT_B32 写 lane0 + vld BRC_B32 广播重载 + mem_bar("VST_VLD") 定序），一次改写即全量通过，零试错。

**测试结果**：L0 8/8、L1 36/36、L2 6/6（0 WARN）、Boundary 19/19（含 determinism 3× 位级一致），max_diff 全为 0.0（fp32 用例 ~9.5e-7）。wrapper 落点复算 assert 通过（(33,33000)→32512 等 3 档）——落点漂移防护生效，v0 的 16640 错误已被门槛拦截。

### 价值点（供蒸馏）

**D（实测数据）**：
- D-ALLOCVAR-DISALLOWED：`T.simd.alloc_var` **跨 SimdVF 块**（块外声明、多块读写）在当前 ascend 后端被 VFChecker 判定非法（需 pointer-type capture），报错字符串见上。与 single 变体「单块内声明+重赋值」合法形成对照——constraint 边界 = **VF 块边界**，非行/函数边界。DESIGN 风险 1 的确定性结论（此前机制论证支持可行、仅无先例，现首编裁决为不可行）。
- D-STATS-ROUNDTRIP：stats-UB 往返 fallback 端到端精度无损（全档 max_diff=0.0，含 63-chunk 长链与跨 chunk maxgap 131008 角点）；代价与 DESIGN 估算一致（~0.14%）。

**P（模式方法）**：
- P-STATS-UB-ROUNDTRIP：跨 VF 块的 loop-carried 标量（running max/sum 类在线统计）落地范式 = 2 元素 fp32 UB buffer + 「vsts(ONEPT_B32) 写 lane0 → mem_bar("VST_VLD") → vld(BRC_B32) 广播重载」，SSA 新名（m_run_old/m_run_new 等）承接跨块数据流；逐 chunk store+bar 后再由下个块 load。flash_attention core.py L103/L117/L120-126 先例，本任务验证其对「每 chunk 一次合并」的在线归约同样适用（flash 是 KV 块内合并，这里是 chunk 间合并）。
- P-RISK1-FALLBACK-PREPLAN：design 阶段「主选无先例但机制可证 + fallback 有先例」的两手准备，使 Stage 3 首编命中风险后一次切换即收敛（无试错成本）——风险 1 的确定性结论由「机制论证 + 先例检索 + 首编裁决」三件套共同锚定。

**R（流程规则）**：
- R-CAPGAP-BOUNDARY：alloc_var 跨块禁用是**语义约束**（VFChecker 按 scope 规则正确执法，非后端能力缺失），且 fallback 已预规划、算法未被迫换、代价可忽略 → 按 AGENTS.md Capability Gap Rule 判定**不登记 capability-gaps.md**（该簿收「无法实现/被迫换算法/触及性能上限」），约束本体与绕法走 pattern-library（P/D 类）。若 evolver 复核认为需登记，可据此段独立裁决。

**C（案例索引）**：
- logsumexp SIMD tiled 变体完成：`examples/logsumexp_simd/_logsumexp_kernel_tiled/`（Stage 3 PRECISION_PASS，69 用例全过）。三变体现状：SIMD single（86 PASS）/ SIMD tiled（本任务 69 PASS）/ SIMT tiled（61 PASS + Stage 4）。

### Transferable Lessons

- SimdVF 的 loop-carried 标量不能跨 VF 块驻留（VFChecker 强制），须用 stats-UB 往返；约束边界是 VF 块而非行/函数——设计期就要把「跨块在线统计」列为风险并预备 fallback。
- 首次实现命中预规划风险时，fallback 落地即收敛的代价趋近于零；真正昂贵的是首跑试错后才补设计。
