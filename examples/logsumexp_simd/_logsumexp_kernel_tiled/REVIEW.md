# _logsumexp_kernel_tiled（SIMD tiled 变体）设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/logsumexp_simd/_logsumexp_kernel_tiled/DESIGN.md`（**v1**，987 行，revision_index=1；§0–§11 齐全，无占位符；v0 备份 `history_version/design_v0.md`）
- 任务类型: **迁移**（含 §0，0.1–0.7 小节齐全）
- 源算子: `/home/z00875089/TileOPs-fork/tileops/kernels/reduction/logsumexp.py`（`_logsumexp_kernel_tiled` L105–194、dispatch L196–212）+ 同目录 `_primitives.py`（`align_up` L91–107、`compute_tile_n` L109–194、`DEFAULT_ALIGNMENT=256` L30、`MAX_SINGLE_TILE_COLS=32512` L39）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: **0**
- 建议级问题数: **0**
- 检视轮次: **第 2 轮复检**（对 v1 修订稿，覆盖上一轮「不通过」报告〔1 阻塞 + 2 建议〕；上轮 3 问题修复验证 **3/3 通过**，见「上轮问题修复验证」节）
- 检视日期: 2026-10-17；检视方式: 亲自 Read 源算子全文 + DESIGN.md v1 全文（987 行分段）+ Glob/Grep 核对全部引用工件（25 个文件实存，含 `history_version/` 备份与单源脚本）+ 机械复核三件重跑（verify_equiv / design_calc_check / design_budget_calc）+ wrapper 落点 7 个独立脚本复算（ast 提取仓内镜像纯函数）+ D-2 roofline 独立复算 + 核数本机复测 + 源码机制锚点 6 处亲验
- 维度判定汇总: 0 pass / 1 pass / 2 pass / 3 pass / 4 pass / 5 pass / 6 pass / 7 pass / 8 pass

## 上轮问题修复验证（3/3 修复）

### 问题 1（阻塞）：wrapper 落点 (33,33000) 手算错误（(1,16640) 应为 (1,32512)）→ **已修复**

- §0.2 L69 推导链修正：`N_padded = align_up(33000, 256) = 33,024 = 129×256`（v0 误算 33,280=130×256）；`eff = min(221184, 32512×1×2) = 65,024B → tile_n_max = 32,512`；`best_dividing = 11,008`（129=3×43 的 ≤127 最大因子 43）→ `div_tiles = 33,024/11,008 = 3 > max_tiles = ceil(33,024/32,512) = 2` → **拒绝 divisor，返回 tile_n_max = 32,512** —— 与本轮独立脚本复算**逐步一致**（见机械复核「落点独立复算」节）。
- 下游联动全部一致传播：§4.5 L673（预算行 x_ub 双版本 130,048B / 总占用 138,304B / 60.51%）、§4.7 L694、§5.2 L704、§8.2 L850（L0 #4 tile_n=32512 + 派生量 num_full=1/rem=488/G=508/G_tail=7/k_tail=40）、兄弟对照 chunk 数（32512 档 2 chunks）。
- **防再犯机制落地**：§8.2 L856 落点 assert 由硬编码常数改为 `import compute_tile_n` 复算（仓内镜像 `examples/TileOPs/tileops/kernels/reduction/_primitives.py`）——落点漂移将在 Stage 3 首跑暴露。
- 残留值语境判定（grep 全扫，逐处命中判定）：`16640`/`33280` 现存命中均为**合法语境**——① §0.2 L54 可达性注记（N ∈ (33,024, 33,280] 时 N_padded=33,280=130×256、因子 65 → 16,640；本轮独立复算验证该注记**内容正确**）；② V2/V4 与 §8.2 Boundary 第 3 档的**显式 config 泛化档**（与内嵌表声明一致，扩大契约空间覆盖，非落点错误残留）；③ §11 修订说明的 v0 错误记录（历史对照）。
- verify_equiv 生产落点档机器证据缺口已根治：V1 (33,33000, **tn=32512**) fp64 ulp=0 / fp32 ulp=1；V6 扩至 14 行含 32512 档全 ≤1 ulp（上轮「修正落点不在任何扫描档内」问题消除）。

### 问题 2（建议）：BLOCK_N 直保留未与 P-LAYOUT-ROW-RESIDENCY 甜点实测对账 → **已修复**

- §1.6.3 L425 甜点对账段：数字与 pattern-library `layout.md` L29 原文**逐字一致**（双向偏离 16384 回退：8192 档 −21.6~−36.4%、32768 档 −24~−29%；「甜点须实测扫描」口径）；显式承认 wl3=25600（+56%）与 (33,33000)=32512（+98%，邻近 32768 回退档）落在回退区间附近的**潜在张力**；保留直保留的三条理由成立（① 契约保真——tile_n 是 wrapper 契约参数；② 实测点为 SimtVF 结构，SimdVF 甜点未实测、不可跨结构外推——与条目自身口径一致；③ V6 划分不变性保证换档零语义成本）。
- §8.3 L870–873 Stage 4 A/B 观察项三件套齐：主选 + 备选（25600 vs 16384、32512 vs 16384）+ 裁决计划（代表 shape、do_bench(msprof) + npusim 归因口径、**判定阈值 >5%**、切换路径〔仅改 BLOCK_N host 常数 + §4.5 单源重算〕、回写路径〔opt_log.md + pattern-library 补 SimdVF 结构实测点〕）。
- §9.1 L889 风险 9 登记（张力显式文档化 + Stage 4 裁决入口）。上轮价值点建议「须对账并声明结构差异」完整落实。

### 问题 3（建议）：数值/口径微偏差 5 处 → **已修复（5/5）**

1. V1 内嵌表 fp32「4 shape 为 0」——与本轮重跑实测一致（ulp=0：(4,32768)/(8,100)/(3,257)/(2,1024000)）✓；
2. V6 域式如实标注 `bn ≤ min((tn//64)*64, N)` 且扩为 14 行（含 32512 生产档）✓；
3. G 展开分档三处一致（wrapper 落点 ≤508 / fp32 clamp ≤430 / 16-bit 显式 config clamp ≤860；L389/L754/L888）✓；
4. O5 merge 占比 0.39%（脚本 `design_budget_calc.py` L102–105 计数修正为 full + 尾 chunk 各计一次——v0 漏计尾 chunk 低估 2×）✓；
5. 源码行号修正（S1a L150–154+L163–167 / S1b L155–162 / `vf_regions.h` L39–40）——与源码逐一核对一致 ✓。

## 机械复核（D-1/D-5）

### verify_equiv 重跑（等价性机器验证）

命令：`python3 examples/logsumexp_simd/_logsumexp_kernel_tiled/verify_equiv.py`（torch CPU，seed 固定 → 确定性可重现）

结果：**VERDICT: ALL EQUIV_PASS**，且与 §1.6.1 内嵌结果表**全部一致**（v0 的两处不符已随 v1 修订消除）：

| 项 | 重跑结论 | 与 §1.6.1 内嵌表一致性 |
|----|---------|------------------------|
| V1 partition/grouping equivalence | EQUIV_PASS（16 行：8 shape × fp32/fp64；fp64 全 8 shape `max_ulp=0`；fp32 全档 ≤1 ulp，其中 **4 shape ulp=0**；**(33,33000) 档 tn=32512**〔v1 更新为生产落点〕fp32 ulp=1 / fp64 ulp=0） | **一致**（含「4 shape 为 0」修正；机器证据已字面覆盖生产落点档） |
| V2 reduction-grouping | EQUIV_PASS | 一致（(33,33000) tn=16640 为显式 config 泛化档，与内嵌表声明一致——合法语境） |
| V3 golden equivalence | EQUIV_PASS | 一致 |
| V4 source-fidelity bitwise | EQUIV_PASS（18 类特殊行 × fp16/bf16：位级 0 差、NaN/±inf 模式一致） | 一致（tn=16640 同 V2，泛化档合法） |
| V5 first/last-chunk-all-inf | EQUIV_PASS ×2（tiled 特有角点：首 chunk 全 −inf → NaN；尾 chunk 全 −inf → 中性） | 一致 |
| V6 partition invariance | EQUIV_PASS（**14 行**，域式 `bn ≤ min((tn//64)*64, N)`，全档 ≤1 ulp；**含 (33,33000) 32512 生产档**——v1 新增） | **一致**（上轮「域描述不符 + 生产落点未覆盖」双缺口已根治） |
| V7 saturation closure（info） | EQUIV_PASS（all-65504 行 N=102400/1024000：y=65515.54/65517.84 < 65520 IEEE-inf 阈值，0 分歧） | 一致 |

### design_calc_check（算术复算）

命令：`python3 .agents/tools/design_calc_check.py --design examples/logsumexp_simd/_logsumexp_kernel_tiled/DESIGN.md`

| check | status | detail（JSON 摘要） | 检视判定 |
|-------|--------|--------------------|---------|
| `core_split` | **pass** | recomputed `ceil(1/1) × ceil(56/1) = 56 blocks` vs declared grid 56（1-D `T.Kernel(N)`） | 采纳 → 维度 3 pass 依据之一 |
| `r3_metrics` | **pass** | R3 四口径齐全（FLOPs / access bytes / scan passes / buffer peak） | 采纳 → 维度 8 调研完整性依据之一 |
| `ub_budget` | **skip** | "no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)" | **机械跳过，人工已核**：重跑单源脚本 `history_version/design_budget_calc.py`（v1 已修正 tile_n=32512 输入，L138）→ §4.5 全表与 D-2 全部数字**逐位一致**（明细见下节）；上轮 garbage-in-garbage-out 缺口（硬编码 16640）已随 v1 脚本修正根治 |
| `l0c_budget` | **warn** | "block 1x1 x fp32 accum … WARN: block_M/block_N not multiples of 16 (C0 fractal alignment)" | **误报，人工判定 pass**：纯 Vector（SimdVF）算子——无 `T.gemm`、无 `T.alloc_l0a/l0b/l0c`、无 Cube 路径（§4.3/§4.4 仅 GM↔UB），C0 分形对齐约束不适用（与上轮判定一致，工具适配缺口仍在） |

### 单源预算脚本重跑（design_budget_calc.py，人工复核 ub_budget skip 项）

`python3 history_version/design_budget_calc.py` 输出与 §4.5 全表及 D-2 全部数字**逐位一致**：wl2 计算项 122.7K cyc/core（最重核 19 行 = 127.5K）、special_func 占 61%、MTE 带宽项 12.0K（计算/流量比 10.2×）、UB 重读 4.68K（3.8%、26× 余量）、merge 指令占比 **0.39%**（修正后 full+尾各计一次）、wl3 merge 0.25%、fp32@clamp 上限档 99.9%（228,416/228,556B）、**(33,33000) x_ub 双版本 130,048B / 总占用 138,304B / 60.51%**（≤ 228,556B = 248KB×90%，`CONST-capacity-a5` assumption 口径，§4.5 已披露）。

### wrapper 落点独立脚本复算（不采信文档手算值）

方法：因仓内镜像 `_primitives.py` 直接 import 触发 `tilelang.ascend` ModuleNotFoundError（检视环境路径解析），用 `ast` 从 `examples/TileOPs/tileops/kernels/reduction/_primitives.py` 提取 `align_up`/`compute_tile_n` 纯函数，`BUDGET=221184`（216KiB，A5 适配 `device_smem_budget`）逐 block_m 遍历复算。**全部 7 个落点与 §0.2 表一致**：

- **(33,33000) → (1, 32512)**：N_padded=33,024=129×256；eff=65,024B；tile_n_max=32,512；best_dividing=11,008；div_tiles=3 > max_tiles=2 → 拒绝 → 32,512 ✓（派生量 num_full=1/rem=488/G=508/G_tail=7/k_tail=40 与 §0.2/§8.2 一致）
- wl2 (1024,32768) → (1,16384) ✓；wl3 (4,102400) → (1,25600) ✓；(4,32768) → (1,16384) ✓；(33,32768) → (1,16384) ✓；fp32 clamp 55,040 → 27,520 ✓
- 16,640 可达性注记（§0.2 L54）独立验证：N ∈ (33,024, 33,280] 时 N_padded=33,280=130×256、因子 65 → 16,640 ✓——注记内容正确，仅在非 wrapper 主流落点语境可达。

### D-2 roofline 独立复算（不依赖设计脚本）

wl2 (1024, 32768) bf16：GM 读 67,108,864B ÷ (56 核 × 100 B/cycle) = **11,983 ≈ 12.0K cyc/core** ✓；计算项 (1030 ALU × 64 lane ÷ 51) + (257 special × 64 ÷ 8) = 3351.9 cyc/(行·chunk) × 2 chunk × (1024/56 行) = **122,588 ≈ 122.7K** ✓（最重核 19 行 = 127,372 ≈ 127.5K ✓）；special 占比 **61.3% ≈ 61%** ✓；计算/流量比 **10.2×** ✓；UB 重读 1.198MB/core ÷ 256 B/cycle = **4,681 ≈ 4.68K**（3.8%、26.2× 余量）✓。粒度口径（lane-op 折算，64× 边界）与上轮复核结论一致；不确定性已由设计自披露为 §9.1 风险 4 + Stage 4 npusim 标定入口，符合 assumption 口径。

### 硬件常数本机复测

`torch.npu.get_device_properties()`（本次复检期执行）：Ascend950PR_9579 → `cube_core_num=28`、`vector_core_num=56` —— 与 §5.5 声明的 56 及 pattern-library `CONST-arch-a5`（status: measured，28/56）**一致** ✓。

## 检视详情

### 0. 源算子理解与迁移分析 — **pass**

**核对方式**：亲自 Read 源算子全文（kernel builder L105–194 + dispatch L196–212 + `_primitives.py` `align_up`/`compute_tile_n` L91–194），逐小节对照 §0.1–§0.7；全部 7 个 wrapper 落点独立脚本复算（上节）；本轮为 v1 复检，同时验证上轮阻塞问题（落点错误）修复。7 检查项全 pass：

| 检查项 | 判定 | 证据（源码 ↔ DESIGN.md） |
|--------|------|--------------------------|
| 语义理解正确性 | pass | §0.1 语义要素表 7 行与源码逐项一致：数学映射（L180–188 `y=row_max+log(row_sum)`）、I/O 契约（kernel 签名 L125–128）、规约语义（chunk 内 `T.reduce_sum` 树形 + chunk 间按序 flash 重缩放合并、**全程 fp32 中间**，L131–138/L147–183）、dtype 语义（L152–167/L180–185）、边界语义 ①–⑥（NaN 传播 / 全 −inf → NaN 源固有分歧 / +inf → NaN / **首 chunk 全 −inf → NaN tiled 特有角点**〔L150–162 递推状态推演 + V5 位级锚定〕/ 合并式 exp 参数 ≤0 无溢出〔`TRAP-NAIVE-ONLINE-MERGE` 88.7/131008 口径一致〕/ M 尾块行谓词 L155–157）、尾块非整除（L116–120/L143–167：仅尾 chunk 可越界、快慢双路径）、host 侧语义（L116–121 常量折叠 + L338–360 wrapper 契约）——无臆造、无遗漏 |
| 算法解读完整性 | pass | §0.3 步骤分解表（S0/S-init/S1a/S1b/S2/S3a/S3b/S4/epilogue）覆盖源码 L116–188 全部计算语句；v1 行号修正后与源码一致（S1a L150–154+L163–167 快路径、S1b L155–162 masked 分支——本轮亲核）；host 侧逻辑（N_padded/num_tiles/total_cols/_needs_mask 折叠 + `compute_tile_n` 契约）已完整列出 |
| 优化手段覆盖完整性 | pass | §0.4 识别源码全部显式优化并逐项标注硬件依赖：online 流式递推（L135–183）、快慢双路径装载（`T.copy` 快路径 vs masked 逐元素，L143–167）、N_padded 256 对齐（L116–118）、host 常量折叠（L116–121）；算法层/实现层区分正确（online 递推=算法层保留；对齐 padding=GPU 向量化器实现层，R2 重解释） |
| 耦合性判定合理性 | pass | §0.5 13 条目 = 保留 4 + 等价替换 3 + 重新设计 5（R1–R6）+ 舍弃 1（threads，K3），四态处置齐全且逐条有依据；与 backend-glossary §1 概念迁移锚不矛盾（shared→UB、reduce→simd intrinsic、T.copy 路径）；舍弃项理由充分（SimdVF 无线程概念，调用面契约已注明） |
| NPU 重设计可行性 | pass | §0.6 每个重设计项（R1 行串行×chunk 内层、R2 BLOCK_N=tile_n 紧凑、R3 alloc_var stats 驻留、R4 静态切片、R5 静态二分支、R6 一维 grid）均有「源方案 → NPU 新算法 → 语义保持论证」三要素；满足 Ascend 约束：一维 `T.Kernel(56)`、内存层级 alloc API（alloc_shared=UB / alloc_var=local.var VRF）、容量 90% 折扣口径（228,556B，60.51%/99.9% 合规）、无 C0 约束（纯 Vector）；累加顺序变化的精度影响已论证（chunk 粒度 online 序保持 → V1 fp64 ulp=0 位级锚定） |
| 设计与决策一致性 | pass | §1–§7 描述的是 §0.5/§0.6 决策后的 NPU 算法（SimdVF 2048-bit 指令流、alloc_var stats、BLOCK_N=tile_n、T.Persistent 任务调度），无「§0 判重设计、§3 照抄源方案」脱节；v1 落点修正一致传播至 §4.5/§4.7/§5.2/§8.2（问题 1 验证节） |
| golden 独立性 | pass | §8.1 golden = `torch.logsumexp(x.float(), dim=-1).to(x.dtype)`（源仓测试基准 `ref_program` 直移植，NPU 上计算），**未复刻** §0.6 NPU 数据流；角点保真由独立的 `_source_lse_tiled_sim` 源语义模拟器承担（§9.2-10 要求先对 verify_equiv 源链自测防模拟器污染）；golden 分歧角点（3 类）显式 assert 分歧形态——精度对比非自证 |

**微小观察（不构成 warn）**：§0.2 L54 与 §11 引「`_primitives.py` L176–183」（divisor 拒绝规则）指向该规则的注释区，规则代码实际位于 L188–194——引用语义正确，仅行号粒度微偏，无需修改。

### 1. API 可行性 — **pass**

- **API 存在性**：§3.2 映射表全部 19 个 `T.simd.*` API（vexpdif/vcadd/vcmax/vcvt/vsel/vdup/vld/vsts/pset/mem_bar/alloc_var/reduce_max/reduce_sum 等）经 Grep `tilelang/ascend/language/simd.py` + `src/tl_templates/ascend/simd_inst.h` 逐一确认实存且签名一致；`exp2` 类 API 在方言**零命中**——「无 exp2 快档、vexpdif fp32-only」负向断言有 `VP-2026-0014`（`.agents/evolution/queue.md` L562）evidence 印证。
- **编程模式匹配**：`programming_mode: simd` + `auto_schedule: yes`（§2.1/§10.3），全文无手动 `T.ascend_set_flag`/`T.ascend_wait_flag`/`T.ascend_pipe_barrier` 混写；`mem_bar("VST_VLD")` 为指令流内 barrier（SimdVF MicroAPI 语义），非跨核手动同步。
- **SimtVF 优先例外成立**：SimdVF 为**用户显式指定**的编程模式契约（DESIGN.md 头部声明「非默认 simt」），非同族 single 变体（86 用例 PASS）已实证该指令链——非任意降级。
- **先例锚点全部核实**：flash `core.py` L103/L115–118/L120–126/L130–137（stats UB 往返 fallback 形态）、`transform_sf.py` L205/215/249、`test_simdvf_arithmetic.py` serial_vreg_accum L86、cast 测试 L64–80、predicate 测试 L261、single 变体 L115/154–211/215–219、SIMT tiled 兄弟 L156–159/222/231——引用文件实存、行号语义相符。
- **源码机制锚点 6 处亲验**：`vf_regions.h` L39–40（enclosing-scope 域检查）、`insert_sync.cc` L1308（local.var 跳同步）、`allocate.py` L103（alloc_var → scope="local.var"）、`simd_inst.h` L688–692/L704–708（vcadd/vcmax → `asc_reduce_*`，无 dtype 声明——风险 4 口径一致）、`estimate_latency.h` L58–70（51/8/256 常数）、`ascend_simdvf_lower_parallel.cc` L818+（lower 路径）——风险 1（alloc_var 跨块）的机制证据链成立，fallback 全先例。

### 2. 内存层级规划 — **pass**

- **搬运路径完整**：§4.4 仅 GM↔UB（`T.copy` 装载 + UB 写回），在 tilelang-ascend skill §6 支持集内；无 Cube/ND2NZ/越级路径（纯 Vector）。
- **UB 预算**：`ub_budget` 机械 skip → **人工已核**（单源脚本重跑逐位一致，见机械复核节）：wl2 73,792B/32.3%、wl3 110,656B/48.4%、(33,33000) 138,304B/**60.51%**、fp32 clamp 档 228,416B/**99.9%**——全部 ≤ 228,556B（248KB × 90% 折扣；`CONST-capacity-a5` L10 assumption 口径与条目 status 已在 §4.5 披露）；x_ub 双版本（multibuffering num_stages=2）已计入（130,048B = 2×65,024B）；99.9% 极限档的膨胀系数风险已由 §9.1 风险 7 披露（降 BLOCK_N 路径 + 回写 constants.md）。
- **L1/L0A/L0B/L0C**：无 Cube buffer——`l0c_budget` 机械 warn 为**误报**（C0 分形对齐不适用），人工判定 pass。
- **Buffer 层级标注**：逐 buffer 标注层级 + alloc API（x_ub/lse_ub/stats_ub(fallback) = `T.alloc_shared` UB；m_run/s_run/acc/常量 = `T.simd.alloc_var` local.var VRF——VRF 水位 ~15 寄存器 vs `CONST-vrf-capacity-a5` L149 容量，无压力）。

### 3. Tiling 策略 — **pass**

- **Block 划分与 grid 映射**：§5.2 block_m（wrapper 契约参数，tiled 落点偏好 1；行串行结构天然兜底任意 block_m）与 BLOCK_N=tile_n 直保留（64 倍数 clamp 兜底显式 config 档）取值及理由明确；grid = 一维 `T.Kernel(56)` + `T.Persistent([num_tiles], 56, num_stages=2)`，tile → (行块, chunk) 一维映射自行计算。
- **对齐约束**：纯 Vector 无分形约束；BLOCK_N 为 64（组宽）倍数——tile_n 256 倍数自动满足，clamp 机制兜底非契约档。
- **非整除处理**：尾 chunk masked 逐元素载入（rem/k_tail 编译期常量）+ M 尾批行谓词（rem_m/rows_this 静态推导）双逻辑齐备（§6.4）。
- **分核三要素**（core-split-strategy §1 核对）：① grid 56 blocks = `vector_core_num` **本机实查**（Ascend950PR_9579：cube 28 / vector 56，本轮复检期复测一致；查询代码与返回值已记录 §5.5）；② AIC-AIV 分工：纯 Vector 算子**仅占 AIV**（已注明）；③ 同步策略：`auto_schedule: yes`（AutoSchedule，无手动 flag、无 flag id 预算问题）。核内串行边界（num_tiles/rows/chunks/G）全部编译期静态值 ✓。`core_split` 机械 pass（56 blocks 复算一致）。

### 4. 技术约束检测 — **pass**

- **一维 grid**：`T.Kernel(56)` 无 `threads=`（§9.2-6 显式记录 TypeError 陷阱）；SimdVF 模式无线程域概念（K3 舍弃项一致）。
- **GEMM 分形/NT 约定、L0C 溢出**：n/a（无 Cube 路径，§4.3 无 `T.gemm`/`T.alloc_l0*`）。
- **方言技术约束清单完备**：SSA 跨块值 → alloc_var（§9.2-1，single 姊妹 Stage 3 教训）、TIR if vs Python if 域区分（§9.2-4：运行期 t vs 编译期 rem/k_tail）、切片全静态推导不依赖 `T.copy` OOB（§9.2-7，R4 重设计一致）、域内常量重物化（§9.2-9）、`T.annotate_buffer_versions` 在 `T.Persistent` 前（§9.2-6）——与源码机制锚点核对无矛盾。

### 5. 循环与同步 — **pass**

- **循环结构**：§6 三层选择明确——`T.Persistent`（任务级，num_stages=2）× 行串行（尾批 TIR if 谓词）× chunk 内层（Python `range(G)` 组展开 + chunk 间串行递推）；选择理由充分（行串行 + 单行 x_ub (1, BLOCK_N) 使 block_m 与 UB 容量解耦——P-BUDGET-UNCOUPLING；G 分档上界 ≤508/≤430/≤860 展开体积已评估，§9.1 风险 8 + `T.serial` 备选形态）。
- **同步与模式匹配**：§7 `auto_schedule: yes` 与全文一致，无手动 flag；块间数据依赖（x_ub 装载 → 计算 → stats 更新）由 AutoSchedule 插同步，local.var stats 跳同步语义有 `insert_sync.cc` L1308 机制证据。
- **尾块处理**：§6.4 尾 chunk（编译期常量 → Python if 静态二分支）+ 尾批（运行期 → TIR if）区分正确；num_full=0 全尾路径（N < BLOCK_N）与 M < block_m 全尾批路径均在契约空间（§0.2）与 L1 扫描覆盖。

### 6. 验证方案 — **pass**

- **Golden 函数**：§8.1 PyTorch 参考实现草案（`torch.logsumexp(x.float(), dim=-1).to(x.dtype)`，NPU 上计算 `device="npu"`）+ `_source_lse_tiled_sim` 源语义模拟器分工明确（golden 分歧角点对模拟器断言，防 torch 分歧误判——风险 2 缓解一致）。
- **L0 门槛测试计划**：8 用例含具体 shape/dtype/block_m/覆盖点标注——#4 **(33,33000) tile_n=32512**（v1 修正落点，覆盖尾 chunk G_tail=7+k_tail=40 + M 非整除尾批 + 非 pow2 + UB 60.5% 档）、#7 fp32 直连 + num_full=0 纯尾路径、#8 bm=4 行串行；**落点 assert 改为 import `compute_tile_n` 复算**（L856——期望值非硬编码，落点漂移 Stage 3 首跑即暴露，上轮问题 1 防再犯建议落实）。
- **精度标准**：fp16/bf16 → 1e-2、fp32 → 1e-3（累积密集型 5e-3 口径注记）、`torch.testing.assert_close`——符合 AGENTS.md 分档。
- **L1/L2/Boundary 覆盖面**：BLOCK_N 档扫描 {64, 256, 4096, 16384, 16640, 25600, 32512}（含生产落点档 + 泛化档 + 非 pow2）× 6 shape × block_m 三态 × 3 dtype；Boundary 18 类特殊行 × 4 档 tn/N 组合（第 2 档 = v1 修正落点档、第 3 档 = 显式 config 泛化档——语境标注清晰）× fp16/bf16；golden 分歧回归 3 类显式 assert 分歧形态；L2 鲁棒性 4 类（lru_cache 位级一致 / 连续异 shape / M=N=1 / clamp 上界档）。

### 7. 完整性与一致性 — **pass**

- **章节齐全**：§0–§11 齐全（迁移 §0 含 0.1–0.7；v1 新增 §11 修订说明——修订性质/明细/超范围事项记录规范）。
- **无占位符**：对 DESIGN.md v1 执行占位符三类模式扫描（花括号占位 / 未完成类标记 / 留空类标记）→ **0 命中**。
- **同类引用真实**：本轮 Glob 复核 **25/25 实存**（正文 23 个唯一文件：examples/ascend 3 + testing/ascend 3 + examples/TileOPs 6 + examples/logsumexp_simd 2 + examples/pilot 1 + src/tl_templates 1 + src/ascend 5 + tilelang 2；另 `history_version/design_v0.md` 备份 + `design_budget_calc.py` 单源脚本）；pattern-library 引用条目全部实存且语义相符（`CONST-capacity-a5` L10〔assumption，248KB/216KB〕、`CONST-arch-a5` L35〔measured〕、`CONST-valu-model` L125+L141〔special 档位注记〕、`CONST-mte-bandwidth-model` L103〔AIV MTE2 100B/cycle〕、`CONST-vrf-capacity-a5` L149、`TRAP-NAIVE-ONLINE-MERGE`〔88.7/131008〕、`D-LOGSUMEXP-MIGRATION-FIDELITY`〔fp16-e 50–60×〕、`P-LAYOUT-TAIL-STATIC-EXTENT`、`P-LAYOUT-ROW-RESIDENCY` 甜点注记 layout.md L29、90% 折扣规则 L7）；manifest `reduction.yaml` L124 `LogSumExpFwdOpSimd`、`test_logsumexp_simd.py` L75–76/L80/L82 落点注记核实。
- **内部一致**：落点数字（16384/25600/32512）跨 §0.2/§4.5/§5.2/§8.2 四处一致（grep 扫描 0 漂移——单源脚本治理生效）；API 映射（§3.2）/内存规划（§4）/Tiling（§5）三处相互一致；16640/33280 残留全部为合法语境（可达性注记/泛化档/修订对照，逐处判定见问题 1 验证节）；SIMT 兄弟 DESIGN L66 仍含同源落点错误——v1 §11.2–4 正确注明**超本任务范围**（职责边界：兄弟文档严禁改动），处置得当。

### 8. 算法优化分析 — **pass**

- **§1.6 章节存在性**：1.6.0（算法调研）/1.6.1（数学等价优化）/1.6.2（向量化替代分析）/1.6.3（向量化轴与数据布局决策）四子节齐全。
- **算法调研存在性与完整性**：R1 候选表 **15 项含基线**（源 online 递推 chunk 粒度），对照 `algorithm-candidates.md` **ALG-softmax L10–16 命中行**核对——候选覆盖一致，**无漏候选**；R2 在线算法有明确结论（**存在且源即在线**——chunk 粒度选定、组粒度弃选 #8，与「源算法已用 online 手段」无矛盾）；R3 复杂度对比表**四口径齐全**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值，基线在表，口径写清）——`r3_metrics` 机械 pass；R4 硬件亲和逐候选评估（单元匹配/容量/对齐/静态边界/流水融合/跨核），容量/带宽淘汰引用 `CONST-capacity-a5`/`CONST-mte-bandwidth-model` 条目（D-2 口径）。
- **调研结论正确性（独立复核）**：① **等价性独立推演成立**——flash 合并式全部 exp 参数 ≤0（p ≤ m_run、x^c ≤ m_c ≤ m_run）无溢出，与 `TRAP-NAIVE-ONLINE-MERGE`（朴素 m_c 基间隙 >88.7 溢出）口径互证，V1 fp64 ulp=0 机器证据一致；② **复杂度算术独立复算一致**——D-2 roofline 五项手工复算全部吻合（见机械复核节）；中间缓冲 GM 往返漏计核查：UB 重读项已单列（4.68K、3.8%、26× 余量），无漏计；③ **负向断言逐条复核成立**——「无 exp2 快档/vexpdif fp32-only」：simd.py/simd_inst.h 零命中 + VP-2026-0014 evidence 印证；「fp16-e 通道 50–60× 回退」：引 `D-LOGSUMEXP-MIGRATION-FIDELITY` measured 条目；「跨核 N-split 无益」：(max,sum) 非 additive 结构论证（R1 #12）+ 兄弟 Stage 4 `[DESIGN_LIMIT]` 同族记录；「wl3 M=4 并行度天花板」为结构事实（行任务粒度 4/56 核），缓解路径已列 Stage 4 观察项；④ **并列候选有裁决路径**——BLOCK_N 直保留 vs SimtVF 甜点档张力**非纸面拍板**：§1.6.3 L425 甜点对账 + §8.3 A/B 观察项（>5% 阈值 + 回写路径）+ §9.1 风险 9 三件套闭环（上轮问题 2 修复）。
- **调研→设计一致性**：§1.6.0 选定（单遍 online、chunk 粒度、N 轴）实际驱动 §1.4 算法描述、§3.1 公式拆解（以 O1–O5 优化后公式为输入）、§6 循环结构（chunk 串行 + 组展开，无两遍扫描残留）；**迁移一致性**：§0.5/§0.6 处置与调研选族一致——源 online 递推保留（R2 粒度重解释）、承载结构重设计（R1/R3–R6），§0.4 优化手段意图**逐项承接有说明**（快慢双路径 → 静态二分支 R5、N_padded 对齐 → BLOCK_N 紧凑 R2、fragment 跨 chunk 驻留 → alloc_var R3），无静默丢弃。
- **数学等价优化完整性**：§1.6.1 对选定算法系统审视（O1–O5 采纳项四要素齐全：原式 → 优化后公式 → 等价性论证 → 收益量化——O4 延迟归约 −28% chunk 指令、O5 merge 0.39%〔v1 修正计数口径〕等）；否决项（exp2 替换等）有 API 缺失佐证（维度 1 同口径核实）。
- **等价性论证独立推演**：逐项推演成立——base-2 指数换算为常数折叠（exp(x−m) = 2^((x−m)·log2e)，vexpdif 语义核对）、延迟归约（max 结合/交换律，组内先 max 后单次跨 lane 等价）、merge 计数修正（full+尾各一次——尾 chunk 同样执行完整合并链）；fp16/bf16 舍入影响由机器验证锚定：**verify_equiv 重跑 ALL EQUIV_PASS 且与内嵌表一致**（V1 fp64 全 ulp=0 / fp32 ≤1 ulp 含 32512 生产档；V6 14 行全档 ≤1 ulp；V4 位级 0 差；V7 饱和封闭 0 分歧），与 §8.2 精度标准一致。
- **向量化替代完整性（§3.3/§6 交叉核对）**：§1.6.2 覆盖方案全部循环/标量点——组内逐元素计算全为 `T.simd.*` 2048-bit 指令流、跨 lane 归约延迟单次（vcmax/vcadd）、无 §6 存在而 §1.6.2 未涵盖的标量循环；残留标量点仅 host 元数据/块索引/编译期静态二分支判定——**保留理由具体充分**（编译期常量非运行期标量、Python 展开即向量化形态本身）。
- **替代 API 佐证**：全部替代 API 实存（simd.py/simd_inst.h Grep 核实）+ 先例（testing/ascend 3 个测试文件 + examples/ascend 3 例 + 同族 single/SIMT 兄弟）；SimdVF 为用户显式契约（非热点降级争议）。
- **向量化轴与布局决策完整性**：§1.6.3 候选矩阵 8 行——规约类必选候选齐全（水平归约 N 轴 lane 映射 vs 垂直扫描 M 轴分别枚举 + 两遍/online 交互 #7/#8）；逐候选量化评分（整除性/尾 lane 浪费/累加链形态/UB 影响）；弃选全部量化理由（#2 >10×、#3 保真失效、#4 repack 无收益、#5 +75% chunk 数、#6 −28%、#7 VRF 26× 余量、#8 37× merge）——无「未考虑」式论证；**源码轴选择对照**已做（GPU N 轴并行是输入非结论，独立复核一致）；**弃选前提核对**：pow2 重划弃选前提「SimdVF 无 fragment」与方言源码/skill §2 一致、VRF 驻留弃选前提与 `CONST-vrf-capacity-a5` L149 一致、甜点注记**全量消费**（L425 对账——上轮问题 2 核心）；负向论断均有 pattern-library measured 条目或源码佐证，无与实测矛盾。
- **实验裁决模式**：主选与弃选判定裕度均不依赖未实证常数（结构性指令计数差/容量水位量级差）→ 不触发强制三件套；唯一实测敏感项（跨结构甜点张力）已具备主选 + 备选（16384 档，结构完整：仅改 BLOCK_N host 常数 + §4.5 单源重算，零语义成本）+ 裁决计划（代表 shape wl3/(33,33000)、do_bench(msprof)+npusim 指标、**>5% 判定阈值**、opt_log + pattern-library 回写路径）——合规。
- **布局决策与实现一致**：选定（x_ub (1, BLOCK_N=tile_n) dtype 紧凑双版本 + VRF 流式 stats alloc_var + N 轴 64 元素组粒度 + 延迟单次归约）与 §3.3 伪代码、§4 buffer 形状/UB 预算（130,048B 双版本）、§6 循环结构（内层组展开 = N 轴）三方一致；§3.1 以 O1–O5 优化后公式为输入，§6 无未论证标量循环——下游一致性成立。

## 附：微小观察（不构成 warn，无需修改）

1. §0.2 L54 / §11 引 `_primitives.py L176–183`（divisor 拒绝规则）指向规则注释区，规则代码实际在 L188–194——引用语义正确，行号粒度微偏。
2. V2/V4 及 §8.2 Boundary 第 3 档的 (33,33000) `tn=16640` 为**显式 config 泛化档**（与内嵌表声明一致、语境标注清晰）——非落点错误残留；保留该档扩大契约空间覆盖，恰当。
