# online_softmax 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: examples/pilot/online_softmax/DESIGN.md
- 任务类型: 迁移（migration-plain，GPU TileLang → Ascend A5）
- 源算子: examples/online_softmax/online_softmax.py（已亲自 Read 全文逐行核对）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 6（均为文档精度/引用行号/常数口径类，随维度说明列出，不阻塞 Stage 3；建议 Stage 3 开工前顺手修正或由 revision 统一处理）

## 机械复核（D-1/D-5）

### verify_equiv.py 重跑（2026-10-16，本检视执行）

命令：`cd examples/pilot/online_softmax && PYTHONPATH=/home/z00875089/tilelang-fork python3 verify_equiv.py`

| 验证项 | 重跑实测 | §1.6.1 内嵌表 | 一致性 |
|--------|---------|--------------|--------|
| V1（O3 去换底） | max_ulp=460437042, max_rel=6.162e-08, fp16 违反率 0, EQUIV_PASS | 4.6e8 / 6.16e-08 / 0 / EQUIV_PASS | ✅ |
| V2（O2 倒数乘） | max_ulp=1, max_rel=2.220e-16, 违反率 0, EQUIV_PASS | 1 / 2.22e-16 / 0 / EQUIV_PASS | ✅ |
| V3（O1 驻留链 vs 源链） | max_ulp=460437042, max_rel=6.162e-08, 违反率 0, EQUIV_PASS | 同 V1 注记 / EQUIV_PASS | ✅ |
| V4（O6 稳健合并） | max_ulp=11605, max_rel=1.316e-12, 违反率 0, EQUIV_PASS | 11605 / 1.32e-12 / 0 / EQUIV_PASS | ✅ |
| V4b（诊断：源朴素合并） | fp16 违反率 fp64=0.136364 / fp32=0.181818，角点发散 | 13.6% / 18.2% | ✅ |
| V5（备选 fp16-e 预估） | max_rel(\|y\|≥1e-3)=2.258e-03, 违反率 0, 5e-3 守卫内 | 2.26e-03 / 0 | ✅ |

**结论：全部采纳项 EQUIV_PASS，重跑输出与 §1.6.1 内嵌表逐项一致（含 ulp 数字与违反率），等价性机器验证有效。**

### design_calc_check.py（JSON checks 摘要）

命令：`python3 .agents/tools/design_calc_check.py --design examples/pilot/online_softmax/DESIGN.md`

| check | 工具输出 | 检视判定 |
|-------|---------|---------|
| ub_budget | **skip**（"no per-level buffer tables parsed"——§4.5 表无工具可识别的层级桶头） | 机械跳过，**人工已核**：x_ub (4,8192) fp16 = 65536B/版本 × 2 版本 = 131072B（128KiB）+ 标量暂存 ~1KB = 132096B ≈ 129KiB ≤ 194.4KB（SimtVF 216KB × 90%，CONST-capacity-a5 assumption，设计已披露条目与折扣口径）→ **pass** |
| l0c_budget | **warn**（"block 4x8192 x fp32 accum = 128KB vs L0C 256KB；WARN: block_M/block_N not multiples of 16"） | **工具误报，人工已核**：工具 `CUBE_PATH_RE` 命中了设计文本中**否定语境**的 API token（§1.5 "无 Cube/T.gemm"、§2.2 "T.MixedKernel 不适用"、§5.3 "仅约束 T.gemm 路径"），把纯 Vector 设计误判为 Cube 路径；解析出的 4×8192 是行块 BM×TILE（非 GEMM tile），本设计**无任何 L0C 分配**（§1.5/§3.5.1 明确），C0 分形对齐不适用（CONST-gemm-dim-constraints 仅约束 T.gemm 路径）→ **n/a→pass** |
| core_split | **fail**（"recomputed ceil(8192/4) x ceil(1000/8192) = 2048 blocks vs declared grid 56"） | **工具误报，人工已核**（已 Read 工具源码 `check_core_split`）：① 工具模型假设 grid = tile 数直接映射，**未识别 `T.Persistent` 常驻调度**——本设计为 `T.Kernel(56)` + `T.Persistent([num_tiles=2048], 56, core_id, num_stages=2)`，grid=56=vector_core_num 实查值、2048 tiles 多波（36.6 tiles/核）由调度器分配，正是 core-split-strategy 推荐的 persistent 形态（其 §2.4 明列 "persistent 化摊薄 launch 开销"；维度 3 的 fail 判据是 "grid 远超物理核数被串行调度却无 persistent 设计"——本设计恰为规避该反模式）；② 工具将 §5.3 举例句 "如 N=1000 → 行 2000B" 中的 1000 误解析为主 shape N。人工核对分核三要素齐全且自洽（§5.5：grid 划分含实查记录 / AIC-AIV 分工 / auto_schedule 声明）→ **pass**（误报已记入 RETROSPECTIVE Skill Flow Issues） |
| r3_metrics | **pass**（四口径齐全） | 与人工复核一致 ✅ |

## 检视详情

### 0. 源算子理解与迁移分析 — pass

已亲自 Read `examples/online_softmax/online_softmax.py` 全文（61 行），逐项核对：

- **语义理解正确性 — pass**：§0.1 与源码逐行一致：① lse 递推——源码 L35 `lse[i] = max_x[i]*scale + T.log2(T.exp2(lse[i] - max_x[i]*scale) + sum_exp_x[i])` 与 §0.1/§0.3 步骤 6 的公式逐项对应（首块 −inf 幺元 L23 → 退化为 m·s+log2(S)，§0.1 表述正确）；② exp2 换底——源码 L14 `scale = 1.44269504 # log2(e)` 确为截断字面量（完整 log2e=1.4426950408889634，绝对偏差 8.89e-10；§0.1 称 "~8.9e-10 相对量级" 实为**绝对**偏差、相对约 6.2e-10——措辞小疵，见建议 5，不影响任何结论）；③ 两遍结构——L25-35 pass-1 / L37-43 pass-2 重读 X，§0.3 步骤 7 标注 "重读 GM" 正确；④ threads=128（L16）、BLOCK_M=1/BLOCK_N=8192 默认参数（L8）、8192 元素/128 线程=64 elem/thread、主 shape 下 Pipelined 退化 1 次迭代——全部与源码一致；⑤ dtype 语义——`max_x` 为 fp16（L20 `dtype`），`exp_x/sum_exp_x/lse` fp32（L19/L21/L22 `accum_dtype`），y fp16 赋值降精度（L41）——§0.1 描述精确；⑥ 边界语义（NaN 传播/+inf 行 NaN/全 −inf 行 NaN）与 "以 golden 实测为准" 的谨慎表述合理；⑦ §0.2 I/O 与源码签名一致（X/Y (M,N) fp16，L53 `X.softmax(dim=1)` 为源仓基准，行号引用准确）。
- **算法解读完整性 — pass**：§0.3 的 9 个步骤与源码计算语句一一对应无遗漏无臆造（L23→步骤1，L26→2，L27→3，L29-30→4，L32→5，L34-35→6，L38→7，L40-41→8，L43→9）；数据流图覆盖全部 6 个 fragment buffer（x/y/lse/max_x/exp_x/sum_exp_x）与 "全程寄存器驻留、X 二遍 GM 读、Y 一遍写" 的访问模式正确；host 侧逻辑（randn 构造/softmax 对比 1e-2/do_bench、无 pass_configs）已列出且与 L48-61 一致。
- **优化手段覆盖完整性 — pass**：§0.4 的 9 项在源码中全部可定位（exp2 换底 L14/L30、除法消除 L41、online lse L34-35、fragment 驻留 L17-22、Pipelined L25/L37、warp 归约 L27/L32、BLOCK_M=1×threads=128 L16、fp32 累加+fp16 max L12/L20、二遍重读 L37-43），算法层/实现层划分正确（online lse 判算法层、fragment 驻留判实现层等），未漏识别。
- **耦合性判定合理性 — pass**：§0.5 逐项有四态处置+依据，与 backend-glossary §1 概念迁移锚一致（warp shuffle→reduce intrinsic：设计用 ascend 方言 `T.reduce_max/reduce_sum`，reduce_op.py L95/L111 签名核实、fp32 属 §10.11 硬件直连白名单〔skill L735-739 亲核：仅 fp16/fp32/int32/uint32 直连 asc_reduce_*〕；cp.async/Pipelined→T.Persistent num_stages；shared→UB）。**重点核对项——exp2 换底判「舍弃」**：依据经亲自查证成立——`src/ascend/codegen/intrin_rule_ascend.cc`：`REGISTER_ASCEND_INTRIN("exp")` 与 `("exp2")` 均注册、fp16 走 "h"+name 派发（AscendMath case 16，L31-32）；`src/ascend/transform/estimate_latency.cc` L1267-1272：`op_name == "exp2" || op_name == "log2" || op_name == "exp" || ...` 同入 `special_func_throughput` 分支——**A5 上 exp/exp2 同档证据确凿**，换底无单位收益反增 1 乘/elem，且优化意图（廉价 exp 求值）由 O3 原生 exp 显式承接（非静默丢弃）✓。**唯一瑕疵**：判定统计行 "等价替换 5 项" 与表内实际 4 项（fragment 驻留/warp 归约/Pipelined/host 环境）不符，括号内自列也仅 4 项（建议 1）——计数笔误，12 行处置本身逐项合理（#8 fp32 累加折叠进 "计算语义" 行保留、max fp32 化在 §1.4 差异⑥ 有显式承接说明，无静默丢弃）。
- **NPU 重设计可行性 — pass**：R1/R2/R3 三项均具「源方案 → NPU 新算法 → 语义保持论证」三要素。R1（一维 `T.Kernel(56)` + `T.SimtVF(threads=1024)` + persistent 行批）满足一维 grid/无 threads= 约束，per_token L44/L45/L51 同构先例亲核属实；语义论证成立（行间独立；max 精确；fp32 树形 sum 误差 ≤ 13×2^-24 ≈ 7.7e-7 << 1e-2，树深 13 = log2(8192) 复核正确）。**R2 重点核对（驻留单读 vs 源两遍流式在 N≤32768 的等价性）**：独立推演 `y = e·(1/S) ≡ e/S ≡ exp(x−m−lnS) ≡ exp2(x·s − m·s − log2 S) ≡ 源式 exp2(x·s − lse)`（lse = m·s + log2 S 代入）——恒等成立，差异仅 fp32 舍入路径，V3 机器验证（源链 vs 驻留链 fp16 输出 max_rel 6.16e-08、违反率 0，本次重跑复核）支撑；驻留上限 N≤32768 复算成立：VRF 口径 BM·TILE·8B ≤ 256KB → BM=1 时 TILE ≤ 32768（pow2 边界，N=32769 → TILE=65536 → 512KB 超限），UB 口径 2·BM·N·2B ≤ 194.4KB → BM=1 时 N ≤ 49766——绑定约束为 VRF，上限值正确（但 §5.2 把系数误写为 "6B"，见建议 2）。**fp16 输入下 lse/max 精度行为**：源 max_x fp16 → 设计 m_frag fp32——fp16 值集的 max 仍是 fp16 可表示值、fp32 承载无损，论证成立；x−m 的 fp32 舍入与源 x·s−m·s 的 fp32 舍入同阶（大 max 下微小差值的丢失两条路径行为一致），V1/V3 角点行（subnormal/深度下溢）实证覆盖。R2 泛化路径保留源 online 结构且按 O6 稳健化——O6 必要性证据链独立复核成立：fp32 exp2 溢出阈 arg>128 → 块间 max 间隙 > 128/log2e = **88.7** ✓（设计数字精确）、fp16 值域间隙上限 65504−(−65504)=**131008** ✓、V4b 重跑复现源式角点发散（13.6%/18.2%）✓、flash 形式全部 exp 参数 ≤0 无溢出 + 哨兵钳位消 NaN 的论证数学成立 ✓。R3 纯调度变换不动数值路径 ✓。
- **设计与决策一致性 — pass**：§1.4 主路径 = R1/R2/R3 落地形态（六条结构差异逐条注明 §0.6/§1.6 出处）；§3.1 公式 = §1.6.1 优化后公式；§3.3/§6 无 exp2/无两遍重读/无 threads=128 残留；源 online 结构仅存于泛化路径（与 §0.5 行 3 "保留作为泛化路径" 一致）。无 "§0 判重设计、§3 照抄源方案" 脱节。
- **golden 独立性 — pass**：§8.1 golden = `X.softmax(dim=1)`（torch on NPU），即源仓自身测试基准（源文件 L53/L55 亲核属实），以 §0.1 语义为依据、未复刻 §0.6 NPU 数据流（驻留三段式/e 复用/倒数乘均不出现在 golden 中），验证独立性成立。

### 1. API 可行性 — pass

- §3.2/§3.4 全部 API 亲核佐证成立：`T.Kernel(N)` 一维无 threads（skill §2 L64-65 陷阱条文 + vecadd L21）；`T.Persistent(domain, wave_size, index, group_size=8, num_stages=0)` 签名与 loop.py L82 **精确一致**（设计按关键字传 num_stages=2，group_size 缺省，调用形态与 per_token L45 同构）；`T.SimtVF(threads=)`（rmsnorm L54、per_token L51 threads=1024）；`T.alloc_shared/alloc_fragment`（rmsnorm L41/L43/L55）；`T.copy` 2D 行块（per_token L50）；`T.reduce_max/reduce_sum(frag, out, dim=1, clear=True)`（ascend 方言 reduce_op.py L95/L111 签名精确匹配；per_token L73 SimtVF 内 dim=1 用例；testing/ascend/language/test_tilelang_ascend_reduce.py fp32 sum/max/min 回归亲核 L67-75）；`T.cast`（test_tilelang_ascend_simtvf_cast.py L96 精确命中）；`T.annotate_buffer_versions`（rmsnorm L47/per_token L49/vecadd L25）；`T.if_then_else` 掩码载入（rmsnorm L56-57 同构——含 "fragment pow2 填充、DMA 紧凑" 的 L42 注释先例）；另以 hasattr 实查复核 ascend 方言导出：exp/exp2/infinity/if_then_else/cast/reduce_*/Persistent/SimtVF/Kernel/alloc_*/annotate_buffer_versions/copy/Parallel/serial/ceildiv/fill 全部 True。
- 编程模式匹配：simt + auto_schedule: yes，全文无手动 flag 混写 ✓。SimtVF 优先 ✓；SimdVF 降级仅作为 Stage 4 热点预案且给出具体路径（`T.simd.vexpdif`——simd.py L1182 `def vexpdif(src0, src1, mask=None)` 亲核存在；flash_attention/core.py L130-133 e=S.vexpdif(p,max)、L169-171 alpha 重缩放用例亲核属实）。
- 引用行号小偏移（内容佐证全部属实）：per_token `T.reduce_max` 实际在 **L73**（设计两处引 L68）、2D 行块 `T.copy` 实际 L50（引 L49）、Persistent+versions 三件套实际 L45-49（引 L45-47）——建议 3。

### 2. 内存层级规划 — pass（1 项 warn 注记）

- 搬运路径：GM→UB（MTE2 `T.copy` 2D 行块）→ VRF（SimtVF masked 载入）→ UB 原位覆写（VALU）→ GM（MTE3）——全部在 skill §6 支持集内；无 L1/L0 段（纯 Vector，§1.5 明确）；无手写 NZ；无中间量落 GM ✓。原位覆写 reuse_input 形态有 rmsnorm L72-78 亲核先例（`if reuse_input: x_ub[...] = ...` → `T.copy(x_ub, Y[...])` 完全同构），WAR 依赖已列入 §7.2 自动同步边界，且 §9.2 备有独立 y_ub fallback（128+64=192KB ≤ 194.4KB 边缘通过，已如实标注 "边缘"）。
- UB 预算：x_ub 双版本 131072B + ~1KB 暂存 ≈ 129KiB ≤ 194.4KB（216KB SimtVF × 90%，CONST-capacity-a5 条目已亲核：248−32=216KB、assumption、90% 折扣纪律），**buffer versions 已计入**（skill §4.3 要求）✓，条目与折扣口径已披露 ✓。机械项 ub_budget skip 已人工复核（见机械复核段）。§4.5 "总计 ≈132KB" 是 132096B 的十进制读法（=129KiB），单位口径混用但方向保守不改变结论（建议级附注）。
- masked 载入越界读风险已核：x_ub (BM,N) 紧凑而 x_frag (BM,TILE) pow2 填充时，`if_then_else(i<N, cast(x_ub[r,i]), −inf)` 对 i≥N 产生跨行/末行越界读——rmsnorm L56-57 为**同构已验证先例**（select 掩掉非法值、UB 分配对齐吸收），主 shape N=TILE=8192 完全无此路径，非 pow2 N 由 L1 用例（1000/8191）覆盖 ✓。
- **warn 注记（VRF 预算，建议 4）**：设计口径 "VRF 256KB/1024 threads = 64 reg/thread"——① VRF 256KB 容量在 pattern-library constants.md 与 tilelang-ascend skill 中**均无条目**（CONST-capacity-a5 只覆盖 UB/L1/L0A/L0B/L0C；CONST-valu-model 只有寄存器宽度 256B），属未文档化常数，设计未按 assumption 显式标注；② 主 shape x_frag+e_frag = 256KB 合计恰达该假设上限 **100%**（§4.5 求和口径），零余量；③ "rmsnorm 80 reg/thread 先例" 是 **per-thread** 口径（其 rows=4/threads=256 配置总量 ~80KB），不能佐证 1024 threads 下 256KB 总量驻留。缓解已充分：§9.1-3 明示 spill 风险 + 两条具体退路（BM=2/threads=512 总量减半、fp16-e 备选 e_frag 减半），且 spill 只劣化性能不损正确性（L0 门禁兜底）；实际编译器对 x_frag（步骤③后死）与 e_frag 的寄存器复用可能进一步压低峰值。**不阻塞**，但 Stage 3 编译后应第一时间核查 spill（编译告警/npusim），RETROSPECTIVE 已提 VRF 容量常数缺口。

### 3. Tiling 策略 — pass

- Block 划分：BM=4（DMA 合并 64KB/tile、VRF 预算内最大行批、整除 M=8192）+ TILE=next_pow2(N)=8192，理由具体 ✓；grid 一维 `T.Kernel(56)` ✓。
- 分核三要素（core-split-strategy §1 逐项）：① grid 划分——num_blocks=56 有**真机实查记录**（查询代码 + 返回值 Ascend950PR_9579 / cube_core_num=28 / vector_core_num=56，与调度方提供的真机信息一致），block 数与核数关系明确（persistent 单波常驻、2048 tiles 多波 36.6/核、惯例初值 32 被实测取代并注明 CONST-arch-a5 条款）✓；② AIC/AIV 分工——纯 Vector 仅占 AIV（56），无 Cube 工作、pure kernel 由 codegen 自动判定（core-split §1 ② 原文一致），无 dual_copy 所有权问题 ✓；③ 同步策略——auto_schedule: yes + 5 条数据依赖边界枚举 + 调度提示位置（T.Persistent num_stages=2；不用 T.Stage/T.Task 有说明）✓。
- 对齐约束：无 GEMM → C0/NT 不适用（§3.5.1/§5.3 明确）✓；GM→UB 行 16KB 32B 对齐 ✓；非 32B 行（N=1000）正确性不受影响、带宽档降级列 §9.3 观察项 ✓；向量宽度 TILE % 128（fp16 lane）= 0、% 64（fp32 lane）= 0 ✓。
- 非整除处理：M 尾块由 BM 自适应（4/2/1 必有除数 1）消除、N 尾由 masked 载入/存储吸收、泛化路径 chunk 尾 masked+哨兵 ✓。核内串行边界全部 host 静态化（lru_cache per-shape）✓。
- 机械项 core_split fail 为工具误报（未建模 T.Persistent + 误解析 N=1000），人工复核三要素自洽，详见机械复核段。
- 附注（建议 6）：§5.4 尾波不均衡 "~2.4%" 复算应为 ~1.2%（37 波 × 56 核 = 2072 槽位，空闲 24 → 1.16%）——量级不变，Stage 4 调优提示仍成立。

### 4. 技术约束检测 — pass

- 一维 grid ✓；无 `T.Kernel(N, threads=)`（源 threads=128 已按 §0.6 R1 重设计为 SimtVF 线程域，§9.2 明确列为常见错误 + skill §10 陷阱 2 条文亲核 L43/L64-65）✓。
- GEMM 分形/NT/L0C：不适用（纯 Vector，无 T.gemm/alloc_l0*/dual_copy 正向使用——机械项 l0c_budget warn 系否定语境 token 误报，见机械复核段）✓。
- ascend-constraints §1 已知限制表逐条对照（§3.5.1）：OOB padding 互斥条款理解正确（不用 DMA padding、计算侧掩码，理由——行完整存在无需 32B 填充——与 skill §6.1 opt-in 语义一致）✓；flag 预算不占用 ✓；核数实查 ✓；静态边界 ✓；reduce dtype 白名单（§10.11 亲核：fp16/fp32/int32/uint32 直连）——本设计归约全 fp32 ✓，bf16 扩展路径（先升 fp32）预留正确 ✓。

### 5. 循环与同步 — pass

- §6.1 循环类型明确：persistent 调度循环 + T.Parallel 向量化（×4 处）+ T.reduce_*（×2）+ 泛化路径 T.serial×2（online 循环携带依赖，结构必需且主 shape 不执行）；与 §3.3 伪代码逐条对应 ✓。
- §7 与 auto_schedule: yes 声明一致：无手动 flag；5 条自动同步边界（MTE2→VALU、VALU→MTE3、版本交替、SimtVF 跨线程 hazard→自动 asc_syncthreads〔skill §2.2 + test_ascend_thread_sync.py 存在亲核〕、reduce 终值→inv/y 同引擎顺序）枚举完整 ✓；pass_configs 不传 TL_ENABLE_AUTO_SCHEDULE（正确——仅完全手动核需要）✓。
- §6.4 尾块三分支（行尾/列尾/泛化 chunk 尾）齐备 ✓。

### 6. 验证方案 — pass

- §8.1 golden：torch on NPU（`assert X.device.type == "npu"`）+ `X.softmax(dim=1)` = 源仓基准直移植，独立于 NPU 算法 ✓。
- L0 门槛：具体规则 shape（[8192,8192] 主 shape + [128,4096] smoke + 源仓对照方式）、fp16、`torch.testing.assert_close(rtol=1e-2, atol=1e-2)` 符合 AGENTS.md dtype 分档 ✓。
- L1/L2/Boundary 分层计划完整：BM 自适应/非 pow2 掩码/非 32B 行/泛化路径 [128,65536]/路径切换边界（32768/32769）/±inf/NaN（equal_nan 对照）/subnormal/深度下溢/重复调用确定性/lru_cache 生效——覆盖面与 §0.1 边界语义、§1.6.1 角点、§9 风险点一一呼应 ✓。L2 的 ±inf/NaN 预期表与 §0.1 语义及 verify_equiv V3 角点实证一致 ✓。

### 7. 完整性与一致性 — warn（建议级，不阻塞）

- 章节齐全：§0（0.1–0.7 全）+ §1–§10 共 11 章 ✓；占位符扫描零命中（grep 复核；"待实现（Stage 3）" 为交付清单状态标注、非占位符）✓。
- 同类引用真实：§3.5.3 全部 6 个路径存在（ls 亲核）✓，关键行号内容属实；per_token 3 处行号偏移（L68→L73、L49→L50、L45-47→L45-49，建议 3）。
- 内部一致性：API 映射（§3.2）↔ 内存规划（§4.3 buffer 形状）↔ Tiling（§5.2 参数）三方一致 ✓；发现 3 处文档精度问题（均建议级）：① §0.5 统计行 "等价替换 5 项" vs 表内 4 项（建议 1）；② §5.2 VRF 约束系数 "BM·TILE·6B" 与 §4.5 的 x_frag+e_frag=8B/elem（256KB 合计）不一致——推导出的上限 N≤32768 按 8B 口径复核**正确**（6B 口径反而得 43690 与声明矛盾），系数为笔误（建议 2）；③ §0.1 scale 截断 "8.9e-10 相对量级" 实为绝对偏差（建议 5）。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性 — pass**：1.6.0（调研）/1.6.1（等价优化）/1.6.2（向量化替代）/1.6.3（轴与布局）/1.6.4（实验裁决三件套）五个子节齐全，深度与算子复杂度匹配（完整调研档）。
- **算法调研存在性与完整性 — pass**：R1 候选表 8 行**含基线**（源两遍流式）且 algorithm-candidates.md ALG-softmax 命中行候选全部在表（减 max→候选1、log-softmax 直出→候选7、exp→exp2·log2e→候选2、online 变体→候选6/R2、三遍/两遍/单遍→R3 表、水平 vs 垂直→§1.6.3、整行 UB 驻留→选定方案；attention mixed 融合不适用非 attention 场景）；另含超出参考表的候选（倒数乘/e 驻留）。R2 有**明确结论**（"有，三个变体"，且正确引用 "源算法已用 online 手段是直接证据" 的判据——与 algorithm-research.md §3 R2 回答方式②原文一致）。R3 四口径齐全（FLOPs/GM Bytes/扫描遍数/中间缓冲，基线在表、口径逐列写清）。R4 逐候选六项评估、负向淘汰附依据。调研范围声明明确（互联网未使用有结构判据说明——本地候选表已覆盖 softmax 族全部已知变体，经查证属实）。
- **调研结论正确性（独立复核）— pass**：① 等价性：选定算法与源语义恒等（维度 0 已独立推演 `e·(1/S) ≡ exp2(x·s−lse)`），fp16 舍入影响由 V3 机器验证（重跑复核）+ §8.2 容差 1e-2 一致；② 复杂度独立复算全部吻合：基线 GM 6B/elem（读2×2B+写2B）→ 选定 4B（−33% ✓）、special 2→1/elem（−50% ✓）、遍数 3→2 ✓、无中间缓冲 GM 往返漏计（选定方案中间量零落 GM、基线全 fragment ✓）；D-2 估算复算：读 134.2MB/(56×100)=24.0K ✓、写 134.2MB/(56×115)=20.9K ✓（CONST-mte-bandwidth-model 条目 100/115 B/cycle 亲核一致）、计算 67.1M×(6/51+1/8)/56=290K ✓（CONST-valu-model 51/13/8 亲核一致）、比值 6.5–12× ✓、主选/基线 VALU 0.57× ✓、流量 0.67× ✓；special op 粒度不确定性的双解释披露诚实且结构结论对两种解释均稳健 ✓；③ 亲和性：容量淘汰引用 CONST-capacity-a5 条目（〔D-2〕要求）✓，与 ascend-constraints 无矛盾；④ **负向断言复核**：核心断言 "A5 上 exp/exp2 同档" 经两处源码亲核成立（intrin_rule_ascend.cc exp/exp2 均注册 + estimate_latency.cc L1267 同 special_func 列表）——该断言方向为 "舍弃源优化" 而非 "否决候选能力"，证据等级充分；"无新增候选空间" 对照 ALG-softmax 命中行逐候选核对无遗漏（softmax 族已知变体：三遍/两遍/单遍 online/驻留式/换底/log 域全部在表）；候选 6 弃选理由量化成立（1 cmp@51 档 = 0.0196 cycle/elem ÷ 总计算 0.2426 = 8.1% ✓ "收益 ~8%" 复核精确，vs 引入行内块序依赖+重舍入——弃选合理且列 Stage 4 观察项）；⑤ 并列裁决：fp32-e vs fp16-e 裕度落在未实证常数（special 吞吐 dtype 粒度）→ **未纸面拍板**，进 §1.6.4 实验裁决 ✓。
- **调研→设计一致性 — pass**：§1.4 主路径 = 选定 "驻留三段式"（七步与 R4 表形态一致）；§3.1 四步公式 = §1.6.1 优化后公式原文；§6 循环结构 = 单遍驻留（无两遍扫描残留，泛化路径才回退 online——与调研 "N>32768 回退" 结论一致）；迁移任务专项：§0.5/§0.6 处置与调研选族一致（#9 二遍重读判重设计的依据即 §1.6.0 R3/R4），§0.4 九项优化意图**逐项承接有说明**（§1.6.0 调研结论行末的承接清单：换底→O3/除法消除→O1+O2/online→泛化路径/寄存器→VRF/流水→Persistent/归约→T.reduce_*），无静默丢弃 ✓。
- **数学等价优化完整性与等价论证 — pass**：O1–O6 六项全部四要素齐全，逐项独立推演成立——O1（恒等式 + V3 机器验证）、O2（≤1 ulp + V2 max_ulp=1；div 13 档→mul 51 档收益量化与 estimate_latency.h 亲核值一致）、O3（换底逆变换恒等 + 同档证据 + 截断误差消除 + V1；注：V1 max_rel 6.16e-08 ≈ 相对截断误差 6.2e-10 × |x−m|~100 放大，量级自洽）、O4（e^11>65504 溢出阈复核正确：ln 65504 = 11.09）、O5（−inf/0 幺元 + NaN 语义 + rmsnorm 同构先例）、O6（溢出阈 88.7 = 128/log2e 独立复算精确、fp16 间隙域 131008 ✓、V4b 发散证据链重跑复现、flash 形式 exp 参数 ≤0 论证成立、哨兵钳位消 NaN 成立；对良性数据与源式恒等由 V4 max_rel 1.32e-12 支撑——角点行为向 golden/torch 对齐属正当改进且 §8.2 L2 以 golden 实测为准）。收益量化全部有数字（cycle/elem、B/elem、op 数）✓。
- **向量化替代完整性 — pass**：§1.6.2 十行与 §3.3 伪代码/§6.1 循环表**交叉核对无遗漏**（①-⑥ 步 → #1-#6，persistent → #7，泛化 serial → #8/#9，host → #10）；#1/#2/#3/#4/#6 全向量化且有 examples/testing 双佐证；保留 3 处理由逐项充分——#5 tile 级每行倒数（依赖 reduce 终值、占比 1/TILE，且已用 T.Parallel(BM) 行维向量化——非 "实现简单" 类借口）、#8/#9 泛化路径 online 循环携带依赖（在线算法本质结构，占比 1/8192，主 shape 不执行）✓；§6 无未论证的逐元素标量循环 ✓。替代 API 全部 SimtVF 优先（T.Parallel 自动向量化），无过早 SimdVF ✓。
- **向量化轴与布局决策完整性 — pass**：规约类必选候选齐全——水平归约（#1 主选）vs 垂直扫描（#2）两种 lane 映射**分别枚举** ✓，另含跨核分块变体（#3）与泛化分块（#4）；两遍/online 交互在 R2/候选 6/#4 覆盖 ✓；轴质量评分五项量化（整除性/尾 lane 0%/累加链 stride-1/repack 无〔I/O 原生行主、核内同构——无重排即无 repack 代价机制误述风险〕/UB 影响 128KB）✓；弃选量化理由成立——#2：lane 利用率 BM/128 < 4% → >25× 劣化（128/4=32 上界，保守标注 "估"）+ repo 无先例；#3：行已整驻留（16KB）跨核拆分零收益 + partial 交换与同步净损（遍数 2→3 口径 +50%）+ "容量不构成约束" 结构性论据决定性 ✓；**源码轴选择对照已做**（GPU threads=128 沿 N → 轴结论一致但承载结构完全重设计，"GPU 轴是输入非结论" 明示）✓；负向论断与 pattern-library 无矛盾（constants.md 亲核：无布局/转置实测条目与之冲突）✓。
- **实验裁决三件套 — pass**：判定裕度依赖未实证常数（special 吞吐 dtype 粒度——经查 pattern-library 确无 fp16/fp32 粒度实测，不属 "漏查已有数据"）→ 三件套齐全：备选 fp16-e **结构完整可实现**（dtype 路径 h-前缀 intrinsic 佐证 intrin_rule L27-32 亲核、buffer/UB/循环/分核按新粒度说明、精度预估 V5 2.26e-03 在守卫内）；裁决计划含代表 shape（[8192,8192] 驻留 + [128,65536] 泛化）、指标（do_bench(msprof) + max_rel_err）、**明确阈值**（<0.90× 且 ≤5e-3）、回写路径（CONST-valu-model Tier 0 + repro、附录补记）；按 shape 分派可行性已评估（lru_cache 分派）✓。
- **布局决策与实现一致 — pass**：§1.6.3 选定（x_ub (BM,N) 紧凑 + fragment (BM,TILE) pow2、N 轴、无 repack）与 §3.3 伪代码（Parallel 内层维 = TILE、buffer 形状一致）、§4.3（buffer 表同形状）、§6（向量化轴 = N）三方一致 ✓；§3.1 以优化后公式为输入 ✓。

（建议级注记共 6 条，已全部内嵌于上列各维度说明中：① §0.5 统计 "等价替换 5 项" 应为 4 项〔维度 0/7〕；② §5.2 VRF 系数 "6B" 应为 8B/elem、上限 N≤32768 结论不变〔维度 0/7〕；③ per_token 行号引用偏移 L68→L73、L49→L50、L45-47→L45-49〔维度 1/7〕；④ VRF 256KB 为未文档化常数且主 shape 达假设上限 100%，Stage 3 编译后首查 spill〔维度 2〕；⑤ §0.1 scale 截断 8.9e-10 为绝对偏差、相对 ~6.2e-10〔维度 0/7〕；⑥ §5.4 尾波 ~2.4% 应为 ~1.2%〔维度 3〕。均不阻塞，供 Stage 1 revision 或 Stage 3 实现时顺手修正。）
