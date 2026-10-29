# _logsumexp_kernel_tiled 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: `examples/logsumexp/_logsumexp_kernel_tiled/DESIGN.md`（1003 行，含 §0 迁移分析与 §1.6.0–§1.6.4）
- 任务类型: 迁移（migration-harness，逐函数——logsumexp tiled 变体）
- 源算子: `examples/TileOPs/tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py`（`_log_sum_exp_fwd_kernel_tiled` L101–189）；调用面/启发式依据 `logsumexp.py` + `_primitives.py`（`compute_tile_n` L116–203）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 5

## 机械复核（D-1/D-5）

### verify_equiv.py 重跑（等价性，D-1）

命令：`PYTHONPATH=/home/z00875089/tilelang-fork python3 examples/logsumexp/_logsumexp_kernel_tiled/verify_equiv.py`（torch CPU，2026-10-16 执行）——**全部通过，与 §1.6.1/§1.6.4 内嵌结果表逐项一致**：

| 段 | 重跑结果 | 设计内嵌值 | 一致性 |
|---|---|---|---|
| V1（7 组 shape×划分组合：(33,33000)/(33,32768)/(4,102400)/(4,32768)/(3,65540)/(2,32513)/(5,5000)） | fp64 与 fp32 位级一致 ulp=0 | §0.6 R2/§1.6.1 O2「位级一致（ulp=0）」 | ✓ |
| V3 | ≤1 ulp，max_rel 6.471e-08 | 6.47e-08 | ✓（逐字） |
| V5（角点全谱，含 fp16 跨 chunk max-gap 131008） | 全 EQUIV_PASS（部分 −inf 完全中性；全 −inf 行 NaN 为源固有语义） | §1.6.0 R1 #11/§9.3 | ✓ |
| V6（BLOCK_N 256…16384 × tile_n 16640/25000/32512 划分扫描） | ≤1 ulp，max_rel 6.32e-08 | §1.6.1 表 L333「6.32e-08」 | ✓（逐字） |
| V7（fp16-e 备选模拟精度） | 3.645e-05（1024×32768）/ 1.402e-05（4×102400），GUARD_PASS @1e-2 违反率 0 | §1.6.4「3.6e-5/1.4e-5」 | ✓ |
| 末行 | `ALL ADOPTED-ITEM CHECKS: EQUIV_PASS` | — | ✓ |

### design_calc_check.py（算术复算，D-5）

命令：`python3 .agents/tools/design_calc_check.py --design examples/logsumexp/_logsumexp_kernel_tiled/DESIGN.md`。JSON checks 摘要与逐项消费（伪影消解依据 = 亲读解析器源码，方法论同 single 变体 REVIEW 判例）：

| check | 工具输出 | 检视判定 | 依据 |
|---|---|---|---|
| `ub_budget` | fail（576.2KB / 17 entries） | **解析伪影，人工复算 pass** | 解析器 `check_ub_budget`（design_calc_check.py L108 起）把 §4.5 节内**全部表行跨 workload 求和入单桶**（多 workload 表 + VRF 表 17 行混加）。按 workload 分桶人工复算：wl2 ≈ 72.1KB、fp32 档 ≈ 136.1KB、显式 bm16 ≈ 136.1KB——均 ≤ 194.4KB（216KB×90% 折扣，CONST-capacity-a5 口径设计已披露） |
| `l0c_budget` | warn（block 4x4） | **解析伪影，n/a** | 纯 Vector 算子无 L0C/Cube 路径；解析器从文本抓取「4x4」类 token 误判 |
| `core_split` | fail（推算 1 block vs 声明 56） | **解析伪影，人工复算 pass** | 解析器 `check_core_split`（L259 起）正则抓示例数字（M=1, bm=1, N=1, bn=4096）反推 grid=1；实际 `T.Kernel(56)` grid = vector_core_num 实查值（§5.5 含查询代码与返回记录），num_tiles 是 `T.Persistent` 调度迭代数而非 grid——persistent 形态该解析器不识别。人工按 core-split-strategy.md §1 三要素核对通过（见维度 3） |
| `r3_metrics` | pass | ✓ | R3 复杂度表四口径存在 |

无 skip 项。两处 fail 与一处 warn 均为解析器口径伪影（与 single 变体检视时确认的伪影形态一致——本任务为第二次复现，已记入 RETROSPECTIVE 供工具修正），**不构成维度 2/3 fail**；人工复算证据见对应维度。

### 检视补充实证（reviewer probe，真机）

设计尾 chunk 使用 `(SUB, TILE_R)` 小 fragment（TILE_R = next_pow2(rem) 可远小于 BLOCK_N），而 threads 按 full chunk 取 `min(1024, SUB·BLOCK_N)`——(33,33000) 落点下尾 chunk 为 **(1,256) fragment @ SimtVF(threads=1024)，元素数 256 < 线程数 1024（replication=4）**。设计所引先例（online_softmax L150–152、single L150–155）支持的是 masked 载入模式本身，但两先例的尾 chunk 均保持全尺寸 fragment（extent ≥ threads）——该配置类无既有 measured 先例，检视以最小探针真机实测闭合：

- 探针完整复刻设计尾 chunk 计算体（静态 rem 切片 copy → `T.And(r<rows, i<rem)` masked 载入 pad=−inf → `T.reduce_max(dim=1, clear=True)` → exp 原位 → `T.reduce_sum(dim=1, clear=True)` → lse/sum 写回），SUB=1, TILE_R=256, rem=232, threads=1024，fp16 输入 (1,232)。
- 结果（Ascend950PR_9579，2026-10-16）：编译运行通过；`sum ratio=1.000000`（复制 lane 无双重计数）；lse=5.82410145 与 fp64 参考 5.82410145 一致——**PROBE_PASS**。
- 源码机理佐证（亲读）：`src/backend/common/op/reduce.h` 对复制布局显式建模（`ReplicateExtent` L92–101、`ThreadReduceStep{extent,scale}` L396–409）；`CheckAllReduceWidth`（L188–213）要求 scale 与 logical_width 均为 2 的幂——本配置 extent=256（pow2）、scale=4（pow2）、reducing_threads=1024，约束满足；`AllReduceNeedsWorkspace`（src/ascend/op/reduce.cc L21–26，reducing_threads>32）→ ub_reduce workspace 路径按 scale-stride 折叠复制槽位，SUM 语义正确。

结论：尾 chunk reduce 在 extent<threads 下受支持且数值正确（详见维度 5/8 与建议 5）。

## 检视详情

### 0. 源算子理解与迁移分析 — warn

检视方式：亲读源算子 `_log_sum_exp_fwd_kernels.py` tiled 变体（L101–189）全文、wrapper `logsumexp.py` 全文、`_primitives.py`（`compute_tile_n` L116–203、`align_up`、MAX_SINGLE_TILE_COLS=32512、216KB 预算），与 DESIGN.md §0.1–§0.7 逐项核对；wrapper 落点以独立复刻脚本逐 shape 复算。

| 检查项 | 判定 | 证据 |
|---|---|---|
| 语义理解正确性 | pass | §0.1 行级 `lse = max + log(Σexp(x−max))`（flash 合并式）、fp32 内部计算/dtype I/O、flatten(dim) 落点语义与源 L101–189 一致（running row_max/row_sum 跨 chunk 递推、末段 `prev_max + log(row_sum)` cast 回 dtype）；§0.2 kernel 签名 `(M,N,dtype,tile_n)(block_m)(x)→y` 与 wrapper dispatch（tile_n≠0 → tiled）一致 |
| 算法解读完整性 | pass | §0.3 步骤分解覆盖源全部计算语句（chunk 载入→行 max→s_run 重缩放→exp→行 sum→lse 收尾→cast 写回），无遗漏无臆造；host 侧逻辑（align_up/num_tiles/_needs_mask 常量折叠、lru_cache JIT、threads 契约）已在 §0.2 L105 列出 |
| 优化手段覆盖完整性 | pass | §0.4 识别 8 项且逐项标注硬件依赖：online 递推、chunk 分块（tile_n 256 倍数——GPU SMEM 对齐驱动）、`_needs_mask` 快慢双路径、SMEM staging、稳定化（减 max）、warp 树形归约（`T.reduce_*`）、合并式标量、行批×grid 并行——与源码显式优化一一对应 |
| 耦合性判定合理性 | pass | §0.5 14 项四态处置（保留 4 / 等价替换 4 / 重新设计 5 / 舍弃 1）逐项有依据；对照 backend-glossary §1 概念迁移锚：shared→UB（`T.alloc_shared`）✓、warp shuffle 归约→`T.reduce_*`/AscendAllReduce ✓、多维 grid+threads→一维 `T.Kernel` + `T.SimtVF` ✓、register 跨循环 fragment→stats_ub UB 驻留 ✓；舍弃项（host threads 契约）有 K3 依据 + 意图承接说明（L141：行内并行度由 SimtVF 线程域承接）——无矛盾、无静默丢弃 |
| NPU 重设计可行性 | pass | §0.6 R1–R5 各含「源方案→NPU 新算法→语义保持论证」三要素；一维 `T.Kernel(56)`（实查）、容量按 90% 折扣独立复算达标（wl2 72.1KB / fp32 档 136.1KB ≤ 194.4KB；VRF ≤128KB）；累加顺序变化（tile_n→BLOCK_N pow2 重划）的精度影响以 V1 位级 + V6 划分扫描 ≤1 ulp 机器验证论证（本次重跑复核通过） |
| 设计与决策一致性 | pass | §1.4/§3.1/§3.3/§4/§5/§6/§7 描述的均为 §0.5/§0.6 决策后的 NPU 算法（BLOCK_N pow2 chunk、SUB 子批、persistent 56 核、stats_ub、full/tail 静态二分支），无「§0 判重设计、§3 照抄源方案」脱节 |
| golden 独立性 | pass | §8.1 golden = `torch.logsumexp`（NPU）+ fp64 高精度参考双口径，以 §0.1 数学语义为依据，未复刻 A5 数据流；bench baseline_fn=`torch.logsumexp` 与 test_softmax.py L99 ref_program 同构——精度对比非自证 |

**warn 项（唯一，建议级不阻塞）**：§0.2 L66 对测试 `(33, 33000)` fp16 的 wrapper 落点复算存在算术错误——设计称「N_padded=33280=130×256，130 的 ≤127 最大因子 65 → tile_n=16640」；独立复算：`align_up(33000,256) = 33024 = 129×256`（129=3×43），整除偏好候选 11008（43×256）因 `div_tiles=3 > max_tiles=2` 不满足接受条件（`_primitives.py` L188–203），实际返回预算上限 **tile_n=32512**，落点 **(1, 32512)**。根因有二：① align_up 算术误（33000→33280 而非 33024）；② L51 对 `compute_tile_n` 规则的描述省略了「div_tiles ≤ max_tiles 才接受整除候选」条件。**功能影响为零**：16640 与 32512 两个 tile_n 下 kernel host 常量完全相同（BLOCK_N=16384、num_full=2、rem=232、TILE_R=256、SUB=1、threads=1024——同一编译产物），传播处（§1.6.0 R3/R4 基线归因、§8.2 L0 ⑤）的量化结论亦巧合近似成立（详见建议 1）。语义/算法/优化理解本体均正确，故判 warn 而非 fail。

### 1. API 可行性 — pass

- §3.4 API 表逐条行级核对（亲读源码/先例）：`T.Kernel(56)` 一维无 threads（online_softmax L57、single L134 measured；`T.Kernel(N, threads=)` 为 TypeError——设计 L563 已按 K3 重设计）；`T.Persistent(domain, wave_size, index, group_size, num_stages)` 签名与 `tilelang/ascend/language/loop.py` L82 一致；`T.SimtVF(threads=)`＝frame.py L107；`T.And` 导出＝ir.py L57/L2209；`T.reduce_max/reduce_sum(dim=, clear=)`＝reduce_op.py L95–108；`T.exp/T.log/T.cast` 的 h 前缀 dtype 派发与 exp/log 注册＝intrin_rule_ascend.cc L27–32/L34–49；`T.simd.vexpdif` 存在但设计未用（SimtVF 路径 `tir.exp→simd_vexp` 自动降级＝ascend_simdvf_lower_parallel.cc L409–413）；`T.annotate_buffer_versions`＝annotations.py L19。
- 编程模式：纯 simt（SimtVF 默认优先，无 SimdVF 降级——special/exp 依赖编译器自动向量化，与 `D-ELEM-SIMTVF-AUTOVEC` measured 条目一致）；`auto_schedule: yes`，全文无手动 `T.ascend_set_flag/wait_flag/pipe_barrier`——合规。
- 瑕疵（不改变判定）：L550「`T.max` single L169 同款在用」引用错位——single 无 `T.max`（grep 零命中），L169 为 `T.copy`；实际佐证 online_softmax L128/L133 已在同一行引用，API 佐证本身成立（建议 2）。
- 尾 chunk reduce extent<threads 配置类由本检视探针补实测（机械复核段，PROBE_PASS）——先例空隙已闭合（建议 5）。

### 2. 内存层级规划 — pass

- 搬运路径：GM→UB（`T.copy` 静态切片，MTE2）→VRF fragment（masked/直载 cast）→原位覆写→UB→GM（MTE3），全在 tilelang-ascend skill §6 支持集内；无 Cube/L1/L0（纯 Vector，n/a）；无手写 NZ。
- UB 预算（机械检查 fail 为解析伪影，人工按 workload 分桶复算——机械复核段）：wl2 = x_ub 双版本 2×1×16384×2B = 64KB + stats_ub 8B + 输出暂存 ≈ **72.1KB**；fp32 档 x_ub 128KB + … ≈ **136.1KB**；显式 bm16（SUB 分裂后）≈ **136.1KB**——均 ≤ 194.4KB（216KB×90%，CONST-capacity-a5，设计已披露条目与折扣口径）。尾 chunk x_ub (SUB, TILE_R) 同域更小（L665）。
- VRF：x_frag ≤ SUB·BLOCK_N·4B，SUB 公式含 `SUB·BLOCK_N·4B ≤ 128KB` 联合封顶；主流落点 64KB=16 reg/thread、上界 32 reg/thread，远低于 76 reg/thread 实测水位（CONST-vrf-capacity-a5——条目亲验一致）。
- Buffer 层级标注：§4.3 逐 buffer 标注层级 + alloc API（x_ub/stats_ub=`T.alloc_shared` UB、x_frag=`T.alloc_fragment` VRF），双版本 `annotate_buffer_versions({x_ub: 2})` 计入预算 ✓。

### 3. Tiling 策略 — pass

- Block 划分：BLOCK_N/SUB/TILE_R/threads 公式明确且经独立复算各场景一致——wl2 (1024,32768) bm=1: BLOCK_N=16384, SUB=1；L0 ⑥ (33,32768) bm=4: SUB=2, num_sub=2；显式 bm16+tile_n6912: BLOCK_N=4096, SUB=8；fp32 档 BLOCK_N 上界 16384 ✓。
- Grid：一维 `T.Kernel(56)`，56 = `torch.npu.get_device_properties().vector_core_num` 实查（§5.5 含查询代码与返回值记录——core-split-strategy.md §1 要素①合规）；`T.Persistent([num_tiles], 56, core_id, num_stages=2)` 承接任意 num_tiles（多波由调度器分配），无核启动风暴。
- 非整除处理：N 尾（rem 静态切片 + TILE_R pow2 masked）、M 尾（rem_m → tsub/tsub_rem/last_sub_rows 静态分解 + TIR if）双路径齐备；TRAP-GRID-DIVISIBILITY 条目已对照（persistent 形态免疫 grid 整除性）。
- 分核三要素（机械检查 fail 为解析伪影，人工核对——机械复核段）：① grid 56 实查 ✓；② 纯 AIV（无 Cube 工作，要素②以「仅占 AIV」注明）✓；③ `auto_schedule: yes` + 跨引擎依赖边界枚举（§7.2 MTE2→VALU→MTE3）✓。
- 瑕疵（不改变判定）：L767「波次量化 ~8.7% 于 18 波尾」不可复算——复算 makespan 19 波 vs 理想 1024/56≈18.29 波 ≈ **+3.9%**（建议 4；single 判例注记 7 同形态复现）。

### 4. 技术约束检测 — pass

- 一维 grid ✓；源 `T.Kernel(grid, threads=threads)` 形态已重设计（§0.5 L138、L563 自检行）✓。
- GEMM 分形/NT/L0C：纯 Vector 算子 n/a。
- fragment pow2 约束满足（BLOCK_N/TILE_R 均 pow2——O2 重划的动因本身）；向量宽度整除（BLOCK_N % 128（fp16 lane）=0、% 64（fp32）=0；尾 lane 由 masked 吸收，浪费摊薄 ≤0.07%（(33,33000)）/0（wl3 rem=4096 pow2)——L367 复算一致）。
- 静态循环边界：num_tiles/num_sub/num_full/rem/TILE_R/BLOCK_N/SUB/threads/tsub/tsub_rem/last_sub_rows 全 JIT 常量（L569/L759 清单独立核过——核内无动态 shape 依赖）。
- GM 列窗口对齐：wl2/wl3 行内 32KB/2048B 对齐（L287）；非对齐 stride 测试 shape（如 (33,33000) 行 stride 66000B=16B 对齐）有 single 变体 N=300（stride 600B）PRECISION_PASS 的同类 measured 先例兜底。

### 5. 循环与同步 — pass

- 循环结构（§6）：`T.Persistent`（主调度，num_stages=2）→ TIR if（full/rem_m 尾块静态分支）→ `T.serial(num_sub)`（子批）→ chunk full/tail Python 静态分支 → `T.SimtVF` 域内 `T.Parallel`——五层嵌套逐层明确；主流落点（bm=1→num_sub=1）退化为与 online_softmax measured 先例同构（设计已论证）。
- 同步：`auto_schedule: yes`，无手动 flag；跨引擎依赖由 AutoSchedule 插 flag 对；双缓冲 = `annotate_buffer_versions({x_ub: 2})` + `num_stages=2`（编译器 multibuffering 优先——tilelang-ascend skill §4.2 合规），chunk c+1 MTE2 与 chunk c VALU 重叠（L828）。
- 尾块处理（§6.4）：N 尾静态 rem 切片 + masked（pad=−inf 三幂元）；M 尾 tsub/tsub_rem/last_sub_rows 静态分支（single 判例注记 2/3 教训已吸收——§3.3 伪代码 L519–525 含对应分支）；垃圾行（r≥rows）NaN 不污染有效行论证在案（−inf pad → exp→0）。
- 尾 chunk reduce extent<threads（本维度重点核对项）：探针实测 PASS + reduce.h/CheckAllReduceWidth 源码佐证（机械复核段）——受支持且 SUM 无复制 lane 双重计数；pow2 约束（extent=256、scale=4、width=256 均 pow2）在设计常量体系下恒满足（TILE_R=next_pow2(rem)、threads=min(1024, SUB·BLOCK_N) 均 pow2）。

### 6. 验证方案 — pass

- Golden（§8.1）：`torch.logsumexp`（`device="npu"`）+ fp64 参考双口径；`torch.testing.assert_close`；容差分档 fp16/bf16→1e-2、fp32→1e-3（累积密集 5e-3 口径与 gpu_to_ascend_adaptation.md §1.8 一致）。
- L0 门槛（§8.2）：7 用例覆盖 wl2/wl3 两 dtype、测试落点、尾 chunk 静态切片路径（②rem=4096 / ⑤rem=232）、M 尾块 TIR if 路径（⑥）、fp32 档（⑦）——shape/dtype/精度标准具体；L1 扫描（M×N×dtype×tile_n×block_m 全维）+ L2 回归（testing/ascend 两文件存在性已核）。⑤ 的 tile_n 引用值受建议 1 影响（16640→32512；kernel 常量与验证意图不变，16640 可保留为任意非 pow2 扫描值）。
- 机器验证：verify_equiv.py 重跑全 PASS 且与内嵌表逐项一致（机械复核段）；V4 源链 vs A5 链位级 + NaN 模式一致为 golden 独立性提供额外佐证。

### 7. 完整性与一致性 — warn

- 章节齐全：模板 11 章 + §0（0.1–0.7）+ §1.6.0–1.6.4 全在 ✓；占位符扫描（待办标记 / 模板花括号残留）零命中 ✓。
- 同类引用真实：online_softmax（全部行号引用 L57/60/73/83/92/102–115/123/140/142/150–152/169/215–223 逐条命中）、single 实现与 REVIEW、examples/ascend 四例（rmsnorm L42 注释逐字、per_token_cast_to_fp8 L81–83、gemm L47、flash_attention core.py L130–133）、testing/ascend 两回归、pattern-library 九条目（D-REDUCE-SIMTVF-API / TRAP-NAIVE-ONLINE-MERGE / D-ELEM-SIMTVF-AUTOVEC / CONST-capacity-a5 / CONST-arch-a5 / CONST-mte-bandwidth-model / CONST-valu-model / CONST-vrf-capacity-a5 / TRAP-GRID-DIVISIBILITY——亲验数值与状态标注一致）、algorithm-candidates.md ALG-softmax 命中行——全部真实存在。
- 内部一致：§3 API 映射 ↔ §4 buffer 形状 ↔ §5 Tiling 常量 ↔ §6 循环结构相互一致（buffer 形状=布局决策=循环内层维）。
- **warn 项（3 处文档瑕疵，均不影响设计决策与实现正确性）**：① L550「`T.max` single L169 同款在用」引用错位（single 无 `T.max`；实际佐证 online_softmax L128/L133 同行已引——建议 2）；② L295/L828「1.14MB/(100 B/cycle) ≈ 11.7K cycle」中间数应为 ≈11.4K（比值 ~11.7× 与 −8% 结论按 11.43K 口径成立——建议 3）；③ L767「波次量化 ~8.7%」不可复算，复算 ≈+3.9%（建议 4）。

### 8. 算法优化分析 — pass

检视方式：以 §1.6 为对象逐项独立推演/复算/对照参考表复核，并交叉核对 §1.4/§3.1/§3.3/§6 双向一致性；机械复核（verify_equiv 重跑 + r3_metrics）与独立推演互补。

- **§1.6 章节存在性**：1.6.0（调研四问）/ 1.6.1（O1–O4 + 内嵌机器验证表）/ 1.6.2（14 点向量化替代表）/ 1.6.3（4 候选布局矩阵 + 源码轴对照）/ 1.6.4（fp16-e 备选）——四子节齐全 ✓。
- **算法调研存在性与完整性**：R1 候选表 11 项**含基线**（源 flash 合并式在表；#11 哨兵钳位显式弃选且理由为语义优先——V5 佐证 flash 形式角点本已安全）；R2 在线算法**明确结论**（源即 online flash 合并，算法族一致，单遍驻留为超驻留大 N 唯一单遍族）；R3 复杂度表**四口径**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值）且基线在表、口径写清（中间缓冲无 GM 往返——chunk 流式单遍，亲核无漏计）；R4 逐候选单元匹配/容量/对齐/静态边界/流水融合/跨核评估；调研范围写明（ALG-softmax 命中行 + pilot measured + single 全流程实证 + examples/ascend 四同类 + pattern-library 七条目 + 源仓双变体 + kb_search）✓。
- **调研结论正确性（独立复核）**：
  - 等价性独立推演成立：flash 合并对 chunk 划分的结合律——每元素恰计入一个 chunk、合并 exp 参数 `x−mn`/`m_c−mn` 恒 ≤0 无溢出、−inf 为 max/exp/sum 三幂元、重缩放乘法分组仅舍入差（V1 位级重跑佐证）；
  - 复杂度算术复算：D-2 计算项 33.55M elem×(5/51+1/8)/56 ≈ **133.6K cycle/核** ✓、special 占比 0.125/(5/51+1/8) ≈ **56%** ✓、wl3 单核序列 102400×0.223 ≈ **22.9K** ✓、V7 余量 ~280×（复算 ~275×）✓、流量项 1.14MB/(100 B/cycle) ≈ **11.4K**（设计写 11.7K——中间数瑕疵，比值 11.7× 成立，建议 3）；
  - 负向断言对照 `tilelang-op-design` references/algorithm-candidates.md **ALG-softmax 命中行**复核：R1 减 max/exp2 化简、R2 online 变体、R3 三遍/两遍/单遍、R4 水平 vs 垂直 lane 映射——命中行候选全部在 §1.6.0 评估，**无漏候选**；「两遍法 X 读 +100%、partial+merge 往返+同步数量级劣于合并标量 0.02%」与 CONST-mte-bandwidth-model 口径一致，无与 pattern-library measured 条目矛盾的负向论断；「源算法已用 online 手段」与 R2 结论一致（无矛盾）；
  - 亲和淘汰引用常数条目：容量（CONST-capacity-a5 90% 折扣）、带宽（CONST-mte-bandwidth-model AIV MTE2 100 B/cycle）、VALU（CONST-valu-model 51/8 档）——均具名引用且亲验条目一致〔D-2 口径〕；estimate_latency.h L58–70 `special_func_throughput{8}` 无 dtype 粒度——设计风险 5 诚实披露（亲读证实）；
  - 并列候选裁决路径：主选 vs fp16-e 备选的判定依赖未实证常数（special fp16 2× 吞吐）→ §1.6.4 **实验裁决三件套**齐备（备选结构完整至可实现深度、判定阈值、Stage 4 npusim/msprof 实测反解 + 回写路径）✓——非纸面拍板。
- **调研→设计一致性**：§1.4 算法描述 = §1.6.0 选定（BLOCK_N pow2 chunk 流式 online logsumexp）；§3.1 公式拆解以 §1.6.1 优化后公式为输入（合并 exp 参数、full chunk 无谓词、合并式标量）；§6 = 单遍 chunk 流式结构（非两遍）；迁移一致性：§0.5/§0.6 处置与选定算法族一致，源 8 项优化意图逐项承接（L293 尾列，无静默丢弃）✓。R3/R4 基线行「测试 (33,33000) tile_n=16640 → ×1.97」的 tile_n 引用值有误（根因 §0.2，建议 1），但聚合开销量级对实际落点巧合近似成立（实际 2 chunks × 32768 lane / 33024 列 ≈ ×1.98），且机制归因需从「每 chunk 50.8% 有效」修正为「chunk1 99.2% 有效 + 近空 chunk2（512 列）仍耗 32768 lane」——**不改变任何调研判定**（O2 收益 ×~1.98→×1.0 依旧成立），故本维度 pass 带注记。
- **数学等价优化完整性 + 等价论证独立推演**：O1（合并 exp 参数 + full chunk 无谓词直载：省 select 1 op/elem——恒等）/ O2（BLOCK_N pow2 重划 + 紧凑驻留：结合律 + 三幂元，V1/V6 机器验证重跑通过）/ O3（exp 原位覆写：读写同 fragment 逐元素一一对应，同舍入链）/ O4（合并式标量 `T.Parallel(SUB)`：代数恒等、舍入顺序与源同构）——四要素「原式→优化后→等价论证→收益量化」逐项齐备，独立推演全部成立，无「显然等价」式空话；容差内等价项（划分重排）与 §8.2 精度标准一致（V6 ≤1 ulp << 1e-2）。
- **向量化替代完整性（与 §3.3/§6 交叉核对）**：§1.6.2 14 点覆盖方案全部循环/标量计算点——元素级段（载入 cast/exp/原位覆写/lse cast 写回）全 `T.Parallel(SUB, BLOCK_N|TILE_R)`；行归约 `T.reduce_max/reduce_sum`（硬件树）；合并/lse 行维段 `T.Parallel(SUB)`；保留标量 3 点（host 元数据=JIT 常量不在核内、chunk 循环控制=循环携带依赖、stats 初始化=一次性）理由具体到本算子——§3.3 伪代码与 §6 无未涵盖的逐元素标量循环 ✓。
- **替代 API 佐证**：全部 measured（online_softmax/single 同构段行号逐条命中 + D-REDUCE-SIMTVF-API / D-ELEM-SIMTVF-AUTOVEC 条目亲验）；SimtVF 优先、无 SimdVF 降级（无热点证据前不降级——合规）✓。
- **§1.6.3 布局决策完整性**：4 候选矩阵（选定 #1 N 轴 chunk 流式；#2 串行化 >8×、#3 流量非瓶颈+往返、#4 repack 无收益——弃选均有量化理由，无「未考虑」式论证）；规约类必选候选（水平归约 vs 垂直扫描 lane 映射、两遍/online 交互）在 R4/矩阵中覆盖；I/O layout 作契约、核内布局作设计变量（x_ub 紧凑 dtype + x_frag pow2 fp32 分离）；迁移源码轴对照已做（GPU N 轴是输入非结论，独立复核）；弃选前提亲核（repack 代价机制与 skill §6.1 transpose 路径语义一致，无 GM 级 transpose 误写；与 pattern-library 实测条目无矛盾）✓。
- **布局决策与实现一致**：§1.6.3 选定轴（N）= §3.3 `T.Parallel` 内层维（BLOCK_N/TILE_R）= §4.3 buffer 形状 = §6 循环内层 ✓；§4.5 UB/VRF 预算按选定布局计算（人工分桶复算通过）✓。

## 附：建议级问题汇总（不阻塞，供 Stage 1 择机修订 / Stage 3 实现注意）

1. **§0.2 wrapper 落点算术修订（根因项，含传播处）**——L66：`(33,33000)` fp16 实际落点为 **(1, 32512)** 而非 (1, 16640)：`align_up(33000,256)=33024=129×256`（非 33280=130×256）；整除偏好候选 11008 因 div_tiles=3 > max_tiles=2 被拒（`_primitives.py` L188–203 接受条件），返回预算上限 32512。同步修订：① L51 规则描述补「整除候选仅在不增加 tile 数（div_tiles ≤ max_tiles）时被接受」条件；② L275/L286/L293/L295/L304/L368/L667 的「16640 落点」引用改为 32512（或标注 16640 为假想/扫描值），×1.97 归因改为聚合口径 ≈×1.98（2×32768 lane/33024 列），机制描述改「近空尾 chunk 仍耗全 fragment lane」；③ L897 L0 ⑤ tile_n=16640→32512（BLOCK_N=16384/rem=232/TILE_R=256 等 kernel 常量与验证意图不变——两个 tile_n 值编译产物相同）；④ L898 L1 tile_n 扫描建议补实际落点值 {32512, 27648, 11008}（16640 可保留为任意非 pow2 输入扫描）。
2. **§3.4 L550 `T.max` 引用错位**——「single L169 同款在用」不成立（single 无 `T.max`，L169 为 `T.copy`）；改为 online_softmax L128/L133（`m_c[r] = T.max(m_c[r], sentinel)` / `mn_frag[r] = T.max(stats_ub[r,0], m_c[r])`，同行已引，仅需删除 single 引用）。
3. **D-2 流量项中间数**——L295「1.14MB/(100 B/cycle) ≈ 11.7K cycle」应为 ≈11.4K cycle（1.14e6/100=11400）；「~11.7 倍」比值与 L828「−8%」结论按 11.43K 口径成立，仅需改中间数（两处）。
4. **§5.4 L767 波次量化数字**——「~8.7%」不可复算；建议改为「makespan 19 波 vs 理想 1024/56≈18.29 波 ≈ +3.9%」或给出推导口径（single 判例注记 7 同形态，属复现性数字瑕疵，Stage 4 实测为准）。
5. **Stage 3 实现注记：尾 chunk reduce extent<threads 配置类**——设计所引先例（online_softmax L150–152 / single L150–155）的尾 chunk 均保持全尺寸 fragment（extent ≥ threads）；本设计 `(SUB, TILE_R)` 尾 fragment 在 (33,33000) 落点下为 256 元素 @ threads=1024（replication=4），该配置类已由本检视探针真机实测通过（sum ratio=1.000000，无复制 lane 双重计数；机理＝reduce.h ReplicateExtent 建模 + CheckAllReduceWidth pow2 约束满足 + ub_reduce scale 折叠，见机械复核段）。Stage 3 请保持 L0 ⑤（(33,33000)，rem=232）作为该路径首验用例；设计可选在 §3.4/L441 补「reviewer 实测」标注，无需结构改动。

## 检视执行记录

- 检视人：tilelang-design-reviewer（Stage 2 subagent）；日期：2026-10-16；环境：Ascend950PR_9579（28 cube / 56 vector 核，实查）+ tilelang-fork 工作区。
- 方法：① 亲读 DESIGN.md 全文（1003 行）与源算子 tiled 变体全文（L101–189）+ wrapper/`_primitives.py`/test_softmax.py/manifest；② 机械复核两项强制（verify_equiv.py 重跑全 PASS、design_calc_check.py JSON 逐项消费，伪影消解经亲读解析器源码 check_ub_budget L108 起/check_core_split L259 起）；③ 引用行级核对（online_softmax/single/examples/ascend 四例/pattern-library 九条目/algorithm-candidates ALG-softmax 命中行/tilelang 源码 API 导出与 lowering 路径）；④ 独立复算（wrapper 落点逐 shape、BLOCK_N/SUB/threads 公式各场景、UB/VRF 分桶预算、D-2 三项、V7 余量、波次量化）；⑤ 等价性独立推演（O1–O4 逐项 + flash 结合律/幂元/溢出界）；⑥ 补充实证探针一枚（尾 chunk reduce extent<threads，最小 kernel 复刻设计配置，真机 PROBE_PASS）。
- 判定汇总：维度 0 warn / 1 pass / 2 pass / 3 pass / 4 pass / 5 pass / 6 pass / 7 warn / 8 pass；阻塞级 0、建议级 5 → 结论: 通过。warn 项均为文档层算术/引用瑕疵，kernel host 常量与验证意图不受影响，不构成 Stage 3 风险。
