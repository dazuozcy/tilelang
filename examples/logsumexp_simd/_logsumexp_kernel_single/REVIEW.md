# _logsumexp_kernel_single 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/logsumexp_simd/_logsumexp_kernel_single/DESIGN.md（951 行，Stage 1 产出于 2026-10-17）
- 任务类型: 迁移（GPU TileOPs `LogSumExpFwdOp` → Ascend A5，**SimdVF 独立变体**，用户显式指定 programming_mode: simd）
- 源算子: `/home/z00875089/TileOPs-fork/tileops/kernels/reduction/logsumexp.py`（`_logsumexp_kernel_single` L40–99；`LogSumExpKernel` L246–557；`_primitives.py` DEFAULT_ALIGNMENT=256 L30 / MAX_SINGLE_TILE_COLS=32512 L39——本次检视已亲自全文读取核对）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 6（均记录于对应维度详情，不影响任何采纳/弃选/门禁结论）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: **PASS**——`PYTHONPATH=/home/z00875089/tilelang-fork python examples/logsumexp_simd/_logsumexp_kernel_single/verify_equiv.py`（torch CPU，本次检视独立重跑）输出 `Total 52 gated rows, EQUIV_FAIL count = 0 -> ALL ADOPTED ITEMS EQUIV_PASS`，与 §1.6.1 内嵌表**门控值全部一致**：V1 max ulp ≤1 / max_rel ≤1.086e-07 / 违反率 0（@1e-2）；V2 max ulp ≤2（128×16384）/ ≤1.869e-07 / 0（@1e-3）；V3 ≤1 / 0（@1e-3）；V4 位级 0 差 + NaN/±inf 模式一致（4 行）；V5 分歧 lane 0/2（上界 65514.4 < 65520 复核成立）。脚本建模方式与声明一致（`_pad_groups` 64 元素组分桶 + 尾组 −inf；`simd_chain` lane 内跨组累加 + 单次跨 lane max/sum——已读脚本源码核对）。**一处描述性瑕疵（建议级）**：内嵌表 V1 零值清单将 33×300 列为 0，重跑实测 fp16/(33,300)=0 但 **bf16/(33,300)=1 ulp**（仍 ≤1 门控阈值、违反率 0）——不改变任何门控结论，建议 Stage 3 顺手修正表格措辞。
- **design_calc_check**: JSON checks 摘要（`python3 .agents/tools/design_calc_check.py --design examples/logsumexp_simd/_logsumexp_kernel_single/DESIGN.md`）：
  - `ub_budget`: **fail**（"11 buffer entries sum = 258.7 KB > budget 223 KB"）→ **人工复核判定为工具解析假阳性**，理由（已读 `.agents/tools/design_calc_check.py` 源码定位）：① 工具按 `\*\*(UB|L1|...)` 标记切换层级，"**VRF 预算**"标题不匹配任何层级标记 → §4.5 VRF 对照表两行（"对照 SIMT 兄弟变体 x_frag ≤128KB / 128KB"）被计入 UB 桶，2×128KB 即占假阳性总量 256/258.7KB；② 逗号千分位 "131,072B" 被 SIZE_RE 解析为 72B（真实 x_ub 项几乎未计入）；③ 设计真实 UB 占用人工复算 = x_ub 双版本 2×SUB×N_pad×elem（wl1/wl4 均 131,072B）+ lse_ub 16B + 编译器保留 ~8KB ≈ **139,280B ≤ 228,556B（=int(248×1024×0.9)，CONST-capacity-a5 assumption 90% 折扣，SimdVF 248KB 口径已在 §4.5 披露）**，占 60.9% ✓。工具 docstring 自述 "advisory report... NOT a gate"。**人工复核通过**。
  - `l0c_budget`: **warn**（"block 16x4 x fp32 accum"）→ 假阳性：CUBE_PATH_RE 被 §5.1 否定句"无 `T.gemm`/L1/L0 段"触发（正则无否定语境识别），16x4 为无关文本数字。本设计纯 Vector 无 Cube 段，L0C **n/a，人工复核通过**。
  - `core_split`: **fail**（"recomputed ceil(1024/16)×ceil(300/1)=19200 vs declared grid 56"）→ **人工复核判定为工具模型不适用**：工具按非持久 grid 模型（grid=输出 tile 数）复算，而本设计为 **persistent 形态**——`T.Kernel(56)`（56=vector_core_num 实查值）+ `T.Persistent([num_batches], 56, core_id, num_stages=2)`，任务域 num_batches=ceil(M/SUB)（wl1=128）由 persistent 调度器分派，grid=物理核数恰为 core-split-strategy §1 ① 要求说明的"block 数与核数关系（单波/多波、persistent 化与否）"之合规形态；且工具解析数字跨 workload 混杂（M=1024 取自 wl1、N=300 取自非对齐示例、bn 缺省 1）。**人工复核通过**（维度 3 三要素详核见下）。
  - `r3_metrics`: **pass**（"all four R3 metrics present"）✓ 与人工核对一致（§1.6.0 R3 表含 FLOPs/GM Bytes/扫描遍数/中间缓冲四口径 + 本变体特有 UB 重读补充口径）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

（检视方式：亲自 Read 源算子 `logsumexp.py` 全文 557 行 + `_primitives.py` 全文 445 行，与 DESIGN.md §0 逐项核对。）

- **语义理解正确性 — pass**：§0.1 语义要素表逐行核对源码成立——数学映射 `y[i]=row_max[i]+log(row_sum[i])`（源 L85–93：`T.fill(row_max,−inf)`→`reduce_max(clear=False)`→原位 `exp(x−m)`→`reduce_sum`→`out_local[i]=row_max[i]+T.log(row_sum[i])`）✓；I/O 契约 x (M,N)/y (M,) 同 dtype（L57–60、L91）✓；**fp32 中间计算**（L64–66 `x_f32/row_max/row_sum` 均 "float32" fragment，输出 fragment 为输入 dtype）✓；边界语义五点独立推演核对：masked 载入 `T.And(pid_m*block_m+i<M, j<N)` 越界 −inf（L72–77）✓、全 −inf 行 → m=−inf → −inf−(−inf)=NaN → S=NaN → y=NaN（源算法固有，与 torch −inf 分歧——§9.3 如实登记且取舍有据）✓、含 +inf 行 → inf−inf=NaN ✓、部分 −inf → exp(−inf−m)=0 中性 ✓。§0.2 single 变体选中条件与源 `_tile_n_for_block_m`（L338–360：`N_padded ≤ MAX_SINGLE_TILE_COLS ∧ compute_tile_n 返回 N_padded`，后者即 `block_m·elem·N_padded ≤ budget`，`_primitives.compute_tile_n` L156–157 核对）一致 ✓；MAX_SINGLE_TILE_COLS=32512 的"32768 向量化边界安全回退"注记与 `_primitives.py` L32–38 注释原文一致 ✓；manifest 四 workload 落点独立复算无误（[32,32,32768]→N_padded=32768>32512→tiled；[4,128,4096] dim=[0,2]→(128,16384)→single）✓。
- **算法解读完整性 — pass**：§0.3 步骤表 S0–S6 与源 kernel 计算语句**逐语句映射无遗漏无臆造**（S0↔L50–52、S1a↔L72–77、S1b↔L79–82、S2↔L84–85、S3↔L87–88、S4↔L89、S5↔L91–93、S6↔L95）；数据流图覆盖全部 6 个 buffer（shared_buf/x_local/x_f32/row_max/row_sum/out_local）✓；循环并行结构（grid=ceildiv(M,block_m) + threads + `T.Parallel(block_m,N_padded)`，L61/L72）与 host 侧逻辑（lru_cache L40、`_neg_inf` 字面量、threads 由 wrapper 传入 L395 默认 256）均已列出 ✓。
- **优化手段覆盖完整性 — pass**：§0.4 七项与源码显式结构一一对应（#1 单 tile 全驻留↔L62 shared_buf (block_m,N_padded)；#2 256 对齐双路径↔L68 `if _needs_pad`＋源模块 docstring L10–11 "256-element alignment required by T.copy() shared memory"；#4 原位覆写↔L87–88 写回 x_f32；#5 树形归约↔L85/L89；#7 非持久 grid↔L61），层级标注恰当（#1/#3 算法层、余优化层）；风险自检（无 warp shuffle 手写/swizzle/cp.async/persistent）与源码事实一致 ✓。
- **耦合性判定合理性 — pass**：§0.5 12 条目四态处置齐全且与 backend-glossary §1 概念迁移锚一致——warp shuffle 树归约→**等价替换**为 vcmax/vcadd 跨 lane 全归约内在（非"保留"，佐证 `simd_inst.h` L688–710 `asc_reduce_sum/asc_reduce_max` 已亲自核对 + flash core.py L115–118/L134–137、per_token L142–146 实测在用）✓；`T.Kernel(grid,threads)`→**重新设计**为一维 persistent（A5 一维约束）✓；SMEM staging→UB 单跳 + UNPK_B16/vcvt 装载 **等价替换**（cast 位级回归 L64–80 已核对）✓；两个舍弃项（exp 原位覆写、host threads）均有理由 + 意图承接说明（"零寄存器驻留"超量承接 / SimdVF 无线程概念），**无静默丢弃** ✓。
- **NPU 重设计可行性 — pass**：§0.6 R1–R5 每项「源方案→NPU 新算法→语义保持论证」三要素齐备；满足 Ascend 约束——一维 `T.Kernel(56)`（56=vector_core_num 实查，CONST-arch-a5 measured 条目一致）✓、UB 228,556B 折扣预算内（§4.5）✓、静态循环边界全 JIT 常量 ✓、无 C0/分形适用段 ✓；语义保持论证非空话——以 verify_equiv.py V1（≤1 ulp/违反率 0）、V2（≤2 ulp）、V4（位级 + NaN 模式一致）机器证据支撑，本次已重跑核实 ✓。R4 尾批不依赖 `T.copy` 越界行为的决策正确（`pad_value=`/`data_select=` 为列向行尾 padding——copy_op.py L278–287 + skill §6.1 原文已核对，行向 OOB 确实不适用）✓。
- **设计与决策一致性 — pass**：§1.4 主路径与结构差异 ①–⑥ 逐条回指 §0.6 R1–R5；§3.2 API 映射、§4 内存规划（x_ub (SUB,N_pad) 紧凑 align64）、§5.2（SUB 单约束封顶）、§6（persistent+TIR if 二分支+range 串行）全部是重设计后形态，无"§0 判重设计、§3 照抄源方案"脱节 ✓。
- **golden 独立性 — pass**：§8.1 golden = `torch.logsumexp(x.float(),dim=-1).to(x.dtype)`（源仓 ref_program 直移植，NPU 上计算，device assert 在位），未复刻 §0.6 NPU 算法 ✓；`_source_lse_sim` 仅为 Boundary ④ 源继承分歧角点的对照器（源链语义模拟，非 NPU 数据流翻译），与 SIMT 兄弟变体 L64–78 实文件同构（已读该文件核对）✓。
- **建议级备注 1/6**：§0.1 host 侧语义行引用 "wrapper L193–207" 指向的实际是源文件 Dispatch 选择器（`_logsumexp_kernel`），dim 归一化/reshape 主张的真实佐证在同文件 class docstring L259–261（"M: Number of rows (product of all dims except last)"）与 forward docstring L543–545（"Accepts an (M, N) tensor"）——主张本身正确，行号引用偏移，建议 Stage 3 修订时顺手更正。

### 1. API 可行性 — pass

- **API 存在性（逐条亲核）**：§3.4 表全部 API 均有源码行号 + examples/testing 实测用法双佐证，本次逐一打开核对无一处失实——`pset`(simd.py L111，PAT_VL1..VL128)✓、`update_mask`(L125，"lanes [0,value) active")✓、`vld`+UNPK_B16(L302–321 dist 表含 UNPK_B16、L324 签名)✓、`vadd`(L554)✓、`vmax`(L815)✓、`vexp/vln`(L860/L867)✓、`vdup`(L922)✓、`vdupv`(L933，POS_LOWEST)✓、`vcadd`(L957)✓、`vcmax`(L969)✓、`vexpdif`(L1182–1195，**fp32 同宽强校验 TypeError 属实**)✓、`vcvt`(L1209 签名、L1340–1349 f16/bf16→f32 part_x2 路径)✓、`vsel`(L1429，"mask ? src0 : src1")✓、`vsts`(L457，ONEPT dist→`vsts_1st` codegen_ascend.cc L2403–2404)✓、`T.Persistent`(tilelang/language/loop.py L82)✓、`T.Kernel` 一维✓、`@tilelang.jit(target="ascend",out_idx=[1])` 懒工厂(online_softmax.py L215–223 实测形态)✓、`-T.infinity(T.float32)`(online_softmax L73)✓。
- **vsel+update_mask 组合语义方向核对**（尾组掩码正确性关键）：`update_mask(k_tail,32)` 前 k_tail lane 为真 + `vsel(c,−inf,mask)`=mask?c:−inf → 有效 lane 保留真实数据、尾部 lane 置 −inf——与 §3.3 伪代码意图一致 ✓；同构先例 `test_tilelang_ascend_simdvf_integer_predicate.py` L261–262、`transform_sf.py` L205/L215/L249（已亲读）✓。
- **编程模式匹配 — pass**：声明 auto_schedule: yes 且全文无手动 `T.ascend_set_flag/wait_flag/pipe_barrier`、无 `TL_ENABLE_AUTO_SCHEDULE: False` ✓；纯 Vector 无 mixed 需求，无 `T.MixedKernel` 误用 ✓。
- **SimdVF 使用理由 — pass**：SimdVF 非默认降级而是**用户显式指定**（任务契约），且 §2.2 给出结构性论证（显式内在使 O4 延迟归约可行——自动 lower 形态 `ascend_simdvf_lower_parallel.cc` L818–841 确为逐组 vcmax+vmax 合并，已亲核，显式形态 max 遍归约指令 2G→G+1 即 −(G−1)×2，wl1 G=64 → −126 指令/行，−50% 结论复算成立）✓。

### 2. 内存层级规划 — pass

- **搬运路径 — pass**：仅 GM→UB（MTE2 2D 静态切片，src 行距 N/dst 行距 N_pad 独立 pitch——skill §6.1 原文 "byte-granular rows with independent source/destination pitches" 已核对）与 UB→GM（MTE3 1D 切片）+ VRF↔UB 向量装载/单点存储，全部在支持集内 ✓；无手写 NZ、无越级路径、无 L1/L0A/L0B/L0C 段（纯 Vector，§1.5/§4.4 明示）✓。
- **UB 预算 — pass（机械 fail 已裁定为解析假阳性，见机械复核段）**：人工复算 wl1 x_ub=(8,4096)×2B×2 版本=131,072B、wl4=(2,16384)×2B×2=131,072B、lse_ub≤16B、保留 ~8KB → ≈139,280B ≤ 228,556B（60.9%）✓；所用条目（CONST-capacity-a5，assumption）与折扣口径（90%）、SimdVF 248KB 无 32KB 线程上下文扣减的差异均已在 §4.5 显式披露 ✓；multibuffering 版本数（×2）已计入 ✓。SUB 上界公式全域独立复解（fp16 N=4096→8、N=16384→2、N=32512→1、N=300→128；fp32 N=4096→4；(32,256,fp32,bm=16)→16）全部满足约束 ✓。
- **VRF 预算 — pass**：~10 活跃向量寄存器×256B≈2.5KB（256B/寄存器有 CONST-valu-model "SIMD 向量寄存器 256B" 佐证）；CONST-vrf-capacity-a5（assumption 256KB）已披露且给出 Stage 3 `get_kernel_source()` spill 首查计划 ✓；对照候选 #6 的 128KB 贴顶论据与 constants.md "76 reg/thread 无 spill 实测样本 + x_frag ≤128KB 封顶" 条目原文一致 ✓。
- **Buffer 层级标注 — pass**：§4.3 每 buffer 标注层级 + alloc API（x_ub/lse_ub=`T.alloc_shared`(UB)、VRF=SSA 绑定无 alloc）✓。

### 3. Tiling 策略 — pass

- **Block 划分与 grid 映射 — pass**：§5.2 给出 N_pad/G_full/k_tail/SUB/full/rem/num_batches/NUM_BLOCKS 全套推导公式 + wl1/wl4/角点数值示例（复算无误）；grid 一维 `T.Kernel(56)`，任务域经 `T.Persistent` 一维分派，无多维分解需求 ✓。
- **对齐约束 — pass**：无 GEMM → C0 分形 n/a（`CONST-gemm-dim-constraints` 仅约束 T.gemm 路径，与 constants.md 原文一致）；组宽 64 元素对齐（UNPK_B16 footprint 128B/fp32 NORM 256B）+ 组起始偏移恒对齐论证成立 ✓；GM→UB 32B 行长对齐主 workload 满足，非对齐 N=300（600B/行）的字节级行长支持与带宽档降级已如实列 §9.1 风险 2 ✓。
- **非整除处理 — pass**：M 尾批静态 full/rem 二分支 + 双侧静态切片（不依赖 T.copy OOB——对照源 GPU 隐式依赖的正确修正）；N 尾组 fp32 域 vsel −inf 掩码；M<SUB、N<64 退化形态显式覆盖（§5.4）✓。
- **分核策略三要素（对照 core-split-strategy §1/§2.2）— pass**：① grid 划分含**核数实查依据**——§5.5 记录查询代码与返回值（2026-10-17，Ascend950PR_9579：cube_core_num=28 / vector_core_num=56），与 CONST-arch-a5 measured 条目一致；grid=56=物理 AIV 数 persistent 单波常驻（无超发串行），批数与核数关系（wl1 128/56≈2.29 批/核 + 尾波 16 批不均衡 ~1.8%）已量化说明 ✓；② AIC/AIV 分工——纯 Vector 仅占 AIV、不占 AIC、无 dual_copy 所有权问题，pure kernel 核型由 codegen 自动判定 ✓；③ 同步策略——auto_schedule: yes 声明 + 五条数据依赖边界列举（§5.5/§7.2），无手动 flag（预算不占用）✓。核内串行边界全静态（JIT 常量）✓。**机械 core_split fail 为工具非持久模型误用，人工裁定通过**（详见机械复核段）。

### 4. 技术约束检测 — pass

- 一维 grid：`T.Kernel(56)` 无 threads= 参数（源 `T.Kernel(grid, threads=threads)` 已在 §0.5/§0.6 R1 识别并重设计；§3.5.1 首行 + §9.2 首条防回退）✓。
- GEMM 分形/NT/L0C：n/a（无 Cube 段；§5.1 判定依据充分——候选 10 Cube GEMV 归并已在调研中量化弃选）✓。
- ascend-constraints §2 强制检测表（§3.5.1）九项逐项有结论：GPU 专用 API 清零（SMEM staging/warp shuffle 均等价替换）、OOB padding 不误用（pad_value 列向语义已亲核 copy_op.py L278–287）、host 侧仅元数据推导 ✓。
- 静态循环边界：num_batches/G_full/k_tail/SUB/rows∈{SUB,rem}/full/rem 全 per-shape JIT 编译期常量；尾组分支为 Python 静态 `if k_tail>0`（源 `_needs_pad` L68 同法先例——prim_func 体内闭包常量分支在源码本身即在用，已核对）✓。

### 5. 循环与同步 — pass

- **循环结构 — pass**：§6.1 六层结构（persistent 批/TIR if 尾批/range 行/range 组/显式向量元素级/单次跨 lane 归约）逐层有 API + 理由 + §1.6.2 编号回指；`T.Parallel`/`T.Pipelined` 不使用的结构性理由成立（SimdVF 无线程域；批间流水由 num_stages=2 承接）；SimdVF 域内 `range` 串行有 per_token L135–136 实测先例 + P-LAYOUT-TAIL-STATIC-EXTENT 原文注记（"TVMScript 把 range 也解析成串行循环而非 IR 构建期展开"）双重支撑 ✓。
- **同步与模式匹配 — pass**：auto_schedule: yes 与 §7.2 依赖边界表一致（MTE2→VALU、VALU→MTE3、版本交替、lse_ub WAR、VALU 内顺序——五条边界覆盖完整）；无手动 flag ✓；"无 mem_bar 需求"论证经与 flash core.py 实况对比成立（flash L103/L117–126 的 VST_VLD barrier 服务于 m_ub **UB 往返**形态——L120/L123 已亲核；本设计 m_brc/s 全程 VRF SSA 经 vdupv 直接消费，无 UB 往返即无该 barrier 需求）✓；风险 3（Persistent+TIR if+SimdVF 嵌套无单一先例）如实披露且 fallback 预案具体（拆双 persistent 域）——各层分别有先例的论断经核对属实（SIMT 兄弟变体 L138–150 Persistent+TIR if+serial+SimtVF 嵌套实文件已读 ✓、per_token Persistent+SimdVF+range ✓）。
- **尾块处理 — pass**：§6.4 行尾/列尾/组尾/批内行尾四类齐备，与 §5.4 一致 ✓。

### 6. 验证方案 — pass

- **Golden — pass**：§8.1 含 PyTorch 参考实现草案（`device.type=="npu"` assert 在位）+ L0 对比语句（`torch.testing.assert_close`）✓；源仓 ref_program 直移植保证迁移验证独立性 ✓。
- **L0 测试计划 — pass**：五个门槛用例全部具体到 shape/dtype/block_m/SUB/G/k_tail 且逐一对应风险首触点（③ bf16 宽化首验→风险 1、④ (33,256) 尾批→风险 3、⑤ (32,300) 尾组→风险 6）✓；L1 扫描矩阵（M×N×dtype×block_m + SUB 分裂点两形态 + N<64 + M=1 + 多维 flatten）、L2 压力（19 波 persistent、契约边界 (2,32512)、N=32513 负向 assert、fp32 大 N 紧约束）、Boundary 六组（含源继承分歧角点以 `_source_lse_sim` 对照 + 尾组尾批双触发 (33,300)）分层完整 ✓。
- **精度标准 — pass**：容差分档符合 AGENTS.md（fp16/bf16→1e-2、fp32→1e-3）；fp32 取 1e-3 档（非 5e-3）的"非 accumulation-heavy"判断有 verify_equiv 实测支撑（全 shape 相对差 ≤1.9e-7，两个数量级余量）✓。

### 7. 完整性与一致性 — pass

- **章节齐全 — pass**：§0（0.1–0.7 全）+ §1–§10 共 11 章节；§1.6 含 1.6.0/1.6.1/1.6.2/1.6.3 四子节 ✓。
- **无占位符 — pass**：grep TODO/待补充/待完善/FIXME 零命中；`{[a-z_]*}` 4 处命中均为 LaTeX 数学记号（`e^{x}`、`\sum_{j=0}`），非模板占位符 ✓。
- **同类引用真实 — pass**：§3.5.3/§10.5 全部引用经 Glob/Read 验证存在——`examples/ascend/`{example_per_token_cast_to_fp8.py, flash_attention/core.py, deepgemm/kernels/transform_sf.py, example_gemm.py, example_rmsnorm.py}、`examples/pilot/online_softmax/online_softmax.py`、`testing/ascend/language/`{test_tilelang_ascend_simdvf_cast.py, test_tilelang_ascend_simdvf_integer_predicate.py}、SIMT 兄弟变体三文件（`examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp_kernel/`{__init__.py, _logsumexp_kernel_single.py, _logsumexp_kernel_single_DESIGN.md, perf_opt/}）、`examples/TileOPs/tileops/manifest/reduction.yaml`（L124–172 `LogSumExpFwdOpSimd` 段实读核对：workloads/roofline/source 路径与 §0.2/§10.5 转述一致）✓。**引用行号抽查 20+ 处零失实**（per_token L117/L118–124/L128/L129/L132/L133–134/L135–136/L142–143/L146/L147/L154；flash L94/L96/L115–118/L120/L130–133/L134–137/L139–142；transform_sf L205/L215/L249；example_gemm L47；online_softmax L57–60/L73/L140–142/L215–223；cast test L50–53/L64–80；integer_predicate L261；SIMT 兄弟变体 L64–78/L138–150）。
- **内部一致 — pass**：SUB 公式/NUM_BLOCKS=56/buffer 形状 (SUB,N_pad)/预算 228,556B 在 §1.4/§3.3/§4.5/§5.2/§5.5/§10.3 各处同值一致；§4.5 "≈139KB/223.2KB(61%)" 字节级复算 139,280/228,556=60.9% ✓（139KB 为十进制口径、223.2KB 为二进制口径的混用属排版口径小疵，比值按字节计算无误）；风险表 8 项与正文回指闭环（风险 1↔§3.4 cast 回归覆盖实况、风险 5↔D-2 口径披露、风险 6↔§9.2 fallback）✓。

### 8. 算法优化分析 — warn（6 条建议级备注，无阻塞项）

- **§1.6 章节存在性 — pass**：1.6.0/1.6.1/1.6.2/1.6.3 四子节齐备且相互引用闭环。
- **算法调研存在性与完整性 — pass**：R1 候选表 11 项**含基线候选 #0**（源驻留三段式）✓；R2 在线算法**明确结论"有"** + 三层证据（① 源仓 `_logsumexp_kernel_tiled` L105–194 online 递推——本次已亲读源码核实 running max/sum + `row_sum·exp(prev_max−row_max)+tile_sum` 重缩放合并确在 L140–183；② ALG-softmax R2 命中行；③ 结构判据）+ 契约空间内"非必需"论证（驻留可行 + 算法族分工=源仓双变体结构）✓；R3 复杂度表**四口径齐全**（FLOPs/GM Bytes/扫描遍数/中间缓冲峰值，基线在表，口径逐列写清，另补本变体特有 UB 重读口径）——机检 r3_metrics pass ✓；R4 六检查项逐候选评估、负向淘汰均附依据 ✓；调研范围明示（候选表命中行 + 4 个 SimdVF 先例 + 源仓双变体 + SIMT 兄弟变体调研 + pattern-library 8 条目 + 知识预注入 3 条目）；互联网来源未使用且给出"本地源已覆盖全部已知变体"的范围论证 ✓。调研深度与规约/统计类算子匹配（完整调研档）✓。
- **调研结论正确性（独立复核）— pass**：
  - **负向断言逐条亲证**：① 候选 2（exp2 换底）弃选——`grep exp2|tanh tilelang/ascend/language/simd.py` **零命中**（无 exp2 快档内在、vtanh 不存在，VP-2026-0014 属实）+ `intrin_rule_ascend.cc` L45–52 exp/exp2/log2 同文件同宏注册 + `estimate_latency.h` L66 special_func_throughput{8} 单档——"换底同档无收益反多 1 乘"结论成立，与 ALG-softmax R1 参考行"A5 上此候选无收益"及 CONST-valu-model measured 补注完全一致 ✓（小注：设计引"L34–49"，log2 实际注册在 L51，主张不受影响）；② 候选 9（fp16 域 max）弃选之"bf16 vcmax 未文档化"——CONST-valu-model "硬件 warp-reduce dtype 仅 fp16/fp32/int32/uint32 直接 lower 到 asc_reduce_*" 条目同向佐证，且本设计 fp32 累加器用法在支持域内 ✓；③ 候选 4/1.6.3 #4（通用转置）——`copy_op.py` L268–269 docstring 原文 "**Ascend GM→L1 only**. Emit dn2nz..."，`ls tilelang/ascend/language/` 无 transpose 模块，负向论断按"未文档化假设"规范标注且弃选稳健（N 已是最长连续轴，转置无轴收益）✓；④ R2 无"源已用 online 却断言无在线变体"矛盾——设计正确断言"有"并以源仓 tiled 变体为直接证据 ✓；⑤ 候选 8（组粒度 online）弃选的**角点推演独立复核成立**：m 初始化 −inf → 首组全 −inf 行 `exp(−inf−(−inf))=exp(NaN)=NaN` 污染累加器（破坏 §0.1 ④ 部分 −inf 中性语义）；初始化 −3.402823e+38（`ascend_simdvf_lower_parallel.cc` L799–801 实值已亲核）→ 全 −inf 行输出 −inf ≠ 源 NaN（破坏 §0.1 ②）——两难真实存在，与 TRAP-NAIVE-ONLINE-MERGE measured 条目（fp64 口径违反率 0.136——设计表述"fp16 违反率 13.6%"为数据域/口径的宽松转述，数值与条目一致）及 ALG-softmax"logsumexp 族弃用哨兵钳位"判据同构 ✓。
  - **复杂度算术独立复算**：D-2 wl1——流量 8,388,608B/(56×100 B/cycle)=1,498≈1.5K cycle ✓（100 B/cycle=CONST-mte-bandwidth-model AIV MTE2 档，条目实读一致）；计算 4,194,304×(4/51+1/8)=4.19M×0.2034→/56=15,236≈15.2K cycle ✓（51/8 档=estimate_latency.h L58–66 源码实值）；special 占比 0.125/0.2034=61.4%≈61% ✓；UB 重读 2B/elem÷256B/cycle=0.0078 cycle/elem→0.6K cycle ✓（256B/cycle=estimate_latency.h L70 valu_bandwidth 源码实值，**非未实证常数**）；容量项可忽略（单遍 GM 读、无 L0C、无中间落 GM——无漏计 GM 往返）✓；瓶颈反转讨论（special op 粒度 element vs 向量指令级两种解释下主选均最优）与 estimate_latency.h L58–70 确无粒度注明的实况相符，风险 5 如实列账 ✓。
  - **并列候选裁决路径**：主选 vs #6（寄存器驻留）差异为结构性 VRF 2.5KB vs 128KB（50×量级，CONST-vrf-capacity-a5 assumption + 76reg/thread 实测水位双佐证）+ UB 重读余量（256B/cycle 有源码佐证）——判定裕度不依赖未实证常数，"无实验裁决三件套"的豁免论证成立 ✓；#2/#3/#4 为量级差弃选（>4× / +50% 流量 / 无轴收益）✓；block_m/SUB 甜点属 config 级旋钮，已按 P-LAYOUT-ROW-RESIDENCY measured 修订规则给出 Stage 4 扫描计划（§5.2/§8.3——wl1 甜点预期 SUB=4、+50.9% vs bm=16 与条目原文逐字一致）✓。
  - **调研→设计一致性**：§1.6.0 选定"UB 驻留两遍式（align64 单路径+fp32 尾组掩码+延迟归约+vexpdif 融合+VRF 流式+子批双缓冲）"= §1.4 主路径 = §3.1 四步拆解 = §3.3 伪代码 = §6 循环结构，五处同构无脱节 ✓；§0.5/§0.6 处置与调研选族一致，源 7 项优化意图承接逐项说明（#4 舍弃有"零寄存器驻留超量承接 + D-2 量化代价"论证），**无静默丢弃** ✓。
- **数学等价优化完整性 — pass**：O1–O6 覆盖稳定化/布局/指令融合/归约时机/cast 链/饱和角点，采纳 3 项（O2/O3/O4）四要素齐备；源公式层已最优的结论有据（R1 候选 2/3 弃选证据同上）✓。
- **等价性论证正确性（独立推演）— pass**：O2——−inf 为 max 幂元 ∧ exp 后 sum 幂元，m=−inf 时 NaN 传播两链一致（推演成立），pad 列数变化（N_padded−N → align64(N)−N）不改变幂元中性 ✓；O3——融合减法舍入时机差 ≤0.5 ulp，独立推演最坏界：δlnS = δS/S ≤ N·max_{d<0}(e^d·|d|)·2⁻²⁴ ≈ 32512×0.368×6e-8 ≈ 7.2e-4 ≪ 1e-2 容差 ✓（设计的 11.1·2⁻²⁴≈6.6e-7 为更紧的单项界，结论同向成立）；O4——max 结合/交换律精确无舍入 ✓、sum 为 fp32 分组重排树深级差（V2 ≤2 ulp 机器证实）✓；O6——上界复算 m+ln(Σ) ≤ 65504+ln(32512)=65514.39 < 65520（fp16 round-to-inf 阈值）且 Σ≥1（max 项贡献 exp(0)=1），契约空间封闭成立 ✓；fp16/bf16 舍入影响经 V4 位级 + NaN 模式对照覆盖，与 §8.2 容差标准一致 ✓。**机器验证与推演互补**：verify_equiv 重跑全 PASS（见机械复核段）。
- **向量化替代覆盖完整性 — pass**：§1.6.2 #1–#10 与 §3.3 伪代码、§6.1 循环表**三方交叉核对无遗漏**（装载提升/尾组掩码/max/exp/sum/行 epilogue/行循环/组循环/尾批分支/host 元数据——伪代码中不存在第 11 个计算点）；#1–#5 100% 2048-bit 显式内在（API 佐证全部亲核，见维度 1）✓。
- **不可替代理由充分性 — pass**：#6 行 epilogue（依赖归约终值的 tile 级标量，4 指令/行占比 ~0.9% 已量化，无法并入元素循环——理由具体）；#7 行循环（SimdVF 无行级并行原语 + 行间交错属编译器调度层，Stage 4 观察项——结构性理由，非"实现简单"）；#8 组循环（两遍扫描 pass2 依赖 pass1 m 终值的行内顺序依赖 + 静态边界 gate 要求 + P-LAYOUT-TAIL-STATIC-EXTENT measured 注记）；#9 尾批 TIR if（运行时边界语义，example_gemm L47 + SIMT 兄弟变体 86+61 用例实证先例）；#10 host 元数据 n/a ✓。
- **替代 API 佐证 — pass**：全部替代 API 佐证已亲核（维度 1）；SimdVF 为用户指定 + 热点/结构理由齐备（O4 收益量化），不违反 SimtVF 优先原则（该原则适用于无指定时的默认推导）✓。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 按规约类必选清单枚举**水平归约（lane→N）vs 垂直扫描（lane→M）**两种 lane 映射（#1/#2）+ SIMD 特有正交轴（归约时机 #5、驻留层级 #6）+ 跨核分解 #3 + 转置布局 #4，六候选全量化弃选理由（#2 利用率 rows/64≤25% → 吞吐 >4× 劣化；#3 流量 +50% 起步 + (max,sum) 合并语义超出 additive atomic 表达力〔SIMT Stage 4 [DESIGN_LIMIT] 实测结论引用〕；#4 repack 纯开销无轴收益；#5 被 O4 取代 −50%；#6 VRF 128KB 贴顶回归）✓；轴质量评分五项齐备（整除性/尾 lane 浪费/累加链形态/repack 代价/容量影响）✓；**源码轴选择对照已做**（GPU N 轴并行是输入非结论，独立复核声明在位）✓；弃选前提亲证（transpose 仅 GM→L1，见上）✓，无"未考虑"式空行、无把 I/O 契约当核内布局依据 ✓。
- **实验裁决模式 — pass（豁免成立）**：判定裕度不依赖未实证常数——256B/cycle（VRF↔UB）与 51/8 档均有 `estimate_latency.h` 源码实值佐证；VRF 压力差为 50× 量级结构差；唯一真不确定量（special op 粒度）已按风险 5 披露且证明两种解释下选择不变；config 级甜点交 Stage 4 扫描（§5.2 校准指引 + §8.3 A/B 计划含判定口径）✓。
- **布局决策与实现一致 — pass**：选定布局（x_ub (SUB,N_pad) 紧凑 + VRF 流式）= §3.3 伪代码 buffer 形状 = §4.3/§4.5 预算 = §6 组循环内层向量维（64 元素组），三方一致 ✓；无重排路径需求（I/O 原生行主连续同构）✓。
- **与下游章节一致 — pass**：§3.1 四步以 §1.6.1 优化后公式为输入（vexpdif 融合/单次 vcmax/vcadd/ONEPT 存储均入表）✓；§6 无 §1.6.2 未论证的标量循环 ✓。
- **建议级备注（3–6/6，均为估算行口径小疵，不改变任何采纳/弃选结论）**：
  - 备注 2（V1 内嵌表描述瑕疵）：见机械复核段——33×300 bf16 实测 1 ulp 而内嵌表零值清单列为 0（门控值 ≤1/违反率 0 不受影响）。
  - 备注 3：D-2 "主选 vs 基线直译 VALU 计算项 0.93×" 独立复算为 **≈0.91×**（(4/51+1/8)/(5/51+1/8+2/64/51)=0.2034/0.2237）——2pp 估算口径漂移，方向与量级结论（省 ~7–9% VALU）不变。
  - 备注 4：D-2 UB 重读行 "0.0078 cycle/elem ≈0.6K（13×）" 混用口径——0.0078 为 fp16 2B/elem 值（对应余量 26×），13× 对应 fp32 4B/elem 的 0.016（§0.6 R2 口径）；两口径均支撑"非瓶颈"，取 13× 为保守方向，无碍。
  - 备注 5：R3 基线行 "GM Bytes/elem 3.4（非对齐 N=300）" 将 N_padded 驻留膨胀计入 GM 流量口径——masked 载入的 GM 实读仍 2B/elem（if_then_else 谓词只读 j<N），3.4 实为驻留口径（R4 行 "+71% 驻留浪费" 已正确单列）；该行仅用于弃选候选对比，排序结论不受影响。
  - 备注 6：O3 论证中 "|x−m| ≤ ~11.1（fp16 域 max-gap）" 措辞不精确——11.1≈ln(65504) 是**输出精度有效贡献域**上界（exp(−11.1)≈1.5e-5），而 "fp16 域 max-gap" 在 pattern-library 语境为 131008（TRAP-NAIVE-ONLINE-MERGE）；数值结论（相对误差 ≪ 容差）经独立推演成立（见上），建议措辞更正以免误读。

## 综合判定

全部阻塞级维度（0/1/2/3/4/6/8）pass，建议级维度（5/7）pass，维度 8 带 6 条建议级备注（warn，不阻塞）。机械复核 D-1 重跑全 PASS；D-5 两项 fail 经工具源码级归因判定为解析假阳性（VRF 对照行误计入 UB 桶 / 非持久 grid 模型不适用于 persistent 设计），人工复算通过并已在机械复核段完整披露。**结论: 通过**——DESIGN.md 可作为 Stage 3 实现输入冻结；6 条建议级备注（行号引用偏移 ×1、估算口径小疵 ×3、表述措辞 ×2）供 Stage 3 实现期顺手修正，无需回退 Stage 1。
