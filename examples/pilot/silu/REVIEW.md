# silu 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/pilot/silu/DESIGN.md
- 任务类型: 新算子开发（非迁移，first_design）
- 检视维度: 8 项（API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析；维度 0 标记 n/a）
- 阻塞级问题数: 0
- 建议级问题数: 3（详见各维度说明内「建议级小疵」，不构成阻塞）
- 检视日期: 2026-10-16
- 环境: Ascend950PR_9579（cube_core_num=28 / vector_core_num=56，与 §5.5 实查记录一致）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: 不适用（无采纳优化项）——豁免依据核对成立：§1.6.1 明确「采纳 0 项——无优化空间」且逐项量化否决（6 候选见 §1.6.0 R1 表）、调度指令确认豁免、§10.2 交付清单标注「不适用」；补充等价性证据为设计期探针实测（`history_version/design_probe_simt_exp.py`，2026-10-16，vs fp32-opmath golden max abs diff 3.906e-3 < 1e-2，vs direct-fp16 golden 亦 PASS——探针源码已 Read 核对，公式/golden 口径/容差与 §3.4、§8.1、§8.2 陈述完全一致）。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design examples/pilot/silu/DESIGN.md`）:
  - `ub_budget`: **skip**（"no per-level buffer tables parsed"）→ 人工已核：§4.5 预算表 a_ub (2048,)×2B×2ver=8192B + b_ub 同=8192B，总计 16384B=16KB ≤ 194.4KB（SimtVF 216KB × 90% 折扣，`CONST-capacity-a5` assumption，占用 8.2%）——算术复算无误，buffer versions 已计入，条目与折扣口径已在 §4.5 披露 ✓。
  - `l0c_budget`: **skip**（"no block_M x block_N values parsed"）→ 人工已核：纯 Vector 算子无 Cube 段（§3.5.1/§5.1），L0C/L0A/L0B/L1 均不适用，skip 合理 ✓。
  - `core_split`: **skip**（"insufficient parsed dims"）→ 人工已核：TOTAL=131072=2^17，NUM_BLOCKS=32，TILE=2048，ITERATIONS=131072/(32×2048)=2 精确整除（无尾块）；grid 32 ≤ 56 AIV 单波（占用率 57.1%）；核数实查记录（查询代码 + 返回值 28/56）与检视环境一致；核内串行边界 ITERATIONS=2 为编译期常量 ✓。
  - `r3_metrics`: **pass**（"all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"）。

## 检视详情

### 0. 源算子理解与迁移分析 — n/a
非迁移任务（DESIGN.md 无 §0，调度未传 source_op_path），本维度不适用。golden 独立性在维度 6 内核对。

### 1. API 可行性 — pass
- **API 存在性（逐条核对佐证，全部命中）**：`T.Kernel(32)` 一维（example_vecadd.py:21 同构 + ascend-constraints.md §1「一维 Kernel，无 threads 参数，threads= 直接 TypeError」原文核实）；`T.alloc_shared`（vecadd:22-24）；`T.annotate_buffer_versions`（vecadd:25）；`T.Pipelined(iterations, num_stages=2)`（vecadd:26）；`T.copy` GM↔UB（vecadd:28,42）；`T.SimtVF(threads=)`+`T.Parallel`（vecadd:31-33、rmsnorm:54）；`T.exp`/`T.float16`（`tilelang/ascend/language/__init__.py:5` 经 `from tilelang.language.common import *` 再导出，运行时静态内省确认 `T.exp/T.float16/T.Pipelined/T.Parallel/T.Tensor/T.prim_func` 全部存在于 ascend 方言命名空间；探针编译+运行通过为实证）；`tir.exp → simd_vexp()` 升降低径亲核 `src/ascend/transform/ascend_simdvf_lower_parallel.cc:412-413`（设计引 409-412，块起于 409，内容一致）、`DivNode → simd_vdiv()` 同文件 :405-406 ✓。
- **编程模式匹配**：simt + `auto_schedule: yes`，§7 无手动 flag 混写 ✓；无 GEMM 段，不涉 mixed/sid/dual_copy ✓。
- **SimtVF 优先**：默认 SimtVF（threads=512，4 el/thread 与 vecadd 8192/2048 同比例——已核 vecadd:12 `tile=8192, num_threads=2048`）；SimdVF 唯一候选收益 vexpdif 已量化否决且 API 限制亲核属实（simd.py:1191-1192 `raise TypeError("T.simd.vexpdif requires matching float32 vectors")` + docstring「Widening float16 ... not represented by this API」）✓。§3.4 表 9 项 API 逐条有 example/源码/探针佐证，无凭记忆断言。

### 2. 内存层级规划 — pass
- **搬运路径**：GM→UB（MTE2）/ UB→GM（MTE3）via `T.copy`，在支持集内（vecadd 同构实证）；无手写 NZ、无越级路径 ✓。
- **UB 预算**：16KB（含双版本）≤ 194.4KB（216KB×90%，`CONST-capacity-a5` assumption）——机器 skip 已人工复算通过（见机械复核段）；条目引用与折扣口径在 §4.5/§5.3/§9.1 三处一致披露 ✓。rmsnorm:17-19「SIMT-usable 216 KiB UB」注释为 216KB 口径的 examples 级旁证 ✓。
- **L1/L0**：不适用（纯 Vector）✓。**Buffer 层级标注**：a_ub/b_ub 均标注 UB + `T.alloc_shared` ✓；无 GM 中间缓冲（§4.3）与 R3「无中间 GM 往返」一致 ✓。

### 3. Tiling 策略 — pass
- **Block 划分与 grid**：TILE=2048 三重依据成立（2048=16×128 el fp16 向量宽整除——`CONST-valu-model`「2048-bit=128×fp16」已核；4KB 连续 MTE 事务；UB 驻留 16KB）；grid 一维 `T.Kernel(32)` ✓。
- **对齐约束**：GEMM 分形对齐不适用；扁平连续访问 32B 行对齐 ✓。
- **非整除处理**：契约 shape 精确整除（131072=32×2048×2）+ host assert 显式防御 + §5.4 泛化策略（padding/shape 特化列为 Stage 3/4 项）✓。grid 整除性论证独立复核成立：2^17 无因子 7 → 28(=4×7)/56(=8×7) 网格必不整除，探针 v1 28-blocks 静默漏写实证（RETROSPECTIVE.md Stage 1 D 类条目）记录在案 ✓。
- **分核三要素（core-split-strategy.md §1 逐项核对）**：① grid 划分含**核数实查**（§5.5 记录查询代码 `torch.npu.get_device_properties(0)` + 返回值 cube 28/vector 56，与本检视环境一致）+ 单波判定（32≤56）+ 非 persistent 理由（仅 2 迭代）+ 未选 56 的量化理由（整除性硬约束）✓；② AIC/AIV 分工：纯 Vector 仅占 AIV，AIC 不参与，codegen 自动指派（§1 ②口径）✓；③ 同步策略声明 `auto_schedule: yes` + 依赖边界枚举（MTE2→VF→MTE3 + 双版本冲突交 AutoSchedule；跨 block 无依赖）+ 调度提示位置（`T.Pipelined`）✓。核内串行边界静态（ITERATIONS/TILE/NUM_BLOCKS 编译期常量）✓。参数级 A/B 变体（64/128 网格、TILE∈{1024,4094}… §5.4）预注册给 Stage 4 按 §2.4 裁决，处置得当。

### 4. 技术约束检测 — pass
- 一维 grid ✓（`T.Kernel(NUM_BLOCKS)` 无 threads=；线程域在 `T.SimtVF(threads=512)` 内——ascend-constraints.md §1/§2 原文核对一致）；GEMM 分形/NT/L0C 约束均不适用（无 Cube 段，§3.5.1 自查表已覆盖且与检视判断一致）；Host 侧 flatten/reshape 为纯 stride/shape 元数据视图，符合 ascend-constraints.md §4「禁止改动输入内容/物理排布」原文 ✓。§9.2 常见错误表（grid 整除违例/fp16 exp 精度/饱和 flush/threads 误用/阴影导入）与本项目已知陷阱记录吻合。

### 5. 循环与同步 — pass
- **循环结构**：§6.1 三层明确（grid 隐式并行 / `T.Pipelined(2, num_stages=2)` tile 流水 / `T.SimtVF+T.Parallel(2048)` 元素级向量化），与 §3.2/§3.3/§6.2 伪代码完全一致 ✓。ITERATIONS=2 下流水收益有限（§6.3 自认 ~15% 量级）但保留理由合理（保持 TILE 最优形态 + vecadd 模板一致性 + Stage 4 可平滑扫参）——非阻塞。
- **同步与模式匹配**：`auto_schedule: yes` 与 §7 一致，无手动 flag；依赖边界表（§7.2）覆盖装载/计算/写回/双版本四类 ✓；`pass_configs={}` 正确（完全手动核才需 False）✓。
- **尾块处理**：§6.4 无尾块（精确整除）+ 泛化防御指引 §5.4 ✓。

### 6. 验证方案 — pass
- **Golden**：§8.1 `torch.nn.functional.silu(x.float()).to(x.dtype)`，device="npu"，fp32 opmath 路径与 kernel fp16 直算路径相互独立（非自证）✓；direct-fp16 golden 双口径探针实证 ✓。
- **L0 测试计划**：L0-1 契约 [128,1024] fp16 + L0-2 同整除类 [512,512]（复算：2^18/(32×2048)=4=ITERATIONS ✓）；`torch.testing.assert_close` ✓。
- **精度标准**：fp16 → atol/rtol 1e-2，符合 AGENTS.md 分档 ✓；探针 max diff 3.906e-3 < 1e-2 实证 ✓。
- **边界语义独立推演**：±inf/NaN 行逐项复核成立（kernel：+inf→+inf/1=+inf、-inf→-inf/inf=NaN、NaN 传播；torch：sigmoid(inf)=1→inf、sigmoid(-inf)=0→-inf×0=NaN——两路径同结果 ✓）；深饱和 flush 最大偏差 ≤1.7e-4 论证复核成立（|x·e^x| 在 fp16 exp 上溢边界 x≈-11.09〔e^11.09≈65513≈65504=fp16 max ✓〕处取峰 ≈1.69e-4，深负区单调衰减）≪ atol ✓。
- **建议级小疵（不阻塞）**：§8.3 表两处期望值数字与独立计算有微小出入——① x=-11.0 行「≈-1.66e-4」：精算 -11/(1+e^11)≈-1.84e-4（若意指 -11.09 边界则 ≈-1.69e-4）；② x=-30 行「真值 -6.9e-12」：精算 -30·e^-30≈-2.8e-12。两处偏差（~1.8e-5 / ~4e-12）均比 atol 1e-2 小 ≥3 个数量级，且 Stage 3 Boundary 层判定以 golden 实算值为准而非查表常数，不影响任何判定与结论。建议 Stage 3 实现 Boundary 用例时以 golden 实算为期望值来源（或修正表中数字）。

### 7. 完整性与一致性 — pass
- **章节齐全**：非迁移 10 章齐备（§1 概述含 §1.6 / §2 模式 / §3 API / §4 数据与内存 / §5 Tiling 含 §5.5 分核 / §6 循环 / §7 同步 / §8 验证 / §9 风险 / §10 交付）✓。
- **无占位残留**：全文经扫描无任何模板占位/待办标记（§10 的「待实现（Stage 3）」为交付状态标注，非占位性质）✓。
- **同类引用真实（Glob/Read 逐一核对）**：`examples/ascend/example_vecadd.py` ✓、`examples/ascend/example_rmsnorm.py` ✓、`examples/pilot/silu/history_version/design_probe_simt_exp.py` ✓；标准/参考文件 `ascend-constraints.md`（tilelang-op-design/references/）✓、`core-split-strategy.md` ✓、`hardware-cost-model.md §2` ✓、`algorithm-candidates.md` ✓、`pattern-library/constants.md` ✓；源码行号 `simd.py:860`（vexp）/`simd.py:1182-1195`（vexpdif fp32-only）/`ascend_simdvf_lower_parallel.cc:409-412` 全部实核命中 ✓。
- **内部一致**：§1.4=§1.6.0 选定算法=§3.1 公式；§3.2=§3.3=§6.2 结构；§4.5=§5.3=§9.1 容量口径；§5.5③=§7.1；§1.6.3 布局→§3.3 线性索引→§4 一维 buffer→§6.1 内层向量维，交叉核对无矛盾 ✓。
- **建议级小疵（不阻塞）**：§1.6.0 调研结论称「『除法乘倒数 / CSE / cast 链合并』候选已逐项评估见 R1 #2/#6」——「cast 链合并」的指针不全：实质评估在 R1 #5（vexpdif 的 fp16↔fp32 双向 cast 净亏论证），且 kernel 路径纯 fp16 本无 cast 链可合并（trivially 不适用）。实质覆盖成立，仅引用指针建议补正为「#2/#6/#5 + 纯 fp16 路径无 cast 链」。

### 8. 算法优化分析 — pass ⭐
- **§1.6 章节存在性**：1.6.0/1.6.1/1.6.2/1.6.3 四子节齐全 ✓；轻量调研分级符合 algorithm-research.md §2（单步逐元素/无规约/无窗口/无矩阵 →「四问各一行结论即可，但必须逐问回答并给依据」）且实际深度超出行级要求（R1 六候选量化表）。
- **调研完整性（四问逐项）**：R1 候选表含基线行（#0）+ 6 个等价变形候选，覆盖类别完整（代数重排 #1/#2、降代换 #3、SFU 融合指令 #5、换底 #4、CSE #6）✓；R2 有明确结论（「有——基线本身即单遍」+ 结构依据，与 algorithm-research.md §3 R2 ③「逐元素映射无跨元素依赖，单遍即天然在线」判据同源）✓；R3 四口径齐全（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值，基线在表，折算口径显式声明），`r3_metrics` 机检 pass ✓；R4 六项检查清单逐候选评估（单元匹配/容量/对齐/静态边界/流水融合/跨核），容量淘汰引用 `CONST-capacity-a5`、带宽引用 `CONST-mte-bandwidth-model`（D-2 条目纪律满足）✓。
- **调研结论正确性（独立复核，不轻信自述）**：
  - **负向断言复核**：① 「A5 无 tanh 直连 SFU intrinsic」——亲自查证：`tilelang/ascend/` 全目录大小写不敏感检索 `tanh` **零命中**，simd.py 一元 SFU 仅 vexp/vln/vsqrt（+vabs/vneg/vrelu），断言成立；② 「候选库无 silu/sigmoid 专条目」——algorithm-candidates.md 全文 grep 无 silu/sigmoid 命中，命中行为 ALG-elementwise（family: [elementwise, activation]），其 R1 三候选（CSE/除法乘倒数/cast 链合并）均已在设计 R1 表实质评估（#6/#2/#5，指针小疵见维度 7）；检视侧复跑 `kb_search.py "silu sigmoid elementwise"` 返回 no entries matched，与设计「无专条目」结论一致（设计所述「命中族行」以直接 grep 候选库为准核实）；③ 「exp2 换底收益不成立」——softmax 族 exp2 缩放的前提是硬件 exp2 快于 exp，A5 `vexp` 即自然指数直连（simd.py:860，special_func 8 ops/cyc 档），方言无独立更快的 exp2 intrinsic（simd.py def 清单核实），换底徒增 1 vmul，断言成立。
  - **否决论证独立推演（调度重点：rcp/tanh/exp2 在 fp16 精度与 A5 VALU/SFU 吞吐下的代价）**：#2 rcp+mul——复算 1/13+1/51−1/13=1/51≈0.0196≈设计「+0.02 cyc/el」✓，fp16 下 rcp+mul 引入两次舍入 vs div 单次（~1 ulp 级，1e-2 容差可覆盖）但零收益纯增 op，否决成立；#3 tanh——无 intrinsic 时须 exp 合成（如 1−2/(1+e^x) 形态 ≥1 exp+1 div+2 常规，且 e^x 在 x>11.09 上溢使有效域劣于基线 e^{-x} 路径），op 数与数值域双劣，否决成立；#4 exp2——见上③，+1 mul（1/51 cyc/el）无对冲收益，否决成立；#5 vexpdif——源码实证 fp32-only+SimdVF-only，省 1 vneg≈0.02/0.24≈8%（复算 0.0196/0.2411=8.1% ✓）vs fp16 双向 cast ≥2 cvt（≈0.04 cyc/el）+ 模式切换，净亏成立；#1 乘法形式——op 数 4→4 无净收益成立（另 e^x 上溢域劣化，设计未提但不影响结论）；#6 CSE——`-x` 单用、`x` 寄存器驻留，无重复子式 ✓。**结论：6 项否决全部经独立推演/源码查证成立，无「明显漏掉的已知候选」**（一元逐元素映射的等价变形类已穷举；近似类〔hard-sigmoid/多项式〕非数学等价，不属 R1 范畴，排除正确）。
  - **复杂度算术复算**：0.52M 向量 op（131072×4）✓；当量 6.375+3.923+2=12.298/el ×131072≈1.612M≈「1.61M」✓；访存 256KB 读+256KB 写=512KB、无中间 GM 往返 ✓；单遍 ✓；缓冲峰值 8KB（算法级单版本）与 §4.5 16KB（实现级双版本）口径自洽 ✓。**D-2 估算复算**：计算项 4096×(1/8+1/13+2/51)=4096×0.2411=987.7≈988 cyc ✓；流量项 2×(4096/100+90)+2×(4096/115+180)=261.9+431.2=693.2≈694 cyc ✓；比值 988/694=1.42≈「1.4×」✓——常数取用与 `CONST-valu-model`（51/13/8 ops/cyc）、`CONST-mte-bandwidth-model`（AIV MTE2 100 B/cyc base 90 / MTE3 115 B/cyc base 180）逐值核对一致，assumption 状态与「只做相对比较、绝对时延以 do_bench 为准」纪律已声明 ✓。ALG-elementwise「通常 MTE 带宽主导」代价提示与本设计「SFU 计算主导」结论的偏离有量化依据（exp/div 慢档 op 占比高）且对照了搬运下界口径，属合理偏离 ✓。
  - **并列候选裁决**：R3 仅基线存活（其余 op 数只增不减），无判定裕度问题 ✓；grid/TILE 参数级变体的 assumption 敏感性已显式声明并预注册 Stage 4 A/B（§5.4，三变体+整除约束+§2.4 判据）——参数扫描而非结构裁决，处置得当。建议级小疵（不阻塞）：§5.4 A/B 未给明确数值判定阈值（如「延迟优 ≥5% 则采纳」），建议 Stage 4 执行时补充收敛判据。
- **调研→设计一致性**：§1.6.0 选定基线公式族 = §1.4 算法描述（4 op 单遍）= §3.1 公式拆解（neg/exp/add/div 四步）= §6 循环结构（单遍 Pipelined + Parallel，无两遍扫描残留）✓；非迁移任务无 §0 处置核对项。
- **数学等价优化完整性（§1.6.1）**：「无优化空间」结论有依据且系统化（op 组合最小性逐项否决 + 数值双向饱和自稳定论证 + 访存单遍 I/O 下界）；稳定化不适用论证独立成立（sigmoid 形式无 softmax 式 exp(正大数) 溢出路径：x 大正→e^{-x} 下溢→y=x；x 大负→e^{-x} 上溢→y=-0.0，偏差 ≤1.7e-4——推演与 §8.3/§9.3 一致）✓。采纳 0 项 → verify_equiv 豁免合规（见机械复核段）。
- **向量化替代覆盖（§1.6.2 ↔ §3.3/§6 交叉核对）**：盘点表 3 个计算点与 §3.3 伪代码/§6.2 逐行比对**无遗漏**（逐元素映射→T.Parallel #1；begin 标量索引 #2；T.copy 搬运 #3）；#1 已替代（lowering 源码佐证 cc:412-413 + 探针实证）✓；#2 保留理由充分且具体（block 级索引/任务映射类：每 block 每迭代 ≤3 标量 op、全 kernel ≤192 次〔复算 32×2×3=192 ✓〕vs 131072 元素级，非热点，属 skill 认可的理由类别）✓；§6 无逐元素标量循环 ✓。
- **替代 API 佐证**：T.Parallel 自动向量化（SimtVF 优先 ✓，无 SimdVF 降级故无热点理由义务）；§1.6.3 方案 3 所引 `T.simd.vintlv/vdintlv` 实存（simd.py:1052/1067）——弃选行的 repack 机制前提真实 ✓。
- **向量化轴与布局决策（§1.6.3）**：纯单轴逐元素算子（豁免类）仍给出 4 候选矩阵，超出行级要求；逐元素类必选候选（广播轴×向量轴对齐）不适用声明明确（无广播/窗口/规约轴）✓；弃选理由量化：#2「同一连续轴等价切分、访存与向量效率相同、纯增索引」（结构论证，不依赖未实证常数）✓、#3「跨步 1024 破坏连续装载 + 重排开销」（MTE 跨步低档带宽有 constants.md:119 文档化依据；I/O 契约未被当作内部布局依据——设计显式声明「I/O layout 是契约、核内布局是设计变量」）✓、#4「host permute ≥512KB 额外流量 + 违反 ascend-constraints §4」（原文核对属实）✓。**pattern-library 对照**：layout.md/elementwise.md 均为 A5 冷启动清零状态（无实测条目可矛盾）✓；代价论断全部落在 constants.md assumption 条目并已披露 ✓。
- **实验裁决模式**：§1.6.3 布局判定不依赖未实证常数（方案 1 vs 2 结构等价、3/4 文档化+约束否决），无三件套义务；参数级变体已预注册 Stage 4（见上）✓。
- **布局决策与实现一致**：选定扁平 1D 轴 → §3.3 线性偏移索引、§4 一维 buffer (2048,)、§6.1 T.Parallel(TILE) 内层向量维=扁平元素轴，三方一致 ✓。
- **与下游章节一致**：§3.1 以 §1.6.1 优化后公式（=基线）为输入 ✓；§6 无 §1.6.2 未论证的标量循环 ✓。
