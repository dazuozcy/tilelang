# gemm_silu — RETROSPECTIVE

> 追加式日志：每个 Stage 执行追加一个章节，不覆盖历史。

## Stage 1 (design) Retrospective — first_design — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| other（探针执行环境） | 脚本式执行 python3 <script>.py 时 import tilelang 解析到 tilelang-mlir-fork（环境 PYTHONPATH 含该路径且无 ascend 模块），报 ModuleNotFoundError: No module named 'tilelang.ascend'；-c 交互式不受影响（cwd 优先） | 本任务探针首次运行失败日志（design_probe_silu_form.py 未加 PYTHONPATH 前缀时 traceback） | tilelang-op-design skill Phase 2 第 5 步（D-1/D-3 执行指引）补一句：「脚本式执行需 PYTHONPATH=<repo_root> 前置，环境 PYTHONPATH 指向 tilelang-mlir-fork 时 import 会错仓」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| silu/sigmoid 类激活公式在 SimtVF 路径选直除式 z/(1+exp(-z))：实测 0.902× 优于乘倒数式；tanh 恒等变形 1.80× 劣化——标量 tanh lowering 非 CONST-valu-model special_func 8 ops/cycle 档位（推断为软件例程），激活 epilogue 禁用 tanh 形式 | D | examples/pilot/gemm_silu/DESIGN.md §1.6.1 探针表；examples/pilot/gemm_silu/history_version/design_probe_silu_form.py | PYTHONPATH=/home/z00875089/tilelang-fork python3 examples/pilot/gemm_silu/history_version/design_probe_silu_form.py（输出三形式 µs 与比值） | tilelang-fork fc32f47c / CANN 9.x / Ascend950PR_9579, 2026-10-16 | tilelang-op-optimize/references/pattern-library/elementwise.md（或 constants.md 作 CONST-valu-model 的 tanh 反例注记） |
| gemm+非线性 epilogue 融合算子设计案例（CV 融合 + dual_copy M-split 双 AIV epilogue + cast 后置语义），Stage 3 实现结果回填后可作案例索引 | C | examples/pilot/gemm_silu/DESIGN.md（§1.4 五步表 / §3.3 伪代码 / §5.5 分核三要素） | none（待 Stage 3/4 结果回填后转正） | tilelang-fork fc32f47c / CANN 9.x | pattern-library/cases.md（待回填） |

### Transferable Lessons（可迁移教训）

- 脚本式执行 python3 脚本时必须前置 PYTHONPATH=<tilelang-fork 仓库根>：本环境默认 PYTHONPATH 含 tilelang-mlir-fork（无 tilelang.ascend 模块），会把 import tilelang 解析到旧栈；`python3 -c` 形式不受影响（cwd 优先解析）。
- silu/sigmoid 类逐元素激活在 Ascend SimtVF 路径用直除式 z/(1+exp(-z))，不要用 tanh 恒等变形（0.5z(1+tanh(z/2))）——实测 1.80× 劣化，标量 tanh 是软件例程而非 SFU 档位。
- 设计期探针（D-3）用最小元素级 kernel（UB 驻留数据 × N 次迭代，do_bench msprof）可在 ~3 分钟内裁决公式级选型，比把不确定性留到 Stage 4 A/B 便宜；本任务由此在设计期就否决了成本模型看好的 tanh 形式。
- GEMM epilogue 注意 dav-3510 约束：L0C→UB dual_copy 禁止类型转换——fp32 累加器以 fp32 落 UB、cast 在 VF 段完成；这恰好同时是精度最优路径（cast 后置消 z 预舍入）。
- 512³ 小规模 GEMM 的 grid 取舍：最小 L1 重读流量（大 tile）vs 核占用率（多 blocks）在 assumption 常数下估算无净收益时，选与模板结构一致的大 tile 单波方案，把 grid/tile 扫描明确列入 Stage 4 调优项并在设计文档记录权衡数字。

## Stage 2 (review) Retrospective — first_review — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| other（机检工具 design_calc_check.py） | `check_r3_metrics` 的 `_section(text,"1.6.0")` 取**第一个标题含 marker 的章节**——本设计 §1.4 标题「算法描述（与 §1.6.0 调研选定一致…）」先于真 §1.6.0 命中，提取到 §1.4 正文（不含 R3 表）→ r3_metrics 误报 fail（missing 全部四指标）；真 §1.6.0 章节体四指标正则复跑全 True | .agents/tools/design_calc_check.py:67-81（_section 实现）、:311-331（check_r3_metrics）；examples/pilot/gemm_silu/REVIEW.md「机械复核」段（误报定性证据与复跑输出） | design_calc_check.py：_section 匹配改为「标题以章节号开头」（如 `^#{1,6}\s*§?1\.6\.0\b`）而非子串包含；或 check_r3_metrics 改为定位含四口径词的表格行所在章节 | R |
| other（机检工具 design_calc_check.py） | `ub_budget` 对本设计 §4.5 表格式（Buffer/Shape/dtype/大小(Bytes) + 小计/×2 版本行）解析 skip（"no per-level buffer tables parsed"），UB 预算只能人工复算——机检拦截位（VP-2026-0009 混合字节口径）对 shape×dtype 型预算表失效 | examples/pilot/gemm_silu/DESIGN.md §4.5（表格式）；design_calc_check 输出 ub_budget=skip；REVIEW.md 机械复核段（人工复算 96KB ≤ 194.4KB） | design_calc_check.py ub 解析支持「shape+dtype 列」推算字节，或 design-template.md §4.5 建议表内附每 buffer 的 KB 数值列以便机检 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 机检 fail 消费前须做误报定性：design_calc_check 的章节提取可被「标题内引用目标章节号」劫持（本例 §1.4 标题含 "§1.6.0" → r3_metrics 误报 fail）——按真章节体复跑检查正则确认真伪后再决定是否阻塞，误报证据记入 REVIEW.md 机械复核段 | R | examples/pilot/gemm_silu/REVIEW.md「机械复核」r3_metrics 行；.agents/tools/design_calc_check.py:67-81 | python3 .agents/tools/design_calc_check.py --design examples/pilot/gemm_silu/DESIGN.md（观察 r3_metrics=fail）；再按 `#### 1.6.0` 至 `#### 1.6.1` 章节体复跑四指标正则（全 True） | tilelang-fork（工作副本）2026-10-16 | tilelang-design-review/SKILL.md Phase 1 第 7 步（补一句：fail 项先做误报定性——按真目标章节复跑检查逻辑，确认为工具解析缺陷时记录证据、不阻塞，并提交工具修复价值点） |

### Transferable Lessons（可迁移教训）

- design_calc_check.py 的 r3_metrics/章节类检查按「第一个标题含 marker 的章节」提取正文——设计文档中其他章节标题若引用了 "§1.6.0" 等编号（如 §1.4 标题写「与 §1.6.0 调研选定一致」）会劫持提取导致误报 fail；检视方消费机检 fail 前必须按真章节体复核定性，设计方宜避免在非目标章节标题内嵌入其他章节编号。
- 设计文档内嵌机器验证结果表应与验证脚本的**输出口径逐列对齐**：verify_equiv.py 打印 fp32 max ulp（相对口径），DESIGN.md §1.6.1 表却写 fp32 max|Δ|（绝对口径）——判定列虽一致，追溯时无法直接核对；内嵌表照抄脚本打印列可消除该缺口。
- 检视核对源码行号引用要亲自 sed/grep 落位：本设计 intrin_rule_ascend.cc 引用行号偏移 1-3 行（exp2 实际 :48 vs 引 :46-47；tanh 实际 :54 vs 引 :57），论断实质为真但按行号跳转会落空——行号级引用属易腐信息，检视以「论断真伪」为准、行号偏差记建议级。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| other（conductor 分层测试指令 vs 设计契约） | conductor 的 L1 指令写「相邻 shape（16 对齐）」，但本算子 DESIGN.md 明确契约 shape 特化、非契约 shape 拒绝（无尾块/padding 路径），kernel 按 128 block 整除编译；16 对齐但非 128 整除的 shape（如 528）无法用「无尾块」kernel 覆盖，只能测 128 整除的「16 对齐」子集（256/128/384） | examples/pilot/gemm_silu/gemm_silu.py build_gemm_silu 的 M%TILE_M 校验与 L1 用例；DESIGN.md §8.4 | conductor 分层测试指令在「算子有静态契约 shape 且 kernel 特化」任务中，将「16 对齐」放宽表述为「block 整除的 16 对齐子集」或注明「若需任意 16 对齐需尾块/padding 路径，属契约外重设计」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| mixed kernel epilogue 模式「T.dual_copy L0C→UB → SimtVF 逐元素段（含 T.exp）→ T.dual_copy UB→GM」在 dav-3510 上直接编译运行通过，无需 §1.6.4 的 F1/F2 降级——此前仓内无直接先例（example_gemm 的 MIXED epilogue 为纯搬运、deepgemm 元素变换走 plain T.copy 链），本实现证明 AutoSchedule 能正确处理 dual_copy 间插 SimtVF 元素段的所有权与同步 | D | examples/pilot/gemm_silu/gemm_silu.py（§3.3 主选路径全量编译运行、L0/L1/L2/Boundary 全 PASS）；DESIGN.md §1.6.4/§9.2 风险项 | PYTHONPATH=/home/z00875089/tilelang-fork python3 examples/pilot/gemm_silu/gemm_silu.py --level all | tilelang-fork fc32f47c / CANN 9.x / Ascend950PR_9579, 2026-10-16 | tilelang-op-optimize/references/pattern-library/cases.md（gemm+非线性 epilogue 案例，§1.6.4 风险项回填为「已证实可行」） |
| 融合 GEMM+silu 的 fp16 输出 max|Δ| 由 fp16 出口量化主导而非累加/公式差：|z|~100 时 1 ULP=2^-4=6.25e-2、|z|~2000 时 1 ULP=2.0，实测各级 max_diff 恰落在 1-2 ULP；直除式 vs golden sigmoid 式仅在 fp16 舍入边界差 ≤1-2 ULP（端到端证实 verify_equiv 的 ulp≤14 结论）——精度报告应带量级上下文用相对容差（rtol=1e-2），裸 max|Δ| 值（如 6.25e-2/2.0）会被误读为超差 | D | examples/pilot/gemm_silu/gemm_silu.py --level all 输出的各级 max_diff（L0 3.125e-2~6.25e-2、L1-range 8× 2.0、Boundary-extreme 2.0）；DESIGN.md §1.6.1 verify_equiv 表 | PYTHONPATH=/home/z00875089/tilelang-fork python3 examples/pilot/gemm_silu/gemm_silu.py --level all | tilelang-fork fc32f47c / CANN 9.x, 2026-10-16 | tilelang-op-develop/SKILL.md Phase 4（补：报告 max_diff 时标注其所在量级与相对误差，fp16 出口 1 ULP 随 |z| 放大是预期行为） |

### Transferable Lessons（可迁移教训）

- mixed GEMM epilogue 用「T.dual_copy L0C→UB(fp32) → with T.SimtVF(threads=128): T.Parallel(TILE_M//2, TILE_N) 逐元素变换（含 T.exp/直除）→ T.dual_copy UB→GM」可直接编译运行，无需 deepgemm 式「for _ in range(1) + T.assume_no_conflict」隔离或 MixedKernel sid guard；fp32 累加器以 fp32 落 UB（dual_copy 禁类型转换），cast 在 SimtVF 段隐式完成（fp32 表达式赋值 fp16 buffer）。
- 带非线性 epilogue 的 fp16 输出精度判定：max 绝对差会随 |z| 增大按 fp16 ULP 阶梯放大（|z|~100 → 6.25e-2、|z|~2000 → 2.0），属正常量化；用 rtol=atol=1e-2 相对容差判定，勿被裸绝对差吓退，也勿把裸绝对差写进「超差」结论。
- 静态契约 shape 算子做分层测试时，L1「相邻 shape」只能是 block 整除的 shape 子集（无尾块/padding 路径）；任意 16 对齐但非 block 整除的 shape 属契约外，需尾块/padding 重设计而非直接塞进 L1。
