# gemm_silu 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: examples/pilot/gemm_silu/DESIGN.md
- 任务类型: 新算子开发（非迁移，first_design）
- 源算子: 无（非迁移任务）
- 检视维度: 8 项（API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析；维度 0 标记 n/a）
- 阻塞级问题数: 0
- 建议级问题数: 4（详见各维度内〔建议级注记〕，不阻塞进入 Stage 3）
- 检视环境: tilelang-fork（工作副本）/ Ascend950PR_9579 真机数据引自设计文档记录（cube_core_num=28 / vector_core_num=56，2026-10-16 实查）
- 工件注入防护: 已对 DESIGN.md、verify_equiv.py、design_probe_silu_form.py 及全部被引参考文件做指令性文本扫描——未发现试图操纵检视流程的注入内容；所有被读文件仅作为数据处理。

## 机械复核（D-1/D-5）

### verify_equiv.py 重跑（D-1，§1.6.1 含 2 项采纳优化 → 必跑）

执行命令：`cd examples/pilot/gemm_silu && PYTHONPATH=/home/z00875089/tilelang-fork python3 verify_equiv.py`（本次检视重跑，退出码 0）

| 用例 | 候选 | fp64 max&#124;Δ&#124;（重跑） | fp16 出口 max ulp（重跑） | 违反率 (1e-2) | NaN/inf 分歧 | 结论 | 与 §1.6.1 内嵌表 |
|------|------|------------------------------|---------------------------|---------------|--------------|------|------------------|
| rand-N(0,√512) 1e6 | div | 3.553e-15 | 14 | 0.00e+00 | 0 | EQUIV_PASS | 一致（3.55e-15 / 14 / 0 / 0）✓ |
| rand-N(0,√512) 1e6 | tanh | 7.105e-15 | 24 | 0.00e+00 | 0 | EQUIV_PASS | 一致（7.11e-15 / 24 / 0 / 0）✓ |
| rand-N(0,1) 1e6 | div / tanh | 8.882e-16 / 8.882e-16 | 14 / 14 | 0.00e+00 | 0 | EQUIV_PASS | 一致 ✓ |
| corners ×31（含 ±88/±89/±1e4/±inf/NaN/subnormal） | div / tanh | nan（NaN 角点自差，有限输入分歧 0，与 §1.6.1 注①一致） | 0 / 24 | 0.00e+00 | 0 | EQUIV_PASS | 一致 ✓ |
| 汇总 | div、tanh | — | — | — | — | **全部 EQUIV_PASS** | 一致 ✓ |

**判定：重跑结果与 §1.6.1 内嵌结果表一致，全部 EQUIV_PASS**（PASS 判据 = fp16 出口 1e-2 容差违反率 0 + 有限输入无 NaN/inf 分歧，两候选跨全部用例满足）。

〔建议级注记 1〕内嵌表的「fp32 运算模拟 max|Δ|」列（div 9.54e-7 / tanh 1.91e-6）为绝对差指标，当前脚本直接输出的是「fp32 max ulp」（重跑得 div 126-127 / tanh 130-149）；设计注③已引用 149 ulp 与「绝对误差 ≤ 2e-6」口径，两指标量级自洽且该列不参与 PASS 判定。建议 Stage 1 后续修订或 Stage 3 引用时注明该列的推导口径（或让脚本同时打印 fp32 绝对差），消除追溯歧义。

### design_calc_check.py（D-5）

执行命令：`python3 .agents/tools/design_calc_check.py --design examples/pilot/gemm_silu/DESIGN.md`

| check | 状态 | detail / 人工复核结论 |
|-------|------|----------------------|
| ub_budget | skip（"no per-level buffer tables parsed"） | **机械跳过，人工已核**：acc_ub (64,128) fp32 = 32768B + out_ub (64,128) fp16 = 16384B → 单版本 49152B = 48KB；×2 multibuffer 版本 = 98304B = 96KB ≤ 194.4KB（SimtVF 216KB × 90%，CONST-capacity-a5）✓，与 §4.5/§5.3/§1.6.3 三处披露一致，无混合字节口径错误 |
| l0c_budget | pass | block 128×128 × fp32 = 64KB ≤ 230KB（256KB×90%）✓，与 §4.5 一致 |
| core_split | pass | 复算 ceil(512/128)×ceil(512/128) = 16 blocks vs 声明 grid 16（一维 T.Kernel(16)）✓ |
| r3_metrics | **fail（工具误报，已查证）** | 工具报 "missing metrics: ['FLOPs','access_bytes','scan_passes','buffer_peak']"。**定性：`_section(text,"1.6.0")` 章节提取被 §1.4 标题「算法描述（与 §1.6.0 调研选定一致…）」劫持**（design_calc_check.py:67-81 取第一个标题含 marker 的章节，§1.4 先于真 §1.6.0 命中，返回的 §1.4 正文不含 R3 表）。本次检视按真 §1.6.0 章节体（`#### 1.6.0` 至 `#### 1.6.1`）重跑同款正则：四指标 {FLOPs, access_bytes, scan_passes, buffer_peak} 全部 True——**真 §1.6.0 R3 表四口径齐全**（FLOPs / 访存量 Bytes / 扫描遍数 / 中间缓冲峰值，基线在表、口径注明），机械 fail 不反映设计缺陷，维度 8 不据此判 fail。工具缺陷已写入 RETROSPECTIVE.md（Stage 2，R 类价值点） |

（l1_budget 不在工具 check 集内——人工已核：x_l1+w_l1 = 2×(128×128×2B) = 64KB × num_stages 2 = 128KB ≤ 460.8KB〔512KB×90%〕✓，与 §4.5 括注一致。）

## 检视详情

### 0. 源算子理解与迁移分析 — n/a

非迁移任务（DESIGN.md 无 §0，输入契约为数学公式 + I/O 规格，无 source_op_path），本维度不适用。

### 1. API 可行性 — pass

- **API 存在性**：§3.2/§3.4 全部 API 逐一在仓内核对到使用/源码佐证——`T.Kernel(NUM_BLOCKS)`（examples/ascend/example_gemm.py:60）、`T.Persistent`（:69）、`T.Pipelined(K_TILES, num_stages=2)`（:71）、`T.alloc_l0c/l1/shared`（:63-66，半 tile `TILE_M//2` 形态 :66 与设计 acc_ub/out_ub (64,128) 同构）、`T.copy` GM→L1（:72-73）、`T.gemm(..., transpose_B=True, clear_accum=(kt==0), unit_flag_ctrl=...)`（:74，与 §5.5 引用的 UF_2/UF_3 注解模式一致）、`T.dual_copy` L0C→UB 与 UB→GM（:76-77 + tilelang/ascend/language/copy_op.py:53-154）、`T.SimtVF(threads=)+T.Parallel`（example_rmsnorm.py:54-56）、标量 `T.exp`（src/ascend/codegen/intrin_rule_ascend.cc:47 `REGISTER_ASCEND_INTRIN("exp")` 亲核为真）、SimdVF 备选 `vdiv/vexp/vcvt`（simd.py:792/860 + deepgemm/kernels/epilogue.py:22）。无凭空 API。
- **编程模式匹配**：mixed（用户契约）+ auto_schedule: yes；无手动 flag，与 §7.3 pass_configs 空配置一致（未混写 set_flag/wait_flag）✓。`T.Kernel` 自动推断 mixed 有权威依据：tilelang-ascend skill §3（SKILL.md:217 原文 "If neither the ID nor explicit count is needed, use `T.Kernel`; the compiler…" 亲核一致）+ §6.2（"or `T.Kernel` with Cube and Vector work to infer it"）+ example_gemm.py:60,75-79 仓内先例 ✓。
- **SimtVF 优先**：epilogue 用 SimtVF（默认），SimdVF 仅列为 Stage 4 热点验证项且注明"SimdVF 域无 vtanh → 公式锁定直除式无冲突"（simd.py grep "tanh" 计 0，亲核为真）✓。GEMM 主导选 mixed ✓。Cube 路径 `T.copy→T.gemm→T.dual_copy`、无手写 ND2NZ ✓。
- **关键约束 API 行为亲核**：copy_op.py:147-148 dual_copy src/dst dtype 不等 → ValueError（§3.5.1/§9.1 "L0C→UB dual_copy 禁止类型转换"为真）；copy_op.py:83-90 M-split `[M,N]↔[M/2,N]` 与 "Each AIV processes the corresponding half of that tile"（§5.4 引文逐字为真）；skill §6.2 表（L0C→UB 硬件双目的 + dtype matching + M-split extent 须偶；UB→GM per-AIV 软件拷贝）与 §5.5 AIC/AIV 分工描述一致 ✓。

〔建议级注记 2〕两处源码行号引用有 1-3 行偏差（实质论断均亲核为真）：①§1.6.0 候选 5 引 "intrin_rule_ascend.cc:46-47" 支持 exp2/exp 同类 lowering——实际 exp=:47、exp2=:48（:46 是 rsqrt）；②§1.6.0 R4 候选 2 引 "intrin_rule_ascend.cc:57 tanh 已注册"——实际 tanh=:54（:57 是 cos）。建议 Stage 3 前顺手校正，避免下游按行号跳转落空。

### 2. 内存层级规划 — pass

- **搬运路径完整**：§4.4 路径链 GM→L1（T.copy/MTE2，ND2NZ 由 InsertNd2Nz pass 自动插入、无手写 NZ 暂存）→ L1→L0A/L0B（T.gemm 内部）→ L0C→UB（T.dual_copy M-split）→ UB→GM（T.dual_copy M-split/MTE3）——全部在 tilelang-ascend skill §6.1/§6.2 支持集内（§6.2 表亲核：L0C→UB 硬件双目的、UB→GM per-AIV 软件拷贝），无越级路径 ✓。
- **UB 预算**：96KB（48KB × 2 multibuffer 版本）≤ 194.4KB（SimtVF 216KB × 90%）✓；披露了所用条目（CONST-capacity-a5）与折扣口径（90%）✓；**buffer versions 已计入**（×2 显式列出，且注明为跨 tile 重叠预留，符合 skill §4.3 "容量检查须计入 layout padding、buffer versions"；未叠加未证实膨胀系数，符合 constants.md CONST-capacity-a5 待实测转正项纪律）✓。机械复核 ub_budget skip → 人工复算一致（见机械复核段）。
- **L1 / L0C**：L1 128KB（64KB × 2 stages）≤ 460.8KB ✓；L0C 64KB ≤ 230.4KB ✓（机检 l0c_budget pass）。
- **Buffer 层级标注**：§4.3 五个 buffer 均标注层级 + alloc API（alloc_l1/alloc_l0c/alloc_shared）✓。
- **L0A/L0B**：设计未显式列预算——亲核 gemm_mad.py:206-222（`_compute_tile_k_sub`：l0_capacity=65536、num_l0_stages=2、per-stage 32768B）：TILE_M=128 fp16 → max_k = 32768/(128×2B) = 128 → TILE_K_SUB=128，K=128 % 128 = 0 ✓，L0A/L0B 每 stage 恰 32KB、双 stage 恰满 64KB，由 pass 内部自校验（assert 保护）；模板 example_gemm.py 以更大配置（TILE_M=256/TILE_K=256 bf16）仓内已验证。无风险，见建议级注记。

〔建议级注记 3〕建议 §4.5 补一行 L0A/L0B 披露（TILE_K_SUB=128 由 gemm_mad 自动求得、2 stages × 32KB = 64KB 恰为容量上限、pass 内 assert 保护），使 Cube 侧四级预算完整可审。非阻塞。

### 3. Tiling 策略 — pass

- **Block 划分与 grid**：§5.2 block_M/N/K=128 各附量化依据（L0C 预算 / dual_copy 偶 extent / C0 整除 / 0 尾 lane / L1 预算）；grid 一维 `T.Kernel(16)`，tile_idx → (m_tile, n_tile) 核内自行一维映射（§3.3/§6.2 `//N_TILES`、`%N_TILES`）✓。机检 core_split pass（复算 16 = 声明 16）✓。
- **对齐约束**：M=N=K=512 均 16 倍数；TILE_K=128：512%128=0 且 128%16=0（fp16 C0=16）——与 CONST-gemm-dim-constraints（constants.md:72-78 亲核：C0=32B/元素、M/N 16 倍数 static_assert、K%TILE_K_SUB==0 且 TILE_K_SUB%C0==0）逐项相符 ✓。**NT 约定核对**：L1-input 路径 `trans_A=False, transpose_B=True`、权重 GM 布局 [N,K]——B 契约 (512,512) 按 [N,K] 声明（§1.3/§4.1），T.gemm 调用与 example_gemm.py:74 同形 ✓；512 全对齐无尾轴 ✓。
- **非整除处理**：契约 shape 特化（512³ 全静态全对齐、无尾块），非契约 shape wrapper 显式 ValueError 拒绝（§8.4，与 ascend-constraints.md §4 host 禁 padding 内容一致）+ 泛化路线（kernel 内 pad_value= 右 padding）已写明——静态特化 + 显式拒绝是合法处置 ✓。
- **分核三要素**（对照 core-split-strategy.md §1，三要素齐全）：① grid 划分——16 blocks，核数**真机实查**且查询代码 + 返回值记录在案（§5.5：`torch.npu.get_device_properties()` → cube_core_num=28 / vector_core_num=56 / L2 128MB，2026-10-16），单波判定 16 ≤ 28、占用 57% 的取舍有 §5.2 权衡记录（备选 128×64/64×64 的流量与尾波代价对比 + Stage 4 扫描裁决路径）✓；② AIC/AIV 分工——AIC（MTE2 copy×2 + gemm×4 + dual_copy 发射）/ 双 AIV（SimtVF 元素段各持 64×128 半 tile 实例 + UB→GM per-AIV 拷贝），**dual_copy 所有权专门说明**（L0C→UB = 单条硬件双目的指令两 AIV 协作接收、UB→GM = per-AIV 独立目的选择）——与 skill §6.2 语义一致；无 sid guard（控制流均匀，sids=2 默认）✓；③ 同步策略——auto_schedule: yes 显式声明 + 5 条数据依赖边界列举 + 调度提示位置（T.Pipelined K 维 / T.Persistent tile 维）✓。核内串行边界全静态（K_TILES=4、tiles=16、元素域 64×128）✓。grid=16 未超发物理核、无核启动风暴 ✓。
- **备选 tile 估算复核**：主选口径全部精确复现（MTE2 4MB÷16÷100B/cyc = 2621 cyc；Cube 2·128·128·512÷8192 = 2048 cyc）；128×64 的 "2301 cyc" 与本次复算 6MB÷28÷100 = 2247 cyc 有 ~2.4% 偏差，且若按波次口径（2 波 × 192KB/AIC ÷ 100 ≈ 3932 cyc）备选更劣——偏差不改变"无净收益"结论方向，且设计已将 grid/tile 扫描列入 Stage 4 裁决（§5.2/§9.2）✓。

### 4. 技术约束检测 — pass

- 一维 grid：`T.Kernel(NUM_BLOCKS)` 无 threads 参数（§3.3:250 注释显式声明）✓；多维输出网格自行一维映射 ✓（skill §10 陷阱 3 处置一致）。
- GEMM 分形对齐与 NT 约定：已在维度 3 逐项核对（512 全对齐、TILE_K=128 合规、transpose_B=True、B [N,K]、clear_accum=(kt==0) 首拍清零）✓。
- L0C 溢出：64KB ≤ 230.4KB ✓（机检 pass）。
- L0C→UB dtype 约束：acc_ub 声明 fp32 与 L0C 一致，cast 移至 VF 段——copy_op.py:147-148 ValueError 分支 + skill §6.2 "For L0C→UB dtype conversion, use a separate supported copy or cast in a VF" 亲核一致，设计恰好把约束转化为精度最优路径（cast 后置消 z 预舍入）✓。
- host 侧约束：wrapper 仅 shape 校验 + 只读视图（§3.5.1/§8.4），与 ascend-constraints.md §4（:71-73 亲核）一致 ✓；INT8 路径声明不存在（CONST-gemm-dim-constraints "INT8 MMAD 不支持" 相符）✓。

### 5. 循环与同步 — pass

- 循环结构：§6.1 四层结构明确（T.Persistent tile 维 / T.Pipelined(K_TILES, num_stages=2) K 维 / T.SimtVF+T.Parallel 元素维 / 标量索引），逐层有 API 与理由，与 example_gemm.py:69-77 同构 ✓。
- 同步与模式匹配：auto_schedule: yes，§7.2 列出 5 条依赖边界 → AutoSchedule 覆盖点映射表（MTE2→MAD multibuffer / MAD→FixPipe / dual_copy→SimtVF RAW / SimtVF→MTE3 RAW / AIC↔AIV InsertSync），无手动 flag、无 flag 预算占用（跨核 16 预算仅作上限参考，CONST-arch-a5 相符）✓；unit_flag_ctrl 注解为可选调优提示（非手动同步），与 example_gemm.py:74,76 先例一致 ✓。
- 尾块处理：§6.4 无尾块论证（契约整除）+ wrapper 拒绝路径 ✓。
- 风险提示已识别：§1.6.4 承认"dual_copy–SimtVF 元素段–dual_copy"组合无仓内直接先例（亲核属实：example_gemm.py:75-79 MIXED epilogue 为纯搬运、deepgemm epilogue.py:59-72 走 plain T.copy 链），skill §6.2 要求 L0C→UB dual_copy "control flow uniform across sub-cores"（SimtVF 均匀元素段满足该形态，但 AutoSchedule 所有权指派行为确需 Stage 3 验证）——设计给出 F1（deepgemm 式 plain copy 链 + loop-body 隔离 `for _ in range(1)` + `T.assume_no_conflict`，先例 epilogue.py:63-64 亲核为真）→ F2（MixedKernel + sid guard）两级降级 + 触发条件 + 回写路径 ✓。风险处置充分，维持建议级以下。

### 6. 验证方案 — pass

- Golden：§8.1 PyTorch 参考实现草案完整（torch on NPU：`A.to("npu")`；`A.float() @ B.float().t()` fp32 累加 + NT 语义正确〔B [N,K] → B.t() [K,N]〕；`(Z * torch.sigmoid(Z)).to(torch.float16)` 与 §1.4 cast 后置语义一致）；**独立性**：以 §1.3 用户公式（sigmoid 库原语 = 基线形态）为依据，未复刻 kernel 的直除优化式与 NPU 数据流拓扑，两式等价性由 verify_equiv.py 机器验证背书 ✓。
- L0 测试计划：契约 shape 512³（block 128 整除）、randn ×3 seeds、fp16 档 atol=rtol=1e-2、assert_close ✓；L1 语义边界覆盖值域构造（±88 exp 溢出边界两侧）、denormal、NaN/inf 传播（equal_nan=True）、bitwise 确定性、fp64 交叉核对——与 verify_equiv 角点集呼应，覆盖充分 ✓；L2 性能基线 do_bench(msprof) + 与设计期估算对账（偏差 >2× 注明失准项）✓。
- 精度标准：fp16 → 1e-2/1e-2、fp32 → 1e-3（累积密集 5e-3）——与 AGENTS.md 分档一致 ✓；误差预算论证（fp32 累加 ~1e-6 相对、silu Lipschitz ≈1.1 不放大、fp16 出口 2^-11 ≈ 4.9e-4 < 1e-2）成立 ✓。

### 7. 完整性与一致性 — pass

- 章节齐全：对照 design-template.md（非迁移 = §1–§10，§0 正确删除），10 章全在且子节完整（1.1-1.6 / 2.1-2.3 / 3.1-3.5 / 4.1-4.7 / 5.1-5.5 / 6.1-6.4 / 7.1-7.3 / 8.1-8.4 / 9.1-9.3 / 10.1-10.4）✓。
- 无占位符：对 DESIGN.md 全文扫描常见占位标记（英文待办缩写、中文"待补"类字样、花括号占位符等）计 0 命中（§10.2 "待实现（Stage 3）"为交付状态标记，非设计占位）✓。
- 同类引用真实：§3.5.3 五个参考文件 ls 全部存在；抽核行号——example_gemm.py 60/63-66/66(半 tile temp)/69/71/72-73/74/75-79 全部准确；epilogue.py 22(vcvt)/59-72(loop-body 隔离) 准确；bf16_gemm.py 61(MixedKernel sids=1)/74-82(annotate_buffer_versions)/113(store_output) 准确；rmsnorm 54-56(SimtVF+Parallel)/66(T.rsqrt) 准确 ✓。
- 内部一致：§3.3 伪代码 buffer 形状 = §4.3 表 = §1.6.3 布局决策（acc_ub/out_ub (64,128)、res (128,128)、x_l1/w_l1 (128,128)）；§6.2 循环伪代码与 §3.3 同构；UB 96KB 口径在 §1.6.0 R4 / §1.6.3 / §4.5 / §5.3 四处一致；§2.1 mixed + auto_schedule 与 §5.5③/§7.1 一致；§1.4 五步表与 §3.1/§3.2 六步映射一致（§3.1 步骤 1 合并了 A/B 两次搬运）✓。
- 常数引用披露：§9.1 声明全部 CONST-* 为 assumption + 90% 折扣 + 绝对值以 Stage 4 实测为准——符合 constants.md A5 冷启动纪律 ✓。

### 8. 算法优化分析 — pass

**§1.6 章节存在性**：1.6.0（调研）/ 1.6.1（数学等价优化）/ 1.6.2（向量化替代）/ 1.6.3（向量化轴与布局决策）四子节齐全，另有 1.6.4（结构备选与降级路径）加分项 ✓。

**算法调研存在性与完整性（四问）**：
- R1：候选表 7 行**含基线**（#0 融合 + 自然展开式）；覆盖候选库命中行——ALG-elementwise R1（除法乘倒数↔直除、cast 链合并 → 候选 1/#3 采纳项②）、ALG-softmax R1（exp→exp2·log2e → 候选 5 评估否决）、ALG-gemv R2（split-K → 候选 4 评估否决）、ALG-attention R4（mixed GEMM+Vector 流水结构 → 基线形态）、ALG-transpose R1/R4（vintlv/vdintlv 与 transpose= 路径特定性 → §1.6.3 候选 #2 否决依据逐字对应候选库原文）✓。
- R2：**明确结论"无"+ 结构依据**——GEMM K 归约已是 L0C 流式 running 累加、silu 逐元素天然单遍、唯一多遍结构（候选 3）已被融合消除；与 algorithm-research.md §3 R2 结构判据（running 统计量可在线化 / 逐元素单遍即天然在线）独立比对一致，无漏判 ✓。
- R3：**四口径齐全**（FLOPs 含/不含口径、访存 Bytes 分层〔GM/L1/L2〕、扫描遍数、中间缓冲峰值，基线在表）——机检 r3_metrics fail 为工具章节提取误报（见机械复核段，真 §1.6.0 四项正则全 True）。**独立复算全部吻合**：GEMM 2·512³ = 2.68e8 ✓；silu 5 op/elem × 262144 = 1.31e6 ✓；GM 流量基线 A+B+C = 3×512KB = 1.5MB ✓、候选 3 +Z 写读 2×512KB = 2.5MB（+67% ✓ / 融合 −40% = 1−1.5/2.5 ✓）；L1 装载 16 blocks × 256KB = 4MB（A/B 面板各 4× 重读）✓；中间缓冲零 GM 往返（融合）vs Z 512KB GM（分离）✓；L2 128MB ≫ 1.5MB 工作集判断与实查返回值一致 ✓。
- R4：逐候选六项清单（单元匹配/容量/对齐/静态边界/流水融合/跨核），容量淘汰引用 CONST-capacity-a5（90% 折扣披露）✓、带宽引用 CONST-mte-bandwidth-model（MTE2 100 B/cycle）、Cube 引用 CONST-cube-throughput-model（fp16 8192 ops/cycle/AIC）、VALU 引用 CONST-valu-model（51/13/8 档）——四条常数数值与 constants.md 原文逐项相符 ✓；候选 3 淘汰对应 BP_redundant_work_or_roundtrip（bottleneck-patterns.md:23/:218 亲核存在）。
- 调研结论：选定算法（融合 mixed + 直除式）+ 关键依据 + 调研范围声明齐全 ✓。

**调研结论正确性（独立复核）**：
- ① 等价性：直除式 a·(1/b) ≡ a/b 数学恒等，舍入路径差异经 verify_equiv 重跑背书（fp16 出口 max ulp 14、违反率 0）；溢出边界 z ≤ −88.7 退化 −0.0 两式一致（注②，角点 ±88/±89 覆盖）✓。
- ② 复杂度算术：见上，全部独立复现；**中间缓冲 GM 往返未漏计**（候选 3 行显式 +Z 写读 2×512KB，R3 小结口径自洽）✓。
- ③ 亲和性：与 ascend-constraints / constants.md 无矛盾；负向淘汰有佐证（候选 6 "epilogue 仅 scale+cast"亲核 epilogue.py:20-23 apply_alpha = vmuls+vcvt ✓；候选 3 "L0C 出口无超越函数能力"——方言 copy 路径无任何变换参数、skill §6.2 dtype 转换须另走 copy 或 VF cast ✓）✓。
- ④ **负向断言复核**：R2 "无在线变体"对照候选库命中行 + 结构判据成立（见上）；候选 5 "无独立更廉价 exp2 通道"——simd.py grep 无 vexp2（仅 vexp:860）、标量 exp/exp2 同文件相邻注册（intrin_rule_ascend.cc:47/:48）同类 lowering，断言为真 ✓；候选 4 split-K 否决依据双重成立（grid 16 < 28 并行度不缺 + silu(a+b) ≠ silu(a)+silu(b) 非线性不可施加于 partial 和，分裂即退化为候选 3 两 kernel 结构）——数学上正确 ✓；候选 2 tanh "SimdVF 域无 vtanh"——simd.py tanh 计 0，为真 ✓。未发现漏掉的已知候选。
- ⑤ 并列候选裁决路径：公式级三候选（基线/直除/tanh）未纸面拍板——**D-3 真机探针实测裁决**（design_probe_silu_form.py 存在且形态与描述一致：128×128 fp32 UB 驻留 × 64 迭代、SimtVF、do_bench(msprof)、环境+日期戳齐全）；模型预测（tanh −22%，独立复算：4×1/51+1/8 = 0.2034 vs 直除 3×1/51+1/8+1/13 = 0.2607 cyc/elem，比值 0.78 = −22% ✓）与实测（1.80×劣化；比值自洽：388.42/215.67 = 1.801、215.67/239.23 = 0.902、388.42/239.23 = 1.624 ✓）矛盾被如实披露并以实测为准，同时反推 "标量 tanh 非 SFU 档位（软件例程）" 的机理解释——**这正是"判定依赖未实证常数时不得纸面拍板"的正确执行**。0.2607 cyc/elem 精确复现（3 op×64/51 + 64/8 + 64/13 = 16.69 cyc/向量 ÷ 64 lane）✓。
- D-2 设计期估算独立复算：MTE2 4MB÷16÷100 = 2621 ✓；Cube 16.78e6÷8192 = 2048 ✓；Vector 0.2607×16384÷2 = 2136 ✓；dual_copy 65536÷256+55 = 311 ✓；max(2.6e3, 2.0e3/2.1e3, 3.1e2) 流量边际主导结论成立 ✓；assumption 声明与相对量级比较纪律符合 hardware-cost-model 协议 ✓。

**调研→设计一致性**：§1.6.0 选定（融合 mixed + 直除）→ §1.4 五步表（步骤 4 直除式）→ §3.1 步骤 4（s = z/(1+exp(−z))，显式标注"输入公式为 §1.6.1 优化后公式"）→ §6 循环结构（单遍 epilogue，无 Z 落 GM 的第二遍扫描）——全链一致，无"调研选 A 设计做 B" ✓。

**数学等价优化完整性（四要素）**：3 个优化项均四要素齐全——#1 直除（原式/优化式/等价论证〔恒等 + 三口径机器验证〕/收益〔省 1 乘 + 探针 0.902×，239.23→215.67 µs 数字自洽〕）；#2 tanh（等价论证成立但**探针否决不采纳**，否决理由量化：1.80× 劣化 + 机理推断）；#3 融合+cast 后置（结构恒等论证 + 收益量化：−1MB GM 往返 / −40% / −1 次 launch / 消 z 预舍入〔z ~10² 量级 fp16 预舍入 2^-11 相对误差进入激活值的论证成立〕；cast 时机如实标注为语义决策而非等价变换，golden 同语义）✓。候选库 ALG-elementwise R1 三项（CSE/除法乘倒数/cast 链合并）均有对应处置（CSE：直除式已无公共子表达式；乘倒数：本场景倒数即一次 div，乘倒数式反而多 1 次 mul——直除严格更优且探针证实，与候选库提示不矛盾〔该提示适用于倒数复用场景如 softmax 分母〕；cast 链合并：#3 采纳）✓。

**等价性论证独立推演**：直除恒等式成立；σ(z)=(1+tanh(z/2))/2 恒等式成立；fp16/bf16 舍入影响已评估（fp16 出口 ulp 14/24、subnormal 区绝对差 ≤2e-6、与 §8.2 容差 1e-2 一致）；E[x²]−E[x]² 类 cancellation 风险不适用（无统计量分解）✓。verify_equiv 重跑与内嵌表一致（判定列精确吻合；见建议级注记 1 的 fp32 列口径说明）✓。

**向量化替代完整性**：§1.6.2 六点盘点与 §3.3 伪代码 / §6 循环结构交叉核对**无遗漏**（伪代码中全部计算点：silu 元素段 #1、cast #2、K 循环 #3、tile_idx 映射 #4、GM 偏移 #5、host wrapper #6；T.Parallel 本身即 #1 的向量化载体、T.Persistent 为 block 级调度每 block 1 迭代）；§6.1 明确声明"无逐元素标量循环"且表格仅含 block/tile 级调度循环 ✓。不可替代项 #4/#5（block 索引/任务映射，每 block 每 tile 一次，非逐元素热点）、#6（host 元数据 + ascend-constraints §4 只读约束）理由具体到本算子、类别合规 ✓。替代 API 佐证齐全（T.SimtVF+T.Parallel → rmsnorm:54-75 含特殊函数 T.rsqrt:66 同模式；T.exp → intrin_rule_ascend.cc:47 注册；vcvt → epilogue.py:22；T.gemm/T.Pipelined → example_gemm.py:71-74）；SimtVF 优先、SimdVF 仅留 Stage 4 热点验证 ✓。

**向量化轴与布局决策完整性（§1.6.3）**：多轴算子（M/N ≥2 非 batch 维）候选矩阵 4 行，属 Cube/mixed 类别——必选项覆盖：ND2NZ 自动转换（InsertNd2Nz，禁止手写 NZ 声明在前）+ L0C fractal 直出 vs UB epilogue 归属（候选 #3）+ epilogue 单/双 AIV 归属侧（候选 #4）✓；轴质量五维评分（整除性 128%64=0/128%128=0、尾 lane 0%、累加链形态、repack 代价、UB 影响）量化齐全 ✓。**弃选论证前提亲核**：#2 否决依据 "T.copy transpose= 仅特定路径（GM→L1 dn2nz / L1→L0 请求转置 XOR）、非通用转置"——skill §6.1 SKILL.md:412-417 原文逐字核对一致，且与 ALG-transpose R4 候选库原文一致，无"核内重排误写成 GM 级 transpose"类前提错误；silu 轴对称 → 转置零收益的结构论证成立 ✓；#3 否决依据（L0C 出口仅 dtype 转换无超越函数 + dual_copy 禁 dtype 转换）——copy_op.py:147-148 ValueError + skill §6.2 "cast in a VF" 亲核为真 ✓；#4 否决依据量化（单 AIV 4271 cyc > Cube 2048 cyc，2.08× 裕度，复算 16384×0.2607 = 4271 ✓，对应 BP_dual_copy_utilization 模式——bottleneck-patterns.md:18/:137 亲核存在）✓。无"I/O 是行主序所以核内也行主序"式论证（#1 采纳理由是 dual_copy M-split 自然产出 + 0 尾 lane + 双 AIV 均衡的量化组合）✓。

**实验裁决模式**：公式级裁决已由 D-3 真机探针在设计期完成（强于三件套要求）；结构级不确定性（dual_copy–SimtVF–dual_copy 组合可行性）如实定性为**可行性风险而非成本不确定**（判断正确：F1 有完整结构描述——deepgemm 式 plain copy 链 + loop-body 隔离先例精确到行、代价估算 4271 cyc / ~1.5-2× 劣化标注 assumption；触发条件明确〔编译失败/结果错误/npusim 串行化〕；回写路径明确〔§1.6.4 + §9 风险表〕）；CV 均衡与 tile 选型的常数依赖（2136≈2048、57% 占用）裕度处置合规——不改变布局选择（N 轴/ND 为 dual_copy M-split 结构性产出），性能面留 Stage 4 扫描且 §8.3 L2 有估算对账机制（偏差 >2× 注明失准项）✓。

**布局决策与实现一致**：§1.6.3 选定（ND 行主序、N 轴向量化、每 AIV (64,128) 实例、无 repack）= §3.3 buffer 形状与 T.Parallel(TILE_M//2, TILE_N) 内层维 = §4.3/§4.5 预算（96KB 含 ×2）= §6.1 "内层向量维 = N 轴" ✓。

**与下游章节一致**：§3.1 以 §1.6.1 优化后公式为输入（步骤 4 直除式 + 步骤 5 cast 融合声明）✓；§6 无 §1.6.2 未论证的标量循环 ✓。

〔建议级注记 4〕§5.2 备选 tile 的 MTE2 估算 2301 cyc（128×64）与复算 2247 cyc（6MB÷28 AIC÷100 B/cyc）有 ~2.4% 偏差；若按波次口径（2 波串行 ≈ 3932 cyc）备选劣势更显著。建议 Stage 4 grid 扫描对账时统一口径（总量÷busy 核 vs 波次串行），并在 §5.2 权衡记录补一句口径说明。不影响"无净收益"结论与主选依据。

## 检视总评

设计文档质量高：调研四问齐全且全部关键数字（复杂度四口径、D-2 四项估算、0.2607 cyc/elem、tanh −22% 模型预测、探针三比值）经本次独立复算精确复现；负向断言（无在线变体 / 无 exp2 廉价通道 / 无 vtanh / L0C 出口无超越函数 / transpose 路径特定性 / epilogue 仅 scale+cast）逐条对照候选库、方言源码、skill 文档与 pattern-library 亲核为真；公式级选型用真机探针替代纸面拍板并如实披露模型失准；唯一结构新颖点（dual_copy 间插 SimtVF 元素段）风险定性准确且有两级可落地降级路径。4 项建议级注记（源码行号偏差 ×2、verify 表 fp32 列口径、L0A/L0B 披露、备选 tile 估算口径）均不阻塞，可随 Stage 3 顺手修正。机械复核中 design_calc_check 的 r3_metrics fail 已查证为工具章节提取缺陷（§1.4 标题含 "§1.6.0" 劫持 _section），非设计问题，已提交 RETROSPECTIVE 价值点。

结论: 通过
