# online_softmax — RETROSPECTIVE

## Stage 1 (design) Retrospective — first_design (migration-plain) — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | gate_lint 的 REF_PATH_RE 会把散文中的「examples/testing」连写误判为仓库路径并判 S1-REF-PATHS 失败（examples/testing 目录不存在） | examples/pilot/online_softmax/DESIGN.md §1.6.2 向量化结论行（初稿连写触发 1 failure，改「examples 与 testing」后 0 failure）；gate_lint.py REF_PATH_RE 定义 | 在 quality-checklist.md 或 design-template.md 的引用规范中加一条：并列引用多个仓库目录时用「与」或顿号分隔，禁止 examples/testing、docs/agents 类斜杠连写 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| softmax 源朴素 online 合并（m_c 基 lse 式）在角点发散：块间 max 间隙 > 88.7 时 fp32 exp 溢出→全 0 输出、块内全 −inf 时 NaN（fp16 数据域间隙可达 131008 远超 88.7）；flash 形式 max 基合并 + fp16 哨兵钳位可修复且良性数据恒等 | D | examples/pilot/online_softmax/verify_equiv.py V4b（源式角点 fp16 违反率 fp64=0.136 / fp32=0.182）与 V4（稳健式 0 违反、max_rel 1.32e-12）；DESIGN.md §1.6.1 O6 | python3 examples/pilot/online_softmax/verify_equiv.py | tilelang-fork（仓库工作区，fc32f47c 后）；torch CPU fp64；2026-10-16 | algorithm-candidates.md ALG-softmax 行补注（evolver 蒸馏时合入） |
| A5 上 exp 与 exp2 同为注册 intrinsic 且同 special_func 代价档——GPU softmax 的 exp2·log2e 换底是 SFU 专属优化，迁移到 A5 应去换底直用 exp(x−m)（省 1 乘/elem 并消除 scale 字面量截断误差） | P | DESIGN.md §1.6.1 O3 + §0.5（src/ascend/codegen/intrin_rule_ascend.cc L34-49、src/ascend/transform/estimate_latency.cc L1267 佐证链；verify_equiv V1 max_rel 6.16e-08） | 核对两处源码 + python3 examples/pilot/online_softmax/verify_equiv.py | tilelang-fork（仓库工作区）；2026-10-16 | algorithm-candidates.md ALG-softmax R1 候选 2 行补注 |
| 行块驻留翻转两遍流式 softmax：A5 UB 216KB〔SimtVF〕驻留整行（N=8192 fp16 行 16KB）时单遍 GM 读 + e 单次求值寄存器复用，GM 流量 −33%、special −50%；online 结构降级为 N>32768 的泛化路径而非默认选择 | P | DESIGN.md §1.6.0 R3/R4 对比表 + §0.6 R2（CONST-capacity-a5 预算推导） | 按 DESIGN.md §1.6.0 R3 口径复算任意 N 的驻留可行性 | tilelang-fork（仓库工作区）；Ascend950PR_9579 实查（cube 28 / vector 56）；2026-10-16 | pattern-library layout/elementwise.md 候选条目（evolver 蒸馏时合入） |

### Transferable Lessons（可迁移教训）

- GPU→A5 迁移 softmax/attention 族时，exp2 换底、log2 域 lse 等「换底类」优化必须先核对目标后端的超越函数代价关系（本仓 A5：exp/exp2 同档，见 intrin_rule_ascend.cc + estimate_latency.cc），不能默认 GPU 结论成立；被舍弃的源优化必须写明「意图由什么承接」（本设计 §0.5 exp2 行：意图 = 廉价 exp，承接 = 原生 exp intrinsic）。
- 「online 算法」不是无条件最优：A5 大容量 UB（216KB）使行块驻留成为比流式 online 更优的默认结构；判定依据 = 行工作集（输入 tile 双版本 + 中间量）对照 CONST-capacity-a5 的驻留预算，超限才回退 online。row-reduction 族通用（ALG-layernorm R4 的 row-reduction 形态与此一致）。
- 本仓 row-reduction 迁移的两个锚点先例：example_rmsnorm.py（fragment 掩码载入 + 原位覆写 + 行块流水）与 example_per_token_cast_to_fp8.py（T.Kernel(N_CORES) + T.Persistent + SimtVF(threads=1024) + reduce_max(dim=1) 同构），二者拼接即覆盖 softmax 的全部 API 需求，无需发明新形态。
- 设计期在真机可用时应优先实查核数（本任务 torch.npu.get_device_properties() 一手实查 Ascend950PR_9579 / cube 28 / vector 56，取代惯例初值 32 的 assumption），并把查询代码原样记入 §5.5。
- verify_equiv.py 不只验证采纳项：给被替换的源公式（朴素 online 合并）留一个「诊断对照臂」（V4b），用数字证明重设计的必要性，是 Stage 2 检视维度 8 的直接证据。

## Stage 2 (design-review) Retrospective — review — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-tooling | design_calc_check.py `check_l0c_budget` 的 CUBE_PATH_RE 会被**否定语境**的 API token 触发（设计写 "无 Cube/T.gemm"、"T.MixedKernel 不适用" 即命中），把纯 Vector 设计当 Cube 路径检查，并将行块 BM×TILE 误当 GEMM tile 报 C0 对齐 warn | 本次输出 l0c_budget warn "block 4x8192 ... not multiples of 16"，而 DESIGN.md §1.5/§3.5.1 明确无任何 alloc_l0*；工具 L199-215 注释自称 "Negations don't count" 但正则未实现否定过滤 | CUBE_PATH_RE 增加同句否定词过滤（无/不适用/不涉及），或改为要求 `alloc_l0[abc]` 正向分配语句命中才启用 l0c 检查 | R |
| Review-tooling | design_calc_check.py `check_core_split` 假设 grid = ceil(M/bm)×ceil(N/bn) 直接映射，**不识别 T.Persistent 常驻调度**（persistent 设计下 declared(核数) ≠ recomputed(tile 数) 是必然且正确的），且 `\bN\s*=\s*(\d+)` 会把举例句中的数值误解析为主 shape | 本次输出 core_split fail "ceil(8192/4) x ceil(1000/8192) = 2048 vs declared 56"：1000 来自 §5.3 举例 "如 N=1000 → 行 2000B"；设计为 T.Kernel(56)+T.Persistent([2048],56,...)（core-split-strategy §2.4 推荐形态）；工具 L259-308 无 Persistent 分支 | check_core_split 增加 persistent 分支：文本含 `T.Persistent` 时改核 "declared grid == 实查核数 && tiles 由 domain 调度" 的自洽性；M/N 解析限定主配置章节并排除 "如/例" 语境 | R |
| Review-coverage | VRF（向量寄存器文件）总容量在 pattern-library constants.md 与 tilelang-ascend skill 均无条目——维度 2 的 VRF 预算检视只能对照设计自述假设（256KB）与 per-thread 先例，无总量口径可核；本次主 shape fragment 合计恰达假设上限 100% 而无法判定是否会 spill | DESIGN.md §3.5.1/§4.5 "VRF 256KB" 无引用条目；constants.md CONST-capacity-a5 仅覆盖 UB/L1/L0A/L0B/L0C；CONST-valu-model 仅寄存器宽度 256B；grep skill SKILL.md+references 零命中 VRF 容量 | constants.md 增设 CONST-vrf-capacity-a5（assumption 起步：容量值 + per-thread/总量双口径预算规则 + Stage 3 spill 观测/npusim 反解转正方法）；design skill 内存规划节同步要求 VRF 预算标注 assumption | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 机械复核 fail/warn 项的「工具误报判定协议」：工具 docstring 自述 advisory/NOT-a-gate 时，reviewer 须 Read 工具 check 函数源码，定位误报根因（解析错误/模型假设未覆盖设计形态），在 REVIEW.md 机械复核段记录「工具输出 + 误报源码证据 + 人工复算结论」三栏后才可推翻映射（fail→维度 fail），不得直接采信也不得无证据无视 | R | 本次 l0c_budget warn 与 core_split fail 均经 Read design_calc_check.py L199-308 确认为误报（否定语境 token / Persistent 未建模 + N=1000 误解析），人工复算 ub 129KiB≤194.4KB、分核三要素自洽后 override；REVIEW.md 机械复核段留痕 | 对照 .agents/tools/design_calc_check.py 源码与 REVIEW.md 机械复核段 | tilelang-fork；2026-10-16 | tilelang-design-review SKILL.md Phase 1 第 7 步补一句判定协议 |
| 寄存器预算检视须区分 per-thread 与总量两个口径：先例的 "N reg/thread" 不等于总量佐证（rmsnorm 80 reg/thread 是 256 threads 配置、总量 ~80KB；本设计 1024 threads×64 reg = 256KB 总量达假设 VRF 上限 100%），总量口径无实测条目时应降为 warn 并要求 Stage 3 编译后首查 spill | R | DESIGN.md §3.5.1/§4.5 引 "rmsnorm 80 reg 先例" 佐证 64 reg/thread；example_rmsnorm.py L11-33 配置复算（rows=4/threads=256 → x_frag 64KB + w_frag 16KB = 80KB 总量、80 reg/thread）；REVIEW.md 维度 2 warn 注记 | 按 rmsnorm host 配置复算 fragment 总量；对照 constants.md 无 VRF 条目 | tilelang-fork；2026-10-16 | tilelang-design-review SKILL.md 维度 2 检查项补注（VRF 双口径） |
| （D 类）本次检视无新增实测常数——检视全程未跑真机测量，重跑 verify_equiv 的数字属设计侧已有 D 类 | — | none | — | — | — |

### Transferable Lessons（可迁移教训）

- advisory 工具的 fail 不能机械映射为维度 fail，也不能无证据无视：正确动作是读工具源码定位其模型假设（本次 core_split 的直映射假设 vs T.Persistent 形态），再用维度表原文判据独立裁决——本次两项误报均三栏留痕后 override，同时 ub/r3 人工复算全部吻合，未漏真错误。
- 阈值类数字是迁移检视中性价比最高的独立复算锚点：O6 溢出阈 88.7（=128/log2e）、fp16 间隙域 131008（=2×65504）、树深 13（=log2 8192）、D-2 的 24.0K/20.9K/290K cycle 全部精确复现——阈值能复算说明设计方真做了推演而非编数字。
- 「舍弃源优化」类处置（exp2 换底）的证据必须双点同查：codegen intrinsic 注册表（exp/exp2 均注册）+ latency 模型代价档（同入 special_func 列表）——单点一致不足以支撑 "同档" 结论；同时核对意图承接链（O3）防止静默丢弃。
- 设计文档的统计行/系数类笔误（"等价替换 5 项" vs 表内 4 项、"6B" vs 8B/elem）要以**表格与推导链为准**逐项复算后再定级：本次数笔误均未污染任何下游结论（处置逐项合理、驻留上限 32768 按正确口径成立），故降为建议级而非阻塞——复算先于定级。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| none | none | none | none | - |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| A5 VRF 无 spill 实证：online_softmax 主 shape [8192,8192]（BM=4/TILE=8192、threads=1024）生成的 AscendC 为 `float x_frag[32]; float e_frag[32];` + 12 个标量 fragment = 76 fp32 reg/thread，`spill` 计数 0，在 rmsnorm 80-reg 先例水位内——REVIEW 维度 2 的「VRF 100% 假设上限」warn 实际不触发 spill，x_frag 步骤③后死的寄存器复用未被编译器回收（两个 (BM,TILE) fp32 fragment 各 32 reg 并置） | D | examples/pilot/online_softmax/online_softmax.py（`_build_resident`）；`kernel.get_kernel_source()` 首 60 行实测 `float x_frag[32]; float e_frag[32]; float m_frag[4]; float s_frag[4]; float inv_frag[4]`、`__launch_bounds__(1024)`、spill=0 | `PYTHONPATH=/home/z00875089/tilelang-fork python3 -c "import examples.pilot.online_softmax.online_softmax as m; k=tilelang.compile(m._build_program(8192,8192),target='ascend',out_idx=-1); print(k.get_kernel_source()[:2000])"` | tilelang-fork（build dev root，Ascend950PR_9579，CANN 9.x）；2026-10-16 | constants.md 增设 CONST-vrf-capacity-a5（REVIEW 已提案）：本次补「主 shape 76 reg/thread 不 spill」实测样本 |
| 驻留三段式 + 流式 online 两遍双路径全绿：63 用例 max_diff ≤ 6.1e-5（fp16，容差 1e-2）；驻留路径主 shape max_diff 1.9e-6、流式路径 [128,65536] 2.4e-7、含 +inf/−inf/NaN/subnormal/深度下溢边界与 [128,32769] 尾块（rem=1）全 PASS——O6 flash 合并 + 哨兵钳位 + 尾块静态处理的设计落地正确 | D | examples/pilot/online_softmax/online_softmax.py `--level all` 全绿输出（L0 2 + L1 44 + L2 4 + Boundary 13 = 63 用例） | `PYTHONPATH=/home/z00875089/tilelang-fork python3 examples/pilot/online_softmax/online_softmax.py --level all` | tilelang-fork（Ascend950PR_9579，CANN 9.x）；2026-10-16 | cases.md 增 online_softmax 案例索引（evolver 蒸馏时合入） |
| 流式分块 kernel 的尾块 DMA 用「静态 rem = N % BLOCK_N + `if rem > 0` 元编程分支 + 缩窄静态 extent（`x_ub[0:BM, 0:rem]`）」处理，不用动态 `T.min` extent——GM↔UB 的 `T.copy` 区域 extent 须为编译期常量（动态 extent 会走非预期 DMA 形态），元编程 `if <python-int/bool>` 在 `@T.prim_func` 体内可用（rmsnorm `if reuse_input:` 先例） | P | examples/pilot/online_softmax/online_softmax.py `_build_streaming`（num_full/rem 推导 + `if rem > 0:` 尾块分支 + `T.copy(X[base:base+BM, col:N], x_ub[0:BM, 0:rem])`）；[128,32769]（rem=1）PASS 实证 | 同上 `--level L2`（覆盖 [128,32769] 尾块） | tilelang-fork（Ascend950PR_9579，CANN 9.x）；2026-10-16 | tilelang-ascend SKILL.md §6（数据搬运）补尾块静态 extent 处理范式 |
| flash 合并哨兵钳位须在 exp 之前：`m_c = max(reduce_max(x), −65504)` 后再 `e = exp(x − m_c)`（全 −inf chunk → e=exp(−inf)=0 非 NaN），若先 exp 再钳位则 `exp(−inf−(−inf))=NaN` 污染 s_run → 全行 NaN 与 torch 全 −inf 行语义仍一致但中间路径多一次 NaN 传播 | P | examples/pilot/online_softmax/online_softmax.py `_build_streaming`（`m_c[r] = T.max(m_c[r], sentinel)` 位于 reduce_max 与 exp 之间）；Boundary-all-neg-inf / Boundary-nan PASS | `--level boundary` | tilelang-fork（Ascend950PR_9579，CANN 9.x）；2026-10-16 | algorithm-candidates.md ALG-softmax 行补注（与 DESIGN §1.6.1 O6 呼应） |
| `T.infinity(T.float32)` 接受 dtype 对象（内部字符串化）、`T.serial(n)` 单参即 `range(n)`、`T.reduce_max/reduce_sum(dim=1, clear=True)` 在 SimtVF 内对 (BM,TILE) fragment 降级为 AscendAllReduce 跨线程树归约（源码可见 `tl::AscendAllReduce<tl::MaxOp,1024,1,0>`）——三者均为 ascend 方言常用形态的实证确认 | D | examples/pilot/online_softmax/online_softmax.py（`-T.infinity(T.float32)` / `T.serial(num_full)` / `T.reduce_*(...,dim=1,clear=True)`）；`get_kernel_source` 含 `AscendAllReduce<tl::MaxOp>` 与 `AscendAllReduce<tl::SumOp>` | 同上 get_kernel_source 命令 | tilelang-fork（Ascend950PR_9579，CANN 9.x）；2026-10-16 | tilelang-ascend SKILL.md §10 陷阱/§5 归约补实证（evolver 蒸馏时合入） |

### Transferable Lessons（可迁移教训）

- 流式/分块 kernel 的循环尾块（N % BLOCK_N ≠ 0）不要用动态 extent 的 `T.copy`：把 `rem = N - (N // BLOCK_N) * BLOCK_N` 在 host 侧算成 Python 常量，用 `if rem > 0:`（元编程分支，parse 期决定）单独处理尾块，copy 两侧都写显式静态 region（`X[base:base+BM, col:N]` ↔ `x_ub[0:BM, 0:rem]`）——GM↔UB DMA 的 region extent 必须是编译期常量，否则落到非预期搬运路径。
- row-reduction 族的驻留式三段（reduce_max → exp(x−m) → reduce_sum → 每行倒数 → 原位 cast(e·inv)）在 SimtVF 内直接映射即可，无需 `alloc_reducer`/`reducer_update` 那套显式 reducer API——`T.reduce_max/reduce_sum(dim=1, clear=True)` 对 (BM,TILE) fragment 自动降为 AscendAllReduce 跨线程归约，fp32 全程硬件直连。
- 掩码载入（pad 用 −inf 做 max 幺元、exp(−inf−m)=0 做 sum 幺元）+ 掩码存储（`if i < N:` 谓词 store）组合可同时吸收非 pow2 N 与尾块，且与 DMA 紧凑拷贝（x_ub 存 (BM,N) 紧凑、fragment 存 (BM,TILE) pow2）正交——先例 rmsnorm L42/L56-57 已验证，迁移时直接复用该形态，无需自造 padding。
- 全 −inf / +inf / NaN 输入行的 softmax 语义「逐元素 NaN 传播」在驻留式与流式两路径天然一致（驻留：m=−inf → exp(nan)；流式：m 钳 −65504 → s_run=0 → lse=−inf → exp(x−(−inf))=nan），golden 用 `torch.softmax(dim=1)` + `assert_close(equal_nan=True)` 对照即可全覆盖，无需逐角点手写期望。

## Stage 3 (develop) Retrospective — retry_impl (attempt=2, 门禁窄修复) — 2026-10-16

### Gate Fix Record（规范入口 vs compile 形态的门禁教训）

- 触发：statectl gate 3 机械检查 `gate_lint.py` 规则 `S3-JIT` 判定「未找到 @tilelang.jit 装饰的 kernel 函数」。根因 = 首次实现用了 `tilelang.compile(_build_program(M, N), target="ascend", out_idx=-1)` 作为编译入口（DESIGN.md 的 decision-tree 把 `tilelang.compile(...)` 列为合法别名形态），但 `gate_lint.py` 的 `kernel_lint`（`_jit_decorators`）只做 AST 扫描 `@tilelang.jit`（含 `from tilelang import jit` 的 `@jit`）装饰器，不认 `tilelang.compile` 形态——规范入口与门禁识别器不一致，必须视为「门禁只认规范入口」。
- 修复（窄、不改算法/golden）：把 `@functools.lru_cache` + `_get_kernel` + `tilelang.compile(...)` 三件替换为规范的参数化工厂函数 `@tilelang.jit(target="ascend", out_idx=[1])` 装饰的 `online_softmax_kernel(M, N)`（lazy mode：返回 `_build_program(M, N)` 的 PrimFunc），`online_softmax` 改为 `online_softmax_kernel(M, N)(X)`。`_build_program`/`_build_resident`/`_build_streaming` 双路径逻辑零改动；`out_idx` 由 `-1` 显式化为 `[1]`（Y 为第 2 个参数，与参考 `examples/pilot/silu/silu.py` 的 `out_idx=[1]` 一致）。`functools` import 一并删除（`@tilelang.jit` 自带 `_kernel_cache`）。
- 验证：`gate_lint.kernel_lint(...)` 0 failure / 0 warning；`--level all` 真实重跑 63 用例全 PASS（L0 2 + L1 44 + L2 4 + Boundary 13，含 streaming 与尾块），无 WARN/FAIL。
- 教训：Stage 3 落盘前应以 `gate_lint.kernel_lint` 做本地自检（而非只看 DESIGN decision-tree 的「合法形态」清单）；`tilelang.compile` 虽被 decision-tree 列为别名，但 `S3-JIT` 机械规则只认 `@tilelang.jit`，二者冲突时以**门禁规则为准**——规范入口唯一。参数化 shape 工厂在 `@tilelang.jit` 下用 lazy mode（工厂返回 PrimFunc）+ `out_idx=[...]` 即可保留分层测试所需的各 shape 参数化，`_is_lazy_style` 通过「调用工厂、判断返回值是 PrimFunc」的 fallback 路径正确识别（不要求被装饰函数体内直接书写 `@T.prim_func`）。

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `@tilelang.jit` 懒模式工厂可透明委托给外部 PrimFunc 构建函数：被装饰函数体不必直接内嵌 `@T.prim_func`，`return _build_program(M, N)` 即可——`has_internal_prim_func` 为 False 时会走「调用工厂并判断返回值为 PrimFunc」的 fallback，模式推断仍正确判为 lazy；参数化 shape 工厂 + `out_idx=[1]` 保留编译期缓存（`_kernel_cache` 以 (M,N) 为 key） | P | examples/pilot/online_softmax/online_softmax.py L215-227；tilelang/language/eager/builder.py `_is_lazy_style` L1621-1659；tilelang/jit/__init__.py `JITImpl.__call__` L521-566 | `PYTHONPATH=/home/z00875089/tilelang-fork python3 examples/pilot/online_softmax/online_softmax.py --level all`（63 PASS） | tilelang-fork（Ascend950PR_9579，CANN 9.x）；2026-10-16 | tilelang-op-develop SKILL.md 第 37 行「tilelang.compile 亦是合法形态」改为「门禁 S3-JIT 只认 @tilelang.jit，规范入口唯一；tilelang.compile 供内部/调试用，不作为落盘入口」 |
| 门禁规则与设计决策树的「合法形态清单」冲突时以门禁为准：S3-JIT 只认 AST 层的 `@tilelang.jit`/`@jit` 装饰器，不认 `tilelang.compile` 调用（decision-tree 将后者列为别名即埋下 first_impl 落盘后 gate 3 机械失败） | R | gate_lint.py `kernel_lint`/`_jit_decorators` L520-651（仅 AST 扫装饰器）；design-reference/decision-tree.md L20 列 compile 为别名 | 复跑 `python3 .agents/tools/gate_lint.py` + `gate_lint.kernel_lint(...)` 前后对比 | tilelang-fork；2026-10-16 | tilelang-op-design references/decision-tree.md L20 与 tilelang-op-develop SKILL.md L37 同步收紧为「规范入口唯一=@tilelang.jit」 |

### Transferable Lessons（可迁移教训）

- 首次实现交付前必须用 `gate_lint.kernel_lint(op.py, op_name)` 做本地自检——`S3-JIT` 的 AST 扫描只认 `@tilelang.jit(...)` 装饰器，`tilelang.compile(...)` 再等价也过不了门禁；开发者不应在「编译等价」与「门禁可识别」之间假设二者一致。
