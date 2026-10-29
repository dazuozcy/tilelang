# RETROSPECTIVE.md — _logsumexp_kernel_tiled（migration-harness）

## Stage 1 (design) Retrospective — first_design（调度重试，前次中断未产出工件）— 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | tiled 变体的契约空间三入口（大 N / 驻留超限 / 显式 config）与 tile_n 启发式（256 倍数 + 整除偏好 + effective_budget ∝ block_m → 落点偏好 block_m=1）须从 wrapper `default_config`/`_tile_n_for_block_m`/`compute_tile_n` 三方复算才能定界，且**wrapper 产生的 tile_n 不保证 2 的幂**（wl3=25600、测试 (33,33000)=16640）——这一事实直接驱动本设计核心重设计 R2（BLOCK_N pow2 重划），调度 prompt 的「tile_n 启发式」提示未点明非 pow2 风险 | `logsumexp.py` L176–233；`_primitives.py` L116–203；DESIGN.md §0.2/§0.6 R2 | conductor 调度 prompt 模板对带 tile 参数的迁移任务补一句「wrapper 产生的 tile_n 可能非 2 的幂，Stage 1 须评估 fragment pow2 约束下的 chunk 粒度重划」 | R |
| Tooling | 前一次调度异常中断未产出任何工件——本任务为重试；无中间产物可复用，全部工件（verify_equiv.py / DESIGN.md）本次重建 | `.stage_state.json` retry_count=0 但无 DESIGN.md；调度 prompt「前一次调度异常中断」 | none（流程性事件，非 skill 缺陷） | - |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| flash 形式 online 合并（max 基重缩放）对 chunk 划分满足结合律：tile_n 任意 256 倍数划分（源 GPU）与 BLOCK_N pow2 重划分（A5）在 7 组 shape×精度组合上 fp64/fp32 位级一致（ulp=0），划分敏感性扫描（BLOCK_N 256…16384 × tile_n 16640/25000/32512）≤1 ulp——「BLOCK_N pow2 重划替代 tile_n 直译」的等价性证据（非 pow2 tile_n 直译将使 fragment next_pow2 翻倍、有效 lane 50.8%–78.1%，special 主导项同比例劣化） | D | `examples/logsumexp/_logsumexp_kernel_tiled/verify_equiv.py` V1/V6（全组合 bitwise/≤1 ulp）；DESIGN.md §1.6.1 O2 | `PYTHONPATH=/home/z00875089/tilelang-fork python3 examples/logsumexp/_logsumexp_kernel_tiled/verify_equiv.py` | tilelang-fork 工作区 + torch CPU，2026-10-16 | pattern-library/elementwise.md（D-REDUCE-SIMTVF-API 条目补注：online 合并链的 chunk 划分不变性） |
| logsumexp online tiled 链（源式 flash 合并）的角点全谱：全 −inf 行 → NaN（源固有，与 single 一致）；部分 −inf（含整 chunk 全 −inf 两种顺序）完全中性；fp16 跨 chunk max-gap 131008 无溢出（flash 形式 exp 参数 ≤ 0——与 TRAP-NAIVE-ONLINE-MERGE 朴素式 88.7 溢出界对照）；主导值跨 chunk 位级正确——**哨兵钳位（pilot O6 式）在 logsumexp 族应弃用**（唯一作用是改变全 −inf 行为 = 语义变更，且 flash 式角点本已安全） | D | `verify_equiv.py` V5（8 corner 行）；DESIGN.md §1.6.0 R1 #11 / §9.3 | 同上（V5 段） | 同上 | pattern-library/traps-runtime.md（TRAP-NAIVE-ONLINE-MERGE 条目补注：logsumexp 变体的钳位弃选判据） |
| logsumexp tiled fp16-e 备选（chunk 内 fp16 exp/sum + 合并链保 fp32 rescale）模拟精度 max_rel 3.6e-5（1024×32768）/ 1.4e-5（4×102400），@1e-2 违反率 0——合并域保 fp32 是 tiled 备选与 single 备选的关键差异点（跨 chunk 重缩放链次数多，s_c 可 fp16 但 s_run 合并必须 fp32） | D | `verify_equiv.py` V7；DESIGN.md §1.6.4 | 同上（V7 段） | 同上 | pattern-library/elementwise.md |
| 「wrapper 契约参数含非 pow2 tile 值」的 row-reduction 流式 kernel 设计模式：BLOCK_N = 满足以下约束的最大 2 的幂（BLOCK_N ≤ tile_n 且 2·BLOCK_N·elem ≤ UB 90% 折扣预算）+ SUB 联合封顶 + 尾 chunk 静态 rem 切片（TILE_R = next_pow2(rem) masked）——tile 参数降级为预算上限（签名不变），任意显式 tile_n 由 clamp 兜底 | P | DESIGN.md §0.6 R2 / §5.2 | none（设计模式，Stage 3 实现后转实证） | tilelang-fork + Ascend950PR_9579 / CANN 9.2.0，2026-10-16 | pattern-library/layout.md 或 elementwise.md（流式 row-reduction 族模板） |
| design_lint（gate 1）的 TEMPLATE_BRACE_RE 只匹配花括号内含中文的模式（`r"\{[^{}\n]*[\u4e00-\u9fff][^{}\n]*\}"`）——Python dict 字面量（如 `T.annotate_buffer_versions({x_ub: 2})`）不触发 S1-PLACEHOLDER；调度 prompt 的「集合记号 {X:Y} 误判」提示精确指向**中文标签集合构建记号**，纯 ASCII 代码字面量可安全使用 | D | `.agents/tools/gate_lint.py` L37/L210–217 亲读；本任务 DESIGN.md 含 `{x_ub: 2, stats_ub: 1}` 字面量通过 gate（0 fail 0 warn） | `PYTHONPATH=<repo> python3 -c "import sys; sys.path.insert(0,'.agents/tools'); from gate_lint import design_lint; print(design_lint('examples/logsumexp/_logsumexp_kernel_tiled/DESIGN.md', True, '.'))"` | tilelang-fork，2026-10-16 | 调度 prompt 门禁规避提示语修正（ conductor 模板）或 tilelang-design-review skill 机械复核段 |

### Transferable Lessons（可迁移教训）

- **tile_n 类后端相关参数的适配方向**：签名保留、语义降级为预算上限（BLOCK_N pow2 ≤ tile_n）——与 K3 移除 threads 同类；迁移 prompt「不要改变接口的参数(除非该参数跟后端实现相关)」的适用判据 = 参数值是否由源硬件约束（SMEM 256 对齐）驱动且目标硬件无该约束（A5 fragment pow2）。
- **online 流式 kernel 的 running stats 驻留**：A5 SimtVF 域 per-chunk 开闭（域间有 MTE2）——stats 必须驻留 UB（`stats_ub (SUB,2)` + 域内读写），不能照抄源 GPU 的跨循环 register fragment；online_softmax streaming L102–111 是该结构的唯一先例形态。
- **chunk 串行循环的保留理由**要写「循环携带依赖 + 行块并行度已饱和」双论证：仅写顺序依赖会被追问「为何不 partial+merge」——行块任务数（M/block_m）≥ 核数时 chunk 维并行化零收益且添 partial 往返（R1 #10 弃选逻辑）。
- **五层嵌套（Persistent + TIR if + 子批 serial + chunk serial + SimtVF）的风险消解法**：论证主流落点的退化形态（block_m=1 → num_sub=1 → 与 measured 先例同构）+ 拆分双先例拼接覆盖 + fallback 预案（循环层次互换 + stats 扩容）三件套，比笼统标风险更有检视价值。
- **契约落点复算必须逐 block_m 展开 wrapper 启发式**（effective_budget ∝ block_m 使小 bm 得大 tile_n——tiled 落点偏好 bm=1 与 single 优先最大 bm 相反）；只看 dispatch 条件不看 default_config 遍历会算错落点。
- 同族先例的 Stage 3 RETROSPECTIVE「本任务后续函数复用同一 per-chunk 计算体」直接构成 tiled 设计的组件复用清单——迁移任务设计前先读同族前序任务的复盘，可省大量重复论证。
- /tmp/opencode 不可写时，超长文档分段落盘可改用算子目录内 `history_version/` 临时段文件再 cat 合并（本任务 5 段 1003 行实操验证）。

## Stage 2 (review) Retrospective — first_review — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | design_calc_check.py 4 项检查中 3 项为解析伪影且需亲读解析器源码才能消解：`check_ub_budget`（L108 起）把 §4.5 节内全部表行**跨 workload 求和入单桶**（17 entries → 576.2KB 假 fail）；`check_core_split`（L259 起）正则抓示例数字（M=1,bm=1,N=1,bn=4096）反推 grid=1（persistent 形态 grid=56 与调度迭代数 num_tiles 不区分 → 假 fail）；`l0c_budget` 对纯 Vector 算子文本误配「block 4x4」→ 假 warn。**single 变体检视已确认同类伪影，本次为第二复现（recurring）** | REVIEW.md 机械复核段；`.agents/tools/design_calc_check.py` L108/L259 亲读；single REVIEW.md 同段判例 | design_calc_check.py：ub_budget 按表行归属分桶（workload 分组解析）、core_split 识别 persistent 形态（grid 声明 vs 调度迭代数）、l0c_budget 对无 Cube 算子直接 skip；或 tilelang-design-review skill 机械复核段维护「已知伪影形态」清单 | R |
| Info-source | SimtVF 行归约在 fragment 元素数 < threads（复制布局）下的行为无任何文档/measured 先例覆盖：D-REDUCE-SIMTVF-API 条目与 online_softmax/single 两先例均为 extent ≥ threads（尾 chunk 保持全尺寸 fragment），检视只能自建探针真机实测 + 亲读 C++（reduce.h ReplicateExtent/CheckAllReduceWidth）闭合 | REVIEW.md 机械复核段「检视补充实证」；`review_probe_tail_reduce.py`（本目录）；`src/backend/common/op/reduce.h` L92–101/L188–213/L396–409 | pattern-library/elementwise.md D-REDUCE-SIMTVF-API 条目补 replication 配置注记（extent<threads 合法条件：extent/scale/width 均 pow2） | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| SimtVF 行归约 extent<threads（复制布局）实测正确：(1,256) fragment @ SimtVF(threads=1024)（replication=4）的 reduce_max/reduce_sum(dim=1, clear=True) 编译运行通过，sum ratio=1.000000（复制 lane 无双重计数）、lse 与 fp64 参考一致；机理 = reduce.h ReplicateExtent 建模 + CheckAllReduceWidth 要求 scale/logical_width 均 pow2（extent=256、scale=4 满足）+ AllReduceNeedsWorkspace(>32)→ub_reduce 按 scale-stride 折叠复制槽位 | D | REVIEW.md 机械复核段；`examples/logsumexp/_logsumexp_kernel_tiled/review_probe_tail_reduce.py`；reduce.h/reduce.cc 行号见左 | `PYTHONPATH=/home/z00875089/tilelang-fork python3 examples/logsumexp/_logsumexp_kernel_tiled/review_probe_tail_reduce.py` | tilelang-fork 工作区 + Ascend950PR_9579 / CANN 9.2.0，2026-10-16 | pattern-library/elementwise.md（D-REDUCE-SIMTVF-API 补注） |
| 迁移检视中 wrapper/host 启发式派生数值（tile_n 落点、block 划分）必须用独立复刻脚本逐 shape 复算——读源码不重算算术会漏掉级联错误：本次 §0.2 一处 align_up 算术误（33000→33280 而非 33024）+ 规则描述省略整除接受条件（div_tiles≤max_tiles），级联至设计文档 5+ 处归因/测试引用；复算脚本（复刻 align_up/compute_tile_n/default_config）一次运行即可全量核对落点 | R | REVIEW.md 维度 0 warn 项与建议 1；DESIGN.md L51/L66 vs `_primitives.py` L116–203 | 检视侧动作规范，无独立脚本工件（复刻逻辑 ~30 行，见 REVIEW 维度 0 证据链） | tilelang-fork，2026-10-16 | tilelang-design-review SKILL.md 维度 0 核对要点（补「host 启发式数值逐 shape 复算」）；对称补 tilelang-op-design skill | R |
| 「检视探针实测」是闭合先例空隙的标准动作：设计引用的先例支持某模式（masked 载入）但先例配置类与设计新配置不同（全尺寸 fragment vs 缩小尾 fragment）时，写 ~70 行最小 kernel 精确复刻设计配置真机跑一遍（编译 ~10s），把不确定风险点转为 measured 证据，避免「按先例推断」或「保守判阻塞」两个极端 | P | REVIEW.md 机械复核段「检视补充实证」+ 建议 5 | 方法论（触发判据：设计配置 ∉ 先例配置类且无 pattern-library 覆盖） | tilelang-fork + Ascend950PR_9579，2026-10-16 | tilelang-design-review SKILL.md（Phase 1 补可选探针步骤） | P |

### Transferable Lessons（可迁移教训）

- **SimtVF 行归约允许 fragment 元素数 < threads**：extent 与 threads 均为 2 的幂时复制布局（scale = threads/extent）被 AscendAllReduce 正确处理（SUM 不重复计数复制 lane、MAX 幂等）——实测 (1,256)@1024 通过；无需为小尾 fragment 收缩 threads 或 pad extent（threads=min(1024, SUB·BLOCK_N) 类全 chunk 公式可直接复用于尾 chunk）。
- **design_calc_check.py 三类已知伪影形态**：多 workload 预算表被节内求和入单桶（ub_budget）、persistent grid 的示例数字被反推为 grid（core_split）、纯 Vector 算子被文本误配 L0C（l0c_budget）——消费机检结果前先按 workload 分桶人工复算并与解析器源码核对，伪影不判维度 fail，但消解依据必须写入 REVIEW 机械复核段。
- **启发式派生数值的错误是级联性的**：host 侧一条算术（align_up）+ 一条规则省略（整除接受条件）可级联到调研归因、复杂度对比、测试计划多处引用；检视复算时应以「复刻脚本输出 vs 设计声明值」逐 shape 对表，而非抽验单点。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | `@T.macro` 定义在 `@T.prim_func` 内部会 `NameError`（AST 解析器不内联非 macro 的嵌套 def）；macro 必须定义在 builder 函数层（prim_func 外层），编译期常量走闭包捕获、buffer 走参数传入——本任务用 macro 干净地消除了子批体的 4× 重复（full/partial × 全块/尾块），若不知此约束会退化为 240 行内联或误用 helper | `_logsumexp_kernel_tiled.py` sub_batch macro；首次尝试（macro 定义在 main 内）报 `NameError: name 'sub_batch' is not defined` | tilelang-ascend skill §2/§6 或 tilelang-op-develop skill 补「`@T.macro` 须定义在 prim_func 外层、buffer 作参数、编译期常量闭包捕获」注记 | P |
| Develop-flow | `@T.macro` 体内的 Python `if`（编译期常量条件，如 `if num_full > 0` / `if rows == SUB`）与 `T.serial`/`T.SimtVF`/`T.And` 组合在 A5 端到端编译运行通过——macro 并非「仅 SimdVF」专用（flash_attention 先例全是 SimdVF），SimtVF + T.serial chunk 循环 + Python 静态分支均可安全内联 | `_logsumexp_kernel_tiled.py` sub_batch macro 全量路径；L0 ⑤⑥⑦ 全 PASS | tilelang-ascend skill §2 macro 用法补 SimtVF/T.serial 兼容性实证 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| logsumexp tiled 流式 kernel（BLOCK_N pow2 chunk + flash 合并 + stats_ub 驻留）对 randn 域 fp16/bf16 输出与 `torch.logsumexp(x.float()).to(dtype)` **位级一致（max_diff=0.0）**，fp32 输出 max_diff ≈ 9.5e-7（树形分组差，与 verify_equiv V3 的 ≤1 ulp/max_rel 6.5e-8 同量级）——源式 m_run 基 flash 合并（prev=m_run; m_run=max(m_run,m_c); e=exp(x−m_run) 原位; s_run=s_run·exp(prev−m_run)+s_c）在测试域与 torch 位级重合，迁移保真达成 | D | `_logsumexp_kernel_tiled.py` L0/L1/L2 全部 PASS 输出（fp16/bf16 max_diff=0.000e+00；fp32 9.537e-07）；verify_equiv.py V1/V3/V4 | `PYTHONPATH=/home/z00875089/tilelang-fork python examples/logsumexp/_logsumexp_kernel_tiled/_logsumexp_kernel_tiled.py --level all` | tilelang 0.1.15+cuda.git40c9d668 + Ascend950PR_9579 / CANN 9.2.0，2026-10-16 | pattern-library/elementwise.md |
| tiled logsumexp 五层嵌套（Persistent + TIR if + 子批 T.serial + chunk T.serial + SimtVF）+ `annotate_buffer_versions({x_ub:2, stats_ub:1})` 在 A5 端到端编译运行通过（L0 ⑤ (33,33000) 尾 chunk extent<threads、⑥ (33,32768,bm=4) M 尾块 TIR if + 子批、⑦ fp32 128KB UB 全触发）——消除 DESIGN 风险 3 的「无单一先例」不确定性，主流落点 bm=1 子批层退化不影响显式 bm>1 深层嵌套的正确性 | P | `_logsumexp_kernel_tiled.py` main() 结构；L0 7 用例 + L1 33 用例 + L2 5 用例 + Boundary 17 用例全 PASS | 同上 `--level all` | 同上 | pattern-library/layout.md 或 elementwise.md（row-reduction 流式族模板） |
| 集成形态尾 chunk reduce extent<threads（(33,33000) 落点：SUB=1, BLOCK_N=16384, rem=232, TILE_R=256, threads=1024 → 256 元素 fragment @ 1024 threads，replication=4）在完整 kernel 内正确——reviewer 探针（review_probe_tail_reduce.py，独立最小 kernel）的 PROBE_PASS 在集成形态 L0 首用例闭合，sum 无复制 lane 双重计数 | D | `_logsumexp_kernel_tiled.py` L0 首用例 (33,33000) max_diff=0.0；review_probe_tail_reduce.py | 同上 `--level L0` | 同上 | pattern-library/elementwise.md（D-REDUCE-SIMTVF-API 补注确认） |
| tiled logsumexp 无寄存器 spill：生成 AscendC `float x_frag[16]`（SUB=1×BLOCK_N=16384/1024 threads = 16 regs/thread）+ m_c/s_c/prev_frag/out_frag 标量，远低于 CONST-vrf-capacity-a5 实测 76 reg 水位——BLOCK_N pow2 重划 + SUB 联合封顶（SUB·BLOCK_N ≤ 32768）把 VRF 从贴顶（128KB）压到舒适水位 | D | `_logsumexp_kernel_tiled.py --show-source`（`float x_frag[16]`）；DESIGN §4.5 VRF 预算表 | `PYTHONPATH=<repo> python examples/logsumexp/_logsumexp_kernel_tiled/_logsumexp_kernel_tiled.py --show-source \| grep 'float x_frag'` | 同上 | pattern-library/constants.md（CONST-vrf-capacity-a5 补样本） |

### Transferable Lessons（可迁移教训）

- **`@T.macro` 必须定义在 `@T.prim_func` 外层**（builder 函数体内、prim_func 之前），编译期常量（threads/num_full/rem/TILE_R/SUB/BLOCK_N/dtype）走闭包捕获、运行期 buffer（x_ub/stats_ub/lse_ub/x/y）走 macro 参数传入、行基址（TIR 表达式）也可作参数；定义在 prim_func 内部会 `NameError`。macro 体内可用 `T.serial`/`T.SimtVF`/`T.Parallel`/`T.reduce_*`/`T.And` 与 Python `if`（编译期常量条件）。
- **flash 合并（源式 m_run 基）在 logsumexp 族对 randn 域与 torch.logsumexp 位级一致（max_diff=0.0）**：`prev=m_run; m_run=max(m_run, m_c); e=exp(x−m_run) 原位; s_run=s_run·exp(prev−m_run)+s_c`——迁移 logsumexp 类算子时 golden 可直接用 `torch.logsumexp(x.float()).to(dtype)`，无需 fp64 参照（fp16/bf16 位级、fp32 ≤1 ulp 树形差）。
- **full chunk 无谓词直载 + 尾 chunk masked 二分支**比源 GPU 的 `_needs_mask` 快慢双路径更简洁且省 select：full chunk（全列有效）直接 `T.cast(x_ub[r,i])`，尾 chunk `T.if_then_else(T.And(r<rows, i<rem), cast, -inf)`——`-inf` 同时是 max/exp/sum 三幂元，单一 masked 载入消化行尾+列尾。
- 被丢弃行（partial 子批 r≥rows）的垃圾值（含 NaN）**不污染有效行**——reduce_* 沿 dim=1 逐行独立、合并沿行维逐行独立，只要写出切片 `lse_ub[0:rows] → y[slice]` 静态截断即可；partial 子批仍加行谓词 `r<rows` 是对齐 single 变体惯例，非严格必需。
