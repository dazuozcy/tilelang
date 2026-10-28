# RETROSPECTIVE.md — _logsumexp_kernel_single（migration-harness）

## Stage 1 (design) Retrospective — first_design — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | wrapper 契约 shape 空间（single 变体何时被 dispatch 选中）未在任何 conductor 预注入信息中给出，需设计者自行追溯 `logsumexp.py` `_tile_n_for_block_m` + `_primitives.py`（MAX_SINGLE_TILE_COLS=32512、compute_tile_n num_buffers=1、device_smem_budget=216KB）三方才能定界；同类 harness 迁移任务的调度 prompt 宜提示「先读 wrapper dispatch 逻辑定契约空间」 | `examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp.py` L176–198；DESIGN.md §0.2 | add-ascend-op skill / conductor 调度 prompt 模板补一句「single/tiled 变体分界由 wrapper dispatch 决定，Stage 1 须先读 dispatch 函数」 | R |
| Constraint-doc | A5 方言 `T.copy` 的行方向 OOB（M 尾块越界切片）行为未文档化；skill §6.1 只覆盖列向 `pad_value=`/`data_select=` 行尾 padding——设计只能假设越界不安全并规避（静态切片 + 行谓词），该假设本身值得实测转正 | `.agents/skills/tilelang-ascend/SKILL.md` L429–461（仅列向）；DESIGN.md §0.6 R4 | tilelang-ascend skill §6.1 补一行「行向 OOB 行为未验证，尾块处理用静态切片/谓词规避」或在 testing/ascend 补行向 OOB 数值用例 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| logsumexp 减 max 稳定化链的 −inf pad 幂元中性在任意 pad 宽度下位级成立（N_padded 对齐 pad 与 pow2 TILE pad 两种布局 fp64/fp32 全部位级一致，ulp=0）——「紧凑 UB + pow2 fragment masked」替代「GPU 256 元素对齐 N_padded 驻留」的等价性证据 | D | `examples/logsumexp/_logsumexp_kernel_single/verify_equiv.py` V1/V4（全 shape bitwise=True） | `PYTHONPATH=/home/z00875089/tilelang-fork python3 examples/logsumexp/_logsumexp_kernel_single/verify_equiv.py` | tilelang-fork 工作区（fc32f47c 后）+ torch CPU，2026-10-16 | pattern-library/elementwise.md（D-REDUCE-SIMTVF-API 条目补注） |
| logsumexp 源算法（m+log(Σexp(x−m)) 链）在全 −inf 行 / 含 +inf 行输出 NaN，与 `torch.logsumexp`（−inf / +inf）固有分歧——randn 测试域不触发；迁移保真应保持源语义并在 Boundary 用源语义模拟对照 | D | `verify_equiv.py` V3b（chain=NaN vs torch=−inf/+inf；V4 链与源位级一致） | 同上（V3b 段） | 同上 | pattern-library/traps-runtime.md（候选新 TRAP 条目） |
| fp16-e 精简链（arg fp16 + fp16 树形和 + fp32 log）模拟精度 max_rel ≈ 4.2e-5，远优于树深最坏估计（~6e-3）——fp16 输入限定的 logsumexp/softmax 备选变体精度守卫余量 ~100× | D | `verify_equiv.py` V5（1024×4096: 4.211e-05；128×16384: 3.589e-05） | 同上（V5 段） | 同上 | pattern-library/elementwise.md |
| block_m 为 wrapper 契约参数（不可整除 M、不可内部改）的 row-reduction kernel：行子批 SUB（min(block_m, UB 双版本约束, VRF 约束) 最大 2 幂）+ TIR if 尾块分支（full/rem 静态分解）双机制可同时消化「任意 block_m × 任意 M」的容量与边界问题 | P | DESIGN.md §5.2/§5.4/§0.6 R2/R4 | none（设计模式，Stage 3 实现后转实证） | tilelang-fork + Ascend950PR_9579 / CANN 9.2.0，2026-10-16 | pattern-library/layout.md 或 elementwise.md |
| 本机 wrapper 侧 single/tiled 分界实测口径：N_padded ≤ 32512 ∧ block_m·elem·N_padded ≤ 216KB（tileops.device NPUBackend 返回 UB_SIMTVF_BUDGET_BYTES），manifest 四 workload 中 wl1 (1024,4096) fp16/bf16 block_m=16 与 wl4 (128,16384) fp16 block_m=4 落 single | D | DESIGN.md §0.2 workload 落点表（依据 logsumexp.py L176–233 推导） | `PYTHONPATH=<repo> python3 -c "from tileops.device import get_device_backend; print(get_device_backend().shared_memory_budget())"`（于 examples/TileOPs 下） | tileops A5 移植仓 + Ascend950PR_9579，2026-10-16 | 待定（TileOPs 侧知识，pattern-library cases.md 候选） |

### Transferable Lessons（可迁移教训）

- A5 方言 `T.Kernel` 一维且无 `threads=` 参数（TypeError）；线程域在 `T.SimtVF(threads=N)` 内声明；harness 调用面按 K3 移除 GPU 版 `threads` 参数（kernel 内 `min(1024, SUB·TILE)` 自推导）。
- `T.copy` 的 `pad_value=`/`data_select=` 是**列向**行尾 padding（补到 32B 边界），不能用于行方向 OOB（M 尾块）；M 尾块用「host 静态 full/rem 分解 + 循环变量 TIR if（`example_gemm.py` aswt_swizzle L47 先例）+ 静态 rem 切片 copy + fragment 行谓词」规避，不依赖未文档化的 DMA 越界行为。
- fragment 形状必须 pow2（`example_rmsnorm.py` L42 注释）；DMA/UB 侧可紧凑任意 N——「UB 紧凑 + fragment pow2 masked（pad=−inf 双幂元）」是 row-reduction 族替代 GPU N_padded 对齐驻留的标准形态（D-REDUCE-SIMTVF-API measured 佐证「掩码载入与 DMA 紧凑拷贝正交」）。
- block_m 由 wrapper 启发式传入（本算子族 {1,2,4,8,16}，single 优先最大），kernel 侧不可改、不可假设整除 M——设计须内建子批/尾块双消化机制。
- 本机核数实查：`torch.npu.get_device_properties()` → cube_core_num=28 / vector_core_num=56（Ascend950PR_9579）；纯 Vector 算子 grid=56 persistent（TRAP-GRID-DIVISIBILITY：28=4×7 整除性陷阱对 persistent 形态免疫）。
- persistent + 循环变量 TIR if + T.serial 子批 + SimtVF 的嵌套组合各层分别有先例但无单一完整先例——首个尾块用例（M%block_m≠0）应尽早进 L0 验证，fallback 预案（拆双 persistent 域）须在设计期写明。
- S3-JIT 门禁要求 `@tilelang.jit(target="ascend")` 懒工厂形态（工厂返回 PrimFunc），`tilelang.compile` 形态会被拒（CASE-pilot-online-softmax 教训）。
- 所有 python 命令前置 `PYTHONPATH=/home/z00875089/tilelang-fork`（全局 PYTHONPATH 被污染）。

## Stage 2 (design-review) Retrospective — review — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Tooling | `design_calc_check.py` 对多 workload / VRF 预算表 / persistent 形态的设计产出系统性伪影 fail：`check_ub_budget` 把 §4.5 节内**所有**表格行求和进单一 UB 桶（并列 workload wl1+wl4 叠加、VRF 预算表无层级标记被归入 UB、约束说明文本与 "4B" 元素字节乘子被当独立尺寸 → 640.2KB fail，真实单 workload 136KB）；`core_split` 用文中示例数字按朴素 `ceil(M/bm)×ceil(N/bn)` 反推 grid（300 vs 56 fail），未识别 `T.Persistent` 形态下 num_tiles 是调度迭代数而非 grid。检视者被迫亲读解析器源码逐项消解，机械结论不可直接消费 | `.agents/tools/design_calc_check.py` check_ub_budget/core_split 解析逻辑亲读；REVIEW.md 机械复核段（640.15KB = 256.03 wl1+wl4 + 384.1 VRF 表叠加复算吻合）；DESIGN.md §4.5 | ① tilelang-op-design skill 的 §4.5 预算表模板增加机器可读层级/workload 标注（如表题 `<!-- ub-budget: wl1 -->`），使工具可按 workload 分桶、识别 VRF 行；② 工具 docstring 或 tilelang-design-review skill 机械复核段记录已知伪影模式清单 | R |
| Info-source | fragment 层级归属知识冲突：TileOPs harness `_primitives.py` 注释宣称 `alloc_fragment` planned into UB 预算，与本仓实测证据矛盾（online_softmax UB 128KB + fragment 2×128KB fp32 并存真机编译运行 63/63 PASS；CONST-vrf-capacity-a5 记录生成 AscendC 为 per-thread 寄存器数组 `float x_frag[32]`，76 reg/thread spill 0——若 fragment 计入 UB 216KB 则 384KB 先例不可能编译）。消解规则「实测生成代码 > harness 成本模型注释」本次由检视者临场建立，无成文依据 | REVIEW.md 机械复核段与维度 2；`.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md` CONST-vrf-capacity-a5；`examples/pilot/online_softmax/`；`examples/TileOPs/tileops/kernels/reduction/logsumexp/_primitives.py` 注释 | pattern-library/constants.md CONST-vrf-capacity-a5 补注「A5 SimtVF fragment 落 VRF per-thread 寄存器数组，不占 UB 216KB 预算；TileOPs _primitives.py 的 planned-into-UB 注释为规划口径，与实测冲突时以生成代码为准」 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| A5 SimtVF 下 `T.alloc_fragment` 落 VRF per-thread 寄存器数组、不占 UB 216KB 预算——实测样本：online_softmax 主 shape UB 128KB + x_frag/e_frag 2×128KB fp32 fragment 并存编译运行 63/63 PASS，生成 AscendC `float x_frag[32]; float e_frag[32]`、76 reg/thread、spill 0；harness `_primitives.py` "planned into UB" 注释与实测冲突按实测消解 | D | `examples/pilot/online_softmax/online_softmax.py` + 其 DESIGN.md §4；pattern-library/constants.md CONST-vrf-capacity-a5 | `PYTHONPATH=/home/z00875089/tilelang-fork python examples/pilot/online_softmax/online_softmax.py`；`get_kernel_source()` 查寄存器数组形态 | tilelang-fork 工作区 + Ascend950PR_9579 / CANN 9.2.0，2026-10-16 | pattern-library/constants.md（CONST-vrf-capacity-a5 补注） |
| 「wrapper 契约参数兜底声明 vs 伪代码覆盖」检视检查点：设计宣称 host 侧兜底机制消化任意契约参数值（如任意 block_m ≥ 1 由 SUB 子批机制兜底）时，须交叉核对伪代码是否渲染了兜底机制的**全部静态分支**——§3.3 全块路径缺 block_m % SUB ≠ 0 的末子批分支，仅在非 pow2 显式 config 下暴露越界切片隐患（wrapper 启发式 pow2 路径可证安全），声明级兜底 ≠ 实现级覆盖 | P | DESIGN.md §4.6/§6.4 vs §3.3；REVIEW.md 建议注记 2 | none（检视方法论） | n/a | tilelang-design-review skill 检视清单（维度 5/7 交叉核对项增补） |
| design_calc_check 机械 fail 的三类解析伪影模式与人工消解法：①多 workload 并列表叠加（按 workload 分桶人工复算）②VRF/约束文本/元素字节误归 UB 桶（按层级标注剔除）③persistent 形态朴素 grid 反推（num_tiles 为调度迭代数非 grid）——检视者须先读解析器源码定位叠加来源、再人工复算真实值，两者记入 REVIEW.md 机械复核段，不得凭机械 fail 直接判设计不合格，亦不得盲目 dismiss | R | `.agents/tools/design_calc_check.py` 解析逻辑；REVIEW.md 机械复核段（640.15KB 叠加复算吻合、300 blocks 示例数字反推定位） | `python3 .agents/tools/design_calc_check.py --design examples/logsumexp/_logsumexp_kernel_single/DESIGN.md` | tilelang-fork，2026-10-16 | tilelang-design-review skill 机械复核段（已知伪影模式表） |

### Transferable Lessons（可迁移教训）

- 机械复核 fail 的消解前置动作是**亲读工具解析器源码**：design_calc_check 的 ub_budget 按节内全表行求和、core_split 按文中示例数字反推 grid——不读解析逻辑就无法区分「设计错误」与「解析伪影」，两种误判方向（盲信 fail / 盲目 dismiss）都会污染检视结论。
- fragment/buffer 层级归属问题以**生成的 AscendC 源码为最终权威**（`get_kernel_source()` / CONST 条目记录的实测样本）；harness 成本模型注释只是规划口径，与实测冲突时按「实测 > 注释」消解并记录冲突本身。
- 设计文档中描述源算子的数字（如 threads=128）须回 GPU 源仓 grep 亲证——`default_config` 实为 256，128 是同族 logical_reduce/cumulative/argreduce 的默认值；同族算子默认值互相污染是迁移文档的高频误记模式。
- 设计文档中的性能预估数字必须独立复推：「末波 ~1.6% 不均衡」在任何标准负载口径下不可复算（persistent 轮转下实际为 makespan 2τ vs 理想 64/56τ≈1.14τ 的波次量化）；此类数字若入档会误导 Stage 4 归因基线，属 warn 级而非可忽略笔误。
- 同族先例 REVIEW 判例（online_softmax：warn 级 VRF/单位问题不阻塞、机器 check skip/fail 须附人工复核注记）为本次结论尺度提供一致性锚——判定前先对照判例，避免同类问题在不同任务间宽严漂移。
- 行号级引用抽检 ~30 处命中、3 处偏差（±1 行 / 文件错位 / 数字误记）：引用错位属建议级常态，但须逐条验证**证据本身有效性**（如 per_token 两段写实际在 L82/L84，佐证成立仅行号偏），佐证有效性与引用精度分开判定。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| none | none | none | none | - |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| logsumexp 单 tile 驻留 kernel（fp32 中间 + cast 回输出 dtype）对 randn 域 fp16/bf16 输出 max_diff = 0（位级一致），fp32 输出 max_diff ≈ 4.8e-7–9.5e-7——源算法 m+log(Σexp(x−m)) 链与 torch.logsumexp(x.float()).to(dtype) 在测试域位级重合，验证迁移保真（与 verify_equiv V3 的 ~9.5e-8 树形差预估同量级） | D | `examples/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py` L0/L1 全部 PASS 输出（max_diff=0.000e+00 / 4.768e-07 / 9.537e-07） | `PYTHONPATH=/home/z00875089/tilelang-fork python examples/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py --level L1` | tilelang 0.1.15+cuda.git40c9d668 + Ascend950PR_9579 / CANN 9.2.0，2026-10-16 | pattern-library/elementwise.md |
| 「Persistent + 循环变量 TIR if（`if t < full`）+ T.serial 子批 + SimtVF + annotate_buffer_versions({x_ub:2})」嵌套组合在 A5 端到端编译运行通过（含尾块路径，L0 ④ (33,256) 即触发）——消除 DESIGN 风险 3 的「无单一先例」不确定性，四层组合不再是排查点 | P | `_logsumexp_kernel_single.py` main() 结构；L0/L1 63+ 用例全 PASS（含 M%block_m≠0、M<block_m、非 pow2 N） | `PYTHONPATH=/home/z00875089/tilelang-fork python examples/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py --level all` | 同上 | pattern-library/layout.md 或 elementwise.md（row-reduction 族模板） |
| `T.And(a < b, c < d)` 在 `tilelang.ascend.language`（shadow 通用面）可用，SimtVF masked 载入中正确降级——行尾 + 列尾联合谓词（`T.And(r < rows, i < N)`）无需绕行；源 GPU kernel 的 T.And 用法可直译 | D | `_logsumexp_kernel_single.py` masked 载入段（T.And 联合谓词）；`tilelang/language/ast/ir.py` L57/L2209 导出 | `python -c "import tilelang.ascend.language as T; print(hasattr(T,'And'))"`（True） | 同上 | tilelang-ascend skill §2 或 §6（API surface 注记） |
| SUB 封顶（x_frag ≤ 128KB）后 wl1 无寄存器 spill：生成 AscendC `float x_frag[32]`（SUB=8×TILE=4096/1024 threads = 32 regs/thread）+ m/s/out_frag 标量 ≈ 48 regs/thread，远低于 CONST-vrf-capacity-a5 实测 76 reg 水位——行子批机制把 VRF 从贴顶（256KB）压到舒适水位 | D | `_logsumexp_kernel_single.py` `--show-source` 生成源码（`float x_frag[32]`）；DESIGN §4.5 VRF 预算表 | `PYTHONPATH=/home/z00875089/tilelang-fork python examples/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py --show-source \| grep x_frag` | 同上 | pattern-library/constants.md（CONST-vrf-capacity-a5 补样本） |
| Ascend A5 NPU 设备不支持 double dtype：torch 张量 `.double()` 在 NPU 上静默降级为 float32 并打 warning（`ToKernelNpu.cpp:40 Device do not support double dtype`）——fp64 参照/模拟器须在 CPU 侧跑，NPU 侧 fp64 是伪 fp64 | D | `_logsumexp_kernel_single.py` `_source_lse_sim` 由 `.double()` 改 `.float()` 的修复记录；运行 warning 原文 | `PYTHONPATH=<repo> python -c "import torch,torch_npu; x=torch.ones(2,2,device='npu'); print((x.double()).dtype)"`（观察 warning） | 同上 | tilelang-ascend skill 或 pattern-library/traps-runtime.md |

### Transferable Lessons（可迁移教训）

- 尾块 copy 目的切片必须写成 `x_ub[0:rows, 0:N]`（源区域 (rows,N) 与 x_ub (SUB,N) 形状匹配），否则编译期报形状错——REVIEW 建议注记 3 的实战确认（先例 online_softmax L142 `x_ub[0:BM, 0:rem]` 同构）。
- `-inf` pad 同时是 max 幂元与 exp 后 sum 幂元，单一 masked 载入 `T.if_then_else(T.And(r < rows, i < N), cast, -inf)` 同时消化行尾与列尾；被丢弃行（r ≥ rows）内部链产生 NaN 但不写出，reduce_* 沿 dim=1 逐行独立故不污染有效行。
- 尾块 TIR if（`if t < full`）直接写在 `T.Persistent` 循环体内即可（t 为循环变量、full 为 host 常量），无需 `T.macro` 包装——比源 GPU 的 `with T.If/T.Then/T.Else` 旧语法更简洁；`full/rem/tsub/tsub_rem/SUB` 全部 host JIT 常量才能让切片静态化。
- 行子批串行循环用 `T.serial`（保持 x_ub 双版本流水顺序依赖），不要用 Python range 展开（TVMScript 会把 range 也解析成串行循环而非 IR 构建期展开）；末子批（rows < SUB）用 Python 静态 `if rows > 0:` 分支 + 静态切片，与尾块 tsub_rem 分支同构。
- 本任务后续函数（`_logsumexp_kernel_tiled`）复用同一 per-chunk 计算体：把本 kernel 的「单 tile 驻留」换成「N 向 chunk 串行 + running (max,sum) flash 合并」，persistent 域 / SimtVF masked 载入 / reduce_max/exp/reduce_sum / lse+cast / 两段写结构全部原样保留。
