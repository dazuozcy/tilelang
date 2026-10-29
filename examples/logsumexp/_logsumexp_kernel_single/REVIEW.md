# _logsumexp_kernel_single 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/logsumexp/_logsumexp_kernel_single/DESIGN.md（878 行，Stage 1 首版 + 1 次机械门禁窄修复）
- 任务类型: 迁移（migration-harness：GPU TileOPs → Ascend A5，dav-3510）
- 源算子: examples/TileOPs/tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py（`_logsumexp_kernel_single` L42–100，已亲读全文）；wrapper: 同目录 logsumexp.py（已亲读全文）；GPU 源仓 /home/z00875089/TileOPs-fork（只读，用于 threads 契约查证）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 7（均以「建议注记」内嵌于对应维度详情，供 Stage 1 择机修订与 Stage 3 实现时注意；不阻塞进入 Stage 3）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: PASS。命令 `PYTHONPATH=/home/z00875089/tilelang-fork python examples/logsumexp/_logsumexp_kernel_single/verify_equiv.py`（2026-10-16 本检视执行）。执行结果与 §1.6.1 内嵌表**逐项一致**：V1（6 shape × fp64/fp32）ulp=0 位级一致 EQUIV_PASS；V2（原位覆写 6 shape）ulp=0 EQUIV_PASS；V4（源 GPU N_padded 链 vs A5 紧凑链，fp16 输出含特殊行）ulp=0 + NaN 模式一致 EQUIV_PASS；V3（A5 链 vs torch.logsumexp fp32）max ulp=1 @N=300、max_rel=9.526e-08、违反率 0 EQUIV_PASS——与设计表数字（9.526e-08）逐字一致；V3b 四角点（全 −inf→chain NaN vs torch −inf；含 +inf→chain NaN vs torch +inf；NaN→双方 NaN；部分 −inf→一致）与 §9.3 分歧文档一致；V5 fp16-e 备选模拟 4.211e-05 / 3.589e-05 GUARD_PASS，与 §1.6.4 引用一致。末行 `ALL ADOPTED-ITEM CHECKS: EQUIV_PASS`。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design examples/logsumexp/_logsumexp_kernel_single/DESIGN.md` JSON checks 摘要：
  - `ub_budget`: **fail**（"15 buffer entries sum = 640.2 KB" vs 194 KB 预算）——**人工复核判定为解析伪影，非设计错误**。已亲读工具解析代码（`.agents/tools/design_calc_check.py` `check_ub_budget`）：它把 §4.5 节内**所有**表格行尺寸求和进单一 "UB" 桶——含 ① wl1 与 wl4 两个**并列 workload**（二者非同时分配，各自 ≈136KB）；② VRF 预算表（工具无 VRF 层级标记，x_frag 行 128+128KB 被归入 UB）；③ 约束说明文本（"≤128KB 封顶" 说明列）；④ 元素字节数（"4B/2B" 乘子被当独立尺寸）。15 条 ≈ 256.03（UB 表 wl1+wl4）+ 384.1（VRF 表）= 640.15KB，与工具输出吻合，确认重复叠加。**人工复算真实单 workload 分配**：UB = x_ub 双版本 128KB（wl1: 2×8×4096×2B；wl4: 2×2×16384×2B）+ lse_ub 16B + 编译器保留 ~8KB ≈ **136KB ≤ 194.4KB**（216KB SimtVF × 90% 折扣，CONST-capacity-a5 口径已披露）✓；fragment 归 VRF 不占 UB 预算的口径有**实测先例支撑**：online_softmax 主 shape（UB 128KB + x_frag/e_frag 2×128KB fp32 fragment 并存）在真机编译运行 63/63 PASS，CONST-vrf-capacity-a5 实测样本记录生成 AscendC 为 per-thread 寄存器数组（`float x_frag[32]; float e_frag[32]`，76 reg/thread，spill 0）——若 fragment 计入 UB 216KB 预算则该先例（384KB）不可能编译通过。TileOPs harness `_primitives.py` 的 "alloc_fragment planned into UB" 注释与该实测证据冲突，按「实测 > harness 成本模型注释」口径以本仓生成代码实证为准。
  - `l0c_budget`: **warn**（"block 56x100 x fp32 accum"）——**解析伪影**：本设计为纯 Vector 算子，无任何 Cube/L0C 段（§1.5/§4.4/§5.1 明确宣告不适用）；工具把 D-2 估算文本中的 "56 核 × 100 B/cycle" 误读为 block_M×block_N。人工复核：L0C 检查不适用。
  - `core_split`: **fail**（"ceil(1/16)×ceil(300/1)=300 blocks vs declared grid 56"）——**解析伪影**：工具用文中示例数字（M≥1 下界、N=300 用例）按朴素 `ceil(M/bm)×ceil(N/bn)` grid 反推，未识别 `T.Persistent` 形态。本设计 grid=56（=vector_core_num 实查值）为 persistent 单波常驻，num_tiles=ceil(M/block_m) 是**调度迭代数**而非 grid（core-split-strategy §1 认可的 persistent 化形态；同族先例 online_softmax 同为 grid 56 + 2048 tiles，Stage 2 通过并 Stage 3 实测 PASS）。人工复核：三要素齐备且自洽（详见维度 3）。
  - `r3_metrics`: **pass**（四口径 FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值俱全）✓。
  - 无 skip 项；两项 fail 与一项 warn 均经上述人工复核消解，结论按人工复算（机械结果与独立推演互补口径）。

## 检视详情

### 0. 源算子理解与迁移分析 — warn

检视方式：亲读源 kernel 全文（`_log_sum_exp_fwd_kernels.py` L42–100 single 变体 + L101–189 tiled 变体）、A5 wrapper 全文（`logsumexp.py` 251 行）、`_primitives.py`（MAX_SINGLE_TILE_COLS/compute_tile_n/device_smem_budget）、op 层 `_validate`、GPU 源仓 `/home/z00875089/TileOPs-fork` threads 契约，与 §0 逐项核对。

- **语义理解正确性 — pass**：§0.1 语义要素表与源码逐条一致。数学映射 `y[i]=row_max[i]+log(row_sum[i])`、`row_sum=Σexp(x−row_max)` ↔ 源 L85–94 ✓；fp32 中间（`x_f32`/`row_max`/`row_sum` 均 "float32" fragment，源 L65–67）✓；输出 cast 回输入 dtype（`out_local` 输入 dtype fragment，L92–94）✓；dtype 白名单 {fp16,bf16,fp32} ↔ op 层 `_validate` L111（`x.dtype not in (torch.float16, torch.bfloat16, torch.float32)` 亲验）✓；边界语义五项独立推演成立：NaN 行→y=NaN（robust——即使 max 硬件语义吞 NaN，NaN 元素经 exp→sum 加法链仍传播）、全 −inf 行→NaN（m=−inf→exp(−inf−(−inf))=NaN，IEEE 直推 ✓，与 torch 的 −inf 分歧为源固有，verify_equiv V3b 重跑实证 ✓）、含 +inf 行→NaN（inf−inf=NaN ✓ V3b 实证）、部分 −inf 中性（exp(−inf−m)=0 ✓）、M 尾块 masked 路径行谓词 `T.And(pid_m*block_m+i<M, j<N)` ↔ 源 L75 逐字 ✓；尾块/非整除双路径（`_needs_pad` L51–52/L69–83）✓；−inf 双幂元（max 幂元 ∧ exp 后 sum 幂元）论证 ✓ V1 位级验证。
- **I/O 与契约空间（§0.2）— pass**：kernel 签名 `_logsumexp_kernel_single(M,N,dtype)` ↔ 源 L42 ✓；A5 调用面 `(block_m)(x)` ↔ wrapper L83 `_logsumexp_kernel(M,N,dtype_str,tile_n)(block_m)(x)` 亲验 ✓；K3 threads 移除 ↔ wrapper 模块头 L26–28（"threads removed … callable takes block_m only"）+ add-ascend-op SKILL.md L146（K3 规则定义）✓；single 选中两条件亲验：① `N_padded ≤ MAX_SINGLE_TILE_COLS=32512`（`_primitives.py` L51 + wrapper L188）✓；② `compute_tile_n` 早返回 `num_buffers(=1)·block_m·elem·N_padded ≤ budget`，budget=`device_smem_budget()`=`UB_SIMTVF_BUDGET_BYTES`=248−32=216KiB（device.py L50–52/L176 亲验）✓；`default_config` single 优先最大 block_m ∈ {1,2,4,8,16} ↔ wrapper L212–233 ✓；manifest 四 workload 落点表算术复算全部成立（wl1: 16×2×4096=128KB≤216KB→single bm=16 ✓；32k: 32768>32512→tiled ✓；lm-head: 102400>32512→tiled ✓；3d [4,128,4096] dim=[0,2]→(128,16384): 4×2×16384=128KB ✓、8 行 256KB 超预算→bm=4 ✓）；test_softmax.py 落点抽查 ✓（(32,300)/(33,256)/(33,300) single、(4,32768)/(33,33000)/(33,32768) tiled、1D 300、均亲验测试文件 L60–83/L198–200）。
- **算法解读完整性 — warn（一处数字误记，建议级）**：§0.3 步骤表 S0–S6 覆盖源码全部计算语句（S0↔L51–52、S1a↔L73–78、S1b↔L80–83、S2↔L85–86 含 `T.fill`+`clear=False` 细节、S3↔L88–89 原位、S4↔L90、S5↔L92–94、S6↔L96），无遗漏无臆造 ✓；数据流图覆盖全部 buffer（shared_buf/x_local/x_f32/row_max/row_sum/out_local）✓；host 侧逻辑（align_up 常量折叠/_neg_inf/JIT+lru_cache）✓。**建议注记 1**：§0.3 "threads=128，行内 64 元素/线程量级" 中 threads 值与源仓不符——GPU 源仓 `TileOPs-fork/tileops/kernels/reduction/logsumexp.py` L395 `default_config` 返回 `"threads": 256`（128 是同族 logical_reduce/cumulative/argreduce 的默认值，非 logsumexp）。该数字仅出现在描述性括注中，threads 契约已按 K3 舍弃、行内并行度由 SimtVF 线程域承接（§0.5 末行意图承接结论不受影响），属引用级误记——建议 Stage 1 修订时改为 256 或删除具体值。
- **优化手段覆盖完整性 — pass**：§0.4 七项手段逐条与源码对应（#1 驻留↔L63 shared_buf 整行块、#2 双路径↔L69–83、#3 稳定化+fp32↔L76/L88–89、#4 原位↔L89 写回 x_f32、#5 树形归约↔L86/L90、#6 行批↔block_m 参数化、#7 非持久 grid↔L62）；层级归属（算法层/优化层）合理；自检声明（无手写 warp shuffle/swizzle/cp.async/persistent）经亲读源码证实 ✓。
- **耦合性判定合理性 — pass**：§0.5 十一条目全部有四态处置+依据；与 backend-glossary §1 概念迁移锚一致（shared→UB、warp shuffle→reduce intrinsic〔T.reduce_* 经 D-REDUCE-SIMTVF-API measured 实证降级 AscendAllReduce 硬件树，fp32 在直连 dtype 集内〕、CUDA 线程模型→SimtVF、核数实查）；与 ascend-constraints §1 一致（一维 Kernel/threads= TypeError、UB 216KB SimtVF、静态循环边界、flag 预算、pad_value 列向限定——亲验约束文档逐条）；舍弃项（host threads）有理由+意图承接 ✓；"未文档化" 项（A5 T.copy 行向 OOB）按保守规避处置（重新设计 R4）符合负向断言证据规则 ✓。判定统计行 "无静默丢弃" 经逐项核对成立。
- **NPU 重设计可行性 — pass（附 3 处建议注记）**：R1–R4 均有「源方案→NPU 新算法→语义保持论证」三要素。R1（一维 T.Kernel(56)+Persistent+SimtVF）：56=vector_core_num 实查（与 CONST-arch-a5 measured 条目及本任务调度 prompt 环境信息一致）✓，行独立性→映射变化无数值影响论证 ✓，归约树深舍入差 ≤13×2⁻²⁴ 估计与 V3 实测 9.5e-8 相符 ✓，先例 online_softmax L57–63 同构亲验 ✓。R2（紧凑 UB+pow2 masked+行子批）：SUB 公式独立复算四例全对（wl1→8、wl4→2、(32,256,fp32)→16、N=32512→1）✓；−inf 幂元中性论证经 V1/V4 位级机器验证 ✓；**建议注记 2（伪代码与 §6.4 不一致）**：§3.3 全块路径 `T.serial(num_sub)` 统一 SUB 行切片，未渲染 §6.4 已声明的 "全块路径末子批行数 = block_m−(num_sub−1)·SUB" 静态分支——wrapper 启发式路径（block_m∈{1,2,4,8,16} 皆 2 幂→SUB|block_m，num_sub·SUB=block_m 恰尽）可证安全，但 §4.6 声称的 "显式 config 任意 ≥1 由 SUB 机制兜底" 在 block_m 非 SUB 整数倍时（如 block_m=12→SUB=8）末子批切片 `[base+(num_sub−1)·SUB : +SUB]` 会越过 tile 边界乃至 M（末个全块 tile 时越界 DMA，A5 行为未文档化——设计 R4 自己的立论依据）。建议：§3.3 全块路径补静态末子批分支（与尾块路径 tsub_rem 分支同构，host 常量 if），或 §4.7 host assert 增补 `num_sub·SUB == block_m or 末子批分支已实现` 类约束条款。**建议注记 3（尾块 copy 缺目的切片）**：§3.3 尾块末子批 `T.copy(x[row_base:M, 0:N], x_ub)` 源区域 (tsub_rem,N) 与 x_ub (SUB,N) 形状不匹配，应为 `x_ub[0:tsub_rem, 0:N]`（同构先例 online_softmax L142 `x_ub[0:BM, 0:rem]` 亲验）；Stage 3 会在编译期报形状错，属伪代码精度问题。R3（persistent+双版本+AutoSchedule）：纯调度变换论证 ✓，三件套先例亲验 ✓。R4（静态 full/rem+TIR if+静态切片）：先例链亲验（example_gemm L47 `if row_idx < MAIN_ROW:` 确在 T.Persistent 循环体内经 L69–70 aswt_swizzle 调用；online_softmax L141 `if rem > 0:` 闭包常量分支）✓；fallback 预案（拆双 persistent 域）API 全先例 ✓；**建议注记 4（单位口径）**：§4.5 N=32512 推导 "194.4K/(2·32512·2)=1.49 ∧ 128K/(32768·4)=0.97→SUB=1" 中比值用十进制 KB（194400/128000）而结论 "x_frag 128KB ✓" 用 KiB（131072B 恰达 128KiB 封顶）——SUB=1 结论在两种口径下均成立，建议统一口径标注避免 Stage 3 实现歧义。
- **设计与决策一致性 — pass**：§1.4 主路径 8 步 = R1–R4 落地（结构差异 6 条逐条注明 §0.6 出处）；§3.2/§3.3 API 映射与伪代码 = §0.5/§0.6 决策后形态（无 SMEM staging、无 `_needs_pad` 分支、无 threads 参数、persistent 而非 ceildiv grid）；§5.2 SUB/§6 循环/§7 同步与 R2/R3 一致；§3.5.2 明示 "不复用源码任何 GPU 形态结构" ✓。无 "§0 判重设计、§3 照抄源方案" 脱节。
- **golden 独立性 — pass**：§8.1 golden = `torch.logsumexp(x.float(), dim=-1).to(x.dtype)` on NPU——源仓 `tests/ops/test_softmax.py` L98–99 `ref_program` 直移植（亲验逐字一致），以 §0.1 语义为依据，未复刻 §0.6 NPU 数据流（无子批/masked/persistent 任何痕迹）✓；±inf 角点另以源语义模拟对照（Boundary ④），不与 golden 混用 ✓。
- **建议注记 5（引用错位）**：§0.1 host 语义行引 "`logsumexp.py` L193–207" 支撑 "wrapper 层负责 dim 归一化/transpose/contiguous/reshape"——该行段实为 kernel wrapper 的 `_tile_n_for_block_m` 预算逻辑；dim 归一化/flatten/reshape 职责在 op 层 `tileops/ops/reduction/logsumexp.py`（其模块头 L17–19 明列该流水线，§10.5 已正确列出该文件、§4.1 亦正确归因 "op 层"）。建议修正 §0.1 该行引用指向。

维度 0 总评：语义/契约空间/算法步骤/优化手段/耦合性/重设计/一致性/golden 八项核对全部成立，重设计四项经独立复算与先例亲验可行；warn 仅源于 5 处建议级注记（1 处源数字误记 + 3 处伪代码/口径精度 + 1 处引用错位），均不动摇迁移决策与数值语义，不阻塞。

### 1. API 可行性 — pass

- **API 存在性**：§3.2/§3.4 全部 API 亲验佐证——`T.Kernel(56)` 一维无 threads（online_softmax L57 measured 先例 + ascend-constraints §1 条 1）；`T.Persistent([num_tiles], 56, core_id, num_stages=2)`（签名亲验 `tilelang/language/loop.py` L82–110：domain/wave_size/index/num_stages 形参吻合；online_softmax L60 + per_token L45 在用）；`T.SimtVF(threads=)`（`tilelang/ascend/language/frame.py` L107）；`T.reduce_max/reduce_sum(dim=1, clear=True)`（`tilelang/ascend/language/reduce_op.py` L95/L111，clear 形参在 L48/L96 亲验；per_token L73 + online_softmax L75 measured）；`T.if_then_else`/`T.And` 掩码（源 L74–75 同构 + online_softmax L71–74）；`T.exp`/`T.log`（intrin_rule_ascend.cc L45–50 REGISTER_ASCEND_INTRIN 亲验注册；ascend_simdvf_lower_parallel.cc L411–415 tir.exp→simd_vexp / tir.log 亲验；D-ELEM-SIMTVF-AUTOVEC measured）；`T.infinity(T.float32)`（tilelang/language/tir/op.py L361 + D-REDUCE-SIMTVF-API measured 注记 "接受 dtype 对象"）；`T.copy` GM→UB 2D 静态切片/fragment→UB/UB→GM 1D 切片（per_token L50/L82/L84、rmsnorm L76/L82、online_softmax L62/L142 亲验）；`T.annotate_buffer_versions`（ascend/language/annotations.py L19 + 两先例）；核内 TIR if 循环变量条件（example_gemm L47 于 Persistent 体内调用亲验）；Python 级常量 if（online_softmax L141）；`@tilelang.jit(target="ascend", out_idx=[1])` 懒工厂（online_softmax L215–223 `_is_lazy_style` 形态亲验）。testing 回归引用真实存在（test_tilelang_ascend_reduce.py / test_ascend_thread_sync.py 亲验 ls）。**建议注记 6（行号 ±1）**：per_token "L81–83 两段写" 实际语句在 L82（`T.copy(y_s_local, y_s_ub)`）与 L84（`T.copy(y_s_ub, scales[...])`），佐证有效，行号建议校正。
- **编程模式匹配**：simt + `auto_schedule: yes`（§2.1/§7.1），无手动 set_flag/wait_flag/pipe_barrier 混写 ✓；无 `T.Kernel(N, threads=)` 误用（§9.2 首条自查清单）✓；pass_configs 不传 TL_ENABLE_AUTO_SCHEDULE ✓。
- **SimtVF 优先**：默认 SimtVF+T.Parallel ✓；SimdVF 仅作为 Stage 4 证实热点后的降级选项且给出量化优先级判断（vexpdif 仅省 1 sub @51 档，低于 fp16-e 备选优先级）——`T.simd.vexpdif` 定义亲验（simd.py L1182–1195：dst=exp(src0−src1)、要求 matching float32 向量，设计 "fp32 同宽形态" 表述准确；flash_attention/core.py L130–133 在用亲验）✓。

### 2. 内存层级规划 — pass

- **搬运路径完整**：§4.4 GM→UB（MTE2 2D 行子批）→VRF（masked 载入）→UB（fragment→UB）→GM（MTE3 1D 切片）——全部在 tilelang-ascend skill §6 支持集内 ✓；无 L1/L0A/L0B/L0C 段（纯 Vector，正确宣告）✓；无手写 NZ 分形 ✓；无中间量落 GM ✓。
- **UB 预算**：x_ub 双版本（版本数已计入，skill §4.3 要求）+ lse_ub + ~8KB 保留 ≈136KB ≤ 194.4KB（CONST-capacity-a5 assumption、216KB SimtVF、90% 折扣——条目与折扣口径已在 §4.5 披露 ✓）；SUB 上界公式第一条约束（2·SUB·N·elem ≤ 194.4KB）保证契约空间全域 UB 达标（fp32 更紧情形 N=4096→SUB=4→128KB 亲算 ✓）。机械 ub_budget fail 为解析伪影（见机械复核段，人工复算通过）。
- **VRF 预算**：x_frag ≤128KB（SUB 公式第二条封顶）≈32 reg/thread + ~12 标量 ≈44 reg，对照 CONST-vrf-capacity-a5 实测下界样本（online_softmax 76 reg/thread spill 0）留有余量 ✓；基线对照行（block_m=16 整批 = 256KB 贴假设上限 + 76 reg 贴实测水位）诚实披露 ✓；§9.1 风险 1 + §5.3 给出 Stage 3 spill 首查与降 SUB 退路（条目 "转正方法" 口径一致）✓。harness `_primitives.py` "fragment planned into UB" 注释与实测先例冲突，按生成代码实证口径消解（见机械复核段）。
- **Buffer 层级标注**：§4.3 五个 buffer 全部标注层级+alloc API（x_ub/lse_ub=UB T.alloc_shared；x_frag/m_frag/s_frag/out_frag=VRF T.alloc_fragment）✓。l0c 机械 warn 为伪影（无 Cube 段）。

### 3. Tiling 策略 — warn

- **Block 划分与 grid 映射**：§5.2 给出 TILE=next_pow2(N)（fragment pow2 约束，rmsnorm L42 注释亲验）+ SUB 双约束封顶公式（独立复算 4 例全对，见维度 0）+ num_sub/full/rem/num_tiles/tsub/tsub_rem/threads 全套 host 静态化推导 ✓；grid 一维 T.Kernel(56)，行块经 T.Persistent 调度（多波由调度器分配，无需手写一维映射）✓。
- **对齐约束**：无 GEMM→C0/NT/分形不适用（§5.3 明确，CONST-gemm-dim-constraints 仅约束 T.gemm 路径——口径正确）✓；GM→UB 行 32B 对齐分析（wl1 8KB/wl4 32KB ✓；非 32B 行长 N=300→600B 字节级行长支持引 skill §6.1 "byte-granular rows" 亲验原文，带宽档降级列 §9.1 风险 2 观察项——CONST-mte-bandwidth-model 平台/未知档 50/48 引用与条目一致）✓；向量宽度 TILE%128（fp16 lane）/%64（fp32 lane）=0 ✓。
- **非整除处理**：M 尾块（静态 full/rem+TIR if+静态切片+行谓词，§5.4/§6.4/R4）✓；列尾（masked −inf 幂元，V1 位级验证）✓；M<block_m（full=0 全尾块）✓；子批尾——§6.4 已声明处理但 §3.3 伪代码未渲染全块路径末子批分支（建议注记 2，见维度 0）。
- **分核策略三要素（core-split-strategy §1 逐项）**：① grid 划分——NUM_BLOCKS=56 有真机实查记录（§5.5 内嵌查询代码 + 返回值 Ascend950PR_9579 / cube 28 / vector 56 / 126GB / L2 128MB，与 CONST-arch-a5 measured 条目及调度 prompt 环境信息一致），惯例初值 32 被实测取代并注明 ✓，persistent 单波常驻 + num_tiles 多波关系明确 ✓；② AIC/AIV 分工——纯 Vector 仅占 AIV（56），无 Cube 工作，pure kernel codegen 自动判定、无 MixedKernel/sid/dual_copy 所有权问题 ✓；③ 同步策略——auto_schedule: yes + 5 条数据依赖边界枚举（MTE2→VALU、VALU→MTE3、版本交替、SimtVF 跨线程 asc_syncthreads、reduce 终值顺序）+ 调度提示位置（Persistent num_stages=2；不用 T.Stage/T.Task 有说明）+ flag 预算不占用声明 ✓。核内串行边界全 JIT 静态常量 ✓。机械 core_split fail 为 persistent 形态伪影（见机械复核段）。
- **建议注记 7（尾波不均衡数字不可复算）**：§5.4 "wl1 num_tiles=64 = 56+8（末波 8/56 核，~1.6% 不均衡）"——独立复推：persistent 轮转分配下 8 核各承担 2 tile、48 核各 1 tile，makespan = 2τ vs 理想 (64/56)τ ≈ 1.14τ（波次量化开销至多 ~75%；核时空闲率 (48×1τ)/(56×2τ) ≈ 43%），1.6%（≈1/64）无法由任何标准负载口径复算得出。该数字不影响结构正确性（persistent 对任意 num_tiles 有效）且设计已列 Stage 4 block_m 对齐调优观察项（core-split-strategy §2.4 提示、block_m 为 wrapper config 可调），但作为 bench 主对象（wl1）的性能预期表述会误导 Stage 4 归因基线——建议修订为波次量化口径（makespan 2 波 vs 理想 64/56 波）或改定性描述；D-2 估算 "除以 56" 的均衡假设宜附同口径注记。wl4 "32<56、24 核空转一轮、绝对影响可忽略" 的定性表述反而准确。

### 4. 技术约束检测 — pass

- 一维 grid：T.Kernel(56) 单参 ✓，无 threads= ✓（源 `T.Kernel(grid, threads=threads)` L62 已在 R1 重设计，§3.5.1 首行明示）；多维分解不适用（行块一维任务流）。
- GEMM 分形/NT/L0C：不适用且已正确宣告（§3.5.1/§5.3——无 T.gemm/L1/L0 段）。
- ascend-constraints §2 强制检测表：§3.5.1 十行全覆盖（一维 Kernel/GPU 专用 API/C0/L0C-UB 溢出/OOB padding 误用〔pad_value=、data_select= 列向行尾 padding 语义亲验 skill §6.1 原文 "fill the row tail up to the next 32B boundary"，行 OOB 用谓词+静态切片规避——口径正确〕/分核策略/静态循环边界/flag 预算/Host 侧输入操作〔仅元数据推导，op 层已 contiguous 化〕）✓。GPU 专用 API 检测：源仅用 T.copy/T.reduce_*/T.Parallel/T.if_then_else 表层共享 API，无 TMA/cp.async/warp shuffle/cluster 直接依赖 ✓（亲读源码证实）。

### 5. 循环与同步 — warn

- **循环结构**：§6.1 五行表（persistent 行块 / serial 子批 / TIR if 尾块 / Parallel 元素级×3 / reduce 硬件归约×2）选择与理由明确 ✓，与 §1.6.2 向量化结论一致（无逐元素标量循环，§6.1 表头明示）✓。
- **同步与模式匹配**：§7.1 auto_schedule: yes ↔ §2.1/§5.5 声明一致 ✓；§7.2 依赖边界表 5 行与 §5.5 ③ 一致 ✓；无手动 flag ✓；multibuffering 优先编译器管理（annotate_buffer_versions 固定 2 版本 + AutoSchedule，skill §4.2 口径）✓。**风险 3 诚实披露**（Persistent+TIR if+serial+SimtVF 嵌套组合无单一先例，各层分别有先例）+ fallback 预案（拆双 persistent 域，语义等价、API 全先例）+ L0 ④ (33,256) 首个尾块用例即触发验证——风险管理充分 ✓。
- **尾块处理**：§6.4 三层（行尾/列尾/子批尾）说明在案 ✓；但 §3.3 伪代码未渲染全块路径末子批分支（建议注记 2）与尾块 copy 目的切片（建议注记 3）——两处均为伪代码精度问题，§6.4 文字与先例（online_softmax L142）已给出正确形态，Stage 3 按 §6.4+先例实现即可，不阻塞。

### 6. 验证方案 — pass

- **Golden 函数**：§8.1 含 PyTorch 参考实现草案，`x.device.type == "npu"` 断言 + torch on NPU ✓；源仓 ref_program 直移植（test_softmax.py L98–99 亲验逐字一致）✓；独立性成立（维度 0 已核）。
- **L0 测试计划**：5 个具体门槛用例（wl1 (1024,4096) fp16 bm=16 / wl4 (128,16384) fp16 bm=4 / smoke (32,256) 三 dtype / (33,256) M 尾块 rem=1 / (32,300) 非 pow2 N）——block 整除与尾块两形态、三 dtype、契约代表 shape 全覆盖 ✓；L1（shape/config 扫描含 SUB 分裂点、M=1、1D flatten、多维 flatten）、L2（多波 M=8192、契约边界 N=32512 SUB=1 退化、tiled 分界负向 assert）、Boundary（有限域角点 9 项 + 部分 −inf + NaN equal_nan + 源继承分歧角点以源语义模拟对照 + 非连续输入 + 确定性 3× 位级）分层完整 ✓，与 §0.2 落点分析对齐 ✓。
- **精度标准**：容差分档 fp16/bf16→1e-2、fp32→1e-3 ✓（AGENTS.md + gpu_to_ascend_adaptation.md §1.8 亲验一致）；fp32 取 1e-3 档（非 accumulation-heavy）有依据——单遍行归约、V3 实测 max_rel 9.5e-8 余量 4 个数量级 ✓；`torch.testing.assert_close` ✓。性能口径 §8.3：do_bench(backend="msprof") 唯一权威 + bench_softmax.py baseline_fn=torch.logsumexp（亲验 L58–59）+ §1.6.4 A/B 必做项 ✓。

### 7. 完整性与一致性 — warn

- **章节齐全**：迁移模板 §0（0.1–0.7 全）+ §1–§10 全 ✓；§1.6 四子节 + 1.6.4 三件套 ✓。
- **无占位符**：grep TODO/待补充/FIXME/placeholder 零命中 ✓。
- **同类引用真实**：§3.5.3/§10.5 全部路径亲验存在（online_softmax 目录、example_per_token_cast_to_fp8.py、example_rmsnorm.py、example_gemm.py、flash_attention/core.py、TileOPs 六工件、pattern-library 三文件八条目、testing 两回归文件）；行号级引用抽检 ~30 处，除建议注记 5（§0.1 L193–207 文件错位）、注记 6（per_token ±1 行）、注记 1（threads=128→256）外全部命中。
- **内部一致**：§1.4/§3.1/§3.2/§3.3/§4.3/§4.4/§4.5/§5.2/§6/§7 交叉核对基本一致（公式链、buffer 形状、SUB/threads 推导、同步声明三处一致）；不一致点即建议注记 2/3/4/7（§3.3 vs §6.4 子批尾、尾块 copy dst 切片、KB/KiB 混用、1.6% 数字）——均为局部精度问题，无设计级矛盾。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性**：1.6.0（调研四问）/1.6.1（数学等价优化）/1.6.2（向量化替代）/1.6.3（轴与布局决策）/1.6.4（实验裁决三件套）齐全 ✓。
- **算法调研存在性与完整性**：R1 候选表 10 行**含基线**（#0 源驻留三段式）✓；algorithm-candidates.md 命中行亲核（ALG-softmax：减 max 必选→#1、exp2 换底→#2、online→R2/#7、三遍/两遍/单遍+缓冲→R3、水平 vs 垂直+UB 驻留→R4/§1.6.3；ALG-reduction：max+sum 合并扫描→#7 在线重缩放形态、分块归并→R2、水平/垂直 lane→§1.6.3；ALG-layernorm row-reduction 形态→R2 结论引用亲验原文一致）——命中行候选全覆盖，无漏候选 ✓；另自评 Cube GEMV(#8)/两 kernel(#9)/fp16-e(#6) 三个超出候选表的变体，调研深度与规约/统计类"完整调研"定级匹配 ✓。R2 明确结论"**有**"且三层证据（源仓 tiled 变体 L101–189 running max+rescaled sum 亲读证实、参考表命中行、结构判据）✓。R3 四口径表（FLOPs/GM Bytes/扫描遍数/中间缓冲+可并行度列）基线在表、口径逐条写清 ✓（r3_metrics 机械 pass）。R4 逐候选六项清单 ✓。调研范围声明 ✓（含 kb_search 命中与 VP-2026-0017 pending 交叉印证——queue.md L60–66 亲验内容一致；互联网未使用有理由声明）。
- **调研结论正确性（独立复核）**：① 等价性——选定链与源链公式级恒等，结构变换（O2/O3）经 V1/V2/V4 位级机器验证（重跑 PASS）+ 独立推演（−inf 双幂元含 m=−inf NaN 路径两链一致）✓；② 复杂度复算——基线/选定 FLOPs 持平（公式同）✓、GM Bytes：对齐 2B/elem、非对齐 N=300 基线 512 列 3.41B/elem（设计 3.4 ✓）选定 2B（−41% ✓）、两 kernel 6B（2 读+1 写+max 往返可忽略，口径自洽）✓、扫描遍数 2=读 1+写 1（与 §4.4 口径一致）✓、缓冲列（基线 wl1 128KB UB+256KB VRF 贴顶 / 选定 ≤194.4+≤128）复算一致 ✓、**中间缓冲无 GM 往返漏计**（驻留单遍结构，无落 GM 中间量——§1.5/§4.4 双处声明一致）✓；③ 亲和性淘汰依据——容量/带宽淘汰全部引 constants.md 条目（CONST-capacity-a5/vrf/valu/mte，D-2 纪律 ✓）；④ **负向断言逐条复核成立**："exp2 换底 A5 无益"——estimate_latency.cc L1267 exp/exp2/log/log2 同列 special_func_throughput 亲验 + intrin_rule_ascend.cc L45–50 同宏注册亲验 + CONST-valu-model measured 补注 + D-ELEM-ACT-FORMULA-PROBE "A5 有直连自然指数 vexp" 四重佐证 ✓；"online 契约空间内非必需"——非"不存在"断言（R2 明言有），分工依据（wrapper 契约边界 + tiled 另任务）亲验 wrapper L186–198 成立 ✓；行内 online #7 弃选量化（1 cmp @51 档 0.02 vs 2K special @8 档 0.125 cycle/elem，K=2 净亏——数字与 estimate_latency.h L58–66 档位亲验一致）+ TRAP-NAIVE-ONLINE-MERGE 角点防护代价（13.6%=条目实测 0.136 亲验）✓；Cube GEMV #8（ND2NZ 62 cycles base 条目亲验 + reduce 已硬件直连非瓶颈）✓；两 kernel #9（读流量 +100%）✓；⑤ 并列候选裁决——fp16-e vs fp32-e 判定裕度落在未实证常数（special dtype 粒度，CONST-valu-model 未区分——亲验条目属实）→ 未纸面拍板，指向 §1.6.4 实验裁决 ✓。D-2 估算复算：GM 8.39MB/(56×100 B/cycle)=1.5K ✓、计算 4.19M×(5/51+1/8)/56=16.7K ✓、比值 ~11× ✓、special 占计算项 0.125/0.223=56% ✓、口径不确定披露（element vs 向量指令粒度，estimate_latency.h L58–70 确未注明——亲验）且两种解释下主选均优的论证成立（VALU 主导→公式已最小；MTE 主导→紧凑 N 流量已最小+流水）✓。附注：fp16-e "计算项 0.72×" 预估在单一口径下不可精确复算（3/51+1/16→0.54×；4/51+1/16→0.63×；不换 special→0.82×）——该项仅为备选预估且 §1.6.4 有阈值化实验裁决兜底，不构成结论依赖，建议 Stage 1 顺手补口径。
- **调研→设计一致性**：§1.6.0 选定 "紧凑驻留+行子批三段式" 实际驱动 §1.4（8 步主路径逐一对应）✓、§3.1（四步公式 = §1.6.1 优化后公式，非源 N_padded 直译形态）✓、§6（persistent+子批 serial+元素级 Parallel，与选定结构一致，无两遍扫描残留）✓；迁移一致性：§0.5 判定输入含 §1.6.0（源算法与调研候选同台）✓，源 7 项优化手段意图承接逐项映射（#1→R2/#2→R2 统一单路径含"对齐快路径免掩码"意图的 pow2 谓词恒真承接说明/#3→保留/#4→保留/#5→等价替换/#6→persistent+子批/#7→persistent）✓，无静默丢弃 ✓；源算法未被替换（公式层恒等）故无额外承接缺口 ✓。
- **数学等价优化完整性**：§1.6.1 对选定算法系统审视（恒等变形/稳定化/CSE/降代换/访存削减/归约合并视角均触及）；采纳 O2/O3 四要素齐备（原式→优化后→等价论证→收益量化：−41% GM 读 @N=300、VRF −50%）✓；O1/O4 必选/确认项定位准确（源已含稳定化；cast 链载入 cast 与 select 融合于同一 if_then_else、输出 1 cast/行——亲核伪代码属实，无冗余 cast）✓；"源公式层已最优"结论有 R1 #2/#3 弃选证据支撑（每行 1 次 log 非热点、换底同档无益）——非空话 ✓。
- **等价性论证正确性（独立推演）**：O2——pad 宽度变化（N_padded−N → TILE−N）下 −inf 同时为 max 幂元与 exp 后 sum 幂元，含 m=−inf 时 exp(NaN) 传播路径两链一致，数学恒等 ✓ + V1 全 shape fp64/fp32 位级（重跑 ulp=0）✓；O3——原位覆写同读-算-写舍入链，无额外舍入步骤 ✓ + V2 位级 ✓；V3 vs golden 1 ulp 树形分组差与 §8.2 容差标准一致（9.5e-8 << 1e-3）✓；V4 源链 vs A5 链 fp16 位级 + NaN 模式一致——**迁移保真的机器证明** ✓；fp16/bf16 舍入影响评估在案（V5 备选守卫 + §9.1 风险 6 bf16 尾数 8 位排除论证，~1.4e-2 预估与 sqrt(树深)×2⁻⁹ 量级相符）✓。
- **向量化替代覆盖完整性**：§1.6.2 十点与 §3.3 伪代码、§6.1 循环表交叉核对**无遗漏**——伪代码全部计算点（masked 载入 #1/reduce_max #2/exp 原位 #3/reduce_sum #4/行级 lse+cast #5/两段写出 #6）与全部结构循环（Persistent #7/serial 子批 #8/TIR if #9/host #10）逐一对应 ✓；§6 无未论证的逐元素标量循环 ✓。
- **不可替代理由充分性**：#5（依赖 reduce 终值的 tile 级标量，SUB 个/子批 vs SUB·TILE 元素，占比 1/TILE=0.02%@4096，且行维已 T.Parallel(SUB) 向量化——理由具体到本算子结构）✓；#8（子批间 x_ub 版本交替的流水顺序依赖 = 双缓冲结构本质，≤16 次迭代非热点）✓；#9（循环变量运行时控制流 + 分支体两侧静态切片 + example_gemm L47 先例，每 tile 1 次）✓——三处均非"实现简单"类借口，类别（tile 级标量/顺序依赖/边界处理）归纳准确 ✓。
- **替代 API 佐证**：全部有 examples + pattern-library measured 双佐证（#1 rmsnorm L56–57/online_softmax L71–74 亲验同构；#2/#4 D-REDUCE-SIMTVF-API measured + per_token L73 亲验；#3 D-ELEM-SIMTVF-AUTOVEC measured + 源码行号亲验；#5 intrin_rule 注册亲验 + online_softmax L169 T.log 在用亲验；#6 per_token L82/L84 亲验）✓；SimtVF 优先、无未论证 SimdVF 降级 ✓。
- **向量化轴与布局决策完整性（§1.6.3）**：规约类必选候选齐全——水平归约（#1 lane→N）与垂直扫描（#2 lane→M）两种 lane 映射**分别枚举** ✓ + 分块跨核（#3）+ 转置布局（#4）；轴质量评分表五项（整除性/尾 lane 浪费率/累加链形态/repack 代价/UB-VRF 影响）量化 ✓；弃选量化理由独立复核：#2 垂直扫描——SUB≤16 行映射 128 lane 利用率 ≤12.5% → ≥8× 劣化，算术自洽 ✓；#3 跨核——驻留已可行（容量不构成约束）+ partial 统计量 GM 往返流量 +50% 起步，违反流量最小化 ✓；#4 转置——**弃选前提亲验**：设计表述 "T.copy transpose 仅特定路径非通用" 与 skill §6.1 原文一致（"Transpose is path-specific… not a general transpose operation for arbitrary memory paths"），通用转置 API 存在性检索未命中后按规则标注 "未文档化假设" ✓，且弃选主依据是量化的 "repack 纯开销+M 轴短无收益"（wl1 M=1024 vs N=4096）不依赖该负向断言 ✓；与 pattern-library 实测条目无矛盾（layout 相关无冲突条目）✓；源码轴选择对照（GPU N 轴 = 输入非结论，独立复核声明）✓；I/O 契约与核内布局变量区分明确（核内 = 紧凑 UB+pow2 fragment 为设计选择而非契约直译）✓。
- **实验裁决模式完整性（§1.6.4）**：判定裕度依赖未实证常数（special dtype 粒度）已识别 ✓；三件套齐备——主选（fp32-e 全设计展开）+ 备选（fp16-e：buffer 形状 fp16 x_frag 64KB、UB 不变、循环/分核/persistent 结构不变、dtype 分派条件 `dtype=="float16"`、bf16 排除有量化理由、精度守卫 V5 实测 4.2e-5 余量 ~100×——结构完整达 Stage 4 可直接实现深度）+ 裁决计划（代表 shape wl1/wl4、do_bench msprof+max_rel 指标、**明确阈值** 0.90× ∧ 5e-3、按 dtype 分派的工厂裁决而非全局二选一、回写路径 CONST-valu-model Tier 0 含 repro）✓；shape/dtype 特化工厂分派可行性已评估 ✓。
- **布局决策与实现一致**：选定轴 N = §3.3 Parallel 内层维（TILE）✓ = §6.1 元素级循环维 ✓；buffer 形状 = §4.3（x_ub (SUB,N) 紧凑/x_frag (SUB,TILE) pow2）✓ = §4.5 预算口径 ✓；无 repack 路径故 §3.3 无转置链 ✓——三方一致。
- **与下游章节一致**：§3.1 四步以 §1.6.1 优化后公式为输入（原位覆写/紧凑 masked 已体现）✓；§6 与 §1.6.2 结论一致（判定向量化的 #1/#3/#5/#6 在 §6.1 均为 Parallel/copy 形态，保留项 #8/#9 为 serial/TIR if）✓。

## 附：建议级问题汇总（不阻塞，供 Stage 1 择机修订 / Stage 3 实现注意）

1. §0.3 threads=128 误记（GPU 源 default_config 实为 256，`TileOPs-fork/.../logsumexp.py` L395；K3 处置结论不受影响）。
2. §3.3 全块路径伪代码补 "block_m % SUB ≠ 0 时末子批" 静态分支（与 §6.4 对齐；或 §4.7 host assert 增补约束），消除显式非 2 幂 block_m 下的越界切片隐患。
3. §3.3 尾块末子批 copy 补目的切片 `x_ub[0:tsub_rem, 0:N]`（先例 online_softmax L142 同构）。
4. §5.4 "末波 ~1.6% 不均衡" 数字修订为波次量化口径（makespan 2 波 vs 理想 64/56 波）或改定性；D-2 均衡假设附注。
5. §4.5 KB/KiB 口径统一（1.49/0.97 比值 vs 128KiB 封顶结论）；§1.6.0 D-2 fp16-e "0.72×" 补估算口径。
6. §0.1 host 语义行引用改指 op 层 `tileops/ops/reduction/logsumexp.py`（现引 wrapper L193–207 为预算逻辑行段）。
7. §0.6 R4/§3.4 per_token 两段写引用行号校正（L82/L84）。

## 检视执行记录

- 检视人: tilelang-design-reviewer（Stage 2）
- 执行日期: 2026-10-16
- 检视方法: skill tilelang-design-review 全流程——亲读 DESIGN.md 878 行全文；亲读源 kernel（single+tiled 两变体）/A5 wrapper/_primitives.py/op 层 _validate/GPU 源仓 threads 契约；Glob/Read 亲验全部 examples、testing、pattern-library（8 条目）、constants、源码（intrin_rule_ascend.cc/estimate_latency.{h,cc}/ascend_simdvf_lower_parallel.cc/simd.py/loop.py/frame.py/reduce_op.py/annotations.py/device.py）与标准文档（backend-glossary/ascend-constraints/core-split-strategy/algorithm-candidates/gpu_to_ascend_adaptation/add-ascend-op SKILL.md）引用；重跑 verify_equiv.py 与 design_calc_check.py 并逐项消费 JSON；等价性/复杂度/SUB 公式/D-2 估算/弃选量化理由独立推演复算；同族先例 online_softmax（63/63 measured）及其 REVIEW.md 判例对照。
