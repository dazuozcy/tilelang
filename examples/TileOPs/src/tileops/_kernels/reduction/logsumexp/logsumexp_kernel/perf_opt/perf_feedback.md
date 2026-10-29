# 性能反馈（[DESIGN_LIMIT]）

## 触发判定

- 设计层归因: `examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp_kernel/_logsumexp_kernel_tiled_DESIGN.md` §1.6.0 R3/R4 算法族选定——「多 kernel partial+merge」候选经 **R1 #10 弃选**（弃选理由原文：「partial 往返 + 同步；**行块并行度已饱和 56 核，chunk 串行非瓶颈**」），选定单 kernel BLOCK_N pow2 chunk 流式结构（行块维并行、chunk 维核内串行）；叠加 §0.2 冻结的 kernel 签名 `(x, y)`（无 GM workspace 形参，跨核 N-split 的 partial 交换无落点）。lm-head-logits（M=4, N=102400）的 4/56 核激活由该两项设计决策直接决定——非 tiling/参数可解（threads=256 已采，BLOCK_N 双向 sweep 已闭，block_m 无法从 M=4 造出 >4 个行块任务）。
- 结构性加速估计: **~2.5x**（依据: 实测外推——32k workload（1024×32768）实测每 chunk-iteration ≈ 3,884 core-cycle〔7,955,325 aggregate AIV cycles / 2,048 chunk-iteration，`examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp_kernel/perf_opt/profiles/round2/_logsumexp_kernel_tiled/attn-weights-32k-bfloat16_merged_final/msprof_detail.json`〕；lm-head 为 4 核 × 7 chunk 串行链 = 28 chunk-iteration 的 163K aggregate cycle 单波；N-split 至 28 核（4 行 × 7 split）后计算段 ≈ 1 个 chunk-iteration ≈ 2µs 量级 + GM partial 有序合并段（28×8B 往返 + flag 同步，参照 example_gemm_splitk deterministic 模式结构）≈ 合计 5-7µs vs 实测 17.7µs）。**且实测与设计假设直接矛盾**: R1 #10 弃选前提「行块并行度已饱和 56 核」在 lm-head workload 上实测为 **4/56 核（7%）**——该前提仅对 M≥56 的 workload 成立，M=4 的契约 shape（lm-head-logits，TileOPs benchmark 在集案例）落在前提失效区。

## 参照锚定

- 参照实现: `examples/ascend/example_gemm_splitk.py`（本仓实测先例：split-K 跨核 partial + GM 合并，含 `T.set_atomic("add", "float32")` 原子模式与 deterministic 有序归并双模式）+ `examples/ascend/test_gemm_splitk.py`；pattern-library `cases.md` CASE-pilot-online-softmax / CASE-logsumexp-harness（row-reduction 单 kernel 流式族先例，无跨核 split 形态）。
- 不可移植论证: ① gemm_splitk 的合并算子是**加法**（`C += partial`，atomic add 直达），logsumexp 的 flash 合并是 **(max, sum) 二元组结合合并**（`m=max_k m_k; s=Σ_k s_k·exp(m_k−m)`）——additive atomic 不可表达 max，只能走 deterministic 有序归并（publish + 按序 read-modify-write）或 designated-merger 模式（文档条款依据：`tilelang/ascend/language/__init__.py` atomic 原语仅 add 类；非实测否决的部分标注为未实证假设：ordered (max,sum) GM 合并的正确性协议需 Stage 1 设计）；② 两种模式都需要 **GM partial workspace**（M×splits×8B），而 §0.2 冻结签名 `(x, y)` 无 workspace 形参、wrapper 契约亦不提供——kernel 内无 GM scratch 可用（UB 为核内私有，跨核不可见），复用 x 作 scratch 违反输入不变式；③ wrapper 层双 kernel（partial kernel + merge kernel）方案被 conductor 约束排除（Stage 4 不得改 wrapper/op 层分派）。结论：参照结构**存在且已实测**（gemm_splitk），不可移植点是 DESIGN 层签名契约与合并算子差异，非工具链能力缺口。

## 假设

`examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp_kernel/_logsumexp_kernel_tiled_DESIGN.md`：
- §1.6.0 R3 候选表「多 kernel partial+merge」行 + R4 硬件亲和性表「❌ 弃选（R1 #10：partial 往返 + 同步；**行块并行度已饱和 56 核，chunk 串行非瓶颈**）」——弃选前提按 M≥56 的代表 workload（wl2 M=1024）成立，未对 M≪56 的契约 shape（wl3 lm-head M=4）做分域检验。
- §0.2 kernel 签名契约 `(M, N, dtype, tile_n)(block_m)(x) → y`——无 workspace 形参（继承源 GPU 算子签名）。
- §1.6.0 D-2「wl3 并行度注记：M=4 → num_tiles=4 < 56（仅 4 核活跃）——并行度受限是 workload 特性（M 小）而非算法结构问题，Stage 4 记录」——设计已预告该现象但将其归为 workload 特性；实测（下节）表明该现象正是 R1 #10 前提失效的实证，且参数空间内无解。

## 实测

（全部 do_bench(backend="msprof") 口径，µs = ms×1000；深挖 `backend="msprof_detail"`，归档 `examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp_kernel/perf_opt/profiles/round2/`）

| 指标 | lm-head-logits-fp16 | lm-head-logits-bf16 | 对照 attn-32k-bf16（M=1024） |
|---|---:|---:|---:|
| baseline（round 0 记录） | 20.334 | 20.349 | 110.626 |
| final（phase=final 记录，候选 `final` = OP1/OP6 合并版） | 17.757 | 17.714 | 100.226 |
| 独立复测 pooled 中位数（5 run × 5 sample） | 17.685（散布 0.53%） | 17.786（散布 0.86%） | 100.430（散布 0.11%） |
| msprof_detail（merged_final 归档） dur / aiv_total_cycles | 17.681µs / **162,959** | 17.56µs / — | 101.197µs / 7,955,325 |
| pipe ratios（vec/scalar/mte2/mte3） | **51.7% / 32.0% / 16.6% / 1.0%** | 47.6% / — / — / — | 82.1% / 2.2% / 29.2% / 47.7% |

- **4/56 核激活的量化证据**：lm-head aggregate AIV 速率 ≈ 162,959 cycle / 17.681µs ≈ 9.2 GHz，对照 32k 的 7,955,325 / 101.197 ≈ 78.6 GHz（≈56 核满载）——lm-head 仅 ~4 核有效运转（7%），与 M=4 → 4 tiles 的行块映射一致。
- **chunk 串行链归因**：lm-head = 4 行 × 7 chunk 串行 = 28 chunk-iteration，163K cycle ≈ 5.8K cycle/chunk（含 per-row stats/lse 与 launch 摊薄，对照 32k 的 3.9K cycle/chunk 满摊薄值）；vec 51.7% + scalar 32.0% 双低占用证实「核数不足 + 每 chunk 开销」叠加，核内已无参数空间（threads=256 已采 +13.1/13.3%，BLOCK_N∈{8192,16384,32768} 实测 16384 局部最优〔E6/E2〕，fp16-e 链实测否决〔E4，−2808%〕）。
- 调优内已穷尽的参数/结构候选：E2/E6（BLOCK_N 双向）、E4（fp16-e）、E7/E9/E11（threads sweep，256 采纳）、merged/final 复测（全 6 tune workload 无回退）。

## 影响面

- 算子族/shape 范围：reduction 族 row-wise streaming 变体在 **M < 56 且 N 超大**的契约 shape（lm-head-logits M=4/N=102400 为在集 benchmark 案例；解码 batch 场景 M=1~8 常见）上的并行度天花板——单 kernel 行块并行结构下核数 = M，端到端 ≈ 单核 chunk 串行链。
- 改设计预期影响：Stage 1（tiled 变体算法族重选：对 M≪56 域补充「跨核 N-split partial+merge」候选，按 gemm_splitk 先例设计 (max,sum) 有序合并协议 + workspace 签名扩展或 op 层双 kernel 分派）+ Stage 3（tiled 变体重实现）+ wrapper 签名/分派修订；M≥56 域（wl2 类）结构不变。预估收益 ~2.5x（lm-head 17.7µs → ~5-7µs），风险为合并协议的数值一致性（须保 V1/V5 等价角点）与 flag 同步开销。
- 不影响：single 变体（N 驻留域）、M≥56 的 tiled workload（已 56 核饱和，R1 #10 前提成立域）。

## 反馈结论

lm-head-logits（M=4）的 ~2.5x 结构性加速空间被 DESIGN §1.6.0 R1 #10 的「行块并行度已饱和 56 核」弃选前提与 §0.2 冻结签名（无 workspace）共同锁死——该前提经 do_bench(msprof)/msprof_detail 实测在 M≪56 域失效（4/56 核、9.2GHz vs 78.6GHz），属设计层（算法族选定 + 签名契约）天花板而非调优参数空间可解。

建议路由: 附录补记
