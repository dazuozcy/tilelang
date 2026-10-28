# Elementwise 多输入搬运效率与带宽曲线

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)，front-matter schema 见其 §5）。
>
> **A5 冷启动（2026-10-16）**：本文件自 tilelang-mlir-fork（npuir/910B2C 实测）复制后**内容清零**——原条目绑定 npuir 工具链实测，不适用于 tilelang-fork（ascend/A5，dav-3510）。schema 保留，条目由 A5 任务蒸馏积累。

---

id: D-ELEM-SIMTVF-AUTOVEC
kind: pattern
family: [elementwise, activation]
apis: [T.exp, T.SimtVF, T.Parallel]
dtype: [fp16, fp32]
device: A5/950 (dav-3510)
status: measured
origin_task: pilot-silu-20261016T025344Z（溯源：该任务 Stage 1 探针 history_version/design_probe_simt_exp.py 全量 PASS 记录 + DESIGN.md §1.6.2/§9.2）
toolchain: tilelang-fork dev build（fc32f47c 时期源码）+ CANN 9.2 + bisheng dav-3510 + Ascend950PR_9579，2026-10-16
repro: repro-missing
---

### SimtVF `T.Parallel` 内标准数学 intrinsic 自动向量化实证（激活类直写公式即可）

- `T.Parallel` 循环内的 `tir.*` 名标准数学 intrinsic（`T.exp`/`T.log`/`T.sqrt`/`T.fabs`）由 `ascend_simdvf_lower_parallel` pass 自动向量化为 2048-bit SIMD 指令（`tir.exp` → `simd_vexp()`、除法 → `vdiv`；源码 `src/ascend/transform/ascend_simdvf_lower_parallel.cc:409-412`）；fp16 输入可用。
- 实测精度（2026-10-16，silu `x/(1+exp(-x))` fp16、131072 随机元素）：vs fp32-opmath→fp16 golden max abs diff **3.906e-3**（< fp16 容差 1e-2；硬件 vexp 为 "hw" 精度而非精确舍入，属预期）。
- 结论：激活类算子（sigmoid/gelu/tanh 合成）simt 模式直写公式即可，SimdVF 显式 intrinsic 仅在需要掩码/舍入模式控制时介入。
- fp16 出口边界数值事实（同任务 Stage 3 实测）：次正规输入（silu(±6e-8)≈3e-8 < fp16 最小次正规 5.96e-8）输出下溢 flush 为 ±0，kernel 与 fp32-opmath→fp16 golden 双方同 flush、位级一致（finite diff=0）——设计期边界表按 fp16 出口量化预期值；深饱和区 x < −11.09 时 exp(−x) 上溢 inf → 输出 flush −0.0，最大偏差 1.7e-4（容差内）。

---

id: D-ELEM-ACT-FORMULA-PROBE
kind: pattern
family: [elementwise, activation]
apis: [T.exp, T.SimdVF]
dtype: [fp16, fp32]
device: A5/950 (dav-3510)
status: measured
origin_task: pilot-gemm_silu-20261016T034510Z（溯源：该任务 DESIGN.md §1.6.1 探针表 + history_version/design_probe_silu_form.py；交叉见 pilot-silu-20261016T025344Z DESIGN §1.6.0 R1 否决表）
toolchain: tilelang-fork fc32f47c / CANN 9.2 / bisheng dav-3510 / Ascend950PR_9579，2026-10-16
repro: repro-missing
---

### silu/sigmoid 类激活公式选型实测（A5，直除最优、tanh 形式实测否决）

- 探针口径：最小元素级 kernel（UB 驻留数据 × 64 次迭代 × 128×128 fp32、单 AIV、do_bench msprof），设计期 ~3 分钟裁决公式级选型。
- 实测：直除式 `z/(1+exp(−z))` **215.67µs**（= 乘倒数式 `z·(1/(1+exp(−z)))` 239.23µs 的 **0.902×**，每元素省 1 次乘）；tanh 恒等变形 `0.5z(1+tanh(0.5z))` **388.42µs**（= 直除的 **1.80×** 劣化）——代价模型按 special_func 8 ops/cycle 档预测 −22%，实测方向相反（标量 tanh 的 Ascend lowering 为软件例程而非 SFU 直连，注记见 constants.md CONST-valu-model）。
- 形态约束事实：SimdVF 域无 `vtanh` intrinsic（`tilelang/ascend/language/simd.py` 无定义；标量 tanh 仅 SimtVF 路径，`src/ascend/codegen/intrin_rule_ascend.cc:57` 注册）；`vexpdif` 融合仅 SimdVF+fp32（simd.py:1182-1195），fp16 需双向 cast 净亏；A5 有直连自然指数 vexp，exp2 换底无单位收益。
- 结论：silu/sigmoid 基线 `x/(1+exp(-x))`（1 SFU exp + 1 div + 2 常规 = 4 向量 op/元素）在此类路径已最小；带 GEMM epilogue 时 tanh 形式为负优化。

---

id: D-REDUCE-SIMTVF-API
kind: pattern
family: [reduction, softmax]
apis: [T.reduce_max, T.reduce_sum, T.infinity, T.serial, T.SimtVF, T.if_then_else, T.And]
dtype: [fp16, fp32]
device: A5/950 (dav-3510)
status: measured
origin_task: pilot-online_softmax-20261016T044814Z（溯源：该任务 Stage 3 `kernel.get_kernel_source()` 实测记录 + online_softmax.py `_build_resident`/`_build_streaming`）；logsumexp-harness 2026-10-16 追加证据（两函数 verify_equiv.py V1/V4/V6 + review_probe_tail_reduce.py PROBE_PASS + Stage 3 实测；溯源允许失效）
toolchain: tilelang-fork dev build + CANN 9.x + bisheng dav-3510 + Ascend950PR_9579，2026-10-16；logsumexp 追加证据：tilelang 0.1.15+cuda.git40c9d668 + CANN 9.2.0 + Ascend950PR_9579，2026-10-16；Stage 4 threads 旋钮证据：tilelang 0.1.15+cuda.git73a4b92d + CANN 9.2.0 + Ascend950PR_9579，2026-10-17
repro: repro/PATT-simtvf-reduce-threads-width.py
---

### row-reduction 常用 API 形态实证（SimtVF 内，硬件直连）

- `T.reduce_max` / `T.reduce_sum(dim=1, clear=True)` 在 SimtVF 内对 (BM,TILE) fragment 自动降级为 AscendAllReduce 跨线程树归约（get_kernel_source 实见 `tl::AscendAllReduce<tl::MaxOp,1024,1,0>` 与 `SumOp` 实例）——row-reduction 三段式（reduce_max → exp(x−m) → reduce_sum → 每行倒数 → 原位 cast）直接映射即可，无需 `alloc_reducer`/`reducer_update` 显式 reducer API，fp32 全程硬件直连。
- `T.infinity(T.float32)` 接受 dtype 对象（内部字符串化）；`T.serial(n)` 单参等价 `range(n)`。
- 全 −inf / +inf / NaN 输入行的 softmax「逐元素 NaN 传播」语义在驻留式与流式两路径天然一致（驻留：m=−inf → exp(nan)；流式：哨兵钳位 → s_run=0 → lse=−inf → exp(x+inf)=nan）；golden 用 `torch.softmax(dim=1)` + `assert_close(equal_nan=True)` 对照可全覆盖角点。
- 掩码载入（pad 用 −inf 做 max 幂元、exp(−inf−m)=0 做 sum 幂元）+ 掩码存储（`if i < N:` 谓词 store）组合可同时吸收非 pow2 N 与尾块（example_rmsnorm.py 先例同形态）；与 DMA 紧凑拷贝（x_ub 存 (BM,N) 紧凑、fragment 存 (BM,TILE) pow2）正交。
- **−inf pad 幂元中性（任意 pad 宽度位级成立，logsumexp verify_equiv V1/V4）**：N_padded 对齐 pad（GPU 256 对齐）与 pow2 TILE pad（A5 fragment）两种布局的减 max 稳定化链 fp64/fp32 全部位级一致（ulp=0）——「紧凑 UB + pow2 fragment masked」可替代「GPU N_padded 对齐驻留」，pad 宽度不引入数值差异。
- **online 合并链 chunk 划分不变性（logsumexp verify_equiv V1/V6）**：flash 形式 max 基重缩放对 chunk 划分满足结合律，BLOCK_N pow2 重划分与 tile_n 直译划分在 7 组 shape×精度上 fp64/fp32 位级一致（≤1 ulp；划分敏感性扫描 BLOCK_N 256…16384 × tile_n 16640/25000/32512）。
- **extent < threads 的复制布局合法（尾 chunk reduce 配置，logsumexp reviewer 探针 + 集成形态实测）**：(1,256) fragment @ SimtVF(threads=1024)（replication=4）的 reduce_max/reduce_sum(dim=1, clear=True) 编译运行通过，sum ratio=1.000000（复制 lane 无双重计数）、lse 与 fp64 参考一致；机理：`src/backend/common/op/reduce.h` ReplicateExtent 建模 + CheckAllReduceWidth 要求 scale/logical_width 均 pow2（extent=256、scale=4 满足）+ AllReduceNeedsWorkspace(>32)→ub_reduce 按 scale-stride 折叠复制槽位。threads=min(1024, SUB·BLOCK_N) 类全 chunk 公式可直接复用于尾 chunk，无需为小尾 fragment 收缩 threads 或 pad extent（探针 PROBE_PASS 独立最小 kernel + 完整 kernel L0 首用例 (33,33000) 双重闭合）。
- 行尾 + 列尾联合谓词 `T.And(r < rows, i < N)`（`tilelang.ascend.language` shadow 通用面有导出，tilelang/language/ast/ir.py L57）在 SimtVF masked 载入中正确降级，单一 masked 载入同时消化行尾与列尾——源 GPU kernel 的 T.And 用法可直译。
- **threads 宽度是归约摊薄的一等旋钮（Stage 4 实测 U 曲线，LogSumExpFwdOp 双变体 6 workload）**：`AscendAllReduce<Op, threads, 1, 0>` 的树宽 = `T.SimtVF(threads=)` 参数——默认满宽 1024 并非最优。sweep {1024→512→256→128}：512 改善、**256 全 workload 最优**、128 回退（每行归约树深/同步规模随宽度减半收窄 vs 每线程串行元素数翻倍的两力平衡点）。实测（do_bench msprof，交错 A/B 3×3）：single 4k +16.1/16.3%、3d +25.9%；tiled 32k +10.1%、lm-head +13.0/13.1%（vec busy 88.4%→73.4%、88.6%→84.4%）；128 档全回退（3d −51% 最烈：SUB·TILE/128 = 每线程 512 元素）。窄宽度下小 shape 自动 clamp `min(threads, SUB·TILE)` 不需特判。慢→快关键更改：`threads = min(1024, SUB*TILE)` → `min(256, SUB*TILE)`（repro/PATT-simtvf-reduce-threads-width.py 完整对照，重验 +15.8%）。

---
id: D-LOGSUMEXP-MIGRATION-FIDELITY
kind: pattern
family: [reduction, logsumexp, softmax]
apis: [T.reduce_max, T.reduce_sum, T.exp, T.log, T.copy, T.Persistent, T.SimtVF]
dtype: [fp16, bf16, fp32]
device: A5/950 (dav-3510)
status: measured
origin_task: logsumexp-harness（_logsumexp_kernel_single/_logsumexp_kernel_tiled，2026-10-16；溯源：两函数 Stage 3 L0/L1/L2/Boundary 全 PASS 输出 + verify_equiv.py V1–V7——允许失效）
toolchain: tilelang 0.1.15+cuda.git40c9d668 + Ascend950PR_9579 / CANN 9.2.0，2026-10-16；Stage 4 fp16-e 时延裁决：tilelang 0.1.15+cuda.git73a4b92d + CANN 9.2.0 + Ascend950PR_9579，2026-10-17
repro: repro/TRAP-fp16-fragment-exp-slow.py
---

### logsumexp 族 A5 迁移保真实测（驻留/流式双变体；golden 可直用 torch.logsumexp）

- 单 tile 驻留变体（fp32 中间 + cast 回输出 dtype）与流式变体（BLOCK_N pow2 chunk + 源式 flash 合并 `prev=m_run; m_run=max(m_run,m_c); e=exp(x−m_run) 原位; s_run=s_run·exp(prev−m_run)+s_c` + stats_ub 驻留）对 randn 域 fp16/bf16 输出与 `torch.logsumexp(x.float()).to(dtype)` **位级一致（max_diff=0.000e+00）**，fp32 输出 max_diff ≈ 4.8e-7–9.5e-7（树形分组差，≤1 ulp）——迁移 logsumexp 类算子 golden 可直接用 torch.logsumexp，无需 fp64 参照（NPU 无 double dtype，见 traps-runtime.md TRAP-NPU-DOUBLE-SILENT-FP32）；源链固有角点（全 −inf 行 NaN）与 torch 的分歧见 traps-runtime.md TRAP-SOURCE-VS-TORCH-GOLDEN。
- 用例规模：single 86 用例（L0/L1/L2/Boundary，含 M%block_m≠0、M<block_m、非 pow2 N）+ tiled 61 用例（含尾 chunk extent<threads、M 尾块+子批双触发、fp32 128KB UB 满载）全 PASS；集成侧 TileOPs report 45/45 正确性（avg_max_abs_err 1.47e-07）。
- fp16-e 备选变体精度守卫（Stage 4 A/B 备选，本任务未采纳）：arg fp16 + fp16 树形和 + fp32 log，模拟精度 max_rel 1.4e-5–4.2e-5（1024×4096: 4.211e-05；128×16384: 3.589e-05；1024×32768: 3.6e-5；4×102400: 1.4e-5），@1e-2 违反率 0——守卫余量 ~100×（远优于树深最坏估计 ~6e-3）；**关键差异**：流式变体跨 chunk 重缩放链次数多，s_c 可 fp16 但 s_run 合并必须保 fp32（合并域保 fp32 是 tiled 备选与 single 备选的分界）。
- **fp16-e 备选 Stage 4 时延裁决：50–60x 慢，实测否决（2026-10-17，tilelang 0.1.15+cuda.git73a4b92d + CANN 9.2.0）**：精度合规（实测 max_rel ≤1.8e-3，在 1e-2 guard 内）但 do_bench(msprof) 灾难性回退——single 4k-fp16 52.97→2579.8µs（−4771%）、3d 19.1→350.2µs；tiled lm-head-fp16 20.4→594.2µs、custom (1024,32768,fp16) 111.0→3222.9µs。**机制实证**（生成 AscendC 源对照，profiles/source/*__r1_fp16e.c）：fp16-e 分支与 fp32 基线结构逐字相同（同持久调度、同宽度 AscendAllReduce、同循环嵌套），唯一差异 = 逐元素标量 `hexp`（fp32 为 `expf`）+ half fragment 上的 AllReduce——归因 bisheng 对半精度标量超越函数/半精度归约的软件化 lowering。**bf16 同构对照（fp32 链）三 workload 全 tie（−0.7%/+0.0%/+0.1%）**证实测量隔离。结论：row-reduction 中间链保持 fp32 fragment（cast 进/出各一次），窄 dtype 只用于输入输出 buffer（repro/TRAP-fp16-fragment-exp-slow.py：独立最小 kernel 实测 20.6x，--bench 断言 ≥10x）。

---

id: D-LOGSUMEXP-SIMD-F16MAX
kind: pattern
family: [reduction, softmax, logsumexp]
apis: [T.SimdVF, T.simd.vld, T.simd.vmax, T.simd.vcvt, T.simd.vcmax, T.simd.vdupv]
dtype: [fp16, bf16]
device: A5/950 (dav-3510)
status: measured
origin_task: logsumexp-simd-harness（TileOPs LogSumExpFwdOpSimd Stage 4，2026-10-17；evidence: perf_opt/opt_log.md Round 1/2 + perf_records.jsonl，5 workload +4.0~+15.5%，max_abs=0）
toolchain: tilelang dev build（fc32f47c 后）/ CANN 9.2.0 + bisheng dav-3510 + Ascend950PR_9579，2026-10-17
repro: repro/PATT-simd-f16-domain-max.py
---

### SimdVF 两遍结构 pass-1 在 16-bit 输入域做 max（位级等价的 vec 指令削减）

- **优化点**：两遍 row-reduction（pass1 max + pass2 exp-sum）的 pass1 对 fp16/bf16 输入直接用 128-lane 16-bit 向量（`vld` NORM 256B + `vmax`），仅最终 winner 广播后单次 `vcvt` 升 fp32——pass1 vec 指令 3 ops/64 elems → 2 ops/128 elems（÷3），UB 读流量不变。**max 无舍入 → 16-bit 域 max 与 fp32 detour 选出同一输入值，位级等价**（5 workload max_abs=0）。
- **bf16 例外**：`asc_reduce_max` 有 `vector_half` 重载、**无 bfloat16 重载**（CANN 9.2.0 `reg_reduce.h` L28-38）——bf16 的跨 lane max 须 EVEN/ODD 拆分到 fp32 归约：`vdupv(vcmax(vmax(vcvt(acc,f32,EVEN), vcvt(acc,f32,ODD))))`（同样精确）。`vmax`/`vlds NORM` 对 half/bf16 128-lane 均支持（`asc_max_impl.h`）。
- **bf16/fp16 宽化装载位级实证（`vld(dist="UNPK_B16")+vcvt("float32",part=0)`，2026-10-17）**：生成 `vlds_unpack<half>`+`vcvt<float>` 链，fp16/bf16 全用例 max_diff=0（L0 3-dtype + L1 42 组扫形 + Boundary 角点），fp32 直用 `vld()` NORM 时 ≤9.537e-7——补齐 bf16→f32 宽化直接位级证据（此前仅 bf16→e4m3 链中间级组合覆盖；`test_tilelang_ascend_simdvf_cast.py` 无 bf16→fp32 直接用例，同族设计期该缺口可按本条关闭）。16-bit 单点回存 `vsts(...,ONEPT_B16)`、fp32 用 `ONEPT_B32`（mask `pset(16/32,"PAT_VL1")`）。
- **同轮反模式实证（refuted，勿叠加）**：① 组循环步进展开（`for g in range(0,G-u,4)` 内联 4 组）实测 **−4.5~−14%**——深依赖链破坏软件流水；② pass2 EVEN/ODD 配对（1 次 128-lane NORM 载入 + 2 次 vcvt）实测 **−4.7~−17%**——同因；③ 显式提升 b16 谓词掩码（每迭代 `asc_create_mask_b16` 重建）**无效果**（+0.0~0.3%）——掩码非 scalar 瓶颈主项。三者均为「减少迭代数/指令数但不改善关键路径」的伪优化。
- 慢→快关键更改（delta 见 repro）：pass1 循环体 `vld(UNPK_B16)+vcvt(EVEN)+vmax(f32)` ×G → `vld(NORM)+vmax(half)` ×(N_pad/128) + winner 单次 `vcvt`。

---

id: D-SIMD-PERSISTENT-ROWS-LSE-BATCH
kind: pattern
family: [reduction, logsumexp]
apis: [T.Persistent, T.alloc_shared, T.copy]
dtype: [fp16, bf16, fp32]
device: A5/950 (dav-3510)
status: measured
origin_task: logsumexp-simd-harness（同上，Round 1/2；attn-32k +30.1% 累计）
toolchain: tilelang dev build（fc32f47c 后）/ CANN 9.2.0 + bisheng dav-3510 + Ascend950PR_9579，2026-10-17
repro: repro/PATT-persistent-rows-overhead.py
---

### SimdVF 流式变体每 tile 行数 ROWS：lse 微写批量化 + M≤56 保并行策略

- block_m=1 × M 行的持久化流式 kernel 每 tile 经 MTE3 写 1 个 lse 标量（M 次 2 字节微写，aiv_mte3 64.3% busy）；kernel 内 ROWS 行/tile（`lse_ub (ROWS,)` 尾部单次 `T.copy` 写回）把 MTE3 次数除以 ROWS。**策略（实测）**：M ≤ 56（单波域）→ ROWS=1（保核并行，lm-head M=4 若 ROWS>1 会把 4 核砍到 <4）；M > 56 → makespan `ceil(ceil(M/ROWS)/56)·ROWS ≤ min+1` 内最大 pow2、cap 8（4 与 8 tie）。慢→快关键更改 = `block_m` 改为 `_pick_rows(M)` 派生 + `lse_ub` 按 ROWS 定形（host 静态化重推导，kernel 体不变）。
- 实测：M=1024 ROWS∈{2,4,8} → +16.4/+25.8/+26.6%（mte3 64.3→6.2%，mte2 升至 85.7% 逼近搬运墙）；与 P-LAYOUT-ROW-RESIDENCY 驻留变体 ROWS 规则（per-tile 开销主导、makespan+1 内最大 pow2）同构，流式变体额外多一个 lse-MTE3 摊薄项。
