# _route_stats 算子设计文档（IndexedExpertMLPFwdOp 路由统计 kernel，TileOPs 迁移）

> **任务标识**：project = `indexed_expert_mlp`，op = `_route_stats`（`@tilelang.jit` 工厂函数名，保持不变），scenario = migration（TileOPs harness，family=moe）。上层算子 `IndexedExpertMLPFwdOp`（indexed/grouped 融合 MoE 专家 MLP 前向，DeepSeek-V3 / Qwen3-235B 规模）。整体算子编程模式为 mixed（GEMM 主导 kernel 走 Cube）；本函数是**向量型路由统计 kernel**，按用户指令以 **SimdVF（simd）** 设计。
>
> **源码版本**：GPU 仓 `/home/l00970450/z00875089/TileOPs` @ `3979278e2da6ac4fa84d8c7c800dec64ea9c0425`（clean）。Stage 0 转换记录：`examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/conversion_record.md`；提取参考（逐字）：`examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/_indexed_expert_m_l_p_fwd_kernels.py`。
>
> **修订标识**：revision_index = **1**（2026-10-29）。§0 与 §1–§10 基础结构承袭 design_v0（`history_version/design_v0.md`）；本次修订由 Stage 2 检视不通过触发（REVIEW.md：维度 8 阻塞级问题 1「负向 API 断言被源码推翻」+ 3 项建议级），核心变更 = 事实层修正（`T.simd.vgatherb/vgather2/vscatter` 原语族在位）+ 扫描主体向量化判定重推（依据换真）+ 新增 §1.6.4 实验裁决三件套 + E5 等价性验证。逐项见下节。

## 修订说明（相对 design_v0 的关键调整）⭐

| # | v0 的错误 / 不足（REVIEW.md 定位） | 本版关键调整 | 为何不会再犯同一错误 |
|---|-----------------------------------|--------------|---------------------|
| 1 | 断言「`tilelang/ascend/language/simd.py` 无 per-lane 动态索引 gather / 跨 lane scatter 原语」（§1.6.0 R1 行 6、§1.6.2 #3(b)、§1.6.3 #2/#3、调研结论「gather 墙」、§0.6 R1 连带淘汰）——被源码推翻：`vgatherb`（simd.py L1144）/ `vgather2`（L1159，"Gather elements from base using per-lane offsets"）/ `vscatter`（L1184，"base[index[lane]] = src[lane]"）均存在 | 事实层全面修正：如实记载 `T.simd.vgatherb/vgather2/vscatter` 及关联原语 `vselr`（L1452）/`vci`（L1038）/`vusqz`（L1029）/`vsqz`（L1022），五重佐证（simd.py 行号 + `__init__.py:69` 导出 + codegen_ascend.cc:2294/2468–2480 + 真实用例 ×3：`examples/ascend/deepgemm/kernels/transform_sf.py（L286）`、`testing/ascend/language/test_tilelang_ascend_buffer_reuse.py（L270）`、`testing/ascend/auto_schedule/test_access_footprints.py（L527）`）——v0 引为标量先例的同一文件 transform_sf.py 恰含 vgather2 用例 | **查证纪律修正**：负向 API 断言一律先对原语族关键词机械 grep（`grep -n -E "gather\|scatter\|sort\|prefix\|scan\|hist\|select" tilelang/ascend/language/*.py`），命中函数逐一核对 docstring 语义后再下结论。本次修订已按该流程重跑（命中清单见 §1.6.0 查证纪律注记）；「无排序原语」断言经同一 grep 确认成立（sort/argsort 零命中） |
| 2 | §1.6.2 #3 扫描主体「不可向量化」判定的理由 (b) 建立在假前提上（review 判定：允许维持结论，但依据必须换真） | 在修正后 API 事实上重推：给出批式分段扫描变体结构草案 + lane-op/cycle 量化（§1.6.2 #3 + §1.6.4 完整结构）——批内秩需 Θ(B)=63 步 vselr 位移比较链（**log 步草图不成立**，反例 [7,5,5,7]：非相邻同值组的块摘要无法回答任意值计数查询）；备选段周期估算 ≈4.5–7.5K cyc vs 主选标量链 2.6–7.7K cyc **区间重叠、无纸面优势** → 主选维持标量（依据换真），备选进实验裁决 | 判定不再依赖任何「API 缺失」负向断言，改为区间量化 + 未实测常数显式披露 + 裁决阈值在位；Stage 4 A/B 实测定夺（§1.6.4 裁决计划） |
| 3 | §1.6.3 #2/#3 弃选行含「gather 墙（API 缺失）」「跨 lane scatter 无原语」不实分句 | #2 改为量化口径（估算区间重叠 + 复杂度 vs 收益上限 ≤4% op）并保留为 §1.6.4 实验裁决备选（结构齐备 + E5 语义机器验证）；#3 保留 E×R≈131K lane-op ~30× 计算量劣化主理由、删除 scatter 缺失分句（原语在位，既不构成障碍也不构成收益） | 弃选理由全部换成可复算数字；「API 缺失」类论断从弃选依据中清除 |
| 4 | 判定裕度落在 vgather2/vscatter/vselr 吞吐未实证区间（pattern-library constants.md 无对应实测条目，已核对），v0 未按实验裁决模式处理 | 新增 **§1.6.4 实验裁决三件套**：① 主选（标量 scan，进 Stage 3）+ ② 备选完整结构（批式向量变体：buffer/UB 预算/循环结构/dtype 路径/分核三要素按新粒度重算）+ ③ 裁决计划（代表 shape ds-v3-t64、指标 do_bench(msprof)+npusim、未知常数实测/反解方法、翻转阈值〔kernel 级 >20% 且 op 级 ≥1% 同时满足〕、shape 分派评估、回写路径） | 三件套随 DESIGN.md 冻结进入 Stage 4 的 A/B 清单；阈值与全部数字在位，Stage 4 可直接执行 |
| 5 | （建议级，问题 2）D-2 行「T=32 时占比 ≈2× 恶化至 ≈4–5%」无法由自身代价模型复现（scan 与 Cube 段同 ∝ T·K） | 改为「占比随 T 近似不变，T=32 复算 ≈2.6–3.9%」；「<4%」改「≈2.5–4%（上界 8K/197K=4.1%）」；另披露标量链流水乐观界 ~5 cyc/iter → ≈2.6K（§1.6.4 裁决区间下端） | 占比口径统一为可复算公式，两 shape 复算入文 |
| 6 | （建议级，问题 3）§8.1 golden 草案设备放置未声明（CPU 默认 vs kernel 输出在 NPU） | 补设备约定：golden 张量建在 `expert_ids.device`（AGENTS.md golden-on-NPU 惯例）；int32 逐位相等方式下与「对照前统一 `.cpu()`」等价，全链路统一即可 | 消除 Stage 3 设备不匹配报错面 |
| 7 | （建议级，问题 4）UB→GM「字节粒度尾区」引用方向偏窄（SKILL.md §6.1 原文针对 GM→UB；非 32B 尾区无直接先例） | §5.3/§9.2 标注「UB→GM 非 32B 尾区：文档未单列，L0 编译冒烟首验项」+ fallback（SIZE 向下 32B 对齐段 bulk + 尾段 ≤7 元素标量小段拷贝） | 首验项 + fallback 双保险，Stage 3 命中即有出路 |
| 8 | 等价性机器验证 E1–E4（v0 已全过，检视重跑确认） | 新增 **E5**：§1.6.4 备选变体（批式分段扫描 + last-occurrence counts 回写）的数学等价性模拟对照——**14 组模式（12 共享 + 2 批边界压力）bitwise 一致，EQUIV_PASS**（2026-10-29 重跑） | 备选变体的语义保持论证由散文升级为机器验证（D-1 纪律）；备选若在 Stage 4 启用，其语义正确性已有设计期证据 |

**保持不变的结论**（检视确认 pass 的维度，承重结构不动）：选定算法族（确定性稳定序单遍扫描 + UB 全量 staging + 单次 bulk 搬出）、编程模式（simd + `auto_schedule: yes`）、API 映射（§3）、内存规划（§4）、Tiling/分核（§5）、循环/同步（§6/§7）、验证方案主体（§8）、OPT-1/2/3 数学等价优化（E1–E4 重跑仍全过）、§0.1–§0.5 源算子解读与耦合性判定。

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 tilelang-op-design skill 的 `references/migration-analysis.md`）：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§10 的所有设计决策。解读对象为源文件 `src/tileops/_kernels/moe/indexed_expert_gemm.py`（GPU 仓）的 `_route_stats` 工厂（L36–63）、`_route_metadata_size` helper（L30–32）、`IndexedRouteStatsKernel` wrapper（L255–300），以及 host 侧调用点 `src/tileops/_ops/moe/routed_expert/indexed_routed_expert.py` 的 `IndexedExpertMLPFwdOp.forward`（L122–126）与 `src/tileops/_kernels/moe/call_spec.py` 的 `IndexedRouteStatsFwdInterface`（L409 起，`output_size` 契约）。

### 0.1 源算子语义（做什么）

**数学语义**：给定 `expert_ids[T, K]`（int32，每 token 的 K 个路由专家 id，调用方契约保证取值在 `[0, E)` 内——manifest `requires: ["in_range(0, E)"]`，kernel 不做值校验，与源一致），在 int32 元数据缓冲 `metadata[E + R + E·T]`（`R = T·K`）上原位写入三段路由描述：

- **counts 段** `metadata[0:E]`：`counts[e]` = 路由到专家 e 的 route 总数（每 route 恰贡献 1）。计数与处理顺序无关，**精确确定**。
- **ranks 段** `metadata[E : E+R]`：`rank[route]` = 该 route 在其专家内部的到达序号。语义要求仅为：**对每个专家 e，其全部 route 的 rank 构成 `[0, counts[e])` 的一个双射**（源用原子 fetch-add 实现，具体双射取决于线程竞态顺序，**顺序是实现自由度**，见下）。
- **lists 段** `metadata[E+R : E+R+E·T]`：`list[e·T + rank] = route`（当且仅当 `rank < T`；每专家列表只有 T 个槽位，`rank ≥ T` 的重复 route 不入列表，下游按"独立计算"回退处理——即测试中的 duplicate-fallback 分支）。

**双射自由度与消费不变性（本迁移的核心语义结论）**：下游消费者 `_indexed_expert_gemm`（dispatch_mode="grouped"，block_m=16）只要求三段**内部自洽**：`counts` 精确；`ranks` 为每专家双射；`lists` 与 `ranks` 配对。任意满足自洽性的双射选择，最终 MoE 前向输出**逐位相同**——因为每条 route 的 GEMM 结果行 = 该 route 自身激活行与该 route 专家权重的点积，K 维累加序由 `ceildiv(k, block_k)` 的 K 循环决定、与 route 无关（route 落在 16 行分组的哪个 m 槽位不改变该行的 K 累加顺序）；分组只改变"哪个 leader 块计算哪批行"，不改变每行的数学内容。该不变性已由 `examples/indexed_expert_mlp/_route_stats/verify_equiv.py` 的 E3 检查机器验证（稳定序双射 vs 随机双射 → counts / 分组划分 / fp64 消费模拟输出全部 bitwise 一致，12 组路由模式含全部移植测试模式）。

**规约语义**：无浮点规约；整数计数（每 route +1），累加顺序不影响结果（整数加法交换结合）。
**dtype 语义**：全 int32；中间无 cast；计数上限 R = T·K 远小于 2^31，rank/route 同理，无溢出风险（E4 已验证 int32 值域）。
**边界语义**：(a) `rank ≥ T` 的 route 跳过 lists 写入（duplicate-fallback）；(b) 零 route 的专家 counts=0（源显式清零 [0,E) 段保证）；(c) lists 段中未被写入的槽位在源实现中保持 `torch.empty` 的**未初始化垃圾值**——下游只读已写槽（E2 已机器验证消费读集 ⊆ 写集），故垃圾值不可观察；(d) expert_ids 越界（e ≥ E 或负值）为未定义行为（调用方契约禁止，与源一致）。
**确定性**：源实现的唯一非确定性 = 双射选择顺序；由上述不变性，其**不影响任何可观察输出**。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现（规范实现 = 路由顺序稳定双射——双射自由度下唯一无需硬件时序即可定义的规范选择），并附性质断言（不依赖稳定序选择的 §0.1 契约校验）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| expert_ids | (T, K) | int32 | 每 token 的 top-k 专家 id；调用方契约取值 ∈ [0, E) |
| metadata | (E + T·K + E·T,) | int32 | **原位写回**的路由元数据（out 模式，`out_idx=[]`，wrapper 返回传入的 out）；输出 shape 依据：`_route_metadata_size(num_experts, num_routes, num_tokens) = E + R + E·T`（源文件 L30–32）+ wrapper `forward` 的 `(size,) int32` 校验（L292–294） |

- 契约来源：`IndexedRouteStatsFwdInterface.output_size(call)`（call_spec.py）与 A5 wrapper `IndexedRouteStatsKernel.forward`（`examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/indexed_expert_mlp.py` L134–142，校验逻辑逐字保留）。
- A5 wrapper 调用面（冻结，不改）：`self.kernel(**self.config)(expert_ids, out)`，`config = {}`（threads 已按 K3/K9 移除）；工厂签名 `_route_stats(num_tokens, top_k, num_experts)` 返回无参 callable。
- 上层分支谓词：`grouped_dispatch = (num_tokens > 1)`（conversion_record.md §3）——T=1 时不运行本 kernel（direct 分派，metadata=None）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每个计算语句归入某步骤，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `for expert in T.Parallel(E): metadata[expert] = 0`（L51–52） | — | counts 段清零 | counts 段（含零 route 专家的 0） |
| 2 | `T.sync_threads()`（L53） | — | — | 步骤 1→3 的块内屏障（清零与原子读改写之间的 RAW/WAR 依赖） |
| 3 | `for route in T.Parallel(R): expert = expert_ids[route // top_k, route % top_k]`（L54–55） | expert_ids (GM) | 每 route 的专家 id e | 路由读取（div/mod 索引） |
| 4 | `rank = T.atomic_add(metadata[expert], 1, return_prev=True)`（L56） | counts[e] (GM) | rank + counts[e]++ | 每专家 running 计数（在线）+ 双射序号分配 |
| 5 | `metadata[rank_base + route] = rank`（L57，rank_base=E） | rank | ranks 段 | ranks 段 |
| 6 | `if rank < num_tokens: metadata[list_base + expert·T + rank] = route`（L58–59，list_base=E+R） | rank, e | lists 段 | lists 段（T 槽位守卫 / duplicate-fallback） |

**数据流与内存访问模式**（源硬件 = CUDA SM 视角）：

```
GM[expert_ids] --逐线程标量 load（div/mod 寻址）--> 寄存器
GM[metadata] <--并行标量 store（步骤1 清零）--
GM[metadata[expert]] <--原子 fetch-add RMW（步骤4）--（读改写直落 GM，无 staging）
GM[metadata] <--分散标量 store（步骤5/6，4B 粒度 scatter）--
```

无 SMEM / 无 fragment / 无分块：全 kernel 直接在 GM 上以标量粒度读写，单 block（`T.Kernel(1, threads=128)`）。

**循环与并行结构**：grid = 1 block × threads = 128（default_config `{"threads": 128}`）；步骤 1 的 `T.Parallel(E)` 把 E 个清零分布到 128 线程；步骤 3–6 的 `T.Parallel(R)` 把 R 条 route 分布到 128 线程；两段之间 `T.sync_threads()` 显式屏障；核内串行边界（E、R）为工厂闭包烘焙的编译期常量。

**host 侧逻辑**：`functools.lru_cache(maxsize=32)` 按 `(num_tokens, top_k, num_experts)` 特化缓存工厂；`@tilelang.jit(out_idx=[])`（原位写、无返回值）；wrapper `forward` 校验 shape/dtype 后调 `self.program(self._compile)(expert_ids, out)`（A5 版为 `self.kernel()(expert_ids, out)`）；Op 侧 `metadata = torch.empty(size, int32)` 后先于两个 GEMM 调用本 kernel（indexed_routed_expert.py L122–126）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | 单 kernel 单遍融合（清零 + 计数 + 秩 + 列表一次 launch 完成） | 消额外 kernel / host 往返 | 一个 prim_func 内三段 | 无（launch 结构） | 可移植（结构保留） |
| 2 | 单遍在线 running-counter | 单遍扫描出全部三段，无预扫描 | fetch-add 返回前值作 rank | 无（纯算法层） | 可移植（保留） |
| 3 | 原子 fetch-add 并行秩分配 | 128 线程并行分配双射序号，无需排序/归并 pass | `T.atomic_add(..., return_prev=True)` | GPU 全局内存原子（硬件 fetch-add） | 硬件强相关（A5 有等价原语，见 §0.5 条目 3） |
| 4 | `T.Parallel` 线程域两级并行（block×thread） | 清零与 route 处理的线程级并行 | SIMT 线程分布 | CUDA SIMT 线程模型 | 硬件强相关（A5 一维 grid 无 threads 参数） |
| 5 | `T.sync_threads()` 块内屏障 | 清零段→原子段的跨线程顺序 | CUDA block 级 barrier | CUDA 块同步原语 | 硬件强相关（A5 SimtVF 自动 asc_syncthreads，见 §0.5 条目 5） |
| 6 | GM 原位直写（无 staging、只写触及槽位） | 最小 GM 写量（≤ E+2R 个 4B 槽） | 分散标量 scatter store | GPU 标量 GM store + 写合并 | 硬件相关（A5 上分散标量 GM store 无 MTE bulk 优势，见 §0.5 条目 6） |

> 注：R = T·K ≤ 512（harness 全部用例），本 kernel 在源平台上即为轻量元数据 kernel；上述手段的意图是"并行摊薄 µs 级延迟 + 最小写量"，不是算力热点。迁移承接判断以 §1.6.0 的定量口径为准。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级（语义/算法/优化） | 源硬件依赖 | NPU 有等价能力？ | 处置（保留/等价替换/重新设计/舍弃） | NPU 对应方案 / 依据 |
|------|----------------------|-----------|-----------------|-----------------------------------|---------------------|
| 1. metadata 三段布局 + 消费契约（counts 精确 / ranks 双射 / lists 配对 / rank<T 守卫） | 语义 | 无 | — | 保留 | 语义层无条件保留；消费方 `_indexed_expert_gemm` grouped 分支（本仓同批迁移）按同一契约消费 |
| 2. 单遍在线 running-counter 算法结构 | 算法 | 无（纯算法层） | — | 保留 | migration-analysis.md §5.3「online 算法 / 分块累加 → 保留」；迁移后仍为单遍（§1.6.0 R2） |
| 3. 原子 fetch-add 并行秩分配（§0.4 #3） | 优化 | GPU 全局原子 | **有**：Ascend 方言 `T.atomic_add(dst, val, return_prev=True)` 在 SimtVF 内对 GM 值可用——`testing/ascend/language/simtvf/test_simtvf_atomics.py`（add/max/min × return_prev 回归）；codegen 降为 `asc_atomic_add`（src/ascend/codegen/codegen_ascend.cc:2526–2561）；int32 GM 重载存在于 CANN 9.2.0 AscendC 头文件 `/home/z00875089/Ascend/cann-9.2.0/x86_64-linux/ascendc/include/basic_api/impl/simt_api/device_atomic_functions_impl.h:168`（`asc_atomic_add(__gm__ int32_t*, int32_t)`，返回前值）。注意：本仓回归仅覆盖 fp32，int32 原子未在 `testing/ascend/` 出现（查证：grep atomic_add 于 examples/ascend 与 testing/ascend 仅命中 test_simtvf_atomics.py 与 set_atomic DMA 路径） | **重新设计**（不用原子路径） | → §0.6 R1：以确定性顺序扫描替代。弃用理由不是 API 缺失（等价原语存在，已举证），而是：(a) 原子序双射非确定 → 元数据 golden 只能性质对照、不能精确对照；(b) 已知能力缺口 CG-2026-0002 的底层机制（SimtVF `T.Parallel` 逐线程标量化）提示该形态在 SimtVF 下落入低效/未覆盖区（该缺口本体为 16-bit 域归约，本算子 int32 不直接命中——已核对 `.agents/evolution/capability-gaps.md` CG-2026-0002 条目，此处仅作风险背景）；(c) R ≤ 512 时并行收益为 µs 级且占比 ≈2.5–4%（§1.6.0 R3/D-2 定量），确定性收益更大 |
| 4. `T.Parallel` + threads=128 两级并行（§0.4 #4） | 优化 | CUDA SIMT | 无（本项目 `T.Kernel` 一维、无 `threads=` 参数，传入即 TypeError） | **重新设计** | → §0.6 R2：`T.Kernel(1)` 单 block + `T.SimdVF()` 向量域（tilelang-ascend skill §2.1/§2.3；migration-analysis.md §5.3「block × thread 两级并行 → 重新设计」）。threads 已从 wrapper/default_config 移除（conversion_record.md §1 K3/K9），线程域由重实现内部声明——本设计判定**无需线程域**（scan 顺序依赖，见 §2.2） |
| 5. `T.sync_threads()` 显式块内屏障（§0.4 #5） | 优化 | CUDA barrier | 部分（SimtVF 跨线程数据冒险自动插 `asc_syncthreads`——tilelang-ascend skill §2.2，回归 `testing/ascend/transform/test_ascend_thread_sync.py`；但 GM 缓冲清零→原子段的跨线程 RAW 检测覆盖性无文档化保证） | **舍弃**（被结构消除） | 重新设计后核内无线程级共享（单执行流：DMA→fill→scan→DMA 链），跨 pipe 依赖由 AutoSchedule 自动同步承接（auto_schedule: yes，§7）；不再需要任何显式屏障 |
| 6. GM 原位分散直写（§0.4 #6） | 优化 | GPU 标量 store + 写合并 | 有更优路径：UB staging + 单次 MTE3 bulk 搬出（`T.copy` UB→GM 连续区，115 B/cycle 档，`CONST-mte-bandwidth-model`） | **等价替换** | → §0.6 R1 内的 staging 方案：GM 写路径从 ≤ E+2R 个 4B 分散标量 store 换为 1 次连续 bulk DMA；已写槽内容逐槽恒等（E2 机器验证），未写槽零填 vs 垃圾对下游不可观察（E2）；附带收益 = 全缓冲确定性（golden 精确对照可行）。写量 3.5KB→68.6KB（T=64）但写路径 cycle 数不升反降（§1.6.1 OPT-2 定量） |
| 7. host 侧：lru_cache (T,K,E) 特化 + wrapper 校验 + `out_idx=[]` + torch.empty 元数据分配 | host | 无 | — | 保留 | A5 wrapper 已按同契约脚手架化（indexed_expert_mlp.py L112–142）；本设计沿用（§3.3/§4.7） |
| 8. `default_config = {"threads": 128}` | host | CUDA threads | 无 | **舍弃** | K3/K9：A5 wrapper `default_config = {}`（indexed_expert_mlp.py L130–132 已落地），工厂 callable 无参 |

> 每项处置依据：migration-analysis.md §5.3 映射表条目（warp shuffle/两级并行/online 行）/ `testing/`+CANN 头文件佐证（条目 3）/ tilelang-ascend skill §2.1/§2.2（条目 4/5）/ pattern-library `CONST-mte-bandwidth-model`（条目 6）。舍弃项（5/8）理由已写明；无静默丢弃。

### 0.6 NPU 算法重设计

对 §0.5 中判定为「重新设计」的条目逐项给出 NPU 新算法（判定输入含 §1.6.0 调研结论——源算法与调研候选同台比较，选定算法族为「确定性稳定序单遍扫描 + UB 全量 staging」，见 §1.6.0 调研结论）：

**重设计项 R1: 并行原子秩分配 → 确定性稳定序顺序扫描 + UB 全量 staging**

- **源方案**：单 block 128 线程，`T.Parallel(R)` 内 `T.atomic_add(metadata[e], 1, return_prev=True)` 并行分配双射序号；清零段与原子段之间 `T.sync_threads()`；全部读写直落 GM（分散 4B scatter）。性能意图（§0.4 #3/#6）：线程级并行摊薄秩分配延迟 + 最小 GM 写量。
- **NPU 新算法**（承接单遍在线结构与最小延迟意图，替换并行机制与写路径）：
  1. `T.copy(expert_ids, ids_ub)`：GM→UB 一次搬入（[T,K] int32，≤2KB）。
  2. `with T.SimdVF(): T.fill(meta_ub, 0)`：UB 上全量零填（缓冲按 64-lane 对齐 padding，§4.3）——承接源清零段，并把 lists 未写槽也纳入确定性合约。
  3. **顺序扫描**（核内标量段，静态边界）：嵌套 `for token in T.serial(T) / for slot in T.serial(K)`，`e = ids_ub[token, slot]`；`rank = meta_ub[e]`；`meta_ub[e] = rank + 1`；`meta_ub[E + token·K + slot] = rank`；`if rank < T: meta_ub[E+R + e·T + rank] = token·K + slot`。route 下标由嵌套循环变量重构（消除源每 route 的 div/mod，§1.6.1 OPT-1）。
  4. `T.copy(meta_ub[0:SIZE], metadata)`：UB→GM 单次 bulk 搬出（字节粒度尾区，只拷 SIZE 个元素）。
  - 单 block（`T.Kernel(1)`）单 AIV；写回路径换 MTE3 bulk（§0.5 条目 6 的等价替换并入本项）。
- **语义保持论证**：
  1. counts 段：整数计数与处理顺序无关，`meta_ub[e]` 终值 = 该专家 route 数，与源逐位相同（E4：`counts == bincount` 精确，12 组模式含全部移植测试模式）。
  2. ranks 段：稳定序双射是源"任意双射"契约的一个实现（§0.1 双射自由度）；消费不变性由两点构成：(a) 分组划分性质——任意自洽双射下每条 route 恰被计算一次（leader 组覆盖 / 独立回退），E3 机器验证稳定序与随机双射的划分与 (route→expert, 激活来源) 映射完全一致；(b) 每 route 结果行的 K 累加序与分组无关（源 GEMM 的 `ceildiv(k, block_k)` K 循环对全部 leader 块一致，属消费方结构性质，§0.3 引证），E3 的 fp64 消费模拟按固定 K 序逐行计算，两双射输出 **bitwise 一致**（max |diff| = 0）。
  3. lists 段：与各自 ranks 配对恒等（E4 配对断言）；`rank < T` 守卫逐槽一致（duplicate-fallback 模式覆盖 count_e = 8 > T = 4 的用例）。
  4. 未写槽：零填 vs 源未初始化垃圾——消费读集 ⊆ 确定写集（E2 对稳定序与随机双射分别验证，0 uncovered reads），差异不可观察。
  5. 边界核对：零 route 专家（counts=0，fill 保证）、全同专家（rank 链最长）、K=1、E=1、token 内重复专家（within-token-duplicates 模式）、rank == T-1 / rank == T 槽位边界（second-route-group 模式 T=17、E=18、block_m=16 组边界）均入 E2/E4 用例。
  6. 验证方式：§8 golden（规范稳定序 + 性质断言）+ L0 精确对照（int32 逐位相等）+ op 级移植测试（`examples/TileOPs/tests/ops/test_indexed_expert_mlp.py` 全部 grouped 用例）。
- **批式向量替代的处置（v1 修订；v0「候选 6 因 gather 原语缺失被淘汰」的依据作废）**：调研事实修正后（§1.6.0 R1 行 6 + 查证纪律注记），`vgather2`/`vscatter`/`vselr` 原语族在位（五重佐证），扫描主体的批式向量变体在原语层**可表达**且语义等价（E5 机器验证：批式模拟 vs 标量稳定序 14 组模式 bitwise 一致）。R1 维持标量主选的依据**换真**为：量化估算区间重叠（备选段 ≈4.5–7.5K cyc vs 主选标量 2.6–7.7K cyc，§1.6.2 #3）+ 复杂度/正确性面（尾批掩码 / last-occurrence counts 回写 / int32↔uint32 索引 reinterpret）；两变体同为稳定序、确定性合约不受影响。是否启用备选由 §1.6.4 实验裁决（Stage 4 A/B，阈值在位）。

**重设计项 R2: block×thread 两级并行组织 → 单 block SimdVF 向量-标量混合核**

- **源方案**：`T.Kernel(1, threads=128)`，两段 `T.Parallel`（清零 E / 路由 R）分布到 128 线程。性能意图：线程级并行。
- **NPU 新算法**：`T.Kernel(1)`（一维、无 threads 参数——tilelang-ascend skill §2.1）；VF 域声明 `T.SimdVF()` 仅包住向量段（T.fill 零填）；顺序扫描为核内标量段（VF 域外，标量核执行——`testing/ascend/language/test_tilelang_ascend_fill.py` 模块 docstring 明确「VF 块外的逐元素标量工作运行于标量核」；标量 UB 访问先例 `examples/ascend/deepgemm/kernels/transform_sf.py` 的 `group_ranges_ub[group_idx, 0]` 与 `examples/ascend/deepgemm/kernels/epilogue.py` 的 `T.serial` 标量循环）。
- **语义保持论证**：并行组织是纯实现层（migration-analysis.md §5.3「block × thread 两级并行 → 重新设计」）；单执行流消除全部跨线程共享与屏障需求（§0.5 条目 5 随之舍弃），计算内容与 R1 相同。源并行意图的承接判断见 §1.6.0 调研结论（R ≤ 512 时单核顺序链的占比 ≈2.5–4%，线程并行的收益不抵非确定性 + 已知 SimtVF 标量化风险的代价；其**向量化形态**由 §1.6.4 实验裁决备选承接——原子并行变体仍是 Stage 4 的既备回退路径之一，§9.2）。

（§0.5 无其他「重新设计」条目；条目 6 的写路径等价替换已并入 R1。）

### 0.7 标杆实现

- **源算子定义**（GPU 仓，绝对路径）：`/home/l00970450/z00875089/TileOPs/src/tileops/_kernels/moe/indexed_expert_gemm.py`（`_route_stats` L36–63、`_route_metadata_size` L30–32、`IndexedRouteStatsKernel` L255–300）；host 调用点 `/home/l00970450/z00875089/TileOPs/src/tileops/_ops/moe/routed_expert/indexed_routed_expert.py`（`IndexedExpertMLPFwdOp.forward`）；接口契约 `/home/l00970450/z00875089/TileOPs/src/tileops/_kernels/moe/call_spec.py`（`IndexedRouteStatsFwdInterface`，含 `output_size`）。
- **提取参考**（逐字，A5 仓）：`examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/_indexed_expert_m_l_p_fwd_kernels.py`。
- **golden 依据**：源仓**无 metadata 级参考实现**（源测试全部为 op 级输出对照——`examples/TileOPs/tests/ops/test_indexed_expert_mlp.py` 移植自 GPU `tests/ops/test_fused_moe_experts.py` 的 route-pattern 用例，对照 `ref_routed_experts`，见 `examples/TileOPs/workloads/moe.py` L58–91）。故 §8.1 golden 按 §0.1 语义自行给出规范实现（稳定序 + 性质断言），op 级 golden 沿用 `ref_routed_experts`（本函数语义的上层消费闭环）。
- **A5 wrapper（冻结契约）**：`examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/indexed_expert_mlp.py`（`IndexedRouteStatsKernel`，L103–142）。

## 1. 概述

### 1.1 算子名称

`_route_stats`（IndexedExpertMLPFwdOp 的路由统计 kernel；TileOPs kernel class `IndexedRouteStatsKernel`）

### 1.2 功能描述

对 `[T, K]` int32 专家路由表做单遍统计，在长度 `E + T·K + E·T` 的 int32 元数据缓冲上产出三段路由描述：每专家 route 计数（counts）、每 route 在其专家内部的稳定秩（ranks）、每专家至多 T 条的路由列表（lists）——供后续 grouped GEMM kernel 按 16 行同专家组复用权重。

### 1.3 数学公式

设 `R = T·K`，`e_r = expert_ids[r // K, r % K]`，`SIZE = E + R + E·T`：

$$
\text{counts}[e] = \#\{\,r \in [0,R) : e_r = e\,\},\qquad \text{rank}[r] = \#\{\,r' < r : e_{r'} = e_r\,\}
$$

$$
\text{list}[e \cdot T + \text{rank}[r]] = r \;\;(\text{iff } \text{rank}[r] < T),\qquad \text{metadata} = [\text{counts}; \text{rank}; \text{list}]
$$

语义约束（§0.1）：counts 精确；每专家 rank 构成 `[0, counts[e])` 双射；list 与 rank 配对；`rank ≥ T` 不入列表。整数域，无浮点。

### 1.4 算法描述

**迁移决策后的 NPU 侧算法**（= §1.6.0 选定算法族，经 §0.6 R1/R2 重设计落地；与源算法的结构差异见行末标注）：

1. **搬入**：`T.copy` 将 `expert_ids[T,K]` 一次性 GM→UB（差异：源逐线程标量 load）。
2. **零填**：`T.SimdVF()` 域内 `T.fill(meta_ub, 0)` 全量清零（SIZE_PAD 按 64-lane 对齐；差异：源只清 counts 段且为线程并行标量清零；本设计把 lists 未写槽纳入确定性合约）。
3. **顺序扫描**（核内标量段，静态边界 T×K）：嵌套 `T.serial` 循环按 (token, slot) 顺序处理每条 route：读 `e = ids_ub[token, slot]` → `rank = meta_ub[e]` → `meta_ub[e] = rank+1` → `meta_ub[E + token·K + slot] = rank` → `if rank < T: meta_ub[E+R + e·T + rank] = token·K + slot`（差异：源为 128 线程 `T.Parallel` + GM 原子 fetch-add 分配任意双射；本设计为稳定序双射 + UB 读写，消除原子与非确定性，来源 §0.6 R1）。**本步骤的向量化替代判定见 §1.6.2 #3（主选维持标量，依据换真）与 §1.6.4 实验裁决备选**。
4. **搬出**：`T.copy(meta_ub[0:SIZE], metadata)` 单次 UB→GM bulk（差异：源为分散 4B 标量 scatter store，来源 §0.5 条目 6 等价替换）。

单遍在线结构（步骤 3 内 counts/ranks/lists 同遍产出）与源一致（§0.4 #2 保留）。

### 1.5 数据流图

```
GM[expert_ids T×K int32] --T.copy(MTE2)--> UB[ids_ub]
                                         (无依赖，可与下一行 fill 重叠)
UB[meta_ub SIZE_PAD int32] <--T.fill(SimdVF 向量广播, 0)--
UB[ids_ub] + UB[meta_ub] --标量顺序扫描(T.serial×T.serial, 读改写 UB)--> UB[meta_ub]
UB[meta_ub[0:SIZE]] --T.copy(MTE3, 字节粒度尾区)--> GM[metadata SIZE int32]
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先**调研**（调研四问，§1.6.0）——同一数学语义（"路由统计：计数 + 每专家稳定/任意双射秩 + 列表"）的算法族系统调研，源算法（GPU 原子 fetch-add）只是基线候选之一；再数学等价地优化公式（§1.6.1）；再做循环 / 标量计算的向量化替代分析（§1.6.2）；然后回答向量化轴与核内布局（§1.6.3）；判定裕度落在未实证常数区间时产出实验裁决三件套（§1.6.4，v1 新增）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。本算子为**统计/规约类**（算法调研深度分级：完整调研——R1 候选表 + R3 四口径对比表 + R4 逐候选评估）。

#### 1.6.0 算法调研（Algorithm Research）⭐

**调研对象**：同一数学语义的算法族——所有产出满足 §0.1 契约（counts 精确 / 每专家 rank 双射 / lists 配对 / rank<T 守卫）的算法结构。双射自由度（§0.1，E3 机器验证）使"任意序分配"与"稳定序分配"在同一等价类内，均合法。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1 完成）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线**：源算法（GPU 原子 fetch-add） | 128 线程 `T.Parallel(R)`，`rank = atomic_add(counts[e], 1, return_prev)`，GM 直写 | —（基线；任意双射实现，§0.1 契约内） | — | ✅（基线） |
| 1 | 确定性稳定序顺序扫描 | `rank[r] = #{r' < r : e_{r'} = e_r}`，按 route 顺序单遍 running counter | 数学恒等（整数恒等式；稳定序是合法双射，E3） | 消原子 / 消非确定性 / UB 驻留 | ✅（选定核心） |
| 2 | 逐对比较秩（矩阵法） | `rank[r] = Σ_{r'<r} (e_{r'} == e_r)`，向量化比较累加 | 数学恒等（同上） | 可向量化的秩计算，但仍需 scatter 写 lists | ✅（R3 淘汰） |
| 3 | 稳定排序法 | 按 e 稳定排序 route，段内位置即秩 | 数学恒等 | 无 running 依赖 | ❌：Ascend 方言无排序原语——机械 grep `sort\|argsort` 于 `tilelang/ascend/language/*.py` 零命中（查证纪律见下注；O(R log R) 亦无复杂度优势） |
| 4 | 分块 + 跨块归并（并行确定性） | P 块各自顺序扫描 + 私有直方图，前缀偏移合并后二次遍历改写 | 数学恒等（合并偏移恒等式） | 并行化 scan | ✅（R3 淘汰） |
| 5 | dhistv2 向量直方图计数 | `T.simd.dhistv2(dst_u16x128, src_u8x256)` 硬件直方图（`tilelang/ascend/language/simd.py` L756；回归 `testing/ascend/language/simdvf/test_simdvf_histv2.py`） | 仅覆盖 counts 子问题（u8 值域、双 bin）且不产 ranks/lists | counts 向量化 | ❌：counts 是 scan 的副产品（scan 必须存在以产 ranks/lists），无净收益 |
| 6 | 核内批式分段扫描（lanes=routes 的向量变体，64-lane 批） | 批内秩 = 63 步 `vselr` 位移比较链（`simd.py` L1452；**Θ(B) 步——非相邻同值组使 log 步草图不成立**：反例 [7,5,5,7]，锚定值块摘要无法回答"该块内等于任意指定值的元素数"查询）；跨批 base = `vgather2(meta_ub[0], e_u32)`（L1159，**原语在位——v0「无 per-lane gather」断言作废**）；ranks 连续 `vsts`；lists/counts = masked `vscatter`（L1184） | 数学恒等（批内 pairwise 秩 + 跨批 base ≡ 稳定序秩；**E5 机器验证**，14 组模式 bitwise 一致） | 批内秩向量化（原语齐备；吞吐未实测） | ✅（R3 行 1c → §1.6.4 实验裁决备选） |

**R2 在线算法**：**有**。本算子天然在线：全部输出可分解为 running 计数统计量（每专家计数器）+ 逐 route 独立标注（rank 写 / list 写），单遍扫描即可产出——源算法已用 online 手段（§0.4 #2 running counter，直接证据）；选定算法同为单遍在线（分块归并候选 4 需要 2 遍 ids + GM workspace，是**更差**的两遍变体，被 R3 淘汰）。无 max/sum 类需要预扫描的规约，不存在"两遍→单遍"的额外收益空间。收益口径：扫描遍数 1（全部候选中最优）；中间缓冲 O(E+R) 计数/秩驻留 UB（非 O(R²)）。

**R3 复杂度对比**（代表 workload ds-v3-t64：T=64, K=8, E=256 → R=512, SIZE=17,152 int = 68,608B；口径：FLOPs 列含寻址/索引整数运算与 RMW，不含 DMA 搬运〔计入访存列〕；访存列 = GM 读+写；遍数 = 对 ids 的扫描次数；并行度 = 单 launch 内）：

| 算法候选 | FLOPs（标量/向量 op 数） | 访存量 (Bytes, GM) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|------------------------|-------------------|---------|-------------|---------------------|
| 0 基线：原子 fetch-add | ≈2K 标量 RMW+写（R×〔原子 + 2 写 + 寻址〕） | 读 2,048 + 写 ≤5,120（E+2R 槽，**4B 分散标量事务**） | 1 | 0（GM 原地） | 128 线程 / 1 block；需清零→原子跨线程屏障；**非确定**（A5 可行性见 R4） |
| 1 **选定**：稳定序扫描 + UB 全量 staging | fill 268 向量 op（17,152 lane / 64）+ scan ≈4K 标量 op（R×~8） | 读 2,048（bulk）+ 写 68,608（**bulk MTE3, 115 B/cycle 档**） | 1 | UB 70,656B（ids+meta，§4.5 预算内） | 1 核（AIV）；无线程/跨核同步；**确定** |
| 1b 最小写变体（counts/ranks bulk + lists 分散直写 GM） | scan ≈4K 标量 op + 512 个 4B 分散 GM store | 读 2,048 + 写 3,072（bulk）+ ≤2,048（分散） | 1 | UB ≈10KB | 1 核；lists 未写槽非确定（垃圾）——确定性合约受损，换不来收益（写路径 cycle 数与 1 持平，§1.6.1 OPT-2 定量） |
| 1c **批式向量变体**（64-lane 批 + vgather2/vscatter，= §1.6.4 备选） | 每批 ≈400–580 向量 op（63 步链 × ~6–9 op/步 + ~10 固定 + last-occurrence 折叠）→ 8 批 ≈ **3.2–4.6K 向量 op（≈205–295K lane-op）** | 同候选 1（读 2,048 + 写 68,608，bulk；批内 vgather2/vscatter 命中 UB〔VRF↔UB 256 B/cyc 类〕，非 GM） | 1 | 同候选 1 UB 70,656B + 活跃 vreg ≈10–16（VRF 假设 256KB，`CONST-vrf-capacity-a5`） | 1 核（AIV 向量管线）；批间独立链可交错；**确定**（同为稳定序，E5）；段周期估算 **≈4.5–7.5K cyc**（51 档乐观 4.0–5.8K + 未实测原语附加），与候选 1 的 2.6–7.7K cyc **区间重叠** |
| 2 矩阵法 O(R²) | ≈12K 向量 lane-op（R²/64 × 3 op）+ 仍需 lists scatter | 同候选 1 | 1 | UB 70,656B | 1 核；op 数为 scan 的 ~3×，纯劣化 |
| 4 分块 + 跨块归并（P=4..8） | P×(R/P) 标量 + P·E 偏移归并 + 二次遍历 | 读 2×2,048（两遍 ids） | 2 | GM workspace P×E int（跨核） | P 核；2 次跨核同步 + 块间前缀串行链；scan 段至多 ~3× 加速（5–8K → ~2–3K cyc），占 op 比例 <1.5%→<0.6%，复杂度/收益失衡 |
| 3 稳定排序法 | O(R log R) 比较 | 同候选 1 量级 | 1–2 | — | API 缺失（R1 行 3，grep 佐证），不参与数值对比 |

**R4 硬件亲和性评估**（逐候选对照检查清单；常数引自 `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`，A5 冷启动期 status=assumption，容量按 90% 折扣）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 跨核结构 | 结论 |
|---------|-------------|---------|------------|---------|------------|---------|------|
| 0 基线：原子 | AIV 标量原子（`asc_atomic_add` int32 GM 重载存在——CANN 头文件 + codegen 路径已引证，§0.5 条目 3；本仓回归仅 fp32）；int32 元数据无 16-bit 归约域 → 不命中 CG-2026-0002 缺口本体（该缺口为 bf16/fp16 域 SimtVF 归约） | 0 UB（GM 原地） | 标量无向量对齐需求 | ✅（工厂烘焙） | 无流水（单段） | 清零→原子需跨线程屏障：SimtVF 自动 asc_syncthreads 依赖检测对 GM RAW 的覆盖无文档化保证（tilelang-ascend skill §2.2 仅给出机制与 UB 冒险回归 testing/ascend/transform/test_ascend_thread_sync.py，GM 缓冲场景未单列）——**未文档化假设**，列 §9 风险 | ⚠️ 可行但非确定 + 屏障覆盖存疑 + int32 原子无本仓回归先例 |
| 1 **选定**：稳定序扫描 + staging | AIV：fill/DMA 走 Vector/MTE，scan 走标量核（全部 int32，无 16-bit 域）；计算落位与单元匹配（无 MAC 密集段，无需 Cube） | UB 70,656B ≤ 228,556B（248KB×0.9，SimdVF 无 32KB 线程上下文扣减，`CONST-capacity-a5`）= 31% | fill 需 64-lane 整除（SIZE_PAD padding 处理，§4.3；`src/ascend/transform/ascend_simdvf_lower_parallel.cc` CanLowerFillRegion 的 vreg_size 整除 + 32B 基对齐规则）+ 32B 基对齐（UB 分配天然满足）；搬出字节粒度尾区（GM↔UB DMA 支持字节粒度行，tilelang-ascend skill §6.1；UB→GM 方向尾区状态见 §5.3 首验项） | ✅（T/K/E/SIZE 全为工厂编译期常量，`核内串行循环边界静态` 满足 core-split-strategy.md §2.1） | MTE2(入) 与 fill 无数据依赖可重叠；scan 依赖两者；AutoSchedule 自动插 pipe 同步（§7） | grid=1 单核，零跨核同步 | ✅ **选定** |
| 1b 最小写 | 同 1 + 512 个 4B 分散 GM 标量 store（无 DMA 聚合） | UB ≈10KB | 同 1 | ✅ | 同 1 | 无 | ❌ 确定性受损且无 cycle 收益 |
| 1c 批式向量变体（§1.6.4 备选） | AIV 向量管线：vcmp/vadd/vand 类在 51 lane-op/cyc 档（`CONST-valu-model`）；**vselr 63/批 + vgather2/vscatter 3/批（≈66 op/批）不在 51 档——constants.md 无对应吞吐条目（未实测）** | UB 同 1（31%）+ VRF 活跃 vreg ≈10–16 ≪ 256KB 假设（`CONST-vrf-capacity-a5`） | 64-lane 批整除（R%64 尾批 valid 掩码）；索引 int32→uint32 位重解释（vscatter 要求 32-bit 元素偏移 uint32，simd.py L1187；库内同机制先例 vselr L1455）；ids 平铺装载 Stage 4 首查项（§1.6.4 ②） | ✅（nb = ceil(R/64) 工厂烘焙） | 批间独立链可交错；MTE2∥fill 同主选 | grid=1 零跨核 | ⚠️ **原语齐备可表达**（五重佐证，§1.6.2 #3）但段周期估算 4.5–7.5K cyc 与主选 2.6–7.7K cyc 重叠、关键吞吐未实测 → **实验裁决备选（§1.6.4），不选为主选** |
| 2 矩阵法 | Vector 密集比较（可向量化）但 op 数 ~3× 于 scan | 同 1 | 64-lane 整除（R=512 ✓；任意 R 需掩尾） | ✅ | 同 1 | 无 | ❌ 复杂度劣化（R3） |
| 4 分块归并 | 标量 scan × P + 归并段 | GM workspace | 同 1 | ✅ | 两阶段结构破坏单遍性 | 2 次跨核同步 + 块间前缀串行链 | ❌ 收益 <0.6% op 占比 vs 结构复杂度 |
| 5 dhistv2 | 硬件直方图指令（u8x256→u16x128，双 bin） | 寄存器 | uint8 收窄约束 | ✅ | — | — | ❌ R1 行 5 |
| 6 vselr 链（= 1c 的批内链形态） | Vector 寄存器内位移比较链（`testing/ascend/language/simdvf/test_simdvf_mutable_reads.py` vselr 先例；`vci` lane ramp L1038） | 寄存器 + counts 段 UB（跨批 base 取数 = `vgather2(meta_ub[0], e_u32)`——**原语在位 L1159，v0「无 gather 原语」断言作废**） | 64-lane 批 + 尾批掩码 | ✅ | 批间独立链可交错；链内 63 步 | 无 | ⚠️ 原语齐备但批内秩需 Θ(B) 链（R1 行 6 反例注记）→ 并入 1c 实验裁决（§1.6.4） |

**调研结论**: 选定算法族 = **候选 1「确定性稳定序单遍顺序扫描 + UB 全量 staging + 单次 bulk 搬出」**（候选 1 的核心 + 候选 0 的单遍在线结构），关键依据：R3 表——写路径 bulk 化（分散 4B 事务 → 115 B/cycle 档 MTE3，`CONST-mte-bandwidth-model`）+ 确定性（全缓冲逐位可复现，golden 精确对照）；R4 表——唯一同时满足「容量 31% / 边界静态 / 零跨核同步 / 无未文档化依赖」的候选（基线的屏障覆盖与 int32 原子先例均为未文档化/无回归项）。**批式向量变体（1c）不选为主选的量化依据（v1 修正）**：段周期估算 ≈4.5–7.5K cyc（vselr/vgather2/vscatter ≈66 op/批 不在 51 档、吞吐未实测；51 档乐观折算 4.0–5.8K cyc）与主选标量链 2.6–7.7K cyc **区间重叠——无纸面优势**，而尾批掩码 / last-occurrence counts 回写 / 索引 reinterpret 的复杂度与正确性面显著增加；因判定裕度落在未实测常数区间（pattern-library constants.md 无对应条目，已核对），1c 以**实验裁决备选**保留（§1.6.4 三件套，E5 已机器验证其语义 = 稳定序 bitwise 一致），Stage 4 A/B 定夺。与基线的结构差异一句话：把"128 线程原子竞争分配任意双射 + GM 分散直写"换成"单核按 route 顺序分配稳定双射 + UB 驻留 + 单次 bulk 写回"。**调研范围**（"无更优替代"的反向确认）：`algorithm-candidates.md` ALG-reduction（statistics 族：行块并行/跨核 N-split/Welford 形态——针对浮点行归约，本算子为 int32 scatter 统计，结构判据不匹配）/ ALG-topk（部分选择 vs 全排序——排序原语缺失同 R1 行 3，机械 grep 佐证）；`examples/ascend/`（example_gemm_splitk.py 的 deterministic-ordered vs atomic 双模式先例、deepgemm 标量 T.serial + UB 标量访问 + **vgather2 用例 transform_sf.py:286**、vecadd/rmsnorm 归约形态）；`testing/ascend/`（simtvf/test_simtvf_atomics、simdvf/test_simdvf_fill、simdvf/test_simdvf_histv2、simdvf/test_simdvf_mutable_reads（vselr）、test_tilelang_ascend_buffer_reuse.py:270（vgather2）、auto_schedule/test_access_footprints.py:527（vgather2/vgatherb））；pattern-library constants.md（CONST-valu-model / CONST-mte-bandwidth-model / CONST-capacity-a5 / CONST-arch-a5 / CONST-cube-throughput-model / CONST-vrf-capacity-a5——**vgather2/vscatter/vselr 吞吐无实测条目，已 grep 核对**）；`tilelang/ascend/language/` 原语族机械 grep（见下注）；结构分析（双射自由度 + 顺序依赖链 + 批式变体区间量化，E1–E5 机器验证）；`tilelang/ascend/language/tile_schedule.py` 的 `AscendMGroupedTileScheduler`（MoE m-grouped prefix-sum 布局调度器——**消费方** tile scheduler，需调用方提供 `grouped_layout` prefix-sum 输入，不产出路由统计，判据不匹配，已核对 L345–352 docstring）。**源算法优化手段承接**（§0.4 → 选定算法）：#1 单遍融合 → 保留（单 kernel 四段）；#2 在线 running-counter → 保留（scan 即 running counter）；#3 并行秩分配意图 → 主选不再需要（R ≤ 512 时单核顺序链占 op ≈2.5–4%，见 D-2），其向量化形态由 1c 备选承接（实验裁决），原子并行变体为 Stage 4 既备回退（§9.2）；#6 最小写量意图 → 由 bulk MTE3 承接（写路径 cycle 更优，OPT-2 定量）。

**查证纪律注记（v1 新增，防 v0 失误再发）**：负向 API 断言一律先对原语族关键词机械 grep 再下结论——本算子执行 `grep -n -E "gather|scatter|sort|prefix|scan|hist|select" tilelang/ascend/language/*.py`，命中函数逐一读取 docstring 核对语义。命中清单：`vgatherb`（simd.py L1144，32B 块 gather）、`vgather2`（L1159，per-lane offsets gather）、`vscatter`（L1184，per-lane offsets scatter-store）、`vsel`（L1447）/`vselr`（L1452，per-lane 索引选择）、`vci`（L1038，lane ramp）、`vusqz`（L1029，mask 前缀计数）/`vsqz`（L1022）、`dhistv2`/`chistv2`（L756/L765，直方图）。**v0 的「无 per-lane gather / 跨 lane scatter 原语」断言即因跳过该流程而失效**（v0 读取了 transform_sf.py L199–200 的标量访问却漏掉同文件 L286 的 vgather2 用法）；「无排序原语」断言经同一 grep 确认成立（sort/argsort 零命中）。后续任何「API 不存在 / 无先例」类负向断言一律附 grep 命令与命中清单或显式标注「未文档化假设 + 估算依据」。

**设计期估算（D-2）**: 估算下界(相对): **≈ 2–8K cycles，主导项 = 顺序扫描标量依赖链**（流量：读 2,048B ÷ 100 B/cycle + 写 68,608B ÷ 115 B/cycle ≈ 618 cyc，`CONST-mte-bandwidth-model`〔assumption〕；计算：fill 17,152 lane-op ÷ 51 ops/cycle ≈ 337 cyc〔`CONST-valu-model`，assumption，lane-op 粒度口径〕+ scan 512 iter × 依赖链 10–15 cyc ≈ 5–8K cyc〔**未实测假设**：UB 标量 load-to-use 延迟按 10–15 cyc 计、不同专家迭代间可流水重叠，实际趋向下界；流水乐观界 ~5 cyc/iter → ≈2.6K cyc，作为 §1.6.4 裁决区间下端披露〕；容量：70,656B ≤ 228,556B 单遍驻留无 tiling 循环〔`CONST-capacity-a5`，assumption，90% 折扣〕）。对照：op 的 Cube 段 ≈ 45.08 GFLOP ÷ (8192 ops/cycle/AIC × 28 AIC) ≈ 197K cycles（T=64，`CONST-cube-throughput-model`〔assumption〕）→ 本 kernel 占比 **≈2.5–4%（上界 8K/197K=4.1%）**；**占比随 T 近似不变**（scan 与 Cube 段同 ∝ T·K），T=32 复算 ≈2.6–3.9%（scan 2.6–3.8K cyc / Cube ≈98K cyc = 22.54 GFLOP ÷ 229,376 ops/cyc）。协议见 `_shared/standards/hardware-cost-model.md` §2（相对量级比较，禁止编造频率换算绝对时延；A5 冷启动期为 assumption 基估算，常数待标定）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（稳定序扫描 + UB staging，经 §0.6 R1 落地）。本算子为纯整数索引/计数问题，无浮点公式与超越函数；等价优化集中于整数恒等变形与访存路径削减。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | OPT-1 嵌套循环消除 div/mod | `e = expert_ids[route // K, route % K]`（每 route 1 div + 1 mod） | `for token in T.serial(T): for slot in T.serial(K): e = ids_ub[token, slot]; route = token·K + slot` | 整数恒等式 `route = token·K + slot ⟹ route//K = token ∧ route%K = slot`（除法定义，K>0）；机器验证 E1（7 组 (T,K) 跨度 0 失配） | R 次 div+mod（13 ops/cycle 档，`CONST-valu-model`）→ 0 次 div/mod + 1 次乘加（乘常数 K 被强度削减）；scan 关键链上每 iter 省 2 个慢档 op（R=512 → ≈79 cycle 当量，占比小但零风险） |
| 2 | OPT-2 写路径 bulk 化（GM 分散 scatter → UB staging + 单次 MTE3） | 每条 route 3 个 4B 分散标量 GM store（counts RMW + ranks + lists） | 全部写驻留 UB `meta_ub`，末尾 `T.copy(meta_ub[0:SIZE], metadata)` 单次连续搬出 | 已写槽内容逐槽恒等（同一写值集合，仅落点路径不同）；消费读集 ⊆ 写集（E2：稳定序与任意双射下 0 uncovered reads，12 组模式）→ 写路径变化对任何可观察输出无影响 | 写路径 cycle：≈(E+2R) 个 4B 分散标量 store（标量发射 + 无聚合，估算 ≈1.5–4K cyc，**未实测假设**：每分散 store 2–5 cyc 发射）→ fill 337 + MTE3 597 ≈ 934 cyc（`CONST-mte-bandwidth-model`〔assumption〕）；GM 写字节 3.5KB→68.6KB（HBM 容量无关紧要），**换全缓冲确定性**（golden 逐位精确对照可行，验证强度升级为性质+精确双保险） |
| 3 | OPT-3 lists 未写槽零填（确定性合约） | lists 未写槽 = `torch.empty` 垃圾（源行为，§0.1 边界 (c)） | `T.fill(meta_ub, 0)` 全量零填覆盖未写槽 | 未写槽不被下游消费（E2：消费读集 ⊆ 写集，0 uncovered）→ 零值与垃圾值同属不可观察；机器验证 E2 | 消除输出缓冲的非确定区域 → 整个 metadata 缓冲逐位可复现（L0 可用 `torch.equal` 精确断言，含未写槽）；成本已含在 OPT-2 的 fill 内（零额外） |

**等价性机器验证（D-1，`examples/indexed_expert_mlp/_route_stats/verify_equiv.py`，torch CPU + fp64 消费模拟，2026-10-29 revision 1 重跑）**——12 组路由模式（4 组移植测试 route-pattern：dispersed / reused / duplicate-fallback / second-route-group + 4 角点 + 4 组 manifest 形态随机路由，每组含稳定序与随机双射两个 metadata 变体；E5 另加 2 组批边界压力模式，共 14 组）：

| 验证项 | 最大差 | 违反率 | 结论 |
|---|---|---|---|
| E1 OPT-1 `route=token·K+slot ⟹ route//K==token ∧ route%K==slot`（7 组跨度） | 0 | 0% | EQUIV_PASS |
| E2 OPT-2/OPT-3 消费读集 ⊆ 确定写集（稳定序与任意双射，12 组模式） | 0 | 0% | EQUIV_PASS |
| E3 §0.6 R1 双射自由度：counts / 分组划分 / fp64 消费输出，稳定序 vs 随机双射 | 0（bitwise） | 0% | EQUIV_PASS |
| E4 确定性合约：counts==bincount / 稳定秩 / lists 配对 / rank<T 守卫 / 零填 / int32 值域（12 组模式） | 0 | 0% | EQUIV_PASS |
| E5 §1.6.4 备选变体批式分段扫描（64-lane 批 + last-occurrence counts 回写）= 稳定序 metadata（14 组模式：12 共享 + 批边界压力 R512 重跨批重复 / 单尾批） | 0（bitwise） | 0% | EQUIV_PASS |

**优化结论**: 采纳 3 项（OPT-1 / OPT-2 / OPT-3，均 EQUIV_PASS；E5 为 v1 新增的备选变体语义验证，非主选采纳项）；优化后"公式"（供 §3.1 拆解）即 §1.4 步骤 2–4 的整数扫描式 + 全量 staging 写回。无浮点公式可化简（纯整数索引/计数，无公共浮点子式、无高代价超越函数——调研范围：§1.6.0 R1 全表 + 恒等变形自查）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态（选定算法内） | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|------------------------|-------------|---------|------------------------------|
| 1 | metadata 全量零填 | —（OPT-3 引入） | `T.fill(meta_ub, 0)` in `T.SimdVF()`（向量广播 fill；佐证 `testing/ascend/language/simdvf/test_simdvf_fill.py`：T.fill 于 SimdVF 内向量降级，整缓冲/子区域/流水槽位回归） | ✅ | — |
| 2 | ids 搬入 / metadata 搬出 | GM↔UB DMA | `T.copy`（MTE2/MTE3 任务，非 VF 计算） | ✅（DMA 天然批量） | — |
| 3 | **顺序扫描主体**（R 次：读 e → 读 rank → 写 counts → 写 ranks → 条件写 lists） | 嵌套 `T.serial` 标量循环（UB 标量读改写） | 候选替代：(a) `T.Parallel` 向量化——**不可行**：rank 依赖同专家先前 route 的 counts 更新（循环携带依赖，跨迭代 RAW on `meta_ub[e]`），朴素并行破坏 running-counter 语义；(b) **批式分段扫描向量化（64-lane 批，§1.6.4 备选）——原语齐备、可表达（v1 修正）**：`vgather2`（simd.py L1159）/`vscatter`（L1184）/`vselr`（L1452）/`vci`（L1038）在位（codegen_ascend.cc:2294/2468–2480；用例 transform_sf.py:286 / test_tilelang_ascend_buffer_reuse.py:270 / test_access_footprints.py:527）——批内秩 = 63 步 vselr 位移比较链（Θ(B) 步，log 步草图不成立——反例 [7,5,5,7]，R1 行 6），跨批 base = `vgather2(meta_ub[0], e_u32)`，ranks 连续 `vsts`，lists/counts = masked `vscatter`；(c) 原子并行（源基线路径）——存在且合法（§0.5 条目 3 举证），但引入非确定性，被 §0.6 R1 重设计弃用；(d) 矩阵法（E×R 比较）——R3/R4 定量淘汰（~30× 计算量） | ❌ **主选维持标量**；备选 (b) 进 §1.6.4 实验裁决（结构齐备 + E5 语义机器验证 + 裁决阈值在位） | **判定依据（量化，v1 换真）**：① tile 级顺序依赖——rank 分配对每专家是 running counter 的串行链，备选 (b) 以「批内 Θ(B) 位移比较链 + 批间 base gather / counts 回写 scatter」表达该依赖，**结构上可表达但非归约可消除**；② 量化对比——备选 (b) 段周期 ≈4.5–7.5K cyc（每批 ≈400–580 向量 op → 8 批 ≈3.2–4.6K op ≈205–295K lane-op；51 档乐观 4.0–5.8K cyc + vselr 63/批、vgather2/vscatter 3/批不在 51 档〔吞吐未实测〕的附加 0.5–1.7K cyc），主选标量链 2.6–7.7K cyc（c_scan ∈ [5,15] cyc/iter 未实测假设）——**两区间重叠，备选无纸面优势**；③ 收益上限 = 消除扫描段本身 ≤4% op 时延（D-2），而复杂度/正确性面增加（尾批掩码、last-occurrence 回写、索引 reinterpret、批装载平铺路径待验证）；④ 关键吞吐常数（vselr/vgather2/vscatter、标量链重叠度）pattern-library 均无实测条目（已核对）→ 判定裕度落在未实证区间，按「实验裁决模式」交 §1.6.4 三件套，Stage 4 A/B 定夺，不以纸面单选拍板 |
| 4 | 循环边界 / 基址常量（rank_base=E、list_base=E+R、route=token·K+slot） | 标量乘加 | 无（编译期常量折叠 + 循环变量仿射式） | ❌ | 每核少量索引计算（block 级索引类，非逐元素热点；K 为编译期常数，乘法被强度削减） |
| 5 | `if rank < T` 槽位守卫 | 标量运行时分支 | 向量掩码（`T.simd.pset/pld/pst`） | ❌（主选链内） | 依赖运行时数据（rank）的逐 route 边界处理；主选标量链内分支粒度为 1（#3 主选维持标量）；**备选变体中该守卫由 `vscatter` 掩码自然承载**（§1.6.4 ②，E5 覆盖 rank<T 路径） |

**向量化结论**: 计算点 5 处，向量化/DMA 化 2 处（零填 fill、双端搬运），标量保留 3 处（扫描主体 / 索引常量 / 槽位守卫）。**#3 的保留依据（v1 换真）**：批式向量变体原语齐备、语义等价（E5 bitwise），但量化估算区间重叠（备选 ≈4.5–7.5K vs 主选 2.6–7.7K cyc）且关键吞吐常数未实测 → 无纸面优势，按实验裁决模式保留主选 + §1.6.4 备选结构（判定裕度显式交 Stage 4 实测，阈值在位）。理由类别：**tile 级顺序依赖 + 量化区间重叠（实验裁决）**（#3）、**block 级索引计算**（#4）、**动态数据边界**（#5，主选链内；备选由 vscatter 掩码承载）。逐元素可向量化部分（零填、搬运）已全部向量化/DMA 化；替代方案 API 均有 `testing/ascend/` 佐证。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> 本算子为一维 int32 元数据 kernel（单条 metadata 轴 + [T,K] 路由表），属"单轴算子"情形，但仍按规约类条目核对正交决策（水平归约 vs 垂直扫描）与核内布局候选。

**轴质量评分维度**（对主选布局）：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | fill 区域 64-lane 整除（int32 vreg = 2048bit/32bit = 64 lane，`CONST-valu-model`） | SIZE_PAD = ceil(SIZE/64)·64 后整除 ✓（harness 最大 shape T=64/E=256: SIZE=17,152 = 268×64 天然整除；测试小 shape 如 SIZE=48/358 需 padding，§4.3）；搬出 extent=SIZE 字节粒度（非 32B 行不强制 padding，tilelang-ascend skill §6.1；UB→GM 尾区状态见 §5.3 首验项） |
| 尾 lane 浪费率 | (SIZE_PAD−SIZE)/SIZE | ≤ 64/358 ≈ 17.9%（最小测试 shape）；harness 主 workload 全部 0% |
| 累加链形态 | 串行累加维 = route 序（stride 1 于 [T,K] 行主展开）；向量维 = fill/move 的 metadata 连续轴（stride 1） | fill/copy 连续无跨步；scan 串行链已由 §1.6.2 #3 论证（主选标量 + 备选裁决） |
| repack 代价 | 核内布局变化的重排代价 | 无需 repack：ids 保持 [T,K] 原生 2-D（核内 div/mod 由嵌套循环消除，§1.6.1 OPT-1），metadata 保持 1-D 连续（三段基址常量偏移） |
| UB 容量影响 | 驻留缓冲尺寸 | 70,656B ≤ 228,556B（§4.5） |

**候选矩阵**（规约类正交决策：lane 映射到归约轴〔水平〕 vs 映射到独立行〔垂直扫描〕；gather/scatter 类的连续批量粒度）：

| # | 布局方案 | 向量化轴 / lane 映射 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|--------------------|------------|--------------|---------|
| 1 | **主选**：ids 原生 [T,K] + metadata 1-D 连续 staging；fill/copy 向量轴 = metadata 连续轴；scan = 垂直逐 route（lane 概念不适用，标量链） | fill/copy: metadata 轴（64-lane 连续）；scan: route 序标量 | 无 | 基线（选定）；确定性全缓冲；§1.6.0 估算 2–8K cyc | ✅ |
| 2 | lanes=routes（水平）：64-lane 批式分段扫描（vselr 位移比较链 + vgather2/vscatter，= §1.6.4 备选结构） | route 轴（水平批形态，批间按批序串行） | ids 按 64-lane 批装载（vld NORM 连续 ✓；lane/route 索引 `vci` ramp L1038） | 批内秩需 **Θ(B)=63 步链**（**log 步草图不成立**——非相邻同值组反例 [7,5,5,7]：锚定值块摘要无法回答任意值计数查询）；段周期估算 ≈4.5–7.5K cyc vs 主选 2.6–7.7K cyc **区间重叠**；vselr/vgather2/vscatter 吞吐未实测（constants.md 无条目，已核对）；收益上限被 R≤512 规模封顶（≤4% op） | ❌ 作主选（量化口径：区间重叠无纸面优势 + 尾批掩码/last-occurrence 回写/索引 reinterpret 复杂度）；✅ **作 §1.6.4 实验裁决备选**（结构齐备 + E5 语义机器验证 + 裁决阈值在位）。v1 修正：原「gather 墙（API 缺失）」论断与 simd.py 源码矛盾，已删除 |
| 3 | lanes=experts（水平）：64 专家并行 × R 步垂直扫描（每 lane 一个专家计数器，广播比较计数） | expert 轴 | ids 需与专家 lane 向量逐块比较（vld + vcmp） | E×R 比较 = 131K lane-op（vs scan ≈4K 标量 op，**~30× 计算量**；51 档折算 ≈2.6K cyc + 每步秩提取/回写附加 → 段周期不低于主选）；秩/列表回写**可由 `vscatter` 表达**（L1184，原语在位）但被比较工作量支配 | ❌ 计算量 ~30× 劣化（主理由，量化在位；v1 删除「跨 lane scatter 无原语」不实分句——原语存在，既不构成障碍也不构成收益） |
| 4 | host permute / 重排进核 | — | host 侧 reshape/连续化 | 无需（输入已连续，[T,K] 行主 = route 序展开） | ❌ 无收益 |
| 5 | ids 1-D 展平变体（[T·K] 单轴 UB） | 同主选 | `expert_ids.reshape(-1)` 视图（host 元数据操作，零拷贝） | 与嵌套循环等价（div/mod vs 单层循环 + 下标乘加）；主选已用嵌套循环达到同一效果且免 host 视图操作 | ❌ 等价无增益（备注：若 Stage 3 实现时 host 侧 reshape 视图更顺手，二者在 §0.1 语义下等价可换；§1.6.4 备选的批装载即采用平铺形态，其实现路径见 ②） |

> 迁移任务核对：GPU 源码的并行轴（128 线程分布 route 维）是输入而非结论——已作为基线候选 0 在 §1.6.0 同台评估后被替换（线程轴 ≠ NPU Vector 轴）。

**布局决策结论**: 选定方案 #1：核内布局 = ids 原生 [T,K]（2-D UB）+ metadata 1-D 连续（SIZE_PAD 对齐分配、SIZE 粒度搬出）；向量化轴 = metadata 连续轴（fill 与两端搬运，64-lane 整除由 SIZE_PAD 保证）；scan 为 route 序标量垂直链（lane 映射不适用，理由 §1.6.2 #3 + §1.6.4 裁决）；repack 路径 = 无。该结论同步落入 §3.3 伪代码与 §6 循环结构（scan 的内层为 slot 序、fill/copy 为 metadata 连续轴）；弃选方案的量化理由见候选矩阵（#2 以「区间重叠 + 复杂度 vs 收益上限」口径弃选为主选、保留为 §1.6.4 实验裁决备选；#3 以 E×R≈131K lane-op ~30× 计算量劣化弃选——v1 已删除其 scatter 缺失不实分句）。

#### 1.6.4 实验裁决三件套（scan 主体：标量链主选 vs 批式向量备选）⭐（v1 新增）

> **触发条件**（skill Phase 2 实验裁决模式）：主选与备选的判定裕度落在未实证常数区间——① `vgather2`/`vscatter`/`vselr` 吞吐在 pattern-library constants.md 无实测条目（grep gather|scatter|vsqz|vusqz|prefix|scan 于 constants.md：仅 L78 一处 GEMM ND2NZ 约束无关行，已核对）；② 标量链每迭代成本 c_scan 无实测（10–15 cyc/iter 为假设，流水乐观界 ~5）；③ 两侧段周期估算区间重叠（4.5–7.5K vs 2.6–7.7K cyc，§1.6.2 #3）——不得仅凭先例或下界估算纸面单选，产出三件套。

**① 主选方案（进 Stage 3）**：标量稳定序扫描（§1.4 / §3.3 / §6 已承载）——当前证据下的最优判断：确定性合约 + 估算区间不劣于备选 + 零未实测原语依赖（fill/copy/serial 均有回归佐证）+ 正确性面最小。

**② 备选方案（批式分段扫描向量变体，冻结进 Stage 4 A/B 清单，结构达可直接实现深度）**：

- **buffer 与 UB 预算**：与主选相同——ids_ub + meta_ub，ds-v3-t64 合计 70,656B ≤ 228,556B（31.0%，`CONST-capacity-a5` assumption 90% 折扣）；**无新增 UB 缓冲**（批内临时量驻留 VRF：活跃 vreg ≈10–16 ≪ 假设容量 256KB/256B = 1024 vreg，`CONST-vrf-capacity-a5` assumption）。
- **ids 平铺装载（Stage 4 首查项）**：批装载需按 route 轴平铺访问。两条候选路径（均静态边界，择一验证）：(a) `ids_flat_ub` 1-D (R_PAD,)（R_PAD = ceil(R/64)·64）+ `T.copy(expert_ids, ids_flat_ub[0:R])` 同元素数区间拷贝（[T,K] 行主连续 → 平铺；logsumexp_simd 部分范围拷出同族先例），尾段 [R, R_PAD) 零填（pad 值 0 ∈ [0,E) 在范围内，全部副作用被 valid 掩码屏蔽）；(b) 保持 (T,K) `T.copy` + 批装载 `vld(ids_ub[route0//K, route0%K])` NORM 连续 64 元素跨行读（`transform_sf.py:288` 的 vsts NORM_B16 同族先例）+ ids_ub 行 padding 至 R_PAD。
- **循环结构**（`nb = ceil(R/64)` 工厂烘焙静态边界；全部向量 op 在 SimdVF 域内）：

```python
with T.SimdVF():
    lane0 = T.simd.vci(0, "int32")                 # lane 索引基（L1038：dst[lane]=index+lane）
    one = T.simd.vdup(T.int32(1), "int32"); zero = T.simd.vdup(T.int32(0), "int32")
    for b in T.serial(nb):                         # 批循环（静态边界）
        route = T.simd.vci(b * 64, "int32")        # per-lane route id ramp
        valid = route < R                           # 尾批掩码（R % 64 ≠ 0 时）
        e_vec = T.simd.vld(ids_flat_ub + b * 64)   # 批 ids 装载（NORM 连续 64 元素）
        e_u32 = reinterpret(e_vec, "uint32x64")     # gather/scatter 索引域（32-bit 元素偏移须 uint32；库内同机制先例 vselr L1455）
        base = T.simd.vgather2(meta_ub[0], e_u32)   # 跨批 base = counts[e]（批首快照，L1159）
        cnt = zero; fut = zero                      # 批内秩累加 / last-occurrence 探测
        for t in T.serial(63):                     # 位移比较链（Θ(B)，见 §1.6.3 #2 反例注记）
            idx_t = T.simd.vci(-t, "int32")         # [−t, 63−t]
            e_shift = T.simd.vselr(e_vec, idx_t)   # 位移取值（L1452）
            eq = T.simd.vcmp(e_shift, e_vec) & (idx_t >= 0) & valid
            cnt = cnt + T.simd.vsel(one, zero, eq)  # 批内 pairwise 秩（eq 位独立，可树形累加消串行链）
            eq_fut = T.simd.vselr(eq, lane0 + t) & (lane0 + t < 64)
            fut = fut | T.simd.vsel(one, zero, eq_fut)  # 未来同值存在 → 非最后出现
        rank = base + cnt
        T.simd.vsts(meta_ub[E + b * 64], rank, mask=valid)          # ranks 段连续写
        rank_u32 = reinterpret(rank, "uint32x64")
        T.simd.vscatter(route, meta_ub[E + R],                       # lists 段（L1184）
                       e_u32 * T + rank_u32, mask=valid & (rank < T))
        T.simd.vscatter(base + cnt + one, meta_ub[0],               # counts 回写（仅 last-occurrence lane）
                       e_u32, mask=valid & (fut == 0))
```

- **dtype 路径**：int32 主域；索引 `reinterpret` uint32x64（`vscatter` 要求 32-bit 元素偏移 uint32，simd.py L1187；库内位重解释机制先例：vselr 的 index 前置转换 L1455、float32 通道 L1459–1462；前端表面形态〔tirx.reinterpret / 库内 helper〕为 Stage 4 实现细节）；**无 u8 收窄依赖**（last-occurrence 用链式折叠而非 dhistv2 直方图路径——E>256 亦可用；dhistv2 的 u8x256 值域约束不引入）。
- **语义保持论证（E5 机器验证，2026-10-29）**：批内 pairwise 秩 + 跨批 base（批首 gather 的 counts）≡ 稳定序全局秩；counts 经 last-occurrence lane 散写 = base + 批内总数（每专家每批恰一个 last lane，跨批累加恒等）；lists 掩码散写同主选——`verify_equiv.py` E5：批式模拟 vs 标量稳定序，**14 组模式（12 共享 + 2 批边界压力：R=512/4 专家重跨批重复、单尾批 R=6）bitwise 一致，EQUIV_PASS**。
- **分核三要素（按备选粒度重算）**：grid=1（`T.Kernel(1)` 一维，同主选）；纯 Vector → 1 AIV（批循环 + 全部向量 op 在 SimdVF 域内；无 Cube/无 sid）；`auto_schedule: yes`——依赖边：MTE2（ids）→ 批循环、fill（meta）→ 批循环、批循环 → MTE3（meta），MTE2∥fill 重叠自由度同主选；无手动 flag（核内 8 / 跨核 16 预算零消耗）；核内串行边界（nb、R、E、T、SIZE_PAD）工厂烘焙静态 ✓。
- **量化估算**（ds-v3-t64：R=512 → nb=8 全整批）：每批 ≈400–580 向量 op（63 步链 × ~6–9 op/步：vci/vselr/vcmp/vand/vsel/vadd + fut 折叠 vselr/vsel/vor + ~10 固定：vld/vgather2/vsts/vscatter×2/掩码）→ 8 批 ≈ **3.2–4.6K 向量 op（≈205–295K lane-op）**；51 档（`CONST-valu-model` assumption）乐观折算 ≈ **4.0–5.8K cyc**；其中 vselr 63/批 + vgather2 1/批 + vscatter 2/批（≈66 op/批）**不在 51 档、吞吐未实测** → 总估 **≈4.5–7.5K cyc**。对照主选标量链 2.6–7.7K cyc：区间重叠。

**③ 裁决计划（Stage 4 执行）**：

| 项 | 内容 |
|---|---|
| 代表 shape | **ds-v3-t64**（T=64,K=8,E=256 → R=512，8 个整批无尾掩码干扰，主 workload）；辅助：ds-v3-t32（4 批）、(17,2,18)（R=34 单尾批——尾掩码路径专项） |
| 指标 | kernel 级 latency（`tilelang.profiler.do_bench(fn, backend="msprof")`，AGENTS.md 唯一端到端权威）+ npusim 周期归因（scan 段隔离对照）；op 级端到端为次级指标（metadata 生产者串在关键路径，§5.5） |
| 未知常数实测/反解 | c_scan（标量/迭代）= 主选 npusim scan 段周期 ÷ R；c_vselr / c_gs（vgather2、vscatter 单 op）= 备选段周期反解（按 ② 的 op 清单扣除 51 档 ALU 份额）或独立微探针（D-3 预算内：≤2 个 × ≤10min，N 次原语的最小 kernel，落 `history_version/design_probe_*.py`） |
| 判定阈值 | **翻转备选 ⟺ 同时满足**：(i) 备选 kernel latency < 0.8 × 主选（kernel 级 >20% 改善）**且** (ii) 节省周期 ≥ 2K cyc（≈ op 时延 ≥1%，对照 Cube 段 197K cyc）。任一不满足 → 维持主选（复杂度不抵 <1% op 收益） |
| shape 分派可行性 | `lru_cache` 工厂按 (T,K,E) 特化——若裁决显示分派优势（如 R≥256 大 shape 备选胜、R≤64 单批 shape 主选胜——批式链开销不摊薄），工厂内 host 分支按 shape 分派两变体（各自核内边界静态），总体最优而非全局二选一 |
| 回写路径 | 实测数字 → §1.6.2/§1.6.3 判定修订 + 胜出变体晋升主选（DESIGN.md 修订，走 conductor 回退链）；c_scan/c_vselr/c_gs 实测值 → pattern-library constants.md 候选条目（`CONST-valu-model` 扩充，蒸馏时合入）；A/B 数据落 Stage 4 perf_records.jsonl |


## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: **simd**（`T.SimdVF()`，无线程概念）（auto_schedule: **yes**）

> 整体算子 `IndexedExpertMLPFwdOp` 为 mixed（GEMM 主导 kernel 走 Cube 路径，本设计不涉及）；本函数 `_route_stats` 为向量型路由统计 kernel，按用户确认的编程模式指令取 simd。kernel 模块 docstring 必须声明 `programming_mode: simd` 与 `auto_schedule: yes`（§3.3 伪代码头部）。

### 2.2 选型理由

1. **算子特征**：单 block、单核、纯 int32 统计（无 matmul、无浮点规约）；核心扫描段为循环携带依赖的标量链（§1.6.2 #3 主选 + §1.6.4 裁决），**无线程并行可展开面**——SimtVF 的线程域在此无益：`T.Parallel` 在 SimtVF 下逐线程标量化（CG-2026-0002 的底层机制，风险背景），且线程域引入 32KB UB 扣减（248KB→216KB，`CONST-capacity-a5`）；`T.serial` 在 SimtVF 线程域内按 SIMT 语义被每线程冗余执行，需 tid 守卫才能单流化（额外复杂度）。
2. **向量能力需求集中在 fill/搬运**：SimdVF 提供显式 2048-bit 向量控制（T.fill 向量广播，64 int32 lane/指令），且无 SimtVF 的线程上下文开销。
3. **同代 wrapper 先例**：`examples/TileOPs/src/tileops/_kernels/reduction/logsumexp_simd/logsumexp_simd_kernel/perf_opt/_logsumexp_kernel_single.py`（TileOPs SimdVF 归约重实现：`@tilelang.jit(target="ascend")` 工厂面 + `programming_mode: simd` docstring 惯例 + 56 AIV 核数注释 + padded 分配/部分范围拷出形态）。
4. **与重设计方向一致**：SimdVF 块禁止直接访问 GM（`testing/ascend/analysis/test_ascend_vf_checker.py`："SIMD_VF blocks cannot access global memory"），强制全量 UB staging——正是 §0.6 R1 确定性 staging 方案的结构前提。
5. **备选变体兼容**（v1 补充）：§1.6.4 批式向量备选的全部原语（vgather2/vscatter/vselr/vci）同为 SimdVF 域内向量 op——两变体共享同一模式与 wrapper 调用面，裁决翻转无模式迁移成本。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| VF 入口 | `with T.SimdVF():` 仅包住向量段（`T.fill` 零填）；扫描段为 VF 域外核内标量代码（标量核执行，`testing/ascend/language/test_tilelang_ascend_fill.py` 模块 docstring：「VF 块外的逐元素标量工作运行于标量核」）。备选变体（§1.6.4 ②）则将批循环整体置于 VF 域内 |
| 内存分配 | `T.alloc_shared`（UB）：ids_ub + meta_ub（§4.3/§4.5）；无 L1/L0 分配（无 Cube 段） |
| 同步 | auto_schedule: yes（AutoSchedule 自动插 MTE2→V→标量→MTE3 的 pipe 同步，§7） |

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 优化后的整数扫描式（OPT-1/2/3 已并入）：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `ids_ub[token, slot] = expert_ids[token, slot]`（全量 [T,K]） | GM→UB 搬入 |
| 2 | `meta_ub[j] = 0`，j ∈ [0, SIZE_PAD) | 向量零填（含 lists 未写槽，OPT-3） |
| 3a | `e = ids_ub[token, slot]`；`rank = meta_ub[e]`；`meta_ub[e] = rank + 1` | running 计数（在线） |
| 3b | `meta_ub[E + token·K + slot] = rank` | ranks 段（route = token·K + slot，OPT-1） |
| 3c | `if rank < T: meta_ub[E+R + e·T + rank] = token·K + slot` | lists 段（T 槽位守卫） |
| 4 | `metadata[j] = meta_ub[j]`，j ∈ [0, SIZE) | UB→GM 单次 bulk 搬出（OPT-2） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | GM→UB 全量 | `T.copy(expert_ids, ids_ub)` | 2-D [T,K] int32 整缓冲 | simd（VF 域外 DMA 任务） |
| 2 | 向量零填 | `T.fill(meta_ub, T.int32(0))`（在 `with T.SimdVF():` 内） | 整缓冲，SIZE_PAD 64-lane 整除 | simd |
| 3 | 标量顺序扫描 | 嵌套 `for token in T.serial(T) / for slot in T.serial(K)` + UB 标量读改写 + 运行时 `if` | 边界为工厂编译期常量 | simd（VF 域外标量段） |
| 4 | UB→GM bulk | `T.copy(meta_ub[0:SIZE], metadata)` | 1-D 区间访问，extent=SIZE（字节粒度尾区） | simd（VF 域外 DMA 任务） |
| — | kernel 骨架 | `with T.Kernel(1):`（一维、无 threads） | num_blocks=1 | — |
| — | JIT 装饰 | `@tilelang.jit(target="ascend", out_idx=[])` | 原位写、无返回值（源同款 out_idx=[]） | — |

> §1.6.4 备选变体的 API 映射（若裁决翻转后启用）：批循环 `T.serial(nb)` 置 SimdVF 域内，装载/取数/散写走 `T.simd.vld/vgather2/vscatter/vselr/vci/vsts/vsel/vcmp/vadd/vand/vor/vdup`（逐原语行号佐证见 §1.6.4 ② 与 §1.6.2 #3）——主选与备选共享 kernel 骨架 / 双端 `T.copy` / `T.fill`，仅扫描段形态不同。

### 3.3 计算伪代码

```python
"""_route_stats — IndexedExpertMLPFwdOp 路由统计 kernel（Ascend A5 重实现）。

programming_mode: simd
auto_schedule: yes
"""
import functools

import tilelang
import tilelang.ascend.language as T


def _route_metadata_size(num_experts, num_routes, num_tokens):
    """Per-expert route counts, per-route ranks, then up to num_tokens routes listed per expert."""
    return num_experts + num_routes + num_experts * num_tokens


@functools.lru_cache(maxsize=32)
def _route_stats(num_tokens: int, top_k: int, num_experts: int):
    """Build the single-block route-statistics kernel (deterministic stable ranks)."""
    num_routes = num_tokens * top_k                       # R
    size = _route_metadata_size(num_experts, num_routes, num_tokens)
    size_pad = ((size + 63) // 64) * 64                   # 64-lane fill 整除（§4.3）
    rank_base = num_experts
    list_base = num_experts + num_routes

    @tilelang.jit(target="ascend", out_idx=[])
    def _kernel():
        @T.prim_func
        def main(
            expert_ids: T.Tensor((num_tokens, top_k), "int32"),
            metadata: T.Tensor((size,), "int32"),
        ):
            with T.Kernel(1):
                ids_ub = T.alloc_shared((num_tokens, top_k), "int32")
                meta_ub = T.alloc_shared((size_pad,), "int32")
                T.copy(expert_ids, ids_ub)                 # GM -> UB (MTE2)
                with T.SimdVF():
                    T.fill(meta_ub, T.int32(0))            # 向量零填（OPT-3）
                for token in T.serial(num_tokens):         # 标量扫描段（VF 域外）
                    for slot in T.serial(top_k):
                        expert = ids_ub[token, slot]
                        rank = meta_ub[expert]
                        meta_ub[expert] = rank + 1
                        route = token * top_k + slot        # OPT-1：免 div/mod
                        meta_ub[rank_base + route] = rank
                        if rank < num_tokens:
                            meta_ub[list_base + expert * num_tokens + rank] = route
                T.copy(meta_ub[0:size], metadata)          # UB -> GM (MTE3, bulk)

        return main

    return _kernel
```

> 迁移前 `@T.prim_func` 签名的 dtype 采用第二位置参数形式 `T.Tensor((T, K), "int32")`（与 `examples/ascend/` 一致，源即此形态，无需改写）。参数名与顺序（expert_ids, metadata）保持不变。wrapper 调用面 `_route_stats(T, K, E)()`（config 为空 dict，threads 已按 K3/K9 移除）。

### 3.4 API 可行性确认

| API | 用途 | 来源确认 | 验证状态 |
|-----|------|----------|---------|
| `T.Kernel(1)` | 一维单 block launch | tilelang-ascend skill §2.1（无 threads 参数） | 已验证（skill + 全部 ascend 例程形态） |
| `T.copy`（GM→UB / UB→GM） | 双端搬运 | tilelang-ascend skill §6.1；`examples/TileOPs/src/tileops/_kernels/reduction/logsumexp_simd/logsumexp_simd_kernel/perf_opt/_logsumexp_kernel_single.py`（部分范围拷出先例：拷 N 列 of N_pad） | 已验证 |
| `T.fill(buf, val)` in SimdVF | 向量零填 | `testing/ascend/language/simdvf/test_simdvf_fill.py`（整缓冲/2-D/流水槽位/对齐子区域回归） | 回归覆盖 fp32；int32 走同一 LowerFill 降级路径（dtype 取自 buffer，`src/ascend/transform/ascend_simdvf_lower_parallel.cc` MakeReg/LowerFill），fallback 见 §9.2 R1 |
| `T.SimdVF()` | 向量域 | tilelang-ascend skill §2.3；logsumexp_simd 先例 | 已验证 |
| `T.serial` 标量循环 + UB 标量读改写 | 顺序扫描 | `examples/ascend/deepgemm/kernels/epilogue.py`（SimdVF 内/外 T.serial + 运行时索引 UB 访问）；`examples/ascend/deepgemm/kernels/transform_sf.py`（serial 循环变量索引的 UB 标量读写） | 已验证（运行时载入值作索引的先例深度略浅，列 §9.2 R2 首查项） |
| 运行时 `if`（标量分支） | rank<T 守卫 | 源 kernel 同款（T.Parallel 内 if）；`examples/ascend/deepgemm/kernels/epilogue.py` T.serial + 运行时边界 | 已验证 |
| `@tilelang.jit(target="ascend", out_idx=[])` | 原位写 JIT | 源同款 out_idx=[]；A5 形态见 logsumexp_simd（out_idx=[1] 输出式对照） | 已验证 |
| `T.simd.vgather2 / vscatter / vselr / vci`（v1 新增行） | §1.6.4 备选变体原语 | `tilelang/ascend/language/simd.py` L1038/L1159/L1184/L1452（docstring 语义：per-lane gather/scatter/索引选择/lane ramp）；`tilelang/ascend/language/__init__.py:69`（`T.simd.*` 导出）；codegen `src/ascend/codegen/codegen_ascend.cc:2294`（vgather2，`__ubuf__` 基址）/`:2468–2480`（vscatter）；用例 `examples/ascend/deepgemm/kernels/transform_sf.py（L286）`、`testing/ascend/language/test_tilelang_ascend_buffer_reuse.py（L270）`、`testing/ascend/auto_schedule/test_access_footprints.py（L527）` | 存在性已验证（吞吐未实测——§1.6.4 裁决对象） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 一维 Kernel / `threads=` 禁用 | Yes | `T.Kernel(1)`，无 threads；线程域不使用（§2.2） |
| C0 分形对齐 / GEMM 非整除 | No | 纯 int32 Vector kernel，无 `T.gemm` |
| UB 容量（248KB，SimdVF 无 32KB 扣减） | Yes | 70,656B ≤ 228,556B（90% 折扣口径，§4.5）；构建期容量 guard 与降级分支见 §5.4 |
| OOB padding（`pad_value`/`data_select`） | No（主选不使用；§1.6.4 备选 ids 平铺尾段零填由分配 padding + valid 掩码承接，非 OOB padding API） | 主选无越界读需求；尾区用字节粒度 extent 拷出（GM↔UB DMA 支持字节粒度行，tilelang-ascend skill §6.1；UB→GM 方向尾区状态见 §5.3 首验项） |
| SimdVF 块内 GM 访问禁令 | Yes | 两端 `T.copy` 均置于 VF 域外；VF 域内仅 `T.fill`（UB）（`testing/ascend/analysis/test_ascend_vf_checker.py`）；备选变体（§1.6.4）VF 域内仅 UB 访问（vgather2/vscatter 基址均为 UB），同一约束满足 |
| `T.Parallel` 须在 VF 域内 | Yes（规避） | 本设计无 T.Parallel（fill 由 T.fill 向量降级承接） |
| fill 对齐/整除 | Yes | SIZE_PAD 64-lane 整除 + 整缓冲 32B 基对齐（`src/ascend/transform/ascend_simdvf_lower_parallel.cc` CanLowerFillRegion / ValidateFillBaseAlignment） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 维度 | `T.Kernel(1, threads=128)` | `T.Kernel(1)`（一维、无 threads） | §0.6 R2 |
| 秩分配 | `T.atomic_add(metadata[e], 1, return_prev=True)`（SimtVF GM 原子，A5 有等价原语但弃用） | 确定性顺序扫描（UB 标量读改写） | §0.6 R1 |
| 跨段同步 | `T.sync_threads()` | 舍弃（AutoSchedule pipe 同步） | §0.5 条目 5 |
| 写路径 | GM 分散 4B scatter 直写 | UB staging + 单次 MTE3 bulk | §0.5 条目 6 / OPT-2 |
| 内存分配 | GM 原地（无 SMEM） | `T.alloc_shared`（UB）staging | §4 |
| 工厂参数 | `_kernel(threads)` + `{"threads": 128}` | `_kernel()`（无参）+ `config = {}` | K3/K9（wrapper 已落地） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/src/tileops/_kernels/reduction/logsumexp_simd/logsumexp_simd_kernel/perf_opt/_logsumexp_kernel_single.py` | 高度相似（同代 TileOPs SimdVF 重实现） | `@tilelang.jit(target="ascend")` 工厂面、`programming_mode: simd` docstring、padded 分配 + 部分范围拷出、T.simd 显式向量 + 标量控制流共存、56 AIV 核数注释口径 |
| `examples/ascend/deepgemm/kernels/transform_sf.py` | 高（标量+向量混合） | `T.serial` 循环内 UB 标量读写（`group_ranges_ub[group_idx, 0]`）、SimdVF 域内外混合结构；**`S.vgather2` 向量 gather 用例（L286，§1.6.4 备选原语佐证——v0 曾引此文件标量先例而漏此用法，v1 已补全）** |
| `examples/ascend/deepgemm/kernels/epilogue.py` | 高（标量循环 + VF） | SimdVF 内/外 `T.serial` + 运行时索引 UB 访问 + 标量分支 |
| `examples/ascend/example_gemm_splitk.py` | 中（设计取向） | deterministic 有序归并 vs atomic 双模式——确定性优先先例 |
| `testing/ascend/language/simdvf/test_simdvf_fill.py` | API 佐证 | T.fill 于 SimdVF 的向量降级与对齐约束 |
| `testing/ascend/language/simtvf/test_simtvf_atomics.py` | API 佐证（回退变体） | SimtVF GM 原子 + return_prev |
| `testing/ascend/language/test_tilelang_ascend_buffer_reuse.py` | API 佐证（v1 新增） | `T.simd.vgather2` 直接用例（L270） |
| `examples/ascend/example_vecadd.py` | 基础形态 | SimdVF 基本结构 |

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| expert_ids | (T, K) | int32 | 行主连续；调用方契约取值 ∈ [0, E)（manifest `requires: in_range(0, E)`，kernel 不校验值，与源一致） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| metadata | (E + T·K + E·T,) | int32 | 原位写回（out 模式，`out_idx=[]`）；全缓冲确定性（未写 lists 槽为零，OPT-3） |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| ids_ub | (T, K) | int32 | UB | 路由表驻留（≤2KB），扫描段标量读取 |
| meta_ub | (SIZE_PAD,) | int32 | UB | 元数据全量 staging（counts/ranks/lists 三段 + 零填），SIZE_PAD = ceil(SIZE/64)·64 |

### 4.4 内存搬运路径

```
GM[expert_ids T×K] --T.copy(MTE2)--> UB[ids_ub]
UB[meta_ub] <--T.fill(SimdVF 向量广播, int32 0)--        （与 MTE2 无依赖，可重叠）
UB[ids_ub] --标量读--> 扫描段 --标量读改写--> UB[meta_ub]
UB[meta_ub[0:SIZE]] --T.copy(MTE3, extent=SIZE 字节粒度)--> GM[metadata]
```

无 L1/L0 参与（纯 Vector）；无 ND2NZ/fractal 转换（无 Cube 段）。

### 4.5 UB 内存预算

| Buffer | Shape（最坏 = ds-v3-t64: T=64,K=8,E=256） | dtype | 大小 (Bytes) |
|--------|------------------------------------------|-------|-------------|
| ids_ub | (64, 8) | int32 | 2,048 |
| meta_ub | (17,152,)（SIZE_PAD=SIZE，天然 64 整除） | int32 | 68,608 |
| **总计** | | | **70,656** / UB 预算 228,556B（248KB × 90%，SimdVF 无 32KB 线程上下文扣减，`CONST-capacity-a5` assumption）= **31.0%** |

其余 harness shape：ds-v3-t32（34,816B staging）、qwen3-235b-t32（17,920B）、测试小 shape（≤1.6KB）。备选变体（§1.6.4 ②）UB 预算与主选相同（向量临时量驻留 VRF，不占 UB）。

### 4.6 动态轴定义

**无**。`(T, K, E)` 由 `lru_cache` 工厂特化为编译期常量（源同款机制，§0.5 条目 7 保留）；核内全部循环边界静态（core-split-strategy.md §2.1「核内串行循环边界必须静态」满足）。harness 展开：4 条 manifest 行中 3 条 grouped（T>1）→ 3 个 (T,K,E) 特化 × 2 dtype（本 kernel 与 dtype 无关，同特化复用）+ 测试小 shape 若干。

### 4.7 JIT 配置

```python
@tilelang.jit(
    target="ascend",          # 规范入口
    out_idx=[],               # 原位写（metadata 为入参），无返回值——与源一致
)
```

pass_configs 默认（AutoSchedule 开启，不传 `TL_ENABLE_AUTO_SCHEDULE`）。

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector（int32 统计 + DMA 搬运；无 matmul → 不占 Cube）。

**判定依据**: 全部计算为标量统计与向量零填；无 `T.gemm`；codegen 自动判 `__vector__` kernel 落 AIV（tilelang-ascend skill §3.1）。

### 5.2 Block 划分

无空间分块：元数据工作集全量驻留 UB（§4.5，31% 预算），单遍完成。`num_blocks = 1`（`T.Kernel(1)`），与源一致（单 block 语义）。

### 5.3 约束分析

- **C0 分形对齐**: 不适用（无 GEMM / 无 Cube 段）。
- **UB 容量**: 70,656B ≤ 228,556B（90% 折扣口径，`CONST-capacity-a5`）✓；无 multibuffering 版本膨胀项（单遍直线依赖，无 `T.annotate_buffer_versions` 需求——AutoSchedule 单版本即可）。
- **L0 容量**: 不适用。
- **fill 整除/对齐**: SIZE_PAD 64-lane 整除 ✓；整缓冲基址 32B 对齐（UB 分配天然满足）✓。
- **拷出尾区**: extent=SIZE 的 1-D 区间（SIZE 非 32B 倍数的小 shape）。**UB→GM 方向非 32B 整数倍 extent 的文档化状态：未单列**（tilelang-ascend skill §6.1 的 byte-granular rows 表述针对 GM→UB 方向；logsumexp 先例行宽 32B 对齐，未直接覆盖非对齐尾区）——列为 **L0 编译冒烟首验项**（§8.3 第 1 项含 (17,2,18) → SIZE=358 int = 1,432B 非 32B 倍数用例）；fallback 见 §9.2 R7（SIZE 向下 32B 对齐段 bulk + 尾段标量小段拷贝，GM 侧多分配不可行——wrapper 契约冻结 size）。

### 5.4 注意事项

- **非整除 fill**：SIZE % 64 ≠ 0 的测试 shape（如 T=4/E=8/K=2 → SIZE=48；T=17/E=18/K=2 → SIZE=358）由 SIZE_PAD 覆盖（fill 整个 padded 缓冲，拷出仅 SIZE 个）。
- **构建期容量 guard（harness 外防御）**：若 `size_pad·4 + R·4 > 228,556B`（假想的超大 T·E 组合，harness 8 条 manifest 调用与全部测试 shape 均不触发，最大 68.6KB），工厂 host Python 层分支降级为 **direct-GM 变体**：ids 仍 staging，扫描段直接读改写 GM metadata（counts/ranks/lists 逐槽写、lists 未写槽回到源语义的未初始化），无全量零填与 bulk 搬出——两个变体各自核内边界静态。该分支为规格完备性防御，非主路径。

### 5.5 分核策略（grid 划分 + AIC/AIV 分工 + 同步策略）⭐

> 三要素判定标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（§2.1 designer 要求）。

- **① grid 划分（含核数依据）**: `num_blocks = 1`（`T.Kernel(1)`；一维）。核数实查（2026-10-29，本机）：
  ```python
  import torch
  props = torch.npu.get_device_properties()   # device: Ascend950PR_9579
  props.cube_core_num, props.vector_core_num  # → 28, 56
  ```
  实查值 **cube_core_num = 28 / vector_core_num = 56**（与 `CONST-arch-a5` 2026-10-16 试点实测一致，status: measured）。block 数与核数关系：单 block 占 1 个 AIV（1/56 单波，无多波、无 persistent 化——单任务直线依赖，无遍历复用面）。多 block 候选（分块归并）已在 §1.6.0 R3/R4 定量淘汰（2 遍 ids + 2 次跨核同步 + 块间前缀串行链，换取 <0.6% op 占比的收益）。
- **② AIC/AIV 分工**: 纯 Vector kernel（无 `T.gemm`）→ codegen 自动判 `__vector__`，全部工作落 **1 个 AIV**：MTE2 搬入（ids）→ SimdVF 向量零填（meta）→ 标量扫描（标量核）→ MTE3 搬出（metadata）。**AIC 不参与**（本 launch 的 27 AIC 空闲，交由 op 内后续 GEMM kernel 使用——本 kernel 是 op 流水线的 metadata 生产者，其单核延迟直接串在关键路径上，§1.6.0 估算 ≈2.5–4% 占比）。无 MixedKernel / sid / dual_copy（无 Cube 段）。备选变体（§1.6.4 ②）同分核拓扑：1 AIV，批循环在 SimdVF 域内执行。
- **③ 同步策略（auto|manual 声明）**: **`auto_schedule: yes`**（默认，不传 `TL_ENABLE_AUTO_SCHEDULE`）。数据依赖边界（AutoSchedule 据此自动插跨 pipe flag）：`T.copy(ids)`〔MTE2〕→ 扫描段〔标量，读 ids_ub〕；`T.fill`〔Vector，写 meta_ub〕→ 扫描段〔读改写 meta_ub〕；扫描段 → `T.copy(out)`〔MTE3，读 meta_ub〕。MTE2 与 fill 互不依赖，可被 AutoSchedule 重叠。无手动 `T.ascend_set_flag/wait_flag`，flag id 预算（核内 8 / 跨核 16，`CONST-arch-a5`）零消耗。**核内串行循环边界静态**：num_tokens / top_k / size / size_pad 均为工厂编译期常量（§4.6）。无 `T.Stage` / `T.Task` 调度提示（单遍直线结构无需）。

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| token 维 | 核内串行 | `T.serial(num_tokens)` | 顺序依赖（每专家 running counter 链，§1.6.2 #3）；边界静态 |
| slot 维 | 核内串行（嵌套内层） | `T.serial(top_k)` | 同上；嵌套形式消除 div/mod（OPT-1） |
| 零填 | 向量化（VF 域内） | `T.fill`（SimdVF 向量广播降级，内部 vreg 循环 64 lane/步） | §1.6.2 #1 |
| 搬入/搬出 | DMA 任务（无循环） | `T.copy` | 全量单发 |
| 元素级 | 无 `T.Parallel` | — | 扫描主体主选维持标量（§1.6.2 #3 量化依据 + §1.6.4 实验裁决备选）；fill 由 T.fill 承接 |

### 6.2 循环伪代码

```python
with T.Kernel(1):                                   # 一维 grid，1 block（§5.5）
    ids_ub = T.alloc_shared((T_, K), "int32")
    meta_ub = T.alloc_shared((SIZE_PAD,), "int32")
    T.copy(expert_ids, ids_ub)                      # MTE2（与 fill 无依赖）
    with T.SimdVF():
        T.fill(meta_ub, T.int32(0))                 # 向量零填
    for token in T.serial(T_):                      # 标量扫描（VF 域外）
        for slot in T.serial(K):
            expert = ids_ub[token, slot]            # UB 标量读
            rank = meta_ub[expert]                  # UB 标量读（数据依赖索引）
            meta_ub[expert] = rank + 1              # UB 标量写
            route = token * K + slot
            meta_ub[E + route] = rank
            if rank < T_:                           # 运行时标量分支
                meta_ub[E + R + expert * T_ + rank] = route
    T.copy(meta_ub[0:SIZE], metadata)               # MTE3 bulk（extent=SIZE）
```

> 备选变体的循环结构（批循环替代 token/slot 双层循环）见 §1.6.4 ②——两变体循环结构互斥，裁决后择一冻结；Stage 3 按主选实现。

### 6.3 流水线优化

不使用 `T.Pipelined`（单遍直线依赖：搬入→零填→扫描→搬出各执行一次，无迭代 tile 复用面）；MTE2（ids 搬入）与 `T.fill`（meta 零填）互不依赖，其重叠由 AutoSchedule 自动调度（`auto_schedule: yes`，§7）。不使用 `T.Persistent`（grid=1 无工作遍历循环）。

### 6.4 尾块处理

- fill 尾区：SIZE_PAD 覆盖（64-lane 整除，§5.3）。
- 拷出尾区：`T.copy(meta_ub[0:SIZE], metadata)` 的 extent = SIZE（字节粒度，padded 尾 [SIZE, SIZE_PAD) 不落 GM）；非 32B 尾区首验项与 fallback 见 §5.3 / §9.2 R7。
- 扫描边界：`rank < T` 守卫覆盖 count_e > T 的 duplicate-fallback；`counts=0` 专家由零填保证。

## 7. 同步策略

### 7.1 同步模式

**模式**: auto_schedule: **yes**（AutoSchedule 自动同步；不设 `TL_ENABLE_AUTO_SCHEDULE: False`）

### 7.2 同步点说明

auto 模式——依赖边界（AutoSchedule 自动插 flag，无需手动配对）：

| 依赖边 | 生产者 → 消费者 | 说明 |
|--------|----------------|------|
| ids 搬入 → 扫描 | MTE2（ids_ub 写）→ 标量段（ids_ub 读） | 跨 pipe RAW |
| 零填 → 扫描 | Vector fill（meta_ub 写）→ 标量段（meta_ub 读改写） | 跨 pipe RAW/WAR |
| 扫描 → 搬出 | 标量段（meta_ub 写）→ MTE3（meta_ub 读） | 跨 pipe RAW |
| （可选重叠）MTE2 ∥ fill | 互不依赖 | AutoSchedule 自由度 |

grid=1 单核 → 无跨核同步；核内无线程域 → 无 `asc_syncthreads` 需求（源 `T.sync_threads` 已随 §0.5 条目 5 舍弃）。flag id 预算（核内 8 / 跨核 16）：零消耗。备选变体（§1.6.4 ②）同步拓扑相同（批循环为 Vector pipe 内串行，依赖边集不变）。

### 7.3 pass_configs 配置

```python
pass_configs = {}   # 默认（AutoSchedule 开启）；完全手动核才需要 TL_ENABLE_AUTO_SCHEDULE: False
```

## 8. 验证方案

### 8.1 Golden 函数

> golden 以 **§0.1 源算子语义**为唯一依据实现：规范实现取双射自由度下的**路由顺序稳定双射**（§0.1 论证：唯一无需硬件时序即可定义的规范选择），另附不依赖稳定序选择的**性质断言**（源契约校验）——两者合起来强于任何单一形式，且不复刻 NPU 实现细节（staging / fill / DMA 路径均不出现于 golden）。
>
> **设备约定（v1 新增）**：golden 张量统一建在 `expert_ids.device`（AGENTS.md golden-on-NPU 惯例：golden 以 torch 在 NPU 上计算）——`torch.zeros(size, dtype=torch.int32, device=expert_ids.device)`；`check_route_stats_contract` 的 `bincount/nonzero` 等派生张量随 `metadata`/`expert_ids` 同设备。int32 逐位相等的对照方式下，与「对照前统一 `.cpu()` 后执行」两形态等价（设备无关精确算术），Stage 3 任选其一但**须全链路统一**——避免 NPU kernel 输出与 CPU golden 混用时的设备不匹配报错（int32 无精度风险，纯设备一致性问题）。

```python
def golden_route_stats(expert_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Stable-order route statistics (canonical realization of §0.1 semantics).

    expert_ids: [T, K] int32 (values in [0, E)); returns int32 [E + T*K + E*T].
    Device: golden stays on expert_ids.device (AGENTS.md golden-on-NPU convention).
    """
    T_, K = expert_ids.shape
    R = T_ * K
    size = num_experts + R + num_experts * T_
    metadata = torch.zeros(size, dtype=torch.int32, device=expert_ids.device)
    flat = expert_ids.reshape(-1).to(torch.int64)
    for route in range(R):
        e = int(flat[route])
        rank = int(metadata[e])
        metadata[e] = rank + 1                                # running count
        metadata[num_experts + route] = rank                  # ranks 段
        if rank < T_:
            metadata[num_experts + R + e * T_ + rank] = route # lists 段（守卫）
    return metadata


def check_route_stats_contract(metadata: torch.Tensor, expert_ids: torch.Tensor, E: int) -> None:
    """§0.1 property assertions (bijection-order agnostic; same-device tensors)."""
    T_, K = expert_ids.shape
    R, base_r, base_l = T_ * K, E, E + T_ * K
    flat = expert_ids.reshape(-1).to(torch.int64)
    assert torch.equal(metadata[:E], torch.bincount(flat, minlength=E).to(torch.int32))
    for e in range(E):
        routes = (flat == e).nonzero(as_tuple=True)[0]
        ranks = metadata[base_r + routes].to(torch.int64)
        assert sorted(ranks.tolist()) == list(range(len(routes)))     # 双射
        for rank in range(min(len(routes), T_)):
            assert int(metadata[base_l + e * T_ + rank]) == int(routes[rank])  # 配对
```

op 级 golden（上层消费闭环，Stage 3/5 使用）：`ref_routed_experts`（`examples/TileOPs/workloads/moe.py`，移植自源仓）。

### 8.2 精度标准

本算子为 **int32 精确语义**——无浮点容差概念，对照标准为**逐位相等**：

| dtype | atol | rtol | 对照方式 |
|-------|------|------|----------|
| int32（本算子唯一 dtype） | 0 | 0 | `torch.equal`（全缓冲，含未写槽零值；对照前统一设备，§8.1）；性质断言（§8.1 check_route_stats_contract）兜底双射自由度 |

> 浮点 dtype 分档容差（fp8 1e-1 / fp16·bf16 1e-2 / fp32 1e-3）不适用于本 kernel；op 级输出的浮点容差由上层算子的验证方案管辖（`_moe_compare`：atol=rtol=1e-2 + 归一化平方误差 ≤1e-4，见 `examples/TileOPs/tests/ops/test_indexed_expert_mlp.py`）。

### 8.3 L0 门槛测试计划（Stage 3 首跑必过）

1. **编译冒烟**：3 个 manifest grouped 特化（(32,8,256) / (64,8,256) / (32,8,128)）+ 测试 shape 集（(4,2,8) / (17,2,18) / (8,4,16) / (2,2,8)）JIT 编译通过（`target="ascend"`）——**含 (17,2,18) → SIZE=358（1,432B，非 32B 整数倍）的 UB→GM 尾区行为首验**（§5.3 首验项；命中即启用 §9.2 R7 fallback）。
2. **精确对照**：对 §8.3 用例 + 随机路由（manifest 形态：每行 K 个互异专家）跑 kernel，`torch.equal(kernel_out, golden_route_stats(...))` 全缓冲逐位相等（设备统一，§8.1）。
3. **性质断言**：`check_route_stats_contract`（双射自由度下的源契约）。
4. **确定性回归**：同一输入连跑 3 次，输出逐位一致（重设计 R1 的确定性合约）。
5. **op 级闭环**：`examples/TileOPs/tests/ops/test_indexed_expert_mlp.py` grouped 用例（small_route 4 模式 × 2 dtype、`test_dims_off_the_tile_grid`、`test_manifest_shaped_call_matches_reference`）。
6. **边界专项**：duplicate-fallback（count_e > T）、second-route-group（T=17 跨 block_m=16 组边界）、K=1、E=1、全同专家、token 内重复专家、零 route 专家。

完整分层套件（L1/L2/Boundary 枚举）由 `tilelang-op-develop` 阶段交付，不在此展开。

## 9. 风险点与注意事项

### 9.1 已知约束

- `T.Kernel` 一维、`threads=` 禁用（TypeError）——本设计 `T.Kernel(1)` 无 threads。
- SimdVF 块内禁止 GM 访问（VFChecker）——全部 GM 访问在 VF 域外（备选变体的 vgather2/vscatter 基址均为 UB，§1.6.4 ②）。
- `T.Parallel` 必须在 VF 域内——本设计无 T.Parallel。
- `T.fill` 区域须 64-lane 整除 + 32B 基对齐——SIZE_PAD 处理。
- UB 248KB（SimdVF 无扣减；90% 折扣 228,556B）——最坏 70,656B（31%）。
- 核内串行循环边界必须静态——工厂特化保证（§4.6；备选变体的 nb = ceil(R/64) 同为工厂常量）。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| R1: `T.fill` int32 降级异常 | fill 值类型/整除不满足 | 编译失败（ValueError "32-byte-aligned" 或类型错误） | SIZE_PAD 已保证整除；值用 `T.int32(0)` 显式定型；若 int32 fill 仍被拒（回归仅覆盖 fp32），fallback：SimdVF 内 `for i in T.Parallel(size_pad): meta_ub[i] = 0`（自动向量化，`testing/ascend/language/simdvf/test_simdvf_fill.py` 第 3 用例形态）或显式 `T.simd.vdup` + `vsts` 循环 |
| R2: 数据依赖标量 UB 访问（`meta_ub[expert]`，expert 为运行时载入值） | 前端/codegen 对非仿射标量索引限制 | 编译失败 | 先例支持（`examples/ascend/deepgemm/kernels/epilogue.py` 运行时 row 索引、`examples/ascend/deepgemm/kernels/transform_sf.py` 循环变量索引——TIR BufferLoad 任意 PrimExpr 索引合法）；若命中未文档化限制：fallback = 扫描段直接读改写 GM metadata（ids 仍 staging），语义不变，OPT-2 收益退化为分散写——Stage 3 首查项，发现即报 conductor |
| R3: 原子回退变体（若 Stage 4 启用） | SimtVF GM RAW 屏障检测覆盖性未文档化 + int32 原子无本仓回归 | counts 竞态错误 | 启用前先在 `testing/ascend/` 补最小复现（清零→原子跨线程屏障 + int32 fetch-add）；本设计主路径不依赖 |
| R4: 容量 guard 触发（harness 外大 T·E） | size_pad·4 + R·4 > 228,556B | 编译期 UB 超限 | 构建期分支降级 direct-GM 变体（§5.4） |
| R5: expert_ids 值越界 | 调用方违约（e ≥ E / 负值） | UB 越界访问（未定义行为，与源 GM 越界同） | 调用方契约（manifest `in_range(0, E)`）；wrapper 与源一致不校验值 |
| R6: 编译期膨胀 | size_pad 极大时 fill 向量循环展开 | 编译变慢 | harness 最大 268 向量步，无风险；防御由 R4 guard 间接限制 |
| R7: **UB→GM 非 32B 尾区**（v1 新增，首验项） | SIZE·4 非 32B 整数倍的小 shape（(17,2,18) → SIZE=358 → 1,432B；文档未单列，§5.3） | 编译失败或尾区行为未定义（L0 冒烟即暴露） | fallback：**SIZE 向下 32B 对齐段 bulk 拷出（`floor(SIZE·4/32)×8` 个元素）+ 尾段（≤7 元素，≤28B）标量小段拷贝**（VF 域外标量核直写 GM，量级无性能面）；GM 侧多分配不可行（wrapper 契约冻结 size）——Stage 3 命中即切换并回报 conductor |
| R8: **备选变体原语吞吐不达预期**（v1 新增） | §1.6.4 裁决启用备选后 vselr/vgather2/vscatter 实测吞吐低于 51 档假设 | 备选劣于主选（裁决即翻转回主选） | 裁决阈值双条件（kernel >20% 且 op ≥1%）已含此情形——不满足即维持主选；实测常数回写 pattern-library（§1.6.4 ③ 回写路径） |

### 9.3 特殊场景处理

- **CG-2026-0002 能力缺口核对**：已亲自核对 `.agents/evolution/capability-gaps.md`——缺口本体为 SimtVF 16-bit（bf16/fp16）域归约的标量化/编译失败；本算子全 int32、无 16-bit fragment 归约、主路径无 SimtVF 线程域，**不命中**（作为 §0.5 条目 3 弃用原子路径的风险背景记录，无需登记新缺口）。备选变体（§1.6.4）亦全 int32，不命中。
- **非整除/极小 shape**：SIZE=48（T=4/E=8/K=2）等由 SIZE_PAD + 字节粒度 extent 拷出覆盖；尾区风险由 §9.2 R7 兜底。
- **dtype 无关性**：本 kernel 只处理 int32 元数据，与上层 D（fp16/bf16）无关——同 (T,K,E) 特化被 2 个 dtype case 复用（wrapper 层天然如此）。
- **能力缺口登记判定（v1 复核）**：备选变体所依赖的 vgather2/vscatter/vselr 原语**存在**（五重佐证）——不构成「无法实现 / 被迫换算法」型能力缺口；其吞吐未实测属 pattern-library constants 待标定项（§1.6.4 ③ 回写路径处理），**不登记 capability-gaps.md**。

## 10. 交付清单

### 10.1 目录结构

```
examples/indexed_expert_mlp/_route_stats/
├── DESIGN.md            # 本设计文档（本文件，revision 1）
├── verify_equiv.py      # §1.6.1 等价性机器验证脚本（v1 已执行，E1–E5 全部 EQUIV_PASS）
└── _route_stats.py      # 算子实现 + golden + 分层测试（Stage 3 交付）
```

> 本目录为 conductor 工作区（`examples/{project}/{op}/`）；Stage 5 经 `integrate_kernel.py` 将实现接回 TileOPs 包（`examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/`，wrapper 契约 §0.2 已冻结）。

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成（revision 1） | 本设计文档（v1：负向断言修正 + 判定重推 + §1.6.4 实验裁决三件套 + E5） |
| `verify_equiv.py` | 已完成（v1 已重跑） | E1–E5 全部 EQUIV_PASS（§1.6.1 结果表；E5 为 v1 新增备选变体验证） |
| `_route_stats.py` | 待实现（Stage 3） | kernel + golden_route_stats + L0 分层测试；模块 docstring 声明 `programming_mode: simd` / `auto_schedule: yes` |

### 10.3 命名规范

- 项目目录名: `indexed_expert_mlp`（snake_case，conductor 解析）
- 算子目录名: `_route_stats`（与 `@tilelang.jit` 工厂函数名严格一致，不裁剪不改名）
- 实现文件: `_route_stats.py`；golden: `golden_route_stats`（+ `check_route_stats_contract` 性质断言）

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md，revision 1——v0 经 Stage 2 检视回退修订）
2. ✅ 等价性机器验证（verify_equiv.py，E1–E5 全部 EQUIV_PASS，2026-10-29 重跑）
3. ⬜ Golden 函数（§8.1，验证基准——设备约定见 §8.1 v1 注）
4. ⬜ 算子实现（`_route_stats.py`，按 §3.3 主选标量 scan）+ L0 精确对照（int32 逐位相等）+ op 级闭环
5. ⬜ 实验裁决（Stage 4，§1.6.4 ③ 计划——主选 vs 备选 A/B，阈值与回写路径在位）
