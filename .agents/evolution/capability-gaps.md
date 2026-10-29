# 编译器能力缺口登记簿（Capability Gaps）

> 编译器层面自进化机制的登记簿。与 skill 层自进化（`queue.md` / pattern-library）平行：skill 层沉淀「如何在现有能力下绕」，本簿登记「ascend 后端缺什么能力」并驱动补齐。
>
> - **用途**：任务过程中识别出「当前 ascend 后端能力不足以实现所选算法」（无法实现 / 被迫换算法 / 触及性能上限）时，登记具体缺失能力、证据与绕法，跟踪复发次数与补齐状态。
> - **写入者**：任务各阶段 Agent（design 调研与 API 映射 / develop / optimize / review）在得出能力不足结论时**直接追加条目或为既有条目计数**；`tilelang-skill-evolver` 在任务终态蒸馏时查重、冲突消解与升级裁决（与 queue.md 的维护分工同口径）。
> - **证据规则**：与 AGENTS.md「Docs Auto Routing / Negative claims」同口径——须引用具体 `docs/` 路径与限制条款、`testing/`/`examples/` 佐证或复现命令；无法佐证时显式标注「未文档化假设 + 估算依据」。禁止凭记忆或 GPU 先验断言能力缺失。
> - **版本戳**：一切「能力缺失」结论绑定工具链版本（tilelang commit + CANN/设备版本）；工具链升级后相关条目自动待重验，不得沿用旧结论否决新设计。
> - **升级阈值**：同一缺口被 **≥2 个不同任务**独立识别 → `recurring`，**必须推动补齐**：产出能力补齐提案（`proposal` 字段），并在任务终态报告中向用户显式列出（与 Tier 1 的 2 次独立证据阈值同口径）。

## 字段速查

```yaml
gap_id: CG-{YYYY}-{NNNN}           # 唯一 ID，追加时按现有最大号递增
layer: Frontend API（ascend 方言）/ Ascend pass / codegen / bisheng 编译器 / ACL runtime
capability: <缺失能力的具体描述：缺什么 API / 优化 / dtype / 布局 / 同步原语 / 性能上限，越具体越好>
blocked_algo: <受阻的算法/算法族与结论形态：无法实现 / 被迫换算法（换成什么）/ 性能上限（差多少）>
evidence:                          # 证据链，规则同负向断言
  - <docs 路径#限制条款 / 复现命令 / 源码位置 / pattern-library 条目 / 实测数据>
workaround: <当前任务的实际绕法及其代价量化（如"两遍算法替代单遍，访存 +1.8x"），或 none（被迫弃用）>
occurrences: {n}                   # 独立任务识别次数（同一任务反复命中只计 1）
tasks: [<task_id>, ...]
toolchain_stamp: <tilelang commit + CANN 版本 + 设备，或 latest-rebuild>
status: open / recurring / proposal / in-progress / fixed / closed / withdrawn
proposal: <达 recurring 后必填：目标层 + 建议改动（API 签名 / pass 行为）+ 收益量化（引用各任务实测）+ 受影响算子清单 + issue 草稿（上游 tile-ai/tilelang 仓 issue，按其模板）>
created_by: <task_id + 日期>
last_seen: <最近一次识别的任务与日期>
```

## 生命周期

```
open ──(≥2 独立任务识别)──> recurring ──(提案产出)──> proposal ──> in-progress ──> fixed ──> closed
  │                                                                            │
  └── withdrawn（证据被证伪/复核不成立）            fixed 后曾绕行的任务回访评估去绕化 ┘
```

- **追加即计数**：新识别的缺口先查重（本簿 + pattern-library §2），命中既有条目则 `occurrences+1` 并更新 `last_seen`/`tasks`，不新建重复条目。
- **推动补齐的形式**：提案正文落在条目 `proposal` 字段；对外建 issue（遵循 tilelang-fork 贡献流程，见 tilelang-github-operations skill）**前须用户批准**。
- **fixed 判定**：新工具链实测确认能力可用（附版本戳与验证记录）；fixed 后记录过 workaround 的任务可回访评估去绕化收益。
- **与 pattern-library 的边界**：绕法经验（怎么绕、绕的代价、陷阱）按 D 类进 pattern-library §1/§2；本簿只登记缺口本体与补齐跟踪，两处以 `gap_id` / 条目路径互相引用，不复制内容。

---


## Open

### CG-2026-0001

```yaml
gap_id: CG-2026-0001
layer: ACL runtime / CANN 工具链（msopprof 观测层——roofline 与流水图的数据面；非 kernel 执行语义缺口）
capability: >-
  Ascend 950PR&950DT 上 msopprof Roofline 的「Vector 内存单元视图」仅支持 SIMT 视图，暂不支持 SIMD 视图。
  后果：纯 SimdVF（T.SimdVF()，核内无 Cube T.gemm）算子采集时，visualize_data.bin 的
  Roofline 数据块为空（multiple_rooflines 为空列表，仅剩一句 advice 文本），Vector 侧不产出任何
  (AI, Perf) 归因点。同代谱系中 SIMT 算子可产出 GM/L2 视图归因点；Cube/Mix 算子的
  Cube内存单元视图/Cube内存通路视图/Cube Pipeline视图 不受影响。
blocked_algo: >-
  纯 SimdVF reduction / elementwise 算子族的 roofline 口径性能归因无法实现——不是"算法被迫替换"，
  而是"归因通道被迫替换"（结论形态：被迫换归因通道）。代价：拿不到与 SIMT 兄弟算子同口径的
  Ratio(%)/AI(Ops/Byte)/Perf，跨编程模式的 report 列不可比。
evidence:
  - "限制条款（权威，官方文档而非本仓）：msopprof 用户指南 docs/zh/user_guide/msopprof_user_guide.md
    （gitcode.com/Ascend/msopprof，master，访问 2026-10-19）Roofline瓶颈分析图章节「表 6 Ascend
    950PR&950DT系列产品支持Roofline视图情况列表」后的 NOTE 原文：`Ascend 950PR&950DT系列产品的
    Vector内存单元视图仅支持SIMT视图，暂不支持SIMD视图。`（同表：GM/L2视图 对 Vector/Cube/Mix
    三类算子均 √；Cube内存单元视图 / Cube内存通路视图 / Cube Pipeline视图 仅 Cube 与 Mix 算子 √）"
  - "同文档旁证（同一观测层限制）：流水图表「[Warp流水图] 展示各warp的耗时情况 | 当前仅支持SIMT类型」；
    指令流水图使用说明「SIMT VF、SIMD VF内的指令无法展示。」；--aic-metrics 表
    「PcSampling：展示simt算子上板运行的 stall 信息」"
  - "框架侧消费点（本仓，tracked）：examples/TileOPs/tileops/benchmark/msprof_roofline.py
    （collect_roofline_msprof 子进程归因）+ examples/TileOPs/tileops/benchmark/benchmark_base.py
    （_attach_roofline_metrics：roofline 优先、不可得时回退管线占用率，生效源落 ratio_source 元数据）"
  - "真机佐证（task logsumexp_simd，2026-10-19）：纯 SimdVF kernel
    examples/TileOPs/tileops/kernels/reduction/logsumexp_simd/（with T.SimdVF()，溯源：任务期工作区产物，
    未归档）经 msprof op --aic-metrics=Roofline 采集，有效 launches 全部 multiple_rooflines 为空；
    同期 SIMT 兄弟算子同机制的归档 report
    examples/TileOPs/reports/tileops/20261017_111528_067406_LogSumExpFwdOp 为 6/6 case
    msprof_roofline_bin 源（与 VP-2026-0031 证据同源）"
workaround: >-
  周期域回退通道：msprof_detail 管线占用率（最忙 pipe busy ratio，ratio_source: msprof_detail_pipe），
  由 examples/TileOPs/tileops/benchmark/benchmark_base.py 在 roofline 不可得时自动启用。
  代价量化：失去 roofline 口径的 Ratio(%)/AI/Perf 三元组，SimdVF 算子与 SIMT 兄弟算子 report 列不可比。
  补充边界：Cube+SimdVF 混合核（如 examples/ascend/flash_attention/example_mha.py）虽有 roofline，
  但只反映 Cube 半边，SimdVF 侧工作量在 roofline 中不可见——整机瓶颈判定仍须读 msprof_detail。
occurrences: 1
tasks: [logsumexp_simd]
toolchain_stamp: tilelang 0.1.15+gita646c7fee（本仓 HEAD，2026-10-19）+ CANN 9.2.0 + Ascend950PR（npu-smi 25.7.0），2026-10-19
status: open
created_by: logsumexp_simd（Stage 4 optimize，2026-10-19）
last_seen: logsumexp_simd（Stage 4 optimize，2026-10-19）
```

> 备注：本条目只登记缺口本体与官方限制条款；具体实现层的判定机制不入簿。达 `recurring` 后 `proposal` 字段与上游 issue 草稿另拟（上游为 CANN msopprof，非 tile-ai/tilelang，故补齐路径为 CANN 侧需求反馈，须用户批准）。与 `queue.md` 的 VP-2026-0031（tileops Ratio(%) 双通道回填机制）互为消费者/被消费者，两处只以 `gap_id` 互引，不复制内容。

### CG-2026-0002

```yaml
gap_id: CG-2026-0002
layer: Ascend pass / codegen（SimtVF Parallel 逐线程标量化 + reduce 降低路径）/ bisheng 编译器
capability: >-
  SimtVF 无法表达「16-bit 输入域的分片归约（f16max）」：
  (a) bf16——T.reduce_max 作用在 bfloat16 fragment 上时，SimtVF 降低把逐线程串行 max 生成
  AscendC 标量调用 max(bfloat16_t, bfloat16_t)，bisheng 报 `error: call to 'max' is ambiguous`
  （候选 half max(half,half)@__clang_cce_aicore_functions.h:3618 vs
  long long/int/... max(...)@asc/include/simt_api/math_functions.h），编译直接失败；
  (b) fp16——能编译，但 AscendAllReduce<MaxOp, threads,1,0> 与逐线程 max 的 half 形态落到软件路径，
  实测整体比 fp32 域 max 慢 4.2x~5.9x（见 evidence）。
  即：SimtVF 组件下 16-bit 域归约既无合法标量 max、也无硬件 warp-reduce 重载可用。
blocked_algo: >-
  pass-1 max 在 16-bit 输入域计算（SIMD 兄弟成员 OP-F16MAX：其 SimdVF 侧 +4.0% attn-32k /
  +15.5% lm-head-fp16）无法迁移到 SimtVF 成员——结论形态：无法实现（该优化在 SimtVF 组件下被迫弃用，
  只能保留 fp32 fragment 迂回：逐元素 widen + fp32 reduce）。
evidence:
  - "复现命令（本仓，2026-10-19）：
    cd examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp_kernel/perf_opt &&
    TILELANG_LSE_HMAX=1 python _logsumexp_kernel_tiled_simdopt.py --level L0
    → 第 1 例 (33,33000,fp16) PASS（fp16 可编译），第 2 例 (1024,32768,bfloat16) 在
    bisheng 阶段失败：`/tmp/.../tl_kernel.asc:23:15: error: call to 'max' is ambiguous`，
    生成行 `m_ch[0] = max(m_ch[0], xh_frag[...])`"
  - "实测（同文件，TILELANG_LSE_HMAX=1 vs 默认，A/B 交错 3 轮 x 3 采样 do_bench(msprof) 中位数）：
    lm-head fp16 13.201 -> 55.446us（-76.2%）；attn-32k 转 fp16 83.834 -> 496.289us（-83.1%）；
    max|diff|=0（数值等价，纯性能缺口）。同任务 SIMT round-1 E3/E4 已独立复现同型结论
    （fp16 中间链 30-60x / 50x 慢，半精度 AllReduce 软件路径）——perf_opt/opt_log.md。"
  - "对照（SimdVF 侧无此缺口，故该优化在 SIMD 成员成立）：logsumexp_simd/.../perf_opt/opt_log.md
    Round 1 OP-F16MAX（vld NORM 128-lane + vmax，bf16 走 EVEN/ODD 拆分），bf16 无 asc_reduce_max
    重载亦可在 SimdVF 内绕开；SimtVF 无等价绕法（逐线程标量 max 无 16-bit 合法重载）。"
workaround: >-
  SimtVF 成员保留 fp32 fragment 迂回（x_ub -> cast fp32 -> fp32 reduce_max）。代价：pass-1 max 每元素
  一次逐 lane 标量 max（vs SimdVF 2 ops/128 elems），是 attn-32k SIMT 成员 vec 占用率 95.4% 的构成之一；
  无法量化「若可用将省多少」（该组件下候选本身就不可执行）。
occurrences: 1
tasks: [logsumexp]
toolchain_stamp: tilelang 0.1.15+gita646c7fee（本仓 HEAD，2026-10-19）+ CANN 9.2.0 + Ascend950PR，2026-10-19
status: open
created_by: logsumexp（Stage 4 派生实验：SIMT 成员叠加 SIMD 可复用优化，2026-10-19）
last_seen: logsumexp（Stage 4 派生实验：SIMT 成员叠加 SIMD 可复用优化，2026-10-19）
```

> 备注：本条目只登记缺口本体；同任务已把 SIMD 成员的另两项可复用优化（OP-ROWS 持久 tile 行驻留批量化 lse 写回、OP-BLOCKN = tile_n 直保留）成功迁移到 SimtVF 成员并实测增益（attn-32k -16.4%、lm-head -25%），故本缺口不影响「跨模式优化迁移」这一结论的主体，只界定其边界。

## Recurring（达升级阈值，待推动补齐）

（暂无条目）

## Resolved（proposal / in-progress / fixed / closed / withdrawn 归档）

（暂无条目）

> **清零说明（2026-10-16，A5 适配 P0）**：本簿自 tilelang-mlir-fork（npuir/A2A3）整套复制后清零——原 17 条缺口绑定 npuir 工具链（TileLangIR pass/BishengIR/tladapter 分层），不适用于 tilelang-fork（ascend/A5，dav-3510）。schema 与生命周期协议保留，条目由 A5 任务重新登记。
