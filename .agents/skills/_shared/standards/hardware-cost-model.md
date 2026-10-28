# 设计期硬件成本模型与 roofline 估算标准（D-2）⭐

> **单一事实源**：本文件定义**设计期定量性能估算协议**（何时算、怎么算、算完落哪）。硬件常数**数值表**的唯一事实源是 `tilelang-op-optimize` skill 的 [pattern-library/constants.md](../../tilelang-op-optimize/references/pattern-library/constants.md)（条目带版本戳与来源证据，evolver 蒸馏新 D 类常数时追加进该表，Tier 0）——本文件不复制数值，只定义使用协议；两文件同步演进。
>
> **A5 冷启动特别规则**：当前 constants.md 条目全部为 `status: assumption`（源码/文档证据，未经真机实测）——① 容量按 **90% 折扣**预算；② 估算行必须标注「assumption 基估算，常数待标定」；③ 绝对时延/吞吐换算禁止使用编造频率（代价模型纯 cycle 域，无静态频率常数）——设计期只做**相对比较**（候选间比值、结构级量级判断），绝对值以 do_bench(msprof) 实测为准；④ 首批真机标定任务（vecadd copy-floor / GEMM roofline / flag 与 launch 开销）转正条目后逐步恢复绝对估算。

## 1. 估算协议（何时算）

| 触发位置 | 动作 |
|---------|------|
| Phase R R4（硬件亲和性逐候选评估） | 对 R3 存活候选计算**相对量级比较**，写入 §1.6.0 R4 列；违反容量/带宽量级的候选在此淘汰（淘汰理由引用具体常数条目 ID） |
| Phase 2 §1.6.3（布局决策逐候选评分） | 主选与备选的取舍分歧涉及搬运/发射代价时，用常数表量化（引用条目 ID）；判定裕度仍落在未实证常数内 → 走实验裁决模式（不纸面拍板） |
| Phase 4 §5（Tiling/分核） | 容量约束复算：UB/L1/L0A/L0B/L0C 驻留预算对照 `CONST-capacity-a5`（assumption 条目按 90% 折扣；AutoSchedule multibuffering 的版本数计入） |

## 2. 估算公式（怎么算）

**相对量级比较 = max(三项)**，全部按 dispatch 代表 workload 计算（A5 早期以流量项为主，发射/容量项按 cycle 域模型相对折算）：

1. **流量项** = GM 读写总字节 ÷ 有效带宽（MTE 模型按访问模式取档位，`CONST-mte-bandwidth-model`：N-strided/宽行等非连续访问按对应折算档，勿按峰值；UB→L1 行 <160B 按 3/4 速率）；
2. **计算项** = Vector op 数 ÷ VALU 吞吐（`CONST-valu-model`：51 ops/cycle/AIV 常规、div/mod 13、special 8）+ Cube 段按 `CONST-cube-throughput-model`（fp16/bf16 8192 ops/cycle/AIC 等）折算 ops 地板；
3. **容量项** = 容量约束决定的 tile 循环数 × 每轮往返代价（L0C→UB 经 `T.dual_copy`/FixPipe 的带宽与延迟，`CONST-mte-bandwidth-model`；fractal/ND2NZ 转换开销计入）。

计算规则：
- 每项引用**常数条目 ID**（如 `CONST-mte-bandwidth-model`）+ 该条目状态（assumption/measured）；常数缺失时显式标注「未实测假设 + 估算依据」，不得冒充实测值；
- 结果落入 DESIGN.md **§1.6.0 末尾的「设计期估算」行**（格式：`估算下界(相对): <量级/比值>（流量 <x> + 计算 <y> + 容量 <z>，引用 CONST-*，status 标注）`）——轻量调研（单步逐元素/纯搬运类）可豁免容量项，但流量项必算（elementwise 主导项）；
- 硬件亲和性淘汰与 §1.6.3 弃选理由引用常数时**先查 constants.md**——库内已有数字不得当作"未实证常数"重新假设（与 algorithm-research.md §3 R4 同口径）。

## 3. Stage 4 回填（实测 vs 估算对账）

1. optimizer Phase 1 baseline 采集完成后，在 `perf_opt/opt_log.md` Baseline 段追加**「设计估算 vs 实测」偏差行**：DESIGN.md §1.6.0 估算下界 vs baseline 实测 do_bench(msprof) 时延，偏差 >2x 时注明偏差项（流量/计算/容量哪项失准）；
2. 该偏差行成为 `[DESIGN_LIMIT]` 归因的量化证据（perf-feedback.md §1 触发条件①的支撑材料——设计期估算系统性乐观/悲观是设计假设被实测推翻的直接形态）；
3. 调优中发现**新的硬件常数**（如新的带宽折算档/发射代价/容量上限/UB 膨胀系数）→ Phase 4 回写 pattern-library/constants.md 新条目（Tier 0，含三件套 + repro，ED-C 责任前移；assumption 条目被实测修正时 update 并转 measured），并在偏差行注明"估算失准根源 = 该常数设计期未知"。

## 4. 与现有机制的衔接

- **负向断言举证**（[negative-claim-evidence.md](negative-claim-evidence.md)）：以"违反容量/带宽量级"淘汰候选的论断，其证据 = constants.md 条目（含版本戳与 status）；constants.md 未覆盖的量仍按「未文档化假设 + 估算依据」标注；
- **实验裁决模式**：roofline 估算不能替代实验裁决——判定裕度落在未实证常数的不确定区间内时仍须产出主选+备选+裁决计划三件套（设计 skill Phase 2 第 3 项）；A5 冷启动期 assumption 常数的不确定区间默认放宽；
- **版本失效**：tilelang 重编译 / CANN/bisheng 升级后，constants.md 相关条目自动待重验（`kb_stale_check.py` 检测，repro 套件重验）——设计期引用待重验常数须在估算行标注。
