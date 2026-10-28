# Session 耗时分析 — logsumexp harness 迁移（LogSumExpFwdOp）

> **数据源**：`examples/logsumexp/full.json`（`opencode session export --standalone ses_ebd910775ffeMV08NXcmtrYscJ` 导出，279 条消息）；
> 阶段边界以 subagent 会话时间戳（opencode session db）与 `examples/logsumexp/.task_timeline.jsonl`、
> `examples/logsumexp/_logsumexp_kernel_{single,tiled}/.task_timeline.jsonl` 的 statectl 事件交叉核对。
> 统计区间 = 本任务（harness 迁移 logsumexp）首次执行活动（08:34:28Z，Stage 0 scaffolder 调度）至 Stage 5 终态信号
> （12:15:29Z，`migration-phase DONE`）。同 session 中先行的 P3 框架移植段（08:03:49–08:33:xxZ）不计入本任务区间。

<!-- TILEOPS_SESSION_TIMING_V1
{"schema_version":1,"operator":"LogSumExpFwdOp","total_duration_s":13261.0,"breakdown":[{"name":"Stage 0 脚手架（scaffolder）","duration_s":945.0,"ratio_percent":7.1,"detail":"08:34:28–08:50:13Z；Phase 0 通读 GPU 仓 + S1-S7 七文件 + 抽取脚本 + Tier 1 全过"},{"name":"函数1 S1 设计（含 1 次门禁窄修复）","duration_s":1542.0,"ratio_percent":11.6,"detail":"08:50:59–09:17:28Z；first_design 1298s + S1-PLACEHOLDER/CORES-LOGICAL/REF-PATHS 三项机械门禁误报窄修复 244s"},{"name":"函数1 S2 设计检视","duration_s":2882.0,"ratio_percent":21.7,"detail":"09:17:48–10:05:50Z；9 维度迁移检视 + 维度 8 独立推演 + 机械复核伪影消解（亲读解析器源码）"},{"name":"函数1 S3 实现","duration_s":1141.0,"ratio_percent":8.6,"detail":"10:06:17–10:25:18Z；首过 PRECISION_PASS，L0/L1/L2/Boundary 86 用例，fp16/bf16 位级一致"},{"name":"函数2 S1 设计（1 次异常中断 + 重试首过）","duration_s":2258.0,"ratio_percent":17.0,"detail":"10:25:54–11:03:59Z；att1 subagent 异常终止 603s（无工件产出，非门禁失败）+ att2 完整设计 1655s（gate 1 零失败零警告——前序教训规避误报生效）"},{"name":"函数2 S2 设计检视","duration_s":2994.0,"ratio_percent":22.6,"detail":"11:04:16–11:54:10Z；9 维度检视 + 真机探针闭合尾 chunk reduce 先例空隙（review_probe_tail_reduce.py PROBE_PASS）"},{"name":"函数2 S3 实现","duration_s":997.0,"ratio_percent":7.5,"detail":"11:54:28–12:11:05Z；首过 PRECISION_PASS，61 用例，fp16/bf16 位级一致"},{"name":"Stage 5 集成验证（integrator）","duration_s":235.0,"ratio_percent":1.8,"detail":"12:11:19–12:15:14Z；一次通过（无调试闭环）：45/45 正确性 + 6/6 benchmark（do_bench(msprof)）+ report passed"},{"name":"conductor 编排开销","duration_s":267.0,"ratio_percent":2.0,"detail":"状态机迁移（migration init/func/phase/integrate）、机械门禁执行、dispatch 间隔；9 次 dispatch 合计 12994s，总区间 13261s 之差"}]}
-->

## 1. 总耗时

- **本任务区间**：2026-10-16 08:34:28Z → 12:15:29Z，**13261s（3h41m）**。
- 同 session 前置段（不计入）：P3 TileOPs 框架移植 08:03:49Z 起 ~30 min（4 个并行 subagent + 验收门禁 + 提交 40c9d668）。

## 2. 阶段明细（按开始时间升序）

| 序号 | 阶段 | 开始时间 (UTC) | 结束时间 (UTC) | 耗时 (s) | 占比 | 证据摘要 |
|---|---|---|---|---|---|---|
| 1 | Stage 0 脚手架 | 08:34:28 | 08:50:13 | 945 | 7.1% | scaffolder 会话 `ses_ebc26d52…`；`migration-init` 08:50:27Z |
| 2 | 函数1 S1 设计（2 att） | 08:50:59 | 09:17:28 | 1542 | 11.6% | 设计 1298s + 门禁窄修复 244s（S1-PLACEHOLDER 集合记号误报等 3 项）；timeline `complete 1` 09:17:38Z |
| 3 | 函数1 S2 检视 | 09:17:48 | 10:05:50 | 2882 | 21.7% | reviewer 会话 `ses_ebbff264…`；结论「通过」0 阻塞 |
| 4 | 函数1 S3 实现 | 10:06:17 | 10:25:18 | 1141 | 8.6% | `[PRECISION_PASS]` 首过；timeline `complete 3` 10:25:27Z |
| 5 | 函数2 S1 设计（2 att） | 10:25:54 | 11:03:59 | 2258 | 17.0% | att1 异常中断 603s（subagent 无响应终止，无工件）；att2 重试 1655s 首过 |
| 6 | 函数2 S2 检视 | 11:04:16 | 11:54:10 | 2994 | 22.6% | reviewer 会话 `ses_ebb9dae0…`；含真机探针 PROBE_PASS |
| 7 | 函数2 S3 实现 | 11:54:28 | 12:11:05 | 997 | 7.5% | `[PRECISION_PASS]` 首过 |
| 8 | Stage 5 集成 | 12:11:19 | 12:15:14 | 235 | 1.8% | `INTEGRATE_COMPLETED` 一次通过；`migration-integrate pass` 12:15:19Z |
| 9 | conductor 编排开销 | — | — | 267 | 2.0% | dispatch 间隔 + statectl 状态迁移 + 机械门禁执行 |

注：各 subagent 会话时间戳与 `.task_timeline.jsonl` 的 statectl 事件互证一致（偏差 ≤ 30s 的状态记账延迟）；阶段间无并行重叠（串行 DEV_LOOP）。

## 3. 减少耗时的方案

1. **S2 检视是最大头（44.3% = 5876s）**，其中大量时间花在「机械复核伪影消解」（两次检视均须亲读 design_calc_check 解析器源码来区分设计错误与解析伪影）。**治本**：修复 design_calc_check 已入队的三类误报（VP-2026-0003 一带：`_section` 标记劫持/否定语境/未建模 T.Persistent），可将每函数 S2 缩短约 20–30%。
2. **函数2 S1 异常中断浪费 603s（4.5%）**：subagent 无响应终止且无工件产出。建议 conductor 对「无文本返回且无工件」的 dispatch 做快速失败检测（timeline 已能识别），并在重试 prompt 中免去重新注入全部上下文以外的工作。无法从本仓根除（属执行环境稳定性），但可缩短检测间隔。
3. **S1 门禁窄修复 244s 可避免**：三类误报（集合记号/num_blocks 字面词/路径前缀）已在函数2 通过「前序教训注入」规避（函数2 S1 零门禁重试）——教训传递机制有效性已实证；进一步可将该三条写入 tilelang-op-design skill 的门禁自查清单（已作为 VP-2026 提案入队）。
4. **S3 实现很快（合计 2138s，16.1%）**且两函数均首过——得益于「前序函数教训逐字注入」（single 的 per-chunk 计算体复用指引直接落到 tiled 实现中）。同族多函数迁移应保持该机制。
5. **Stage 0/5 极快（945s/235s）**：框架脚手架与集成链路无优化必要。
