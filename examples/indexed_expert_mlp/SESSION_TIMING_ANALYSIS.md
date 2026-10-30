<!-- TILEOPS_SESSION_TIMING_V1
{"schema_version":1,"operator":"IndexedExpertMLPFwdOp","total_duration_s":44062.0,"breakdown":[{"name":"阶段准备与场景路由","duration_s":796.0,"ratio_percent":1.8,"detail":"09:44:35 会话创建 → 09:57:51 首个子 Agent 调度（statectl init / 场景路由 / kb_stale_check / 预算预留）"},{"name":"_route_stats 设计-检视-开发","duration_s":10478.0,"ratio_percent":23.8,"detail":"09:57:51 → 12:56:29，Stage 1(42.6min)+Stage 2(25.7min)+Path A 修订(28.0min)+复检(62.8min)+Stage 3(15.5min)"},{"name":"_indexed_expert_gemm 设计-检视-开发","duration_s":14935.0,"ratio_percent":33.9,"detail":"12:58:27 → 17:07:22，Stage 1(56.2min)+Stage 2(49.4min)+Path A 修订(16.9min)+门禁修复(0.9min)+复检(23.8min)+Stage 3(94.3min)"},{"name":"_weighted_reduce 设计-检视-开发","duration_s":12594.0,"ratio_percent":28.6,"detail":"17:10:10 → 20:40:04，Stage 1(42.1min)+门禁修复(5.9min)+Stage 2(22.4min)+检视重调(34.0min)+Path A 修订(29.1min)+复检(41.3min)+Stage 3(21.7min)"},{"name":"Stage 5 集成验证","duration_s":4457.0,"ratio_percent":10.1,"detail":"20:44:40 → 21:58:57，integrate_kernel.py 集成 + 单算子 TileOPs report（correctness 18/18 + benchmark 8/8，msprof 采集约 65min）"},{"name":"编排间隙（statectl/调度/结果处理）","duration_s":802.0,"ratio_percent":1.8,"detail":"函数间与阶段间的 conductor 编排开销（statectl 状态迁移、dispatch 预注入、结果核验），约 1-4min/次"}]}
-->

# Session 耗时分析 — IndexedExpertMLPFwdOp（harness 迁移）

## 数据来源声明

- **session id**：`ses_e7aab6a5effeYBUsVb47oJ5HOA`（标题 "IndexedExpertMLPFwdOp Ascend SIMD tuning"，agent=tilelang-op-conductor，model=deepseek-v4-pro/gateway）
- **full.json**：`examples/indexed_expert_mlp/full.json`
- **导出方式说明**：`opencode session export` CLI 在本共享机环境**导出失败**——托管服务端口 `127.0.0.1:49376` 被同机其他用户的 opencode 服务占用，CLI 无法启动托管服务；显式 `--server http://127.0.0.1:49376` 则要求 `OPENCODE_PASSWORD`（本会话未暴露该凭据）。因此本次耗时数据为**只读回退**：直接从本地会话库 `/home/l00970450/.local/share/opencode/opencode.db`（表 `session_v2` / `session_message`）读取同一份底层数据（与 `opencode export` 序列化源一致），按 §7.1 要求不得以 `.task_timeline.jsonl` 替代，这里也未使用后者。除导出机制外，时间戳与耗时均为本次 session 的原始记录，非估造。

## 总耗时

- **开始时间**：2026-10-29 09:44:35（本 session 首次执行活动，会话创建）
- **结束时间**：2026-10-29 21:58:57（Stage 5 终态信号 `INTEGRATE_COMPLETED`，integrator 子会话结束）
- **总耗时**：**12h 14m 22s（44062 s）**

> 会话 `time_updated` 为 2026-10-30 08:39:19（含 Stage 5 后置的 Session 耗时分析、报告重渲染、终态蒸馏等可恢复后处理），不计入上述「任务执行」统计区间。

## 耗时细分（按开始时间升序）

| 序号 | 阶段 | 开始 | 结束 | 耗时(min) | 占比 | 证据摘要 |
|------|------|------|------|-----------|------|---------|
| 1 | 阶段准备与场景路由 | 09:44:35 | 09:57:51 | 13.3 | 1.8% | 会话创建 → 首个 designer 调度（statectl init / 场景识别 / migration_mode 探测 / kb_stale_check / dispatch 收据预留） |
| 2 | `_route_stats` 设计-检视-开发 | 09:57:51 | 12:56:29 | 174.6 | 23.8% | Stage 1 设计 42.6min → Stage 2 检视 25.7min → Path A 修订 28.0min → 复检 62.8min → Stage 3 开发 15.5min（`[PRECISION_PASS]`） |
| 3 | `_indexed_expert_gemm` 设计-检视-开发 | 12:58:27 | 17:07:22 | 248.9 | 33.9% | Stage 1 设计 56.2min → Stage 2 检视 49.4min → Path A 修订 16.9min → 门禁修复 0.9min → 复检 23.8min → Stage 3 开发 94.3min（`[PRECISION_PASS]` 40/40） |
| 4 | `_weighted_reduce` 设计-检视-开发 | 17:10:10 | 20:40:04 | 209.9 | 28.6% | Stage 1 设计 42.1min → 门禁修复 5.9min → Stage 2 检视 22.4min → 检视重调 34.0min（infra 错误）→ Path A 修订 29.1min → 复检 41.3min → Stage 3 开发 21.7min（`[PRECISION_PASS]`） |
| 5 | Stage 5 集成验证 | 20:44:40 | 21:58:57 | 74.3 | 10.1% | integrate_kernel.py 集成 + 单算子 report（correctness 18/18、benchmark 8/8、roofline 80.03%–97.75%，msprof 采集约 65min） |
| 6 | 编排间隙（statectl/调度/结果处理） | —（分散） | — | ~13.4 | 1.8% | 19 次子 Agent 调度间的 conductor 编排开销（状态迁移、dispatch 预注入、收据/结果核验），平均 ~1-4min/次 |

> 阶段 2/3/4 的「耗时」为该函数的完整墙钟跨度（含其内部多次 dispatch 之间的小间隙）；各子 Agent 的净执行时长已在证据摘要中单列。总子 Agent 净时长约 686.9min（11.45h），编排与间隙约 35.5min。

## 减少耗时的方案

1. **检视基础设施前置自检（最高收益，预计省 ~34min）**：`_weighted_reduce` 的 Stage 2 检视因「compaction 模板不匹配」基础设施错误导致一次 34min 的重调。按 `execution-efficiency.md` §2 的返回前只读自检（reviewer 产出 REVIEW.md 前先核对模板/机械字段），可在首次产出即通过，消除这次纯浪费的重调。
2. **设计门禁对齐预检（预计省 ~7min + 减少碎片化）**：F2/F3 各触发一次设计门禁修复（0.9min / 5.9min，S1-PLACEHOLDER 花括号、S1-REF-PATHS `path:line` 格式）。在 designer 调度 prompt 的「知识预注入」中附门禁示例（gate-and-retry.md §1 Stage 1 行的机械要求），可让首版 DESIGN.md 即满足机械门禁，避免微 dispatch。
3. **独立函数的并行设计/检视**：本 op 三函数因数据流（route_stats → gemm → weighted_reduce）串行依赖，无法并行。若未来 op 有独立函数，harness §4 已支持并行 design/review（最多 2 个静态 agent），可将设计/检视墙钟显著压缩。
4. **设计修订概率抑制（预计省 ~200min）**：三个函数各触发 1 次 Path A 修订，合计约 200min 修订+复检开销。加强 designer 首次调度的 kb 预注入（top-K 同族 GEMM/归约 pattern-library 案例 + 分核策略 §2.1 示例），可降低首版设计被检视判不通过的概率。
5. **msprof benchmark 时间（设备绑定，收益有限）**：Stage 5 的 8 个 workload msprof 采集约 65min（6-10min/case），属设备端性能探测的固有成本；迁移场景 benchmark 仅报告不修，若只需正确性门禁可考虑轻量 prof-mode，但会丢失性能基线，需权衡。
