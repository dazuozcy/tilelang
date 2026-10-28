# 双仓长期同步机制（tilelang-fork ⇄ tilelang-mlir-fork）

> 依据《tilelang-op-conductor A5 适配方案》§四落地。本仓（tilelang-fork，ascend/A5，
> `--npu-arch=dav-3510`）的 conductor 框架移植自 tilelang-mlir-fork（旧后端仓）；后者
> 仍是 conductor 框架**流程性演进**的可能来源，两仓长期并存必然分叉——本文档规定同步
> 纪律，把分叉成本控制在「目录 diff + 人工 review cherry-pick」的可控范围。

## 1. 同步范围分层

| 层 | 路径（mlir-fork 侧） | 是否同步 | 说明 |
|---|---|---|---|
| **L0 流程层** | `.agents/tools/`（statectl / gate_lint / kb_lint / kb_search / kb_stale_check / repro_runner / ab_test 等**流程逻辑**）；`.opencode/agents/` conductor 骨架与场景文件的**结构性变更**（调度流、出口判定、门禁接入点）；`.agents/skills/_shared/standards/` 流程标准 | **同步** | 与后端解耦的流程性变更；cherry-pick 后需过本仓门禁（禁用词 / `pytest .agents/tools/tests` / `standards_check.py`） |
| **L1 词汇层** | L0 变更中携带的后端词汇 | 同步时**适配** | 一律对照 `.agents/skills/_shared/standards/backend-glossary.md`（单一事实源）换词，禁止旧栈术语进入本仓规则文件 |
| **L2 后端层** | 后端专属 skill / references / pattern-library 实测数据 / evals 评测集 | **不同步** | 两仓绑定不同工具链与硬件，实测结论互不适用（版本戳规则见 `kb_stale_check.py`） |
| **L3 数据层** | `.agents/evolution/`（queue / stats / 快照） | **不同步** | 各仓自进化数据独立积累；仅**提案的思路**（非证据）可人工参考 |

TileOPs（`examples/TileOPs/`）与算子工作区属本仓自有产物，不在同步范围。

## 2. 同步流程

1. **来源侧打标**：mlir-fork 侧对 conductor 框架的流程性改动提交时打
   `conductor-framework` 标签（建议逐提交打，或按同步窗口打批量标签
   `conductor-framework-<YYYYMMDD>`）。
2. **变更枚举**（本仓侧）：
   ```bash
   git -C <mlir-fork 路径> log conductor-framework-baseline.. \
       -- .agents/tools .opencode/agents .agents/skills/_shared/standards
   ```
3. **目录 diff + 人工 review**：逐变更判定「纯流程（cherry-pick）」还是
   「流程+后端耦合（手工移植 + 词汇适配）」。判据：改动是否触及后端方言 /
   工具链命令 / 硬件常数；触及即按 L1/L2 处置。
4. **落地**：cherry-pick 或手工移植 → `bash format.sh --files <changed>` →
   门禁三连（禁用词 grep、`pytest .agents/tools/tests`、
   `python3 .agents/tools/standards_check.py check`）。
5. **基线前移**：同步完成后把本文档 §4 的基线记录更新为已同步到的
   mlir-fork 提交，并在 mlir-fork 侧把 `conductor-framework-baseline`
   标签前移到该提交。

## 3. 锁文件纪律

任一侧修改 `_shared/standards/` 后重生成 `standards.lock.json`；跨仓同步时以
「内容合并后**重生成**」为准，**禁止直接拷贝锁文件**（哈希绑定路径与内容，
拷贝即说谎）。

## 4. 基线记录

| 项 | 值 |
|---|---|
| 首次建立 | 2026-10-17（P4） |
| mlir-fork 基线提交 | `9309edb`（2026-10-15 18:27:55 +0800，"fix partial+merge error (#202)"） |
| mlir-fork 基线标签 | `conductor-framework-baseline` |
| tilelang-fork 侧锚点 | P4 提交（见 git log 本文档引入提交） |
| 上次同步 | 2026-10-17（P3 移植即最近一次全量同步；此后按 §2 增量） |

## 5. 建议节奏

- mlir-fork 侧出现 `conductor-framework*` 标签推送时择机同步；
- 至少每个工具链升级窗口（CANN / bisheng / tilelang 上游 rebase）执行一次
  §2 步骤 2 的变更枚举，即便无标签（防止漏标导致的静默分叉）；
- 同步动作本身作为价值点可入 `.agents/evolution/`（R 类），由 evolver 蒸馏。
