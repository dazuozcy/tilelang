---
name: tilelang-github-operations
description: >
  tilelang-fork GitHub 工作流技能。用户提及 commit、push、PR、rebase、
  upstream、issue、GitHub Actions、gh CLI、分支同步时必须使用本技能。
  本仓是上游 tile-ai/tilelang 的 fork；PR/issue 遵循 CONTRIBUTING.md 与
  docs/developer_guide/；PR readiness 检查的权威是 tilelang-submit-pr skill。
---

# TileLang GitHub Operations（tilelang-fork）

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## 仓库拓扑

- **上游**：`tile-ai/tilelang`（upstream remote）。
- **本仓**：上游的 fork（origin 为个人/组织 fork remote，如
  `dazuozcy/tilelang`）。日常工作在 fork 上的 feature branch 进行，
  PR 目标为上游 `main`。
- 与上游的合并冲突高发区：`examples/`、`docs/`、`.github/` 属上游同步范围，
  fork 侧自有的算子工作区与工具产出应避免混入上游路径（见 AGENTS.md 与
  conductor 相关约定）。

## 工作流基线

1. **同步上游**：`git fetch upstream` → feature branch 上
   `git rebase upstream/main`（保持提交历史线性、便于 review）。
2. **本地验证**：跑改动区域的聚焦测试；**必跑**格式校验（Pre-PR 强制）：
   ```bash
   bash format.sh --files <changed_file>...
   ```
   （从仓库根目录运行；AGENTS.md「Pre-PR Formatting Rule (Mandatory)」）。
3. **提交**：聚焦改动、约定式前缀（见下）；不混入无关 cleanup 与生成物。
4. **推送并建 PR**：`git push origin <feature-branch>`，PR 目标
   `tile-ai/tilelang` 的 `main`。PR readiness（scoped diff、测试可移植性、
   构建新鲜度、CI 验证）**以 tilelang-submit-pr skill 为权威**，本 skill
   不重复其清单。
5. **跟踪 CI**：失败快速响应；PR 文案只列实际完成的验证。

## Commit 前缀惯例

以 `git log --oneline` 实例为准，方括号组件标签开头（可叠加第二层标签）：

| 前缀示例 | 用途 |
|---|---|
| `[BugFix][JIT]` / `[BugFix][Transform]` / `[BugFix][Windows]` | 缺陷修复（可带子系统/平台二层标签） |
| `[Feature][Autotune]` | 新功能 |
| `[Ascend]` / `[ROCm]` / `[CPU]` / `[CUDA]` | 后端/目标相关改动 |
| `[Codegen]` | 代码生成 |
| `[Test]` | 测试 |
| `[Docs]` | 文档 |
| `[Chore]` / `[Maint]` | 工程杂务 / 维护脚本 |

亦存在少量 `fix(scope): ...` conventional 风格历史提交；新提交优先使用
方括号标签风格。Ascend 相关改动统一用 `[Ascend]`。

## Issue 约定

- 上游 issue 仓库：`tile-ai/tilelang`。**开 issue 前先搜索**是否已有人报告
  （CONTRIBUTING.md「Report Bugs」）。
- 必含：最小复现代码片段 + 问题的清晰描述（模板见
  `references/issue-template.txt`）。
- 疑似编译器缺陷类 issue：先按 tilelang-ascend-debug skill ⑦ 的保全纪律
  备齐最小失败 kernel、生成源码/lowering trace 与环境戳再上报；
  **对外建 issue 前须用户批准**（AGENTS.md Capability Gap Rule）。
- conductor 任务中发现的能力缺口先登记
  `.agents/evolution/capability-gaps.md`，达 recurring 后由提案驱动建 issue。

## Docs to consult first

- `CONTRIBUTING.md`（贡献总入口：测试政策、编码风格、pre-commit、本地测试）
- `docs/developer_guide/testing_style.md`（测试放置与案例选择）
- `docs/developer_guide/cpp_style.md`（C++ 命名/TVM 对象/头文件边界）
- `.github/workflows/ci.yml`（CI 矩阵，判断测试可移植性）

## References

- `references/pr-workflow.txt`：分支同步 → 本地验证 → 提交推送 → PR 创建与
  CI 跟踪的分步操作卡。
- `references/issue-template.txt`：issue 标题与正文模板。

## Related skills

- tilelang-submit-pr（PR readiness 权威，勿与此处重复）
- tilelang-review-skill（代码审查与格式校验）
- tilelang-build（构建/重装/测试命令）
