# 信息收集与信息源优先级

## 目录

- [信息收集与信息源优先级](#信息收集与信息源优先级)
  - [目录](#目录)
  - [1. 强制步骤 0：搜索本项目同类实现](#1-强制步骤-0搜索本项目同类实现)
  - [2. 信息收集步骤](#2-信息收集步骤)
  - [3. 禁止行为](#3-禁止行为)
  - [4. 信息源优先级](#4-信息源优先级)

---

## 1. 强制步骤 0：搜索本项目同类实现

在生成 DESIGN.md 前，**必须**执行以下工具调用：

```bash
# 1. 搜索同类算子（根据算子名称 / 类别）
glob examples/ascend/**/*{算子名称}*.py
glob examples/ascend/**/*{算子类别}*.py    # 如 gemm, attention, norm, vecadd, fp8
glob testing/ascend/**/*.py                # 特性回归（按 language/layout/analysis/
                                           # auto_schedule/transform/target/runtime 边界找）

# 2. 如果找到同类实现，完整阅读
read examples/ascend/{找到的同类实现路径}

# 3. 检查关键技术点
grep "T.Kernel\|T.MixedKernel" {同类实现}    # grid 维度与核类型（一维 / mixed）
grep "T.gemm" {同类实现}                      # GEMM API（NT 约定 transpose_B=True）
grep "T.copy\|T.dual_copy" {同类实现}         # 数据搬运路径
grep "T.alloc" {同类实现}                     # 内存层级分配（shared/l1/l0*/fragment）
grep "SimtVF\|SimdVF\|T\.simd\." {同类实现}   # VF 编程模式
grep "ascend_set_flag\|ascend_wait_flag\|ascend_pipe_barrier\|T.Stage\|T.Pipelined\|T.Persistent" {同类实现}  # 同步与调度
```

> 算子族级导航（族 → 模式 → 参考例子 → 搬运套路）见 tilelang-ascend-opdev skill 的算子族导航表。

## 2. 信息收集步骤

> **算法调研（Phase R）的信息源与查证规则**单独见 [algorithm-research.md](algorithm-research.md)（§5 常见算子族参考表命中行必查、§6 信息源清单）；本文件的优先级表与禁止行为同样适用于调研阶段的候选查证——外部算法只取思路，API 存在性与代价论断必须本地佐证。

1. 查阅 `examples/ascend/` 同类算子实现（**强制步骤 0**；vecadd / rmsnorm / gemm / gemm_splitk / per_token_cast_to_fp8 / deepgemm / flash_attention）
2. 查阅 `testing/ascend/` 特性回归（DSL/VF 行为、布局、调度、target/runtime 边界）
3. 查阅 **pattern-library（强制步骤 0.5）**：`tilelang-op-optimize` skill 的 `references/pattern-library/`（入口 INDEX.md：layout/elementwise/attention.md 实测模式与代价、traps-*.md 编译器陷阱〔注意版本戳〕、constants.md 硬件常数、cases.md 案例索引〔同类算子正/反例参考〕；检索可用 `python3 .agents/tools/kb_search.py "<查询>"`。**A5 冷启动**：当前仅 constants.md 有 assumption 级条目，其余主题文件为空骨架待积累）
4. 按 **AGENTS.md Docs Auto Routing** 查文档确认 API 可用性和用法：`docs/programming_guides/`（语言基础/类型/控制流/软件流水/autotuning）→ `tilelang/ascend/README.md` + `.agents/skills/tilelang-ascend/SKILL.md` 及其 references（**API 语义与约束权威**）→ `docs/tools/`（lower_trace / pass_diff / compile_only）→ `docs/get_started/targets.md`（target/runtime/env）
5. 如有参考实现，分析其计算步骤（**仅用于理解数学逻辑，不可直接使用 API**）

## 3. 禁止行为

- ❌ 在没有执行强制步骤 0 的情况下，直接使用外部参考实现的 API
- ❌ 凭记忆猜测 API 名称或参数（含 `T.simd.*` intrinsic——未索引的 intrinsic 必须查 `tilelang/ascend/language/simd.py` 确认）

## 4. 信息源优先级

| 优先级 | 信息源 | 用途 | 说明 |
|--------|--------|------|------|
| **0** | **`examples/ascend/` 同类实现** | **主要参考：API、编程模式、Kernel 结构** | **最权威**，直接可用 |
| **0.3** | `testing/ascend/` | DSL/VF 行为、布局、调度、target/runtime 边界的特性回归 | 不得把应用 kernel 导入该套件 |
| **0.5** | **pattern-library/**（`tilelang-op-optimize` skill 的 `references/pattern-library/`，入口 INDEX.md） | **实测代价数据、已验证模式（布局/轴/陷阱）、硬件常数与案例索引（cases.md）** | Stage 4 调优实测积累；代价类论断优先于 docs 规格与 examples 先例；**条目绑定工具链版本戳**（kb_stale_check 检测），重编译后旧结论待重验，勿引用 `status: overturned/stale` 条目；A5 冷启动期仅 constants.md 有条目（assumption 级） |
| 1 | `.agents/skills/tilelang-ascend/SKILL.md` + references（§1–§12） | **API 语义与约束权威**（线程层级/内存/GEMM/搬运/同步/SIMD/陷阱/性能测量） | Ascend 方言问题的第一权威；算子族导航另见 tilelang-ascend-opdev skill |
| 2 | `docs/programming_guides/` | 语言基础、类型系统、控制流、软件流水、autotuning | AGENTS.md Docs Auto Routing 第 1 层 |
| 3 | `docs/tools/` | lower_trace / pass_diff / compile_only / analyzer 等工具文档 | 调试与 pass 追踪 |
| 3.5 | `docs/get_started/targets.md` + `tilelang/ascend/README.md` | target / runtime / env 说明 | |
| 4 | **外部参考实现** | **仅用于理解数学逻辑** | **不可直接使用 API** |
| 4.5 | **互联网检索（webfetch，若环境可用）** | **算法调研（Phase R）的候选补充**：论文 / 开源实现 / 厂商文档 / 技术博客中的算法族变体与复杂度结论 | 只取算法思路；外部 API 不可直接采用（存在性须本地佐证）；外部性能数字仅作候选排序参考（GPU 加速比 ≠ NPU 判定依据）；引用须记录来源（URL/论文/仓库 + 访问日期），检索未命中如实记录；与本地任何信息源矛盾时以本地为准；详见 [algorithm-research.md](algorithm-research.md) §6 |
| 5 | `tilelang/ascend/language/` 源码（`simd.py`、`copy_op.py`、`gemm_op.py`、`mode.py`、`schedule_hint.py`、`tile_schedule.py` 等） | 公开 API 导出与前端定义——**实现真理** | docs 与源码冲突时以源码为准（AGENTS.md Conflict Resolution）；API 未被任何映射覆盖时先枚举本目录模块再下「不支持」结论 |
| 6 | `src/ascend/`（pass / codegen / AutoSchedule） | Ascend pass 与后端实现状态 | 实现验证（容量上限见 `src/ascend/transform/auto_schedule.cc`） |

**规则**：当信息源之间矛盾时，以 `examples/ascend/` 为准。若 examples 未覆盖，以 tilelang-ascend skill 与 `docs/`（按 Docs Auto Routing）为准。仍未覆盖时，以 `tilelang/ascend/language/` 源码实际实现为准。**性能代价类论断**（某操作快/慢/阻碍向量化）矛盾时，以 pattern-library 实测条目为准（未覆盖时标注「未实证假设」并进实验裁决流程，见 SKILL.md Phase 2）。**互联网检索结果**与上述任何信息源矛盾时一律以本地为准，且不得直接作为 API 存在性与性能代价的判定依据（见优先级 4.5 行与 algorithm-research.md §6 纪律）。
