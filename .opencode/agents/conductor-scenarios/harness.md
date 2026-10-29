---
disable: true
---

# conductor 场景文件：migration-harness（TileOPs 同构仓迁移）

> **适用范围**：`scenario=migration` + `migration_mode=harness`（GPU 仓为 TileOPs 同构工程）。**载入顺序**：先 `conductor-scenarios/migration.md`（公共规则），再本文件。通用骨架（门禁/重试/设计修订/状态写入/恢复/自进化）见主文件 `.opencode/agents/tilelang-op-conductor.md`。

## 1. 状态机与阶段计划

`stage_plan=[0, (1→2→3)×N函数, 5]`（Stage 4 跳过）：

```
INIT --> SCAFFOLD --> (DEV_LOOP: 每函数 DESIGN --> REVIEW --> DEVELOP) --> INTEGRATE --> DONE
                        ^                |
                        |____ 修订循环 ___|  (retry_count < max_retry，仅当前函数)
  超限/失败 ____________________________> FAILED (BLOCKED_SCAFFOLD / BLOCKED_IMPL / BLOCKED_ACCURACY / BLOCKED_INTEGRATION)
```

## 2. 命名与预检特例

- **多函数命名约定**：逐函数独立 Stage 1-3，`project_name={op_slug}`、`op_name={func}`，算子目录 `examples/{op_slug}/{func}/`（**不使用** new-op.md §2 的用户消息解析规则）。
- GPU 参考实现位置与规格（manifest workloads、test/bench 路径）从 Stage 0 的 `.migration_meta.json` 读取，调度 designer 时随 prompt 传入。
- 预检不问用户：规格从 GPU 仓 meta 与源码推断（迁移公共规则见 migration.md §1）。

## 2.1 源仓与目标仓边界

当前 GPU 源仓为 `/home/zuo/code/TileOPs`：`src/tileops/manifest/spec/*.yaml`、`src/tileops/_ops/`、`src/tileops/_kernels/`，workloads/tests/benchmarks 位于仓库根目录。Ascend 目标 `examples/TileOPs/` 同样采用 `src/tileops/{_ops,_kernels,manifest/spec}`，workloads/tests/benchmarks 在目标根目录；旧布局不再支持。

Stage 0 与 Stage 5 都必须读 `examples/TileOPs/.agents/skills/add-ascend-op/references/tileops-src-layout.md`。源 manifest 无 `source`，规格语法、Kernel/Op 接口及 bench API 均需按该文档转换；meta 中 wrapper/test/bench 等路径始终指向 **A5 目标**。新源 meta 必须遵循 `.agents/skills/_shared/standards/migration-source.md`（含逐函数来源和转换记录）；调度 Stage 1/2/3 及修订时完整传递 `source_context`。

无 NPU 或用户明确禁止设备操作时，只进行静态适配，不运行 import 冒烟、pytest collection、report 或 msprof；静态结果不能替代精度和 Stage 5 门禁，不产生 `INTEGRATE_COMPLETED`。

## 3. Stage 0 — 迁移脚手架 Agent（`@tileops-scaffolder`，仅 harness）

- **触发条件**：场景路由判定 `migration` + `migration_mode=harness`，状态 `init` 后
- **输入**：`op_name`、`gpu_repo_root`、`family`（可选，从 GPU manifest 推导）
- **输出/交付件**：TileOPs 7 文件脚手架（Tier 1 结构校验通过）+ `examples/TileOPs/src/tileops/_kernels/{family}/{op_slug}/.migration_meta.json` + 逐函数迁移 prompt
- **完成信号**：`SCAFFOLD_COMPLETED`（携带 meta_path、extracted_functions、migration_prompts）
- **失败信号**：`[SCAFFOLD_FAIL]` + 原因
- **编排层动作**：
  - `SCAFFOLD_COMPLETED` → `complete_stage(0)` → 解析 `extracted_functions`，经 `statectl migration init` 建 `.migration_state.json`（见 §4），进入逐函数 DEV_LOOP
  - `[SCAFFOLD_FAIL]` 且属结构问题 → `fail_stage(0)` 重试（≤3 次）；"GPU 侧无 `@tilelang.jit` 实现" → `phase=FAILED`、`failure_reason=BLOCKED_SPEC`；GPU repo 缺失 → `BLOCKED_ENVIRONMENT`；结构重试超限 → `BLOCKED_SCAFFOLD`

## 3.1 Stage 0 快速执行契约（首次启动 / 仅验证脚手架）

**调用接口优先**：正常路径不 Read/Grep `statectl.py`、`gate_lint.py`、提取器实现或其测试，不翻历史任务推断用法。按本节调用并消费 JSON；只有具体命令报错或本节未覆盖的接口问题才允许定点读实现，并说明错误码/缺失契约。源算子及适配必需的目标基类不受此限制。

1. **环境只确定一次**：优先用户指定的 Python，否则检查当前 `python3` 的版本与 PyYAML：`python3 -c 'import sys, yaml; assert sys.version_info >= (3, 10); print(sys.executable)'`。失败时只查已知环境中的解释器，不递归扫描磁盘或逐环境导入 torch。保存输出绝对路径为 `stage0_python`，所有状态/发现/验收命令及子 Agent 复用它。用户明确无设备/禁止设备操作时设 `validation_mode=static_only`，不做设备探测；否则仅在选定环境检查一次，结果透传。Stage 0 静态发现不依赖 torch/torch_npu。
2. **工具决定身份与路径**：在初始化状态之前执行提取器 `--inspect --inspect-out /tmp/stage0_inspect.json`（并发任务用独立临时路径）。JSON 顶层 `op_slug` / `task_dir` / `wrapper_path` / `meta_path` / test/bench 路径是唯一命名依据，不自行转换类名或沿用源文件名。按返回的 `task_dir` 保存完整 inspect JSON 并透传；不截断 JSON、不重复 inspect。已有 meta 的 op_slug 与工具不一致时报告路径冲突，不删除旧目录或重建状态来掩盖冲突。
3. **重复验证走复用路径**：若工具返回的 meta/report 已存在且用户未要求强制重新移植，先用 `stage0_check.py --meta "$meta_path" --reuse-report "$scaffold_dir/stage0_static_checks.json" --emit-prompts --output "$scaffold_dir/stage0_static_checks.json"`。仅 `static_pass` 且 `reuse_scaffold=true` 时复用已转换脚手架，不重读业务源码、不调度 scaffolder、不重写已完成状态；核对已有状态并报告“已有脚手架复验通过”。工具将源树、目标代码、meta/转换记录和检查器内容指纹一起比较；缺报告/指纹变化/检查失败均回到正常生成路径，不凭文件存在判定可复用。复用只证明 static-only，不升级运行时结论。
4. **新任务固定命令**（仓库根执行；`$stage0_python` 是上述解释器，`$task_dir=examples/{op_slug}`；每条成功后再执行下一条）：

```bash
"$stage0_python" .agents/tools/statectl.py init --dir "$task_dir" --project "$op_slug" --op "$op_name" --scenario migration --migration-mode harness --stage-plan 0,1,2,3,5 --requirement "$requirement"
"$stage0_python" .agents/tools/kb_stale_check.py --json
"$stage0_python" .agents/tools/statectl.py start 0 --dir "$task_dir"
```

这是 **op 级 Stage 0 状态**，不是逐函数 Stage 1 状态。已有状态先 `verify`，禁止重复 init/start。调度 scaffolder 时传 `stage0_python`、`validation_mode`、inspect 路径与“仅 Stage 0”停止边界；conductor 不重复阅读子 skill 和六类业务源码。

5. **返回后固定收束**：scaffolder 先完成下述静态验收（输出 wrapper 同目录的 `stage0_static_checks.json`），conductor 消费报告；`next_action=return_to_conductor` 后不再补写 YAML/路径检查，不再重复 py_compile、文件清单扫描或相同范围的 lint。只在产物再次修改或检查器报告具体缺口时重验。`complete 0` 内含 gate 0，无须再单独 gate 一次。`$meta_path`、`$family`、函数列表取子 Agent 返回且已验收的 meta：

```bash
"$stage0_python" .agents/tools/statectl.py complete 0 --dir "$task_dir" --meta "$meta_path"
"$stage0_python" .agents/tools/statectl.py migration init --dir "$task_dir" --op-slug "$op_slug" --family "$family" --meta-path "$meta_path" --functions {extracted_functions逐项参数} --phase DEV_LOOP
```

仅 Stage 0 请求到此停止；`phase=DESIGN` 和聚合 `DEV_LOOP` 表示后续可继续，不启动 Stage 1，不创建逐函数状态，不执行终态蒸馏。报告只列产物、函数、静态/运行时验证范围及下一阶段，不逐字复述全部迁移 prompt（完整 prompt 作为文件工件传递）。失败仍走既有门禁/重试路由。

## 4. 多函数组织（DEV_LOOP 与聚合状态）

- Stage 0 产出的 `.migration_meta.json` 中 `extracted_functions` 为函数列表（N ≥ 1）。
- **每个函数独立跑 Stage 1→2→3**：算子目录 `examples/{op_slug}/{func}/`；每函数独立 `.stage_state.json` 与独立 `stage_retry_count` 预算。
- 函数间不共享 DESIGN.md；某函数 `[DESIGN_ERROR]` 只修订该函数。
- 全部函数 Stage 3 通过（含全量执行收据核验）后才进入 Stage 5。
- Stage 0 后由 scaffolder 根据实际调用/数据依赖产出，再由 conductor 消费 `function_dependencies.json`：完整对象 `{函数名: [前置函数名]}`，覆盖全部 extracted_functions；不能以列表顺序或函数名猜依赖。未知依赖维持原串行流程，不生成虚假的空依赖图。
- 调度前运行 `statectl migration ready --dir examples/{op_slug} --graph <绝对路径>`，工具拒绝漏项、未知节点和环。ready 中独立函数的 Stage 1/2 可并行（默认最多 2 个静态 agent），各写独立目录；每函数内部仍先设计后检视。Stage 3 及设计探针等任何设备操作进入同一全局串行队列；静态并发的 designer 禁用设备探针或排队执行。
- statectl 更新与共享修订预算由 conductor 串行处理；并行组使用同一任务上下文快照，教训只取已完成依赖。工具不支持并发、存在共享写入或依赖待定时退回串行；失败函数的依赖节点暂停，不取消独立分支的已验证产物。
- Stage 0 固化逐函数签名/返回结构/调用分支与 wrapper 接线契约到来源上下文；每个 Stage 3 返回前静态核对目标适配接口。Stage 5 不再首次发现可静态识别的接口缺陷。
- 聚合状态由你维护在 `examples/{op_slug}/.migration_state.json`：

```json
{
  "op_name": "{op_name}", "op_slug": "{op_slug}", "family": "{family}",
  "meta_path": "examples/TileOPs/src/tileops/_kernels/{family}/{op_slug}/.migration_meta.json",
  "phase": "SCAFFOLD | DEV_LOOP | INTEGRATE | DONE | FAILED",
  "functions": {"{func}": {"stage_state": "examples/{op_slug}/{func}/.stage_state.json", "status": "pending | in_progress | done | failed"}},
  "integration": {"attempts": 0, "status": null}
}
```

- 聚合状态的迁移与函数状态更新一律经 `statectl migration` 命令组执行：`migration init --op-slug S --family F --meta-path P --functions f1 f2`（Stage 0 完成、解析函数列表后创建，phase=DEV_LOOP）→ 逐函数 `migration func {name} --status pending|in_progress|done|failed` → 全函数 done 后 `migration phase --phase INTEGRATE` → 集成结果 `migration integrate --status pass|fail`（fail 超 2 次自动 `BLOCKED_INTEGRATION`）→ pass 后 `migration phase --phase DONE`。Stage 0/5 的工件门禁用 `statectl gate 0 --meta ...` / `statectl gate 5 --migration-dir ...` 单独执行。

## 5. Stage 3 通过后的分支

`[PRECISION_PASS]` 且全量执行收据核验通过 → `migration func {func} --status done` → 还有未完成函数则按 §4 依赖 ready 集合推进各自未完成阶段；全部函数 done → `migration phase --phase INTEGRATE` → `start_stage(5)` 调度 integrator。

## 6. Stage 4 跳过

harness 迁移**不询问调优、不进入 Stage 4**（bench 由 Stage 5 仅报告）；最终报告附本次单算子 report 路径、bench 状态与有效数值（无有效数值时写明原因），以及"可另起 optimize 场景"提示。

## 7. Stage 5 — 迁移集成 Agent（`@tilelang-op-integrator`，仅 harness）

- **触发条件**：全部提取函数 Stage 3 通过且执行收据核验完成（`.migration_state.json` 的 `functions` 全部 `done`）
- **输入**：`meta_path`、`op_name`、`op_slug`、`family`、`attempt_index`、`max_attempts`（默认 5）
- **输出/交付件**：`src/tileops/_kernels/{family}/{op_slug}/{op_slug}_kernel/`（集成 kernel 文件 + 每函数 `{func}_DESIGN.md` 设计文档快照（源自 `examples/{op_slug}/{func}/DESIGN.md`）+ 聚合 `__init__.py` + `integration_log.md` + `integration_report.json`），wrapper import 已改写为 baseline/perf_opt 双 import 切换块（baseline 默认激活，perf_opt 注释占位）；另有本次单算子 TileOPs `run.json`、`report.md`、`report.html`
- **完成信号**：三态之一：`INTEGRATE_COMPLETED`（从 `examples/TileOPs/` 运行 `PYTHONPATH=src:.:../.. python -m tileops.reporting.cli run --op {op_name} --prof-mode msprof`；本次 report 全量正确性通过、benchmark 已运行并报告；report 为 `partial` 时明确披露 benchmark 失败/无效但不阻断集成）/ `[INTEGRATE_FAIL]` / `[DESIGN_ERROR]`
- **编排层动作**：
  - `INTEGRATE_COMPLETED` → `complete_stage(5)` → `phase=DONE`（harness 迁移不询问调优；最终报告附本次 report 路径、bench 状态与有效数值，以及"可另起 optimize 场景"提示）
  - `[INTEGRATE_FAIL]` → `fail_stage(5)` → 重新调度 integrator 传入 `last_failure_summary`（`stage_retry_count[5]` 上限 2；integrator 内部已有 5 次调试闭环，两级预算独立）；超限 → `phase=FAILED`、`failure_reason=BLOCKED_INTEGRATION`
  - `[DESIGN_ERROR]` → 设计修订循环路径 B：对**失败根因指向的函数**备份其 `DESIGN.md` → `retry_count += 1` → 该函数重跑 Stage 1→2→3 → 通过后**重新执行 Stage 5**（全量重集成，集成脚本幂等）
- **Stage 5 终态后置钩子**：Stage 5 进入 `DONE`，或因 Stage 5 重试耗尽进入 `FAILED` 后，必须先按主文件「自进化机制」的 Session 耗时分析 prompt 生成 `examples/{op_slug}/SESSION_TIMING_ANALYSIS.md`，再重新渲染报告并开始终态蒸馏；该钩子不是新 Stage，不修改 Stage 状态。

### 7.1 Stage 5 后 Session 耗时分析 prompt（必执行）

Stage 5 进入上述终态后，在重新渲染 TileOPs report、终态蒸馏和最终报告之前，由 conductor 在当前 Primary 会话执行以下 prompt。该环节不调用算子 Subagent；写入此文件是 conductor「禁止自行编辑算子工件」的显式例外。

```text
获取本次 OpenCode session id，记为 `<session_id>`。

先在仓库根目录执行以下命令，将本次 session 的完整数据导出到 harness 聚合时间线 `examples/{op_slug}/.task_timeline.jsonl` 所在的固定目录：

opencode export <session_id> > examples/{op_slug}/full.json

导出完成后，只基于 `examples/{op_slug}/full.json` 进行时间分析：

1. 整个过程耗时多久？
2. 对耗时进行细分，并按实际发生时间从早到晚排列各个阶段的耗时；每个阶段列出开始时间、结束时间、耗时和占总耗时比例。
3. 给出减少耗时的方案。
4. 将上面问题的回答输出到：
   examples/{op_slug}/SESSION_TIMING_ANALYSIS.md
```

执行要求：

1. `<session_id>` 必须是**本次执行对应的当前 OpenCode session id**。harness 聚合 `.task_timeline.jsonl` 由 `statectl migration ... --dir examples/{op_slug}` 固定写入 `examples/{op_slug}/`，因此不得再搜索或推断目录；导出目标固定为 `examples/{op_slug}/full.json`。
2. 必须先执行 `opencode export`，确认 `full.json` 已生成且可解析，再开始分析。时间事实源只允许使用本次导出的 `full.json`；统计区间从本任务首次执行活动到 Stage 5 终态信号，不得混入其他 session，也不得以 `.task_timeline.jsonl` 或新增 Stage/Phase 埋点替代 `full.json`。
3. Markdown 必须写明 `full.json` 路径和 session id、开始/结束时间、总耗时、主要环节耗时与占比、减少耗时方案。耗时细分须使用按开始时间升序排列的阶段表，至少包含“序号、阶段、开始时间、结束时间、耗时、占比、证据摘要”；阶段存在重叠时保持实际开始时间顺序并明确标注，不得通过重排掩盖并行或等待。`full.json` 不能证实的值写 `N/A`，不得估造；若导出失败、文件缺失或无法解析，仍要生成 `SESSION_TIMING_ANALYSIS.md`，明确失败原因及无法确认的字段。
4. Markdown 顶部须包含以下机器可读块，供报告渲染器提取；`total_duration_s` 无法确认时为 `null`，`breakdown` 无法确认时为空数组。自然语言分析写在该块之后。

```text
<!-- TILEOPS_SESSION_TIMING_V1
{"schema_version":1,"operator":"{op_name}","total_duration_s":0.0,"breakdown":[{"name":"环节名","duration_s":0.0,"ratio_percent":0.0,"detail":"日志证据摘要"}]}
-->
```

5. `breakdown` 按各阶段实际开始时间升序排列，只保留对理解总耗时有用的主要环节，避免把报告扩展成逐事件流水账；每项 `detail` 写明该阶段的开始/结束时间和日志证据摘要。减少耗时方案留在 Markdown 正文，报告只展示总耗时和上述时间明细。
6. Markdown 写入成功后，从集成包的 `integration_report.json` 读取本次 `run.json` 仓库相对路径；文件有效时从 `examples/TileOPs/` 执行 `PYTHONPATH=src:.:../.. python -m tileops.reporting.cli render {run_json} --with-session-timing`，原地更新该 run 的 `run.json` / `report.md` / `report.html`，**不得重跑 correctness 或 benchmark**。`run.json` 缺失或无效时不伪造报告，最终汇报披露无法重渲染，然后仍将 Markdown 交给 evolver。

## 8. harness 设计修订特例

- `retry_count` 为**全 op 共享预算**（跨函数累计），达 `max_retry` 即整个 op `FAILED`。
- 修订只回退**失败根因指向的当前函数**，其他已完成函数的工件不动。
- 修订后该函数重跑 Stage 1→2→3，通过后 Stage 5 **全量重集成**（集成脚本幂等）。

## 9. harness 目录结构

```text
examples/{op_slug}/               # op 级目录（project = op_slug）
├── .migration_state.json         # conductor 维护的多函数聚合状态
├── full.json                     # 由 opencode export 导出的当前 session 完整数据
├── RETROSPECTIVE.md              # Stage 5 集成复盘（op 级单份，自进化钩子）
├── SESSION_TIMING_ANALYSIS.md     # Stage 5 后基于当前 session 的 full.json 生成的耗时分析
└── {func}/                       # 每个提取函数一个算子目录（结构同主文件标准目录，无 Stage 4；含函数级 RETROSPECTIVE.md）

examples/TileOPs/                              # 集成侧
├── src/tileops/manifest/spec/{family}.yaml             # Stage 0 产物（S1）
├── workloads/{family}.py              # Stage 0 产物（S2）
├── src/tileops/_kernels/{family}/{op_slug}/
│   ├── {op_slug}.py                            # wrapper（Stage 0 移植，Stage 5 改写为 baseline/perf_opt 双 import 切换块）
│   ├── .migration_meta.json                    # Stage 0 机器模式产物
│   └── {op_slug}_kernel/                       # Stage 5 集成包
│       ├── {func}.py                           #   集成 kernel（源自 examples/{op_slug}/{func}/）
│       ├── {func}_DESIGN.md                    #   集成设计文档（源自 examples/{op_slug}/{func}/DESIGN.md，Stage 1 交付件快照）
│       ├── __init__.py                         #   聚合 re-export（integrate_kernel.py 生成）
│       ├── perf_opt/                           #   Stage 4 调优产物（optimize 场景：{func}.py + opt_log.md；wrapper 切换块的 perf_opt import 指向此处）
│       ├── integration_log.md                  #   集成验证与调试日志
│       ├── integration_report.json             #   本次 TileOPs run.json 的仓库相对路径
│       └── history_version/                    #   Stage 5 调试备份
├── tests/ops/test_{test_slug}.py               # Stage 0 产物（S5，仅含本算子用例）
├── benchmarks/ops/bench_{bench_slug}.py        # Stage 0 产物（S6）
└── reports/tileops/{run_id}/                   # Stage 5 单算子报告（run.json / report.md / report.html / pytest / profiler）
```

### harness 专属工件衔接（Owner / Consumer）

| 工件 | Owner | 主要消费者 | 消费者需要的信息 |
|------|-------|------------|-----------------|
| TileOPs 7 文件脚手架 | Stage 0 | Stage 1（规格来源）、Stage 5（集成目标） | manifest workloads、wrapper/Kernel class、test/bench 路径 |
| `.migration_meta.json` | Stage 0 | conductor（函数循环）、Stage 5（集成参数与单算子 report 核对） | manifest `op_name`、op_slug / family / extracted_functions / wrapper_path / test_path / bench_path |
| `.migration_state.json` | conductor | conductor | 多函数聚合状态（见 §4） |
| `full.json` | conductor（`opencode export`） | conductor（Session 耗时分析） | 当前 OpenCode session 的完整导出数据；固定为 `examples/{op_slug}/full.json`，与 harness 聚合 `.task_timeline.jsonl` 同目录 |
| `SESSION_TIMING_ANALYSIS.md` | conductor（Stage 5 后置 prompt） | TileOPs report、evolver | 当前 session 总耗时、耗时细分、减少耗时方案及机器可读摘要 |
| `{op_slug}_kernel/`（集成包） | Stage 5 | 用户、TileOPs 框架 | 集成 kernel 文件 + `{func}_DESIGN.md` 设计文档快照 + 聚合 `__init__.py` + `integration_log.md` + `integration_report.json` |

## 10. 带记忆重试与 session 教训传递（harness 专属）

- 主文件「自进化机制」第 2 条的重试前必读标准段，在 harness 多函数任务中**另加**一条检索位置：`examples/{op_slug}/{前序函数}/RETROSPECTIVE.md` 的 Transferable Lessons 小节。
- 调度第 N 个函数（N ≥ 2）的 designer / developer 时：
  1. Read 已完成依赖函数 `examples/{op_slug}/{前序函数}/RETROSPECTIVE.md` 的「Transferable Lessons（可迁移教训）」小节（文件或小节不存在则跳过）。
  2. 将其内容**逐字追加**到调度 prompt 的「前序函数教训」段——你只搬运不加工不筛选（领域判断属于 Subagent，你保持零领域推理）。
  3. 该教训仅在本迁移任务内有效；**不得**写入任何持久文件（跨任务沉淀由 evolver 在终态蒸馏时统一裁决）。
