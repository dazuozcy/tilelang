# 流程执行效率契约

本标准适用于 new_op、migration（plain/harness）、optimize。精度、独立设计检视、最终完整 benchmark 覆盖及集成验收保持原标准；优化执行方式和重复工作。conductor 是状态唯一 owner，执行工具不读写 `.stage_state.json` 或 `.migration_state.json`。

## 1. 输入、上下文与授权

- Primary 一次询问全部无法推导的关键缺项；标准语义可推导的输出规格注明依据。编程模式无用户约束时默认 simt，GEMM 主导时由 designer 论证 mixed；不为默认模式单独等待用户选择。
- 启动时记录 `perf_tuning_requested`、目标和预算；用户已要求调优时后续不重复询问。没有调优意图时，在开发通过后只询问一次，明确否定直接交付。
- 调度传递任务上下文清单：需求/来源路径、环境与工具链版本戳、已查文档路径和内容指纹、知识检索结果、当前失败摘要、剩余预算。复用相同查询和未变化文件，工具链或相关输入变化后重查。每个独立 agent 首次仍需读取自己的约束；reviewer 独立核对关键源码，不以 designer 摘要替代。
- 多函数教训按依赖已完成函数传递；仅搬运任务内经验，不并发修改知识库。

## 2. 返回前自检与风险分层

Stage 1/2/3 agent 返回前运行只读预检，不读取状态文件：

```bash
python3 .agents/tools/gate_lint.py --stage 1 --artifact <DESIGN.md> [--migration]
python3 .agents/tools/gate_lint.py --stage 2 --artifact <REVIEW.md> [--migration]
python3 .agents/tools/gate_lint.py --stage 3 --artifact <kernel.py> --op <op>
```

修正机械问题再返回；conductor 的 `complete` 门禁仍是最终权威，不因自检通过跳过。已成功 complete 且工件指纹未变时，不额外再 gate 同一上游。

成熟模板的低风险参数变体可以精简设计叙述，用可定位的参考实现和当前版本证据回答必需问题；所有必需章节、调研四问、容量预算和独立检视维度仍保留，不使用空泛的“同模板”。算法、布局、同步、精度路径变化做完整分析。revision 附变更章节及依赖；reviewer 重点重验受影响维度，其余维度保留可复核的旧证据引用及输入指纹；来源、工具链或前提变化时重做相应核对。

Stage 2 的报告机械错误用 `statectl fail 2 --reason report_repair` → `start 2` → reviewer 补报告；最多 3 次，超限 BLOCKED_DESIGN，不消耗设计修订预算。只有明确的设计不通过才用 `--reason design_revision`。报告修复不得把未解决的设计问题改为通过。

Stage 3 单次 dispatch 允许最多 2 次局部实现修复，且累计不超过 10min（服从更紧的任务剩余预算）。每次先备份、记录失败指纹和修复假设；同一失败指纹且无新证据时停止。设计错误立即返回；耗尽局部预算返回原四出口，由 conductor 记 dispatch 级重试。内部尝试记录到执行日志，不能隐藏或重置计数。

## 3. 执行收据与断点

通用执行工具 `.agents/tools/execution_receipt.py` 运行命令、保存 stdout/stderr、输入 SHA256、命令/cwd、环境、退出码和超时状态。`verify` 不执行命令。收据只证明执行结果，测试命令必须真实覆盖要求且通过退出码表达 L0/L1 失败；不能把静态命令包装成精度证明。

```bash
python3 .agents/tools/execution_receipt.py run --receipt <dir>/execution/stage3-all.json --cwd <repo> --suite stage3-all --input <kernel.py> --input <DESIGN.md> --input <environment-stamp.json> --timeout 900 -- <python> <kernel.py> --level all
python3 .agents/tools/execution_receipt.py verify --receipt <dir>/execution/stage3-all.json --cwd <repo> --suite stage3-all --input <kernel.py> --input <DESIGN.md> --input <environment-stamp.json> -- <python> <kernel.py> --level all
```

所有路径使用绝对路径，命令参数逐项传递。环境戳由 Primary 在任务启动实查后生成并传入，含解释器及 torch/torch_npu/TileLang/CANN/bisheng 版本、设备型号/设备号、源码版本与工作区改动、实际加载编译库的指纹。相关环境变化时重新生成，不能只写 git HEAD。声明全部影响执行的本地导入、测试配置、编译器/共享库和测试数据为 `--input`（大工件可用经实际复查生成的内容清单），缺依赖或无法确认环境新鲜度时禁用复用。

developer 的全量测试经 runner 生成收据；conductor 用相同参数 verify，检查 suite/命令、测试覆盖和日志后替代固定二次全量跑测。收据缺失/失败、输入变化、环境未确认或稳定性要求时重新执行并更新收据。任何代码改动均使此前通过结论失效。无设备不产生精度通过。

生成文件通过重复 `--output <绝对路径>` 声明，执行后记录输出 SHA256；目标缺失或漂移同样使收据失效，不把生成目标列为会被本命令修改的 input。`run --reuse` 仅复用输入/输出指纹、日志和成功退出码均匹配的执行单元；失败/超时不复用。适用于已验证测试、静态检查和确定性文件集成；性能样本只用于同一次实验断点，不将历史收据当作当前环境的新测量。所有设备命令通过 `run --device-lock <Primary 透传的全局设备锁绝对路径>` 排队（同一设备跨函数/任务使用同一路径）；锁等待计入任务墙钟，获锁后重新检查预算。Stage 5 最终 report 必须验证当前完整集成包，不能用 Stage 3 收据替代。

## 4. 预算与恢复

任务初始化后，Primary 设置 `statectl set --budget-json`，默认 dispatch 总额 40、Stage 4 实验总额 30，用户给出的上限优先；wallclock 仅用户指定时设置。并用同一有效预算初始化共享执行账本（harness 放 op 级目录，各函数共用）：

```bash
python3 .agents/tools/execution_receipt.py budget --budget <task>/execution/budget.json --max-experiments 30 --max-dispatches 40 [--max-wallclock-s <seconds>]
```

Primary 用同一组值同步 statectl 和执行账本；恢复或调高上限不会清空已用次数或任务开始时间。每次真实调度前由 Primary 执行 `execution_receipt.py reserve --budget <budget.json> --kind dispatch` 预占任务总额，成功后才调用 Subagent；statectl 继续记录各阶段转换与局部计数。harness 各函数共用同一账本，不能每函数重新获得 40 次调度。实验候选由共享账本在执行前预占；失败/超时也计一次，复用成功单元不再次计数。轮次按 workload 记录，但 max_experiments 是任务内所有 kernel/workload/重调度合计，非每个 workload 重新获得 30 次。常规测试和报告不计实验数，但受墙钟限制。

每个候选通过 `execution_receipt.py run --budget <budget.json> --experiment ... -- <runner command>` 执行。一次预占对应一个候选配置；autotune 的每个配置同样记账，不能用一个父 runner 把全部配置计作一个实验。runner 将共享预算路径传入内部子循环，每个测量/调试单元检查剩余时间；外层 timeout 不超过剩余 wallclock，超时结束整个子进程组并留失败收据。预算不足只交付已验证结果和未完成清单，不宣称全量调优完成。

Stage 4 保存完整 pair 的 checkpoint，身份包含 A/B 代码、runner、inventory、workload、采样参数、环境戳；不复用中断的半个 pair。`--resume --context-stamp <stamp>` 恢复匹配实验，仍保留最终完整复测。Stage 5 文件落位可以按输入指纹复用，集成包任何修复后仍跑完整 report。

## 5. 调优、并发与后处理

- 完整 workload inventory 不缩减。可以按同一 kernel 的结构/dtype/尾块分组，在候选分支上探索；中间检查受影响用例及 smoke，暂不更新已验收 current best。每批默认最多 3 个 workload 或到预算边界（先到者），合并后全量 tune 性能/精度和 smoke 精度验收；失败回退整批，再拆分定位。最终交付仍要求同一最终候选全量复测。
- 实验 runner 与 `ab_test.py` 共用 `summarize_pairs`。已满足 ≥3 对、交替 A/B 与 B/A、有效正样本和同一上下文的采样直接复用裁决；不再为相同协议重采一遍。inconclusive 时预算内追加完整 pairs；详细 profiling 仅按瓶颈归因需求采集。
- harness 依赖图、独立函数静态阶段并发与设备队列规则见 harness 场景 §4。conductor 串行更新状态，agent 各写自己的目录；任何设备执行和 profiling 全局串行。无并发调度能力、依赖不清或共享文件冲突时保持串行。
- 终态先报告已验证的交付路径、精度与性能范围；耗时分析、重渲染和蒸馏为同会话可恢复后处理，随后最终报告包含结果。不能声称已经后台执行；若会话中断，明确 pending 和恢复路径。进化审批简报在任务交付后汇总，除非用户正在要求审批，不等待审批完成当前任务。
- 三场景统一统计墙钟、人工等待、dispatch 次数、内部尝试/实验数、收据复用次数、重复验证耗时与恢复耗时。`python3 .agents/tools/execution_receipt.py summary --dir <task>` 汇总执行次数、失败、复用和锁等待（不重复累加被复用结果的执行耗时）。执行收据提供执行耗时，statectl 提供转换事实；session 导出提供人工等待和端到端时间，缺数据写 N/A，不能把并行阶段时长相加当总耗时。
