# RETROSPECTIVE -- indexed_expert_mlp (op 级)

## Stage 5 (integrate) Retrospective — migration-harness — 2026-10-29

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | Stage 0 产出的 wrapper 使用绝对导入（`from tileops._kernels.moe...{extracted_module} import ...`），而 integrate_kernel.py 的 `parse_wrapper_imports` 只识别 level==1 相对导入、`rewrite_wrapper_import` 的 regex 只匹配 `from \.{module}`，直接运行报 `[error] wrapper ... has no 'from .{module} import'`，需 Stage 5 手工把 import 改为相对形式才能集成 | examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/indexed_expert_mlp_kernel/history_version/indexed_expert_mlp_s5_attempt1.py#L57-62（改前绝对导入）；examples/TileOPs/.agents/skills/add-ascend-op/scripts/integrate_kernel.py#L146-155,L248-272（仅消费相对导入） | 让 add-ascend-op Stage 0 生成相对导入（`from .{extracted_module} import ...`），或让 integrate_kernel.py 的解析/改写同时接受绝对包内导入形式 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| wrapper import 形式适配：integrate_kernel.py 只认相对导入，集成前先把 Stage 0 wrapper 的绝对包内导入改写为 `from .{extracted_module} import ...` | P | examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/indexed_expert_mlp_kernel/integration_log.md#集成前置修正 | 对任意 Stage 0 产物 wrapper 含 `from tileops._kernels.<family>.<op>._<op>_kernels import ...`（绝对形式）直接运行 integrate_kernel.py → `[error] wrapper has no 'from ._...' import`；改为相对导入后一次通过 | tilelang 0.1.15（fork build /home/l00970450/z00875089/tilelang-fork，report git_commit a2a92cf9）+ Ascend950PR_9579 + CANN 9.2.0 | pattern-library / 待定 |
| MoE 大权重 workload 的 msprof 屋顶线采集耗时量级：8 bench case（E=256/128，权重 0.9–15GB 级）约 65 分钟，约 6–10 分钟/case | D | examples/TileOPs/reports/tileops/20261029_204631_566537_IndexedExpertMLPFwdOp/run.json（benchmark 8 case）+ 进程树观察（逐 case `msprof op --aic-metrics=Roofline`） | `cd examples/TileOPs && PYTHONPATH=src:.:../.. python -m tileops.reporting.cli run --op IndexedExpertMLPFwdOp --prof-mode msprof`，观察 benchmark 阶段耗时 | tilelang 0.1.15（build /home/l00970450/z00875089/tilelang-fork）+ Ascend950PR_9579 + CANN 9.2.0 + msprof（cann-9.2.0） | 待定（编排/预算估计用） |
| auto_schedule Z3 unsat 回退 topological sort 告警在本算子 bench 全程反复出现，正确性与 benchmark 结论不受影响（status=passed，profiler_fallback_count=0）——此类告警可按良性记录，不作失败信号 | C | examples/TileOPs/reports/tileops/20261029_204631_566537_IndexedExpertMLPFwdOp/pytest/benchmark.log（多次 Z3 fallback 告警）对照同目录 run.json summary（全绿） | 同上 report 命令；观察 benchmark.log 中 `Z3 scheduling failed: solver returned unsat, falling back to topological sort` 与最终 status=passed 并存 | tilelang 0.1.15（build /home/l00970450/z00875089/tilelang-fork，src/ascend/transform/auto_schedule.cc:901）+ Ascend950PR_9579 + CANN 9.2.0 | pattern-library / 待定 |

### Transferable Lessons（可迁移教训）

- 集成前检查 wrapper 的 extracted import 形式：integrate_kernel.py 只消费相对导入 `from .{extracted_module} import ...`；Stage 0 wrapper 若写成 `from tileops._kernels....{extracted_module} import ...`（绝对形式），先改写为相对形式再运行集成脚本（语义等价，属允许的 import 胶水层修改，改前备份）。
- bench/report 期间 `Z3 scheduling failed: solver returned unsat, falling back to topological sort` 是 auto_schedule 的良性回退告警，不要当作失败信号；以 run.json 的 status 与 summary 为准。
- 大权重 MoE 算子（专家权重 GB 级）的单算子 msprof report 全程约 1 小时以上（8 case × 6–10 分钟屋顶线采集），编排阶段应为 Stage 5 benchmark 预留充足时间预算，勿按小算子分钟级估计。
