# RETROSPECTIVE — logsumexp_simd (op-level)

> migration-harness 场景：Stage 1/2/3 逐函数复盘见 `_logsumexp_kernel_single/RETROSPECTIVE.md` 与 `_logsumexp_kernel_tiled/RETROSPECTIVE.md`；本文件为 Stage 5 集成复盘（op 级单份）。

## Stage 5 (integrate) Retrospective — attempt1 — 2026-10-17

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| none | none | none | none | - |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| SIMD 双 kernel（single+tiled）logsumexp 一次集成通过，msprof 6 case 全有效，roofline util 34–66% | D | examples/TileOPs/reports/tileops/20261017_191533_378170_LogSumExpFwdOpSimd/run.json#summary（correctness 45/45, benchmark 6/6）；integration_log.md#验证结果 | `cd examples/TileOPs && python -m tileops.reporting.cli run --op LogSumExpFwdOpSimd --prof-mode msprof` | tilelang-fork commit f487023f + Ascend950PR_9579 + CANN 9.2.0 | pattern-library（reduction 族迁移基准数据） |
| wrapper 双 custom_op 共存模式：variant 算子注册独立 op 名（ascend::logsumexp_simd_fwd）即可与 SIMT 版（ascend::logsumexp_fwd）同进程共存，集成零冲突 | P | examples/TileOPs/tileops/kernels/reduction/logsumexp_simd/logsumexp_simd.py#custom_op 注释与注册块 | 集成包冒烟 + 全量 report 通过即证 | 同上 | pattern-library / integration-patterns |
| benchmark 观测：lm-head-logits [4,102400] fp16 延迟 6.65us vs bf16 0.66us（10x 差异，util 均正常）；3d-multidim-reduce [4,128,4096] ratio 仅 34.33%（全 shape 最低）——Stage 4 调优候选点 | D | examples/TileOPs/reports/tileops/20261017_191533_378170_LogSumExpFwdOpSimd/benchmark.json#records；run.json#performance | 同 run.json 复现命令；查看 perf records latency_us 与 roofline_utilization_percent | 同上 | pattern-library / bottleneck-patterns |

### Transferable Lessons（可迁移教训）

- 集成 SIMD 变体算子时，custom_op 注册名必须与既有变体区分（`ascend::{op}_simd_fwd` 风格），否则同进程 torch custom_op 注册冲突；wrapper 侧无需其他特殊处理。
- 两函数（single/tiled 分派）kernel 集成流程与单函数完全一致：integrate_kernel.py 按 extracted_functions 列表逐个复制并生成聚合 `__init__.py`，wrapper 的分派函数（`_logsumexp_simd_kernel`）保留在 wrapper 内即可，无需改动。
- Stage 5 前核对编程模式声明应以 kernel 文件头 docstring 为准（`programming_mode: simd` + `auto_schedule: yes`），integrate_kernel.py 的 `[prog-mode]` 输出即核对结果，一次通过无需人工重读 DESIGN 全文。
