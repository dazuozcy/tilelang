# Integration Log -- IndexedExpertMLPFwdOp
- meta: examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/.migration_meta.json
- integrated:
  - `_route_stats` -> `src/tileops/_kernels/moe/indexed_expert_mlp/indexed_expert_mlp_kernel/_route_stats.py`
  - `_indexed_expert_gemm` -> `src/tileops/_kernels/moe/indexed_expert_mlp/indexed_expert_mlp_kernel/_indexed_expert_gemm.py`
  - `_weighted_reduce` -> `src/tileops/_kernels/moe/indexed_expert_mlp/indexed_expert_mlp_kernel/_weighted_reduce.py`
- design_docs:
  - `_route_stats` -> `_route_stats_DESIGN.md`
  - `_indexed_expert_gemm` -> `_indexed_expert_gemm_DESIGN.md`
  - `_weighted_reduce` -> `_weighted_reduce_DESIGN.md`
- wrapper: `src/tileops/_kernels/moe/indexed_expert_mlp/indexed_expert_mlp.py` rewritten -> baseline/perf_opt kernel-source selection block (baseline active, perf_opt commented placeholders for all 4 re-exported names incl. `_route_metadata_size`)
- programming_mode (kernel docstring, 与 DESIGN.md 一致，集成时与重复集成均核对):
  - `_route_stats`: simd, auto_schedule: yes
  - `_indexed_expert_gemm`: mixed, auto_schedule: yes
  - `_weighted_reduce`: simd, auto_schedule: yes
- attempts: 0（调试闭环未触发；集成前 1 次胶水层预处理 + 1 次集成脚本运行 + 1 次 report 全绿）

## 集成前置修正（非调试闭环 attempt）
- 症状：wrapper（Stage 0 产物）使用绝对导入 `from tileops._kernels.moe.indexed_expert_mlp._indexed_expert_m_l_p_fwd_kernels import (...)`，而 integrate_kernel.py 的 `parse_wrapper_imports`（仅识别 `level==1` 相对导入）与 `rewrite_wrapper_import`（regex `from \.{module}`）均只消费相对导入形式；直接运行会报 `[error] wrapper ... has no 'from ._...' import`。
- 分类：wrapper 胶水错误（import 形式），非前置产物缺失——提取函数与模块均存在且名字完全一致，语义等价，属允许修改范围（import 胶水层）。
- 修复：改写为相对导入 `from ._indexed_expert_m_l_p_fwd_kernels import (...)`（备份：`history_version/indexed_expert_mlp_s5_attempt1.py`）。
- 结果：集成脚本一次通过，import 冒烟 pass（含 4 个重导出名与 4 个 wrapper Kernel 类断言）。

## 验证结果
- report_run_json: examples/TileOPs/reports/tileops/20261029_204631_566537_IndexedExpertMLPFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261029_204631_566537_IndexedExpertMLPFwdOp/report.md
- run_id: 20261029_204631_566537_IndexedExpertMLPFwdOp（CLI 输出 status=passed，run.json status 字段为 passed，warnings=[]）
- correctness: pass — 18/18 用例通过（0 failed / 0 errors / 0 skipped），avg max abs error 2.14e-06（来自 report.md 算子明细）
- benchmark: pass — 8/8 case（4 workload × fp16/bf16）msprof 采集成功，profiler_fallback_count=0，roofline ratio 80.03%–97.75%：
  - ds-v3-t1:     10032.22 us (fp16)  / 10020.90 us (bf16), ratio 97.71%/97.75%
  - ds-v3-t32:   466984.31 us (fp16)  / 466913.19 us (bf16), ratio 90.31%/90.09%
  - ds-v3-t64:  1113695.56 us (fp16)  / 1113780.11 us (bf16), ratio 85.58%/85.61%
  - qwen3-235b-t32: 232647.98 us (fp16) / 358497.83 us (bf16), ratio 80.03%/83.62%
- 备注（仅记录，不修复）：
  - bench 期间 auto_schedule 反复告警 "Python Z3 scheduler failed ... solver returned unsat, falling back to topological sort"——良性回退，正确性与 benchmark 结论均通过，未做处理。
  - msprof 屋顶线采集较慢（约 6–10 分钟/case，8 case 全程约 65 分钟），为大规模权重 MoE workload 的正常开销。

## 调试历史
- 无（attempt 0；集成脚本与 report 均一次通过）。
