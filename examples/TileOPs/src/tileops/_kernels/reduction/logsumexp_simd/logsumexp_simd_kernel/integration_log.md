# Integration Log -- LogSumExpFwdOpSimd

- meta: examples/TileOPs/tileops/kernels/reduction/logsumexp_simd/.migration_meta.json
- integrated:
  - `_logsumexp_kernel_single` -> `tileops/kernels/reduction/logsumexp_simd/logsumexp_simd_kernel/_logsumexp_kernel_single.py`
  - `_logsumexp_kernel_tiled` -> `tileops/kernels/reduction/logsumexp_simd/logsumexp_simd_kernel/_logsumexp_kernel_tiled.py`
- design_docs:
  - `_logsumexp_kernel_single` -> `_logsumexp_kernel_single_DESIGN.md`
  - `_logsumexp_kernel_tiled` -> `_logsumexp_kernel_tiled_DESIGN.md`
  - missing: none
- programming_mode: simd (SimdVF, auto_schedule yes) — 两函数 kernel 文件头 docstring 声明与 DESIGN.md / meta 一致（integrate_kernel.py `[prog-mode]` 输出核对，report 前复核）
- wrapper: `tileops/kernels/reduction/logsumexp_simd/logsumexp_simd.py` 改写为 baseline/perf_opt 双 import 切换块（baseline 激活，perf_opt 注释占位——本任务未做 Stage 4 调优）；`ascend::logsumexp_simd_fwd` custom_op 独立注册，未触碰 SIMT 版本 `tileops/kernels/reduction/logsumexp/` 与 `ascend::logsumexp_fwd`
- attempts: 0（一次集成成功，无调试闭环）

## 验证结果

- report_run_json: examples/TileOPs/reports/tileops/20261017_191533_378170_LogSumExpFwdOpSimd/run.json
- report_md: examples/TileOPs/reports/tileops/20261017_191533_378170_LogSumExpFwdOpSimd/report.md
- report_html: examples/TileOPs/reports/tileops/20261017_191533_378170_LogSumExpFwdOpSimd/report.html
- 命令: `python -m tileops.reporting.cli run --op LogSumExpFwdOpSimd --prof-mode msprof`（从 examples/TileOPs/ 运行，PYTHONPATH=/home/z00875089/tilelang-fork）
- status: passed；exit code 0；warnings: []
- correctness: pass — 45/45 用例（pytest junitxml：tests=45, passed=45, failed=0, errors=0, skipped=0），Avg Max Abs Error 9.17e-08
- benchmark: pass — 6/6 case（msprof 全部有效，`latency_source=do_bench(msprof)`，profiler_fallback_count=0）

### benchmark 数值（run.json / benchmark.json 摘要）

| Label | Shape | DType | tileops latency (us) | torch baseline (us) | roofline util (%) | BW (TB/s) | config |
|---|---|---|---:|---:|---:|---:|---|
| attn-weights-4k | [32,32,4096] | fp16 | 11.09 | 48.62 | 56.03 | 0.757 | block_m=16, tile_n=0 (single) |
| attn-weights-4k | [32,32,4096] | bf16 | 11.07 | 49.00 | 58.35 | 0.758 | block_m=16, tile_n=0 (single) |
| attn-weights-32k | [32,32,32768] | bf16 | 72.49 | 400.08 | 66.14 | 0.926 | block_m=1, tile_n=16384 (tiled) |
| lm-head-logits | [4,102400] | fp16 | 6.65 | 29.54 | 61.06 | 0.123 | block_m=1, tile_n=25600 (tiled) |
| lm-head-logits | [4,102400] | bf16 | 0.66 | 29.93 | 57.26 | 1.250 | block_m=1, tile_n=25600 (tiled) |
| 3d-multidim-reduce | [4,128,4096] | fp16 | 16.74 | 38.35 | 34.33 | 0.251 | block_m=4, tile_n=0 (single) |

## 调试历史

无。首次集成即通过（integrate_kernel.py 幂等执行 + import smoke pass → report 一次通过）。

## 集成产物清单

- tileops/kernels/reduction/logsumexp_simd/logsumexp_simd_kernel/_logsumexp_kernel_single.py
- tileops/kernels/reduction/logsumexp_simd/logsumexp_simd_kernel/_logsumexp_kernel_single_DESIGN.md
- tileops/kernels/reduction/logsumexp_simd/logsumexp_simd_kernel/_logsumexp_kernel_tiled.py
- tileops/kernels/reduction/logsumexp_simd/logsumexp_simd_kernel/_logsumexp_kernel_tiled_DESIGN.md
- tileops/kernels/reduction/logsumexp_simd/logsumexp_simd_kernel/__init__.py（聚合 re-export，脚本生成）
- tileops/kernels/reduction/logsumexp_simd/logsumexp_simd_kernel/integration_report.json
- tileops/kernels/reduction/logsumexp_simd/logsumexp_simd.py（wrapper import 切换块改写）

## 备注

- 编程模式核对：两函数均为 `programming_mode: simd` + `auto_schedule: yes`，单一 Kernel 类 `LogSumExpSimdKernel` 内未混用模式，无需类级模式字段处理。
- benchmark 数值仅记录（Stage 4 调优建议关注：3d-multidim-reduce ratio 34.33% 偏低；lm-head-logits fp16 与 bf16 延迟差异 6.65 vs 0.66 us）。
