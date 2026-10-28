# Integration Log -- LogSumExpFwdOp
- meta: tileops/kernels/reduction/logsumexp/.migration_meta.json
- integrated:
  - `_logsumexp_kernel_single` -> `tileops/kernels/reduction/logsumexp/logsumexp_kernel/_logsumexp_kernel_single.py`
  - `_logsumexp_kernel_tiled` -> `tileops/kernels/reduction/logsumexp/logsumexp_kernel/_logsumexp_kernel_tiled.py`
- design_docs:
  - `_logsumexp_kernel_single` -> `_logsumexp_kernel_single_DESIGN.md`（present）
  - `_logsumexp_kernel_tiled` -> `_logsumexp_kernel_tiled_DESIGN.md`（present）
- programming_mode（脚本 [prog-mode] 校验，与 DESIGN.md 一致）:
  - `_logsumexp_kernel_single`: simt（auto_schedule: yes）
  - `_logsumexp_kernel_tiled`: simt（auto_schedule: yes）
- aggregation `__init__.py`: 生成（re-export 两函数）
- wrapper `logsumexp.py`: 改写为 baseline/perf_opt 双 import 切换块（baseline 生效，perf_opt 两行注释待 Stage 4）
- import smoke: pass（`smoke-ok ['_logsumexp_kernel_single', '_logsumexp_kernel_tiled']` + LogSumExpKernel 类存在）
- attempts: 0（一次通过，未进调试闭环）

## 验证结果
- report_run_json: examples/TileOPs/reports/tileops/20261016_201213_480829_LogSumExpFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261016_201213_480829_LogSumExpFwdOp/report.md
- report_status: passed（CLI 退出码 0；结论以 run.json 为准）
- correctness: pass（45/45 用例，0 failed / 0 errors / 0 skipped；avg_max_abs_err=1.47e-07，max_abs_err=4.77e-07）
  - 覆盖 test_logsumexp_op（26 shape×dtype 组合）、test_logsumexp_keepdim（6）、test_logsumexp_non_contiguous（6）、test_logsumexp_1d（6）、test_logsumexp_accepts_multidim（1）
- benchmark: pass（6/6，profiler_fallback_count=0，口径 do_bench(backend="msprof")，device Ascend950PR_9579，CANN 9.2.0）
  - attn-weights-4k   float16  [32,32,4096]   config block_m=16, tile_n=0     -> latency 53.004 us, bw 0.158 TB/s
  - attn-weights-4k   bfloat16 [32,32,4096]   config block_m=16, tile_n=0     -> latency 52.966 us, bw 0.158 TB/s
  - attn-weights-32k  bfloat16 [32,32,32768]  config block_m=1,  tile_n=16384 -> latency 110.394 us, bw 0.608 TB/s
  - lm-head-logits    float16  [4,102400]     config block_m=1,  tile_n=25600 -> latency 20.350 us, bw 0.040 TB/s
  - lm-head-logits    bfloat16 [4,102400]     config block_m=1,  tile_n=25600 -> latency 20.377 us, bw 0.040 TB/s
  - 3d-multidim-reduce float16 [4,128,4096] dim=[0,2] config block_m=4, tile_n=0 -> latency 24.806 us, bw 0.169 TB/s
- run.json warnings: 无

## 调试历史
- 无（首次 report 即通过，无 attempt）
