# TileOPs Evaluation Report: LogSumExpFwdOpSimd

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-19T08:51:45.978666+08:00 |
| evaluation_scope | LogSumExpFwdOpSimd |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261019_084947_709899_LogSumExpFwdOpSimd |
| git_commit | f487023f3aa3f000d35f5f98efdb711705fac167 |
| correctness_target | tests/ops/test_logsumexp_simd.py |
| benchmark_target | benchmarks/ops/bench_logsumexp_simd.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend950PR_9579 x 2 |
| cpu | x86_64 |
| cann | 9.2.0 |
| driver | 25.7.0 |
| pytorch | 2.10.0+cpu |
| pytorch_npu | 2.10.0.post5.dev20260815 |
| python | 3.11.16 |
| os | Linux-5.4.0-216-generic-x86_64-with-glibc2.31 |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 45 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| LogSumExpFwdOpSimd | 100.0% (45/45) | 9.17e-08 | 6 | 35.02% – 75.83% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的 roofline 性能占比范围（msprof 工具内置 SoC 模型计算，GM Read + Write 点；不可得时回退瓶颈管线占用率）。

## Operator Details / 算子明细

### LogSumExpFwdOpSimd

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---:|
| attn-weights-4k | 10.3301 | 61.40 | [32, 32, 4096] | float16 | msprof | 0.8123 |
| attn-weights-4k | 10.2991 | 61.83 | [32, 32, 4096] | bfloat16 | msprof | 0.8147 |
| attn-weights-32k | 51.2489 | 75.83 | [32, 32, 32768] | bfloat16 | msprof | 1.3095 |
| lm-head-logits | 5.7869 | 62.58 | [4, 102400] | float16 | msprof | 0.1416 |
| lm-head-logits | 5.7961 | 61.16 | [4, 102400] | bfloat16 | msprof | 0.1413 |
| 3d-multidim-reduce | 16.3098 | 35.02 | [4, 128, 4096] | float16 | msprof | 0.2572 |
