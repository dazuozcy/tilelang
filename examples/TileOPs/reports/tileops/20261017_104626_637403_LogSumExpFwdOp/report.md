# TileOPs Evaluation Report: LogSumExpFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-17T10:47:25.733666+08:00 |
| evaluation_scope | LogSumExpFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261017_104626_637403_LogSumExpFwdOp |
| git_commit | 73a4b92d5fe88ab730e32a1eaa944de875ce617c |
| correctness_target | tests/ops/test_softmax.py |
| benchmark_target | benchmarks/ops/bench_softmax.py |

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
| LogSumExpFwdOp | 100.0% (45/45) | 1.47e-07 | 6 | 44.81% – 82.28% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的瓶颈管线占用率范围（msprof_detail 周期口径，实测最忙 pipe 的 busy ratio）。

## Operator Details / 算子明细

### LogSumExpFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---:|
| attn-weights-4k | 21.4302 | 72.99 | [32, 32, 4096] | float16 | msprof | 0.3915 |
| attn-weights-4k | 21.3533 | 71.88 | [32, 32, 4096] | bfloat16 | msprof | 0.3929 |
| attn-weights-32k | 100.2530 | 82.28 | [32, 32, 32768] | bfloat16 | msprof | 0.6694 |
| lm-head-logits | 17.6544 | 51.63 | [4, 102400] | float16 | msprof | 0.0464 |
| lm-head-logits | 17.5977 | 51.92 | [4, 102400] | bfloat16 | msprof | 0.0466 |
| 3d-multidim-reduce | 19.7425 | 44.81 | [4, 128, 4096] | float16 | msprof | 0.2125 |
