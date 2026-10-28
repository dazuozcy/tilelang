# TileOPs Evaluation Report: LogSumExpFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-17T11:17:47.802409+08:00 |
| evaluation_scope | LogSumExpFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261017_111528_067406_LogSumExpFwdOp |
| git_commit | fe70f2322499613d3ecbd2206a7559f9848fdefd |
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
| LogSumExpFwdOp | 100.0% (45/45) | 1.47e-07 | 6 | 4.98% – 78.53% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的 roofline 性能占比范围（msprof 工具内置 SoC 模型计算，GM Read + Write 点；不可得时回退瓶颈管线占用率）。

## Operator Details / 算子明细

### LogSumExpFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---:|
| attn-weights-4k | 21.4967 | 44.11 | [32, 32, 4096] | float16 | msprof | 0.3903 |
| attn-weights-4k | 21.4931 | 36.04 | [32, 32, 4096] | bfloat16 | msprof | 0.3904 |
| attn-weights-32k | 87.3101 | 78.53 | [32, 32, 32768] | bfloat16 | msprof | 0.7687 |
| lm-head-logits | 17.6954 | 4.99 | [4, 102400] | float16 | msprof | 0.0463 |
| lm-head-logits | 17.6743 | 4.98 | [4, 102400] | bfloat16 | msprof | 0.0464 |
| 3d-multidim-reduce | 20.3819 | 28.23 | [4, 128, 4096] | float16 | msprof | 0.2058 |
