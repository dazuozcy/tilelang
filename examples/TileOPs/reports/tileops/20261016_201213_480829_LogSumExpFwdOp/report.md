# TileOPs Evaluation Report: LogSumExpFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-16T20:13:39.497060+08:00 |
| evaluation_scope | LogSumExpFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261016_201213_480829_LogSumExpFwdOp |
| git_commit | 40c9d668776dbc9fd42bd7ed698a0c0091c81418 |
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

| Operator | Total Time | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|---:|
| LogSumExpFwdOp | 3h41m01s | 100.0% (45/45) | 1.47e-07 | 6 | N/A |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### LogSumExpFwdOp

<details>
<summary>Session Timing / Session 耗时 — 3h41m01s</summary>

| Phase / 环节 | Duration / 耗时 | Ratio / 占比 | Detail / 说明 |
|---|---:|---:|---|
| Stage 0 脚手架（scaffolder） | 15m45s | 7.10% | 08:34:28–08:50:13Z；Phase 0 通读 GPU 仓 + S1-S7 七文件 + 抽取脚本 + Tier 1 全过 |
| 函数1 S1 设计（含 1 次门禁窄修复） | 25m42s | 11.60% | 08:50:59–09:17:28Z；first_design 1298s + S1-PLACEHOLDER/CORES-LOGICAL/REF-PATHS 三项机械门禁误报窄修复 244s |
| 函数1 S2 设计检视 | 48m02s | 21.70% | 09:17:48–10:05:50Z；9 维度迁移检视 + 维度 8 独立推演 + 机械复核伪影消解（亲读解析器源码） |
| 函数1 S3 实现 | 19m01s | 8.60% | 10:06:17–10:25:18Z；首过 PRECISION_PASS，L0/L1/L2/Boundary 86 用例，fp16/bf16 位级一致 |
| 函数2 S1 设计（1 次异常中断 + 重试首过） | 37m38s | 17.00% | 10:25:54–11:03:59Z；att1 subagent 异常终止 603s（无工件产出，非门禁失败）+ att2 完整设计 1655s（gate 1 零失败零警告——前序教训规避误报生效） |
| 函数2 S2 设计检视 | 49m54s | 22.60% | 11:04:16–11:54:10Z；9 维度检视 + 真机探针闭合尾 chunk reduce 先例空隙（review_probe_tail_reduce.py PROBE_PASS） |
| 函数2 S3 实现 | 16m37s | 7.50% | 11:54:28–12:11:05Z；首过 PRECISION_PASS，61 用例，fp16/bf16 位级一致 |
| Stage 5 集成验证（integrator） | 3m55s | 1.80% | 12:11:19–12:15:14Z；一次通过（无调试闭环）：45/45 正确性 + 6/6 benchmark（do_bench(msprof)）+ report passed |
| conductor 编排开销 | 4m27s | 2.00% | 状态机迁移（migration init/func/phase/integrate）、机械门禁执行、dispatch 间隔；9 次 dispatch 合计 12994s，总区间 13261s 之差 |

</details>

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---:|
| attn-weights-4k | 53.0042 | N/A | [32, 32, 4096] | float16 | msprof | 0.1583 |
| attn-weights-4k | 52.9656 | N/A | [32, 32, 4096] | bfloat16 | msprof | 0.1584 |
| attn-weights-32k | 110.3945 | N/A | [32, 32, 32768] | bfloat16 | msprof | 0.6079 |
| lm-head-logits | 20.3497 | N/A | [4, 102400] | float16 | msprof | 0.0403 |
| lm-head-logits | 20.3773 | N/A | [4, 102400] | bfloat16 | msprof | 0.0402 |
| 3d-multidim-reduce | 24.8058 | N/A | [4, 128, 4096] | float16 | msprof | 0.1691 |
