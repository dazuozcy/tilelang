# TileOPs Evaluation Report: LogSumExpFwdOpSimd

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-17T19:19:11.032375+08:00 |
| evaluation_scope | LogSumExpFwdOpSimd |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261017_191533_378170_LogSumExpFwdOpSimd |
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

| Operator | Total Time | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|---:|
| LogSumExpFwdOpSimd | 6h23m49s | 100.0% (45/45) | 9.17e-08 | 6 | 34.33% – 66.14% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的 roofline 性能占比范围（msprof 工具内置 SoC 模型计算，GM Read + Write 点；不可得时回退瓶颈管线占用率）。

## Operator Details / 算子明细

### LogSumExpFwdOpSimd

<details>
<summary>Session Timing / Session 耗时 — 6h23m49s</summary>

| Phase / 环节 | Duration / 耗时 | Ratio / 占比 | Detail / 说明 |
|---|---:|---:|---|
| 初始化·场景路由·需求预检·AskUserQuestion | 16m13s | 4.20% | 05:01:03–05:17:16 UTC，会话启动至 Stage 0 首个子 Agent 调度前 |
| Stage 0 脚手架（tileops-scaffolder） | 13m04s | 3.40% | 05:17:16–05:30:20 UTC，7 文件移植 + .migration_meta.json |
| [single] Stage 1-3 设计+检视+开发 | 1h30m21s | 23.50% | 05:32:57–07:05:30 UTC，含 Stage 1 gate 修复 63s |
| [tiled] Stage 1 设计（含 gate 修复 + 一次修订） | 1h08m00s | 17.70% | 07:07:55–09:16:47 UTC，3 次 dispatch（2960s+193s+927s） |
| [tiled] Stage 2 检视（含 2 次复检，其中 2 次 compaction 失败） | 2h25m45s | 38.00% | 08:01:18–10:45:50 UTC，4 次 dispatch（3470s+1608s+1265s+2303s） |
| [tiled] Stage 3 开发 | 25m30s | 6.60% | 10:46:36–11:12:07 UTC，[PRECISION_PASS] 69 用例 |
| Stage 5 集成（tilelang-op-integrator） | 6m24s | 1.70% | 11:14:29–11:20:53 UTC，correctness 45/45 + benchmark 6/6 |
| 编排间隙与收尾 | 18m32s | 4.80% | 各阶段间 conductor 状态机/statectl/kb_search/收尾等 |

</details>

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---:|
| attn-weights-4k | 11.0899 | 56.03 | [32, 32, 4096] | float16 | msprof | 0.7566 |
| attn-weights-4k | 11.0714 | 58.35 | [32, 32, 4096] | bfloat16 | msprof | 0.7579 |
| attn-weights-32k | 72.4916 | 66.14 | [32, 32, 32768] | bfloat16 | msprof | 0.9258 |
| lm-head-logits | 6.6517 | 61.06 | [4, 102400] | float16 | msprof | 0.1232 |
| lm-head-logits | 0.6553 | 57.26 | [4, 102400] | bfloat16 | msprof | 1.2500 |
| 3d-multidim-reduce | 16.7352 | 34.33 | [4, 128, 4096] | float16 | msprof | 0.2506 |
