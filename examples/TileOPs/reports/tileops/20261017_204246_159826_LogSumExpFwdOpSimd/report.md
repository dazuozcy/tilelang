# TileOPs Evaluation Report: LogSumExpFwdOpSimd

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-17T20:46:02.267891+08:00 |
| evaluation_scope | LogSumExpFwdOpSimd |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261017_204246_159826_LogSumExpFwdOpSimd |
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
| 97.8% | 1 | 45 | 1 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| LogSumExpFwdOpSimd | 97.8% (44/45) | 7.63e-08 | 0 | N/A |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的 roofline 性能占比范围（msprof 工具内置 SoC 模型计算，GM Read + Write 点；不可得时回退瓶颈管线占用率）。

## Operator Details / 算子明细

### LogSumExpFwdOpSimd

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---:|
| - | No performance records | - | - | - | - | - |

## Failed Cases / 失败用例

| Stage | Test | Outcome | Message |
|---|---|---|---|
| Correctness | tests.ops.test_logsumexp_simd::test_logsumexp_simd_op[shape0--1-dtype0] | failed | RuntimeError: LazySetDevice:../torch_npu/csrc/core/npu/NPUFunctions.cpp:481 NPU function error: err, error code is 507033 [ERROR] 2026-10-17-20:45:44 (PID:2293492, Device:0, RankID:-1) ERR00100 PTA call acl api failed [Error]: Failed to start the device.          Rectify the fault based on the error information in the ascend log. [PID: 2293492] 2026-10-17-20:45:31.280.628 Inner_Error_Device_Subprocess_Startup_Timeout(E39007): Starting a subprocess on the device timed out.         Possible Cause: 1.Failed to start the subprocess. 2. The HDC link is faulty.         Solution: Obtain the host and device logs, and contact technical support at https://www.hiascend.com/support. TraceBack (most recent call last):         tsd client wait response fail, hostpid:2293492, receive device response data failed, check hdc service.[FUNC:WaitRsp][FILE:tsd_process_controller.cpp][LINE:196]         Starting a subprocess on the device timed out.         TsdOpen failed. devId=0, tdt error=31.[FUNC:PrintfTsdError][FILE:runtime.cc][LINE:2146]         Check param failed, dev can not be NULL![FUNC:DeviceRetain][FILE:runtime.cc][LINE:3186]         rtSetDevice execution failed, reason=device retain error[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:72] |
