# Autotune 引用索引

## 用途

当 Phase 2 的候选优化点包含 `autotune 参数搜索` 时，读取本文件。

本文件不复制 tilelang-fork autotuner 的完整 API 和示例，避免仓库更新后两边内容不同步。具体用法、参数、示例和实现细节，以仓库文档与源码为准（负向断言须按 AGENTS.md「Docs Auto Routing」举证）。

## 先读这些源文件（tilelang-fork 仓库内）

优先读取官方文档：

1. `docs/programming_guides/autotuning.md`（权威使用指南：`@tilelang.autotune` 装饰器式 + `AutoTuner.from_kernel(...).set_*().run()` 程序式两种工作流、输入张量供给、容差、缓存与环境变量）

需要确认实现细节时读取：

1. `tilelang/autotuner/`（tuner.py / param.py）
2. `tilelang/profiler/bench.py`（do_bench API 形态与单位）

需要参考可运行示例时读取（仓库内）：

1. `examples/gdn/example_chunk_delta_h.py`（`@autotune` 扫描配置）
2. `examples/deepseek_nsa/benchmark/benchmark_nsa_fwd.py`（`@tilelang.autotune`）
3. `examples/quickstart.py`（tuned kernel 的 profiler 用法）
4. `examples/hadamard_transform/example_hadamard.py`、`examples/dynamic_shape/example_dynamic.py`

需要参考结构性优化模式时，优先搜索并阅读：

1. `examples/ascend/`（deepgemm 持久化遍历/L0 staging/混合 epilogue、gemm_splitk、flash_attention、vecadd/rmsnorm/per_token_cast_to_fp8 的 simt/simd 对照）
2. `testing/ascend/`（语言特性回归；勿将应用 kernel 导入该 suite）
3. 历史优化目录或同类算子的 `perf_opt/opt_log.md`（A5 冷启动期可能不存在，检索 pattern-library cases.md）

如果历史目录存在 `Optimize.md`，只能把它当作快速摘要；完整过程仍以 `perf_opt/opt_log.md` 为准。

## 在本 skill 中的使用规则

Autotune 属于 Phase 2 的一个实验分支，通常放在结构性优化之后执行。结构性分支通过正确性且方向有效后，应先对该结构做一轮 coarse autotune 或等价手动粗搜；autotune winner 不是最终结论，之后还要检查 top-k 配置和 winner 邻域，再进入精测和 winner 收束。

执行前必须满足：

- baseline kernel 已经通过正确性测试。
- 已经用 `do_bench(fn, backend="msprof")` 得到目标 kernel 的 baseline 时延。
- kernel 代码中已经暴露显式可调参数，例如 tile shape（block_M/block_N/block_K）、`T.Pipelined` num_stages、SimtVF threads、grid 大小或 buffer 数量。
- 如果结构性优化已识别为持久化/grid 划分方向，应显式暴露 grid 大小（`T.Kernel(N)` 的 N）或 `iters` 类等价参数。
- 若搜索 grid 大小，搜索空间必须覆盖核数实查值附近、整数倍核数、低 imbalance 候选和 flat-grid 端点，不能只测单点（核数以 `torch.npu.get_device_properties()` 实查为准，惯例初值 32 待标定——CONST-arch-a5）。
- 搜索空间小而合法，优先过滤明显超过约束的配置：UB/L1/L0 容量（CONST-capacity-a5，含 SimtVF 时 UB 上限更低）、C0 分形对齐（CONST-gemm-dim-constraints）、一维 grid 与 shape 整除性。

结构后 coarse search：

- `T.Pipelined` / multibuffering 分支至少粗搜 tile size × num_stages 中实际暴露的关键参数。
- 持久化/grid 划分分支至少粗搜 grid 大小曲线；`iters` 若由 `ceildiv(num_logical, grid)` 推导，不必单独暴露。
- coarse search 只服务当前结构，不混入其它结构性改动；若 coarse winner 周围存在甜点区，再扩展邻近参数精搜。
- coarse search 结果只用于筛候选，最终 winner 必须单独用 do_bench(msprof) 复测。

Autotune 后邻域精搜：

- 记录 autotune winner 和 top-k 配置：程序式路径用 `AutoTuner.run(on_trial=...)` 回调收集逐 config 记录（`status` / `index` / `config` / `latency_ms` / `validation` / `error`）；若不能直接导出 top-k，则至少保留所有已测 config 的 latency / correctness / failure reason。
- 围绕 winner 做邻域搜索，例如 tile 上下相邻合法档位、grid 大小上下相邻档位、低 imbalance 候选和资源余量更好的候选。
- 若 autotune latency 与 do_bench(msprof) 时延排序冲突，以独立复测的 do_bench(msprof) 决胜。
- 邻域精搜可以手动或脚本执行，但必须记录 search space、结果和为什么停止扩展。

执行时必须记录：

- search space。
- best config。
- top-k config 或所有已测 config 的摘要。
- autotune latency（注意单位：autotuner 与 do_bench 均报告毫秒；`perf_records.jsonl` 的 `duration_us` = 毫秒值 ×1000）。
- 正确性结果。
- 编译失败或 correctness 失败的配置数量。
- 每个候选的 grid 大小、理论 logical tile 数、UB/L1/L0 buffer 占用估算。

执行后必须：

- 用 best config 生成当前 autotune 实验分支，例如 `perf_opt/{op}_round{iter}_{opt_id}.py`。
- 若执行了邻域精搜，用精搜 winner 生成最终候选分支，并记录它与 autotune winner 的关系。
- 跑 L0 精度回归。
- 对 winner 单独跑 `do_bench(fn, backend="msprof")`。
- 最终性能报告使用单独复测的 do_bench(msprof)，不直接使用 autotune latency 作为最终结果。

## Ascend 特有注意

- compile 目标传 `target="ascend"`（`AutoTuner.set_compile_args(target=...)`；decorator 场景由 `@tilelang.jit(target="ascend")` 决定）。
- 用 `set_autotune_inputs(A, B, C)` 或 `supply_prog` 固定输入，保证各 config 在同一数据上测量；动态维度（symbolic shape）必须显式供给具体输入。
- 搜索空间中的 num_stages 影响 AutoSchedule multibuffering 的 buffer versions 与 UB 占用（版本维度由编译器加），容量超限的 config 会编译失败——这正是预先过滤的意义。
- 缓存：`TILELANG_CACHE_DIR`（默认 `~/.tilelang/cache`）、`TILELANG_AUTO_TUNING_DISABLE_CACHE`；比较结果时注意 cache hit（`on_trial` 只发一个 `cache_hit` 标记，不重建 trial 历史）。
- 手动批量编译替代路径：`JITImpl.par_compile`（见 autotuning.md「Alternative: Manual Sweeps」），适合自驱 do_bench 测量的受控实验。

## 实验批处理 runner（T-4）

Phase 2 的实验分支执行（L0 回归 → do_bench(msprof) A/B 测量 → perf_records.jsonl 追加 → 汇总表）应交给批处理脚本而非 agent 逐个往返驱动：

1. **首轮生成**：先建立 `workload_inventory.json`（benchmark 展开或用户指定 workload），再从模板 [run_experiments_template.py](run_experiments_template.py) 拷贝到 `perf_opt/run_experiments.py`，适配 `kernel_id`、被测 kernel 的构造方式（`load_runner`）与 A/B 实现文件；模板只测量 inventory 中该 kernel 的 tune workload，smoke 精度回归另按 skill 执行。
2. **后续轮直接调用**：模板对同一 workload 下的两个实现（baseline/current best 与候选）做 do_bench(msprof) **交错重复测量**（A/B/A/B 次序、每轮多次采样取中位），输出相对比较表；脚本逐测量追加性能记录（append-only，字段契约 signal-registry §5：`duration_us` = 毫秒 ×1000、`profiler_output`）。
3. **决策留在 agent**：读表决策 winner/rollback/下一轮候选——脚本只执行不判断；`<5%` 差异与平区决胜仍按 SKILL.md Phase 2 第 9 步走 `ab_test.py` 交错多 run 协议（do_bench(msprof) 封装）。
4. 收益：单轮 agent 交互轮次下降 60%+；大工件会话超限类失败（VP-2026-0021：两次空返回各耗 1579s/930s，旧栈实证）被根治——长链路工具调用收敛为一次脚本执行。
5. 脚本输出与 perf_records.jsonl 对账（append-only 纪律不变），gate 4 机械校验照常生效。

## 口径约束

不要把 autotune 理解为自动完成所有性能优化。它只是在给定搜索空间中选择候选配置；winner 需要经过 top-k / 邻域检查和独立复测（do_bench(msprof)）。

对 elementwise / activation / 轻量 vector kernel，先判断是否存在派发/驻留开销问题。若 grid 大小远超核数实查值，优先把持久化遍历（`T.Persistent` + `tile_schedule` 核-tile 映射，参考 `examples/ascend/deepgemm/`）作为结构性改写，再 autotune tile 与 grid 参数，不要只扩大 tile 或 num_stages 搜索空间。grid 大小不只是划分数，也是在给硬件提供合适的软件任务并发；太少会并发不足，太多会派发开销过高。

如果所有 config 的 do_bench(msprof) 时延接近，先用更多采样与独立运行复测（同 workload 重复运行方差 ≤1%）。复测后仍打平时，使用 imbalance、整除性和资源余量决胜；若仍无法区分，回到 Phase 2 重新分析当前现象，而不是继续扩大搜索空间。

如果 tilelang-fork 中 autotune API、参数名或示例发生变化，以仓库文档与源码为准，更新本文件中的引用路径或极少量执行规则即可。
