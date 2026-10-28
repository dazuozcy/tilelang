# 性能采集流程

本文档服务 `tilelang-op-optimize` 的 Stage 4。性能调优对象是 benchmark 实际参数化的、可运行且非 smoke 的 workload；合并对象是它们实际调用的 Stage 3 kernel。**时延口径：`tilelang.profiler.do_bench(fn, backend="msprof")` 为唯一端到端时延口径（返回毫秒，×1000 记 µs），npusim 为唯一周期级归因口径**（权威：`tilelang-ascend` skill §12 + `references/profiling.md`）；不使用 NPU event、host wall-clock 等其他时延口径。

> **TileOPs 依赖检查**：确认本地 A5 集成算子与迁移后 benchmark 入口已存在；缺失时报告依赖，不能用 GPU 源仓 API 代替。来源边界遵循 `_shared/standards/migration-source.md`。

---

## 1. 建立 benchmark workload 清单

TileOPs 迁移/集成场景（本地 A5 算子已集成时）读取迁移后的 benchmark 参数化入口，展开每个案例的 `workload_id`、shape、dtype、影响 kernel 的参数和 benchmark 来源。benchmark 可以通过 helper 从 manifest 生成参数，但**只取 benchmark 实际展开的案例**；不得额外枚举 manifest、L0/L1/L2/Boundary 测试或补造 benchmark 中不存在的 case。只提取 workload 参数，不要求先运行正式 benchmark、接入 wrapper 或注入候选实现。非 TileOPs、确无迁移 benchmark 的独立算子使用用户明确指定的性能 workload，逐 kernel/逐 workload 建立 inventory；`benchmark_source` 记为 `explicit_target`。

逐案例确认它实际调用的 Stage 3 kernel，记录可唯一定位实现文件及函数的 `kernel_id` 与被测 kernel 的构造/调用方式（用于测量时的 callable 边界与追溯）。一个案例若调用多个目标 kernel，分别登记 workload–kernel 对；无法确认映射时记录原因并停止该案例的性能采集。TileOPs 原有 op 层路由只用于查明映射，不是 Stage 4 调优单位，也不应修改。

分类顺序：benchmark 显式 skip → `skipped`（保留 reason）；`pytest.mark.smoke` 或案例 ID / 明确 workload label 中独立的 `smoke` 词（大小写不敏感，以 `-`、`_` 等分隔，如 `smoke-1m`）→ `smoke`；其余 → `tune`。若 `full` 标记与 `smoke` ID/label 冲突，记录冲突，仍归为 `smoke`。不得凭 shape 小、参数化顺序或 manifest 注释推断 smoke。

将完整清单写入 `perf_opt/workload_inventory.json`，契约见 `_shared/standards/signal-registry.md` §5。`tune` 项进入性能 baseline、调优与性能回退门禁，并在完成后写 `tuning_status`；`smoke` 项仅用于每次合并后及最终的精度回归，不采集性能 baseline、不选 winner、不参加性能回退门禁；`skipped` 项只记录原因。无 `tune` 项的 kernel 记录 `no_tunable_workload`，不虚构性能结果。

---

## 2. 逐 kernel 采集 baseline 并排序

一个 optimizer subagent 逐个 kernel 执行。对当前 kernel 的**所有** `tune` workload 串行采集 baseline，确认每项测量有效后再决定调优顺序：优先实测耗时较大、瓶颈明确且可能有结构性收益的大 shape；随后尽早处理小 shape、其他 dtype、尾块和资源临界案例；其余按实测瓶颈与收益机会排序。排序与理由写入 `opt_log.md`，不得因排序跳过 workload。每个 workload 完成迭代后才尝试核内合并，并复测当前 kernel 的全部 `tune` workload。

---

## 3. 串行运行 do_bench（msprof）

对每个 `kernel_id + workload_id` 逐个测量，**禁止并发 profiling**。使用 benchmark workload 参数直接构造 Stage 3 kernel 或 `perf_opt/` 下实际候选文件的 kernel 对象，**不运行需要 Stage 5 wrapper 的正式 benchmark**。

测量模板（权威 API 形态以 `tilelang/profiler/bench.py` 当前版本为准）：

```python
import statistics
from tilelang.profiler import do_bench


def run():  # ADAPT: 用 workload 参数调用被测 kernel 对象（单次发射）
    kernel(x, w)


# 端到端时延（msprof backend）：do_bench 返回毫秒
samples_ms = [do_bench(run, backend="msprof", warmup=25, rep=100, early_stop_baseline=None) for _ in range(5)]
latency_us = statistics.median(samples_ms) * 1000  # ms -> us

# 深挖（可选）：KernelProfile —— dur_ns、AIC/AIV cycle counts、pipe ratios
profile = do_bench(run, backend="msprof_detail", early_stop_baseline=None)
detail_us = profile.dur_ns / 1_000
```

要点：

- `early_stop_baseline=None`：受控对比必须禁用提前停止估计（事件估计是不同测量路径，不得与 msprof 样本混合）。
- `warmup`/`rep` 单位毫秒；比较既有结果时沿用原 harness 参数，并在日志记录所用参数。
- **被测 callable 边界在两个实现间必须一致**（是否含分配、输出初始化、多 kernel、同步）；首 JIT、模块加载、profiler 初始化与稳态测量分离；NPU 设备在准备张量前选定并与测量上下文一致。
- helper 自带 cache flush（预热 flush kernel 并从解析结果中过滤该 kernel）；若 callable 发射多个 kernel，先确认聚合语义再解释（必要时用 `msprof_detail` 逐 kernel 检查个体记录）。
- 每次测量由多次 do_bench 采样取中位数；采样数与独立运行次数写入日志。复测稳定性以**同 workload 重复运行方差 ≤1%** 为对账阈值（signal-registry §5）。
- 实验 stdout/stderr 不写到 `perf_opt/` 顶层，统一写入 `perf_opt/logs/{profile_stage}/{run_id}.log`。

深挖数据（`msprof_detail`）归档：

- pipe ratios（AIC：mad/scalar/mte1/mte2/mte3/fixpipe；AIV：vec/scalar/mte2/mte3——相对各自 AIC/AIV cycles，不可相加）、`dur_ns`、AIC/AIV cycle counts 记录到：

```text
perf_opt/profiles/{profile_stage}/{kernel_id}/{workload_id}_{candidate_id}/
```

微架构归因（npusim，按需）：

- 对 VF/数据搬运/Cube 操作/同步交接做周期级定位时，按 `tilelang-ascend` skill `references/profiling.md`「Operation cycle measurements」节执行：隔离操作并编译最小 launcher，选择与编译目标匹配的模型（A5/950 用 `-s Ascend950`）、全新输出目录：

```bash
npusim record -s Ascend950 -o ./cycle-profile -g -n 0 \
  'python /absolute/path/to/kernel_launcher.py'
```

- `-n 0` 启用 core 0 日志（不改变 launch grid）；跨核测量用 `-n all` 或所需核范围；确认捕获的 kernel 身份与发射次数、trace 完整后再解读周期区间。
- npusim trace 与解码产物归档到同 `perf_opt/profiles/` 目录；**npusim 周期数据用于归因与 `T.Task(latency=, ii=)`/VF `latency=` 注解反馈，不替代 do_bench(msprof) 时延**；模拟器宿主墙钟不是设备时延。

---

## 4. 校验采集结果

每次采集后检查：

- do_bench(msprof) 返回有效数值（非空、非零、量级合理），采样数足够。
- 被测 callable 确实调用目标 kernel：能通过 kernel 对象来源、构造代码、运行日志（及必要时 `msprof_detail` 记录的 kernel 名）追溯到本次目标 TileLang kernel 与候选版本。
- 采到的不是 cache flush 辅助 kernel 或 Cast / Mul / OnesLike / Random 等框架小算子。
- 检查 profiler 启动与解析日志：空记录、启动失败或缺失目标 kernel，即使外层测试通过也判定 invalid。
- 同 workload 重复运行方差 ≤1%；超阈值先排查设备竞争/输入生成差异，再重测。

无效测量记录 reason 并重新采集；无效数据不能进入诊断。

---

## 5. 记录格式

把逐 workload 采集结果写入 `perf_opt/opt_log.md` 的 `Performance Test Data` 章节。路径中的 ID 须安全转义并保持唯一；baseline 以 `phase=baseline`、`round=0` 追加到 `perf_records.jsonl`（`duration_us` = do_bench(msprof) 毫秒值 ×1000；`profiler_output` 为 do_bench/msprof 输出目录引用，深挖数据归档路径或 `none`）。

最小格式：

```markdown
## Performance Test Data

| kernel_id | workload_id | candidate_id | duration_us | samples | variance_pct | profile_status | profiler_output | command |
|---|---|---|---:|---:|---:|---|---|---|
| {kernel} | {benchmark_case} | baseline | {v} | {n} | {v} | valid | {profiler_output_or_none} | {how_the_runner_was_built_and_invoked} |
```

另列 smoke/skip 排除原因与 smoke 精度结果。最终同一 kernel 的全部 `tune` workload 必须来自同一个 `phase=final` 候选版本，不能拼接各自最快的实验数字。

返回：

```text
PERF_DATA_COLLECTED
```
