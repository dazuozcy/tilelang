---
name: tilelang-op-develop
description: "根据冻结的 DESIGN.md 生成算子实现（{op}.py：kernel + golden），执行测试并返回四出口判定。触发：实现算子、生成 kernel、算子开发、跑精度。"
---

# TileLang-Ascend 算子开发与验证

迁移来源与阶段衔接遵循 [migration-source.md](../_shared/standards/migration-source.md)。harness 开发消费 `source_context`，golden 对应当前内部函数；保持目标 wrapper 的符号、参数及返回结构。

> 执行效率与恢复遵循 [execution-efficiency.md](../_shared/standards/execution-efficiency.md)。首次加载后按指纹复用上下文；仅影响执行方式，不跳过最终精度、独立检视或完整 workload 验收。

## 1. 目标

根据 Stage 1 冻结的 `DESIGN.md` 与 Stage 2 通过的 `REVIEW.md`，生成算子实现文件 `{op}.py`（含 `@tilelang.jit(target="ascend")` kernel + 内嵌 PyTorch golden + main 入口），执行测试，并返回四出口判定（[PRECISION_PASS] / [PRECISION_FAIL] / [DESIGN_ERROR] / RUNTIME_FAIL）供 conductor 路由。

> **环境前提**：本 skill 运行在已具备 Ascend A5/950 设备的环境中，`tilelang` 与 `torch_npu` 可正常导入（`torch.npu.is_available()` 为 True），CANN 9.x 与 bisheng（`--npu-arch=dav-3510`，`BISHENG_HOME`）可用。kernel 编译与执行在 NPU 上真实进行，精度校验为真实结果。

---

## 2. 输入

| 字段 | 说明 |
|------|------|
| `design_md_path` | 冻结的 `DESIGN.md`（含 L0 测试计划） |
| `review_md_path` | Stage 2 通过的 `REVIEW.md`（设计已检视通过） |
| `mode` | 调度模式，由 conductor 传入；枚举唯一出处 `_shared/standards/signal-registry.md` §2 |
| `attempt_index` | 当前 Stage 3 attempt 序号 |
| `last_failure_summary` | 重试时传入的失败信息（stderr 摘要 / 精度失败详情） |
| `design_revision_count` | 设计修订次数（用于回退后清零判断） |

---

## 3. 工作流程

### Phase 1：读取设计
1. Read `DESIGN.md` 全文，提取：算子名、I/O 规格、编程模式（`simt` / `simd` / `mixed` + `auto_schedule`，枚举见 `_shared/standards/signal-registry.md` §3 与 `_shared/standards/backend-glossary.md` §2）、API 映射、Tiling、内存层级、同步策略、L0 测试计划、精度标准。
2. Read `REVIEW.md`，确认检视已通过（如有 warn 项记录但不阻塞）。
3. Read `tilelang-op-optimize` skill 的 [pattern-library/INDEX.md](../tilelang-op-optimize/references/pattern-library/INDEX.md)（条目索引）并按命中精读主题文件（[constants.md](../tilelang-op-optimize/references/pattern-library/constants.md) A5 容量/吞吐常数、已验证模式、编译器/运行时陷阱，**注意版本戳**——重编译后旧结论待重验，`kb_stale_check.py` 检测）——实现与调试前必读；调试中命中的条目在返回的 `skills_consulted` 中注明引用。

### Phase 2：生成 kernel
1. 按 DESIGN.md §3 API 映射 + §6 循环结构生成 kernel，规范入口 `@tilelang.jit(target="ascend")`（`tilelang.compile(program, target="ascend", out_idx=-1)` 亦是合法形态）。
2. **方言与编程模式**：
   - 一律 `import tilelang.ascend.language as T`（方言 shadowing common surface，禁用 `import tilelang.language as T`）。
   - 按 DESIGN.md `programming_mode` 实现：`simt` = `T.SimtVF(threads=N)` + `T.Parallel` 循环（默认推荐）；`simd` = `T.SimdVF()` + `T.simd.*` 2048-bit intrinsic（仅性能热点）；`mixed` = `T.MixedKernel(sids=1|2)` + `T.gemm` + SimdVF 后处理（GEMM 主导）。
   - Cube 路径：`T.copy` GM↔L1（ND2NZ fractal 转换由 InsertNd2Nz pass 自动插入，**禁止手写 NZ 分形搬运**）→ `T.gemm(A_l1, B_l1, C_l0c, transpose_B=True)`（L1-input 路径要求 NT，权重 GM 布局 `[N, K]`）→ `T.dual_copy`/`T.copy` L0C→UB。
   - 内存层级：`T.alloc_shared`=UB / `T.alloc_l1`=CBuf / `T.alloc_l0a/l0b/l0c`=Cube 缓冲（容量见 pattern-library constants.md CONST-capacity-a5，注意 SimtVF 下 UB 上限 216KB）。
   - 核启动一维：`T.Kernel(N)`，**不带 `threads=`**（TypeError）；2-D 分解自行计算 `row = idx // N_TILES; col = idx % N_TILES`。
   - 同步默认依赖 AutoSchedule；仅当 DESIGN.md `auto_schedule: no` 时写手动 `T.ascend_set_flag`/`T.ascend_wait_flag`（须配对并注释），且编译需 `pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False}`。
3. 遵循项目根 AGENTS.md："不要凭记忆猜 API"、"从示例入手"——先查 `tilelang-ascend-opdev` skill 的算子族导航表（elementwise/gemm/attention/reduction-norm → 推荐模式 → 参考实现 → 布局/搬运套路），据此 Glob `examples/ascend/` 同类实现参考；API 语义权威 `.agents/skills/tilelang-ascend/SKILL.md`（§2 线程层级 / §4 内存 / §5 GEMM / §6 搬运 / §7 同步 / §10 陷阱）。

### Phase 3：生成 golden
1. 按 DESIGN.md §8.1 生成 PyTorch 参考实现 `golden_{op}(...)`。
2. Golden 用 torch 在 **NPU 上**计算（输入张量 `device="npu"`），与 kernel 同设备对比；对比用 `torch.testing.assert_close`。
3. 容差按 dtype 分档：fp8→1e-1、bf16/fp16→1e-2、fp32→1e-3（累加密集型 5e-3）；DESIGN.md 另有精度标准时以其为准。

### Phase 4：执行测试
1. 跑 L0：`python {op}.py --level L0`。
2. L0 通过后扩展 L1/L2/Boundary，经 execution_receipt runner 跑全量 `--level all` 并返回收据路径（execution-efficiency.md §3）；L0/L1 失败必须体现在测试入口的退出码中。
3. 收集结果：max_diff、失败用例 shape、层级。

### Phase 5：四出口判定与返回

| 条件 | 返回标记 |
|------|----------|
| 输出与 golden 函数输出对比精度正常 | `[PRECISION_PASS]` |
| 输出与 golden 函数输出对比精度未过 | `[PRECISION_FAIL]` |
| 发现设计层错误（API 不可用、L0C/UB 溢出、内存层级冲突等实现层无法修复） | `[DESIGN_ERROR]` + 原因 |
| 无标记且 exit code ≠ 0 | 运行失败（conductor 按 retry_impl 路由） |

### Phase 6：任务复盘（Retrospective，自进化钩子）⭐

返回 `[PRECISION_PASS]` 或 `[DESIGN_ERROR]` 前（`[PRECISION_FAIL]` / 运行失败**不写**——失败摘要已由 conductor 路由），向 `examples/{project}/{op}/RETROSPECTIVE.md` **追加**复盘章节（先 Read 既有内容，整文件写回，只追加不覆盖历史章节）：

- 章节模板与字段规范（canonical）：`tilelang-skill-evolution` skill 的 [references/retrospective-schema.md](../tilelang-skill-evolution/references/retrospective-schema.md)。两个标准表（Skill Flow Issues / Value Point Proposals）+ Transferable Lessons 小节。
- **PASS 复盘覆盖整个 attempt 链**（含中间失败中的发现）：
  - API 实际行为实证（D 类，须带三件套）：如某 API 的隐藏限制、对齐触发条件、静默 dtype 转换、合法形态边界；
  - 有效的调试手法（P 类）：如精度问题的定位顺序、`kernel.get_kernel_source()`（AscendC 源码）/ IR Lower Trace / Pass Diff 关键点；
  - `[DESIGN_ERROR]` 的根因与设计判断教训（P/R 类）：哪类设计判断错了、正确依据是什么。
- **探针转正（ED-C，D 类发现的 repro 责任前移）**：调试中发现 API 陷阱/行为实证时，探针不再只写 `/tmp`——有价值的最小复现**转正**为 `examples/{project}/{op}/repro/<TRAP-or-CONST-id>.py`（自包含 + 自描述头 + 断言，ED-B 规范见 pattern-library [repro/README.md](../tilelang-op-optimize/references/pattern-library/repro/README.md)），RETROSPECTIVE 的 Value Point Proposals 以该路径为 `repro` 字段（不再引用 `/tmp` session-local 路径）；evolver 终态蒸馏时机械拷入知识域 repro/ 并登记。本 skill 有 NPU 环境，转正前**实际跑一遍**确认断言成立。
- **Transferable Lessons**（迁移多函数任务必写，其他场景可 none）：写给本迁移任务**后续函数**实施者的跨函数教训，一条一行，自包含（不依赖本函数上下文可理解）。
- 质量红线：无则如实写 `none`；单任务偶然现象不得写成通用规则；不得硬凑。

---

## 4. `{op}.py` 结构规范

生成的文件必须在模块头 docstring 中记录最终通过验证的编程模式，供 Stage 4/5 与
report 读取：`programming_mode: simt | simd | mixed` 与 `auto_schedule: yes | no`
（枚举出处 `_shared/standards/signal-registry.md` §3）。该值是 Stage 3 实际通过
验证的实现模式；实现过程中切换模式时，以最终实现为准。不要把 Stage 3 调度字段
`mode`（调度三 mode 枚举，见 signal-registry.md §2）当作编程模式。`auto_schedule:
no`（完全手动核）时，文件中须保留
`pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False}` 的编译调用。

生成的文件必须包含以下组成部分（顺序）：
注意：*.py 中的注释只能使用英文

```python
# 1. Copyright (c) Huawei Technologies Co., Ltd. 2026.
# 2. imports (tilelang + tilelang.ascend.language + torch/torch_npu)
# 2. golden_{op}(...) function
# 3. @tilelang.jit(target="ascend") kernel
# 4. precision comparing func：run_case()
# 5. main()
```

完整可运行模板见 [templates/op_template.py](templates/op_template.py).

### main 块结构

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"])
    args, _ = parser.parse_known_args()

    if args.level == "L0":
        run_L0()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
```

每个 `run_LX()` 内部在 NPU 上创建张量（`device="npu"`）、调用 kernel 执行、与 golden 对比精度（`torch.testing.assert_close`，容差按 §Phase 3 分档）。详见模板。

---

## 5. 失败处理

错误按 ascend 编译链四类定位（详见 `tilelang-ascend-error-fixer` skill；调试设施见 `tilelang-ascend` skill §11 与 `docs/tools/lower_trace.md`、`docs/tools/pass_diff.md`）：

| 失败类型 | 识别 | 处理 |
|---------|------|------|
| ① 方言前端（tilelang-semantic 语义校验） | stderr 含语义校验报错（buffer 越界/未初始化、数据竞争、TileOp 契约、循环嵌套等） | 实现层语义违规 → 返回运行失败 + stderr 摘要，conductor 走 retry_impl；方言表达能力缺口 → `[DESIGN_ERROR]` + 原因 |
| ② Ascend pass 断言 | stderr 含布局推理 / VFChecker / AutoSchedule / 容量检查断言 | 按 pass 报错修正实现；属设计层（L0C/UB 溢出、布局组合不可实现）→ `[DESIGN_ERROR]` + 原因 |
| ③ bisheng 编译报错 | stderr 含 bisheng/`--npu-arch=dav-3510` 编译错误 | 用 `kernel.get_kernel_source()` 对照 AscendC 源码定位；实现层可修 → 运行失败 + 摘要；codegen 缺口 → `[DESIGN_ERROR]` + 原因 |
| ④ ACL 运行时/精度 | `aclrtLaunchKernelWithHostArgs` 等运行时报错、launch 失败、段错误 / `assert_close` 失败 | 资源不足改 tiling；同步缺陷复核 AutoSchedule/手动 flag 配对；精度不达标 → `[PRECISION_FAIL]` + max_diff/失败 shape |
| API 不存在 | `AttributeError` / 设计用 API 无导出 | 返回 `[DESIGN_ERROR]` + 原因 |
| 内存层级越级 | stderr 提示 GM/L1/UB/L0 访问违规 | 返回 `[DESIGN_ERROR]` + 原因 |
| 环境问题 | `ImportError` 指向 tilelang/torch_npu 未安装，或 CANN 9.x / `BISHENG_HOME` 未配置 | 返回运行失败，提示检查环境 |

---

## 6. 备份规则

`precision_fix` 模式每次修改 `{op}.py` 前，必须先备份：
```bash
cp {op}.py history_version/{op}_impl_s3_attempt{N}.py
```
（`{N}` = 当前 attempt_index）

---

## 7. 完成报告

返回结构化摘要：

```markdown
## Stage Result
- stage: 3
- mode: first_impl / retry_impl / precision_fix
- programming_mode: simt / simd / mixed（+ auto_schedule: yes / no）
- project: {project}
- operator: {op}
- output: examples/{project}/{op}/{op}.py
- verdict: [PRECISION_PASS] / [PRECISION_FAIL] / [DESIGN_ERROR] / RUNTIME_FAIL
- test_results:
  - L0: pass / fail (N cases)
  - L1: pass / fail (N cases)
  - L2: pass / warn (N cases, 不阻塞)
  - Boundary: pass / warn (N cases, 不阻塞)
- max_diff: <精度数值>
- design_error_summary: <仅 DESIGN_ERROR 时填>
- skills_consulted: <引用的 skill 路径>
- summary: <一句话>
- issues: <若无则 none>
```
