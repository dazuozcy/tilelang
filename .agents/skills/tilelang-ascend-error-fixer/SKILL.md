---
name: tilelang-ascend-error-fixer
description: >
  TileLang Ascend 错误诊断与修复技能。用户提及编译失败、运行错误、pass 断言、
  布局推理/VFChecker/AutoSchedule 报错、bisheng 编译报错（dav-3510）、
  ACL 运行时错误、结果错误、精度失败、sync 死锁、load/store 维度不一致、
  容量超限时必须使用本技能。
---

# TileLang Ascend Error Fixer

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## 诊断总流程

1. **确认环境与入口**：`@tilelang.jit(target="ascend")`（或
   `tilelang.compile(program, target="ascend")`）；`torch.npu.is_available()`
   为真；CANN 与 `BISHENG_HOME` 环境就绪。
2. **最小复现**：缩到单文件 kernel（方法见 tilelang-ascend-debug skill ⑥）。
3. **分类**：按下述四类错误树定位层——方言前端 / Ascend pass / bisheng
   编译 / ACL 运行时与精度。
4. **取证**：报错全文、`kernel.get_kernel_source()`、`TL_LOWER_TRACE` 报告。
5. **修复并验证**：最小改动 → 精度回归（NPU torch golden + `assert_close`
   分档容差）→ 相关 `testing/ascend/` 套件。

## 错误分类树

| 层 | 识别特征 | 权威依据 |
|---|---|---|
| ① 方言前端 | Python 异常 / `[TileLang Semantic Check]` 前缀诊断，发生在 `tilelang.compile` 早期 | tilelang-semantic skill |
| ② Ascend pass | pass 名 + 断言消息（布局推理 / VF 校验 / AutoSchedule / 容量检查），发生在 lowering 中段 | tilelang-ascend skill + `TL_LOWER_TRACE` |
| ③ bisheng 编译 | bisheng 编译器报错（`--npu-arch=dav-3510`），AscendC 源码级（语法/类型/资源），发生在设备编译段 | `kernel.get_kernel_source()` |
| ④ ACL 运行时/精度 | `aclrt*` 错误、段错误、挂死，或跑通但数值超容差 | tilelang-ascend skill §11 + debug skill ④⑤ |

---

## ① 方言前端（语义校验）

**识别特征**：报错早于任何 pass 输出；消息带可操作的改写建议；常见于循环
嵌套、buffer 边界/初始化/所有权、数据竞争、TileOp shape/dtype/scope 契约、
reducer 生命周期、kernel launch 约束。

**定位工具**：报错消息本身 + tilelang-semantic skill 及其 references
（`loop-rules.md` / `memory-concurrency.md` / `operations-layout.md`）；
边界不明时用 `TL_LOWER_TRACE=terminal` 确认失败发生在 `PreLowerSemanticCheck`
阶段。

**常见根因**：

- `T.Kernel` 带了 `threads=`（TypeError）或用了多维 grid（仅支持一维）。
- reducer 误用：句柄直接读/赋值/`T.clear`/`T.fill`/原地 finalize；
  init 与 finalize 不在同一作用域。
- 循环嵌套/所有权违反：owner 外普通访问、嵌套 owner、pipeline 契约。
- `T.Parallel` 与 `T.vectorized` 混淆（两者是不同执行层）。
- buffer 越界/未初始化/别名竞争被 `legalize_safe_memory_access` 拦截。

**判别原则**（tilelang-semantic 核心）：区分"对所有 lowering 都非法"与
"当前 lowering 不支持"——后者不是语义错误，走能力缺口登记。

**上报/登记**：确属前端能力缺失 → `.agents/evolution/capability-gaps.md`
（layer: Frontend API（ascend 方言））。

---

## ② Ascend pass（断言类）

**识别特征**：报错含 pass 名或调度/布局断言；发生在 lowering 中段。涉及
`AscendLayoutInference`（fractal/NZ 布局推理）、`NormalizeAscendFractalStorage`、
`InsertNd2Nz`（GM→L1 分形转换插入）、`InsertOOBPadding`（越界填充）、
VFChecker（VF 模式校验）、AutoSchedule（`AssignCore`/`EstimateLatency`/
`InsertSync`）、容量检查。

**定位工具**：`TL_LOWER_TRACE=1` 找首个失败/异常 pass（tilelang-ascend-debug
skill ②）；`kernel.get_kernel_source()` 看已生成部分（①工具）；程序化
`lower_trace(func, [可疑pass])` 单独重放。

**常见根因**：

- **布局推理失败**：`T.copy` 源/目标 region 与推理出的 fractal/NZ 布局不
  匹配；高维 DMA / 非紧凑 L0 region / MX scale 存储违反指令契约。
- **InsertOOBPadding 相关**：尾块/非 32B 行触发填充路径，但目标 UB 分配或
  行 pitch 未按填充宽度预留（tilelang-ascend skill §6.1 pad_value/data_select
  的两模式与互斥约束）。
- **InsertNd2Nz 相关**：GM→L1 搬运的 region 形状/步长不被分形转换支持；
  `transpose=True` 在 GM→L1 选 dn2nz 并反转逻辑轴（路径相关，非通用转置）。
- **VFChecker**：`T.SimdVF` 内不支持的 pattern；`T.Parallel` in SimdVF 的
  lowering 拒绝（改显式 `T.simd.*` 表达，§8.2）。
- **AutoSchedule/容量**：UB 248KB（SimtVF 216KB）/ L1 512KB / L0 超限——
  注意分配尺寸 ≠ 逻辑 shape：要算布局填充、buffer versions、编译器保留
  （§4.3）。排查时可临时 `T.annotate_unlimit_memory` 解除调度容量约束以
  确认是容量问题（不扩大物理内存，启动前须核实物理分配）。
- **`T.PerCoreTask` 契约**：候选守卫不满足"每核恰好执行一次"的互斥完备
  要求（§2.5）。

**上报/登记**：pass 层缺失/误报 → capability-gaps.md（layer: Ascend pass），
附最小 kernel + trace 证据。

---

## ③ bisheng 编译报错

**识别特征**：设备编译段报错，`--npu-arch=dav-3510`（A5/950）工具链；
错误指向 AscendC 源码行（语法/类型/内联 intrinsic 参数/资源超限）。

**定位工具**：`kernel.get_kernel_source()` 拿到同一段 AscendC 源码对照报错
行号；compile_only 在无设备环境复现纯编译问题（tilelang-ascend-debug skill ③）。

**常见根因**：

- 生成的 intrinsic 调用参数组合不被 SDK 头支持（lane 宽度/dtype/舍入）——
  先对照 `tilelang/ascend/language/simd.py` 与安装 SDK 头文件。
- L0/UB 资源在 bisheng 侧判定超限（与 ② 的编译器容量拒绝是两类失败，
  分开记录，tilelang-ascend skill §11）。
- tilelang 模板 include 路径 / CANN 版本不匹配。

**上报/登记**：bisheng 工具链问题 → capability-gaps.md（layer: bisheng
编译器），附 `.asc` 级最小片段与 bisheng 版本。

---

## ④ ACL 运行时 / 精度

**识别特征**：kernel 编译通过，运行期 `aclrtBinaryLoadFromData` /
`aclrtLaunchKernelWithHostArgs`（ascendcl stub 惰性 dlopen）报错、段错误、
挂死；或无报错但数值超容差。

**定位工具**：`T.print`/`device_assert` 圈定出错数据段（debug skill ④）；
挂死用生成源码审计 flag 配对（tilelang-ascend skill references/
synchronization.md「Audit flags through control flow」五条清单）；
周期/死锁归因用 npusim（debug skill ⑤）。

**常见根因**：

- **同步类**：手动 `T.ascend_set_flag/wait_flag` 不配对、flag ID 复用错、
  零_trip 循环内 set 被跳过而外部 drain wait 仍执行、跨核 fan-out/fan-in
  与 AIV 子核归属不符。
- **dual_copy/归属类**：L0C→UB M-split 目标只有半行；两 AIV 分摊同一 tile
  的 GM 区域划分不清；`sid` 守卫与 `sub_blockid` 不一致导致 copy 被消除。
- **精度类**：golden 未在 NPU 上算或 dtype 混用；累加 dtype 契约（fp16/bf16
  GEMM 累加 fp32）；HF32 舍入模式影响；padded lane 未初始化即被消费。
- **环境类**：CANN/驱动版本、device 选择与进程可见性、竞争负载。

**精度验证口径**（AGENTS.md）：golden 用 torch 在 NPU（`device="npu"`）计算，
`torch.testing.assert_close`，容差分档 fp8→1e-1、bf16/fp16→1e-2、fp32→1e-3
（累加重 5e-3）。

**上报/登记**：runtime 层缺失 → capability-gaps.md（layer: ACL runtime）；
精度问题若定位到编译器缺陷，按 debug skill ⑦ 保全纪律归档最小失败证据。

---

## 通用 Issue Report 模板

```markdown
## TileLang Ascend Issue Report

### Summary
- Symptom:        <报错原文/错误数值现象，一句话>
- Repro script:   <单文件最小复现路径或粘贴>
- Impact:         <阻塞的算子/场景与规避代价>

### Environment
- tilelang commit: <hash>；CANN/bisheng 版本；arch: dav-3510（A5/950）
- device: <型号与所选 device>；torch/torch_npu 版本

### Classification
- Layer:          frontend / Ascend pass / bisheng / ACL runtime（四选一）
- First failing pass（若有）: <TL_LOWER_TRACE 定位的 pass 名与序号>

### Evidence
- kernel source:  <get_kernel_source 或 codegen.cpp 关键段>
- lowering trace: <报告目录路径>
- Expected vs actual: <精度问题附 golden 与容差判据>

### Root Cause
- <定位到的具体机制，注明证据来源>

### Fix
- Minimal change: <最小补丁>
- Why this fixes it: <因果链>

### Verification
- Repro after fix: <结果>
- Numerical check: <assert_close 分档容差结果>
- Regression:     <跑过的 testing/ascend/ 套件；回归风险说明>
```

## 案例区

**A5 冷启动声明**：本 skill 随 tilelang-fork（ascend/A5，dav-3510）适配
重建，旧后端案例已随数据层清零，不适用于本工具链。典型案例（报错原文 →
根因 → 修复 → 回归）由 conductor 自进化机制在任务终态蒸馏积累
（tilelang-skill-evolution skill），并按 backend-glossary 词汇登记，禁止
回填旧后端案例。

## Related skills

- tilelang-ascend-debug（定位工具链：trace/源码/npusim/保全纪律）
- tilelang-semantic（前端语义校验权威）
- tilelang-ascend（编程权威，§4.3 容量 / §6 搬运 / §11 编译执行）
