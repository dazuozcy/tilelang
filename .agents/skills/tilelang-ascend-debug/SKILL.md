---
name: tilelang-ascend-debug
description: >
  TileLang Ascend kernel 调试技能。用户提及调试 ascend kernel、查看 AscendC
  源码（get_kernel_source）、IR Lower Trace、pass 阶段定位、Pass Diff、
  compile_only、T.print、device_assert、精度异常定位、npusim 周期定位、
  最小复现缩减时必须使用本技能。
---

# TileLang Ascend Kernel 调试

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## 调试症状分流（先分类，再选工具）

| 症状 | 首选工具 | 详见 |
|---|---|---|
| 编译失败（pass 断言 / bisheng 报错） | 错误分类走 tilelang-ascend-error-fixer；`TL_LOWER_TRACE` 定位首个出问题的 pass | 本文件 ②③ + error-fixer |
| 生成的调度/指令与预期不符 | `kernel.get_kernel_source()` + IR Lower Trace 对比「调度改变 vs 指令 lowering 改变」 | ①② |
| 运行结果错误 / 精度异常 | `T.print` / `device_assert` 圈定出错范围 → 最小复现缩减 | ④⑥ |
| 挂死 / 疑似同步缺陷 | 生成源码审计 flag 配对（对照 tilelang-ascend skill references/synchronization.md） | ① + synchronization.md |
| 性能回退 / 周期归因 | tilelang-ascend skill §12 + references/profiling.md（do_bench(msprof)、msprof_detail、npusim） | ⑤ |
| 疑似编译器缺陷 | 最小失败 kernel + 生成源码 / lowering trace 保全纪律 | ⑦ |

## 工具链速览

| # | 工具 | 用途 | 权威文档 |
|---|---|---|---|
| ① | `kernel.get_kernel_source()` | 查看最终生成的 AscendC C++ 源码，定位 codegen 层问题 | tilelang-ascend skill §11 |
| ② | IR Lower Trace（`TL_LOWER_TRACE`） | 逐 pass 前后 IR 对比 + codegen 源码落盘 + 编辑重编译 | `docs/tools/lower_trace.md` |
| ③ | Pass Diff（`TILELANG_PASS_DIFF`，已 superseded）与 compile_only | 选定 pass 链对比；无设备环境下只降级出源码 | `docs/tools/pass_diff.md`、`docs/tools/compile_only.md` |
| ④ | `T.print` / `device_assert` | 运行期缓冲区/表达式打印与断言 | `tilelang/ascend/language/print.py`、`tilelang/ascend/debug.py` |
| ⑤ | npusim | 单操作周期定位（微架构归因） | tilelang-ascend skill references/profiling.md「Operation cycle measurements」 |
| ⑥ | 最小复现缩减 | 把失败压缩到最小可独立复现的 kernel | references/debug-toolchain.md §6 |
| ⑦ | 缺陷保全纪律 | 疑似编译器缺陷时保全证据链 | tilelang-ascend skill 头部规则；references/debug-toolchain.md §7 |

## 各工具要点（详细步骤见 references/debug-toolchain.md）

### ① get_kernel_source：看 AscendC 源码

```python
kernel = tilelang.compile(program, target="ascend", out_idx=-1)
print(kernel.get_kernel_source())
```

用途：确认 VF 模式实际 lowering、`T.copy`/`T.dual_copy` 的指令形态与路由、
调度顺序、flag 插入位置、分配尺寸。源码与预期不符时先于一切猜测。

### ② IR Lower Trace：定位「哪个 pass 改坏了」

```bash
TL_LOWER_TRACE=1 python3 my_script.py        # HTML 报告（默认）
TL_LOWER_TRACE=terminal python3 my_script.py # 终端着色 diff
```

- 每个 pass 带 pipeline phase 标签与全局序号；失败 pass 会显示异常信息与
  崩溃前 IR。
- **「调度改变 vs 指令 lowering 改变」的判定法**：对比两版（改动前后）的
  trace，找到**首个发生表示变化的 pass**——若变化始于 AutoSchedule/
  InsertSync 类 pass，是调度问题；若始于 LowerTileOp/codegen 层，是指令
  lowering 问题。两者修复方向完全不同（tilelang-ascend skill §12.5）。
- codegen 源码落在 `<script_dir>/codegen.cpp`，可编辑后重跑注入（PATCHED，
  注意后端限制，见 `docs/tools/lower_trace.md`）。
- 程序化 API：`tilelang.tools.lower_trace.lower_trace(func, passes, mode=...)`
  可对指定 pass 链做一次性 diff。

### ③ Pass Diff 与 compile_only

- Pass Diff（`TILELANG_PASS_DIFF`）已被 IR Lower Trace 取代，仅向后兼容时
  使用；新调试一律用 ②。
- compile_only（`python -I -m tilelang.tools.compile_only --output_file out.c
  example.py`）：无设备环境下降级出源码，适合 CI 与纯 lowering 检查。

### ④ T.print / device_assert：运行期调试

- `T.print(buffer_or_expr, msg=...)`：支持 UB（shared）、GM（global）、
  fragment（local）缓冲区、PrimExpr 与纯消息；UB/L1 为核内私有，
  无需单线程门控（`tilelang/ascend/language/print.py`）。
- `device_assert(condition, msg=...)`：运行期断言，带栈信息。
- 回归样例：`testing/ascend/language/test_tilelang_ascend_print.py`、
  `test_tilelang_ascend_device_assert.py`。

### ⑤ npusim：单操作周期定位

协议权威在 tilelang-ascend skill references/profiling.md
「Operation cycle measurements」，要点：

```bash
npusim record -s Ascend950 -o ./cycle-profile -g -n 0 \
  'python /absolute/path/to/kernel_launcher.py'
```

- 先把待测操作隔离成最小 `.asc` + 独立 launcher，复现原 kernel 的
  CANN/bisheng 版本、target 与编译旗标。
- 定位 `instr.bin`，用同一 CANN 安装的 `cannsim.core.public.instr_decoder`
  解码；事件配对用 launch/chip/core/subcore/指令 ID/PC，**PC 单独不足以
  区分循环迭代**。
- `is_popped` 语义：已验证的 CANN 9.2 reader 中 `1`=dispatch、`0`=completion；
  换工具版本须重验，禁止凭名字猜语义。
- 周期间隔 = 区间首条指令 start 到末条指令 completion（`max(start+duration)
  - min(start)`）；区界内含 load/store/同步。周期数保留原始值，换算 ns 前先
  确认时钟域。

### ⑥ 最小复现缩减（通用方法）

见 references/debug-toolchain.md §6：二分 pass、二分语句、固定 seed 与形状、
剥离应用层包装。

### ⑦ 疑似编译器缺陷时的保全纪律

tilelang-ascend skill 头部规则：**保全最小失败 kernel 及其生成源码或
lowering trace**。上报前按 error-fixer 的 Issue Report 模板整理；
能力缺口登记到 `.agents/evolution/capability-gaps.md`（对齐 AGENTS.md
Capability Gap Rule 的 layer 枚举与证据规则）。

## References

- [references/debug-toolchain.md](references/debug-toolchain.md)：①–⑦ 的
  操作步骤、判定准则与保全清单。

## Related skills

- tilelang-ascend（编程权威；§11 编译执行、§12 性能测量）
- tilelang-ascend-error-fixer（错误分类树与修复路径）
- tilelang-semantic（前端语义校验权威）
