# Ascend 调试工具链详解

> 配套 [tilelang-ascend-debug SKILL.md](../SKILL.md) 的操作参考。命令与环境
> 变量的权威语义以 `docs/tools/` 各文档与 tilelang-ascend skill
> references/profiling.md 为准；本文给出 Ascend 场景下的使用步骤与判定准则。

## 1. `kernel.get_kernel_source()`：AscendC 源码检查

```python
import tilelang
kernel = tilelang.compile(program, target="ascend", out_idx=-1)
print(kernel.get_kernel_source())   # 最终 AscendC C++ 源码
```

检查点清单：

- VF 模式实际形态：`T.SimtVF`/`T.SimdVF` 区域 lowering 成什么指令序列
  （`asc_vf_call` / CCE MicroAPI 向量指令）。
- `T.copy`/`T.dual_copy` 的指令形态与路由（MTE2/Fixpipe、目标 AIV、
  `sub_blockid`），ND2NZ 转换是否如预期由 pass 插入。
- 调度顺序：任务次序、prologue/epilogue 守卫、flag 的 set/wait 位置。
- 分配尺寸：UB/L1/L0 的物理分配（含布局填充、版本数、编译器保留），
  与容量预算对账。
- 完全手动核须显式传 `pass_configs={tilelang.PassConfigKey.
  TL_ENABLE_AUTO_SCHEDULE: False}`（tilelang-ascend skill §11）。

源码与预期不符是后续一切排查的起点；先看源码，再开 trace。

## 2. IR Lower Trace（`TL_LOWER_TRACE`）

权威文档：`docs/tools/lower_trace.md`。

```bash
TL_LOWER_TRACE=1 python3 my_script.py          # HTML 报告
TL_LOWER_TRACE=terminal python3 my_script.py   # 终端着色 diff
TL_LOWER_TRACE=both python3 my_script.py       # 两者
TL_LOWER_TRACE_DIR=/tmp/opencode/trace python3 my_script.py   # 产物目录
```

产物布局：`<TL_LOWER_TRACE_DIR>/<script_name>/.run_records/run_*/`
下按 phase 分目录、pass 全局序号编号（`00_BindTarget_before.tir` …）；
codegen 产物为 `.cpp`。`<script_dir>/report.html` 符号链接指向最新一次。

用法要点：

1. **首个变化 pass 定位**：在 HTML 报告里按序号找第一个 `changed` 的 pass，
   读它的 before/after diff。失败 pass 会附异常消息与崩溃前 IR。
2. **调度改变 vs 指令 lowering 改变**：准备改动前后两份 trace，对齐 pass
   序号后找首个分歧——
   - 分歧出现在 AutoSchedule / InsertSync / buffer version 规划类 pass：
     **调度问题**（改 `T.Pipelined`/`T.Stage`/`num_stages`/latency 注解/
     `T.Task(latency=, ii=)`）；
   - 分歧出现在布局推理（fractal/NZ）、LowerTileOp、codegen 层：**指令
     lowering 问题**（改搬运 API 参数/布局选择/路径）。
   该判定法与 tilelang-ascend skill §12.5「Attribute the difference」同口径。
3. **编辑重编译**：编辑 `<script_dir>/codegen.cpp`（勿动 `.original`）后
   重跑，日志出现 `PATCHED` 即注入；注意该工作流要求源码编译型执行后端，
   默认二进制后端只记录不重编（见 `docs/tools/lower_trace.md` 的后端限制节）。
4. **程序化单链 diff**：
   `tilelang.tools.lower_trace.lower_trace(func, [pass1, pass2], mode="both")`，
   适合对单个可疑 pass 做假设检验。

注意：trace 有开销，仅调试时开启；文本 diff 变化本身不证明语义/性能变化。

## 3. Pass Diff 与 compile_only

- Pass Diff（`TILELANG_PASS_DIFF`，`docs/tools/pass_diff.md`）已被 IR Lower
  Trace 取代（superseded），仅存量脚本兼容时使用。
- compile_only（`docs/tools/compile_only.md`）：

```bash
python -I -m tilelang.tools.compile_only --output_file out.c example.py
```

  无 NPU/驱动环境也能降级出 kernel 源码；输入文件顶层不得启动 kernel 或
  分配张量；只编译模块内第一个 `@tilelang.jit` 函数或 PrimFunc。
  程序化：`from tilelang.tools.compile_only import compile_kernel_source`。

## 4. `T.print` / `device_assert`：运行期调试

- `T.print(obj, msg="")`：`obj` 为 Buffer（local/shared/global scope）、
  PrimExpr 或 None（纯消息）。实现见 `tilelang/ascend/language/print.py`：
  Ascend 上 kernel body 每 AI 核执行一次、UB/L1 为核内私有，缓冲区打印
  无需单线程门控。
- `device_assert(condition, msg="")`：运行期断言，可带栈信息
  （`tilelang/ascend/debug.py`，与 CUDA 侧同一 `tl.device_assert` builtins）。
- 用法套路：先在搬运边界（copy 进 UB 后 / 写回 GM 前）打印缓冲区切片，
  圈定「数据进 VF 前就错」还是「VF 算错」还是「写回错」；再用断言钉住
  循环变量/索引假设。
- 回归样例：`testing/ascend/language/test_tilelang_ascend_print.py`、
  `test_tilelang_ascend_device_assert.py`。

## 5. npusim 单操作周期定位

协议权威：tilelang-ascend skill references/profiling.md
「Operation cycle measurements」节。步骤摘要：

1. **选区**：声明测的是单操作 / 依赖链 / 独立重复流 / 整核——不同边界回答
   不同问题；保留区内的 shape、dtype、布局、线程维度、掩码、循环次数、
   缓冲生命周期与依赖。
2. **隔离**：把目标操作连同 includes 与依赖 helper 拷进最小 `.asc`；
   VF 用 `__global__ __vector__` 入口并保留原 SIMD 调用或 SIMT
   `asc_vf_call<vf>(cce::dim3(...), ...)` 与捕获参数；对齐分配原始内存空间
   的存储并初始化代表性输入。
3. **编译**：复用原 kernel 的 CANN 安装、target（`--npu-arch=dav-3510`）、
   优化旗标、TileLang 模板 include 路径与启动 ABI；必要时加载所用 CANN
   安装自带的 set_env.sh（CANN 官方环境脚本）。
4. **录制**：
   ```bash
   npusim record -s Ascend950 -o ./cycle-profile -g -n 0 \
     'python /absolute/path/to/kernel_launcher.py'
   ```
   `-n 0` 开 core 0 日志（不选 kernel、不改 grid）；跨核测量用 `-n all` 或
   指定核范围——单核完成不等于整核完成。Python launcher 在 CPU 备好输入与
   参考结果再上设备，捕获前禁止 NPU 随机生成/量化/warmup kernel。
5. **解码与配对**：定位 `instr.bin`（可能在 `npusim_*/record` 下），用同一
   CANN 安装的 `cannsim.core.public.instr_decoder` 解码或用其指令时间线
   报告。事件配对依据 launch/chip/core/subcore/指令 ID/PC；**PC 单独不足以
   区分循环迭代**。`is_popped`：已验证 CANN 9.2 reader 中 `1`=dispatch、
   `0`=completion——禁止按名字猜，换版本须重验。
6. **读数**：区间 = 首条指令 start → 末条指令 completion
   （`max(start+duration) - min(start)`）；VF 用 RVEC/VECTOR 事件，copy/
   Cube/组合区用对应 pipe 事件与必要 handoff。区分 dispatch 起点与实际执行
   起点；保留原始 cycle，换算 ns 前确认时钟域与频率。
7. **落地**：测得成本经 `T.Task(latency=, ii=)`（整任务体）或 VF 的
   `latency=` 注解回馈调度模型（tilelang-ascend skill §8.3、
   references/profiling.md「Override automatic task estimates」），改后必须
   重编译、验精度、重测整核。

## 6. 最小复现缩减方法（通用）

目标：把失败压缩到「最小可独立复现的 kernel + 一段日志」。

1. **固定变量**：seed、shape（优先缩到最小触发尺寸）、dtype、pass 配置、
   tile 参数全部字面量化；剥离 CLI/配置层，直排成单文件脚本。
2. **二分 pass**：`TL_LOWER_TRACE=terminal` 下观察失败发生在哪个 phase；
   用程序化 `lower_trace(func, [可疑pass])` 单独重放验证。
3. **二分语句**：kernel 内逐块注释/删除（VF 体、copy、gemm、epilogue），
   保持每步仍可编译；同步类问题注意"删掉后不再死锁"不等于定位了根因。
4. **降维对照**：关 AutoSchedule（完全手动）对照、关 pipeline（`num_stages`
   降 1）、单核化（`T.Kernel(1)`）、去掉 dual_copy 改普通 copy——每次只改
   一个变量。
5. **应用层剥离**：从测试框架/算子工程里抽出 PrimFunc 直接
   `tilelang.compile(..., target="ascend")`。
6. **保留物**：缩减过程中每一步的脚本、报错全文、trace 产物归档，防止
   "修好了但不知道为什么"。

## 7. 疑似编译器缺陷：保全纪律

tilelang-ascend skill 头部规则：**For a suspected compiler defect, preserve
the smallest failing kernel together with its generated source or lowering
trace.**

上报/登记前的保全清单：

- [ ] 最小失败 kernel（单文件、无外部依赖、字面量参数）
- [ ] `kernel.get_kernel_source()` 全文（或 codegen.cpp 产物）
- [ ] `TL_LOWER_TRACE=1` 的报告目录（含失败 pass 的 before IR）
- [ ] 环境戳：tilelang commit、CANN/bisheng 版本、`--npu-arch=dav-3510`、
      设备型号与所选 device
- [ ] 复现命令（逐字可粘贴）与预期 vs 实际（精度问题附 golden 对照与容差）
- [ ] 若涉及容量/启动失败：生成的分配尺寸、buffer versions、VF 模式、
      compile_flags（tilelang-ascend skill §11 的三类失败要分开记录）

去向：上游 issue 走 tilelang-github-operations skill 的模板（需用户批准）；
能力缺口按 AGENTS.md Capability Gap Rule 登记
`.agents/evolution/capability-gaps.md`（layer 枚举：Frontend API（ascend
方言）/ Ascend pass / codegen / bisheng 编译器 / ACL runtime）。
