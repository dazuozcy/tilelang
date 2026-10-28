# 技术约束清单（必须遵守）

本项目使用 TileLang Ascend 方言（`import tilelang.ascend.language as T`；tilelang-fork ascend 后端，A5/950，bisheng `--npu-arch=dav-3510`），与 GPU 版 TileLang 在 Kernel 维度、VF 编程模式、GEMM 约定、内存层级 API 上有显著差异。
**外部参考实现不可直接使用，必须转换为 Ascend 兼容方案。**

> 硬件常数（容量 / 吞吐 / 带宽）引用 `tilelang-op-optimize` skill pattern-library [constants.md](../../tilelang-op-optimize/references/pattern-library/constants.md) 条目 ID——当前全部 `status: assumption`（A5 冷启动），设计期容量按 **90% 折扣**预算并在 DESIGN.md 披露。API 语义与约束权威：`.agents/skills/tilelang-ascend/SKILL.md`（§2 线程层级 / §4 内存 / §5 GEMM / §6 搬运 / §7 同步）。

## 目录

- [技术约束清单（必须遵守）](#技术约束清单必须遵守)
  - [目录](#目录)
  - [1. 本项目已知限制](#1-本项目已知限制)
  - [2. 强制检测规则](#2-强制检测规则)
  - [3. 警告输出格式](#3-警告输出格式)
  - [4. Host 侧 输入 操作约束](#4-host-侧-输入-操作约束)

---

## 1. 本项目已知限制

| 约束 | 说明 | 影响 | 替代方案 |
|------|------|------|----------|
| **一维 Kernel，无 threads 参数** | `T.Kernel(N)` 只接受一维 block 数；`threads=` 直接 TypeError。线程域在 `T.SimtVF(threads=N)` 内声明（tilelang-ascend skill §2.1/§2.2） | 多维并行设计无法直接表达 | 多维乘积展开为一维 block 数，核内自行计算映射（`row = tile_idx // N_TILES; col = tile_idx % N_TILES`，skill §10 陷阱 3） |
| **部分 GPU API 不可用** | CUDA 专用概念（TMA / cp.async / warp shuffle / cluster 等）在 Ascend 方言无对应 API | 直接移植 GPU 代码失败 | 查 `examples/ascend/` + `testing/ascend/` + tilelang-ascend skill 确认 Ascend API；能力映射表见 [migration-analysis.md](migration-analysis.md) §5.3 |
| **GEMM C0 分形对齐** | M、N 必须是 16 的倍数；`K % TILE_K_SUB == 0` 且 `TILE_K_SUB % C0 == 0`（C0 = 32B/元素：bf16/fp16=16、fp32=8、int8=32、fp4=64）；FRAC_N=16（`CONST-gemm-dim-constraints`） | 违反触发 static_assert 编译失败 | 设计文档 §4/§5 必须明确处理策略：GM→UB `pad_value=`/`data_select=` 右 padding（32B 行边界）或 host 侧 padding+crop |
| **量化 GEMM（blockscaled/MX）约束** | K 必须 64 整除（一个 SF pair 覆盖 64 个 K 元素）；仅 bf16/fp8e4m3fn/fp4；GEMM 输入 dtype：fp16/bf16/fp32/fp8e4m3fn/fp8e5m2/fp4e2m1fn/int8 | 不满足时量化路径不可用 | 调整 K 分块或换 dtype 路径 |
| **L1-input GEMM 仅支持 NT** | `T.gemm(A_l1, B_l1, C_l0c, transpose_B=True)`（`trans_A=False`）；权重 GM 布局 **[N, K]**（skill §5.1/§5.2） | 其他转置组合在 L1 路径不可用 | 保持 [N, K] 权重布局或显式变换；L0-input 路径有独立布局处理，勿把 L1 限制推广 |
| **INT8 MMAD 不支持** | A5 上 INT8 矩乘吞吐为 0（`CONST-cube-throughput-model`） | int8 GEMM 无 Cube 路径 | 另寻算法路线；按 AGENTS.md Capability Gap Rule 登记 `.agents/evolution/capability-gaps.md` |
| **HF32 模式精度权衡** | `T.set_hf32_mode("nearest_even"/"nearest_zero")` 用精度换 fp32 GEMM 吞吐（skill §5.4） | fp32 结果精度下降 | 实测算子级影响并按 §8.2 精度标准验证容差 |
| **L0C 容量上限** | A5 L0C = 256KB（`CONST-capacity-a5`，assumption 按 90% 折扣） | `block_M × block_N × sizeof(accum)` 超限导致编译/launch 失败 | 设计 block 时满足容量预算（计入 buffer versions 与编译器保留，skill §4.3） |
| **UB/L1 容量上限** | UB = 248KB（含 `T.SimtVF` 时 216KB——SIMT 线程上下文占 32KB）；L1 = 512KB；L0A = L0B = 64KB（`CONST-capacity-a5`） | 超限编译失败或调度劣化 | §4.5 UB 预算表逐 buffer 核对；AutoSchedule multibuffering 版本数计入容量 |
| **核数无编译期静态常数** | AIC/AIV 总数须运行时实查：`torch.npu.get_device_properties()` 的 `cube_core_num` / `vector_core_num`；examples/ascend 惯例初值 32 blocks（assumption 待标定，`CONST-arch-a5`）；AIC:AIV 物理编组 1:2 | 超发 block 被串行调度引入额外开销；block 数非核数整数倍造成负载不均 | 分核三要素（grid 划分含核数依据 / AIC-AIV 分工 / 同步策略）见 `_shared/standards/core-split-strategy.md` §1 |
| **静态循环边界** | 核内串行循环边界必须为静态值（`CONST-arch-a5`；gate 约束） | 动态边界破坏调度与同步 | 静态化（编译期已知）或 padding 到静态边界 |
| **OOB padding 模式约束** | `pad_value=` 与 `data_select=True` 互斥（同传 ValueError）；仅 GM→UB 路径；dtype 限 8/16/32-bit；`data_select` 复用 pad 寄存器须先 `T.ascend_set_copy_pad_value` 设值（skill §6.1） | 误用报错或 pad 值错误 | 按 skill §6.1 两种模式择一；数值覆盖见 `testing/ascend/language/test_tilelang_ascend_dma_copy.py` |
| **flag id 预算** | 核内 flag 8 个、跨核 flag 16 个（`CONST-arch-a5`；溢出经 RewriteFlagToBuf 压缩/spill） | 手动同步方案溢出导致性能劣化 | 手动同步须做 flag 配对预算；默认走 AutoSchedule 自动插 flag 对 |

## 2. 强制检测规则

在设计文档生成前，**必须**执行以下检测：

| 检测项 | 触发条件 | 处理方式 |
|--------|----------|---------|
| 一维 Kernel / threads= 误用 | 参考实现含 `T.Kernel(m, n, batch)` 三维形式或 `threads=` 参数 | **立即警告**，改为一维 + 核内映射方案 |
| GPU 专用 API | CUDA 专属 API（TMA / cp.async / warp shuffle / cluster 等） | **立即警告**，查 `examples/ascend/` 确认 Ascend API |
| C0 分形对齐风险 | M/N 非 16 倍数，或 K 不满足 `K % TILE_K_SUB == 0` 且 `TILE_K_SUB % C0 == 0` | **立即警告**，要求 design 明确 padding 策略 |
| L0C/UB 溢出风险 | `block_M × block_N × sizeof(accum)` > L0C 256KB × 90%；UB 预算超 216KB（SimtVF 场景） | **立即警告**，减小 block 或拆分 |
| OOB padding 误用 | `pad_value` 与 `data_select` 同传 / 非 GM→UB 路径使用 / `data_select` 前未设 pad 寄存器 | **立即警告**，按 skill §6.1 修正 |
| 分核策略缺失 | grid 划分无核数依据（无实查记录且未标明 assumption）、无 AIC/AIV 分工、无 `auto_schedule` 声明 | **立即警告**，要求 design §5 补分核三要素（grid 划分 / AIC-AIV 分工 / 同步策略） |

## 3. 警告输出格式

```
⚠️ 技术限制检测警告

检测到参考实现包含本项目不支持的功能：

1. 三维 Kernel（本项目只支持一维 T.Kernel(N)，且无 threads 参数）
   - 参考实现：T.Kernel(m_num, n_num, batch_count)
   - 本项目方案：T.Kernel(total_blocks) + 核内一维映射
   - 参考：examples/ascend/example_gemm.py

建议：
- 先查阅 examples/ascend/ 与 testing/ascend/ 中的同类实现
- 确认 Ascend API 用法（tilelang-ascend skill）后再生成设计文档

是否继续生成设计文档？
```

## 4. Host 侧 输入 操作约束

> **⚠️ 核心原则：host 侧禁止改动输入 NPU 张量 内的真实内容，禁止触发任何 Padding 操作**
>
> 算子的所有核心计算逻辑（数据搬运、数学运算、归约、维度重排、padding 等）必须在 kernel 函数内部完成（规范入口 `@tilelang.jit(target="ascend")` 或 `tilelang.compile(program, target="ascend")`）。**host 侧（kernel 外的 Python 代码）对 NPU 侧输入张量内的真实内容（数据值、物理排布、数据指针）一律不得改动**——只允许做只改 stride/shape 元数据的视图操作。**约束范围覆盖 kernel 调用前（输入预处理）和 kernel 调用后（输出后处理）的完整 host 代码路径。**
