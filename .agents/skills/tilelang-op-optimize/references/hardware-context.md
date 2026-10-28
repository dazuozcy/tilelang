# Ascend A5/950 Hardware Context（Stage 4 调优口径）

> 本文件是 `tilelang-op-optimize` 的硬件上下文速查。**硬件常数的唯一事实源是 [pattern-library/constants.md](pattern-library/constants.md)**——本文件只做条目 ID 导航与使用纪律，不复制数值大段；引用常数时以下列条目 ID 为准。当前全部条目 `status: assumption`（A5 冷启动，无实测基线），遵守 [INDEX.md](pattern-library/INDEX.md) §1 assumption 折扣纪律：设计期容量按 90% 预算并在 DESIGN.md 披露；**频率/绝对 TFLOPS/GBs 无静态值（代价模型纯 cycle 域）——绝对时延与吞吐换算必须 do_bench 实测，禁止编造频率**。

## Target Platform

- 设备：Ascend A5/950（npu-arch `dav-3510`）。
- 工具链：CANN 9.x + bisheng 编译器（`--npu-arch=dav-3510`，`BISHENG_HOME`）；toolchain_stamp 组装口径 "tilelang X + CANN 9.x + dav-3510(A5)"（`kb_stale_check.py`）。
- 规范入口：`@tilelang.jit(target="ascend")`（`pto` AOT 线不在 conductor 范围）。
- 核数实查：`torch.npu.get_device_properties()` 的 `cube_core_num` / `vector_core_num`（无静态常数；examples/ascend 惯例默认 32 blocks，待标定）。
- 测量工具：`tilelang.profiler.do_bench(backend="msprof")`（端到端时延唯一口径，返回毫秒）/ `backend="msprof_detail"`（`KernelProfile` 深挖）/ npusim（周期级归因唯一口径）。权威用法：`tilelang-ascend` skill §12 + `references/profiling.md`。

## 核组织与调度（CONST-arch-a5）

- AIC:AIV 物理编组 = 1:2（编译器可见 {Vector0, Cube, Vector1}）；AIV sub-core `sids=1|2`（默认 2，`sids=1` 只激活 AIV0）。
- `T.Kernel(N)` **一维 grid**（无 `threads=` 参数）；AIC/AIV 指派与同步默认 **AutoSchedule**（AssignCore/EstimateLatency/InsertSync）；调度提示：`T.Task(latency=, ii=)`、`T.Stage`、`T.PerCoreTask`、`T.Pipelined(num_stages=)`、`T.Persistent`。
- 完全手动核（显式 `T.Cube()`/`T.Vector()` + 手写 flag）需 `PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False`；手动 flag（`T.ascend_set_flag/wait_flag`/`T.ascend_pipe_barrier`）必须配对并注释。
- **flag/event 预算**：核内 flag 8 个、跨核 flag 16 个（溢出经 RewriteFlagToBuf 压缩/spill）——手动同步方案须做 flag 配对预算。
- 编程模式枚举（signal-registry §3 权威）：`simt`（`T.SimtVF(threads=N)` + `T.Parallel`，默认推荐）/ `simd`（`T.SimdVF()` + `T.simd.*` 2048-bit 显式向量，热点）/ `mixed`（`T.MixedKernel(sids=1|2)` + `T.gemm`），正交字段 `auto_schedule: yes|no`。

## 片上容量（CONST-capacity-a5）

- UB（`T.alloc_shared`）= 248KB，**含 `T.SimtVF` 时上限 216KB**（SIMT 线程上下文 32KB）；L1（`T.alloc_l1`）= 512KB；L0C（`T.alloc_l0c`）= 256KB；L0A/L0B = 64KB（GEMM K-sub 切分参考 `tilelang/ascend/op/gemm/gemm_mad.py`）。
- 缓冲对齐：L0C 1024B；L1/L0A/L0B 512B。
- 容量检查须计入 layout padding、buffer versions（AutoSchedule multibuffering）、编译器保留（`tilelang-ascend` skill §4.3）；GM/L2/HBM 容量运行时查询。
- 待实测转正项：multibuffering 下 UB 实际膨胀系数未知——设计期在 90% 折扣之外勿叠加未证实膨胀系数。

## GEMM 与 C0 分形约束（CONST-gemm-dim-constraints）

- C0（Cube fractal 宽度）= 32B / 元素大小（bf16/fp16=16、fp32=8、int8=32、fp4=64）；FRAC_N = 16；L0C 固定 16×16 fractal。
- M、N 须 16 倍数；`K % TILE_K_SUB == 0` 且 `TILE_K_SUB % C0 == 0`。
- L1-input GEMM 仅支持 NT（`trans_A=False, trans_B=True`），权重 GM 布局 **[N, K]**；L0-input GEMM 有独立布局处理，勿把 L1 限制推广。
- Blockscaled(MX)：K 须 64 整除；INT8 MMAD 不支持（A5）；HF32 模式（`T.set_hf32_mode`）为 fp32 吞吐换精度，须实测容差。
- L0C→UB dual_copy N-split：完整 N 须被 32 整除。

## 吞吐与带宽代价模型（设计期 roofline，cycle 域）

- CONST-cube-throughput-model：Cube ops/cycle/AIC（按 dtype 分档；一次乘加 = 2 ops）。
- CONST-mte-bandwidth-model：MTE2/MTE3/FixPipe/dual_copy 带宽（B/cycle）+ base cycles；N-strided/平台/宽行档位；ND2NZ 转换 base cycles（fractal 转换非免费）。
- CONST-valu-model：VALU ops/cycle/AIV（按操作类别）、VRF↔UB 带宽、2048-bit SIMD 向量寄存器（64×fp32 / 128×fp16）与整除性。
- 估算协议：`_shared/standards/hardware-cost-model.md` §3。**solved II / 编译器估计 latency 是调度模型输入，不是设备时延**；绝对时延以 do_bench(msprof) 实测为准。

## 使用纪律

1. 引用常数须带版本戳（工具链变更后自动待重验，`kb_stale_check.py` 检测）；外推前核对条目适用条件。
2. assumption 值与实测冲突时以实测为准，并 update 对应条目（转 `measured` 附 repro，SKILL.md Phase 4 第 6 条）。
3. 首次真机标定任务（vecadd copy-floor、GEMM roofline、flag/launch 开销）后转正 constants.md 待实测项。
