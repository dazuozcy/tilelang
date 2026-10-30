# _weighted_reduce 算子设计文档（IndexedExpertMLPFwdOp 加权归约 kernel，TileOPs 迁移）

> **任务标识**：project = `indexed_expert_mlp`，op = `_weighted_reduce`（`@tilelang.jit` 工厂函数名，保持不变），scenario = migration（TileOPs harness，family=moe）。上层算子 `IndexedExpertMLPFwdOp`（indexed/grouped 融合 MoE 专家 MLP 前向，DeepSeek-V3 / Qwen3-235B 规模）。整体算子编程模式为 mixed（GEMM 主导 kernel 走 Cube）；本函数是 **3 个提取函数之末的纯向量加权归约 kernel**，按用户指令以 **SimdVF（simd）** 设计（无 GEMM，不用 Cube/MixedKernel）。
>
> **源码版本**：GPU 仓 `/home/l00970450/z00875089/TileOPs` @ `3979278e2da6ac4fa84d8c7c800dec64ea9c0425`（clean）。Stage 0 转换记录：`examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/conversion_record.md`；提取参考（逐字）：`examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/_indexed_expert_m_l_p_fwd_kernels.py`（`_weighted_reduce` L205–237）。
>
> **修订标识**：revision_index = **1**（revision，2026-10-29）。上一版备份 `history_version/design_v0.md`（first_design，同日）；本次修订由 Stage 2 检视不通过触发（REVIEW.md 问题 1〔阻塞：OPT-2 lane 映射置换错误 + E3 自证式验证〕+ 问题 2〔建议：尾声 op 计数拆解口径不一致〕），已逐条修复。
>
> **相对上一版（v0）的关键调整与防复发机制**（revision 专属章节）：
> 1. **问题 1 修复——方案 (a)：保留 OPT-2，尾声改 flash-core 式 vor 重合并**。v0 的错误：假设 `vcvt(part=0/1)` 16→32 加宽为「连续前/后 64 元素半」并按此把两半连续存储；而四源硬件证据确证真实语义为**偶/奇 lane 交错拆分**——`src/tl_templates/ascend/nd2nz_copy.h` L149–154（PART_EVEN/PART_ODD 加宽后必须 `vintlv` 重交错才能连续存储）、`examples/ascend/flash_attention/core.py` L128–152（DINTLV 偶奇拆分 + narrowing part=0/1 + `vor` 位或重合并，order-sensitive 路径真机验证）、CANN `basic_block_vf_nd.h`（PART_EVEN 奇偶注释 + FP32_TO_F16_EVEN/ODD + `Or<uint16_t>` 合并范式）、`tilelang/ascend/language/simd.py` vcvt docstring（"even/odd 2-way splits"）。v0 尾声按连续半存储必然产出 de-interleave 置换。修复后：装载侧偶奇拆分（acc_even[m] ↔ 列 g·128+2m、acc_odd[m] ↔ 列 g·128+2m+1）与尾声侧偶奇分置 + `vor` 重合并 + `vsts(NORM_B16)` 单发全宽存储**互为逆置换，输出位置恒等**；组 op 总数口径不变（尾声 4 op：vor +1 与 vsts −1 相抵）。防复发机制：lane 语义全文统一为偶奇拆分（消除 v0 §1.6.1 与 §3.1 的互斥表述），全部 lane 映射断言锚定四源证据链而非单一 docstring。
> 2. **问题 1 必做项——E3 重建为独立 lane-mapping 模型**。v0 的 `run_e3()` 两路径为同一表达式 `x16.float()`（自证式：只证数值加宽精确、未建模 lane 索引映射，无法捕获置换错误）。重建后：**E3a**（采纳形态：NORM-128 装载 → 偶奇加宽 → FMA → 偶奇收窄 → vor 重合并 → NORM_B16 存储 → extent-BH 写回，显式 gather/interleave 索引张量）与 **E3b**（fallback 形态：UNPK-64 装载 + part=0 收窄 + PK_B32 存储，cast 测试 L47–53/L77–80 两处保序真机形态）对照**参考公式直算**（[T,K,H] 张量式，结构独立的第二条代码路径），并注入 padded-lane 垃圾值；**E3neg 负对照**模拟 v0 错误形态并断言其被检出（bit_mismatch 98.4% / viol 91.0%）——证明新 E3 具备置换检出能力（v0 缺的就是这一步）。E1–E6 全量重跑通过。
> 3. **问题 2 修复——op 计数拆解口径统一**。按伪代码逐段列 op：每 slot 装载 3（1 vld NORM + 2 vcvt 加宽）+ FMA 2–4（2 vmula 指令 / 1–2 lane-op 双口径）+ 广播 1（vld BRC）；每组 vdup 2 + 尾声 4（2 vcvt 收窄 + 1 vor + 1 vsts）+ vmuls 2（仅 s≠1.0）。分项之和 = 组总数 **54–70**（K=8、s=1.0；FMA 指令/双 lane-op 口径），与 v0 总数一致但分项拆解修正（v0 的「尾声 3」/「2 op 尾声」错误已消除）；§1.6.0 R3 表、§1.6.1 结论行、§1.6.3/§5.4 微调项 op 数字（vaxpy 46、GU 展开 48–50）同步对齐。
> 4. **B0 选定算法族不变**（纵向 lanes=H Vector-FMA 累加）；OPT-1（scaling 编译期特化）与 OPT-3（首槽乘，Stage 4 微调项）不受波及、原样保留；§0.1–§0.5/§0.7 源算子三问解读与 §5 分核三要素完整保留（§0.6 R1 语义保持论证随问题 1 修复重新出具）。
>
> **上游数据流**：`_route_stats`（元数据，grouped 分支）→ `_indexed_expert_gemm`（gate/up + down 两次调用，产出 `expert_output[T,K,H]`）→ **本函数**（加权求和 × scaling）。本函数不消费 route metadata，只消费 `expert_output` 与 `topk_weights`。

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 tilelang-op-design skill 的 `references/migration-analysis.md`）：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§10 的所有设计决策。解读对象为源文件 `/home/l00970450/z00875089/TileOPs/src/tileops/_kernels/moe/indexed_expert_gemm.py` 的 `_weighted_reduce` 工厂（L219–252）与 `IndexedWeightedReduceKernel` wrapper（L430–501），以及 host 侧调用点 `src/tileops/_ops/moe/routed_expert/indexed_routed_expert.py` 的 `IndexedExpertMLPFwdOp.forward` 末步（conversion_record.md §3 步骤 4）。

### 0.1 源算子语义（做什么）

**数学语义**：给定 `expert_output[T, K, H]`（dtype ∈ {fp16, bf16}，K = top_k 条路由的专家输出）与 `topk_weights[T, K]`（fp32 路由权重），对每个 (token, col) 独立计算加权和并缩放：

$$
\text{output}[t, h] = \text{cast}_{\text{dtype}}\Big( \text{scaling} \cdot \sum_{k=0}^{K-1} \text{f32}\big(\text{expert\_output}[t, k, h]\big) \cdot \text{topk\_weights}[t, k] \Big)
$$

**规约语义**：归约轴 = K（中间轴，长度为编译期常量，harness 恒 K=8，测试含 K∈{2,4}）；累加顺序 = **slot k 升序串行**（源 `for slot in T.serial(top_k)`）；累加域 = **fp32**（`T.Cast("float32", x)` 后逐项乘加）。

**dtype 语义**（逐元素 cast 链，源 L245–248）：
1. 每个输入元素 up-cast 到 fp32：`f32(x)`（fp16/bf16 → fp32 精确，无舍入）；
2. fp32 乘 `topk_weights[t,k]`（fp32 舍入）→ fp32 累加（fp32 舍入）；
3. 求和后乘 `scaling`（fp32 舍入，工厂闭包烘焙的 Python float 编译期常量）；
4. **单次** narrowing cast 回 dtype（`T.Cast(dtype, acc*scaling)`，ROUND 最近偶数）。

即每元素恰 1 次 up-cast + 1 次 down-cast，中间全 fp32——cast 次数已是该语义下的最小值。

**边界语义**：NaN/Inf 按 IEEE 传播（`inf × 0 = NaN` 等在 fp32 乘加链中自然发生）；计算路径无守卫条件（源 kernel 唯一 `if` 是 launch 边界 `index < T*H`）；K=1 时退化为纯数乘；scaling 可为任意 float（harness 全部行默认 1.0；测试 `test_the_indexed_op_scales_its_output` 用 2.5）。

**I/O 语义**：原位写回模式（`out_idx=[]`，`output` 为入参，wrapper 返回传入的 out）。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现（fp32 域乘加、slot 升序、scaling 后置、单次 narrowing cast），不复刻 NPU 实现细节（UB staging / FMA / BRC 广播均不出现于 golden）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| expert_output | (T, K, H) | fp16 / bf16 | 行主连续；K 为归约轴（中间轴），H 为最内连续轴；由 `_indexed_expert_gemm`（down 投影）产出 |
| topk_weights | (T, K) | **float32** | 路由权重（workload 生成 `softmax(randn(T,K))`）；源码强制 fp32（wrapper L494–495 校验） |
| output | (T, H) | fp16 / bf16（同 expert_output） | **原位写回**（out 模式，`out_idx=[]`）；输出 shape 依据：工厂张量声明（源 L233）+ wrapper `forward` 的 `(T, H)` 校验（源 L490–491） |

- 契约来源：GPU `IndexedWeightedReduceKernel.forward`（源 L480–497）逐字校验 shape/dtype；A5 wrapper（冻结，不改）：`examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/indexed_expert_mlp.py` `IndexedWeightedReduceKernel`（L336 起）——`self.kernel = _weighted_reduce(num_tokens, top_k, hidden_size, dtype_str, float(scaling))`，`default_config = {}`（threads 已按 K3/K9 移除），调用面 `self.kernel()(expert_output, topk_weights, output)`。
- 工厂签名（冻结）：`_weighted_reduce(num_tokens, top_k, hidden_size, dtype, scaling)` → 返回无参 callable（GPU 版为 `_kernel(threads)`，A5 版按 conversion_record §1 K1/K3 去 threads）。
- 上层分支：本 kernel **无条件执行**（T=1 direct 与 T>1 grouped 两条路径的最终汇合步，conversion_record §3 步骤 4）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每个计算语句归入某步骤，无遗漏、无臆造；行号指源文件 `_weighted_reduce`）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `with T.Kernel(ceildiv(T*H, threads), threads=threads) as bid`（L235） | — | — | 输出元素空间 T·H 的一维 block 划分（256 threads/块，default_config L477–478） |
| 2 | `tid = T.get_thread_binding(); index = bid*threads + tid`（L236–237） | bid, tid | 全局元素下标 index | 元素 → (token, col) 的扁平映射 |
| 3 | `if index < T*H: token = index // H; col = index % H`（L238–240） | index | token, col | 边界守卫 + div/mod 索引分解 |
| 4 | `acc = T.alloc_local((1,), "float32"); acc[0] = 0`（L241–242） | — | 寄存器累加器 | fp32 累加域初始化 |
| 5 | `for slot in T.serial(top_k): acc[0] += T.Cast("float32", expert_output[token, slot, col]) * topk_weights[token, slot]`（L243–247） | expert_output (GM 标量读), topk_weights (GM 标量读) | acc | 加权累加（§0.1 cast 链步骤 1–2，slot 升序） |
| 6 | `output[token, col] = T.Cast(dtype, acc[0] * scaling)`（L248） | acc, scaling | output (GM 标量写) | scaling 乘 + narrowing cast（cast 链步骤 3–4） |

**数据流与内存访问模式**（源硬件 = CUDA SM 视角）：

```
GM[expert_output] --逐线程标量 load（同 warp 内 col 连续 → coalesced 2B 事务）--> 寄存器
GM[topk_weights]  --逐线程标量 load（warp 内同 token 广播读）--> 寄存器
寄存器 acc[1] fp32 --K 步串行乘加-->
GM[output] <--逐线程标量 store（coalesced）--
```

无 SMEM / 无 fragment / 无 staging：全 kernel 直接在 GM 上以标量粒度读写；累加器驻留寄存器（`T.alloc_local`）。

**循环与并行结构**：grid = `ceildiv(T*H, 256)` blocks × 256 threads（1-D grid + thread 两级并行）；每 thread 恰一个 (token, col) 输出元素；核内唯一循环 = K 步串行（编译期常量边界）；无块间/线程间交互（输出元素空间两两不相交）。

**host 侧逻辑**：`functools.lru_cache(maxsize=32)` 按 `(T, K, H, dtype, scaling)` 特化缓存工厂；`@tilelang.jit(out_idx=[])` 原位写；wrapper 校验 shape/dtype 后 `self.program(self._compile)(expert_output, topk_weights, output)`；Op 侧在 down 投影后调用（`out=` 语义：结果写入调用方给定缓冲并返回）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | 1-D grid 扁平化 + thread-per-element（T·H 展平分派） | 输出空间满并行；warp 内 col 连续 → coalesced 事务合并 | `T.Kernel(ceildiv(T*H, threads), threads)` + 扁平 index | CUDA SIMT 两级并行 + memory coalescing | 硬件强相关（A5 一维 grid 无 threads 参数） |
| 2 | 寄存器累加器（`T.alloc_local((1,), "float32")`） | K 步累加零访存（数据驻留寄存器） | register tiling | GPU 大寄存器堆 | 模式可移植（NPU VRF 向量寄存器等价承接） |
| 3 | fp32 累加 + 末尾单次 narrowing cast | 精度语义（§0.1）+ cast 次数最小化 | cast 链设计 | 无（纯算法层） | 保留 |
| 4 | scaling 融合在 cast 前（fp32 域一次乘） | 免 per-slot 缩放（K 次乘 → 1 次乘） | 乘法对加法的分配顺序选择 | 无（纯算法层） | 保留 |
| 5 | 每线程 div/mod 索引分解（`token = index // H` 等） | 扁平下标 → 2-D 索引 | GPU 标量 ALU 富余（每线程 1 div + 1 mod 摊薄） | 标量算术与线程并行的相对价格 | 硬件相关（A5 标量 div/mod 走 13 ops/cycle 慢档且无线程摊薄，`CONST-valu-model`） |
| 6 | GM 直读直写（无 staging） | 最小代码路径；coalescing + L2 已足够 | 标量 load/store 直落 GM | GPU coalescing / L2 cache | 硬件强相关（A5 SimdVF 块内禁 GM 访问，须 UB staging） |
| 7 | host `lru_cache` 特化 + `out_idx=[]` 原位写 | 编译期烘焙 T/K/H/dtype/scaling；免输出分配 | 工厂闭包 | 无 | 保留 |

> 注：本 kernel 在源平台即为带宽型轻量 kernel（T=64 时读 7.34MB / 写 0.92MB，FLOPs ≈ 2·T·H·K + O(T·H)）；上述手段的意图是"满并行 + coalesced 访存 + 寄存器驻留"。迁移承接判断以 §1.6.0 的定量口径为准。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级（语义/算法/优化） | 源硬件依赖 | NPU 有等价能力？ | 处置（保留/等价替换/重新设计/舍弃） | NPU 对应方案 / 依据 |
|------|----------------------|-----------|-----------------|-----------------------------------|---------------------|
| 1. 加权求和公式 + fp32 累加域 + slot 升序 + 单次 narrowing cast（§0.1 全部 cast 链） | 语义 | 无 | — | **保留** | 语义层无条件保留；§1.6.1 各 OPT 均在此链内做恒等变形（E1–E5 机器验证） |
| 2. scaling 后置单次乘（§0.4 #4） | 算法 | 无 | — | **保留** | 纯算法层（migration-analysis.md §5.3「online/算法层 → 保留」）；==1.0 时编译期跳过（OPT-1，E1 验证） |
| 3. K 步串行累加结构（寄存器驻留） | 算法 | 无（寄存器累加是通用模式） | 有：VRF 向量寄存器 + `T.simd.vmula` FMA（`testing/ascend/language/simdvf/test_simdvf_arithmetic.py L106` serial_vreg_accum：T.serial 循环 + vmula 寄存器累加回归） | **等价替换** | 每线程 1 个 fp32 标量累加器 → 每向量寄存器 64/128 lane 并行累加器（`T.simd.alloc_var` + vmula）；累加顺序同为 slot 升序（§0.6 R1 语义论证） |
| 4. 1-D grid + threads=256 两级并行（§0.4 #1） | 优化 | CUDA SIMT | 无（本项目 `T.Kernel` 一维、无 `threads=` 参数，传入即 TypeError——tilelang-ascend skill §2.1；migration-analysis.md §5.3「block × thread 两级并行 → 重新设计」） | **重新设计** | → §0.6 R1：block = (token, h-chunk) 二维分解的一维映射 + SimdVF 128-lane 纵向向量域；threads 已从 wrapper 移除（conversion_record §1 K3/K9），线程域不引入 |
| 5. 每线程 div/mod 索引（§0.4 #5） | 优化 | GPU 标量 ALU 价格 | 有更优路径：block 级 2 次 div/mod（bid → token, h_blk）+ 组内仿射寻址（`g*128` 循环变量仿射式，免 div/mod） | **等价替换** | div/mod 从 T·H 次降到 T·nh 次（block 级，每 block 2 次）；组内 `x_ub[k, g*128]` 为循环变量仿射式 |
| 6. GM 标量直读直写（§0.4 #6） | 优化 | coalescing + L2 | 无（SimdVF 块内禁止 GM 访问——`testing/ascend/analysis/test_ascend_vf_checker.py`："SIMD_VF blocks cannot access global memory"） | **等价替换** | → §0.6 R2：`T.copy` GM↔UB bulk 搬运（MTE2/MTE3，100/115 B/cycle 档，`CONST-mte-bandwidth-model`）+ UB 上向量计算；coalescing 意图由 2-D 连续区拷贝承接 |
| 7. 边界守卫 `if index < T*H`（§0.3 步骤 3） | 优化 | GPU 末块线程闲置处理 | 不需要（块划分取 BH \| H 整除 + UB padding，无越界读；见 §5.2/§5.4） | **舍弃** | 被结构消除：host 侧除数选取保证 BH \| H（无尾块），padded UB 吸收 lane 尾区（E6 机器验证：GM 写集恰为 [0,H)、padded lane 不落 GM）；源守卫的意图（防 OOB）由静态 extent 保证 |
| 8. host lru_cache / out_idx=[] / wrapper 校验 | host | 无 | — | **保留** | A5 wrapper 已按同契约脚手架化（indexed_expert_mlp.py L336–380） |
| 9. `default_config = {"threads": 256}` | host | CUDA threads | 无 | **舍弃** | K3/K9：A5 wrapper `default_config = {}`（已落地），工厂 callable 无参 |

> 每项处置依据：migration-analysis.md §5.3 映射表条目（两级并行/寄存器累加/直访 GM 行）+ `testing/` 佐证（条目 3/6）+ tilelang-ascend skill §2.1（条目 4）+ 本设计 §5.2（条目 7）。舍弃项（7/9）理由已写明；无静默丢弃。

### 0.6 NPU 算法重设计

对 §0.5 中判定为「重新设计」的条目逐项给出 NPU 新算法（判定输入含 §1.6.0 调研结论——源算法与调研候选同台比较，选定算法族为「纵向 lanes=H 的 Vector FMA 加权累加（BRC 广播权重 + vmula 寄存器累加 + UB 全块 staging）」，见 §1.6.0 调研结论）：

**重设计项 R1: thread-per-element 两级并行 → (token, h-chunk) block + SimdVF 纵向向量域**

- **源方案**：`T.Kernel(ceildiv(T*H, 256), threads=256)`，每 thread 一个 (token, col)，K 步寄存器串行累加。性能意图（§0.4 #1/#2）：输出空间满并行 + coalesced 访存 + 累加驻留寄存器。
- **NPU 新算法**（承接满并行 + 寄存器驻留意图，替换并行载体）：
  1. host 编译期选取 `nh`（h-chunk 数，取 H 的除数，§5.2 启发式：目标总 block 数 ≈ 4×56 AIV、每 chunk ≥ 128 列）与 `BH = H // nh`；
  2. `T.Kernel(T * nh)` 一维 grid；`token = bid // nh`，`h0 = (bid % nh) * BH`（block 级标量分解，§0.5 条目 5 的等价替换）；
  3. `with T.SimdVF()` 内 `for g in T.serial(BH_pad // 128)`：每 128 元素组双累加器 `acc_even/acc_odd`（`alloc_var` fp32），`for k in range(K)` 编译期展开：`w_vec = vld(w_ub[k], "BRC_B32")`（权重广播）→ `x_h = vld(x_ub[k, g*128])`（128-lane 16-bit NORM 装载，寄存器 lane j ↔ 列 g·128+j）→ `vcvt(part=0)` **偶 lane 加宽**（acc_even[m] ↔ 列 g·128+2m）→ `vcvt(part=1)` **奇 lane 加宽**（acc_odd[m] ↔ 列 g·128+2m+1）→ `vmula(acc_even/acc_odd, x_half, w_vec)` FMA 累加（slot 升序）；
  4. 尾声：`vmuls`（scaling ≠ 1.0 时，OPT-1）→ `vcvt(acc_even, dtype, part=0)` + `vcvt(acc_odd, dtype, part=1)` 偶/奇 16-bit lane 分置 → `vor`（`T.reinterpret(·, "uint16x128")` 位或）重合并为自然交错序 → `vsts(dist="NORM_B16")` 单发存 128 连续元素（**逆交错步骤显式化**——flash-core L128–152 先例）。
- **语义保持论证**：
  1. **累加顺序**：k 展开为 slot 升序的 FMA 链，与源 `T.serial(top_k)` 顺序一致；vmula 的硬件舍入行为（融合单舍入 vs 乘加两舍入）对两种可能均做了机器验证（E4f/E4s：vs 源链 vs fp64 参照，全部 ≤1 ulp fp16、0% 违反 @1e-2，EQUIV_PASS）——对 §8.2 分档容差（fp16/bf16 1e-2）鲁棒；
  2. **cast 链与 lane 映射**：`vcvt` 16→32 位加宽为精确映射（加宽转换不涉 round/sat——`tilelang/ascend/language/simd.py` vcvt docstring）且 `part=0/1` 为**偶/奇 lane 拆分**（PART_EVEN/PART_ODD，四源佐证：`src/tl_templates/ascend/nd2nz_copy.h` L149–154〔PART_EVEN/ODD 加宽 + vintlv 重交错〕、CANN `basic_block_vf_nd.h`〔奇偶注释 + EVEN/ODD 收窄 + Or 合并范式〕、`examples/ascend/flash_attention/core.py` L128–152〔vor 重合并真机〕、simd.py docstring）；尾声收窄 part=0/1 把两半 64 值分置偶/奇 16-bit lane，`vor` 按位或重合并（偶奇 lane 位置不相交 → OR = 合并）后 `vsts(NORM_B16)` 存 128 连续元素——**装载拆分与尾声合并互为逆置换，输出位置恒等**（E3a lane-mapping 机器验证：58 用例位级恒等，含 fp16 全 65536 位模式 + bf16 全位模式扫描与 padded-lane 垃圾注入；E3neg 负对照证明该 check 具备置换检出能力）；narrowing `vcvt(f32→fp16/bf16)` 与源 `T.Cast` 同为 ROUND 最近偶数；
  3. **边界**：NaN/Inf 传播在同序 fp32 链中一致（E4/E5 角点用例 nonfinite mask 全等）；padded lane [BH, BH_pad) 的计算结果不落 GM（E6 布局断言：每 token GM 写集恰 [0,H) 一次覆盖、无越界）；
  4. **验证方式**：§8 golden（§0.1 语义）+ L0 分档容差对照 + op 级移植测试（`examples/TileOPs/tests/ops/test_indexed_expert_mlp.py` 全部用例经 op 链路覆盖本 kernel）。
- 源 §0.4 #1 意图承接：满并行 → 56–256 blocks 覆盖 56 AIV（T=1 时 nh=56 专项填补，§5.2）；#2 意图承接：累加驻留 VRF 寄存器（vmula dst）。

**重设计项 R2: GM 标量直访 → 三段 T.copy staging + 静态 extent 搬运**

- **源方案**：每 thread 直接 GM 标量 load/store（coalesced）。性能意图：最小代码路径，访存合并交给硬件。
- **NPU 新算法**：
  1. `T.copy(expert_output[token, 0:K, h0:h0+BH], x_ub[0:K, 0:BH])`：3-D GM 张量标量行 + 双切片区（2-D [K, BH] 块）GM→UB bulk——形态先例：同仓 `examples/indexed_expert_mlp/_indexed_expert_gemm/_indexed_expert_gemm.py` L288–292（`weights[expert, nb*block_n:…, ko*block_k:…]` 3-D 标量索引 + 双切片 T.copy，Stage 3 已验证）；extent 全编译期静态（BH \| H 无尾块——`_indexed_expert_gemm` RETROSPECTIVE 教训「运行期钳制 extent 静默 OOB」的正面规避）；
  2. `T.copy(topk_weights[token, 0:K], w_ub[0:K])`：1-D 区（K fp32 ≤ 32B，字节粒度行——logsumexp_simd L1 先例含 N=1 极小行通过）；
  3. `T.copy(out_ub[0:BH], output[token, h0:h0+BH])`：UB→GM 静态 extent 搬出（部分范围拷出先例：logsumexp_simd `T.copy(lse_ub[0:SUB], y[t*SUB:…])`）。
- **语义保持论证**：staging 不改变任何计算内容（数据逐元素恒等搬运，layout 保持 [K, BH] 行主——输入 [T,K,H] 的 K-中间布局无需转置，ALG-transpose 行的核内重排/host permute 候选均不需要，§1.6.3 候选 4/5 弃选论证）；coalescing 意图由 2-D 连续区 bulk DMA 承接（每行 BH·2 ≥ 256B 连续，宽行档 100 B/cycle，`CONST-mte-bandwidth-model`）。
- 源 §0.4 #6 意图承接：访存路径从「per-thread coalesced 标量事务」换「per-block bulk DMA」，GM 流量不变（§1.6.0 R3：读 T·K·H·2 + T·K·4 / 写 T·H·2，每元素恰读一次写一次——流量下界）。

（§0.5 无其他「重新设计」条目；条目 3/5/6 的等价替换已并入 R1/R2。）

### 0.7 标杆实现

- **源算子定义**（GPU 仓，绝对路径）：`/home/l00970450/z00875089/TileOPs/src/tileops/_kernels/moe/indexed_expert_gemm.py`（`_weighted_reduce` L219–252、`IndexedWeightedReduceKernel` L430–501）；host 调用点 `/home/l00970450/z00875089/TileOPs/src/tileops/_ops/moe/routed_expert/indexed_routed_expert.py`（`IndexedExpertMLPFwdOp.forward` 末步）；接口契约 `/home/l00970450/z00875089/TileOPs/src/tileops/_kernels/moe/call_spec.py`（`IndexedWeightedReduceFwdInterface`）。
- **提取参考**（逐字，A5 仓）：`examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/_indexed_expert_m_l_p_fwd_kernels.py`（`_weighted_reduce` L205–237）。
- **golden 依据**：源仓无 kernel 级 weighted-reduce 参考实现（GPU 测试全部 op 级输出对照，`test_weighted_reduce_noop*` 属未迁移的共享 ABC 用例——conversion_record §4 排除清单）；op 级 golden `ref_routed_experts`（`examples/TileOPs/workloads/moe.py`）覆盖本函数语义的上层闭环。故 §8.1 kernel 级 golden 按 §0.1 语义自行给出（fp32 域 slot 升序乘加 + scaling 后置 + 单次 narrowing cast）。
- **A5 wrapper（冻结契约）**：`examples/TileOPs/src/tileops/_kernels/moe/indexed_expert_mlp/indexed_expert_mlp.py`（`IndexedWeightedReduceKernel`，L336 起；`default_config = {}`）。
- **测试/bench**：`examples/TileOPs/tests/ops/test_indexed_expert_mlp.py`、`examples/TileOPs/benchmarks/ops/bench_indexed_expert_mlp.py`。
## 1. 概述

### 1.1 算子名称

`_weighted_reduce`（IndexedExpertMLPFwdOp 的加权归约 kernel；TileOPs kernel class `IndexedWeightedReduceKernel`）

### 1.2 功能描述

对路由维 K 展开的专家输出 `expert_output[T, K, H]` 做逐 (token, col) 的 fp32 加权求和并乘 `scaling`，narrowing cast 回 fp16/bf16 写入 `output[T, H]`（原位）——MoE 前向的最终 combine 步：把 top_k 条路由的专家 MLP 输出按路由权重汇成单 token 表示。

### 1.3 数学公式

设 K = top_k，s = scaling（编译期常量）：

$$
\text{output}[t, h] = \text{cast}_{\text{dtype}}\Big( s \cdot \sum_{k=0}^{K-1} \text{f32}\big(\text{expert\_output}[t,k,h]\big) \cdot \text{f32}\big(\text{topk\_weights}[t,k]\big) \Big), \qquad \forall\, t \in [0,T),\ h \in [0,H)
$$

语义约束（§0.1）：slot 升序 fp32 累加；scaling 在 fp32 域后置单次乘；单次 narrowing cast；NaN/Inf IEEE 传播。

### 1.4 算法描述

**迁移决策后的 NPU 侧算法**（= §1.6.0 选定算法族，经 §0.6 R1/R2 重设计落地；与源算法的结构差异见行末标注）：

1. **块划分**（host 编译期）：选 `nh`（H 的除数，目标总 block ≈ 4×56、每 chunk ≥ 128 列，§5.2）→ `BH = H // nh`、`BH_pad = align_up(BH, 128)`、`ng = BH_pad // 128`（差异：源为 T·H 元素扁平线程划分，无块内向量结构）。
2. **搬入**：`T.copy` 三段静态 extent bulk——`expert_output[token, 0:K, h0:h0+BH]`（2-D [K, BH] 块）→ `x_ub`；`topk_weights[token, 0:K]` → `w_ub`（差异：源逐线程 GM 标量直读）。
3. **向量累加**（SimdVF 域，`for g in T.serial(ng)` × `for k in range(K)` 编译期展开）：`w_vec = vld(w_ub[k], "BRC_B32")` 权重广播 → `x_h = vld(x_ub[k, g*128])` 128-lane 16-bit 装载（寄存器 lane j ↔ 列 g·128+j）→ `vcvt(part=0)` 偶 lane 加宽（acc_even[m] ↔ 列 g·128+2m）+ `vcvt(part=1)` 奇 lane 加宽（acc_odd[m] ↔ 列 g·128+2m+1）→ `vmula(acc_even/acc_odd, x_half, w_vec)` FMA 累加（slot 升序，累加器驻留 VRF；差异：源为每元素 K 步标量乘加链——本设计 lane 沿 H 纵向并行、K 归约在寄存器内闭合）。
4. **尾声与搬出**：`vmuls`（仅 s ≠ 1.0，OPT-1）→ `vcvt(acc_even/acc_odd, dtype, part=0/1)` 偶/奇 16-bit lane 分置 → `vor` 位或重合并（逆交错，flash-core 先例）→ `vsts(dist="NORM_B16")` 单发写 128 连续元素入 `out_ub`（padded lane 计算结果滞留 UB，不落 GM，E6）→ `T.copy(out_ub[0:BH], output[token, h0:h0+BH])`。

cast 链（每元素 1 up-cast + 1 down-cast、中间 fp32）与源逐点一致（§0.4 #3 保留）。

### 1.5 数据流图

```
GM[expert_output T×K×H] --T.copy(MTE2, 2-D [K,BH] 静态 extent)--> UB[x_ub K×BH_pad]
GM[topk_weights T×K]   --T.copy(MTE2, 1-D K fp32)-->              UB[w_ub K]
UB[x_ub] + UB[w_ub] --SimdVF 向量链(vld NORM/BRC + vcvt 偶奇加宽 + vmula, K 展开升序)--> VRF[acc_even/acc_odd fp32×64]
VRF[acc] --vmuls(仅 s≠1) + vcvt(part=0/1 偶奇分置) + vor 重合并 + vsts(NORM_B16)--> UB[out_ub BH_pad]
UB[out_ub[0:BH]] --T.copy(MTE3, 静态 extent BH)--> GM[output T×H]
```

无 L1/L0 参与（纯 Vector）；无 ND2NZ/fractal 转换（无 Cube 段）。

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先**调研**（调研四问，§1.6.0）——同一数学语义（"K 路加权求和 + 编译期缩放 + 单次窄化 cast"）的算法族系统调研，源算法（GPU thread-per-element 串行累加）只是基线候选之一；再数学等价地优化公式（§1.6.1）；再做循环 / 标量计算的向量化替代分析（§1.6.2）；然后回答向量化轴与核内布局（§1.6.3）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。本算子为**规约/加权求和类**（多输入逐元素乘加 + 中间轴归约，非单步逐元素），按调研深度分级取**完整调研**（R1 候选表 + R3 四口径对比表 + R4 逐候选评估）。

#### 1.6.0 算法调研（Algorithm Research）⭐

**调研对象**：同一数学语义的算法族——所有满足 §0.1 契约（slot 升序 fp32 累加〔或容差内等价〕+ scaling 后置 + 单次 narrowing cast + NaN/Inf 传播）的计算结构。源算法（逐元素串行 K 步累加）为基线候选 0。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1 完成）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线**：源算法（thread-per-element 串行累加） | 每 (t,h) 一执行单元，`acc = 0; for k: acc += f32(x_k)·w_k; out = cast(s·acc)` | —（基线） | — | ✅（基线） |
| 1 | **纵向 Vector-FMA**（lanes=H，K 寄存器累加） | lane ↔ 输出列 h；slot 升序 FMA 链（`vmula`）于 VRF；权重 `BRC_B32` 广播 | 数学恒等（同序同域累加；FMA 舍入差异容差内，E4 双口径机器验证） | 56 AIV × 51 lane-ops/cycle 满载；K 归约零额外 op（寄存器内闭合） | ✅（选定核心） |
| 2 | Cube 批式 GEMV / GEMM 分形（ALG-gemv 行） | 每 token 一条 GEMV：`output[t,:] = W_row[t,1×K] @ X[t,K×H]`；或 M=16-pad 批 GEMM | 数学恒等（padding 区零贡献） | Cube 算力（8192 ops/cycle/AIC） | ✅（R3/R4 定量淘汰） |
| 3 | 水平 lane=K 向量归约 | lane ↔ slot k（64 lane 装 K 个 x）；`vcadd` 跨 lane 求和 | 数学恒等（同集合求和；跨 lane 加法序变化在 fp32 容差内） | K 步串行 → 1 条跨 lane 归约指令 | ✅（R3/R4 定量淘汰） |
| 4 | scaling 预折叠进权重（`Σ x·(s·w)`） | 先乘 s·w 再累加 | **非等价**（舍入位置改变：每 slot 一次额外 fp32 舍入；且 s≠1 时多 K−1 次乘） | — | ❌ 语义拒绝（§0.1 cast 链步骤 3 为「求和后乘」） |
| 5 | fp16/bf16 域累加（免 up-cast） | `acc_h += x_h · w_h` 全 16-bit 域 | **非等价**（fp32 累加语义，§0.1 dtype 语义） | −K cast/elem | ❌ 语义拒绝（语义层不可妥协） |
| 6 | ALG-pooling 行命中：「sum pool × 常数 = avg pool」同构 → s == 1.0 时化简 | s=1.0：`cast(1.0 · acc)` → `cast(acc)`（乘法跳过） | 恒等（IEEE fp32 x·1.0 ≡ x，NaN payload 差异不可观察；E1 机器验证 28 用例 0 差异） | −1 向量 op / 128 元素组（harness 全部行 s=1.0 → 热路径零开销） | ✅（并入候选 1 为 OPT-1） |
| 7 | 16-bit 域装载形态二选一：UNPK-64（`vld(dist="UNPK_B16") + vcvt(part=0)` 每 64 元素 ×2）vs NORM-128（`vld` 128-lane + `vcvt(part=0/1)` 偶奇加宽 ×2 + 尾声 `vor` 重合并） | 装载链 op 数：4 ops/128e/slot vs 3 ops/128e/slot（尾声均 4 op/组：UNPK = 2 vcvt + 2 vsts PK_B32；NORM = 2 vcvt + 1 vor + 1 vsts NORM_B16） | 恒等（均为逐元素精确加宽 + 输出位置恒等——NORM 形态装载偶奇拆分与尾声偶奇重合并互为逆置换；E3a/E3b lane-mapping 双形态机器验证位级恒等，E3neg 负对照证明置换可检出） | 装载链 op 数 −25%/slot；组总数 −8 op（−12.9% 指令口径） | ✅（并入候选 1 为 OPT-2） |

> 查证纪律注记（负向断言先机械 grep）：「T.simd 无 16-bit→32-bit 融合乘加（cast+FMA 一体）原语」——grep `axpy\|vmla\|muladd\|vmula\|vmadd` 于 `tilelang/ascend/language/simd.py`：FMA 族仅 `vmula`（L653，dst+src0·src1 同 dtype）/ `vmadd`（L667）/ `vaxpy`（L681，标量乘加）；无带 vcvt 融合变体 → 逐元素 up-cast 不可消除（计入 §1.6.1 成本模型）。候选 5 的拒绝与此独立（语义层）。

**R2 在线算法**：**无更优在线变体**。结构依据：(a) 归约轴 K 为编译期常量（≤8），每块工作集（K×BH_pad×2B ≤ 28.7KB）全量驻留 UB，K 步累加在 VRF 寄存器内闭合——不存在「归约轴超驻留需分块 running 统计量」的场景；(b) 本算子天然单遍流式：每个输入元素恰被读一次、每个输出元素恰被写一次（R3 流量下界），无两遍/三遍结构可在线化（对比 softmax 的 max→sum→normalize 三遍依赖——本函数无跨元素统计量依赖）；(c) 源算法即单遍（§0.4 无 online 手段也无需引入）。收益口径（遍数 1 / 中间缓冲 O(block) 驻留 UB）已在候选 1 达到该口径下限。ALG-reduction 行的「分块归并 + 跨核 N-split」候选仅当归约轴超驻留或输出维度 M≪56 需拆列并行时适用——本算子输出按 (token, h-chunk) 已天然跨 56 核划分（§5.5），归约轴在寄存器内闭合，无跨核归并需求，判据不匹配。

**R3 复杂度对比**（代表 workload ds-v3-t64：T=64, K=8, H=7168, fp16 → 输出元素 458,752，真 MAC = T·H·K = 3.67M；口径：FLOPs 列按 lane-op/elem 计（含 cast 与广播装载，乘加 = 2 lane-op；FMA 指令按 1–2 lane-op 双口径标注）；访存列 = GM 读+写（分母 = 契约 I/O，不可消除）；遍数 = 同一 GM 元素被读次数；缓冲 = 片上驻留峰值）：

| 算法候选 | FLOPs（lane-op/elem） | 访存量 (Bytes, GM) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|----------------------|-------------------|---------|-------------|---------------------|
| 0 基线（源，thread-per-elem） | 26（cast 1 + mul 1 + add 1 per slot ×8 + scale 1 + down-cast 1） | 读 7,340,032 + 2,048；写 917,504（**8.26MB**） | 1 | O(1)/thread（寄存器） | 1,792 blocks × 256 threads（源机）；**A5 不可直接映射**（§0.5 条目 4） |
| 1 **选定**：纵向 Vector-FMA + UB staging | **27–35**（54–70 向量 op/128e 组，s=1.0，s≠1.0 再 +2 vmuls；分项：每 slot 装载 3〔1 vld NORM + 2 vcvt 加宽〕+ FMA 2–4〔2 vmula 指令 / 1–2 lane-op 双口径〕+ 广播 1〔vld BRC〕，×8 slot + 每组 vdup 2 + 尾声 4〔2 vcvt 收窄 + 1 vor + 1 vsts NORM_B16〕——分项之和 = 54（指令口径）–70（FMA 双 lane-op 口径）；`CONST-valu-model` 无融合档，双口径披露） | 同上（8.26MB，**流量下界**：每元素恰一读一写） | 1 | UB ≤ 32,288B/块（harness 最坏，§4.5）；VRF 活跃 ≤ 8 vreg | **T×nh = 56–256 blocks**（1–5 波于 56 AIV）；零跨核同步（输出空间不相交） |
| 2 Cube 批式 GEMV（M=1 batch ×T） | 真 MAC 3.67M → **padded 16×（M=1→16 行 pad）× 2×（K=8→16 pad）= 118M MAC**（有效利用率 3.1%）+ ND2NZ 重排 op | 输入 ND2NZ 重排中转（GM→L1 staging 全量）+ L0C fp32 → UB → GM 输出（epilogue 额外 TH·4B 中转） | **2–3**（ND2NZ + GEMM + epilogue） | L1 staging [16, H] + L0C 累加器 | 28 AIC；每 token 独立 batch（B 矩阵逐 token 不同，无法合并单 GEMM）→ T 次小 GEMM 链 |
| 3 水平 lane=K 向量归约 | 有效 lane 8/64 = **12.5%**（×8 计算量浪费）+ 每元素跨步 gather（x[t,k,h] 沿 k 步长 H·2B，非连续）+ 每 64-lane 组 1 条 vcadd | 同候选 1 | 1 | 同候选 1 | 同候选 1（但 gather 装载抵消向量收益） |

**R4 硬件亲和性评估**（逐候选对照检查清单；常数引自 `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`，A5 冷启动期 status=assumption〔`CONST-arch-a5` 核数为 measured〕，容量按 90% 折扣）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 跨核结构 | 结论 |
|---------|-------------|---------|------------|---------|------------|---------|------|
| 0 基线（源形态） | A5 无 256-thread SIMT 域；逐元素标量链若直译到标量核则 3.67M×3 lane-op 全走标量核（51 lane-ops/cycle 的 VALU 完全闲置）——主导计算落错单元 | 0 UB（GM 直访在 SimdVF 被禁） | — | ✅ | — | — | ❌ 不可直接映射（§0.5 条目 4/6） |
| 1 **选定**：纵向 Vector-FMA | AIV VALU：vmula/vcvt/vld 全在 51 ops/cycle 档（`CONST-valu-model`）；无 MAC 密集段 → 不需 Cube；累加驻留 VRF（`CONST-vrf-capacity-a5`：活跃 ≤8 vreg ≈ 2KB ≪ 假设 256KB） | UB 32,288B ≤ 228,556B（248KB×90%，SimdVF 无 32KB 扣减，`CONST-capacity-a5`）= **14.1%**（harness 最坏 T=64） | vld NORM 128-lane：行基址 k·BH_pad·2B（BH_pad≥128 → ≥256B 整倍，32B 对齐 ✓）；组内偏移 g·128·2 = 256B 倍 ✓；BRC_B32 单元素 ✓；vsts NORM_B16 128 元素 = 256B、落点 g·128·2 = 256B 倍 ✓；T.copy 行宽 BH·2 ≥ 256B（宽行档；H=100 测试 200B 行——logsumexp_simd L1 (32,300) 600B 行实测通过先例） | ✅（T/K/H/BH/nh/ng 全工厂烘焙；BH \| H 无尾块 → 全部 extent 静态，规避「运行期钳制 extent 静默 OOB」陷阱） | 单遍直线（copy→VF→copy）；MTE2 两段互不依赖可由 AutoSchedule 重叠；无 T.Pipelined 需求（每块一次通过） | grid = T×nh 不相交输出空间 → 零跨核同步；AutoSchedule 自动 pipe 同步（§7） | ✅ **选定** |
| 2 Cube 批式 GEMV | Cube MAC 档 8192 ops/cycle/AIC（`CONST-cube-throughput-model`）——但 M=1 batch 的 16 行 pad + K=8→16 pad 使利用率 3.1%；且 B 矩阵逐 token 不同 → 无单 GEMM 合并面；**M=1 违反 M%16 静态断言**（`CONST-gemm-dim-constraints`：M、N 须 16 倍数——M=1 必 pad；K % TILE_K_SUB == 0 且 % C0=16：K=8 不满足，必 pad ×2） | L1 [16, H] 分形 staging + L0C（每 batch 独立） | C0 分形全约束命中（M-pad/K-pad + ND2NZ scatter 行 16 整除）；[T,K,H] K-中间布局的 ND2NZ 重排为非平凡变换 | ✅ | T 次 batch 链无法流水合并；epilogue L0C→UB→GM 每批一次 | 28 AIC | ❌ **定量淘汰**：padded MAC 32×（利用率 3.1%）+ ND2NZ 重排多 1 遍 + T 次独立 GEMM 链的固定开销 ≫ 候选 1 总量（候选 1 全量 12.4–16.1M lane-ops / 56 AIV ≈ 4.3–5.7K cyc vs 候选 2 仅 ND2NZ 62-cyc base × T=64 批 ≈ 4K cyc 起步 + MAC + epilogue） |
| 3 水平 lane=K | Vector 可表达（vld + vcadd）但 87.5% lane 浪费 + x 沿 k 跨步 H·2B 的 gather 装载（跨步 UB 访问代价未实测 + 非 32B 连续事务） | 同 1 | 64-lane 组需 K ≤ 64 掩尾 | ✅ | 同 1 | 同 1 | ❌ **定量淘汰**：有效吞吐 ÷8（12.5% lane 利用率）+ 跨步 gather；候选 1 连续 NORM 装载零浪费 |

**调研结论**: 选定算法族 = **候选 1「纵向 lanes=H 的 Vector-FMA 加权累加：UB 全块 staging + BRC_B32 权重广播 + 128-lane NORM 装载/偶奇 vcvt 加宽 + vmula 寄存器 FMA 累加（slot 升序）+ 偶奇收窄 + vor 位或重合并 + NORM_B16 全宽存储」**，关键依据：R3 表——GM 流量已达下界（8.26MB，每元素恰一读一写，全部候选相同），分歧全在**计算侧落位**：候选 1 是唯一把 3.67M MAC 放上 56×51 lane-ops/cycle VALU 且零 lane 浪费的结构；R4 表——候选 2 的 C0 分形约束（M%16 / K%C0）在 M=1、K=8 的本算子契约 shape 上直接触发 32× padding 浪费 + ND2NZ 重排额外遍；候选 3 的 12.5% lane 利用率 + 跨步 gather 双重劣化。与基线的结构差异一句话：把「每输出元素一个标量执行单元串行 K 步累加」换成「每 (token, 128 列 chunk) 一条 VRF 寄存器 FMA 链（lane=列，K 归约寄存器内闭合）+ UB 块 staging」。**源算法优化手段承接**（§0.4 → 选定算法）：#1 满并行/thread-per-element → 由 (token, h-chunk) block × 128-lane 向量域承接（T=1 时 nh=56 专项填补 AIV，§5.2）；#2 寄存器累加 → vmula 的 VRF dst（同意图同形态）；#3 fp32 累加 + 单次 narrowing cast → 逐点保留（cast 链不变）；#4 scaling 后置 → 保留 + OPT-1（s=1.0 编译期跳过，E1）；#5 div/mod 索引 → block 级 2 次（§0.5 条目 5）；#6 GM 直访 → bulk T.copy（§0.6 R2，流量不变）；#7 host 特化 → 原样保留。**调研范围**（"无更优替代"的反向确认）：`algorithm-candidates.md` 命中行 ALG-pooling（R1 行 6 同构化简已采纳）/ ALG-reduction（R2 结构判据核对本算子不匹配跨核归并场景）/ ALG-gemv（R1 行 2 Cube-vs-Vector 已定量评估淘汰）/ ALG-elementwise（R3/R4 口径参照：MTE 带宽主导提示——本算子复算为 lane-op 计算侧主导，见 D-2）/ ALG-transpose（§1.6.3 候选 4/5 布局重排评估）；`examples/ascend/`（logsumexp_simd SimdVF 归约重实现〔128-lane 装载/HMAX 同族优化先例〕、flash_attention/core.py〔BRC_B32 广播 × 向量乘加 accumulate_output 同构先例；L128–152 偶奇拆分 + vor 重合并真机先例——本设计尾声直接参照〕、deepgemm/epilogue.py〔vmula 用例 + UNPK_B16 偏置装载 + reinterpret/NORM_B16 全宽存储先例〕）；`testing/ascend/`（simdvf_arithmetic〔serial_vreg_accum = 本 kernel 内层骨架回归〕、test_tilelang_ascend_simdvf_cast〔双向 cast 惯例〕、test_tilelang_ascend_simd_vld_brc〔BRC 广播语义〕）；pattern-library constants.md（`CONST-valu-model` / `CONST-mte-bandwidth-model` / `CONST-capacity-a5` / `CONST-cube-throughput-model` / `CONST-gemm-dim-constraints` / `CONST-arch-a5` / `CONST-vrf-capacity-a5`）；结构分析（K 编译期常量 ≤8 → 寄存器内闭合；输出空间不相交 → 零跨核）；`tilelang/ascend/language/simd.py` 原语族机械 grep（FMA 族 / 广播族 / 装载族见 R1 表与查证纪律注记）。

**设计期估算（D-2）**: 估算(相对): **T=64（ds-v3-t64）≈ 4.3–8.8K cycles；主导项 = Vector lane-op 吞吐（计算侧 4.3–5.7K）与每块 MTE base 开销（串行链上界 8.1–8.8K）**（流量：8.26MB ÷ (100–115 B/cycle × 56 AIV) ≈ 1.3–1.5K cyc 下界〔`CONST-mte-bandwidth-model`，assumption〕；计算：458,752 elem × 27–35 lane-ops/elem ÷ (51 ops/cycle × 56 AIV) ≈ 4.3–5.7K cyc〔`CONST-valu-model`，assumption，lane-op 粒度口径；FMA 指令 1/2 lane-op 双口径——constants.md 无融合档〕；串行链上界：每块（BH=1792，14 组 × 67.8–87.8 cyc 计算 + MTE 链 ≈678 cyc〔2×MTE2 base 90 + MTE3 base 180 + 传输 318〕）≈ 1.6–1.8K cyc × ⌈256/56⌉=5 波 ≈ 8.1–8.8K；容量：32,288B ≤ 228,556B〔`CONST-capacity-a5`，assumption，90% 折扣〕14.1% 单遍驻留无 tiling 循环）。**对照 op Cube 段**（T=64：45.08 GFLOP ÷ (8192 ops/cycle/AIC × 28 AIC) ≈ 197K cyc，`CONST-cube-throughput-model` assumption；同口径承袭 `_indexed_expert_gemm`/`_route_stats` D-2）→ 本 kernel 占比 **≈ 2.2–4.5%**（T=64）；T=32 复算 ≈ 4.3–5.0K / Cube ≈ 98.4K cyc = **4.4–5.1%**；**T=1（decode 敏感 shape）**：BH=128 → 56 blocks 单波，每块 ≈ 68–88 计算 + MTE ≈ 380–470 cyc → **≈ 0.47K cyc**，对照 T=1 Cube 段（eval_roofline FLOPs = K·(6FH+6F+2H) ≈ 704.9 MFLOP ÷ 229,376 ops/cyc ≈ 3.07K cyc）占比 **≈ 15%**——该 shape 正是 §5.2 shape-aware 分块（T=1 → nh=56）的存在理由（整行单 block 时 ≈ 5K cyc，劣化 ≈10×）。协议见 `_shared/standards/hardware-cost-model.md` §2（相对量级比较 = max(流量项, 计算项, 容量项)，禁止编造频率换算绝对时延；A5 冷启动期 assumption 值只做相对比较）。
#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（纵向 Vector-FMA 链，经 §0.6 R1/R2 落地）。本算子公式为线性加权求和——无超越函数、无除法、无条件结构；等价优化集中于 cast 链特种化、装载形态（op 数削减）与 lane 映射闭合（装载偶奇拆分 ↔ 尾声偶奇重合并互为逆置换），全部在 §0.1 语义链内做恒等变形。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益量化 |
|---|--------|------|-----------|-----------|---------|
| 1 | **OPT-1**: scaling == 1.0 编译期特化（乘法跳过） | `out = cast_d(s · acc)`（s = 1.0 时仍执行 `acc · 1.0` 一次 fp32 乘） | s == 1.0 时：`out = cast_d(acc)`（host Python `if scaling != 1.0` 分支烘焙，两条特化各自静态） | IEEE-754 fp32 乘 1.0 对全体有限值/±inf 为恒等（精确无舍入）；NaN payload 规范化差异在容差对照下不可观察（assert_close equal_nan + 值级 0 差）；**E1 机器验证：28 用例（2 dtype × 7 shape × rand/corner）max_abs = 0 / 0 ulp / 0% 违反，EQUIV_PASS** | −2 向量 op/128e 组（vmuls×2：−3.7% 组 op）；harness 4 行 workload 全部 s=1.0（conversion_record §2）→ 热路径恒零开销；s=2.5 测试路径保留乘法（语义两全） |
| 2 | **OPT-2**: 128-lane NORM 装载 + 偶奇 vcvt 加宽 + vor 重合并尾声（替换 UNPK-64 单半装载） | 每 128 元素/slot（UNPK-64 形态）：`vld(x_ub[k, g·128+j·64], "UNPK_B16")` ×2 + `vcvt(part=0)` ×2 = 4 op；尾声 2×（`vcvt(part=0)` + `vsts(PK_B32)`）= 4 op | 每 128 元素/slot（NORM-128 形态）：`vld(x_ub[k, g·128])`（128-lane NORM，寄存器 lane j ↔ 列 g·128+j）+ `vcvt(part=0)`（**偶 lane {0,2,…,126}** → acc_even[m] ↔ 列 g·128+2m）+ `vcvt(part=1)`（**奇 lane {1,3,…,127}** → acc_odd[m] ↔ 列 g·128+2m+1）= 3 op；尾声改为 flash-core 式重合并：`vcvt(acc_even, dtype, part=0)`（值落偶 16-bit lane）+ `vcvt(acc_odd, dtype, part=1)`（值落奇 16-bit lane）+ `vor(T.reinterpret(·, "uint16x128") ×2)`（位或重合并——偶奇 lane 位置不相交，OR = 合并）+ `vsts(out_ub[g·128], T.reinterpret(merged, dtype+"x128"), dist="NORM_B16")` 单发存 128 连续元素 = 4 op | 两条 lane 索引链均为输出位置恒等置换：(装载) part=0/1 = PART_EVEN/PART_ODD **偶/奇 lane 交错拆分**（非连续半——四源佐证：`src/tl_templates/ascend/nd2nz_copy.h` L149–154〔加宽后须 `vintlv` 重交错〕、CANN `basic_block_vf_nd.h`〔PART_EVEN 奇偶注释 + FP32_TO_F16_EVEN/ODD + `Or<uint16_t>` 合并〕、`examples/ascend/flash_attention/core.py` L128–152〔vor 重合并真机〕、`tilelang/ascend/language/simd.py` vcvt docstring〔"even/odd 2-way splits"〕）；(尾声) 收窄 part=0/1 分置偶/奇 16-bit lane，vor 重合并为自然交错序 → merged[j] = 列 g·128+j 的结果——**装载拆分与尾声合并互为逆置换**。数值侧两种形态均为逐元素精确加宽（16→32 位无舍入）。**E3a 机器验证（独立 lane-mapping 模型）：58 用例（7 shape × 2 dtype × rand/corner × s∈{1.0, 2.5} + fp16 全 65536 位模式 + bf16 全位模式扫描，含 padded-lane 垃圾注入），显式 gather/interleave 建模整链索引映射，对照参考公式直算——位级恒等（max_ulp = 0 / bit_mismatch = 0 / 0% 违反），EQUIV_PASS；E3b 同款模型验证 UNPK-64 fallback 形态（58 用例位级恒等）；E3neg 负对照（v0 连续半尾声形态）置换检出 bit_mismatch 98.4%——证明 check 具备置换检出能力** | 装载链 op：4 → 3 op/128e/slot（−25%）；组总数（K=8, s=1.0 指令口径）**62 → 54**（−8 op/组 = −12.9%；FMA 双 lane-op 口径 78 → 70 = −10.3%）；尾声 op 数不变（4 ≡ 4：2 vcvt + 1 vor + 1 vsts 替换 2 vcvt + 2 vsts）；先例：flash core softmax P 矩阵 bf16 路径同款 vor 重合并（真机）、CANN weight_quant 反量化同范式 |
| 3 | **OPT-3**（微优化，Stage 4 调优项非主选形态）: 首槽乘替代零初始化 | `acc = vdup(0); for k in 0..K-1: vmula(acc, x_k, w_k)`（组首 2 条 vdup 初始化） | `acc = vmul(x_0, w_0); for k in 1..K-1: vmula(acc, x_k, w_k)`（首槽直接 vmul，免 vdup） | 0 + a 的 fp32 加法对全体值恒等除 ±0 符号位（(−0)+(+0) = +0 vs vmul 直通 −0）——值级差 0（\|0 − (−0)\| = 0），容差内恒等；**E2 机器验证：56 用例 max_abs = 0 / 0 ulp / 0% 违反，EQUIV_PASS** | −2 向量 op/128e 组（−3.7%）；**列为主选之外的 Stage 4 微调项**：主选保持 `vdup-init + all-vmula`（与 `serial_vreg_accum` 回归骨架逐点同形，首过实现风险最低） |

**FMA 舍入行为鲁棒性（E4，非优化项——容差边界论证）**：`vmula` 的硬件舍入（融合单舍入 FMA vs 乘加两舍入）为微架构未文档化行为——两种可能均已机器验证：E4f（融合：`acc = fl32(acc + x·w)`，乘积 fp64 精确后单舍入）与 E4s（两舍入 = 源链本身）对照源链与 fp64 参照，各 56 用例 max ≤ 1 ulp fp16 / 0% 违反 @1e-2 / nonfinite mask 全等，**双双 EQUIV_PASS** → 设计对硬件舍入行为不敏感，§8.2 分档容差（fp16/bf16 1e-2）下两条路径均合法。

**等价性机器验证（D-1，`examples/indexed_expert_mlp/_weighted_reduce/verify_equiv.py`，torch CPU + fp64 参照，revision 1 重建后 2026-10-29 全量执行，运行留档 `history_version/verify_equiv_r1_run.log`）**——7 组 shape（harness 代表 (2,8,7168)〔T 缩样保 H 全量〕+ 测试全集 (4,2,128)/(17,2,128)/(4,2,100)/(8,4,512)/(2,2,128)/(1,4,96)）× 2 dtype（fp16/bf16）×「rand / corner」两类（±inf/NaN/±0/subnormal/fp16·bf16 极值 + 权重含 0/负/大数）× scaling ∈ {1.0, 2.5}；E3a/E3b 另含 fp16/bf16 全 65536 位模式扫描用例，lane-mapping 模型对 UB padded lane 注入垃圾值（真实 kernel 形态）：

| 验证项 | 内容 | 最大差（fp16 值/ulp） | 违反率 @1e-2 | 结论 |
|---|---|---|---|---|
| E1 OPT-1 | s=1.0 跳乘 vs 强制乘 1.0（28 用例） | 0 / 0 ulp | 0% | EQUIV_PASS |
| E2 OPT-3 | 首槽乘 + FMA 余 vs 零初始化 + 全 FMA（56 用例） | 0 / 0 ulp | 0% | EQUIV_PASS |
| E3a OPT-2（采纳形态） | **独立 lane-mapping 模型**：NORM-128 装载 → 偶奇 vcvt 加宽 → FMA 累加 → 偶奇收窄 → vor 重合并 → NORM_B16 存储 → extent-BH 写回（显式 gather/interleave 索引张量建模）vs 参考公式直算（[T,K,H] 张量式，结构独立路径）（58 用例） | 0 / 0 ulp（**位级恒等**，bit_mismatch = 0） | 0% | EQUIV_PASS |
| E3b（fallback 形态） | 同款 lane-mapping 模型：UNPK-64 装载 + `vcvt(part=0)` 加宽 + `vcvt(part=0)` 收窄 + `vsts(PK_B32)` 存储（cast 测试 L47–53/L77–80 两处保序真机形态的模型化）vs 参考公式直算（58 用例） | 0 / 0 ulp（位级恒等） | 0% | EQUIV_PASS |
| E3neg（负对照） | v0 错误尾声形态（硬件偶奇装载拆分 + v0 连续半存储、缺逆交错）vs 参考公式直算（2 用例：fp16 (4,2,128) / bf16 (8,4,512)） | bit_mismatch **98.4%**（max 38538 ulp） | 91.0% | **PERM_DETECTED**（负对照按预期置换检出——E3 检出能力证明；此即 v0 设计若直上真机将产生的置换错误） |
| E4s | 两舍入 FMA 链 vs 源链（56 用例） | 0 / 0 ulp（同序同算） | 0% | EQUIV_PASS |
| E4f | 融合单舍入 FMA 链 vs 源链（56 用例） | 6.1e-05 / 1 ulp（subnormal 角点量级；位差元素 0.03%） | 0% | EQUIV_PASS |
| E4s/E4f-vs64 | 两种 FMA 链 vs fp64 参照（112 用例） | ≤1 ulp（2.4e-04） | 0% | EQUIV_PASS |
| E5 | 全采纳候选链（fused + 全 OPT）vs fp64 参照（56 用例） | ≤1 ulp | 0% | EQUIV_PASS |
| E6 | 尾区 padding 管控：每 token GM 写集恰 [0,H) 单次覆盖、无 OOB、padded lane [BH,BH_pad) 不落 GM（7 shape × BH 启发式实际取值） | — | 0% | EQUIV_PASS |

**优化结论**: 采纳 2 项进主选（OPT-1 / OPT-2，均 EQUIV_PASS；OPT-3 已验证、列为 Stage 4 微调项）；E3a/E3b 为采纳与 fallback 形态的 lane-mapping 结构验证（位置恒等 + 位级恒等），E3neg 为负对照（置换检出能力证明），E4 为容差边界鲁棒性论证，E5/E6 为全链与布局 containment 验证。优化后「公式」（供 §3.1 拆解，偶奇 lane 映射显式化）：

$$
\text{out\_ub}^{(g)} = \text{vsts}_{\text{NORM\_B16}}\Big( \text{vor}\big( \text{vcvt}_{\text{part}=0}\big(\text{acc}_{\text{even}} \cdot [s]\big),\ \text{vcvt}_{\text{part}=1}\big(\text{acc}_{\text{odd}} \cdot [s]\big) \big) \Big), \qquad \text{acc}_{\text{even/odd}} = \text{vmula}^{*K\uparrow}\big( \text{vcvt}_{\text{part}=0/1}\big(\text{vld}(\text{x\_ub}[k, g{\cdot}128])\big),\ \text{vld}(\text{w\_ub}[k], \text{BRC}) \big)
$$

（[s] 仅 s≠1.0；vor 前后各一次 `T.reinterpret(·, "uint16x128")` / `T.reinterpret(·, dtype+"x128")` 位型转换，不计入数值语义。）

无进一步优化空间的依据：公式为线性（无超越函数/除法/公共子表达式——权重与 scaling 均已最小次数化）；候选 4/5（R1 表）为语义层拒绝；装载/存储 op 数已按 E3 形态取下限（装载链 3 op/128e/slot + 尾声 4 op/组〔2 vcvt + 1 vor + 1 vsts〕——再减需 cast-FMA 融合原语，机械 grep 佐证不存在，见 R1 查证纪律注记）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点选定算法实现方案中的全部循环与标量计算点（§1.4 四步），逐项判定能否用向量操作等价替代；替代方案 API 均有 `examples/ascend/` / `testing/ascend/` / `tilelang/ascend/language/` 源码佐证。

| # | 计算点 | 原实现形态（源） | 向量替代方案（选定算法内） | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|----------------|---------------------------|---------|------------------------------|
| 1 | **K 步加权累加主体**（源 §0.3 步骤 5：每元素 T.serial(top_k) 标量乘加链） | 每元素 3K 标量 op | SimdVF 向量 FMA：`for k in range(K)` 编译期展开 × `vmula(acc, vcvt(vld(...)), vld(w_ub[k], "BRC_B32"))`——K 归约在 VRF 寄存器闭合，lane 沿 H 纵向 | ✅（核心替代） | —（佐证：`testing/ascend/language/simdvf/test_simdvf_arithmetic.py L85–111` serial_vreg_accum = 同构骨架回归：T.serial 循环 + vld + vmula 寄存器累加；`examples/ascend/flash_attention/core.py L188–196` accumulate_output = BRC 广播 × 向量乘加同构先例；`examples/ascend/deepgemm/kernels/epilogue.py L181–182` vmula 用例） |
| 2 | 逐元素 up-cast（cast 链步骤 1） | 每 `T.Cast("float32", x)` 标量 | `vcvt(x_h, "float32", part=0/1)` 偶/奇 lane 向量加宽（OPT-2：part=0 = 偶 lane {0,2,…,126}、part=1 = 奇 lane {1,3,…,127}——位级恒等经 E3a lane-mapping 验证） | ✅ | —（佐证：`testing/ascend/language/test_tilelang_ascend_simdvf_cast.py` half→fp32 UNPK+part=0 保序用例 L77–80；`src/tl_templates/ascend/nd2nz_copy.h` L149–154 PART_EVEN/PART_ODD 加宽 + vintlv 重交错模板级佐证；`examples/ascend/deepgemm/kernels/epilogue.py L38` 偏置装载同构；bf16 路径 logsumexp_simd L213–217 双 dtype 同支） |
| 3 | 逐元素 down-cast + scaling + 打包存储（cast 链步骤 3–4） | 标量乘 + `T.Cast(dtype, …)` | `vmuls(acc, s)`（仅 s≠1.0，OPT-1）+ `vcvt(acc_even/acc_odd, dtype, part=0/1)` 偶奇分置 + `vor` 位或重合并（逆交错）+ `vsts(..., dist="NORM_B16")` 全宽单发存储（128 连续元素） | ✅ | —（佐证：`examples/ascend/flash_attention/core.py L139–152` bf16 收窄 part=0/1 + vor 重合并真机先例；`examples/ascend/deepgemm/kernels/epilogue.py L116` reinterpret + NORM_B16 全宽存储真机先例；`vmuls` 编译期标量乘 flash core.py L110 用例；cast 测试 L47–53 narrowing+PK_B32 保序形态〔UNPK-64 fallback 用〕） |
| 4 | 权重逐元素读取/广播（源每 thread 2 次/lane-group GM 标量读） | 标量 GM 读 | `T.copy` 整行 staging + `vld(w_ub[k], "BRC_B32")` 单元素广播装载（每 slot 每 128e 组 1 op） | ✅ | —（佐证：`examples/ascend/flash_attention/core.py L126–127, L192, L303, L314` BRC_B32 广播装载 4 处用例；`testing/ascend/language/test_tilelang_ascend_simd_vld_brc.py` BRC 语义回归） |
| 5 | GM↔核内数据搬运（源标量直读直写） | per-thread 标量 load/store | `T.copy` 三段 bulk（2-D [K,BH] / 1-D [K] / 1-D [BH]，全静态 extent） | ✅（DMA 天然批量） | —（佐证：§0.6 R2 三处先例引证） |
| 6 | block 索引分解（`token = bid // nh`、`h0 = (bid % nh) · BH`） | —（选定算法引入，block 级标量） | 无需替代（每 block 2 次 div/mod + 2 次乘加，非逐元素热点：T·nh ≤ 256 次 vs 源 T·H = 458,752 次） | ❌（保留标量） | **block 级索引 / 任务映射计算**（可接受理由类别：每 block 少量标量、非逐元素热点——源形态同类计算量放大 1,792×） |
| 7 | host 侧 BH / nh / BH_pad / ng 启发式计算 | —（host Python） | 无需替代（不在 kernel 内；编译期烘焙为张量形状与循环边界常量） | ❌（保留 host） | **host 侧 shape / tiling 元数据计算**（不在 kernel 内，工厂闭包烘焙） |
| 8 | 组循环（`for g in T.serial(ng)`） | —（选定算法引入） | 循环体全向量 op（#1–#4）；循环变量 g 仅作仿射寻址（`g·128`）——`vld(a_ub[i·LANES])` 同构先例（serial_vreg_accum L103） | ✅（循环体向量化；循环本身为向量批处理骨架，非标量计算点） | — |
| 9 | slot 循环（`for k in range(K)`） | 源 `T.serial(top_k)`（运行时循环，编译期常量边界） | Python `range(K)` 编译期展开（K ≤ 8 代码膨胀可控）——**消除运行时循环结构**（无循环控制开销、寻址全编译期常量折叠） | ✅（结构消除） | — |

**向量化结论**: 计算点 9 处，向量化/DMA 化 7 处（累加主体 / 双向 cast / 权重广播 / 三段搬运 / 组循环体 / slot 循环结构消除），标量保留 2 处——#6 block 级索引（每 block ≤4 次标量算术，源形态同功能计算量 1,792×于本设计）、#7 host tiling 元数据（不在 kernel 内）。**源算法的全部逐元素标量计算（乘加链 / cast 链 / 索引 div-mod / GM 直访）100% 被向量或 DMA 形态替代**，无任何逐元素标量循环残留；替代 API 均有 `testing/ascend/` 回归或 `examples/ascend/` 用例佐证（逐条见上表）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> 本算子 I/O 为 [T, K, H] 三维（归约轴 K 在中间）+ [T, K] 权重 → 具有 ≥2 个非 batch 维度，按规约类条目枚举正交决策（水平归约 lane↔归约轴 vs 垂直扫描 lane↔独立输出列）与核内布局候选。**I/O layout 是契约（[T,K,H] 行主连续，不可更改）；核内布局与 lane 映射是设计变量。**

**轴质量评分维度**（对主选布局）：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | vld NORM 128-lane 组：BH_pad = align_up(BH,128) 保证组界整除；vsts NORM_B16 128 元素/发 | harness 全部 shape BH ∈ {128, 512, 1024, 1792}（128 倍数，0 padding）；测试 off-grid H=100 → BH=100，BH_pad=128（1 组，28 padded lane 由 E6 containment 吸收）✓ |
| 尾 lane 浪费率 | (BH_pad−BH)/BH_pad | harness 0%；最坏测试 shape 28/128 = 21.9%（H=100，µs 级 kernel 可忽略） |
| 累加链形态 | 串行累加维 = K（编译期常量 ≤8），全在 VRF 寄存器（无跨步、无访存）；向量维 = H 连续（stride 1 于 UB 行内） | 零跨步累加链（对比候选 3 的跨步 gather）✓ |
| repack 代价 | 核内布局变化的重排开销 | **无 repack**：x_ub 保持 [K, BH_pad]（GM [T,K,H] 的原生块布局）；K-中间归约轴由 2-D T.copy 的行跳步（row stride H）原生处理，无转置（ALG-transpose 行布局重排候选全部弃选，见候选矩阵 #4/#5） |
| UB 容量影响 | 驻留缓冲 | 32,288B ≤ 228,556B（14.1%，§4.5） |

**候选矩阵**（规约类正交决策 + 布局重排评估）：

| # | 布局方案 | 向量化轴 / lane 映射 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|--------------------|------------|--------------|---------|
| 1 | **主选**：纵向 lanes=H（lane ↔ 输出列），K 归约寄存器闭合；x_ub 原生 [K, BH_pad] | H 轴（连续 NORM 装载 128 lane/组）；K = VRF FMA 链 | 无 | §1.6.0 D-2：T=64 ≈ 4.3–8.8K cyc；零 lane 浪费（harness）；流量下界 | ✅ |
| 2 | 水平 lanes=K（lane ↔ slot，64-lane 组内装 K 个 x[t,k,h]） | K 轴（`vcadd` 跨 lane 归约） | x 需沿 k 跨步 gather（stride H·2B 非 32B 连续） | **12.5% lane 利用率（K=8/64）→ 有效吞吐 ÷8** + 跨步 gather 装载（跨步 UB 访问代价未实测 + 权重装载反而连续）；净劣化 ≥8×（量化在 R3/R4 行 3） | ❌ 定量淘汰（lane 浪费主导，非未实证常数区间——无需实验裁决） |
| 3 | Cube 批式 GEMV（归约轴进 Cube K 维） | —（fractal 路径） | [T,K,H] → ND2NZ 分形重排（K-中间布局非平凡变换） | M=1 → 16 行 pad（M%16 断言）+ K=8 → 16 pad（C0）：利用率 3.1%（32× padded MAC）；+1 遍 ND2NZ 重排 + T 次 batch 链 | ❌ 定量淘汰（R1 行 2 / R3 / R4；`CONST-gemm-dim-constraints` 约束直接命中） |
| 4 | host permute 输入 [T,K,H] → [T,H,K]（归约轴移至最内，lane 连续 K） | 归约轴连续化后水平归约 | host `permute/transpose`（全量 GM 重排：读 7.34MB + 写 7.34MB） | **+14.7MB GM 流量（+178%）** + host kernel 往返；ALG-transpose 行：host permute 百 µs 级通常净亏；且归约轴连续化后仍需跨 lane 归约（#2 的代价不变） | ❌ 流量主导淘汰（2× 读 + 1× 写 vs 下界 1 读 1 写——无需实测即可判死） |
| 5 | 核内重排：UB 内 [K, BH] → [BH, K] 转置（vintlv/vdintlv 交错链）后水平归约 | 同 #2 但 repack 进核 | `T.simd.vintlv/vdintlv` 交错链（`tilelang/ascend/language/simd.py` 存在性已核——ALG-transpose 行佐证） | repack op 数 ≈ K×BH/64 × ~3 op（≥ 24 op/64e）**超过其欲优化的归约收益**（K 步 FMA 仅 2–4 op/lane-elem）；结构净亏 | ❌ 复杂度/收益倒挂（repack 链 op 数 > 归约节省，量化在左） |
| 6 | GU 组展开（广播摊销：每 slot 每 GU 组 1 次广播装载 → 1 op/GU·128e） | 同主选，寄存器数组 [GU×2] acc | 无 | 组 op 数 54 → 48–50（GU=2–4：−4~−6 op/组，7.4–11.1%）；代码 ×GU 膨胀 + acc 寄存器数组索引 | ❌ 作主选（收益 7–11% 不抵实现复杂度）；✅ 记入 Stage 4 微调清单（与 OPT-3 / vaxpy 标量变体同档） |

> 迁移任务核对：GPU 源码的并行轴（256 threads 展开 T·H 元素空间）是**输入而非结论**——已作为基线候选 0 在 §1.6.0 同台评估后被替换（thread 轴 ≠ NPU Vector 轴；一维 grid 保留但映射对象从「元素」换「(token, h-chunk) 块 + 128-lane 向量」）。

**布局决策结论**: 选定方案 #1：核内布局 = x_ub 原生 [K, BH_pad]（GM 块布局直拷，零 repack）+ w_ub [K] + out_ub [BH_pad]；lane 映射 = H 纵向（128-lane NORM 装载 + 偶奇 vcvt 拆分/重合并 + NORM_B16 全宽存储）；K 归约 = VRF 寄存器 FMA 链（slot 升序）。弃选方案全部给出量化理由（#2 lane 利用率 12.5%、#3 利用率 3.1% + 分形约束命中、#4 流量 +178%、#5 repack 倒挂、#6 收益 7–11% 复杂度不抵）——判定裕度均不在未实证常数区间内（主导项是整数量化关系：lane 数、padding 倍数、流量倍数），**无需实验裁决三件套**；结构内微调项（GU 展开 / OPT-3 / vaxpy 标量乘加变体〔`vaxpy(acc, x, w_scalar)` 免广播装载，组 op 54 → 46（−8 op = 免 8×BRC）；运行时 UB 标量作 simd 标量操作数的先例为 transform_sf.py L201–203 的 vcmps——同构但非同形，列 L0 首验项〕）统一记入 §5.4 / §9.2 的 Stage 4 调优清单，属参数级调优而非结构候选。该结论同步落入 §3.3 伪代码与 §6 循环结构（累加链内层向量维 = H 轴 128-lane 组）。

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: **simd**（`T.SimdVF()`，无线程概念）（auto_schedule: **yes**）

> 整体算子 `IndexedExpertMLPFwdOp` 为 mixed（GEMM 主导 kernel 走 Cube 路径，由 `_indexed_expert_gemm` 承载）；本函数 `_weighted_reduce` 为纯向量加权归约 kernel，按用户确认的编程模式指令取 simd（调度 prompt：「纯向量走 SimdVF；无 GEMM，不用 Cube/MixedKernel」）。kernel 模块 docstring 必须声明 `programming_mode: simd` 与 `auto_schedule: yes`（§3.3 伪代码头部）。

### 2.2 选型理由

1. **算子特征**：无 matmul（加权求和的 MAC 强度为 2·T·H·K ≈ 7.3M，与 Cube 8192 ops/cycle/AIC 的分形路径不匹配——R1 行 2 定量淘汰）；全部计算为逐元素乘加 + cast + 归约（K ≤ 8 寄存器内闭合）→ 纯 Vector 落位 AIV。
2. **向量能力需求**：核心链（vld/vcvt/vmula/vsts）全部为显式 2048-bit 向量原语——SimdVF 的原生形态；无 T.Parallel / 线程域需求（对比 SimtVF：线程域引入 32KB UB 扣减〔248KB→216KB，`CONST-capacity-a5`〕且 `T.Parallel` 逐线程标量化机制〔CG-2026-0002 风险背景〕对本算子无益——本算子的并行完全由 block 级 (token, h-chunk) 划分 + 128-lane 向量承载）。
3. **同代 wrapper 先例**：`examples/TileOPs/src/tileops/_kernels/reduction/logsumexp_simd/logsumexp_simd_kernel/perf_opt/_logsumexp_kernel_single.py`（TileOPs SimdVF 归约重实现：`@tilelang.jit(target="ascend")` 工厂面 + `programming_mode: simd` docstring 惯例 + 56 AIV 核数注释 + padded 分配/部分范围拷出形态——本设计 §3.3/§4.3 直接沿用该形态族）；同批迁移 `_route_stats`（simd + auto_schedule: yes，已过 Stage 2/3）。
4. **结构与重设计方向一致**：SimdVF 块禁止直接访问 GM（`testing/ascend/analysis/test_ascend_vf_checker.py`），强制 UB staging——正是 §0.6 R2 bulk staging 方案的结构前提。
5. **与上游一致性**：Op 的 4 个 kernel 中 3 个非 GEMM 的（route_stats / weighted_reduce）均取 simd，GEMM 取 mixed（`_indexed_expert_gemm`）——调用面与 wrapper 装饰完全同族。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| VF 入口 | `with T.SimdVF():` 包住全部计算段（g 循环 + k 展开 + 尾声）；三段 `T.copy` 在 VF 域外（DMA 任务；SimdVF 块内禁 GM） |
| 内存分配 | `T.alloc_shared`（UB）：x_ub / w_ub / out_ub（§4.3）；无 L1/L0 分配（无 Cube 段）；累加器 `T.simd.alloc_var`（VRF local.var，不占 UB 预算，`CONST-vrf-capacity-a5`） |
| 同步 | auto_schedule: yes（AutoSchedule 自动插 MTE2→V→MTE3 的 pipe 同步，§7） |
| 循环原语 | `T.serial`（组循环，静态边界）+ Python `range` 编译期展开（slot 循环）；无 T.Parallel（并行由 grid × 向量宽度承载） |

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 优化后的向量链式（OPT-1/OPT-2 已并入；T2 = [T,K,H]，W = [T,K]）：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_ub[k, j] = expert_output[token, k, h0+j]`，k ∈ [0,K)，j ∈ [0,BH)；`w_ub[k] = topk_weights[token, k]` | GM→UB 搬入（2-D 块 + 1-D 行，静态 extent） |
| 2a | `x_h = vld(x_ub[k, g·128])`：寄存器 lane j ↔ 列 g·128+j（128-lane NORM 装载）；`x_even = f32(x_h 的偶 lane)`——x_even[m] = x[t, k, g·128+2m]；`x_odd = f32(x_h 的奇 lane)`——x_odd[m] = x[t, k, g·128+2m+1] | NORM 128-lane 装载 + `vcvt(part=0/1)` **偶/奇 lane 加宽拆分**（OPT-2；PART_EVEN/PART_ODD，四源佐证见 §1.6.1） |
| 2b | `w_vec = broadcast(w_ub[k])`（BRC_B32） | 权重广播（每 slot） |
| 2c | `acc_even += x_even · w_vec`；`acc_odd += x_odd · w_vec`（slot k 升序，vmula FMA；acc 驻留 VRF；acc_even[m] ↔ 列 g·128+2m、acc_odd[m] ↔ 列 g·128+2m+1） | 加权累加（§0.1 cast 链步骤 1–2） |
| 3 | `merged[2m] = cast_dtype(acc_even[m] · [s])`；`merged[2m+1] = cast_dtype(acc_odd[m] · [s])`（[s] 仅 s≠1.0）——经 `vcvt(part=0/1)` 偶奇分置 + `vor` 位或重合并实现；`out_ub[g·128+j] = merged[j]`（`vsts(dist="NORM_B16")` 单发 128 连续元素，逆交错显式化） | scaling + narrowing cast + **偶奇重合并存储**（OPT-1/OPT-2；flash-core L128–152 先例） |
| 4 | `output[token, h0+j] = out_ub[j]`，j ∈ [0,BH) | UB→GM 静态 extent 搬出（padded lane 不落 GM，E6） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1a | GM→UB 2-D 块 | `T.copy(expert_output[token, 0:K, h0:h0+BH], x_ub[0:K, 0:BH])` | 3-D GM 标量行 + 双切片（extent K×BH 编译期静态；offset 运行时） | simd（VF 域外 DMA） |
| 1b | GM→UB 1-D 行 | `T.copy(topk_weights[token, 0:K], w_ub[0:K])` | K fp32（≤32B） | simd（VF 域外 DMA） |
| 2a | 128-lane 装载 + 偶奇加宽 | `T.simd.vld(x_ub[k, g*128])` + `T.simd.vcvt(x_h, "float32", part=0)` + `T.simd.vcvt(x_h, "float32", part=1)` | fp16/bf16 128 lane → 2×fp32×64 lane（part=0 = 偶 lane {0,2,…,126}、part=1 = 奇 lane {1,3,…,127}——PART_EVEN/PART_ODD） | simd（VF 域内） |
| 2b | 权重广播 | `T.simd.vld(w_ub[k], "BRC_B32")` | 单 fp32 元素 → 64 lane 广播 | simd（VF 域内） |
| 2c | FMA 累加 | `T.simd.vmula(acc_even, x_even, w_vec)` / `T.simd.vmula(acc_odd, x_odd, w_vec)` | dst = dst + src0·src1（local.var 寄存器；acc_even[m] ↔ 列 g·128+2m、acc_odd[m] ↔ 列 g·128+2m+1） | simd（VF 域内） |
| 累加器声明/初始化 | acc_even/acc_odd | `T.simd.alloc_var("float32")` + `T.simd.vdup(T.float32(0.0), "float32")` | 每 128e 组偶/奇列双累加器（组循环内重初始化） | simd（VF 域内） |
| 3a | scaling（条件） | `T.simd.vmuls(acc, T.float32(scaling))`（host Python `if scaling != 1.0` 烘焙） | 编译期标量乘 | simd（VF 域内） |
| 3b | narrowing 偶奇分置 + vor 重合并 + 存储 | `T.simd.vcvt(acc_even, dtype, part=0)` + `T.simd.vcvt(acc_odd, dtype, part=1)` + `T.simd.vor(T.reinterpret(even16, "uint16x128"), T.reinterpret(odd16, "uint16x128"))` + `T.simd.vsts(out_ub[g*128], T.reinterpret(merged, dtype+"x128"), dist="NORM_B16")` | f32×64 偶奇分置 16-bit lane → vor 位或重合并（逆交错）→ 128 连续元素全宽单发存 | simd（VF 域内） |
| 4 | UB→GM | `T.copy(out_ub[0:BH], output[token, h0:h0+BH])` | 1-D 静态 extent | simd（VF 域外 DMA） |
| — | kernel 骨架 | `with T.Kernel(T * nh) as bid:`（一维、无 threads） | num_blocks = T×nh | — |
| — | JIT 装饰 | `@tilelang.jit(target="ascend", out_idx=[])` | 原位写、无返回值（源同款 out_idx=[]） | — |

### 3.3 计算伪代码

```python
"""_weighted_reduce — IndexedExpertMLPFwdOp 加权归约 kernel（Ascend A5 重实现）。

programming_mode: simd
auto_schedule: yes
"""
import functools

import tilelang
import tilelang.ascend.language as T


def _pick_nh(num_tokens: int, hidden_size: int, *, target_blocks: int = 224,
             min_cols: int = 128) -> int:
    """Host compile-time chunk-count heuristic (DESIGN 5.2): pick a divisor n
    of H (n <= H // 128 so every chunk keeps >= 128 columns) minimizing
    |T*n - target| with ties toward larger n; fallback n = 1 (whole row)."""
    cands = [n for n in range(1, hidden_size + 1)
             if hidden_size % n == 0 and n <= max(1, hidden_size // min_cols)]
    if not cands:
        return 1
    return min(cands, key=lambda n: (abs(num_tokens * n - target_blocks), -n))


@functools.lru_cache(maxsize=32)
def _weighted_reduce(num_tokens: int, top_k: int, hidden_size: int,
                     dtype: str, scaling: float):
    """Build the token-major weighted-reduce kernel (fp32 slot-ascending FMA)."""
    nh = _pick_nh(num_tokens, hidden_size)          # h-chunks per token
    bh = hidden_size // nh                          # bh | H -> no tail block
    bh_pad = ((bh + 127) // 128) * 128              # 128-lane group padding
    ng = bh_pad // 128                              # groups per chunk

    @tilelang.jit(target="ascend", out_idx=[])
    def _kernel():
        @T.prim_func
        def main(
            expert_output: T.Tensor((num_tokens, top_k, hidden_size), dtype),
            topk_weights: T.Tensor((num_tokens, top_k), "float32"),
            output: T.Tensor((num_tokens, hidden_size), dtype),
        ):
            with T.Kernel(num_tokens * nh) as bid:
                token = bid // nh                    # block-level scalar (DESIGN 1.6.2 #6)
                h0 = (bid % nh) * bh
                x_ub = T.alloc_shared((top_k, bh_pad), dtype)
                w_ub = T.alloc_shared((top_k,), "float32")
                out_ub = T.alloc_shared((bh_pad,), dtype)
                # ---- staging: static extents only (DESIGN 0.6 R2) ----
                T.copy(expert_output[token, 0:top_k, h0:h0 + bh], x_ub[0:top_k, 0:bh])
                T.copy(topk_weights[token, 0:top_k], w_ub[0:top_k])
                # ---- vector reduce: lanes = H columns, K in registers ----
                with T.SimdVF():
                    acc_even = T.simd.alloc_var("float32")  # VRF, not UB
                    acc_odd = T.simd.alloc_var("float32")
                    for g in T.serial(ng):
                        acc_even = T.simd.vdup(T.float32(0.0), "float32")
                        acc_odd = T.simd.vdup(T.float32(0.0), "float32")
                        for k in range(top_k):             # compile-time unroll
                            w_vec = T.simd.vld(w_ub[k], "BRC_B32")     # broadcast
                            x_h = T.simd.vld(x_ub[k, g * 128])         # 128 lanes,
                            #   register lane j <-> column g*128 + j
                            x_even = T.simd.vcvt(x_h, "float32", part=0)  # OPT-2:
                            #   PART_EVEN -> even lanes {0,2,...,126}
                            #   acc_even[m] <-> column g*128 + 2*m
                            x_odd = T.simd.vcvt(x_h, "float32", part=1)
                            #   PART_ODD  -> odd lanes {1,3,...,127}
                            #   acc_odd[m]  <-> column g*128 + 2*m + 1
                            T.simd.vmula(acc_even, x_even, w_vec)      # slot ascending
                            T.simd.vmula(acc_odd, x_odd, w_vec)
                        if scaling != 1.0:                    # OPT-1 specialization
                            acc_even = T.simd.vmuls(acc_even, T.float32(scaling))
                            acc_odd = T.simd.vmuls(acc_odd, T.float32(scaling))
                        # epilogue: even/odd narrowing placement + bitwise
                        # re-merge (inverse of the load split; flash core
                        # L128-152 precedent) + one full-width 128-lane store
                        even16 = T.simd.vcvt(acc_even, dtype, part=0)  # even 16b lanes
                        odd16 = T.simd.vcvt(acc_odd, dtype, part=1)   # odd 16b lanes
                        merged = T.simd.vor(
                            T.reinterpret(even16, "uint16x128"),
                            T.reinterpret(odd16, "uint16x128"),
                        )  # disjoint lane positions -> OR == merge
                        T.simd.vsts(out_ub[g * 128],
                                    T.reinterpret(merged, dtype + "x128"),
                                    dist="NORM_B16")          # 128 contiguous elements
                # ---- writeback: extent bh (padded lanes stay in UB, E6) ----
                T.copy(out_ub[0:bh], output[token, h0:h0 + bh])

        return main

    return _kernel
```

> 参数名与顺序（expert_output, topk_weights, output）及工厂签名与源逐点一致；`@T.prim_func` dtype 采用第二位置参数形式（与 `examples/ascend/` 一致，源即此形态）。wrapper 调用面 `_weighted_reduce(T, K, H, dtype, scaling)()`（config 为空 dict，threads 已按 K3/K9 移除）。`if scaling != 1.0` 为 **host Python 编译期分支**（工厂闭包内 trace 时求值——两条特化各自静态，规避「条件绑定标量跨帧」陷阱〔`_indexed_expert_gemm` Stage 3 教训：运行时条件下的 sel 绑定 raise Immutable variable〕；此处条件是 Python float 比较，天然编译期）。

### 3.4 API 可行性确认

| API | 用途 | 来源确认 | 验证状态 |
|-----|------|----------|---------|
| `T.Kernel(T * nh)` | 一维 grid launch（无 threads） | tilelang-ascend skill §2.1；logsumexp_simd（NUM_BLOCKS 形态） | 已验证 |
| `T.copy` 3-D 标量行 + 双切片（`expert_output[token, 0:K, h0:h0+bh]`） | 2-D [K, BH] 块 GM→UB | `examples/indexed_expert_mlp/_indexed_expert_gemm/_indexed_expert_gemm.py L288–292`（`weights[expert, nb*block_n:…, ko*block_k:…]` 同形，Stage 3 实测通过） | 已验证（同族先例） |
| `T.copy` 1-D 区（`topk_weights[token, 0:K]` / `out_ub[0:bh]`） | 行 staging / 部分范围搬出 | logsumexp_simd `T.copy(lse_ub[0:SUB], y[t*SUB:(t+1)*SUB])`（L279） | 已验证 |
| `T.SimdVF()` | 向量域 | tilelang-ascend skill §2.3；logsumexp_simd / flash core / deepgemm 全族 | 已验证 |
| `T.simd.vld`（NORM 128-lane 16-bit） | 输入装载 | `examples/ascend/deepgemm/kernels/epilogue.py L38`（偏置 UNPK 同族）+ logsumexp_simd HMAX（`vld(x_ub[r, 0])` NORM 128-lane 实测胜出形态，L175–177）；`tilelang/ascend/language/simd.py L324` | 已验证 |
| `T.simd.vcvt(src, "float32", part=0/1)` | 16→32 位偶/奇 lane 加宽（PART_EVEN=0 / PART_ODD=1） | logsumexp_simd L213–217（fp16/bf16 同支）+ L186–192（bf16 双 part 加宽）；`testing/ascend/language/test_tilelang_ascend_simdvf_cast.py`（half→fp32 UNPK+part=0 保序用例 L77–80，真机 atol=0）；偶奇拆分语义佐证：`src/tl_templates/ascend/nd2nz_copy.h` L149–154（PART_EVEN/ODD 加宽 + vintlv 重交错）+ `tilelang/ascend/language/simd.py` vcvt docstring（"even/odd 2-way splits"） | 已验证（保序形态真机 + 偶奇拆分语义模板级佐证；E3a lane-mapping 模型位级验证） |
| `T.simd.vcvt(acc, dtype, part=0/1)` | f32→fp16/bf16 narrowing（part=0 = 偶 16-bit lane、part=1 = 奇 lane 分置） | `examples/ascend/flash_attention/core.py L139–142`（bf16 part=0/1 收窄真机）；logsumexp_simd L265（`vcvt(out, dtype)` 双 dtype）；cast 测试 f32→f16（L47–53） | 已验证 |
| `T.simd.vld(addr, "BRC_B32")` | 权重单元素广播 | `examples/ascend/flash_attention/core.py L126–127, L192, L303, L314`（4 处 BRC_B32 广播先例）；`testing/ascend/language/test_tilelang_ascend_simd_vld_brc.py` | 已验证 |
| `T.simd.vmula(dst, src0, src1)` | FMA 累加（寄存器 dst） | `testing/ascend/language/simdvf/test_simdvf_arithmetic.py L85–111`（serial_vreg_accum：alloc_var + vdup + T.serial + vmula + vsts 全骨架回归）；`examples/ascend/deepgemm/kernels/epilogue.py L181–182` | 已验证（回归） |
| `T.simd.vdup(T.float32(0.0), "float32")` | 累加器零初始化 | serial_vreg_accum L100（`vdup(0.0, T.float32)` 同款） | 已验证 |
| `T.simd.vmuls(acc, T.float32(scaling))` | 编译期标量乘 | `examples/ascend/flash_attention/core.py L110–113`（`vmuls(vld(...), T.float32(scale), …)` 同款） | 已验证 |
| `T.simd.vsts(addr, src, dist="PK_B32")` | 64×16-bit 打包存储（**UNPK-64 fallback 形态用**；主选尾声为 NORM_B16） | `testing/ascend/language/test_tilelang_ascend_simdvf_cast.py L49–51`（f32→f16 + PK_B32 精确同形，保序真机） | 已验证 |
| `T.reinterpret(vreg, "uint16x128" / dtype+"x128")` | VReg 位型重解释（vor 前置 / 存储前置） | `examples/ascend/flash_attention/core.py L144–154`（uint16x128 重解释 + 回转 "bfloat16x128" 真机）；`examples/ascend/deepgemm/kernels/epilogue.py L101/L116`；`tilelang/language/tir/op.py L382` | 已验证（真机先例） |
| `T.simd.vor(src0, src1)` | 偶/奇 lane 位或重合并（位置不相交 → OR = 合并；装载拆分的逆置换） | `examples/ascend/flash_attention/core.py L143–152`（真机，order-sensitive P 矩阵路径）；CANN `basic_block_vf_nd.h`（FP32_TO_F16_EVEN/ODD + `Or<uint16_t>` 合并范式）；`tilelang/ascend/language/simd.py L851` | 已验证（真机先例） |
| `T.simd.vsts(addr, src, dist="NORM_B16")` | 16-bit 128-lane 全宽连续存储（主选尾声，单发 128 元素） | `examples/ascend/deepgemm/kernels/epilogue.py L116`（`T.reinterpret(·, "int16x128")` + NORM_B16 真机同形）；`tilelang/ascend/language/simd.py L457` | 已验证（同族先例；fp16/bf16 缓冲组合列 L0 首验项，§9.2 R2） |
| `T.simd.alloc_var("float32")` | VRF 累加器 | `tilelang/ascend/language/simd.py L234`（local.var 语义）；logsumexp_simd / serial_vreg_accum | 已验证 |
| `@tilelang.jit(target="ascend", out_idx=[])` | 原位写 JIT | 源同款 out_idx=[]；logsumexp_simd（out_idx=[1] 输出式对照）+ `_route_stats`（out_idx=[] 同款已过 Stage 3） | 已验证 |
| 组循环内 SSA 重初始化（`acc_even = vdup(...)` in T.serial） | 每组重置累加器 | `examples/ascend/deepgemm/kernels/transform_sf.py L200–204`（T.serial 循环内 alloc_var 重赋值先例：`valid_group = S.por(...)`） | 已验证（同构先例） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 一维 Kernel / `threads=` 禁用 | Yes | `T.Kernel(T * nh)`，无 threads；线程域不使用（§2.2） |
| C0 分形对齐 / GEMM 非整除 | No | 纯 Vector kernel，无 `T.gemm`（Cube 路径已在 §1.6.0 R1 行 2 定量淘汰） |
| UB 容量（248KB，SimdVF 无 32KB 扣减 → 90% 折扣 228,556B） | Yes | 32,288B ≤ 228,556B（14.1%，§4.5）；harness 外最坏界（nh=1 且 K=8、H=7168 假想 T≥224）129,056B 仍 ≤ 预算 |
| OOB padding（`pad_value`/`data_select`） | No（不使用 OOB padding API） | 尾区由**分配 padding**（BH_pad）+ 静态 extent 搬运吸收：全部 `T.copy` extent 编译期静态（BH \| H 整除保证无尾块）；padded lane 计算结果滞留 UB 不落 GM（E6 机器验证）——正面规避「运行期钳制 extent 静默 OOB」陷阱（`_indexed_expert_gemm` Stage 3 教训） |
| SimdVF 块内 GM 访问禁令 | Yes | 三段 `T.copy` 均置于 VF 域外；VF 域内仅 UB 访问（vld/vsts 基址 x_ub/w_ub/out_ub） |
| `T.Parallel` 须在 VF 域内 | Yes（规避） | 本设计无 T.Parallel（并行由 grid × 向量宽度承载，§2.3） |
| fill 对齐/整除 | No | 无 `T.fill`（累加器初始化用 vdup 常量广播——不涉 fill 的 64-lane 整除/32B 基对齐约束；UB 分配天然 32B 对齐） |
| 核内串行循环边界静态 | Yes | ng / top_k / bh / bh_pad / nh 全为工厂烘焙编译期常量（core-split-strategy.md §2.1 满足） |
| 累加 dtype 语义 | Yes | fp32 累加域全程保持（VRF acc fp32×64；候选 5 语义拒绝） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 维度 | `T.Kernel(ceildiv(T*H, threads), threads=256)` | `T.Kernel(T * nh)`（一维、无 threads） | §0.6 R1 |
| 执行单元 | thread-per-element（SIMT 标量链） | SimdVF 128-lane 纵向向量（lane=列） | §0.6 R1 |
| 数据搬运 | GM 标量直读直写（coalesced） | `T.copy` 三段 bulk（UB staging） | §0.6 R2 |
| 累加器 | `T.alloc_local((1,), "float32")`（每 thread 1 标量） | `T.simd.alloc_var`（每 128e 组偶/奇列 64-lane 向量寄存器 ×2） | §0.5 条目 3 等价替换 |
| 权重访问 | 每 thread 每 slot GM 标量读 | UB staging + BRC_B32 广播（每 slot 每 128e 组 1 向量装载） | §1.6.2 #4 |
| 边界守卫 | `if index < T*H`（末块线程闲置） | 无守卫（BH \| H 静态整除 + padding containment，E6） | §0.5 条目 7 舍弃 |
| 工厂参数 | `_kernel(threads)` + `{"threads": 256}` | `_kernel()`（无参）+ `config = {}` | K3/K9（wrapper 已落地） |
| 索引计算 | 每 thread div/mod（T·H 次） | 每 block div/mod（T·nh 次）+ 仿射组寻址 | §0.5 条目 5 等价替换 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/src/tileops/_kernels/reduction/logsumexp_simd/logsumexp_simd_kernel/perf_opt/_logsumexp_kernel_single.py` | 高度相似（同代 TileOPs SimdVF 归约重实现） | `@tilelang.jit(target="ascend")` 工厂面、`programming_mode: simd` docstring、padded 分配 + 部分范围拷出（N_pad/UNPK 装载/vcvt part/vsts 双 dtype 路径）、56 AIV 核数注释、HMAX 128-lane 装载形态（OPT-2 同族先例） |
| `testing/ascend/language/simdvf/test_simdvf_arithmetic.py` | 核心骨架（serial_vreg_accum） | `alloc_var` + `vdup(0)` + `T.serial` 循环 + `vmula` 寄存器累加 + `vsts` —— 本 kernel 内层链的回归级同构 |
| `examples/ascend/flash_attention/core.py` | 高（BRC 广播 × 向量乘加 + vor 重合并） | `accumulate_output` 宏（L188–196）：`vld(alpha_ub[i], "BRC_B32")` 广播 × `vmul/vadd` 逐 chunk —— 权重广播×数据块的同构先例；`vmuls(vld(...), T.float32(scale))` 编译期标量乘；**L128–152：DINTLV 偶奇拆分 + narrowing part=0/1 + `vor` 位或重合并——本设计尾声（§3.3）的直接真机先例** |
| `examples/ascend/deepgemm/kernels/epilogue.py` | 高（向量乘加 + 打包存储） | `vmula` 用例（L181–182）；`vcvt(vld(bias[…], "UNPK_B16"), "float32")` 偏置装载（L38）；`vsts(..., dist="NORM_B16")` 16-bit 全宽存储（L116，`T.reinterpret(·, "int16x128")` 同形）；`T.reinterpret` VReg 位型转换先例（L101） |
| `testing/ascend/language/test_tilelang_ascend_simdvf_cast.py` | API 佐证（cast 惯例，两形态均保序真机） | f32→f16 + `vsts(dist="PK_B32")` 精确同形（L47–53）；half→fp32 `vld(dist="UNPK_B16")` + `vcvt(part=0)` 形态（L77–80）——分别为 E3b fallback 的存储/装载佐证与 E3a 数值佐证 |
| `examples/indexed_expert_mlp/_indexed_expert_gemm/_indexed_expert_gemm.py` | 同批同族（上游 producer，Stage 3 已过） | 3-D GM 标量行 + 双切片 `T.copy`（L288–292）；「运行期 extent OOB」教训的 per-element fallback 对照（本设计以静态 extent 正面规避） |
| `examples/indexed_expert_mlp/_route_stats/_route_stats.py` | 同批同族（Stage 3 已过） | `@tilelang.jit(target="ascend", out_idx=[])` 无参工厂面 + `T.Kernel(1)` + SimdVF/标量混合结构 + wrapper 调用面 |
| `examples/ascend/example_per_token_cast_to_fp8.py` | 中（SimdVF 数据流形态） | 逐元素 cast 类 kernel 的 SimdVF 组织（VF 域内外划分） |
## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| expert_output | (T, K, H) | fp16 / bf16 | 行主连续（K-中间归约轴，H 最内连续轴）；wrapper 冻结校验 |
| topk_weights | (T, K) | float32 | 行主连续；wrapper 冻结强制 fp32 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| output | (T, H) | fp16 / bf16（同 expert_output） | 原位写回（out 模式，`out_idx=[]`，wrapper 返回传入缓冲） |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| x_ub | (K, BH_pad) | 输入 dtype | UB | expert_output [K, BH] 块 staging（padded 至 128 倍数吸收尾组 lane；行跳步 = GM 的 H 轴 → UB 的 BH_pad 轴） |
| w_ub | (K,) | float32 | UB | 本 token 权重行 staging（BRC 广播源） |
| out_ub | (BH_pad,) | 输出 dtype | UB | 输出块 staging（vor 重合并后 NORM_B16 全宽写入；搬出仅 [0, BH)） |
| acc_even / acc_odd | 向量寄存器 | float32 | VRF（local.var） | 每 128e 组偶/奇列双累加器（acc_even[m] ↔ 列 g·128+2m、acc_odd[m] ↔ 列 g·128+2m+1；不占 UB 预算，`CONST-vrf-capacity-a5`） |

### 4.4 内存搬运路径

```
GM[expert_output T×K×H] --T.copy(MTE2, 2-D [K,BH] 静态 extent)--> UB[x_ub K×BH_pad]
GM[topk_weights T×K]   --T.copy(MTE2, 1-D K fp32，与上行无依赖可重叠)--> UB[w_ub]
UB[x_ub] + UB[w_ub] --SimdVF 向量链(vld NORM/BRC + vcvt 偶奇加宽 + vmula)--> VRF[acc_even/acc_odd]
VRF[acc] --vmuls(仅 s≠1) + vcvt(part=0/1 偶奇分置) + vor 重合并 + vsts(NORM_B16)--> UB[out_ub BH_pad]
UB[out_ub[0:BH]] --T.copy(MTE3, 静态 extent BH)--> GM[output T×H]
```

无 L1/L0 参与（纯 Vector）；无 ND2NZ/fractal 转换（无 Cube 段）。

### 4.5 UB 内存预算

| Buffer | Shape（最坏 harness = ds-v3-t64: T=64,K=8,H=7168 → nh=4, BH=1792） | dtype | 大小 (Bytes) |
|--------|------------------------------------------|-------|-------------|
| x_ub | (8, 1792)（BH_pad=1792 = 14×128 天然整除） | fp16/bf16 | 28,672 |
| w_ub | (8,) | fp32 | 32 |
| out_ub | (1792,) | fp16/bf16 | 3,584 |
| **总计** | | | **32,288** / UB 预算 228,556B（248KB × 90%，SimdVF 无 32KB 线程上下文扣减，`CONST-capacity-a5` assumption）= **14.1%** |

其余 harness shape：ds-v3-t32（nh=7, BH=1024）：18,464B；qwen3-235b-t32（H=4096 → nh=8, BH=512）：9,248B；ds-v3-t1（nh=56, BH=128）：2,336B；测试 shape（H=128/100/512/96）≤ 2,336B。harness 外最坏界（假想 T ≥ 224 → nh=1、BH=H=7168、K=8）：114,688 + 14,336 + 32 = 129,056B（56.5%）仍 ≤ 预算——容量不构成任何 (T,K,H) 契约 shape 的约束（K ≤ 8 界内）。VRF 侧：活跃寄存器 ≤ 8 vreg（2 acc + w_vec + x_h + 2 x_half + 尾声临时）≈ 2KB ≪ 256KB 假设（`CONST-vrf-capacity-a5`）。

### 4.6 动态轴定义

**无**。`(T, K, H, dtype, scaling)` 由 `lru_cache` 工厂特化为编译期常量（源同款机制，§0.5 条目 8 保留）；`nh / BH / BH_pad / ng` 为 host 编译期启发式产物（§5.2）；核内全部循环边界静态（`ng`、`top_k` 均工厂烘焙——core-split-strategy.md §2.1「核内串行循环边界必须静态」满足）。harness 展开：4 manifest 行 × 2 dtype = 8 特化 + 测试 shape 若干（(4,2,128)/(17,2,128)/(4,2,100)/(8,4,512)/(2,2,128)/(1,2,128)/(4,2,128)-scales 等经 op 链路）。

### 4.7 JIT 配置

```python
@tilelang.jit(
    target="ascend",          # 规范入口
    out_idx=[],               # 原位写（output 为入参），无返回值——与源一致
)
```

pass_configs 默认（AutoSchedule 开启，不传 `TL_ENABLE_AUTO_SCHEDULE`）；无 compile_flags 需求（源 GEMM 工厂的 `-DENABLE_BF16` 为其 Cube 路径所需；本 kernel bf16 走 vcvt 通用路径，无编译宏依赖——logsumexp_simd bf16 同形态无 flags 先例）。

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector（逐元素乘加 + cast + 寄存器内 K 归约 + DMA 搬运；无 matmul → 不占 Cube）。

**判定依据**: 全部计算为 SimdVF 向量原语；无 `T.gemm`；codegen 自动判 `__vector__` kernel 落 AIV（tilelang-ascend skill §3.1）。Cube 替代路径已在 §1.6.0 R1 行 2 / R4 定量淘汰（M=1/K=8 的分形 padding 浪费 32×）。

### 5.2 Block 划分

**块 = (token, h-chunk)**：`num_blocks = T × nh`，`bid → token = bid // nh`、`h_blk = bid % nh`、`h0 = h_blk · BH`。h-chunk 快变（相邻 bid 处理同 token 相邻列块——L2 局部性友好）。

**BH 选取启发式（host 编译期，§3.3 `_pick_nh`）**：`nh` 取 **H 的除数**（保证 BH = H/nh 整除 → **无尾块、全部拷贝 extent 静态**——规避运行期 extent OOB 陷阱）且 `nh ≤ max(1, H // 128)`（每 chunk ≥ 128 列，保持 per-block 数据量 ≥ K×256B 摊薄 MTE base 90/180 cyc），目标 `|T·nh − 224|` 最小（224 = 4×56 AIV 的 4 波中点——块数太少不满核〔T=1 整行单块劣化 ≈10×，§1.6.0 D-2〕、太多则每块 MTE base 占比上升〔BH=896 时 +18% 时延，D-2 复算〕）；无除数落窗时回落 nh=1（整行）。逐 workload 取值与验证：

| workload / 测试 | (T, K, H) | nh | BH | blocks | 波数（÷56 AIV） | UB/块 (B) |
|---|---|---|---|---|---|---|
| ds-v3-t1 | (1, 8, 7168) | 56 | 128 | 56 | 1 | 2,336 |
| ds-v3-t32 | (32, 8, 7168) | 7 | 1024 | 224 | 4 | 18,464 |
| ds-v3-t64 | (64, 8, 7168) | 4 | 1792 | 256 | 5 | 32,288 |
| qwen3-235b-t32 | (32, 8, 4096) | 8 | 512 | 256 | 5 | 9,248 |
| small-route / scales / out-buffer | (4, 2, 128) 等 | 1 | 128 | 4 | 1 | 776 |
| second-route-group | (17, 2, 128) | 1 | 128 | 17 | 1 | 776 |
| dims_off_the_tile_grid | (4, 2, 100) | 1 | 100（BH_pad=128） | 4 | 1 | 776 |
| manifest-shaped | (8, 4, 512) | 4 | 128 | 32 | 1 | 1,296 |

（nh/BH/布局 containment 已由 verify_equiv.py E6 按同启发式逐 shape 机器验证。BH 为 Stage 4 调优自由变量——§5.4。）

### 5.3 约束分析

- **C0 分形对齐**: 不适用（无 GEMM / 无 Cube 段）。
- **UB 容量**: 32,288B ≤ 228,556B（90% 折扣口径，`CONST-capacity-a5`）✓；无 multibuffering 版本膨胀项（单遍直线依赖，默认单版本；§6.3）。
- **L0 容量**: 不适用。
- **装载/存储对齐**: vld NORM 行基址 = k·BH_pad·2B（BH_pad ≥ 128 → ≥ 256B 整倍 → 32B 对齐 ✓）；组内偏移 g·256B ✓；vsts NORM_B16 落点 g·128·2B = 256B 倍、单发 128 元素 ✓；BRC_B32 单元素 ✓。T.copy 源行宽 BH·2B：harness 全部 ≥ 256B（宽行档 100 B/cycle，`CONST-mte-bandwidth-model`）；测试 off-grid H=100 → 200B 行（非 32B 对齐步长）——**logsumexp_simd L1 (32,300)/(33,300) 600B 行实测通过先例**（非 32B 对齐行步长的 2-D T.copy 真机通过）；已列 L0 首验项（§8.3 第 1 项含该 shape）。
- **拷出尾区**: extent = BH 的 1-D 区（BH 非 32B 倍数的 shape：H=100 → BH=100 → 200B）。UB→GM 非 32B 整数倍 extent 的文档化状态未单列（tilelang-ascend skill §6.1 byte-granular rows 表述针对 GM→UB 方向；logsumexp 先例行宽 32B 对齐）——列 **L0 编译冒烟首验项**；fallback：SIZE 向下 32B 对齐段 bulk + 尾段标量小段拷贝（§9.2 R3）。
- **边界/尾块**: 无尾块（BH \| H）；lane 尾区 [BH, BH_pad) 由分配 padding 吸收（计算结果滞留 UB，E6 containment 机器验证）。

### 5.4 注意事项

- **BH 为 Stage 4 调优自由变量**：§1.6.3 微调清单——① per-shape BH 重选（表值为基础值；D-2 显示 T=64 在 BH=1792 附近最优、偏离 ±2× 劣化 10–30%）；② GU 组展开（广播摊销，−4~−6 op/组）；③ OPT-3 首槽乘（−2 op/组 = −3.7%）；④ vaxpy 标量乘加变体（`T.simd.vaxpy(acc, x, w_ub[k])` 免 BRC 广播，组 op 54 → 46——运行时 UB 标量作 simd 标量操作数的同构先例 transform_sf.py L201–203 vcmps；同构非同形，L0 首验）；⑤ x_ub 双缓冲（`T.annotate_buffer_versions({x_ub: 2})` + 组级流水，D-2 串行链上界 8.1–8.8K → 计算下界 4.3–5.7K 的重叠空间）。均不改结构（§1.6.3 已论证：属参数级调优而非结构候选）。
- **构建期容量 guard（harness 外防御）**：`K·BH_pad·2 + BH_pad·2 + K·4 > 228,556B` 时（需 K·H > ~14K 元素 × 契约 dtype——K ≤ 8 界内 H > 1.8M 才触发，harness/测试 shape 均远未触及）工厂 host Python 层强制降 nh（最小化 BH 至除数窗下界）；仍超界则断言报错（容量契约外 shape）。

### 5.5 分核策略（grid 划分 + AIC/AIV 分工 + 同步策略）⭐

> 三要素判定标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（§2.1 designer 要求）。

- **① grid 划分（含核数依据）**: `num_blocks = T × nh`（一维 `T.Kernel(T * nh)`；逐 workload 值见 §5.2 表——harness 范围 56–256 blocks）。核数实查（2026-10-29，本机，与同批 `_route_stats` DESIGN §5.5 同日同机记录）：
  ```python
  import torch
  props = torch.npu.get_device_properties()   # device: Ascend950PR_9579
  props.cube_core_num, props.vector_core_num  # → 28, 56
  ```
  实查值 **cube_core_num = 28 / vector_core_num = 56**（与 `CONST-arch-a5` 2026-10-16 试点实测一致，status: measured；本任务复查因 dev-build 首次 tensor 初始化耗时而沿用同日同机实测记录——查询代码与返回值如上，Stage 3 冒烟时可复跑确认）。block 数与核数关系：harness 全部 shape 单核单块、1–5 波（§5.2 表），无 persistent 化需求（块粒度已细至 128 列/chunk，launch 队列开销由块数 ≤ 256 界定；T.Persistent 为 Stage 4 可选项，参照 logsumexp_simd 形态）。
- **② AIC/AIV 分工**: 纯 Vector kernel（无 `T.gemm`）→ codegen 自动判 `__vector__`，全部工作落 **AIV**：每 block = 1 AIV 执行「MTE2 搬入（x_ub + w_ub）→ SimdVF 向量归约（vld/vcvt/vmula/vsts）→ MTE3 搬出（out_ub）」。**AIC 不参与**（本 launch 的 28 AIC 空闲——上游 `_indexed_expert_gemm` 的 Cube 段已执行完毕，本 kernel 为 op 流水线的 Vector 收尾段；其 56-AIV 满载并行是 §5.2 shape-aware 分块的设计目标）。无 MixedKernel / sid / dual_copy（无 Cube 段）。
- **③ 同步策略（auto|manual 声明）**: **`auto_schedule: yes`**（默认，不传 `TL_ENABLE_AUTO_SCHEDULE`）。数据依赖边界（AutoSchedule 据此自动插跨 pipe flag）：`T.copy(x)`〔MTE2，写 x_ub〕→ SimdVF 段〔读 x_ub〕；`T.copy(w)`〔MTE2，写 w_ub〕→ SimdVF 段〔读 w_ub〕；SimdVF 段〔写 out_ub〕→ `T.copy(out)`〔MTE3，读 out_ub〕。两段 MTE2 互不依赖可被 AutoSchedule 重叠。无手动 `T.ascend_set_flag/wait_flag`，flag id 预算（核内 8 / 跨核 16，`CONST-arch-a5`）零消耗。**核内串行循环边界静态**：num_tokens / top_k / bh / bh_pad / ng / nh 均为工厂编译期常量（§4.6）。无 `T.Stage` / `T.Task` 调度提示（单遍直线结构无需；`T.simd` 链的 `latency=` 注解为 Stage 4 可选项，flash core `T.SimdVF(latency=318)` 先例）。

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 组循环（g：128 列/步） | 核内串行（VF 域内） | `T.serial(ng)` | 静态边界（工厂烘焙）；循环体全向量 op；`vld(a_ub[i·LANES])` 同构先例（serial_vreg_accum） |
| slot 循环（k） | **编译期展开**（无运行时循环） | Python `range(top_k)` | K ≤ 8：消除循环控制开销 + 寻址常量折叠；§1.6.2 #9 |
| 累加器重初始化 | 组循环内 SSA 重赋值 | `acc = vdup(0.0)` | transform_sf.py T.serial 内 alloc_var 重赋值先例 |
| 搬入/搬出 | DMA 任务（无循环） | `T.copy` ×3 | 全量单发（静态 extent） |
| 元素级 | 无 `T.Parallel` | — | 并行由 grid（T×nh blocks）× 向量宽度（128 lane/组）承载；§1.6.2 全表 |

### 6.2 循环伪代码

```python
with T.Kernel(T_ * nh) as bid:                        # 一维 grid（§5.5）
    token = bid // nh; h0 = (bid % nh) * bh           # block 级标量（§1.6.2 #6）
    x_ub = T.alloc_shared((K, bh_pad), dtype)
    w_ub = T.alloc_shared((K,), "float32")
    out_ub = T.alloc_shared((bh_pad,), dtype)
    T.copy(expert_output[token, 0:K, h0:h0+bh], x_ub[0:K, 0:bh])   # MTE2
    T.copy(topk_weights[token, 0:K], w_ub[0:K])                    # MTE2（可重叠）
    with T.SimdVF():
        acc_even = T.simd.alloc_var("float32"); acc_odd = T.simd.alloc_var("float32")
        for g in T.serial(ng):                         # 128-lane 组循环（静态）
            acc_even = T.simd.vdup(T.float32(0.0), "float32")
            acc_odd = T.simd.vdup(T.float32(0.0), "float32")
            for k in range(K):                         # 编译期展开，slot 升序
                w_vec = T.simd.vld(w_ub[k], "BRC_B32")
                x_h = T.simd.vld(x_ub[k, g*128])       # lane j <-> col g*128+j
                T.simd.vmula(acc_even, T.simd.vcvt(x_h, "float32", part=0), w_vec)  # even lanes
                T.simd.vmula(acc_odd, T.simd.vcvt(x_h, "float32", part=1), w_vec)   # odd lanes
            if_s = scaling != 1.0                      # host 编译期（OPT-1）
            if if_s:
                acc_even = T.simd.vmuls(acc_even, T.float32(scaling))
                acc_odd = T.simd.vmuls(acc_odd, T.float32(scaling))
            # epilogue: even/odd narrowing + vor re-merge (inverse permutation
            # of the load split) + one full-width NORM_B16 store
            even16 = T.simd.vcvt(acc_even, dtype, part=0)
            odd16 = T.simd.vcvt(acc_odd, dtype, part=1)
            merged = T.simd.vor(T.reinterpret(even16, "uint16x128"),
                                T.reinterpret(odd16, "uint16x128"))
            T.simd.vsts(out_ub[g*128], T.reinterpret(merged, dtype + "x128"),
                        dist="NORM_B16")
    T.copy(out_ub[0:bh], output[token, h0:h0+bh])      # MTE3（extent bh）
```

（`if_s` 为示意注释——实际代码为工厂闭包内 Python `if scaling != 1.0:` 编译期分支，两条特化各自静态，§3.3。）

### 6.3 流水线优化

不使用 `T.Pipelined`（每块单遍直线：搬入→归约→搬出各一次，无迭代 tile 复用面）；两段 MTE2（x_ub 与 w_ub）互不依赖，其重叠由 AutoSchedule 自动调度（`auto_schedule: yes`，§7）。不使用 `T.Persistent`（块粒度已细、块数 ≤ 256，无工作遍历复用需求；Stage 4 可选，logsumexp_simd 形态参照）。**Stage 4 可选项**：x_ub 双缓冲 + 组级流水（`T.annotate_buffer_versions({x_ub: 2})`，§5.4 ⑤——D-2 显示串行链上界与计算下界间有 3–4K cyc 的重叠空间）。

### 6.4 尾块处理

- **块级尾块**：无（BH \| H 整除，§5.2 启发式——所有拷贝 extent 静态）。
- **lane 级尾区**：BH_pad 吸收（[BH, BH_pad) 的装载垃圾→计算→存储全链滞留 UB，搬出 extent=BH 不外泄；E6 机器验证）。垃圾 lane 的 fp 运算（可能含 NaN 模式）无跨 lane 污染（逐元素 op），且 16-bit 域无陷阱。
- **累加边界**：K=1 退化（单 vmula）、T=1 单 token、H < 128（单 padded 组）均由结构覆盖（E6 含 (1,4,96) shape）。

## 7. 同步策略

### 7.1 同步模式

**模式**: auto_schedule: **yes**（AutoSchedule 自动同步；不设 `TL_ENABLE_AUTO_SCHEDULE: False`）

### 7.2 同步点说明

auto 模式——依赖边界（AutoSchedule 自动插 flag，无需手动配对）：

| 依赖边 | 生产者 → 消费者 | 说明 |
|--------|----------------|------|
| x 块搬入 → 向量段 | MTE2（x_ub 写）→ SimdVF（x_ub 读） | 跨 pipe RAW |
| w 行搬入 → 向量段 | MTE2（w_ub 写）→ SimdVF（w_ub 读） | 跨 pipe RAW |
| 向量段 → 搬出 | SimdVF（out_ub 写）→ MTE3（out_ub 读） | 跨 pipe RAW |
| （可重叠）x 搬入 ∥ w 搬入 | 互不依赖 | AutoSchedule 自由度 |

grid = T×nh 且输出空间两两不相交 → **无跨 block / 跨核同步**；VF 域内无 store→load 冒险（acc 驻留寄存器；vsts 只写 out_ub、vld 只读 x_ub/w_ub——无 flash core 那种 UB 回读所需的 `S.mem_bar("VST_VLD")`）。flag id 预算（核内 8 / 跨核 16，`CONST-arch-a5`）：零消耗。

### 7.3 pass_configs 配置

```python
pass_configs = {}   # 默认（AutoSchedule 开启）；完全手动核才需要 TL_ENABLE_AUTO_SCHEDULE: False
```

## 8. 验证方案

### 8.1 Golden 函数

> golden 以 **§0.1 源算子语义**为唯一依据实现（fp32 域 slot 升序乘加 + scaling 后置 + 单次 narrowing cast），不复刻 NPU 实现细节（staging / FMA / BRC / PK 打包均不出现）。设备约定：golden 张量建在输入同设备（AGENTS.md golden-on-NPU 惯例，torch 在 NPU 上计算）；对照用 `torch.testing.assert_close`（分档容差，§8.2）。

```python
def golden_weighted_reduce(expert_output: torch.Tensor, topk_weights: torch.Tensor,
                           scaling: float) -> torch.Tensor:
    """Source-semantics weighted reduce (DESIGN 0.1).

    expert_output: [T, K, H] fp16/bf16; topk_weights: [T, K] fp32 -> [T, H].
    fp32 accumulation over slots (ascending), scaling applied once in fp32,
    single narrowing cast at the end. Device: stays on expert_output.device.
    """
    x32 = expert_output.float()                              # per-element up-cast
    acc = torch.zeros(expert_output.shape[0], expert_output.shape[2],
                      dtype=torch.float32, device=expert_output.device)
    for k in range(expert_output.shape[1]):                  # slot ascending
        acc = acc + x32[:, k, :] * topk_weights[:, k:k + 1]  # fp32 mul + add
    return (acc * scaling).to(expert_output.dtype)           # one narrowing cast
```

> 注：`for k` 显式升序循环保持 §0.1 累加顺序语义（向量化写法 `x32.mul(w.unsqueeze(-1)).sum(1)` 的 torch 求和序为实现自由度——两者在 1e-2 分档容差下等价，显式序更贴近源语义，供 L0 严格对照；op 级 golden 沿用 `ref_routed_experts`（`examples/TileOPs/workloads/moe.py`）。

### 8.2 精度标准

fp16/bf16 输出，fp32 累加——按 AGENTS.md 分档容差：

| dtype | atol | rtol | 对照方式 |
|-------|------|------|----------|
| float16 / bfloat16（契约 dtype） | 1e-2 | 1e-2 | `torch.testing.assert_close(got, golden)`；附加 nonfinite mask 相等断言（NaN/Inf 位置一致，§0.1 边界语义）；设计期证据：verify_equiv E3a/E3b lane-mapping 位级恒等（采纳与 fallback 形态的输出位置正确性）+ E4f/E4s/E5 全部 ≤1 ulp / 0% 违反（FMA 舍入行为两口径 + 全采纳链 vs fp64 参照） |

> FMA 融合舍入与源两舍入的差异已由 E4 机器验证在 1 ulp fp16 内（0.03% 元素、0% 容差违反）——分档容差充分覆盖；不适用位级精确标准（同批 `_route_stats` 的 int32 逐位标准不适用于本浮点 kernel——其 RETROSPECTIVE 教训明示「后续涉及 fp32/bf16 累加的函数须改用 skill 分档容差」）。op 级：`_moe_compare`（atol=rtol=1e-2 + 归一化平方误差 ≤1e-4，fp64 计算——`examples/TileOPs/tests/ops/test_indexed_expert_mlp.py` 移植策略）。

### 8.3 L0 门槛测试计划（Stage 3 首跑必过）

1. **编译冒烟**：8 个 manifest 特化（4 行 × 2 dtype：(1,8,7168) / (32,8,7168) / (64,8,7168) / (32,8,4096) × fp16/bf16）+ 测试 shape 集（(4,2,128) / (17,2,128) / (4,2,100) / (8,4,512) / (2,2,128) / (1,2,128)）JIT 编译通过（`target="ascend"`）——**含 (4,2,100) 的 200B 行 2-D T.copy 与 200B extent 搬出首验**（§5.3 首验项；命中即启用 §9.2 R3 fallback）＋ **vor 重合并尾声首验**（2 dtype × s∈{1.0, 2.5}：`vcvt(part=0/1)` 偶奇分置 + `vor` + `T.reinterpret(merged, dtype+"x128")` + `vsts(dist="NORM_B16")` 全宽存储组合链，§9.2 R2；命中即启用 UNPK-64 fallback 形态）。
2. **golden 对照**：上述 shape ×「rand / corner」两类值（±inf/NaN/±0/subnormal/极值 + 权重含 0/负/大数——verify_equiv corner 集同款）跑 kernel vs `golden_weighted_reduce`：`assert_close(atol=rtol=1e-2)` + nonfinite mask 相等。
3. **scaling 双路**：s=1.0（OPT-1 跳乘特化）与 s=2.5（乘法路径）分别对照（E1 语义在真机的复核）。
4. **确定性回归**：同一输入连跑 3 次，输出逐位一致（grid 划分不相交 + 无原子 → 结构性确定）。
5. **op 级闭环**：`examples/TileOPs/tests/ops/test_indexed_expert_mlp.py` 全部用例（small_route 4 模式 × 2 dtype、single_token、dims_off_the_tile_grid、manifest_shaped、scales、out-buffer、roofline）——经 op 链路覆盖本 kernel 的端到端正确性。
6. **边界专项**：K=1（单 slot 退化）、T=1（56 blocks 分块路径）、H=100（padded 组路径）、H=96（BH_pad 单组全 padding 吸收）。

完整分层套件（L1/L2/Boundary 枚举）由 `tilelang-op-develop` 阶段交付，不在此展开。

## 9. 风险点与注意事项

### 9.1 已知约束

- `T.Kernel` 一维、`threads=` 禁用（TypeError）——本设计 `T.Kernel(T * nh)` 无 threads。
- SimdVF 块内禁止 GM 访问（VFChecker）——三段 `T.copy` 均在 VF 域外。
- `T.Parallel` 必须在 VF 域内——本设计无 T.Parallel。
- UB 248KB（SimdVF 无扣减；90% 折扣 228,556B）——harness 最坏 32,288B（14.1%）。
- 核内串行循环边界必须静态——工厂特化保证（§4.6）。
- `Z3 scheduler failed... falling back to topological sort` 警告为良性（同批 `_indexed_expert_gemm` Stage 3 实证）。
- dev build 首次 tensor/npu 初始化与每 shape 编译（2–11s）是时间大头：L0 冒烟 shell 探针预算 ≥600s、后台 + tee 落盘（同批教训）。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| R1: `vmula` dst 非法（传入不可变寄存器值） | dst 用 SSA 表达式而非 buffer 访问 | 编译期 ValueError（simd.py L657–662） | dst 用 `T.simd.alloc_var` 变量（serial_vreg_accum 同款；本设计 §3.3 已按此形态） |
| R2: vor 重合并尾声组合（`vcvt(part=0/1)` + `vor` + `T.reinterpret` + `vsts(NORM_B16)`）无单点直测先例 | fp16/bf16 narrowing 偶奇分置 + 位或合并 + 16-bit 全宽存储的**完整组合**（组件级真机先例各自成立：flash core bf16 vor 链 L139–152 / deepgemm NORM_B16 L116 / cast 测试 narrowing L47–53，但三者未在同一示例中串联） | 编译期/运行期未知（组合层面） | L0 首验项（§8.3 第 1 项，2 dtype × 双 scaling）；fallback：**UNPK-64 对称形态**（option b——`vld(dist="UNPK_B16")` + `vcvt(part=0)` 装载 + `vcvt(part=0)` + `vsts(PK_B32)` 存储，cast 测试 L47–53/L77–80 两条保序真机形态、E3b lane-mapping 已验证位级恒等；组 op 数 +8/组〔54 → 62〕）；更深 fallback：逐半 ONEPT 存储（logsumexp bf16 实证形态，+2 op/组） |
| R3: UB→GM 非 32B 整数倍 extent（BH=100 → 200B） | off-grid 测试 shape 搬出 | 未知（文档未单列该方向） | L0 首验项（§8.3 第 1 项）；fallback：SIZE 向下 32B 对齐段 bulk（192B）+ 尾段 8B 标量小段拷出（VF 域外标量核，`_route_stats` 同族先例） |
| R4: 200B 行 2-D T.copy 慢档 | H=100 的 GM 行步长非 32B 对齐 | 带宽降档（platform/unknown 档 40–50 B/cycle，`CONST-mte-bandwidth-model`） | 可接受（µs 级测试 shape）；logsumexp (32,300) 600B 行真机通过先例；生产 shape（H≥4096）全为宽行档 |
| R5: `vaxpy` 运行时标量变体首验失败（若 Stage 4 启用） | `T.simd.vaxpy(acc, x, w_ub[k])` 的标量操作数为 UB 运行时载入值 | 编译期/运行期未知 | 未启用则无风险（主选 BRC+vmula 全先例）；启用时 L0 首验 + fallback = 主选形态（BRC 广播，+8 op/组） |
| R6: 组循环内 SSA 重初始化不被支持（极端情形） | `acc = vdup(...)` 于 T.serial 体内 | 编译期错误 | transform_sf.py T.serial 内 alloc_var 重赋值先例（L200–204）；fallback：改 `T.simd.alloc_local((2,), "float32")` 数组 + `vmula(acc_arr[i], ...)` 索引形态（merging 测试 `dst[0]` 形态） |
| R7: K 极大时编译期展开膨胀 | 契约外 K（如 64） | 编译时间/代码膨胀 | harness/测试 K ≤ 8；防御：K > 16 时降级 `T.serial(top_k)` 运行时循环（边界仍静态；vmula 链同形） |

### 9.3 性能风险

- **D-2 占比 2.2–5.1%（T=32/64）**：本 kernel 为 op 的 Vector 收尾段，与 Cube 段串行——非关键路径主导项；Stage 4 若 op 级 profiling 显示本段占比超 D-2 上界（>5%），按 §5.4 调优清单处置（BH 重选优先）。
- **T=1 decode shape 占比 ≈15%**（D-2）：已由 shape-aware 分块缓解（56 blocks vs 整行单块 10× 劣化）；进一步优化空间（x_ub 双缓冲 / GU 展开）在 Stage 4。
- 所有估算基于 assumption 档常数（`CONST-valu-model` / `CONST-mte-bandwidth-model`）：D-2 数字为相对量级比较用途，绝对值以 `do_bench(msprof)` 实测为准（hardware-cost-model.md §2 协议）。

## 10. 交付清单

| 交付物 | 路径 | 状态 |
|--------|------|------|
| 设计文档 | `examples/indexed_expert_mlp/_weighted_reduce/DESIGN.md` | 本文件 |
| 等价性机器验证脚本 + 结果 | `examples/indexed_expert_mlp/_weighted_reduce/verify_equiv.py`（revision 1 重建：E3 为独立 lane-mapping 模型 + E3neg 负对照；E1–E6 全 EQUIV_PASS，2026-10-29 执行；运行留档 `history_version/verify_equiv_r1_run.log`） | 已交付 |
| kernel 实现（`_weighted_reduce` A5 重实现 + golden + 分层测试） | kernel 实现文件（`_weighted_reduce.py`，Stage 3 由 tilelang-op-develop 交付至本算子目录） | Stage 3（tilelang-op-develop）交付 |
| 集成验证 | TileOPs wrapper 接线 + op 级 report | Stage 5（tilelang-op-integrator）交付 |
| 性能调优（BH/GU/OPT-3/vaxpy/双缓冲 调优清单） | `perf_opt/` + perf_records.jsonl | Stage 4（tilelang-op-optimizer）交付 |

> 本设计冻结的 Stage 3 实现要点：编程模式 simd + `auto_schedule: yes`（docstring 声明）；工厂签名与 wrapper 调用面不变（§0.2）；§3.3 伪代码为实现蓝本（**尾声为偶奇分置 + vor 重合并 + NORM_B16 全宽存储形态，装载拆分与尾声合并互为逆置换**；API 全表 §3.4 已逐条佐证）；L0 门槛计划 §8.3（含两项首验项与 fallback）；§5.4/§1.6.3 微调清单为 Stage 4 专属，Stage 3 不实现。
