# gemm_silu 算子设计文档

> 任务：pilot / gemm_silu（非迁移，first_design）。输入契约：A [512,512] fp16、B [512,512] fp16（ascend NT GEMM 约定的 [N,K] 权重布局）、C [512,512] fp16；编程模式 mixed（用户指定）；auto_schedule: yes。
> 真机环境：Ascend950PR_9579（dav-3510），cube_core_num=28 / vector_core_num=56（实查，见 §5.5）。
> 工具链：tilelang-fork（fc32f47c）+ CANN 9.x（bisheng --npu-arch=dav-3510）。

## 1. 概述

### 1.1 算子名称

gemm_silu

### 1.2 功能描述

对 fp16 输入矩阵做带 fp32 累加的 GEMM，并在矩阵乘结果上逐元素施加 SiLU 激活（silu(z) = z·sigmoid(z)），单遍融合输出 fp16 结果：C = silu(A @ B)。属于"GEMM + 非线性 epilogue"融合算子族（Cube 主导计算 + Vector epilogue，CV 融合结构）。

### 1.3 数学公式

$$
Z = A \cdot B^{T}, \qquad C_{ij} = \mathrm{silu}(Z_{ij}) = Z_{ij}\cdot\sigma(Z_{ij}) = \frac{Z_{ij}}{1+e^{-Z_{ij}}}
$$

（Z 为 fp16 输入下的 fp32 累加 GEMM 结果；B 按 NT 约定取 [N,K] 布局，逻辑上参与运算的是 B 的转置。）

- 输入：A [M,K]=[512,512] fp16；B [N,K]=[512,512] fp16（L1-input NT 约定下权重 GM 布局为 [N,K]，逻辑上参与运算的是 B 的转置，见 §3.5.1）。
- 输出：C [M,N]=[512,512] fp16。
- 精度语义：GEMM 以 fp32 累加（L0C 累加器 fp32）；silu 在 **fp32 累加值上**施加（cast 后置），最终单点 cast 到 fp16。z·sigmoid(z) 与 z/(1+e^{-z}) 的恒等性经机器验证，见 §1.6.1。

### 1.4 算法描述（与 §1.6.0 调研选定一致：融合单 kernel mixed 算法）

选定算法族为**单 kernel 融合结构**（GEMM Cube 段 + silu Vector epilogue，L0C/UB 片上直连，中间量零 GM 往返），而非"分离两算子"（GEMM kernel 落 GM + silu kernel 读回）结构。计算分五步：

| 步骤 | 计算 | 落位 | 精度域 |
|------|------|------|--------|
| 1 | A/B 的 K 分块 tile 由 GM 搬入 L1（ND2NZ 分形由 InsertNd2Nz pass 自动插入） | AIC（MTE2） | fp16 |
| 2 | z = Σ_k a·b 沿 K 流式累加（K_TILES 次矩阵 MAC，首拍清零） | AIC（Cube MAD） | fp32 累加 |
| 3 | z 的 128×128 tile 由 L0C 搬入 UB（M-split 双 AIV 分担） | AIC 发射 / 双 AIV 接收 | fp32 |
| 4 | s = z/(1+exp(−z))，逐元素，直除形式（§1.6.1 优化后公式） | AIV ×2（SimtVF 向量化） | fp32 |
| 5 | c = fp16(s)，同一遍内完成 cast 并写回 GM（M-split 双 AIV） | AIV ×2 | fp32→fp16 |

关键语义决策（cast 时机）：silu 施加于 fp32 累加器、cast 后置到写出口——消除 z 的 fp16 预舍入误差（z 量级可达 ~10²，fp16 预舍入相对误差 2^-11 会直接进入激活值）；该路径同时是 dav-3510 硬件约束的自然结果（L0C→UB dual_copy 禁止类型转换，见 §3.5.1），golden（§8.1）与此语义一致。

### 1.5 数据流图

```
GM[A] --T.copy(MTE2+ND2NZ)--> L1[x_l1] --T.gemm--> L0C[res fp32] --T.dual_copy(M-split)--> UB[acc_ub fp32]
GM[B] --T.copy(MTE2+ND2NZ)--> L1[w_l1]                                      |
                                                                             v
                                                            SimtVF: s = z/(1+exp(-z)); cast fp16
                                                                             |
                                                                             v
GM[C] <--T.dual_copy(M-split, MTE3)-------- UB[out_ub fp16] <---------------+
```

（16 个输出 tile 相互独立，无跨 tile 依赖、无跨核归并。）

### 1.6 算法调研与优化分析 ⭐

> 设计第一优先级：先调研（1.6.0，调研四问）——输入公式"逐元素 silu 作用于 GEMM 结果"只是基线候选之一；再做公式级数学等价优化（1.6.1）；再做循环/标量计算的向量化替代分析（1.6.2）；最后做向量化轴与数据布局决策（1.6.3）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：同一数学语义（C = silu(A@B)，fp32 累加、silu 于 fp32 域施加、fp16 输出）的算法族。调研深度：**完整调研**（矩阵 + 融合类算子，Phase 1 判定）。查证来源：algorithm-candidates.md 参考表（ALG-elementwise / ALG-attention / ALG-gemv / ALG-transpose 条目）、examples/ascend/ 同类实现（example_gemm.py、deepgemm 族、example_rmsnorm.py、example_vecadd.py）、pattern-library（constants.md / bottleneck-patterns.md BP_dual_copy_utilization、BP_redundant_work_or_roundtrip）、结构分析与恒等变形自查。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线：融合单 kernel，silu 取自然展开 z·(1/(1+exp(−z))) | C=silu(A@B)，epilogue 逐元素 | —（基线） | — | ✅（基线） |
| 1 | silu 直除形式 | z/(1+exp(−z)) | 数学恒等（乘倒数 vs 直除，fp32 差 ≤2 ulp） | 每元素省 1 次乘（VALU 51 档） | ✅（§1.6.1 采纳） |
| 2 | silu tanh 形式 | 0.5·z·(1+tanh(0.5·z)) | 数学恒等（σ(z)=(1+tanh(z/2))/2） | 纸面消除除法（13 档）换 2 乘（51 档） | ✅（探针实测后否决，见 §1.6.1） |
| 3 | 分离两算子 | GEMM kernel 落 GM 的 Z + 独立 silu kernel 读回 | 数学等价、结构不融合 | 实现/复用简单；代价是 GM 往返 | ✅（R3/R4 对比后弃） |
| 4 | split-K 并行 + 终核施加 silu | K=512 分裂到多核 | 数学等价（partial 和后施加） | 更高并行度 | ❌：grid 16 < 28 AIC，并行度不缺；且 silu 需全 K 和，分裂即破坏 epilogue 融合（partial 和不能过 silu），退化为候选 3 的两 kernel 结构 |
| 5 | exp 换底 exp2·log2e | e^{-z} = 2^{-z·log2e} | 数学恒等 | 省超越函数？否 | ❌：SimdVF 域 vexp 本身就是 SFU 原语（tilelang/ascend/language/simd.py:860），无独立更廉价 exp2 通道；标量域 exp2 与 exp 同类 lowering（src/ascend/codegen/intrin_rule_ascend.cc:46-47），无收益 |
| 6 | silu 融入 GEMM 线性 epilogue（deepgemm apply_alpha 式 alpha 缩放） | C = alpha·(A@B) | **不等价**：silu 非线性，FixPipe/量化前处理只支持线性缩放与 dtype 转换 | — | ❌：语义不可行（examples/ascend/deepgemm/kernels/epilogue.py 的 epilogue 变换仅 scale+cast） |

**R2 在线算法**：结论——**无**（不存在需要在线化的多遍结构）。结构依据：① GEMM 的 K 维归约本身就是流式 running 累加（L0C 累加器单遍扫 K，clear_accum 首拍清零），已是"在线"形态；② silu 是逐元素映射，无跨元素依赖，天然单遍；③ 该算子族唯一的多遍结构就是 R1 候选 3（中间量 Z 落 GM 后再读回施加 silu），融合结构（基线）已将其消除；融合后算法达到最小扫描遍数（A、B 各一遍 K 流式读入，C 一遍写出），不存在可进一步单遍化的 running 统计量分解（输出元素 c_ij 依赖全 K 归约，该依赖已由 L0C 累加器以 O(1) 状态承载）。

**R3 复杂度对比**（M=N=K=512，fp16；四口径，含什么/不含什么已注明）：

| 算法候选 | FLOPs（含/不含口径） | 访存量 Bytes | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|---------------------|--------------|---------|-------------|---------------------|
| 基线（融合单 kernel） | 2·512³ ≈ 2.68e8（GEMM MAC，Cube）+ ~1.3e6（silu 向量 ops，5 op/elem × 2.62e5 elem；不含 index 计算） | GM 流量 1.5MB（A 读 0.5 + B 读 0.5 + C 写 0.5；L2 128MB ≫ 1.5MB 工作集，tile 重读在 L2 吸收）；L1 装载 4MB（A/B 面板各重读 4×，经 MTE2） | GM 级 A/B/C 各 1 遍；L1 级 A/B 各 4 遍（tile 重读） | 片上：L0C 64KB + UB 48KB（零 GM 中间量） | 16 blocks ×（1 AIC + 2 AIV），tile 相互独立，无跨核归并 |
| 候选 3（分离两 kernel） | 同上（FLOPs 不变） | GM 流量 2.5MB（+Z 写 0.5MB + Z 读 0.5MB，**+67%**）；另 +1 次 kernel launch（开销未实测，assumption：µs 量级） | Z 多写 1 遍 + 多读 1 遍 | GM 中间量 Z 512KB（fp16） | 两 kernel 串行，无重叠；AIC/AIV 各自独占时段，CV 无法流水 |
| 候选 1/2（公式变体） | 与基线同结构，仅 epilogue 每元素 op 数变化（候选 1 省 1 乘；候选 2 纸面省 1 除加 2 乘） | 同基线 | 同基线 | 同基线 | 同基线 |

R3 小结：融合结构对分离结构的优势在**访存口径**（GM 流量 −40%，2.5MB→1.5MB）与**遍数口径**（消除 Z 的 GM 往返两遍）——FLOPs 口径两者相同（融合不省计算）；512³ 规模下 launch 开销与 L2 失效的相对占比高，融合同时消除两者。候选 1/2 与基线的差异在计算口径常数项，交 §1.6.1 量化。

**R4 硬件亲和性评估**（逐候选对照检查清单；常数引用 pattern-library/constants.md，A5 冷启动期全部 status=assumption，容量按 90% 折扣）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（融合 mixed） | MAC 密集段 → Cube（L1-input NT 路径）；silu 逐元素/超越函数 → Vector（AIV）✓ | L1 128KB（含 2 级流水）≤ 460KB；L0C 64KB ≤ 230KB；UB 48KB（×2 版本 96KB）≤ 194KB（CONST-capacity-a5，90% 折扣）✓ | M=N=K=512 均 16 倍数；TILE_K=128，512%128=0 且 128%16=0（CONST-gemm-dim-constraints）✓；dual_copy M-split extent 128 为偶 ✓；N 轴 128 fp32=2×2048b、fp16=1×2048b，0 尾 lane ✓ | 全静态（无动态轴）✓ | K 维 T.Pipelined(num_stages=2)；Cube 段与 Vector epilogue 经 mixed 结构跨 tile 重叠 ✓；无跨核归并 ✓ | ✅ 选定 |
| 候选 3（分离两 kernel） | 各 kernel 单元匹配也成立（GEMM 纯 Cube / silu 纯 Vector） | 同样满足 | 同样满足 | 同样满足 | CV 无法流水（两 kernel 串行），GM 往返 +67%；对应 BP_redundant_work_or_roundtrip 模式（多小 kernel 经 GM 传递中间结果） | ❌ R3 访存/遍数劣势 |
| 候选 1（直除） | 同基线 | 同基线 | 同基线 | 同基线 | 同基线 | ✅（在基线之上采纳，探针实测 0.902×，见 §1.6.1） |
| 候选 2（tanh） | Vector 段无 simd 域 vtanh（tilelang/ascend/language/simd.py 无该 intrinsic），仅 SimtVF 标量路径（src/ascend/codegen/intrin_rule_ascend.cc:57 已注册）——代价未知 | 同基线 | 同基线 | 同基线 | 同基线 | ⚠️ 等价性 PASS 但探针实测 1.80× 劣化 → 否决（§1.6.1） |

**调研结论**: 选定算法族 = **融合单 kernel 的 mixed 结构（基线）+ silu 直除公式（候选 1）**。关键依据：R3 表中融合对分离的 GM 流量 −40%（2.5MB→1.5MB）与遍数优势（消除 Z 往返两遍）、R4 表中融合结构全项亲和（NT 路径 / 容量 / 对齐 / 静态边界 / CV 流水全 ✓）且无跨核归并；候选 1 在等价性机器验证通过（§1.6.1）与设计期探针实测 0.902×（epilogue 向量段）支持下采纳。与基线的结构差异一句话：算法结构与基线融合形态相同，仅 silu 公式由"乘倒数"改为"直除"，并由探针否决了纸面更优的 tanh 形式（实测 1.80× 劣化，证明标量 tanh lowering 代价远超 CONST-valu-model 的 special_func 8 ops/cycle 假设档位）。调研范围：algorithm-candidates.md 命中条目（ALG-elementwise/ALG-attention/ALG-gemv/ALG-transpose）全部评估、恒等变形空间自查（R1 表 6 候选）、examples/ascend 五个同类实现（example_gemm.py / deepgemm epilogue 与 kmn kernel / example_rmsnorm.py / example_vecadd.py）、pattern-library（constants.md / bottleneck-patterns.md 两条相关模式）、结构分析（R2）。split-K 与 exp2 换底的弃选依据见 R1 表。

**设计期估算（D-2）**: 估算下界(相对): **max(流量 ≈2.6e3 cyc, 计算 ≈2.0e3 cyc〔Cube〕/ ≈2.1e3 cyc〔Vector 双 AIV〕, 容量 ≈3.1e2 cyc) —— MTE2 流量项与 Cube 计算项同量级、流量项边际主导**（流量：L1 装载 4MB（A/B 面板各 4× 重读）÷ 16 busy AIC × 100 B/cycle/AIC ≈ 2621 cyc；计算：Cube 每块 2·128·128·512 ÷ 8192 ops/cycle/AIC = 2048 cyc，Vector 直除式 0.2607 cyc/elem × 16384 elem ÷ 2 AIV ≈ 2136 cyc；容量：L0C→UB dual_copy 每 tile 64KB ÷ 256 B/cycle + base 55 ≈ 311 cyc/tile，可被下一 tile Cube 段覆盖；引用 CONST-mte-bandwidth-model / CONST-cube-throughput-model / CONST-valu-model / CONST-capacity-a5，全部 status=assumption——A5 冷启动期 assumption 基估算、常数待标定、容量按 90% 折扣、只做相对量级比较；Vector 项与 Cube 项相当 → CV 重叠质量与 epilogue 向量段效率〔§1.6.1 直除式收益〕对端到端显著）。协议见 .agents/skills/_shared/standards/hardware-cost-model.md §2。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（融合 mixed 结构）的 epilogue 公式。所有对照经 verify_equiv.py 机器验证（examples/pilot/gemm_silu/verify_equiv.py，torch CPU，fp64 参照 + fp32 运算模拟 + fp16 出口三口径；随机 N(0,√512)（z 实际分布尺度）与 N(0,1) 各 1e6 样本 + 31 角点，角点含 ±inf/NaN/subnormal/±88（exp fp32 溢出边界）/±1e4）。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | silu 直除形式 | z·(1/(1+exp(−z)))（乘倒数） | z/(1+exp(−z))（直除） | 数学恒等：a·(1/b) ≡ a/b；差异仅舍入路径——fp64 参照两式 max&#124;Δ&#124; = 3.55e-15；fp32 运算模拟 max&#124;Δ&#124; ≤ 9.5e-7（≈1-2 ulp @ &#124;s&#124;~1-17）；fp16 出口 max ulp ≤ 14（0.7% 相对，< 1e-2 容差） | 每元素省 1 次乘（VALU 51 档，CONST-valu-model）；探针实测 epilogue 向量段 **0.902×**（239.23→215.67 µs，64 次迭代 × 128×128 fp32 单 AIV，见下方探针表） |
| 2 | tanh 形式消除除法（**探针否决，不采纳**） | z·(1/(1+exp(−z))) | 0.5·z·(1+tanh(0.5·z)) | 数学恒等：σ(z) = (1+tanh(z/2))/2；verify_equiv 全用例 EQUIV_PASS（fp64 max&#124;Δ&#124; = 7.11e-15，fp16 出口 max ulp ≤ 24，违反率 0） | 纸面：消除 div（13 档）换 2 mul（51 档），模型预测 epilogue −22%；**实测相反**：探针 388.42 µs，为直除形式 215.67 µs 的 **1.80×**——标量 tanh 的 Ascend lowering 代价远超 special_func 8 ops/cycle 假设（推断为软件例程而非 SFU 直连）。等价性成立但收益为负，否决 |
| 3 | epilogue 单遍融合 + cast 后置 | silu 与 cast 分离（Z 落 GM，二次遍历施加 silu 再 cast，即 R1 候选 3 结构） | 同一 SimtVF 遍内读 fp32 acc_ub → 直除式 silu → 写 fp16 out_ub | 结构恒等：同公式、同 fp32 精度路径，仅消除搬运与二次遍历；cast 时机为语义决策（§1.4：silu 于 fp32 累加器上施加）而非等价变换，golden（§8.1）同语义 | 省 Z 的 GM 往返 2×512KB = 1MB（GM 流量 −40%，2.5MB→1.5MB，R3 表）+ 1 次 kernel launch；消除 z 的 fp16 预舍入误差（z 可达 ~10² 量级） |

**verify_equiv.py 结果表**（n = 1e6×2 随机 + 31 角点；PASS 判据：fp16 出口在 §8.2 容差 atol=rtol=1e-2 下违反率 = 0 且有限输入上无 NaN/inf 分歧）：

| 候选式（vs 基线 z·(1/(1+exp(−z)))） | fp64 参照 max&#124;Δ&#124; | fp32 运算模拟 max&#124;Δ&#124; | fp16 出口 max ulp | 违反率 (1e-2) | NaN/inf 分歧 | 结论 |
|----------------------------------------|--------------------------|------------------------------|--------------------|---------------|--------------|------|
| 直除 z/(1+exp(−z)) | 3.55e-15 | 9.54e-7 | 14 | 0.00e+00 | 0 | **EQUIV_PASS** |
| tanh 0.5z(1+tanh(0.5z)) | 7.11e-15 | 1.91e-6 | 24 | 0.00e+00 | 0 | **EQUIV_PASS** |

注：①角点含 z=±1e4/±inf/NaN——fp64 列的 nan 来自 NaN 角点自身（NaN−NaN），有限输入上全部分歧为 0；②z ≤ −88.7 时 exp(−z) 在 fp32 溢出为 inf，直除形式与基线形式一致地退化为正确极限 −0.0（1/inf=0 → z·0 与 z/inf 同为 −0.0），角点用例已覆盖；③近零/次正规结果区的相对 ulp 指标会放大（≤149 ulp @ &#124;s&#124;~1e-37 量级），绝对误差 ≤ 2e-6，对 fp16 出口无影响。

**设计期探针（D-3）结果**（examples/pilot/gemm_silu/history_version/design_probe_silu_form.py；UB 驻留 128×128 fp32 × 64 次原地迭代，单 AIV SimtVF 路径，do_bench(msprof)；Ascend950PR_9579 / tilelang-fork fc32f47c / CANN 9.x，2026-10-16）：

| 形式 | 延迟 (µs) | 相对比值（基线=1.00） |
|------|-----------|------------------------|
| 基线 z·(1/(1+exp(−z))) | 239.23 | 1.000 |
| 直除 z/(1+exp(−z)) | 215.67 | **0.902** |
| tanh 0.5z(1+tanh(0.5z)) | 388.42 | 1.624（= 直除的 1.80×） |

**优化结论**: 采纳 2 项——①silu 直除形式（z/(1+exp(−z))，等价性 EQUIV_PASS + 探针实测 epilogue 向量段 0.902×）；②epilogue 单遍融合 + cast 后置（GM 流量 −40% + 消 z 预舍入）。tanh 形式（#2）等价性 EQUIV_PASS 但被探针实测代价否决（1.80× 劣化，标量 tanh lowering 非 SFU 档位），不采纳。**优化后公式（供 §3.1 使用）：z_ij = Σ_k a_ik·b_jk（fp16 输入，fp32 累加，Cube 段）；c_ij = fp16( z_ij / (1 + exp(−z_ij)) )（fp32 计算，Vector 段单遍完成 silu 与 cast，cast 后置）。**

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点选定算法实现方案中全部循环与标量计算点。替代方案所用 API 均有仓内佐证（examples/ascend/ 或 tilelang/ascend/language/ 源码）。

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | silu 逐元素计算（16384 elem/tile） | 标量循环 exp/div/mul | T.SimtVF(threads=128) + T.Parallel(64,128)——编译器自动向量化；T.exp 标量 intrinsic 已注册 Ascend lowering（src/ascend/codegen/intrin_rule_ascend.cc:47，REGISTER_ASCEND_INTRIN("exp")，vectorized dtype 支持任意 lane 数）；SimtVF+T.Parallel 含特殊函数的模式佐证 examples/ascend/example_rmsnorm.py（54-75 行）（T.rsqrt:66） | ✅ | — |
| 2 | fp32→fp16 cast | 逐元素独立 cast 遍 / 独立 cast copy | 融入 #1 的同一 T.Parallel 遍（fp32 值赋给 fp16 buffer 隐式 cast）；SimdVF 变体 T.simd.vcvt 佐证 examples/ascend/deepgemm/kernels/epilogue.py（22 行） | ✅ | — |
| 3 | K 维分块累加循环（K_TILES=4 迭代） | 逐迭代标量控制 | T.Pipelined(K_TILES, num_stages=2) + T.gemm——循环体是 Cube 矩阵算子而非标量计算，Cube MAD 单元即该累加的矩阵级向量化本体（T.gemm 佐证 examples/ascend/example_gemm.py（71-74 行）） | ✅（结构级） | — |
| 4 | tile_idx → (m_tile, n_tile) 映射 | 标量整除/取模 | 无逐元素等价 API | ❌ | block 级索引/任务映射计算：每 block 每 tile 执行 1 次（16 blocks × 1 tile），非逐元素热点（理由类别：block 索引/任务映射） |
| 5 | GM 区域偏移（m0=TILE_M·m_tile 等） | 标量乘加 | 无 | ❌ | 同 #4，block 级元数据，每 tile 一次（理由类别：block 索引/任务映射） |
| 6 | host 侧 wrapper（shape 契约校验、编译缓存、输入只读视图） | host Python | 不在 kernel 内 | ❌（不适用） | host 侧元数据计算（理由类别：host 元数据；且按 ascend-constraints.md §4 host 禁改输入内容，仅允许 stride/shape 视图操作） |

**向量化结论**: 逐元素计算 100% 向量化（#1 silu、#2 cast、#3 K 累加分别为 SimtVF 向量化 / 同遍融合 / Cube MAD 本体，API 均有佐证）；保留 3 处标量/循环点（#4/#5 block 级索引与任务映射、#6 host 元数据），理由类别均为"block 索引/host 元数据"，无逐元素热点残留、无 tile 级顺序依赖（tile 相互独立）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（GM 侧 A [M,K] 行主序、B [N,K] 行主序、C [M,N] 行主序）；核内布局与向量化轴是设计变量。本算子 ≥2 个非 batch 维度，必须完成本节分析。Cube 侧 fractal 布局（ND2NZ）由 InsertNd2Nz pass 自动插入（禁止手写 NZ staging，AGENTS.md）；本节决策集中在 **epilogue 侧 UB 布局与向量化轴** 及 **epilogue 归属结构**。

**轴质量评分维度**（对主选轴）：

| 评分项 | 判定 |
|--------|------|
| 整除性 | N 轴 128 元素：fp32 = 2×2048b 向量（64 lane），128%64=0 ✓；fp16 = 1×2048b（128 lane），128%128=0 ✓（CONST-valu-model：2048b = 128×fp16 / 64×fp32） |
| 尾 lane 浪费率 | 0%（128 恰为两种 dtype 向量宽整数倍） |
| 累加链形态 | epilogue 无累加链（逐元素映射）；K 累加在 Cube 侧（L0C），不占向量轴 |
| repack 代价 | 无需 repack：L0C→UB dual_copy 直出 ND 行主序（M-split 沿 M，N 保持连续）；silu 轴对称，任何重排零收益 |
| UB 容量影响 | acc_ub 32KB + out_ub 16KB = 48KB（×2 multibuffer 版本 96KB）≤ 194KB（216KB×90%，SimtVF 场景，CONST-capacity-a5）✓ |

**候选矩阵**（全部枚举，弃选给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | UB ND 行主序（dual_copy M-split 自然产出），N 轴向量化 | N（行内连续） | 无 | 基线；0 尾 lane；无重排开销；双 AIV 分担向量段（2136 cyc ≈ Cube 2048 cyc，CV 均衡） | ✅ 主选 |
| 2 | UB 内转置布局，M 轴向量化 | M（跨行 stride 128） | 需通用转置——T.copy 的 transpose= 仅特定路径（GM→L1 dn2nz / L1→L0 装载请求转置），非通用转置（ALG-transpose 条目、tilelang-ascend skill §6.1）；VF 内交错 vintlv/vdintlv 仅双向量交错 | silu 逐元素轴对称，转置零收益；跨行跨步阻碍向量化；repack 纯代价 | ❌ |
| 3 | L0C 直出 GM（无 UB epilogue：T.copy(res, C) 携带 fp32→fp16 转换） | — | 无 | 不可行：L0C 出口路径仅支持 dtype 转换（fixpipe 量化前处理），无超越函数能力；且 dav-950 的 L0C→UB dual_copy 显式禁止类型转换（tilelang/ascend/language/copy_op.py 的 ValueError 分支）——silu 必须在 VF 内计算 | ❌ 语义不可行 |
| 4 | epilogue 单 AIV（deepgemm sids=1 模式：plain T.copy L0C→UB → VF 后处理 → plain T.copy UB→GM） | N | 无 | 仓内最强先例（examples/ascend/deepgemm/kernels/epilogue.py（59-72 行）、examples/ascend/deepgemm/kernels/kmn/bf16_gemm.py（61,113 行））；代价：单 AIV 承担全部向量段 16384×0.2607 ≈ 4271 cyc > Cube 2048 cyc → epilogue 成瓶颈（对应 BP_dual_copy_utilization 模式：单 AIV epilogue 负载不均） | ❌ 主选不用；保留为 §1.6.4 结构 fallback |

**布局决策结论**: 选定方案 #1：epilogue UB 采用 ND 行主序（TILE_M/2 × TILE_N 的 fp32/fp16 半 tile，每 AIV 一份实例），向量化轴 = **N（行内连续轴）**，无 repack；L0C→UB 与 UB→GM 均走 dual_copy M-split（双 AIV 分担向量段与写回）；Cube 侧 ND2NZ 分形由 pass 自动插入。该结论同步落入 §3.3 伪代码与 §6 循环结构（元素级循环的内层向量维 = N 轴）；弃选方案的量化理由见候选矩阵（#2 repack 纯代价 / #3 语义不可行 / #4 单 AIV epilogue 成瓶颈 4271 cyc）。

#### 1.6.4 结构备选与降级路径（风险 fallback）

主选 epilogue 结构（dual_copy L0C→UB → 均匀 SimtVF 元素段 → dual_copy UB→GM）中，"两个 dual_copy 之间插入 SimtVF 元素段"在仓内无直接先例（examples/ascend/example_gemm.py（75-79 行） 的 MIXED epilogue 为纯搬运；deepgemm 的元素变换走 plain T.copy 链）。这是**可行性风险**（AutoSchedule 对该元素段的所有权指派行为待 Stage 3 验证），不是成本不确定（无需实验裁决三件套）。降级路径按序：

- **Fallback F1（deepgemm 式单 AIV 链）**：plain T.copy(L0C→UB) → SimdVF/SimtVF 后处理（照搬 examples/ascend/deepgemm/kernels/epilogue.py（59-72 行） 的 loop-body 隔离模式：`for _ in range(1)` + `T.assume_no_conflict`）→ plain T.copy(UB→GM)。结构代价：向量段 4271 cyc 由单 AIV 承担 → epilogue 主导（估算整体劣化 ~1.5-2×，assumption 常数）。
- **Fallback F2（显式 MixedKernel + sid guard）**：T.MixedKernel(16, sids=2) + sid 条件分派（AIV0/AIV1 各认领半 tile 的独立 T.copy 与 VF 段），规避均匀控制流要求；代价是实现复杂度与手动分派的负载不均风险。
- 触发条件：Stage 3 编译失败 / 结果错误 / npusim 显示 epilogue 段严重串行化时按 F1→F2 顺序降级，并回写本节与 §9 风险表。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: mixed（auto_schedule: yes）

### 2.2 选型理由

- 算子特征（Phase 1 判定）：matmul 主导（2.68e8 MAC FLOPs，Cube）+ 逐元素非线性后处理（silu，Vector）→ 决策树"matmul + element-wise 后处理 → mixed 混合核"（tilelang-op-design skill decision-tree.md §2）；与用户指定的 mixed 一致。
- mixed 的收益已在 §1.6.0 R3/R4 量化：融合消除 Z 的 GM 往返（−40% GM 流量）并使 Cube 段与 Vector epilogue 跨 tile 流水重叠（向量段 2136 cyc ≈ Cube 2048 cyc，重叠后接近 max 而非 sum）。
- Kernel 入口形态：**T.Kernel(NUM_BLOCKS) 纯 kernel 形式 + codegen 自动判定 mixed**——本算子不需要 sid ID、不需要显式控制 AIV 数（sids=2 为默认行为），按 tilelang-ascend skill §3.1/§3.2 的决策（"If neither the ID nor explicit count is needed, use T.Kernel; the compiler handles core splitting and the appropriate T.dual_copy lowering"）。仓内先例：examples/ascend/example_gemm.py（60,75-79 行）（T.Kernel + Cube/Vector 工作共存 + dual_copy epilogue，auto 判定 mixed）。T.MixedKernel(16, sids=2) 为等价显式形式，仅在需要 sid guard 时切换（§1.6.4 F2）。
- epilogue VF 模式：**SimtVF**（默认；AGENTS.md：SimdVF 仅用于证实的热点）。本设计的公式级裁决已在 SimtVF 路径用探针完成（§1.6.1）；SimdVF 变体（T.simd.vexp/vdiv/vcvt，tilelang/ascend/language/simd.py:792,860 + examples/ascend/deepgemm/kernels/epilogue.py（22 行） 先例）保留为 Stage 4 的热点验证项——注意 SimdVF 域无 vtanh，若走 SimdVF 则公式被锁定为直除形式（该形式本就是主选，无冲突）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| VF 入口 | T.Kernel(16)（一维，无 threads 参数；auto 检测 __mix__）+ Cube 段 T.copy/T.gemm + Vector 段 T.SimtVF(threads=128) + T.Parallel(64,128) 元素级 |
| 内存分配 | Cube 侧 T.alloc_l1（L1）/ T.alloc_l0c（L0C 累加器）；Vector 侧 T.alloc_shared（UB：acc_ub fp32 / out_ub fp16） |
| 同步 | auto_schedule: yes——AutoSchedule 自动插 flag 对（AssignCore/InsertSync pass），数据依赖边界见 §7.2；无手动 flag |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的优化后公式。每步优先向量化映射，与 §1.6.2 的向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | x_tile = A[m0:m1, k0:k1]；w_tile = B[n0:n1, k0:k1] | A/B 的 GM→L1 tile 搬运（ND2NZ 自动） |
| 2 | z = Σ_k x_tile·w_tile^T | fp32 累加 GEMM（L1-input NT 路径，clear_accum 首拍清零） |
| 3 | z_ub = z（M-split 半 tile） | L0C→UB 搬运（fp32 原样驻留） |
| 4 | s = z_ub / (1 + exp(−z_ub)) | silu 直除形式，fp32，SimtVF 向量化 |
| 5 | c = fp16(s) | cast 融入步骤 4 同一遍，写 fp16 UB |
| 6 | C[m0:m1, n0:n1] = c（M-split） | UB→GM 写回 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | GM→L1 tile 搬运 | T.copy(A[m0:m1, k0:k1], x_l1) / T.copy(B[n0:n1, k0:k1], w_l1) | 区域切片；ND2NZ 由 InsertNd2Nz pass 自动插入，禁止手写 NZ | mixed（AIC/MTE2） |
| 2 | fp32 累加 GEMM | T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0)) | L1-input NT：trans_A=False、trans_B=True；权重 GM 布局 [N,K] | mixed（AIC/Cube） |
| 3 | L0C→UB（半 tile） | T.dual_copy(res, acc_ub) | M-split（src 128×128 → dst 64×128），fp32→fp32（dav-950 此路径禁止类型转换，本设计恰好要求 fp32 驻留） | mixed（AIC 发射/双 AIV 接收） |
| 4 | silu 直除式 | with T.SimtVF(threads=128): for i, j in T.Parallel(64, 128): out_ub[i,j] = z/(1.0 + T.exp(−z))，z = acc_ub[i,j] | T.exp 标量 intrinsic（Ascend lowering 已注册）；fp32 计算 | mixed（AIV ×2） |
| 5 | cast fp16 | 融入步骤 4（fp32 表达式赋值 fp16 buffer 隐式 cast） | — | mixed（AIV ×2） |
| 6 | UB→GM 写回 | T.dual_copy(out_ub, C[m0:m1, n0:n1]) | UB→GM M-split（per-AIV 软件拷贝，fp16→fp16 无转换） | mixed（AIV ×2） |

### 3.3 计算伪代码

> Tensor 类型按 examples/ascend/ 形态：dtype 作为第二个位置参数（T.Tensor((M, N), "float16")），不用 dtype= 关键字。

```python
import tilelang.ascend.language as T

M_DIM = N_DIM = K_DIM = 512
TILE_M = TILE_N = TILE_K = 128          # 见 §5.2 取值依据
M_TILES = N_TILES = 4                   # 512 // 128
K_TILES = 4                             # 512 // 128
NUM_BLOCKS = 16                         # M_TILES * N_TILES，一维 grid
NUM_STAGES = 2

@T.prim_func
def gemm_silu(
    A: T.Tensor((M_DIM, K_DIM), "float16"),   # [M, K]
    B: T.Tensor((N_DIM, K_DIM), "float16"),   # [N, K]（NT 约定权重布局）
    C: T.Tensor((M_DIM, N_DIM), "float16"),   # [M, N]
):
    with T.Kernel(NUM_BLOCKS) as bx:          # 一维 grid，无 threads 参数；Cube+Vector 工作共存 → auto mixed
        res = T.alloc_l0c((TILE_M, TILE_N), "float32")           # fp32 累加器（64KB）
        x_l1 = T.alloc_l1((TILE_M, TILE_K), "float16")           # 32KB ×stages
        w_l1 = T.alloc_l1((TILE_N, TILE_K), "float16")           # 32KB ×stages
        acc_ub = T.alloc_shared((TILE_M // 2, TILE_N), "float32")  # dual_copy M-split 半 tile，每 AIV 一份实例
        out_ub = T.alloc_shared((TILE_M // 2, TILE_N), "float16")
        for tile_idx in T.Persistent([M_TILES * N_TILES], NUM_BLOCKS, bx):
            m_tile = tile_idx // N_TILES                        # block 级标量索引（§1.6.2 #4）
            n_tile = tile_idx % N_TILES
            m0, n0 = m_tile * TILE_M, n_tile * TILE_N           # block 级标量（§1.6.2 #5）
            for kt in T.Pipelined(K_TILES, num_stages=NUM_STAGES):
                T.copy(A[m0:(m_tile + 1) * TILE_M, kt * TILE_K:(kt + 1) * TILE_K], x_l1)  # ND2NZ 自动
                T.copy(B[n0:(n_tile + 1) * TILE_N, kt * TILE_K:(kt + 1) * TILE_K], w_l1)
                T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))
            # ---- epilogue：L0C → UB(M-split 双 AIV) → silu+cast(SimtVF) → GM(M-split) ----
            T.dual_copy(res, acc_ub)                            # fp32 → fp32（无转换）
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(TILE_M // 2, TILE_N):    # 每 AIV 处理自己的半 tile 实例
                    z = acc_ub[i, j]
                    out_ub[i, j] = z / (1.0 + T.exp(-z))        # fp32 计算 + 隐式 cast fp16
            T.dual_copy(out_ub, C[m0:(m_tile + 1) * TILE_M, n0:(n_tile + 1) * TILE_N])
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|----------|---------|
| T.Kernel(N)（一维，无 threads） | examples/ascend/example_gemm.py（60 行）；tilelang-ascend skill §2.1 | 仓内已验证形态 |
| T.Persistent([tiles], N, bx) | examples/ascend/example_gemm.py（69 行） | 仓内已验证形态 |
| T.Pipelined(K_TILES, num_stages=2) | examples/ascend/example_gemm.py（71 行） | 仓内已验证形态 |
| T.alloc_l1 / T.alloc_l0c / T.alloc_shared | examples/ascend/example_gemm.py（63-66 行） | 仓内已验证形态 |
| T.copy（GM→L1，ND2NZ 自动） | examples/ascend/example_gemm.py（72-73 行） | 仓内已验证形态 |
| T.gemm(..., transpose_B=True, clear_accum=) | examples/ascend/example_gemm.py（74 行）；tilelang/ascend/op/gemm/gemm_mad.py | 仓内已验证形态 |
| T.dual_copy（L0C→UB M-split；UB→GM M-split） | examples/ascend/example_gemm.py（76-77 行）；tilelang/ascend/language/copy_op.py（M-split 语义与 dtype 约束源码） | 仓内已验证形态（元素段插入为设计新组合，风险见 §1.6.4/§9） |
| T.SimtVF(threads=) + T.Parallel | examples/ascend/example_rmsnorm.py（54-56 行）；examples/ascend/example_vecadd.py | 仓内已验证形态 |
| T.exp（标量 intrinsic，Ascend lowering） | src/ascend/codegen/intrin_rule_ascend.cc:47（REGISTER_ASCEND_INTRIN("exp")，vectorized dtype 任意 lane） | 源码注册确认；本任务探针（design_probe_silu_form.py）已在真机编译运行通过 |
| T.simd.vexp / vdiv / vcvt（SimdVF 备选） | tilelang/ascend/language/simd.py:792,860；examples/ascend/deepgemm/kernels/epilogue.py（22 行） | 源码 + 仓内先例（本设计非主选路径） |

### 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 一维 Kernel / 无 threads 参数 | Yes | T.Kernel(16)；多维输出网格（4×4 tiles）自行一维映射：m_tile = tile_idx // 4、n_tile = tile_idx % 4（skill §10 陷阱 3） |
| GEMM C0 分形对齐（M/N 16 倍数；K % TILE_K_SUB == 0 且 TILE_K_SUB % C0 == 0） | Yes | M=N=K=512 全 16 倍数；TILE_K=128：512%128=0、128%16=0（fp16 C0=16，CONST-gemm-dim-constraints）——契约 shape 全对齐，无 padding 需求（非契约 shape 的拒绝行为见 §8.4） |
| L1-input GEMM 仅 NT（trans_A=False, trans_B=True），权重 GM [N,K] | Yes | B 按 [N,K] 契约输入，直接满足；T.gemm(..., transpose_B=True) |
| L0C→UB dual_copy 禁止类型转换（dav-950） | Yes | acc_ub 声明 fp32 与 L0C 一致（cast 在 VF 段完成）——约束恰好强制了最优精度路径（cast 后置） |
| L0C 容量 256KB / UB 248KB（SimtVF 216KB）/ L1 512KB | Yes | 64KB / 48KB(×2 版本 96KB) / 128KB(含 2 级流水)，全部 ≤ 90% 折扣预算（§4.5/§5.3） |
| OOB padding（pad_value/data_select 互斥、仅 GM→UB） | No | 契约 shape 全对齐、无尾块，不使用 padding 路径 |
| 静态循环边界 | Yes | 全静态：K_TILES=4、tiles=16、元素域 64×128（核内无动态边界） |
| 核数无编译期静态常数 | Yes | §5.5 实查记录（cube 28 / vector 56），grid=16 单波 |
| flag id 预算（核内 8 / 跨核 16） | No | auto_schedule: yes，无手动 flag（§7） |
| host 侧禁改输入内容 | Yes | wrapper 仅做 shape 契约校验与只读视图，无任何数据预处理/后处理 |

### 3.5.2 参考实现差异说明

无外部 GPU 参考实现（需求输入为数学公式与 I/O 契约，非源码迁移）。PyTorch 侧仅作为 golden 参照（§8.1），不构成 API 映射输入；GPU 生态（CUTLASS epilogue fusion 等）仅作算法族调研的思路来源（§1.6.0 R1 已对照），其 API 不在本项目使用。

### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| examples/ascend/example_gemm.py | 高度相似（结构母版） | persistent GEMM + T.Pipelined K 流水 + MIXED dual_copy epilogue 的完整骨架（60-79 行）；半 tile UB 缓冲模式（66 行 temp = TILE_M//2） |
| examples/ascend/deepgemm/kernels/epilogue.py | 高度相似（epilogue 元素变换先例） | mixed kernel 中 Vector epilogue 变换（scale + vcvt cast）的 loop-body 隔离模式（59-72 行：for range(1) + T.assume_no_conflict）——§1.6.4 F1 fallback 与 AutoSchedule 所有权处理依据 |
| examples/ascend/deepgemm/kernels/kmn/bf16_gemm.py | 相似 | MixedKernel(sids=1) + scheduler + epilogue 调用的完整组织方式（61-124 行）；T.annotate_buffer_versions 的 multibuffer 版本管理（74-82 行） |
| examples/ascend/example_rmsnorm.py | 参考 | SimtVF + T.Parallel + 特殊函数（T.rsqrt）元素级向量化模式（54-66 行） |
| examples/ascend/example_vecadd.py | 参考 | simt 元素向量化与 T.Pipelined 的基础模式 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| A | (512, 512) | float16 | 输入矩阵 [M, K]，行主序，静态 |
| B | (512, 512) | float16 | 权重矩阵 [N, K]（NT 约定布局），行主序，静态 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| C | (512, 512) | float16 | 输出矩阵 [M, N]，行主序，静态；silu 于 fp32 累加器上施加后 cast |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| x_l1 | (128, 128) | float16 | L1（CBuf） | A 的 K 分块 tile（ND2NZ 后分形驻留，×num_stages=2 双缓冲） |
| w_l1 | (128, 128) | float16 | L1（CBuf） | B 的 K 分块 tile（同上） |
| res | (128, 128) | float32 | L0C | fp32 GEMM 累加器（clear_accum 首拍清零） |
| acc_ub | (64, 128) | float32 | UB | L0C→UB dual_copy M-split 的半 tile 落点（每 AIV 一份实例），silu 的 fp32 输入 |
| out_ub | (64, 128) | float16 | UB | silu+cast 输出（每 AIV 一份实例），dual_copy 写回 GM 的源 |

### 4.4 内存搬运路径

```
GM[A] --T.copy(MTE2, ND2NZ 自动插入)--> L1[x_l1] --|
GM[B] --T.copy(MTE2, ND2NZ 自动插入)--> L1[w_l1] --|--T.gemm(NT, fp32 累加)--> L0C[res]
                                                                        |
                     T.dual_copy(M-split, FixPipe 双目的, fp32→fp32)   |
                                                                        v
                                              UB[acc_ub 64×128 fp32]（每 AIV 一份）
                                                                        |
                                     SimtVF: s = z/(1+exp(-z)); cast fp16
                                                                        v
                                              UB[out_ub 64×128 fp16]（每 AIV 一份）
                                                                        |
                     T.dual_copy(M-split, per-AIV 软件拷贝, MTE3)       |
                                                                        v
                                                                 GM[C]（128×128 tile）
```

### 4.5 UB 内存预算

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| acc_ub | (64, 128) | float32 | 32768 |
| out_ub | (64, 128) | float16 | 16384 |
| 小计（单版本） | | | 49152 |
| ×2（epilogue multibuffer 版本，跨 tile 重叠） | | | 98304 |
| **总计** | | | **98304 B ≈ 96KB** / UB 容量 216KB（SimtVF 场景，`CONST-capacity-a5`，A5，按 90% 折扣预算 194KB）✓ 余量 98KB |

（L1 预算：x_l1 + w_l1 = 64KB × num_stages 2 = 128KB ≤ 460KB〔512KB×90%〕✓；L0C 预算：res 64KB ≤ 230KB〔256KB×90%〕✓；AutoSchedule multibuffering 版本数已计入 UB 项，L1/L0C 的版本膨胀余量充足。）

### 4.6 动态轴定义

无（契约 shape 全静态 512×512×512；wrapper 校验输入 shape 严格等于契约值，非契约 shape 拒绝并报错，见 §8.4）。

### 4.7 JIT 配置

```python
import tilelang

@tilelang.jit(
    target="ascend",          # 规范入口（别名 asc；pto AOT 线不在 conductor 范围）
    out_idx=[2],              # C 为第 3 个参数
)
def gemm_silu_kernel(...):
    ...
# 等价编译式：tilelang.compile(gemm_silu_prim_func, target="ascend", out_idx=[2])
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 混合（Cube GEMM 主导 + Vector 逐元素 epilogue，CV 融合）

**判定依据**: 算子含 matmul（2.68e8 MAC FLOPs → Cube/L1/L0A/L0B/L0C 路径）+ 逐元素非线性后处理（silu → Vector/UB 路径），需核内 CV 协作（Phase 1 判定）。

### 5.2 Block 划分

```python
block_M = 128   # 依据：L0C 预算 128×128×4B=64KB ≤ 230KB（90% 折扣）；dual_copy M-split 需偶数 extent；
                #       epilogue 向量段均衡（双 AIV 2136 cyc ≈ Cube 2048 cyc）；M=512/128=4 整除
block_N = 128   # 依据：N 轴 128 fp32=2×2048b / fp16=1×2048b，0 尾 lane；UB 半 tile 32KB 预算友好；N=512/128=4 整除
block_K = 128   # 依据：512%128=0 且 128%16=0（C0=16，CONST-gemm-dim-constraints）；K_TILES=4 流水粒度适中；
                #       L1 预算 (32+32)KB×2 stages = 128KB ≤ 460KB（TILE_K=256 亦可行，Stage 4 扫描项）
block_num = (512 // 128) * (512 // 128) = 16   # 一维 T.Kernel(16)，核内 tile_idx → (m_tile, n_tile) 映射
```

grid 与 tile 的权衡记录：16 blocks 单波占用 16/28 AIC（57% Cube 利用率）。备选 tile 形态已按 §1.6.0 估算口径比较——128×64（32 blocks，2 波尾 4/28，L1 装载 6MB）与 64×64（64 blocks，persistent 2.3 轮，L1 装载 8MB）以更高的 MTE2 重读流量与尾波不平衡换取更多 busy 核，模型估算无净收益（MTE2 项 2621/2301/3010 cyc vs Cube 项 2048/1170/1170 cyc，assumption 常数）；主选 128×128 的依据是最低 L1 装载流量 + 单波无尾波 + 与模板结构（example_gemm.py）最大一致性。grid/tile 扫描列入 Stage 4 调优项（§9）。

### 5.3 约束分析

- **C0 分形对齐**: M=N=K=512 均 16 倍数 ✓；TILE_K=128：512 % 128 == 0 且 128 % 16 == 0（fp16 C0=16）✓（CONST-gemm-dim-constraints）；dual_copy M-split：源 extent M=128 为偶 ✓；UB→GM M-split 沿 M，无 N-split 的 N%32 约束（未用 N-split）。
- **UB 容量**: 96KB（含 ×2 multibuffer 版本）≤ 194KB（216KB×90%，SimtVF 场景，CONST-capacity-a5）✓。
- **L1 容量**: 128KB（含 2 级流水双缓冲）≤ 460KB（512KB×90%）✓。
- **L0C 容量**: 64KB ≤ 230KB（256KB×90%）✓。
- **静态边界**: 核内全部循环边界静态（K_TILES=4、tiles=16、元素域 64×128）✓。

### 5.4 注意事项

- 契约 shape 全 16 对齐且被 block 整除——**无尾块、无 padding 路径**；非契约 shape（含非 16 倍数）在 wrapper 层显式拒绝（§8.4），kernel 不做动态边界处理（静态边界约束）。
- 双 AIV epilogue 的半 tile 实例语义：acc_ub/out_ub 的 (64,128) 分配为每 AIV 各一份实例（dual_copy M-split 语义，tilelang/ascend/language/copy_op.py："Each AIV processes the corresponding half of that tile"），UB 预算按单实例 96KB 计（每 AIV 的物理 UB 独立），元素段在两 AIV 上均匀执行同一控制流、各处理自己的实例——无跨 AIV 数据竞争。
- L2 策略：T.copy 默认 L2 策略（GM→L1 NORMAL_FV / UB→GM NOTALLOC_CI，tilelang-ascend skill §6.1）；工作集 1.5MB ≪ L2 128MB，A/B 面板重读由 L2 吸收——Stage 4 可按 BP_l2_cache_policy 模式扫描 l2_cache_ctrl。

### 5.5 分核策略（grid 划分 + AIC/AIV 分工 + 同步策略）⭐

> 三要素判定标准：.agents/skills/_shared/standards/core-split-strategy.md §1（依据 tilelang-ascend skill §2/§3/§7）。

- **grid 划分（含核数依据）**: num_blocks = (M // block_M) × (N // block_N) = (512/128) × (512/128) = **16**，一维 T.Kernel(16)（T.Persistent([16], 16, bx) 分发，每 block 恰处理 1 个输出 tile）。核数**真机实查**（查询代码与返回值记录）：

  ```python
  import torch
  props = torch.npu.get_device_properties()
  # 实查返回（2026-10-16，本任务执行）：
  # name='Ascend950PR_9579', total_memory=126208MB,
  # cube_core_num=28, vector_core_num=56, L2_cache_size=128MB
  ```

  block 数与核数关系：16 blocks ≤ 28 AIC → **单波**（无多波尾波、无需 persistent 多轮；T.Persistent 保留模板一致性并为 Stage 4 grid 扫描免改结构）；AIC 占用 16/28 = 57%、AIV 占用 16 组 × 2 = 32/56 = 57%（AIC:AIV 物理编组 1:2，CONST-arch-a5）。57% 利用率为 512³ 规模 + 最低重读流量取舍的已知代价（§5.2 权衡记录），Stage 4 裁决。
- **AIC/AIV 分工**: 
  - **AIC（每 block 1 个）**: T.copy GM→L1 ×2/K 迭代（MTE2 + ND2NZ 自动）+ T.gemm ×4（Cube MAD，clear_accum=(kt==0)）+ L0C→UB dual_copy 发射（FixPipe 双目的指令，AIC 侧 pipe）。
  - **AIV（每 block 2 个，sids=2 默认行为；控制流均匀，无 sid guard）**: SimtVF silu+cast 元素段（各处理自己 64×128 半 tile 的 acc_ub/out_ub 实例，N 轴向量化）+ UB→GM dual_copy（per-AIV 软件拷贝按 cthread sid 索引，各写 C 的 64 行）。
  - **dual_copy 所有权**: L0C→UB = 单条硬件双目的指令、两 AIV 协作接收（无独立目的选择）；UB→GM = per-AIV 软件拷贝（独立选择目的，无需均匀控制流约束）。
  - 纯 Vector 段（silu）与纯 Cube 段（gemm）的落位即 §1.4 五步表的落位列；本算子无纯 AIV-only 或 AIC-only 的整 kernel 分工。
- **同步策略（auto|manual）**: **auto_schedule: yes**（默认，用户契约一致）。依赖的数据依赖边界（AutoSchedule 自动插 flag 对覆盖点）：① K 流水内 MTE2→MAD（L1 multibuffer 就绪，T.Pipelined num_stages=2 声明）；② MAD→FixPipe/dual_copy（L0C 累加器就绪，clear_accum 生命周期）；③ dual_copy→SimtVF（acc_ub RAW）；④ SimtVF→MTE3（out_ub RAW）；⑤ AIC↔AIV 组内跨核同步（InsertSync pass）。无手动 T.ascend_set_flag/wait_flag（flag 预算不占用，CONST-arch-a5 核内 8/跨核 16 预算仅作手动方案上限参考）。调度提示使用位置：T.Pipelined（K 维）、T.Persistent（tile 维）；可选 Stage 3/4 调优提示——照搬 examples/ascend/example_gemm.py（74-76 行） 的 unit_flag_ctrl 注解模式（UF_2/UF_3 的 MAD→FixPipe 流水提示），非手动同步。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化（#1/#2/#3）——本表仅 block 级 / tile 级调度循环，无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| tile 维（16 tiles / 16 blocks） | block 级持久调度 | T.Persistent([16], 16, bx) | 每 block 1 tile；保留模板形态与 Stage 4 grid 扫描的免改结构能力 |
| K 维（4 迭代） | tile 级软件流水 | T.Pipelined(K_TILES, num_stages=2) | L1 双缓冲：copy A / copy B / gemm 三段流水（example_gemm.py:71-74 同构） |
| epilogue 元素维（64×128） | 向量化（线程域并行） | T.SimtVF(threads=128) + T.Parallel(64, 128) | §1.6.2 #1/#2：silu+cast 单遍向量化，内层向量维 = N 轴（§1.6.3 主选轴） |
| tile_idx → (m_tile, n_tile) | 标量索引 | tile_idx // N_TILES、% N_TILES | §1.6.2 #4/#5：block 级索引，非热点 |

### 6.2 循环伪代码

```python
with T.Kernel(16) as bx:                       # Block 级并行（一维 grid）
    # ... buffer 分配（§4.3）...
    for tile_idx in T.Persistent([16], 16, bx):        # tile 维（每 block 1 tile）
        m_tile, n_tile = tile_idx // 4, tile_idx % 4   # 标量索引
        for kt in T.Pipelined(4, num_stages=2):        # K 维流水（双缓冲）
            T.copy(A[...], x_l1); T.copy(B[...], w_l1)
            T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))
        T.dual_copy(res, acc_ub)                       # L0C→UB（M-split 双 AIV）
        with T.SimtVF(threads=128):                    # 元素维向量化（N 轴）
            for i, j in T.Parallel(64, 128):
                z = acc_ub[i, j]
                out_ub[i, j] = z / (1.0 + T.exp(-z))
        T.dual_copy(out_ub, C[...])                    # UB→GM（M-split 双 AIV）
```

### 6.3 流水线优化

- **K 维 T.Pipelined(num_stages=2)**：L1 上 copy A / copy B 与 Cube MAD 的双缓冲流水（MTE2 与 MAD 重叠），与 example_gemm.py:71-74 同构。
- **CV 跨 tile 重叠**：mixed 结构下，tile t 的 Vector epilogue（dual_copy→silu→dual_copy）与 tile t+1 的 Cube 段（copy→gemm）由 AutoSchedule 调度重叠（epilogue_ub multibuffer ×2 版本即为此预留，deepgemm 的 T.annotate_buffer_versions 模式为可选显式化手段）。
- 不使用 T.Stage/T.Task 显式调度（auto_schedule: yes 下交 AutoSchedule；deepgemm 的 with T.Task() 模式为 Stage 4 对照项）。

### 6.4 尾块处理

无尾块：契约 shape（512×512×512）被 block（128×128×128）整除，M_TILES=N_TILES=K_TILES 均为整数；非契约 shape 在 wrapper 层拒绝（§8.4），kernel 内无动态边界分支（静态循环边界约束）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: auto_schedule: yes（AutoSchedule 自动同步；与 §5.5 ③ 一致）

### 7.2 同步点说明

auto_schedule: yes——无手动同步点。AutoSchedule（AssignCore/InsertSync pass）自动插入的 flag 对覆盖以下数据依赖边界（依赖方向 → 覆盖点）：

| 数据依赖边界 | 生产者 → 消费者 | AutoSchedule 处理 |
|--------------|----------------|-------------------|
| L1 tile 就绪 | MTE2（T.copy GM→L1）→ Cube MAD（T.gemm） | K 流水 multibuffer 版本同步（T.Pipelined num_stages=2 声明） |
| L0C 累加器就绪 | Cube MAD（末拍 kt=3）→ FixPipe dual_copy | clear_accum 生命周期 + MAD→FixPipe flag 对 |
| acc_ub 就绪 | dual_copy（L0C→UB）→ SimtVF silu | RAW flag 对（PIPE_MTE3→PIPE_V） |
| out_ub 就绪 | SimtVF silu → dual_copy（UB→GM） | RAW flag 对（PIPE_V→PIPE_MTE3） |
| AIC↔AIV 组内 | Cube 段 ↔ Vector 段（mixed __mix__(1,2) 编组） | InsertSync 跨核 flag 对（预算充足：跨核 16，CONST-arch-a5） |

### 7.3 pass_configs 配置

```python
pass_configs = {}
# 默认自动调度：不传 TL_ENABLE_AUTO_SCHEDULE（保持 True）。
# 仅完全手动核（本设计不采用）才需要：
# pass_configs[tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE] = False
```

---

## 8. 验证方案

### 8.1 Golden 函数

> 非迁移任务：golden 以 §1.3/§1.4 的算子语义（用户公式 z·sigmoid(z)、fp32 累加、silu 于 fp32 域施加、cast 后置）为依据，用 PyTorch 在 NPU（device="npu"）上独立实现，不复刻 §1.4 的 NPU kernel 算法（无 T.copy/dual_copy/T.exp 拓扑复用），保证验证独立性。silu 采用 torch.sigmoid 库原语（基线公式形态，与 kernel 的直除优化式相互独立——两者等价性已由 verify_equiv.py 机器验证）。

```python
import torch

def golden_gemm_silu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Ascend NPU 参考实现。
    A [M,K] fp16；B [N,K] fp16（NT 约定，逻辑上使用 B 的转置）；返回 C [M,N] fp16。
    语义：fp32 累加 GEMM → silu 于 fp32 域施加（z*sigmoid(z)）→ cast fp16（后置）。
    """
    A = A.to("npu")
    B = B.to("npu")
    Z = A.float() @ B.float().t()          # [M,N] fp32，K=512 fp32 累加
    return (Z * torch.sigmoid(Z)).to(torch.float16)
```

### 8.2 精度标准

golden 以 torch 在 NPU 上计算参考，`torch.testing.assert_close` 对比；容差按 dtype 分档（AGENTS.md API Convention）：

| dtype | atol | rtol |
|-------|------|------|
| float16（本算子输出） | 1e-2 | 1e-2 |
| float32（若 L1 加测 fp32 出口变体） | 1e-3 | 1e-3（累加重算子放宽至 5e-3） |

依据：fp32 累加（K=512）的 GEMM 差异在 1e-6 相对量级；silu 平滑（Lipschitz ≈ 1.1）不放大误差；fp16 出口量化误差 2^-11 ≈ 4.9e-4 < 1e-2。verify_equiv.py 已证明公式变体间 fp16 出口违反率 0（§1.6.1）。

### 8.3 分层测试计划（L0/L1/L2）

| 层级 | 内容 | 判据 |
|------|------|------|
| **L0 门槛**（Stage 3 必过） | 契约 shape 512×512×512，randn 输入 ×3 seeds；kernel 输出 vs golden（NPU 上 assert_close） | rtol=atol=1e-2 通过 |
| **L1 语义边界**（Stage 3） | ①值域构造：缩放 A/B（如 8×/0.02×）使 z 覆盖 ~±100（含 exp fp32 溢出边界 ±88 两侧）与 ~±1e-3 近零区；②denormal/极小输入；③NaN/inf 传播（A/B 注入 inf/NaN，核验输出与 golden 的 NaN 位置一致）；④确定性：同输入连续两次运行 bitwise 一致；⑤fp64 交叉核对：golden 的 fp32 路径 vs fp64 参照的误差分布统计（max/mean 相对误差） | ①-③ assert_close（equal_nan=True）；④ bitwise；⑤ 误差 ≤ 1e-2 相对 |
| **L2 性能基线**（Stage 4） | do_bench(backend="msprof") 端到端延迟记录；与 §1.6.0 设计期估算行对账（偏差 >2x 时注明失准项，hardware-cost-model.md §3） | 记录进 perf_opt/opt_log.md Baseline 段 |

### 8.4 Boundary（契约外行为说明）

- **非 16 倍数 shape 的拒绝行为**：本算子契约 shape 固定 512×512×512（全 16 对齐），kernel 按 512 特化编译；wrapper 校验 A/B/C 的 shape 严格等于契约值，不匹配即抛出明确 ValueError（契约外拒绝，不做静默 padding/crop）。若未来泛化到任意 16 倍数 shape：按 CONST-gemm-dim-constraints，非 16 倍数 M/N 在 GEMM 路径触发 static_assert 编译期拒绝——正确泛化路线是 kernel 内 GM→UB `pad_value=` 右 padding 或按 shape 特化重编译（host 侧禁止对输入内容做 padding，ascend-constraints.md §4）；非 16 倍数场景不在本算子契约内，按显式拒绝处理。
- **K 边界**：契约 K=512 % TILE_K(128) == 0，无 K 尾块；契约内不存在 OOB/越界访问。
- **极值语义**：z ≤ −88.7 时 fp32 exp 溢出 → 输出退化为正确极限 −0.0（直除式 z/inf；verify_equiv.py 角点已核，与 golden 一致）；z → +∞ 时 sigmoid → 1，输出 → z；NaN 输入全路径传播 NaN（三公式形态一致）。

---

## 9. 风险点与注意事项

### 9.1 已知约束（本算子相关）

- L1-input GEMM 仅 NT 路径（trans_A=False, trans_B=True，权重 GM [N,K]）；L0C→UB dual_copy 禁止类型转换（dav-950，copy_op.py ValueError 分支）——本设计 acc_ub 取 fp32 恰好规避且获得最优精度路径。
- M/N/K 必须 16 倍数、K % TILE_K_SUB == 0 且 TILE_K_SUB % C0 == 0（CONST-gemm-dim-constraints）；一维 T.Kernel、无 threads 参数；核内串行循环边界必须静态（全满足）。
- 常数引用（CONST-capacity-a5 / CONST-mte-bandwidth-model / CONST-cube-throughput-model / CONST-valu-model / CONST-arch-a5 / CONST-gemm-dim-constraints）当前全部 status=assumption（A5 冷启动期）——设计期容量按 90% 折扣、性能只做相对比较，绝对值以 Stage 4 do_bench(msprof) 实测为准。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| dual_copy 前后元素段的 AutoSchedule 所有权/同步异常 | "两个 dual_copy 之间插入 SimtVF 元素段"无仓内先例（§1.6.4） | 编译失败 / 结果错误 / epilogue 串行化 | 按 §1.6.4 降级：F1（deepgemm 式 plain T.copy 链 + loop-body 隔离：for range(1) + T.assume_no_conflict）→ F2（MixedKernel + sid guard）；回写 §1.6.4 与本表 |
| epilogue 使用 tanh 形式 | 误信 CONST-valu-model 的 special_func 统一档位 | 向量段 ~1.8× 劣化（探针实测 388.42 vs 215.67 µs） | 禁用 tanh 形式；silu 类激活用直除式 z/(1+exp(−z))（§1.6.1 探针结论） |
| L0C→UB dual_copy 带类型转换 | acc_ub 误声明 fp16 | copy_op.py 显式 ValueError / 静默错误数据 | acc_ub 保持 fp32，cast 在 VF 段完成（本设计已规避） |
| UB/L1/L0C 超预算 | block 过大或 multibuffer 版本数膨胀 | 编译失败或调度劣化 | §4.5 预算表留有余量（96/128/64KB vs 194/460/230KB）；block 变更时重算预算 |
| 57% AIC 利用率被误判为实现缺陷 | grid=16 < 28 AIC 是设计取舍（§5.2 权衡记录） | 无（非缺陷） | Stage 4 grid/tile 扫描裁决（128×64/32 blocks、64×64/64 blocks、TILE_K=256），注意 tile 缩小 → L1 装载流量上升 |
| 非 16 倍数 shape 输入 | 契约外使用 | wrapper ValueError（预期行为） | 契约内拒绝；泛化需求按 §8.4 路线重设计 |

### 9.3 特殊场景处理

- **极小/极值输入**：近零区（&#124;z&#124; < 1e-3）直除式与基线一致（verify_equiv 角点覆盖）；z ≤ −88.7 的 exp 溢出退化为 −0.0（正确极限，与 golden 一致）；z 大正数单调饱和到 z。
- **denormal**：fp32 次正规结果区两公式绝对差 ≤ 2e-6（§1.6.1 注③），fp16 出口无影响。
- **混合精度语义**：本算子固定 fp16 in / fp32 累加 / silu fp32 域 / fp16 out；不存在 fp8/bf16/INT8 路径（INT8 MMAD 本平台不支持且契约外）。
- **多实例复用**：wrapper 以 lru_cache 按 shape 特化编译缓存（契约内仅 512 一种），无跨 shape 状态污染。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/pilot/gemm_silu/
├── gemm_silu.py             # 算子实现（kernel + golden + 测试入口，Stage 3 产出）
├── DESIGN.md                # 本设计文档
├── verify_equiv.py          # §1.6.1 等价性机器验证脚本（已产出并执行，EQUIV_PASS）
├── RETROSPECTIVE.md         # Stage 1 复盘（追加式）
├── history_version/
│   └── design_probe_silu_form.py   # D-3 设计期探针（silu 三公式代价，已执行）
└── README.md                # 使用说明（可选）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| DESIGN.md | 已完成 | 本设计文档（含 §1.6 调研与算法级优化分析） |
| verify_equiv.py | 已完成（已执行，全用例 EQUIV_PASS） | 等价性机器验证（D-1，S1-EQUIV-EXEC） |
| history_version/design_probe_silu_form.py | 已完成（已执行，真机实测） | 设计期探针（D-3）：直除 0.902× / tanh 1.80× |
| gemm_silu.py | 待实现（Stage 3） | 算子实现 + golden + 分层测试 |
| test_gemm_silu.py | 待实现（Stage 3，可选） | 测试文件 |

### 10.3 命名规范

- 项目目录名: pilot（snake_case，conductor 解析）
- 算子目录名: gemm_silu（snake_case）
- 实现文件: gemm_silu.py
- 测试文件: test_gemm_silu.py

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md，含算法调研 §1.6.0 与算法级优化 §1.6.1-§1.6.4）
2. ✅ 等价性机器验证（verify_equiv.py，EQUIV_PASS ×2 候选）
3. ✅ 设计期探针（design_probe_silu_form.py，公式裁决完成）
4. ⬜ Golden 函数（§8.1，随 gemm_silu.py）
5. ⬜ 算子实现（gemm_silu.py：§3.3 伪代码 → 实现，dual_copy 主选路径，§1.6.4 fallback 备用）+ 分层测试（§8.3）与精度比对
