#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定、经 §0.6 R1/R2/R5/R6 落地的 online flash（chunk 粒度）算法。公式层源已是该族最优形态（e-base 减 max flash 合并、log 每行 1 次——R1 候选 2/3 弃选），本节优化集中在**划分 / 指令 / 归约时机 / 驻留层级**，每项四要素齐备。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| O1 | 减 max 稳定化 + flash 形式合并基（必选确认） | `y = log(Σ e^x)` | §1.3 递推式：合并基 = 合并后 m_run（非朴素 m_c 基） | 指数平移恒等；合并式全部 exp 参数 ≤ 0（p ≤ m_run、x^c ≤ m_c ≤ m_run）——**朴素 m_c 基在 fp16 域 max 间隙 131008 下溢出**（`TRAP-NAIVE-ONLINE-MERGE` measured：间隙 > 88.7 即 fp32 溢出），flash 形式安全 | 消溢出（源已含，语义层保留）；V4/V5 corner `max-gap 131008` 实测位级正确 |
| O2 | **BLOCK_N = tile_n 直保留 + 尾 chunk 组级 −inf 掩码 + 消 N_padded**（布局级）⭐ | 源：chunk 宽 tile_n 但 N_padded = align_up(N,256) 对齐（(33,33000) 搬 33,280 列，+280 pad 列）；`_needs_mask` 运行期双路径（最后 chunk 逐元素 masked）；〔对照〕SIMT 兄弟变体 fragment pow2 约束下 BLOCK_N = pow2 ≤ tile_n（16640→16384、25600→16384） | ① BLOCK_N = 最大 64 倍数 s.t. BLOCK_N ≤ tile_n ∧ 2·BLOCK_N·elem ≤ UB 预算（§4.5）——SimdVF 无 fragment，**不重划**；② full chunk 无谓词直载（tile_n 256 倍数 → 64 整除零尾组）；③ 尾 chunk 静态 rem 切片 DMA（只搬 rem 列）+ 满组直载 + 尾组 `vsel(c, vdup(−inf,f32), update_mask(k_tail,32))` | flash 合并对 chunk 划分满足**结合律**（每元素恰计入一个 chunk、合并代数恒等；浮点下仅重缩放乘法分组舍入差）——**V1**：8 组 (M,N,tile_n)（含 25600 非 pow2 4-chunk、16640 masked-tail、tiny、63-chunk 长链 1024000/16384）候选链 vs 源链 **fp64 位级一致（ulp=0）、fp32 ≤1 ulp、违反率 0**；**V6** 划分敏感性扫描（BLOCK_N ∈ {4096, 8192, 16384, 16640, 25600, 32768} × 3 shape，**非 pow2 BLOCK_N 首次纳入**）全部 ≤1 ulp；−inf 掩码双幂元中性（max(−inf,·)=· ∧ exp(−inf−m)=0，V1 尾 shape 覆盖） | 非 pow2 tile_n 下 chunk 数保持源划分：wl3 4 chunks（vs 兄弟 pow2 7，merge/DMA 迭代 −43%）、(33,33000) 2（vs 3）；GM 读消 pad 列（33000：−0.85%）；消 `_needs_mask` 双路径（统一 full/tail 静态二分支）；全 chunk 零尾组零掩码开销 |
| O3 | vexpdif 融合（指令级，SIMD 特有；两处受益） | `d = vsub(c, m); e = vexp(d)`（两指令）；rescale 因子 `vsub(prev, m); vexp` | `e = vexpdif(c, m_run)`（单指令，dst = exp(src0−src1)）；rescale `vexpdif(prev_brc, m_run)` | 公式恒等（同式融合——verify_equiv 候选链按融合式建模，V1 已含）；硬件差异仅为融合内部减法舍入时机 ≤0.5 ulp（fp32 减法）→ 经 flash 形式保证的参数 ≤ 0 上界下相对误差 ≤ 11.1·2⁻²⁴ ≈ 6.6e-7 ≪ 16-bit 容差 1e-2（4 个数量级余量；fp32 容差 1e-3 同 3 个数量级余量）——single 姊妹 O3 同款论证 + 本变体 V1/V3 复验 | −1 VALU op/64 元素（vsub @51 档）×pass2 + −1/chunk rescale：wl2 每 chunk −257 ALU 指令 ≈ 计算项 −9.6%（单源脚本复算）；`VP-2026-0014`：vexpdif fp32-only（16-bit 装载须先 vcvt——R1 #10 弃选的 API 侧佐证） |
| O4 | 延迟跨 lane 归约（归约时机，SIMD 特有） | 逐组跨 lane 归约再合并（`T.reduce` 在 SimdVF 的自动 lower 形态：每组 `vcmax(c)` + `vmax(acc, reduced)`，`src/ascend/transform/ascend_simdvf_lower_parallel.cc` L818–841） | 逐组元素级累加 `acc = vmax(acc, c)` / `accs = vadd(accs, e)` + 每 chunk 遍末**单次** `vcmax`/`vcadd` | max：结合/交换律精确成立（任何分组同值，无舍入）；sum：fp32 累加分组重排（lane 内跨组序贯 + 单次 64-lane 全归约 vs 逐组全归约+标量合并）→ 树深级 ulp 差；**V2**：3 shape（含 25600/16640 非 pow2 chunk）≤1 ulp、违反率 0 @1e-3 | wl2（G=256）每 chunk −(G−1)×2 = −510 指令 ≈ chunk 指令 −28%；计算项 −15.7%（vs 逐组形态，单源脚本复算）；比自动 lower 形态更优——显式内在的价值点 |
| O5 | 合并式标量最小化保持（寄存器向量合并） | 源：`for i in T.Parallel(block_m)` per-row 标量合并体（GPU 标量管线）；〔对照〕组粒度 online 每 64 元素组合并（R1 #8） | 每 chunk 每行 7 条向量指令完成 (max,sum) 双合并：`vcmax`+`vdupv`（chunk max 广播）+ `vmax`（max 合并）+ `vcadd`（chunk sum）+ `vexpdif`+`vmul`+`vadd`（flash 重缩放）——m_run broadcast 形态 / s_run lane-0 形态的 alloc_var 寄存器运算 | 合并代数逐式与源 S3/S6 恒等（fp32；broadcast/lane-0 仅是 64-lane 寄存器内的值布局，算术语义逐 lane 独立，lane-0 链自洽）——V1 fp64 位级（ulp=0）+ V4 位级复验 | merge 指令占比 wl2 0.39%（7/1,799 per chunk）、(33,33000) 0.19%（chunk 更大摊薄更薄）；vs 组粒度 online 每行 −512 指令（37×）；承接源 #7「重缩放开销 O(rows·chunks)」意图 |
| O6 | UB 两遍重读（驻留层级确认项） | 〔对照〕VRF 整 chunk 驻留单装载（pass2 免重读重转，R1 #7） | pass1/pass2 各自从 x_ub 重装载（`vld UNPK_B16`+`vcvt` ×2 遍） | 数值同链（两遍读同一 UB 版本，V1 已含该形态） | UB 重读 4.68K cyc/core（wl2，VRF↔UB 256B/cycle）= 计算项 3.8%（26× 余量）非瓶颈；换来 VRF ~3.3KB 零行数据驻留（vs R1 #7 的 256–400 数据寄存器 = fp32 chunk 64–100KB）+ 结构简单（无 alloc_local 数组索引）+ 不与 m_run/s_run 跨块驻留（R5）叠加无先例风险 |
| O7 | pass-1 累加器跨 chunk 折叠 max-merge（拒绝项） | 〔候选〕acc 不按 chunk 复位，chunk 末直接 `m_run = vdupv(vcmax(acc))`，省 per-chunk `vmax` 合并 + acc 复位 | —（不采纳，保留源式显式 m_c 合并） | 等价（max 精确、结合律）但收益微小 | 省约 2 指令/chunk/行（128 lane-ops/51 ≈ 2.5 cyc of 3,348 cyc/chunk）= **0.07% 计算项**；代价：pass1 acc 生命周期跨 chunk + 失去与源 S2/S3 逐式同构的保真锚点（V1 fp64 ulp=0 链按源式建模）——不值得 |
| O8 | 输出饱和角点封闭（确认项，非优化；tiled 契约 N 无上限） | — | `vcvt(out, fp16/bf16)` 默认 sat=True | **契约空间内数学封闭**：out = m + ln(Σ) ≤ 65504 + ln(N) 且 Σ ≥ 1（max 元素贡献 exp(0)=1）→ 饱和阈值 65520（fp16 IEEE 舍入到 inf 的中点）⟺ **N ≤ e^16 ≈ 8.86M**；**V7**：全 65504 行 N=102400 → y=65515.54 < 65520 ✓、N=1,024,000 → y=65517.84 < 65520 ✓（0 分歧 lane）；N > 8.86M 时 sat 钳位与 IEEE 溢出可能分歧——**源同样经 `T.cast` 共享该行为**（迁移保真），§9.3 文档化 | 契约空间（wrapper tile_n ≤ 32,512、显式 config 任意大 N 但 < 8.86M 实际不可达）内免额外钳位指令 |

**优化结论**：**采纳 4 项（O2 布局级 / O3 指令融合 / O4 归约时机 / O5 寄存器合并保持）+ 3 项必选/确认（O1 稳定化、O6 驻留层级、O8 饱和封闭）+ 1 项拒绝（O7，0.07% 不值得）**。优化后主路径公式（§3.1 拆解的唯一输入）：

```
per row:  m_run = −inf; s_run = 0                                    # alloc_var 寄存器
per chunk c (BLOCK_N = tile_n, 64 整除):
  pass1:  acc = vmax* 逐组累加 → m_c = vcmax(acc)（广播 vdupv）       # 延迟跨 lane 归约
  merge:  prev = m_run; m_run = max(m_run, m_c)                       # 源 S3
  pass2:  accs = vadd* 逐组累加 vexpdif(c, m_run)                     # 融合指数，UB 重读
  merge:  s_run = s_run·vexpdif(prev, m_run) + vcadd(accs)            # 源 S6 flash 重缩放
  tail:   尾组 vsel(−inf) 掩码后同上（rem 列静态切片）
epilogue: y = cast(m_run + vln(s_run), dtype)（ONEPT 单点存储）
```

**等价性机器验证（D-1）**：已生成并执行 `examples/logsumexp_simd/_logsumexp_kernel_tiled/verify_equiv.py`（fp64 参照 + 角点值：全零/全等/单主导/±65504 极值/subnormal/深度下溢 −745/部分 −inf/全 −inf/含 +inf/含 NaN/**首 chunk 全 −inf**/**尾 chunk 全 −inf**/N=1/非 64 倍数 N（257、33000 的 rem=16360）/超长链 N=1024000（63 chunks）/非 pow2 tile_n（16640、25600）；torch CPU；候选链按「BLOCK_N=tile_n 划分 + 64 元素组分桶 + lane 内跨组累加 + 单次跨 lane 归约 + vexpdif 同式融合 + 尾组 −inf + flash 逐 chunk 合并」逐要素模拟，基线链按源 tiled 算法（tile_n chunk + fp32 中间 + flash 合并序）模拟）。执行结果（2026-10-17，tilelang-fork 工作区 + torch CPU，**全部 EQUIV_PASS**）：

| 验证项 | 对照（候选 vs 基线/精确） | 最大 ulp 差（fp32 位级） | 最大相对差 | 违反率（@容差） | 结论 |
|--------|--------------------------|--------------------------|------------|---------------------|------|
| V1（O1+O2+O3+O4，8 shape × 2 参照 = 16 行） | SIMD online 链（BLOCK_N=tile_n、含融合/延迟归约/尾组掩码建模）vs 源 tile_n 链（fp32 链对照 + fp64 参照）：(8,32768) tn=16384、(4,102400) tn=25600、(33,33000) tn=16640、(4,32768) tn=32768、(8,100) tn=64、(3,257) tn=256、(2,1024000) tn=16384（63-chunk 长链）、(57,32768) tn=16384 | fp64 参照 **0（全部 8 shape）**；fp32 链 **≤1**（5 shape 为 0） | ≤6.6e-08 | 0.000000（@1e-2 / @1e-12 fp64） | **EQUIV_PASS** |
| V2（O4，3 shape） | 延迟跨 lane 归约 vs 逐组跨 lane 归约（两种 fp32 分组） | **≤1**（(4,102400) tn=25600 为 0） | ≤6.5e-08 | 0.000000（@1e-3） | **EQUIV_PASS** |
| V3（A5 链 vs golden，8 shape） | SIMD online 链 vs `torch.logsumexp`（fp32） | ≤1 | ≤6.5e-08 | 0.000000（@1e-3） | **EQUIV_PASS** |
| V4（源链 vs 候选链，16-bit 输出，3 N × 2 dtype = 6 行） | 18 类特殊行（含 NaN/±inf/部分 −inf/全 −inf/极值/subnormal/深下溢/max-gap 131008）fp64→fp16/bf16 输出对照 + NaN/±inf 模式逐位比对（N=32768 tn=16384、N=33000 tn=16640、N=102400 tn=25600） | **0（位级一致）**；NaN/±inf 模式全部一致 | 0 | 0.000000（@1e-2） | **EQUIV_PASS** |
| V5（tiled 特有角点，信息项） | ① **首 chunk 全 −inf 且后续含有限值**行：候选 vs 源 tiled（NaN 位级比对，int64 位模式——torch.equal 对 NaN 恒 false 须位级比较）vs single 形态 vs torch；② **尾 chunk 全 −inf**（首个含有限值 chunk 之后）：候选 vs torch | ① 候选 = 源 = **NaN（位级一致）**；single 形态 = torch = 9.704（有限）——分歧为**源固有**（§0.1 ④、§9.3），非采纳项失败；② 候选 = 9.704 = torch（**中性**，m_run 有限 → exp(−inf−m)=0） | — | — | **EQUIV_PASS**（迁移保真达成） |
| V6（O2 划分敏感性，12 行） | BLOCK_N ∈ {4096, 8192, 16384, 16640, 25600, 32768} × {(4,102400), (33,33000), (8,32768)}（BLOCK_N ≤ N 取值域内）vs 源链 | **≤1**（(4,102400) 全档 0；(33,33000) 全档 1；含非 pow2 档 16640/25600——**首次纳入非 pow2 扫描**，SIMT 兄弟 V6 仅 pow2） | ≤6.5e-08 | 0.000000（@1e-3） | **EQUIV_PASS** |
| V7（O8，信息项） | sat=True 钳位 vs IEEE 溢出（全 65504 行 × N ∈ {102400, 1024000}） | — | — | — | **0 分歧 lane**（y ≤ 65504+ln(N) < 65520；封闭域 N ≤ e^16 ≈ 8.86M） |

> **全部采纳项 EQUIV_PASS**（V1 fp64 位级 + fp32 ≤1 ulp、V2/V3/V6 ≤1 ulp、V4 位级一致 + NaN 模式一致、V5 角点位级保真）。V5 同时证实：首 chunk 全 −inf → NaN 是源 tiled online 递推的固有语义（single 形态与 torch 给有限值），本设计迁移保真保持（§9.3 文档化 + §8 Boundary ④ 用源语义模拟器验证，不用 golden）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 本变体的模式本身即「显式向量化」——逐元素计算 100% 以 `T.simd.*` 2048-bit 向量内在表达（每指令 64 fp32 / 128 fp16 元素）。本节盘点全部循环与标量计算点，确认无逐元素标量残留、逐项说明保留的非向量化点（与 §6 循环结构一致）。

| # | 计算点 | 实现形态 | 向量化状态 | 不可替代理由（非向量化点必填） |
|---|--------|-----------|-----------|------------------------------|
| 1 | 输入装载 + fp32 提升（逐元素） | `vld(x_ub[0, g·64], dist="UNPK_B16")` + `vcvt(part=0)` 每 64 元素组 1+1 指令（fp32 输入 `vld` NORM 直连） | ✅ 2048-bit 显式 | —（`testing/ascend/language/test_tilelang_ascend_simdvf_cast.py` L64–80 位级回归；single 姊妹 L154–162 Stage 3 实证） |
| 2 | 尾 chunk 边界掩码（逐元素条件） | 满组直载零掩码；尾组 `vsel(c, vdup(−inf,f32), update_mask(k_tail,32))`——向量谓词选择，仅尾 chunk 最后 1 组 × 2 遍 | ✅ 谓词化 | —（`deepgemm/transform_sf.py` L205/L215/L249 vsel 先例；`test_simdvf_integer_predicate.py` L261 update_mask 先例；比源逐元素 `T.if_then_else` 载入更优——掩码只付 1 指令/遍而非 1 select/元素） |
| 3 | chunk 内行 max（逐元素比较） | `vmax(acc, c)` 逐组 + 遍末单次 `vcmax`（+`vdupv` 广播） | ✅ | —（flash core.py L115–118 同构） |
| 4 | 稳定化指数（逐元素 exp） | `vexpdif(c, m_run)` 融合单指令/组 | ✅ | —（flash core.py L130–133 同构；O3） |
| 5 | chunk 内行和（逐元素累加） | `vadd(accs, e)` 逐组 + 遍末单次 `vcadd` | ✅ | —（flash core.py L134–137 同构） |
| 6 | (max, sum) flash 合并（跨 chunk 统计量） | 7 条向量指令/chunk/行（O5）：`vmax` 合并 + `vexpdif`/`vmul`/`vadd` 重缩放，m_run broadcast / s_run lane-0 形态 alloc_var 寄存器运算 | ✅（寄存器向量合并） | —（vs 源 GPU 标量管线合并：等价向量形态；vs 组粒度 online：37× 指令差，R1 #8 弃选） |
| 7 | 每行 epilogue：y = m + log(S) + cast + 写出 | `vadd(m_run, vln(s_run))` → `vcvt` → `vsts(ONEPT)`——4–5 向量指令/行，值在 lane 0 有效（其余 lane 垃圾不逃逸——ONEPT 只取 lane 0） | ✅（行粒度向量指令） | 依赖归约终值（s_run 定稿后才能求 log）——tile 级标量（4–5 指令/行 vs 1,799 指令/chunk/行元素段，wl2 占比 ≈ 0.07%），无法并入元素循环 |
| 8 | 行循环（任务内 rows_this 行串行） | `for r in range(rows_this)` Python 展开（rows_this 静态 ∈ {block_m, rem_m}；主流 bm=1 → 单迭代退化） | ❌（保留串行） | **alloc_var 是单寄存器语义（一个 2048-bit 向量寄存器），无法按行索引为数组**——行批在线 running stats 只能落 UB（= fallback 形态，§0.6 R5）；行串行使 stats 退化为每行一对寄存器（主选形态）；行间独立但 SimdVF 无行级并行原语（无线程域，行到核映射由 persistent 调度承接——M 维并行在任务层已消耗） |
| 9 | chunk 循环（行内 num_full chunk 串行） | `for c in T.serial(num_full)` + 尾 chunk Python 静态分支 | ❌（保留串行） | **running stats 循环携带依赖**（s_run 更新依赖前一 chunk 终值——online 递推的算法本质，顺序不可交换）；chunk 间的并行性已由 x_ub 双版本流水**重叠**利用（MTE2 c+1 ∥ VALU c，重叠非并行）；`P-LAYOUT-ROW-RESIDENCY` measured：串行循环用 T.serial 保持双版本流水顺序依赖 |
| 10 | 组循环（chunk 内 G 组 × 2 遍） | `for g in range(G)` Python 展开（G ≤ 430）+ 尾组静态分支 | ❌（循环载体，体内全向量化） | 组循环体 = 3–4 条向量指令（#1–#5 已全向量化）；两遍扫描数据流本质（pass2 依赖 pass1 的 m_run 终值）；G 静态、展开形态与 single 姊妹（G ≤ 508 Stage 3 实证）一致 |
| 11 | 尾批 TIR if 分支（`if t < full_m`） | 运行时控制流，分支体 = `row_task` 宏两次实例化（rows = block_m / rem_m 静态） | ❌（保留分支） | 依赖循环变量 t 的 M 尾批边界语义；分支体两侧均为静态切片 + 静态行数（SIMT tiled 兄弟 L233–250 同款实证）；每任务 1 次分支判断，非热点 |
| 12 | host 侧 BLOCK_N/num_full/rem/full_m/rem_m/G/k_tail 计算 | Python 算术（builder 闭包静态化） | n/a | host 元数据计算，不在 kernel 内（per-shape JIT）；`_tile_n_for_block_m` 的 wrapper 侧启发式不属于本 kernel 契约（§0.2） |
| 13 | lse 写回（任务尾 rows_this 元素） | `T.copy(lse_ub[0:rows_this], y[row_base:+rows_this])` MTE3 DMA | ✅（DMA 块搬运） | 数据搬运非 VALU 计算；静态切片（兄弟 L222 同款） |

**向量化结论**：逐元素计算 100% 显式向量化（#1–#6 全部 2048-bit 内在，API 均有 `examples/ascend` / `testing/ascend` 位级或实测佐证）；保留 4 处非向量化点——#7 行级 epilogue（4–5 向量指令/行，依赖归约终值，占比 0.07%）、#8 行循环（alloc_var 单寄存器语义不可行索引 + 主流单迭代退化）、#9 chunk 循环（online 递推顺序依赖，已由双缓冲流水重叠利用）、#10/#11 组循环载体与尾批分支（数据流本质 + 运行时边界语义，先例佐证）——理由类别 = tile 级标量 + 算法顺序依赖 + 寄存器语义限制 + 边界处理，均非「实现简单」类借口；与 §6 循环结构一致（§6 无逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（[M, N] 行主进 / (M,) 行主出），核内布局与 lane 映射是设计变量。规约类算子必选候选：水平归约（lane 映射到归约轴 N）vs 垂直扫描（lane 映射到独立行 M）；另评估 tiled 特有轴决策（chunk 划分粒度 / running stats 驻留层级——§1.6.0 已裁，此处收口布局面）。

**轴质量评分**（对选定轴 N）：

| 评分项 | 判定 |
|--------|------|
| 整除性 | wrapper tile_n 为 256 倍数 → 64 整除 ✓ full chunk **零尾组**（组宽 64）；尾 chunk rem 任意（组级掩码吸收）；显式 config 任意 tile_n → BLOCK_N 取 64 倍数 clamp（§4.5）仍零尾组 |
| 尾 lane 浪费率 | full chunk 0%；尾 chunk 仅最后 1 组（k_tail/64 < 100%，wl3 类整除 rem=0 完全无尾 chunk；(33,33000) 尾 chunk k_tail=40/64 = 62.5% 有效——pad lane 的 exp(−inf−m)=0 不污染 sum） |
| 累加链形态 | 归约维 N = chunk 内 stride-1 连续向量维（组内 64 lane 连续装载）；无跨步系数；行切片 (1, BLOCK_N) DMA 连续 |
| repack 代价 | 无（I/O 原生行主连续，核内布局与 I/O 同构；UNPK_B16 装载即提升，无交错重排——规约序不敏感，flash 合并对划分结合律〔V6〕） |
| UB/VRF 容量影响 | chunk 驻留：x_ub (1, BLOCK_N) 双版本 65.5KB@wl2 / 102.4KB@wl3 / ≤220KB 上界（fp32 clamp）+ lse_ub ≤64B + VRF 流式 ~3.3KB（§4.5 预算表）——**block_m 与 UB 解耦**（x_ub 单行粒度，行数不进容量式；无 single 变体的 SUB 封顶机制） |

**候选矩阵**（规约类候选全枚举 + tiled 特有轴，弃选给量化理由）：

| # | 布局方案 | 向量化轴 / 驻留层级 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | 行主 [M,N] 原生 + chunk 内**水平归约**（lane→N 连续轴，64 元素组，UB 两遍重读 + alloc_var stats + BLOCK_N=tile_n，persistent 行块任务） | N / UB 两遍 + 寄存器 stats | 无 | 连续 GM 行切片（wl2 行 64KB/chunk）；归约维 = stride-1 组内连续；flash/cast/arithmetic 三先例 + single 姊妹 Stage 3 实证 + SIMT tiled 兄弟（tiled 契约）实证；vcmax/vcadd 单指令全归约；任意大 N（chunk 流式） | ✅ **主选** |
| 2 | 垂直扫描（lane→M 行，N 向串行步进） | M | 无 | M 维短（wl3 M=4 → 4/64 lane 利用率 6%；wl2 1024 行可行但 N 串行 32768 步/行）；行内并行度归零；repo 无先例 | ❌ 串行化 N 维（32768 步 vs 256 组×64 lane），量化劣化 >10×；且 vertical 形态下 running stats 须跨行 维护（更复杂） |
| 3 | 跨核 N-split + partial (m,s) 合并（2D 分块 grid） | N（块内）+ 跨核 M×Nsplit | partial 统计量 GM 往返 | wl3 (M=4) 并行度 4→56 理论受益；(max,sum) 合并非 additive（须定序合并段）；partial GM 往返 + 跨核 flag / 第二 kernel；**浮点合并顺序变化破坏 V1/V4 位级保真论证** | ❌ 弃选（R1 #12 同源：复杂度 + 保真失效；wl2 满载零收益；兄弟 Stage 4 `[DESIGN_LIMIT]` M≪56 天花板同族记录；列 §9.1 风险 3 Stage 4 观察项） |
| 4 | 转置布局（列主驻留 + M 向归约） | M（转置后） | 核内转置链（通用转置 API 存在性未命中——未文档化假设） | repack 纯开销 + UB 翻倍；收益仅换归约轴到 M（M 短无收益） | ❌ repack 无收益（N 已是最长连续轴） |
| 5 | pow2 BLOCK_N 重划（SIMT 兄弟被迫形态） | N（同主选） | 无 | 非 pow2 tile_n 下 chunk 数虚增（wl3 7 vs 4：merge/DMA 迭代 +75%）；有效 lane 浪费（fragment 50.8%/78.1%——SimdVF 无此约束） | ❌ 弃选（fragment 约束在 SimdVF 不存在——重划是约束残留纯负优化，O2 反向论证 + V6 非 pow2 位级/≤1 ulp） |
| 6 | 逐组跨 lane 归约（自动 lower 形态直译） | N / 归约时机=逐组 | 无 | 每组 +1 vcmax +1 vmax（R1 #6 弃选形态） | ❌ 被 O4 延迟归约取代（−28% chunk 指令） |
| 7 | VRF 整 chunk 驻留单装载（SIMT 形态直译） | N / VRF 驻留 | 无 | −10% VALU（免 pass2 重读重转）vs **VRF 64–100KB 行数据驻留**（256–400 寄存器）+ alloc_local 数组复杂度 + 与 alloc_var 跨块驻留叠加无先例风险（R1 #7） | ❌ 弃选（D-2：UB 重读 26× 余量非瓶颈；VRF 压力 + 风险叠加得不偿失） |
| 8 | 组粒度 online 更新（每 64 组合并 stats） | N / 更新粒度=组 | 无 | 每 64 元素 +3 指令 ×G（37× merge 指令）+ 角点保真分歧（R1 #8） | ❌ 弃选（chunk 粒度已最优：7 指令/chunk vs 3 指令/组×G） |

**源码轴选择对照**（迁移任务必做）：源 GPU 也是 N 轴并行（`T.Parallel(block_m, tile_n)` 行内元素线程化 + `T.reduce_max/sum` dim=1）+ 行级 grid + chunk 串行——**轴结论一致**（N 是唯一高带宽连续轴、M 是任务轴，两代硬件共识；chunk 串行顺序依赖亦一致），GPU 轴选择是输入而非结论，已独立复核。承载结构完全重设计：GPU block×threads=256 → SimdVF 2048-bit 指令流；GPU fragment 跨 chunk 驻留 → alloc_var 寄存器；GPU N_padded 对齐 + 快慢双路径 → BLOCK_N=tile_n 紧凑 + 静态二分支（§0.6 R1/R2/R5）。

**布局决策结论**：选定方案 #1：核内布局 = `x_ub (1, BLOCK_N=tile_n)` dtype 紧凑（双版本）+ `lse_ub (block_m,)` + **VRF 流式零行数据驻留**（仅 stats 2 寄存器 + 累加器/常量 ~13 个）；向量化轴 = **N（chunk 内连续轴，64 元素组粒度）**；跨 lane 归约时机 = **延迟单次**（每 chunk 1 vcmax + 1 vcadd）；running stats 驻留 = **行作用域 alloc_var 寄存器**（跨 chunk SimdVF 块，风险 1 + fallback）；chunk 粒度 = **BLOCK_N = tile_n**（64 倍数 clamp 兜底）。该结论同步落入 §3.3 伪代码与 §6 循环结构；弃选方案量化理由见候选矩阵（#2 >10×、#3 保真失效 + 复杂度、#4 repack 无收益、#5 +75% chunk 数、#6 −28%、#7 VRF 压力、#8 37× merge）。**无实验裁决三件套**：主选与全部弃选候选的判定裕度均不依赖未实证常数（#5/#6/#8 为结构性指令计数差；#7 为容量水位与 26× 余量的量级差；#2/#4 为轴量级差；#3 为语义保真 + 复杂度弃选），不触发 skill Phase 2 第 3 项的强制裁决条件。

---
