## 8. 验证方案

### 8.1 Golden 函数

**主 golden（精度对照基准）**——源仓测试基准 `ref_program` 直移植，torch on NPU（fp32 计算域，保证验证独立性——不复刻 §0.6 的 NPU 算法）：

```python
def golden_logsumexp(x: torch.Tensor) -> torch.Tensor:
    """Row-wise logsumexp, torch on NPU (source test baseline ref_program)."""
    assert x.device.type == "npu" and x.ndim == 2
    return torch.logsumexp(x.float(), dim=-1).to(x.dtype)
```

**源语义模拟器（迁移保真基准）**——tiled 源算法的 fp32 逐要素复刻（BLOCK_N=tile_n chunk 划分 + fp32 中间 + flash 合并序 + 尾列 −inf），用于 golden 与源语义**分歧**的角点（±inf / 全 −inf / 首 chunk 全 −inf——torch 在这些角点与源 tiled 不同，§9.3）：

```python
def _source_lse_tiled_sim(x: torch.Tensor, tile_n: int) -> torch.Tensor:
    """fp32 re-implementation of the SOURCE tiled online algorithm semantics.
    Replicates chunk-serial flash merge order (m_run/s_run recursion, pad=-inf
    tail columns) so source-inherent divergence corners (all -inf row -> NaN,
    +inf row -> NaN, first-chunk-all -inf -> NaN) validate against THIS, not
    against torch.logsumexp (DESIGN 9.3 / verify_equiv V5)."""
    M, N = x.shape
    out = torch.empty(M, dtype=x.dtype)
    for i in range(M):  # 行独立（与 kernel 调度无关）
        m_run, s_run = float("-inf"), 0.0  # fp32 链（源 dtype 语义）
        for c0 in range(0, N, tile_n):
            cols = x[i, c0 : c0 + tile_n].float()
            cols = torch.cat([cols, torch.full((tile_n - cols.numel(),), float("-inf"))])  # 越界列 = -inf（源 masked 语义）
            m_c = cols.max()
            prev = m_run
            m_run = max(m_run, m_c)
            e = torch.exp(cols - m_run)
            s_c = e.sum()
            s_run = s_run * torch.exp(torch.tensor(prev - m_run)) + s_c
        out[i] = torch.tensor(m_run + torch.log(torch.tensor(s_run))).to(x.dtype)
    return out
```

> 两基准分工：randn 数据域（L0 主门槛）golden 与源语义模拟器一致（V3：≤1 ulp）；±inf/NaN 角点（Boundary ④）**只对源语义模拟器**（V4/V5：候选链与源位级一致 = 迁移保真达成；与 golden 的分歧为源固有，§9.3 文档化）。模拟器自身以 verify_equiv.py 的源链（V1/V4/V5 基线式）交叉锚定。

### 8.2 精度标准与分层测试计划（L0 门槛；L1/L2/Boundary 完整套件由 tilelang-op-develop 落地）

**容差**（AGENTS.md 口径）：fp16/bf16 → `torch.testing.assert_close` rtol=1e-2；fp32 → 1e-3（fp32 累加链 102,400 元素归约属 accumulation-heavy，若 L0 抖动超限按 5e-3 档复核并在 RETROSPECTIVE 记录）。

**L0（本设计规定的门槛用例，全过才算实现成功）**：

| # | shape (M, N) | dtype | tile_n（wrapper 复算） | block_m | 覆盖点 |
|---|-------------|-------|------------------------|---------|--------|
| 1 | (1024, 32768) | bf16 | 16384 | 1 | wl2 主 workload；2 full chunk；rem=0；**首用例 = 最简主流形态（bm=1/rem=0，风险 1 首编隔离）** |
| 2 | (4, 102400) | fp16 | 25600 | 1 | wl3；4 full chunk；非 pow2 BLOCK_N |
| 3 | (4, 102400) | bf16 | 25600 | 1 | wl3 dtype 变体 |
| 4 | (33, 33000) | fp16 | 16640 | 1 | 尾 chunk（G_tail=255 + k_tail=40）+ M 非整除尾批 + 非 pow2 |
| 5 | (4, 32768) | fp16 | 16384 | 1 | test 形态；2 full chunk |
| 6 | (33, 32768) | fp16 | 16384 | 1 | M 非整除 + 大 N |
| 7 | (8, 100) | fp32 | 64（显式 config） | 1 | fp32 直连装载 + 纯尾 chunk（N < tile_n → num_full=0 路径）+ k_tail=36 |
| 8 | (1024, 32768) | bf16 | 16384 | 4 | bm>1 行串行（wrapper default_config 遍历形态）；M 整除 |

L0 断言：`torch.testing.assert_close(y, golden(x), rtol=tol, atol=tol)`（randn 域）+ wrapper 落点复算 assert（tile_n/block_m 与 §0.2 表一致）。

**L1/L2/Boundary（Stage 3 完整套件，设计规定覆盖面）**：

- **形状/粒度扫描**：BLOCK_N ∈ {64, 256, 4096, 16384, 16640, 25600}（含非 pow2）× shape ∈ {(3,257), (2,1024000), (57,32768), (1,1), (5,63), (7,65)}——num_full=0 / k_tail∈{0,1,63} / 单 chunk / 63-chunk 长链；block_m ∈ {1,2,4,8,16} × {M 整除, M 非整除, M<block_m}；dtype × {fp16, bf16, fp32}。
- **Boundary（角点行，对 `_source_lse_tiled_sim`）**：18 类特殊行复用 verify_equiv V4/V5 集——全零/全等/单主导/±65504 极值/subnormal ±6e-8/深度下溢 −745/部分 −inf 散布/**全 −inf**/**含 +inf**/**含 NaN**/max-gap 131008/**首 chunk 全 −inf 且后续有限**（→ NaN，tiled 特有）/尾 chunk 全 −inf（→ 中性）/N=1/末元素最大（跨 chunk max 边界）/首元素最大；每类 × {tn=16384/N=32768, tn=16640/N=33000, tn=25600/N=102400} × {fp16, bf16}。
- **golden 分歧回归**（对 golden，预期分歧并 assert 分歧形态）：全 −inf 行 → 源 NaN vs torch −inf；+inf 行 → 源 NaN vs torch +inf；首 chunk 全 −inf → 源 NaN vs torch 有限值（**tiled 特有**，single 变体无此分歧）。
- **L2（鲁棒性）**：lru_cache 复用（同 shape 二次调用位级一致）；连续调用不同 shape；M=1/N=1；显式 config tile_n > clamp 上界（fp32：BLOCK_N=27,520 clamp 路径 + rem=5248/k_tail=0）。

### 8.3 性能验证（Stage 4，设计期仅记录口径）

- 入口：`tilelang.profiler.do_bench(fn, backend="msprof")`（端到端时延唯一权威）；`benchmarks/ops/bench_logsumexp_simd.py`（manifest 驱动，wl2/wl3 为 tiled 覆盖对象）。
- 对照基线：SIMT tiled 兄弟 Stage 4 采纳值（wl2 100.2µs / wl3 17.6µs）——同 workload 家族跨模式参考（非门禁，记录口径）。
- 归因：npusim 周期定位（special_func vs MTE 瓶颈裁决——§9.1 风险 4 的实测标定入口）；设计估算 vs 实测偏差行按 hardware-cost-model.md §3 回填 `perf_opt/opt_log.md`。

## 9. 风险点与注意事项

### 9.1 已知约束（本算子相关，按风险度排序）

| # | 风险 | 影响 | 缓解 / 裁决计划 |
|---|------|------|----------------|
| 1 | **alloc_var 跨 SimdVF 块声明+访问（m_run/s_run）无单一先例**——行作用域声明、init/chunk/epilogue 多块读写；若 VFChecker/buffer 作用域报错则设计主形态受阻 | 首编失败 → 阻塞 | 机制链三证据（`allocate.py` L103–168 声明无域检查 / `vf_regions.h` L50–53 enclosing-scope / `insert_sync.cc` L1308 local.var 跳同步）；Stage 3 首用例（L0 #1 最简形态）首编即触发验证；**fallback 全先例**：stats UB 往返（`stats_ub (2,) fp32` + flash core.py L103/L117/L120–126 形态：chunk 末 `vsts(ONEPT_B32)` / chunk 首 `vld(BRC_B32)` + `mem_bar("VST_VLD")`），语义等价（V1 覆盖该链形态——模拟器按值链验证与驻留位置无关），代价 ~5 指令/chunk/行 = wl2 0.14% 计算项 |
| 2 | **首 chunk 全 −inf → NaN 角点（源固有 tiled 分歧）**——golden（torch）在该角点给有限值，误用 golden 对照会把源语义判为 bug | Boundary ④ 误判 | 已 V5 位级锚定（候选=源=NaN）；Boundary ④ 只对 `_source_lse_tiled_sim`（§8.1 分工）；golden 分歧回归用例显式 assert 分歧形态（§8.2）；§9.3 文档化 |
| 3 | **wl3 M=4 并行度天花板**——行任务粒度下仅 4/56 核活跃（D-2：20.9K cyc/行 vs wl2 满载 122.7K/核均值） | wl3 时延受限于 M | 跨核 N-split + partial 合并已评估弃选（R1 #12：(max,sum) 非 additive + 位级保真失效 + 复杂度；兄弟 Stage 4 `[DESIGN_LIMIT]` 同族记录）；列 Stage 4 观察项——若 wl3 实测成瓶颈且用户接受合并序容差语义，再评估 N-split + 定序合并变体 |
| 4 | **special_func 吞吐常数粒度不确定**——`estimate_latency.h` L58–70「8 ops/cycle」未区分 element vs 向量指令粒度；若为指令级则 D-2 瓶颈反转为 MTE 主导 | 设计期 D-2 归因失准（结构选型不受影响——两种解释下主选均最优，§1.6.0 末段） | Stage 4 npusim 周期归因实测标定；偏差 >2× 时按 hardware-cost-model.md §3 回填偏差行并回写 pattern-library 常数条目 |
| 5 | **block_m>1 时 per-row (1, BLOCK_N) DMA**——行串行使 DMA 描述符数 ∝ rows×chunks（兄弟为 (SUB, BLOCK_N) 整批，描述符 ∝ chunks）；字节数不变 | bm>1 显式 config 下 MTE base 项增长（wl2 bm=4 估算 +3.3K cyc/核 ≈ 2.7%——描述符数与 bm=1 相同的总量级，因任务内行循环本就逐行拷贝） | wrapper 主流落点 bm=1（tiled 路径偏好，§0.2）不受影响；bm>1 为显式 config 边缘路径；Stage 4 若 bm>1 shape 进 bench 再评估批拷贝形态（语义等价、x_ub 需 (bm, BLOCK_N)——容量 ×bm 须重算 §4.5） |
| 6 | **饱和封闭域 N ≤ e^16 ≈ 8.86M**——超出后 `vcvt` sat 钳位与 IEEE 溢出可能分歧（y > 65520 时） | 超大 N 显式 config 的 fp16 输出 | V7 实测 102400/1024000 两档 0 分歧；源经 `T.cast` 共享同行为（迁移保真非缺陷）；§9.3-⑧ 文档化；契约空间（wrapper tile_n ≤ 32,512）实际不可达 |
| 7 | **fp32 / 超大 tile_n clamp 路径的 UB 99.9% 档**——BLOCK_N=27,520 时总占用 228,416/228,556B；AutoSchedule multibuffering 膨胀系数未实测（constants.md 待标定项） | 编译期容量报错 | clamp 公式保证显式算术不超预算；若编译报 UB 溢出（膨胀系数 >1.0 首次实测）→ 降 BLOCK_N（64 的倍数递减，语义中性 V6）并回写 constants.md 膨胀系数条目 |
| 8 | **Python 展开体积**——`range(G)`（≤430 × 2 遍）+ `range(rows)`（block_m）× 任务体宏双实例化；大 block_m 显式 config（>64）展开膨胀 | 编译时间增长 | G 展开与 single 姊妹（≤508 Stage 3 实证）同量级；block_m > 64 非契约主流（wrapper ≤16）——builder 不加人为上限，Stage 3 遇大 block_m 显式 config 再评估 `T.serial` 行循环形态（语义等价） |

### 9.2 常见错误（实现期检查清单）

1. **循环携带 / 跨块值必须 `T.simd.alloc_var`**——SSA 重赋值报 `Immutable variable used outside its defining region`（single 姊妹 Stage 3 教训）；m_run/s_run/acc/accs 全部 alloc_var。
2. **掩码只在 fp32 域**：`vsel(cv, vdup(−inf,"float32"), update_mask(k_tail, 32))`——不在 16-bit 域掩码（NaN/−inf 编码差异）。
3. **`vexpdif` fp32-only**：16-bit 装载先 `vcvt(part=0)` 提升；fp32 输入 `vld` NORM 直连无 vcvt。
4. **尾批用 TIR if**：`if t < full_m`（t 为运行期变量，Python if 会 trace 期求值报错/误消除）；尾 chunk/尾组用 Python if（rem/k_tail 编译期常量）。
5. **ONEPT 存储按 dtype 配对**：16-bit `pset(16,"PAT_VL1")` + `"ONEPT_B16"`；fp32 `pset(32,…)` + `"ONEPT_B32"`。
6. **`T.Kernel(56)` 一维、无 threads=**（TypeError）；`T.annotate_buffer_versions` 在 `T.Persistent` 前。
7. **切片全静态推导**：行/列切片边界全部由编译期常量表达式给出（rem/rem_m/rows_this）——不依赖 `T.copy` OOB 行为（§0.6 R4）。
8. **alloc_var 声明位置**：行作用域（chunk 循环外、SimdVF 块外）——若首编报 buffer 作用域/VFChecker 错误，按风险 1 fallback 切 stats UB 往返（先备份后修改）。
9. **域内常量重物化**：neg_inf/zero_f/one_pt/tail_mask 在每个 SimdVF 块内声明（SSA 不跨域）；不要在域外引用域内 SSA 值。
10. **模拟器交叉锚定**：`_source_lse_tiled_sim` 必须先对 verify_equiv 源链自测（torch CPU 对照）再用于 Boundary——防止模拟器自身 bug 污染保真判定。

### 9.3 特殊场景处理（边界语义文档）

| # | 场景 | 源 tiled 行为（= 本设计行为，迁移保真） | torch（golden）行为 | 依据 |
|---|------|--------------------------------------|---------------------|------|
| ① | 任一元素 NaN | y = NaN（逐元素传播：m_c 或 e 任一 NaN → s_run=NaN 不可恢复） | NaN | V4 位级 + NaN 模式 |
| ② | 全 −inf 行 | **NaN**（每 chunk m_c=−inf → m_run 恒 −inf → 首 chunk `exp(−inf−(−inf))=NaN` 污染） | −inf（**分歧**） | V4；single 姊妹同源分歧 |
| ③ | 含 +inf 行 | **NaN**（inf 位置 `exp(inf−inf)=NaN` 污染） | +inf（**分歧**） | V4 |
| ④ | **首 chunk 全 −inf 且后续 chunk 含有限值** | **NaN**（该 chunk 处理时 m_run 仍 −inf → NaN 污染 s_run，此后 `NaN·0+s_c=NaN` 不可恢复）——**tiled 特有**：single 形态（全行 max 先行）与 torch 均给有限值 9.70（V5 实测） | 有限值（**tiled 特有分歧**） | **V5 位级**（本设计新发现角点：兄弟变体 DESIGN V5 与其测试均未覆盖——首 chunk 覆盖率依赖 tile_n） |
| ⑤ | 尾 chunk 全 −inf（首个含有限值 chunk 之后） | 中性（m_run 有限 → `exp(−inf−m)=0` → s_c=0） | 有限值（一致） | V5 |
| ⑥ | 部分元素 −inf 散布（首个有限 chunk 之后） | 中性（同⑤逐元素级） | 一致 | V4 |
| ⑦ | 跨 chunk max 间隙 131008（±65504 极值对，fp16 域最大间隙） | flash 形式安全（合并式全部 exp 参数 ≤ 0，无溢出——朴素 m_c 基会溢出，`TRAP-NAIVE-ONLINE-MERGE`） | —（fp32 域恒安全） | V4 corner 位级 |
| ⑧ | 输出饱和（y > 65520，需 N > e^16 ≈ 8.86M） | `vcvt` sat 钳位（与源 `T.cast` 同行为）；契约空间内不可达（V7 封闭证明：N=102400 → y=65515.54 < 65520） | — | V7 |
| ⑨ | 退化 shape：N=1 / M=1 / N < BLOCK_N（num_full=0 纯尾路径）/ M < block_m（全尾批） | 正常路径承接（静态分支组合，§5.4） | 一致 | V1 tiny shape + L2 |

## 10. 交付清单

### 10.1 目录结构

```
examples/logsumexp_simd/_logsumexp_kernel_tiled/
├── _logsumexp_kernel_tiled.py    # Stage 3 产出：kernel + golden + 模拟器 + 分层测试
├── DESIGN.md                     # 本设计文档
├── RETROSPECTIVE.md              # 阶段复盘（Stage 1 章节随本设计落盘）
├── verify_equiv.py               # D-1 等价性机器验证（已执行，ALL EQUIV_PASS）
└── history_version/              # 设计分段稿 + 预算复算脚本（design_budget_calc.py）
```

### 10.2 文件清单（Stage 3 实现输入）

- `_logsumexp_kernel_tiled.py`：kernel builder（§3.3 伪代码 → 实现，docstring 声明 §10.3）+ `golden_logsumexp` / `_source_lse_tiled_sim`（§8.1）+ L0/L1/L2/Boundary 测试入口（§8.2，文件组织 mirror single 姊妹：单文件 kernel + golden + `run_L0`/`run_boundary` + `__main__` argparse）。
- 输入依赖：`DESIGN.md`（本文件，冻结基线）、`verify_equiv.py`（等价性锚点）、`history_version/design_budget_calc.py`（容量/复杂度数字复算源）。

### 10.3 命名与声明规范

- 模块 docstring 首部声明（mandatory）：

  ```python
  programming_mode: simd
  auto_schedule: yes
  ```

- 调用面（与 SIMT 兄弟变体一致，GPU threads 按 K3 移除）：`_logsumexp_kernel_tiled(M, N, dtype, tile_n)(block_m)(x) -> y`。
- `import tilelang.ascend.language as T`（Ascend 方言，AGENTS.md API 约定）；NUM_BLOCKS=56 常量注释注明实查来源（§5.5）。

### 10.4 实现顺序建议（Stage 3）

1. **Step 0**：golden + `_source_lse_tiled_sim` 落地，torch CPU 上对 verify_equiv 源链自测（§9.2-10）。
2. **Step 1**：builder 静态化 + L0 #1（(1024,32768) bf16 tn=16384 bm=1，rem=0 最简主流形态）——**首编即触发风险 1 验证**（alloc_var 跨块）；报错则按 fallback 切 stats UB 往返（备份后修改，§9.1-1）。
3. **Step 2**：尾 chunk 路径（L0 #4 (33,33000)：G_tail/k_tail/静态 rem 切片）+ L0 #2/#3（非 pow2 4-chunk）。
4. **Step 3**：M 尾批 + bm>1（L0 #8）+ 纯尾路径（L0 #7 fp32）+ L2 形态扫描。
5. **Step 4**：Boundary 18 类角点行（对模拟器）+ golden 分歧回归；L0 全量门槛。
6. **Step 5**：交 Stage 4（do_bench/msprof + npusim 归因，风险 4 标定）。

### 10.5 本设计引用的外部工件

- 源算子：`/home/z00875089/TileOPs-fork/tileops/kernels/reduction/logsumexp.py`（`_logsumexp_kernel_tiled` L105–194 + wrapper L338–395）+ `_primitives.py`。
- 同族先例：`examples/logsumexp_simd/_logsumexp_kernel_single/`（DESIGN.md / 实现 86 用例 PASS / RETROSPECTIVE）；`examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp_kernel/_logsumexp_kernel_tiled.py` + `_logsumexp_kernel_tiled_DESIGN.md`（SIMT 兄弟，61 用例 PASS + Stage 4）。
- API/结构先例：`examples/ascend/flash_attention/core.py`；`examples/pilot/online_softmax/online_softmax.py`；`examples/ascend/example_gemm.py`；`testing/ascend/language/test_tilelang_ascend_simdvf_cast.py`；`testing/ascend/language/simdvf/test_simdvf_arithmetic.py`；`testing/ascend/language/test_tilelang_ascend_simdvf_integer_predicate.py`；`examples/ascend/deepgemm/transform_sf.py`。
- manifest/测试：`examples/TileOPs/tileops/manifest/reduction.yaml`（LogSumExpFwdOpSimd L124–172）；`examples/TileOPs/tests/ops/test_logsumexp_simd.py`；`benchmarks/ops/bench_logsumexp_simd.py`。
- pattern-library / 标准：`CONST-capacity-a5` / `CONST-arch-a5` / `CONST-valu-model` / `CONST-mte-bandwidth-model` / `CONST-vrf-capacity-a5` / `CONST-gemm-dim-constraints`；`TRAP-NAIVE-ONLINE-MERGE` / `P-LAYOUT-ROW-RESIDENCY` / `P-LAYOUT-TAIL-STATIC-EXTENT` / `D-LOGSUMEXP-MIGRATION-FIDELITY` / `D-REDUCE-SIMTVF-API`；`VP-2026-0014`；`_shared/standards/core-split-strategy.md` / `algorithm-research.md` / `hardware-cost-model.md` / `negative-claim-evidence.md` / `gate-and-retry.md`。
- 源码机制锚点：`tilelang/language/allocate.py` L103–168；`tilelang/ascend/language/simd.py`；`src/ascend/transform/vf_regions.h` L50–53；`src/ascend/transform/insert_sync.cc` L1308；`src/ascend/transform/ascend_simdvf_lower_parallel.cc` L818–841；`src/tl_templates/ascend/simd_inst.h` L688–710；`src/ascend/transform/estimate_latency.h` L58–70。
