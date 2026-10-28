## 2. 编程模式选型

### 2.1 模式结论

**simd（SimdVF）**——用户显式指定（非默认 simt）；`auto_schedule: yes`（AutoSchedule 自动同步，无手动 flag）。kernel 模块 docstring 声明 `programming_mode: simd`、`auto_schedule: yes`（§10.3）。

### 2.2 选型理由

1. **用户显式指定**：本变体是 SIMD 家族（`examples/logsumexp_simd/`）的成员，用户在任务契约中指定 simd 承载——编程模式为输入约束，非选型变量（与 single 姊妹同族口径）。
2. **算子特征匹配**：纯 Vector 归约负载（无 GEMM/矩阵乘段）——special_func 主导（D-2 61%）+ 跨 lane 全归约（`vcmax`/`vcadd` 单指令）+ 逐元素向量算术，全部落在 AIV VALU；无 Cube 段 → 不需要 MixedKernel（mixed 仅当单 kernel 含 Cube `T.gemm` + Vector 段时使用，tilelang-ascend skill §3）。
3. **先例实证**：同族 single 姊妹以 SimdVF 实现 Stage 3 全量 PASS（86 用例，2026-10-17）——pass1/pass2/epilogue 指令链逐指令复用其验证过的形态；flash core.py / online_softmax streaming 为 chunk 流式 + flash 合并的 SimdVF/同构先例。
4. **SimdVF vs SimtVF 对本算子的差异**：显式 2048-bit 指令流（无线程域、编译器不自动向量化）换 per-chunk 指令级确定性（延迟归约/融合/掩码形态全部显式可控，O3/O4 的收益即来自显式形态优于自动 lower）。

### 2.3 模式影响

- **无 `threads=`**：`T.Kernel` 一维、无 threads 参数（TypeError）；行内并行度由 2048-bit 指令承载（每指令 64 fp32 / 128 fp16 lane），GPU `threads` 契约按 K3 舍弃（§0.5）。
- **`with T.SimdVF():` 域结构**：元素计算必须在域内以 `T.simd.*` 表达；`T.copy`（MTE2/MTE3）在域外；跨域值传递 = UB buffer 或 `T.simd.alloc_var`（local.var 寄存器）——本设计 m_run/s_run 跨 chunk 域驻留是后者（风险 1）。
- **SSA 限制**：域内 SSA 值不能跨域引用（`Immutable variable used outside its defining region`——single 姊妹 Stage 3 教训）；循环携带累加器必须 `alloc_var`（`test_simdvf_arithmetic.py` L86–110 先例）。
- **`T.simd.*` API 面**：装载（vld/vsts）、算术（vadd/vmax/vmul）、归约（vcmax/vcadd/vdupv）、超越（vexpdif/vln）、转换（vcvt）、谓词（vsel/update_mask）、常量（vdup/pset）——全部有先例（§3.4）。
- **AutoSchedule**：MTE2→VALU→MTE3 跨引擎依赖由编译器自动插 flag 对（§7）；完全手动同步不需要（无 `TL_ENABLE_AUTO_SCHEDULE: False`）。

## 3. API 映射设计

### 3.1 公式拆解

§1.6.1 优化后主链（逐式 → §3.2 API）：

```
per row:  (m_run, s_run) = (−inf, 0)                                   # alloc_var + vdup
per chunk c:
  A  载入:    x_ub ← T.copy(x[row, c·BLOCK_N : +BLOCK_N])              # MTE2 单行切片
  B  pass1:   acc = vmax(acc, vcvt(vld(x_ub[0, g·64]))) ×G → m_c = vdupv(vcmax(acc))
  C  max 合并: prev = m_run;  m_run = vmax(m_run, m_c)                  # 源 S3
  D  pass2:   e = vexpdif(vcvt(vld(x_ub[0, g·64])), m_run) ×G;  accs = vadd(accs, e) ×G
  E  sum 合并: s_run = vadd(vmul(s_run, vexpdif(prev, m_run)), vcadd(accs))   # 源 S6
  A′ 尾 chunk: T.copy rem 列切片; B–E 同上（组循环 G_tail + 尾组 vsel(−inf)）
epilogue:  y[r] = vsts(ONEPT, vcvt(vadd(m_run, vln(s_run))))           # lane 0
任务尾:    y[base:+rows] ← T.copy(lse_ub[0:rows])                      # MTE3
```

### 3.2 TileLang API 映射

| 计算步骤 | TileLang Ascend API（`import tilelang.ascend.language as T`） | 参数 | 先例 / 依据 |
|---------|------------------------------------------------|------|------------|
| kernel 入口 | `with T.Kernel(NUM_BLOCKS)` | NUM_BLOCKS=56（§5.5 实查） | single L215；一维强制（ascend-constraints §1 条 1） |
| 任务调度 | `for t in T.Persistent([num_tiles], NUM_BLOCKS, core_id, num_stages=2)` | num_tiles=ceil(M/block_m) | single L219 / SIMT tiled L231 同构 |
| UB 分配 | `T.alloc_shared((1, BLOCK_N), dtype)`；`T.alloc_shared((block_m,), dtype)` | x_ub / lse_ub | single L216–217（形状不同：单行 chunk 粒度） |
| 双缓冲声明 | `T.annotate_buffer_versions({x_ub: 2, lse_ub: 1})` | — | single L218 / SIMT tiled L230 |
| GM→UB chunk 载入 | `T.copy(x[row:row+1, c·BLOCK_N:(c+1)·BLOCK_N], x_ub[0:1, 0:BLOCK_N])` | (1, BLOCK_N) 单行切片（row 为 TIR expr） | SIMT tiled L156–159（rows=1 特例同构） |
| UB→VRF 装载 + 提升 | `T.simd.vld(x_ub[0, g·64], dist="UNPK_B16")` + `T.simd.vcvt(·, "float32", part=0)`；fp32 输入 `T.simd.vld(x_ub[0, g·64])` NORM 直连 | 组宽 64 | cast 回归 L64–80 位级；single L154–162 |
| 行 max 累加 | `acc = T.simd.vmax(acc, cv)` | — | single L163 |
| 跨 lane 全归约 + 广播 | `T.simd.vcmax(acc)` + `T.simd.vdupv(·)` | — | flash core.py L115–118；single L175 |
| max 合并（寄存器） | `m_run = T.simd.vmax(m_run, m_c_brc)` | broadcast 形态 | vmax 同上；alloc_var 承载（下行） |
| running stats 驻留 | `m_run = T.simd.alloc_var("float32")`；`s_run = T.simd.alloc_var("float32")`（行作用域声明、域内读写） | local.var 可变寄存器 | `tilelang/language/allocate.py` L103–168；`test_simdvf_arithmetic.py` L86–110 循环携带先例；**跨块声明+访问 = 本设计新组合（风险 1 + fallback）** |
| 融合稳定化指数 | `e = T.simd.vexpdif(cv, m_run)` | fp32-only 强校验 | flash core.py L130–133；VP-2026-0014 |
| 行 sum 累加 + 全归约 | `accs = T.simd.vadd(accs, e)`；`s_c = T.simd.vcadd(accs)` | lane-0 形态 | flash core.py L134–137；single L190/L205 |
| flash 重缩放合并 | `s_run = T.simd.vadd(T.simd.vmul(s_run, T.simd.vexpdif(prev_brc, m_run)), s_c)` | — | vmul/vadd 单指令；代数 = 源 S6 |
| 尾组掩码 | `cv = T.simd.vsel(cv, neg_inf, T.simd.update_mask(k_tail, 32))`（neg_inf = `T.simd.vdup(−inf, "float32")`） | fp32 域 | deepgemm transform_sf L205/L215/L249；single L173 |
| epilogue | `out = T.simd.vadd(m_run, T.simd.vln(s_run))`；16-bit 输出 `out_h = T.simd.vcvt(out, dtype)` | — | single L206/L210 |
| 单点存储 | `T.simd.vsts(lse_ub[r], out_h, T.simd.pset(16, "PAT_VL1"), "ONEPT_B16")`（fp32：`pset(32,…)` + `"ONEPT_B32"`） | lane 0 | single L208/L211 |
| lse 写回 | `T.copy(lse_ub[0:rows], y[row_base:row_base+rows])` | 静态切片 | SIMT tiled L222 |
| JIT 入口 | `@tilelang.jit(target="ascend", out_idx=[1])` 懒工厂：`_logsumexp_kernel_tiled(M, N, dtype, tile_n)(block_m)(x) -> y` | — | single L115 同款（§4.7 调用面契约） |

### 3.3 计算伪代码

```python
# host 静态化（builder 闭包 `_logsumexp_kernel_tiled(M, N, dtype, tile_n)`，
#   per-(M, N, dtype, tile_n) JIT + lru_cache；`_func(block_m)` 懒编译，`(x) -> y`，§4.7）：
#   elem     = 4 if dtype == "float32" else 2
#   BLOCK_N  = 最大 64 倍数 s.t. BLOCK_N <= tile_n 且 2*BLOCK_N*elem <= 228556 - 8256
#              （clamp 上界：16-bit 55,040 / fp32 27,520；wrapper tile_n <= 32,512
#               → 16-bit 恒不触发；fp32 或显式超大 tile_n 由 clamp 兜底，§4.5）
#   G        = BLOCK_N // 64                     # full chunk 组数
#   num_full = N // BLOCK_N                      # full chunk 数（N < BLOCK_N 时 0）
#   rem      = N % BLOCK_N                       # 尾 chunk 列数（0 = 无尾 chunk）
#   G_tail   = rem // 64;  k_tail = rem % 64     # 尾 chunk 满组数 / 尾组有效元素
#   full_m   = M // block_m;  rem_m = M % block_m
#   num_tiles = full_m + (1 if rem_m > 0 else 0)
#   NUM_BLOCKS = 56                              # = vector_core_num（实查，§5.5）
# 契约防御（builder 内 raise）：M >= 1; N >= 1; tile_n >= 1; block_m >= 1
#   （无 N 上限——任意大 N 是本变体契约；无 BLOCK_N <= N 假设——N < BLOCK_N 走纯尾 chunk 路径）

@T.prim_func
def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M,), dtype)):
    with T.Kernel(NUM_BLOCKS) as core_id:            # 一维 grid，无 threads=（K3）
        x_ub = T.alloc_shared((1, BLOCK_N), dtype)   # UB：单行 chunk 粒度，双版本流水
        lse_ub = T.alloc_shared((block_m,), dtype)
        T.annotate_buffer_versions({x_ub: 2, lse_ub: 1})

        @T.macro
        def row_task(row_base, rows):                # row_base: TIR expr; rows: Python 常量
            for r in range(rows):                    # 行串行（主流 bm=1 → 单迭代退化）
                row = row_base + r
                # —— running stats：行作用域 alloc_var（SimdVF 块外声明，风险 1）——
                m_run = T.simd.alloc_var("float32")  # broadcast 形态（vdupv 产出/更新）
                s_run = T.simd.alloc_var("float32")  # lane-0 形态（vcadd 产出，仅 lane 0 消费）
                with T.SimdVF():                     # init（源 S-init）
                    m_run = T.simd.vdup(-T.infinity(T.float32), "float32")
                    s_run = T.simd.vdup(T.float32(0.0), "float32")
                # —— full chunks：online 流式（源 S1a→S6）——
                for c in T.serial(num_full):
                    T.copy(x[row:row+1, c*BLOCK_N:(c+1)*BLOCK_N],
                           x_ub[0:1, 0:BLOCK_N])     # (1, BLOCK_N) 单行 MTE2
                    with T.SimdVF():                 # per-chunk 两遍 + flash 合并
                        neg_inf = T.simd.vdup(-T.infinity(T.float32), "float32")
                        zero_f  = T.simd.vdup(T.float32(0.0), "float32")
                        acc = T.simd.alloc_var("float32"); acc = neg_inf
                        for g in range(G):           # pass 1: chunk max（延迟归约）
                            cv = vld+vcvt(x_ub[0, g*64])      # fp32 输入: vld NORM 直连
                            acc = T.simd.vmax(acc, cv)
                        m_c_brc = T.simd.vdupv(T.simd.vcmax(acc))
                        prev_brc = m_run             # SSA 快照（源 S3 的 prev_max）
                        m_run = T.simd.vmax(m_run, m_c_brc)
                        accs = T.simd.alloc_var("float32"); accs = zero_f
                        for g in range(G):           # pass 2: sum of exp（UB 重读 + 融合）
                            cv = vld+vcvt(x_ub[0, g*64])
                            e = T.simd.vexpdif(cv, m_run)
                            accs = T.simd.vadd(accs, e)
                        s_c = T.simd.vcadd(accs)     # lane-0 形态
                        s_run = T.simd.vadd(         # flash 重缩放（源 S6）
                            T.simd.vmul(s_run, T.simd.vexpdif(prev_brc, m_run)), s_c)
                # —— 尾 chunk（rem > 0 时 Python 静态实例化；源 S1b 的静态化形态）——
                if rem > 0:
                    T.copy(x[row:row+1, num_full*BLOCK_N:N], x_ub[0:1, 0:rem])
                    with T.SimdVF():
                        neg_inf = ...; zero_f = ...  # 同上（域内常量重物化）
                        if k_tail > 0:
                            tail_mask = T.simd.update_mask(k_tail, 32)
                        acc = ...; acc[初始化 neg_inf]
                        for g in range(G_tail):      # 满组同 full
                            cv = vld+vcvt(x_ub[0, g*64]); acc = T.simd.vmax(acc, cv)
                        if k_tail > 0:               # 尾组：−inf 掩码后进算术
                            cv = vld+vcvt(x_ub[0, G_tail*64])
                            cv = T.simd.vsel(cv, neg_inf, tail_mask)
                            acc = T.simd.vmax(acc, cv)
                        m_c_brc = ...; prev_brc = m_run; m_run = ...   # 同 full
                        accs = ...[初始化 zero_f]
                        for g in range(G_tail):
                            cv = vld+vcvt(x_ub[0, g*64])
                            accs = T.simd.vadd(accs, T.simd.vexpdif(cv, m_run))
                        if k_tail > 0:
                            cv = vld+vcvt(x_ub[0, G_tail*64])
                            cv = T.simd.vsel(cv, neg_inf, tail_mask)
                            accs = T.simd.vadd(accs, T.simd.vexpdif(cv, m_run))
                        s_c = T.simd.vcadd(accs)
                        s_run = ...                  # 同 full flash 重缩放
                # —— 行尾 epilogue（源 S7/S8）——
                with T.SimdVF():
                    out = T.simd.vadd(m_run, T.simd.vln(s_run))
                    if dtype == "float32":
                        one_pt = T.simd.pset(32, "PAT_VL1")
                        T.simd.vsts(lse_ub[r], out, one_pt, "ONEPT_B32")
                    else:
                        one_pt = T.simd.pset(16, "PAT_VL1")
                        out_h = T.simd.vcvt(out, dtype)
                        T.simd.vsts(lse_ub[r], out_h, one_pt, "ONEPT_B16")
            T.copy(lse_ub[0:rows], y[row_base:row_base+rows])   # 任务尾 MTE3（静态切片）

        for t in T.Persistent([num_tiles], NUM_BLOCKS, core_id, num_stages=2):
            if t < full_m:                           # TIR if：M 尾批边界（源 R4）
                row_task(t * block_m, block_m)       # 全块（rows=block_m 静态）
            else:
                row_task(M - rem_m, rem_m)           # 尾块（rows=rem_m 静态）
```

注：`vld+vcvt(x_ub[0, g*64])` 展开为 `T.simd.vcvt(T.simd.vld(x_ub[0, g*64], dist="UNPK_B16"), "float32", part=0)`（fp32 输入为 `T.simd.vld(x_ub[0, g*64])` 单指令，无 vcvt）；`acc = ...[初始化 neg_inf]` 等省略号处 = 与 full chunk 完全相同的语句。宏两次实例化（rows=block_m / rem_m）与 SIMT tiled 兄弟 `sub_batch(row_base, rows)` 同构（L145–250 Stage 3 实证）；`if rem > 0` / `if k_tail > 0` 为 Python 静态分支（trace 期消除，非 TIR 运行时分支）。

### 3.4 API 可行性确认

**逐 API 先例锚点**（指令链整体 = single 姊妹 Stage 3 全量 PASS 的复用 + tiled 契约结构 = SIMT tiled 兄弟 Stage 3 实证的复用；`tilelang/ascend/language/simd.py` 为签名权威）：

| API | 证据 | 状态 |
|-----|------|------|
| vdup / vmax / vadd / vmul / vcvt(part=0) / vld(UNPK_B16) / vsts(ONEPT_B16/B32) / pset(PAT_VL1) / vsel / update_mask / vexpdif / vln / vcmax / vcadd / vdupv | single 姊妹 `_logsumexp_kernel_single.py` L135–211 同族指令链（86 用例 Stage 3 全量 PASS，2026-10-17）；cast 位级回归 `test_tilelang_ascend_simdvf_cast.py` L64–80；flash core.py L94–137（vcmax/vcadd/vexpdif/BRC 实测）；deepgemm transform_sf（vsel/update_mask） | ✅ 实证 |
| `T.copy` (1, BLOCK_N) 单行切片 + 静态 rem 切片 + lse_ub 切片 | SIMT tiled 兄弟 L156–159/L191/L222（tiled 契约同款切片，61 用例 PASS）；`D-REDUCE-SIMTVF-API`：掩码载入与 DMA 紧凑拷贝正交 | ✅ 实证 |
| `T.Persistent` + TIR if 尾批 + `T.macro(row_base: TIR, rows: int)` | single L219–229 + SIMT tiled L231–250（宏带 TIR row_base 实证） | ✅ 实证 |
| `T.simd.alloc_var` 域内声明 + `T.serial` 循环携带 | `test_simdvf_arithmetic.py` L86–110 serial_vreg_accum（NPU PASS） | ✅ 实证（域内形态） |
| **`T.simd.alloc_var` 域外声明 + 多个 per-chunk SimdVF 块内读写（m_run/s_run）** | 机制链：`allocate.py` L103–168（声明无 SimdVF 上下文检查）+ `vf_regions.h` L50–53（SIMD_VF keeps the enclosing scope——域体可访问外围作用域分配的 buffer）+ `insert_sync.cc` L1308（local.var 依赖跳过跨引擎同步插入）；**无单一先例覆盖该组合** | ⚠️ **风险 1**：Stage 3 首编验证；fallback = stats UB 往返（`stats_ub (2,) fp32` + flash core.py L103/L117/L120–126 形态：chunk 末 `vsts(m_ub, ·, ONEPT_B32)` / chunk 首 `vld(m_ub, "BRC_B32")` 广播装载 + `mem_bar("VST_VLD")`，~5 指令/chunk/行 = wl2 0.14%，API 全先例、语义等价） |

### 3.5 技术约束确认

**3.5.1 本项目已知限制检查（ascend-constraints §2 强制检测）**：

| 约束 | 本设计处置 | 状态 |
|------|-----------|------|
| `T.Kernel` 一维、无 `threads=` | `T.Kernel(56)` 一维；GPU `threads` 契约按 K3 舍弃（§0.5）；调用面 `(block_m)` 单参数 | ✅ |
| C0 分形对齐（GEMM 类 16 倍数） | n/a——纯 Vector 归约，无 `T.gemm`/L0 路径 | ✅ |
| UB 容量（248KB 物理；SimdVF 无 32KB 线程上下文扣减） | x_ub 双版本 + lse_ub + 8,256B 预留 ≤ 228,556B（90% 折扣预算，§4.5 表：最大 fp32 clamp 档 99.9%） | ✅ |
| `T.copy` OOB 行为未文档化 | 全部切片静态（full/rem/rows_this 推导自编译期常量；行循环只迭代有效行）——不依赖任何越界保护（§0.6 R4） | ✅ |
| SimdVF 域 SSA 限制 | 循环携带/跨块值一律 alloc_var；域内 SSA（prev_brc、m_c_brc、e、cv）仅域内消费 | ✅ |
| `vexpdif` fp32-only 强校验 | 装载即 vcvt 提升 fp32（fp16/bf16）；fp32 输入直连——全链 fp32 算术 | ✅ |
| flag/event 预算（核内 8 / 跨核 16） | auto_schedule: yes——AutoSchedule 管理，无手动 flag（§7） | ✅ |
| L1/L0A/L0B/L0C | 不使用（纯 UB+VRF 路径） | ✅ |

**3.5.2 参考实现差异说明**：与 GPU 源的结构差异（并行结构 / chunk 粒度 / 双路径 / 驻留 / OOB）全部经 §0.5 四态判定 + §0.6 重设计论证，数值链经 V1–V7 机器验证——非逐 API 语法翻译（§0 迁移决策驱动 §1–§7）。

**3.5.3 本项目同类实现参考**：`examples/logsumexp_simd/_logsumexp_kernel_single/`（指令链 + Persistent 嵌套 + 契约防御 + golden 组织）；`examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp_kernel/_logsumexp_kernel_tiled.py`（tiled 契约的切片/宏/尾批结构）；`examples/pilot/online_softmax/online_softmax.py`（streaming 路径）；`examples/ascend/flash_attention/core.py`（stats 访问形态 + fallback）。

---
