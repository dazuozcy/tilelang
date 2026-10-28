# 算子族开发指南（family-guide）

> 导航层参考：每族一节，给出模式决策、参考实现锚点、结构骨架（示意伪代码，
> 非可直接运行代码）与易错点映射。API 语义与约束权威在
> [tilelang-ascend skill](../../tilelang-ascend/SKILL.md)（下称"§"），
> 本文件不复制其内容。

## 1. elementwise（逐元素 / 量化 / cast）

### 模式决策

- 默认 `simt`：`T.SimtVF(threads=N)` + `T.Parallel`，编译器自动向量化与自动调度
  （§2.2），跨线程数据冒险由编译器自动插 `asc_syncthreads()`（§10.8）。
- 降级 `simd` 的条件：profiling 证实 VF 是瓶颈，或需要特定掩码/舍入/求值顺序
  （§8.2 的 per_token_cast_to_fp8 SIMD 版即为此例）。
- 带宽主导型算子的优化重点是 `T.Pipelined` 流水深度、tile 大小与
  buffer versions（§4.2），而非指令选择。

### 参考实现锚点

- `examples/ascend/example_vecadd.py`：simt/simd 双模式对照 + `T.Pipelined`
  双缓冲流水，是本族的标准模板。
- `examples/ascend/example_per_token_cast_to_fp8.py`：分组量化，simt 与显式
  simd 两版实现。
- 回归：`testing/ascend/language/simtvf/`、`simdvf/`、
  `test_tilelang_ascend_simdvf_predicate_load_store.py`、
  `test_tilelang_ascend_simdvf_merging.py`。

### 结构骨架（示意）

```python
@T.prim_func
def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
    with T.Kernel(num_blocks) as bx:
        a = T.alloc_shared((tile,), dtype)      # UB
        b = T.alloc_shared((tile,), dtype)
        c = T.alloc_shared((tile,), dtype)
        T.annotate_buffer_versions({a: num_stages, b: num_stages, c: num_stages})
        for step in T.Pipelined(iterations, num_stages=num_stages):
            begin = (step * num_blocks + bx) * tile
            T.copy(A[begin : begin + tile], a)   # GM→UB（§6.1）
            T.copy(B[begin : begin + tile], b)
            with T.SimtVF(threads=num_threads):  # simt 模式（§2.2）
                for i in T.Parallel(tile):
                    c[i] = a[i] + b[i]           # 热点处可降级为 T.simd.*（§2.3/§8）
            T.copy(c, C[begin : begin + tile])   # UB→GM
```

### 易错点

`threads=` 不能出现在 `T.Kernel` 上（§10.2）；UB 容量按 tile×stages×3 份预算
（§4.3）；GM→UB 尾部 32B 填充用 `pad_value=`/`data_select=`（§6.1）；
L2 策略默认值随路径不同（§6.1 表）。

## 2. gemm（含 split-K / 量化 GEMM）

### 模式决策

- `mixed`：`T.Kernel(N)` 内同时含 `T.gemm`（Cube 侧）与 Vector 侧工作时，
  codegen 自动识别；需要 AIV 子核 ID 或显式两 AIV 启动时用
  `T.MixedKernel(N, sids=2)`（§3.1/§3.2）。
- 纯 Cube 计算无需显式 `T.Cube()` 域；完全手动核（`TL_ENABLE_AUTO_SCHEDULE:
  False` + 手写 flag）仅在自动调度被证实不足时使用（§3.3）。
- split-K：`T.set_atomic("add", dtype)` 累加式或分块有序归约，
  注意初始化归属与确定性差异（§5.3）。

### 参考实现锚点

- `examples/ascend/example_gemm.py`：persistent 遍历 + swizzle + NT GEMM +
  mixed epilogue（`T.dual_copy` 双段回写）+ HF32/累加选项，是本族标准模板。
- `examples/ascend/example_gemm_splitk.py`：split-K 原子/有序归约。
- `examples/ascend/deepgemm/`：进阶（L0 staging、量化 GEMM、mixed epilogue
  见 `kernels/epilogue.py`）。
- 回归：`testing/ascend/language/test_tilelang_ascend_l0_gemm_effective_k.py`。

### 结构骨架（示意）

```python
@T.prim_func
def main(X: T.Tensor((M, K), dtype), W: T.Tensor((N, K), dtype), C: T.Tensor((M, N), out_dtype)):
    with T.Kernel(NUM_BLOCKS) as bx:            # 或 T.MixedKernel(NUM_BLOCKS, sids=2)
        res = T.alloc_l0c((TILE_M, TILE_N), "float32")   # 累加器（§4.1）
        x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)       # CBuf
        w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)       # 权重 [N, K]（§5.2）
        temp = T.alloc_shared((TILE_M // 2, TILE_N), out_dtype)  # dual_copy 半行
        for tile_idx in T.Persistent([OUT_TILES], NUM_BLOCKS, bx):
            m_tile, n_tile = swizzle(tile_idx)  # 2-D 分解自行计算（§10.3）
            for kt in T.Pipelined(K_TILES, num_stages=NUM_STAGES):
                T.copy(X[m_tile * TILE_M : ...], x_l1)   # GM→L1，ND2NZ 自动（§6.1）
                T.copy(W[n_tile * TILE_N : ...], w_l1)
                T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))
            T.dual_copy(res, temp)              # L0C→UB，M-split 双 AIV（§6.2）
            T.dual_copy(temp, C[m_tile * TILE_M : ..., ...])  # UB→GM
```

### 易错点

L1 输入路径强制 `trans_A=False, trans_B=True`（§10.1/§5.1）；权重 `[N, K]`
布局（§10.4）；fp16/bf16 累加 dtype 契约（§10.6）；auto_schedule 下编译器会
加版本维度，勿与显式 slot 维度重复（§10.5）；L0C→UB 的 M-split 目标只有半行
（§10.9）；两 AIV 分摊输出时的归属划分（§10.10）。

## 3. attention（MHA / GQA）

### 模式决策

- `mixed`：QK^T 与 PV 两次 GEMM 走 Cube，在线 softmax（行 max / 指数 / 归一
  化）走 Vector，softmax 打包用显式 `T.simd.*`。
- 自动调度起步；需要显式阶段约束时用 `T.Stage`（保持 AutoSchedule 开启），
  对照 `example_gqa_manual_schedule.py`，协议见
  tilelang-ascend skill references/synchronization.md。

### 参考实现锚点

- `examples/ascend/flash_attention/example_mha.py` / `example_gqa.py`：
  前向标准模板（`core.py` 为共享 builder）。
- `examples/ascend/flash_attention/example_gqa_manual_schedule.py`：显式
  `T.Stage` 流水版。
- `examples/ascend/flash_attention/example_gqa_bwd.py` + `core_bwd.py`：反向。
- 关键结构：`core.py` 将 softmax 概率**直接写 NZ 布局**供第二 GEMM 消费，
  省一次中间转换；该路径当前绑定 `head_dim == 128`（读 `core.py` 头注）。

### 结构骨架（示意）

```python
with T.Kernel(NUM_BLOCKS) as bx:                 # MixedKernel 或自动识别
    q_l1, k_l1, v_l1 = ..., ..., ...            # T.alloc_l1 分块（§4.1）
    s_l0c = T.alloc_l0c((BLOCK_Q, BLOCK_KV), "float32")
    o_l0c = T.alloc_l0c((BLOCK_Q, HEAD_DIM), "float32")
    m_ub, l_ub = ..., ...                        # 在线 softmax 状态（UB）
    p_ub = ...                                   # 概率 tile，直接写 NZ 布局
    for kv in T.Pipelined(KV_TILES, num_stages=PIPELINE_STAGES):
        T.copy(K[kv * BLOCK_KV : ...], k_l1)     # GM→L1（§6.1）
        T.copy(V[kv * BLOCK_KV : ...], v_l1)
        T.gemm(q_l1, k_l1, s_l0c, transpose_B=True, clear_accum=...)  # QK^T
        # Vector 侧（SimdVF + T.simd.*）：行 max 更新 → vexp → 概率打包 NZ
        #   （vmax/vexp/vcvt + vintlv/vdintlv，见 core.py 实际用法，§8）
        T.gemm(p_nz, v_l1, o_l0c, transpose_B=True, ...)              # PV 累加
    # epilogue：L0C→UB→GM；scale 放置与行归一化在 Vector 侧收尾
```

### 易错点

保持 head 映射、mask、scale 位置、归一化与累加顺序（§9 对 attention 的要求）；
AIV 输出归属与 dual_copy 形状关系（§10.9/§10.10）；非 32B 对齐行列与尾块
（§6.3）；改反向时必须验证全部梯度输出（§9）。

## 4. reduction-norm（归约 / 归一化）

### 模式决策

- 默认 `simt`：`T.alloc_fragment` 向量化加载（2 的幂填充 + 谓词掩尾）+
  `T.Parallel` 归约；VF 内延迟归约用 reducer epoch API
  （`T.alloc_reducer` + `reducer_init/update/finalize_reducer`，§4.1）。
- 跨线程归约不需要手写 barrier（编译器自动，§10.8）。
- 小规模硬件归约：`T.warp_reduce_sum/max/min` 直连硬件 intrinsic，仅支持
  float16/float32/int32/uint32（§10.11）；其余 dtype 走 fallback，勿假设直连。

### 参考实现锚点

- `examples/ascend/example_rmsnorm.py`：fragment 加载、reducer、行批处理
  （多行批摊薄 VF 启动与 RSTD 小写出）、UB 预算推导，是本族标准模板。
- 回归：`testing/ascend/language/test_tilelang_ascend_reduce.py`、
  `test_tilelang_ascend_warp_reduce.py`、`test_tilelang_ascend_warp_vote.py`。

### 结构骨架（示意）

```python
with T.Kernel(N_CORES) as core_id:
    w_ub = T.alloc_shared((d,), dtype)
    x_ub = T.alloc_shared(rows * d, "float32")
    T.copy(W[:d], w_ub[:d])
    for t in T.Pipelined(iterations, num_stages=2):
        T.copy(X[base : base + rows * d], x_ub)  # GM→UB
        with T.SimtVF(threads=threads):
            x_frag = T.alloc_fragment((rows, TILE), "float32")  # 2 的幂填充
            for r, i in T.Parallel(rows, TILE):
                x_frag[r, i] = T.if_then_else(i < d, x_ub[r * d + i], T.float32(0))
            acc = T.alloc_reducer((rows,), "float32", op="sum")
            T.reducer_init(acc)
            for r, i in T.Parallel(rows, TILE):
                if i < d:
                    T.reducer_update(acc[r], x_frag[r, i] * x_frag[r, i])
            res = T.alloc_fragment((rows,), "float32")
            T.finalize_reducer(acc, res)         # finalize 后才可读 res（§10.7）
            for r in T.Parallel(rows):
                res[r] = T.rsqrt(res[r] / d + eps)
            # 乘权重写回 x_ub/y_ub → T.copy 回 GM
```

### 易错点

reducer 句柄不可直接读/赋值/`T.clear`/`T.fill`/原地 finalize，init 与 finalize
须在同一作用域（§10.7）；归一化与其他算子融合时保持中间 cast 与精度（§9）；
`T.Parallel` extent 不必等于线程数（tilelang-semantic：分区/复制/掩尾允许差异）。
