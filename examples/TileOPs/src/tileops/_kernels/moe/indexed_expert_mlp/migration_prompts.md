# Migration prompts — IndexedExpertMLPFwdOp (Stage 0 scaffold output)

One prompt per extracted TileLang kernel function (3 functions), for the
TileLang-Ascend 算子端到端开发编排 Agent (`tilelang-op-conductor`), Stage 1 → 4.
Shared context: extracted reference file
`src/tileops/_kernels/moe/indexed_expert_mlp/_indexed_expert_m_l_p_fwd_kernels.py`
(auto-generated name from `extract_tl_kernel.py --out <dir>`); conversion record
`src/tileops/_kernels/moe/indexed_expert_mlp/conversion_record.md`; migration meta
`src/tileops/_kernels/moe/indexed_expert_mlp/.migration_meta.json`.

---

## Prompt 1 — `_route_stats`

```text
迁移 indexed_expert_mlp 算子 IndexedExpertMLPFwdOp。 其 TileLang 实现函数 _route_stats 在 src/tileops/_kernels/moe/indexed_expert_mlp/_indexed_expert_m_l_p_fwd_kernels.py。
_route_stats 是对外接口，不要改变接口的参数(除非该参数跟后端实现相关，在迁移时需要适配)。
IndexedExpertMLPFwdOp 的规格见 examples/TileOPs/src/tileops/manifest/spec/moe.yaml中 IndexedExpertMLPFwdOp 的 workloads 部分。
精度用例见 examples/TileOPs/tests/ops/test_indexed_expert_mlp.py，
性能用例见 examples/TileOPs/benchmarks/ops/bench_indexed_expert_mlp.py
编程模式由 Stage 1 设计裁定（默认 simt，auto_schedule yes）；kernel 模块 docstring 须声明 programming_mode: simt|simd|mixed。
上下文：原始定义 src/tileops/_kernels/moe/indexed_expert_gemm.py（GPU 仓根 /home/l00970450/z00875089/TileOPs，见 meta gpu_source.function_sources）；本函数是单 block 路由统计 kernel（per-expert 计数 + per-route 秩 + 每 expert 路由列表，metadata 布局 E + T*K + E*T int32，含 T.atomic_add return_prev=True），A5 wrapper 调用形如 self.kernel(**self.config)(expert_ids, out)，config 为空 dict（threads 已按 K3/K9 移除，线程域由重实现内部 T.SimtVF 声明）；分支谓词 grouped_dispatch = (T > 1) 见 conversion_record.md §3。
```

---

## Prompt 2 — `_indexed_expert_gemm`

```text
迁移 indexed_expert_mlp 算子 IndexedExpertMLPFwdOp。 其 TileLang 实现函数 _indexed_expert_gemm 在 src/tileops/_kernels/moe/indexed_expert_mlp/_indexed_expert_m_l_p_fwd_kernels.py。
_indexed_expert_gemm 是对外接口，不要改变接口的参数(除非该参数跟后端实现相关，在迁移时需要适配)。
IndexedExpertMLPFwdOp 的规格见 examples/TileOPs/src/tileops/manifest/spec/moe.yaml中 IndexedExpertMLPFwdOp 的 workloads 部分。
精度用例见 examples/TileOPs/tests/ops/test_indexed_expert_mlp.py，
性能用例见 examples/TileOPs/benchmarks/ops/bench_indexed_expert_mlp.py
编程模式由 Stage 1 设计裁定（默认 simt，auto_schedule yes）；kernel 模块 docstring 须声明 programming_mode: simt|simd|mixed。
上下文：原始定义 src/tileops/_kernels/moe/indexed_expert_gemm.py（GPU 仓根 /home/l00970450/z00875089/TileOPs，见 meta gpu_source.function_sources）；本函数是 route 索引 expert GEMM 模板（block_m=16、dispatch_mode direct|grouped、route_input 直通|组行装载、gated silu_and_mul epilogue 融合，T.Kernel(num_routes*n_blocks, threads=threads) 二维索引 pid 分解 + T.gemm(transpose_B=True)），A5 wrapper 调用形如 self.kernel(**{"block_n","block_k","num_stages"})(a, weights, expert_ids, route_metadata, out)（threads 按 K3/K9 移除；A5 为一维 T.Kernel，GEMM 主导候选 MixedKernel/Cube 路径见 docs/gpu_to_ascend_adaptation.md §2.3）；A5 wrapper 侧两个实例化：IndexedExpertGateUpKernel（activation=silu_and_mul, route_input=False, n=F, k=H）与 IndexedExpertDownKernel（activation=none, route_input=True, n=H, k=F）见 conversion_record.md §3。
```

---

## Prompt 3 — `_weighted_reduce`

```text
迁移 indexed_expert_mlp 算子 IndexedExpertMLPFwdOp。 其 TileLang 实现函数 _weighted_reduce 在 src/tileops/_kernels/moe/indexed_expert_mlp/_indexed_expert_m_l_p_fwd_kernels.py。
_weighted_reduce 是对外接口，不要改变接口的参数(除非该参数跟后端实现相关，在迁移时需要适配)。
IndexedExpertMLPFwdOp 的规格见 examples/TileOPs/src/tileops/manifest/spec/moe.yaml中 IndexedExpertMLPFwdOp 的 workloads 部分。
精度用例见 examples/TileOPs/tests/ops/test_indexed_expert_mlp.py，
性能用例见 examples/TileOPs/benchmarks/ops/bench_indexed_expert_mlp.py
编程模式由 Stage 1 设计裁定（默认 simt，auto_schedule yes）；kernel 模块 docstring 须声明 programming_mode: simt|simd|mixed。
上下文：原始定义 src/tileops/_kernels/moe/indexed_expert_gemm.py（GPU 仓根 /home/l00970450/z00875089/TileOPs，见 meta gpu_source.function_sources）；本函数是加权归约 kernel（fp32 累加 sum_k expert_output[t,k,:]*topk_weights[t,k] × scaling 后 cast 写回 [T,H]；GPU 以 threads=256 一维分块 + T.get_thread_binding 展平索引），A5 wrapper 调用形如 self.kernel(**self.config)(expert_output, topk_weights, output)，config 为空 dict（threads 已按 K3/K9 移除）；elementwise 归约族参考 docs/gpu_to_ascend_adaptation.md §2.1/§2.2。
```
