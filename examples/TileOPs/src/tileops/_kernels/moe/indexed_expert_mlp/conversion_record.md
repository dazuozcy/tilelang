# Stage 0 conversion record — IndexedExpertMLPFwdOp (GPU → Ascend A5)

Source: `/home/l00970450/z00875089/TileOPs` @ `3979278e2da6ac4fa84d8c7c800dec64ea9c0425` (clean).
Target: `examples/TileOPs/` (A5, `target="ascend"`, dav-3510). Meta:
`.migration_meta.json` in this directory. Discovery sidecar: `stage0_discovery.json`
(extractor `--source-info-out`; merged reviewed fields into the meta).

## 1. Interface / schema conversion table

| GPU source (`indexed_expert_gemm.py` / `indexed_routed_expert.py`) | A5 target (`indexed_expert_mlp.py` / `_ops/moe/indexed_expert_mlp.py`) | Note |
| --- | --- | --- |
| `@tilelang.jit` factories `_route_stats(T,K,E)` / `_indexed_expert_gemm(T,K,E,n,k,dtype,activation,route_input,dispatch_mode)` / `_weighted_reduce(T,K,H,dtype,scaling)`, inner callables take `threads` (+ `block_n, block_k, num_stages` for the GEMM) | Extracted verbatim into `_indexed_expert_m_l_p_fwd_kernels.py` (+ `_route_metadata_size` appended manually). Stage 1-3 re-implements each with `@tilelang.jit(target="ascend")`, Ascend dialect, callable minus `threads` (K1/K3) | Extracted reference only; not runnable on A5 until re-implemented |
| `supported_archs = [80, 86, 89, 90]` | `supported_archs = ["Ascend950"]` | K5 |
| `get_shared_memory_optin(self.device_index)` (`torch.cuda` props) | `get_device_backend().shared_memory_budget(device_index)` (UB: 248 KiB / 216 KiB SimtVF) | K6 |
| `default_config`: route_stats `{"threads": 128}`; GEMM `{"block_n", "block_k", "threads", "num_stages"}`; weighted_reduce `{"threads": 256}` | route_stats `{}`; GEMM `{"block_n", "block_k", "num_stages"}`; weighted_reduce `{}` — `threads` removed everywhere | K3/K9 |
| `self.program(self._compile)(args...)` (generations/CompiledProgram) | `self.kernel(**self.config)(args...)` (plain jit factory call) | A5 Kernel base has no program machinery |
| `Entry` / `entry_for(call)` build identities + `Op.kernel_for` / `call_spec` | `Op._kernel_for(key, call, device_index)` dict cache; identities preserved verbatim as cache keys (incl. `grouped_dispatch`) | adapter §S3/S4 |
| `IndexedExpertCall` (frozen dataclass on `CallSpec`) + 4 `KernelInterface` classes | Local frozen dataclass `IndexedExpertCall` (same fields + `grouped_dispatch` property); interface contracts preserved as wrapper `forward` validations | no CallSpec/KernelInterface on A5 |
| GPU op `__init__(routed_scaling_factor=1.0, *, target: Target)` | `__init__(routed_scaling_factor=1.0, *, kernel_map=None)` | Target registry → A5 kernel_map override (O3/O4) |
| `kernel_types` / `interfaces` ClassVar maps | `default_kernel_map` property: `route_stats`/`expert_gate_up`/`expert_down`/`weighted_reduce` → the four A5 kernel classes | same keys |
| Op.forward: `self.call_spec("expert_gate_up", lambda: IndexedExpertCall(...))` + `kernel_for(...)` chain | Op.forward: builds `IndexedExpertCall`, runs `route_stats` (grouped only) → `expert_gate_up` → `expert_down` → `weighted_reduce`; `out=` semantics preserved | branch predicate `grouped_dispatch = num_tokens > 1` preserved |
| `roofline_data_terms()` via `routed_active_experts(last_call)` (lazy metadata read) | `roofline_data_terms()` reading stored `_last_topk_ids` lazily (`.to("cpu")` sync at call time, not in forward) | identical terms: `{"active_experts": n}` |
| `roof_key()` via `tensor_core_roof` (`tileops._perf.profile`) | not ported — GPU roof-oracle machinery; A5 reporting uses `eval_roofline()` only | documented exclusion |
| Manifest `forall/params/inputs/outputs/shape_rules` (symbolic, `values`/`requires` generators) | A5 concrete schema: dtype unions, workload dim rows, `shape_rules` translated (incl. `topk_ids` in-range rule, restated as caller contract — no per-forward device sync, same as GPU) | adapter §S1/S2 |
| `routed_expert_mlp_roofline` (`tileops._perf.formulas`) | `eval_roofline()`: `flops = T*K*(6*F*H + 6*F + 2*H) + (T*H if scale != 1)`; `bytes = active*3*F*H*elem + 2*T*H*elem + T*K*8` | silu_and_mul = 6 flops/elem; per-expert weights (2F·H + H·F)/E = 3F·H |

## 2. Manifest / workload expansion comparison

| GPU workload row (moe.yaml) | A5 row (spec/moe.yaml) | Expansion |
| --- | --- | --- |
| `{T: 1, E: 256, K: 8, H: 7168, F: 2048, dtype_cases: [{D: float16}, {D: bfloat16}], label: "ds-v3-t1"}` | `{T: 1, E: 256, K: 8, H: 7168, F: 2048, dtypes: [float16, bfloat16], label: "ds-v3-t1"}` | 2 calls (fp16, bf16) |
| `{T: 32, E: 256, K: 8, H: 7168, F: 2048, ... "ds-v3-t32"}` | same | 2 calls |
| `{T: 64, E: 256, K: 8, H: 7168, F: 2048, ... "ds-v3-t64"}` | same | 2 calls |
| `{T: 32, E: 128, K: 8, H: 4096, F: 1536, ... "qwen3-235b-t32"}` | same | 2 calls |

Call inventory: GPU 4 rows × 2 dtype_cases = 8; A5 4 rows × 2 dtypes = 8. No extra
smoke rows added; labels preserved. `routed_scaling_factor` stays at its default 1.0
on every row (as on GPU). Input count 5 (signature order:
`hidden_states, w_gate_up, w_down, topk_weights, topk_ids`) preserved.

Workload input-generation semantics (GPU `CallWorkload.materialize` +
`MoEExpertsWorkload.gen_inputs` → A5 `IndexedExpertMLPWorkload.gen_inputs`):
hidden ×0.1, weights ×0.02, `topk_weights = softmax(randn(T,K))`, `topk_ids` = K
distinct experts in `[0, E)` per row (GPU manifest `topk_ids(T,K,E)` generator). The
GPU `_scale_routed` step is a no-op under this op's contract (fixed `silu_and_mul`,
no `b_gate_up`/`b_down` in the signature) and is not carried over.

## 3. Kernel / Op call mapping (source params ↔ target params)

Op.forward: `(hidden_states, w_gate_up, w_down, topk_weights, topk_ids, *, out=None)`
→ output `[T, H]` (dtype `same_as(hidden_states)`). Route flow:

1. `T > 1` (grouped): alloc `metadata[E + T*K + E*T] int32`, run `route_stats(topk_ids, metadata)`.
   `T == 1`: `metadata=None` (direct dispatch).
2. `hidden_rows[T,K,F] = expert_gate_up(hidden_states, w_gate_up, topk_ids, metadata)` (silu fused).
3. `route_output[T,K,H] = expert_down(hidden_rows, w_down, topk_ids, metadata)`.
4. `output[T,H] = weighted_reduce(route_output, topk_weights)` × `routed_scaling_factor`;
   written into `out` when provided and returned.

Each extracted factory's caller-facing contract (unchanged by this scaffold; the
conductor must keep signatures minus `threads`):

- `_route_stats(num_tokens, top_k, num_experts)` → program `(expert_ids[T,K] i32, metadata[size] i32)`; `size = E + T*K + E*T`.
- `_indexed_expert_gemm(num_tokens, top_k, num_experts, n, k, dtype, activation, route_input, dispatch_mode)` → program `(a, weights, expert_ids, route_metadata, out)`;
  gate/up: `a=[T,H]`, `weights=[E,2F,H]`, `out=[T,K,F]`, `activation="silu_and_mul"`;
  down: `a=[T,K,F]` (route_input), `weights=[E,H,F]`, `out=[T,K,H]`, `activation="none"`;
  `dispatch_mode="grouped"` iff `T > 1`.
- `_weighted_reduce(num_tokens, top_k, hidden_size, dtype, scaling)` → program `(expert_output[T,K,H], topk_weights[T,K] fp32, output[T,H])`.

## 4. Test case inventory (source → target)

Source files: `tests/ops/test_fused_moe_experts.py` (primary),
`tests/test_roofline_oracle.py`, `tests/ops/test_moe_compile.py`.

| GPU case | A5 case (`tests/ops/test_indexed_expert_mlp.py`) | Disposition |
| --- | --- | --- |
| `test_abc_imports`, `test_weighted_reduce_noop*`, `TestMoEPrepareAndFinalizeNoDPEP`, `TestFusedMoEExpertsFwdOp.*` (staged-op cases), `TestFusedMoEActivationInjection.*`, `test_the_shared_expert_refuses_*` | — | excluded: other ops / shared ABCs not migrated |
| `test_small_route_branch_matches_reference` (staged op dispatches indexed branch; ids: dispersed / reused / duplicate-fallback / second-route-group × bf16/fp16) | `test_small_route_matches_reference` (same route matrix × dtypes, direct `IndexedExpertMLPFwdOp` calls) | ported with documented substitution — the dispatch predicate belongs to the unmigrated staged op; kernel-side coverage preserved |
| `test_the_indexed_op_scales_its_output` | same name | ported verbatim |
| — | `test_single_token_direct_dispatch` (T=1 direct path, both dtypes) | added: grouped-vs-direct branch of the migrated op (GPU covered it only through the staged op) |
| `test_dims_off_the_tile_grid` (staged op, F=96) | `test_dims_off_the_tile_grid` (indexed op, H=100/F=96, both dtypes) | ported; boundary/tail coverage |
| — | `test_manifest_shaped_call_matches_reference` (workload-generated grouped call) | added: exercises the A5 workload class + grouped pipeline |
| — | `test_rejects_unsupported_dtype` (fp32 → ValueError) | added: manifest dtype contract (fp16\|bf16) |
| — | `test_writes_into_out_buffer` (`out=` aliasing semantics) | added: `out` contract of the migrated op |
| `test_a_call_prices_the_experts_its_routing_reads` (staged op bytes formula) + `test_roofline_oracle.py` `test_routed_expert_mlp_counts_active_experts_and_the_routing` | `test_roofline_prices_the_experts_its_routing_reads` (indexed op; same expected-bytes formula `active*3*F*H*elem + 2*T*H*elem + T*K*8`, `roofline_data_terms`) | ported/adapted |
| `test_moe_compile.py` `test_*` using `IndexedExpertMLPFwdOp` (torch.compile fullgraph, graph-node ownership) | — | excluded: CUDA compile-boundary infrastructure (A5 removes compile_boundary; O5) |
| `test_small_route_dispatch_replays_in_cuda_graph` (`@pytest.mark.sm90`, CUDA graphs) | — | excluded: CUDA-graph capture is CUDA-specific |

Numerical policy preserved: `moe_verification(2)` = elementwise atol=rtol=1e-2 +
normalized squared error ≤ 1e-4 (fp64 chunked), ported as `_moe_compare` +
`assert_normalized_error`. GPU negative controls (`gate_up_swapped`) are
verification-harness machinery (workloads.numerics) not ported to the A5 test
framework; the elementwise + normalized-error comparison is preserved.

## 5. Benchmark case inventory (source → target)

Source files: `benchmarks/ops/bench_fused_moe_experts.py`
(`test_indexed_expert_mlp_bench`), `benchmarks/_cases/moe.py`
(`IndexedExpertMLPFwdOp` Entry: `IndexedExpertMLPWorkload`, `count_copies=True`,
`binder=_private_output`).

| GPU implementation | A5 implementation | Disposition |
| --- | --- | --- |
| op itself (`bench.Runner` primary, private `out` buffer, copy counting) | `bm.profile(op, *inputs)`, `tag="tileops"` | ported (copy counting/out-binder is GPU benchmarks.api machinery; A5 ManifestBenchmark measures the op call directly) |
| `staged` comparator (pre-permute → expert MLP → post-permute) | — | excluded: staged ops not migrated to A5 |
| `vllm-triton` baseline | — | excluded: CUDA-only |
| `flashinfer-cutlass` baseline (skip-reason gated) | — | excluded: CUDA-only |
| — | `tag="torch"`: `_torch_routed_experts` (gathered bmm + silu + fp32 weighted `index_add_`) | added: native-torch performance baseline runnable on A5 (not the fp32 accuracy golden, which stays in the workload) |

Parametrization: GPU `bench.cases(IndexedExpertMLPFwdOp)` (manifest rows × dtype
cases) → A5 `_shape_dtype_params("IndexedExpertMLPFwdOp")` (multi-input custom
helper; `workloads_to_params` raises KeyError for 5-input ops). Same 8 cases
(4 rows × 2 dtypes), same labels.

## 6. Branch coverage / exclusion rationale

Kernel-level branches (all preserved, to be served by the Stage 1-3
re-implementation):

- `dispatch_mode`: `"direct"` (T == 1) vs `"grouped"` (T > 1) — both in tests
  (`test_single_token_direct_dispatch`, all grouped cases); build identity keeps
  `grouped_dispatch`.
- `route_input`: False (gate/up) / True (down) — both specializations ported.
- `activation`: `"silu_and_mul"` (gate/up) / `"none"` (down) — both; the
  `IndexedExpertGEMMTemplate` constructor still validates the pair
  (`none`/`silu_and_mul`) as on GPU.
- Leader/duplicate handling: routes with `rank >= num_tokens` (unlisted duplicates)
  computed independently (`duplicate-fallback` case); leader grouping exercised by
  `reused` / `second-route-group`.
- Workspace-budget fit loop (`num_stages`/`block_k` reduction) — ported against the
  A5 UB budget (K6), branch structure unchanged.

Excluded from migration (with proof no retained call selects them):
`MoEPermuteAlignFwdOp`, `MoEPrePermuteFwdOp`, `MoEExpertMLPFwdOp`,
`MoEPostPermuteFwdOp`, `FusedMoEExpertsFwdOp`, `FusedMoEFwdOp`,
`FusedMoESharedExpertFwdOp`, `SharedExpertMLPFwdOp`, `FusedTopKFwdOp`,
`MoEGroupedGEMM(FP8)FwdOp` — none is referenced by `IndexedExpertMLPFwdOp` or its
four kernels (GPU op imports only the four indexed kernel classes and
`IndexedExpertCall`); the staged-pipeline comparator appears only in the GPU bench
file, whose A5 port replaces it with the native-torch baseline.

## 7. Extracted-function provenance and helper completeness

- `_route_stats` → `src/tileops/_kernels/moe/indexed_expert_gemm.py` (GPU repo).
- `_indexed_expert_gemm` → same file.
- `_weighted_reduce` → same file.
- `_route_metadata_size` (helper, called inside the `_route_stats` and
  `_indexed_expert_gemm` factory closures) → same file; **manually appended** to the
  extracted file (extractor did not auto-include it), verbatim, GPU import order kept
  top-down. Not an extracted kernel function, so it is absent from
  `extracted_functions`/`function_sources` by contract.
- Helper scan result: every non-builtin, non-`T.*`/`tilelang.*`/`torch.*` callable
  referenced by the three extracted factories is defined in the extracted file.
- Extraction warnings: none reported by the extractor; no unresolved internal
  package dependency, no same-name collisions (single source file for all three
  factories).
