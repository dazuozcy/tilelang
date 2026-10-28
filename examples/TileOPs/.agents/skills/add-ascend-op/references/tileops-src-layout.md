# Current TileOPs source → local Ascend harness

This adapter applies to Stage 0 and Stage 5. The cross-stage field contract is
`.agents/skills/_shared/standards/migration-source.md` in the outer repository. The source is
`/home/zuo/code/TileOPs` (caller overrides are allowed); the destination is
`examples/TileOPs/`. Only this source layout is supported. Both sides use src packages and root workloads.
Read the source `docs/design/manifest.md`, `docs/design/ops-design.md` and
`docs/design/testing.md`, then inspect the actual implementation. Never import
the GPU package to discover paths or evaluate YAML expressions with `eval`.

## Phase 0: discover, then record

| Artifact | Current GPU source | A5 destination |
| --- | --- | --- |
| Manifest | `src/tileops/manifest/spec/<family>*.yaml` (exclude `types.yaml`) | `src/tileops/manifest/spec/<family>.yaml` |
| Op | `src/tileops/_ops/**/*.py`; public re-exports in `src/tileops/<family>.py` | `src/tileops/_ops/<family>/<op_slug>.py` |
| Kernel | `src/tileops/_kernels/**/*.py`; follow Op `kernel_types` and imports, including base classes | `src/tileops/_kernels/<family>/<op_slug>/` |
| Workload | `workloads/**/*.py`, including package families | `workloads/<family>.py` |
| Correctness | `tests/ops/**/*.py` and imported fixtures/case helpers | `tests/ops/test_<test_slug>.py` |
| Benchmark | `benchmarks/ops/**/*.py`, often `benchmarks.api` / `bench.cases(Op)` | `benchmarks/ops/bench_<bench_slug>.py` using local harness API |

Run from the A5 root with Python >= 3.10:

```bash
python .agents/skills/add-ascend-op/scripts/extract_tl_kernel.py \
  --gpu-repo-root /home/zuo/code/TileOPs --op-name {op_name} --inspect
```

`--inspect` emits JSON with the entry and resolved `kernel_bindings`, without
writing files or executing source code. Its synthesized `entry.source.op`
and `entry.source.kernel` point to the Op discovery root, not a manifest
field or a claim that all kernel classes live in that file. The bindings
contain actual kernel files/classes. Old directory layouts and metadata fallback are not supported. Ambiguous Op definitions, dynamic `kernel_types`, or unresolved
classes fail explicitly; inspect and extend discovery rather than guessing.
`spec-only` alone does not prove there is no implementation: inspect the code
and verify reachable TileLang builders before rejecting migration.

Locate tests/benchmarks by imports and actual Op calls, not matching filenames.
Follow aliases, shared helpers, inherited fixtures and package workloads.
Record every relevant file/node id; if multiple source files serve this Op,
port all relevant cases into target entries or consolidate them without dropping
coverage. Select all reachable kernel implementations by default; any exclusion
needs an explicit scope decision and proof that retained calls cannot select it.

Record source provenance in `.migration_meta.json.gpu_source` (required for the
current source layout; old metadata must be regenerated):

```json
{
  "manifest_path": "src/tileops/manifest/spec/reduction.yaml",
  "op_path": "src/tileops/_ops/reduction/softmax.py",
  "function_sources": {"example_builder": "src/tileops/_kernels/reduction/logsumexp.py"},
  "kernel_bindings": [{"path": "src/tileops/_kernels/reduction/logsumexp.py", "class": "LogSumExpKernel"}],
  "workload_paths": ["workloads/reduction.py"],
  "test_paths": ["tests/ops/test_softmax.py"],
  "bench_paths": ["benchmarks/ops/bench_softmax.py"],
  "revision": "<source HEAD>"
}
```

The example binding/function lists are illustrative; record the complete actual lists.
Run normal extraction with `--source-info-out <discovery.json>` to generate
`source_layout`, GPU root, original per-function paths and revision/dirty state.
Complete its workload/test/bench lists after reading the actual cases; add a
`conversion_path` pointing to the A5 conversion record. It is a discovery sidecar,
not a complete Stage 0 meta; merge only reviewed fields into that meta.
Existing top-level meta paths always name A5 files. Keep required
`kernel_class_name` as an actual exported A5 wrapper class for the integrator's
smoke assertion; for multiple classes add optional `kernel_class_names` with
all target exports and validate every one during Tier 1. Never join class names
into the singular string. `kernel_bindings` describes source classes and need
not have the same names as these target exports. Per-function prompts must
identify the extracted reference and original source file(s), plus the A5
spec/test/bench paths. Keep an interface/schema conversion table alongside the
extracted reference; pass it to design and integration.

## S1/S2: translate the contract and calls

The new manifest deliberately has no `source` section. Generate A5 `source.op`,
`source.kernel`, `source.kernel_map`, `source.test`, `source.bench` and
`bench_manifest_driven` from the files actually created, not from GPU paths.
Derive `family` from the entry and reject conflicting caller input.

New `signature.forall`, symbolic tensor shapes, `let`, ADTs/types, tensor
`requires`/effects and `dtype_cases` are not the legacy A5 schema. Bind each
workload row's indices and constructor params, expand all `dtype_cases`, and
translate into A5 concrete `<tensor>_shape`, `dtypes` and op params. Preserve
labels, input count, dtype pairings, defaults, optional inputs, axis/keepdim,
mutation/alias behavior and shape constraints. Do not replace a missing required
constructor argument with a guessed default. Compare the expanded source and
target call inventories; extra smoke cases must be labeled separately.

Port the actual workload Case/input-generation logic and its helper dependencies
to the local workload API; changing `cuda` to `npu` alone is insufficient.
Implement constraints/shape checks the new source generates from its manifest
explicitly in the A5 Op when its framework cannot generate them. A contract
that cannot be represented must be reported as a scaffold failure, not silently
weakened. For roofline, bind symbolic indices and translate formulas to the
local evaluator's vocabulary. An absent byte formula must be derived from the
contract's tensor traffic, documented and checked; never invent a number.

## S3/S4: adapt dispatch, not just import spelling

The extraction script resolves `kernel_types` through base classes and imports,
then gathers builders and dependencies across all bound kernel files. It uses
`src/` for absolute `tileops.*` imports and preserves relative import resolution.
Run its normal `--out` command after inspect. Inspect output warnings and AST:
there must be no unresolved internal package dependency or missing helper.
Conflicting same-name definitions from different files need explicit renaming
with all references updated; never accept an extraction with collisions.

Current source uses `kernel_types`, `interfaces`, `kernel_for`, call-spec objects,
`CompiledProgram` and target selection. These APIs differ from local
`default_kernel_map`, Kernel construction/config and forward dispatch. Read
`src/tileops/_ops/op_base.py`, `src/tileops/_kernels/kernel_base.py` and a local adapted
case before constructing the wrapper. Preserve source branch predicates,
layout/reshape and semantic behavior while mapping to the local execution API.
Do not copy GPU registry decorators, interface registration, device capability
probes, CUDA stream management or `tileops._*` imports into the A5 wrapper.
The wrapper must call extracted functions through the import form consumed by
`integrate_kernel.py`; keep every Stage 1-3 function in `extracted_functions`.

## S5/S6: adapt tests and benchmarks

Port only the selected Op's cases and their transitive helpers, including cases
that use public `tileops.<family>` imports. Convert `workloads.device.run_device`,
`run_device_available`, numeric comparison helpers and execution-policy arguments
to local equivalents while preserving assertions and coverage. Do not carry
unrelated Ops or blanket GPU skip markers into A5 tests.

Current benchmarks use `bench.cases(Op)`, `bench.Case`, `bench.Runner`, and
optional CUDA-only baselines (flag_gems/quack/torch compile). Translate their
complete workload-call inventory to the local benchmark API consumed by
`tileops.reporting.cli`. Preserve the operator call and torch reference;
GPU-only comparison implementations are not required on Ascend. Do not copy
`benchmarks.api` imports into a target that does not provide that API, and do
not claim source performance numbers as A5 measurements.

## Stage 5: consume target metadata only

Keep integration output and baseline/perf_opt switches under the target
`src/tileops/_kernels/` layout. The report CLI still uses the `tileops` namespace. Check wrapper signatures/config forwarding against the
conversion table and final Stage 3 artifacts. `source.test`/`source.bench` in
the **A5** manifest must match meta `test_path`/`bench_path`; the new GPU
manifest has no such fields. No source repo install or GPU benchmark is needed.

On a machine without NPU, static integration may use `--skip-smoke`; do not run
package import smoke, pytest collection, report or profiling. Record runtime as
unverified and do not emit `INTEGRATE_COMPLETED`. On NPU hardware, the existing
full single-Op report and gate 5 remain mandatory; static checks do not satisfy
them. Source-schema or test-porting errors belong to Stage 0, not Stage 5's
restricted kernel/glue repair scope.
