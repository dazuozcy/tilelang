# TileLang Ascend (A5) Agent Guide

This repository (tilelang-fork, Ascend A5/950, `--npu-arch=dav-3510`) uses AGENTS skills for TileLang Ascend operator development. The authoritative operator-programming reference is `.agents/skills/tilelang-ascend/SKILL.md` (+ its `references/`).

## Scope

The conductor-related skills under `.agents/skills/` are designed for `target="ascend"` workflows (canonical entry: `@tilelang.jit(target="ascend")`). They prioritize the Ascend dialect `import tilelang.ascend.language as T` with SimtVF/SimdVF/MixedKernel programming modes and the `T.copy`/`T.gemm`/`T.dual_copy` Cube path. The `pto` AOT target line is **out of conductor scope** (recorded in `.agents/evolution/capability-gaps.md` if needed).

## API Convention (Mandatory)

- New operator code uses `import tilelang.ascend.language as T` (the Ascend dialect shadows the common surface).
- Default programming mode is **SimtVF** (`T.SimtVF(threads=N)` + `T.Parallel` loops; compiler vectorizes and auto-schedules). Downgrade to **SimdVF** (`T.SimdVF()` + explicit `T.simd.*` 2048-bit intrinsics, optionally `latency=`) only for proven hot spots. GEMM-dominant operators use **mixed** kernels (`T.MixedKernel(sids=1|2)`).
- Cube path: `T.copy` GM↔L1 (ND2NZ fractal conversion is inserted automatically by the InsertNd2Nz pass — never hand-write NZ staging) → `T.gemm(A_l1, B_l1, C_l0c, transpose_B=True)` (L1-input path requires NT: `trans_A=False, trans_B=True`; weights in GM are `[N, K]`) → `T.dual_copy`/`T.copy` L0C→UB. Accumulators: `T.alloc_l0c` + `clear_accum=(kt == 0)`.
- Memory hierarchy: `T.alloc_shared` = UB (248KB physical; 216KB with SimtVF), `T.alloc_l1` = CBuf (512KB), `T.alloc_l0a/l0b/l0c` = Cube buffers. See `pattern-library/constants.md` (CONST-capacity-a5).
- Kernel launch is 1-D only: `T.Kernel(N)`; no `threads=` on `T.Kernel` (TypeError).
- Synchronization: rely on AutoSchedule by default; manual `T.ascend_set_flag`/`T.ascend_wait_flag`/`T.ascend_pipe_barrier` only with matched pairs and comments (fully-manual kernels need `PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False`).
- Precision validation: golden computed with torch on NPU (`device="npu"`), `torch.testing.assert_close`, tolerance by dtype: fp8→1e-1, bf16/fp16→1e-2, fp32→1e-3 (5e-3 for accumulation-heavy).
- Performance measurement: `tilelang.profiler.do_bench(fn, backend="msprof")` is the sole end-to-end latency authority; `msprof_detail` and npusim serve cycle-level attribution. Never invent clock frequencies — the cost model is cycle-domain only.

## Skill Index

Existing repository skills (unchanged):

1. tilelang-ascend — authoritative Ascend NPU programming skill (mental model, thread hierarchy, memory, GEMM, data movement, sync, SIMD MicroAPI, pitfalls, perf measurement).
2. tilelang-backend — target/backend integration architecture (four layers).
3. tilelang-build — repository build/install/test instructions.
4. tilelang-dev-guide — C++ development conventions and regression planning.
5. tilelang-layout — Layout/Fragment classes, layout inference, CuTe algebra.
6. tilelang-semantic — TileLang semantic rules and validation.
7. tilelang-simplification — simplification audits of compiler/runtime surface.
8. tilelang-submit-pr — PR readiness checks for this repository.
9. tilelang-tvm-ir — TVM TIRX ObjectRef/NodeRef editing conventions.

Conductor skills (ported from tilelang-mlir-fork, adapted to ascend):

10. tilelang-op-design — Stage 1 operator design (DESIGN.md): algorithm research (four questions), programming-mode selection (simt/simd/mixed), algorithm-level optimization, API mapping, memory planning, tiling, sync strategy, verification plan.
11. tilelang-op-develop — Stage 3 implementation (kernel + golden + layered tests), four-exit verdict.
12. tilelang-op-optimize — Stage 4 performance tuning, do_bench(msprof)/npusim evidence, perf_records.jsonl.
13. tilelang-design-review — Stage 2 risk-first DESIGN.md review producing REVIEW.md.
14. tilelang-ascend-opdev — operator-family navigation (elementwise/gemm/attention/reduction): mode selection → reference example → layout/movement idioms; delegates to tilelang-ascend skill sections.
15. tilelang-ascend-debug — Ascend debugging: get_kernel_source (AscendC source), IR Lower Trace, Pass Diff, T.print/device_assert, npusim cycle localization.
16. tilelang-ascend-error-fixer — error diagnosis and repair: dialect frontend (tilelang-semantic) / Ascend pass assertions / bisheng compile errors / ACL runtime & precision.
17. tilelang-review-skill — risk-first code review and format checks (`bash format.sh --files`).
18. tilelang-github-operations — upstream (tile-ai/tilelang) commit/PR/issue workflow.
19. tilelang-skill-evolution — self-evolution distillation for tilelang-op-conductor (unchanged mechanism).

## Trigger Guidance

Use the matching skill whenever the user asks for:

- Ascend kernel writing, performance tuning, SimtVF/SimdVF, cube gemm, MixedKernel
- operator design / implementation / review / tuning under the conductor pipeline (`.opencode/agents/tilelang-op-conductor.md`)
- Ascend pass debugging, AscendC source inspection, lowering trace, cycle-level profiling
- compile/runtime error analysis on the ascend target (bisheng / ACL runtime)
- code review, lint/format checks, PR readiness
- commit/push/rebase/upstream sync, PR or issue workflow
- reviewing or merging evolution proposals (`.agents/evolution/queue.md`), value-point distillation

Mixed-kernel trigger rule: if one kernel contains Cube-side `T.gemm` and Vector-side work (SimtVF/SimdVF region), treat it as a mixed kernel and consult tilelang-ascend skill §3 + tilelang-ascend-opdev.

## Operator Implementation Baseline (Mandatory)

For operator-writing tasks, always start from existing examples and tests:

- First consult `examples/ascend/` (application examples: vecadd, rmsnorm, gemm, gemm_splitk, per_token_cast_to_fp8, deepgemm, flash_attention) and `testing/ascend/` (feature regression) for the closest existing pattern.
- Prefer modifying an existing operator case instead of generating a brand-new kernel from scratch.
- If no close template exists, explicitly state that and then build the minimal new kernel.
- Do not import application kernels into `testing/ascend/` regression suites.

## Compiler Capability Gap Rule (Mandatory)

When any task stage (design algorithm research / API mapping, develop, optimize, review) concludes that current ascend-backend capability is insufficient to implement the chosen algorithm (cannot implement / forced algorithm swap / performance ceiling):

1. Identify the specific missing capability and its layer (Frontend API (ascend dialect) / Ascend pass / codegen / bisheng compiler / ACL runtime).
2. Record it in `.agents/evolution/capability-gaps.md`: dedupe first, then append a new entry or increment `occurrences` of the existing one, following the schema in that file.
3. Evidence rule is identical to negative claims in Docs Auto Routing: cite concrete doc path and limiting clause, or explicitly mark as undocumented assumption with estimation basis.
4. Escalation: a gap independently identified in 2 or more different tasks becomes `recurring` and must produce a capability-fill proposal (target layer, proposed change, quantified benefit, affected operators) surfaced to the user at task end. Filing an upstream issue requires user approval.
5. Gap conclusions bind to toolchain version stamps; re-verify before reusing them after toolchain upgrades (mechanized: `.agents/tools/kb_stale_check.py` detects stale entries; `.agents/tools/repro_runner.py --filter stale` re-verifies repro-backed ones).

## Pre-PR Formatting Rule (Mandatory)

Before creating or updating a PR, run format validation for changed files from repository root:

- bash format.sh --files changed_files

Notes:

- This is a required self-check for clean code and style consistency.
- The changed_files placeholder represents the file list modified in the current branch.

## Docs Auto Routing Rules (Mandatory)

When any skill answers technical questions, it must route references by docs directory first.

### Routing Priority

1. `docs/programming_guides/` (language basics, instructions, control flow, software pipeline, autotuning)
2. `tilelang/ascend/README.md` + `.agents/skills/tilelang-ascend/` (SKILL.md + references: API semantics, constraints, sync, profiling — the authority for Ascend dialect questions)
3. `docs/tools/` (lower_trace, pass_diff, compile_only, analyzer, autodd, layout_visualization)
4. `docs/get_started/targets.md` (target/runtime/env)
5. `docs/developer_guide/` + `docs/compiler_internals/` + `docs/runtime_internals/` (contribution, testing style, internals)

### Keyword to Docs Mapping

- Language basics, types, control flow: `docs/programming_guides/language_basics.md`, `type_system.md`, `control_flow.md`
- Pipeline/stage/persistent: `docs/programming_guides/software_pipeline.md`, `.agents/skills/tilelang-ascend/references/synchronization.md`
- Autotuning: `docs/programming_guides/autotuning.md`
- Ascend dialect API (SimtVF/SimdVF/MixedKernel/T.copy/T.dual_copy/T.gemm/T.simd.*/alloc_*): `.agents/skills/tilelang-ascend/SKILL.md` §2–§8; source of truth for exact signatures: `tilelang/ascend/language/` (e.g. `simd.py`, `copy_op.py`, `kernel.py`)
- Memory hierarchy / capacity: tilelang-ascend skill §4; `src/ascend/transform/auto_schedule.cc` (capacity limits)
- GEMM constraints: tilelang-ascend skill §5; `tilelang/ascend/op/gemm/gemm_mad.py`
- Synchronization (flags/barriers/Stage/AutoSchedule): tilelang-ascend skill §7 + `references/synchronization.md`
- Profiling, do_bench, msprof, npusim, latency annotation: tilelang-ascend skill §12 + `references/profiling.md`
- Debug, compile failure, lowering trace, pass diff, AscendC source: `docs/tools/lower_trace.md`, `docs/tools/pass_diff.md`, `docs/tools/compile_only.md`, tilelang-ascend skill §11
- Runtime target, env, execution: `docs/get_started/targets.md`, `tilelang/ascend/README.md`
- PR, rebase, commit, issue, CI workflow: `docs/developer_guide/` + CONTRIBUTING.md

### Conflict Resolution

- If multiple mappings match, select by priority and keep at most 3 primary doc references.
- For API-related questions, always include at least 1 reference under `.agents/skills/tilelang-ascend/` or `tilelang/ascend/` source.
- If docs and source differ, `tilelang/ascend/` source is the implementation truth; docs issues should be recorded.
- If a question involves an API not covered by any mapping above, enumerate `ls tilelang/ascend/language/` and check the modules (including `simd.py`, `copy_op.py`, `gemm_op.py`, `mode.py`, `schedule_hint.py`, `tile_schedule.py`) before concluding the API is unsupported or undocumented.
- Negative claims (API does not exist / not supported / too costly / no precedent) must cite the concrete doc path and the limiting clause read from it, or be explicitly marked as "undocumented assumption" with the estimation basis. Never reject a design candidate from memory or GPU prior before the retrieval.

## Backend Glossary (Single Source of Truth)

Vocabulary mapping between the legacy npuir stack (tilelang-mlir-fork) and this repository is maintained in `.agents/skills/_shared/standards/backend-glossary.md`. All conductor agents/skills reference it instead of inlining mappings. Forbidden legacy terms (npuir, v-prefix APIs, load_nd2nz, store_fixpipe, TileLangIR, BishengIR, tladapter, 910B2C, NPUUtils, set_env.sh, ASCEND_MODE) must not appear in `.opencode/` + `.agents/` rule files outside the glossary and explicit historical notes.
