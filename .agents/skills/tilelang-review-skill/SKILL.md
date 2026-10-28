---
name: tilelang-review-skill
description: TileLang Ascend 代码审查与格式校验技能。用户提及 review、代码审查、PR 前检查、lint、format、ruff、clang-format、规范检查、CI 不通过时必须使用本技能。优先识别行为回归、数值风险、同步风险与测试缺口，其次才是风格问题。
---

# TileLang Review Skill (Ascend)

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## Scope

- pre-PR code review for the Ascend backend (tilelang-fork, A5/950, dav-3510)
- format and lint checks aligned with CI
- risk-focused review for correctness, precision, synchronization, and capacity

## Review priorities

1. Behavior regressions
2. Precision and dtype risks
3. Synchronization and pipeline hazards (AutoSchedule assumptions, manual flag pairing)
4. Missing tests
5. Style and format consistency

## Docs to consult first

- `.agents/skills/tilelang-ascend/SKILL.md` (§2 thread hierarchy / §4 memory /
  §5 GEMM / §6 data movement / §7 synchronization / §10 pitfalls)
- `docs/programming_guides/` (language basics, control flow, software pipeline)
- `docs/developer_guide/` + `CONTRIBUTING.md` (contribution and testing style)
- `docs/get_started/targets.md` (target/runtime/env)

Ascend-specific checklist items (SimtVF/SimdVF mode discipline, NT GEMM
convention, 1-D grid, alloc API mapping, capacity budget with buffer versions
and layout padding, NPU-side golden with dtype-tiered tolerance) live in
[references/checklist.txt](references/checklist.txt).

## Documentation drift check (agent/skill markdown changes)

When the change set touches `.opencode/agents/`, `.agents/skills/` (including
`_shared/standards/`), or the conductor orchestration docs, run the shared
standards drift checker before review conclusion:

```bash
python3 .agents/tools/standards_check.py check
```

- Exit 0: no drift. Exit 1: inspect the single-line JSON `failures[]`:
  - `SC-INLINE-DUP` — a consumer inlined canonical rule text instead of
    referencing `.agents/skills/_shared/standards/*.md` (reject: single
    source of truth violated);
  - `SC-HASH-MISMATCH` / `SC-UNTRACKED` — a standard file was edited
    without regenerating the lock (require the author to run
    `standards_check.py update`, and to confirm consumers of the changed
    standard were synchronized);
  - `SC-DANGLING-REF` / `SC-REF-PATH` / `SC-FP-STALE` — broken references
    (reject).
- Review focus for standards edits: the mechanical gate rules in
  `.agents/tools/gate_lint.py` must be updated in the same change when the
  edited standard carries a mechanically checkable subset.

## Knowledge base lint (pattern-library / evolution queue changes)

When the change set touches the knowledge base -- `.agents/skills/
tilelang-op-optimize/references/pattern-library/` (topic files, INDEX.md,
`repro/`), `bottleneck-patterns.md`, `algorithm-candidates.md`, or
`.agents/evolution/queue.md` -- run the KB lint (K-4, same level as
standards_check; used by optimizer write-back self-check, evolver
pre-merge check, and this review path):

```bash
python3 .agents/tools/kb_lint.py
```

- Exit 0: clean (warnings allowed). Exit 1: inspect `KB-*` failures:
  - `KB-FM-REQUIRED` / `KB-FM-ENUM` — entry missing front-matter fields or
    illegal kind/status enum (K-1 schema; reject);
  - `KB-REPRO-*` — repro registered but missing / header lacks entry id or
    version stamp / syntax error / code references task-workspace paths
    (ED-F decoupling; reject);
  - `KB-QUEUE-SCHEMA` — malformed proposal fields (reject).
- Warnings (`KB-BUDGET` byte budgets, `KB-IMPERATIVE` heuristic,
  `KB-INDEX-COVERED`, `KB-REPRO-ORPHAN`) do not block review but should be
  queued for the next distill consolidate.
- Repro suite execution is on-demand (real NPU probes):
  `python3 .agents/tools/repro_runner.py` after toolchain changes.

## References

- references/checklist.txt

## Related skills

- tilelang-ascend-error-fixer
- tilelang-ascend-debug
