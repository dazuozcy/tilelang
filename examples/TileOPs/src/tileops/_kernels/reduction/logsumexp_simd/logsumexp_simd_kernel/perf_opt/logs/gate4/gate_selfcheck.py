#!/usr/bin/env python3
"""Gate-4 self-check (READ-ONLY): invoke gate_lint.perf_lint directly with the
state-contract artifact paths.  Does not read/write .stage_state.json."""

import sys

sys.path.insert(0, "/home/z00875089/tilelang-fork/.agents/tools")
import gate_lint  # noqa: E402

REPO = "/home/z00875089/tilelang-fork"
BASE = (
    REPO + "/examples/TileOPs/tileops/kernels/reduction/logsumexp_simd/"
    "logsumexp_simd_kernel/perf_opt"
)

failures, warnings = gate_lint.perf_lint(
    BASE + "/LogSumExpFwdOpSimd.py",  # perf_py_path (S4-EXISTS / kernel_lint)
    BASE + "/opt_log.md",  # opt_log_path
    BASE + "/perf_feedback.md",  # feedback_path (absent -> skipped)
    BASE + "/perf_records.jsonl",  # records_path
    "LogSumExpFwdOpSimd",  # op_name
    REPO,  # repo_root
)

print(f"gate 4 lint: {len(failures)} failures, {len(warnings)} warnings")
for f in failures:
    print("  FAIL", f)
for w in warnings:
    print("  WARN", w)
sys.exit(1 if failures else 0)
