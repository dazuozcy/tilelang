#!/usr/bin/env python3
"""Verify the consolidated artifact's kernel bodies are byte-identical to the
per-function drop-in files (Stage 4 gate self-check).

Kernel code body (verbatim set): both kernel builders + their host-staticize
helpers (_load_group/_pick_rows/_env_int/_wrapper_tile_n) + shared golden /
source-sim / tolerance / dtype helpers + call helpers.  Test scaffolding is
rename-only (sibling template) and checked as such.
"""

import ast
import sys

PERF_OPT = (
    "/home/z00875089/tilelang-fork/examples/TileOPs/tileops/kernels/"
    "reduction/logsumexp_simd/logsumexp_simd_kernel/perf_opt"
)
SINGLE = PERF_OPT + "/_logsumexp_kernel_single.py"
TILED = PERF_OPT + "/_logsumexp_kernel_tiled.py"
CONS = PERF_OPT + "/LogSumExpFwdOpSimd.py"


def func_sources(path):
    src = open(path, encoding="utf-8").read()
    lines = src.splitlines()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = "\n".join(lines[node.lineno - 1 : node.end_lineno])
    return out, src


s_src, s_full = func_sources(SINGLE)
t_src, t_full = func_sources(TILED)
c_src, c_full = func_sources(CONS)

failures = []

# ---- exact (byte-identical) kernel-body functions ----
exact_checks = [
    (SINGLE, "_logsumexp_kernel_single"),
    (SINGLE, "_call"),
    (SINGLE, "golden_logsumexp"),
    (SINGLE, "_source_lse_sim"),
    (SINGLE, "_tolerance"),
    (SINGLE, "_elem_bytes"),
    (SINGLE, "_dtype_str"),
    (TILED, "_logsumexp_kernel_tiled"),
    (TILED, "_load_group"),
    (TILED, "_pick_rows"),
    (TILED, "_env_int"),
    (TILED, "_wrapper_tile_n"),
    (TILED, "golden_logsumexp"),
    (TILED, "_source_lse_tiled_sim"),
    (TILED, "_tolerance"),
    (TILED, "_elem_bytes"),
]
for path, name in exact_checks:
    origin = s_src if path == SINGLE else t_src
    if origin[name] != c_src[name]:
        failures.append(f"BYTE MISMATCH: {name} ({path.split('/')[-1]})")
    else:
        print(f"  OK byte-identical: {name} ({len(origin[name].splitlines())} lines)")

# ---- _call_tiled: tiled _call modulo the sibling rename ----
expect = t_src["_call"].replace("def _call(", "def _call_tiled(", 1)
if expect != c_src["_call_tiled"]:
    failures.append("BYTE MISMATCH: _call_tiled vs tiled _call (rename-only)")
else:
    print("  OK rename-only: _call -> _call_tiled")


# ---- test scaffolding: rename-only equivalence ----
def renamed(origin_text, pairs):
    for old, new in pairs:
        origin_text = origin_text.replace(old, new)
    return origin_text


scaffold_checks = [
    (s_src["_run_case"], c_src["_run_single_case"], [("def _run_case(", "def _run_single_case(")]),
    (
        s_src["_run_special_case"],
        c_src["_run_single_special_case"],
        [("def _run_special_case(", "def _run_single_special_case(")],
    ),
    (
        s_src["run_L0"],
        c_src["_run_single_L0"],
        [("def run_L0(", "def _run_single_L0("), ("_run_case(", "_run_single_case(")],
    ),
    (
        s_src["run_L1"],
        c_src["_run_single_L1"],
        [("def run_L1(", "def _run_single_L1("), ("_run_case(", "_run_single_case(")],
    ),
    (
        s_src["run_L2"],
        c_src["_run_single_L2"],
        [("def run_L2(", "def _run_single_L2("), ("_run_case(", "_run_single_case(")],
    ),
    (
        s_src["run_boundary"],
        c_src["_run_single_boundary"],
        [
            ("def run_boundary(", "def _run_single_boundary("),
            ("_run_case(", "_run_single_case("),
            ("_run_special_case(", "_run_single_special_case("),
        ],
    ),
    (
        s_src["_show_source"],
        c_src["_show_single_source"],
        [("def _show_source(", "def _show_single_source(")],
    ),
    (
        t_src["_run_case"],
        c_src["_run_tiled_case"],
        [("def _run_case(", "def _run_tiled_case("), ("_call(", "_call_tiled(")],
    ),
    (
        t_src["_run_special_case"],
        c_src["_run_tiled_special_case"],
        [("def _run_special_case(", "def _run_tiled_special_case("), ("_call(", "_call_tiled(")],
    ),
    (
        t_src["run_L0"],
        c_src["_run_tiled_L0"],
        [("def run_L0(", "def _run_tiled_L0("), ("_run_case(", "_run_tiled_case(")],
    ),
    (
        t_src["run_L1"],
        c_src["_run_tiled_L1"],
        [("def run_L1(", "def _run_tiled_L1("), ("_run_case(", "_run_tiled_case(")],
    ),
    (
        t_src["run_L2"],
        c_src["_run_tiled_L2"],
        [("def run_L2(", "def _run_tiled_L2("), ("_run_case(", "_run_tiled_case(")],
    ),
    (
        t_src["run_boundary"],
        c_src["_run_tiled_boundary"],
        [
            ("def run_boundary(", "def _run_tiled_boundary("),
            ("_run_special_case(", "_run_tiled_special_case("),
            ("_call(", "_call_tiled("),
        ],
    ),
    (
        t_src["_show_source"],
        c_src["_show_tiled_source"],
        [("def _show_source(", "def _show_tiled_source(")],
    ),
]
for origin_text, cons_text, pairs in scaffold_checks:
    if renamed(origin_text, pairs) != cons_text:
        failures.append("SCAFFOLD MISMATCH (rename-only expected)")
    else:
        print("  OK rename-only scaffold")

# ---- tuned constants values ----
import re

cons_const = dict(re.findall(r"^(\w+) = (.+)$", c_full, re.M))
for name, expect_val in [
    ("NUM_BLOCKS", "56"),
    ("MAX_SINGLE_TILE_COLS", "32512"),
    ("UB_BUDGET_BYTES", "int(248 * 1024 * 0.9)  # 228556"),
    ("UB_RESERVE_BYTES", "8256"),
    ("WRAPPER_BUDGET_BYTES", "216 * 1024"),
    ("GROUP", "64"),
    ("HMAX_ENABLE", "True"),
]:
    got = cons_const.get(name)
    if got != expect_val:
        failures.append(f"CONST MISMATCH: {name} = {got!r}, expected {expect_val!r}")
    else:
        print(f"  OK constant: {name} = {got}")

print()
if failures:
    print("FAILURES:")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("ALL VERBATIM CHECKS PASSED")
