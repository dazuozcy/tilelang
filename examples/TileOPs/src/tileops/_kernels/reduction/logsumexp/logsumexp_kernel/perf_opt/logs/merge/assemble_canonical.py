"""Assemble perf_opt/LogSumExpFwdOp.py (canonical consolidated dual-kernel artifact)
from the two frozen per-kernel final files. AST-guided splice, symbol-renamed suites."""
import ast

BASE = "/home/z00875089/tilelang-fork/examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp_kernel/perf_opt"


def sections(path):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = (node.lineno - 1, node.end_lineno)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = (node.lineno - 1, node.end_lineno)
    return src, lines, out


def grab(lines, sec, name):
    a, b = sec[name]
    return "".join(lines[a:b])


s_src, s_lines, s_sec = sections(f"{BASE}/_logsumexp_kernel_single.py")
t_src, t_lines, t_sec = sections(f"{BASE}/_logsumexp_kernel_tiled.py")

# --- single: constants + helpers + builder + _call ---
single_consts = "".join(
    grab(s_lines, s_sec, n) for n in
    ("NUM_BLOCKS", "MAX_SINGLE_TILE_COLS", "UB_BUDGET_BYTES", "VRF_BUDGET_BYTES",
     "TUNED_SIMTVF_THREADS", "TUNED_ROWS_CANDIDATES")
)
single_helpers = grab(s_lines, s_sec, "_elem_bytes") + grab(s_lines, s_sec, "_pick_rows")
single_builder = grab(s_lines, s_sec, "_logsumexp_kernel_single") + "\n\n" + grab(s_lines, s_sec, "_call")
single_suite = "".join(
    grab(s_lines, s_sec, n) + "\n\n" for n in
    ("_run_case", "_run_special_case", "run_L0", "run_L1", "run_L2", "run_boundary")
)
# rename single suite symbols -> _single_*
renames_s = [
    ("def _run_case(", "def _run_single_case("),
    ("def _run_special_case(", "def _run_single_special_case("),
    ("def run_L0(", "def _run_single_L0("),
    ("def run_L1(", "def _run_single_L1("),
    ("def run_L2(", "def _run_single_L2("),
    ("def run_boundary(", "def _run_single_boundary("),
    ("_run_case(", "_run_single_case("),
    ("_run_special_case(", "_run_single_special_case("),
]
for a, b in renames_s:
    single_suite = single_suite.replace(a, b)
# fix self-calls inside renamed defs (run_L0 body etc. call the helpers, already renamed)

# --- tiled: builder + _call + suite ---
tiled_helpers = grab(t_lines, t_sec, "_pick_block_n") + "\n\n" + grab(t_lines, t_sec, "_pick_threads")
tiled_builder = grab(t_lines, t_sec, "_logsumexp_kernel_tiled")
tiled_call = grab(t_lines, t_sec, "_call").replace("def _call(", "def _call_tiled(")
tiled_suite = "".join(
    grab(t_lines, t_sec, n) + "\n\n" for n in
    ("_run_case", "_run_special_case", "run_L0", "run_L1", "run_L2", "run_boundary")
)
renames_t = [
    ("def _run_case(", "def _run_tiled_case("),
    ("def _run_special_case(", "def _run_tiled_special_case("),
    ("def run_L0(", "def _run_tiled_L0("),
    ("def run_L1(", "def _run_tiled_L1("),
    ("def run_L2(", "def _run_tiled_L2("),
    ("def run_boundary(", "def _run_tiled_boundary("),
    ("_run_case(", "_run_tiled_case("),
    ("_run_special_case(", "_run_tiled_special_case("),
]
for a, b in renames_t:
    tiled_suite = tiled_suite.replace(a, b)

# --- shared golden + dtype/tolerance helpers (from single; identical in tiled) ---
shared = "".join(
    grab(s_lines, s_sec, n) + "\n\n" for n in
    ("golden_logsumexp", "_source_lse_sim", "_tolerance", "_dtype_str")
)

header = '''# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""LogSumExpFwdOp -- Stage 4 perf_opt FINAL canonical artifact (dual kernel).

This is the state-contract final artifact (``kernel_opt_py_path`` =
``perf_opt/LogSumExpFwdOp.py``, see ``.stage_state.json``): the consolidated
module containing BOTH tuned kernels of the LogSumExpFwdOp integration:

  * ``_logsumexp_kernel_single`` -- single-tile residency variant, tuned
    (round 1 OP1 ROWS re-derivation + round 2 OP5 threads=256).
  * ``_logsumexp_kernel_tiled`` -- N-chunk streaming variant, tuned
    (round 2 OP6 threads=256; BLOCK_N keeps the baseline rule -- E2/E6
    rejected both deviations).

The per-kernel drop-in files ``perf_opt/_logsumexp_kernel_single.py`` /
``perf_opt/_logsumexp_kernel_tiled.py`` contain the same frozen kernel code
(byte-frozen measurement artifacts; the TileOPs wrapper's perf_opt import
switch references those paths).  This module is the single-file consolidated
view required by the Stage 4 gate; keep the three files in sync (they are
frozen deliverables -- no further edits expected).

Tuned defaults are exposed as module-level constants (Phase 3 item 3):
``TUNED_SIMTVF_THREADS = 256`` (both variants) and ``TUNED_ROWS_CANDIDATES``
(single ROWS heuristic).  Call surfaces are wrapper-compatible and unchanged:

    _logsumexp_kernel_single(M, N, dtype)(block_m)(x) -> y
    _logsumexp_kernel_tiled(M, N, dtype, tile_n)(block_m)(x) -> y

golden = torch.logsumexp(x.float(), dim=-1).to(x.dtype) computed on NPU.
Precision: full L0/L1/L2/boundary suites of both variants (see run_L0/run_L1/
run_L2/run_boundary -- each runs the single and tiled suites).
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T


'''

out = [header, single_consts, "\n\n", single_helpers, "\n\n", grab(s_lines, s_sec, "_pick_threads"), "\n\n",
       shared, single_builder, "\n\n", tiled_helpers, "\n\n", tiled_builder, "\n\n", tiled_call, "\n\n",
       "# ---------- Layered testing (both variants) ----------\n",
       single_suite, tiled_suite,
       '''# ---------- Combined suite entrypoints (Stage 4 gate contract) ----------
def run_L0():
    """L0 gate cases of BOTH variants (single first, then tiled)."""
    _run_single_L0()
    _run_tiled_L0()


def run_L1():
    _run_single_L1()
    _run_tiled_L1()


def run_L2():
    _run_single_L2()
    _run_tiled_L2()


def run_boundary():
    _run_single_boundary()
    _run_tiled_boundary()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"])
    args, _ = parser.parse_known_args()
    if args.level == "L0":
        run_L0()
    elif args.level == "L1":
        run_L1()
    elif args.level == "L2":
        run_L2()
    elif args.level == "boundary":
        run_boundary()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\\033[92mAll check passed!\\033[0m")


if __name__ == "__main__":
    main()
''']

open(f"{BASE}/LogSumExpFwdOp.py", "w", encoding="utf-8").write("".join(out))
print("assembled LogSumExpFwdOp.py")
import subprocess
r = subprocess.run(["python3", "-c", "import ast; ast.parse(open('{}').read()); print('syntax OK')".format(f"{BASE}/LogSumExpFwdOp.py")], capture_output=True, text=True)
print(r.stdout, r.stderr)
