#!/usr/bin/env python3
"""Assemble perf_opt/LogSumExpFwdOpSimd.py (Stage 4 gate consolidated artifact).

Byte-exact extraction of the frozen kernel bodies + tuned constants from the
per-function drop-in files; test scaffolding gets the SIMT-sibling rename
treatment (_run_case -> _run_single_case/_run_tiled_case, tiled _call ->
_call_tiled, run_L0..boundary -> _run_single_*/_run_tiled_* + combined
entrypoints).  Every segment asserts its first line to catch off-by-one.
"""

import os

PERF_OPT = (
    "/home/z00875089/tilelang-fork/examples/TileOPs/tileops/kernels/"
    "reduction/logsumexp_simd/logsumexp_simd_kernel/perf_opt"
)
SINGLE = os.path.join(PERF_OPT, "_logsumexp_kernel_single.py")
TILED = os.path.join(PERF_OPT, "_logsumexp_kernel_tiled.py")
OUT = os.path.join(PERF_OPT, "LogSumExpFwdOpSimd.py")


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def seg(lines, start, end, expect_first, expect_last=None):
    """1-based inclusive slice with boundary assertions."""
    first = lines[start - 1]
    assert first == expect_first, (
        f"segment {start}-{end}: first line mismatch\n  got: {first!r}\n  exp: {expect_first!r}"
    )
    if expect_last is not None:
        last = lines[end - 1]
        assert last == expect_last, (
            f"segment {start}-{end}: last line mismatch\n  got: {last!r}\n  exp: {expect_last!r}"
        )
    return "\n".join(lines[start - 1 : end])


def apply_renames(text, renames):
    for old, new in renames:
        assert old in text, f"rename target {old!r} not found in segment"
        text = text.replace(old, new)
    return text


s = read_lines(SINGLE)
t = read_lines(TILED)

# ---------------------------------------------------------------- authored
HEADER = '''# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""LogSumExpFwdOpSimd -- Stage 4 perf_opt FINAL canonical artifact (dual kernel).

This is the state-contract final artifact (``kernel_opt_py_path`` =
``perf_opt/LogSumExpFwdOpSimd.py``, see ``.stage_state.json``): the
consolidated module containing BOTH tuned kernels of the LogSumExpFwdOpSimd
integration (explicit-SIMD / SimdVF variant, 2048-bit ``T.simd.*``
intrinsics):

  * ``_logsumexp_kernel_single`` -- single-tile residency variant, tuned
    (round 1 f16max: pass-1 row max computed in the 16-bit input domain --
    exact / bit-identical output; bf16 goes through an EVEN/ODD fp32 split
    because asc_reduce_max has no bf16 overload).
  * ``_logsumexp_kernel_tiled`` -- N-chunk streaming variant, tuned
    (round 1 f16max + ROWS policy via ``_pick_rows``: M <= 56 -> ROWS=1 for
    core parallelism, else the largest pow2 whose makespan stays within
    min+1, capped 8 -- batches the 2-byte lse UB->GM writes; BLOCK_N keeps
    the wrapper tile_n direct-retention rule: the DESIGN 8.3 A/B sweet-spot
    candidate 16384 was measured and rejected, -52%).

The per-kernel drop-in files ``perf_opt/_logsumexp_kernel_single.py`` /
``perf_opt/_logsumexp_kernel_tiled.py`` contain the same frozen kernel code
(byte-frozen measurement artifacts; the TileOPs wrapper's perf_opt import
switch references those paths).  This module is the single-file consolidated
view required by the Stage 4 gate; keep the three files in sync (they are
frozen deliverables -- no further edits expected).

Tuned defaults are exposed as module-level constants: ``HMAX_ENABLE = True``
(16-bit-domain pass-1 max, shared by both variants) and the tiled
``_pick_rows`` ROWS policy (env override ``TILELANG_LSE_ROWS`` kept for
sweep reproducibility).  Call surfaces are wrapper-compatible and unchanged:

    _logsumexp_kernel_single(M, N, dtype)(block_m)(x) -> y
    _logsumexp_kernel_tiled(M, N, dtype, tile_n)(block_m)(x) -> y

golden = torch.logsumexp(x.float(), dim=-1).to(x.dtype) computed on NPU.
Precision: full L0/L1/L2/boundary suites of both variants (see run_L0/
run_L1/run_L2/run_boundary -- each runs the single and tiled suites).
"""

import argparse
import os
import sys

import torch
import tilelang
import tilelang.ascend.language as T'''

CONSTANTS = """# 56 = vector_core_num of Ascend950PR_9579 (torch.npu.get_device_properties()).
# Pure-Vector operator -> persistent grid of 56 AIV cores (DESIGN section 5.5).
NUM_BLOCKS = 56

# Single-tile column cap from the TileOPs wrapper (_primitives.py MAX_SINGLE_TILE_COLS).
MAX_SINGLE_TILE_COLS = 32512

# UB budget under SimdVF: 248 KiB * 0.9 discount (DESIGN section 4.5,
# CONST-capacity-a5 assumption; SimdVF has no SimtVF 32 KiB thread-context
# deduction).  Single variant: VRF is streaming (~2.5 KiB), so SUB is capped
# by UB alone.  Tiled variant: same budget with the UB_RESERVE_BYTES static
# reserve (lse_ub + alignment/misc headroom) deducted in the chunk clamp.
UB_BUDGET_BYTES = int(248 * 1024 * 0.9)  # 228556
UB_RESERVE_BYTES = 8256

# Wrapper heuristic budget (216 KiB = SimtVF UB budget; the wrapper tiled
# path sizes tile_n against this).  Used only in run_L0 to recompute the
# wrapper tile_n expectation via the in-repo _primitives mirror (DESIGN 8.2).
WRAPPER_BUDGET_BYTES = 216 * 1024

# SimdVF fp32 lane count == group width (64 fp32 / 128 fp16 per instruction).
GROUP = 64"""

COMBINED_MAIN = '''# ---------- Combined suite entrypoints (Stage 4 gate contract) ----------
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
    parser.add_argument(
        "--show-source",
        default=None,
        choices=["single", "tiled"],
        help="print the generated AscendC source of one variant",
    )
    args, _ = parser.parse_known_args()

    if args.show_source == "single":
        _show_single_source()
        return
    if args.show_source == "tiled":
        _show_tiled_source()
        return
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
    main()'''

# ------------------------------------------------------------- segments
pieces = [HEADER, CONSTANTS]

# ---- shared helpers (identical text in both files; taken from single) ----
pieces.append(seg(s, 38, 44, "# Stage 4 tuned final (SimdVF single variant):"))
pieces.append(seg(s, 68, 69, "def _elem_bytes(dtype: str) -> int:"))
pieces.append(seg(s, 72, 76, "# ---------- Golden (PyTorch reference, computed on NPU) ----------"))
pieces.append(seg(s, 79, 92, "def _source_lse_sim(x: torch.Tensor) -> torch.Tensor:"))
pieces.append(
    seg(t, 111, 136, "def _source_lse_tiled_sim(x: torch.Tensor, tile_n: int) -> torch.Tensor:")
)
pieces.append(seg(s, 95, 98, "def _tolerance(dtype: torch.dtype) -> float:"))

# ---- single kernel section (verbatim) ----
pieces.append(seg(s, 101, 110, "# ---------- Kernel builder ----------"))
pieces.append(
    seg(
        s, 113, 291, "def _logsumexp_kernel_single(M: int, N: int, dtype: str):", "    return _func"
    )
)
pieces.append(seg(s, 294, 301, "# ---------- Call helpers ----------"))

# ---- tiled kernel section (verbatim; _call renamed per sibling template) ----
pieces.append(seg(t, 52, 61, "# Stage 4 tuned final (SimdVF tiled variant):"))
pieces.append(seg(t, 139, 149, "# ---------- Kernel builder ----------"))
pieces.append(
    seg(
        t,
        152,
        160,
        "# ---------- Stage 4 Round 1: pass-1 max in the 16-bit input domain ----------",
    )
)
pieces.append(
    seg(
        t,
        163,
        182,
        "# ---------- Stage 4 Round 2: + ROWS policy (rows per persistent tile) --------",
    )
)
pieces.append(seg(t, 185, 187, "def _env_int(name, default):"))
pieces.append(
    seg(
        t,
        190,
        404,
        "def _logsumexp_kernel_tiled(M: int, N: int, dtype: str, tile_n: int):",
        "    return _func",
    )
)
pieces.append(
    apply_renames(
        seg(t, 408, 410, "def _call(M, N, dtype, tile_n, block_m, x):"),
        [("def _call(", "def _call_tiled(")],
    )
)
pieces.append(
    seg(t, 413, 440, "# ---------- Wrapper tile_n recomputation (DESIGN 8.2 / 11.2) ----------")
)

# ---- single layered tests (rename to _run_single_*) ----
pieces.append(
    apply_renames(
        seg(s, 304, 314, "# ---------- Layered testing ----------"),
        [("def _run_case(", "def _run_single_case(")],
    )
)
pieces.append(
    apply_renames(
        seg(s, 317, 337, "def _run_special_case(x, block_m, tag, *, source_semantics=False):"),
        [("def _run_special_case(", "def _run_single_special_case(")],
    )
)
pieces.append(
    apply_renames(
        seg(s, 340, 349, "def run_L0():"),
        [("def run_L0(", "def _run_single_L0("), ("_run_case(", "_run_single_case(")],
    )
)
pieces.append(
    apply_renames(
        seg(s, 352, 369, "def run_L1():"),
        [("def run_L1(", "def _run_single_L1("), ("_run_case(", "_run_single_case(")],
    )
)
pieces.append(
    apply_renames(
        seg(s, 372, 392, "def run_L2():"),
        [("def run_L2(", "def _run_single_L2("), ("_run_case(", "_run_single_case(")],
    )
)
pieces.append(
    apply_renames(
        seg(s, 395, 453, "def run_boundary():"),
        [
            ("def run_boundary(", "def _run_single_boundary("),
            ("_run_case(", "_run_single_case("),
            ("_run_special_case(", "_run_single_special_case("),
        ],
    )
)
pieces.append(
    apply_renames(
        seg(s, 456, 459, "def _show_source():"), [("def _show_source(", "def _show_single_source(")]
    )
)

# ---- tiled layered tests (rename to _run_tiled_* + _call_tiled) ----
pieces.append(
    apply_renames(
        seg(t, 443, 456, "# ---------- Layered testing ----------"),
        [("def _run_case(", "def _run_tiled_case("), ("_call(", "_call_tiled(")],
    )
)
pieces.append(
    apply_renames(
        seg(
            t,
            459,
            478,
            "def _run_special_case(x, block_m, tile_n, tag, *, source_semantics=False):",
        ),
        [("def _run_special_case(", "def _run_tiled_special_case("), ("_call(", "_call_tiled(")],
    )
)
pieces.append(
    apply_renames(
        seg(t, 481, 499, "def run_L0():"),
        [("def run_L0(", "def _run_tiled_L0("), ("_run_case(", "_run_tiled_case(")],
    )
)
pieces.append(
    apply_renames(
        seg(t, 502, 530, "def run_L1():"),
        [("def run_L1(", "def _run_tiled_L1("), ("_run_case(", "_run_tiled_case(")],
    )
)
pieces.append(
    apply_renames(
        seg(t, 533, 549, "def run_L2():"),
        [("def run_L2(", "def _run_tiled_L2("), ("_run_case(", "_run_tiled_case(")],
    )
)
pieces.append(
    apply_renames(
        seg(t, 552, 647, "def run_boundary():"),
        [
            ("def run_boundary(", "def _run_tiled_boundary("),
            ("_run_special_case(", "_run_tiled_special_case("),
            ("_call(", "_call_tiled("),
        ],
    )
)
pieces.append(
    apply_renames(
        seg(t, 650, 653, "def _show_source():"), [("def _show_source(", "def _show_tiled_source(")]
    )
)

# ---- combined entrypoints + main ----
pieces.append(COMBINED_MAIN)

body = "\n\n\n".join(p.strip("\n") for p in pieces) + "\n"

with open(OUT, "w", encoding="utf-8") as f:
    f.write(body)

import ast  # noqa: E402

ast.parse(body)
print(f"wrote {OUT} ({len(body.splitlines())} lines), syntax OK")
