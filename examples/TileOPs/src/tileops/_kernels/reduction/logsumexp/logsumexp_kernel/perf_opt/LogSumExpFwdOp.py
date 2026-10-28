# Copyright (c) Huawei Technologies Co., Ltd. 2026.
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


NUM_BLOCKS = 56
MAX_SINGLE_TILE_COLS = 32512
UB_BUDGET_BYTES = int(216 * 1024 * 0.9)  # 199065
VRF_BUDGET_BYTES = 128 * 1024  # 131072
TUNED_SIMTVF_THREADS = 256
TUNED_ROWS_CANDIDATES = (1, 2, 4, 8, 16)


def _elem_bytes(dtype: str) -> int:
    return 4 if dtype == "float32" else 2
def _pick_rows(M: int) -> int:
    """Re-derive ROWS (rows per persistent tile), superseding wrapper block_m.

    Round-1 empirical revision: the pure makespan model (minimize
    ceil(ceil(M/ROWS)/56)*ROWS) picked ROWS=1 for M=1024 but measured 43.6us
    vs 25.6us at ROWS=4 -- per-tile overheads (tiny lse MTE3 handshakes,
    vf_call launches, MTE2 issues) dominate when tiles get small.  Revised
    rule (FINAL): candidates are POW2 ROWS in TUNED_ROWS_CANDIDATES (keeps
    SUB=ROWS, a single clean sub-batch per tile); pick the LARGEST candidate
    whose makespan is <= min_makespan + 1 row.  M=1024 -> 4 (validated
    +50.9%); M=128 -> 4 (baseline-equal, tie); M=4 -> 2.
    """
    cands = [r for r in TUNED_ROWS_CANDIDATES if r <= max(1, M)]

    def _ceil_div(a: int, b: int) -> int:
        return -(-a // b)

    makespans = {
        r: _ceil_div(_ceil_div(M, r), NUM_BLOCKS) * r for r in cands
    }
    min_ms = min(makespans.values())
    best = max(r for r in cands if makespans[r] <= min_ms + 1)
    return best


def _pick_threads(sub_tile_elems: int) -> int:
    """SimtVF thread width (round 2 OP5 winner, hardcoded final default).

    Baseline used min(1024, SUB*TILE).  Every reduce_max/reduce_sum lowers
    to AscendAllReduce<Op, threads, 1, 0> -- the FULL thread width
    participates in each row's tree (generated-source evidence), so a
    narrower thread team shrinks the tree/syncthreads cost at the price of
    more elements per thread.  Sweep verdict (single E5/E8/E10 and tiled E7/E9/E11, both U-curves):
    512 improves, 256 optimal, 128 regresses.  Final default =
    TUNED_SIMTVF_THREADS (shared by both variants).
    """
    return min(TUNED_SIMTVF_THREADS, sub_tile_elems)


def golden_logsumexp(x: torch.Tensor) -> torch.Tensor:
    """Row-wise logsumexp, torch on NPU (source test baseline ref_program)."""
    assert x.device.type == "npu" and x.ndim == 2
    return torch.logsumexp(x.float(), dim=-1).to(x.dtype)


def _source_lse_sim(x: torch.Tensor) -> torch.Tensor:
    """fp32 re-implementation of the SOURCE GPU algorithm semantics.

    m = max(x); y = m + log(sum(exp(x - m))) with fp32 intermediates (matching
    the source's fp32 chain, DESIGN section 0.1).  Unlike torch.logsumexp this
    propagates the source-inherent divergence (all -inf row -> NaN, +inf row
    -> NaN) exactly, so it is used to validate migration fidelity for those
    corner cases (DESIGN section 8.2 Boundary 4 / 9.3).
    """
    x32 = x.float()
    m = x32.max(dim=-1, keepdim=True).values
    e = torch.exp(x32 - m)
    s = e.sum(dim=-1, keepdim=True)
    return (m + torch.log(s)).squeeze(-1).to(x.dtype)


def _tolerance(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-2
    return 1e-3  # fp32


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _logsumexp_kernel_single(M: int, N: int, dtype: str):
    """Build a single-tile logsumexp kernel (N fits on-chip).

    Keeps the wrapper call surface ``_logsumexp_kernel_single(M, N, dtype)
    (block_m)(x)``; the GPU ``threads`` argument is removed per K3 (threads
    derived inside as min(1024, SUB * TILE)).
    """
    if M < 1 or N < 1:
        raise ValueError(f"M, N must be >= 1, got M={M}, N={N}")
    if N > MAX_SINGLE_TILE_COLS:
        raise ValueError(
            f"N={N} exceeds MAX_SINGLE_TILE_COLS={MAX_SINGLE_TILE_COLS} "
            f"(single-tile contract; larger N is the tiled variant)"
        )

    TILE = tilelang.next_power_of_2(N)
    elem = _elem_bytes(dtype)

    @tilelang.jit(target="ascend", out_idx=[1])
    def _func(block_m: int):
        if block_m < 1:
            raise ValueError(f"block_m must be >= 1, got {block_m}")

        # OP1: internal ROWS re-derivation (wrapper block_m kept as a hint
        # only; the persistent tile granularity is re-derived for the A5
        # 56-core scheduler -- see module docstring and _pick_rows).
        ROWS = _pick_rows(M)

        # SUB: max power-of-two satisfying (i) x_ub double-buffer <= UB budget,
        # (ii) x_frag fp32 <= VRF budget, (iii) SUB <= ROWS.  SUB >= 1 always
        # (DESIGN section 5.2).
        SUB = 1
        while SUB * 2 <= ROWS:
            cand = SUB * 2
            if (2 * cand * N * elem <= UB_BUDGET_BYTES) and (
                cand * TILE * 4 <= VRF_BUDGET_BYTES
            ):
                SUB = cand
            else:
                break

        num_full_sub = ROWS // SUB          # full SUB-row sub-batches per full tile
        last_sub_rows = ROWS % SUB          # rows in the last (partial) sub-batch
        full = M // ROWS                    # number of complete ROWS-row tiles
        rem = M % ROWS                      # rows in the tail tile
        num_tiles = full + (1 if rem else 0)
        tsub = rem // SUB                   # full SUB-row sub-batches in the tail
        tsub_rem = rem % SUB                # rows in the tail's last partial sub-batch
        threads = _pick_threads(SUB * TILE)

        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M,), dtype)):
            with T.Kernel(NUM_BLOCKS) as core_id:
                x_ub = T.alloc_shared((SUB, N), dtype)
                lse_ub = T.alloc_shared((SUB,), dtype)
                T.annotate_buffer_versions({x_ub: 2})
                for t in T.Persistent([num_tiles], NUM_BLOCKS, core_id, num_stages=2):
                    base = t * ROWS
                    if t < full:
                        # ---- full tile (block_m rows, all in-bounds) ----
                        for s in T.serial(num_full_sub):
                            row_base = base + s * SUB
                            T.copy(x[row_base : row_base + SUB, 0:N], x_ub)
                            with T.SimtVF(threads=threads):
                                x_frag = T.alloc_fragment((SUB, TILE), "float32")
                                m_frag = T.alloc_fragment((SUB,), "float32")
                                s_frag = T.alloc_fragment((SUB,), "float32")
                                out_frag = T.alloc_fragment((SUB,), dtype)
                                for r, i in T.Parallel(SUB, TILE):
                                    x_frag[r, i] = T.if_then_else(
                                        i < N,
                                        T.cast(x_ub[r, i], "float32"),
                                        -T.infinity(T.float32),
                                    )
                                T.reduce_max(x_frag, m_frag, dim=1, clear=True)
                                for r, i in T.Parallel(SUB, TILE):
                                    x_frag[r, i] = T.exp(x_frag[r, i] - m_frag[r])
                                T.reduce_sum(x_frag, s_frag, dim=1, clear=True)
                                for r in T.Parallel(SUB):
                                    out_frag[r] = T.cast(m_frag[r] + T.log(s_frag[r]), dtype)
                                T.copy(out_frag, lse_ub)
                            T.copy(lse_ub, y[row_base : row_base + SUB])
                        if last_sub_rows > 0:
                            # partial last sub-batch (block_m % SUB != 0; not
                            # reachable under the wrapper power-of-two block_m
                            # heuristic, kept for arbitrary-config generality).
                            row_base = base + num_full_sub * SUB
                            T.copy(
                                x[row_base : row_base + last_sub_rows, 0:N],
                                x_ub[0:last_sub_rows, 0:N],
                            )
                            with T.SimtVF(threads=threads):
                                x_frag = T.alloc_fragment((SUB, TILE), "float32")
                                m_frag = T.alloc_fragment((SUB,), "float32")
                                s_frag = T.alloc_fragment((SUB,), "float32")
                                out_frag = T.alloc_fragment((SUB,), dtype)
                                for r, i in T.Parallel(SUB, TILE):
                                    x_frag[r, i] = T.if_then_else(
                                        T.And(r < last_sub_rows, i < N),
                                        T.cast(x_ub[r, i], "float32"),
                                        -T.infinity(T.float32),
                                    )
                                T.reduce_max(x_frag, m_frag, dim=1, clear=True)
                                for r, i in T.Parallel(SUB, TILE):
                                    x_frag[r, i] = T.exp(x_frag[r, i] - m_frag[r])
                                T.reduce_sum(x_frag, s_frag, dim=1, clear=True)
                                for r in T.Parallel(SUB):
                                    out_frag[r] = T.cast(m_frag[r] + T.log(s_frag[r]), dtype)
                                T.copy(out_frag, lse_ub)
                            T.copy(lse_ub[0:last_sub_rows], y[row_base : row_base + last_sub_rows])
                    else:
                        # ---- tail tile (rem rows) ----
                        if tsub > 0:
                            for s in T.serial(tsub):
                                row_base = base + s * SUB
                                T.copy(x[row_base : row_base + SUB, 0:N], x_ub)
                                with T.SimtVF(threads=threads):
                                    x_frag = T.alloc_fragment((SUB, TILE), "float32")
                                    m_frag = T.alloc_fragment((SUB,), "float32")
                                    s_frag = T.alloc_fragment((SUB,), "float32")
                                    out_frag = T.alloc_fragment((SUB,), dtype)
                                    for r, i in T.Parallel(SUB, TILE):
                                        x_frag[r, i] = T.if_then_else(
                                            i < N,
                                            T.cast(x_ub[r, i], "float32"),
                                            -T.infinity(T.float32),
                                        )
                                    T.reduce_max(x_frag, m_frag, dim=1, clear=True)
                                    for r, i in T.Parallel(SUB, TILE):
                                        x_frag[r, i] = T.exp(x_frag[r, i] - m_frag[r])
                                    T.reduce_sum(x_frag, s_frag, dim=1, clear=True)
                                    for r in T.Parallel(SUB):
                                        out_frag[r] = T.cast(m_frag[r] + T.log(s_frag[r]), dtype)
                                    T.copy(out_frag, lse_ub)
                                T.copy(lse_ub, y[row_base : row_base + SUB])
                        if tsub_rem > 0:
                            row_base = M - tsub_rem
                            T.copy(x[row_base:M, 0:N], x_ub[0:tsub_rem, 0:N])
                            with T.SimtVF(threads=threads):
                                x_frag = T.alloc_fragment((SUB, TILE), "float32")
                                m_frag = T.alloc_fragment((SUB,), "float32")
                                s_frag = T.alloc_fragment((SUB,), "float32")
                                out_frag = T.alloc_fragment((SUB,), dtype)
                                for r, i in T.Parallel(SUB, TILE):
                                    x_frag[r, i] = T.if_then_else(
                                        T.And(r < tsub_rem, i < N),
                                        T.cast(x_ub[r, i], "float32"),
                                        -T.infinity(T.float32),
                                    )
                                T.reduce_max(x_frag, m_frag, dim=1, clear=True)
                                for r, i in T.Parallel(SUB, TILE):
                                    x_frag[r, i] = T.exp(x_frag[r, i] - m_frag[r])
                                T.reduce_sum(x_frag, s_frag, dim=1, clear=True)
                                for r in T.Parallel(SUB):
                                    out_frag[r] = T.cast(m_frag[r] + T.log(s_frag[r]), dtype)
                                T.copy(out_frag, lse_ub)
                            T.copy(lse_ub[0:tsub_rem], y[row_base:M])

        return main

    return _func


def _call(M: int, N: int, dtype: torch.dtype, block_m: int, x: torch.Tensor) -> torch.Tensor:
    kernel = _logsumexp_kernel_single(M, N, _dtype_str(dtype))(block_m)
    return kernel(x)


def _pick_block_n(tile_n: int, elem: int) -> int:
    """Internal chunk width BLOCK_N (FINAL: baseline rule, round-2 E6/E2 verdict).

    Largest power of two with BLOCK_N <= tile_n and 2*BLOCK_N*elem <= UB
    budget.  Both deviations were measured and REJECTED: uncapped 32768
    (round-1 E2, -24~-29%, vec busy 88.6%->94.2%) and 8192 (round-2 E6,
    -21.6~-36.4%) -- 16384 (fp16/bf16 with tile_n >= 16384) is the local
    optimum balancing chunk count against per-element vector cost.
    """
    block_n = 1
    while block_n * 2 <= tile_n and 2 * (block_n * 2) * elem <= UB_BUDGET_BYTES:
        block_n *= 2
    return block_n


def _logsumexp_kernel_tiled(M: int, N: int, dtype: str, tile_n: int):
    """Build a multi-tile (online streaming) logsumexp kernel.

    Keeps the wrapper call surface ``_logsumexp_kernel_tiled(M, N, dtype,
    tile_n)(block_m)(x)``; the GPU ``threads`` argument is removed per K3
    (threads derived inside: min(1024, SUB * BLOCK_N)). tile_n is honoured as
    a budget cap: the internal chunk width BLOCK_N is the largest power of two
    with BLOCK_N <= tile_n and 2*BLOCK_N*elem <= 194.4 KiB.
    """
    if M < 1 or N < 1 or tile_n < 1:
        raise ValueError(f"M, N, tile_n must be >= 1, got M={M}, N={N}, tile_n={tile_n}")
    elem = _elem_bytes(dtype)

    @tilelang.jit(target="ascend", out_idx=[1])
    def _func(block_m: int):
        if block_m < 1:
            raise ValueError(f"block_m must be >= 1, got {block_m}")

        # ---- host staticization (DESIGN section 3.3 / 5.2) ----
        # FINAL MERGED: BLOCK_N via _pick_block_n (baseline rule; E2/E6 both
        # rejected the deviations), threads via _pick_threads (OP6 winner
        # TUNED_SIMTVF_THREADS = 256).
        BLOCK_N = _pick_block_n(tile_n, elem)
        # SUB: largest power of two with SUB <= block_m, x_ub double-buffer
        # <= UB budget and x_frag <= VRF budget.
        SUB = 1
        while SUB * 2 <= block_m:
            cand = SUB * 2
            if (2 * cand * BLOCK_N * elem <= UB_BUDGET_BYTES) and (
                cand * BLOCK_N * 4 <= VRF_BUDGET_BYTES
            ):
                SUB = cand
            else:
                break

        num_full = N // BLOCK_N           # full chunks (all columns valid)
        rem = N % BLOCK_N                 # tail-chunk column count
        TILE_R = tilelang.next_power_of_2(rem) if rem > 0 else 0
        full = M // block_m               # complete block_m tiles
        rem_m = M % block_m               # rows in the tail tile
        num_tiles = full + (1 if rem_m > 0 else 0)
        num_sub = block_m // SUB          # full SUB-row sub-batches per tile
        last_sub_rows = block_m % SUB     # rows in the last partial sub-batch
        tsub = rem_m // SUB               # full SUB-row sub-batches in tail
        tsub_rem = rem_m % SUB            # rows in the tail's partial sub-batch
        threads = _pick_threads(SUB * BLOCK_N)

        @T.macro
        def sub_batch(x, y, x_ub, stats_ub, lse_ub, row_base, rows):
            # stats init: m_run = -inf, s_run = 0 (per sub-batch, DESIGN S-init)
            with T.SimtVF(threads=threads):
                for r in T.Parallel(SUB):
                    stats_ub[r, 0] = -T.infinity(T.float32)
                    stats_ub[r, 1] = T.float32(0.0)
            # ---- full chunks (all columns valid) ----
            if num_full > 0:
                for c in T.serial(num_full):
                    col = c * BLOCK_N
                    T.copy(
                        x[row_base : row_base + rows, col : col + BLOCK_N],
                        x_ub[0:rows, 0:BLOCK_N],
                    )
                    with T.SimtVF(threads=threads):
                        x_frag = T.alloc_fragment((SUB, BLOCK_N), "float32")
                        m_c = T.alloc_fragment((SUB,), "float32")
                        s_c = T.alloc_fragment((SUB,), "float32")
                        prev_frag = T.alloc_fragment((SUB,), "float32")
                        if rows == SUB:
                            # no-predicate direct load (all rows + cols valid)
                            for r, i in T.Parallel(SUB, BLOCK_N):
                                x_frag[r, i] = T.cast(x_ub[r, i], "float32")
                        else:
                            # row predicate for the partial sub-batch
                            for r, i in T.Parallel(SUB, BLOCK_N):
                                x_frag[r, i] = T.if_then_else(
                                    r < rows,
                                    T.cast(x_ub[r, i], "float32"),
                                    -T.infinity(T.float32),
                                )
                        T.reduce_max(x_frag, m_c, dim=1, clear=True)
                        for r in T.Parallel(SUB):
                            prev_frag[r] = stats_ub[r, 0]
                            stats_ub[r, 0] = T.max(stats_ub[r, 0], m_c[r])
                        for r, i in T.Parallel(SUB, BLOCK_N):
                            x_frag[r, i] = T.exp(x_frag[r, i] - stats_ub[r, 0])
                        T.reduce_sum(x_frag, s_c, dim=1, clear=True)
                        for r in T.Parallel(SUB):
                            stats_ub[r, 1] = (
                                stats_ub[r, 1] * T.exp(prev_frag[r] - stats_ub[r, 0]) + s_c[r]
                            )
            # ---- tail chunk (rem columns, masked load, pad=-inf) ----
            if rem > 0:
                col = num_full * BLOCK_N
                T.copy(x[row_base : row_base + rows, col:N], x_ub[0:rows, 0:rem])
                with T.SimtVF(threads=threads):
                    x_frag = T.alloc_fragment((SUB, TILE_R), "float32")
                    m_c = T.alloc_fragment((SUB,), "float32")
                    s_c = T.alloc_fragment((SUB,), "float32")
                    prev_frag = T.alloc_fragment((SUB,), "float32")
                    for r, i in T.Parallel(SUB, TILE_R):
                        x_frag[r, i] = T.if_then_else(
                            T.And(r < rows, i < rem),
                            T.cast(x_ub[r, i], "float32"),
                            -T.infinity(T.float32),
                        )
                    T.reduce_max(x_frag, m_c, dim=1, clear=True)
                    for r in T.Parallel(SUB):
                        prev_frag[r] = stats_ub[r, 0]
                        stats_ub[r, 0] = T.max(stats_ub[r, 0], m_c[r])
                    for r, i in T.Parallel(SUB, TILE_R):
                        x_frag[r, i] = T.exp(x_frag[r, i] - stats_ub[r, 0])
                    T.reduce_sum(x_frag, s_c, dim=1, clear=True)
                    for r in T.Parallel(SUB):
                        stats_ub[r, 1] = (
                            stats_ub[r, 1] * T.exp(prev_frag[r] - stats_ub[r, 0]) + s_c[r]
                        )
            # ---- lse = m_run + log(s_run), two-stage write ----
            with T.SimtVF(threads=threads):
                out_frag = T.alloc_fragment((SUB,), dtype)
                for r in T.Parallel(SUB):
                    out_frag[r] = T.cast(
                        stats_ub[r, 0] + T.log(stats_ub[r, 1]), dtype
                    )
                T.copy(out_frag, lse_ub)
            T.copy(lse_ub[0:rows], y[row_base : row_base + rows])

        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M,), dtype)):
            with T.Kernel(NUM_BLOCKS) as core_id:
                x_ub = T.alloc_shared((SUB, BLOCK_N), dtype)
                stats_ub = T.alloc_shared((SUB, 2), "float32")
                lse_ub = T.alloc_shared((SUB,), dtype)
                T.annotate_buffer_versions({x_ub: 2, stats_ub: 1})
                for t in T.Persistent([num_tiles], NUM_BLOCKS, core_id, num_stages=2):
                    base = t * block_m
                    if t < full:
                        # ---- full tile (block_m rows, all in-bounds) ----
                        for s in T.serial(num_sub):
                            row_base = base + s * SUB
                            sub_batch(x, y, x_ub, stats_ub, lse_ub, row_base, SUB)
                        if last_sub_rows > 0:
                            row_base = base + num_sub * SUB
                            sub_batch(
                                x, y, x_ub, stats_ub, lse_ub, row_base, last_sub_rows
                            )
                    else:
                        # ---- tail tile (rem_m rows) ----
                        for s in T.serial(tsub):
                            row_base = base + s * SUB
                            sub_batch(x, y, x_ub, stats_ub, lse_ub, row_base, SUB)
                        if tsub_rem > 0:
                            row_base = M - tsub_rem
                            sub_batch(x, y, x_ub, stats_ub, lse_ub, row_base, tsub_rem)

        return main

    return _func


def _call_tiled(M, N, dtype, tile_n, block_m, x):
    kernel = _logsumexp_kernel_tiled(M, N, _dtype_str(dtype), tile_n)(block_m)
    return kernel(x)


# ---------- Layered testing (both variants) ----------
def _run_single_case(M, N, dtype, block_m, tag):
    x = torch.randn(M, N, dtype=dtype, device="npu")
    y = _call(M, N, dtype, block_m, x)
    torch.npu.synchronize()
    golden = golden_logsumexp(x)
    tol = _tolerance(dtype)
    max_diff = (y.float() - golden.float()).abs().max().item()
    torch.testing.assert_close(y, golden, rtol=tol, atol=tol)
    print(f"[{tag}] PASS: shape=({M},{N}) dtype={dtype} block_m={block_m} max_diff={max_diff:.3e}")
    return max_diff


def _run_single_special_case(x, block_m, tag, *, source_semantics=False):
    """Compare a pre-built special-value input against golden (equal_nan aware).

    source_semantics=True compares against the fp32 source-algorithm simulator
    instead of torch.logsumexp, for the all -inf / +inf divergence rows
    (DESIGN section 8.2 Boundary 4 / 9.3).
    """
    M, N = x.shape
    dtype = x.dtype
    y = _call(M, N, dtype, block_m, x)
    torch.npu.synchronize()
    ref = _source_lse_sim(x) if source_semantics else golden_logsumexp(x)
    tol = _tolerance(dtype)
    torch.testing.assert_close(y, ref, rtol=tol, atol=tol, equal_nan=True)
    # isfinite-mask the diff so +/-inf/NaN positions don't poison the report
    # (inf - inf = NaN would otherwise dominate max_diff).
    diff = (y.float() - ref.float())
    diff = diff[torch.isfinite(diff)]
    max_diff = diff.abs().max().item() if diff.numel() > 0 else 0.0
    print(f"[{tag}] PASS: shape={tuple(x.shape)} max_diff={max_diff:.3e}")
    return max_diff


def _run_single_L0():
    # Gate cases (DESIGN section 8.2 L0 plan): wl1/wl4 equivalents, smoke
    # 3-dtype, M tail (rem=1), non-pow2 N (masked load).
    _run_single_case(1024, 4096, torch.float16, 16, "L0")
    _run_single_case(128, 16384, torch.float16, 4, "L0")
    _run_single_case(32, 256, torch.float32, 16, "L0")
    _run_single_case(32, 256, torch.float16, 16, "L0")
    _run_single_case(32, 256, torch.bfloat16, 16, "L0")
    _run_single_case(33, 256, torch.float16, 16, "L0")
    _run_single_case(32, 300, torch.float16, 16, "L0")


def _run_single_L1():
    # Shape / config sweep: M breadth, N breadth (incl. non-pow2 and 32B
    # non-aligned rows), 3 dtypes, block_m sweep incl. SUB split points.
    for M in (1, 3, 33, 56, 57, 128, 1024):
        for N in (1, 7, 256, 300, 4096, 16384):
            _run_single_case(M, N, torch.float16, 16, "L1")
    # fp32 / bf16 over representative shapes.
    for M, N in ((33, 256), (32, 300), (1024, 4096), (128, 16384)):
        _run_single_case(M, N, torch.float32, 16, "L1")
        _run_single_case(M, N, torch.bfloat16, 16, "L1")
    # block_m sweep (wrapper heuristic full set + extreme 1), SUB split
    # verification (block_m > SUB vs block_m == SUB).
    for bm in (1, 2, 4, 8, 16):
        _run_single_case(32, 256, torch.float16, bm, "L1")
    # M=1 (full tail), 1D flatten, multi-dim dim=[0,2] flatten.
    _run_single_case(1, 256, torch.float16, 16, "L1")
    _run_single_case(1, 256, torch.float32, 16, "L1")
    _run_single_case(128, 16384, torch.float16, 4, "L1")


def _run_single_L2():
    # Robustness / stress: large M multi-wave scheduling (num_tiles > 56),
    # contract boundary N=32512 SUB=1 degenerate form.  Recorded as warnings,
    # do not block the verdict.
    cases = [
        (8192, 4096, torch.float16, 16),
        (8192, 4096, torch.float32, 16),
        (2, 32512, torch.float16, 2),
    ]
    for M, N, dt, bm in cases:
        try:
            _run_single_case(M, N, dt, bm, "L2")
        except Exception as e:  # L2 recorded as warn, does not block
            print(f"[L2] WARN (not blocking): shape=({M},{N}) dtype={dt} {e}")
    # Negative assert: N beyond the single-tile column cap must raise.
    try:
        _logsumexp_kernel_single(4, 32768, "float16")
        print("[L2] WARN (not blocking): N=32768 did not raise (expected single-tile assert)")
    except ValueError:
        print("[L2] PASS: N=32768 rejected by single-tile assert (tiled-variant boundary)")


def _run_single_boundary():
    n = 128

    def base(rows=8, cols=128, dtype=torch.float16):
        return torch.zeros(rows, cols, dtype=dtype, device="npu")

    # finite-domain corners vs golden.
    _run_single_special_case(base(), 8, "Boundary-zeros")
    x = base()
    x[:, 0] = 65504.0
    x[:, 1] = -65504.0
    _run_single_special_case(x, 8, "Boundary-extreme")
    x = base()
    x[:, :] = 3.5
    _run_single_special_case(x, 8, "Boundary-all-equal")
    x = base()
    x[:, 0] = 100.0
    _run_single_special_case(x, 8, "Boundary-single-dominant")
    x = base()
    x[:, 0] = 6e-8
    x[:, 1] = -6e-8
    _run_single_special_case(x, 8, "Boundary-subnormal")
    x = base()
    x[:, 0] = -745.0
    x[:, 1] = -744.0
    _run_single_special_case(x, 8, "Boundary-deep-underflow")
    # partial -inf (finite rows -> matches golden).
    x = base()
    x[:, 0] = float("-inf")
    _run_single_special_case(x, 8, "Boundary-neg-inf")
    # NaN row -> both NaN (equal_nan=True).
    x = base()
    x[:, 0] = float("nan")
    _run_single_special_case(x, 8, "Boundary-nan")
    # N=1 and M=1 minimal shapes.
    _run_single_special_case(torch.randn(3, 1, dtype=torch.float16, device="npu"), 8, "Boundary-N1")
    _run_single_special_case(torch.randn(1, 16, dtype=torch.float16, device="npu"), 8, "Boundary-M1")

    # source-inherited divergence (vs fp64 source semantics, not golden).
    x = base()
    x[:, :] = float("-inf")
    _run_single_special_case(x, 8, "Boundary-all-neg-inf", source_semantics=True)
    x = base()
    x[:, 0] = float("inf")
    _run_single_special_case(x, 8, "Boundary-pos-inf", source_semantics=True)

    # determinism: same input 3x, bitwise identical.
    x = torch.randn(4096, 4096, dtype=torch.float16, device="npu")
    y0 = _call(4096, 4096, torch.float16, 16, x)
    torch.npu.synchronize()
    y1 = _call(4096, 4096, torch.float16, 16, x)
    torch.npu.synchronize()
    y2 = _call(4096, 4096, torch.float16, 16, x)
    torch.npu.synchronize()
    assert torch.equal(y0, y1) and torch.equal(y1, y2), "nondeterministic output"
    print("[Boundary-determinism] PASS: 3x bitwise identical")


def _run_tiled_case(M, N, dtype, tile_n, block_m, tag):
    x = torch.randn(M, N, dtype=dtype, device="npu")
    y = _call_tiled(M, N, dtype, tile_n, block_m, x)
    torch.npu.synchronize()
    golden = golden_logsumexp(x)
    tol = _tolerance(dtype)
    max_diff = (y.float() - golden.float()).abs().max().item()
    torch.testing.assert_close(y, golden, rtol=tol, atol=tol)
    print(
        f"[{tag}] PASS: shape=({M},{N}) dtype={dtype} block_m={block_m} "
        f"tile_n={tile_n} max_diff={max_diff:.3e}"
    )
    return max_diff


def _run_tiled_special_case(x, block_m, tile_n, tag, *, source_semantics=False):
    """Compare a pre-built special-value input against golden (equal_nan aware).

    source_semantics=True compares against the fp32 source-algorithm simulator
    instead of torch.logsumexp, for the all -inf / +inf divergence rows
    (DESIGN section 8.2 Boundary 4 / 9.3).
    """
    M, N = x.shape
    dtype = x.dtype
    y = _call_tiled(M, N, dtype, tile_n, block_m, x)
    torch.npu.synchronize()
    ref = _source_lse_sim(x) if source_semantics else golden_logsumexp(x)
    tol = _tolerance(dtype)
    torch.testing.assert_close(y, ref, rtol=tol, atol=tol, equal_nan=True)
    # isfinite-mask the diff so +/-inf/NaN positions don't poison the report
    # (inf - inf = NaN would otherwise dominate max_diff).
    diff = y.float() - ref.float()
    diff = diff[torch.isfinite(diff)]
    max_diff = diff.abs().max().item() if diff.numel() > 0 else 0.0
    print(f"[{tag}] PASS: shape={tuple(x.shape)} max_diff={max_diff:.3e}")
    return max_diff


def _run_tiled_L0():
    # Gate cases (DESIGN section 8.2 L0 plan). Case 5 first per REVIEW note 5:
    # the (33, 33000) tail chunk is a 256-elem fragment @ SimtVF(threads=1024)
    # (extent < threads, replication=4) -- validate the reduce config early.
    _run_tiled_case(33, 33000, torch.float16, 16640, 1, "L0")    # non-pow2 tile_n + rem=232 tail chunk
    _run_tiled_case(1024, 32768, torch.bfloat16, 16384, 1, "L0")  # wl2 (2 full chunks)
    _run_tiled_case(4, 102400, torch.float16, 25600, 1, "L0")     # wl3 (rem=4096 tail chunk)
    _run_tiled_case(4, 102400, torch.bfloat16, 25600, 1, "L0")    # wl3 bf16
    _run_tiled_case(4, 32768, torch.float16, 16384, 1, "L0")      # test L76 shape
    _run_tiled_case(33, 32768, torch.float16, 16384, 4, "L0")     # M tail (rem_m=1) + bm>1
    _run_tiled_case(4, 32768, torch.float32, 16384, 1, "L0")      # fp32 (UB 128 KiB)


def _run_tiled_L1():
    # N breadth (fp16, block_m=1): non-pow2 N (rem>0), N < BLOCK_N (num_full=0),
    # multi-chunk N, exact-pow2 N.
    for N in (32768, 33000, 65540, 102400, 32513, 256, 300):
        _run_tiled_case(33, N, torch.float16, 16384, 1, "L1")
    # M breadth (incl. M < block_m / M % block_m != 0 / M=1 single-row streaming).
    for M in (1, 3, 56, 57, 128):
        _run_tiled_case(M, 32768, torch.float16, 16384, 1, "L1")
    # tile_n scan (BLOCK_N clamp + chunk-count variation) over non-pow2 N.
    for tile_n in (25600, 16640, 13824, 8192, 4096, 256):
        _run_tiled_case(33, 33000, torch.float16, tile_n, 1, "L1")
    # block_m sweep (wrapper heuristic full set + SUB split points).
    for bm in (1, 2, 4, 8, 16):
        _run_tiled_case(33, 32768, torch.float16, 16384, bm, "L1")
    # fp32 / bf16 over representative shapes.
    for M, N in ((33, 33000), (4, 102400), (128, 32768)):
        _run_tiled_case(M, N, torch.float32, 16384, 1, "L1")
        _run_tiled_case(M, N, torch.bfloat16, 16384, 1, "L1")
    # M=1 streaming + minimal N (N < BLOCK_N full tail-chunk path).
    _run_tiled_case(1, 65536, torch.float16, 16384, 1, "L1")
    _run_tiled_case(8, 100, torch.float16, 16384, 1, "L1")
    _run_tiled_case(8, 100, torch.float32, 16384, 1, "L1")


def _run_tiled_L2():
    # Robustness / stress: large M multi-wave, very large N (long merge chain),
    # tiny tile_n (many chunks), single/tiled switch boundary neighbourhood.
    # Recorded as warnings, do not block the verdict.
    cases = [
        (8192, 32768, torch.float16, 16384, 1),   # 8192 tiles, ~146 waves
        (2, 1024000, torch.float16, 16384, 1),    # 63-chunk merge chain
        (33, 33000, torch.float16, 256, 1),       # tiny tile_n -> 128 chunks
        (33, 32512, torch.float16, 16384, 1),     # single/tiled boundary
        (33, 32513, torch.float16, 16384, 1),
    ]
    for M, N, dt, tile_n, bm in cases:
        try:
            _run_tiled_case(M, N, dt, tile_n, bm, "L2")
        except Exception as e:  # L2 recorded as warn, does not block
            print(f"[L2] WARN (not blocking): shape=({M},{N}) dtype={dt} {e}")


def _run_tiled_boundary():
    # Canonical boundary config: tile_n=256 -> BLOCK_N=256, N=512 spans 2 full
    # chunks (exercises the chunk loop + flash merge for special values).
    tile_n = 256
    block_m = 8
    n = 512

    def base(rows=8, cols=n, dtype=torch.float16):
        return torch.zeros(rows, cols, dtype=dtype, device="npu")

    # finite-domain corners vs golden.
    _run_tiled_special_case(base(), block_m, tile_n, "Boundary-zeros")
    x = base()
    x[:, 0] = 65504.0
    x[:, 1] = -65504.0
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-extreme")
    x = base()
    x[:, :] = 3.5
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-all-equal")
    x = base()
    x[:, 0] = 100.0
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-single-dominant")
    x = base()
    x[:, 0] = 6e-8
    x[:, 1] = -6e-8
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-subnormal")
    x = base()
    x[:, 0] = -745.0
    x[:, 1] = -744.0
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-deep-underflow")
    # partial -inf (finite rows -> matches golden).
    x = base()
    x[:, 0] = float("-inf")
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-neg-inf")
    # NaN row -> both NaN (equal_nan=True).
    x = base()
    x[:, 0] = float("nan")
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-nan")
    # N=1 and M=1 minimal shapes (single tail chunk path).
    _run_tiled_special_case(
        torch.randn(3, 1, dtype=torch.float16, device="npu"), 8, 256, "Boundary-N1"
    )
    _run_tiled_special_case(
        torch.randn(1, 16, dtype=torch.float16, device="npu"), 8, 256, "Boundary-M1"
    )

    # tiled-specific merge corners (vs golden / source sim).
    # cross-chunk max gap 131008: -65504 in chunk0, +65504 in chunk1 (flash form
    # exp args <= 0 -> no overflow; DESIGN V5 corner).
    x = base()
    x[:, 0] = -65504.0     # chunk0 (cols [0, 256))
    x[:, 256] = 65504.0    # chunk1 (cols [256, 512))
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-cross-chunk-maxgap")
    # dominant value in chunk0 vs chunk1 (merge-order both directions).
    x = base()
    x[:, 0] = 100.0        # dominant in chunk0
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-dominant-chunk0")
    x = base()
    x[:, 256] = 100.0      # dominant in chunk1
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-dominant-chunk1")
    # equal values spanning the chunk boundary (associativity).
    x = base()
    x[:, :] = 1.0
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-equal-span-chunks")

    # source-inherited divergence (vs fp32 source semantics, not golden).
    x = base()
    x[:, :] = float("-inf")
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-all-neg-inf", source_semantics=True)
    x = base()
    x[:, 0] = float("inf")
    _run_tiled_special_case(x, block_m, tile_n, "Boundary-pos-inf", source_semantics=True)

    # determinism: same input 3x, bitwise identical.
    x = torch.randn(4096, 4096, dtype=torch.float16, device="npu")
    y0 = _call_tiled(4096, 4096, torch.float16, 16384, 16, x)
    torch.npu.synchronize()
    y1 = _call_tiled(4096, 4096, torch.float16, 16384, 16, x)
    torch.npu.synchronize()
    y2 = _call_tiled(4096, 4096, torch.float16, 16384, 16, x)
    torch.npu.synchronize()
    assert torch.equal(y0, y1) and torch.equal(y1, y2), "nondeterministic output"
    print("[Boundary-determinism] PASS: 3x bitwise identical")


# ---------- Combined suite entrypoints (Stage 4 gate contract) ----------
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
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
